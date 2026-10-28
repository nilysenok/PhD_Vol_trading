"""Быстрый слой этапа 1 (Ж-083): `patched()` подменяет горячие места оценщика, медленный путь — эталон.

Профиль «до» (docs/otchety/2026-10-28_skorost_etap1.md): время уходило на (1) `E1Inputs.digest` — хэш всех баров
на каждую точку, (2) проверку баров в `exec_e.run` (np.stack на каждый день каждого вызова), (3) автомат
`exec_e_run.Run` в Python, (4) сигналы Sk и позиции на каждую новую сигнальную точку.
Подмены (результат тот же, побайтно):
 - `exec_e.run` → `run`: тот же автомат одним numba-проходом (`fast_exec.kernel`); бары тикера раскладываются
   в массивы один раз (кэш по тождеству объектов Day, ссылки держатся — id не переиспользуется);
 - `E1Inputs.digest` → тот же хэш, посчитанный один раз на набор входов (ключ — тождество всех полей и комиссия);
 - `etap1_eval.align_b` → тот же флаг, один раз на (ряд, даты тикера);
 - `Evaluator._positions` → тот же расчёт, кэш позиций по Д5 (хэш рядов Sk, кода, sid, тикера, параметров) в памяти
   и на диске (`<основная копия>/data/fastbt/etap1_cache/`, FASTBT_E1_CACHE — другая папка, FASTBT_NO_CACHE=1 — без диска).
Параллельно — `map_points`: fork с общими массивами баров (копия при записи), по умолчанию 10 процессов (Ж-082).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import operator
import os
import pickle
from pathlib import Path

import numpy as np

from fastbt import fast_exec as fx
from fastbt.exec_e_types import Params, Result, Trade

TAIL = 512                                  # запас баров под последний день вызова (у оценщика — одна свеча close)
_FLAT: dict = {}
_ALIGN: dict = {}
_MEM: dict = {}


def _stack(days) -> tuple:
    """(B[4, бары + TAIL], off, nbar, pusto, плохой день) по списку Day."""
    nbar = np.fromiter((len(d.c) for d in days), np.int64, len(days))
    pusto = np.fromiter((d.pusto for d in days), np.bool_, len(days))
    full = [d for d in days if not d.pusto and len(d.c)]
    B = np.empty((4, int(nbar[~pusto].sum()) + TAIL))
    n = B.shape[1] - TAIL
    if full:
        B[:, :n] = np.concatenate([np.stack([d.o, d.h, d.l, d.c]) for d in full], axis=1)
    off = np.zeros(len(days), np.int64)
    off[~pusto] = np.r_[0, np.cumsum(nbar[~pusto])[:-1]] if (~pusto).any() else 0
    bad = ~pusto & (nbar == 0)
    if n:
        good = np.isfinite(B[:, :n]).all(0) & (B[:, :n] > 0).all(0)
        ok = np.add.reduceat(good, off[~pusto & (nbar > 0)]) == nbar[~pusto & (nbar > 0)]
        bad[np.flatnonzero(~pusto & (nbar > 0))[~ok]] = True
    return B, off, nbar, pusto, bad


def flat(days) -> tuple:
    """Массивы дней: голова (все, кроме последнего) — из кэша, последний день — в хвост буфера."""
    head, last = days[:-1], days[-1]
    key = (id(head[0]) if head else 0, len(head))
    hit = _FLAT.get(key)
    if hit is None or not all(map(operator.is_, hit[0], head)):
        if len(_FLAT) > 256:
            _FLAT.clear()
        hit = _FLAT[key] = (list(head), _stack(head))
    B, off, nbar, pusto, bad = hit[1]
    lb, lo, ln, lp, lbad = _stack([last])
    if ln[0] > TAIL:
        return _stack(days)
    n0 = B.shape[1] - TAIL
    B[:, n0:n0 + ln[0]] = lb[:, :ln[0]]
    return B, np.r_[off, n0], np.r_[nbar, ln], np.r_[pusto, lp], np.r_[bad, lbad]


def _params(p: Params, nal: float) -> tuple:
    f = lambda v: 0.0 if v is None else float(v)                                    # noqa: E731
    pf = np.array([f(p.k), f(p.a), f(p.b), p.comm, f(p.tick), f(p.R), f(p.h), f(p.k_tr), nal])
    pi = np.array([p.k is not None, fx.F_CODES.get(p.f, 3), p.a is not None, p.b is not None, p.tau is not None,
                   p.tau or 0, p.s == "ezhedn", p.exit_signal, p.side, p.tick is not None, p.stop_touch,
                   p.R is not None, p.h is not None, p.k_tr is not None, p.be, p.check == "zakrytie", p.reentry,
                   p.vyhod_lim], np.int64)
    return pf, pi


def _trades(ticker, ti, tf, comm) -> list:
    out = []
    for r, (epx, xpx) in zip(ti.tolist(), tf.tolist()):
        done = r[fx.XD] >= 0
        out.append(Trade(ticker, r[fx.DS], r[fx.ED], r[fx.EB], epx, r[fx.XD] if done else None,
                         r[fx.XB] if done else None, xpx if done else None, fx.REASONS[r[fx.RS]] if done else None,
                         r[fx.SDC], comm, comm if done else 0.0, r[fx.R0]))
    return out


def run(ticker, days, sig, sigma, p: Params, rf_cal=None, close=None, div=None, nalog=None) -> Result:
    """Drop-in `exec_e.run`: те же проверки и тот же Result (сделки — объекты Trade)."""
    from fastbt import dividendy, exec_e
    sig, sigma = np.asarray(sig), np.asarray(sigma, float)
    if not (len(days) == len(sig) == len(sigma)) or (close is not None and len(close) != len(days)):
        raise ValueError("days, sig, sigma, close разной длины")
    exec_e._check_params(p)
    if not len(days) or not np.isfinite(sig.astype(float)).all():
        return _ORIG["run"](ticker, days, sig, sigma, p, rf_cal=rf_cal, close=close, div=div, nalog=nalog)
    B, off, nbar, pusto, bad = flat(days)
    cl = B[3, off + np.maximum(nbar - 1, 0)] if close is None else np.asarray(close, float)
    if close is not None:
        bad_c = ~pusto & ~(np.isfinite(cl) & (cl > 0))
        if (bad | bad_c).any():
            i = int(np.flatnonzero(bad | bad_c)[0])
            raise ValueError(f"день {i}: " + ("пустой или нечисловой набор баров (нужен pusto=True)" if bad[i]
                                                 else "close Н6 нечисловой"))
    elif bad.any():
        raise ValueError(f"день {int(np.flatnonzero(bad)[0])}: пустой или нечисловой набор баров (нужен pusto=True)")
    if div is not None and len(div) != len(days):
        raise ValueError("days, div разной длины")
    ds, ix = np.zeros(len(days)), np.flatnonzero(~pusto)
    if div is not None:
        for i in np.flatnonzero(np.asarray(div, float)):
            k = np.searchsorted(ix, i)
            if k < len(ix):
                ds[ix[k]] += float(div[i])
    dirs = (p.side * np.sign(sig.astype(float))).astype(np.int64)
    pf, pi = _params(p, float(dividendy.DIV_NALOG if nalog is None else nalog))
    gross, cost, pos, free, ti, tf, st = fx.kernel(B, off, nbar, pusto, cl, dirs, sigma, ds, pf, pi)
    rf = np.zeros(len(days)) if rf_cal is None else np.asarray(rf_cal, float)
    return Result(_trades(ticker, ti, tf, p.comm), gross, cost, pos, free, free * rf, int(st[fx.NO]), int(st[fx.NF]),
                  int(st[fx.XN]), int(st[fx.XF]))
