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

from fastbt import exec_e, fast_exec as fx
from fastbt.exec_e_types import Params, Result, Trade
from fastbt.exec_e_v6 import v6_on

_SLOW_RUN = exec_e.run                      # эталон; запасной путь для входов, которые ядро не берёт (NaN в sig)
TAIL = 512                                 # запас баров под последний день вызова (у оценщика — одна свеча close)
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
        good = (np.isfinite(B[:, :n]).all(0) & (B[:, :n] > 0).all(0)).astype(np.int64)
        ok = np.add.reduceat(good, off[~pusto & (nbar > 0)]) == nbar[~pusto & (nbar > 0)]
        bad[np.flatnonzero(~pusto & (nbar > 0))[~ok]] = True
    return B, off, nbar, pusto, bad


def flat(days) -> tuple:
    """Массивы дней: голова (все, кроме последнего) — из кэша, последний день — в хвост буфера."""
    head, last = days[:-1], days[-1]
    key = (id(head[0]) if head else 0, len(head))
    hit = _FLAT.get(key)
    if hit is None or not all(map(operator.is_, hit[0], head)):
        if len(_FLAT) >= 64:                # ≈ 4–5 МБ на тикер train4: не больше ~3 наборов входов в памяти
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


def _slow_only(p: Params, ed) -> bool:
    """Оси прогона Л-2 (Run2: ed, zhizn, gap_propusk, t_vyhod, bar_plus, stop_tik) ядро не повторяет — эталон."""
    f = getattr(exec_e, "_new_axes", None)
    return ed is not None or p.tejk != "net" or p.vhod_tip != "podtverzhdenie" or p.atr_m is not None or p.chast_1r or v6_on(p) or (f is not None and f(p, ed))     # tejk (Ж-084) — только эталон


def run(ticker, days, sig, sigma, p: Params, rf_cal=None, close=None, div=None, nalog=None, ed=None, tejk_M=None, urovni=None, rang=None) -> Result:
    """Drop-in `exec_e.run`: те же проверки и тот же Result (сделки — объекты Trade); оси Run2, tejk, stop_zayavka — медленным путём."""
    from fastbt import dividendy
    sig, sigma = np.asarray(sig), np.asarray(sigma, float)
    if _slow_only(p, ed) or not len(days) or not np.isfinite(sig.astype(float)).all():
        kw = {k: v for k, v in (("ed", ed), ("tejk_M", tejk_M), ("urovni", urovni), ("rang", rang)) if v is not None}
        return _SLOW_RUN(ticker, days, sig, sigma, p, rf_cal=rf_cal, close=close, div=div, nalog=nalog, **kw)
    if not (len(days) == len(sig) == len(sigma)) or (close is not None and len(close) != len(days)):
        raise ValueError("days, sig, sigma, close разной длины")
    exec_e._check_params(p)
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


# --- хэш входов один раз на набор (Д5: подмена любого поля или листа — новый id — пересчёт) ---
def _leaves(x, out: list) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            out.append(k)
            _leaves(v, out)
    elif isinstance(x, list):
        out.append(x)
        if x and not isinstance(x[0], (dict, list)):
            out.extend(x)
        else:
            for v in x:
                _leaves(v, out)
    else:
        out.append(x)


def _digest_fast(inp) -> str:
    """Тот же `E1Inputs.digest`; повтор — если все листы входов те же объекты (ссылки держатся — id не переиспользуется).
    Замена массива внутри листа на месте (без нового объекта) не ловится — входы оценщика так не меняются."""
    from fastbt import dividendy
    leaves: list = []
    for x in (inp.sk, inp.sigma, inp.bars, inp.close_n6, inp.rezhim, inp.sobytiya, inp.tick, inp.rate, inp.meta,
              inp.div):
        _leaves(x, leaves)
    tag = (inp.commission, float(dividendy.DIV_NALOG), len(leaves))
    hit = _MEM.get(("digest", id(inp)))
    if hit is not None and hit[1] == tag and all(map(operator.is_, hit[0], leaves)):
        return hit[2]
    d = _ORIG["digest"](inp)
    _MEM[("digest", id(inp))] = (leaves, tag, d, inp)
    return d


def _align_b_fast(series, dates):
    """`etap1_data.align_b` один раз на (ряд, даты тикера): ключ — объект ряда и адрес/длина дат (держатся в кэше)."""
    d = np.asarray(dates)
    k = (id(series), d.__array_interface__["data"][0], len(d), d.dtype.str)
    hit = _ALIGN.get(k)
    if hit is None or hit[0] is not series:
        hit = _ALIGN[k] = (series, d, _ORIG["align_b"](series, dates))
    return hit[2].copy()


# --- кэш позиций Sk по Д5 (память + диск, по образцу fs_cache) ---
def _cache_dir() -> Path:
    from fastbt.cache import MAIN
    return Path(os.environ.get("FASTBT_E1_CACHE", MAIN / "data" / "fastbt" / "etap1_cache"))


def _group_key(evr, sp: dict, q: float, anchor: bool) -> str:
    """sha1: код fastbt, ряды Sk (series_digest), sid, база точки, сигнальные параметры, q, якорь, RM_OFF."""
    from fastbt import cache
    from fastbt.etap1_eval import RM_OFF
    sk = evr.sk
    sd = _MEM.get(("sk", id(sk)))
    if sd is None or sd[0] is not sk:
        sd = _MEM[("sk", id(sk))] = (sk, sk.series_digest())
    p = {**evr.base, **sp, "q_sily": q, **({} if anchor else RM_OFF[evr.sid])}
    parts = [cache.code_hash(), sd[1], evr.sid, sorted(sk.sig_names), list(sk.tickers), bool(anchor)]
    s = json.dumps([parts, sorted((str(k), repr(v)) for k, v in p.items())], default=str)
    return "e1pos_" + hashlib.sha1(s.encode()).hexdigest()[:24]


def _positions_fast(self, t: str, sp: dict, q: float, anchor: bool, gi1: int) -> np.ndarray:
    k = (t, tuple(sorted(sp.items())), q, anchor, gi1)
    if k in self._pos:
        return self._pos[k]
    gk = _group_key(self, sp, q, anchor)
    grp = _MEM.get(gk)
    if grp is None:
        f = _cache_dir() / f"{gk}.pkl"
        use = os.environ.get("FASTBT_NO_CACHE", "") != "1"
        grp = _MEM[gk] = pickle.loads(f.read_bytes()) if use and f.exists() else {}
    if (t, gi1) not in grp:
        grp[(t, gi1)] = _ORIG["positions"](self, t, sp, q, anchor, gi1)
        if len(grp) % len(self.sk.tickers) == 0:
            _save(gk, grp)
    self._pos[k] = grp[(t, gi1)]
    return self._pos[k]


def _save(gk: str, grp: dict) -> None:
    if os.environ.get("FASTBT_NO_CACHE", "") == "1":
        return
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{gk}.tmp{os.getpid()}"
    tmp.write_bytes(pickle.dumps(grp, protocol=4))
    os.replace(tmp, d / f"{gk}.pkl")


_ORIG: dict = {}


@contextlib.contextmanager
def patched():
    """Подменяет горячие функции оценщика этапа 1 быстрыми (результат тот же) и возвращает всё назад."""
    from fastbt import etap1, etap1_eval
    saved = [(exec_e, "run", exec_e.run), (etap1.E1Inputs, "digest", etap1.E1Inputs.digest),
             (etap1_eval, "align_b", etap1_eval.align_b), (etap1_eval.Evaluator, "_positions",
                                                            etap1_eval.Evaluator._positions)]
    _ORIG.update(run=_SLOW_RUN, digest=saved[1][2], align_b=saved[2][2], positions=saved[3][2])
    try:
        exec_e.run = run
        etap1.E1Inputs.digest = _digest_fast
        etap1_eval.align_b = _align_b_fast
        etap1_eval.Evaluator._positions = _positions_fast
        yield
    finally:
        for mod, name, v in saved:
            setattr(mod, name, v)


def pokryto(pt: dict) -> bool:
    """§1.8 контракта ledger v2.0: быстрый слой покрывает точки осей zh047 (v1.0); новые оси §1.1 — только в умолчании,
    `filtr` — только net | cbr (прежние `sobytiya`), заём — только 0. Остальное драйвер считает медленным путём."""
    from fastbt.etap1_eval import NET
    umolch = dict(ed_lim="sigma", zhizn="den", lestnica=NET, gap="vhod", t_vyhod="close", k_x=NET, storona="obe",
                  tejk_signal=NET)                  # Ж-084: tejk_signal ≠ net — медленный путь (эталон)
    return (all(str(pt.get(k, v)) == str(v) for k, v in umolch.items()) and pt.get("filtr", NET) in (NET, "cbr")
            and pt.get("sobytiya", NET) in (NET, "cbr") and pt.get("vyhod", NET) in (NET, "lim")
            and pt.get("zaem", 0) in (0, "0", "0.0", NET, None))


def ryady(evr, pt: dict, period: str) -> tuple:
    """Ряды ledger v2.0 §1.4 (docs/otchety/2026-10-28_progon_etap1.md): (port float64 [T], tik float32 [17, T], daty).
    port — df["excess"] годов периода (как v1.0 excess_d подряд); tik — ex_t = g − sides·comm − u·rf·cal, u = expo
    (fdr = None) или noch; nan — у тикера нет строки на дату. Писатель — ledger_v2.Pisatel (Л-1), когда появится."""
    from fastbt.etap1_eval import years_of
    with patched():
        df, inv, eng, _, rs = evr.excess(pt, period)
    keep = np.isin(eng.dates.year, years_of(period))
    tik = np.full((len(rs), len(eng.dates)), np.nan)
    a = 0
    for ti, r in enumerate(rs):
        d = inv[a:a + len(r.g)]
        u = r.expo if r.noch is None else r.noch
        tik[ti, d] = r.g - r.sides * evr.comm - u * eng.rf[d] * eng.cal[d]
        a += len(r.g)
    return df["excess"].to_numpy()[keep], tik[:, keep].astype(np.float32), eng.dates[keep].to_numpy()


# --- параллельно: fork с общими массивами баров ---
_W: dict = {}


def _one(args):
    pt, period = args
    with patched():
        return _W["evr"].evaluate(pt, period)


def map_points(evr, points: list, period: str, jobs: int = 10) -> list:
    """evaluate по точкам в `jobs` процессах fork: входы (бары, σ) построены до fork и общие (копия при записи)."""
    import multiprocessing as mp
    _W["evr"] = evr
    if jobs <= 1:
        return [_one((p, period)) for p in points]
    with mp.get_context("fork").Pool(jobs) as pool:
        return pool.map(_one, [(p, period) for p in points], chunksize=max(1, len(points) // (jobs * 4)))
