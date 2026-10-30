"""Л1.4 (Ж-146): семьи S7–S9 (сечение) с интерфейсом сигнала S1–S6: fn(td, ind, grid) -> SignalSet.

Сечение (17 тикеров) в td одного тикера не помещается: панель подаётся заранее (Panel), как внешние ряды v6.
semya(sid, panel)(td, ind, grid) берёт столбец тикера td.ticker из kross_v7.storona на датах td. Решение дня d
использует данные ≤ d (kross_v7); исполнение d+1 — в оценщике. Реестр — локальный REESTR_V7, не interface._SETKI.
Ось точки: rezhim (den|nedelya), porog, bufer, cap. strength = |скор|/σ; exit_l/exit_s — сторона снята.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fastbt import kross_v7 as kv
from fastbt.signals.interface import WARMUP, SignalSet, TickerData

SIDS = ("S7", "S8", "S9")


@dataclass
class Panel:
    """Панель сечения на общих датах: c — закрытия [T, N] (колонки = tikery), imoex — ряд на тех же датах,
    metki — сектор [T, N], sigma — σ̂ [T, N], ok — допуск входа [T, N]."""
    daty: np.ndarray
    tikery: list
    c: pd.DataFrame
    imoex: pd.Series
    metki: np.ndarray
    sigma: np.ndarray
    ok: np.ndarray


def _tochki(td: TickerData, pn: Panel):
    """(столбец тикера, индексы дат td в панели). Нет тикера или даты — KeyError, не нули."""
    if td.ticker not in pn.tikery:
        raise KeyError(f"тикер {td.ticker!r} вне панели")
    daty = np.asarray(pn.daty, "datetime64[ns]")
    pos = np.searchsorted(daty, td.dates)
    bad = (pos >= len(daty)) | (daty[np.minimum(pos, len(daty) - 1)] != td.dates)
    if bad.any():
        raise KeyError(f"{int(bad.sum())} дат {td.ticker!r} вне панели")
    return pn.tikery.index(td.ticker), pos


def semya(sid: str, pn: Panel):
    """Фабрика семьи sid ∈ S7–S9 над панелью pn -> fn(td, ind, grid) -> SignalSet."""
    if sid not in SIDS:
        raise KeyError(f"семья {sid!r}: есть только {SIDS}")
    kesh_rez: dict = {}

    def fn(td: TickerData, ind: dict, grid: list) -> SignalSet:
        j, pos = _tochki(td, pn)
        n, K = td.n, len(grid)
        sig, stg = np.zeros((K, n), np.int8), np.zeros((K, n))
        for k, p in enumerate(grid):
            if p["rezhim"] not in kv.REZHIMY:
                raise ValueError(f"rezhim {p['rezhim']!r} не из {kv.REZHIMY}")
            if p["rezhim"] not in kesh_rez:
                kesh_rez[p["rezhim"]] = kv.rebal(kv.panel(pn.c, pn.imoex, pn.metki, sid), pn.daty, p["rezhim"])
            sk = kesh_rez[p["rezhim"]]
            sg = kv.storona(sk, pn.sigma, pn.metki, pn.ok, p["porog"], p["bufer"], p["cap"])
            sig[k] = sg[pos, j]
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.abs(sk[pos, j]) / pn.sigma[pos, j]
            stg[k] = np.where(np.isfinite(s), s, 0.0)
        sig[:, :WARMUP] = 0
        return SignalSet(sig, sig != 1, sig != -1, stg, list(grid), split=True)

    return fn


REESTR_V7 = {sid: (lambda pn, _s=sid: semya(_s, pn)) for sid in SIDS}
