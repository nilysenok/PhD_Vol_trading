"""Л1.5 (Ж-146): семьи S10–S12 с интерфейсом сигнала S1–S6 — `semya(sid, ist)(td, ind, grid) -> SignalSet`.

`ist` — {тикер: 10-минутки (date, slot, OHLCV)}; сигнал дня — `impuls_v7.signal_dnya`. Оси точки:
tau ∈ {1,2,3}, vhod ∈ {d1, T}, T ∈ {17:00, 18:00, 18:40}. vhod=d1: решение в 18:30, вход не раньше d+1 (T не влияет);
vhod=T: решение по барам, закрытым ≤ T, вход в T дня d. Выход — close дня d+tau: exit_l = exit_s = True на баре
d+tau (за концом ленты — нет). strength = |sig|. Движок не трогает; реестр v7 — волна 5. Статус: разработка, до аудита.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import impuls_v7 as I
from fastbt.signals.interface import WARMUP, SignalSet, TickerData

TAU = (1, 2, 3)
VHOD = ("d1", "T")
T_DOP = ("17:00", "18:00", "18:40")
T_D1 = "18:30"


def _tochka(p: dict) -> tuple:
    tau, vhod, T = p.get("tau"), p.get("vhod"), p.get("T")
    if tau not in TAU or vhod not in VHOD or T not in T_DOP:
        raise ValueError(f"точка {p!r}: tau ∈ {TAU}, vhod ∈ {VHOD}, T ∈ {T_DOP}")
    return tau, vhod, T


def semya(sid: str, ist: dict):
    """Фабрика семьи sid ∈ {S10, S11, S12} над лентами `ist` по тикерам."""
    if sid not in ("S10", "S11", "S12"):
        raise ValueError(f"sid {sid!r}: ожидается S10, S11 или S12")

    def fn(td: TickerData, ind: dict, grid: list) -> SignalSet:
        if td.ticker not in ist:
            raise KeyError(f"{td.ticker}: нет 10-минуток в ленте семьи {sid}")
        b10, n = ist[td.ticker], td.n
        dni = pd.DatetimeIndex(td.dates).normalize()
        kesh, sig = {}, np.zeros((len(grid), n), dtype=np.int8)
        ex = np.zeros((len(grid), n), dtype=bool)
        for k, p in enumerate(grid):
            tau, vhod, T = _tochka(p)
            t_sig = T_D1 if vhod == "d1" else T
            if t_sig not in kesh:
                kesh[t_sig] = I.signal_dnya(b10, sid, t_sig).reindex(dni).fillna(0).astype(np.int8).to_numpy()
            sig[k] = kesh[t_sig]
            sig[k, :WARMUP] = 0
            idx = np.flatnonzero(sig[k]) + tau
            ex[k, idx[idx < n]] = True
        return SignalSet(sig=sig, exit_l=ex, exit_s=ex.copy(), strength=np.abs(sig).astype(np.float64),
                         params=list(grid), split=True, extra={})

    return fn
