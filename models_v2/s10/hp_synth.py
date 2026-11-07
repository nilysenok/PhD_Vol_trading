"""Синтетика для тестов P0 (в сеть не ходит, данных проекта не читает): 10-минутные бары основной сессии (10:00…18:40, 53 бара, режим 18:40) со случайным блужданием,
персистентной дневной волатильностью и общим «рыночным» множителем (чтобы корреляция с индексом была, а HAR-J — обучаем). Даты до 2025 — правило nonworking_2025 не срабатывает."""
from __future__ import annotations

import numpy as np
import pandas as pd

START = "2016-01-04"
SLOTS = pd.timedelta_range("10:00:00", "18:40:00", freq="10min")


def weekdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(START, periods=n)


def market(n_all: int, seed: int = 1) -> tuple:
    """Общий ряд ln-волатильности рынка (AR(1)) и доходностей бара индекса: (vol[n_all], r_m[n_all, 53])."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n_all)
    for t in range(1, n_all):
        z[t] = 0.9 * z[t - 1] + 0.3 * rng.normal()
    return z, rng.normal(size=(n_all, len(SLOTS)))


def make_bars(days: pd.DatetimeIndex, mkt: tuple, base: pd.DatetimeIndex, seed: int, beta: float = 0.8, price0: float = 100.0) -> pd.DataFrame:
    """Бары бумаги на днях `days` (подмножество `base` — календаря рынка); доходность = β·рынок + идио, волатильность дня персистентна."""
    z, rm = mkt
    rng = np.random.default_rng(seed)
    own = np.zeros(len(base))
    for t in range(1, len(base)):
        own[t] = 0.85 * own[t - 1] + 0.25 * rng.normal()
    rows, px = [], price0
    pos = {d: k for k, d in enumerate(base)}
    for d in days:
        k = pos[d]
        sig = 0.0006 * np.exp(0.5 * (z[k] + own[k]))
        r = sig * (beta * rm[k] + 0.6 * rng.normal(size=len(SLOTS)))
        c = px * np.exp(np.cumsum(r))
        o = np.r_[px, c[:-1]]
        hi, lo = np.maximum(o, c) * (1 + 0.0002), np.minimum(o, c) * (1 - 0.0002)
        rows.append(pd.DataFrame({"begin": d + SLOTS, "open": o, "high": hi, "low": lo, "close": c, "volume": rng.integers(1000, 5000, len(SLOTS)),
                                  "value": c * 1000.0}))
        px = c[-1]
    return pd.concat(rows, ignore_index=True)


def make_index(base: pd.DatetimeIndex, mkt: tuple, seed: int = 99) -> pd.DataFrame:
    return make_bars(base, mkt, base, seed, beta=1.0)


def world(n_days: int = 900, tickers=("A", "B", "C", "D", "E", "F"), skip: dict | None = None, seed: int = 5) -> tuple:
    """(бары по бумагам, бары индекса, календарь рынка). skip: {бумага: [даты-дыры]} — дни, где у бумаги нет торгов (остальные бумаги торгуют)."""
    base = weekdays(n_days)
    mkt = market(len(base), seed)
    skip = skip or {}
    bars = {}
    for i, t in enumerate(tickers):
        days = base[~base.isin(pd.DatetimeIndex(skip.get(t, [])))]
        bars[t] = make_bars(days, mkt, base, seed * 100 + i, beta=0.6 + 0.1 * i)
    return bars, make_index(base, mkt), base
