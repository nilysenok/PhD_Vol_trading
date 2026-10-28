"""Синтетика для проверок FA2b (test_20…25): дневные бары одного тикера без чтения данных."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from fastbt.signals.interface import TickerData


def bars(n=560, seed=3, start="2015-01-05", ticker="ZZZ", sigma=True) -> TickerData:
    """Случайное блуждание с внутридневным размахом и объёмом; σ h1/h5/h22 — свой ряд."""
    rng = np.random.RandomState(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.018, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0.0, 0.006, n))
    h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0.0, 0.01, n)))
    l = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0.0, 0.01, n)))
    v = np.exp(rng.normal(12.0, 0.6, n))
    dates = pd.bdate_range(start, periods=n).values.astype("datetime64[ns]")
    lr = np.zeros(n)
    lr[:-1] = np.log(c[1:] / c[:-1])
    sg = {}
    if sigma:
        for k, hz in enumerate(("h1", "h5", "h22")):
            sg[hz] = 0.015 + 0.005 * np.abs(np.sin(np.arange(n) / (7.0 + 5 * k)))
    return TickerData(ticker, dates, o, h, l, c, v, lr, sg)


def prefix(td: TickerData, t: int) -> TickerData:
    """Ряд, обрезанный по бару t включительно (данные после t не существуют)."""
    cut = lambda a: np.asarray(a)[:t + 1].copy()
    lr = np.zeros(t + 1)
    lr[:-1] = np.log(td.close[1:t + 1] / td.close[:t])
    return dataclasses.replace(td, dates=cut(td.dates), open=cut(td.open), high=cut(td.high),
                               low=cut(td.low), close=cut(td.close), volume=cut(td.volume),
                               log_ret=lr, sigma={k: cut(v) for k, v in td.sigma.items()})


def perturb_after(td: TickerData, t: int, seed=11, scale=0.05) -> TickerData:
    """Цены и объём после бара t заменены шумом; бары ≤ t не тронуты."""
    rng = np.random.RandomState(seed)
    n = td.n
    f = np.ones(n)
    f[t + 1:] = np.exp(np.cumsum(rng.normal(0.0, scale, n - t - 1)))
    o, h, l, c = (np.asarray(a, float) * f for a in (td.open, td.high, td.low, td.close))
    v = np.asarray(td.volume, float).copy()
    v[t + 1:] *= np.exp(rng.normal(0.0, 0.7, n - t - 1))
    lr = np.zeros(n)
    lr[:-1] = np.log(c[1:] / c[:-1])
    return dataclasses.replace(td, open=o, high=h, low=l, close=c, volume=v, log_ret=lr)
