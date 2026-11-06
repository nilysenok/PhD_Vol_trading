"""Ж-214: синтетические M10-бары с известными доходностями (для тестов hv2). Не читает данные."""
from __future__ import annotations

import numpy as np
import pandas as pd


def day_bars(day: str, last_min: int = 1120, drop=(), alt_amp: float | None = None, rng=None, amp: float = 0.001,
             p0: float = 100.0, start_min: int = 600, extra_edges: bool = True) -> pd.DataFrame:
    """Бары одного дня: 10:00…last_min шаг 10 мин (минус `drop` — минуты, которых нет). alt_amp — доходности ±alt_amp поочерёдно
    (|r| постоянен: BV > RV); иначе нормальные с sd=amp. extra_edges — бары 09:50 и 18:50 (вне основной сессии, в канон не входят)."""
    rng = rng or np.random.default_rng(0)
    mins = [m for m in range(start_min, last_min + 1, 10) if m not in drop]
    n = len(mins)
    r = alt_amp * np.where(np.arange(n) % 2 == 0, 1.0, -1.0) if alt_amp is not None else rng.normal(0, amp, n)
    c = p0 * np.exp(np.cumsum(r))
    o = np.r_[p0, c[:-1]]
    rows = [(m, o[i], c[i]) for i, m in enumerate(mins)]
    if extra_edges:
        rows = [(590, p0 * 1.002, p0)] + rows + [(1130, c[-1], c[-1] * 1.003)]
    d = pd.Timestamp(day)
    df = pd.DataFrame(rows, columns=["m", "open", "close"])
    df["begin"] = d + pd.to_timedelta(df["m"], unit="m")
    df["high"] = df[["open", "close"]].max(axis=1) * 1.0001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.9999
    df["volume"], df["value"] = 1000, 1e5
    return df[["begin", "open", "high", "low", "close", "volume", "value"]]


def make(specs, seed: int = 0) -> pd.DataFrame:
    """specs: список (дата, kwargs day_bars). Бары подряд, уникальные begin."""
    rng = np.random.default_rng(seed)
    return pd.concat([day_bars(d, rng=rng, **kw) for d, kw in specs], ignore_index=True)


def bdays(start: str, n: int) -> list:
    return [str(x.date()) for x in pd.bdate_range(start, periods=n)]
