"""Синтетические 10-минутные бары для тестов Ж-200 (известный ответ, без сети и без реальных данных)."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_MAIN = 51          # begin 10:00 … 18:40
EVENING = (50, 60)   # begin 18:50, 19:00 (вне основной сессии)


def days(n: int, start: str = "2019-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def from_returns(dates, R: np.ndarray, gaps=None, p0: float = 100.0, evening: np.ndarray | None = None) -> pd.DataFrame:
    """R[i, k] — ln-доходность бара k дня i (open_k = close_{k−1}); gaps[i] — ln(open первого бара / close вчера)."""
    rows, last = [], p0
    for i, d in enumerate(pd.DatetimeIndex(dates)):
        g = 0.0 if gaps is None else float(gaps[i])
        rets = list(R[i])
        mins = [10 * 60 + 10 * k for k in range(N_MAIN)]
        if evening is not None:
            rets += list(evening[i])
            mins += [18 * 60 + m for m in EVENING][:len(evening[i])]
        p = last * np.exp(g)
        for m, r in zip(mins, rets):
            q = p * np.exp(r)
            rows.append((d + pd.Timedelta(minutes=m), p, max(p, q), min(p, q), q))
            p = q
        last = rows[-1][4] if evening is None else _main_close(rows)
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close"])


def _main_close(rows) -> float:
    return [r[4] for r in rows if r[0].hour * 60 + r[0].minute <= 18 * 60 + 40][-1]


def random_returns(n_days: int, seed: int, vol: float = 0.0015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = vol * np.exp(0.4 * rng.standard_normal((n_days, 1)))
    return scale * rng.standard_normal((n_days, N_MAIN))


def random_bars(n_days: int, seed: int, vol: float = 0.0015, with_evening: bool = True) -> pd.DataFrame:
    dts = days(n_days)
    rng = np.random.default_rng(seed + 1)
    ev = 0.001 * rng.standard_normal((n_days, 2)) if with_evening else None
    return from_returns(dts, random_returns(n_days, seed, vol), 0.002 * rng.standard_normal(n_days), evening=ev)


def damage_after(bars: pd.DataFrame, d: pd.Timestamp, seed: int = 7) -> pd.DataFrame:
    """Порча: все бары дней > d и бары дня d с begin >= 18:50 заменены шумом (цены ×случайный множитель)."""
    b = bars.copy()
    t = b["begin"].dt.hour * 60 + b["begin"].dt.minute
    bad = (b["begin"].dt.normalize() > d) | ((b["begin"].dt.normalize() == d) & (t >= 18 * 60 + 50))
    f = np.exp(np.random.default_rng(seed).normal(0, 0.5, int(bad.sum())))
    for c in ("open", "high", "low", "close"):
        b.loc[bad, c] = b.loc[bad, c].to_numpy() * f
    return b
