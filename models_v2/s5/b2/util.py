"""B2 (Ж-204): окна внутри сегмента (без склейки через остановку 2022) и сдвиги. Строка i видит только строки <= i."""
from __future__ import annotations

import numpy as np
import pandas as pd


def roll(x, seg, w: int, how: str, mp: int | None = None) -> np.ndarray:
    """Скользящее mean/std(ddof=1)/sum по w строкам с i включительно внутри сегмента; меньше mp конечных (по умолчанию w) — NaN."""
    x, seg = np.asarray(x, float), np.asarray(seg)
    out = np.full(len(x), np.nan)
    for s in np.unique(seg):
        m = np.flatnonzero(seg == s)
        r = pd.Series(x[m]).rolling(w, min_periods=w if mp is None else mp)
        out[m] = getattr(r, how)().to_numpy()
    return out


def roll_corr(x, y, seg, w: int, mp: int) -> np.ndarray:
    x, y, seg = np.asarray(x, float), np.asarray(y, float), np.asarray(seg)
    out = np.full(len(x), np.nan)
    for s in np.unique(seg):
        m = np.flatnonzero(seg == s)
        out[m] = pd.Series(x[m]).rolling(w, min_periods=mp).corr(pd.Series(y[m])).to_numpy()
    return out


def lag(x, seg, k: int = 1) -> np.ndarray:
    """x_{i-k}; NaN, если строка i-k в другом сегменте или её нет."""
    x, seg = np.asarray(x, float), np.asarray(seg)
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = np.where(seg[k:] == seg[:-k], x[:-k], np.nan)
    return out


def ln_floor(x, floor: float = 1e-6) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, float), floor))
