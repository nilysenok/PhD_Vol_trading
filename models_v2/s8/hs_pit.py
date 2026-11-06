"""Ж-212: PIT-примитивы. Строка k (дата d_k) видит только закрытые прогнозы: target_date_j <= d_k, т.е. строки j < m[k].
`extra` — подложенная ошибка («закрыто + extra»): при extra > 0 в расчёт попадают ещё не закрытые строки."""
from __future__ import annotations

import numpy as np


def closed_count(d, td, extra: int = 0) -> np.ndarray:
    """m[k] = число строк с target_date <= d[k] (d, td возрастают), плюс extra, не больше n."""
    d, td = np.asarray(d, "datetime64[ns]"), np.asarray(td, "datetime64[ns]")
    return np.minimum(np.searchsorted(td, d, side="right") + extra, len(d))


def past_mean(x, d, td, window: int, min_obs: int = 5, extra: int = 0) -> np.ndarray:
    """Среднее x по последним `window` закрытым строкам; меньше min_obs закрытых — NaN."""
    x = np.asarray(x, float)
    m = closed_count(d, td, extra)
    fin = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(fin, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(fin)])
    lo = np.maximum(m - window, 0)
    n = cn[m] - cn[lo]
    out = np.full(len(x), np.nan)
    ok = n >= min_obs
    out[ok] = (c[m] - c[lo])[ok] / n[ok]
    return out


def past_last(x, d, td, extra: int = 0) -> np.ndarray:
    """Последнее закрытое значение x (строка m[k]−1); нет закрытых — NaN."""
    x = np.asarray(x, float)
    m = closed_count(d, td, extra)
    out = np.full(len(x), np.nan)
    ok = m >= 1
    out[ok] = x[m[ok] - 1]
    return out


def truncate_future(x, d, td, k: int) -> np.ndarray:
    """Копия x, где все строки, не закрытые на d_k (j >= m[k]), заменены NaN (порча «будущего» для теста обрезки)."""
    x = np.array(x, float, copy=True)
    x[int(closed_count(d, td)[k]):] = np.nan
    return x
