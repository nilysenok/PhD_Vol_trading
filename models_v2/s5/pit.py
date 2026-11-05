"""PIT-примитивы волны A: всё на дате d строится только по данным <= d (пороги — по значениям строго до d)."""
from __future__ import annotations

import numpy as np

from models_v2.s3 import post

WIN = 252
EPS_K = 0.1
MIN_CLOSED = 252
SHIFT_DAYS = 5   # деклъстеризация событий


def q_prior_nan(x, q: float, win: int = WIN, min_finite: int = WIN - 10) -> np.ndarray:
    """Квантиль q по win предыдущим значениям (без текущего), NaN пропускаются (после остановки 2022 в окне 4 NaN); меньше min_finite — NaN."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for k in range(win, len(x)):
        w = x[k - win:k]
        if np.isfinite(w).sum() >= min_finite:
            out[k] = np.nanquantile(w, q)
    return out


def past_bucket(x, inner, win: int = WIN, include_current: bool = False) -> np.ndarray:
    """Номер группы 0..len(inner) значения x[k] относительно квантилей inner по win предыдущим значениям.
    include_current=True — мутация (квантиль с текущим днём); в рабочем коде всегда False."""
    x = np.asarray(x, float)
    thr = []
    for q in inner:
        if include_current:
            t = np.full(len(x), np.nan)
            for k in range(win, len(x)):
                t[k] = np.nanquantile(x[k - win + 1:k + 1], q)
            thr.append(t)
        else:
            thr.append(q_prior_nan(x, q, win))
    T = np.vstack(thr)
    with np.errstate(invalid="ignore"):
        b = (x[None, :] > T).sum(axis=0).astype(float)
    return np.where(np.isnan(T).any(axis=0) | np.isnan(x), np.nan, b)


def eps_floor(rv_daily) -> np.ndarray:
    """ε_i = 0,1 · min(RV по строкам <= i): граница только из известного на i."""
    return EPS_K * np.minimum.accumulate(np.asarray(rv_daily, float))


def scale_shift(rv, s2, d, td, min_closed: int = MIN_CLOSED, extra_closed: int = 0) -> np.ndarray:
    """c_k = ln mean(RV/σ̂²) по закрытым прогнозам (target_date <= d_k); меньше min_closed — 0.
    QLIKE(y, f e^c) минимален при e^c = mean(y/f). extra_closed=1 — мутация (пара d в окне)."""
    n = len(rv)
    cc = np.minimum(post.closed_count(np.asarray(d), np.asarray(td)) + extra_closed, n)
    cs = np.cumsum(np.asarray(rv, float) / np.asarray(s2, float))
    out = np.zeros(n)
    ok = cc >= min_closed
    out[ok] = np.log(cs[cc[ok] - 1] / cc[ok])
    return out


def decluster(flag, gap: int = SHIFT_DAYS) -> np.ndarray:
    """Индексы событий; следующее событие принимается не раньше чем через gap строк после принятого."""
    out, last = [], -10 ** 9
    for i in np.flatnonzero(np.asarray(flag, bool)):
        if i - last >= gap:
            out.append(int(i))
            last = int(i)
    return np.array(out, int)
