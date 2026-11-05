"""Б-1 (= Г1, Ж-198): остаток к HAR-J(б). e_t = ln RV_{t+1} − lb_t, lb — PIT-подгонка HAR-J по данным <= t."""
from __future__ import annotations

import numpy as np

from models_v2 import har
from models_v2.s2 import features as ft


def harj_ab(p: har.Panel, min_pairs: int = ft.MIN_PAIRS_HAR, full: bool = False) -> tuple:
    """(ln σ̂²_a, ln σ̂²_b) HAR-J на строке t по данным <= t; NaN при < min_pairs пар.
    full=True — одна подгонка на всей выборке (мутация «остаток по HAR на всей выборке»; не используется в расчёте)."""
    n, cols = len(p.rv), list(har.COLS_HARJ)
    la, lb = np.full(n, np.nan), np.full(n, np.nan)
    if full:
        beta, s2, _ = har.ols(*har.pairs(p, n - 1, har.COLS_HARJ))
    for i in range(n):
        if not np.isfinite(p.X[i]).all():
            continue
        if not full:
            Xt, yt = har.pairs(p, i, har.COLS_HARJ)
            if len(yt) < min_pairs:
                continue
            beta, s2, _ = har.ols(Xt, yt)
        la[i] = float(beta[0] + (beta[1:] * p.X[i, cols]).sum())
        lb[i] = la[i] + s2 / 2
    return la, lb


def target_minus(F: ft.Fp, base: np.ndarray) -> np.ndarray:
    """y_t = ln RV_{t+1} − base_t (NaN, где t+1 нет, другой сегмент или base не определён)."""
    y = np.full(len(F.rv), np.nan)
    same = F.seg[:-1] == F.seg[1:]
    y[:-1] = np.where(same, F.lnrv[1:] - base[:-1], np.nan)
    return y


def sigma2_b1(ln_hb, f, c: float, s2_es, bias: bool = True) -> np.ndarray:
    """Б-1: σ̂² = HAR-J(б)·exp(c·f + s²/2); bias=False — без поправки (колонка (а) / чувствительность)."""
    return np.exp(np.asarray(ln_hb) + c * np.asarray(f) + (np.asarray(s2_es) / 2 if bias else 0.0))
