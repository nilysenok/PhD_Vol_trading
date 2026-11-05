"""Б-2/Б-4 (Ж-198): нормировка масштаба. RV-признаки и цель — отклонения от ln S, уровень возвращается сложением."""
from __future__ import annotations

import numpy as np

from models_v2.s2 import features as ft

WIN, MIN_DAYS = 252, 22
LN_COLS = (0, 1, 2, 4, 5, 6)   # ln RV_d/w/m, ln BV, ln RV первого/последнего часа: минус ln S
SEMI_COLS = (7, 8)              # полудисперсии (уровни) -> ln(max(.,пол)) − ln S


def rv252(rv: np.ndarray, seg: np.ndarray, win: int = WIN, min_days: int = MIN_DAYS) -> np.ndarray:
    """Среднее уровней RV за min(win, доступно в сегменте) строк с i включительно; < min_days — NaN; через разрыв не склеивается."""
    n = len(rv)
    start = np.zeros(n, int)
    for i in range(1, n):
        start[i] = start[i - 1] if seg[i] == seg[i - 1] else i
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(i - win + 1, start[i])
        if i - lo + 1 >= min_days:
            out[i] = rv[lo:i + 1].mean()
    return out


def scale(F: ft.Fp, kind: str) -> np.ndarray:
    """S_d: 'm' — RV_m (среднее за 22 дня, из столбца ln_rv_m), '252' — rv252."""
    return np.exp(F.X[:, 2]) if kind == "m" else rv252(F.rv, F.seg)


def normalize(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Копия X[:, :N_BASE] с RV-признаками в отклонениях от ln S; ln(1+J) -> ln(1+J/S); гэп и dow как есть."""
    Z, lS = X[:, :ft.N_BASE].copy(), np.log(S)
    for c in LN_COLS:
        Z[:, c] = X[:, c] - lS
    for c in SEMI_COLS:
        Z[:, c] = np.log(np.maximum(X[:, c], ft.HOUR_FLOOR)) - lS
    Z[:, 3] = np.log1p(np.expm1(X[:, 3]) / S)
    return Z
