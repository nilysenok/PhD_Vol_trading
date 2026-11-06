"""Обучение волны D: окно (последние N пар) и/или экспоненциальные веса поверх схемы И1 (`fitx.fit_i1x`, seed 42). Без окна и весов побитно равно C (`c_fit.fit_c`, l2)."""
from __future__ import annotations

import numpy as np

from models_v2.s4 import fit
from models_v2.s5.b2 import loyo
from models_v2.s5.c import fitx


def fit_d(X, y, idx, hp, window: int, hl: int, T: int, seed: int = fit.SEED) -> dict:
    """Блок прогноза: пары idx (t+1 <= T) → последние `window`; веса 0,5^((T−t)/hl) по возрасту относительно конца окна T."""
    idx = idx[-window:] if window else idx
    w = fit.weights(np.arange(len(y)), T, hl) if hl else None
    return fitx.fit_i1x(X, y, idx, hp, w, None, seed)


def fold_train(tr: np.ndarray, val: np.ndarray, window: int, hl: int, n_rows: int) -> tuple:
    """Фолд LOYO: окно/веса — по расстоянию до проверяемого года (как B2, `loyo.nearest`). -> (пары, веса по строкам | None)."""
    if not (window or hl):
        return tr, None
    tr, dist = loyo.nearest(tr, val, window, hl)
    if not hl:
        return tr, None
    w = np.zeros(n_rows)
    w[tr] = 0.5 ** (dist / hl)
    return tr, w
