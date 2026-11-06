"""Ж-212: причинные правила выбора по закрытым прогнозам (HS4). Кандидат на дату d выбирается по сумме QLIKE закрытых строк (target_date <= d);
меньше `min_closed` закрытых — базовый кандидат. Ничья (|Δ| <= 1e-12 от минимума) — кандидат раньше в списке (порядок = предпочтение).
`extra` — подложенная ошибка «закрыто + extra» (мутация, должна ловиться)."""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s8 import hs_pit

MIN_CLOSED = 252
TIE = 1e-12


def qlike_matrix(rv, S) -> np.ndarray:
    """Q[:, j] = QLIKE(rv, S[:, j]) по строкам даты d (цель d+1)."""
    S = np.asarray(S, float)
    rv = np.asarray(rv, float)
    return np.column_stack([np.asarray(metrics.qlike(rv, S[:, j]), float) for j in range(S.shape[1])])


def causal_pick(Q, d, td, base: int = 0, min_closed: int = MIN_CLOSED, extra: int = 0) -> np.ndarray:
    """Индекс кандидата на каждую дату: argmin суммы Q по первым m[k] строкам (m = число закрытых на d_k)."""
    Q = np.asarray(Q, float)
    n, K = Q.shape
    m = hs_pit.closed_count(d, td, extra)
    cum = np.vstack([np.zeros(K), np.cumsum(Q, axis=0)])
    out = np.full(n, base, dtype=int)
    for k in range(n):
        if m[k] >= min_closed:
            s = cum[m[k]]
            out[k] = int(np.flatnonzero(s <= s.min() + TIE)[0])
    return out


def pick_series(S, idx) -> np.ndarray:
    S = np.asarray(S, float)
    return S[np.arange(len(S)), idx]
