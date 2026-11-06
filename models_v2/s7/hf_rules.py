"""Ж-211: чистые правила (без обучения). R2 clip, R3 shrinkage, выбор (c, λ) по прошлому (HF4/HF5), EMA и ограничение скорости весов (HF7).
Причинность: параметр на строке k — функция первых m[k] строк (прогнозы с target_date <= d_k, `ens.closed`); extra — мутация «закрыто + extra»."""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s5.d import ens

MIN_CLOSED = 252
C_GRID = (0.10, 0.20, 0.35, 0.50)
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def apply(har, g, c=None, lam: float = 1.0) -> np.ndarray:
    """σ̂² = HAR·exp(λ·clip(g, −c, c)); c=None — без клипа (c = ∞), λ = 0 — побитно HAR, λ = 1, c=None — R1."""
    g = np.asarray(g, float)
    gc = g if c is None else np.clip(g, -c, c)
    return np.asarray(har, float) * np.exp(lam * gc)


def cfg_grid(cs=C_GRID, lams=LAM_GRID) -> list:
    """Конфигурации (c, λ) в порядке «меньшая поправка раньше»: первая — (0, 1) = HAR-J (умолчание до 252 закрытых и ничья)."""
    out = [(0.0, 1.0)]
    for c in list(cs) + [None]:
        for lam in lams:
            if lam > 0 and (c, lam) not in out:
                out.append((c, lam))
    return out


def wf_select(rv, har, g, m, cfgs, min_closed: int = MIN_CLOSED) -> tuple:
    """Для строки k: cfg с минимальной суммой QLIKE по первым m[k] строкам; m[k] < min_closed — cfgs[0] (HAR-J). Ничья — раньше в списке.
    -> (σ̂² (n,), индексы выбора (n,))."""
    S = np.column_stack([apply(har, g, c, lam) for c, lam in cfgs])
    C = np.vstack([np.zeros((1, len(cfgs))), np.cumsum(metrics.qlike(np.asarray(rv, float)[:, None], S), axis=0)])
    n = len(rv)
    idx = np.zeros(n, int)
    for k in range(n):
        if m[k] >= min_closed:
            idx[k] = int(np.argmin(C[min(int(m[k]), n)]))
    return S[np.arange(n), idx], idx


def closed_of(rv, d, td, extra: int = 0) -> np.ndarray:
    return ens.closed({"rv": np.asarray(rv), "d": np.asarray(d), "td": np.asarray(td)}, extra)


def ema_weights(W: np.ndarray, half_life: float) -> np.ndarray:
    """w_k = (1−α)·w_{k−1} + α·W_k, α = 1 − 2^(−1/h), w_{−1} = равные веса; W_k — целевой вес, посчитанный по закрытым на d_k."""
    a = 1.0 - 2.0 ** (-1.0 / half_life)
    out, prev = np.empty_like(W), np.full(W.shape[1], 1.0 / W.shape[1])
    for k in range(len(W)):
        prev = (1 - a) * prev + a * W[k]
        out[k] = prev
    return out


def rate_limit_weights(W: np.ndarray, r: float) -> np.ndarray:
    """w_k = w_{k−1} + clip(W_k − w_{k−1}, ±r), нормировка на симплекс; w_{−1} = равные."""
    out, prev = np.empty_like(W), np.full(W.shape[1], 1.0 / W.shape[1])
    for k in range(len(W)):
        prev = prev + np.clip(W[k] - prev, -r, r)
        prev = np.maximum(prev, 0.0)
        prev = prev / prev.sum()
        out[k] = prev
    return out


def mix(w: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.einsum("nk,nk->n", w, S)
