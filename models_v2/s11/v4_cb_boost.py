"""V4-C, лист C-b, C7: малый QLIKE-бустинг гейта — собственная реализация (без внешних библиотек, детерминированная, seed не нужен). Логит веса η (w = σ(η), F = H·exp(w·ℓ)) — сумма 200 деревьев
глубины 2, шаг 0,05, лист ≥ 500 пар; старт η = 0 (w = 0,5). Потеря на паре: e + ln F, e = RV/F. Градиент по η: g = (1 − e)·u, u = ℓ·w(1 − w). Гессиан: гауссово-ньютоновская часть h = e·u² (полный
гессиан e·u² + (1 − e)·u' может быть отрицательным, лист был бы неустойчив); значение листа −ΣG/(ΣH + L), L = 250·s²/16 (250 пар-эквивалентов с кривизной одной пары по логиту s²/16 при w = 0,5, как в
гребне C-a). Пороги — по 32 квантильным корзинам признака. Ранней остановки нет. Гиперпараметры фиксированы объявлением."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from models_v2.s11 import v4_b_fit as FT

N_TREES, DEPTH, SHRINK, MIN_LEAF, NBINS = 200, 2, 0.05, 500, 32


def grad_hess(rv, lH, ell, eta) -> tuple:
    w = expit(eta)
    u = ell * w * (1.0 - w)
    e = rv * np.exp(-(lH + w * ell))
    return (1.0 - e) * u, e * u * u


def _edges(X: np.ndarray) -> list:
    return [np.unique(np.quantile(X[:, j], np.linspace(0, 1, NBINS + 1)[1:-1])) for j in range(X.shape[1])]


def _bin(X: np.ndarray, edges: list) -> np.ndarray:
    return np.stack([np.searchsorted(edges[j], X[:, j], side="left") for j in range(X.shape[1])], 1)


def _best_split(Xb, g, h, idx, L, min_leaf):
    """-> (gain, признак, корзина: влево — корзины ≤ b) или None, если ни одно разбиение не оставляет ≥ min_leaf пар по обе стороны."""
    G, H, n = g[idx].sum(), h[idx].sum(), len(idx)
    best = None
    for j in range(Xb.shape[1]):
        b = Xb[idx, j]
        nb = int(b.max()) + 1
        cl = np.cumsum(np.bincount(b, minlength=nb))[:-1]
        gl, hl = np.cumsum(np.bincount(b, g[idx], minlength=nb))[:-1], np.cumsum(np.bincount(b, h[idx], minlength=nb))[:-1]
        ok = (cl >= min_leaf) & (n - cl >= min_leaf)
        if not ok.any():
            continue
        gain = np.where(ok, gl ** 2 / (hl + L) + (G - gl) ** 2 / (H - hl + L) - G ** 2 / (H + L), -np.inf)
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[0]:
            best = (float(gain[i]), j, i)
    return best if best is not None and best[0] > 0 else None


def _grow(Xb, g, h, idx, depth, L, min_leaf):
    s = _best_split(Xb, g, h, idx, L, min_leaf) if depth > 0 else None
    if s is None:
        return ("leaf", float(-g[idx].sum() / (h[idx].sum() + L)))
    _, j, b = s
    m = Xb[idx, j] <= b
    return ("split", j, b, _grow(Xb, g, h, idx[m], depth - 1, L, min_leaf), _grow(Xb, g, h, idx[~m], depth - 1, L, min_leaf))


def _apply(node, Xb: np.ndarray) -> np.ndarray:
    if node[0] == "leaf":
        return np.full(len(Xb), node[1])
    m = Xb[:, node[1]] <= node[2]
    out = np.empty(len(Xb))
    out[m], out[~m] = _apply(node[3], Xb[m]), _apply(node[4], Xb[~m])
    return out


def fit_boost(X, rv, lH, ell, n_trees: int | None = None, min_leaf: int | None = None) -> dict:
    """X (N, k) — строки обучения с конечными признаками. -> {edges, med, trees, n_leaves}."""
    n_trees, min_leaf = N_TREES if n_trees is None else n_trees, MIN_LEAF if min_leaf is None else min_leaf
    L = FT.LAM * float(np.mean(ell ** 2)) / 16.0
    edges = _edges(X)
    Xb = _bin(X, edges)
    eta, trees, idx = np.zeros(len(X)), [], np.arange(len(X))
    for _ in range(n_trees):
        g, h = grad_hess(rv, lH, ell, eta)
        t = _grow(Xb, g, h, idx, DEPTH, L, min_leaf)
        eta = eta + SHRINK * _apply(t, Xb)
        trees.append(t)
    return {"edges": edges, "med": np.median(X, 0), "trees": trees, "eta_train": eta}


def predict_eta(model: dict, X: np.ndarray) -> np.ndarray:
    """Нефинитный признак заменяется медианой окна обучения (в обучение такие строки не входят)."""
    Xf = np.where(np.isfinite(X), X, model["med"])
    Xb = _bin(Xf, model["edges"])
    return SHRINK * sum(_apply(t, Xb) for t in model["trees"])


def n_splits(model: dict) -> int:
    def cnt(t):
        return 0 if t[0] == "leaf" else 1 + cnt(t[3]) + cnt(t[4])
    return sum(cnt(t) for t in model["trees"])
