"""Ж-212 (HS9, G12): Mixture of Experts. Эксперты HAR, XJ, LF (замороженные σ̂²). Геометрическая смесь σ̂² = HAR^(1−w)·ML^w (F1: ML = XJ, F2: ML = LF), w ∈ {0; 0,1; …; 1} выбирается
причинно по закрытым (>= 252; ничья — ближе к 0,5; до этого w = 0,5). Hedge (F3/F4): w_i ∝ exp(−η·L̄_i/L̄), L̄_i — средний QLIKE последних 252 закрытых, L̄ — среднее по экспертам,
смесь геометрическая, η ∈ {0,5; 2; 8} выбирается причинно (до 252 закрытых — равные веса, η = 2). Арифметическая смесь (Ж-210 A1–A4) — не используется."""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s8 import hs_pit, hs_rules

WS = tuple(sorted(np.round(np.linspace(0, 1, 11), 1), key=lambda w: (abs(w - 0.5), w)))   # порядок предпочтения: ближе к 0,5
ETAS = (2.0, 0.5, 8.0)
WIN = 252


def geo(a, b, w):
    """HAR^(1−w)·ML^w (a, b — σ̂²; w скаляр или массив)."""
    return np.exp((1.0 - np.asarray(w)) * np.log(a) + np.asarray(w) * np.log(b))


def geo_mix(rv, har, ml, d, td, extra: int = 0) -> tuple:
    """-> (σ̂², w по датам) причинной геометрической смеси двух экспертов."""
    S = np.column_stack([geo(har, ml, w) for w in WS])
    i = hs_rules.causal_pick(hs_rules.qlike_matrix(rv, S), d, td, base=0, extra=extra)
    return hs_rules.pick_series(S, i), np.array(WS)[i]


def hedge_weights(Q, d, td, eta: float, extra: int = 0) -> np.ndarray:
    """Q (n×K) — QLIKE экспертов. w_k,i ∝ exp(−η L̄_i/L̄), L̄_i — среднее по последним WIN закрытым; < WIN закрытых — равные веса."""
    n, K = Q.shape
    m = hs_pit.closed_count(d, td, extra)
    cum = np.vstack([np.zeros(K), np.cumsum(Q, axis=0)])
    W = np.full((n, K), 1.0 / K)
    for k in range(n):
        if m[k] >= WIN:
            lb = (cum[m[k]] - cum[m[k] - WIN]) / WIN
            e = np.exp(-eta * lb / lb.mean())
            W[k] = e / e.sum()
    return W


def hedge_mix(rv, S, d, td, extra: int = 0) -> tuple:
    """Hedge экспертов S (n×K), геометрическая смесь, η причинно из ETAS. -> (σ̂², веса (n×K), η по датам)."""
    Q = hs_rules.qlike_matrix(rv, S)
    cand, Ws = [], []
    for eta in ETAS:
        W = hedge_weights(Q, d, td, eta, extra)
        Ws.append(W)
        cand.append(np.exp((W * np.log(S)).sum(axis=1)))
    C = np.column_stack(cand)
    i = hs_rules.causal_pick(hs_rules.qlike_matrix(rv, C), d, td, base=0, extra=extra)
    return hs_rules.pick_series(C, i), np.stack(Ws)[i, np.arange(len(i))], np.array(ETAS)[i]


def third_expert_x(xj, lf) -> np.ndarray:
    return np.log(lf) - np.log(xj)
