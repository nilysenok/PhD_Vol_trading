"""Тест Диболда–Мариано по разности QLIKE: Ньюи–Уэст (Бартлетт), лаг 5, N(0;1), двусторонний p."""
from __future__ import annotations

import math

import numpy as np

from models_v2 import metrics

LAG = 5


def nw_variance(x: np.ndarray, lag: int = LAG) -> float:
    """Долгосрочная дисперсия среднего: (γ0 + 2 Σ_{l=1..lag} (1 − l/(lag+1)) γ_l) / n."""
    n = len(x)
    u = x - x.mean()
    lrv = float((u * u).sum()) / n
    for l in range(1, lag + 1):
        lrv += 2 * (1 - l / (lag + 1)) * float((u[l:] * u[:-l]).sum()) / n
    return lrv / n


def dm(rv, s2_base, s2_model, lag: int = LAG) -> dict:
    """d_t = QLIKE(base) − QLIKE(model); плюс = модель лучше. -> mean_d, stat, p, sign, n."""
    d = metrics.qlike(rv, s2_base) - metrics.qlike(rv, s2_model)
    if len(d) < 2:  # пустая или одноточечная подвыборка: статистики нет
        return {"n": len(d), "mean_d": float("nan"), "stat": float("nan"), "p": float("nan"), "sign": 0}
    v = nw_variance(d, lag)
    m = float(d.mean())
    if v <= 0:
        return {"n": len(d), "mean_d": m, "stat": float("nan"), "p": float("nan"), "sign": int(np.sign(m))}
    z = m / math.sqrt(v)
    return {"n": len(d), "mean_d": m, "stat": z, "p": math.erfc(abs(z) / math.sqrt(2)), "sign": int(np.sign(m))}


def holm(p: np.ndarray) -> np.ndarray:
    """Поправка Холма (step-down), монотонная, не меньше исходного p."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj
