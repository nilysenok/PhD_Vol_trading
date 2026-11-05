"""L4 (Ж-204 B1): прямой QLIKE в лог-параметризации F = ln σ̂² (на шкале y_ratio = RV/S).

Потеря l(y, F) = y·e^(−F) + F; grad = 1 − y·e^(−F); hess = y·e^(−F). Минимум по F при постоянной F — ln mean(y).
QLIKE (models_v2.metrics) = l(y, F) − 1 − ln y: отличается только функцией y (Д7).
"""
from __future__ import annotations

import numpy as np

HESS_MIN = 1e-16


def loss(y, F) -> np.ndarray:
    y, F = np.asarray(y, float), np.asarray(F, float)
    return y * np.exp(-F) + F


def grad_hess(y, F) -> tuple:
    e = np.asarray(y, float) * np.exp(-np.asarray(F, float))
    return 1.0 - e, np.maximum(e, HESS_MIN)


def numeric_grad_hess(y, F, h: float = 1e-5) -> tuple:
    """Конечные разности потери: (g, H) — для теста градиента."""
    g = (loss(y, F + h) - loss(y, F - h)) / (2 * h)
    H = (loss(y, F + h) - 2 * loss(y, F) + loss(y, F - h)) / h ** 2
    return g, H


def qlike_obj(preds, dtrain):
    """Пользовательская цель XGBoost: preds — margin (с base_margin); веса применяются здесь (для custom obj XGBoost их не применяет)."""
    g, h = grad_hess(dtrain.get_label(), preds)
    w = dtrain.get_weight()
    if len(w):
        g, h = g * w, h * w
    return g, h


def qlike_metric(preds, dmat):
    w = dmat.get_weight()
    return "qlike", float(np.average(loss(dmat.get_label(), preds), weights=w if len(w) else None))
