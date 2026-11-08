"""V4-B: операторы объединения H = K1 (HAR-J) и M = √(K2·K3) (B1–B5) и три эксперта (B6). Только формулы и аналитические производные; ничего не обучается, RV сюда не приходит.
B1 арифм. F = wH + (1−w)M; B2 геом. F = H^w·M^(1−w); B3 гарм. 1/F = w/H + (1−w)/M; B4 степенное (wH^p + (1−w)M^p)^(1/p), p ∈ [−2, 2], p → 0 — предел B2;
B5 = B2 · exp(b), b ∈ [−0,15; 0,15]; B6 F = H^a·K2^b·K3^c, a+b+c = 1. На концах (w = 1 → H, w = 0 → M) значение берётся точно (побитно)."""
from __future__ import annotations

import numpy as np

CFGS = ("B1", "B2", "B3", "B4", "B5", "B6")
EPS_P = 1e-4
P_BOUND, B_BOUND = 2.0, 0.15
W0, B0 = 0.5, 0.0


def mean_m(k2, k3):
    """M = √(K2·K3) — геометрическое среднее двух ML-прогнозов."""
    return np.sqrt(np.asarray(k2, float) * np.asarray(k3, float))


def forward(cfg: str, H, M, w, p=0.0, b=0.0):
    """-> (F, dF/dw, dF/dp, dF/db) по строкам; H, M, w — массивы одной длины (w — вес на H)."""
    H, M, w = (np.asarray(x, float) for x in (H, M, w))
    lH, lM = np.log(H), np.log(M)
    dp = np.zeros_like(H)
    db = np.zeros_like(H)
    if cfg == "B1":
        F, dw = w * H + (1 - w) * M, H - M
    elif cfg in ("B2", "B5"):
        F = np.exp(w * lH + (1 - w) * lM) * (np.exp(b) if cfg == "B5" else 1.0)
        dw = F * (lH - lM)
        db = F if cfg == "B5" else db
    elif cfg == "B3":
        F = 1.0 / (w / H + (1 - w) / M)
        dw = -F * F * (1 / H - 1 / M)
    elif cfg == "B4":
        F, dw, dp = _power(H, M, lH, lM, w, p)
    else:
        raise ValueError("forward: неизвестная конфигурация " + cfg)
    e = np.exp(b) if cfg == "B5" else 1.0           # B5: на концах F = H·e^b / M·e^b; для остальных e = 1.0 (умножение на 1.0 побитно точное)
    F = np.where(w >= 1.0, H * e, np.where(w <= 0.0, M * e, F))
    return F, dw, dp, db


def _power(H, M, lH, lM, w, p):
    if abs(p) < EPS_P:       # p → 0: B2 (p = 0 точно) с поправкой первого порядка ln F ≈ u + p·w(1−w)(lnH−lnM)²/2, ошибка O(p²) ≈ 1e-11; формула степенного среднего при малом p теряет точность
        d = lH - lM
        F = np.exp(w * lH + (1 - w) * lM + p * 0.5 * w * (1 - w) * d * d)
        return F, F * (d + 0.5 * p * (1 - 2 * w) * d * d), 0.5 * w * (1 - w) * d * d * F
    a, c = H ** p, M ** p
    S = w * a + (1 - w) * c
    F = S ** (1.0 / p)
    return F, F / (p * S) * (a - c), F * (-np.log(S) / p ** 2 + (w * a * lH + (1 - w) * c * lM) / (p * S))


def forward_b6(theta, lH, lK2, lK3):
    """theta (n, 3) = (a, b, c) по строкам -> F = exp(a·lnH + b·lnK2 + c·lnK3)."""
    return np.exp(theta[:, 0] * lH + theta[:, 1] * lK2 + theta[:, 2] * lK3)


def qloss(rv, F):
    """QLIKE без константы по RV: rv/F + ln F (сумма совпадает с `models_v2.metrics.qlike` с точностью до слагаемого, не зависящего от F)."""
    return rv / F + np.log(F)
