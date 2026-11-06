"""Ж-212 (HS8, G11): Conditional Correction. σ̂² = HAR·exp(a_t·g), a_t = logistic(θ0 + Σ θ_i z_i,t) ∈ [0, 1]. θ — QLIKE-подгонка по закрытым строкам (target_date <= d), расширяющееся окно,
ridge на наклоны, ≥ 252 закрытых (до этого a = 0 → HAR), переподгонка каждые 5 строк; z стандартизован по закрытым строкам момента подгонки. Показатели Z1–Z4 известны к 18:50 d."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models_v2.s8 import hs_pit

STEP, MIN_CLOSED = 5, 252


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def loss_grad(theta, Zc, rv, har, g, ridge: float) -> tuple:
    """Средний QLIKE + ridge·|θ[1:]|²/2 и аналитический градиент. Zc — столбцы [1, z…]."""
    a = logistic(np.einsum("ij,j->i", Zc, theta))
    m = np.log(har) + a * g
    r = rv * np.exp(-m)
    L = float(np.mean(m + r)) + 0.5 * ridge * float(np.sum(theta[1:] ** 2))
    dm = (1.0 - r) * g * a * (1.0 - a)
    grad = np.einsum("ij,i->j", Zc, dm) / len(rv)
    grad[1:] += ridge * theta[1:]
    return L, grad


def fit_theta(Zc, rv, har, g, ridge: float, theta0=None) -> np.ndarray:
    k = Zc.shape[1]
    t0 = np.zeros(k) if theta0 is None else theta0
    r = minimize(loss_grad, t0, args=(Zc, rv, har, g, ridge), jac=True, method="L-BFGS-B")
    return r.x


def activity(Z, rv, har, g, d, td, ridge: float = 0.01, step: int = STEP, min_closed: int = MIN_CLOSED, extra: int = 0) -> np.ndarray:
    """a_t на каждую дату. Z (n×p) — показатели на дату d_t; строки j < m[k0] (закрытые на момент подгонки) — единственное, что видит подгонка."""
    Z = np.asarray(Z, float).reshape(len(rv), -1)
    n, p = Z.shape
    m = hs_pit.closed_count(d, td, extra)
    a, theta, mu, sd = np.zeros(n), None, np.zeros(p), np.ones(p)
    for k0 in range(0, n, step):
        if m[k0] >= min_closed:
            mm = int(m[k0])
            ok = np.isfinite(Z[:mm]).all(axis=1)
            mu, sd = Z[:mm][ok].mean(axis=0), Z[:mm][ok].std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            Zc = np.column_stack([np.ones(ok.sum()), (Z[:mm][ok] - mu) / sd])
            theta = fit_theta(Zc, rv[:mm][ok], har[:mm][ok], g[:mm][ok], ridge, theta)
        if theta is not None:
            zs = np.nan_to_num((Z[k0:k0 + step] - mu) / sd, nan=0.0)
            a[k0:k0 + step] = logistic(theta[0] + np.einsum("ij,j->i", zs, theta[1:]))
    return a


def expanding_dev(x) -> np.ndarray:
    """ln-показатель минус расширяющаяся медиана (по строкам <= d)."""
    s = pd.Series(np.asarray(x, float))
    return (s - s.expanding(min_periods=20).median()).to_numpy()
