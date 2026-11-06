"""Ж-212: QLIKE-согласованная подгонка. f = base·exp(X·γ) > 0 всегда; L = ln f + RV/f (с точностью до константы = QLIKE metrics.qlike).
По g (при f = HAR·exp(g)): градиент 1 − RV/f, гессиан RV/f (> 0). По γ: ∇ = Xᵀ(1 − RV/f), H = Xᵀ diag(RV/f) X.
Произведения — через einsum/сумму (matmul Accelerate на macOS даёт ложные RuntimeWarning, как в models_v2/har.py)."""
from __future__ import annotations

import numpy as np


def lin(X, gam) -> np.ndarray:
    return (np.asarray(X, float) * np.asarray(gam, float)).sum(axis=1)


def obj(rv, base, g):
    """Поэлементно по g: (L, grad, hess) для f = base·exp(g). Положительность f и hess обеспечена exp."""
    rv, base, g = (np.asarray(a, float) for a in (rv, base, g))
    f = base * np.exp(g)
    r = rv / f
    return np.log(f) + r, 1.0 - r, r


def mean_loss(gam, X, rv, off) -> float:
    f_ln = off + lin(X, gam)
    return float(np.mean(f_ln + rv * np.exp(-f_ln)))


def fit(X, rv, off, ridge: float = 0.0, gam0=None, iters: int = 30, tol: float = 1e-10, pen0: int = 0) -> np.ndarray:
    """Newton с обратным шагом по среднему QLIKE + ridge·|γ[pen0:]|²/2; off = ln base. Выпуклая задача."""
    X, rv, off = np.asarray(X, float), np.asarray(rv, float), np.asarray(off, float)
    n, k = X.shape
    if n == 0 or k == 0 or not (rv > 0).all() or not np.isfinite(X).all() or not np.isfinite(off).all():
        raise ValueError("QLIKE-подгонка: пусто, rv <= 0 или нефинитные X/off (n=%d, k=%d)" % (n, k))
    gam = np.zeros(k) if gam0 is None else np.array(gam0, float)
    pen = np.r_[np.zeros(pen0), np.ones(k - pen0)] * ridge

    def L(g):
        return mean_loss(g, X, rv, off) + 0.5 * float(np.sum(pen * g * g))

    for _ in range(iters):
        r = rv * np.exp(-(off + lin(X, gam)))
        grad = np.einsum("ij,i->j", X, 1.0 - r) / n + pen * gam
        H = np.einsum("ij,i,ik->jk", X, r, X) / n + np.diag(pen) + 1e-12 * np.eye(k)
        step = np.linalg.solve(H, grad)
        t, l0 = 1.0, L(gam)
        while L(gam - t * step) > l0 and t > 1e-8:
            t /= 2
        gam = gam - t * step
        if np.max(np.abs(t * step)) < tol:
            break
    return gam


def sigma2(X, off, gam) -> np.ndarray:
    return np.exp(np.asarray(off, float) + lin(X, gam))
