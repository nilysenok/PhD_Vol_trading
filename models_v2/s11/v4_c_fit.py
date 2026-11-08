"""V4-C: гейт условной смеси экспертов. F = H·exp(w·ell), ell = ln(M/H), w = σ(α + δ_i + βᵀz) — вес на M (w = 0,5 — приор B2). Признаки z стандартизованы по закрытому окну оценки.
Цель — средний QLIKE объединённого прогноза (rv/F + ln F) + штрафы: (c/2)·Σδ_i² (c = κ·s²), (l/2)·α² (приор α = 0, w = 0,5) и гребень (l/2)·‖β‖² к 0 (β = 0 — постоянный вес); κ = λ = 250 пар-эквивалентов,
s² = mean ell² на закрытых парах (кривизна одной пары по весу, как в B). Решатель — L-BFGS-B с аналитическим градиентом по (α, δ, β); флаг сходимости — max|∇| < 1e-6. Гиперпараметры фиксированы (v4_b_fit)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from models_v2.s11 import v4_b_fit as FT

GTOL = 1e-6


BOUND = 20.0          # ящик по параметрам логита: страховка линейного поиска от расхождения; на реальных данных не достигается (проверяется флагом сходимости)


def sigmoid(x):
    return expit(x)


def forward(lH, ell, eta):
    """-> (F, w) по строкам: w = σ(η), F = exp(lH + w·ell)."""
    w = sigmoid(eta)
    return np.exp(lH + w * ell), w


def logit_w_of_b2(w_h):
    """α, при котором постоянный вес на M равен 1 − w_h: F = H^w_h · M^(1−w_h) (тождество с B2)."""
    return np.log((1.0 - np.asarray(w_h, float)) / np.asarray(w_h, float))


def mv(Z, b):
    """Z·b через einsum: matmul numpy 2.0.2 + Accelerate (macOS) даёт ложные RuntimeWarning (divide by zero / overflow) на конечных данных — проверено отдельным скриптом."""
    return np.einsum("ij,j->i", Z, b)


def _obj(rv, lH, ell, Z, tid, n, c, l):
    N, k = len(rv), Z.shape[1]

    def f(x):
        a, dl, b = x[0], x[1:n + 1], x[n + 1:]
        eta = a + dl[tid] + mv(Z, b) if k else a + dl[tid]
        w = sigmoid(eta)
        lF = lH + w * ell
        e = rv * np.exp(-lF)
        ge = (1.0 - e) * ell * w * (1.0 - w)
        val = (np.sum(e + lF) + 0.5 * c * float(dl @ dl) + 0.5 * l * a * a + 0.5 * l * float(b @ b)) / N
        g = np.empty_like(x)
        g[0] = (ge.sum() + l * a) / N
        g[1:n + 1] = (np.bincount(tid, ge, minlength=n) + c * dl) / N
        g[n + 1:] = (np.einsum("ij,i->j", Z, ge) + l * b) / N if k else 0.0
        return val, g
    return f


def fit(rv, lH, ell, Z, tid, n, x0=None, kappa=None, lam=None) -> dict:
    """Оценка на закрытых парах. Z (N, k) уже стандартизована (k может быть 0), tid — номера бумаг 0..n−1. -> {alpha, delta (n), beta (k), ok, grad, nit}."""
    kappa, lam = FT.KAPPA if kappa is None else kappa, FT.LAM if lam is None else lam
    s2 = float(np.mean(ell ** 2))
    k = Z.shape[1]
    x0 = np.zeros(n + 1 + k) if x0 is None or len(x0) != n + 1 + k else np.asarray(x0, float)
    r = minimize(_obj(rv, lH, ell, Z, tid, n, kappa * s2, lam * s2), x0, jac=True, method="L-BFGS-B", bounds=[(-BOUND, BOUND)] * len(x0),
                 options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 20})
    g = float(np.abs(r.jac).max())              # непроекционный градиент: на границе ящика флаг сходимости = False
    return {"alpha": float(r.x[0]), "delta": r.x[1:n + 1].copy(), "beta": r.x[n + 1:].copy(), "ok": bool(g < GTOL), "grad": g, "nit": int(r.nit), "_x": r.x}


def standardize(Z: np.ndarray) -> tuple:
    """-> (mu, sd) по строкам окна оценки; нулевой разброс -> sd = 1 (признак константен, β на него не влияет)."""
    mu = Z.mean(0) if len(Z) else np.zeros(Z.shape[1])
    sd = Z.std(0) if len(Z) else np.ones(Z.shape[1])
    return mu, np.where(sd > 1e-12, sd, 1.0)
