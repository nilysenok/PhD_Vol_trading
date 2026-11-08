"""V4-H: поправка к равновесной смеси F = K8·exp(g), g = θᵀz̃ (без свободного члена), z̃ — признак, ортогонализованный к базовой информации (log H, log M, ℓ, |ℓ|) ТОЛЬКО на закрытом окне оценки.
Цель — средний QLIKE `x·exp(−g) + g` (x = RV/K8; ln K8 — константа) + гребень (λ/2)‖θ‖² (+ (c/2)Σ‖η_i‖² для иерархического H8: θ_i = θ + η_i), λ = c = κ·s², κ = 250 пар-эквивалентов, s² = mean z̃² на окне.
Решатель — L-BFGS-B с аналитическим градиентом; флаг сходимости max|∇| < 1e-6. Гиперпараметры (KAPPA, LAM_PROJ, CLIP) фиксированы объявлением и не выбираются по результатам."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2.s11 import v4_c_fit as CF

KAPPA = 250.0
LAM_PROJ = 250.0          # гребень проекции z на базовые величины (в единицах стандартизованных величин, пар-эквивалентов)
CLIP = 0.5                # g ∈ [−0,5; 0,5] при прогнозе
GTOL = 1e-6
BOUND = 20.0


def clip_g(g):
    """Обрезка поправки при прогнозе (в обучении не применяется — гладкая цель)."""
    return np.clip(g, -CLIP, CLIP)


def _obj(x_ratio, Z, tid, n, hier, lam, c):
    N, k = Z.shape

    def f(p):
        th, eta = p[:k], (p[k:].reshape(n, k) if hier else None)
        g = CF.mv(Z, th) + (np.einsum("ij,ij->i", Z, eta[tid]) if hier else 0.0)
        e = x_ratio * np.exp(-g)
        val = (np.sum(e + g) + 0.5 * lam * float(th @ th) + (0.5 * c * float((eta * eta).sum()) if hier else 0.0)) / N
        r = 1.0 - e
        gr = np.empty_like(p)
        gr[:k] = (np.einsum("ij,i->j", Z, r) + lam * th) / N
        if hier:
            gi = np.zeros((n, k))
            np.add.at(gi, tid, Z * r[:, None])
            gr[k:] = ((gi + c * eta) / N).ravel()
        return val, gr
    return f


def fit(x_ratio, Z, tid=None, n=0, hier=False, x0=None, kappa=KAPPA) -> dict:
    """Оценка на закрытых парах. Z (N, k) — z̃ (стандартизованные остатки), tid — номера бумаг 0..n−1 (для hier). -> {theta (k), eta (n, k) | None, ok, grad, nit}."""
    N, k = Z.shape
    s2 = float(np.mean(Z * Z)) if N else 1.0
    lam = kappa * s2
    size = k + (n * k if hier else 0)
    x0 = np.zeros(size) if x0 is None or len(x0) != size else np.asarray(x0, float)
    r = minimize(_obj(np.asarray(x_ratio, float), Z, tid, n, hier, lam, lam), x0, jac=True, method="L-BFGS-B", bounds=[(-BOUND, BOUND)] * size,
                 options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 20})
    return {"theta": r.x[:k].copy(), "eta": r.x[k:].reshape(n, k).copy() if hier else None, "ok": bool(float(np.abs(r.jac).max()) < GTOL), "grad": float(np.abs(r.jac).max()), "nit": int(r.nit), "_x": r.x}


def orthogonalize_fit(Zr, B, lam=None) -> dict:
    """Параметры проекции по окну оценки: стандартизация Z и B, гребневая регрессия Z на B (центрированные: свободного члена нет), стандартизация остатка. Z, B — только строки закрытых пар."""
    lam = LAM_PROJ if lam is None else lam
    mz, sz = CF.standardize(Zr)
    mb, sb = CF.standardize(B)
    Zs, Bs = (Zr - mz) / sz, (B - mb) / sb
    mm = lambda a, b: np.einsum("ij,jk->ik", a, b)        # matmul numpy + Accelerate (macOS) даёт ложные RuntimeWarning, см. v4_c_fit.mv
    gam = np.linalg.solve(mm(Bs.T, Bs) + lam * np.eye(B.shape[1]), mm(Bs.T, Zs))
    R = Zs - mm(Bs, gam)
    mr, sr = CF.standardize(R)
    return {"mz": mz, "sz": sz, "mb": mb, "sb": sb, "gam": gam, "mr": mr, "sr": sr}


def orthogonalize_apply(p: dict, Zr, B):
    """z̃ для любых строк по параметрам окна оценки; нефинитный признак -> 0 (среднее окна), как в C."""
    fin = np.isfinite(Zr)
    Zs = np.where(fin, (Zr - p["mz"]) / p["sz"], 0.0)
    Bs = (B - p["mb"]) / p["sb"]
    R = (Zs - np.einsum("ij,jk->ik", Bs, p["gam"]) - p["mr"]) / p["sr"]
    return np.where(fin, R, 0.0)
