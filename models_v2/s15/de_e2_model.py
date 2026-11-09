"""E2 (X2): модель хвостовой смеси, numpy + L-BFGS (без torch/sklearn). F = H·m, m = c_n + p·(c_s − c_n) (равно (1−p)·c_n + p·c_s; запись через разность даёт m = 1 ровно при c_n = c_s = 1 → F = H побитно).
c_k = exp(c_max·tanh(a_k·[1, Zs])), c_max = 1; p = σ(u·[1, Xs]); Zs — первые NZ колонок стандартизованной матрицы Xs (NP колонок).
Потеря: mean(ln F + RV/F) [QLIKE с точностью до константы] + β·BCE(p, y) + λ₂·Σθ² (λ₂ = 1e-3, как V7). Параметры: a_n (NZ+1), a_s (NZ+1), u (NP+1)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from models_v2.s15.de_e2_feat import NP, NZ

CMAX, LAM2, BETAS, MAXITER, SEEDS = 1.0, 1e-3, (0.1, 1.0), 300, (0, 1, 2, 3, 4)
NC = NZ + 1
NTH = 2 * NC + NP + 1


def split(th):
    return th[:NC], th[NC:2 * NC], th[2 * NC:]


def design(Xs: np.ndarray) -> tuple:
    """-> (Xc с единицей (n, NZ+1), Xp с единицей (n, NP+1))."""
    one = np.ones((len(Xs), 1))
    return np.hstack([one, Xs[:, :NZ]]), np.hstack([one, Xs])


def forward(th, Xs) -> dict:
    an, as_, u = split(np.asarray(th, float))
    Xc, Xp = design(Xs)
    tn, ts = np.tanh(Xc @ an), np.tanh(Xc @ as_)
    cn, cs, p = np.exp(CMAX * tn), np.exp(CMAX * ts), expit(Xp @ u)
    return dict(cn=cn, cs=cs, p=p, m=cn + p * (cs - cn), tn=tn, ts=ts, Xc=Xc, Xp=Xp)


def loss_grad(th, Xs, R, y, beta) -> tuple:
    """-> (L, ∂L/∂θ); R = RV/H, y — метка стресса. ∂/∂m [ln m + R/m] = 1/m − R/m²."""
    th, n = np.asarray(th, float), len(R)
    f = forward(th, Xs)
    cn, cs, p, m = f["cn"], f["cs"], f["p"], f["m"]
    z = f["Xp"] @ split(th)[2]
    L = np.mean(np.log(m) + R / m) + beta * np.mean(np.logaddexp(0.0, z) - y * z) + LAM2 * float(th @ th)
    gm = (1.0 / m - R / m ** 2) / n
    gan = f["Xc"].T @ (gm * (1.0 - p) * cn * CMAX * (1.0 - f["tn"] ** 2))
    gas = f["Xc"].T @ (gm * p * cs * CMAX * (1.0 - f["ts"] ** 2))
    gu = f["Xp"].T @ (gm * (cs - cn) * p * (1.0 - p) + beta * (p - y) / n)
    return float(L), np.concatenate([gan, gas, gu]) + 2.0 * LAM2 * th


def fit(Xs, R, y, beta: float, seed: int, maxiter: int = MAXITER) -> np.ndarray:
    th0 = 0.1 * np.random.RandomState(seed).randn(NTH)
    r = minimize(loss_grad, th0, args=(Xs, R, y, beta), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return r.x


def predict(models, Xs) -> dict:
    """Ансамбль seeds: ln m и p, c_n, c_s — средние по моделям. F = H·exp(lnm)."""
    fs = [forward(th, Xs) for th in models]
    return dict(lnm=np.mean([np.log(f["m"]) for f in fs], axis=0), p=np.mean([f["p"] for f in fs], axis=0),
                cn=np.mean([f["cn"] for f in fs], axis=0), cs=np.mean([f["cs"] for f in fs], axis=0))


def qlike_val(R, lnm) -> float:
    """QLIKE (RV/F − ln(RV/F) − 1) ансамбля на валидации; R = RV/H."""
    r = np.asarray(R, float) / np.exp(lnm)
    return float(np.mean(r - np.log(r) - 1.0))
