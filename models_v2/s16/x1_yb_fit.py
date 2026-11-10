"""Y-B, обучение Y3/Y4 (план раздел 3 и 8). F = X1·exp(g), g = θ·h, h = ln B_d − ln RV_d, B_d = Σ_k w_k·r²_k (по присутствующим барам).
Y3: w = n·softmax(Φγ) по присутствующим (среднее w = 1), β = (θ, γ₁…γ₄); старт θ = 0,01, γ ~ N(0; 0,1²) с seed. Y4: w фиксирован (профиль), обучается θ.
Цель: mean QLIKE(RV, X1·e^g) + λ(θ² + ‖γ‖²), QLIKE = u·e^{−g} + g − ln u − 1; аналитический градиент. L-BFGS-B ≤ 500 итераций; θ ∈ [−5; 5], γ свободны.
Выбор λ и переобучение — как Y-A (`x1_ya_fit`: LAMS, V, последние V закрытых пар, при равенстве больший λ). Чистые функции от массивов; D — словарь пар (u, Phi, M, R2, n, lrv, h4)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2.s16 import x1_ya_fit as FT

THETA0, GAMMA_SD, BOUND_T, MAXITER, SEEDS = 0.01, 0.1, 5.0, 500, (0, 1, 2, 3, 4)
NG = 4


def weights(gam, Phi, M, n) -> np.ndarray:
    """w = n·softmax_k(γᵀφ_k) по присутствующим (отсутствующие — 0). Деление (n·e)/Σe: при γ = 0 w ≡ 1 побитно."""
    z = np.where(M, Phi @ np.asarray(gam, float), -np.inf)
    e = np.exp(z - np.max(np.where(M, z, -np.inf), axis=1, keepdims=True)) * M
    return (n[:, None] * e) / e.sum(1, keepdims=True)


def h_y3(gam, D) -> np.ndarray:
    w = weights(gam, D["Phi"], D["M"], D["n"])
    return np.log((w * D["R2"]).sum(1)) - D["lrv"]


def loss_grad3(x, D, lam):
    """x = (θ, γ₁…γ₄). -> (L, ∇L)."""
    x = np.asarray(x, float)
    th, gam = x[0], x[1:]
    Phi, M, R2, n, u = D["Phi"], D["M"], D["R2"], D["n"], D["u"]
    p = weights(gam, Phi, M, n) / n[:, None]
    B = n * (p * R2).sum(1)
    h = np.log(B) - D["lrv"]
    g = th * h
    ug = u * np.exp(-g)
    L = float(np.mean(ug + g - np.log(u) - 1.0)) + lam * float(x @ x)
    dg = 1.0 - ug
    phibar = np.einsum("ns,nsf->nf", p, Phi)
    dB = n[:, None] * (np.einsum("ns,nsf->nf", p * R2, Phi) - (B / n)[:, None] * phibar)
    gr = np.concatenate([[np.mean(dg * h)], np.mean((dg * th / B)[:, None] * dB, axis=0)]) + 2 * lam * x
    return L, gr


def loss_grad4(x, D, lam):
    th = float(x[0])
    h = D["h4"]
    g = th * h
    ug = D["u"] * np.exp(-g)
    return float(np.mean(ug + g - np.log(D["u"]) - 1.0)) + lam * th * th, np.array([np.mean((1.0 - ug) * h) + 2 * lam * th])


def start(mode: str, seed: int) -> np.ndarray:
    if mode == "Y4":
        return np.array([THETA0])
    return np.concatenate([[THETA0], np.random.default_rng(seed).normal(0.0, GAMMA_SD, NG)])


def fit(D, lam, mode, seed=0) -> np.ndarray:
    lg = loss_grad3 if mode == "Y3" else loss_grad4
    x0 = start(mode, seed)
    r = minimize(lg, x0, args=(D, lam), jac=True, method="L-BFGS-B", bounds=[(-BOUND_T, BOUND_T)] + [(None, None)] * (len(x0) - 1), options=dict(maxiter=MAXITER))
    return np.asarray(r.x, float)


def h_of(x, D, mode) -> np.ndarray:
    return D["h4"] if mode == "Y4" else h_y3(x[1:], D)


def qlike_mean(x, D, mode) -> float:
    g = x[0] * h_of(x, D, mode)
    return float(np.mean(D["u"] * np.exp(-g) + g - np.log(D["u"]) - 1.0))


def sub(D: dict, idx) -> dict:
    return {k: v[idx] for k, v in D.items()}


def choose_lambda(pool, D, mode, seed=0, lams=FT.LAMS) -> float:
    """pool — индексы закрытых пар в порядке дат; обучение без последних V -> потеря на последних V; минимум, ничья — больший λ."""
    v = FT.n_valid(len(pool))
    val, tr = FT.valid_block(pool, v), pool[: len(pool) - v]
    best, lam_best = np.inf, lams[-1]
    for lam in sorted(lams, reverse=True):
        q = qlike_mean(fit(sub(D, tr), lam, mode, seed), sub(D, val), mode)
        if q < best:
            best, lam_best = q, lam
    return float(lam_best)


def refit(pool, D, mode, seed=0) -> tuple:
    """-> (λ, x) по закрытым парам pool."""
    lam = choose_lambda(pool, D, mode, seed)
    return lam, fit(sub(D, pool), lam, mode, seed)
