"""Y-A, обучение Y1/Y2 (план раздел 3 и 8, Ж-243). F = X1·exp(g), g = (a_n·(1 − e_d) + a_e·e_d)·s_d·exp(−κ·Δ_d); Y2: κ ≡ 0.
Цель: mean QLIKE(RV, X1·e^g) + λ·(a_n² + a_e²) (κ не штрафуется); QLIKE = u·e^{−g} + g − ln u − 1, u = RV/X1. Аналитический градиент: dl/dg = 1 − u·e^{−g}.
L-BFGS-B (scipy), ≤ 500 итераций; границы κ ∈ [0; 5], a ∈ [−5; 5]; старт a = 0, κ = 0,5 (Y1) — κ оживает только после a (при a = 0 градиент по κ равен 0).
λ ∈ LAMS: обучение на закрытых парах без последних V, средняя QLIKE на последних V (V = 126 при n ≥ 252, иначе ⌊n/2⌋); минимум, при равенстве — больший λ; затем переобучение на всех закрытых парах. Чистые функции."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

LAMS, V_FULL, MAXITER, KAPPA0, BOUND_A, BOUND_K = (0.0, 0.001, 0.01, 0.1), 126, 500, 0.5, 5.0, 5.0
MODES = {"Y1": 3, "Y2": 2}          # число обучаемых параметров


def _g(th, e, s, delta, mode):
    k = th[2] if mode == "Y1" else 0.0
    w = s * np.exp(-k * delta)
    return (th[0] * (1.0 - e) + th[1] * e) * w, w


def loss_grad(th, u, e, s, delta, lam, mode):
    """Средняя QLIKE + λ·(a_n² + a_e²) и аналитический градиент по th = (a_n, a_e[, κ])."""
    th = np.asarray(th, float)
    g, w = _g(th, e, s, delta, mode)
    ug = u * np.exp(-g)
    L = float(np.mean(ug + g - np.log(u) - 1.0)) + lam * float(th[0] ** 2 + th[1] ** 2)
    dg = 1.0 - ug
    gr = [np.mean(dg * (1.0 - e) * w) + 2 * lam * th[0], np.mean(dg * e * w) + 2 * lam * th[1]]
    if mode == "Y1":
        gr.append(np.mean(dg * (-delta) * g))
    return L, np.asarray(gr, float)


def fit(u, e, s, delta, lam, mode) -> np.ndarray:
    """-> th. Старт: a = 0 (и κ = 0,5 у Y1)."""
    x0 = np.array([0.0, 0.0, KAPPA0][: MODES[mode]])
    bounds = [(-BOUND_A, BOUND_A)] * 2 + ([(0.0, BOUND_K)] if mode == "Y1" else [])
    r = minimize(loss_grad, x0, args=(u, e, s, delta, lam, mode), jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=MAXITER))
    return np.asarray(r.x, float)


def qlike_mean(th, u, e, s, delta, mode) -> float:
    g, _ = _g(np.asarray(th, float), e, s, delta, mode)
    return float(np.mean(u * np.exp(-g) + g - np.log(u) - 1.0))


def n_valid(n: int) -> int:
    return V_FULL if n >= 2 * V_FULL else n // 2


def valid_block(pool: np.ndarray, v: int) -> np.ndarray:
    """Строки внутренней валидации: последние V ЗАКРЫТЫХ пар (мутация: V пар ПОСЛЕ последней закрытой — будущее)."""
    return pool[-v:]


def choose_lambda(pool: np.ndarray, u, e, s, delta, mode, lams=LAMS) -> float:
    """pool — индексы закрытых пар в порядке дат. Обучение на pool без последних V -> потеря на последних V; минимум, ничья — больший λ."""
    v = n_valid(len(pool))
    val = valid_block(pool, v)
    tr = pool[: len(pool) - v]
    best, lam_best = np.inf, lams[-1]
    for lam in sorted(lams, reverse=True):
        th = fit(u[tr], e[tr], s[tr], delta[tr], lam, mode)
        q = qlike_mean(th, u[val], e[val], s[val], delta[val], mode)
        if q < best:
            best, lam_best = q, lam
    return float(lam_best)


def refit(pool: np.ndarray, u, e, s, delta, mode) -> tuple:
    """-> (λ, th) по закрытым парам pool: выбор λ, затем переобучение на всех закрытых."""
    lam = choose_lambda(pool, u, e, s, delta, mode)
    return lam, fit(u[pool], e[pool], s[pool], delta[pool], lam, mode)
