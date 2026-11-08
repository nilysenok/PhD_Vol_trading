"""V4-I: оценка компонент по закрытым парам панели 17 бумаг (Ж-222 п. 10). Гиперпараметры фиксированы объявлением: λ = κ = 250 пар-эквивалентов, поправка логнормальности exp(σ̂²/2).
HAR-C: ln C_{t⁺} = b0 + βᵀz + α_i + ε, z — стандартизованные ln C_d, ln C̄_5, ln C̄_22; гребень λ на наклонах (β), стягивание свободных членов бумаг κ·Σα_i² (b0 не штрафуется); Ĉ = exp(f + σ̂²/2).
Скачок: логистическая регрессия события J_{t⁺} > 0 на z = (EWMA₂₀ частоты, J_d/RV_d, ln RV_d), те же штрафы; Ньютон. m̂_i = (n_i·m_i + κ·m_пан)/(n_i + κ), m_i — EWMA₆₀ величины J при J > 0, m_пан — среднее J при J > 0 по закрытым парам панели.
I3: θ ∈ [0, 1] — минимум среднего QLIKE K8·exp(θu), u = ln S − ln K8, + гребень (λ/2)θ², λ = κ·mean(u²), раз в месяц."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

KAPPA = 250.0
LAM = 250.0
GTOL = 1e-8


def _mm(a, b):
    return np.einsum("ij,jk->ik", a, b)         # matmul numpy + Accelerate (macOS) даёт ложные RuntimeWarning (см. v4_c_fit.mv)


def design(Z, tid, n):
    """[1, Z, дамми бумаг] и диагональ штрафа [0, λ…, κ…]."""
    N, k = Z.shape
    D = np.zeros((N, n))
    D[np.arange(N), tid] = 1.0
    return np.hstack([np.ones((N, 1)), Z, D]), np.concatenate([[0.0], np.full(k, LAM), np.full(n, KAPPA)])


def standardize(X):
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def har_fit(X, y, tid, n) -> dict:
    mu, sd = standardize(X)
    A, pen = design((X - mu) / sd, tid, n)
    th = np.linalg.solve(_mm(A.T, A) + np.diag(pen), _mm(A.T, y[:, None])[:, 0])
    r = y - _mm(A, th[:, None])[:, 0]
    return {"mu": mu, "sd": sd, "theta": th, "sig2": float(np.mean(r * r)), "n": n}


def har_predict(p, X, tix):
    """exp(f + σ̂²/2); бумага без обучения (tix = −1) — без своего свободного члена."""
    k = X.shape[1]
    f = p["theta"][0] + _mm((X - p["mu"]) / p["sd"], p["theta"][1:1 + k, None])[:, 0] + np.where(tix >= 0, p["theta"][1 + k:][np.maximum(tix, 0)], 0.0)
    return np.exp(f + 0.5 * p["sig2"])


def logit_fit(X, y, tid, n) -> dict:
    mu, sd = standardize(X)
    A, pen = design((X - mu) / sd, tid, n)
    th = np.zeros(A.shape[1])
    th[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    ok, gmax = False, np.inf
    for it in range(60):
        p = 1.0 / (1.0 + np.exp(-np.clip(_mm(A, th[:, None])[:, 0], -30, 30)))
        g = _mm(A.T, (p - y)[:, None])[:, 0] + pen * th
        gmax = float(np.abs(g).max() / len(y))
        if gmax < GTOL:
            ok = True
            break
        H = _mm(A.T * (p * (1 - p) + 1e-12), A) + np.diag(pen)
        th = th - np.linalg.solve(H, g)
    return {"mu": mu, "sd": sd, "theta": th, "ok": ok, "grad": gmax, "nit": it + 1, "n": n}


def logit_predict(p, X, tix):
    k = X.shape[1]
    f = p["theta"][0] + _mm((X - p["mu"]) / p["sd"], p["theta"][1:1 + k, None])[:, 0] + np.where(tix >= 0, p["theta"][1 + k:][np.maximum(tix, 0)], 0.0)
    return 1.0 / (1.0 + np.exp(-np.clip(f, -30, 30)))


def shrink_m(m_i, n_i, m_pan):
    """m̂ = (n·m_i + κ·m_пан)/(n + κ); нет собственной оценки (NaN) — панельная."""
    m_i = np.asarray(m_i, float)
    n_i = np.where(np.isfinite(m_i), np.asarray(n_i, float), 0.0)
    return (n_i * np.where(np.isfinite(m_i), m_i, 0.0) + KAPPA * m_pan) / (n_i + KAPPA)


def theta_fit(x_ratio, u, kappa: float = KAPPA) -> float:
    """θ ∈ [0, 1] для F = K8·exp(θu): минимум среднего QLIKE x·e^{−θu} + θu + гребень; x = RV/K8. Выпуклая задача на отрезке."""
    x, u = np.asarray(x_ratio, float), np.asarray(u, float)
    N, lam = len(x), kappa * float(np.mean(u * u))
    f = lambda t: float(np.sum(x * np.exp(-t * u) + t * u) + 0.5 * lam * t * t) / N
    return float(minimize_scalar(f, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10}).x)
