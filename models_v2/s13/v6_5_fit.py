"""V6-5, G41: робастные цели F1 и F2 (план V6, п. 7; уточнения дирижёра, объявлены до счёта F). Форма та же, что у LIN (g = θ₀ + θ'z, clip в прогнозе — движок листа 1a), цель другая.
Среда = календарный год t⁺ ЗАКРЫТЫХ пар (среда < 20 пар сливается с предыдущей; первая — со следующей); Q_e(θ) — средний QLIKE пар среды.
F1: L = mean_e Q_e + 0,5·sd_e Q_e (ddof = 0) + λ‖θ‖²;  F2: L = (1/η)·ln Σ_e exp(η·Q_e) + λ‖θ‖², η = 20;  θ₀ не штрафуется. L-BFGS-B от решения LIN той же даты и того же λ, аналитический градиент.
λ — та же процедура, что у LIN: обучение на всех закрытых парах, кроме последних 252, проверка на последних 252 (средний QLIKE с clip), сетка {1e-3, 1e-2, 1e-1, 1}, затем переоценка на всех закрытых парах."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models_v2.s13 import v6_1_fit as FT

MIN_ENV, ETA, KSD = 20, 20.0, 0.5
KINDS = ("F1", "F2")
OPT = dict(maxiter=2000, ftol=1e-15, gtol=1e-9)


def env_labels(tp, min_n: int = MIN_ENV) -> np.ndarray:
    """Номер среды (0…K−1) каждой пары по году t⁺. tp — ТОЛЬКО закрытые пары на дату переоценки (МУТАНТ «среда из будущих пар» подаёт сюда лишнее)."""
    y = pd.DatetimeIndex(tp).year.to_numpy()
    groups, carry = [], []
    for yr in np.unique(y):
        small = int((y == yr).sum()) < min_n
        if small and groups:
            groups[-1].append(int(yr))
        elif small:
            carry.append(int(yr))
        else:
            groups.append(carry + [int(yr)])
            carry = []
    if carry:
        groups = [carry] if not groups else groups[:-1] + [groups[-1] + carry]
    lab = np.zeros(len(y), int)
    for k, gr in enumerate(groups):
        lab[np.isin(y, gr)] = k
    return lab


def env_q(A: np.ndarray, e: np.ndarray, th: np.ndarray, lab: np.ndarray) -> tuple:
    """-> (Q_e (K), G_e (K × p)): средний QLIKE и его градиент по θ в каждой среде."""
    k = int(lab.max()) + 1
    M = np.zeros((len(e), k))
    M[np.arange(len(e)), lab] = 1.0
    M /= M.sum(0)
    g = np.einsum("ij,j->i", A, th)
    q, r = FT.qlike_g(e, g), FT.grad_hess(e, g)[0]
    return M.T @ q, np.einsum("ik,i,ij->kj", M, r, A)


def robust_obj(th, A, e, lab, lam: float, kind: str) -> tuple:
    """-> (L, ∇L). kind: F1 | F2 | LIN (LIN — среднее по парам при одной среде; для проверки совпадения)."""
    Q, G = env_q(A, e, th, lab)
    if kind == "F2":
        s = ETA * Q
        m = s.max()
        p = np.exp(s - m)
        val, grad = m / ETA + np.log(p.sum()) / ETA, (p / p.sum()) @ G
    else:
        sd = Q.std()
        val, grad = Q.mean() + (KSD * sd if kind == "F1" else 0.0), G.mean(0)
        if kind == "F1" and sd > 1e-12:
            grad = grad + KSD * ((Q - Q.mean()) @ G) / (len(Q) * sd)
    pen = np.r_[0.0, np.full(len(th) - 1, lam)]
    return float(val + lam * (th[1:] ** 2).sum()), grad + 2 * pen * th


def fit_robust_fixed(Z: np.ndarray, e: np.ndarray, lab: np.ndarray, lam: float, kind: str) -> tuple:
    """θ для фиксированного λ: L-BFGS-B от решения LIN. -> (θ, info: nit, grad_norm (2-норма ∇L в решении), success, L)."""
    A = np.column_stack([np.ones(len(Z)), Z])
    th0 = FT.fit_lin_fixed(Z, e, lam)
    f = lambda th: robust_obj(th, A, e, lab, lam, kind)
    res = minimize(f, th0, jac=True, method="L-BFGS-B", options=OPT)
    th = res.x if res.fun <= f(th0)[0] else th0
    L, gr = f(th)
    return th, dict(nit=int(res.nit), grad_norm=float(np.linalg.norm(gr)), success=bool(res.success), L=L)


class RobModel(FT.LinModel):
    def __init__(self, mu, sd, theta, lam, scores, info):
        super().__init__(mu, sd, theta, lam, scores)
        self.info = info


def select_lambda(X: np.ndarray, e: np.ndarray, tp, kind: str, grid=FT.GRID, hold: int = FT.HOLD) -> tuple:
    """Индекс λ по среднему QLIKE (clip g) на последних `hold` парах; обучение, его нормировка и среды — только на предыдущих парах. -> (idx, scores)."""
    tr, va = FT.split_es(len(e), hold)
    mu, sd = FT.norm_stats(X[tr])
    Zt, Zv = (X[tr] - mu) / sd, (X[va] - mu) / sd
    lab = env_labels(np.asarray(tp)[tr])
    sc = []
    for lam in grid:
        th = fit_robust_fixed(Zt, e[tr], lab, lam, kind)[0]
        sc.append(float(FT.qlike_g(e[va], FT.clip_g(np.einsum("ij,j->i", FT._design(Zv), th))).mean()))
    return int(np.argmin(sc)), sc


def fit_robust(X: np.ndarray, e: np.ndarray, tp, kind: str, lam_shift: int = 0) -> RobModel:
    """F1/F2 на закрытых парах (X, e, tp — t⁺ тех же пар). lam_shift = ±1 — соседняя точка сетки (чувствительность, не конфигурация)."""
    if len(e) < FT.MIN_TRAIN:
        raise ValueError("мало закрытых пар: %d < %d" % (len(e), FT.MIN_TRAIN))
    i, sc = select_lambda(X, e, tp, kind)
    lam = FT.GRID[int(np.clip(i + lam_shift, 0, len(FT.GRID) - 1))]
    mu, sd = FT.norm_stats(X)
    lab = env_labels(tp)
    th, info = fit_robust_fixed((X - mu) / sd, e, lab, lam, kind)
    return RobModel(mu, sd, th, lam, sc, {**info, "n_env": int(lab.max()) + 1})
