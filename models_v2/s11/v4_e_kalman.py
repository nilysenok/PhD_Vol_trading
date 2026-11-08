"""V4-E, E5: фильтр Калмана «локальный уровень» для u = ln(RV/F0): u_t = μ_t + ε (r), μ_{t+1} = μ_t + η (q). Строго фильтр (данные ≤ d), не сглаживание.
Прогнозное распределение u_{d+1}: N(m, s²), s² = r̂·(P' + ρ + 1) (P' — апостериорная дисперсия уровня в единицах r, ρ = q/r); b = m + s²/2 — оптимум QLIKE для логнормального RV/F0.
ρ и r̂ оцениваются максимумом правдоподобия (концентрированного по r) раз в год (первый торговый день года; пуск — первый день, когда у всех бумаг ≥ MIN_PAIRS закрытых пар) по закрытым парам;
ρ бумаги стягивается к панельному в лог-масштабе: ln ρ_i = (n_i ln ρ̂_i + κ ln ρ_p)/(n_i + κ). Начальное наблюдение — диффузная инициализация (в правдоподобие не входит)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from models_v2.s11 import v4_e_calib as CAL

LRHO = (np.log(1e-4), np.log(1.0))


def filt(u, rho):
    """-> (m, P', v2/Fv, log Fv) по наблюдениям; первое — диффузное (m = u₀, P' = 1)."""
    n = len(u)
    m, P = np.empty(n), np.empty(n)
    q, lf = np.zeros(n), np.zeros(n)
    if n == 0:
        return m, P, q, lf
    mm, pp = float(u[0]), 1.0
    m[0], P[0] = mm, pp
    for t in range(1, n):
        pr = pp + rho
        fv = pr + 1.0
        v = u[t] - mm
        mm += pr / fv * v
        pp = pr / fv
        m[t], P[t], q[t], lf[t] = mm, pp, v * v / fv, np.log(fv)
    return m, P, q, lf


def nll(u, rho) -> tuple:
    """-> (−loglik без констант, r̂) концентрированного правдоподобия."""
    _, _, q, lf = filt(u, rho)
    n = max(len(u) - 1, 1)
    rhat = float(q[1:].sum() / n) if len(u) > 1 else 1.0
    return 0.5 * n * np.log(max(rhat, 1e-300)) + 0.5 * float(lf[1:].sum()), rhat


def fit_rho(us) -> float:
    """Общий ρ по списку рядов (панель): минимум суммы концентрированных nll."""
    f = lambda lr: sum(nll(u, np.exp(lr))[0] for u in us if len(u) > 2)
    return float(np.exp(minimize_scalar(f, bounds=LRHO, method="bounded", options={"xatol": 1e-3}).x))


def params(us, member, kappa=CAL.KAPPA) -> tuple:
    """us — ряды u закрытых пар каждой бумаги; member — бумага входит в панель. -> (rho_i, rhat_i, rho_panel)."""
    rp = fit_rho([u for u, m in zip(us, member) if m])
    rho, rhat = [], []
    for u in us:
        n = len(u)
        r_i = fit_rho([u]) if n > 2 else rp
        ri = float(np.exp((n * np.log(r_i) + kappa * np.log(rp)) / (n + kappa)))
        rho.append(ri)
        rhat.append(nll(u, ri)[1] if n > 2 else 1.0)
    return rho, rhat, rp


def b_from_state(m, P, rho, rhat):
    s2 = rhat * (P + rho + 1.0)
    return np.clip(m + 0.5 * s2, -CAL.B_CLIP, CAL.B_CLIP)
