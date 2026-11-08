"""V5 лист 1a: ежемесячная оценка θ = (ρ, α, β) и путь состояния (план п. 3.1). Гиперпараметры объявлены и не подбираются: границы ρ ∈ [0; 0,99], α, β ∈ [0; 0,3]; приор θ₀ = (0,9; 0; 0); κ = 250 пар-эквивалентов; < 60 закрытых дней — θ₀ (тогда F = основа).
Целевая функция — как в V4 (`v4_b_fit._obj`): (Σ QLIKE + ½·κ·Σ_j s2_j·(θ_j − θ0_j)²) / N, N — число закрытых пар. Масштаб s2_j — аналог `v4_b_fit.curvature`: средний квадрат производной ln F по θ_j (= кривизна QLIKE по ln F ≈ 1)
в опорной точке θ_ref — середине ящика (0,495; 0,15; 0,15); в θ₀ производная по ρ тождественно нулевая (u ≡ 0), поэтому опора вынесена из θ₀. Для GS1/GS2 (передачи нет) β = 0 фиксирована.
Закрытые пары месяца m: цель τ < первого числа месяца (`closed_n`). Путь u месяца — фильтр от начала ряда с θ_m и W_m, λ_m, оценёнными по скорам s̃, ВЫДАННЫМИ прежними месяцами (живая лента `Xlive`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models_v2.s12 import v5_gs_core as CO, v5_gs_graph as GR

KAPPA = 250.0
TH0 = np.array([0.9, 0.0, 0.0])
LO = np.array([0.0, 0.0, 0.0])
HI = np.array([0.99, 0.3, 0.3])
REF = np.array([0.495, 0.15, 0.15])
MIN_DAYS = 60
VAR = {"GS1": "A", "GS2": "A", "GS3": "B", "GS4": "C", "GS5": "D"}


def closed_n(axis: pd.DatetimeIndex, m0: pd.Timestamp) -> int:
    """Число дат оси, закрытых к началу месяца: τ < m0 (тест подменяет на «закрыто + 1»)."""
    return int(np.searchsorted(axis.to_numpy(), m0.to_datetime64(), side="left"))


def _free(var: str) -> list:
    return [0, 1] if var == "A" else [0, 1, 2]


def fit_theta(P, base, n_use, W, c, clip, lm, fm, var, prev=None) -> np.ndarray:
    """θ по закрытым парам первых n_use дат: L-BFGS-B из θ₀ и из прошлого решения, лучший по значению. Аналитический градиент через рекурсию."""
    free, ref = _free(var), REF.copy()
    ref[2] = REF[2] if var != "A" else 0.0
    acc0 = CO.run(P.rv, base, P.pres, ref, W, c, clip, lm, fm, n_use)[3]
    s2 = np.maximum(acc0[5:8] / max(acc0[1], 1.0), 1e-12)

    def f(z):
        th = TH0.copy()
        th[free] = z
        acc = CO.run(P.rv, base, P.pres, th, W, c, clip, lm, fm, n_use)[3]
        d = th - TH0
        val = (acc[0] + 0.5 * KAPPA * float((s2 * d * d)[free].sum())) / acc[1]
        return val, ((acc[2:5] + KAPPA * s2 * d)[free]) / acc[1]
    starts = [TH0[free]] + ([prev[free]] if prev is not None else [])
    best = None
    for x0 in starts:
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(LO[free], HI[free])), options={"maxiter": 400, "ftol": 1e-13, "gtol": 1e-9})
        best = r if best is None or r.fun < best.fun else best
    th = TH0.copy()
    th[free] = best.x
    return th


def run_config(P, base, var: str, clip: bool, excl: int | None = None) -> dict:
    """Весь путь конфигурации: по месяцам дня прогноза. excl — индекс исключённой бумаги (LOTO). -> dict(U, S, X, params DataFrame)."""
    N, m = P.rv.shape
    lm, fm = np.ones(m, bool), np.ones(m, bool)
    if excl is not None:
        lm[excl] = fm[excl] = False
    U, S, X = (np.full((N, m), np.nan) for _ in range(3))
    live = np.full((N, m), np.nan)
    rows, th = [], TH0.copy()
    for mo in sorted(set(P.mon[P.pres].tolist())):
        m0 = pd.Timestamp(year=(mo - 1) // 12, month=(mo - 1) % 12 + 1, day=1)
        nu = closed_n(P.axis, m0)
        W, c, n_eq, summ = GR.graph(var, live[:nu], excl)
        days = int((P.pres[:nu][:, lm]).any(axis=1).sum())
        fitted = days >= MIN_DAYS
        th = fit_theta(P, base, nu, W, c, clip, lm, fm, var, th) if fitted else TH0.copy()
        u, s, x, _ = CO.run(P.rv, base, P.pres, th, W, c, clip, lm, fm)
        sel = (P.mon == mo) & P.pres
        U[sel], S[sel], X[sel], live[sel] = u[sel], s[sel], x[sel], x[sel]
        rows.append(dict(month=m0.strftime("%Y-%m"), n_closed_dates=nu, n_closed_days=days, fitted=fitted, rho=th[0], alpha=th[1], beta=th[2], n_equal_rows=n_eq, w_summary=summ))
    return dict(U=U, S=S, X=X, params=pd.DataFrame(rows))
