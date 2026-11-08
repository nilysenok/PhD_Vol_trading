"""V5 лист 2, SP2–SP4: общая регуляризованная поправка полос. F = K8·M, M = Σ_b π_b·exp(g_b) = 1 + Σ_b π_b·expm1(g_b) (тождество при Σπ_b = 1; при θ = 0 M = 1.0 ровно, F = K8 побитно), g_b = θ_bᵀ z_b.
θ ∈ R^{3×L} общий для панели; ежемесячно по закрытым парам (цель τ < первого числа месяца): минимум (Σ QLIKE + ½·κ·Σ s2·θ²)/N, κ = 250 (штраф V4 `v4_b_fit._obj`), s2_bl = средний квадрат ∂ln F/∂θ_bl в θ = 0 (= (π_b·z_bl)²), N — число закрытых пар.
Меньше 60 закрытых дней — θ = 0 (F = K8). Старт L-BFGS-B — θ = 0 (без подбора); аналитический градиент. LOTO: исключённая бумага не входит ни в потери, ни в межбумажный фактор SP3."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models_v2.s12 import v5_sp_feat as FE
from models_v2.s12.v5_gs_fit import closed_n

KAPPA = 250.0
MIN_DAYS = 60
GCLIP = 30.0


def mult(Z: np.ndarray, pi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """M = 1 + Σ_b π_b·expm1(θ_bᵀ z_b); Z (…, 3, L), π (…, 3), θ (3, L)."""
    g = np.clip(np.einsum("...bl,bl->...b", Z, theta), -GCLIP, GCLIP)
    return 1.0 + (pi * np.expm1(g)).sum(axis=-1)


def objective(th, Z, pi, rv, k8, s2):
    """-> (значение, градиент) — (Σ QLIKE + ½κ Σ s2 θ²)/N и его ∂θ."""
    th = th.reshape(Z.shape[1:])
    g = np.clip(np.einsum("rbl,bl->rb", Z, th), -GCLIP, GCLIP)
    eg = np.exp(g)
    mu = 1.0 + (pi * (eg - 1.0)).sum(axis=1)
    r = rv / (k8 * mu)
    n = len(rv)
    val = float((r - np.log(r) - 1.0).sum() + 0.5 * KAPPA * (s2 * th * th).sum()) / n
    w = ((1.0 - r) / mu)[:, None] * pi * eg                               # ∂QLIKE/∂ln F = 1 − r; ∂ln F/∂g_b = π_b e^{g_b}/M
    grad = (np.einsum("rb,rbl->bl", w, Z) + KAPPA * s2 * th) / n
    return val, grad.ravel()


def fit_theta(Z, pi, rv, k8) -> np.ndarray:
    s2 = np.maximum(((pi[:, :, None] * Z) ** 2).mean(axis=0), 1e-12)
    r = minimize(objective, np.zeros(Z.shape[1] * Z.shape[2]), args=(Z, pi, rv, k8, s2), jac=True, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-13, "gtol": 1e-9})
    return r.x.reshape(Z.shape[1:])


def run_config(P, F: FE.Feat, cfg: str, excl: int | None = None) -> dict:
    """Весь путь SP2–SP4. excl — индекс исключённой бумаги (LOTO). -> dict(F (N, m) с NaN вне панели, M (N, m), params DataFrame)."""
    N, m = P.rv.shape
    keep = np.ones(m, bool)
    if excl is not None:
        keep[excl] = False
    Z, pi = FE.panel_z(P, F, cfg, keep)
    nn, jj = np.nonzero(P.pres & keep[None, :])
    Zf, pif, rvf, k8f = Z[nn, jj], pi[nn, jj], P.rv[nn, jj], P.k8[nn, jj]
    L = Z.shape[-1]
    names = ["%s_%s" % (b, l) for b in FE.BANDS for l in FE.LAG_NAMES[cfg]]
    th, out_M, rows = np.zeros((3, L)), np.full((N, m), np.nan), []
    for mo in sorted(set(P.mon[P.pres].tolist())):
        m0 = pd.Timestamp(year=(mo - 1) // 12, month=(mo - 1) % 12 + 1, day=1)
        nu = closed_n(P.axis, m0)
        e = int(np.searchsorted(nn, nu))
        days = int(P.pres[:nu][:, keep].any(axis=1).sum())
        fitted = days >= MIN_DAYS and e > 0
        th = fit_theta(Zf[:e], pif[:e], rvf[:e], k8f[:e]) if fitted else np.zeros((3, L))
        sel = (P.mon == mo) & P.pres
        out_M[sel] = mult(Z[sel], pi[sel], th)
        rows.append(dict(month=m0.strftime("%Y-%m"), n_closed_dates=nu, n_closed_days=days, n_pairs=e, fitted=fitted, **{"th_" + n: float(v) for n, v in zip(names, th.ravel())}))
    return dict(F=P.k8 * out_M, M=out_M, params=pd.DataFrame(rows))
