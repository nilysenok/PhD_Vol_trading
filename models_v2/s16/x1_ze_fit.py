"""Лист ZE, обучение (план ZE, разд. 3.2–3.3). g = b + a·s + c·G на строках e_d = 1; потеря QLIKE F = X1·e^g (как в `x1_z_fit.lin_loss_grad`: l = u·e^{−g} + g − ln u − 1, u = RV(t⁺)/X1; при c = 0 значение и градиент по (b, a) совпадают побитно);
цель = mean l + λ·(b² + a² + c²); L-BFGS-B (scipy), ≤ 500 итераций, старт b = a = c = 0; |b|, |a| ≤ 5, c ∈ [0, 1] (`C_BOUNDS`; мутация: без границы). Выбор λ, V_e, ≥ 30 пар — `x1_z_fit` (импорт). Чистые функции."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2.s16 import x1_z_fit as FT

C_BOUNDS = (0.0, 1.0)


def loss_grad(th, u, s, G, lam):
    """th = (b, a, c); g = b + a·s + c·G."""
    g = th[0] + th[1] * s + th[2] * G
    ug = u * np.exp(-g)
    dg = 1.0 - ug
    pen = lam * float(th[0] * th[0] + th[1] * th[1] + th[2] * th[2])
    return float(np.mean(ug + g - np.log(u) - 1.0)) + pen, np.array([np.mean(dg) + 2 * lam * th[0], np.mean(dg * s) + 2 * lam * th[1], np.mean(dg * G) + 2 * lam * th[2]])


def fit_ze(u, s, G, lam, c_bounds=None) -> np.ndarray:
    """-> (b, a, c)."""
    bnd = [(-FT.BOUND_BA, FT.BOUND_BA)] * 2 + [tuple(C_BOUNDS if c_bounds is None else c_bounds)]
    return np.asarray(minimize(loss_grad, np.zeros(3), args=(u, s, G, lam), jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=FT.MAXITER)).x, float)


def g_of(th, e, s, G) -> np.ndarray:
    """g = e_d·(b + a·s + c·G); вне e_d = 1 — 0."""
    return np.asarray(e, float) * (th[0] + th[1] * s + th[2] * G)
