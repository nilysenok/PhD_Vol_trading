"""V4-C, лист C-b: численные оценщики гейтов C5, C6, C8, C9, C10 (логит-параметры, L-BFGS-B, аналитический градиент). Цель — средний QLIKE объединённого прогноза (rv/F + ln F) + штрафы C-a:
(c/2)·Σ(отклонения по бумагам)² с c = κ·s², (l/2)·(панельные параметры)² с l = λ·s², κ = λ = 250 пар-эквивалентов, s² = mean ln²(M/H) на закрытых парах (v4_c_fit, v4_b_fit). Гиперпараметры фиксированы.
Линейная часть (C5, C6, C10) использует целевую функцию C-a `v4_c_fit._obj` как есть: без ограничений и без η результат совпадает с C1 до погрешности решателя."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit

from models_v2.s11 import v4_b_fit as FT, v4_c_fit as CF

BOUND, GTOL = CF.BOUND, CF.GTOL
DEG, KNOT_Q = 3, (1.0 / 3.0, 2.0 / 3.0)       # кубический B-сплайн, 2 внутренних узла по квантилям ℓ: без свободного члена 3 + 2 = 5 степеней свободы


def softmax(eta: np.ndarray) -> np.ndarray:
    z = eta - eta.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def lbfgs(fun, x0, lo, hi) -> tuple:
    """-> (результат, max|∇| с проекцией на ящик): градиент на активной границе не считается нарушением сходимости."""
    r = minimize(fun, np.clip(x0, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 20})
    g = r.jac
    pg = np.where(((r.x <= lo + 1e-12) & (g > 0)) | ((r.x >= hi - 1e-12) & (g < 0)), 0.0, g)
    return r, float(np.abs(pg).max())


def scales(ell, kappa=None, lam=None) -> tuple:
    kappa, lam = FT.KAPPA if kappa is None else kappa, FT.LAM if lam is None else lam
    s2 = float(np.mean(ell ** 2))
    return kappa * s2, lam * s2


def spline_knots(ell: np.ndarray) -> dict:
    return {"lo": float(ell.min()), "hi": float(ell.max()), "inner": np.quantile(ell, KNOT_Q)}


def spline_basis(ell: np.ndarray, kn: dict) -> np.ndarray:
    """Кубический B-сплайн по ℓ (значения вне окна обучения прижаты к его границам), без первого столбца (константа входит в α) -> (N, 5)."""
    t = np.r_[[kn["lo"]] * (DEG + 1), kn["inner"], [kn["hi"]] * (DEG + 1)]
    return BSpline.design_matrix(np.clip(ell, kn["lo"], kn["hi"]), t, DEG).toarray()[:, 1:]


def _obj_eta(rv, lH, ell, Z, tid, n, c, l):
    """C10: η_i = β + H_i; штраф на H_i с тем же c = κ·s², что на δ_i. x = [α, δ (n), β (k), H (n·k)]."""
    N, k = len(rv), Z.shape[1]

    def f(x):
        a, dl, b, E = x[0], x[1:n + 1], x[n + 1:n + 1 + k], x[n + 1 + k:].reshape(n, k)
        eta = a + dl[tid] + CF.mv(Z, b) + (Z * E[tid]).sum(1)
        w = expit(eta)
        lF = lH + w * ell
        e = rv * np.exp(-lF)
        ge = (1.0 - e) * ell * w * (1.0 - w)
        val = (np.sum(e + lF) + 0.5 * c * (float(dl @ dl) + float((E * E).sum())) + 0.5 * l * (a * a + float(b @ b))) / N
        g = np.empty_like(x)
        g[0] = (ge.sum() + l * a) / N
        g[1:n + 1] = (np.bincount(tid, ge, minlength=n) + c * dl) / N
        g[n + 1:n + 1 + k] = (np.einsum("ij,i->j", Z, ge) + l * b) / N
        GE = np.stack([np.bincount(tid, Z[:, j] * ge, minlength=n) for j in range(k)], 1)
        g[n + 1 + k:] = ((GE + c * E) / N).ravel()
        return val, g
    return f


def fit_lin(rv, lH, ell, Z, tid, n, x0=None, nonneg=(), eta=False, eta_zero=False, kappa=None, lam=None) -> dict:
    """w = σ(α + δ_i + Zβ [+ Z·H_i]). nonneg — номера столбцов Z с β ≥ 0 (C6: γ); eta — наклоны по бумагам (C10); eta_zero — H_i заморожены в 0 (тождество C10 → C1).
    -> {alpha, delta (n), beta (k), eta (n,k)|None, ok, grad, nit}."""
    c, l = scales(ell, kappa, lam)
    k = Z.shape[1]
    m = n + 1 + k
    free_eta = eta and not eta_zero             # eta_zero: H_i = 0 тождественно -> задача C-a (C1) без наклонов по бумагам, результат совпадает с C1 побитно
    ne = n * k if free_eta else 0
    x0 = np.zeros(m) if x0 is None or len(x0) != m else np.asarray(x0, float)
    x0 = np.r_[x0, np.zeros(ne)]
    lo, hi = np.full(m + ne, -BOUND), np.full(m + ne, BOUND)
    for j in nonneg:
        lo[n + 1 + j] = 0.0
    f = _obj_eta(rv, lH, ell, Z, tid, n, c, l) if free_eta else CF._obj(rv, lH, ell, Z, tid, n, c, l)
    r, g = lbfgs(f, x0, lo, hi)
    H = r.x[m:].reshape(n, k).copy() if free_eta else (np.zeros((n, k)) if eta else None)
    return {"alpha": float(r.x[0]), "delta": r.x[1:n + 1].copy(), "beta": r.x[n + 1:m].copy(), "eta": H, "ok": bool(g < GTOL), "grad": g, "nit": int(r.nit)}


def fit_regime(rv, lH, ell, z, tid, n, b_zero=False, kappa=None, lam=None) -> dict:
    """C8: w = π·w_calm_i + (1 − π)·w_crisis_i, π = σ(a + b·z), w_·_i = σ(a_· + d_·i). x = [a, b, a_calm, a_crisis, d_calm (n), d_crisis (n)]; старт a = 0, b = −1 (рост рыночного уровня снижает π) —
    симметрия «calm ↔ crisis» при равных стартах даёт седловую точку; b_zero — b заморожен в 0 (постоянный вес)."""
    c, l = scales(ell, kappa, lam)
    N = len(rv)

    def f(x):
        a, b, ac, ar, dc, dr = x[0], x[1], x[2], x[3], x[4:4 + n], x[4 + n:]
        p, wc, wr = expit(a + b * z), expit(ac + dc[tid]), expit(ar + dr[tid])
        w = p * wc + (1.0 - p) * wr
        lF = lH + w * ell
        e = rv * np.exp(-lF)
        gw = (1.0 - e) * ell
        gp = gw * (wc - wr) * p * (1.0 - p)
        gc, gr = gw * p * wc * (1.0 - wc), gw * (1.0 - p) * wr * (1.0 - wr)
        val = (np.sum(e + lF) + 0.5 * c * (float(dc @ dc) + float(dr @ dr)) + 0.5 * l * (a * a + b * b + ac * ac + ar * ar)) / N
        g = np.concatenate([[gp.sum() + l * a, (gp * z).sum() + l * b, gc.sum() + l * ac, gr.sum() + l * ar], np.bincount(tid, gc, minlength=n) + c * dc, np.bincount(tid, gr, minlength=n) + c * dr]) / N
        return val, g
    x0 = np.zeros(4 + 2 * n)
    x0[1] = 0.0 if b_zero else -1.0
    lo, hi = np.full(len(x0), -BOUND), np.full(len(x0), BOUND)
    if b_zero:
        lo[1], hi[1] = 0.0, 0.0
    r, g = lbfgs(f, x0, lo, hi)
    x = r.x
    return {"a": float(x[0]), "b": float(x[1]), "a_calm": float(x[2]), "a_crisis": float(x[3]), "d_calm": x[4:4 + n].copy(), "d_crisis": x[4 + n:].copy(), "ok": bool(g < GTOL), "grad": g, "nit": int(r.nit)}


def fit_three(rv, lK, Z, tid, n, ell, kappa=None, lam=None) -> dict:
    """C9: вес эксперта k = softmax_k(α_k + δ_ik + Z·β_k), F = exp(Σ w_k·lK_k), lK = (ln H, ln K2, ln K3) (N, 3), Z (N, 4). Избыточность softmax снимает гребень. x = [α (3), δ (n·3), β (4·3)]."""
    c, l = scales(ell, kappa, lam)
    N, k = len(rv), Z.shape[1]

    def f(x):
        a, dl, B = x[:3], x[3:3 + 3 * n].reshape(n, 3), x[3 + 3 * n:].reshape(k, 3)
        w = softmax(a + dl[tid] + np.einsum("ij,jk->ik", Z, B))
        lF = (w * lK).sum(1)
        e = rv * np.exp(-lF)
        gk = (1.0 - e)[:, None] * w * (lK - lF[:, None])
        val = (np.sum(e + lF) + 0.5 * c * float((dl * dl).sum()) + 0.5 * l * (float(a @ a) + float((B * B).sum()))) / N
        GD = np.stack([np.bincount(tid, gk[:, j], minlength=n) for j in range(3)], 1)
        g = np.concatenate([gk.sum(0) + l * a, (GD + c * dl).ravel(), (np.einsum("ij,ik->jk", Z, gk) + l * B).ravel()]) / N
        return val, g
    x0 = np.zeros(3 + 3 * n + 3 * k)
    r, g = lbfgs(f, x0, np.full(len(x0), -BOUND), np.full(len(x0), BOUND))
    x = r.x
    return {"alpha": x[:3].copy(), "delta": x[3:3 + 3 * n].reshape(n, 3).copy(), "beta": x[3 + 3 * n:].reshape(k, 3).copy(), "ok": bool(g < GTOL), "grad": g, "nit": int(r.nit)}
