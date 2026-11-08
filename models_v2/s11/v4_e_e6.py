"""V4-E, E6: совместная адаптивная (b, w): F = H·exp(b_i + w_i·ℓ), ℓ = ln(M/H), M = √(K2·K3). Раз в месяц (первый торговый день d0) по парам, закрытым к d0, минимизируется дисконтированный QLIKE
Σ ω·(RV/F + ln F), ω = 0,5^(возраст/250) (возраст — число более поздних закрытых пар той же бумаги), с иерархией θ_i = θ_panel + δ_i (штраф ½c·Σδ², c_w = κ·mean ℓ², c_b = κ; κ = 250) и приором
(w₀ = 0,5; b₀ = 0; сила 250); w ∈ [0, 1], b ∈ [−0,5; 0,5]. До MIN_PAIRS закрытых пар панели — приор. Исключённая бумага (LOTO) и бумага без закрытых пар получают панельные параметры."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2.s11 import v4_e_calib as CAL

HL, W0, B0, KAPPA, LAM = 250.0, 0.5, 0.0, 250.0, 250.0


def _obj(rv, H, ell, tid, om, n, c_w, c_b):
    def f(x):
        wp, w, bp, b = x[0], x[1:n + 1], x[n + 1], x[n + 2:]
        lF = np.log(H) + b[tid] + w[tid] * ell
        e = rv / np.exp(lF)
        dw, db = w - wp, b - bp
        val = (om * (e + lF)).sum() + 0.5 * c_w * (dw @ dw) + 0.5 * c_b * (db @ db) + 0.5 * LAM * c_w * (wp - W0) ** 2 / KAPPA + 0.5 * LAM * (bp - B0) ** 2
        g1 = om * (1 - e)
        g = np.zeros_like(x)
        g[1:n + 1] = np.bincount(tid, g1 * ell, minlength=n) + c_w * dw
        g[0] = -c_w * dw.sum() + LAM * c_w * (wp - W0) / KAPPA
        g[n + 2:] = np.bincount(tid, g1, minlength=n) + c_b * db
        g[n + 1] = -c_b * db.sum() + LAM * (bp - B0)
        return val / om.sum(), g / om.sum()
    return f


def fit(rv, H, M, tid, n, kappa=KAPPA):
    """Закрытые пары (tid — 0..n−1, внутри бумаги по возрастанию даты). -> {w_p, w (n), b_p, b (n), ok}."""
    ell = np.log(M / H)
    om = np.empty(len(rv))
    for i in range(n):
        j = np.where(tid == i)[0]
        om[j] = CAL.lam(HL) ** np.arange(len(j) - 1, -1, -1)
    c_w, c_b = kappa * float(np.mean(ell ** 2)), kappa
    lo = np.r_[0.0, np.zeros(n), -CAL.B_CLIP, np.full(n, -CAL.B_CLIP)]
    hi = np.r_[1.0, np.ones(n), CAL.B_CLIP, np.full(n, CAL.B_CLIP)]
    x0 = np.r_[W0, np.full(n, W0), B0, np.full(n, B0)]
    r = minimize(_obj(rv, H, ell, tid, om, n, c_w, c_b), x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
    g = r.jac
    pg = np.where(((r.x <= lo + 1e-12) & (g > 0)) | ((r.x >= hi - 1e-12) & (g < 0)), 0.0, g)
    return {"w_p": float(r.x[0]), "w": r.x[1:n + 1], "b_p": float(r.x[n + 1]), "b": r.x[n + 2:], "ok": bool(np.abs(pg).max() < 1e-6)}


def monthly_base(rv, H, K2, K3, tix, date, t_plus, nT, exclude=None):
    """-> (ln-база lnF6 = ln H + b + w·ℓ по строкам, w по строкам, b по строкам, список записей месяцев). Строки отсортированы по (date, ticker)."""
    avail = np.asarray(CAL.closed_date(np.asarray(t_plus), np.asarray(date)))
    M = np.sqrt(K2 * K3)
    mon = date.astype("datetime64[M]")
    w, b = np.full(len(rv), W0), np.full(len(rv), B0)
    recs = []
    for m in np.unique(mon):
        sel = mon == m
        d0 = date[sel].min()
        cl = avail <= d0
        if exclude is not None:
            cl &= tix != exclude
        par = {"w_p": W0, "b_p": B0, "ok": True, "w": np.full(nT, W0), "b": np.full(nT, B0)}
        if cl.sum() >= CAL.MIN_PAIRS:
            present, tid = np.unique(tix[cl], return_inverse=True)
            f = fit(rv[cl], H[cl], M[cl], tid, len(present))
            par.update(w_p=f["w_p"], b_p=f["b_p"], ok=f["ok"], w=np.full(nT, f["w_p"]), b=np.full(nT, f["b_p"]))
            par["w"][present], par["b"][present] = f["w"], f["b"]
        w[sel], b[sel] = par["w"][tix[sel]], par["b"][tix[sel]]
        recs.append({"month": str(m), "w_p": par["w_p"], "b_p": par["b_p"], "converged": par["ok"], "n_pairs": int(cl.sum())})
    return np.log(H) + b + w * np.log(M / H), w, b, recs
