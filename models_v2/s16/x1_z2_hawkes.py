"""Z2, ступень 1: правдоподобие баров (план Z2 раздел 2.2, 3; цели RV нет). λ_{d,k} = μ·f_{D,k} + H_{d,k}; P(N_{d,k} = 1 | прошлое) = 1 − exp(−δλ), δ = 10 мин; событие бара k возбуждает бары > k.
H_{d,0} = ρ^{1+Δ_{d−1}}·H^{cl}_{d−1}; H_{d,k+1} = e^{−βδ}·H_{d,k} + α·N_{d,k}·e^{η·m_{d,k}}; слот без бара: N = 0, в правдоподобии нет, затухание идёт. α = n·(1 − e^{−βδ})/(δ·M_η), M_η — среднее e^{ηm} по событиям окна.
Параметры θ = (μ, n, β, η, ρ); границы: n ∈ [0; 0,9], β ∈ [ln 2/530; ln 2/10], η ∈ [0; 2], ρ ∈ [0; 1]; L-BFGS-B, аналитический градиент (прямые чувствительности), ≤ 500 итераций, без seed. Старт: n = 0,3, полураспад 60 мин, η = 0,5, ρ = 0,5."""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import chi2

from models_v2.s16.x1_z2_ev import DELTA, NS

N_MAX, ETA_MAX, MAXITER, WIN, WARM, PROF = 0.9, 2.0, 500, 500, 20, 250
BETA_LO, BETA_HI, N0, B0, ETA0, RHO0 = math.log(2) / 530.0, math.log(2) / 10.0, 0.3, math.log(2) / 60.0, 0.5, 0.5


def win_end(p: int) -> int:
    """Конец (исключительно) окна ступени 1 для переоценки на позиции p склада: СТРОГО до p (мутация: p + 1 — сессия D в окне)."""
    return p


def _kernel(lnmu, alpha, beta, eta, rho, N, m, pres, f, inlik, dg, delta, g):
    mu, E, H, ll = math.exp(lnmu), math.exp(-beta * delta), 0.0, 0.0
    s0 = s1 = s2 = s3 = 0.0                       # dH/dα, dH/dβ, dH/dη, dH/dρ
    for i in range(5):
        g[i] = 0.0
    for j in range(N.shape[0]):
        if j > 0:
            pw, dpw = rho ** (1.0 + dg[j]), (1.0 + dg[j]) * rho ** dg[j]
            s3 = dpw * H + pw * s3
            s0, s1, s2, H = s0 * pw, s1 * pw, s2 * pw, H * pw
        for k in range(N.shape[1]):
            ev = 1.0 if (pres[j, k] and N[j, k] > 0.5) else 0.0
            if pres[j, k] and inlik[j]:
                lam = mu * f[k] + H                  # ORDER: λ бара k — ДО добавления события бара k
                if ev > 0.0:
                    x = delta * max(lam, 1e-12)
                    em = -math.expm1(-x)
                    ll += math.log(em)
                    dl = delta * math.exp(-x) / em
                else:
                    ll -= delta * lam
                    dl = -delta
                g[0] += dl * mu * f[k]
                g[1] += dl * s0
                g[2] += dl * s1
                g[3] += dl * s2
                g[4] += dl * s3
            w = math.exp(eta * m[j, k]) if ev > 0.0 else 0.0
            s0 = E * s0 + ev * w
            s1 = E * s1 - delta * E * H
            s2 = E * s2 + alpha * ev * w * m[j, k]
            s3 = E * s3
            H = E * H + alpha * ev * w
    return ll


def _hcl(alpha, beta, eta, rho, N, m, pres, dg, delta, out):
    E, H = math.exp(-beta * delta), 0.0
    for j in range(N.shape[0]):
        if j > 0:
            H *= rho ** (1.0 + dg[j])
        for k in range(N.shape[1]):
            ev = 1.0 if (pres[j, k] and N[j, k] > 0.5) else 0.0
            H = E * H + alpha * ev * (math.exp(eta * m[j, k]) if ev > 0.0 else 0.0)
        out[j] = H


kernel, hcl_kernel = njit(_kernel), njit(_hcl)


def make_ctx(days, r, q, N, m) -> dict:
    """Контекст склада: days, pres, N, m (события; для Hc — переставленные), r2, dg (Δ между j − 1 и j), dnext (Δ между j и j + 1), base_inlik (дни с порогом)."""
    pres = np.isfinite(r)
    gaps = np.diff(days.values).astype("timedelta64[D]").astype(int) - 1
    return dict(days=days, pres=pres, N=N.astype(float), m=m, r2=np.where(pres, r, 0.0) ** 2, dg=np.r_[0.0, gaps], dnext=np.r_[gaps, 0].astype(float), base_inlik=np.isfinite(q))


def profile(ctx, pos_D: int) -> np.ndarray:
    """f_{D,k}: средняя доля r²_k/RV по сессиям < D (до 250) с присутствующим слотом k, нормировка на среднее по слотам; слот без наблюдений — 0."""
    a = max(0, pos_D - PROF)
    r2, pr = ctx["r2"][a:pos_D], ctx["pres"][a:pos_D]
    rv = r2.sum(1)
    ok = rv > 0
    sh = np.where(pr[ok], r2[ok] / rv[ok][:, None], 0.0)
    cnt = pr[ok].sum(0)
    pbar = np.where(cnt > 0, sh.sum(0) / np.maximum(cnt, 1), 0.0)
    return pbar / pbar.mean()


def rows(ctx, a: int, b: int, warm: int = WARM) -> dict:
    """Окно: сессии a − warm … b − 1; первые `warm` строк — прогрев рекурсии (в правдоподобие не входят)."""
    s = slice(a - warm, b)
    inl = ctx["base_inlik"][s].copy()
    inl[:warm] = False
    ev = ctx["N"][s] > 0.5
    mev = ctx["m"][s][ev & ctx["pres"][s] & inl[:, None]]
    return dict(N=ctx["N"][s], m=ctx["m"][s], pres=ctx["pres"][s], inlik=inl, dg=ctx["dg"][s], mev=mev, nslots=int((ctx["pres"][s] & inl[:, None]).sum()))


def window(ctx, pos_D: int) -> dict:
    e = win_end(pos_D)
    return rows(ctx, e - WIN, e)


def alpha_of(n, beta, eta, mev) -> tuple:
    M, dM = (float(np.exp(eta * mev).mean()), float((mev * np.exp(eta * mev)).mean())) if len(mev) else (1.0, 0.0)
    return n * (1.0 - math.exp(-beta * DELTA)) / (DELTA * M), M, dM


def nll(x, W, f):
    """x = (ln μ, n, β, η, ρ) -> (−ll/слотов, градиент по x)."""
    n, beta, eta = x[1], x[2], x[3]
    al, M, dM = alpha_of(n, beta, eta, W["mev"])
    E, g = math.exp(-beta * DELTA), np.zeros(5)
    ll = kernel(x[0], al, beta, eta, x[4], W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"], DELTA, g)
    gx = np.array([g[0], g[1] * (1.0 - E) / (DELTA * M), g[2] + g[1] * n * E / M, g[3] - g[1] * al * dM / M, g[4]])
    return -ll / W["nslots"], -gx / W["nslots"]


def nll0(lnmu, W, f):
    g = np.zeros(5)
    ll = kernel(float(lnmu[0]), 0.0, B0, ETA0, RHO0, W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"], DELTA, g)
    return -ll / W["nslots"], np.array([-g[0] / W["nslots"]])


def start(W) -> np.ndarray:
    frac = max(float((W["N"] > 0.5)[W["pres"] & W["inlik"][:, None]].sum()), 1.0) / W["nslots"]
    return np.array([math.log(frac * (1.0 - N0) / DELTA), N0, B0, ETA0, RHO0])


def bounds() -> list:
    return [(-25.0, 2.0), (0.0, N_MAX), (BETA_LO, BETA_HI), (0.0, ETA_MAX), (0.0, 1.0)]


def fit(W, f) -> dict:
    """Ступень 1: MLE θ и пуассоновская база (α = 0). -> μ, n, β, η, ρ, α, ll, ll0, nev, nslots."""
    r = minimize(nll, start(W), args=(W, f), jac=True, method="L-BFGS-B", bounds=bounds(), options=dict(maxiter=MAXITER, ftol=1e-12, gtol=1e-9))
    r0 = minimize(nll0, [start(W)[0]], args=(W, f), jac=True, method="L-BFGS-B", bounds=[(-25.0, 2.0)], options=dict(maxiter=MAXITER, ftol=1e-12, gtol=1e-9))
    ns = W["nslots"]
    al, M, _ = alpha_of(r.x[1], r.x[2], r.x[3], W["mev"])
    return dict(mu=float(np.exp(r.x[0])), n=float(r.x[1]), beta=float(r.x[2]), eta=float(r.x[3]), rho=float(r.x[4]), alpha=float(al), M=float(M), ll=float(-r.fun * ns), ll0=float(-r0.fun * ns),
                nev=int((W["N"] > 0.5)[W["pres"] & W["inlik"][:, None]].sum()), nslots=int(ns), nit=int(r.nit))


def lr_test(ll1: float, ll0: float) -> dict:
    """LR = 2(ll1 − ll0) ≥ 0. p_mix — смесь 0,5χ²₀ + 0,5χ²₁ (n на границе; план); p_strict — χ²₄ (β, η, ρ не определены при n = 0: строже). Условие L3 читает p_strict (неясность — в пользу строгого, отступление)."""
    lr = max(2.0 * (ll1 - ll0), 0.0)
    return dict(lr=lr, p_mix=float(0.5 * chi2.sf(lr, 1)) if lr > 0 else 1.0, p_strict=float(chi2.sf(lr, 4)))


def hcl_all(par: dict, ctx) -> np.ndarray:
    """H^{cl}_j для ВСЕХ сессий склада при θ (рекурсия от начала склада, H = 0)."""
    out = np.zeros(len(ctx["N"]))
    hcl_kernel(par["alpha"], par["beta"], par["eta"], par["rho"], ctx["N"], ctx["m"], ctx["pres"], ctx["dg"], DELTA, out)
    return out


def xi_row(p: int) -> int:
    """Строка, чьё закрытие несёт ξ дня d на позиции p: СТРОГО день d (мутация: p + 1 — бары дня t⁺)."""
    return p


def xi_all(par: dict, ctx, f: np.ndarray, dnext) -> np.ndarray:
    """ξ_j = ρ^{1+Δ_j}·H^{cl}_j·Σ_k e^{−βδk} / [(1 − n)·μ·Σ_k f_k] ≥ 0 для всех сессий склада; Δ_j — `dnext`."""
    h = hcl_all(par, ctx)
    pos = np.array([xi_row(p) for p in range(len(h))])
    h = np.where(pos < len(h), h[np.minimum(pos, len(h) - 1)], 0.0)
    es = float(np.exp(-par["beta"] * DELTA * np.arange(NS)).sum())
    return par["rho"] ** (1.0 + np.asarray(dnext)) * h * es / ((1.0 - par["n"]) * par["mu"] * float(f.sum()))
