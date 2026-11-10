"""Z2, помесячная переоценка H1 / Hc (план Z2 раздел 5; считается ТОЛЬКО при истинном условии L1–L3). В первый торговый день D месяца с 2018-07-02: ступень 1 — θ_D по 500 сессиям < D (+ 20 прогрева), f_D по 250 сессиям < D;
ξ_j ВСЕХ строк (и пула, и прогнозов месяца) — с ОДНИМ θ_D (согласованные координаты, класс № 146); ступень 2 — a по ВСЕМ закрытым парам (t⁺ ≤ D): QLIKE + λ·a², λ ∈ {0; 0,001; 0,01; 0,1} по последним V закрытым парам (V = 126 при n ≥ 252, иначе ⌊n/2⌋), ничья — больший λ.
Включение — ≥ 30 закрытых пар с e_d = 1, до этого F = X1 побитно, active = 0. g = a·ln(1 + ξ_d); a = 0 ⇒ F = X1 побитно."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models_v2.s16 import x1_z2_hawkes as HK, x1_z_fit as FT, x1_z_walk as W

OUT_COLS = ["d", "t_plus", "rv", "X1", "F", "g", "xi", "active", "lam", "a", "mu", "n", "beta", "eta", "rho", "nev", "n_ext"]
BOUND_A = 5.0


def loss_grad(th, u, x, lam):
    """g = a·x, x = ln(1 + ξ); QLIKE u·e^{−g} + g − ln u − 1; dl/dg = 1 − u·e^{−g}."""
    g = th[0] * x
    ug = u * np.exp(-g)
    return float(np.mean(ug + g - np.log(u) - 1.0)) + lam * float(th[0] ** 2), np.array([float(np.mean((1.0 - ug) * x)) + 2.0 * lam * th[0]])


def fit_a(u, x, lam) -> np.ndarray:
    return np.asarray(minimize(loss_grad, np.zeros(1), args=(u, x, lam), jac=True, method="L-BFGS-B", bounds=[(-BOUND_A, BOUND_A)], options=dict(maxiter=FT.MAXITER)).x, float)


def refit(pool: np.ndarray, u, x) -> tuple:
    """-> (λ, a): λ по последним V закрытым парам пула, затем переобучение на всём пуле."""
    v = FT.n_valid(len(pool), FT.V_FULL)
    lam, _ = FT.select(pool[: len(pool) - v], FT.val_idx(pool, v), lambda i, l: fit_a(u[i], x[i], l), lambda th, i, o: FT.qlike_g(u[i], th[0] * x[i]))
    return lam, float(fit_a(u[pool], x[pool], lam)[0])


def pool_xi(par: dict, ctx: dict, f: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """ξ строк прогнозов на позициях склада `pos` при ОДНОМ θ_D (мутация № 146: θ месяца самой пары)."""
    return HK.xi_all(par, ctx, f, ctx["dnext"])[pos]


def forecasts(X1, g) -> np.ndarray:
    """F = X1·exp(g); g = 0 ⇒ F = X1 побитно."""
    return X1 * np.exp(g)


def stage1(ctx: dict, d) -> dict:
    """Ступень 1 на каждую переоценку D = d[a] (первый торговый день месяца с 2018-07-02): {a: (θ_D, f_D)}; только бары склада < D."""
    pos, fits = ctx["days"].get_indexer(pd.DatetimeIndex(d)), {}
    for a in (p for p in W.month_starts(d) if pd.DatetimeIndex(d)[p] >= W.ACTIVATION):
        f = HK.profile(ctx, int(pos[a]))
        fits[a] = (HK.fit(HK.window(ctx, int(pos[a])), f), f)
    return fits


def walk(x1: pd.DataFrame, ctx: dict, fits: dict, e: np.ndarray) -> pd.DataFrame:
    """x1: d, t_plus, rv, F (= X1); fits: {позиция строки x1 первого дня месяца: (θ_D, f_D)} (ступень 1); e — e_d по строкам x1. -> прогнозы (OUT_COLS)."""
    d, tp, X1, rv = pd.DatetimeIndex(x1["d"]), pd.DatetimeIndex(x1["t_plus"]), x1["F"].to_numpy(float), x1["rv"].to_numpy(float)
    n, u = len(x1), x1["rv"].to_numpy(float) / X1
    pos = ctx["days"].get_indexer(d)
    g, xi, act, par = np.zeros(n), np.full(n, np.nan), np.zeros(n, int), np.full((n, 8), np.nan)
    starts = [p for p in W.month_starts(d) if d[p] >= W.ACTIVATION] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        pool = W.closed_pool(tp, d, d[a])
        nx = FT.ext_pool(pool, e).size
        par[a:b, 7] = nx
        if nx < FT.MIN_EXT:
            continue
        th, f = fits[a]
        xi_all = pool_xi(th, ctx, f, pos)
        lam, ah = refit(pool, u, np.log1p(xi_all))
        g[a:b], xi[a:b], act[a:b] = ah * np.log1p(xi_all[a:b]), xi_all[a:b], 1
        par[a:b, :7] = [lam, ah, th["mu"], th["n"], th["beta"], th["eta"], th["rho"]]
    cols = dict(zip(("lam", "a", "mu", "n", "beta", "eta", "rho"), par[:, :7].T))
    nev = np.full(n, np.nan)
    for a, b in zip(starts[:-1], starts[1:]):
        if a in fits and act[a]:
            nev[a:b] = fits[a][0]["nev"]
    return pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, X1=X1, F=forecasts(X1, g), g=g, xi=xi, active=act, nev=nev, n_ext=par[:, 7], **cols))[OUT_COLS]
