# -*- coding: utf-8 -*-
"""Zh-190 stage 3 layers on top of G*: A-1 (per-ticker non-negative ridge in logs), A-2 (stress/calm logistic mixture in logs), A-3 (EWMA recalibration).
Frame `a`: rows sorted by (ticker, date). Annual cutoffs use rows whose target ends before the cut (PIT); 2016 has no history and falls back to G*."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import expit
import gl3_r6b_layers as L
import gl3_zh190_stage23_core as Q

ALPHA, LOGIT_REG, MIN_LOGIT = 1., .5e-4, 100
HALF_LIVES, WS = (20, 60, 120), (.5, .75)


def nnls_ridge(X, y, alpha=ALPHA):
    """min |y - b - Xw|^2 + alpha |w|^2, w >= 0, intercept b free (not penalised): NNLS on centred, augmented data."""
    xm, ym = X.mean(0), y.mean()
    A = np.vstack([X - xm, np.sqrt(alpha) * np.eye(X.shape[1])])
    w, _ = nnls(A, np.r_[y - ym, np.zeros(X.shape[1])])
    return w, float(ym - xm @ w)


def a1_forecast(a, lns, y, end, h, base_level, rec=None):
    """lns: (n, k) log forecasts [HAR_i, B* panel, B* ticker, G*]. Weights of the ticker = 0.5 own + 0.5 pool (coefficients and intercept);
    back-transform exp(s2/2), s2 = variance of the log residual of the ticker on the same past rows."""
    with np.errstate(all='ignore'):  # spurious BLAS (Accelerate) matmul warnings on finite data, as in 6b
        return _a1(a, lns, y, end, h, base_level, rec)


def _a1(a, lns, y, end, h, base_level, rec):
    out = np.asarray(base_level, float).copy()
    ly, tk, year = np.log(np.clip(y, 1e-12, None)), a.ticker.values, a.year.values
    for Y in Q.YEARS:
        cut = pd.Timestamp(f'{Y}-01-01')
        tr = Q.past_mask(end, cut) & np.isfinite(ly) & (y > 1e-12) & np.isfinite(lns).all(axis=1)
        wp, bp = nnls_ridge(lns[tr], ly[tr])
        if rec:
            rec(h, 'A1_pool', cut, int(tr.sum()), np.asarray(end)[tr].max())
        coef, res = {}, np.full(len(a), np.nan)
        for t in np.unique(tk):
            m = tr & (tk == t)
            wi, bi = nnls_ridge(lns[m], ly[m]) if m.sum() >= Q.MIN_OBS else (wp, bp)
            coef[t] = (Q.SHRINK * wi + (1 - Q.SHRINK) * wp, Q.SHRINK * bi + (1 - Q.SHRINK) * bp)
            res[m] = ly[m] - (coef[t][1] + lns[m] @ coef[t][0])
            if rec and m.sum() >= Q.MIN_OBS:
                rec(h, 'A1_ticker', cut, int(m.sum()), np.asarray(end)[m].max(), detail=dict(ticker=t))
        pool_s2 = float(np.var(res[tr]))
        for t in np.unique(tk):
            va, m = (year == Y) & (tk == t), tr & (tk == t)
            s2 = float(np.var(res[m])) if m.sum() >= Q.MIN_OBS else pool_s2
            out[va] = Q.level(coef[t][1] + lns[va] @ coef[t][0] + s2 / 2)
    return out


def fit_logit_ln(zz, y, e, f):
    """Weight of the stress expert g = sigma(zz b) in logs: ln p = ln e + g (ln f - ln e); QLIKE + 0.5e-4 sum(b[1:]^2), BFGS (as G2_logistic of 6b)."""
    d = np.log(f) - np.log(e)

    def fg(b):
        g = expit(zz @ b)
        r = y / np.exp(np.log(e) + g * d)
        return np.mean(r - np.log(r) - 1) + LOGIT_REG * np.sum(b[1:] ** 2), zz.T @ ((1 - r) * d * g * (1 - g)) / len(y) + np.r_[0., 2 * LOGIT_REG * b[1:]]
    with np.errstate(all='ignore'):
        return minimize(fg, np.zeros(zz.shape[1]), method='BFGS', jac=True, options={'maxiter': 200, 'gtol': 1e-8})


def z_features(b_full):
    """z_lnRVI and z_lnRV_w from the full frame (deviation from the past median / 2/3-quantile over closed dates, regime of day d: RVI_LEAD = 0 of 6b)."""
    x, m = L.rvi_pair(b_full)
    w, q = L.rvw_pair(b_full)
    lg = lambda v: np.log(np.clip(v, 1e-12, None))
    return np.column_stack([lg(x) - lg(m), lg(w) - lg(q)])


def a2_forecast(a, z, y, end, h, calm, stress, rec=None):
    """Stress weight sigma(a + b1 z1 + b2 z2) on day d, a, b fitted at each 1 January on the closed windows of the three past years (QLIKE)."""
    out = np.asarray(calm, float).copy()
    year, date = a.year.values, a.date.values
    for Y in Q.YEARS:
        cut = pd.Timestamp(f'{Y}-01-01')
        tr = (Q.past_mask(end, cut) & (date >= np.datetime64(cut - pd.DateOffset(years=3))) & np.isfinite(y) & (y > 1e-12) & np.isfinite(z).all(axis=1))
        if tr.sum() < MIN_LOGIT:
            continue
        mu, sd = z[tr].mean(0), z[tr].std(0)
        sd[sd < 1e-8] = 1.
        zz = np.column_stack([np.ones(len(a)), np.nan_to_num((z - mu) / sd)])
        r = fit_logit_ln(zz[tr], y[tr], calm[tr], stress[tr])
        va = year == Y
        with np.errstate(all='ignore'):
            g = expit(zz[va] @ r.x)
        out[va] = np.exp(np.log(calm[va]) + g * (np.log(stress[va]) - np.log(calm[va])))
        if rec:
            rec(h, 'A2_logit', cut, int(tr.sum()), np.asarray(end)[tr].max(), detail=dict(coef=r.x.tolist(), success=bool(r.success)))
    return out


def a3_points(a, base, y, h):
    """k_i,t = EWMA(RV/sigma^2) over closed windows (lag h), half-life {20, 60, 120}, w in {0.5, 0.75}, k in [0.7; 1.5] (layer14 of 6b)."""
    return {f'hl{hl}_w{int(w * 100)}': L.layer14(a, base, y, hl, w, lag=h) for hl in HALF_LIVES for w in WS}
