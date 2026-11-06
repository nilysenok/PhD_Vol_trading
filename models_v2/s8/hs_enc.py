"""Ж-212, этап A, вопрос 8: условная ценность регрессоров сверх базового прогноза, QLIKE-согласованно.
Модель σ̂² = base·exp(X·γ). Тест H0: γ = 0 — M-оценка QLIKE, сэндвич Ньюи–Уэста (лаг 5, Бартлетт): V = A⁻¹ B A⁻¹ / n,
A = mean(r·XXᵀ), B = NW-ковариация скоров s_t = X_t(1 − r_t), r = RV/f. Положительный γ при X = ln HAR − ln base — HAR добавляет информацию.
OOS «по прошлому»: γ подгоняется только по закрытым прогнозам (target_date <= d), расширяющееся окно, переподгонка каждые `step` строк."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from models_v2 import metrics
from models_v2.s3 import dm as dmod
from models_v2.s8 import hs_pit, hs_qfit

LAG = dmod.LAG


def nw_cov(S: np.ndarray, lag: int = LAG) -> np.ndarray:
    """Долгосрочная ковариация рядов скоров (n×k), Бартлетт."""
    S = S - S.mean(axis=0)
    n = len(S)
    G = np.einsum("ij,ik->jk", S, S) / n
    for l in range(1, lag + 1):
        Gl = np.einsum("ij,ik->jk", S[l:], S[:-l]) / n
        G = G + (1 - l / (lag + 1)) * (Gl + Gl.T)
    return G


def _mm(*ms) -> np.ndarray:
    out = ms[0]
    for m in ms[1:]:
        out = np.einsum("ij,jk->ik", out, m)
    return out


def wald(rv, base, X, ridge: float = 0.0) -> dict:
    """QLIKE-оценка γ и HAC-тест. -> gamma, se, t, p (по компонентам), wald, p_joint, gain_in_sample (потолок), n."""
    rv, base, X = np.asarray(rv, float), np.asarray(base, float), np.asarray(X, float)
    X = X.reshape(len(rv), -1)
    off = np.log(base)
    gam = hs_qfit.fit(X, rv, off, ridge)
    r = rv * np.exp(-(off + hs_qfit.lin(X, gam)))
    A = np.einsum("ij,i,ik->jk", X, r, X) / len(rv)
    Ai = np.linalg.inv(A)
    V = _mm(Ai, nw_cov(X * (1.0 - r)[:, None]), Ai) / len(rv)
    se = np.sqrt(np.diag(V))
    t = gam / se
    w = float(np.sum(gam * np.linalg.solve(V, gam)))
    gain = hs_qfit.mean_loss(np.zeros(X.shape[1]), X, rv, off) - hs_qfit.mean_loss(gam, X, rv, off)
    return {"gamma": gam, "se": se, "t": t, "p": 2 * sps.norm.sf(np.abs(t)), "wald": w, "p_joint": float(sps.chi2.sf(w, X.shape[1])),
            "gain_in_sample": float(gain), "n": len(rv)}


def oos_expanding(rv, base, X, d, td, step: int = 5, min_closed: int = 252, ridge: float = 0.0, extra: int = 0) -> tuple:
    """σ̂² на строке k: base_k·exp(X_k·γ), γ подогнана на строках j < m[k0] (закрытые на d_{k0}), k0 — последняя переподгонка <= k.
    До min_closed закрытых γ = 0 (σ̂² = base). -> (σ̂² (n,), γ по строкам (n×k))."""
    rv, base, X = np.asarray(rv, float), np.asarray(base, float), np.asarray(X, float).reshape(len(rv), -1)
    n, k = X.shape
    m = hs_pit.closed_count(d, td, extra)
    off = np.log(base)
    G, gam = np.zeros((n, k)), np.zeros(k)
    for k0 in range(0, n, step):
        if m[k0] >= min_closed:
            mm = int(m[k0])
            gam = hs_qfit.fit(X[:mm], rv[:mm], off[:mm], ridge, gam0=gam)
        G[k0:k0 + step] = gam
    return np.exp(off + hs_qfit.lin(X, G)), G


def dm_vs(rv, s2_base, s2_new, mask=None) -> dict:
    """DM (Ньюи–Уэст лаг 5): Δ = mean(QLIKE(new) − QLIKE(base)), минус — new лучше."""
    rv, a, b = np.asarray(rv, float), np.asarray(s2_base, float), np.asarray(s2_new, float)
    if mask is not None:
        rv, a, b = rv[mask], a[mask], b[mask]
    r = dmod.dm(rv, a, b)
    return {"delta": -r["mean_d"], "dm": r["stat"], "p": r["p"], "n": r["n"]}


def mz_hac(y, x, lag: int = LAG) -> dict:
    """y = a + b·x + e, OLS, сэндвич Ньюи–Уэста. -> a, b, t_b, p_b, r2."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    A = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    u = y - hs_qfit.lin(A, beta)
    Q = np.linalg.inv(np.einsum("ij,ik->jk", A, A) / len(y))
    V = _mm(Q, nw_cov(A * u[:, None], lag), Q) / len(y)
    t = beta[1] / np.sqrt(V[1, 1])
    return {"a": float(beta[0]), "b": float(beta[1]), "t_b": float(t), "p_b": float(2 * sps.norm.sf(abs(t))),
            "r2": float(1 - np.sum(u * u) / np.sum((y - y.mean()) ** 2))}


def qlike_mean(rv, s2) -> float:
    return float(metrics.qlike(rv, s2).mean())
