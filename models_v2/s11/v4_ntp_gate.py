"""V4-NTP, NTP3: QLIKE-обученный гейт доверия гибриду. F = M^(1−π)·C8^π, π = σ(a_i + b·(m_t/s_t) + c·|ℓ_d| + e·z_mkt_level), ℓ_d = ln(M/K1), z_mkt_level = медиана ln RV по бумагам (`med_lrv`).
Оценка — в первый торговый день d0 месяца по парам t⁺ < d0 всех бумаг панели (расширяющееся окно), как в C (`v4_c_run`): цель — средний QLIKE объединённого прогноза + штрафы κ = λ = 250·s²
(иерархия a_i = α + δ_i, гребень на (b, c, e)); решатель и целевая функция — `v4_c_fit.fit` (база ln M, «ℓ» = ln(C8/M), вес на C8). До первых 60 закрытых пар — приор π = 0,5.
Исключённая бумага (LOTO) не входит в оценку и получает δ = 0. Признак m_t/s_t — из `v4_ntp_state` (по закрытым парам бумаги, панельная часть — без исключённой)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from models_v2.s11 import v4_b_fit as FT, v4_b_run as BR, v4_c_fit as CF

FEATS = ["zmq", "abs_ell", "med_lrv"]


def _fit_month(P, X, lM, ell, tix, nT, d0, ex, prev) -> tuple:
    k = X.shape[1]
    closed = BR.closed_mask(P["t_plus"].to_numpy(), d0) & (tix != ex)
    m = closed & np.isfinite(X).all(1)
    base = {"alpha": 0.0, "delta": np.zeros(nT), "beta": np.zeros(k), "mu": np.zeros(k), "sd": np.ones(k), "ok": True, "grad": 0.0, "nit": 0, "n_train": int(m.sum())}
    npairs = np.bincount(tix[closed], minlength=nT)
    if m.sum() < FT.MIN_PAIRS:
        return base, npairs, prev
    mu, sd = CF.standardize(X[m])
    present, tid = np.unique(tix[m], return_inverse=True)
    x0 = None if prev is None else np.concatenate([[prev["alpha"]], [prev["delta"].get(i, 0.0) for i in present], prev["beta"]])
    r = CF.fit(P["rv"].to_numpy(float)[m], lM[m], ell[m], (X[m] - mu) / sd, tid, len(present), x0)
    d = np.zeros(nT)
    d[present] = r["delta"]
    base.update(alpha=r["alpha"], delta=d, beta=r["beta"], mu=mu, sd=sd, ok=r["ok"], grad=r["grad"], nit=r["nit"])
    return base, npairs, {"alpha": r["alpha"], "beta": r["beta"], "delta": dict(zip(present.tolist(), r["delta"]))}


def _predict(par, X, tix, sel) -> np.ndarray:
    z = np.where(np.isfinite(X[sel]), (X[sel] - par["mu"]) / par["sd"], 0.0)
    return expit(par["alpha"] + par["delta"][tix[sel]] + CF.mv(z, par["beta"]))


def _wrows(month, d0, tickers, par, npairs, pi, tix, ex) -> list:
    base = {"month": month, "d0": str(d0)[:10], "converged": par["ok"]}
    out = [{**base, "ticker": "PANEL", "alpha": par["alpha"], "mean_pi": float(np.mean(pi)), "n_pairs": int(npairs.sum()), "n_train": par["n_train"], "grad": par["grad"], "nit": par["nit"],
            **{"b_" + f: float(b) for f, b in zip(FEATS, par["beta"])}}]
    for i, t in enumerate(tickers):
        q = tix == i
        out.append({**base, "ticker": t, "alpha": par["alpha"] + par["delta"][i], "delta": par["delta"][i], "mean_pi": float(np.mean(pi[q])) if q.any() else np.nan, "n_pairs": int(npairs[i]),
                    "excluded": bool(ex == i)})
    return out


def fit_predict(P, tix, tickers, zmq, aux, ex=None) -> tuple:
    """P — строки по (date, ticker) с rv, t_plus, date, m (= M), c8; zmq — признак m/s (NaN до 60 пар); aux (N, 2) — |ℓ_d| и z_mkt_level; ex — индекс исключённой бумаги или None. -> (π, строки весов)."""
    X = np.column_stack([zmq, aux])
    lM, ell = np.log(P["m"].to_numpy(float)), np.log(P["c8"].to_numpy(float)) - np.log(P["m"].to_numpy(float))
    mon = P["date"].dt.to_period("M")
    pi, wrows, prev = np.full(len(P), np.nan), [], None
    for p in mon.unique():
        sel = (mon == p).to_numpy()
        d0 = P["date"][sel].min()
        par, npairs, prev = _fit_month(P, X, lM, ell, tix, len(tickers), d0, ex, prev)
        pi[sel] = _predict(par, X, tix, sel)
        wrows += _wrows(str(p), d0, tickers, par, npairs, pi[sel], tix[sel], ex)
    return pi, wrows
