"""V4-H: walk-forward прогноз F = K8·exp(clip(θᵀz̃, ±0,5)). Параметры месяца m оцениваются один раз, в первый торговый день d0 месяца, по парам, закрытым ДО d0 (t⁺ < d0; `v4_b_run.closed_mask`), всех бумаг панели,
расширяющееся окно с 2017-01; проекция z на (log H, log M, ℓ, |ℓ|) и стандартизация — по тем же закрытым парам (`v4_h_fit.orthogonalize_*`). До 60 закрытых пар g = 0 (F = K8 побитно). Исключённая бумага (LOTO) не входит в оценку;
в H8 θ_i = θ + η_i, у исключённой и у бумаги без пар η = 0. Строка обучения с нефинитным признаком из оценки исключается; в прогнозе нефинитный признак -> z̃ = 0. Панель P: ticker, date, t_plus, rv, h, k2, k3; Zh — длинная таблица признаков."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_fit as FT, v4_b_run as BR, v4_h_feat as HF, v4_h_fit as HFIT, v4_rqc_calib as CAL

CFGS = ["H%d" % i for i in range(1, 9)]


def prepare(P: pd.DataFrame, Zh: pd.DataFrame, feats: list) -> tuple:
    """-> (P по (date, ticker), X признаки, B базовые величины [ln H, ln M, ℓ, |ℓ|], K8, tickers, tix)."""
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    X = P[["ticker", "date"]].merge(Zh[["ticker", "date"] + feats], on=["ticker", "date"], how="left")[feats].to_numpy(float)
    lH, lM = np.log(P["h"].to_numpy(float)), 0.5 * (np.log(P["k2"].to_numpy(float)) + np.log(P["k3"].to_numpy(float)))
    B = np.column_stack([lH, lM, lM - lH, np.abs(lM - lH)])
    tk = sorted(P["ticker"].unique())
    return P, X, B, CAL.k8(P["h"], P["k2"], P["k3"]), tk, P["ticker"].map({t: i for i, t in enumerate(tk)}).to_numpy()


def _fit_month(x, X, B, tix, nT, tp, d0, ex, hier, prev) -> tuple:
    k = X.shape[1]
    cl = BR.closed_mask(tp, d0) & (tix != ex)
    m = cl & np.isfinite(X).all(1)
    base = {"theta": np.zeros(k), "eta": np.zeros((nT, k)), "proj": None, "ok": True, "grad": 0.0, "nit": 0, "n_train": int(m.sum())}
    if m.sum() < FT.MIN_PAIRS:
        return base, np.bincount(tix[cl], minlength=nT), prev
    pj = HFIT.orthogonalize_fit(X[m], B[m])
    Z = HFIT.orthogonalize_apply(pj, X[m], B[m])
    present, tid = np.unique(tix[m], return_inverse=True)
    x0 = None if prev is None else (np.concatenate([prev["theta"], np.array([prev["eta"].get(t, np.zeros(k)) for t in present]).ravel()]) if hier else prev["theta"])
    r = HFIT.fit(x[m], Z, tid, len(present), hier, x0)
    eta = np.zeros((nT, k))
    if hier:
        eta[present] = r["eta"]
    base.update(theta=r["theta"], eta=eta, proj=pj, ok=r["ok"], grad=r["grad"], nit=r["nit"])
    return base, np.bincount(tix[cl], minlength=nT), {"theta": r["theta"], "eta": {int(t): r["eta"][i] for i, t in enumerate(present)} if hier else {}}


def _predict(par, x, X, B, tix, sel):
    if par["proj"] is None:
        return np.zeros(int(sel.sum())), np.zeros((int(sel.sum()), X.shape[1]))
    Z = HFIT.orthogonalize_apply(par["proj"], X[sel], B[sel])
    g = np.einsum("ij,j->i", Z, par["theta"]) + np.einsum("ij,ij->i", Z, par["eta"][tix[sel]])
    return g, Z


def forecast(cfg: str, P: pd.DataFrame, Zh: pd.DataFrame, exclude: str | None = None, zero_theta: bool = False) -> tuple:
    """-> (прогнозы: ticker, date, t_plus, rv, h, F, g; коэффициенты по месяцам: θ панели, n_train, сходимость). zero_theta — тождество θ ≡ 0 (F = K8 побитно), только для проверки."""
    feats = HF.cfg_feats(cfg)
    P, X, B, K8, tk, tix = prepare(P, Zh, feats)
    x, tp = P["rv"].to_numpy(float) / K8, P["t_plus"].to_numpy()
    ex = -1 if exclude is None else tk.index(exclude)
    mon = P["date"].dt.to_period("M")
    g_all, rows, prev = np.zeros(len(P)), [], None
    for mth in mon.unique():
        sel = (mon == mth).to_numpy()
        d0 = P["date"][sel].min()
        par, npairs, prev = _fit_month(x, X, B, tix, len(tk), tp, d0, ex, cfg == "H8", prev)
        g_all[sel] = 0.0 if zero_theta else _predict(par, x, X, B, tix, sel)[0]
        rows.append({"config": cfg, "month": str(mth), "d0": str(d0)[:10], "converged": par["ok"], "n_train": par["n_train"], "n_pairs": int(npairs.sum()), "grad": par["grad"], "nit": par["nit"],
                     **{"th_" + f: float(v) for f, v in zip(feats, par["theta"])}})
    g = HFIT.clip_g(g_all)
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["g"] = K8 * np.exp(g), g
    return out, pd.DataFrame(rows)


RAW_SHIFT = ["ell", "abs_ell", "lerr_h", "sur_mkt", "med_lrv", "jshare", "lasth", "herf"]


def _perturb_after(Zl: pd.DataFrame, fh: pd.DataFrame, day) -> tuple:
    """Подмена исходных величин (Z) и доли первого часа строго после `day`."""
    Z, F = Zl.copy(), fh.copy()
    late = Z["date"] > day
    Z.loc[late, RAW_SHIFT] = Z.loc[late, RAW_SHIFT].to_numpy() + 0.7
    F.loc[F["date"] > day, "firsth"] += 0.3
    return Z, F


def pit_violation(cfg: str, P: pd.DataFrame, Zl: pd.DataFrame, fh: pd.DataFrame, day, scale: float = 7.3, lead: int = 0) -> bool:
    """Обрезка боевого пути (признаки из Z + оценка + прогноз) на полной панели: подмена RV пар с t⁺ > day и исходных величин после day не должна менять F ни одной строки с date <= day. True — нарушение.
    day — первый торговый день месяца (ловит «закрыто + 1» в оценке). lead — только для теста-мутации («признак дня d+1»); боевое 0."""
    day = pd.Timestamp(day)
    lim = day + pd.Timedelta(days=31)                  # строки после day (до lim) нужны, чтобы проверить, что они не влияют; дальше — не нужны (скорость)
    P, Zl, fh = P[P["date"] <= lim], Zl[Zl["date"] <= lim], fh[fh["date"] <= lim]
    B = P.copy()
    B.loc[B["t_plus"] > day, "rv"] *= scale
    Z2, F2 = _perturb_after(Zl, fh, day)
    fa = forecast(cfg, P, HF.build(Zl, fh, lead=lead))[0]
    fb = forecast(cfg, B, HF.build(Z2, F2, lead=lead))[0]
    m = (fa["date"] <= day).to_numpy()
    return not np.array_equal(fa["F"].to_numpy()[m], fb["F"].to_numpy()[m])
