"""V4-I: walk-forward прогноз компонент и конфигураций I1–I3. Параметры месяца m оцениваются один раз, в первый торговый день d0 месяца, по парам, закрытым ДО d0 (t⁺ < d0; `v4_b_run.closed_mask`) всех бумаг панели
(кроме исключённой в LOTO), расширяющееся окно с 2017-01. До 60 закрытых пар (и в строках с нефинитным признаком) компонентный прогноз S = K8. Панель P: ticker, date, t_plus, rv, h, k2, k3; Fe — признаки и цели (`v4_i_comp.build`).
I1: F = S = Ĉ + Ĵ; I2: F = √(K8·S); I3: F = K8·exp(θ(ln S − ln K8)), θ — раз в месяц по закрытым парам с компонентным прогнозом (прогнозы прошлых месяцев — вне выборки)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_run as BR, v4_i_comp as IC, v4_i_fit as FT, v4_rqc_calib as CAL

CFGS = ["I1", "I2", "I3"]
MIN_PAIRS = 60


def prepare(P: pd.DataFrame, Fe: pd.DataFrame) -> tuple:
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    Fe = P[["ticker", "date"]].merge(Fe, on=["ticker", "date"], how="left")
    tk = sorted(P["ticker"].unique())
    return P, Fe, CAL.k8(P["h"], P["k2"], P["k3"]), tk, P["ticker"].map({t: i for i, t in enumerate(tk)}).to_numpy()


def _month(Fe, tix, tp, d0, ex, nT, prev):
    """Оценка месяца -> параметры (или None до 60 пар)."""
    cl = BR.closed_mask(tp, d0) & (tix != ex)
    hc = cl & np.isfinite(Fe[IC.HAR_F + ["c_t"]].to_numpy(float)).all(1) & (Fe["c_t"].to_numpy(float) > 0)
    jc = cl & np.isfinite(Fe[IC.JMP_F + ["ev_t"]].to_numpy(float)).all(1)
    if min(hc.sum(), jc.sum()) < MIN_PAIRS:
        return None
    ph, pj = np.unique(tix[hc]), np.unique(tix[jc])
    har = FT.har_fit(Fe.loc[hc, IC.HAR_F].to_numpy(float), np.log(Fe.loc[hc, "c_t"].to_numpy(float)), np.searchsorted(ph, tix[hc]), len(ph))
    lg = FT.logit_fit(Fe.loc[jc, IC.JMP_F].to_numpy(float), Fe.loc[jc, "ev_t"].to_numpy(float), np.searchsorted(pj, tix[jc]), len(pj))
    pos = jc & (Fe["ev_t"].to_numpy(float) > 0) & np.isfinite(Fe["j_t"].to_numpy(float))
    return {"har": har, "lg": lg, "ph": ph, "pj": pj, "m_pan": float(Fe.loc[pos, "j_t"].mean()) if pos.any() else np.nan, "n_h": int(hc.sum()), "n_j": int(jc.sum())}


def _coef_row(mth, d0, par) -> dict:
    r = {"month": str(mth), "d0": str(d0)[:10], "fitted": par is not None, "n_har": 0, "n_jump": 0, "sig2_c": np.nan, "m_pan": np.nan, "jump_ok": False, "jump_grad": np.nan,
         **{"b_" + f: np.nan for f in IC.HAR_F}, **{"g_" + f: np.nan for f in IC.JMP_F}}
    if par is not None:
        r.update(n_har=par["n_h"], n_jump=par["n_j"], sig2_c=par["har"]["sig2"], m_pan=par["m_pan"], jump_ok=bool(par["lg"]["ok"]), jump_grad=par["lg"]["grad"],
                 **{"b_" + f: float(v) for f, v in zip(IC.HAR_F, par["har"]["theta"][1:4])}, **{"g_" + f: float(v) for f, v in zip(IC.JMP_F, par["lg"]["theta"][1:4])})
    return r


def _slot(present, tix):
    """Номер бумаги среди обученных (−1 — не было в обучении)."""
    pos = np.searchsorted(present, tix)
    ok = (pos < len(present)) & (present[np.minimum(pos, len(present) - 1)] == tix)
    return np.where(ok, pos, -1)


def components(P: pd.DataFrame, Fe: pd.DataFrame, exclude: str | None = None) -> tuple:
    """-> (таблица: ticker, date, t_plus, rv, h, K8, S, chat, jhat, phat, mhat; коэффициенты по месяцам). Строки без компонентного прогноза: S = K8, chat = jhat = phat = mhat = NaN."""
    P, Fe, K8, tk, tix = prepare(P, Fe)
    ex = -1 if exclude is None else tk.index(exclude)
    tp, mon = P["t_plus"].to_numpy(), P["date"].dt.to_period("M")
    n = len(P)
    chat, phat, mhat = (np.full(n, np.nan) for _ in range(3))
    rows, prev = [], None
    for mth in mon.unique():
        sel = (mon == mth).to_numpy()
        d0 = P["date"][sel].min()
        par = _month(Fe, tix, tp, d0, ex, len(tk), prev)
        if par is not None:
            Xh, Xj = Fe.loc[sel, IC.HAR_F].to_numpy(float), Fe.loc[sel, IC.JMP_F].to_numpy(float)
            oh = np.isfinite(Xh).all(1)
            c = np.full(int(sel.sum()), np.nan)
            c[oh] = FT.har_predict(par["har"], Xh[oh], _slot(par["ph"], tix[sel][oh]))
            oj = np.isfinite(Xj).all(1) & np.isfinite(Fe.loc[sel, "m_ew"].to_numpy(float))
            p, m = np.full(len(c), np.nan), np.full(len(c), np.nan)
            p[oj] = FT.logit_predict(par["lg"], Xj[oj], _slot(par["pj"], tix[sel][oj]))
            m[oj] = FT.shrink_m(Fe.loc[sel, "m_ew"].to_numpy(float)[oj], Fe.loc[sel, "n_ev"].to_numpy(float)[oj], par["m_pan"])
            chat[sel], phat[sel], mhat[sel] = c, p, m
        rows.append(_coef_row(mth, d0, par))
    jhat = phat * mhat
    S = np.where(np.isfinite(chat + jhat), chat + jhat, K8)
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["K8"], out["S"], out["chat"], out["jhat"], out["phat"], out["mhat"] = K8, S, chat, jhat, phat, mhat
    return out, pd.DataFrame(rows)


def thetas(cmp: pd.DataFrame, P: pd.DataFrame, exclude: str | None = None) -> pd.DataFrame:
    """θ месяца для I3: по закрытым парам панели с компонентным прогнозом (u = ln S − ln K8; NaN-компонент — строка не участвует). До 60 пар θ = 0."""
    tk = sorted(cmp["ticker"].unique())
    tix = cmp["ticker"].map({t: i for i, t in enumerate(tk)}).to_numpy()
    ex = -1 if exclude is None else tk.index(exclude)
    tp, mon = cmp["t_plus"].to_numpy(), cmp["date"].dt.to_period("M")
    u = np.log(cmp["S"].to_numpy()) - np.log(cmp["K8"].to_numpy())
    x, have = cmp["rv"].to_numpy(float) / cmp["K8"].to_numpy(float), np.isfinite(cmp["chat"].to_numpy(float) + cmp["jhat"].to_numpy(float))
    rows = []
    for mth in mon.unique():
        d0 = cmp["date"][(mon == mth).to_numpy()].min()
        m = BR.closed_mask(tp, d0) & (tix != ex) & have
        rows.append({"month": str(mth), "d0": str(d0)[:10], "n_train": int(m.sum()), "theta": FT.theta_fit(x[m], u[m]) if m.sum() >= MIN_PAIRS else 0.0})
    return pd.DataFrame(rows)


def combine(cfg: str, cmp: pd.DataFrame, th: pd.DataFrame, zero_theta: bool = False) -> pd.DataFrame:
    """I1/I2/I3 из компонентного прогноза. zero_theta — тождество θ ≡ 0 -> K8 побитно (I3), только для проверки."""
    K8, S = cmp["K8"].to_numpy(), cmp["S"].to_numpy()
    if cfg == "I1":
        F = S
    elif cfg == "I2":
        F = np.sqrt(K8 * S)
    else:
        t = cmp["date"].dt.to_period("M").astype(str).map(dict(zip(th["month"], th["theta"]))).to_numpy(float)
        F = K8 * np.exp((0.0 if zero_theta else t) * (np.log(S) - np.log(K8)))
    out = cmp[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"] = F
    return out


def forecast(cfg: str, P: pd.DataFrame, Fe: pd.DataFrame, exclude: str | None = None, zero_theta: bool = False) -> tuple:
    """-> (прогноз cfg: ticker, date, t_plus, rv, h, F; компонентный прогноз; коэффициенты месяцев: HAR-C, скачок, θ)."""
    cmp, co = components(P, Fe, exclude)
    th = thetas(cmp, P, exclude)
    return combine(cfg, cmp, th, zero_theta), cmp, co.merge(th[["month", "n_train", "theta"]], on="month")


def pit_violation(cfg: str, P: pd.DataFrame, dailies: dict, day, scale: float = 7.3, lead: int = 0) -> bool:
    """Обрезка боевого пути на реальных/синтетических данных: подмена RV пар с t⁺ > day и дневных RV, J (и C) после day не меняет F ни одной строки с date <= day. True — нарушение.
    day — первый торговый день месяца (ловит «закрыто + 1»). lead — только для теста-мутации; боевое 0."""
    day = pd.Timestamp(day)
    lim = day + pd.Timedelta(days=31)
    P = P[P["date"] <= lim]
    B = P.copy()
    B.loc[B["t_plus"] > day, "rv"] *= scale
    D2 = {}
    for t, d in dailies.items():
        e = d.copy()
        late = e.index > day
        e.loc[late, ["rv", "j"]] = e.loc[late, ["rv", "j"]].to_numpy() * scale
        D2[t] = e
    fa = forecast(cfg, P, IC.build(P, dailies, lead))[0]
    fb = forecast(cfg, B, IC.build(B, D2, lead))[0]
    m = (fa["date"] <= day).to_numpy()
    return not np.array_equal(fa["F"].to_numpy()[m], fb["F"].to_numpy()[m])
