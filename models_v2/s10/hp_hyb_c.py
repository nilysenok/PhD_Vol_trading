"""P2, G9/G10, архитектура C (У-11): Market / Idiosyncratic.
Реализованные части на t⁺: m = β²_{i,t}·RV_IMOEX,t⁺ (β — как K7, известна в t), u = max(RV − m, ε·RV), ε = 0,01. База: Ĥ^m = β²·Ĥ(IMOEX), Ĥ^u = HAR-J(u) — части K7 (hp_ctrl.k7_series), K7 = Ĥ^m + Ĥ^u.
g_m, g_u — две панельные QLIKE-поправки движка к целям m и u с offsets ln Ĥ^m, ln Ĥ^u; f = Ĥ^m·exp(λ_m·clip(g_m)) + Ĥ^u·exp(λ_u·clip(g_u)); (глубина, c, λ_m, λ_u) — раз в год по QLIKE итоговой f к RV за Y−1.
C2 = C1 при g_m ≡ 0. λ = 0 ≡ K7 побитно."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s10 import hp_ctrl as CT, hp_engine as E, hp_hyb_core as C

DEFAULT_C = dict(depth=2, c=1.0, lam_m=1.0, lam_u=1.0)


def beta2_at(d: pd.DataFrame) -> np.ndarray:
    """β²_t по прошлому (окно 252 строки, включая день t, но не t⁺). Мутант «β с днём t⁺» подменяет эту функцию."""
    return (CT.beta_past(d) ** 2).to_numpy(float)


def k7_parts(d: pd.DataFrame, imo: pd.DataFrame, dates, k1_imo: pd.DataFrame) -> pd.DataFrame:
    """На датах `dates` бумаги: hm = β²·Ĥ(IMOEX), hu = HAR-J(u), b2 = β². hm + hu = σ̂²_K7 побитно (тот же порядок операций, что в hp_ctrl.k7_series)."""
    pu = H.make_panel(CT.u_daily(d, imo))
    rows = np.flatnonzero(np.isin(d.index.to_numpy(), np.asarray(dates)))
    hu = np.array([CT.harj_at(pu, int(i)) for i in rows])
    h_imo = k1_imo.set_index("d")["sigma2"].reindex(d.index[rows]).to_numpy(float)
    b2 = beta2_at(d)[rows]
    return pd.DataFrame({"date": d.index[rows], "hm": b2 * h_imo, "hu": hu, "b2": b2})


def targets(panel: pd.DataFrame, b2: np.ndarray, eps: float = CT.EPS_U) -> tuple:
    """(m, u) — реализованные части на t⁺; нет β² или RV индекса на t⁺ — NaN."""
    rv, rm = panel["rv_tplus"].to_numpy(float), panel["mkt_rv_tplus"].to_numpy(float)
    m = b2 * rm
    return m, np.where(np.isfinite(m), np.maximum(rv - m, eps * rv), np.nan)


def forecast_c(hm, hu, gm, gu, c, lam_m, lam_u) -> np.ndarray:
    """f = Ĥ^m·exp(λ_m·clip(g_m)) + Ĥ^u·exp(λ_u·clip(g_u)); нет g — поправка 0."""
    return C.apply_lam(hm, gm, c, lam_m) + C.apply_lam(hu, gu, c, lam_u)


def select_c(panel, hm, hu, runs_m, runs_u, Y: int, cs=E.CS, lams=E.LAMS, g_m_zero: bool = False) -> dict:
    """(depth, c, λ_m, λ_u) на год Y по закрытым прогнозам Y−1; g_m_zero — C2 (λ_m не выбирается, g_m ≡ 0)."""
    mk = E.closed_mask(panel["date"], panel["t_plus"], Y)
    tk, rv = panel["ticker"].to_numpy()[mk], panel["rv_tplus"].to_numpy(float)[mk]
    cands = []
    for dp in sorted(runs_u):
        gu = runs_u[dp]["g"].to_numpy()[mk]
        gm = np.zeros(mk.sum()) if g_m_zero else runs_m[dp]["g"].to_numpy()[mk]
        for c, lm, lu in itertools.product(cs, (1.0,) if g_m_zero else lams, lams):
            f = forecast_c(hm[mk], hu[mk], gm, gu, c, lm, lu)
            cands.append((C.mean_q(rv, f, tk, np.ones(len(rv), bool)), dict(depth=dp, c=c, lam_m=lm if not g_m_zero else 0.0, lam_u=lu)))
    return C.pick_min(cands, {**DEFAULT_C, "lam_m": 0.0 if g_m_zero else 1.0})


def finalize_c(panel, hm, hu, runs_m, runs_u, g_m_zero: bool = False, years=None, cs=E.CS, lams=E.LAMS) -> tuple:
    """-> (σ̂² по строкам панели — NaN до 2016, таблица выбора по годам)."""
    yr = pd.DatetimeIndex(panel["date"]).year.to_numpy()
    out, ch = np.full(len(panel), np.nan), []
    for Y in sorted(set(yr[yr >= E.FIRST_YEAR])) if years is None else years:
        s = select_c(panel, hm, hu, runs_m, runs_u, Y, cs, lams, g_m_zero)
        ch.append({"year": int(Y), **s})
        m = yr == Y
        gm = np.zeros(len(panel)) if g_m_zero else runs_m[s["depth"]]["g"].to_numpy()
        out[m] = forecast_c(hm, hu, gm, runs_u[s["depth"]]["g"].to_numpy(), s["c"], s["lam_m"], s["lam_u"])[m]
    return out, pd.DataFrame(ch)
