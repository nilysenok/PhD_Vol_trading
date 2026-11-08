"""V4-E: прогнозы E1–E6. F = F0·exp(b) (Ж-222 п. 6); F0 — H (E1) или G = B2 (E2–E5; месячный вес B2 из `B_B2.parquet`, без изменений); E6 — H·exp(b_m + w_m·ℓ) с месячной оценкой и дневной EWMA поверх.
Спецификации (полураспады, κ) фиксированы объявлением в GATES_V4.md («Этап E»). Панель P: ticker, date, t_plus, rv, h, k2, k3; G — таблица ticker, date, F (B2). exclude — бумага вне панельных величин (LOTO)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_e_calib as CAL, v4_e_e6 as E6, v4_e_k5 as K5

CFGS = ["E1", "E2", "E3", "E4", "E5", "E6"]
SPEC = {"E1": ("H", "own", 60, 60), "E2": ("G", "own", 60, 60), "E3": ("G", "panel", 20, 20), "E4": ("G", "two", 120, 20), "E5": ("G", "kalman", None, None), "E6": ("H+(b,w)", "own", 20, 20)}


def prepare(P: pd.DataFrame, G: pd.DataFrame) -> tuple:
    """-> (строки по (date, ticker), список бумаг, tix). F0-колонка g — B2-прогноз (NaN, если нет)."""
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    tk = sorted(P["ticker"].unique())
    P = P.merge(G[["ticker", "date", "F"]].rename(columns={"F": "g"}), on=["ticker", "date"], how="left").sort_values(["date", "ticker"]).reset_index(drop=True)
    return P, tk, P["ticker"].map({t: i for i, t in enumerate(tk)}).to_numpy()


def forecast(cfg: str, P: pd.DataFrame, G: pd.DataFrame, exclude: str | None = None, zero_b: bool = False) -> tuple:
    """-> (прогнозы: ticker, date, t_plus, rv, h, F, b[, w], c_panel; мета-информация {'panel': бумаги панели, ...}). zero_b — калибровка выключена (b ≡ 0; проверка тождеств E1 = H, E2 = G)."""
    P, tk, tix = prepare(P, G)
    nT = len(tk)
    pm = np.array([t != exclude for t in tk])
    rv, H, date, tp = P["rv"].to_numpy(float), P["h"].to_numpy(float), P["date"].to_numpy(), P["t_plus"].to_numpy()
    meta = {"panel": [t for t in tk if t != exclude], "cfg": cfg}
    w = np.full(len(P), np.nan)
    if cfg == "E6":
        lnF, w, bm, meta["months"] = E6.monthly_base(rv, H, P["k2"].to_numpy(float), P["k3"].to_numpy(float), tix, date, tp, nT, None if exclude is None else tk.index(exclude))
        F0, cpan = np.exp(lnF), np.ones(len(P))
    else:
        F0 = H if cfg == "E1" else P["g"].to_numpy(float)
        cpan = np.ones(len(P))
    _, mode, hl_o, hl_p = SPEC[cfg]
    if cfg == "E5":
        b, meta["kalman"] = K5.forecast_b(rv, F0, tix, date, tp, nT, pm)
    else:
        b, cpan = CAL.ewma_b(rv, F0, tix, date, tp, nT, mode, hl_o, hl_p, pm)
    if zero_b:
        b = np.zeros(len(P))
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["b"], out["w"], out["c_panel"], out["F0"] = F0 * np.exp(b), b, w, cpan, F0
    return out, meta


def pit_violation(cfg: str, P: pd.DataFrame, G: pd.DataFrame, day) -> bool:
    """Обрезка: подмена RV пар, закрытых после `day` (t⁺ > day), не должна менять F ни одного дня ≤ day (E6 и G — на тех же входах). True — нарушение."""
    P = P[P["date"] <= day].reset_index(drop=True)
    Q = P.copy()
    Q.loc[Q["t_plus"] > day, "rv"] *= 7.3
    a, b = forecast(cfg, P, G)[0], forecast(cfg, Q, G)[0]
    s = a["date"] <= day
    return not np.array_equal(a.loc[s, "F"].to_numpy(), b.loc[s, "F"].to_numpy())
