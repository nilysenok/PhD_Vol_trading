"""P2, G9/G10, архитектура B (У-11): Global + Individual поверх A1.
B1: α̂_{i,d} = ln среднее RV/f_A1 по последним 252 ЗАКРЫТЫМ парам бумаги (t⁺ <= d); α = α̂·n/(n + κ); f = f_A1·exp(α).
B2: ридж ln(RV/f_A1) на [1, z(ln RV_d/RV_m)] по тем же парам, штраф κ на оба коэффициента (в т. ч. свободный член); f = f_A1·exp(α + γ·z_d).
κ ∈ {50; 250; 1000} выбирается раз в год Y по закрытым прогнозам года Y−1 (t⁺ < 01.01.Y), критерий — средний по бумагам QLIKE; первый год без закрытой валидации — κ = 1000 (максимум сжатия, R8).
λ = 0 ≡ K1 (А1 при λ = 0, поправки α, γ нулевые: κ → ∞): exp(0) = 1.0 побитно."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_engine as E, hp_hyb_core as C

KAPPAS, WIN = (50.0, 250.0, 1000.0), 252
Z_COL = "z_x_d_m"


def alpha_b1(panel: pd.DataFrame, f_a1: np.ndarray, win: int = WIN) -> tuple:
    """(α̂, n) на строку: ln среднего RV/f_A1 по последним win закрытым парам бумаги; нет пар — α̂ = 0, n = 0."""
    rv = panel["rv_tplus"].to_numpy(float)
    ok = np.isfinite(f_a1) & (f_a1 > 0) & np.isfinite(rv) & (rv > 0)
    m, n = C.trail_agg(np.where(ok, rv / np.where(ok, f_a1, 1.0), 0.0), C.trail_index(panel, ok, win))
    return np.where(n > 0, np.log(np.where(m > 0, m, 1.0)), 0.0), n


def adj_b1(ahat: np.ndarray, n: np.ndarray, kappa: float) -> np.ndarray:
    return ahat * n / (n + kappa)


def adj_b2(panel: pd.DataFrame, f_a1: np.ndarray, kappa: float, win: int = WIN) -> np.ndarray:
    """α + γ·z_d из ридж-регрессии y = ln(RV/f_A1) на [1, z] по закрытым парам (нормальные уравнения 2×2); z строки нет — γ·0."""
    rv, z = panel["rv_tplus"].to_numpy(float), panel[Z_COL].to_numpy(float)
    ok = np.isfinite(f_a1) & (f_a1 > 0) & np.isfinite(rv) & (rv > 0) & np.isfinite(z)
    idx = C.trail_index(panel, ok, win)
    y = np.where(ok, np.log(np.where(ok, rv, 1.0) / np.where(ok, f_a1, 1.0)), 0.0)
    zz = np.where(ok, z, 0.0)
    sy, n = C.trail_agg(y, idx, "sum")
    sz, _ = C.trail_agg(zz, idx, "sum")
    szz, _ = C.trail_agg(zz * zz, idx, "sum")
    szy, _ = C.trail_agg(zz * y, idx, "sum")
    n = n.astype(float)
    sy, sz, szz, szy = (np.nan_to_num(a) for a in (sy, sz, szz, szy))
    a11, a22 = n + kappa, szz + kappa
    det = a11 * a22 - sz * sz
    al, ga = (a22 * sy - sz * szy) / det, (a11 * szy - sz * sy) / det
    return al + ga * np.nan_to_num(z, nan=0.0)


def select_kappa(panel: pd.DataFrame, f_by_kappa: dict, Y: int) -> dict:
    """κ на год Y по закрытым прогнозам года Y−1; f_by_kappa {κ: σ̂² по строкам панели}."""
    m = E.closed_mask(panel["date"], panel["t_plus"], Y)
    tk, rv = panel["ticker"].to_numpy(), panel["rv_tplus"].to_numpy(float)
    return C.pick_min([(C.mean_q(rv, f, tk, m), {"kappa": k}) for k, f in f_by_kappa.items()], {"kappa": max(f_by_kappa)})


def forecast_b(panel: pd.DataFrame, f_a1: np.ndarray, kind: str, kappas=KAPPAS, years=None) -> tuple:
    """B1 / B2: -> (σ̂² по строкам панели — NaN до 2016 и где нет f_A1; таблица выбора κ по годам). f = f_A1·exp(поправка), κ по правилу выше."""
    ahat, n = alpha_b1(panel, f_a1) if kind == "b1" else (None, None)
    adj = {k: (adj_b1(ahat, n, k) if kind == "b1" else adj_b2(panel, f_a1, k)) for k in kappas}
    fk = {k: f_a1 * np.exp(a) for k, a in adj.items()}
    yr = pd.DatetimeIndex(panel["date"]).year.to_numpy()
    out, ch = np.full(len(panel), np.nan), []
    for Y in sorted(set(yr[yr >= E.FIRST_YEAR])) if years is None else years:
        sel = select_kappa(panel, fk, Y)
        ch.append({"year": int(Y), **sel})
        m = yr == Y
        out[m] = fk[sel["kappa"]][m]
    return out, pd.DataFrame(ch)
