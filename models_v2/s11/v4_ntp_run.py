"""V4-NTP: прогнозы NTP1–NTP4 — F = M^(1−π)·C8^π, M = √(K2·K3), C8 — сохранённый прогноз (`C8.parquet`, не переобучается). π_t строится только по закрытым парам (t⁺ ≤ d; `v4_ntp_state`, NTP3 — `v4_ntp_gate`).
NTP1 π = σ(−m/s); NTP2 fixed share (α = 0,02); NTP3 QLIKE-гейт; NTP4 — NTP1 с шагом |Δπ| ≤ 0,02. Панель P: ticker, date, t_plus, rv, h, k2, k3; G — ticker, date, F (C8); Zl — длинная Z (NTP3).
exclude — бумага вне панельных величин и вне оценки (LOTO). Диагностика (`v4_dx_*`) сюда не импортируется (тест изоляции)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_c_run as CR, v4_ntp_gate as GT, v4_ntp_state as SS, v4_z_feat as ZF

CFGS = ["NTP1", "NTP2", "NTP3", "NTP4"]


def prepare(P: pd.DataFrame, G: pd.DataFrame, Zl: pd.DataFrame | None) -> tuple:
    """-> (строки по (date, ticker) с m = √(K2·K3), c8[, abs_ell, med_lrv]; список бумаг; tix). Строки без прогноза C8 не входят."""
    P = P.merge(G[["ticker", "date", "F"]].rename(columns={"F": "c8"}), on=["ticker", "date"], how="inner")
    if Zl is not None:
        P = P.merge(Zl[["ticker", "date", "abs_ell", "med_lrv"]], on=["ticker", "date"], how="left")
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    P["m"] = np.sqrt(P["k2"].to_numpy(float) * P["k3"].to_numpy(float))
    tk = sorted(P["ticker"].unique())
    return P, tk, P["ticker"].map({t: i for i, t in enumerate(tk)}).to_numpy()


def _pi(cfg, P, tix, tk, ex):
    """-> (π, панельная величина дня, m/s или NaN, строки весов NTP3)."""
    a = (P["rv"].to_numpy(float), P["m"].to_numpy(float), P["c8"].to_numpy(float), tix, P["date"].to_numpy(), P["t_plus"].to_numpy(), len(tk), np.array([t != ex for t in tk]))
    if cfg == "NTP2":
        pi, eta = SS.fixed_share(*a)
        return pi, eta, np.full(len(P), np.nan), []
    m, s, pan, _ = SS.ewma_state(*a)
    pis, z = SS.pi_discount(m, s)
    if cfg == "NTP3":
        pi, wrows = GT.fit_predict(P, tix, tk, z, P[["abs_ell", "med_lrv"]].to_numpy(float), None if ex is None else tk.index(ex))
        return pi, pan, z, wrows
    return (SS.smooth(pis, tix, len(tk)) if cfg == "NTP4" else pis), pan, z, []


def forecast(cfg: str, P: pd.DataFrame, G: pd.DataFrame, Zl: pd.DataFrame | None = None, exclude: str | None = None, force_pi: float | None = None) -> tuple:
    """-> (прогнозы: ticker, date, t_plus, rv, h, F, pi, pan, zmq; таблица весов по месяцам × бумагам). force_pi — только тест тождеств (π ≡ const); боевое значение None."""
    if cfg not in CFGS:
        raise ValueError("неизвестная конфигурация NTP: %s" % cfg)
    P, tk, tix = prepare(P, G, Zl if cfg == "NTP3" else None)
    pi, pan, z, wr = _pi(cfg, P, tix, tk, exclude)
    pi = pi if force_pi is None else np.full(len(P), float(force_pi))
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["pi"], out["pan"], out["zmq"] = SS.combine(P["m"], P["c8"], pi), pi, pan, z
    return out, weights(cfg, out, wr)


def weights(cfg: str, out: pd.DataFrame, wr: list) -> pd.DataFrame:
    """Средний π по месяцам × бумагам (+ PANEL); для NTP3 — параметры гейта (alpha, b_*, converged, delta, excluded)."""
    mon = out["date"].dt.to_period("M").astype(str)
    g = out.assign(month=mon).groupby(["month", "ticker"])["pi"].mean().rename("mean_pi").reset_index()
    pn = out.assign(month=mon).groupby("month")["pi"].mean().rename("mean_pi").reset_index().assign(ticker="PANEL")
    base = pd.DataFrame(wr) if wr else pd.concat([pn, g], ignore_index=True)
    base.insert(0, "config", cfg)
    return base


def pit_violation(cfg: str, P: pd.DataFrame, G: pd.DataFrame, inputs: dict, day, scale: float = 7.3) -> bool:
    """Обрезка боевого пути: подмена RV пар с t⁺ > day (и дневных величин после day) и отбрасывание всех строк после day не меняют π, F ни одной строки с date <= day. day — первый торговый день месяца
    (ловит «закрыто + 1» в состоянии и в оценке NTP3). True — нарушение."""
    day = pd.Timestamp(day)
    B = P.copy()
    B.loc[B["t_plus"] > day, "rv"] *= scale
    alt = {t: {"daily": CR._scale_after(v["daily"], day, scale), "herf": v["herf"].where(v["herf"].index <= day, 0.5)} for t, v in inputs.items()}
    Pc = P[P["date"] <= day].reset_index(drop=True)
    z = (lambda X, inp: ZF.build_panel_z(X, inp)) if cfg == "NTP3" else (lambda X, inp: None)
    fa, fb, fc = forecast(cfg, P, G, z(P, inputs))[0], forecast(cfg, B, G, z(B, alt))[0], forecast(cfg, Pc, G, z(Pc, inputs))[0]
    m = (fa["date"] <= day).to_numpy()
    same = lambda x, y: np.array_equal(x["F"].to_numpy(), y["F"].to_numpy()) and np.array_equal(x["pi"].to_numpy(), y["pi"].to_numpy())
    return not (same(fa[m], fb[m]) and same(fa[m], fc))
