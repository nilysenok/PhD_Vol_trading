"""V4-RQC: прогнозы K8, RQC1–RQC3 (F = K8·a). Панель P: ticker, date, t_plus, rv, h, k2, k3 (общие даты H∩K2∩K3); Zl — Z-таблица (ticker, date, jshare, dlrv; определение J — как в K1).
Базовые ряды не переобучаются: K1, K2, K3 читаются как есть. exclude — бумага вне панельных величин (LOTO); zero_a — калибровка выключена (a ≡ 1; тождество с K8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_rqc_calib as CAL, v4_rqc_r3 as R3

CFGS = ["K8", "RQC1", "RQC2", "RQC3"]


def prepare(P: pd.DataFrame, Zl: pd.DataFrame) -> tuple:
    """-> (строки по (date, ticker) с jshare, dlrv; список бумаг; tix)."""
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    tk = sorted(P["ticker"].unique())
    P = P.merge(Zl[["ticker", "date", "jshare", "dlrv"]], on=["ticker", "date"], how="left").sort_values(["date", "ticker"]).reset_index(drop=True)
    return P, tk, P["ticker"].map({t: i for i, t in enumerate(tk)}).to_numpy()


def make_pre(P: pd.DataFrame, Zl: pd.DataFrame) -> tuple:
    """Часть RQC3, не зависящая от панели и от моментов закрытия (winsorized y, флаги скачков) — один раз на LOTO."""
    P, tk, tix = prepare(P, Zl)
    return _pre(P, tix, len(tk))


def _pre(P: pd.DataFrame, tix, nT: int) -> tuple:
    return R3.prep(P["rv"].to_numpy(float), CAL.k8(P["h"], P["k2"], P["k3"]), tix, P["jshare"].to_numpy(float), P["dlrv"].to_numpy(float), nT)


def forecast(cfg: str, P: pd.DataFrame, Zl: pd.DataFrame, exclude: str | None = None, zero_a: bool = False, pre: tuple | None = None) -> tuple:
    """-> (ticker, date, t_plus, rv, h, F, a, F0, frozen; мета {'panel': бумаги панели, 'cfg'})."""
    P, tk, tix = prepare(P, Zl)
    nT, n = len(tk), len(P)
    F0 = CAL.k8(P["h"], P["k2"], P["k3"])
    date, tp = P["date"].to_numpy("datetime64[ns]"), P["t_plus"].to_numpy("datetime64[ns]")
    avail = np.asarray(CAL.closed_date(tp, date)).astype("datetime64[ns]")
    pm = np.array([t != exclude for t in tk])
    x, fz = P["rv"].to_numpy(float) / F0, np.zeros(n, bool)
    if cfg == "RQC1":
        a = CAL.global_a(x, avail, date, pm[tix])
    elif cfg == "RQC2":
        a = CAL.yearly_a(x, avail, date, pm[tix])
    elif cfg == "RQC3":
        yw, flag = pre if pre is not None else _pre(P, tix, nT)
        la, fz = R3.run(yw, flag, avail, tix, date, nT, pm)
        a = np.exp(la)
    elif cfg == "K8":
        a = np.ones(n)
    else:
        raise ValueError(cfg)
    if zero_a:
        a = np.ones(n)
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["a"], out["F0"], out["frozen"] = F0 * a, a, F0, fz
    return out, {"panel": [t for t in tk if t != exclude], "cfg": cfg}


def pit_violation(cfg: str, P: pd.DataFrame, Zl: pd.DataFrame, day) -> bool:
    """Обрезка: подмена RV пар, закрытых после `day` (t⁺ > day), не должна менять F ни одного дня ≤ day. True — нарушение."""
    P = P[P["date"] <= day].reset_index(drop=True)
    Q = P.copy()
    Q.loc[Q["t_plus"] > day, "rv"] *= 7.3
    a, b = forecast(cfg, P, Zl)[0], forecast(cfg, Q, Zl)[0]
    s = a["date"] <= day
    return not np.array_equal(a.loc[s, "F"].to_numpy(), b.loc[s, "F"].to_numpy())
