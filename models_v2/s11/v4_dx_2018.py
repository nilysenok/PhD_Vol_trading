"""V4-DX, G26: разбор 2018 (год цели; все гибриды C хуже HAR-J) — ЗАДНИМ ЧИСЛОМ, описательно. Для C8 (лучшая C) и C1: помесячный вклад в проигрыш K1 (месяц цели t⁺; вклад = Σ строк (QLIKE(C) − QLIKE(K1)) / число строк года, M16 пулом — сумма вкладов
= средняя Δ года), вклад бумаг (17), путь веса на M (средний вес месяца против лучшего постоянного веса месяца и года задним числом, геометрическая смесь H^(1−w)·M^w, M = √(K2·K3), пул M16)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_cache as BC, v4_dx_years as DY, v4_io as IO

SERIES = ("C8", "C1")
YEAR = 2018


def load(P: pd.DataFrame, series: str) -> pd.DataFrame:
    """Прогнозы ряда в 2018 (цель) + k2, k3 панели: ticker, date, t_plus, rv, h, F, w, k2, k3, d, month, ell."""
    f = pd.read_parquet(BC.OUT / (series + ".parquet"))[["ticker", "date", "t_plus", "rv", "h", "F", "w"]]
    f = f.merge(P[["ticker", "date", "k2", "k3"]], on=["ticker", "date"])
    f = f[f["t_plus"].dt.year == YEAR].copy()
    f["d"] = DY.ql(f["rv"], f["F"]) - DY.ql(f["rv"], f["h"])
    f["month"] = f["t_plus"].dt.to_period("M").astype(str)
    f["ell"] = 0.5 * np.log(f["k2"] * f["k3"]) - np.log(f["h"])
    return f


def best_w(g: pd.DataFrame) -> float:
    """Лучший постоянный вес на M (геометрическая смесь, [−1; 2]) на строках g задним числом."""
    return float(DY.solve_w((g["rv"] / g["h"]).to_numpy(float), g["ell"].to_numpy(float), np.ones((1, len(g))))[0])


def table(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SERIES:
        f = load(P, s)
        m16 = f[f["ticker"].isin(IO.M16)]
        n_year, w_year = len(m16), best_w(m16)
        for mo, g in m16.groupby("month"):
            rows.append({"kind": "month", "series": s, "key": mo, "n": len(g), "d_vs_k1": float(g["d"].mean()), "contrib_to_year": float(g["d"].sum() / n_year), "mean_w": float(g["w"].mean()),
                         "w_best_period": best_w(g), "w_best_year": w_year})
        rows.append({"kind": "year", "series": s, "key": str(YEAR), "n": n_year, "d_vs_k1": float(m16["d"].mean()), "contrib_to_year": float(m16["d"].sum() / n_year), "mean_w": float(m16["w"].mean()),
                     "w_best_period": w_year, "w_best_year": w_year})
        for t, g in f.groupby("ticker"):
            rows.append({"kind": "ticker", "series": s, "key": t, "n": len(g), "d_vs_k1": float(g["d"].mean()), "contrib_to_year": float(g["d"].sum() / n_year) if t in IO.M16 else np.nan,
                         "mean_w": float(g["w"].mean()), "w_best_period": best_w(g), "w_best_year": w_year})
    return pd.DataFrame(rows)
