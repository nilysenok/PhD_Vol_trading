"""Ж-214, A0: фиксация HAR-J(б) — контрольной версии. Формула (models_v2/har.py, daily.py), восстановлена и сверена с components.HAR:

  ln RV_{t+1} = b0 + b_d·ln RV_t + b_w·ln RV_w,t + b_m·ln RV_m,t + b_J·ln(1 + J_t) + ε,   σ̂²_b = exp(f + s²/2),  s² = SSR/(n − 5),
  RV_w, RV_m — средние УРОВНЕЙ RV за 5/22 дня сегмента (включая t); J_t = max(RV − BV, 0), BV = (π/2)·Σ|r_i||r_{i−1}|; RV, BV, J — в %².
OLS на парах (t, t+1), t+1 <= d, один сегмент (разрыв 28.02–24.03.2022 не склеивается), расширяющееся окно, ПЕРЕОЦЕНКА КАЖДЫЙ ДЕНЬ d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D, har as H, metrics
from models_v2.s9 import hv2_paths as P

BCOLS = ["b0", "b_d", "b_w", "b_m", "b_J"]


def panel_grid(bars: pd.DataFrame):
    d, _ = D.build_daily(bars)
    p = H.make_panel(d)
    return d, p, H.eligible(p)


def fit_at(p: H.Panel, i: int, extra: int = 0):
    """(beta, s2, n) по парам t+1 <= i + extra. extra > 0 — МУТАНТ «будущие пары»: должен ловиться тестом."""
    Xt, yt = H.pairs(p, i, H.COLS_HARJ, upto=i + extra)
    return H.ols(Xt, yt)


def harj_series(p: H.Panel, el: np.ndarray, extra: int = 0) -> pd.DataFrame:
    rows = []
    for i in el:
        b, s2, n = fit_at(p, int(i), extra)
        sa, sb = H.predict(b, p.X[i, list(H.COLS_HARJ)], s2)
        rows.append([p.dates[i], p.dates[i + 1], *b, s2, n, sa, sb, p.rv[i + 1]])
    return pd.DataFrame(rows, columns=["d", "target_date", *BCOLS, "s2", "n_pairs", "sigma2_a", "sigma2_b", "rv"])


def compare(s: pd.DataFrame, comp: pd.DataFrame) -> dict:
    same_grid = bool((s["d"].to_numpy() == comp["d"].to_numpy()).all() and (s["target_date"].to_numpy() == comp["target_date"].to_numpy()).all())
    return {"same_grid": same_grid, "n": len(s), "max_rel_rv": float(np.abs(s["rv"] / comp["rv"] - 1).max()),
            "max_rel_har": float(np.abs(s["sigma2_b"] / comp["HAR"] - 1).max()),
            "qlike_har_comp": float(metrics.qlike(comp["rv"], comp["HAR"]).mean()),
            "qlike_har_rebuilt": float(metrics.qlike(s["rv"], s["sigma2_b"]).mean()),
            "qlike_variant_a": float(metrics.qlike(s["rv"], s["sigma2_a"]).mean())}


def spec_facts(p: H.Panel, el: np.ndarray, s: pd.DataFrame) -> dict:
    """Окна и сегменты для A0_SPEC: первая/последняя дата обучения, пар на первой/последней дате, строк в сегментах."""
    return {"first_row_index": int(el[0]), "first_date": str(pd.Timestamp(p.dates[0]).date()), "n_pairs_first": int(s["n_pairs"].iloc[0]),
            "n_pairs_last": int(s["n_pairs"].iloc[-1]), "rows_seg0": int((p.seg == 0).sum()), "rows_seg1": int((p.seg == 1).sum()),
            "beta_last": {k: float(s[k].iloc[-1]) for k in BCOLS}, "s2_last": float(s["s2"].iloc[-1])}
