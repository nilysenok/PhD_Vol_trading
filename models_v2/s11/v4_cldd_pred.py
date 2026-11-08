"""V4-CLD-диагностика, G33: насколько предсказуем D = QLIKE(H) − QLIKE(M) вне выборки (`CLD_PRED.csv`). Модуль читает реализованный D задним числом — вне путей прогноза (тест изоляции).
Строка прогноза d: D̂ — по закрытым парам до начала месяца, D — на цели пары (t⁺). По M16 (SBER — отдельная диагностика), только месяцы с оценённой моделью (не приор).
R² = 1 − Σ(D − D̂)² / Σ(D − D̄)², D̄ — среднее D закрытых пар окна оценки (историческое среднее; R² > 0 — модель лучше «ничего не знающего» прогноза); sign — доля строк, где знак D̂ = знак D (D ≠ 0);
share_pos — доля D > 0 (контроль знака: «всегда плюс»); corr — Пирсон D̂ и D. Группировка — по году/месяцу цели (t⁺), как окна выбора; строки панели не независимы (оценка — описательная)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_stats as ST, v4_io as IO

COLS = ["config", "kind", "period", "n", "r2_oos", "sign_share", "share_pos", "corr", "mean_dhat", "mean_d"]


def metrics(g: pd.DataFrame) -> dict:
    d, dh, db = (g[c].to_numpy(float) for c in ("d_act", "dhat", "dbar"))
    sst = float(np.sum((d - db) ** 2))
    nz = d != 0
    return {"n": len(g), "r2_oos": 1.0 - float(np.sum((d - dh) ** 2)) / sst if sst > 0 else np.nan, "sign_share": float(np.mean(np.sign(dh[nz]) == np.sign(d[nz]))) if nz.any() else np.nan,
            "share_pos": float(np.mean(d > 0)), "corr": float(np.corrcoef(d, dh)[0, 1]) if len(g) > 2 and np.std(dh) > 0 and np.std(d) > 0 else np.nan, "mean_dhat": float(dh.mean()), "mean_d": float(d.mean())}


def active(fc: pd.DataFrame) -> pd.DataFrame:
    """M16, строки с оценённой моделью (dbar конечен), D̂ и D конечны."""
    g = fc[fc["ticker"].isin(IO.M16) & fc["dbar"].notna() & fc["dhat"].notna() & fc["d_act"].notna()]
    return g.reset_index(drop=True)


def pred_table(fcs: dict) -> pd.DataFrame:
    rows = []
    for cfg, fc in fcs.items():
        g = active(fc)
        for w in ST.WINS:
            s = g[ST.win_mask(g, w)]
            if len(s):
                rows.append({"config": cfg, "kind": "window", "period": w, **metrics(s)})
        for p, s in g.groupby(g["t_plus"].dt.to_period("M").astype(str)):
            rows.append({"config": cfg, "kind": "month", "period": p, **metrics(s)})
    return pd.DataFrame(rows, columns=COLS)
