"""ZD: таблица `ZD_CAUSES` — по каждой из 6 причин плана Z (раздел 1) число ZD или путь к готовой таблице. Файлов не пишет.
Колонки: cause, item, period, value, source. Параметры Y2/Y3 — из готовых `YA_PARAMS.csv`, `YB_PARAMS.csv` (только чтение)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_zd_tab as TB

NOT_ZD = {4: "ZD не проверяет (план Z, раздел 1: в этом цикле не проверяется); готовое — docs/otchety/harj_x1/YB_PAIRS.csv (Y3 против Y4)",
          5: "ZD не проверяет (план Z, раздел 1: проверка — лист ZC); готовое — docs/otchety/harj_x1/YA_CELLS.csv"}


def year_means(par: pd.DataFrame, y: str, col: str) -> dict:
    g = par[par["y"] == y]
    return {int(k): float(v) for k, v in g.groupby(g["month"].str[:4].astype(int))[col].mean().items()}


def causes_table(states: pd.DataFrame, pred: pd.DataFrame, pools: pd.DataFrame, ya: pd.DataFrame, yb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    add = lambda c, i, p, v, s: rows.append((c, i, p, float(v), s))      # noqa: E731
    for yr, v in year_means(ya, "Y2", "a_n").items():
        add(1, "a_n Y2, среднее по месяцам", str(yr), v, "YA_PARAMS.csv")
    for yr, g in states[(states["y"] == "Y2") & (states["state"] == "N")].groupby("period"):
        if g["n"].iloc[0] and (yr in map(str, TB.YEARS)):
            add(1, "средний D Y2 в обычные дни N (D > 0 — вред)", yr, g["mean_D"].iloc[0], "ZD_STATES.csv")
    for yr, v in year_means(ya, "Y2", "a_e").items():
        add(2, "a_e Y2, среднее по месяцам", str(yr), v, "YA_PARAMS.csv")
    g2 = ya[ya["y"] == "Y2"]
    add(2, "доля месяцев Y2 с λ = 0,1 (верх решётки), 2018–2021", "2018-2021", (g2[g2["month"] < "2022"]["lam"] >= 0.1).mean(), "YA_PARAMS.csv")
    for yr, v in year_means(ya, "Y2", "lam").items():
        add(3, "λ Y2, среднее по месяцам", str(yr), v, "YA_PARAMS.csv")
    for yr, g in pools.groupby("year"):
        add(3, "доля дней E в пулах (среднее по месяцам года)", str(yr), (g["n_e_pool"] / g["n_pool"]).mean(), "ZD_PRED (пулы walk)")
    for c, text in NOT_ZD.items():
        rows.append((c, text, "—", float("nan"), "не ZD"))
    p2 = pred[(pred["y"] == "Y2") & pred["period"].isin([str(v) for v in TB.YEARS] + list(TB.WINDOWS))]
    for r in p2.itertuples():
        add(6, "AUC (а) Y2: 1[D<0] по z", r.period, r.auc_a, "ZD_PRED.csv")
    s2 = states[(states["y"] == "Y2") & (states["state"] == "all") & states["period"].isin([str(v) for v in range(2018, 2026)])]
    add(6, "число лет 2018–2025, где средний D Y2 > 0 (хуже X1)", "2018-2025", (s2["mean_D"] > 0).sum(), "ZD_STATES.csv")
    for yr, v in year_means(yb, "Y3", "theta").items():
        add(6, "θ Y3, среднее по месяцам", str(yr), v, "YB_PARAMS.csv")
    for yr, g in yb[yb["y"] == "Y3"].groupby(yb["month"].str[:4]):
        add(6, "θ Y3, σ по месяцам", yr, np.std(g["theta"], ddof=0), "YB_PARAMS.csv")
    return pd.DataFrame(rows, columns=["cause", "item", "period", "value", "source"])
