"""Таблицы волны C потока XGB (C10): HP_AXES_C.csv (ось → Δcrit_pen), HP_RESULTS_C.csv (все строки C + опоры), синхронизация CHECKPOINT."""
from __future__ import annotations

import pandas as pd

from models_v2.s6.xgb import c_rows, cfgx

AXIS_COLS = ["id", "axis", "u", "hp_2023", "clamped_years", "duplicate", "d_crit_pen", "axis_min_d_crit_pen", "crit_pen", "qlike_all", "qlike_2017_2021", "qlike_2022",
             "qlike_2023_2025", "qlike_no_240222", "d_harj_no_240222", "beats_base", "perspective", "unstable", "status"]
RES_COLS = ["id", "kind", "family", "moves", "trial_id", "crit_pen", "d_crit_pen", "crit_mean", "qlike_all", "qlike_2017_2021", "qlike_2022", "qlike_2023_2025",
            "qlike_2026", "qlike_no_240222", "qlike_no_2022", "mean_r", "d_harj", "p_harj", "d_harj_no_240222", "p_harj_no_240222", "d_f4", "p_f4", "d_f4_no_240222",
            "p_f4_no_240222", "d_base", "p_base", "d_base_no_240222", "p_base_no_240222", "share_years_win_harj", "beats_base", "perspective", "unstable", "n_dates",
            "J_guide", "years_not_i2", "choice", "status"]
NOTE_J = "строки с J_guide / x2_annual: колонки 2017–2021 — потолок (кандидаты видели 2017–2021); p — сырые"


def axes_table(rows: dict) -> pd.DataFrame:
    x1 = pd.DataFrame([r for r in rows.values() if r.get("kind") == "x1"])
    if x1.empty:
        return pd.DataFrame(columns=AXIS_COLS)
    if "duplicate" not in x1:
        x1["duplicate"] = False
    ok = x1[x1["duplicate"].isna() | (x1["duplicate"] == False)]  # noqa: E712 — NaN/False = не дубль
    x1 = x1.assign(axis_min_d_crit_pen=x1["axis"].map(ok.groupby("axis")["d_crit_pen"].min()))
    return x1.reindex(columns=AXIS_COLS).reset_index(drop=True)


def results_table(rows: dict) -> pd.DataFrame:
    keep = [r for r in rows.values() if not r.get("duplicate") and r.get("kind") != "neighbor"]
    order = {"control": 0, "x1": 1, "x2_best_j": 2, "x2_annual": 3, "x3": 4}
    keep.sort(key=lambda r: (order.get(r["kind"], 9), r["id"]))
    return pd.DataFrame(keep).reindex(columns=RES_COLS)


def stage_tables(rows: dict) -> int:
    ax, rs = axes_table(rows), results_table(rows)
    c_rows.write_csv(ax, cfgx.XGB_REP / "HP_AXES_C.csv", "C-X1: одно-осевые развёртки вокруг годового выбора И2 на Б-2+F4; Δcrit_pen = точка − ctrl_f4 (минус — лучше); u — шаги оси")
    c_rows.write_csv(rs, cfgx.XGB_REP / "HP_RESULTS_C.csv", "опоры ctrl_f4 / x3_gamma_f4, X1, X2, X3; Δ к опоре своей семьи; " + NOTE_J)
    ck = c_rows.sync_checkpoint()
    dup = rs["id"].duplicated().any() or rs["trial_id"].dropna().duplicated().any()
    print("C10: HP_AXES_C %d строк, HP_RESULTS_C %d строк; used.hp = %d; дубли id/trial_id: %s" % (len(ax), len(rs), ck["used"]["hp"], "ЕСТЬ" if dup else "нет"))
    return 1 if dup else 0
