"""Перенос строк Ж-204 для XGBoost в FEATURE_SETS / OBJECTIVE_RESULTS БЕЗ пересчёта: числа берутся из CSV Ж-204 по trial_id, crit_pen — из сохранённых фолдов B1 (где есть).
LightGBM-строки (obj_l3l), HAR-гибриды, смеси с HAR и Optuna в таблицы потока не входят."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from models_v2.s5 import common
from models_v2.s6.xgb import critx, rowsx

OLD = common.REP
B1_DATA = common.DEEP / "b1"
STATUS_T = "перенос Ж-204 без пересчёта (trial_id Ж-204); " + rowsx.STATUS
OBJ_IDS = {"ctrl_b2": "L2, ln-цель (Б-2, X1/X2-база)", "obj_l1n": "X1: MSE по уровню RV/S", "obj_l1raw": "X1: MSE по уровню RV (%²)", "obj_l3x": "X3: reg:gamma",
           "obj_l4": "X4: custom QLIKE", "obj_l5a": "X5: L2, веса RV_w^(−0,5)", "obj_l5b": "X5: L2, веса RV_w^(−1)", "obj_l5c": "X5: QLIKE, веса RV_w^(−0,5)"}
COLS = ["id", "trial_id", "transferred", "source", "objective", "features", "crit_mean", "crit_pen", "qlike_all", "qlike_2017_2021", "qlike_2022",
        "qlike_2023_2025", "qlike_2026", "qlike_no_240222", "qlike_no_2022", "mean_r", "d_harj_no_240222", "d_b2", "status"]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(OLD / name, comment="#")


def _pen(row_id: str) -> float:
    p = B1_DATA / ("loyo_%s.parquet" % row_id)
    if not p.exists():
        return float("nan")
    f = pd.read_parquet(p)
    return critx.crit(f[f["ci"] == 0])["crit_pen"]


def _base_q() -> float:
    fa = _read("FEATURE_ABLATIONS.csv")
    return float(fa[fa["model"] == "BASE"]["qlike_all"].iloc[0])


def objectives() -> pd.DataFrame:
    oe, base, out = _read("OBJECTIVE_EXPERIMENTS.csv"), _base_q(), []
    base_tid = _read("FEATURE_ABLATIONS.csv").query("model == 'BASE'")["trial_id"].iloc[0]
    for rid, desc in OBJ_IDS.items():
        r = oe[oe["row_id"] == rid].iloc[0]
        tid = r["trial_id"] if isinstance(r["trial_id"], str) else base_tid
        out.append({"id": rid, "trial_id": tid, "transferred": True, "source": "OBJECTIVE_EXPERIMENTS.csv (Ж-204 B1)", "objective": desc, "features": "Б-2",
                    "crit_mean": r["q_loyo"] if np.isfinite(r["q_loyo"]) else r["q_int"], "crit_pen": _pen(rid), "qlike_all": r["qlike_all"],
                    "qlike_2017_2021": r["qlike_2017_2021"], "qlike_2022": r["qlike_2022"], "qlike_2023_2025": r["qlike_2023_2025"],
                    "qlike_2026": r["qlike_2026"], "qlike_no_240222": r["qlike_no_240222"], "qlike_no_2022": r["qlike_no_2022"], "mean_r": r["mean_r"],
                    "d_harj_no_240222": r["d_harj_no_240222"], "d_b2": r["qlike_all"] - base, "status": STATUS_T})
    return pd.DataFrame(out)[COLS]


def _feat_name(cfg: str) -> str:
    m = re.search(r"add=\(([^)]*)\)", cfg)
    adds = sorted({a.strip(" '").split("_")[0] for a in m.group(1).split(",") if a.strip()}) if m else []
    d = re.search(r"drop='([^']*)'", cfg)
    return "Б-2" + ("+" + "+".join(adds) if adds else "") + (" без " + d.group(1) if d and d.group(1) else "")


def features() -> pd.DataFrame:
    fa, tr, base, out = _read("FEATURE_ABLATIONS.csv"), pd.read_csv(OLD / "OPTUNA_TRIALS_B2.csv"), _base_q(), []
    cfg = dict(zip(tr["trial_id"].astype(str), tr["конфигурация"]))
    for _, r in fa.iterrows():
        if r["model"] == "BASE" or str(r["model"]).startswith("w_") or str(r["trial_id"]) not in cfg:
            continue
        out.append({"id": r["model"], "trial_id": r["trial_id"], "transferred": True, "source": "FEATURE_ABLATIONS.csv (Ж-204 B2)", "objective": "L2, ln-цель",
                    "features": _feat_name(cfg[str(r["trial_id"])]), "crit_mean": r["crit_loyo"], "crit_pen": float("nan"), "qlike_all": r["qlike_all"],
                    "qlike_2017_2021": r["qlike_p2017_2021"], "qlike_2022": r["qlike_y2022"], "qlike_2023_2025": r["qlike_p2023_2025"],
                    "qlike_2026": r["qlike_y2026"], "qlike_no_240222": r["qlike_no_20220224"], "qlike_no_2022": r["qlike_no_2022"], "mean_r": r["mean_r"],
                    "d_harj_no_240222": r["dloss_harj_no_20220224"], "d_b2": r["qlike_all"] - base, "status": STATUS_T})
    return pd.concat([pd.DataFrame(out)[COLS], c_rows(base)], ignore_index=True)


def verify(df: pd.DataFrame) -> list:
    """Каждая перенесённая строка: transferred=True, trial_id есть в источниках Ж-204 и qlike_all / qlike_2022 равны исходным (по trial_id). -> список нарушений."""
    src = pd.concat([_read("OBJECTIVE_EXPERIMENTS.csv")[["trial_id", "qlike_all", "qlike_2022"]],
                     _read("FEATURE_ABLATIONS.csv").rename(columns={"qlike_y2022": "qlike_2022"})[["trial_id", "qlike_all", "qlike_2022"]],
                     _read("C_ALL.csv")[["trial_id", "qlike_all", "qlike_2022"]]]).dropna(subset=["trial_id"]).drop_duplicates("trial_id").set_index("trial_id")
    bad = []
    for _, r in df.iterrows():
        if not r["transferred"]:
            bad.append("%s: transferred=false в таблице переноса" % r["id"])
        elif r["trial_id"] not in src.index:
            bad.append("%s: trial_id %s нет в Ж-204" % (r["id"], r["trial_id"]))
        elif not (np.isclose(src.loc[r["trial_id"], "qlike_all"], r["qlike_all"], rtol=0, atol=1e-15) and
                  np.isclose(src.loc[r["trial_id"], "qlike_2022"], r["qlike_2022"], rtol=0, atol=1e-15)):
            bad.append("%s: число не равно исходному по trial_id" % r["id"])
    return bad


C_IDS = {"c1_f4_range": ("Б-2+F4 range (одиночный)", "L2, ln-цель"), "c1_f4_park": ("Б-2+F4 park (одиночный)", "L2, ln-цель"),
         "c1_f4_f1": ("Б-2+F4+F1", "L2, ln-цель"), "c3_f3calm_rv": ("смесь двух XGBoost: Б-2+F4 / Б-2+F4+F3, F3 в калме (защита в стрессе)", "смесь XGB+XGB"),
         "c3_f3calm_g4": ("смесь двух XGBoost: Б-2+F4 / Б-2+F4+F3, F3 в калме (защита Г4)", "смесь XGB+XGB")}


def c_rows(base: float) -> pd.DataFrame:
    ca, out = _read("C_ALL.csv"), []
    for rid, (feat, obj) in C_IDS.items():
        r = ca[ca["row_id"] == rid].iloc[0]
        out.append({"id": rid, "trial_id": r["trial_id"], "transferred": True, "source": "C_ALL.csv (Ж-204 C)", "objective": obj, "features": feat,
                    "crit_mean": r["q_loyo"], "crit_pen": float("nan"), "qlike_all": r["qlike_all"],   # смеси: LOYO нет (NaN), q_wf — не сопоставим
                    "qlike_2017_2021": r["qlike_2017_2021"], "qlike_2022": r["qlike_2022"], "qlike_2023_2025": r["qlike_2023_2025"],
                    "qlike_2026": r["qlike_2026"], "qlike_no_240222": r["qlike_no_240222"], "qlike_no_2022": r["qlike_no_2022"], "mean_r": r["mean_r"],
                    "d_harj_no_240222": r["d_harj_no_240222"], "d_b2": r["qlike_all"] - base, "status": STATUS_T})
    return pd.DataFrame(out)[COLS]
