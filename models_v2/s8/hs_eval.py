"""Ж-212: оценка строки направления (QLIKE, crit_wf, QLIKE по годам, Δ к контролю, знак по периодам, DM) и правило HS11 «продолжать направление».
Δ = QLIKE(строка) − QLIKE(контроль), минус — строка лучше. crit_wf — `d_comp.crit_wf` (причинный критерий)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5.b1 import evalrows
from models_v2.s6.e import crit_e

YEARS = tuple(range(2017, 2027))
PERIODS = {"G_2019_2021": (2019, 2021), "P2_2022": (2022, 2022), "P3_2023_2025": (2023, 2025)}
CRIT_STEP = 0.0005
MIN_YEARS = 6


def years_of(df: pd.DataFrame) -> np.ndarray:
    return pd.Series(df["target_date"]).dt.year.to_numpy()


def period_mask(df: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    y = years_of(df)
    return (y >= lo) & (y <= hi)


def row(df: pd.DataFrame, name: str, s2, ctrl, extra: dict | None = None) -> dict:
    """Строка таблицы: name, qlike_all, crit_wf, q_<год>, d_crit/d_q к контролю, wins (лет лучше контроля из 10), знак по трём периодам, ex 24.02, DM."""
    s2, ctrl = np.asarray(s2, float), np.asarray(ctrl, float)
    r = crit_e.describe(name, s2, df, {"ctrl": ctrl})
    q, qc = np.asarray(metrics.qlike(df["rv"], s2), float), np.asarray(metrics.qlike(df["rv"], ctrl), float)
    y = years_of(df)
    out = {"model": name, "qlike_all": r["qlike_all"], "crit_wf": r["crit_wf"], "crit_wf_ctrl": crit_e.cw(qc, df["target_date"])}
    out["d_crit"] = out["crit_wf"] - out["crit_wf_ctrl"]
    out["d_qlike"], out["d_qlike_no_240222"], out["p_dm"] = r["d_ctrl"], r["d_ctrl_no_240222"], r["p_ctrl"]
    for yy in YEARS:
        out["q_%d" % yy] = float(q[y == yy].mean())
    out["years_better"] = int(sum(q[y == yy].mean() < qc[y == yy].mean() for yy in YEARS))
    for k, (lo, hi) in PERIODS.items():
        m = period_mask(df, lo, hi)
        out["d_" + k] = float((q - qc)[m].mean())
    out["sign_ok_2of3"] = bool(sum(out["d_" + k] < 0 for k in PERIODS) >= 2)
    out.update(extra or {})
    return out


def robust_gain(r: dict) -> bool:
    """Блок «полезен»: Δcrit_wf <= −0,0005 и Δ QLIKE < 0 на всей истории и без 24.02.2022."""
    return bool(r["d_crit"] <= -CRIT_STEP and r["d_qlike"] < 0 and r["d_qlike_no_240222"] < 0)


def hs11(rows: list) -> dict:
    """HS11: лучшая (мин. d_crit) строка исполненного набора продолжает направление, если d_crit <= −0,0005 и лет лучше контроля >= 6 из 10."""
    best = min(rows, key=lambda r: r["d_crit"])
    ok = bool(best["d_crit"] <= -CRIT_STEP and best["years_better"] >= MIN_YEARS)
    return {"continue": ok, "best": best["model"], "d_crit": best["d_crit"], "years_better": best["years_better"],
            "rule": "d_crit <= -0.0005 и лет лучше контроля >= 6 из 10"}


def mask_no_feb24(df: pd.DataFrame) -> np.ndarray:
    return evalrows.masks(df["target_date"])["no_240222"]
