"""Строка таблицы B1: QLIKE по периодам и срезам чувствительности, RV/σ̂², потери обучение/блок, разности и DM к HAR-J(б) и Б-2, критерии."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import dm as dmod
from models_v2.s5 import common

FEB24 = np.datetime64("2022-02-24")
YEARS = range(2017, 2026)


def masks(td: pd.Series) -> dict:
    m = common.period_masks(td)
    t = pd.Series(td).to_numpy("datetime64[ns]").astype("datetime64[D]")
    m["no_240222"] = t != FEB24
    m["no_2022"] = ~m["y2022"]
    return m


def q_wf(q: np.ndarray, td: pd.Series) -> float:
    """Walk-forward: score_Y = среднее годовых средних QLIKE по y ∈ [2017, Y−1]; Q_wf — среднее score_Y по Y ∈ 2018…2026."""
    y = pd.Series(td).dt.year.to_numpy()
    ann = {k: float(q[y == k].mean()) for k in range(2017, 2027) if (y == k).any()}
    sc = [np.mean([ann[k] for k in range(2017, Y) if k in ann]) for Y in range(2018, 2027)]
    return float(np.mean(sc))


def q_loyo(folds: pd.DataFrame) -> float:
    """folds: колонки Y, y, q. Q_loyo = среднее по Y из среднего по y ∈ [2015, Y−1] (NaN-фолды пропускаются, как в I2)."""
    return float(folds.groupby("Y")["q"].mean().mean())


def vs(rv, s2_base, s2_cfg, mask) -> dict:
    d = dmod.dm(rv[mask], s2_base[mask], s2_cfg[mask])
    return {"d": -d["mean_d"], "dm": d["stat"], "p": d["p"]}


def row(fc: pd.DataFrame, ctrl: pd.DataFrame, extra: dict | None = None) -> dict:
    """fc: d, target_date, rv, sigma2 (сетка Б-2); ctrl: s2_b2, s2_harj той же сетки. d_* = mean(QLIKE_конфиг − QLIKE_база): минус — конфиг лучше."""
    rv, s2 = fc["rv"].to_numpy(float), fc["sigma2"].to_numpy(float)
    q = metrics.qlike(rv, s2)
    mk = masks(fc["target_date"])
    out = {"qlike_all": float(q.mean()), "qlike_2017_2021": float(q[mk["p2017_2021"]].mean()), "qlike_2022": float(q[mk["y2022"]].mean()),
           "qlike_2023_2025": float(q[mk["p2023_2025"]].mean()), "qlike_2026": float(q[mk["y2026"]].mean()),
           "qlike_no_240222": float(q[mk["no_240222"]].mean()), "qlike_no_2022": float(q[mk["no_2022"]].mean()),
           "mean_r": float((rv / s2).mean()), "q_wf": q_wf(q, fc["target_date"])}
    for name, col in (("harj", "s2_harj"), ("b2", "s2_b2")):
        base = ctrl[col].to_numpy(float)
        for lab, key in (("", "all"), ("_no_240222", "no_240222")):
            r = vs(rv, base, s2, mk[key])
            out.update({"d_%s%s" % (name, lab): r["d"], "dm_%s%s" % (name, lab): r["dm"], "p_%s%s" % (name, lab): r["p"]})
    yr, qh = fc["target_date"].dt.year.to_numpy(), metrics.qlike(rv, ctrl["s2_harj"].to_numpy(float))
    out["share_years_win_harj"] = float(np.mean([q[yr == k].mean() < qh[yr == k].mean() for k in YEARS]))
    out.update(extra or {})
    return out


def perspective(q_int: float, q_int_b2: float, d_harj_no_240222: float) -> bool:
    """Правило GATES_B1: Q_int строго лучше Б-2 (> 1e-12) И разность к HAR-J(б) «без 24.02.2022» <= 0."""
    return bool(q_int_b2 - q_int > 1e-12 and d_harj_no_240222 <= 0)
