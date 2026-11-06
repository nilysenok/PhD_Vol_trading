"""Контроль R6 для XGBoost Б-2: seed 43–46 тем же кодом (engine_d, lib=xgb); seed 42 — сохранённый Б-2. Контроль, не trial."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common
from models_v2.s5.d import engine_d
from models_v2.s6 import engine6, lgb_b2, paths6

XGB = paths6.OUT / "xgboost"
SEEDS = lgb_b2.SEEDS


def _q(rv, s2) -> float:
    return float(np.mean(metrics.qlike(rv, s2)))


def stage(a) -> int:
    paths6.mkdirs()
    L = paths6.world()
    w = lgb_b2.mk_world(L)
    base = pd.read_parquet(paths6.REP / "xgboost" / "PREDICTIONS.parquet")
    rows = [{"seed": 42, "qlike_all": _q(base["RV_actual"], base["XGB_forecast"]), "trees_mean": json.loads((common.DEEP / "b2" / "st_BASE.json").read_text())["trees"],
             "source": "сохранённый Б-2 fx_b2_g132_b"}]
    with engine_d.pool(w, engine6.PROCS) as p:
        for s in SEEDS[1:]:
            fc, st = engine_d.daily(p, w, "b2", L["i2"], s, "xgb")
            fc.to_parquet(XGB / ("fc_xgb_b2_s%d.parquet" % s))
            rows.append({"seed": s, "qlike_all": _q(fc["rv"], fc["sigma2"]), "trees_mean": st["trees"], "source": "пересчёт тем же кодом (контроль)"})
            print("seed", s, rows[-1]["qlike_all"], flush=True)
    df = pd.DataFrame(rows)
    paths6.write_csv(df, paths6.REP / "xgboost" / "XGB_SEEDS.csv", "XGBoost Б-2, 5 seed 42–46, контроль R6, не выбор")
    print(df.to_string(), "\nmean %.5f std %.5f min %.5f max %.5f" % (df.qlike_all.mean(), df.qlike_all.std(ddof=1), df.qlike_all.min(), df.qlike_all.max()))
    return 0
