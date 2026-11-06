"""Компоненты волны E (GATES_E R1, G3): сохранённые ряды σ̂² финальных моделей и baseline, 5 seed; собираются один раз в components.parquet (без обучения)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s5.c import mix
from models_v2.s6 import paths6
from models_v2.s6.e import pe
from models_v2.s6.xgb import c_calc, c_space, cfgx, worldx

SEEDS = (42, 43, 44, 45, 46)
FILE = pe.OUT / "components.parquet"
LDIR, XDIR = paths6.OUT / "lightgbm", paths6.OUT / "xgboost"
NAMES = {"XJ": "x2_best_j (XGB final; потолок 2017–21)", "XF": "ctrl_f4 (XGB Б-2+F4, чистая)", "LF": "C0_inh_F4 (LGBM final)",
         "B2X": "Б-2 (baseline XGB)", "B2L": "LGBM-аналог Б-2 (baseline LGBM)", "HAR": "HAR-J(б)"}


def x2_moves() -> tuple:
    d = json.loads((cfgx.XGB_REP / "X2_DECISION.json").read_text())
    return d["best_trial_id"], d["best_moves"]


def xgb_series(w: dict, moves: dict, seed: int, tid: str) -> np.ndarray:
    raw = pd.read_parquet(c_calc.DATA / ("raw_%s_s%d.parquet" % (tid, seed)))
    return worldx.sigma_frame(w, c_calc.spec_for(w, "l2"), raw)["sigma2"].to_numpy(float)


def lgbm_path(name: str, seed: int):
    if name == "LF":
        return LDIR / "c" / ("fc_C0_inh_F4.parquet" if seed == 42 else "fc_N_C0_inh_F4_s%d.parquet" % seed)
    return LDIR / ("fc_lgbm_b2.parquet" if seed == 42 else "fc_lgbm_b2_s%d.parquet" % seed)


def build(w: dict) -> pd.DataFrame:
    ctrl = w["ctrl"]
    df = ctrl[["d", "target_date", "rv"]].copy()
    df["HAR"] = ctrl["s2_harj"].to_numpy(float)
    x2_tid, x2_mv = x2_moves()
    if c_space.trial_id("l2", x2_mv) != x2_tid:
        raise SystemExit("стоп: trial_id x2_best_j не совпал с X2_DECISION")
    for s in SEEDS:
        sfx = "" if s == 42 else "_s%d" % s
        df["XJ" + sfx] = xgb_series(w, x2_mv, s, x2_tid)
        df["XF" + sfx] = xgb_series(w, {}, s, c_space.trial_id("l2", {}))
        df["B2X" + sfx] = ctrl["s2_b2"].to_numpy(float) if s == 42 else pd.read_parquet(XDIR / ("fc_xgb_b2_s%d.parquet" % s))["sigma2"].to_numpy(float)
        for nm in ("LF", "B2L"):
            fc = pd.read_parquet(lgbm_path(nm, s))
            if not (fc["d"].to_numpy() == ctrl["d"].to_numpy()).all():
                raise SystemExit("стоп: сетка %s s%d не совпала" % (nm, s))
            df[nm + sfx] = fc["sigma2"].to_numpy(float)
    c = mix.context(w["F"], w["grid"], ctrl)
    df["st_g4"], df["st_rv"] = c["st_g4"].astype(bool), c["st_rv"].astype(bool)
    return df.reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    pe.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FILE)


def load() -> pd.DataFrame:
    return pd.read_parquet(FILE)


def ctx_of(df: pd.DataFrame) -> dict:
    """Контекст правил (rv, d, td, st_g4, st_rv): всё известно на d (стресс — по прошлому)."""
    return {"rv": df["rv"].to_numpy(float), "h": df["HAR"].to_numpy(float), "d": df["d"].to_numpy("datetime64[ns]"),
            "td": df["target_date"].to_numpy("datetime64[ns]"), "st_g4": df["st_g4"].to_numpy(bool), "st_rv": df["st_rv"].to_numpy(bool)}


def col(df: pd.DataFrame, name: str, seed: int = 42) -> np.ndarray:
    return df[name if seed == 42 or name == "HAR" else "%s_s%d" % (name, seed)].to_numpy(float)


def stack(df: pd.DataFrame, names: tuple, seed: int = 42) -> np.ndarray:
    return np.column_stack([col(df, n, seed) for n in names])
