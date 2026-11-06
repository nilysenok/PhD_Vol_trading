"""Хранилище рядов σ̂² волны E: компоненты (`comp`) + ряды стадий a / b / c (parquet в `data/regen/xgb_lgbm_full_v1/e/`); все на сетке 2271 дат, σ̂² > 0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6 import paths6
from models_v2.s6.e import comp, crit_e, pe

STAGES = ("a", "b", "c")
BASE = ["d", "target_date", "rv"]


def file(stage: str):
    return pe.OUT / ("series_%s.parquet" % stage)


def save(stage: str, base: pd.DataFrame, series: dict) -> None:
    """Дописать ряды стадии (существующие столбцы перезаписываются по имени); проверка σ̂² > 0 и конечности."""
    for k, v in series.items():
        v = np.asarray(v, float)
        if len(v) != paths6.N_GRID or not (np.isfinite(v).all() and (v > 0).all()):
            raise ValueError("%s: ряд не 2271 дата или σ̂² не положительна/конечна" % k)
    cur = pd.read_parquet(file(stage)) if file(stage).exists() else base[BASE].copy()
    for k, v in series.items():
        cur[k] = np.asarray(v, float)
    cur.to_parquet(file(stage))


def load(stages=STAGES) -> pd.DataFrame:
    """Компоненты (seed 42) + ряды стадий; столбцы seed 43–46 компонент — только через `comp.col(df, name, seed)`."""
    df = comp.load()
    keep = BASE + ["HAR", "XJ", "XF", "LF", "B2X", "B2L", "st_g4", "st_rv"]
    out = df[keep].copy()
    for s in stages:
        if file(s).exists():
            x = pd.read_parquet(file(s))
            for c in x.columns:
                if c not in BASE:
                    out[c] = x[c].to_numpy()
    return out


def series(df: pd.DataFrame) -> dict:
    return {c: df[c].to_numpy(float) for c in df.columns if c not in BASE + ["st_g4", "st_rv"]}


def qloss(df: pd.DataFrame, name: str) -> np.ndarray:
    return crit_e.qlike_series(df["rv"].to_numpy(float), df[name].to_numpy(float))


def crit(df: pd.DataFrame, name: str) -> float:
    return crit_e.cw(qloss(df, name), df["target_date"])
