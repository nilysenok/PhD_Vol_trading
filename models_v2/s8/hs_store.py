"""Ж-212: хранилище рядов σ̂² направления (parquet: d, target_date, rv + по столбцу на конфигурацию) и запись таблиц направления."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.e import comp
from models_v2.s8 import hs_paths as P


def path(direction: str):
    return P.OUT / ("%s_series.parquet" % direction)


def load(direction: str) -> pd.DataFrame:
    p = path(direction)
    return pd.read_parquet(p) if p.exists() else comp.load()[["d", "target_date", "rv"]].copy()


def put(direction: str, name: str, s2, positive: bool = True) -> None:
    s2 = np.asarray(s2, float)
    if not (np.isfinite(s2).all() and ((s2 > 0).all() if positive else True)):
        raise ValueError("%s: ряд не конечен%s" % (name, "/неположителен" if positive else ""))
    df = load(direction)
    df[name] = s2
    P.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path(direction))


def get(direction: str, name: str):
    df = load(direction)
    return df[name].to_numpy(float) if name in df.columns else None


def frozen() -> pd.DataFrame:
    """d, target_date, rv, HAR, XJ, LF (замороженные)."""
    return comp.load()[["d", "target_date", "rv", "HAR", "XJ", "LF"]].copy()
