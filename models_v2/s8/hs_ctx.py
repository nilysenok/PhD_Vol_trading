"""Ж-212: общий контекст счёта B–G: мир потока XGB (F, сетка, Б-2+F4, I2, eps), PIT HAR-J на всех строках F (lb, beta; кэш), замороженные компоненты, ML на строках F."""
from __future__ import annotations

import numpy as np

from models_v2.s6.e import comp
from models_v2.s6.xgb import worldx
from models_v2.s8 import hs_feat, hs_paths as P

HAR_PIT = P.OUT / "har_pit.npz"


def har_cached(F) -> tuple:
    if HAR_PIT.exists():
        z = np.load(HAR_PIT)
        if len(z["lb"]) == len(F.rv):
            return z["lb"], z["beta"]
    lb, beta = hs_feat.har_pit(F)
    P.OUT.mkdir(parents=True, exist_ok=True)
    np.savez(HAR_PIT, lb=lb, beta=beta)
    return lb, beta


def ml_rows(F, grid, df) -> dict:
    n = len(F.rv)
    return {"XGB": hs_feat.ml_on_rows(n, grid, df["XJ"]), "LGBM": hs_feat.ml_on_rows(n, grid, df["LF"])}


def build() -> dict:
    w = worldx.load()
    df = comp.load()
    lb, beta = har_cached(w["F"])
    if not (df["d"].to_numpy("datetime64[ns]") == w["F"].dates[w["grid"]].astype("datetime64[ns]")).all():
        raise SystemExit("стоп: сетка comp != сетка мира")
    return {"w": w, "df": df, "lb": lb, "beta": beta, "ml": ml_rows(w["F"], w["grid"], df)}


def extra_for(F, lb, beta, ml, fam: str, blocks: tuple, extra: int = 0) -> tuple:
    """Матрица добавок (n×k) и имена: блоки по порядку; ML для Bg — свой самостоятельный ряд модели (XGB — XJ, LGBM — LF)."""
    bl = hs_feat.blocks(F, lb, beta, ml[fam], extra)
    cols = np.column_stack([bl[b][0] for b in blocks])
    return cols, [n for b in blocks for n in bl[b][1]]
