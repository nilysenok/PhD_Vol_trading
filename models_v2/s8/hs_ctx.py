"""Ж-212: общий контекст счёта B–G: мир потока XGB (F, сетка, Б-2+F4, I2, eps), PIT HAR-J на всех строках F (lb, beta; кэш), замороженные компоненты, ML на строках F."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from models_v2 import har
from models_v2.s2 import features as ft
from models_v2.s6.e import comp
from models_v2.s6.xgb import worldx
from models_v2.s8 import hs_feat, hs_paths as P, hs_pit

HAR_PIT = P.OUT / "har_pit.npz"


CODE = (hs_feat, hs_pit, har, ft)


def har_key(F) -> str:
    """Д5: SHA-256 входов (даты, сегменты, RV, X[:, :4], min_pairs) и исходников модулей, строящих PIT HAR-J."""
    h = hashlib.sha256()
    for a in (np.asarray(F.dates, "datetime64[ns]"), np.asarray(F.seg), np.asarray(F.rv, float), np.asarray(F.X[:, :4], float)):
        h.update(str(a.shape).encode() + np.ascontiguousarray(a).tobytes())
    h.update(str(ft.MIN_PAIRS_HAR).encode())
    for m in CODE:
        h.update(Path(m.__file__).read_bytes())
    return h.hexdigest()


def har_cached(F) -> tuple:
    key = har_key(F)
    if HAR_PIT.exists():
        z = np.load(HAR_PIT)
        if "key" in z.files and str(z["key"]) == key:
            return z["lb"], z["beta"]
    lb, beta = hs_feat.har_pit(F)
    P.OUT.mkdir(parents=True, exist_ok=True)
    np.savez(HAR_PIT, lb=lb, beta=beta, key=np.array(key))
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
