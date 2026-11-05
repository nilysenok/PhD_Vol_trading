"""Единственный подбор гиперпараметров (Б5): обучение — цели <= 2015-12-31, проверка — цели 2016 по QLIKE (б).

Сетка объявлена в GATES.md до подбора: глубина {3, 4}; δ Хьюбера {0.5, 1.0}; подбор на nohar/free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.metrics import qlike
from models_v2.s2 import boost
from models_v2.s2 import features as ft

DEPTHS, DELTAS = (3, 4), (0.5, 1.0)
TRAIN_END = np.datetime64("2015-12-31")


def candidates() -> list:
    c = []
    for lib in ("xgb", "lgbm"):
        c += [(lib, "mse", d, 0.0) for d in DEPTHS]
        c += [(lib, "hub", d, dl) for d in DEPTHS for dl in DELTAS]
    return c


def tune_setup(F: ft.Fp):
    """(T, d_idx): последняя строка <= 2015-12-31 и строки d с целью в 2016 году."""
    T = int(np.flatnonzero(F.dates <= TRAIN_END)[-1])
    n = len(F.dates)
    d = np.arange(T, n - 1)
    yr = F.dates[d + 1].astype("datetime64[Y]") == np.datetime64("2016", "Y")
    ok = yr & (F.seg[d] == F.seg[d + 1]) & np.isfinite(F.X[d][:, list(ft.CORE)]).all(axis=1)
    return T, d[ok]


def score(F: ft.Fp, d_idx, res: dict) -> float:
    s2 = np.exp(res["f"] + res["s2_es"] / 2)
    return float(qlike(F.rv[d_idx + 1], s2).mean())


def choose(rows: pd.DataFrame) -> dict:
    """{(lib, loss): (depth, delta)}: min QLIKE (б), ничья — меньшая глубина, затем меньшее δ."""
    out = {}
    for (lib, loss), g in rows.groupby(["lib", "loss"]):
        b = g.sort_values(["qlike_b", "depth", "delta"]).iloc[0]
        out[(lib, loss)] = (int(b["depth"]), float(b["delta"]))
    return out
