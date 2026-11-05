"""И2/И6 (Ж-200): годовой выбор по LOYO внутри доступного прошлого (PIT: цели < 01.01.Y)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2.s4 import runner
from models_v2.s4.fit import HP

TIE = 1e-12
FIRST_Y = 2015
GRID = tuple(HP(d, m, l2, lr, 0.8) for d, m, l2, lr in
             itertools.product((2, 3, 5), (10, 30), (0.5, 2.0), (0.05, 0.1)))   # 3×2×2×2 = 24


NEW = tuple(HP(d, m, l2, lr, ss, 0.5, 300) for d, m, l2, ss, lr in
            itertools.product((1, 2, 3), (30, 100, 200), (2.0, 10.0, 30.0), (0.6, 0.8), (0.02, 0.05)))   # Ж-202: 108
GRID132 = GRID + NEW
HARD = tuple(h for h in GRID132 if h.depth <= 2)   # «Б-1 жёсткий»: глубина {1, 2}


def order_key(hp: HP) -> tuple:
    """Ничья: меньше глубина, больше min_data, больше L2, меньше lr, затем colsample 0,5 раньше 1, меньший потолок, subsample."""
    return (hp.depth, -hp.min_child, -hp.l2, hp.lr, hp.colsample, hp.max_trees, hp.subsample)


def cut_row(F, Y: int) -> int:
    """Последняя строка с датой < 01.01.Y (пары с t+1 <= этой строки)."""
    return int(np.searchsorted(F.dates, np.datetime64("%d-01-01" % Y))) - 1


def pick(scores: dict, key=None):
    """scores: {метка: средний QLIKE}; ничья |Δ| <= 1e-12 решается key (по умолчанию — порядок меток)."""
    ok = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not ok:
        return None
    lo = min(ok.values())
    tied = [k for k, v in ok.items() if v - lo <= TIE]
    return min(tied, key=key) if key else tied[0]


def run_loyo(F, DZ, cands: dict, years, procs: int = 4) -> pd.DataFrame:
    """cands[Y] = {метка: (dz_key, hp, wflag)} -> строки (Y, label, score, n_folds); score — среднее по y ∈ [2015, Y−1]."""
    jobs = [((Y, lab, y), dzk, hp, cut_row(F, Y), y, w) for Y in years for lab, (dzk, hp, w) in cands[Y].items()
            for y in range(FIRST_Y, Y)]
    res = runner.map_jobs(F, DZ, jobs, "fold", procs)
    df = pd.DataFrame([{"Y": k[0], "label": k[1], "y": k[2], "q": q} for k, q in res])
    g = df.groupby(["Y", "label"], sort=False)["q"]
    out = pd.DataFrame({"score": g.mean(), "n_folds": g.count()}).reset_index()
    return out, df
