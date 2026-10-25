#!/usr/bin/env python3
"""etap0: QLIKE и DM на сохранённых прогнозах СВ 2017-2026 (только чтение).

QLIKE = mean(y/p - ln(y/p) - 1), пул по тикерам, rv_actual > 0,
28.02-24.03.2022 исключены. DM: простой, лаг max(1, h-1), как в
moex_volatility/scripts/walk_forward_adaptive_v3.py:28.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2] / "moex_volatility"
WF = ROOT / "results/final/data/predictions_walkforward"
HALT = ("2022-02-28", "2022-03-24")


def load(name, h):
    d = pd.read_parquet(WF / f"{name}_h{h}_annual.parquet")
    d = d[(d.rv_actual > 0) & ~d.date.between(*HALT)]
    return d.set_index(["date", "ticker"])


def loss(d):
    p = d.rv_pred.clip(lower=1e-10)
    return d.rv_actual / p - np.log(d.rv_actual / p) - 1


def dm(l1, l2, h):
    d = (l1 - l2).dropna().sort_index(level="date").values
    n, lag = len(d), max(1, h - 1)
    g = np.var(d, ddof=1)
    for k in range(1, lag + 1):
        g += 2 * (1 - k / (lag + 1)) * np.cov(d[k:], d[:-k])[0, 1]
    t = d.mean() / np.sqrt(g / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def main():
    names = ["har", "hybrid_adaptive", "hybrid_best", "hybrid_v2"]
    for h in (1, 5, 22):
        ls = {n: loss(load(n, h)) for n in names}
        row = {n: f"{ls[n].mean():.4f} (n={len(ls[n])})" for n in names}
        print(f"h={h} QLIKE", row)
        for n in names[1:]:
            t, p = dm(ls["har"], ls[n], h)  # t>0: n лучше HAR-J
            d = (ls["har"] - ls[n]).dropna().values
            t0 = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))  # простой DM без HAC
            print(f"  DM {n} vs har: HAC t={t:+.2f} p={p:.4f}; простой t={t0:+.2f}")
    t19 = pd.read_parquet(ROOT / "data/predictions/test_2019/hybrid_h1.parquet")
    print("test 2019 hybrid_h1 rows:", len(t19))
    print("hybrid_best h1 rows (все даты):", len(pd.read_parquet(WF / "hybrid_best_h1_annual.parquet")))


if __name__ == "__main__":
    main()
