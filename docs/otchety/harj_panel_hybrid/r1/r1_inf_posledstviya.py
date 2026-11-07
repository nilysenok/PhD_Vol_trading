"""Р-1, находка 2, последствия: что видят K2/K3 и z-столбец A4 вокруг 26.01.2024 (только чтение `k23_<T>.parquet`, `hyb_a4_feats.parquet`).

Запуск (из корня): python3 -W ignore docs/otchety/harj_panel_hybrid/r1/r1_inf_posledstviya.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from models_v2.s10 import hp_paths as P  # noqa: E402

D0, D1 = pd.Timestamp("2024-01-22"), pd.Timestamp("2024-02-02")

if __name__ == "__main__":
    have = sorted(p.name for p in P.OUT.glob("k23_*.parquet"))
    print("файлы K2/K3 по бумагам:", have)
    for t in ("SBER", "AFLT", "ALRS", "GCHE", "RGSS"):
        f = P.OUT / ("k23_%s.parquet" % t)
        if not f.exists():
            print(t, ": k23 не посчитан (K2/K3 RGSS не считался до этого листа)")
            continue
        k = pd.read_parquet(f)
        c = k[["sigma2_k2", "sigma2_k3"]].to_numpy(float)
        print(t, ": строк", len(k), "inf в K2/K3:", int(np.isinf(c).sum()), "NaN:", int(np.isnan(c).sum()))
    a4 = pd.read_parquet(P.OUT / "hyb_a4_feats.parquet")
    z = a4[(a4["ticker"] == "RGSS") & (a4["date"] >= D0) & (a4["date"] <= D1)]
    print("A4 z_a4_4 RGSS вокруг даты:")
    print(z[["date", "z_a4_4"]].to_string())
    print("inf в кэше A4 (все числовые столбцы):", int(np.isinf(a4.select_dtypes("number")).sum().sum()))
