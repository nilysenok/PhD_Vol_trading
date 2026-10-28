#!/usr/bin/env python3
"""Д17.1, h=1: нога top5 (среднее RV пяти точек B v2b, seed 42) — шаги 2-5 и 2021 (шаг 6); шаги 7-9 не читаются.
python3 scripts/analysis/d17_top5.py D12_h1_A_d812:sm D17_1_h1_G1X:raw D17_1_h1_G1X:sm"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L


def avg(d):
    return d.groupby(['date', 'ticker', 'step'], as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


for arg in sys.argv[1:]:
    b, code = arg.split(':')
    for m in ('xgb', 'lgbm'):
        t = avg(pd.read_parquet(L.REGEN / f'{b}_{m}_h1_top5_val.parquet').query('step <= 6'))
        q = L.evaluate(t, 2, 6)[code][1]
        print(f'{b} {m} top5 {code}: шаги 2-5 {np.mean(q[:4]):.4f}, 2021 {q[4]:.4f}; по шагам', ' / '.join(f'{x:.4f}' for x in q))
