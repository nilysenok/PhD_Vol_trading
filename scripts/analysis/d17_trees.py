#!/usr/bin/env python3
"""Д17.1: число деревьев (best_iteration, seed 42, шаги 1-6) на h=22 до/после — D12_h22 против вариантов gamma с офсетом."""
from pathlib import Path
import pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
rows = []
for name, f, kind in (('D12_h22 (V9 + linear_tree, MSE на ln; optuna-30)', R / 'D12_h22_selected_h22.csv', 'sel'),
                      ('G22 (gamma + офсет HAR-J, точки D12_h22)', R / 'D17/h22/G22_meta.csv', 'meta'),
                      ('G22n (то же без linear_tree)', R / 'D17/h22/G22n_meta.csv', 'meta'),
                      ('G22X (gamma + офсет HAR-X, точки D12_h22)', R / 'D17/h22/G22X_meta.csv', 'meta'),
                      ('G22opt (gamma + HAR-J, optuna-30)', R / 'D17_1_h22_G22opt_selected_h22.csv', 'sel'),
                      ('G22Xopt (gamma + HAR-X, optuna-30)', R / 'D17_1_h22_G22Xopt_selected_h22.csv', 'sel')):
    d = pd.read_csv(f)
    d = d[d.step <= 6]
    if kind == 'meta':
        d = d[(d.seed == 42) & (d['rank'] == 1)]
    for m, g in d.groupby('model'):
        g = g.sort_values('step')
        rows.append(dict(variant=name, model=m, **{f's{int(s)}': int(b) for s, b in zip(g.step, g.best_iteration)}, mean=g.best_iteration.mean()))
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format='%.1f'))
T.to_csv(R / 'D17' / 'trees_h22.csv', index=False)
