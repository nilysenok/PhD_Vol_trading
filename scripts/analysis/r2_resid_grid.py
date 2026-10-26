#!/usr/bin/env python3
"""R2 протокол v3, агент B: сетка остаточного ученика (кластер 4, фаза D, boost.grid = resid_small_v1).
LHS-60: scipy.stats.qmc.LatinHypercube(d=7, seed=20261026), уровень = floor(u·k) по уровням models/designs/r2/r2_B_roli.yaml (lhs.levels),
порядок осей: num_leaves, max_depth, min_data_in_leaf, learning_rate, lambda_l2, feature_fraction, bagging_fraction.
Точки 1–60 — LHS, точка 61 — текущая точка ячейки (контроль; r2_roli.points). Список коммитится до фазы D: models/designs/r2/r2_B_lhs60.csv (+ SHA-256 в журнале).
Счёт точки — r2_roli.build(design, ctx) с design.boost.grid = resid_small_v1, early_stop = qlike_total и ctx['_lhs_point'] = i
(❓ для S/O: ключ схемы boost.grid_point ∈ 1..61, чтобы точка входила в хэш дизайна; до schema_v2 — ctx). ≤ 300 деревьев, patience 30.
Отсечение 60 → 30 по val 2016 (шаг 1), 30 → 15 → 8 по R̄ — у O (протокол v3 п. 4).
python3 scripts/analysis/r2_resid_grid.py lhs              # записать список (проверка: повтор даёт тот же SHA)
python3 scripts/analysis/r2_resid_grid.py smoke [--h 5]    # дым: точки 1, 2, 61 на 3 тикерах, шаги 1–3"""
import argparse, hashlib, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, yaml

YB = yaml.safe_load(open(REPO / 'models' / 'designs' / 'r2' / 'r2_B_roli.yaml'))
AXES = ('num_leaves', 'max_depth', 'min_data_in_leaf', 'learning_rate', 'lambda_l2', 'feature_fraction', 'bagging_fraction')


def lhs_table():
    from scipy.stats import qmc
    c = YB['lhs']
    lv = c['levels']
    u = qmc.LatinHypercube(d=len(AXES), seed=int(c['seed'])).random(int(c['n']))
    rows = []
    for i, r in enumerate(u, 1):
        row = {'point': i}
        for j, ax in enumerate(AXES):
            k = len(lv[ax])
            row[ax] = lv[ax][min(int(np.floor(r[j] * k)), k - 1)]
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_lhs(a):
    T = lhs_table()
    p = REPO / YB['lhs']['file']
    T.to_csv(p, index=False)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f'{p}: {len(T)} точек, SHA-256 {sha}; уникальных комбинаций {len(T.drop(columns="point").drop_duplicates())}')
    for ax in AXES:
        print(f'  {ax}: ' + ', '.join(f'{v}×{n}' for v, n in T[ax].value_counts().sort_index().items()))


def cmd_smoke(a):
    import r2_roli as B
    rows = []
    for h in a.h or YB['smoke']['hs']:
        ctx = B.smoke_ctx(h)
        for pt in (1, 2, 61):
            ctx['_lhs_point'] = pt
            t0 = time.time()
            o = B.build(dict(design='grid_smoke', h=h, seed=42, boost=dict(role='residual', grid='resid_small_v1', early_stop='qlike_total',
                                                                          lib='lightgbm'), data={}), ctx, steps=YB['smoke']['steps'])
            rows.append(dict(h=h, point=pt, rows=len(o), sec=round(time.time() - t0, 1)))
            print(time.strftime('%T'), f'сетка h={h} точка {pt}: строк {len(o)}, {time.time() - t0:.1f} с', flush=True)
    p = B.R / 'R2' / 'B' / 'smoke_grid_B.csv'
    pd.DataFrame(rows).to_csv(p, index=False)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('lhs')
    p = sp.add_parser('smoke'); p.add_argument('--h', type=int, nargs='*')
    a = ap.parse_args()
    {'lhs': cmd_lhs, 'smoke': cmd_smoke}[a.cmd](a)


if __name__ == '__main__':
    main()
