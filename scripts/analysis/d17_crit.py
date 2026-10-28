#!/usr/bin/env python3
"""Д17.1: сводка критериев гибрида (gibrid_v17 --dev-eval oos, эталон same) — лучший дизайн по правилу и одиночные V0-дизайны:
нарушенные c1-c4, средний QLIKE шагов 2-5 (val 2017-2020) и шага 6 (2021) по q_hyb из designs_val.
python3 scripts/analysis/d17_crit.py --h 5 --boost D17_1_h5_G5X [--dir data/regen/D17/gibrid_oos]"""
import argparse
from pathlib import Path
import pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')


def summary(h, boost, d):
    p = Path(d) / boost
    rk = pd.read_csv(p / f'ranking_{boost}_h{h}.csv')
    dv = pd.read_csv(p / f'designs_val_{boost}_h{h}.csv')
    q = dv[dv.step.between(2, 6)].pivot_table(index='design', columns='step', values='q_hyb')
    rk = rk.merge(q.reset_index(), on='design', how='left')
    rk['q2_5'] = rk[[2, 3, 4, 5]].mean(axis=1)
    rk['q2021'] = rk[6]
    rk['crit'] = rk.apply(lambda r: ''.join(c[1] if bool(r[c]) else '-' for c in ('c1', 'c2', 'c3', 'c4')), axis=1)
    rk['rank'] = range(1, len(rk) + 1)
    return rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--boost', required=True)
    ap.add_argument('--dir', default=str(R / 'D17' / 'gibrid_oos'))
    a = ap.parse_args()
    rk = summary(a.h, a.boost, a.dir)
    cols = ['rank', 'design', 'crit', 'n_failed', 'mean_q', 'q2_5', 'q2021']
    pd.set_option('display.width', 200)
    print(f'{a.boost} h={a.h}: лучший по правилу')
    print(rk.head(3)[cols].to_string(index=False, float_format='%.4f'))
    print(rk[rk.design.str.match(r'^(single|top5|seeds3)\|V0_(XGB|LGB)\|')].head(6)[cols].to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
