#!/usr/bin/env python3
"""Д12 h=1: сводка правила выбора по кандидатам (разработочный набор, шаги 2-6).
Дизайн кандидата = <нога>|V0_XGB или V0_LGB|<s²>|L0 на его прогнозах (сам кандидат — одна модель; схемы смешивания с HAR-J — справка).
Рейтинг: число нарушенных критериев 1-4 (gibrid_v17, эталон — модели с той же s²), затем средний QLIKE.
Tie-break владельца: лучший гибрид берётся, если отстаёт от лучшего V0 не больше чем на 1%.
python3 scripts/analysis/d12_h1_select.py V0 A_d812 B_b0.25 ..."""
import sys
from pathlib import Path
import pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
rows = []
for c in sys.argv[1:]:
    p = R / 'D12_h1_gibrid' / c / f'ranking_D12_h1_{c}_h1.csv'
    if not p.exists():
        print('нет', p); continue
    F = pd.read_csv(p)
    F['cand'] = c
    F['rank_in_cand'] = range(1, len(F) + 1)
    F['single_model'] = F.design.str.contains(r'\|V0_(?:XGB|LGB)\|')
    rows.append(F)
A = pd.concat(rows, ignore_index=True)
S = A[A.single_model].sort_values(['n_failed', 'mean_q']).reset_index(drop=True)
best = S.groupby('cand', sort=False).head(1)
pd.set_option('display.width', 250)
print('Лучший дизайн-одиночка каждого кандидата по правилу:')
print(best[['cand', 'design', 'failed', 'n_failed', 'mean_q', 'rank_in_cand']].to_string(index=False))
print('\nЛучший дизайн каждого кандидата по всему пространству гибрида (справка):')
print(A.sort_values(['n_failed', 'mean_q']).groupby('cand', sort=False).head(1)[['cand', 'design', 'failed', 'mean_q']].to_string(index=False))
if 'V0' in set(best.cand):
    v0 = best[best.cand == 'V0'].iloc[0]
    hy = best[~best.cand.str.startswith('V0')]
    print(f'\nV0: {v0.design} q={v0.mean_q:.5f} нарушено {v0.n_failed}')
    for _, r in hy.iterrows():
        print(f'{r.cand}: {r.design} q={r.mean_q:.5f} ({r.mean_q / v0.mean_q - 1:+.2%} к V0), нарушено {r.n_failed}')
best.to_csv(R / 'D12_h1_gibrid' / 'select_summary.csv', index=False)
