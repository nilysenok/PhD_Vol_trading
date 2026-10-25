#!/usr/bin/env python3
"""etap5, вариант B: ТОЛЬКО валидационная сводка. QLIKE и сравнения B/A/П1 — по шагам 1–6 (val 2016–2021, разработочный набор);
шаги 7–9 (val 2022–2024) скрыты: для них только параметры/best_iteration. Масштаб прогнозов — по всем годам. Тестовые QLIKE не считаются.
A (json): xgb = xgb_h*_val (параметры json, ES), lgbm = P1_h*_val (параметры json, gbdt+ES). Запуск из корня: python3 scripts/analysis/etap5_B_val.py
Выход: data/out/etap5_B_qlike.csv, etap5_B_masshtab.csv, etap5_B_stability.csv, etap5_B_ranges.csv"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; REG = ROOT / 'data/regen'; OUT = ROOT / 'data/out'; OUT.mkdir(parents=True, exist_ok=True)
DEV = range(1, 7)
qlike = lambda y, p: float(np.mean((r := y / np.clip(p, 1e-10, None)) - np.log(r) - 1))
SRC = {('xgb', 'B'): 'B_xgb', ('xgb', 'A'): 'xgb', ('lgbm', 'B'): 'B_lgbm', ('lgbm', 'A'): 'P1'}  # A lgbm = П1 (json + ES)

rows, sc = [], []
for h in (1, 5, 22):
    for (m, role), pre in SRC.items():
        for kind in ('val', 'test'):
            f = REG / f'{pre}_h{h}_{kind}.parquet'
            if not f.exists(): continue
            d = pd.read_parquet(f); name = f'{m}_{role}'
            r = d.groupby(['ticker', 'year']).apply(lambda g: g.rv_pred.median() / g.rv_actual.median())
            sc.append(dict(variant=name, h=h, kind=kind, n=len(r), out=int(((r < .5) | (r > 2)).sum()), min=r.min(), max=r.max()))
            if kind == 'val':
                for s, g in d[d.step.isin(DEV)].groupby('step'):
                    rows.append(dict(variant=name, h=h, step=int(s), val_year=int(g.year.iloc[0]), qlike_val=qlike(g.rv_actual.values, g.rv_pred.values)))
q = pd.DataFrame(rows); q.to_csv(OUT / 'etap5_B_qlike.csv', index=False)
pd.DataFrame(sc).to_csv(OUT / 'etap5_B_masshtab.csv', index=False)
pv = q.pivot_table(index=['h', 'variant'], columns='step', values='qlike_val'); pv['mean'] = pv.mean(axis=1)
print('Валидационный QLIKE, шаги 1–6 (val 2016–2021)'); print(pv.round(4).to_string())
print('\nB относительно A, шаги 1–6 (% изменения среднего QLIKE; <0 — B лучше)')
for h in (1, 5, 22):
    for m in ('xgb', 'lgbm'):
        a, b = pv.loc[(h, f'{m}_A'), 'mean'], pv.loc[(h, f'{m}_B'), 'mean']
        w = (pv.loc[(h, f'{m}_B'), list(DEV)] < pv.loc[(h, f'{m}_A'), list(DEV)]).sum()
        print(f'h{h} {m}: A {a:.4f} B {b:.4f} ({100*(b/a-1):+.1f}%), B лучше на {w}/6 шагах')
print('\nМасштаб прогнозов, тикер-годов вне [0,5; 2] (все годы)'); print(pd.DataFrame(sc).round(2).to_string(index=False))

sel = pd.read_csv(REG / 'B_selected.csv'); tr = pd.read_csv(REG / 'B_trials.csv').drop_duplicates(['model', 'h', 'step', 'point'])
st = []
for (m, h), g in sel.groupby(['model', 'h']):
    t = tr[(tr.model == m) & (tr.h == h) & tr.step.isin(DEV)]
    gap = t.groupby('step').apply(lambda x: (lambda s: s.iloc[4] / s.iloc[0] - 1)(x.qlike_val.sort_values())).mean()  # top-5 vs top-1, только шаги 1–6
    st.append(dict(model=m, h=h, distinct_points=g.point.nunique(), steps=len(g), median_best_iter=g.best_iteration.median(), top5_vs_top1_qlike_pct_dev=100 * gap))
pd.DataFrame(st).to_csv(OUT / 'etap5_B_stability.csv', index=False); print('\nСтабильность выбора'); print(pd.DataFrame(st).round(2).to_string(index=False))
par = [c for c in sel.columns if c not in ('model', 'h', 'step', 'point', 'best_iteration', 'qlike_val')]
rg = sel.groupby('model')[par + ['best_iteration']].agg(['min', 'median', 'max']).T
rg.to_csv(OUT / 'etap5_B_ranges.csv'); print('\nФактические диапазоны выбранных параметров (шаги 1–9)'); print(rg.round(4).to_string())
