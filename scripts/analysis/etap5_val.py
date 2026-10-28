#!/usr/bin/env python3
"""etap5: сравнение вариантов бустинга. По умолчанию --scope val: QLIKE ТОЛЬКО валидационных лет шагов (пул) + масштаб.
--scope test требует docs/otchety/FINAL_DESIGN.md (слепой тест). Запуск из корня: python3 scripts/analysis/etap5_val.py [--scope val]
Выход: data/out/etap5_val_qlike.csv, data/out/etap5_val_masshtab.csv"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument('--scope', default='val', choices=['val', 'test'])
scope = ap.parse_args().scope
if scope == 'test' and not (ROOT / 'docs/otchety/FINAL_DESIGN.md').exists():
    sys.exit('ОШИБКА: нет docs/otchety/FINAL_DESIGN.md — тестовые метрики запрещены (слепой тест).')
if scope == 'test':
    sys.exit('Тестовый режим не реализован в этом скрипте (метрики на тесте считаются после FINAL_DESIGN).')

REG = ROOT / 'data/regen'; OLD = ROOT / 'moex_volatility/results/final/data/predictions_walkforward'
OUT = ROOT / 'data/out'; OUT.mkdir(parents=True, exist_ok=True)

def qlike(y, p):
    r = y / np.clip(p, 1e-10, None); return float(np.mean(r - np.log(r) - 1))

rows, sc = [], []
def scale(name, h, kind, d, col='rv_pred'):
    g = d.groupby(['ticker', 'year']).agg(mp=(col, 'median'), mr=('rv_actual', 'median'))
    r = g.mp / g.mr
    sc.append(dict(variant=name, h=h, kind=kind, n=len(r), out=int(((r < .5) | (r > 2)).sum()),
                   min=r.min(), max=r.max()))

for h in (1, 5, 22):
    for v in ('P1', 'P2', 'xgb'):
        pv = REG / f'{v}_h{h}_val.parquet'
        if not pv.exists(): continue
        d = pd.read_parquet(pv)
        for s, g in d.groupby('step'):
            rows.append(dict(variant=v, h=h, step=s, val_year=int(g.year.iloc[0]), qlike_val=qlike(g.rv_actual.values, g.rv_pred.values)))
        scale(v, h, 'val', d); scale(v, h, 'test', pd.read_parquet(REG / f'{v}_h{h}_test.parquet'))
    rows.append(dict(variant='old_dart', h=h, step=np.nan, val_year=np.nan, qlike_val=np.nan))  # в base_models только тестовые годы: нет
    o = pd.read_parquet(OLD / f'base_models_h{h}.parquet')
    scale('old_lightgbm', h, 'test', o, 'pred_lightgbm'); scale('old_xgboost', h, 'test', o, 'pred_xgboost')

q = pd.DataFrame(rows); q.to_csv(OUT / 'etap5_val_qlike.csv', index=False)
s = pd.DataFrame(sc); s.to_csv(OUT / 'etap5_val_masshtab.csv', index=False)
print('Валидационный QLIKE по шагам (старый dart: нет валидации в base_models)')
print(q.dropna(subset=['step']).pivot_table(index=['h', 'variant'], columns='step', values='qlike_val').round(3).assign(
    mean=lambda t: t.mean(axis=1)).round(3).to_string())
print('\nМасштаб: тикер-годов вне [0,5; 2]'); print(s.round(2).to_string(index=False))
