#!/usr/bin/env python3
"""Д16.1, выживший h=1: нога top5 (шаги 1-6 - среднее RV пяти точек ЛГК B v2b, шаги 7-10 - точка шага 6, seed 42), Д8.12 на HAR-X,
поправка Б0.1 sm. Сборка val (шаги 1-10) и теста (шаги 1-10) БЕЗ МЕТРИК: val шагов 7-10 используется только как процедура оценки s²
для теста (как в final2), QLIKE на них не считается.
Выход: data/regen/D16_h1_d812x_top5_val.parquet; data/regen/test_locked/D16_h1_d812x/D16_h1_test.parquet (rv_pred = с поправкой sm,
rv_pred_raw, F, design); data/regen/D16/D16_h1_step_factors.csv."""
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harx as HX

R = HX.REGEN
KEY = ['date', 'ticker', 'step']


def leg(parts):
    d = pd.concat(parts, ignore_index=True)
    return d.groupby(KEY, as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


def main():
    V = leg([pd.read_parquet(R / f'D16_h1_d812x{s}_val.parquet') for s in ('', '_s79', '_s10')])
    T = leg([pd.read_parquet(R / 'test_locked' / f'D16_h1_d812x{s}' / f'D16_h1_d812x{s}_test.parquet') for s in ('', '_s79', '_s10')])
    v = V[V.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    F = {st: float(np.exp(HX.s2_est('sm', E, st) / 2)) for st in sorted(T.step.unique())}
    T = T.rename(columns={'rv_pred': 'rv_pred_raw'})
    T['F'] = T.step.map(F)
    T['rv_pred'] = T.rv_pred_raw * T.F
    T['design'] = 'D16_h1_d812x top5/V0_XGB/B-sm/L0 (Д8.12 на HAR-X)'
    assert sorted(V.step.unique()) == list(range(1, 11)) and sorted(T.step.unique()) == list(range(1, 11))
    V.to_parquet(R / 'D16_h1_d812x_top5_val.parquet', index=False)
    d = R / 'test_locked' / 'D16_h1_d812x'
    T.to_parquet(d / 'D16_h1_test.parquet', index=False)
    pd.DataFrame([dict(step=k, F=f) for k, f in F.items()]).to_csv(R / 'D16' / 'D16_h1_step_factors.csv', index=False)
    print(len(V), len(T), {k: round(f, 3) for k, f in F.items()}, '; метрики не считались')


def build_h5(cand, code):
    """h=5 (условный выживший по среднему 3 seed): нога seeds3-wrap2 = среднее RV XGB и LGBM по seed 42/43/44, шаги 1-10, Б0.1 code."""
    src = [('', 'val'), ('_s4344', 'val'), ('_s710', 'val')]
    V = leg([pd.read_parquet(R / f'D16_{cand}{s}_val.parquet') for s, _ in src])
    T = leg([pd.read_parquet(R / 'test_locked' / f'D16_{cand}{s}' / f'D16_{cand}{s}_test.parquet') for s, _ in src])
    v = V[V.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    F = {st: float(np.exp(HX.s2_est(code, E, st) / 2)) for st in sorted(T.step.unique())}
    T = T.rename(columns={'rv_pred': 'rv_pred_raw'})
    T['F'] = T.step.map(F); T['rv_pred'] = T.rv_pred_raw * T.F
    T['design'] = f'D16_{cand} seeds3-wrap2/V0/B-{code}/L0 (V9 на HAR-X)'
    assert sorted(V.step.unique()) == list(range(1, 11)) and sorted(T.step.unique()) == list(range(1, 11))
    V.to_parquet(R / f'D16_{cand}_seeds3wrap2_val.parquet', index=False)
    T.to_parquet(R / 'test_locked' / f'D16_{cand}' / f'D16_{cand}_seeds3wrap2_test.parquet', index=False)
    print(cand, len(V), len(T), {int(k): round(f, 3) for k, f in F.items()}, '; метрики не считались')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'h5':
        build_h5('h5_F2mx', 'smrob'); build_h5('h5_D12x', 'sm')
    else:
        main()
