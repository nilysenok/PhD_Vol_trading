#!/usr/bin/env python3
"""Д16.8: сводка — деревья по шагам до/после (нативная ES против ES по QLIKE гибрида), OOS QLIKE шагов 2-6 (val 2017-2021) лучшего бустинга
с поправкой Б0.1 ord (и лучший из 4 s2), по seed 42/43/44; справочно — дизайны финалистов (seeds3 / wrap2 со своей s2). Сверка: нативная ES
воспроизводит сохранённые прогнозы кандидатов (шаги 1-6). Только шаги 1-6.
python3 scripts/analysis/d16_8_eval.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

CANDS = {'D12_h22': (22, ('seeds3', 'xgb', 'smrob')), 'D12_h5': (5, ('wrap2', 'xgb', 'sm')), 'F2m_h5': (5, ('wrap2', 'lgbm', 'smrob'))}
OUT = L.REGEN / 'D16_8'


def check_repro(tag, h, V):
    res = {}
    for m in ('xgb', 'lgbm'):
        S = pd.read_parquet(L.REGEN / f'{tag}_{m}_h{h}_val.parquet')
        S = S[S.step <= 6]
        S4 = pd.read_parquet(L.REGEN / f'{tag}_{m}_h{h}_seeds4344_val.parquet')
        S4 = S4[S4.step <= 6]
        for sd, ref in ((42, S), (43, S4[S4.seed == 43]), (44, S4[S4.seed == 44])):
            mine = V[(V.es == 'native') & (V.model == m) & (V.seed == sd)]
            j = mine.merge(ref[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], suffixes=('', '_ref'))
            res[f'{m}_s{sd}'] = (len(j), len(mine), float(np.max(np.abs(np.log(j.rv_pred) - np.log(j.rv_pred_ref)))))
    return res


def design_val(V, es, leg, model, slot=None):
    """Нога финалиста: seeds3 — среднее RV по 3 seed модели; wrap2 — среднее RV XGB и LGBM (seed 42; слот базовой модели gibrid_io)."""
    k = ['date', 'ticker', 'step']
    g = V[(V.es == es) & (V.model == model)] if leg == 'seeds3' else V[(V.es == es) & (V.seed == 42)]
    return g.groupby(k, as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


def agg(qs):
    """qs — QLIKE шагов 2..6 -> (отбор 2017-2020 = шаги 2-5, проверка 2021 = шаг 6, все 2017-2021)."""
    return float(np.mean(qs[:4])), float(qs[4]), float(np.mean(qs))


def main():
    rows, trees_all, rep = [], [], {}
    for tag, (h, (leg, dm, code)) in CANDS.items():
        V = pd.read_parquet(L.REGEN / f'D16_8_h{h}_{tag}_val.parquet')
        T = pd.read_csv(L.REGEN / f'D16_8_h{h}_{tag}_trees.csv')
        rep[tag] = check_repro(tag, h, V)
        trees_all.append(T)
        for es in ('native', 'qlike'):
            for sd in (42, 43, 44):
                ev = {m: L.evaluate(V[(V.es == es) & (V.model == m) & (V.seed == sd)]) for m in ('xgb', 'lgbm')}
                for m in ev:
                    for c in ('ord',) + tuple(x for x in L.CODES if x != 'ord'):
                        a = agg(ev[m][c][1])
                        rows.append(dict(tag=tag, h=h, kind='model', model=m, code=c, seed=sd, es=es, q1720=a[0], q21=a[1], q1721=a[2],
                                         per_step=json.dumps([round(x, 5) for x in ev[m][c][1]])))
            ev = L.evaluate(design_val(V, es, leg, dm))
            a = agg(ev[code][1])
            rows.append(dict(tag=tag, h=h, kind=f'design_{leg}', model=dm if leg == 'seeds3' else 'avg', code=code, seed=-1 if leg == 'seeds3' else 42, es=es,
                             q1720=a[0], q21=a[1], q1721=a[2], per_step=json.dumps([round(x, 5) for x in ev[code][1]])))
            if leg == 'wrap2':  # та же нога на каждом seed отдельно (среднее XGB и LGBM одного seed) — для порога по 3 seed
                for sd in (43, 44):
                    ev = L.evaluate(design_val(V[V.seed == sd].assign(seed=42), es, leg, dm))
                    a = agg(ev[code][1])
                    rows.append(dict(tag=tag, h=h, kind=f'design_{leg}', model='avg', code=code, seed=sd, es=es,
                                     q1720=a[0], q21=a[1], q1721=a[2], per_step=json.dumps([round(x, 5) for x in ev[code][1]])))
    R = pd.DataFrame(rows)
    # лучший бустинг (Б0.1 ord) по seed: модель выбирается по отборочным годам 2017-2020
    B = []
    for (tag, es, sd), g in R[(R.kind == 'model') & (R.code == 'ord')].groupby(['tag', 'es', 'seed']):
        b = g.loc[g.q1720.idxmin()]
        B.append(dict(b, kind='best_boost'))
    R = pd.concat([R, pd.DataFrame(B)], ignore_index=True)
    T = pd.concat(trees_all, ignore_index=True)
    for h in (5, 22):
        R[R.h == h].to_csv(L.REGEN / f'D16_8_h{h}_metrics.csv', index=False, float_format='%.6f')
    W = T.pivot_table(index=['tag', 'model', 'seed', 'es'], columns='step', values='trees').reset_index()
    for h in (5, 22):
        W[W.tag.map(lambda t: CANDS[t][0]) == h].to_csv(L.REGEN / f'D16_8_h{h}_trees_wide.csv', index=False)
    pd.set_option('display.width', 250)
    print('сверка нативной ES с сохранёнными прогнозами (строк совпало, строк, max |d ln|):')
    for t, r in rep.items():
        print(' ', t, r)
    print('\nдеревья по шагам 1-6:'); print(W.to_string(index=False))
    X = R[R.kind.isin(['best_boost']) | R.kind.str.startswith('design')]
    print('\nметрики (OOS; отбор 2017-2020, проверка 2021):')
    print(X[['tag', 'kind', 'model', 'code', 'seed', 'es', 'q1720', 'q21', 'q1721']].sort_values(['tag', 'kind', 'seed', 'es']).to_string(index=False, float_format='%.4f'))
    M = X.groupby(['tag', 'kind', 'es'])[['q1720', 'q21', 'q1721']].mean().unstack('es')
    print('\nсреднее по seed:'); print(M.to_string(float_format='%.4f'))


if __name__ == '__main__':
    main()
