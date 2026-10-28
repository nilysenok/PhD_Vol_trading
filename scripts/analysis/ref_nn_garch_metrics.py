#!/usr/bin/env python3
"""Метрики справочных LSTM / GRU / GARCH / GJR только на разработочном наборе: годы оценки 2017-2021 (шаги 2-6), OOS.
QLIKE (как gibrid_schemes.qlike), RMSE и MAE (RV x 1e4) по годам и среднее по годам; общие строки (date, ticker) всех моделей.
Эталон — HAR-J с поправкой sm (как gibrid_bias: s² по объединённым остаткам до трёх val-лет шагов k-3..k-1, F = exp(s²/2)),
прогнозы HAR-J — data/regen/D16_10_harj_h{h}_val.parquet. Подтверждающий набор и тест не читаются.
Выход: data/regen/ref_nn_garch/REF_metrics_dev.csv; таблица в stdout (markdown).
python3 scripts/analysis/ref_nn_garch_metrics.py"""
from pathlib import Path
import numpy as np, pandas as pd

REG = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
OUT = REG / 'ref_nn_garch'
HS, YEARS, SEEDS = (1, 5, 22), list(range(2017, 2022)), (42, 43, 44)
K = ['date', 'ticker']


def qlike(y, p):
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def harj_sm(h):
    d = pd.read_parquet(REG / f'D16_10_harj_h{h}_val.parquet')
    d = d[d.rv_actual > 1e-12].copy()  # фильтр gibrid_io.merged
    E = {k: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for k, g in d.groupby('step')}
    s2 = lambda st: float(np.var(np.concatenate([E[j] for j in range(max(1, st - 2), st + 1)])))
    d['F'] = d.step.map(lambda k: np.exp(s2(k - 1) / 2) if k > 1 else 1.0)
    d['p'] = d.rv_pred * d.F
    return d[d.step.between(2, 6)][K + ['step', 'rv_actual', 'p']]


def nn_seed_variant(d, s):
    """Прогноз одного seed с собственной цепочкой s² (как основной: s² шага k-1, эмбарго)."""
    v = d[d.emb & (d.rv_actual > 0)]
    s2 = {k: float(np.var(np.log(g.rv_actual.values + 1e-10) - g[f'lp_s{s}'].values)) for k, g in v.groupby('step')}
    return np.exp(d[f'lp_s{s}']) * np.exp(d.step.map(lambda k: s2.get(k - 1, 0.0)) / 2)


def variants(m, h):
    fp = OUT / f'REF_{m}_h{h}_val.parquet'
    if not fp.exists():
        return {}
    d = pd.read_parquet(fp)
    d['date'] = pd.to_datetime(d.date)
    v = {f'{m}': d.rv_pred, f'{m} без Б0.1': d.rv_pred_raw}
    if m in ('GRU', 'LSTM'):
        v.update({f'{m} seed {s}': nn_seed_variant(d, s) for s in SEEDS if f'lp_s{s}' in d})
    else:
        v[f'{m} масштаб без F'] = d.rv_pred_a
        v[f'{m} без калибровки'] = d.rv_pred_raw
        del v[f'{m} без Б0.1']
    base = d[K + ['step', 'rv_actual']]
    return {k: base.assign(p=x.values)[base.step.between(2, 6).values] for k, x in v.items()}


def main():
    rows = []
    for h in HS:
        V = {'HAR-J с поправкой (sm)': harj_sm(h)}
        for m in ('LSTM', 'GRU', 'GJR', 'GARCH'):
            V.update(variants(m, h))
        keys = None
        for d in V.values():
            k = d[K].astype(str).agg('|'.join, axis=1)
            keys = set(k) if keys is None else keys & set(k)
        for name, d in V.items():
            d = d[d[K].astype(str).agg('|'.join, axis=1).isin(keys).values]
            yr = pd.to_datetime(d.date).dt.year
            for y in YEARS:
                g = d[yr == y]
                e = (g.rv_actual - g.p).values * 1e4
                rows.append(dict(h=h, model=name, year=y, n=len(g), qlike=qlike(g.rv_actual.values, g.p.values),
                                 rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e)))))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'REF_metrics_dev.csv', index=False)
    MAIN = ['HAR-J с поправкой (sm)', 'LSTM', 'GRU', 'GJR', 'GARCH']
    for met, only in (('qlike', MAIN), ('rmse', MAIN), ('mae', MAIN), ('qlike', None), ('rmse', None), ('mae', None)):
        print(f'\n### {met.upper()} (OOS, шаги 2-6){", RV x 1e4" if met != "qlike" else ""}{": основные варианты" if only else ": все варианты"}')
        print('| h | модель | ' + ' | '.join(map(str, YEARS)) + ' | средн. | к HAR-J |')
        print('|---|---|' + '---|' * (len(YEARS) + 2))
        for h in HS:
            P = R[R.h == h].pivot(index='model', columns='year', values=met)
            P['mean'] = P[YEARS].mean(axis=1)
            ref = P.loc['HAR-J с поправкой (sm)', 'mean']
            order = ['HAR-J с поправкой (sm)'] + [m for m in P.index if m != 'HAR-J с поправкой (sm)']
            if only:
                order = [m for m in only if m in P.index]
            for m in order:
                f = lambda x: f'{x:.4f}'.replace('.', ',')
                rel = '' if m.startswith('HAR-J') else f'{(P.loc[m, "mean"] / ref - 1) * 100:+.1f}%'.replace('.', ',')
                print(f'| {h} | {m} | ' + ' | '.join(f(P.loc[m, y]) for y in YEARS) + f' | {f(P.loc[m, "mean"])} | {rel} |')
    print('\nстрок на h (общих):', R.groupby('h').n.sum().to_dict() if len(R) else {}, '— делить на число моделей')


if __name__ == '__main__':
    main()
