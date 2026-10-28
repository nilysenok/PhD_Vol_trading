#!/usr/bin/env python3
"""Д16.4 / Д16.1 / Д16.6: оценка на разработочном наборе (OOS, шаги 2-6: выбор по 2017-2020, 2021 - отложенный). Шаги 7-9 отбрасываются
до расчёта (подтверждающий набор закрыт), тест не читается.
Модели (ноги - как у финалистов): HAR-J без поправки, HAR-J с поправкой, HAR-X с поправкой, финалист-гибрид, Д16.1 (гибрид на HAR-X).
Поправка Б0.1 - своя s² модели (gibrid_bias: оценка по её val-остаткам), val шага k - оценка шага k-1.
DM на каждом шаге: дневной ряд средней по тикерам разности QLIKE, t-статистика Ньюи-Уэста (лаг max(h-1, 5)); <0 - первая лучше.
Выход: data/regen/D16/d16_eval_h{h}.csv, d16_dm_h{h}.csv; печать - таблица для отчёта."""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harx as HX

R = HX.REGEN
KEY = ['date', 'ticker', 'step']
CODE = {1: 'sm', 5: 'sm', 22: 'smrob'}


def leg(df, how):
    df = df[df.step <= 6]
    return df.groupby(KEY, as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


def stored(h):
    """Финалисты-гибриды (сохранённые файлы, только шаги <= 6)."""
    rd = lambda n: pd.read_parquet(R / n)
    if h == 1:
        return {'гибрид (финал D12_h1_A_d812 top5)': leg(rd('D12_h1_A_d812_xgb_h1_top5_val.parquet'), 'mean')}
    if h == 5:
        out = {}
        for nm in ('D12_h5', 'F2m_h5'):
            x, l = rd(f'{nm}_xgb_h5_val.parquet'), rd(f'{nm}_lgbm_h5_val.parquet')
            out[f'гибрид ({nm} wrap2)'] = leg(pd.concat([x, l]), 'mean')
        return out
    x = pd.concat([rd('D12_h22_xgb_h22_val.parquet'), rd('D12_h22_xgb_h22_seeds4344_val.parquet')])
    return {'гибрид (D12_h22 seeds3, справка; финал - HAR-J)': leg(x, 'mean')}


def d16(h, suffix=''):
    out = {}
    for cand, nm in {1: [('h1_d812x', 'Д16.1 d812 на HAR-X top5')], 5: [('h5_D12x', 'Д16.1 D12_h5 на HAR-X wrap2'), ('h5_F2mx', 'Д16.1 F2m_h5 на HAR-X wrap2')],
                     22: [('h22_D12x', 'Д16.1 D12_h22 на HAR-X seeds3')]}[h]:
        p = R / f'D16_{cand}{suffix}_val.parquet'
        if p.exists():
            out[nm + suffix] = leg(pd.read_parquet(p), 'mean')
    return out


def bias(df, code):
    v = df[df.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    F = {st: (np.exp(HX.s2_est(code, E, st - 1) / 2) if st > 1 else 1.0) for st in df.step.unique()}
    return df.assign(rv_pred=df.rv_pred * df.step.map(F))


def nw_t(d, lag):
    d = np.asarray(d, float); n = len(d); m = d.mean(); e = d - m
    s = np.dot(e, e) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * np.dot(e[k:], e[:-k]) / n
    return m / np.sqrt(s / n)


def ql(y, p):
    r = y / p
    return r - np.log(r) - 1


def run(h, extra_suffixes=()):
    A, harj = HX.load()
    hv, _, _ = HX.har_steps(h, A, harj)
    hv = hv[hv.step <= 6][KEY + ['rv_actual', 'rv_pred']]
    xv = pd.read_parquet(R / f'HARX_h{h}_val.parquet'); xv = xv[xv.step <= 6][KEY + ['rv_actual', 'rv_pred']]
    x0 = pd.read_parquet(R / f'HARX0_h{h}_val.parquet'); x0 = x0[x0.step <= 6][KEY + ['rv_actual', 'rv_pred']]
    M = {'HAR-J без поправки': ('raw', hv), 'HAR-J с поправкой': (None, hv), 'HAR-X с поправкой (Д16.4)': (None, xv),
         'HAR-X0 (без Д17.3), справка': (None, x0)}
    for k, v in stored(h).items():
        M[k] = (None, v)
    for sfx in ('',) + tuple(extra_suffixes):
        for k, v in d16(h, sfx).items():
            M[k] = (None, v)
    rows, P = [], {}
    for code in ('sm', 'smrob'):
        for nm, (c0, df) in M.items():
            d = df if c0 == 'raw' else bias(df, code)
            P[(nm, code)] = d
    # общий набор строк
    base = None
    for d in P.values():
        k = d[(d.step >= 2) & (d.rv_actual > 1e-12)][KEY]
        base = k if base is None else base.merge(k, on=KEY)
    for (nm, code), d in P.items():
        d = base.merge(d, on=KEY)
        q = {int(s): float(ql(g.rv_actual.values, g.rv_pred.values).mean()) for s, g in d.groupby('step')}
        rows.append(dict(h=h, model=nm, code=code, **{f'q{2015 + s}': q[s] for s in range(2, 7)},
                         mean17_20=np.mean([q[s] for s in range(2, 6)]), n=len(d)))
    T = pd.DataFrame(rows)
    hj = T[T.model == 'HAR-J с поправкой'].set_index('code')
    T['ratio_harj17_20'] = T.apply(lambda r: r.mean17_20 / hj.loc[r.code, 'mean17_20'], axis=1)
    T['ratio_harj2021'] = T.apply(lambda r: r.q2021 / hj.loc[r.code, 'q2021'], axis=1)
    T.to_csv(R / 'D16' / f'd16_eval_h{h}.csv', index=False)
    # DM по шагам: цепочка + Д16.1 против финалиста
    code = CODE[h]
    names = list(M)
    chain = [('HAR-J с поправкой', 'HAR-J без поправки'), ('HAR-X с поправкой (Д16.4)', 'HAR-J с поправкой')]
    hyb = [n for n in names if n.startswith('гибрид')]
    for n in hyb:
        chain.append((n, 'HAR-X с поправкой (Д16.4)'))
    for n in [n for n in names if n.startswith('Д16.1')]:
        ref = {1: hyb[0], 22: hyb[0]}.get(h) or (hyb[0] if 'D12_h5' in n else hyb[1])
        chain += [(n, ref), (n, 'HAR-X с поправкой (Д16.4)')]
    lag = max(h - 1, 5)
    dm = []
    for a_, b_ in chain:
        da = base.merge(P[(a_, code)], on=KEY); db = base.merge(P[(b_, code)], on=KEY)
        m = da.merge(db[KEY + ['rv_pred']], on=KEY, suffixes=('_a', '_b'))
        m['d'] = ql(m.rv_actual, m.rv_pred_a) - ql(m.rv_actual, m.rv_pred_b)
        r = dict(h=h, code=code, a=a_, b=b_)
        for s, g in m.groupby('step'):
            ds = g.groupby('date').d.mean().sort_index().values
            r[f't{2015 + int(s)}'] = nw_t(ds, lag)
        dm.append(r)
    DM = pd.DataFrame(dm)
    DM.to_csv(R / 'D16' / f'd16_dm_h{h}.csv', index=False)
    return T, DM


def seeds_check():
    """Порог по 3 seed: QLIKE (поправка финалиста) по каждому seed 42/43/44 и нога seeds3 (среднее RV трёх seed); Д16.1 против финалиста
    в одинаковой ноге (h=1 - точка ранга 1 XGB; h=5 - среднее XGB и LGBM; h=22 - XGB). 2017-2020 и 2021."""
    rd = lambda n: pd.read_parquet(R / n)

    def fin(nm, h, models):
        out = []
        for m in models:
            a = rd(f'{nm}_{m}_h{h}_val.parquet')
            a = a.assign(seed=a['seed'] if 'seed' in a else 42, model=m)
            out.append(pd.concat([a, rd(f'{nm}_{m}_h{h}_seeds4344_val.parquet').assign(model=m)]))
        return pd.concat(out)

    def mine(cand):
        a = rd(f'D16_{cand}_val.parquet'); a = a[a['rank'] == 1]
        p = R / f'D16_{cand}_s4344_val.parquet'
        return pd.concat([a] + ([rd(p.name)] if p.exists() else []))
    rows = []
    for h, nm, cand, models in ((1, 'D12_h1_A_d812', 'h1_d812x', ('xgb',)), (5, 'D12_h5', 'h5_D12x', ('xgb', 'lgbm')),
                                (5, 'F2m_h5', 'h5_F2mx', ('xgb', 'lgbm')), (22, 'D12_h22', 'h22_D12x', ('xgb',))):
        code = 'smrob' if nm in ('F2m_h5', 'D12_h22') else 'sm'
        for lab, df in (('финалист ' + nm, fin(nm, h, models)), ('Д16.1 ' + cand, mine(cand))):
            df = df[(df.step <= 6) & df.model.isin(models)]
            res = {f's{sd}': g for sd, g in df.groupby('seed')}
            res['seeds3'] = df
            r = dict(h=h, model=lab, code=code, n_seeds=df.seed.nunique())
            for k, g in res.items():
                d = g.groupby(KEY, as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))
                d = bias(d, code); d = d[(d.rv_actual > 1e-12) & (d.step >= 2)]
                q = {int(s): float(ql(x.rv_actual.values, x.rv_pred.values).mean()) for s, x in d.groupby('step')}
                r[f'{k}_17_20'] = np.mean([q[s] for s in range(2, 6)]); r[f'{k}_2021'] = q[6]
            r['mean3_17_20'] = np.mean([r[f's{s}_17_20'] for s in (42, 43, 44) if f's{s}_17_20' in r])
            r['mean3_2021'] = np.mean([r[f's{s}_2021'] for s in (42, 43, 44) if f's{s}_2021' in r])
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(R / 'D16' / 'd16_1_seeds3.csv', index=False)
    return S


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, nargs='+', default=[1, 5, 22])
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--seeds', action='store_true', help='только порог по 3 seed (Д16.1)')
    a = ap.parse_args()
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 60)
    if a.seeds:
        print(seeds_check().round(4).to_string()); sys.exit()
    for h in a.h:
        T, DM = run(h, a.extra)
        print(T.round(4).to_string()); print(DM.round(2).to_string())
