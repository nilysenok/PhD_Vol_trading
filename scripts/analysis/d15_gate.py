#!/usr/bin/env python3
"""Д15: оценка вариантов по трём seed и порог (решение владельца 25.10, 21:40).
Метрика seed s — средний OOS QLIKE шагов 2-6 (val 2017-2021) лучшего из двух бустингов с поправкой Б0.1 ord (s² — по val-году шага t−1,
l2_lib.evaluate); seed 42 — {TAG}_{m}_h{h}_val.parquet, seed 43/44 — {TAG}_{m}_h{h}_seeds4344_val.parquet (выбранная точка шага).
Порог: вариант принимается, если среднее по трём seed лучше текущего на ≥ 1% и метрика лучше на каждом seed.
Эталон цели (3) — HAR-J на той же матрице (l2_lib.har_ln: МНК на ln, обучение на годах окна без val-года) с той же поправкой.
Читаются только шаги 1-6 (разработочный набор); val шагов 7-9 и тест не читаются.
python3 scripts/analysis/d15_gate.py --h 5 --tag D15v_h5_base [--cur D15v_h5_base] [--label base]"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

OUT = L.REGEN / 'D15'
SEEDS = (42, 43, 44)
CODES = ('ord', 'rob', 'sm', 'smrob')


def dev(df):
    return df[df.step <= 6]


def seed_vals(tag, h):
    V = {}
    for m in ('xgb', 'lgbm'):
        main = dev(pd.read_parquet(L.REGEN / f'{tag}_{m}_h{h}_val.parquet'))
        s = dev(pd.read_parquet(L.REGEN / f'{tag}_{m}_h{h}_seeds4344_val.parquet'))
        V[(m, 42)] = main
        for sd in (43, 44):
            V[(m, sd)] = s[s.seed == sd]
    return V


def wrap(a, b):
    k = ['step', 'date', 'ticker']
    m = a[k + ['rv_actual', 'rv_pred']].merge(b[k + ['rv_pred']], on=k, suffixes=('', '_b'))
    return m.assign(rv_pred=(m.rv_pred + m.rv_pred_b) / 2)


def score(tag, h):
    V = seed_vals(tag, h)
    rows = []
    for sd in SEEDS:
        S = L.summarize({m: V[(m, sd)] for m in ('xgb', 'lgbm')})
        W = L.evaluate(wrap(V[('xgb', sd)], V[('lgbm', sd)]))
        rows.append(dict(seed=sd, metric=S['metric'], best_model=S['best_model'], best4=S['best4'], best4_mc=S['best4_mc'],
                         **{f'{m}_{c}': S[f'{m}_{c}'] for m in ('xgb', 'lgbm') for c in CODES}, wrap_ord=W['ord'][0], wrap_smrob=W['smrob'][0],
                         per_step=' '.join(f'{q:.4f}' for q in S['per_step'][S['best_model']])))
    return pd.DataFrame(rows)


def harj(h):
    p = OUT / f'harj_v3_h{h}_dev.parquet'
    if not p.exists():
        A, FC = L.load_base(None, (), h)
        hc = L.harj_cols(A)
        fr = []
        for st in range(1, 7):
            tp, vi, *_ = L.split(A, h, st)
            pv = L.har_ln(tp, [vi], h, hc)[0]
            fr.append(pd.DataFrame({'step': st, 'date': vi['date'].values, 'ticker': vi['ticker'].values,
                                    'rv_actual': vi[L.tcol(h)].values, 'rv_pred': np.exp(pv)}))
        pd.concat(fr).to_parquet(p, index=False)
    E = L.evaluate(pd.read_parquet(p))
    return {c: E[c][0] for c in ('raw',) + CODES}, {c: E[c][1] for c in ('raw',) + CODES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--cur', default=None, help='тег текущей комбинации (для порога)'); ap.add_argument('--label', default='')
    a = ap.parse_args()
    h = a.h
    S = score(a.tag, h)
    Hm, _ = harj(h)
    print(f'{a.tag} ({a.label}) h={h}:')
    print(S.drop(columns=['per_step']).round(4).to_string(index=False))
    mean = S.metric.mean()
    row = dict(time=time.strftime('%H:%M'), h=h, tag=a.tag, label=a.label, cur=a.cur or '', **{f'm{sd}': v for sd, v in zip(S.seed, S.metric)},
               mean=mean, models=','.join(S.best_model), best4_mean=S.best4.mean(), wrap_ord_mean=S.wrap_ord.mean(),
               xgb_ord_mean=S.xgb_ord.mean(), lgbm_ord_mean=S.lgbm_ord.mean(), xgb_smrob_mean=S.xgb_smrob.mean(), lgbm_smrob_mean=S.lgbm_smrob.mean(),
               per_step42=S.per_step.iloc[0], harj_ord=Hm['ord'], harj_smrob=Hm['smrob'], harj_raw=Hm['raw'])
    print(f'среднее по seed {mean:.4f}; HAR-J ord {Hm["ord"]:.4f}, smrob {Hm["smrob"]:.4f}, без поправки {Hm["raw"]:.4f}; '
          f'к HAR-J ord {mean / Hm["ord"] - 1:+.1%}')
    if a.cur:
        C = score(a.cur, h)
        cm = C.metric.mean()
        d = mean / cm - 1
        each = bool((S.metric.values < C.metric.values).all())
        ok = d <= -0.01 and each
        row.update(cur_mean=cm, delta=d, each_seed=each, accepted=ok,
                   seed_deltas=' '.join(f'{x:+.2%}' for x in S.metric.values / C.metric.values - 1))
        print(f'к текущему {a.cur}: {d:+.2%} (по seed {row["seed_deltas"]}); лучше на каждом seed: {each} → {"ПРИНЯТ" if ok else "отброшен"}')
    f = OUT / f'levers_h{h}.csv'
    pd.concat([pd.read_csv(f), pd.DataFrame([row])] if f.exists() else [pd.DataFrame([row])]).to_csv(f, index=False, float_format='%.6g')


if __name__ == '__main__':
    main()
