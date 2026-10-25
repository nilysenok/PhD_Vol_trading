#!/usr/bin/env python3
"""Д10: gain-отбор блока d10_* на точке Bv2b (LightGBM, параметры {Bv2b}_selected_h{h}.csv, шаги 1-6, только val).
Суммарный gain по шагам 1-6 для каждого d10_* признака; нулевой -> исключить. Выход: data/regen/d10/gain_h{h}.csv, gain_summary.csv.
python3 scripts/analysis/d10_gain.py [--workers 2]"""
import argparse, multiprocessing as mp, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

G = {}


def task(a):
    h, step = a
    A, cols, p = G['A'], G['cols'], G['P'][h][('lgbm', step)]
    ref = L.make_ref('lgbm', A, cols, h, {}, 'Bv2b') if step > 1 else None
    D = L.step_data(A, cols, h, step, 'lgbm', {}, 'Bv2b')
    pv, _, bi, gain, _ = L.fit('lgbm', p, D, {}, ref=ref)
    q = L.qlike(D['y']['va'], np.exp(pv))
    return h, step, bi, q, dict(zip(D['names'], gain))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); a = ap.parse_args()
    A = pd.read_parquet(L.REGEN / 'features_pit_v2_d10.parquet').sort_values('date').reset_index(drop=True)
    A['year'] = pd.to_datetime(A['date']).dt.year
    _, FC = L.load_base()
    d10 = [c for c in A.columns if c.startswith('d10_')]
    G.update(A=A, cols=FC + d10, P={h: L.selected_params('Bv2b', h) for h in (5, 22)})
    with mp.get_context('fork').Pool(a.workers) as pool:
        res = pool.map(task, [(h, s) for h in (22, 5) for s in range(6, 0, -1)], chunksize=1)
    out = L.REGEN / 'd10'; out.mkdir(exist_ok=True, parents=True)
    rows = []
    for h in (5, 22):
        R = sorted([r for r in res if r[0] == h], key=lambda r: r[1])
        df = pd.DataFrame([dict(h=h, step=r[1], feature=k, gain=v) for r in R for k, v in r[4].items()])
        df.to_csv(out / f'gain_h{h}.csv', index=False)
        tot = df.groupby('feature')['gain'].sum()
        for f in d10:
            rows.append(dict(h=h, feature=f, gain_sum=tot.get(f, 0.0), share_pct=100 * tot.get(f, 0.0) / tot.sum(),
                             rank=int((tot > tot.get(f, 0.0)).sum() + 1), n_steps_used=int((df[df.feature == f].gain > 0).sum())))
        print(h, 'qlike val по шагам (без поправки):', [round(r[3], 4) for r in R], 'best_it:', [r[2] for r in R], 'n_feat:', len(tot), flush=True)
    S = pd.DataFrame(rows); S['exclude'] = S.gain_sum == 0
    S.to_csv(out / 'gain_summary.csv', index=False)
    print(S.round(3).to_string())
