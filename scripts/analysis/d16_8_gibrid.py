#!/usr/bin/env python3
"""Д16.8: критерии c1-c4 ноги финалиста до/после (гибрид gibrid_v17.py, только val шагов 1-6; тест закрыт — load_boost не читает test).
Готовит файлы ног в data/regen/D16_8/gib/: D16_8{n|q}_{tag}_{xgb|lgbm}_h{h}_val.parquet (seed 42), _seeds4344_val, _wrap2_val (h=5: слот базовой
модели — среднее RV XGB и LGBM, другой слот — вторая модель, как final2_wrap) и запускает гибрид в режимах --dev-eval in (критерии на 2016-2021,
решение владельца 00:55) и oos (2017-2021, как во втором круге).
python3 scripts/analysis/d16_8_gibrid.py [--workers 8]"""
import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
GD = R / 'D16_8' / 'gib'
CANDS = {'D12_h22': (22, 'seeds3', None), 'D12_h5': (5, 'wrap2', 'xgb'), 'F2m_h5': (5, 'wrap2', 'lgbm')}
FIN = {'D12_h22': 'seeds3|V0_XGB|B-smrob|L0', 'D12_h5': 'wrap2|V0_XGB|B-sm|L0', 'F2m_h5': 'wrap2|V0_LGB|B-smrob|L0'}  # дизайны финалистов (final2)
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']


def prep():
    GD.mkdir(parents=True, exist_ok=True)
    jobs = []
    for tag, (h, leg, base) in CANDS.items():
        V = pd.read_parquet(R / f'D16_8_h{h}_{tag}_val.parquet')
        V = V[V.step <= 6]
        for es in ('native', 'qlike'):
            b = f'D16_8{es[0]}_{tag}'
            S = {m: V[(V.es == es) & (V.model == m) & (V.seed == 42)][COLS].sort_values(['step', 'date', 'ticker']) for m in ('xgb', 'lgbm')}
            for m in ('xgb', 'lgbm'):
                S[m].to_parquet(GD / f'{b}_{m}_h{h}_val.parquet', index=False)
                V[(V.es == es) & (V.model == m) & (V.seed != 42)][COLS + ['seed']].to_parquet(GD / f'{b}_{m}_h{h}_seeds4344_val.parquet', index=False)
            if leg == 'wrap2':
                k = ['date', 'ticker', 'step']
                avg = S['xgb'].merge(S['lgbm'][k + ['rv_pred']], on=k, suffixes=('', '_l'))
                avg = avg.assign(rv_pred=(avg.rv_pred + avg.rv_pred_l) / 2)[COLS]
                other = 'lgbm' if base == 'xgb' else 'xgb'
                avg.to_parquet(GD / f'{b}_{base}_h{h}_wrap2_val.parquet', index=False)
                S[other].to_parquet(GD / f'{b}_{other}_h{h}_wrap2_val.parquet', index=False)
            for mode in ('in', 'oos'):
                jobs.append((b, h, leg, mode))
    return jobs


def run(job):
    b, h, leg, mode = job
    out = R / 'D16_8' / f'gibrid_{mode}' / b
    cmd = [sys.executable, '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', b, '--h', str(h), '--regen', str(GD),
           '--out', str(out), '--devlog', str(R / 'D16_8' / f'gibrid_devlog_{mode}.md'), '--features', str(R / 'features_pit_v3.parquet'),
           '--legs', 'single', leg, '--dev-eval', mode]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return job, r.returncode, (r.stdout + r.stderr)[-1500:]


def summary():
    """Нога финалиста и лучший дизайн по правилу: c1-c4, QLIKE по годам (in: 2016-2021; oos: 2017-2021), до (n) / после (q)."""
    rows = []
    for tag, (h, leg, _) in CANDS.items():
        for mode in ('in', 'oos'):
            for es in ('native', 'qlike'):
                b = f'D16_8{es[0]}_{tag}'
                d = R / 'D16_8' / f'gibrid_{mode}' / b
                rk = pd.read_csv(d / f'ranking_{b}_h{h}.csv')
                dv = pd.read_csv(d / f'designs_val_{b}_h{h}.csv')
                for role, nm in (('финалист', FIN[tag]), ('лучший по правилу', rk.iloc[0].design)):
                    r = rk[rk.design == nm].iloc[0]
                    g = dv[(dv.design == nm) & (dv.step <= 6) & (dv.step >= (1 if mode == 'in' else 2))]
                    rows.append(dict(tag=tag, h=h, mode=mode, es=es, role=role, design=nm, rank=int(rk.index[rk.design == nm][0]) + 1,
                                     crit=''.join(c[1] if bool(r[c]) else '-' for c in ('c1', 'c2', 'c3', 'c4')), mean_q=r.mean_q,
                                     **{f'q{int(y)}': q for y, q in zip(g.val_year, g.q_hyb)},
                                     **{f'har{int(y)}': q for y, q in zip(g.val_year, g.q_har)}))
    S = pd.DataFrame(rows)
    for h in (5, 22):
        S[S.h == h].to_csv(R / f'D16_8_h{h}_criteria.csv', index=False, float_format='%.6f')
    pd.set_option('display.width', 300)
    print(S.drop(columns=[c for c in S.columns if c.startswith('har')]).to_string(index=False, float_format='%.4f'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.summary:
        return summary()
    jobs = prep()
    with ThreadPoolExecutor(a.workers) as ex:
        for job, rc, tail in ex.map(run, jobs):
            print(job, 'rc', rc, flush=True)
            if rc:
                print(tail, flush=True)


if __name__ == '__main__':
    main()
