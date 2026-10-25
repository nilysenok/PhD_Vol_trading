#!/usr/bin/env python3
"""Д17: сравнение кандидата с текущим финалом по 3 seed (правило владельца 26.10, 00:55).
Метрика отбора — средний OOS QLIKE шагов 2-5 (val 2017-2020) лучшего из XGB/LGBM; шаг 6 (2021) — отложенный год, «не хуже текущего».
Поправка: кандидат gamma с офсетом — 'raw' (без поправки), остальные — код текущего финала (h=1 sm, h=5 sm, h=22 smrob); справочно ord.
Шаги 7-9 (подтверждающий набор) отбрасываются до расчёта.
python3 scripts/analysis/d17_eval.py --h 22 --cand G22 [--code raw]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

R = L.REGEN
BASE = {1: 'D12_h1_A_d812', 5: 'D12_h5', 22: 'D12_h22'}
BCODE = {1: 'sm', 5: 'sm', 22: 'smrob'}


def base_frames(h, m, tag=None):
    b = tag or BASE[h]
    v = pd.read_parquet(R / f'{b}_{m}_h{h}_val.parquet')
    s = pd.read_parquet(R / f'{b}_{m}_h{h}_seeds4344_val.parquet')
    v = v[v.get('rank', 1) == 1] if 'rank' in v else v
    out = {42: v[v.step <= 6]}
    for sd in (43, 44):
        out[sd] = s[(s.seed == sd) & (s.step <= 6)]
    return out


def cand_frames(h, name, m):
    V = pd.read_parquet(R / 'D17' / f'h{h}' / f'{name}_val.parquet')
    V = V[(V.model == m) & (V['rank'] == 1) & (V.step <= 6)]
    return {sd: g for sd, g in V.groupby('seed')}


def score(V, code):
    e = L.evaluate(V, lo=2, hi=6)
    q = e[code][1]  # шаги 2..6
    return float(np.mean(q[:4])), float(q[4]), q


def compare(h, name, code, bcode=None, base_tag=None, cand_loader=None):
    bcode = bcode or BCODE[h]
    rows = []
    for sd in (42, 43, 44):
        r = {'seed': sd}
        for who, loader, cd in (('base', lambda m: base_frames(h, m, base_tag)[sd], bcode), ('cand', lambda m: (cand_loader or cand_frames)(h, name, m)[sd], code)):
            best = None
            for m in ('xgb', 'lgbm'):
                try:
                    sel, h6, q = score(loader(m), cd)
                except (KeyError, FileNotFoundError):
                    continue
                r[f'{who}_{m}'] = sel
                if best is None or sel < best[0]:
                    best = (sel, h6, m, q)
            r[f'{who}_sel'], r[f'{who}_2021'], r[f'{who}_model'] = best[0], best[1], best[2]
            r[f'{who}_steps'] = ' / '.join(f'{x:.4f}' for x in best[3])
        r['delta_sel'] = r['cand_sel'] / r['base_sel'] - 1
        r['delta_2021'] = r['cand_2021'] / r['base_2021'] - 1
        rows.append(r)
    df = pd.DataFrame(rows)
    mean_d = df.cand_sel.mean() / df.base_sel.mean() - 1
    ok = (mean_d <= -0.01) and bool((df.delta_sel < 0).all())
    hold = df.cand_2021.mean() <= df.base_2021.mean()
    return df, mean_d, ok, hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--cand', required=True)
    ap.add_argument('--code', default='raw'); ap.add_argument('--bcode', default=None); ap.add_argument('--base-tag', default=None)
    ap.add_argument('--cand-tag', action='store_true', help='кандидат — выход tune (data/regen/<cand>_{m}_h{h}_val + _seeds4344)')
    a = ap.parse_args()
    cl = (lambda h, n, m: base_frames(h, m, n)) if a.cand_tag else None
    df, md, ok, hold = compare(a.h, a.cand, a.code, a.bcode, a.base_tag, cl)
    pd.set_option('display.width', 250)
    print(df[['seed', 'base_sel', 'base_model', 'cand_sel', 'cand_model', 'delta_sel', 'base_2021', 'cand_2021', 'delta_2021']].to_string(float_format='%.4f'))
    print(df[['seed', 'base_steps', 'cand_steps']].to_string())
    print(f'h={a.h} {a.cand} ({a.code}) против {a.base_tag or BASE[a.h]} ({a.bcode or BCODE[a.h]}): среднее шагов 2-5 {df.base_sel.mean():.4f} -> '
          f'{df.cand_sel.mean():.4f} ({md * 100:+.2f}%); порог (>=1% и лучше на каждом seed): {"да" if ok else "нет"}; 2021 не хуже: {"да" if hold else "нет"}')
    mp = R / 'D17' / f'h{a.h}' / f'{a.cand}_meta.csv'
    if mp.exists():
        M = pd.read_csv(mp)
        M = M[(M['rank'] == 1) & (M.step <= 6)]
        print('деревья (seed 42, шаги 1-6):', M[M.seed == 42].pivot(index='model', columns='step', values='best_iteration').to_string())


if __name__ == '__main__':
    main()
