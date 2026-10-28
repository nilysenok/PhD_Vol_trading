#!/usr/bin/env python3
"""Д17.1 с основой HAR-X: ln-прогноз HAR-X (harx.py агента d16-harx, без изменений: слагаемые по умолчанию, эластичная сеть, α/l1 по году vy-1)
на ВСЕХ строках шага — обучение (внутри выборки, для init_score/base_margin), val-год, тест-год. Схема шага та же, что у harx.forecast_step
и l2_lib.split (обучение — годы < vy). Сверка: val-прогноз совпадает с data/regen/HARX_h{h}_val.parquet (rv_pred).
Выход: data/regen/D17/harx_base_h{h}.parquet — date, ticker, step, part (tr/va/te), base_ln.
python3 scripts/analysis/d17_1_harx_base.py --h 1 5 22"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harx as HX
import gibrid_har as GH

REGEN = HX.REGEN


def step_all(A, h, k, harj, terms, Xt):
    """harx.forecast_step(all_rows=True): ln-прогноз модели шага k на всех строках; режем на обучение/val/тест шага."""
    target = f'rv_target_h{h}'
    _, _, ln = HX.forecast_step(A, h, k, harj, terms, Xt, all_rows=True)
    ok = A[target].notna()
    ty, vy = 2016 + k, 2015 + k
    out = []
    for m, part in (((A.year < vy) & ok, 'tr'), ((A.year == vy) & ok, 'va'), (A.year == ty, 'te')):
        out.append(pd.DataFrame({'date': A.loc[m, 'date'].values, 'ticker': A.loc[m, 'ticker'].values, 'step': k, 'part': part,
                                 'base_ln': ln[m].values}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, nargs='+', default=[1, 5, 22])
    ap.add_argument('--steps', type=int, default=9, help='10 — ещё шаг 10 (обучение <= 2024, val 2025, тест 01-02.2026)')
    a = ap.parse_args()
    A, harj = HX.load()
    allt = list(dict.fromkeys(HX.BASE_TERMS + sum(HX.D17_TERMS.values(), [])))
    Xall = HX.build_terms(A, allt)
    (REGEN / 'D17').mkdir(exist_ok=True)
    for h in a.h:
        terms = [t for t in HX.default_terms(h, True) if not (h == 1 and t == 'ln_sig1')]
        B = pd.concat([step_all(A, h, k, harj, terms, Xall) for k in range(1, a.steps + 1)], ignore_index=True)
        ref = pd.read_parquet(REGEN / f'HARX_h{h}_val.parquet')
        ref = ref[ref.step <= 9]
        m = B[B.part == 'va'].merge(ref[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'])
        d = np.abs(np.clip(np.exp(m.base_ln), 1e-10, None) / m.rv_pred - 1).max()
        print(f'h={h}: терминов {terms}; строк {len(B)}; сверка val с HARX_h{h}_val: {len(m)} строк, макс. отн. расхождение {d:.2e}', flush=True)
        f = REGEN / 'D17' / f'harx_base_h{h}.parquet'
        if f.exists():  # сверка с прежним файлом на общих шагах
            O = pd.read_parquet(f)
            mm = O.merge(B, on=['date', 'ticker', 'step', 'part'], suffixes=('_o', ''))
            print(f'  сверка с прежним файлом: {len(mm)} из {len(O)} строк, макс. |Δ| {np.abs(mm.base_ln - mm.base_ln_o).max():.2e}', flush=True)
        B.to_parquet(f, index=False)


if __name__ == '__main__':
    main()
