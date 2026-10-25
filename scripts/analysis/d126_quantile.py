#!/usr/bin/env python3
"""Д12.6: квантильный остаточный ученик на h=5 (80-й перцентиль) как вход σ_pred для фильтра C главы 4.

Точка — L2_h5 (V9 + Д8.12 + блок Д7; параметры LightGBM `data/regen/L2_h5_selected_h5.csv` по шагам 1-9, без поиска).
Отличие от L2_h5 одно: objective=quantile, alpha=0,8 (цель — ln RV_avg(t+1..t+5) − ln HAR-J, ранняя остановка по pinball-потере val-года, patience 30).
Матрица: основная — features_pit_v2 + d7-вариант h=5 (ровно признаки L2_h5, 318 + ln HAR-J): features_pit_v3 появилась в 20:39, когда счёт уже шёл,
а блока Д7 на v3 нет (d7_block.py строит его от v2), так что точка L2_h5 на v3 — другой набор признаков; --matrix v3 — справочный прогон на v3 без Д7
(242 признака) в другой файл.
Прогноз: rv_q80 = HAR-J × exp(квантильный прогноз остатка) (без поправки Б0.1: это квантиль, не среднее), σ_pred_q80 = sqrt(rv_q80) — дневная σ,
как sigma_pred_h1_v2. QLIKE не считается. Тестовые годы — только прогнозы; val шагов 7-9 не печатается.
python3 scripts/analysis/d126_quantile.py [--workers 3] [--check] [--matrix v2d7|v3]"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

H, ALPHA = 5, 0.8
CFG = {'v9': True, 'harf': True}
G = {}


def task(a):
    step, quant = a
    cfg = dict(CFG, **({'quantile': ALPHA} if quant else {}))
    p = G['P'][('lgbm', step)]
    ref = L.make_ref('lgbm', G['A'], G['cols'], H, cfg, 'Bv2b') if step > 1 else None
    D = L.step_data(G['A'], G['cols'], H, step, 'lgbm', cfg, 'Bv2b')
    pv, pt, bi, _, _ = L.fit('lgbm', p, D, cfg, ref=ref)
    va, te = D['parts']['va'], D['parts']['te']
    fv = pd.DataFrame({'date': va.date.values, 'ticker': va.ticker.values, 'step': step, 'rv_actual': D['y']['va'], 'rv_pred': np.exp(pv)})
    ft = pd.DataFrame({'date': te.date.values, 'ticker': te.ticker.values, 'step': step, 'rv_q80': np.exp(pt)})
    return step, bi, D['ty'], len(D['names']), fv, ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--matrix', choices=('v2d7', 'v3'), default='v2d7')
    ap.add_argument('--check', action='store_true', help='сверка: тот же конвейер с objective=mse воспроизводит L2_h5 (val шагов 2-6) побитно')
    a = ap.parse_args()
    t0 = time.time()
    matrix = L.REGEN / 'features_pit_v3.parquet' if a.matrix == 'v3' else None
    A, FC = L.load_base(matrix, ('d7',) if matrix is None else (), H)
    A, cols, _, _ = L.build_features(A, FC, H, CFG)
    G.update(A=A, cols=cols, P=L.selected_params('L2_h5', H))
    print(f"матрица: {'features_pit_v3 (без Д7)' if matrix else 'features_pit_v2 + d7_h5'}, признаков {len(cols)} (+ ln HAR-J)", flush=True)
    ctx = mp.get_context('fork')
    if a.check:
        with ctx.Pool(a.workers) as P:
            R = P.map(task, [(s, False) for s in (2, 3, 4)], chunksize=1)
        ref = pd.read_parquet(L.REGEN / 'L2_h5_lgbm_h5_val.parquet')
        for step, bi, ty, nf, fv, _ in R:
            r = ref[ref.step == step].merge(fv, on=['date', 'ticker'], suffixes=('_ref', ''))
            print(f'сверка шаг {step}: строк {len(r)}, max |Δ ln rv_pred| = {np.abs(np.log(r.rv_pred_ref) - np.log(r.rv_pred)).max():.2e}', flush=True)
        return
    with ctx.Pool(a.workers) as P:
        R = P.map(task, [(s, True) for s in range(9, 0, -1)], chunksize=1)
    R.sort(key=lambda r: r[0])
    steps = pd.DataFrame([dict(step=r[0], test_year=r[2], best_iteration=r[1], n_features=r[3], n_test=len(r[5])) for r in R])
    steps.to_csv(L.REGEN / ('sigma_pred_h5_q80_steps.csv' if a.matrix == 'v2d7' else 'sigma_pred_h5_q80_v3_steps.csv'), index=False)
    # диагностика только на val шагов 1-6 (разработочный набор): доля фактических RV ниже прогноза q80 (ориентир 0,8; ES шёл по этому же val)
    cov = [(r[0], float((r[4].rv_actual <= r[4].rv_pred).mean())) for r in R if r[0] <= 6]
    print('доля RV <= q80 на val шагов 1-6 (разработочный набор):', ' '.join(f'ш{s}: {c:.3f}' for s, c in cov), flush=True)
    OUT = L.REGEN / ('sigma_pred_h5_q80.parquet' if a.matrix == 'v2d7' else 'sigma_pred_h5_q80_v3.parquet')
    T = pd.concat([r[5] for r in R], ignore_index=True)
    T['sigma_pred_h5'] = np.sqrt(T['rv_q80']).astype('float32')
    S = T[['date', 'ticker', 'sigma_pred_h5']].sort_values(['date', 'ticker']).reset_index(drop=True)
    assert not S.duplicated(['date', 'ticker']).any() and S.sigma_pred_h5.notna().all()
    S.to_parquet(OUT, index=False)
    print(f'{OUT.name}: {len(S)} строк, {S.date.min().date()} … {S.date.max().date()}, деревьев по шагам {steps.best_iteration.tolist()}, {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
