#!/usr/bin/env python3
"""Д16.8 (диагностика): ранняя остановка остаточного ученика V9 по QLIKE итогового гибрида HAR-J × exp(f(x)) на уровне RV val-года шага
(l2_lib.fit, cfg['es_qlike']: LightGBM — feval при metric='None', XGBoost — custom_metric при disable_default_eval_metric) вместо нативной
метрики (l2 / rmse; у Huber — huber / mphe). Метка обучения не меняется (ln RV − ln HAR-J), patience 30, потолок 1000 деревьев.
Точки — optuna-30 кандидатов на v3 ({tag}_selected_h{h}.csv), без нового подбора. Только разработочный набор: шаги 1-6 (шаг 1 — для s2
поправки шага 2), метрика — OOS QLIKE шагов 2-6 (val 2017-2021) с поправкой Б0.1 (s2 по val-году шага t-1). Шаги 7-9 и тест не считаются:
тестовые прогнозы fit отбрасываются, не сохраняются и не оцениваются.
python3 scripts/analysis/d16_8_run.py --workers 8"""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

CANDS = {  # tag: (h, cfg, blocks) — как final2_step10.sh / final2_opt.sh
    'D12_h22': (22, {'v9': True, 'linear_lambda': 1}, ''),
    'D12_h5': (5, {'v9': True, 'harf': True, 'huber': True}, 'd7'),
    'F2m_h5': (5, {'v9': True, 'harf': True, 'mono': True, 'mono_sign': -1}, 'd7'),
}
SEEDS = (42, 43, 44)
VARS = ('native', 'qlike')
G = {}


def task(a):
    tag, model, step, seed, var = a
    h, cfg0, _ = CANDS[tag]
    A, cols = G['built'][tag]
    cfg = dict(cfg0, es_qlike=(var == 'qlike'))
    D = L.step_data(A, cols, h, step, model, cfg, 'Bv2b')
    if cfg.get('mono_sign') == -1:  # как tune_L2_optuna._sd
        D['mono'] = [-m for m in D['mono']]
    ref = L.make_ref(model, A, cols, h, cfg, 'Bv2b') if step > 1 else None
    t0 = time.time()
    pv, _pt, bi, _, _ = L.fit(model, G['P'][tag][(model, step)], D, cfg, seed=seed, ref=ref)
    del _pt  # тест не сохраняется и не оценивается
    trees = bi + 1 if model == 'xgb' else bi
    d = D['parts']['va']
    V = pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': D['vy'], 'step': step, 'model': model, 'seed': seed, 'es': var,
                      'rv_actual': D['y']['va'], 'rv_pred': np.clip(np.exp(pv), 1e-10, None), 'har_ln': D['off']['va']})
    return dict(tag=tag, h=h, model=model, step=step, seed=seed, es=var, best_iteration=int(bi), trees=int(trees), sec=round(time.time() - t0, 1)), V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cands', default=','.join(CANDS))
    a = ap.parse_args()
    cands = a.cands.split(',')
    t0 = time.time()
    G.update(built={}, P={})
    for tag in cands:
        h, cfg, blocks = CANDS[tag]
        A, FC = L.load_base(None, tuple(x for x in blocks.split(',') if x), h)
        A2, cols, _, _ = L.build_features(A, FC, h, cfg)
        G['built'][tag] = (A2, cols)
        G['P'][tag] = L.selected_params(str(L.REGEN / f'{tag}_selected_h{h}.csv'), h)
        print(f'{tag}: h={h}, cfg={cfg}, матрица {L.FEAT.name}{"+" + blocks if blocks else ""}, признаков {len(cols)}', flush=True)
    tasks = [(t, m, s, sd, v) for s in range(6, 0, -1) for t in cands for m in ('lgbm', 'xgb') for sd in SEEDS for v in VARS]
    print(f'задач {len(tasks)}, воркеров {a.workers}, старт {time.strftime("%T")}', flush=True)
    rows, vals = [], []
    with mp.get_context('fork').Pool(a.workers) as P:
        for i, (r, V) in enumerate(P.imap_unordered(task, tasks), 1):
            rows.append(r); vals.append(V)
            if i % 12 == 0:
                print(f'  {i}/{len(tasks)} {time.time() - t0:.0f} с', flush=True)
    T = pd.DataFrame(rows).sort_values(['tag', 'model', 'es', 'seed', 'step'])
    for tag in cands:
        h = CANDS[tag][0]
        T[T.tag == tag].to_csv(L.REGEN / f'D16_8_h{h}_{tag}_trees.csv', index=False)
        Vt = pd.concat([v for v, r in zip(vals, rows) if r['tag'] == tag], ignore_index=True)  # шаги 1-6, val 2016-2021
        Vt.sort_values(['es', 'model', 'seed', 'step', 'date', 'ticker']).to_parquet(L.REGEN / f'D16_8_h{h}_{tag}_val.parquet', index=False)
    print(f'готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
