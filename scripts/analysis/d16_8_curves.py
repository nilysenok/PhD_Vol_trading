#!/usr/bin/env python3
"""Д16.8, диагностика h=22: кривые val-метрик остаточного ученика V9 (D12_h22, точки optuna-30, seed 42) по числу деревьев без ранней
остановки (300 деревьев): l2 по остаткам ln RV − ln HAR-J и QLIKE гибрида exp(ln HAR-J + f) на уровне RV; k=0 — только константа
(среднее остатка окна обучения), «HAR-J» — exp(ln HAR-J) без константы. Только val шагов 1-6.
python3 scripts/analysis/d16_8_curves.py [--workers 8]"""
import argparse, multiprocessing as mp, os, sys
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
import lightgbm as lgb, xgboost as xgb

TAG, H, CFG = 'D12_h22', 22, {'v9': True, 'linear_lambda': 1, 'nmax': 300}
K = 300
G = {}
_cap = {}
_lt, _xt = lgb.train, xgb.train


def _lgb_train(*a, **k):
    # без callback ранней остановки и с keep_training_booster: иначе lgb.train обрезает модель до best_iteration (model_to_string)
    k['callbacks'] = []; k['keep_training_booster'] = True
    b = _lt(*a, **k); _cap['b'] = b; return b


def _xgb_train(*a, **k):
    b = _xt(*a, **k); _cap['b'] = b; return b


def task(a):
    model, step = a
    L.PAT = 10 ** 9  # без ранней остановки
    lgb.train, xgb.train = _lgb_train, _xgb_train
    A, cols = G['A'], G['cols']
    D = L.step_data(A, cols, H, step, model, CFG, 'Bv2b')
    ref = L.make_ref(model, A, cols, H, CFG, 'Bv2b') if step > 1 else None
    L.fit(model, G['P'][(model, step)], D, CFG, seed=42, ref=ref)
    b = _cap['b']
    y, off = D['y']['va'], D['off']['va']
    m = y > 1e-12
    yl = np.log(y + 1e-10) - off
    c0 = float(np.mean(D['ytr_ln'] - D['off']['tr']))
    rows = [dict(model=model, step=step, k=-1, l2=float(np.mean(yl[m] ** 2)), qlike=L.qlike(y[m], np.exp(off[m]))),
            dict(model=model, step=step, k=0, l2=float(np.mean((yl[m] - c0) ** 2)), qlike=L.qlike(y[m], np.exp(off[m] + c0)))]
    if model == 'xgb':
        dm = xgb.DMatrix(D['X']['va'], feature_types=L._ref_xgb(D, L.MAX_BIN))
        pr = lambda k: b.predict(dm, iteration_range=(0, k), output_margin=True)
    else:
        pr = lambda k: b.predict(D['X']['va'], num_iteration=k, raw_score=True)
    for k in range(1, K + 1):
        f = pr(k)[m]
        rows.append(dict(model=model, step=step, k=k, l2=float(np.mean((yl[m] - f) ** 2)), qlike=L.qlike(y[m], np.exp(off[m] + f))))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    A, FC = L.load_base(None, (), H)
    A2, cols, _, _ = L.build_features(A, FC, H, CFG)
    G.update(A=A2, cols=cols, P=L.selected_params(str(L.REGEN / f'{TAG}_selected_h{H}.csv'), H))
    with mp.get_context('fork').Pool(a.workers) as P:
        res = P.map(task, [(m, s) for m in ('lgbm', 'xgb') for s in range(1, 7)], chunksize=1)
    C = pd.DataFrame([r for x in res for r in x])
    C.to_csv(L.REGEN / 'D16_8_h22_curves.csv', index=False, float_format='%.7g')
    S = []
    for (m, s), g in C.groupby(['model', 'step']):
        gk = g[g.k >= 1]
        S.append(dict(model=m, step=s, qlike_harj=g[g.k == -1].qlike.iloc[0], qlike_k0=g[g.k == 0].qlike.iloc[0], qlike_k1=gk.qlike.iloc[0],
                      k_min_l2=int(gk.loc[gk.l2.idxmin()].k), k_min_qlike=int(gk.loc[gk.qlike.idxmin()].k), qlike_min=gk.qlike.min(),
                      qlike_k50=gk[gk.k == 50].qlike.iloc[0], qlike_k300=gk[gk.k == 300].qlike.iloc[0],
                      l2_k0=g[g.k == 0].l2.iloc[0], l2_min=gk.l2.min()))
    S = pd.DataFrame(S)
    S.to_csv(L.REGEN / 'D16_8_h22_curves_summary.csv', index=False, float_format='%.6f')
    pd.set_option('display.width', 250)
    print(S.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
