#!/usr/bin/env python3
"""Д22, слой 2: h=1 — Д8.12 на HAR-X⁺ (ln HAR-X⁺₁ модели шага как признак; на обучении — внутри выборки, как HAR-J в Д8.12),
признаки v3 + календарь окна h=1 (как слой 1) + RRV PK/GK d/w/m + ночной гэп; винзоризация цели 1/99 по тикеру (l2_lib cfg wins);
переподбор по протоколу B v2 (tune_L2.py: ЛГК LGBM-100 / XGB-40 с отсечением, шаги 1-6 подбор, 7-9 точка шага 6; max_bin 255).
Режимы:
  --tune     протокол B v2 (tag D22_h1) -> data/regen/D22_h1_{xgb,lgbm}_h1{,_top5,_seeds4344}_val.parquet, тест -> test_locked/D22_h1/
  --step10   шаг 10 (обучение <= 2024, val 2025, тест 2026; точка шага 6; seed 42/43/44) -> data/regen/D22/step10/, test_locked/D22_h1/step10/
  --clf      классификатор направления P(ln RV_h > ln HAR-X⁺_h), h ∈ {5, 22}: LightGBM фиксированной конфигурации (без ранней остановки на
             годе оценки), обучение — строки < vy с эмбарго горизонта, seed 42/43/44 -> data/regen/D22/clf_h{h}_{val,test}.parquet (p по seed)
  --build    hyb₁ (среднее ln XGB и ln LGBM, по seed), r̂₁ = ln hyb₁ − ln HAR-X⁺₁, m̂_t -> data/regen/D22/L2_h1_{val,test}.parquet
python3 scripts/analysis/d22_h1.py --tune --workers 4"""
import os
os.environ['L2_FEAT'] = '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3_rrv.parquet'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, json, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
import d22_harxp as P

R = L.REGEN
OUT = R / 'D22'
TAG = 'D22_h1'
LOCK = R / 'test_locked' / TAG
CFG = {'harf': True, 'wins': True}
EXTRA = ['rrv_pk_d', 'rrv_pk_w', 'rrv_pk_m', 'rrv_gk_d', 'rrv_gk_w', 'rrv_gk_m', 'r2_on'] + [f'{f}@1' for f in P.CAL_FIELDS]
SEEDS = (42, 43, 44)

# ---------------------------------------------------------------- подмены l2_lib (только в этом процессе)
_har_ln0, _split0, _bf0, _lb0 = L.har_ln, L.split, L.build_features, L.load_base
_LNCOL = {'v': 'hxp_1_{k}'}


def har_ln_x(tp, parts, h, hc, w=None, huber=False, y=None):
    """ln HAR-X⁺ модели шага k = (последний год обучения + 1) − 2015 на строках каждой части (обучение — внутри выборки)."""
    k = int(tp['year'].max()) + 1 - 2015
    col = _LNCOL['v'].format(k=k)
    return [d[col].values for d in parts]


def _split10(A, h, step):
    if step != 10:
        return _split0(A, h, step)
    t = L.tcol(h)
    tr = A[A.year < 2026]
    return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026


def load_base_x(matrix=None, blocks=(), h=None):
    A, fc = _lb0(matrix, blocks, h)
    for nm in ('HXP', 'HJ'):
        f = OUT / f'{nm}_lnall.parquet'
        if f.exists():
            A = A.merge(pd.read_parquet(f), on=['date', 'ticker'], how='left', validate='one_to_one')
    assert A['hxp_1_1'].notna().all()
    return A, fc


def build_features_x(A, FC, h, cfg):
    A2, cols, cats, mono = _bf0(A, FC, h, cfg)
    X = P.build_terms(A2, EXTRA)
    for c in EXTRA:
        A2[c] = X[c].values
    return A2, cols + [c for c in EXTRA if c not in cols], cats, mono


def patch():
    L.har_ln, L.split, L.build_features, L.load_base = har_ln_x, _split10, build_features_x, load_base_x


# ---------------------------------------------------------------- режимы
def tune(workers):
    patch()
    import tune_L2 as T
    sys.argv = ['tune_L2.py', '--h', '1', '--cfg', json.dumps(CFG), '--space', 'B', '--workers', str(workers), '--tag', TAG]
    T.main()


G = {}


def _job10(a):
    m, seed = a
    D = L.step_data(G['A'], G['cols'], 1, 10, m, CFG, TAG)
    ref = L.make_ref(m, G['A'], G['cols'], 1, CFG, TAG)
    pv, pt, bi, _, _ = L.fit(m, G['P'][m], D, CFG, seed=seed, ref=ref)
    out = []
    for part, pred, yr in (('va', pv, 2025), ('te', pt, 2026)):
        d = D['parts'][part]
        out.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': yr, 'step': 10, 'seed': seed, 'rank': 1,
                                 'rv_actual': d[L.tcol(1)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return m, seed, bi, out[0], out[1]


def step10(workers):
    patch()
    from tune_L2 import SPACEB
    sel = L.selected_params(TAG, 1)
    Pm = {m: {k: sel[(m, 6)][k] for k, *_ in SPACEB[m]} for m in ('xgb', 'lgbm')}
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, 1, CFG)
    G.update(A=A2, cols=cols, P=Pm)
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job10, [(m, s) for m in ('lgbm', 'xgb') for s in SEEDS], chunksize=1)
    od, lock = OUT / 'step10', LOCK / 'step10'
    od.mkdir(parents=True, exist_ok=True); lock.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        rr = [r for r in res if r[0] == m]
        pd.concat([r[3] for r in rr]).to_parquet(od / f'{TAG}_{m}_h1_s10_val.parquet', index=False)
        pd.concat([r[4] for r in rr]).to_parquet(lock / f'{TAG}_{m}_h1_s10_test.parquet', index=False)
    pd.DataFrame([dict(model=r[0], seed=r[1], best_iteration=r[2]) for r in res]).to_csv(od / f'{TAG}_h1_s10_trees.csv', index=False)
    print('шаг 10 готов: ' + ', '.join(f'{r[0]}/s{r[1]}: {r[2]}' for r in res), flush=True)


# ---------------------------------------------------------------- классификатор направления
CLF_PRM = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, max_bin=255, verbosity=-1, num_threads=1, force_col_wise=True)
CLF_TREES = 300


def _job_clf(a):
    import lightgbm as lgb
    h, step, seed = a
    A, cols = G['A'], G['cols']
    tg = str(h)
    vy, ty = (2015 + step, 2016 + step)
    tgt = L.tcol(h)
    base = A[f"{G['base']}_{tg}_{step}"].values
    ok = A[tgt].notna().values
    tend = A[f'_tend_{tg}'].values
    tr = (A.year.values < vy) & ok & (tend < vy)
    X = L.clean(np.column_stack([A[cols].values.astype(np.float64), base])).astype(np.float32)
    ylab = (np.log(A[tgt].values + 1e-10) > base).astype(float)
    b = lgb.train(dict(CLF_PRM, seed=seed), lgb.Dataset(X[tr], ylab[tr], params={'max_bin': 255, 'verbosity': -1}), CLF_TREES)
    out = []
    for yr in (vy, ty):
        m = (A.year.values == yr)
        out.append(pd.DataFrame({'date': A.loc[m, 'date'].values, 'ticker': A.loc[m, 'ticker'].values, 'year': yr, 'step': step, 'seed': seed,
                                 'p': b.predict(X[m])}))
    return h, step, seed, out[0], out[1]


def clf(workers, hs=(5, 22), base='hxp', tag='clf', steps=range(1, 11)):
    A, FC = load_base_x()
    A = P.add_targets(A)
    A2, cols, _, _ = build_features_x(A, FC, 1, {})
    G.update(A=A2, cols=cols, base=base)
    jobs = [(h, s, sd) for h in hs for s in steps for sd in SEEDS]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job_clf, jobs, chunksize=1)
    for h in hs:
        for i, k in ((3, 'val'), (4, 'test')):
            d = pd.concat([r[i] for r in res if r[0] == h], ignore_index=True)
            d.to_parquet((OUT if k == 'val' else LOCK) / f'{tag}_h{h}_{k}.parquet', index=False)
    print('классификатор готов', flush=True)


# ---------------------------------------------------------------- абляции слоя 2 (точки подбора D22_h1, без нового перебора; шаги 1-6)
ABL = {'HJ': dict(lncol='hj_1_{k}', cfg=CFG, extra=True),        # минус слой 1: основа Д8.12 — HAR-J (тот же эмбарго), признаки и точки D22
       'nofeat': dict(lncol='hxp_1_{k}', cfg=CFG, extra=False),  # минус новые признаки слоя 2 (календарь, RRV, гэп)
       'nowins': dict(lncol='hxp_1_{k}', cfg={'harf': True}, extra=True)}  # минус винзоризация цели


def _job_abl(a):
    var, m, step, seed = a
    v = ABL[var]
    _LNCOL['v'] = v['lncol']
    cols = G['cols'] if v['extra'] else [c for c in G['cols'] if c not in EXTRA]
    D = L.step_data(G['A'], cols, 1, step, m, v['cfg'], TAG)
    ref = L.make_ref(m, G['A'], cols, 1, v['cfg'], TAG) if step > 1 else None
    pv, pt, bi, _, _ = L.fit(m, G['sel'][(m, step)], D, v['cfg'], seed=seed, ref=ref)
    out = []
    for part, pred in (('va', pv), ('te', pt)):
        d = D['parts'][part]
        out.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': d['year'].values, 'step': step, 'seed': seed,
                                 'model': m, 'rv_actual': d[L.tcol(1)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return var, out[0], out[1]


def abl(workers, variants=('HJ', 'nofeat', 'nowins'), steps=range(1, 7)):
    patch()
    from tune_L2 import SPACEB
    sel = L.selected_params(TAG, 1)
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, 1, CFG)
    G.update(A=A2, cols=cols, sel={(m, s): {k: sel[(m, s)][k] for k, *_ in SPACEB[m]} for m in ('xgb', 'lgbm') for s in steps})
    jobs = [(v, m, s, sd) for v in variants for m in ('lgbm', 'xgb') for s in steps for sd in SEEDS]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job_abl, jobs, chunksize=1)
    for v in variants:
        pd.concat([r[1] for r in res if r[0] == v]).to_parquet(OUT / f'abl_{v}_h1_val.parquet', index=False)
        pd.concat([r[2] for r in res if r[0] == v]).to_parquet(LOCK / f'abl_{v}_h1_test.parquet', index=False)
    print('абляции слоя 2 готовы', flush=True)


# ---------------------------------------------------------------- hyb₁, r̂₁, m̂
def _read(m, k, sfx):
    d = (R if k == 'val' else LOCK) / f'{TAG}_{m}_h1{sfx}_{k}.parquet'
    return pd.read_parquet(d)


def build(src=None, base='HXP', out='L2_h1'):
    """L2_h1_{val,test}: date, ticker, year, step, seed, rv_actual, ln_hyb1, ln_hxp1, r1, m1 (шаги 1-10; seed 42/43/44 и 0 = среднее
    6 ln-прогнозов). Шаг 10 — из step10/. src — префикс файлов абляции abl_{var}_h1 (столбцы model, seed), base — основа r̂₁ (HXP | HJ)."""
    LOCK.mkdir(parents=True, exist_ok=True)
    for k in ('val', 'test'):
        fr = []
        for m in (('xgb', 'lgbm') if src is None else ()):
            a = _read(m, k, ''); a['seed'] = 42
            b = _read(m, k, '_seeds4344')
            s10 = pd.read_parquet((OUT / 'step10' if k == 'val' else LOCK / 'step10') / f'{TAG}_{m}_h1_s10_{k}.parquet')
            d = pd.concat([a, b, s10], ignore_index=True)[['date', 'ticker', 'year', 'step', 'seed', 'rv_actual', 'rv_pred']]
            fr.append(d.assign(model=m))
        d = pd.concat(fr, ignore_index=True) if src is None else pd.read_parquet((OUT if k == 'val' else LOCK) / f'{src}_{k}.parquet')
        d['ln'] = np.log(d.rv_pred)
        w = d.pivot_table(index=['date', 'ticker', 'year', 'step', 'seed'], columns='model', values='ln').reset_index()
        w['ln_hyb1'] = (w.xgb + w.lgbm) / 2
        avg = w.groupby(['date', 'ticker', 'year', 'step'], as_index=False)['ln_hyb1'].mean().assign(seed=0)  # seed 0 = среднее 3 seed
        H = pd.concat([w[['date', 'ticker', 'year', 'step', 'seed', 'ln_hyb1']], avg], ignore_index=True)
        hx = pd.read_parquet((OUT if k == 'val' else R / 'test_locked' / f'D22_{base}') / f'{base}_1_{k}.parquet')[['date', 'ticker', 'step', 'rv_actual', 'rv_pred']]
        hx['ln_hxp1'] = np.log(hx.rv_pred)
        H = H.merge(hx.drop(columns='rv_pred'), on=['date', 'ticker', 'step'], how='left', validate='many_to_one')
        H['r1'] = H.ln_hyb1 - H.ln_hxp1
        H['m1'] = H.groupby(['date', 'seed', 'step'])['r1'].transform('mean')
        H.to_parquet((OUT if k == 'val' else LOCK) / f'{out}_{k}.parquet', index=False)
        print(k, len(H), H.groupby('seed').size().to_dict(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', action='store_true'); ap.add_argument('--step10', action='store_true')
    ap.add_argument('--clf', action='store_true'); ap.add_argument('--build', action='store_true')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--abl', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    if a.tune:
        tune(a.workers)
    if a.step10:
        step10(a.workers)
    if a.clf:
        clf(a.workers)
    if a.build:
        build()
    if a.abl:
        abl(a.workers)
        build('abl_HJ_h1', 'HJ', 'abl_HJ_L2_h1')
        clf(a.workers, base='hj', tag='abl_HJ_clf', steps=range(1, 7))


if __name__ == '__main__':
    main()
