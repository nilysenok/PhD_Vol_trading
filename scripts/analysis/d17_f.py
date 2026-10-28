#!/usr/bin/env python3
"""Д17 (агент F, третий круг): рычаги Д17.2 (RRV), Д17.5 (винзоризация цели), Д17.4 (итерированный HAR) на текущих финалистах.

Финалисты и их cfg (l2_lib, без изменений общих файлов конвейера):
  h=1  d812     {"harf": true}                                   точка B v2b ранга 1 шага (Bv2b_selected_h1.csv -> Bv2b_grid_*), XGB
  h=5  D12_h5   {"v9": true, "harf": true, "huber": true} + Д7     точки D12_h5_selected_h5.csv, XGB и LGBM (wrap2)
  h=5  F2m_h5   {"v9": true, "harf": true, "mono": true, "mono_sign": -1} + Д7   точки F2m_h5_selected_h5.csv, XGB и LGBM (wrap2)
  h=22 D12_h22  {"v9": true, "linear_lambda": 1}                  точки D12_h22_selected_h22.csv, XGB (seeds3)
  V0_h{h}       {} — чистый бустинг на ln RV; точки Bv2b (h=1) / Bv3_selected_h{h}.csv (h=5, 22), XGB и LGBM
Варианты (одна точка на шаг, без переподбора; base — тот же прогон без рычага, сравнение «яблоки к яблокам»):
  base       — финалист как есть
  rrv_pk     — + признаки rrv_pk_d/w/m (ln Parkinson, features_pit_v3_rrv)
  rrv_gk     — + признаки rrv_gk_d/w/m (ln Garman–Klass)
  tgt_pk     — обучение на ln RRV_PK (цель rrv_pk_target_*), ранняя остановка по ln RRV_PK val-года; прогноз RV =
               exp(ŷ + c_тикер), c_тикер = среднее ln(RV_цель) − ln(RRV_цель) по обучающему окну шага (только прошлое);
               оценка по RV. Для V9 смещение — ln HAR-J на RV, как у финалиста.
  wins       — Д17.5: ln-цель обучения винзоризуется 1/99 по тикеру на обучающем окне (l2_lib cfg wins), оценка по сырой RV
  iterhar    — Д17.4 (h=5/22): признак ln итерированного HAR-прогноза среднего RV за [t+1, t+h] (HAR(1) на ln, МНК на
               обучающем окне шага; итерация по дням с протяжкой прогнозов в rv_d/w/m). Ряд -> data/regen/iterhar_h{h}.parquet
Шаги 1–6 (val 2016–2021); тестовые части шага не считаются вовсе (X теста подменён одной строкой val) — подтверждающий
и тест не открываются. Seeds 42/43/44.
Выход: data/regen/D17F_h{h}_{design}_{variant}_{model}_val.parquet (date, ticker, year, step, seed, rv_actual, rv_pred).
python3 scripts/analysis/d17_f.py --h 22 --designs D12_h22 --variants base rrv_pk ... [--seeds 42 43 44] [--workers 3]
"""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path

os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
import l2_lib as L

R = L.REGEN
RRV = R / 'features_pit_v3_rrv.parquet'
DESIGNS = {
    'd812': dict(h=1, cfg={'harf': True}, blocks=(), models=('xgb',), sel='Bv2b'),
    'D12_h5': dict(h=5, cfg={'v9': True, 'harf': True, 'huber': True}, blocks=('d7',), models=('xgb', 'lgbm'), sel='D12_h5'),
    'F2m_h5': dict(h=5, cfg={'v9': True, 'harf': True, 'mono': True, 'mono_sign': -1}, blocks=('d7',), models=('xgb', 'lgbm'), sel='F2m_h5'),
    'D12_h22': dict(h=22, cfg={'v9': True, 'linear_lambda': 1}, blocks=(), models=('xgb',), sel='D12_h22'),
    # чистый бустинг V0 (ln RV, без HAR-J): точки B v2b (h=1, как D12_h1_V0) / Bv3 (h=5, 22), XGB и LGBM
    'V0_h1': dict(h=1, cfg={}, blocks=(), models=('xgb', 'lgbm'), sel='Bv2b'),
    'V0_h5': dict(h=5, cfg={}, blocks=(), models=('xgb', 'lgbm'), sel='Bv3'),
    'V0_h22': dict(h=22, cfg={}, blocks=(), models=('xgb', 'lgbm'), sel='Bv3'),
}
VARIANTS = ('base', 'rrv_pk', 'rrv_gk', 'tgt_pk', 'wins', 'iterhar')
PCOLS = ('num_leaves', 'learning_rate', 'bagging_fraction', 'feature_fraction', 'min_data_in_leaf', 'reg_alpha', 'reg_lambda',
         'max_depth', 'subsample', 'colsample_bytree', 'min_child_weight')
G = {}


def rrv_tcol(h, m='pk'):
    return f'rrv_{m}_target_h1' if h == 1 else f'rrv_{m}_target_mean_h{h}'


# ---------------------------------------------------------------- Д17.4: итерированный HAR
def iterhar(A, h, steps=range(1, 11)):
    """ln итерированного HAR-прогноза среднего RV за [t+1, t+h] для каждой строки A и каждого шага (параметры — МНК на
    обучающем окне шага, цель ln RV(t+1)). История — RV дня t и 21 прошлых строк тикера (колонка rv; только <= t)."""
    o = A.sort_values(['ticker', 'date'])
    rv = o['rv'].clip(lower=1e-10)
    hist = np.column_stack([rv.groupby(o['ticker']).shift(j).values for j in range(21, -1, -1)])  # [t-21 .. t]
    first = o.groupby('ticker').cumcount().values  # min_periods = 1 как у rv_w/rv_m
    hist = np.where(np.isnan(hist), np.nan, hist)

    def feats(H):
        d = H[:, -1]
        w = np.nanmean(H[:, -5:], axis=1)
        m = np.nanmean(H, axis=1)
        return np.column_stack([np.ones(len(H)), np.log(d), np.log(w), np.log(m)])

    X0 = feats(hist)
    y1 = np.log(o['rv_target_h1'].values + 1e-10)
    yr = o['year'].values
    out = []
    for st in steps:
        ty = L.TEST_YEARS[st - 1] if st <= len(L.TEST_YEARS) else 2026
        vy = ty - 1
        trm = (yr < vy) & np.isfinite(y1) & np.isfinite(X0).all(1)
        beta = np.linalg.lstsq(X0[trm], y1[trm], rcond=None)[0]
        H = hist.copy()
        acc = np.zeros(len(H))
        for _ in range(h):
            f = np.exp(feats(H) @ beta)
            acc += f
            H = np.column_stack([H[:, 1:], f])
        out.append(pd.DataFrame({'date': o['date'].values, 'ticker': o['ticker'].values, 'step': st,
                                 'iterhar_ln': np.log(acc / h)}))
    del first
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- данные шага
def params(design, model, step):
    d = DESIGNS[design]
    if d['sel'] == 'Bv2b':
        S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
        pt = int(S[(S.model == model) & (S.step == step)].point.iloc[0])
        g = pd.concat([pd.read_csv(R / f'Bv2b_grid_{x}.csv') for x in ('xgb40', 'xgb100', 'lgbm100')])
        r = g[(g.model == model) & (g.point == pt)].iloc[0]
        return {k: r[k] for k in PCOLS if k in r and pd.notna(r[k])}
    s = pd.read_csv(R / f"{d['sel']}_selected_h{d['h']}.csv")
    r = s[(s.model == model) & (s.step == step) & (s.seed == 42)].iloc[0] if 'seed' in s else s[(s.model == model) & (s.step == step)].iloc[0]
    return {k: r[k] for k in PCOLS if k in r and pd.notna(r[k])}


def augment(D, A, step, variant, h):
    """Добавка признака iterhar (значения шага) в X всех частей."""
    if 'iterhar' not in variant.split('+'):
        return D
    ih = G['ih'][G['ih'].step == step][['date', 'ticker', 'iterhar_ln']]
    for k, d in D['parts'].items():
        v = d[['date', 'ticker']].merge(ih, on=['date', 'ticker'], how='left')['iterhar_ln'].values
        D['X'][k] = np.column_stack([D['X'][k], L.clean(v)]).astype(np.float32)
    D['names'] = D['names'] + ['iterhar_ln']
    D['mono'] = D['mono'] + [0]
    return D


def sd(design, variant, model, step, ex_step=None):
    d = DESIGNS[design]
    A, cols, cfg, h = G['A'][design], G['cols'][(design, variant)], G['cfg'][(design, variant)], d['h']
    D = L.step_data(A, cols, h, step, model, cfg, 'Bv2b', ex_step=ex_step)
    if cfg.get('mono_sign') == -1:
        D['mono'] = [-m for m in D['mono']]
    return augment(D, A, step, variant, h)


_REF = {}


def ref(design, variant, model, step):
    """Бины окна шага 1 — один раз на (дизайн, набор признаков, модель) в процессе; метка на бины не влияет."""
    if step == 1:
        return None
    key = (design, tuple(G['cols'][(design, variant)]), 'iterhar' in variant.split('+'), model)
    if key not in _REF:
        _REF[key] = _make_ref(design, variant, model)
    return _REF[key]


def _make_ref(design, variant, model):
    cfg = G['cfg'][(design, variant)]
    D = sd(design, variant, model, 1, ex_step=2)
    if model == 'xgb':
        return xgb.QuantileDMatrix(D['X']['tr'], D['ytr_ln'], max_bin=L.MAX_BIN, feature_types=L._ref_xgb(D, L.MAX_BIN))
    dp = {'max_bin': L.MAX_BIN, 'feature_pre_filter': False, 'verbosity': -1}
    if cfg.get('linear_lambda') is not None:
        dp['linear_tree'] = True
    return lgb.Dataset(D['X']['tr'], D['ytr_ln'], params=dp, categorical_feature='auto', free_raw_data=False)


def task(a):
    """Одна (дизайн, вариант, модель, шаг): данные шага и бины строятся один раз, затем обучения по всем seed."""
    design, variant, model, step, seeds = a
    t0 = time.time()
    h = DESIGNS[design]['h']
    cfg = G['cfg'][(design, variant)]
    D = sd(design, variant, model, step)
    if not G.get('final'):  # отбор: тестовую часть не трогаем — подмена одной строкой val
        D['X']['te'] = D['X']['va'][:1]
        D['off']['te'] = D['off']['va'][:1]
    c = None
    if variant == 'tgt_pk':
        tc = rrv_tcol(h)
        tp, va = D['parts']['tr'], D['parts']['va']
        lr = np.log(tp[tc].values.clip(1e-10))
        D['ytr_ln'] = lr
        lrv = np.log(tp[L.tcol(h)].values + 1e-10) - lr
        cmap = pd.Series(lrv).groupby(tp['ticker'].values).mean()
        c = va['ticker'].map(cmap).fillna(float(np.mean(lrv))).values
        yva_rv = D['y']['va']
        D['y']['va'] = va[tc].values.clip(1e-10)
    rf, p = ref(design, variant, model, min(step, 9)), params(design, model, step)
    va, te = D['parts']['va'], D['parts']['te']
    out, outt = [], []
    for seed in seeds:
        pv, pt, bi, _, _ = L.fit(model, p, D, cfg, seed=seed, ref=rf)
        if variant == 'tgt_pk':
            pv = pv + c
        out.append(pd.DataFrame({'date': va['date'].values, 'ticker': va['ticker'].values, 'year': D['vy'], 'step': step, 'seed': seed,
                                 'best_iteration': bi, 'rv_actual': yva_rv if variant == 'tgt_pk' else D['y']['va'],
                                 'rv_pred': np.clip(np.exp(pv), 1e-10, None), 'sec': time.time() - t0}))
        if G.get('final'):
            outt.append(pd.DataFrame({'date': te['date'].values, 'ticker': te['ticker'].values, 'year': D['ty'], 'step': step, 'seed': seed,
                                      'best_iteration': bi, 'rv_actual': D['y']['te'], 'rv_pred': np.clip(np.exp(pt), 1e-10, None)}))
    if G.get('final'):
        return pd.concat(out, ignore_index=True), pd.concat(outt, ignore_index=True)
    return pd.concat(out, ignore_index=True)


def setup(designs, variants):
    G.update(A={}, cols={}, cfg={})
    rr = pd.read_parquet(RRV)
    new = [c for c in rr.columns if c.startswith('rrv_')]
    for ds in designs:
        d = DESIGNS[ds]
        A, FC = L.load_base(None, d['blocks'], d['h'])
        A = A.merge(rr[['date', 'ticker'] + new], on=['date', 'ticker'], how='left', validate='one_to_one')
        A2, cols, _, _ = L.build_features(A, FC, d['h'], d['cfg'])
        G['A'][ds] = A2
        for v in variants:  # комбинация рычагов — через «+», например rrv_gk+iterhar
            cfg = dict(d['cfg'])
            c = list(cols)
            parts = v.split('+')
            if 'rrv_pk' in parts:
                c += ['rrv_pk_d', 'rrv_pk_w', 'rrv_pk_m']
            if 'rrv_gk' in parts:
                c += ['rrv_gk_d', 'rrv_gk_w', 'rrv_gk_m']
            if 'wins' in parts:
                cfg['wins'] = True
            G['cols'][(ds, v)], G['cfg'][(ds, v)] = c, cfg


def write_final(boost, h, tasks, res):
    """{boost}_{m}_h{h}_val.parquet (seed 42) и _seeds4344; тест — test_locked/{boost}/ (не оценивается)."""
    lock = R / 'test_locked' / boost
    lock.mkdir(parents=True, exist_ok=True)
    cols = ['date', 'ticker', 'year', 'step', 'trial', 'seed', 'rank', 'rv_actual', 'rv_pred']
    for m in sorted({t[2] for t in tasks}):
        rr = [r for t, r in zip(tasks, res) if t[2] == m]
        for k, ix, d in (('val', 0, R), ('test', 1, lock)):
            x = pd.concat([r[ix] for r in rr], ignore_index=True).assign(trial=-1, rank=1)[cols]
            x = x.sort_values(['seed', 'step', 'date', 'ticker'])
            x[x.seed == 42].to_parquet(d / f'{boost}_{m}_h{h}_{k}.parquet', index=False)
            x[x.seed != 42].to_parquet(d / f'{boost}_{m}_h{h}_seeds4344_{k}.parquet', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--designs', nargs='+', required=True)
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--steps', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--final', default=None, help='имя boost: шаги 1-9 (7-9 — точка шага 6), val и тест в формате кандидата '
                    'третьего круга; один дизайн и один вариант; метрики не печатаются')
    a = ap.parse_args()
    setup(a.designs, a.variants)
    if a.final:
        G['final'] = True
        a.steps = list(range(1, 10))
    if any('iterhar' in v.split('+') for v in a.variants):
        A0 = G['A'][a.designs[0]]
        p = R / f'iterhar_h{a.h}.parquet'
        G['ih'] = iterhar(A0, a.h)
        G['ih'].to_parquet(p, index=False)
        print(f'iterhar_h{a.h}: {len(G["ih"])} строк -> {p}', flush=True)
    # тяжёлые шаги первыми; бины кэшируются в процессе воркера
    tasks = [(ds, v, m, st, tuple(a.seeds)) for st in sorted(a.steps, reverse=True) for ds in a.designs for v in a.variants
             for m in DESIGNS[ds]['models']]
    t0 = time.time()
    print(f'D17F h={a.h}: {len(tasks)} обучений, воркеров {a.workers}', flush=True)
    with mp.get_context('fork').Pool(a.workers) as P:
        res = P.map(task, tasks, chunksize=1)
    if a.final:
        write_final(a.final, a.h, tasks, res)
        print(f'{a.final}: готово за {time.time() - t0:.0f} с', flush=True)
        return
    by = {}
    for t, r in zip(tasks, res):
        by.setdefault(t[:3], []).append(r)
    for (ds, v, m), rs in by.items():
        p = R / f'D17F_h{a.h}_{ds}_{v}_{m}_val.parquet'
        new = pd.concat(rs, ignore_index=True)
        new = new.drop_duplicates(['date', 'ticker', 'step', 'seed'])
        if p.exists():  # дописывание других seed/шагов
            old = pd.read_parquet(p)
            old = old[~old.set_index(['step', 'seed']).index.isin(new.set_index(['step', 'seed']).index)]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(['seed', 'step', 'date', 'ticker']).to_parquet(p, index=False)
    print(f'готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
