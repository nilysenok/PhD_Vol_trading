#!/usr/bin/env python3
"""Этап 5, вариант B: перебор 100 точек ЛГК (seed 42) на каждом (модель, h, шаг СВ); выбор по QLIKE валидационного года (пул).
Запуск из корня репо: python3 moex_volatility/scripts/tune_B.py [--bench | --workers 9]
Данные/шаги/qlike/clean_features — как в regen_boosting.py (walk_forward_full.py). ES: patience 50, метрика l2 на ln(rv) (rmse у xgboost)."""
import argparse, ast, json, os, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]; MV = ROOT / 'moex_volatility'; OUT = ROOT / 'data/regen'
ns = {'np': np}
for n in ast.parse((MV / 'scripts/walk_forward_full.py').read_text()).body:
    if isinstance(n, ast.FunctionDef) and n.name in ('qlike', 'clean_features'):
        exec(compile(ast.Module([n], []), 'wff', 'exec'), ns)
qlike, clean_features = ns['qlike'], ns['clean_features']
N_PTS, HS, MODELS = 100, (1, 5, 22), ('xgb', 'lgbm')
# (имя, lo, hi, шкала) по протоколу B
SPACE = {'xgb': [('max_depth', 3, 12, 'int'), ('learning_rate', .01, .2, 'log'), ('subsample', .5, 1, 'lin'), ('colsample_bytree', .4, 1, 'lin'),
                 ('min_child_weight', 1, 100, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')],
         'lgbm': [('num_leaves', 15, 255, 'int'), ('learning_rate', .01, .2, 'log'), ('bagging_fraction', .5, 1, 'lin'), ('feature_fraction', .4, 1, 'lin'),
                  ('min_data_in_leaf', 5, 200, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')]}


def lhs(d, n, seed=42):
    try:
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    except ImportError:
        rng = np.random.default_rng(seed)
        return np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(d)])


def make_grid():  # один куб (7 измерений) для обеих моделей, масштабирование по SPACE модели
    u = lhs(7, N_PTS); rows = []
    for m in MODELS:
        for i in range(N_PTS):
            r = {'model': m, 'point': i}
            for j, (k, lo, hi, sc) in enumerate(SPACE[m]):
                x = u[i, j]
                v = lo + x * (hi - lo) if sc in ('lin', 'int') else float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
                r[k] = int(round(v)) if sc in ('int', 'logint') else v
            rows.append(r)
    return pd.DataFrame(rows)


def load():
    os.chdir(MV)
    parts = [pd.read_parquet(f'data/prepared/{n}.parquet') for n in ('train', 'val', 'test')]
    fc = json.load(open('data/prepared/config.json'))['feature_cols']
    for y in range(2020, 2027):
        fp = f'data/prepared/walkforward_{y}.parquet'
        if os.path.exists(fp):
            d = pd.read_parquet(fp)
            if len(d) > 100: parts.append(d)
    ad = pd.concat(parts).sort_values('date').reset_index(drop=True)
    ad['year'] = pd.to_datetime(ad['date']).dt.year
    return ad, fc, [y for y in range(2017, 2026) if y in ad.year.values]


ALL, FC, TEST_YEARS = load()
GRID = make_grid().set_index(['model', 'point'])
_cache = {}


def step_data(h, step):
    if (h, step) not in _cache:
        _cache.clear(); ty = TEST_YEARS[step - 1]; t = f'rv_target_h{h}'
        tr = ALL[ALL.year < ty]; vy = sorted(tr.year.unique())[-1]
        tp, vi = tr[tr.year != vy].dropna(subset=[t]), tr[tr.year == vy].dropna(subset=[t])
        td = ALL[ALL.year == ty].dropna(subset=[t])
        _cache[(h, step)] = dict(vi=vi, td=td, vy=vy, ty=ty, t=t, X=[clean_features(x[FC].values) for x in (tp, vi, td)],
                                 ytp=np.log(tp[t].values + 1e-10), yvi=np.log(vi[t].values + 1e-10))
    return _cache[(h, step)]


def fit(model, pt, h, step, seed=42):
    s = step_data(h, step); p = GRID.loc[(model, pt)].to_dict(); Xtp, Xvi, _ = s['X']
    if model == 'xgb':
        m = xgb.XGBRegressor(n_estimators=2000, tree_method='hist', max_depth=int(p['max_depth']), learning_rate=p['learning_rate'], subsample=p['subsample'],
                             colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'], reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
                             verbosity=0, random_state=seed, n_jobs=1, early_stopping_rounds=50).fit(Xtp, s['ytp'], eval_set=[(Xvi, s['yvi'])], verbose=False)
        bi = int(m.best_iteration)
    else:
        if 'dtr' not in s:  # один lgb.Dataset на шаг; feature_pre_filter=False — чтобы датасет подходил любому min_data_in_leaf
            s['dtr'] = lgb.Dataset(Xtp, s['ytp'], free_raw_data=False, params={'feature_pre_filter': False, 'verbosity': -1})
            s['dva'] = lgb.Dataset(Xvi, s['yvi'], reference=s['dtr'], free_raw_data=False)
        prm = dict(objective='regression', boosting_type='gbdt', max_depth=-1, num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'],
                   bagging_fraction=p['bagging_fraction'], bagging_freq=1, feature_fraction=p['feature_fraction'], min_data_in_leaf=int(p['min_data_in_leaf']),
                   reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], verbosity=-1, seed=seed, num_threads=1, metric='l2')
        m = lgb.train(prm, s['dtr'], 2000, valid_sets=[s['dva']], callbacks=[lgb.early_stopping(50, verbose=False)])  # = LGBMRegressor, проверено (maxdiff 0.0)
        bi = int(m.best_iteration)
    return m, s, bi


def task(a):
    model, h, step, pt = a; t0 = time.time(); m, s, bi = fit(model, pt, h, step)
    q = qlike(s['vi'][s['t']].values, np.clip(np.exp(m.predict(s['X'][1])), 1e-10, None))
    return dict(model=model, h=h, step=step, point=pt, qlike_val=q, best_iteration=bi, sec=time.time() - t0)


def extra_task(a):  # прогнозы val/test для (модель, h, шаг, точка, seed); метки rank/seed добавляются в extra()
    m, h, s, pt, seed, kind, k = a; mod, sd, _ = fit(m, pt, h, s, seed); out = []
    for nm, df, X, yr in (('val', sd['vi'], sd['X'][1], sd['vy']), ('test', sd['td'], sd['X'][2], sd['ty'])):
        out.append((nm, pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': s, 'point': pt, 'seed': seed, kind: k,
            'rv_actual': df[sd['t']].values, 'rv_pred': np.clip(np.exp(mod.predict(X)), 1e-10, None)})))
    return (m, h, kind), out


def extra(workers, tf):  # этап 2: топ-5 точек по val QLIKE и 3 seed (42,43,44) лучшей точки
    tr = pd.read_csv(tf).drop_duplicates(['model', 'h', 'step', 'point']); tasks = []
    for (m, h, s), g in tr.groupby(['model', 'h', 'step']):
        top = g.sort_values('qlike_val').point.astype(int).tolist()[:5]
        tasks += [(m, h, s, pt, 42, 'rank', r + 1) for r, pt in enumerate(top)] + [(m, h, s, top[0], sd, 'seed_k', sd) for sd in (43, 44)]
    res = {}
    with Pool(workers) as P:
        for key, out in P.imap_unordered(extra_task, sorted(tasks, key=lambda t: -t[2]), chunksize=1):
            for nm, df in out: res.setdefault((*key, nm), []).append(df)
    for (m, h, kind, nm), L in res.items():
        d = pd.concat(L)
        if kind == 'rank': d.drop(columns='seed_k', errors='ignore').to_parquet(OUT / f'B_{m}_h{h}_top5_{nm}.parquet', index=False)
        else: d.to_parquet(OUT / f'B_{m}_h{h}_seeds4344_{nm}.parquet', index=False)  # seed 42 лучшей точки = B_{m}_h{h}_{nm}.parquet
    print('extra saved', flush=True)


def fin_task(a):  # детерминированный рефит выбранной точки (seed 42) + прогнозы val/test
    m, h, s, pt, q = a; mod, sd, bi = fit(m, pt, h, s)
    row = dict(model=m, h=h, step=s, point=pt, **GRID.loc[(m, pt)].to_dict(), best_iteration=bi, qlike_val=q); pr = {}
    for k, df, X, yr in (('val', sd['vi'], sd['X'][1], sd['vy']), ('test', sd['td'], sd['X'][2], sd['ty'])):
        pr[k] = pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': s,
                              'rv_actual': df[sd['t']].values, 'rv_pred': np.clip(np.exp(mod.predict(X)), 1e-10, None)})
    return row, pr


def complete(tr, h):  # все 100 точек x шаги x модели для этого h посчитаны
    return len(tr[tr.h == h].drop_duplicates(['model', 'h', 'step', 'point'])) == len(MODELS) * len(TEST_YEARS) * N_PTS


def finalize(h, workers=2):  # выбор лучших точек по val QLIKE, запись B_{m}_h{h}_{val,test}.parquet и строк B_selected.csv для h
    tr = pd.read_csv(OUT / 'B_trials.csv').drop_duplicates(['model', 'h', 'step', 'point'])
    if not complete(tr, h): return False
    tasks = [(m, h, int(s), int(g.loc[g.qlike_val.idxmin(), 'point']), g.qlike_val.min()) for (m, s), g in tr[tr.h == h].groupby(['model', 'step'])]
    with Pool(workers) as P: res = P.map(fin_task, tasks, chunksize=1)
    for m in MODELS:
        for k in ('val', 'test'): pd.concat([pr[k] for r, pr in res if r['model'] == m]).sort_values('step', kind='stable').to_parquet(OUT / f'B_{m}_h{h}_{k}.parquet', index=False)
    f = OUT / 'B_selected.csv'; old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    new = pd.concat([old[old.h != h] if len(old) else old, pd.DataFrame([r for r, _ in res])]).sort_values(['model', 'h', 'step'])
    new.to_csv(str(f) + '.tmp', index=False); os.replace(str(f) + '.tmp', f); return True


def watch(workers=2, every=120):  # наблюдатель: раз в 2 мин финализирует готовые h (по одному разу)
    sel = OUT / 'B_selected.csv'; done = set(pd.read_csv(sel).h.unique()) if sel.exists() else set()
    while len(done) < len(HS):
        for h in HS:
            if h not in done and (OUT / 'B_trials.csv').exists() and finalize(h, workers):
                done.add(h); print(f'h={h} готов {time.strftime("%F %T")}', flush=True)
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bench', action='store_true'); ap.add_argument('--extra', action='store_true'); ap.add_argument('--finalize-h', type=int); ap.add_argument('--watch', action='store_true'); ap.add_argument('--workers', type=int, default=9)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    GRID.reset_index().to_csv(OUT / 'B_grid_100.csv', index=False)
    tf = OUT / 'B_trials.csv'
    if a.extra: return extra(a.workers, tf)
    if a.watch: return watch()
    if a.finalize_h: return print('готов' if finalize(a.finalize_h) else 'не все точки посчитаны')
    if a.bench:  # 10 задач на шаге 2025 + оценка полного времени
        rng = np.random.default_rng(0); tasks = [(MODELS[i % 2], HS[i % 3], 9, int(rng.integers(N_PTS))) for i in range(10)]
        t0 = time.time()
        with Pool(a.workers) as P: res = P.map(task, tasks, chunksize=1)
        wall = time.time() - t0; mean = np.mean([r['sec'] for r in res]); tot = 5400 * mean / a.workers
        print(pd.DataFrame(res).to_string()); print(f'wall10={wall:.0f}s mean_task={mean:.1f}s -> est_total(5400 tasks, {a.workers} workers, step-9 cost)={tot/3600:.2f} h'); return
    done = set(map(tuple, pd.read_csv(tf)[['model', 'h', 'step', 'point']].values)) if tf.exists() else set()
    tasks = [(m, h, s, p) for s in range(1, len(TEST_YEARS) + 1) for h in HS for m in MODELS for p in range(N_PTS) if (m, h, s, p) not in done]
    print(f'tasks left {len(tasks)} of {len(MODELS)*len(HS)*len(TEST_YEARS)*N_PTS}', flush=True); t0 = time.time()
    with Pool(a.workers) as P:
        for i, r in enumerate(P.imap_unordered(task, tasks, chunksize=1), 1):
            pd.DataFrame([r]).to_csv(tf, mode='a', header=not tf.exists(), index=False)
            if i % 50 == 0: print(f'{i}/{len(tasks)} {time.time()-t0:.0f}s', flush=True)
    for h in HS: finalize(h, a.workers)
    print('saved', flush=True)


if __name__ == '__main__':
    main()
