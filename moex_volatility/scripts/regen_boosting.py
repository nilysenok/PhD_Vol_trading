#!/usr/bin/env python3
"""Этап 5: перегенерация бустинга (П1 gbdt+ES, П2 dart с выбором n, xgb как есть) с сохранением валидационных прогнозов.
Запуск из корня репо: python3 moex_volatility/scripts/regen_boosting.py --variant P1|P2|xgb --h 1|5|22
Выход: data/regen/{variant}_h{h}_{val,test}.parquet, data/regen/{variant}_h{h}_meta.csv.
Схема данных и шагов — как в walk_forward_full.py (annual, тест 2017-2025, валидация = последний год обучающего окна);
qlike/clean_features берутся из walk_forward_full.py через ast (скрипт исполняется на уровне модуля, импортом не подключить).
Ранняя остановка П1/XGB: patience 50 (как в коде), метрика l2 на ln(rv) валидационного года (lightgbm eval_metric=l2; xgboost rmse)."""
import argparse, ast, json, os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
MV = ROOT / 'moex_volatility'
src = (MV / 'scripts/walk_forward_full.py').read_text()
ns = {'np': np}
for n in ast.parse(src).body:
    if isinstance(n, ast.FunctionDef) and n.name in ('qlike', 'clean_features'):
        exec(compile(ast.Module([n], []), 'wff', 'exec'), ns)
qlike, clean_features = ns['qlike'], ns['clean_features']

ap = argparse.ArgumentParser()
ap.add_argument('--variant', required=True, choices=['P1', 'P2', 'xgb'])
ap.add_argument('--h', required=True, type=int, choices=[1, 5, 22])
a = ap.parse_args()
h, V = a.h, a.variant
os.chdir(MV)  # пути данных как в walk_forward_full.py
out = ROOT / 'data/regen'
out.mkdir(parents=True, exist_ok=True)

parts = [pd.read_parquet(f'data/prepared/{n}.parquet') for n in ('train', 'val', 'test')]
feature_cols = json.load(open('data/prepared/config.json'))['feature_cols']
for y in range(2020, 2027):
    fp = f'data/prepared/walkforward_{y}.parquet'
    if os.path.exists(fp):
        d = pd.read_parquet(fp)
        if len(d) > 100:
            parts.append(d)
all_data = pd.concat(parts).sort_values('date').reset_index(drop=True)
all_data['year'] = pd.to_datetime(all_data['date']).dt.year
test_years = [y for y in range(2017, 2026) if y in all_data.year.values]

lp = json.load(open(f'models/lightgbm/params_h{h}.json'))
lp.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
xp = json.load(open(f'models/xgboost/params_h{h}.json'))
xp.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})
xp['early_stopping_rounds'] = 50
target = f'rv_target_h{h}'
vals, tests, meta = [], [], []
t0 = time.time()
for step, ty in enumerate(test_years, 1):
    tr = all_data[all_data.year < ty]
    vy = sorted(tr.year.unique())[-1]
    tp = tr[tr.year != vy].dropna(subset=[target])
    vi = tr[tr.year == vy].dropna(subset=[target])
    td = all_data[all_data.year == ty].dropna(subset=[target])
    ytp = np.log(tp[target].values + 1e-10); yvi = np.log(vi[target].values + 1e-10)
    Xtp, Xvi, Xtd = (clean_features(x[feature_cols].values) for x in (tp, vi, td))
    row = {'step': step, 'test_year': ty, 'val_year': vy, 'n_json': lp['n_estimators']}
    if V == 'P1':
        p = dict(lp); p['boosting_type'] = 'gbdt'
        m = lgb.LGBMRegressor(**p)
        m.fit(Xtp, ytp, eval_set=[(Xvi, yvi)], eval_metric='l2', callbacks=[lgb.early_stopping(50, verbose=False)])
        row.update(best_iteration=m.best_iteration_, n_trees=m.booster_.num_trees())
        pv, pt = np.exp(m.predict(Xvi)), np.exp(m.predict(Xtd))  # predict использует best_iteration_
    elif V == 'P2':
        best = None
        for n in sorted({50, 100, 200, 400, lp['n_estimators']}):  # верхняя точка = json (h=5: 634)
            p = dict(lp); p['n_estimators'] = n
            m = lgb.LGBMRegressor(**p).fit(Xtp, ytp)
            pv_n = np.exp(m.predict(Xvi)); q = qlike(vi[target].values, pv_n)
            row[f'qlike_val_n{n}'] = q
            if best is None or q < best[0]:
                best = (q, n, pv_n, np.exp(m.predict(Xtd)))
        _, nsel, pv, pt = best
        row['n_selected'] = nsel
    else:
        row['n_json'] = xp['n_estimators']
        m = xgb.XGBRegressor(**xp).fit(Xtp, ytp, eval_set=[(Xvi, yvi)], verbose=False)
        row['best_iteration'] = m.best_iteration
        pv, pt = np.exp(m.predict(Xvi)), np.exp(m.predict(Xtd))
    row['qlike_val'] = qlike(vi[target].values, np.clip(pv, 1e-10, None))
    meta.append(row)
    for df, pr, lst, yr in ((vi, pv, vals, vy), (td, pt, tests, ty)):
        lst.append(pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': step,
                                 'rv_actual': df[target].values, 'rv_pred': np.clip(pr, 1e-10, None)}))
    print(f'{V} h{h} step {step} test {ty} val {vy} done {time.time()-t0:.0f}s', flush=True)
pd.concat(vals).to_parquet(out / f'{V}_h{h}_val.parquet', index=False)
pd.concat(tests).to_parquet(out / f'{V}_h{h}_test.parquet', index=False)
pd.DataFrame(meta).to_csv(out / f'{V}_h{h}_meta.csv', index=False)
print('saved', flush=True)
