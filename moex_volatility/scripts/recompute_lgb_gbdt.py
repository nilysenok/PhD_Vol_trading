#!/usr/bin/env python3
"""Recompute LightGBM walk-forward predictions with gbdt instead of dart.
Saves predictions to data/predictions/walk_forward/lightgbm_gbdt_h*_annual.parquet
and prints QLIKE comparison."""

import pandas as pd
import numpy as np
import json
import os
import time
import warnings
from pathlib import Path
import lightgbm as lgb_lib

warnings.filterwarnings('ignore')
np.random.seed(42)

def qlike(y_true, y_pred):
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def clean_features(X):
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)

# ============================================================
# LOAD DATA
# ============================================================
print('Loading data...')
train = pd.read_parquet('data/prepared/train.parquet')
val = pd.read_parquet('data/prepared/val.parquet')
test = pd.read_parquet('data/prepared/test.parquet')

with open('data/prepared/config.json') as f:
    config = json.load(f)
feature_cols = config['feature_cols']

wf_data = {}
for year in range(2020, 2027):
    fp = f'data/prepared/walkforward_{year}.parquet'
    if os.path.exists(fp):
        df = pd.read_parquet(fp)
        if len(df) > 100:
            wf_data[year] = df

all_data = pd.concat([train, val, test] + [wf_data[y] for y in sorted(wf_data)]).sort_values('date').reset_index(drop=True)
all_data['year'] = pd.to_datetime(all_data['date']).dt.year

test_years = [y for y in range(2017, 2026) if y in all_data.year.values]
print(f'Test years: {test_years}')
print(f'Total rows: {len(all_data)}')

# ============================================================
# LOAD PARAMS — override boosting_type to gbdt
# ============================================================
lgb_params = {}
for h in [1, 5, 22]:
    with open(f'models/lightgbm/params_h{h}.json') as f:
        params = json.load(f)
    # Override dart → gbdt
    params['boosting_type'] = 'gbdt'
    # Remove dart-specific params if any
    for key in ['drop_rate', 'skip_drop', 'max_drop', 'xgboost_dart_mode', 'uniform_drop']:
        params.pop(key, None)
    # Ensure early stopping works
    params.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
    lgb_params[h] = params
    print(f'H={h} params: n_estimators={params.get("n_estimators")}, '
          f'max_depth={params.get("max_depth")}, boosting={params["boosting_type"]}')

# ============================================================
# WALK-FORWARD: LightGBM gbdt only
# ============================================================
print('\n' + '=' * 70)
print('WALK-FORWARD: LightGBM gbdt')
print('=' * 70)

Path('data/predictions/walk_forward').mkdir(parents=True, exist_ok=True)

all_preds = {}  # (h,) -> list of DataFrames
t0 = time.time()

for test_year in test_years:
    train_mask = all_data['year'] < test_year
    test_mask = all_data['year'] == test_year
    train_full = all_data[train_mask]
    test_data = all_data[test_mask]

    if len(test_data) < 10 or len(train_full) < 100:
        continue

    train_years_avail = sorted(train_full['year'].unique())
    val_year = train_years_avail[-1]
    train_pure = train_full[train_full['year'] != val_year]
    val_inner = train_full[train_full['year'] == val_year]

    for h in [1, 5, 22]:
        target = f'rv_target_h{h}'

        tp = train_pure.dropna(subset=[target])
        vi = val_inner.dropna(subset=[target])
        td = test_data.dropna(subset=[target])

        if len(tp) < 50 or len(vi) < 10 or len(td) < 10:
            continue

        y_tp_log = np.log(tp[target].values + 1e-10)
        y_vi_log = np.log(vi[target].values + 1e-10)
        y_td = td[target].values

        X_tp = clean_features(tp[feature_cols].values)
        X_vi = clean_features(vi[feature_cols].values)
        X_td = clean_features(td[feature_cols].values)

        lgb_m = lgb_lib.LGBMRegressor(**lgb_params[h])
        lgb_m.fit(X_tp, y_tp_log, eval_set=[(X_vi, y_vi_log)],
                  callbacks=[lgb_lib.early_stopping(50, verbose=False)])
        lgb_td = np.clip(np.exp(lgb_m.predict(X_td)), 1e-10, None)

        n_trees = lgb_m.best_iteration_ if lgb_m.best_iteration_ else lgb_m.n_estimators
        q = qlike(y_td, lgb_td)
        print(f'  {test_year} H={h}: QLIKE={q:.4f}, trees={n_trees}')

        if h not in all_preds:
            all_preds[h] = []
        all_preds[h].append(pd.DataFrame({
            'date': td['date'].values,
            'ticker': td['ticker'].values,
            'year': test_year,
            'rv_actual': y_td,
            'rv_pred': lgb_td,
        }))

elapsed = time.time() - t0
print(f'\nDone in {elapsed:.0f}s')

# ============================================================
# SAVE & COMPARE
# ============================================================
print('\n' + '=' * 70)
print('RESULTS: gbdt vs dart')
print('=' * 70)

for h in [1, 5, 22]:
    if h not in all_preds:
        continue

    # Save gbdt predictions
    combined = pd.concat(all_preds[h]).reset_index(drop=True)
    out_fp = f'data/predictions/walk_forward/lightgbm_gbdt_h{h}_annual.parquet'
    combined.to_parquet(out_fp)

    # Load dart predictions for comparison
    dart_fp = f'data/predictions/walk_forward/lightgbm_h{h}_annual.parquet'
    dart_df = pd.read_parquet(dart_fp)
    dart_df = dart_df[dart_df['rv_actual'] > 0]

    # Overall QLIKE
    gbdt_q = qlike(combined['rv_actual'].values, combined['rv_pred'].values)
    dart_q = qlike(dart_df['rv_actual'].values, dart_df['rv_pred'].values)

    print(f'\nH={h}: dart={dart_q:.4f}  gbdt={gbdt_q:.4f}  diff={gbdt_q-dart_q:+.4f}')

    # By year
    for year in test_years:
        g = combined[combined['year'] == year]
        d = dart_df[dart_df['year'] == year]
        if len(g) > 0 and len(d) > 0:
            gq = qlike(g['rv_actual'].values, g['rv_pred'].values)
            dq = qlike(d['rv_actual'].values, d['rv_pred'].values)
            winner = 'gbdt' if gq < dq else 'dart'
            print(f'  {year}: dart={dq:.4f}  gbdt={gq:.4f}  {winner}')
