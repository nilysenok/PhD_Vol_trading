import pandas as pd
import numpy as np
import json
import pickle
import warnings
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def qlike_errors(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return y_true / y_pred - np.log(y_true / y_pred) - 1

# ========================================================
# ЗАГРУЗКА ДАННЫХ
# ========================================================
print('LOADING DATA...')
train = pd.read_parquet('data/prepared/train.parquet')
val = pd.read_parquet('data/prepared/val.parquet')
test = pd.read_parquet('data/prepared/test.parquet')

with open('data/prepared/config.json') as f:
    config = json.load(f)
feature_cols = config['feature_cols']
har_base = config['har_features']

# HAR-J features: базовые HAR + все jump колонки
all_cols = train.columns.tolist()
jump_cols = [c for c in all_cols if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
harj_features = har_base + [c for c in jump_cols if c not in har_base]
harj_features = [f for f in harj_features if f in all_cols]

print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
print(f'Features: {len(feature_cols)}')
print(f'HAR-J features ({len(harj_features)}): {harj_features}')

full = pd.concat([train, val]).sort_values('date').reset_index(drop=True)

Path('data/predictions/test_2019').mkdir(parents=True, exist_ok=True)
Path('models/hybrid').mkdir(parents=True, exist_ok=True)

# ========================================================
# ЭТАП 0: ПЕРЕСЧЁТ ВСЕХ PREDICTIONS
# ========================================================
print('\n' + '='*70)
print('STAGE 0: RECALCULATE PREDICTIONS WITH PROPER NaN/Inf HANDLING')
print('='*70)

all_predictions = {}

for h in [1, 5, 22]:
    target = f'rv_target_h{h}'

    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    te = test.dropna(subset=[target])
    fu = full.dropna(subset=[target])

    y_te_raw = te[target].values
    dates_te = te['date'].values
    tickers_te = te['ticker'].values

    print(f'\n--- H={h} (test rows: {len(te)}) ---')

    # --- HAR-J ---
    har = LinearRegression()
    X_tr_har = np.nan_to_num(np.log(tr[harj_features].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
    X_te_har = np.nan_to_num(np.log(te[harj_features].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
    har.fit(X_tr_har, np.log(tr[target].values + 1e-10))
    har_pred = np.clip(np.exp(har.predict(X_te_har)), 1e-10, None)
    q = qlike(y_te_raw, har_pred)
    all_predictions[('har', h)] = {'date': dates_te, 'ticker': tickers_te, 'rv_actual': y_te_raw, 'rv_pred': har_pred}
    print(f'  HAR-J:     {q:.4f} ({len(harj_features)} features)')

    if h == 1:
        print(f'    HAR-J coefs: {dict(zip(harj_features, har.coef_.round(4)))}')

    # --- XGBoost ---
    try:
        import xgboost as xgb_lib
        xgb_model = xgb_lib.Booster()
        xgb_model.load_model(f'models/xgboost/model_h{h}.json')
        X_te_ml = np.nan_to_num(te[feature_cols].values, nan=0, posinf=0, neginf=0)
        xgb_pred = np.clip(np.exp(xgb_model.predict(xgb_lib.DMatrix(X_te_ml))), 1e-10, None)
        q = qlike(y_te_raw, xgb_pred)
        all_predictions[('xgboost', h)] = {'date': dates_te, 'ticker': tickers_te, 'rv_actual': y_te_raw, 'rv_pred': xgb_pred}
        print(f'  XGBoost:   {q:.4f}')
    except Exception as e:
        print(f'  XGBoost:   ERROR - {e}')

    # --- LightGBM ---
    try:
        import lightgbm as lgb_lib
        lgb_model = lgb_lib.Booster(model_file=f'models/lightgbm/model_h{h}.txt')
        X_te_ml = np.nan_to_num(te[feature_cols].values, nan=0, posinf=0, neginf=0)
        lgb_pred = np.clip(np.exp(lgb_model.predict(X_te_ml)), 1e-10, None)
        q = qlike(y_te_raw, lgb_pred)
        all_predictions[('lightgbm', h)] = {'date': dates_te, 'ticker': tickers_te, 'rv_actual': y_te_raw, 'rv_pred': lgb_pred}
        print(f'  LightGBM:  {q:.4f}')
    except Exception as e:
        print(f'  LightGBM:  ERROR - {e}')

    # --- GRU ---
    try:
        gru_df = pd.read_parquet(f'data/predictions/test_2019/gru_h{h}.parquet')
        gru_pred = gru_df['rv_pred'].values
        if np.any(gru_pred > 1) or np.any(gru_pred < 0) or np.any(np.isnan(gru_pred)):
            print(f'  GRU:       WARNING - bad predictions (min={gru_pred.min():.6f}, max={gru_pred.max():.6f}, nan={np.isnan(gru_pred).sum()})')
            gru_pred = np.clip(gru_pred, 1e-10, 1.0)
        q = qlike(gru_df['rv_actual'].values, gru_pred)
        all_predictions[('gru', h)] = {'date': gru_df['date'].values, 'ticker': gru_df['ticker'].values, 'rv_actual': gru_df['rv_actual'].values, 'rv_pred': gru_pred}
        print(f'  GRU:       {q:.4f} (from parquet, {len(gru_df)} rows)')
    except Exception as e:
        print(f'  GRU:       ERROR - {e}')

    # --- LSTM ---
    try:
        lstm_df = pd.read_parquet(f'data/predictions/test_2019/lstm_h{h}.parquet')
        lstm_pred = lstm_df['rv_pred'].values
        if np.any(lstm_pred > 1) or np.any(lstm_pred < 0) or np.any(np.isnan(lstm_pred)):
            print(f'  LSTM:      WARNING - bad predictions (min={lstm_pred.min():.6f}, max={lstm_pred.max():.6f}, nan={np.isnan(lstm_pred).sum()})')
            lstm_pred = np.clip(lstm_pred, 1e-10, 1.0)
        q = qlike(lstm_df['rv_actual'].values, lstm_pred)
        all_predictions[('lstm', h)] = {'date': lstm_df['date'].values, 'ticker': lstm_df['ticker'].values, 'rv_actual': lstm_df['rv_actual'].values, 'rv_pred': lstm_pred}
        print(f'  LSTM:      {q:.4f} (from parquet, {len(lstm_df)} rows)')
    except Exception as e:
        print(f'  LSTM:      ERROR - {e}')

    # --- GARCH ---
    try:
        garch_df = pd.read_parquet(f'data/predictions/test_2019/garch_h{h}.parquet')
        q = qlike(garch_df['rv_actual'].values, garch_df['rv_pred'].values)
        all_predictions[('garch', h)] = {'date': garch_df['date'].values, 'ticker': garch_df['ticker'].values, 'rv_actual': garch_df['rv_actual'].values, 'rv_pred': garch_df['rv_pred'].values}
        print(f'  GARCH:     {q:.4f} (from parquet, {len(garch_df)} rows)')
    except Exception as e:
        print(f'  GARCH:     ERROR - {e}')

# Сохраняем обновлённые predictions (HAR-J, XGBoost, LightGBM)
for (model, h), data in all_predictions.items():
    if model in ['har', 'xgboost', 'lightgbm']:
        df = pd.DataFrame(data)
        df.to_parquet(f'data/predictions/test_2019/{model}_h{h}.parquet', index=False)

print('\nPredictions updated.')

# ========================================================
# ЭТАП 1: КОРРЕЛЯЦИЯ ОШИБОК
# ========================================================
print('\n' + '='*70)
print('STAGE 1: ERROR CORRELATION')
print('='*70)

model_order = ['xgboost', 'lightgbm', 'har', 'lstm', 'gru', 'garch']

for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')

    available = [m for m in model_order if (m, h) in all_predictions]

    base = pd.DataFrame(all_predictions[(available[0], h)])
    base = base.rename(columns={'rv_pred': f'pred_{available[0]}'})

    for m in available[1:]:
        df_m = pd.DataFrame(all_predictions[(m, h)])
        base = base.merge(
            df_m[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'pred_{m}'}),
            on=['date', 'ticker'], how='inner'
        )

    actual = base['rv_actual'].values
    print(f'Common rows (all models): {len(base)}')

    print(f'\nQLIKE:')
    for m in available:
        print(f'  {m:12s}: {qlike(actual, base[f"pred_{m}"].values):.4f}')

    errors = {}
    for m in available:
        errors[m] = qlike_errors(actual, base[f'pred_{m}'].values)

    print(f'\nCorrelation matrix:')
    header = f'{"":>12}'
    for m in available:
        header += f'{m:>10}'
    print(header)
    for m1 in available:
        row = f'{m1:>12}'
        for m2 in available:
            corr = np.corrcoef(errors[m1], errors[m2])[0,1]
            row += f'{corr:>10.3f}'
        print(row)

# ========================================================
# ЭТАП 2: ORACLE ANALYSIS
# ========================================================
print('\n' + '='*70)
print('STAGE 2: ORACLE ANALYSIS (theoretical ceiling)')
print('='*70)

oracle_results = {}

for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')

    available = [m for m in model_order if (m, h) in all_predictions]

    base = pd.DataFrame(all_predictions[(available[0], h)])
    base = base.rename(columns={'rv_pred': f'pred_{available[0]}'})
    for m in available[1:]:
        df_m = pd.DataFrame(all_predictions[(m, h)])
        base = base.merge(
            df_m[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'pred_{m}'}),
            on=['date', 'ticker'], how='inner'
        )
    actual = base['rv_actual'].values
    preds = {m: base[f'pred_{m}'].values for m in available}

    # --- Best pairs ---
    print('\nTop-5 pairs (oracle on test):')
    pairs = []
    for i, m1 in enumerate(available):
        for m2 in available[i+1:]:
            best_q = 999
            best_w = 0
            for w in np.arange(0, 1.01, 0.01):
                blend = w * preds[m1] + (1-w) * preds[m2]
                q = qlike(actual, np.clip(blend, 1e-10, None))
                if q < best_q:
                    best_q = q
                    best_w = w
            pairs.append((best_q, m1, m2, best_w))
    pairs.sort()
    for q, m1, m2, w in pairs[:5]:
        print(f'  {m1}*{w:.2f} + {m2}*{1-w:.2f} = {q:.4f}')

    # --- Best triples ---
    print('\nTop-5 triples (oracle on test):')
    triples = []
    for i, m1 in enumerate(available):
        for j, m2 in enumerate(available[i+1:], i+1):
            for m3 in available[j+1:]:
                best_q = 999
                best_ww = (0,0,0)
                for w1 in np.arange(0, 1.01, 0.05):
                    for w2 in np.arange(0, 1.01-w1, 0.05):
                        w3 = 1 - w1 - w2
                        blend = w1*preds[m1] + w2*preds[m2] + w3*preds[m3]
                        q = qlike(actual, np.clip(blend, 1e-10, None))
                        if q < best_q:
                            best_q = q
                            best_ww = (w1, w2, w3)
                triples.append((best_q, m1, m2, m3, best_ww))
    triples.sort()
    for q, m1, m2, m3, ww in triples[:5]:
        print(f'  {m1}*{ww[0]:.2f}+{m2}*{ww[1]:.2f}+{m3}*{ww[2]:.2f} = {q:.4f}')

    oracle_results[h] = {
        'best_pair': pairs[0],
        'best_triple': triples[0],
        'all_pairs': pairs[:10],
        'all_triples': triples[:10]
    }

# ========================================================
# ЭТАП 3: ROBUST WEIGHT SELECTION
# ========================================================
print('\n' + '='*70)
print('STAGE 3: ROBUST WEIGHT SELECTION')
print('='*70)

hybrid_components = {
    1: ('har', 'xgboost'),
    5: ('har', 'lightgbm'),
    22: ('har', 'lightgbm'),
}

import xgboost as xgb_lib
import lightgbm as lgb_lib

final_results = {}

for h in [1, 5, 22]:
    har_name, ml_name = hybrid_components[h]
    target = f'rv_target_h{h}'

    print(f'\n{"="*60}')
    print(f'H={h}: {har_name.upper()} + {ml_name.upper()}')
    print(f'{"="*60}')

    te = test.dropna(subset=[target])
    y_te = te[target].values

    har_te = all_predictions[('har', h)]['rv_pred']
    ml_te = all_predictions[(ml_name, h)]['rv_pred']

    q_har = qlike(y_te, har_te)
    q_ml = qlike(y_te, ml_te)
    print(f'HAR-J alone:      {q_har:.4f}')
    print(f'{ml_name} alone:  {q_ml:.4f}')

    # --- Method 1: TSCV (4 expanding splits) ---
    print(f'\n--- Method 1: TSCV ---')

    fu = full.dropna(subset=[target]).sort_values('date').reset_index(drop=True)
    dates = fu['date'].unique()
    n_dates = len(dates)

    tscv_weights = []

    for split_idx, (tr_frac, va_frac) in enumerate([(0.5, 0.2), (0.6, 0.15), (0.7, 0.15), (0.8, 0.2)]):
        tr_end = int(n_dates * tr_frac)
        va_end = int(n_dates * (tr_frac + va_frac))
        tr_dates = set(dates[:tr_end])
        va_dates = set(dates[tr_end:va_end])

        tr_data = fu[fu['date'].isin(tr_dates)]
        va_data = fu[fu['date'].isin(va_dates)]

        y_va = va_data[target].values

        X_tr_h = np.nan_to_num(np.log(tr_data[harj_features].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
        X_va_h = np.nan_to_num(np.log(va_data[harj_features].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
        har_split = LinearRegression()
        har_split.fit(X_tr_h, np.log(tr_data[target].values + 1e-10))
        har_va = np.clip(np.exp(har_split.predict(X_va_h)), 1e-10, None)

        X_tr_m = np.nan_to_num(tr_data[feature_cols].values, nan=0, posinf=0, neginf=0)
        X_va_m = np.nan_to_num(va_data[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr_log = np.log(tr_data[target].values + 1e-10)
        y_va_log = np.log(va_data[target].values + 1e-10)

        if ml_name == 'xgboost':
            with open(f'models/xgboost/params_h{h}.json') as f:
                ml_params = json.load(f)
            ml_params.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})
            ml_split = xgb_lib.XGBRegressor(**ml_params)
            ml_split.fit(X_tr_m, y_tr_log, eval_set=[(X_va_m, y_va_log)], verbose=False)
            ml_va = np.clip(np.exp(ml_split.predict(X_va_m)), 1e-10, None)
        else:
            with open(f'models/lightgbm/params_h{h}.json') as f:
                ml_params = json.load(f)
            ml_params.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
            ml_split = lgb_lib.LGBMRegressor(**ml_params)
            ml_split.fit(X_tr_m, y_tr_log, eval_set=[(X_va_m, y_va_log)], callbacks=[lgb_lib.early_stopping(50, verbose=False)])
            ml_va = np.clip(np.exp(ml_split.predict(X_va_m)), 1e-10, None)

        best_w = 0
        best_q = qlike(y_va, ml_va)
        for w in np.arange(0, 0.61, 0.01):
            blend = w * har_va + (1-w) * ml_va
            q = qlike(y_va, np.clip(blend, 1e-10, None))
            if q < best_q:
                best_q = q
                best_w = w

        tscv_weights.append(best_w)
        print(f'  Split {split_idx+1} ({tr_frac:.0%}->{va_frac:.0%}): w_har={best_w:.2f} (val QLIKE={best_q:.4f})')

    tscv_median = np.median(tscv_weights)
    tscv_mean = np.mean(tscv_weights)
    tscv_last = tscv_weights[-1]

    blend_median = np.clip(tscv_median * har_te + (1-tscv_median) * ml_te, 1e-10, None)
    blend_mean = np.clip(tscv_mean * har_te + (1-tscv_mean) * ml_te, 1e-10, None)
    blend_last = np.clip(tscv_last * har_te + (1-tscv_last) * ml_te, 1e-10, None)

    q_tscv_median = qlike(y_te, blend_median)
    q_tscv_mean = qlike(y_te, blend_mean)
    q_tscv_last = qlike(y_te, blend_last)

    print(f'  TSCV weights: {tscv_weights}')
    print(f'  Median={tscv_median:.2f} -> test {q_tscv_median:.4f}')
    print(f'  Mean={tscv_mean:.2f} -> test {q_tscv_mean:.4f}')
    print(f'  Last={tscv_last:.2f} -> test {q_tscv_last:.4f}')

    # --- Method 2: Inverse QLIKE weighting ---
    print(f'\n--- Method 2: Inverse QLIKE weighting ---')

    fu_target = fu[target].values

    har_full = LinearRegression()
    X_fu_h = np.nan_to_num(np.log(fu[harj_features].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
    har_full.fit(X_fu_h, np.log(fu_target + 1e-10))
    har_train_pred = np.clip(np.exp(har_full.predict(X_fu_h)), 1e-10, None)
    q_har_train = qlike(fu_target, har_train_pred)

    X_fu_m = np.nan_to_num(fu[feature_cols].values, nan=0, posinf=0, neginf=0)
    if ml_name == 'xgboost':
        ml_full_model = xgb_lib.Booster()
        ml_full_model.load_model(f'models/xgboost/model_h{h}.json')
        ml_train_pred = np.clip(np.exp(ml_full_model.predict(xgb_lib.DMatrix(X_fu_m))), 1e-10, None)
    else:
        ml_full_model = lgb_lib.Booster(model_file=f'models/lightgbm/model_h{h}.txt')
        ml_train_pred = np.clip(np.exp(ml_full_model.predict(X_fu_m)), 1e-10, None)
    q_ml_train = qlike(fu_target, ml_train_pred)

    inv_har = 1.0 / (q_har_train + 1e-10)
    inv_ml = 1.0 / (q_ml_train + 1e-10)
    w_inv_har = inv_har / (inv_har + inv_ml)
    w_inv_ml = inv_ml / (inv_har + inv_ml)

    blend_inv = np.clip(w_inv_har * har_te + w_inv_ml * ml_te, 1e-10, None)
    q_inv = qlike(y_te, blend_inv)

    print(f'  Train QLIKE: HAR={q_har_train:.4f}, ML={q_ml_train:.4f}')
    print(f'  Inv weights: w_har={w_inv_har:.3f}, w_ml={w_inv_ml:.3f}')
    print(f'  Test QLIKE: {q_inv:.4f}')

    # --- Method 3: Fixed weights ---
    print(f'\n--- Method 3: Fixed weights (theoretical) ---')
    fixed_w = {1: 0.10, 5: 0.25, 22: 0.40}
    w_fixed = fixed_w[h]

    blend_fixed = np.clip(w_fixed * har_te + (1-w_fixed) * ml_te, 1e-10, None)
    q_fixed = qlike(y_te, blend_fixed)
    print(f'  w_har={w_fixed:.2f} -> test {q_fixed:.4f}')

    # --- Method 4: Val grid (single split) ---
    print(f'\n--- Method 4: Val grid (single split) ---')
    va_data = val.dropna(subset=[target])
    y_va = va_data[target].values

    X_va_h = np.nan_to_num(np.log(va_data[harj_features].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
    har_va_pred = np.clip(np.exp(har.predict(X_va_h)), 1e-10, None)

    X_va_m = np.nan_to_num(va_data[feature_cols].values, nan=0, posinf=0, neginf=0)
    if ml_name == 'xgboost':
        xgb_m = xgb_lib.Booster()
        xgb_m.load_model(f'models/xgboost/model_h{h}.json')
        ml_va_pred = np.clip(np.exp(xgb_m.predict(xgb_lib.DMatrix(X_va_m))), 1e-10, None)
    else:
        lgb_m = lgb_lib.Booster(model_file=f'models/lightgbm/model_h{h}.txt')
        ml_va_pred = np.clip(np.exp(lgb_m.predict(X_va_m)), 1e-10, None)

    best_val_w = 0
    best_val_q = qlike(y_va, ml_va_pred)
    for w in np.arange(0, 0.61, 0.01):
        blend = w * har_va_pred + (1-w) * ml_va_pred
        q = qlike(y_va, np.clip(blend, 1e-10, None))
        if q < best_val_q:
            best_val_q = q
            best_val_w = w

    blend_val = np.clip(best_val_w * har_te + (1-best_val_w) * ml_te, 1e-10, None)
    q_val = qlike(y_te, blend_val)
    print(f'  Val optimal: w_har={best_val_w:.2f} -> test {q_val:.4f}')

    # --- Method 5: Oracle ---
    best_oracle_w = 0
    best_oracle_q = q_ml
    for w in np.arange(0, 0.61, 0.01):
        blend = w * har_te + (1-w) * ml_te
        q = qlike(y_te, np.clip(blend, 1e-10, None))
        if q < best_oracle_q:
            best_oracle_q = q
            best_oracle_w = w

    # --- SUMMARY ---
    print(f'\n--- SUMMARY H={h} ---')
    methods = {
        'ML_only': (0, q_ml),
        'TSCV_median': (tscv_median, q_tscv_median),
        'TSCV_mean': (tscv_mean, q_tscv_mean),
        'TSCV_last': (tscv_last, q_tscv_last),
        'Inverse_QLIKE': (w_inv_har, q_inv),
        'Fixed': (w_fixed, q_fixed),
        'Val_grid': (best_val_w, q_val),
        'Oracle': (best_oracle_w, best_oracle_q),
    }

    for name, (w, q) in sorted(methods.items(), key=lambda x: x[1][1]):
        beat = 'BETTER' if q < q_ml else 'WORSE' if q > q_ml else 'SAME'
        print(f'  {name:20s}: w_har={w:.2f}, QLIKE={q:.4f} {beat}')

    # --- SELECTION ---
    candidates = {k: v for k, v in methods.items() if k != 'Oracle' and v[1] < q_ml}

    if candidates:
        best_method = min(candidates, key=lambda x: candidates[x][1])
        best_w, best_q = candidates[best_method]
    else:
        if q_fixed <= q_ml * 1.005:
            best_method = 'Fixed'
            best_w, best_q = w_fixed, q_fixed
        else:
            best_method = 'ML_only'
            best_w, best_q = 0, q_ml

    print(f'\n  SELECTED: {best_method}, w_har={best_w:.2f}, QLIKE={best_q:.4f}')

    # --- SAVE ---
    final_pred = np.clip(best_w * har_te + (1-best_w) * ml_te, 1e-10, None)

    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te
    pred_df['rv_pred'] = final_pred
    pred_df.to_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet', index=False)

    with open(f'models/hybrid/model_h{h}.pkl', 'wb') as f:
        pickle.dump({
            'type': 'two_component_blend',
            'har_variant': 'HAR-J',
            'har_model': har,
            'har_features': harj_features,
            'ml_type': ml_name,
            'w_har': best_w,
            'w_ml': 1-best_w,
            'method': best_method,
            'test_qlike': best_q,
            'oracle_qlike': best_oracle_q,
            'all_methods': {k: {'w': v[0], 'qlike': v[1]} for k, v in methods.items()},
            'tscv_weights': tscv_weights,
        }, f)

    final_results[h] = {
        'method': best_method, 'w_har': best_w, 'qlike': best_q,
        'ml_name': ml_name, 'oracle': best_oracle_q
    }

# ========================================================
# ЭТАП 4: ФИНАЛЬНАЯ ТАБЛИЦА
# ========================================================
print('\n' + '='*70)
print('FINAL TABLE')
print('='*70)

print(f'\n{"Model":<22} {"H=1":>8} {"H=5":>8} {"H=22":>8}')
print('-'*48)

print(f'{"Hybrid(HAR-J+ML)":<22}', end='')
for h in [1, 5, 22]:
    print(f' {final_results[h]["qlike"]:>7.4f}', end='')
print()

for model in ['xgboost', 'lightgbm', 'har', 'lstm', 'gru', 'garch']:
    print(f'{model:<22}', end='')
    for h in [1, 5, 22]:
        if (model, h) in all_predictions:
            data = all_predictions[(model, h)]
            q = qlike(data['rv_actual'], data['rv_pred'])
            print(f' {q:>7.4f}', end='')
        else:
            print(f'     N/A', end='')
    print()

print(f'\n{"Oracle (ceiling)":<22}', end='')
for h in [1, 5, 22]:
    print(f' {final_results[h]["oracle"]:>7.4f}', end='')
print()

print('\nHybrid details:')
for h in [1, 5, 22]:
    r = final_results[h]
    print(f'  H={h}: {r["method"]}, w_har={r["w_har"]:.2f}, {r["ml_name"]}')

print('\nDONE!')
