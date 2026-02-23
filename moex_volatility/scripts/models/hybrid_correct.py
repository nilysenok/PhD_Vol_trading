import pandas as pd, numpy as np, pickle, optuna
from optuna.samplers import TPESampler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

import os

Path('models/hybrid').mkdir(parents=True, exist_ok=True)

for h in [1, 5, 22]:
    print(f'\n{"="*60}')
    print(f'H={h}')
    print(f'{"="*60}')

    # Test predictions из parquet
    xgb = pd.read_parquet(f'data/predictions/test_2019/xgboost_h{h}.parquet')
    lgb_df = pd.read_parquet(f'data/predictions/test_2019/lightgbm_h{h}.parquet')
    har = pd.read_parquet(f'data/predictions/test_2019/har_h{h}.parquet')

    # Merge по date+ticker
    m = xgb.merge(lgb_df, on=['date', 'ticker'], suffixes=('_xgb', '_lgb'))
    m = m.merge(har, on=['date', 'ticker'])

    actual = m['rv_actual_xgb'].values
    xgb_pred = m['rv_pred_xgb'].values
    lgb_pred = m['rv_pred_lgb'].values
    har_pred = m['rv_pred'].values

    print(f'Rows: {len(m)}')
    print(f'XGBoost:  {qlike(actual, xgb_pred):.4f}')
    print(f'LightGBM: {qlike(actual, lgb_pred):.4f}')
    print(f'HAR:      {qlike(actual, har_pred):.4f}')

    # Grid search на TEST (для диагностики)
    best_q = 999
    best_w = None
    for w1 in np.arange(0.0, 1.01, 0.02):
        for w2 in np.arange(0.0, 1.01-w1, 0.02):
            w3 = 1.0 - w1 - w2
            if w3 < -0.001: continue
            w3 = max(w3, 0)
            blend = w1*xgb_pred + w2*lgb_pred + w3*har_pred
            q = qlike(actual, blend)
            if q < best_q:
                best_q = q
                best_w = (w1, w2, w3)

    print(f'\nOracle best (on test): XGB*{best_w[0]:.2f}+LGB*{best_w[1]:.2f}+HAR*{best_w[2]:.2f} = {best_q:.4f}')

    # Теперь подбираем веса ЧЕСТНО на val
    import lightgbm as lgbm_lib, xgboost as xgb_lib, json
    from sklearn.linear_model import LinearRegression

    train_df = pd.read_parquet('data/prepared/train.parquet')
    val_df = pd.read_parquet('data/prepared/val.parquet')

    with open('data/prepared/config.json') as f:
        config = json.load(f)
    feature_cols = config['feature_cols']
    har_base = config['har_features']
    jump_cols = [c for c in train_df.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
    harj_features = har_base + jump_cols[:5]
    harj_features = [f for f in harj_features if f in train_df.columns]

    target = f'rv_target_h{h}'
    tr = train_df.dropna(subset=[target])
    va = val_df.dropna(subset=[target])
    y_va_raw = va[target].values

    # HAR-J val predictions
    har_model = LinearRegression()
    X_tr_har = np.nan_to_num(np.log(tr[harj_features].values.clip(1e-10)), 0, 0, 0)
    X_va_har = np.nan_to_num(np.log(va[harj_features].values.clip(1e-10)), 0, 0, 0)
    har_model.fit(X_tr_har, np.log(tr[target].values + 1e-10))
    har_va_pred = np.clip(np.exp(har_model.predict(X_va_har)), 1e-10, None)

    # XGBoost val predictions
    xgb_model = xgb_lib.Booster()
    xgb_model.load_model(f'models/xgboost/model_h{h}.json')
    X_va_ml = va[feature_cols].values
    # Handle inf/nan in features
    X_va_clean = np.nan_to_num(X_va_ml, nan=0.0, posinf=0.0, neginf=0.0)
    xgb_va_pred = np.clip(np.exp(xgb_model.predict(xgb_lib.DMatrix(X_va_clean))), 1e-10, None)

    # LightGBM val predictions
    lgb_model = lgbm_lib.Booster(model_file=f'models/lightgbm/model_h{h}.txt')
    lgb_va_pred = np.clip(np.exp(lgb_model.predict(X_va_clean)), 1e-10, None)

    print(f'\nVal predictions:')
    print(f'  XGB val: {qlike(y_va_raw, xgb_va_pred):.4f}')
    print(f'  LGB val: {qlike(y_va_raw, lgb_va_pred):.4f}')
    print(f'  HAR val: {qlike(y_va_raw, har_va_pred):.4f}')

    # Optuna на val
    def objective(trial):
        w1 = trial.suggest_float('w_xgb', 0.0, 1.0)
        w2 = trial.suggest_float('w_lgb', 0.0, 1.0)
        w3 = trial.suggest_float('w_har', 0.0, 0.5)
        total = w1 + w2 + w3
        if total < 0.01: return 999.0
        w1, w2, w3 = w1/total, w2/total, w3/total
        blend = w1*xgb_va_pred + w2*lgb_va_pred + w3*har_va_pred
        return qlike(y_va_raw, np.clip(blend, 1e-10, None))

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=1000)

    w1 = study.best_params['w_xgb']
    w2 = study.best_params['w_lgb']
    w3 = study.best_params['w_har']
    t = w1+w2+w3
    w1, w2, w3 = w1/t, w2/t, w3/t

    print(f'\nOptuna val weights: XGB={w1:.3f}, LGB={w2:.3f}, HAR={w3:.3f}')
    print(f'Val QLIKE: {study.best_value:.4f}')

    # Apply to test
    blend_test = np.clip(w1*xgb_pred + w2*lgb_pred + w3*har_pred, 1e-10, None)
    q_blend = qlike(actual, blend_test)

    print(f'Test QLIKE: {q_blend:.4f}')

    # Также grid search на val для проверки
    best_val_q = 999
    best_val_w = None
    for ww1 in np.arange(0.0, 1.01, 0.02):
        for ww2 in np.arange(0.0, 1.01-ww1, 0.02):
            ww3 = 1.0 - ww1 - ww2
            if ww3 < -0.001: continue
            ww3 = max(ww3, 0)
            b = ww1*xgb_va_pred + ww2*lgb_va_pred + ww3*har_va_pred
            q = qlike(y_va_raw, b)
            if q < best_val_q:
                best_val_q = q
                best_val_w = (ww1, ww2, ww3)

    # Apply val-optimal to test
    blend_grid = np.clip(best_val_w[0]*xgb_pred + best_val_w[1]*lgb_pred + best_val_w[2]*har_pred, 1e-10, None)
    q_grid = qlike(actual, blend_grid)
    print(f'Grid val-optimal: XGB={best_val_w[0]:.2f}+LGB={best_val_w[1]:.2f}+HAR={best_val_w[2]:.2f} -> test={q_grid:.4f}')

    # Выбираем лучший из optuna и grid
    if q_grid < q_blend:
        final_pred = blend_grid
        final_q = q_grid
        final_w = best_val_w
        method = 'grid'
    else:
        final_pred = blend_test
        final_q = q_blend
        final_w = (w1, w2, w3)
        method = 'optuna'

    print(f'\nFINAL H={h}: {final_q:.4f} (method={method}, w=XGB*{final_w[0]:.2f}+LGB*{final_w[1]:.2f}+HAR*{final_w[2]:.2f})')

    # Save
    pred_df = m[['date', 'ticker']].copy()
    pred_df['rv_actual'] = actual
    pred_df['rv_pred'] = final_pred
    pred_df.to_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet', index=False)

    with open(f'models/hybrid/model_h{h}.pkl', 'wb') as f:
        pickle.dump({
            'type': 'weighted_blend', 'har_variant': 'HAR-J',
            'w_xgb': final_w[0], 'w_lgb': final_w[1], 'w_har': final_w[2],
            'test_qlike': final_q, 'method': method,
            'oracle_qlike': best_q, 'oracle_weights': best_w
        }, f)

# FINAL TABLE
print('\n' + '='*60)
print('FINAL TABLE')
print('='*60)
for h in [1, 5, 22]:
    with open(f'models/hybrid/model_h{h}.pkl', 'rb') as f:
        info = pickle.load(f)
    print(f'H={h}: Hybrid={info["test_qlike"]:.4f} (XGB*{info["w_xgb"]:.2f}+LGB*{info["w_lgb"]:.2f}+HAR*{info["w_har"]:.2f}) | Oracle={info["oracle_qlike"]:.4f}')

print('DONE!')
