import pandas as pd, numpy as np, json, optuna, pickle, lightgbm as lgb, xgboost
from sklearn.linear_model import LinearRegression
from optuna.samplers import TPESampler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

train = pd.read_parquet('data/prepared/train.parquet')
val = pd.read_parquet('data/prepared/val.parquet')
test = pd.read_parquet('data/prepared/test.parquet')

with open('data/prepared/config.json') as f:
    config = json.load(f)
feature_cols = config['feature_cols']
har_base = config['har_features']

all_cols = train.columns.tolist()

def find(patterns):
    return [c for c in all_cols if any(p in c.lower() for p in patterns) and 'target' not in c.lower() and 'idx_' not in c.lower()]

# Лучшие HAR варианты по горизонтам
best_har = {
    1: ('HAR-BV', har_base + find(['bv'])[:3], 'XGB'),
    5: ('HAR-J', har_base + find(['jump'])[:3], 'LGB'),
    22: ('HAR-SK', har_base + find(['rskew', 'rkurt'])[:4], 'LGB'),
}

Path('models/hybrid').mkdir(parents=True, exist_ok=True)

for h in [1, 5, 22]:
    har_name, har_feats, ml_type = best_har[h]
    har_feats = [f for f in har_feats if f in all_cols]
    target = f'rv_target_h{h}'

    print(f'\n{"="*60}')
    print(f'H={h}: {har_name} + {ml_type} (features: {har_feats})')
    print(f'{"="*60}')

    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    te = test.dropna(subset=[target])

    y_tr_log = np.log(tr[target].values + 1e-10)
    y_va_log = np.log(va[target].values + 1e-10)
    y_va_raw = va[target].values
    y_te_raw = te[target].values

    # HAR
    X_tr_har = np.nan_to_num(np.log(tr[har_feats].values.clip(1e-10)), 0, 0, 0)
    X_va_har = np.nan_to_num(np.log(va[har_feats].values.clip(1e-10)), 0, 0, 0)
    X_te_har = np.nan_to_num(np.log(te[har_feats].values.clip(1e-10)), 0, 0, 0)

    har = LinearRegression()
    har.fit(X_tr_har, y_tr_log)

    har_tr_log = har.predict(X_tr_har)
    har_va_log = har.predict(X_va_har)
    har_te_log = har.predict(X_te_har)

    # HAR-only QLIKE
    q_har = qlike(y_te_raw, np.clip(np.exp(har_te_log), 1e-10, None))
    print(f'{har_name} only: QLIKE={q_har:.4f}')

    # Residuals
    resid_tr = y_tr_log - har_tr_log
    resid_va = y_va_log - har_va_log

    X_tr_ml = tr[feature_cols].values
    X_va_ml = va[feature_cols].values
    X_te_ml = te[feature_cols].values

    if ml_type == 'XGB':
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 40),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 5.0, log=True),
                'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1
            }
            w = trial.suggest_float('w_resid', 0.1, 1.5)

            model = xgboost.XGBRegressor(**params)
            model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], verbose=False)
            pred = model.predict(X_va_ml)
            combined = np.exp(har_va_log + w * pred)
            return qlike(y_va_raw, np.clip(combined, 1e-10, None))
    else:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 128),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 5.0, log=True),
                'boosting_type': 'gbdt',
                'verbosity': -1, 'random_state': 42, 'n_jobs': -1
            }
            w = trial.suggest_float('w_resid', 0.1, 1.5)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = model.predict(X_va_ml)
            combined = np.exp(har_va_log + w * pred)
            return qlike(y_va_raw, np.clip(combined, 1e-10, None))

    print(f'Optuna 150 trials...')
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, n_startup_trials=20))
    study.optimize(objective, n_trials=150, show_progress_bar=True)

    bp = {k:v for k,v in study.best_params.items() if k != 'w_resid'}
    w = study.best_params['w_resid']
    print(f'Best val QLIKE: {study.best_value:.4f}, w_resid={w:.3f}')

    # Final model
    if ml_type == 'XGB':
        bp.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})
        ml_model = xgboost.XGBRegressor(**bp)
        ml_model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], verbose=False)
    else:
        bp.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1, 'boosting_type': 'gbdt'})
        ml_model = lgb.LGBMRegressor(**bp)
        ml_model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

    resid_te = ml_model.predict(X_te_ml)
    hybrid_pred = np.clip(np.exp(har_te_log + w * resid_te), 1e-10, None)
    q_hybrid = qlike(y_te_raw, hybrid_pred)

    print(f'\nRESULTS H={h}:')
    print(f'  {har_name} only: {q_har:.4f}')
    print(f'  Hybrid ({har_name}+{ml_type}): {q_hybrid:.4f}')
    print(f'  Improvement: {((q_har-q_hybrid)/q_har)*100:.1f}%')

    # Save
    with open(f'models/hybrid/model_h{h}.pkl', 'wb') as f:
        pickle.dump({
            'type': 'residual', 'har_variant': har_name, 'har_model': har,
            'har_features': har_feats, 'ml_type': ml_type, 'ml_model': ml_model,
            'ml_params': bp, 'w_resid': w, 'test_qlike': q_hybrid
        }, f)

    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = hybrid_pred
    pred_df.to_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet', index=False)

print('\n' + '='*60)
print('ИТОГО')
print('='*60)
print('Model         H=1     H=5     H=22')
print(f'XGBoost       0.278   0.396   0.470')
print(f'LightGBM      0.289   0.381   0.455')
for h in [1, 5, 22]:
    with open(f'models/hybrid/model_h{h}.pkl', 'rb') as f:
        info = pickle.load(f)
    print(f'Hybrid H={h}    {info["test_qlike"]:.3f}   ({info["har_variant"]}+{info["ml_type"]})')
print(f'HAR           0.314   0.432   0.468')
print(f'GARCH         1.802   1.051   0.732')
print('DONE!')
