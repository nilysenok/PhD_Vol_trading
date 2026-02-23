import pandas as pd, numpy as np, json, optuna, pickle
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
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

# HAR features
har_features = ['rv_d', 'rv_w', 'rv_m']
# Если har_features в config — используй их
if 'har_features' in config:
    har_features = config['har_features']

for h in [1, 5, 22]:
    target = f'rv_target_h{h}'
    print(f'\n========== HYBRID H={h} ==========')

    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    te = test.dropna(subset=[target])

    y_tr_raw, y_va_raw, y_te_raw = tr[target].values, va[target].values, te[target].values
    y_tr_log = np.log(y_tr_raw + 1e-10)
    y_va_log = np.log(y_va_raw + 1e-10)

    # ====== Step 1: HAR прогноз ======
    X_tr_har = tr[har_features].apply(lambda x: np.log(x + 1e-10)).values
    X_va_har = va.loc[va.index.isin(va.dropna(subset=[target]).index), har_features].apply(lambda x: np.log(x + 1e-10)).values
    X_te_har = te.loc[te.index.isin(te.dropna(subset=[target]).index), har_features].apply(lambda x: np.log(x + 1e-10)).values

    har = LinearRegression()
    har.fit(X_tr_har, y_tr_log)

    har_pred_tr = har.predict(X_tr_har)
    har_pred_va = har.predict(X_va_har)
    har_pred_te = har.predict(X_te_har)

    # ====== Step 2: Residuals ======
    residuals_tr = y_tr_log - har_pred_tr
    residuals_va = y_va_log - har_pred_va

    # ====== Step 3: ML модель на residuals ======
    # H=1 -> XGBoost (лучше на краткосрочном)
    # H=5, H=22 -> LightGBM (лучше на средне/долгосрочном)

    X_tr_ml = tr[feature_cols].values
    X_va_ml = va.loc[va.index.isin(va.dropna(subset=[target]).index), feature_cols].values
    X_te_ml = te.loc[te.index.isin(te.dropna(subset=[target]).index), feature_cols].values

    if h == 1:
        ml_name = 'XGBoost'
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'tree_method': 'hist',
                'verbosity': 0, 'random_state': 42, 'n_jobs': -1
            }
            w_har = trial.suggest_float('w_har', 0.0, 1.0)

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr_ml, residuals_tr, eval_set=[(X_va_ml, residuals_va)], verbose=False)

            resid_pred = model.predict(X_va_ml)
            # combined = har + (1 - w_har) * resid_pred
            combined_log2 = har_pred_va + (1 - w_har) * resid_pred
            combined = np.exp(combined_log2)
            return qlike(y_va_raw, np.clip(combined, 1e-10, None))
    else:
        ml_name = 'LightGBM'
        def objective(trial):
            boosting = trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'boosting_type': boosting,
                'verbosity': -1, 'random_state': 42, 'n_jobs': -1
            }
            w_har = trial.suggest_float('w_har', 0.0, 1.0)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr_ml, residuals_tr, eval_set=[(X_va_ml, residuals_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

            resid_pred = model.predict(X_va_ml)
            combined_log = har_pred_va + (1 - w_har) * resid_pred
            combined = np.exp(combined_log)
            return qlike(y_va_raw, np.clip(combined, 1e-10, None))

    print(f'Hybrid = HAR + {ml_name}, Optuna 150 trials...')
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, n_startup_trials=20))
    study.optimize(objective, n_trials=150, show_progress_bar=True)

    best_params = {k:v for k,v in study.best_params.items() if k != 'w_har'}
    w_har = study.best_params['w_har']

    print(f'Best val QLIKE: {study.best_value:.4f}, w_har={w_har:.3f}')

    # Final model
    if h == 1:
        best_params.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})
        ml_model = xgb.XGBRegressor(**best_params)
        ml_model.fit(X_tr_ml, residuals_tr, eval_set=[(X_va_ml, residuals_va)], verbose=False)
    else:
        best_params.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
        ml_model = lgb.LGBMRegressor(**best_params)
        ml_model.fit(X_tr_ml, residuals_tr, eval_set=[(X_va_ml, residuals_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

    # Predict test
    resid_pred_te = ml_model.predict(X_te_ml)
    final_log = har_pred_te + (1 - w_har) * resid_pred_te
    final_pred = np.clip(np.exp(final_log), 1e-10, None)

    q_hybrid = qlike(y_te_raw, final_pred)

    # Сравнение
    har_only = np.clip(np.exp(har_pred_te), 1e-10, None)
    q_har = qlike(y_te_raw, har_only)

    print(f'\nH={h} Results:')
    print(f'  HAR only:     {q_har:.4f}')
    print(f'  Hybrid:       {q_hybrid:.4f}')
    print(f'  ML component: {ml_name}')
    print(f'  w_har:        {w_har:.3f}')

    # Save
    Path('models/hybrid').mkdir(parents=True, exist_ok=True)
    hybrid_data = {
        'har_model': har,
        'ml_model': ml_model,
        'ml_type': ml_name,
        'w_har': w_har,
        'best_params': best_params,
        'val_qlike': study.best_value,
        'test_qlike': q_hybrid
    }
    with open(f'models/hybrid/model_h{h}.pkl', 'wb') as f:
        pickle.dump(hybrid_data, f)

    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = final_pred
    pred_df.to_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet', index=False)

# Final summary
print('\n========== FINAL SUMMARY ==========')
for h in [1, 5, 22]:
    df = pd.read_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet')
    q = qlike(df['rv_actual'].values, df['rv_pred'].values)
    with open(f'models/hybrid/model_h{h}.pkl', 'rb') as f:
        info = pickle.load(f)
    print(f'H={h}: QLIKE={q:.4f}, ML={info["ml_type"]}, w_har={info["w_har"]:.3f}')

print('DONE!')
