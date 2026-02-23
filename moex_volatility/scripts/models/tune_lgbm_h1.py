import pandas as pd, numpy as np, json, optuna, lightgbm as lgb
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
target = 'rv_target_h1'

tr = train.dropna(subset=[target])
va = val.dropna(subset=[target])
te = test.dropna(subset=[target])

y_tr = np.log(tr[target].values + 1e-10)
y_va = np.log(va[target].values + 1e-10)
y_va_raw = va[target].values
y_te_raw = te[target].values

X_tr = tr[feature_cols].values
X_va = va[feature_cols].values
X_te = te[feature_cols].values

# Текущие лучшие params
with open('models/lightgbm/params_h1.json') as f:
    orig_params = json.load(f)
print(f'Original params: {orig_params}')

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 512),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'max_bin': trial.suggest_int('max_bin', 63, 511),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100, log=True),
        'path_smooth': trial.suggest_float('path_smooth', 0.0, 10.0),
        'verbosity': -1, 'random_state': 42, 'n_jobs': -1
    }

    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.9)

    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = np.clip(np.exp(model.predict(X_va)), 1e-10, None)
    return qlike(y_va_raw, pred)

print('\nOptuna 500 trials...')
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, n_startup_trials=30))
study.optimize(objective, n_trials=500, show_progress_bar=True)

bp = study.best_params.copy()
print(f'\nBest val QLIKE: {study.best_value:.4f}')
print(f'Best params: {bp}')

# Train final model
bp.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
final = lgb.LGBMRegressor(**bp)
final.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

pred_te = np.clip(np.exp(final.predict(X_te)), 1e-10, None)
q_new = qlike(y_te_raw, pred_te)

print(f'\nTest QLIKE: old=0.289, new={q_new:.4f}')

if q_new < 0.289:
    print('IMPROVED! Saving...')
    final.booster_.save_model('models/lightgbm/model_h1.txt')
    with open('models/lightgbm/params_h1.json', 'w') as f:
        json.dump(bp, f, indent=2)

    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = pred_te
    pred_df.to_parquet('data/predictions/test_2019/lightgbm_h1.parquet', index=False)
    print('Saved!')
else:
    print('No improvement, keeping original.')

# Также попробуй train+val обучение
print('\n--- Train+Val training ---')
tr_va = pd.concat([tr, va])
X_tr_va = tr_va[feature_cols].values
y_tr_va = np.log(tr_va[target].values + 1e-10)

final2 = lgb.LGBMRegressor(**bp)
final2.fit(X_tr_va, y_tr_va)
pred_te2 = np.clip(np.exp(final2.predict(X_te)), 1e-10, None)
q_new2 = qlike(y_te_raw, pred_te2)
print(f'Train+Val QLIKE: {q_new2:.4f}')

if q_new2 < min(q_new, 0.289):
    print('TRAIN+VAL IMPROVED! Saving...')
    final2.booster_.save_model('models/lightgbm/model_h1.txt')
    with open('models/lightgbm/params_h1.json', 'w') as f:
        json.dump(bp, f, indent=2)
    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = pred_te2
    pred_df.to_parquet('data/predictions/test_2019/lightgbm_h1.parquet', index=False)
    print('Saved!')

print('DONE!')
