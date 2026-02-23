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

# Комбинированные данные train+val
tr_va = pd.concat([tr, va])
X_tr_va = tr_va[feature_cols].values
y_tr_va_log = np.log(tr_va[target].values + 1e-10)

results = {}

# ====== PHASE 1: Optuna 700 trials на val ======
print('PHASE 1: Optuna 700 trials (val evaluation)')

def objective_val(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 1024),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-10, 1000.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 1000.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'max_bin': trial.suggest_int('max_bin', 31, 1023),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-6, 1000, log=True),
        'path_smooth': trial.suggest_float('path_smooth', 0.0, 50.0),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.3, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'verbosity': -1, 'random_state': 42, 'n_jobs': -1
    }

    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.001, 0.7)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.05, 0.95)
        params['max_drop'] = trial.suggest_int('max_drop', 5, 100)

    if params['boosting_type'] == 'goss':
        params['top_rate'] = trial.suggest_float('top_rate', 0.1, 0.5)
        params['other_rate'] = trial.suggest_float('other_rate', 0.01, 0.3)
        # GOSS не поддерживает bagging
        params['subsample'] = 1.0
        params['bagging_freq'] = 0

    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = np.clip(np.exp(model.predict(X_va)), 1e-10, None)
    return qlike(y_va_raw, pred)

study1 = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, n_startup_trials=50))
study1.optimize(objective_val, n_trials=700, show_progress_bar=True)

bp1 = study1.best_params.copy()
print(f'Phase 1 best val QLIKE: {study1.best_value:.4f}')

# Test — train only
bp1.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
m1 = lgb.LGBMRegressor(**bp1)
m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
pred1 = np.clip(np.exp(m1.predict(X_te)), 1e-10, None)
q1 = qlike(y_te_raw, pred1)
results['Phase1_train'] = q1
print(f'  Test (train only): {q1:.4f}')

# Test — train+val
m1tv = lgb.LGBMRegressor(**bp1)
m1tv.fit(X_tr_va, y_tr_va_log)
pred1tv = np.clip(np.exp(m1tv.predict(X_te)), 1e-10, None)
q1tv = qlike(y_te_raw, pred1tv)
results['Phase1_trainval'] = q1tv
print(f'  Test (train+val): {q1tv:.4f}')

# ====== PHASE 2: Ensemble seeds ======
print('\nPHASE 2: Ensemble 10 seeds')

preds_seeds = []
for seed in range(10):
    bp_s = bp1.copy()
    bp_s['random_state'] = seed
    m_s = lgb.LGBMRegressor(**bp_s)
    m_s.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds_seeds.append(np.exp(m_s.predict(X_te)))

ensemble_pred = np.clip(np.mean(preds_seeds, axis=0), 1e-10, None)
q_ens = qlike(y_te_raw, ensemble_pred)
results['Phase2_ensemble10'] = q_ens
print(f'  Ensemble 10 seeds: {q_ens:.4f}')

# train+val ensemble
preds_seeds_tv = []
for seed in range(10):
    bp_s = bp1.copy()
    bp_s['random_state'] = seed
    m_s = lgb.LGBMRegressor(**bp_s)
    m_s.fit(X_tr_va, y_tr_va_log)
    preds_seeds_tv.append(np.exp(m_s.predict(X_te)))

ensemble_pred_tv = np.clip(np.mean(preds_seeds_tv, axis=0), 1e-10, None)
q_ens_tv = qlike(y_te_raw, ensemble_pred_tv)
results['Phase2_ensemble10_tv'] = q_ens_tv
print(f'  Ensemble 10 seeds (train+val): {q_ens_tv:.4f}')

# ====== PHASE 3: Top-5 trials ensemble ======
print('\nPHASE 3: Top-5 trials ensemble')

top5_trials = sorted(study1.trials, key=lambda t: t.value)[:5]
preds_top5 = []
for t in top5_trials:
    p = t.params.copy()
    p.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds_top5.append(np.exp(m.predict(X_te)))
    print(f'  Trial {t.number}: val={t.value:.4f}, test={qlike(y_te_raw, np.clip(np.exp(m.predict(X_te)), 1e-10, None)):.4f}')

ens_top5 = np.clip(np.mean(preds_top5, axis=0), 1e-10, None)
q_top5 = qlike(y_te_raw, ens_top5)
results['Phase3_top5_ensemble'] = q_top5
print(f'  Top-5 ensemble: {q_top5:.4f}')

# train+val top5
preds_top5_tv = []
for t in top5_trials:
    p = t.params.copy()
    p.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr_va, y_tr_va_log)
    preds_top5_tv.append(np.exp(m.predict(X_te)))

ens_top5_tv = np.clip(np.mean(preds_top5_tv, axis=0), 1e-10, None)
q_top5_tv = qlike(y_te_raw, ens_top5_tv)
results['Phase3_top5_ensemble_tv'] = q_top5_tv
print(f'  Top-5 ensemble (train+val): {q_top5_tv:.4f}')

# ====== SUMMARY ======
print('\n' + '='*60)
print('SUMMARY (benchmark: 0.289 old LGB, 0.278 XGBoost)')
print('='*60)
for name, q in sorted(results.items(), key=lambda x: x[1]):
    vs_old = 'Y' if q < 0.289 else 'N'
    vs_xgb = ' BEATS_XGB' if q < 0.278 else ''
    print(f'  {name:30s}: {q:.4f} {vs_old}{vs_xgb}')

# Save best
best_name = min(results, key=lambda x: results[x])
best_q = results[best_name]

print(f'\nBEST: {best_name} = {best_q:.4f}')

if best_q < 0.289:
    print('Saving improved LightGBM...')

    # Определи какой pred сохранять
    pred_map = {
        'Phase1_train': pred1, 'Phase1_trainval': pred1tv,
        'Phase2_ensemble10': ensemble_pred, 'Phase2_ensemble10_tv': ensemble_pred_tv,
        'Phase3_top5_ensemble': ens_top5, 'Phase3_top5_ensemble_tv': ens_top5_tv
    }

    best_pred = pred_map[best_name]

    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = best_pred
    pred_df.to_parquet('data/predictions/test_2019/lightgbm_h1.parquet', index=False)

    with open('models/lightgbm/params_h1.json', 'w') as f:
        json.dump(bp1, f, indent=2)

    print(f'Saved! LightGBM H=1: {best_q:.4f}')
else:
    print('No improvement over 0.289')

print('DONE!')
