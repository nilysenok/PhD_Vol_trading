import pandas as pd, numpy as np, json, optuna, pickle, lightgbm as lgb, xgboost
from sklearn.linear_model import LinearRegression, Ridge
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
jump_cols = [c for c in all_cols if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
harj_features = har_base + jump_cols[:5]
harj_features = [f for f in harj_features if f in all_cols]

target = 'rv_target_h1'

# Объединяем train+val для TimeSeriesCV
full = pd.concat([train, val]).dropna(subset=[target]).sort_values('date').reset_index(drop=True)
te = test.dropna(subset=[target])

y_te_raw = te[target].values
X_te_ml = te[feature_cols].values
X_te_har = np.nan_to_num(np.log(te[harj_features].values.clip(1e-10)), 0, 0, 0)

print(f'Full train+val: {len(full)}, Test: {len(te)}')
print(f'Date range: {full.date.min()} to {full.date.max()}')

# TimeSeriesCV: 4 expanding splits
dates = full['date'].unique()
n_dates = len(dates)
splits = []

for tr_frac, va_frac in [(0.5, 0.2), (0.6, 0.15), (0.7, 0.15), (0.8, 0.2)]:
    tr_end = int(n_dates * tr_frac)
    va_end = int(n_dates * (tr_frac + va_frac))
    tr_dates = set(dates[:tr_end])
    va_dates = set(dates[tr_end:va_end])
    tr_mask = full['date'].isin(tr_dates)
    va_mask = full['date'].isin(va_dates)
    splits.append((full[tr_mask].index.values, full[va_mask].index.values))
    print(f'Split: train {sum(tr_mask)}, val {sum(va_mask)}')

# Загружаем params из сохранённых моделей
with open('models/xgboost/params_h1.json') as f:
    xgb_params = json.load(f)
xgb_params.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})

with open('models/lightgbm/params_h1.json') as f:
    lgb_params = json.load(f)
lgb_params.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})

X_full_ml = full[feature_cols].values
X_full_har = np.nan_to_num(np.log(full[harj_features].values.clip(1e-10)), 0, 0, 0)
y_full_log = np.log(full[target].values + 1e-10)
y_full_raw = full[target].values

# ====== APPROACH 1: TSCV Stacking с Ridge ======
print('\n===== APPROACH 1: TSCV Ridge Stacking =====')

def get_predictions_for_split(tr_idx, va_idx):
    # HAR-J
    har = LinearRegression()
    har.fit(X_full_har[tr_idx], y_full_log[tr_idx])
    har_pred = har.predict(X_full_har[va_idx])

    # XGBoost
    xgb_m = xgboost.XGBRegressor(**xgb_params)
    xgb_m.fit(X_full_ml[tr_idx], y_full_log[tr_idx], eval_set=[(X_full_ml[va_idx], y_full_log[va_idx])], verbose=False)
    xgb_pred = xgb_m.predict(X_full_ml[va_idx])

    # LightGBM
    lgb_m = lgb.LGBMRegressor(**lgb_params)
    lgb_m.fit(X_full_ml[tr_idx], y_full_log[tr_idx], eval_set=[(X_full_ml[va_idx], y_full_log[va_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_pred = lgb_m.predict(X_full_ml[va_idx])

    return har_pred, xgb_pred, lgb_pred

# Собираем OOF predictions по всем splits
oof_meta_X = []
oof_meta_y = []

for i, (tr_idx, va_idx) in enumerate(splits):
    har_p, xgb_p, lgb_p = get_predictions_for_split(tr_idx, va_idx)
    oof_meta_X.append(np.column_stack([har_p, xgb_p, lgb_p]))
    oof_meta_y.append(y_full_log[va_idx])
    print(f'  Split {i+1}: HAR={qlike(y_full_raw[va_idx], np.clip(np.exp(har_p),1e-10,None)):.4f}, XGB={qlike(y_full_raw[va_idx], np.clip(np.exp(xgb_p),1e-10,None)):.4f}, LGB={qlike(y_full_raw[va_idx], np.clip(np.exp(lgb_p),1e-10,None)):.4f}')

meta_X_all = np.vstack(oof_meta_X)
meta_y_all = np.concatenate(oof_meta_y)

# Full models for test prediction
har_full = LinearRegression()
har_full.fit(X_full_har, y_full_log)
har_te = har_full.predict(X_te_har)

xgb_params_noES = {k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'}
xgb_full_m = xgboost.XGBRegressor(**xgb_params_noES)
xgb_full_m.fit(X_full_ml, y_full_log)
xgb_te = xgb_full_m.predict(X_te_ml)

lgb_full_m = lgb.LGBMRegressor(**lgb_params)
lgb_full_m.fit(X_full_ml, y_full_log)
lgb_te = lgb_full_m.predict(X_te_ml)

meta_X_te = np.column_stack([har_te, xgb_te, lgb_te])

# Ridge на TSCV OOF
def obj_ridge(trial):
    alpha = trial.suggest_float('alpha', 1e-10, 1000, log=True)
    meta = Ridge(alpha=alpha)
    # CV score на splits
    scores = []
    for i, (_, va_idx) in enumerate(splits):
        # Обучаем meta на всех OOF КРОМЕ текущего split
        other_X = np.vstack([oof_meta_X[j] for j in range(len(splits)) if j != i])
        other_y = np.concatenate([oof_meta_y[j] for j in range(len(splits)) if j != i])
        meta.fit(other_X, other_y)
        pred = np.exp(meta.predict(oof_meta_X[i]))
        scores.append(qlike(y_full_raw[va_idx], np.clip(pred, 1e-10, None)))
    return np.mean(scores)

study_r = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_r.optimize(obj_ridge, n_trials=300)

meta_r = Ridge(alpha=study_r.best_params['alpha'])
meta_r.fit(meta_X_all, meta_y_all)
pred_r = np.clip(np.exp(meta_r.predict(meta_X_te)), 1e-10, None)
q_r = qlike(y_te_raw, pred_r)
print(f'TSCV Ridge: {q_r:.4f}, weights={meta_r.coef_}, intercept={meta_r.intercept_:.4f}')

# ====== APPROACH 2: TSCV Direct Weights (log-space) ======
print('\n===== APPROACH 2: TSCV Direct Weights =====')

def obj_w(trial):
    w1 = trial.suggest_float('w_har', 0.0, 0.3)
    w2 = trial.suggest_float('w_xgb', 0.0, 1.0)
    w3 = trial.suggest_float('w_lgb', 0.0, 1.0)
    total = w1 + w2 + w3
    if total < 0.01: return 999.0
    w1, w2, w3 = w1/total, w2/total, w3/total

    scores = []
    for i, (_, va_idx) in enumerate(splits):
        blend = w1 * oof_meta_X[i][:,0] + w2 * oof_meta_X[i][:,1] + w3 * oof_meta_X[i][:,2]
        pred = np.exp(blend)
        scores.append(qlike(y_full_raw[va_idx], np.clip(pred, 1e-10, None)))
    return np.mean(scores)

study_w = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_w.optimize(obj_w, n_trials=500)

w1 = study_w.best_params['w_har']
w2 = study_w.best_params['w_xgb']
w3 = study_w.best_params['w_lgb']
t = w1+w2+w3
w1,w2,w3 = w1/t, w2/t, w3/t

blend_te = np.exp(w1*har_te + w2*xgb_te + w3*lgb_te)
q_w = qlike(y_te_raw, np.clip(blend_te, 1e-10, None))
print(f'TSCV Weights (log): {q_w:.4f}, w_har={w1:.3f}, w_xgb={w2:.3f}, w_lgb={w3:.3f}')

# ====== APPROACH 3: TSCV Direct Weights (exp-space) ======
print('\n===== APPROACH 3: TSCV Direct Weights (exp) =====')

def obj_we(trial):
    w1 = trial.suggest_float('w_har', 0.0, 0.3)
    w2 = trial.suggest_float('w_xgb', 0.0, 1.0)
    w3 = trial.suggest_float('w_lgb', 0.0, 1.0)
    total = w1 + w2 + w3
    if total < 0.01: return 999.0
    w1, w2, w3 = w1/total, w2/total, w3/total

    scores = []
    for i, (_, va_idx) in enumerate(splits):
        pred = w1*np.exp(oof_meta_X[i][:,0]) + w2*np.exp(oof_meta_X[i][:,1]) + w3*np.exp(oof_meta_X[i][:,2])
        scores.append(qlike(y_full_raw[va_idx], np.clip(pred, 1e-10, None)))
    return np.mean(scores)

study_we = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_we.optimize(obj_we, n_trials=500)

w1e = study_we.best_params['w_har']
w2e = study_we.best_params['w_xgb']
w3e = study_we.best_params['w_lgb']
te2 = w1e+w2e+w3e
w1e,w2e,w3e = w1e/te2, w2e/te2, w3e/te2

blend_te_exp = w1e*np.exp(har_te) + w2e*np.exp(xgb_te) + w3e*np.exp(lgb_te)
q_we = qlike(y_te_raw, np.clip(blend_te_exp, 1e-10, None))
print(f'TSCV Weights (exp): {q_we:.4f}, w_har={w1e:.3f}, w_xgb={w2e:.3f}, w_lgb={w3e:.3f}')

# ====== APPROACH 4: Простое среднее XGB+LGB (без подбора) ======
print('\n===== APPROACH 4: Simple averages =====')

avg2 = np.exp(0.5*xgb_te + 0.5*lgb_te)
q_avg2 = qlike(y_te_raw, np.clip(avg2, 1e-10, None))
print(f'XGB+LGB avg (log): {q_avg2:.4f}')

avg2e = 0.5*np.exp(xgb_te) + 0.5*np.exp(lgb_te)
q_avg2e = qlike(y_te_raw, np.clip(avg2e, 1e-10, None))
print(f'XGB+LGB avg (exp): {q_avg2e:.4f}')

avg3e = (np.exp(har_te) + np.exp(xgb_te) + np.exp(lgb_te))/3
q_avg3e = qlike(y_te_raw, np.clip(avg3e, 1e-10, None))
print(f'HAR+XGB+LGB avg (exp): {q_avg3e:.4f}')

# ====== SUMMARY ======
print('\n' + '='*60)
print('ALL H=1 RESULTS (benchmark XGBoost=0.278)')
print('='*60)

results = {
    'TSCV_Ridge': (q_r, pred_r),
    'TSCV_Weights_log': (q_w, blend_te),
    'TSCV_Weights_exp': (q_we, blend_te_exp),
    'Simple_XGB_LGB_log': (q_avg2, avg2),
    'Simple_XGB_LGB_exp': (q_avg2e, avg2e),
    'Simple_3model_exp': (q_avg3e, avg3e),
}

for name, (q, _) in sorted(results.items(), key=lambda x: x[1][0]):
    beat = 'BEATS 0.278' if q < 0.278 else ''
    print(f'  {name:25s}: {q:.4f} {beat}')

best_name = min(results, key=lambda x: results[x][0])
best_q, best_pred = results[best_name]

print(f'\nBEST: {best_name} = {best_q:.4f}')

# Save
pred_df = te[['date', 'ticker']].copy()
pred_df['rv_actual'] = y_te_raw
pred_df['rv_pred'] = np.clip(best_pred, 1e-10, None)
pred_df.to_parquet('data/predictions/test_2019/hybrid_h1.parquet', index=False)

with open('models/hybrid/model_h1.pkl', 'wb') as f:
    pickle.dump({'strategy': best_name, 'test_qlike': best_q, 'all_results': {k:v[0] for k,v in results.items()}}, f)

print('Saved!')
print('DONE!')
