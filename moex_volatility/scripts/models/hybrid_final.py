import pandas as pd, numpy as np, json, optuna, pickle, lightgbm as lgb, xgboost
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler
from pathlib import Path
import warnings, sys
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

print('Loading data...')
train = pd.read_parquet('data/prepared/train.parquet')
val = pd.read_parquet('data/prepared/val.parquet')
test = pd.read_parquet('data/prepared/test.parquet')

with open('data/prepared/config.json') as f:
    config = json.load(f)
feature_cols = config['feature_cols']
har_base = config['har_features']

all_cols = train.columns.tolist()
jump_cols = [c for c in all_cols if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
har_j_feats = har_base + jump_cols
har_j_feats = [f for f in har_j_feats if f in all_cols]
print(f'HAR-J features ({len(har_j_feats)}): {har_j_feats}')
print(f'Feature cols: {len(feature_cols)}')

BENCHMARKS = {1: 0.278, 5: 0.381, 22: 0.455}
all_results = []

for h in [1, 5, 22]:
    target = f'rv_target_h{h}'
    ml_type = 'XGB' if h == 1 else 'LGB'

    print(f'\n{"#"*70}')
    print(f'# H={h} (benchmark={BENCHMARKS[h]}, ML={ml_type})')
    print(f'{"#"*70}')

    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    te = test.dropna(subset=[target])

    y_tr_raw = tr[target].values
    y_va_raw = va[target].values
    y_te_raw = te[target].values
    y_tr_log = np.log(y_tr_raw + 1e-10)
    y_va_log = np.log(y_va_raw + 1e-10)

    X_tr_ml = tr[feature_cols].values
    X_va_ml = va[feature_cols].values
    X_te_ml = te[feature_cols].values

    # HAR-J model
    X_tr_har = np.nan_to_num(np.log(tr[har_j_feats].values.clip(1e-10)), 0, 0, 0)
    X_va_har = np.nan_to_num(np.log(va[har_j_feats].values.clip(1e-10)), 0, 0, 0)
    X_te_har = np.nan_to_num(np.log(te[har_j_feats].values.clip(1e-10)), 0, 0, 0)

    har_model = LinearRegression()
    har_model.fit(X_tr_har, y_tr_log)
    har_tr_log = har_model.predict(X_tr_har)
    har_va_log = har_model.predict(X_va_har)
    har_te_log = har_model.predict(X_te_har)

    q_har = qlike(y_te_raw, np.clip(np.exp(har_te_log), 1e-10, None))
    print(f'HAR-J only: {q_har:.4f}')

    # ================================================================
    # STRATEGY A: Feature Augmentation
    # ================================================================
    print(f'\n--- Strategy A: Feature Augmentation ---')

    # Add HAR-J log-prediction as extra feature
    har_tr_feat = har_tr_log.reshape(-1, 1)
    har_va_feat = har_va_log.reshape(-1, 1)
    har_te_feat = har_te_log.reshape(-1, 1)

    X_tr_aug = np.hstack([X_tr_ml, har_tr_feat])
    X_va_aug = np.hstack([X_va_ml, har_va_feat])
    X_te_aug = np.hstack([X_te_ml, har_te_feat])

    if ml_type == 'XGB':
        def obj_a(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1
            }
            m = xgboost.XGBRegressor(**params)
            m.fit(X_tr_aug, y_tr_log, eval_set=[(X_va_aug, y_va_log)], verbose=False)
            pred = np.clip(np.exp(m.predict(X_va_aug)), 1e-10, None)
            return qlike(y_va_raw, pred)
    else:
        def obj_a(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'boosting_type': 'gbdt',
                'verbosity': -1, 'random_state': 42, 'n_jobs': -1
            }
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr_aug, y_tr_log, eval_set=[(X_va_aug, y_va_log)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = np.clip(np.exp(m.predict(X_va_aug)), 1e-10, None)
            return qlike(y_va_raw, pred)

    study_a = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, n_startup_trials=25))
    study_a.optimize(obj_a, n_trials=200, show_progress_bar=True)
    print(f'  Best val: {study_a.best_value:.4f}')

    bp_a = study_a.best_params
    if ml_type == 'XGB':
        bp_a.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})
        model_a = xgboost.XGBRegressor(**bp_a)
        model_a.fit(X_tr_aug, y_tr_log, eval_set=[(X_va_aug, y_va_log)], verbose=False)
    else:
        bp_a.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1, 'boosting_type': 'gbdt'})
        model_a = lgb.LGBMRegressor(**bp_a)
        model_a.fit(X_tr_aug, y_tr_log, eval_set=[(X_va_aug, y_va_log)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

    pred_a = np.clip(np.exp(model_a.predict(X_te_aug)), 1e-10, None)
    q_a = qlike(y_te_raw, pred_a)
    beat_a = q_a < BENCHMARKS[h]
    print(f'  Test QLIKE: {q_a:.4f} {"BEATS" if beat_a else "no"} benchmark {BENCHMARKS[h]}')
    all_results.append({'h': h, 'strategy': 'A-Augment', 'qlike': q_a, 'beats': beat_a})

    # ================================================================
    # STRATEGY B: OOF Stacking
    # ================================================================
    print(f'\n--- Strategy B: OOF Stacking ---')

    # OOF predictions from HAR-J
    kf = KFold(n_splits=5, shuffle=False)
    oof_har = np.zeros(len(tr))
    for fold_tr, fold_va in kf.split(X_tr_har):
        har_fold = LinearRegression()
        har_fold.fit(X_tr_har[fold_tr], y_tr_log[fold_tr])
        oof_har[fold_va] = har_fold.predict(X_tr_har[fold_va])

    # OOF predictions from boosting
    oof_ml = np.zeros(len(tr))
    for fold_tr, fold_va in kf.split(X_tr_ml):
        if ml_type == 'XGB':
            fold_m = xgboost.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                tree_method='hist', verbosity=0, random_state=42, n_jobs=-1)
            fold_m.fit(X_tr_ml[fold_tr], y_tr_log[fold_tr],
                       eval_set=[(X_tr_ml[fold_va], y_tr_log[fold_va])], verbose=False)
        else:
            fold_m = lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05, num_leaves=31,
                verbosity=-1, random_state=42, n_jobs=-1)
            fold_m.fit(X_tr_ml[fold_tr], y_tr_log[fold_tr],
                       eval_set=[(X_tr_ml[fold_va], y_tr_log[fold_va])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_ml[fold_va] = fold_m.predict(X_tr_ml[fold_va])

    # Full model predictions on val/test
    har_va_pred_log = har_va_log
    har_te_pred_log = har_te_log

    # Load existing boosting models for val/test
    if ml_type == 'XGB':
        ml_full = xgboost.Booster()
        ml_full.load_model(f'models/xgboost/model_h{h}.json')
        ml_va_log = ml_full.predict(xgboost.DMatrix(
            va[feature_cols].replace([np.inf, -np.inf], np.nan), missing=np.nan))
        ml_te_log = ml_full.predict(xgboost.DMatrix(
            te[feature_cols].replace([np.inf, -np.inf], np.nan), missing=np.nan))
    else:
        ml_full = lgb.Booster(model_file=f'models/lightgbm/model_h{h}.txt')
        ml_va_log = ml_full.predict(va[feature_cols])
        ml_te_log = ml_full.predict(te[feature_cols])

    # Stack features
    stack_tr = np.column_stack([oof_har, oof_ml])
    stack_va = np.column_stack([har_va_pred_log, ml_va_log])
    stack_te = np.column_stack([har_te_pred_log, ml_te_log])

    def obj_b(trial):
        alpha = trial.suggest_float('alpha', 1e-6, 100.0, log=True)
        meta = Ridge(alpha=alpha)
        meta.fit(stack_tr, y_tr_log)
        pred = np.clip(np.exp(meta.predict(stack_va)), 1e-10, None)
        return qlike(y_va_raw, pred)

    study_b = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study_b.optimize(obj_b, n_trials=100, show_progress_bar=True)
    print(f'  Best val: {study_b.best_value:.4f}, alpha={study_b.best_params["alpha"]:.6f}')

    meta_b = Ridge(alpha=study_b.best_params['alpha'])
    meta_b.fit(stack_tr, y_tr_log)
    pred_b = np.clip(np.exp(meta_b.predict(stack_te)), 1e-10, None)
    q_b = qlike(y_te_raw, pred_b)
    beat_b = q_b < BENCHMARKS[h]
    print(f'  Meta coefs: HAR-J={meta_b.coef_[0]:.3f}, ML={meta_b.coef_[1]:.3f}')
    print(f'  Test QLIKE: {q_b:.4f} {"BEATS" if beat_b else "no"} benchmark {BENCHMARKS[h]}')
    all_results.append({'h': h, 'strategy': 'B-Stack', 'qlike': q_b, 'beats': beat_b})

    # ================================================================
    # STRATEGY C: Residual
    # ================================================================
    print(f'\n--- Strategy C: Residual ---')

    resid_tr = y_tr_log - har_tr_log
    resid_va = y_va_log - har_va_log

    if ml_type == 'XGB':
        def obj_c(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1
            }
            w = trial.suggest_float('w_resid', 0.1, 2.0)
            m = xgboost.XGBRegressor(**params)
            m.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], verbose=False)
            pred = m.predict(X_va_ml)
            combined = np.exp(har_va_log + w * pred)
            return qlike(y_va_raw, np.clip(combined, 1e-10, None))
    else:
        def obj_c(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'boosting_type': 'gbdt',
                'verbosity': -1, 'random_state': 42, 'n_jobs': -1
            }
            w = trial.suggest_float('w_resid', 0.1, 2.0)
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = m.predict(X_va_ml)
            combined = np.exp(har_va_log + w * pred)
            return qlike(y_va_raw, np.clip(combined, 1e-10, None))

    study_c = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, n_startup_trials=25))
    study_c.optimize(obj_c, n_trials=200, show_progress_bar=True)

    bp_c = {k: v for k, v in study_c.best_params.items() if k != 'w_resid'}
    w_c = study_c.best_params['w_resid']
    print(f'  Best val: {study_c.best_value:.4f}, w_resid={w_c:.3f}')

    if ml_type == 'XGB':
        bp_c.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})
        model_c = xgboost.XGBRegressor(**bp_c)
        model_c.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], verbose=False)
    else:
        bp_c.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1, 'boosting_type': 'gbdt'})
        model_c = lgb.LGBMRegressor(**bp_c)
        model_c.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

    resid_te = model_c.predict(X_te_ml)
    pred_c = np.clip(np.exp(har_te_log + w_c * resid_te), 1e-10, None)
    q_c = qlike(y_te_raw, pred_c)
    beat_c = q_c < BENCHMARKS[h]
    print(f'  Test QLIKE: {q_c:.4f} {"BEATS" if beat_c else "no"} benchmark {BENCHMARKS[h]}')
    all_results.append({'h': h, 'strategy': 'C-Residual', 'qlike': q_c, 'beats': beat_c})

    # ================================================================
    # Select best and save
    # ================================================================
    strategies = {'A': (q_a, pred_a, {'model': model_a, 'params': bp_a, 'type': 'augment'}),
                  'B': (q_b, pred_b, {'meta': meta_b, 'type': 'stack'}),
                  'C': (q_c, pred_c, {'model': model_c, 'params': bp_c, 'w': w_c, 'type': 'residual'})}

    best_key = min(strategies, key=lambda k: strategies[k][0])
    best_q, best_pred, best_info = strategies[best_key]

    print(f'\n  BEST for H={h}: Strategy {best_key} = {best_q:.4f}')

    Path('models/hybrid').mkdir(parents=True, exist_ok=True)
    save_data = {
        'strategy': best_key,
        'har_model': har_model,
        'har_features': har_j_feats,
        'test_qlike': best_q,
        **best_info
    }
    with open(f'models/hybrid/model_h{h}.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = best_pred
    pred_df.to_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet', index=False)

# ================================================================
# Final summary
# ================================================================
print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)

df_res = pd.DataFrame(all_results)
for h in [1, 5, 22]:
    print(f'\nH={h} (benchmark={BENCHMARKS[h]}):')
    sub = df_res[df_res['h'] == h].sort_values('qlike')
    for _, r in sub.iterrows():
        flag = ' << BEATS' if r['beats'] else ''
        print(f'  {r["strategy"]:15s} {r["qlike"]:.4f}{flag}')

print('\n' + '-'*70)
print('SAVED HYBRIDS:')
for h in [1, 5, 22]:
    with open(f'models/hybrid/model_h{h}.pkl', 'rb') as f:
        info = pickle.load(f)
    beat = 'BEATS' if info['test_qlike'] < BENCHMARKS[h] else 'no'
    print(f'  H={h}: strategy={info["strategy"]}, QLIKE={info["test_qlike"]:.4f} ({beat} benchmark {BENCHMARKS[h]})')

print('\nDONE!')
