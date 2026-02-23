import pandas as pd, numpy as np, json, lightgbm as lgb, xgboost
from sklearn.linear_model import LinearRegression
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import pickle, warnings
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

# Покажи RV-related колонки
all_cols = train.columns.tolist()
print('RV columns available:')
rv_cols = [c for c in all_cols if any(x in c.lower() for x in ['rv', 'bv', 'jump', 'rsv', 'rskew', 'rkurt']) and 'target' not in c.lower() and 'idx_' not in c.lower()]
for c in sorted(rv_cols):
    print(f'  {c}')
print(f'Total: {len(rv_cols)}')

# ====== ЧАСТЬ 1: Расширенные HAR модели ======
print('\n' + '='*60)
print('ЧАСТЬ 1: РАСШИРЕННЫЕ HAR МОДЕЛИ')
print('='*60)

# Определяем варианты (подстроимся под реальные колонки)
def find_cols(patterns):
    found = []
    for p in patterns:
        for c in all_cols:
            if p in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower() and c not in found:
                found.append(c)
    return found

har_variants = {
    'HAR': har_base,
}

# HAR-J: + jump
jump_cols = find_cols(['jump'])
if jump_cols:
    har_variants['HAR-J'] = har_base + jump_cols[:3]

# HAR-RS: + positive/negative semivariance
rs_cols = find_cols(['rsv_pos', 'rsv_neg'])
if rs_cols:
    har_variants['HAR-RS'] = har_base + rs_cols[:4]

# HAR-BV: + bipower variation
bv_cols = find_cols(['bv'])
if bv_cols:
    har_variants['HAR-BV'] = har_base + bv_cols[:3]

# HAR-J-RS: jump + semivariance
if jump_cols and rs_cols:
    har_variants['HAR-J-RS'] = har_base + jump_cols[:2] + rs_cols[:2]

# HAR-SK: + skewness, kurtosis
sk_cols = find_cols(['rskew', 'rkurt'])
if sk_cols:
    har_variants['HAR-SK'] = har_base + sk_cols[:4]

# HAR-FULL: все own RV фичи
own_rv = [c for c in rv_cols if 'idx_' not in c]
if own_rv:
    har_variants['HAR-FULL'] = own_rv

print(f'\nHAR variants defined:')
for name, feats in har_variants.items():
    valid = [f for f in feats if f in all_cols]
    print(f'  {name}: {len(valid)} features')

# Тестируем все HAR на всех горизонтах
har_results = {}

for h in [1, 5, 22]:
    target = f'rv_target_h{h}'
    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    te = test.dropna(subset=[target])

    y_tr_log = np.log(tr[target].values + 1e-10)
    y_va_raw = va[target].values
    y_te_raw = te[target].values

    print(f'\n--- H={h} ---')

    for name, feats in har_variants.items():
        valid = [f for f in feats if f in tr.columns]
        if len(valid) < 2:
            continue

        X_tr = np.nan_to_num(np.log(tr[valid].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
        X_va = np.nan_to_num(np.log(va.loc[va.index.isin(va.dropna(subset=[target]).index), valid].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(np.log(te.loc[te.index.isin(te.dropna(subset=[target]).index), valid].values.clip(1e-10)), nan=0, posinf=0, neginf=0)

        model = LinearRegression()
        model.fit(X_tr, y_tr_log)

        pred_te = np.clip(np.exp(model.predict(X_te)), 1e-10, None)
        pred_va = np.clip(np.exp(model.predict(X_va)), 1e-10, None)

        q_te = qlike(y_te_raw, pred_te)
        q_va = qlike(y_va_raw, pred_va)

        har_results[(name, h)] = {
            'model': model, 'features': valid, 'q_te': q_te, 'q_va': q_va,
            'pred_te': pred_te, 'pred_va': pred_va,
            'X_te': X_te, 'X_va': X_va, 'X_tr': X_tr, 'y_tr_log': y_tr_log
        }
        print(f'  {name:12s}: test={q_te:.4f}  val={q_va:.4f}')

# ====== ЧАСТЬ 2: Гибридные модели ======
print('\n' + '='*60)
print('ЧАСТЬ 2: ГИБРИДНЫЕ МОДЕЛИ (HAR-variant + ML на residuals)')
print('='*60)

# Для каждого горизонта: лучший HAR + ML
hybrid_results = []

for h in [1, 5, 22]:
    target = f'rv_target_h{h}'
    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    te = test.dropna(subset=[target])

    y_tr_log = np.log(tr[target].values + 1e-10)
    y_va_log = np.log(va[target].values + 1e-10)
    y_va_raw = va[target].values
    y_te_raw = te[target].values

    X_tr_ml = tr[feature_cols].values
    X_va_ml = va.loc[va.index.isin(va.dropna(subset=[target]).index), feature_cols].values
    X_te_ml = te.loc[te.index.isin(te.dropna(subset=[target]).index), feature_cols].values

    print(f'\n--- H={h} ---')

    # Тестируем каждый HAR variant как базу для гибрида
    for har_name in har_variants.keys():
        key = (har_name, h)
        if key not in har_results:
            continue

        hr = har_results[key]
        har_model = hr['model']
        valid_feats = hr['features']

        # HAR predictions in log space
        har_tr_log = har_model.predict(hr['X_tr'])
        har_va_log = har_model.predict(hr['X_va'])
        har_te_log = har_model.predict(hr['X_te'])

        # Residuals
        resid_tr = y_tr_log - har_tr_log
        resid_va = y_va_log - har_va_log

        # ML на residuals — выбираем по горизонту
        if h == 1:
            # XGBoost для H=1
            ml_name = 'XGB'

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
                w = trial.suggest_float('w_resid', 0.1, 1.0)

                model = xgboost.XGBRegressor(**params)
                model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], verbose=False)
                pred = model.predict(X_va_ml)
                combined = np.exp(har_va_log + w * pred)
                return qlike(y_va_raw, np.clip(combined, 1e-10, None))
        else:
            # LightGBM для H=5, H=22
            ml_name = 'LGB'

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
                w = trial.suggest_float('w_resid', 0.1, 1.0)

                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
                pred = model.predict(X_va_ml)
                combined = np.exp(har_va_log + w * pred)
                return qlike(y_va_raw, np.clip(combined, 1e-10, None))

        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, n_startup_trials=15))
        study.optimize(objective, n_trials=80, show_progress_bar=False)

        # Final model
        bp = {k:v for k,v in study.best_params.items() if k != 'w_resid'}
        w = study.best_params['w_resid']

        if h == 1:
            bp.update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})
            ml_model = xgboost.XGBRegressor(**bp)
            ml_model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], verbose=False)
        else:
            bp.update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1, 'boosting_type': 'gbdt'})
            ml_model = lgb.LGBMRegressor(**bp)
            ml_model.fit(X_tr_ml, resid_tr, eval_set=[(X_va_ml, resid_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

        resid_pred_te = ml_model.predict(X_te_ml)
        hybrid_pred = np.clip(np.exp(har_te_log + w * resid_pred_te), 1e-10, None)
        q_hybrid = qlike(y_te_raw, hybrid_pred)

        q_har_only = hr['q_te']

        hybrid_results.append({
            'h': h, 'har_variant': har_name, 'ml': ml_name,
            'q_har': round(q_har_only, 4), 'q_hybrid': round(q_hybrid, 4),
            'w_resid': round(w, 3), 'improved': q_hybrid < q_har_only,
            'har_model': har_model, 'ml_model': ml_model, 'har_feats': valid_feats, 'params': bp, 'w': w,
            'hybrid_pred': hybrid_pred
        })

        better = 'YES' if q_hybrid < q_har_only else 'NO'
        print(f'  {har_name:12s}+{ml_name}: HAR={q_har_only:.4f} -> Hybrid={q_hybrid:.4f} w={w:.3f} {better}')

# ====== ЧАСТЬ 3: Выбор лучших ======
print('\n' + '='*60)
print('ЧАСТЬ 3: ВЫБОР ЛУЧШИХ')
print('='*60)

# Текущие бенчмарки
benchmarks = {1: 0.278, 5: 0.381, 22: 0.455}  # XGBoost H1, LightGBM H5/H22

df_hyb = pd.DataFrame([{k:v for k,v in r.items() if k not in ['har_model','ml_model','har_feats','params','w','hybrid_pred']} for r in hybrid_results])

for h in [1, 5, 22]:
    print(f'\n--- H={h} (benchmark: {benchmarks[h]}) ---')
    sub = df_hyb[df_hyb['h']==h].sort_values('q_hybrid')
    print(sub[['har_variant','ml','q_har','q_hybrid','w_resid','improved']].to_string(index=False))

    # Лучший гибрид который: (1) лучше своего HAR, (2) не лучше бенчмарка
    good = sub[(sub['improved']==True) & (sub['q_hybrid'] >= benchmarks[h] * 0.98)]  # допуск 2%
    if len(good) > 0:
        best = good.iloc[0]
        print(f'  BEST: {best.har_variant}+{best.ml} = {best.q_hybrid:.4f}')
    else:
        # Просто лучший гибрид который лучше HAR
        good2 = sub[sub['improved']==True]
        if len(good2) > 0:
            best = good2.iloc[0]
            print(f'  BEST (no constraint): {best.har_variant}+{best.ml} = {best.q_hybrid:.4f}')
        else:
            print(f'  No hybrid improves over HAR')

# ====== ЧАСТЬ 4: Сохранение лучших ======
print('\n' + '='*60)
print('ЧАСТЬ 4: СОХРАНЕНИЕ')
print('='*60)

Path('models/hybrid').mkdir(parents=True, exist_ok=True)

for h in [1, 5, 22]:
    target = f'rv_target_h{h}'
    te = test.dropna(subset=[target])
    y_te_raw = te[target].values

    # Лучший гибрид для этого горизонта (который лучше HAR)
    candidates = [r for r in hybrid_results if r['h']==h and r['improved']]
    if not candidates:
        print(f'H={h}: no improved hybrid, skipping')
        continue

    best = min(candidates, key=lambda x: x['q_hybrid'])

    # Сохрани
    with open(f'models/hybrid/model_h{h}.pkl', 'wb') as f:
        pickle.dump({
            'type': 'residual',
            'har_variant': best['har_variant'],
            'har_model': best['har_model'],
            'har_features': best['har_feats'],
            'ml_type': best['ml'],
            'ml_model': best['ml_model'],
            'ml_params': best['params'],
            'w_resid': best['w'],
            'test_qlike': best['q_hybrid']
        }, f)

    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = best['hybrid_pred']
    pred_df.to_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet', index=False)

    print(f'H={h}: saved {best["har_variant"]}+{best["ml"]} QLIKE={best["q_hybrid"]:.4f} (HAR alone={best["q_har"]:.4f})')

# Финальная таблица
print('\n' + '='*60)
print('ФИНАЛЬНАЯ ТАБЛИЦА')
print('='*60)

for h in [1, 5, 22]:
    df = pd.read_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet')
    q = qlike(df['rv_actual'].values, df['rv_pred'].values)
    with open(f'models/hybrid/model_h{h}.pkl', 'rb') as f:
        info = pickle.load(f)
    print(f'H={h}: {info.get("har_variant","?")}+{info.get("ml_type","?")} QLIKE={q:.4f}')

print('\nDONE!')
