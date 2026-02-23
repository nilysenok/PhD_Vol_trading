import pandas as pd, numpy as np, json, pickle, optuna
from optuna.samplers import TPESampler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

# Загружаем val predictions из моделей
import lightgbm as lgb, xgboost as xgb_lib
from sklearn.linear_model import LinearRegression

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
print(f'HAR-J features: {harj_features}')

Path('models/hybrid').mkdir(parents=True, exist_ok=True)

# Какой бустинг для какого горизонта
best_ml = {1: 'xgboost', 5: 'lightgbm', 22: 'lightgbm'}

for h in [1, 5, 22]:
    ml_name = best_ml[h]
    target = f'rv_target_h{h}'

    print(f'\n{"="*60}')
    print(f'H={h}: HAR-J + {ml_name}')
    print(f'{"="*60}')

    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    te = test.dropna(subset=[target])

    y_va_raw = va[target].values
    y_te_raw = te[target].values

    # === HAR-J predictions ===
    har_model = LinearRegression()
    X_tr_har = np.nan_to_num(np.log(tr[harj_features].values.clip(1e-10)), 0, 0, 0)
    X_va_har = np.nan_to_num(np.log(va[harj_features].values.clip(1e-10)), 0, 0, 0)
    X_te_har = np.nan_to_num(np.log(te[harj_features].values.clip(1e-10)), 0, 0, 0)

    har_model.fit(X_tr_har, np.log(tr[target].values + 1e-10))
    har_va = np.clip(np.exp(har_model.predict(X_va_har)), 1e-10, None)
    har_te = np.clip(np.exp(har_model.predict(X_te_har)), 1e-10, None)

    # === ML predictions из сохранённых моделей ===
    X_va_ml = np.nan_to_num(va[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_ml = np.nan_to_num(te[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

    if ml_name == 'xgboost':
        ml_model = xgb_lib.Booster()
        ml_model.load_model(f'models/xgboost/model_h{h}.json')
        ml_va = np.clip(np.exp(ml_model.predict(xgb_lib.DMatrix(X_va_ml))), 1e-10, None)
        ml_te = np.clip(np.exp(ml_model.predict(xgb_lib.DMatrix(X_te_ml))), 1e-10, None)
    else:
        ml_model = lgb.Booster(model_file=f'models/lightgbm/model_h{h}.txt')
        ml_va = np.clip(np.exp(ml_model.predict(X_va_ml)), 1e-10, None)
        ml_te = np.clip(np.exp(ml_model.predict(X_te_ml)), 1e-10, None)

    q_har = qlike(y_te_raw, har_te)
    q_ml = qlike(y_te_raw, ml_te)
    print(f'HAR-J:     test={q_har:.4f}')
    print(f'{ml_name}: test={q_ml:.4f}')

    # === Подбор веса на val: w*HAR + (1-w)*ML ===
    print(f'\nGrid search w (val)...')

    best_val_q = 999
    best_w = 0
    for w in np.arange(0.0, 1.001, 0.01):
        blend_va = w * har_va + (1-w) * ml_va
        q = qlike(y_va_raw, np.clip(blend_va, 1e-10, None))
        if q < best_val_q:
            best_val_q = q
            best_w = w

    print(f'Val optimal: w_har={best_w:.2f}, w_ml={1-best_w:.2f}, val_qlike={best_val_q:.4f}')

    # Apply to test
    blend_te = np.clip(best_w * har_te + (1-best_w) * ml_te, 1e-10, None)
    q_blend = qlike(y_te_raw, blend_te)

    # Oracle (best w on test)
    best_test_q = 999
    best_test_w = 0
    for w in np.arange(0.0, 1.001, 0.01):
        blend = w * har_te + (1-w) * ml_te
        q = qlike(y_te_raw, np.clip(blend, 1e-10, None))
        if q < best_test_q:
            best_test_q = q
            best_test_w = w

    print(f'\nTest results:')
    print(f'  HAR-J only:        {q_har:.4f}')
    print(f'  {ml_name} only:    {q_ml:.4f}')
    print(f'  Hybrid (val w):    {q_blend:.4f} (w_har={best_w:.2f})')
    print(f'  Oracle (test w):   {best_test_q:.4f} (w_har={best_test_w:.2f})')

    beats_ml = 'YES' if q_blend < q_ml else 'NO'
    beats_har = 'YES' if q_blend < q_har else 'NO'
    print(f'  Beats {ml_name}? {beats_ml}')
    print(f'  Beats HAR-J? {beats_har}')

    # Save
    pred_df = te[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_te_raw
    pred_df['rv_pred'] = blend_te
    pred_df.to_parquet(f'data/predictions/test_2019/hybrid_h{h}.parquet', index=False)

    with open(f'models/hybrid/model_h{h}.pkl', 'wb') as f:
        pickle.dump({
            'type': 'two_component_blend',
            'har_variant': 'HAR-J',
            'har_model': har_model,
            'har_features': harj_features,
            'ml_type': ml_name,
            'w_har': best_w,
            'w_ml': 1-best_w,
            'val_qlike': best_val_q,
            'test_qlike': q_blend,
            'oracle_qlike': best_test_q,
            'har_coefs': dict(zip(harj_features, har_model.coef_))
        }, f)

# FINAL
print('\n' + '='*60)
print('FINAL TABLE')
print('='*60)
print(f'{"Model":<20} {"H=1":>8} {"H=5":>8} {"H=22":>8}')
print(f'{"Hybrid(HAR-J+ML)":<20}', end='')
for h in [1, 5, 22]:
    with open(f'models/hybrid/model_h{h}.pkl', 'rb') as f:
        info = pickle.load(f)
    print(f' {info["test_qlike"]:>7.4f}', end='')
print()
print(f'{"XGBoost":<20} {"0.278":>8} {"0.396":>8} {"0.470":>8}')
print(f'{"LightGBM":<20} {"0.289":>8} {"0.381":>8} {"0.455":>8}')
print(f'{"HAR-J":<20} {"0.305":>8} {"0.425":>8} {"0.468":>8}')
print(f'{"GARCH":<20} {"1.802":>8} {"1.051":>8} {"0.732":>8}')
print('DONE!')
