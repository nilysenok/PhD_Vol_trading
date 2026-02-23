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
har_features = config['har_features']
target = 'rv_target_h1'

tr = train.dropna(subset=[target])
va = val.dropna(subset=[target])
te = test.dropna(subset=[target])

# HAR predictions
har_model = LinearRegression()
har_model.fit(np.log(tr[har_features]+1e-10), np.log(tr[target]+1e-10))
har_va = np.exp(har_model.predict(np.log(va.loc[va.index.isin(va.dropna(subset=[target]).index), har_features]+1e-10)))
har_te = np.exp(har_model.predict(np.log(te.loc[te.index.isin(te.dropna(subset=[target]).index), har_features]+1e-10)))

# LightGBM predictions
lgbm_model = lgb.Booster(model_file='models/lightgbm/model_h1.txt')
lgbm_va = np.exp(lgbm_model.predict(va.loc[va.index.isin(va.dropna(subset=[target]).index), feature_cols]))
lgbm_te = np.exp(lgbm_model.predict(te.loc[te.index.isin(te.dropna(subset=[target]).index), feature_cols]))

# XGBoost predictions
xgb_model = xgboost.Booster()
xgb_model.load_model('models/xgboost/model_h1.json')
xgb_va_data = va.loc[va.index.isin(va.dropna(subset=[target]).index), feature_cols].replace([np.inf, -np.inf], np.nan)
xgb_te_data = te.loc[te.index.isin(te.dropna(subset=[target]).index), feature_cols].replace([np.inf, -np.inf], np.nan)
xgb_va = np.exp(xgb_model.predict(xgboost.DMatrix(xgb_va_data, missing=np.nan)))
xgb_te = np.exp(xgb_model.predict(xgboost.DMatrix(xgb_te_data, missing=np.nan)))

y_va = va.dropna(subset=[target])[target].values
y_te = te.dropna(subset=[target])[target].values

# Optuna: веса трёх моделей
def objective(trial):
    w1 = trial.suggest_float('w_har', 0.0, 0.5)
    w2 = trial.suggest_float('w_lgbm', 0.0, 1.0)
    w3 = 1.0 - w1 - w2
    if w3 < 0: return 999.0
    blend = np.clip(w1*har_va + w2*lgbm_va + w3*xgb_va, 1e-10, None)
    return qlike(y_va, blend)

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=500)

w_har = study.best_params['w_har']
w_lgbm = study.best_params['w_lgbm']
w_xgb = 1.0 - w_har - w_lgbm

blend_te = np.clip(w_har*har_te + w_lgbm*lgbm_te + w_xgb*xgb_te, 1e-10, None)
q = qlike(y_te, blend_te)

print(f'Weights: HAR={w_har:.3f}, LGB={w_lgbm:.3f}, XGB={w_xgb:.3f}')
print(f'H=1 results:')
print(f'  HAR:      {qlike(y_te, np.clip(har_te,1e-10,None)):.4f}')
print(f'  LightGBM: {qlike(y_te, np.clip(lgbm_te,1e-10,None)):.4f}')
print(f'  XGBoost:  {qlike(y_te, np.clip(xgb_te,1e-10,None)):.4f}')
print(f'  Blend:    {q:.4f}')

Path('models/hybrid').mkdir(parents=True, exist_ok=True)
with open('models/hybrid/model_h1.pkl', 'wb') as f:
    pickle.dump({'type':'blend', 'w_har':w_har, 'w_lgbm':w_lgbm, 'w_xgb':w_xgb, 'qlike':q}, f)

pred_df = te[['date','ticker']].copy()
pred_df['rv_actual'] = y_te
pred_df['rv_pred'] = blend_te
pred_df.to_parquet('data/predictions/test_2019/hybrid_h1.parquet', index=False)
print('DONE!')
