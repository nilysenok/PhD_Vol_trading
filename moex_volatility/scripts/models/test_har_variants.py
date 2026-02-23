import pandas as pd, numpy as np, json
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

train = pd.read_parquet('data/prepared/train.parquet')
val = pd.read_parquet('data/prepared/val.parquet')
test = pd.read_parquet('data/prepared/test.parquet')

with open('data/prepared/config.json') as f:
    config = json.load(f)
har_base = config['har_features']

all_cols = train.columns.tolist()
rv_cols = [c for c in all_cols if any(x in c.lower() for x in ['rv', 'bv', 'jump', 'rsv', 'rskew', 'rkurt']) and 'target' not in c.lower() and 'idx_' not in c.lower()]
print(f'RV columns ({len(rv_cols)}): {sorted(rv_cols)}')

def find(patterns):
    return [c for c in all_cols if any(p in c.lower() for p in patterns) and 'target' not in c.lower() and 'idx_' not in c.lower()]

variants = {
    'HAR': har_base,
    'HAR-J': har_base + find(['jump'])[:3],
    'HAR-RS': har_base + find(['rsv_pos', 'rsv_neg'])[:4],
    'HAR-BV': har_base + find(['bv'])[:3],
    'HAR-J-RS': har_base + find(['jump'])[:2] + find(['rsv_pos', 'rsv_neg'])[:2],
    'HAR-SK': har_base + find(['rskew', 'rkurt'])[:4],
    'HAR-FULL': rv_cols,
}

# Убираем варианты без новых фичей
variants = {k:v for k,v in variants.items() if len([f for f in v if f in all_cols]) >= len(har_base)}

for name, feats in variants.items():
    valid = [f for f in feats if f in all_cols]
    print(f'{name}: {len(valid)} features -> {valid}')

results = []
for h in [1, 5, 22]:
    target = f'rv_target_h{h}'
    tr = train.dropna(subset=[target])
    te = test.dropna(subset=[target])

    y_tr = np.log(tr[target].values + 1e-10)
    y_te = te[target].values

    for name, feats in variants.items():
        valid = [f for f in feats if f in tr.columns]
        X_tr = np.nan_to_num(np.log(tr[valid].values.clip(1e-10)), 0, 0, 0)
        X_te = np.nan_to_num(np.log(te.loc[te.index.isin(te.dropna(subset=[target]).index), valid].values.clip(1e-10)), 0, 0, 0)

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        pred = np.clip(np.exp(model.predict(X_te)), 1e-10, None)
        q = qlike(y_te, pred)
        results.append({'model': name, 'h': h, 'qlike': round(q, 4), 'n_feat': len(valid)})

df = pd.DataFrame(results)
pivot = df.pivot_table(index='model', columns='h', values='qlike')
print('\n=== QLIKE по горизонтам ===')
print(pivot.round(4).to_string())
print('\nБенчмарки: H=1 XGBoost=0.278, H=5 LGB=0.381, H=22 LGB=0.455')
print('DONE!')
