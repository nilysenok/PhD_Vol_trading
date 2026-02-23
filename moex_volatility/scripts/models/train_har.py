#!/usr/bin/env python3
"""Train HAR model with internal parallelism.

HAR: log(RV_{t+h}) = α + β_d·log(RV_d) + β_w·log(RV_w) + β_m·log(RV_m)

Usage:
    python scripts/models/train_har.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import pickle
import json


def qlike(y_true, y_pred):
    """QLIKE loss function."""
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def train_har_horizon(h, train_df, val_df, test_df, har_features, target_col):
    """Train HAR for one horizon."""

    # Remove NaN
    train = train_df.dropna(subset=[target_col])
    test = test_df.dropna(subset=[target_col])

    # Log transform
    X_train = np.log(train[har_features].values + 1e-10)
    y_train = np.log(train[target_col].values + 1e-10)
    X_test = np.log(test[har_features].values + 1e-10)
    y_test = test[target_col].values

    # Fit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_pred = np.clip(y_pred, 1e-10, None)

    # Metrics
    qlike_val = qlike(y_test, y_pred)

    # Coefficients
    coefs = dict(zip(har_features, model.coef_))
    coefs['intercept'] = float(model.intercept_)

    # Convert numpy floats to Python floats for JSON serialization
    coefs = {k: float(v) for k, v in coefs.items()}

    # Save predictions
    pred_df = test[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_test
    pred_df['rv_pred'] = y_pred

    return {
        'h': h,
        'qlike': qlike_val,
        'coefs': coefs,
        'model': model,
        'predictions': pred_df
    }


if __name__ == '__main__':
    # Load prepared data
    train = pd.read_parquet('data/prepared/train.parquet')
    val = pd.read_parquet('data/prepared/val.parquet')
    test = pd.read_parquet('data/prepared/test.parquet')

    with open('data/prepared/config.json') as f:
        config = json.load(f)
    har_features = config['har_features']

    # Parallel training for h=1,5,22
    results = Parallel(n_jobs=3, verbose=10)(
        delayed(train_har_horizon)(
            h, train, val, test, har_features, f'rv_target_h{h}'
        ) for h in [1, 5, 22]
    )

    # Save
    Path('models/har').mkdir(parents=True, exist_ok=True)
    Path('data/predictions/test_2019').mkdir(parents=True, exist_ok=True)

    print('\nHAR Model Results:')
    print('=' * 50)

    all_coefs = {}
    for r in sorted(results, key=lambda x: x['h']):
        h = r['h']
        print(f"H={h}: QLIKE = {r['qlike']:.4f}")
        print(f"   Coefs: {r['coefs']}")

        # Save model
        with open(f'models/har/model_h{h}.pkl', 'wb') as f:
            pickle.dump(r['model'], f)

        # Save predictions
        r['predictions'].to_parquet(f'data/predictions/test_2019/har_h{h}.parquet')

        all_coefs[f'h{h}'] = r['coefs']

    # Save all coefficients
    with open('models/har/coefficients.json', 'w') as f:
        json.dump(all_coefs, f, indent=2)

    print('\nFiles saved!')
