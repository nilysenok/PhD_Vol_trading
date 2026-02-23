#!/usr/bin/env python3
"""Train XGBoost with WIDE Optuna grid.

Usage:
    python scripts/models/train_xgboost.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def qlike(y_true, y_pred):
    """QLIKE loss function."""
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def optimize_xgboost(h, X_train, y_train, X_val, y_val, n_trials=100):
    """Optuna optimization for XGBoost."""

    y_train_log = np.log(y_train + 1e-10)
    y_val_log = np.log(y_val + 1e-10)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 2.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': 4
        }

        params['early_stopping_rounds'] = 50
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False
        )

        pred_log = model.predict(X_val)
        pred = np.exp(pred_log)
        return qlike(y_val, pred)

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def train_xgboost_horizon(h, train_df, val_df, test_df, feature_cols, n_trials=100):
    """Train XGBoost for one horizon."""

    target = f'rv_target_h{h}'

    train = train_df.dropna(subset=[target])
    val = val_df.dropna(subset=[target])
    test = test_df.dropna(subset=[target])

    X_train = train[feature_cols].values
    y_train = train[target].values
    X_val = val[feature_cols].values
    y_val = val[target].values
    X_test = test[feature_cols].values
    y_test = test[target].values

    print(f'\n=== XGBoost H={h} ===')
    print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # Optuna
    best_params, best_val_qlike = optimize_xgboost(h, X_train, y_train, X_val, y_val, n_trials)
    print(f'Best val QLIKE: {best_val_qlike:.4f}')

    # Final model
    best_params['verbosity'] = 0
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    y_train_log = np.log(y_train + 1e-10)
    y_val_log = np.log(y_val + 1e-10)

    best_params['early_stopping_rounds'] = 50
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False
    )

    y_pred = np.exp(model.predict(X_test))
    y_pred = np.clip(y_pred, 1e-10, None)

    test_qlike = qlike(y_test, y_pred)
    print(f'Test QLIKE: {test_qlike:.4f}')

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    pred_df = test[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_test
    pred_df['rv_pred'] = y_pred

    return {
        'h': h, 'qlike': test_qlike, 'best_params': best_params,
        'model': model, 'importance': importance, 'predictions': pred_df
    }


if __name__ == '__main__':
    train = pd.read_parquet('data/prepared/train.parquet')
    val = pd.read_parquet('data/prepared/val.parquet')
    test = pd.read_parquet('data/prepared/test.parquet')

    with open('data/prepared/config.json') as f:
        config = json.load(f)
    feature_cols = config['feature_cols']

    results = []
    for h in [1, 5, 22]:
        r = train_xgboost_horizon(h, train, val, test, feature_cols, n_trials=100)
        results.append(r)

    # Save
    Path('models/xgboost').mkdir(parents=True, exist_ok=True)
    Path('data/predictions/test_2019').mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('XGBoost Results:')
    print('='*60)

    for r in results:
        h = r['h']
        print(f"H={h}: QLIKE = {r['qlike']:.4f}")
        print(f"   Top-5: {r['importance'].head(5)['feature'].tolist()}")

        r['model'].save_model(f'models/xgboost/model_h{h}.json')

        # Convert numpy types to Python types for JSON
        params_json = {k: (int(v) if isinstance(v, np.integer) else
                          float(v) if isinstance(v, np.floating) else v)
                      for k, v in r['best_params'].items()}
        with open(f'models/xgboost/params_h{h}.json', 'w') as f:
            json.dump(params_json, f, indent=2)

        r['importance'].to_csv(f'models/xgboost/importance_h{h}.csv', index=False)
        r['predictions'].to_parquet(f'data/predictions/test_2019/xgboost_h{h}.parquet')

    print('\nXGBoost training complete!')
