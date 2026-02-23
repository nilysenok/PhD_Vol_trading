#!/usr/bin/env python3
"""Train LightGBM with WIDE Optuna grid and parallelism.

Usage:
    python scripts/models/train_lightgbm.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
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


def optimize_lightgbm(h, X_train, y_train, X_val, y_val, n_trials=150):
    """Optuna optimization for LightGBM with WIDE grid."""

    y_train_log = np.log(y_train + 1e-10)
    y_val_log = np.log(y_val + 1e-10)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.1),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': 4,
            'force_col_wise': True
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        pred_log = model.predict(X_val)
        pred = np.exp(pred_log)
        return qlike(y_val, pred)

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    return study.best_params, study.best_value


def train_lightgbm_horizon(h, train_df, val_df, test_df, feature_cols, n_trials=150):
    """Train LightGBM for one horizon."""

    target = f'rv_target_h{h}'

    # Prepare data
    train = train_df.dropna(subset=[target])
    val = val_df.dropna(subset=[target])
    test = test_df.dropna(subset=[target])

    X_train = train[feature_cols].values
    y_train = train[target].values
    X_val = val[feature_cols].values
    y_val = val[target].values
    X_test = test[feature_cols].values
    y_test = test[target].values

    print(f'\n=== LightGBM H={h} ===')
    print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # Optuna optimization
    best_params, best_val_qlike = optimize_lightgbm(h, X_train, y_train, X_val, y_val, n_trials)
    print(f'Best val QLIKE: {best_val_qlike:.4f}')

    # Train final model with best params
    best_params['verbosity'] = -1
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    y_train_log = np.log(y_train + 1e-10)
    y_val_log = np.log(y_val + 1e-10)

    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Predict
    y_pred = np.exp(model.predict(X_test))
    y_pred = np.clip(y_pred, 1e-10, None)

    test_qlike = qlike(y_test, y_pred)
    print(f'Test QLIKE: {test_qlike:.4f}')

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Predictions
    pred_df = test[['date', 'ticker']].copy()
    pred_df['rv_actual'] = y_test
    pred_df['rv_pred'] = y_pred

    return {
        'h': h,
        'qlike': test_qlike,
        'best_params': best_params,
        'model': model,
        'importance': importance,
        'predictions': pred_df
    }


if __name__ == '__main__':
    train = pd.read_parquet('data/prepared/train.parquet')
    val = pd.read_parquet('data/prepared/val.parquet')
    test = pd.read_parquet('data/prepared/test.parquet')

    with open('data/prepared/config.json') as f:
        config = json.load(f)
    feature_cols = config['feature_cols']

    # Sequential by h (Optuna uses parallelism internally)
    results = []
    for h in [1, 5, 22]:
        r = train_lightgbm_horizon(h, train, val, test, feature_cols, n_trials=150)
        results.append(r)

    # Save
    Path('models/lightgbm').mkdir(parents=True, exist_ok=True)
    Path('data/predictions/test_2019').mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('LightGBM Results:')
    print('='*60)

    for r in results:
        h = r['h']
        print(f"H={h}: QLIKE = {r['qlike']:.4f}")
        print(f"   Top-5: {r['importance'].head(5)['feature'].tolist()}")

        # Save model
        r['model'].booster_.save_model(f'models/lightgbm/model_h{h}.txt')

        # Save params
        with open(f'models/lightgbm/params_h{h}.json', 'w') as f:
            # Convert numpy types to Python types for JSON
            params_json = {k: (int(v) if isinstance(v, np.integer) else
                              float(v) if isinstance(v, np.floating) else v)
                          for k, v in r['best_params'].items()}
            json.dump(params_json, f, indent=2)

        # Save importance
        r['importance'].to_csv(f'models/lightgbm/importance_h{h}.csv', index=False)

        # Save predictions
        r['predictions'].to_parquet(f'data/predictions/test_2019/lightgbm_h{h}.parquet')

    print('\nLightGBM training complete!')
