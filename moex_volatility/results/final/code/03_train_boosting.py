#!/usr/bin/env python3
"""Train LightGBM and XGBoost models for volatility forecasting.

Usage:
    python scripts/03_train_boosting.py
    python scripts/03_train_boosting.py --n-trials 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.models.boosting import LightGBMModel, XGBoostModel, SklearnBoostingModel, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE
from src.training.optimizer import BoostingOptimizer, qlike_metric
from src.utils.logger import setup_logger

# Check available boosting libraries
print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")


# Configuration
DATA_PATH = Path('data/features/features_all.parquet')
MODELS_PATH = Path('models')
PREDICTIONS_PATH = Path('data/predictions/test_2019')

# Data split dates
TRAIN_END = '2017-12-31'
VAL_END = '2018-12-31'
TEST_END = '2019-12-31'

# Horizons
HORIZONS = [1, 5, 22]

# Reference results (from HAR/GARCH)
HAR_QLIKE = {1: -7.527, 5: -7.414, 22: -7.340}
GARCH_QLIKE = {1: -7.444, 5: -6.145, 22: -4.248}


def prepare_features(df: pd.DataFrame, horizon: int) -> tuple:
    """Prepare features and create target variable.

    Args:
        df: Full feature DataFrame.
        horizon: Forecast horizon.

    Returns:
        Tuple of (df_with_target, feature_cols, target_col).
    """
    df = df.copy()

    # Create target: RV shifted forward by horizon days
    # For each ticker, shift RV forward
    df['rv_target'] = df.groupby('ticker')['rv'].shift(-horizon)

    # Exclude columns that shouldn't be features
    exclude_cols = ['date', 'ticker', 'rv', 'rv_target']

    # Also exclude any existing target columns
    exclude_cols += [c for c in df.columns if c.startswith('rv_target')]

    # Exclude non-numeric columns (object, bool types)
    exclude_cols += [c for c in df.columns if df[c].dtype == 'object']
    exclude_cols += [c for c in df.columns if c.startswith('low_bars')]
    exclude_cols += [c for c in df.columns if c.startswith('source')]

    # Get numeric columns only
    feature_cols = [c for c in df.columns
                    if c not in exclude_cols
                    and np.issubdtype(df[c].dtype, np.number)]

    return df, feature_cols, 'rv_target'


def main():
    parser = argparse.ArgumentParser(description="Train boosting models")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per model")
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    args = parser.parse_args()

    # Setup
    base_dir = Path(__file__).parent.parent
    logger = setup_logger("train_boosting", log_file=str(base_dir / "results" / "train_boosting.log"))

    logger.info("=" * 70)
    logger.info("Training Boosting Models (LightGBM, XGBoost)")
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Optuna trials: {args.n_trials}")
    logger.info("=" * 70)

    # Load features
    features_path = base_dir / DATA_PATH
    logger.info(f"Loading features from {features_path}")

    df = pd.read_parquet(features_path)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")

    # Ensure directories exist
    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)

    results = []

    for horizon in args.horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"HORIZON: {horizon} day(s)")
        logger.info("=" * 60)

        # Prepare data
        df_h, feature_cols, target_col = prepare_features(df, horizon)
        logger.info(f"Features: {len(feature_cols)}")

        # Split by date
        train_mask = df_h['date'] <= TRAIN_END
        val_mask = (df_h['date'] > TRAIN_END) & (df_h['date'] <= VAL_END)
        test_mask = (df_h['date'] > VAL_END) & (df_h['date'] <= TEST_END)

        # Drop NaN targets
        train_df = df_h[train_mask].dropna(subset=[target_col])
        val_df = df_h[val_mask].dropna(subset=[target_col])
        test_df = df_h[test_mask].dropna(subset=[target_col])

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Prepare arrays
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        # Prepare train+val for final model
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        lgb_qlike = float('inf')
        xgb_qlike = float('inf')
        use_lgb = LIGHTGBM_AVAILABLE
        use_xgb = XGBOOST_AVAILABLE

        # ===== Model 1: LightGBM or Sklearn HistGradientBoosting =====
        if use_lgb:
            logger.info(f"\n--- LightGBM (Optuna {args.n_trials} trials) ---")
            try:
                lgb_optimizer = BoostingOptimizer(
                    model_type='lightgbm',
                    n_trials=args.n_trials,
                    metric='qlike',
                    verbose=True
                )
                lgb_params = lgb_optimizer.optimize(X_train, y_train, X_val, y_val)
                logger.info(f"Best LightGBM params: {lgb_params}")

                lgb_model = LightGBMModel(horizon=horizon, params=lgb_params)
                lgb_model.fit(X_train_val, y_train_val, feature_names=feature_cols)
                lgb_pred = lgb_model.predict(X_test)
                lgb_qlike = qlike_metric(y_test, lgb_pred)
            except Exception as e:
                logger.warning(f"LightGBM failed: {e}")
                use_lgb = False

        if not use_lgb:
            logger.info(f"\n--- HistGradientBoosting (sklearn fallback) ---")
            # Use default params for sklearn
            lgb_params = {'max_iter': 500, 'max_depth': 8, 'learning_rate': 0.05}
            lgb_model = SklearnBoostingModel(horizon=horizon, params=lgb_params, model_type='hist')
            lgb_model.fit(X_train_val, y_train_val, X_val=X_val, y_val=y_val, feature_names=feature_cols)
            lgb_pred = lgb_model.predict(X_test)
            lgb_qlike = qlike_metric(y_test, lgb_pred)

        logger.info(f"Model 1 Test QLIKE: {lgb_qlike:.4f}")

        # Save model
        lgb_path = MODELS_PATH / 'lightgbm' / f'h{horizon}'
        lgb_path.mkdir(parents=True, exist_ok=True)
        lgb_model.save(str(lgb_path / 'model.pkl'))
        with open(lgb_path / 'best_params.json', 'w') as f:
            json.dump(lgb_params, f, indent=2, default=str)

        # Feature importance
        importance_df = lgb_model.get_feature_importance()
        importance_df.to_csv(lgb_path / 'feature_importance.csv', index=False)
        logger.info(f"Top-5 features: {importance_df.head(5)['feature'].tolist()}")

        # Save predictions
        lgb_pred_df = test_df[['date', 'ticker']].copy()
        lgb_pred_df['y_true'] = y_test
        lgb_pred_df['y_pred_lgb'] = lgb_pred
        lgb_pred_df.to_parquet(PREDICTIONS_PATH / f'lightgbm_h{horizon}.parquet', index=False)

        # ===== Model 2: XGBoost or Sklearn GradientBoosting =====
        if use_xgb:
            logger.info(f"\n--- XGBoost (Optuna {args.n_trials} trials) ---")
            try:
                xgb_optimizer = BoostingOptimizer(
                    model_type='xgboost',
                    n_trials=args.n_trials,
                    metric='qlike',
                    verbose=True
                )
                xgb_params = xgb_optimizer.optimize(X_train, y_train, X_val, y_val)
                logger.info(f"Best XGBoost params: {xgb_params}")

                xgb_model = XGBoostModel(horizon=horizon, params=xgb_params)
                xgb_model.fit(X_train_val, y_train_val, feature_names=feature_cols)
                xgb_pred = xgb_model.predict(X_test)
                xgb_qlike = qlike_metric(y_test, xgb_pred)
            except Exception as e:
                logger.warning(f"XGBoost failed: {e}")
                use_xgb = False

        if not use_xgb:
            logger.info(f"\n--- GradientBoosting (sklearn fallback) ---")
            # Use default params for sklearn
            xgb_params = {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05}
            xgb_model = SklearnBoostingModel(horizon=horizon, params=xgb_params, model_type='gbr')
            xgb_model.fit(X_train_val, y_train_val, feature_names=feature_cols)
            xgb_pred = xgb_model.predict(X_test)
            xgb_qlike = qlike_metric(y_test, xgb_pred)

        logger.info(f"Model 2 Test QLIKE: {xgb_qlike:.4f}")

        # Save model
        xgb_path = MODELS_PATH / 'xgboost' / f'h{horizon}'
        xgb_path.mkdir(parents=True, exist_ok=True)
        xgb_model.save(str(xgb_path / 'model.pkl'))
        with open(xgb_path / 'best_params.json', 'w') as f:
            json.dump(xgb_params, f, indent=2, default=str)

        # Feature importance
        importance_df = xgb_model.get_feature_importance()
        importance_df.to_csv(xgb_path / 'feature_importance.csv', index=False)
        logger.info(f"Top-5 features: {importance_df.head(5)['feature'].tolist()}")

        # Save predictions
        xgb_pred_df = test_df[['date', 'ticker']].copy()
        xgb_pred_df['y_true'] = y_test
        xgb_pred_df['y_pred_xgb'] = xgb_pred
        xgb_pred_df.to_parquet(PREDICTIONS_PATH / f'xgboost_h{horizon}.parquet', index=False)

        results.append({
            'horizon': horizon,
            'lgb_qlike': lgb_qlike,
            'xgb_qlike': xgb_qlike,
            'har_qlike': HAR_QLIKE[horizon],
            'garch_qlike': GARCH_QLIKE[horizon]
        })

        logger.info(f"\n--- Horizon {horizon} Summary ---")
        logger.info(f"HAR QLIKE:      {HAR_QLIKE[horizon]:.4f}")
        logger.info(f"GARCH QLIKE:    {GARCH_QLIKE[horizon]:.4f}")
        logger.info(f"LightGBM QLIKE: {lgb_qlike:.4f}")
        logger.info(f"XGBoost QLIKE:  {xgb_qlike:.4f}")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: QLIKE (lower is better)")
    print("=" * 70)
    print(f"\n{'Model':<12} {'H=1':>10} {'H=5':>10} {'H=22':>10}")
    print("-" * 44)

    # HAR
    print(f"{'HAR':<12} {HAR_QLIKE[1]:>10.3f} {HAR_QLIKE[5]:>10.3f} {HAR_QLIKE[22]:>10.3f}")

    # GARCH
    print(f"{'GARCH':<12} {GARCH_QLIKE[1]:>10.3f} {GARCH_QLIKE[5]:>10.3f} {GARCH_QLIKE[22]:>10.3f}")

    # LightGBM
    lgb_scores = {r['horizon']: r['lgb_qlike'] for r in results}
    print(f"{'LightGBM':<12} {lgb_scores.get(1, float('nan')):>10.3f} "
          f"{lgb_scores.get(5, float('nan')):>10.3f} {lgb_scores.get(22, float('nan')):>10.3f}")

    # XGBoost
    xgb_scores = {r['horizon']: r['xgb_qlike'] for r in results}
    print(f"{'XGBoost':<12} {xgb_scores.get(1, float('nan')):>10.3f} "
          f"{xgb_scores.get(5, float('nan')):>10.3f} {xgb_scores.get(22, float('nan')):>10.3f}")

    print("\n" + "=" * 70)

    # Find best model per horizon
    print("\nBest model per horizon:")
    for h in args.horizons:
        models = {
            'HAR': HAR_QLIKE[h],
            'GARCH': GARCH_QLIKE[h],
            'LightGBM': lgb_scores.get(h, float('inf')),
            'XGBoost': xgb_scores.get(h, float('inf'))
        }
        best = min(models, key=models.get)
        print(f"  H={h}: {best} (QLIKE={models[best]:.4f})")

    logger.info("\n" + "=" * 70)
    logger.info("Boosting model training complete!")
    logger.info("=" * 70)

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(base_dir / 'results' / 'boosting_summary.csv', index=False)
    print(f"\nSummary saved to results/boosting_summary.csv")


if __name__ == "__main__":
    main()
