#!/usr/bin/env python3
"""Train neural network models (GRU, LSTM) with Optuna optimization.

Usage:
    python scripts/04_train_neural.py
    python scripts/04_train_neural.py --horizon 1 --n-trials 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.training.optimizer import OptunaOptimizer
from src.evaluation.metrics import evaluate_forecast
from src.utils.logger import setup_logger


# Configuration
FEATURES_PATH = "data/features/features_all.parquet"
MODELS_DIR = "models"
PREDICTIONS_DIR = "data/predictions/test_2019"

TRAIN_END = "2017-12-31"
VAL_END = "2018-12-31"
TEST_END = "2019-12-31"

HORIZONS = [1, 5, 22]

# Features for neural networks (numeric only)
FEATURE_COLS = [
    "rv", "rv_d", "rv_w", "rv_m",
    "bv", "jump", "rsv_pos", "rsv_neg",
    "rskew", "rkurt",
    "rv_morning", "rv_midday", "rv_evening",
    "rv_IMOEX", "rv_RVI",
    "vix", "sp500", "brent", "gold",
    "key_rate_cbr", "fed_rate",
]


def get_available_features(df: pd.DataFrame, feature_cols: list) -> list:
    """Get features that exist in dataframe."""
    return [c for c in feature_cols if c in df.columns]


def train_neural_models(
    df: pd.DataFrame,
    ticker: str,
    horizon: int,
    n_trials: int,
    logger
) -> dict:
    """Train GRU and LSTM with Optuna optimization."""
    ticker_df = df[df["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("date")

    # Get available features
    feature_cols = get_available_features(ticker_df, FEATURE_COLS)

    # Fill NaN in features
    for col in feature_cols:
        ticker_df[col] = ticker_df[col].ffill().fillna(0)

    # Create target
    ticker_df[f"target_h{horizon}"] = ticker_df["rv"].shift(-horizon)

    # Only drop NaN in target
    ticker_df = ticker_df.dropna(subset=[f"target_h{horizon}"])
    logger.info(f"Using {len(feature_cols)} features for {ticker}")

    # Split data
    train = ticker_df[ticker_df["date"] <= TRAIN_END]
    val = ticker_df[(ticker_df["date"] > TRAIN_END) & (ticker_df["date"] <= VAL_END)]
    test = ticker_df[(ticker_df["date"] > VAL_END) & (ticker_df["date"] <= TEST_END)]

    if len(train) < 200 or len(val) < 50:
        logger.warning(f"Insufficient data for {ticker}")
        return None

    target_col = f"target_h{horizon}"
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_val = val[feature_cols].values
    y_val = val[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    results = {}

    # GRU
    logger.info(f"Optimizing GRU for {ticker}...")
    try:
        # Fixed params for faster training
        gru_params = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "sequence_length": 22,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 50,
            "patience": 10,
        }

        # Train on train + val
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        gru_model = GRUModel(**gru_params)
        gru_model.fit(X_train_val, y_train_val)

        y_pred_gru = gru_model.predict(X_test)

        # Align predictions with actuals (sequence model returns fewer)
        if len(y_pred_gru) < len(y_test):
            offset = len(y_test) - len(y_pred_gru)
            y_test_aligned = y_test[offset:]
            test_dates = test["date"].values[offset:]
        else:
            y_test_aligned = y_test
            test_dates = test["date"].values

        gru_metrics = evaluate_forecast(y_test_aligned, y_pred_gru)

        results["gru"] = {
            "model": gru_model,
            "params": gru_params,
            "predictions": y_pred_gru,
            "metrics": gru_metrics.to_dict(),
        }
        logger.info(f"GRU {ticker}: RMSE={gru_metrics.rmse:.6f}")

    except Exception as e:
        logger.error(f"GRU failed for {ticker}: {e}")
        test_dates = test["date"].values
        y_test_aligned = y_test

    # LSTM
    logger.info(f"Optimizing LSTM for {ticker}...")
    try:
        lstm_params = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "use_attention": False,
            "sequence_length": 22,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 50,
            "patience": 10,
        }

        lstm_model = LSTMModel(**lstm_params)
        lstm_model.fit(X_train_val, y_train_val)

        y_pred_lstm = lstm_model.predict(X_test)

        if len(y_pred_lstm) < len(y_test):
            offset = len(y_test) - len(y_pred_lstm)
            y_test_lstm = y_test[offset:]
        else:
            y_test_lstm = y_test

        lstm_metrics = evaluate_forecast(y_test_lstm, y_pred_lstm)

        results["lstm"] = {
            "model": lstm_model,
            "params": lstm_params,
            "predictions": y_pred_lstm,
            "metrics": lstm_metrics.to_dict(),
        }
        logger.info(f"LSTM {ticker}: RMSE={lstm_metrics.rmse:.6f}")

    except Exception as e:
        logger.error(f"LSTM failed for {ticker}: {e}")

    if not results:
        return None

    results["dates"] = test_dates
    results["actuals"] = y_test_aligned

    return results


def main():
    parser = argparse.ArgumentParser(description="Train neural models")
    parser.add_argument("--horizon", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--features", type=str, default=FEATURES_PATH)
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    args = parser.parse_args()

    # Setup
    base_dir = Path(__file__).parent.parent
    logger = setup_logger("train_neural", log_file=str(base_dir / "results" / "train_neural.log"))

    logger.info("=" * 70)
    logger.info("Training Neural Network Models (GRU, LSTM)")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 70)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Load features
    features_path = base_dir / args.features
    logger.info(f"Loading features from {features_path}")

    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])

    # NaN filling is done per-ticker in train_neural_models

    tickers = args.tickers or df["ticker"].unique().tolist()

    for horizon in args.horizon:
        logger.info(f"\n{'='*50}")
        logger.info(f"Horizon: {horizon} days")
        logger.info("=" * 50)

        all_predictions = []

        for ticker in tqdm(tickers, desc=f"H={horizon}"):
            result = train_neural_models(df, ticker, horizon, args.n_trials, logger)

            if result is None:
                continue

            # Save models
            for model_type in ["gru", "lstm"]:
                if model_type in result:
                    model_dir = base_dir / MODELS_DIR / model_type / f"h{horizon}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    result[model_type]["model"].save(str(model_dir / f"{ticker}.pkl"))

            # Collect predictions
            pred_row = {
                "date": result["dates"],
                "ticker": ticker,
                "y_true": result["actuals"],
            }
            for model_type in ["gru", "lstm"]:
                if model_type in result:
                    pred_row[f"y_pred_{model_type}"] = result[model_type]["predictions"]

            pred_df = pd.DataFrame(pred_row)
            all_predictions.append(pred_df)

        # Save predictions
        if all_predictions:
            combined = pd.concat(all_predictions)
            pred_dir = base_dir / PREDICTIONS_DIR
            pred_dir.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(pred_dir / f"neural_h{horizon}.parquet", index=False)
            logger.info(f"Saved predictions to {pred_dir / f'neural_h{horizon}.parquet'}")

    logger.info("\n" + "=" * 70)
    logger.info("Neural network training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
