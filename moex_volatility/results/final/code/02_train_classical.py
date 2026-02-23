#!/usr/bin/env python3
"""Train classical models (HAR-RV-J, GARCH-GJR).

Usage:
    python scripts/02_train_classical.py
    python scripts/02_train_classical.py --horizon 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.models.har import HARModel
from src.models.garch import GARCHModel
from src.evaluation.metrics import evaluate_forecast
from src.utils.logger import setup_logger


# Configuration
FEATURES_PATH = "data/features/features_all.parquet"
MODELS_DIR = "models"
PREDICTIONS_DIR = "data/predictions/test_2019"

# Data split dates
TRAIN_END = pd.Timestamp("2017-12-31")
VAL_END = pd.Timestamp("2018-12-31")
TEST_END = pd.Timestamp("2019-12-31")

# Horizons to forecast
HORIZONS = [1, 5, 22]

# HAR grid search
HAR_GRID = {
    "use_log": [True, False],
    "add_constant": [True],
}

# GARCH grid search
GARCH_GRID = {
    "p": [1, 2],
    "q": [1, 2],
    "vol": ["GARCH", "GJRGARCH", "EGARCH"],
    "dist": ["normal", "t"],
    "mean": ["Constant", "Zero"],
}


def prepare_har_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Prepare features for HAR model."""
    ticker_df = df[df["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("date")
    ticker_df["date"] = pd.to_datetime(ticker_df["date"])

    # Ensure we have the required columns
    required = ["rv", "rv_d", "rv_w", "rv_m", "jump"]
    for col in required:
        if col not in ticker_df.columns:
            if col == "rv_d":
                ticker_df["rv_d"] = ticker_df["rv"].shift(1)
            elif col == "rv_w":
                ticker_df["rv_w"] = ticker_df["rv"].rolling(5).mean().shift(1)
            elif col == "rv_m":
                ticker_df["rv_m"] = ticker_df["rv"].rolling(22).mean().shift(1)
            elif col == "jump":
                ticker_df["jump"] = 0

    # Calculate returns from close prices for GARCH
    if "close" in ticker_df.columns:
        ticker_df["returns"] = np.log(ticker_df["close"] / ticker_df["close"].shift(1))
    else:
        # Approximate returns from RV
        ticker_df["returns"] = np.sign(np.random.randn(len(ticker_df))) * np.sqrt(ticker_df["rv"])

    return ticker_df


def train_har_models(
    df: pd.DataFrame,
    ticker: str,
    horizon: int,
    logger
) -> dict:
    """Train HAR models with grid search."""
    ticker_df = prepare_har_features(df, ticker)

    # Create target
    ticker_df[f"target_h{horizon}"] = ticker_df["rv"].shift(-horizon)

    # Only drop NaN for the specific columns we need
    required_cols = ["rv_d", "rv_w", "rv_m", f"target_h{horizon}"]
    ticker_df = ticker_df.dropna(subset=required_cols)

    # Split data
    train = ticker_df[ticker_df["date"] <= TRAIN_END]
    val = ticker_df[(ticker_df["date"] > TRAIN_END) & (ticker_df["date"] <= VAL_END)]
    test = ticker_df[(ticker_df["date"] > VAL_END) & (ticker_df["date"] <= TEST_END)]

    if len(train) < 100 or len(val) < 50:
        return None

    # Feature columns
    feature_cols = ["rv_d", "rv_w", "rv_m"]
    target_col = f"target_h{horizon}"

    # Grid search on validation
    best_score = float("inf")
    best_params = {}
    best_model = None

    for use_log in HAR_GRID["use_log"]:
        for add_constant in HAR_GRID["add_constant"]:
            try:
                model = HARModel(use_log=use_log, add_constant=add_constant)
                model.fit(train[feature_cols], train[target_col])
                y_pred = model.predict(val[feature_cols])
                score = np.mean((val[target_col].values - y_pred) ** 2)

                if score < best_score:
                    best_score = score
                    best_params = {"use_log": use_log, "add_constant": add_constant}
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        return None

    # Retrain on train + val
    train_val = pd.concat([train, val])
    final_model = HARModel(**best_params)
    final_model.fit(train_val[feature_cols], train_val[target_col])

    # Predict on test
    y_pred = final_model.predict(test[feature_cols])
    y_true = test[target_col].values

    metrics = evaluate_forecast(y_true, y_pred)

    return {
        "model": final_model,
        "params": best_params,
        "predictions": y_pred,
        "actuals": y_true,
        "dates": test["date"].values,
        "metrics": metrics.to_dict(),
    }


def train_garch_models(
    df: pd.DataFrame,
    ticker: str,
    horizon: int,
    logger
) -> dict:
    """Train GARCH-GJR models with grid search."""
    ticker_df = prepare_har_features(df, ticker)

    # Need returns for GARCH
    ticker_df = ticker_df.dropna(subset=["returns", "rv"])

    # Split data
    train = ticker_df[ticker_df["date"] <= TRAIN_END]
    val = ticker_df[(ticker_df["date"] > TRAIN_END) & (ticker_df["date"] <= VAL_END)]
    test = ticker_df[(ticker_df["date"] > VAL_END) & (ticker_df["date"] <= TEST_END)]

    if len(train) < 200 or len(val) < 50 or len(test) < 50:
        return None

    # Get returns
    returns_train = train["returns"].values * 100  # Scale for numerical stability
    returns_val = val["returns"].values * 100
    returns_train_val = np.concatenate([returns_train, returns_val])

    # RV targets for evaluation
    rv_test = test["rv"].values

    # Grid search on validation
    best_aic = float("inf")
    best_params = {}
    best_model = None

    for p in GARCH_GRID["p"]:
        for q in GARCH_GRID["q"]:
            for vol in GARCH_GRID["vol"]:
                for dist in GARCH_GRID["dist"]:
                    for mean in GARCH_GRID["mean"]:
                        try:
                            # Use GJR parameter for asymmetric models
                            # GJR-GARCH uses vol='GARCH' with o > 0
                            o = 1 if vol == "GJRGARCH" else 0
                            vol_type = "GARCH" if vol == "GJRGARCH" else vol

                            model = GARCHModel(
                                p=p, q=q, o=o,
                                vol=vol_type,
                                dist=dist,
                                mean=mean,
                                rescale=False  # Already scaled
                            )
                            model.fit(None, returns_train)

                            aic = model.get_aic()
                            if aic < best_aic:
                                best_aic = aic
                                best_params = {
                                    "p": p, "q": q, "o": o,
                                    "vol": vol_type, "dist": dist, "mean": mean,
                                    "original_vol": vol  # Keep original for naming
                                }
                                best_model = model
                        except Exception:
                            continue

    if best_model is None:
        return None

    # Retrain on train + val
    try:
        final_model = GARCHModel(**best_params, rescale=False)
        final_model.fit(None, returns_train_val)
    except Exception as e:
        logger.warning(f"GARCH retrain failed for {ticker}: {e}")
        return None

    # Multi-step ahead forecast
    # GARCH forecasts variance, we need to convert to RV proxy
    try:
        # Get forecast for each test day using rolling window
        predictions = []
        returns_history = returns_train_val.tolist()

        for i in range(len(test)):
            # Refit on current history (or use rolling forecast)
            if i > 0:
                returns_history.append(test["returns"].values[i-1] * 100)

            # Forecast h steps ahead
            # Use simulation for multi-step ahead (analytic not available for EGARCH)
            method = "analytic" if horizon == 1 else "simulation"
            try:
                forecast_var = final_model.predict(None, horizon=horizon, method=method)
            except Exception:
                # Fallback to simulation if analytic fails
                forecast_var = final_model.predict(None, horizon=horizon, method="simulation")

            # Sum of variance over horizon approximates cumulative RV
            if horizon == 1:
                pred_rv = forecast_var[0] / 10000  # Rescale back
            else:
                pred_rv = np.sum(forecast_var[:horizon]) / 10000

            predictions.append(pred_rv)

        predictions = np.array(predictions)

        # Shift target for h-step ahead
        rv_target = test["rv"].shift(-horizon + 1).values if horizon > 1 else rv_test
        rv_target = rv_target[:len(predictions)]

        # Remove NaN at the end
        valid_mask = ~np.isnan(rv_target)
        predictions = predictions[valid_mask]
        rv_target = rv_target[valid_mask]
        test_dates = test["date"].values[valid_mask]

        if len(predictions) < 10:
            return None

        metrics = evaluate_forecast(rv_target, predictions)

        return {
            "model": final_model,
            "params": best_params,
            "predictions": predictions,
            "actuals": rv_target,
            "dates": test_dates,
            "metrics": metrics.to_dict(),
            "aic": best_aic,
        }
    except Exception as e:
        logger.warning(f"GARCH forecast failed for {ticker}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train classical models")
    parser.add_argument("--horizon", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--features", type=str, default=FEATURES_PATH)
    parser.add_argument("--skip-garch", action="store_true", help="Skip GARCH training")
    args = parser.parse_args()

    # Setup
    base_dir = Path(__file__).parent.parent
    logger = setup_logger("train_classical", log_file=str(base_dir / "results" / "train_classical.log"))

    logger.info("=" * 70)
    logger.info("Training Classical Models (HAR, GARCH-GJR)")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 70)

    # Load features
    features_path = base_dir / args.features
    logger.info(f"Loading features from {features_path}")

    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")

    tickers = df["ticker"].unique().tolist()

    for horizon in args.horizon:
        logger.info(f"\n{'='*50}")
        logger.info(f"Horizon: {horizon} days")
        logger.info("=" * 50)

        har_results = {}
        garch_results = {}

        # Train HAR
        logger.info("\n--- Training HAR models ---")
        for ticker in tqdm(tickers, desc=f"HAR H={horizon}"):
            result = train_har_models(df, ticker, horizon, logger)
            if result:
                har_results[ticker] = result
                # Save model
                model_dir = base_dir / MODELS_DIR / "har" / f"h{horizon}"
                model_dir.mkdir(parents=True, exist_ok=True)
                result["model"].save(str(model_dir / f"{ticker}.pkl"))

        logger.info(f"HAR: Trained {len(har_results)}/{len(tickers)} tickers")

        # Train GARCH
        if not args.skip_garch:
            logger.info("\n--- Training GARCH-GJR models ---")
            for ticker in tqdm(tickers, desc=f"GARCH H={horizon}"):
                result = train_garch_models(df, ticker, horizon, logger)
                if result:
                    garch_results[ticker] = result
                    # Save model
                    model_dir = base_dir / MODELS_DIR / "garch" / f"h{horizon}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    with open(model_dir / f"{ticker}.pkl", "wb") as f:
                        pickle.dump({
                            "params": result["params"],
                            "aic": result.get("aic"),
                        }, f)

            logger.info(f"GARCH: Trained {len(garch_results)}/{len(tickers)} tickers")

        # Save combined predictions
        pred_dir = base_dir / PREDICTIONS_DIR
        pred_dir.mkdir(parents=True, exist_ok=True)

        # HAR predictions
        har_preds = []
        for ticker, res in har_results.items():
            pred_df = pd.DataFrame({
                "date": res["dates"],
                "ticker": ticker,
                "y_true": res["actuals"],
                "y_pred_har": res["predictions"],
            })
            har_preds.append(pred_df)

        if har_preds:
            har_combined = pd.concat(har_preds)
            har_combined.to_parquet(pred_dir / f"har_h{horizon}.parquet", index=False)
            logger.info(f"Saved HAR predictions: {pred_dir / f'har_h{horizon}.parquet'}")

        # GARCH predictions
        if garch_results:
            garch_preds = []
            for ticker, res in garch_results.items():
                pred_df = pd.DataFrame({
                    "date": res["dates"],
                    "ticker": ticker,
                    "y_true": res["actuals"],
                    "y_pred_garch": res["predictions"],
                })
                garch_preds.append(pred_df)

            garch_combined = pd.concat(garch_preds)
            garch_combined.to_parquet(pred_dir / f"garch_h{horizon}.parquet", index=False)
            logger.info(f"Saved GARCH predictions: {pred_dir / f'garch_h{horizon}.parquet'}")

        # Merge and save classical predictions (HAR + GARCH)
        if har_preds and garch_results:
            # Merge HAR and GARCH on date and ticker
            merged = har_combined.merge(
                garch_combined[["date", "ticker", "y_pred_garch"]],
                on=["date", "ticker"],
                how="left"
            )
            merged.to_parquet(pred_dir / f"classical_h{horizon}.parquet", index=False)
        elif har_preds:
            har_combined.to_parquet(pred_dir / f"classical_h{horizon}.parquet", index=False)

        # Summary
        logger.info(f"\n--- Horizon {horizon} Summary ---")
        logger.info("\nHAR Results (top 5):")
        for ticker, res in list(har_results.items())[:5]:
            m = res["metrics"]
            logger.info(f"  {ticker}: RMSE={m['RMSE']:.6f}, R²={m['R2']:.4f}, QLIKE={m['QLIKE']:.4f}")

        if garch_results:
            logger.info("\nGARCH Results (top 5):")
            for ticker, res in list(garch_results.items())[:5]:
                m = res["metrics"]
                logger.info(f"  {ticker}: RMSE={m['RMSE']:.6f}, R²={m['R2']:.4f}, AIC={res.get('aic', 'N/A')}")

    logger.info("\n" + "=" * 70)
    logger.info("Classical model training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
