"""Training utilities for volatility forecasting models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Type, Union
from pathlib import Path
import json
from datetime import datetime

from ..models.base import BaseVolatilityModel
from ..evaluation.metrics import evaluate_forecast, ForecastMetrics
from ..utils.logger import TrainingLogger, get_logger
from .walk_forward import WalkForwardCV, WalkForwardSplit


class Trainer:
    """Trainer for volatility forecasting models."""

    def __init__(
        self,
        model_class: Type[BaseVolatilityModel],
        model_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "results",
        experiment_name: Optional[str] = None,
        seed: int = 42
    ):
        """Initialize trainer.

        Args:
            model_class: Model class to train.
            model_params: Model initialization parameters.
            output_dir: Directory for saving results.
            experiment_name: Name for this experiment.
            seed: Random seed.
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.seed = seed

        # Setup
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger(name=self.experiment_name)

        # Results storage
        self.results: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.models: Dict[str, BaseVolatilityModel] = {}

    def train_single_ticker(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        ticker: str = "unknown",
        **fit_kwargs
    ) -> Tuple[BaseVolatilityModel, ForecastMetrics]:
        """Train model for a single ticker.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            ticker: Ticker symbol.
            **fit_kwargs: Additional fitting parameters.

        Returns:
            Tuple of (trained_model, validation_metrics).
        """
        self.logger.start_training(
            model_name=self.model_class.__name__,
            ticker=ticker,
            n_samples=len(X_train)
        )

        # Initialize model
        model = self.model_class(**self.model_params)

        # Fit
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs)
        else:
            model.fit(X_train, y_train, **fit_kwargs)

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)

            # Handle sequence models
            if len(y_pred) < len(y_val):
                y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
                offset = len(y_val_arr) - len(y_pred)
                y_val_arr = y_val_arr[offset:]
            else:
                y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val

            metrics = evaluate_forecast(y_val_arr, y_pred)
        else:
            # Use in-sample metrics
            y_pred = model.predict(X_train)
            y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
            if len(y_pred) < len(y_train_arr):
                offset = len(y_train_arr) - len(y_pred)
                y_train_arr = y_train_arr[offset:]
            metrics = evaluate_forecast(y_train_arr, y_pred)

        self.logger.end_training(metrics.to_dict())

        # Store
        self.models[ticker] = model
        self.results[ticker] = metrics.to_dict()

        return model, metrics

    def train_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: WalkForwardCV,
        ticker: str = "unknown",
        save_models: bool = False,
        **fit_kwargs
    ) -> Dict[str, Any]:
        """Train model using walk-forward cross-validation.

        Args:
            X: Features.
            y: Target.
            cv: WalkForwardCV instance.
            ticker: Ticker symbol.
            save_models: Whether to save models for each fold.
            **fit_kwargs: Additional fitting parameters.

        Returns:
            Dictionary with results.
        """
        logger = get_logger()
        logger.info(f"Walk-forward CV for {ticker} with {cv.get_n_splits(len(X))} folds")

        dates = X.index if isinstance(X.index, pd.DatetimeIndex) else None

        fold_metrics = []
        all_preds = []
        all_actuals = []
        all_dates = []

        for X_train, X_test, y_train, y_test, split in cv.split_arrays(X, y, dates):
            logger.info(f"Fold {split.fold + 1}: Train={len(X_train)}, Test={len(X_test)}")

            # Initialize and fit model
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train, **fit_kwargs)

            # Predict
            y_pred = model.predict(X_test)

            # Handle sequence models
            if len(y_pred) < len(y_test):
                offset = len(y_test) - len(y_pred)
                y_test = y_test[offset:]
                if dates is not None:
                    test_dates = dates[split.test_start + offset:split.test_end]
                else:
                    test_dates = None
            else:
                test_dates = dates[split.test_start:split.test_end] if dates is not None else None

            # Store predictions
            all_preds.extend(y_pred)
            all_actuals.extend(y_test)
            if test_dates is not None:
                all_dates.extend(test_dates)

            # Calculate fold metrics
            fold_metric = evaluate_forecast(y_test, y_pred)
            fold_metrics.append({
                "fold": split.fold,
                "train_size": len(X_train),
                "test_size": len(y_pred),
                **fold_metric.to_dict()
            })

            logger.info(f"  RMSE: {fold_metric.rmse:.6f}, R²: {fold_metric.r2:.4f}")

            # Save model if requested
            if save_models:
                model_path = self.output_dir / "models" / f"{ticker}_fold{split.fold}.pkl"
                model.save(str(model_path))

        # Aggregate metrics
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        overall_metrics = evaluate_forecast(all_actuals, all_preds)

        results = {
            "ticker": ticker,
            "n_folds": len(fold_metrics),
            "fold_metrics": fold_metrics,
            "overall_metrics": overall_metrics.to_dict(),
            "predictions": all_preds,
            "actuals": all_actuals,
            "dates": all_dates if all_dates else None,
        }

        # Store
        self.results[ticker] = results
        self.predictions[ticker] = all_preds

        return results

    def train_all_tickers(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_h1",
        ticker_col: str = "ticker",
        cv: Optional[WalkForwardCV] = None,
        tickers: Optional[List[str]] = None,
        **fit_kwargs
    ) -> pd.DataFrame:
        """Train models for all tickers.

        Args:
            data: DataFrame with all data.
            feature_cols: Feature column names.
            target_col: Target column name.
            ticker_col: Ticker column name.
            cv: WalkForwardCV instance (if None, simple train/test split).
            tickers: List of tickers to train (None for all).
            **fit_kwargs: Additional fitting parameters.

        Returns:
            DataFrame with results for all tickers.
        """
        if tickers is None:
            tickers = data[ticker_col].unique().tolist()

        all_results = []

        for ticker in tickers:
            ticker_data = data[data[ticker_col] == ticker].copy()
            ticker_data = ticker_data.dropna(subset=feature_cols + [target_col])

            if len(ticker_data) < 100:
                get_logger().warning(f"Skipping {ticker}: insufficient data ({len(ticker_data)} rows)")
                continue

            X = ticker_data[feature_cols]
            y = ticker_data[target_col]

            if cv is not None:
                results = self.train_walk_forward(X, y, cv, ticker=ticker, **fit_kwargs)
                all_results.append({
                    "ticker": ticker,
                    "n_samples": len(ticker_data),
                    **results["overall_metrics"]
                })
            else:
                # Simple train/test split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                model, metrics = self.train_single_ticker(
                    X_train, y_train, X_test, y_test, ticker=ticker, **fit_kwargs
                )
                all_results.append({
                    "ticker": ticker,
                    "n_samples": len(ticker_data),
                    **metrics.to_dict()
                })

        return pd.DataFrame(all_results)

    def save_results(self, filename: Optional[str] = None) -> None:
        """Save results to JSON.

        Args:
            filename: Output filename.
        """
        if filename is None:
            filename = f"{self.experiment_name}_results.json"

        output_path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_json[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                results_json[key] = value

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, default=str)

        get_logger().info(f"Results saved to {output_path}")

    def save_predictions(self, filename: Optional[str] = None) -> None:
        """Save predictions to parquet.

        Args:
            filename: Output filename.
        """
        if filename is None:
            filename = f"{self.experiment_name}_predictions.parquet"

        output_path = self.output_dir / filename

        # Combine all predictions
        dfs = []
        for ticker, results in self.results.items():
            if isinstance(results, dict) and "predictions" in results:
                df = pd.DataFrame({
                    "ticker": ticker,
                    "prediction": results["predictions"],
                    "actual": results["actuals"],
                })
                if results.get("dates"):
                    df["date"] = results["dates"]
                dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_parquet(output_path)
            get_logger().info(f"Predictions saved to {output_path}")


def train_and_evaluate(
    model_class: Type[BaseVolatilityModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_params: Optional[Dict] = None,
    fit_params: Optional[Dict] = None
) -> Tuple[BaseVolatilityModel, ForecastMetrics, np.ndarray]:
    """Convenience function for quick train and evaluate.

    Args:
        model_class: Model class.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        model_params: Model parameters.
        fit_params: Fitting parameters.

    Returns:
        Tuple of (model, metrics, predictions).
    """
    model_params = model_params or {}
    fit_params = fit_params or {}

    model = model_class(**model_params)
    model.fit(X_train, y_train, **fit_params)

    y_pred = model.predict(X_test)

    # Handle sequence models
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
    if len(y_pred) < len(y_test_arr):
        offset = len(y_test_arr) - len(y_pred)
        y_test_arr = y_test_arr[offset:]

    metrics = evaluate_forecast(y_test_arr, y_pred)

    return model, metrics, y_pred
