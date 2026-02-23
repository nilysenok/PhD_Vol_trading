"""Evaluation metrics for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    qlike: float
    correlation: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "MSE": self.mse,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE": self.mape,
            "R2": self.r2,
            "QLIKE": self.qlike,
            "Correlation": self.correlation,
        }

    def __str__(self) -> str:
        lines = [
            f"MSE:         {self.mse:.6f}",
            f"RMSE:        {self.rmse:.6f}",
            f"MAE:         {self.mae:.6f}",
            f"MAPE:        {self.mape:.2f}%",
            f"R²:          {self.r2:.4f}",
            f"QLIKE:       {self.qlike:.6f}",
            f"Correlation: {self.correlation:.4f}",
        ]
        return "\n".join(lines)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MSE value.
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero.

    Returns:
        MAPE value in percentage.
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        R² value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def qlike(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """QLIKE (Quasi-Likelihood) loss function.

    Robust loss function for volatility forecasting that handles
    the asymmetry between under and over-prediction.

    QLIKE = E[log(σ²_pred) + σ²_true / σ²_pred]

    Reference:
        Patton, A. J. (2011). Volatility forecast comparison using
        imperfect volatility proxies.

    Args:
        y_true: Actual volatility (RV).
        y_pred: Predicted volatility.
        epsilon: Small value to avoid log(0).

    Returns:
        QLIKE value (lower is better).
    """
    y_pred_safe = np.maximum(y_pred, epsilon)
    return np.mean(np.log(y_pred_safe) + y_true / y_pred_safe)


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Correlation coefficient.
    """
    if len(y_true) < 2:
        return 0.0

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> ForecastMetrics:
    """Evaluate forecast with all metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        ForecastMetrics object with all metrics.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return ForecastMetrics(
        mse=mse(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        qlike=qlike(y_true, y_pred),
        correlation=correlation(y_true, y_pred),
    )


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    primary_metric: str = "RMSE"
) -> pd.DataFrame:
    """Compare multiple models.

    Args:
        y_true: Actual values.
        predictions: Dictionary mapping model names to predictions.
        primary_metric: Metric to sort by.

    Returns:
        DataFrame with metrics for each model.
    """
    results = []

    for name, y_pred in predictions.items():
        metrics = evaluate_forecast(y_true, y_pred)
        row = {"Model": name, **metrics.to_dict()}
        results.append(row)

    df = pd.DataFrame(results)

    # Sort by primary metric (lower is better for most metrics)
    ascending = primary_metric not in ["R2", "Correlation"]
    df = df.sort_values(primary_metric, ascending=ascending)

    return df.reset_index(drop=True)


def heteroscedasticity_adjusted_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Heteroscedasticity-adjusted MSE.

    Weights errors by inverse variance to account for
    heteroscedasticity in volatility forecasts.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        weights: Custom weights (if None, use 1/y_true²).

    Returns:
        Weighted MSE.
    """
    if weights is None:
        weights = 1 / (y_true ** 2 + 1e-10)

    weights = weights / weights.sum()
    return np.sum(weights * (y_true - y_pred) ** 2)
