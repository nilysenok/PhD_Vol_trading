"""Helper utilities for moex_volatility project."""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save object to pickle file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Dict, path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_trading_days(
    start: Union[str, datetime],
    end: Union[str, datetime],
    dates: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Get trading days between start and end from available dates."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return dates[(dates >= start) & (dates <= end)]


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = "date"
) -> tuple:
    """Split DataFrame by date into train, validation, and test sets."""
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        train = df[df[date_col] <= train_end]
        val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
        test = df[df[date_col] > val_end]
    else:
        # Assume index is date
        df.index = pd.to_datetime(df.index)
        train = df[df.index <= train_end]
        val = df[(df.index > train_end) & (df.index <= val_end)]
        test = df[df.index > val_end]

    return train, val, test


def create_target(
    df: pd.DataFrame,
    target_col: str = "rv",
    horizon: int = 1,
    use_log: bool = True
) -> pd.DataFrame:
    """Create forward-looking target variable."""
    df = df.copy()

    if use_log:
        target = np.log(df[target_col] + 1e-8)
    else:
        target = df[target_col]

    # Shift target backwards (forward-looking)
    df[f"y_h{horizon}"] = target.shift(-horizon)

    return df


def annualize_volatility(daily_rv: float, trading_days: int = 252) -> float:
    """Convert daily realized volatility to annualized."""
    return np.sqrt(daily_rv * trading_days)


def deannualize_volatility(annual_vol: float, trading_days: int = 252) -> float:
    """Convert annualized volatility to daily."""
    return (annual_vol ** 2) / trading_days


def winsorize(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99
) -> pd.Series:
    """Winsorize series to remove extreme values."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def standardize(
    series: pd.Series,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> tuple:
    """Standardize series to zero mean and unit variance."""
    if mean is None:
        mean = series.mean()
    if std is None:
        std = series.std()

    standardized = (series - mean) / (std + 1e-8)
    return standardized, mean, std


def inverse_standardize(
    series: pd.Series,
    mean: float,
    std: float
) -> pd.Series:
    """Inverse standardization."""
    return series * std + mean


def check_gpu_available() -> Dict[str, bool]:
    """Check if GPU is available for different frameworks."""
    result = {"cuda": False, "lightgbm_gpu": False, "xgboost_gpu": False}

    # Check PyTorch CUDA
    try:
        import torch
        result["cuda"] = torch.cuda.is_available()
        if result["cuda"]:
            result["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Check LightGBM GPU
    try:
        import lightgbm as lgb
        # LightGBM GPU check is indirect
        result["lightgbm_gpu"] = result["cuda"]
    except ImportError:
        pass

    # Check XGBoost GPU
    try:
        import xgboost as xgb
        result["xgboost_gpu"] = result["cuda"]
    except ImportError:
        pass

    return result


def timer(func):
    """Decorator to time function execution."""
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    return wrapper


def format_number(num: float, decimals: int = 4) -> str:
    """Format number for display."""
    if abs(num) < 0.0001:
        return f"{num:.2e}"
    return f"{num:.{decimals}f}"


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    prefix: str = ""
) -> pd.DataFrame:
    """Create lagged features for specified columns."""
    df = df.copy()

    for col in columns:
        for lag in lags:
            lag_name = f"{prefix}{col}_lag{lag}" if prefix else f"{col}_lag{lag}"
            df[lag_name] = df[col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    agg_funcs: List[str] = ["mean", "std"]
) -> pd.DataFrame:
    """Create rolling window features."""
    df = df.copy()

    for col in columns:
        for window in windows:
            for func in agg_funcs:
                feat_name = f"{col}_roll{window}_{func}"
                if func == "mean":
                    df[feat_name] = df[col].rolling(window).mean()
                elif func == "std":
                    df[feat_name] = df[col].rolling(window).std()
                elif func == "min":
                    df[feat_name] = df[col].rolling(window).min()
                elif func == "max":
                    df[feat_name] = df[col].rolling(window).max()
                elif func == "sum":
                    df[feat_name] = df[col].rolling(window).sum()

    return df


def get_feature_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
    """Get feature column names, excluding target and metadata columns."""
    exclude_patterns = exclude_patterns or ["y_", "date", "ticker", "target"]

    feature_cols = []
    for col in df.columns:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in col.lower():
                exclude = True
                break
        if not exclude:
            feature_cols.append(col)

    return feature_cols


class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = datetime.now()

    def update(self, n: int = 1) -> None:
        self.current += n
        pct = self.current / self.total * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = elapsed / self.current * (self.total - self.current) if self.current > 0 else 0
        print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.1f}%) - ETA: {eta:.0f}s", end="")

        if self.current >= self.total:
            print()  # New line when done

    def reset(self) -> None:
        self.current = 0
        self.start_time = datetime.now()
