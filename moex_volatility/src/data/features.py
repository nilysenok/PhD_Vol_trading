"""Feature engineering for volatility forecasting."""

import pandas as pd
import numpy as np
from typing import List, Optional, Union


class FeatureEngineer:
    """Feature engineering for realized volatility forecasting."""

    def __init__(self, df: pd.DataFrame, ticker_col: str = "ticker", date_col: str = "date"):
        """Initialize feature engineer.

        Args:
            df: Input DataFrame with RV data.
            ticker_col: Name of ticker column.
            date_col: Name of date column.
        """
        self.df = df.copy()
        self.ticker_col = ticker_col
        self.date_col = date_col
        self._ensure_sorted()

    def _ensure_sorted(self) -> None:
        """Ensure data is sorted by ticker and date."""
        self.df = self.df.sort_values([self.ticker_col, self.date_col])

    def add_lags(
        self,
        columns: Union[str, List[str]],
        lags: List[int],
        prefix: Optional[str] = None
    ) -> "FeatureEngineer":
        """Add lagged features.

        Args:
            columns: Column(s) to create lags for.
            lags: List of lag periods.
            prefix: Optional prefix for new column names.

        Returns:
            Self for chaining.
        """
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            for lag in lags:
                new_col = f"{prefix or col}_lag{lag}"
                self.df[new_col] = self.df.groupby(self.ticker_col)[col].shift(lag)

        return self

    def add_rolling_stats(
        self,
        columns: Union[str, List[str]],
        windows: List[int],
        stats: List[str] = ["mean", "std"],
        min_periods: Optional[int] = None
    ) -> "FeatureEngineer":
        """Add rolling statistics.

        Args:
            columns: Column(s) to compute stats for.
            windows: List of window sizes.
            stats: Statistics to compute ('mean', 'std', 'min', 'max', 'median').
            min_periods: Minimum periods for valid calculation.

        Returns:
            Self for chaining.
        """
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            for window in windows:
                for stat in stats:
                    new_col = f"{col}_roll{window}_{stat}"
                    min_p = min_periods or window

                    if stat == "mean":
                        self.df[new_col] = self.df.groupby(self.ticker_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=min_p).mean()
                        )
                    elif stat == "std":
                        self.df[new_col] = self.df.groupby(self.ticker_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=min_p).std()
                        )
                    elif stat == "min":
                        self.df[new_col] = self.df.groupby(self.ticker_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=min_p).min()
                        )
                    elif stat == "max":
                        self.df[new_col] = self.df.groupby(self.ticker_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=min_p).max()
                        )
                    elif stat == "median":
                        self.df[new_col] = self.df.groupby(self.ticker_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=min_p).median()
                        )

        return self

    def add_har_features(
        self,
        rv_col: str = "rv_daily",
        daily_lag: int = 1,
        weekly_window: int = 5,
        monthly_window: int = 22
    ) -> "FeatureEngineer":
        """Add HAR (Heterogeneous Autoregressive) model features.

        HAR model uses RV at different aggregation levels:
        - RV_d: Daily RV (lag 1)
        - RV_w: Weekly average (5-day rolling)
        - RV_m: Monthly average (22-day rolling)

        Args:
            rv_col: Name of RV column.
            daily_lag: Lag for daily component.
            weekly_window: Window for weekly average.
            monthly_window: Window for monthly average.

        Returns:
            Self for chaining.
        """
        # Daily lag
        self.df["rv_d"] = self.df.groupby(self.ticker_col)[rv_col].shift(daily_lag)

        # Weekly average (shifted to avoid lookahead)
        self.df["rv_w"] = self.df.groupby(self.ticker_col)[rv_col].transform(
            lambda x: x.rolling(weekly_window, min_periods=1).mean().shift(daily_lag)
        )

        # Monthly average (shifted to avoid lookahead)
        self.df["rv_m"] = self.df.groupby(self.ticker_col)[rv_col].transform(
            lambda x: x.rolling(monthly_window, min_periods=1).mean().shift(daily_lag)
        )

        return self

    def add_log_features(
        self,
        columns: Union[str, List[str]],
        prefix: str = "log_"
    ) -> "FeatureEngineer":
        """Add log-transformed features.

        Args:
            columns: Column(s) to transform.
            prefix: Prefix for new column names.

        Returns:
            Self for chaining.
        """
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            new_col = f"{prefix}{col}"
            # Use log1p for numerical stability with small values
            self.df[new_col] = np.log1p(self.df[col].clip(lower=0))

        return self

    def add_returns(
        self,
        price_col: str = "close",
        periods: List[int] = [1, 5, 22]
    ) -> "FeatureEngineer":
        """Add return features.

        Args:
            price_col: Name of price column.
            periods: List of periods for return calculation.

        Returns:
            Self for chaining.
        """
        for period in periods:
            # Simple returns
            self.df[f"return_{period}d"] = self.df.groupby(self.ticker_col)[price_col].pct_change(period)

            # Log returns
            self.df[f"log_return_{period}d"] = self.df.groupby(self.ticker_col)[price_col].transform(
                lambda x: np.log(x / x.shift(period))
            )

        return self

    def add_volatility_of_volatility(
        self,
        rv_col: str = "rv_daily",
        windows: List[int] = [5, 22]
    ) -> "FeatureEngineer":
        """Add volatility of volatility (VoV) features.

        Args:
            rv_col: Name of RV column.
            windows: Windows for VoV calculation.

        Returns:
            Self for chaining.
        """
        for window in windows:
            new_col = f"vov_{window}d"
            self.df[new_col] = self.df.groupby(self.ticker_col)[rv_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        return self

    def add_relative_rv(
        self,
        rv_col: str = "rv_daily",
        benchmark_col: str = "rv_IMOEX"
    ) -> "FeatureEngineer":
        """Add relative RV (stock RV / benchmark RV).

        Args:
            rv_col: Name of stock RV column.
            benchmark_col: Name of benchmark RV column.

        Returns:
            Self for chaining.
        """
        # Avoid division by zero
        benchmark = self.df[benchmark_col].replace(0, np.nan)
        self.df["rv_relative"] = self.df[rv_col] / benchmark

        return self

    def add_day_of_week(self) -> "FeatureEngineer":
        """Add day of week features.

        Returns:
            Self for chaining.
        """
        self.df["day_of_week"] = pd.to_datetime(self.df[self.date_col]).dt.dayofweek

        # One-hot encoding
        for i in range(5):  # Monday to Friday
            self.df[f"dow_{i}"] = (self.df["day_of_week"] == i).astype(int)

        return self

    def add_month(self) -> "FeatureEngineer":
        """Add month features.

        Returns:
            Self for chaining.
        """
        self.df["month"] = pd.to_datetime(self.df[self.date_col]).dt.month
        return self

    def add_jump_indicator(
        self,
        rv_col: str = "rv_daily",
        threshold_std: float = 3.0
    ) -> "FeatureEngineer":
        """Add jump indicator (extreme RV days).

        Args:
            rv_col: Name of RV column.
            threshold_std: Number of std deviations for jump threshold.

        Returns:
            Self for chaining.
        """
        # Calculate rolling mean and std
        rolling_mean = self.df.groupby(self.ticker_col)[rv_col].transform(
            lambda x: x.rolling(66, min_periods=22).mean()
        )
        rolling_std = self.df.groupby(self.ticker_col)[rv_col].transform(
            lambda x: x.rolling(66, min_periods=22).std()
        )

        threshold = rolling_mean + threshold_std * rolling_std
        self.df["rv_jump"] = (self.df[rv_col] > threshold).astype(int)

        return self

    def shift_target(
        self,
        target_col: str = "rv_daily",
        horizon: int = 1
    ) -> "FeatureEngineer":
        """Shift target variable for prediction horizon.

        Args:
            target_col: Name of target column.
            horizon: Forecast horizon (positive = future).

        Returns:
            Self for chaining.
        """
        self.df[f"target_h{horizon}"] = self.df.groupby(self.ticker_col)[target_col].shift(-horizon)
        return self

    def dropna(self, subset: Optional[List[str]] = None) -> "FeatureEngineer":
        """Drop rows with NaN values.

        Args:
            subset: Columns to check for NaN. If None, check all.

        Returns:
            Self for chaining.
        """
        self.df = self.df.dropna(subset=subset)
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """Get the processed DataFrame.

        Returns:
            Processed DataFrame with features.
        """
        return self.df


def create_features_for_har(
    df: pd.DataFrame,
    rv_col: str = "rv_daily",
    ticker_col: str = "ticker",
    date_col: str = "date"
) -> pd.DataFrame:
    """Create features for HAR model.

    Args:
        df: Input DataFrame.
        rv_col: Name of RV column.
        ticker_col: Name of ticker column.
        date_col: Name of date column.

    Returns:
        DataFrame with HAR features.
    """
    fe = FeatureEngineer(df, ticker_col=ticker_col, date_col=date_col)
    fe.add_har_features(rv_col=rv_col)
    fe.shift_target(target_col=rv_col, horizon=1)
    return fe.get_dataframe()


def create_features_for_ml(
    df: pd.DataFrame,
    rv_col: str = "rv_daily",
    ticker_col: str = "ticker",
    date_col: str = "date",
    lags: List[int] = [1, 2, 3, 4, 5, 10, 22],
    rolling_windows: List[int] = [5, 10, 22],
    include_external: bool = True
) -> pd.DataFrame:
    """Create features for ML models (LightGBM, XGBoost).

    Args:
        df: Input DataFrame.
        rv_col: Name of RV column.
        ticker_col: Name of ticker column.
        date_col: Name of date column.
        lags: Lags to create.
        rolling_windows: Rolling windows for statistics.
        include_external: Whether to add external features.

    Returns:
        DataFrame with ML features.
    """
    fe = FeatureEngineer(df, ticker_col=ticker_col, date_col=date_col)

    # HAR features
    fe.add_har_features(rv_col=rv_col)

    # Extended lags
    fe.add_lags(rv_col, lags, prefix="rv")

    # Rolling statistics
    fe.add_rolling_stats(rv_col, rolling_windows, stats=["mean", "std"])

    # Volatility of volatility
    fe.add_volatility_of_volatility(rv_col)

    # Log features
    fe.add_log_features([rv_col])

    # Time features
    fe.add_day_of_week()
    fe.add_month()

    # Jump indicator
    fe.add_jump_indicator(rv_col)

    # Returns if close price available
    if "close" in df.columns:
        fe.add_returns("close", periods=[1, 5, 22])

    # Relative RV if benchmark available
    if include_external and "rv_IMOEX" in df.columns:
        fe.add_relative_rv(rv_col, "rv_IMOEX")

    # Target
    fe.shift_target(target_col=rv_col, horizon=1)

    return fe.get_dataframe()
