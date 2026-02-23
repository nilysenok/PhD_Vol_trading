"""Intraday feature engineering from 10-minute candles using Polars."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
from datetime import time

# Try polars first, fall back to pandas
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    import pandas as pd


class IntradayFeatureCalculator:
    """Calculate volatility features from 10-minute intraday candles.

    Features calculated:
    - RV: Realized Volatility = Σ(log_return²)
    - BV: Bipower Variation = (π/2) · Σ(|r_i| · |r_{i-1}|)
    - Jump: max(RV - BV, 0)
    - RSV+: Positive semi-variance = Σ(r² | r > 0)
    - RSV-: Negative semi-variance = Σ(r² | r < 0)
    - RSkew: Realized Skewness = Σ(r³) / RV^1.5
    - RKurt: Realized Kurtosis = Σ(r⁴) / RV²
    - RV_morning: RV of first 2 hours (10:00-12:00)
    - RV_midday: RV of middle hours (12:00-16:00)
    - RV_evening: RV of last 2 hours (16:00-18:50)
    """

    # MOEX trading hours
    MORNING_START = time(10, 0)
    MORNING_END = time(12, 0)
    MIDDAY_START = time(12, 0)
    MIDDAY_END = time(16, 0)
    EVENING_START = time(16, 0)
    EVENING_END = time(18, 50)

    # Bipower constant
    BV_CONSTANT = np.pi / 2

    def __init__(
        self,
        candles_path: str,
        use_polars: bool = True,
        annualize: bool = True,
        trading_days: int = 252
    ):
        """Initialize calculator.

        Args:
            candles_path: Path to directory with candle parquet files.
            use_polars: Use polars for faster computation.
            annualize: Whether to annualize volatility measures.
            trading_days: Trading days per year for annualization.
        """
        self.candles_path = Path(candles_path)
        self.use_polars = use_polars and POLARS_AVAILABLE
        self.annualize = annualize
        self.trading_days = trading_days
        self._annualization_factor = np.sqrt(trading_days)

    def load_candles(self, ticker: str) -> Union["pl.DataFrame", "pd.DataFrame"]:
        """Load candle data for a ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            DataFrame with candles.
        """
        path = self.candles_path / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Candles file not found: {path}")

        if self.use_polars:
            df = pl.read_parquet(path)
            # Normalize datetime column name to 'begin'
            if "timestamp" in df.columns and "begin" not in df.columns:
                df = df.rename({"timestamp": "begin"})
            if "begin" in df.columns:
                df = df.with_columns(pl.col("begin").cast(pl.Datetime))
        else:
            df = pd.read_parquet(path)
            # Normalize datetime column name to 'begin'
            if "timestamp" in df.columns and "begin" not in df.columns:
                df = df.rename(columns={"timestamp": "begin"})
            if "begin" in df.columns:
                df["begin"] = pd.to_datetime(df["begin"])

        return df

    def calculate_features_polars(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """Calculate all intraday features using Polars.

        Args:
            df: DataFrame with columns: begin, open, high, low, close, volume.

        Returns:
            DataFrame with daily features.
        """
        # Calculate log returns
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),
            pl.col("begin").dt.date().alias("date"),
            pl.col("begin").dt.time().alias("time")
        ])

        # Remove first row (no return) and null returns
        df = df.filter(pl.col("log_return").is_not_null())

        # Calculate squared, cubed, and fourth power returns
        df = df.with_columns([
            (pl.col("log_return") ** 2).alias("r2"),
            (pl.col("log_return") ** 3).alias("r3"),
            (pl.col("log_return") ** 4).alias("r4"),
            pl.col("log_return").abs().alias("r_abs"),
        ])

        # Lagged absolute return for BV
        df = df.with_columns([
            pl.col("r_abs").shift(1).over("date").alias("r_abs_lag1")
        ])

        # Time period indicators
        df = df.with_columns([
            ((pl.col("time") >= pl.time(10, 0)) & (pl.col("time") < pl.time(12, 0))).alias("is_morning"),
            ((pl.col("time") >= pl.time(12, 0)) & (pl.col("time") < pl.time(16, 0))).alias("is_midday"),
            ((pl.col("time") >= pl.time(16, 0)) & (pl.col("time") <= pl.time(18, 50))).alias("is_evening"),
            (pl.col("log_return") > 0).alias("is_positive"),
            (pl.col("log_return") < 0).alias("is_negative"),
        ])

        # Aggregate to daily
        daily = df.group_by("date").agg([
            # Basic RV
            pl.col("r2").sum().alias("rv"),

            # Bipower Variation: (π/2) * Σ(|r_i| * |r_{i-1}|)
            (pl.col("r_abs") * pl.col("r_abs_lag1")).sum().alias("bv_raw"),

            # Higher moments
            pl.col("r3").sum().alias("r3_sum"),
            pl.col("r4").sum().alias("r4_sum"),

            # Signed RV
            pl.col("r2").filter(pl.col("is_positive")).sum().alias("rsv_pos"),
            pl.col("r2").filter(pl.col("is_negative")).sum().alias("rsv_neg"),

            # Intraday patterns
            pl.col("r2").filter(pl.col("is_morning")).sum().alias("rv_morning"),
            pl.col("r2").filter(pl.col("is_midday")).sum().alias("rv_midday"),
            pl.col("r2").filter(pl.col("is_evening")).sum().alias("rv_evening"),

            # Count
            pl.col("log_return").count().alias("n_bars"),

            # Volume
            pl.col("volume").sum().alias("volume"),

            # Close price (last of day)
            pl.col("close").last().alias("close"),
        ])

        # Calculate derived features
        daily = daily.with_columns([
            # BV with constant
            (self.BV_CONSTANT * pl.col("bv_raw")).alias("bv"),

            # Jump = max(RV - BV, 0)
            pl.max_horizontal(pl.col("rv") - self.BV_CONSTANT * pl.col("bv_raw"), pl.lit(0)).alias("jump"),

            # Realized Skewness: Σr³ / RV^1.5
            (pl.col("r3_sum") / (pl.col("rv") ** 1.5 + 1e-10)).alias("rskew"),

            # Realized Kurtosis: Σr⁴ / RV²
            (pl.col("r4_sum") / (pl.col("rv") ** 2 + 1e-10)).alias("rkurt"),

            # Fill nulls for intraday patterns
            pl.col("rsv_pos").fill_null(0),
            pl.col("rsv_neg").fill_null(0),
            pl.col("rv_morning").fill_null(0),
            pl.col("rv_midday").fill_null(0),
            pl.col("rv_evening").fill_null(0),
        ])

        # Annualize if requested
        if self.annualize:
            daily = daily.with_columns([
                (pl.col("rv").sqrt() * self._annualization_factor).alias("rv_annualized"),
                (pl.col("bv").sqrt() * self._annualization_factor).alias("bv_annualized"),
                (pl.col("jump").sqrt() * self._annualization_factor).alias("jump_annualized"),
            ])

        # Sort by date
        daily = daily.sort("date")

        # Drop intermediate columns
        daily = daily.drop(["bv_raw", "r3_sum", "r4_sum"])

        return daily

    def calculate_features_pandas(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Calculate all intraday features using Pandas.

        Args:
            df: DataFrame with columns: begin, open, high, low, close, volume.

        Returns:
            DataFrame with daily features.
        """
        df = df.copy()

        # Calculate log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["date"] = df["begin"].dt.date

        # Remove nulls
        df = df.dropna(subset=["log_return"])

        # Calculate powers
        df["r2"] = df["log_return"] ** 2
        df["r3"] = df["log_return"] ** 3
        df["r4"] = df["log_return"] ** 4
        df["r_abs"] = df["log_return"].abs()
        df["r_abs_lag1"] = df.groupby("date")["r_abs"].shift(1)

        # Time periods
        df["time"] = df["begin"].dt.time
        df["is_morning"] = (df["time"] >= self.MORNING_START) & (df["time"] < self.MORNING_END)
        df["is_midday"] = (df["time"] >= self.MIDDAY_START) & (df["time"] < self.MIDDAY_END)
        df["is_evening"] = (df["time"] >= self.EVENING_START) & (df["time"] <= self.EVENING_END)

        # Aggregate
        daily = df.groupby("date").agg({
            "r2": "sum",
            "r3": "sum",
            "r4": "sum",
            "r_abs": lambda x: (x * df.loc[x.index, "r_abs_lag1"]).sum(),
            "log_return": "count",
            "volume": "sum",
            "close": "last",
        }).rename(columns={
            "r2": "rv",
            "log_return": "n_bars",
            "r_abs": "bv_raw",
        })

        # Signed RV
        daily["rsv_pos"] = df[df["log_return"] > 0].groupby("date")["r2"].sum()
        daily["rsv_neg"] = df[df["log_return"] < 0].groupby("date")["r2"].sum()

        # Intraday patterns
        daily["rv_morning"] = df[df["is_morning"]].groupby("date")["r2"].sum()
        daily["rv_midday"] = df[df["is_midday"]].groupby("date")["r2"].sum()
        daily["rv_evening"] = df[df["is_evening"]].groupby("date")["r2"].sum()

        # Fill NaN
        for col in ["rsv_pos", "rsv_neg", "rv_morning", "rv_midday", "rv_evening"]:
            daily[col] = daily[col].fillna(0)

        # Derived features
        daily["bv"] = self.BV_CONSTANT * daily["bv_raw"]
        daily["jump"] = np.maximum(daily["rv"] - daily["bv"], 0)
        daily["rskew"] = daily["r3"] / (daily["rv"] ** 1.5 + 1e-10)
        daily["rkurt"] = daily["r4"] / (daily["rv"] ** 2 + 1e-10)

        if self.annualize:
            daily["rv_annualized"] = np.sqrt(daily["rv"]) * self._annualization_factor
            daily["bv_annualized"] = np.sqrt(daily["bv"]) * self._annualization_factor
            daily["jump_annualized"] = np.sqrt(daily["jump"]) * self._annualization_factor

        # Clean up
        daily = daily.drop(columns=["bv_raw", "r3", "r4"], errors="ignore")
        daily = daily.reset_index()

        return daily

    def calculate_features(
        self,
        ticker: str
    ) -> Union["pl.DataFrame", "pd.DataFrame"]:
        """Calculate all features for a ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            DataFrame with daily features.
        """
        df = self.load_candles(ticker)

        if self.use_polars:
            result = self.calculate_features_polars(df)
            result = result.with_columns(pl.lit(ticker).alias("ticker"))
        else:
            result = self.calculate_features_pandas(df)
            result["ticker"] = ticker

        return result

    def calculate_all_tickers(
        self,
        tickers: List[str],
        verbose: bool = True
    ) -> Union["pl.DataFrame", "pd.DataFrame"]:
        """Calculate features for all tickers.

        Args:
            tickers: List of ticker symbols.
            verbose: Print progress.

        Returns:
            Combined DataFrame with all features.
        """
        results = []

        for i, ticker in enumerate(tickers):
            if verbose:
                print(f"Processing {ticker} ({i+1}/{len(tickers)})...")

            try:
                df = self.calculate_features(ticker)
                results.append(df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        if self.use_polars:
            return pl.concat(results)
        else:
            return pd.concat(results, ignore_index=True)

    def add_har_lags(
        self,
        df: Union["pl.DataFrame", "pd.DataFrame"],
        rv_col: str = "rv",
        lags: Dict[str, int] = None
    ) -> Union["pl.DataFrame", "pd.DataFrame"]:
        """Add HAR-style lagged features.

        Args:
            df: DataFrame with RV data.
            rv_col: Name of RV column.
            lags: Dictionary of lag names and windows.
                  Default: {rv_d: 1, rv_w: 5, rv_m: 22, rv_q: 66}

        Returns:
            DataFrame with lag features.
        """
        if lags is None:
            lags = {"rv_d": 1, "rv_w": 5, "rv_2w": 10, "rv_m": 22, "rv_q": 66}

        if self.use_polars:
            # Daily lag (shift by 1)
            df = df.with_columns([
                pl.col(rv_col).shift(1).over("ticker").alias("rv_d")
            ])

            # Rolling averages (shifted to avoid lookahead)
            for name, window in lags.items():
                if name == "rv_d":
                    continue
                df = df.with_columns([
                    pl.col(rv_col)
                    .rolling_mean(window_size=window, min_periods=1)
                    .shift(1)
                    .over("ticker")
                    .alias(name)
                ])
        else:
            df = df.copy()
            df = df.sort_values(["ticker", "date"])

            # Daily lag
            df["rv_d"] = df.groupby("ticker")[rv_col].shift(1)

            # Rolling averages
            for name, window in lags.items():
                if name == "rv_d":
                    continue
                df[name] = df.groupby("ticker")[rv_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )

        return df

    def add_extended_lags(
        self,
        df: Union["pl.DataFrame", "pd.DataFrame"],
        columns: List[str],
        lags: List[int] = [1, 2, 3, 4, 5, 10, 22, 44, 66]
    ) -> Union["pl.DataFrame", "pd.DataFrame"]:
        """Add extended lag features for ML models.

        Args:
            df: DataFrame with features.
            columns: Columns to create lags for.
            lags: List of lag periods.

        Returns:
            DataFrame with lag features.
        """
        if self.use_polars:
            for col in columns:
                for lag in lags:
                    df = df.with_columns([
                        pl.col(col).shift(lag).over("ticker").alias(f"{col}_lag{lag}")
                    ])
        else:
            df = df.copy()
            for col in columns:
                for lag in lags:
                    df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)

        return df


def build_stock_features(
    candles_path: str,
    tickers: List[str],
    output_path: str,
    use_polars: bool = True,
    verbose: bool = True
) -> None:
    """Build and save stock features from 10-minute candles.

    Args:
        candles_path: Path to candles directory.
        tickers: List of tickers.
        output_path: Path to save output parquet.
        use_polars: Use polars for computation.
        verbose: Print progress.
    """
    calculator = IntradayFeatureCalculator(
        candles_path=candles_path,
        use_polars=use_polars,
        annualize=True
    )

    # Calculate base features
    if verbose:
        print("Calculating intraday features...")
    df = calculator.calculate_all_tickers(tickers, verbose=verbose)

    # Add HAR lags
    if verbose:
        print("Adding HAR lags...")
    df = calculator.add_har_lags(df)

    # Add extended lags for jump
    if verbose:
        print("Adding extended lags...")
    df = calculator.add_extended_lags(df, columns=["jump"], lags=[1, 5, 22])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if use_polars and POLARS_AVAILABLE:
        df.write_parquet(output_path)
    else:
        df.to_parquet(output_path, index=False)

    if verbose:
        print(f"Saved to {output_path}")
        print(f"Shape: {df.shape}")
