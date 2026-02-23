"""External factor feature engineering."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

import pandas as pd


class ExternalFeatureBuilder:
    """Build features from external factors (indices, Yahoo, macro).

    For each external factor X, calculates:
    - X_lag1: X(t-1)
    - X_ret1: log(X_t / X_{t-1})
    - X_ret5: log(X_t / X_{t-5})
    - X_ret22: log(X_t / X_{t-22})
    - X_ma5: 5-day moving average
    - X_ma22: 22-day moving average
    """

    def __init__(self, base_path: str, use_polars: bool = True):
        """Initialize builder.

        Args:
            base_path: Path to dataset_final directory.
            use_polars: Use polars for computation.
        """
        self.base_path = Path(base_path)
        self.use_polars = use_polars and POLARS_AVAILABLE

    def load_moex_indices_rv(self) -> Union["pl.DataFrame", pd.DataFrame]:
        """Load MOEX indices RV data.

        Returns:
            DataFrame with index RV.
        """
        path = self.base_path / "02_external" / "rv_moex.parquet"

        if self.use_polars:
            df = pl.read_parquet(path)
        else:
            df = pd.read_parquet(path)

        return df

    def load_yahoo(self) -> Union["pl.DataFrame", pd.DataFrame]:
        """Load Yahoo Finance data.

        Returns:
            DataFrame with global market data.
        """
        path = self.base_path / "02_external" / "yahoo.parquet"

        if self.use_polars:
            df = pl.read_parquet(path)
        else:
            df = pd.read_parquet(path)

        return df

    def load_macro(self) -> Union["pl.DataFrame", pd.DataFrame]:
        """Load macro data.

        Returns:
            DataFrame with macro indicators.
        """
        path = self.base_path / "02_external" / "macro.parquet"

        if self.use_polars:
            df = pl.read_parquet(path)
        else:
            df = pd.read_parquet(path)

        return df

    def add_factor_features_polars(
        self,
        df: "pl.DataFrame",
        columns: List[str],
        prefix: str = ""
    ) -> "pl.DataFrame":
        """Add lag and return features for factors using Polars.

        Args:
            df: DataFrame with factors.
            columns: Factor columns to process.
            prefix: Prefix for new column names.

        Returns:
            DataFrame with added features.
        """
        for col in columns:
            col_name = f"{prefix}{col}" if prefix else col

            df = df.with_columns([
                # Lag 1
                pl.col(col).shift(1).alias(f"{col_name}_lag1"),

                # Log returns
                (pl.col(col) / pl.col(col).shift(1)).log().alias(f"{col_name}_ret1"),
                (pl.col(col) / pl.col(col).shift(5)).log().alias(f"{col_name}_ret5"),
                (pl.col(col) / pl.col(col).shift(22)).log().alias(f"{col_name}_ret22"),

                # Moving averages
                pl.col(col).rolling_mean(window_size=5, min_periods=1).alias(f"{col_name}_ma5"),
                pl.col(col).rolling_mean(window_size=22, min_periods=1).alias(f"{col_name}_ma22"),

                # Volatility (rolling std)
                pl.col(col).rolling_std(window_size=22, min_periods=5).alias(f"{col_name}_vol22"),
            ])

        return df

    def add_factor_features_pandas(
        self,
        df: pd.DataFrame,
        columns: List[str],
        prefix: str = ""
    ) -> pd.DataFrame:
        """Add lag and return features for factors using Pandas.

        Args:
            df: DataFrame with factors.
            columns: Factor columns to process.
            prefix: Prefix for new column names.

        Returns:
            DataFrame with added features.
        """
        df = df.copy()

        for col in columns:
            col_name = f"{prefix}{col}" if prefix else col

            # Lag 1
            df[f"{col_name}_lag1"] = df[col].shift(1)

            # Log returns
            df[f"{col_name}_ret1"] = np.log(df[col] / df[col].shift(1))
            df[f"{col_name}_ret5"] = np.log(df[col] / df[col].shift(5))
            df[f"{col_name}_ret22"] = np.log(df[col] / df[col].shift(22))

            # Moving averages
            df[f"{col_name}_ma5"] = df[col].rolling(5, min_periods=1).mean()
            df[f"{col_name}_ma22"] = df[col].rolling(22, min_periods=1).mean()

            # Volatility
            df[f"{col_name}_vol22"] = df[col].rolling(22, min_periods=5).std()

        return df

    def build_moex_features(self) -> Union["pl.DataFrame", pd.DataFrame]:
        """Build features from MOEX indices RV.

        Returns:
            DataFrame with MOEX index features.
        """
        df = self.load_moex_indices_rv()

        # Get unique tickers
        if self.use_polars:
            tickers = df["ticker"].unique().to_list()
        else:
            tickers = df["ticker"].unique().tolist()

        # Process each ticker
        results = []
        for ticker in tickers:
            if self.use_polars:
                ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
                # Add features for rv_annualized
                ticker_df = self.add_factor_features_polars(
                    ticker_df,
                    columns=["rv_annualized"],
                    prefix=f"idx_{ticker}_"
                )
                # Rename rv columns
                ticker_df = ticker_df.rename({
                    "rv_annualized": f"rv_{ticker}",
                    "close": f"close_{ticker}",
                })
                ticker_df = ticker_df.drop(["ticker", "rv_daily", "n_bars"])
            else:
                ticker_df = df[df["ticker"] == ticker].copy().sort_values("date")
                ticker_df = self.add_factor_features_pandas(
                    ticker_df,
                    columns=["rv_annualized"],
                    prefix=f"idx_{ticker}_"
                )
                ticker_df = ticker_df.rename(columns={
                    "rv_annualized": f"rv_{ticker}",
                    "close": f"close_{ticker}",
                })
                ticker_df = ticker_df.drop(columns=["ticker", "rv_daily", "n_bars"], errors="ignore")

            results.append(ticker_df)

        # Merge all on date
        if self.use_polars:
            merged = results[0]
            for i, df in enumerate(results[1:]):
                # Use suffix to avoid duplicate column names
                merged = merged.join(df, on="date", how="outer", suffix=f"_dup{i}")
            # Remove any duplicate date columns
            merged = merged.select([c for c in merged.columns if not c.startswith("date_dup")])
            return merged.sort("date")
        else:
            merged = results[0]
            for df in results[1:]:
                merged = merged.merge(df, on="date", how="outer")
            return merged.sort_values("date").reset_index(drop=True)

    def build_yahoo_features(self) -> Union["pl.DataFrame", pd.DataFrame]:
        """Build features from Yahoo Finance data.

        Returns:
            DataFrame with Yahoo features.
        """
        df = self.load_yahoo()

        # Get price columns (exclude date)
        if self.use_polars:
            price_cols = [c for c in df.columns if c != "date"]
            df = df.sort("date")
            df = self.add_factor_features_polars(df, columns=price_cols, prefix="yahoo_")
        else:
            price_cols = [c for c in df.columns if c != "date"]
            df = df.sort_values("date")
            df = self.add_factor_features_pandas(df, columns=price_cols, prefix="yahoo_")

        return df

    def build_macro_features(self) -> Union["pl.DataFrame", pd.DataFrame]:
        """Build features from macro data.

        Returns:
            DataFrame with macro features.
        """
        df = self.load_macro()

        if self.use_polars:
            macro_cols = [c for c in df.columns if c != "date"]
            df = df.sort("date")

            # For macro, add simpler features (most are already levels)
            for col in macro_cols:
                df = df.with_columns([
                    pl.col(col).shift(1).alias(f"{col}_lag1"),
                    (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_diff1"),
                    (pl.col(col) - pl.col(col).shift(22)).alias(f"{col}_diff22"),
                ])
        else:
            macro_cols = [c for c in df.columns if c != "date"]
            df = df.sort_values("date")

            for col in macro_cols:
                df[f"{col}_lag1"] = df[col].shift(1)
                df[f"{col}_diff1"] = df[col] - df[col].shift(1)
                df[f"{col}_diff22"] = df[col] - df[col].shift(22)

        return df

    def build_all_external_features(
        self,
        verbose: bool = True
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """Build all external features and merge.

        Args:
            verbose: Print progress.

        Returns:
            DataFrame with all external features merged on date.
        """
        if verbose:
            print("Building MOEX index features...")
        moex_df = self.build_moex_features()

        if verbose:
            print("Building Yahoo features...")
        yahoo_df = self.build_yahoo_features()

        if verbose:
            print("Building macro features...")
        macro_df = self.build_macro_features()

        # Merge all
        if verbose:
            print("Merging external features...")

        if self.use_polars:
            # Ensure consistent date types (cast all to date)
            moex_df = moex_df.with_columns(pl.col("date").cast(pl.Date))
            yahoo_df = yahoo_df.with_columns(pl.col("date").cast(pl.Date))
            macro_df = macro_df.with_columns(pl.col("date").cast(pl.Date))

            merged = moex_df.join(yahoo_df, on="date", how="outer", suffix="_y")
            merged = merged.join(macro_df, on="date", how="outer", suffix="_m")
            # Remove any duplicate date columns created by join
            merged = merged.select([c for c in merged.columns if c not in ["date_y", "date_m"]])
            merged = merged.sort("date")
        else:
            merged = moex_df.merge(yahoo_df, on="date", how="outer")
            merged = merged.merge(macro_df, on="date", how="outer")
            merged = merged.sort_values("date").reset_index(drop=True)

        if verbose:
            print(f"External features shape: {merged.shape}")

        return merged


def merge_stock_and_external_features(
    stock_features_path: str,
    external_features: Union["pl.DataFrame", pd.DataFrame],
    output_path: str,
    use_polars: bool = True,
    verbose: bool = True
) -> None:
    """Merge stock features with external features.

    Args:
        stock_features_path: Path to stock features parquet.
        external_features: DataFrame with external features.
        output_path: Path to save merged output.
        use_polars: Use polars.
        verbose: Print progress.
    """
    if verbose:
        print("Loading stock features...")

    if use_polars and POLARS_AVAILABLE:
        stock_df = pl.read_parquet(stock_features_path)
        merged = stock_df.join(external_features, on="date", how="left")
        merged = merged.sort(["ticker", "date"])
    else:
        stock_df = pd.read_parquet(stock_features_path)
        if isinstance(external_features, pl.DataFrame):
            external_features = external_features.to_pandas()
        merged = stock_df.merge(external_features, on="date", how="left")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if use_polars and POLARS_AVAILABLE:
        merged.write_parquet(output_path)
    else:
        merged.to_parquet(output_path, index=False)

    if verbose:
        print(f"Saved merged features to {output_path}")
        print(f"Final shape: {merged.shape}")


def build_all_features(
    dataset_path: str,
    stock_tickers: List[str],
    output_dir: str,
    use_polars: bool = True,
    verbose: bool = True
) -> str:
    """Build all features (stock + external) and save.

    Args:
        dataset_path: Path to dataset_final directory.
        stock_tickers: List of stock tickers.
        output_dir: Output directory.
        use_polars: Use polars.
        verbose: Print progress.

    Returns:
        Path to final features file.
    """
    from .intraday_features import build_stock_features

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build stock features
    stock_features_path = output_dir / "stock_features.parquet"
    candles_path = Path(dataset_path) / "01_stocks" / "candles_10m"

    if verbose:
        print("=" * 60)
        print("Step 1: Building stock features from 10-min candles")
        print("=" * 60)

    build_stock_features(
        candles_path=str(candles_path),
        tickers=stock_tickers,
        output_path=str(stock_features_path),
        use_polars=use_polars,
        verbose=verbose
    )

    # Step 2: Build external features
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: Building external features")
        print("=" * 60)

    builder = ExternalFeatureBuilder(dataset_path, use_polars=use_polars)
    external_df = builder.build_all_external_features(verbose=verbose)

    # Step 3: Merge
    final_path = output_dir / "features_all.parquet"

    if verbose:
        print("\n" + "=" * 60)
        print("Step 3: Merging stock and external features")
        print("=" * 60)

    merge_stock_and_external_features(
        stock_features_path=str(stock_features_path),
        external_features=external_df,
        output_path=str(final_path),
        use_polars=use_polars,
        verbose=verbose
    )

    return str(final_path)
