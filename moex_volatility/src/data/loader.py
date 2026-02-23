"""Data loading utilities for MOEX volatility dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union


class DataLoader:
    """Loader for MOEX volatility dataset."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize data loader.

        Args:
            base_path: Base path to dataset_final folder.
                       If None, uses default relative path.
        """
        if base_path is None:
            # Default path relative to project root
            self.base_path = Path(__file__).parent.parent.parent.parent / "moex_discovery" / "data" / "dataset_final"
        else:
            self.base_path = Path(base_path)

        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate that required data files exist."""
        required = [
            "03_master/master_long.parquet",
            "03_master/master_wide.parquet",
            "01_stocks/rv_daily.parquet",
        ]
        for rel_path in required:
            full_path = self.base_path / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"Required file not found: {full_path}")

    def load_master_long(self) -> pd.DataFrame:
        """Load master dataset in long format.

        Returns:
            DataFrame with columns: date, ticker, rv_daily, rv_annualized, ...
        """
        path = self.base_path / "03_master" / "master_long.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_master_wide(self) -> pd.DataFrame:
        """Load master dataset in wide format.

        Returns:
            DataFrame with date index and columns for each variable.
        """
        path = self.base_path / "03_master" / "master_wide.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_stock_rv(self) -> pd.DataFrame:
        """Load stock RV data.

        Returns:
            DataFrame with stock RV in long format.
        """
        path = self.base_path / "01_stocks" / "rv_daily.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_yahoo(self) -> pd.DataFrame:
        """Load Yahoo Finance external data.

        Returns:
            DataFrame with global market data.
        """
        path = self.base_path / "02_external" / "yahoo.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_macro(self) -> pd.DataFrame:
        """Load macroeconomic data.

        Returns:
            DataFrame with macro indicators.
        """
        path = self.base_path / "02_external" / "macro.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_moex_indices_rv(self, wide: bool = True) -> pd.DataFrame:
        """Load MOEX indices RV data.

        Args:
            wide: If True, return wide format. Otherwise long format.

        Returns:
            DataFrame with MOEX index RV.
        """
        filename = "rv_moex_wide.parquet" if wide else "rv_moex.parquet"
        path = self.base_path / "02_external" / filename
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_candles_10m(
        self,
        ticker: str,
        source: str = "stocks"
    ) -> pd.DataFrame:
        """Load 10-minute candles for a specific ticker.

        Args:
            ticker: Ticker symbol (e.g., 'SBER', 'IMOEX').
            source: 'stocks' or 'external'.

        Returns:
            DataFrame with OHLCV candles.
        """
        folder = "01_stocks" if source == "stocks" else "02_external"
        path = self.base_path / folder / "candles_10m" / f"{ticker}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Candles file not found: {path}")

        df = pd.read_parquet(path)
        if "begin" in df.columns:
            df["begin"] = pd.to_datetime(df["begin"])
        if "end" in df.columns:
            df["end"] = pd.to_datetime(df["end"])
        return df

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        """Load all data for a single ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            DataFrame with all features for the ticker.
        """
        df = self.load_master_long()
        return df[df["ticker"] == ticker].copy()

    def get_tickers(self) -> List[str]:
        """Get list of available stock tickers.

        Returns:
            List of ticker symbols.
        """
        df = self.load_stock_rv()
        return sorted(df["ticker"].unique().tolist())

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get date range of the dataset.

        Returns:
            Tuple of (start_date, end_date).
        """
        path = self.base_path / "03_master" / "dates.csv"
        dates = pd.read_csv(path)
        dates["date"] = pd.to_datetime(dates["date"])
        return dates["date"].min(), dates["date"].max()

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        date_col: str = "date"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets by date.

        Args:
            df: Input DataFrame.
            test_ratio: Fraction of data for test set.
            date_col: Name of date column.

        Returns:
            Tuple of (train_df, test_df).
        """
        dates = df[date_col].unique()
        dates = np.sort(dates)

        split_idx = int(len(dates) * (1 - test_ratio))
        split_date = dates[split_idx]

        train = df[df[date_col] < split_date].copy()
        test = df[df[date_col] >= split_date].copy()

        return train, test
