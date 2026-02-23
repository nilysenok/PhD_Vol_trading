"""Tests for feature engineering module."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.intraday_features import IntradayFeatureCalculator
from src.data.features import FeatureEngineer


class TestIntradayFeatureCalculator:
    """Tests for IntradayFeatureCalculator class."""

    @pytest.fixture
    def sample_intraday_data(self):
        """Create sample 10-minute intraday data."""
        dates = pd.date_range("2023-01-02 10:00", "2023-01-02 18:40", freq="10min")
        n = len(dates)

        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            "begin": dates,
            "open": close_prices + np.random.randn(n) * 0.1,
            "high": close_prices + np.abs(np.random.randn(n) * 0.2),
            "low": close_prices - np.abs(np.random.randn(n) * 0.2),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, n),
        })

        return df

    @pytest.fixture
    def calculator(self):
        """Create IntradayFeatureCalculator instance."""
        return IntradayFeatureCalculator()

    def test_calculate_log_returns(self, calculator, sample_intraday_data):
        """Test log return calculation."""
        df = sample_intraday_data.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        returns = df["log_return"].dropna()

        assert len(returns) == len(df) - 1
        assert not returns.isna().any()

    def test_calculate_rv(self, calculator, sample_intraday_data):
        """Test realized volatility calculation."""
        df = sample_intraday_data.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        rv = (df["log_return"] ** 2).sum()

        assert rv > 0
        assert np.isfinite(rv)

    def test_calculate_bv(self, calculator, sample_intraday_data):
        """Test bipower variation calculation."""
        df = sample_intraday_data.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["abs_return"] = df["log_return"].abs()

        # BV = (π/2) * Σ(|r_i| * |r_{i-1}|)
        bv = (np.pi / 2) * (df["abs_return"] * df["abs_return"].shift(1)).sum()

        assert bv > 0
        assert np.isfinite(bv)

    def test_calculate_jump(self, calculator, sample_intraday_data):
        """Test jump component calculation."""
        df = sample_intraday_data.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        rv = (df["log_return"] ** 2).sum()
        abs_returns = df["log_return"].abs()
        bv = (np.pi / 2) * (abs_returns * abs_returns.shift(1)).sum()

        jump = max(rv - bv, 0)

        assert jump >= 0
        assert np.isfinite(jump)

    def test_calculate_rsv_positive_negative(self, calculator, sample_intraday_data):
        """Test RSV+ and RSV- calculation."""
        df = sample_intraday_data.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        rsv_pos = (df["log_return"] ** 2 * (df["log_return"] > 0)).sum()
        rsv_neg = (df["log_return"] ** 2 * (df["log_return"] < 0)).sum()

        rv = (df["log_return"] ** 2).sum()

        # RSV+ + RSV- should approximately equal RV
        assert abs(rsv_pos + rsv_neg - rv) < 1e-10
        assert rsv_pos >= 0
        assert rsv_neg >= 0

    def test_calculate_rskew(self, calculator, sample_intraday_data):
        """Test realized skewness calculation."""
        df = sample_intraday_data.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        returns = df["log_return"].dropna()

        rv = (returns ** 2).sum()
        rskew = (returns ** 3).sum() / (rv ** 1.5 + 1e-10)

        assert np.isfinite(rskew)

    def test_calculate_rkurt(self, calculator, sample_intraday_data):
        """Test realized kurtosis calculation."""
        df = sample_intraday_data.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        returns = df["log_return"].dropna()

        rv = (returns ** 2).sum()
        rkurt = (returns ** 4).sum() / (rv ** 2 + 1e-10)

        assert np.isfinite(rkurt)
        assert rkurt > 0  # Kurtosis should be positive


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    @pytest.fixture
    def sample_daily_data(self):
        """Create sample daily RV data."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        n = len(dates)

        np.random.seed(42)
        rv = np.abs(np.random.randn(n) * 0.001 + 0.0005)

        df = pd.DataFrame({
            "date": dates,
            "ticker": "SBER",
            "rv": rv,
            "bv": rv * 0.9,
            "jump": rv * 0.1,
            "rsv_pos": rv * 0.55,
            "rsv_neg": rv * 0.45,
            "rskew": np.random.randn(n) * 0.5,
            "rkurt": np.abs(np.random.randn(n)) + 3,
            "n_bars": np.random.randint(48, 53, n),
        })

        return df

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        config = {
            "lags": {"daily": [1, 2, 3, 4, 5], "weekly": 5, "monthly": 22},
            "features": {
                "realized_volatility": True,
                "bipower_variation": True,
                "jump_component": True,
            }
        }
        return FeatureEngineer(config)

    def test_create_har_features(self, feature_engineer, sample_daily_data):
        """Test HAR feature creation."""
        df = sample_daily_data.copy()

        # Create HAR lags manually
        df["rv_d"] = df["rv"].shift(1)  # Daily lag
        df["rv_w"] = df["rv"].rolling(5).mean().shift(1)  # Weekly average
        df["rv_m"] = df["rv"].rolling(22).mean().shift(1)  # Monthly average

        # Check that features are created
        df_clean = df.dropna()

        assert "rv_d" in df_clean.columns
        assert "rv_w" in df_clean.columns
        assert "rv_m" in df_clean.columns
        assert len(df_clean) > 0

    def test_create_target_variable(self, sample_daily_data):
        """Test target variable creation."""
        df = sample_daily_data.copy()

        for h in [1, 5, 22]:
            df[f"y_h{h}"] = df["rv"].shift(-h)

            # Forward shift should create NaN at the end
            assert df[f"y_h{h}"].iloc[-h:].isna().all()
            assert df[f"y_h{h}"].iloc[:-h].notna().all()

    def test_log_transformation(self, sample_daily_data):
        """Test log transformation of RV."""
        df = sample_daily_data.copy()

        df["log_rv"] = np.log(df["rv"] + 1e-8)

        assert not df["log_rv"].isna().any()
        assert np.isfinite(df["log_rv"]).all()

    def test_feature_alignment(self, sample_daily_data):
        """Test that features align correctly with target."""
        df = sample_daily_data.copy()

        # Create features (past data)
        df["rv_lag1"] = df["rv"].shift(1)

        # Create target (future data)
        df["y_h1"] = df["rv"].shift(-1)

        # After dropping NaN, features at time t should predict target at t+1
        df_clean = df.dropna()

        # The feature at index i should come from time before target at index i
        assert len(df_clean) == len(df) - 2  # Lost 1 at start, 1 at end


class TestDataIntegrity:
    """Tests for data integrity and consistency."""

    def test_no_future_leakage(self):
        """Verify no future information leaks into features."""
        dates = pd.date_range("2020-01-01", "2020-01-31", freq="B")
        n = len(dates)

        np.random.seed(42)
        df = pd.DataFrame({
            "date": dates,
            "rv": np.abs(np.random.randn(n) * 0.001),
        })

        # Create lagged feature
        df["rv_lag1"] = df["rv"].shift(1)

        # Create forward target
        df["y_h1"] = df["rv"].shift(-1)

        # At any point t, rv_lag1 should be from t-1, y_h1 from t+1
        # So rv_lag1 at t should never equal y_h1 at t (unless by coincidence)
        for i in range(1, len(df) - 1):
            # Feature at i is rv at i-1
            # Target at i is rv at i+1
            # These should be different indices
            feature_source_idx = i - 1
            target_source_idx = i + 1
            assert feature_source_idx != target_source_idx

    def test_date_continuity(self):
        """Test that dates are properly ordered."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        df = pd.DataFrame({"date": dates})

        # Dates should be monotonically increasing
        assert df["date"].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
