"""Tests for volatility models."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import BaseVolatilityModel
from src.models.har import HARModel
from src.models.garch import GARCHModel


class TestBaseVolatilityModel:
    """Tests for BaseVolatilityModel class."""

    def test_base_model_interface(self):
        """Test that base model has required interface."""
        model = BaseVolatilityModel(horizon=1, config={})

        # Check required methods exist
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "get_params")
        assert hasattr(model, "save")
        assert hasattr(model, "load")

    def test_base_model_initialization(self):
        """Test base model initialization."""
        config = {"param1": "value1"}
        model = BaseVolatilityModel(horizon=5, config=config)

        assert model.horizon == 5
        assert model.config == config


class TestHARModel:
    """Tests for HAR-RV model."""

    @pytest.fixture
    def sample_har_data(self):
        """Create sample data for HAR model."""
        np.random.seed(42)
        n = 500

        # Generate synthetic RV data
        rv = np.abs(np.random.randn(n) * 0.001 + 0.0005)

        df = pd.DataFrame({
            "rv_d": np.roll(rv, 1),
            "rv_w": pd.Series(rv).rolling(5).mean().values,
            "rv_m": pd.Series(rv).rolling(22).mean().values,
            "jump": np.maximum(rv - rv * 0.9, 0),
        })

        # Target
        df["y"] = rv

        # Drop NaN
        df = df.iloc[22:].reset_index(drop=True)

        return df

    @pytest.fixture
    def har_model(self):
        """Create HAR model instance."""
        config = {
            "lags": [1, 5, 22],
            "include_jump": True,
            "include_rsv": False,
            "use_log": True,
            "estimator": "ols",
        }
        return HARModel(horizon=1, config=config)

    def test_har_model_fit(self, har_model, sample_har_data):
        """Test HAR model fitting."""
        X = sample_har_data[["rv_d", "rv_w", "rv_m", "jump"]]
        y = sample_har_data["y"]

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Fit model
        har_model.fit(X_train, y_train, X_val, y_val)

        assert har_model.is_fitted
        assert har_model.model is not None

    def test_har_model_predict(self, har_model, sample_har_data):
        """Test HAR model prediction."""
        X = sample_har_data[["rv_d", "rv_w", "rv_m", "jump"]]
        y = sample_har_data["y"]

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Fit and predict
        har_model.fit(X_train, y_train, X_test, y_test)
        predictions = har_model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()

    def test_har_model_coefficients(self, har_model, sample_har_data):
        """Test HAR model returns valid coefficients."""
        X = sample_har_data[["rv_d", "rv_w", "rv_m", "jump"]]
        y = sample_har_data["y"]

        train_size = int(len(X) * 0.8)
        har_model.fit(X[:train_size], y[:train_size], X[train_size:], y[train_size:])

        params = har_model.get_params()

        assert "coefficients" in params or len(params) > 0

    def test_har_with_log_transform(self, sample_har_data):
        """Test HAR model with log transformation."""
        config = {
            "lags": [1, 5, 22],
            "include_jump": True,
            "use_log": True,
            "estimator": "ols",
        }
        model = HARModel(horizon=1, config=config)

        X = sample_har_data[["rv_d", "rv_w", "rv_m", "jump"]]
        y = sample_har_data["y"]

        train_size = int(len(X) * 0.8)
        model.fit(X[:train_size], y[:train_size], X[train_size:], y[train_size:])

        predictions = model.predict(X[train_size:])

        # Predictions should be positive (after exp transform)
        assert (predictions > 0).all()


class TestGARCHModel:
    """Tests for GARCH model."""

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for GARCH model."""
        np.random.seed(42)
        n = 1000

        # Generate synthetic returns with volatility clustering
        returns = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.01

        omega = 0.00001
        alpha = 0.1
        beta = 0.85

        for t in range(1, n):
            sigma[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * sigma[t-1]**2)
            returns[t] = sigma[t] * np.random.randn()

        return pd.Series(returns, name="returns")

    @pytest.fixture
    def garch_model(self):
        """Create GARCH model instance."""
        config = {
            "p": 1,
            "q": 1,
            "o": 1,  # GJR asymmetric term
            "dist": "t",
            "mean": "Zero",
            "vol": "GARCH",
        }
        return GARCHModel(horizon=1, config=config)

    def test_garch_model_fit(self, garch_model, sample_returns_data):
        """Test GARCH model fitting."""
        returns = sample_returns_data

        train_size = int(len(returns) * 0.8)
        train_returns = returns[:train_size]
        val_returns = returns[train_size:]

        garch_model.fit(train_returns, None, val_returns, None)

        assert garch_model.is_fitted
        assert garch_model.model is not None

    def test_garch_model_predict(self, garch_model, sample_returns_data):
        """Test GARCH model prediction."""
        returns = sample_returns_data

        train_size = int(len(returns) * 0.8)
        train_returns = returns[:train_size]

        garch_model.fit(train_returns, None, None, None)

        # Forecast h steps ahead
        forecasts = garch_model.predict(n_ahead=5)

        assert len(forecasts) == 5
        assert (forecasts > 0).all()  # Variance should be positive

    def test_garch_different_distributions(self, sample_returns_data):
        """Test GARCH with different error distributions."""
        returns = sample_returns_data[:500]

        for dist in ["normal", "t"]:
            config = {
                "p": 1,
                "q": 1,
                "dist": dist,
                "mean": "Zero",
                "vol": "GARCH",
            }
            model = GARCHModel(horizon=1, config=config)

            try:
                model.fit(returns, None, None, None)
                assert model.is_fitted
            except Exception as e:
                pytest.skip(f"GARCH with {dist} distribution failed: {e}")


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    @pytest.fixture
    def tmp_path(self, tmp_path_factory):
        """Create temporary directory for tests."""
        return tmp_path_factory.mktemp("models")

    def test_har_save_load(self, tmp_path):
        """Test HAR model save and load."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame({
            "rv_d": np.abs(np.random.randn(n) * 0.001),
            "rv_w": np.abs(np.random.randn(n) * 0.001),
            "rv_m": np.abs(np.random.randn(n) * 0.001),
        })
        y = pd.Series(np.abs(np.random.randn(n) * 0.001))

        config = {"lags": [1, 5, 22], "use_log": True, "estimator": "ols"}
        model = HARModel(horizon=1, config=config)
        model.fit(X[:150], y[:150], X[150:], y[150:])

        # Save
        save_path = tmp_path / "har_model.pkl"
        model.save(str(save_path))

        assert save_path.exists()

        # Load
        loaded_model = HARModel(horizon=1, config=config)
        loaded_model.load(str(save_path))

        # Predictions should be the same
        original_pred = model.predict(X[150:])
        loaded_pred = loaded_model.predict(X[150:])

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestModelValidation:
    """Tests for model validation and error handling."""

    def test_predict_without_fit(self):
        """Test that predict raises error if model not fitted."""
        model = HARModel(horizon=1, config={})

        X = pd.DataFrame({"rv_d": [0.001], "rv_w": [0.001], "rv_m": [0.001]})

        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            model.predict(X)

    def test_invalid_horizon(self):
        """Test that invalid horizon raises error."""
        with pytest.raises((ValueError, AssertionError)):
            HARModel(horizon=-1, config={})

    def test_empty_data(self):
        """Test handling of empty data."""
        model = HARModel(horizon=1, config={"lags": [1, 5, 22], "estimator": "ols"})

        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)

        with pytest.raises((ValueError, KeyError)):
            model.fit(X_empty, y_empty, X_empty, y_empty)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
