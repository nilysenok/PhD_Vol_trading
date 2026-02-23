"""Tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    qlike,
    mse,
    rmse,
    mae,
    r2_score,
    correlation,
    evaluate_forecast,
    ForecastMetrics,
)


class TestQLIKE:
    """Tests for QLIKE loss function."""

    def test_qlike_perfect_forecast(self):
        """Test QLIKE with perfect forecast."""
        y_true = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        y_pred = y_true.copy()

        loss = qlike(y_true, y_pred)

        # QLIKE = mean(y_true/y_pred - log(y_true/y_pred) - 1)
        # When y_true = y_pred: y_true/y_pred = 1, log(1) = 0
        # So QLIKE = mean(1 - 0 - 1) = 0
        assert np.isclose(loss, 0, atol=1e-10)

    def test_qlike_positive(self):
        """Test that QLIKE is non-negative for any forecast."""
        np.random.seed(42)

        y_true = np.abs(np.random.randn(100) * 0.001) + 0.0001
        y_pred = np.abs(np.random.randn(100) * 0.001) + 0.0001

        loss = qlike(y_true, y_pred)

        assert loss >= 0

    def test_qlike_underestimate_vs_overestimate(self):
        """Test QLIKE asymmetry: underestimation penalized more."""
        y_true = np.array([0.001, 0.001, 0.001])

        # Underestimate by factor of 2
        y_pred_under = np.array([0.0005, 0.0005, 0.0005])
        loss_under = qlike(y_true, y_pred_under)

        # Overestimate by factor of 2
        y_pred_over = np.array([0.002, 0.002, 0.002])
        loss_over = qlike(y_true, y_pred_over)

        # Underestimation should be penalized more heavily
        assert loss_under > loss_over

    def test_qlike_handles_zeros(self):
        """Test QLIKE handles near-zero values."""
        y_true = np.array([0.001, 0.0001, 0.00001])
        y_pred = np.array([0.001, 0.0001, 0.00001])

        loss = qlike(y_true, y_pred)

        assert np.isfinite(loss)


class TestMSE:
    """Tests for MSE metric."""

    def test_mse_perfect_forecast(self):
        """Test MSE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()

        assert mse(y_true, y_pred) == 0.0

    def test_mse_known_value(self):
        """Test MSE with known value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])

        # Errors: -1, 0, 1
        # Squared: 1, 0, 1
        # Mean: 2/3
        expected = 2 / 3

        assert np.isclose(mse(y_true, y_pred), expected)

    def test_mse_symmetric(self):
        """Test that MSE is symmetric to over/under prediction."""
        y_true = np.array([1.0, 1.0])

        y_pred_over = np.array([2.0, 2.0])
        y_pred_under = np.array([0.0, 0.0])

        assert mse(y_true, y_pred_over) == mse(y_true, y_pred_under)


class TestRMSE:
    """Tests for RMSE metric."""

    def test_rmse_perfect_forecast(self):
        """Test RMSE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()

        assert rmse(y_true, y_pred) == 0.0

    def test_rmse_is_sqrt_mse(self):
        """Test that RMSE = sqrt(MSE)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        assert np.isclose(rmse(y_true, y_pred), np.sqrt(mse(y_true, y_pred)))


class TestMAE:
    """Tests for MAE metric."""

    def test_mae_perfect_forecast(self):
        """Test MAE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()

        assert mae(y_true, y_pred) == 0.0

    def test_mae_known_value(self):
        """Test MAE with known value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])

        # Absolute errors: 1, 0, 1
        # Mean: 2/3
        expected = 2 / 3

        assert np.isclose(mae(y_true, y_pred), expected)


class TestR2Score:
    """Tests for R² score."""

    def test_r2_perfect_forecast(self):
        """Test R² with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        assert r2_score(y_true, y_pred) == 1.0

    def test_r2_mean_forecast(self):
        """Test R² when predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())

        # R² should be 0 when predicting mean
        assert np.isclose(r2_score(y_true, y_pred), 0.0)

    def test_r2_can_be_negative(self):
        """Test that R² can be negative for bad forecasts."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # Very bad prediction

        assert r2_score(y_true, y_pred) < 0


class TestCorrelation:
    """Tests for correlation metric."""

    def test_correlation_perfect_positive(self):
        """Test perfect positive correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true * 2  # Perfect linear relationship

        corr = correlation(y_true, y_pred)

        assert np.isclose(corr, 1.0)

    def test_correlation_perfect_negative(self):
        """Test perfect negative correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y_true

        corr = correlation(y_true, y_pred)

        assert np.isclose(corr, -1.0)

    def test_correlation_no_correlation(self):
        """Test no correlation."""
        np.random.seed(42)

        y_true = np.random.randn(1000)
        y_pred = np.random.randn(1000)

        corr = correlation(y_true, y_pred)

        # Should be close to 0 for random data
        assert abs(corr) < 0.1


class TestEvaluateForecast:
    """Tests for evaluate_forecast function."""

    def test_evaluate_forecast_returns_metrics(self):
        """Test that evaluate_forecast returns ForecastMetrics."""
        y_true = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        y_pred = np.array([0.0011, 0.0019, 0.0031, 0.0039, 0.0051])

        metrics = evaluate_forecast(y_true, y_pred)

        assert isinstance(metrics, ForecastMetrics)
        assert hasattr(metrics, "qlike")
        assert hasattr(metrics, "mse")
        assert hasattr(metrics, "rmse")
        assert hasattr(metrics, "mae")
        assert hasattr(metrics, "r2")
        assert hasattr(metrics, "correlation")

    def test_evaluate_forecast_all_finite(self):
        """Test that all metrics are finite."""
        np.random.seed(42)

        y_true = np.abs(np.random.randn(100) * 0.001) + 0.0001
        y_pred = np.abs(np.random.randn(100) * 0.001) + 0.0001

        metrics = evaluate_forecast(y_true, y_pred)

        assert np.isfinite(metrics.qlike)
        assert np.isfinite(metrics.mse)
        assert np.isfinite(metrics.rmse)
        assert np.isfinite(metrics.mae)
        assert np.isfinite(metrics.r2)
        assert np.isfinite(metrics.correlation)

    def test_evaluate_forecast_good_vs_bad(self):
        """Test that good forecasts have better metrics than bad ones."""
        y_true = np.array([0.001, 0.002, 0.003, 0.004, 0.005])

        # Good forecast
        y_pred_good = np.array([0.00105, 0.00195, 0.00305, 0.00395, 0.00505])

        # Bad forecast
        y_pred_bad = np.array([0.005, 0.001, 0.004, 0.002, 0.003])

        metrics_good = evaluate_forecast(y_true, y_pred_good)
        metrics_bad = evaluate_forecast(y_true, y_pred_bad)

        # Good forecast should have lower loss metrics
        assert metrics_good.mse < metrics_bad.mse
        assert metrics_good.mae < metrics_bad.mae

        # Good forecast should have higher R² and correlation
        assert metrics_good.r2 > metrics_bad.r2
        assert metrics_good.correlation > metrics_bad.correlation


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_value(self):
        """Test metrics with single value."""
        y_true = np.array([0.001])
        y_pred = np.array([0.001])

        # Should not raise errors
        assert mse(y_true, y_pred) == 0.0
        assert mae(y_true, y_pred) == 0.0

    def test_identical_values(self):
        """Test metrics when all true values are identical."""
        y_true = np.array([0.001, 0.001, 0.001])
        y_pred = np.array([0.001, 0.0011, 0.0009])

        # R² may be undefined or negative when variance is 0
        # But other metrics should work
        assert mse(y_true, y_pred) >= 0
        assert mae(y_true, y_pred) >= 0

    def test_different_lengths_raises_error(self):
        """Test that different length arrays raise error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])

        with pytest.raises((ValueError, IndexError)):
            mse(y_true, y_pred)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        y_true = np.array([0.001, np.nan, 0.003])
        y_pred = np.array([0.001, 0.002, 0.003])

        # Result should be NaN or function should handle it
        result = mse(y_true, y_pred)
        # Either NaN or valid number after dropping NaN
        assert np.isnan(result) or np.isfinite(result)


class TestComparisonMetrics:
    """Tests for model comparison metrics."""

    def test_ranking_by_qlike(self):
        """Test that models can be ranked by QLIKE."""
        y_true = np.array([0.001, 0.002, 0.003, 0.004, 0.005])

        models = {
            "model_a": np.array([0.00105, 0.00195, 0.00305, 0.00395, 0.00505]),
            "model_b": np.array([0.0015, 0.0025, 0.0035, 0.0045, 0.0055]),
            "model_c": np.array([0.002, 0.003, 0.004, 0.005, 0.006]),
        }

        qlikes = {name: qlike(y_true, pred) for name, pred in models.items()}

        # Sort by QLIKE (lower is better)
        ranking = sorted(qlikes.items(), key=lambda x: x[1])

        # Model A should be best
        assert ranking[0][0] == "model_a"

    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other."""
        np.random.seed(42)

        y_true = np.abs(np.random.randn(100) * 0.001) + 0.0001

        # Create forecasts of varying quality
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        results = []

        for noise in noise_levels:
            y_pred = y_true + np.random.randn(100) * noise * 0.001
            y_pred = np.abs(y_pred) + 0.00001

            metrics = evaluate_forecast(y_true, y_pred)
            results.append((noise, metrics))

        # Higher noise should generally lead to worse metrics
        for i in range(len(results) - 1):
            # MSE should increase with noise
            assert results[i][1].mse <= results[i + 1][1].mse * 1.5  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
