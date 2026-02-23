"""LightGBM model for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings
from .base import BaseVolatilityModel

# Conditional import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("lightgbm package not installed. LightGBM models will not be available.")


class LightGBMModel(BaseVolatilityModel):
    """LightGBM model for volatility forecasting.

    LightGBM is a gradient boosting framework that uses tree-based learning.
    It's particularly efficient for large datasets and supports GPU acceleration.
    """

    def __init__(
        self,
        objective: str = "regression",
        metric: str = "rmse",
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        n_estimators: int = 1000,
        early_stopping_rounds: Optional[int] = 50,
        verbose: int = -1,
        random_state: int = 42,
        device: str = "cpu",
        **kwargs
    ):
        """Initialize LightGBM model.

        Args:
            objective: Objective function.
            metric: Evaluation metric.
            boosting_type: Boosting type (gbdt, dart, goss).
            num_leaves: Maximum number of leaves.
            learning_rate: Learning rate.
            feature_fraction: Fraction of features for each tree.
            bagging_fraction: Fraction of data for bagging.
            bagging_freq: Frequency of bagging.
            n_estimators: Number of boosting iterations.
            early_stopping_rounds: Early stopping rounds.
            verbose: Verbosity level.
            random_state: Random seed.
            device: Device to use (cpu, gpu).
            **kwargs: Additional LightGBM parameters.
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm package is required. Install with: pip install lightgbm")

        super().__init__(name="LightGBM", **kwargs)

        self.lgb_params = {
            "objective": objective,
            "metric": metric,
            "boosting_type": boosting_type,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "n_estimators": n_estimators,
            "verbose": verbose,
            "random_state": random_state,
            "device": device,
        }
        self.lgb_params.update(kwargs)

        self.early_stopping_rounds = early_stopping_rounds
        self._model = None
        self._best_iteration = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> "LightGBMModel":
        """Fit LightGBM model.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features (for early stopping).
            y_val: Validation target.
            **kwargs: Additional fitting parameters.

        Returns:
            Self.
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Create datasets
        train_data = lgb.Dataset(X, label=y)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Prepare callbacks
        callbacks = []
        if self.early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        # Train
        self._model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            **kwargs
        )

        self._best_iteration = self._model.best_iteration
        self.is_fitted = True

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._model.predict(X, num_iteration=self._best_iteration)

    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> pd.Series:
        """Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'split').

        Returns:
            Series with feature importance.
        """
        if not self.is_fitted:
            return pd.Series()

        importance = self._model.feature_importance(importance_type=importance_type)
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]

        return pd.Series(importance, index=names).sort_values(ascending=False)

    def get_best_iteration(self) -> Optional[int]:
        """Get best iteration from early stopping.

        Returns:
            Best iteration number.
        """
        return self._best_iteration

    def save_model(self, path: str) -> None:
        """Save LightGBM model in native format.

        Args:
            path: Path to save model.
        """
        if self._model is not None:
            self._model.save_model(path)

    def load_model(self, path: str) -> "LightGBMModel":
        """Load LightGBM model from native format.

        Args:
            path: Path to saved model.

        Returns:
            Self.
        """
        self._model = lgb.Booster(model_file=path)
        self.is_fitted = True
        return self


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Optional[Dict[str, List]] = None,
    n_trials: int = 50,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """Tune LightGBM hyperparameters using simple random search.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        param_grid: Parameter grid for search.
        n_trials: Number of trials.
        random_state: Random seed.

    Returns:
        Tuple of (best_params, best_score).
    """
    if param_grid is None:
        param_grid = {
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "feature_fraction": [0.6, 0.7, 0.8, 0.9],
            "bagging_fraction": [0.6, 0.7, 0.8, 0.9],
            "min_child_samples": [10, 20, 50, 100],
        }

    np.random.seed(random_state)

    best_score = float("inf")
    best_params = {}

    for _ in range(n_trials):
        # Sample parameters
        params = {
            key: np.random.choice(values)
            for key, values in param_grid.items()
        }

        # Train model
        model = LightGBMModel(**params, n_estimators=1000, early_stopping_rounds=50)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Evaluate
        y_pred = model.predict(X_val)
        score = np.sqrt(np.mean((y_val - y_pred) ** 2))

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score
