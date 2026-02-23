"""XGBoost model for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings
from .base import BaseVolatilityModel

# Conditional import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost package not installed. XGBoost models will not be available.")


class XGBoostModel(BaseVolatilityModel):
    """XGBoost model for volatility forecasting.

    XGBoost is an optimized gradient boosting library designed for
    speed and performance. It supports GPU acceleration.
    """

    def __init__(
        self,
        objective: str = "reg:squarederror",
        eval_metric: str = "rmse",
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: Optional[int] = 50,
        verbosity: int = 0,
        random_state: int = 42,
        tree_method: str = "auto",
        **kwargs
    ):
        """Initialize XGBoost model.

        Args:
            objective: Objective function.
            eval_metric: Evaluation metric.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate (eta).
            n_estimators: Number of boosting rounds.
            subsample: Row subsampling ratio.
            colsample_bytree: Column subsampling ratio.
            min_child_weight: Minimum sum of instance weight in a child.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            early_stopping_rounds: Early stopping rounds.
            verbosity: Verbosity level.
            random_state: Random seed.
            tree_method: Tree construction algorithm (auto, exact, approx, hist, gpu_hist).
            **kwargs: Additional XGBoost parameters.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost package is required. Install with: pip install xgboost")

        super().__init__(name="XGBoost", **kwargs)

        self.xgb_params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "verbosity": verbosity,
            "random_state": random_state,
            "tree_method": tree_method,
        }
        self.xgb_params.update(kwargs)

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
    ) -> "XGBoostModel":
        """Fit XGBoost model.

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

        # Convert to numpy if needed
        if isinstance(y, pd.Series):
            y = y.values

        # Prepare eval set
        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            eval_set.append((X_val, y_val))

        # Create model
        self._model = xgb.XGBRegressor(**self.xgb_params)

        # Fit
        fit_params = {
            "eval_set": eval_set,
            "verbose": False,
        }
        if self.early_stopping_rounds and X_val is not None:
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self._model.fit(X, y, **fit_params, **kwargs)

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

        return self._model.predict(X)

    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> pd.Series:
        """Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover').

        Returns:
            Series with feature importance.
        """
        if not self.is_fitted:
            return pd.Series()

        importance = self._model.get_booster().get_score(importance_type=importance_type)

        # Fill missing features with 0
        names = self._feature_names or [f"f{i}" for i in range(self._model.n_features_in_)]
        importance_series = pd.Series(0.0, index=names)

        for key, value in importance.items():
            if key in importance_series.index:
                importance_series[key] = value
            elif key.startswith("f"):
                # Handle numeric feature names
                idx = int(key[1:])
                if idx < len(names):
                    importance_series[names[idx]] = value

        return importance_series.sort_values(ascending=False)

    def get_best_iteration(self) -> Optional[int]:
        """Get best iteration from early stopping.

        Returns:
            Best iteration number.
        """
        return self._best_iteration

    def save_model(self, path: str) -> None:
        """Save XGBoost model in native format.

        Args:
            path: Path to save model.
        """
        if self._model is not None:
            self._model.save_model(path)

    def load_model(self, path: str) -> "XGBoostModel":
        """Load XGBoost model from native format.

        Args:
            path: Path to saved model.

        Returns:
            Self.
        """
        self._model = xgb.XGBRegressor()
        self._model.load_model(path)
        self.is_fitted = True
        return self


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Optional[Dict[str, List]] = None,
    n_trials: int = 50,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """Tune XGBoost hyperparameters using random search.

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
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5, 10],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [0.1, 1.0, 10.0],
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
        model = XGBoostModel(**params, n_estimators=1000, early_stopping_rounds=50)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Evaluate
        y_pred = model.predict(X_val)
        score = np.sqrt(np.mean((y_val - y_pred) ** 2))

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score
