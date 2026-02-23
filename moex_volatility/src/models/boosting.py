"""Gradient Boosting models for volatility forecasting."""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    # Test if it actually works
    _lgb_test = lgb.LGBMRegressor()
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError, Exception):
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    # Test if it actually works
    _xgb_test = xgb.XGBRegressor()
    XGBOOST_AVAILABLE = True
except (ImportError, OSError, Exception):
    XGBOOST_AVAILABLE = False

# Fallback to sklearn
try:
    from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LightGBMModel:
    """LightGBM model for volatility forecasting."""

    def __init__(self, horizon: int = 1, params: Optional[Dict[str, Any]] = None):
        """Initialize LightGBM model.

        Args:
            horizon: Forecast horizon.
            params: Model hyperparameters.
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is not available. Install with: pip install lightgbm")

        self.horizon = horizon
        self.params = params or {}
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> "LightGBMModel":
        """Fit LightGBM model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features for early stopping.
            y_val: Validation targets for early stopping.
            feature_names: Feature names for importance.

        Returns:
            Self.
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        # Convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Replace inf with nan, then fill nan
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_train = np.nan_to_num(X_train, nan=0.0)

        # Default params
        default_params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1
        }

        # Merge with user params
        model_params = {**default_params, **self.params}

        # Create model
        self.model = lgb.LGBMRegressor(**model_params)

        # Fit with or without validation
        if X_val is not None and y_val is not None:
            X_val = np.where(np.isinf(X_val), np.nan, X_val)
            X_val = np.nan_to_num(X_val, nan=0.0)

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict volatility.

        Args:
            X: Features.

        Returns:
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Handle inf/nan
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)

        predictions = self.model.predict(X)
        # Ensure non-negative predictions for volatility
        predictions = np.clip(predictions, 1e-10, None)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance.

        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names,
                'horizon': self.horizon
            }, f)

    def load(self, path: str) -> "LightGBMModel":
        """Load model from disk.

        Args:
            path: File path.

        Returns:
            Self.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.params = data['params']
            self.feature_names = data.get('feature_names', [])
            self.horizon = data.get('horizon', 1)
            self.is_fitted = True
        return self


class XGBoostModel:
    """XGBoost model for volatility forecasting."""

    def __init__(self, horizon: int = 1, params: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model.

        Args:
            horizon: Forecast horizon.
            params: Model hyperparameters.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not available. Install with: pip install xgboost")

        self.horizon = horizon
        self.params = params or {}
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> "XGBoostModel":
        """Fit XGBoost model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features for early stopping.
            y_val: Validation targets for early stopping.
            feature_names: Feature names for importance.

        Returns:
            Self.
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        # Convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Replace inf with nan, then fill nan
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_train = np.nan_to_num(X_train, nan=0.0)

        # Default params
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0,
            'random_state': 42,
            'n_jobs': -1
        }

        # Merge with user params
        model_params = {**default_params, **self.params}

        # Create model
        self.model = xgb.XGBRegressor(**model_params)

        # Fit with or without validation
        if X_val is not None and y_val is not None:
            X_val = np.where(np.isinf(X_val), np.nan, X_val)
            X_val = np.nan_to_num(X_val, nan=0.0)

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict volatility.

        Args:
            X: Features.

        Returns:
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Handle inf/nan
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)

        predictions = self.model.predict(X)
        # Ensure non-negative predictions for volatility
        predictions = np.clip(predictions, 1e-10, None)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance.

        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names,
                'horizon': self.horizon
            }, f)

    def load(self, path: str) -> "XGBoostModel":
        """Load model from disk.

        Args:
            path: File path.

        Returns:
            Self.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.params = data['params']
            self.feature_names = data.get('feature_names', [])
            self.horizon = data.get('horizon', 1)
            self.is_fitted = True
        return self


class SklearnBoostingModel:
    """Sklearn-based gradient boosting model (fallback when LightGBM/XGBoost unavailable)."""

    def __init__(self, horizon: int = 1, params: Optional[Dict[str, Any]] = None, model_type: str = 'hist'):
        """Initialize sklearn boosting model.

        Args:
            horizon: Forecast horizon.
            params: Model hyperparameters.
            model_type: 'hist' for HistGradientBoostingRegressor, 'gbr' for GradientBoostingRegressor.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        self.horizon = horizon
        self.params = params or {}
        self.model_type = model_type
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> "SklearnBoostingModel":
        """Fit model."""
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        # Convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Ensure float64 and handle inf/nan
        X_train = np.asarray(X_train, dtype=np.float64)
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_train = np.nan_to_num(X_train, nan=0.0)

        # Default params
        if self.model_type == 'hist':
            default_params = {
                'max_iter': 500,
                'max_depth': 8,
                'learning_rate': 0.05,
                'min_samples_leaf': 20,
                'l2_regularization': 0.1,
                'random_state': 42
            }
            model_params = {k: v for k, v in {**default_params, **self.params}.items()
                          if k in ['max_iter', 'max_depth', 'learning_rate', 'min_samples_leaf',
                                  'l2_regularization', 'max_leaf_nodes', 'random_state', 'validation_fraction',
                                  'n_iter_no_change', 'tol']}
            self.model = HistGradientBoostingRegressor(**model_params)
        else:
            default_params = {
                'n_estimators': 500,
                'max_depth': 5,
                'learning_rate': 0.05,
                'min_samples_leaf': 20,
                'subsample': 0.8,
                'random_state': 42
            }
            model_params = {k: v for k, v in {**default_params, **self.params}.items()
                          if k in ['n_estimators', 'max_depth', 'learning_rate', 'min_samples_leaf',
                                  'subsample', 'random_state', 'min_samples_split', 'max_features']}
            self.model = GradientBoostingRegressor(**model_params)

        # Fit
        if X_val is not None and y_val is not None and self.model_type == 'hist':
            # Use early stopping for hist
            X_val_arr = np.asarray(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, dtype=np.float64)
            X_val_arr = np.where(np.isinf(X_val_arr), np.nan, X_val_arr)
            X_val_arr = np.nan_to_num(X_val_arr, nan=0.0)
            y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
            # Combine for validation split
            X_combined = np.vstack([X_train, X_val_arr])
            y_combined = np.concatenate([y_train, y_val_arr])
            val_fraction = len(X_val_arr) / len(X_combined)
            self.model.set_params(validation_fraction=val_fraction, n_iter_no_change=10)
            self.model.fit(X_combined, y_combined)
        else:
            self.model.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict volatility."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)

        predictions = self.model.predict(X)
        predictions = np.clip(predictions, 1e-10, None)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_names))

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names,
                'horizon': self.horizon,
                'model_type': self.model_type
            }, f)

    def load(self, path: str) -> "SklearnBoostingModel":
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.params = data['params']
            self.feature_names = data.get('feature_names', [])
            self.horizon = data.get('horizon', 1)
            self.model_type = data.get('model_type', 'hist')
            self.is_fitted = True
        return self
