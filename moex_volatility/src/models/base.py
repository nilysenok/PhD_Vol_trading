"""Base model class for volatility forecasting."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json


class BaseVolatilityModel(ABC):
    """Abstract base class for volatility forecasting models."""

    def __init__(self, name: str, **kwargs):
        """Initialize base model.

        Args:
            name: Model name.
            **kwargs: Additional model parameters.
        """
        self.name = name
        self.params = kwargs
        self.is_fitted = False
        self._feature_names: List[str] = []
        self._training_history: Dict[str, List] = {}

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> "BaseVolatilityModel":
        """Fit the model.

        Args:
            X: Feature matrix.
            y: Target values.
            **kwargs: Additional fitting parameters.

        Returns:
            Self.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        pass

    def fit_predict(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Fit model and predict on test set.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            **kwargs: Additional fitting parameters.

        Returns:
            Predictions on test set.
        """
        self.fit(X_train, y_train, **kwargs)
        return self.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameters.
        """
        return self.params.copy()

    def set_params(self, **params) -> "BaseVolatilityModel":
        """Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            Self.
        """
        self.params.update(params)
        return self

    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return self._feature_names

    @feature_names.setter
    def feature_names(self, names: List[str]) -> None:
        """Set feature names."""
        self._feature_names = list(names)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance (if available).

        Returns:
            Series with feature importance, or None.
        """
        return None

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(path, "wb") as f:
            pickle.dump(self, f)

        # Save metadata
        metadata = {
            "name": self.name,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "feature_names": self._feature_names,
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "BaseVolatilityModel":
        """Load model from disk.

        Args:
            path: Path to saved model.

        Returns:
            Loaded model instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"


class TimeSeriesModel(BaseVolatilityModel):
    """Base class for univariate time series models (HAR, GARCH)."""

    def __init__(self, name: str, use_log: bool = True, **kwargs):
        """Initialize time series model.

        Args:
            name: Model name.
            use_log: Whether to use log transformation.
            **kwargs: Additional parameters.
        """
        super().__init__(name, **kwargs)
        self.use_log = use_log

    def _transform(self, y: np.ndarray) -> np.ndarray:
        """Transform target variable.

        Args:
            y: Target values.

        Returns:
            Transformed values.
        """
        if self.use_log:
            return np.log(np.maximum(y, 1e-10))
        return y

    def _inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform predictions.

        Args:
            y: Transformed values.

        Returns:
            Original scale values.
        """
        if self.use_log:
            return np.exp(y)
        return y


class EnsembleModel(BaseVolatilityModel):
    """Base class for ensemble models."""

    def __init__(
        self,
        name: str,
        models: List[BaseVolatilityModel],
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        """Initialize ensemble model.

        Args:
            name: Model name.
            models: List of base models.
            weights: Weights for each model. If None, equal weights.
            **kwargs: Additional parameters.
        """
        super().__init__(name, **kwargs)
        self.models = models

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> "EnsembleModel":
        """Fit all models in the ensemble.

        Args:
            X: Feature matrix.
            y: Target values.
            **kwargs: Additional fitting parameters.

        Returns:
            Self.
        """
        for model in self.models:
            model.fit(X, y, **kwargs)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make weighted average predictions.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average predictions.
        """
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)

    def predict_all(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Get predictions from all models.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary mapping model names to predictions.
        """
        return {
            model.name: model.predict(X)
            for model in self.models
        }
