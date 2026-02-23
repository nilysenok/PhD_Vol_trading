"""Hybrid models combining multiple approaches for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Literal
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle
from .base import BaseVolatilityModel, EnsembleModel

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class HybridHARML:
    """
    Гибридная модель: HAR-RV + ML на остатках

    Идея:
    - HAR хорошо ловит авторегрессионную структуру RV (персистентность)
    - ML ловит влияние внешних факторов на остатки HAR

    RV_final = exp(log_HAR_pred + ML_pred(residuals))

    Uses log-transform for stability (standard in HAR literature).
    """

    def __init__(self, horizon: int, har_features: List[str], external_features: List[str],
                 use_log: bool = True):
        """
        har_features: список фичей для HAR (rv_d, rv_w, rv_m и тд)
        external_features: список внешних фичей для ML (VIX, Brent, ставки, индексы...)
        use_log: использовать log-преобразование (рекомендуется)
        """
        self.horizon = horizon
        self.har_features = har_features
        self.external_features = external_features
        self.use_log = use_log
        self.har_model = LinearRegression()
        self.ml_model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None):
        """
        X: DataFrame с har_features и external_features
        y: target (RV)
        """
        # Clip values to avoid log(0)
        y_safe = np.clip(y, 1e-10, None)

        if self.use_log:
            y_train = np.log(y_safe)
            X_har = np.log(np.clip(X[self.har_features].values, 1e-10, None))
        else:
            y_train = y_safe
            X_har = X[self.har_features].values

        # Step 1: Fit HAR on log-RV
        self.har_model.fit(X_har, y_train)
        har_pred = self.har_model.predict(X_har)

        # Step 2: Calculate residuals (in log space if use_log)
        residuals = y_train - har_pred

        # Step 3: Fit ML on residuals using external features
        X_ext = X[self.external_features].values
        # Replace inf/nan with 0
        X_ext = np.nan_to_num(X_ext, nan=0.0, posinf=0.0, neginf=0.0)
        self.ml_model.fit(X_ext, residuals)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict: exp(HAR + ML(residuals)) if use_log else HAR + ML"""
        if self.use_log:
            X_har = np.log(np.clip(X[self.har_features].values, 1e-10, None))
        else:
            X_har = X[self.har_features].values

        X_ext = X[self.external_features].values
        X_ext = np.nan_to_num(X_ext, nan=0.0, posinf=0.0, neginf=0.0)

        har_pred = self.har_model.predict(X_har)
        ml_pred = self.ml_model.predict(X_ext)

        combined = har_pred + ml_pred

        if self.use_log:
            # Transform back from log space
            return np.exp(combined)
        else:
            return np.maximum(combined, 1e-10)

    def get_har_coefs(self) -> Dict[str, float]:
        """Коэффициенты HAR модели"""
        return dict(zip(self.har_features, self.har_model.coef_))

    def get_ml_importance(self) -> Optional[pd.DataFrame]:
        """Feature importance из ML модели"""
        if hasattr(self.ml_model, 'feature_importances_'):
            importance = self.ml_model.feature_importances_
            return pd.DataFrame({
                'feature': self.external_features,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'har_model': self.har_model,
                'ml_model': self.ml_model,
                'har_features': self.har_features,
                'external_features': self.external_features,
                'horizon': self.horizon
            }, f)

    @classmethod
    def load(cls, path: str) -> "HybridHARML":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(
            horizon=data['horizon'],
            har_features=data['har_features'],
            external_features=data['external_features']
        )
        model.har_model = data['har_model']
        model.ml_model = data['ml_model']
        model.is_fitted = True
        return model


class HybridHARLGBM(HybridHARML):
    """
    Гибридная модель HAR + LightGBM на остатках.

    Использует LightGBM вместо HistGradientBoosting для лучшей производительности.
    """

    def __init__(self, horizon: int, har_features: List[str], external_features: List[str],
                 use_log: bool = True):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")

        self.horizon = horizon
        self.har_features = har_features
        self.external_features = external_features
        self.use_log = use_log
        self.har_model = LinearRegression()
        self.ml_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        self.is_fitted = False

    def get_ml_importance(self) -> Optional[pd.DataFrame]:
        """Feature importance из LightGBM модели"""
        if hasattr(self.ml_model, 'feature_importances_'):
            importance = self.ml_model.feature_importances_
            return pd.DataFrame({
                'feature': self.external_features,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None


class HybridModel(BaseVolatilityModel):
    """Hybrid model combining econometric and ML approaches.

    Strategies:
    1. GARCH + ML: Use GARCH residuals/conditional vol as features for ML
    2. HAR + ML: Use HAR predictions as features for ML
    3. Stacking: Train meta-model on base model predictions
    4. Weighted Average: Combine predictions with learned weights
    """

    def __init__(
        self,
        base_models: List[BaseVolatilityModel],
        meta_model: Optional[BaseVolatilityModel] = None,
        combine_method: Literal["stack", "average", "weighted"] = "stack",
        weights: Optional[List[float]] = None,
        use_base_features: bool = True,
        **kwargs
    ):
        """Initialize hybrid model.

        Args:
            base_models: List of base models.
            meta_model: Meta-learner for stacking.
            combine_method: How to combine predictions.
            weights: Weights for weighted average.
            use_base_features: Include original features in meta-model.
            **kwargs: Additional parameters.
        """
        super().__init__(name="Hybrid", **kwargs)

        self.base_models = base_models
        self.meta_model = meta_model
        self.combine_method = combine_method
        self.use_base_features = use_base_features

        if weights is None:
            self.weights = [1.0 / len(base_models)] * len(base_models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> "HybridModel":
        """Fit hybrid model.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features.
            y_val: Validation target.
            **kwargs: Additional parameters.

        Returns:
            Self.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_np = X.values
        else:
            X_np = X

        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y

        # Fit base models
        for model in self.base_models:
            model.fit(X, y, **kwargs)

        if self.combine_method == "stack" and self.meta_model is not None:
            # Get base model predictions for training meta-model
            base_preds = self._get_base_predictions(X)

            # Create meta-features
            if self.use_base_features:
                meta_X = np.column_stack([X_np, base_preds])
            else:
                meta_X = base_preds

            # Fit meta-model
            self.meta_model.fit(meta_X, y_np, **kwargs)

        elif self.combine_method == "weighted":
            # Learn optimal weights using validation set
            if X_val is not None and y_val is not None:
                self._optimize_weights(X_val, y_val)

        self.is_fitted = True
        return self

    def _get_base_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Get predictions from all base models.

        Args:
            X: Features.

        Returns:
            Array of shape (n_samples, n_models) with predictions.
        """
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)

        return np.column_stack(predictions)

    def _optimize_weights(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray]
    ) -> None:
        """Optimize combination weights using validation data.

        Args:
            X_val: Validation features.
            y_val: Validation target.
        """
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Get base predictions
        base_preds = self._get_base_predictions(X_val)

        # Simple optimization: minimize MSE
        # Using closed-form solution for linear combination
        try:
            # Solve: min ||y - Pw||^2 where P is predictions matrix
            # Solution: w = (P'P)^{-1} P'y
            P = base_preds
            weights = np.linalg.lstsq(P, y_val, rcond=None)[0]
            # Normalize to sum to 1
            weights = np.maximum(weights, 0)  # Non-negative
            weights = weights / weights.sum()
            self.weights = weights.tolist()
        except np.linalg.LinAlgError:
            # Keep equal weights if optimization fails
            pass

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
            X_np = X.values
        else:
            X_np = X

        # Get base predictions
        base_preds = self._get_base_predictions(X)

        if self.combine_method == "stack" and self.meta_model is not None:
            # Create meta-features
            if self.use_base_features:
                meta_X = np.column_stack([X_np, base_preds])
            else:
                meta_X = base_preds

            return self.meta_model.predict(meta_X)

        elif self.combine_method == "average":
            return base_preds.mean(axis=1)

        elif self.combine_method == "weighted":
            return np.average(base_preds, axis=1, weights=self.weights)

        return base_preds.mean(axis=1)

    def get_base_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Get predictions from each base model.

        Args:
            X: Features.

        Returns:
            Dictionary mapping model names to predictions.
        """
        return {
            model.name: model.predict(X)
            for model in self.base_models
        }

    def get_weights(self) -> Dict[str, float]:
        """Get combination weights.

        Returns:
            Dictionary mapping model names to weights.
        """
        return {
            model.name: w
            for model, w in zip(self.base_models, self.weights)
        }


class GARCHMLHybrid(HybridModel):
    """Hybrid model using GARCH features with ML.

    Uses GARCH conditional volatility and residuals as additional
    features for the ML model.
    """

    def __init__(
        self,
        ml_model: BaseVolatilityModel,
        garch_p: int = 1,
        garch_q: int = 1,
        **kwargs
    ):
        """Initialize GARCH-ML hybrid.

        Args:
            ml_model: ML model for final prediction.
            garch_p: GARCH p parameter.
            garch_q: GARCH q parameter.
            **kwargs: Additional parameters.
        """
        super().__init__(
            base_models=[],
            meta_model=ml_model,
            combine_method="stack",
            **kwargs
        )
        self.name = "GARCH-ML-Hybrid"
        self.garch_p = garch_p
        self.garch_q = garch_q
        self._garch_model = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> "GARCHMLHybrid":
        """Fit GARCH-ML hybrid model.

        Args:
            X: Features for ML model.
            y: Target (RV).
            returns: Return series for GARCH (if None, computed from X).
            **kwargs: Additional parameters.

        Returns:
            Self.
        """
        # Import here to avoid circular dependency
        from .garch import GARCHModel

        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_np = X.values
        else:
            X_np = X

        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y

        # Fit GARCH if returns provided
        if returns is not None:
            self._garch_model = GARCHModel(p=self.garch_p, q=self.garch_q)
            self._garch_model.fit(None, returns)

            # Get GARCH features
            garch_vol = self._garch_model.get_conditional_volatility()
            garch_resid = self._garch_model.get_standardized_residuals()

            # Align lengths
            min_len = min(len(garch_vol), len(X_np))
            X_np = X_np[-min_len:]
            y_np = y_np[-min_len:]
            garch_vol = garch_vol[-min_len:]
            garch_resid = garch_resid[-min_len:]

            # Add GARCH features
            X_enhanced = np.column_stack([X_np, garch_vol, garch_resid])
        else:
            X_enhanced = X_np

        # Fit ML model
        self.meta_model.fit(X_enhanced, y_np, **kwargs)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        returns: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features.
            returns: Returns for GARCH features.

        Returns:
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X

        # Add GARCH features if available
        if self._garch_model is not None and returns is not None:
            # Would need to fit GARCH on test returns
            # For simplicity, use last known volatility
            garch_vol = np.full(len(X_np), self._garch_model.get_conditional_volatility()[-1])
            garch_resid = np.zeros(len(X_np))

            X_enhanced = np.column_stack([X_np, garch_vol, garch_resid])
        else:
            X_enhanced = X_np

        return self.meta_model.predict(X_enhanced)


def create_ensemble(
    models: List[BaseVolatilityModel],
    method: Literal["simple", "weighted", "stacked"] = "simple",
    meta_model: Optional[BaseVolatilityModel] = None
) -> EnsembleModel:
    """Create an ensemble of models.

    Args:
        models: List of base models.
        method: Ensemble method.
        meta_model: Meta-learner for stacking.

    Returns:
        Ensemble model.
    """
    if method == "stacked" and meta_model is not None:
        return HybridModel(
            base_models=models,
            meta_model=meta_model,
            combine_method="stack"
        )

    return EnsembleModel(
        name="Ensemble",
        models=models,
        weights=None if method == "simple" else None
    )
