"""GARCH models for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, Literal
import warnings
from .base import TimeSeriesModel

# Conditional import for arch package
try:
    from arch import arch_model
    from arch.univariate import ConstantMean, ZeroMean, GARCH, EGARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch package not installed. GARCH models will not be available.")


class GARCHModel(TimeSeriesModel):
    """GARCH(p,q) model for volatility forecasting.

    Supports:
    - GARCH(1,1): Standard GARCH
    - GJR-GARCH: Asymmetric GARCH (leverage effect)
    - EGARCH: Exponential GARCH

    Reference:
        Bollerslev, T. (1986). Generalized autoregressive conditional
        heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        o: int = 0,
        vol: Literal["GARCH", "EGARCH", "GJR"] = "GARCH",
        mean: Literal["Zero", "Constant", "AR"] = "Zero",
        dist: Literal["normal", "t", "skewt", "ged"] = "normal",
        rescale: bool = True,
        **kwargs
    ):
        """Initialize GARCH model.

        Args:
            p: Lag order of GARCH terms.
            q: Lag order of ARCH terms.
            o: Lag order for asymmetric terms (GJR-GARCH).
            vol: Volatility model type.
            mean: Mean model type.
            dist: Error distribution.
            rescale: Rescale data for numerical stability.
            **kwargs: Additional parameters.
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package is required for GARCH models. Install with: pip install arch")

        name = f"{vol}({p},{q})" if o == 0 else f"GJR-GARCH({p},{o},{q})"
        super().__init__(name=name, use_log=False, **kwargs)

        self.p = p
        self.q = q
        self.o = o
        self.vol = vol
        self.mean = mean
        self.dist = dist
        self.rescale = rescale
        self._model = None
        self._results = None
        self._scale = 1.0

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> "GARCHModel":
        """Fit GARCH model.

        Note: For GARCH, X is typically ignored as it's a univariate model.
        The target y should be returns (not RV).

        Args:
            X: Ignored (univariate model).
            y: Return series.
            **kwargs: Additional fitting parameters.

        Returns:
            Self.
        """
        # Convert to numpy
        if isinstance(y, pd.Series):
            y = y.values

        # Rescale for numerical stability
        if self.rescale:
            self._scale = y.std()
            y = y / self._scale * 100  # Scale to percentage

        # Create model
        self._model = arch_model(
            y,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            o=self.o,
            dist=self.dist
        )

        # Fit
        self._results = self._model.fit(disp="off", **kwargs)
        self.is_fitted = True

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        horizon: int = 1,
        method: Literal["analytic", "simulation", "bootstrap"] = "analytic"
    ) -> np.ndarray:
        """Predict volatility.

        Args:
            X: Ignored for univariate model.
            horizon: Forecast horizon.
            method: Forecasting method.

        Returns:
            Predicted variance (not volatility).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Forecast
        forecasts = self._results.forecast(horizon=horizon, method=method)

        # Get variance forecast
        variance = forecasts.variance.iloc[-1].values

        # Rescale back
        if self.rescale:
            variance = variance * (self._scale / 100) ** 2

        return variance

    def forecast_volatility(
        self,
        horizon: int = 1,
        annualize: bool = True,
        trading_days: int = 252
    ) -> np.ndarray:
        """Forecast volatility (standard deviation).

        Args:
            horizon: Forecast horizon.
            annualize: Whether to annualize the forecast.
            trading_days: Number of trading days per year.

        Returns:
            Volatility forecast.
        """
        variance = self.predict(None, horizon=horizon)
        volatility = np.sqrt(variance)

        if annualize:
            volatility = volatility * np.sqrt(trading_days)

        return volatility

    def get_summary(self) -> str:
        """Get model summary.

        Returns:
            Summary string.
        """
        if self._results is None:
            return "Model not fitted"
        return str(self._results.summary())

    def get_conditional_volatility(self) -> np.ndarray:
        """Get fitted conditional volatility.

        Returns:
            Conditional volatility series.
        """
        if self._results is None:
            return np.array([])

        vol = self._results.conditional_volatility

        # Rescale
        if self.rescale:
            vol = vol * (self._scale / 100)

        return vol

    def get_standardized_residuals(self) -> np.ndarray:
        """Get standardized residuals.

        Returns:
            Standardized residuals.
        """
        if self._results is None:
            return np.array([])
        return self._results.std_resid

    def get_parameters(self) -> Dict[str, float]:
        """Get estimated parameters.

        Returns:
            Dictionary of parameter estimates.
        """
        if self._results is None:
            return {}
        return dict(self._results.params)

    def get_aic(self) -> float:
        """Get AIC."""
        if self._results is None:
            return np.nan
        return self._results.aic

    def get_bic(self) -> float:
        """Get BIC."""
        if self._results is None:
            return np.nan
        return self._results.bic


class RealizedGARCH(TimeSeriesModel):
    """Realized GARCH model combining returns and RV.

    This model uses realized volatility as an additional measurement
    equation to improve volatility forecasting.

    Reference:
        Hansen, P. R., Huang, Z., & Shek, H. H. (2012). Realized GARCH:
        A joint model for returns and realized measures of volatility.
        Journal of Applied Econometrics, 27(6), 877-906.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        **kwargs
    ):
        """Initialize Realized GARCH model.

        Args:
            p: GARCH lag order.
            q: ARCH lag order.
            **kwargs: Additional parameters.
        """
        super().__init__(name=f"Realized-GARCH({p},{q})", use_log=True, **kwargs)
        self.p = p
        self.q = q
        # Note: Full implementation requires custom estimation
        # This is a placeholder for the structure
        raise NotImplementedError(
            "Realized GARCH requires custom implementation. "
            "Consider using standard GARCH with RV as proxy."
        )

    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X):
        pass


def fit_garch_for_rv(
    returns: Union[pd.Series, np.ndarray],
    rv: Union[pd.Series, np.ndarray],
    p: int = 1,
    q: int = 1,
    vol: str = "GARCH"
) -> Dict[str, Any]:
    """Fit GARCH on returns and compare with RV.

    Args:
        returns: Return series.
        rv: Realized volatility series.
        p: GARCH p parameter.
        q: GARCH q parameter.
        vol: Volatility model type.

    Returns:
        Dictionary with model, results, and comparison metrics.
    """
    model = GARCHModel(p=p, q=q, vol=vol)
    model.fit(None, returns)

    # Get conditional volatility
    cond_vol = model.get_conditional_volatility()

    # Compare with RV
    if isinstance(rv, pd.Series):
        rv = rv.values

    # Align lengths
    min_len = min(len(cond_vol), len(rv))
    cond_vol = cond_vol[-min_len:]
    rv_aligned = rv[-min_len:]

    # Calculate correlation
    corr = np.corrcoef(cond_vol, rv_aligned)[0, 1]

    return {
        "model": model,
        "conditional_volatility": cond_vol,
        "correlation_with_rv": corr,
        "aic": model.get_aic(),
        "bic": model.get_bic(),
    }
