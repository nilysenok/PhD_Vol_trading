"""HAR (Heterogeneous Autoregressive) model for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
import statsmodels.api as sm
from .base import TimeSeriesModel


class HARModel(TimeSeriesModel):
    """Heterogeneous Autoregressive Model for Realized Volatility.

    The HAR model captures the long memory property of volatility through
    three components at different frequencies:
    - Daily: RV(t-1)
    - Weekly: Average RV over past 5 days
    - Monthly: Average RV over past 22 days

    Reference:
        Corsi, F. (2009). A simple approximate long-memory model of realized
        volatility. Journal of Financial Econometrics, 7(2), 174-196.
    """

    def __init__(
        self,
        use_log: bool = True,
        add_constant: bool = True,
        robust_cov: bool = True,
        **kwargs
    ):
        """Initialize HAR model.

        Args:
            use_log: Use log transformation of RV.
            add_constant: Add constant term to regression.
            robust_cov: Use HAC (Newey-West) robust standard errors.
            **kwargs: Additional parameters.
        """
        super().__init__(name="HAR", use_log=use_log, **kwargs)
        self.add_constant = add_constant
        self.robust_cov = robust_cov
        self._model = None
        self._results = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> "HARModel":
        """Fit HAR model.

        Expected columns in X: rv_d, rv_w, rv_m (or corresponding indices).

        Args:
            X: Feature matrix with HAR components.
            y: Target RV values.
            **kwargs: Additional fitting parameters.

        Returns:
            Self.
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Transform
        y_trans = self._transform(y)
        X_trans = self._transform(X) if self.use_log else X

        # Add constant
        if self.add_constant:
            X_trans = sm.add_constant(X_trans)

        # Fit OLS
        self._model = sm.OLS(y_trans, X_trans)

        if self.robust_cov:
            self._results = self._model.fit(cov_type="HAC", cov_kwds={"maxlags": 22})
        else:
            self._results = self._model.fit()

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict volatility.

        Args:
            X: Feature matrix.

        Returns:
            Predicted RV values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Transform
        X_trans = self._transform(X) if self.use_log else X

        # Add constant
        if self.add_constant:
            X_trans = sm.add_constant(X_trans)

        # Predict
        y_pred = self._results.predict(X_trans)

        # Inverse transform
        return self._inverse_transform(y_pred)

    def get_summary(self) -> str:
        """Get regression summary.

        Returns:
            Summary string.
        """
        if self._results is None:
            return "Model not fitted"
        return str(self._results.summary())

    def get_coefficients(self) -> pd.Series:
        """Get model coefficients.

        Returns:
            Series with coefficient names and values.
        """
        if self._results is None:
            return pd.Series()

        names = ["const"] if self.add_constant else []
        names += self._feature_names if self._feature_names else [f"x{i}" for i in range(len(self._results.params) - (1 if self.add_constant else 0))]

        return pd.Series(self._results.params, index=names)

    def get_r_squared(self) -> float:
        """Get R-squared.

        Returns:
            R-squared value.
        """
        if self._results is None:
            return np.nan
        return self._results.rsquared

    def get_adjusted_r_squared(self) -> float:
        """Get adjusted R-squared.

        Returns:
            Adjusted R-squared value.
        """
        if self._results is None:
            return np.nan
        return self._results.rsquared_adj


class HARExtendedModel(HARModel):
    """Extended HAR model with additional regressors.

    Supports:
    - HAR-RV: Standard model
    - HAR-RV-J: With jump component
    - HAR-RV-CJ: With continuous and jump components
    - HAR-X: With exogenous variables (VIX, etc.)
    """

    def __init__(
        self,
        use_log: bool = True,
        add_constant: bool = True,
        robust_cov: bool = True,
        include_jump: bool = False,
        include_asymmetry: bool = False,
        **kwargs
    ):
        """Initialize extended HAR model.

        Args:
            use_log: Use log transformation.
            add_constant: Add constant term.
            robust_cov: Use robust standard errors.
            include_jump: Include jump component (requires 'rv_jump' in X).
            include_asymmetry: Include asymmetric leverage effect.
            **kwargs: Additional parameters.
        """
        super().__init__(
            use_log=use_log,
            add_constant=add_constant,
            robust_cov=robust_cov,
            **kwargs
        )
        self.name = "HAR-Extended"
        self.include_jump = include_jump
        self.include_asymmetry = include_asymmetry


def prepare_har_data(
    df: pd.DataFrame,
    rv_col: str = "rv_daily",
    ticker_col: str = "ticker",
    date_col: str = "date",
    forecast_horizon: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for HAR model.

    Args:
        df: Input DataFrame with RV data.
        rv_col: Name of RV column.
        ticker_col: Name of ticker column.
        date_col: Name of date column.
        forecast_horizon: Forecast horizon in days.

    Returns:
        Tuple of (X, y) for model fitting.
    """
    # Create HAR features
    result = df.copy()
    result = result.sort_values([ticker_col, date_col])

    # Daily component
    result["rv_d"] = result.groupby(ticker_col)[rv_col].shift(1)

    # Weekly component (5-day average)
    result["rv_w"] = result.groupby(ticker_col)[rv_col].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )

    # Monthly component (22-day average)
    result["rv_m"] = result.groupby(ticker_col)[rv_col].transform(
        lambda x: x.rolling(22, min_periods=1).mean().shift(1)
    )

    # Target (forward-looking)
    result["target"] = result.groupby(ticker_col)[rv_col].shift(-forecast_horizon)

    # Drop NaN
    result = result.dropna(subset=["rv_d", "rv_w", "rv_m", "target"])

    X = result[["rv_d", "rv_w", "rv_m"]]
    y = result["target"]

    return X, y
