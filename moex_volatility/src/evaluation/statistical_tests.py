"""Statistical tests for volatility forecast evaluation."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings

# Conditional imports
try:
    from scipy import stats
    from scipy.stats import ttest_rel, wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not installed. Statistical tests will not be available.")

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. Some tests will not be available.")


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    conclusion: str
    details: Optional[Dict] = None

    def __str__(self) -> str:
        return (
            f"{self.test_name}:\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value:   {self.p_value:.4f}\n"
            f"  Conclusion: {self.conclusion}"
        )


def mincer_zarnowitz_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> TestResult:
    """Mincer-Zarnowitz regression test for forecast efficiency.

    Tests H0: α=0, β=1 in the regression:
    y_true = α + β * y_pred + ε

    An efficient forecast should have α=0 and β=1.

    Reference:
        Mincer, J., & Zarnowitz, V. (1969). The evaluation of economic
        forecasts. In Economic forecasts and expectations.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        alpha: Significance level.

    Returns:
        TestResult with F-statistic and p-value.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for Mincer-Zarnowitz test")

    # Run regression
    X = sm.add_constant(y_pred)
    model = sm.OLS(y_true, X).fit()

    # Test joint hypothesis: α=0, β=1
    # Using Wald test
    r_matrix = np.array([[1, 0], [0, 1]])  # Restrictions matrix
    q = np.array([0, 1])  # Null hypothesis values

    wald_test = model.wald_test((r_matrix, q))
    f_stat = wald_test.statistic
    p_val = wald_test.pvalue

    # Individual t-tests
    alpha_est = model.params[0]
    beta_est = model.params[1]
    alpha_pval = model.pvalues[0]
    beta_pval = 2 * (1 - stats.t.cdf(abs((beta_est - 1) / model.bse[1]), model.df_resid))

    if p_val < alpha:
        conclusion = f"Reject H0: Forecast is NOT efficient (p={p_val:.4f})"
    else:
        conclusion = f"Cannot reject H0: Forecast may be efficient (p={p_val:.4f})"

    return TestResult(
        test_name="Mincer-Zarnowitz Test",
        statistic=float(f_stat),
        p_value=float(p_val),
        conclusion=conclusion,
        details={
            "alpha": alpha_est,
            "beta": beta_est,
            "alpha_pvalue": alpha_pval,
            "beta_pvalue": beta_pval,
            "r_squared": model.rsquared,
        }
    )


def diebold_mariano_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss_func: str = "mse",
    h: int = 1,
    alpha: float = 0.05
) -> TestResult:
    """Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t)
    is the loss differential.

    Reference:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
        accuracy. Journal of Business & Economic Statistics.

    Args:
        y_true: Actual values.
        pred1: Predictions from model 1.
        pred2: Predictions from model 2.
        loss_func: Loss function ('mse', 'mae', 'qlike').
        h: Forecast horizon.
        alpha: Significance level.

    Returns:
        TestResult with DM statistic and p-value.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Diebold-Mariano test")

    # Calculate errors
    e1 = y_true - pred1
    e2 = y_true - pred2

    # Calculate loss
    if loss_func == "mse":
        d = e1**2 - e2**2
    elif loss_func == "mae":
        d = np.abs(e1) - np.abs(e2)
    elif loss_func == "qlike":
        d = (np.log(pred1) + y_true/pred1) - (np.log(pred2) + y_true/pred2)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    # DM statistic with Newey-West standard errors
    n = len(d)
    d_mean = np.mean(d)

    # Autocovariance
    gamma = np.zeros(h)
    for i in range(h):
        gamma[i] = np.mean((d[i:] - d_mean) * (d[:n-i] - d_mean))

    # Long-run variance (Newey-West)
    var_d = gamma[0] + 2 * np.sum(gamma[1:])
    var_d = max(var_d, 1e-10)  # Ensure positive

    # DM statistic
    dm_stat = d_mean / np.sqrt(var_d / n)

    # Two-sided p-value
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if p_val < alpha:
        if dm_stat < 0:
            conclusion = f"Model 1 significantly better (p={p_val:.4f})"
        else:
            conclusion = f"Model 2 significantly better (p={p_val:.4f})"
    else:
        conclusion = f"No significant difference (p={p_val:.4f})"

    return TestResult(
        test_name="Diebold-Mariano Test",
        statistic=dm_stat,
        p_value=p_val,
        conclusion=conclusion,
        details={
            "mean_loss_diff": d_mean,
            "model1_mean_loss": np.mean(e1**2) if loss_func == "mse" else np.mean(np.abs(e1)),
            "model2_mean_loss": np.mean(e2**2) if loss_func == "mse" else np.mean(np.abs(e2)),
        }
    )


def giacomini_white_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    instruments: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> TestResult:
    """Giacomini-White test for conditional predictive ability.

    Tests whether model 1 outperforms model 2 conditionally on
    available information.

    Reference:
        Giacomini, R., & White, H. (2006). Tests of conditional
        predictive ability. Econometrica.

    Args:
        y_true: Actual values.
        pred1: Predictions from model 1.
        pred2: Predictions from model 2.
        instruments: Conditioning instruments (if None, uses constant).
        alpha: Significance level.

    Returns:
        TestResult with GW statistic and p-value.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for Giacomini-White test")

    # Loss differential
    d = (y_true - pred1)**2 - (y_true - pred2)**2

    # Instruments
    if instruments is None:
        Z = np.ones((len(d), 1))
    else:
        Z = instruments
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

    # Regression: d_t = Z_{t-1}' * delta + u_t
    # Under H0: delta = 0
    model = sm.OLS(d, Z).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    # Wald test
    wald = model.wald_test(np.eye(Z.shape[1]))
    gw_stat = float(wald.statistic)
    p_val = float(wald.pvalue)

    if p_val < alpha:
        conclusion = f"Reject equal conditional predictive ability (p={p_val:.4f})"
    else:
        conclusion = f"Cannot reject equal ability (p={p_val:.4f})"

    return TestResult(
        test_name="Giacomini-White Test",
        statistic=gw_stat,
        p_value=p_val,
        conclusion=conclusion,
        details={"coefficients": model.params.tolist()}
    )


def model_confidence_set(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    alpha: float = 0.10,
    loss_func: str = "mse"
) -> Dict[str, bool]:
    """Model Confidence Set (MCS) procedure.

    Determines the set of models that contains the best model with
    probability (1 - alpha).

    Reference:
        Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model
        confidence set. Econometrica.

    Args:
        y_true: Actual values.
        predictions: Dictionary of model predictions.
        alpha: Significance level.
        loss_func: Loss function.

    Returns:
        Dictionary indicating which models are in the MCS.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for MCS")

    model_names = list(predictions.keys())
    n_models = len(model_names)

    # Calculate losses
    losses = {}
    for name, pred in predictions.items():
        if loss_func == "mse":
            losses[name] = (y_true - pred)**2
        elif loss_func == "mae":
            losses[name] = np.abs(y_true - pred)
        elif loss_func == "qlike":
            losses[name] = np.log(pred) + y_true / pred

    # Initialize MCS with all models
    mcs = set(model_names)

    # Elimination procedure
    while len(mcs) > 1:
        models_in_mcs = list(mcs)
        n_in_mcs = len(models_in_mcs)

        # Calculate loss differentials
        d_matrix = np.zeros((n_in_mcs, n_in_mcs))
        p_matrix = np.zeros((n_in_mcs, n_in_mcs))

        for i, m1 in enumerate(models_in_mcs):
            for j, m2 in enumerate(models_in_mcs):
                if i != j:
                    d = losses[m1] - losses[m2]
                    # t-test for d != 0
                    t_stat, p_val = stats.ttest_1samp(d, 0)
                    d_matrix[i, j] = np.mean(d)
                    p_matrix[i, j] = p_val

        # Find worst model (highest average loss)
        avg_losses = {m: np.mean(losses[m]) for m in models_in_mcs}
        worst_model = max(avg_losses, key=avg_losses.get)

        # Test if worst model is significantly worse
        worst_idx = models_in_mcs.index(worst_model)
        min_p_val = 1.0

        for j, m2 in enumerate(models_in_mcs):
            if j != worst_idx:
                min_p_val = min(min_p_val, p_matrix[worst_idx, j])

        # Eliminate if significantly worse
        if min_p_val < alpha / (n_in_mcs - 1):  # Bonferroni correction
            mcs.remove(worst_model)
        else:
            break

    return {name: name in mcs for name in model_names}


def ljung_box_test(
    residuals: np.ndarray,
    lags: int = 10,
    alpha: float = 0.05
) -> TestResult:
    """Ljung-Box test for autocorrelation in residuals.

    Args:
        residuals: Forecast residuals.
        lags: Number of lags to test.
        alpha: Significance level.

    Returns:
        TestResult with Q-statistic and p-value.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for Ljung-Box test")

    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    q_stat = result["lb_stat"].iloc[0]
    p_val = result["lb_pvalue"].iloc[0]

    if p_val < alpha:
        conclusion = f"Reject H0: Residuals are autocorrelated (p={p_val:.4f})"
    else:
        conclusion = f"Cannot reject H0: No significant autocorrelation (p={p_val:.4f})"

    return TestResult(
        test_name="Ljung-Box Test",
        statistic=q_stat,
        p_value=p_val,
        conclusion=conclusion
    )


def forecast_encompassing_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    alpha: float = 0.05
) -> TestResult:
    """Forecast encompassing test.

    Tests whether forecast 1 encompasses forecast 2, i.e., whether
    forecast 2 adds any information beyond forecast 1.

    H0: λ = 0 in y = (1-λ)*f1 + λ*f2 + ε

    Args:
        y_true: Actual values.
        pred1: Predictions from model 1.
        pred2: Predictions from model 2.
        alpha: Significance level.

    Returns:
        TestResult.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for encompassing test")

    # Create combination forecast
    # y = (1-λ)*f1 + λ*f2 = f1 + λ*(f2 - f1)
    X = pred2 - pred1
    y = y_true - pred1

    # Regression
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    lambda_est = model.params[1]
    t_stat = model.tvalues[1]
    p_val = model.pvalues[1]

    if p_val < alpha:
        if lambda_est > 0:
            conclusion = f"Model 2 adds information beyond Model 1 (λ={lambda_est:.4f}, p={p_val:.4f})"
        else:
            conclusion = f"Model 2 reduces accuracy (λ={lambda_est:.4f}, p={p_val:.4f})"
    else:
        conclusion = f"Model 1 encompasses Model 2 (λ={lambda_est:.4f}, p={p_val:.4f})"

    return TestResult(
        test_name="Forecast Encompassing Test",
        statistic=t_stat,
        p_value=p_val,
        conclusion=conclusion,
        details={
            "lambda": lambda_est,
            "optimal_weight_model2": max(0, min(1, lambda_est)),
        }
    )
