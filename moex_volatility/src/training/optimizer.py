"""Optuna hyperparameter optimization for volatility models."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Type, Callable, List, Union
from pathlib import Path
import pickle
import json
import warnings

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not installed. Hyperparameter optimization will not be available.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, OSError):
    XGBOOST_AVAILABLE = False

from ..models.base import BaseVolatilityModel
from ..evaluation.metrics import mse, rmse, mae, qlike


def qlike_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate QLIKE loss for volatility forecasting.

    QLIKE = mean(log(σ²_pred) + σ²_true / σ²_pred)

    This is the standard QLIKE from Patton (2011).
    Lower (more negative) is better.

    Args:
        y_true: True realized volatility values.
        y_pred: Predicted volatility values.

    Returns:
        QLIKE loss value (lower/more negative is better).
    """
    y_pred = np.clip(y_pred, 1e-10, None)
    y_true = np.clip(y_true, 1e-10, None)
    return np.mean(np.log(y_pred) + y_true / y_pred)


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for volatility models.

    Supports:
    - Multiple objective metrics (QLIKE, MSE, MAE)
    - Pruning of unpromising trials
    - Parallel optimization
    - Study persistence
    """

    # Default search spaces for different model types
    SEARCH_SPACES = {
        "lightgbm": {
            "n_estimators": ("int", 100, 2000),
            "max_depth": ("int", -1, 15),
            "learning_rate": ("log_float", 0.005, 0.2),
            "num_leaves": ("int", 7, 127),
            "min_child_samples": ("int", 5, 100),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("log_float", 1e-8, 10.0),
            "reg_lambda": ("log_float", 1e-8, 10.0),
        },
        "xgboost": {
            "n_estimators": ("int", 100, 2000),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("log_float", 0.005, 0.2),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("log_float", 1e-8, 10.0),
            "reg_lambda": ("log_float", 1e-8, 10.0),
        },
        "gru": {
            "hidden_size": ("categorical", [16, 32, 64, 128, 256]),
            "num_layers": ("int", 1, 3),
            "dropout": ("float", 0.0, 0.5),
            "sequence_length": ("categorical", [5, 10, 22, 44, 66]),
            "learning_rate": ("log_float", 0.0001, 0.01),
            "batch_size": ("categorical", [16, 32, 64, 128]),
            "bidirectional": ("categorical", [True, False]),
        },
        "lstm": {
            "hidden_size": ("categorical", [16, 32, 64, 128, 256]),
            "num_layers": ("int", 1, 3),
            "dropout": ("float", 0.0, 0.5),
            "sequence_length": ("categorical", [5, 10, 22, 44, 66]),
            "learning_rate": ("log_float", 0.0001, 0.01),
            "batch_size": ("categorical", [16, 32, 64, 128]),
            "bidirectional": ("categorical", [True, False]),
            "use_attention": ("categorical", [True, False]),
        },
        "har": {
            "lags": ("categorical", [[1, 5, 22], [1, 5, 10, 22], [1, 2, 5, 10, 22]]),
            "include_jump": ("categorical", [True, False]),
            "use_log": ("categorical", [True, False]),
        },
    }

    # Metric functions
    METRICS = {
        "qlike": qlike,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }

    def __init__(
        self,
        model_class: Type[BaseVolatilityModel],
        model_type: str,
        metric: str = "qlike",
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
        seed: int = 42,
        search_space: Optional[Dict[str, tuple]] = None,
        pruner: str = "median",
        study_name: Optional[str] = None,
    ):
        """Initialize optimizer.

        Args:
            model_class: Model class to optimize.
            model_type: Type of model for default search space.
            metric: Optimization metric ('qlike', 'mse', 'rmse', 'mae').
            n_trials: Number of optimization trials.
            n_jobs: Number of parallel jobs.
            timeout: Timeout in seconds.
            seed: Random seed.
            search_space: Custom search space (overrides defaults).
            pruner: Pruner type ('median', 'hyperband', 'none').
            study_name: Name for the Optuna study.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna required. Install with: pip install optuna")

        self.model_class = model_class
        self.model_type = model_type
        self.metric = metric
        self.metric_func = self.METRICS.get(metric, mse)
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.seed = seed
        self.study_name = study_name or f"{model_type}_optimization"

        # Search space
        if search_space is not None:
            self.search_space = search_space
        else:
            self.search_space = self.SEARCH_SPACES.get(model_type, {})

        # Pruner
        if pruner == "median":
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == "hyperband":
            self.pruner = HyperbandPruner()
        else:
            self.pruner = None

        # Sampler
        self.sampler = TPESampler(seed=seed)

        # Results
        self._study: Optional[optuna.Study] = None
        self._best_params: Optional[Dict[str, Any]] = None

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters from search space.

        Args:
            trial: Optuna trial.

        Returns:
            Dictionary of sampled parameters.
        """
        params = {}

        for name, spec in self.search_space.items():
            param_type = spec[0]

            if param_type == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif param_type == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif param_type == "log_float":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])

        return params

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fit_kwargs: Dict[str, Any]
    ) -> float:
        """Objective function for optimization.

        Args:
            trial: Optuna trial.
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            fit_kwargs: Additional fit parameters.

        Returns:
            Metric value (lower is better).
        """
        # Sample parameters
        params = self._sample_params(trial)

        try:
            # Create and fit model
            model = self.model_class(**params)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs)

            # Predict
            y_pred = model.predict(X_val)

            # Handle sequence models with shorter output
            if len(y_pred) < len(y_val):
                y_val_aligned = y_val[-len(y_pred):]
            else:
                y_val_aligned = y_val

            # Calculate metric
            score = self.metric_func(y_val_aligned, y_pred)

            # Report for pruning (if applicable)
            if hasattr(model, "_training_history") and "val_loss" in model._training_history:
                for step, val_loss in enumerate(model._training_history["val_loss"]):
                    trial.report(val_loss, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            return score

        except Exception as e:
            # Return worst possible score on error
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")

    def optimize(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        fit_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            fit_kwargs: Additional fit parameters.
            verbose: Print progress.

        Returns:
            Best parameters.
        """
        # Convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        fit_kwargs = fit_kwargs or {}

        # Create study
        self._study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            sampler=self.sampler,
            pruner=self.pruner,
        )

        # Optimize
        verbosity = optuna.logging.INFO if verbose else optuna.logging.WARNING
        optuna.logging.set_verbosity(verbosity)

        self._study.optimize(
            lambda trial: self._objective(
                trial, X_train, y_train, X_val, y_val, fit_kwargs
            ),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=verbose,
        )

        self._best_params = self._study.best_params

        if verbose:
            print(f"\nBest {self.metric}: {self._study.best_value:.6f}")
            print(f"Best parameters: {self._best_params}")

        return self._best_params

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from optimization.

        Returns:
            Best parameters dictionary.
        """
        if self._best_params is None:
            raise ValueError("No optimization has been run yet")
        return self._best_params.copy()

    def get_study(self) -> optuna.Study:
        """Get Optuna study object.

        Returns:
            Optuna Study.
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")
        return self._study

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get trials as DataFrame.

        Returns:
            DataFrame with all trials.
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")
        return self._study.trials_dataframe()

    def save_study(self, path: str) -> None:
        """Save study to pickle file.

        Args:
            path: Path to save study.
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._study, f)

    def save_best_params(self, path: str) -> None:
        """Save best parameters to JSON.

        Args:
            path: Path to save parameters.
        """
        if self._best_params is None:
            raise ValueError("No optimization has been run yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self._best_params, f, indent=2)

    def load_study(self, path: str) -> None:
        """Load study from pickle file.

        Args:
            path: Path to saved study.
        """
        with open(path, "rb") as f:
            self._study = pickle.load(f)
        self._best_params = self._study.best_params

    def plot_importance(self, save_path: Optional[str] = None):
        """Plot parameter importance.

        Args:
            save_path: Path to save figure.
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")

        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self._study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except ImportError:
            print("plotly required for visualization")
            return None

    def plot_history(self, save_path: Optional[str] = None):
        """Plot optimization history.

        Args:
            save_path: Path to save figure.
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")

        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self._study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except ImportError:
            print("plotly required for visualization")
            return None

    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """Plot parallel coordinate.

        Args:
            save_path: Path to save figure.
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")

        try:
            from optuna.visualization import plot_parallel_coordinate
            fig = plot_parallel_coordinate(self._study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except ImportError:
            print("plotly required for visualization")
            return None


def quick_optimize(
    model_class: Type[BaseVolatilityModel],
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    metric: str = "qlike",
    verbose: bool = True
) -> Dict[str, Any]:
    """Quick hyperparameter optimization.

    Args:
        model_class: Model class.
        model_type: Model type for search space.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        n_trials: Number of trials.
        metric: Optimization metric.
        verbose: Print progress.

    Returns:
        Best parameters.
    """
    optimizer = OptunaOptimizer(
        model_class=model_class,
        model_type=model_type,
        metric=metric,
        n_trials=n_trials,
    )

    return optimizer.optimize(X_train, y_train, X_val, y_val, verbose=verbose)


class BoostingOptimizer:
    """Simple Optuna optimizer for boosting models (LightGBM/XGBoost).

    This is a standalone optimizer that doesn't require the model classes.
    """

    def __init__(
        self,
        model_type: str = 'lightgbm',
        n_trials: int = 50,
        metric: str = 'qlike',
        random_state: int = 42,
        verbose: bool = True
    ):
        """Initialize optimizer.

        Args:
            model_type: 'lightgbm' or 'xgboost'.
            n_trials: Number of optimization trials.
            metric: Optimization metric ('qlike' or 'mse').
            random_state: Random seed.
            verbose: Print progress.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna required. Install with: pip install optuna")

        self.model_type = model_type
        self.n_trials = n_trials
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose
        self.study = None
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float('inf')

        if metric == 'qlike':
            self.metric_fn = qlike_metric
        else:
            self.metric_fn = lambda y, p: np.mean((y - p) ** 2)

    def _get_lgb_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Get LightGBM search space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': -1
        }

    def _get_xgb_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Get XGBoost search space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 0.5, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'verbosity': 0,
            'random_state': self.random_state,
            'n_jobs': -1
        }

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Best hyperparameters.
        """
        # Ensure float type and handle inf/nan
        X_train = np.asarray(X_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.where(np.isinf(X_val), np.nan, X_val)
        X_val = np.nan_to_num(X_val, nan=0.0)

        def objective(trial: "optuna.Trial") -> float:
            try:
                if self.model_type == 'lightgbm':
                    if not LIGHTGBM_AVAILABLE:
                        raise ImportError("lightgbm not available")
                    params = self._get_lgb_params(trial)
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )
                else:
                    if not XGBOOST_AVAILABLE:
                        raise ImportError("xgboost not available")
                    params = self._get_xgb_params(trial)
                    model = xgb.XGBRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )

                y_pred = model.predict(X_val)
                y_pred = np.clip(y_pred, 1e-10, None)
                return self.metric_fn(y_val, y_pred)

            except Exception:
                return float('inf')

        # Create study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            n_jobs=1
        )

        self.best_params = self.study.best_params.copy()
        self.best_score = self.study.best_value

        # Add fixed params
        if self.model_type == 'lightgbm':
            self.best_params['verbosity'] = -1
        else:
            self.best_params['verbosity'] = 0
        self.best_params['random_state'] = self.random_state
        self.best_params['n_jobs'] = -1

        return self.best_params
