"""Walk-forward cross-validation for time series models."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Generator, Union, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WalkForwardSplit:
    """A single train/test split in walk-forward CV."""
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_dates: Optional[Tuple[datetime, datetime]] = None
    test_dates: Optional[Tuple[datetime, datetime]] = None


class WalkForwardCV:
    """Walk-forward cross-validation for time series.

    Implements expanding or rolling window cross-validation
    appropriate for time series data.

    Example:
        Expanding window (initial=3, step=1, test=1):
        Fold 1: [0,1,2] -> [3]
        Fold 2: [0,1,2,3] -> [4]
        Fold 3: [0,1,2,3,4] -> [5]

        Rolling window (initial=3, step=1, test=1):
        Fold 1: [0,1,2] -> [3]
        Fold 2: [1,2,3] -> [4]
        Fold 3: [2,3,4] -> [5]
    """

    def __init__(
        self,
        initial_train_size: int = 756,
        step_size: int = 63,
        test_size: int = 63,
        expanding: bool = True,
        min_train_size: Optional[int] = None
    ):
        """Initialize walk-forward CV.

        Args:
            initial_train_size: Initial training window size.
            step_size: Step size between folds.
            test_size: Test window size.
            expanding: Use expanding (True) or rolling (False) window.
            min_train_size: Minimum training size for rolling window.
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.test_size = test_size
        self.expanding = expanding
        self.min_train_size = min_train_size or initial_train_size

    def get_n_splits(self, n_samples: int) -> int:
        """Get number of splits.

        Args:
            n_samples: Total number of samples.

        Returns:
            Number of splits.
        """
        available = n_samples - self.initial_train_size - self.test_size
        if available < 0:
            return 0
        return 1 + available // self.step_size

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Generator[WalkForwardSplit, None, None]:
        """Generate train/test splits.

        Args:
            X: Feature matrix.
            y: Target (optional, not used).
            dates: Date index for date information.

        Yields:
            WalkForwardSplit objects.
        """
        n_samples = len(X)
        n_splits = self.get_n_splits(n_samples)

        for fold in range(n_splits):
            # Calculate indices
            test_end = self.initial_train_size + self.test_size + fold * self.step_size
            test_start = test_end - self.test_size

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.initial_train_size - fold * self.step_size)
                # Ensure minimum training size
                if test_start - train_start < self.min_train_size:
                    train_start = max(0, test_start - self.min_train_size)

            train_end = test_start

            # Get dates if available
            train_dates = None
            test_dates = None
            if dates is not None:
                train_dates = (dates[train_start], dates[train_end - 1])
                test_dates = (dates[test_start], dates[test_end - 1])

            yield WalkForwardSplit(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_dates=train_dates,
                test_dates=test_dates,
            )

    def split_arrays(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, WalkForwardSplit], None, None]:
        """Generate train/test arrays.

        Args:
            X: Feature matrix.
            y: Target.
            dates: Date index.

        Yields:
            Tuple of (X_train, X_test, y_train, y_test, split_info).
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        for split in self.split(X, y, dates):
            X_train = X_arr[split.train_start:split.train_end]
            X_test = X_arr[split.test_start:split.test_end]
            y_train = y_arr[split.train_start:split.train_end]
            y_test = y_arr[split.test_start:split.test_end]

            yield X_train, X_test, y_train, y_test, split

    def __repr__(self) -> str:
        return (
            f"WalkForwardCV(initial={self.initial_train_size}, "
            f"step={self.step_size}, test={self.test_size}, "
            f"expanding={self.expanding})"
        )


class TimeSeriesSplit:
    """Simple time series split (train before test)."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """Initialize time series split.

        Args:
            n_splits: Number of splits.
            test_size: Size of test set (None for equal sizes).
            gap: Gap between train and test.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices.

        Args:
            X: Feature matrix.
            y: Target (optional).

        Yields:
            Tuple of (train_indices, test_indices).
        """
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        for i in range(self.n_splits):
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices


def create_walk_forward_splits(
    dates: pd.DatetimeIndex,
    initial_years: float = 3.0,
    step_months: int = 3,
    test_months: int = 3,
    expanding: bool = True
) -> List[WalkForwardSplit]:
    """Create walk-forward splits based on dates.

    Args:
        dates: DatetimeIndex of trading dates.
        initial_years: Initial training period in years.
        step_months: Step size in months.
        test_months: Test period in months.
        expanding: Use expanding window.

    Returns:
        List of WalkForwardSplit objects.
    """
    # Convert to trading days (approximate)
    initial_days = int(initial_years * 252)
    step_days = int(step_months * 21)
    test_days = int(test_months * 21)

    cv = WalkForwardCV(
        initial_train_size=initial_days,
        step_size=step_days,
        test_size=test_days,
        expanding=expanding
    )

    return list(cv.split(np.zeros(len(dates)), dates=dates))


def walk_forward_predict(
    model_class,
    X: pd.DataFrame,
    y: pd.Series,
    cv: WalkForwardCV,
    model_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Run walk-forward prediction.

    Args:
        model_class: Model class to instantiate.
        X: Features.
        y: Target.
        cv: WalkForwardCV instance.
        model_params: Parameters for model initialization.
        fit_params: Parameters for model fitting.
        verbose: Print progress.

    Returns:
        Tuple of (all_predictions, all_actuals, fold_results).
    """
    model_params = model_params or {}
    fit_params = fit_params or {}

    all_preds = []
    all_actuals = []
    fold_results = []

    dates = X.index if isinstance(X.index, pd.DatetimeIndex) else None

    for X_train, X_test, y_train, y_test, split in cv.split_arrays(X, y, dates):
        if verbose:
            print(f"Fold {split.fold + 1}: Train={len(X_train)}, Test={len(X_test)}")
            if split.train_dates:
                print(f"  Train: {split.train_dates[0].date()} to {split.train_dates[1].date()}")
                print(f"  Test:  {split.test_dates[0].date()} to {split.test_dates[1].date()}")

        # Initialize and fit model
        model = model_class(**model_params)
        model.fit(X_train, y_train, **fit_params)

        # Predict
        y_pred = model.predict(X_test)

        # Handle sequence models that return shorter predictions
        if len(y_pred) < len(y_test):
            offset = len(y_test) - len(y_pred)
            y_test = y_test[offset:]

        all_preds.extend(y_pred)
        all_actuals.extend(y_test)

        # Calculate fold metrics
        mse = np.mean((y_test - y_pred) ** 2)
        fold_results.append({
            "fold": split.fold,
            "train_size": len(X_train),
            "test_size": len(y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
        })

        if verbose:
            print(f"  RMSE: {np.sqrt(mse):.6f}")

    return np.array(all_preds), np.array(all_actuals), fold_results
