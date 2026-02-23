"""Volatility forecasting models."""

from .base import BaseVolatilityModel, TimeSeriesModel, EnsembleModel
from .har import HARModel, HARExtendedModel, prepare_har_data

# Optional imports - handle missing dependencies gracefully
try:
    from .garch import GARCHModel
except ImportError:
    GARCHModel = None

try:
    from .lightgbm_model import LightGBMModel
except (ImportError, OSError):
    LightGBMModel = None

try:
    from .xgboost_model import XGBoostModel
except (ImportError, OSError, Exception):
    XGBoostModel = None

try:
    from .gru_model import GRUModel
except ImportError:
    GRUModel = None

try:
    from .lstm_model import LSTMModel
except ImportError:
    LSTMModel = None

try:
    from .hybrid import HybridModel, GARCHMLHybrid, create_ensemble
except ImportError:
    HybridModel = None
    GARCHMLHybrid = None
    create_ensemble = None

__all__ = [
    "BaseVolatilityModel",
    "TimeSeriesModel",
    "EnsembleModel",
    "HARModel",
    "HARExtendedModel",
    "prepare_har_data",
    "GARCHModel",
    "LightGBMModel",
    "XGBoostModel",
    "GRUModel",
    "LSTMModel",
    "HybridModel",
    "GARCHMLHybrid",
    "create_ensemble",
]
