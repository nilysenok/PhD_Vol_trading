# MOEX Volatility Forecasting

Volatility forecasting framework for Russian equities using multiple modeling approaches.

## Project Structure

```
moex_volatility/
├── configs/
│   ├── model_config.yaml      # Model hyperparameters
│   └── training_config.yaml   # Training settings
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading utilities
│   │   └── features.py        # Feature engineering
│   ├── models/
│   │   ├── base.py            # Base model classes
│   │   ├── har.py             # HAR model
│   │   ├── garch.py           # GARCH models
│   │   ├── lightgbm_model.py  # LightGBM
│   │   ├── xgboost_model.py   # XGBoost
│   │   ├── gru_model.py       # GRU neural network
│   │   ├── lstm_model.py      # LSTM neural network
│   │   └── hybrid.py          # Hybrid/ensemble models
│   ├── training/
│   │   ├── trainer.py         # Training utilities
│   │   └── walk_forward.py    # Walk-forward CV
│   ├── evaluation/
│   │   ├── metrics.py         # Forecast metrics
│   │   └── statistical_tests.py # Statistical tests
│   └── utils/
│       ├── config.py          # Configuration loader
│       └── logger.py          # Logging utilities
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Training scripts
├── data/                       # Data directory
├── models/                     # Saved models
├── results/                    # Results and figures
└── tests/                      # Unit tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Models

### Econometric Models

- **HAR** (Heterogeneous Autoregressive): Captures long memory through daily, weekly, and monthly RV components
- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity models

### Machine Learning Models

- **LightGBM**: Gradient boosting with leaf-wise tree growth
- **XGBoost**: Gradient boosting with level-wise tree growth

### Deep Learning Models

- **GRU**: Gated Recurrent Units
- **LSTM**: Long Short-Term Memory networks

### Hybrid Models

- **GARCH-ML**: GARCH features combined with ML models
- **Stacking**: Meta-learning ensemble

## Usage

### Quick Start

```python
from src.data import DataLoader, create_features_for_ml
from src.models import HARModel, LightGBMModel
from src.training import WalkForwardCV, Trainer
from src.evaluation import evaluate_forecast

# Load data
loader = DataLoader("../moex_discovery/data/dataset_final")
df = loader.load_master_long()

# Create features
from src.data.features import FeatureEngineer
fe = FeatureEngineer(df)
fe.add_har_features()
fe.add_lags("rv_daily", [1, 2, 3, 5, 10, 22])
fe.shift_target("rv_daily", horizon=1)
df = fe.get_dataframe()

# Train HAR model
har = HARModel(use_log=True)
X = df[["rv_d", "rv_w", "rv_m"]]
y = df["target_h1"]
har.fit(X, y)

# Walk-forward CV
cv = WalkForwardCV(
    initial_train_size=756,  # 3 years
    step_size=63,            # 3 months
    test_size=63,            # 3 months
    expanding=True
)

trainer = Trainer(LightGBMModel, model_params={"n_estimators": 500})
results = trainer.train_walk_forward(X, y, cv, ticker="SBER")
```

### Evaluation

```python
from src.evaluation import evaluate_forecast, diebold_mariano_test

# Metrics
metrics = evaluate_forecast(y_true, y_pred)
print(metrics)

# Statistical tests
dm_test = diebold_mariano_test(y_true, pred_har, pred_lgbm)
print(dm_test)
```

## Data

The framework uses data from `moex_discovery/data/dataset_final/`:

- **01_stocks/**: Stock RV and 10-min candles
- **02_external/**: MOEX indices, Yahoo Finance, macro factors
- **03_master/**: Combined datasets

### Features

- Realized Volatility (RV) from 10-minute returns
- HAR components (daily, weekly, monthly)
- External factors (VIX, S&P500, Brent, Gold)
- Macro indicators (key rate, CPI, Fed rate)

## Metrics

- **MSE/RMSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **QLIKE**: Quasi-likelihood loss
- **R²**: Coefficient of determination

## Statistical Tests

- **Mincer-Zarnowitz**: Forecast efficiency
- **Diebold-Mariano**: Comparing forecast accuracy
- **Giacomini-White**: Conditional predictive ability
- **Model Confidence Set**: Identify best models

## References

1. Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
2. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*.
3. Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
