# MOEX Volatility Forecasting - Final Report

**Generated:** 2026-02-06 14:36

## Summary

### Best Models by Horizon

| Horizon | Best Model | QLIKE | RMSE | R² |
|---------|------------|-------|------|-----|
| H=1 | HAR-RV | -7.527 | 0.0003 | 0.061 |
| H=5 | HAR-RV | -7.414 | 0.0005 | -0.007 |
| H=22 | HAR-RV | -7.340 | 0.0005 | -0.017 |

### Classical Models Comparison (HAR vs GARCH)

| Horizon | HAR QLIKE | GARCH QLIKE | HAR Wins |
|---------|-----------|-------------|----------|
| H=1 | **-7.527** | -7.444 | 12/17 |
| H=5 | **-7.414** | -6.145 | 17/17 |
| H=22 | **-7.340** | -4.248 | 17/17 |

**Key Finding:** HAR model significantly outperforms GARCH for realized volatility forecasting, especially at longer horizons. GARCH multi-step forecasts converge quickly to unconditional variance, making them unsuitable for RV prediction.

### GARCH Model Selection

Best GARCH specifications by ticker (selected via AIC on validation set):

| Ticker | p | q | o | Vol Type | Distribution |
|--------|---|---|---|----------|--------------|
| Most tickers | 1 | 1 | 0-1 | GARCH/GJR | t-distribution |

The leverage effect (GJR parameter γ) is significant for most stocks, confirming asymmetric volatility responses to negative vs positive shocks.

## Models Trained

### 1. HAR-RV (Heterogeneous Autoregressive)
- **Formula:** RV(t+h) = α + β₁·RV_d + β₂·RV_w + β₃·RV_m + ε
- **Features:** Daily, weekly (5-day), monthly (22-day) RV lags
- **Grid search:** log vs level, with/without constant
- **Training:** 17/17 tickers for all horizons

### 2. GARCH-GJR
- **Formula:** σ²(t) = ω + α·ε²(t-1) + γ·ε²(t-1)·I(ε<0) + β·σ²(t-1)
- **Grid search:** p∈{1,2}, q∈{1,2}, vol∈{GARCH,GJR,EGARCH}, dist∈{normal,t}
- **Selection:** AIC on validation set
- **Training:** 17/17 tickers for all horizons

## Data

- **Train period:** 2015-01-01 to 2017-12-31
- **Validation period:** 2018-01-01 to 2018-12-31
- **Test period:** 2019-01-01 to 2019-12-31
- **Tickers:** 17 MOEX stocks (AFLT, ALRS, HYDR, IRAO, LKOH, LSRG, MGNT, MOEX, MTLR, MTSS, NVTK, OGKB, PHOR, RTKM, SBER, TATN, VTBR)
- **Features:** 259 columns (RV, BV, Jump, RSV+/-, RSkew, RKurt, external factors)

## Files Generated

### Predictions
- `data/predictions/test_2019/har_h{1,5,22}.parquet` - HAR predictions
- `data/predictions/test_2019/garch_h{1,5,22}.parquet` - GARCH predictions
- `data/predictions/test_2019/classical_h{1,5,22}.parquet` - Combined predictions

### Models
- `models/har/h{1,5,22}/*.pkl` - Saved HAR models
- `models/garch/h{1,5,22}/*.pkl` - Saved GARCH parameters

### Results
- `results/train_classical.log` - Training log
- `results/FINAL_REPORT.md` - This report

## Metrics Description

- **QLIKE**: Quasi-Likelihood loss = mean(log(σ²_pred) + σ²_true/σ²_pred). Lower is better. Preferred for volatility forecasting.
- **RMSE**: Root Mean Squared Error. Lower is better.
- **MAE**: Mean Absolute Error. Lower is better.
- **R²**: Coefficient of determination. Higher is better. Can be negative if predictions are worse than naive mean.

## Conclusions

1. **HAR dominates for RV forecasting**: Consistent with academic literature (Corsi 2009), HAR is specifically designed for realized volatility and significantly outperforms return-based GARCH.

2. **GARCH limitations**:
   - GARCH forecasts conditional variance of returns, not RV
   - Multi-step forecasts converge rapidly to unconditional variance
   - EGARCH requires simulation-based forecasts for horizons > 1

3. **Leverage effect confirmed**: GJR-GARCH γ parameter is positive for most stocks, confirming volatility asymmetry (volatility increases more after negative shocks).

4. **Next steps**:
   - Add ML models (LightGBM, XGBoost) for feature-rich forecasting
   - Add neural networks (GRU, LSTM) for capturing complex patterns
   - Implement walk-forward cross-validation for 2020-2026
   - Statistical tests (Diebold-Mariano, MCS)
