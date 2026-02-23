#!/usr/bin/env python3
"""Train GARCH(1,1) model with parallelism across tickers.

GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Usage:
    python scripts/models/train_garch.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


def qlike(y_true, y_pred):
    """QLIKE loss function."""
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def fit_garch_ticker(ticker, train_data, test_data, horizons=[1, 5, 22]):
    """Fit GARCH for one ticker, forecast for all horizons."""

    try:
        # Get returns for this ticker
        ticker_train = train_data[train_data['ticker'] == ticker].sort_values('date')
        ticker_test = test_data[test_data['ticker'] == ticker].sort_values('date')

        # Need returns — if not available, compute from rv or close
        if 'return' in ticker_train.columns:
            returns = ticker_train['return'].values * 100
        else:
            # Use sqrt(rv) as proxy for returns
            returns = np.sqrt(ticker_train['rv'].values) * 100

        # Fit GARCH(1,1)
        model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
        result = model.fit(disp='off', show_warning=False)

        # Params
        params = {
            'omega': result.params.get('omega', 0),
            'alpha': result.params.get('alpha[1]', 0),
            'beta': result.params.get('beta[1]', 0)
        }

        # Forecast for each horizon
        forecasts = {}
        for h in horizons:
            # Multi-step forecast
            fc = result.forecast(horizon=h, reindex=False)
            # Variance forecast (last train day → h days ahead)
            var_forecast = fc.variance.iloc[-1, h-1] / 10000  # back from *100
            forecasts[h] = var_forecast

        # Match with test actuals
        test_rv = ticker_test['rv'].values

        return {
            'ticker': ticker,
            'params': params,
            'forecasts': forecasts,
            'test_dates': ticker_test['date'].values,
            'test_rv': test_rv,
            'success': True
        }

    except Exception as e:
        return {
            'ticker': ticker,
            'error': str(e),
            'success': False
        }


if __name__ == '__main__':
    train = pd.read_parquet('data/prepared/train.parquet')
    test = pd.read_parquet('data/prepared/test.parquet')

    tickers = train['ticker'].unique()
    print(f'Training GARCH for {len(tickers)} tickers...')

    # Parallel across tickers
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(fit_garch_ticker)(t, train, test) for t in tickers
    )

    # Aggregate results
    Path('models/garch').mkdir(parents=True, exist_ok=True)
    Path('data/predictions/test_2019').mkdir(parents=True, exist_ok=True)

    # Save params
    params_list = []
    for r in results:
        if r['success']:
            params_list.append({'ticker': r['ticker'], **r['params']})
    pd.DataFrame(params_list).to_csv('models/garch/params.csv', index=False)

    # Create predictions for each h
    for h in [1, 5, 22]:
        preds = []
        for r in results:
            if r['success'] and h in r['forecasts']:
                # Replicate forecast for all test days (GARCH gives one forecast)
                n_days = len(r['test_dates'])
                pred_df = pd.DataFrame({
                    'date': r['test_dates'],
                    'ticker': r['ticker'],
                    'rv_actual': r['test_rv'],
                    'rv_pred': [r['forecasts'][h]] * n_days
                })
                preds.append(pred_df)

        if preds:
            all_preds = pd.concat(preds, ignore_index=True)
            all_preds.to_parquet(f'data/predictions/test_2019/garch_h{h}.parquet')

            q = qlike(all_preds['rv_actual'].values, all_preds['rv_pred'].values)
            print(f'H={h}: QLIKE = {q:.4f}')

    print('\nGARCH training complete!')
