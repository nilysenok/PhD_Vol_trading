#!/usr/bin/env python3
"""Full walk-forward validation pipeline with statistical tests and visualizations."""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
import os
import time
import scipy.stats
from pathlib import Path
from sklearn.linear_model import LinearRegression
import xgboost as xgb_lib
import lightgbm as lgb_lib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# UTILITIES
# ============================================================
def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def qlike_losses(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return y_true / y_pred - np.log(y_true / y_pred) - 1

def dm_test(loss1, loss2, h=1):
    """Diebold-Mariano test with Newey-West HAC standard errors."""
    d = loss1 - loss2
    T = len(d)
    d_mean = np.mean(d)
    lag = max(1, h - 1)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, lag + 1):
        if k < T:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += 2 * (1 - k / (lag + 1)) * gamma_k
    var_d = (gamma_0 + gamma_sum) / T
    dm_stat = d_mean / np.sqrt(max(var_d, 1e-20))
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def clean_features(X):
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)

# ============================================================
# STAGE 0: DATA EXPLORATION
# ============================================================
print('=' * 70)
print('STAGE 0: DATA EXPLORATION')
print('=' * 70)

train = pd.read_parquet('data/prepared/train.parquet')
val = pd.read_parquet('data/prepared/val.parquet')
test = pd.read_parquet('data/prepared/test.parquet')

with open('data/prepared/config.json') as f:
    config = json.load(f)
feature_cols = config['feature_cols']
har_base = config['har_features']

all_cols = train.columns.tolist()
jump_cols = [c for c in all_cols if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
harj_features = har_base + [c for c in jump_cols if c not in har_base]
harj_features = [f for f in harj_features if f in all_cols]

# Load walkforward data
wf_years = []
wf_data = {}
for year in range(2020, 2027):
    fp = f'data/prepared/walkforward_{year}.parquet'
    if os.path.exists(fp):
        df = pd.read_parquet(fp)
        if len(df) > 100:  # skip tiny files
            wf_data[year] = df
            wf_years.append(year)
            print(f'  WF {year}: {len(df)} rows, {df.date.min().date()} to {df.date.max().date()}, tickers={df.ticker.nunique()}')

# Build full dataset: train + val + test + walkforward
all_data = pd.concat([train, val, test] + [wf_data[y] for y in wf_years]).sort_values('date').reset_index(drop=True)
all_data['year'] = pd.to_datetime(all_data['date']).dt.year

print(f'\nFull dataset: {len(all_data)} rows')
print(f'Date range: {all_data.date.min()} to {all_data.date.max()}')
print(f'Years: {sorted(all_data.year.unique())}')
print(f'Tickers: {all_data.ticker.nunique()}')
print(f'Features: {len(feature_cols)}, HAR-J: {len(harj_features)}')

# Test years for walk-forward: 2017-2025
# (2026 has only 1 month - skip, need at least ~3 months)
test_years = [y for y in range(2017, 2026) if y in all_data.year.values]
print(f'\nWalk-forward test years: {test_years}')

# Create output directories
for d in ['data/predictions/walk_forward', 'results/tables', 'results/figures']:
    Path(d).mkdir(parents=True, exist_ok=True)

# Load hyperparameters
xgb_params = {}
lgb_params = {}
for h in [1, 5, 22]:
    with open(f'models/xgboost/params_h{h}.json') as f:
        xgb_params[h] = json.load(f)
    xgb_params[h].update({'tree_method': 'hist', 'verbosity': 0, 'random_state': 42, 'n_jobs': -1})

    with open(f'models/lightgbm/params_h{h}.json') as f:
        lgb_params[h] = json.load(f)
    lgb_params[h].update({'verbosity': -1, 'random_state': 42, 'n_jobs': -1})

best_ml = {1: 'xgboost', 5: 'lightgbm', 22: 'lightgbm'}

# ============================================================
# STAGE 1: WALK-FORWARD PREDICTIONS
# ============================================================
print('\n' + '=' * 70)
print('STAGE 1: WALK-FORWARD PREDICTIONS')
print('=' * 70)

# Storage for all predictions
all_preds = {}  # (model, h, strategy) -> list of DataFrames
hybrid_weights = []  # list of dicts

t0 = time.time()

for strategy in ['annual', 'quarterly']:
    print(f'\n--- Strategy: {strategy.upper()} ---')

    for test_year in test_years:
        if strategy == 'annual':
            periods = [(test_year, None)]  # None = full year
        else:
            periods = [(test_year, q) for q in range(1, 5)]

        for test_year_p, quarter in periods:
            if quarter is not None:
                # Quarterly: test = specific quarter
                test_mask = (all_data['year'] == test_year_p) & (pd.to_datetime(all_data['date']).dt.quarter == quarter)
                train_end_month = (quarter - 1) * 3  # months before this quarter
                if train_end_month == 0:
                    train_mask = all_data['year'] < test_year_p
                else:
                    q_start = pd.Timestamp(f'{test_year_p}-{(quarter-1)*3+1:02d}-01')
                    train_mask = pd.to_datetime(all_data['date']) < q_start
                label = f'{test_year_p}-Q{quarter}'
            else:
                test_mask = all_data['year'] == test_year_p
                train_mask = all_data['year'] < test_year_p
                label = str(test_year_p)

            test_data = all_data[test_mask]
            train_full = all_data[train_mask]

            if len(test_data) < 10 or len(train_full) < 100:
                continue

            # Split train into train_pure + val (val = last year of train)
            train_years_avail = sorted(train_full['year'].unique())
            if len(train_years_avail) < 2:
                continue
            val_year = train_years_avail[-1]
            val_mask_inner = train_full['year'] == val_year
            train_pure = train_full[~val_mask_inner]
            val_inner = train_full[val_mask_inner]

            for h in [1, 5, 22]:
                target = f'rv_target_h{h}'

                # Drop NaN targets
                tp = train_pure.dropna(subset=[target])
                vi = val_inner.dropna(subset=[target])
                td = test_data.dropna(subset=[target])

                if len(tp) < 50 or len(vi) < 10 or len(td) < 10:
                    continue

                y_tp_log = np.log(tp[target].values + 1e-10)
                y_vi_log = np.log(vi[target].values + 1e-10)
                y_vi = vi[target].values
                y_td = td[target].values

                X_tp_ml = clean_features(tp[feature_cols].values)
                X_vi_ml = clean_features(vi[feature_cols].values)
                X_td_ml = clean_features(td[feature_cols].values)

                X_tp_har = clean_features(np.log(tp[harj_features].values.clip(1e-10)))
                X_vi_har = clean_features(np.log(vi[harj_features].values.clip(1e-10)))
                X_td_har = clean_features(np.log(td[harj_features].values.clip(1e-10)))

                base_info = {'date': td['date'].values, 'ticker': td['ticker'].values,
                             'year': test_year_p, 'rv_actual': y_td}

                # --- HAR-J ---
                try:
                    har = LinearRegression()
                    har.fit(X_tp_har, y_tp_log)
                    har_td = np.clip(np.exp(har.predict(X_td_har)), 1e-10, None)
                    har_vi = np.clip(np.exp(har.predict(X_vi_har)), 1e-10, None)

                    key = ('har', h, strategy)
                    if key not in all_preds:
                        all_preds[key] = []
                    all_preds[key].append(pd.DataFrame({**base_info, 'rv_pred': har_td}))
                except Exception as e:
                    print(f'    ERROR HAR-J {label} H={h}: {e}')
                    har_td = None
                    har_vi = None

                # --- XGBoost ---
                try:
                    params = {k: v for k, v in xgb_params[h].items() if k != 'early_stopping_rounds'}
                    params['early_stopping_rounds'] = 50
                    xgb_m = xgb_lib.XGBRegressor(**params)
                    xgb_m.fit(X_tp_ml, y_tp_log, eval_set=[(X_vi_ml, y_vi_log)], verbose=False)
                    xgb_td = np.clip(np.exp(xgb_m.predict(X_td_ml)), 1e-10, None)
                    xgb_vi = np.clip(np.exp(xgb_m.predict(X_vi_ml)), 1e-10, None)

                    key = ('xgboost', h, strategy)
                    if key not in all_preds:
                        all_preds[key] = []
                    all_preds[key].append(pd.DataFrame({**base_info, 'rv_pred': xgb_td}))
                except Exception as e:
                    print(f'    ERROR XGB {label} H={h}: {e}')
                    xgb_td = None
                    xgb_vi = None

                # --- LightGBM ---
                try:
                    lgb_m = lgb_lib.LGBMRegressor(**lgb_params[h])
                    lgb_m.fit(X_tp_ml, y_tp_log, eval_set=[(X_vi_ml, y_vi_log)],
                              callbacks=[lgb_lib.early_stopping(50, verbose=False)])
                    lgb_td = np.clip(np.exp(lgb_m.predict(X_td_ml)), 1e-10, None)
                    lgb_vi = np.clip(np.exp(lgb_m.predict(X_vi_ml)), 1e-10, None)

                    key = ('lightgbm', h, strategy)
                    if key not in all_preds:
                        all_preds[key] = []
                    all_preds[key].append(pd.DataFrame({**base_info, 'rv_pred': lgb_td}))
                except Exception as e:
                    print(f'    ERROR LGB {label} H={h}: {e}')
                    lgb_td = None
                    lgb_vi = None

                # --- Hybrid ---
                try:
                    ml_name = best_ml[h]
                    ml_td = xgb_td if ml_name == 'xgboost' else lgb_td
                    ml_vi = xgb_vi if ml_name == 'xgboost' else lgb_vi

                    if ml_td is not None and har_td is not None and ml_vi is not None and har_vi is not None:
                        # Grid search w_har on val
                        best_w = 0
                        best_q = qlike(y_vi, ml_vi)
                        for w in np.arange(0, 0.61, 0.01):
                            blend = w * har_vi + (1 - w) * ml_vi
                            q = qlike(y_vi, np.clip(blend, 1e-10, None))
                            if q < best_q:
                                best_q = q
                                best_w = w

                        # Fallback: inverse QLIKE if w=0
                        if best_w == 0:
                            q_har_v = qlike(y_vi, har_vi)
                            q_ml_v = qlike(y_vi, ml_vi)
                            inv_har = 1.0 / (q_har_v + 1e-10)
                            inv_ml = 1.0 / (q_ml_v + 1e-10)
                            best_w = inv_har / (inv_har + inv_ml)
                            method = 'inverse_qlike'
                        else:
                            method = 'val_grid'

                        hybrid_td = np.clip(best_w * har_td + (1 - best_w) * ml_td, 1e-10, None)

                        key = ('hybrid', h, strategy)
                        if key not in all_preds:
                            all_preds[key] = []
                        all_preds[key].append(pd.DataFrame({**base_info, 'rv_pred': hybrid_td}))

                        hybrid_weights.append({
                            'year': test_year_p, 'quarter': quarter, 'horizon': h,
                            'strategy': strategy, 'w_har': best_w, 'method': method,
                            'ml_model': ml_name
                        })
                except Exception as e:
                    print(f'    ERROR Hybrid {label} H={h}: {e}')

            if strategy == 'annual':
                elapsed = time.time() - t0
                print(f'  {label}: done ({elapsed:.0f}s)')

    print(f'  Strategy {strategy} done in {time.time()-t0:.0f}s')

# Save predictions
print('\nSaving predictions...')
for (model, h, strategy), dfs in all_preds.items():
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(f'data/predictions/walk_forward/{model}_h{h}_{strategy}.parquet', index=False)

# Save hybrid weights
hw_df = pd.DataFrame(hybrid_weights)
hw_df.to_csv('results/tables/hybrid_weights.csv', index=False)
print(f'Saved {len(all_preds)} prediction files, {len(hybrid_weights)} weight records')

# ============================================================
# STAGE 2: QLIKE TABLES
# ============================================================
print('\n' + '=' * 70)
print('STAGE 2: QLIKE TABLES')
print('=' * 70)

models_list = ['har', 'xgboost', 'lightgbm', 'hybrid']
model_labels = {'har': 'HAR-J', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'hybrid': 'Hybrid'}

for h in [1, 5, 22]:
    print(f'\n--- H={h}: QLIKE by Year (annual) ---')
    rows = []
    for model in models_list:
        key = (model, h, 'annual')
        if key not in all_preds:
            continue
        combined = pd.concat(all_preds[key], ignore_index=True)
        row = {'Model': model_labels[model]}
        for year in test_years:
            yr_data = combined[combined['year'] == year]
            if len(yr_data) > 0:
                row[str(year)] = qlike(yr_data['rv_actual'].values, yr_data['rv_pred'].values)
        # Overall
        row['Mean'] = np.mean([v for k, v in row.items() if k != 'Model' and isinstance(v, float)])
        rows.append(row)

    df_table = pd.DataFrame(rows)
    df_table.to_csv(f'results/tables/qlike_by_year_h{h}.csv', index=False)

    # Print
    print(df_table.to_string(index=False, float_format='%.4f'))

    # Mark best per column
    for col in df_table.columns[1:]:
        vals = df_table[col].values
        best_idx = np.argmin(vals)
        print(f'  Best {col}: {df_table.iloc[best_idx]["Model"]}')

# Annual vs Quarterly
print('\n--- Annual vs Quarterly ---')
aq_rows = []
for model in models_list:
    for h in [1, 5, 22]:
        key_a = (model, h, 'annual')
        key_q = (model, h, 'quarterly')
        if key_a in all_preds and key_q in all_preds:
            ca = pd.concat(all_preds[key_a], ignore_index=True)
            cq = pd.concat(all_preds[key_q], ignore_index=True)
            qa = qlike(ca['rv_actual'].values, ca['rv_pred'].values)
            qq = qlike(cq['rv_actual'].values, cq['rv_pred'].values)
            aq_rows.append({'Model': model_labels[model], 'H': h, 'Annual': qa, 'Quarterly': qq,
                            'Diff%': (qq - qa) / qa * 100})

aq_df = pd.DataFrame(aq_rows)
aq_df.to_csv('results/tables/annual_vs_quarterly.csv', index=False)
print(aq_df.to_string(index=False, float_format='%.4f'))

# Hybrid weights table
print('\n--- Hybrid Weights by Year ---')
hw_annual = hw_df[hw_df['strategy'] == 'annual'][['year', 'horizon', 'w_har', 'method', 'ml_model']]
print(hw_annual.to_string(index=False))

# ============================================================
# STAGE 3: STATISTICAL TESTS
# ============================================================
print('\n' + '=' * 70)
print('STAGE 3: DIEBOLD-MARIANO TESTS')
print('=' * 70)

dm_results = []

for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')

    # Get annual predictions, merge all models
    preds_by_model = {}
    for model in models_list:
        key = (model, h, 'annual')
        if key in all_preds:
            preds_by_model[model] = pd.concat(all_preds[key], ignore_index=True)

    if len(preds_by_model) < 2:
        print('  Not enough models')
        continue

    # Merge all on date+ticker
    base = preds_by_model[models_list[0]].rename(columns={'rv_pred': f'pred_{models_list[0]}'})
    for model in models_list[1:]:
        if model in preds_by_model:
            df_m = preds_by_model[model][['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'pred_{model}'})
            base = base.merge(df_m, on=['date', 'ticker'], how='inner')

    actual = base['rv_actual'].values

    # Compute losses
    losses = {}
    for model in models_list:
        if f'pred_{model}' in base.columns:
            losses[model] = qlike_losses(actual, base[f'pred_{model}'].values)

    # DM tests
    pairs = [('hybrid', 'xgboost'), ('hybrid', 'lightgbm'), ('hybrid', 'har'),
             ('xgboost', 'lightgbm'), ('xgboost', 'har'), ('lightgbm', 'har')]

    print(f'  {"Pair":<30} {"DM_stat":>10} {"p-value":>10} {"Winner":>12}')
    print(f'  {"-"*62}')

    for m1, m2 in pairs:
        if m1 in losses and m2 in losses:
            stat, pval = dm_test(losses[m1], losses[m2], h=h)
            winner = model_labels[m1] if stat < 0 else model_labels[m2]
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            print(f'  {model_labels[m1]+" vs "+model_labels[m2]:<30} {stat:>10.3f} {pval:>10.4f} {winner:>10}{sig}')

            dm_results.append({
                'Horizon': h, 'Model1': model_labels[m1], 'Model2': model_labels[m2],
                'DM_stat': stat, 'p_value': pval, 'Winner': winner
            })

    # DM p-value matrix
    print(f'\n  P-value matrix (positive DM = model2 better):')
    mlist = [m for m in models_list if m in losses]
    header = f'  {"":>12}' + ''.join(f'{model_labels[m]:>12}' for m in mlist)
    print(header)
    for m1 in mlist:
        row = f'  {model_labels[m1]:>12}'
        for m2 in mlist:
            if m1 == m2:
                row += f'{"---":>12}'
            else:
                stat, pval = dm_test(losses[m1], losses[m2], h=h)
                row += f'{pval:>12.4f}'
        print(row)

dm_df = pd.DataFrame(dm_results)
for h in [1, 5, 22]:
    dm_h = dm_df[dm_df['Horizon'] == h]
    if len(dm_h) > 0:
        dm_h.to_csv(f'results/tables/dm_tests_h{h}.csv', index=False)

# ============================================================
# STAGE 4: VISUALIZATIONS
# ============================================================
print('\n' + '=' * 70)
print('STAGE 4: VISUALIZATIONS')
print('=' * 70)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

colors = {'HAR-J': '#1f77b4', 'XGBoost': '#ff7f0e', 'LightGBM': '#2ca02c', 'Hybrid': '#d62728'}

# --- FIGURE 1: QLIKE Comparison Bar Chart ---
print('  Fig 1: QLIKE comparison...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    means = []
    stds = []
    labels = []
    for model in models_list:
        key = (model, h, 'annual')
        if key not in all_preds:
            continue
        combined = pd.concat(all_preds[key], ignore_index=True)
        year_qlikes = []
        for year in test_years:
            yr = combined[combined['year'] == year]
            if len(yr) > 0:
                year_qlikes.append(qlike(yr['rv_actual'].values, yr['rv_pred'].values))
        means.append(np.mean(year_qlikes))
        stds.append(np.std(year_qlikes))
        labels.append(model_labels[model])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[l] for l in labels], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(f'H={h}', fontsize=14)
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)

fig.suptitle('Walk-Forward QLIKE (mean +/- std across years)', fontsize=14)
fig.tight_layout()
fig.savefig('results/figures/qlike_comparison.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/qlike_comparison.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 2: QLIKE by Year ---
print('  Fig 2: QLIKE by year...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    for model in models_list:
        key = (model, h, 'annual')
        if key not in all_preds:
            continue
        combined = pd.concat(all_preds[key], ignore_index=True)
        years_plot = []
        qlikes_plot = []
        for year in test_years:
            yr = combined[combined['year'] == year]
            if len(yr) > 0:
                years_plot.append(year)
                qlikes_plot.append(qlike(yr['rv_actual'].values, yr['rv_pred'].values))
        ax.plot(years_plot, qlikes_plot, 'o-', label=model_labels[model],
                color=colors[model_labels[model]], linewidth=2, markersize=6)

    # Crisis shading
    if 2020 in test_years:
        ax.axvspan(2019.8, 2020.2, alpha=0.15, color='gray', label='COVID-19')
    if 2022 in test_years:
        ax.axvspan(2021.8, 2022.2, alpha=0.15, color='red', label='Sanctions')

    ax.set_title(f'H={h}', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)
    ax.legend(fontsize=10)

fig.suptitle('Walk-Forward QLIKE by Year', fontsize=14)
fig.tight_layout()
fig.savefig('results/figures/qlike_by_year.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/qlike_by_year.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 3: Actual vs Predicted ---
print('  Fig 3: Actual vs predicted...')
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
# Use 2019 (test year) and SBER ticker
ticker_plot = 'SBER'
year_plot = 2019

for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    key = ('hybrid', h, 'annual')
    if key in all_preds:
        combined = pd.concat(all_preds[key], ignore_index=True)
        mask = (combined['year'] == year_plot) & (combined['ticker'] == ticker_plot)
        sub = combined[mask].sort_values('date')
        if len(sub) > 0:
            dates = pd.to_datetime(sub['date'])
            ax.plot(dates, sub['rv_actual'], 'k-', alpha=0.7, linewidth=1, label='Actual RV')
            ax.plot(dates, sub['rv_pred'], 'r-', alpha=0.8, linewidth=1.5, label='Hybrid Pred')
            ax.set_title(f'{ticker_plot} — H={h} ({year_plot})', fontsize=13)
            ax.legend(fontsize=10)
            ax.set_ylabel('Realized Variance', fontsize=11)

fig.tight_layout()
fig.savefig('results/figures/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/actual_vs_predicted.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 4: Hybrid Weights Over Time ---
print('  Fig 4: Hybrid weights...')
fig, ax = plt.subplots(figsize=(10, 6))
hw_annual = hw_df[hw_df['strategy'] == 'annual']
for h in [1, 5, 22]:
    sub = hw_annual[hw_annual['horizon'] == h].sort_values('year')
    if len(sub) > 0:
        ax.plot(sub['year'], sub['w_har'], 'o-', label=f'H={h}', linewidth=2, markersize=8)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('w_har (HAR-J weight)', fontsize=12)
ax.set_title('Hybrid Weights Over Time (Annual Walk-Forward)', fontsize=14)
ax.legend(fontsize=12)
ax.set_ylim(-0.05, 0.65)
fig.tight_layout()
fig.savefig('results/figures/hybrid_weights.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/hybrid_weights.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 5: Feature Importance ---
print('  Fig 5: Feature importance...')
try:
    xgb_model = xgb_lib.Booster()
    xgb_model.load_model('models/xgboost/model_h1.json')
    importance = xgb_model.get_score(importance_type='gain')
    # Map f0, f1... to feature names
    imp_named = {}
    for k, v in importance.items():
        idx_num = int(k.replace('f', ''))
        if idx_num < len(feature_cols):
            imp_named[feature_cols[idx_num]] = v
    top20 = sorted(imp_named.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [x[0] for x in reversed(top20)]
    vals = [x[1] for x in reversed(top20)]
    ax.barh(range(len(names)), vals, color='#2ca02c', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Gain', fontsize=12)
    ax.set_title('XGBoost Feature Importance (H=1, Top 20)', fontsize=14)
    fig.tight_layout()
    fig.savefig('results/figures/feature_importance.png', dpi=150, bbox_inches='tight')
    fig.savefig('results/figures/feature_importance.pdf', bbox_inches='tight')
    plt.close(fig)
except Exception as e:
    print(f'    Feature importance error: {e}')

# --- FIGURE 6: Model Rankings Heatmap ---
print('  Fig 6: Model rankings...')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    rank_data = []
    for year in test_years:
        year_qlikes = {}
        for model in models_list:
            key = (model, h, 'annual')
            if key in all_preds:
                combined = pd.concat(all_preds[key], ignore_index=True)
                yr = combined[combined['year'] == year]
                if len(yr) > 0:
                    year_qlikes[model_labels[model]] = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
        if year_qlikes:
            sorted_models = sorted(year_qlikes.items(), key=lambda x: x[1])
            ranks = {m: r + 1 for r, (m, _) in enumerate(sorted_models)}
            rank_data.append({'Year': year, **ranks})

    if rank_data:
        rank_df = pd.DataFrame(rank_data).set_index('Year')
        im = ax.imshow(rank_df.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=4)
        ax.set_xticks(range(len(rank_df.columns)))
        ax.set_xticklabels(rank_df.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(rank_df.index)))
        ax.set_yticklabels(rank_df.index, fontsize=10)
        ax.set_title(f'H={h}', fontsize=14)
        # Add text annotations
        for i in range(len(rank_df.index)):
            for j in range(len(rank_df.columns)):
                ax.text(j, i, f'{int(rank_df.values[i, j])}', ha='center', va='center',
                        fontsize=12, fontweight='bold')

fig.suptitle('Model Rankings by Year (1=Best, 4=Worst)', fontsize=14)
fig.colorbar(im, ax=axes, shrink=0.6)
fig.tight_layout()
fig.savefig('results/figures/model_rankings.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/model_rankings.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 7: Error Correlation ---
print('  Fig 7: Error correlation...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    preds_by_model = {}
    for model in models_list:
        key = (model, h, 'annual')
        if key in all_preds:
            preds_by_model[model] = pd.concat(all_preds[key], ignore_index=True)

    if len(preds_by_model) < 2:
        continue

    # Merge
    base = preds_by_model[models_list[0]].rename(columns={'rv_pred': f'pred_{models_list[0]}'})
    for model in models_list[1:]:
        if model in preds_by_model:
            df_m = preds_by_model[model][['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'pred_{model}'})
            base = base.merge(df_m, on=['date', 'ticker'], how='inner')

    actual = base['rv_actual'].values
    mlist = [m for m in models_list if f'pred_{m}' in base.columns]
    errors = {model_labels[m]: qlike_losses(actual, base[f'pred_{m}'].values) for m in mlist}

    labels_corr = list(errors.keys())
    corr_matrix = np.zeros((len(labels_corr), len(labels_corr)))
    for i, l1 in enumerate(labels_corr):
        for j, l2 in enumerate(labels_corr):
            corr_matrix[i, j] = np.corrcoef(errors[l1], errors[l2])[0, 1]

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(labels_corr)))
    ax.set_xticklabels(labels_corr, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(labels_corr)))
    ax.set_yticklabels(labels_corr, fontsize=10)
    ax.set_title(f'H={h}', fontsize=14)
    for i in range(len(labels_corr)):
        for j in range(len(labels_corr)):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=11)

fig.suptitle('Error Correlation (QLIKE losses)', fontsize=14)
fig.colorbar(im, ax=axes, shrink=0.6)
fig.tight_layout()
fig.savefig('results/figures/error_correlation.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/error_correlation.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n' + '=' * 70)
print('FINAL SUMMARY')
print('=' * 70)

print('\n1. Walk-Forward QLIKE (Annual, all years):')
for h in [1, 5, 22]:
    print(f'\n  H={h}:')
    for model in models_list:
        key = (model, h, 'annual')
        if key in all_preds:
            combined = pd.concat(all_preds[key], ignore_index=True)
            q = qlike(combined['rv_actual'].values, combined['rv_pred'].values)
            print(f'    {model_labels[model]:>12}: {q:.4f}')

print('\n2. DM Test Summary (Hybrid vs others):')
for _, row in dm_df[dm_df['Model1'] == 'Hybrid'].iterrows():
    sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.1 else 'ns'
    print(f'  H={row["Horizon"]}: Hybrid vs {row["Model2"]}: DM={row["DM_stat"]:.3f}, p={row["p_value"]:.4f} {sig}')

print('\n3. Saved files:')
for d in ['data/predictions/walk_forward', 'results/tables', 'results/figures']:
    files = sorted(os.listdir(d))
    print(f'  {d}/: {len(files)} files')
    for f in files:
        print(f'    {f}')

print(f'\nTotal time: {time.time()-t0:.0f}s')
print('DONE!')
