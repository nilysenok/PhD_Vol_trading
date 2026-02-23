#!/usr/bin/env python3
"""Adaptive hybrid v3: trimmed ensemble + rolling inverse error weighting."""

import pandas as pd
import numpy as np
import warnings
import os
import time
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# UTILITIES
# ============================================================
def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def qlike_per_obs(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return y_true / y_pred - np.log(y_true / y_pred) - 1

def dm_test(loss1, loss2, h_horizon=1):
    mask = np.isfinite(loss1) & np.isfinite(loss2)
    loss1, loss2 = loss1[mask], loss2[mask]
    d = loss1 - loss2
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    d_mean = np.mean(d)
    lag = max(1, h_horizon - 1)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, lag + 1):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 1 else 0
        gamma_sum += 2 * (1 - k / (lag + 1)) * gamma_k
    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def sig_stars(p):
    if not np.isfinite(p):
        return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''

# ============================================================
# LOAD DATA
# ============================================================
print('=' * 70)
print('LOADING WALK-FORWARD PREDICTIONS')
print('=' * 70)

preds = {}
for model in ['har', 'xgboost', 'lightgbm']:
    for h in [1, 5, 22]:
        df = pd.read_parquet(f'data/predictions/walk_forward/{model}_h{h}_annual.parquet')
        df = df[df['rv_actual'] > 1e-12].copy()
        preds[(model, h)] = df
        print(f'  {model} H={h}: {len(df)} rows')

Path('data/predictions/walk_forward').mkdir(parents=True, exist_ok=True)
Path('results/tables').mkdir(parents=True, exist_ok=True)
Path('results/figures').mkdir(parents=True, exist_ok=True)

# Merge all 3 models
merged = {}
for h in [1, 5, 22]:
    df = preds[('har', h)][['date', 'ticker', 'year', 'rv_actual', 'rv_pred']].rename(
        columns={'rv_pred': 'har'})
    df = df.merge(preds[('xgboost', h)][['date', 'ticker', 'rv_pred']].rename(
        columns={'rv_pred': 'xgb'}), on=['date', 'ticker'])
    df = df.merge(preds[('lightgbm', h)][['date', 'ticker', 'rv_pred']].rename(
        columns={'rv_pred': 'lgb'}), on=['date', 'ticker'])
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    merged[h] = df
    print(f'  Merged H={h}: {len(df)} rows, years {int(df.year.min())}-{int(df.year.max())}')

years = sorted(merged[1]['year'].unique())
print(f'\nYears: {years}')

# ============================================================
# STRATEGY V6: TRIMMED ENSEMBLE
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY V6: TRIMMED ENSEMBLE')
print('=' * 70)

def trimmed_ensemble(har, xgb, lgb):
    """Drop the most deviant model per observation, average remaining two."""
    stack = np.column_stack([har, xgb, lgb])
    med = np.median(stack, axis=1, keepdims=True)
    dev = np.abs(stack - med) / np.clip(med, 1e-10, None)
    worst = np.argmax(dev, axis=1)
    mask = np.ones_like(stack)
    mask[np.arange(len(worst)), worst] = 0
    result = (stack * mask).sum(axis=1) / mask.sum(axis=1)
    return np.clip(result, 1e-10, None)

def trimmed_ensemble_weighted(har, xgb, lgb):
    """Drop the most deviant, weight remaining two by inverse deviation."""
    stack = np.column_stack([har, xgb, lgb])
    med = np.median(stack, axis=1, keepdims=True)
    dev = np.abs(stack - med) / np.clip(med, 1e-10, None)
    worst = np.argmax(dev, axis=1)

    mask = np.ones_like(stack)
    mask[np.arange(len(worst)), worst] = 0

    # Inverse deviation weights for remaining two
    inv_dev = 1.0 / np.clip(dev, 1e-8, None)
    inv_dev = inv_dev * mask
    w = inv_dev / inv_dev.sum(axis=1, keepdims=True)

    result = (stack * w).sum(axis=1)
    return np.clip(result, 1e-10, None)

v6_preds = {}
v6w_preds = {}

for h in [1, 5, 22]:
    df = merged[h].copy()

    # Simple trimmed
    df['v6_pred'] = trimmed_ensemble(df['har'].values, df['xgb'].values, df['lgb'].values)
    v6_preds[h] = df[['date', 'ticker', 'year', 'rv_actual', 'v6_pred']].rename(
        columns={'v6_pred': 'rv_pred'}).copy()

    # Weighted trimmed
    df['v6w_pred'] = trimmed_ensemble_weighted(df['har'].values, df['xgb'].values, df['lgb'].values)
    v6w_preds[h] = df[['date', 'ticker', 'year', 'rv_actual', 'v6w_pred']].rename(
        columns={'v6w_pred': 'rv_pred'}).copy()

    # Which model gets dropped most often?
    stack = np.column_stack([df['har'].values, df['xgb'].values, df['lgb'].values])
    med = np.median(stack, axis=1, keepdims=True)
    dev = np.abs(stack - med) / np.clip(med, 1e-10, None)
    worst = np.argmax(dev, axis=1)
    model_names = ['HAR-J', 'XGBoost', 'LightGBM']
    drop_counts = {m: np.sum(worst == i) for i, m in enumerate(model_names)}

    print(f'\n  H={h}: Dropped model counts: {drop_counts}')

    for year in years:
        yr = df[df['year'] == year]
        q_trim = qlike(yr['rv_actual'].values, yr['v6_pred'].values)
        q_trim_w = qlike(yr['rv_actual'].values, yr['v6w_pred'].values)
        q_har = qlike(yr['rv_actual'].values, yr['har'].values)
        q_xgb = qlike(yr['rv_actual'].values, yr['xgb'].values)
        q_lgb = qlike(yr['rv_actual'].values, yr['lgb'].values)

        # Year-level drop stats
        yr_worst = worst[df['year'] == year]
        yr_drops = {m: np.sum(yr_worst == i) for i, m in enumerate(model_names)}
        top_drop = max(yr_drops, key=yr_drops.get)

        print(f'    {year}: Trim={q_trim:.4f} TrimW={q_trim_w:.4f} | '
              f'HAR={q_har:.4f} XGB={q_xgb:.4f} LGB={q_lgb:.4f} | drop:{top_drop}({yr_drops[top_drop]})')

# ============================================================
# STRATEGY V7: ROLLING INVERSE ERROR WEIGHTING
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY V7: ROLLING INVERSE QLIKE WEIGHTING')
print('=' * 70)

windows = [20, 40, 60, 90, 120]
v7_preds = {}  # (h, window) -> DataFrame

for h in [1, 5, 22]:
    df = merged[h].copy()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Pre-compute per-obs QLIKE components for each model
    comps = {}
    for name in ['har', 'xgb', 'lgb']:
        pred_c = np.clip(df[name].values, 1e-10, None)
        comps[name] = df['rv_actual'].values / pred_c - np.log(df['rv_actual'].values / pred_c) - 1

    tickers = df['ticker'].unique()

    for window in windows:
        t0 = time.time()
        result = np.zeros(len(df))

        for ticker in tickers:
            mask = (df['ticker'] == ticker).values
            idx = np.where(mask)[0]
            n = len(idx)

            # Per-ticker predictions
            p = {name: df[name].values[idx] for name in ['har', 'xgb', 'lgb']}

            # Per-ticker QLIKE components + cumsum
            cs = {}
            for name in ['har', 'xgb', 'lgb']:
                c = comps[name][idx]
                cs[name] = np.cumsum(c)

            for i in range(n):
                if i < window:
                    result[idx[i]] = (p['har'][i] + p['xgb'][i] + p['lgb'][i]) / 3
                else:
                    weights = {}
                    for name in ['har', 'xgb', 'lgb']:
                        if i - window - 1 >= 0:
                            roll_q = (cs[name][i - 1] - cs[name][i - window - 1]) / window
                        else:
                            roll_q = cs[name][i - 1] / i
                        weights[name] = 1.0 / max(roll_q, 1e-6)

                    total = sum(weights.values())
                    result[idx[i]] = sum(
                        (weights[name] / total) * p[name][i] for name in ['har', 'xgb', 'lgb'])

        result = np.clip(result, 1e-10, None)
        v7_df = df[['date', 'ticker', 'year', 'rv_actual']].copy()
        v7_df['rv_pred'] = result
        v7_preds[(h, window)] = v7_df

        elapsed = time.time() - t0
        q_overall = qlike(v7_df['rv_actual'].values, v7_df['rv_pred'].values)
        print(f'  H={h} W={window:>3}: QLIKE={q_overall:.4f} ({elapsed:.1f}s)')

        for year in years:
            yr = v7_df[v7_df['year'] == year]
            q_yr = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
            print(f'    {year}: {q_yr:.4f}')

# ============================================================
# ALSO COMPUTE BASELINES
# ============================================================
print('\n--- Computing baselines ---')

# V1_Adaptive
v1_preds = {}
for h in [1, 5, 22]:
    fp = f'data/predictions/walk_forward/hybrid_adaptive_h{h}_annual.parquet'
    if os.path.exists(fp):
        df = pd.read_parquet(fp)
        df = df[df['rv_actual'] > 1e-12].copy()
        v1_preds[h] = df

# V2_MultiVal
v2_preds = {}
for h in [1, 5, 22]:
    fp = f'data/predictions/walk_forward/hybrid_v2_h{h}_annual.parquet'
    if os.path.exists(fp):
        df = pd.read_parquet(fp)
        df = df[df['rv_actual'] > 1e-12].copy()
        v2_preds[h] = df

# BestSingle
best_single_preds = {}
for h in [1, 5, 22]:
    all_bs = []
    for test_year in years:
        val_year = test_year - 1
        candidates = {}
        for model in ['har', 'xgboost', 'lightgbm']:
            val_data = preds[(model, h)]
            val_data = val_data[val_data['year'] == val_year]
            if len(val_data) > 0:
                candidates[model] = qlike(val_data['rv_actual'].values, val_data['rv_pred'].values)
        best_model = min(candidates, key=candidates.get) if candidates else 'har'
        test_data = preds[(best_model, h)]
        test_data = test_data[test_data['year'] == test_year]
        all_bs.append(test_data[['date', 'ticker', 'year', 'rv_actual', 'rv_pred']].copy())
    best_single_preds[h] = pd.concat(all_bs, ignore_index=True)

# SimpleAvg
simple_avg_preds = {}
for h in [1, 5, 22]:
    df = merged[h].copy()
    df['rv_pred'] = np.clip((df['har'] + df['xgb'] + df['lgb']) / 3, 1e-10, None)
    simple_avg_preds[h] = df[['date', 'ticker', 'year', 'rv_actual', 'rv_pred']].copy()

# ============================================================
# ROLLING WINDOW SENSITIVITY TABLE
# ============================================================
print('\n' + '=' * 70)
print('ROLLING WINDOW SENSITIVITY')
print('=' * 70)

roll_rows = []
for window in windows:
    row = {'window': window}
    for h in [1, 5, 22]:
        df = v7_preds[(h, window)]
        row[f'H={h}'] = qlike(df['rv_actual'].values, df['rv_pred'].values)
    roll_rows.append(row)
    print(f'  W={window:>3}: H=1={row["H=1"]:.4f}  H=5={row["H=5"]:.4f}  H=22={row["H=22"]:.4f}')

roll_df = pd.DataFrame(roll_rows)
roll_df.to_csv('results/tables/rolling_sensitivity.csv', index=False)

# Best window per horizon
best_windows = {}
for h_col in ['H=1', 'H=5', 'H=22']:
    best_idx = roll_df[h_col].idxmin()
    best_windows[h_col] = int(roll_df.loc[best_idx, 'window'])
    print(f'  Best window for {h_col}: {best_windows[h_col]}')

# ============================================================
# COMPREHENSIVE COMPARISON TABLE
# ============================================================
print('\n' + '=' * 70)
print('COMPREHENSIVE COMPARISON — ALL STRATEGIES')
print('=' * 70)

# Pick best rolling window per horizon for V7
v7_best = {}
for h in [1, 5, 22]:
    best_w = best_windows[f'H={h}']
    v7_best[h] = v7_preds[(h, best_w)]

all_strategies = {
    'HAR-J': {h: preds[('har', h)] for h in [1, 5, 22]},
    'XGBoost': {h: preds[('xgboost', h)] for h in [1, 5, 22]},
    'LightGBM': {h: preds[('lightgbm', h)] for h in [1, 5, 22]},
    'SimpleAvg': simple_avg_preds,
    'BestSingle': best_single_preds,
    'V1_Adaptive': v1_preds,
    'V6_Trimmed': v6_preds,
    'V6_TrimmedW': v6w_preds,
    'V7_Rolling': v7_best,
}

if v2_preds:
    all_strategies['V2_MultiVal'] = v2_preds

strat_rows = []
for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')
    snames = list(all_strategies.keys())
    header = f'  {"Year":>6}' + ''.join(f' {s:>12}' for s in snames)
    print(header)
    print('  ' + '-' * (6 + len(snames) * 13))

    for year in years:
        row = {'year': year, 'horizon': h}
        for sname, spreds in all_strategies.items():
            if h in spreds:
                yr = spreds[h]
                yr = yr[yr['year'] == year]
                if len(yr) > 0:
                    row[sname] = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                else:
                    row[sname] = np.nan
            else:
                row[sname] = np.nan
        strat_rows.append(row)

        line = f'  {year:>6}'
        vals_row = [row.get(s, np.nan) for s in snames]
        finite_vals = [v for v in vals_row if np.isfinite(v)]
        best_val = min(finite_vals) if finite_vals else np.inf
        for s in snames:
            v = row.get(s, np.nan)
            if np.isfinite(v):
                mark = '*' if abs(v - best_val) < 1e-6 else ' '
                line += f' {v:>11.4f}{mark}'
            else:
                line += f' {"N/A":>12}'
        print(line)

    # Mean row
    mean_vals = {}
    for s in snames:
        vals = [r[s] for r in strat_rows if r['horizon'] == h and np.isfinite(r.get(s, np.nan))]
        mean_vals[s] = np.mean(vals) if vals else np.nan

    line = f'  {"Mean":>6}'
    finite_means = [v for v in mean_vals.values() if np.isfinite(v)]
    best_mean = min(finite_means) if finite_means else np.inf
    for s in snames:
        v = mean_vals.get(s, np.nan)
        if np.isfinite(v):
            mark = '*' if abs(v - best_mean) < 1e-6 else ' '
            line += f' {v:>11.4f}{mark}'
        else:
            line += f' {"N/A":>12}'
    print(line)

    best_s = min(mean_vals, key=lambda k: mean_vals[k] if np.isfinite(mean_vals[k]) else 999)
    print(f'  >>> Best: {best_s} = {mean_vals[best_s]:.4f}')

strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv('results/tables/strategies_v3.csv', index=False)

# ============================================================
# FINAL SUMMARY TABLE
# ============================================================
print('\n' + '=' * 70)
print('FINAL SUMMARY')
print('=' * 70)

summary_rows = []
for sname, spreds in all_strategies.items():
    row = {'Model': sname}
    for h in [1, 5, 22]:
        if h in spreds:
            df = spreds[h]
            row[f'H={h}'] = qlike(df['rv_actual'].values, df['rv_pred'].values)
        else:
            row[f'H={h}'] = np.nan
    summary_rows.append(row)

# Add V7 with different windows
for window in windows:
    row = {'Model': f'V7_W{window}'}
    for h in [1, 5, 22]:
        df = v7_preds[(h, window)]
        row[f'H={h}'] = qlike(df['rv_actual'].values, df['rv_pred'].values)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/tables/final_best.csv', index=False)

print(f'\n  {"Model":<16} {"H=1":>10} {"H=5":>10} {"H=22":>10}')
print('  ' + '-' * 46)
for _, row in summary_df.iterrows():
    h1 = f'{row["H=1"]:.4f}' if np.isfinite(row["H=1"]) else 'N/A'
    h5 = f'{row["H=5"]:.4f}' if np.isfinite(row["H=5"]) else 'N/A'
    h22 = f'{row["H=22"]:.4f}' if np.isfinite(row["H=22"]) else 'N/A'
    print(f'  {row["Model"]:<16} {h1:>10} {h5:>10} {h22:>10}')

for col in ['H=1', 'H=5', 'H=22']:
    best_idx = summary_df[col].idxmin()
    print(f'\n  Best {col}: {summary_df.loc[best_idx, "Model"]} = {summary_df.loc[best_idx, col]:.4f}')

# ============================================================
# SAVE BEST PREDICTIONS PER HORIZON
# ============================================================
print('\n' + '=' * 70)
print('SAVING BEST PREDICTIONS')
print('=' * 70)

# Determine best strategy per horizon (among all including new ones)
core_strategies = {k: v for k, v in all_strategies.items()}
for h in [1, 5, 22]:
    best_name = min(core_strategies,
                    key=lambda s: qlike(core_strategies[s][h]['rv_actual'].values,
                                        core_strategies[s][h]['rv_pred'].values)
                    if h in core_strategies[s] else 999)
    best_df = core_strategies[best_name][h]
    q = qlike(best_df['rv_actual'].values, best_df['rv_pred'].values)
    best_df.to_parquet(f'data/predictions/walk_forward/hybrid_best_h{h}_annual.parquet', index=False)
    print(f'  H={h}: {best_name} (QLIKE={q:.4f}, {len(best_df)} rows)')

# ============================================================
# DM TESTS
# ============================================================
print('\n' + '=' * 70)
print('DM TESTS')
print('=' * 70)

dm_all = []

# Test key strategies against base models
key_new = ['V6_Trimmed', 'V6_TrimmedW', 'V7_Rolling']
base_models = ['HAR-J', 'XGBoost', 'LightGBM', 'V1_Adaptive']
if 'V2_MultiVal' in all_strategies:
    base_models.append('V2_MultiVal')

for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')
    print(f'  {"Pair":<40} {"DM":>8} {"p":>8} {"Winner":>16}')
    print(f'  {"-" * 75}')

    for new_s in key_new:
        if h not in all_strategies.get(new_s, {}):
            continue
        for base_s in base_models:
            if h not in all_strategies.get(base_s, {}):
                continue

            df1 = all_strategies[new_s][h]
            df2 = all_strategies[base_s][h]

            m = df1[['date', 'ticker', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'p1'})
            m = m.merge(df2[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'p2'}),
                        on=['date', 'ticker'], how='inner')

            loss1 = qlike_per_obs(m['rv_actual'].values, m['p1'].values)
            loss2 = qlike_per_obs(m['rv_actual'].values, m['p2'].values)
            stat, pval = dm_test(loss1, loss2, h_horizon=h)

            if np.isfinite(stat):
                winner = new_s if stat < 0 else base_s
                stars = sig_stars(pval)
                print(f'  {new_s+" vs "+base_s:<40} {stat:>8.3f} {pval:>8.4f} {winner:>13}{stars}')
            else:
                winner = 'N/A'
                print(f'  {new_s+" vs "+base_s:<40} {"nan":>8} {"nan":>8} {"N/A":>13}')

            dm_all.append({'Horizon': h, 'Model1': new_s, 'Model2': base_s,
                           'DM_stat': stat, 'p_value': pval, 'Winner': winner})

dm_all_df = pd.DataFrame(dm_all)
for h in [1, 5, 22]:
    dm_h = dm_all_df[dm_all_df['Horizon'] == h]
    dm_h.to_csv(f'results/tables/dm_tests_v3_h{h}.csv', index=False)

# ============================================================
# VISUALIZATIONS
# ============================================================
print('\n' + '=' * 70)
print('GENERATING FIGURES')
print('=' * 70)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# --- FIGURE 1: Rolling Window Sensitivity ---
print('  Fig 1: Rolling window sensitivity...')
fig, ax = plt.subplots(figsize=(10, 6))
colors_h = {1: '#d62728', 5: '#9467bd', 22: '#8c564b'}
markers_h = {1: 'o', 5: 's', 22: '^'}

for h in [1, 5, 22]:
    qs = [qlike(v7_preds[(h, w)]['rv_actual'].values, v7_preds[(h, w)]['rv_pred'].values)
          for w in windows]
    ax.plot(windows, qs, f'{markers_h[h]}-', color=colors_h[h],
            linewidth=2.5, markersize=10, label=f'H={h}')
    best_w = windows[np.argmin(qs)]
    best_q = min(qs)
    ax.annotate(f'W={best_w}', (best_w, best_q), textcoords='offset points',
                xytext=(10, -10), fontsize=9, color=colors_h[h], fontweight='bold')

ax.set_xlabel('Rolling Window Size (days)', fontsize=12)
ax.set_ylabel('Mean QLIKE', fontsize=12)
ax.set_title('Rolling Inverse QLIKE Weighting: Window Sensitivity', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig('results/figures/rolling_window_sensitivity.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/rolling_window_sensitivity.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 2: Strategy Comparison ---
print('  Fig 2: Strategy comparison v3...')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

compare_strategies = ['HAR-J', 'XGBoost', 'LightGBM', 'BestSingle',
                       'V1_Adaptive', 'V6_Trimmed', 'V7_Rolling']
if 'V2_MultiVal' in all_strategies:
    compare_strategies.insert(5, 'V2_MultiVal')

bar_colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#7f7f7f',
                   '#d62728', '#9467bd', '#8c564b', '#17becf']

for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    means_fig, stds_fig, labels_fig = [], [], []

    for i, sname in enumerate(compare_strategies):
        if sname not in all_strategies or h not in all_strategies[sname]:
            continue
        year_qs = []
        for year in years:
            yr = all_strategies[sname][h]
            yr = yr[yr['year'] == year]
            if len(yr) > 0:
                q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                if np.isfinite(q):
                    year_qs.append(q)
        if year_qs:
            means_fig.append(np.mean(year_qs))
            stds_fig.append(np.std(year_qs))
            labels_fig.append(sname.replace('_', '\n'))

    x = np.arange(len(labels_fig))
    bars = ax.bar(x, means_fig, yerr=stds_fig, capsize=3,
                  color=bar_colors_list[:len(labels_fig)], alpha=0.85,
                  edgecolor='black', linewidth=0.5)

    # Highlight best
    best_idx_fig = np.argmin(means_fig)
    bars[best_idx_fig].set_edgecolor('red')
    bars[best_idx_fig].set_linewidth(2.5)

    for bar, val in zip(bars, means_fig):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_fig, fontsize=7, rotation=45, ha='right')
    ax.set_title(f'H={h}', fontsize=14, fontweight='bold')
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)

fig.suptitle('Walk-Forward QLIKE: All Strategies (2017-2025)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/strategy_comparison_v3.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/strategy_comparison_v3.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 3: Trimmed Analysis for H=5 ---
print('  Fig 3: Trimmed analysis (H=5)...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

plot_models = {
    'V6_Trimmed': v6_preds,
    'V1_Adaptive': v1_preds,
    'LightGBM': {h: preds[('lightgbm', h)] for h in [1, 5, 22]},
    'HAR-J': {h: preds[('har', h)] for h in [1, 5, 22]},
    'V7_Rolling': v7_best,
}
plot_colors = {'V6_Trimmed': '#9467bd', 'V1_Adaptive': '#d62728',
               'LightGBM': '#2ca02c', 'HAR-J': '#1f77b4', 'V7_Rolling': '#17becf'}

for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    for mname, mpreds in plot_models.items():
        if h not in mpreds:
            continue
        year_qs = []
        year_list = []
        for year in years:
            yr = mpreds[h]
            yr = yr[yr['year'] == year]
            if len(yr) > 0:
                q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                if np.isfinite(q):
                    year_list.append(year)
                    year_qs.append(q)

        lw = 3 if mname in ['V6_Trimmed', 'V7_Rolling'] else 1.5
        ms = 8 if mname in ['V6_Trimmed', 'V7_Rolling'] else 5
        ax.plot(year_list, year_qs, 'o-', label=mname, color=plot_colors[mname],
                linewidth=lw, markersize=ms)

    ax.axvspan(2021.8, 2022.2, alpha=0.12, color='red')
    ax.set_title(f'H={h}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Trimmed & Rolling vs Baselines — QLIKE by Year', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/trimmed_analysis.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/trimmed_analysis.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FINAL OUTPUT
# ============================================================
print('\n' + '=' * 70)
print('FINAL RESULTS')
print('=' * 70)

# 1. Full summary
print('\n1. Walk-Forward QLIKE Summary (mean 2017-2025):')
print(f'\n  {"Model":<16} {"H=1":>10} {"H=5":>10} {"H=22":>10}')
print('  ' + '-' * 46)

# Only show key strategies
key_models = ['HAR-J', 'XGBoost', 'LightGBM', 'SimpleAvg', 'BestSingle',
              'V1_Adaptive', 'V6_Trimmed', 'V6_TrimmedW', 'V7_Rolling']
if 'V2_MultiVal' in all_strategies:
    key_models.insert(6, 'V2_MultiVal')

for sname in key_models:
    if sname in all_strategies:
        vals = {}
        for h in [1, 5, 22]:
            if h in all_strategies[sname]:
                df = all_strategies[sname][h]
                vals[h] = qlike(df['rv_actual'].values, df['rv_pred'].values)
        h1 = f'{vals.get(1, np.nan):.4f}' if np.isfinite(vals.get(1, np.nan)) else 'N/A'
        h5 = f'{vals.get(5, np.nan):.4f}' if np.isfinite(vals.get(5, np.nan)) else 'N/A'
        h22 = f'{vals.get(22, np.nan):.4f}' if np.isfinite(vals.get(22, np.nan)) else 'N/A'
        print(f'  {sname:<16} {h1:>10} {h5:>10} {h22:>10}')

# 2. Best per horizon
print('\n2. Best strategy per horizon:')
for h in [1, 5, 22]:
    best_name = min(all_strategies,
                    key=lambda s: qlike(all_strategies[s][h]['rv_actual'].values,
                                        all_strategies[s][h]['rv_pred'].values)
                    if h in all_strategies[s] else 999)
    best_q = qlike(all_strategies[best_name][h]['rv_actual'].values,
                    all_strategies[best_name][h]['rv_pred'].values)
    print(f'  H={h}: {best_name} = {best_q:.4f}')

# 3. DM highlights
print('\n3. DM Tests — Key results:')
for h in [1, 5, 22]:
    dm_h = dm_all_df[dm_all_df['Horizon'] == h]
    if len(dm_h) > 0:
        print(f'\n  H={h}:')
        for _, r in dm_h.iterrows():
            if np.isfinite(r['DM_stat']):
                stars = sig_stars(r['p_value'])
                print(f'    {r["Model1"]:<14} vs {r["Model2"]:<14}: DM={r["DM_stat"]:>7.3f} p={r["p_value"]:.4f}{stars:>4} -> {r["Winner"]}')

# 4. Rolling window recommendation
print('\n4. Rolling Window Sensitivity:')
for _, row in roll_df.iterrows():
    w = int(row['window'])
    print(f'  W={w:>3}: H=1={row["H=1"]:.4f}  H=5={row["H=5"]:.4f}  H=22={row["H=22"]:.4f}')

# 5. V6 Trimmed: what it drops
print('\n5. Trimmed Ensemble — Drop Analysis:')
for h in [1, 5, 22]:
    df = merged[h]
    stack = np.column_stack([df['har'].values, df['xgb'].values, df['lgb'].values])
    med = np.median(stack, axis=1, keepdims=True)
    dev = np.abs(stack - med) / np.clip(med, 1e-10, None)
    worst = np.argmax(dev, axis=1)
    model_names = ['HAR-J', 'XGBoost', 'LightGBM']
    pcts = {m: np.mean(worst == i) * 100 for i, m in enumerate(model_names)}
    print(f'  H={h}: Dropped HAR={pcts["HAR-J"]:.1f}% XGB={pcts["XGBoost"]:.1f}% LGB={pcts["LightGBM"]:.1f}%')

# 6. Recommendation
print('\n6. RECOMMENDATION:')
for h in [1, 5, 22]:
    best_name = min(all_strategies,
                    key=lambda s: qlike(all_strategies[s][h]['rv_actual'].values,
                                        all_strategies[s][h]['rv_pred'].values)
                    if h in all_strategies[s] else 999)
    best_q = qlike(all_strategies[best_name][h]['rv_actual'].values,
                    all_strategies[best_name][h]['rv_pred'].values)

    # Compare to HAR-J baseline
    har_q = qlike(preds[('har', h)]['rv_actual'].values, preds[('har', h)]['rv_pred'].values)
    improvement = (har_q - best_q) / har_q * 100

    print(f'  H={h}: Use {best_name} (QLIKE={best_q:.4f}, {improvement:+.1f}% vs HAR-J)')

# 7. Saved files
print('\n7. Saved files:')
for d in ['data/predictions/walk_forward', 'results/tables', 'results/figures']:
    files = sorted([f for f in os.listdir(d)
                     if any(x in f.lower() for x in ['v3', 'best', 'rolling', 'trimmed', 'final'])])
    if files:
        print(f'  {d}/:')
        for f in files:
            print(f'    {f}')

print('\nDONE!')
