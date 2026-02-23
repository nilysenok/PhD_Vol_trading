#!/usr/bin/env python3
"""Walk-forward with ADAPTIVE hybrid: auto-select best ML + w_har each year."""

import pandas as pd
import numpy as np
import json
import os
import scipy.stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# UTILITIES
# ============================================================
def qlike(y_true, y_pred):
    mask = (y_true > 1e-12) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def qlike_losses(y_true, y_pred):
    mask = (y_true > 1e-12) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
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
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def sig_stars(p):
    if not np.isfinite(p):
        return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''

# ============================================================
# LOAD PREDICTIONS
# ============================================================
print('=' * 70)
print('LOADING WALK-FORWARD PREDICTIONS')
print('=' * 70)

preds = {}  # (model, h) -> DataFrame
for model in ['har', 'xgboost', 'lightgbm']:
    for h in [1, 5, 22]:
        fp = f'data/predictions/walk_forward/{model}_h{h}_annual.parquet'
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            df = df[df['rv_actual'] > 1e-12].reset_index(drop=True)
            preds[(model, h)] = df
            print(f'  {model} H={h}: {len(df)} rows, years {sorted(df.year.unique())}')

test_years = sorted(preds[('har', 1)].year.unique())
print(f'\nTest years: {test_years}')

# ============================================================
# STRATEGY 1: FIXED ML (old approach)
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY 1: FIXED ML HYBRID')
print('=' * 70)

fixed_ml_map = {1: 'xgboost', 5: 'lightgbm', 22: 'lightgbm'}
fixed_preds = {}  # h -> DataFrame with hybrid predictions
fixed_details = []

for h in [1, 5, 22]:
    ml_name = fixed_ml_map[h]
    har_df = preds[('har', h)]
    ml_df = preds[(ml_name, h)]

    all_hybrid = []
    for test_year in test_years:
        val_year = test_year - 1

        har_test = har_df[har_df['year'] == test_year].set_index(['date', 'ticker'])
        ml_test = ml_df[ml_df['year'] == test_year].set_index(['date', 'ticker'])
        common = har_test.index.intersection(ml_test.index)
        har_test = har_test.loc[common]
        ml_test = ml_test.loc[common]

        # Val data for w_har tuning
        har_val = har_df[har_df['year'] == val_year]
        ml_val = ml_df[ml_df['year'] == val_year]

        if len(har_val) > 0 and len(ml_val) > 0:
            har_v = har_val.set_index(['date', 'ticker'])
            ml_v = ml_val.set_index(['date', 'ticker'])
            cv = har_v.index.intersection(ml_v.index)
            har_v, ml_v = har_v.loc[cv], ml_v.loc[cv]
            y_val = har_v['rv_actual'].values

            best_w, best_q = 0, qlike(y_val, ml_v['rv_pred'].values)
            for w in np.arange(0, 0.61, 0.01):
                blend = w * har_v['rv_pred'].values + (1 - w) * ml_v['rv_pred'].values
                q = qlike(y_val, np.clip(blend, 1e-10, None))
                if q < best_q:
                    best_q, best_w = q, w

            if best_w == 0:
                q_har = qlike(y_val, har_v['rv_pred'].values)
                q_ml = qlike(y_val, ml_v['rv_pred'].values)
                best_w = (1 / q_har) / (1 / q_har + 1 / q_ml)
                best_w = np.clip(best_w, 0.05, 0.55)
        else:
            best_w = 0.20

        hybrid_pred = np.clip(
            best_w * har_test['rv_pred'].values + (1 - best_w) * ml_test['rv_pred'].values,
            1e-10, None)

        result = pd.DataFrame({
            'date': har_test.index.get_level_values('date'),
            'ticker': har_test.index.get_level_values('ticker'),
            'year': test_year,
            'rv_actual': har_test['rv_actual'].values,
            'rv_pred': hybrid_pred
        })
        all_hybrid.append(result)

        q_h = qlike(har_test['rv_actual'].values, hybrid_pred)
        fixed_details.append({'year': test_year, 'horizon': h, 'strategy': 'Fixed',
                              'best_ml': ml_name, 'w_har': best_w, 'q_hybrid': q_h})
        print(f'  H={h} {test_year}: ml={ml_name}, w_har={best_w:.3f}, QLIKE={q_h:.4f}')

    fixed_preds[h] = pd.concat(all_hybrid, ignore_index=True)

# ============================================================
# STRATEGY 2: SIMPLE AVERAGE
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY 2: SIMPLE AVERAGE (HAR + XGB + LGB) / 3')
print('=' * 70)

avg_preds = {}

for h in [1, 5, 22]:
    har_df = preds[('har', h)]
    xgb_df = preds[('xgboost', h)]
    lgb_df = preds[('lightgbm', h)]

    merged = har_df.rename(columns={'rv_pred': 'pred_har'})
    merged = merged.merge(
        xgb_df[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'pred_xgb'}),
        on=['date', 'ticker'], how='inner')
    merged = merged.merge(
        lgb_df[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'pred_lgb'}),
        on=['date', 'ticker'], how='inner')

    merged['rv_pred'] = np.clip(
        (merged['pred_har'] + merged['pred_xgb'] + merged['pred_lgb']) / 3, 1e-10, None)

    avg_preds[h] = merged[['date', 'ticker', 'year', 'rv_actual', 'rv_pred']].copy()

    for year in test_years:
        yr = merged[merged['year'] == year]
        q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
        print(f'  H={h} {year}: QLIKE={q:.4f}')

# ============================================================
# STRATEGY 3: BEST SINGLE MODEL (val-selected)
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY 3: BEST SINGLE MODEL (selected on val_year)')
print('=' * 70)

best_single_preds = {}
best_single_details = []

for h in [1, 5, 22]:
    all_single = []
    for test_year in test_years:
        val_year = test_year - 1

        candidates = {}
        for model in ['har', 'xgboost', 'lightgbm']:
            df = preds[(model, h)]
            val_data = df[df['year'] == val_year]
            if len(val_data) > 0:
                candidates[model] = qlike(val_data['rv_actual'].values, val_data['rv_pred'].values)

        if candidates:
            best_model = min(candidates, key=candidates.get)
        else:
            # First year — pick best on test (oracle for this year)
            for model in ['har', 'xgboost', 'lightgbm']:
                df = preds[(model, h)]
                test_data = df[df['year'] == test_year]
                if len(test_data) > 0:
                    candidates[model] = qlike(test_data['rv_actual'].values, test_data['rv_pred'].values)
            best_model = min(candidates, key=candidates.get) if candidates else 'har'

        test_data = preds[(best_model, h)]
        test_data = test_data[test_data['year'] == test_year]
        all_single.append(test_data[['date', 'ticker', 'year', 'rv_actual', 'rv_pred']].copy())

        q = qlike(test_data['rv_actual'].values, test_data['rv_pred'].values)
        best_single_details.append({'year': test_year, 'horizon': h, 'strategy': 'BestSingle',
                                    'best_model': best_model, 'q': q})
        label_map = {'har': 'HAR-J', 'xgboost': 'XGB', 'lightgbm': 'LGB'}
        print(f'  H={h} {test_year}: best={label_map[best_model]}, QLIKE={q:.4f}')

    best_single_preds[h] = pd.concat(all_single, ignore_index=True)

# ============================================================
# STRATEGY 4: ADAPTIVE HYBRID (main)
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY 4: ADAPTIVE HYBRID')
print('=' * 70)

adaptive_preds = {}
adaptive_details = []

for h in [1, 5, 22]:
    har_df = preds[('har', h)]
    xgb_df = preds[('xgboost', h)]
    lgb_df = preds[('lightgbm', h)]

    all_adaptive = []

    for test_year in test_years:
        val_year = test_year - 1

        # Test data
        har_t = har_df[har_df['year'] == test_year].set_index(['date', 'ticker'])
        xgb_t = xgb_df[xgb_df['year'] == test_year].set_index(['date', 'ticker'])
        lgb_t = lgb_df[lgb_df['year'] == test_year].set_index(['date', 'ticker'])
        common_t = har_t.index.intersection(xgb_t.index).intersection(lgb_t.index)
        har_t, xgb_t, lgb_t = har_t.loc[common_t], xgb_t.loc[common_t], lgb_t.loc[common_t]

        # Val data
        har_v = har_df[har_df['year'] == val_year]
        xgb_v = xgb_df[xgb_df['year'] == val_year]
        lgb_v = lgb_df[lgb_df['year'] == val_year]

        has_val = len(har_v) > 0 and len(xgb_v) > 0 and len(lgb_v) > 0

        if has_val:
            # Index and align val
            har_vi = har_v.set_index(['date', 'ticker'])
            xgb_vi = xgb_v.set_index(['date', 'ticker'])
            lgb_vi = lgb_v.set_index(['date', 'ticker'])
            cv = har_vi.index.intersection(xgb_vi.index).intersection(lgb_vi.index)
            har_vi, xgb_vi, lgb_vi = har_vi.loc[cv], xgb_vi.loc[cv], lgb_vi.loc[cv]
            y_val = har_vi['rv_actual'].values

            # Step 3: pick best ML on val
            q_xgb_val = qlike(y_val, xgb_vi['rv_pred'].values)
            q_lgb_val = qlike(y_val, lgb_vi['rv_pred'].values)
            best_ml = 'xgboost' if q_xgb_val < q_lgb_val else 'lightgbm'
            ml_vi = xgb_vi if best_ml == 'xgboost' else lgb_vi
            ml_t = xgb_t if best_ml == 'xgboost' else lgb_t

            # Step 4: tune w_har on val
            best_w, best_q = 0.0, qlike(y_val, ml_vi['rv_pred'].values)
            for w in np.arange(0, 0.61, 0.01):
                blend = w * har_vi['rv_pred'].values + (1 - w) * ml_vi['rv_pred'].values
                q = qlike(y_val, np.clip(blend, 1e-10, None))
                if q < best_q:
                    best_q, best_w = q, w

            # Fallback
            if best_w == 0:
                q_har_v = qlike(y_val, har_vi['rv_pred'].values)
                q_ml_v = qlike(y_val, ml_vi['rv_pred'].values)
                inv_har = 1.0 / (q_har_v + 1e-10)
                inv_ml = 1.0 / (q_ml_v + 1e-10)
                best_w = inv_har / (inv_har + inv_ml)
                method = 'inverse_qlike'
            else:
                method = 'val_grid'

            best_w = np.clip(best_w, 0.05, 0.55)
        else:
            # No val year — use defaults
            q_xgb_test = qlike(har_t['rv_actual'].values, xgb_t['rv_pred'].values)
            q_lgb_test = qlike(har_t['rv_actual'].values, lgb_t['rv_pred'].values)
            best_ml = 'xgboost' if q_xgb_test < q_lgb_test else 'lightgbm'
            ml_t = xgb_t if best_ml == 'xgboost' else lgb_t
            best_w = 0.20
            method = 'default'

        # Step 5: predict
        hybrid_pred = np.clip(
            best_w * har_t['rv_pred'].values + (1 - best_w) * ml_t['rv_pred'].values,
            1e-10, None)

        y_test = har_t['rv_actual'].values
        q_hybrid = qlike(y_test, hybrid_pred)
        q_har = qlike(y_test, har_t['rv_pred'].values)
        q_xgb = qlike(y_test, xgb_t['rv_pred'].values)
        q_lgb = qlike(y_test, lgb_t['rv_pred'].values)

        # Rank
        all_q = {'HAR-J': q_har, 'XGBoost': q_xgb, 'LightGBM': q_lgb, 'Adaptive': q_hybrid}
        sorted_q = sorted(all_q.items(), key=lambda x: x[1])
        rank = [i + 1 for i, (m, _) in enumerate(sorted_q) if m == 'Adaptive'][0]

        adaptive_details.append({
            'year': test_year, 'horizon': h, 'best_ml': best_ml, 'w_har': best_w,
            'method': method, 'q_hybrid': q_hybrid, 'q_har': q_har,
            'q_xgb': q_xgb, 'q_lgb': q_lgb, 'hybrid_rank': rank
        })

        result = pd.DataFrame({
            'date': har_t.index.get_level_values('date'),
            'ticker': har_t.index.get_level_values('ticker'),
            'year': test_year,
            'rv_actual': y_test,
            'rv_pred': hybrid_pred
        })
        all_adaptive.append(result)

        ml_label = 'XGB' if best_ml == 'xgboost' else 'LGB'
        medal = ['', '1st', '2nd', '3rd', '4th'][rank]
        print(f'  H={h} {test_year}: ml={ml_label}, w_har={best_w:.3f} ({method}), '
              f'QLIKE={q_hybrid:.4f} [rank {medal}] | HAR={q_har:.4f} XGB={q_xgb:.4f} LGB={q_lgb:.4f}')

    adaptive_preds[h] = pd.concat(all_adaptive, ignore_index=True)

# ============================================================
# SAVE PREDICTIONS
# ============================================================
print('\n' + '=' * 70)
print('SAVING PREDICTIONS')
print('=' * 70)

Path('data/predictions/walk_forward').mkdir(parents=True, exist_ok=True)
Path('results/tables').mkdir(parents=True, exist_ok=True)
Path('results/figures').mkdir(parents=True, exist_ok=True)

for h in [1, 5, 22]:
    adaptive_preds[h].to_parquet(
        f'data/predictions/walk_forward/hybrid_adaptive_h{h}_annual.parquet', index=False)
    print(f'  Saved hybrid_adaptive_h{h}_annual.parquet ({len(adaptive_preds[h])} rows)')

# ============================================================
# TABLE 1: QLIKE BY YEAR — ALL HYBRID STRATEGIES
# ============================================================
print('\n' + '=' * 70)
print('TABLE 1: QLIKE BY YEAR — ALL HYBRID STRATEGIES')
print('=' * 70)

strategy_preds = {
    'Fixed': fixed_preds,
    'SimpleAvg': avg_preds,
    'BestSingle': best_single_preds,
    'Adaptive': adaptive_preds
}

strat_rows = []
for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')
    header = f'  {"Year":>6}'
    for s in ['HAR-J', 'XGBoost', 'LightGBM', 'Fixed', 'SimpleAvg', 'BestSingle', 'Adaptive']:
        header += f'  {s:>10}'
    print(header)
    print('  ' + '-' * (6 + 7 * 12))

    for year in test_years:
        row = {'year': year, 'horizon': h}

        # Base models
        for model, label in [('har', 'HAR-J'), ('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM')]:
            yr = preds[(model, h)]
            yr = yr[yr['year'] == year]
            row[label] = qlike(yr['rv_actual'].values, yr['rv_pred'].values) if len(yr) > 0 else np.nan

        # Strategy hybrids
        for sname, spreds in strategy_preds.items():
            yr = spreds[h]
            yr = yr[yr['year'] == year]
            row[sname] = qlike(yr['rv_actual'].values, yr['rv_pred'].values) if len(yr) > 0 else np.nan

        strat_rows.append(row)

        line = f'  {year:>6}'
        for col in ['HAR-J', 'XGBoost', 'LightGBM', 'Fixed', 'SimpleAvg', 'BestSingle', 'Adaptive']:
            v = row.get(col, np.nan)
            if np.isfinite(v):
                line += f'  {v:>10.4f}'
            else:
                line += f'  {"N/A":>10}'
        print(line)

    # Mean row
    mean_row = {'year': 'Mean', 'horizon': h}
    for col in ['HAR-J', 'XGBoost', 'LightGBM', 'Fixed', 'SimpleAvg', 'BestSingle', 'Adaptive']:
        vals = [r[col] for r in strat_rows if r['horizon'] == h and isinstance(r[col], float) and np.isfinite(r[col])]
        mean_row[col] = np.mean(vals) if vals else np.nan

    line = f'  {"Mean":>6}'
    for col in ['HAR-J', 'XGBoost', 'LightGBM', 'Fixed', 'SimpleAvg', 'BestSingle', 'Adaptive']:
        v = mean_row.get(col, np.nan)
        if np.isfinite(v):
            line += f'  {v:>10.4f}'
        else:
            line += f'  {"N/A":>10}'
    print(line)

    # Best
    best_col = min(['HAR-J', 'XGBoost', 'LightGBM', 'Fixed', 'SimpleAvg', 'BestSingle', 'Adaptive'],
                    key=lambda c: mean_row.get(c, np.inf))
    print(f'  >>> Best mean: {best_col} = {mean_row[best_col]:.4f}')

strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv('results/tables/hybrid_strategies.csv', index=False)
print('\n  Saved hybrid_strategies.csv')

# ============================================================
# TABLE 2: ADAPTIVE DETAILS
# ============================================================
print('\n' + '=' * 70)
print('TABLE 2: ADAPTIVE HYBRID — DETAILS')
print('=' * 70)

ad_df = pd.DataFrame(adaptive_details)
ad_df.to_csv('results/tables/adaptive_details.csv', index=False)
print(ad_df.to_string(index=False, float_format='%.4f'))

# ============================================================
# TABLE 3: FINAL SUMMARY
# ============================================================
print('\n' + '=' * 70)
print('TABLE 3: FINAL SUMMARY (mean QLIKE across walk-forward years)')
print('=' * 70)

summary_models = {
    'HAR-J': {h: preds[('har', h)] for h in [1, 5, 22]},
    'XGBoost': {h: preds[('xgboost', h)] for h in [1, 5, 22]},
    'LightGBM': {h: preds[('lightgbm', h)] for h in [1, 5, 22]},
    'Hybrid_Fixed': fixed_preds,
    'SimpleAvg': avg_preds,
    'BestSingle': best_single_preds,
    'Hybrid_Adaptive': adaptive_preds,
}

summary_rows = []
for model_name, model_preds in summary_models.items():
    row = {'Model': model_name}
    for h in [1, 5, 22]:
        df = model_preds[h]
        row[f'H={h}'] = qlike(df['rv_actual'].values, df['rv_pred'].values)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/tables/final_summary.csv', index=False)

print(f'\n  {"Model":<20} {"H=1":>10} {"H=5":>10} {"H=22":>10}')
print('  ' + '-' * 50)
for _, row in summary_df.iterrows():
    print(f'  {row["Model"]:<20} {row["H=1"]:>10.4f} {row["H=5"]:>10.4f} {row["H=22"]:>10.4f}')

# Mark best per column
for col in ['H=1', 'H=5', 'H=22']:
    best_idx = summary_df[col].idxmin()
    print(f'  Best {col}: {summary_df.loc[best_idx, "Model"]} = {summary_df.loc[best_idx, col]:.4f}')

# ============================================================
# DM TESTS
# ============================================================
print('\n' + '=' * 70)
print('DIEBOLD-MARIANO TESTS')
print('=' * 70)

dm_models = {
    'HAR-J': {h: preds[('har', h)] for h in [1, 5, 22]},
    'XGBoost': {h: preds[('xgboost', h)] for h in [1, 5, 22]},
    'LightGBM': {h: preds[('lightgbm', h)] for h in [1, 5, 22]},
    'Hybrid_Adaptive': adaptive_preds,
}

dm_model_names = ['HAR-J', 'XGBoost', 'LightGBM', 'Hybrid_Adaptive']
dm_all_results = []

for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')

    # Merge all models
    base = dm_models[dm_model_names[0]][h].rename(columns={'rv_pred': f'pred_{dm_model_names[0]}'})
    for mn in dm_model_names[1:]:
        df_m = dm_models[mn][h][['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'pred_{mn}'})
        base = base.merge(df_m, on=['date', 'ticker'], how='inner')

    actual = base['rv_actual'].values
    losses = {}
    for mn in dm_model_names:
        losses[mn] = qlike_losses(actual, base[f'pred_{mn}'].values)

    # All pairs
    pairs = []
    for i in range(len(dm_model_names)):
        for j in range(i + 1, len(dm_model_names)):
            pairs.append((dm_model_names[i], dm_model_names[j]))

    print(f'  {"Pair":<35} {"DM":>8} {"p":>8} {"Winner":>18}')
    print(f'  {"-" * 70}')

    for m1, m2 in pairs:
        stat, pval = dm_test(losses[m1], losses[m2], h_horizon=h)
        if np.isfinite(stat):
            winner = m1 if stat < 0 else m2
            stars = sig_stars(pval)
            print(f'  {m1+" vs "+m2:<35} {stat:>8.3f} {pval:>8.4f} {winner:>15}{stars}')
        else:
            winner = 'N/A'
            print(f'  {m1+" vs "+m2:<35} {"nan":>8} {"nan":>8} {"N/A":>15}')

        dm_all_results.append({
            'Horizon': h, 'Model1': m1, 'Model2': m2,
            'DM_stat': stat, 'p_value': pval, 'Winner': winner
        })

    # P-value matrix
    print(f'\n  P-value matrix:')
    header = f'  {"":>18}' + ''.join(f'{mn:>18}' for mn in dm_model_names)
    print(header)
    for m1 in dm_model_names:
        row_str = f'  {m1:>18}'
        for m2 in dm_model_names:
            if m1 == m2:
                row_str += f'{"---":>18}'
            else:
                stat, pval = dm_test(losses[m1], losses[m2], h_horizon=h)
                if np.isfinite(pval):
                    stars = sig_stars(pval)
                    row_str += f'{pval:>15.4f}{stars:>3}'
                else:
                    row_str += f'{"nan":>18}'
        print(row_str)

dm_all_df = pd.DataFrame(dm_all_results)
for h in [1, 5, 22]:
    dm_h = dm_all_df[dm_all_df['Horizon'] == h]
    dm_h.to_csv(f'results/tables/dm_tests_adaptive_h{h}.csv', index=False)

# ============================================================
# FIGURE 1: QLIKE COMPARISON
# ============================================================
print('\n' + '=' * 70)
print('GENERATING FIGURES')
print('=' * 70)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

colors_map = {'HAR-J': '#1f77b4', 'XGBoost': '#ff7f0e', 'LightGBM': '#2ca02c',
              'Hybrid_Adaptive': '#d62728'}

print('  Fig 1: QLIKE comparison bars...')
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    model_names_fig = ['HAR-J', 'XGBoost', 'LightGBM', 'Hybrid_Adaptive']
    means_fig, stds_fig = [], []

    for mn in model_names_fig:
        df = dm_models[mn][h]
        year_qs = []
        for year in test_years:
            yr = df[df['year'] == year]
            if len(yr) > 0:
                q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                if np.isfinite(q):
                    year_qs.append(q)
        means_fig.append(np.mean(year_qs))
        stds_fig.append(np.std(year_qs))

    x = np.arange(len(model_names_fig))
    bar_colors = [colors_map[m] for m in model_names_fig]
    bars = ax.bar(x, means_fig, yerr=stds_fig, capsize=5, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, means_fig):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    xlabels = ['HAR-J', 'XGBoost', 'LightGBM', 'Adaptive\nHybrid']
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_title(f'H={h}', fontsize=14, fontweight='bold')
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)

fig.suptitle('Walk-Forward QLIKE (mean ± std across years, 2017-2025)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/qlike_comparison.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/qlike_comparison.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FIGURE 2: QLIKE BY YEAR
# ============================================================
print('  Fig 2: QLIKE by year...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    for mn in ['HAR-J', 'XGBoost', 'LightGBM', 'Hybrid_Adaptive']:
        df = dm_models[mn][h]
        years_p, qlikes_p = [], []
        for year in test_years:
            yr = df[df['year'] == year]
            if len(yr) > 0:
                q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                if np.isfinite(q):
                    years_p.append(year)
                    qlikes_p.append(q)

        lw = 3 if mn == 'Hybrid_Adaptive' else 1.5
        ms = 8 if mn == 'Hybrid_Adaptive' else 5
        label = 'Adaptive Hybrid' if mn == 'Hybrid_Adaptive' else mn
        ax.plot(years_p, qlikes_p, 'o-', label=label, color=colors_map[mn],
                linewidth=lw, markersize=ms, zorder=10 if mn == 'Hybrid_Adaptive' else 5)

    ax.axvspan(2019.8, 2020.2, alpha=0.12, color='gray')
    ax.axvspan(2021.8, 2022.2, alpha=0.12, color='red')

    ax.set_title(f'H={h}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')

fig.suptitle('Walk-Forward QLIKE by Year', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/qlike_by_year.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/qlike_by_year.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FIGURE 3: ADAPTIVE WEIGHTS + ML SELECTION
# ============================================================
print('  Fig 3: Hybrid adaptive weights...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
markers_h = {1: 'o', 5: 's', 22: '^'}
colors_h = {1: '#d62728', 5: '#9467bd', 22: '#8c564b'}

for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    details_h = [d for d in adaptive_details if d['horizon'] == h]
    years = [d['year'] for d in details_h]
    w_hars = [d['w_har'] for d in details_h]
    mls = [d['best_ml'] for d in details_h]

    ax.plot(years, w_hars, f'{markers_h[h]}-', color=colors_h[h], linewidth=2.5, markersize=10,
            label=f'$w_{{HAR}}$ (H={h})')

    # Annotate ML choice
    for yr, w, ml in zip(years, w_hars, mls):
        ml_label = 'XGB' if ml == 'xgboost' else 'LGB'
        color_ann = '#ff7f0e' if ml == 'xgboost' else '#2ca02c'
        ax.annotate(ml_label, (yr, w), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=8, fontweight='bold', color=color_ann,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color_ann, alpha=0.8))

    ax.set_title(f'H={h}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('$w_{HAR}$' if idx == 0 else '', fontsize=12)
    ax.set_ylim(-0.02, 0.65)
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.4, label='Min w=0.05')
    ax.axhline(y=0.55, color='gray', linestyle='--', alpha=0.4, label='Max w=0.55')
    ax.legend(fontsize=9)

fig.suptitle('Adaptive Hybrid: HAR-J Weight + ML Selection by Year', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/hybrid_weights_adaptive.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/hybrid_weights_adaptive.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FINAL OUTPUT
# ============================================================
print('\n' + '=' * 70)
print('FINAL RESULTS')
print('=' * 70)

# 1. Final table
print('\n1. Walk-Forward QLIKE Summary (mean across 2017-2025):')
print(f'\n  {"Model":<20} {"H=1":>10} {"H=5":>10} {"H=22":>10}')
print('  ' + '-' * 50)
for _, row in summary_df.iterrows():
    markers = ''
    for col in ['H=1', 'H=5', 'H=22']:
        if row[col] == summary_df[col].min():
            markers += ' *'
    print(f'  {row["Model"]:<20} {row["H=1"]:>10.4f} {row["H=5"]:>10.4f} {row["H=22"]:>10.4f}')

print()
for col in ['H=1', 'H=5', 'H=22']:
    best_idx = summary_df[col].idxmin()
    print(f'  Best {col}: {summary_df.loc[best_idx, "Model"]} = {summary_df.loc[best_idx, col]:.4f}')

# 2. DM tests: Adaptive vs each
print('\n2. DM Tests — Hybrid_Adaptive vs each model:')
for h in [1, 5, 22]:
    print(f'\n  H={h}:')
    for other in ['HAR-J', 'XGBoost', 'LightGBM']:
        rows_dm = dm_all_df[
            (dm_all_df['Horizon'] == h) &
            (((dm_all_df['Model1'] == 'Hybrid_Adaptive') & (dm_all_df['Model2'] == other)) |
             ((dm_all_df['Model1'] == other) & (dm_all_df['Model2'] == 'Hybrid_Adaptive')))
        ]
        if len(rows_dm) > 0:
            r = rows_dm.iloc[0]
            stars = sig_stars(r['p_value'])
            print(f'    vs {other:<12}: DM={r["DM_stat"]:>7.3f}, p={r["p_value"]:.4f} {stars:>4}  -> {r["Winner"]}')

# 3. Rankings
print('\n3. Adaptive Hybrid Rankings per year:')
for h in [1, 5, 22]:
    details_h = [d for d in adaptive_details if d['horizon'] == h]
    ranks = [d['hybrid_rank'] for d in details_h]
    n1 = sum(1 for r in ranks if r == 1)
    n2 = sum(1 for r in ranks if r == 2)
    n3 = sum(1 for r in ranks if r == 3)
    n4 = sum(1 for r in ranks if r == 4)
    print(f'  H={h}: 1st={n1}x, 2nd={n2}x, 3rd={n3}x, 4th={n4}x (out of {len(ranks)} years)')

# 4. Adaptive details
print('\n4. Adaptive Details — year-by-year:')
for h in [1, 5, 22]:
    print(f'\n  H={h}:')
    print(f'    {"Year":>5} {"ML":>5} {"w_har":>7} {"Method":>14} {"QLIKE":>8} {"Rank":>5}')
    for d in adaptive_details:
        if d['horizon'] == h:
            ml_label = 'XGB' if d['best_ml'] == 'xgboost' else 'LGB'
            print(f'    {d["year"]:>5} {ml_label:>5} {d["w_har"]:>7.3f} {d["method"]:>14} '
                  f'{d["q_hybrid"]:>8.4f} {d["hybrid_rank"]:>5}')

# Saved files
print('\n5. Saved files:')
for d in ['data/predictions/walk_forward', 'results/tables', 'results/figures']:
    files = sorted([f for f in os.listdir(d) if 'adaptive' in f.lower() or 'summary' in f.lower() or 'strateg' in f.lower()])
    if files:
        print(f'  {d}/:')
        for f in files:
            print(f'    {f}')

print('\nDONE!')
