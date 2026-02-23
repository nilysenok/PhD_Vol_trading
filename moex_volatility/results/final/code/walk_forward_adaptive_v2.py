#!/usr/bin/env python3
"""Adaptive hybrid v2: multi-val, regime-aware, shrinkage strategies."""

import pandas as pd
import numpy as np
import warnings
import os
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

def find_best_weights_3comp(actual, har_pred, xgb_pred, lgb_pred, step=0.02):
    best_q = 999
    best_w = (1/3, 1/3, 1/3)
    for w_har in np.arange(0, 0.61, step):
        for w_xgb in np.arange(0, 1.01 - w_har, step):
            w_lgb = round(1 - w_har - w_xgb, 4)
            if w_lgb < -0.001:
                continue
            w_lgb = max(w_lgb, 0)
            blend = w_har * har_pred + w_xgb * xgb_pred + w_lgb * lgb_pred
            q = qlike(actual, np.clip(blend, 1e-10, None))
            if q < best_q:
                best_q = q
                best_w = (w_har, w_xgb, w_lgb)
    return best_w, best_q

def merge_3models(har_yr, xgb_yr, lgb_yr):
    """Merge 3 model predictions on date+ticker, return aligned arrays."""
    m = har_yr[['date', 'ticker', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'har'})
    m = m.merge(xgb_yr[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'xgb'}),
                on=['date', 'ticker'], how='inner')
    m = m.merge(lgb_yr[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'lgb'}),
                on=['date', 'ticker'], how='inner')
    return m

# ============================================================
# LOAD DATA
# ============================================================
print('=' * 70)
print('LOADING WALK-FORWARD PREDICTIONS')
print('=' * 70)

models = {}
for model in ['har', 'xgboost', 'lightgbm']:
    for h in [1, 5, 22]:
        fp = f'data/predictions/walk_forward/{model}_h{h}_annual.parquet'
        df = pd.read_parquet(fp)
        df = df[df['rv_actual'] > 1e-12].copy()
        models[(model, h)] = df
        print(f'  {model} H={h}: {len(df)} rows')

# Load baseline adaptive
baseline = {}
for h in [1, 5, 22]:
    fp = f'data/predictions/walk_forward/hybrid_adaptive_h{h}_annual.parquet'
    if os.path.exists(fp):
        df = pd.read_parquet(fp)
        df = df[df['rv_actual'] > 1e-12].copy()
        baseline[h] = df

years = sorted(models[('har', 1)]['year'].unique())
print(f'\nYears: {years}')

Path('data/predictions/walk_forward').mkdir(parents=True, exist_ok=True)
Path('results/tables').mkdir(parents=True, exist_ok=True)
Path('results/figures').mkdir(parents=True, exist_ok=True)

# ============================================================
# STRATEGY V2: MULTI-VAL (2 years for weights)
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY V2: MULTI-VAL (2 val years)')
print('=' * 70)

v2_preds = {}
v2_details = []

for h in [1, 5, 22]:
    har_df = models[('har', h)]
    xgb_df = models[('xgboost', h)]
    lgb_df = models[('lightgbm', h)]
    all_results = []

    for test_year in years:
        val_years_avail = [y for y in [test_year - 2, test_year - 1] if y in years]

        # Get test data merged
        har_t = har_df[har_df['year'] == test_year]
        xgb_t = xgb_df[xgb_df['year'] == test_year]
        lgb_t = lgb_df[lgb_df['year'] == test_year]
        mt = merge_3models(har_t, xgb_t, lgb_t)

        if len(val_years_avail) == 0:
            # Fallback: equal weights
            w_final = np.array([1/3, 1/3, 1/3])
            method = 'equal_default'
        else:
            val_weights = []
            for vy in val_years_avail:
                har_v = har_df[har_df['year'] == vy]
                xgb_v = xgb_df[xgb_df['year'] == vy]
                lgb_v = lgb_df[lgb_df['year'] == vy]
                mv = merge_3models(har_v, xgb_v, lgb_v)
                if len(mv) > 10:
                    w, _ = find_best_weights_3comp(
                        mv['rv_actual'].values, mv['har'].values,
                        mv['xgb'].values, mv['lgb'].values)
                    val_weights.append(w)

            if len(val_weights) == 0:
                w_final = np.array([1/3, 1/3, 1/3])
                method = 'equal_fallback'
            else:
                w_final = np.mean(val_weights, axis=0)
                w_final = w_final / w_final.sum()
                method = f'multi_val_{len(val_weights)}yr'

        blend = np.clip(
            w_final[0] * mt['har'].values + w_final[1] * mt['xgb'].values + w_final[2] * mt['lgb'].values,
            1e-10, None)
        q = qlike(mt['rv_actual'].values, blend)

        result_df = pd.DataFrame({
            'date': mt['date'].values, 'ticker': mt['ticker'].values,
            'year': test_year, 'rv_actual': mt['rv_actual'].values, 'rv_pred': blend})
        all_results.append(result_df)

        v2_details.append({
            'year': test_year, 'horizon': h, 'strategy': 'V2_MultiVal',
            'w_har': w_final[0], 'w_xgb': w_final[1], 'w_lgb': w_final[2],
            'method': method, 'qlike': q})
        print(f'  H={h} {test_year}: w=({w_final[0]:.2f},{w_final[1]:.2f},{w_final[2]:.2f}) '
              f'{method} QLIKE={q:.4f}')

    v2_preds[h] = pd.concat(all_results, ignore_index=True)

# ============================================================
# STRATEGY V3: REGIME-AWARE
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY V3: REGIME-AWARE')
print('=' * 70)

v3_preds = {}
v3_details = []

for h in [1, 5, 22]:
    har_df = models[('har', h)]
    xgb_df = models[('xgboost', h)]
    lgb_df = models[('lightgbm', h)]
    all_results = []

    for test_year in years:
        val_year = test_year - 1

        # Test data
        har_t = har_df[har_df['year'] == test_year]
        xgb_t = xgb_df[xgb_df['year'] == test_year]
        lgb_t = lgb_df[lgb_df['year'] == test_year]
        mt = merge_3models(har_t, xgb_t, lgb_t)

        # Determine regime
        historical = har_df[har_df['year'] < test_year]['rv_actual'].values
        recent = har_df[har_df['year'] == val_year]['rv_actual'].values if val_year in years else historical

        if len(historical) > 0 and len(recent) > 0:
            regime_ratio = np.mean(recent) / np.mean(historical)
            if regime_ratio > 1.3:
                regime = 'HIGH_VOL'
            elif regime_ratio < 0.7:
                regime = 'LOW_VOL'
            else:
                regime = 'NORMAL'
        else:
            regime_ratio = 1.0
            regime = 'NORMAL'

        # Base weights from val_year grid search
        if val_year in years:
            har_v = har_df[har_df['year'] == val_year]
            xgb_v = xgb_df[xgb_df['year'] == val_year]
            lgb_v = lgb_df[lgb_df['year'] == val_year]
            mv = merge_3models(har_v, xgb_v, lgb_v)
            if len(mv) > 10:
                w_base, _ = find_best_weights_3comp(
                    mv['rv_actual'].values, mv['har'].values,
                    mv['xgb'].values, mv['lgb'].values)
            else:
                w_base = (1/3, 1/3, 1/3)
        else:
            w_base = (1/3, 1/3, 1/3)

        w = np.array(w_base)

        # Regime adjustment
        if regime == 'HIGH_VOL':
            w_har_adj = min(w[0] * 1.5, 0.6)
            remainder = 1.0 - w_har_adj
            ml_sum = w[1] + w[2]
            if ml_sum > 0:
                w[1] = remainder * (w[1] / ml_sum)
                w[2] = remainder * (w[2] / ml_sum)
            else:
                w[1] = remainder / 2
                w[2] = remainder / 2
            w[0] = w_har_adj
        elif regime == 'LOW_VOL':
            w_har_adj = max(w[0] * 0.7, 0.05)
            remainder = 1.0 - w_har_adj
            ml_sum = w[1] + w[2]
            if ml_sum > 0:
                w[1] = remainder * (w[1] / ml_sum)
                w[2] = remainder * (w[2] / ml_sum)
            else:
                w[1] = remainder / 2
                w[2] = remainder / 2
            w[0] = w_har_adj

        w = w / w.sum()

        blend = np.clip(
            w[0] * mt['har'].values + w[1] * mt['xgb'].values + w[2] * mt['lgb'].values,
            1e-10, None)
        q = qlike(mt['rv_actual'].values, blend)

        result_df = pd.DataFrame({
            'date': mt['date'].values, 'ticker': mt['ticker'].values,
            'year': test_year, 'rv_actual': mt['rv_actual'].values, 'rv_pred': blend})
        all_results.append(result_df)

        v3_details.append({
            'year': test_year, 'horizon': h, 'strategy': 'V3_Regime',
            'w_har': w[0], 'w_xgb': w[1], 'w_lgb': w[2],
            'regime': regime, 'regime_ratio': regime_ratio, 'qlike': q})
        print(f'  H={h} {test_year}: regime={regime} (ratio={regime_ratio:.2f}) '
              f'w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f}) QLIKE={q:.4f}')

    v3_preds[h] = pd.concat(all_results, ignore_index=True)

# ============================================================
# STRATEGY V4: SHRINKAGE TO EQUAL WEIGHTS
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY V4: SHRINKAGE (lambda=0.3)')
print('=' * 70)

shrinkage_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
v4_preds = {}
v4_details = []
shrinkage_results = {s: {h: [] for h in [1, 5, 22]} for s in shrinkage_levels}
FIXED_SHRINKAGE = 0.3

for h in [1, 5, 22]:
    har_df = models[('har', h)]
    xgb_df = models[('xgboost', h)]
    lgb_df = models[('lightgbm', h)]
    all_results = []

    for test_year in years:
        val_year = test_year - 1

        har_t = har_df[har_df['year'] == test_year]
        xgb_t = xgb_df[xgb_df['year'] == test_year]
        lgb_t = lgb_df[lgb_df['year'] == test_year]
        mt = merge_3models(har_t, xgb_t, lgb_t)

        # Optimal weights from val
        if val_year in years:
            har_v = har_df[har_df['year'] == val_year]
            xgb_v = xgb_df[xgb_df['year'] == val_year]
            lgb_v = lgb_df[lgb_df['year'] == val_year]
            mv = merge_3models(har_v, xgb_v, lgb_v)
            if len(mv) > 10:
                w_opt, _ = find_best_weights_3comp(
                    mv['rv_actual'].values, mv['har'].values,
                    mv['xgb'].values, mv['lgb'].values)
            else:
                w_opt = (1/3, 1/3, 1/3)
        else:
            w_opt = (1/3, 1/3, 1/3)

        w_eq = np.array([1/3, 1/3, 1/3])

        # Compute QLIKE for all shrinkage levels
        for s in shrinkage_levels:
            w_s = (1 - s) * np.array(w_opt) + s * w_eq
            w_s = w_s / w_s.sum()
            blend_s = np.clip(
                w_s[0] * mt['har'].values + w_s[1] * mt['xgb'].values + w_s[2] * mt['lgb'].values,
                1e-10, None)
            q_s = qlike(mt['rv_actual'].values, blend_s)
            shrinkage_results[s][h].append(q_s)

        # Use fixed shrinkage for V4 predictions
        w_final = (1 - FIXED_SHRINKAGE) * np.array(w_opt) + FIXED_SHRINKAGE * w_eq
        w_final = w_final / w_final.sum()

        blend = np.clip(
            w_final[0] * mt['har'].values + w_final[1] * mt['xgb'].values + w_final[2] * mt['lgb'].values,
            1e-10, None)
        q = qlike(mt['rv_actual'].values, blend)

        result_df = pd.DataFrame({
            'date': mt['date'].values, 'ticker': mt['ticker'].values,
            'year': test_year, 'rv_actual': mt['rv_actual'].values, 'rv_pred': blend})
        all_results.append(result_df)

        v4_details.append({
            'year': test_year, 'horizon': h, 'strategy': 'V4_Shrinkage',
            'w_har': w_final[0], 'w_xgb': w_final[1], 'w_lgb': w_final[2],
            'shrinkage': FIXED_SHRINKAGE, 'qlike': q})
        print(f'  H={h} {test_year}: w_opt=({w_opt[0]:.2f},{w_opt[1]:.2f},{w_opt[2]:.2f}) '
              f'-> shrunk=({w_final[0]:.2f},{w_final[1]:.2f},{w_final[2]:.2f}) QLIKE={q:.4f}')

    v4_preds[h] = pd.concat(all_results, ignore_index=True)

# Shrinkage sensitivity table
print('\n--- Shrinkage Sensitivity ---')
shrink_table = []
for s in shrinkage_levels:
    row = {'shrinkage': s}
    for h in [1, 5, 22]:
        row[f'H={h}'] = np.mean(shrinkage_results[s][h])
    shrink_table.append(row)
    print(f'  lambda={s:.1f}: H=1={row["H=1"]:.4f}  H=5={row["H=5"]:.4f}  H=22={row["H=22"]:.4f}')

shrink_df = pd.DataFrame(shrink_table)
shrink_df.to_csv('results/tables/shrinkage_sensitivity.csv', index=False)

# ============================================================
# STRATEGY V5: COMBINED (multi-val + regime + shrinkage)
# ============================================================
print('\n' + '=' * 70)
print('STRATEGY V5: COMBINED (MultiVal + Regime + Shrinkage)')
print('=' * 70)

v5_preds = {}
v5_details = []

for h in [1, 5, 22]:
    har_df = models[('har', h)]
    xgb_df = models[('xgboost', h)]
    lgb_df = models[('lightgbm', h)]
    all_results = []

    for test_year in years:
        har_t = har_df[har_df['year'] == test_year]
        xgb_t = xgb_df[xgb_df['year'] == test_year]
        lgb_t = lgb_df[lgb_df['year'] == test_year]
        mt = merge_3models(har_t, xgb_t, lgb_t)

        # Step 1: Multi-val weights (2 years)
        val_years_avail = [y for y in [test_year - 2, test_year - 1] if y in years]
        if len(val_years_avail) == 0:
            w = np.array([1/3, 1/3, 1/3])
        else:
            val_weights = []
            for vy in val_years_avail:
                har_v = har_df[har_df['year'] == vy]
                xgb_v = xgb_df[xgb_df['year'] == vy]
                lgb_v = lgb_df[lgb_df['year'] == vy]
                mv = merge_3models(har_v, xgb_v, lgb_v)
                if len(mv) > 10:
                    wv, _ = find_best_weights_3comp(
                        mv['rv_actual'].values, mv['har'].values,
                        mv['xgb'].values, mv['lgb'].values)
                    val_weights.append(wv)
            if len(val_weights) == 0:
                w = np.array([1/3, 1/3, 1/3])
            else:
                w = np.mean(val_weights, axis=0)
                w = w / w.sum()

        # Step 2: Regime adjustment
        historical = har_df[har_df['year'] < test_year]['rv_actual'].values
        val_year = test_year - 1
        recent = har_df[har_df['year'] == val_year]['rv_actual'].values if val_year in years else historical

        if len(historical) > 0 and len(recent) > 0:
            regime_ratio = np.mean(recent) / np.mean(historical)
            if regime_ratio > 1.3:
                regime = 'HIGH_VOL'
            elif regime_ratio < 0.7:
                regime = 'LOW_VOL'
            else:
                regime = 'NORMAL'
        else:
            regime_ratio = 1.0
            regime = 'NORMAL'

        if regime == 'HIGH_VOL':
            w_har_adj = min(w[0] * 1.5, 0.6)
            remainder = 1.0 - w_har_adj
            ml_sum = w[1] + w[2]
            if ml_sum > 0:
                w[1] = remainder * (w[1] / ml_sum)
                w[2] = remainder * (w[2] / ml_sum)
            else:
                w[1] = w[2] = remainder / 2
            w[0] = w_har_adj
        elif regime == 'LOW_VOL':
            w_har_adj = max(w[0] * 0.7, 0.05)
            remainder = 1.0 - w_har_adj
            ml_sum = w[1] + w[2]
            if ml_sum > 0:
                w[1] = remainder * (w[1] / ml_sum)
                w[2] = remainder * (w[2] / ml_sum)
            else:
                w[1] = w[2] = remainder / 2
            w[0] = w_har_adj

        # Step 3: Shrinkage
        w_eq = np.array([1/3, 1/3, 1/3])
        w = (1 - FIXED_SHRINKAGE) * w + FIXED_SHRINKAGE * w_eq
        w = w / w.sum()

        blend = np.clip(
            w[0] * mt['har'].values + w[1] * mt['xgb'].values + w[2] * mt['lgb'].values,
            1e-10, None)
        q = qlike(mt['rv_actual'].values, blend)

        result_df = pd.DataFrame({
            'date': mt['date'].values, 'ticker': mt['ticker'].values,
            'year': test_year, 'rv_actual': mt['rv_actual'].values, 'rv_pred': blend})
        all_results.append(result_df)

        v5_details.append({
            'year': test_year, 'horizon': h, 'strategy': 'V5_Combined',
            'w_har': w[0], 'w_xgb': w[1], 'w_lgb': w[2],
            'regime': regime, 'regime_ratio': regime_ratio, 'qlike': q})
        print(f'  H={h} {test_year}: {regime}(ratio={regime_ratio:.2f}) '
              f'w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f}) QLIKE={q:.4f}')

    v5_preds[h] = pd.concat(all_results, ignore_index=True)

# ============================================================
# ALSO: BestSingle baseline
# ============================================================
print('\n--- Computing BestSingle baseline ---')
best_single_preds = {}
for h in [1, 5, 22]:
    all_bs = []
    for test_year in years:
        val_year = test_year - 1
        candidates = {}
        for model in ['har', 'xgboost', 'lightgbm']:
            df = models[(model, h)]
            val_data = df[df['year'] == val_year]
            if len(val_data) > 0:
                candidates[model] = qlike(val_data['rv_actual'].values, val_data['rv_pred'].values)
        if candidates:
            best_model = min(candidates, key=candidates.get)
        else:
            best_model = 'har'
        test_data = models[(best_model, h)]
        test_data = test_data[test_data['year'] == test_year]
        all_bs.append(test_data[['date', 'ticker', 'year', 'rv_actual', 'rv_pred']].copy())
    best_single_preds[h] = pd.concat(all_bs, ignore_index=True)

# ============================================================
# TABLE 1: ALL STRATEGIES BY YEAR
# ============================================================
print('\n' + '=' * 70)
print('TABLE 1: QLIKE BY YEAR — ALL STRATEGIES')
print('=' * 70)

all_strategies = {
    'V1_Adaptive': baseline,
    'V2_MultiVal': v2_preds,
    'V3_Regime': v3_preds,
    'V4_Shrinkage': v4_preds,
    'V5_Combined': v5_preds,
    'HAR-J': {h: models[('har', h)] for h in [1, 5, 22]},
    'XGBoost': {h: models[('xgboost', h)] for h in [1, 5, 22]},
    'LightGBM': {h: models[('lightgbm', h)] for h in [1, 5, 22]},
    'BestSingle': best_single_preds,
}

strat_rows = []
for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')
    snames = list(all_strategies.keys())
    header = f'  {"Year":>6}' + ''.join(f'  {s:>12}' for s in snames)
    print(header)
    print('  ' + '-' * (6 + len(snames) * 14))

    for year in years:
        row = {'year': year, 'horizon': h}
        for sname, spreds in all_strategies.items():
            yr = spreds[h]
            yr = yr[yr['year'] == year]
            if len(yr) > 0:
                row[sname] = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
            else:
                row[sname] = np.nan
        strat_rows.append(row)

        line = f'  {year:>6}'
        vals_row = [row.get(s, np.nan) for s in snames]
        best_val = min(v for v in vals_row if np.isfinite(v))
        for s in snames:
            v = row.get(s, np.nan)
            if np.isfinite(v):
                mark = '*' if abs(v - best_val) < 1e-6 else ' '
                line += f'  {v:>11.4f}{mark}'
            else:
                line += f'  {"N/A":>12}'
        print(line)

    # Mean
    mean_vals = {}
    for s in snames:
        vals = [r[s] for r in strat_rows if r['horizon'] == h and np.isfinite(r.get(s, np.nan))]
        mean_vals[s] = np.mean(vals) if vals else np.nan

    line = f'  {"Mean":>6}'
    best_mean = min(v for v in mean_vals.values() if np.isfinite(v))
    for s in snames:
        v = mean_vals[s]
        mark = '*' if np.isfinite(v) and abs(v - best_mean) < 1e-6 else ' '
        line += f'  {v:>11.4f}{mark}' if np.isfinite(v) else f'  {"N/A":>12}'
    print(line)
    best_s = min(mean_vals, key=lambda k: mean_vals[k] if np.isfinite(mean_vals[k]) else 999)
    print(f'  >>> Best: {best_s} = {mean_vals[best_s]:.4f}')

strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv('results/tables/hybrid_strategies_v2.csv', index=False)

# ============================================================
# TABLE 2: FINAL SUMMARY
# ============================================================
print('\n' + '=' * 70)
print('TABLE 2: FINAL SUMMARY (mean QLIKE)')
print('=' * 70)

summary_rows = []
for sname, spreds in all_strategies.items():
    row = {'Model': sname}
    for h in [1, 5, 22]:
        df = spreds[h]
        row[f'H={h}'] = qlike(df['rv_actual'].values, df['rv_pred'].values)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/tables/final_summary_v2.csv', index=False)

print(f'\n  {"Model":<16} {"H=1":>10} {"H=5":>10} {"H=22":>10}')
print('  ' + '-' * 46)
for _, row in summary_df.iterrows():
    marks = ''
    print(f'  {row["Model"]:<16} {row["H=1"]:>10.4f} {row["H=5"]:>10.4f} {row["H=22"]:>10.4f}')

for col in ['H=1', 'H=5', 'H=22']:
    best_idx = summary_df[col].idxmin()
    print(f'  Best {col}: {summary_df.loc[best_idx, "Model"]} = {summary_df.loc[best_idx, col]:.4f}')

# ============================================================
# TABLE 3: DETAILS OF BEST STRATEGY PER HORIZON
# ============================================================
print('\n' + '=' * 70)
print('TABLE 3: ADAPTIVE DETAILS (all V2-V5)')
print('=' * 70)

all_details = v2_details + v3_details + v4_details + v5_details
details_df = pd.DataFrame(all_details)
details_df.to_csv('results/tables/adaptive_details_v2.csv', index=False)

# Show best strategy details per horizon
for h in [1, 5, 22]:
    # Determine best V-strategy for this horizon
    v_strategies = {'V2_MultiVal': v2_preds, 'V3_Regime': v3_preds,
                    'V4_Shrinkage': v4_preds, 'V5_Combined': v5_preds}
    best_v = min(v_strategies, key=lambda s: qlike(
        v_strategies[s][h]['rv_actual'].values, v_strategies[s][h]['rv_pred'].values))
    best_q = qlike(v_strategies[best_v][h]['rv_actual'].values, v_strategies[best_v][h]['rv_pred'].values)
    print(f'\n  H={h} best V-strategy: {best_v} (QLIKE={best_q:.4f})')

    det = [d for d in all_details if d['horizon'] == h and d['strategy'] == best_v.replace('_', '_')]
    # Find matching details
    det_match = details_df[(details_df['horizon'] == h) & (details_df['strategy'].str.contains(best_v.split('_')[0]))]
    if len(det_match) > 0:
        for _, d in det_match.iterrows():
            regime_str = f' regime={d.get("regime", "N/A")}' if 'regime' in d and pd.notna(d.get('regime')) else ''
            print(f'    {int(d["year"])}: w=({d["w_har"]:.2f},{d["w_xgb"]:.2f},{d["w_lgb"]:.2f}){regime_str} '
                  f'QLIKE={d["qlike"]:.4f}')

# ============================================================
# SAVE BEST PREDICTIONS
# ============================================================
print('\n' + '=' * 70)
print('SAVING BEST PREDICTIONS')
print('=' * 70)

# Determine overall best strategy
for h in [1, 5, 22]:
    v_strats = {
        'V2_MultiVal': v2_preds[h], 'V3_Regime': v3_preds[h],
        'V4_Shrinkage': v4_preds[h], 'V5_Combined': v5_preds[h]}
    best_v = min(v_strats, key=lambda s: qlike(v_strats[s]['rv_actual'].values, v_strats[s]['rv_pred'].values))
    best_df = v_strats[best_v]
    best_df.to_parquet(f'data/predictions/walk_forward/hybrid_v2_h{h}_annual.parquet', index=False)
    q = qlike(best_df['rv_actual'].values, best_df['rv_pred'].values)
    print(f'  H={h}: saved {best_v} as hybrid_v2 (QLIKE={q:.4f}, {len(best_df)} rows)')

# ============================================================
# DM TESTS
# ============================================================
print('\n' + '=' * 70)
print('DM TESTS')
print('=' * 70)

dm_all = []
# Test each V-strategy vs base models
v_all = {'V1_Adaptive': baseline, 'V2_MultiVal': v2_preds, 'V3_Regime': v3_preds,
         'V4_Shrinkage': v4_preds, 'V5_Combined': v5_preds}
base_models = {'HAR-J': {h: models[('har', h)] for h in [1, 5, 22]},
               'XGBoost': {h: models[('xgboost', h)] for h in [1, 5, 22]},
               'LightGBM': {h: models[('lightgbm', h)] for h in [1, 5, 22]}}

for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')

    # Determine best V
    best_v_name = min(v_all, key=lambda s: qlike(
        v_all[s][h]['rv_actual'].values, v_all[s][h]['rv_pred'].values))
    best_v_df = v_all[best_v_name][h]
    best_v_q = qlike(best_v_df['rv_actual'].values, best_v_df['rv_pred'].values)
    print(f'  Best V-strategy: {best_v_name} (QLIKE={best_v_q:.4f})')

    # DM: best V vs all base models
    print(f'\n  {"Pair":<35} {"DM":>8} {"p":>8} {"Winner":>18}')
    print(f'  {"-" * 70}')

    # Also test V strategies against each other
    all_test_models = {**base_models, **v_all}
    test_pairs = []
    # Best V vs base models
    for bm in ['HAR-J', 'XGBoost', 'LightGBM']:
        test_pairs.append((best_v_name, bm))
    # Best V vs other V strategies
    for vn in v_all:
        if vn != best_v_name:
            test_pairs.append((best_v_name, vn))

    for m1, m2 in test_pairs:
        df1 = all_test_models[m1][h]
        df2 = all_test_models[m2][h]

        # Merge on date+ticker
        merged = df1[['date', 'ticker', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'pred1'})
        merged = merged.merge(
            df2[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'pred2'}),
            on=['date', 'ticker'], how='inner')

        loss1 = qlike_per_obs(merged['rv_actual'].values, merged['pred1'].values)
        loss2 = qlike_per_obs(merged['rv_actual'].values, merged['pred2'].values)
        stat, pval = dm_test(loss1, loss2, h_horizon=h)

        if np.isfinite(stat):
            winner = m1 if stat < 0 else m2
            stars = sig_stars(pval)
            print(f'  {m1+" vs "+m2:<35} {stat:>8.3f} {pval:>8.4f} {winner:>15}{stars}')
        else:
            winner = 'N/A'
            print(f'  {m1+" vs "+m2:<35} {"nan":>8} {"nan":>8} {"N/A":>15}')

        dm_all.append({'Horizon': h, 'Model1': m1, 'Model2': m2,
                       'DM_stat': stat, 'p_value': pval, 'Winner': winner})

dm_all_df = pd.DataFrame(dm_all)
for h in [1, 5, 22]:
    dm_h = dm_all_df[dm_all_df['Horizon'] == h]
    dm_h.to_csv(f'results/tables/dm_tests_v2_h{h}.csv', index=False)

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

# --- FIGURE 1: Strategy Comparison Grouped Bar ---
print('  Fig 1: Strategy comparison...')
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
bar_strategies = ['HAR-J', 'XGBoost', 'LightGBM', 'V1_Adaptive', 'V2_MultiVal', 'V3_Regime', 'V4_Shrinkage', 'V5_Combined']
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    means_fig, stds_fig = [], []
    labels_fig = []

    for sname in bar_strategies:
        spreds = all_strategies[sname]
        year_qs = []
        for year in years:
            yr = spreds[h]
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
                  color=bar_colors[:len(labels_fig)], alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, means_fig):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_fig, fontsize=7, rotation=45, ha='right')
    ax.set_title(f'H={h}', fontsize=14, fontweight='bold')
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)

fig.suptitle('Walk-Forward QLIKE: All Hybrid Strategies (2017-2025)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/strategy_comparison.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/strategy_comparison.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 2: Stacked Area Weights ---
print('  Fig 2: Adaptive weights v2...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Use best V strategy details
for idx, h in enumerate([1, 5, 22]):
    ax = axes[idx]
    best_v_name = min(v_all, key=lambda s: qlike(
        v_all[s][h]['rv_actual'].values, v_all[s][h]['rv_pred'].values))

    # Get details for this strategy
    if best_v_name == 'V2_MultiVal':
        det = [d for d in v2_details if d['horizon'] == h]
    elif best_v_name == 'V3_Regime':
        det = [d for d in v3_details if d['horizon'] == h]
    elif best_v_name == 'V4_Shrinkage':
        det = [d for d in v4_details if d['horizon'] == h]
    elif best_v_name == 'V5_Combined':
        det = [d for d in v5_details if d['horizon'] == h]
    else:
        det = [d for d in v2_details if d['horizon'] == h]  # fallback

    det_years = [d['year'] for d in det]
    w_hars = [d['w_har'] for d in det]
    w_xgbs = [d['w_xgb'] for d in det]
    w_lgbs = [d['w_lgb'] for d in det]

    ax.stackplot(det_years, w_hars, w_xgbs, w_lgbs,
                 labels=['HAR-J', 'XGBoost', 'LightGBM'],
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)

    # Regime background
    for d in det:
        if d.get('regime') == 'HIGH_VOL':
            ax.axvspan(d['year'] - 0.4, d['year'] + 0.4, alpha=0.15, color='red')
        elif d.get('regime') == 'LOW_VOL':
            ax.axvspan(d['year'] - 0.4, d['year'] + 0.4, alpha=0.15, color='blue')

    ax.set_title(f'H={h} ({best_v_name})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Weight' if idx == 0 else '', fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc='upper right')

fig.suptitle('Model Weights Over Time (Best Strategy)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/adaptive_weights_v2.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/adaptive_weights_v2.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 3: Shrinkage Sensitivity ---
print('  Fig 3: Shrinkage sensitivity...')
fig, ax = plt.subplots(figsize=(10, 6))
colors_h = {1: '#d62728', 5: '#9467bd', 22: '#8c564b'}
markers_h = {1: 'o', 5: 's', 22: '^'}

for h in [1, 5, 22]:
    means = [np.mean(shrinkage_results[s][h]) for s in shrinkage_levels]
    ax.plot(shrinkage_levels, means, f'{markers_h[h]}-', color=colors_h[h],
            linewidth=2.5, markersize=10, label=f'H={h}')

    best_s = shrinkage_levels[np.argmin(means)]
    best_q = min(means)
    ax.annotate(f'best={best_s:.1f}', (best_s, best_q),
                textcoords='offset points', xytext=(10, -10),
                fontsize=9, color=colors_h[h])

ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='Fixed (0.3)')
ax.set_xlabel('Shrinkage Level ($\\lambda$)', fontsize=12)
ax.set_ylabel('Mean QLIKE', fontsize=12)
ax.set_title('Shrinkage Sensitivity Analysis', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig('results/figures/shrinkage_sensitivity.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/shrinkage_sensitivity.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FINAL OUTPUT
# ============================================================
print('\n' + '=' * 70)
print('FINAL RESULTS')
print('=' * 70)

# 1. Summary table
print('\n1. Walk-Forward QLIKE Summary (mean 2017-2025):')
print(f'\n  {"Model":<16} {"H=1":>10} {"H=5":>10} {"H=22":>10}')
print('  ' + '-' * 46)
for _, row in summary_df.iterrows():
    print(f'  {row["Model"]:<16} {row["H=1"]:>10.4f} {row["H=5"]:>10.4f} {row["H=22"]:>10.4f}')

# 2. Best strategy per horizon
print('\n2. Best strategy per horizon:')
for col in ['H=1', 'H=5', 'H=22']:
    best_idx = summary_df[col].idxmin()
    print(f'  {col}: {summary_df.loc[best_idx, "Model"]} = {summary_df.loc[best_idx, col]:.4f}')

# 3. DM tests for best V vs base models
print('\n3. DM Tests — Best V-strategy vs base models:')
for h in [1, 5, 22]:
    best_v_name = min(v_all, key=lambda s: qlike(
        v_all[s][h]['rv_actual'].values, v_all[s][h]['rv_pred'].values))
    print(f'\n  H={h} ({best_v_name}):')
    dm_h = dm_all_df[(dm_all_df['Horizon'] == h) & (dm_all_df['Model1'] == best_v_name)]
    for _, r in dm_h.iterrows():
        if r['Model2'] in ['HAR-J', 'XGBoost', 'LightGBM']:
            stars = sig_stars(r['p_value'])
            print(f'    vs {r["Model2"]:<12}: DM={r["DM_stat"]:>7.3f}, p={r["p_value"]:.4f} {stars:>4} -> {r["Winner"]}')

# 4. Shrinkage sensitivity
print('\n4. Shrinkage Sensitivity:')
for _, row in shrink_df.iterrows():
    marker = ' <-- fixed' if row['shrinkage'] == 0.3 else ''
    print(f'  lambda={row["shrinkage"]:.1f}: H=1={row["H=1"]:.4f}  H=5={row["H=5"]:.4f}  H=22={row["H=22"]:.4f}{marker}')

# 5. Recommendation
print('\n5. Recommendation for dissertation:')
for h_col, h_val in [('H=1', 1), ('H=5', 5), ('H=22', 22)]:
    best_idx = summary_df[h_col].idxmin()
    best_name = summary_df.loc[best_idx, 'Model']
    best_q = summary_df.loc[best_idx, h_col]

    # Compare V strategies only
    v_summary = summary_df[summary_df['Model'].str.startswith('V')]
    best_v_idx = v_summary[h_col].idxmin()
    best_v_name = v_summary.loc[best_v_idx, 'Model']
    best_v_q = v_summary.loc[best_v_idx, h_col]

    # Improvement over V1
    v1_q = summary_df[summary_df['Model'] == 'V1_Adaptive'][h_col].values[0]
    improvement = (v1_q - best_v_q) / v1_q * 100

    print(f'  H={h_val}: Overall best = {best_name} ({best_q:.4f})')
    print(f'         Best V-strategy = {best_v_name} ({best_v_q:.4f}), '
          f'improvement over V1: {improvement:+.1f}%')

# 6. Saved files
print('\n6. Saved files:')
for d in ['data/predictions/walk_forward', 'results/tables', 'results/figures']:
    files = sorted([f for f in os.listdir(d) if 'v2' in f.lower() or 'shrink' in f.lower()
                     or 'strategy' in f.lower() or 'summary' in f.lower()])
    if files:
        print(f'  {d}/:')
        for f in files:
            print(f'    {f}')

print('\nDONE!')
