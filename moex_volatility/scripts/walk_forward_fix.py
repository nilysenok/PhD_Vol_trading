#!/usr/bin/env python3
"""Fix walk-forward results: filter rv_actual<=0 rows, recompute stages 2-4 from saved parquets."""

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
# UTILITIES (fixed: clip y_true too)
# ============================================================
def qlike(y_true, y_pred):
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def qlike_losses(y_true, y_pred):
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_pred = np.clip(y_pred, 1e-10, None)
    return y_true / y_pred - np.log(y_true / y_pred) - 1

def dm_test(loss1, loss2, h=1):
    """Diebold-Mariano test with Newey-West HAC."""
    mask = np.isfinite(loss1) & np.isfinite(loss2)
    loss1, loss2 = loss1[mask], loss2[mask]
    d = loss1 - loss2
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    d_mean = np.mean(d)
    lag = max(1, h - 1)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, lag + 1):
        if k < T:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += 2 * (1 - k / (lag + 1)) * gamma_k
    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

# ============================================================
# LOAD SAVED PREDICTIONS
# ============================================================
print('=' * 70)
print('LOADING SAVED PREDICTIONS')
print('=' * 70)

models_list = ['har', 'xgboost', 'lightgbm', 'hybrid']
model_labels = {'har': 'HAR-J', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'hybrid': 'Hybrid'}

all_preds = {}
for model in models_list:
    for h in [1, 5, 22]:
        for strategy in ['annual', 'quarterly']:
            fp = f'data/predictions/walk_forward/{model}_h{h}_{strategy}.parquet'
            if os.path.exists(fp):
                df = pd.read_parquet(fp)
                # Filter out rv_actual <= 0
                n_before = len(df)
                df = df[df['rv_actual'] > 0].reset_index(drop=True)
                n_dropped = n_before - len(df)
                all_preds[(model, h, strategy)] = df
                if n_dropped > 0:
                    print(f'  {model} H={h} {strategy}: loaded {len(df)} rows (dropped {n_dropped} with rv_actual<=0)')
                else:
                    print(f'  {model} H={h} {strategy}: loaded {len(df)} rows')

# Load hybrid weights
hw_df = pd.read_csv('results/tables/hybrid_weights.csv')

test_years = sorted(pd.concat([all_preds[k] for k in all_preds]).year.unique())
print(f'\nTest years: {test_years}')
print(f'Total prediction sets: {len(all_preds)}')

# Load config for feature importance
with open('data/prepared/config.json') as f:
    config = json.load(f)
feature_cols = config['feature_cols']

# ============================================================
# STAGE 2: QLIKE TABLES
# ============================================================
print('\n' + '=' * 70)
print('STAGE 2: QLIKE TABLES')
print('=' * 70)

for h in [1, 5, 22]:
    print(f'\n--- H={h}: QLIKE by Year (annual) ---')
    rows = []
    for model in models_list:
        key = (model, h, 'annual')
        if key not in all_preds:
            continue
        combined = all_preds[key]
        row = {'Model': model_labels[model]}
        for year in test_years:
            yr_data = combined[combined['year'] == year]
            if len(yr_data) > 0:
                row[str(year)] = qlike(yr_data['rv_actual'].values, yr_data['rv_pred'].values)
        row['Mean'] = np.mean([v for k, v in row.items() if k != 'Model' and isinstance(v, float)])
        rows.append(row)

    df_table = pd.DataFrame(rows)
    df_table.to_csv(f'results/tables/qlike_by_year_h{h}.csv', index=False)
    print(df_table.to_string(index=False, float_format='%.4f'))

    for col in df_table.columns[1:]:
        vals = pd.to_numeric(df_table[col], errors='coerce').values
        if np.any(np.isfinite(vals)):
            best_idx = np.nanargmin(vals)
            print(f'  Best {col}: {df_table.iloc[best_idx]["Model"]}')

# Annual vs Quarterly
print('\n--- Annual vs Quarterly ---')
aq_rows = []
for model in models_list:
    for h in [1, 5, 22]:
        key_a = (model, h, 'annual')
        key_q = (model, h, 'quarterly')
        if key_a in all_preds and key_q in all_preds:
            ca = all_preds[key_a]
            cq = all_preds[key_q]
            qa = qlike(ca['rv_actual'].values, ca['rv_pred'].values)
            qq = qlike(cq['rv_actual'].values, cq['rv_pred'].values)
            diff = (qq - qa) / qa * 100
            aq_rows.append({'Model': model_labels[model], 'H': h,
                            'Annual': qa, 'Quarterly': qq, 'Diff%': diff})

aq_df = pd.DataFrame(aq_rows)
aq_df.to_csv('results/tables/annual_vs_quarterly.csv', index=False)
print(aq_df.to_string(index=False, float_format='%.4f'))

# Hybrid weights
print('\n--- Hybrid Weights by Year (annual) ---')
hw_annual = hw_df[hw_df['strategy'] == 'annual'][['year', 'horizon', 'w_har', 'method', 'ml_model']]
print(hw_annual.to_string(index=False))

# ============================================================
# STAGE 2b: LaTeX TABLES
# ============================================================
print('\n--- Generating LaTeX tables ---')

for h in [1, 5, 22]:
    key_data = {}
    for model in models_list:
        key = (model, h, 'annual')
        if key in all_preds:
            combined = all_preds[key]
            year_vals = {}
            for year in test_years:
                yr = combined[combined['year'] == year]
                if len(yr) > 0:
                    year_vals[year] = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
            year_vals['Mean'] = np.mean(list(year_vals.values()))
            key_data[model_labels[model]] = year_vals

    # Find best per column
    all_cols = list(test_years) + ['Mean']
    best_per_col = {}
    for col in all_cols:
        vals = {m: key_data[m].get(col, np.inf) for m in key_data}
        best_per_col[col] = min(vals, key=vals.get)

    # Build LaTeX
    cols_str = ' & '.join([str(y) for y in test_years] + ['Mean'])
    latex = f'\\begin{{table}}[htbp]\n\\centering\n\\caption{{Walk-Forward QLIKE by Year, $h={h}$}}\n'
    latex += f'\\label{{tab:wf_qlike_h{h}}}\n'
    latex += '\\begin{tabular}{l' + 'c' * (len(test_years) + 1) + '}\n\\hline\n'
    latex += f'Model & {cols_str} \\\\\n\\hline\n'

    for model_name in ['HAR-J', 'XGBoost', 'LightGBM', 'Hybrid']:
        if model_name not in key_data:
            continue
        vals = key_data[model_name]
        parts = []
        for col in all_cols:
            v = vals.get(col, np.nan)
            if np.isfinite(v):
                s = f'{v:.4f}'
                if best_per_col.get(col) == model_name:
                    s = f'\\textbf{{{s}}}'
                parts.append(s)
            else:
                parts.append('---')
        latex += f'{model_name} & ' + ' & '.join(parts) + ' \\\\\n'

    latex += '\\hline\n\\end{tabular}\n\\end{table}\n'

    with open(f'results/tables/comparison_h{h}.tex', 'w') as f:
        f.write(latex)
    print(f'  Saved comparison_h{h}.tex')

# Summary LaTeX table
latex_sum = '\\begin{table}[htbp]\n\\centering\n\\caption{Walk-Forward QLIKE Summary}\n'
latex_sum += '\\label{tab:wf_summary}\n'
latex_sum += '\\begin{tabular}{lccc}\n\\hline\n'
latex_sum += 'Model & $h=1$ & $h=5$ & $h=22$ \\\\\n\\hline\n'

summary_rows = []
for model in models_list:
    row_data = {'Model': model_labels[model]}
    for h in [1, 5, 22]:
        key = (model, h, 'annual')
        if key in all_preds:
            combined = all_preds[key]
            row_data[h] = qlike(combined['rv_actual'].values, combined['rv_pred'].values)
    summary_rows.append(row_data)

# Find best per horizon
best_per_h = {}
for h in [1, 5, 22]:
    vals = {r['Model']: r.get(h, np.inf) for r in summary_rows}
    best_per_h[h] = min(vals, key=vals.get)

for row_data in summary_rows:
    parts = []
    for h in [1, 5, 22]:
        v = row_data.get(h, np.nan)
        if np.isfinite(v):
            s = f'{v:.4f}'
            if best_per_h[h] == row_data['Model']:
                s = f'\\textbf{{{s}}}'
            parts.append(s)
        else:
            parts.append('---')
    latex_sum += f'{row_data["Model"]} & ' + ' & '.join(parts) + ' \\\\\n'

latex_sum += '\\hline\n\\end{tabular}\n\\end{table}\n'
with open('results/tables/summary.tex', 'w') as f:
    f.write(latex_sum)

# Summary CSV
sum_df = pd.DataFrame(summary_rows)
sum_df.to_csv('results/tables/summary.csv', index=False)
print('  Saved summary.tex, summary.csv')

# ============================================================
# STAGE 3: DIEBOLD-MARIANO TESTS
# ============================================================
print('\n' + '=' * 70)
print('STAGE 3: DIEBOLD-MARIANO TESTS')
print('=' * 70)

dm_results = []

for h in [1, 5, 22]:
    print(f'\n--- H={h} ---')

    preds_by_model = {}
    for model in models_list:
        key = (model, h, 'annual')
        if key in all_preds:
            preds_by_model[model] = all_preds[key]

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

    losses = {}
    for model in models_list:
        if f'pred_{model}' in base.columns:
            losses[model] = qlike_losses(actual, base[f'pred_{model}'].values)

    pairs = [('hybrid', 'xgboost'), ('hybrid', 'lightgbm'), ('hybrid', 'har'),
             ('xgboost', 'lightgbm'), ('xgboost', 'har'), ('lightgbm', 'har')]

    print(f'  {"Pair":<30} {"DM_stat":>10} {"p-value":>10} {"Winner":>12}')
    print(f'  {"-"*62}')

    for m1, m2 in pairs:
        if m1 in losses and m2 in losses:
            stat, pval = dm_test(losses[m1], losses[m2], h=h)
            if np.isfinite(stat):
                winner = model_labels[m1] if stat < 0 else model_labels[m2]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                print(f'  {model_labels[m1]+" vs "+model_labels[m2]:<30} {stat:>10.3f} {pval:>10.4f} {winner:>10}{sig}')
            else:
                winner = 'N/A'
                print(f'  {model_labels[m1]+" vs "+model_labels[m2]:<30} {"nan":>10} {"nan":>10} {"N/A":>10}')

            dm_results.append({
                'Horizon': h, 'Model1': model_labels[m1], 'Model2': model_labels[m2],
                'DM_stat': stat, 'p_value': pval, 'Winner': winner
            })

    # P-value matrix
    print(f'\n  P-value matrix (positive DM = model2 better):')
    mlist = [m for m in models_list if m in losses]
    header = f'  {"":>12}' + ''.join(f'{model_labels[m]:>12}' for m in mlist)
    print(header)
    for m1 in mlist:
        row_str = f'  {model_labels[m1]:>12}'
        for m2 in mlist:
            if m1 == m2:
                row_str += f'{"---":>12}'
            else:
                stat, pval = dm_test(losses[m1], losses[m2], h=h)
                if np.isfinite(pval):
                    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    row_str += f'{pval:>10.4f}{sig:>2}'
                else:
                    row_str += f'{"nan":>12}'
        print(row_str)

dm_df = pd.DataFrame(dm_results)
for h in [1, 5, 22]:
    dm_h = dm_df[dm_df['Horizon'] == h]
    if len(dm_h) > 0:
        dm_h.to_csv(f'results/tables/dm_tests_h{h}.csv', index=False)

# DM LaTeX table
latex_dm = '\\begin{table}[htbp]\n\\centering\n\\caption{Diebold-Mariano Test Results}\n'
latex_dm += '\\label{tab:dm_tests}\n'
latex_dm += '\\begin{tabular}{llrrrl}\n\\hline\n'
latex_dm += 'Horizon & Pair & DM stat & p-value & Winner & Sig. \\\\\n\\hline\n'

for _, row in dm_df.iterrows():
    if np.isfinite(row['DM_stat']):
        sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.1 else ''
        latex_dm += f'$h={int(row["Horizon"])}$ & {row["Model1"]} vs {row["Model2"]} & '
        latex_dm += f'{row["DM_stat"]:.3f} & {row["p_value"]:.4f} & {row["Winner"]} & {sig} \\\\\n'

latex_dm += '\\hline\n\\end{tabular}\n\\end{table}\n'
with open('results/tables/dm_tests.tex', 'w') as f:
    f.write(latex_dm)
print('\n  Saved dm_tests.tex')

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
        combined = all_preds[key]
        year_qlikes = []
        for year in test_years:
            yr = combined[combined['year'] == year]
            if len(yr) > 0:
                q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                if np.isfinite(q):
                    year_qlikes.append(q)
        if year_qlikes:
            means.append(np.mean(year_qlikes))
            stds.append(np.std(year_qlikes))
            labels.append(model_labels[model])

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=[colors[l] for l in labels], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(f'H={h}', fontsize=14)
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)

fig.suptitle('Walk-Forward QLIKE (mean ± std across years)', fontsize=14)
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
        combined = all_preds[key]
        years_plot = []
        qlikes_plot = []
        for year in test_years:
            yr = combined[combined['year'] == year]
            if len(yr) > 0:
                q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                if np.isfinite(q):
                    years_plot.append(year)
                    qlikes_plot.append(q)
        ax.plot(years_plot, qlikes_plot, 'o-', label=model_labels[model],
                color=colors[model_labels[model]], linewidth=2, markersize=6)

    # Crisis shading
    ax.axvspan(2019.8, 2020.2, alpha=0.15, color='gray', label='COVID-19')
    ax.axvspan(2021.8, 2022.2, alpha=0.15, color='red', label='Sanctions')

    ax.set_title(f'H={h}', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('QLIKE' if idx == 0 else '', fontsize=12)
    ax.legend(fontsize=9)

fig.suptitle('Walk-Forward QLIKE by Year', fontsize=14)
fig.tight_layout()
fig.savefig('results/figures/qlike_by_year.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/qlike_by_year.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 3: Actual vs Predicted (multiple tickers, 2019 + 2022) ---
print('  Fig 3: Actual vs predicted...')
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
plot_configs = [
    ('SBER', 2019, 'SBER 2019 (normal)'),
    ('SBER', 2022, 'SBER 2022 (sanctions)'),
]

for col_idx, (ticker_plot, year_plot, title_extra) in enumerate(plot_configs):
    for row_idx, h in enumerate([1, 5, 22]):
        ax = axes[row_idx, col_idx]
        key = ('hybrid', h, 'annual')
        if key in all_preds:
            combined = all_preds[key]
            mask = (combined['year'] == year_plot) & (combined['ticker'] == ticker_plot)
            sub = combined[mask].sort_values('date')
            if len(sub) > 0:
                dates = pd.to_datetime(sub['date'])
                ax.plot(dates, sub['rv_actual'], 'k-', alpha=0.7, linewidth=1, label='Actual RV')
                ax.plot(dates, sub['rv_pred'], 'r-', alpha=0.8, linewidth=1.5, label='Hybrid Pred')
                ax.set_title(f'{title_extra}, H={h}', fontsize=12)
                ax.legend(fontsize=9)
                ax.set_ylabel('Realized Variance', fontsize=10)

fig.suptitle('Actual vs Predicted Realized Variance', fontsize=14)
fig.tight_layout()
fig.savefig('results/figures/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/actual_vs_predicted.pdf', bbox_inches='tight')
plt.close(fig)

# Per-ticker predictions for AFLT
for h in [1, 5, 22]:
    key = ('hybrid', h, 'annual')
    if key in all_preds:
        combined = all_preds[key]
        for ticker in ['AFLT']:
            mask = combined['ticker'] == ticker
            sub = combined[mask].sort_values('date')
            if len(sub) > 0:
                fig, ax = plt.subplots(figsize=(14, 4))
                dates = pd.to_datetime(sub['date'])
                ax.plot(dates, sub['rv_actual'], 'k-', alpha=0.6, linewidth=0.8, label='Actual')
                ax.plot(dates, sub['rv_pred'], 'r-', alpha=0.7, linewidth=1, label='Hybrid')
                ax.set_title(f'{ticker} — H={h} (2017-2025)', fontsize=13)
                ax.legend()
                ax.set_ylabel('RV')
                fig.tight_layout()
                fig.savefig(f'results/figures/predictions_{ticker}_h{h}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

# --- FIGURE 4: Hybrid Weights ---
print('  Fig 4: Hybrid weights...')
fig, ax = plt.subplots(figsize=(10, 6))
hw_annual = hw_df[hw_df['strategy'] == 'annual']
markers = {1: 'o', 5: 's', 22: '^'}
for h in [1, 5, 22]:
    sub = hw_annual[hw_annual['horizon'] == h].sort_values('year')
    if len(sub) > 0:
        ax.plot(sub['year'], sub['w_har'], f'{markers[h]}-', label=f'H={h}', linewidth=2, markersize=8)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('$w_{HAR}$ (HAR-J weight)', fontsize=12)
ax.set_title('Hybrid Weights Over Time (Annual Walk-Forward)', fontsize=14)
ax.legend(fontsize=12)
ax.set_ylim(-0.05, 0.65)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
fig.tight_layout()
fig.savefig('results/figures/hybrid_weights.png', dpi=150, bbox_inches='tight')
fig.savefig('results/figures/hybrid_weights.pdf', bbox_inches='tight')
plt.close(fig)

# --- FIGURE 5: Feature Importance ---
print('  Fig 5: Feature importance...')
try:
    import xgboost as xgb_lib
    xgb_model = xgb_lib.Booster()
    xgb_model.load_model('models/xgboost/model_h1.json')
    importance = xgb_model.get_score(importance_type='gain')
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
                combined = all_preds[key]
                yr = combined[combined['year'] == year]
                if len(yr) > 0:
                    q = qlike(yr['rv_actual'].values, yr['rv_pred'].values)
                    if np.isfinite(q):
                        year_qlikes[model_labels[model]] = q
        if year_qlikes:
            sorted_models = sorted(year_qlikes.items(), key=lambda x: x[1])
            ranks = {m: r + 1 for r, (m, _) in enumerate(sorted_models)}
            rank_data.append({'Year': year, **ranks})

    if rank_data:
        rank_df = pd.DataFrame(rank_data).set_index('Year')
        n_models = len(rank_df.columns)
        im = ax.imshow(rank_df.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=n_models)
        ax.set_xticks(range(len(rank_df.columns)))
        ax.set_xticklabels(rank_df.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(rank_df.index)))
        ax.set_yticklabels(rank_df.index, fontsize=10)
        ax.set_title(f'H={h}', fontsize=14)
        for i in range(len(rank_df.index)):
            for j in range(len(rank_df.columns)):
                ax.text(j, i, f'{int(rank_df.values[i, j])}', ha='center', va='center',
                        fontsize=12, fontweight='bold')

fig.suptitle('Model Rankings by Year (1=Best)', fontsize=14)
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
            preds_by_model[model] = all_preds[key]

    if len(preds_by_model) < 2:
        continue

    base = preds_by_model[models_list[0]].rename(columns={'rv_pred': f'pred_{models_list[0]}'})
    for model in models_list[1:]:
        if model in preds_by_model:
            df_m = preds_by_model[model][['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'pred_{model}'})
            base = base.merge(df_m, on=['date', 'ticker'], how='inner')

    actual = base['rv_actual'].values
    mlist = [m for m in models_list if f'pred_{m}' in base.columns]
    errors = {}
    for m in mlist:
        e = qlike_losses(actual, base[f'pred_{m}'].values)
        errors[model_labels[m]] = e

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
# STAGE 5: METRICS TABLE (multiple metrics)
# ============================================================
print('\n' + '=' * 70)
print('STAGE 5: ADDITIONAL METRICS')
print('=' * 70)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

for h in [1, 5, 22]:
    print(f'\n--- H={h}: Multiple Metrics (annual, all years) ---')
    rows = []
    for model in models_list:
        key = (model, h, 'annual')
        if key not in all_preds:
            continue
        combined = all_preds[key]
        y_true = combined['rv_actual'].values
        y_pred = combined['rv_pred'].values
        rows.append({
            'Model': model_labels[model],
            'QLIKE': qlike(y_true, y_pred),
            'MSE': mse(y_true, y_pred),
            'MAE': mae(y_true, y_pred),
            'R2': r2(y_true, y_pred),
            'N': len(y_true)
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f'results/tables/metrics_h{h}.csv', index=False)
    print(metrics_df.to_string(index=False, float_format='%.4f'))

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n' + '=' * 70)
print('FINAL SUMMARY')
print('=' * 70)

print('\n1. Walk-Forward QLIKE (Annual, all years, rv_actual>0):')
for h in [1, 5, 22]:
    print(f'\n  H={h}:')
    for model in models_list:
        key = (model, h, 'annual')
        if key in all_preds:
            combined = all_preds[key]
            q = qlike(combined['rv_actual'].values, combined['rv_pred'].values)
            print(f'    {model_labels[model]:>12}: {q:.4f}')

print('\n2. DM Test Summary (Hybrid vs others):')
for _, row in dm_df[dm_df['Model1'] == 'Hybrid'].iterrows():
    if np.isfinite(row['p_value']):
        sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.1 else 'ns'
        print(f'  H={int(row["Horizon"])}: Hybrid vs {row["Model2"]}: DM={row["DM_stat"]:.3f}, p={row["p_value"]:.4f} {sig}')

print('\n3. Annual vs Quarterly:')
for _, row in aq_df.iterrows():
    better = 'Quarterly' if row['Diff%'] < 0 else 'Annual'
    print(f'  {row["Model"]} H={int(row["H"])}: Annual={row["Annual"]:.4f}, Quarterly={row["Quarterly"]:.4f} ({row["Diff%"]:+.1f}%) -> {better}')

print('\n4. Saved files:')
for d in ['data/predictions/walk_forward', 'results/tables', 'results/figures']:
    files = sorted(os.listdir(d))
    print(f'  {d}/: {len(files)} files')
    for f in files:
        print(f'    {f}')

print('\nDONE!')
