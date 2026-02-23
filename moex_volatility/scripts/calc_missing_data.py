"""
Compute missing data for final summary:
- Descriptive statistics (17 tickers, OOS 2022-2026)
- Strategy correlations (S1-S6)
- Annual returns breakdown
- Win rate / Sortino
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
BPY = 252
COMM = 0.0040

TICKERS = ['AFLT','ALRS','HYDR','IRAO','LKOH','LSRG','MGNT','MOEX',
           'MTLR','MTSS','NVTK','OGKB','PHOR','RTKM','SBER','TATN','VTBR']
STRATEGIES = ['S1_MeanRev','S2_Bollinger','S3_Donchian',
              'S4_Supertrend','S5_PivotPoints','S6_VWAP']
BEST_MAP = {'S1_MeanRev':'D', 'S2_Bollinger':'C', 'S3_Donchian':'D',
            'S4_Supertrend':'D', 'S5_PivotPoints':'B', 'S6_VWAP':'D'}

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
print("Loading daily_positions.parquet...")
pos_path = 'results/final/strategies/walkforward_v4/daily_positions.parquet'
df = pd.read_parquet(pos_path)
df['date'] = pd.to_datetime(df['date'])
print(f"  Loaded {len(df):,} rows")

# Filter: daily timeframe, BCD years
daily = df[df['tf'] == 'daily'].copy()
bcd = daily[daily['test_year'].isin([2022, 2023, 2024, 2025, 2026])].copy()
print(f"  BCD daily rows: {len(bcd):,}")

# Capital efficiency details
det = pd.read_csv('results/capital_efficiency_details.csv',
                   index_col=0, parse_dates=True)


# ──────────────────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS (17 tickers, OOS 2022-2026)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. DESCRIPTIVE STATISTICS (17 tickers)")
print("=" * 80)

# Get daily gross returns per ticker (from approach A, any strategy — they share same returns)
# Use S1_MeanRev approach A as source for daily_gross_return
ticker_stats = []
for ticker in TICKERS:
    mask = (bcd['ticker'] == ticker) & (bcd['strategy'] == 'S1_MeanRev') & (bcd['approach'] == 'A')
    sub = bcd[mask].sort_values('date').drop_duplicates('date')
    r = sub['daily_gross_return'].values
    r = r[np.isfinite(r)]
    if len(r) < 50:
        print(f"  WARNING: {ticker} only {len(r)} obs")
        continue
    ann_ret = np.mean(r) * BPY * 100
    ann_vol = np.std(r, ddof=1) * np.sqrt(BPY) * 100
    skew = pd.Series(r).skew()
    kurt = pd.Series(r).kurtosis()
    min_r = np.min(r) * 100
    max_r = np.max(r) * 100
    n_days = len(r)
    ticker_stats.append(dict(
        Ticker=ticker, N=n_days,
        AnnReturn_pct=round(ann_ret, 2),
        AnnVol_pct=round(ann_vol, 2),
        Skewness=round(skew, 3),
        Kurtosis=round(kurt, 2),
        MinDaily_pct=round(min_r, 2),
        MaxDaily_pct=round(max_r, 2)))

ts_df = pd.DataFrame(ticker_stats)
print(ts_df.to_string(index=False))

# Cross-sectional summary
print(f"\nCross-sectional summary:")
print(f"  Mean annual return: {ts_df['AnnReturn_pct'].mean():.2f}%")
print(f"  Mean annual vol: {ts_df['AnnVol_pct'].mean():.2f}%")
print(f"  Mean skewness: {ts_df['Skewness'].mean():.3f}")
print(f"  Mean kurtosis: {ts_df['Kurtosis'].mean():.2f}")


# ──────────────────────────────────────────────────────────
# 8. WIN RATE, SORTINO
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("8. WIN RATE, SORTINO, AVG TRADE (per-strategy EW portfolios)")
print("=" * 80)

def net_returns_series(pos, gross_r):
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * COMM

def strategy_portfolio_returns(strat, approach):
    """EW net returns for a strategy×approach across 17 tickers."""
    mask = (bcd['strategy'] == strat) & (bcd['approach'] == approach)
    sub = bcd[mask].copy()
    all_dates = sorted(sub['date'].unique())
    ret_per_ticker = {}
    for ticker in TICKERS:
        t_mask = sub['ticker'] == ticker
        t_df = sub[t_mask].sort_values('date').drop_duplicates('date').set_index('date')
        if len(t_df) < 10:
            continue
        pos = t_df['position'].values
        gross_r = t_df['daily_gross_return'].values
        net_r = net_returns_series(pos, gross_r)
        ret_per_ticker[ticker] = pd.Series(net_r, index=t_df.index)
    if not ret_per_ticker:
        return pd.Series(dtype=float)
    ret_df = pd.DataFrame(ret_per_ticker)
    return ret_df.mean(axis=1)

winrate_rows = []
for strat in STRATEGIES:
    for approach in ['A', 'B', 'C', 'D']:
        r = strategy_portfolio_returns(strat, approach)
        if len(r) < 50:
            continue
        r_arr = r.values
        # Win rate: % of non-zero return days that are positive
        nonzero = r_arr[np.abs(r_arr) > 1e-10]
        win_rate = np.mean(nonzero > 0) * 100 if len(nonzero) > 0 else 0

        # Sortino (downside deviation)
        ann_ret = np.mean(r_arr) * BPY
        down_r = r_arr[r_arr < 0]
        downside_dev = np.std(down_r, ddof=1) * np.sqrt(BPY) if len(down_r) > 1 else 1e-12
        sortino = ann_ret / downside_dev

        # Avg trade (mean of nonzero returns)
        avg_trade = np.mean(nonzero) * 100 if len(nonzero) > 0 else 0

        # Sharpe
        ann_vol = np.std(r_arr, ddof=1) * np.sqrt(BPY)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0

        winrate_rows.append(dict(
            Strategy=strat, Approach=approach,
            WinRate_pct=round(win_rate, 1),
            Sortino=round(sortino, 2),
            AvgTrade_bps=round(avg_trade * 100, 2),
            Sharpe=round(sharpe, 2)))

wr_df = pd.DataFrame(winrate_rows)
print(wr_df.to_string(index=False))

# Meta-level win rate / Sortino
print("\n--- Meta-portfolio level ---")
for meta_name in ['META-A', 'META-B', 'META-D', 'META-BEST', 'META-MEAN(BCD)']:
    col = f'{meta_name}_strat'
    if col not in det.columns:
        continue
    r = det[col].values
    r = r[np.isfinite(r)]
    nonzero = r[np.abs(r) > 1e-10]
    win_rate = np.mean(nonzero > 0) * 100 if len(nonzero) > 0 else 0
    ann_ret = np.mean(r) * BPY
    down_r = r[r < 0]
    downside_dev = np.std(down_r, ddof=1) * np.sqrt(BPY) if len(down_r) > 1 else 1e-12
    sortino = ann_ret / downside_dev
    print(f"  {meta_name:20s}: WinRate={win_rate:.1f}%, Sortino={sortino:.2f}")


# ──────────────────────────────────────────────────────────
# 9. CORRELATIONS BETWEEN STRATEGIES (S1-S6)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("9. STRATEGY CORRELATIONS (BEST approach, EW)")
print("=" * 80)

strat_returns = {}
for strat in STRATEGIES:
    best_approach = BEST_MAP[strat]
    r = strategy_portfolio_returns(strat, best_approach)
    if len(r) > 0:
        strat_returns[strat] = r

corr_df = pd.DataFrame(strat_returns)
corr_matrix = corr_df.corr()
print("\nCorrelation matrix (BEST approach):")
print(corr_matrix.round(3).to_string())

# Also for approach A
print("\nCorrelation matrix (Approach A):")
strat_returns_a = {}
for strat in STRATEGIES:
    r = strategy_portfolio_returns(strat, 'A')
    if len(r) > 0:
        strat_returns_a[strat] = r

corr_a_df = pd.DataFrame(strat_returns_a)
corr_a_matrix = corr_a_df.corr()
print(corr_a_matrix.round(3).to_string())

print(f"\nMean pairwise correlation (BEST): {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
print(f"Mean pairwise correlation (A): {corr_a_matrix.values[np.triu_indices_from(corr_a_matrix.values, k=1)].mean():.3f}")


# ──────────────────────────────────────────────────────────
# 10. ANNUAL RETURNS BREAKDOWN
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("10. ANNUAL RETURNS BREAKDOWN")
print("=" * 80)

meta_names = ['META-A', 'META-C', 'META-B', 'META-D', 'META-BEST', 'META-MEAN(BCD)']
years = [2022, 2023, 2024, 2025]

annual_rows = []
for year in years:
    mask = det.index.year == year
    year_data = det[mask]
    n_days_year = len(year_data)
    row = {'Year': year, 'N_days': n_days_year}

    # IMOEX
    imoex_r = year_data['IMOEX'].values
    cum_ret = (1 + imoex_r).prod() - 1
    row['IMOEX'] = round(cum_ret * 100, 2)

    for meta in meta_names:
        col = f'{meta}_strat'
        if col in year_data.columns:
            r = year_data[col].values
            cum_ret = (1 + r).prod() - 1
            row[meta] = round(cum_ret * 100, 2)

    annual_rows.append(row)

# Full period
mask_full = det.index.year.isin(years)
full_data = det[mask_full]
full_row = {'Year': 'Full', 'N_days': len(full_data)}
imoex_full = (1 + full_data['IMOEX'].values).prod() - 1
full_row['IMOEX'] = round(imoex_full * 100, 2)
for meta in meta_names:
    col = f'{meta}_strat'
    if col in full_data.columns:
        r = full_data[col].values
        cum_ret = (1 + r).prod() - 1
        full_row[meta] = round(cum_ret * 100, 2)
annual_rows.append(full_row)

ann_df = pd.DataFrame(annual_rows)
print(ann_df.to_string(index=False))

# Also with MMF
print("\n--- With MMF ---")
annual_mmf_rows = []
for year in years:
    mask = det.index.year == year
    year_data = det[mask]
    row = {'Year': year}
    row['IMOEX'] = round(((1 + year_data['IMOEX'].values).prod() - 1) * 100, 2)
    for meta in ['META-A', 'META-B', 'META-D', 'META-BEST']:
        total_col = f'{meta}_total'
        if total_col in year_data.columns:
            r = year_data[total_col].values
            cum_ret = (1 + r).prod() - 1
            row[f'{meta}+MMF'] = round(cum_ret * 100, 2)
    annual_mmf_rows.append(row)

ammf_df = pd.DataFrame(annual_mmf_rows)
print(ammf_df.to_string(index=False))


print("\n\nDone!")
