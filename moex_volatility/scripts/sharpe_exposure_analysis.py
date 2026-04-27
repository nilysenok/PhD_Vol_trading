"""
Comprehensive Sharpe / Exposure-Scaling Analysis for Hourly Approach-A Strategies.
Computes old vs new Sharpe, exposure scaling optimization, and year-by-year metrics.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
POSITIONS_PATH = '/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility/results/final/strategies/walkforward_v4/daily_positions.parquet'
KEY_RATE_PATH  = '/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet'

COMM_LEVELS = [0.0004, 0.0005]
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
STRATEGIES = ['S1_MeanRev', 'S2_Bollinger', 'S3_Donchian',
              'S4_Supertrend', 'S5_PivotPoints', 'S6_VWAP']
TEST_YEARS = [2022, 2023, 2024, 2025]
BPY = 252  # business days per year (after daily aggregation)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
pos_all = pd.read_parquet(POSITIONS_PATH)
key_rate = pd.read_parquet(KEY_RATE_PATH)

# Filter to hourly, approach A
pos = pos_all[(pos_all.tf == 'hourly') & (pos_all.approach == 'A')].copy()
print(f"Hourly Approach-A rows: {len(pos):,}")

# Create calendar date for aggregation
pos['cal_date'] = pos['date'].dt.normalize()

# ── Aggregate hourly bars to daily ─────────────────────────────────────────────
print("Aggregating hourly bars to daily...")

def agg_to_daily(grp):
    """Aggregate hourly bars within a day to daily level."""
    return pd.Series({
        'daily_gross_return': grp['daily_gross_return'].sum(),
        'position_last': grp['position'].iloc[-1],     # end-of-day position
        'position_first': grp['position'].iloc[0],      # start-of-day position
    })

daily = (pos
    .groupby(['strategy', 'ticker', 'test_year', 'cal_date'])
    .apply(agg_to_daily, include_groups=False)
    .reset_index()
    .rename(columns={'cal_date': 'date'})
)

# Sort properly for delta-position computation
daily = daily.sort_values(['strategy', 'ticker', 'date']).reset_index(drop=True)

# Compute |Δposition| — change from previous day's last position to today's first bar,
# plus intraday position changes are already captured in gross_return.
# For commission: what matters is the position CHANGE day-over-day (end to end).
# |Δposition| = |position_last_today - position_last_yesterday|
daily['prev_pos'] = daily.groupby(['strategy', 'ticker'])['position_last'].shift(1).fillna(0)
daily['abs_delta_pos'] = (daily['position_last'] - daily['prev_pos']).abs()

# But we also need intraday position changes for commission.
# Actually, looking at the hourly data: daily_gross_return already includes all intraday
# position * log_return products. For commission, we need total turnover.
# Let's compute total absolute position changes within each day from hourly data.

print("Computing intraday turnover from hourly data...")

def compute_intraday_turnover(grp):
    """Sum of |Δposition| within a day (all hourly transitions)."""
    positions = grp['position'].values
    # Include transition from previous day's last position
    # We'll handle that separately; here just intraday changes
    diffs = np.abs(np.diff(positions))
    return pd.Series({
        'intraday_turnover': diffs.sum(),
        'position_last': positions[-1],
    })

# We need the full turnover including day boundaries.
# Re-compute: for each (strategy, ticker), compute |Δposition| at every hourly step.
pos_sorted = pos.sort_values(['strategy', 'ticker', 'date']).reset_index(drop=True)

print("Computing total turnover per day...")
turnover_list = []
for (strat, tick), grp in pos_sorted.groupby(['strategy', 'ticker']):
    positions = grp['position'].values
    dates = grp['cal_date'].values
    test_years = grp['test_year'].values

    # Prepend 0 for the first bar (start from flat)
    prev_positions = np.concatenate([[0.0], positions[:-1]])
    abs_deltas = np.abs(positions - prev_positions)

    # But at day boundaries within same test_year, the prev is the last bar of previous day
    # which is already handled by the shift above. However across test_year boundaries,
    # we should reset. Let's detect test_year changes.
    year_changes = np.concatenate([[True], test_years[1:] != test_years[:-1]])
    abs_deltas[year_changes] = np.abs(positions[year_changes])  # from flat

    tmpdf = pd.DataFrame({
        'strategy': strat, 'ticker': tick,
        'date': dates, 'test_year': test_years,
        'abs_delta': abs_deltas,
    })
    daily_turnover = tmpdf.groupby(['strategy', 'ticker', 'test_year', 'date']).agg(
        total_turnover=('abs_delta', 'sum')
    ).reset_index()
    turnover_list.append(daily_turnover)

turnover_daily = pd.concat(turnover_list, ignore_index=True)
turnover_daily.rename(columns={'date': 'date'}, inplace=True)

# Merge turnover into daily
daily = daily.merge(turnover_daily, on=['strategy', 'ticker', 'test_year', 'date'], how='left')
daily['total_turnover'] = daily['total_turnover'].fillna(daily['abs_delta_pos'])

print(f"Daily aggregated rows: {len(daily):,}")

# ── Align key rate ─────────────────────────────────────────────────────────────
key_rate = key_rate.sort_values('date')
key_rate.set_index('date', inplace=True)

# ── Helper: build EW portfolio returns ─────────────────────────────────────────
def build_ew_portfolio(daily_df, comm, strategies=STRATEGIES, test_years=TEST_YEARS):
    """
    For each strategy, build EW portfolio net returns across tickers.
    Returns dict: strategy -> DataFrame(date, ew_return, exposure_frac, n_tickers)
    """
    results = {}
    for strat in strategies:
        sdf = daily_df[(daily_df.strategy == strat) & (daily_df.test_year.isin(test_years))].copy()
        if len(sdf) == 0:
            continue

        # Net return per ticker per day
        sdf['net_return'] = sdf['daily_gross_return'] - sdf['total_turnover'] * comm

        # EW portfolio: mean across tickers per date
        port = sdf.groupby('date').agg(
            ew_return=('net_return', 'mean'),
            n_tickers=('ticker', 'nunique'),
            n_in_position=('position_last', lambda x: (x != 0).sum()),
            total_tickers=('ticker', 'count'),
        ).reset_index()
        port['exposure_frac'] = port['n_in_position'] / port['n_tickers']
        port = port.sort_values('date').reset_index(drop=True)

        # Align key rate
        port = port.set_index('date')
        port['key_rate'] = key_rate['key_rate'].reindex(port.index).ffill().bfill()
        port = port.reset_index()

        results[strat] = port
    return results


def compute_sharpe_old(daily_returns):
    """Old Sharpe: no rf subtraction."""
    mu = daily_returns.mean()
    sigma = daily_returns.std(ddof=1)
    if sigma == 0:
        return 0.0
    return mu / sigma * np.sqrt(BPY)


def compute_sharpe_new(port_df, k=1, comm=0.0004, leverage_spread=0.02):
    """
    New Sharpe with rf, FDR, and optional leverage cost.
    port_df must have: ew_return (net 1x), exposure_frac, key_rate, date
    """
    dr = port_df.copy()

    # Scale trading return
    trading_ret = k * dr['ew_return']

    # Capital exposure
    capital_exp = k * dr['exposure_frac']

    # Daily rf
    rf_daily = dr['key_rate'] / 252

    # FDR on idle capital
    idle_frac = np.maximum(0, 1 - capital_exp)
    fdr = idle_frac * rf_daily

    # Leverage cost when capital_exposure > 1
    leverage_cost = np.maximum(0, capital_exp - 1) * (dr['key_rate'] + leverage_spread) / 252

    # Total daily return
    total_daily = trading_ret + fdr - leverage_cost

    # Annualized metrics
    cumulative = (1 + total_daily).prod()
    n_days = len(total_daily)
    n_years = n_days / 252
    ann_return = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = total_daily.std(ddof=1) * np.sqrt(252)
    ann_rf = (rf_daily * 252).mean()

    sharpe = (ann_return - ann_rf) / ann_vol if ann_vol > 0 else 0

    # MDD
    cum_ret = (1 + total_daily).cumprod()
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    mdd = drawdown.min()

    return {
        'sharpe': sharpe,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'ann_rf': ann_rf,
        'mdd': mdd,
        'avg_exposure': capital_exp.mean(),
        'max_exposure': capital_exp.max(),
        'total_daily': total_daily,
        'cum_ret': cum_ret,
    }


# ── STEP 1 & 2: Build portfolios and compute old vs new Sharpe ────────────────
print("\n" + "="*80)
print("STEP 1 & 2: Old vs New Sharpe at 1x")
print("="*80)

for comm in COMM_LEVELS:
    portfolios = build_ew_portfolio(daily, comm)

    print(f"\n{'─'*70}")
    print(f"Commission = {comm*100:.2f}%")
    print(f"{'─'*70}")
    print(f"{'Strategy':<18} | {'Old Sharpe':>10} | {'New Sharpe':>10} | {'Δ':>8} | {'rf_avg%':>7} | {'AnnRet%':>8} | {'MDD%':>7}")
    print(f"{'─'*18}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}-+-{'─'*7}-+-{'─'*8}-+-{'─'*7}")

    for strat in STRATEGIES:
        if strat not in portfolios:
            continue
        port = portfolios[strat]

        old_sharpe = compute_sharpe_old(port['ew_return'])
        new_metrics = compute_sharpe_new(port, k=1, comm=comm)
        new_sharpe = new_metrics['sharpe']
        delta = new_sharpe - old_sharpe
        rf_avg = new_metrics['ann_rf'] * 100
        ann_ret = new_metrics['ann_return'] * 100
        mdd = new_metrics['mdd'] * 100

        print(f"{strat:<18} | {old_sharpe:>10.4f} | {new_sharpe:>10.4f} | {delta:>+8.4f} | {rf_avg:>6.1f}% | {ann_ret:>+7.2f}% | {mdd:>6.2f}%")


# ── STEP 3: Exposure scaling optimization ─────────────────────────────────────
print("\n\n" + "="*80)
print("STEP 3 & 4: Exposure Scaling Optimization")
print("="*80)

# First, compute max exposure on TRAIN period (pre-2022) for each strategy
# to determine max allowed k
print("\nComputing train-period max exposure fractions...")

train_portfolios = build_ew_portfolio(daily, comm=0.0004,
                                       test_years=[2020, 2021])

# Also need to check pre-2022 data in the full daily set
# Actually, test_years 2020, 2021 are the training period data we have.
# For each strategy, max_exposure_frac = max across training days
max_exp_train = {}
for strat in STRATEGIES:
    if strat in train_portfolios:
        port = train_portfolios[strat]
        max_exp = port['exposure_frac'].max()
        max_exp_train[strat] = max_exp
    else:
        # If no training data, use OOS data to estimate
        max_exp_train[strat] = 1.0  # conservative

print(f"\n{'Strategy':<18} | {'Max Exp(train)':>14} | {'Max k allowed':>13}")
print(f"{'─'*18}-+-{'─'*14}-+-{'─'*13}")
for strat in STRATEGIES:
    me = max_exp_train[strat]
    max_k = int(1.0 / me) if me > 0 else 999
    print(f"{strat:<18} | {me:>13.2%} | {max_k:>13d}")


for comm_idx, comm in enumerate(COMM_LEVELS):
    portfolios_oos = build_ew_portfolio(daily, comm)

    print(f"\n{'='*80}")
    print(f"TABLE {'2' if comm_idx==0 else '2b'}: Full Sharpe map across k values (comm={comm*100:.2f}%)")
    print(f"{'='*80}")

    # Header
    k_display = [1, 2, 3, 5, 7, 10, 15, 20]
    header = f"{'Strategy':<18} |"
    for k in k_display:
        header += f" k={k:<4}"
    header += f" | {'k*':>3} {'max_k':>5}"
    print(header)
    print("─" * len(header))

    # Store results for Tables 3-5
    table3_data = []
    table5_data = []

    for strat in STRATEGIES:
        if strat not in portfolios_oos:
            continue
        port = portfolios_oos[strat]

        me = max_exp_train.get(strat, 1.0)
        max_k_allowed = int(1.0 / me) if me > 0 else 999

        sharpe_map = {}
        best_k = 1
        best_sharpe = -999
        best_metrics = None

        for k in K_VALUES:
            metrics = compute_sharpe_new(port, k=k, comm=comm)
            sharpe_map[k] = metrics['sharpe']

            # Check constraint: k <= max_k_allowed
            if k <= max_k_allowed and metrics['sharpe'] > best_sharpe:
                best_sharpe = metrics['sharpe']
                best_k = k
                best_metrics = metrics

        # Print Sharpe map row
        row = f"{strat:<18} |"
        for k in k_display:
            s = sharpe_map.get(k, float('nan'))
            row += f" {s:>5.2f}"
        row += f" | {best_k:>3} {max_k_allowed:>5}"
        print(row)

        # Compute sharpe at 1x without rf for Table 3
        sharpe_1x_old = compute_sharpe_old(port['ew_return'])
        metrics_1x = compute_sharpe_new(port, k=1, comm=comm)

        table3_data.append({
            'strategy': strat,
            'k_star': best_k,
            'sharpe_1x_no_rf': sharpe_1x_old,
            'sharpe_1x_rf': metrics_1x['sharpe'],
            'sharpe_kstar_rf': best_sharpe,
            'ret_kstar': best_metrics['ann_return'] * 100,
            'mdd_kstar': best_metrics['mdd'] * 100,
            'avg_exp_kstar': best_metrics['avg_exposure'] * 100,
            'max_exp_kstar': best_metrics['max_exposure'] * 100,
        })

        # Year-by-year returns at k*
        year_rets = {}
        for yr in TEST_YEARS:
            port_yr = build_ew_portfolio(daily, comm, strategies=[strat], test_years=[yr])
            if strat in port_yr and len(port_yr[strat]) > 0:
                m = compute_sharpe_new(port_yr[strat], k=best_k, comm=comm)
                year_rets[yr] = m['ann_return'] * 100
            else:
                year_rets[yr] = float('nan')
        table5_data.append({'strategy': strat, 'k_star': best_k, **year_rets})

    # ── TABLE 3 / 4 ───────────────────────────────────────────────────────────
    tnum = '3' if comm_idx == 0 else '4'
    print(f"\n{'='*80}")
    print(f"TABLE {tnum}: Optimal k* metrics (comm={comm*100:.2f}%)")
    print(f"{'='*80}")
    print(f"{'Strategy':<18} | {'k*':>3} | {'Sh(1x,no rf)':>12} | {'Sh(1x,+rf)':>10} | {'Sh(k*,+rf)':>10} | {'Ret%(k*)':>8} | {'MDD%(k*)':>8} | {'AvgExp%':>7} | {'MaxExp%':>7}")
    print(f"{'─'*18}-+-{'─'*3}-+-{'─'*12}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}-+-{'─'*7}-+-{'─'*7}")
    for r in table3_data:
        print(f"{r['strategy']:<18} | {r['k_star']:>3} | {r['sharpe_1x_no_rf']:>12.4f} | {r['sharpe_1x_rf']:>10.4f} | {r['sharpe_kstar_rf']:>10.4f} | {r['ret_kstar']:>+7.2f}% | {r['mdd_kstar']:>7.2f}% | {r['avg_exp_kstar']:>6.1f}% | {r['max_exp_kstar']:>6.1f}%")

    # ── TABLE 5 ────────────────────────────────────────────────────────────────
    if comm_idx == 0:
        print(f"\n{'='*80}")
        print(f"TABLE 5: Year-by-year returns at optimal k* (comm={comm*100:.2f}%)")
        print(f"{'='*80}")
        print(f"{'Strategy':<18} | {'k*':>3} | {'2022':>8} | {'2023':>8} | {'2024':>8} | {'2025':>8}")
        print(f"{'─'*18}-+-{'─'*3}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}")
        for r in table5_data:
            print(f"{r['strategy']:<18} | {r['k_star']:>3} | {r.get(2022, float('nan')):>+7.2f}% | {r.get(2023, float('nan')):>+7.2f}% | {r.get(2024, float('nan')):>+7.2f}% | {r.get(2025, float('nan')):>+7.2f}%")

print("\n\nDone.")
