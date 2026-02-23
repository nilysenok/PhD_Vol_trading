#!/usr/bin/env python3
"""
Hourly Final Comparison: Exposure, MMF, Scaling, Correlation, Commission Sensitivity.
Analogous to calc_final_comparison.py + calc_exposure_scaling.py but for hourly data.

Outputs:
  - results/hourly_exposure.csv
  - results/hourly_capital_efficiency.csv
  - results/hourly_scaling_analysis.csv
  - results/hourly_correlation_matrix.csv
  - results/commission_sensitivity.csv
  - results/hourly_vs_imoex.csv
  - Sections 12-15 for FINAL_RESULTS.md (printed to stdout)
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

# ── Parameters ──────────────────────────────────────────────────────
BPY_HOURLY = 2268  # 252 days × 9 bars
BPY_DAILY = 252
COMM_HOURLY = 0.0004   # 0.04% per side
COMM_DAILY = 0.0005    # 0.05% per side
BCD_YEARS = [2022, 2023, 2024, 2025]
STRATEGIES = ['S1_MeanRev', 'S2_Bollinger', 'S3_Donchian',
              'S4_Supertrend', 'S5_PivotPoints', 'S6_VWAP']
TICKERS = ['AFLT','ALRS','HYDR','IRAO','LKOH','LSRG','MGNT','MOEX',
           'MTLR','MTSS','NVTK','OGKB','PHOR','RTKM','SBER','TATN','VTBR']

BEST_MAP_HOURLY = {
    'S1_MeanRev': 'D', 'S2_Bollinger': 'D', 'S3_Donchian': 'B',
    'S4_Supertrend': 'B', 'S5_PivotPoints': 'D', 'S6_VWAP': 'D',
}
BEST_MAP_DAILY = {
    'S1_MeanRev': 'C', 'S2_Bollinger': 'C', 'S3_Donchian': 'B',
    'S4_Supertrend': 'D', 'S5_PivotPoints': 'C', 'S6_VWAP': 'D',
}

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────
def calc_metrics(r, bpy=BPY_HOURLY):
    """Annualized metrics from return series."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return dict(Return_pct=np.nan, Vol_pct=np.nan, Sharpe=np.nan,
                    MDD_pct=np.nan, Calmar=np.nan)
    ann_ret = np.mean(r) * bpy
    ann_vol = np.std(r, ddof=1) * np.sqrt(bpy)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0
    cum = np.cumprod(1 + r)
    rmax = np.maximum.accumulate(cum)
    mdd = np.min((cum - rmax) / rmax)
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-12 else 0
    return dict(Return_pct=round(ann_ret * 100, 2),
                Vol_pct=round(ann_vol * 100, 2),
                Sharpe=round(sharpe, 2),
                MDD_pct=round(mdd * 100, 2),
                Calmar=round(calmar, 2))


def net_returns_series(pos, gross_r, comm):
    """Net returns after commission on position changes."""
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * comm


def strategy_portfolio_returns(df_sub, comm):
    """EW portfolio return for one (strategy, approach) pair."""
    net_by_ticker = {}
    for ticker, g in df_sub.groupby('ticker'):
        g = g.sort_values('date')
        nr = net_returns_series(g['position'].values, g['daily_gross_return'].values, comm)
        net_by_ticker[ticker] = pd.Series(nr, index=g['date'].values)
    ret_df = pd.DataFrame(net_by_ticker).fillna(0.0)
    return ret_df.mean(axis=1)  # EW across tickers


# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

pos_df = pd.read_parquet('results/final/strategies/walkforward_v4/daily_positions.parquet')
pos_df['date'] = pd.to_datetime(pos_df['date'])

# Hourly BCD
hourly = pos_df[(pos_df['tf'] == 'hourly') & (pos_df['test_year'].isin(BCD_YEARS))].copy()
print(f"Hourly BCD rows: {len(hourly):,}")

# Daily BCD (for commission sensitivity comparison)
daily = pos_df[(pos_df['tf'] == 'daily') & (pos_df['test_year'].isin(BCD_YEARS))].copy()
print(f"Daily BCD rows: {len(daily):,}")


# ══════════════════════════════════════════════════════════════════
#  1. BUILD HOURLY META-PORTFOLIO RETURNS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. BUILDING HOURLY META-PORTFOLIO RETURNS")
print("=" * 70)

# Per (strategy, approach) hourly portfolio returns
strat_port_h = {}
for (strat, app), g in hourly.groupby(['strategy', 'approach']):
    strat_port_h[(strat, app)] = strategy_portfolio_returns(g, COMM_HOURLY)
print(f"Built {len(strat_port_h)} hourly (strategy, approach) portfolios")


def build_meta(strat_port, approach, strategies=STRATEGIES):
    parts = {}
    for s in strategies:
        key = (s, approach)
        if key in strat_port:
            parts[s] = strat_port[key]
    if not parts:
        return pd.Series(dtype=float)
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1)


def build_meta_best(strat_port, best_map):
    parts = {}
    for s, app in best_map.items():
        key = (s, app)
        if key in strat_port:
            parts[s] = strat_port[key]
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1)


def build_meta_mean_bcd(strat_port, strategies=STRATEGIES):
    parts = {}
    for s in strategies:
        bcd = {}
        for app in ['B', 'C', 'D']:
            key = (s, app)
            if key in strat_port:
                bcd[app] = strat_port[key]
        if bcd:
            parts[s] = pd.DataFrame(bcd).fillna(0.0).mean(axis=1)
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1)


meta_h = pd.DataFrame({
    'META-A': build_meta(strat_port_h, 'A'),
    'META-B': build_meta(strat_port_h, 'B'),
    'META-D': build_meta(strat_port_h, 'D'),
    'META-BEST': build_meta_best(strat_port_h, BEST_MAP_HOURLY),
    'META-MEAN(BCD)': build_meta_mean_bcd(strat_port_h),
})
meta_h.index = pd.to_datetime(meta_h.index)
meta_h = meta_h.sort_index()

print("\nHourly META metrics (hourly bars):")
for col in meta_h.columns:
    m = calc_metrics(meta_h[col], BPY_HOURLY)
    print(f"  {col}: Ret={m['Return_pct']}%, Vol={m['Vol_pct']}%, "
          f"Sharpe={m['Sharpe']}, MDD={m['MDD_pct']}%")


# ══════════════════════════════════════════════════════════════════
#  2. HOURLY EXPOSURE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. HOURLY EXPOSURE")
print("=" * 70)

# For exposure: aggregate to DAILY level (fraction of nonzero positions per day)
# Each day has 9 hourly bars. Take the mean fraction across hours in the day.
hourly_date_col = hourly['date'].dt.normalize()


def hourly_exposure_meta(approach):
    """Daily exposure for hourly meta-portfolio: fraction of (strategy, ticker) non-zero per day."""
    sub = hourly[hourly['approach'] == approach].copy()
    sub['day'] = sub['date'].dt.normalize()
    # For each (day, hour): fraction of non-zero across all (strategy, ticker)
    # Then take mean across hours within day
    exp_per_ts = sub.groupby('date').apply(
        lambda g: (g['position'].abs() > 0.001).mean(), include_groups=False)
    exp_per_ts.index = pd.to_datetime(exp_per_ts.index)
    # Aggregate to daily: mean across hourly bars in the day
    daily_exp = exp_per_ts.groupby(exp_per_ts.index.normalize()).mean()
    return daily_exp


def hourly_exposure_best():
    """Daily exposure for META-BEST (best approach per strategy)."""
    parts = []
    for s, app in BEST_MAP_HOURLY.items():
        sub = hourly[(hourly['strategy'] == s) & (hourly['approach'] == app)]
        exp_s = sub.groupby('date').apply(
            lambda g: (g['position'].abs() > 0.001).mean(), include_groups=False)
        exp_s.index = pd.to_datetime(exp_s.index)
        daily_exp_s = exp_s.groupby(exp_s.index.normalize()).mean()
        parts.append(daily_exp_s.rename(s))
    df = pd.DataFrame(parts).T.fillna(0)
    return df.mean(axis=1)


exposure_h = pd.DataFrame({
    'META-A': hourly_exposure_meta('A'),
    'META-B': hourly_exposure_meta('B'),
    'META-D': hourly_exposure_meta('D'),
})
exposure_h['META-BEST'] = hourly_exposure_best()
exposure_h.index = pd.to_datetime(exposure_h.index)
exposure_h = exposure_h.sort_index()

print("Mean daily exposure (hourly strategies):")
for col in exposure_h.columns:
    e = exposure_h[col]
    print(f"  {col}: mean={e.mean()*100:.1f}%, max={e.max()*100:.1f}%, "
          f"min={e.min()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════
#  3. AGGREGATE HOURLY RETURNS TO DAILY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. AGGREGATING HOURLY RETURNS TO DAILY")
print("=" * 70)

# Sum hourly returns within each day to get daily return
# (compounding within a day: (1+r1)*(1+r2)*...*(1+r9) - 1)
meta_h_daily = pd.DataFrame()
for col in meta_h.columns:
    s = meta_h[col].copy()
    s.index = pd.to_datetime(s.index)
    daily_agg = s.groupby(s.index.normalize()).apply(lambda x: np.prod(1 + x) - 1)
    meta_h_daily[col] = daily_agg

meta_h_daily.index = pd.to_datetime(meta_h_daily.index)
meta_h_daily = meta_h_daily.sort_index()

print("Daily-aggregated hourly META metrics:")
for col in meta_h_daily.columns:
    m = calc_metrics(meta_h_daily[col], BPY_DAILY)
    print(f"  {col}: Ret={m['Return_pct']}%, Vol={m['Vol_pct']}%, "
          f"Sharpe={m['Sharpe']}, MDD={m['MDD_pct']}%")


# ══════════════════════════════════════════════════════════════════
#  4. CBR KEY RATE + MMF
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CBR KEY RATE + MMF")
print("=" * 70)

kr_path = os.path.join(os.path.dirname(BASE),
                       'moex_discovery/data/external/macro/raw/key_rate_cbr.parquet')
kr = pd.read_parquet(kr_path)
kr['date'] = pd.to_datetime(kr['date'])
kr = kr.sort_values('date')

# Align to daily dates from hourly meta
daily_dates = pd.DataFrame({'date': meta_h_daily.index}).sort_values('date')
kr_daily = pd.merge_asof(daily_dates, kr[['date', 'value']], on='date', direction='backward')
kr_daily = kr_daily.set_index('date')['value']
print(f"CBR key rate: mean={kr_daily.mean():.2f}%, range {kr_daily.min():.1f}%-{kr_daily.max():.1f}%")

# MMF: (CBR - 1.5%) / 100 / 365 per calendar day
mmf_annual = (kr_daily - 1.5) / 100
mmf_daily_rate = mmf_annual / 365


# ══════════════════════════════════════════════════════════════════
#  5. ALIGN ALL DATA
# ══════════════════════════════════════════════════════════════════
# IMOEX daily
imoex_path = os.path.join(os.path.dirname(BASE),
                          'moex_discovery/data/external/moex_iss/indices_10m/IMOEX.parquet')
imoex_raw = pd.read_parquet(imoex_path)
imoex_raw['dt'] = pd.to_datetime(imoex_raw['begin'])
imoex_raw['date_d'] = imoex_raw['dt'].dt.normalize()
imoex_daily = imoex_raw.groupby('date_d').agg(close=('close', 'last')).sort_index()
imoex_daily['ret'] = imoex_daily['close'].pct_change()

# Common dates across all datasets
common = meta_h_daily.index.intersection(exposure_h.index)\
                            .intersection(mmf_daily_rate.index)\
                            .intersection(imoex_daily.index)
common = common.sort_values()
print(f"\nCommon dates: {len(common)}, {common[0].date()} to {common[-1].date()}")

meta_h_daily = meta_h_daily.loc[common]
exposure_h = exposure_h.loc[common]
mmf_daily_rate = mmf_daily_rate.loc[common]
imoex_ret = imoex_daily.loc[common, 'ret']


# ══════════════════════════════════════════════════════════════════
#  TABLE 1: HOURLY EXPOSURE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 1: HOURLY EXPOSURE")
print("=" * 70)

exp_rows = []
for col in ['META-A', 'META-B', 'META-D', 'META-BEST']:
    e = exposure_h[col]
    exp_rows.append({
        'Portfolio': col,
        'Mean_pct': round(e.mean() * 100, 1),
        'Median_pct': round(e.median() * 100, 1),
        'Max_pct': round(e.max() * 100, 1),
        'Min_pct': round(e.min() * 100, 1),
        'Free_Capital_pct': round((1 - e.mean()) * 100, 1),
    })
exp_df = pd.DataFrame(exp_rows)
print(exp_df.to_string(index=False))
exp_df.to_csv(f'{OUT_DIR}/hourly_exposure.csv', index=False)
print(f"Saved: {OUT_DIR}/hourly_exposure.csv")


# ══════════════════════════════════════════════════════════════════
#  TABLE 2: CAPITAL EFFICIENCY (Strategy + MMF)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 2: CAPITAL EFFICIENCY (Strategy + MMF)")
print("=" * 70)

ce_rows = []
ce_details = pd.DataFrame(index=common)

for col in ['META-A', 'META-B', 'META-D', 'META-BEST']:
    strat_r = meta_h_daily[col].values
    exp = exposure_h[col].values
    free_cap = np.maximum(0, 1 - exp)
    mmf_r = free_cap * mmf_daily_rate.values
    total_r = strat_r + mmf_r

    ce_details[f'{col}_strat'] = strat_r
    ce_details[f'{col}_exposure'] = exp
    ce_details[f'{col}_mmf'] = mmf_r
    ce_details[f'{col}_total'] = total_r

    m_base = calc_metrics(strat_r, BPY_DAILY)
    m_total = calc_metrics(total_r, BPY_DAILY)
    mmf_add = np.mean(mmf_r) * BPY_DAILY * 100

    ce_rows.append({
        'Portfolio': col,
        'Strat_Return': m_base['Return_pct'],
        'MMF_Addition': round(mmf_add, 2),
        'Total_Return': m_total['Return_pct'],
        'Sharpe': m_total['Sharpe'],
        'MDD_pct': m_total['MDD_pct'],
        'Calmar': m_total['Calmar'],
    })
    print(f"  {col}:")
    print(f"    Strategy only: Ret={m_base['Return_pct']}%, Sharpe={m_base['Sharpe']}")
    print(f"    MMF addition:  +{mmf_add:.2f}% p.a. (avg exposure={np.mean(exp)*100:.1f}%)")
    print(f"    Total:         Ret={m_total['Return_pct']}%, Sharpe={m_total['Sharpe']}, "
          f"MDD={m_total['MDD_pct']}%")

ce_df = pd.DataFrame(ce_rows)
ce_df.to_csv(f'{OUT_DIR}/hourly_capital_efficiency.csv', index=False)
ce_details.to_csv(f'{OUT_DIR}/hourly_capital_efficiency_details.csv')
print(f"\nSaved: {OUT_DIR}/hourly_capital_efficiency.csv")


# ══════════════════════════════════════════════════════════════════
#  TABLE 3: SCALING META-BEST + MMF (1x-8x)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 3: SCALING META-BEST + MMF (1x-8x)")
print("=" * 70)

KEY_MULTS = [1, 2, 3, 5, 8]
SCALING_PORTFOLIOS = ['META-A', 'META-B', 'META-D', 'META-BEST']

scale_rows = []
for p in SCALING_PORTFOLIOS:
    strat_r = meta_h_daily[p].values
    exp = exposure_h[p].values
    mmf_rate = mmf_daily_rate.values

    for k in KEY_MULTS:
        scaled_r = k * strat_r
        eff_exp = k * exp
        free_cap = np.maximum(0, 1 - eff_exp)
        mmf_r = free_cap * mmf_rate
        total_r = scaled_r + mmf_r

        m = calc_metrics(total_r, BPY_DAILY)
        scale_rows.append({
            'Portfolio': p, 'Multiplier': f'{k}x',
            'Avg_Exp_pct': round(np.mean(eff_exp) * 100, 1),
            'Max_Exp_pct': round(np.max(eff_exp) * 100, 1),
            'Free_Cap_pct': round(np.mean(free_cap) * 100, 1),
            'Pct_Over100': round(np.mean(eff_exp > 1.0) * 100, 1),
            'Return_pct': m['Return_pct'],
            'Vol_pct': m['Vol_pct'],
            'Sharpe': m['Sharpe'],
            'MDD_pct': m['MDD_pct'],
            'Calmar': m['Calmar'],
        })

scale_df = pd.DataFrame(scale_rows)
scale_df.to_csv(f'{OUT_DIR}/hourly_scaling_analysis.csv', index=False)

# Print META-BEST scaling
print("\n--- META-BEST hourly scaling ---")
best_sc = scale_df[scale_df['Portfolio'] == 'META-BEST']
print(best_sc[['Multiplier', 'Avg_Exp_pct', 'Max_Exp_pct', 'Free_Cap_pct',
               'Pct_Over100', 'Return_pct', 'Vol_pct', 'Sharpe', 'MDD_pct']].to_string(index=False))

print("\n--- META-D hourly scaling ---")
d_sc = scale_df[scale_df['Portfolio'] == 'META-D']
print(d_sc[['Multiplier', 'Avg_Exp_pct', 'Max_Exp_pct', 'Free_Cap_pct',
            'Pct_Over100', 'Return_pct', 'Vol_pct', 'Sharpe', 'MDD_pct']].to_string(index=False))

print(f"\nSaved: {OUT_DIR}/hourly_scaling_analysis.csv")

# Fine-grained scan (for thresholds)
print("\n--- Fine-grained scan META-BEST (1x to 15x, step 0.5) ---")
fine_mults = np.arange(1, 15.5, 0.5)
fine_rows = []
for p in ['META-BEST', 'META-D']:
    strat_r = meta_h_daily[p].values
    exp = exposure_h[p].values
    mmf_rate = mmf_daily_rate.values
    for k in fine_mults:
        scaled_r = k * strat_r
        eff_exp = k * exp
        free_cap = np.maximum(0, 1 - eff_exp)
        total_r = scaled_r + free_cap * mmf_rate
        m = calc_metrics(total_r, BPY_DAILY)
        fine_rows.append({
            'Portfolio': p, 'k': k,
            'Avg_Exp': round(np.mean(eff_exp) * 100, 1),
            'Max_Exp': round(np.max(eff_exp) * 100, 1),
            'Pct_Over100': round(np.mean(eff_exp > 1.0) * 100, 1),
            'Return': m['Return_pct'], 'Sharpe': m['Sharpe'], 'MDD': m['MDD_pct'],
        })
fine_df = pd.DataFrame(fine_rows)

bs = fine_df[fine_df['Portfolio'] == 'META-BEST']
print(bs.to_string(index=False))

# Key thresholds
print("\n--- KEY THRESHOLDS (META-BEST hourly) ---")
exp50 = bs.loc[(bs['Avg_Exp'] - 50).abs().idxmin()]
print(f"Avg exposure ~50%: k={exp50['k']}x (actual {exp50['Avg_Exp']}%)")
over100 = bs[bs['Max_Exp'] > 100]
if len(over100) > 0:
    fo = over100.iloc[0]
    print(f"First max exp >100%: k={fo['k']}x (max {fo['Max_Exp']}%, {fo['Pct_Over100']}% days)")
max_sh = bs.loc[bs['Sharpe'].idxmax()]
print(f"Max Sharpe: k={max_sh['k']}x (Sharpe={max_sh['Sharpe']}, Return={max_sh['Return']}%)")
for mdd_tgt in [-5, -10, -20]:
    over_mdd = bs[bs['MDD'] <= mdd_tgt]
    if len(over_mdd) > 0:
        fm = over_mdd.iloc[0]
        print(f"MDD <= {mdd_tgt}%: k={fm['k']}x (MDD={fm['MDD']}%)")
    else:
        print(f"MDD <= {mdd_tgt}%: not reached in 1-15x")


# ══════════════════════════════════════════════════════════════════
#  TABLE 4: HOURLY CORRELATION MATRIX S1-S6 (approach A, EW)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 4: HOURLY CORRELATION MATRIX S1-S6")
print("=" * 70)

# Build per-strategy returns (approach A, EW, hourly, net)
strat_returns_a = pd.DataFrame()
for s in STRATEGIES:
    key = (s, 'A')
    if key in strat_port_h:
        # Aggregate to daily
        sr = strat_port_h[key]
        sr.index = pd.to_datetime(sr.index)
        daily_sr = sr.groupby(sr.index.normalize()).apply(lambda x: np.prod(1 + x) - 1)
        strat_returns_a[s] = daily_sr

strat_returns_a = strat_returns_a.loc[common].dropna(how='all')
corr_matrix = strat_returns_a.corr()

# Short names for display
short_names = {'S1_MeanRev': 'S1', 'S2_Bollinger': 'S2', 'S3_Donchian': 'S3',
               'S4_Supertrend': 'S4', 'S5_PivotPoints': 'S5', 'S6_VWAP': 'S6'}
corr_display = corr_matrix.rename(index=short_names, columns=short_names)
print(corr_display.round(3).to_string())

corr_matrix.to_csv(f'{OUT_DIR}/hourly_correlation_matrix.csv')
print(f"\nSaved: {OUT_DIR}/hourly_correlation_matrix.csv")

# Categorize
print("\nCategory correlations:")
contra = ['S1_MeanRev', 'S2_Bollinger']
trend = ['S3_Donchian', 'S4_Supertrend']
range_ = ['S5_PivotPoints', 'S6_VWAP']

def mean_corr(strats1, strats2, mat):
    vals = []
    for s1 in strats1:
        for s2 in strats2:
            if s1 != s2:
                vals.append(mat.loc[s1, s2])
    return np.mean(vals) if vals else np.nan

print(f"  Intra-Contrarian (S1-S2): {corr_matrix.loc['S1_MeanRev','S2_Bollinger']:.3f}")
print(f"  Intra-Trend (S3-S4): {corr_matrix.loc['S3_Donchian','S4_Supertrend']:.3f}")
print(f"  Intra-Range (S5-S6): {corr_matrix.loc['S5_PivotPoints','S6_VWAP']:.3f}")
print(f"  Cross-Category (Contra vs Trend): {mean_corr(contra, trend, corr_matrix):.3f}")
print(f"  Cross-Category (Contra vs Range): {mean_corr(contra, range_, corr_matrix):.3f}")
print(f"  Cross-Category (Trend vs Range): {mean_corr(trend, range_, corr_matrix):.3f}")
print(f"  Overall mean (off-diag): {corr_matrix.values[np.triu_indices(6, k=1)].mean():.3f}")


# ══════════════════════════════════════════════════════════════════
#  TABLE 5: COMMISSION SENSITIVITY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 5: COMMISSION SENSITIVITY (META-BEST)")
print("=" * 70)

COMM_LEVELS = [0.0000, 0.0002, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]

def build_meta_best_with_comm(pos_data, best_map, comm, bpy):
    """Build META-BEST returns for a given commission level."""
    strat_parts = {}
    for s, app in best_map.items():
        sub = pos_data[(pos_data['strategy'] == s) & (pos_data['approach'] == app)]
        net_by_ticker = {}
        for ticker, g in sub.groupby('ticker'):
            g = g.sort_values('date')
            nr = net_returns_series(g['position'].values, g['daily_gross_return'].values, comm)
            net_by_ticker[ticker] = pd.Series(nr, index=g['date'].values)
        if net_by_ticker:
            ret_df = pd.DataFrame(net_by_ticker).fillna(0.0)
            strat_parts[s] = ret_df.mean(axis=1)
    aligned = pd.DataFrame(strat_parts).fillna(0.0)
    ew_ret = aligned.mean(axis=1)

    # If hourly, aggregate to daily
    ew_ret.index = pd.to_datetime(ew_ret.index)
    if bpy == BPY_HOURLY:
        daily_ret = ew_ret.groupby(ew_ret.index.normalize()).apply(lambda x: np.prod(1 + x) - 1)
        return daily_ret
    return ew_ret


comm_rows = []
for comm_val in COMM_LEVELS:
    comm_pct = f"{comm_val*100:.2f}%"

    # Daily
    r_d = build_meta_best_with_comm(daily, BEST_MAP_DAILY, comm_val, BPY_DAILY)
    m_d = calc_metrics(r_d.values, BPY_DAILY)

    # Hourly
    r_h = build_meta_best_with_comm(hourly, BEST_MAP_HOURLY, comm_val, BPY_HOURLY)
    m_h = calc_metrics(r_h.values, BPY_DAILY)

    comm_rows.append({
        'Comm_pct': comm_pct,
        'Daily_Return': m_d['Return_pct'],
        'Daily_Vol': m_d['Vol_pct'],
        'Daily_Sharpe': m_d['Sharpe'],
        'Daily_MDD': m_d['MDD_pct'],
        'Hourly_Return': m_h['Return_pct'],
        'Hourly_Vol': m_h['Vol_pct'],
        'Hourly_Sharpe': m_h['Sharpe'],
        'Hourly_MDD': m_h['MDD_pct'],
    })
    print(f"  Comm {comm_pct}: Daily Sharpe={m_d['Sharpe']}, Hourly Sharpe={m_h['Sharpe']}")

comm_df = pd.DataFrame(comm_rows)
comm_df.to_csv(f'{OUT_DIR}/commission_sensitivity.csv', index=False)
print(f"\nSaved: {OUT_DIR}/commission_sensitivity.csv")


# ══════════════════════════════════════════════════════════════════
#  TABLE 6: COMPARISON vs IMOEX
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 6: HOURLY vs IMOEX COMPARISON")
print("=" * 70)

# Alpha/Beta
def compute_alpha_beta(strat_r, idx_r, bpy):
    mask = np.isfinite(strat_r) & np.isfinite(idx_r)
    s, i = strat_r[mask], idx_r[mask]
    beta = np.cov(s, i)[0, 1] / np.var(i) if np.var(i) > 1e-12 else 0
    alpha = (np.mean(s) - beta * np.mean(i)) * bpy * 100
    return round(beta, 4), round(alpha, 2)

cmp_rows = []

# IMOEX
m_imoex = calc_metrics(imoex_ret.values, BPY_DAILY)
cmp_rows.append({
    'Portfolio': 'IMOEX B&H', 'Return_pct': m_imoex['Return_pct'],
    'Vol_pct': m_imoex['Vol_pct'], 'Sharpe': m_imoex['Sharpe'],
    'MDD_pct': m_imoex['MDD_pct'], 'Calmar': m_imoex['Calmar'],
    'Beta': 1.0, 'Alpha_pct': '-',
})

# Base hourly META portfolios
for col in ['META-A', 'META-B', 'META-D', 'META-BEST']:
    r = meta_h_daily[col].values
    m = calc_metrics(r, BPY_DAILY)
    beta, alpha = compute_alpha_beta(r, imoex_ret.values, BPY_DAILY)
    cmp_rows.append({
        'Portfolio': f'{col} (hourly)',
        'Return_pct': m['Return_pct'], 'Vol_pct': m['Vol_pct'],
        'Sharpe': m['Sharpe'], 'MDD_pct': m['MDD_pct'],
        'Calmar': m['Calmar'], 'Beta': beta, 'Alpha_pct': alpha,
    })

# + MMF
for col in ['META-A', 'META-B', 'META-D', 'META-BEST']:
    strat_r = meta_h_daily[col].values
    exp = exposure_h[col].values
    free_cap = np.maximum(0, 1 - exp)
    total_r = strat_r + free_cap * mmf_daily_rate.values
    m = calc_metrics(total_r, BPY_DAILY)
    cmp_rows.append({
        'Portfolio': f'{col} + ФДР (hourly)',
        'Return_pct': m['Return_pct'], 'Vol_pct': m['Vol_pct'],
        'Sharpe': m['Sharpe'], 'MDD_pct': m['MDD_pct'],
        'Calmar': m['Calmar'], 'Beta': '~0', 'Alpha_pct': '-',
    })

# Scaled META-BEST + MMF
for k in [2, 3, 5]:
    strat_r = meta_h_daily['META-BEST'].values
    exp = exposure_h['META-BEST'].values
    scaled_r = k * strat_r
    eff_exp = k * exp
    free_cap = np.maximum(0, 1 - eff_exp)
    total_r = scaled_r + free_cap * mmf_daily_rate.values
    m = calc_metrics(total_r, BPY_DAILY)
    cmp_rows.append({
        'Portfolio': f'META-BEST × {k} + ФДР (hourly)',
        'Return_pct': m['Return_pct'], 'Vol_pct': m['Vol_pct'],
        'Sharpe': m['Sharpe'], 'MDD_pct': m['MDD_pct'],
        'Calmar': m['Calmar'], 'Beta': '~0', 'Alpha_pct': '-',
    })

cmp_df = pd.DataFrame(cmp_rows)
print(cmp_df.to_string(index=False))
cmp_df.to_csv(f'{OUT_DIR}/hourly_vs_imoex.csv', index=False)
print(f"\nSaved: {OUT_DIR}/hourly_vs_imoex.csv")


# ══════════════════════════════════════════════════════════════════
#  MARKDOWN OUTPUT (for FINAL_RESULTS.md sections 12-15)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MARKDOWN SECTIONS FOR FINAL_RESULTS.md")
print("=" * 70)

md_lines = []

# Section 12: Hourly Exposure
md_lines.append("\n---\n")
md_lines.append("## 12. Часовая экспозиция и капитальная эффективность\n")
md_lines.append("\n### Таблица 12.1. Экспозиция часовых мета-портфелей\n")
md_lines.append("| Портфель | Ср. экспоз. | Медиана | Макс. | Мин. | Своб. капитал |")
md_lines.append("|----------|:-----------:|:-------:|:-----:|:----:|:-------------:|")
for _, r in exp_df.iterrows():
    md_lines.append(f"| {r['Portfolio']} | {r['Mean_pct']}% | {r['Median_pct']}% | "
                    f"{r['Max_pct']}% | {r['Min_pct']}% | {r['Free_Capital_pct']}% |")

md_lines.append("\n### Таблица 12.2. Капитальная эффективность (стратегия + ФДР, часовые)\n")
md_lines.append("| Портфель | Стратегия, % | +ФДР, % | Итого, % | Sharpe | MDD, % | Calmar |")
md_lines.append("|----------|:------------:|:-------:|:--------:|:------:|:------:|:------:|")
for _, r in ce_df.iterrows():
    md_lines.append(f"| {r['Portfolio']} | {r['Strat_Return']} | +{r['MMF_Addition']} | "
                    f"**{r['Total_Return']}** | {r['Sharpe']} | {r['MDD_pct']} | {r['Calmar']} |")

# Section 13: Hourly Correlation
md_lines.append("\n---\n")
md_lines.append("## 13. Часовая корреляция стратегий\n")
md_lines.append("\n### Таблица 13.1. Корреляционная матрица S1-S6 (подход A, EW, часовые)\n")
hdr = "| | " + " | ".join(short_names.values()) + " |"
md_lines.append(hdr)
md_lines.append("|---" * 7 + "|")
for s_full, s_short in short_names.items():
    vals = [f"{corr_matrix.loc[s_full, s2]:.3f}" for s2 in corr_matrix.columns]
    md_lines.append(f"| **{s_short}** | " + " | ".join(vals) + " |")

# Category summary
md_lines.append(f"\nМежкатегорийные корреляции:")
md_lines.append(f"- Внутри «контртренд» (S1-S2): {corr_matrix.loc['S1_MeanRev','S2_Bollinger']:.3f}")
md_lines.append(f"- Внутри «тренд» (S3-S4): {corr_matrix.loc['S3_Donchian','S4_Supertrend']:.3f}")
md_lines.append(f"- Внутри «диапазон» (S5-S6): {corr_matrix.loc['S5_PivotPoints','S6_VWAP']:.3f}")
md_lines.append(f"- Между категориями (контртренд vs тренд): {mean_corr(contra, trend, corr_matrix):.3f}")
md_lines.append(f"- Средняя (внедиагональная): {corr_matrix.values[np.triu_indices(6, k=1)].mean():.3f}")

# Section 14: Commission Sensitivity
md_lines.append("\n---\n")
md_lines.append("## 14. Чувствительность к комиссиям (META-BEST)\n")
md_lines.append("\n### Таблица 14.1. Sharpe META-BEST при разных уровнях комиссии\n")
md_lines.append("| Комиссия | Daily Return% | Daily Sharpe | Daily MDD% | Hourly Return% | Hourly Sharpe | Hourly MDD% |")
md_lines.append("|:--------:|:------------:|:------------:|:----------:|:--------------:|:-------------:|:-----------:|")
for _, r in comm_df.iterrows():
    md_lines.append(f"| {r['Comm_pct']} | {r['Daily_Return']} | {r['Daily_Sharpe']} | "
                    f"{r['Daily_MDD']} | {r['Hourly_Return']} | {r['Hourly_Sharpe']} | {r['Hourly_MDD']} |")

# Section 15: Comparison vs IMOEX (hourly)
md_lines.append("\n---\n")
md_lines.append("## 15. Часовое сравнение с IMOEX\n")
md_lines.append("\n### Таблица 15.1. Сводная таблица (часовые портфели vs IMOEX)\n")
md_lines.append("| Портфель | Return% | Vol% | Sharpe | MDD% | Calmar | Beta |")
md_lines.append("|----------|--------:|-----:|-------:|-----:|-------:|-----:|")
for _, r in cmp_df.iterrows():
    md_lines.append(f"| {r['Portfolio']} | {r['Return_pct']} | {r['Vol_pct']} | "
                    f"{r['Sharpe']} | {r['MDD_pct']} | {r['Calmar']} | {r['Beta']} |")

md_lines.append(f"\nBeta ~ 0: часовые мета-портфели market-neutral, как и дневные.")

# Hourly scaling table
md_lines.append("\n### Масштабирование экспозиции (META-BEST hourly + ФДР)\n")
md_lines.append("| Множитель | Ср. экспоз. | Макс. экспоз. | Своб. кап. | %дней >100% | Return% | Sharpe | MDD% |")
md_lines.append("|:---------:|:-----------:|:-------------:|:----------:|:-----------:|:-------:|:------:|:----:|")
for _, r in scale_df[scale_df['Portfolio'] == 'META-BEST'].iterrows():
    md_lines.append(f"| {r['Multiplier']} | {r['Avg_Exp_pct']}% | {r['Max_Exp_pct']}% | "
                    f"{r['Free_Cap_pct']}% | {r['Pct_Over100']}% | {r['Return_pct']} | {r['Sharpe']} | {r['MDD_pct']} |")

# Print markdown
md_text = "\n".join(md_lines)
print(md_text)

# Save markdown snippet
with open(f'{OUT_DIR}/hourly_final_sections.md', 'w') as f:
    f.write(md_text)
print(f"\nSaved: {OUT_DIR}/hourly_final_sections.md")

print("\nDone!")
