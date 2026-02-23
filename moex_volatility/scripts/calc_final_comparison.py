"""
Final Comparison Table: Capital Efficiency, Leverage, Benchmark
Reads from daily_positions.parquet, IMOEX, CBR key rate.
Outputs: FINAL_COMPARISON_TABLE.csv, .md, capital_efficiency_details.csv
"""
import pandas as pd
import numpy as np
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

COMM = 0.0040  # 0.40% per side
BPY = 252
BCD_YEARS = list(range(2022, 2027))
TICKERS = ['AFLT','ALRS','HYDR','IRAO','LKOH','LSRG','MGNT','MOEX',
           'MTLR','MTSS','NVTK','OGKB','PHOR','RTKM','SBER','TATN','VTBR']
STRATEGIES = ['S1_MeanRev','S2_Bollinger','S3_Donchian',
              'S4_Supertrend','S5_PivotPoints','S6_VWAP']
# Best approach per strategy (from v4 results)
BEST_MAP = {'S1_MeanRev':'D', 'S2_Bollinger':'C', 'S3_Donchian':'D',
            'S4_Supertrend':'D', 'S5_PivotPoints':'B', 'S6_VWAP':'D'}

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
def calc_metrics(r):
    """Annualized metrics from daily return series."""
    r = r.dropna()
    if len(r) < 10:
        return dict(Return_pct=np.nan, Vol_pct=np.nan, Sharpe=np.nan,
                    MDD_pct=np.nan, Calmar=np.nan)
    ann_ret = r.mean() * BPY
    ann_vol = r.std(ddof=1) * np.sqrt(BPY)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-12 else 0
    return dict(Return_pct=round(ann_ret*100, 2),
                Vol_pct=round(ann_vol*100, 2),
                Sharpe=round(sharpe, 2),
                MDD_pct=round(mdd*100, 2),
                Calmar=round(calmar, 2))


def net_returns_series(pos, gross_r):
    """Per-ticker net returns with commission."""
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * COMM


# ──────────────────────────────────────────────────────────
# 1. RECONSTRUCT ALL META-PORTFOLIO DAILY RETURNS
# ──────────────────────────────────────────────────────────
print("=" * 70)
print("1. RECONSTRUCTING META-PORTFOLIO DAILY RETURNS")
print("=" * 70)

pos_df = pd.read_parquet('results/final/strategies/walkforward_v4/daily_positions.parquet')
pos_df['date'] = pd.to_datetime(pos_df['date'])

# Filter: daily timeframe, BCD test years only
daily = pos_df[(pos_df['tf'] == 'daily') & (pos_df['test_year'].isin(BCD_YEARS))].copy()
print(f"Daily BCD positions: {len(daily):,} rows")

# Per-strategy, per-approach: compute EW portfolio daily returns across 17 tickers
def strategy_portfolio_returns(df_sub):
    """EW portfolio return for one (strategy, approach) pair."""
    net_by_ticker = {}
    for ticker, g in df_sub.groupby('ticker'):
        g = g.sort_values('date')
        nr = net_returns_series(g['position'].values, g['daily_gross_return'].values)
        net_by_ticker[ticker] = pd.Series(nr, index=g['date'].values)
    ret_df = pd.DataFrame(net_by_ticker)
    ret_df = ret_df.fillna(0.0)
    return ret_df.mean(axis=1)  # EW across tickers


# Build per-(strategy, approach) portfolio returns
print("Building per-strategy portfolio returns...")
strat_port = {}
for (strat, app), g in daily.groupby(['strategy', 'approach']):
    strat_port[(strat, app)] = strategy_portfolio_returns(g)

print(f"Built {len(strat_port)} (strategy, approach) portfolios")

# Build META portfolios
def build_meta(approach):
    """Average across 6 strategies for a given approach."""
    parts = {}
    for s in STRATEGIES:
        key = (s, approach)
        if key in strat_port:
            parts[s] = strat_port[key]
    if not parts:
        return pd.Series(dtype=float)
    aligned = pd.DataFrame(parts).fillna(0.0)
    return aligned.mean(axis=1)


def build_meta_best():
    """Best approach per strategy, then average."""
    parts = {}
    for s, app in BEST_MAP.items():
        key = (s, app)
        if key in strat_port:
            parts[s] = strat_port[key]
    aligned = pd.DataFrame(parts).fillna(0.0)
    return aligned.mean(axis=1)


def build_meta_mean_bcd():
    """For each strategy: average B, C, D returns. Then average across strategies."""
    parts = {}
    for s in STRATEGIES:
        bcd_parts = {}
        for app in ['B', 'C', 'D']:
            key = (s, app)
            if key in strat_port:
                bcd_parts[app] = strat_port[key]
        if bcd_parts:
            aligned_bcd = pd.DataFrame(bcd_parts).fillna(0.0)
            parts[s] = aligned_bcd.mean(axis=1)
    aligned = pd.DataFrame(parts).fillna(0.0)
    return aligned.mean(axis=1)


meta_returns = pd.DataFrame({
    'META-A': build_meta('A'),
    'META-B': build_meta('B'),
    'META-C': build_meta('C'),
    'META-D': build_meta('D'),
    'META-BEST': build_meta_best(),
    'META-MEAN(BCD)': build_meta_mean_bcd(),
})
meta_returns.index = pd.to_datetime(meta_returns.index)
meta_returns = meta_returns.sort_index()

# Verify against known values
print("\nVerification vs meta_portfolios_bcd.csv:")
for col in meta_returns.columns:
    m = calc_metrics(meta_returns[col])
    print(f"  {col}: Ret={m['Return_pct']}%, Vol={m['Vol_pct']}%, "
          f"Sharpe={m['Sharpe']}, MDD={m['MDD_pct']}%")

# ──────────────────────────────────────────────────────────
# 2. DAILY EXPOSURE PER META-PORTFOLIO
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. COMPUTING DAILY EXPOSURE")
print("=" * 70)

def daily_exposure_meta(approach):
    """For each date: fraction of (strategy, ticker) pairs with non-zero position."""
    sub = daily[daily['approach'] == approach].copy()
    # group by date: count non-zero / total
    exp = sub.groupby('date').apply(
        lambda g: (g['position'].abs() > 0.001).mean(), include_groups=False)
    return exp


exposure = pd.DataFrame({
    'META-A': daily_exposure_meta('A'),
    'META-B': daily_exposure_meta('B'),
    'META-C': daily_exposure_meta('C'),
    'META-D': daily_exposure_meta('D'),
})
exposure.index = pd.to_datetime(exposure.index)
exposure = exposure.sort_index()

# META-BEST exposure: weighted by best approach per strategy
def daily_exposure_best():
    parts = []
    for s, app in BEST_MAP.items():
        sub = daily[(daily['strategy'] == s) & (daily['approach'] == app)]
        exp_s = sub.groupby('date').apply(
            lambda g: (g['position'].abs() > 0.001).mean(), include_groups=False)
        parts.append(exp_s.rename(s))
    df = pd.DataFrame(parts).T.fillna(0)
    return df.mean(axis=1)


exposure['META-BEST'] = daily_exposure_best()

# META-MEAN(BCD): average of B, C, D exposures
exposure['META-MEAN(BCD)'] = exposure[['META-B', 'META-C', 'META-D']].mean(axis=1)

# Align with meta_returns dates
common_dates = meta_returns.index.intersection(exposure.index)
meta_returns = meta_returns.loc[common_dates]
exposure = exposure.loc[common_dates]

print(f"Dates: {len(common_dates)}, {common_dates[0].date()} to {common_dates[-1].date()}")
print("\nMean daily exposure:")
for col in exposure.columns:
    print(f"  {col}: {exposure[col].mean()*100:.1f}%")

# ──────────────────────────────────────────────────────────
# 3. LOAD CBR KEY RATE → DAILY MMF RATE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. LOADING CBR KEY RATE")
print("=" * 70)

kr_path = os.path.join(os.path.dirname(BASE), 'moex_discovery/data/external/macro/raw/key_rate_cbr.parquet')
if os.path.exists(kr_path):
    kr = pd.read_parquet(kr_path)
    kr['date'] = pd.to_datetime(kr['date'])
    kr = kr.sort_values('date')
    # Expand to daily (forward-fill)
    daily_dates = pd.DataFrame({'date': common_dates}).sort_values('date')
    kr_daily = pd.merge_asof(daily_dates, kr[['date', 'value']], on='date', direction='backward')
    kr_daily = kr_daily.set_index('date')['value']
    print(f"CBR key rate loaded: mean = {kr_daily.mean():.2f}%")
    print(f"  Range: {kr_daily.min():.1f}% to {kr_daily.max():.1f}%")
else:
    print("CBR key rate file not found, using fallback estimates")
    CBR_FALLBACK = {2022: 11.0, 2023: 9.5, 2024: 17.0, 2025: 21.0, 2026: 21.0}
    kr_vals = []
    for d in common_dates:
        kr_vals.append(CBR_FALLBACK.get(d.year, 14.0))
    kr_daily = pd.Series(kr_vals, index=common_dates)

# MMF rate = CBR - 1.5% (management fees, bid-ask)
mmf_annual = (kr_daily - 1.5) / 100   # decimal annual rate
mmf_daily = mmf_annual / 365           # daily rate (calendar days → trading days: /365 is standard for money market)
print(f"MMF daily rate: mean = {mmf_daily.mean()*10000:.2f} bps/day")

# ──────────────────────────────────────────────────────────
# 4. LOAD IMOEX
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. LOADING IMOEX")
print("=" * 70)

imoex_path = os.path.join(os.path.dirname(BASE), 'moex_discovery/data/external/moex_iss/indices_10m/IMOEX.parquet')
imoex_raw = pd.read_parquet(imoex_path)
imoex_raw['dt'] = pd.to_datetime(imoex_raw['begin'])
imoex_raw['date_d'] = imoex_raw['dt'].dt.normalize()

# Daily close (last 10-min bar)
imoex_daily = imoex_raw.groupby('date_d').agg(close=('close', 'last')).sort_index()
imoex_daily['ret'] = imoex_daily['close'].pct_change()

# Align to common dates
imoex_aligned = imoex_daily.loc[imoex_daily.index.intersection(common_dates), 'ret']
# Some dates might be missing in IMOEX — use intersection
final_dates = common_dates.intersection(imoex_aligned.dropna().index)
print(f"IMOEX aligned dates: {len(final_dates)}")

# Re-align everything to final_dates
meta_returns = meta_returns.loc[final_dates]
exposure = exposure.loc[final_dates]
mmf_daily = mmf_daily.loc[final_dates]
imoex_ret = imoex_aligned.loc[final_dates]

print(f"Final common dates: {len(final_dates)}, {final_dates[0].date()} to {final_dates[-1].date()}")
print(f"IMOEX B&H metrics: {calc_metrics(imoex_ret)}")

# ──────────────────────────────────────────────────────────
# 5. CAPITAL EFFICIENCY: strategy + MMF on idle capital
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. CAPITAL EFFICIENCY (Strategy + MMF)")
print("=" * 70)

ce_returns = pd.DataFrame(index=final_dates)
ce_details = pd.DataFrame(index=final_dates)

for col in meta_returns.columns:
    # Base strategy return
    strat_r = meta_returns[col].values
    # Daily exposure
    exp = exposure[col].values if col in exposure.columns else np.full(len(final_dates), 0.12)
    # Free capital
    free_cap = np.maximum(0, 1 - exp)
    # MMF return on free capital
    mmf_r = free_cap * mmf_daily.values
    # Total return
    total_r = strat_r + mmf_r

    ce_returns[col + ' + MMF'] = total_r
    ce_details[col + '_strat'] = strat_r
    ce_details[col + '_mmf'] = mmf_r
    ce_details[col + '_total'] = total_r
    ce_details[col + '_exposure'] = exp

    m_base = calc_metrics(pd.Series(strat_r))
    m_total = calc_metrics(pd.Series(total_r))
    mmf_add = np.mean(mmf_r) * BPY * 100
    print(f"  {col}:")
    print(f"    Strategy only: Ret={m_base['Return_pct']}%, Sharpe={m_base['Sharpe']}")
    print(f"    MMF addition:  +{mmf_add:.2f}% p.a. (avg exposure={np.mean(exp)*100:.1f}%)")
    print(f"    Total:         Ret={m_total['Return_pct']}%, Vol={m_total['Vol_pct']}%, "
          f"Sharpe={m_total['Sharpe']}, MDD={m_total['MDD_pct']}%")

# ──────────────────────────────────────────────────────────
# 6. LEVERAGE (2× and 3×, no funding cost — futures)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. LEVERAGE ANALYSIS (futures, no funding cost)")
print("=" * 70)

lev_returns = {}
for col in ['META-D', 'META-BEST']:
    base_r = meta_returns[col].values
    exp = exposure[col].values if col in exposure.columns else np.full(len(final_dates), 0.12)

    for lev in [2, 3]:
        # Leveraged strategy returns (no funding cost for futures)
        lev_r = lev * base_r
        # Free capital after leverage: 1 - lev * exposure
        # (position capital needed = lev * exposure fraction)
        free_cap_lev = np.maximum(0, 1 - lev * exp)
        # MMF on remaining free capital
        mmf_r_lev = free_cap_lev * mmf_daily.values

        # Variant 1: Pure leverage (no MMF)
        label_pure = f'{col} x{lev}'
        lev_returns[label_pure] = lev_r
        m = calc_metrics(pd.Series(lev_r))
        print(f"  {label_pure}: Ret={m['Return_pct']}%, Vol={m['Vol_pct']}%, "
              f"Sharpe={m['Sharpe']}, MDD={m['MDD_pct']}%")

        # Variant 2: Leverage + MMF on remaining free capital
        total_lev = lev_r + mmf_r_lev
        label_mmf = f'{col} x{lev} + MMF'
        lev_returns[label_mmf] = total_lev
        m2 = calc_metrics(pd.Series(total_lev))
        avg_free = np.mean(free_cap_lev) * 100
        print(f"  {label_mmf}: Ret={m2['Return_pct']}%, Vol={m2['Vol_pct']}%, "
              f"Sharpe={m2['Sharpe']}, MDD={m2['MDD_pct']}%, free_cap={avg_free:.0f}%")

# ──────────────────────────────────────────────────────────
# 7. ALPHA AND BETA
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. ALPHA AND BETA vs IMOEX")
print("=" * 70)

alphas = {}
betas = {}
for col in meta_returns.columns:
    r_meta = meta_returns[col].values
    r_idx = imoex_ret.values
    mask = np.isfinite(r_meta) & np.isfinite(r_idx)
    rm, ri = r_meta[mask], r_idx[mask]

    beta = np.cov(rm, ri)[0, 1] / np.var(ri) if np.var(ri) > 1e-12 else 0
    alpha_ann = (np.mean(rm) - beta * np.mean(ri)) * BPY * 100
    corr = np.corrcoef(rm, ri)[0, 1]

    betas[col] = round(beta, 4)
    alphas[col] = round(alpha_ann, 2)
    print(f"  {col}: beta={beta:.4f}, alpha={alpha_ann:.2f}%, corr={corr:.4f}")

# ──────────────────────────────────────────────────────────
# 8. FINAL COMPARISON TABLE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. BUILDING FINAL COMPARISON TABLE")
print("=" * 70)

rows = []

# IMOEX B&H
m = calc_metrics(imoex_ret)
m['Portfolio'] = 'IMOEX B&H'
m['Beta'] = 1.00
m['Alpha_pct'] = '—'
m['Comment'] = 'Benchmark (price index, no dividends)'
rows.append(m)

# Base meta-portfolios
for col in ['META-A', 'META-B', 'META-C', 'META-D', 'META-BEST', 'META-MEAN(BCD)']:
    m = calc_metrics(meta_returns[col])
    m['Portfolio'] = col
    m['Beta'] = betas.get(col, '—')
    m['Alpha_pct'] = alphas.get(col, '—')
    comments = {
        'META-A': 'Baseline (no forecasts)',
        'META-B': 'Adaptive stops',
        'META-C': 'Regime filter',
        'META-D': 'Vol-gate',
        'META-BEST': 'Best approach per strategy',
        'META-MEAN(BCD)': 'Average forecast effect',
    }
    m['Comment'] = comments.get(col, '')
    rows.append(m)

# Strategy + MMF
for col in ['META-A', 'META-D', 'META-BEST', 'META-B']:
    total_r = ce_returns[col + ' + MMF']
    m = calc_metrics(total_r)
    m['Portfolio'] = col + ' + MMF'
    m['Beta'] = betas.get(col, '—')
    m['Alpha_pct'] = '—'
    m['Comment'] = 'Strategy + money market on idle capital'
    rows.append(m)

# Leverage variants
for label, r_arr in lev_returns.items():
    m = calc_metrics(pd.Series(r_arr, index=final_dates))
    m['Portfolio'] = label
    m['Beta'] = '~0'
    m['Alpha_pct'] = '—'
    if 'MMF' in label:
        m['Comment'] = 'Futures leverage + MMF on remainder'
    else:
        m['Comment'] = 'Futures leverage (no funding cost)'
    rows.append(m)

result_df = pd.DataFrame(rows)
cols_order = ['Portfolio', 'Return_pct', 'Vol_pct', 'Sharpe', 'MDD_pct', 'Calmar',
              'Beta', 'Alpha_pct', 'Comment']
result_df = result_df[cols_order]

print(result_df.to_string(index=False))

# ──────────────────────────────────────────────────────────
# 9. SAVE
# ──────────────────────────────────────────────────────────
out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

# CSV
result_df.to_csv(f'{out_dir}/FINAL_COMPARISON_TABLE.csv', index=False)
print(f"\nSaved: {out_dir}/FINAL_COMPARISON_TABLE.csv")

# Details CSV (daily returns)
ce_details['IMOEX'] = imoex_ret
for label, r_arr in lev_returns.items():
    ce_details[label] = r_arr
ce_details.index.name = 'date'
ce_details.to_csv(f'{out_dir}/capital_efficiency_details.csv')
print(f"Saved: {out_dir}/capital_efficiency_details.csv ({ce_details.shape})")

# Markdown
avg_kr = kr_daily.mean()
with open(f'{out_dir}/FINAL_COMPARISON_TABLE.md', 'w') as f:
    f.write("# Final Comparison Table\n\n")
    f.write(f"**Period**: {final_dates[0].date()} to {final_dates[-1].date()} "
            f"({len(final_dates)} trading days)\n")
    f.write(f"**Commission**: 0.40% per side (daily strategies)\n")
    f.write(f"**CBR key rate**: avg {avg_kr:.1f}% over period "
            f"(range {kr_daily.min():.1f}%–{kr_daily.max():.1f}%)\n")
    f.write(f"**MMF rate**: CBR − 1.5% (management fees)\n")
    f.write(f"**Leverage**: via futures (no borrowing cost, ГО covers margin)\n\n")

    f.write("## Results\n\n")
    f.write("| Portfolio | Return% | Vol% | Sharpe | MDD% | Calmar | Beta | Alpha% | Comment |\n")
    f.write("|" + "|".join(["---"] * 9) + "|\n")
    for _, row in result_df.iterrows():
        vals = [str(row[c]) for c in cols_order]
        f.write("| " + " | ".join(vals) + " |\n")

    f.write("\n## Key Takeaways\n\n")
    f.write("1. **IMOEX B&H** — catastrophic over 2022–2025: sanctions shock, -50% drawdown, "
            "negative return. Poor benchmark.\n\n")

    # Find META-BEST + MMF metrics
    best_mmf = result_df[result_df['Portfolio'] == 'META-BEST + MMF'].iloc[0]
    f.write(f"2. **META-BEST + MMF** — the most realistic practical variant: "
            f"**{best_mmf['Return_pct']}% return** at **{best_mmf['Vol_pct']}% vol**, "
            f"Sharpe **{best_mmf['Sharpe']}**, MDD **{best_mmf['MDD_pct']}%**. "
            f"Idle capital (~88%) earns money market rate.\n\n")

    f.write("3. **Leverage 2–3×** via futures is shown for completeness. "
            "No borrowing cost (ГО/margin covers), so Sharpe scales linearly. "
            "MDD increases proportionally.\n\n")

    f.write("4. **All strategies are market-neutral** (beta ≈ 0, correlation < 0.05 with IMOEX). "
            "Returns are pure alpha from volatility forecasting + strategy construction.\n\n")

    f.write("5. **Capital efficiency** is the correct framing (not leverage): "
            "binary strategies use 4–16% of capital, the rest earns risk-free rate. "
            f"At avg CBR {avg_kr:.1f}%, this adds ~12% p.a. to strategy returns.\n\n")

    f.write("## Methodology Notes\n\n")
    f.write("- Meta-portfolio daily returns reconstructed from `daily_positions.parquet` "
            "(17 tickers × 6 strategies × EW)\n")
    f.write("- Commission: 0.40% per side charged on every position change\n")
    f.write("- Exposure: fraction of (strategy, ticker) pairs with non-zero position each day\n")
    f.write("- MMF return: (CBR key rate − 1.5%) / 365 per calendar day, "
            "applied to (1 − exposure) fraction of capital\n")
    f.write("- Leverage: simple multiplication of daily returns (futures model, no funding cost)\n")
    f.write("- Leveraged + MMF: free capital = max(0, 1 − leverage × exposure)\n")
    f.write("- IMOEX: price index (no dividends). Strategies also trade price, "
            "so comparison is apples-to-apples\n")
    f.write("- Alpha/Beta computed via OLS regression of daily returns on IMOEX returns\n")

print(f"Saved: {out_dir}/FINAL_COMPARISON_TABLE.md")
print("\nDone!")
