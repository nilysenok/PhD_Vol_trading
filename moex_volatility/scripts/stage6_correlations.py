#!/usr/bin/env python3
"""
stage6_correlations.py — 5 types of return correlations.

1. Within-strategy:   avg pairwise corr of daily returns between 17 tickers, per S1-S6
2. Between-strategy:  per ticker, corr of returns across S1-S6
3. Daily vs hourly:   per strategy×ticker, corr of daily vs hourly-aggregated returns
4. Crisis vs calm:    avg pairwise corr in 2022 vs 2024
5. A vs BCD:          avg pairwise correlations under A vs B/C/D

Saves to: correlations/corr_{within_strategy,between_strategy,timeframe,crisis_vs_normal,A_vs_BCD}.csv
"""
import os, sys, warnings, datetime
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from itertools import combinations

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
STRAT_DIR = BASE / "results/final/strategies"
CORR_DIR = STRAT_DIR / "correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = STRAT_DIR / "MASTER_PROGRESS_LOG.txt"

WF_V4_POS = STRAT_DIR / "walkforward_v4/daily_positions.parquet"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")

BPY = 252
TICKERS = ["AFLT","ALRS","HYDR","IRAO","LKOH","LSRG","MGNT","MOEX",
           "MTLR","MTSS","NVTK","OGKB","PHOR","RTKM","SBER","TATN","VTBR"]
STRATEGIES = ["S1_MeanRev","S2_Bollinger","S3_Donchian",
              "S4_Supertrend","S5_PivotPoints","S6_VWAP"]
APPROACHES = ["A","B","C","D"]

COMM_DAILY = 0.0005
COMM_HOURLY = 0.0004


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage6_corr] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════
# Immutable excess-Sharpe formula
# ══════════════════════════════════════════════════════════════════════
def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm):
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    n = len(pm)
    if n < 20:
        return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exposure = np.abs(pm)
    fdr = (1.0 - exposure) * rf * cd
    daily_total = net + fdr
    excess = daily_total - rf * cd
    s = np.std(daily_total, ddof=1)
    if s < 1e-12:
        return 0.0, {}
    sharpe = np.mean(excess) * np.sqrt(252) / s
    return sharpe, {"exposure": round(np.mean(exposure) * 100, 1)}


# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
def load_positions():
    """Load position data. Returns DataFrame or creates dummy."""
    if WF_V4_POS.exists():
        log(f"Loading from {WF_V4_POS}")
        df = pd.read_parquet(WF_V4_POS)
        df["date"] = pd.to_datetime(df["date"])
        return df, False
    log("WARNING: No position file found — creating DUMMY data")
    return _create_dummy(), True


def _create_dummy():
    dates = pd.bdate_range("2020-01-02", "2025-12-31")
    rows = []
    for tf in ["daily", "hourly"]:
        for app in APPROACHES:
            for strat in STRATEGIES:
                for tick in TICKERS:
                    n = len(dates)
                    pos = np.random.choice([-1.0, 0.0, 1.0], size=n, p=[0.15, 0.60, 0.25])
                    lr = np.random.normal(0.0003, 0.02, size=n)
                    rows.append(pd.DataFrame({
                        "date": dates, "strategy": strat, "tf": tf,
                        "ticker": tick, "approach": app,
                        "test_year": [d.year for d in dates],
                        "position": pos, "daily_gross_return": lr,
                    }))
    return pd.concat(rows, ignore_index=True)


def compute_net_returns(sub_df, comm):
    """Compute net returns for a sorted sub-DataFrame."""
    pos = sub_df["position"].values
    lr = sub_df["daily_gross_return"].values
    dp = np.diff(pos, prepend=0.0)
    return pos * lr - np.abs(dp) * comm


def pivot_returns(pos_df, tf, approach, comm):
    """Build a (date × ticker) matrix of net returns for a given strategy/tf/approach.
    Returns dict: strategy -> DataFrame(date × ticker).
    """
    sub = pos_df[(pos_df["tf"] == tf) & (pos_df["approach"] == approach)]
    result = {}
    for strat in STRATEGIES:
        st_data = sub[sub["strategy"] == strat]
        frames = {}
        for tick in TICKERS:
            tk_data = st_data[st_data["ticker"] == tick].sort_values("date")
            if tk_data.empty:
                continue
            net = compute_net_returns(tk_data, comm)
            frames[tick] = pd.Series(net, index=tk_data["date"].values, name=tick)
        if frames:
            result[strat] = pd.DataFrame(frames).sort_index()
    return result


def avg_pairwise_corr(df):
    """Average pairwise correlation among columns of df."""
    if df.shape[1] < 2:
        return np.nan
    corr = df.corr()
    vals = []
    for i, j in combinations(range(len(corr)), 2):
        v = corr.iloc[i, j]
        if not np.isnan(v):
            vals.append(v)
    return np.mean(vals) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════
# Correlation 1: Within-strategy
# ══════════════════════════════════════════════════════════════════════
def corr_within_strategy(pos_df, tf="daily", approach="A"):
    """Avg pairwise correlation of daily returns between 17 tickers, per strategy."""
    log("  Computing within-strategy correlations")
    comm = COMM_DAILY if tf == "daily" else COMM_HOURLY
    strat_rets = pivot_returns(pos_df, tf, approach, comm)

    rows = []
    for strat, df_ret in strat_rets.items():
        rho = avg_pairwise_corr(df_ret)
        rows.append({"strategy": strat, "avg_pairwise_corr": round(rho, 4),
                      "n_tickers": df_ret.shape[1], "n_days": df_ret.shape[0]})
        log(f"    {strat}: rho={rho:.4f} ({df_ret.shape[1]} tickers, {df_ret.shape[0]} days)")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Correlation 2: Between-strategy
# ══════════════════════════════════════════════════════════════════════
def corr_between_strategy(pos_df, tf="daily", approach="A"):
    """For each ticker, correlation of returns across S1-S6."""
    log("  Computing between-strategy correlations")
    comm = COMM_DAILY if tf == "daily" else COMM_HOURLY
    sub = pos_df[(pos_df["tf"] == tf) & (pos_df["approach"] == approach)]

    rows = []
    for tick in TICKERS:
        tk_sub = sub[sub["ticker"] == tick]
        frames = {}
        for strat in STRATEGIES:
            st_data = tk_sub[tk_sub["strategy"] == strat].sort_values("date")
            if st_data.empty:
                continue
            net = compute_net_returns(st_data, comm)
            frames[strat] = pd.Series(net, index=st_data["date"].values, name=strat)
        if len(frames) >= 2:
            df_strats = pd.DataFrame(frames).sort_index().dropna()
            rho = avg_pairwise_corr(df_strats)
            rows.append({"ticker": tick, "avg_between_strat_corr": round(rho, 4),
                          "n_strategies": len(frames), "n_days": len(df_strats)})

    result = pd.DataFrame(rows)
    if not result.empty:
        log(f"    Mean between-strategy corr: {result['avg_between_strat_corr'].mean():.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════
# Correlation 3: Daily vs Hourly
# ══════════════════════════════════════════════════════════════════════
def corr_timeframe(pos_df, approach="A"):
    """Per strategy×ticker, correlation of daily vs hourly-aggregated returns."""
    log("  Computing daily vs hourly correlations")
    rows = []
    for strat in STRATEGIES:
        for tick in TICKERS:
            # Daily returns
            d_sub = pos_df[(pos_df["tf"] == "daily") & (pos_df["approach"] == approach) &
                           (pos_df["strategy"] == strat) & (pos_df["ticker"] == tick)].sort_values("date")
            # Hourly returns (aggregate to daily)
            h_sub = pos_df[(pos_df["tf"] == "hourly") & (pos_df["approach"] == approach) &
                           (pos_df["strategy"] == strat) & (pos_df["ticker"] == tick)].copy()

            if d_sub.empty or h_sub.empty:
                continue

            d_net = compute_net_returns(d_sub, COMM_DAILY)
            d_series = pd.Series(d_net, index=pd.to_datetime(d_sub["date"].values))

            h_sub = h_sub.sort_values("date")
            h_net = compute_net_returns(h_sub, COMM_HOURLY)
            h_series = pd.Series(h_net, index=pd.to_datetime(h_sub["date"].values))
            # Aggregate hourly to daily
            h_daily = h_series.groupby(h_series.index.date).sum()
            h_daily.index = pd.to_datetime(h_daily.index)

            # Align
            common = d_series.index.intersection(h_daily.index)
            if len(common) < 30:
                continue

            rho = np.corrcoef(d_series.loc[common].values, h_daily.loc[common].values)[0, 1]
            rows.append({"strategy": strat, "ticker": tick, "corr_daily_hourly": round(rho, 4),
                          "n_days": len(common)})

    result = pd.DataFrame(rows)
    if not result.empty:
        log(f"    Mean daily-hourly corr: {result['corr_daily_hourly'].mean():.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════
# Correlation 4: Crisis vs Calm
# ══════════════════════════════════════════════════════════════════════
def corr_crisis_vs_normal(pos_df, tf="daily", approach="A"):
    """Avg pairwise correlation in 2022 (crisis) vs 2024 (calm)."""
    log("  Computing crisis vs calm correlations")
    comm = COMM_DAILY if tf == "daily" else COMM_HOURLY
    strat_rets = pivot_returns(pos_df, tf, approach, comm)

    rows = []
    for strat, df_ret in strat_rets.items():
        for yr, label in [(2022, "crisis_2022"), (2024, "calm_2024")]:
            mask = df_ret.index.year == yr if hasattr(df_ret.index, "year") else pd.to_datetime(df_ret.index).year == yr
            sub = df_ret[mask]
            if sub.shape[1] < 2 or len(sub) < 20:
                rho = np.nan
            else:
                rho = avg_pairwise_corr(sub)
            rows.append({"strategy": strat, "period": label, "avg_pairwise_corr": round(rho, 4) if not np.isnan(rho) else np.nan,
                          "n_days": len(sub)})

    result = pd.DataFrame(rows)
    if not result.empty:
        crisis = result[result["period"] == "crisis_2022"]["avg_pairwise_corr"].mean()
        calm = result[result["period"] == "calm_2024"]["avg_pairwise_corr"].mean()
        log(f"    Crisis(2022) avg corr: {crisis:.4f}, Calm(2024) avg corr: {calm:.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════
# Correlation 5: A vs BCD
# ══════════════════════════════════════════════════════════════════════
def corr_A_vs_BCD(pos_df, tf="daily"):
    """Compare average pairwise correlations under A vs B/C/D."""
    log("  Computing A vs BCD correlations")
    comm = COMM_DAILY if tf == "daily" else COMM_HOURLY
    rows = []

    for app in APPROACHES:
        strat_rets = pivot_returns(pos_df, tf, app, comm)
        for strat, df_ret in strat_rets.items():
            rho = avg_pairwise_corr(df_ret)
            rows.append({"approach": app, "strategy": strat,
                          "avg_pairwise_corr": round(rho, 4) if not np.isnan(rho) else np.nan,
                          "n_tickers": df_ret.shape[1], "n_days": df_ret.shape[0]})

    result = pd.DataFrame(rows)
    if not result.empty:
        for app in APPROACHES:
            sub = result[result["approach"] == app]
            mean_corr = sub["avg_pairwise_corr"].mean()
            log(f"    Approach {app}: mean pairwise corr = {mean_corr:.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("STAGE 6: Correlation Analysis (5 types)")
    log("=" * 80)

    result = load_positions()
    if isinstance(result, tuple):
        pos_df, is_dummy = result
    else:
        pos_df, is_dummy = result, False

    if is_dummy:
        log("*** RUNNING WITH DUMMY DATA — results are placeholders ***")

    # 1. Within-strategy
    df1 = corr_within_strategy(pos_df, tf="daily", approach="A")
    df1.to_csv(CORR_DIR / "corr_within_strategy.csv", index=False)
    log(f"  Saved corr_within_strategy.csv ({len(df1)} rows)")

    # 2. Between-strategy
    df2 = corr_between_strategy(pos_df, tf="daily", approach="A")
    df2.to_csv(CORR_DIR / "corr_between_strategy.csv", index=False)
    log(f"  Saved corr_between_strategy.csv ({len(df2)} rows)")

    # 3. Daily vs Hourly
    df3 = corr_timeframe(pos_df, approach="A")
    df3.to_csv(CORR_DIR / "corr_timeframe.csv", index=False)
    log(f"  Saved corr_timeframe.csv ({len(df3)} rows)")

    # 4. Crisis vs Calm
    df4 = corr_crisis_vs_normal(pos_df, tf="daily", approach="A")
    df4.to_csv(CORR_DIR / "corr_crisis_vs_normal.csv", index=False)
    log(f"  Saved corr_crisis_vs_normal.csv ({len(df4)} rows)")

    # 5. A vs BCD
    df5 = corr_A_vs_BCD(pos_df, tf="daily")
    df5.to_csv(CORR_DIR / "corr_A_vs_BCD.csv", index=False)
    log(f"  Saved corr_A_vs_BCD.csv ({len(df5)} rows)")

    log("STAGE 6 COMPLETE")


if __name__ == "__main__":
    main()
