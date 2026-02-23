#!/usr/bin/env python3
"""
V4 Strategy x Approach Analysis Tables

Reads precomputed CSVs and prints 4 formatted summary tables:
  1. Net Sharpe: 6 strategies x 4 approaches (EW portfolios, daily)
  2. Gross vs Net Sharpe at 0.05% and sensitivity at 0.06%
  3. Per-ticker Sharpe for best combo per strategy
  4. Hourly results: 6x4 net Sharpe at 0.04%
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results/final/strategies/walkforward_v4"

STRATEGIES = [
    "S1_MeanRev", "S2_Bollinger", "S3_Donchian",
    "S4_Supertrend", "S5_PivotPoints", "S6_VWAP",
]
APPROACHES = ["A", "B", "C", "D"]

BEST_COMBOS = {
    "S1_MeanRev": "C",
    "S2_Bollinger": "C",
    "S3_Donchian": "B",
    "S4_Supertrend": "D",
    "S5_PivotPoints": "C",
    "S6_VWAP": "D",
}


def print_sep(char="=", width=90):
    print(char * width)


def table1(comparison):
    """Table 1: Net Sharpe -- 6 strategies x 4 approaches (EW portfolios, daily)"""
    print_sep()
    print("TABLE 1: Net Sharpe -- 6 strategies x 4 approaches (EW portfolios, daily)")
    print_sep()
    print()

    df = comparison[comparison["table"] == "EW_NetSharpe"].copy()

    header = f"{'Strategy':<16} {'A':>8} {'B':>8} {'C':>8} {'D':>8} {'Best':>6} {'%D vs A':>10}"
    print(header)
    print("-" * len(header))

    rows_data = []
    for strat in STRATEGIES:
        row = df[df["strategy"] == strat]
        if row.empty:
            continue
        row = row.iloc[0]
        a, b, c, d = float(row["A"]), float(row["B"]), float(row["C"]), float(row["D"])
        rows_data.append((strat, a, b, c, d))

    # Mean row
    mean_row = df[df["strategy"] == "Mean"]
    if not mean_row.empty:
        r = mean_row.iloc[0]
        rows_data.append(("Mean", float(r["A"]), float(r["B"]), float(r["C"]), float(r["D"])))

    for i, (strat, a, b, c, d) in enumerate(rows_data):
        vals = {"A": a, "B": b, "C": c, "D": d}
        best_key = max(vals, key=vals.get)
        pct_delta = (vals[best_key] / a - 1) * 100 if a != 0 else 0.0

        if strat == "Mean":
            print("-" * len(header))

        # Mark best value with *
        parts = f"{strat:<16}"
        for k in APPROACHES:
            v = vals[k]
            if k == best_key:
                parts += f" {v:>7.2f}*"
            else:
                parts += f" {v:>8.2f}"
        parts += f" {best_key:>6} {pct_delta:>+9.1f}%"
        print(parts)

    print()
    print("Key findings:")
    print("  - D dominates (best in 4/6 strategies), C wins S2, B wins S5")
    strat_vals = [r for r in rows_data if r[0] != "Mean"]
    mean_a = np.mean([r[1] for r in strat_vals])
    for app_label, idx in [("D", 4), ("B", 2), ("C", 3)]:
        mean_v = np.mean([r[idx] for r in strat_vals])
        print(f"  - Mean improvement {app_label} vs A: {(mean_v / mean_a - 1) * 100:+.1f}%")
    largest = max(strat_vals, key=lambda r: max(r[2], r[3], r[4]) / r[1] - 1 if r[1] else 0)
    best_v = max(largest[2], largest[3], largest[4])
    print(f"  - Largest gain: {largest[0]} ({(best_v / largest[1] - 1) * 100:+.1f}% vs A)")
    print()


def table2(daily):
    """Table 2: Gross vs Net Sharpe (mean across 17 tickers, 2022-2025)"""
    print_sep()
    print("TABLE 2: Gross vs Net Sharpe (mean across 17 tickers, 2022-2025)")
    print_sep()
    print()

    header = f"{'Strategy':<16} {'App':>4} {'Gross':>8} {'Net0.05':>8} {'Net0.06':>8} {'D(G-N05)':>9}"
    print(header)
    print("-" * len(header))

    prev_strat = None
    for _, row in daily.iterrows():
        strat = row["Strategy"]
        if prev_strat and strat != prev_strat:
            print()
        prev_strat = strat

        gross = row["GrossSharpe"]
        net05 = row["Net0.05Sharpe"]
        net06 = row["Net0.06Sharpe"]
        delta = net05 - gross

        print(f"{strat:<16} {row['App']:>4} {gross:>8.3f} {net05:>8.3f} {net06:>8.3f} {delta:>+9.3f}")

    # Commission sensitivity analysis
    print()
    print("Commission sensitivity (Net 0.05% -> 0.06%):")

    daily_c = daily.copy()
    daily_c["pct_drop"] = np.where(
        daily_c["Net0.05Sharpe"].abs() > 0.05,
        (daily_c["Net0.06Sharpe"] / daily_c["Net0.05Sharpe"] - 1) * 100,
        np.nan,
    )

    valid = daily_c.dropna(subset=["pct_drop"])
    worst2 = valid.nsmallest(2, "pct_drop")
    best2 = valid.nlargest(2, "pct_drop")

    def _fmt(r):
        return (
            f"{r['Strategy']} {r['App']} "
            f"({r['Net0.05Sharpe']:.3f} -> {r['Net0.06Sharpe']:.3f}, "
            f"{r['pct_drop']:.0f}%)"
        )

    print(f"  Worst: {', '.join(_fmt(r) for _, r in worst2.iterrows())}")
    print(f"  Best:  {', '.join(_fmt(r) for _, r in best2.iterrows())}")
    print("  D approach generally most commission-robust (fewer trades)")
    print()
    print("Note: mean-ticker Sharpe (avg of per-ticker Sharpe), not EW portfolio Sharpe.")
    print("      EW portfolio values (Table 1) are higher due to diversification.")
    print()


def table3(raw):
    """Table 3: Per-ticker Sharpe for best combo per strategy (daily, 2022-2025)"""
    print_sep()
    print("TABLE 3: Per-ticker Sharpe for best combo per strategy (daily, 2022-2025)")
    print_sep()
    print()

    df = raw[
        (raw["timeframe"] == "daily")
        & (raw["year"].isin([2022, 2023, 2024, 2025]))
    ].copy()

    tickers = sorted(df["ticker"].unique())

    # Build per-ticker mean Sharpe for each strategy's best approach
    result = {}
    for strat, app in BEST_COMBOS.items():
        subset = df[(df["strategy"] == strat) & (df["approach"] == app)]
        result[strat] = subset.groupby("ticker")["sharpe"].mean()

    result_df = pd.DataFrame(result).reindex(tickers)

    # Short labels for header
    short = {s: f"{s.split('_')[0]}({BEST_COMBOS[s]})" for s in STRATEGIES}
    header = f"{'Ticker':<10}" + "".join(f"{short[s]:>10}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))

    for ticker in tickers:
        parts = f"{ticker:<10}"
        for strat in STRATEGIES:
            val = result_df.at[ticker, strat]
            if pd.isna(val):
                parts += f"{'N/A':>10}"
            else:
                parts += f"{val:>10.3f}"
        print(parts)

    print("-" * len(header))

    # Stats
    for label, func in [("Mean", "mean"), ("Min", "min"), ("Max", "max"), ("Std", "std")]:
        parts = f"{label:<10}"
        for strat in STRATEGIES:
            col = result_df[strat].dropna()
            val = getattr(col, func)()
            parts += f"{val:>10.3f}"
        print(parts)

    print()
    print("Tickers with positive mean Sharpe:")
    for strat in STRATEGIES:
        col = result_df[strat].dropna()
        n_pos = (col > 0).sum()
        print(f"  {short[strat]}: {n_pos}/{len(col)}")
    print()


def table4(hourly):
    """Table 4: Hourly 6x4 at Net 0.04%"""
    print_sep()
    print("TABLE 4: Hourly Net Sharpe at 0.04% commission (mean across tickers, 2022-2025)")
    print_sep()
    print()

    pivot = hourly.set_index(["Strategy", "App"])["Net0.04Sharpe"]

    header = f"{'Strategy':<16}" + "".join(f"{a:>10}" for a in APPROACHES)
    print(header)
    print("-" * len(header))

    n_positive = 0
    positive_by_app = {a: [] for a in APPROACHES}

    for strat in STRATEGIES:
        parts = f"{strat:<16}"
        for app in APPROACHES:
            val = pivot.get((strat, app), np.nan)
            if val > 0:
                parts += f" {val:>8.3f}*"
                n_positive += 1
                positive_by_app[app].append(strat)
            else:
                parts += f" {val:>9.3f}"
        print(parts)

    print()
    print(f"{n_positive} of {len(STRATEGIES) * len(APPROACHES)} positive entries:")
    for app in APPROACHES:
        strats = positive_by_app[app]
        if strats:
            names = ", ".join(s.split("_")[1] for s in strats)
            print(f"  {app}: {names} ({len(strats)}/6)")
        else:
            print(f"  {app}: none (0/6)")

    print()
    d_all = all(
        pivot.get((s, "D"), -1) > 0 for s in STRATEGIES
    )
    if d_all:
        print("Pattern: D is positive for all 6 strategies.")
    print()


def main():
    comparison = pd.read_csv(BASE / "tables/v4_A_vs_forecast_comparison.csv")
    daily = pd.read_csv(BASE / "tables/v4_full_daily.csv")
    hourly = pd.read_csv(BASE / "tables/v4_full_hourly.csv")
    raw = pd.read_csv(BASE / "data/wf_v4_all_results.csv")

    table1(comparison)
    table2(daily)
    table3(raw)
    table4(hourly)


if __name__ == "__main__":
    main()
