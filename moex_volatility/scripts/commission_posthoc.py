#!/usr/bin/env python3
"""
commission_posthoc.py — Post-hoc commission recalculation from saved positions.

Uses daily_positions.parquet (no pipeline rerun).
Real MOEX commission rates per side:
  Daily:  0.40% (min), 0.50% (max)
  Hourly: 0.35% (min), 0.45% (max)
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
POS_PATH = BASE / "results" / "final" / "strategies" / "walkforward_v3" / "daily_positions.parquet"
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v3" / "data"
RESULTS_DIR = BASE / "results"

BCD_YEARS = [2022, 2023, 2024, 2025]

# Commission per side
COMM_DAILY = {"min": 0.0040, "max": 0.0050}    # 0.40%, 0.50%
COMM_HOURLY = {"min": 0.0035, "max": 0.0045}   # 0.35%, 0.45%

STRATEGY_ORDER = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
                  "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACH_ORDER = ["A", "B", "C", "D"]


def compute_metrics(positions, gross_returns, bars_per_year):
    """Compute Sharpe, ann_return%, max_dd% from position/return arrays."""
    n = len(positions)
    if n < 10:
        return {"sharpe": np.nan, "ann_ret_pct": np.nan, "max_dd_pct": np.nan,
                "n_trades": 0, "exposure_pct": 0.0}

    # Trades: count entries (0 → non-zero)
    trades = 0
    in_trade = False
    for i in range(n):
        if positions[i] != 0 and not in_trade:
            trades += 1
            in_trade = True
        if in_trade and positions[i] == 0:
            in_trade = False

    exposure_pct = np.count_nonzero(positions) / n * 100
    years = max(n / bars_per_year, 0.5)
    trades_per_year = trades / years

    ann_factor = np.sqrt(bars_per_year)
    mean_r = np.mean(gross_returns)
    std_r = np.std(gross_returns, ddof=1) if n > 1 else 1e-10
    sharpe = mean_r / std_r * ann_factor if std_r > 1e-12 else 0.0
    ann_ret_pct = mean_r * bars_per_year * 100
    cum = np.cumsum(gross_returns)
    rmax = np.maximum.accumulate(cum)
    dd = cum - rmax
    max_dd_pct = dd.min() * 100 if len(dd) > 0 else 0.0

    return {"sharpe": sharpe, "ann_ret_pct": ann_ret_pct, "max_dd_pct": max_dd_pct,
            "n_trades": trades, "trades_per_year": trades_per_year,
            "exposure_pct": exposure_pct}


def compute_net_returns(positions, gross_returns, commission):
    """Compute net returns: gross - |Δposition| × commission."""
    dpos = np.diff(positions, prepend=0.0)
    comm_cost = np.abs(dpos) * commission
    return gross_returns - comm_cost


def main():
    print("Loading positions...")
    df = pd.read_parquet(POS_PATH)
    print(f"  {len(df):,} rows, {df['strategy'].nunique()} strategies, "
          f"{df['tf'].nunique()} TFs, {df['approach'].nunique()} approaches")

    # Filter to BCD years
    df = df[df["test_year"].isin(BCD_YEARS)].copy()
    print(f"  BCD years {BCD_YEARS}: {len(df):,} rows")

    # Sort for correct diff computation
    df = df.sort_values(["strategy", "tf", "approach", "ticker", "test_year", "date"])

    # Group keys
    group_cols = ["strategy", "tf", "approach", "ticker", "test_year"]

    results = []
    for (strat, tf, appr, ticker, year), grp in df.groupby(group_cols):
        pos = grp["position"].values
        gross_r = grp["daily_gross_return"].values
        n = len(pos)

        # Bars per year estimate
        if tf == "daily":
            bpy = 252
            comm_rates = COMM_DAILY
        else:
            bpy = n  # use actual bar count as bars_per_year for this year
            comm_rates = COMM_HOURLY

        # Gross metrics
        gross_met = compute_metrics(pos, gross_r, bpy)

        row = {
            "strategy": strat, "tf": tf, "approach": appr,
            "ticker": ticker, "year": year,
            "gross_sharpe": gross_met["sharpe"],
            "gross_ann_ret_pct": gross_met["ann_ret_pct"],
            "gross_max_dd_pct": gross_met["max_dd_pct"],
            "trades_per_year": gross_met["trades_per_year"],
            "exposure_pct": gross_met["exposure_pct"],
        }

        # Net metrics for each commission variant
        for label, rate in comm_rates.items():
            net_r = compute_net_returns(pos, gross_r, rate)
            net_met = compute_metrics(pos, net_r, bpy)
            pct = rate * 100
            row[f"net_{label}_sharpe"] = net_met["sharpe"]
            row[f"net_{label}_ann_ret_pct"] = net_met["ann_ret_pct"]
            row[f"net_{label}_max_dd_pct"] = net_met["max_dd_pct"]

        results.append(row)

    res_df = pd.DataFrame(results)
    print(f"  Computed {len(res_df)} group metrics")

    # ── Verify gross Sharpe matches existing results ──
    print("\n  Verifying gross Sharpe vs existing results...")
    existing = pd.read_csv(OUT_DIR / "wf_v3_all_results.csv")
    existing_bcd = existing[existing["year"].isin(BCD_YEARS)]

    for tf in ["daily", "hourly"]:
        for strat in STRATEGY_ORDER:
            for appr in APPROACH_ORDER:
                # Our gross
                our = res_df[(res_df["strategy"] == strat) &
                             (res_df["tf"] == tf) &
                             (res_df["approach"] == appr)]
                our_sh = our.groupby("year")["gross_sharpe"].mean().mean()

                # Existing
                ex = existing_bcd[(existing_bcd["strategy"] == strat) &
                                  (existing_bcd["timeframe"] == tf) &
                                  (existing_bcd["approach"] == appr)]
                ex_sh = ex.groupby("year")["sharpe"].mean().mean() if len(ex) > 0 else np.nan

                if not np.isnan(our_sh) and not np.isnan(ex_sh):
                    diff = abs(our_sh - ex_sh)
                    if diff > 0.05:
                        print(f"    MISMATCH {strat} {tf} {appr}: "
                              f"posthoc={our_sh:.4f} vs csv={ex_sh:.4f} (Δ={diff:.4f})")

    # ── Aggregate: mean across tickers per year, then mean across years ──
    agg_cols_gross = ["gross_sharpe", "gross_ann_ret_pct", "gross_max_dd_pct",
                      "trades_per_year"]
    agg_cols_net = []
    for tf_key in ["daily", "hourly"]:
        comm = COMM_DAILY if tf_key == "daily" else COMM_HOURLY
        for label in comm:
            agg_cols_net.extend([
                f"net_{label}_sharpe", f"net_{label}_ann_ret_pct", f"net_{label}_max_dd_pct"
            ])

    # Step 1: mean across tickers per (strategy, tf, approach, year)
    year_agg = res_df.groupby(["strategy", "tf", "approach", "year"]).agg(
        {c: "mean" for c in agg_cols_gross + [c for c in res_df.columns
                                                if c.startswith("net_")]}
    ).reset_index()

    # Step 2: mean across years per (strategy, tf, approach)
    final = year_agg.groupby(["strategy", "tf", "approach"]).agg(
        {c: "mean" for c in agg_cols_gross + [c for c in year_agg.columns
                                                if c.startswith("net_")]}
    ).reset_index()

    # ══════════════════════════════════════════════════════════════
    # TABLE 1: DAILY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 100)
    print("ТАБЛИЦА 1: ДНЕВКИ (commission per side: 0.40% min, 0.50% max)")
    print("═" * 100)

    daily = final[final["tf"] == "daily"].copy()
    daily["strategy"] = pd.Categorical(daily["strategy"], categories=STRATEGY_ORDER, ordered=True)
    daily["approach"] = pd.Categorical(daily["approach"], categories=APPROACH_ORDER, ordered=True)
    daily = daily.sort_values(["strategy", "approach"])

    header = (f"  {'Strategy':<18s} {'App':>3s} │ {'Gross':>7s} {'Net0.4%':>7s} {'Net0.5%':>7s} │"
              f" {'Tr/yr':>5s} │ {'GrossR%':>7s} {'Net0.4R%':>8s} {'Net0.5R%':>8s} │"
              f" {'GrosDD%':>7s} {'N0.4DD%':>7s} {'N0.5DD%':>7s}")
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    prev_strat = ""
    for _, r in daily.iterrows():
        s = r["strategy"] if r["strategy"] != prev_strat else ""
        prev_strat = r["strategy"]
        print(f"  {s:<18s} {r['approach']:>3s} │ "
              f"{r['gross_sharpe']:7.3f} {r['net_min_sharpe']:7.3f} {r['net_max_sharpe']:7.3f} │"
              f" {r['trades_per_year']:5.1f} │"
              f" {r['gross_ann_ret_pct']:7.2f} {r['net_min_ann_ret_pct']:8.2f} {r['net_max_ann_ret_pct']:8.2f} │"
              f" {r['gross_max_dd_pct']:7.2f} {r['net_min_max_dd_pct']:7.2f} {r['net_max_max_dd_pct']:7.2f}")

    # ══════════════════════════════════════════════════════════════
    # TABLE 2: HOURLY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 105)
    print("ТАБЛИЦА 2: ЧАСОВИКИ (commission per side: 0.35% min, 0.45% max)")
    print("═" * 105)

    hourly = final[final["tf"] == "hourly"].copy()
    hourly["strategy"] = pd.Categorical(hourly["strategy"], categories=STRATEGY_ORDER, ordered=True)
    hourly["approach"] = pd.Categorical(hourly["approach"], categories=APPROACH_ORDER, ordered=True)
    hourly = hourly.sort_values(["strategy", "approach"])

    header2 = (f"  {'Strategy':<18s} {'App':>3s} │ {'Gross':>7s} {'Net.35%':>7s} {'Net.45%':>7s} │"
               f" {'Tr/yr':>5s} │ {'GrossR%':>7s} {'N.35R%':>8s} {'N.45R%':>8s} │"
               f" {'GrosDD%':>7s} {'N.35DD%':>7s} {'N.45DD%':>7s}")
    sep2 = "  " + "─" * (len(header2) - 2)
    print(header2)
    print(sep2)

    prev_strat = ""
    for _, r in hourly.iterrows():
        s = r["strategy"] if r["strategy"] != prev_strat else ""
        prev_strat = r["strategy"]
        print(f"  {s:<18s} {r['approach']:>3s} │ "
              f"{r['gross_sharpe']:7.3f} {r['net_min_sharpe']:7.3f} {r['net_max_sharpe']:7.3f} │"
              f" {r['trades_per_year']:5.1f} │"
              f" {r['gross_ann_ret_pct']:7.2f} {r['net_min_ann_ret_pct']:8.2f} {r['net_max_ann_ret_pct']:8.2f} │"
              f" {r['gross_max_dd_pct']:7.2f} {r['net_min_max_dd_pct']:7.2f} {r['net_max_max_dd_pct']:7.2f}")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("ИТОГО")
    print("═" * 80)

    for tf_label, tf_key in [("DAILY", "daily"), ("HOURLY", "hourly")]:
        sub = final[final["tf"] == tf_key]
        comm = COMM_DAILY if tf_key == "daily" else COMM_HOURLY

        for appr in APPROACH_ORDER:
            a = sub[sub["approach"] == appr]
            g = a["gross_sharpe"].mean()
            n_min = a["net_min_sharpe"].mean()
            n_max = a["net_max_sharpe"].mean()
            pos_g = (a["gross_sharpe"] > 0).sum()
            pos_min = (a["net_min_sharpe"] > 0).sum()
            pos_max = (a["net_max_sharpe"] > 0).sum()
            total = len(a)

            min_pct = comm["min"] * 100
            max_pct = comm["max"] * 100
            print(f"  {tf_label} {appr}: Gross={g:.3f}  "
                  f"Net({min_pct:.2f}%)={n_min:.3f}  "
                  f"Net({max_pct:.2f}%)={n_max:.3f}  "
                  f"Positive: {pos_g}/{total} → {pos_min}/{total} → {pos_max}/{total}")

    # Strategies that flip sign
    print("\n  ИНВЕРСИЯ ЗНАКА (gross>0 → net<0):")
    for tf_key, comm in [("daily", COMM_DAILY), ("hourly", COMM_HOURLY)]:
        sub = final[final["tf"] == tf_key]
        for _, r in sub.iterrows():
            if r["gross_sharpe"] > 0:
                for label, rate in comm.items():
                    net_sh = r[f"net_{label}_sharpe"]
                    if net_sh < 0:
                        print(f"    {r['strategy']} {tf_key} {r['approach']}: "
                              f"Gross={r['gross_sharpe']:.3f} → Net({rate*100:.2f}%)={net_sh:.3f}")

    # ── Save CSVs ──
    # Detailed per ticker-year
    res_df.to_csv(OUT_DIR / "wf_v3_commission_posthoc_detail.csv", index=False)
    # Summary
    final.to_csv(OUT_DIR / "wf_v3_commission_posthoc_summary.csv", index=False)
    print(f"\n  Saved: wf_v3_commission_posthoc_detail.csv ({len(res_df)} rows)")
    print(f"  Saved: wf_v3_commission_posthoc_summary.csv ({len(final)} rows)")


if __name__ == "__main__":
    main()
