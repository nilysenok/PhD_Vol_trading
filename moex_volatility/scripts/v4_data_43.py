#!/usr/bin/env python3
"""
Section 4.3 Data Extraction — 4 tables for dissertation.

1. Portfolio weighting methods: mean Sharpe/Return/MDD by method
2. Correlation matrix S1–S6 (daily returns, approach A, EW)
3. Yearly returns for META portfolios (A, B, C, D, BEST, MEAN(BCD), IMOEX)
4. Per-strategy EW net Sharpe: strategy × approach

Usage: python3 scripts/v4_data_43.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
V4_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
TBL_DIR = V4_DIR / "tables"
OUT_DIR = BASE / "output_4_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
BCD_YEARS = [2022, 2023, 2024, 2025, 2026]
COMM = 0.0005
BPY = 252

BEST_APP = {
    "S1_MeanRev": "C", "S2_Bollinger": "C", "S3_Donchian": "B",
    "S4_Supertrend": "D", "S5_PivotPoints": "C", "S6_VWAP": "D",
}


def net_returns(pos, gross_r, comm):
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * comm


def calc_sharpe(r, bpy=252):
    if len(r) < 2:
        return 0.0
    s = np.std(r, ddof=1)
    return np.mean(r) / s * np.sqrt(bpy) if s > 1e-12 else 0.0


def calc_maxdd(equity):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return dd.min() * 100


# ══════════════════════════════════════════════════════════════════════
#  TABLE 1: Weighting methods — mean metrics
# ══════════════════════════════════════════════════════════════════════
def table_1_weighting_methods():
    print("=" * 60)
    print("TABLE 1: Portfolio Weighting Methods — Mean Metrics")
    print("=" * 60)

    df = pd.read_csv(TBL_DIR / "v4_portfolios_daily.csv")

    methods = ["EW", "InvVol", "MinVar", "MaxSharpe"]
    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        rows.append({
            "Method": m,
            "Mean_Sharpe": sub["net_sharpe"].mean(),
            "Mean_Return%": sub["ann_ret_pct"].mean(),
            "Mean_MDD%": sub["maxdd_pct"].mean(),
            "Wins_Best": 0,  # count later
        })

    # Count how many times each method is best per (strategy, approach)
    for (strat, appr), grp in df.groupby(["strategy", "approach"]):
        best_idx = grp["net_sharpe"].idxmax()
        best_method = grp.loc[best_idx, "method"]
        for r in rows:
            if r["Method"] == best_method:
                r["Wins_Best"] += 1

    print(f"\n{'Method':>12s} | {'Mean Sharpe':>12s} {'Mean Ret%':>10s} {'Mean MDD%':>10s} {'#Best':>6s}")
    print(f"{'-'*12}-+-{'-'*12}-{'-'*10}-{'-'*10}-{'-'*6}")
    for r in rows:
        print(f"{r['Method']:>12s} | {r['Mean_Sharpe']:12.4f} {r['Mean_Return%']:10.2f} {r['Mean_MDD%']:10.2f} {r['Wins_Best']:6d}")

    # Also compute per-approach breakdown
    print(f"\n  Per-approach breakdown (Mean Sharpe):")
    print(f"{'Method':>12s} | {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s}")
    print(f"{'-'*12}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}")
    for m in methods:
        vals = []
        for appr in APPROACHES:
            sub = df[(df["method"] == m) & (df["approach"] == appr)]
            vals.append(sub["net_sharpe"].mean())
        print(f"{m:>12s} | {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_DIR / "table_1_weighting_methods.csv", index=False)
    return rows


# ══════════════════════════════════════════════════════════════════════
#  TABLE 2: Correlation matrix S1–S6 (daily returns, EW, approach A)
# ══════════════════════════════════════════════════════════════════════
def table_2_correlation_matrix():
    print("\n" + "=" * 60)
    print("TABLE 2: Correlation Matrix S1–S6 (EW daily returns, approach A)")
    print("=" * 60)

    # Build from daily_positions.parquet
    pos_df = pd.read_parquet(V4_DIR / "daily_positions.parquet")
    pos_df = pos_df[(pos_df["tf"] == "daily") &
                    (pos_df["test_year"].isin(BCD_YEARS))].copy()

    strat_rets = {}
    for strat in STRATEGIES:
        sub = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == "A")]
        tickers = sorted(sub["ticker"].unique())
        ticker_rets = {}
        for tkr in tickers:
            g = sub[sub["ticker"] == tkr].sort_values("date")
            nr = net_returns(g["position"].values,
                             g["daily_gross_return"].values, COMM)
            ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)
        ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
        strat_rets[strat] = ret_df.mean(axis=1)  # EW

    all_rets = pd.DataFrame(strat_rets).sort_index().fillna(0.0)

    # Short names for display
    short = {
        "S1_MeanRev": "S1", "S2_Bollinger": "S2", "S3_Donchian": "S3",
        "S4_Supertrend": "S4", "S5_PivotPoints": "S5", "S6_VWAP": "S6",
    }
    all_rets.columns = [short[c] for c in all_rets.columns]

    corr = all_rets.corr()
    print(f"\n{corr.round(3).to_string()}")

    corr.to_csv(OUT_DIR / "table_2_correlation_matrix.csv")

    # Also compute for approach D
    print("\n  Also for approach D:")
    strat_rets_d = {}
    for strat in STRATEGIES:
        sub = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == "D")]
        tickers = sorted(sub["ticker"].unique())
        ticker_rets = {}
        for tkr in tickers:
            g = sub[sub["ticker"] == tkr].sort_values("date")
            nr = net_returns(g["position"].values,
                             g["daily_gross_return"].values, COMM)
            ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)
        ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
        strat_rets_d[strat] = ret_df.mean(axis=1)

    all_rets_d = pd.DataFrame(strat_rets_d).sort_index().fillna(0.0)
    all_rets_d.columns = [short[c] for c in all_rets_d.columns]
    corr_d = all_rets_d.corr()
    print(f"\n{corr_d.round(3).to_string()}")
    corr_d.to_csv(OUT_DIR / "table_2_correlation_matrix_D.csv")

    return corr, corr_d, all_rets, all_rets_d


# ══════════════════════════════════════════════════════════════════════
#  TABLE 3: Yearly returns for META portfolios + IMOEX
# ══════════════════════════════════════════════════════════════════════
def table_3_yearly_returns(all_rets_A, all_rets_D):
    print("\n" + "=" * 60)
    print("TABLE 3: Yearly Returns for META Portfolios")
    print("=" * 60)

    # Load positions for all approaches
    pos_df = pd.read_parquet(V4_DIR / "daily_positions.parquet")
    pos_df = pos_df[(pos_df["tf"] == "daily") &
                    (pos_df["test_year"].isin(BCD_YEARS))].copy()

    # Build per-strategy EW returns for ALL approaches
    port_cache = {}
    for strat in STRATEGIES:
        for appr in APPROACHES:
            sub = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == appr)]
            tickers = sorted(sub["ticker"].unique())
            ticker_rets = {}
            for tkr in tickers:
                g = sub[sub["ticker"] == tkr].sort_values("date")
                nr = net_returns(g["position"].values,
                                 g["daily_gross_return"].values, COMM)
                ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)
            ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
            port_cache[(strat, appr)] = ret_df.mean(axis=1)

    # Build META portfolios
    def build_meta(approach_map):
        """approach_map: {strat: appr} or single string"""
        rets = []
        for s in STRATEGIES:
            appr = approach_map if isinstance(approach_map, str) else approach_map[s]
            rets.append(port_cache[(s, appr)])
        df = pd.DataFrame({s: r for s, r in zip(STRATEGIES, rets)}).sort_index().fillna(0)
        return df.mean(axis=1)

    meta_A = build_meta("A")
    meta_B = build_meta("B")
    meta_C = build_meta("C")
    meta_D = build_meta("D")
    meta_BEST = build_meta(BEST_APP)

    # META-MEAN(BCD): for each strategy, average B/C/D returns, then average across strategies
    bcd_means = []
    for s in STRATEGIES:
        bcd_df = pd.DataFrame({
            "B": port_cache[(s, "B")],
            "C": port_cache[(s, "C")],
            "D": port_cache[(s, "D")],
        }).sort_index().fillna(0)
        bcd_means.append(bcd_df.mean(axis=1))
    aligned_bcd = pd.DataFrame({s: r for s, r in zip(STRATEGIES, bcd_means)}).sort_index().fillna(0)
    meta_MEAN_BCD = aligned_bcd.mean(axis=1)

    # Load IMOEX
    imoex_path = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/dataset_final/02_external/candles_10m/IMOEX.parquet")
    imoex_yearly = {}
    if imoex_path.exists():
        imoex = pd.read_parquet(imoex_path)
        # Convert to daily close prices
        if "close" in imoex.columns and "begin" in imoex.columns:
            imoex["date"] = pd.to_datetime(imoex["begin"]).dt.date
            daily_close = imoex.groupby("date")["close"].last()
            daily_close.index = pd.to_datetime(daily_close.index)
            daily_ret = daily_close.pct_change().dropna()
            # Filter to 2022-2025
            daily_ret = daily_ret[(daily_ret.index >= "2022-01-01") &
                                  (daily_ret.index <= "2025-12-31")]
            for year in BCD_YEARS:
                yr_ret = daily_ret[daily_ret.index.year == year]
                if len(yr_ret) > 0:
                    cum = (1 + yr_ret).prod() - 1
                    imoex_yearly[year] = cum * 100
            # Total
            cum_total = (1 + daily_ret).prod() - 1
            imoex_yearly["Total"] = cum_total * 100
            imoex_yearly["Ann"] = ((1 + cum_total) ** (1 / len(BCD_YEARS)) - 1) * 100
            print(f"  IMOEX loaded: {len(daily_ret)} daily returns")
        else:
            print(f"  IMOEX columns: {list(imoex.columns)}")
    else:
        print(f"  IMOEX not found at {imoex_path}")

    # Compute yearly returns for each META
    portfolios = {
        "META-A": meta_A,
        "META-B": meta_B,
        "META-C": meta_C,
        "META-D": meta_D,
        "META-BEST": meta_BEST,
        "META-MEAN(BCD)": meta_MEAN_BCD,
    }

    results = []
    for name, ret_s in portfolios.items():
        ret_s.index = pd.to_datetime(ret_s.index)
        row = {"Portfolio": name}
        for year in BCD_YEARS:
            yr_ret = ret_s[ret_s.index.year == year]
            if len(yr_ret) > 0:
                cum = (1 + yr_ret).prod() - 1
                row[str(year)] = cum * 100
            else:
                row[str(year)] = np.nan
        # Total (compound all years)
        cum_total = (1 + ret_s).prod() - 1
        row["Total"] = cum_total * 100
        ann = ((1 + cum_total) ** (1 / len(BCD_YEARS)) - 1) * 100
        row["Ann"] = ann
        # Sharpe
        row["Sharpe"] = calc_sharpe(ret_s.values, BPY)
        results.append(row)

    # Add IMOEX
    if imoex_yearly:
        imoex_row = {"Portfolio": "IMOEX B&H"}
        for year in BCD_YEARS:
            imoex_row[str(year)] = imoex_yearly.get(year, np.nan)
        imoex_row["Total"] = imoex_yearly.get("Total", np.nan)
        imoex_row["Ann"] = imoex_yearly.get("Ann", np.nan)
        imoex_row["Sharpe"] = -0.12  # from FINAL_COMPARISON_TABLE_V2
        results.append(imoex_row)

    res_df = pd.DataFrame(results)

    # Print
    cols = ["Portfolio"] + [str(y) for y in BCD_YEARS] + ["Total", "Ann", "Sharpe"]
    print(f"\n{'Portfolio':>18s}", end="")
    for year in BCD_YEARS:
        print(f" {year:>8d}", end="")
    print(f" {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s}")
    print("-" * 80)
    for _, r in res_df.iterrows():
        print(f"{r['Portfolio']:>18s}", end="")
        for year in BCD_YEARS:
            v = r.get(str(year), np.nan)
            if pd.notna(v):
                print(f" {v:8.2f}", end="")
            else:
                print(f" {'N/A':>8s}", end="")
        print(f" {r.get('Total', np.nan):8.2f} {r.get('Ann', np.nan):8.2f} {r.get('Sharpe', np.nan):8.4f}")

    res_df.to_csv(OUT_DIR / "table_3_yearly_returns.csv", index=False)

    # Also compute yearly Sharpe per portfolio
    print(f"\n  Yearly Sharpe:")
    print(f"{'Portfolio':>18s}", end="")
    for year in BCD_YEARS:
        print(f" {year:>8d}", end="")
    print()
    print("-" * 60)
    for name, ret_s in portfolios.items():
        ret_s.index = pd.to_datetime(ret_s.index)
        print(f"{name:>18s}", end="")
        for year in BCD_YEARS:
            yr_ret = ret_s[ret_s.index.year == year]
            if len(yr_ret) > 0:
                sh = calc_sharpe(yr_ret.values, BPY)
                print(f" {sh:8.4f}", end="")
            else:
                print(f" {'N/A':>8s}", end="")
        print()

    return res_df


# ══════════════════════════════════════════════════════════════════════
#  TABLE 4: Per-strategy EW net Sharpe: strategy × approach
# ══════════════════════════════════════════════════════════════════════
def table_4_strategy_approach():
    print("\n" + "=" * 60)
    print("TABLE 4: Per-strategy EW Net Sharpe (strategy × approach)")
    print("=" * 60)

    comp = pd.read_csv(TBL_DIR / "v4_A_vs_forecast_comparison.csv")
    ew = comp[comp["table"] == "EW_NetSharpe"].copy()

    # Exclude Mean and Delta rows
    ew = ew[~ew["strategy"].isin(["Mean", "Delta vs A", "Delta% vs A"])]

    print(f"\n{'Strategy':>16s} | {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s} | {'Best':>5s} {'Δ% vs A':>8s}")
    print(f"{'-'*16}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*5}-{'-'*8}")

    for _, r in ew.iterrows():
        a, b, c, d = r["A"], r["B"], r["C"], r["D"]
        best = r["best_app"]
        best_val = r["best_BCD"]
        delta = (best_val / a - 1) * 100
        print(f"{r['strategy']:>16s} | {a:8.4f} {b:8.4f} {c:8.4f} {d:8.4f} | {best:>5s} {delta:+7.1f}%")

    # Mean row
    mean_row = comp[(comp["table"] == "EW_NetSharpe") & (comp["strategy"] == "Mean")]
    if not mean_row.empty:
        r = mean_row.iloc[0]
        a, b, c, d = r["A"], r["B"], r["C"], r["D"]
        best_val = r["best_BCD"]
        delta = (best_val / a - 1) * 100
        print(f"{'Среднее':>16s} | {a:8.4f} {b:8.4f} {c:8.4f} {d:8.4f} | {'D':>5s} {delta:+7.1f}%")

    # Also get AnnRet table
    print(f"\n  Annual Return (%):")
    ann = comp[comp["table"] == "EW_AnnRet"].copy()
    ann = ann[~ann["strategy"].isin(["Mean", "Delta vs A", "Delta% vs A"])]

    print(f"{'Strategy':>16s} | {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s}")
    print(f"{'-'*16}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}")
    for _, r in ann.iterrows():
        print(f"{r['strategy']:>16s} | {r['A']:8.2f} {r['B']:8.2f} {r['C']:8.2f} {r['D']:8.2f}")

    # MaxDD
    print(f"\n  Max Drawdown (%):")
    mdd = comp[comp["table"] == "EW_MaxDD"].copy()
    mdd = mdd[~mdd["strategy"].isin(["Mean", "Delta vs A", "Delta% vs A"])]

    print(f"{'Strategy':>16s} | {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s}")
    print(f"{'-'*16}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}")
    for _, r in mdd.iterrows():
        print(f"{r['strategy']:>16s} | {r['A']:8.2f} {r['B']:8.2f} {r['C']:8.2f} {r['D']:8.2f}")

    # Save combined
    combined = ew[["strategy", "A", "B", "C", "D", "best_app"]].copy()
    combined.to_csv(OUT_DIR / "table_4_strategy_approach.csv", index=False)

    return combined


# ══════════════════════════════════════════════════════════════════════
def main():
    print("Section 4.3 Data Extraction")
    print("=" * 60)

    table_1_weighting_methods()
    corr_A, corr_D, all_rets_A, all_rets_D = table_2_correlation_matrix()
    table_3_yearly_returns(all_rets_A, all_rets_D)
    table_4_strategy_approach()

    print(f"\nAll saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
