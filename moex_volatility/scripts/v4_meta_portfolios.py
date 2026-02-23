#!/usr/bin/env python3
"""V4 Meta-portfolios, stat tests, equity curves, and plots."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
BCD_YEARS = [2022, 2023, 2024, 2025]
COMM_LEVELS = [0.0005, 0.0006]
BPY = 252

# Best approach per strategy (from previous analysis)
BEST_APP = {
    "S1_MeanRev": "C", "S2_Bollinger": "C", "S3_Donchian": "B",
    "S4_Supertrend": "D", "S5_PivotPoints": "C", "S6_VWAP": "D",
}

# Hourly parameters
COMM_LEVELS_HOURLY = [0.0004, 0.0005]
BPY_HOURLY = 2268

BEST_APP_HOURLY = {
    "S1_MeanRev": "D", "S2_Bollinger": "D", "S3_Donchian": "B",
    "S4_Supertrend": "B", "S5_PivotPoints": "D", "S6_VWAP": "D",
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


def build_ew_strategy_portfolio(pos_df, strat, appr, comm, ticker_universe=None):
    """Build EW portfolio of 17 tickers for one (strategy, approach).
    Returns Series indexed by date with daily net returns.
    """
    sub = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == appr)].copy()
    tickers = sorted(sub["ticker"].unique())
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)

    # Per-ticker net returns
    ticker_rets = {}
    for tkr in tickers:
        g = sub[sub["ticker"] == tkr].sort_values("date")
        nr = net_returns(g["position"].values, g["daily_gross_return"].values, comm)
        ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)

    ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
    # Ensure consistent ticker universe (fix for missing tickers in approach C)
    if ticker_universe is not None:
        for tkr in ticker_universe:
            if tkr not in ret_df.columns:
                ret_df[tkr] = 0.0
    # EW
    port_ret = ret_df.mean(axis=1)
    return port_ret


def bootstrap_delta_sharpe(r_a, r_b, n_boot=10000, bpy=252, seed=42):
    rng = np.random.RandomState(seed)
    T = len(r_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, T, T)
        sa = calc_sharpe(r_a[idx], bpy)
        sb = calc_sharpe(r_b[idx], bpy)
        deltas[i] = sb - sa
    return deltas


def main():
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pos_df = pd.read_parquet(OUT_DIR / "daily_positions.parquet")
    pos_df = pos_df[(pos_df["tf"] == "daily") & (pos_df["test_year"].isin(BCD_YEARS))].copy()
    print(f"  {len(pos_df):,} daily BCD rows")

    # Get ticker universe from approach A for each strategy
    ticker_universes = {}
    for strat in STRATEGIES:
        sub_a = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == "A")]
        ticker_universes[strat] = sorted(sub_a["ticker"].unique())

    # ── Build all EW strategy portfolios ──
    print("Building EW strategy portfolios...")
    port_cache = {}  # (strat, appr, comm) -> Series
    for comm in COMM_LEVELS:
        for strat in STRATEGIES:
            for appr in APPROACHES:
                port_cache[(strat, appr, comm)] = build_ew_strategy_portfolio(
                    pos_df, strat, appr, comm,
                    ticker_universe=ticker_universes[strat])

    # ════════════════════════════════════════════════════════════
    # 1. META-PORTFOLIOS
    # ════════════════════════════════════════════════════════════
    print("\nBuilding meta-portfolios...")

    meta_results = []
    meta_returns = {}  # name -> Series (at comm=0.004)

    for comm in COMM_LEVELS:
        # META-A
        strat_rets_a = [port_cache[(s, "A", comm)] for s in STRATEGIES]
        aligned_a = pd.DataFrame({s: r for s, r in zip(STRATEGIES, strat_rets_a)}).sort_index().fillna(0)
        meta_a = aligned_a.mean(axis=1)

        # META-D
        strat_rets_d = [port_cache[(s, "D", comm)] for s in STRATEGIES]
        aligned_d = pd.DataFrame({s: r for s, r in zip(STRATEGIES, strat_rets_d)}).sort_index().fillna(0)
        meta_d = aligned_d.mean(axis=1)

        # META-BEST
        strat_rets_best = [port_cache[(s, BEST_APP[s], comm)] for s in STRATEGIES]
        aligned_best = pd.DataFrame({s: r for s, r in zip(STRATEGIES, strat_rets_best)}).sort_index().fillna(0)
        meta_best = aligned_best.mean(axis=1)

        # META-MEAN(BCD)
        bcd_means = []
        for s in STRATEGIES:
            b_ret = port_cache[(s, "B", comm)]
            c_ret = port_cache[(s, "C", comm)]
            d_ret = port_cache[(s, "D", comm)]
            bcd_df = pd.DataFrame({"B": b_ret, "C": c_ret, "D": d_ret}).sort_index().fillna(0)
            bcd_means.append(bcd_df.mean(axis=1))
        aligned_bcd = pd.DataFrame({s: r for s, r in zip(STRATEGIES, bcd_means)}).sort_index().fillna(0)
        meta_mean_bcd = aligned_bcd.mean(axis=1)

        for name, ret_s in [("META-A", meta_a), ("META-D", meta_d),
                             ("META-BEST", meta_best), ("META-MEAN(BCD)", meta_mean_bcd)]:
            r = ret_s.values
            sh = calc_sharpe(r, BPY)
            ann_ret = np.mean(r) * BPY * 100
            ann_vol = np.std(r, ddof=1) * np.sqrt(BPY) * 100
            eq = (1 + ret_s).cumprod()
            mdd = calc_maxdd(eq.values)
            calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-6 else 0.0

            meta_results.append({
                "portfolio": name, "comm": comm,
                "net_sharpe": round(sh, 4),
                "ann_ret_pct": round(ann_ret, 2),
                "ann_vol_pct": round(ann_vol, 2),
                "maxdd_pct": round(mdd, 2),
                "calmar": round(calmar, 2),
            })

            if comm == COMM_LEVELS[0]:
                meta_returns[name] = ret_s

    # Print meta table
    mr = pd.DataFrame(meta_results)
    print(f"\n{'='*85}")
    print(f"  META-PORTFOLIOS (EW across 6 strategies, each EW across 17 tickers)")
    print(f"{'='*85}")
    print(f"{'Portfolio':>16s} | {'Net0.05%':>9s} {'Net0.06%':>9s} | {'Ret%':>7s} {'Vol%':>7s} {'MDD%':>7s} {'Calmar':>7s}")
    print(f"{'-'*16}-+-{'-'*9}-{'-'*9}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")

    for name in ["META-A", "META-D", "META-BEST", "META-MEAN(BCD)"]:
        r1 = mr[(mr["portfolio"] == name) & (mr["comm"] == COMM_LEVELS[0])].iloc[0]
        r2 = mr[(mr["portfolio"] == name) & (mr["comm"] == COMM_LEVELS[1])].iloc[0]
        print(f"{name:>16s} | {r1['net_sharpe']:9.4f} {r2['net_sharpe']:9.4f} | "
              f"{r1['ann_ret_pct']:7.2f} {r1['ann_vol_pct']:7.2f} {r1['maxdd_pct']:7.2f} {r1['calmar']:7.2f}")

    # Delta rows
    print(f"{'-'*16}-+-{'-'*9}-{'-'*9}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")
    a04 = mr[(mr["portfolio"] == "META-A") & (mr["comm"] == COMM_LEVELS[0])].iloc[0]
    a05 = mr[(mr["portfolio"] == "META-A") & (mr["comm"] == COMM_LEVELS[1])].iloc[0]
    for dname, comp_name in [("D vs A", "META-D"), ("BEST vs A", "META-BEST")]:
        c04 = mr[(mr["portfolio"] == comp_name) & (mr["comm"] == COMM_LEVELS[0])].iloc[0]
        c05 = mr[(mr["portfolio"] == comp_name) & (mr["comm"] == COMM_LEVELS[1])].iloc[0]
        ds04 = c04["net_sharpe"] - a04["net_sharpe"]
        ds05 = c05["net_sharpe"] - a05["net_sharpe"]
        dr = c04["ann_ret_pct"] - a04["ann_ret_pct"]
        dv = c04["ann_vol_pct"] - a04["ann_vol_pct"]
        dm = c04["maxdd_pct"] - a04["maxdd_pct"]
        dc = c04["calmar"] - a04["calmar"]
        print(f"{'Delta '+dname:>16s} | {ds04:+9.4f} {ds05:+9.4f} | "
              f"{dr:+7.2f} {dv:+7.2f} {dm:+7.2f} {dc:+7.2f}")

    mr.to_csv(TBL_DIR / "meta_portfolios.csv", index=False)

    # ════════════════════════════════════════════════════════════
    # 2. STATISTICAL TESTS
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print(f"  STATISTICAL TESTS")
    print(f"{'='*75}")

    comm = COMM_LEVELS[0]
    test_rows = []

    # META tests
    for comp_name, comp_label in [("META-D", "META: A vs D"), ("META-BEST", "META: A vs BEST")]:
        r_a = meta_returns["META-A"].values
        r_b = meta_returns[comp_name].values
        # Align
        idx = meta_returns["META-A"].index.intersection(meta_returns[comp_name].index)
        r_a = meta_returns["META-A"].reindex(idx).fillna(0).values
        r_b = meta_returns[comp_name].reindex(idx).fillna(0).values

        d = r_b - r_a
        t_stat, p_ttest = stats.ttest_rel(r_b, r_a)
        dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))) if np.std(d, ddof=1) > 0 else 0
        p_dm = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        delta_sh = calc_sharpe(r_b, BPY) - calc_sharpe(r_a, BPY)
        boot = bootstrap_delta_sharpe(r_a, r_b)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        test_rows.append({
            "test": comp_label, "delta_sharpe": round(delta_sh, 4),
            "t_stat": round(t_stat, 3), "p_ttest": round(p_ttest, 6),
            "dm_stat": round(dm_stat, 3), "p_dm": round(p_dm, 6),
            "boot_ci_lo": round(ci_lo, 4), "boot_ci_hi": round(ci_hi, 4),
        })

    # Per-strategy tests
    for strat in STRATEGIES:
        best_appr = BEST_APP[strat]
        r_a = port_cache[(strat, "A", comm)]
        r_b = port_cache[(strat, best_appr, comm)]
        idx = r_a.index.intersection(r_b.index)
        ra = r_a.reindex(idx).fillna(0).values
        rb = r_b.reindex(idx).fillna(0).values

        d = rb - ra
        t_stat, p_ttest = stats.ttest_rel(rb, ra)
        dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))) if np.std(d, ddof=1) > 0 else 0
        p_dm = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        delta_sh = calc_sharpe(rb, BPY) - calc_sharpe(ra, BPY)
        boot = bootstrap_delta_sharpe(ra, rb)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        label = f"{strat.split('_')[0]}: A vs {best_appr}"
        test_rows.append({
            "test": label, "delta_sharpe": round(delta_sh, 4),
            "t_stat": round(t_stat, 3), "p_ttest": round(p_ttest, 6),
            "dm_stat": round(dm_stat, 3), "p_dm": round(p_dm, 6),
            "boot_ci_lo": round(ci_lo, 4), "boot_ci_hi": round(ci_hi, 4),
        })

    # Print
    print(f"{'Test':>18s} | {'dSharpe':>8s} | {'t-stat':>7s} {'p(t)':>10s} | "
          f"{'DM':>7s} {'p(DM)':>10s} | {'Boot 95% CI':>18s}")
    print(f"{'-'*18}-+-{'-'*8}-+-{'-'*7}-{'-'*10}-+-{'-'*7}-{'-'*10}-+-{'-'*18}")
    for r in test_rows:
        sig = ""
        p = r["p_ttest"]
        if p < 0.001: sig = " ***"
        elif p < 0.01: sig = "  **"
        elif p < 0.05: sig = "   *"

        print(f"{r['test']:>18s} | {r['delta_sharpe']:+8.4f} | {r['t_stat']:7.3f} {r['p_ttest']:10.6f} | "
              f"{r['dm_stat']:7.3f} {r['p_dm']:10.6f} | [{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}]{sig}")

    pd.DataFrame(test_rows).to_csv(TBL_DIR / "stat_tests.csv", index=False)

    # ════════════════════════════════════════════════════════════
    # 3. EQUITY CURVES
    # ════════════════════════════════════════════════════════════
    print("\nSaving equity curves...")
    comm = COMM_LEVELS[0]

    # Meta equity curves
    meta_eq = pd.DataFrame(index=meta_returns["META-A"].index)
    for name in ["META-A", "META-D", "META-BEST"]:
        meta_eq[name.replace("-", "_")] = (1 + meta_returns[name]).cumprod()
    meta_eq.index.name = "date"
    meta_eq = meta_eq.sort_index()
    meta_eq.to_csv(TBL_DIR / "equity_curves_meta.csv")

    # Per-strategy equity curves
    strat_eq = pd.DataFrame()
    pairs = [
        ("S1_MeanRev", "A"), ("S1_MeanRev", "C"),
        ("S2_Bollinger", "A"), ("S2_Bollinger", "C"),
        ("S3_Donchian", "A"), ("S3_Donchian", "B"),
        ("S4_Supertrend", "A"), ("S4_Supertrend", "D"),
        ("S5_PivotPoints", "A"), ("S5_PivotPoints", "C"),
        ("S6_VWAP", "A"), ("S6_VWAP", "D"),
    ]
    for strat, appr in pairs:
        s_short = strat.split("_")[0]
        col = f"{s_short}_{appr}"
        ret_s = port_cache[(strat, appr, comm)]
        eq_s = (1 + ret_s).cumprod()
        strat_eq[col] = eq_s
    strat_eq.index.name = "date"
    strat_eq = strat_eq.sort_index().ffill()
    strat_eq.to_csv(TBL_DIR / "equity_curves_strategies.csv")

    # ════════════════════════════════════════════════════════════
    # 4. PLOTS
    # ════════════════════════════════════════════════════════════
    print("Generating plots...")
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})

    # -- PLOT 1: Meta Equity Curves --
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {"META_A": "#888888", "META_D": "#d62728", "META_BEST": "#2ca02c"}
    labels_map = {}
    for name in ["META-A", "META-D", "META-BEST"]:
        col = name.replace("-", "_")
        sh = calc_sharpe(meta_returns[name].values, BPY)
        labels_map[col] = f"{name} (Sharpe={sh:.2f})"

    dates = pd.to_datetime(meta_eq.index)
    for col in ["META_A", "META_D", "META_BEST"]:
        ax.plot(dates, meta_eq[col].values, label=labels_map[col],
                color=colors[col], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Equity")
    ax.set_title("Meta-Portfolio: Baseline (A) vs Volatility Forecasts (D, BEST)")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meta_equity_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: meta_equity_curves.png")

    # -- PLOT 2: Per-Strategy Sharpe Barplot --
    port_df = pd.read_csv(TBL_DIR / "v4_portfolios_daily.csv")
    ew_df = port_df[port_df["method"] == "EW"].copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(STRATEGIES))
    width = 0.18
    bar_colors = {"A": "#999999", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}

    for i, appr in enumerate(APPROACHES):
        vals = []
        for strat in STRATEGIES:
            row = ew_df[(ew_df["strategy"] == strat) & (ew_df["approach"] == appr)]
            vals.append(row.iloc[0]["net_sharpe"] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width - 1.5 * width, vals, width,
                      label=f"Approach {appr}", color=bar_colors[appr], edgecolor="white")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Net Sharpe (@ 0.05%)")
    ax.set_title("Per-Strategy EW Portfolio: Net Sharpe by Approach")
    ax.set_xticks(x)
    short_names = [s.replace("S1_", "S1\n").replace("S2_", "S2\n").replace("S3_", "S3\n")
                   .replace("S4_", "S4\n").replace("S5_", "S5\n").replace("S6_", "S6\n")
                   for s in STRATEGIES]
    ax.set_xticklabels(short_names)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sharpe_comparison_barplot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: sharpe_comparison_barplot.png")

    # -- PLOT 3: Per-Strategy Equity Curves (2x3 grid) --
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
    strat_pairs = [
        ("S1_MeanRev", "A", "C"), ("S2_Bollinger", "A", "C"), ("S3_Donchian", "A", "B"),
        ("S4_Supertrend", "A", "D"), ("S5_PivotPoints", "A", "C"), ("S6_VWAP", "A", "D"),
    ]
    dates_s = pd.to_datetime(strat_eq.index)

    for idx, (strat, app_a, app_best) in enumerate(strat_pairs):
        ax = axes[idx // 3, idx % 3]
        s_short = strat.split("_")[0]
        col_a = f"{s_short}_A"
        col_b = f"{s_short}_{app_best}"

        sh_a = calc_sharpe(port_cache[(strat, "A", comm)].values, BPY)
        sh_b = calc_sharpe(port_cache[(strat, app_best, comm)].values, BPY)

        ax.plot(dates_s, strat_eq[col_a].values, color="#888888", linewidth=1.5,
                label=f"A ({sh_a:.2f})")
        ax.plot(dates_s, strat_eq[col_b].values, color="#d62728", linewidth=1.5,
                label=f"{app_best} ({sh_b:.2f})")
        ax.set_title(strat, fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    fig.suptitle("Per-Strategy Equity: Baseline A vs Best Forecast", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategy_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: strategy_equity_curves.png")

    # -- PLOT 4: Drawdown Comparison Barplot --
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(STRATEGIES))
    width = 0.3

    dd_a = []
    dd_best = []
    best_labels = []
    for strat in STRATEGIES:
        best_appr = BEST_APP[strat]
        # A drawdown
        eq_a = (1 + port_cache[(strat, "A", comm)]).cumprod()
        dd_a.append(calc_maxdd(eq_a.values))
        # Best drawdown
        eq_b = (1 + port_cache[(strat, best_appr, comm)]).cumprod()
        dd_best.append(calc_maxdd(eq_b.values))
        best_labels.append(best_appr)

    ax.bar(x - width / 2, dd_a, width, label="Approach A", color="#999999", edgecolor="white")
    ax.bar(x + width / 2, dd_best, width, label="Best (B/C/D)", color="#d62728", edgecolor="white")

    # Add best approach labels
    for i, lbl in enumerate(best_labels):
        ax.text(i + width / 2, dd_best[i] - 0.3, lbl, ha="center", va="top", fontsize=10, fontweight="bold")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Max Drawdown: Baseline A vs Best Forecast Approach")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drawdown_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: drawdown_comparison.png")

    # ════════════════════════════════════════════════════════════
    # HOURLY PIPELINE
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  HOURLY PIPELINE")
    print(f"{'='*85}")

    pos_all = pd.read_parquet(OUT_DIR / "daily_positions.parquet")
    pos_hourly = pos_all[(pos_all["tf"] == "hourly") &
                         (pos_all["test_year"].isin(BCD_YEARS))].copy()
    print(f"  Hourly BCD rows: {len(pos_hourly):,}")

    if len(pos_hourly) == 0:
        print("  No hourly data found, skipping.")
        print(f"\nAll done. Files saved to:\n  {TBL_DIR}/\n  {FIG_DIR}/")
        return

    # Get hourly ticker universe from approach A
    hticker_universes = {}
    for strat in STRATEGIES:
        sub_a = pos_hourly[(pos_hourly["strategy"] == strat) & (pos_hourly["approach"] == "A")]
        hticker_universes[strat] = sorted(sub_a["ticker"].unique())

    # ── Build all EW strategy portfolios (hourly) ──
    print("Building hourly EW strategy portfolios...")
    hport_cache = {}  # (strat, appr, comm) -> Series
    for comm_h in COMM_LEVELS_HOURLY:
        for strat in STRATEGIES:
            for appr in APPROACHES:
                hport_cache[(strat, appr, comm_h)] = build_ew_strategy_portfolio(
                    pos_hourly, strat, appr, comm_h,
                    ticker_universe=hticker_universes[strat])

    # ── 1. META-PORTFOLIOS (hourly) ──
    print("\nBuilding hourly meta-portfolios...")
    hmeta_results = []
    hmeta_returns = {}

    for comm_h in COMM_LEVELS_HOURLY:
        # META-A
        strat_rets_a = [hport_cache[(s, "A", comm_h)] for s in STRATEGIES]
        aligned_a = pd.DataFrame({s: r for s, r in zip(STRATEGIES, strat_rets_a)}).sort_index().fillna(0)
        meta_a_h = aligned_a.mean(axis=1)

        # META-D
        strat_rets_d = [hport_cache[(s, "D", comm_h)] for s in STRATEGIES]
        aligned_d = pd.DataFrame({s: r for s, r in zip(STRATEGIES, strat_rets_d)}).sort_index().fillna(0)
        meta_d_h = aligned_d.mean(axis=1)

        # META-BEST
        strat_rets_best = [hport_cache[(s, BEST_APP_HOURLY[s], comm_h)] for s in STRATEGIES]
        aligned_best = pd.DataFrame({s: r for s, r in zip(STRATEGIES, strat_rets_best)}).sort_index().fillna(0)
        meta_best_h = aligned_best.mean(axis=1)

        # META-MEAN(BCD)
        bcd_means = []
        for s in STRATEGIES:
            b_ret = hport_cache[(s, "B", comm_h)]
            c_ret = hport_cache[(s, "C", comm_h)]
            d_ret = hport_cache[(s, "D", comm_h)]
            bcd_df = pd.DataFrame({"B": b_ret, "C": c_ret, "D": d_ret}).sort_index().fillna(0)
            bcd_means.append(bcd_df.mean(axis=1))
        aligned_bcd = pd.DataFrame({s: r for s, r in zip(STRATEGIES, bcd_means)}).sort_index().fillna(0)
        meta_mean_bcd_h = aligned_bcd.mean(axis=1)

        for name, ret_s in [("META-A", meta_a_h), ("META-D", meta_d_h),
                             ("META-BEST", meta_best_h), ("META-MEAN(BCD)", meta_mean_bcd_h)]:
            r = ret_s.values
            sh = calc_sharpe(r, BPY_HOURLY)
            ann_ret = np.mean(r) * BPY_HOURLY * 100
            ann_vol = np.std(r, ddof=1) * np.sqrt(BPY_HOURLY) * 100
            eq = (1 + ret_s).cumprod()
            mdd = calc_maxdd(eq.values)
            calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-6 else 0.0

            hmeta_results.append({
                "portfolio": name, "comm": comm_h,
                "net_sharpe": round(sh, 4),
                "ann_ret_pct": round(ann_ret, 2),
                "ann_vol_pct": round(ann_vol, 2),
                "maxdd_pct": round(mdd, 2),
                "calmar": round(calmar, 2),
            })

            if comm_h == COMM_LEVELS_HOURLY[0]:
                hmeta_returns[name] = ret_s

    # Print hourly meta table
    hmr = pd.DataFrame(hmeta_results)
    c1_lbl = f"Net{COMM_LEVELS_HOURLY[0]*100:.2f}%"
    c2_lbl = f"Net{COMM_LEVELS_HOURLY[1]*100:.2f}%"
    print(f"\n{'='*85}")
    print(f"  HOURLY META-PORTFOLIOS (EW across 6 strategies, each EW across 17 tickers)")
    print(f"{'='*85}")
    print(f"{'Portfolio':>16s} | {c1_lbl:>9s} {c2_lbl:>9s} | {'Ret%':>7s} {'Vol%':>7s} {'MDD%':>7s} {'Calmar':>7s}")
    print(f"{'-'*16}-+-{'-'*9}-{'-'*9}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")

    for name in ["META-A", "META-D", "META-BEST", "META-MEAN(BCD)"]:
        r1 = hmr[(hmr["portfolio"] == name) & (hmr["comm"] == COMM_LEVELS_HOURLY[0])].iloc[0]
        r2 = hmr[(hmr["portfolio"] == name) & (hmr["comm"] == COMM_LEVELS_HOURLY[1])].iloc[0]
        print(f"{name:>16s} | {r1['net_sharpe']:9.4f} {r2['net_sharpe']:9.4f} | "
              f"{r1['ann_ret_pct']:7.2f} {r1['ann_vol_pct']:7.2f} {r1['maxdd_pct']:7.2f} {r1['calmar']:7.2f}")

    # Delta rows
    print(f"{'-'*16}-+-{'-'*9}-{'-'*9}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")
    ha04 = hmr[(hmr["portfolio"] == "META-A") & (hmr["comm"] == COMM_LEVELS_HOURLY[0])].iloc[0]
    ha05 = hmr[(hmr["portfolio"] == "META-A") & (hmr["comm"] == COMM_LEVELS_HOURLY[1])].iloc[0]
    for dname, comp_name in [("D vs A", "META-D"), ("BEST vs A", "META-BEST")]:
        hc04 = hmr[(hmr["portfolio"] == comp_name) & (hmr["comm"] == COMM_LEVELS_HOURLY[0])].iloc[0]
        hc05 = hmr[(hmr["portfolio"] == comp_name) & (hmr["comm"] == COMM_LEVELS_HOURLY[1])].iloc[0]
        ds04 = hc04["net_sharpe"] - ha04["net_sharpe"]
        ds05 = hc05["net_sharpe"] - ha05["net_sharpe"]
        dr = hc04["ann_ret_pct"] - ha04["ann_ret_pct"]
        dv = hc04["ann_vol_pct"] - ha04["ann_vol_pct"]
        dm = hc04["maxdd_pct"] - ha04["maxdd_pct"]
        dc = hc04["calmar"] - ha04["calmar"]
        print(f"{'Delta '+dname:>16s} | {ds04:+9.4f} {ds05:+9.4f} | "
              f"{dr:+7.2f} {dv:+7.2f} {dm:+7.2f} {dc:+7.2f}")

    hmr.to_csv(TBL_DIR / "meta_portfolios_hourly.csv", index=False)

    # ── 2. STATISTICAL TESTS (hourly) ──
    print(f"\n{'='*75}")
    print(f"  HOURLY STATISTICAL TESTS")
    print(f"{'='*75}")

    comm_h = COMM_LEVELS_HOURLY[0]
    htest_rows = []

    # META tests
    for comp_name, comp_label in [("META-D", "META: A vs D"), ("META-BEST", "META: A vs BEST")]:
        idx = hmeta_returns["META-A"].index.intersection(hmeta_returns[comp_name].index)
        r_a = hmeta_returns["META-A"].reindex(idx).fillna(0).values
        r_b = hmeta_returns[comp_name].reindex(idx).fillna(0).values

        d = r_b - r_a
        t_stat, p_ttest = stats.ttest_rel(r_b, r_a)
        dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))) if np.std(d, ddof=1) > 0 else 0
        p_dm = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        delta_sh = calc_sharpe(r_b, BPY_HOURLY) - calc_sharpe(r_a, BPY_HOURLY)
        boot = bootstrap_delta_sharpe(r_a, r_b, bpy=BPY_HOURLY)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        htest_rows.append({
            "test": comp_label, "delta_sharpe": round(delta_sh, 4),
            "t_stat": round(t_stat, 3), "p_ttest": round(p_ttest, 6),
            "dm_stat": round(dm_stat, 3), "p_dm": round(p_dm, 6),
            "boot_ci_lo": round(ci_lo, 4), "boot_ci_hi": round(ci_hi, 4),
        })

    # Per-strategy tests
    for strat in STRATEGIES:
        best_appr = BEST_APP_HOURLY[strat]
        r_a_s = hport_cache[(strat, "A", comm_h)]
        r_b_s = hport_cache[(strat, best_appr, comm_h)]
        idx = r_a_s.index.intersection(r_b_s.index)
        ra = r_a_s.reindex(idx).fillna(0).values
        rb = r_b_s.reindex(idx).fillna(0).values

        d = rb - ra
        t_stat, p_ttest = stats.ttest_rel(rb, ra)
        dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))) if np.std(d, ddof=1) > 0 else 0
        p_dm = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        delta_sh = calc_sharpe(rb, BPY_HOURLY) - calc_sharpe(ra, BPY_HOURLY)
        boot = bootstrap_delta_sharpe(ra, rb, bpy=BPY_HOURLY)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        label = f"{strat.split('_')[0]}: A vs {best_appr}"
        htest_rows.append({
            "test": label, "delta_sharpe": round(delta_sh, 4),
            "t_stat": round(t_stat, 3), "p_ttest": round(p_ttest, 6),
            "dm_stat": round(dm_stat, 3), "p_dm": round(p_dm, 6),
            "boot_ci_lo": round(ci_lo, 4), "boot_ci_hi": round(ci_hi, 4),
        })

    # Print
    print(f"{'Test':>18s} | {'dSharpe':>8s} | {'t-stat':>7s} {'p(t)':>10s} | "
          f"{'DM':>7s} {'p(DM)':>10s} | {'Boot 95% CI':>18s}")
    print(f"{'-'*18}-+-{'-'*8}-+-{'-'*7}-{'-'*10}-+-{'-'*7}-{'-'*10}-+-{'-'*18}")
    for r in htest_rows:
        sig = ""
        p = r["p_ttest"]
        if p < 0.001: sig = " ***"
        elif p < 0.01: sig = "  **"
        elif p < 0.05: sig = "   *"
        print(f"{r['test']:>18s} | {r['delta_sharpe']:+8.4f} | {r['t_stat']:7.3f} {r['p_ttest']:10.6f} | "
              f"{r['dm_stat']:7.3f} {r['p_dm']:10.6f} | [{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}]{sig}")

    pd.DataFrame(htest_rows).to_csv(TBL_DIR / "stat_tests_hourly.csv", index=False)

    # ── 3. EQUITY CURVES (hourly) ──
    print("\nSaving hourly equity curves...")

    hmeta_eq = pd.DataFrame(index=hmeta_returns["META-A"].index)
    for name in ["META-A", "META-D", "META-BEST"]:
        hmeta_eq[name.replace("-", "_")] = (1 + hmeta_returns[name]).cumprod()
    hmeta_eq.index.name = "date"
    hmeta_eq = hmeta_eq.sort_index()
    hmeta_eq.to_csv(TBL_DIR / "equity_curves_meta_hourly.csv")

    # ── 4. HOURLY PLOTS ──
    print("Generating hourly plots...")

    # PLOT 1: Hourly meta equity curves
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_h = {"META_A": "#888888", "META_D": "#d62728", "META_BEST": "#2ca02c"}
    hlabels = {}
    for name in ["META-A", "META-D", "META-BEST"]:
        col = name.replace("-", "_")
        sh = calc_sharpe(hmeta_returns[name].values, BPY_HOURLY)
        hlabels[col] = f"{name} (Sharpe={sh:.2f})"

    hdates = pd.to_datetime(hmeta_eq.index)
    for col in ["META_A", "META_D", "META_BEST"]:
        ax.plot(hdates, hmeta_eq[col].values, label=hlabels[col],
                color=colors_h[col], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Equity")
    ax.set_title("Hourly Meta-Portfolio: Baseline (A) vs Volatility Forecasts (D, BEST)")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meta_equity_curves_hourly.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: meta_equity_curves_hourly.png")

    # PLOT 2: Hourly sharpe barplot
    hport_df = pd.read_csv(TBL_DIR / "v4_portfolios_hourly.csv")
    hew_df = hport_df[hport_df["method"] == "EW"].copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(STRATEGIES))
    width = 0.18
    bar_colors = {"A": "#999999", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}

    for i, appr in enumerate(APPROACHES):
        vals = []
        for strat in STRATEGIES:
            row = hew_df[(hew_df["strategy"] == strat) & (hew_df["approach"] == appr)]
            vals.append(row.iloc[0]["net_sharpe"] if len(row) > 0 else 0)
        ax.bar(x + i * width - 1.5 * width, vals, width,
               label=f"Approach {appr}", color=bar_colors[appr], edgecolor="white")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Net Sharpe (@ 0.04%)")
    ax.set_title("Hourly Per-Strategy EW Portfolio: Net Sharpe by Approach")
    ax.set_xticks(x)
    short_names_h = [s.replace("S1_", "S1\n").replace("S2_", "S2\n").replace("S3_", "S3\n")
                     .replace("S4_", "S4\n").replace("S5_", "S5\n").replace("S6_", "S6\n")
                     for s in STRATEGIES]
    ax.set_xticklabels(short_names_h)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sharpe_comparison_barplot_hourly.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: sharpe_comparison_barplot_hourly.png")

    # PLOT 3: Hourly per-strategy equity curves (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
    hstrat_pairs = [
        ("S1_MeanRev", "A", BEST_APP_HOURLY["S1_MeanRev"]),
        ("S2_Bollinger", "A", BEST_APP_HOURLY["S2_Bollinger"]),
        ("S3_Donchian", "A", BEST_APP_HOURLY["S3_Donchian"]),
        ("S4_Supertrend", "A", BEST_APP_HOURLY["S4_Supertrend"]),
        ("S5_PivotPoints", "A", BEST_APP_HOURLY["S5_PivotPoints"]),
        ("S6_VWAP", "A", BEST_APP_HOURLY["S6_VWAP"]),
    ]

    for idx_p, (strat, app_a, app_best) in enumerate(hstrat_pairs):
        ax = axes[idx_p // 3, idx_p % 3]
        ret_a = hport_cache[(strat, "A", comm_h)]
        ret_b = hport_cache[(strat, app_best, comm_h)]
        eq_a_s = (1 + ret_a).cumprod()
        eq_b_s = (1 + ret_b).cumprod()
        sh_a = calc_sharpe(ret_a.values, BPY_HOURLY)
        sh_b = calc_sharpe(ret_b.values, BPY_HOURLY)
        dates_p = pd.to_datetime(eq_a_s.index)

        ax.plot(dates_p, eq_a_s.values, color="#888888", linewidth=1.5,
                label=f"A ({sh_a:.2f})")
        ax.plot(dates_p, eq_b_s.values, color="#d62728", linewidth=1.5,
                label=f"{app_best} ({sh_b:.2f})")
        ax.set_title(strat, fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    fig.suptitle("Hourly Per-Strategy Equity: Baseline A vs Best Forecast", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategy_equity_curves_hourly.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: strategy_equity_curves_hourly.png")

    # PLOT 4: Hourly drawdown comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(STRATEGIES))
    width = 0.3

    hdd_a = []
    hdd_best = []
    hbest_labels = []
    for strat in STRATEGIES:
        best_appr = BEST_APP_HOURLY[strat]
        eq_a_s = (1 + hport_cache[(strat, "A", comm_h)]).cumprod()
        hdd_a.append(calc_maxdd(eq_a_s.values))
        eq_b_s = (1 + hport_cache[(strat, best_appr, comm_h)]).cumprod()
        hdd_best.append(calc_maxdd(eq_b_s.values))
        hbest_labels.append(best_appr)

    ax.bar(x - width / 2, hdd_a, width, label="Approach A", color="#999999", edgecolor="white")
    ax.bar(x + width / 2, hdd_best, width, label="Best (B/C/D)", color="#d62728", edgecolor="white")

    for i, lbl in enumerate(hbest_labels):
        ax.text(i + width / 2, hdd_best[i] - 0.3, lbl, ha="center", va="top", fontsize=10, fontweight="bold")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Hourly Max Drawdown: Baseline A vs Best Forecast Approach")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names_h)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drawdown_comparison_hourly.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: drawdown_comparison_hourly.png")

    print(f"\nAll done. Files saved to:\n  {TBL_DIR}/\n  {FIG_DIR}/")


if __name__ == "__main__":
    main()
