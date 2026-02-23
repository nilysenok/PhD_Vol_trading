#!/usr/bin/env python3
"""V4 Meta-portfolios: separate B, C, D + bootstrap tests."""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
TBL_DIR = OUT_DIR / "tables"

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
BCD_YEARS = [2022, 2023, 2024, 2025]
COMM = 0.0005
BPY = 252

BEST_APP = {
    "S1_MeanRev": "C", "S2_Bollinger": "C", "S3_Donchian": "B",
    "S4_Supertrend": "D", "S5_PivotPoints": "C", "S6_VWAP": "D",
}

# Hourly parameters
COMM_HOURLY = 0.0004
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


def bootstrap_delta_sharpe(r_a, r_b, n_boot=10000, bpy=252, seed=42):
    rng = np.random.RandomState(seed)
    T = len(r_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, T, T)
        deltas[i] = calc_sharpe(r_b[idx], bpy) - calc_sharpe(r_a[idx], bpy)
    return deltas


def build_ew_portfolio(pos_df, strat, appr, comm=COMM, ticker_universe=None):
    sub = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == appr)]
    ticker_rets = {}
    for tkr in sorted(sub["ticker"].unique()):
        g = sub[sub["ticker"] == tkr].sort_values("date")
        nr = net_returns(g["position"].values, g["daily_gross_return"].values, comm)
        ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)
    ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
    # Ensure consistent ticker universe (fix for missing tickers in approach C)
    if ticker_universe is not None:
        for tkr in ticker_universe:
            if tkr not in ret_df.columns:
                ret_df[tkr] = 0.0
    return ret_df.mean(axis=1)


def metrics(r, bpy=BPY):
    v = r.values
    sh = calc_sharpe(v, bpy)
    ann_ret = np.mean(v) * bpy * 100
    ann_vol = np.std(v, ddof=1) * np.sqrt(bpy) * 100
    eq = (1 + r).cumprod()
    mdd = calc_maxdd(eq.values)
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-6 else 0.0
    return sh, ann_ret, ann_vol, mdd, calmar


def main():
    print("Loading data...")
    pos_df = pd.read_parquet(OUT_DIR / "daily_positions.parquet")
    pos_df = pos_df[(pos_df["tf"] == "daily") & (pos_df["test_year"].isin(BCD_YEARS))].copy()

    # Get ticker universe from approach A for each strategy
    ticker_universes = {}
    for strat in STRATEGIES:
        sub_a = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == "A")]
        ticker_universes[strat] = sorted(sub_a["ticker"].unique())

    # Build all strategy-level EW portfolios
    print("Building EW portfolios...")
    cache = {}
    for strat in STRATEGIES:
        for appr in APPROACHES:
            cache[(strat, appr)] = build_ew_portfolio(
                pos_df, strat, appr, ticker_universe=ticker_universes[strat])

    # Build meta-portfolios
    def build_meta(approach):
        rets = [cache[(s, approach)] for s in STRATEGIES]
        aligned = pd.DataFrame({s: r for s, r in zip(STRATEGIES, rets)}).sort_index().fillna(0)
        return aligned.mean(axis=1)

    meta = {}
    for appr in APPROACHES:
        meta[f"META-{appr}"] = build_meta(appr)

    # META-MEAN(BCD)
    meta_bcd = pd.DataFrame({
        "B": meta["META-B"], "C": meta["META-C"], "D": meta["META-D"]
    }).sort_index().fillna(0)
    meta["META-MEAN(BCD)"] = meta_bcd.mean(axis=1)

    # META-BEST
    best_rets = [cache[(s, BEST_APP[s])] for s in STRATEGIES]
    aligned_best = pd.DataFrame({s: r for s, r in zip(STRATEGIES, best_rets)}).sort_index().fillna(0)
    meta["META-BEST"] = aligned_best.mean(axis=1)

    # Compute metrics
    names_order = ["META-A", "META-B", "META-C", "META-D", "META-MEAN(BCD)", "META-BEST"]
    rows = []
    for name in names_order:
        sh, ar, av, mdd, cal = metrics(meta[name])
        rows.append({
            "portfolio": name,
            "net_sharpe": round(sh, 4), "ann_ret_pct": round(ar, 2),
            "ann_vol_pct": round(av, 2), "maxdd_pct": round(mdd, 2),
            "calmar": round(cal, 2),
        })

    # Print table
    print(f"\n{'='*75}")
    print(f"  META-PORTFOLIOS (EW 6 strategies x EW 17 tickers, Net @ {COMM*100:.2f}%)")
    print(f"{'='*75}")
    print(f"{'Portfolio':>17s} | {'Sharpe':>8s} {'Ret%':>7s} {'Vol%':>7s} {'MDD%':>7s} {'Calmar':>7s}")
    print(f"{'-'*17}-+-{'-'*8}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")

    for r in rows:
        print(f"{r['portfolio']:>17s} | {r['net_sharpe']:8.4f} {r['ann_ret_pct']:7.2f} "
              f"{r['ann_vol_pct']:7.2f} {r['maxdd_pct']:7.2f} {r['calmar']:7.2f}")

    # Deltas
    print(f"{'-'*17}-+-{'-'*8}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")
    a_m = {k: v for k, v in rows[0].items() if k != "portfolio"}

    delta_names = [
        ("Delta B vs A", "META-B"),
        ("Delta C vs A", "META-C"),
        ("Delta D vs A", "META-D"),
        ("Delta MEAN vs A", "META-MEAN(BCD)"),
        ("Delta BEST vs A", "META-BEST"),
    ]
    for dname, comp in delta_names:
        cr = next(r for r in rows if r["portfolio"] == comp)
        ds = cr["net_sharpe"] - a_m["net_sharpe"]
        dr = cr["ann_ret_pct"] - a_m["ann_ret_pct"]
        dv = cr["ann_vol_pct"] - a_m["ann_vol_pct"]
        dm = cr["maxdd_pct"] - a_m["maxdd_pct"]
        dc = cr["calmar"] - a_m["calmar"]
        print(f"{dname:>17s} | {ds:+8.4f} {dr:+7.2f} {dv:+7.2f} {dm:+7.2f} {dc:+7.2f}")

    # Bootstrap tests
    print(f"\n{'='*65}")
    print(f"  BOOTSTRAP 95% CI for Delta Sharpe vs A (10,000 samples)")
    print(f"{'='*65}")
    print(f"{'Comparison':>17s} | {'dSharpe':>8s} | {'95% CI':>22s}")
    print(f"{'-'*17}-+-{'-'*8}-+-{'-'*22}")

    r_a = meta["META-A"].values
    for label, comp in [("A vs B", "META-B"), ("A vs C", "META-C"),
                        ("A vs D", "META-D"), ("A vs MEAN(BCD)", "META-MEAN(BCD)"),
                        ("A vs BEST", "META-BEST")]:
        idx = meta["META-A"].index.intersection(meta[comp].index)
        ra = meta["META-A"].reindex(idx).fillna(0).values
        rb = meta[comp].reindex(idx).fillna(0).values
        ds = calc_sharpe(rb, BPY) - calc_sharpe(ra, BPY)
        boot = bootstrap_delta_sharpe(ra, rb)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = ""
        if lo > 0: sig = " *"
        print(f"{label:>17s} | {ds:+8.4f} | [{lo:+8.4f}, {hi:+8.4f}]{sig}")

    # Save
    out_df = pd.DataFrame(rows)
    out_df.to_csv(TBL_DIR / "meta_portfolios_bcd.csv", index=False)
    print(f"\nSaved: {TBL_DIR / 'meta_portfolios_bcd.csv'}")

    # Save ALL 6 equity curves (daily)
    eq_all = {}
    for name in names_order:
        eq_all[name.replace("META-", "META_").replace("(BCD)", "BCD")] = (1 + meta[name]).cumprod()
    eq_df = pd.DataFrame(eq_all)
    eq_df.index.name = "date"
    eq_df.to_csv(TBL_DIR / "equity_curves_meta_all.csv")
    print(f"Saved: {TBL_DIR / 'equity_curves_meta_all.csv'}")

    # Save daily strategy-level returns for correlation (all approaches A)
    strat_rets_daily = {}
    for strat in STRATEGIES:
        strat_rets_daily[strat] = cache[(strat, "A")]
    strat_df = pd.DataFrame(strat_rets_daily).sort_index().fillna(0)
    strat_df.index.name = "date"
    strat_df.to_csv(TBL_DIR / "strategy_returns_daily.csv")
    print(f"Saved: {TBL_DIR / 'strategy_returns_daily.csv'}")

    # ════════════════════════════════════════════════════════════
    # HOURLY PIPELINE
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print(f"  HOURLY PIPELINE")
    print(f"{'='*75}")

    pos_all = pd.read_parquet(OUT_DIR / "daily_positions.parquet")
    pos_hourly = pos_all[(pos_all["tf"] == "hourly") &
                         (pos_all["test_year"].isin(BCD_YEARS))].copy()
    print(f"  Hourly BCD rows: {len(pos_hourly):,}")

    if len(pos_hourly) == 0:
        print("  No hourly data found, skipping.")
        return

    # Get hourly ticker universe from approach A
    hticker_universes = {}
    for strat in STRATEGIES:
        sub_a = pos_hourly[(pos_hourly["strategy"] == strat) & (pos_hourly["approach"] == "A")]
        hticker_universes[strat] = sorted(sub_a["ticker"].unique())

    # Build all strategy-level EW portfolios (hourly)
    print("Building hourly EW portfolios...")
    hcache = {}
    for strat in STRATEGIES:
        for appr in APPROACHES:
            hcache[(strat, appr)] = build_ew_portfolio(
                pos_hourly, strat, appr, COMM_HOURLY,
                ticker_universe=hticker_universes[strat])

    # Build hourly meta-portfolios
    def build_hmeta(approach):
        rets = [hcache[(s, approach)] for s in STRATEGIES]
        aligned = pd.DataFrame({s: r for s, r in zip(STRATEGIES, rets)}).sort_index().fillna(0)
        return aligned.mean(axis=1)

    hmeta = {}
    for appr in APPROACHES:
        hmeta[f"META-{appr}"] = build_hmeta(appr)

    # META-MEAN(BCD)
    hmeta_bcd = pd.DataFrame({
        "B": hmeta["META-B"], "C": hmeta["META-C"], "D": hmeta["META-D"]
    }).sort_index().fillna(0)
    hmeta["META-MEAN(BCD)"] = hmeta_bcd.mean(axis=1)

    # META-BEST
    hbest_rets = [hcache[(s, BEST_APP_HOURLY[s])] for s in STRATEGIES]
    haligned_best = pd.DataFrame({s: r for s, r in zip(STRATEGIES, hbest_rets)}).sort_index().fillna(0)
    hmeta["META-BEST"] = haligned_best.mean(axis=1)

    # Compute metrics
    hnames_order = ["META-A", "META-B", "META-C", "META-D", "META-MEAN(BCD)", "META-BEST"]
    hrows = []
    for name in hnames_order:
        sh, ar, av, mdd, cal = metrics(hmeta[name], BPY_HOURLY)
        hrows.append({
            "portfolio": name,
            "net_sharpe": round(sh, 4), "ann_ret_pct": round(ar, 2),
            "ann_vol_pct": round(av, 2), "maxdd_pct": round(mdd, 2),
            "calmar": round(cal, 2),
        })

    # Print table
    print(f"\n{'='*75}")
    print(f"  HOURLY META-PORTFOLIOS (EW 6 strategies x EW 17 tickers, Net @ {COMM_HOURLY*100:.2f}%)")
    print(f"{'='*75}")
    print(f"{'Portfolio':>17s} | {'Sharpe':>8s} {'Ret%':>7s} {'Vol%':>7s} {'MDD%':>7s} {'Calmar':>7s}")
    print(f"{'-'*17}-+-{'-'*8}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")

    for r in hrows:
        print(f"{r['portfolio']:>17s} | {r['net_sharpe']:8.4f} {r['ann_ret_pct']:7.2f} "
              f"{r['ann_vol_pct']:7.2f} {r['maxdd_pct']:7.2f} {r['calmar']:7.2f}")

    # Deltas
    print(f"{'-'*17}-+-{'-'*8}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")
    ha_m = {k: v for k, v in hrows[0].items() if k != "portfolio"}

    hdelta_names = [
        ("Delta B vs A", "META-B"),
        ("Delta C vs A", "META-C"),
        ("Delta D vs A", "META-D"),
        ("Delta MEAN vs A", "META-MEAN(BCD)"),
        ("Delta BEST vs A", "META-BEST"),
    ]
    for dname, comp in hdelta_names:
        cr = next(r for r in hrows if r["portfolio"] == comp)
        ds = cr["net_sharpe"] - ha_m["net_sharpe"]
        dr = cr["ann_ret_pct"] - ha_m["ann_ret_pct"]
        dv = cr["ann_vol_pct"] - ha_m["ann_vol_pct"]
        dm = cr["maxdd_pct"] - ha_m["maxdd_pct"]
        dc = cr["calmar"] - ha_m["calmar"]
        print(f"{dname:>17s} | {ds:+8.4f} {dr:+7.2f} {dv:+7.2f} {dm:+7.2f} {dc:+7.2f}")

    # Hourly bootstrap tests
    print(f"\n{'='*65}")
    print(f"  HOURLY BOOTSTRAP 95% CI for Delta Sharpe vs A (10,000 samples)")
    print(f"{'='*65}")
    print(f"{'Comparison':>17s} | {'dSharpe':>8s} | {'95% CI':>22s}")
    print(f"{'-'*17}-+-{'-'*8}-+-{'-'*22}")

    for label, comp in [("A vs B", "META-B"), ("A vs C", "META-C"),
                        ("A vs D", "META-D"), ("A vs MEAN(BCD)", "META-MEAN(BCD)"),
                        ("A vs BEST", "META-BEST")]:
        idx = hmeta["META-A"].index.intersection(hmeta[comp].index)
        hra = hmeta["META-A"].reindex(idx).fillna(0).values
        hrb = hmeta[comp].reindex(idx).fillna(0).values
        ds = calc_sharpe(hrb, BPY_HOURLY) - calc_sharpe(hra, BPY_HOURLY)
        boot = bootstrap_delta_sharpe(hra, hrb, bpy=BPY_HOURLY)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = ""
        if lo > 0: sig = " *"
        print(f"{label:>17s} | {ds:+8.4f} | [{lo:+8.4f}, {hi:+8.4f}]{sig}")

    # Save hourly
    hout_df = pd.DataFrame(hrows)
    hout_df.to_csv(TBL_DIR / "meta_portfolios_bcd_hourly.csv", index=False)
    print(f"\nSaved: {TBL_DIR / 'meta_portfolios_bcd_hourly.csv'}")

    # Save ALL 6 equity curves (hourly)
    heq_all = {}
    for name in hnames_order:
        heq_all[name.replace("META-", "META_").replace("(BCD)", "BCD")] = (1 + hmeta[name]).cumprod()
    heq_df = pd.DataFrame(heq_all)
    heq_df.index.name = "date"
    heq_df.to_csv(TBL_DIR / "equity_curves_meta_all_hourly.csv")
    print(f"Saved: {TBL_DIR / 'equity_curves_meta_all_hourly.csv'}")

    # Save hourly strategy-level returns for correlation
    strat_rets_hourly = {}
    for strat in STRATEGIES:
        strat_rets_hourly[strat] = hcache[(strat, "A")]
    hstrat_df = pd.DataFrame(strat_rets_hourly).sort_index().fillna(0)
    hstrat_df.index.name = "date"
    hstrat_df.to_csv(TBL_DIR / "strategy_returns_hourly.csv")
    print(f"Saved: {TBL_DIR / 'strategy_returns_hourly.csv'}")


if __name__ == "__main__":
    main()
