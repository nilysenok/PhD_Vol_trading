#!/usr/bin/env python3
"""
fix_msp_b.py — Regenerate MSP_B_inst.csv and MSP_B_std.csv using the exact
same method as stage5_msp.py (build_msp), then recompute bootstrap B>A and CE B>A.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

BASE = Path(__file__).resolve().parent.parent
WF_DIR = BASE / "results/final/strategies/walkforward"
MSP_DIR = BASE / "results/final/strategies/msp"
PERSTRAT_RET_DIR = MSP_DIR / "perstrat_daily_returns"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BPY = 252
COMM_DAILY = {"inst": 0.0005, "std": 0.0006}


def load_key_rate():
    if KR_PATH.exists():
        kr = pd.read_parquet(KR_PATH)
        kr["date"] = pd.to_datetime(kr["date"])
        return kr.set_index("date")["key_rate"].sort_index()
    dates = pd.bdate_range("2019-01-01", "2026-06-01")
    return pd.Series(0.16, index=dates, name="key_rate")


def build_portfolio_returns(pos_df, kr_series, comm):
    """Exact copy of stage5_msp.build_portfolio_returns."""
    ticker_frames = []
    for tick in pos_df["ticker"].unique():
        sub = pos_df[pos_df["ticker"] == tick].sort_values("date").reset_index(drop=True)
        pos = sub["position"].values
        gross = sub["daily_gross_return"].values
        dp = np.diff(pos, prepend=0.0)
        net = gross - np.abs(dp) * comm
        sub = sub.copy()
        sub["net_ret"] = net
        sub["abs_pos"] = np.abs(pos)
        sub["turnover"] = np.abs(dp)
        ticker_frames.append(sub[["date", "ticker", "net_ret", "abs_pos", "turnover"]])

    if not ticker_frames:
        return None

    all_tk = pd.concat(ticker_frames, ignore_index=True)
    port = all_tk.groupby("date").agg(
        portfolio_return=("net_ret", "mean"),
        portfolio_exposure=("abs_pos", "mean"),
        turnover=("turnover", "mean"),
    ).sort_index()

    rf_daily = kr_series.reindex(port.index).ffill().bfill() / 365.0
    idx = port.index
    cal_days = np.ones(len(idx))
    for i in range(1, len(idx)):
        cal_days[i] = (idx[i] - idx[i - 1]).days

    port["rf_daily"] = rf_daily.values
    port["cal_days"] = cal_days
    exp_arr = port["portfolio_exposure"].values

    # FDR: cash earns RF (ONCE at portfolio level)
    port["fdr"] = np.maximum(0.0, 1.0 - exp_arr) * port["rf_daily"].values * cal_days
    kr_ann = kr_series.reindex(port.index).ffill().bfill().values
    port["margin_cost"] = np.maximum(0.0, exp_arr - 1.0) * (kr_ann + 0.02) / 365.0

    port["portfolio_total"] = port["portfolio_return"] + port["fdr"] - port["margin_cost"]
    port["excess"] = port["portfolio_total"] - port["rf_daily"] * port["cal_days"]

    return port


def sharpe_from_series(xs):
    s = np.std(xs, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(xs) * sqrt(BPY) / s


def bootstrap_sharpe_diff(ret_bcd, ret_a, n_boot=10000):
    """One-sided test: Sharpe(BCD) > Sharpe(A)."""
    n = len(ret_bcd)
    observed = sharpe_from_series(ret_bcd) - sharpe_from_series(ret_a)
    np.random.seed(42)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        deltas[i] = sharpe_from_series(ret_bcd[idx]) - sharpe_from_series(ret_a[idx])
    p = np.mean(deltas <= 0)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return observed, p, ci_lo, ci_hi


def ce_bootstrap(ret_bcd, ret_a, gamma, n_boot=10000):
    """CE difference BCD - A, bootstrap CI."""
    def ce(r, g):
        w = 1.0 + r
        w = np.maximum(w, 1e-10)
        if g == 1:
            return np.exp(np.mean(np.log(w))) - 1
        u = np.mean(w ** (1 - g))
        return u ** (1 / (1 - g)) - 1

    obs = (ce(ret_bcd, gamma) - ce(ret_a, gamma)) * BPY * 100  # ann bps
    np.random.seed(42)
    n = len(ret_bcd)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        deltas[i] = (ce(ret_bcd[idx], gamma) - ce(ret_a[idx], gamma)) * BPY * 100
    p = np.mean(deltas <= 0)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return obs, p, ci_lo, ci_hi


def main():
    print("=" * 70)
    print("Regenerating MSP-B from wf_positions_B.parquet")
    print("=" * 70)

    kr_series = load_key_rate()

    # Load B positions
    fp = WF_DIR / "wf_positions_B.parquet"
    pos_b = pd.read_parquet(fp)
    pos_b["date"] = pd.to_datetime(pos_b["date"])
    pos_b["approach"] = "B"
    print(f"Loaded {fp.name}: {len(pos_b)} rows")

    # Aggregate: mean across 6 strategies per (date, ticker)
    grp = pos_b.groupby(["date", "ticker"]).agg(
        position=("position", "mean"),
        daily_gross_return=("daily_gross_return", "mean"),
    ).reset_index()
    print(f"Aggregated to {len(grp)} date×ticker rows")
    print(f"Mean |position|: {grp['position'].abs().mean():.4f}")

    for comm_label, comm_val in COMM_DAILY.items():
        port = build_portfolio_returns(grp, kr_series, comm_val)
        if port is None:
            print(f"ERROR: build_portfolio_returns returned None for {comm_label}")
            continue

        exp = port["portfolio_exposure"].mean() * 100
        tr_vol = port["portfolio_return"].std() * sqrt(BPY) * 100
        tot_vol = port["portfolio_total"].std() * sqrt(BPY) * 100
        xs_sharpe = sharpe_from_series(port["excess"].values)
        tr_sharpe = sharpe_from_series(port["portfolio_return"].values)
        ann_ret = port["portfolio_total"].mean() * BPY * 100

        # MDD
        cum = np.cumsum(port["portfolio_total"].values)
        mdd = (cum - np.maximum.accumulate(cum)).min() * 100

        print(f"\n--- MSP-B {comm_label} ({comm_val}) ---")
        print(f"  Exposure:       {exp:.1f}%")
        print(f"  Trading Vol:    {tr_vol:.2f}%")
        print(f"  Total Vol:      {tot_vol:.2f}%")
        print(f"  XS Sharpe:      {xs_sharpe:.2f}")
        print(f"  Trading Sharpe: {tr_sharpe:.2f}")
        print(f"  Ann Return:     {ann_ret:.1f}%")
        print(f"  MDD:            {mdd:.1f}%")

        # Per-year
        port["year"] = port.index.year
        for yr in [2022, 2023, 2024, 2025]:
            ydf = port[port["year"] == yr]
            if len(ydf) < 20:
                continue
            yxs = sharpe_from_series(ydf["excess"].values)
            yret = ydf["portfolio_total"].mean() * BPY * 100
            print(f"  {yr}: XS Sh={yxs:+.2f}  Return={yret:+.1f}%")

        # Save corrected file
        ret_df = port[["portfolio_return", "portfolio_total", "excess", "portfolio_exposure"]].copy()
        ret_df.columns = ["trading_ret", "total_ret", "excess_ret", "exposure"]
        out_path = PERSTRAT_RET_DIR / f"MSP_B_{comm_label}.csv"
        ret_df.to_csv(out_path)
        print(f"  Saved: {out_path}")

    # ── Bootstrap B>A ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Bootstrap B>A (10,000 replications)")
    print("=" * 70)

    a_inst = pd.read_csv(PERSTRAT_RET_DIR / "MSP_A_inst.csv", parse_dates=["date"])
    b_inst = pd.read_csv(PERSTRAT_RET_DIR / "MSP_B_inst.csv", parse_dates=["date"])

    common = a_inst.set_index("date").index.intersection(b_inst.set_index("date").index)
    ret_a = a_inst.set_index("date").loc[common, "excess_ret"].values
    ret_b = b_inst.set_index("date").loc[common, "excess_ret"].values

    delta, p, ci_lo, ci_hi = bootstrap_sharpe_diff(ret_b, ret_a)
    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
    print(f"  ΔSharpe(B>A): {delta:+.4f}")
    print(f"  p-value:      {p:.4f} {stars}")
    print(f"  95% CI:       [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── CE Bootstrap B>A ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CE Bootstrap B>A")
    print("=" * 70)

    # Use total_ret for CE (includes FDR)
    tot_a = a_inst.set_index("date").loc[common, "total_ret"].values
    tot_b = b_inst.set_index("date").loc[common, "total_ret"].values

    for gamma in [3, 5, 10]:
        ce_diff, p, ci_lo, ci_hi = ce_bootstrap(tot_b, tot_a, gamma)
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"  γ={gamma:2d}: ΔCE={ce_diff:+.2f}%/yr  p={p:.4f} {stars}  CI=[{ci_lo:.2f}, {ci_hi:.2f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
