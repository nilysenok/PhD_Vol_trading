#!/usr/bin/env python3
"""V4 Portfolio analysis: 6 strategies × 4 approaches × 4 weighting methods."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
BCD_YEARS = [2022, 2023, 2024, 2025]
COMM = 0.0005          # per-side commission for net Sharpe (0.05%)
LOOKBACK_IV = 252      # InvVol lookback
W_CAP = 0.20           # max weight per ticker
N_TICKERS = 17
BPY = 252              # bars per year (daily)

# Hourly parameters
COMM_HOURLY = 0.0004         # 0.04% per side
BPY_HOURLY = 2268            # 252 × 9 hourly bars
LOOKBACK_IV_HOURLY = 2268    # 1 year of hourly bars


def net_returns_series(pos_s, gross_r_s, comm):
    """Compute net returns from position and gross return series."""
    dpos = pos_s.diff().fillna(pos_s.iloc[0] if len(pos_s) > 0 else 0)
    return gross_r_s - np.abs(dpos) * comm


def calc_sharpe(r, bpy=252):
    if len(r) < 2:
        return 0.0
    s = np.std(r, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(r) / s * np.sqrt(bpy)


def calc_maxdd(cum_r):
    """Max drawdown from cumulative return series (1+r).cumprod()."""
    peak = np.maximum.accumulate(cum_r)
    dd = (cum_r - peak) / np.where(peak > 0, peak, 1)
    return dd.min() * 100  # percent


def clip_weights(w, cap):
    """Clip weights to cap and renormalize."""
    w = np.maximum(w, 0)
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        w[over] = cap
        residual = 1.0 - w.sum()
        under = ~over
        if under.sum() == 0:
            break
        w[under] += residual * (w[under] / (w[under].sum() + 1e-15))
    w = w / (w.sum() + 1e-15)
    return w


def weight_ew(n):
    return np.ones(n) / n


def weight_invvol(ret_matrix, cap=0.20):
    """Inverse volatility weights."""
    n = ret_matrix.shape[1]
    vols = np.std(ret_matrix, axis=0, ddof=1)
    vols = np.where(vols < 1e-10, 1e-10, vols)
    inv = 1.0 / vols
    w = inv / inv.sum()
    return clip_weights(w, cap)


def weight_max_sharpe(ret_matrix, cap=0.20):
    """Max Sharpe portfolio with Ledoit-Wolf covariance."""
    n = ret_matrix.shape[1]
    if ret_matrix.shape[0] < n + 5:
        return np.ones(n) / n

    mu = np.mean(ret_matrix, axis=0)
    try:
        lw = LedoitWolf().fit(ret_matrix)
        cov = lw.covariance_
    except Exception:
        return np.ones(n) / n

    def neg_sharpe(w):
        pr = w @ mu
        pv = np.sqrt(w @ cov @ w + 1e-15)
        return -pr / pv

    w0 = np.ones(n) / n
    bounds = [(0, cap)] * n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    try:
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            return clip_weights(res.x, cap)
    except Exception:
        pass
    return np.ones(n) / n


def weight_minvar(ret_matrix, cap=0.20):
    """Minimum variance portfolio with Ledoit-Wolf covariance."""
    n = ret_matrix.shape[1]
    if ret_matrix.shape[0] < n + 5:
        return np.ones(n) / n

    try:
        lw = LedoitWolf().fit(ret_matrix)
        cov = lw.covariance_
    except Exception:
        return np.ones(n) / n

    def port_var(w):
        return w @ cov @ w

    w0 = np.ones(n) / n
    bounds = [(0, cap)] * n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    try:
        res = minimize(port_var, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-15})
        if res.success:
            return clip_weights(res.x, cap)
    except Exception:
        pass
    return np.ones(n) / n


WEIGHT_FUNCS = {
    "EW":        lambda ret_mat, cap: weight_ew(ret_mat.shape[1]),
    "InvVol":    weight_invvol,
    "MaxSharpe": weight_max_sharpe,
    "MinVar":    weight_minvar,
}


def build_portfolio(ticker_net_rets, method, cap=W_CAP, lookback_iv=252,
                    rebal_comm=0.0):
    """
    Build monthly-rebalanced walk-forward portfolio.

    ticker_net_rets: DataFrame, index=date, columns=tickers, values=net returns
    rebal_comm: commission for weight rebalancing (per side). When > 0, the cost
                of changing portfolio weights at each month boundary is deducted.
                EW always targets 1/N → minimal rebalancing; MinVar/InvVol/MaxSharpe
                change targets each month → higher turnover cost.
    Returns: Series of portfolio returns
    """
    tickers = list(ticker_net_rets.columns)
    n = len(tickers)
    dates = ticker_net_rets.index
    ret_arr = ticker_net_rets.values  # (T, n)

    # Get month boundaries for rebalancing
    months = pd.Series(dates).dt.to_period("M")
    month_starts = []
    prev_m = None
    for i, m in enumerate(months):
        if m != prev_m:
            month_starts.append(i)
            prev_m = m

    wf = WEIGHT_FUNCS[method]
    port_rets = np.zeros(len(dates))
    w_actual = np.zeros(n)  # start from cash

    for k, start_idx in enumerate(month_starts):
        end_idx = month_starts[k + 1] if k + 1 < len(month_starts) else len(dates)

        # Expanding window: all data before this month
        if start_idx == 0:
            w_target = np.ones(n) / n
        else:
            hist = ret_arr[:start_idx]
            if method == "InvVol":
                hist = hist[-lookback_iv:]
            w_target = wf(hist, cap)

        # Rebalancing cost: turnover = sum(|w_target - w_actual|)
        rebal_cost = 0.0
        if rebal_comm > 0:
            turnover = np.sum(np.abs(w_target - w_actual))
            rebal_cost = turnover * rebal_comm

        # Apply target weights for this month
        for t in range(start_idx, end_idx):
            port_rets[t] = np.sum(ret_arr[t] * w_target)

        # Deduct rebalancing cost from first bar of the month
        port_rets[start_idx] -= rebal_cost

        # Track weight drift during the month
        # At month end: w_actual[i] = w_target[i] * prod(1 + r_i) / sum(w_j * prod(1 + r_j))
        cum_growth = np.ones(n)
        for t in range(start_idx, end_idx):
            cum_growth *= (1 + ret_arr[t])
        w_drifted = w_target * cum_growth
        total = w_drifted.sum()
        if total > 1e-15:
            w_actual = w_drifted / total
        else:
            w_actual = w_target.copy()

    return pd.Series(port_rets, index=dates)


def compute_portfolios(pos_df, comm, bpy, lookback_iv, rebal_comm=None):
    """Compute all portfolios for given position data. Returns DataFrame.
    rebal_comm: commission for weight rebalancing (comm + half-spread).
                If None, defaults to 2 * comm (commission + half-spread estimate)."""
    if rebal_comm is None:
        rebal_comm = 2 * comm
    tickers = sorted(pos_df["ticker"].unique())

    # Precompute net returns per (strategy, approach, ticker)
    net_cache = {}
    for (strat, appr, tkr), grp in pos_df.groupby(["strategy", "approach", "ticker"]):
        grp = grp.sort_values("date")
        pos_s = grp["position"].reset_index(drop=True)
        gross_s = grp["daily_gross_return"].reset_index(drop=True)
        dates = grp["date"].reset_index(drop=True)
        net_s = net_returns_series(pos_s, gross_s, comm)
        net_cache[(strat, appr, tkr)] = pd.Series(net_s.values, index=dates.values)

    all_results = []
    total = len(STRATEGIES) * len(APPROACHES) * len(WEIGHT_FUNCS)
    done = 0

    for strat in STRATEGIES:
        for appr in APPROACHES:
            ticker_rets = {}
            for tkr in tickers:
                key = (strat, appr, tkr)
                if key in net_cache:
                    ticker_rets[tkr] = net_cache[key]

            if len(ticker_rets) < 2:
                for method in WEIGHT_FUNCS:
                    all_results.append({
                        "strategy": strat, "approach": appr, "method": method,
                        "net_sharpe": np.nan, "ann_ret_pct": np.nan, "maxdd_pct": np.nan,
                    })
                    done += 1
                continue

            ret_df = pd.DataFrame(ticker_rets)
            ret_df = ret_df.sort_index().fillna(0.0)

            for method in WEIGHT_FUNCS:
                port_ret = build_portfolio(ret_df, method, lookback_iv=lookback_iv,
                                           rebal_comm=rebal_comm)
                sh = calc_sharpe(port_ret.values, bpy)
                ann = np.mean(port_ret.values) * bpy * 100
                cum = (1 + port_ret).cumprod()
                mdd = calc_maxdd(cum.values)

                all_results.append({
                    "strategy": strat, "approach": appr, "method": method,
                    "net_sharpe": round(sh, 4),
                    "ann_ret_pct": round(ann, 2),
                    "maxdd_pct": round(mdd, 2),
                })
                done += 1

            if done % 16 == 0:
                print(f"  {done}/{total} portfolios done")

    print(f"  {done}/{total} portfolios done")
    return pd.DataFrame(all_results)


def print_strategy_tables(res_df, tf_label, comm):
    """Print per-strategy tables and summary. Returns summary_rows."""
    METHODS = ["EW", "InvVol", "MaxSharpe", "MinVar"]
    summary_rows = []

    for strat in STRATEGIES:
        sdf = res_df[res_df["strategy"] == strat]

        print(f"\n{'='*62}")
        print(f"  {strat} {tf_label} — Net Sharpe @ {comm*100:.2f}%")
        print(f"{'='*62}")
        print(f"{'Method':>11s} | {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s}")
        print(f"{'-'*11}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}")

        best_sh = -999
        best_method = ""
        best_appr = ""
        best_ann = 0.0
        best_mdd = 0.0

        for method in METHODS:
            vals = []
            for appr in APPROACHES:
                row = sdf[(sdf["approach"] == appr) & (sdf["method"] == method)]
                if len(row) == 0 or pd.isna(row.iloc[0]["net_sharpe"]):
                    vals.append("  N/A   ")
                else:
                    sh = row.iloc[0]["net_sharpe"]
                    vals.append(f"{sh:8.4f}")
                    if sh > best_sh:
                        best_sh = sh
                        best_method = method
                        best_appr = appr
                        best_ann = row.iloc[0]["ann_ret_pct"]
                        best_mdd = row.iloc[0]["maxdd_pct"]
            print(f"{method:>11s} | {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

        # Ann return row
        print(f"\n{'':>11s}   Ann Return %:")
        for method in METHODS:
            vals = []
            for appr in APPROACHES:
                row = sdf[(sdf["approach"] == appr) & (sdf["method"] == method)]
                if len(row) == 0 or pd.isna(row.iloc[0]["ann_ret_pct"]):
                    vals.append("  N/A   ")
                else:
                    vals.append(f"{row.iloc[0]['ann_ret_pct']:8.2f}")
            print(f"{method:>11s} | {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

        # MaxDD row
        print(f"\n{'':>11s}   Max Drawdown %:")
        for method in METHODS:
            vals = []
            for appr in APPROACHES:
                row = sdf[(sdf["approach"] == appr) & (sdf["method"] == method)]
                if len(row) == 0 or pd.isna(row.iloc[0]["maxdd_pct"]):
                    vals.append("  N/A   ")
                else:
                    vals.append(f"{row.iloc[0]['maxdd_pct']:8.2f}")
            print(f"{method:>11s} | {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

        print(f"\n  >>> Best: {best_method} {best_appr} → "
              f"Sharpe={best_sh:.4f}, Ret={best_ann:.2f}%, MDD={best_mdd:.2f}%")

        summary_rows.append({
            "Strategy": strat, "Method": best_method, "Approach": best_appr,
            "Net_Sharpe": best_sh, "Ann_Ret%": best_ann, "MaxDD%": best_mdd,
        })

    # Summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY ({tf_label}): Best portfolio per strategy (Net Sharpe @ {comm*100:.2f}%)")
    print(f"{'='*72}")
    print(f"{'Strategy':>16s} | {'Method':>10s} {'App':>4s} | {'Sharpe':>8s} {'Ret%':>8s} {'MDD%':>8s}")
    print(f"{'-'*16}-+-{'-'*10}-{'-'*4}-+-{'-'*8}-{'-'*8}-{'-'*8}")
    for r in summary_rows:
        print(f"{r['Strategy']:>16s} | {r['Method']:>10s} {r['Approach']:>4s} | "
              f"{r['Net_Sharpe']:8.4f} {r['Ann_Ret%']:8.2f} {r['MaxDD%']:8.2f}")

    return summary_rows


def main():
    print("Loading data...")
    pos_all = pd.read_parquet(OUT_DIR / "daily_positions.parquet")
    tables_dir = OUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Commission scenarios per timeframe:
    #   Gross (comm=0, rebal=0), Net primary, Net secondary
    DAILY_SCENARIOS = [
        ("gross",    0.0,    0.0,    BPY, LOOKBACK_IV),
        ("net_0.05", COMM,   2*COMM, BPY, LOOKBACK_IV),
        ("net_0.06", 0.0006, 2*0.0006, BPY, LOOKBACK_IV),
    ]
    HOURLY_SCENARIOS = [
        ("gross",    0.0,    0.0,    BPY_HOURLY, LOOKBACK_IV_HOURLY),
        ("net_0.04", COMM_HOURLY, 2*COMM_HOURLY, BPY_HOURLY, LOOKBACK_IV_HOURLY),
        ("net_0.05", 0.0005, 2*0.0005, BPY_HOURLY, LOOKBACK_IV_HOURLY),
    ]

    # ═══════════════════════════════════════════════════════════════
    # DAILY
    # ═══════════════════════════════════════════════════════════════
    pos_daily = pos_all[(pos_all["tf"] == "daily") &
                        (pos_all["test_year"].isin(BCD_YEARS))].copy()
    print(f"  Daily BCD rows: {len(pos_daily):,}")
    tickers = sorted(pos_daily["ticker"].unique())
    print(f"  Tickers: {len(tickers)} — {', '.join(tickers)}")

    all_daily = []
    for label, comm, rebal, bpy, lookback in DAILY_SCENARIOS:
        print(f"\nComputing daily portfolios ({label}, comm={comm*100:.2f}%, rebal={rebal*100:.2f}%)...")
        res = compute_portfolios(pos_daily, comm, bpy, lookback, rebal_comm=rebal)
        res["comm_level"] = label
        all_daily.append(res)
        comm_display = comm if comm > 0 else 0
        print_strategy_tables(res, f"daily {label}", comm_display)

    df_daily = pd.concat(all_daily, ignore_index=True)
    df_daily.to_csv(tables_dir / "v4_portfolios_daily.csv", index=False)
    print(f"\nSaved: v4_portfolios_daily.csv ({len(df_daily)} rows)")

    # ═══════════════════════════════════════════════════════════════
    # HOURLY
    # ═══════════════════════════════════════════════════════════════
    pos_hourly = pos_all[(pos_all["tf"] == "hourly") &
                         (pos_all["test_year"].isin(BCD_YEARS))].copy()
    print(f"\n{'='*62}")
    print(f"  Hourly BCD rows: {len(pos_hourly):,}")
    if len(pos_hourly) == 0:
        print("  No hourly data found, skipping.")
        return
    tickers_h = sorted(pos_hourly["ticker"].unique())
    print(f"  Tickers: {len(tickers_h)} — {', '.join(tickers_h)}")

    all_hourly = []
    for label, comm, rebal, bpy, lookback in HOURLY_SCENARIOS:
        print(f"\nComputing hourly portfolios ({label}, comm={comm*100:.2f}%, rebal={rebal*100:.2f}%)...")
        res = compute_portfolios(pos_hourly, comm, bpy, lookback, rebal_comm=rebal)
        res["comm_level"] = label
        all_hourly.append(res)
        comm_display = comm if comm > 0 else 0
        print_strategy_tables(res, f"hourly {label}", comm_display)

    df_hourly = pd.concat(all_hourly, ignore_index=True)
    df_hourly.to_csv(tables_dir / "v4_portfolios_hourly.csv", index=False)
    print(f"\nSaved: v4_portfolios_hourly.csv ({len(df_hourly)} rows)")


if __name__ == "__main__":
    main()
