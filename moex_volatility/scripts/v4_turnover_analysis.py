#!/usr/bin/env python3
"""
Turnover analysis for portfolio weighting methods.
Computes average monthly turnover per method and Sharpe at various rebalancing cost levels.
"""
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
W_CAP = 0.20

# Import weight functions from v4_portfolios
from v4_portfolios import (WEIGHT_FUNCS, net_returns_series, calc_sharpe,
                           calc_maxdd, clip_weights)


def compute_turnover_and_sharpe(ticker_net_rets, method, lookback_iv=252,
                                 rebal_comms=[0.0, 0.0005, 0.001, 0.0015]):
    """
    Compute turnover statistics and Sharpe at various rebalancing cost levels.
    Returns: dict with turnover stats and Sharpe per cost level.
    """
    tickers = list(ticker_net_rets.columns)
    n = len(tickers)
    dates = ticker_net_rets.index
    ret_arr = ticker_net_rets.values

    months = pd.Series(dates).dt.to_period("M")
    month_starts = []
    prev_m = None
    for i, m in enumerate(months):
        if m != prev_m:
            month_starts.append(i)
            prev_m = m

    wf = WEIGHT_FUNCS[method]
    turnovers = []
    w_actual = np.zeros(n)

    # Pre-compute target weights per month
    target_weights = []
    for k, start_idx in enumerate(month_starts):
        if start_idx == 0:
            w_target = np.ones(n) / n
        else:
            hist = ret_arr[:start_idx]
            if method == "InvVol":
                hist = hist[-lookback_iv:]
            w_target = wf(hist, W_CAP)
        target_weights.append(w_target)

    # Compute turnovers
    w_actual = np.zeros(n)
    for k, start_idx in enumerate(month_starts):
        end_idx = month_starts[k + 1] if k + 1 < len(month_starts) else len(dates)
        w_target = target_weights[k]

        turnover = np.sum(np.abs(w_target - w_actual))
        turnovers.append(turnover)

        # Track drift
        cum_growth = np.ones(n)
        for t in range(start_idx, end_idx):
            cum_growth *= (1 + ret_arr[t])
        w_drifted = w_target * cum_growth
        total = w_drifted.sum()
        if total > 1e-15:
            w_actual = w_drifted / total
        else:
            w_actual = w_target.copy()

    # Skip first month (initial buy-in, same for all methods)
    monthly_turnovers = turnovers[1:]  # exclude initial buy-in
    avg_monthly_turnover = np.mean(monthly_turnovers) if monthly_turnovers else 0
    total_turnover = sum(turnovers)
    n_months = len(month_starts)
    ann_turnover = avg_monthly_turnover * 12 * 100  # annualized, %

    # Compute Sharpe at various rebal_comm levels
    sharpe_by_cost = {}
    for rc in rebal_comms:
        # Rebuild portfolio returns with this cost
        port_rets = np.zeros(len(dates))
        w_actual2 = np.zeros(n)
        for k, start_idx in enumerate(month_starts):
            end_idx = month_starts[k + 1] if k + 1 < len(month_starts) else len(dates)
            w_target = target_weights[k]

            rebal_cost = np.sum(np.abs(w_target - w_actual2)) * rc

            for t in range(start_idx, end_idx):
                port_rets[t] = np.sum(ret_arr[t] * w_target)
            port_rets[start_idx] -= rebal_cost

            cum_growth = np.ones(n)
            for t in range(start_idx, end_idx):
                cum_growth *= (1 + ret_arr[t])
            w_drifted = w_target * cum_growth
            total = w_drifted.sum()
            if total > 1e-15:
                w_actual2 = w_drifted / total
            else:
                w_actual2 = w_target.copy()

        sharpe_by_cost[rc] = round(calc_sharpe(port_rets, 252), 4)

    return {
        "avg_monthly_turnover": avg_monthly_turnover,
        "ann_turnover_pct": ann_turnover,
        "total_turnover": total_turnover,
        "n_rebalances": n_months,
        "sharpe_by_cost": sharpe_by_cost,
    }


def main():
    print("Loading data...")
    pos_all = pd.read_parquet(OUT_DIR / "daily_positions.parquet")

    for tf_label, tf_filter, comm, bpy, lookback_iv in [
        ("DAILY", "daily", 0.0005, 252, 252),
        ("HOURLY", "hourly", 0.0004, 2268, 2268),
    ]:
        pos = pos_all[(pos_all["tf"] == tf_filter) &
                      (pos_all["test_year"].isin(BCD_YEARS))].copy()
        if len(pos) == 0:
            continue

        print(f"\n{'='*80}")
        print(f"  TURNOVER ANALYSIS — {tf_label}")
        print(f"{'='*80}")

        tickers = sorted(pos["ticker"].unique())

        # Pre-compute net returns
        net_cache = {}
        for (strat, appr, tkr), grp in pos.groupby(["strategy", "approach", "ticker"]):
            grp = grp.sort_values("date")
            pos_s = grp["position"].reset_index(drop=True)
            gross_s = grp["daily_gross_return"].reset_index(drop=True)
            dates = grp["date"].reset_index(drop=True)
            net_s = net_returns_series(pos_s, gross_s, comm)
            net_cache[(strat, appr, tkr)] = pd.Series(net_s.values, index=dates.values)

        rebal_costs = [0.0, comm, 2 * comm, 3 * comm]
        cost_labels = ["0 (no rebal)", f"{comm*100:.2f}% (comm)",
                       f"{2*comm*100:.2f}% (comm+spread)", f"{3*comm*100:.2f}% (high)"]

        # Collect per method statistics
        METHODS = ["EW", "InvVol", "MaxSharpe", "MinVar"]
        method_turnovers = {m: [] for m in METHODS}
        method_sharpes = {m: {rc: [] for rc in rebal_costs} for m in METHODS}

        for strat in STRATEGIES:
            for appr in APPROACHES:
                ticker_rets = {}
                for tkr in tickers:
                    key = (strat, appr, tkr)
                    if key in net_cache:
                        ticker_rets[tkr] = net_cache[key]

                if len(ticker_rets) < 2:
                    continue

                ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)

                for method in METHODS:
                    result = compute_turnover_and_sharpe(
                        ret_df, method, lookback_iv=lookback_iv,
                        rebal_comms=rebal_costs
                    )
                    # Override Sharpe with correct BPY for hourly
                    if bpy != 252:
                        # Recompute with correct BPY
                        sharpes_corrected = {}
                        for rc in rebal_costs:
                            # Rebuild for correct BPY
                            from v4_portfolios import build_portfolio
                            port_ret = build_portfolio(ret_df, method,
                                                       lookback_iv=lookback_iv,
                                                       rebal_comm=rc)
                            sharpes_corrected[rc] = calc_sharpe(port_ret.values, bpy)
                        result["sharpe_by_cost"] = sharpes_corrected

                    method_turnovers[method].append(result["ann_turnover_pct"])
                    for rc in rebal_costs:
                        method_sharpes[method][rc].append(result["sharpe_by_cost"][rc])

        # Print turnover summary
        print(f"\n### Среднегодовой turnover по методам (%, средний по 24 комбинациям)")
        print(f"\n| Метод | Ср. годовой turnover% | Медиана | Мин | Макс |")
        print(f"|-------|:--------------------:|:-------:|:---:|:----:|")
        for m in METHODS:
            tv = method_turnovers[m]
            if len(tv) > 0:
                print(f"| {m} | {np.mean(tv):.1f}% | {np.median(tv):.1f}% | "
                      f"{np.min(tv):.1f}% | {np.max(tv):.1f}% |")

        # Print Sharpe at various cost levels
        print(f"\n### Средний Sharpe при различных уровнях rebalancing cost")
        print()
        header = "| Метод |"
        for lab in cost_labels:
            header += f" {lab} |"
        print(header)
        sep = "|-------|"
        for _ in cost_labels:
            sep += "---:|"
        print(sep)
        for m in METHODS:
            row = f"| {m} |"
            for rc in rebal_costs:
                avg_sh = np.mean(method_sharpes[m][rc])
                row += f" {avg_sh:.2f} |"
            print(row)

        # Print EW advantage at each cost level
        print(f"\n### Преимущество EW над MinVar при различных rebal costs")
        print()
        print(f"| Уровень | EW Sharpe | MinVar Sharpe | Δ | Δ% |")
        print(f"|---------|:---------:|:------------:|---:|---:|")
        for rc, lab in zip(rebal_costs, cost_labels):
            ew_sh = np.mean(method_sharpes["EW"][rc])
            mv_sh = np.mean(method_sharpes["MinVar"][rc])
            delta = ew_sh - mv_sh
            pct = delta / mv_sh * 100 if mv_sh > 0 else 0
            print(f"| {lab} | {ew_sh:.3f} | {mv_sh:.3f} | "
                  f"{'+' if delta >= 0 else ''}{delta:.3f} | "
                  f"{'+' if pct >= 0 else ''}{pct:.1f}% |")

        # Win counts at each cost level
        print(f"\n### Победы EW vs MinVar при различных rebal costs")
        print()
        print(f"| Уровень | EW побед | MinVar побед |")
        print(f"|---------|:-------:|:-----------:|")
        for rc, lab in zip(rebal_costs, cost_labels):
            ew_wins = 0
            mv_wins = 0
            for i in range(len(method_sharpes["EW"][rc])):
                if method_sharpes["EW"][rc][i] > method_sharpes["MinVar"][rc][i]:
                    ew_wins += 1
                else:
                    mv_wins += 1
            print(f"| {lab} | {ew_wins}/24 | {mv_wins}/24 |")


if __name__ == "__main__":
    main()
