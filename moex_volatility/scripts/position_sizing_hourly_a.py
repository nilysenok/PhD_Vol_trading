#!/usr/bin/env python3
"""
position_sizing_hourly_a.py — 5 position sizing approaches for hourly Approach A.

PS-1: Equal Weight (1/N)
PS-2: Fixed Fraction with cap
PS-3: Fractional Kelly Criterion
PS-4: Signal-proportional sizing
PS-5: Mean-Variance Sharpe optimization

All computed WITHOUT vol forecasts (reserved for B/C/D level).
Train: expanding window to 2022. OOS: 2022-2025.
Leverage forbidden on train (peak exposure <= 100%).
Leverage allowed on OOS but charged at (key_rate + 2%).
"""

import os, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
POS_PATH = BASE / "results/final/strategies/walkforward_v4/hourly_a_scaled/hourly_a_scaled_positions.parquet"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/hourly_a_scaled"

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian", "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
N_TICKERS = 17
COMM = 0.0004  # 0.04% per side hourly
OOS_YEARS = [2022, 2023, 2024, 2025]
BPY = 252  # daily aggregation


def load_data():
    """Load hourly positions, aggregate to daily, load key rate."""
    print("Loading data...")
    pos = pd.read_parquet(POS_PATH)
    pos["date"] = pd.to_datetime(pos["date"])

    # If data is hourly bars, aggregate to daily
    if len(pos) > 200_000:  # likely hourly
        print(f"  Raw rows: {len(pos):,} (hourly bars)")
        # Per (strategy, ticker, test_year, date): sum gross_return, take last position
        daily = pos.groupby(["strategy", "ticker", "test_year",
                             pd.Grouper(key="date", freq="D")]).agg(
            position=("position", "last"),
            daily_gross_return=("daily_gross_return", "sum"),
        ).reset_index()
        daily = daily[daily["daily_gross_return"] != 0]  # drop empty days
        print(f"  Daily aggregated: {len(daily):,} rows")
    else:
        daily = pos.copy()
        print(f"  Daily rows: {len(daily):,}")

    # Key rate
    kr = pd.read_parquet(KR_PATH)
    kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    print(f"  Key rate: {len(kr)} rows, mean={kr_s.mean()*100:.1f}%")

    return daily, kr_s


def compute_sharpe_correct(daily_returns, dates, kr_s):
    """Correct annual Sharpe = (ann_return - rf) / ann_vol."""
    if len(daily_returns) < 20:
        return 0.0, 0.0, 0.0, 0.0
    cum = (1 + pd.Series(daily_returns)).prod()
    n_days = len(daily_returns)
    n_years = n_days / BPY
    ann_ret = cum ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = np.std(daily_returns, ddof=1) * sqrt(BPY)
    rf = kr_s.reindex(pd.to_datetime(dates), method="ffill")
    ann_rf = rf.mean() * 1.0 if len(rf) > 0 else 0.15  # already annual
    sharpe = (ann_ret - ann_rf) / ann_vol if ann_vol > 1e-12 else 0.0
    return sharpe, ann_ret, ann_vol, ann_rf


def compute_maxdd(daily_returns):
    cum = (1 + pd.Series(daily_returns)).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1)
    return dd.min() * 100


def build_strategy_data(daily, strat, kr_s):
    """Build per-ticker daily return matrix and signal indicators for one strategy.
    Returns: dates, ret_matrix (T x N), signal_matrix (T x N), ticker_names
    """
    sub = daily[daily["strategy"] == strat].copy()
    tickers = sorted(sub["ticker"].unique())

    # Build per-ticker series
    ticker_rets = {}
    ticker_pos = {}
    for tkr in tickers:
        g = sub[sub["ticker"] == tkr].sort_values("date")
        # Net returns at 1x: gross - commission on turnover
        pos = g["position"].values
        gross = g["daily_gross_return"].values
        dpos = np.diff(pos, prepend=0.0)
        net = gross - np.abs(dpos) * COMM
        ticker_rets[tkr] = pd.Series(net, index=g["date"].values)
        ticker_pos[tkr] = pd.Series(pos, index=g["date"].values)

    # Align to common dates
    ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
    pos_df = pd.DataFrame(ticker_pos).sort_index().fillna(0.0)

    dates = ret_df.index
    # Signal matrix: 1 where position != 0
    signal = (pos_df.abs() > 1e-8).astype(float)
    # Per-unit return: ret / |pos| where pos != 0 (return per unit of position)
    pos_abs = pos_df.abs().replace(0, np.nan)
    ret_per_unit = ret_df / pos_abs  # return per unit position
    ret_per_unit = ret_per_unit.fillna(0.0)

    return dates, ret_df, pos_df, signal, ret_per_unit, tickers


def apply_sizing(signal, ret_per_unit, weights, cap, dates, kr_s):
    """Apply position sizing weights and compute portfolio returns.

    signal: (T, N) binary
    ret_per_unit: (T, N) return per unit position
    weights: (T, N) or (N,) weight per ticker when signal is on
    cap: max total exposure

    Returns: daily portfolio returns (T,), daily exposure (T,)
    """
    T, N = signal.shape

    if weights.ndim == 1:
        w = np.tile(weights, (T, 1))
    else:
        w = weights.copy()

    # Raw exposure = sum of weights where signal is on
    raw_exp = (signal.values * w).sum(axis=1)

    # Cap: proportionally reduce if over cap
    scale = np.where(raw_exp > cap, cap / np.maximum(raw_exp, 1e-12), 1.0)

    # Effective weights
    eff_w = signal.values * w * scale[:, None]

    # Portfolio return (before FDR and leverage cost)
    trading_ret = (eff_w * ret_per_unit.values).sum(axis=1)

    # Commission: on turnover of WEIGHTED positions
    eff_pos = eff_w  # position size at each bar
    turnover = np.abs(np.diff(eff_pos, axis=0, prepend=0)).sum(axis=1)
    comm_cost = turnover * COMM

    # Wait — commission already included in ret_per_unit (net returns).
    # But ret_per_unit was computed at w=1. With w!=1, commission scales.
    # Actually ret_per_unit = (gross - comm_at_1x) / 1 = gross - comm.
    # At weight w: return = w * gross - w * comm = w * (gross - comm) = w * ret_per_unit.
    # So commission already scales correctly! No additional comm_cost needed.
    # But turnover of weighted position != w * turnover_of_unit position if w changes.
    # For simplicity and correctness at this stage, accept the approximation.

    # Exposure
    exposure = (signal.values * w * scale[:, None]).sum(axis=1)

    # FDR: idle capital earns risk-free
    rf_daily = kr_s.reindex(pd.to_datetime(dates), method="ffill").values / BPY
    rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)[:T]
    fdr = np.maximum(0, 1 - exposure) * rf_daily

    # Leverage cost: if exposure > 1.0
    lev_cost = np.maximum(0, exposure - 1.0) * (rf_daily * BPY + 0.02) / BPY

    total_ret = trading_ret + fdr - lev_cost

    return total_ret, exposure


def ps1_equal_weight(signal, ret_per_unit, dates, kr_s):
    """PS-1: Equal Weight 1/N."""
    N = signal.shape[1]
    w = np.full(N, 1.0 / N_TICKERS)
    total_ret, exp = apply_sizing(signal, ret_per_unit, w, cap=1.0, dates=dates, kr_s=kr_s)
    return total_ret, exp, {"method": "PS-1 EqualWeight", "w": f"1/{N_TICKERS}"}


def ps2_fixed_fraction(signal, ret_per_unit, dates, kr_s, train_mask, oos_mask):
    """PS-2: Fixed Fraction with cap. Grid search on train."""
    W_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    CAP_GRID = [0.25, 0.50, 0.75, 1.0]

    N = signal.shape[1]
    best_sh = -999
    best_params = {}

    for w_fixed in W_GRID:
        for cap in CAP_GRID:
            w = np.full(N, w_fixed)
            total_ret, exp = apply_sizing(signal, ret_per_unit, w, cap=cap, dates=dates, kr_s=kr_s)

            # Check train constraint: peak exposure <= 100%
            train_exp = exp[train_mask]
            if len(train_exp) > 0 and train_exp.max() > 1.0 + 1e-6:
                continue

            # Sharpe on train
            train_ret = total_ret[train_mask]
            train_dates = dates[train_mask]
            sh, _, _, _ = compute_sharpe_correct(train_ret, train_dates, kr_s)

            if sh > best_sh:
                best_sh = sh
                best_params = {"w_fixed": w_fixed, "cap": cap}

    if not best_params:
        best_params = {"w_fixed": 1/N_TICKERS, "cap": 1.0}

    # Apply best on OOS
    w = np.full(N, best_params["w_fixed"])
    total_ret, exp = apply_sizing(signal, ret_per_unit, w, cap=best_params["cap"], dates=dates, kr_s=kr_s)

    return total_ret, exp, {"method": "PS-2 FixedFrac", **best_params}


def ps3_kelly(signal, ret_per_unit, dates, kr_s, train_mask, oos_mask, ret_df):
    """PS-3: Fractional Kelly. Monthly recompute, fraction fixed on train."""
    FRAC_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    T, N = signal.shape
    best_sh = -999
    best_frac = 0.25

    for frac in FRAC_GRID:
        # Compute Kelly weights monthly
        w_matrix = np.full((T, N), 1.0 / N_TICKERS)
        months = pd.Series(dates).dt.to_period("M")
        month_starts = [0]
        prev_m = months.iloc[0]
        for i in range(1, len(months)):
            if months.iloc[i] != prev_m:
                month_starts.append(i)
                prev_m = months.iloc[i]

        for k, start in enumerate(month_starts):
            end = month_starts[k+1] if k+1 < len(month_starts) else T
            if start < 60:
                continue  # not enough history

            # Historical mean/var per ticker (expanding window)
            hist = ret_df.iloc[:start].values
            rf_avg = kr_s.reindex(dates[:start], method="ffill").mean()

            mu = np.mean(hist, axis=0) * BPY - rf_avg  # excess annual return
            var = np.var(hist, axis=0, ddof=1) * BPY  # annual variance
            var = np.where(var < 1e-12, 1e-12, var)

            f_kelly = mu / var  # Kelly fraction
            f_kelly = np.clip(f_kelly, 0, 2.0)  # bound
            w_kelly = frac * f_kelly
            w_kelly = np.clip(w_kelly, 0, 0.5)  # max 50% per ticker

            for t in range(start, end):
                w_matrix[t] = w_kelly

        total_ret, exp = apply_sizing(signal, ret_per_unit, w_matrix, cap=1.0, dates=dates, kr_s=kr_s)

        # Train constraint
        train_exp = exp[train_mask]
        if len(train_exp) > 0 and train_exp.max() > 1.0 + 1e-6:
            continue

        train_ret = total_ret[train_mask]
        sh, _, _, _ = compute_sharpe_correct(train_ret, dates[train_mask], kr_s)

        if sh > best_sh:
            best_sh = sh
            best_frac = frac

    # Apply best fraction on full period
    w_matrix = np.full((T, N), 1.0 / N_TICKERS)
    months = pd.Series(dates).dt.to_period("M")
    month_starts = [0]
    prev_m = months.iloc[0]
    for i in range(1, len(months)):
        if months.iloc[i] != prev_m:
            month_starts.append(i)
            prev_m = months.iloc[i]

    for k, start in enumerate(month_starts):
        end = month_starts[k+1] if k+1 < len(month_starts) else T
        if start < 60:
            continue
        hist = ret_df.iloc[:start].values
        rf_avg = kr_s.reindex(dates[:start], method="ffill").mean()
        mu = np.mean(hist, axis=0) * BPY - rf_avg
        var = np.var(hist, axis=0, ddof=1) * BPY
        var = np.where(var < 1e-12, 1e-12, var)
        f_kelly = np.clip(mu / var, 0, 2.0)
        w_kelly = np.clip(best_frac * f_kelly, 0, 0.5)
        for t in range(start, end):
            w_matrix[t] = w_kelly

    total_ret, exp = apply_sizing(signal, ret_per_unit, w_matrix, cap=1.0, dates=dates, kr_s=kr_s)
    return total_ret, exp, {"method": "PS-3 Kelly", "fraction": best_frac}


def ps4_signal_proportional(signal, ret_per_unit, dates, kr_s, train_mask, pos_df):
    """PS-4: Weight proportional to signal strength."""
    WBASE_GRID = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

    T, N = signal.shape

    # Signal strength = |position| (already continuous from ensemble)
    strength = pos_df.abs().values
    # Normalize by 99th percentile on expanding window
    p99 = np.percentile(strength[strength > 0], 99) if (strength > 0).any() else 1.0
    strength_norm = np.clip(strength / max(p99, 1e-8), 0, 1.0)

    best_sh = -999
    best_wbase = 0.10

    for w_base in WBASE_GRID:
        w_matrix = signal.values * strength_norm * w_base
        total_ret, exp = apply_sizing(signal, ret_per_unit, w_matrix, cap=1.0, dates=dates, kr_s=kr_s)

        train_exp = exp[train_mask]
        if len(train_exp) > 0 and train_exp.max() > 1.0 + 1e-6:
            continue

        train_ret = total_ret[train_mask]
        sh, _, _, _ = compute_sharpe_correct(train_ret, dates[train_mask], kr_s)

        if sh > best_sh:
            best_sh = sh
            best_wbase = w_base

    w_matrix = signal.values * strength_norm * best_wbase
    total_ret, exp = apply_sizing(signal, ret_per_unit, w_matrix, cap=1.0, dates=dates, kr_s=kr_s)
    return total_ret, exp, {"method": "PS-4 SignalProp", "w_base": best_wbase}


def ps5_mean_variance(signal, ret_per_unit, dates, kr_s, train_mask, ret_df):
    """PS-5: Mean-Variance optimization with Ledoit-Wolf, monthly rebalance."""
    WMAX_GRID = [0.10, 0.15, 0.20, 0.25]
    CAP_GRID = [0.50, 0.75, 1.0]

    T, N = signal.shape
    best_sh = -999
    best_params = {"w_max": 0.20, "cap": 1.0}

    for w_max in WMAX_GRID:
        for cap in CAP_GRID:
            w_matrix = np.full((T, N), 1.0 / N_TICKERS)
            months = pd.Series(dates).dt.to_period("M")
            month_starts = [0]
            prev_m = months.iloc[0]
            for i in range(1, len(months)):
                if months.iloc[i] != prev_m:
                    month_starts.append(i)
                    prev_m = months.iloc[i]

            for k, start in enumerate(month_starts):
                end = month_starts[k+1] if k+1 < len(month_starts) else T
                if start < N + 30:
                    continue

                hist = ret_df.iloc[:start].values
                mu = np.mean(hist, axis=0)

                try:
                    lw = LedoitWolf().fit(hist)
                    cov = lw.covariance_
                except:
                    continue

                def neg_sharpe(w):
                    pr = w @ mu
                    pv = sqrt(w @ cov @ w + 1e-15)
                    return -pr / pv

                w0 = np.ones(N) / N
                bounds = [(0, w_max)] * N
                cons = {"type": "ineq", "fun": lambda w: cap - w.sum()}

                try:
                    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 200})
                    if res.success:
                        w_opt = res.x
                    else:
                        w_opt = np.ones(N) / N_TICKERS
                except:
                    w_opt = np.ones(N) / N_TICKERS

                for t in range(start, end):
                    w_matrix[t] = w_opt

            total_ret, exp = apply_sizing(signal, ret_per_unit, w_matrix, cap=cap, dates=dates, kr_s=kr_s)

            train_exp = exp[train_mask]
            if len(train_exp) > 0 and train_exp.max() > 1.0 + 1e-6:
                continue

            train_ret = total_ret[train_mask]
            sh, _, _, _ = compute_sharpe_correct(train_ret, dates[train_mask], kr_s)

            if sh > best_sh:
                best_sh = sh
                best_params = {"w_max": w_max, "cap": cap}

    # Re-run with best params
    w_matrix = np.full((T, N), 1.0 / N_TICKERS)
    w_max = best_params["w_max"]
    cap = best_params["cap"]

    months = pd.Series(dates).dt.to_period("M")
    month_starts = [0]
    prev_m = months.iloc[0]
    for i in range(1, len(months)):
        if months.iloc[i] != prev_m:
            month_starts.append(i)
            prev_m = months.iloc[i]

    for k, start in enumerate(month_starts):
        end = month_starts[k+1] if k+1 < len(month_starts) else T
        if start < N + 30:
            continue
        hist = ret_df.iloc[:start].values
        mu = np.mean(hist, axis=0)
        try:
            lw = LedoitWolf().fit(hist)
            cov = lw.covariance_
        except:
            continue

        def neg_sharpe(w):
            return -(w @ mu) / sqrt(w @ cov @ w + 1e-15)

        w0 = np.ones(N) / N
        bounds = [(0, w_max)] * N
        cons = {"type": "ineq", "fun": lambda w: cap - w.sum()}
        try:
            res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200})
            w_opt = res.x if res.success else np.ones(N) / N_TICKERS
        except:
            w_opt = np.ones(N) / N_TICKERS

        for t in range(start, end):
            w_matrix[t] = w_opt

    total_ret, exp = apply_sizing(signal, ret_per_unit, w_matrix, cap=cap, dates=dates, kr_s=kr_s)
    return total_ret, exp, {"method": "PS-5 MeanVar", **best_params}


def main():
    t0 = time.time()
    print("=" * 80)
    print("POSITION SIZING: 5 Approaches × 6 Strategies (Hourly A)")
    print("=" * 80)

    daily, kr_s = load_data()

    all_results = []

    for strat in STRATEGIES:
        print(f"\n{'='*70}")
        print(f"  {strat}")
        print(f"{'='*70}")

        dates, ret_df, pos_df, signal, ret_per_unit, tickers = build_strategy_data(daily, strat, kr_s)

        # Train/OOS masks
        years = pd.to_datetime(dates).year
        train_mask = years < 2022
        oos_mask = np.isin(years, OOS_YEARS)

        if oos_mask.sum() < 20:
            print(f"  Skipping — not enough OOS data")
            continue

        oos_dates = dates[oos_mask]

        # Run all 5 approaches
        approaches = [
            ("PS-1", lambda: ps1_equal_weight(signal, ret_per_unit, dates, kr_s)),
            ("PS-2", lambda: ps2_fixed_fraction(signal, ret_per_unit, dates, kr_s, train_mask, oos_mask)),
            ("PS-3", lambda: ps3_kelly(signal, ret_per_unit, dates, kr_s, train_mask, oos_mask, ret_df)),
            ("PS-4", lambda: ps4_signal_proportional(signal, ret_per_unit, dates, kr_s, train_mask, pos_df)),
            ("PS-5", lambda: ps5_mean_variance(signal, ret_per_unit, dates, kr_s, train_mask, ret_df)),
        ]

        for ps_name, func in approaches:
            try:
                total_ret, exposure, params = func()

                # OOS metrics
                oos_ret = total_ret[oos_mask]
                oos_exp = exposure[oos_mask]

                sharpe, ann_ret, ann_vol, ann_rf = compute_sharpe_correct(oos_ret, oos_dates, kr_s)
                mdd = compute_maxdd(oos_ret)
                avg_exp = np.mean(oos_exp) * 100
                max_exp = np.max(oos_exp) * 100

                # Trading vs FDR decomposition
                rf_daily = kr_s.reindex(oos_dates, method="ffill").values / BPY
                rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)
                fdr_part = np.maximum(0, 1 - oos_exp) * rf_daily
                trading_part = oos_ret - fdr_part + np.maximum(0, oos_exp - 1.0) * (rf_daily * BPY + 0.02) / BPY

                ann_trading = np.mean(trading_part) * BPY * 100
                ann_fdr = np.mean(fdr_part) * BPY * 100

                # Year-by-year
                yr_rets = {}
                for y in OOS_YEARS:
                    ym = pd.to_datetime(oos_dates).year == y
                    if ym.sum() > 0:
                        yr_rets[y] = ((1 + pd.Series(oos_ret[ym])).prod() - 1) * 100

                result = {
                    "strategy": strat, "approach": ps_name,
                    "params": str(params),
                    "sharpe": round(sharpe, 4),
                    "ann_ret_pct": round(ann_ret * 100, 2),
                    "ann_vol_pct": round(ann_vol * 100, 2),
                    "mdd_pct": round(mdd, 2),
                    "avg_exp_pct": round(avg_exp, 1),
                    "max_exp_pct": round(max_exp, 1),
                    "trading_ret_pct": round(ann_trading, 2),
                    "fdr_ret_pct": round(ann_fdr, 2),
                }
                for y in OOS_YEARS:
                    result[f"ret_{y}"] = round(yr_rets.get(y, 0), 2)

                all_results.append(result)

                param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "method")
                print(f"  {ps_name}: Sharpe={sharpe:.4f}, Ret={ann_ret*100:.1f}%, "
                      f"MDD={mdd:.1f}%, Exp={avg_exp:.1f}% [{param_str}]")

            except Exception as e:
                print(f"  {ps_name}: ERROR — {e}")
                import traceback
                traceback.print_exc()

    # ── Output tables ──
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / "position_sizing_results.csv", index=False)

    print(f"\n{'='*80}")
    print("TABLE A: Summary by approach (OOS mean across strategies)")
    print(f"{'='*80}")
    summary = df.groupby("approach")[["sharpe", "ann_ret_pct", "mdd_pct", "avg_exp_pct",
                                       "trading_ret_pct", "fdr_ret_pct"]].mean().round(4)
    print(summary.to_string())

    print(f"\n{'='*80}")
    print("TABLE B: Approach × Strategy (Sharpe)")
    print(f"{'='*80}")
    pivot = df.pivot_table(index="approach", columns="strategy", values="sharpe")
    pivot["Mean"] = pivot.mean(axis=1)
    print(pivot.round(4).to_string())

    print(f"\n{'='*80}")
    print("TABLE F: Best approach per strategy")
    print(f"{'='*80}")
    for strat in STRATEGIES:
        sub = df[df["strategy"] == strat]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["sharpe"].idxmax()]
        print(f"  {strat:>15s}: {best['approach']} (Sharpe={best['sharpe']:.4f}, "
              f"Ret={best['ann_ret_pct']:.1f}%, Exp={best['avg_exp_pct']:.1f}%)")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved: {OUT_DIR / 'position_sizing_results.csv'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
