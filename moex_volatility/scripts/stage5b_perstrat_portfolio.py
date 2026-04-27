#!/usr/bin/env python3
"""
stage5b_perstrat_portfolio.py — Per-strategy portfolios with 4 weighting schemes.

For each of 6 strategies (S1-S6) × 4 approaches (A/B/C/D) × 2 commissions (0.0005/0.0006):
  - Walk-forward weight estimation: train on data before test_year, apply on test_year
  - 4 weighting schemes: EW, InvVol, RiskParity, MaxSharpe
  - 3 Sharpe metrics: trading, excess (real RF), excess (@9%)
  - Save portfolio daily returns for bootstrap

Output:
  msp/perstrat_results.csv
  msp/perstrat_daily_returns/{strategy}_{approach}_{weight}_{comm_label}.csv
"""
import os, sys, warnings, datetime
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy.optimize import minimize

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
STRAT_DIR = BASE / "results/final/strategies"
MSP_DIR = STRAT_DIR / "msp"
MSP_DIR.mkdir(parents=True, exist_ok=True)
PERSTRAT_RET_DIR = MSP_DIR / "perstrat_daily_returns"
PERSTRAT_RET_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = STRAT_DIR / "MASTER_PROGRESS_LOG.txt"

WF_DIR = STRAT_DIR / "walkforward"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")

BPY = 252
RF_MEAN_HIST = 0.09
TICKERS = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
           "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"]
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
APPROACHES = ["A", "B", "C", "D"]
OOS_YEARS = [2022, 2023, 2024, 2025]
WEIGHT_SCHEMES = ["W1_EW", "W2_InvVol", "W3_RiskParity", "W4_MaxSharpe"]

COMM_DAILY = {"inst": 0.0005, "std": 0.0006}


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage5b_perstrat] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════
# Load key rate
# ══════════════════════════════════════════════════════════════════════
def load_key_rate():
    if KR_PATH.exists():
        kr = pd.read_parquet(KR_PATH)
        kr["date"] = pd.to_datetime(kr["date"])
        return kr.set_index("date")["key_rate"].sort_index()
    log("WARNING: key rate file not found, using 0.16 flat")
    dates = pd.bdate_range("2019-01-01", "2026-06-01")
    return pd.Series(0.16, index=dates, name="key_rate")


# ══════════════════════════════════════════════════════════════════════
# Load positions (daily only)
# ══════════════════════════════════════════════════════════════════════
def load_positions():
    """Load daily WF positions for all approaches.
    Returns DataFrame: date, ticker, strategy, year, position, daily_gross_return, approach.
    """
    frames = []
    for app in APPROACHES:
        fp = WF_DIR / f"wf_positions_{app}.parquet"
        if fp.exists():
            tmp = pd.read_parquet(fp)
            tmp["approach"] = app
            tmp["date"] = pd.to_datetime(tmp["date"])
            frames.append(tmp)
            log(f"  Loaded {fp.name}: {len(tmp)} rows")
        else:
            log(f"  WARNING: {fp} not found")

    if not frames:
        raise FileNotFoundError("No position files found. Stage 3/4 must run first.")

    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# Weight computation functions
# ══════════════════════════════════════════════════════════════════════
def compute_equal_weights(n):
    """W1: Equal weight — 1/N."""
    return np.ones(n) / n


def compute_inv_vol_weights(vols):
    """W2: Inverse volatility — 1/vol_i / sum(1/vol_i)."""
    vols = np.maximum(vols, 1e-10)
    inv = 1.0 / vols
    return inv / inv.sum()


def compute_risk_parity_weights(cov, n_iter=100):
    """W3: Risk parity — equal risk contribution via iterative algorithm."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(n_iter):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return np.ones(n) / n
        # Marginal risk contribution
        mrc = cov @ w / port_vol
        # Risk contribution
        rc = w * mrc
        # Target: equal risk = port_vol / n
        target = port_vol / n
        # Update: scale weights proportional to target / rc
        rc = np.maximum(rc, 1e-12)
        w_new = w * (target / rc)
        w_new = np.maximum(w_new, 0.0)
        s = w_new.sum()
        if s > 1e-12:
            w_new = w_new / s
        else:
            w_new = np.ones(n) / n
        w = w_new
    return w


def compute_max_sharpe_weights(mu, cov):
    """W4: Max Sharpe — Markowitz mean-variance optimal (long only, sum=1)."""
    n = len(mu)
    if n == 0:
        return np.array([])

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        return -port_ret / port_vol

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        w = result.x
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 1e-12:
            w = w / s
        else:
            w = np.ones(n) / n
        return w
    else:
        # Fallback to equal weight
        return np.ones(n) / n


def compute_all_weights(train_returns_matrix, tickers_active):
    """
    Compute 4 weight vectors from training data.

    train_returns_matrix: DataFrame with columns = tickers, rows = dates, values = net daily returns
    tickers_active: list of tickers with data

    Returns dict: {scheme_name: np.array of weights aligned to tickers_active}
    """
    n = len(tickers_active)
    if n == 0:
        return {ws: np.array([]) for ws in WEIGHT_SCHEMES}

    mu = train_returns_matrix.mean().values
    cov = train_returns_matrix.cov().values
    vols = train_returns_matrix.std().values

    # Regularize covariance
    cov = cov + np.eye(n) * 1e-10

    weights = {}
    weights["W1_EW"] = compute_equal_weights(n)
    weights["W2_InvVol"] = compute_inv_vol_weights(vols)
    weights["W3_RiskParity"] = compute_risk_parity_weights(cov)
    weights["W4_MaxSharpe"] = compute_max_sharpe_weights(mu, cov)

    return weights


# ══════════════════════════════════════════════════════════════════════
# Calendar-days helper
# ══════════════════════════════════════════════════════════════════════
def compute_cal_days(dates_index):
    """Return array of calendar days between consecutive trading days."""
    cal_days = np.ones(len(dates_index))
    for i in range(1, len(dates_index)):
        cal_days[i] = (dates_index[i] - dates_index[i - 1]).days
    return cal_days


# ══════════════════════════════════════════════════════════════════════
# Build per-strategy weighted portfolio (walk-forward)
# ══════════════════════════════════════════════════════════════════════
def build_weighted_portfolio(strat_data, kr_series, comm, weight_scheme):
    """
    Build a weighted portfolio for one strategy using walk-forward weight estimation.

    strat_data: DataFrame with date, ticker, position, daily_gross_return
                (already filtered to one strategy + one approach)
    kr_series: key rate series
    comm: commission per unit
    weight_scheme: one of WEIGHT_SCHEMES

    Returns: (portfolio_df indexed by date, n_tickers_used)
    """
    # Step 1: compute per-ticker net returns
    ticker_data = {}
    for tick in TICKERS:
        sub = strat_data[strat_data["ticker"] == tick].sort_values("date").reset_index(drop=True)
        if len(sub) == 0:
            continue
        pos = sub["position"].values
        gross = sub["daily_gross_return"].values
        dp = np.diff(pos, prepend=0.0)
        net = gross - np.abs(dp) * comm

        df_tick = pd.DataFrame({
            "date": sub["date"].values,
            "net_ret": net,
            "abs_pos": np.abs(pos),
            "position": pos,
        }).set_index("date")
        ticker_data[tick] = df_tick

    if not ticker_data:
        return None, 0

    # Step 2: build a wide matrix of net returns (date × ticker)
    all_dates = sorted(set().union(*(df.index for df in ticker_data.values())))
    ret_wide = pd.DataFrame(index=all_dates)
    pos_wide = pd.DataFrame(index=all_dates)
    abspos_wide = pd.DataFrame(index=all_dates)

    for tick, df_tick in ticker_data.items():
        ret_wide[tick] = df_tick["net_ret"]
        pos_wide[tick] = df_tick["position"]
        abspos_wide[tick] = df_tick["abs_pos"]

    # Fill NaN with 0 (ticker not trading on that date → zero return, zero position)
    ret_wide = ret_wide.fillna(0.0)
    pos_wide = pos_wide.fillna(0.0)
    abspos_wide = abspos_wide.fillna(0.0)

    active_tickers = list(ticker_data.keys())
    n_tickers = len(active_tickers)

    # Step 3: Walk-forward weight estimation per test year
    port_chunks = []

    for test_year in OOS_YEARS:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + 1}-01-01")

        # Test period mask
        test_mask = (ret_wide.index >= test_start) & (ret_wide.index < test_end)
        test_dates = ret_wide.index[test_mask]
        if len(test_dates) == 0:
            continue

        # Train period: all data before test_start
        train_mask = ret_wide.index < test_start
        train_data = ret_wide.loc[train_mask, active_tickers]

        # Determine which tickers have data in test period
        test_data = ret_wide.loc[test_mask, active_tickers]
        # A ticker "has data" if it has any non-zero return in test period
        # But we keep all 17 tickers for simplicity as specified
        tickers_for_weights = active_tickers

        # Compute weights on train data
        if len(train_data) < 30:
            # Not enough training data (e.g., test_year=2022 is the first year)
            # Fall back to equal weights
            n_w = len(tickers_for_weights)
            weights_dict = {ws: np.ones(n_w) / n_w for ws in WEIGHT_SCHEMES}
        else:
            train_sub = train_data[tickers_for_weights]
            weights_dict = compute_all_weights(train_sub, tickers_for_weights)

        # Get weights for this scheme
        w = weights_dict[weight_scheme]
        w_series = pd.Series(w, index=tickers_for_weights)

        # Apply weights on test period
        test_ret = ret_wide.loc[test_mask, tickers_for_weights]
        test_abspos = abspos_wide.loc[test_mask, tickers_for_weights]
        test_pos = pos_wide.loc[test_mask, tickers_for_weights]

        # Weighted position for commission on weight changes
        # weighted_pos_i = w_i * pos_i
        weighted_pos = test_pos.multiply(w_series, axis=1)

        # Portfolio daily return = sum(w_i * net_ret_i)
        port_ret = test_ret.multiply(w_series, axis=1).sum(axis=1)

        # Portfolio exposure = sum(w_i * |pos_i|)
        port_exp = test_abspos.multiply(w_series, axis=1).sum(axis=1)

        chunk = pd.DataFrame({
            "portfolio_return": port_ret,
            "portfolio_exposure": port_exp,
        }, index=test_dates)
        port_chunks.append(chunk)

    if not port_chunks:
        return None, 0

    port = pd.concat(port_chunks).sort_index()

    # Step 4: Compute FDR, margin, excess returns
    rf_daily = kr_series.reindex(port.index).ffill().bfill() / 365.0
    cal_days = compute_cal_days(port.index)

    port["rf_daily"] = rf_daily.values
    port["cal_days"] = cal_days
    exp_arr = port["portfolio_exposure"].values

    # FDR: cash earns RF
    port["fdr"] = np.maximum(0.0, 1.0 - exp_arr) * port["rf_daily"].values * cal_days
    # Margin cost for leverage > 1
    kr_ann = kr_series.reindex(port.index).ffill().bfill().values
    port["margin_cost"] = np.maximum(0.0, exp_arr - 1.0) * (kr_ann + 0.02) / 365.0

    port["portfolio_total"] = port["portfolio_return"] + port["fdr"] - port["margin_cost"]
    port["excess"] = port["portfolio_total"] - port["rf_daily"] * port["cal_days"]

    return port, n_tickers


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(port):
    """Compute all performance metrics from portfolio DataFrame."""
    total = port["portfolio_total"].values
    excess = port["excess"].values
    exp_arr = port["portfolio_exposure"].values
    trading_ret = port["portfolio_return"].values
    cd = port["cal_days"].values
    n = len(total)

    if n < 30:
        return {k: np.nan for k in [
            "sharpe_excess", "sharpe_trading", "sharpe_excess_9pct",
            "sortino", "calmar", "max_dd",
            "ann_ret", "ann_vol", "exposure"]}

    # Annual return
    n_years = max(n / BPY, 0.01)
    cum = np.sum(total)
    ann_ret = cum / n_years

    # Annual vol
    ann_vol = np.std(total, ddof=1) * sqrt(BPY)

    # Excess Sharpe (real RF)
    s_total = np.std(total, ddof=1)
    sharpe_excess = np.mean(excess) * sqrt(BPY) / s_total if s_total > 1e-12 else 0.0

    # Trading Sharpe (no RF, no FDR)
    s_trad = np.std(trading_ret, ddof=1)
    sharpe_trading = np.mean(trading_ret) * sqrt(BPY) / s_trad if s_trad > 1e-12 else 0.0

    # Counterfactual excess (RF = 9%)
    rf_cf = RF_MEAN_HIST / 365.0
    fdr_cf = np.maximum(0.0, 1.0 - exp_arr) * rf_cf * cd
    margin_cf = np.maximum(0.0, exp_arr - 1.0) * (RF_MEAN_HIST + 0.02) / 365.0
    total_cf = trading_ret + fdr_cf - margin_cf
    excess_cf = total_cf - rf_cf * cd
    s_cf = np.std(total_cf, ddof=1)
    sharpe_excess_9pct = np.mean(excess_cf) * sqrt(BPY) / s_cf if s_cf > 1e-12 else 0.0

    # Sortino
    downside = total[total < 0]
    ds_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-12
    sortino = np.mean(excess) * sqrt(BPY) / ds_std if ds_std > 1e-12 else 0.0

    # Max drawdown
    cum_arr = np.cumsum(total)
    peak = np.maximum.accumulate(cum_arr)
    dd = peak - cum_arr
    max_dd = np.max(dd) * 100  # percent

    # Calmar
    calmar = ann_ret / (max_dd / 100) if max_dd > 1e-6 else 0.0

    # Exposure
    avg_exp = np.mean(exp_arr) * 100

    return {
        "sharpe_trading": round(sharpe_trading, 3),
        "sharpe_excess": round(sharpe_excess, 3),
        "sharpe_excess_9pct": round(sharpe_excess_9pct, 3),
        "ann_ret": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "max_dd": round(max_dd, 2),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "exposure": round(avg_exp, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("STAGE 5b: Per-strategy portfolios with 4 weighting schemes")
    log("=" * 80)

    kr_series = load_key_rate()
    pos_df = load_positions()

    all_rows = []
    total_combos = len(STRATEGIES) * len(APPROACHES) * len(COMM_DAILY) * len(WEIGHT_SCHEMES)
    done = 0

    for app in APPROACHES:
        app_data = pos_df[pos_df["approach"] == app]
        if app_data.empty:
            log(f"  No data for approach {app}")
            continue

        for comm_label, comm_val in COMM_DAILY.items():
            for strat in STRATEGIES:
                strat_data = app_data[app_data["strategy"] == strat]
                if strat_data.empty:
                    log(f"  No data for {strat}_{app}_{comm_label}")
                    continue

                for ws in WEIGHT_SCHEMES:
                    done += 1
                    variant = f"{strat}_{app}_{ws}_{comm_label}"

                    port, n_tickers = build_weighted_portfolio(
                        strat_data, kr_series, comm_val, ws
                    )

                    if port is None or len(port) < 30:
                        log(f"  [{done}/{total_combos}] Skipped {variant}: insufficient data")
                        continue

                    metrics = compute_metrics(port)

                    row = {
                        "strategy": strat,
                        "approach": app,
                        "comm_label": comm_label,
                        "comm": comm_val,
                        "weight_scheme": ws,
                        "n_tickers": n_tickers,
                    }
                    row.update(metrics)
                    row["best_weight"] = False  # Will be set later
                    all_rows.append(row)

                    # Save daily returns for bootstrap
                    ret_df = port[["portfolio_return", "portfolio_total", "excess"]].copy()
                    ret_df.columns = ["trading_ret", "total_ret", "excess_ret"]
                    ret_path = PERSTRAT_RET_DIR / f"{variant}.csv"
                    ret_df.to_csv(ret_path)

                    if done % 20 == 0 or done == total_combos:
                        log(f"  [{done}/{total_combos}] {variant}: "
                            f"ShTr={metrics['sharpe_trading']:.3f}  "
                            f"ShEx={metrics['sharpe_excess']:.3f}  "
                            f"Sh9={metrics['sharpe_excess_9pct']:.3f}  "
                            f"Exp={metrics['exposure']:.1f}%")

    if not all_rows:
        log("ERROR: No results computed. Check data availability.")
        return

    results_df = pd.DataFrame(all_rows)

    # Mark best weight_scheme per (strategy, approach, comm_label) by highest trading Sharpe
    results_df["best_weight"] = False
    for (strat, app, cl), grp in results_df.groupby(["strategy", "approach", "comm_label"]):
        idx_best = grp["sharpe_trading"].idxmax()
        if pd.notna(grp.loc[idx_best, "sharpe_trading"]):
            results_df.loc[idx_best, "best_weight"] = True

    # Save
    out_path = MSP_DIR / "perstrat_results.csv"
    results_df.to_csv(out_path, index=False)
    log(f"Saved perstrat_results.csv: {len(results_df)} rows")

    # Summary
    log("")
    log("=== SUMMARY: Best weight scheme per strategy (trading Sharpe) ===")
    best = results_df[results_df["best_weight"]]
    for _, r in best.iterrows():
        log(f"  {r['strategy']}_{r['approach']}_{r['comm_label']}: "
            f"{r['weight_scheme']} → ShTr={r['sharpe_trading']:.3f}")

    log("")
    log("=== SUMMARY: Mean trading Sharpe by weight scheme ===")
    for ws in WEIGHT_SCHEMES:
        sub = results_df[results_df["weight_scheme"] == ws]
        mean_sh = sub["sharpe_trading"].mean()
        log(f"  {ws}: {mean_sh:.3f}")

    log("")
    log("STAGE 5b COMPLETE")

    print(f"\n{'=' * 80}")
    print(results_df.to_string(index=False))
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
