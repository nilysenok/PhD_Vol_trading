#!/usr/bin/env python3
"""
stage5c_extended_portfolio.py — Extended portfolio optimization.

PART 1: Extended Per-Strategy Portfolio
  For each strategy S1-S6, optimize across:
    - 3 ticker selections: ALL, POSITIVE (train Sharpe > 0), TOP12 (top 12 by train Sharpe)
    - 5 weighting schemes: EW, InvVol, RiskParity, MaxSharpe, MaxDiv
    - 4 approaches: A, B, C, D
    - 2 commission levels: inst (0.0005), std (0.0006)
  Walk-forward: for each test_year (2022-2025), train on prior years.
  Total: 6 x 3 x 5 x 4 x 2 = 720 variants.

PART 2: Extended MSP
  For each strategy_set x MSP_weight x approach x comm:
    - 4 strategy sets: ALL6, TOP4, NO_S5, DIVERSE3 (S1+S3+S6)
    - 5 MSP weights: EW, IV, RP, MS, MaxDiv
    - 4 approaches x 2 comms
  Uses best per-strategy portfolio (from Part 1) as inputs.

Output:
  msp/extended_perstrat.csv
  msp/extended_msp.csv
  msp/extended_daily_returns/  (daily returns for bootstrap)
"""
import os, sys, warnings, datetime, time
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
EXT_RET_DIR = MSP_DIR / "extended_daily_returns"
EXT_RET_DIR.mkdir(parents=True, exist_ok=True)
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

SELECTIONS = ["ALL", "POSITIVE", "TOP12"]
WEIGHT_SCHEMES = ["EW", "IV", "RP", "MS", "MaxDiv"]
COMM_DAILY = {"inst": 0.0005, "std": 0.0006}

# MSP strategy sets
MSP_STRATEGY_SETS = {
    "ALL6":     ["S1", "S2", "S3", "S4", "S5", "S6"],
    "TOP4":     None,   # determined dynamically per (approach, comm)
    "NO_S5":    ["S1", "S2", "S3", "S4", "S6"],
    "DIVERSE3": ["S1", "S3", "S6"],
}
MSP_WEIGHTS = ["EW", "IV", "RP", "MS", "MaxDiv"]


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage5c_extended] {msg}"
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
# Load positions
# ══════════════════════════════════════════════════════════════════════
def load_positions():
    """Load daily WF positions for all approaches."""
    pos_data = {}
    for app in APPROACHES:
        fp = WF_DIR / f"wf_positions_{app}.parquet"
        if fp.exists():
            tmp = pd.read_parquet(fp)
            tmp["date"] = pd.to_datetime(tmp["date"])
            pos_data[app] = tmp
            log(f"  Loaded {fp.name}: {len(tmp)} rows")
        else:
            log(f"  WARNING: {fp} not found")
    if not pos_data:
        raise FileNotFoundError("No position files found.")
    return pos_data


# ══════════════════════════════════════════════════════════════════════
# Weight computation functions
# ══════════════════════════════════════════════════════════════════════
def w_equal(n):
    return np.ones(n) / n


def w_inv_vol(vols):
    vols = np.maximum(vols, 1e-10)
    inv = 1.0 / vols
    return inv / inv.sum()


def w_risk_parity(cov, n_iter=100):
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(n_iter):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return np.ones(n) / n
        mrc = cov @ w / port_vol
        rc = w * mrc
        target = port_vol / n
        rc = np.maximum(rc, 1e-12)
        w_new = w * (target / rc)
        w_new = np.maximum(w_new, 0.0)
        s = w_new.sum()
        w_new = w_new / s if s > 1e-12 else np.ones(n) / n
        w = w_new
    return w


def w_max_sharpe(mu, cov):
    n = len(mu)
    if n == 0:
        return np.array([])

    def neg_sharpe(w):
        pr = w @ mu
        pv = np.sqrt(w @ cov @ w)
        return -pr / pv if pv > 1e-12 else 0.0

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                      constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    if result.success:
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        return w / s if s > 1e-12 else np.ones(n) / n
    return np.ones(n) / n


def w_max_diversification(cov):
    """Max diversification ratio: max (w'sigma) / sqrt(w'Cov w), long-only."""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    vols = np.sqrt(np.diag(cov))
    vols = np.maximum(vols, 1e-10)

    def neg_div_ratio(w):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        return -(w @ vols) / port_vol

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(neg_div_ratio, w0, method="SLSQP", bounds=bounds,
                      constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    if result.success:
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        return w / s if s > 1e-12 else np.ones(n) / n
    return np.ones(n) / n


def compute_weights(w_name, mu, cov, vols, n):
    """Dispatch to the right weight function."""
    if w_name == "EW":
        return w_equal(n)
    elif w_name == "IV":
        return w_inv_vol(vols)
    elif w_name == "RP":
        return w_risk_parity(cov)
    elif w_name == "MS":
        return w_max_sharpe(mu, cov)
    elif w_name == "MaxDiv":
        return w_max_diversification(cov)
    return w_equal(n)


# ══════════════════════════════════════════════════════════════════════
# Calendar-days helper
# ══════════════════════════════════════════════════════════════════════
def compute_cal_days(dates_index):
    cal_days = np.ones(len(dates_index))
    for i in range(1, len(dates_index)):
        cal_days[i] = (dates_index[i] - dates_index[i - 1]).days
    return cal_days


# ══════════════════════════════════════════════════════════════════════
# Ticker-level trading Sharpe on training data
# ══════════════════════════════════════════════════════════════════════
def ticker_train_sharpe(train_df):
    """Compute trading Sharpe per ticker from training data.
    train_df: filtered to one strategy, columns: date, ticker, daily_gross_return.
    Returns pd.Series indexed by ticker.
    """
    result = {}
    for tk, grp in train_df.groupby("ticker"):
        rets = grp["daily_gross_return"].values
        if len(rets) < 20:
            result[tk] = -999.0
            continue
        mu = np.mean(rets)
        sd = np.std(rets, ddof=1)
        result[tk] = mu * sqrt(BPY) / sd if sd > 1e-12 else 0.0
    return pd.Series(result)


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(total, excess, trading_ret, exp_arr, cal_days):
    """Compute performance metrics from arrays."""
    n = len(total)
    if n < 30:
        return {k: np.nan for k in [
            "sharpe_excess", "sharpe_trading", "sharpe_excess_9pct",
            "sortino", "calmar", "max_dd", "ann_ret", "ann_vol", "exposure"]}

    n_years = max(n / BPY, 0.01)
    cum = np.sum(total)
    ann_ret = cum / n_years
    ann_vol = np.std(total, ddof=1) * sqrt(BPY)

    s_total = np.std(total, ddof=1)
    sharpe_excess = np.mean(excess) * sqrt(BPY) / s_total if s_total > 1e-12 else 0.0

    s_trad = np.std(trading_ret, ddof=1)
    sharpe_trading = np.mean(trading_ret) * sqrt(BPY) / s_trad if s_trad > 1e-12 else 0.0

    # Counterfactual excess (RF = 9%)
    rf_cf = RF_MEAN_HIST / 365.0
    fdr_cf = np.maximum(0.0, 1.0 - exp_arr) * rf_cf * cal_days
    margin_cf = np.maximum(0.0, exp_arr - 1.0) * (RF_MEAN_HIST + 0.02) / 365.0
    total_cf = trading_ret + fdr_cf - margin_cf
    excess_cf = total_cf - rf_cf * cal_days
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
    max_dd = np.max(dd) * 100

    calmar = ann_ret / (max_dd / 100) if max_dd > 1e-6 else 0.0
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
# PART 1: Extended Per-Strategy Portfolio
# ══════════════════════════════════════════════════════════════════════
def run_part1(pos_data, kr_series):
    """Run extended per-strategy portfolio optimization.
    Returns (results_df, daily_returns_cache).
    daily_returns_cache: dict of (strat, app, sel, w_name, comm_label) -> pd.DataFrame
    """
    log("=" * 70)
    log("PART 1: Extended Per-Strategy Portfolio")
    log("=" * 70)

    all_rows = []
    # Cache: key = (strat, app, sel, w_name, comm_label) -> portfolio DataFrame
    daily_returns_cache = {}

    total_combos = len(STRATEGIES) * len(APPROACHES) * len(SELECTIONS) * \
                   len(WEIGHT_SCHEMES) * len(COMM_DAILY)
    done = 0
    t0 = time.time()

    for app in APPROACHES:
        if app not in pos_data:
            continue
        app_df = pos_data[app]

        for sid in STRATEGIES:
            strat_df = app_df[app_df["strategy"] == sid].copy()
            if strat_df.empty:
                done += len(SELECTIONS) * len(WEIGHT_SCHEMES) * len(COMM_DAILY)
                continue

            # Pre-compute per-ticker data structures for this strategy+approach
            # Build wide matrices per year for reuse
            ticker_frames = {}
            for tk in TICKERS:
                sub = strat_df[strat_df["ticker"] == tk].sort_values("date").reset_index(drop=True)
                if len(sub) > 0:
                    ticker_frames[tk] = sub

            # Build wide DataFrames: date x ticker for gross_return and position
            all_dates = sorted(strat_df["date"].unique())
            ret_wide = pd.DataFrame(0.0, index=all_dates, columns=TICKERS)
            pos_wide = pd.DataFrame(0.0, index=all_dates, columns=TICKERS)

            for tk, sub in ticker_frames.items():
                sub_idx = sub.set_index("date")
                ret_wide.loc[sub_idx.index, tk] = sub_idx["daily_gross_return"].values
                pos_wide.loc[sub_idx.index, tk] = sub_idx["position"].values

            ret_wide.index = pd.DatetimeIndex(ret_wide.index)
            pos_wide.index = pd.DatetimeIndex(pos_wide.index)

            # Pre-compute year labels
            year_arr = ret_wide.index.year

            for comm_label, comm_val in COMM_DAILY.items():
                # Net returns: gross - |delta_pos| * comm
                net_wide = ret_wide.copy()
                for tk in TICKERS:
                    p = pos_wide[tk].values
                    dp = np.abs(np.diff(p, prepend=0.0))
                    net_wide[tk] = ret_wide[tk].values - dp * comm_val

                for sel in SELECTIONS:
                    for w_name in WEIGHT_SCHEMES:
                        done += 1
                        variant = f"{sid}_{app}_{sel}_{w_name}_{comm_label}"

                        # Walk-forward per year
                        yearly_port_ret = []
                        yearly_trad_ret = []
                        yearly_exp = []
                        yearly_dates = []

                        for test_year in OOS_YEARS:
                            train_mask = year_arr < test_year
                            test_mask = year_arr == test_year

                            test_dates_idx = ret_wide.index[test_mask]
                            if len(test_dates_idx) == 0:
                                continue

                            train_net = net_wide.loc[train_mask]

                            # --- Ticker selection ---
                            if sel == "ALL":
                                tickers_sel = list(TICKERS)
                            elif sel == "POSITIVE":
                                if train_net.shape[0] >= 20:
                                    # Compute per-ticker trading Sharpe on train net returns
                                    train_mu = train_net.mean()
                                    train_sd = train_net.std(ddof=1)
                                    train_sh = train_mu * sqrt(BPY) / train_sd.replace(0, np.nan)
                                    train_sh = train_sh.fillna(0.0)
                                    tickers_sel = train_sh[train_sh > 0].index.tolist()
                                else:
                                    tickers_sel = list(TICKERS)
                            elif sel == "TOP12":
                                if train_net.shape[0] >= 20:
                                    train_mu = train_net.mean()
                                    train_sd = train_net.std(ddof=1)
                                    train_sh = train_mu * sqrt(BPY) / train_sd.replace(0, np.nan)
                                    train_sh = train_sh.fillna(0.0)
                                    tickers_sel = train_sh.nlargest(12).index.tolist()
                                else:
                                    tickers_sel = list(TICKERS)
                            else:
                                tickers_sel = list(TICKERS)

                            # Fallback if too few
                            if len(tickers_sel) < 5:
                                tickers_sel = list(TICKERS)

                            n_tk = len(tickers_sel)

                            # --- Compute weights on train ---
                            if train_net.shape[0] >= 30:
                                tr = train_net[tickers_sel]
                                mu_arr = tr.mean().values
                                cov_arr = tr.cov().values + np.eye(n_tk) * 1e-10
                                vol_arr = tr.std().values
                                w = compute_weights(w_name, mu_arr, cov_arr, vol_arr, n_tk)
                            else:
                                w = w_equal(n_tk)

                            # --- Apply on test ---
                            test_net = net_wide.loc[test_mask, tickers_sel].values
                            test_gross = ret_wide.loc[test_mask, tickers_sel].values
                            test_abs_pos = pos_wide.loc[test_mask, tickers_sel].abs().values

                            port_ret_arr = test_net @ w
                            trad_ret_arr = test_gross @ w
                            exp_arr = test_abs_pos @ w

                            yearly_port_ret.append(port_ret_arr)
                            yearly_trad_ret.append(trad_ret_arr)
                            yearly_exp.append(exp_arr)
                            yearly_dates.append(test_dates_idx)

                        if not yearly_dates:
                            continue

                        # Concatenate
                        all_dates_concat = yearly_dates[0].append(yearly_dates[1:]) if len(yearly_dates) > 1 \
                            else yearly_dates[0]
                        all_port_ret = np.concatenate(yearly_port_ret)
                        all_trad_ret = np.concatenate(yearly_trad_ret)
                        all_exp = np.concatenate(yearly_exp)

                        # Rebuild DatetimeIndex
                        all_dates_idx = pd.DatetimeIndex(np.concatenate(
                            [d.values for d in yearly_dates]))

                        # FDR + margin
                        rf_daily = kr_series.reindex(all_dates_idx).ffill().bfill().values / 365.0
                        cal_days = compute_cal_days(all_dates_idx)
                        kr_ann = kr_series.reindex(all_dates_idx).ffill().bfill().values

                        fdr = np.maximum(0.0, 1.0 - all_exp) * rf_daily * cal_days
                        margin = np.maximum(0.0, all_exp - 1.0) * (kr_ann + 0.02) / 365.0

                        total_ret = all_port_ret + fdr - margin
                        excess_ret = total_ret - rf_daily * cal_days

                        metrics = compute_metrics(total_ret, excess_ret,
                                                  all_trad_ret, all_exp, cal_days)

                        row = {
                            "strategy": sid,
                            "approach": app,
                            "selection": sel,
                            "weight_scheme": w_name,
                            "comm_label": comm_label,
                            "comm": comm_val,
                        }
                        row.update(metrics)
                        all_rows.append(row)

                        # Cache daily returns for MSP (Part 2)
                        cache_key = (sid, app, sel, w_name, comm_label)
                        daily_returns_cache[cache_key] = pd.DataFrame({
                            "trading_ret": all_trad_ret,
                            "total_ret": total_ret,
                            "excess_ret": excess_ret,
                            "exposure": all_exp,
                        }, index=all_dates_idx)

                        # Save daily returns
                        ret_path = EXT_RET_DIR / f"perstrat_{variant}.csv"
                        daily_returns_cache[cache_key].to_csv(ret_path)

                        if done % 60 == 0 or done == total_combos:
                            elapsed = time.time() - t0
                            log(f"  [{done}/{total_combos}] {variant}: "
                                f"ShTr={metrics['sharpe_trading']:.3f}  "
                                f"ShEx={metrics['sharpe_excess']:.3f}  "
                                f"Sh9={metrics['sharpe_excess_9pct']:.3f}  "
                                f"Exp={metrics['exposure']:.1f}%  "
                                f"({elapsed:.0f}s)")

    if not all_rows:
        log("ERROR: No per-strategy results computed.")
        return pd.DataFrame(), {}

    results_df = pd.DataFrame(all_rows)

    # Mark best variant per (strategy, approach, comm_label)
    results_df["best_variant"] = False
    for (sid, app, cl), grp in results_df.groupby(["strategy", "approach", "comm_label"]):
        idx_best = grp["sharpe_trading"].idxmax()
        if pd.notna(grp.loc[idx_best, "sharpe_trading"]):
            results_df.loc[idx_best, "best_variant"] = True

    # Save
    out_path = MSP_DIR / "extended_perstrat.csv"
    results_df.to_csv(out_path, index=False)
    log(f"Saved extended_perstrat.csv: {len(results_df)} rows")

    elapsed = time.time() - t0
    log(f"PART 1 complete in {elapsed:.1f}s")

    return results_df, daily_returns_cache


# ══════════════════════════════════════════════════════════════════════
# PART 2: Extended MSP
# ══════════════════════════════════════════════════════════════════════
def run_part2(perstrat_df, daily_returns_cache, kr_series):
    """Run extended MSP optimization."""
    log("")
    log("=" * 70)
    log("PART 2: Extended MSP")
    log("=" * 70)

    if perstrat_df.empty:
        log("ERROR: No per-strategy results. Skipping Part 2.")
        return pd.DataFrame()

    all_rows = []
    t0 = time.time()

    # For each (approach, comm_label), find best per-strategy variant
    # and determine TOP4 strategies
    def find_best_key(sid, app, comm_label):
        """Find the cache key for the best per-strategy variant."""
        mask = (perstrat_df["strategy"] == sid) & \
               (perstrat_df["approach"] == app) & \
               (perstrat_df["comm_label"] == comm_label)
        sub = perstrat_df.loc[mask]
        if sub.empty:
            return None
        best_idx = sub["sharpe_trading"].idxmax()
        if pd.isna(sub.loc[best_idx, "sharpe_trading"]):
            return None
        r = sub.loc[best_idx]
        return (sid, app, r["selection"], r["weight_scheme"], comm_label)

    total_combos = 4 * len(MSP_WEIGHTS) * len(APPROACHES) * len(COMM_DAILY)
    done = 0

    for app in APPROACHES:
        for comm_label, comm_val in COMM_DAILY.items():
            # Load best per-strategy daily returns
            best_strat_rets = {}
            best_strat_sharpes = {}
            for sid in STRATEGIES:
                key = find_best_key(sid, app, comm_label)
                if key is not None and key in daily_returns_cache:
                    best_strat_rets[sid] = daily_returns_cache[key]
                    mask = (perstrat_df["strategy"] == sid) & \
                           (perstrat_df["approach"] == app) & \
                           (perstrat_df["comm_label"] == comm_label)
                    sub = perstrat_df.loc[mask]
                    best_strat_sharpes[sid] = sub["sharpe_trading"].max()

            if len(best_strat_rets) < 2:
                done += len(MSP_STRATEGY_SETS) * len(MSP_WEIGHTS)
                continue

            # Determine TOP4 dynamically
            sorted_strats = sorted(best_strat_sharpes.items(), key=lambda x: x[1], reverse=True)
            top4 = [s for s, _ in sorted_strats[:4]]

            strategy_sets = {
                "ALL6": [s for s in STRATEGIES if s in best_strat_rets],
                "TOP4": [s for s in top4 if s in best_strat_rets],
                "NO_S5": [s for s in ["S1", "S2", "S3", "S4", "S6"] if s in best_strat_rets],
                "DIVERSE3": [s for s in ["S1", "S3", "S6"] if s in best_strat_rets],
            }

            for set_name, strat_set in strategy_sets.items():
                if len(strat_set) < 2:
                    done += len(MSP_WEIGHTS)
                    continue

                # Build strategy-level return matrix (common dates)
                # Use total_ret for strategy-level returns
                common_dates = None
                for sid in strat_set:
                    idx = best_strat_rets[sid].index
                    common_dates = idx if common_dates is None else common_dates.intersection(idx)

                if common_dates is None or len(common_dates) < 30:
                    done += len(MSP_WEIGHTS)
                    continue

                common_dates = common_dates.sort_values()

                # Walk-forward MSP: for each test_year, train on prior years
                strat_total = pd.DataFrame(
                    {sid: best_strat_rets[sid].loc[common_dates, "total_ret"].values
                     for sid in strat_set},
                    index=common_dates)
                strat_trading = pd.DataFrame(
                    {sid: best_strat_rets[sid].loc[common_dates, "trading_ret"].values
                     for sid in strat_set},
                    index=common_dates)
                strat_exp = pd.DataFrame(
                    {sid: best_strat_rets[sid].loc[common_dates, "exposure"].values
                     for sid in strat_set},
                    index=common_dates)

                year_arr = common_dates.year

                for msp_w_name in MSP_WEIGHTS:
                    done += 1
                    variant = f"MSP_{set_name}_{msp_w_name}_{app}_{comm_label}"

                    yearly_total = []
                    yearly_trading = []
                    yearly_exp = []
                    yearly_dates = []

                    for test_year in OOS_YEARS:
                        train_mask = year_arr < test_year
                        test_mask = year_arr == test_year

                        test_dates_idx = common_dates[test_mask]
                        if len(test_dates_idx) == 0:
                            continue

                        train_data = strat_total.loc[train_mask]
                        n_s = len(strat_set)

                        if len(train_data) >= 30:
                            mu_s = train_data.mean().values
                            cov_s = train_data.cov().values + np.eye(n_s) * 1e-10
                            vol_s = train_data.std().values
                            w = compute_weights(msp_w_name, mu_s, cov_s, vol_s, n_s)
                        else:
                            w = w_equal(n_s)

                        test_total = strat_total.loc[test_mask].values
                        test_trading = strat_trading.loc[test_mask].values
                        test_exposure = strat_exp.loc[test_mask].values

                        yearly_total.append(test_total @ w)
                        yearly_trading.append(test_trading @ w)
                        yearly_exp.append(test_exposure @ w)
                        yearly_dates.append(test_dates_idx)

                    if not yearly_dates:
                        continue

                    all_dates_idx = pd.DatetimeIndex(np.concatenate(
                        [d.values for d in yearly_dates]))
                    all_total = np.concatenate(yearly_total)
                    all_trading = np.concatenate(yearly_trading)
                    all_exp_arr = np.concatenate(yearly_exp)

                    # For MSP, total_ret already includes FDR/margin from per-strategy level
                    # So excess = total - rf_daily * cal_days
                    rf_daily = kr_series.reindex(all_dates_idx).ffill().bfill().values / 365.0
                    cal_days = compute_cal_days(all_dates_idx)
                    excess = all_total - rf_daily * cal_days

                    metrics = compute_metrics(all_total, excess,
                                              all_trading, all_exp_arr, cal_days)

                    row = {
                        "strategy_set": set_name,
                        "n_strategies": len(strat_set),
                        "strategies": "+".join(strat_set),
                        "msp_weight": msp_w_name,
                        "approach": app,
                        "comm_label": comm_label,
                        "comm": comm_val,
                    }
                    row.update(metrics)
                    all_rows.append(row)

                    # Save daily returns
                    ret_df = pd.DataFrame({
                        "trading_ret": all_trading,
                        "total_ret": all_total,
                        "excess_ret": excess,
                        "exposure": all_exp_arr,
                    }, index=all_dates_idx)
                    ret_path = EXT_RET_DIR / f"msp_{variant}.csv"
                    ret_df.to_csv(ret_path)

                    if done % 20 == 0:
                        elapsed = time.time() - t0
                        log(f"  [{done}/{total_combos}] {variant}: "
                            f"ShTr={metrics['sharpe_trading']:.3f}  "
                            f"ShEx={metrics['sharpe_excess']:.3f}  "
                            f"Exp={metrics['exposure']:.1f}%  "
                            f"({elapsed:.0f}s)")

    if not all_rows:
        log("ERROR: No MSP results computed.")
        return pd.DataFrame()

    msp_df = pd.DataFrame(all_rows)

    # Mark best per (strategy_set, approach, comm_label)
    msp_df["best_variant"] = False
    for (ss, app, cl), grp in msp_df.groupby(["strategy_set", "approach", "comm_label"]):
        idx_best = grp["sharpe_trading"].idxmax()
        if pd.notna(grp.loc[idx_best, "sharpe_trading"]):
            msp_df.loc[idx_best, "best_variant"] = True

    out_path = MSP_DIR / "extended_msp.csv"
    msp_df.to_csv(out_path, index=False)
    log(f"Saved extended_msp.csv: {len(msp_df)} rows")

    elapsed = time.time() - t0
    log(f"PART 2 complete in {elapsed:.1f}s")

    return msp_df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("STAGE 5c: Extended Portfolio Optimization")
    log("=" * 80)
    t_start = time.time()

    kr_series = load_key_rate()
    pos_data = load_positions()

    # PART 1
    perstrat_df, daily_returns_cache = run_part1(pos_data, kr_series)

    # PART 2
    msp_df = run_part2(perstrat_df, daily_returns_cache, kr_series)

    # ── Summary ──
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)

    if not perstrat_df.empty:
        log("")
        log("--- Best per-strategy variant (by trading Sharpe) ---")
        best_ps = perstrat_df[perstrat_df["best_variant"]]
        for _, r in best_ps.iterrows():
            log(f"  {r['strategy']}_{r['approach']}_{r['comm_label']}: "
                f"sel={r['selection']}, w={r['weight_scheme']} -> "
                f"ShTr={r['sharpe_trading']:.3f}  ShEx={r['sharpe_excess']:.3f}  "
                f"Sh9={r['sharpe_excess_9pct']:.3f}")

        log("")
        log("--- Mean trading Sharpe by selection method ---")
        for sel in SELECTIONS:
            sub = perstrat_df[perstrat_df["selection"] == sel]
            log(f"  {sel}: {sub['sharpe_trading'].mean():.3f}")

        log("")
        log("--- Mean trading Sharpe by weight scheme ---")
        for ws in WEIGHT_SCHEMES:
            sub = perstrat_df[perstrat_df["weight_scheme"] == ws]
            log(f"  {ws}: {sub['sharpe_trading'].mean():.3f}")

    if not msp_df.empty:
        log("")
        log("--- Best MSP variant (by trading Sharpe) ---")
        best_msp = msp_df[msp_df["best_variant"]]
        for _, r in best_msp.iterrows():
            log(f"  {r['strategy_set']}_{r['approach']}_{r['comm_label']}: "
                f"w={r['msp_weight']} -> "
                f"ShTr={r['sharpe_trading']:.3f}  ShEx={r['sharpe_excess']:.3f}")

        log("")
        log("--- Mean trading Sharpe by strategy set ---")
        for ss in MSP_STRATEGY_SETS:
            sub = msp_df[msp_df["strategy_set"] == ss]
            if not sub.empty:
                log(f"  {ss}: {sub['sharpe_trading'].mean():.3f}")

        log("")
        log("--- Mean trading Sharpe by MSP weight ---")
        for mw in MSP_WEIGHTS:
            sub = msp_df[msp_df["msp_weight"] == mw]
            if not sub.empty:
                log(f"  {mw}: {sub['sharpe_trading'].mean():.3f}")

    elapsed = time.time() - t_start
    log("")
    log(f"STAGE 5c COMPLETE in {elapsed:.1f}s")
    log("=" * 80)


if __name__ == "__main__":
    main()
