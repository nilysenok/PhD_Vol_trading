#!/usr/bin/env python3
"""
stage4b_hourly_wf.py — Walk-forward on hourly data for all 6 strategies (A/B/C/D).

For each strategy S1-S6:
  1. Load hourly data for all 17 tickers
  2. For each test_year (2022-2025):
     a. Adapt daily configs using agent_hourly_adapter(cfg, scale=BEST_SCALE[sid])
     b. Generate signal on hourly data
     c. Run backtest on hourly data
     d. Aggregate hourly -> daily using hourly_to_daily_agg
     e. Compute 3 Sharpe metrics on daily-aggregated returns
  3. Save results and positions

Uses BEST daily configs from FINAL_best_configs.json, adapts windows x scale.
"""
import os, sys, warnings, time, json, datetime, gc
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from archivist import agent_hourly_adapter

from stage4_hourly import (
    gen_signal_hourly, run_bt, apply_confirmation, hourly_to_daily_agg,
    precompute_indicators, IS_TREND, STRATEGIES, STRATEGY_TYPE,
    WINDOWS_D, MA_TYPES, DC_WINDOWS, ST_PERIODS, ST_MULTS,
    PIVOT_TYPES, VWAP_WINDOWS_ALL,
    _apply_B_numba, apply_B, apply_C, apply_D,
    _year_sharpe,
)

from strategies_walkforward import (
    TICKERS, WARMUP_HOURLY, load_data,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily,
    compute_daily_pivots_for_hourly, _year_bounds, _compute_nhours_per_day,
)

from daily_upper_bound import bt_universal

# ================================================================
# PATHS & CONSTANTS
# ================================================================

BASE = SCRIPT_DIR.parent
HOURLY_PATH = BASE / "results/final/strategies/data/ohlcv_hourly_full.parquet"
DAILY_PATH = BASE / "results/final/strategies/data/ohlcv_daily_full.parquet"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
VPRED_PATH = BASE / "results/final/strategies/data/vpred_aligned.parquet"
CONFIGS_PATH = BASE / "results/final/strategies/test_phases/FINAL_best_configs.json"
OUT_DIR = BASE / "results/final/strategies/hourly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = BASE / "results/final/strategies/MASTER_PROGRESS_LOG.txt"

BPY = 252
WARMUP_H = 1400  # 200 days x 7 bars
OOS_YEARS = [2022, 2023, 2024, 2025]
RF_MEAN_HIST = 0.09
COMM_INST = 0.0004
COMM_STD = 0.0005

# Best scale per strategy (from pilot)
BEST_SCALES = {'S1': 7, 'S2': 7, 'S3': 8, 'S4': 6, 'S5': 7, 'S6': 8}

# BCD grids (coarse)
HORIZONS = ['h1', 'h5', 'h22']

B_GRID = dict(
    k_sl=[0.5, 1.0, 2.0, 3.0],
    ratio=[0.5, 1.5, 3.0],
    base_hold=[5, 15, 50],
    hold_gamma=[0.25, 1.0],
    be_trigger=[0, 1.5],
)

C_GRID = dict(
    threshold=[0.25, 0.55, 0.85],
    lookback=[63, 252],
    direction=['off', 'rise', 'fall'],
    term_str=[False, True],
    hysteresis=[0, 10],
)

D_GRID = dict(
    sigma_target=[0.05, 0.10, 0.15, 0.20, 0.30],
    gamma=[0.25, 0.5, 1.0, 2.0],
    smooth=[0, 10],
    vol_floor=[0.1, 0.3],
    vol_cap=[1.5, 2.0, 3.0],
)


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] HOURLY_WF: {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def grid_combos(grid_dict):
    """Generate all combos from a grid dict."""
    from itertools import product
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))


# ================================================================
# DATA PREPARATION
# ================================================================

def prepare_ticker_data(ticker, hourly_df, daily_df, vpred_df):
    """Prepare all data structures for one ticker on hourly timeframe."""
    hdf = hourly_df[hourly_df["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
    if len(hdf) < WARMUP_H + 200:
        return None

    close = hdf["close"].values.astype(np.float64)
    high = hdf["high"].values.astype(np.float64)
    low = hdf["low"].values.astype(np.float64)
    open_arr = hdf["open"].values.astype(np.float64)
    volume = hdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(hdf["datetime"].values)
    n = len(close)

    # Log returns (forward: lr[i] = log(close[i+1]/close[i]))
    lr = np.zeros(n)
    lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    # Indicators
    ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
    atr = ind["atr14"]
    rh = ind.get("rh", np.zeros(n))
    rl = ind.get("rl", np.zeros(n))

    # Caches for signal generation
    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS_ALL)

    # Pivot data (daily pivots aligned to hourly)
    pivot_data = {}
    for pv in PIVOT_TYPES:
        P, S1p, R1p, S2p, R2p, S3p, R3p = compute_daily_pivots_for_hourly(
            daily_df, ticker, hdf["datetime"].values, pv)
        pivot_data[pv] = (P, S1p, R1p, S2p, R2p, S3p, R3p)

    # Vpred alignment (daily -> hourly via searchsorted)
    vp_tk = vpred_df[vpred_df.ticker == ticker].copy()
    vp_tk["date"] = pd.to_datetime(vp_tk["date"])
    vp_tk = vp_tk.set_index("date").sort_index()
    hourly_bar_dates = dates.normalize()
    vp_dates = pd.to_datetime(vp_tk.index)

    # Daily-level sigma for BCD (NOT scaled by sqrt(nhours))
    sigma_daily = {}
    for h in ["h1", "h5", "h22"]:
        col = f"sigma_{h}"
        if col in vp_tk.columns:
            vp_vals = vp_tk[col].values
            sigma_arr = np.full(n, np.nan)
            idx = np.searchsorted(vp_dates, hourly_bar_dates, side="right") - 1
            valid = idx >= 0
            sigma_arr[valid] = vp_vals[idx[valid]]
            sigma_daily[h] = sigma_arr
        else:
            sigma_daily[h] = np.full(n, 0.02)

    return dict(
        ticker=ticker, close=close, high=high, low=low, open_arr=open_arr,
        volume=volume, dates=dates, n=n, lr=lr,
        ind=ind, atr=atr, rh=rh, rl=rl,
        dc_cache=dc_cache, st_cache=st_cache, vwap_cache=vwap_cache,
        pivot_data=pivot_data, sigma_daily=sigma_daily,
    )


# ================================================================
# SHARPE HELPERS
# ================================================================

def compute_three_sharpes(daily_net, daily_exp, daily_dates, kr_s, year):
    """Compute sharpe_trading, sharpe_excess, sharpe_excess_9pct for one year."""
    year_mask = daily_dates.year == year
    if year_mask.sum() < 20:
        return None

    dn = daily_net[year_mask]
    de = daily_exp[year_mask]
    dd = daily_dates[year_mask]
    ny = len(dn)

    # 1) Trading Sharpe (no RF)
    s_t = np.std(dn, ddof=1)
    sharpe_trading = np.mean(dn) * sqrt(BPY) / s_t if s_t > 1e-12 else 0.0

    # 2) Excess Sharpe (actual RF from CBR)
    rf_aligned = kr_s.reindex(dd, method="ffill").values / 365.0
    rf_aligned = np.where(np.isfinite(rf_aligned), rf_aligned, 0.15 / 365.0)
    cal_days = np.ones(ny)
    for i in range(1, ny):
        cal_days[i] = (dd[i] - dd[i - 1]).days
    fdr = (1.0 - de) * rf_aligned * cal_days
    daily_total = dn + fdr
    excess = daily_total - rf_aligned * cal_days
    s_e = np.std(daily_total, ddof=1)
    sharpe_excess = np.mean(excess) * sqrt(BPY) / s_e if s_e > 1e-12 else 0.0

    # 3) Excess Sharpe with fixed RF = 9%
    rf_9 = RF_MEAN_HIST / 365.0
    fdr_9 = (1.0 - de) * rf_9 * cal_days
    daily_total_9 = dn + fdr_9
    excess_9 = daily_total_9 - rf_9 * cal_days
    s_9 = np.std(daily_total_9, ddof=1)
    sharpe_excess_9pct = np.mean(excess_9) * sqrt(BPY) / s_9 if s_9 > 1e-12 else 0.0

    exp_pct = np.mean(de) * 100

    return dict(
        sharpe_trading=round(sharpe_trading, 4),
        sharpe_excess=round(sharpe_excess, 4),
        sharpe_excess_9pct=round(sharpe_excess_9pct, 4),
        exposure=round(exp_pct, 1),
    )


# ================================================================
# WALK-FORWARD A
# ================================================================

def run_walkforward_A(ticker_data_list, best_configs, kr_s, comm):
    """
    Walk-forward A on hourly: for each test_year, use adapted daily config,
    generate signal on full data, run BT, aggregate to daily, compute Sharpe.
    """
    log(f"=== Walk-Forward A (comm={comm}) ===")
    results = []
    positions_all = []

    for ti, td in enumerate(ticker_data_list):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        lr = td["lr"]; dates = td["dates"]; n = td["n"]
        volume = td["volume"]

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None:
                continue

            scale = BEST_SCALES[sid]
            sig_params = agent_hourly_adapter(cfg, scale=scale)
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"]) if isinstance(cfg["rm"], str) else cfg["rm"]

            # Generate signal on hourly data
            sig_raw = gen_signal_hourly(sid, td, sig_params, WARMUP_H)
            if sig_raw is None:
                continue

            rsi = td["ind"]["rsi14"]
            vol_sma20 = td["ind"]["vol_sma20"]
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)

            # Run backtest on full hourly data
            pos_h = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP_H, n)

            # Aggregate hourly -> daily
            daily_dates, daily_net, daily_exp, daily_pos, daily_gross = \
                hourly_to_daily_agg(pos_h, lr, dates, comm)

            if len(daily_dates) < 20:
                continue

            # Evaluate per year
            for year in OOS_YEARS:
                metrics = compute_three_sharpes(daily_net, daily_exp, daily_dates, kr_s, year)
                if metrics is None:
                    continue

                results.append(dict(
                    strategy=sid, ticker=ticker, year=year,
                    sharpe_trading=metrics["sharpe_trading"],
                    sharpe_excess=metrics["sharpe_excess"],
                    sharpe_excess_9pct=metrics["sharpe_excess_9pct"],
                    exposure=metrics["exposure"],
                    comm=comm,
                ))

                # Collect ALL daily positions (including zeros)
                year_mask = daily_dates.year == year
                dd_y = daily_dates[year_mask]
                dp_y = daily_pos[year_mask]
                dg_y = daily_gross[year_mask]
                for j in range(len(dd_y)):
                    positions_all.append(dict(
                        date=dd_y[j], ticker=ticker, strategy=sid, year=year,
                        position=round(float(dp_y[j]), 6),
                        daily_gross_return=round(float(dg_y[j]), 8),
                    ))

        elapsed = time.time() - t0
        log(f"  [{ti+1}/{len(ticker_data_list)}] {ticker}: A done in {elapsed:.1f}s")

    return pd.DataFrame(results), positions_all


# ================================================================
# WALK-FORWARD BCD
# ================================================================

def run_walkforward_BCD(ticker_data_list, best_configs, kr_s, comm, a_positions_list):
    """
    Walk-forward B/C/D on hourly:
    For each strategy x ticker x year:
      - Regenerate A positions on hourly
      - Apply B/C/D overlays using vpred
      - Aggregate to daily and compute Sharpe
      - For B: grid search per year (pick best on train), apply to test
      - For C/D: same grid search approach
    Simplified: apply best BCD params per year (no train/test split for BCD grid
    since A config is already fixed; we just pick best BCD on train years).
    """
    log(f"=== Walk-Forward BCD (comm={comm}) ===")
    results_all = {"B": [], "C": [], "D": []}
    positions_all = {"B": [], "C": [], "D": []}

    for ti, td in enumerate(ticker_data_list):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        lr = td["lr"]; dates = td["dates"]; n = td["n"]
        volume = td["volume"]
        sigma_daily = td["sigma_daily"]

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            stype = STRATEGY_TYPE[sid]
            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None:
                continue

            scale = BEST_SCALES[sid]
            sig_params = agent_hourly_adapter(cfg, scale=scale)
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"]) if isinstance(cfg["rm"], str) else cfg["rm"]

            # Generate A signal and positions on hourly
            sig_raw = gen_signal_hourly(sid, td, sig_params, WARMUP_H)
            if sig_raw is None:
                continue
            rsi = td["ind"]["rsi14"]
            vol_sma20 = td["ind"]["vol_sma20"]
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
            a_pos = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP_H, n)

            for test_year in OOS_YEARS:
                # Train: all data before test_year
                train_end_idx, _ = _year_bounds(dates, test_year)
                if train_end_idx <= WARMUP_H:
                    continue
                test_start, test_end = _year_bounds(dates, test_year)
                if test_year < 2026:
                    _, test_end = _year_bounds(dates, test_year + 1)
                else:
                    test_end = n
                if test_start >= test_end:
                    continue

                # ---- B ----
                best_b_sh = -999.0
                best_b_params = None
                best_b_horizon = None

                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    vm = np.nanmedian(vp[~np.isnan(vp)]) if np.any(~np.isnan(vp)) else 0.02

                    for bp in grid_combos(B_GRID):
                        # Evaluate on train
                        b_pos_train = apply_B(a_pos[:train_end_idx], close[:train_end_idx],
                                              atr[:train_end_idx], vp[:train_end_idx], vm,
                                              bp["k_sl"], bp["ratio"], bp["base_hold"],
                                              bp["hold_gamma"], bp["be_trigger"])
                        dd_t, dn_t, de_t, _, _ = hourly_to_daily_agg(
                            b_pos_train, lr[:train_end_idx], dates[:train_end_idx], comm)
                        if len(dd_t) < 30:
                            continue
                        # Full train sharpe
                        rf_t = kr_s.reindex(dd_t, method="ffill").values / 365.0
                        rf_t = np.where(np.isfinite(rf_t), rf_t, 0.15 / 365.0)
                        cd_t = np.ones(len(dd_t))
                        for i in range(1, len(dd_t)):
                            cd_t[i] = (dd_t[i] - dd_t[i - 1]).days
                        full_mask = np.ones(len(dd_t), dtype=bool)
                        sh_train = _year_sharpe(dn_t, de_t, rf_t, cd_t, full_mask)
                        if sh_train > best_b_sh:
                            best_b_sh = sh_train
                            best_b_params = bp
                            best_b_horizon = horizon

                if best_b_params is not None:
                    # Apply best B to full data, evaluate on test year
                    vp_best = sigma_daily.get(best_b_horizon, np.full(n, 0.02))
                    vm_best = np.nanmedian(vp_best[~np.isnan(vp_best)]) if np.any(~np.isnan(vp_best)) else 0.02
                    b_pos_full = apply_B(a_pos, close, atr, vp_best, vm_best,
                                         best_b_params["k_sl"], best_b_params["ratio"],
                                         best_b_params["base_hold"], best_b_params["hold_gamma"],
                                         best_b_params["be_trigger"])
                    dd_b, dn_b, de_b, dp_b, dg_b = hourly_to_daily_agg(b_pos_full, lr, dates, comm)
                    if len(dd_b) > 20:
                        metrics = compute_three_sharpes(dn_b, de_b, dd_b, kr_s, test_year)
                        if metrics is not None:
                            results_all["B"].append(dict(
                                strategy=sid, ticker=ticker, year=test_year,
                                **metrics, comm=comm,
                            ))
                            ym_b = dd_b.year == test_year
                            dd_y = dd_b[ym_b]; dp_y = dp_b[ym_b]; dg_y = dg_b[ym_b]
                            for j in range(len(dd_y)):
                                positions_all["B"].append(dict(
                                    date=dd_y[j], ticker=ticker, strategy=sid, year=test_year,
                                    position=round(float(dp_y[j]), 6),
                                    daily_gross_return=round(float(dg_y[j]), 8),
                                ))

                # ---- C ----
                best_c_sh = -999.0
                best_c_params = None
                best_c_horizon = None

                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))

                    for cp in grid_combos(C_GRID):
                        c_pos_train = apply_C(
                            a_pos[:train_end_idx], vp[:train_end_idx],
                            cp["threshold"], cp["lookback"], stype,
                            cp["direction"],
                            sigma_daily.get("h22", np.full(n, 0.02))[:train_end_idx],
                            sigma_daily.get("h1", np.full(n, 0.02))[:train_end_idx],
                            cp["term_str"], cp["hysteresis"])
                        dd_t, dn_t, de_t, _, _ = hourly_to_daily_agg(
                            c_pos_train, lr[:train_end_idx], dates[:train_end_idx], comm)
                        if len(dd_t) < 30:
                            continue
                        rf_t = kr_s.reindex(dd_t, method="ffill").values / 365.0
                        rf_t = np.where(np.isfinite(rf_t), rf_t, 0.15 / 365.0)
                        cd_t = np.ones(len(dd_t))
                        for i in range(1, len(dd_t)):
                            cd_t[i] = (dd_t[i] - dd_t[i - 1]).days
                        full_mask = np.ones(len(dd_t), dtype=bool)
                        sh_train = _year_sharpe(dn_t, de_t, rf_t, cd_t, full_mask)
                        if sh_train > best_c_sh:
                            best_c_sh = sh_train
                            best_c_params = cp
                            best_c_horizon = horizon

                if best_c_params is not None:
                    vp_best = sigma_daily.get(best_c_horizon, np.full(n, 0.02))
                    c_pos_full = apply_C(
                        a_pos, vp_best, best_c_params["threshold"],
                        best_c_params["lookback"], stype, best_c_params["direction"],
                        sigma_daily.get("h22", np.full(n, 0.02)),
                        sigma_daily.get("h1", np.full(n, 0.02)),
                        best_c_params["term_str"], best_c_params["hysteresis"])
                    dd_c, dn_c, de_c, dp_c, dg_c = hourly_to_daily_agg(c_pos_full, lr, dates, comm)
                    if len(dd_c) > 20:
                        metrics = compute_three_sharpes(dn_c, de_c, dd_c, kr_s, test_year)
                        if metrics is not None:
                            results_all["C"].append(dict(
                                strategy=sid, ticker=ticker, year=test_year,
                                **metrics, comm=comm,
                            ))
                            ym_c = dd_c.year == test_year
                            dd_y = dd_c[ym_c]; dp_y = dp_c[ym_c]; dg_y = dg_c[ym_c]
                            for j in range(len(dd_y)):
                                positions_all["C"].append(dict(
                                    date=dd_y[j], ticker=ticker, strategy=sid, year=test_year,
                                    position=round(float(dp_y[j]), 6),
                                    daily_gross_return=round(float(dg_y[j]), 8),
                                ))

                # ---- D ----
                best_d_sh = -999.0
                best_d_params = None
                best_d_horizon = None

                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))

                    for dp in grid_combos(D_GRID):
                        d_pos_train = apply_D(
                            a_pos[:train_end_idx], vp[:train_end_idx],
                            dp["sigma_target"], dp["gamma"],
                            dp["smooth"], dp["vol_floor"], dp["vol_cap"])
                        dd_t, dn_t, de_t, _, _ = hourly_to_daily_agg(
                            d_pos_train, lr[:train_end_idx], dates[:train_end_idx], comm)
                        if len(dd_t) < 30:
                            continue
                        rf_t = kr_s.reindex(dd_t, method="ffill").values / 365.0
                        rf_t = np.where(np.isfinite(rf_t), rf_t, 0.15 / 365.0)
                        cd_t = np.ones(len(dd_t))
                        for i in range(1, len(dd_t)):
                            cd_t[i] = (dd_t[i] - dd_t[i - 1]).days
                        full_mask = np.ones(len(dd_t), dtype=bool)
                        sh_train = _year_sharpe(dn_t, de_t, rf_t, cd_t, full_mask)
                        if sh_train > best_d_sh:
                            best_d_sh = sh_train
                            best_d_params = dp
                            best_d_horizon = horizon

                if best_d_params is not None:
                    vp_best = sigma_daily.get(best_d_horizon, np.full(n, 0.02))
                    d_pos_full = apply_D(
                        a_pos, vp_best, best_d_params["sigma_target"],
                        best_d_params["gamma"], best_d_params["smooth"],
                        best_d_params["vol_floor"], best_d_params["vol_cap"])
                    dd_d, dn_d, de_d, dp_d, dg_d = hourly_to_daily_agg(d_pos_full, lr, dates, comm)
                    if len(dd_d) > 20:
                        metrics = compute_three_sharpes(dn_d, de_d, dd_d, kr_s, test_year)
                        if metrics is not None:
                            results_all["D"].append(dict(
                                strategy=sid, ticker=ticker, year=test_year,
                                **metrics, comm=comm,
                            ))
                            ym_d = dd_d.year == test_year
                            dd_y = dd_d[ym_d]; dp_y = dp_d[ym_d]; dg_y = dg_d[ym_d]
                            for j in range(len(dd_y)):
                                positions_all["D"].append(dict(
                                    date=dd_y[j], ticker=ticker, strategy=sid, year=test_year,
                                    position=round(float(dp_y[j]), 6),
                                    daily_gross_return=round(float(dg_y[j]), 8),
                                ))

        elapsed = time.time() - t0
        log(f"  [{ti+1}/{len(ticker_data_list)}] {ticker}: BCD done in {elapsed:.1f}s")

    return {k: pd.DataFrame(v) for k, v in results_all.items()}, positions_all


# ================================================================
# MAIN
# ================================================================

def main():
    t_start = time.time()
    log("=" * 70)
    log("stage4b_hourly_wf.py — Walk-forward on hourly data (A/B/C/D)")
    log("=" * 70)

    # Load data
    log("Loading data...")
    hourly = pd.read_parquet(HOURLY_PATH)
    daily = pd.read_parquet(DAILY_PATH)
    vpred = pd.read_parquet(VPRED_PATH)
    kr = pd.read_parquet(KR_PATH)
    kr_s = kr.set_index("date")["key_rate"]
    kr_s.index = pd.to_datetime(kr_s.index)

    with open(CONFIGS_PATH) as f:
        best_configs = json.load(f)

    log(f"Hourly: {hourly.shape}, Daily: {daily.shape}, Vpred: {vpred.shape}")
    log(f"Configs: {len(best_configs)} strategies")
    log(f"WARMUP_H={WARMUP_H}, OOS_YEARS={OOS_YEARS}")
    log(f"BEST_SCALES={BEST_SCALES}")

    # Prepare ticker data
    log("Preparing ticker data...")
    ticker_data_list = []
    for ticker in TICKERS:
        td = prepare_ticker_data(ticker, hourly, daily, vpred)
        if td is not None:
            ticker_data_list.append(td)
            log(f"  {ticker}: {td['n']} hourly bars")
        else:
            log(f"  {ticker}: SKIPPED (too few bars)")
    log(f"Prepared {len(ticker_data_list)} tickers")

    # Free raw data
    del hourly
    gc.collect()

    # --- A ---
    comm = COMM_STD
    results_A, positions_A = run_walkforward_A(ticker_data_list, best_configs, kr_s, comm)

    # Save A
    if len(results_A) > 0:
        results_A.to_csv(OUT_DIR / "hourly_wf_A_results.csv", index=False)
        log(f"A results: {len(results_A)} rows, mean Sharpe(excess)={results_A['sharpe_excess'].mean():.3f}")
    if positions_A:
        pos_A_df = pd.DataFrame(positions_A)
        pos_A_df.to_parquet(OUT_DIR / "hourly_wf_positions_A.parquet", index=False)
        log(f"A positions: {len(pos_A_df)} rows")
    else:
        pos_A_df = pd.DataFrame()

    # --- BCD ---
    results_BCD, positions_BCD = run_walkforward_BCD(
        ticker_data_list, best_configs, kr_s, comm, positions_A)

    # Save BCD
    for letter in ["B", "C", "D"]:
        rdf = results_BCD[letter]
        if len(rdf) > 0:
            rdf.to_csv(OUT_DIR / f"hourly_wf_{letter}_results.csv", index=False)
            log(f"{letter} results: {len(rdf)} rows, mean Sharpe(excess)={rdf['sharpe_excess'].mean():.3f}")

        plist = positions_BCD[letter]
        if plist:
            pdf = pd.DataFrame(plist)
            pdf.to_parquet(OUT_DIR / f"hourly_wf_positions_{letter}.parquet", index=False)
            log(f"{letter} positions: {len(pdf)} rows")

    # Summary
    elapsed = time.time() - t_start
    log(f"DONE in {elapsed / 60:.1f} min")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Walk-Forward Hourly")
    print("=" * 80)
    for letter in ["A", "B", "C", "D"]:
        if letter == "A":
            rdf = results_A
        else:
            rdf = results_BCD[letter]
        if len(rdf) > 0:
            avg = rdf.groupby("strategy")["sharpe_excess"].mean()
            print(f"\n--- {letter} ---")
            for sid in STRATEGIES:
                if sid in avg.index:
                    print(f"  {sid}: avg Sharpe(excess) = {avg[sid]:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
