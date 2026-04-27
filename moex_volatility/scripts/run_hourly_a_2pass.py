#!/usr/bin/env python3
"""
run_hourly_a_2pass.py — Fast hourly Approach A with scaled signal grids.

2-pass grid search:
  Pass 1: Coarse grid (every 2nd RM param) → top-5 (sig, rm) regions
  Pass 2: Fine-tune around top-5 regions → ensemble

Key changes vs original:
  - SIGNAL_GRIDS_HOURLY: windows ×9 (9 hourly bars per MOEX day)
  - WARMUP_HOURLY = 1800 (200 days × 9)
  - Only approach A, only hourly
  - 2-pass coarse→fine for ~3x speedup
"""

import os, time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
import multiprocessing as mp
from multiprocessing import cpu_count

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# Import everything from existing pipeline
from strategies_walkforward import (
    TICKERS, STRATEGY_IDS, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, WARMUP,
    RM_GRIDS_V3, TREND_RM_GRID,
    SIGNAL_GRIDS_HOURLY, WARMUP_HOURLY,
    expand_grid, load_data,
    compute_base, precompute_sma_cache, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    precompute_session_vwap,
    calc_pivot_daily, compute_daily_pivots_for_hourly,
    build_daily_trend, align_daily_to_hourly,
    dispatch_signals, dispatch_bt_atr_v3,
    calc_sharpe_comm, warmup_numba,
    _year_bounds, _store_metrics, _extract_trades_from_pos,
    _trades_to_rows, generate_outputs,
    COMMISSION, _pctile_cache,
)

N_WORKERS = min(cpu_count(), 8)
COARSE_STEP = 1  # full grid (no coarsening — fast enough at ~2min)
TOP_K_PASS1 = 10  # top regions from pass 1
FINE_RADIUS = 2   # expand ±2 around best RM indices


def coarsen_rm_grid(rm_grid, step=2):
    """Take every step-th element from RM grid."""
    return rm_grid[::step]


def fine_tune_rm(rm_grid, top_indices, radius=1):
    """Get RM params around top indices ± radius."""
    n = len(rm_grid)
    idx_set = set()
    for i in top_indices:
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            idx_set.add(j)
    return [rm_grid[i] for i in sorted(idx_set)]


def approach_a_hourly_2pass(sid, close, high, low, volume, open_arr, ind,
                            sma_cache, dc_cache, st_cache, vwap_cache,
                            pivot_data, daily_trend, dates, log_ret,
                            session_vwap_data=None):
    """Approach A for hourly with 2-pass grid search and scaled signal grids."""
    ann = sqrt(252 * 9)
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)
    warmup = WARMUP_HOURLY

    # Use HOURLY signal grids (×9 scaled)
    sig_grid = expand_grid(SIGNAL_GRIDS_HOURLY[sid])
    cat = CATEGORY[sid]
    if cat == "Trend":
        rm_grid_full = TREND_RM_GRID
    else:
        rm_grid_full = expand_grid(RM_GRIDS_V3[(cat, "hourly")])

    rm_grid_coarse = coarsen_rm_grid(rm_grid_full, COARSE_STEP)

    results_by_year = {}
    params_by_year = {}
    trades_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        if test_year == 2026:
            test_end = n
        else:
            _, test_end = _year_bounds(dates, test_year + 1)
        if train_end <= warmup or train_end >= n:
            continue

        train_mask[:] = False
        train_mask[warmup:train_end] = True

        # Cache all signals (once per year)
        cached_sigs = {}
        for sp in sig_grid:
            key = tuple(sorted(sp.items()))
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, dc_cache, st_cache, vwap_cache,
                                              pivot_data, daily_trend, sp, is_hourly=True,
                                              session_vwap_data=session_vwap_data)
            cached_sigs[key] = (sig, exit_info)

        # ── PASS 1: Coarse grid ──
        coarse_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            for rm_idx, rm in enumerate(rm_grid_coarse):
                # Map coarse index back to full index
                full_idx = rm_idx * COARSE_STEP
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, warmup, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                active = np.sum(pos[train_mask] != 0)
                exposure = active / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                if exposure > 1:  # pre-filter: skip zero-exposure combos
                    coarse_results.append((sp, full_idx, sh, exposure))

        # Sort by Sharpe, take top-K signal+region combos
        coarse_results.sort(key=lambda x: x[2], reverse=True)
        top_coarse = coarse_results[:TOP_K_PASS1 * 3]  # take more for diversity

        # Identify top RM regions
        top_rm_indices = list(set(r[1] for r in top_coarse[:TOP_K_PASS1]))
        top_sig_params = list(set(tuple(sorted(r[0].items())) for r in top_coarse[:TOP_K_PASS1 * 2]))

        # ── PASS 2: Fine-tune around top regions ──
        rm_grid_fine = fine_tune_rm(rm_grid_full, top_rm_indices, FINE_RADIUS)

        fine_results = []
        for sp_key_tuple in top_sig_params:
            if sp_key_tuple not in cached_sigs:
                continue
            sp = dict(sp_key_tuple)
            sig, exit_info = cached_sigs[sp_key_tuple]
            for rm in rm_grid_fine:
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, warmup, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                dpos = np.diff(pos[train_mask], prepend=0.0)
                net_r = dr - np.abs(dpos) * COMMISSION
                cum = np.cumsum(net_r)
                rmax = np.maximum.accumulate(cum)
                max_dd = (cum - rmax).min() if len(cum) > 0 else 0.0
                active = np.sum(pos[train_mask] != 0)
                exposure = active / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                fine_results.append((sp, rm, sh, max_dd, exposure))

        # Select top-10 for ensemble
        passing = [(sp, rm, sh, mdd, exp) for sp, rm, sh, mdd, exp in fine_results
                   if sh > 0 and mdd > -0.30 and exp > 3]
        if len(passing) == 0:
            passing = sorted(fine_results, key=lambda x: x[2], reverse=True)[:3]
        passing = sorted(passing, key=lambda x: x[2], reverse=True)[:10]

        # Run ensemble on test period
        test_positions = []
        test_trades = []
        for sp, rm, sh, mdd, exp in passing:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            pos, tr = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                         rm, warmup, test_end, 1)
            test_positions.append(pos)
            test_trades.append(tr)

        if test_positions:
            ensemble = np.mean(test_positions, axis=0)
        else:
            ensemble = np.zeros(n)

        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm) for sp, rm, sh, mdd, exp in passing]
        trades_by_year[test_year] = test_trades

    return results_by_year, params_by_year, trades_by_year


def process_ticker_hourly_a(ticker, daily_df, hourly_df, vpred_df):
    """Process one ticker: hourly approach A only, with scaled grids."""
    results = {}
    trade_rows = []
    position_rows = []

    tf = "hourly"
    is_hourly = True

    tdf = hourly_df[hourly_df["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
    if len(tdf) == 0:
        return {"results": results, "trades": trade_rows, "positions": position_rows}

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["datetime"].values)
    n = len(close)

    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    ind = compute_base(close, high, low, open_arr, volume, is_hourly)

    # Use HOURLY signal grids for cache precomputation
    sma_windows = sorted(set(
        SIGNAL_GRIDS_HOURLY["S1"].get("ma_window", []) +
        SIGNAL_GRIDS_HOURLY["S2"].get("bb_window", [])
    ))
    sma_cache = precompute_sma_cache(close, sma_windows)

    dc_windows = SIGNAL_GRIDS_HOURLY["S3"].get("dc_window", [])
    dc_cache = precompute_donchian_cache(high, low, dc_windows)

    st_periods = SIGNAL_GRIDS_HOURLY["S4"].get("atr_period", [])
    st_mults = SIGNAL_GRIDS_HOURLY["S4"].get("multiplier", [])
    st_cache = precompute_supertrend_cache(high, low, close, st_periods, st_mults)

    vwap_windows = [w for w in SIGNAL_GRIDS_HOURLY["S6"].get("vwap_window", []) if w > 0]
    vwap_cache = precompute_vwap_cache(close, high, low, volume, vwap_windows)

    # Session VWAP (daily reset) for S6
    h_dts = tdf["datetime"].values
    session_vwap_data = precompute_session_vwap(close, high, low, volume, h_dts)

    # Pivot (all variants) + daily trend
    pivot_data = {}
    for pv in ["classic", "camarilla", "fibonacci"]:
        P, S1p, R1p, S2p, R2p, S3p, R3p = compute_daily_pivots_for_hourly(daily_df, ticker, h_dts, pv)
        pivot_data[pv] = (P, S1p, R1p, S2p, R2p, S3p, R3p)

    d_dates, d_above = build_daily_trend(daily_df, ticker)
    daily_trend = align_daily_to_hourly(d_dates, d_above, h_dts)

    ann = sqrt(252 * 9)
    bpy = 252 * 9

    _pctile_cache.clear()  # reset signal strength cache per ticker

    for sid in STRATEGY_IDS:
        sname = STRATEGY_NAMES[sid]

        a_results, a_params, a_trades = approach_a_hourly_2pass(
            sid, close, high, low, volume, open_arr, ind,
            sma_cache, dc_cache, st_cache, vwap_cache,
            pivot_data, daily_trend, dates, log_ret,
            session_vwap_data=session_vwap_data)

        for year, pos_arr in a_results.items():
            _store_metrics(results, sid, tf, "A", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        for year, trades_list in a_trades.items():
            trade_rows.extend(_trades_to_rows(trades_list, sname, tf, ticker, "A", year, dates))

        for year in A_TEST_YEARS:
            pos_arr = a_results.get(year)
            if pos_arr is None:
                continue
            ys = _year_bounds(dates, year)[0]
            ye = _year_bounds(dates, year + 1)[0] if year < 2026 else n
            for bar in range(ys, ye):
                if bar < n:
                    position_rows.append((
                        dates[bar], sname, tf, ticker, "A", year,
                        round(float(pos_arr[bar]), 6),
                        round(float(pos_arr[bar] * log_ret[bar]), 8),
                    ))

    return {"results": results, "trades": trade_rows, "positions": position_rows}


# ── Pool worker ──
_G_DAILY = None
_G_HOURLY = None
_G_VPRED = None

def _pool_init(daily, hourly, vpred):
    global _G_DAILY, _G_HOURLY, _G_VPRED
    _G_DAILY = daily
    _G_HOURLY = hourly
    _G_VPRED = vpred

def _worker(ticker):
    return ticker, process_ticker_hourly_a(ticker, _G_DAILY, _G_HOURLY, _G_VPRED)


def main():
    t0 = time.time()
    print("=" * 70)
    print("Hourly Approach A — 2-Pass Grid Search (Scaled ×9 Windows)")
    print(f"Tickers: {len(TICKERS)}, Strategies: {len(STRATEGY_IDS)}")
    print(f"Test years: {A_TEST_YEARS[0]}-{A_TEST_YEARS[-1]}")
    print(f"WARMUP: {WARMUP_HOURLY} bars ({WARMUP_HOURLY//9} days)")
    print(f"Workers: {N_WORKERS}")
    print(f"2-pass: coarse step={COARSE_STEP}, top-K={TOP_K_PASS1}, fine radius={FINE_RADIUS}")
    print("=" * 70)

    daily, hourly, vpred = load_data()

    # Print grid sizes
    for sid in STRATEGY_IDS:
        sg = len(expand_grid(SIGNAL_GRIDS_HOURLY[sid])) if SIGNAL_GRIDS_HOURLY[sid] else 1
        cat = CATEGORY[sid]
        if cat == "Trend":
            rg_full = len(TREND_RM_GRID)
        else:
            rg_full = len(expand_grid(RM_GRIDS_V3[(cat, "hourly")]))
        rg_coarse = len(coarsen_rm_grid(list(range(rg_full)), COARSE_STEP))
        print(f"  {STRATEGY_NAMES[sid]}: {sg} sig × {rg_coarse}/{rg_full} RM (coarse/full)")

    print(f"\nWarming up numba...")
    warmup_numba()

    print(f"\nProcessing {len(TICKERS)} tickers...")
    all_results = {}
    all_trade_rows = []
    all_position_rows = []

    if N_WORKERS > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(N_WORKERS, initializer=_pool_init, initargs=(daily, hourly, vpred)) as pool:
            for i, (ticker, result_dict) in enumerate(pool.imap_unordered(_worker, TICKERS)):
                all_results[ticker] = result_dict["results"]
                all_trade_rows.extend(result_dict["trades"])
                all_position_rows.extend(result_dict["positions"])
                elapsed = time.time() - t0
                n_met = len(result_dict["results"])
                print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {n_met} metrics ({elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(TICKERS):
            result_dict = process_ticker_hourly_a(ticker, daily, hourly, vpred)
            all_results[ticker] = result_dict["results"]
            all_trade_rows.extend(result_dict["trades"])
            all_position_rows.extend(result_dict["positions"])
            elapsed = time.time() - t0
            n_met = len(result_dict["results"])
            print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {n_met} metrics ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")

    # Save results
    OUT_DIR = Path("../results/final/strategies/walkforward_v4/hourly_a_scaled")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Results CSV
    rows = []
    for ticker, ticker_results in all_results.items():
        for key, met in ticker_results.items():
            rows.append(met)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "hourly_a_scaled_results.csv", index=False)
        print(f"\nResults: {len(df)} rows → {OUT_DIR / 'hourly_a_scaled_results.csv'}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY: Hourly A (scaled ×9 windows)")
        print(f"{'='*70}")
        common = df[df["year"].isin([2022, 2023, 2024, 2025])]
        if len(common) > 0:
            agg_dict = {"sharpe": ("sharpe", "mean"), "ann_ret": ("ann_ret_pct", "mean"),
                        "max_dd": ("max_dd_pct", "mean"), "exposure": ("exposure_pct", "mean"),
                        "win_rate": ("win_rate_pct", "mean")}
            if "n_trades" in common.columns:
                agg_dict["trades"] = ("n_trades", "mean")
            summary = common.groupby("strategy").agg(**agg_dict).round(4)
            print(summary.to_string())

    # Positions parquet
    if all_position_rows:
        pos_df = pd.DataFrame(all_position_rows,
                              columns=["date", "strategy", "tf", "ticker", "approach",
                                       "test_year", "position", "daily_gross_return"])
        pos_df.to_parquet(OUT_DIR / "hourly_a_scaled_positions.parquet", index=False)
        print(f"Positions: {len(pos_df):,} rows → hourly_a_scaled_positions.parquet")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
