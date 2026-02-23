#!/usr/bin/env python3
"""
Extract 2026 positions from V4 pipeline.

The V4 pipeline calibrates for test_year=2025 using data up to start of 2026,
then generates positions for the FULL time series (including 2026 dates).
But the collection step only saves dates within [start_2025, start_2026).

This script re-runs test_year=2025 and extracts positions for 2026 dates.

Usage: python3 scripts/run_v4_2026_fix.py
"""
import os, sys, time, warnings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Patch year lists to only process 2025
import strategies_walkforward as swf
swf.A_TEST_YEARS = [2025]
swf.BCD_TEST_YEARS = [2025]

import strategies_walkforward_v4 as v4

# Ensure v4 functions see patched years
v4.A_TEST_YEARS = [2025]
v4.BCD_TEST_YEARS = [2025]
for attr in dir(v4):
    obj = getattr(v4, attr)
    if callable(obj) and hasattr(obj, '__globals__'):
        g = obj.__globals__
        if 'A_TEST_YEARS' in g:
            g['A_TEST_YEARS'] = [2025]
        if 'BCD_TEST_YEARS' in g:
            g['BCD_TEST_YEARS'] = [2025]

OUT_DIR = v4.OUT_DIR
POS_PATH = OUT_DIR / "daily_positions.parquet"

STRATEGIES = v4.STRATEGY_IDS
STRATEGY_NAMES = v4.STRATEGY_NAMES
TICKERS = v4.TICKERS
TIMEFRAMES = v4.TIMEFRAMES
COMM_DAILY = v4.COMM_DAILY
COMM_HOURLY = v4.COMM_HOURLY
BPY = 252


def main():
    t0 = time.time()
    print("=" * 70)
    print("V4 Pipeline — Extract 2026 positions from test_year=2025")
    print(f"A_TEST_YEARS = {swf.A_TEST_YEARS}")
    print(f"BCD_TEST_YEARS = {swf.BCD_TEST_YEARS}")
    print("=" * 70)

    daily, hourly, vpred = v4.load_data()

    # Verify 2026 data exists
    daily_dates = pd.to_datetime(daily["date"])
    n26 = (daily_dates >= "2026-01-01").sum()
    print(f"2026 daily rows in raw data: {n26}")
    if n26 == 0:
        print("ERROR: No 2026 data!")
        return

    print(f"\nWarming up numba...")
    v4.warmup_numba_v4()

    print(f"\nProcessing {len(TICKERS)} tickers (single-threaded for position extraction)...")

    all_position_rows_2026 = []

    for i, ticker in enumerate(TICKERS):
        # Process ticker (only test_year=2025)
        result = v4.process_ticker_v4(ticker, daily, hourly, vpred)
        elapsed = time.time() - t0

        # The result contains positions for test_year=2025 (dates in 2025 only)
        # But internally, approach functions generated position arrays for full date range.
        # We need to re-extract positions for 2026 dates.

        # Re-run approaches to get raw position arrays
        for tf in TIMEFRAMES:
            is_hourly = tf == "hourly"
            commission = COMM_HOURLY if is_hourly else COMM_DAILY
            ann = sqrt(252 * 9) if is_hourly else sqrt(252)
            bpy_tf = 252 * 9 if is_hourly else 252

            if is_hourly:
                tdf = hourly[hourly["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
                if len(tdf) == 0:
                    continue
                close = tdf["close"].values.astype(np.float64)
                high = tdf["high"].values.astype(np.float64)
                low = tdf["low"].values.astype(np.float64)
                open_arr = tdf["open"].values.astype(np.float64)
                volume = tdf["volume"].values.astype(np.float64)
                dates = pd.to_datetime(tdf["datetime"].values)
            else:
                tdf = daily[daily["ticker"] == ticker].sort_values("date").reset_index(drop=True)
                if len(tdf) == 0:
                    continue
                close = tdf["close"].values.astype(np.float64)
                high = tdf["high"].values.astype(np.float64)
                low = tdf["low"].values.astype(np.float64)
                open_arr = tdf["open"].values.astype(np.float64)
                volume = tdf["volume"].values.astype(np.float64)
                dates = pd.to_datetime(tdf["date"].values)

            n = len(close)
            log_ret = np.zeros(n)
            log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

            # Find 2026 date range
            ys_2026 = swf._year_bounds(dates, 2026)[0]
            ye_2026 = n  # to end of data

            if ys_2026 >= n:
                continue  # no 2026 data for this ticker/tf

            n_2026 = ye_2026 - ys_2026
            if n_2026 <= 0:
                continue

            # Precompute indicators
            ind = swf.compute_base(close, high, low, open_arr, volume, is_hourly)
            sma_windows = sorted(set(swf.SIGNAL_GRIDS["S1"].get("ma_window", []) +
                                     swf.SIGNAL_GRIDS["S2"].get("bb_window", [])))
            sma_cache = swf.precompute_sma_cache(close, sma_windows)
            dc_windows = swf.SIGNAL_GRIDS["S3"].get("dc_window", [])
            dc_cache = swf.precompute_donchian_cache(high, low, dc_windows)
            st_periods = swf.SIGNAL_GRIDS["S4"].get("atr_period", [])
            st_mults = swf.SIGNAL_GRIDS["S4"].get("multiplier", [])
            st_cache = swf.precompute_supertrend_cache(high, low, close, st_periods, st_mults)
            vwap_windows = swf.SIGNAL_GRIDS["S6"].get("vwap_window", [])
            vwap_cache = swf.precompute_vwap_cache(close, high, low, volume, vwap_windows)

            pivot_data = {}
            if is_hourly:
                h_dts = tdf["datetime"].values
                P, S1p, R1p = swf.compute_daily_pivots_for_hourly(daily, ticker, h_dts, "classic")
            else:
                P, S1p, R1p = swf.calc_pivot_daily(high, low, close, "classic")
            pivot_data["classic"] = (P, S1p, R1p)

            daily_trend = None
            if is_hourly:
                d_dates, d_above = swf.build_daily_trend(daily, ticker)
                daily_trend = swf.align_daily_to_hourly(d_dates, d_above, tdf["datetime"].values)

            sigma_dict = {}
            if vpred is not None:
                vp = vpred[vpred["ticker"] == ticker].sort_values("date")
                vp_dates = pd.to_datetime(vp["date"].values)
                for hname in ["h1", "h5", "h22"]:
                    col = f"sigma_{hname}"
                    if col not in vp.columns:
                        continue
                    vp_vals = vp[col].values
                    sigma_arr = np.full(n, np.nan)
                    if is_hourly:
                        bar_dates = pd.to_datetime(tdf["datetime"].values).normalize()
                    else:
                        bar_dates = pd.to_datetime(tdf["date"].values)
                    idx = np.searchsorted(vp_dates, bar_dates, side="right") - 1
                    valid = idx >= 0
                    sigma_arr[valid] = vp_vals[idx[valid]]
                    if is_hourly:
                        nhours = swf._compute_nhours_per_day(tdf["datetime"].values)
                        nhours_safe = np.maximum(nhours, 1.0)
                        sigma_arr = sigma_arr / np.sqrt(nhours_safe)
                    sigma_dict[hname] = sigma_arr

            for sid in STRATEGIES:
                sname = STRATEGY_NAMES[sid]

                # Approach A
                a_results, a_params, a_trades, a_exec_params = v4.approach_a_v4(
                    sid, tf, close, high, low, volume, open_arr, ind,
                    sma_cache, dc_cache, st_cache, vwap_cache,
                    pivot_data, daily_trend, dates, is_hourly, log_ret)

                # Approach B
                b_results, b_params, b_trades = v4.approach_b_v4(
                    sid, tf, close, high, low, volume, ind,
                    sma_cache, dc_cache, st_cache, vwap_cache,
                    pivot_data, daily_trend, dates, sigma_dict, is_hourly,
                    log_ret, a_params, a_exec_params)

                # Approach C
                c_results, c_params, c_trades = v4.approach_c_v4(
                    sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

                # Approach D
                d_results, d_params, d_trades = v4.approach_d_v4(
                    sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

                # Extract 2026 positions from each approach
                for approach, res in [("A", a_results), ("B", b_results),
                                       ("C", c_results), ("D", d_results)]:
                    # res is {test_year: pos_array}
                    pos_arr = res.get(2025)  # test_year=2025 contains full pos array
                    if pos_arr is None:
                        continue

                    for bar in range(ys_2026, ye_2026):
                        if bar < n:
                            all_position_rows_2026.append((
                                dates[bar], sname, tf, ticker, approach, 2026,
                                round(float(pos_arr[bar]), 6),
                                round(float(pos_arr[bar] * log_ret[bar]), 8),
                            ))

        print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {len(all_position_rows_2026)} total 2026 rows ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")
    print(f"Total 2026 position rows: {len(all_position_rows_2026)}")

    if all_position_rows_2026:
        pos_2026 = pd.DataFrame(all_position_rows_2026,
                                columns=["date", "strategy", "tf", "ticker", "approach",
                                         "test_year", "position", "daily_gross_return"])
        pos_2026_path = OUT_DIR / "daily_positions_2026.parquet"
        pos_2026.to_parquet(pos_2026_path, index=False)
        print(f"2026 positions saved: {len(pos_2026)} rows -> {pos_2026_path}")
        print(f"  Date range: {pos_2026['date'].min()} — {pos_2026['date'].max()}")
        print(f"  Strategies: {sorted(pos_2026['strategy'].unique())}")
        print(f"  Approaches: {sorted(pos_2026['approach'].unique())}")
        print(f"  TFs: {sorted(pos_2026['tf'].unique())}")

        # Merge with existing
        if POS_PATH.exists():
            pos_existing = pd.read_parquet(POS_PATH)
            pos_existing = pos_existing[pos_existing["test_year"] != 2026]
            pos_merged = pd.concat([pos_existing, pos_2026], ignore_index=True)

            backup_path = OUT_DIR / "daily_positions_pre2026.parquet"
            if not backup_path.exists():
                pd.read_parquet(POS_PATH).to_parquet(backup_path, index=False)
                print(f"  Backup saved: {backup_path}")

            pos_merged.to_parquet(POS_PATH, index=False)
            print(f"  Merged: {len(pos_merged)} rows (was {len(pos_existing)}, +{len(pos_2026)})")
            print(f"  Years: {sorted(pos_merged['test_year'].unique())}")
    else:
        print("  WARNING: No 2026 positions generated!")

    print(f"\nDone in {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
