#!/usr/bin/env python3
"""
Extract 2026 positions from V4 walk-forward pipeline (multiprocessing).

Strategy: run test_year=2025 (calibrate on data through end-2025),
then collect positions for ALL dates from start_2025 onward (including 2026)
by patching _year_bounds so year=2026 maps to (n, n).

After pipeline finishes, split out 2026 dates and save separately.

Usage: python3 scripts/run_v4_2026_mp.py
"""
import os, sys, time, warnings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Step 1: Patch year lists BEFORE importing V4 ──
import strategies_walkforward as swf
swf.A_TEST_YEARS = [2025]
swf.BCD_TEST_YEARS = [2025]

# ── Step 2: Patch _year_bounds so year>=2026 returns (n, n) ──
# This makes collection for test_year=2025 extend to include 2026 dates.
_orig_year_bounds = swf._year_bounds

def _patched_year_bounds(dates, year):
    if year >= 2026:
        n = len(dates)
        return n, n
    return _orig_year_bounds(dates, year)

swf._year_bounds = _patched_year_bounds

# ── Step 3: Import V4 (it imports _year_bounds from swf) ──
import strategies_walkforward_v4 as v4

# Patch year lists in V4 namespace
v4.A_TEST_YEARS = [2025]
v4.BCD_TEST_YEARS = [2025]

# Patch _year_bounds in V4 namespace and all function globals
v4._year_bounds = _patched_year_bounds
for attr in dir(v4):
    obj = getattr(v4, attr)
    if callable(obj) and hasattr(obj, '__globals__'):
        g = obj.__globals__
        if '_year_bounds' in g:
            g['_year_bounds'] = _patched_year_bounds
        if 'A_TEST_YEARS' in g:
            g['A_TEST_YEARS'] = [2025]
        if 'BCD_TEST_YEARS' in g:
            g['BCD_TEST_YEARS'] = [2025]

# Also patch in swf function globals
for attr in dir(swf):
    obj = getattr(swf, attr)
    if callable(obj) and hasattr(obj, '__globals__'):
        g = obj.__globals__
        if '_year_bounds' in g:
            g['_year_bounds'] = _patched_year_bounds
        if 'A_TEST_YEARS' in g:
            g['A_TEST_YEARS'] = [2025]
        if 'BCD_TEST_YEARS' in g:
            g['BCD_TEST_YEARS'] = [2025]


OUT_DIR = v4.OUT_DIR
POS_PATH = OUT_DIR / "daily_positions.parquet"


def main():
    t0 = time.time()
    print("=" * 70)
    print("V4 Pipeline — Extract 2026 positions (multiprocessing)")
    print(f"A_TEST_YEARS = {v4.A_TEST_YEARS}")
    print(f"BCD_TEST_YEARS = {v4.BCD_TEST_YEARS}")
    print(f"N_WORKERS = {v4.N_WORKERS}")
    print("=" * 70)

    # Verify patch
    test_dates = pd.to_datetime(["2025-06-01", "2025-12-01", "2026-01-05", "2026-02-01"])
    yb = _patched_year_bounds(test_dates, 2026)
    print(f"Patch check: _year_bounds(dates, 2026) = {yb} (should be ({len(test_dates)}, {len(test_dates)}))")

    daily, hourly, vpred = v4.load_data()

    daily_dates = pd.to_datetime(daily["date"])
    n26 = (daily_dates >= "2026-01-01").sum()
    print(f"2026 daily rows in raw data: {n26}")
    if n26 == 0:
        print("ERROR: No 2026 data!")
        return

    print(f"\nWarming up numba...")
    v4.warmup_numba_v4()

    print(f"\nProcessing {len(v4.TICKERS)} tickers with {v4.N_WORKERS} workers...")

    all_position_rows = []

    if v4.N_WORKERS > 1:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(v4.N_WORKERS, initializer=v4._pool_init_v4,
                       initargs=(daily, hourly, vpred)) as pool:
            for i, (ticker, result_dict) in enumerate(
                    pool.imap_unordered(v4._worker_func_v4, v4.TICKERS)):
                all_position_rows.extend(result_dict["positions"])
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(v4.TICKERS)}] {ticker} done "
                      f"({len(result_dict['positions'])} pos rows, {elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(v4.TICKERS):
            result_dict = v4.process_ticker_v4(ticker, daily, hourly, vpred)
            all_position_rows.extend(result_dict["positions"])
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(v4.TICKERS)}] {ticker} done "
                  f"({len(result_dict['positions'])} pos rows, {elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")
    print(f"Total position rows (2025+2026): {len(all_position_rows)}")

    if not all_position_rows:
        print("ERROR: No position rows generated!")
        return

    # Build DataFrame
    pos_all = pd.DataFrame(all_position_rows,
                           columns=["date", "strategy", "tf", "ticker", "approach",
                                    "test_year", "position", "daily_gross_return"])

    # Convert dates
    pos_all["date"] = pd.to_datetime(pos_all["date"])

    # Split: 2026 dates
    mask_2026 = pos_all["date"] >= "2026-01-01"
    pos_2026 = pos_all[mask_2026].copy()
    pos_2026["test_year"] = 2026

    print(f"\n2026 positions extracted: {len(pos_2026)} rows")
    if len(pos_2026) == 0:
        print("WARNING: No 2026 positions found!")
        return

    print(f"  Date range: {pos_2026['date'].min()} — {pos_2026['date'].max()}")
    print(f"  Strategies: {sorted(pos_2026['strategy'].unique())}")
    print(f"  Approaches: {sorted(pos_2026['approach'].unique())}")
    print(f"  TFs: {sorted(pos_2026['tf'].unique())}")
    print(f"  Tickers: {sorted(pos_2026['ticker'].unique())}")

    # Save 2026-only file
    pos_2026_path = OUT_DIR / "daily_positions_2026.parquet"
    pos_2026.to_parquet(pos_2026_path, index=False)
    print(f"\n2026 positions saved: {pos_2026_path}")

    # Merge with existing
    if POS_PATH.exists():
        pos_existing = pd.read_parquet(POS_PATH)
        n_before = len(pos_existing)
        pos_existing = pos_existing[pos_existing["test_year"] != 2026]
        pos_merged = pd.concat([pos_existing, pos_2026], ignore_index=True)

        # Backup
        backup_path = OUT_DIR / "daily_positions_pre2026.parquet"
        if not backup_path.exists():
            pd.read_parquet(POS_PATH).to_parquet(backup_path, index=False)
            print(f"  Backup saved: {backup_path}")

        pos_merged.to_parquet(POS_PATH, index=False)
        print(f"  Merged: {len(pos_merged)} rows (was {n_before}, +{len(pos_2026)} new)")
        print(f"  Test years: {sorted(pos_merged['test_year'].unique())}")
    else:
        print("  WARNING: No existing positions file to merge!")

    print(f"\nDone in {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
