#!/usr/bin/env python3
"""
Run V4 walk-forward pipeline for test_year=2026 ONLY.

Monkey-patches year constants so the full pipeline only processes 2026.
Saves 2026 positions separately, then merges with existing positions.

Usage: python3 scripts/run_v4_2026.py
"""
import sys, time
from pathlib import Path

# Patch year lists BEFORE importing V4 pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategies_walkforward as swf
swf.A_TEST_YEARS = [2026]
swf.BCD_TEST_YEARS = [2026]

import strategies_walkforward_v4 as v4
v4.A_TEST_YEARS = [2026]
v4.BCD_TEST_YEARS = [2026]

# Also patch in the imported namespace
import importlib
# Force the V4 module to see patched years
for attr in dir(v4):
    obj = getattr(v4, attr)
    if callable(obj) and hasattr(obj, '__globals__'):
        if 'A_TEST_YEARS' in obj.__globals__:
            obj.__globals__['A_TEST_YEARS'] = [2026]
        if 'BCD_TEST_YEARS' in obj.__globals__:
            obj.__globals__['BCD_TEST_YEARS'] = [2026]

import numpy as np
import pandas as pd

OUT_DIR = v4.OUT_DIR
POS_PATH = OUT_DIR / "daily_positions.parquet"
POS_PATH_2026 = OUT_DIR / "daily_positions_2026.parquet"


def main():
    t0 = time.time()
    print("=" * 70)
    print("V4 Pipeline — 2026 ONLY")
    print(f"A_TEST_YEARS = {swf.A_TEST_YEARS}")
    print(f"BCD_TEST_YEARS = {swf.BCD_TEST_YEARS}")
    print("=" * 70)

    daily, hourly, vpred = v4.load_data()

    # Check 2026 data exists
    daily_dates = pd.to_datetime(daily["date"])
    n26 = (daily_dates >= "2026-01-01").sum()
    print(f"2026 daily rows in raw data: {n26}")
    if n26 == 0:
        print("ERROR: No 2026 data found!")
        return

    print(f"\nWarming up numba...")
    v4.warmup_numba_v4()

    print(f"\nProcessing {len(v4.TICKERS)} tickers with {v4.N_WORKERS} workers...")

    all_results = {}
    all_trade_rows = []
    all_position_rows = []

    if v4.N_WORKERS > 1:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(v4.N_WORKERS, initializer=v4._pool_init_v4,
                       initargs=(daily, hourly, vpred)) as pool:
            for i, (ticker, result_dict) in enumerate(
                    pool.imap_unordered(v4._worker_func_v4, v4.TICKERS)):
                all_results[ticker] = result_dict["results"]
                all_trade_rows.extend(result_dict["trades"])
                all_position_rows.extend(result_dict["positions"])
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(v4.TICKERS)}] {ticker} done ({elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(v4.TICKERS):
            result_dict = v4.process_ticker_v4(ticker, daily, hourly, vpred)
            all_results[ticker] = result_dict["results"]
            all_trade_rows.extend(result_dict["trades"])
            all_position_rows.extend(result_dict["positions"])
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(v4.TICKERS)}] {ticker} done ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")

    # Save 2026-only positions
    if all_position_rows:
        pos_2026 = pd.DataFrame(all_position_rows,
                                columns=["date", "strategy", "tf", "ticker", "approach",
                                         "test_year", "position", "daily_gross_return"])
        pos_2026.to_parquet(POS_PATH_2026, index=False)
        print(f"\n2026 positions saved: {len(pos_2026)} rows -> {POS_PATH_2026}")
        print(f"  Strategies: {sorted(pos_2026['strategy'].unique())}")
        print(f"  Approaches: {sorted(pos_2026['approach'].unique())}")
        print(f"  TFs: {sorted(pos_2026['tf'].unique())}")
        print(f"  Date range: {pos_2026['date'].min()} — {pos_2026['date'].max()}")

        # Merge with existing positions
        if POS_PATH.exists():
            pos_existing = pd.read_parquet(POS_PATH)
            # Remove any existing 2026 data (shouldn't be any, but safety)
            pos_existing = pos_existing[pos_existing["test_year"] != 2026]
            pos_merged = pd.concat([pos_existing, pos_2026], ignore_index=True)
            # Save backup first
            backup_path = OUT_DIR / "daily_positions_backup.parquet"
            pd.read_parquet(POS_PATH).to_parquet(backup_path, index=False)
            print(f"  Backup saved: {backup_path}")
            # Save merged
            pos_merged.to_parquet(POS_PATH, index=False)
            print(f"  Merged positions saved: {len(pos_merged)} rows "
                  f"(was {len(pos_existing)}, added {len(pos_2026)})")
        else:
            print("  WARNING: No existing positions file to merge with!")
    else:
        print("  WARNING: No position rows generated!")

    print(f"\nDone in {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
