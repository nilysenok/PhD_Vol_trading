#!/usr/bin/env python3
"""
run_hourly_bcd.py — Run B/C/D approaches on hourly data using NEW A positions.

Uses scaled signal grids (×9) and WARMUP_HOURLY=1800.
Loads A results from hourly_a_scaled, then runs B/C/D on top.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
import multiprocessing as mp
from multiprocessing import cpu_count

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

from strategies_walkforward import (
    TICKERS, STRATEGY_IDS, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, BCD_TEST_YEARS, WARMUP_HOURLY,
    SIGNAL_GRIDS_HOURLY, RM_GRIDS_V3, TREND_RM_GRID,
    expand_grid, load_data,
    compute_base, precompute_sma_cache, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    precompute_session_vwap,
    calc_pivot_daily, compute_daily_pivots_for_hourly,
    build_daily_trend, align_daily_to_hourly,
    dispatch_signals, dispatch_bt_atr_v3,
    calc_sharpe_comm, warmup_numba,
    _year_bounds, _store_metrics,
    _extract_trades_from_pos, _trades_to_rows,
    approach_a_one,
    approach_b_one, approach_c_one, approach_d_one,
    _compute_nhours_per_day,
    COMMISSION,
)

N_WORKERS = min(cpu_count(), 8)
OUT_DIR = Path("../results/final/strategies/walkforward_v4/hourly_a_scaled")


def process_ticker_hourly_bcd(ticker, daily_df, hourly_df, vpred_df):
    """Process one ticker: re-run A (scaled) then B/C/D on hourly."""
    results = {}
    position_rows = []

    tf = "hourly"
    is_hourly = True

    tdf = hourly_df[hourly_df["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
    if len(tdf) == 0:
        return {"results": results, "positions": position_rows}

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

    # Signal caches (hourly-scaled)
    _sg = SIGNAL_GRIDS_HOURLY
    sma_windows = sorted(set(_sg["S1"].get("ma_window", []) + _sg["S2"].get("bb_window", [])))
    sma_cache = precompute_sma_cache(close, sma_windows)
    dc_windows = _sg["S3"].get("dc_window", [])
    dc_cache = precompute_donchian_cache(high, low, dc_windows)
    st_periods = _sg["S4"].get("atr_period", [])
    st_mults = _sg["S4"].get("multiplier", [])
    st_cache = precompute_supertrend_cache(high, low, close, st_periods, st_mults)
    vwap_windows = [w for w in _sg["S6"].get("vwap_window", []) if w > 0]
    vwap_cache = precompute_vwap_cache(close, high, low, volume, vwap_windows)

    h_dts = tdf["datetime"].values
    session_vwap_data = precompute_session_vwap(close, high, low, volume, h_dts)

    pivot_data = {}
    for pv in ["classic", "camarilla", "fibonacci"]:
        P, S1p, R1p, S2p, R2p, S3p, R3p = compute_daily_pivots_for_hourly(daily_df, ticker, h_dts, pv)
        pivot_data[pv] = (P, S1p, R1p, S2p, R2p, S3p, R3p)

    d_dates, d_above = build_daily_trend(daily_df, ticker)
    daily_trend = align_daily_to_hourly(d_dates, d_above, h_dts)

    # Sigma dict
    sigma_dict = {}
    if vpred_df is not None:
        vp = vpred_df[vpred_df["ticker"] == ticker].sort_values("date")
        vp_dates = pd.to_datetime(vp["date"].values)
        for hname in ["h1", "h5", "h22"]:
            col = f"sigma_{hname}"
            if col not in vp.columns:
                continue
            vp_vals = vp[col].values
            sigma_arr = np.full(n, np.nan)
            bar_dates = pd.to_datetime(tdf["datetime"].values).normalize()
            idx = np.searchsorted(vp_dates, bar_dates, side="right") - 1
            valid = idx >= 0
            sigma_arr[valid] = vp_vals[idx[valid]]
            # v2_sqrtN for hourly
            nhours = _compute_nhours_per_day(tdf["datetime"].values)
            nhours_safe = np.maximum(nhours, 1.0)
            sigma_arr = sigma_arr / np.sqrt(nhours_safe)
            sigma_dict[hname] = sigma_arr

    ann = sqrt(252 * 9)
    bpy = 252 * 9

    for sid in STRATEGY_IDS:
        sname = STRATEGY_NAMES[sid]

        # Re-run A with scaled grids
        a_results, a_params, a_trades = approach_a_one(
            sid, tf, close, high, low, volume, open_arr, ind,
            sma_cache, dc_cache, st_cache, vwap_cache,
            pivot_data, daily_trend, dates, is_hourly, log_ret,
            session_vwap_data=session_vwap_data)

        # Store A metrics
        for year, pos_arr in a_results.items():
            _store_metrics(results, sid, tf, "A", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        # B
        b_results, b_params, b_trades = approach_b_one(
            sid, tf, close, high, low, volume, ind,
            sma_cache, dc_cache, st_cache, vwap_cache,
            pivot_data, daily_trend, dates, sigma_dict, is_hourly,
            log_ret, a_params, session_vwap_data=session_vwap_data)

        # C
        c_results, c_params, c_trades = approach_c_one(
            sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

        # D
        d_results, d_params, d_trades = approach_d_one(
            sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

        for approach, res in [("A", a_results), ("B", b_results),
                               ("C", c_results), ("D", d_results)]:
            for year, pos_arr in res.items():
                if approach != "A":
                    _store_metrics(results, sid, tf, approach, year, pos_arr,
                                   log_ret, dates, n, ann, bpy, ticker)

                # Collect positions
                test_years = A_TEST_YEARS if approach == "A" else BCD_TEST_YEARS
                if year not in test_years:
                    continue
                ys = _year_bounds(dates, year)[0]
                ye = _year_bounds(dates, year + 1)[0] if year < 2026 else n
                for bar in range(ys, ye):
                    if bar < n:
                        position_rows.append((
                            dates[bar], sname, tf, ticker, approach, year,
                            round(float(pos_arr[bar]), 6),
                            round(float(pos_arr[bar] * log_ret[bar]), 8),
                        ))

    return {"results": results, "positions": position_rows}


_G_DAILY = None
_G_HOURLY = None
_G_VPRED = None

def _pool_init(daily, hourly, vpred):
    global _G_DAILY, _G_HOURLY, _G_VPRED
    _G_DAILY = daily
    _G_HOURLY = hourly
    _G_VPRED = vpred

def _worker(ticker):
    return ticker, process_ticker_hourly_bcd(ticker, _G_DAILY, _G_HOURLY, _G_VPRED)


def main():
    t0 = time.time()
    print("=" * 70)
    print("Hourly A+B+C+D — Scaled Signal Grids")
    print(f"Tickers: {len(TICKERS)}, Strategies: {len(STRATEGY_IDS)}")
    print(f"Workers: {N_WORKERS}")
    print("=" * 70)

    daily, hourly, vpred = load_data()

    print(f"\nWarming up numba...")
    warmup_numba()

    print(f"\nProcessing {len(TICKERS)} tickers...")
    all_results = {}
    all_position_rows = []

    if N_WORKERS > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(N_WORKERS, initializer=_pool_init, initargs=(daily, hourly, vpred)) as pool:
            for i, (ticker, result_dict) in enumerate(pool.imap_unordered(_worker, TICKERS)):
                all_results[ticker] = result_dict["results"]
                all_position_rows.extend(result_dict["positions"])
                elapsed = time.time() - t0
                n_met = len(result_dict["results"])
                print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {n_met} metrics ({elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(TICKERS):
            result_dict = process_ticker_hourly_bcd(ticker, daily, hourly, vpred)
            all_results[ticker] = result_dict["results"]
            all_position_rows.extend(result_dict["positions"])
            elapsed = time.time() - t0
            n_met = len(result_dict["results"])
            print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {n_met} metrics ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker, ticker_results in all_results.items():
        for key, met in ticker_results.items():
            rows.append(met)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "hourly_abcd_results.csv", index=False)
        print(f"\nResults: {len(df)} rows → hourly_abcd_results.csv")

        # Summary
        common = df[df["year"].isin([2022, 2023, 2024, 2025])]
        if len(common) > 0:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            summary = common.groupby(["strategy", "approach"]).agg(
                sharpe=("sharpe", "mean"),
                ann_ret=("ann_ret_pct", "mean"),
                max_dd=("max_dd_pct", "mean"),
                exposure=("exposure_pct", "mean"),
            ).round(4)
            print(summary.to_string())

    # Positions
    if all_position_rows:
        pos_df = pd.DataFrame(all_position_rows,
                              columns=["date", "strategy", "tf", "ticker", "approach",
                                       "test_year", "position", "daily_gross_return"])
        pos_df.to_parquet(OUT_DIR / "hourly_abcd_positions.parquet", index=False)
        print(f"\nPositions: {len(pos_df):,} rows → hourly_abcd_positions.parquet")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
