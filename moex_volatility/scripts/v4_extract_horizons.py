#!/usr/bin/env python3
"""Extract horizon selections from V4 approaches B, C, D.
Re-runs the pipeline (daily only) capturing params that were discarded."""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
import multiprocessing as mp

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

BASE = SCRIPTS_DIR.parent
DATA_DIR = BASE / "data" / "prepared"
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"

# Import from V4 and base modules
from strategies_walkforward import (
    TICKERS, STRATEGY_IDS, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, BCD_TEST_YEARS,
    expand_grid, SIGNAL_GRIDS, B_GRIDS_V3,
    compute_base, precompute_sma_cache, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    calc_pivot_daily,
    load_data,
)

from strategies_walkforward_v4 import (
    COMM_DAILY,
    approach_a_v4, approach_b_v4, approach_c_v4, approach_d_v4,
    _compute_nhours_per_day,
)

from strategies_walkforward_v4_1 import (
    approach_a_s5_v41, approach_a_s6_v41,
    approach_b_s5_v41, approach_b_s6_v41,
    approach_c_v41,
    prepare_pivot_cache, prepare_vwap_cache_ext,
)

_G_DAILY = None
_G_VPRED = None


def extract_horizons_ticker(ticker):
    """Extract horizon selections for one ticker, daily TF only."""
    daily_df = _G_DAILY
    vpred_df = _G_VPRED
    tf = "daily"
    is_hourly = False

    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) == 0:
        return ticker, []

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)  # DatetimeIndex
    n = len(close)

    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    ind = compute_base(close, high, low, open_arr, volume, False)

    # Caches
    sma_windows = sorted(set(SIGNAL_GRIDS["S1"].get("ma_window", []) +
                             SIGNAL_GRIDS["S2"].get("bb_window", [])))
    sma_cache = precompute_sma_cache(close, sma_windows)
    dc_windows = SIGNAL_GRIDS["S3"].get("dc_window", [])
    dc_cache = precompute_donchian_cache(high, low, dc_windows)
    st_periods = SIGNAL_GRIDS["S4"].get("atr_period", [])
    st_mults = SIGNAL_GRIDS["S4"].get("multiplier", [])
    st_cache = precompute_supertrend_cache(high, low, close, st_periods, st_mults)
    vwap_windows = SIGNAL_GRIDS["S6"].get("vwap_window", [])
    vwap_cache_basic = precompute_vwap_cache(close, high, low, volume, vwap_windows)

    P, S1p, R1p = calc_pivot_daily(high, low, close, "classic")
    pivot_data = {"classic": (P, S1p, R1p)}
    daily_trend = None

    # sigma_dict
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
            bar_dates = pd.to_datetime(tdf["date"].values)
            idx = np.searchsorted(vp_dates, bar_dates, side="right") - 1
            valid = idx >= 0
            sigma_arr[valid] = vp_vals[idx[valid]]
            sigma_dict[hname] = sigma_arr

    horizon_records = []

    # ──── S1-S4: V4 approach A/B/C/D ────
    for sid in ["S1", "S2", "S3", "S4"]:
        sname = STRATEGY_NAMES[sid]
        try:
            a_results, a_params, _, a_exec_params = approach_a_v4(
                sid, tf, close, high, low, volume, open_arr, ind,
                sma_cache, dc_cache, st_cache, vwap_cache_basic,
                pivot_data, daily_trend, dates, is_hourly, log_ret)
        except Exception as e:
            print(f"    {ticker} {sname} A: {e}")
            continue

        # B
        try:
            _, b_params, _ = approach_b_v4(
                sid, tf, close, high, low, volume, ind,
                sma_cache, dc_cache, st_cache, vwap_cache_basic,
                pivot_data, daily_trend, dates, sigma_dict, is_hourly,
                log_ret, a_params, a_exec_params)
            for year, bp in b_params.items():
                horizon_records.append({
                    "strategy": sname, "approach": "B",
                    "ticker": ticker, "test_year": year,
                    "horizon": bp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} {sname} B: {e}")

        # C (V4.1 relaxed for S1/S2, V4 for S3/S4)
        try:
            if sid in ("S1", "S2"):
                _, c_params, _ = approach_c_v41(
                    sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)
            else:
                _, c_params, _ = approach_c_v4(
                    sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)
            for year, cp in c_params.items():
                horizon_records.append({
                    "strategy": sname, "approach": "C",
                    "ticker": ticker, "test_year": year,
                    "horizon": cp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} {sname} C: {e}")

        # D
        try:
            _, d_params, _ = approach_d_v4(
                sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)
            for year, dp in d_params.items():
                horizon_records.append({
                    "strategy": sname, "approach": "D",
                    "ticker": ticker, "test_year": year,
                    "horizon": dp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} {sname} D: {e}")

    # ──── S5: V4.1 approach A/B/C/D ────
    try:
        pivot_cache = prepare_pivot_cache(daily_df, ticker, dates, is_hourly, tdf)

        s5_a, s5_a_params, _, s5_a_exec = approach_a_s5_v41(
            tf, close, high, low, open_arr, volume, ind,
            dates, is_hourly, log_ret, pivot_cache)

        # S5 B
        try:
            _, s5_b_params, _ = approach_b_s5_v41(
                tf, close, high, low, volume, ind, dates, sigma_dict,
                is_hourly, log_ret, s5_a_params, s5_a_exec, pivot_cache)
            for year, bp in s5_b_params.items():
                horizon_records.append({
                    "strategy": "S5_PivotPoints", "approach": "B",
                    "ticker": ticker, "test_year": year,
                    "horizon": bp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} S5 B: {e}")

        # S5 C
        try:
            _, s5_c_params, _ = approach_c_v41(
                "S5", tf, dates, s5_a, sigma_dict, is_hourly, log_ret, n)
            for year, cp in s5_c_params.items():
                horizon_records.append({
                    "strategy": "S5_PivotPoints", "approach": "C",
                    "ticker": ticker, "test_year": year,
                    "horizon": cp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} S5 C: {e}")

        # S5 D
        try:
            _, s5_d_params, _ = approach_d_v4(
                "S5", tf, dates, s5_a, sigma_dict, is_hourly, log_ret, n)
            for year, dp in s5_d_params.items():
                horizon_records.append({
                    "strategy": "S5_PivotPoints", "approach": "D",
                    "ticker": ticker, "test_year": year,
                    "horizon": dp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} S5 D: {e}")

    except Exception as e:
        print(f"    {ticker} S5 A: {e}")

    # ──── S6: V4.1 approach A/B/C/D ────
    try:
        vwap_cache_ext, session_vwap_data, dir_bias_arr = prepare_vwap_cache_ext(
            close, high, low, volume, is_hourly, tdf,
            daily_df, ticker)

        s6_a, s6_a_params, _, s6_a_exec = approach_a_s6_v41(
            tf, close, high, low, open_arr, volume, ind,
            dates, is_hourly, log_ret,
            vwap_cache_ext, session_vwap_data, dir_bias_arr)

        # S6 B
        try:
            _, s6_b_params, _ = approach_b_s6_v41(
                tf, close, high, low, volume, ind, dates, sigma_dict,
                is_hourly, log_ret, s6_a_params, s6_a_exec,
                vwap_cache_ext, session_vwap_data, dir_bias_arr)
            for year, bp in s6_b_params.items():
                horizon_records.append({
                    "strategy": "S6_VWAP", "approach": "B",
                    "ticker": ticker, "test_year": year,
                    "horizon": bp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} S6 B: {e}")

        # S6 C
        try:
            _, s6_c_params, _ = approach_c_v41(
                "S6", tf, dates, s6_a, sigma_dict, is_hourly, log_ret, n)
            for year, cp in s6_c_params.items():
                horizon_records.append({
                    "strategy": "S6_VWAP", "approach": "C",
                    "ticker": ticker, "test_year": year,
                    "horizon": cp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} S6 C: {e}")

        # S6 D
        try:
            _, s6_d_params, _ = approach_d_v4(
                "S6", tf, dates, s6_a, sigma_dict, is_hourly, log_ret, n)
            for year, dp in s6_d_params.items():
                horizon_records.append({
                    "strategy": "S6_VWAP", "approach": "D",
                    "ticker": ticker, "test_year": year,
                    "horizon": dp.get("horizon", "?")})
        except Exception as e:
            print(f"    {ticker} S6 D: {e}")

    except Exception as e:
        print(f"    {ticker} S6 A: {e}")

    return ticker, horizon_records


def _pool_init(daily, vpred):
    global _G_DAILY, _G_VPRED
    _G_DAILY = daily
    _G_VPRED = vpred


def main():
    t0 = time.time()
    print("=" * 70)
    print("V4 Horizon Extraction (daily only, B/C/D)")
    print("=" * 70)

    daily, _, vpred = load_data()

    # Numba warmup
    print("Numba warmup...")
    from strategies_walkforward_v4 import warmup_numba_v4
    warmup_numba_v4()
    print("  done")

    n_workers = min(8, len(TICKERS))
    print(f"\nProcessing {len(TICKERS)} tickers with {n_workers} workers...")

    ctx = mp.get_context("fork")
    all_records = []

    with ctx.Pool(n_workers, initializer=_pool_init, initargs=(daily, vpred)) as pool:
        for i, (ticker, records) in enumerate(
                pool.imap_unordered(extract_horizons_ticker, TICKERS)):
            all_records.extend(records)
            print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {len(records)} horizon records ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(all_records)
    # Filter to BCD years only (2022-2025)
    df = df[df["test_year"].isin([2022, 2023, 2024, 2025])].copy()
    print(f"\nTotal records (BCD 2022-2025): {len(df)}")

    strategies_order = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
                        "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]

    # ── TABLE 1: Strategy × Approach ──
    print(f"\n{'='*75}")
    print(f"  HORIZON SELECTION: Strategy x Approach (% of ticker-year selections)")
    print(f"{'='*75}")
    print(f"{'Strategy':>16s} {'App':>4s} | {'h1%':>6s} {'h5%':>6s} {'h22%':>6s} | {'N':>4s} | {'mode':>4s}")
    print(f"{'-'*16}-{'-'*4}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*4}-+-{'-'*4}")

    prev_strat = ""
    for strat in strategies_order:
        for appr in ["B", "C", "D"]:
            sub = df[(df["strategy"] == strat) & (df["approach"] == appr)]
            total = len(sub)
            s_disp = strat if strat != prev_strat else ""
            prev_strat = strat
            if total == 0:
                print(f"{s_disp:>16s} {appr:>4s} |   N/A    N/A    N/A |    0 |  N/A")
                continue
            h1 = (sub["horizon"] == "h1").sum() / total * 100
            h5 = (sub["horizon"] == "h5").sum() / total * 100
            h22 = (sub["horizon"] == "h22").sum() / total * 100
            mode = sub["horizon"].mode().iloc[0]
            print(f"{s_disp:>16s} {appr:>4s} | {h1:5.1f}  {h5:5.1f}  {h22:5.1f}  | {total:4d} |  {mode:>3s}")

    # ── TABLE 2: By approach ──
    print(f"\n{'='*55}")
    print(f"  AGGREGATE BY APPROACH")
    print(f"{'='*55}")
    print(f"{'Approach':>10s} | {'h1%':>6s} {'h5%':>6s} {'h22%':>6s} | {'N':>5s}")
    print(f"{'-'*10}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*5}")
    for appr in ["B", "C", "D"]:
        sub = df[df["approach"] == appr]
        total = len(sub)
        if total == 0:
            continue
        h1 = (sub["horizon"] == "h1").sum() / total * 100
        h5 = (sub["horizon"] == "h5").sum() / total * 100
        h22 = (sub["horizon"] == "h22").sum() / total * 100
        print(f"{appr:>10s} | {h1:5.1f}  {h5:5.1f}  {h22:5.1f}  | {total:5d}")

    # ── TABLE 3: By strategy ──
    print(f"\n{'='*55}")
    print(f"  AGGREGATE BY STRATEGY")
    print(f"{'='*55}")
    print(f"{'Strategy':>16s} | {'h1%':>6s} {'h5%':>6s} {'h22%':>6s} | {'N':>5s}")
    print(f"{'-'*16}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*5}")
    for strat in strategies_order:
        sub = df[df["strategy"] == strat]
        total = len(sub)
        if total == 0:
            continue
        h1 = (sub["horizon"] == "h1").sum() / total * 100
        h5 = (sub["horizon"] == "h5").sum() / total * 100
        h22 = (sub["horizon"] == "h22").sum() / total * 100
        print(f"{strat:>16s} | {h1:5.1f}  {h5:5.1f}  {h22:5.1f}  | {total:5d}")

    # Save
    df.to_csv(OUT_DIR / "tables" / "v4_horizon_selections.csv", index=False)
    print(f"\nSaved: v4_horizon_selections.csv")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
