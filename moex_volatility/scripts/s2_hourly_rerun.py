#!/usr/bin/env python3
"""
s2_hourly_rerun.py — Re-run S2 Bollinger hourly with expanded parameter grid.

Changes from V4 baseline:
  - bb_window = [15, 25, 50, 100, 150, 200] (was [15, 20, 25])
  - bb_std = [1.0, 1.5, 2.0, 2.5] (was [1.5, 2.0, 2.5])
  - 24 signal combos (was 9) x 108 RM = 2,592 total per ticker/year
  - Calibration commission: 0.04% per side (was 0.35%)
  - Reporting commissions: 0.04%, 0.05% per side (was 0.35%, 0.45%)

Steps:
  1. Process S2 hourly with expanded grid for all 17 tickers
  2. Update daily_positions.parquet (replace S2_Bollinger hourly rows)
  3. Rebuild v4_full_hourly.csv with new commissions (all strategies)
  4. Rebuild v4_S2_hourly.csv
"""
import os, sys, time, warnings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ═══════════════════════════════════════════════════════════════════
# Patch SIGNAL_GRIDS["S2"] BEFORE importing V4 (which reads it)
# ═══════════════════════════════════════════════════════════════════
from strategies_walkforward import SIGNAL_GRIDS

S2_EXPANDED = dict(
    bb_window=[15, 25, 50, 100, 150, 200],
    bb_std=[1.0, 1.5, 2.0, 2.5],
)
_ORIG_S2 = SIGNAL_GRIDS["S2"].copy()
SIGNAL_GRIDS["S2"] = S2_EXPANDED

# Patch calibration commission
import strategies_walkforward_v4 as v4mod
v4mod.COMM_HOURLY = 0.0004  # 0.04% per side

from strategies_walkforward_v4 import (
    approach_a_v4, approach_b_v4, approach_c_v4, approach_d_v4,
    calc_metrics_v4, _store_metrics_v4,
    warmup_numba_v4,
    RM_GRIDS_V4, EXEC_GRID_HOURLY,
    EXIT_REASON_NAMES,
)
from strategies_walkforward import (
    load_data, TICKERS, WARMUP, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, BCD_TEST_YEARS,
    expand_grid, compute_base, precompute_sma_cache,
    compute_daily_pivots_for_hourly,
    build_daily_trend, align_daily_to_hourly,
    _year_bounds, _compute_nhours_per_day,
    _extract_trades_from_pos,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
OUT_TABLES = OUT_DIR / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

COMM_CALIBRATION = 0.0004  # 0.04% per side
COMM_REPORT_HOURLY = [0.0, 0.0004, 0.0005]  # gross, 0.04%, 0.05%

SID = "S2"
TF = "hourly"
SNAME = STRATEGY_NAMES[SID]  # "S2_Bollinger"

BCD_YEARS = [2022, 2023, 2024, 2025]
ALL_STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
                  "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
ALL_APPROACHES = ["A", "B", "C", "D"]


# ═══════════════════════════════════════════════════════════════════
# Process one ticker — S2 hourly only
# ═══════════════════════════════════════════════════════════════════
def process_ticker_s2_hourly(ticker, daily_df, hourly_df, vpred_df):
    """Process one ticker for S2 Bollinger hourly only."""
    results = {}
    position_rows = []
    is_hourly = True
    commission = COMM_CALIBRATION

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

    # SMA cache with EXPANDED windows
    sma_windows = sorted(set(
        SIGNAL_GRIDS["S1"].get("ma_window", []) +
        SIGNAL_GRIDS["S2"].get("bb_window", [])
    ))
    sma_cache = precompute_sma_cache(close, sma_windows)

    # Other caches — not needed for S2, pass empty
    dc_cache = {}
    st_cache = {}
    vwap_cache = {}

    # Pivots (needed by dispatch functions)
    pivot_data = {}
    h_dts = tdf["datetime"].values
    P, S1p, R1p = compute_daily_pivots_for_hourly(daily_df, ticker, h_dts, "classic")
    pivot_data["classic"] = (P, S1p, R1p)

    # Daily trend
    d_dates, d_above = build_daily_trend(daily_df, ticker)
    daily_trend = align_daily_to_hourly(d_dates, d_above, tdf["datetime"].values)

    # Sigma predictions (v2_sqrtN for hourly)
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
            dates_arr = tdf["datetime"].values
            nhours = _compute_nhours_per_day(dates_arr)
            nhours_safe = np.maximum(nhours, 1.0)
            sigma_arr = sigma_arr / np.sqrt(nhours_safe)
            sigma_dict[hname] = sigma_arr

    ann = sqrt(252 * 9)
    bpy = 252 * 9

    # ── Approach A ──
    a_results, a_params, a_trades, a_exec_params = approach_a_v4(
        SID, TF, close, high, low, volume, open_arr, ind,
        sma_cache, dc_cache, st_cache, vwap_cache,
        pivot_data, daily_trend, dates, is_hourly, log_ret)

    for year, pos_arr in a_results.items():
        _store_metrics_v4(results, SID, TF, "A", year, pos_arr,
                          log_ret, dates, n, ann, bpy, ticker, commission)

    # ── Approach B ──
    b_results, b_params, b_trades = approach_b_v4(
        SID, TF, close, high, low, volume, ind,
        sma_cache, dc_cache, st_cache, vwap_cache,
        pivot_data, daily_trend, dates, sigma_dict, is_hourly,
        log_ret, a_params, a_exec_params)

    # ── Approach C ──
    c_results, c_params, c_trades = approach_c_v4(
        SID, TF, dates, a_results, sigma_dict, is_hourly, log_ret, n)
    for year, pos_arr in c_results.items():
        test_start_c = _year_bounds(dates, year)[0]
        test_end_c = _year_bounds(dates, year + 1)[0] if year < 2026 else n
        c_trades[year] = _extract_trades_from_pos(pos_arr, close, test_start_c, test_end_c)

    # ── Approach D ──
    d_results, d_params, d_trades = approach_d_v4(
        SID, TF, dates, a_results, sigma_dict, is_hourly, log_ret, n)
    for year, pos_arr in d_results.items():
        test_start_d = _year_bounds(dates, year)[0]
        test_end_d = _year_bounds(dates, year + 1)[0] if year < 2026 else n
        d_trades[year] = _extract_trades_from_pos(pos_arr, close, test_start_d, test_end_d)

    # Store B/C/D metrics
    for approach, res in [("B", b_results), ("C", c_results), ("D", d_results)]:
        for year, pos_arr in res.items():
            _store_metrics_v4(results, SID, TF, approach, year, pos_arr,
                              log_ret, dates, n, ann, bpy, ticker, commission)

    # Collect position rows
    for approach, res, test_years in [
        ("A", a_results, A_TEST_YEARS),
        ("B", b_results, BCD_TEST_YEARS),
        ("C", c_results, BCD_TEST_YEARS),
        ("D", d_results, BCD_TEST_YEARS),
    ]:
        for year in test_years:
            pos_arr = res.get(year)
            if pos_arr is None:
                continue
            ys = _year_bounds(dates, year)[0]
            ye = _year_bounds(dates, year + 1)[0] if year < 2026 else n
            for bar in range(ys, ye):
                if bar < n:
                    position_rows.append((
                        dates[bar], SNAME, TF, ticker, approach, year,
                        round(float(pos_arr[bar]), 6),
                        round(float(pos_arr[bar] * log_ret[bar]), 8),
                    ))

    return {"results": results, "positions": position_rows}


# ═══════════════════════════════════════════════════════════════════
# Table builders
# ═══════════════════════════════════════════════════════════════════
def compute_metrics_report(grp, commission, is_hourly):
    """Compute Sharpe, AnnReturn, Trades/yr for a ticker-year group."""
    pos = grp["position"].values
    gross_r = grp["daily_gross_return"].values
    n = len(pos)
    if n == 0:
        return None
    dpos = np.diff(pos, prepend=0.0)
    comm_cost = np.abs(dpos) * commission
    net_r = gross_r - comm_cost
    bpy = 252 * 9 if is_hourly else 252
    ann = np.sqrt(bpy)
    std_r = np.std(net_r, ddof=1) if n > 1 else 1e-10
    sharpe = np.mean(net_r) / std_r * ann if std_r > 1e-12 else 0.0
    ann_ret = np.mean(net_r) * bpy
    trades = 0
    in_t = False
    for i in range(n):
        if pos[i] != 0 and not in_t:
            trades += 1
            in_t = True
        if in_t and pos[i] == 0:
            in_t = False
    years = max(n / bpy, 0.5)
    tpy = trades / years
    return {"sharpe": sharpe, "ann_ret": ann_ret * 100, "tpy": tpy}


def build_full_hourly_table(pos_df, comm_levels):
    """Build full hourly table for ALL strategies with given commission levels."""
    is_hourly = True
    sub = pos_df[(pos_df["tf"] == "hourly") & (pos_df["test_year"].isin(BCD_YEARS))].copy()

    rows = []
    for strat in ALL_STRATEGIES:
        for appr in ALL_APPROACHES:
            chunk = sub[(sub["strategy"] == strat) & (sub["approach"] == appr)]
            if len(chunk) == 0:
                rows.append({
                    "Strategy": strat, "App": appr,
                    "GrossSharpe": np.nan,
                    f"Net{comm_levels[1]*100:.2f}Sharpe": np.nan,
                    f"Net{comm_levels[2]*100:.2f}Sharpe": np.nan,
                    "Tr/yr": np.nan,
                    "GrossRet%": np.nan,
                    f"Net{comm_levels[1]*100:.2f}Ret%": np.nan,
                    f"Net{comm_levels[2]*100:.2f}Ret%": np.nan,
                })
                continue

            results_by_comm = {c: {"sharpes": [], "rets": [], "tpys": []}
                               for c in comm_levels}

            for ticker in sorted(chunk["ticker"].unique()):
                for year in BCD_YEARS:
                    grp = chunk[(chunk["ticker"] == ticker) & (chunk["test_year"] == year)]
                    if len(grp) == 0:
                        continue
                    for c in comm_levels:
                        m = compute_metrics_report(grp, c, is_hourly)
                        if m is not None:
                            results_by_comm[c]["sharpes"].append(m["sharpe"])
                            results_by_comm[c]["rets"].append(m["ann_ret"])
                            results_by_comm[c]["tpys"].append(m["tpy"])

            def safe_mean(lst):
                return np.mean(lst) if lst else np.nan

            gross_c, net1_c, net2_c = comm_levels
            row = {
                "Strategy": strat,
                "App": appr,
                "GrossSharpe": round(safe_mean(results_by_comm[gross_c]["sharpes"]), 4),
                f"Net{net1_c*100:.2f}Sharpe": round(safe_mean(results_by_comm[net1_c]["sharpes"]), 4),
                f"Net{net2_c*100:.2f}Sharpe": round(safe_mean(results_by_comm[net2_c]["sharpes"]), 4),
                "Tr/yr": round(safe_mean(results_by_comm[gross_c]["tpys"]), 2),
                "GrossRet%": round(safe_mean(results_by_comm[gross_c]["rets"]), 2),
                f"Net{net1_c*100:.2f}Ret%": round(safe_mean(results_by_comm[net1_c]["rets"]), 2),
                f"Net{net2_c*100:.2f}Ret%": round(safe_mean(results_by_comm[net2_c]["rets"]), 2),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def build_s2_detail_table(all_results):
    """Build S2 hourly detail table from raw results (same format as v4_S2_hourly.csv)."""
    rows = []
    for ticker, ticker_results in all_results.items():
        for key, met in ticker_results.items():
            row = {k: v for k, v in met.items() if k != "_net_returns"}
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregate: year mean across tickers, then approach mean across years
    agg = df.groupby(["approach", "year"]).agg(
        sharpe=("sharpe", "mean"),
        ann_ret_pct=("ann_ret_pct", "mean"),
        ann_vol_pct=("ann_vol_pct", "mean"),
        max_dd_pct=("max_dd_pct", "mean"),
        exposure_pct=("exposure_pct", "mean"),
        n_trades=("n_trades", "mean"),
        win_rate_pct=("win_rate_pct", "mean"),
    ).reset_index()

    summary = agg.groupby("approach").agg(
        MeanSharpe=("sharpe", "mean"),
        AnnReturn=("ann_ret_pct", "mean"),
        AnnVol=("ann_vol_pct", "mean"),
        MaxDD=("max_dd_pct", "mean"),
        Exposure=("exposure_pct", "mean"),
        TradesPerYr=("n_trades", "mean"),
        WinRate=("win_rate_pct", "mean"),
    ).reset_index().round(4)

    return summary


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 70)
    print("S2 Bollinger Hourly Re-run — Expanded Parameter Grid")
    print(f"Signal grid: bb_window={S2_EXPANDED['bb_window']}, bb_std={S2_EXPANDED['bb_std']}")
    sig_combos = len(expand_grid(S2_EXPANDED))
    rm_combos = len(expand_grid(RM_GRIDS_V4[("Contrarian", "hourly")]))
    print(f"Grid: {sig_combos} signal x {rm_combos} RM = {sig_combos * rm_combos} combos")
    print(f"Calibration commission: {COMM_CALIBRATION*100:.2f}% per side")
    print(f"Report commissions: {[f'{c*100:.2f}%' for c in COMM_REPORT_HOURLY]}")
    print(f"Tickers: {len(TICKERS)}")
    print(f"A test years: {A_TEST_YEARS}")
    print(f"BCD test years: {BCD_TEST_YEARS}")
    print("=" * 70)

    daily, hourly, vpred = load_data()

    print("\nWarming up numba...")
    warmup_numba_v4()

    print(f"\nProcessing {len(TICKERS)} tickers (S2 hourly only)...")
    all_results = {}
    all_position_rows = []

    for i, ticker in enumerate(TICKERS):
        result = process_ticker_s2_hourly(ticker, daily, hourly, vpred)
        all_results[ticker] = result["results"]
        all_position_rows.extend(result["positions"])
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({elapsed:.0f}s)")

    # ── Build new S2 positions DataFrame ──
    new_pos_df = pd.DataFrame(
        all_position_rows,
        columns=["date", "strategy", "tf", "ticker", "approach",
                 "test_year", "position", "daily_gross_return"]
    )
    print(f"\nNew S2 hourly positions: {len(new_pos_df):,} rows")

    # ── Update daily_positions.parquet ──
    pos_parquet = OUT_DIR / "daily_positions.parquet"
    if pos_parquet.exists():
        existing = pd.read_parquet(pos_parquet)
        print(f"Existing positions: {len(existing):,} rows")
        # Remove old S2 hourly rows
        mask_remove = (existing["strategy"] == SNAME) & (existing["tf"] == "hourly")
        n_removed = mask_remove.sum()
        existing = existing[~mask_remove]
        print(f"Removed {n_removed:,} old S2 hourly rows -> {len(existing):,} remaining")
        # Add new
        merged = pd.concat([existing, new_pos_df], ignore_index=True)
        print(f"After adding new S2 hourly: {len(merged):,} rows")
    else:
        print("WARNING: daily_positions.parquet not found, using only new data")
        merged = new_pos_df

    merged.to_parquet(pos_parquet, index=False)
    print(f"Saved: {pos_parquet}")

    # ── Rebuild v4_full_hourly.csv with new commissions ──
    print("\nRebuilding v4_full_hourly.csv with new commissions...")
    full_hourly = build_full_hourly_table(merged, COMM_REPORT_HOURLY)
    full_hourly.to_csv(OUT_TABLES / "v4_full_hourly.csv", index=False)
    print(f"Saved: {OUT_TABLES / 'v4_full_hourly.csv'}")
    print("\n  === Full Hourly Table (new commissions: 0.04% / 0.05%) ===")
    for _, r in full_hourly.iterrows():
        cols = list(full_hourly.columns)
        print(f"  {r['Strategy']:>16s} {r['App']:>1s}  "
              f"Gross={r[cols[2]]:7.4f}  "
              f"Net04={r[cols[3]]:7.4f}  "
              f"Net05={r[cols[4]]:7.4f}  "
              f"Tr/yr={r[cols[5]]:5.2f}  "
              f"GrossR={r[cols[6]]:6.2f}%")

    # ── Rebuild v4_S2_hourly.csv ──
    print("\nRebuilding v4_S2_hourly.csv...")
    s2_detail = build_s2_detail_table(all_results)
    if not s2_detail.empty:
        s2_detail.to_csv(OUT_TABLES / "v4_S2_hourly.csv", index=False)
        print(f"Saved: {OUT_TABLES / 'v4_S2_hourly.csv'}")
        print(s2_detail.to_string(index=False))

    # ── Verification ──
    print("\n  === VERIFICATION ===")
    s2_rows = full_hourly[full_hourly["Strategy"] == "S2_Bollinger"]
    if len(s2_rows) > 0:
        a_row = s2_rows[s2_rows["App"] == "A"]
        if len(a_row) > 0:
            gross = a_row.iloc[0]["GrossSharpe"]
            cols = list(full_hourly.columns)
            net04 = a_row.iloc[0][cols[3]]
            tpy = a_row.iloc[0]["Tr/yr"]
            print(f"  S2 hourly A: GrossSharpe={gross:.4f} (was 0.0627)")
            print(f"  S2 hourly A: Net0.04Sharpe={net04:.4f}")
            print(f"  S2 hourly A: Trades/yr={tpy:.2f} (was 1.41)")
            if gross > 0.063:
                print(f"  PASS: GrossSharpe improved ({gross:.4f} > 0.063)")
            else:
                print(f"  WARN: GrossSharpe NOT improved ({gross:.4f} <= 0.063)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"S2 HOURLY RE-RUN DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
