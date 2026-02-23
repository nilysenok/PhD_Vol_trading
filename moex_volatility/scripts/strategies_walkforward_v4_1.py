#!/usr/bin/env python3
"""
strategies_walkforward_v4_1.py — Selective Fix for V4

Fixes:
  1. S5 — Full pivot support (weekly/monthly, fibonacci, relaxed filters, pivot SL/TP)
  2. S6 — Full VWAP support (filter_set 1-4, dir_bias, session VWAP, VWAP-band SL/TP)
  3. C  — Relaxed constraints + final fallback to A
  4. MIN_TRADES — Reject combos with <1 trade/yr

Re-runs: S5 daily/hourly (A+B+C+D), S6 daily/hourly (A+B+C+D),
         S1 hourly (C only), S2 hourly (A+B+C+D with MIN_TRADES).
Merges results with V4 positions parquet.
"""
import os, warnings, time, sys
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import multiprocessing as mp
from multiprocessing import cpu_count
from math import sqrt

warnings.filterwarnings("ignore")

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator
    print("WARNING: numba not installed")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ═══════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════
from strategies_walkforward import (
    bt_range_v3, bt_range_vpred_v3,
    gen_s1_signals, gen_s2_signals,
    compute_base, precompute_sma_cache, precompute_vwap_cache,
    calc_pivot_daily, compute_daily_pivots_for_hourly,
    build_daily_trend, align_daily_to_hourly,
    dispatch_signals, dispatch_bt_atr_v3, dispatch_bt_vpred_v3,
    SIGNAL_GRIDS, B_GRIDS_V3, expand_grid,
    C_THRESHOLDS, C_RANGE_BANDS, C_LOOKBACKS, C_HORIZONS, C_DIRECTION,
    C_TERM_FILTER, C_HYSTERESIS,
    D_TARGET_VOLS, D_MAX_LEVS, D_GAMMA, D_VOL_FLOORS, D_VOL_CAPS,
    D_SMOOTH, D_HORIZONS, D_INV_LOOKBACKS,
    _year_bounds, _compute_nhours_per_day, _apply_hysteresis,
    _build_c_mask_v3, _compute_d_scale_v3, _quick_exposure_trades,
    _extract_trades_from_pos,
    load_data,
    TICKERS, STRATEGY_IDS, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, BCD_TEST_YEARS, WARMUP, TIMEFRAMES,
    RM_GRIDS_V3,
)

from s5_rerun import (
    bt_range_pivot_v3, precompute_pivot_sl_tp, gen_s5_signals_v2,
    calc_pivot_weekly, calc_pivot_monthly,
    calc_pivot_daily_ext, map_daily_pivots_to_hourly,
    precompute_pivot_base_distances,
    S5_SIG_DAILY, S5_RM_PIVOT_DAILY, S5_RM_ATR_DAILY,
    S5_RM_PIVOT_HOURLY, S5_RM_ATR_HOURLY,
)

from s5s6_rerun import (
    gen_s5_signals_v3, compute_session_vwap, gen_s6_signals_v2,
    bt_range_vwap_v3, precompute_vwap_sl_tp,
    precompute_vwap_cache_ext,
    map_monthly_pivots_to_hourly, map_weekly_pivots_to_hourly,
    S5H_SIG, S6D_SIG, S6H_SIG,
    S5H_RM_PIVOT, S5H_RM_ATR,
    S6_RM_ATR_DAILY, S6_RM_ATR_HOURLY,
    S6_RM_VWAP_DAILY, S6_RM_VWAP_HOURLY,
)

from strategies_walkforward_v4 import (
    calc_sharpe_v4, calc_metrics_v4, compute_max_dd, compute_exposure,
    execution_layer, EXEC_GRID_DAILY, EXEC_GRID_HOURLY,
    signal_strength_s1, signal_strength_s2, signal_strength_s5, signal_strength_s6,
    RM_GRIDS_V4, approach_b_v4, approach_d_v4, _store_metrics_v4,
    COMM_DAILY, COMM_HOURLY, D_GATE_THRESHOLDS,
    get_rm_neighbors, dispatch_strength,
    _trades_to_rows_v4,
)

# ═══════════════════════════════════════════════════════════════════
# §0. Constants
# ═══════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
V4_POS_PATH = OUT_DIR / "daily_positions.parquet"

N_WORKERS = min(cpu_count(), 8)

# Fix 4: MIN_TRADES constraint
MIN_TRADES_PER_YEAR = 1.0

# Fix 3: Relaxed C constraints
C_MIN_EXPOSURE_V41 = 2.0
C_MIN_TRADES_YR_V41 = 3.0
C_FALLBACK_EXPOSURE_V41 = 1.0
C_FALLBACK_TRADES_YR_V41 = 1.0
C_THRESHOLDS_V41 = [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)]
C_RANGE_BANDS_V41 = list(C_RANGE_BANDS) + [(0.45, 0.55)]


# ═══════════════════════════════════════════════════════════════════
# §1. Pivot Cache
# ═══════════════════════════════════════════════════════════════════

def prepare_pivot_cache(daily_df, ticker, dates, is_hourly, tdf):
    """Precompute all pivot types needed for S5."""
    cache = {}
    if is_hourly:
        h_dts = tdf["datetime"].values
        for ptype in ["classic", "fibonacci"]:
            cache[("weekly", ptype)] = map_weekly_pivots_to_hourly(
                daily_df, ticker, h_dts, ptype)
            cache[("monthly", ptype)] = map_monthly_pivots_to_hourly(
                daily_df, ticker, h_dts, ptype)
    else:
        # For daily: calc_pivot_weekly/monthly return (P, S1, S2, R1, R2, dates_arr)
        # But these are aligned to daily_df bars, not necessarily to our ticker's bars.
        # We need to re-align via date matching.
        tdf_daily = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        n = len(tdf_daily)
        for ptype in ["classic", "fibonacci"]:
            pw_result = calc_pivot_weekly(daily_df, ticker, ptype)
            # pw_result = (P, S1, S2, R1, R2, pw_dates)
            # Already aligned to ticker's daily bars by calc_pivot_weekly
            cache[("weekly", ptype)] = pw_result[:5]
            pm_result = calc_pivot_monthly(daily_df, ticker, ptype)
            cache[("monthly", ptype)] = pm_result[:5]
    return cache


# ═══════════════════════════════════════════════════════════════════
# §2. Extended VWAP Cache
# ═══════════════════════════════════════════════════════════════════

def prepare_vwap_cache_ext(close, high, low, volume, is_hourly, tdf,
                            daily_df=None, ticker=None):
    """Prepare extended VWAP cache + session VWAP + dir_bias."""
    vwap_windows = [5, 10, 20, 60]
    vwap_cache = precompute_vwap_cache_ext(close, high, low, volume, vwap_windows)

    session_vwap_data = None
    if is_hourly:
        session_vwap_data = compute_session_vwap(close, high, low, volume,
                                                  tdf["datetime"].values)

    dir_bias_arr = None
    if not is_hourly and daily_df is not None and ticker is not None:
        n = len(close)
        d_dates, d_above = build_daily_trend(daily_df, ticker)
        d_dates_np = np.array(d_dates, dtype="datetime64[D]")
        dates = pd.to_datetime(tdf["date"].values)
        our_dates = np.array(dates.values, dtype="datetime64[D]")
        idx = np.searchsorted(d_dates_np, our_dates, side="right") - 1
        valid = idx >= 0
        dir_bias_arr = np.full(n, np.nan)
        dir_bias_arr[valid] = d_above[idx[valid]]

    return vwap_cache, session_vwap_data, dir_bias_arr


# ═══════════════════════════════════════════════════════════════════
# §3. Approach A — S5 V4.1 (Coarse-to-fine + Exec layer)
# ═══════════════════════════════════════════════════════════════════

def approach_a_s5_v41(tf, close, high, low, open_arr, volume, ind,
                      dates, is_hourly, log_ret, pivot_cache):
    """Approach A for S5 with full pivot support + V4 exec layer."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    bpy = 252 * 9 if is_hourly else 252
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)

    sig_grid_def = S5H_SIG if is_hourly else S5_SIG_DAILY
    sig_grid = expand_grid(sig_grid_def)
    rm_pivot_grid = expand_grid(S5H_RM_PIVOT if is_hourly else S5_RM_PIVOT_DAILY)
    rm_atr_grid = expand_grid(S5H_RM_ATR if is_hourly else S5_RM_ATR_DAILY)

    # Combined RM grid for coarse-to-fine (tag with type)
    full_rm = []
    for rm in rm_pivot_grid:
        d = dict(rm); d["_rm_type"] = "pivot"; full_rm.append(d)
    for rm in rm_atr_grid:
        d = dict(rm); d["_rm_type"] = "atr"; full_rm.append(d)

    results_by_year = {}
    params_by_year = {}
    trades_by_year = {}
    exec_params_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        if train_end <= WARMUP or train_end >= n:
            continue
        train_mask[:] = False
        train_mask[WARMUP:train_end] = True

        # Cache signals & base distances
        cached_sigs = {}
        base_dist_cache = {}
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            if sp_key in cached_sigs:
                continue
            piv_key = (sp["pivot_period"], sp["pivot_type"])
            P, S1, S2, R1, R2 = pivot_cache[piv_key]

            if is_hourly:
                sig, exit_arr = gen_s5_signals_v3(
                    close, ind, P, S1, R1, sp["filter_set"],
                    sp["confirmation"], sp["entry_buffer"])
            else:
                sig, exit_arr = gen_s5_signals_v2(
                    close, ind, P, S1, R1, sp["filter_set"])

            if piv_key not in base_dist_cache:
                base_dist_cache[piv_key] = precompute_pivot_base_distances(P, S1, S2, R1, R2)
            cached_sigs[sp_key] = (sig, exit_arr, piv_key)

        # Cache strength (S5: distance to nearest level)
        cached_strength = {}
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            if sp_key in cached_strength:
                continue
            piv_key = (sp["pivot_period"], sp["pivot_type"])
            P, S1, S2, R1, R2 = pivot_cache[piv_key]
            span = np.abs(R1 - S1)
            dist = np.minimum(np.abs(close - S1), np.abs(close - R1))
            span_safe = np.where(np.isnan(span) | (span < 1e-12), 1e-12, span)
            dist_safe = np.where(np.isnan(dist), 0.0, dist)
            cached_strength[sp_key] = np.minimum(dist_safe / span_safe, 1.0)

        # ──── PHASE 1: Coarse grid ────
        coarse_rm = full_rm[::2] if len(full_rm) > 20 else full_rm
        coarse_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, piv_key = cached_sigs[sp_key]
            base_dist = base_dist_cache[piv_key]

            for rm in coarse_rm:
                pos = _run_s5_bt(rm, sig, exit_arr, close, high, low, ind,
                                 base_dist, train_end)
                sh = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                mdd = compute_max_dd(pos, log_ret, train_mask, commission)
                exp = compute_exposure(pos, train_mask)
                # MIN_TRADES check
                _, tpy = _quick_exposure_trades(pos, train_mask, bpy)
                coarse_results.append((sp, rm, sh, mdd, exp, tpy))

        passing = [r for r in coarse_results
                   if r[2] > 0 and r[3] > -0.30 and r[4] > 5 and r[5] >= MIN_TRADES_PER_YEAR]
        if not passing:
            passing = sorted(coarse_results, key=lambda x: x[2], reverse=True)[:10]
        top50 = sorted(passing, key=lambda x: x[2], reverse=True)[:50]

        # ──── PHASE 2: Fine neighbors ────
        fine_results = []
        seen = set()
        for sp, rm, *_ in top50:
            rm_idx = _find_rm_index(rm, full_rm)
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, piv_key = cached_sigs[sp_key]
            base_dist = base_dist_cache[piv_key]

            for delta in [-2, -1, 1, 2]:
                idx = rm_idx + delta
                if idx < 0 or idx >= len(full_rm):
                    continue
                rm_n = full_rm[idx]
                combo_key = (sp_key, str(sorted(rm_n.items())))
                if combo_key in seen:
                    continue
                seen.add(combo_key)
                pos = _run_s5_bt(rm_n, sig, exit_arr, close, high, low, ind,
                                  base_dist, train_end)
                sh = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                mdd = compute_max_dd(pos, log_ret, train_mask, commission)
                exp = compute_exposure(pos, train_mask)
                fine_results.append((sp, rm_n, sh, mdd, exp, 0))

        all_phase2 = top50 + fine_results
        dedup = {}
        for sp, rm, sh, mdd, exp, *_ in all_phase2:
            key = (tuple(sorted(sp.items())), str(sorted(rm.items())))
            if key not in dedup or sh > dedup[key][2]:
                dedup[key] = (sp, rm, sh, mdd, exp)
        top20 = sorted(dedup.values(), key=lambda x: x[2], reverse=True)[:20]

        # ──── PHASE 3: Execution layer ────
        exec_grid = expand_grid(EXEC_GRID_HOURLY if is_hourly else EXEC_GRID_DAILY)
        cached_raw_pos = {}
        for sp, rm, *_ in top20:
            sp_key = tuple(sorted(sp.items()))
            rm_key = str(sorted(rm.items()))
            if (sp_key, rm_key) not in cached_raw_pos:
                sig, exit_arr, piv_key = cached_sigs[sp_key]
                base_dist = base_dist_cache[piv_key]
                pos_raw = _run_s5_bt(rm, sig, exit_arr, close, high, low, ind,
                                      base_dist, train_end)
                cached_raw_pos[(sp_key, rm_key)] = pos_raw

        best_combos = []
        for sp, rm, sh_raw, mdd, exp in top20:
            sp_key = tuple(sorted(sp.items()))
            rm_key = str(sorted(rm.items()))
            pos_raw = cached_raw_pos[(sp_key, rm_key)]
            strength = cached_strength[sp_key]
            for ep in exec_grid:
                pos_exec = execution_layer(pos_raw, strength,
                                           ep["consec_entry"], ep["min_hold"],
                                           ep["cooldown_bars"], ep["min_strength"])
                sh_exec = calc_sharpe_v4(pos_exec, log_ret, train_mask, ann, commission)
                best_combos.append((sp, rm, ep, sh_exec))

        top10 = sorted(best_combos, key=lambda x: x[3], reverse=True)[:10]
        if not top10:
            continue

        # ──── Build ensemble ────
        test_positions = []
        test_trades = []
        for sp, rm, ep, sh in top10:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, piv_key = cached_sigs[sp_key]
            base_dist = base_dist_cache[piv_key]
            pos_raw = _run_s5_bt(rm, sig, exit_arr, close, high, low, ind,
                                  base_dist, test_end, log_trades=1)
            strength = cached_strength[sp_key]
            pos_exec = execution_layer(pos_raw, strength,
                                       ep["consec_entry"], ep["min_hold"],
                                       ep["cooldown_bars"], ep["min_strength"])
            test_positions.append(pos_exec)

        # Majority vote
        pos_stack = np.array(test_positions)
        ens_mag = np.mean(np.abs(pos_stack), axis=0)
        ens_dir = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir

        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm, ep) for sp, rm, ep, sh in top10]
        exec_params_by_year[test_year] = top10[0][2] if top10 else {}

    return results_by_year, params_by_year, {}, exec_params_by_year


def _run_s5_bt(rm, sig, exit_arr, close, high, low, ind, base_dist, end_idx,
               log_trades=0):
    """Run S5 backtest for either pivot or ATR RM."""
    if rm["_rm_type"] == "pivot":
        sl_dist, tp_dist = precompute_pivot_sl_tp(
            close, sig, base_dist,
            rm["sl_target"], rm["tp_target"], rm["buffer"])
        be = 0.5 if rm["breakeven"] else -1.0
        pos, tr = bt_range_pivot_v3(
            sig, exit_arr, close, high, low, ind["adx14"],
            sl_dist, tp_dist, 30.0, int(rm["max_hold"]),
            be, rm["cooldown_bars"], WARMUP, end_idx, log_trades)
    else:
        be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
        td = 1 if rm.get("time_decay") else 0
        pos, tr = bt_range_v3(
            sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
            rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
            be, rm["cooldown_bars"], td, WARMUP, end_idx, log_trades)
    return pos


def _find_rm_index(rm, rm_list):
    """Find index of rm dict in rm_list."""
    rm_str = str(sorted(rm.items()))
    for i, r in enumerate(rm_list):
        if str(sorted(r.items())) == rm_str:
            return i
    return 0


# ═══════════════════════════════════════════════════════════════════
# §4. Approach A — S6 V4.1 (Coarse-to-fine + Exec layer)
# ═══════════════════════════════════════════════════════════════════

def approach_a_s6_v41(tf, close, high, low, open_arr, volume, ind,
                      dates, is_hourly, log_ret,
                      vwap_cache, session_vwap_data, dir_bias_arr):
    """Approach A for S6 with full VWAP support + V4 exec layer."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    bpy = 252 * 9 if is_hourly else 252
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)

    sig_grid_def = S6H_SIG if is_hourly else S6D_SIG
    sig_grid = expand_grid(sig_grid_def)
    rm_atr_grid = expand_grid(S6_RM_ATR_HOURLY if is_hourly else S6_RM_ATR_DAILY)
    rm_vwap_grid = expand_grid(S6_RM_VWAP_HOURLY if is_hourly else S6_RM_VWAP_DAILY)

    full_rm = []
    for rm in rm_atr_grid:
        d = dict(rm); d["_rm_type"] = "atr"; full_rm.append(d)
    for rm in rm_vwap_grid:
        d = dict(rm); d["_rm_type"] = "vwap"; full_rm.append(d)

    results_by_year = {}
    params_by_year = {}
    exec_params_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        if train_end <= WARMUP or train_end >= n:
            continue
        train_mask[:] = False
        train_mask[WARMUP:train_end] = True

        # Cache signals
        cached_sigs = {}
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            if sp_key in cached_sigs:
                continue
            w = sp["vwap_window"]; dm = sp["dev_mult"]; fs = sp["filter_set"]

            if is_hourly:
                vtype = sp["vwap_type"]
                if vtype == "session":
                    vwap, dev = session_vwap_data
                else:
                    vwap, dev = vwap_cache[w]
                dba = None
            else:
                dba = dir_bias_arr if sp["dir_bias"] == "ON" else None
                vwap, dev = vwap_cache[w]

            sig, exit_arr = gen_s6_signals_v2(close, ind, vwap, dev, dm, fs, dba)
            cached_sigs[sp_key] = (sig, exit_arr, vwap, dev)

        # Cache strength
        cached_strength = {}
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            if sp_key in cached_strength:
                continue
            w = sp.get("vwap_window", 10)
            if w in vwap_cache:
                vwap, dev = vwap_cache[w]
            else:
                vwap, dev = list(vwap_cache.values())[0]
            dev_safe = np.maximum(dev, 1e-12)
            vwap_safe = np.where(np.isnan(vwap), close, vwap)
            cached_strength[sp_key] = np.minimum(
                np.abs(close - vwap_safe) / (2.0 * dev_safe), 1.0)

        # ──── PHASE 1: Coarse grid ────
        coarse_rm = full_rm[::2] if len(full_rm) > 20 else full_rm
        coarse_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, vwap, dev = cached_sigs[sp_key]

            for rm in coarse_rm:
                pos = _run_s6_bt(rm, sig, exit_arr, close, high, low, ind,
                                 vwap, dev, train_end)
                sh = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                mdd = compute_max_dd(pos, log_ret, train_mask, commission)
                exp = compute_exposure(pos, train_mask)
                _, tpy = _quick_exposure_trades(pos, train_mask, bpy)
                coarse_results.append((sp, rm, sh, mdd, exp, tpy))

        passing = [r for r in coarse_results
                   if r[2] > 0 and r[3] > -0.30 and r[4] > 5 and r[5] >= MIN_TRADES_PER_YEAR]
        if not passing:
            passing = sorted(coarse_results, key=lambda x: x[2], reverse=True)[:10]
        top50 = sorted(passing, key=lambda x: x[2], reverse=True)[:50]

        # ──── PHASE 2: Fine neighbors ────
        fine_results = []
        seen = set()
        for sp, rm, *_ in top50:
            rm_idx = _find_rm_index(rm, full_rm)
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, vwap, dev = cached_sigs[sp_key]

            for delta in [-2, -1, 1, 2]:
                idx = rm_idx + delta
                if idx < 0 or idx >= len(full_rm):
                    continue
                rm_n = full_rm[idx]
                combo_key = (sp_key, str(sorted(rm_n.items())))
                if combo_key in seen:
                    continue
                seen.add(combo_key)
                pos = _run_s6_bt(rm_n, sig, exit_arr, close, high, low, ind,
                                  vwap, dev, train_end)
                sh = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                mdd = compute_max_dd(pos, log_ret, train_mask, commission)
                exp = compute_exposure(pos, train_mask)
                fine_results.append((sp, rm_n, sh, mdd, exp, 0))

        all_phase2 = top50 + fine_results
        dedup = {}
        for sp, rm, sh, mdd, exp, *_ in all_phase2:
            key = (tuple(sorted(sp.items())), str(sorted(rm.items())))
            if key not in dedup or sh > dedup[key][2]:
                dedup[key] = (sp, rm, sh, mdd, exp)
        top20 = sorted(dedup.values(), key=lambda x: x[2], reverse=True)[:20]

        # ──── PHASE 3: Execution layer ────
        exec_grid = expand_grid(EXEC_GRID_HOURLY if is_hourly else EXEC_GRID_DAILY)
        cached_raw_pos = {}
        for sp, rm, *_ in top20:
            sp_key = tuple(sorted(sp.items()))
            rm_key = str(sorted(rm.items()))
            if (sp_key, rm_key) not in cached_raw_pos:
                sig, exit_arr, vwap, dev = cached_sigs[sp_key]
                pos_raw = _run_s6_bt(rm, sig, exit_arr, close, high, low, ind,
                                      vwap, dev, train_end)
                cached_raw_pos[(sp_key, rm_key)] = pos_raw

        best_combos = []
        for sp, rm, sh_raw, mdd, exp in top20:
            sp_key = tuple(sorted(sp.items()))
            rm_key = str(sorted(rm.items()))
            pos_raw = cached_raw_pos[(sp_key, rm_key)]
            strength = cached_strength[sp_key]
            for ep in exec_grid:
                pos_exec = execution_layer(pos_raw, strength,
                                           ep["consec_entry"], ep["min_hold"],
                                           ep["cooldown_bars"], ep["min_strength"])
                sh_exec = calc_sharpe_v4(pos_exec, log_ret, train_mask, ann, commission)
                best_combos.append((sp, rm, ep, sh_exec))

        top10 = sorted(best_combos, key=lambda x: x[3], reverse=True)[:10]
        if not top10:
            continue

        # ──── Build ensemble ────
        test_positions = []
        for sp, rm, ep, sh in top10:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, vwap, dev = cached_sigs[sp_key]
            pos_raw = _run_s6_bt(rm, sig, exit_arr, close, high, low, ind,
                                  vwap, dev, test_end)
            strength = cached_strength[sp_key]
            pos_exec = execution_layer(pos_raw, strength,
                                       ep["consec_entry"], ep["min_hold"],
                                       ep["cooldown_bars"], ep["min_strength"])
            test_positions.append(pos_exec)

        pos_stack = np.array(test_positions)
        ens_mag = np.mean(np.abs(pos_stack), axis=0)
        ens_dir = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir

        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm, ep) for sp, rm, ep, sh in top10]
        exec_params_by_year[test_year] = top10[0][2] if top10 else {}

    return results_by_year, params_by_year, {}, exec_params_by_year


def _run_s6_bt(rm, sig, exit_arr, close, high, low, ind, vwap, dev, end_idx,
               log_trades=0):
    """Run S6 backtest for either ATR or VWAP-band RM."""
    if rm["_rm_type"] == "vwap":
        sl_dist, tp_dist = precompute_vwap_sl_tp(
            close, sig, vwap, dev, rm["sl_dev"])
        be = 0.5 if rm["breakeven"] else -1.0
        pos, tr = bt_range_vwap_v3(
            sig, exit_arr, close, high, low, ind["adx14"],
            sl_dist, tp_dist, 30.0, int(rm["max_hold"]),
            be, rm["cooldown_bars"], WARMUP, end_idx, log_trades)
    else:
        be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
        td = 1 if rm.get("time_decay") else 0
        pos, tr = bt_range_v3(
            sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
            rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
            be, rm["cooldown_bars"], td, WARMUP, end_idx, log_trades)
    return pos


# ═══════════════════════════════════════════════════════════════════
# §5. Approach B — S5/S6 V4.1 (vpred + exec layer)
# ═══════════════════════════════════════════════════════════════════

def approach_b_s5_v41(tf, close, high, low, volume, ind, dates, sigma_dict,
                      is_hourly, log_ret, a_params_by_year, exec_params_by_year,
                      pivot_cache):
    """Approach B for S5 with commission and execution layer."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    n = len(close)
    b_grid = expand_grid(B_GRIDS_V3["Range"])

    results_by_year = {}
    best_params_by_year = {}

    for test_year in BCD_TEST_YEARS:
        _, val_start = _year_bounds(dates, 2020)
        _, val_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True
        if val_mask.sum() == 0:
            continue

        a_params = a_params_by_year.get(test_year, [])
        if not a_params:
            continue

        ep_a = exec_params_by_year.get(test_year, {})
        consec = ep_a.get("consec_entry", 1)
        mhold = ep_a.get("min_hold", 0)
        cool = ep_a.get("cooldown_bars", 0)
        mstr = ep_a.get("min_strength", 0.0)

        # Regenerate A's signals
        a_sigs = []
        a_strengths = []
        for item in a_params:
            sp = item[0]
            sp_key = tuple(sorted(sp.items()))
            piv_key = (sp["pivot_period"], sp["pivot_type"])
            P, S1, S2, R1, R2 = pivot_cache[piv_key]
            if is_hourly:
                sig, exit_arr = gen_s5_signals_v3(
                    close, ind, P, S1, R1, sp["filter_set"],
                    sp["confirmation"], sp["entry_buffer"])
            else:
                sig, exit_arr = gen_s5_signals_v2(
                    close, ind, P, S1, R1, sp["filter_set"])
            a_sigs.append((sig, exit_arr))
            span = np.abs(R1 - S1)
            dist = np.minimum(np.abs(close - S1), np.abs(close - R1))
            span_safe = np.where(np.isnan(span) | (span < 1e-12), 1e-12, span)
            dist_safe = np.where(np.isnan(dist), 0.0, dist)
            a_strengths.append(np.minimum(dist_safe / span_safe, 1.0))

        best_sh = -999.0
        best_bp = b_grid[0] if b_grid else {}

        for bp in b_grid:
            horizon = bp.get("horizon", "h1")
            sigma = sigma_dict.get(horizon, sigma_dict.get("h1", np.full(n, 0.02)))
            sigma_pos = sigma[sigma > 1e-6]
            sigma_median = np.nanmedian(sigma_pos) if len(sigma_pos) > 0 else 0.02
            k_be = -1.0 if bp.get("k_be") is None else bp["k_be"]
            gh = -1.0 if bp.get("gamma_hold") is None else bp["gamma_hold"]
            cd = bp.get("cooldown_bars", 5)
            k_tp = bp.get("k_tp", bp.get("k_sl", 1.0) * bp.get("ratio", 1.5))

            val_positions = []
            for (sig, exit_arr), strength in zip(a_sigs, a_strengths):
                pos_raw, _ = bt_range_vpred_v3(
                    sig, exit_arr, close, high, low, sigma, ind["adx14"],
                    bp["k_sl"], k_tp, 30.0, 20 if is_hourly else 10,
                    k_be, gh, sigma_median, cd, WARMUP, val_end, 0)
                pos_exec = execution_layer(pos_raw, strength, consec, mhold, cool, mstr)
                val_positions.append(pos_exec)

            pos_stack = np.array(val_positions)
            ens_mag = np.mean(np.abs(pos_stack), axis=0)
            ens_dir = np.sign(np.sum(pos_stack, axis=0))
            ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir
            sh = calc_sharpe_v4(ensemble, log_ret, val_mask, ann, commission)
            if sh > best_sh:
                best_sh = sh
                best_bp = dict(bp)

        # Run best on test
        best_sigma = sigma_dict.get(best_bp.get("horizon", "h1"),
                                    sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        bsm = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        k_be = -1.0 if best_bp.get("k_be") is None else best_bp["k_be"]
        gh = -1.0 if best_bp.get("gamma_hold") is None else best_bp["gamma_hold"]
        cd = best_bp.get("cooldown_bars", 5)
        k_tp = best_bp.get("k_tp", best_bp.get("k_sl", 1.0) * best_bp.get("ratio", 1.5))

        test_positions = []
        for (sig, exit_arr), strength in zip(a_sigs, a_strengths):
            pos_raw, _ = bt_range_vpred_v3(
                sig, exit_arr, close, high, low, best_sigma, ind["adx14"],
                best_bp["k_sl"], k_tp, 30.0, 20 if is_hourly else 10,
                k_be, gh, bsm, cd, WARMUP, test_end, 0)
            pos_exec = execution_layer(pos_raw, strength, consec, mhold, cool, mstr)
            test_positions.append(pos_exec)

        pos_stack = np.array(test_positions)
        ens_mag = np.mean(np.abs(pos_stack), axis=0)
        ens_dir = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir

        results_by_year[test_year] = ensemble
        best_params_by_year[test_year] = best_bp

    return results_by_year, best_params_by_year, {}


def approach_b_s6_v41(tf, close, high, low, volume, ind, dates, sigma_dict,
                      is_hourly, log_ret, a_params_by_year, exec_params_by_year,
                      vwap_cache, session_vwap_data, dir_bias_arr):
    """Approach B for S6 with commission and execution layer."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    n = len(close)
    b_grid = expand_grid(B_GRIDS_V3["Range"])

    results_by_year = {}
    best_params_by_year = {}

    for test_year in BCD_TEST_YEARS:
        _, val_start = _year_bounds(dates, 2020)
        _, val_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True
        if val_mask.sum() == 0:
            continue

        a_params = a_params_by_year.get(test_year, [])
        if not a_params:
            continue

        ep_a = exec_params_by_year.get(test_year, {})
        consec = ep_a.get("consec_entry", 1)
        mhold = ep_a.get("min_hold", 0)
        cool = ep_a.get("cooldown_bars", 0)
        mstr = ep_a.get("min_strength", 0.0)

        # Regenerate A's signals
        a_sigs = []
        a_strengths = []
        for item in a_params:
            sp = item[0]
            w = sp["vwap_window"]; dm = sp["dev_mult"]; fs = sp["filter_set"]
            if is_hourly:
                vtype = sp["vwap_type"]
                if vtype == "session":
                    vwap, dev = session_vwap_data
                else:
                    vwap, dev = vwap_cache[w]
                dba = None
            else:
                dba = dir_bias_arr if sp["dir_bias"] == "ON" else None
                vwap, dev = vwap_cache[w]
            sig, exit_arr = gen_s6_signals_v2(close, ind, vwap, dev, dm, fs, dba)
            a_sigs.append((sig, exit_arr))
            dev_safe = np.maximum(dev, 1e-12)
            vwap_safe = np.where(np.isnan(vwap), close, vwap)
            a_strengths.append(np.minimum(np.abs(close - vwap_safe) / (2.0 * dev_safe), 1.0))

        best_sh = -999.0
        best_bp = b_grid[0] if b_grid else {}

        for bp in b_grid:
            horizon = bp.get("horizon", "h1")
            sigma = sigma_dict.get(horizon, sigma_dict.get("h1", np.full(n, 0.02)))
            sigma_pos = sigma[sigma > 1e-6]
            sigma_median = np.nanmedian(sigma_pos) if len(sigma_pos) > 0 else 0.02
            k_be = -1.0 if bp.get("k_be") is None else bp["k_be"]
            gh = -1.0 if bp.get("gamma_hold") is None else bp["gamma_hold"]
            cd = bp.get("cooldown_bars", 5)
            k_tp = bp.get("k_tp", bp.get("k_sl", 1.0) * bp.get("ratio", 1.5))

            val_positions = []
            for (sig, exit_arr), strength in zip(a_sigs, a_strengths):
                pos_raw, _ = bt_range_vpred_v3(
                    sig, exit_arr, close, high, low, sigma, ind["adx14"],
                    bp["k_sl"], k_tp, 30.0, 20 if is_hourly else 10,
                    k_be, gh, sigma_median, cd, WARMUP, val_end, 0)
                pos_exec = execution_layer(pos_raw, strength, consec, mhold, cool, mstr)
                val_positions.append(pos_exec)

            pos_stack = np.array(val_positions)
            ens_mag = np.mean(np.abs(pos_stack), axis=0)
            ens_dir = np.sign(np.sum(pos_stack, axis=0))
            ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir
            sh = calc_sharpe_v4(ensemble, log_ret, val_mask, ann, commission)
            if sh > best_sh:
                best_sh = sh
                best_bp = dict(bp)

        best_sigma = sigma_dict.get(best_bp.get("horizon", "h1"),
                                    sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        bsm = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        k_be = -1.0 if best_bp.get("k_be") is None else best_bp["k_be"]
        gh = -1.0 if best_bp.get("gamma_hold") is None else best_bp["gamma_hold"]
        cd = best_bp.get("cooldown_bars", 5)
        k_tp = best_bp.get("k_tp", best_bp.get("k_sl", 1.0) * best_bp.get("ratio", 1.5))

        test_positions = []
        for (sig, exit_arr), strength in zip(a_sigs, a_strengths):
            pos_raw, _ = bt_range_vpred_v3(
                sig, exit_arr, close, high, low, best_sigma, ind["adx14"],
                best_bp["k_sl"], k_tp, 30.0, 20 if is_hourly else 10,
                k_be, gh, bsm, cd, WARMUP, test_end, 0)
            pos_exec = execution_layer(pos_raw, strength, consec, mhold, cool, mstr)
            test_positions.append(pos_exec)

        pos_stack = np.array(test_positions)
        ens_mag = np.mean(np.abs(pos_stack), axis=0)
        ens_dir = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir

        results_by_year[test_year] = ensemble
        best_params_by_year[test_year] = best_bp

    return results_by_year, best_params_by_year, {}


# ═══════════════════════════════════════════════════════════════════
# §6. Approach C V4.1 — Relaxed constraints + fallback to A
# ═══════════════════════════════════════════════════════════════════

def _c_grid_search_v41(cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
                       bars_per_year, n, min_exp, min_tpy, commission):
    """C grid search with V4.1 relaxed thresholds."""
    best_sh = -999.0
    best_cp = {}

    if cat in ("Trend", "Contrarian"):
        thresholds = C_THRESHOLDS_V41
        for threshold in thresholds:
            for lookback_base in C_LOOKBACKS:
                lookback = lookback_base * 9 if is_hourly else lookback_base
                for horizon in C_HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None:
                        continue
                    sigma_h22 = sigma_dict.get("h22")
                    sigma_h1 = sigma_dict.get("h1")
                    for direction in C_DIRECTION:
                        for term_f in C_TERM_FILTER:
                            if term_f and horizon == "h22":
                                continue
                            for hyst in C_HYSTERESIS:
                                mask = _build_c_mask_v3(cat, sigma, lookback,
                                                        threshold, direction, 5,
                                                        sigma_h22, sigma_h1,
                                                        term_f, hyst, n)
                                masked_pos = a_pos.copy()
                                masked_pos[~mask] = 0.0
                                sh = calc_sharpe_v4(masked_pos, log_ret, val_mask, ann, commission)
                                if sh > best_sh:
                                    exp_pct, tpy = _quick_exposure_trades(
                                        masked_pos, val_mask, bars_per_year)
                                    if exp_pct < min_exp or tpy < min_tpy:
                                        continue
                                    best_sh = sh
                                    best_cp = {"threshold": threshold, "horizon": horizon,
                                               "lookback": lookback_base, "direction": direction,
                                               "term_filter": term_f, "hysteresis": hyst}
    else:
        bands = C_RANGE_BANDS_V41
        for band in bands:
            for lookback_base in C_LOOKBACKS:
                lookback = lookback_base * 9 if is_hourly else lookback_base
                for horizon in C_HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None:
                        continue
                    sigma_h22 = sigma_dict.get("h22")
                    sigma_h1 = sigma_dict.get("h1")
                    for direction in C_DIRECTION:
                        for term_f in C_TERM_FILTER:
                            if term_f and horizon == "h22":
                                continue
                            for hyst in C_HYSTERESIS:
                                mask = _build_c_mask_v3(cat, sigma, lookback,
                                                        band, direction, 5,
                                                        sigma_h22, sigma_h1,
                                                        term_f, hyst, n)
                                masked_pos = a_pos.copy()
                                masked_pos[~mask] = 0.0
                                sh = calc_sharpe_v4(masked_pos, log_ret, val_mask, ann, commission)
                                if sh > best_sh:
                                    exp_pct, tpy = _quick_exposure_trades(
                                        masked_pos, val_mask, bars_per_year)
                                    if exp_pct < min_exp or tpy < min_tpy:
                                        continue
                                    best_sh = sh
                                    best_cp = {"band": band, "horizon": horizon,
                                               "lookback": lookback_base, "direction": direction,
                                               "term_filter": term_f, "hysteresis": hyst}
    return best_sh, best_cp


def approach_c_v41(sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n):
    """Approach C V4.1 — relaxed constraints + fallback to A."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    bpy = 252 * 9 if is_hourly else 252
    cat = CATEGORY[sid]
    results_by_year = {}
    best_params_by_year = {}

    for test_year in BCD_TEST_YEARS:
        _, val_start = _year_bounds(dates, 2020)
        _, val_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]

        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True

        a_pos = a_results.get(test_year)
        if a_pos is None:
            continue

        # 3-pass: strict → relaxed → fallback to A
        best_sh, best_cp = _c_grid_search_v41(
            cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
            bpy, n, C_MIN_EXPOSURE_V41, C_MIN_TRADES_YR_V41, commission)

        if not best_cp:
            best_sh, best_cp = _c_grid_search_v41(
                cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
                bpy, n, C_FALLBACK_EXPOSURE_V41, C_FALLBACK_TRADES_YR_V41, commission)

        if not best_cp:
            # Final fallback: C = A
            results_by_year[test_year] = a_pos.copy()
            best_params_by_year[test_year] = {"_fallback": "A"}
            continue

        sigma = sigma_dict.get(best_cp.get("horizon", "h1"))
        if sigma is None:
            results_by_year[test_year] = a_pos.copy()
            best_params_by_year[test_year] = {"_fallback": "A"}
            continue

        lookback_actual = best_cp.get("lookback", 252)
        lookback_actual = lookback_actual * 9 if is_hourly else lookback_actual
        if cat in ("Trend", "Contrarian"):
            tob = best_cp.get("threshold", 0.5)
        else:
            tob = best_cp.get("band", (0.3, 0.7))
        mask = _build_c_mask_v3(cat, sigma, lookback_actual, tob,
                                best_cp.get("direction", "OFF"), 5,
                                sigma_dict.get("h22"), sigma_dict.get("h1"),
                                best_cp.get("term_filter", False),
                                best_cp.get("hysteresis", 0), n)
        final_pos = a_pos.copy()
        final_pos[~mask] = 0.0
        results_by_year[test_year] = final_pos
        best_params_by_year[test_year] = best_cp

    return results_by_year, best_params_by_year, {}


# ═══════════════════════════════════════════════════════════════════
# §7. Approach A — S2 hourly with MIN_TRADES (full V4 approach)
# ═══════════════════════════════════════════════════════════════════

def approach_a_s2h_v41(close, high, low, volume, open_arr, ind, sma_cache,
                       dates, log_ret, pivot_data, daily_trend, vwap_cache):
    """Approach A for S2 hourly with MIN_TRADES constraint."""
    sid = "S2"
    tf = "hourly"
    is_hourly = True
    commission = COMM_HOURLY
    ann = sqrt(252 * 9)
    bpy = 252 * 9
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)

    sig_grid = expand_grid(SIGNAL_GRIDS[sid])
    cat = CATEGORY[sid]
    rm_grid = expand_grid(RM_GRIDS_V4.get((cat, tf), RM_GRIDS_V3.get((cat, tf), {})))

    results_by_year = {}
    params_by_year = {}
    exec_params_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        if train_end <= WARMUP or train_end >= n:
            continue
        train_mask[:] = False
        train_mask[WARMUP:train_end] = True

        cached_sigs = {}
        for sp in sig_grid:
            key = tuple(sorted(sp.items()))
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, {}, {}, vwap_cache,
                                              pivot_data, daily_trend, sp)
            cached_sigs[key] = (sig, exit_info)

        cached_strength = {}
        for sp in sig_grid:
            key = tuple(sorted(sp.items()))
            cached_strength[key] = signal_strength_s2(close, ind, sma_cache, sp)

        # Phase 1: Coarse
        coarse_rm = rm_grid[::2] if len(rm_grid) > 20 else rm_grid
        coarse_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            for rm in coarse_rm:
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, WARMUP, train_end, 0)
                sh = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                mdd = compute_max_dd(pos, log_ret, train_mask, commission)
                exp = compute_exposure(pos, train_mask)
                _, tpy = _quick_exposure_trades(pos, train_mask, bpy)
                coarse_results.append((sp, rm, sh, mdd, exp, tpy))

        # Filter with MIN_TRADES
        passing = [r for r in coarse_results
                   if r[2] > 0 and r[3] > -0.30 and r[4] > 5 and r[5] >= MIN_TRADES_PER_YEAR]
        if not passing:
            passing = sorted(coarse_results, key=lambda x: x[2], reverse=True)[:10]
        top50 = sorted(passing, key=lambda x: x[2], reverse=True)[:50]

        # Phase 2: Fine
        fine_results = []
        seen = set()
        for sp, rm, *_ in top50:
            neighbors = get_rm_neighbors(rm, rm_grid)
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            for rm_n in neighbors:
                combo_key = (sp_key, str(sorted(rm_n.items())))
                if combo_key in seen:
                    continue
                seen.add(combo_key)
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm_n, WARMUP, train_end, 0)
                sh = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                mdd = compute_max_dd(pos, log_ret, train_mask, commission)
                exp = compute_exposure(pos, train_mask)
                fine_results.append((sp, rm_n, sh, mdd, exp, 0))

        all_phase2 = top50 + fine_results
        dedup = {}
        for sp, rm, sh, mdd, exp, *_ in all_phase2:
            key = (tuple(sorted(sp.items())), str(sorted(rm.items())))
            if key not in dedup or sh > dedup[key][2]:
                dedup[key] = (sp, rm, sh, mdd, exp)
        top20 = sorted(dedup.values(), key=lambda x: x[2], reverse=True)[:20]

        # Phase 3: Exec layer
        exec_grid = expand_grid(EXEC_GRID_HOURLY)
        cached_raw_pos = {}
        for sp, rm, *_ in top20:
            sp_key = tuple(sorted(sp.items()))
            rm_key = str(sorted(rm.items()))
            if (sp_key, rm_key) not in cached_raw_pos:
                sig, exit_info = cached_sigs[sp_key]
                pos_raw, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                                rm, WARMUP, train_end, 0)
                cached_raw_pos[(sp_key, rm_key)] = pos_raw

        best_combos = []
        for sp, rm, sh_raw, mdd, exp in top20:
            sp_key = tuple(sorted(sp.items()))
            rm_key = str(sorted(rm.items()))
            pos_raw = cached_raw_pos[(sp_key, rm_key)]
            strength = cached_strength[sp_key]
            for ep in exec_grid:
                pos_exec = execution_layer(pos_raw, strength,
                                           ep["consec_entry"], ep["min_hold"],
                                           ep["cooldown_bars"], ep["min_strength"])
                sh_exec = calc_sharpe_v4(pos_exec, log_ret, train_mask, ann, commission)
                # MIN_TRADES in exec layer
                _, tpy_exec = _quick_exposure_trades(pos_exec, train_mask, bpy)
                if tpy_exec >= MIN_TRADES_PER_YEAR:
                    best_combos.append((sp, rm, ep, sh_exec))

        if not best_combos:
            # Fallback: allow any combo
            for sp, rm, sh_raw, mdd, exp in top20:
                sp_key = tuple(sorted(sp.items()))
                rm_key = str(sorted(rm.items()))
                pos_raw = cached_raw_pos[(sp_key, rm_key)]
                strength = cached_strength[sp_key]
                ep = exec_grid[0]  # default
                pos_exec = execution_layer(pos_raw, strength,
                                           ep["consec_entry"], ep["min_hold"],
                                           ep["cooldown_bars"], ep["min_strength"])
                sh_exec = calc_sharpe_v4(pos_exec, log_ret, train_mask, ann, commission)
                best_combos.append((sp, rm, ep, sh_exec))

        top10 = sorted(best_combos, key=lambda x: x[3], reverse=True)[:10]
        if not top10:
            continue

        test_positions = []
        for sp, rm, ep, sh in top10:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            pos_raw, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, WARMUP, test_end, 0)
            strength = cached_strength[sp_key]
            pos_exec = execution_layer(pos_raw, strength,
                                       ep["consec_entry"], ep["min_hold"],
                                       ep["cooldown_bars"], ep["min_strength"])
            test_positions.append(pos_exec)

        pos_stack = np.array(test_positions)
        ens_mag = np.mean(np.abs(pos_stack), axis=0)
        ens_dir = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir

        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm, ep) for sp, rm, ep, sh in top10]
        exec_params_by_year[test_year] = top10[0][2] if top10 else {}

    return results_by_year, params_by_year, {}, exec_params_by_year


# ═══════════════════════════════════════════════════════════════════
# §8. Approach D V4.1 (reuse V4)
# ═══════════════════════════════════════════════════════════════════

# approach_d_v4 is imported from V4 and used as-is


# ═══════════════════════════════════════════════════════════════════
# §9. process_ticker_v41 — selective per-strategy processing
# ═══════════════════════════════════════════════════════════════════

def process_ticker_v41(ticker, daily_df, hourly_df, vpred_df, v4_pos_df):
    """Process one ticker: selectively re-run S5, S6, S1h(C), S2h(full).
    Returns list of position rows to REPLACE in V4 data.
    """
    new_position_rows = []

    for tf in TIMEFRAMES:
        is_hourly = tf == "hourly"
        commission = COMM_HOURLY if is_hourly else COMM_DAILY

        if is_hourly:
            tdf = hourly_df[hourly_df["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
            if len(tdf) == 0:
                continue
            close = tdf["close"].values.astype(np.float64)
            high = tdf["high"].values.astype(np.float64)
            low = tdf["low"].values.astype(np.float64)
            open_arr = tdf["open"].values.astype(np.float64)
            volume = tdf["volume"].values.astype(np.float64)
            dates = pd.to_datetime(tdf["datetime"].values)
        else:
            tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
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

        ind = compute_base(close, high, low, open_arr, volume, is_hourly)

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
                if is_hourly:
                    bar_dates = pd.to_datetime(tdf["datetime"].values).normalize()
                else:
                    bar_dates = pd.to_datetime(tdf["date"].values)
                idx = np.searchsorted(vp_dates, bar_dates, side="right") - 1
                valid = idx >= 0
                sigma_arr[valid] = vp_vals[idx[valid]]
                if is_hourly:
                    nhours = _compute_nhours_per_day(tdf["datetime"].values)
                    sigma_arr = sigma_arr / np.sqrt(np.maximum(nhours, 1.0))
                sigma_dict[hname] = sigma_arr

        ann = sqrt(252 * 9) if is_hourly else sqrt(252)
        bpy = 252 * 9 if is_hourly else 252

        # ──── S5: Full re-run (A+B+C+D) ────
        pivot_cache = prepare_pivot_cache(daily_df, ticker, dates, is_hourly, tdf)

        s5_a, s5_a_params, _, s5_a_exec = approach_a_s5_v41(
            tf, close, high, low, open_arr, volume, ind,
            dates, is_hourly, log_ret, pivot_cache)
        _collect_positions(new_position_rows, s5_a, "S5_PivotPoints", tf, ticker,
                           "A", A_TEST_YEARS, dates, log_ret, n)

        s5_b, _, _ = approach_b_s5_v41(
            tf, close, high, low, volume, ind, dates, sigma_dict,
            is_hourly, log_ret, s5_a_params, s5_a_exec, pivot_cache)
        _collect_positions(new_position_rows, s5_b, "S5_PivotPoints", tf, ticker,
                           "B", BCD_TEST_YEARS, dates, log_ret, n)

        s5_c, _, _ = approach_c_v41("S5", tf, dates, s5_a, sigma_dict, is_hourly, log_ret, n)
        _collect_positions(new_position_rows, s5_c, "S5_PivotPoints", tf, ticker,
                           "C", BCD_TEST_YEARS, dates, log_ret, n)

        s5_d, _, _ = approach_d_v4("S5", tf, dates, s5_a, sigma_dict, is_hourly, log_ret, n)
        _collect_positions(new_position_rows, s5_d, "S5_PivotPoints", tf, ticker,
                           "D", BCD_TEST_YEARS, dates, log_ret, n)

        # ──── S6: Full re-run (A+B+C+D) ────
        vwap_cache, session_vwap_data, dir_bias_arr = prepare_vwap_cache_ext(
            close, high, low, volume, is_hourly, tdf,
            daily_df if not is_hourly else None,
            ticker if not is_hourly else None)

        s6_a, s6_a_params, _, s6_a_exec = approach_a_s6_v41(
            tf, close, high, low, open_arr, volume, ind,
            dates, is_hourly, log_ret,
            vwap_cache, session_vwap_data, dir_bias_arr)
        _collect_positions(new_position_rows, s6_a, "S6_VWAP", tf, ticker,
                           "A", A_TEST_YEARS, dates, log_ret, n)

        s6_b, _, _ = approach_b_s6_v41(
            tf, close, high, low, volume, ind, dates, sigma_dict,
            is_hourly, log_ret, s6_a_params, s6_a_exec,
            vwap_cache, session_vwap_data, dir_bias_arr)
        _collect_positions(new_position_rows, s6_b, "S6_VWAP", tf, ticker,
                           "B", BCD_TEST_YEARS, dates, log_ret, n)

        s6_c, _, _ = approach_c_v41("S6", tf, dates, s6_a, sigma_dict, is_hourly, log_ret, n)
        _collect_positions(new_position_rows, s6_c, "S6_VWAP", tf, ticker,
                           "C", BCD_TEST_YEARS, dates, log_ret, n)

        s6_d, _, _ = approach_d_v4("S6", tf, dates, s6_a, sigma_dict, is_hourly, log_ret, n)
        _collect_positions(new_position_rows, s6_d, "S6_VWAP", tf, ticker,
                           "D", BCD_TEST_YEARS, dates, log_ret, n)

        # ──── S1 hourly: C only (re-run with relaxed C) ────
        if is_hourly:
            # Load A from V4 positions
            s1h_a = _load_v4_positions(v4_pos_df, "S1_MeanRev", "hourly", ticker, dates, n)
            if s1h_a:
                s1h_c, _, _ = approach_c_v41("S1", tf, dates, s1h_a, sigma_dict,
                                              is_hourly, log_ret, n)
                _collect_positions(new_position_rows, s1h_c, "S1_MeanRev", tf, ticker,
                                   "C", BCD_TEST_YEARS, dates, log_ret, n)

        # ──── S2 hourly: Full re-run (A+B+C+D) with MIN_TRADES ────
        if is_hourly:
            sma_windows = sorted(set(SIGNAL_GRIDS["S1"].get("ma_window", []) +
                                     SIGNAL_GRIDS["S2"].get("bb_window", [])))
            sma_cache = precompute_sma_cache(close, sma_windows)

            pivot_data_simple = {}
            h_dts = tdf["datetime"].values
            P, S1p, R1p = compute_daily_pivots_for_hourly(daily_df, ticker, h_dts, "classic")
            pivot_data_simple["classic"] = (P, S1p, R1p)
            d_dates, d_above = build_daily_trend(daily_df, ticker)
            daily_trend = align_daily_to_hourly(d_dates, d_above, h_dts)

            s2h_a, s2h_a_params, _, s2h_a_exec = approach_a_s2h_v41(
                close, high, low, volume, open_arr, ind, sma_cache,
                dates, log_ret, pivot_data_simple, daily_trend, vwap_cache)
            _collect_positions(new_position_rows, s2h_a, "S2_Bollinger", tf, ticker,
                               "A", A_TEST_YEARS, dates, log_ret, n)

            # B for S2h
            s2h_b, _, _ = _approach_b_s2h_v41(
                close, high, low, volume, ind, sma_cache,
                dates, sigma_dict, log_ret, s2h_a_params, s2h_a_exec,
                pivot_data_simple, daily_trend, vwap_cache)
            _collect_positions(new_position_rows, s2h_b, "S2_Bollinger", tf, ticker,
                               "B", BCD_TEST_YEARS, dates, log_ret, n)

            s2h_c, _, _ = approach_c_v41("S2", tf, dates, s2h_a, sigma_dict,
                                          is_hourly, log_ret, n)
            _collect_positions(new_position_rows, s2h_c, "S2_Bollinger", tf, ticker,
                               "C", BCD_TEST_YEARS, dates, log_ret, n)

            s2h_d, _, _ = approach_d_v4("S2", tf, dates, s2h_a, sigma_dict,
                                         is_hourly, log_ret, n)
            _collect_positions(new_position_rows, s2h_d, "S2_Bollinger", tf, ticker,
                               "D", BCD_TEST_YEARS, dates, log_ret, n)

        # ──── S1 daily / S2 daily: C only ────
        if not is_hourly:
            sma_windows = sorted(set(SIGNAL_GRIDS["S1"].get("ma_window", []) +
                                     SIGNAL_GRIDS["S2"].get("bb_window", [])))
            sma_cache = precompute_sma_cache(close, sma_windows)

            for sid_label, sname in [("S1", "S1_MeanRev"), ("S2", "S2_Bollinger")]:
                a_res = _load_v4_positions(v4_pos_df, sname, tf, ticker, dates, n)
                if a_res:
                    c_res, _, _ = approach_c_v41(sid_label, tf, dates, a_res, sigma_dict,
                                                  is_hourly, log_ret, n)
                    _collect_positions(new_position_rows, c_res, sname, tf, ticker,
                                       "C", BCD_TEST_YEARS, dates, log_ret, n)

    return new_position_rows


def _approach_b_s2h_v41(close, high, low, volume, ind, sma_cache,
                        dates, sigma_dict, log_ret, a_params_by_year,
                        exec_params_by_year, pivot_data, daily_trend, vwap_cache):
    """Approach B for S2 hourly with V4 patterns."""
    sid = "S2"
    is_hourly = True
    commission = COMM_HOURLY
    ann = sqrt(252 * 9)
    n = len(close)
    cat = CATEGORY[sid]
    b_grid = expand_grid(B_GRIDS_V3[cat])

    results_by_year = {}
    best_params_by_year = {}

    for test_year in BCD_TEST_YEARS:
        _, val_start = _year_bounds(dates, 2020)
        _, val_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True
        if val_mask.sum() == 0:
            continue

        a_params = a_params_by_year.get(test_year, [])
        if not a_params:
            continue

        ep_a = exec_params_by_year.get(test_year, {})
        consec = ep_a.get("consec_entry", 1)
        mhold = ep_a.get("min_hold", 0)
        cool = ep_a.get("cooldown_bars", 0)
        mstr = ep_a.get("min_strength", 0.0)

        a_sigs = []
        a_strengths = []
        for item in a_params:
            sp = item[0]
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, {}, {}, vwap_cache,
                                              pivot_data, daily_trend, sp)
            strength = signal_strength_s2(close, ind, sma_cache, sp)
            a_sigs.append((sig, exit_info))
            a_strengths.append(strength)

        best_sh = -999.0
        best_bp = b_grid[0] if b_grid else {}

        for bp in b_grid:
            horizon = bp.get("horizon", "h1")
            sigma = sigma_dict.get(horizon, sigma_dict.get("h1", np.full(n, 0.02)))
            sigma_pos = sigma[sigma > 1e-6]
            sigma_median = np.nanmedian(sigma_pos) if len(sigma_pos) > 0 else 0.02
            bp_engine = {k: v for k, v in bp.items() if k not in ("horizon", "ratio")}
            if "ratio" in bp:
                bp_engine["k_tp"] = bp["k_sl"] * bp["ratio"]

            val_positions = []
            for (sig, exit_info), strength in zip(a_sigs, a_strengths):
                pos_raw, _ = dispatch_bt_vpred_v3(sid, sig, exit_info, close, high, low, ind,
                                                  sigma, bp_engine, WARMUP, val_end,
                                                  is_hourly, sigma_median, 0)
                pos_exec = execution_layer(pos_raw, strength, consec, mhold, cool, mstr)
                val_positions.append(pos_exec)

            pos_stack = np.array(val_positions)
            ens_mag = np.mean(np.abs(pos_stack), axis=0)
            ens_dir = np.sign(np.sum(pos_stack, axis=0))
            ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir
            sh = calc_sharpe_v4(ensemble, log_ret, val_mask, ann, commission)
            if sh > best_sh:
                best_sh = sh
                best_bp = dict(bp)

        best_sigma = sigma_dict.get(best_bp.get("horizon", "h1"),
                                    sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        bsm = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        best_bp_engine = {k: v for k, v in best_bp.items() if k not in ("horizon", "ratio")}
        if "ratio" in best_bp:
            best_bp_engine["k_tp"] = best_bp["k_sl"] * best_bp["ratio"]

        test_positions = []
        for (sig, exit_info), strength in zip(a_sigs, a_strengths):
            pos_raw, _ = dispatch_bt_vpred_v3(sid, sig, exit_info, close, high, low, ind,
                                               best_sigma, best_bp_engine, WARMUP, test_end,
                                               is_hourly, bsm, 0)
            pos_exec = execution_layer(pos_raw, strength, consec, mhold, cool, mstr)
            test_positions.append(pos_exec)

        pos_stack = np.array(test_positions)
        ens_mag = np.mean(np.abs(pos_stack), axis=0)
        ens_dir = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir

        results_by_year[test_year] = ensemble
        best_params_by_year[test_year] = best_bp

    return results_by_year, best_params_by_year, {}


# ═══════════════════════════════════════════════════════════════════
# §10. Helper: Collect positions & Load V4 positions
# ═══════════════════════════════════════════════════════════════════

def _collect_positions(rows_list, results_by_year, strategy_name, tf, ticker,
                       approach, test_years, dates, log_ret, n):
    """Collect position rows in V4 format."""
    for year in test_years:
        pos_arr = results_by_year.get(year)
        if pos_arr is None:
            continue
        ys = _year_bounds(dates, year)[0]
        ye = _year_bounds(dates, year + 1)[0] if year < 2026 else n
        for bar in range(ys, ye):
            if bar < n:
                rows_list.append((
                    dates[bar], strategy_name, tf, ticker, approach, year,
                    round(float(pos_arr[bar]), 6),
                    round(float(pos_arr[bar] * log_ret[bar]), 8),
                ))


def _load_v4_positions(v4_pos_df, strategy, tf, ticker, dates, n):
    """Load A positions from V4 parquet for a given strategy/tf/ticker.
    Returns dict {year: pos_arr} or empty dict.
    """
    sub = v4_pos_df[(v4_pos_df["strategy"] == strategy) &
                     (v4_pos_df["tf"] == tf) &
                     (v4_pos_df["ticker"] == ticker) &
                     (v4_pos_df["approach"] == "A")]
    if len(sub) == 0:
        return {}

    results = {}
    for year in sub["test_year"].unique():
        ysub = sub[sub["test_year"] == year].sort_values("date")
        pos_arr = np.zeros(n)
        ys = _year_bounds(dates, year)[0]
        ye = _year_bounds(dates, year + 1)[0] if year < 2026 else n

        if len(ysub) == 0:
            continue

        # Map V4 positions back to bar indices
        v4_dates = pd.to_datetime(ysub["date"].values)
        v4_pos = ysub["position"].values
        for i, d in enumerate(v4_dates):
            # Find matching bar
            matches = np.where(dates == d)[0]
            if len(matches) > 0:
                pos_arr[matches[0]] = v4_pos[i]

        results[year] = pos_arr

    return results


# ═══════════════════════════════════════════════════════════════════
# §11. Merge results
# ═══════════════════════════════════════════════════════════════════

def merge_results(v4_pos_df, new_rows):
    """Merge V4.1 new position rows with V4 data.
    Replace rows where (strategy, tf, ticker, approach, test_year, date) match.
    """
    if not new_rows:
        return v4_pos_df

    new_df = pd.DataFrame(new_rows,
                           columns=["date", "strategy", "tf", "ticker", "approach",
                                    "test_year", "position", "daily_gross_return"])

    # Identify which (strategy, tf, approach) combos to replace
    replace_keys = new_df.groupby(["strategy", "tf", "approach"]).size().reset_index()
    replace_keys = set(zip(replace_keys["strategy"], replace_keys["tf"], replace_keys["approach"]))

    # Keep V4 rows that are NOT being replaced
    keep_mask = ~v4_pos_df.apply(
        lambda r: (r["strategy"], r["tf"], r["approach"]) in replace_keys, axis=1)
    kept = v4_pos_df[keep_mask]

    merged = pd.concat([kept, new_df], ignore_index=True)
    merged = merged.sort_values(["strategy", "tf", "ticker", "approach", "test_year", "date"]).reset_index(drop=True)
    return merged


# ═══════════════════════════════════════════════════════════════════
# §12. Multiprocessing
# ═══════════════════════════════════════════════════════════════════

_G_DAILY = None
_G_HOURLY = None
_G_VPRED = None
_G_V4POS = None


def _pool_init_v41(daily, hourly, vpred, v4pos):
    global _G_DAILY, _G_HOURLY, _G_VPRED, _G_V4POS
    _G_DAILY = daily
    _G_HOURLY = hourly
    _G_VPRED = vpred
    _G_V4POS = v4pos


def _worker_func_v41(ticker):
    return ticker, process_ticker_v41(ticker, _G_DAILY, _G_HOURLY, _G_VPRED, _G_V4POS)


def warmup_numba_v41():
    """JIT-compile numba functions."""
    if not HAS_NUMBA:
        return
    n = 50
    sig = np.zeros(n, dtype=np.int8)
    sig[10] = 1
    ex = np.zeros(n, dtype=bool)
    c = np.random.randn(n).cumsum() + 100
    h = c + 1; l = c - 1
    a = np.ones(n)
    adx = np.ones(n) * 15
    sigma = np.ones(n) * 0.02
    sl_d = np.ones(n) * 0.01; tp_d = np.ones(n) * 0.01

    bt_range_v3(sig, ex, c, h, l, a, adx, 1.0, 1.5, 30.0, 10, 1.0, 5, 1, 5, n, 0)
    bt_range_vpred_v3(sig, ex, c, h, l, sigma, adx, 1.0, 2.0, 30.0, 10, 1.0, 0.5, 0.02, 5, 5, n, 0)
    bt_range_pivot_v3(sig, ex, c, h, l, adx, sl_d, tp_d, 30.0, 10, 1.0, 5, 5, n, 0)
    bt_range_vwap_v3(sig, ex, c, h, l, adx, sl_d, tp_d, 30.0, 10, 1.0, 5, 5, n, 0)
    pctrank = np.random.rand(n)
    _apply_hysteresis(pctrank, 1, 0.5, 0.4, n)
    print("  Numba warmup done (V4.1)")


# ═══════════════════════════════════════════════════════════════════
# §13. main()
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("Walk-Forward V4.1 — Selective Fix (S5/S6/C/MIN_TRADES)")
    print(f"Tickers: {len(TICKERS)}")
    print(f"Re-running: S5 (A+B+C+D), S6 (A+B+C+D), S1h(C), S2h(A+B+C+D), S1d/S2d(C)")
    print(f"MIN_TRADES_PER_YEAR: {MIN_TRADES_PER_YEAR}")
    print(f"C relaxed: exp>={C_MIN_EXPOSURE_V41}%, tpy>={C_MIN_TRADES_YR_V41}, fallback to A")
    print("=" * 70)

    # Load data
    daily, hourly, vpred = load_data()

    # Load V4 positions
    print(f"\nLoading V4 positions from {V4_POS_PATH}...")
    v4_pos_df = pd.read_parquet(V4_POS_PATH)
    print(f"  Loaded: {len(v4_pos_df):,} rows")

    # Grid sizes
    for label, sg_def, rmp, rma in [
        ("S5 daily", S5_SIG_DAILY, S5_RM_PIVOT_DAILY, S5_RM_ATR_DAILY),
        ("S5 hourly", S5H_SIG, S5H_RM_PIVOT, S5H_RM_ATR),
    ]:
        ns = len(expand_grid(sg_def))
        nrp = len(expand_grid(rmp))
        nra = len(expand_grid(rma))
        print(f"  {label}: {ns} sig x ({nrp} pivot + {nra} ATR) = {ns*(nrp+nra)} combos")
    for label, sg_def, rma, rmv in [
        ("S6 daily", S6D_SIG, S6_RM_ATR_DAILY, S6_RM_VWAP_DAILY),
        ("S6 hourly", S6H_SIG, S6_RM_ATR_HOURLY, S6_RM_VWAP_HOURLY),
    ]:
        ns = len(expand_grid(sg_def))
        nra = len(expand_grid(rma))
        nrv = len(expand_grid(rmv))
        print(f"  {label}: {ns} sig x ({nra} ATR + {nrv} VWAP) = {ns*(nra+nrv)} combos")

    print(f"\nWarming up numba...")
    warmup_numba_v41()

    print(f"\nProcessing {len(TICKERS)} tickers with {N_WORKERS} workers...")

    all_new_rows = []

    if N_WORKERS > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(N_WORKERS, initializer=_pool_init_v41,
                       initargs=(daily, hourly, vpred, v4_pos_df)) as pool:
            for i, (ticker, new_rows) in enumerate(
                    pool.imap_unordered(_worker_func_v41, TICKERS)):
                all_new_rows.extend(new_rows)
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {len(new_rows):,} new rows ({elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(TICKERS):
            new_rows = process_ticker_v41(ticker, daily, hourly, vpred, v4_pos_df)
            all_new_rows.extend(new_rows)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {len(new_rows):,} new rows ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")
    print(f"Total new rows: {len(all_new_rows):,}")

    # Merge
    print("\nMerging with V4 data...")
    merged = merge_results(v4_pos_df, all_new_rows)
    out_path = OUT_DIR / "daily_positions.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"  Saved merged: {len(merged):,} rows -> {out_path}")

    # ──── Verification ────
    print("\n" + "=" * 70)
    print("=== VERIFICATION ===")
    print("=" * 70)

    # 1. Check no NaN in C
    for strat in ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
                   "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]:
        for tf in ["daily", "hourly"]:
            c_sub = merged[(merged["strategy"] == strat) & (merged["tf"] == tf) &
                            (merged["approach"] == "C")]
            if len(c_sub) == 0:
                print(f"  {strat} {tf} C: MISSING!")
            else:
                nan_pct = c_sub["daily_gross_return"].isna().mean()
                print(f"  {strat} {tf} C: {len(c_sub):,} rows, NaN={nan_pct:.1%}")

    # 2. S2 hourly trades
    s2h = merged[(merged["strategy"] == "S2_Bollinger") & (merged["tf"] == "hourly")]
    for app in ["A", "B", "C", "D"]:
        sub = s2h[s2h["approach"] == app]
        if len(sub) == 0:
            print(f"  S2 hourly {app}: no data")
            continue
        trades = (sub["position"].diff().abs() > 0).sum()
        days = sub.groupby("test_year").apply(len).sum()
        bpy = 252 * 9
        years = max(days / bpy, 0.5)
        tpy = trades / years
        print(f"  S2 hourly {app}: {trades} trades, ~{tpy:.2f} trades/yr")

    # 3. Binary check
    unique_pos = merged["position"].unique()
    non_binary = [p for p in unique_pos if abs(p) > 0.01 and abs(abs(p) - 1.0) > 0.01]
    if non_binary:
        print(f"  WARNING: Non-binary positions: {non_binary[:10]}")
    else:
        print("  All positions binary {-1, 0, +1}")

    # 4. BEFORE vs AFTER comparison
    print("\n" + "=" * 70)
    print("=== BEFORE vs AFTER (BCD years, mean Sharpe) ===")
    print("=" * 70)

    bcd_years = [2022, 2023, 2024, 2025]
    bpy_d = 252; bpy_h = 252 * 9

    for strat in ["S5_PivotPoints", "S6_VWAP", "S1_MeanRev", "S2_Bollinger"]:
        for tf in ["daily", "hourly"]:
            for app in ["A", "B", "C", "D"]:
                # Before (V4)
                v4_sub = v4_pos_df[(v4_pos_df["strategy"] == strat) &
                                    (v4_pos_df["tf"] == tf) &
                                    (v4_pos_df["approach"] == app) &
                                    (v4_pos_df["test_year"].isin(bcd_years))]
                # After (merged)
                m_sub = merged[(merged["strategy"] == strat) &
                                (merged["tf"] == tf) &
                                (merged["approach"] == app) &
                                (merged["test_year"].isin(bcd_years))]

                # Compute gross Sharpe
                def _mean_sharpe(df, is_h):
                    if len(df) == 0:
                        return float('nan')
                    sharpes = []
                    for (t, y), grp in df.groupby(["ticker", "test_year"]):
                        r = grp["daily_gross_return"].values
                        if len(r) > 1:
                            s = np.std(r, ddof=1)
                            ann_f = sqrt(bpy_h) if is_h else sqrt(bpy_d)
                            sh = np.mean(r) / s * ann_f if s > 1e-12 else 0.0
                            sharpes.append(sh)
                    return np.mean(sharpes) if sharpes else float('nan')

                is_h = tf == "hourly"
                before = _mean_sharpe(v4_sub, is_h)
                after = _mean_sharpe(m_sub, is_h)
                delta = after - before if not (np.isnan(before) or np.isnan(after)) else float('nan')
                if np.isnan(before) and np.isnan(after):
                    continue
                b_str = f"{before:.4f}" if not np.isnan(before) else "  N/A "
                a_str = f"{after:.4f}" if not np.isnan(after) else "  N/A "
                d_str = f"{delta:+.4f}" if not np.isnan(delta) else " N/A "
                print(f"  {strat:>16s} {tf:>6s} {app}: BEFORE={b_str}  AFTER={a_str}  delta={d_str}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"V4.1 DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
