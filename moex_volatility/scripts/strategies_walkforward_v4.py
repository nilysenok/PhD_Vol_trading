#!/usr/bin/env python3
"""
strategies_walkforward_v4.py — Commission-Optimized Walk-Forward Pipeline (V4)

Key changes from V3:
  - Binary positions {-1, 0, +1} — no fractional ensemble weights
  - Explicit commission in calibration (COMM_DAILY=0.40%, COMM_HOURLY=0.35%)
  - Signal strength functions per strategy (6 functions)
  - Execution layer: strength_filter → discretize → consec_entry → min_hold → cooldown
  - Coarse-to-fine grid search in Approach A (Phase 1 coarse → Phase 2 fine → Phase 3 exec)
  - Majority-vote ensemble (mean > 0.5 → 1, else 0)
  - D: binary vol-gate (scale < threshold → off, else keep)
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

# ═══════════════════════════════════════════════════════════════════
# Imports from V3 (strategies_walkforward.py)
# ═══════════════════════════════════════════════════════════════════
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    # BT engines (ATR)
    bt_contrarian_v3, bt_trend_v3, bt_range_v3, bt_range_split_v3,
    # BT engines (vpred)
    bt_contrarian_vpred_v3, bt_trend_vpred_v3, bt_range_vpred_v3, bt_range_split_vpred_v3,
    # Signal generators
    gen_s1_signals, gen_s2_signals, gen_s3_signals, gen_s4_signals,
    gen_s5_signals, gen_s6_signals,
    # Caches & indicators
    compute_base, precompute_sma_cache, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    # Pivots & trend
    calc_pivot_daily, compute_daily_pivots_for_hourly,
    build_daily_trend, align_daily_to_hourly,
    # Dispatch
    dispatch_signals, dispatch_bt_atr_v3, dispatch_bt_vpred_v3,
    # Grids
    SIGNAL_GRIDS, TREND_RM_GRID, B_GRIDS_V3, expand_grid,
    # C/D grids
    C_THRESHOLDS, C_RANGE_BANDS, C_LOOKBACKS, C_HORIZONS, C_DIRECTION,
    C_TERM_FILTER, C_HYSTERESIS,
    C_MIN_EXPOSURE, C_MIN_TRADES_YR, C_FALLBACK_EXPOSURE, C_FALLBACK_TRADES_YR,
    D_TARGET_VOLS, D_MAX_LEVS, D_GAMMA, D_VOL_FLOORS, D_VOL_CAPS,
    D_SMOOTH, D_HORIZONS, D_INV_LOOKBACKS,
    # Helpers
    _year_bounds, _compute_nhours_per_day, _apply_hysteresis,
    _build_c_mask_v3, _compute_d_scale_v3, _quick_exposure_trades,
    _extract_trades_from_pos,
    dm_test, bootstrap_sharpe_diff,
    # Data loading
    load_data,
    # Constants
    TICKERS, STRATEGY_IDS, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, BCD_TEST_YEARS, WARMUP, TIMEFRAMES,
    EXIT_REASON_NAMES, RM_GRIDS_V3,
)

# ═══════════════════════════════════════════════════════════════════
# Imports from s5_rerun.py
# ═══════════════════════════════════════════════════════════════════
from s5_rerun import (
    bt_range_pivot_v3, precompute_pivot_sl_tp, gen_s5_signals_v2,
    calc_pivot_ext, calc_pivot_weekly, calc_pivot_monthly,
    calc_pivot_daily_ext, map_daily_pivots_to_hourly,
    precompute_pivot_base_distances,
    S5_SIG_DAILY, S5_SIG_HOURLY, S5_RM_PIVOT_DAILY, S5_RM_PIVOT_HOURLY,
    S5_RM_ATR_DAILY, S5_RM_ATR_HOURLY,
)

# ═══════════════════════════════════════════════════════════════════
# Imports from s5s6_rerun.py
# ═══════════════════════════════════════════════════════════════════
from s5s6_rerun import (
    gen_s5_signals_v3, compute_session_vwap, gen_s6_signals_v2,
    bt_range_vwap_v3, precompute_vwap_sl_tp,
    precompute_vwap_cache_ext,
    map_monthly_pivots_to_hourly,
    S5H_SIG, S6D_SIG, S6H_SIG,
    S5H_RM_PIVOT, S5H_RM_ATR,
    S6_RM_ATR_DAILY, S6_RM_ATR_HOURLY,
    S6_RM_VWAP_DAILY, S6_RM_VWAP_HOURLY,
)


# ═══════════════════════════════════════════════════════════════════
# §0. Constants & Commission
# ═══════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results" / "final" / "strategies" / "data"
OUT_DIR  = BASE / "results" / "final" / "strategies" / "walkforward_v4"
OUT_TABLES = OUT_DIR / "tables"
OUT_DATA   = OUT_DIR / "data"

COMM_DAILY  = 0.0040   # 0.40% per side
COMM_HOURLY = 0.0035   # 0.35% per side

N_WORKERS = min(cpu_count(), 8)

ALL_APPROACHES = ["A", "B", "C", "D"]
BCD_APPROACHES = ["B", "C", "D"]


# ═══════════════════════════════════════════════════════════════════
# §1. Signal Strength Functions
# ═══════════════════════════════════════════════════════════════════

def signal_strength_s1(close, ind, sma_cache, sig_params):
    """S1 MeanRev: strength = min(|z| / z_strong, 1.0), z_strong=3.0"""
    w = sig_params["ma_window"]
    sma, std = sma_cache[w]
    z = (close - sma) / np.maximum(std, 1e-12)
    return np.minimum(np.abs(z) / 3.0, 1.0)


def signal_strength_s2(close, ind, sma_cache, sig_params):
    """S2 Bollinger: strength = min(penetration / (k_str * std), 1.0), k_str=0.5"""
    w = sig_params["bb_window"]
    sma, std = sma_cache[w]
    bb_std = sig_params["bb_std"]
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    pen = np.maximum(np.maximum(lower - close, close - upper), 0.0) / np.maximum(std, 1e-12)
    return np.minimum(pen / 0.5, 1.0)


def signal_strength_s3(close, ind, dc_cache, sig_params):
    """S3 Donchian: strength = min(adx / 35, 1.0)"""
    adx = ind["adx14"]
    return np.minimum(np.where(np.isnan(adx), 0.0, adx) / 35.0, 1.0)


def signal_strength_s4(close, ind, st_cache, sig_params):
    """S4 Supertrend: strength = min(|close-ST| / (k_str * ATR), 1.0), k_str=1.5"""
    key = (sig_params["atr_period"], sig_params["multiplier"])
    st, _ = st_cache[key]
    atr = ind["atr14"]
    gap = np.abs(close - st)
    return np.minimum(gap / (1.5 * np.maximum(atr, 1e-12)), 1.0)


def signal_strength_s5(close, pivot_data):
    """S5 Pivot: strength = min(|close - nearest_level| / |R1 - S1|, 1.0)"""
    P, S1, R1 = pivot_data["classic"][:3]
    span = np.abs(R1 - S1)
    dist = np.minimum(np.abs(close - S1), np.abs(close - R1))
    # Handle nans
    span_safe = np.where(np.isnan(span) | (span < 1e-12), 1e-12, span)
    dist_safe = np.where(np.isnan(dist), 0.0, dist)
    return np.minimum(dist_safe / span_safe, 1.0)


def signal_strength_s6(close, ind, vwap_cache, sig_params):
    """S6 VWAP: strength = min(|close - vwap| / (k_str * dev), 1.0), k_str=2.0"""
    w = sig_params.get("vwap_window", 10)
    if w in vwap_cache:
        vwap, dev = vwap_cache[w]
    else:
        # Fallback: first available in cache
        first_key = next(iter(vwap_cache), None)
        if first_key is None:
            return np.zeros(len(close))
        vwap, dev = vwap_cache[first_key]
    dev_safe = np.maximum(dev, 1e-12)
    vwap_safe = np.where(np.isnan(vwap), close, vwap)
    return np.minimum(np.abs(close - vwap_safe) / (2.0 * dev_safe), 1.0)


def dispatch_strength(sid, close, ind, sma_cache, dc_cache, st_cache,
                      vwap_cache, pivot_data, sig_params):
    """Route to correct signal strength function."""
    if sid == "S1":
        return signal_strength_s1(close, ind, sma_cache, sig_params)
    elif sid == "S2":
        return signal_strength_s2(close, ind, sma_cache, sig_params)
    elif sid == "S3":
        return signal_strength_s3(close, ind, dc_cache, sig_params)
    elif sid == "S4":
        return signal_strength_s4(close, ind, st_cache, sig_params)
    elif sid == "S5":
        return signal_strength_s5(close, pivot_data)
    elif sid == "S6":
        return signal_strength_s6(close, ind, vwap_cache, sig_params)
    return np.ones(len(close))


# ═══════════════════════════════════════════════════════════════════
# §2. Execution Layer
# ═══════════════════════════════════════════════════════════════════

def execution_layer(raw_pos, strength, consec_entry, min_hold,
                    cooldown_bars, min_strength):
    """
    Pipeline: strength_filter -> discretize -> consecutive -> min_hold -> cooldown
    Input:  raw_pos[n] from bt_engine (fractional or binary)
    Output: binary_pos[n] in {-1, 0, +1} (direction-aware)
    """
    n = len(raw_pos)
    pos = np.abs(raw_pos).copy()

    # 1. Strength filter: zero out bars where strength < min_strength
    if min_strength > 0:
        pos[strength < min_strength] = 0.0

    # 2. Discretize: threshold = 0.5
    binary = (pos > 0.5).astype(np.float64)

    # 3. Consecutive entry confirmation
    if consec_entry > 1:
        confirmed = np.zeros(n)
        count = 0
        in_pos = False
        for i in range(n):
            if binary[i] > 0:
                if in_pos:
                    confirmed[i] = 1.0
                else:
                    count += 1
                    if count >= consec_entry:
                        confirmed[i] = 1.0
                        in_pos = True
            else:
                count = 0
                in_pos = False
        binary = confirmed

    # 4. Minimum hold: once entered, hold for min_hold bars
    if min_hold > 0:
        held = np.zeros(n)
        remaining = 0
        for i in range(n):
            if binary[i] > 0 and remaining == 0:
                remaining = min_hold
            if remaining > 0:
                held[i] = 1.0
                remaining -= 1
        binary = held

    # 5. Cooldown: after exit, wait cooldown_bars before re-entry
    if cooldown_bars > 0:
        result = np.zeros(n)
        cool = 0
        for i in range(n):
            if cool > 0:
                cool -= 1
                continue
            if binary[i] > 0:
                result[i] = 1.0
            elif i > 0 and binary[i] == 0 and result[i-1] > 0:
                cool = cooldown_bars
        binary = result

    # Restore direction from raw_pos sign
    direction = np.sign(raw_pos)
    return binary * direction


# ═══════════════════════════════════════════════════════════════════
# §3. Execution Grid
# ═══════════════════════════════════════════════════════════════════

EXEC_GRID_DAILY = dict(
    consec_entry=[1, 2],
    min_hold=[0, 5],
    cooldown_bars=[0, 10],
    min_strength=[0.0, 0.3, 0.5],
)  # 2*2*2*3 = 24 combos

EXEC_GRID_HOURLY = dict(
    consec_entry=[1, 2],
    min_hold=[0, 18],
    cooldown_bars=[0, 18],
    min_strength=[0.0, 0.3, 0.5],
)  # 2*2*2*3 = 24 combos


# ═══════════════════════════════════════════════════════════════════
# §4. Modified RM Grids for V4
# ═══════════════════════════════════════════════════════════════════

RM_GRIDS_V4 = {
    ("Contrarian", "daily"): dict(
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[2.0, 3.0, 5.0], max_hold=[15, 20, 30],
        breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5],
        partial_exit=[False]),
    ("Contrarian", "hourly"): dict(
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[2.0, 3.0, 5.0], max_hold=[30, 40, 60],
        breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5],
        partial_exit=[False]),
    ("Range", "daily"): dict(
        sl_mult=[0.75, 1.0, 1.5], tp_mult=[1.0, 1.5, 2.0], max_hold=[8, 12, 16],
        breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5],
        time_decay=[False]),
    ("Range", "hourly"): dict(
        sl_mult=[0.75, 1.0, 1.5], tp_mult=[1.0, 1.5, 2.0], max_hold=[16, 24, 32],
        breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5],
        time_decay=[False]),
}

def _build_trend_rm_grid_v4():
    """Build Trend RM grid V4: add large max_hold values and wider trail."""
    grid = []
    for isl in [2.0, 2.5, 3.0]:
        for be in [None, 1.0]:
            for cd in [0, 5]:
                for tt in ["fixed_atr", "chandelier"]:
                    for tn in [10, 15, 20]:
                        for tam in [2.0, 3.0, 4.0, 5.0]:
                            grid.append(dict(
                                initial_sl_mult=isl, trail_type=tt,
                                trail_n=tn, trail_atr_mult=tam,
                                breakeven_thresh=be, cooldown_bars=cd,
                                parabolic_step=0.02, parabolic_max=0.15))
                for ps in [0.01, 0.02]:
                    for pm in [0.10, 0.15]:
                        grid.append(dict(
                            initial_sl_mult=isl, trail_type="parabolic_step",
                            trail_n=10, trail_atr_mult=2.5,
                            breakeven_thresh=be, cooldown_bars=cd,
                            parabolic_step=ps, parabolic_max=pm))
    # OFF max_hold variants: disable trail, use max_hold only
    for isl in [2.0, 3.0]:
        for mh in [30, 50, 100, 200]:
            for cd in [0, 5]:
                grid.append(dict(
                    initial_sl_mult=isl, trail_type="fixed_atr",
                    trail_n=mh, trail_atr_mult=99.0,
                    breakeven_thresh=None, cooldown_bars=cd,
                    parabolic_step=0.02, parabolic_max=0.15))
    return grid

TREND_RM_GRID_V4 = _build_trend_rm_grid_v4()


# ═══════════════════════════════════════════════════════════════════
# §5. calc_sharpe_v4 & calc_metrics_v4
# ═══════════════════════════════════════════════════════════════════

def calc_sharpe_v4(positions, log_ret, mask, ann_factor, commission):
    """Net Sharpe with explicit commission (not global)."""
    pos = positions[mask]
    lr = log_ret[mask]
    if len(pos) == 0:
        return -999.0
    dr = pos * lr
    dpos = np.diff(pos, prepend=0.0)
    comm = np.abs(dpos) * commission
    net_r = dr - comm
    s = np.std(net_r, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(net_r) / s * ann_factor


def calc_metrics_v4(positions, log_ret, mask, ann_factor, bars_per_year, commission):
    """Full metrics dict with explicit commission."""
    pos = positions[mask]
    lr = log_ret[mask]
    n = len(pos)
    if n == 0:
        return {}
    dr = pos * lr
    dpos = np.diff(pos, prepend=0.0)
    comm = np.abs(dpos) * commission
    net_r = dr - comm
    active = pos != 0
    n_active = int(active.sum())
    exposure = n_active / n if n > 0 else 0.0
    mean_r = np.mean(net_r)
    std_r = np.std(net_r, ddof=1) if n > 1 else 1e-10
    sharpe = mean_r / std_r * ann_factor if std_r > 1e-12 else 0.0
    ann_ret = mean_r * bars_per_year
    ann_vol = std_r * np.sqrt(bars_per_year)
    cum = np.cumsum(net_r)
    rmax = np.maximum.accumulate(cum)
    dd = cum - rmax
    max_dd = dd.min() if len(dd) > 0 else 0.0
    trades = 0; wins = 0; in_t = False; trade_ret = 0.0
    for i in range(len(pos)):
        if pos[i] != 0 and not in_t:
            trades += 1; in_t = True; trade_ret = 0.0
        if in_t:
            trade_ret += net_r[i]
        if in_t and (pos[i] == 0 or i == len(pos) - 1):
            if trade_ret > 0: wins += 1
            in_t = False
    win_rate = wins / trades * 100 if trades > 0 else 0.0
    tpy = trades / max(n / bars_per_year, 0.5)
    return {"sharpe": round(sharpe, 4), "ann_ret_pct": round(ann_ret * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "max_dd_pct": round(max_dd * 100, 2), "exposure_pct": round(exposure * 100, 2),
            "n_trades": trades, "trades_per_yr": round(tpy, 1),
            "win_rate_pct": round(win_rate, 1)}


def compute_max_dd(positions, log_ret, mask, commission):
    """Compute max drawdown for filtering."""
    pos = positions[mask]
    lr = log_ret[mask]
    if len(pos) == 0:
        return 0.0
    dr = pos * lr
    dpos = np.diff(pos, prepend=0.0)
    net_r = dr - np.abs(dpos) * commission
    cum = np.cumsum(net_r)
    rmax = np.maximum.accumulate(cum)
    dd = cum - rmax
    return dd.min() if len(dd) > 0 else 0.0


def compute_exposure(positions, mask):
    """Compute exposure % on mask."""
    pos = positions[mask]
    n = len(pos)
    if n == 0:
        return 0.0
    return np.count_nonzero(pos) / n * 100


# ═══════════════════════════════════════════════════════════════════
# §6. Coarse-to-Fine Helper
# ═══════════════════════════════════════════════════════════════════

def get_rm_neighbors(rm, rm_grid):
    """Get ±1 step neighbors of rm in the grid.
    Returns list of rm dicts that differ by at most 1 parameter.
    """
    # Find index of rm in grid
    rm_idx = None
    for i, g in enumerate(rm_grid):
        if g == rm:
            rm_idx = i
            break
    if rm_idx is None:
        return []

    neighbors = set()
    # Simple approach: take ±1, ±2 indices nearby
    for delta in [-2, -1, 1, 2]:
        idx = rm_idx + delta
        if 0 <= idx < len(rm_grid):
            neighbors.add(idx)

    return [rm_grid[i] for i in sorted(neighbors)]


# ═══════════════════════════════════════════════════════════════════
# §7. Approach A V4 — Coarse-to-Fine Grid Search
# ═══════════════════════════════════════════════════════════════════

def approach_a_v4(sid, tf, close, high, low, volume, open_arr, ind,
                  sma_cache, dc_cache, st_cache, vwap_cache,
                  pivot_data, daily_trend, dates, is_hourly, log_ret):
    """Approach A with coarse-to-fine + execution layer + binary ensemble."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    bpy = 252 * 9 if is_hourly else 252
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)

    sig_grid = expand_grid(SIGNAL_GRIDS[sid])
    cat = CATEGORY[sid]
    if cat == "Trend":
        rm_grid = TREND_RM_GRID_V4
    else:
        rm_grid = expand_grid(RM_GRIDS_V4.get((cat, tf), RM_GRIDS_V3.get((cat, tf), {})))

    results_by_year = {}
    params_by_year = {}
    trades_by_year = {}
    exec_params_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        if test_year == 2026:
            test_end = n
        else:
            _, test_end = _year_bounds(dates, test_year + 1)
        if train_end <= WARMUP or train_end >= n:
            continue

        train_mask[:] = False
        train_mask[WARMUP:train_end] = True

        # Cache signals
        cached_sigs = {}
        for sp in sig_grid:
            key = tuple(sorted(sp.items()))
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, dc_cache, st_cache, vwap_cache,
                                              pivot_data, daily_trend, sp)
            cached_sigs[key] = (sig, exit_info)

        # Cache strength
        cached_strength = {}
        for sp in sig_grid:
            key = tuple(sorted(sp.items()))
            strength = dispatch_strength(sid, close, ind, sma_cache, dc_cache,
                                         st_cache, vwap_cache, pivot_data, sp)
            cached_strength[key] = strength

        # ──── PHASE 1: Coarse grid (subsample RM) ────
        if len(rm_grid) > 200:
            coarse_rm = rm_grid[::3]  # every 3rd for large grids (Trend)
        elif len(rm_grid) > 20:
            coarse_rm = rm_grid[::2]
        else:
            coarse_rm = rm_grid
        coarse_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            for rm in coarse_rm:
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, WARMUP, train_end, 0)
                sh = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                max_dd = compute_max_dd(pos, log_ret, train_mask, commission)
                exposure = compute_exposure(pos, train_mask)
                coarse_results.append((sp, rm, sh, max_dd, exposure))

        # Filter & select top-50
        passing = [r for r in coarse_results
                   if r[2] > 0 and r[3] > -0.30 and r[4] > 5]
        if not passing:
            passing = sorted(coarse_results, key=lambda x: x[2], reverse=True)[:10]
        top50 = sorted(passing, key=lambda x: x[2], reverse=True)[:50]

        # ──── PHASE 2: Fine grid (neighbors ±1 step) ────
        fine_results = []
        seen = set()
        for sp, rm, sh, mdd, exp in top50:
            seen.add((tuple(sorted(sp.items())), id(rm)))
            neighbors = get_rm_neighbors(rm, rm_grid)
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            for rm_n in neighbors:
                combo_key = (sp_key, id(rm_n))
                if combo_key in seen:
                    continue
                seen.add(combo_key)
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm_n, WARMUP, train_end, 0)
                sh_n = calc_sharpe_v4(pos, log_ret, train_mask, ann, commission)
                mdd_n = compute_max_dd(pos, log_ret, train_mask, commission)
                exp_n = compute_exposure(pos, train_mask)
                fine_results.append((sp, rm_n, sh_n, mdd_n, exp_n))

        # Combine with top50, deduplicate by (sp, rm), select top-30
        all_phase2 = top50 + fine_results
        # Deduplicate: keep best Sharpe per (sp_key, rm dict repr)
        dedup = {}
        for sp, rm, sh, mdd, exp in all_phase2:
            key = (tuple(sorted(sp.items())), str(sorted(rm.items())))
            if key not in dedup or sh > dedup[key][2]:
                dedup[key] = (sp, rm, sh, mdd, exp)
        all_dedup = list(dedup.values())
        top30 = sorted(all_dedup, key=lambda x: x[2], reverse=True)[:20]

        # ──── PHASE 3: Execution layer grid (top-20 × 24 exec combos) ────
        # Cache raw positions to avoid re-running BT
        exec_grid = expand_grid(EXEC_GRID_HOURLY if is_hourly else EXEC_GRID_DAILY)
        cached_raw_pos = {}
        for sp, rm, sh_raw, mdd, exp in top30:
            sp_key = tuple(sorted(sp.items()))
            rm_key = str(sorted(rm.items()))
            cache_key = (sp_key, rm_key)
            if cache_key not in cached_raw_pos:
                sig, exit_info = cached_sigs[sp_key]
                pos_raw, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                                rm, WARMUP, train_end, 0)
                cached_raw_pos[cache_key] = pos_raw

        best_combos = []
        for sp, rm, sh_raw, mdd, exp in top30:
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

        # Select top-10 by net Sharpe
        top10 = sorted(best_combos, key=lambda x: x[3], reverse=True)[:10]

        # ──── Build ensemble ────
        if not top10:
            continue

        test_positions = []
        test_trades = []
        for sp, rm, ep, sh in top10:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            pos_raw, tr = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                             rm, WARMUP, test_end, 1)
            strength = cached_strength[sp_key]
            pos_exec = execution_layer(pos_raw, strength,
                                       ep["consec_entry"], ep["min_hold"],
                                       ep["cooldown_bars"], ep["min_strength"])
            test_positions.append(pos_exec)
            test_trades.append(tr)

        # Majority vote ensemble: binary
        pos_stack = np.array(test_positions)  # shape (K, n)
        ensemble_magnitude = np.mean(np.abs(pos_stack), axis=0)
        ensemble_direction = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ensemble_magnitude > 0.5, 1.0, 0.0) * ensemble_direction

        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm, ep) for sp, rm, ep, sh in top10]
        trades_by_year[test_year] = test_trades
        exec_params_by_year[test_year] = top10[0][2] if top10 else {}

    return results_by_year, params_by_year, trades_by_year, exec_params_by_year


# ═══════════════════════════════════════════════════════════════════
# §8. Approach B V4
# ═══════════════════════════════════════════════════════════════════

def approach_b_v4(sid, tf, close, high, low, volume, ind,
                  sma_cache, dc_cache, st_cache, vwap_cache,
                  pivot_data, daily_trend, dates, sigma_dict, is_hourly,
                  log_ret, a_params_by_year, exec_params_by_year):
    """Approach B with commission and execution layer from A."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    n = len(close)
    cat = CATEGORY[sid]
    b_grid = expand_grid(B_GRIDS_V3[cat])

    results_by_year = {}
    best_params_by_year = {}
    trades_by_year = {}

    for test_year in BCD_TEST_YEARS:
        _, val_start = _year_bounds(dates, 2020)
        _, val_end = _year_bounds(dates, test_year)
        if test_year == 2026:
            test_end = n
        else:
            _, test_end = _year_bounds(dates, test_year + 1)

        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True
        if val_mask.sum() == 0:
            continue

        a_params = a_params_by_year.get(test_year, [])
        if not a_params:
            continue

        # Get exec params from A
        ep = exec_params_by_year.get(test_year, {})
        consec = ep.get("consec_entry", 1)
        mhold = ep.get("min_hold", 0)
        cool = ep.get("cooldown_bars", 0)
        mstr = ep.get("min_strength", 0.0)

        # Prepare A signals
        a_sigs = []
        a_strengths = []
        for item in a_params:
            sp = item[0]
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, dc_cache, st_cache, vwap_cache,
                                              pivot_data, daily_trend, sp)
            strength = dispatch_strength(sid, close, ind, sma_cache, dc_cache,
                                         st_cache, vwap_cache, pivot_data, sp)
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

            # Majority vote
            pos_stack = np.array(val_positions)
            ens_mag = np.mean(np.abs(pos_stack), axis=0)
            ens_dir = np.sign(np.sum(pos_stack, axis=0))
            ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir
            sh = calc_sharpe_v4(ensemble, log_ret, val_mask, ann, commission)
            if sh > best_sh:
                best_sh = sh
                best_bp = dict(bp)

        # Run best on test
        best_horizon = best_bp.get("horizon", "h1")
        best_sigma = sigma_dict.get(best_horizon, sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        best_sigma_median = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        best_bp_engine = {k: v for k, v in best_bp.items() if k not in ("horizon", "ratio")}
        if "ratio" in best_bp:
            best_bp_engine["k_tp"] = best_bp["k_sl"] * best_bp["ratio"]

        test_positions = []
        test_trades = []
        for (sig, exit_info), strength in zip(a_sigs, a_strengths):
            pos_raw, tr = dispatch_bt_vpred_v3(sid, sig, exit_info, close, high, low, ind,
                                               best_sigma, best_bp_engine, WARMUP, test_end,
                                               is_hourly, best_sigma_median, 1)
            pos_exec = execution_layer(pos_raw, strength, consec, mhold, cool, mstr)
            test_positions.append(pos_exec)
            test_trades.append(tr)

        pos_stack = np.array(test_positions)
        ens_mag = np.mean(np.abs(pos_stack), axis=0)
        ens_dir = np.sign(np.sum(pos_stack, axis=0))
        ensemble = np.where(ens_mag > 0.5, 1.0, 0.0) * ens_dir

        results_by_year[test_year] = ensemble
        best_params_by_year[test_year] = best_bp
        trades_by_year[test_year] = test_trades

    return results_by_year, best_params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# §9. Approach C V4 — Regime filter with commission
# ═══════════════════════════════════════════════════════════════════

def _c_grid_search_v4(cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
                      bars_per_year, n, min_exp, min_tpy, commission):
    """Inner grid search for approach C with explicit commission. Returns (best_sharpe, best_params)."""
    best_sh = -999.0
    best_cp = {}

    if cat in ("Trend", "Contrarian"):
        for threshold in C_THRESHOLDS:
            for lookback_base in C_LOOKBACKS:
                lookback = lookback_base * 9 if is_hourly else lookback_base
                for horizon in C_HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None: continue
                    sigma_h22 = sigma_dict.get("h22")
                    sigma_h1 = sigma_dict.get("h1")
                    for direction in C_DIRECTION:
                        for term_f in C_TERM_FILTER:
                            if term_f and horizon == "h22": continue
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
        for band in C_RANGE_BANDS:
            for lookback_base in C_LOOKBACKS:
                lookback = lookback_base * 9 if is_hourly else lookback_base
                for horizon in C_HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None: continue
                    sigma_h22 = sigma_dict.get("h22")
                    sigma_h1 = sigma_dict.get("h1")
                    for direction in C_DIRECTION:
                        for term_f in C_TERM_FILTER:
                            if term_f and horizon == "h22": continue
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


def approach_c_v4(sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n):
    """Approach C V4 — regime filter on binary A positions with commission."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    bpy = 252 * 9 if is_hourly else 252
    cat = CATEGORY[sid]
    results_by_year = {}
    best_params_by_year = {}
    trades_by_year = {}

    for test_year in BCD_TEST_YEARS:
        _, val_start = _year_bounds(dates, 2020)
        _, val_end = _year_bounds(dates, test_year)
        if test_year == 2026:
            test_end = n
        else:
            _, test_end = _year_bounds(dates, test_year + 1)

        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True

        a_pos = a_results.get(test_year)
        if a_pos is None: continue

        # 2-pass: strict constraints first, then fallback
        best_sh, best_cp = _c_grid_search_v4(
            cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
            bpy, n, C_MIN_EXPOSURE, C_MIN_TRADES_YR, commission)
        if not best_cp:
            best_sh, best_cp = _c_grid_search_v4(
                cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
                bpy, n, C_FALLBACK_EXPOSURE, C_FALLBACK_TRADES_YR, commission)

        if not best_cp:
            # Fallback: keep A's positions when regime filter can't improve
            results_by_year[test_year] = a_pos
            best_params_by_year[test_year] = {"fallback": True}
            continue

        sigma = sigma_dict.get(best_cp.get("horizon", "h1"))
        if sigma is None:
            results_by_year[test_year] = a_pos
            best_params_by_year[test_year] = {"fallback": True}
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

    return results_by_year, best_params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# §10. Approach D V4 — Binary Vol-Gate
# ═══════════════════════════════════════════════════════════════════

D_GATE_THRESHOLDS = [0.3, 0.5, 0.7, 1.0]

def approach_d_v4(sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n):
    """Approach D V4 — binary vol-gate: if scale < threshold → off, else keep A's position."""
    commission = COMM_HOURLY if is_hourly else COMM_DAILY
    ann = sqrt(252 * 9) if is_hourly else sqrt(252)
    results_by_year = {}
    best_params_by_year = {}
    trades_by_year = {}

    for test_year in BCD_TEST_YEARS:
        _, val_start = _year_bounds(dates, 2020)
        _, val_end = _year_bounds(dates, test_year)
        if test_year == 2026:
            test_end = n
        else:
            _, test_end = _year_bounds(dates, test_year + 1)

        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True

        a_pos = a_results.get(test_year)
        if a_pos is None: continue

        best_sh = -999.0
        best_dp = {}

        # Standard vol-targeting as binary gate
        for tv in D_TARGET_VOLS:
            for ml in D_MAX_LEVS:
                for gamma in D_GAMMA:
                    for vf in D_VOL_FLOORS:
                        for vc in D_VOL_CAPS:
                            for smooth in D_SMOOTH:
                                for horizon in D_HORIZONS:
                                    sigma = sigma_dict.get(horizon)
                                    if sigma is None: continue
                                    dp = {"target_vol": tv, "max_lev": ml, "gamma": gamma,
                                          "vol_floor": vf, "vol_cap": vc,
                                          "smooth_span": smooth,
                                          "horizon": horizon, "inverse_vol": False}
                                    scale = _compute_d_scale_v3(sigma, dp)
                                    # Binary gate: try each threshold
                                    for gate_th in D_GATE_THRESHOLDS:
                                        gate = (scale >= gate_th).astype(np.float64)
                                        gated_pos = a_pos * gate
                                        sh = calc_sharpe_v4(gated_pos, log_ret, val_mask, ann, commission)
                                        if sh > best_sh:
                                            best_sh = sh
                                            best_dp = dict(dp)
                                            best_dp["gate_threshold"] = gate_th

        # Inverse vol as binary gate
        for inv_lb in D_INV_LOOKBACKS:
            for ml in D_MAX_LEVS:
                for smooth in D_SMOOTH:
                    for horizon in D_HORIZONS:
                        sigma = sigma_dict.get(horizon)
                        if sigma is None: continue
                        dp = {"max_lev": ml, "inv_lookback": inv_lb,
                              "smooth_span": smooth,
                              "horizon": horizon, "inverse_vol": True}
                        scale = _compute_d_scale_v3(sigma, dp)
                        for gate_th in D_GATE_THRESHOLDS:
                            gate = (scale >= gate_th).astype(np.float64)
                            gated_pos = a_pos * gate
                            sh = calc_sharpe_v4(gated_pos, log_ret, val_mask, ann, commission)
                            if sh > best_sh:
                                best_sh = sh
                                best_dp = dict(dp)
                                best_dp["gate_threshold"] = gate_th

        if not best_dp: continue

        sigma = sigma_dict.get(best_dp.get("horizon", "h1"))
        if sigma is None: continue
        scale = _compute_d_scale_v3(sigma, best_dp)
        gate_th = best_dp.get("gate_threshold", 0.5)
        gate = (scale >= gate_th).astype(np.float64)
        final_pos = a_pos * gate
        results_by_year[test_year] = final_pos
        best_params_by_year[test_year] = best_dp

    return results_by_year, best_params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# §11. Metrics storage helper
# ═══════════════════════════════════════════════════════════════════

def _store_metrics_v4(results, sid, tf, approach, year, pos_arr,
                      log_ret, dates, n, ann, bpy, ticker, commission):
    """Store annual metrics in results dict using V4 commission."""
    ys, _ = _year_bounds(dates, year)
    ye_actual = _year_bounds(dates, year + 1)[0] if year < 2026 else n
    test_mask = np.zeros(n, dtype=bool)
    test_mask[ys:ye_actual] = True
    met = calc_metrics_v4(pos_arr, log_ret, test_mask, ann, bpy, commission)
    if not met:
        return
    met["year"] = year
    met["approach"] = approach
    met["strategy"] = STRATEGY_NAMES[sid]
    met["timeframe"] = tf
    met["ticker"] = ticker

    pos_test = pos_arr[test_mask]
    lr_test = log_ret[test_mask]
    dr = pos_test * lr_test
    dpos = np.diff(pos_test, prepend=0.0)
    net_r = dr - np.abs(dpos) * commission
    met["_net_returns"] = net_r

    key = (sid, tf, approach, year)
    results[key] = met


# ═══════════════════════════════════════════════════════════════════
# §12. process_ticker_v4
# ═══════════════════════════════════════════════════════════════════

def process_ticker_v4(ticker, daily_df, hourly_df, vpred_df):
    """Process one ticker: run A/B/C/D for all strategies x TFs (V4).
    Returns dict with keys: results, trades, positions.
    """
    results = {}
    trade_rows = []
    position_rows = []

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

        sma_windows = sorted(set(SIGNAL_GRIDS["S1"].get("ma_window", []) +
                                 SIGNAL_GRIDS["S2"].get("bb_window", [])))
        sma_cache = precompute_sma_cache(close, sma_windows)
        dc_windows = SIGNAL_GRIDS["S3"].get("dc_window", [])
        dc_cache = precompute_donchian_cache(high, low, dc_windows)
        st_periods = SIGNAL_GRIDS["S4"].get("atr_period", [])
        st_mults = SIGNAL_GRIDS["S4"].get("multiplier", [])
        st_cache = precompute_supertrend_cache(high, low, close, st_periods, st_mults)
        vwap_windows = SIGNAL_GRIDS["S6"].get("vwap_window", [])
        vwap_cache = precompute_vwap_cache(close, high, low, volume, vwap_windows)

        pivot_data = {}
        if is_hourly:
            h_dts = tdf["datetime"].values
            P, S1p, R1p = compute_daily_pivots_for_hourly(daily_df, ticker, h_dts, "classic")
        else:
            P, S1p, R1p = calc_pivot_daily(high, low, close, "classic")
        pivot_data["classic"] = (P, S1p, R1p)

        daily_trend = None
        if is_hourly:
            d_dates, d_above = build_daily_trend(daily_df, ticker)
            daily_trend = align_daily_to_hourly(d_dates, d_above, tdf["datetime"].values)

        # Prepare sigma_dict (v2_sqrtN for hourly, raw for daily)
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
                    dates_arr = tdf["datetime"].values
                    nhours = _compute_nhours_per_day(dates_arr)
                    nhours_safe = np.maximum(nhours, 1.0)
                    sigma_arr = sigma_arr / np.sqrt(nhours_safe)
                sigma_dict[hname] = sigma_arr

        ann = sqrt(252 * 9) if is_hourly else sqrt(252)
        bpy = 252 * 9 if is_hourly else 252

        for sid in STRATEGY_IDS:
            sname = STRATEGY_NAMES[sid]

            # Approach A
            a_results, a_params, a_trades, a_exec_params = approach_a_v4(
                sid, tf, close, high, low, volume, open_arr, ind,
                sma_cache, dc_cache, st_cache, vwap_cache,
                pivot_data, daily_trend, dates, is_hourly, log_ret)

            for year, pos_arr in a_results.items():
                _store_metrics_v4(results, sid, tf, "A", year, pos_arr,
                                  log_ret, dates, n, ann, bpy, ticker, commission)

            # Collect A trades
            for year, trades_list in a_trades.items():
                trade_rows.extend(_trades_to_rows_v4(trades_list, sname, tf, ticker, "A", year, dates))

            # Approach B
            b_results, b_params, b_trades = approach_b_v4(
                sid, tf, close, high, low, volume, ind,
                sma_cache, dc_cache, st_cache, vwap_cache,
                pivot_data, daily_trend, dates, sigma_dict, is_hourly,
                log_ret, a_params, a_exec_params)

            for year, trades_list in b_trades.items():
                trade_rows.extend(_trades_to_rows_v4(trades_list, sname, tf, ticker, "B", year, dates))

            # Approach C
            c_results, c_params, c_trades = approach_c_v4(
                sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

            for year, pos_arr in c_results.items():
                test_start_c = _year_bounds(dates, year)[0]
                test_end_c = _year_bounds(dates, year + 1)[0] if year < 2026 else n
                c_trades[year] = _extract_trades_from_pos(pos_arr, close, test_start_c, test_end_c)

            for year, trades_arr in c_trades.items():
                trade_rows.extend(_trades_to_rows_v4(trades_arr, sname, tf, ticker, "C", year, dates))

            # Approach D
            d_results, d_params, d_trades = approach_d_v4(
                sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

            for year, pos_arr in d_results.items():
                test_start_d = _year_bounds(dates, year)[0]
                test_end_d = _year_bounds(dates, year + 1)[0] if year < 2026 else n
                d_trades[year] = _extract_trades_from_pos(pos_arr, close, test_start_d, test_end_d)

            for year, trades_arr in d_trades.items():
                trade_rows.extend(_trades_to_rows_v4(trades_arr, sname, tf, ticker, "D", year, dates))

            for approach, res in [("B", b_results), ("C", c_results), ("D", d_results)]:
                for year, pos_arr in res.items():
                    _store_metrics_v4(results, sid, tf, approach, year, pos_arr,
                                      log_ret, dates, n, ann, bpy, ticker, commission)

            # Collect daily positions for all approaches
            for approach, res, test_years in [("A", a_results, A_TEST_YEARS),
                                               ("B", b_results, BCD_TEST_YEARS),
                                               ("C", c_results, BCD_TEST_YEARS),
                                               ("D", d_results, BCD_TEST_YEARS)]:
                for year in test_years:
                    pos_arr = res.get(year)
                    if pos_arr is None:
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

    return {"results": results, "trades": trade_rows, "positions": position_rows}


# ═══════════════════════════════════════════════════════════════════
# §13. Trade row conversion helper
# ═══════════════════════════════════════════════════════════════════

def _trades_to_rows_v4(trades_list_or_arr, strategy_name, tf, ticker, approach, test_year, dates):
    """Convert trade arrays to list of dicts for DataFrame."""
    rows = []
    if isinstance(trades_list_or_arr, list):
        pos_size = 1.0 / max(len(trades_list_or_arr), 1)
        for trades_arr in trades_list_or_arr:
            if len(trades_arr) == 0:
                continue
            for i in range(len(trades_arr)):
                eb = int(trades_arr[i, 0])
                xb = int(trades_arr[i, 1])
                dirn = int(trades_arr[i, 2])
                ep = trades_arr[i, 3]
                xp = trades_arr[i, 4]
                er = int(trades_arr[i, 5])
                hb = int(trades_arr[i, 6])
                if ep > 0:
                    gross_ret = dirn * np.log(xp / ep) if xp > 0 and ep > 0 else 0.0
                else:
                    gross_ret = 0.0
                rows.append({
                    "strategy": strategy_name, "tf": tf, "ticker": ticker,
                    "approach": approach, "test_year": test_year,
                    "entry_date": dates[eb] if eb < len(dates) else None,
                    "entry_price": round(ep, 4),
                    "exit_date": dates[xb] if xb < len(dates) else None,
                    "exit_price": round(xp, 4),
                    "direction": "long" if dirn > 0 else "short",
                    "position_size": round(pos_size, 4),
                    "gross_return": round(gross_ret, 6),
                    "holding_bars": hb,
                    "exit_reason": EXIT_REASON_NAMES.get(er, str(er)),
                })
    else:
        if len(trades_list_or_arr) == 0:
            return rows
        for i in range(len(trades_list_or_arr)):
            eb = int(trades_list_or_arr[i, 0])
            xb = int(trades_list_or_arr[i, 1])
            dirn = int(trades_list_or_arr[i, 2])
            ep = trades_list_or_arr[i, 3]
            xp = trades_list_or_arr[i, 4]
            er = int(trades_list_or_arr[i, 5])
            hb = int(trades_list_or_arr[i, 6])
            if ep > 0:
                gross_ret = dirn * np.log(xp / ep) if xp > 0 and ep > 0 else 0.0
            else:
                gross_ret = 0.0
            rows.append({
                "strategy": strategy_name, "tf": tf, "ticker": ticker,
                "approach": approach, "test_year": test_year,
                "entry_date": dates[eb] if eb < len(dates) else None,
                "entry_price": round(ep, 4),
                "exit_date": dates[xb] if xb < len(dates) else None,
                "exit_price": round(xp, 4),
                "direction": "long" if dirn > 0 else "short",
                "position_size": 1.0,
                "gross_return": round(gross_ret, 6),
                "holding_bars": hb,
                "exit_reason": EXIT_REASON_NAMES.get(er, str(er)),
            })
    return rows


# ═══════════════════════════════════════════════════════════════════
# §14. Output Generation
# ═══════════════════════════════════════════════════════════════════

def generate_outputs_v4(all_results, all_trade_rows, all_position_rows):
    """Generate 5 required tables + support files."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    rows = []
    net_returns_store = {}
    for ticker, ticker_results in all_results.items():
        for key, met in ticker_results.items():
            sid, tf, approach, year = key
            nr = met.pop("_net_returns", np.array([]))
            rows.append(met)
            nr_key = (STRATEGY_NAMES[sid], tf, approach, year, ticker)
            net_returns_store[nr_key] = nr

    if not rows:
        print("  No results to output!")
        return

    df = pd.DataFrame(rows)

    # ── Table 5: Statistical tests ──
    stat_rows = []
    stat_lookup = {}
    for strat in df["strategy"].unique():
        for tf in TIMEFRAMES:
            for comp_approach in BCD_APPROACHES:
                a_rets = []
                b_rets = []
                for year in BCD_TEST_YEARS:
                    for tkr in TICKERS:
                        a_key = (strat, tf, "A", year, tkr)
                        b_key = (strat, tf, comp_approach, year, tkr)
                        if a_key in net_returns_store and b_key in net_returns_store:
                            a_rets.append(net_returns_store[a_key])
                            b_rets.append(net_returns_store[b_key])
                if a_rets and b_rets:
                    a_all = np.concatenate(a_rets)
                    b_all = np.concatenate(b_rets)
                    min_len = min(len(a_all), len(b_all))
                    if min_len > 10:
                        t_stat, p_val = dm_test(b_all[:min_len], a_all[:min_len])
                        bs_mean, bs_lo, bs_hi = bootstrap_sharpe_diff(
                            b_all[:min_len], a_all[:min_len])
                        sig_str = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
                        row_dict = {
                            "strategy": strat, "timeframe": tf,
                            "comparison": f"{comp_approach}_vs_A",
                            "dm_t": t_stat, "dm_p": p_val,
                            "bs_mean": bs_mean, "bs_ci_lo": bs_lo, "bs_ci_hi": bs_hi,
                            "sig": sig_str
                        }
                        stat_rows.append(row_dict)
                        stat_lookup[(strat, tf, f"{comp_approach}_vs_A")] = row_dict

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(OUT_TABLES / "v4_stat_tests.csv", index=False)
        print("\n  === Statistical Tests (B-D vs A, V4) ===")
        print(stat_df.to_string(index=False))

    # ── Per strategy x TF detailed tables (12 tables) ──
    for sid in STRATEGY_IDS:
        sname = STRATEGY_NAMES[sid]
        for tf in TIMEFRAMES:
            sub = df[(df["strategy"] == sname) & (df["timeframe"] == tf)]
            if len(sub) == 0:
                continue

            agg = sub.groupby(["approach", "year"]).agg(
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

            fname = f"v4_{sid}_{tf}.csv"
            summary.to_csv(OUT_TABLES / fname, index=False)
            print(f"\n  === {sname} {tf} (V4) ===")
            hdr = f"  {'approach':>8s}  {'MeanSharpe':>10s}  {'AnnRet%':>8s}  {'AnnVol%':>8s}  {'MaxDD%':>7s}  {'Exp%':>6s}  {'Tr/yr':>6s}  {'Win%':>6s}"
            print(hdr)
            a_sharpe_val = None
            for _, r in summary.iterrows():
                if r["approach"] == "A":
                    a_sharpe_val = r["MeanSharpe"]
                print(f"  {r['approach']:>8s}  {r['MeanSharpe']:10.4f}  {r['AnnReturn']:8.2f}  {r['AnnVol']:8.2f}  {r['MaxDD']:7.2f}  {r['Exposure']:6.2f}  {r['TradesPerYr']:6.2f}  {r['WinRate']:6.2f}")

            if a_sharpe_val is not None:
                delta_parts = []
                pval_parts = []
                for appr in BCD_APPROACHES:
                    row_s = summary.loc[summary["approach"] == appr]
                    if len(row_s) > 0:
                        delta = row_s["MeanSharpe"].values[0] - a_sharpe_val
                        delta_parts.append(f"{appr}={delta:+.3f}")
                    sl = stat_lookup.get((sname, tf, f"{appr}_vs_A"))
                    if sl:
                        pval_parts.append(f"{appr}={sl['dm_p']:.3f}{sl['sig']}")
                if delta_parts:
                    print(f"  \u0394Sharpe vs A:  {', '.join(delta_parts)}")
                if pval_parts:
                    print(f"  p-value vs A:  {', '.join(pval_parts)}")

            # Top-5 / Worst-5 tickers
            sub_common = sub[sub["year"].isin(BCD_TEST_YEARS)]
            if len(sub_common) > 0:
                ticker_agg = sub_common.groupby(["approach", "ticker"]).agg(
                    sharpe=("sharpe", "median")
                ).reset_index()
                best_appr = summary.loc[summary["MeanSharpe"].idxmax(), "approach"]
                asub = ticker_agg[ticker_agg["approach"] == best_appr].sort_values("sharpe", ascending=False)
                if len(asub) >= 5:
                    top5 = asub.head(5)
                    worst5 = asub.tail(5)
                    top_str = ", ".join(f"{r['ticker']}({r['sharpe']:.2f})" for _, r in top5.iterrows())
                    worst_str = ", ".join(f"{r['ticker']}({r['sharpe']:.2f})" for _, r in worst5.iterrows())
                    print(f"  Top-5 ({best_appr}):   {top_str}")
                    print(f"  Worst-5 ({best_appr}): {worst_str}")

    # ── Table 1: v4_summary_sharpe.csv ──
    common = df[df["year"].isin(BCD_TEST_YEARS)]
    if len(common) > 0:
        mean_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            sharpe=("sharpe", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="sharpe"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in mean_pivot.columns]
        mean_pivot = mean_pivot[["strategy", "timeframe"] + approach_cols].round(4)
        mean_pivot.to_csv(OUT_TABLES / "v4_summary_sharpe.csv", index=False)
        print("\n  === Summary: Mean Net Sharpe (V4) ===")
        print(mean_pivot.to_string(index=False))

        # ── Table 2: v4_summary_trades.csv ──
        trades_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            n_trades=("n_trades", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="n_trades"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in trades_pivot.columns]
        trades_pivot = trades_pivot[["strategy", "timeframe"] + approach_cols].round(2)
        trades_pivot.to_csv(OUT_TABLES / "v4_summary_trades.csv", index=False)
        print("\n  === Summary: Trades/yr (V4) ===")
        print(trades_pivot.to_string(index=False))

        # ── Table 3: v4_summary_return.csv ──
        ret_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            ann_ret_pct=("ann_ret_pct", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="ann_ret_pct"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in ret_pivot.columns]
        ret_pivot = ret_pivot[["strategy", "timeframe"] + approach_cols].round(2)
        ret_pivot.to_csv(OUT_TABLES / "v4_summary_return.csv", index=False)
        print("\n  === Summary: Ann Return % (V4) ===")
        print(ret_pivot.to_string(index=False))

        # Exposure pivot
        exp_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            exposure_pct=("exposure_pct", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="exposure_pct"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in exp_pivot.columns]
        exp_pivot = exp_pivot[["strategy", "timeframe"] + approach_cols].round(2)
        exp_pivot.to_csv(OUT_TABLES / "v4_summary_exposure.csv", index=False)
        print("\n  === Summary: Exposure % (V4) ===")
        print(exp_pivot.to_string(index=False))

    # ── Table 4: v4_comparison_v3.csv ──
    # Load V3 results if available
    v3_path = BASE / "results" / "final" / "strategies" / "walkforward_v3" / "data" / "wf_v3_all_results.csv"
    if v3_path.exists() and len(common) > 0:
        v3_df = pd.read_csv(v3_path)
        v3_common = v3_df[v3_df["year"].isin(BCD_TEST_YEARS)]
        if len(v3_common) > 0:
            v3_agg = v3_common.groupby(["strategy", "timeframe", "approach"]).agg(
                v3_sharpe=("sharpe", "mean"),
            ).reset_index()
            v4_agg = common.groupby(["strategy", "timeframe", "approach"]).agg(
                v4_sharpe=("sharpe", "mean"),
            ).reset_index()
            comp = v4_agg.merge(v3_agg, on=["strategy", "timeframe", "approach"], how="left")
            comp["delta"] = comp["v4_sharpe"] - comp["v3_sharpe"]
            comp = comp.round(4)
            comp.to_csv(OUT_TABLES / "v4_comparison_v3.csv", index=False)
            print("\n  === V4 vs V3 Comparison ===")
            print(comp.to_string(index=False))

    # ── B vs A Holding Bars Diagnosis ──
    if all_trade_rows:
        trade_df_diag = pd.DataFrame(all_trade_rows)
        if "holding_bars" in trade_df_diag.columns and "approach" in trade_df_diag.columns:
            print("\n  === B vs A Holding Bars Diagnosis (V4) ===")
            for appr in ["A", "B"]:
                tsub = trade_df_diag[trade_df_diag["approach"] == appr]
                if len(tsub) > 0:
                    mean_h = tsub["holding_bars"].mean()
                    med_h = tsub["holding_bars"].median()
                    n_tr = len(tsub)
                    print(f"    {appr}: mean_hold={mean_h:.1f}, median_hold={med_h:.0f}, trades={n_tr:,}")

    # ── Raw results ──
    df.to_csv(OUT_DATA / "wf_v4_all_results.csv", index=False)
    print(f"\n  Full results saved: {len(df)} rows")

    # ── Sanity checks ──
    print("\n  === Sanity Checks (V4) ===")
    for appr in ALL_APPROACHES:
        asub = df[df["approach"] == appr]
        if len(asub) > 0:
            exp = asub["exposure_pct"].mean()
            trades = asub["n_trades"].mean()
            print(f"    {appr}: avg exposure={exp:.1f}%, avg trades={trades:.1f}")

    # ── Binary position check ──
    print("\n  === Binary Position Check ===")
    binary_ok = True
    for appr in ALL_APPROACHES:
        asub = df[df["approach"] == appr]
        if len(asub) > 0:
            # Check from position_rows
            pass  # Will check in main after collecting positions
    print("    (checked in main)")

    # ── Trade log ──
    if all_trade_rows:
        trade_df = pd.DataFrame(all_trade_rows)
        trade_path = OUT_DIR / "trade_log.parquet"
        trade_df.to_parquet(trade_path, index=False)
        print(f"\n  Trade log saved: {len(trade_df)} trades -> {trade_path}")
        if "exit_reason" in trade_df.columns:
            reason_counts = trade_df["exit_reason"].value_counts()
            print("  Exit reasons:")
            for reason, count in reason_counts.items():
                print(f"    {reason}: {count} ({count/len(trade_df)*100:.1f}%)")

    # ── Daily positions ──
    if all_position_rows:
        pos_df = pd.DataFrame(all_position_rows,
                              columns=["date", "strategy", "tf", "ticker", "approach",
                                       "test_year", "position", "daily_gross_return"])
        pos_path = OUT_DIR / "daily_positions.parquet"
        pos_df.to_parquet(pos_path, index=False)
        print(f"  Daily positions saved: {len(pos_df)} rows -> {pos_path}")

        # Binary check
        unique_pos = pos_df["position"].unique()
        non_binary = [p for p in unique_pos if abs(p) > 0.01 and abs(abs(p) - 1.0) > 0.01]
        if non_binary:
            print(f"  WARNING: Non-binary positions found: {non_binary[:10]}")
        else:
            print("  All positions are binary {-1, 0, +1}")


# ═══════════════════════════════════════════════════════════════════
# §15. Multiprocessing helpers
# ═══════════════════════════════════════════════════════════════════

_G_DAILY = None
_G_HOURLY = None
_G_VPRED = None


def _pool_init_v4(daily, hourly, vpred):
    global _G_DAILY, _G_HOURLY, _G_VPRED
    _G_DAILY = daily
    _G_HOURLY = hourly
    _G_VPRED = vpred


def _worker_func_v4(ticker):
    return ticker, process_ticker_v4(ticker, _G_DAILY, _G_HOURLY, _G_VPRED)


def warmup_numba_v4():
    """JIT-compile all numba functions with dummy data."""
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
    rh = h.copy(); rl = l.copy()
    ex_l = np.zeros(n, dtype=bool); ex_s = np.zeros(n, dtype=bool)

    bt_contrarian_v3(sig, ex, c, h, l, a, 1.5, 2.0, 10, 1.0, 5, 1, 5, n, 0)
    bt_trend_v3(sig, ex_l, ex_s, c, h, l, a, rh, rl, 2.5, 2.5, 1.5, 0, 0.02, 0.15, 5, 5, n, 0)
    bt_range_v3(sig, ex, c, h, l, a, adx, 1.0, 1.5, 30.0, 10, 1.0, 5, 1, 5, n, 0)
    bt_range_split_v3(sig, ex_l, ex_s, c, h, l, a, adx, 1.0, 1.5, 30.0, 10, 1.0, 5, 1, 5, n, 0)
    bt_contrarian_vpred_v3(sig, ex, c, h, l, sigma, 1.0, 2.0, 20, 1.0, 0.5, 0.02, 5, 5, n, 0)
    bt_trend_vpred_v3(sig, ex_l, ex_s, c, h, l, sigma, rh, rl, 1.0, 1.0, 1.0, 0.5, 0.02, 30, 5, 5, n, 0)
    bt_range_vpred_v3(sig, ex, c, h, l, sigma, adx, 1.0, 2.0, 30.0, 10, 1.0, 0.5, 0.02, 5, 5, n, 0)
    bt_range_split_vpred_v3(sig, ex_l, ex_s, c, h, l, sigma, adx, 1.0, 2.0, 30.0, 10, 1.0, 0.5, 0.02, 5, 5, n, 0)
    # s5/s6 engines
    sl_d = np.ones(n) * 0.01; tp_d = np.ones(n) * 0.01
    bt_range_pivot_v3(sig, ex, c, h, l, adx, sl_d, tp_d, 30.0, 10, 1.0, 5, 5, n, 0)
    bt_range_vwap_v3(sig, ex, c, h, l, adx, sl_d, tp_d, 30.0, 10, 1.0, 5, 5, n, 0)
    pctrank = np.random.rand(n)
    _apply_hysteresis(pctrank, 1, 0.5, 0.4, n)
    print("  Numba warmup done (V4)")


# ═══════════════════════════════════════════════════════════════════
# §16. main()
# ═══════════════════════════════════════════════════════════════════

def main():
    global _G_DAILY, _G_HOURLY, _G_VPRED

    t0 = time.time()
    print("=" * 70)
    print("Walk-Forward A/B/C/D Strategy Pipeline (V4) — Commission-Optimized")
    print(f"Tickers: {len(TICKERS)}, Strategies: {len(STRATEGY_IDS)}, TFs: {len(TIMEFRAMES)}")
    print(f"A test years: {A_TEST_YEARS[0]}-{A_TEST_YEARS[-1]}")
    print(f"B/C/D test years: {BCD_TEST_YEARS[0]}-{BCD_TEST_YEARS[-1]}")
    print(f"Commission: daily={COMM_DAILY*100:.2f}%, hourly={COMM_HOURLY*100:.2f}% per side")
    print(f"Binary positions: YES (majority vote ensemble)")
    print("=" * 70)

    daily, hourly, vpred = load_data()

    # Grid sizes
    for sid in STRATEGY_IDS:
        cat = CATEGORY[sid]
        sg = len(expand_grid(SIGNAL_GRIDS[sid]))
        if cat == "Trend":
            rg = len(TREND_RM_GRID_V4)
            for tf in TIMEFRAMES:
                print(f"  {STRATEGY_NAMES[sid]} {tf}: {sg} signal x {rg} RM = {sg*rg} combos (coarse ~{sg*rg//2})")
        else:
            for tf in TIMEFRAMES:
                rg = len(expand_grid(RM_GRIDS_V4.get((cat, tf), RM_GRIDS_V3.get((cat, tf), {}))))
                print(f"  {STRATEGY_NAMES[sid]} {tf}: {sg} signal x {rg} RM = {sg*rg} combos (coarse ~{sg*rg//2})")

    exec_daily = len(expand_grid(EXEC_GRID_DAILY))
    exec_hourly = len(expand_grid(EXEC_GRID_HOURLY))
    print(f"  Execution grid: daily={exec_daily}, hourly={exec_hourly} combos")

    for cat in sorted(set(CATEGORY.values())):
        bg = len(expand_grid(B_GRIDS_V3[cat]))
        print(f"  B grid ({cat}): {bg} combos")

    n_d_gate = len(D_GATE_THRESHOLDS)
    n_d_std = len(D_TARGET_VOLS) * len(D_MAX_LEVS) * len(D_GAMMA) * len(D_VOL_FLOORS) * len(D_VOL_CAPS) * len(D_SMOOTH) * len(D_HORIZONS) * n_d_gate
    n_d_inv = len(D_INV_LOOKBACKS) * len(D_MAX_LEVS) * len(D_SMOOTH) * len(D_HORIZONS) * n_d_gate
    print(f"  D grid (binary gate): {n_d_std} standard + {n_d_inv} inverse = {n_d_std+n_d_inv} combos")

    print(f"\nWarming up numba...")
    warmup_numba_v4()

    print(f"\nProcessing {len(TICKERS)} tickers with {N_WORKERS} workers...")

    all_results = {}
    all_trade_rows = []
    all_position_rows = []

    if N_WORKERS > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(N_WORKERS, initializer=_pool_init_v4, initargs=(daily, hourly, vpred)) as pool:
            for i, (ticker, result_dict) in enumerate(pool.imap_unordered(_worker_func_v4, TICKERS)):
                all_results[ticker] = result_dict["results"]
                all_trade_rows.extend(result_dict["trades"])
                all_position_rows.extend(result_dict["positions"])
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(TICKERS):
            result_dict = process_ticker_v4(ticker, daily, hourly, vpred)
            all_results[ticker] = result_dict["results"]
            all_trade_rows.extend(result_dict["trades"])
            all_position_rows.extend(result_dict["positions"])
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")

    print("\nGenerating outputs...")
    generate_outputs_v4(all_results, all_trade_rows, all_position_rows)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"V4 DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
