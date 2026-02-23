#!/usr/bin/env python3
"""
s5_rerun.py — Standalone S5_PivotPoints Rerun with Improved Pivot Logic

Fixes:
  1. Daily TF now uses weekly/monthly pivots (not daily pivots on daily bars)
  2. Relaxed filter sets (2-filter and 3-filter options alongside original 4-filter)
  3. Pivot-based SL/TP (structural levels) alongside ATR-based
  4. Extended pivots with S2/R2 levels

Imports shared functions from strategies_walkforward.py.
Runs S5 only for all 17 tickers x 2 TFs x 4 approaches (A/B/C/D).
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

# Import shared functions from main pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    load_data, compute_base, calc_sma, calc_atr, calc_adx,
    calc_sharpe_comm, calc_metrics_comm, _quick_exposure_trades,
    _year_bounds, _store_metrics, _extract_trades_from_pos,
    _build_c_mask_v3, _compute_d_scale_v3, _apply_hysteresis,
    bt_range_v3, bt_range_vpred_v3,
    expand_grid, _compute_nhours_per_day,
    TICKERS, WARMUP, COMMISSION, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, BCD_TEST_YEARS,
    B_GRIDS_V3, C_RANGE_BANDS, C_LOOKBACKS, C_HORIZONS, C_DIRECTION,
    C_TERM_FILTER, C_HYSTERESIS, C_MIN_EXPOSURE, C_MIN_TRADES_YR,
    C_FALLBACK_EXPOSURE, C_FALLBACK_TRADES_YR,
    D_TARGET_VOLS, D_MAX_LEVS, D_GAMMA, D_VOL_FLOORS, D_VOL_CAPS,
    D_SMOOTH, D_HORIZONS, D_INV_LOOKBACKS,
    approach_c_one, approach_d_one,
)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results" / "final" / "strategies" / "data"
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v3"
OUT_TABLES = OUT_DIR / "tables"
OUT_DATA = OUT_DIR / "data"

N_WORKERS = min(cpu_count(), 8)

# ═══════════════════════════════════════════════════════════════════
# Section 1: Extended Pivot Computation
# ═══════════════════════════════════════════════════════════════════

def calc_pivot_ext(H_prev, L_prev, C_prev, variant):
    """Compute extended pivot levels (P, S1, S2, R1, R2) from previous bar data."""
    P = (H_prev + L_prev + C_prev) / 3.0
    rng = H_prev - L_prev
    if variant == "classic":
        S1 = 2 * P - H_prev
        R1 = 2 * P - L_prev
        S2 = P - rng
        R2 = P + rng
    elif variant == "fibonacci":
        S1 = P - 0.382 * rng
        R1 = P + 0.382 * rng
        S2 = P - 0.618 * rng
        R2 = P + 0.618 * rng
    else:
        raise ValueError(f"Unknown pivot variant: {variant}")
    return P, S1, S2, R1, R2


def calc_pivot_weekly(daily_df, ticker, variant):
    """Compute weekly pivots aligned to daily bars.
    Previous week's H/L/C -> current week's pivot levels.
    """
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    n = len(tdf)
    dates = pd.to_datetime(tdf["date"].values)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    close = tdf["close"].values.astype(np.float64)

    # Group by ISO year-week
    iso_weeks = np.array([d.isocalendar()[:2] for d in dates])
    # Compute weekly H/L/C
    week_labels = [f"{yw[0]}-{yw[1]:02d}" for yw in iso_weeks]
    wdf = pd.DataFrame({"week": week_labels, "high": high, "low": low, "close": close})
    weekly_agg = wdf.groupby("week", sort=False).agg(
        H=("high", "max"), L=("low", "min"), C=("close", "last")
    )

    # Map previous week's pivots to current week's bars
    P_out = np.full(n, np.nan)
    S1_out = np.full(n, np.nan)
    S2_out = np.full(n, np.nan)
    R1_out = np.full(n, np.nan)
    R2_out = np.full(n, np.nan)

    unique_weeks = list(dict.fromkeys(week_labels))  # preserves order
    for i, wk in enumerate(unique_weeks):
        if i == 0:
            continue
        prev_wk = unique_weeks[i - 1]
        if prev_wk not in weekly_agg.index:
            continue
        pw = weekly_agg.loc[prev_wk]
        P, S1, S2, R1, R2 = calc_pivot_ext(pw["H"], pw["L"], pw["C"], variant)
        mask = np.array([wl == wk for wl in week_labels])
        P_out[mask] = P
        S1_out[mask] = S1
        S2_out[mask] = S2
        R1_out[mask] = R1
        R2_out[mask] = R2

    return P_out, S1_out, S2_out, R1_out, R2_out, dates


def calc_pivot_monthly(daily_df, ticker, variant):
    """Compute monthly pivots aligned to daily bars.
    Previous month's H/L/C -> current month's pivot levels.
    """
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    n = len(tdf)
    dates = pd.to_datetime(tdf["date"].values)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    close = tdf["close"].values.astype(np.float64)

    month_labels = [f"{d.year}-{d.month:02d}" for d in dates]
    mdf = pd.DataFrame({"month": month_labels, "high": high, "low": low, "close": close})
    monthly_agg = mdf.groupby("month", sort=False).agg(
        H=("high", "max"), L=("low", "min"), C=("close", "last")
    )

    P_out = np.full(n, np.nan)
    S1_out = np.full(n, np.nan)
    S2_out = np.full(n, np.nan)
    R1_out = np.full(n, np.nan)
    R2_out = np.full(n, np.nan)

    unique_months = list(dict.fromkeys(month_labels))
    for i, mo in enumerate(unique_months):
        if i == 0:
            continue
        prev_mo = unique_months[i - 1]
        if prev_mo not in monthly_agg.index:
            continue
        pm = monthly_agg.loc[prev_mo]
        P, S1, S2, R1, R2 = calc_pivot_ext(pm["H"], pm["L"], pm["C"], variant)
        mask = np.array([ml == mo for ml in month_labels])
        P_out[mask] = P
        S1_out[mask] = S1
        S2_out[mask] = S2
        R1_out[mask] = R1
        R2_out[mask] = R2

    return P_out, S1_out, S2_out, R1_out, R2_out, dates


def calc_pivot_daily_ext(high, low, close, variant):
    """Previous-day pivots with S2/R2 (extended version of calc_pivot_daily)."""
    n = len(close)
    P = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    S2 = np.full(n, np.nan)
    R1 = np.full(n, np.nan)
    R2 = np.full(n, np.nan)
    if n < 2:
        return P, S1, S2, R1, R2
    pp, s1, s2, r1, r2 = calc_pivot_ext(high[:-1], low[:-1], close[:-1], variant)
    P[1:] = pp
    S1[1:] = s1
    S2[1:] = s2
    R1[1:] = r1
    R2[1:] = r2
    return P, S1, S2, R1, R2


def map_daily_pivots_to_hourly(daily_df, ticker, h_dts, variant):
    """Map daily pivots (with S2/R2) to hourly bars via date alignment."""
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date")
    high_d = tdf["high"].values.astype(np.float64)
    low_d = tdf["low"].values.astype(np.float64)
    close_d = tdf["close"].values.astype(np.float64)
    dates_d = np.array(tdf["date"].values, dtype="datetime64[D]")
    P_d, S1_d, S2_d, R1_d, R2_d = calc_pivot_daily_ext(high_d, low_d, close_d, variant)

    h_dates = np.array(h_dts, dtype="datetime64[D]")
    indices = np.searchsorted(dates_d, h_dates, side="right") - 1
    valid = indices >= 0
    nh = len(h_dts)
    P_h = np.full(nh, np.nan)
    S1_h = np.full(nh, np.nan)
    S2_h = np.full(nh, np.nan)
    R1_h = np.full(nh, np.nan)
    R2_h = np.full(nh, np.nan)
    P_h[valid] = P_d[indices[valid]]
    S1_h[valid] = S1_d[indices[valid]]
    S2_h[valid] = S2_d[indices[valid]]
    R1_h[valid] = R1_d[indices[valid]]
    R2_h[valid] = R2_d[indices[valid]]
    return P_h, S1_h, S2_h, R1_h, R2_h


def map_weekly_pivots_to_hourly(daily_df, ticker, h_dts, variant):
    """Compute weekly pivots on daily data, then map to hourly bars."""
    P_d, S1_d, S2_d, R1_d, R2_d, dates_d = calc_pivot_weekly(daily_df, ticker, variant)
    dates_d_np = np.array(dates_d.values, dtype="datetime64[D]")
    h_dates = np.array(h_dts, dtype="datetime64[D]")
    indices = np.searchsorted(dates_d_np, h_dates, side="right") - 1
    valid = indices >= 0
    nh = len(h_dts)
    P_h = np.full(nh, np.nan)
    S1_h = np.full(nh, np.nan)
    S2_h = np.full(nh, np.nan)
    R1_h = np.full(nh, np.nan)
    R2_h = np.full(nh, np.nan)
    P_h[valid] = P_d[indices[valid]]
    S1_h[valid] = S1_d[indices[valid]]
    S2_h[valid] = S2_d[indices[valid]]
    R1_h[valid] = R1_d[indices[valid]]
    R2_h[valid] = R2_d[indices[valid]]
    return P_h, S1_h, S2_h, R1_h, R2_h


# ═══════════════════════════════════════════════════════════════════
# Section 2: New Signal Generation
# ═══════════════════════════════════════════════════════════════════

def gen_s5_signals_v2(close, ind, P, S1, R1, filter_set):
    """Generate S5 signals with configurable filter strictness.

    filter_set=2: ADX<30 AND |sma20_slope10|<0.02 (2 filters)
    filter_set=3: above + vol_regime<0.95 (3 filters)
    filter_set=4: ADX<25 AND BB<P30 AND |slope|<0.01 AND vol_regime<0.9 (original)
    """
    n = len(close)
    entry_l = ~np.isnan(S1) & (close < S1)
    entry_s = ~np.isnan(R1) & (close > R1)

    adx = ind["adx14"]
    slope = ind["sma20_slope10"]
    vr = ind["vol_regime"]
    bw = ind["bb_width"]
    bw_t = ind["bw_p30"]

    if filter_set == 2:
        f_adx = ~np.isnan(adx) & (adx < 30)
        f_slope = ~np.isnan(slope) & (np.abs(slope) < 0.02)
        entry_l = entry_l & f_adx & f_slope
        entry_s = entry_s & f_adx & f_slope
    elif filter_set == 3:
        f_adx = ~np.isnan(adx) & (adx < 30)
        f_slope = ~np.isnan(slope) & (np.abs(slope) < 0.02)
        f_vr = ~np.isnan(vr) & (vr < 0.95)
        entry_l = entry_l & f_adx & f_slope & f_vr
        entry_s = entry_s & f_adx & f_slope & f_vr
    elif filter_set == 4:
        f_adx = ~np.isnan(adx) & (adx < 25)
        f_bw = ~np.isnan(bw_t) & (bw < bw_t)
        f_slope = ~np.isnan(slope) & (np.abs(slope) < 0.01)
        f_vr = ~np.isnan(vr) & (vr < 0.9)
        entry_l = entry_l & f_adx & f_bw & f_slope & f_vr
        entry_s = entry_s & f_adx & f_bw & f_slope & f_vr

    # Volume filter
    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    # Exit: price crosses pivot P
    exit_arr = np.zeros(n, dtype=bool)
    prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
    prev_P = np.empty_like(P); prev_P[0] = P[0]; prev_P[1:] = P[:-1]
    valid_P = ~np.isnan(P) & ~np.isnan(prev_P)
    exit_arr[1:] = valid_P[1:] & (
        ((prev_c[1:] < prev_P[1:]) & (close[1:] >= P[1:])) |
        ((prev_c[1:] > prev_P[1:]) & (close[1:] <= P[1:])))

    return sig, exit_arr


# ═══════════════════════════════════════════════════════════════════
# Section 3: Pivot-Based SL/TP Precomputation + Backtest Engine
# ═══════════════════════════════════════════════════════════════════

def precompute_pivot_base_distances(P, S1, S2, R1, R2):
    """Precompute raw base distances for all bars (vectorized, called once per pivot set).

    Returns dict with 6 arrays:
      long_sl_base:  |S1-S2| (SL distance for longs)
      short_sl_base: |R2-R1| (SL distance for shorts)
      tp_P_long:     |P-S1| (TP to pivot for longs)
      tp_P_short:    |R1-P| (TP to pivot for shorts)
      tp_nextR:      |R1-S1| (TP to next level, same for both directions)
    """
    # Fallback S2/R2 where NaN
    s2 = np.where(np.isnan(S2), S1 - np.abs(P - S1), S2)
    r2 = np.where(np.isnan(R2), R1 + np.abs(R1 - P), R2)

    return {
        "long_sl_base":  np.abs(S1 - s2),
        "short_sl_base": np.abs(r2 - R1),
        "tp_P_long":     np.abs(P - S1),
        "tp_P_short":    np.abs(R1 - P),
        "tp_nextR":      np.abs(R1 - S1),
    }


def precompute_pivot_sl_tp(close, sig, base_dist, sl_target, tp_target, buffer):
    """Compute per-bar SL/TP distances using precomputed base distances (vectorized).

    sl_target: "next_level" or "next_level_half"
    tp_target: "P", "next_R", or "half_to_next"
    buffer: extra fraction on SL (0, 0.1, 0.2)

    Returns: (sl_dist, tp_dist) arrays.
    """
    n = len(close)
    min_dist = 0.001 * close

    is_long = sig == 1
    is_short = sig == -1

    # SL distance
    sl_mult = 0.5 if sl_target == "next_level_half" else 1.0
    long_sl = base_dist["long_sl_base"] * sl_mult * (1.0 + buffer)
    short_sl = base_dist["short_sl_base"] * sl_mult * (1.0 + buffer)
    sl_dist = np.where(is_long, long_sl, np.where(is_short, short_sl, min_dist))
    sl_dist = np.maximum(sl_dist, min_dist)

    # TP distance
    if tp_target == "P":
        long_tp = base_dist["tp_P_long"]
        short_tp = base_dist["tp_P_short"]
    elif tp_target == "next_R":
        long_tp = base_dist["tp_nextR"]
        short_tp = base_dist["tp_nextR"]
    else:  # half_to_next
        long_tp = 0.5 * base_dist["tp_nextR"]
        short_tp = 0.5 * base_dist["tp_nextR"]
    tp_dist = np.where(is_long, long_tp, np.where(is_short, short_tp, min_dist))
    tp_dist = np.maximum(tp_dist, min_dist)

    return sl_dist, tp_dist


@njit(cache=True)
def bt_range_pivot_v3(sig_arr, exit_arr, close, high, low, adx14,
                      sl_dist_arr, tp_dist_arr,
                      adx_exit_thresh, max_hold,
                      be_frac, cooldown_bars,
                      warmup, end_idx, log_trades):
    """Range backtest with pivot-based SL/TP distances.

    sl_dist_arr[t], tp_dist_arr[t]: absolute distances from entry price.
    be_frac: breakeven triggers when profit > frac * tp_dist (-1 = disabled).
    """
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0
    ep = 0.0; be_active = False
    orig_tp_dist = 0.0
    cooldown_rem = 0; was_sl_exit = False

    max_t = n // 2 + 1 if log_trades else 1
    trades = np.empty((max_t, 7))
    nt = 0
    entry_bar_t = 0
    direction_t = 0

    for t in range(warmup, n):
        if cooldown_rem > 0:
            cooldown_rem -= 1

        if cp != 0.0:
            held += 1
            closed = False
            was_sl_exit = False
            exit_reason = 0

            # Breakeven
            if be_frac > 0.0 and not be_active:
                if cp == 1.0 and (close[t] - ep) > be_frac * orig_tp_dist:
                    be_active = True; sl = ep
                elif cp == -1.0 and (ep - close[t]) > be_frac * orig_tp_dist:
                    be_active = True; sl = ep

            # SL / TP checks
            if cp == 1.0:
                if low[t] <= sl:
                    exit_reason = 4 if be_active else 0
                    closed = True; was_sl_exit = True
                elif high[t] >= tp:
                    exit_reason = 1
                    closed = True
            else:
                if high[t] >= sl:
                    exit_reason = 4 if be_active else 0
                    closed = True; was_sl_exit = True
                elif low[t] <= tp:
                    exit_reason = 1
                    closed = True

            # ADX exit
            if not closed:
                av = adx14[t]
                if av == av and av > adx_exit_thresh:
                    exit_reason = 6
                    closed = True
            # Signal exit
            if not closed and exit_arr[t]:
                exit_reason = 3
                closed = True
            # Max hold
            if not closed and held >= max_hold:
                exit_reason = 2
                closed = True
            if closed:
                if log_trades and nt < max_t:
                    trades[nt, 0] = entry_bar_t
                    trades[nt, 1] = t
                    trades[nt, 2] = direction_t
                    trades[nt, 3] = ep
                    trades[nt, 4] = close[t]
                    trades[nt, 5] = exit_reason
                    trades[nt, 6] = held
                    nt += 1
                if was_sl_exit and cooldown_bars > 0:
                    cooldown_rem = cooldown_bars
                cp = 0.0; held = 0; be_active = False

        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                sd = sl_dist_arr[t]
                td = tp_dist_arr[t]
                if sd != sd: sd = 0.001 * close[t]
                if td != td: td = 0.001 * close[t]
                cp = 1.0; ep = close[t]; be_active = False
                sl = close[t] - sd
                tp = close[t] + td
                orig_tp_dist = td
                held = 0
                entry_bar_t = t
                direction_t = 1
            elif s == -1:
                sd = sl_dist_arr[t]
                td = tp_dist_arr[t]
                if sd != sd: sd = 0.001 * close[t]
                if td != td: td = 0.001 * close[t]
                cp = -1.0; ep = close[t]; be_active = False
                sl = close[t] + sd
                tp = close[t] - td
                orig_tp_dist = td
                held = 0
                entry_bar_t = t
                direction_t = -1
        pos[t] = cp
    return pos, trades[:nt]


# ═══════════════════════════════════════════════════════════════════
# Section 4: S5 Grid Definitions
# ═══════════════════════════════════════════════════════════════════

S5_SIG_DAILY = dict(
    pivot_period=["weekly", "monthly"],
    pivot_type=["classic", "fibonacci"],
    filter_set=[2, 3, 4],
)  # 2x2x3 = 12 signal combos

S5_SIG_HOURLY = dict(
    pivot_period=["daily"],
    pivot_type=["classic", "fibonacci"],
    filter_set=[2, 3, 4],
)  # 1x2x3 = 6 signal combos

S5_RM_PIVOT_DAILY = dict(
    sl_target=["next_level", "next_level_half"],
    tp_target=["P", "next_R", "half_to_next"],
    buffer=[0, 0.1, 0.2],
    max_hold=[5, 8, 10],
    breakeven=[False, True],
    cooldown_bars=[0, 5],
)  # 2x3x3x3x2x2 = 216

S5_RM_PIVOT_HOURLY = dict(
    sl_target=["next_level", "next_level_half"],
    tp_target=["P", "next_R", "half_to_next"],
    buffer=[0, 0.1, 0.2],
    max_hold=[10, 16, 20],
    breakeven=[False, True],
    cooldown_bars=[0, 5],
)  # 216

S5_RM_ATR_DAILY = dict(
    sl_mult=[0.3, 0.5, 0.75],
    tp_mult=[0.3, 0.5, 0.75, 1.0],
    max_hold=[5, 8, 10],
    breakeven_trigger=[None, 1.0],
    cooldown_bars=[0, 5],
    time_decay=[False, True],
)  # 3x4x3x2x2x2 = 288

S5_RM_ATR_HOURLY = dict(
    sl_mult=[0.3, 0.5, 0.75],
    tp_mult=[0.3, 0.5, 0.75, 1.0],
    max_hold=[10, 16, 20],
    breakeven_trigger=[None, 1.0],
    cooldown_bars=[0, 5],
    time_decay=[False, True],
)  # 288


# ═══════════════════════════════════════════════════════════════════
# Section 5: Approach A
# ═══════════════════════════════════════════════════════════════════

def approach_a_s5(tf, close, high, low, open_arr, volume, ind,
                  daily_df, ticker, dates, is_hourly, log_ret,
                  pivot_cache):
    """Approach A for S5 with new pivot-based signal and RM grids."""
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)

    sig_grid = expand_grid(S5_SIG_HOURLY if is_hourly else S5_SIG_DAILY)
    rm_pivot_grid = expand_grid(S5_RM_PIVOT_HOURLY if is_hourly else S5_RM_PIVOT_DAILY)
    rm_atr_grid = expand_grid(S5_RM_ATR_HOURLY if is_hourly else S5_RM_ATR_DAILY)

    results_by_year = {}
    params_by_year = {}
    trades_by_year = {}

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

        # Pre-cache signals and base distances per signal_params
        cached_sigs = {}
        base_dist_cache = {}  # pivot-level base distances, keyed by (period, type)
        for sp in sig_grid:
            sp_key = (sp["pivot_period"], sp["pivot_type"], sp["filter_set"])
            piv_key = (sp["pivot_period"], sp["pivot_type"])
            P, S1, S2, R1, R2 = pivot_cache[piv_key]
            sig, exit_arr = gen_s5_signals_v2(close, ind, P, S1, R1, sp["filter_set"])
            if piv_key not in base_dist_cache:
                base_dist_cache[piv_key] = precompute_pivot_base_distances(P, S1, S2, R1, R2)
            cached_sigs[sp_key] = (sig, exit_arr, piv_key)

        all_results = []

        for sp in sig_grid:
            sp_key = (sp["pivot_period"], sp["pivot_type"], sp["filter_set"])
            sig, exit_arr, piv_key = cached_sigs[sp_key]
            base_dist = base_dist_cache[piv_key]

            # Pivot-based RM
            for rm in rm_pivot_grid:
                sl_dist, tp_dist = precompute_pivot_sl_tp(
                    close, sig, base_dist,
                    rm["sl_target"], rm["tp_target"], rm["buffer"])
                be = 0.5 if rm["breakeven"] else -1.0
                pos, _ = bt_range_pivot_v3(
                    sig, exit_arr, close, high, low, ind["adx14"],
                    sl_dist, tp_dist, 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"],
                    WARMUP, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                dpos = np.diff(pos[train_mask], prepend=0.0)
                net_r = dr - np.abs(dpos) * COMMISSION
                cum = np.cumsum(net_r)
                rmax = np.maximum.accumulate(cum)
                max_dd = (cum - rmax).min() if len(cum) > 0 else 0.0
                active = np.sum(pos[train_mask] != 0)
                exposure = active / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                rm_full = dict(rm)
                rm_full["_rm_type"] = "pivot"
                all_results.append((sp, rm_full, sh, max_dd, exposure))

            # ATR-based RM
            for rm in rm_atr_grid:
                be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
                cd = rm.get("cooldown_bars", 0)
                td = 1 if rm.get("time_decay") else 0
                pos, _ = bt_range_v3(
                    sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
                    rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
                    be, cd, td, WARMUP, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                dpos = np.diff(pos[train_mask], prepend=0.0)
                net_r = dr - np.abs(dpos) * COMMISSION
                cum = np.cumsum(net_r)
                rmax = np.maximum.accumulate(cum)
                max_dd = (cum - rmax).min() if len(cum) > 0 else 0.0
                active = np.sum(pos[train_mask] != 0)
                exposure = active / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                rm_full = dict(rm)
                rm_full["_rm_type"] = "atr"
                all_results.append((sp, rm_full, sh, max_dd, exposure))

        # Filter and select top-10
        passing = [(sp, rm, sh, mdd, exp) for sp, rm, sh, mdd, exp in all_results
                   if sh > 0 and mdd > -0.30 and exp > 5]
        if len(passing) == 0:
            passing = sorted(all_results, key=lambda x: x[2], reverse=True)[:3]
        passing = sorted(passing, key=lambda x: x[2], reverse=True)[:10]

        # Run top-10 through test period
        test_positions = []
        test_trades = []
        for sp, rm, sh, mdd, exp in passing:
            sp_key = (sp["pivot_period"], sp["pivot_type"], sp["filter_set"])
            sig, exit_arr, piv_key = cached_sigs[sp_key]

            if rm["_rm_type"] == "pivot":
                base_dist = base_dist_cache[piv_key]
                sl_dist, tp_dist = precompute_pivot_sl_tp(
                    close, sig, base_dist,
                    rm["sl_target"], rm["tp_target"], rm["buffer"])
                be = 0.5 if rm["breakeven"] else -1.0
                pos, tr = bt_range_pivot_v3(
                    sig, exit_arr, close, high, low, ind["adx14"],
                    sl_dist, tp_dist, 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"],
                    WARMUP, test_end, 1)
            else:
                be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
                cd = rm.get("cooldown_bars", 0)
                td = 1 if rm.get("time_decay") else 0
                pos, tr = bt_range_v3(
                    sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
                    rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
                    be, cd, td, WARMUP, test_end, 1)
            test_positions.append(pos)
            test_trades.append(tr)

        ensemble = np.mean(test_positions, axis=0)
        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm) for sp, rm, sh, mdd, exp in passing]
        trades_by_year[test_year] = test_trades

    return results_by_year, params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# Section 6: Approach B
# ═══════════════════════════════════════════════════════════════════

def approach_b_s5(tf, close, high, low, volume, ind,
                  daily_df, ticker, dates, sigma_dict, is_hourly,
                  log_ret, a_params_by_year, pivot_cache):
    """Approach B for S5: sigma_pred-based adaptive stops on A's top-10 signals."""
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
    n = len(close)
    b_grid = expand_grid(B_GRIDS_V3["Range"])

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

        # Regenerate A's signals
        a_sigs = []
        for sp, rm in a_params:
            piv_key = (sp["pivot_period"], sp["pivot_type"])
            P, S1, S2, R1, R2 = pivot_cache[piv_key]
            sig, exit_arr = gen_s5_signals_v2(close, ind, P, S1, R1, sp["filter_set"])
            a_sigs.append((sig, exit_arr))

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
            base_hold = 20 if is_hourly else 10

            val_positions = []
            for sig, exit_arr in a_sigs:
                pos, _ = bt_range_vpred_v3(
                    sig, exit_arr, close, high, low, sigma, ind["adx14"],
                    bp["k_sl"], k_tp, 30.0, base_hold,
                    k_be, gh, sigma_median,
                    cd, WARMUP, val_end, 0)
                val_positions.append(pos)
            ensemble = np.mean(val_positions, axis=0)
            sh = calc_sharpe_comm(ensemble, log_ret, val_mask, ann)
            if sh > best_sh:
                best_sh = sh
                best_bp = dict(bp)

        # Run best B through test period
        best_horizon = best_bp.get("horizon", "h1")
        best_sigma = sigma_dict.get(best_horizon, sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        best_sigma_median = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        k_be = -1.0 if best_bp.get("k_be") is None else best_bp["k_be"]
        gh = -1.0 if best_bp.get("gamma_hold") is None else best_bp["gamma_hold"]
        cd = best_bp.get("cooldown_bars", 5)
        k_tp = best_bp.get("k_tp", best_bp.get("k_sl", 1.0) * best_bp.get("ratio", 1.5))
        base_hold = 20 if is_hourly else 10

        test_positions = []
        test_trades = []
        for sig, exit_arr in a_sigs:
            pos, tr = bt_range_vpred_v3(
                sig, exit_arr, close, high, low, best_sigma, ind["adx14"],
                best_bp["k_sl"], k_tp, 30.0, base_hold,
                k_be, gh, best_sigma_median,
                cd, WARMUP, test_end, 1)
            test_positions.append(pos)
            test_trades.append(tr)
        ensemble = np.mean(test_positions, axis=0)
        results_by_year[test_year] = ensemble
        best_params_by_year[test_year] = best_bp
        trades_by_year[test_year] = test_trades

    return results_by_year, best_params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# Section 7: Process Ticker
# ═══════════════════════════════════════════════════════════════════

def process_s5_ticker(ticker, daily_df, hourly_df, vpred_df):
    """Process one ticker: run A/B/C/D for S5 only, both TFs."""
    results = {}
    sid = "S5"
    sname = STRATEGY_NAMES[sid]

    for tf in ["daily", "hourly"]:
        is_hourly = tf == "hourly"
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

        # Pre-compute ALL pivot variants
        pivot_cache = {}
        if is_hourly:
            h_dts = tdf["datetime"].values
            for ptype in ["classic", "fibonacci"]:
                # Hourly uses daily pivots
                P, S1, S2, R1, R2 = map_daily_pivots_to_hourly(
                    daily_df, ticker, h_dts, ptype)
                pivot_cache[("daily", ptype)] = (P, S1, S2, R1, R2)
        else:
            for ptype in ["classic", "fibonacci"]:
                # Daily uses weekly pivots
                P_w, S1_w, S2_w, R1_w, R2_w, _ = calc_pivot_weekly(daily_df, ticker, ptype)
                pivot_cache[("weekly", ptype)] = (P_w, S1_w, S2_w, R1_w, R2_w)
                # Daily uses monthly pivots
                P_m, S1_m, S2_m, R1_m, R2_m, _ = calc_pivot_monthly(daily_df, ticker, ptype)
                pivot_cache[("monthly", ptype)] = (P_m, S1_m, S2_m, R1_m, R2_m)

        # Build sigma_dict
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
                    nhours_safe = np.maximum(nhours, 1.0)
                    sigma_arr = sigma_arr / np.sqrt(nhours_safe)
                sigma_dict[hname] = sigma_arr

        ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
        bpy = 252 * 9 if is_hourly else 252

        # Approach A
        a_results, a_params, a_trades = approach_a_s5(
            tf, close, high, low, open_arr, volume, ind,
            daily_df, ticker, dates, is_hourly, log_ret,
            pivot_cache)

        for year, pos_arr in a_results.items():
            _store_metrics(results, sid, tf, "A", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        # Approach B
        b_results, b_params, b_trades = approach_b_s5(
            tf, close, high, low, volume, ind,
            daily_df, ticker, dates, sigma_dict, is_hourly,
            log_ret, a_params, pivot_cache)

        for year, pos_arr in b_results.items():
            _store_metrics(results, sid, tf, "B", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        # Approach C (imported from main)
        c_results, c_params, c_trades = approach_c_one(
            sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

        for year, pos_arr in c_results.items():
            _store_metrics(results, sid, tf, "C", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        # Approach D (imported from main)
        d_results, d_params, d_trades = approach_d_one(
            sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

        for year, pos_arr in d_results.items():
            _store_metrics(results, sid, tf, "D", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

    return {"results": results}


# ═══════════════════════════════════════════════════════════════════
# Section 8: Multiprocessing & Main
# ═══════════════════════════════════════════════════════════════════

_G_DAILY = None
_G_HOURLY = None
_G_VPRED = None


def _pool_init(daily, hourly, vpred):
    global _G_DAILY, _G_HOURLY, _G_VPRED
    _G_DAILY = daily
    _G_HOURLY = hourly
    _G_VPRED = vpred


def _worker_func(ticker):
    return ticker, process_s5_ticker(ticker, _G_DAILY, _G_HOURLY, _G_VPRED)


def warmup_s5_numba():
    """Warmup numba for S5-specific and shared BT functions."""
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
    sl_d = np.ones(n) * 2.0
    tp_d = np.ones(n) * 3.0

    # Warmup shared bt_range_v3
    bt_range_v3(sig, ex, c, h, l, a, adx, 1.0, 1.5, 30.0, 10, 1.0, 5, 1, 5, n, 0)
    # Warmup shared bt_range_vpred_v3
    bt_range_vpred_v3(sig, ex, c, h, l, sigma, adx, 1.0, 2.0, 30.0, 10, 1.0, 0.5, 0.02, 5, 5, n, 0)
    # Warmup new bt_range_pivot_v3
    bt_range_pivot_v3(sig, ex, c, h, l, adx, sl_d, tp_d, 30.0, 10, 0.5, 5, 5, n, 0)
    # Warmup _apply_hysteresis
    pctrank = np.random.rand(n)
    _apply_hysteresis(pctrank, 1, 0.5, 0.4, n)
    print("  Numba warmup done")


def main():
    t0 = time.time()
    print("=" * 70)
    print("S5_PivotPoints RERUN — Improved Pivot Logic")
    print(f"Tickers: {len(TICKERS)}, TFs: 2, Approaches: A/B/C/D")
    print(f"A test years: {A_TEST_YEARS[0]}-{A_TEST_YEARS[-1]}")
    print(f"B/C/D test years: {BCD_TEST_YEARS[0]}-{BCD_TEST_YEARS[-1]}")
    print("=" * 70)

    # Grid sizes
    for tf_label, sig_grid, rm_p_grid, rm_a_grid in [
        ("daily", S5_SIG_DAILY, S5_RM_PIVOT_DAILY, S5_RM_ATR_DAILY),
        ("hourly", S5_SIG_HOURLY, S5_RM_PIVOT_HOURLY, S5_RM_ATR_HOURLY),
    ]:
        n_sig = len(expand_grid(sig_grid))
        n_rmp = len(expand_grid(rm_p_grid))
        n_rma = len(expand_grid(rm_a_grid))
        total = n_sig * (n_rmp + n_rma)
        print(f"  {tf_label}: {n_sig} signal x ({n_rmp} pivot_RM + {n_rma} atr_RM) = {total} combos")

    n_b = len(expand_grid(B_GRIDS_V3["Range"]))
    print(f"  B grid (Range): {n_b} combos")

    daily, hourly, vpred = load_data()

    print(f"\nWarming up numba...")
    warmup_s5_numba()

    print(f"\nProcessing {len(TICKERS)} tickers with {N_WORKERS} workers...")

    all_results = {}

    if N_WORKERS > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(N_WORKERS, initializer=_pool_init, initargs=(daily, hourly, vpred)) as pool:
            for i, (ticker, result_dict) in enumerate(pool.imap_unordered(_worker_func, TICKERS)):
                all_results[ticker] = result_dict["results"]
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(TICKERS):
            result_dict = process_s5_ticker(ticker, daily, hourly, vpred)
            all_results[ticker] = result_dict["results"]
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")

    # ── Collect new S5 rows ──
    new_rows = []
    for ticker, ticker_results in all_results.items():
        for key, met in ticker_results.items():
            met.pop("_net_returns", None)
            new_rows.append(met)

    if not new_rows:
        print("  No results produced!")
        return

    new_df = pd.DataFrame(new_rows)

    # ── Load previous results and replace S5 ──
    prev_path = OUT_DATA / "wf_v3_all_results.csv"
    if prev_path.exists():
        prev_df = pd.read_csv(prev_path)
        old_s5 = prev_df[prev_df["strategy"] == "S5_PivotPoints"].copy()
        other = prev_df[prev_df["strategy"] != "S5_PivotPoints"]
        combined = pd.concat([other, new_df], ignore_index=True)

        # Save updated
        combined.to_csv(OUT_DATA / "wf_v3_all_results.csv", index=False)
        print(f"\n  Updated results saved: {len(combined)} rows")
        print(f"  (replaced {len(old_s5)} old S5 rows with {len(new_df)} new)")
    else:
        print(f"\n  WARNING: Previous results not found at {prev_path}")
        print(f"  Saving S5-only results")
        new_df.to_csv(OUT_DATA / "wf_v3_s5_rerun_results.csv", index=False)

    # ── Print comparison ──
    print("\n" + "=" * 70)
    print("=== S5_PivotPoints BEFORE vs AFTER ===")
    print("=" * 70)

    if prev_path.exists() and len(old_s5) > 0:
        # Use BCD years for comparison (common years)
        common_years = BCD_TEST_YEARS
        for tf in ["daily", "hourly"]:
            old_sub = old_s5[(old_s5["timeframe"] == tf) & (old_s5["year"].isin(common_years))]
            new_sub = new_df[(new_df["timeframe"] == tf) & (new_df["year"].isin(common_years))]
            if len(old_sub) == 0 and len(new_sub) == 0:
                continue

            print(f"\n  {tf}:")
            for approach in ["A", "B", "C", "D"]:
                old_a = old_sub[old_sub["approach"] == approach]
                new_a = new_sub[new_sub["approach"] == approach]
                old_sh = old_a["sharpe"].mean() if len(old_a) > 0 else float('nan')
                new_sh = new_a["sharpe"].mean() if len(new_a) > 0 else float('nan')
                delta = new_sh - old_sh if not (np.isnan(old_sh) or np.isnan(new_sh)) else float('nan')
                delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
                print(f"    {approach}: BEFORE={old_sh:.4f}  AFTER={new_sh:.4f}  delta={delta_str}")

    # ── Detailed new results ──
    for tf in ["daily", "hourly"]:
        sub = new_df[new_df["timeframe"] == tf]
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

        print(f"\n  === S5_PivotPoints {tf} (NEW) ===")
        hdr = f"  {'approach':>8s}  {'MeanSharpe':>10s}  {'AnnRet%':>8s}  {'AnnVol%':>8s}  {'MaxDD%':>7s}  {'Exp%':>6s}  {'Tr/yr':>6s}  {'Win%':>6s}"
        print(hdr)
        a_sharpe_val = None
        for _, r in summary.iterrows():
            if r["approach"] == "A":
                a_sharpe_val = r["MeanSharpe"]
            print(f"  {r['approach']:>8s}  {r['MeanSharpe']:10.4f}  {r['AnnReturn']:8.2f}  {r['AnnVol']:8.2f}  {r['MaxDD']:7.2f}  {r['Exposure']:6.2f}  {r['TradesPerYr']:6.2f}  {r['WinRate']:6.2f}")

        if a_sharpe_val is not None:
            delta_parts = []
            for appr in ["B", "C", "D"]:
                row_s = summary.loc[summary["approach"] == appr]
                if len(row_s) > 0:
                    delta = row_s["MeanSharpe"].values[0] - a_sharpe_val
                    delta_parts.append(f"{appr}={delta:+.3f}")
            if delta_parts:
                print(f"  \u0394Sharpe vs A:  {', '.join(delta_parts)}")

        # Top-5 / Worst-5 tickers
        sub_common = sub[sub["year"].isin(common_years)]
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

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
