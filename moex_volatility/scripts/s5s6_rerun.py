#!/usr/bin/env python3
"""
s5s6_rerun.py — Rerun S5 hourly + S6 daily/hourly with fixes.

S5 hourly fixes:
  - Weekly/monthly pivots (was daily → band too narrow)
  - Confirmation bars (2 consecutive bars beyond level)
  - Entry buffer (price must exceed S1/R1 by buffer% of close)

S6 fixes:
  - Relaxed filter sets (1/2/3/4 instead of only 4)
  - Expanded VWAP window (5-60 instead of 10-20)
  - Expanded dev_mult (0.5-2.5 instead of 1.0-2.0)
  - VWAP-band SL/TP (structural, not ATR-based)
  - Directional bias (daily only, SMA50 vs SMA200)
  - Session VWAP for hourly (cumulative from session open)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    load_data, compute_base, calc_sma, calc_atr,
    calc_sharpe_comm, calc_metrics_comm, _quick_exposure_trades,
    _year_bounds, _store_metrics, _extract_trades_from_pos,
    _build_c_mask_v3, _compute_d_scale_v3, _apply_hysteresis,
    bt_range_v3, bt_range_vpred_v3,
    expand_grid, _compute_nhours_per_day,
    build_daily_trend, align_daily_to_hourly,
    TICKERS, WARMUP, COMMISSION, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, BCD_TEST_YEARS,
    B_GRIDS_V3, C_RANGE_BANDS, C_LOOKBACKS, C_HORIZONS, C_DIRECTION,
    C_TERM_FILTER, C_HYSTERESIS, C_MIN_EXPOSURE, C_MIN_TRADES_YR,
    C_FALLBACK_EXPOSURE, C_FALLBACK_TRADES_YR,
    D_TARGET_VOLS, D_MAX_LEVS, D_GAMMA, D_VOL_FLOORS, D_VOL_CAPS,
    D_SMOOTH, D_HORIZONS, D_INV_LOOKBACKS,
    approach_c_one, approach_d_one,
)
from s5_rerun import (
    calc_pivot_ext, calc_pivot_weekly, calc_pivot_monthly,
    map_weekly_pivots_to_hourly,
    precompute_pivot_base_distances, precompute_pivot_sl_tp,
    bt_range_pivot_v3,
)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results" / "final" / "strategies" / "data"
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v3"
OUT_TABLES = OUT_DIR / "tables"
OUT_DATA = OUT_DIR / "data"
N_WORKERS = min(cpu_count(), 8)


# ═══════════════════════════════════════════════════════════════════
# Section 1: S5 Hourly — Weekly/Monthly Pivots + Confirmation + Buffer
# ═══════════════════════════════════════════════════════════════════

def map_monthly_pivots_to_hourly(daily_df, ticker, h_dts, variant):
    """Compute monthly pivots on daily data, then map to hourly bars."""
    P_d, S1_d, S2_d, R1_d, R2_d, dates_d = calc_pivot_monthly(daily_df, ticker, variant)
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


def gen_s5_signals_v3(close, ind, P, S1, R1, filter_set, confirmation, entry_buffer):
    """S5 signal generation with confirmation bars and entry buffer.

    confirmation: 0=OFF, 2=require 2 consecutive bars beyond level
    entry_buffer: 0, 0.001, 0.002 — price must exceed S1/R1 by buffer*close
    """
    n = len(close)
    buf = entry_buffer * close
    raw_l = ~np.isnan(S1) & (close < S1 - buf)
    raw_s = ~np.isnan(R1) & (close > R1 + buf)

    if confirmation >= 2:
        # Require 2 consecutive bars beyond level
        consec_l = np.zeros(n, dtype=bool)
        consec_s = np.zeros(n, dtype=bool)
        for t in range(1, n):
            consec_l[t] = raw_l[t] and raw_l[t - 1]
            consec_s[t] = raw_s[t] and raw_s[t - 1]
        entry_l = consec_l
        entry_s = consec_s
    else:
        entry_l = raw_l
        entry_s = raw_s

    # Apply filters
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
# Section 2: S6 Fixes — Filters, Session VWAP, VWAP-band SL/TP
# ═══════════════════════════════════════════════════════════════════

def compute_session_vwap(close, high, low, volume, datetimes):
    """Cumulative VWAP from session open, reset daily at first bar of each day."""
    n = len(close)
    vwap = np.full(n, np.nan)
    dev = np.full(n, 1e-12)
    tp = (high + low + close) / 3.0
    dates = pd.to_datetime(datetimes).normalize()

    cum_tv = 0.0
    cum_v = 0.0
    session_closes = []
    session_start = 0

    for t in range(n):
        if t > 0 and dates[t] != dates[t - 1]:
            # New session — reset
            cum_tv = 0.0
            cum_v = 0.0
            session_closes = []
            session_start = t
        cum_tv += tp[t] * volume[t]
        cum_v += volume[t]
        if cum_v > 1e-12:
            vwap[t] = cum_tv / cum_v
            session_closes.append(close[t])
            if len(session_closes) >= 3:
                diffs = np.array(session_closes) - vwap[t]
                dev[t] = max(np.std(diffs, ddof=1), 1e-12)
    return vwap, dev


def precompute_vwap_cache_ext(close, high, low, volume, windows):
    """Extended VWAP cache with more windows."""
    cache = {}
    for w in windows:
        tp = (high + low + close) / 3.0
        tp_vol = tp * volume
        sum_tv = pd.Series(tp_vol).rolling(w, min_periods=w).sum().values
        sum_v = pd.Series(volume).rolling(w, min_periods=w).sum().values
        sum_v_safe = np.where(np.isnan(sum_v) | (sum_v < 1e-12), 1e-12, sum_v)
        vwap = sum_tv / sum_v_safe
        d = pd.Series(close - vwap).rolling(w, min_periods=w).std(ddof=1).values
        d = np.where(np.isnan(d) | (d < 1e-12), 1e-12, d)
        cache[w] = (vwap, d)
    return cache


def gen_s6_signals_v2(close, ind, vwap, dev, dev_mult, filter_set, dir_bias_arr=None):
    """S6 signal generation with configurable filter strictness + directional bias.

    filter_set=1: ADX<35 only
    filter_set=2: ADX<30 + |slope|<0.02
    filter_set=3: above + vol_regime<0.95
    filter_set=4: ADX<25 + BB<P30 + |slope|<0.01 + vol_regime<0.9 (original)

    dir_bias_arr: if not None, float array where 1.0=bullish (longs only), 0.0=bearish (shorts only), NaN=both
    """
    n = len(close)
    entry_l = ~np.isnan(vwap) & (close < vwap - dev_mult * dev)
    entry_s = ~np.isnan(vwap) & (close > vwap + dev_mult * dev)

    adx = ind["adx14"]
    slope = ind["sma20_slope10"]
    vr = ind["vol_regime"]
    bw = ind["bb_width"]
    bw_t = ind["bw_p30"]

    if filter_set == 1:
        f_adx = ~np.isnan(adx) & (adx < 35)
        entry_l = entry_l & f_adx
        entry_s = entry_s & f_adx
    elif filter_set == 2:
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

    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    # Directional bias
    if dir_bias_arr is not None:
        bullish = ~np.isnan(dir_bias_arr) & (dir_bias_arr >= 0.5)
        bearish = ~np.isnan(dir_bias_arr) & (dir_bias_arr < 0.5)
        # In uptrend: only longs; in downtrend: only shorts
        entry_s = entry_s & bearish
        entry_l = entry_l & bullish

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    # Exit: price returns near VWAP
    exit_arr = ~np.isnan(vwap) & (np.abs(close - vwap) < 0.5 * dev)
    return sig, exit_arr


def precompute_vwap_sl_tp(close, sig, vwap, dev, sl_dev):
    """Compute VWAP-band SL/TP distances (vectorized).

    Long:  SL = entry - |entry - (VWAP - sl_dev*dev)|, TP = |VWAP - entry|
    Short: SL = |(VWAP + sl_dev*dev) - entry| - entry, TP = |entry - VWAP|

    Simplified: SL based on distance to extended VWAP band, TP = distance to VWAP center.
    """
    n = len(close)
    min_dist = 0.001 * close
    is_long = sig == 1
    is_short = sig == -1

    # SL distance: from entry to far VWAP band
    sl_band_low = vwap - sl_dev * dev
    sl_band_high = vwap + sl_dev * dev
    long_sl = np.abs(close - sl_band_low)
    short_sl = np.abs(sl_band_high - close)

    # TP distance: from entry to VWAP center
    long_tp = np.abs(vwap - close)
    short_tp = np.abs(close - vwap)

    sl_dist = np.where(is_long, long_sl, np.where(is_short, short_sl, min_dist))
    tp_dist = np.where(is_long, long_tp, np.where(is_short, short_tp, min_dist))
    sl_dist = np.maximum(sl_dist, min_dist)
    tp_dist = np.maximum(tp_dist, min_dist)
    return sl_dist, tp_dist


@njit(cache=True)
def bt_range_vwap_v3(sig_arr, exit_arr, close, high, low, adx14,
                     sl_dist_arr, tp_dist_arr,
                     adx_exit_thresh, max_hold,
                     be_frac, cooldown_bars,
                     warmup, end_idx, log_trades):
    """Range backtest with VWAP-band-based SL/TP distances.
    Same structure as bt_range_pivot_v3."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0
    ep = 0.0; be_active = False
    orig_tp_dist = 0.0
    cooldown_rem = 0; was_sl_exit = False
    max_t = n // 2 + 1 if log_trades else 1
    trades = np.empty((max_t, 7))
    nt = 0; entry_bar_t = 0; direction_t = 0

    for t in range(warmup, n):
        if cooldown_rem > 0:
            cooldown_rem -= 1
        if cp != 0.0:
            held += 1
            closed = False; was_sl_exit = False; exit_reason = 0
            if be_frac > 0.0 and not be_active:
                if cp == 1.0 and (close[t] - ep) > be_frac * orig_tp_dist:
                    be_active = True; sl = ep
                elif cp == -1.0 and (ep - close[t]) > be_frac * orig_tp_dist:
                    be_active = True; sl = ep
            if cp == 1.0:
                if low[t] <= sl:
                    exit_reason = 4 if be_active else 0; closed = True; was_sl_exit = True
                elif high[t] >= tp:
                    exit_reason = 1; closed = True
            else:
                if high[t] >= sl:
                    exit_reason = 4 if be_active else 0; closed = True; was_sl_exit = True
                elif low[t] <= tp:
                    exit_reason = 1; closed = True
            if not closed:
                av = adx14[t]
                if av == av and av > adx_exit_thresh:
                    exit_reason = 6; closed = True
            if not closed and exit_arr[t]:
                exit_reason = 3; closed = True
            if not closed and held >= max_hold:
                exit_reason = 2; closed = True
            if closed:
                if log_trades and nt < max_t:
                    trades[nt, 0] = entry_bar_t; trades[nt, 1] = t
                    trades[nt, 2] = direction_t; trades[nt, 3] = ep
                    trades[nt, 4] = close[t]; trades[nt, 5] = exit_reason
                    trades[nt, 6] = held; nt += 1
                if was_sl_exit and cooldown_bars > 0:
                    cooldown_rem = cooldown_bars
                cp = 0.0; held = 0; be_active = False
        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                sd = sl_dist_arr[t]; td = tp_dist_arr[t]
                if sd != sd: sd = 0.001 * close[t]
                if td != td: td = 0.001 * close[t]
                cp = 1.0; ep = close[t]; be_active = False
                sl = close[t] - sd; tp = close[t] + td
                orig_tp_dist = td; held = 0
                entry_bar_t = t; direction_t = 1
            elif s == -1:
                sd = sl_dist_arr[t]; td = tp_dist_arr[t]
                if sd != sd: sd = 0.001 * close[t]
                if td != td: td = 0.001 * close[t]
                cp = -1.0; ep = close[t]; be_active = False
                sl = close[t] + sd; tp = close[t] - td
                orig_tp_dist = td; held = 0
                entry_bar_t = t; direction_t = -1
        pos[t] = cp
    return pos, trades[:nt]


# ═══════════════════════════════════════════════════════════════════
# Section 3: Grid Definitions
# ═══════════════════════════════════════════════════════════════════

# --- S5 hourly (now weekly/monthly pivots) ---
S5H_SIG = dict(
    pivot_period=["weekly", "monthly"],
    pivot_type=["classic", "fibonacci"],
    filter_set=[2, 3, 4],
    confirmation=[0, 2],
    entry_buffer=[0, 0.001, 0.002],
)  # 2×2×3×2×3 = 72

S5H_RM_PIVOT = dict(
    sl_target=["next_level", "next_level_half"],
    tp_target=["P", "next_R", "half_to_next"],
    buffer=[0, 0.1, 0.2],
    max_hold=[10, 16, 20],
    breakeven=[False, True],
    cooldown_bars=[0, 5],
)  # 216

S5H_RM_ATR = dict(
    sl_mult=[0.3, 0.5, 0.75],
    tp_mult=[0.3, 0.5, 0.75, 1.0],
    max_hold=[10, 16, 20],
    breakeven_trigger=[None, 1.0],
    cooldown_bars=[0, 5],
    time_decay=[False, True],
)  # 288

# --- S6 daily ---
S6D_SIG = dict(
    vwap_window=[5, 10, 20, 60],
    dev_mult=[0.5, 1.0, 1.5, 2.0],
    filter_set=[1, 2, 3, 4],
    dir_bias=["OFF", "ON"],
)  # 4×4×4×2 = 128

# --- S6 hourly ---
S6H_SIG = dict(
    vwap_window=[5, 10, 20, 60],
    dev_mult=[0.5, 1.0, 1.5, 2.0],
    filter_set=[1, 2, 3, 4],
    vwap_type=["rolling", "session"],
)  # 4×4×4×2 = 128

# --- S6 RM (shared daily/hourly, only max_hold differs) ---
S6_RM_ATR_DAILY = dict(
    sl_mult=[0.75, 1.0, 1.5], tp_mult=[0.75, 1.0, 1.5], max_hold=[5, 8, 10],
    breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5], time_decay=[False, True],
)  # 216

S6_RM_ATR_HOURLY = dict(
    sl_mult=[0.75, 1.0, 1.5], tp_mult=[0.75, 1.0, 1.5], max_hold=[10, 16, 20],
    breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5], time_decay=[False, True],
)  # 216

S6_RM_VWAP_DAILY = dict(
    sl_dev=[1.5, 2.0, 2.5, 3.0],
    max_hold=[5, 8, 10],
    breakeven=[False, True],
    cooldown_bars=[0, 5],
)  # 4×3×2×2 = 48

S6_RM_VWAP_HOURLY = dict(
    sl_dev=[1.5, 2.0, 2.5, 3.0],
    max_hold=[10, 16, 20],
    breakeven=[False, True],
    cooldown_bars=[0, 5],
)  # 48


# ═══════════════════════════════════════════════════════════════════
# Section 4: Approach A — S5 Hourly
# ═══════════════════════════════════════════════════════════════════

def approach_a_s5h(close, high, low, open_arr, volume, ind,
                   dates, log_ret, pivot_cache):
    """Approach A for S5 hourly with weekly/monthly pivots."""
    ann = np.sqrt(252 * 9)
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)
    sig_grid = expand_grid(S5H_SIG)
    rm_pivot_grid = expand_grid(S5H_RM_PIVOT)
    rm_atr_grid = expand_grid(S5H_RM_ATR)

    results_by_year = {}
    params_by_year = {}
    trades_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        if train_end <= WARMUP or train_end >= n:
            continue
        train_mask[:] = False
        train_mask[WARMUP:train_end] = True

        # Pre-cache signals + base distances
        cached_sigs = {}
        base_dist_cache = {}
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            piv_key = (sp["pivot_period"], sp["pivot_type"])
            P, S1, S2, R1, R2 = pivot_cache[piv_key]
            sig, exit_arr = gen_s5_signals_v3(
                close, ind, P, S1, R1, sp["filter_set"],
                sp["confirmation"], sp["entry_buffer"])
            if piv_key not in base_dist_cache:
                base_dist_cache[piv_key] = precompute_pivot_base_distances(P, S1, S2, R1, R2)
            cached_sigs[sp_key] = (sig, exit_arr, piv_key)

        all_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, piv_key = cached_sigs[sp_key]
            base_dist = base_dist_cache[piv_key]

            for rm in rm_pivot_grid:
                sl_dist, tp_dist = precompute_pivot_sl_tp(
                    close, sig, base_dist,
                    rm["sl_target"], rm["tp_target"], rm["buffer"])
                be = 0.5 if rm["breakeven"] else -1.0
                pos, _ = bt_range_pivot_v3(
                    sig, exit_arr, close, high, low, ind["adx14"],
                    sl_dist, tp_dist, 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"], WARMUP, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                net_r = dr - np.abs(np.diff(pos[train_mask], prepend=0.0)) * COMMISSION
                cum = np.cumsum(net_r)
                max_dd = (cum - np.maximum.accumulate(cum)).min() if len(cum) > 0 else 0.0
                exp = np.sum(pos[train_mask] != 0) / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                rm_f = dict(rm); rm_f["_rm_type"] = "pivot"
                all_results.append((sp, rm_f, sh, max_dd, exp))

            for rm in rm_atr_grid:
                be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
                td = 1 if rm.get("time_decay") else 0
                pos, _ = bt_range_v3(
                    sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
                    rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"], td, WARMUP, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                net_r = dr - np.abs(np.diff(pos[train_mask], prepend=0.0)) * COMMISSION
                cum = np.cumsum(net_r)
                max_dd = (cum - np.maximum.accumulate(cum)).min() if len(cum) > 0 else 0.0
                exp = np.sum(pos[train_mask] != 0) / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                rm_f = dict(rm); rm_f["_rm_type"] = "atr"
                all_results.append((sp, rm_f, sh, max_dd, exp))

        passing = [(sp, rm, sh, mdd, exp) for sp, rm, sh, mdd, exp in all_results
                   if sh > 0 and mdd > -0.30 and exp > 5]
        if len(passing) == 0:
            passing = sorted(all_results, key=lambda x: x[2], reverse=True)[:3]
        passing = sorted(passing, key=lambda x: x[2], reverse=True)[:10]

        test_positions = []
        test_trades = []
        for sp, rm, sh, mdd, exp in passing:
            sp_key = tuple(sorted(sp.items()))
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
                    be, rm["cooldown_bars"], WARMUP, test_end, 1)
            else:
                be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
                td = 1 if rm.get("time_decay") else 0
                pos, tr = bt_range_v3(
                    sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
                    rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"], td, WARMUP, test_end, 1)
            test_positions.append(pos)
            test_trades.append(tr)

        ensemble = np.mean(test_positions, axis=0)
        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm) for sp, rm, *_ in passing]
        trades_by_year[test_year] = test_trades

    return results_by_year, params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# Section 5: Approach A — S6
# ═══════════════════════════════════════════════════════════════════

def approach_a_s6(tf, close, high, low, open_arr, volume, ind,
                  dates, is_hourly, log_ret,
                  vwap_cache, session_vwap_data, dir_bias_arr):
    """Approach A for S6 with expanded grid."""
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)
    sig_grid_def = S6H_SIG if is_hourly else S6D_SIG
    sig_grid = expand_grid(sig_grid_def)
    rm_atr_grid = expand_grid(S6_RM_ATR_HOURLY if is_hourly else S6_RM_ATR_DAILY)
    rm_vwap_grid = expand_grid(S6_RM_VWAP_HOURLY if is_hourly else S6_RM_VWAP_DAILY)

    results_by_year = {}
    params_by_year = {}
    trades_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        test_end = n if test_year == 2026 else _year_bounds(dates, test_year + 1)[1]
        if train_end <= WARMUP or train_end >= n:
            continue
        train_mask[:] = False
        train_mask[WARMUP:train_end] = True

        # Pre-cache signals
        cached_sigs = {}
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            if sp_key in cached_sigs:
                continue

            w = sp["vwap_window"]
            dm = sp["dev_mult"]
            fs = sp["filter_set"]

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

        all_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, vwap, dev = cached_sigs[sp_key]

            # ATR-based RM
            for rm in rm_atr_grid:
                be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
                td = 1 if rm.get("time_decay") else 0
                pos, _ = bt_range_v3(
                    sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
                    rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"], td, WARMUP, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                net_r = dr - np.abs(np.diff(pos[train_mask], prepend=0.0)) * COMMISSION
                cum = np.cumsum(net_r)
                max_dd = (cum - np.maximum.accumulate(cum)).min() if len(cum) > 0 else 0.0
                exp = np.sum(pos[train_mask] != 0) / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                rm_f = dict(rm); rm_f["_rm_type"] = "atr"
                all_results.append((sp, rm_f, sh, max_dd, exp))

            # VWAP-band RM
            for rm in rm_vwap_grid:
                sl_dist, tp_dist = precompute_vwap_sl_tp(
                    close, sig, vwap, dev, rm["sl_dev"])
                be = 0.5 if rm["breakeven"] else -1.0
                pos, _ = bt_range_vwap_v3(
                    sig, exit_arr, close, high, low, ind["adx14"],
                    sl_dist, tp_dist, 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"], WARMUP, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                net_r = dr - np.abs(np.diff(pos[train_mask], prepend=0.0)) * COMMISSION
                cum = np.cumsum(net_r)
                max_dd = (cum - np.maximum.accumulate(cum)).min() if len(cum) > 0 else 0.0
                exp = np.sum(pos[train_mask] != 0) / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                rm_f = dict(rm); rm_f["_rm_type"] = "vwap"
                all_results.append((sp, rm_f, sh, max_dd, exp))

        passing = [(sp, rm, sh, mdd, exp) for sp, rm, sh, mdd, exp in all_results
                   if sh > 0 and mdd > -0.30 and exp > 5]
        if len(passing) == 0:
            passing = sorted(all_results, key=lambda x: x[2], reverse=True)[:3]
        passing = sorted(passing, key=lambda x: x[2], reverse=True)[:10]

        test_positions = []
        test_trades = []
        for sp, rm, sh, mdd, exp in passing:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_arr, vwap, dev = cached_sigs[sp_key]
            if rm["_rm_type"] == "vwap":
                sl_dist, tp_dist = precompute_vwap_sl_tp(
                    close, sig, vwap, dev, rm["sl_dev"])
                be = 0.5 if rm["breakeven"] else -1.0
                pos, tr = bt_range_vwap_v3(
                    sig, exit_arr, close, high, low, ind["adx14"],
                    sl_dist, tp_dist, 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"], WARMUP, test_end, 1)
            else:
                be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
                td = 1 if rm.get("time_decay") else 0
                pos, tr = bt_range_v3(
                    sig, exit_arr, close, high, low, ind["atr14"], ind["adx14"],
                    rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
                    be, rm["cooldown_bars"], td, WARMUP, test_end, 1)
            test_positions.append(pos)
            test_trades.append(tr)

        ensemble = np.mean(test_positions, axis=0)
        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm) for sp, rm, *_ in passing]
        trades_by_year[test_year] = test_trades

    return results_by_year, params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# Section 6: Approach B — S5 Hourly + S6
# ═══════════════════════════════════════════════════════════════════

def approach_b_s5h(close, high, low, volume, ind,
                   dates, sigma_dict, log_ret,
                   a_params_by_year, pivot_cache):
    """Approach B for S5 hourly."""
    ann = np.sqrt(252 * 9)
    n = len(close)
    b_grid = expand_grid(B_GRIDS_V3["Range"])
    results_by_year = {}
    best_params_by_year = {}
    trades_by_year = {}

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

        a_sigs = []
        for sp, rm in a_params:
            piv_key = (sp["pivot_period"], sp["pivot_type"])
            P, S1, S2, R1, R2 = pivot_cache[piv_key]
            sig, exit_arr = gen_s5_signals_v3(
                close, ind, P, S1, R1, sp["filter_set"],
                sp["confirmation"], sp["entry_buffer"])
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

            val_positions = []
            for sig, exit_arr in a_sigs:
                pos, _ = bt_range_vpred_v3(
                    sig, exit_arr, close, high, low, sigma, ind["adx14"],
                    bp["k_sl"], k_tp, 30.0, 20,
                    k_be, gh, sigma_median, cd, WARMUP, val_end, 0)
                val_positions.append(pos)
            ensemble = np.mean(val_positions, axis=0)
            sh = calc_sharpe_comm(ensemble, log_ret, val_mask, ann)
            if sh > best_sh:
                best_sh = sh; best_bp = dict(bp)

        best_sigma = sigma_dict.get(best_bp.get("horizon", "h1"),
                                    sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        bsm = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        k_be = -1.0 if best_bp.get("k_be") is None else best_bp["k_be"]
        gh = -1.0 if best_bp.get("gamma_hold") is None else best_bp["gamma_hold"]
        cd = best_bp.get("cooldown_bars", 5)
        k_tp = best_bp.get("k_tp", best_bp.get("k_sl", 1.0) * best_bp.get("ratio", 1.5))

        test_positions = []
        test_trades = []
        for sig, exit_arr in a_sigs:
            pos, tr = bt_range_vpred_v3(
                sig, exit_arr, close, high, low, best_sigma, ind["adx14"],
                best_bp["k_sl"], k_tp, 30.0, 20,
                k_be, gh, bsm, cd, WARMUP, test_end, 1)
            test_positions.append(pos)
            test_trades.append(tr)
        ensemble = np.mean(test_positions, axis=0)
        results_by_year[test_year] = ensemble
        best_params_by_year[test_year] = best_bp
        trades_by_year[test_year] = test_trades

    return results_by_year, best_params_by_year, trades_by_year


def approach_b_s6(tf, close, high, low, volume, ind,
                  dates, sigma_dict, is_hourly, log_ret,
                  a_params_by_year, vwap_cache, session_vwap_data, dir_bias_arr):
    """Approach B for S6."""
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
    n = len(close)
    b_grid = expand_grid(B_GRIDS_V3["Range"])
    results_by_year = {}
    best_params_by_year = {}
    trades_by_year = {}

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

        # Regenerate A's signals
        a_sigs = []
        for sp, rm in a_params:
            w = sp["vwap_window"]; dm = sp["dev_mult"]; fs = sp["filter_set"]
            if is_hourly:
                vtype = sp["vwap_type"]
                vwap, dev = session_vwap_data if vtype == "session" else vwap_cache[w]
                dba = None
            else:
                dba = dir_bias_arr if sp["dir_bias"] == "ON" else None
                vwap, dev = vwap_cache[w]
            sig, exit_arr = gen_s6_signals_v2(close, ind, vwap, dev, dm, fs, dba)
            a_sigs.append((sig, exit_arr))

        best_sh = -999.0
        best_bp = b_grid[0] if b_grid else {}
        base_hold = 20 if is_hourly else 10
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
            for sig, exit_arr in a_sigs:
                pos, _ = bt_range_vpred_v3(
                    sig, exit_arr, close, high, low, sigma, ind["adx14"],
                    bp["k_sl"], k_tp, 30.0, base_hold,
                    k_be, gh, sigma_median, cd, WARMUP, val_end, 0)
                val_positions.append(pos)
            ensemble = np.mean(val_positions, axis=0)
            sh = calc_sharpe_comm(ensemble, log_ret, val_mask, ann)
            if sh > best_sh:
                best_sh = sh; best_bp = dict(bp)

        best_sigma = sigma_dict.get(best_bp.get("horizon", "h1"),
                                    sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        bsm = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        k_be = -1.0 if best_bp.get("k_be") is None else best_bp["k_be"]
        gh = -1.0 if best_bp.get("gamma_hold") is None else best_bp["gamma_hold"]
        cd = best_bp.get("cooldown_bars", 5)
        k_tp = best_bp.get("k_tp", best_bp.get("k_sl", 1.0) * best_bp.get("ratio", 1.5))

        test_positions = []
        test_trades = []
        for sig, exit_arr in a_sigs:
            pos, tr = bt_range_vpred_v3(
                sig, exit_arr, close, high, low, best_sigma, ind["adx14"],
                best_bp["k_sl"], k_tp, 30.0, base_hold,
                k_be, gh, bsm, cd, WARMUP, test_end, 1)
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

def process_ticker(ticker, daily_df, hourly_df, vpred_df):
    """Process one ticker: S5 hourly + S6 daily + S6 hourly."""
    results = {}

    for tf in ["daily", "hourly"]:
        is_hourly = tf == "hourly"
        if is_hourly:
            tdf = hourly_df[hourly_df["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
            if len(tdf) == 0: continue
            close = tdf["close"].values.astype(np.float64)
            high = tdf["high"].values.astype(np.float64)
            low = tdf["low"].values.astype(np.float64)
            open_arr = tdf["open"].values.astype(np.float64)
            volume = tdf["volume"].values.astype(np.float64)
            dates = pd.to_datetime(tdf["datetime"].values)
        else:
            tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if len(tdf) == 0: continue
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
                if col not in vp.columns: continue
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

        ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
        bpy = 252 * 9 if is_hourly else 252

        # ── S5 hourly only ──
        if is_hourly:
            h_dts = tdf["datetime"].values
            pivot_cache = {}
            for ptype in ["classic", "fibonacci"]:
                pivot_cache[("weekly", ptype)] = map_weekly_pivots_to_hourly(
                    daily_df, ticker, h_dts, ptype)
                pivot_cache[("monthly", ptype)] = map_monthly_pivots_to_hourly(
                    daily_df, ticker, h_dts, ptype)

            a_res, a_par, a_tr = approach_a_s5h(
                close, high, low, open_arr, volume, ind,
                dates, log_ret, pivot_cache)
            for year, pos_arr in a_res.items():
                _store_metrics(results, "S5", tf, "A", year, pos_arr,
                               log_ret, dates, n, ann, bpy, ticker)

            b_res, b_par, b_tr = approach_b_s5h(
                close, high, low, volume, ind,
                dates, sigma_dict, log_ret, a_par, pivot_cache)
            for year, pos_arr in b_res.items():
                _store_metrics(results, "S5", tf, "B", year, pos_arr,
                               log_ret, dates, n, ann, bpy, ticker)

            c_res, _, _ = approach_c_one("S5", tf, dates, a_res, sigma_dict, True, log_ret, n)
            for year, pos_arr in c_res.items():
                _store_metrics(results, "S5", tf, "C", year, pos_arr,
                               log_ret, dates, n, ann, bpy, ticker)

            d_res, _, _ = approach_d_one("S5", tf, dates, a_res, sigma_dict, True, log_ret, n)
            for year, pos_arr in d_res.items():
                _store_metrics(results, "S5", tf, "D", year, pos_arr,
                               log_ret, dates, n, ann, bpy, ticker)

        # ── S6 daily + hourly ──
        vwap_windows = [5, 10, 15, 20, 30, 60]
        vwap_cache = precompute_vwap_cache_ext(close, high, low, volume, vwap_windows)

        session_vwap_data = None
        if is_hourly:
            session_vwap_data = compute_session_vwap(close, high, low, volume,
                                                     tdf["datetime"].values)

        dir_bias_arr = None
        if not is_hourly:
            d_dates, d_above = build_daily_trend(daily_df, ticker)
            dir_bias_arr = d_above  # already aligned to daily bars
            # Re-align via date matching (daily_trend uses same ticker daily)
            # build_daily_trend returns (dates, above) for this ticker
            # We need to match to our date array
            d_dates_np = np.array(d_dates, dtype="datetime64[D]")
            our_dates = np.array(dates.values, dtype="datetime64[D]")
            idx = np.searchsorted(d_dates_np, our_dates, side="right") - 1
            valid = idx >= 0
            dir_bias_arr = np.full(n, np.nan)
            dir_bias_arr[valid] = d_above[idx[valid]]

        a_res6, a_par6, a_tr6 = approach_a_s6(
            tf, close, high, low, open_arr, volume, ind,
            dates, is_hourly, log_ret,
            vwap_cache, session_vwap_data, dir_bias_arr)
        for year, pos_arr in a_res6.items():
            _store_metrics(results, "S6", tf, "A", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        b_res6, b_par6, b_tr6 = approach_b_s6(
            tf, close, high, low, volume, ind,
            dates, sigma_dict, is_hourly, log_ret,
            a_par6, vwap_cache, session_vwap_data, dir_bias_arr)
        for year, pos_arr in b_res6.items():
            _store_metrics(results, "S6", tf, "B", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        c_res6, _, _ = approach_c_one("S6", tf, dates, a_res6, sigma_dict, is_hourly, log_ret, n)
        for year, pos_arr in c_res6.items():
            _store_metrics(results, "S6", tf, "C", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

        d_res6, _, _ = approach_d_one("S6", tf, dates, a_res6, sigma_dict, is_hourly, log_ret, n)
        for year, pos_arr in d_res6.items():
            _store_metrics(results, "S6", tf, "D", year, pos_arr,
                           log_ret, dates, n, ann, bpy, ticker)

    return {"results": results}


# ═══════════════════════════════════════════════════════════════════
# Section 8: Main
# ═══════════════════════════════════════════════════════════════════

_G_DAILY = None; _G_HOURLY = None; _G_VPRED = None

def _pool_init(daily, hourly, vpred):
    global _G_DAILY, _G_HOURLY, _G_VPRED
    _G_DAILY = daily; _G_HOURLY = hourly; _G_VPRED = vpred

def _worker_func(ticker):
    return ticker, process_ticker(ticker, _G_DAILY, _G_HOURLY, _G_VPRED)

def warmup_numba_s5s6():
    if not HAS_NUMBA: return
    n = 50
    sig = np.zeros(n, dtype=np.int8); sig[10] = 1
    ex = np.zeros(n, dtype=bool)
    c = np.random.randn(n).cumsum() + 100
    h = c + 1; l = c - 1
    a = np.ones(n); adx = np.ones(n) * 15
    sigma = np.ones(n) * 0.02
    sl_d = np.ones(n) * 2.0; tp_d = np.ones(n) * 3.0
    bt_range_v3(sig, ex, c, h, l, a, adx, 1.0, 1.5, 30.0, 10, 1.0, 5, 1, 5, n, 0)
    bt_range_vpred_v3(sig, ex, c, h, l, sigma, adx, 1.0, 2.0, 30.0, 10, 1.0, 0.5, 0.02, 5, 5, n, 0)
    bt_range_pivot_v3(sig, ex, c, h, l, adx, sl_d, tp_d, 30.0, 10, 0.5, 5, 5, n, 0)
    bt_range_vwap_v3(sig, ex, c, h, l, adx, sl_d, tp_d, 30.0, 10, 0.5, 5, 5, n, 0)
    _apply_hysteresis(np.random.rand(n), 1, 0.5, 0.4, n)
    print("  Numba warmup done")


def main():
    t0 = time.time()
    print("=" * 70)
    print("S5 hourly + S6 daily/hourly RERUN")
    print(f"Tickers: {len(TICKERS)}, Approaches: A/B/C/D")
    print("=" * 70)

    # Grid sizes
    n_s5h_sig = len(expand_grid(S5H_SIG))
    n_s5h_rmp = len(expand_grid(S5H_RM_PIVOT))
    n_s5h_rma = len(expand_grid(S5H_RM_ATR))
    print(f"  S5 hourly: {n_s5h_sig} signal x ({n_s5h_rmp} pivot + {n_s5h_rma} ATR) = {n_s5h_sig * (n_s5h_rmp + n_s5h_rma)} combos")

    for label, sg_def, rma, rmv in [
        ("S6 daily", S6D_SIG, S6_RM_ATR_DAILY, S6_RM_VWAP_DAILY),
        ("S6 hourly", S6H_SIG, S6_RM_ATR_HOURLY, S6_RM_VWAP_HOURLY),
    ]:
        ns = len(expand_grid(sg_def))
        nra = len(expand_grid(rma))
        nrv = len(expand_grid(rmv))
        print(f"  {label}: {ns} signal x ({nra} ATR + {nrv} VWAP) = {ns * (nra + nrv)} combos")

    n_b = len(expand_grid(B_GRIDS_V3["Range"]))
    print(f"  B grid (Range): {n_b} combos")

    daily, hourly, vpred = load_data()

    print(f"\nWarming up numba...")
    warmup_numba_s5s6()

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
            result_dict = process_ticker(ticker, daily, hourly, vpred)
            all_results[ticker] = result_dict["results"]
            print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({time.time()-t0:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")

    # Collect new rows
    new_rows = []
    for ticker, ticker_results in all_results.items():
        for key, met in ticker_results.items():
            met.pop("_net_returns", None)
            new_rows.append(met)

    if not new_rows:
        print("  No results!"); return

    new_df = pd.DataFrame(new_rows)

    # Load previous and replace
    prev_path = OUT_DATA / "wf_v3_all_results.csv"
    if prev_path.exists():
        prev_df = pd.read_csv(prev_path)

        # Identify which (strategy, timeframe) combos to replace
        replace_combos = [
            ("S5_PivotPoints", "hourly"),
            ("S6_VWAP", "daily"),
            ("S6_VWAP", "hourly"),
        ]
        keep_mask = ~prev_df.apply(
            lambda r: (r["strategy"], r["timeframe"]) in replace_combos, axis=1)
        old_replaced = prev_df[~keep_mask]
        other = prev_df[keep_mask]
        combined = pd.concat([other, new_df], ignore_index=True)
        combined.to_csv(prev_path, index=False)
        print(f"\n  Updated: {len(combined)} rows")
        print(f"  Replaced {len(old_replaced)} old rows with {len(new_df)} new")
    else:
        new_df.to_csv(OUT_DATA / "wf_v3_s5s6_rerun.csv", index=False)
        print(f"\n  Saved {len(new_df)} rows (no previous file)")

    # ── BEFORE vs AFTER ──
    print("\n" + "=" * 70)
    print("=== BEFORE vs AFTER ===")
    print("=" * 70)
    common_years = BCD_TEST_YEARS

    if prev_path.exists():
        for sid_name, tf in [("S5_PivotPoints", "hourly"),
                              ("S6_VWAP", "daily"),
                              ("S6_VWAP", "hourly")]:
            old_sub = old_replaced[(old_replaced["strategy"] == sid_name) &
                                   (old_replaced["timeframe"] == tf) &
                                   (old_replaced["year"].isin(common_years))]
            new_sub = new_df[(new_df["strategy"] == sid_name) &
                             (new_df["timeframe"] == tf) &
                             (new_df["year"].isin(common_years))]
            print(f"\n  {sid_name} {tf}:")
            for approach in ["A", "B", "C", "D"]:
                old_a = old_sub[old_sub["approach"] == approach]
                new_a = new_sub[new_sub["approach"] == approach]
                old_sh = old_a["sharpe"].mean() if len(old_a) > 0 else float('nan')
                new_sh = new_a["sharpe"].mean() if len(new_a) > 0 else float('nan')
                delta = new_sh - old_sh if not (np.isnan(old_sh) or np.isnan(new_sh)) else float('nan')
                delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
                old_str = f"{old_sh:.4f}" if not np.isnan(old_sh) else "N/A"
                new_str = f"{new_sh:.4f}" if not np.isnan(new_sh) else "N/A"
                print(f"    {approach}: BEFORE={old_str}  AFTER={new_str}  delta={delta_str}")

    # ── Detailed new results ──
    for sid_name, tf in [("S5_PivotPoints", "hourly"),
                          ("S6_VWAP", "daily"),
                          ("S6_VWAP", "hourly")]:
        sub = new_df[(new_df["strategy"] == sid_name) & (new_df["timeframe"] == tf)]
        if len(sub) == 0: continue

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

        print(f"\n  === {sid_name} {tf} (NEW) ===")
        hdr = f"  {'approach':>8s}  {'MeanSharpe':>10s}  {'AnnRet%':>8s}  {'AnnVol%':>8s}  {'MaxDD%':>7s}  {'Exp%':>6s}  {'Tr/yr':>6s}  {'Win%':>6s}"
        print(hdr)
        a_sh = None
        for _, r in summary.iterrows():
            if r["approach"] == "A": a_sh = r["MeanSharpe"]
            print(f"  {r['approach']:>8s}  {r['MeanSharpe']:10.4f}  {r['AnnReturn']:8.2f}  {r['AnnVol']:8.2f}  {r['MaxDD']:7.2f}  {r['Exposure']:6.2f}  {r['TradesPerYr']:6.2f}  {r['WinRate']:6.2f}")
        if a_sh is not None:
            parts = []
            for appr in ["B", "C", "D"]:
                rs = summary.loc[summary["approach"] == appr]
                if len(rs) > 0:
                    parts.append(f"{appr}={rs['MeanSharpe'].values[0] - a_sh:+.3f}")
            if parts:
                print(f"  \u0394Sharpe vs A:  {', '.join(parts)}")

        # Top/worst tickers
        sub_c = sub[sub["year"].isin(common_years)]
        if len(sub_c) > 0:
            ta = sub_c.groupby(["approach", "ticker"]).agg(sharpe=("sharpe", "median")).reset_index()
            best_appr = summary.loc[summary["MeanSharpe"].idxmax(), "approach"]
            asub = ta[ta["approach"] == best_appr].sort_values("sharpe", ascending=False)
            if len(asub) >= 5:
                top5 = asub.head(5)
                worst5 = asub.tail(5)
                print(f"  Top-5 ({best_appr}):   " + ", ".join(f"{r['ticker']}({r['sharpe']:.2f})" for _, r in top5.iterrows()))
                print(f"  Worst-5 ({best_appr}): " + ", ".join(f"{r['ticker']}({r['sharpe']:.2f})" for _, r in worst5.iterrows()))

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
