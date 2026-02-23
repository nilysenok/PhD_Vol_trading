#!/usr/bin/env python3
"""
strategies_calibration.py — Per-ticker parameter calibration for 5 selected strategies.
Train: 2020-2021 (grid search, max Sharpe per ticker).
Test: 2022-2026 (fixed best params).

Two-stage grid search:
  Stage 1: strategy + RM params (default filters)
  Stage 2: filter params (best strategy+RM from stage 1)
"""

import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results" / "final" / "strategies" / "data"
OUT_DIR = BASE / "results" / "final" / "strategies"
OUT_TABLES = OUT_DIR / "tables"
OUT_DATA = OUT_DIR / "data"

TICKERS = sorted([
    "AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
    "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"
])

WARMUP = 200
TRAIN_START = np.datetime64("2020-01-01")
TRAIN_END = np.datetime64("2022-01-01")  # exclusive
TEST_START = np.datetime64("2022-01-01")

# Default params from v3
V3_DEFAULTS = {
    "S1_MA_Reversion": {
        "ma_window": 20, "z_entry": 2.0, "z_exit": 0.5, "max_hold": 20,
        "sl_mult": 1.5, "tp_mult": 2.0,
        "rsi_long": 40, "rsi_short": 60, "vol_regime_thresh": 1.2, "consec_candles": 3,
    },
    "S4_Donchian_daily": {
        "dc_window": 20,
        "initial_sl_mult": 2.5, "trail_n": 10, "trail_atr_mult": 2.5, "breakeven_thresh": 1.5,
        "adx_thresh": 20, "vol_confirm": 1.0,
    },
    "S4_Donchian_hourly": {
        "dc_window": 20,
        "initial_sl_mult": 2.5, "trail_n": 20, "trail_atr_mult": 2.5, "breakeven_thresh": 1.5,
        "adx_thresh": 20, "vol_confirm": 1.0,
    },
    "S5_Supertrend": {
        "atr_period": 14, "multiplier": 3.0,
        "initial_sl_mult": 2.5, "trail_n": 20, "trail_atr_mult": 2.5, "breakeven_thresh": 1.5,
        "adx_thresh": 20,
    },
    "S6_DualMA": {
        "fast_window": 20, "slow_window": 80,
        "initial_sl_mult": 2.5, "trail_n": 20, "trail_atr_mult": 2.5, "breakeven_thresh": 1.5,
        "adx_thresh": 15,
    },
}

# Grid definitions
GRID = {
    "S1_stage1": {
        "ma_window": [15, 20, 25],
        "z_entry": [1.5, 2.0, 2.5],
        "z_exit": [0.3, 0.5, 0.7],
        "max_hold": [10, 15, 20, 25],
        "sl_mult": [1.0, 1.5, 2.0],
        "tp_mult": [1.5, 2.0, 2.5, 3.0],
    },
    "S1_stage2": {
        "rsi_long": [35, 40, 45],
        "rsi_short": [55, 60, 65],
        "vol_regime_thresh": [1.0, 1.2, 1.5],
        "consec_candles": [2, 3, 4],
    },
    "S4_stage1": {
        "dc_window": [10, 15, 20, 25, 30],
        "initial_sl_mult": [2.0, 2.5, 3.0],
        "trail_n": [10, 15, 20, 25],
        "trail_atr_mult": [2.0, 2.5, 3.0],
        "breakeven_thresh": [1.0, 1.5, 2.0],
    },
    "S4_stage2": {
        "adx_thresh": [15, 20, 25],
        "vol_confirm": [0.8, 1.0, 1.2],
    },
    "S5_stage1": {
        "atr_period": [10, 14, 20],
        "multiplier": [2.0, 2.5, 3.0, 3.5],
        "initial_sl_mult": [2.0, 2.5, 3.0],
        "trail_n": [10, 15, 20, 25],
        "trail_atr_mult": [2.0, 2.5, 3.0],
        "breakeven_thresh": [1.0, 1.5, 2.0],
    },
    "S5_stage2": {
        "adx_thresh": [15, 20, 25],
    },
    "S6_stage1": {
        "fast_window": [10, 15, 20, 25],
        "slow_window": [60, 80, 100, 120],
        "initial_sl_mult": [2.0, 2.5, 3.0],
        "trail_n": [10, 15, 20, 25],
        "trail_atr_mult": [2.0, 2.5, 3.0],
        "breakeven_thresh": [1.0, 1.5, 2.0],
    },
    "S6_stage2": {
        "adx_thresh": [10, 15, 20],
    },
}


# ════════════════════════════════════════════════════════════
# Indicator functions (from v3)
# ════════════════════════════════════════════════════════════

def calc_sma(arr, w):
    return pd.Series(arr).rolling(w, min_periods=w).mean().values

def calc_std(arr, w):
    s = pd.Series(arr).rolling(w).std(ddof=1).values
    return np.where(np.isnan(s) | (s < 1e-12), 1e-12, s)

def calc_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    n = len(close)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-12:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi

def calc_atr(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    return pd.Series(tr).rolling(period).mean().values

def calc_adx(high, low, close, period=14):
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr_s = np.full(n, np.nan)
    pdi_s = np.full(n, np.nan)
    mdi_s = np.full(n, np.nan)
    atr_s[period] = np.sum(tr[1:period + 1])
    pdi_s[period] = np.sum(plus_dm[1:period + 1])
    mdi_s[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdi_s[i] = pdi_s[i - 1] - pdi_s[i - 1] / period + plus_dm[i]
        mdi_s[i] = mdi_s[i - 1] - mdi_s[i - 1] / period + minus_dm[i]
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if atr_s[i] is not None and atr_s[i] > 1e-12:
            pdi = 100.0 * pdi_s[i] / atr_s[i]
            mdi = 100.0 * mdi_s[i] / atr_s[i]
        else:
            pdi, mdi = 0.0, 0.0
        s = pdi + mdi
        dx[i] = 100.0 * abs(pdi - mdi) / s if s > 1e-12 else 0.0
    adx = np.full(n, np.nan)
    start = 2 * period
    if start < n:
        adx[start] = np.nanmean(dx[period:start + 1])
        for i in range(start + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx

def calc_supertrend(high, low, close, period=14, mult=3.0):
    n = len(close)
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    bu = hl2 + mult * atr
    bl = hl2 - mult * atr
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    st = np.full(n, np.nan)
    d = np.zeros(n)
    s = period
    fu[s] = bu[s]; fl[s] = bl[s]
    d[s] = 1.0 if close[s] > bu[s] else -1.0
    st[s] = fl[s] if d[s] == 1 else fu[s]
    for t in range(s + 1, n):
        if np.isnan(atr[t]):
            d[t] = d[t-1]; fu[t] = bu[t]; fl[t] = bl[t]; st[t] = st[t-1]
            continue
        fu[t] = bu[t] if (bu[t] < fu[t-1] or close[t-1] > fu[t-1]) else fu[t-1]
        fl[t] = bl[t] if (bl[t] > fl[t-1] or close[t-1] < fl[t-1]) else fl[t-1]
        if d[t-1] == 1:
            d[t] = -1.0 if close[t] < fl[t] else 1.0
        else:
            d[t] = 1.0 if close[t] > fu[t] else -1.0
        st[t] = fl[t] if d[t] == 1 else fu[t]
    return st, d


# ════════════════════════════════════════════════════════════
# Base indicators (computed once per ticker)
# ════════════════════════════════════════════════════════════

def compute_base(close, high, low, open_arr, volume):
    ind = {}
    ind["rsi14"] = calc_rsi(close, 14)
    ind["atr14"] = calc_atr(high, low, close, 14)
    ind["adx14"] = calc_adx(high, low, close, 14)
    ind["vol_sma20"] = calc_sma(volume, 20)
    ind["vol_sma5"] = calc_sma(volume, 5)
    atr_sma50 = calc_sma(ind["atr14"], 50)
    atr_sma50_safe = np.where(np.isnan(atr_sma50) | (atr_sma50 < 1e-12), 1e-12, atr_sma50)
    ind["vol_regime"] = ind["atr14"] / atr_sma50_safe
    red = (close < open_arr).astype(float)
    green = (close > open_arr).astype(float)
    ind["red_count5"] = pd.Series(red).rolling(5).sum().values
    ind["green_count5"] = pd.Series(green).rolling(5).sum().values
    # Precompute rolling high/low for all possible trail_n values
    for tn in [10, 15, 20, 25]:
        ind[f"rh_{tn}"] = pd.Series(high).rolling(tn + 1, min_periods=1).max().values
        ind[f"rl_{tn}"] = pd.Series(low).rolling(tn + 1, min_periods=1).min().values
    # Volume filter pass mask
    vs20 = ind["vol_sma20"]
    vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), np.inf, vs20)
    ind["vf_pass"] = ~(~np.isnan(vs20) & (vs20 > 0) & (volume < 0.5 * vs20_safe))
    ind["_volume"] = volume
    return ind


# ════════════════════════════════════════════════════════════
# Vectorized signal generators
# ════════════════════════════════════════════════════════════

def gen_s1_signals(close, ind, ma_window, z_entry,
                   rsi_long, rsi_short, vol_regime_thresh, consec_candles):
    """Returns (entry_signal_array, z_array)."""
    sma = calc_sma(close, ma_window)
    std = calc_std(close, ma_window)
    z = (close - sma) / std

    rsi = ind["rsi14"]
    vr = ind["vol_regime"]
    rc5 = ind["red_count5"]
    gc5 = ind["green_count5"]
    vs5 = ind["vol_sma5"]
    vol = ind["_volume"]
    vfp = ind["vf_pass"]

    f1 = ~np.isnan(vr) & (vr < vol_regime_thresh)
    vs5_safe = np.where(np.isnan(vs5) | (vs5 <= 0), np.inf, vs5)
    f3 = ~(~np.isnan(vs5) & (vs5 > 0) & (vol >= vs5_safe))
    valid = ~np.isnan(z) & ~np.isnan(rsi)

    entry_long = (valid & (z < -z_entry) & f1 &
                  ~np.isnan(rc5) & (rc5 >= consec_candles) &
                  f3 & (rsi < rsi_long) & vfp)
    entry_short = (valid & (z > z_entry) & f1 &
                   ~np.isnan(gc5) & (gc5 >= consec_candles) &
                   f3 & (rsi > rsi_short) & vfp)

    sig = np.where(entry_long, 1, np.where(entry_short, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    return sig, z


def gen_s4_signals(close, high, low, ind, dc_window, adx_thresh, vol_confirm,
                   daily_trend=None):
    """Returns (entry_signal, exit_long, exit_short)."""
    high_ch = pd.Series(high).rolling(dc_window).max().values
    low_ch = pd.Series(low).rolling(dc_window).min().values
    prev_hc = np.empty_like(high_ch)
    prev_hc[0] = np.nan
    prev_hc[1:] = high_ch[:-1]
    prev_lc = np.empty_like(low_ch)
    prev_lc[0] = np.nan
    prev_lc[1:] = low_ch[:-1]

    adx = ind["adx14"]
    vol = ind["_volume"]
    vs20 = ind["vol_sma20"]
    vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), 1.0, vs20)
    vol_ratio = vol / vs20_safe
    vfp = ind["vf_pass"]
    valid = ~np.isnan(adx) & ~np.isnan(prev_hc)

    entry_long = valid & (close > prev_hc) & (adx > adx_thresh) & (vol_ratio > vol_confirm) & vfp
    entry_short = valid & (close < prev_lc) & (adx > adx_thresh) & (vol_ratio > vol_confirm) & vfp

    if daily_trend is not None:
        dt_valid = ~np.isnan(daily_trend)
        entry_long = entry_long & (~dt_valid | (daily_trend >= 0.5))
        entry_short = entry_short & (~dt_valid | (daily_trend < 0.5))

    sig = np.where(entry_long, 1, np.where(entry_short, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    exit_long = ~np.isnan(prev_lc) & (close < prev_lc)
    exit_short = ~np.isnan(prev_hc) & (close > prev_hc)
    return sig, exit_long, exit_short


def gen_s5_signals(close, high, low, ind, atr_period, multiplier, adx_thresh,
                   daily_trend=None):
    """Returns (entry_signal, exit_long, exit_short)."""
    st, _ = calc_supertrend(high, low, close, atr_period, multiplier)
    st_prev = np.empty_like(st)
    st_prev[0] = np.nan
    st_prev[1:] = st[:-1]
    close_prev = np.empty_like(close)
    close_prev[0] = close[0]
    close_prev[1:] = close[:-1]

    adx = ind["adx14"]
    vfp = ind["vf_pass"]
    valid = ~np.isnan(st) & ~np.isnan(st_prev) & ~np.isnan(adx)

    entry_long = valid & (close > st) & (close_prev <= st_prev) & (adx > adx_thresh) & vfp
    entry_short = valid & (close < st) & (close_prev >= st_prev) & (adx > adx_thresh) & vfp

    if daily_trend is not None:
        dt_valid = ~np.isnan(daily_trend)
        entry_long = entry_long & (~dt_valid | (daily_trend >= 0.5))
        entry_short = entry_short & (~dt_valid | (daily_trend < 0.5))

    sig = np.where(entry_long, 1, np.where(entry_short, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    exit_long = valid & (close < st) & (close_prev >= st_prev)
    exit_short = valid & (close > st) & (close_prev <= st_prev)
    return sig, exit_long, exit_short


def gen_s6_signals(close, ind, fast_window, slow_window, adx_thresh):
    """Returns (entry_signal, exit_long, exit_short)."""
    ma_f = calc_sma(close, fast_window)
    ma_s = calc_sma(close, slow_window)
    fp = np.empty_like(ma_f)
    fp[0] = np.nan
    fp[1:] = ma_f[:-1]
    sp = np.empty_like(ma_s)
    sp[0] = np.nan
    sp[1:] = ma_s[:-1]

    adx = ind["adx14"]
    vfp = ind["vf_pass"]
    valid = ~np.isnan(ma_f) & ~np.isnan(ma_s) & ~np.isnan(fp) & ~np.isnan(sp) & ~np.isnan(adx)

    entry_long = valid & (ma_f > ma_s) & (fp <= sp) & (adx > adx_thresh) & vfp
    entry_short = valid & (ma_f < ma_s) & (fp >= sp) & (adx > adx_thresh) & vfp

    sig = np.where(entry_long, 1, np.where(entry_short, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    exit_long = valid & (ma_f < ma_s) & (fp >= sp)
    exit_short = valid & (ma_f > ma_s) & (fp <= sp)
    return sig, exit_long, exit_short


# ════════════════════════════════════════════════════════════
# Backtest engines
# ════════════════════════════════════════════════════════════

def bt_contrarian(sig_arr, exit_arr, close, high, low, atr14,
                  sl_mult, tp_mult, max_hold, warmup, end_idx):
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0
    sl = 0.0
    tp = 0.0
    held = 0

    for t in range(warmup, n):
        if cp != 0.0:
            held += 1
            closed = False
            if cp == 1.0:
                if low[t] <= sl:
                    closed = True
                elif high[t] >= tp:
                    closed = True
            else:
                if high[t] >= sl:
                    closed = True
                elif low[t] <= tp:
                    closed = True
            if not closed and exit_arr[t]:
                closed = True
            if not closed and held >= max_hold:
                closed = True
            if closed:
                cp = 0.0
                held = 0

        if cp == 0.0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a:
                    a = 0.0
                cp = 1.0
                sl = close[t] - sl_mult * a
                tp = close[t] + tp_mult * a
                held = 0
            elif s == -1:
                a = atr14[t]
                if a != a:
                    a = 0.0
                cp = -1.0
                sl = close[t] + sl_mult * a
                tp = close[t] - tp_mult * a
                held = 0

        pos[t] = cp

    return pos


def bt_trend(sig_arr, exit_long, exit_short, close, high, low, atr14,
             rolling_high, rolling_low, isl_mult, trail_mult, be_thresh,
             warmup, end_idx):
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0
    sl = np.nan
    ep = 0.0
    ea = 0.0
    be = False

    for t in range(warmup, n):
        if cp != 0.0:
            ca = atr14[t]
            if ca != ca:
                ca = ea
            closed = False

            if cp == 1.0:
                tsl = rolling_high[t] - trail_mult * ca
                if not be and (close[t] - ep) > be_thresh * ea:
                    be = True
                if be:
                    tsl = max(tsl, ep)
                if sl != sl:
                    sl = tsl
                else:
                    sl = max(sl, tsl)
                if low[t] <= sl:
                    closed = True
                elif exit_long[t]:
                    closed = True
            else:
                tsl = rolling_low[t] + trail_mult * ca
                if not be and (ep - close[t]) > be_thresh * ea:
                    be = True
                if be:
                    tsl = min(tsl, ep)
                if sl != sl:
                    sl = tsl
                else:
                    sl = min(sl, tsl)
                if high[t] >= sl:
                    closed = True
                elif exit_short[t]:
                    closed = True

            if closed:
                cp = 0.0
                sl = np.nan
                ep = 0.0
                ea = 0.0
                be = False

        if cp == 0.0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a:
                    a = 0.0
                cp = 1.0
                ep = close[t]
                ea = a
                be = False
                sl = ep - isl_mult * a
            elif s == -1:
                a = atr14[t]
                if a != a:
                    a = 0.0
                cp = -1.0
                ep = close[t]
                ea = a
                be = False
                sl = ep + isl_mult * a

        pos[t] = cp

    return pos


# ════════════════════════════════════════════════════════════
# Sharpe and metrics
# ════════════════════════════════════════════════════════════

def calc_sharpe(positions, log_ret, mask, ann_factor):
    dr = positions[mask] * log_ret[mask]
    if len(dr) == 0:
        return -999.0
    s = np.std(dr, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(dr) / s * ann_factor


def calc_all_metrics(positions, log_ret, mask, ann_factor, bars_per_year):
    dr = positions[mask] * log_ret[mask]
    pos = positions[mask]
    n = len(dr)
    if n == 0:
        return {}
    active = pos != 0
    n_active = int(active.sum())
    exposure = n_active / n if n > 0 else 0.0
    mean_r = np.mean(dr)
    std_r = np.std(dr, ddof=1) if n > 1 else 1e-10
    sharpe = mean_r / std_r * ann_factor if std_r > 1e-12 else 0.0
    ann_ret = mean_r * bars_per_year
    ann_vol = std_r * ann_factor
    cum = np.cumsum(dr)
    rmax = np.maximum.accumulate(cum)
    dd = cum - rmax
    max_dd = dd.min() if len(dd) > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
    if n_active > 0:
        ar = dr[active]
        win_rate = float((ar > 0).sum()) / n_active
        gp = ar[ar > 0].sum()
        gl = abs(ar[ar < 0].sum())
        pf = gp / gl if gl > 1e-12 else (99.0 if gp > 0 else 0.0)
        n_win = (ar > 0).sum()
        n_loss = (ar < 0).sum()
        avg_win = ar[ar > 0].mean() if n_win > 0 else 0.0
        avg_loss = abs(ar[ar < 0].mean()) if n_loss > 0 else 1e-12
        payoff = avg_win / avg_loss if avg_loss > 1e-12 else (99.0 if avg_win > 0 else 0.0)
    else:
        win_rate = pf = payoff = 0.0
    changes = np.sum(np.abs(np.diff(pos)) > 0) if len(pos) > 1 else 0
    n_years = n / bars_per_year if bars_per_year > 0 else 1
    turnover = changes / n_years if n_years > 0 else 0
    trades = 0
    in_t = False
    t_bars = 0
    for i in range(len(pos)):
        if pos[i] != 0 and not in_t:
            trades += 1
            in_t = True
        elif pos[i] == 0:
            in_t = False
        if pos[i] != 0:
            t_bars += 1
    avg_trade = t_bars / trades if trades > 0 else 0
    return {
        "sharpe": round(sharpe, 4), "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2), "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 4), "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(pf, 3), "payoff_ratio": round(payoff, 3),
        "exposure_pct": round(exposure * 100, 2), "turnover": round(turnover, 1),
        "avg_trade_bars": round(avg_trade, 1), "n_trades": trades,
    }


# ════════════════════════════════════════════════════════════
# Per-strategy calibration functions
# ════════════════════════════════════════════════════════════

def calibrate_s1(close, high, low, open_arr, volume, dts, ind):
    """Calibrate S1_MA_Reversion (daily) for one ticker."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    train_mask = (dts >= TRAIN_START) & (dts < TRAIN_END)
    test_mask = dts >= TEST_START
    train_end_idx = int(np.searchsorted(dts, TRAIN_END, side="left"))
    ann = np.sqrt(252)
    bpy = 252
    defaults = V3_DEFAULTS["S1_MA_Reversion"]

    # Stage 1: strategy + RM, default filters
    best_sh = -999.0
    best_p1 = None
    df_rsi_long = defaults["rsi_long"]
    df_rsi_short = defaults["rsi_short"]
    df_vrt = defaults["vol_regime_thresh"]
    df_cc = defaults["consec_candles"]

    for ma_w in GRID["S1_stage1"]["ma_window"]:
        for z_ent in GRID["S1_stage1"]["z_entry"]:
            sig, z_arr = gen_s1_signals(close, ind, ma_w, z_ent,
                                        df_rsi_long, df_rsi_short, df_vrt, df_cc)
            for z_ex in GRID["S1_stage1"]["z_exit"]:
                exit_c = ~np.isnan(z_arr) & (np.abs(z_arr) < z_ex)
                for mh in GRID["S1_stage1"]["max_hold"]:
                    for sl_m in GRID["S1_stage1"]["sl_mult"]:
                        for tp_m in GRID["S1_stage1"]["tp_mult"]:
                            pos = bt_contrarian(sig, exit_c, close, high, low,
                                                ind["atr14"], sl_m, tp_m, mh,
                                                WARMUP, train_end_idx)
                            sh = calc_sharpe(pos, log_ret, train_mask, ann)
                            if sh > best_sh:
                                best_sh = sh
                                best_p1 = {"ma_window": ma_w, "z_entry": z_ent,
                                            "z_exit": z_ex, "max_hold": mh,
                                            "sl_mult": sl_m, "tp_mult": tp_m}

    if best_p1 is None:
        best_p1 = {k: defaults[k] for k in ["ma_window", "z_entry", "z_exit",
                                              "max_hold", "sl_mult", "tp_mult"]}

    # Stage 2: filter params with best strat+RM
    best_sh2 = best_sh
    best_filt = {"rsi_long": df_rsi_long, "rsi_short": df_rsi_short,
                 "vol_regime_thresh": df_vrt, "consec_candles": df_cc}

    for rl in GRID["S1_stage2"]["rsi_long"]:
        for rs in GRID["S1_stage2"]["rsi_short"]:
            for vrt in GRID["S1_stage2"]["vol_regime_thresh"]:
                for cc in GRID["S1_stage2"]["consec_candles"]:
                    sig, z_arr = gen_s1_signals(close, ind,
                                                best_p1["ma_window"], best_p1["z_entry"],
                                                rl, rs, vrt, cc)
                    exit_c = ~np.isnan(z_arr) & (np.abs(z_arr) < best_p1["z_exit"])
                    pos = bt_contrarian(sig, exit_c, close, high, low,
                                        ind["atr14"], best_p1["sl_mult"],
                                        best_p1["tp_mult"], best_p1["max_hold"],
                                        WARMUP, train_end_idx)
                    sh = calc_sharpe(pos, log_ret, train_mask, ann)
                    if sh > best_sh2:
                        best_sh2 = sh
                        best_filt = {"rsi_long": rl, "rsi_short": rs,
                                     "vol_regime_thresh": vrt, "consec_candles": cc}

    # Final run with best params on full data
    all_p = {**best_p1, **best_filt}
    sig, z_arr = gen_s1_signals(close, ind, all_p["ma_window"], all_p["z_entry"],
                                all_p["rsi_long"], all_p["rsi_short"],
                                all_p["vol_regime_thresh"], all_p["consec_candles"])
    exit_c = ~np.isnan(z_arr) & (np.abs(z_arr) < all_p["z_exit"])
    final_pos = bt_contrarian(sig, exit_c, close, high, low, ind["atr14"],
                              all_p["sl_mult"], all_p["tp_mult"], all_p["max_hold"],
                              WARMUP, n)

    # Check if all train combos negative → use defaults
    if best_sh2 < 0:
        all_p = dict(defaults)
        sig, z_arr = gen_s1_signals(close, ind, all_p["ma_window"], all_p["z_entry"],
                                    all_p["rsi_long"], all_p["rsi_short"],
                                    all_p["vol_regime_thresh"], all_p["consec_candles"])
        exit_c = ~np.isnan(z_arr) & (np.abs(z_arr) < all_p["z_exit"])
        final_pos = bt_contrarian(sig, exit_c, close, high, low, ind["atr14"],
                                  all_p["sl_mult"], all_p["tp_mult"], all_p["max_hold"],
                                  WARMUP, n)

    train_m = calc_all_metrics(final_pos, log_ret, train_mask, ann, bpy)
    test_m = calc_all_metrics(final_pos, log_ret, test_mask, ann, bpy)
    return all_p, train_m, test_m, final_pos, log_ret, dts, test_mask


def calibrate_s4(close, high, low, open_arr, volume, dts, ind, is_hourly,
                 daily_trend=None):
    """Calibrate S4_Donchian for one ticker."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    train_mask = (dts >= TRAIN_START) & (dts < TRAIN_END)
    test_mask = dts >= TEST_START
    train_end_idx = int(np.searchsorted(dts, TRAIN_END, side="left"))
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
    bpy = 252 * 9 if is_hourly else 252
    dk = "S4_Donchian_hourly" if is_hourly else "S4_Donchian_daily"
    defaults = V3_DEFAULTS[dk]

    # Stage 1: strategy + RM, default filters
    best_sh = -999.0
    best_p1 = None
    df_adx = defaults["adx_thresh"]
    df_vc = defaults["vol_confirm"]

    for dc_w in GRID["S4_stage1"]["dc_window"]:
        for adx_th_temp in [df_adx]:  # fixed during stage 1
            pass
        sig, ex_l, ex_s = gen_s4_signals(close, high, low, ind, dc_w,
                                          df_adx, df_vc, daily_trend)
        for isl in GRID["S4_stage1"]["initial_sl_mult"]:
            for tn in GRID["S4_stage1"]["trail_n"]:
                rh = ind[f"rh_{tn}"]
                rl = ind[f"rl_{tn}"]
                for tam in GRID["S4_stage1"]["trail_atr_mult"]:
                    for bet in GRID["S4_stage1"]["breakeven_thresh"]:
                        pos = bt_trend(sig, ex_l, ex_s, close, high, low,
                                       ind["atr14"], rh, rl, isl, tam, bet,
                                       WARMUP, train_end_idx)
                        sh = calc_sharpe(pos, log_ret, train_mask, ann)
                        if sh > best_sh:
                            best_sh = sh
                            best_p1 = {"dc_window": dc_w, "initial_sl_mult": isl,
                                        "trail_n": tn, "trail_atr_mult": tam,
                                        "breakeven_thresh": bet}

    if best_p1 is None:
        best_p1 = {k: defaults[k] for k in ["dc_window", "initial_sl_mult",
                                              "trail_n", "trail_atr_mult",
                                              "breakeven_thresh"]}

    # Stage 2: filter params
    best_sh2 = best_sh
    best_filt = {"adx_thresh": df_adx, "vol_confirm": df_vc}

    for adx_th in GRID["S4_stage2"]["adx_thresh"]:
        for vc in GRID["S4_stage2"]["vol_confirm"]:
            sig, ex_l, ex_s = gen_s4_signals(close, high, low, ind,
                                              best_p1["dc_window"], adx_th, vc,
                                              daily_trend)
            rh = ind[f"rh_{best_p1['trail_n']}"]
            rl = ind[f"rl_{best_p1['trail_n']}"]
            pos = bt_trend(sig, ex_l, ex_s, close, high, low, ind["atr14"],
                           rh, rl, best_p1["initial_sl_mult"],
                           best_p1["trail_atr_mult"], best_p1["breakeven_thresh"],
                           WARMUP, train_end_idx)
            sh = calc_sharpe(pos, log_ret, train_mask, ann)
            if sh > best_sh2:
                best_sh2 = sh
                best_filt = {"adx_thresh": adx_th, "vol_confirm": vc}

    # Final
    all_p = {**best_p1, **best_filt}
    if best_sh2 < 0:
        all_p = dict(defaults)
    sig, ex_l, ex_s = gen_s4_signals(close, high, low, ind, all_p["dc_window"],
                                      all_p["adx_thresh"], all_p["vol_confirm"],
                                      daily_trend)
    rh = ind[f"rh_{all_p['trail_n']}"]
    rl = ind[f"rl_{all_p['trail_n']}"]
    final_pos = bt_trend(sig, ex_l, ex_s, close, high, low, ind["atr14"],
                         rh, rl, all_p["initial_sl_mult"],
                         all_p["trail_atr_mult"], all_p["breakeven_thresh"],
                         WARMUP, n)

    train_m = calc_all_metrics(final_pos, log_ret, train_mask, ann, bpy)
    test_m = calc_all_metrics(final_pos, log_ret, test_mask, ann, bpy)
    return all_p, train_m, test_m, final_pos, log_ret, dts, test_mask


def calibrate_s5(close, high, low, open_arr, volume, dts, ind,
                 daily_trend=None):
    """Calibrate S5_Supertrend (hourly) for one ticker."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    train_mask = (dts >= TRAIN_START) & (dts < TRAIN_END)
    test_mask = dts >= TEST_START
    train_end_idx = int(np.searchsorted(dts, TRAIN_END, side="left"))
    ann = np.sqrt(252 * 9)
    bpy = 252 * 9
    defaults = V3_DEFAULTS["S5_Supertrend"]

    best_sh = -999.0
    best_p1 = None
    df_adx = defaults["adx_thresh"]

    for atr_p in GRID["S5_stage1"]["atr_period"]:
        for mult in GRID["S5_stage1"]["multiplier"]:
            sig, ex_l, ex_s = gen_s5_signals(close, high, low, ind, atr_p, mult,
                                              df_adx, daily_trend)
            for isl in GRID["S5_stage1"]["initial_sl_mult"]:
                for tn in GRID["S5_stage1"]["trail_n"]:
                    rh = ind[f"rh_{tn}"]
                    rl = ind[f"rl_{tn}"]
                    for tam in GRID["S5_stage1"]["trail_atr_mult"]:
                        for bet in GRID["S5_stage1"]["breakeven_thresh"]:
                            pos = bt_trend(sig, ex_l, ex_s, close, high, low,
                                           ind["atr14"], rh, rl, isl, tam, bet,
                                           WARMUP, train_end_idx)
                            sh = calc_sharpe(pos, log_ret, train_mask, ann)
                            if sh > best_sh:
                                best_sh = sh
                                best_p1 = {"atr_period": atr_p, "multiplier": mult,
                                            "initial_sl_mult": isl, "trail_n": tn,
                                            "trail_atr_mult": tam, "breakeven_thresh": bet}

    if best_p1 is None:
        best_p1 = {k: defaults[k] for k in ["atr_period", "multiplier",
                                              "initial_sl_mult", "trail_n",
                                              "trail_atr_mult", "breakeven_thresh"]}

    best_sh2 = best_sh
    best_filt = {"adx_thresh": df_adx}

    for adx_th in GRID["S5_stage2"]["adx_thresh"]:
        sig, ex_l, ex_s = gen_s5_signals(close, high, low, ind,
                                          best_p1["atr_period"], best_p1["multiplier"],
                                          adx_th, daily_trend)
        rh = ind[f"rh_{best_p1['trail_n']}"]
        rl = ind[f"rl_{best_p1['trail_n']}"]
        pos = bt_trend(sig, ex_l, ex_s, close, high, low, ind["atr14"],
                       rh, rl, best_p1["initial_sl_mult"],
                       best_p1["trail_atr_mult"], best_p1["breakeven_thresh"],
                       WARMUP, train_end_idx)
        sh = calc_sharpe(pos, log_ret, train_mask, ann)
        if sh > best_sh2:
            best_sh2 = sh
            best_filt = {"adx_thresh": adx_th}

    all_p = {**best_p1, **best_filt}
    if best_sh2 < 0:
        all_p = dict(defaults)
    sig, ex_l, ex_s = gen_s5_signals(close, high, low, ind,
                                      all_p["atr_period"], all_p["multiplier"],
                                      all_p["adx_thresh"], daily_trend)
    rh = ind[f"rh_{all_p['trail_n']}"]
    rl = ind[f"rl_{all_p['trail_n']}"]
    final_pos = bt_trend(sig, ex_l, ex_s, close, high, low, ind["atr14"],
                         rh, rl, all_p["initial_sl_mult"],
                         all_p["trail_atr_mult"], all_p["breakeven_thresh"],
                         WARMUP, n)

    train_m = calc_all_metrics(final_pos, log_ret, train_mask, ann, bpy)
    test_m = calc_all_metrics(final_pos, log_ret, test_mask, ann, bpy)
    return all_p, train_m, test_m, final_pos, log_ret, dts, test_mask


def calibrate_s6(close, high, low, open_arr, volume, dts, ind):
    """Calibrate S6_DualMA (hourly) for one ticker."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    train_mask = (dts >= TRAIN_START) & (dts < TRAIN_END)
    test_mask = dts >= TEST_START
    train_end_idx = int(np.searchsorted(dts, TRAIN_END, side="left"))
    ann = np.sqrt(252 * 9)
    bpy = 252 * 9
    defaults = V3_DEFAULTS["S6_DualMA"]

    best_sh = -999.0
    best_p1 = None
    df_adx = defaults["adx_thresh"]

    for fw in GRID["S6_stage1"]["fast_window"]:
        for sw in GRID["S6_stage1"]["slow_window"]:
            sig, ex_l, ex_s = gen_s6_signals(close, ind, fw, sw, df_adx)
            for isl in GRID["S6_stage1"]["initial_sl_mult"]:
                for tn in GRID["S6_stage1"]["trail_n"]:
                    rh = ind[f"rh_{tn}"]
                    rl = ind[f"rl_{tn}"]
                    for tam in GRID["S6_stage1"]["trail_atr_mult"]:
                        for bet in GRID["S6_stage1"]["breakeven_thresh"]:
                            pos = bt_trend(sig, ex_l, ex_s, close, high, low,
                                           ind["atr14"], rh, rl, isl, tam, bet,
                                           WARMUP, train_end_idx)
                            sh = calc_sharpe(pos, log_ret, train_mask, ann)
                            if sh > best_sh:
                                best_sh = sh
                                best_p1 = {"fast_window": fw, "slow_window": sw,
                                            "initial_sl_mult": isl, "trail_n": tn,
                                            "trail_atr_mult": tam, "breakeven_thresh": bet}

    if best_p1 is None:
        best_p1 = {k: defaults[k] for k in ["fast_window", "slow_window",
                                              "initial_sl_mult", "trail_n",
                                              "trail_atr_mult", "breakeven_thresh"]}

    best_sh2 = best_sh
    best_filt = {"adx_thresh": df_adx}

    for adx_th in GRID["S6_stage2"]["adx_thresh"]:
        sig, ex_l, ex_s = gen_s6_signals(close, ind,
                                          best_p1["fast_window"], best_p1["slow_window"],
                                          adx_th)
        rh = ind[f"rh_{best_p1['trail_n']}"]
        rl = ind[f"rl_{best_p1['trail_n']}"]
        pos = bt_trend(sig, ex_l, ex_s, close, high, low, ind["atr14"],
                       rh, rl, best_p1["initial_sl_mult"],
                       best_p1["trail_atr_mult"], best_p1["breakeven_thresh"],
                       WARMUP, train_end_idx)
        sh = calc_sharpe(pos, log_ret, train_mask, ann)
        if sh > best_sh2:
            best_sh2 = sh
            best_filt = {"adx_thresh": adx_th}

    all_p = {**best_p1, **best_filt}
    if best_sh2 < 0:
        all_p = dict(defaults)
    sig, ex_l, ex_s = gen_s6_signals(close, ind,
                                      all_p["fast_window"], all_p["slow_window"],
                                      all_p["adx_thresh"])
    rh = ind[f"rh_{all_p['trail_n']}"]
    rl = ind[f"rl_{all_p['trail_n']}"]
    final_pos = bt_trend(sig, ex_l, ex_s, close, high, low, ind["atr14"],
                         rh, rl, all_p["initial_sl_mult"],
                         all_p["trail_atr_mult"], all_p["breakeven_thresh"],
                         WARMUP, n)

    train_m = calc_all_metrics(final_pos, log_ret, train_mask, ann, bpy)
    test_m = calc_all_metrics(final_pos, log_ret, test_mask, ann, bpy)
    return all_p, train_m, test_m, final_pos, log_ret, dts, test_mask


# ════════════════════════════════════════════════════════════
# Daily lookup for multi-TF alignment (S4h, S5h)
# ════════════════════════════════════════════════════════════

def build_daily_trend(daily, ticker):
    tdf = daily[daily["ticker"] == ticker].sort_values("date")
    close_d = tdf["close"].values
    dates_d = tdf["date"].values
    ma50 = calc_sma(close_d, 50)
    ma200 = calc_sma(close_d, 200)
    above = np.full(len(close_d), np.nan)
    valid = ~np.isnan(ma50) & ~np.isnan(ma200)
    above[valid] = (ma50[valid] > ma200[valid]).astype(float)
    return np.array(dates_d, dtype="datetime64[D]"), above


def align_daily_to_hourly(d_dates, d_above, h_dts):
    n = len(h_dts)
    h_dates = np.array(h_dts, dtype="datetime64[D]")
    indices = np.searchsorted(d_dates, h_dates, side="left") - 1
    valid = indices >= 0
    result = np.full(n, np.nan)
    result[valid] = d_above[indices[valid]]
    return result


# ════════════════════════════════════════════════════════════
# Main orchestration
# ════════════════════════════════════════════════════════════

STRATEGY_CONFIGS = [
    ("S1_MA_Reversion", "daily"),
    ("S4_Donchian", "daily"),
    ("S4_Donchian", "hourly"),
    ("S5_Supertrend", "hourly"),
    ("S6_DualMA", "hourly"),
]


def process_ticker(ticker, daily, hourly):
    """Calibrate all 5 strategies for one ticker. Returns results dict."""
    results = {}

    # Daily data
    tdf_d = daily[daily["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    close_d = tdf_d["close"].values.astype(np.float64)
    high_d = tdf_d["high"].values.astype(np.float64)
    low_d = tdf_d["low"].values.astype(np.float64)
    open_d = tdf_d["open"].values.astype(np.float64)
    vol_d = tdf_d["volume"].values.astype(np.float64)
    dts_d = tdf_d["date"].values

    # Hourly data
    tdf_h = hourly[hourly["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
    close_h = tdf_h["close"].values.astype(np.float64)
    high_h = tdf_h["high"].values.astype(np.float64)
    low_h = tdf_h["low"].values.astype(np.float64)
    open_h = tdf_h["open"].values.astype(np.float64)
    vol_h = tdf_h["volume"].values.astype(np.float64)
    dts_h = tdf_h["datetime"].values

    # Base indicators
    ind_d = compute_base(close_d, high_d, low_d, open_d, vol_d)
    ind_h = compute_base(close_h, high_h, low_h, open_h, vol_h)

    # Daily trend for multi-TF
    d_dates, d_above = build_daily_trend(daily, ticker)
    daily_trend_h = align_daily_to_hourly(d_dates, d_above, dts_h)

    # S1_MA_Reversion daily
    res = calibrate_s1(close_d, high_d, low_d, open_d, vol_d, dts_d, ind_d)
    results[("S1_MA_Reversion", "daily")] = res

    # S4_Donchian daily
    res = calibrate_s4(close_d, high_d, low_d, open_d, vol_d, dts_d, ind_d,
                       is_hourly=False, daily_trend=None)
    results[("S4_Donchian", "daily")] = res

    # S4_Donchian hourly
    res = calibrate_s4(close_h, high_h, low_h, open_h, vol_h, dts_h, ind_h,
                       is_hourly=True, daily_trend=daily_trend_h)
    results[("S4_Donchian", "hourly")] = res

    # S5_Supertrend hourly
    res = calibrate_s5(close_h, high_h, low_h, open_h, vol_h, dts_h, ind_h,
                       daily_trend=daily_trend_h)
    results[("S5_Supertrend", "hourly")] = res

    # S6_DualMA hourly
    res = calibrate_s6(close_h, high_h, low_h, open_h, vol_h, dts_h, ind_h)
    results[("S6_DualMA", "hourly")] = res

    return results


def load_data():
    daily = pd.read_parquet(DATA_DIR / "ohlcv_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.read_parquet(DATA_DIR / "ohlcv_hourly.parquet")
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    return daily, hourly


PARAM_KEYS = {
    ("S1_MA_Reversion", "daily"): ["ma_window", "z_entry", "z_exit", "max_hold",
                                    "sl_mult", "tp_mult", "rsi_long", "rsi_short",
                                    "vol_regime_thresh", "consec_candles"],
    ("S4_Donchian", "daily"): ["dc_window", "initial_sl_mult", "trail_n",
                                "trail_atr_mult", "breakeven_thresh", "adx_thresh", "vol_confirm"],
    ("S4_Donchian", "hourly"): ["dc_window", "initial_sl_mult", "trail_n",
                                 "trail_atr_mult", "breakeven_thresh", "adx_thresh", "vol_confirm"],
    ("S5_Supertrend", "hourly"): ["atr_period", "multiplier", "initial_sl_mult", "trail_n",
                                   "trail_atr_mult", "breakeven_thresh", "adx_thresh"],
    ("S6_DualMA", "hourly"): ["fast_window", "slow_window", "initial_sl_mult", "trail_n",
                               "trail_atr_mult", "breakeven_thresh", "adx_thresh"],
}


def run_v3_defaults_test(daily, hourly):
    """Run v3 default params on TEST period (2022+) for fair comparison."""
    print("\n  Computing v3 default Sharpe on test period (2022+)...")
    result = {}
    for ticker in TICKERS:
        # Daily
        tdf_d = daily[daily["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        c_d = tdf_d["close"].values.astype(np.float64)
        h_d = tdf_d["high"].values.astype(np.float64)
        l_d = tdf_d["low"].values.astype(np.float64)
        o_d = tdf_d["open"].values.astype(np.float64)
        v_d = tdf_d["volume"].values.astype(np.float64)
        dts_d = tdf_d["date"].values
        ind_d = compute_base(c_d, h_d, l_d, o_d, v_d)
        n_d = len(c_d)
        lr_d = np.zeros(n_d)
        lr_d[:-1] = np.log(c_d[1:] / np.maximum(c_d[:-1], 1e-12))
        test_mask_d = dts_d >= TEST_START

        # Hourly
        tdf_h = hourly[hourly["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
        c_h = tdf_h["close"].values.astype(np.float64)
        h_h = tdf_h["high"].values.astype(np.float64)
        l_h = tdf_h["low"].values.astype(np.float64)
        o_h = tdf_h["open"].values.astype(np.float64)
        v_h = tdf_h["volume"].values.astype(np.float64)
        dts_h = tdf_h["datetime"].values
        ind_h = compute_base(c_h, h_h, l_h, o_h, v_h)
        n_h = len(c_h)
        lr_h = np.zeros(n_h)
        lr_h[:-1] = np.log(c_h[1:] / np.maximum(c_h[:-1], 1e-12))
        test_mask_h = dts_h >= TEST_START

        d_dates, d_above = build_daily_trend(daily, ticker)
        daily_trend_h = align_daily_to_hourly(d_dates, d_above, dts_h)

        # S1 daily
        d = V3_DEFAULTS["S1_MA_Reversion"]
        sig, z_arr = gen_s1_signals(c_d, ind_d, d["ma_window"], d["z_entry"],
                                    d["rsi_long"], d["rsi_short"], d["vol_regime_thresh"],
                                    d["consec_candles"])
        exit_c = ~np.isnan(z_arr) & (np.abs(z_arr) < d["z_exit"])
        pos = bt_contrarian(sig, exit_c, c_d, h_d, l_d, ind_d["atr14"],
                            d["sl_mult"], d["tp_mult"], d["max_hold"], WARMUP, n_d)
        sh = calc_sharpe(pos, lr_d, test_mask_d, np.sqrt(252))
        result.setdefault(("S1_MA_Reversion", "daily"), []).append(sh)

        # S4 daily
        d = V3_DEFAULTS["S4_Donchian_daily"]
        sig, ex_l, ex_s = gen_s4_signals(c_d, h_d, l_d, ind_d, d["dc_window"],
                                          d["adx_thresh"], d["vol_confirm"], None)
        rh = ind_d[f"rh_{d['trail_n']}"]
        rl = ind_d[f"rl_{d['trail_n']}"]
        pos = bt_trend(sig, ex_l, ex_s, c_d, h_d, l_d, ind_d["atr14"], rh, rl,
                       d["initial_sl_mult"], d["trail_atr_mult"], d["breakeven_thresh"],
                       WARMUP, n_d)
        sh = calc_sharpe(pos, lr_d, test_mask_d, np.sqrt(252))
        result.setdefault(("S4_Donchian", "daily"), []).append(sh)

        # S4 hourly
        d = V3_DEFAULTS["S4_Donchian_hourly"]
        sig, ex_l, ex_s = gen_s4_signals(c_h, h_h, l_h, ind_h, d["dc_window"],
                                          d["adx_thresh"], d["vol_confirm"], daily_trend_h)
        rh = ind_h[f"rh_{d['trail_n']}"]
        rl = ind_h[f"rl_{d['trail_n']}"]
        pos = bt_trend(sig, ex_l, ex_s, c_h, h_h, l_h, ind_h["atr14"], rh, rl,
                       d["initial_sl_mult"], d["trail_atr_mult"], d["breakeven_thresh"],
                       WARMUP, n_h)
        sh = calc_sharpe(pos, lr_h, test_mask_h, np.sqrt(252 * 9))
        result.setdefault(("S4_Donchian", "hourly"), []).append(sh)

        # S5 hourly
        d = V3_DEFAULTS["S5_Supertrend"]
        sig, ex_l, ex_s = gen_s5_signals(c_h, h_h, l_h, ind_h, d["atr_period"],
                                          d["multiplier"], d["adx_thresh"], daily_trend_h)
        rh = ind_h[f"rh_{d['trail_n']}"]
        rl = ind_h[f"rl_{d['trail_n']}"]
        pos = bt_trend(sig, ex_l, ex_s, c_h, h_h, l_h, ind_h["atr14"], rh, rl,
                       d["initial_sl_mult"], d["trail_atr_mult"], d["breakeven_thresh"],
                       WARMUP, n_h)
        sh = calc_sharpe(pos, lr_h, test_mask_h, np.sqrt(252 * 9))
        result.setdefault(("S5_Supertrend", "hourly"), []).append(sh)

        # S6 hourly
        d = V3_DEFAULTS["S6_DualMA"]
        sig, ex_l, ex_s = gen_s6_signals(c_h, ind_h, d["fast_window"],
                                          d["slow_window"], d["adx_thresh"])
        rh = ind_h[f"rh_{d['trail_n']}"]
        rl = ind_h[f"rl_{d['trail_n']}"]
        pos = bt_trend(sig, ex_l, ex_s, c_h, h_h, l_h, ind_h["atr14"], rh, rl,
                       d["initial_sl_mult"], d["trail_atr_mult"], d["breakeven_thresh"],
                       WARMUP, n_h)
        sh = calc_sharpe(pos, lr_h, test_mask_h, np.sqrt(252 * 9))
        result.setdefault(("S6_DualMA", "hourly"), []).append(sh)

    # Compute medians
    medians = {}
    for k, v in result.items():
        medians[k] = float(np.median(v))
    return medians


def main():
    t0 = time.time()
    print("=" * 70)
    print("Per-Ticker Calibration: 5 strategies × 17 tickers")
    print("  Train: 2020-2021 | Test: 2022-2026")
    print("  Two-stage grid search: strategy+RM → filters")
    print("=" * 70)

    daily, hourly = load_data()
    print(f"Loaded daily: {len(daily):,} rows, hourly: {len(hourly):,} rows")

    # Process all tickers
    all_results = {}
    for i, ticker in enumerate(TICKERS):
        t1 = time.time()
        print(f"\n  [{i+1}/17] {ticker}...", end="", flush=True)
        results = process_ticker(ticker, daily, hourly)
        all_results[ticker] = results
        elapsed = time.time() - t1
        print(f" done ({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t0
    print(f"\n  Total calibration time: {total_time:.0f}s")

    # ── Collect outputs ──
    print("\nCollecting results...")
    params_rows = []
    train_rows = []
    test_rows = []
    signal_records = []

    for ticker in TICKERS:
        for (sname, tf) in STRATEGY_CONFIGS:
            all_p, train_m, test_m, final_pos, log_ret, dts, test_mask = \
                all_results[ticker][(sname, tf)]

            # Params
            params_rows.append({
                "strategy": sname, "timeframe": tf, "ticker": ticker,
                **{f"p_{k}": v for k, v in all_p.items()},
                "sharpe_train": train_m.get("sharpe", 0),
                "sharpe_test": test_m.get("sharpe", 0),
            })

            # Train metrics
            train_rows.append({
                "strategy": sname, "timeframe": tf, "ticker": ticker,
                **train_m,
            })

            # Test metrics
            test_rows.append({
                "strategy": sname, "timeframe": tf, "ticker": ticker,
                **test_m,
            })

            # Signals (test period only)
            dt_col = "datetime" if tf == "hourly" else "date"
            t_dts = dts[test_mask]
            t_pos = final_pos[test_mask]
            t_ret = (final_pos * log_ret)[test_mask]
            # Generate signal from position changes
            t_sig = np.zeros(len(t_pos), dtype=np.int8)
            for j in range(len(t_pos)):
                if j == 0:
                    if t_pos[j] != 0:
                        t_sig[j] = int(t_pos[j])
                else:
                    if t_pos[j] != 0 and t_pos[j - 1] == 0:
                        t_sig[j] = int(t_pos[j])
                    elif t_pos[j] == 0 and t_pos[j - 1] != 0:
                        t_sig[j] = 0

            for j in range(len(t_dts)):
                signal_records.append({
                    "datetime": t_dts[j], "ticker": ticker,
                    "strategy": sname, "timeframe": tf,
                    "signal": int(t_sig[j]), "position": t_pos[j],
                    "return": t_ret[j],
                })

    # ── Save outputs ──
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    # 1. Optimal params
    params_df = pd.DataFrame(params_rows)
    p = OUT_TABLES / "calibration_optimal_params.csv"
    params_df.to_csv(p, index=False)
    print(f"  1. {p.name}: {len(params_df)} rows")

    # 2. Train results
    train_df = pd.DataFrame(train_rows)
    p = OUT_TABLES / "calibration_train_results.csv"
    train_df.to_csv(p, index=False)
    print(f"  2. {p.name}: {len(train_df)} rows")

    # 3. Test results
    test_df = pd.DataFrame(test_rows)
    p = OUT_TABLES / "calibration_test_results.csv"
    test_df.to_csv(p, index=False)
    print(f"  3. {p.name}: {len(test_df)} rows")

    # 4. Summary — compute v3 defaults on same test period for fair comparison
    v3_defaults = run_v3_defaults_test(daily, hourly)
    summary_rows = []
    for (sname, tf) in STRATEGY_CONFIGS:
        tr_sub = train_df[(train_df["strategy"] == sname) & (train_df["timeframe"] == tf)]
        te_sub = test_df[(test_df["strategy"] == sname) & (test_df["timeframe"] == tf)]
        sh_train_med = tr_sub["sharpe"].median() if "sharpe" in tr_sub.columns else 0
        sh_test_med = te_sub["sharpe"].median() if "sharpe" in te_sub.columns else 0
        v3_default = v3_defaults.get((sname, tf), np.nan)
        delta_pct = ((sh_test_med - v3_default) / abs(v3_default) * 100
                     if not np.isnan(v3_default) and abs(v3_default) > 1e-6 else np.nan)
        n_pos_train = int((tr_sub["sharpe"] > 0).sum()) if "sharpe" in tr_sub.columns else 0
        n_pos_test = int((te_sub["sharpe"] > 0).sum()) if "sharpe" in te_sub.columns else 0
        summary_rows.append({
            "strategy": sname, "timeframe": tf,
            "sharpe_train_median": round(sh_train_med, 4),
            "sharpe_test_median": round(sh_test_med, 4),
            "sharpe_v3default_test_median": round(v3_default, 4) if not np.isnan(v3_default) else np.nan,
            "delta_vs_default_pct": round(delta_pct, 1) if not np.isnan(delta_pct) else np.nan,
            "n_positive_train": n_pos_train,
            "n_positive_test": n_pos_test,
        })

    summary_df = pd.DataFrame(summary_rows)
    p = OUT_TABLES / "calibration_summary.csv"
    summary_df.to_csv(p, index=False)
    print(f"  4. {p.name}: {len(summary_df)} rows")

    # 5. Param distribution (only relevant params per strategy)
    param_dist_rows = []
    for (sname, tf) in STRATEGY_CONFIGS:
        sub = params_df[(params_df["strategy"] == sname) & (params_df["timeframe"] == tf)]
        relevant = PARAM_KEYS.get((sname, tf), [])
        for pname in relevant:
            pc = f"p_{pname}"
            if pc not in sub.columns:
                continue
            for val, cnt in sub[pc].value_counts().items():
                if pd.notna(val):
                    param_dist_rows.append({
                        "strategy": sname, "timeframe": tf,
                        "param_name": pname, "param_value": val, "count": int(cnt),
                    })

    param_dist_df = pd.DataFrame(param_dist_rows)
    p = OUT_TABLES / "calibration_param_distribution.csv"
    param_dist_df.to_csv(p, index=False)
    print(f"  5. {p.name}: {len(param_dist_df)} rows")

    # 6. Signals parquet (test period)
    signals_df = pd.DataFrame(signal_records)
    signals_df["datetime"] = pd.to_datetime(signals_df["datetime"])
    p = OUT_DATA / "signals_calibrated_test.parquet"
    signals_df.to_parquet(p, index=False)
    print(f"  6. {p.name}: {len(signals_df):,} rows")

    # ── Print summary ──
    print("\n" + "=" * 90)
    print("CALIBRATION SUMMARY")
    print("=" * 90)
    print(f"{'Strategy':<20} {'TF':<7} {'Sh_train':>9} {'Sh_test':>9} "
          f"{'Sh_v3def':>9} {'Δ%':>7} {'pos_tr':>6} {'pos_te':>6}")
    print("-" * 90)
    for _, r in summary_df.iterrows():
        v3d = r['sharpe_v3default_test_median']
        v3d_s = f"{v3d:>8.4f}" if not np.isnan(v3d) else "     N/A"
        dp = r['delta_vs_default_pct']
        dp_s = f"{dp:>+6.1f}%" if not np.isnan(dp) else "    N/A"
        print(f"{r['strategy']:<20} {r['timeframe']:<7} "
              f"{r['sharpe_train_median']:>+8.4f} {r['sharpe_test_median']:>+8.4f} "
              f"{v3d_s} {dp_s} "
              f"{r['n_positive_train']:>5}/17 {r['n_positive_test']:>5}/17")

    # Overfitting check
    print("\n" + "-" * 60)
    print("OVERFITTING CHECK:")
    all_sh_train = params_df["sharpe_train"].values
    all_sh_test = params_df["sharpe_test"].values
    valid = ~np.isnan(all_sh_train) & ~np.isnan(all_sh_test)
    if valid.sum() > 2:
        corr = np.corrcoef(all_sh_train[valid], all_sh_test[valid])[0, 1]
        print(f"  Train-Test Sharpe correlation: {corr:.3f}")
        if corr < 0.3:
            print("  ⚠ WARNING: Low correlation — possible overfitting!")
        elif corr > 0.6:
            print("  ✓ Good correlation — calibration generalizes well")
    mean_train = np.nanmean(all_sh_train)
    mean_test = np.nanmean(all_sh_test)
    print(f"  Mean Sharpe train: {mean_train:+.4f}, test: {mean_test:+.4f}")
    if mean_test < mean_train * 0.5 and mean_train > 0:
        print("  ⚠ WARNING: Test Sharpe << Train Sharpe — overfitting risk!")

    # Param modes (only relevant params per strategy)
    print("\n" + "-" * 60)
    print("MOST POPULAR PARAMS (mode across 17 tickers):")
    for (sname, tf) in STRATEGY_CONFIGS:
        sub = params_df[(params_df["strategy"] == sname) & (params_df["timeframe"] == tf)]
        relevant = PARAM_KEYS.get((sname, tf), [])
        modes = []
        for pname in relevant:
            pc = f"p_{pname}"
            if pc not in sub.columns:
                continue
            vals = sub[pc].dropna()
            if len(vals) == 0:
                continue
            mode_val = vals.mode().values[0]
            cnt = int((vals == mode_val).sum())
            modes.append(f"{pname}={mode_val}({cnt})")
        print(f"  {sname} [{tf}]: {', '.join(modes)}")

    # Delta vs default
    print("\n" + "-" * 60)
    for _, r in summary_df.iterrows():
        v3d = r['sharpe_v3default_test_median']
        test_sh = r['sharpe_test_median']
        if not np.isnan(v3d):
            if test_sh > v3d:
                print(f"  ✓ {r['strategy']} [{r['timeframe']}]: "
                      f"calibration helps ({test_sh:+.4f} > {v3d:+.4f})")
            else:
                print(f"  ✗ {r['strategy']} [{r['timeframe']}]: "
                      f"calibration doesn't help ({test_sh:+.4f} <= {v3d:+.4f})")

    print(f"\n{'=' * 70}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print("DONE")


if __name__ == "__main__":
    main()
