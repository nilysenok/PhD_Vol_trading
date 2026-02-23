#!/usr/bin/env python3
"""
strategies_rolling_calib_v2.py — Rolling recalibration V2

Expanded parameter grids, new filters, multiprocessing + numba.
Changes vs V1:
  - N_SAMPLES=5000, META_GRID reduced to 4
  - New filters: ma200, mean_revert_speed, close_vs_bb_pct, squeeze_then_expand,
    momentum, atr_expansion, intraday_range, range_bound_check, vwap_slope_flat
  - Pivot types: woodie, camarilla added
  - S5 target_exit: pivot / mid_SR / opposite_SR
  - multiprocessing.Pool + numba @njit on backtest engines
"""

import warnings
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
from collections import Counter

warnings.filterwarnings("ignore")

# ── numba (optional) ──
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    print("WARNING: numba not installed, backtests will run without JIT")

# ════════════════════════════════════════════════════════════
# Constants and paths
# ════════════════════════════════════════════════════════════

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
N_SAMPLES = 5000
SEED = 42
N_WORKERS = min(cpu_count(), 8)

META_GRID = [(63, 504), (126, 504), (252, 504), (126, 252)]

STRATEGY_IDS = ["S1", "S2", "S3", "S4", "S5", "S6"]
TIMEFRAMES = ["daily", "hourly"]

STRATEGY_NAMES = {
    "S1": "S1_MeanRev", "S2": "S2_Bollinger", "S3": "S3_Donchian",
    "S4": "S4_Supertrend", "S5": "S5_PivotPoints", "S6": "S6_VWAP",
}
CATEGORY = {
    "S1": "Contrarian", "S2": "Contrarian",
    "S3": "Trend", "S4": "Trend",
    "S5": "Range", "S6": "Range",
}

# V1 results for comparison
V1_SHARPES = {
    ("S1", "daily"): 0.0935, ("S1", "hourly"): -0.1023,
    ("S2", "daily"): 0.0785, ("S2", "hourly"): -0.0246,
    ("S3", "daily"): 0.3013, ("S3", "hourly"): 0.4789,
    ("S4", "daily"): 0.4056, ("S4", "hourly"): 0.2576,
    ("S5", "daily"): -0.1779, ("S5", "hourly"): -0.2455,
    ("S6", "daily"): -0.2106, ("S6", "hourly"): -0.0031,
}

# ════════════════════════════════════════════════════════════
# V3 defaults (fallback if all samples negative)
# ════════════════════════════════════════════════════════════

V3_DEFAULTS = {
    ("S1", "daily"): dict(ma_window=20, z_entry=2.0, z_exit=0.5, max_hold=20,
                          sl_mult=1.5, tp_mult=2.0, rsi_thresh=40,
                          vol_regime=1.2, consec_candles=3, vol_exhaustion=True,
                          ma200_filter=None, mean_revert_speed=None),
    ("S1", "hourly"): dict(ma_window=20, z_entry=2.0, z_exit=0.5, max_hold=40,
                           sl_mult=1.5, tp_mult=2.0, rsi_thresh=40,
                           vol_regime=1.2, consec_candles=3, vol_exhaustion=True,
                           ma200_filter=None, mean_revert_speed=None),
    ("S2", "daily"): dict(bb_window=20, bb_std=2.0, max_hold=15,
                          sl_mult=1.5, tp_mult=2.0, bw_percentile=0.25,
                          vol_regime=1.2, consec_candles=3,
                          vol_exhaustion=True, rsi_divergence=True,
                          close_vs_bb_pct=None, squeeze_then_expand=None),
    ("S2", "hourly"): dict(bb_window=20, bb_std=2.0, max_hold=30,
                           sl_mult=1.5, tp_mult=2.0, bw_percentile=0.25,
                           vol_regime=1.2, consec_candles=3,
                           vol_exhaustion=True, rsi_divergence=True,
                           close_vs_bb_pct=None, squeeze_then_expand=None),
    ("S3", "daily"): dict(dc_window=20, initial_sl_mult=2.5, trail_n=10,
                          trail_atr_mult=2.5, breakeven_thresh=1.5,
                          adx_thresh=20, vol_confirm=1.0,
                          momentum_filter=None, atr_expansion=None),
    ("S3", "hourly"): dict(dc_window=20, initial_sl_mult=2.5, trail_n=20,
                           trail_atr_mult=2.5, breakeven_thresh=1.5,
                           adx_thresh=20, vol_confirm=1.0, daily_ma_confirm=True,
                           momentum_filter=None, atr_expansion=None),
    ("S4", "daily"): dict(atr_period=14, multiplier=3.0, initial_sl_mult=2.5,
                          trail_n=10, trail_atr_mult=2.5, breakeven_thresh=1.5,
                          adx_thresh=20,
                          momentum_filter=None, atr_expansion=None),
    ("S4", "hourly"): dict(atr_period=14, multiplier=3.0, initial_sl_mult=2.5,
                           trail_n=20, trail_atr_mult=2.5, breakeven_thresh=1.5,
                           adx_thresh=20, daily_ma_confirm=True,
                           momentum_filter=None, atr_expansion=None),
    ("S5", "daily"): dict(pivot_type="classic", target_exit="pivot",
                          max_hold=10, sl_mult=1.5, tp_mult=1.5,
                          adx_exit_thresh=30, adx_entry_thresh=20,
                          bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                          vol_compression=0.9,
                          intraday_range=None, range_bound_check=None),
    ("S5", "hourly"): dict(pivot_type="classic", target_exit="pivot",
                           max_hold=20, sl_mult=1.5, tp_mult=1.5,
                           adx_exit_thresh=30, adx_entry_thresh=20,
                           bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                           vol_compression=0.9,
                           intraday_range=None, range_bound_check=None),
    ("S6", "daily"): dict(vwap_window=20, dev_mult=2.0, exit_mult=0.5,
                          max_hold=10, sl_mult=1.5, tp_mult=1.5,
                          adx_exit_thresh=30, adx_entry_thresh=20,
                          bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                          vol_compression=0.9, hurst_proxy=True,
                          intraday_range=None, range_bound_check=None,
                          vwap_slope_flat=None),
    ("S6", "hourly"): dict(vwap_window=20, dev_mult=2.0, exit_mult=0.5,
                           max_hold=20, sl_mult=1.5, tp_mult=1.5,
                           adx_exit_thresh=30, adx_entry_thresh=20,
                           bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                           vol_compression=0.9, hurst_proxy=True,
                           intraday_range=None, range_bound_check=None,
                           vwap_slope_flat=None),
}

# ════════════════════════════════════════════════════════════
# Parameter spaces (expanded)
# ════════════════════════════════════════════════════════════

PARAM_SPACE = {
    ("S1", "daily"): dict(
        ma_window=[10, 15, 20, 25, 30], z_entry=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        z_exit=[0.3, 0.5, 0.7], max_hold=[10, 15, 20, 25],
        sl_mult=[1.0, 1.5, 2.0, 2.5, 3.0], tp_mult=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        rsi_thresh=[30, 35, 40, 45, None], vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
        ma200_filter=[True, None], mean_revert_speed=[True, None],
    ),
    ("S1", "hourly"): dict(
        ma_window=[10, 15, 20, 25, 30], z_entry=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        z_exit=[0.3, 0.5, 0.7], max_hold=[20, 30, 40, 50],
        sl_mult=[1.0, 1.5, 2.0, 2.5, 3.0], tp_mult=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        rsi_thresh=[30, 35, 40, 45, None], vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
        ma200_filter=[True, None], mean_revert_speed=[True, None],
    ),
    ("S2", "daily"): dict(
        bb_window=[10, 15, 20, 25, 30], bb_std=[1.5, 1.75, 2.0, 2.25, 2.5],
        max_hold=[10, 15, 20], sl_mult=[1.0, 1.5, 2.0, 2.5, 3.0],
        tp_mult=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        bw_percentile=[0.15, 0.20, 0.25, 0.33, 0.50, None],
        vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
        rsi_divergence=[True, None],
        close_vs_bb_pct=[True, None], squeeze_then_expand=[True, None],
    ),
    ("S2", "hourly"): dict(
        bb_window=[10, 15, 20, 25, 30], bb_std=[1.5, 1.75, 2.0, 2.25, 2.5],
        max_hold=[20, 30, 40], sl_mult=[1.0, 1.5, 2.0, 2.5, 3.0],
        tp_mult=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        bw_percentile=[0.15, 0.20, 0.25, 0.33, 0.50, None],
        vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
        rsi_divergence=[True, None],
        close_vs_bb_pct=[True, None], squeeze_then_expand=[True, None],
    ),
    ("S3", "daily"): dict(
        dc_window=[5, 10, 15, 20, 25, 30, 40], initial_sl_mult=[2.0, 2.5, 3.0],
        trail_n=[3, 5, 10, 15, 20, 25, 30], trail_atr_mult=[2.0, 2.5, 3.0],
        breakeven_thresh=[0.5, 1.0, 1.5, 2.0, None],
        adx_thresh=[15, 20, 25, None], vol_confirm=[0.8, 1.0, 1.2, None],
        momentum_filter=[True, None], atr_expansion=[True, None],
    ),
    ("S3", "hourly"): dict(
        dc_window=[5, 10, 15, 20, 25, 30, 40], initial_sl_mult=[2.0, 2.5, 3.0],
        trail_n=[3, 5, 10, 15, 20, 25, 30], trail_atr_mult=[2.0, 2.5, 3.0],
        breakeven_thresh=[0.5, 1.0, 1.5, 2.0, None],
        adx_thresh=[15, 20, 25, None], vol_confirm=[0.8, 1.0, 1.2, None],
        daily_ma_confirm=[True, None],
        momentum_filter=[True, None], atr_expansion=[True, None],
    ),
    ("S4", "daily"): dict(
        atr_period=[7, 10, 14, 20, 25], multiplier=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        initial_sl_mult=[2.0, 2.5, 3.0], trail_n=[3, 5, 10, 15, 20, 25, 30],
        trail_atr_mult=[2.0, 2.5, 3.0], breakeven_thresh=[0.5, 1.0, 1.5, 2.0, None],
        adx_thresh=[15, 20, 25, None],
        momentum_filter=[True, None], atr_expansion=[True, None],
    ),
    ("S4", "hourly"): dict(
        atr_period=[7, 10, 14, 20, 25], multiplier=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        initial_sl_mult=[2.0, 2.5, 3.0], trail_n=[3, 5, 10, 15, 20, 25, 30],
        trail_atr_mult=[2.0, 2.5, 3.0], breakeven_thresh=[0.5, 1.0, 1.5, 2.0, None],
        adx_thresh=[15, 20, 25, None], daily_ma_confirm=[True, None],
        momentum_filter=[True, None], atr_expansion=[True, None],
    ),
    ("S5", "daily"): dict(
        pivot_type=["classic", "fibonacci", "woodie", "camarilla"],
        target_exit=["pivot", "mid_SR", "opposite_SR"],
        max_hold=[5, 8, 10, 15], sl_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        tp_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35, None],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.15, 0.20, 0.30, 0.40, 0.50, None],
        flat_ma_slope=[0.005, 0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
        intraday_range=[True, None], range_bound_check=[True, None],
    ),
    ("S5", "hourly"): dict(
        pivot_type=["classic", "fibonacci", "woodie", "camarilla"],
        target_exit=["pivot", "mid_SR", "opposite_SR"],
        max_hold=[10, 16, 20, 30], sl_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        tp_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35, None],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.15, 0.20, 0.30, 0.40, 0.50, None],
        flat_ma_slope=[0.005, 0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
        intraday_range=[True, None], range_bound_check=[True, None],
    ),
    ("S6", "daily"): dict(
        vwap_window=[5, 10, 15, 20, 25, 30], dev_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        exit_mult=[0.0, 0.3, 0.5, 0.7], max_hold=[5, 8, 10, 15],
        sl_mult=[0.5, 0.75, 1.0, 1.5, 2.0], tp_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35, None],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.15, 0.20, 0.30, 0.40, 0.50, None],
        flat_ma_slope=[0.005, 0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
        hurst_proxy=[True, None],
        intraday_range=[True, None], range_bound_check=[True, None],
        vwap_slope_flat=[True, None],
    ),
    ("S6", "hourly"): dict(
        vwap_window=[5, 10, 15, 20, 25, 30], dev_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        exit_mult=[0.0, 0.3, 0.5, 0.7], max_hold=[10, 16, 20, 30],
        sl_mult=[0.5, 0.75, 1.0, 1.5, 2.0], tp_mult=[0.5, 0.75, 1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35, None],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.15, 0.20, 0.30, 0.40, 0.50, None],
        flat_ma_slope=[0.005, 0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
        hurst_proxy=[True, None],
        intraday_range=[True, None], range_bound_check=[True, None],
        vwap_slope_flat=[True, None],
    ),
}

PARAM_KEYS = {
    "S1": ["ma_window", "z_entry", "z_exit", "max_hold", "sl_mult", "tp_mult",
           "rsi_thresh", "vol_regime", "consec_candles", "vol_exhaustion",
           "ma200_filter", "mean_revert_speed"],
    "S2": ["bb_window", "bb_std", "max_hold", "sl_mult", "tp_mult",
           "bw_percentile", "vol_regime", "consec_candles", "vol_exhaustion",
           "rsi_divergence", "close_vs_bb_pct", "squeeze_then_expand"],
    "S3": ["dc_window", "initial_sl_mult", "trail_n", "trail_atr_mult",
           "breakeven_thresh", "adx_thresh", "vol_confirm", "daily_ma_confirm",
           "momentum_filter", "atr_expansion"],
    "S4": ["atr_period", "multiplier", "initial_sl_mult", "trail_n",
           "trail_atr_mult", "breakeven_thresh", "adx_thresh", "daily_ma_confirm",
           "momentum_filter", "atr_expansion"],
    "S5": ["pivot_type", "target_exit", "max_hold", "sl_mult", "tp_mult",
           "adx_exit_thresh", "adx_entry_thresh", "bb_squeeze_pctl", "flat_ma_slope",
           "vol_compression", "intraday_range", "range_bound_check"],
    "S6": ["vwap_window", "dev_mult", "exit_mult", "max_hold", "sl_mult", "tp_mult",
           "adx_exit_thresh", "adx_entry_thresh", "bb_squeeze_pctl", "flat_ma_slope",
           "vol_compression", "hurst_proxy", "intraday_range", "range_bound_check",
           "vwap_slope_flat"],
}


# ════════════════════════════════════════════════════════════
# Indicator functions
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
    if s >= n:
        return st, d
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

def calc_hurst_proxy(close, window=20):
    n = len(close)
    hurst = np.full(n, np.nan)
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    for i in range(window, n):
        ret = log_ret[i - window + 1:i + 1]
        m = np.mean(ret)
        dev = ret - m
        cumdev = np.cumsum(dev)
        R = np.max(cumdev) - np.min(cumdev)
        S = np.std(ret, ddof=1)
        if S > 1e-12 and R > 0:
            hurst[i] = np.log(R / S) / np.log(window)
        else:
            hurst[i] = 0.5
    return hurst


# ════════════════════════════════════════════════════════════
# Base indicators (computed once per ticker/TF)
# ════════════════════════════════════════════════════════════

def compute_base(close, high, low, open_arr, volume, is_hourly=False):
    ind = {}
    n = len(close)
    ind["rsi14"] = calc_rsi(close, 14)
    ind["atr14"] = calc_atr(high, low, close, 14)
    ind["adx14"] = calc_adx(high, low, close, 14)
    ind["vol_sma20"] = calc_sma(volume, 20)
    ind["vol_sma5"] = calc_sma(volume, 5)

    # Vol regime
    atr_sma50 = calc_sma(ind["atr14"], 50)
    atr_sma50_safe = np.where(np.isnan(atr_sma50) | (atr_sma50 < 1e-12), 1e-12, atr_sma50)
    ind["vol_regime"] = ind["atr14"] / atr_sma50_safe

    # Consecutive candles
    red = (close < open_arr).astype(float)
    green = (close > open_arr).astype(float)
    ind["red_count5"] = pd.Series(red).rolling(5).sum().values
    ind["green_count5"] = pd.Series(green).rolling(5).sum().values

    # Rolling high/low for trail_n (extended: added 3, 30)
    for tn in [3, 5, 10, 15, 20, 25, 30]:
        ind[f"rh_{tn}"] = pd.Series(high).rolling(tn + 1, min_periods=1).max().values
        ind[f"rl_{tn}"] = pd.Series(low).rolling(tn + 1, min_periods=1).min().values

    # Volume filter
    vs20 = ind["vol_sma20"]
    vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), np.inf, vs20)
    ind["vf_pass"] = ~(~np.isnan(vs20) & (vs20 > 0) & (volume < 0.5 * vs20_safe))
    ind["_volume"] = volume

    # BB width (from standard 20/2.0)
    sma20 = calc_sma(close, 20)
    std20 = calc_std(close, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    sma20_safe = np.where(np.abs(sma20) > 1e-12, sma20, 1e-12)
    bb_width = (bb_upper - bb_lower) / sma20_safe
    ind["bb_width"] = bb_width

    # BW rolling percentiles (extended: added 0.15, 0.50)
    bw_win = 252 * 9 if is_hourly else 252
    for pctl in [0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50]:
        key = f"bw_p{int(pctl*100)}"
        ind[key] = pd.Series(bb_width).rolling(bw_win, min_periods=50).quantile(pctl).values

    # RSI divergence arrays (for S2)
    ind["rsi_min5"] = pd.Series(ind["rsi14"]).rolling(5).min().values
    ind["rsi_max5"] = pd.Series(ind["rsi14"]).rolling(5).max().values
    ind["close_min5"] = pd.Series(close).rolling(5).min().values
    ind["close_max5"] = pd.Series(close).rolling(5).max().values

    # MA slopes for flat detection
    sma20_lag10 = np.full(n, np.nan)
    sma20_lag10[10:] = sma20[:-10]
    sma20_lag_safe = np.where(np.abs(sma20_lag10) > 1e-12, sma20_lag10, 1e-12)
    ind["sma20_slope10"] = (sma20 - sma20_lag10) / sma20_lag_safe

    # Hurst proxy
    ind["hurst20"] = calc_hurst_proxy(close, 20)

    # ── NEW V2 indicators ──

    # SMA200 for ma200_filter (S1)
    ind["sma200"] = calc_sma(close, 200)

    # ROC(10) for momentum_filter (S3, S4)
    close_lag10 = np.full(n, np.nan)
    close_lag10[10:] = close[:-10]
    safe_lag10 = np.where(np.abs(close_lag10) > 1e-12, close_lag10, 1e-12)
    ind["roc10"] = (close - close_lag10) / safe_lag10

    # ATR(50) for atr_expansion (S3, S4)
    ind["atr50"] = calc_atr(high, low, close, 50)

    # Intraday range ratio for S5/S6
    close_safe = np.where(np.abs(close) > 1e-12, close, 1e-12)
    ind["intraday_range_ratio"] = (high - low) / close_safe
    ind["idr_median20"] = pd.Series(ind["intraday_range_ratio"]).rolling(
        20, min_periods=5).median().values

    # Range span 5 for S5/S6
    rh5 = pd.Series(high).rolling(5, min_periods=1).max().values
    rl5 = pd.Series(low).rolling(5, min_periods=1).min().values
    ind["range_span_5"] = rh5 - rl5

    # BB width shifted arrays for squeeze_then_expand (S2)
    ind["bb_width_lag1"] = np.full(n, np.nan)
    ind["bb_width_lag1"][1:] = bb_width[:-1]
    ind["bb_width_lag2"] = np.full(n, np.nan)
    ind["bb_width_lag2"][2:] = bb_width[:-2]
    ind["bb_width_lag3"] = np.full(n, np.nan)
    ind["bb_width_lag3"][3:] = bb_width[:-3]

    return ind


# ════════════════════════════════════════════════════════════
# Precompute caches
# ════════════════════════════════════════════════════════════

def precompute_sma_cache(close, windows):
    cache = {}
    for w in windows:
        cache[w] = (calc_sma(close, w), calc_std(close, w))
    return cache

def precompute_donchian_cache(high, low, windows):
    cache = {}
    for w in windows:
        hc = pd.Series(high).rolling(w).max().values
        lc = pd.Series(low).rolling(w).min().values
        cache[w] = (hc, lc)
    return cache

def precompute_supertrend_cache(high, low, close, atr_periods, multipliers):
    cache = {}
    for ap in atr_periods:
        for m in multipliers:
            cache[(ap, m)] = calc_supertrend(high, low, close, ap, m)
    return cache

def precompute_vwap_cache(close, high, low, volume, windows):
    cache = {}
    for w in windows:
        tp = (high + low + close) / 3.0
        tp_vol = tp * volume
        sum_tv = pd.Series(tp_vol).rolling(w, min_periods=w).sum().values
        sum_v = pd.Series(volume).rolling(w, min_periods=w).sum().values
        sum_v_safe = np.where(np.isnan(sum_v) | (sum_v < 1e-12), 1e-12, sum_v)
        vwap = sum_tv / sum_v_safe
        dev = pd.Series(close - vwap).rolling(w, min_periods=w).std(ddof=1).values
        dev = np.where(np.isnan(dev) | (dev < 1e-12), 1e-12, dev)
        cache[w] = (vwap, dev)
    return cache

def calc_pivot_daily(high, low, close, variant):
    """Compute daily pivots (shifted by 1 day). Returns (P, S1, R1)."""
    n = len(close)
    P = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    R1 = np.full(n, np.nan)
    if variant == "classic":
        P[1:] = (high[:-1] + low[:-1] + close[:-1]) / 3.0
        S1[1:] = 2 * P[1:] - high[:-1]
        R1[1:] = 2 * P[1:] - low[:-1]
    elif variant == "fibonacci":
        P[1:] = (high[:-1] + low[:-1] + close[:-1]) / 3.0
        rng = high[:-1] - low[:-1]
        S1[1:] = P[1:] - 0.382 * rng
        R1[1:] = P[1:] + 0.382 * rng
    elif variant == "woodie":
        P[1:] = (high[:-1] + low[:-1] + 2 * close[:-1]) / 4.0
        S1[1:] = 2 * P[1:] - high[:-1]
        R1[1:] = 2 * P[1:] - low[:-1]
    elif variant == "camarilla":
        P[1:] = (high[:-1] + low[:-1] + close[:-1]) / 3.0
        rng = high[:-1] - low[:-1]
        S1[1:] = close[:-1] - 1.1 * rng / 12.0
        R1[1:] = close[:-1] + 1.1 * rng / 12.0
    return P, S1, R1

def compute_daily_pivots_for_hourly(daily_df, ticker, h_dts, variant):
    """Compute daily pivots and align to hourly bars."""
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date")
    high_d = tdf["high"].values.astype(np.float64)
    low_d = tdf["low"].values.astype(np.float64)
    close_d = tdf["close"].values.astype(np.float64)
    dates_d = np.array(tdf["date"].values, dtype="datetime64[D]")

    P_d, S1_d, R1_d = calc_pivot_daily(high_d, low_d, close_d, variant)

    h_dates = np.array(h_dts, dtype="datetime64[D]")
    indices = np.searchsorted(dates_d, h_dates, side="right") - 1
    valid = indices >= 0

    nh = len(h_dts)
    P_h = np.full(nh, np.nan)
    S1_h = np.full(nh, np.nan)
    R1_h = np.full(nh, np.nan)
    P_h[valid] = P_d[indices[valid]]
    S1_h[valid] = S1_d[indices[valid]]
    R1_h[valid] = R1_d[indices[valid]]
    return P_h, S1_h, R1_h


# ════════════════════════════════════════════════════════════
# Signal generators
# ════════════════════════════════════════════════════════════

def gen_s1_signals(close, ind, params, sma_cache):
    """S1 MA Mean Reversion (Contrarian). Returns (sig, exit_arr)."""
    w = int(params["ma_window"])
    sma, std = sma_cache[w]
    z = (close - sma) / std
    z_ent = params["z_entry"]
    rsi = ind["rsi14"]
    n = len(close)

    valid = ~np.isnan(z)

    f_vr = np.ones(n, dtype=bool)
    vr_t = params.get("vol_regime")
    if vr_t is not None:
        vr = ind["vol_regime"]
        f_vr = ~np.isnan(vr) & (vr < vr_t)

    f_cc_l = np.ones(n, dtype=bool)
    f_cc_s = np.ones(n, dtype=bool)
    cc = params.get("consec_candles")
    if cc is not None:
        f_cc_l = ~np.isnan(ind["red_count5"]) & (ind["red_count5"] >= cc)
        f_cc_s = ~np.isnan(ind["green_count5"]) & (ind["green_count5"] >= cc)

    f_ve = np.ones(n, dtype=bool)
    if params.get("vol_exhaustion"):
        vs5 = ind["vol_sma5"]
        vol = ind["_volume"]
        vs5_safe = np.where(np.isnan(vs5) | (vs5 <= 0), np.inf, vs5)
        f_ve = ~(~np.isnan(vs5) & (vs5 > 0) & (vol >= vs5_safe))

    f_rsi_l = np.ones(n, dtype=bool)
    f_rsi_s = np.ones(n, dtype=bool)
    rt = params.get("rsi_thresh")
    if rt is not None:
        f_rsi_l = ~np.isnan(rsi) & (rsi < rt)
        f_rsi_s = ~np.isnan(rsi) & (rsi > (100 - rt))

    # NEW: ma200_filter
    f_ma200_l = np.ones(n, dtype=bool)
    f_ma200_s = np.ones(n, dtype=bool)
    if params.get("ma200_filter"):
        sma200 = ind["sma200"]
        v200 = ~np.isnan(sma200)
        f_ma200_l = ~v200 | (close > sma200)
        f_ma200_s = ~v200 | (close < sma200)

    # NEW: mean_revert_speed
    f_mrs_l = np.ones(n, dtype=bool)
    f_mrs_s = np.ones(n, dtype=bool)
    if params.get("mean_revert_speed"):
        z_lag1 = np.full(n, np.nan)
        z_lag1[1:] = z[:-1]
        valid_z = ~np.isnan(z) & ~np.isnan(z_lag1)
        f_mrs_l = ~valid_z | (z_lag1 < z)
        f_mrs_s = ~valid_z | (z_lag1 > z)

    vfp = ind["vf_pass"]
    entry_l = valid & (z < -z_ent) & f_vr & f_cc_l & f_ve & f_rsi_l & f_ma200_l & f_mrs_l & vfp
    entry_s = valid & (z > z_ent) & f_vr & f_cc_s & f_ve & f_rsi_s & f_ma200_s & f_mrs_s & vfp

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    z_exit = params["z_exit"]
    exit_arr = ~np.isnan(z) & (np.abs(z) < z_exit)
    return sig, exit_arr


def gen_s2_signals(close, ind, params, sma_cache, is_hourly):
    """S2 Bollinger Bands (Contrarian). Returns (sig, exit_arr)."""
    w = int(params["bb_window"])
    sma, std = sma_cache[w]
    bb_s = params["bb_std"]
    bb_upper = sma + bb_s * std
    bb_lower = sma - bb_s * std
    n = len(close)

    valid = ~np.isnan(sma)
    entry_l = valid & (close <= bb_lower)
    entry_s = valid & (close >= bb_upper)

    vr_t = params.get("vol_regime")
    if vr_t is not None:
        vr = ind["vol_regime"]
        f = ~np.isnan(vr) & (vr < vr_t)
        entry_l = entry_l & f
        entry_s = entry_s & f

    cc = params.get("consec_candles")
    if cc is not None:
        entry_l = entry_l & ~np.isnan(ind["red_count5"]) & (ind["red_count5"] >= cc)
        entry_s = entry_s & ~np.isnan(ind["green_count5"]) & (ind["green_count5"] >= cc)

    if params.get("vol_exhaustion"):
        vs5 = ind["vol_sma5"]
        vol = ind["_volume"]
        vs5_safe = np.where(np.isnan(vs5) | (vs5 <= 0), np.inf, vs5)
        f_ve = ~(~np.isnan(vs5) & (vs5 > 0) & (vol >= vs5_safe))
        entry_l = entry_l & f_ve
        entry_s = entry_s & f_ve

    bw_p = params.get("bw_percentile")
    if bw_p is not None:
        key = f"bw_p{int(bw_p*100)}"
        bw_thresh = ind.get(key)
        if bw_thresh is not None:
            f_bw = ~np.isnan(bw_thresh) & (ind["bb_width"] > bw_thresh)
            entry_l = entry_l & f_bw
            entry_s = entry_s & f_bw

    if params.get("rsi_divergence"):
        rsi = ind["rsi14"]
        f_div_l = (~np.isnan(ind["close_min5"]) & ~np.isnan(ind["rsi_min5"]) &
                   ~np.isnan(rsi) & (close <= ind["close_min5"]) &
                   (rsi > ind["rsi_min5"]))
        f_div_s = (~np.isnan(ind["close_max5"]) & ~np.isnan(ind["rsi_max5"]) &
                   ~np.isnan(rsi) & (close >= ind["close_max5"]) &
                   (rsi < ind["rsi_max5"]))
        entry_l = entry_l & f_div_l
        entry_s = entry_s & f_div_s

    # NEW: close_vs_bb_pct — deep penetration
    if params.get("close_vs_bb_pct"):
        bb_range = bb_upper - bb_lower
        entry_l = entry_l & (close < bb_lower - 0.5 * bb_range)
        entry_s = entry_s & (close > bb_upper + 0.5 * bb_range)

    # NEW: squeeze_then_expand
    if params.get("squeeze_then_expand"):
        bw = ind["bb_width"]
        bw1 = ind["bb_width_lag1"]
        bw2 = ind["bb_width_lag2"]
        bw3 = ind["bb_width_lag3"]
        valid_bw = ~np.isnan(bw) & ~np.isnan(bw1) & ~np.isnan(bw2) & ~np.isnan(bw3)
        f_squeeze = valid_bw & (bw3 > bw2) & (bw2 > bw1) & (bw > bw1)
        entry_l = entry_l & f_squeeze
        entry_s = entry_s & f_squeeze

    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    exit_arr = np.zeros(n, dtype=bool)
    prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
    prev_m = np.empty_like(sma); prev_m[0] = sma[0]; prev_m[1:] = sma[:-1]
    exit_arr[1:] = (((prev_c[1:] < prev_m[1:]) & (close[1:] >= sma[1:])) |
                    ((prev_c[1:] > prev_m[1:]) & (close[1:] <= sma[1:])))
    return sig, exit_arr


def gen_s3_signals(close, high, low, ind, params, dc_cache, daily_trend=None):
    """S3 Donchian Channel (Trend). Returns (sig, exit_long, exit_short)."""
    w = int(params["dc_window"])
    hc, lc = dc_cache[w]
    prev_hc = np.empty_like(hc); prev_hc[0] = np.nan; prev_hc[1:] = hc[:-1]
    prev_lc = np.empty_like(lc); prev_lc[0] = np.nan; prev_lc[1:] = lc[:-1]
    n = len(close)

    valid = ~np.isnan(prev_hc)

    adx_t = params.get("adx_thresh")
    f_adx = np.ones(n, dtype=bool)
    if adx_t is not None:
        adx = ind["adx14"]
        f_adx = ~np.isnan(adx) & (adx > adx_t)

    vc = params.get("vol_confirm")
    f_vc = np.ones(n, dtype=bool)
    if vc is not None:
        vol = ind["_volume"]
        vs20 = ind["vol_sma20"]
        vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), 1.0, vs20)
        f_vc = (vol / vs20_safe) > vc

    # NEW: momentum_filter
    f_mom_l = np.ones(n, dtype=bool)
    f_mom_s = np.ones(n, dtype=bool)
    if params.get("momentum_filter"):
        roc = ind["roc10"]
        v_roc = ~np.isnan(roc)
        f_mom_l = ~v_roc | (roc > 0)
        f_mom_s = ~v_roc | (roc < 0)

    # NEW: atr_expansion
    f_atr_exp = np.ones(n, dtype=bool)
    if params.get("atr_expansion"):
        atr14 = ind["atr14"]
        atr50 = ind["atr50"]
        v_atr = ~np.isnan(atr14) & ~np.isnan(atr50)
        f_atr_exp = ~v_atr | (atr14 > atr50)

    entry_l = valid & (close > prev_hc) & f_adx & f_vc & f_mom_l & f_atr_exp & ind["vf_pass"]
    entry_s = valid & (close < prev_lc) & f_adx & f_vc & f_mom_s & f_atr_exp & ind["vf_pass"]

    if params.get("daily_ma_confirm") and daily_trend is not None:
        dt_valid = ~np.isnan(daily_trend)
        entry_l = entry_l & (~dt_valid | (daily_trend >= 0.5))
        entry_s = entry_s & (~dt_valid | (daily_trend < 0.5))

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    exit_l = ~np.isnan(prev_lc) & (close < prev_lc)
    exit_s = ~np.isnan(prev_hc) & (close > prev_hc)
    return sig, exit_l, exit_s


def gen_s4_signals(close, high, low, ind, params, st_cache, daily_trend=None):
    """S4 Supertrend (Trend). Returns (sig, exit_long, exit_short)."""
    ap = int(params["atr_period"])
    m = float(params["multiplier"])
    st, _ = st_cache[(ap, m)]
    st_prev = np.empty_like(st); st_prev[0] = np.nan; st_prev[1:] = st[:-1]
    c_prev = np.empty_like(close); c_prev[0] = close[0]; c_prev[1:] = close[:-1]
    n = len(close)

    valid = ~np.isnan(st) & ~np.isnan(st_prev)

    adx_t = params.get("adx_thresh")
    f_adx = np.ones(n, dtype=bool)
    if adx_t is not None:
        adx = ind["adx14"]
        f_adx = ~np.isnan(adx) & (adx > adx_t)

    # NEW: momentum_filter
    f_mom_l = np.ones(n, dtype=bool)
    f_mom_s = np.ones(n, dtype=bool)
    if params.get("momentum_filter"):
        roc = ind["roc10"]
        v_roc = ~np.isnan(roc)
        f_mom_l = ~v_roc | (roc > 0)
        f_mom_s = ~v_roc | (roc < 0)

    # NEW: atr_expansion
    f_atr_exp = np.ones(n, dtype=bool)
    if params.get("atr_expansion"):
        atr14 = ind["atr14"]
        atr50 = ind["atr50"]
        v_atr = ~np.isnan(atr14) & ~np.isnan(atr50)
        f_atr_exp = ~v_atr | (atr14 > atr50)

    entry_l = valid & (close > st) & (c_prev <= st_prev) & f_adx & f_mom_l & f_atr_exp & ind["vf_pass"]
    entry_s = valid & (close < st) & (c_prev >= st_prev) & f_adx & f_mom_s & f_atr_exp & ind["vf_pass"]

    if params.get("daily_ma_confirm") and daily_trend is not None:
        dt_valid = ~np.isnan(daily_trend)
        entry_l = entry_l & (~dt_valid | (daily_trend >= 0.5))
        entry_s = entry_s & (~dt_valid | (daily_trend < 0.5))

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    exit_l = valid & (close < st) & (c_prev >= st_prev)
    exit_s = valid & (close > st) & (c_prev <= st_prev)
    return sig, exit_l, exit_s


def _apply_range_filters(entry_l, entry_s, ind, params):
    """Apply common Range filters."""
    adx_e = params.get("adx_entry_thresh")
    if adx_e is not None:
        adx = ind["adx14"]
        f = ~np.isnan(adx) & (adx < adx_e)
        entry_l = entry_l & f
        entry_s = entry_s & f

    bw_p = params.get("bb_squeeze_pctl")
    if bw_p is not None:
        key = f"bw_p{int(bw_p*100)}"
        bw_thresh = ind.get(key)
        if bw_thresh is not None:
            f = ~np.isnan(bw_thresh) & (ind["bb_width"] < bw_thresh)
            entry_l = entry_l & f
            entry_s = entry_s & f

    fms = params.get("flat_ma_slope")
    if fms is not None:
        slope = ind["sma20_slope10"]
        f = ~np.isnan(slope) & (np.abs(slope) < fms)
        entry_l = entry_l & f
        entry_s = entry_s & f

    vc = params.get("vol_compression")
    if vc is not None:
        vr = ind["vol_regime"]
        f = ~np.isnan(vr) & (vr < vc)
        entry_l = entry_l & f
        entry_s = entry_s & f

    # NEW: intraday_range
    if params.get("intraday_range"):
        idr = ind["intraday_range_ratio"]
        idr_med = ind["idr_median20"]
        v = ~np.isnan(idr) & ~np.isnan(idr_med)
        f = ~v | (idr < idr_med)
        entry_l = entry_l & f
        entry_s = entry_s & f

    # NEW: range_bound_check
    if params.get("range_bound_check"):
        span5 = ind["range_span_5"]
        atr14 = ind["atr14"]
        v = ~np.isnan(span5) & ~np.isnan(atr14) & (atr14 > 1e-12)
        f = ~v | (span5 < 2.0 * atr14)
        entry_l = entry_l & f
        entry_s = entry_s & f

    return entry_l, entry_s


def gen_s5_signals(close, ind, params, pivot_data):
    """S5 Pivot Points (Range). Returns (sig, exit_info)."""
    variant = params["pivot_type"]
    P, S1, R1 = pivot_data[variant]
    n = len(close)

    entry_l = ~np.isnan(S1) & (close < S1)
    entry_s = ~np.isnan(R1) & (close > R1)

    entry_l, entry_s = _apply_range_filters(entry_l, entry_s, ind, params)
    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    target_exit = params.get("target_exit", "pivot")

    if target_exit == "pivot":
        exit_arr = np.zeros(n, dtype=bool)
        prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
        prev_P = np.empty_like(P); prev_P[0] = P[0]; prev_P[1:] = P[:-1]
        valid_P = ~np.isnan(P) & ~np.isnan(prev_P)
        exit_arr[1:] = valid_P[1:] & (
            ((prev_c[1:] < prev_P[1:]) & (close[1:] >= P[1:])) |
            ((prev_c[1:] > prev_P[1:]) & (close[1:] <= P[1:])))
        return sig, exit_arr

    elif target_exit == "mid_SR":
        mid = (S1 + R1) / 2.0
        exit_arr = np.zeros(n, dtype=bool)
        prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
        prev_mid = np.empty_like(mid); prev_mid[0] = mid[0]; prev_mid[1:] = mid[:-1]
        valid_mid = ~np.isnan(mid) & ~np.isnan(prev_mid)
        exit_arr[1:] = valid_mid[1:] & (
            ((prev_c[1:] < prev_mid[1:]) & (close[1:] >= mid[1:])) |
            ((prev_c[1:] > prev_mid[1:]) & (close[1:] <= mid[1:])))
        return sig, exit_arr

    elif target_exit == "opposite_SR":
        exit_l = ~np.isnan(R1) & (close >= R1)
        exit_s = ~np.isnan(S1) & (close <= S1)
        return sig, (exit_l, exit_s)

    exit_arr = np.zeros(n, dtype=bool)
    return sig, exit_arr


def gen_s6_signals(close, high, low, volume, ind, params, vwap_cache):
    """S6 VWAP Reversion (Range). Returns (sig, exit_arr)."""
    w = int(params["vwap_window"])
    vwap, dev = vwap_cache[w]
    dm = params["dev_mult"]
    n = len(close)

    entry_l = ~np.isnan(vwap) & (close < vwap - dm * dev)
    entry_s = ~np.isnan(vwap) & (close > vwap + dm * dev)

    entry_l, entry_s = _apply_range_filters(entry_l, entry_s, ind, params)

    if params.get("hurst_proxy"):
        h = ind["hurst20"]
        f = ~np.isnan(h) & (h < 0.45)
        entry_l = entry_l & f
        entry_s = entry_s & f

    # NEW: vwap_slope_flat
    if params.get("vwap_slope_flat"):
        vwap_lag5 = np.full(n, np.nan)
        vwap_lag5[5:] = vwap[:-5]
        vwap_safe = np.where(np.abs(vwap_lag5) > 1e-12, vwap_lag5, 1e-12)
        slope = np.abs(vwap - vwap_lag5) / vwap_safe
        v = ~np.isnan(slope)
        f = ~v | (slope < 0.005)
        entry_l = entry_l & f
        entry_s = entry_s & f

    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    em = params["exit_mult"]
    exit_arr = ~np.isnan(vwap) & (np.abs(close - vwap) < em * dev)
    return sig, exit_arr


# ════════════════════════════════════════════════════════════
# Backtest engines (@njit)
# ════════════════════════════════════════════════════════════

@njit(cache=True)
def bt_contrarian(sig_arr, exit_arr, close, high, low, atr14,
                  sl_mult, tp_mult, max_hold, warmup, end_idx):
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0

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
                cp = 0.0; held = 0

        if cp == 0.0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0
                sl = close[t] - sl_mult * a
                tp = close[t] + tp_mult * a
                held = 0
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0
                sl = close[t] + sl_mult * a
                tp = close[t] - tp_mult * a
                held = 0
        pos[t] = cp
    return pos


@njit(cache=True)
def bt_trend(sig_arr, exit_long, exit_short, close, high, low, atr14,
             rolling_high, rolling_low, isl_mult, trail_mult, be_thresh,
             warmup, end_idx):
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; ep = 0.0; ea = 0.0; be = False
    sl = np.nan

    for t in range(warmup, n):
        if cp != 0.0:
            ca = atr14[t]
            if ca != ca: ca = ea
            closed = False
            if cp == 1.0:
                tsl = rolling_high[t] - trail_mult * ca
                if not be and (close[t] - ep) > be_thresh * ea:
                    be = True
                if be:
                    tsl = max(tsl, ep)
                if sl != sl: sl = tsl
                else: sl = max(sl, tsl)
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
                if sl != sl: sl = tsl
                else: sl = min(sl, tsl)
                if high[t] >= sl:
                    closed = True
                elif exit_short[t]:
                    closed = True
            if closed:
                cp = 0.0; sl = np.nan; ep = 0.0; ea = 0.0; be = False

        if cp == 0.0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0; ep = close[t]; ea = a; be = False
                sl = ep - isl_mult * a
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0; ep = close[t]; ea = a; be = False
                sl = ep + isl_mult * a
        pos[t] = cp
    return pos


@njit(cache=True)
def bt_range(sig_arr, exit_arr, close, high, low, atr14, adx14,
             sl_mult, tp_mult, adx_exit_thresh, max_hold, warmup, end_idx):
    """Range RM: SL > TP > ADX breakout > signal > max_hold."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0

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
            if not closed:
                av = adx14[t]
                if av == av and av > adx_exit_thresh:
                    closed = True
            if not closed and exit_arr[t]:
                closed = True
            if not closed and held >= max_hold:
                closed = True
            if closed:
                cp = 0.0; held = 0

        if cp == 0.0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0
                sl = close[t] - sl_mult * a
                tp = close[t] + tp_mult * a
                held = 0
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0
                sl = close[t] + sl_mult * a
                tp = close[t] - tp_mult * a
                held = 0
        pos[t] = cp
    return pos


@njit(cache=True)
def bt_range_split_exit(sig_arr, exit_l, exit_s, close, high, low, atr14, adx14,
                        sl_mult, tp_mult, adx_exit_thresh, max_hold, warmup, end_idx):
    """Range RM with separate exit arrays for long/short (S5 opposite_SR)."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0

    for t in range(warmup, n):
        if cp != 0.0:
            held += 1
            closed = False
            if cp == 1.0:
                if low[t] <= sl:
                    closed = True
                elif high[t] >= tp:
                    closed = True
                if not closed:
                    av = adx14[t]
                    if av == av and av > adx_exit_thresh:
                        closed = True
                if not closed and exit_l[t]:
                    closed = True
            else:
                if high[t] >= sl:
                    closed = True
                elif low[t] <= tp:
                    closed = True
                if not closed:
                    av = adx14[t]
                    if av == av and av > adx_exit_thresh:
                        closed = True
                if not closed and exit_s[t]:
                    closed = True
            if not closed and held >= max_hold:
                closed = True
            if closed:
                cp = 0.0; held = 0

        if cp == 0.0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0
                sl = close[t] - sl_mult * a
                tp = close[t] + tp_mult * a
                held = 0
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0
                sl = close[t] + sl_mult * a
                tp = close[t] - tp_mult * a
                held = 0
        pos[t] = cp
    return pos


# ════════════════════════════════════════════════════════════
# Metrics
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
    trades = 0; in_t = False; t_bars = 0
    for i in range(len(pos)):
        if pos[i] != 0 and not in_t:
            trades += 1; in_t = True
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
# Daily trend alignment (for S3h, S4h)
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
# Dispatch functions
# ════════════════════════════════════════════════════════════

def dispatch_signals(sid, close, high, low, volume, ind, params,
                     sma_cache, dc_cache, st_cache, vwap_cache,
                     pivot_data, daily_trend, is_hourly):
    """Generate signals for given strategy. Returns (sig, exit_info)."""
    if sid == "S1":
        return gen_s1_signals(close, ind, params, sma_cache)
    elif sid == "S2":
        return gen_s2_signals(close, ind, params, sma_cache, is_hourly)
    elif sid == "S3":
        dt = daily_trend if params.get("daily_ma_confirm") else None
        sig, el, es = gen_s3_signals(close, high, low, ind, params, dc_cache, dt)
        return sig, (el, es)
    elif sid == "S4":
        dt = daily_trend if params.get("daily_ma_confirm") else None
        sig, el, es = gen_s4_signals(close, high, low, ind, params, st_cache, dt)
        return sig, (el, es)
    elif sid == "S5":
        return gen_s5_signals(close, ind, params, pivot_data)
    elif sid == "S6":
        return gen_s6_signals(close, high, low, volume, ind, params, vwap_cache)

def dispatch_backtest(sid, sig, exit_info, close, high, low, ind, params,
                      warmup, end_idx):
    """Run backtest for given strategy. Returns positions array."""
    cat = CATEGORY[sid]
    if cat == "Contrarian":
        return bt_contrarian(sig, exit_info, close, high, low, ind["atr14"],
                             params["sl_mult"], params["tp_mult"],
                             int(params["max_hold"]), warmup, end_idx)
    elif cat == "Trend":
        exit_l, exit_s = exit_info
        tn = int(params["trail_n"])
        rh = ind.get(f"rh_{tn}")
        rl = ind.get(f"rl_{tn}")
        if rh is None:
            rh = pd.Series(high).rolling(tn + 1, min_periods=1).max().values
            rl = pd.Series(low).rolling(tn + 1, min_periods=1).min().values
        be = params.get("breakeven_thresh")
        be_val = 1e12 if be is None else float(be)
        return bt_trend(sig, exit_l, exit_s, close, high, low, ind["atr14"],
                        rh, rl, params["initial_sl_mult"], params["trail_atr_mult"],
                        be_val, warmup, end_idx)
    elif cat == "Range":
        adx_exit = params.get("adx_exit_thresh")
        adx_exit_val = 999.0 if adx_exit is None else float(adx_exit)
        if isinstance(exit_info, tuple):
            exit_l, exit_s = exit_info
            return bt_range_split_exit(sig, exit_l, exit_s, close, high, low,
                                       ind["atr14"], ind["adx14"],
                                       params["sl_mult"], params["tp_mult"],
                                       adx_exit_val, int(params["max_hold"]),
                                       warmup, end_idx)
        else:
            return bt_range(sig, exit_info, close, high, low, ind["atr14"],
                            ind["adx14"], params["sl_mult"], params["tp_mult"],
                            adx_exit_val, int(params["max_hold"]),
                            warmup, end_idx)


# ════════════════════════════════════════════════════════════
# Random search
# ════════════════════════════════════════════════════════════

def random_search_one(sid, tf, close, high, low, volume, ind,
                      train_start, train_end, n_samples, rng,
                      sma_cache, dc_cache, st_cache, vwap_cache,
                      pivot_data, daily_trend, is_hourly):
    """Random search for best params on train window [train_start, train_end)."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_start:train_end] = True
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)

    space = PARAM_SPACE[(sid, tf)]
    space_keys = list(space.keys())
    space_vals = [space[k] for k in space_keys]

    best_sh = -999.0
    best_params = None
    warmup_start = max(WARMUP, train_start)

    for _ in range(n_samples):
        p = {k: rng.choice(v) for k, v in zip(space_keys, space_vals)}

        sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind, p,
                                          sma_cache, dc_cache, st_cache, vwap_cache,
                                          pivot_data, daily_trend, is_hourly)
        pos = dispatch_backtest(sid, sig, exit_info, close, high, low, ind, p,
                                warmup_start, train_end)
        sh = calc_sharpe(pos, log_ret, train_mask, ann)
        if sh > best_sh:
            best_sh = sh
            best_params = dict(p)

    # Fallback to defaults if all negative
    if best_sh < 0:
        best_params = dict(V3_DEFAULTS.get((sid, tf), best_params or {}))
        if best_params:
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              best_params, sma_cache, dc_cache,
                                              st_cache, vwap_cache,
                                              pivot_data, daily_trend, is_hourly)
            pos = dispatch_backtest(sid, sig, exit_info, close, high, low, ind,
                                    best_params, warmup_start, train_end)
            best_sh = calc_sharpe(pos, log_ret, train_mask, ann)

    return best_params, best_sh


# ════════════════════════════════════════════════════════════
# Rolling calibration loop
# ════════════════════════════════════════════════════════════

def rolling_calibrate_one_meta(sid, tf, close, high, low, volume, ind,
                                recalib_freq, train_window, n_samples,
                                sma_cache, dc_cache, st_cache, vwap_cache,
                                pivot_data, daily_trend, is_hourly):
    """Rolling recalibration for one strategy x one ticker x one meta combo."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
    bpy = 252 * 9 if is_hourly else 252

    # Convert freq/window to bar units for hourly
    bar_freq = recalib_freq * 9 if is_hourly else recalib_freq
    bar_window = train_window * 9 if is_hourly else train_window

    first_recalib = WARMUP + bar_window
    if first_recalib >= n:
        return None

    recalib_points = list(range(first_recalib, n, bar_freq))
    oos_positions = np.zeros(n)
    param_history = []

    for i, t in enumerate(recalib_points):
        train_start = t - bar_window
        train_end = t

        rng = np.random.RandomState(SEED + i)
        best_params, train_sh = random_search_one(
            sid, tf, close, high, low, volume, ind,
            train_start, train_end, n_samples, rng,
            sma_cache, dc_cache, st_cache, vwap_cache,
            pivot_data, daily_trend, is_hourly)

        param_history.append({
            "recalib_idx": i, "recalib_bar": t,
            "train_sharpe": round(train_sh, 4),
            "params": best_params,
        })

        # Apply best params on OOS window
        oos_end = recalib_points[i + 1] if i + 1 < len(recalib_points) else n
        if best_params:
            sig, exit_info = dispatch_signals(
                sid, close, high, low, volume, ind, best_params,
                sma_cache, dc_cache, st_cache, vwap_cache,
                pivot_data, daily_trend, is_hourly)
            pos = dispatch_backtest(
                sid, sig, exit_info, close, high, low, ind, best_params,
                max(WARMUP, t), oos_end)
            oos_positions[t:oos_end] = pos[t:oos_end]

    # OOS metrics
    oos_mask = np.zeros(n, dtype=bool)
    if recalib_points:
        oos_mask[recalib_points[0]:] = True
    oos_sh = calc_sharpe(oos_positions, log_ret, oos_mask, ann)
    oos_metrics = calc_all_metrics(oos_positions, log_ret, oos_mask, ann, bpy)

    return {
        "oos_sharpe": oos_sh,
        "oos_metrics": oos_metrics,
        "oos_positions": oos_positions,
        "param_history": param_history,
        "n_recalibs": len(recalib_points),
        "log_ret": log_ret,
        "oos_mask": oos_mask,
    }


# ════════════════════════════════════════════════════════════
# Process one ticker
# ════════════════════════════════════════════════════════════

def process_ticker(ticker, daily, hourly):
    """Process all strategies x timeframes x meta combos for one ticker."""
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
    ind_d = compute_base(close_d, high_d, low_d, open_d, vol_d, is_hourly=False)
    ind_h = compute_base(close_h, high_h, low_h, open_h, vol_h, is_hourly=True)

    # Precompute caches — daily (V2: extended ranges)
    sma_cache_d = precompute_sma_cache(close_d, [10, 15, 20, 25, 30])
    dc_cache_d = precompute_donchian_cache(high_d, low_d, [5, 10, 15, 20, 25, 30, 40])
    st_cache_d = precompute_supertrend_cache(high_d, low_d, close_d,
                                              [7, 10, 14, 20, 25],
                                              [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    vwap_cache_d = precompute_vwap_cache(close_d, high_d, low_d, vol_d,
                                          [5, 10, 15, 20, 25, 30])

    # Precompute caches — hourly (V2: extended ranges)
    sma_cache_h = precompute_sma_cache(close_h, [10, 15, 20, 25, 30])
    dc_cache_h = precompute_donchian_cache(high_h, low_h, [5, 10, 15, 20, 25, 30, 40])
    st_cache_h = precompute_supertrend_cache(high_h, low_h, close_h,
                                              [7, 10, 14, 20, 25],
                                              [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    vwap_cache_h = precompute_vwap_cache(close_h, high_h, low_h, vol_h,
                                          [5, 10, 15, 20, 25, 30])

    # Daily trend for S3h, S4h
    d_dates, d_above = build_daily_trend(daily, ticker)
    daily_trend_h = align_daily_to_hourly(d_dates, d_above, dts_h)

    # Pivot data — daily (V2: 4 variants)
    pivot_d = {
        "classic": calc_pivot_daily(high_d, low_d, close_d, "classic"),
        "fibonacci": calc_pivot_daily(high_d, low_d, close_d, "fibonacci"),
        "woodie": calc_pivot_daily(high_d, low_d, close_d, "woodie"),
        "camarilla": calc_pivot_daily(high_d, low_d, close_d, "camarilla"),
    }
    # Pivot data — hourly (V2: 4 variants)
    pivot_h = {
        "classic": compute_daily_pivots_for_hourly(daily, ticker, dts_h, "classic"),
        "fibonacci": compute_daily_pivots_for_hourly(daily, ticker, dts_h, "fibonacci"),
        "woodie": compute_daily_pivots_for_hourly(daily, ticker, dts_h, "woodie"),
        "camarilla": compute_daily_pivots_for_hourly(daily, ticker, dts_h, "camarilla"),
    }

    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            is_h = tf == "hourly"
            c = close_h if is_h else close_d
            h = high_h if is_h else high_d
            l = low_h if is_h else low_d
            v = vol_h if is_h else vol_d
            ind = ind_h if is_h else ind_d
            sc = sma_cache_h if is_h else sma_cache_d
            dc = dc_cache_h if is_h else dc_cache_d
            stc = st_cache_h if is_h else st_cache_d
            vc = vwap_cache_h if is_h else vwap_cache_d
            pv = pivot_h if is_h else pivot_d
            dt = daily_trend_h if is_h else None

            for freq, window in META_GRID:
                res = rolling_calibrate_one_meta(
                    sid, tf, c, h, l, v, ind,
                    freq, window, N_SAMPLES,
                    sc, dc, stc, vc, pv, dt, is_h)
                results[(sid, tf, freq, window)] = res

    return results


# ════════════════════════════════════════════════════════════
# Meta-grid selection
# ════════════════════════════════════════════════════════════

def select_best_meta(all_ticker_results, sid, tf):
    """Select best (freq, window) by median OOS Sharpe across tickers."""
    best_meta = None
    best_median = -999.0
    for freq, window in META_GRID:
        sharpes = []
        for ticker in TICKERS:
            res = all_ticker_results[ticker].get((sid, tf, freq, window))
            if res is not None:
                sharpes.append(res["oos_sharpe"])
        if sharpes:
            med = np.median(sharpes)
            if med > best_median:
                best_median = med
                best_meta = (freq, window)
    return best_meta


# ════════════════════════════════════════════════════════════
# V3 defaults baseline (run on same period)
# ════════════════════════════════════════════════════════════

def run_v3_defaults_one(sid, tf, close, high, low, volume, ind,
                        oos_mask, sma_cache, dc_cache, st_cache, vwap_cache,
                        pivot_data, daily_trend, is_hourly):
    """Run V3 defaults and compute Sharpe on OOS mask."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)

    defaults = V3_DEFAULTS.get((sid, tf))
    if not defaults:
        return -999.0
    params = dict(defaults)
    sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind, params,
                                      sma_cache, dc_cache, st_cache, vwap_cache,
                                      pivot_data, daily_trend, is_hourly)
    pos = dispatch_backtest(sid, sig, exit_info, close, high, low, ind, params,
                            WARMUP, n)
    return calc_sharpe(pos, log_ret, oos_mask, ann)


# ════════════════════════════════════════════════════════════
# Multiprocessing: worker globals + warmup
# ════════════════════════════════════════════════════════════

_G_DAILY = None
_G_HOURLY = None


def _worker_func(ticker):
    """Worker function for multiprocessing — uses global data."""
    return ticker, process_ticker(ticker, _G_DAILY, _G_HOURLY)


def warmup_numba():
    """Compile numba functions before forking."""
    if not HAS_NUMBA:
        return
    n = 100
    sig = np.zeros(n, dtype=np.int8)
    sig[10] = 1
    ex = np.zeros(n, dtype=bool)
    c = np.random.randn(n).cumsum() + 100
    h = c + 0.5
    l = c - 0.5
    atr = np.full(n, 0.5)
    adx = np.full(n, 25.0)
    rh = np.maximum.accumulate(h)
    rl = np.minimum.accumulate(l)

    bt_contrarian(sig, ex, c, h, l, atr, 1.5, 2.0, 20, 5, n)
    bt_trend(sig, ex, ex, c, h, l, atr, rh, rl, 2.0, 1.5, 1e12, 5, n)
    bt_range(sig, ex, c, h, l, atr, adx, 1.5, 2.0, 999.0, 20, 5, n)
    bt_range_split_exit(sig, ex, ex, c, h, l, atr, adx, 1.5, 2.0, 999.0, 20, 5, n)
    print("  Numba warmup complete.")


# ════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════

def load_data():
    daily = pd.read_parquet(DATA_DIR / "ohlcv_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.read_parquet(DATA_DIR / "ohlcv_hourly.parquet")
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    return daily, hourly


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    global _G_DAILY, _G_HOURLY
    t0 = time.time()
    print(f"Rolling Recalibration V2")
    print(f"  N_SAMPLES={N_SAMPLES}, META_GRID={META_GRID}, N_WORKERS={N_WORKERS}")
    print(f"  numba={'YES' if HAS_NUMBA else 'NO'}")
    print(f"\nLoading data...")
    _G_DAILY, _G_HOURLY = load_data()
    daily, hourly = _G_DAILY, _G_HOURLY
    print(f"  Daily: {len(daily)} rows, Hourly: {len(hourly)} rows")

    # Warmup numba before fork
    warmup_numba()

    # Process tickers with multiprocessing
    all_results = {}
    print(f"\nProcessing {len(TICKERS)} tickers with {N_WORKERS} workers...")
    with Pool(N_WORKERS) as pool:
        for ticker, result in pool.imap_unordered(_worker_func, TICKERS):
            elapsed = time.time() - t0
            print(f"  {ticker} done ({elapsed:.0f}s elapsed)", flush=True)
            all_results[ticker] = result

    total_time = time.time() - t0
    print(f"\nTotal processing: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Output 1: calib_optimal_params_v2.csv ──
    rows_params = []
    for ticker in TICKERS:
        for sid in STRATEGY_IDS:
            for tf in TIMEFRAMES:
                for freq, window in META_GRID:
                    res = all_results[ticker].get((sid, tf, freq, window))
                    if res is None:
                        continue
                    for ph in res["param_history"]:
                        row = {
                            "strategy": STRATEGY_NAMES[sid], "timeframe": tf,
                            "ticker": ticker, "recalib_freq": freq,
                            "train_window": window,
                            "recalib_idx": ph["recalib_idx"],
                            "sharpe_train": ph["train_sharpe"],
                        }
                        if ph["params"]:
                            for k in PARAM_KEYS[sid]:
                                v = ph["params"].get(k)
                                if v is None:
                                    row[f"p_{k}"] = "OFF"
                                elif isinstance(v, bool):
                                    row[f"p_{k}"] = "ON" if v else "OFF"
                                else:
                                    row[f"p_{k}"] = v
                        rows_params.append(row)
    df_params = pd.DataFrame(rows_params)
    df_params.to_csv(OUT_TABLES / "calib_optimal_params_v2.csv", index=False)
    print(f"\n[1] calib_optimal_params_v2.csv: {len(df_params)} rows")

    # ── Output 2: calib_meta_comparison_v2.csv ──
    rows_meta = []
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            for freq, window in META_GRID:
                sharpes = []
                for ticker in TICKERS:
                    res = all_results[ticker].get((sid, tf, freq, window))
                    if res is not None:
                        sharpes.append(res["oos_sharpe"])
                if sharpes:
                    rows_meta.append({
                        "strategy": STRATEGY_NAMES[sid], "timeframe": tf,
                        "recalib_freq": freq, "train_window": window,
                        "sharpe_median": round(np.median(sharpes), 4),
                        "sharpe_mean": round(np.mean(sharpes), 4),
                        "n_positive": sum(1 for s in sharpes if s > 0),
                    })
    df_meta = pd.DataFrame(rows_meta)
    df_meta.to_csv(OUT_TABLES / "calib_meta_comparison_v2.csv", index=False)
    print(f"[2] calib_meta_comparison_v2.csv: {len(df_meta)} rows")

    # ── Select best meta per strategy x TF ──
    best_metas = {}
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = select_best_meta(all_results, sid, tf)
            best_metas[(sid, tf)] = bm

    # ── Output 3: calib_oos_results_v2.csv ──
    rows_oos = []
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            for ticker in TICKERS:
                res = all_results[ticker].get((sid, tf, freq, window))
                if res is None or not res.get("oos_metrics"):
                    continue
                row = {
                    "strategy": STRATEGY_NAMES[sid], "timeframe": tf,
                    "category": CATEGORY[sid], "ticker": ticker,
                }
                row.update(res["oos_metrics"])
                rows_oos.append(row)
    df_oos = pd.DataFrame(rows_oos)
    df_oos.to_csv(OUT_TABLES / "calib_oos_results_v2.csv", index=False)
    print(f"[3] calib_oos_results_v2.csv: {len(df_oos)} rows")

    # ── V3 defaults for comparison (compute on same OOS period) ──
    default_sharpes = {}
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            ticker_sharpes = []
            for ticker in TICKERS:
                res = all_results[ticker].get((sid, tf, freq, window))
                if res is None:
                    continue
                oos_mask = res["oos_mask"]
                # Re-extract arrays
                is_h = tf == "hourly"
                if is_h:
                    tdf = hourly[hourly["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
                    c = tdf["close"].values.astype(np.float64)
                    h = tdf["high"].values.astype(np.float64)
                    l = tdf["low"].values.astype(np.float64)
                    o = tdf["open"].values.astype(np.float64)
                    v = tdf["volume"].values.astype(np.float64)
                    dt = tdf["datetime"].values
                else:
                    tdf = daily[daily["ticker"] == ticker].sort_values("date").reset_index(drop=True)
                    c = tdf["close"].values.astype(np.float64)
                    h = tdf["high"].values.astype(np.float64)
                    l = tdf["low"].values.astype(np.float64)
                    o = tdf["open"].values.astype(np.float64)
                    v = tdf["volume"].values.astype(np.float64)
                    dt = tdf["date"].values

                ind = compute_base(c, h, l, o, v, is_hourly=is_h)
                sc = precompute_sma_cache(c, [10, 15, 20, 25, 30])
                dc = precompute_donchian_cache(h, l, [5, 10, 15, 20, 25, 30, 40])
                stc = precompute_supertrend_cache(h, l, c,
                                                   [7, 10, 14, 20, 25],
                                                   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
                vc = precompute_vwap_cache(c, h, l, v, [5, 10, 15, 20, 25, 30])

                if is_h:
                    d_dates, d_above = build_daily_trend(daily, ticker)
                    daily_tr = align_daily_to_hourly(d_dates, d_above, dt)
                    pv = {
                        "classic": compute_daily_pivots_for_hourly(daily, ticker, dt, "classic"),
                        "fibonacci": compute_daily_pivots_for_hourly(daily, ticker, dt, "fibonacci"),
                        "woodie": compute_daily_pivots_for_hourly(daily, ticker, dt, "woodie"),
                        "camarilla": compute_daily_pivots_for_hourly(daily, ticker, dt, "camarilla"),
                    }
                else:
                    daily_tr = None
                    pv = {
                        "classic": calc_pivot_daily(h, l, c, "classic"),
                        "fibonacci": calc_pivot_daily(h, l, c, "fibonacci"),
                        "woodie": calc_pivot_daily(h, l, c, "woodie"),
                        "camarilla": calc_pivot_daily(h, l, c, "camarilla"),
                    }

                sh = run_v3_defaults_one(sid, tf, c, h, l, v, ind, oos_mask,
                                          sc, dc, stc, vc, pv, daily_tr, is_h)
                ticker_sharpes.append(sh)
            if ticker_sharpes:
                default_sharpes[(sid, tf)] = np.median(ticker_sharpes)

    # ── Output 4: calib_summary_v2.csv ──
    rows_summary = []
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            sharpes = []
            for ticker in TICKERS:
                res = all_results[ticker].get((sid, tf, freq, window))
                if res is not None:
                    sharpes.append(res["oos_sharpe"])
            if not sharpes:
                continue
            med_cal = np.median(sharpes)
            med_def = default_sharpes.get((sid, tf), 0)
            delta = ((med_cal - med_def) / abs(med_def) * 100) if abs(med_def) > 1e-6 else 0.0
            rows_summary.append({
                "strategy": STRATEGY_NAMES[sid], "timeframe": tf,
                "category": CATEGORY[sid],
                "sharpe_default_median": round(med_def, 4),
                "sharpe_calibrated_median": round(med_cal, 4),
                "delta_pct": round(delta, 1),
                "best_freq": freq, "best_window": window,
                "n_positive": sum(1 for s in sharpes if s > 0),
            })
    df_summary = pd.DataFrame(rows_summary)
    df_summary.to_csv(OUT_TABLES / "calib_summary_v2.csv", index=False)
    print(f"[4] calib_summary_v2.csv: {len(df_summary)} rows")

    # ── Output 5: calib_param_stability_v2.csv ──
    rows_stab = []
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            all_param_vals = {}
            for ticker in TICKERS:
                res = all_results[ticker].get((sid, tf, freq, window))
                if res is None:
                    continue
                for ph in res["param_history"]:
                    if not ph["params"]:
                        continue
                    for k in PARAM_KEYS[sid]:
                        v = ph["params"].get(k)
                        if k not in all_param_vals:
                            all_param_vals[k] = []
                        all_param_vals[k].append(v)

            for pname, vals in all_param_vals.items():
                if not vals:
                    continue
                str_vals = [str(v) for v in vals]
                from collections import Counter
                cnt = Counter(str_vals)
                mode_val, mode_count = cnt.most_common(1)[0]
                total = len(str_vals)
                stability = mode_count / total * 100
                rows_stab.append({
                    "strategy": STRATEGY_NAMES[sid], "timeframe": tf,
                    "param_name": pname, "mode_value": mode_val,
                    "mode_count": mode_count, "total_count": total,
                    "stability_pct": round(stability, 1),
                })
    df_stab = pd.DataFrame(rows_stab)
    df_stab.to_csv(OUT_TABLES / "calib_param_stability_v2.csv", index=False)
    print(f"[5] calib_param_stability_v2.csv: {len(df_stab)} rows")

    # ── Output 6: signals_calibrated_v2.parquet ──
    sig_rows = []
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            is_h = tf == "hourly"
            for ticker in TICKERS:
                res = all_results[ticker].get((sid, tf, freq, window))
                if res is None:
                    continue
                oos_pos = res["oos_positions"]
                log_ret = res["log_ret"]
                oos_mask = res["oos_mask"]
                oos_idx = np.where(oos_mask)[0]
                if len(oos_idx) == 0:
                    continue
                if is_h:
                    tdf = hourly[hourly["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
                    dts = tdf["datetime"].values
                else:
                    tdf = daily[daily["ticker"] == ticker].sort_values("date").reset_index(drop=True)
                    dts = tdf["date"].values
                for idx in oos_idx:
                    if oos_pos[idx] != 0:
                        sig_rows.append({
                            "strategy": STRATEGY_NAMES[sid], "timeframe": tf,
                            "ticker": ticker,
                            "datetime": dts[idx],
                            "position": int(oos_pos[idx]),
                            "log_return": round(log_ret[idx], 8),
                        })
    df_sig = pd.DataFrame(sig_rows)
    if not df_sig.empty:
        df_sig.to_parquet(OUT_DATA / "signals_calibrated_v2.parquet", index=False)
    print(f"[6] signals_calibrated_v2.parquet: {len(df_sig)} rows")

    # ════════════════════════════════════════════════════════════
    # Console output: V1 vs V2 comparison
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("ROLLING RECALIBRATION V2 RESULTS")
    print("=" * 90)

    for tf_label in ["daily", "hourly"]:
        print(f"\n{'─' * 80}")
        print(f"  {tf_label.upper()} STRATEGIES — V1 vs V2 Comparison")
        print(f"{'─' * 80}")
        print(f"  {'Strategy':<18} {'V1 Sharpe':>10} {'V2 Sharpe':>10} {'Delta':>8} "
              f"{'Freq':>5} {'Win':>5} {'N+':>5}")
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*5} {'-'*5} {'-'*5}")
        for sid in STRATEGY_IDS:
            row = [r for r in rows_summary
                   if r["strategy"] == STRATEGY_NAMES[sid] and r["timeframe"] == tf_label]
            if not row:
                continue
            r = row[0]
            v1_sh = V1_SHARPES.get((sid, tf_label), 0)
            v2_sh = r["sharpe_calibrated_median"]
            delta = v2_sh - v1_sh
            print(f"  {r['strategy']:<18} {v1_sh:>+10.4f} {v2_sh:>+10.4f} {delta:>+8.4f} "
                  f"{r['best_freq']:>5} {r['best_window']:>5} {r['n_positive']:>4}/17")

    # New filter analysis
    print(f"\n{'─' * 80}")
    print("  NEW FILTER ANALYSIS (how often new V2 filters were selected)")
    print(f"{'─' * 80}")
    new_filters = {
        "S1": ["ma200_filter", "mean_revert_speed"],
        "S2": ["close_vs_bb_pct", "squeeze_then_expand"],
        "S3": ["momentum_filter", "atr_expansion"],
        "S4": ["momentum_filter", "atr_expansion"],
        "S5": ["intraday_range", "range_bound_check", "target_exit", "pivot_type"],
        "S6": ["intraday_range", "range_bound_check", "vwap_slope_flat"],
    }
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            filter_counts = {}
            total_windows = 0
            for ticker in TICKERS:
                res = all_results[ticker].get((sid, tf, freq, window))
                if res is None:
                    continue
                for ph in res["param_history"]:
                    if not ph["params"]:
                        continue
                    total_windows += 1
                    for filt in new_filters.get(sid, []):
                        val = ph["params"].get(filt)
                        if filt not in filter_counts:
                            filter_counts[filt] = {}
                        sval = str(val)
                        filter_counts[filt][sval] = filter_counts[filt].get(sval, 0) + 1

            if filter_counts and total_windows > 0:
                print(f"\n  {STRATEGY_NAMES[sid]} ({tf}) — {total_windows} total windows:")
                for filt, counts in filter_counts.items():
                    parts = []
                    for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                        pct = cnt / total_windows * 100
                        parts.append(f"{val}={pct:.0f}%")
                    print(f"    {filt:25s}: {', '.join(parts)}")

    # Overfitting check
    print(f"\n{'─' * 80}")
    print("  OVERFITTING CHECK: Train vs OOS Sharpe")
    print(f"{'─' * 80}")
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            train_shs = []
            oos_shs = []
            for ticker in TICKERS:
                res = all_results[ticker].get((sid, tf, freq, window))
                if res is None:
                    continue
                oos_shs.append(res["oos_sharpe"])
                avg_train = np.mean([ph["train_sharpe"] for ph in res["param_history"]])
                train_shs.append(avg_train)
            if train_shs:
                corr = np.corrcoef(train_shs, oos_shs)[0, 1] if len(train_shs) > 2 else 0
                print(f"  {STRATEGY_NAMES[sid]:18s} {tf:7s}  "
                      f"Train={np.median(train_shs):+.3f}  OOS={np.median(oos_shs):+.3f}  "
                      f"corr={corr:+.3f}")

    # Summary: mean delta V2 vs V1
    print(f"\n{'─' * 80}")
    print("  SUMMARY: V2 vs V1 mean Sharpe delta")
    print(f"{'─' * 80}")
    deltas = []
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            row = [r for r in rows_summary
                   if r["strategy"] == STRATEGY_NAMES[sid] and r["timeframe"] == tf]
            if row:
                v1_sh = V1_SHARPES.get((sid, tf), 0)
                v2_sh = row[0]["sharpe_calibrated_median"]
                deltas.append(v2_sh - v1_sh)
    if deltas:
        print(f"  Mean delta: {np.mean(deltas):+.4f}")
        print(f"  Positive deltas: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

    # Final selection
    print(f"\n{'─' * 80}")
    print("  FINAL SELECTION: Sharpe > 0 AND Exposure > 5%")
    print(f"{'─' * 80}")
    selected = []
    for _, r in df_oos.iterrows():
        if r.get("sharpe", 0) > 0 and r.get("exposure_pct", 0) > 5:
            selected.append(r)
    if selected:
        df_sel = pd.DataFrame(selected)
        grouped = df_sel.groupby(["strategy", "timeframe"]).agg(
            sharpe_median=("sharpe", "median"),
            n_pass=("sharpe", "count"),
        ).reset_index()
        for _, r in grouped.iterrows():
            print(f"  {r['strategy']:18s} {r['timeframe']:7s}  "
                  f"Sharpe_med={r['sharpe_median']:+.4f}  pass={int(r['n_pass'])}/17")
    else:
        print("  No strategies passed both criteria.")

    print(f"\nDone in {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")
    main()
