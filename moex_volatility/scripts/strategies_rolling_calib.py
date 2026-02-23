#!/usr/bin/env python3
"""
strategies_rolling_calib.py — Rolling recalibration of 6 strategies
on 2 timeframes × 17 tickers with meta-grid optimization.

6 strategies (renumbered):
  S1: MA Mean Reversion (Contrarian)
  S2: Bollinger Bands (Contrarian)
  S3: Donchian Channel (Trend)
  S4: Supertrend (Trend)
  S5: Pivot Points (Range) — NEW
  S6: VWAP Reversion (Range) — NEW

Rolling recalibration:
  recalib_freq ∈ {63, 126, 252} days
  train_window ∈ {252, 504} days
  Random search: 2000 samples per ticker per window
"""

import warnings
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

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
N_SAMPLES = 1000  # Reduced from 2000: 2000 → ~3.3h, 1000 → ~1.7h
SEED = 42

META_GRID = list(product([63, 126, 252], [252, 504]))  # 6 combos

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

# ════════════════════════════════════════════════════════════
# V3 defaults (fallback if all samples negative)
# ════════════════════════════════════════════════════════════

V3_DEFAULTS = {
    ("S1", "daily"): dict(ma_window=20, z_entry=2.0, z_exit=0.5, max_hold=20,
                          sl_mult=1.5, tp_mult=2.0, rsi_thresh=40,
                          vol_regime=1.2, consec_candles=3, vol_exhaustion=True),
    ("S1", "hourly"): dict(ma_window=20, z_entry=2.0, z_exit=0.5, max_hold=40,
                           sl_mult=1.5, tp_mult=2.0, rsi_thresh=40,
                           vol_regime=1.2, consec_candles=3, vol_exhaustion=True),
    ("S2", "daily"): dict(bb_window=20, bb_std=2.0, max_hold=15,
                          sl_mult=1.5, tp_mult=2.0, bw_percentile=0.25,
                          vol_regime=1.2, consec_candles=3,
                          vol_exhaustion=True, rsi_divergence=True),
    ("S2", "hourly"): dict(bb_window=20, bb_std=2.0, max_hold=30,
                           sl_mult=1.5, tp_mult=2.0, bw_percentile=0.25,
                           vol_regime=1.2, consec_candles=3,
                           vol_exhaustion=True, rsi_divergence=True),
    ("S3", "daily"): dict(dc_window=20, initial_sl_mult=2.5, trail_n=10,
                          trail_atr_mult=2.5, breakeven_thresh=1.5,
                          adx_thresh=20, vol_confirm=1.0),
    ("S3", "hourly"): dict(dc_window=20, initial_sl_mult=2.5, trail_n=20,
                           trail_atr_mult=2.5, breakeven_thresh=1.5,
                           adx_thresh=20, vol_confirm=1.0, daily_ma_confirm=True),
    ("S4", "daily"): dict(atr_period=14, multiplier=3.0, initial_sl_mult=2.5,
                          trail_n=10, trail_atr_mult=2.5, breakeven_thresh=1.5,
                          adx_thresh=20),
    ("S4", "hourly"): dict(atr_period=14, multiplier=3.0, initial_sl_mult=2.5,
                           trail_n=20, trail_atr_mult=2.5, breakeven_thresh=1.5,
                           adx_thresh=20, daily_ma_confirm=True),
    ("S5", "daily"): dict(pivot_type="classic", max_hold=10, sl_mult=1.5,
                          tp_mult=1.5, adx_exit_thresh=30, adx_entry_thresh=20,
                          bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                          vol_compression=0.9),
    ("S5", "hourly"): dict(pivot_type="classic", max_hold=20, sl_mult=1.5,
                           tp_mult=1.5, adx_exit_thresh=30, adx_entry_thresh=20,
                           bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                           vol_compression=0.9),
    ("S6", "daily"): dict(vwap_window=20, dev_mult=2.0, exit_mult=0.5,
                          max_hold=10, sl_mult=1.5, tp_mult=1.5,
                          adx_exit_thresh=30, adx_entry_thresh=20,
                          bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                          vol_compression=0.9, hurst_proxy=True),
    ("S6", "hourly"): dict(vwap_window=20, dev_mult=2.0, exit_mult=0.5,
                           max_hold=20, sl_mult=1.5, tp_mult=1.5,
                           adx_exit_thresh=30, adx_entry_thresh=20,
                           bb_squeeze_pctl=0.30, flat_ma_slope=0.01,
                           vol_compression=0.9, hurst_proxy=True),
}

# ════════════════════════════════════════════════════════════
# Parameter spaces
# ════════════════════════════════════════════════════════════

PARAM_SPACE = {
    ("S1", "daily"): dict(
        ma_window=[15, 20, 25], z_entry=[1.5, 2.0, 2.5], z_exit=[0.3, 0.5, 0.7],
        max_hold=[10, 15, 20, 25], sl_mult=[1.0, 1.5, 2.0],
        tp_mult=[1.5, 2.0, 2.5, 3.0],
        rsi_thresh=[35, 40, 45, None], vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
    ),
    ("S1", "hourly"): dict(
        ma_window=[15, 20, 25], z_entry=[1.5, 2.0, 2.5], z_exit=[0.3, 0.5, 0.7],
        max_hold=[20, 30, 40, 50], sl_mult=[1.0, 1.5, 2.0],
        tp_mult=[1.5, 2.0, 2.5, 3.0],
        rsi_thresh=[35, 40, 45, None], vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
    ),
    ("S2", "daily"): dict(
        bb_window=[15, 20, 25], bb_std=[1.5, 2.0, 2.5], max_hold=[10, 15, 20],
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.5, 2.0, 2.5, 3.0],
        bw_percentile=[0.20, 0.25, 0.33, None], vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
        rsi_divergence=[True, None],
    ),
    ("S2", "hourly"): dict(
        bb_window=[15, 20, 25], bb_std=[1.5, 2.0, 2.5], max_hold=[20, 30, 40],
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.5, 2.0, 2.5, 3.0],
        bw_percentile=[0.20, 0.25, 0.33, None], vol_regime=[1.0, 1.2, 1.5, None],
        consec_candles=[2, 3, 4, None], vol_exhaustion=[True, False],
        rsi_divergence=[True, None],
    ),
    ("S3", "daily"): dict(
        dc_window=[10, 15, 20, 25, 30], initial_sl_mult=[2.0, 2.5, 3.0],
        trail_n=[5, 10, 15, 20, 25], trail_atr_mult=[2.0, 2.5, 3.0],
        breakeven_thresh=[1.0, 1.5, 2.0],
        adx_thresh=[15, 20, 25, None], vol_confirm=[0.8, 1.0, 1.2, None],
    ),
    ("S3", "hourly"): dict(
        dc_window=[10, 15, 20, 25, 30], initial_sl_mult=[2.0, 2.5, 3.0],
        trail_n=[5, 10, 15, 20, 25], trail_atr_mult=[2.0, 2.5, 3.0],
        breakeven_thresh=[1.0, 1.5, 2.0],
        adx_thresh=[15, 20, 25, None], vol_confirm=[0.8, 1.0, 1.2, None],
        daily_ma_confirm=[True, None],
    ),
    ("S4", "daily"): dict(
        atr_period=[10, 14, 20], multiplier=[2.0, 2.5, 3.0, 3.5, 4.0],
        initial_sl_mult=[2.0, 2.5, 3.0], trail_n=[5, 10, 15, 20, 25],
        trail_atr_mult=[2.0, 2.5, 3.0], breakeven_thresh=[1.0, 1.5, 2.0],
        adx_thresh=[15, 20, 25, None],
    ),
    ("S4", "hourly"): dict(
        atr_period=[10, 14, 20], multiplier=[2.0, 2.5, 3.0, 3.5, 4.0],
        initial_sl_mult=[2.0, 2.5, 3.0], trail_n=[5, 10, 15, 20, 25],
        trail_atr_mult=[2.0, 2.5, 3.0], breakeven_thresh=[1.0, 1.5, 2.0],
        adx_thresh=[15, 20, 25, None], daily_ma_confirm=[True, None],
    ),
    ("S5", "daily"): dict(
        pivot_type=["classic", "fibonacci"], max_hold=[5, 8, 10, 15],
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.20, 0.30, 0.40, None],
        flat_ma_slope=[0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
    ),
    ("S5", "hourly"): dict(
        pivot_type=["classic", "fibonacci"], max_hold=[10, 16, 20, 30],
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.20, 0.30, 0.40, None],
        flat_ma_slope=[0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
    ),
    ("S6", "daily"): dict(
        vwap_window=[10, 15, 20, 25], dev_mult=[1.0, 1.5, 2.0],
        exit_mult=[0.3, 0.5, 0.7], max_hold=[5, 8, 10, 15],
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.20, 0.30, 0.40, None],
        flat_ma_slope=[0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
        hurst_proxy=[True, None],
    ),
    ("S6", "hourly"): dict(
        vwap_window=[10, 15, 20, 25], dev_mult=[1.0, 1.5, 2.0],
        exit_mult=[0.3, 0.5, 0.7], max_hold=[10, 16, 20, 30],
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.0, 1.5, 2.0],
        adx_exit_thresh=[25, 30, 35],
        adx_entry_thresh=[15, 20, 25, None],
        bb_squeeze_pctl=[0.20, 0.30, 0.40, None],
        flat_ma_slope=[0.01, 0.02, None],
        vol_compression=[0.8, 0.9, 1.0, None],
        hurst_proxy=[True, None],
    ),
}

# Relevant param keys per strategy (for output)
PARAM_KEYS = {
    "S1": ["ma_window", "z_entry", "z_exit", "max_hold", "sl_mult", "tp_mult",
           "rsi_thresh", "vol_regime", "consec_candles", "vol_exhaustion"],
    "S2": ["bb_window", "bb_std", "max_hold", "sl_mult", "tp_mult",
           "bw_percentile", "vol_regime", "consec_candles", "vol_exhaustion",
           "rsi_divergence"],
    "S3": ["dc_window", "initial_sl_mult", "trail_n", "trail_atr_mult",
           "breakeven_thresh", "adx_thresh", "vol_confirm", "daily_ma_confirm"],
    "S4": ["atr_period", "multiplier", "initial_sl_mult", "trail_n",
           "trail_atr_mult", "breakeven_thresh", "adx_thresh", "daily_ma_confirm"],
    "S5": ["pivot_type", "max_hold", "sl_mult", "tp_mult", "adx_exit_thresh",
           "adx_entry_thresh", "bb_squeeze_pctl", "flat_ma_slope", "vol_compression"],
    "S6": ["vwap_window", "dev_mult", "exit_mult", "max_hold", "sl_mult", "tp_mult",
           "adx_exit_thresh", "adx_entry_thresh", "bb_squeeze_pctl", "flat_ma_slope",
           "vol_compression", "hurst_proxy"],
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

    # Rolling high/low for trail_n
    for tn in [5, 10, 15, 20, 25]:
        ind[f"rh_{tn}"] = pd.Series(high).rolling(tn + 1, min_periods=1).max().values
        ind[f"rl_{tn}"] = pd.Series(low).rolling(tn + 1, min_periods=1).min().values

    # Volume filter
    vs20 = ind["vol_sma20"]
    vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), np.inf, vs20)
    ind["vf_pass"] = ~(~np.isnan(vs20) & (vs20 > 0) & (volume < 0.5 * vs20_safe))
    ind["_volume"] = volume

    # BB width (from standard 20/2.0) for squeeze filter
    sma20 = calc_sma(close, 20)
    std20 = calc_std(close, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    sma20_safe = np.where(np.abs(sma20) > 1e-12, sma20, 1e-12)
    bb_width = (bb_upper - bb_lower) / sma20_safe
    ind["bb_width"] = bb_width

    # BW rolling percentiles for squeeze filter
    bw_win = 252 * 9 if is_hourly else 252
    for pctl in [0.20, 0.25, 0.30, 0.33, 0.40]:
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

    return ind


# ════════════════════════════════════════════════════════════
# Precompute caches (expensive per-param indicators)
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
    P[1:] = (high[:-1] + low[:-1] + close[:-1]) / 3.0
    if variant == "classic":
        S1[1:] = 2 * P[1:] - high[:-1]
        R1[1:] = 2 * P[1:] - low[:-1]
    else:  # fibonacci
        rng = high[:-1] - low[:-1]
        S1[1:] = P[1:] - 0.382 * rng
        R1[1:] = P[1:] + 0.382 * rng
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

    valid = ~np.isnan(z)

    # Filters (None = OFF)
    f_vr = np.ones(len(close), dtype=bool)
    vr_t = params.get("vol_regime")
    if vr_t is not None:
        vr = ind["vol_regime"]
        f_vr = ~np.isnan(vr) & (vr < vr_t)

    f_cc_l = np.ones(len(close), dtype=bool)
    f_cc_s = np.ones(len(close), dtype=bool)
    cc = params.get("consec_candles")
    if cc is not None:
        f_cc_l = ~np.isnan(ind["red_count5"]) & (ind["red_count5"] >= cc)
        f_cc_s = ~np.isnan(ind["green_count5"]) & (ind["green_count5"] >= cc)

    f_ve = np.ones(len(close), dtype=bool)
    if params.get("vol_exhaustion"):
        vs5 = ind["vol_sma5"]
        vol = ind["_volume"]
        vs5_safe = np.where(np.isnan(vs5) | (vs5 <= 0), np.inf, vs5)
        f_ve = ~(~np.isnan(vs5) & (vs5 > 0) & (vol >= vs5_safe))

    f_rsi_l = np.ones(len(close), dtype=bool)
    f_rsi_s = np.ones(len(close), dtype=bool)
    rt = params.get("rsi_thresh")
    if rt is not None:
        f_rsi_l = ~np.isnan(rsi) & (rsi < rt)
        f_rsi_s = ~np.isnan(rsi) & (rsi > (100 - rt))

    vfp = ind["vf_pass"]
    entry_l = valid & (z < -z_ent) & f_vr & f_cc_l & f_ve & f_rsi_l & vfp
    entry_s = valid & (z > z_ent) & f_vr & f_cc_s & f_ve & f_rsi_s & vfp

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

    valid = ~np.isnan(sma)
    entry_l = valid & (close <= bb_lower)
    entry_s = valid & (close >= bb_upper)

    # Filters
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
        # Bullish: close at 5-bar low but RSI not at low
        f_div_l = (~np.isnan(ind["close_min5"]) & ~np.isnan(ind["rsi_min5"]) &
                   ~np.isnan(rsi) & (close <= ind["close_min5"]) &
                   (rsi > ind["rsi_min5"]))
        # Bearish: close at 5-bar high but RSI not at high
        f_div_s = (~np.isnan(ind["close_max5"]) & ~np.isnan(ind["rsi_max5"]) &
                   ~np.isnan(rsi) & (close >= ind["close_max5"]) &
                   (rsi < ind["rsi_max5"]))
        entry_l = entry_l & f_div_l
        entry_s = entry_s & f_div_s

    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    # Exit: close crosses SMA
    n = len(close)
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

    valid = ~np.isnan(prev_hc)

    # ADX filter
    adx_t = params.get("adx_thresh")
    f_adx = np.ones(len(close), dtype=bool)
    if adx_t is not None:
        adx = ind["adx14"]
        f_adx = ~np.isnan(adx) & (adx > adx_t)

    # Vol confirm
    vc = params.get("vol_confirm")
    f_vc = np.ones(len(close), dtype=bool)
    if vc is not None:
        vol = ind["_volume"]
        vs20 = ind["vol_sma20"]
        vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), 1.0, vs20)
        f_vc = (vol / vs20_safe) > vc

    entry_l = valid & (close > prev_hc) & f_adx & f_vc & ind["vf_pass"]
    entry_s = valid & (close < prev_lc) & f_adx & f_vc & ind["vf_pass"]

    # Daily MA confirm (hourly only)
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

    valid = ~np.isnan(st) & ~np.isnan(st_prev)

    adx_t = params.get("adx_thresh")
    f_adx = np.ones(len(close), dtype=bool)
    if adx_t is not None:
        adx = ind["adx14"]
        f_adx = ~np.isnan(adx) & (adx > adx_t)

    entry_l = valid & (close > st) & (c_prev <= st_prev) & f_adx & ind["vf_pass"]
    entry_s = valid & (close < st) & (c_prev >= st_prev) & f_adx & ind["vf_pass"]

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
    """Apply common Range filters (ADX entry, BB squeeze, flat MA, vol compression)."""
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

    return entry_l, entry_s


def gen_s5_signals(close, ind, params, pivot_data):
    """S5 Pivot Points (Range). Returns (sig, exit_arr)."""
    variant = params["pivot_type"]
    P, S1, R1 = pivot_data[variant]

    entry_l = ~np.isnan(S1) & (close < S1)
    entry_s = ~np.isnan(R1) & (close > R1)

    entry_l, entry_s = _apply_range_filters(entry_l, entry_s, ind, params)
    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    # Exit: close crosses P
    n = len(close)
    exit_arr = np.zeros(n, dtype=bool)
    prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
    prev_P = np.empty_like(P); prev_P[0] = P[0]; prev_P[1:] = P[:-1]
    valid_P = ~np.isnan(P) & ~np.isnan(prev_P)
    exit_arr[1:] = valid_P[1:] & (
        ((prev_c[1:] < prev_P[1:]) & (close[1:] >= P[1:])) |
        ((prev_c[1:] > prev_P[1:]) & (close[1:] <= P[1:])))
    return sig, exit_arr


def gen_s6_signals(close, high, low, volume, ind, params, vwap_cache):
    """S6 VWAP Reversion (Range). Returns (sig, exit_arr)."""
    w = int(params["vwap_window"])
    vwap, dev = vwap_cache[w]
    dm = params["dev_mult"]

    entry_l = ~np.isnan(vwap) & (close < vwap - dm * dev)
    entry_s = ~np.isnan(vwap) & (close > vwap + dm * dev)

    entry_l, entry_s = _apply_range_filters(entry_l, entry_s, ind, params)

    if params.get("hurst_proxy"):
        h = ind["hurst20"]
        f = ~np.isnan(h) & (h < 0.45)
        entry_l = entry_l & f
        entry_s = entry_s & f

    entry_l = entry_l & ind["vf_pass"]
    entry_s = entry_s & ind["vf_pass"]

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0

    # Exit: |close - VWAP| < exit_mult * dev
    em = params["exit_mult"]
    exit_arr = ~np.isnan(vwap) & (np.abs(close - vwap) < em * dev)
    return sig, exit_arr


# ════════════════════════════════════════════════════════════
# Backtest engines
# ════════════════════════════════════════════════════════════

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


def bt_trend(sig_arr, exit_long, exit_short, close, high, low, atr14,
             rolling_high, rolling_low, isl_mult, trail_mult, be_thresh,
             warmup, end_idx):
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = np.nan; ep = 0.0; ea = 0.0; be = False

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
        return bt_trend(sig, exit_l, exit_s, close, high, low, ind["atr14"],
                        rh, rl, params["initial_sl_mult"], params["trail_atr_mult"],
                        params["breakeven_thresh"], warmup, end_idx)
    elif cat == "Range":
        return bt_range(sig, exit_info, close, high, low, ind["atr14"],
                        ind["adx14"], params["sl_mult"], params["tp_mult"],
                        params["adx_exit_thresh"], int(params["max_hold"]),
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
    """Rolling recalibration for one strategy × one ticker × one meta combo."""
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
    """Process all strategies × timeframes × meta combos for one ticker."""
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

    # Precompute caches — daily
    sma_cache_d = precompute_sma_cache(close_d, [15, 20, 25])
    dc_cache_d = precompute_donchian_cache(high_d, low_d, [10, 15, 20, 25, 30])
    st_cache_d = precompute_supertrend_cache(high_d, low_d, close_d,
                                              [10, 14, 20], [2.0, 2.5, 3.0, 3.5, 4.0])
    vwap_cache_d = precompute_vwap_cache(close_d, high_d, low_d, vol_d, [10, 15, 20, 25])

    # Precompute caches — hourly
    sma_cache_h = precompute_sma_cache(close_h, [15, 20, 25])
    dc_cache_h = precompute_donchian_cache(high_h, low_h, [10, 15, 20, 25, 30])
    st_cache_h = precompute_supertrend_cache(high_h, low_h, close_h,
                                              [10, 14, 20], [2.0, 2.5, 3.0, 3.5, 4.0])
    vwap_cache_h = precompute_vwap_cache(close_h, high_h, low_h, vol_h, [10, 15, 20, 25])

    # Daily trend for S3h, S4h
    d_dates, d_above = build_daily_trend(daily, ticker)
    daily_trend_h = align_daily_to_hourly(d_dates, d_above, dts_h)

    # Pivot data — daily
    pivot_d = {
        "classic": calc_pivot_daily(high_d, low_d, close_d, "classic"),
        "fibonacci": calc_pivot_daily(high_d, low_d, close_d, "fibonacci"),
    }
    # Pivot data — hourly
    pivot_h = {
        "classic": compute_daily_pivots_for_hourly(daily, ticker, dts_h, "classic"),
        "fibonacci": compute_daily_pivots_for_hourly(daily, ticker, dts_h, "fibonacci"),
    }

    combo_idx = 0
    total_combos = len(STRATEGY_IDS) * len(TIMEFRAMES) * len(META_GRID)
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
                combo_idx += 1
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
    t0 = time.time()
    print("Loading data...")
    daily, hourly = load_data()
    print(f"  Daily: {len(daily)} rows, Hourly: {len(hourly)} rows")

    # Process tickers sequentially
    all_results = {}
    for i, ticker in enumerate(TICKERS):
        tt = time.time()
        print(f"  [{i+1}/{len(TICKERS)}] {ticker}...", end="", flush=True)
        all_results[ticker] = process_ticker(ticker, daily, hourly)
        elapsed = time.time() - tt
        print(f" {elapsed:.1f}s", flush=True)

    total_time = time.time() - t0
    print(f"\nTotal processing: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Output 1: calib_optimal_params.csv ──
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
    df_params.to_csv(OUT_TABLES / "calib_optimal_params.csv", index=False)
    print(f"\n[1] calib_optimal_params.csv: {len(df_params)} rows")

    # ── Output 2: calib_meta_comparison.csv ──
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
    df_meta.to_csv(OUT_TABLES / "calib_meta_comparison.csv", index=False)
    print(f"[2] calib_meta_comparison.csv: {len(df_meta)} rows")

    # ── Select best meta per strategy×TF ──
    best_metas = {}
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = select_best_meta(all_results, sid, tf)
            best_metas[(sid, tf)] = bm

    # ── Output 3: calib_oos_results.csv ──
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
    df_oos.to_csv(OUT_TABLES / "calib_oos_results.csv", index=False)
    print(f"[3] calib_oos_results.csv: {len(df_oos)} rows")

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
                sc = precompute_sma_cache(c, [15, 20, 25])
                dc = precompute_donchian_cache(h, l, [10, 15, 20, 25, 30])
                stc = precompute_supertrend_cache(h, l, c, [10, 14, 20], [2.0, 2.5, 3.0, 3.5, 4.0])
                vc = precompute_vwap_cache(c, h, l, v, [10, 15, 20, 25])

                if is_h:
                    d_dates, d_above = build_daily_trend(daily, ticker)
                    daily_tr = align_daily_to_hourly(d_dates, d_above, dt)
                    pv = {
                        "classic": compute_daily_pivots_for_hourly(daily, ticker, dt, "classic"),
                        "fibonacci": compute_daily_pivots_for_hourly(daily, ticker, dt, "fibonacci"),
                    }
                else:
                    daily_tr = None
                    pv = {
                        "classic": calc_pivot_daily(h, l, c, "classic"),
                        "fibonacci": calc_pivot_daily(h, l, c, "fibonacci"),
                    }

                sh = run_v3_defaults_one(sid, tf, c, h, l, v, ind, oos_mask,
                                          sc, dc, stc, vc, pv, daily_tr, is_h)
                ticker_sharpes.append(sh)
            if ticker_sharpes:
                default_sharpes[(sid, tf)] = np.median(ticker_sharpes)

    # ── Output 4: calib_summary.csv ──
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
    df_summary.to_csv(OUT_TABLES / "calib_summary.csv", index=False)
    print(f"[4] calib_summary.csv: {len(df_summary)} rows")

    # ── Output 5: calib_param_stability.csv ──
    rows_stab = []
    for sid in STRATEGY_IDS:
        for tf in TIMEFRAMES:
            bm = best_metas.get((sid, tf))
            if bm is None:
                continue
            freq, window = bm
            # Collect all params across tickers and recalib windows
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
                # Convert to strings for counting
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
    df_stab.to_csv(OUT_TABLES / "calib_param_stability.csv", index=False)
    print(f"[5] calib_param_stability.csv: {len(df_stab)} rows")

    # ── Output 6: signals_calibrated.parquet ──
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
        df_sig.to_parquet(OUT_DATA / "signals_calibrated.parquet", index=False)
    print(f"[6] signals_calibrated.parquet: {len(df_sig)} rows")

    # ── Output 7: Console print ──
    print("\n" + "=" * 80)
    print("ROLLING RECALIBRATION RESULTS")
    print("=" * 80)

    for tf_label in ["daily", "hourly"]:
        print(f"\n{'─' * 60}")
        print(f"  {tf_label.upper()} STRATEGIES")
        print(f"{'─' * 60}")
        print(f"  {'Strategy':<18} {'Default':>8} {'Calibr':>8} {'Delta%':>8} "
              f"{'Freq':>5} {'Win':>5} {'N+':>4}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*4}")
        for sid in STRATEGY_IDS:
            row = [r for r in rows_summary
                   if r["strategy"] == STRATEGY_NAMES[sid] and r["timeframe"] == tf_label]
            if not row:
                continue
            r = row[0]
            print(f"  {r['strategy']:<18} {r['sharpe_default_median']:>8.4f} "
                  f"{r['sharpe_calibrated_median']:>8.4f} {r['delta_pct']:>+7.1f}% "
                  f"{r['best_freq']:>5} {r['best_window']:>5} {r['n_positive']:>4}/17")

    # Overfitting check
    print(f"\n{'─' * 60}")
    print("  OVERFITTING CHECK: Train vs OOS Sharpe")
    print(f"{'─' * 60}")
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

    # Param stability highlights
    print(f"\n{'─' * 60}")
    print("  PARAM STABILITY (top params by stability)")
    print(f"{'─' * 60}")
    if not df_stab.empty:
        top = df_stab.sort_values("stability_pct", ascending=False).head(15)
        for _, r in top.iterrows():
            print(f"  {r['strategy']:18s} {r['timeframe']:7s}  "
                  f"{r['param_name']:20s} = {r['mode_value']:10s}  "
                  f"({r['stability_pct']:.0f}%)")

    # Final selection
    print(f"\n{'─' * 60}")
    print("  FINAL SELECTION: Sharpe > 0 AND Exposure > 5%")
    print(f"{'─' * 60}")
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

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
