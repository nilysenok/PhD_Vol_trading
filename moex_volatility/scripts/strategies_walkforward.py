#!/usr/bin/env python3
"""
strategies_walkforward.py — Walk-Forward A/B/C/D Strategy Pipeline (V3)

A: Baseline (no predictions) - expanding WF, full grid, top-10 ensemble
   Enhanced RM: breakeven, cooldown, partial exit, trail types, time decay
B: Adaptive sigma_pred stops (with horizon selection, dynamic hold, breakeven)
C: Regime filter using sigma_pred (direction, term structure, hysteresis)
D: Vol-targeting using sigma_pred (power scaling, smoothing, inverse vol)

Hourly sigma scaling: v2_sqrtN as default (no variant loop).
Commission = 0.0 (gross returns).
"""
import os, warnings, time, sys
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

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

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results" / "final" / "strategies" / "data"
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v3"
OUT_TABLES = OUT_DIR / "tables"
OUT_DATA = OUT_DIR / "data"

TICKERS = sorted([
    "AFLT","ALRS","HYDR","IRAO","LKOH","LSRG","MGNT","MOEX",
    "MTLR","MTSS","NVTK","OGKB","PHOR","RTKM","SBER","TATN","VTBR"
])

WARMUP = 200
N_WORKERS = min(cpu_count(), 8)
COMMISSION = float(os.environ.get("WF_COMMISSION", "0.0"))

# C approach exposure/trades constraints
C_MIN_EXPOSURE = 5.0        # strict: min exposure % to accept
C_MIN_TRADES_YR = 10.0      # strict: min trades/year to accept
C_FALLBACK_EXPOSURE = 3.0   # relaxed fallback
C_FALLBACK_TRADES_YR = 6.0  # relaxed fallback
EXIT_REASON_NAMES = {0:'SL', 1:'TP', 2:'max_hold', 3:'signal', 4:'breakeven',
                     5:'time_decay', 6:'ADX_exit', 7:'trail_SL', 8:'end_data', 9:'position'}

STRATEGY_IDS = ["S1","S2","S3","S4","S5","S6"]
TIMEFRAMES = ["daily","hourly"]
STRATEGY_NAMES = {
    "S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
    "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP",
}
CATEGORY = {
    "S1":"Contrarian","S2":"Contrarian",
    "S3":"Trend","S4":"Trend",
    "S5":"Range","S6":"Range",
}

A_TEST_YEARS = list(range(2020, 2027))
BCD_TEST_YEARS = list(range(2022, 2027))

# ═══════════════════════════════════════════════════════════════════
# Signal parameter grids
# ═══════════════════════════════════════════════════════════════════
SIGNAL_GRIDS = {
    "S1": dict(ma_window=[15,20,25], z_entry=[1.5,2.0,2.5], z_exit=[0.3,0.5,0.7]),
    "S2": dict(bb_window=[15,20,25], bb_std=[1.5,2.0,2.5]),
    "S3": dict(dc_window=[10,15,20,25]),
    "S4": dict(atr_period=[10,14,20], multiplier=[2.0,2.5,3.0,3.5]),
    "S5": {},
    "S6": dict(vwap_window=[10,15,20], dev_mult=[1.0,1.5,2.0]),
}

# ═══════════════════════════════════════════════════════════════════
# A: RM grids (V3 — enhanced with breakeven, cooldown, etc.)
# ═══════════════════════════════════════════════════════════════════
RM_GRIDS_V3 = {
    ("Contrarian", "daily"): dict(
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.5, 2.0, 2.5], max_hold=[10, 15, 20],
        breakeven_trigger=[None, 1.0],
        cooldown_bars=[0, 5],
        partial_exit=[False, True]),
    ("Contrarian", "hourly"): dict(
        sl_mult=[1.0, 1.5, 2.0], tp_mult=[1.5, 2.0, 2.5], max_hold=[20, 30, 40],
        breakeven_trigger=[None, 1.0],
        cooldown_bars=[0, 5],
        partial_exit=[False, True]),
    ("Trend", "daily"): "special",   # built separately
    ("Trend", "hourly"): "special",
    ("Range", "daily"): dict(
        sl_mult=[0.75, 1.0, 1.5], tp_mult=[0.75, 1.0, 1.5], max_hold=[5, 8, 10],
        breakeven_trigger=[None, 1.0],
        cooldown_bars=[0, 5],
        time_decay=[False, True]),
    ("Range", "hourly"): dict(
        sl_mult=[0.75, 1.0, 1.5], tp_mult=[0.75, 1.0, 1.5], max_hold=[10, 16, 20],
        breakeven_trigger=[None, 1.0],
        cooldown_bars=[0, 5],
        time_decay=[False, True]),
}

def _build_trend_rm_grid():
    """Build Trend RM grid with special handling for trail types."""
    grid = []
    for isl in [2.0, 2.5, 3.0]:
        for be in [None, 1.0]:
            for cd in [0, 5]:
                # fixed_atr & chandelier
                for tt in ["fixed_atr", "chandelier"]:
                    for tn in [5, 10, 15, 20]:
                        for tam in [2.0, 2.5, 3.0]:
                            grid.append(dict(
                                initial_sl_mult=isl, trail_type=tt,
                                trail_n=tn, trail_atr_mult=tam,
                                breakeven_thresh=be, cooldown_bars=cd,
                                parabolic_step=0.02, parabolic_max=0.15))
                # parabolic_step
                for ps in [0.01, 0.02, 0.03]:
                    for pm in [0.10, 0.15, 0.20]:
                        grid.append(dict(
                            initial_sl_mult=isl, trail_type="parabolic_step",
                            trail_n=10, trail_atr_mult=2.5,
                            breakeven_thresh=be, cooldown_bars=cd,
                            parabolic_step=ps, parabolic_max=pm))
    return grid

TREND_RM_GRID = _build_trend_rm_grid()

# ═══════════════════════════════════════════════════════════════════
# B grids (V3)
# ═══════════════════════════════════════════════════════════════════
B_GRIDS_V3 = {
    "Contrarian": dict(
        k_sl=[0.5, 0.75, 1.0, 1.5, 2.0],
        ratio=[0.5, 1.0, 1.5, 2.0, 3.0],
        k_be=[None, 0.5, 1.0, 1.5, 2.0],
        gamma_hold=[None, 0.5, 1.0],
        horizon=["h1", "h5", "h22"]),
    "Trend": dict(
        k_sl=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        k_trail=[0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
        k_be=[None, 0.5, 1.0, 1.5, 2.0],
        gamma_hold=[None, 0.5, 1.0],
        horizon=["h1", "h5", "h22"]),
    "Range": dict(
        k_sl=[0.5, 0.75, 1.0, 1.5, 2.0],
        ratio=[0.5, 1.0, 1.5, 2.0, 3.0],
        k_be=[None, 0.5, 1.0, 1.5, 2.0],
        gamma_hold=[None, 0.5, 1.0],
        horizon=["h1", "h5", "h22"]),
}

# ═══════════════════════════════════════════════════════════════════
# C grids (V3)
# ═══════════════════════════════════════════════════════════════════
C_THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
C_RANGE_BANDS = [(0.3,0.7), (0.25,0.75), (0.2,0.8),
                 (0.35,0.65), (0.15,0.85), (0.4,0.6), (0.1,0.9)]
C_LOOKBACKS = [126, 252, 504]
C_HORIZONS = ["h1", "h5", "h22"]
C_DIRECTION = ["OFF", "rise", "fall"]
C_TERM_FILTER = [False, True]
C_HYSTERESIS = [0, 10]

# ═══════════════════════════════════════════════════════════════════
# D grids (V3)
# ═══════════════════════════════════════════════════════════════════
D_TARGET_VOLS = [0.05, 0.10, 0.15, 0.20, 0.30]
D_MAX_LEVS = [1.0, 2.0, 3.0]
D_GAMMA = [0.5, 1.0, 1.5]
D_VOL_FLOORS = [None, 0.10]
D_VOL_CAPS = [None, 0.60]
D_SMOOTH = [None, 5, 20]
D_HORIZONS = ["h1", "h5", "h22"]
D_INV_LOOKBACKS = [63, 126, 252]


def expand_grid(d):
    """Expand dict of lists to list of dicts (cartesian product)."""
    if not d:
        return [{}]
    keys = sorted(d.keys())
    vals = [d[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Indicator Functions
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Section 2 — compute_base
# ═══════════════════════════════════════════════════════════════════

def compute_base(close, high, low, open_arr, volume, is_hourly=False):
    ind = {}
    n = len(close)
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

    for tn in [5, 10, 15, 20]:
        ind[f"rh_{tn}"] = pd.Series(high).rolling(tn + 1, min_periods=1).max().values
        ind[f"rl_{tn}"] = pd.Series(low).rolling(tn + 1, min_periods=1).min().values

    vs20 = ind["vol_sma20"]
    vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), np.inf, vs20)
    ind["vf_pass"] = ~(~np.isnan(vs20) & (vs20 > 0) & (volume < 0.5 * vs20_safe))
    ind["_volume"] = volume

    sma20 = calc_sma(close, 20)
    std20 = calc_std(close, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    sma20_safe = np.where(np.abs(sma20) > 1e-12, sma20, 1e-12)
    ind["bb_width"] = (bb_upper - bb_lower) / sma20_safe
    bw_win = 252 * 9 if is_hourly else 252
    ind["bw_p30"] = pd.Series(ind["bb_width"]).rolling(bw_win, min_periods=50).quantile(0.30).values

    sma20_lag10 = np.full(n, np.nan)
    sma20_lag10[10:] = sma20[:-10]
    sma20_lag_safe = np.where(np.abs(sma20_lag10) > 1e-12, sma20_lag10, 1e-12)
    ind["sma20_slope10"] = (sma20 - sma20_lag10) / sma20_lag_safe

    return ind


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Precompute Caches + Pivot + Daily Trend
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Signal Generators
# ═══════════════════════════════════════════════════════════════════

def gen_s1_signals(close, ind, sma_cache, *, ma_window, z_entry, z_exit):
    sma, std = sma_cache[ma_window]
    z = (close - sma) / std
    n = len(close)
    valid = ~np.isnan(z)
    vr = ind["vol_regime"]
    f_vr = ~np.isnan(vr) & (vr < 1.2)
    f_cc_l = ~np.isnan(ind["red_count5"]) & (ind["red_count5"] >= 3)
    f_cc_s = ~np.isnan(ind["green_count5"]) & (ind["green_count5"] >= 3)
    vs5 = ind["vol_sma5"]; vol = ind["_volume"]
    vs5_safe = np.where(np.isnan(vs5) | (vs5 <= 0), np.inf, vs5)
    f_ve = ~(~np.isnan(vs5) & (vs5 > 0) & (vol >= vs5_safe))
    vfp = ind["vf_pass"]
    entry_l = valid & (z < -z_entry) & f_vr & f_cc_l & f_ve & vfp
    entry_s = valid & (z > z_entry) & f_vr & f_cc_s & f_ve & vfp
    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_arr = ~np.isnan(z) & (np.abs(z) < z_exit)
    return sig, exit_arr

def gen_s2_signals(close, ind, sma_cache, *, bb_window, bb_std):
    sma, std = sma_cache[bb_window]
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std
    n = len(close)
    valid = ~np.isnan(sma)
    entry_l = valid & (close <= bb_lower)
    entry_s = valid & (close >= bb_upper)
    vr = ind["vol_regime"]
    f_vr = ~np.isnan(vr) & (vr < 1.2)
    f_cc_l = ~np.isnan(ind["red_count5"]) & (ind["red_count5"] >= 3)
    f_cc_s = ~np.isnan(ind["green_count5"]) & (ind["green_count5"] >= 3)
    vs5 = ind["vol_sma5"]; vol = ind["_volume"]
    vs5_safe = np.where(np.isnan(vs5) | (vs5 <= 0), np.inf, vs5)
    f_ve = ~(~np.isnan(vs5) & (vs5 > 0) & (vol >= vs5_safe))
    vfp = ind["vf_pass"]
    entry_l = entry_l & f_vr & f_cc_l & f_ve & vfp
    entry_s = entry_s & f_vr & f_cc_s & f_ve & vfp
    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_arr = np.zeros(n, dtype=bool)
    prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
    prev_m = np.empty_like(sma); prev_m[0] = sma[0]; prev_m[1:] = sma[:-1]
    exit_arr[1:] = (((prev_c[1:] < prev_m[1:]) & (close[1:] >= sma[1:])) |
                    ((prev_c[1:] > prev_m[1:]) & (close[1:] <= sma[1:])))
    return sig, exit_arr

def gen_s3_signals(close, high, low, ind, dc_cache, *, dc_window, daily_trend=None):
    hc, lc = dc_cache[dc_window]
    prev_hc = np.empty_like(hc); prev_hc[0] = np.nan; prev_hc[1:] = hc[:-1]
    prev_lc = np.empty_like(lc); prev_lc[0] = np.nan; prev_lc[1:] = lc[:-1]
    n = len(close)
    valid = ~np.isnan(prev_hc)
    adx = ind["adx14"]
    f_adx = ~np.isnan(adx) & (adx > 20)
    vol = ind["_volume"]; vs20 = ind["vol_sma20"]
    vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), 1.0, vs20)
    f_vc = (vol / vs20_safe) > 1.0
    entry_l = valid & (close > prev_hc) & f_adx & f_vc & ind["vf_pass"]
    entry_s = valid & (close < prev_lc) & f_adx & f_vc & ind["vf_pass"]
    if daily_trend is not None:
        dt_valid = ~np.isnan(daily_trend)
        entry_l = entry_l & (~dt_valid | (daily_trend >= 0.5))
        entry_s = entry_s & (~dt_valid | (daily_trend < 0.5))
    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_l = ~np.isnan(prev_lc) & (close < prev_lc)
    exit_s = ~np.isnan(prev_hc) & (close > prev_hc)
    return sig, (exit_l, exit_s)

def gen_s4_signals(close, high, low, ind, st_cache, *, atr_period, multiplier, daily_trend=None):
    st, _ = st_cache[(atr_period, multiplier)]
    st_prev = np.empty_like(st); st_prev[0] = np.nan; st_prev[1:] = st[:-1]
    c_prev = np.empty_like(close); c_prev[0] = close[0]; c_prev[1:] = close[:-1]
    n = len(close)
    valid = ~np.isnan(st) & ~np.isnan(st_prev)
    adx = ind["adx14"]
    f_adx = ~np.isnan(adx) & (adx > 20)
    entry_l = valid & (close > st) & (c_prev <= st_prev) & f_adx & ind["vf_pass"]
    entry_s = valid & (close < st) & (c_prev >= st_prev) & f_adx & ind["vf_pass"]
    if daily_trend is not None:
        dt_valid = ~np.isnan(daily_trend)
        entry_l = entry_l & (~dt_valid | (daily_trend >= 0.5))
        entry_s = entry_s & (~dt_valid | (daily_trend < 0.5))
    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_l = valid & (close < st) & (c_prev >= st_prev)
    exit_s = valid & (close > st) & (c_prev <= st_prev)
    return sig, (exit_l, exit_s)

def _range_filter(entry_l, entry_s, ind):
    adx = ind["adx14"]
    f_adx = ~np.isnan(adx) & (adx < 25)
    entry_l = entry_l & f_adx; entry_s = entry_s & f_adx
    bw = ind["bb_width"]; bw_t = ind["bw_p30"]
    f_bw = ~np.isnan(bw_t) & (bw < bw_t)
    entry_l = entry_l & f_bw; entry_s = entry_s & f_bw
    slope = ind["sma20_slope10"]
    f_slope = ~np.isnan(slope) & (np.abs(slope) < 0.01)
    entry_l = entry_l & f_slope; entry_s = entry_s & f_slope
    vr = ind["vol_regime"]
    f_vr = ~np.isnan(vr) & (vr < 0.9)
    entry_l = entry_l & f_vr; entry_s = entry_s & f_vr
    return entry_l, entry_s

def gen_s5_signals(close, ind, pivot_data):
    P, S1p, R1p = pivot_data["classic"]
    n = len(close)
    entry_l = ~np.isnan(S1p) & (close < S1p)
    entry_s = ~np.isnan(R1p) & (close > R1p)
    entry_l, entry_s = _range_filter(entry_l, entry_s, ind)
    entry_l = entry_l & ind["vf_pass"]; entry_s = entry_s & ind["vf_pass"]
    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_arr = np.zeros(n, dtype=bool)
    prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
    prev_P = np.empty_like(P); prev_P[0] = P[0]; prev_P[1:] = P[:-1]
    valid_P = ~np.isnan(P) & ~np.isnan(prev_P)
    exit_arr[1:] = valid_P[1:] & (
        ((prev_c[1:] < prev_P[1:]) & (close[1:] >= P[1:])) |
        ((prev_c[1:] > prev_P[1:]) & (close[1:] <= P[1:])))
    return sig, exit_arr

def gen_s6_signals(close, high, low, volume, ind, vwap_cache, *, vwap_window, dev_mult):
    vwap, dev = vwap_cache[vwap_window]
    n = len(close)
    entry_l = ~np.isnan(vwap) & (close < vwap - dev_mult * dev)
    entry_s = ~np.isnan(vwap) & (close > vwap + dev_mult * dev)
    entry_l, entry_s = _range_filter(entry_l, entry_s, ind)
    entry_l = entry_l & ind["vf_pass"]; entry_s = entry_s & ind["vf_pass"]
    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_arr = ~np.isnan(vwap) & (np.abs(close - vwap) < 0.5 * dev)
    return sig, exit_arr


# ═══════════════════════════════════════════════════════════════════
# Section 5a — ATR Backtest Engines V3
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def bt_contrarian_v3(sig_arr, exit_arr, close, high, low, atr14,
                     sl_mult, tp_mult, max_hold,
                     be_trigger, cooldown_bars, partial_exit,
                     warmup, end_idx, log_trades):
    """Contrarian V3: breakeven, cooldown, partial exit."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0
    ep = 0.0; ea = 0.0; be_active = False
    half_exited = False; orig_tp_dist = 0.0
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

            # Breakeven check
            if be_trigger > 0.0 and not be_active:
                if cp == 1.0 and (close[t] - ep) > be_trigger * ea:
                    be_active = True
                    sl = ep
                elif cp == -1.0 and (ep - close[t]) > be_trigger * ea:
                    be_active = True
                    sl = ep

            if cp == 1.0:
                if low[t] <= sl:
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                    closed = True; was_sl_exit = True
                elif high[t] >= tp:
                    # Partial exit check
                    if partial_exit > 0 and not half_exited:
                        half_tp = ep + 0.5 * orig_tp_dist
                        if high[t] >= half_tp and high[t] < tp:
                            half_exited = True
                            cp = 0.5
                        elif high[t] >= tp:
                            exit_reason = 1
                            closed = True
                    else:
                        exit_reason = 1
                        closed = True
            else:
                if high[t] >= sl:
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                    closed = True; was_sl_exit = True
                elif low[t] <= tp:
                    if partial_exit > 0 and not half_exited:
                        half_tp = ep - 0.5 * orig_tp_dist
                        if low[t] <= half_tp and low[t] > tp:
                            half_exited = True
                            cp = -0.5
                        elif low[t] <= tp:
                            exit_reason = 1
                            closed = True
                    else:
                        exit_reason = 1
                        closed = True

            if not closed and exit_arr[t]:
                exit_reason = 3
                closed = True
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
                half_exited = False

        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0; ep = close[t]; ea = a; be_active = False
                sl = close[t] - sl_mult * a
                tp = close[t] + tp_mult * a
                orig_tp_dist = tp_mult * a
                held = 0; half_exited = False
                entry_bar_t = t
                direction_t = 1
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0; ep = close[t]; ea = a; be_active = False
                sl = close[t] + sl_mult * a
                tp = close[t] - tp_mult * a
                orig_tp_dist = tp_mult * a
                held = 0; half_exited = False
                entry_bar_t = t
                direction_t = -1
        pos[t] = cp
    return pos, trades[:nt]


@njit(cache=True)
def bt_trend_v3(sig_arr, exit_long, exit_short, close, high, low, atr14,
                rolling_high, rolling_low,
                isl_mult, trail_mult, be_thresh,
                trail_type, parabolic_step, parabolic_max,
                cooldown_bars, warmup, end_idx, log_trades):
    """Trend V3: trail types (0=fixed_atr, 1=chandelier, 2=parabolic), cooldown."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; ep = 0.0; ea = 0.0; be = False
    sl = np.nan
    cooldown_rem = 0; was_sl_exit = False
    # Parabolic state
    sar_step = 0.0; sar_max = 0.0; extreme_p = 0.0

    max_t = n // 2 + 1 if log_trades else 1
    trades = np.empty((max_t, 7))
    nt = 0
    entry_bar_t = 0
    direction_t = 0
    held = 0

    for t in range(warmup, n):
        if cooldown_rem > 0:
            cooldown_rem -= 1

        if cp != 0.0:
            held += 1
            ca = atr14[t]
            if ca != ca: ca = ea
            closed = False
            was_sl_exit = False
            exit_reason = 0

            if cp == 1.0:
                # Compute trail SL based on type
                if trail_type == 0:  # fixed_atr
                    tsl = rolling_high[t] - trail_mult * ca
                elif trail_type == 1:  # chandelier
                    tsl = rolling_high[t] - trail_mult * ca
                else:  # parabolic_step
                    if high[t] > extreme_p:
                        extreme_p = high[t]
                        sar_step = min(sar_step + parabolic_step, parabolic_max)
                    if sl != sl:
                        tsl = ep - isl_mult * ea
                    else:
                        tsl = sl + sar_step * (extreme_p - sl)

                # Breakeven
                if not be and be_thresh < 1e10:
                    if (close[t] - ep) > be_thresh * ea:
                        be = True
                if be:
                    tsl = max(tsl, ep)

                if sl != sl: sl = tsl
                else: sl = max(sl, tsl)

                if low[t] <= sl:
                    exit_reason = 7
                    closed = True; was_sl_exit = True
                elif exit_long[t]:
                    exit_reason = 3
                    closed = True
            else:
                if trail_type == 0:
                    tsl = rolling_low[t] + trail_mult * ca
                elif trail_type == 1:
                    tsl = rolling_low[t] + trail_mult * ca
                else:
                    if low[t] < extreme_p:
                        extreme_p = low[t]
                        sar_step = min(sar_step + parabolic_step, parabolic_max)
                    if sl != sl:
                        tsl = ep + isl_mult * ea
                    else:
                        tsl = sl - sar_step * (sl - extreme_p)

                if not be and be_thresh < 1e10:
                    if (ep - close[t]) > be_thresh * ea:
                        be = True
                if be:
                    tsl = min(tsl, ep)

                if sl != sl: sl = tsl
                else: sl = min(sl, tsl)

                if high[t] >= sl:
                    exit_reason = 7
                    closed = True; was_sl_exit = True
                elif exit_short[t]:
                    exit_reason = 3
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
                cp = 0.0; sl = np.nan; ep = 0.0; ea = 0.0; be = False
                held = 0

        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0; ep = close[t]; ea = a; be = False
                sl = ep - isl_mult * a
                extreme_p = high[t]
                sar_step = parabolic_step; sar_max = parabolic_max
                entry_bar_t = t
                direction_t = 1
                held = 0
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0; ep = close[t]; ea = a; be = False
                sl = ep + isl_mult * a
                extreme_p = low[t]
                sar_step = parabolic_step; sar_max = parabolic_max
                entry_bar_t = t
                direction_t = -1
                held = 0
        pos[t] = cp
    return pos, trades[:nt]


@njit(cache=True)
def bt_range_v3(sig_arr, exit_arr, close, high, low, atr14, adx14,
                sl_mult, tp_mult, adx_exit_thresh, max_hold,
                be_trigger, cooldown_bars, time_decay,
                warmup, end_idx, log_trades):
    """Range V3: breakeven, cooldown, time decay."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0
    ep = 0.0; ea = 0.0; be_active = False
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
            if be_trigger > 0.0 and not be_active:
                if cp == 1.0 and (close[t] - ep) > be_trigger * ea:
                    be_active = True; sl = ep
                elif cp == -1.0 and (ep - close[t]) > be_trigger * ea:
                    be_active = True; sl = ep

            # Time decay: narrow TP after half max_hold
            td_active = False
            if time_decay > 0 and held > max_hold // 2:
                td_active = True
                if cp == 1.0:
                    tp = ep + 0.5 * orig_tp_dist
                else:
                    tp = ep - 0.5 * orig_tp_dist

            if cp == 1.0:
                if low[t] <= sl:
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                    closed = True; was_sl_exit = True
                elif high[t] >= tp:
                    if td_active:
                        exit_reason = 5
                    else:
                        exit_reason = 1
                    closed = True
            else:
                if high[t] >= sl:
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                    closed = True; was_sl_exit = True
                elif low[t] <= tp:
                    if td_active:
                        exit_reason = 5
                    else:
                        exit_reason = 1
                    closed = True
            if not closed:
                av = adx14[t]
                if av == av and av > adx_exit_thresh:
                    exit_reason = 6
                    closed = True
            if not closed and exit_arr[t]:
                exit_reason = 3
                closed = True
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
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0; ep = close[t]; ea = a; be_active = False
                sl = close[t] - sl_mult * a
                tp = close[t] + tp_mult * a
                orig_tp_dist = tp_mult * a
                held = 0
                entry_bar_t = t
                direction_t = 1
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0; ep = close[t]; ea = a; be_active = False
                sl = close[t] + sl_mult * a
                tp = close[t] - tp_mult * a
                orig_tp_dist = tp_mult * a
                held = 0
                entry_bar_t = t
                direction_t = -1
        pos[t] = cp
    return pos, trades[:nt]


@njit(cache=True)
def bt_range_split_v3(sig_arr, exit_l, exit_s, close, high, low, atr14, adx14,
                      sl_mult, tp_mult, adx_exit_thresh, max_hold,
                      be_trigger, cooldown_bars, time_decay,
                      warmup, end_idx, log_trades):
    """Range V3 with split exit arrays."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0
    ep = 0.0; ea = 0.0; be_active = False
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

            if be_trigger > 0.0 and not be_active:
                if cp == 1.0 and (close[t] - ep) > be_trigger * ea:
                    be_active = True; sl = ep
                elif cp == -1.0 and (ep - close[t]) > be_trigger * ea:
                    be_active = True; sl = ep

            td_active = False
            if time_decay > 0 and held > max_hold // 2:
                td_active = True
                if cp == 1.0:
                    tp = ep + 0.5 * orig_tp_dist
                else:
                    tp = ep - 0.5 * orig_tp_dist

            if cp == 1.0:
                if low[t] <= sl:
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                    closed = True; was_sl_exit = True
                elif high[t] >= tp:
                    if td_active:
                        exit_reason = 5
                    else:
                        exit_reason = 1
                    closed = True
                if not closed:
                    av = adx14[t]
                    if av == av and av > adx_exit_thresh:
                        exit_reason = 6
                        closed = True
                if not closed and exit_l[t]:
                    exit_reason = 3
                    closed = True
            else:
                if high[t] >= sl:
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                    closed = True; was_sl_exit = True
                elif low[t] <= tp:
                    if td_active:
                        exit_reason = 5
                    else:
                        exit_reason = 1
                    closed = True
                if not closed:
                    av = adx14[t]
                    if av == av and av > adx_exit_thresh:
                        exit_reason = 6
                        closed = True
                if not closed and exit_s[t]:
                    exit_reason = 3
                    closed = True
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
                a = atr14[t]
                if a != a: a = 0.0
                cp = 1.0; ep = close[t]; ea = a; be_active = False
                sl = close[t] - sl_mult * a
                tp = close[t] + tp_mult * a
                orig_tp_dist = tp_mult * a
                held = 0
                entry_bar_t = t
                direction_t = 1
            elif s == -1:
                a = atr14[t]
                if a != a: a = 0.0
                cp = -1.0; ep = close[t]; ea = a; be_active = False
                sl = close[t] + sl_mult * a
                tp = close[t] - tp_mult * a
                orig_tp_dist = tp_mult * a
                held = 0
                entry_bar_t = t
                direction_t = -1
        pos[t] = cp
    return pos, trades[:nt]
# ═══════════════════════════════════════════════════════════════════
# Section 5b — VPred Backtest Engines V3
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def bt_contrarian_vpred_v3(sig_arr, exit_arr, close, high, low, sigma,
                           k_sl, k_tp, base_hold,
                           k_be, gamma_hold, sigma_median,
                           cooldown_bars, warmup, end_idx, log_trades):
    """Contrarian vpred V3: breakeven, dynamic hold, cooldown."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0; max_hold_t = base_hold
    ep = 0.0; be_active = False
    cooldown_rem = 0; was_sl_exit = False

    max_t = n // 2 + 1 if log_trades > 0 else 1
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
            exit_reason = -1

            if k_be > 0.0 and not be_active:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                if cp == 1.0 and (close[t] - ep) > k_be * sv * ep:
                    be_active = True; sl = ep
                elif cp == -1.0 and (ep - close[t]) > k_be * sv * ep:
                    be_active = True; sl = ep

            if cp == 1.0:
                if low[t] <= sl:
                    closed = True; was_sl_exit = True
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                elif high[t] >= tp:
                    closed = True; exit_reason = 1
            else:
                if high[t] >= sl:
                    closed = True; was_sl_exit = True
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                elif low[t] <= tp:
                    closed = True; exit_reason = 1
            if not closed and exit_arr[t]:
                closed = True; exit_reason = 3
            if not closed and held >= max_hold_t:
                closed = True; exit_reason = 2
            if closed:
                if was_sl_exit and cooldown_bars > 0:
                    cooldown_rem = cooldown_bars
                if log_trades > 0 and nt < max_t:
                    trades[nt, 0] = entry_bar_t
                    trades[nt, 1] = t
                    trades[nt, 2] = direction_t
                    trades[nt, 3] = ep
                    trades[nt, 4] = close[t]
                    trades[nt, 5] = exit_reason
                    trades[nt, 6] = held
                    nt += 1
                cp = 0.0; held = 0; be_active = False

        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = 1.0; ep = close[t]; be_active = False
                sl = close[t] * (1.0 - k_sl * sv)
                tp = close[t] * (1.0 + k_tp * sv)
                held = 0
                entry_bar_t = t
                direction_t = 1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
            elif s == -1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = -1.0; ep = close[t]; be_active = False
                sl = close[t] * (1.0 + k_sl * sv)
                tp = close[t] * (1.0 - k_tp * sv)
                held = 0
                entry_bar_t = t
                direction_t = -1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
        pos[t] = cp
    return (pos, trades[:nt])


@njit(cache=True)
def bt_trend_vpred_v3(sig_arr, exit_long, exit_short, close, high, low, sigma,
                      rolling_high, rolling_low,
                      k_sl, k_trail, k_be,
                      gamma_hold, sigma_median, base_hold,
                      cooldown_bars, warmup, end_idx, log_trades):
    """Trend vpred V3: dynamic trailing, dynamic hold, cooldown."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; ep = 0.0; es = 0.0; be = False
    sl = np.nan; max_hold_t = base_hold; held = 0
    cooldown_rem = 0; was_sl_exit = False

    max_t = n // 2 + 1 if log_trades > 0 else 1
    trades = np.empty((max_t, 7))
    nt = 0
    entry_bar_t = 0
    direction_t = 0

    for t in range(warmup, n):
        if cooldown_rem > 0:
            cooldown_rem -= 1

        if cp != 0.0:
            held += 1
            sv = sigma[t]
            if sv != sv: sv = 0.02
            closed = False
            was_sl_exit = False
            exit_reason = -1

            if cp == 1.0:
                tsl = rolling_high[t] - k_trail * sv * close[t]
                if not be and k_be > 0.0 and (close[t] - ep) > k_be * es * ep:
                    be = True
                if be: tsl = max(tsl, ep)
                if sl != sl: sl = tsl
                else: sl = max(sl, tsl)
                if low[t] <= sl:
                    closed = True; was_sl_exit = True; exit_reason = 7
                elif exit_long[t]:
                    closed = True; exit_reason = 3
            else:
                tsl = rolling_low[t] + k_trail * sv * close[t]
                if not be and k_be > 0.0 and (ep - close[t]) > k_be * es * ep:
                    be = True
                if be: tsl = min(tsl, ep)
                if sl != sl: sl = tsl
                else: sl = min(sl, tsl)
                if high[t] >= sl:
                    closed = True; was_sl_exit = True; exit_reason = 7
                elif exit_short[t]:
                    closed = True; exit_reason = 3

            if not closed and held >= max_hold_t:
                closed = True; exit_reason = 2

            if closed:
                if was_sl_exit and cooldown_bars > 0:
                    cooldown_rem = cooldown_bars
                if log_trades > 0 and nt < max_t:
                    trades[nt, 0] = entry_bar_t
                    trades[nt, 1] = t
                    trades[nt, 2] = direction_t
                    trades[nt, 3] = ep
                    trades[nt, 4] = close[t]
                    trades[nt, 5] = exit_reason
                    trades[nt, 6] = held
                    nt += 1
                cp = 0.0; sl = np.nan; ep = 0.0; es = 0.0; be = False; held = 0

        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = 1.0; ep = close[t]; es = sv; be = False; held = 0
                sl = ep * (1.0 - k_sl * sv)
                entry_bar_t = t
                direction_t = 1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
            elif s == -1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = -1.0; ep = close[t]; es = sv; be = False; held = 0
                sl = ep * (1.0 + k_sl * sv)
                entry_bar_t = t
                direction_t = -1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
        pos[t] = cp
    return (pos, trades[:nt])


@njit(cache=True)
def bt_range_vpred_v3(sig_arr, exit_arr, close, high, low, sigma, adx14,
                      k_sl, k_tp, adx_exit_thresh, base_hold,
                      k_be, gamma_hold, sigma_median,
                      cooldown_bars, warmup, end_idx, log_trades):
    """Range vpred V3: breakeven, dynamic hold, cooldown."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0; max_hold_t = base_hold
    ep = 0.0; be_active = False
    cooldown_rem = 0; was_sl_exit = False

    max_t = n // 2 + 1 if log_trades > 0 else 1
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
            exit_reason = -1

            if k_be > 0.0 and not be_active:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                if cp == 1.0 and (close[t] - ep) > k_be * sv * ep:
                    be_active = True; sl = ep
                elif cp == -1.0 and (ep - close[t]) > k_be * sv * ep:
                    be_active = True; sl = ep

            if cp == 1.0:
                if low[t] <= sl:
                    closed = True; was_sl_exit = True
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                elif high[t] >= tp:
                    closed = True; exit_reason = 1
            else:
                if high[t] >= sl:
                    closed = True; was_sl_exit = True
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                elif low[t] <= tp:
                    closed = True; exit_reason = 1
            if not closed:
                av = adx14[t]
                if av == av and av > adx_exit_thresh:
                    closed = True; exit_reason = 6
            if not closed and exit_arr[t]:
                closed = True; exit_reason = 3
            if not closed and held >= max_hold_t:
                closed = True; exit_reason = 2
            if closed:
                if was_sl_exit and cooldown_bars > 0:
                    cooldown_rem = cooldown_bars
                if log_trades > 0 and nt < max_t:
                    trades[nt, 0] = entry_bar_t
                    trades[nt, 1] = t
                    trades[nt, 2] = direction_t
                    trades[nt, 3] = ep
                    trades[nt, 4] = close[t]
                    trades[nt, 5] = exit_reason
                    trades[nt, 6] = held
                    nt += 1
                cp = 0.0; held = 0; be_active = False

        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = 1.0; ep = close[t]; be_active = False
                sl = close[t] * (1.0 - k_sl * sv)
                tp = close[t] * (1.0 + k_tp * sv)
                held = 0
                entry_bar_t = t
                direction_t = 1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
            elif s == -1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = -1.0; ep = close[t]; be_active = False
                sl = close[t] * (1.0 + k_sl * sv)
                tp = close[t] * (1.0 - k_tp * sv)
                held = 0
                entry_bar_t = t
                direction_t = -1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
        pos[t] = cp
    return (pos, trades[:nt])


@njit(cache=True)
def bt_range_split_vpred_v3(sig_arr, exit_l, exit_s, close, high, low, sigma, adx14,
                            k_sl, k_tp, adx_exit_thresh, base_hold,
                            k_be, gamma_hold, sigma_median,
                            cooldown_bars, warmup, end_idx, log_trades):
    """Range vpred V3 with split exit arrays."""
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp = 0.0; sl = 0.0; tp = 0.0; held = 0; max_hold_t = base_hold
    ep = 0.0; be_active = False
    cooldown_rem = 0; was_sl_exit = False

    max_t = n // 2 + 1 if log_trades > 0 else 1
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
            exit_reason = -1

            if k_be > 0.0 and not be_active:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                if cp == 1.0 and (close[t] - ep) > k_be * sv * ep:
                    be_active = True; sl = ep
                elif cp == -1.0 and (ep - close[t]) > k_be * sv * ep:
                    be_active = True; sl = ep

            if cp == 1.0:
                if low[t] <= sl:
                    closed = True; was_sl_exit = True
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                elif high[t] >= tp:
                    closed = True; exit_reason = 1
                if not closed:
                    av = adx14[t]
                    if av == av and av > adx_exit_thresh:
                        closed = True; exit_reason = 6
                if not closed and exit_l[t]:
                    closed = True; exit_reason = 3
            else:
                if high[t] >= sl:
                    closed = True; was_sl_exit = True
                    if be_active:
                        exit_reason = 4
                    else:
                        exit_reason = 0
                elif low[t] <= tp:
                    closed = True; exit_reason = 1
                if not closed:
                    av = adx14[t]
                    if av == av and av > adx_exit_thresh:
                        closed = True; exit_reason = 6
                if not closed and exit_s[t]:
                    closed = True; exit_reason = 3
            if not closed and held >= max_hold_t:
                closed = True; exit_reason = 2
            if closed:
                if was_sl_exit and cooldown_bars > 0:
                    cooldown_rem = cooldown_bars
                if log_trades > 0 and nt < max_t:
                    trades[nt, 0] = entry_bar_t
                    trades[nt, 1] = t
                    trades[nt, 2] = direction_t
                    trades[nt, 3] = ep
                    trades[nt, 4] = close[t]
                    trades[nt, 5] = exit_reason
                    trades[nt, 6] = held
                    nt += 1
                cp = 0.0; held = 0; be_active = False

        if cp == 0.0 and cooldown_rem == 0:
            s = sig_arr[t]
            if s == 1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = 1.0; ep = close[t]; be_active = False
                sl = close[t] * (1.0 - k_sl * sv)
                tp = close[t] * (1.0 + k_tp * sv)
                held = 0
                entry_bar_t = t
                direction_t = 1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
            elif s == -1:
                sv = sigma[t]
                if sv != sv: sv = 0.02
                cp = -1.0; ep = close[t]; be_active = False
                sl = close[t] * (1.0 + k_sl * sv)
                tp = close[t] * (1.0 - k_tp * sv)
                held = 0
                entry_bar_t = t
                direction_t = -1
                if gamma_hold > 0.0 and sv > 1e-8 and sigma_median > 1e-8:
                    mh = base_hold * (sigma_median / sv) ** gamma_hold
                    max_hold_t = int(min(max(mh, 3.0), base_hold * 3.0))
                else:
                    max_hold_t = base_hold
        pos[t] = cp
    return (pos, trades[:nt])


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Metrics with Commission
# ═══════════════════════════════════════════════════════════════════

def calc_sharpe_comm(positions, log_ret, mask, ann_factor):
    pos = positions[mask]
    lr = log_ret[mask]
    if len(pos) == 0: return -999.0
    dr = pos * lr
    dpos = np.diff(pos, prepend=0.0)
    comm = np.abs(dpos) * COMMISSION
    net_r = dr - comm
    s = np.std(net_r, ddof=1)
    if s < 1e-12: return 0.0
    return np.mean(net_r) / s * ann_factor


def _quick_exposure_trades(positions, mask, bars_per_year):
    """Fast exposure % and trades/year check for constraint filtering."""
    pos = positions[mask]
    n = len(pos)
    if n == 0:
        return 0.0, 0.0
    exposure_pct = np.count_nonzero(pos) / n * 100
    trades = 0
    in_t = False
    for i in range(n):
        if pos[i] != 0 and not in_t:
            trades += 1
            in_t = True
        if in_t and pos[i] == 0:
            in_t = False
    years = max(n / bars_per_year, 0.5)
    return exposure_pct, trades / years


def calc_metrics_comm(positions, log_ret, mask, ann_factor, bars_per_year):
    pos = positions[mask]
    lr = log_ret[mask]
    n = len(pos)
    if n == 0: return {}
    dr = pos * lr
    dpos = np.diff(pos, prepend=0.0)
    comm = np.abs(dpos) * COMMISSION
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
    return {"sharpe": round(sharpe, 4), "ann_ret_pct": round(ann_ret * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "max_dd_pct": round(max_dd * 100, 2), "exposure_pct": round(exposure * 100, 2),
            "n_trades": trades, "win_rate_pct": round(win_rate, 1)}


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Dispatch Functions (V3)
# ═══════════════════════════════════════════════════════════════════

def dispatch_signals(sid, close, high, low, volume, ind,
                     sma_cache, dc_cache, st_cache, vwap_cache,
                     pivot_data, daily_trend, sig_params):
    if sid == "S1":
        return gen_s1_signals(close, ind, sma_cache, **sig_params)
    elif sid == "S2":
        return gen_s2_signals(close, ind, sma_cache, **sig_params)
    elif sid == "S3":
        return gen_s3_signals(close, high, low, ind, dc_cache, daily_trend=daily_trend, **sig_params)
    elif sid == "S4":
        return gen_s4_signals(close, high, low, ind, st_cache, daily_trend=daily_trend, **sig_params)
    elif sid == "S5":
        return gen_s5_signals(close, ind, pivot_data)
    elif sid == "S6":
        return gen_s6_signals(close, high, low, volume, ind, vwap_cache, **sig_params)


def dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind, rm, warmup, end_idx, log_trades=0):
    cat = CATEGORY[sid]
    if cat == "Contrarian":
        be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
        cd = rm.get("cooldown_bars", 0)
        pe = 1 if rm.get("partial_exit") else 0
        return bt_contrarian_v3(sig, exit_info, close, high, low, ind["atr14"],
                                rm["sl_mult"], rm["tp_mult"], int(rm["max_hold"]),
                                be, cd, pe, warmup, end_idx, log_trades)
    elif cat == "Trend":
        exit_l, exit_s = exit_info
        tn = int(rm.get("trail_n", 10))
        rh = ind.get(f"rh_{tn}", pd.Series(high).rolling(tn+1, min_periods=1).max().values)
        rl = ind.get(f"rl_{tn}", pd.Series(low).rolling(tn+1, min_periods=1).min().values)
        be_val = 1e12 if rm.get("breakeven_thresh") is None else float(rm["breakeven_thresh"])
        tt_map = {"fixed_atr": 0, "chandelier": 1, "parabolic_step": 2}
        tt = tt_map.get(rm.get("trail_type", "fixed_atr"), 0)
        ps = rm.get("parabolic_step", 0.02)
        pm = rm.get("parabolic_max", 0.15)
        cd = rm.get("cooldown_bars", 0)
        return bt_trend_v3(sig, exit_l, exit_s, close, high, low, ind["atr14"],
                           rh, rl, rm["initial_sl_mult"], rm.get("trail_atr_mult", 2.5),
                           be_val, tt, ps, pm, cd, warmup, end_idx, log_trades)
    elif cat == "Range":
        be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
        cd = rm.get("cooldown_bars", 0)
        td = 1 if rm.get("time_decay") else 0
        if isinstance(exit_info, tuple):
            el, es = exit_info
            return bt_range_split_v3(sig, el, es, close, high, low,
                                     ind["atr14"], ind["adx14"],
                                     rm["sl_mult"], rm["tp_mult"],
                                     30.0, int(rm["max_hold"]),
                                     be, cd, td, warmup, end_idx, log_trades)
        return bt_range_v3(sig, exit_info, close, high, low, ind["atr14"], ind["adx14"],
                           rm["sl_mult"], rm["tp_mult"], 30.0, int(rm["max_hold"]),
                           be, cd, td, warmup, end_idx, log_trades)


def dispatch_bt_vpred_v3(sid, sig, exit_info, close, high, low, ind, sigma,
                         bp, warmup, end_idx, is_hourly, sigma_median, log_trades=0):
    cat = CATEGORY[sid]
    k_be = -1.0 if bp.get("k_be") is None else bp["k_be"]
    gh = -1.0 if bp.get("gamma_hold") is None else bp["gamma_hold"]
    cd = bp.get("cooldown_bars", 5)

    if cat == "Contrarian":
        base_hold = 40 if is_hourly else 20
        k_tp = bp.get("k_tp", bp.get("k_sl", 1.0) * bp.get("ratio", 1.5))
        return bt_contrarian_vpred_v3(sig, exit_info, close, high, low, sigma,
                                      bp["k_sl"], k_tp, base_hold,
                                      k_be, gh, sigma_median,
                                      cd, warmup, end_idx, log_trades)
    elif cat == "Trend":
        base_hold = 60 if is_hourly else 30
        tn = 10
        rh = ind.get(f"rh_{tn}", pd.Series(high).rolling(tn+1, min_periods=1).max().values)
        rl = ind.get(f"rl_{tn}", pd.Series(low).rolling(tn+1, min_periods=1).min().values)
        exit_l, exit_s = exit_info
        return bt_trend_vpred_v3(sig, exit_l, exit_s, close, high, low, sigma,
                                 rh, rl, bp["k_sl"], bp["k_trail"], k_be,
                                 gh, sigma_median, base_hold,
                                 cd, warmup, end_idx, log_trades)
    elif cat == "Range":
        base_hold = 20 if is_hourly else 10
        k_tp = bp.get("k_tp", bp.get("k_sl", 1.0) * bp.get("ratio", 1.5))
        if isinstance(exit_info, tuple):
            el, es = exit_info
            return bt_range_split_vpred_v3(sig, el, es, close, high, low, sigma,
                                           ind["adx14"], bp["k_sl"], k_tp,
                                           30.0, base_hold,
                                           k_be, gh, sigma_median,
                                           cd, warmup, end_idx, log_trades)
        return bt_range_vpred_v3(sig, exit_info, close, high, low, sigma, ind["adx14"],
                                 bp["k_sl"], k_tp, 30.0, base_hold,
                                 k_be, gh, sigma_median,
                                 cd, warmup, end_idx, log_trades)


# ═══════════════════════════════════════════════════════════════════
# Section 8: Approach A — Full Grid Search + Top-10 Ensemble (V3)
# ═══════════════════════════════════════════════════════════════════

def _year_bounds(dates, year):
    y_start = pd.Timestamp(f"{year}-01-01")
    y_end = pd.Timestamp(f"{year + 1}-01-01")
    start = np.searchsorted(dates, y_start)
    end = np.searchsorted(dates, y_end)
    return int(start), int(end)


def approach_a_one(sid, tf, close, high, low, volume, open_arr, ind,
                   sma_cache, dc_cache, st_cache, vwap_cache,
                   pivot_data, daily_trend, dates, is_hourly, log_ret):
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)

    sig_grid = expand_grid(SIGNAL_GRIDS[sid])
    cat = CATEGORY[sid]
    if cat == "Trend":
        rm_grid = TREND_RM_GRID
    else:
        rm_grid = expand_grid(RM_GRIDS_V3[(cat, tf)])

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

        cached_sigs = {}
        for sp in sig_grid:
            key = tuple(sorted(sp.items()))
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, dc_cache, st_cache, vwap_cache,
                                              pivot_data, daily_trend, sp)
            cached_sigs[key] = (sig, exit_info)

        all_results = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            for rm in rm_grid:
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, WARMUP, train_end, 0)
                sh = calc_sharpe_comm(pos, log_ret, train_mask, ann)
                dr = pos[train_mask] * log_ret[train_mask]
                dpos = np.diff(pos[train_mask], prepend=0.0)
                net_r = dr - np.abs(dpos) * COMMISSION
                cum = np.cumsum(net_r)
                rmax = np.maximum.accumulate(cum)
                max_dd = (cum - rmax).min() if len(cum) > 0 else 0.0
                active = np.sum(pos[train_mask] != 0)
                exposure = active / train_mask.sum() * 100 if train_mask.sum() > 0 else 0
                all_results.append((sp, rm, sh, max_dd, exposure))

        passing = [(sp, rm, sh, mdd, exp) for sp, rm, sh, mdd, exp in all_results
                   if sh > 0 and mdd > -0.30 and exp > 5]
        if len(passing) == 0:
            passing = sorted(all_results, key=lambda x: x[2], reverse=True)[:3]
        passing = sorted(passing, key=lambda x: x[2], reverse=True)[:10]

        test_positions = []
        test_trades = []
        for sp, rm, sh, mdd, exp in passing:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            pos, tr = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                         rm, WARMUP, test_end, 1)
            test_positions.append(pos)
            test_trades.append(tr)

        ensemble = np.mean(test_positions, axis=0)
        results_by_year[test_year] = ensemble
        params_by_year[test_year] = [(sp, rm) for sp, rm, sh, mdd, exp in passing]
        trades_by_year[test_year] = test_trades

    return results_by_year, params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# Section 9: Approach B — Adaptive sigma_pred Stops (V3)
# ═══════════════════════════════════════════════════════════════════

def approach_b_one(sid, tf, close, high, low, volume, ind,
                   sma_cache, dc_cache, st_cache, vwap_cache,
                   pivot_data, daily_trend, dates, sigma_dict, is_hourly,
                   log_ret, a_params_by_year):
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
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

        a_sigs = []
        for sp, rm in a_params:
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, dc_cache, st_cache, vwap_cache,
                                              pivot_data, daily_trend, sp)
            a_sigs.append((sig, exit_info))

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
            for sig, exit_info in a_sigs:
                pos, _ = dispatch_bt_vpred_v3(sid, sig, exit_info, close, high, low, ind,
                                              sigma, bp_engine, WARMUP, val_end, is_hourly, sigma_median, 0)
                val_positions.append(pos)
            ensemble = np.mean(val_positions, axis=0)
            sh = calc_sharpe_comm(ensemble, log_ret, val_mask, ann)
            if sh > best_sh:
                best_sh = sh
                best_bp = dict(bp)

        best_horizon = best_bp.get("horizon", "h1")
        best_sigma = sigma_dict.get(best_horizon, sigma_dict.get("h1", np.full(n, 0.02)))
        bs_pos = best_sigma[best_sigma > 1e-6]
        best_sigma_median = np.nanmedian(bs_pos) if len(bs_pos) > 0 else 0.02
        best_bp_engine = {k: v for k, v in best_bp.items() if k not in ("horizon", "ratio")}
        if "ratio" in best_bp:
            best_bp_engine["k_tp"] = best_bp["k_sl"] * best_bp["ratio"]

        test_positions = []
        test_trades = []
        for sig, exit_info in a_sigs:
            pos, tr = dispatch_bt_vpred_v3(sid, sig, exit_info, close, high, low, ind,
                                           best_sigma, best_bp_engine, WARMUP, test_end, is_hourly, best_sigma_median, 1)
            test_positions.append(pos)
            test_trades.append(tr)
        ensemble = np.mean(test_positions, axis=0)
        results_by_year[test_year] = ensemble
        best_params_by_year[test_year] = best_bp
        trades_by_year[test_year] = test_trades

    return results_by_year, best_params_by_year, trades_by_year
# ═══════════════════════════════════════════════════════════════════
# Section 10: Approach C — Regime Filter (V3)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _apply_hysteresis(pctrank, is_above_type, entry_thresh, exit_thresh, n):
    mask = np.zeros(n, dtype=np.bool_)
    active = False
    for i in range(n):
        v = pctrank[i]
        if v != v:
            mask[i] = False
            continue
        if is_above_type == 1:
            if not active and v > entry_thresh:
                active = True
            elif active and v < exit_thresh:
                active = False
        else:
            if not active and v < (1.0 - entry_thresh):
                active = True
            elif active and v > (1.0 - exit_thresh):
                active = False
        mask[i] = active
    return mask


def _build_c_mask_v3(cat, sigma, lookback, threshold_or_band,
                     direction, delta_win, sigma_h22, sigma_h1,
                     term_filter, hysteresis, n):
    pctrank = pd.Series(sigma).rolling(lookback, min_periods=max(50, lookback // 5)).rank(pct=True).values

    if cat == "Trend":
        if hysteresis > 0:
            exit_thresh = threshold_or_band - hysteresis / 100.0
            mask = _apply_hysteresis(pctrank, 1, threshold_or_band, exit_thresh, n)
        else:
            mask = ~np.isnan(pctrank) & (pctrank > threshold_or_band)
    elif cat == "Contrarian":
        if hysteresis > 0:
            exit_thresh = threshold_or_band - hysteresis / 100.0
            mask = _apply_hysteresis(pctrank, 0, threshold_or_band, exit_thresh, n)
        else:
            mask = ~np.isnan(pctrank) & (pctrank < (1 - threshold_or_band))
    else:
        lo, hi = threshold_or_band
        mask = ~np.isnan(pctrank) & (pctrank >= lo) & (pctrank <= hi)

    if direction == "rise":
        dw = 5
        delta_sigma = np.full(n, np.nan)
        delta_sigma[dw:] = sigma[dw:] - sigma[:-dw]
        mask = mask & (~np.isnan(delta_sigma)) & (delta_sigma > 0)
    elif direction == "fall":
        dw = 5
        delta_sigma = np.full(n, np.nan)
        delta_sigma[dw:] = sigma[dw:] - sigma[:-dw]
        mask = mask & (~np.isnan(delta_sigma)) & (delta_sigma < 0)

    if term_filter and sigma_h22 is not None and sigma_h1 is not None:
        s_h1_safe = np.maximum(sigma_h1, 1e-8)
        ratio = sigma_h22 / s_h1_safe
        valid_r = ~np.isnan(ratio)
        if cat == "Trend":
            mask = mask & valid_r & (ratio > 1.0)
        elif cat == "Contrarian":
            mask = mask & valid_r & (ratio < 1.0)

    return mask


def _extract_trades_from_pos(pos_arr, close, start_idx, end_idx):
    """Extract trades from a position array. Returns 2D array (n_trades, 7).
    exit_reason = 9 (position-derived).
    """
    trades = []
    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    direction = 0

    for t in range(start_idx, end_idx):
        p = pos_arr[t]
        if not in_trade:
            if abs(p) > 0.01:
                in_trade = True
                entry_bar = t
                entry_price = close[t]
                direction = 1 if p > 0 else -1
        else:
            # Exit when position goes to ~0 or flips direction
            if abs(p) < 0.01 or (direction == 1 and p < -0.01) or (direction == -1 and p > 0.01):
                trades.append([entry_bar, t, direction, entry_price, close[t], 9, t - entry_bar])
                in_trade = False
                # Check if new trade starts immediately
                if abs(p) > 0.01:
                    in_trade = True
                    entry_bar = t
                    entry_price = close[t]
                    direction = 1 if p > 0 else -1

    # Close open trade at end
    if in_trade:
        t_end = min(end_idx, len(pos_arr)) - 1
        trades.append([entry_bar, t_end, direction, entry_price, close[t_end], 8, t_end - entry_bar])

    if trades:
        return np.array(trades)
    return np.empty((0, 7))


def _c_grid_search(cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
                    bars_per_year, n, min_exp, min_tpy):
    """Inner grid search for approach C. Returns (best_sharpe, best_params)."""
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
                                sh = calc_sharpe_comm(masked_pos, log_ret, val_mask, ann)
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
                                sh = calc_sharpe_comm(masked_pos, log_ret, val_mask, ann)
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


def approach_c_one(sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n):
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
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
        best_sh, best_cp = _c_grid_search(
            cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
            bpy, n, C_MIN_EXPOSURE, C_MIN_TRADES_YR)
        if not best_cp:
            best_sh, best_cp = _c_grid_search(
                cat, a_pos, sigma_dict, is_hourly, log_ret, val_mask, ann,
                bpy, n, C_FALLBACK_EXPOSURE, C_FALLBACK_TRADES_YR)

        if not best_cp: continue

        sigma = sigma_dict.get(best_cp.get("horizon", "h1"))
        if sigma is None: continue
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
# Section 11: Approach D — Vol-Targeting (V3)
# ═══════════════════════════════════════════════════════════════════

def _compute_d_scale_v3(sigma_raw, dp):
    sigma = sigma_raw.copy()
    smooth = dp.get("smooth_span")
    if smooth is not None:
        sigma = pd.Series(sigma).ewm(span=smooth, min_periods=1).mean().values
    vf = dp.get("vol_floor")
    vc = dp.get("vol_cap")
    if vf is not None:
        sigma = np.maximum(sigma, vf)
    if vc is not None:
        sigma = np.minimum(sigma, vc)
    sigma = np.where(np.isnan(sigma) | (sigma < 1e-6), 1e-6, sigma)

    if dp.get("inverse_vol", False):
        inv = 1.0 / sigma
        lb = dp.get("inv_lookback", 126)
        norm = pd.Series(inv).rolling(lb, min_periods=20).mean().values
        norm = np.where(np.isnan(norm) | (norm < 1e-6), 1e-6, norm)
        scale = inv / norm
    else:
        gamma = dp.get("gamma", 1.0)
        target_vol = dp.get("target_vol", 0.10)
        scale = (target_vol / sigma) ** gamma

    max_lev = dp.get("max_lev", 2.0)
    scale = np.minimum(scale, max_lev)
    return scale


def approach_d_one(sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n):
    ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
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
                                    scaled_pos = a_pos * scale
                                    sh = calc_sharpe_comm(scaled_pos, log_ret, val_mask, ann)
                                    if sh > best_sh:
                                        best_sh = sh
                                        best_dp = dict(dp)

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
                        scaled_pos = a_pos * scale
                        sh = calc_sharpe_comm(scaled_pos, log_ret, val_mask, ann)
                        if sh > best_sh:
                            best_sh = sh
                            best_dp = dict(dp)

        if not best_dp: continue

        sigma = sigma_dict.get(best_dp.get("horizon", "h1"))
        if sigma is None: continue
        scale = _compute_d_scale_v3(sigma, best_dp)
        final_pos = a_pos * scale
        results_by_year[test_year] = final_pos
        best_params_by_year[test_year] = best_dp

    return results_by_year, best_params_by_year, trades_by_year


# ═══════════════════════════════════════════════════════════════════
# Section 12: Statistical Tests
# ═══════════════════════════════════════════════════════════════════

def dm_test(returns_a, returns_b):
    d = returns_a - returns_b
    n = len(d)
    if n < 10:
        return 0.0, 1.0
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var < 1e-20:
        return 0.0, 1.0
    t_stat = d_mean / np.sqrt(d_var / n)
    from scipy import stats as sp_stats
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - 1))
    return round(t_stat, 4), round(p_val, 4)


def bootstrap_sharpe_diff(ret_a, ret_b, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(ret_a)
    if n < 10:
        return 0.0, 0.0, 0.0
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ra = ret_a[idx]
        rb = ret_b[idx]
        sa = np.mean(ra) / np.std(ra, ddof=1) if np.std(ra, ddof=1) > 1e-12 else 0
        sb = np.mean(rb) / np.std(rb, ddof=1) if np.std(rb, ddof=1) > 1e-12 else 0
        diffs.append(sa - sb)
    diffs = np.array(diffs)
    return round(np.mean(diffs), 4), round(np.percentile(diffs, 2.5), 4), round(np.percentile(diffs, 97.5), 4)


# ═══════════════════════════════════════════════════════════════════
# Section 13: process_ticker + multiprocessing
# ═══════════════════════════════════════════════════════════════════

def _compute_nhours_per_day(dates_hourly):
    bar_days = pd.to_datetime(dates_hourly).normalize()
    day_counts = pd.Series(bar_days).groupby(bar_days).transform("count").values
    return day_counts.astype(np.float64)


def _store_metrics(results, sid, tf, approach, year, pos_arr,
                   log_ret, dates, n, ann, bpy, ticker):
    ys, _ = _year_bounds(dates, year)
    ye_actual = _year_bounds(dates, year + 1)[0] if year < 2026 else n
    test_mask = np.zeros(n, dtype=bool)
    test_mask[ys:ye_actual] = True
    met = calc_metrics_comm(pos_arr, log_ret, test_mask, ann, bpy)
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
    net_r = dr - np.abs(dpos) * COMMISSION
    met["_net_returns"] = net_r

    key = (sid, tf, approach, year)
    results[key] = met


def _trades_to_rows(trades_list_or_arr, strategy_name, tf, ticker, approach, test_year,
                    dates, n_components=1):
    """Convert trade arrays to list of dicts for DataFrame."""
    rows = []
    if isinstance(trades_list_or_arr, list):
        # List of 2D arrays (one per ensemble component)
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
        # Single 2D array (from C/D position extraction)
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


def process_ticker(ticker, daily_df, hourly_df, vpred_df):
    """Process one ticker: run A/B/C/D for all strategies x TFs.
    Hourly sigma: v2_sqrtN only (no variant loop).
    Returns dict with keys: results, trades, positions.
    """
    results = {}
    trade_rows = []
    position_rows = []

    for tf in TIMEFRAMES:
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
                # v2_sqrtN for hourly
                if is_hourly:
                    dates_arr = tdf["datetime"].values
                    nhours = _compute_nhours_per_day(dates_arr)
                    nhours_safe = np.maximum(nhours, 1.0)
                    sigma_arr = sigma_arr / np.sqrt(nhours_safe)
                sigma_dict[hname] = sigma_arr

        ann = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
        bpy = 252 * 9 if is_hourly else 252

        for sid in STRATEGY_IDS:
            sname = STRATEGY_NAMES[sid]

            # Approach A
            a_results, a_params, a_trades = approach_a_one(
                sid, tf, close, high, low, volume, open_arr, ind,
                sma_cache, dc_cache, st_cache, vwap_cache,
                pivot_data, daily_trend, dates, is_hourly, log_ret)

            for year, pos_arr in a_results.items():
                _store_metrics(results, sid, tf, "A", year, pos_arr,
                               log_ret, dates, n, ann, bpy, ticker)

            # Collect A trades
            for year, trades_list in a_trades.items():
                trade_rows.extend(_trades_to_rows(trades_list, sname, tf, ticker, "A", year, dates))

            # Approach B
            b_results, b_params, b_trades = approach_b_one(
                sid, tf, close, high, low, volume, ind,
                sma_cache, dc_cache, st_cache, vwap_cache,
                pivot_data, daily_trend, dates, sigma_dict, is_hourly,
                log_ret, a_params)

            # Collect B trades
            for year, trades_list in b_trades.items():
                trade_rows.extend(_trades_to_rows(trades_list, sname, tf, ticker, "B", year, dates))

            # Approach C
            c_results, c_params, c_trades = approach_c_one(
                sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

            # Extract C trades from positions
            for year, pos_arr in c_results.items():
                test_start_c = _year_bounds(dates, year)[0]
                test_end_c = _year_bounds(dates, year + 1)[0] if year < 2026 else n
                c_trades[year] = _extract_trades_from_pos(pos_arr, close, test_start_c, test_end_c)

            # Collect C trades
            for year, trades_arr in c_trades.items():
                trade_rows.extend(_trades_to_rows(trades_arr, sname, tf, ticker, "C", year, dates))

            # Approach D
            d_results, d_params, d_trades = approach_d_one(
                sid, tf, dates, a_results, sigma_dict, is_hourly, log_ret, n)

            # Extract D trades from positions
            for year, pos_arr in d_results.items():
                test_start_d = _year_bounds(dates, year)[0]
                test_end_d = _year_bounds(dates, year + 1)[0] if year < 2026 else n
                d_trades[year] = _extract_trades_from_pos(pos_arr, close, test_start_d, test_end_d)

            # Collect D trades
            for year, trades_arr in d_trades.items():
                trade_rows.extend(_trades_to_rows(trades_arr, sname, tf, ticker, "D", year, dates))

            for approach, res in [("B", b_results), ("C", c_results), ("D", d_results)]:
                for year, pos_arr in res.items():
                    _store_metrics(results, sid, tf, approach, year, pos_arr,
                                   log_ret, dates, n, ann, bpy, ticker)

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


_G_DAILY = None
_G_HOURLY = None
_G_VPRED = None


def _pool_init(daily, hourly, vpred):
    global _G_DAILY, _G_HOURLY, _G_VPRED
    _G_DAILY = daily
    _G_HOURLY = hourly
    _G_VPRED = vpred


def _worker_func(ticker):
    return ticker, process_ticker(ticker, _G_DAILY, _G_HOURLY, _G_VPRED)


def warmup_numba():
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

    pos, _ = bt_contrarian_v3(sig, ex, c, h, l, a, 1.5, 2.0, 10, 1.0, 5, 1, 5, n, 0)
    pos, _ = bt_trend_v3(sig, ex_l, ex_s, c, h, l, a, rh, rl, 2.5, 2.5, 1.5, 0, 0.02, 0.15, 5, 5, n, 0)
    pos, _ = bt_range_v3(sig, ex, c, h, l, a, adx, 1.0, 1.5, 30.0, 10, 1.0, 5, 1, 5, n, 0)
    pos, _ = bt_range_split_v3(sig, ex_l, ex_s, c, h, l, a, adx, 1.0, 1.5, 30.0, 10, 1.0, 5, 1, 5, n, 0)
    pos, _ = bt_contrarian_vpred_v3(sig, ex, c, h, l, sigma, 1.0, 2.0, 20, 1.0, 0.5, 0.02, 5, 5, n, 0)
    pos, _ = bt_trend_vpred_v3(sig, ex_l, ex_s, c, h, l, sigma, rh, rl, 1.0, 1.0, 1.0, 0.5, 0.02, 30, 5, 5, n, 0)
    pos, _ = bt_range_vpred_v3(sig, ex, c, h, l, sigma, adx, 1.0, 2.0, 30.0, 10, 1.0, 0.5, 0.02, 5, 5, n, 0)
    pos, _ = bt_range_split_vpred_v3(sig, ex_l, ex_s, c, h, l, sigma, adx, 1.0, 2.0, 30.0, 10, 1.0, 0.5, 0.02, 5, 5, n, 0)
    pctrank = np.random.rand(n)
    _apply_hysteresis(pctrank, 1, 0.5, 0.4, n)
    print("  Numba warmup done")


# ═══════════════════════════════════════════════════════════════════
# Section 14: Output Generation
# ═══════════════════════════════════════════════════════════════════

ALL_APPROACHES = ["A", "B", "C", "D"]
BCD_APPROACHES = ["B", "C", "D"]


def generate_outputs(all_results, all_trade_rows, all_position_rows):
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

    # ── Statistical tests (compute BEFORE detailed tables so p-values are available) ──
    stat_rows = []
    stat_lookup = {}  # (strategy, tf, comparison) -> {dm_p, sig, ...}
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
        stat_df.to_csv(OUT_TABLES / "wf_v3_stat_tests.csv", index=False)
        print("\n  === Statistical Tests (B-D vs A) ===")
        print(stat_df.to_string(index=False))

    # ── Per strategy x TF detailed tables (12 tables) ──
    for sid in STRATEGY_IDS:
        sname = STRATEGY_NAMES[sid]
        for tf in TIMEFRAMES:
            sub = df[(df["strategy"] == sname) & (df["timeframe"] == tf)]
            if len(sub) == 0:
                continue

            # Mean across tickers per (approach, year)
            agg = sub.groupby(["approach", "year"]).agg(
                sharpe=("sharpe", "mean"),
                ann_ret_pct=("ann_ret_pct", "mean"),
                ann_vol_pct=("ann_vol_pct", "mean"),
                max_dd_pct=("max_dd_pct", "mean"),
                exposure_pct=("exposure_pct", "mean"),
                n_trades=("n_trades", "mean"),
                win_rate_pct=("win_rate_pct", "mean"),
            ).reset_index()

            # Summary per approach (mean across years)
            summary = agg.groupby("approach").agg(
                MeanSharpe=("sharpe", "mean"),
                AnnReturn=("ann_ret_pct", "mean"),
                AnnVol=("ann_vol_pct", "mean"),
                MaxDD=("max_dd_pct", "mean"),
                Exposure=("exposure_pct", "mean"),
                TradesPerYr=("n_trades", "mean"),
                WinRate=("win_rate_pct", "mean"),
            ).reset_index().round(4)

            fname = f"wf_v3_{sid}_{tf}.csv"
            summary.to_csv(OUT_TABLES / fname, index=False)
            print(f"\n  === {sname} {tf} ===")
            # Print formatted table
            hdr = f"  {'approach':>8s}  {'MeanSharpe':>10s}  {'AnnRet%':>8s}  {'AnnVol%':>8s}  {'MaxDD%':>7s}  {'Exp%':>6s}  {'Tr/yr':>6s}  {'Win%':>6s}"
            print(hdr)
            a_sharpe_val = None
            for _, r in summary.iterrows():
                if r["approach"] == "A":
                    a_sharpe_val = r["MeanSharpe"]
                print(f"  {r['approach']:>8s}  {r['MeanSharpe']:10.4f}  {r['AnnReturn']:8.2f}  {r['AnnVol']:8.2f}  {r['MaxDD']:7.2f}  {r['Exposure']:6.2f}  {r['TradesPerYr']:6.2f}  {r['WinRate']:6.2f}")

            # Delta Sharpe vs A and p-values
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

            # Top-5 / Worst-5 tickers with Sharpe values — for best approach
            sub_common = sub[sub["year"].isin(BCD_TEST_YEARS)]
            if len(sub_common) > 0:
                ticker_agg = sub_common.groupby(["approach", "ticker"]).agg(
                    sharpe=("sharpe", "median")
                ).reset_index()
                # Find best approach by MeanSharpe
                best_appr = summary.loc[summary["MeanSharpe"].idxmax(), "approach"]
                asub = ticker_agg[ticker_agg["approach"] == best_appr].sort_values("sharpe", ascending=False)
                if len(asub) >= 5:
                    top5 = asub.head(5)
                    worst5 = asub.tail(5)
                    top_str = ", ".join(f"{r['ticker']}({r['sharpe']:.2f})" for _, r in top5.iterrows())
                    worst_str = ", ".join(f"{r['ticker']}({r['sharpe']:.2f})" for _, r in worst5.iterrows())
                    print(f"  Top-5 ({best_appr}):   {top_str}")
                    print(f"  Worst-5 ({best_appr}): {worst_str}")

    # ── B vs A Holding Bars Diagnosis ──
    if all_trade_rows:
        trade_df_diag = pd.DataFrame(all_trade_rows)
        if "holding_bars" in trade_df_diag.columns and "approach" in trade_df_diag.columns:
            print("\n  === B vs A Holding Bars Diagnosis ===")
            for appr in ["A", "B"]:
                tsub = trade_df_diag[trade_df_diag["approach"] == appr]
                if len(tsub) > 0:
                    mean_h = tsub["holding_bars"].mean()
                    med_h = tsub["holding_bars"].median()
                    n_tr = len(tsub)
                    print(f"    {appr}: mean_hold={mean_h:.1f}, median_hold={med_h:.0f}, trades={n_tr:,}")

    # ── Summary tables ──
    common = df[df["year"].isin(BCD_TEST_YEARS)]
    if len(common) > 0:
        # Mean Sharpe
        mean_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            sharpe=("sharpe", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="sharpe"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in mean_pivot.columns]
        mean_pivot = mean_pivot[["strategy", "timeframe"] + approach_cols].round(4)
        mean_pivot.to_csv(OUT_TABLES / "wf_v3_summary_mean.csv", index=False)
        print("\n  === Summary: Mean Sharpe ===")
        print(mean_pivot.to_string(index=False))

        # Median Sharpe
        med_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            sharpe=("sharpe", "median")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="sharpe"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in med_pivot.columns]
        med_pivot = med_pivot[["strategy", "timeframe"] + approach_cols].round(4)
        med_pivot.to_csv(OUT_TABLES / "wf_v3_summary_median.csv", index=False)

        # Annualized return
        ret_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            ann_ret_pct=("ann_ret_pct", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="ann_ret_pct"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in ret_pivot.columns]
        ret_pivot = ret_pivot[["strategy", "timeframe"] + approach_cols].round(2)
        ret_pivot.to_csv(OUT_TABLES / "wf_v3_summary_return.csv", index=False)

        # Exposure pivot
        exp_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            exposure_pct=("exposure_pct", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="exposure_pct"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in exp_pivot.columns]
        exp_pivot = exp_pivot[["strategy", "timeframe"] + approach_cols].round(2)
        exp_pivot.to_csv(OUT_TABLES / "wf_v3_summary_exposure.csv", index=False)
        print("\n  === Summary: Exposure % ===")
        print(exp_pivot.to_string(index=False))

        # Trades/yr pivot
        trades_pivot = common.groupby(["strategy", "timeframe", "approach"]).agg(
            n_trades=("n_trades", "mean")
        ).reset_index().pivot_table(
            index=["strategy", "timeframe"], columns="approach", values="n_trades"
        ).reset_index()
        approach_cols = [c for c in ALL_APPROACHES if c in trades_pivot.columns]
        trades_pivot = trades_pivot[["strategy", "timeframe"] + approach_cols].round(2)
        trades_pivot.to_csv(OUT_TABLES / "wf_v3_summary_trades.csv", index=False)
        print("\n  === Summary: Trades/yr ===")
        print(trades_pivot.to_string(index=False))

    # ── Raw results ──
    df.to_csv(OUT_DATA / "wf_v3_all_results.csv", index=False)
    print(f"\n  Full results saved: {len(df)} rows")

    # ── Sanity checks ──
    print("\n  === Sanity Checks ===")
    for appr in ALL_APPROACHES:
        asub = df[df["approach"] == appr]
        if len(asub) > 0:
            exp = asub["exposure_pct"].mean()
            trades = asub["n_trades"].mean()
            print(f"    {appr}: avg exposure={exp:.1f}%, avg trades={trades:.1f}")

    # ── Trade log ──
    if all_trade_rows:
        trade_df = pd.DataFrame(all_trade_rows)
        trade_path = OUT_DIR / "trade_log.parquet"
        trade_df.to_parquet(trade_path, index=False)
        print(f"\n  Trade log saved: {len(trade_df)} trades \u2192 {trade_path}")
        # Summary by exit reason
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
        print(f"  Daily positions saved: {len(pos_df)} rows \u2192 {pos_path}")


# ═══════════════════════════════════════════════════════════════════
# Section 15: main()
# ═══════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    daily = pd.read_parquet(DATA_DIR / "ohlcv_daily_full.parquet")
    hourly = pd.read_parquet(DATA_DIR / "ohlcv_hourly_full.parquet")
    vpred = pd.read_parquet(DATA_DIR / "vpred_aligned.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    vpred["date"] = pd.to_datetime(vpred["date"])
    print(f"  Daily: {len(daily):,} rows, {daily['date'].min().date()} — {daily['date'].max().date()}")
    print(f"  Hourly: {len(hourly):,} rows")
    print(f"  Vpred: {len(vpred):,} rows, {vpred['date'].min().date()} — {vpred['date'].max().date()}")
    return daily, hourly, vpred


def main():
    global _G_DAILY, _G_HOURLY, _G_VPRED

    t0 = time.time()
    print("=" * 70)
    print("Walk-Forward A/B/C/D Strategy Pipeline (V3)")
    print(f"Tickers: {len(TICKERS)}, Strategies: {len(STRATEGY_IDS)}, TFs: {len(TIMEFRAMES)}")
    print(f"A test years: {A_TEST_YEARS[0]}-{A_TEST_YEARS[-1]}")
    print(f"B/C/D test years: {BCD_TEST_YEARS[0]}-{BCD_TEST_YEARS[-1]}")
    print(f"Hourly sigma: v2_sqrtN (no variant loop)")
    print(f"Commission: {COMMISSION} per side")
    print("=" * 70)

    daily, hourly, vpred = load_data()

    # Grid sizes
    for sid in STRATEGY_IDS:
        cat = CATEGORY[sid]
        sg = len(expand_grid(SIGNAL_GRIDS[sid]))
        if cat == "Trend":
            rg = len(TREND_RM_GRID)
        else:
            for tf in TIMEFRAMES:
                rg = len(expand_grid(RM_GRIDS_V3[(cat, tf)]))
                print(f"  {STRATEGY_NAMES[sid]} {tf}: {sg} signal x {rg} RM = {sg * rg} combos")
            continue
        for tf in TIMEFRAMES:
            print(f"  {STRATEGY_NAMES[sid]} {tf}: {sg} signal x {rg} RM = {sg * rg} combos")

    for cat in sorted(set(CATEGORY.values())):
        bg = len(expand_grid(B_GRIDS_V3[cat]))
        print(f"  B grid ({cat}): {bg} combos")

    n_c_trend = len(C_THRESHOLDS) * len(C_LOOKBACKS) * len(C_HORIZONS) * len(C_DIRECTION) * len(C_TERM_FILTER) * len(C_HYSTERESIS)
    n_c_range = len(C_RANGE_BANDS) * len(C_LOOKBACKS) * len(C_HORIZONS) * len(C_DIRECTION) * len(C_TERM_FILTER) * len(C_HYSTERESIS)
    print(f"  C grid (Trend/Contr): {n_c_trend} combos")
    print(f"  C grid (Range): {n_c_range} combos")

    n_d_std = len(D_TARGET_VOLS) * len(D_MAX_LEVS) * len(D_GAMMA) * len(D_VOL_FLOORS) * len(D_VOL_CAPS) * len(D_SMOOTH) * len(D_HORIZONS)
    n_d_inv = len(D_INV_LOOKBACKS) * len(D_MAX_LEVS) * len(D_SMOOTH) * len(D_HORIZONS)
    print(f"  D grid: {n_d_std} standard + {n_d_inv} inverse = {n_d_std + n_d_inv} combos")

    print(f"\nWarming up numba...")
    warmup_numba()

    print(f"\nProcessing {len(TICKERS)} tickers with {N_WORKERS} workers...")

    all_results = {}
    all_trade_rows = []
    all_position_rows = []

    if N_WORKERS > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(N_WORKERS, initializer=_pool_init, initargs=(daily, hourly, vpred)) as pool:
            for i, (ticker, result_dict) in enumerate(pool.imap_unordered(_worker_func, TICKERS)):
                all_results[ticker] = result_dict["results"]
                all_trade_rows.extend(result_dict["trades"])
                all_position_rows.extend(result_dict["positions"])
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({elapsed:.0f}s)")
    else:
        for i, ticker in enumerate(TICKERS):
            result_dict = process_ticker(ticker, daily, hourly, vpred)
            all_results[ticker] = result_dict["results"]
            all_trade_rows.extend(result_dict["trades"])
            all_position_rows.extend(result_dict["positions"])
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(TICKERS)}] {ticker} done ({elapsed:.0f}s)")

    print(f"\nAll tickers done in {time.time() - t0:.0f}s")

    print("\nGenerating outputs...")
    generate_outputs(all_results, all_trade_rows, all_position_rows)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
