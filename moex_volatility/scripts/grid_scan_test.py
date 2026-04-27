#!/usr/bin/env python3
"""
grid_scan_test.py — Phase 1: Comprehensive grid scan on TEST (OOS 2022-2026).

Architecture (3-tier):
  Tier 1: Full signal grid × default RM → find promising signals
  Tier 2: Promising signals × random filters(20) × default RM → find good combos
  Tier 3: Good combos × random RM(100) → find optimal RM

Signal confirmations: none, rsi30, rsi20, volume, close_confirm
Filters: ADX, ATR pctile, Volume, RSI, Momentum, DayOfWeek, Gap
"""
import os, sys, warnings, time, json, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

random.seed(42); np.random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP, load_data, compute_base, calc_rsi,
    calc_sma, calc_std, calc_atr, calc_adx,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)
from daily_upper_bound import bt_universal

KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
BASE = Path(__file__).resolve().parent.parent
BPY = 252


def calc_trading_sharpe(pos, close, log_ret, dates, rf_daily, cal_days, mask, comm):
    """Trading Sharpe: compound return - compound RF, vol on ALL days. No FDR."""
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    dm = dates[mask]; n = len(pm)
    if n < 30: return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm  # trading return with commission, NO FDR
    exp = np.abs(pm); active = exp > 1e-8; na = active.sum()
    if na < 10: return -999.0, {}
    cs = (dm[-1] - dm[0]).days
    if cs < 30: return -999.0, {}
    cum = np.prod(1.0 + net) - 1.0
    ann = (1.0 + cum) ** (365.0 / cs) - 1.0
    rf_c = np.prod(1.0 + rf * cd) - 1.0
    rf_a = (1.0 + rf_c) ** (365.0 / cs) - 1.0
    vol = np.std(net, ddof=1) * sqrt(BPY)  # ALL days (not just active)
    if vol < 1e-6: return 0.0, {}
    ex = ann - rf_a
    exsh = ex / vol
    ny = cs / 365.0
    nt = 0; nw = 0; in_t = False; tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t: in_t = True; tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1
                if tr > 0: nw += 1
                in_t = False
    return exsh, dict(total_ret=round(ann * 100, 2), vol=round(vol * 100, 2),
                      exposure=round(na / n * 100, 1),
                      tpy=round(nt / max(ny, 0.01), 1),
                      wr=round(nw / nt * 100 if nt > 0 else 0, 1))

# ═══════════════════════════════════════════════════════════════
# PARAMETER GRIDS
# ═══════════════════════════════════════════════════════════════

MA_TYPES = ["sma", "ema", "dema", "hull", "kama"]
WINDOWS_D = [5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]
THRESHOLDS_S1 = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
BB_STDS = [0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0]
DC_WINDOWS = [3, 5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120]
ST_PERIODS = [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 35, 40, 50, 60, 80]
ST_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
PIVOT_TYPES = ["classic", "fibonacci", "woodie", "camarilla"]
VWAP_WINDOWS = [3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100]
VWAP_DM = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]

CONFIRMATIONS = ["none", "rsi30", "rsi20", "volume"]
IS_TREND = {"S1": False, "S2": False, "S3": True, "S4": True, "S5": False, "S6": False}

DEFAULT_RM = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=3.0, trail=0, mh=15, be=0, chand=0, rr=0)
DEFAULT_RM_TREND = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=0, trail=2.0, mh=0, be=1.0, chand=0, rr=0)

# Thresholds for tiered search
T1_THRESHOLD = -0.1   # Tier 1 → Tier 2 (try filters)
T2_THRESHOLD = 0.2    # Tier 2 → Tier 3 (try RM variants)
N_FILTERS = 10        # random filter combos per promising signal
N_RM = 50             # random RM combos per good (signal, filter)
MAX_RESULTS_PER_TICKER = 500  # keep only top-N results per ticker


# ═══════════════════════════════════════════════════════════════
# MA HELPERS
# ═══════════════════════════════════════════════════════════════

def _ema(arr, w):
    a = 2.0 / (w + 1); out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def _dema(arr, w):
    e1 = _ema(arr, w); e2 = _ema(e1, w); return 2 * e1 - e2

def _hull(arr, w):
    h = max(w // 2, 1)
    wts_h = np.arange(1, h + 1, dtype=np.float64)
    wts_f = np.arange(1, w + 1, dtype=np.float64)
    wma_h = np.full_like(arr, np.nan); wma_f = np.full_like(arr, np.nan)
    for i in range(h - 1, len(arr)):
        wma_h[i] = np.dot(arr[i - h + 1:i + 1], wts_h) / wts_h.sum()
    for i in range(w - 1, len(arr)):
        wma_f[i] = np.dot(arr[i - w + 1:i + 1], wts_f) / wts_f.sum()
    diff = 2 * wma_h - wma_f
    sw = max(int(sqrt(w)), 1); wts_s = np.arange(1, sw + 1, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    for i in range(sw - 1, len(diff)):
        if not np.isnan(diff[i - sw + 1]):
            out[i] = np.dot(diff[i - sw + 1:i + 1], wts_s) / wts_s.sum()
    return out

def _kama(arr, w):
    fc = 2.0 / 3; sc = 2.0 / 31
    out = np.full_like(arr, np.nan); out[w - 1] = arr[w - 1]
    for i in range(w, len(arr)):
        d = abs(arr[i] - arr[i - w])
        v = np.sum(np.abs(np.diff(arr[i - w:i + 1])))
        er = d / v if v > 1e-12 else 0
        s = (er * (fc - sc) + sc) ** 2
        out[i] = out[i - 1] + s * (arr[i] - out[i - 1])
    return out

def get_ma(arr, w, mt):
    if mt == "sma": return calc_sma(arr, w)
    elif mt == "ema": return _ema(arr, w)
    elif mt == "dema": return _dema(arr, w)
    elif mt == "hull": return _hull(arr, w)
    elif mt == "kama": return _kama(arr, w)
    return calc_sma(arr, w)


# ═══════════════════════════════════════════════════════════════
# PRECOMPUTE INDICATORS (per ticker, ONCE)
# ═══════════════════════════════════════════════════════════════

def precompute_indicators(close, high, low, volume, open_arr, windows):
    n = len(close)
    ind = {}

    ind["atr14"] = calc_atr(high, low, close, 14)
    ind["rsi14"] = calc_rsi(close, 14)
    ind["adx14"] = calc_adx(high, low, close, 14)
    ind["vol_sma20"] = calc_sma(volume, 20)

    # Rolling high/low for Chandelier (11 bars for daily)
    ind["rh"] = pd.Series(high).rolling(11, min_periods=1).max().values
    ind["rl"] = pd.Series(low).rolling(11, min_periods=1).min().values

    # Precompute MAs and stds
    ind["ma"] = {}; ind["std"] = {}
    for mt in MA_TYPES:
        for w in windows:
            if w >= n - 10: continue
            try:
                ind["ma"][(mt, w)] = get_ma(close, w, mt)
                ind["std"][(mt, w)] = calc_std(close, w)
            except:
                pass

    # Z-scores: (close - MA) / ATR  (for S1)
    ind["z_atr"] = {}
    atr = ind["atr14"]
    for key, ma in ind["ma"].items():
        z = (close - ma) / np.where(atr > 1e-12, atr, 1.0)
        z[np.isnan(ma)] = np.nan
        ind["z_atr"][key] = z

    # Z-scores: (close - MA) / std  (for S2)
    ind["z_std"] = {}
    for key in ind["ma"]:
        ma = ind["ma"][key]
        std = ind["std"].get(key)
        if std is not None:
            z = (close - ma) / np.where((std > 1e-12) & ~np.isnan(std), std, 1.0)
            z[np.isnan(ma)] = np.nan
            ind["z_std"][key] = z

    # Day of week (0=Mon ... 4=Fri) — will be set from dates later
    ind["volume"] = volume

    return ind


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════

def gen_s1(z, threshold, warmup):
    """S1 Mean Reversion: buy when z < -threshold, sell when z > threshold."""
    sig = np.zeros(len(z), dtype=np.int8)
    v = ~np.isnan(z)
    sig[v & (z < -threshold)] = 1
    sig[v & (z > threshold)] = -1
    sig[:warmup] = 0
    return sig

def gen_s2(z, bs, warmup):
    """S2 Bollinger: buy when z < -bs, sell when z > bs."""
    sig = np.zeros(len(z), dtype=np.int8)
    v = ~np.isnan(z)
    sig[v & (z < -bs)] = 1
    sig[v & (z > bs)] = -1
    sig[:warmup] = 0
    return sig

def gen_s3(close, dc_cache, w, warmup):
    """S3 Donchian breakout."""
    if w not in dc_cache: return None
    hc, lc = dc_cache[w]
    ph = np.empty_like(hc); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty_like(lc); pl[0] = np.nan; pl[1:] = lc[:-1]
    v = ~np.isnan(ph)
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close > ph)] = 1
    sig[v & (close < pl)] = -1
    sig[:warmup] = 0
    return sig

def gen_s4(close, st_cache, ap, mult, warmup):
    """S4 Supertrend crossover."""
    key = (ap, mult)
    if key not in st_cache: return None
    st, _ = st_cache[key]
    n = len(close)
    sp = np.empty_like(st); sp[0] = np.nan; sp[1:] = st[:-1]
    cp = np.empty_like(close); cp[0] = close[0]; cp[1:] = close[:-1]
    v = ~np.isnan(st) & ~np.isnan(sp)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close > st) & (cp <= sp)] = 1
    sig[v & (close < st) & (cp >= sp)] = -1
    sig[:warmup] = 0
    return sig

def gen_s5(close, pivot_data, pt, el, warmup):
    """S5 PivotPoints."""
    if pt not in pivot_data: return None
    P, S1r, R1r, S2r, R2r, S3r, R3r = pivot_data[pt]
    if el == 2: S, R = S2r, R2r
    elif el == 3: S, R = S3r, R3r
    else: S, R = S1r, R1r
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    sig[~np.isnan(S) & (close < S)] = 1
    sig[~np.isnan(R) & (close > R)] = -1
    sig[:warmup] = 0
    return sig

def gen_s6(close, vwap_cache, vw, dm, warmup):
    """S6 VWAP deviation."""
    if vw not in vwap_cache: return None
    vwap, dev = vwap_cache[vw]
    v = ~np.isnan(vwap)
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close < vwap - dm * dev)] = 1
    sig[v & (close > vwap + dm * dev)] = -1
    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# CONFIRMATIONS (applied to raw signal)
# ═══════════════════════════════════════════════════════════════

def apply_confirmation(sig, conf_type, rsi, volume, vol_sma20, is_trend):
    """Apply confirmation filter to signal. Returns modified copy."""
    if conf_type == "none": return sig
    out = sig.copy()
    if conf_type == "rsi30":
        if is_trend:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi < 50))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi > 50))] = 0
        else:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi > 30))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi < 70))] = 0
    elif conf_type == "rsi20":
        if is_trend:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi < 55))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi > 45))] = 0
        else:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi > 20))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi < 80))] = 0
    elif conf_type == "volume":
        vs = np.where(np.isnan(vol_sma20) | (vol_sma20 < 1), 1.0, vol_sma20)
        out[volume / vs < 1.0] = 0
    return out


# ═══════════════════════════════════════════════════════════════
# FILTERS (random combinations)
# ═══════════════════════════════════════════════════════════════

def random_filter_params(is_trend):
    """Generate random filter parameter dict."""
    fp = {}
    if random.random() < 0.35:
        fp["adx_th"] = random.choice([15, 20, 25, 30, 35, 40])
    if random.random() < 0.25:
        fp["atr_lo"] = random.choice([5, 10, 15, 20, 25])
    if random.random() < 0.25:
        fp["vol_th"] = random.choice([0.5, 0.7, 1.0, 1.2, 1.5])
    if random.random() < 0.20:
        fp["rsi_th"] = random.choice([20, 25, 30, 35, 40])
    if random.random() < 0.20:
        fp["mom_n"] = random.choice([1, 3, 5, 7, 10])
    if random.random() < 0.15:
        fp["dow"] = random.choice([1, 2, 3])
    return fp


def apply_filters(sig, close, volume, ind, atr_arr, day_of_week, fp, is_trend):
    """Apply filter params to signal. Modifies in-place and returns."""
    if not fp: return sig
    n = len(sig)
    mask = np.ones(n, dtype=bool)
    adx = ind["adx14"]
    rsi = ind["rsi14"]

    # ADX filter
    at = fp.get("adx_th", 0)
    if at > 0:
        if is_trend:
            mask &= (~np.isnan(adx)) & (adx > at)
        else:
            mask &= (~np.isnan(adx)) & (adx < at)

    # ATR percentile filter
    al = fp.get("atr_lo", 0)
    if al > 0:
        va = atr_arr[~np.isnan(atr_arr)]
        if len(va) > 50:
            mask &= (~np.isnan(atr_arr)) & (atr_arr > np.percentile(va, al))

    # Volume filter
    vt = fp.get("vol_th", 0)
    if vt > 0:
        vs = ind["vol_sma20"]
        vs_s = np.where(np.isnan(vs) | (vs < 1), 1.0, vs)
        mask &= volume / vs_s > vt

    # RSI filter
    rt = fp.get("rsi_th", 0)
    if rt > 0:
        if not is_trend:
            sig[(sig == 1) & ~((~np.isnan(rsi)) & (rsi < rt))] = 0
            sig[(sig == -1) & ~((~np.isnan(rsi)) & (rsi > (100 - rt)))] = 0
        else:
            sig[(sig == 1) & ~((~np.isnan(rsi)) & (rsi > 50))] = 0
            sig[(sig == -1) & ~((~np.isnan(rsi)) & (rsi < 50))] = 0

    # Momentum filter
    mn = fp.get("mom_n", 0)
    if mn > 0:
        mom = np.zeros(n); mom[mn:] = close[mn:] - close[:-mn]
        if is_trend:
            sig[(sig == 1) & (mom <= 0)] = 0
            sig[(sig == -1) & (mom >= 0)] = 0
        else:
            sig[(sig == 1) & (mom >= 0)] = 0
            sig[(sig == -1) & (mom <= 0)] = 0

    # Day of week filter
    dw = fp.get("dow", 0)
    if dw > 0 and day_of_week is not None:
        if dw == 1: mask &= day_of_week != 0
        elif dw == 2: mask &= day_of_week != 4
        elif dw == 3: mask &= (day_of_week != 0) & (day_of_week != 4)

    sig[~mask] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# RANDOM RM GENERATOR
# ═══════════════════════════════════════════════════════════════

def random_rm_params():
    """Generate random RM params for bt_universal."""
    r = random.random()
    ch = 0.0
    if r < 0.15:
        sl_p = random.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10])
        sl_a = 0.0
    elif r < 0.60:
        sl_p = 0.0
        sl_a = random.choice([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0])
    else:
        sl_p = 0.0; sl_a = 0.0
        ch = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

    r2 = random.random()
    rr = 0.0
    if r2 < 0.25:
        tp_p = 0.0; tp_a = 0.0
        trail = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    elif r2 < 0.50:
        tp_p = 0.0; tp_a = 0.0; trail = 0.0
        rr = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    elif r2 < 0.65:
        trail = 0.0
        tp_p = random.choice([0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15])
        tp_a = 0.0
    else:
        trail = 0.0; tp_p = 0.0
        tp_a = random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0])

    mh = random.choice([0, 3, 5, 7, 10, 15, 20, 30, 50])
    be = random.choice([0.0, 0.5, 1.0, 1.5, 2.0])

    return dict(sl_p=sl_p, tp_p=tp_p, sl_a=sl_a, tp_a=tp_a, trail=trail,
                mh=mh, be=be, chand=ch, rr=rr)


def run_bt(sig, close, high, low, atr, rh, rl, rm, warmup, end_idx):
    return bt_universal(sig, close, high, low, atr, rh, rl,
                        rm["sl_p"], rm["tp_p"], rm["sl_a"], rm["tp_a"],
                        rm["trail"], rm["mh"], rm["be"], warmup, end_idx,
                        chandelier_atr=rm.get("chand", 0.0),
                        rr_ratio=rm.get("rr", 0.0))


# ═══════════════════════════════════════════════════════════════
# SCAN ONE TICKER (3-tier architecture)
# ═══════════════════════════════════════════════════════════════

def scan_ticker_daily(ticker, daily_df, kr_s):
    """3-tier grid scan for one ticker, daily timeframe."""
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP + 200: return []

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)
    lr = np.zeros(n); lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    # Calendar/RF
    cal_days = np.ones(n)
    for i in range(1, n): cal_days[i] = (dates[i] - dates[i - 1]).days
    rf_raw = kr_s.reindex(dates, method="ffill").values
    rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0

    # OOS mask
    oos_start = _year_bounds(dates, 2022)[0]
    if oos_start <= WARMUP: return []
    oos_mask = np.zeros(n, dtype=bool); oos_mask[oos_start:] = True

    # Precompute indicators
    ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
    atr = ind["atr14"]; rh = ind["rh"]; rl = ind["rl"]
    rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
    day_of_week = dates.dayofweek.values

    # Precompute strategy-specific caches
    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS)
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}

    # Pre-generate random samples
    rm_samples = [random_rm_params() for _ in range(N_RM)]
    filter_samples_ct = [random_filter_params(False) for _ in range(N_FILTERS)]
    filter_samples_tr = [random_filter_params(True) for _ in range(N_FILTERS)]

    comm = 0.0005  # daily institutional
    results = []
    n_bt = 0

    def eval_signal(sid, sig_raw, sig_params_str, is_trend):
        """3-tier evaluation of a signal."""
        nonlocal n_bt
        local_results = []

        # Pick default RM based on strategy type
        def_rm = DEFAULT_RM_TREND if is_trend else DEFAULT_RM

        # ── TIER 1: raw signal × default RM ──
        for conf in CONFIRMATIONS:
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
            if np.sum(np.abs(sig[oos_mask]) > 0) < 3: continue

            pos = run_bt(sig, close, high, low, atr, rh, rl, def_rm, WARMUP, n)
            sh, info = calc_trading_sharpe(pos, close, lr, dates, rf_d, cal_days, oos_mask, comm)
            n_bt += 1

            if sh > -0.5:  # only record reasonable results
                local_results.append({
                    "sid": sid, "ticker": ticker, "params": sig_params_str,
                    "conf": conf, "filter": "{}", "rm": "default",
                    "sharpe": round(sh, 3), "ret": info.get("total_ret", 0),
                    "vol": info.get("vol", 0), "exp": info.get("exposure", 0),
                    "wr": info.get("wr", 0), "tpy": info.get("tpy", 0)
                })

            # ── TIER 2: promising signal × random filters ──
            if sh > T1_THRESHOLD:
                fsamples = filter_samples_tr if is_trend else filter_samples_ct
                for fp in fsamples:
                    if not fp: continue
                    sig_f = sig.copy()
                    sig_f = apply_filters(sig_f, close, volume, ind, atr, day_of_week, fp, is_trend)
                    if np.sum(np.abs(sig_f[oos_mask]) > 0) < 3: continue

                    pos_f = run_bt(sig_f, close, high, low, atr, rh, rl, def_rm, WARMUP, n)
                    sh_f, info_f = calc_trading_sharpe(pos_f, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    n_bt += 1

                    if sh_f > sh:  # filter improved over raw
                        fp_str = json.dumps(fp, sort_keys=True)
                        local_results.append({
                            "sid": sid, "ticker": ticker, "params": sig_params_str,
                            "conf": conf, "filter": fp_str, "rm": "default",
                            "sharpe": round(sh_f, 3), "ret": info_f.get("total_ret", 0),
                            "vol": info_f.get("vol", 0), "exp": info_f.get("exposure", 0),
                            "wr": info_f.get("wr", 0), "tpy": info_f.get("tpy", 0)
                        })

                        # ── TIER 3: good combo × random RM ──
                        if sh_f > T2_THRESHOLD:
                            for rm in rm_samples:
                                pos_r = run_bt(sig_f, close, high, low, atr, rh, rl, rm, WARMUP, n)
                                sh_r, info_r = calc_trading_sharpe(pos_r, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                                n_bt += 1
                                if sh_r > sh_f:
                                    rm_str = json.dumps({k: v for k, v in rm.items() if v != 0}, sort_keys=True)
                                    local_results.append({
                                        "sid": sid, "ticker": ticker, "params": sig_params_str,
                                        "conf": conf, "filter": fp_str, "rm": rm_str,
                                        "sharpe": round(sh_r, 3), "ret": info_r.get("total_ret", 0),
                                        "vol": info_r.get("vol", 0), "exp": info_r.get("exposure", 0),
                                        "wr": info_r.get("wr", 0), "tpy": info_r.get("tpy", 0)
                                    })

                # Also try RM search WITHOUT filter for promising signals
                if sh > T2_THRESHOLD:
                    for rm in rm_samples:
                        pos_r = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                        sh_r, info_r = calc_trading_sharpe(pos_r, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                        n_bt += 1
                        if sh_r > sh:
                            rm_str = json.dumps({k: v for k, v in rm.items() if v != 0}, sort_keys=True)
                            local_results.append({
                                "sid": sid, "ticker": ticker, "params": sig_params_str,
                                "conf": conf, "filter": "{}", "rm": rm_str,
                                "sharpe": round(sh_r, 3), "ret": info_r.get("total_ret", 0),
                                "vol": info_r.get("vol", 0), "exp": info_r.get("exposure", 0),
                                "wr": info_r.get("wr", 0), "tpy": info_r.get("tpy", 0)
                            })

        return local_results

    # ── S1: Mean Reversion ──
    for (mt, w), z in ind["z_atr"].items():
        for thresh in THRESHOLDS_S1:
            sig = gen_s1(z, thresh, WARMUP)
            sp = json.dumps({"ma": mt, "w": w, "thresh": thresh})
            results.extend(eval_signal("S1", sig, sp, False))

    # ── S2: Bollinger ──
    for (mt, w), z in ind["z_std"].items():
        for bs in BB_STDS:
            sig = gen_s2(z, bs, WARMUP)
            sp = json.dumps({"ma": mt, "w": w, "bs": bs})
            results.extend(eval_signal("S2", sig, sp, False))

    # ── S3: Donchian ──
    for w in DC_WINDOWS:
        sig = gen_s3(close, dc_cache, w, WARMUP)
        if sig is None: continue
        sp = json.dumps({"w": w})
        results.extend(eval_signal("S3", sig, sp, True))

    # ── S4: Supertrend ──
    for ap in ST_PERIODS:
        for mult in ST_MULTS:
            sig = gen_s4(close, st_cache, ap, mult, WARMUP)
            if sig is None: continue
            sp = json.dumps({"ap": ap, "mult": mult})
            results.extend(eval_signal("S4", sig, sp, True))

    # ── S5: PivotPoints ──
    for pt in PIVOT_TYPES:
        for el in [1, 2, 3]:
            sig = gen_s5(close, pivot_data, pt, el, WARMUP)
            if sig is None: continue
            sp = json.dumps({"pt": pt, "el": el})
            results.extend(eval_signal("S5", sig, sp, False))

    # ── S6: VWAP ──
    for vw in VWAP_WINDOWS:
        for dm in VWAP_DM:
            sig = gen_s6(close, vwap_cache, vw, dm, WARMUP)
            if sig is None: continue
            sp = json.dumps({"vw": vw, "dm": dm})
            results.extend(eval_signal("S6", sig, sp, False))

    # Keep only top results to avoid memory explosion
    if len(results) > MAX_RESULTS_PER_TICKER:
        results.sort(key=lambda x: x["sharpe"], reverse=True)
        results = results[:MAX_RESULTS_PER_TICKER]

    return results, n_bt


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 100)
    print("GRID SCAN — Phase 1: 3-tier (signal+confirm → filters → RM)")
    print("Metric: calc_trading_sharpe (compound - RF, NO FDR, vol ALL days × √252)")
    print(f"Confirmations: {CONFIRMATIONS}")
    print(f"Tiers: T1>{T1_THRESHOLD} → filters({N_FILTERS}) → T2>{T2_THRESHOLD} → RM({N_RM})")
    print("=" * 100)

    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    all_results = []
    total_bt = 0
    for i, ticker in enumerate(TICKERS):
        t1 = time.time()
        res, n_bt = scan_ticker_daily(ticker, daily_df, kr_s)
        elapsed = time.time() - t1
        all_results.extend(res)
        total_bt += n_bt
        n_pos = sum(1 for r in res if r["sharpe"] > 0)
        n_gt1 = sum(1 for r in res if r["sharpe"] > 1.0)
        best_sh = max((r["sharpe"] for r in res), default=-999)
        print(f"  [{i+1}/17] {ticker}: {len(res)} results, {n_bt} bt_calls, "
              f"{n_pos} Sh>0, {n_gt1} Sh>1.0, best={best_sh:+.2f} ({elapsed:.1f}s)")

    # Save to parquet
    df = pd.DataFrame(all_results)
    out_path = BASE / "results/final/strategies/walkforward_v4/grid_scan_daily_phase1.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    # ── Summary ──
    elapsed_total = time.time() - t0
    print(f"\n{'='*100}")
    print(f"SUMMARY — {len(df)} total results, {total_bt} bt_universal calls")
    print(f"{'='*100}")

    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        sub = df[df["sid"] == sid]
        if len(sub) == 0: continue
        sh = sub["sharpe"]
        print(f"  {sid}: N={len(sub)}, mean={sh.mean():+.2f}, median={sh.median():+.2f}, "
              f"max={sh.max():+.2f}, >0={(sh>0).sum()}, >1.0={(sh>1.0).sum()}")

    # ── Top-10 per strategy ──
    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        sub = df[df["sid"] == sid].nlargest(10, "sharpe")
        if len(sub) == 0: continue
        print(f"\n  Top-10 {sid}:")
        for _, r in sub.iterrows():
            print(f"    Sh={r['sharpe']:+.2f}  Ret={r['ret']:+.1f}%  Vol={r['vol']:.0f}%  "
                  f"Exp={r['exp']:.0f}%  WR={r['wr']:.0f}%  Tr/y={r['tpy']:.0f}  "
                  f"{r['ticker']:5s}  conf={r['conf']:6s}  {r['params']}")

    # ── Cross-ticker robustness ──
    print(f"\n{'='*100}")
    print("CROSS-TICKER ROBUSTNESS: params that work on 3+ tickers (Sh>0)")
    print(f"{'='*100}")

    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        sub = df[(df["sid"] == sid) & (df["sharpe"] > 0)]
        if len(sub) == 0: continue
        # Group by params, count distinct tickers
        g = sub.groupby("params").agg(
            n_tickers=("ticker", "nunique"),
            mean_sh=("sharpe", "mean"),
            max_sh=("sharpe", "max"),
            tickers=("ticker", lambda x: ",".join(sorted(x.unique())))
        ).sort_values("n_tickers", ascending=False)
        g3 = g[g["n_tickers"] >= 3]
        if len(g3) > 0:
            print(f"\n  {sid} — {len(g3)} param combos work on 3+ tickers:")
            for params, row in g3.head(10).iterrows():
                print(f"    {row['n_tickers']}tk  mean={row['mean_sh']:+.2f}  max={row['max_sh']:+.2f}  "
                      f"{params}  [{row['tickers']}]")
        else:
            print(f"\n  {sid} — no params work on 3+ tickers")

    print(f"\nDONE in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
