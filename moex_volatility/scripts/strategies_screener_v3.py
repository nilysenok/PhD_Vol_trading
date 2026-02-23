#!/usr/bin/env python3
"""
strategies_screener_v3.py — Aggressive filters for contrarian + range.
Trend strategies unchanged from v2.

Key changes vs v2:
  Contrarian (S1-S3): Exhaustion filters (consecutive candles, volume exhaustion,
                      RSI divergence, MA slope)
  Range (S7-S8): Strict range definition (Bollinger squeeze, flat MA, Hurst proxy,
                 no-recent-breakout)
  Trend (S4-S6): UNCHANGED + multi-TF confirmation for S4h/S5h
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results" / "final" / "strategies" / "data"
OUT_DIR = BASE / "results" / "final" / "strategies"
OUT_DATA = OUT_DIR / "data"
OUT_TABLES = OUT_DIR / "tables"

TICKERS = sorted([
    "AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
    "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"
])

WARMUP = 200
START_DATE = pd.Timestamp("2020-01-01")


# ════════════════════════════════════════════════════════════
# Technical indicators (from v2 + new)
# ════════════════════════════════════════════════════════════

def calc_sma(arr, w):
    return pd.Series(arr).rolling(w).mean().values

def calc_ema(arr, w):
    return pd.Series(arr).ewm(span=w, adjust=False).mean().values

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

def calc_stochastic_k(high, low, close, period=14, smooth=3):
    n = len(close)
    raw_k = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        if hh - ll > 1e-12:
            raw_k[i] = (close[i] - ll) / (hh - ll) * 100
        else:
            raw_k[i] = 50.0
    k = pd.Series(raw_k).rolling(smooth).mean().values
    return k

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


# ── NEW indicators for v3 ──

def calc_hurst_proxy(close, window=20):
    """Rolling Hurst exponent estimate via R/S method."""
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
# Precompute indicators per ticker
# ════════════════════════════════════════════════════════════

def precompute(close, high, low, open_arr, volume, is_hourly=False):
    """Compute all indicators needed by any strategy."""
    n = len(close)
    ind = {}

    # Basics
    ind["sma20"] = calc_sma(close, 20)
    ind["std20"] = calc_std(close, 20)
    ind["z"] = (close - ind["sma20"]) / ind["std20"]

    # Bollinger
    ind["bb_upper"] = ind["sma20"] + 2.0 * ind["std20"]
    ind["bb_lower"] = ind["sma20"] - 2.0 * ind["std20"]
    ma20_safe = np.where(np.abs(ind["sma20"]) > 1e-12, ind["sma20"], 1e-12)
    ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / ma20_safe
    bw_window = 252 * 9 if is_hourly else 252
    ind["bw_p25"] = pd.Series(ind["bb_width"]).rolling(bw_window, min_periods=50).quantile(0.25).values
    ind["bw_p30"] = pd.Series(ind["bb_width"]).rolling(bw_window, min_periods=50).quantile(0.30).values

    # RSI, Stochastic
    ind["rsi14"] = calc_rsi(close, 14)
    ind["stoch_k"] = calc_stochastic_k(high, low, close, 14, 3)

    # ADX, ATR
    ind["adx14"] = calc_adx(high, low, close, 14)
    ind["atr14"] = calc_atr(high, low, close, 14)

    # Vol-regime filter: ATR(14) / SMA(ATR(14), 50)
    atr_sma50 = calc_sma(ind["atr14"], 50)
    atr_sma50_safe = np.where(
        np.isnan(atr_sma50) | (atr_sma50 < 1e-12), 1e-12, atr_sma50
    )
    ind["vol_regime"] = ind["atr14"] / atr_sma50_safe

    # Donchian
    ind["high_ch20"] = pd.Series(high).rolling(20).max().values
    ind["low_ch20"] = pd.Series(low).rolling(20).min().values

    # Volume
    ind["vol_sma20"] = calc_sma(volume, 20)
    ind["vol_sma5"] = calc_sma(volume, 5)

    # Supertrend
    ind["supertrend"], ind["st_dir"] = calc_supertrend(high, low, close, 14, 3.0)

    # Dual MA
    if is_hourly:
        ind["ma_fast"] = calc_sma(close, 20)
        ind["ma_slow"] = calc_sma(close, 80)
    else:
        ind["ma_fast"] = calc_sma(close, 50)
        ind["ma_slow"] = calc_sma(close, 200)

    # Keltner
    ind["ema20"] = calc_ema(close, 20)
    ind["kc_upper"] = ind["ema20"] + 1.5 * ind["atr14"]
    ind["kc_lower"] = ind["ema20"] - 1.5 * ind["atr14"]

    # ── NEW for v3 ──

    # Consecutive candles (red/green count in last 5)
    red = (close < open_arr).astype(float)
    green = (close > open_arr).astype(float)
    ind["red_count5"] = pd.Series(red).rolling(5).sum().values
    ind["green_count5"] = pd.Series(green).rolling(5).sum().values

    # RSI rolling min/max for divergence (5-bar window)
    ind["rsi_min5"] = pd.Series(ind["rsi14"]).rolling(5).min().values
    ind["rsi_max5"] = pd.Series(ind["rsi14"]).rolling(5).max().values
    ind["close_min5"] = pd.Series(close).rolling(5).min().values
    ind["close_max5"] = pd.Series(close).rolling(5).max().values

    # MA slopes for flat detection
    sma20 = ind["sma20"]
    sma20_lag10 = np.full(n, np.nan)
    sma20_lag10[10:] = sma20[:-10]
    sma20_safe = np.where(np.abs(sma20_lag10) > 1e-12, sma20_lag10, 1e-12)
    ind["sma20_slope10"] = (sma20 - sma20_lag10) / sma20_safe

    # SMA(50) slope for S3 daily
    sma50 = calc_sma(close, 50)
    sma50_lag10 = np.full(n, np.nan)
    sma50_lag10[10:] = sma50[:-10]
    sma50_safe = np.where(np.abs(sma50_lag10) > 1e-12, sma50_lag10, 1e-12)
    ind["sma50_slope10"] = (sma50 - sma50_lag10) / sma50_safe

    # Hurst proxy for range strategies
    ind["hurst_proxy"] = calc_hurst_proxy(close, 20)

    # No-recent-breakout for S8 (no Donchian breakout in last 10 bars)
    breakout = np.zeros(n, dtype=bool)
    for i in range(1, n):
        hc = ind["high_ch20"][i - 1]
        lc = ind["low_ch20"][i - 1]
        if not np.isnan(hc) and not np.isnan(lc):
            breakout[i] = (close[i] > hc) or (close[i] < lc)
    no_breakout = np.full(n, False)
    for i in range(9, n):
        no_breakout[i] = not np.any(breakout[i - 9:i + 1])
    ind["no_recent_breakout"] = no_breakout

    return ind


# ════════════════════════════════════════════════════════════
# Daily lookup for multi-TF alignment
# ════════════════════════════════════════════════════════════

def build_daily_lookups(daily):
    """Precompute daily indicators per ticker for multi-TF filters."""
    lookups = {}
    for ticker in TICKERS:
        tdf = daily[daily["ticker"] == ticker].sort_values("date")
        close_d = tdf["close"].values
        dates_d = tdf["date"].values

        ma20_d = calc_sma(close_d, 20)
        ma50_d = calc_sma(close_d, 50)
        ma200_d = calc_sma(close_d, 200)

        # MA50 > MA200 for trend confirmation
        ma50_above = np.full(len(close_d), np.nan)
        valid = ~np.isnan(ma50_d) & ~np.isnan(ma200_d)
        ma50_above[valid] = (ma50_d[valid] > ma200_d[valid]).astype(float)

        # Daily MA20 slope over 5 days for S3 hourly
        ma20_lag5 = np.full_like(ma20_d, np.nan)
        ma20_lag5[5:] = ma20_d[:-5]
        ma20_safe = np.where(np.abs(ma20_lag5) > 1e-12, ma20_lag5, 1e-12)
        ma_slope = np.where(~np.isnan(ma20_lag5), (ma20_d - ma20_lag5) / ma20_safe, np.nan)

        lookups[ticker] = {
            "dates": np.array(dates_d, dtype="datetime64[D]"),
            "ma50_above_ma200": ma50_above,
            "ma_slope_daily": ma_slope,
        }
    return lookups


def align_daily_to_hourly(daily_lookup, hourly_dts):
    """For each hourly datetime, find previous day's daily indicators."""
    d_dates = daily_lookup["dates"]
    n = len(hourly_dts)

    h_dates = np.array(hourly_dts, dtype="datetime64[D]")
    indices = np.searchsorted(d_dates, h_dates, side="left") - 1
    valid = indices >= 0

    ma50_above = np.full(n, np.nan)
    ma_slope = np.full(n, np.nan)
    ma50_above[valid] = daily_lookup["ma50_above_ma200"][indices[valid]]
    ma_slope[valid] = daily_lookup["ma_slope_daily"][indices[valid]]

    return ma50_above, ma_slope


# ════════════════════════════════════════════════════════════
# Strategy entry/exit functions (v3)
# ════════════════════════════════════════════════════════════

# --- CONTRARIAN S1: exhaustion filters ---

def s1_entry(t, c, ind, tracker=None):
    z = ind["z"][t]
    rsi = ind["rsi14"][t]
    if np.isnan(z) or np.isnan(rsi):
        return 0
    sig = 0
    if z < -2.0:
        sig = 1
    elif z > 2.0:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1

    # F1: vol regime < 1.2
    vr = ind["vol_regime"][t]
    if np.isnan(vr) or vr >= 1.2:
        return 0
    if tracker is not None:
        tracker["F1_vol_regime"] += 1

    # F2: consecutive candles (3/5 in move direction)
    if sig == 1:
        cnt = ind["red_count5"][t]
    else:
        cnt = ind["green_count5"][t]
    if np.isnan(cnt) or cnt < 3:
        return 0
    if tracker is not None:
        tracker["F2_consec_candles"] += 1

    # F3: volume exhaustion (volume dropping)
    vs5 = ind["vol_sma5"][t]
    vol = ind["_volume"][t]
    if not np.isnan(vs5) and vs5 > 0 and vol >= vs5:
        return 0
    if tracker is not None:
        tracker["F3_vol_exhaust"] += 1

    # F4: RSI < 40 for long / > 60 for short
    if sig == 1 and rsi >= 40:
        return 0
    if sig == -1 and rsi <= 60:
        return 0
    if tracker is not None:
        tracker["F4_rsi"] += 1

    return sig

def s1_exit(t, c, ind, pos):
    z = ind["z"][t]
    if np.isnan(z):
        return False
    return abs(z) < 0.5

# --- CONTRARIAN S2: exhaustion + RSI divergence ---

def s2_entry(t, c, ind, tracker=None):
    cl = c[t]
    bw = ind["bb_width"][t]
    bw_p25 = ind["bw_p25"][t]
    lo, up = ind["bb_lower"][t], ind["bb_upper"][t]
    if np.isnan(bw_p25):
        return 0
    sig = 0
    if cl <= lo:
        sig = 1
    elif cl >= up:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1

    # F1: vol regime < 1.2
    vr = ind["vol_regime"][t]
    if np.isnan(vr) or vr >= 1.2:
        return 0
    if tracker is not None:
        tracker["F1_vol_regime"] += 1

    # F2: consecutive candles 3/5
    if sig == 1:
        cnt = ind["red_count5"][t]
    else:
        cnt = ind["green_count5"][t]
    if np.isnan(cnt) or cnt < 3:
        return 0
    if tracker is not None:
        tracker["F2_consec_candles"] += 1

    # F3: volume exhaustion
    vs5 = ind["vol_sma5"][t]
    vol = ind["_volume"][t]
    if not np.isnan(vs5) and vs5 > 0 and vol >= vs5:
        return 0
    if tracker is not None:
        tracker["F3_vol_exhaust"] += 1

    # F4: BandWidth > P25
    if bw <= bw_p25:
        return 0
    if tracker is not None:
        tracker["F4_bw_p25"] += 1

    # F5: RSI divergence
    rsi = ind["rsi14"][t]
    if sig == 1:
        # Bullish divergence: Close = new low(5) but RSI NOT new low(5)
        close_min = ind["close_min5"][t]
        rsi_min = ind["rsi_min5"][t]
        if np.isnan(close_min) or np.isnan(rsi_min) or np.isnan(rsi):
            return 0
        if not (cl <= close_min and rsi > rsi_min):
            return 0
    else:
        # Bearish divergence: Close = new high(5) but RSI NOT new high(5)
        close_max = ind["close_max5"][t]
        rsi_max = ind["rsi_max5"][t]
        if np.isnan(close_max) or np.isnan(rsi_max) or np.isnan(rsi):
            return 0
        if not (cl >= close_max and rsi < rsi_max):
            return 0
    if tracker is not None:
        tracker["F5_rsi_divergence"] += 1

    return sig

def s2_exit(t, c, ind, pos):
    ma = ind["sma20"][t]
    if np.isnan(ma):
        return False
    if pos == 1.0 and c[t] >= ma:
        return True
    if pos == -1.0 and c[t] <= ma:
        return True
    return False

# --- CONTRARIAN S3: exhaustion + MA slope ---

def s3_entry(t, c, ind, tracker=None):
    rsi = ind["rsi14"][t]
    if np.isnan(rsi):
        return 0
    sig = 0
    if rsi < 25:
        sig = 1
    elif rsi > 75:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1

    # F1: vol regime < 1.2
    vr = ind["vol_regime"][t]
    if np.isnan(vr) or vr >= 1.2:
        return 0
    if tracker is not None:
        tracker["F1_vol_regime"] += 1

    # F2: ADX < 25 (tightened from 30)
    adx = ind["adx14"][t]
    if np.isnan(adx) or adx >= 25:
        return 0
    if tracker is not None:
        tracker["F2_adx"] += 1

    # F3: consecutive candles 3/5
    if sig == 1:
        cnt = ind["red_count5"][t]
    else:
        cnt = ind["green_count5"][t]
    if np.isnan(cnt) or cnt < 3:
        return 0
    if tracker is not None:
        tracker["F3_consec_candles"] += 1

    # F4: volume exhaustion
    vs5 = ind["vol_sma5"][t]
    vol = ind["_volume"][t]
    if not np.isnan(vs5) and vs5 > 0 and vol >= vs5:
        return 0
    if tracker is not None:
        tracker["F4_vol_exhaust"] += 1

    # F5: MA slope (flat trend)
    daily_slope = ind.get("_daily_ma_slope")
    if daily_slope is not None:
        # Hourly: use daily MA20 slope < 2%
        slope_val = daily_slope[t]
        if not np.isnan(slope_val) and abs(slope_val) >= 0.02:
            return 0
    else:
        # Daily: use SMA(50) slope over 10 bars < 1%
        slope_val = ind["sma50_slope10"][t]
        if not np.isnan(slope_val) and abs(slope_val) >= 0.01:
            return 0
    if tracker is not None:
        tracker["F5_slope"] += 1

    return sig

def s3_exit(t, c, ind, pos):
    rsi = ind["rsi14"][t]
    if np.isnan(rsi):
        return False
    if pos == 1.0 and rsi >= 50:
        return True
    if pos == -1.0 and rsi <= 50:
        return True
    return False

# --- TREND S4-S6: UNCHANGED from v2 (+ multi-TF for S4h/S5h) ---

def s4_entry(t, c, ind, tracker=None):
    if t < 1:
        return 0
    cl, adx = c[t], ind["adx14"][t]
    hc, lc = ind["high_ch20"][t - 1], ind["low_ch20"][t - 1]
    vs = ind["vol_sma20"][t]
    if np.isnan(adx) or np.isnan(hc):
        return 0
    if not np.isnan(vs) and vs > 0:
        v = ind["_volume"][t] / vs
    else:
        v = 1.0
    sig = 0
    if cl > hc and adx > 20 and v > 1.0:
        sig = 1
    elif cl < lc and adx > 20 and v > 1.0:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1

    # Multi-TF confirmation (hourly only)
    daily_trend = ind.get("_daily_ma50_above_ma200")
    if daily_trend is not None:
        dt_val = daily_trend[t]
        if not np.isnan(dt_val):
            if sig == 1 and dt_val < 0.5:
                return 0
            if sig == -1 and dt_val > 0.5:
                return 0
    if tracker is not None:
        tracker["F_multi_tf"] += 1

    return sig

def s4_exit(t, c, ind, pos):
    if t < 1:
        return False
    cl = c[t]
    hc, lc = ind["high_ch20"][t - 1], ind["low_ch20"][t - 1]
    if np.isnan(hc):
        return False
    if pos == 1.0 and cl < lc:
        return True
    if pos == -1.0 and cl > hc:
        return True
    return False

def s5_entry(t, c, ind, tracker=None):
    if t < 1:
        return 0
    cl, adx = c[t], ind["adx14"][t]
    st, st_p = ind["supertrend"][t], ind["supertrend"][t - 1]
    cp = c[t - 1]
    if np.isnan(st) or np.isnan(st_p) or np.isnan(adx):
        return 0
    sig = 0
    if cl > st and cp <= st_p and adx > 20:
        sig = 1
    elif cl < st and cp >= st_p and adx > 20:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1

    # Multi-TF confirmation (hourly only)
    daily_trend = ind.get("_daily_ma50_above_ma200")
    if daily_trend is not None:
        dt_val = daily_trend[t]
        if not np.isnan(dt_val):
            if sig == 1 and dt_val < 0.5:
                return 0
            if sig == -1 and dt_val > 0.5:
                return 0
    if tracker is not None:
        tracker["F_multi_tf"] += 1

    return sig

def s5_exit(t, c, ind, pos):
    if t < 1:
        return False
    cl, st = c[t], ind["supertrend"][t]
    cp, st_p = c[t - 1], ind["supertrend"][t - 1]
    if np.isnan(st) or np.isnan(st_p):
        return False
    if pos == 1.0 and cl < st and cp >= st_p:
        return True
    if pos == -1.0 and cl > st and cp <= st_p:
        return True
    return False

def s6_entry(t, c, ind, tracker=None):
    if t < 1:
        return 0
    f, s = ind["ma_fast"][t], ind["ma_slow"][t]
    fp, sp = ind["ma_fast"][t - 1], ind["ma_slow"][t - 1]
    adx = ind["adx14"][t]
    if np.isnan(f) or np.isnan(s) or np.isnan(fp) or np.isnan(sp) or np.isnan(adx):
        return 0
    sig = 0
    if f > s and fp <= sp and adx > 15:
        sig = 1
    elif f < s and fp >= sp and adx > 15:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1
        tracker["F_multi_tf"] = tracker.get("F_multi_tf", 0) + 1
    return sig

def s6_exit(t, c, ind, pos):
    if t < 1:
        return False
    f, s = ind["ma_fast"][t], ind["ma_slow"][t]
    fp, sp = ind["ma_fast"][t - 1], ind["ma_slow"][t - 1]
    if np.isnan(f) or np.isnan(s) or np.isnan(fp) or np.isnan(sp):
        return False
    if pos == 1.0 and f < s and fp >= sp:
        return True
    if pos == -1.0 and f > s and fp <= sp:
        return True
    return False

# --- RANGE S7: strict range definition ---

def s7_entry(t, c, ind, tracker=None):
    cl, adx = c[t], ind["adx14"][t]
    lo, up = ind["kc_lower"][t], ind["kc_upper"][t]
    if np.isnan(adx) or np.isnan(lo):
        return 0
    sig = 0
    if cl < lo:
        sig = 1
    elif cl > up:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1

    # F1: ADX < 20 (tightened from 25)
    if adx >= 20:
        return 0
    if tracker is not None:
        tracker["F1_adx"] += 1

    # F2: Bollinger squeeze — BW < P30
    bw = ind["bb_width"][t]
    bw_p30 = ind["bw_p30"][t]
    if np.isnan(bw) or np.isnan(bw_p30) or bw >= bw_p30:
        return 0
    if tracker is not None:
        tracker["F2_bw_squeeze"] += 1

    # F3: Flat MA — |SMA(20) slope over 10 bars| < 1%
    slope = ind["sma20_slope10"][t]
    if np.isnan(slope) or abs(slope) >= 0.01:
        return 0
    if tracker is not None:
        tracker["F3_flat_ma"] += 1

    # F4: Vol compression — ATR ratio < 0.9
    vr = ind["vol_regime"][t]
    if np.isnan(vr) or vr >= 0.9:
        return 0
    if tracker is not None:
        tracker["F4_vol_compress"] += 1

    # F5: Hurst proxy < 0.45 (mean-reverting)
    h = ind["hurst_proxy"][t]
    if np.isnan(h) or h >= 0.45:
        return 0
    if tracker is not None:
        tracker["F5_hurst"] += 1

    return sig

def s7_exit(t, c, ind, pos):
    ema = ind["ema20"][t]
    if np.isnan(ema):
        return False
    if pos == 1.0 and c[t] >= ema:
        return True
    if pos == -1.0 and c[t] <= ema:
        return True
    return False

# --- RANGE S8: strict range + no-recent-breakout ---

def s8_entry(t, c, ind, tracker=None):
    rsi, sk, adx = ind["rsi14"][t], ind["stoch_k"][t], ind["adx14"][t]
    if np.isnan(rsi) or np.isnan(sk) or np.isnan(adx):
        return 0
    sig = 0
    if rsi < 30 and sk < 20:
        sig = 1
    elif rsi > 70 and sk > 80:
        sig = -1
    if sig == 0:
        return 0
    if tracker is not None:
        tracker["raw"] += 1

    # F1: ADX < 20
    if adx >= 20:
        return 0
    if tracker is not None:
        tracker["F1_adx"] += 1

    # F2: Bollinger squeeze — BW < P30
    bw = ind["bb_width"][t]
    bw_p30 = ind["bw_p30"][t]
    if np.isnan(bw) or np.isnan(bw_p30) or bw >= bw_p30:
        return 0
    if tracker is not None:
        tracker["F2_bw_squeeze"] += 1

    # F3: Flat MA — slope < 1%
    slope = ind["sma20_slope10"][t]
    if np.isnan(slope) or abs(slope) >= 0.01:
        return 0
    if tracker is not None:
        tracker["F3_flat_ma"] += 1

    # F4: Vol compression — ATR ratio < 0.9
    vr = ind["vol_regime"][t]
    if np.isnan(vr) or vr >= 0.9:
        return 0
    if tracker is not None:
        tracker["F4_vol_compress"] += 1

    # F5: No recent breakout in last 10 bars
    if not ind["no_recent_breakout"][t]:
        return 0
    if tracker is not None:
        tracker["F5_no_breakout"] += 1

    return sig

def s8_exit(t, c, ind, pos):
    rsi = ind["rsi14"][t]
    if np.isnan(rsi):
        return False
    if pos == 1.0 and rsi >= 50:
        return True
    if pos == -1.0 and rsi <= 50:
        return True
    return False


# ════════════════════════════════════════════════════════════
# Strategy registry
# ════════════════════════════════════════════════════════════

STRATEGIES = {
    "S1_MA_Reversion": {
        "cat": "Contrarian", "entry": s1_entry, "exit": s1_exit,
        "max_hold_d": 20, "max_hold_h": 40,
    },
    "S2_Bollinger": {
        "cat": "Contrarian", "entry": s2_entry, "exit": s2_exit,
        "max_hold_d": 15, "max_hold_h": 30,
    },
    "S3_RSI_Reversion": {
        "cat": "Contrarian", "entry": s3_entry, "exit": s3_exit,
        "max_hold_d": 10, "max_hold_h": 20,
    },
    "S4_Donchian": {
        "cat": "Trend", "entry": s4_entry, "exit": s4_exit,
        "max_hold_d": None, "max_hold_h": None,
    },
    "S5_Supertrend": {
        "cat": "Trend", "entry": s5_entry, "exit": s5_exit,
        "max_hold_d": None, "max_hold_h": None,
    },
    "S6_DualMA": {
        "cat": "Trend", "entry": s6_entry, "exit": s6_exit,
        "max_hold_d": None, "max_hold_h": None,
    },
    "S7_Keltner": {
        "cat": "Range", "entry": s7_entry, "exit": s7_exit,
        "max_hold_d": 10, "max_hold_h": 20,
    },
    "S8_RSI_Stoch": {
        "cat": "Range", "entry": s8_entry, "exit": s8_exit,
        "max_hold_d": 8, "max_hold_h": 16,
    },
}

# Filter cascade names per strategy (for stats output)
FILTER_CASCADE = {
    "S1_MA_Reversion": ["raw", "F1_vol_regime", "F2_consec_candles", "F3_vol_exhaust", "F4_rsi"],
    "S2_Bollinger": ["raw", "F1_vol_regime", "F2_consec_candles", "F3_vol_exhaust", "F4_bw_p25", "F5_rsi_divergence"],
    "S3_RSI_Reversion": ["raw", "F1_vol_regime", "F2_adx", "F3_consec_candles", "F4_vol_exhaust", "F5_slope"],
    "S4_Donchian": ["raw", "F_multi_tf"],
    "S5_Supertrend": ["raw", "F_multi_tf"],
    "S6_DualMA": ["raw", "F_multi_tf"],
    "S7_Keltner": ["raw", "F1_adx", "F2_bw_squeeze", "F3_flat_ma", "F4_vol_compress", "F5_hurst"],
    "S8_RSI_Stoch": ["raw", "F1_adx", "F2_bw_squeeze", "F3_flat_ma", "F4_vol_compress", "F5_no_breakout"],
}


# ════════════════════════════════════════════════════════════
# Backtest engine (from v2 — category-specific RM)
# ════════════════════════════════════════════════════════════

def backtest_one(sinfo, close, high, low, open_arr, volume, ind, is_hourly):
    n = len(close)
    cat = sinfo["cat"]
    entry_fn = sinfo["entry"]
    exit_fn = sinfo["exit"]
    max_hold = sinfo["max_hold_h"] if is_hourly else sinfo["max_hold_d"]

    atr14 = ind["atr14"]
    adx14 = ind["adx14"]
    vol_sma20 = ind["vol_sma20"]
    trail_n = 20 if is_hourly else 10

    positions = np.zeros(n, dtype=np.float64)
    cur_pos = 0.0
    cur_sl = np.nan
    cur_tp = np.nan
    entry_price = np.nan
    entry_atr = np.nan
    held = 0
    breakeven_activated = False

    exit_counts = {"sl": 0, "tp": 0, "signal": 0, "trailing": 0,
                   "max_hold": 0, "adx_breakout": 0}

    # Filter tracker
    sname = [k for k, v in STRATEGIES.items() if v is sinfo][0]
    cascade = FILTER_CASCADE[sname]
    tracker = {k: 0 for k in cascade}
    tracker["volume_filter"] = 0
    tracker["final"] = 0

    ind["_volume"] = volume

    for t in range(WARMUP, n):
        if cur_pos != 0:
            held += 1
            closed = False
            exit_type = ""

            if cat == "Trend":
                cur_atr = atr14[t] if not np.isnan(atr14[t]) else entry_atr
                lb_start = max(0, t - trail_n)
                if cur_pos == 1.0:
                    recent_high = np.max(high[lb_start:t + 1])
                    trail_sl = recent_high - 2.5 * cur_atr
                    if not breakeven_activated and (close[t] - entry_price) > 1.5 * entry_atr:
                        breakeven_activated = True
                    if breakeven_activated:
                        trail_sl = max(trail_sl, entry_price)
                    cur_sl = max(cur_sl, trail_sl)
                    if low[t] <= cur_sl:
                        closed = True
                        exit_type = "trailing"
                else:
                    recent_low = np.min(low[lb_start:t + 1])
                    trail_sl = recent_low + 2.5 * cur_atr
                    if not breakeven_activated and (entry_price - close[t]) > 1.5 * entry_atr:
                        breakeven_activated = True
                    if breakeven_activated:
                        trail_sl = min(trail_sl, entry_price)
                    cur_sl = min(cur_sl, trail_sl)
                    if high[t] >= cur_sl:
                        closed = True
                        exit_type = "trailing"
                if not closed and exit_fn(t, close, ind, cur_pos):
                    closed = True
                    exit_type = "signal"

            elif cat == "Contrarian":
                if cur_pos == 1.0:
                    sl_hit = low[t] <= cur_sl
                    tp_hit = high[t] >= cur_tp
                else:
                    sl_hit = high[t] >= cur_sl
                    tp_hit = low[t] <= cur_tp
                if sl_hit:
                    closed = True
                    exit_type = "sl"
                elif tp_hit:
                    closed = True
                    exit_type = "tp"
                if not closed and exit_fn(t, close, ind, cur_pos):
                    closed = True
                    exit_type = "signal"
                if not closed and max_hold is not None and held >= max_hold:
                    closed = True
                    exit_type = "max_hold"

            elif cat == "Range":
                if cur_pos == 1.0:
                    sl_hit = low[t] <= cur_sl
                    tp_hit = high[t] >= cur_tp
                else:
                    sl_hit = high[t] >= cur_sl
                    tp_hit = low[t] <= cur_tp
                if sl_hit:
                    closed = True
                    exit_type = "sl"
                elif tp_hit:
                    closed = True
                    exit_type = "tp"
                if not closed:
                    adx_val = adx14[t]
                    if not np.isnan(adx_val) and adx_val > 30:
                        closed = True
                        exit_type = "adx_breakout"
                if not closed and exit_fn(t, close, ind, cur_pos):
                    closed = True
                    exit_type = "signal"
                if not closed and max_hold is not None and held >= max_hold:
                    closed = True
                    exit_type = "max_hold"

            if closed:
                if exit_type in exit_counts:
                    exit_counts[exit_type] += 1
                cur_pos = 0.0
                cur_sl = np.nan
                cur_tp = np.nan
                entry_price = np.nan
                entry_atr = np.nan
                held = 0
                breakeven_activated = False

        if cur_pos == 0:
            sig = entry_fn(t, close, ind, tracker)
            if sig != 0:
                # Volume filter
                vs = vol_sma20[t]
                if not np.isnan(vs) and vs > 0 and volume[t] < 0.5 * vs:
                    sig = 0
                else:
                    tracker["volume_filter"] += 1

                if sig != 0:
                    tracker["final"] += 1
                    cur_pos = float(sig)
                    atr_val = atr14[t] if not np.isnan(atr14[t]) else 0.0
                    entry_price = close[t]
                    entry_atr = atr_val
                    breakeven_activated = False
                    held = 0

                    if cat == "Trend":
                        if sig == 1:
                            cur_sl = close[t] - 2.5 * atr_val
                        else:
                            cur_sl = close[t] + 2.5 * atr_val
                        cur_tp = np.nan
                    elif cat == "Contrarian":
                        if sig == 1:
                            cur_sl = close[t] - 1.5 * atr_val
                            cur_tp = close[t] + 2.0 * atr_val
                        else:
                            cur_sl = close[t] + 1.5 * atr_val
                            cur_tp = close[t] - 2.0 * atr_val
                    elif cat == "Range":
                        if sig == 1:
                            cur_sl = close[t] - 1.5 * atr_val
                            cur_tp = close[t] + 1.5 * atr_val
                        else:
                            cur_sl = close[t] + 1.5 * atr_val
                            cur_tp = close[t] - 1.5 * atr_val

        positions[t] = cur_pos

    return positions, exit_counts, tracker


# ════════════════════════════════════════════════════════════
# Run all backtests
# ════════════════════════════════════════════════════════════

def load_data():
    daily = pd.read_parquet(DATA_DIR / "ohlcv_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.read_parquet(DATA_DIR / "ohlcv_hourly.parquet")
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    print(f"Loaded daily: {len(daily):,} rows, hourly: {len(hourly):,} rows")
    return daily, hourly


def run_all(daily, hourly):
    print("\n" + "=" * 70)
    print("Running v3: 8 strategies × 2 TF × 17 tickers = 272 backtests")
    print("Aggressive filters for contrarian/range, trend unchanged from v2")
    print("=" * 70)

    # Precompute daily lookups for multi-TF alignment
    print("\n  Precomputing daily lookups for multi-TF...")
    daily_lookups = build_daily_lookups(daily)

    records = []
    exit_records = []
    filter_records = []

    for tf_name, df, is_hourly, dt_col in [
        ("daily", daily, False, "date"),
        ("hourly", hourly, True, "datetime"),
    ]:
        print(f"\n  [{tf_name.upper()}]")

        for sname, sinfo in STRATEGIES.items():
            print(f"    {sname} ({sinfo['cat']}):", end="", flush=True)

            # Aggregate filter tracker across tickers
            cascade = FILTER_CASCADE[sname]
            agg_tracker = {k: 0 for k in cascade}
            agg_tracker["volume_filter"] = 0
            agg_tracker["final"] = 0

            for ticker in TICKERS:
                tdf = df[df["ticker"] == ticker].sort_values(dt_col).reset_index(drop=True)
                close = tdf["close"].values
                high_a = tdf["high"].values
                low_a = tdf["low"].values
                open_a = tdf["open"].values
                vol_a = tdf["volume"].values.astype(np.float64)
                dts = tdf[dt_col].values

                ind = precompute(close, high_a, low_a, open_a, vol_a, is_hourly)

                # Inject daily alignment for hourly
                if is_hourly and ticker in daily_lookups:
                    dl = daily_lookups[ticker]
                    ma50_above, ma_slope = align_daily_to_hourly(dl, dts)
                    ind["_daily_ma50_above_ma200"] = ma50_above
                    ind["_daily_ma_slope"] = ma_slope

                positions, exit_counts, tracker = backtest_one(
                    sinfo, close, high_a, low_a, open_a, vol_a, ind, is_hourly
                )

                # Accumulate tracker
                for k in agg_tracker:
                    agg_tracker[k] += tracker.get(k, 0)

                exit_records.append({
                    "strategy": sname, "timeframe": tf_name,
                    "category": sinfo["cat"], "ticker": ticker,
                    **exit_counts,
                })

                log_ret = np.zeros(len(close))
                log_ret[:-1] = np.log(close[1:] / close[:-1])
                daily_return = positions * log_ret

                mask = dts >= np.datetime64(START_DATE)
                dts_s = dts[mask]
                pos_s = positions[mask]
                dr_s = daily_return[mask]

                for i in range(len(dts_s)):
                    records.append({
                        "datetime": dts_s[i],
                        "ticker": ticker,
                        "strategy": sname,
                        "timeframe": tf_name,
                        "position": pos_s[i],
                        "daily_return": dr_s[i],
                    })

                print(f" {ticker}", end="", flush=True)
            print()

            # Store filter stats
            for step in cascade + ["volume_filter", "final"]:
                filter_records.append({
                    "strategy": sname, "timeframe": tf_name,
                    "filter": step, "pass_count": agg_tracker.get(step, 0),
                })

    signals = pd.DataFrame(records)
    signals["datetime"] = pd.to_datetime(signals["datetime"])
    exit_df = pd.DataFrame(exit_records)
    filter_df = pd.DataFrame(filter_records)
    print(f"\n  Total records: {len(signals):,}")
    return signals, exit_df, filter_df


# ════════════════════════════════════════════════════════════
# Metrics (from v2)
# ════════════════════════════════════════════════════════════

def compute_metrics(signals, exit_df):
    print("\nComputing metrics...")
    rows = []

    for tf_name in ["daily", "hourly"]:
        ann_factor = np.sqrt(252 * 9) if tf_name == "hourly" else np.sqrt(252)
        bars_per_year = 252 * 9 if tf_name == "hourly" else 252

        for sname in STRATEGIES:
            for ticker in TICKERS:
                m = ((signals["strategy"] == sname) &
                     (signals["ticker"] == ticker) &
                     (signals["timeframe"] == tf_name))
                sdf = signals[m].sort_values("datetime")
                dr = sdf["daily_return"].values
                pos = sdf["position"].values
                n = len(dr)
                if n == 0:
                    continue

                active = pos != 0
                n_active = active.sum()
                exposure = n_active / n

                mean_r = np.mean(dr)
                std_r = np.std(dr, ddof=1) if n > 1 else 1e-10
                sharpe = mean_r / std_r * ann_factor if std_r > 1e-12 else 0.0
                ann_ret = mean_r * bars_per_year
                ann_vol = std_r * ann_factor

                cum = np.cumsum(dr)
                rmax = np.maximum.accumulate(cum)
                dd = cum - rmax
                max_dd = dd.min()
                calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

                if n_active > 0:
                    ar = dr[active]
                    win_rate = (ar > 0).sum() / n_active
                    gp = ar[ar > 0].sum()
                    gl = abs(ar[ar < 0].sum())
                    pf = gp / gl if gl > 1e-12 else (99.0 if gp > 0 else 0.0)
                    n_win = (ar > 0).sum()
                    n_loss = (ar < 0).sum()
                    avg_win = ar[ar > 0].mean() if n_win > 0 else 0.0
                    avg_loss = abs(ar[ar < 0].mean()) if n_loss > 0 else 1e-12
                    payoff = avg_win / avg_loss if avg_loss > 1e-12 else (99.0 if avg_win > 0 else 0.0)
                else:
                    win_rate = 0.0
                    pf = 0.0
                    payoff = 0.0

                changes = np.sum(np.abs(np.diff(pos)) > 0)
                n_years = n / bars_per_year
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

                ex = exit_df[
                    (exit_df["strategy"] == sname) &
                    (exit_df["timeframe"] == tf_name) &
                    (exit_df["ticker"] == ticker)
                ]
                exit_pcts = {}
                if len(ex) > 0:
                    ex_row = ex.iloc[0]
                    ex_cols = ["sl", "tp", "signal", "trailing", "max_hold", "adx_breakout"]
                    ex_total = max(sum(ex_row[c] for c in ex_cols), 1)
                    for c in ex_cols:
                        exit_pcts[f"{c}_exit_pct"] = round(ex_row[c] / ex_total * 100, 1)
                else:
                    for c in ["sl", "tp", "signal", "trailing", "max_hold", "adx_breakout"]:
                        exit_pcts[f"{c}_exit_pct"] = 0.0

                rows.append({
                    "strategy": sname, "timeframe": tf_name,
                    "category": STRATEGIES[sname]["cat"], "ticker": ticker,
                    "sharpe": sharpe, "annual_return": ann_ret,
                    "annual_vol": ann_vol, "max_drawdown": max_dd,
                    "calmar": calmar, "win_rate": win_rate,
                    "profit_factor": pf, "payoff_ratio": payoff,
                    "exposure": exposure,
                    "turnover": turnover, "avg_trade_bars": avg_trade,
                    "n_trades": trades,
                    **exit_pcts,
                })

    return pd.DataFrame(rows)


def aggregate(metrics):
    cols = ["sharpe", "annual_return", "annual_vol", "max_drawdown",
            "calmar", "win_rate", "profit_factor", "payoff_ratio",
            "exposure", "turnover", "avg_trade_bars"]
    rows = []
    for (sname, tf), grp in metrics.groupby(["strategy", "timeframe"]):
        r = {"strategy": sname, "timeframe": tf,
             "category": grp["category"].iloc[0]}
        for c in cols:
            r[f"{c}_median"] = grp[c].median()
            r[f"{c}_mean"] = grp[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def select_and_rank(summary):
    summary["selected"] = (
        (summary["sharpe_median"] > 0.0) &
        (summary["exposure_median"] > 0.05) &
        (summary["max_drawdown_median"] > -0.55)
    )
    summary = summary.sort_values("sharpe_median", ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    return summary


# ════════════════════════════════════════════════════════════
# Output functions
# ════════════════════════════════════════════════════════════

def make_exit_analysis(exit_df):
    exit_cols = ["sl", "tp", "signal", "trailing", "max_hold", "adx_breakout"]
    agg = exit_df.groupby(["strategy", "timeframe", "category"])[exit_cols].sum().reset_index()
    total = agg[exit_cols].sum(axis=1).replace(0, 1)
    for c in exit_cols:
        agg[f"{c}_pct"] = (agg[c] / total * 100).round(1)
    return agg


def make_positive_count(metrics):
    rows = []
    for (sname, tf), grp in metrics.groupby(["strategy", "timeframe"]):
        n_pos = (grp["sharpe"] > 0).sum()
        rows.append({
            "strategy": sname, "timeframe": tf,
            "category": grp["category"].iloc[0],
            "positive_tickers": n_pos,
            "total_tickers": len(grp),
            "positive_pct": round(n_pos / len(grp) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values(
        ["timeframe", "positive_tickers"], ascending=[True, False]
    )


def load_v2_summary():
    p = OUT_TABLES / "screener_v2_daily.csv"
    p2 = OUT_TABLES / "screener_v2_hourly.csv"
    if p.exists() and p2.exists():
        return pd.concat([pd.read_csv(p), pd.read_csv(p2)], ignore_index=True)
    return None


def make_comparison(v3_summary, v2_summary):
    if v2_summary is None:
        return None
    rows = []
    for _, r3 in v3_summary.iterrows():
        sname, tf = r3["strategy"], r3["timeframe"]
        v2_match = v2_summary[
            (v2_summary["strategy"] == sname) & (v2_summary["timeframe"] == tf)
        ]
        sharpe_v2 = v2_match["sharpe_median"].values[0] if len(v2_match) > 0 else np.nan
        sharpe_v3 = r3["sharpe_median"]
        rows.append({
            "strategy": sname, "timeframe": tf, "category": r3["category"],
            "sharpe_v2": round(sharpe_v2, 4) if not np.isnan(sharpe_v2) else np.nan,
            "sharpe_v3": round(sharpe_v3, 4),
            "delta": round(sharpe_v3 - sharpe_v2, 4) if not np.isnan(sharpe_v2) else np.nan,
            "maxdd_v2": round(v2_match["max_drawdown_median"].values[0] * 100, 2)
                if len(v2_match) > 0 else np.nan,
            "maxdd_v3": round(r3["max_drawdown_median"] * 100, 2),
            "exp_v3": round(r3["exposure_median"] * 100, 2),
        })
    comp = pd.DataFrame(rows)
    comp = comp.sort_values("delta", ascending=False)
    return comp


def save_all(signals, metrics, summary, exit_df, filter_df, v2_summary):
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    print("\nSaving files:")

    p = OUT_DATA / "signals_screener_v3.parquet"
    signals.to_parquet(p, index=False)
    print(f"  1. {p.name}: {len(signals):,} rows")

    daily_s = summary[summary["timeframe"] == "daily"].copy()
    p = OUT_TABLES / "screener_v3_daily.csv"
    daily_s.to_csv(p, index=False)
    print(f"  2. {p.name}: {len(daily_s)} rows")

    hourly_s = summary[summary["timeframe"] == "hourly"].copy()
    p = OUT_TABLES / "screener_v3_hourly.csv"
    hourly_s.to_csv(p, index=False)
    print(f"  3. {p.name}: {len(hourly_s)} rows")

    comp = make_comparison(summary, v2_summary)
    if comp is not None:
        p = OUT_TABLES / "screener_v3_vs_v2.csv"
        comp.to_csv(p, index=False)
        print(f"  4. {p.name}: {len(comp)} rows")

    pos_count = make_positive_count(metrics)
    p = OUT_TABLES / "screener_v3_positive_count.csv"
    pos_count.to_csv(p, index=False)
    print(f"  5. {p.name}: {len(pos_count)} rows")

    exit_analysis = make_exit_analysis(exit_df)
    p = OUT_TABLES / "screener_v3_exit_analysis.csv"
    exit_analysis.to_csv(p, index=False)
    print(f"  6. {p.name}: {len(exit_analysis)} rows")

    p = OUT_TABLES / "screener_v3_filter_stats.csv"
    filter_df.to_csv(p, index=False)
    print(f"  7. {p.name}: {len(filter_df)} rows")

    sel = summary[summary["selected"]].copy()
    p = OUT_TABLES / "screener_v3_selected.csv"
    sel.to_csv(p, index=False)
    print(f"  8. {p.name}: {len(sel)} rows")


def print_results(summary, exit_df, filter_df, v2_summary):
    # Table A: DAILY
    daily_s = summary[summary["timeframe"] == "daily"].copy()
    print("\n" + "=" * 116)
    print("TABLE A: DAILY — 8 strategies (v3, aggressive filters)")
    print("=" * 116)
    _print_table(daily_s)

    # Table B: HOURLY
    hourly_s = summary[summary["timeframe"] == "hourly"].copy()
    print("\n" + "=" * 116)
    print("TABLE B: HOURLY — 8 strategies (v3, aggressive filters)")
    print("=" * 116)
    _print_table(hourly_s)

    # Table C: v2 vs v3
    comp = make_comparison(summary, v2_summary)
    if comp is not None:
        print("\n" + "=" * 100)
        print("TABLE C: v2 vs v3 COMPARISON (sorted by improvement)")
        print("=" * 100)
        print(f"{'Strategy':<18} {'TF':<7} {'Cat':<11} {'Sh_v2':>8} "
              f"{'Sh_v3':>8} {'Delta':>8} {'MaxDD_v2':>9} {'MaxDD_v3':>9} {'Exp_v3':>7}")
        print("-" * 100)
        for _, r in comp.iterrows():
            d = r["delta"]
            d_str = f"{d:>+7.4f}" if not np.isnan(d) else "    N/A"
            s2 = f"{r['sharpe_v2']:>7.4f}" if not np.isnan(r['sharpe_v2']) else "    N/A"
            print(f"{r['strategy']:<18} {r['timeframe']:<7} {r['category']:<11} "
                  f"{s2} {r['sharpe_v3']:>7.4f} {d_str} "
                  f"{r['maxdd_v2']:>+8.2f}% {r['maxdd_v3']:>+8.2f}% "
                  f"{r['exp_v3']:>6.2f}%")
        avg_delta = comp["delta"].dropna().mean()
        n_improved = (comp["delta"] > 0).sum()
        print(f"\n  Улучшение vs v2: средний Δ Sharpe = {avg_delta:+.4f}")
        print(f"  Improved: {n_improved} / {len(comp)} strategy×TF")

    # Filter stats
    print("\n" + "=" * 90)
    print("FILTER STATS (which filter kills most signals)")
    print("=" * 90)
    for tf in ["daily", "hourly"]:
        print(f"\n  [{tf.upper()}]")
        for sname in STRATEGIES:
            cat = STRATEGIES[sname]["cat"]
            if cat == "Trend":
                continue  # trend filters are minimal
            fdf = filter_df[(filter_df["strategy"] == sname) & (filter_df["timeframe"] == tf)]
            if len(fdf) == 0:
                continue
            cascade = FILTER_CASCADE[sname] + ["volume_filter", "final"]
            counts = []
            for step in cascade:
                row = fdf[fdf["filter"] == step]
                cnt = row["pass_count"].values[0] if len(row) > 0 else 0
                counts.append((step, cnt))
            line = f"    {sname:<18}"
            for step, cnt in counts:
                line += f"  {step}={cnt}"
            print(line)
            # Show biggest killer
            if len(counts) > 1:
                kills = []
                for i in range(1, len(counts)):
                    before = counts[i - 1][1]
                    after = counts[i][1]
                    if before > 0:
                        kill_pct = (1 - after / before) * 100
                        kills.append((counts[i][0], kill_pct))
                if kills:
                    worst = max(kills, key=lambda x: x[1])
                    print(f"    {'':18}  → biggest killer: {worst[0]} (-{worst[1]:.0f}%)")

    # Exposure warnings
    print("\n" + "=" * 60)
    print("EXPOSURE WARNINGS")
    print("=" * 60)
    contrarian_range = summary[summary["category"].isin(["Contrarian", "Range"])]
    for _, r in contrarian_range.iterrows():
        exp = r["exposure_median"] * 100
        if exp < 3.0:
            print(f"  ⚠ {r['strategy']:<18} [{r['timeframe']}] Exposure={exp:.1f}% "
                  "— ФИЛЬТРЫ ПЕРЕГРУЖЕНЫ, стратегия почти не торгует")
        elif exp < 5.0:
            print(f"  ! {r['strategy']:<18} [{r['timeframe']}] Exposure={exp:.1f}% — low")

    # Selected
    n_sel = summary["selected"].sum()
    print(f"\n{'=' * 60}")
    print(f"SELECTED (Sharpe>0, Exp>5%, MaxDD>-55%): {n_sel} / {len(summary)}")
    print(f"{'=' * 60}")
    sel = summary[summary["selected"]]
    if len(sel) > 0:
        for _, r in sel.iterrows():
            print(f"  {r['rank']:>2}. {r['strategy']:<18} [{r['timeframe']}] "
                  f"Sharpe={r['sharpe_median']:.3f}  "
                  f"Return={r['annual_return_median'] * 100:+.1f}%  "
                  f"MaxDD={r['max_drawdown_median'] * 100:+.1f}%")


def _print_table(sub):
    hdr = (f"{'#':>2} {'Strategy':<18} {'Cat':<11} "
           f"{'Sharpe':>7} {'AnnR%':>7} {'MaxDD%':>7} {'Calmar':>7} "
           f"{'WinR%':>6} {'PF':>6} {'PayR':>5} {'Exp%':>5} {'Turn':>5} {'AvgT':>5} {'SEL':>4}")
    print(hdr)
    print("-" * 116)
    printed_sep = False
    for _, r in sub.iterrows():
        if not printed_sep and not r["selected"]:
            print("-" * 40 + " BELOW CUTOFF " + "-" * 62)
            printed_sep = True
        sel_mark = " *" if r["selected"] else ""
        pf = r["profit_factor_median"]
        pf_s = f"{pf:>6.2f}" if pf < 90 else f"{'inf':>6}"
        pr = r["payoff_ratio_median"]
        pr_s = f"{pr:>5.2f}" if pr < 90 else f"{'inf':>5}"
        print(f"{r['rank']:>2} {r['strategy']:<18} {r['category']:<11} "
              f"{r['sharpe_median']:>7.3f} "
              f"{r['annual_return_median'] * 100:>+6.1f}% "
              f"{r['max_drawdown_median'] * 100:>+6.1f}% "
              f"{r['calmar_median']:>7.3f} "
              f"{r['win_rate_median'] * 100:>5.1f}% "
              f"{pf_s} "
              f"{pr_s} "
              f"{r['exposure_median'] * 100:>4.1f}% "
              f"{r['turnover_median']:>5.0f} "
              f"{r['avg_trade_bars_median']:>5.1f}"
              f"{sel_mark}")
    if not printed_sep:
        print("-" * 40 + " BELOW CUTOFF " + "-" * 62)
    print("-" * 116)


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Strategy Screener v3: Aggressive filters")
    print("  Contrarian: exhaustion (consec candles, vol exhaust, divergence)")
    print("  Range: strict (squeeze, flat MA, Hurst, no breakout)")
    print("  Trend: unchanged from v2 + multi-TF for S4h/S5h")
    print("=" * 70)

    daily, hourly = load_data()
    signals, exit_df, filter_df = run_all(daily, hourly)
    metrics = compute_metrics(signals, exit_df)
    summary = aggregate(metrics)
    summary = select_and_rank(summary)

    v2_summary = load_v2_summary()
    if v2_summary is not None:
        print("  v2 summary loaded for comparison")

    save_all(signals, metrics, summary, exit_df, filter_df, v2_summary)
    print_results(summary, exit_df, filter_df, v2_summary)
    print("\nDONE")


if __name__ == "__main__":
    main()
