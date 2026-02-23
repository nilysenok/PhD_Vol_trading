#!/usr/bin/env python3
"""
strategies_screener_v2.py — 8 strategies × 2 timeframes screener
with CATEGORY-SPECIFIC risk management.

Key changes vs v1:
  Trend  (S4-S6): Trailing stop + break-even, NO fixed TP
  Contrarian (S1-S3): Tight SL=1.5×ATR, TP=2.0×ATR
  Range  (S7-S8): Symmetric SL=TP=1.5×ATR, ADX>30 breakout exit

Improved filters: vol-regime, relaxed RSI for S1, BW P25 for S2,
                  ADX<30 for S3, BW<P75 for S7.
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
# Technical indicators
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


# ════════════════════════════════════════════════════════════
# Precompute indicators per ticker
# ════════════════════════════════════════════════════════════

def precompute(close, high, low, volume, is_hourly=False):
    """Compute all indicators needed by any strategy."""
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
    ind["bw_median"] = pd.Series(ind["bb_width"]).rolling(bw_window, min_periods=50).median().values
    # P25 for S2 contrarian filter (was median — too strict)
    ind["bw_p25"] = pd.Series(ind["bb_width"]).rolling(bw_window, min_periods=50).quantile(0.25).values
    # P75 for S7 range filter
    ind["bw_p75"] = pd.Series(ind["bb_width"]).rolling(bw_window, min_periods=50).quantile(0.75).values

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

    return ind


# ════════════════════════════════════════════════════════════
# Strategy entry/exit functions (v2 with improved filters)
# ════════════════════════════════════════════════════════════

# --- CONTRARIAN (S1-S3): vol-regime < 1.2, relaxed thresholds ---

def s1_entry(t, c, ind):
    """MA Mean Reversion — RELAXED RSI 45/55 (was 40/60), vol-regime < 1.2, no MA200."""
    z, rsi = ind["z"][t], ind["rsi14"][t]
    vr = ind["vol_regime"][t]
    if np.isnan(z) or np.isnan(rsi) or np.isnan(vr):
        return 0
    if vr >= 1.2:
        return 0
    if z < -2.0 and rsi < 45:
        return 1
    if z > 2.0 and rsi > 55:
        return -1
    return 0

def s1_exit(t, c, ind, pos):
    z = ind["z"][t]
    if np.isnan(z):
        return False
    return abs(z) < 0.5

def s2_entry(t, c, ind):
    """Bollinger Bands — BW > P25 (was > median), vol-regime < 1.2, no MA200."""
    cl = c[t]
    bw = ind["bb_width"][t]
    bw_p25 = ind["bw_p25"][t]
    lo, up = ind["bb_lower"][t], ind["bb_upper"][t]
    vr = ind["vol_regime"][t]
    if np.isnan(bw_p25) or np.isnan(vr):
        return 0
    if vr >= 1.2:
        return 0
    if cl <= lo and bw > bw_p25:
        return 1
    if cl >= up and bw > bw_p25:
        return -1
    return 0

def s2_exit(t, c, ind, pos):
    ma = ind["sma20"][t]
    if np.isnan(ma):
        return False
    if pos == 1.0 and c[t] >= ma:
        return True
    if pos == -1.0 and c[t] <= ma:
        return True
    return False

def s3_entry(t, c, ind):
    """RSI Mean Reversion — added ADX<30 + vol-regime < 1.2."""
    rsi = ind["rsi14"][t]
    adx = ind["adx14"][t]
    vr = ind["vol_regime"][t]
    if np.isnan(rsi) or np.isnan(adx) or np.isnan(vr):
        return 0
    if vr >= 1.2:
        return 0
    if adx >= 30:
        return 0
    if rsi < 25:
        return 1
    if rsi > 75:
        return -1
    return 0

def s3_exit(t, c, ind, pos):
    rsi = ind["rsi14"][t]
    if np.isnan(rsi):
        return False
    if pos == 1.0 and rsi >= 50:
        return True
    if pos == -1.0 and rsi <= 50:
        return True
    return False

# --- TREND (S4-S6): unchanged filters ---

def s4_entry(t, c, ind):
    if t < 1:
        return 0
    cl, adx = c[t], ind["adx14"][t]
    hc, lc = ind["high_ch20"][t-1], ind["low_ch20"][t-1]
    vs = ind["vol_sma20"][t]
    if np.isnan(adx) or np.isnan(hc):
        return 0
    if not np.isnan(vs) and vs > 0:
        v = ind["_volume"][t] / vs
    else:
        v = 1.0
    if cl > hc and adx > 20 and v > 1.0:
        return 1
    if cl < lc and adx > 20 and v > 1.0:
        return -1
    return 0

def s4_exit(t, c, ind, pos):
    if t < 1:
        return False
    cl = c[t]
    hc, lc = ind["high_ch20"][t-1], ind["low_ch20"][t-1]
    if np.isnan(hc):
        return False
    if pos == 1.0 and cl < lc:
        return True
    if pos == -1.0 and cl > hc:
        return True
    return False

def s5_entry(t, c, ind):
    if t < 1:
        return 0
    cl, adx = c[t], ind["adx14"][t]
    st, st_p = ind["supertrend"][t], ind["supertrend"][t-1]
    cp = c[t-1]
    if np.isnan(st) or np.isnan(st_p) or np.isnan(adx):
        return 0
    if cl > st and cp <= st_p and adx > 20:
        return 1
    if cl < st and cp >= st_p and adx > 20:
        return -1
    return 0

def s5_exit(t, c, ind, pos):
    if t < 1:
        return False
    cl, st = c[t], ind["supertrend"][t]
    cp, st_p = c[t-1], ind["supertrend"][t-1]
    if np.isnan(st) or np.isnan(st_p):
        return False
    if pos == 1.0 and cl < st and cp >= st_p:
        return True
    if pos == -1.0 and cl > st and cp <= st_p:
        return True
    return False

def s6_entry(t, c, ind):
    if t < 1:
        return 0
    f, s = ind["ma_fast"][t], ind["ma_slow"][t]
    fp, sp = ind["ma_fast"][t-1], ind["ma_slow"][t-1]
    adx = ind["adx14"][t]
    if np.isnan(f) or np.isnan(s) or np.isnan(fp) or np.isnan(sp) or np.isnan(adx):
        return 0
    if f > s and fp <= sp and adx > 15:
        return 1
    if f < s and fp >= sp and adx > 15:
        return -1
    return 0

def s6_exit(t, c, ind, pos):
    if t < 1:
        return False
    f, s = ind["ma_fast"][t], ind["ma_slow"][t]
    fp, sp = ind["ma_fast"][t-1], ind["ma_slow"][t-1]
    if np.isnan(f) or np.isnan(s) or np.isnan(fp) or np.isnan(sp):
        return False
    if pos == 1.0 and f < s and fp >= sp:
        return True
    if pos == -1.0 and f > s and fp <= sp:
        return True
    return False

# --- RANGE (S7-S8): vol-regime < 1.0, BW<P75 for S7 ---

def s7_entry(t, c, ind):
    """Keltner Channel — ADX<25, BW<P75 (bands compressed), vol-regime < 1.0."""
    cl, adx = c[t], ind["adx14"][t]
    lo, up = ind["kc_lower"][t], ind["kc_upper"][t]
    bw = ind["bb_width"][t]
    bw_p75 = ind["bw_p75"][t]
    vr = ind["vol_regime"][t]
    if np.isnan(adx) or np.isnan(lo) or np.isnan(bw_p75) or np.isnan(vr):
        return 0
    if adx >= 25:
        return 0
    if bw >= bw_p75:
        return 0
    if vr >= 1.0:
        return 0
    if cl < lo:
        return 1
    if cl > up:
        return -1
    return 0

def s7_exit(t, c, ind, pos):
    ema = ind["ema20"][t]
    if np.isnan(ema):
        return False
    if pos == 1.0 and c[t] >= ema:
        return True
    if pos == -1.0 and c[t] <= ema:
        return True
    return False

def s8_entry(t, c, ind):
    """RSI + Stochastic — ADX<25 (was <30), vol-regime < 1.0."""
    rsi, sk, adx = ind["rsi14"][t], ind["stoch_k"][t], ind["adx14"][t]
    vr = ind["vol_regime"][t]
    if np.isnan(rsi) or np.isnan(sk) or np.isnan(adx) or np.isnan(vr):
        return 0
    if adx >= 25:
        return 0
    if vr >= 1.0:
        return 0
    if rsi < 30 and sk < 20:
        return 1
    if rsi > 70 and sk > 80:
        return -1
    return 0

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


# ════════════════════════════════════════════════════════════
# Backtest engine with category-specific RM
# ════════════════════════════════════════════════════════════

def backtest_one(sinfo, close, high, low, volume, ind, is_hourly):
    """
    Run one strategy on one ticker with category-specific risk management.

    Returns:
        positions: np.array of position values
        exit_counts: dict with counts per exit type
    """
    n = len(close)
    cat = sinfo["cat"]
    entry_fn = sinfo["entry"]
    exit_fn = sinfo["exit"]
    max_hold = sinfo["max_hold_h"] if is_hourly else sinfo["max_hold_d"]

    atr14 = ind["atr14"]
    adx14 = ind["adx14"]
    vol_sma20 = ind["vol_sma20"]

    # Trailing stop lookback for trend strategies
    trail_n = 20 if is_hourly else 10

    positions = np.zeros(n, dtype=np.float64)
    cur_pos = 0.0
    cur_sl = np.nan
    cur_tp = np.nan
    entry_price = np.nan
    entry_atr = np.nan
    held = 0
    breakeven_activated = False

    # Exit tracking
    exit_counts = {"sl": 0, "tp": 0, "signal": 0, "trailing": 0,
                   "max_hold": 0, "adx_breakout": 0}

    ind["_volume"] = volume

    for t in range(WARMUP, n):
        if cur_pos != 0:
            held += 1
            closed = False
            exit_type = ""

            if cat == "Trend":
                # ── TREND RM: trailing stop, break-even, no TP ──
                # Use current ATR for trailing offset (not entry ATR)
                cur_atr = atr14[t] if not np.isnan(atr14[t]) else entry_atr
                lb_start = max(0, t - trail_n)
                if cur_pos == 1.0:
                    recent_high = np.max(high[lb_start:t + 1])
                    trail_sl = recent_high - 2.5 * cur_atr
                    # Break-even: if profit > 1.5×ATR(entry), floor SL at entry
                    if not breakeven_activated and (close[t] - entry_price) > 1.5 * entry_atr:
                        breakeven_activated = True
                    if breakeven_activated:
                        trail_sl = max(trail_sl, entry_price)
                    # SL only tightens, never widens
                    cur_sl = max(cur_sl, trail_sl)
                    if low[t] <= cur_sl:
                        closed = True
                        exit_type = "trailing"
                else:  # short
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

                # Strategy exit (reverse signal)
                if not closed and exit_fn(t, close, ind, cur_pos):
                    closed = True
                    exit_type = "signal"

            elif cat == "Contrarian":
                # ── CONTRARIAN RM: SL=1.5×ATR, TP=2.0×ATR ──
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
                # ── RANGE RM: SL=TP=1.5×ATR, ADX>30 breakout exit ──
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

                # ADX breakout exit
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

        # Check entry if flat
        if cur_pos == 0:
            sig = entry_fn(t, close, ind)
            if sig != 0:
                # Volume filter (applied to all)
                vs = vol_sma20[t]
                if not np.isnan(vs) and vs > 0 and volume[t] < 0.5 * vs:
                    sig = 0

                if sig != 0:
                    cur_pos = float(sig)
                    atr_val = atr14[t] if not np.isnan(atr14[t]) else 0.0
                    entry_price = close[t]
                    entry_atr = atr_val
                    breakeven_activated = False
                    held = 0

                    if cat == "Trend":
                        # Initial SL = 2.5 × ATR, NO TP
                        if sig == 1:
                            cur_sl = close[t] - 2.5 * atr_val
                        else:
                            cur_sl = close[t] + 2.5 * atr_val
                        cur_tp = np.nan  # no TP for trend

                    elif cat == "Contrarian":
                        # SL = 1.5 × ATR, TP = 2.0 × ATR
                        if sig == 1:
                            cur_sl = close[t] - 1.5 * atr_val
                            cur_tp = close[t] + 2.0 * atr_val
                        else:
                            cur_sl = close[t] + 1.5 * atr_val
                            cur_tp = close[t] - 2.0 * atr_val

                    elif cat == "Range":
                        # Symmetric SL = TP = 1.5 × ATR
                        if sig == 1:
                            cur_sl = close[t] - 1.5 * atr_val
                            cur_tp = close[t] + 1.5 * atr_val
                        else:
                            cur_sl = close[t] + 1.5 * atr_val
                            cur_tp = close[t] - 1.5 * atr_val

        positions[t] = cur_pos

    return positions, exit_counts


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
    print(f"Running v2: 8 strategies × 2 timeframes × 17 tickers = 272 backtests")
    print("Category-specific RM: Trend=trailing, Contrarian=tight, Range=symmetric")
    print("=" * 70)

    records = []
    exit_records = []  # per ticker

    for tf_name, df, is_hourly, dt_col in [
        ("daily", daily, False, "date"),
        ("hourly", hourly, True, "datetime"),
    ]:
        print(f"\n  [{tf_name.upper()}]")

        for sname, sinfo in STRATEGIES.items():
            print(f"    {sname} ({sinfo['cat']}):", end="", flush=True)

            for ticker in TICKERS:
                tdf = df[df["ticker"] == ticker].sort_values(dt_col).reset_index(drop=True)
                close = tdf["close"].values
                high_a = tdf["high"].values
                low_a = tdf["low"].values
                vol_a = tdf["volume"].values.astype(np.float64)
                dts = tdf[dt_col].values

                ind = precompute(close, high_a, low_a, vol_a, is_hourly)
                positions, exit_counts = backtest_one(
                    sinfo, close, high_a, low_a, vol_a, ind, is_hourly
                )

                # Store per-ticker exit stats
                exit_records.append({
                    "strategy": sname, "timeframe": tf_name,
                    "category": sinfo["cat"], "ticker": ticker,
                    **exit_counts,
                })

                # Returns: signal on close t → position t+1
                log_ret = np.zeros(len(close))
                log_ret[:-1] = np.log(close[1:] / close[:-1])
                daily_return = positions * log_ret

                # Filter to 2020+
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

    signals = pd.DataFrame(records)
    signals["datetime"] = pd.to_datetime(signals["datetime"])
    exit_df = pd.DataFrame(exit_records)
    print(f"\n  Total records: {len(signals):,}")
    return signals, exit_df


# ════════════════════════════════════════════════════════════
# Metrics
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
                    # Payoff Ratio = avg_win / avg_loss
                    n_win = (ar > 0).sum()
                    n_loss = (ar < 0).sum()
                    avg_win = ar[ar > 0].mean() if n_win > 0 else 0.0
                    avg_loss = abs(ar[ar < 0].mean()) if n_loss > 0 else 1e-12
                    payoff = avg_win / avg_loss if avg_loss > 1e-12 else (99.0 if avg_win > 0 else 0.0)
                else:
                    win_rate = 0.0
                    pf = 0.0
                    payoff = 0.0

                # Turnover & avg trade
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

                # Per-ticker exit % from exit_df
                ex = exit_df[
                    (exit_df["strategy"] == sname) &
                    (exit_df["timeframe"] == tf_name) &
                    (exit_df["ticker"] == ticker)
                ]
                exit_pcts = {}
                if len(ex) > 0:
                    ex_row = ex.iloc[0]
                    ex_cols = ["sl", "tp", "signal", "trailing", "max_hold", "adx_breakout"]
                    ex_total = sum(ex_row[c] for c in ex_cols)
                    ex_total = max(ex_total, 1)
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


# ════════════════════════════════════════════════════════════
# Selection, analysis & output
# ════════════════════════════════════════════════════════════

def select_and_rank(summary):
    """Apply RELAXED selection criteria and rank."""
    summary["selected"] = (
        (summary["sharpe_median"] > 0.0) &
        (summary["exposure_median"] > 0.05) &
        (summary["max_drawdown_median"] > -0.55)
    )
    summary = summary.sort_values("sharpe_median", ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    return summary


def make_exit_analysis(exit_df):
    """Aggregate per-ticker exit counts to strategy×TF and convert to %."""
    exit_cols = ["sl", "tp", "signal", "trailing", "max_hold", "adx_breakout"]
    agg = exit_df.groupby(["strategy", "timeframe", "category"])[exit_cols].sum().reset_index()
    total = agg[exit_cols].sum(axis=1).replace(0, 1)
    for c in exit_cols:
        agg[f"{c}_pct"] = (agg[c] / total * 100).round(1)
    return agg


def make_positive_count(metrics):
    """Count tickers with Sharpe > 0 per strategy × TF."""
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


def make_top10(metrics, col, ascending=False):
    return metrics.nlargest(10, col) if not ascending else metrics.nsmallest(10, col)


def load_v1_summary():
    """Load v1 screener summary for comparison."""
    p = OUT_TABLES / "screener_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def make_comparison(v2_summary, v1_summary):
    """Create v1 vs v2 comparison table."""
    if v1_summary is None:
        return None
    rows = []
    for _, r2 in v2_summary.iterrows():
        sname, tf = r2["strategy"], r2["timeframe"]
        v1_match = v1_summary[
            (v1_summary["strategy"] == sname) & (v1_summary["timeframe"] == tf)
        ]
        sharpe_v1 = v1_match["sharpe_median"].values[0] if len(v1_match) > 0 else np.nan
        sharpe_v2 = r2["sharpe_median"]
        rows.append({
            "strategy": sname, "timeframe": tf, "category": r2["category"],
            "sharpe_v1": round(sharpe_v1, 4) if not np.isnan(sharpe_v1) else np.nan,
            "sharpe_v2": round(sharpe_v2, 4),
            "delta": round(sharpe_v2 - sharpe_v1, 4) if not np.isnan(sharpe_v1) else np.nan,
            "ann_ret_v1": round(v1_match["annual_return_median"].values[0] * 100, 2)
                if len(v1_match) > 0 else np.nan,
            "ann_ret_v2": round(r2["annual_return_median"] * 100, 2),
            "maxdd_v1": round(v1_match["max_drawdown_median"].values[0] * 100, 2)
                if len(v1_match) > 0 else np.nan,
            "maxdd_v2": round(r2["max_drawdown_median"] * 100, 2),
        })
    comp = pd.DataFrame(rows)
    comp = comp.sort_values("delta", ascending=False)
    return comp


def save_all(signals, metrics, summary, exit_df, v1_summary):
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    print("\nSaving files:")

    # 1. Signals parquet
    p = OUT_DATA / "signals_screener_v2.parquet"
    signals.to_parquet(p, index=False)
    print(f"  1. {p.name}: {len(signals):,} rows")

    # 2. Daily summary
    daily_s = summary[summary["timeframe"] == "daily"].copy()
    p = OUT_TABLES / "screener_v2_daily.csv"
    daily_s.to_csv(p, index=False)
    print(f"  2. {p.name}: {len(daily_s)} rows")

    # 3. Hourly summary
    hourly_s = summary[summary["timeframe"] == "hourly"].copy()
    p = OUT_TABLES / "screener_v2_hourly.csv"
    hourly_s.to_csv(p, index=False)
    print(f"  3. {p.name}: {len(hourly_s)} rows")

    # 4. Top-10 by Sharpe
    top_sharpe = make_top10(metrics, "sharpe")
    p = OUT_TABLES / "screener_v2_top10_sharpe.csv"
    top_sharpe.to_csv(p, index=False)
    print(f"  4. {p.name}: {len(top_sharpe)} rows")

    # 5. Top-10 by Calmar
    top_calmar = make_top10(metrics, "calmar")
    p = OUT_TABLES / "screener_v2_top10_calmar.csv"
    top_calmar.to_csv(p, index=False)
    print(f"  5. {p.name}: {len(top_calmar)} rows")

    # 6. Positive ticker count
    pos_count = make_positive_count(metrics)
    p = OUT_TABLES / "screener_v2_positive_count.csv"
    pos_count.to_csv(p, index=False)
    print(f"  6. {p.name}: {len(pos_count)} rows")

    # 7. Exit analysis
    exit_analysis = make_exit_analysis(exit_df)
    p = OUT_TABLES / "screener_v2_exit_analysis.csv"
    exit_analysis.to_csv(p, index=False)
    print(f"  7. {p.name}: {len(exit_analysis)} rows")

    # 8. v1 vs v2 comparison
    comp = make_comparison(summary, v1_summary)
    if comp is not None:
        p = OUT_TABLES / "screener_v2_vs_v1.csv"
        comp.to_csv(p, index=False)
        print(f"  8. {p.name}: {len(comp)} rows")
    else:
        print("  8. screener_v2_vs_v1.csv: SKIPPED (v1 summary not found)")

    # 9. By-ticker metrics
    p = OUT_TABLES / "screener_v2_by_ticker.csv"
    metrics.to_csv(p, index=False)
    print(f"  9. {p.name}: {len(metrics)} rows")


def print_results(summary, exit_df, v1_summary):
    # ── Table A: DAILY ──
    daily_s = summary[summary["timeframe"] == "daily"].copy()
    print("\n" + "=" * 110)
    print("TABLE A: DAILY — 8 strategies (v2, category-specific RM)")
    print("=" * 110)
    _print_table(daily_s)

    # ── Table B: HOURLY ──
    hourly_s = summary[summary["timeframe"] == "hourly"].copy()
    print("\n" + "=" * 110)
    print("TABLE B: HOURLY — 8 strategies (v2, category-specific RM)")
    print("=" * 110)
    _print_table(hourly_s)

    # ── Exit analysis ──
    print("\n" + "=" * 100)
    print("EXIT ANALYSIS (% of all exits)")
    print("=" * 100)
    exit_analysis = make_exit_analysis(exit_df.copy())
    print(f"{'Strategy':<18} {'TF':<7} {'Cat':<11} {'SL%':>6} {'TP%':>6} "
          f"{'Signal%':>8} {'Trail%':>7} {'MaxH%':>6} {'ADX%':>6} {'Total':>6}")
    print("-" * 100)
    for _, r in exit_analysis.iterrows():
        total = sum(r[c] for c in ["sl", "tp", "signal", "trailing", "max_hold", "adx_breakout"])
        print(f"{r['strategy']:<18} {r['timeframe']:<7} {r['category']:<11} "
              f"{r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% "
              f"{r['signal_pct']:>7.1f}% {r['trailing_pct']:>6.1f}% "
              f"{r['max_hold_pct']:>5.1f}% {r['adx_breakout_pct']:>5.1f}% "
              f"{total:>6.0f}")

    # ── Table C: v1 vs v2 ──
    comp = make_comparison(summary, v1_summary)
    if comp is not None:
        print("\n" + "=" * 100)
        print("TABLE C: v1 vs v2 COMPARISON (Sharpe median, sorted by improvement)")
        print("=" * 100)
        print(f"{'Strategy':<18} {'TF':<7} {'Cat':<11} {'Sharpe_v1':>10} "
              f"{'Sharpe_v2':>10} {'Delta':>8} {'AnnR_v1%':>9} {'AnnR_v2%':>9} "
              f"{'MaxDD_v1%':>10} {'MaxDD_v2%':>10}")
        print("-" * 100)
        for _, r in comp.iterrows():
            d = r["delta"]
            d_str = f"{d:>+7.4f}" if not np.isnan(d) else "    N/A"
            s_v1 = f"{r['sharpe_v1']:>9.4f}" if not np.isnan(r['sharpe_v1']) else "      N/A"
            print(f"{r['strategy']:<18} {r['timeframe']:<7} {r['category']:<11} "
                  f"{s_v1} {r['sharpe_v2']:>9.4f} {d_str} "
                  f"{r['ann_ret_v1']:>+8.2f}% {r['ann_ret_v2']:>+8.2f}% "
                  f"{r['maxdd_v1']:>+9.2f}% {r['maxdd_v2']:>+9.2f}%")

        avg_delta = comp["delta"].mean()
        n_improved = (comp["delta"] > 0).sum()
        print(f"\n  Average Δ Sharpe = {avg_delta:+.4f}")
        print(f"  Improved: {n_improved} / {len(comp)} strategy×TF combinations")

    # ── Overall summary ──
    n_sel = summary["selected"].sum()
    print(f"\n{'=' * 60}")
    print(f"SELECTED (Sharpe>0, Exp>5%, MaxDD>-55%): {n_sel} / {len(summary)}")
    print(f"{'=' * 60}")

    sel = summary[summary["selected"]]
    if len(sel) > 0:
        for _, r in sel.iterrows():
            print(f"  {r['rank']:>2}. {r['strategy']:<18} [{r['timeframe']}] "
                  f"Sharpe={r['sharpe_median']:.3f}  "
                  f"Return={r['annual_return_median']*100:+.1f}%  "
                  f"MaxDD={r['max_drawdown_median']*100:+.1f}%")


def _print_table(sub):
    hdr = (f"{'#':>2} {'Strategy':<18} {'Cat':<11} "
           f"{'Sharpe':>7} {'AnnR%':>7} {'MaxDD%':>7} {'Calmar':>7} "
           f"{'WinR%':>6} {'PF':>6} {'PayR':>5} {'Exp%':>5} {'Turn':>5} {'AvgT':>5} {'SEL':>4}")
    print(hdr)
    print("-" * 116)
    printed_sep = False
    for _, r in sub.iterrows():
        if not printed_sep and not r["selected"]:
            print("-" * 40 + " BELOW CUTOFF " + "-" * 76)
            printed_sep = True
        sel_mark = " *" if r["selected"] else ""
        pf = r["profit_factor_median"]
        pf_s = f"{pf:>6.2f}" if pf < 90 else f"{'inf':>6}"
        pr = r["payoff_ratio_median"]
        pr_s = f"{pr:>5.2f}" if pr < 90 else f"{'inf':>5}"
        print(f"{r['rank']:>2} {r['strategy']:<18} {r['category']:<11} "
              f"{r['sharpe_median']:>7.3f} "
              f"{r['annual_return_median']*100:>+6.1f}% "
              f"{r['max_drawdown_median']*100:>+6.1f}% "
              f"{r['calmar_median']:>7.3f} "
              f"{r['win_rate_median']*100:>5.1f}% "
              f"{pf_s} "
              f"{pr_s} "
              f"{r['exposure_median']*100:>4.1f}% "
              f"{r['turnover_median']:>5.0f} "
              f"{r['avg_trade_bars_median']:>5.1f}"
              f"{sel_mark}")
    if not printed_sep:
        print("-" * 40 + " BELOW CUTOFF " + "-" * 76)
    print("-" * 116)


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Strategy Screener v2: Category-specific RM")
    print("  Trend:      trailing stop + break-even, NO TP")
    print("  Contrarian: SL=1.5×ATR, TP=2.0×ATR")
    print("  Range:      SL=TP=1.5×ATR + ADX>30 breakout exit")
    print("  Filters:    vol-regime, relaxed RSI, BW percentiles")
    print("=" * 70)

    daily, hourly = load_data()
    signals, exit_df = run_all(daily, hourly)
    metrics = compute_metrics(signals, exit_df)
    summary = aggregate(metrics)
    summary = select_and_rank(summary)

    v1_summary = load_v1_summary()
    if v1_summary is not None:
        print("  v1 summary loaded for comparison")
    else:
        print("  v1 summary not found — skipping comparison")

    save_all(signals, metrics, summary, exit_df, v1_summary)
    print_results(summary, exit_df, v1_summary)
    print("\nDONE")


if __name__ == "__main__":
    main()
