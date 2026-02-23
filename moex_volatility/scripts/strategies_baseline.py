#!/usr/bin/env python3
"""
strategies_baseline.py — 6 baseline trading strategies WITH FILTERS (no vol forecasts).
Approach A: benchmark for comparison with forecast-based approaches B/C/D.
Period: 2020-01-01 — end of data. 17 tickers.

Risk Management:
  RM1: ATR(14) stops — SL=2×ATR, TP=3×ATR from entry
  RM2: Volume filter — skip entry if Vol < 0.5×SMA(Vol,20)
  RM3: Max holding period per strategy

Filters by category:
  Contrarian (S1,S2): RSI confirmation, MA200 trend filter
  Trend (S3,S4): ADX>20 trend strength, volume breakout, MA200
  Range (S5,S6): ADX<25 flat market, BandWidth filter
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
CANDLE_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/raw/candles_10m")
OUT_DIR = BASE / "results" / "final" / "strategies"
OUT_DATA = OUT_DIR / "data"
OUT_TABLES = OUT_DIR / "tables"

TICKERS = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
           "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"]

WARMUP = 200  # MA200 needs 200 days
START_DATE = pd.Timestamp("2020-01-01")


# ============================================================
# Step 0: Load data
# ============================================================

def load_daily_ohlcv():
    """Aggregate 10-min candles to daily OHLCV for 17 model tickers."""
    print("Loading and aggregating 10-min candles to daily OHLCV...")
    frames = []
    for ticker in TICKERS:
        path = CANDLE_DIR / f"{ticker}.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        daily = df.groupby("date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index()
        daily["ticker"] = ticker
        frames.append(daily)

    ohlcv = pd.concat(frames, ignore_index=True)
    ohlcv = ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  {len(ohlcv)} rows, {ohlcv['ticker'].nunique()} tickers, "
          f"{ohlcv['date'].min().date()} — {ohlcv['date'].max().date()}")
    return ohlcv


# ============================================================
# Technical indicators
# ============================================================

def calc_rsi(close, period=14):
    """RSI (Wilder's smoothing)."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    rsi = np.full(len(close), np.nan)

    if len(close) < period + 1:
        return rsi

    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, len(close)):
        if avg_loss[i] < 1e-12:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def calc_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


def calc_adx(high, low, close, period=14):
    """ADX (Average Directional Index)."""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)

    # +DM, -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    # Wilder's smoothing
    atr_s = np.full(n, np.nan)
    plus_di_s = np.full(n, np.nan)
    minus_di_s = np.full(n, np.nan)

    atr_s[period] = np.sum(tr[1:period + 1])
    plus_di_s[period] = np.sum(plus_dm[1:period + 1])
    minus_di_s[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        plus_di_s[i] = plus_di_s[i - 1] - plus_di_s[i - 1] / period + plus_dm[i]
        minus_di_s[i] = minus_di_s[i - 1] - minus_di_s[i - 1] / period + minus_dm[i]

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for i in range(period, n):
        if atr_s[i] > 1e-12:
            plus_di[i] = 100.0 * plus_di_s[i] / atr_s[i]
            minus_di[i] = 100.0 * minus_di_s[i] / atr_s[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-12:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0

    # ADX = smoothed DX
    adx = np.full(n, np.nan)
    start = 2 * period
    if start < n:
        adx[start] = np.nanmean(dx[period:start + 1])
        for i in range(start + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def calc_supertrend(high, low, close, period=14, mult=3.0):
    """Supertrend indicator. Returns (supertrend_line, direction)."""
    n = len(close)
    atr = calc_atr(high, low, close, period)

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n)

    start = period
    final_upper[start] = basic_upper[start]
    final_lower[start] = basic_lower[start]
    direction[start] = 1.0 if close[start] > basic_upper[start] else -1.0
    supertrend[start] = final_lower[start] if direction[start] == 1 else final_upper[start]

    for t in range(start + 1, n):
        if np.isnan(atr[t]):
            direction[t] = direction[t - 1]
            final_upper[t] = basic_upper[t]
            final_lower[t] = basic_lower[t]
            supertrend[t] = supertrend[t - 1]
            continue

        # Final upper band
        if basic_upper[t] < final_upper[t - 1] or close[t - 1] > final_upper[t - 1]:
            final_upper[t] = basic_upper[t]
        else:
            final_upper[t] = final_upper[t - 1]

        # Final lower band
        if basic_lower[t] > final_lower[t - 1] or close[t - 1] < final_lower[t - 1]:
            final_lower[t] = basic_lower[t]
        else:
            final_lower[t] = final_lower[t - 1]

        # Direction
        if direction[t - 1] == 1:
            direction[t] = -1.0 if close[t] < final_lower[t] else 1.0
        else:
            direction[t] = 1.0 if close[t] > final_upper[t] else -1.0

        supertrend[t] = final_lower[t] if direction[t] == 1 else final_upper[t]

    return supertrend, direction


# ============================================================
# Common indicators precomputation
# ============================================================

def precompute_indicators(close, high, low, volume):
    """Precompute all indicators needed across strategies."""
    n = len(close)
    cs = pd.Series(close)
    vs = pd.Series(volume)

    ind = {}

    # Moving averages
    ind["ma20"] = cs.rolling(20).mean().values
    ind["ma200"] = cs.rolling(200).mean().values
    ind["std20"] = cs.rolling(20).std(ddof=1).values
    ind["std20"] = np.where(ind["std20"] < 1e-12, 1e-12, ind["std20"])

    # Z-score
    ind["z"] = (close - ind["ma20"]) / ind["std20"]

    # Bollinger Bands
    ind["bb_upper"] = ind["ma20"] + 2.0 * ind["std20"]
    ind["bb_lower"] = ind["ma20"] - 2.0 * ind["std20"]
    ind["bb_width"] = np.where(
        np.abs(ind["ma20"]) > 1e-12,
        (ind["bb_upper"] - ind["bb_lower"]) / ind["ma20"],
        0.0
    )
    ind["bw_median"] = pd.Series(ind["bb_width"]).rolling(252, min_periods=50).median().values

    # RSI
    ind["rsi14"] = calc_rsi(close, 14)

    # ADX
    ind["adx14"] = calc_adx(high, low, close, 14)

    # ATR
    ind["atr14"] = calc_atr(high, low, close, 14)

    # Donchian channels
    ind["high_ch20"] = pd.Series(high).rolling(20).max().values
    ind["low_ch20"] = pd.Series(low).rolling(20).min().values

    # Volume filter
    ind["vol_sma20"] = vs.rolling(20).mean().values

    # Volume ratio for breakout confirmation
    ind["vol_ratio"] = np.where(
        ind["vol_sma20"] > 1e-12,
        volume / ind["vol_sma20"],
        0.0
    )

    # Supertrend
    ind["supertrend"], ind["st_direction"] = calc_supertrend(high, low, close, 14, 3.0)

    # Pivot points
    ind["pivot"] = np.full(n, np.nan)
    ind["s1_level"] = np.full(n, np.nan)
    ind["r1_level"] = np.full(n, np.nan)
    for t in range(1, n):
        p = (high[t - 1] + low[t - 1] + close[t - 1]) / 3.0
        ind["pivot"][t] = p
        ind["s1_level"][t] = 2 * p - high[t - 1]
        ind["r1_level"][t] = 2 * p - low[t - 1]

    # VWAP proxy
    tp = (high + low + close) / 3.0
    tp_vol = tp * volume
    sma_tpvol = pd.Series(tp_vol).rolling(20).mean().values
    sma_vol = vs.rolling(20).mean().values
    sma_vol_safe = np.where(sma_vol < 1e-12, 1e-12, sma_vol)
    ind["vwap"] = sma_tpvol / sma_vol_safe
    dev = close - ind["vwap"]
    ind["sigma_vwap"] = pd.Series(dev).rolling(20).std(ddof=1).values
    ind["sigma_vwap"] = np.where(ind["sigma_vwap"] < 1e-12, 1e-12, ind["sigma_vwap"])

    return ind


# ============================================================
# Risk management: ATR stops
# ============================================================

def check_sl_tp(position, entry_price, sl_price, tp_price, high_t1, low_t1, close_t1):
    """Check if SL or TP is hit on day t+1 using High/Low.
    Returns (closed, exit_price_approx, exit_type).
    If both SL and TP hit same day → SL (conservative).
    """
    if position == 0:
        return False, close_t1, None

    if position == 1.0:  # long
        sl_hit = low_t1 <= sl_price
        tp_hit = high_t1 >= tp_price
    else:  # short
        sl_hit = high_t1 >= sl_price
        tp_hit = low_t1 <= tp_price

    if sl_hit and tp_hit:
        return True, sl_price, "SL"  # conservative
    elif sl_hit:
        return True, sl_price, "SL"
    elif tp_hit:
        return True, tp_price, "TP"
    return False, close_t1, None


# ============================================================
# Strategy signal generators (entry conditions only)
# ============================================================
# Each returns (entry_signal, exit_signal) for a given bar t
# entry_signal: +1 (long), -1 (short), 0 (no signal)
# exit_signal: True if strategy-specific exit condition met

def s1_entry(t, close, ind):
    """S1: MA Mean Reversion — entry conditions with filters."""
    z = ind["z"][t]
    rsi = ind["rsi14"][t]
    ma200 = ind["ma200"][t]
    c = close[t]

    if np.isnan(rsi) or np.isnan(ma200) or np.isnan(z):
        return 0
    if z < -2.0 and rsi < 30 and c > ma200:
        return 1
    if z > 2.0 and rsi > 70 and c < ma200:
        return -1
    return 0


def s1_exit(t, close, ind, position):
    """S1: exit when |z| < 0.5."""
    z = ind["z"][t]
    if np.isnan(z):
        return False
    return abs(z) < 0.5


def s2_entry(t, close, ind):
    """S2: Bollinger Bands — entry with RSI, BandWidth, MA200 filters."""
    c = close[t]
    rsi = ind["rsi14"][t]
    ma200 = ind["ma200"][t]
    bw = ind["bb_width"][t]
    bw_med = ind["bw_median"][t]
    lower = ind["bb_lower"][t]
    upper = ind["bb_upper"][t]

    if np.isnan(rsi) or np.isnan(ma200) or np.isnan(bw_med):
        return 0
    if c <= lower and rsi < 35 and bw > bw_med and c > ma200:
        return 1
    if c >= upper and rsi > 65 and bw > bw_med and c < ma200:
        return -1
    return 0


def s2_exit(t, close, ind, position):
    """S2: exit when close crosses MA20."""
    c = close[t]
    ma20 = ind["ma20"][t]
    if np.isnan(ma20):
        return False
    if position == 1.0 and c >= ma20:
        return True
    if position == -1.0 and c <= ma20:
        return True
    return False


def s3_entry(t, close, ind, volume):
    """S3: Donchian Channels — entry with ADX, volume ratio, MA200 filters."""
    if t < 1:
        return 0
    c = close[t]
    adx = ind["adx14"][t]
    ma200 = ind["ma200"][t]
    vr = ind["vol_ratio"][t]
    high_ch_prev = ind["high_ch20"][t - 1]
    low_ch_prev = ind["low_ch20"][t - 1]

    if np.isnan(adx) or np.isnan(ma200) or np.isnan(high_ch_prev):
        return 0
    if c > high_ch_prev and adx > 20 and vr > 1.2 and c > ma200:
        return 1
    if c < low_ch_prev and adx > 20 and vr > 1.2 and c < ma200:
        return -1
    return 0


def s3_exit(t, close, ind, position):
    """S3: exit on reverse breakout (opposite channel breach)."""
    if t < 1:
        return False
    c = close[t]
    high_ch_prev = ind["high_ch20"][t - 1]
    low_ch_prev = ind["low_ch20"][t - 1]
    if np.isnan(high_ch_prev):
        return False
    if position == 1.0 and c < low_ch_prev:
        return True
    if position == -1.0 and c > high_ch_prev:
        return True
    return False


def s4_entry(t, close, ind):
    """S4: Supertrend — entry on crossover with ADX and MA200 filters."""
    if t < 1:
        return 0
    c = close[t]
    st = ind["supertrend"][t]
    st_prev = ind["supertrend"][t - 1]
    c_prev = close[t - 1]
    adx = ind["adx14"][t]
    ma200 = ind["ma200"][t]

    if np.isnan(st) or np.isnan(st_prev) or np.isnan(adx) or np.isnan(ma200):
        return 0

    # Crossover up: close > supertrend AND prev_close <= prev_supertrend
    if c > st and c_prev <= st_prev and adx > 20 and c > ma200:
        return 1
    # Crossover down
    if c < st and c_prev >= st_prev and adx > 20 and c < ma200:
        return -1
    return 0


def s4_exit(t, close, ind, position):
    """S4: exit on reverse supertrend crossover."""
    if t < 1:
        return False
    c = close[t]
    st = ind["supertrend"][t]
    st_prev = ind["supertrend"][t - 1]
    c_prev = close[t - 1]
    if np.isnan(st) or np.isnan(st_prev):
        return False
    if position == 1.0 and c < st and c_prev >= st_prev:
        return True
    if position == -1.0 and c > st and c_prev <= st_prev:
        return True
    return False


def s5_entry(t, close, ind):
    """S5: Pivot Points — entry with ADX<25 and BandWidth filters."""
    c = close[t]
    s1_lev = ind["s1_level"][t]
    r1_lev = ind["r1_level"][t]
    adx = ind["adx14"][t]
    bw = ind["bb_width"][t]
    bw_med = ind["bw_median"][t]

    if np.isnan(s1_lev) or np.isnan(adx) or np.isnan(bw_med):
        return 0
    if c <= s1_lev and adx < 25 and bw < bw_med * 1.5:
        return 1
    if c >= r1_lev and adx < 25 and bw < bw_med * 1.5:
        return -1
    return 0


def s5_exit(t, close, ind, position):
    """S5: exit at Pivot or opposite level."""
    c = close[t]
    pivot = ind["pivot"][t]
    r1_lev = ind["r1_level"][t]
    s1_lev = ind["s1_level"][t]
    if np.isnan(pivot):
        return False
    if position == 1.0 and (c >= pivot or c >= r1_lev):
        return True
    if position == -1.0 and (c <= pivot or c <= s1_lev):
        return True
    return False


def s6_entry(t, close, ind):
    """S6: VWAP Reversion — entry with ADX<25 and BandWidth filters."""
    c = close[t]
    vwap = ind["vwap"][t]
    sigma = ind["sigma_vwap"][t]
    adx = ind["adx14"][t]
    bw = ind["bb_width"][t]
    bw_med = ind["bw_median"][t]

    if np.isnan(vwap) or np.isnan(adx) or np.isnan(bw_med):
        return 0
    if c < vwap - 1.5 * sigma and adx < 25 and bw < bw_med * 1.5:
        return 1
    if c > vwap + 1.5 * sigma and adx < 25 and bw < bw_med * 1.5:
        return -1
    return 0


def s6_exit(t, close, ind, position):
    """S6: exit when |close-VWAP| < 0.5*sigma."""
    c = close[t]
    vwap = ind["vwap"][t]
    sigma = ind["sigma_vwap"][t]
    if np.isnan(vwap):
        return False
    return abs(c - vwap) < 0.5 * sigma


# ============================================================
# Strategy definitions
# ============================================================

STRATEGIES = {
    "S1_MA_Reversion": {
        "category": "Contrarian",
        "entry_func": lambda t, c, h, l, v, ind: s1_entry(t, c, ind),
        "exit_func": lambda t, c, h, l, v, ind, pos: s1_exit(t, c, ind, pos),
        "max_hold": 20,
        "use_vol_filter": True,  # RM2
    },
    "S2_Bollinger": {
        "category": "Contrarian",
        "entry_func": lambda t, c, h, l, v, ind: s2_entry(t, c, ind),
        "exit_func": lambda t, c, h, l, v, ind, pos: s2_exit(t, c, ind, pos),
        "max_hold": 15,
        "use_vol_filter": True,
    },
    "S3_Donchian": {
        "category": "Trend",
        "entry_func": lambda t, c, h, l, v, ind: s3_entry(t, c, ind, v),
        "exit_func": lambda t, c, h, l, v, ind, pos: s3_exit(t, c, ind, pos),
        "max_hold": None,  # no max hold for trend
        "use_vol_filter": True,
    },
    "S4_Supertrend": {
        "category": "Trend",
        "entry_func": lambda t, c, h, l, v, ind: s4_entry(t, c, ind),
        "exit_func": lambda t, c, h, l, v, ind, pos: s4_exit(t, c, ind, pos),
        "max_hold": None,
        "use_vol_filter": True,
    },
    "S5_Pivot": {
        "category": "Range",
        "entry_func": lambda t, c, h, l, v, ind: s5_entry(t, c, ind),
        "exit_func": lambda t, c, h, l, v, ind, pos: s5_exit(t, c, ind, pos),
        "max_hold": 10,
        "use_vol_filter": True,
    },
    "S6_VWAP": {
        "category": "Range",
        "entry_func": lambda t, c, h, l, v, ind: s6_entry(t, c, ind),
        "exit_func": lambda t, c, h, l, v, ind, pos: s6_exit(t, c, ind, pos),
        "max_hold": 10,
        "use_vol_filter": True,
    },
}


# ============================================================
# Backtest engine with full risk management
# ============================================================

def backtest_strategy(sname, sinfo, close, high, low, volume, dates, ind):
    """
    Run one strategy on one ticker with full RM:
      RM1: ATR stops (SL=2×ATR, TP=3×ATR)
      RM2: Volume filter
      RM3: Max holding period

    Timing:
      - Indicators computed on close[t]
      - Position opened at close[t]
      - Return = pos[t] * log(close[t+1]/close[t])
      - SL/TP checked on High/Low of day t+1
    """
    n = len(close)
    entry_func = sinfo["entry_func"]
    exit_func = sinfo["exit_func"]
    max_hold = sinfo["max_hold"]
    use_vol_filter = sinfo["use_vol_filter"]

    atr14 = ind["atr14"]
    vol_sma20 = ind["vol_sma20"]

    # Output arrays
    positions = np.zeros(n, dtype=np.float64)
    sl_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)
    entry_prices = np.full(n, np.nan)
    days_in_pos = np.zeros(n, dtype=np.int32)

    # Filter tracking
    filter_stats = {
        "raw_signals": 0,
        "passed_volume": 0,
        "passed_all": 0,
        "exits_sl": 0,
        "exits_tp": 0,
        "exits_signal": 0,
        "exits_max_hold": 0,
    }

    current_pos = 0.0
    current_entry = np.nan
    current_sl = np.nan
    current_tp = np.nan
    held_days = 0

    for t in range(WARMUP, n):
        # ── Step 1: If in position, check exits ──
        if current_pos != 0:
            held_days += 1
            closed = False

            # RM1: Check SL/TP using today's High/Low
            # (today = t, we entered at t-held_days, checking t's bar)
            if held_days >= 1:  # at least one day in position
                if current_pos == 1.0:
                    sl_hit = low[t] <= current_sl
                    tp_hit = high[t] >= current_tp
                else:
                    sl_hit = high[t] >= current_sl
                    tp_hit = low[t] <= current_tp

                if sl_hit and tp_hit:
                    closed = True
                    filter_stats["exits_sl"] += 1
                elif sl_hit:
                    closed = True
                    filter_stats["exits_sl"] += 1
                elif tp_hit:
                    closed = True
                    filter_stats["exits_tp"] += 1

            # Strategy-specific exit
            if not closed:
                if exit_func(t, close, high, low, volume, ind, current_pos):
                    closed = True
                    filter_stats["exits_signal"] += 1

            # RM3: Max holding period
            if not closed and max_hold is not None:
                if held_days >= max_hold:
                    closed = True
                    filter_stats["exits_max_hold"] += 1

            if closed:
                current_pos = 0.0
                current_entry = np.nan
                current_sl = np.nan
                current_tp = np.nan
                held_days = 0

        # ── Step 2: If flat, check entry ──
        if current_pos == 0:
            signal = entry_func(t, close, high, low, volume, ind)

            if signal != 0:
                filter_stats["raw_signals"] += 1

                # RM2: Volume filter
                vol_ok = True
                if use_vol_filter:
                    vs = vol_sma20[t]
                    if not np.isnan(vs) and vs > 0:
                        if volume[t] < 0.5 * vs:
                            vol_ok = False

                if vol_ok:
                    filter_stats["passed_volume"] += 1
                    filter_stats["passed_all"] += 1

                    current_pos = float(signal)
                    current_entry = close[t]
                    atr_val = atr14[t] if not np.isnan(atr14[t]) else 0.0

                    if signal == 1:  # long
                        current_sl = current_entry - 2.0 * atr_val
                        current_tp = current_entry + 3.0 * atr_val
                    else:  # short
                        current_sl = current_entry + 2.0 * atr_val
                        current_tp = current_entry - 3.0 * atr_val

                    held_days = 0

        # Record state
        positions[t] = current_pos
        sl_prices[t] = current_sl
        tp_prices[t] = current_tp
        entry_prices[t] = current_entry
        days_in_pos[t] = held_days

    return positions, sl_prices, tp_prices, entry_prices, days_in_pos, filter_stats


# ============================================================
# Run all strategies
# ============================================================

def run_backtest(ohlcv):
    """Run all 6 strategies on all 17 tickers."""
    print("\n" + "=" * 60)
    print("Running backtests with full filters & risk management...")
    print("=" * 60)

    all_records = []
    all_filter_stats = []

    for sname, sinfo in STRATEGIES.items():
        cat = sinfo["category"]
        print(f"\n  {sname} ({cat}):", end="")

        for ticker in TICKERS:
            tdf = ohlcv[ohlcv["ticker"] == ticker].sort_values("date").reset_index(drop=True)

            close = tdf["close"].values
            high_arr = tdf["high"].values
            low_arr = tdf["low"].values
            volume_arr = tdf["volume"].values.astype(np.float64)
            dates = tdf["date"].values

            # Precompute indicators
            ind = precompute_indicators(close, high_arr, low_arr, volume_arr)

            # Run strategy
            positions, sl_prices, tp_prices, entry_prices, days_in_pos, fstats = \
                backtest_strategy(sname, sinfo, close, high_arr, low_arr, volume_arr, dates, ind)

            # Compute returns: ret_t = pos_t * log(close_{t+1}/close_t)
            log_ret = np.zeros(len(close))
            log_ret[:-1] = np.log(close[1:] / close[:-1])
            daily_return = positions * log_ret
            # For days where SL/TP was hit, approximate return using SL/TP price
            # (already handled implicitly since position goes flat on exit day)

            # Filter to strategy period (>= START_DATE)
            start_mask = dates >= np.datetime64(START_DATE)

            dates_strat = dates[start_mask]
            pos_strat = positions[start_mask]
            dr_strat = daily_return[start_mask]
            cumret_strat = np.cumsum(dr_strat)
            sl_strat = sl_prices[start_mask]
            tp_strat = tp_prices[start_mask]
            ep_strat = entry_prices[start_mask]
            dip_strat = days_in_pos[start_mask]

            for i in range(len(dates_strat)):
                all_records.append({
                    "date": dates_strat[i],
                    "ticker": ticker,
                    "strategy": sname,
                    "category": cat,
                    "position": pos_strat[i],
                    "daily_return": dr_strat[i],
                    "cumret": cumret_strat[i],
                    "sl_price": sl_strat[i],
                    "tp_price": tp_strat[i],
                    "entry_price": ep_strat[i],
                    "days_in_position": dip_strat[i],
                })

            # Filter stats
            all_filter_stats.append({
                "strategy": sname,
                "ticker": ticker,
                "raw_signals": fstats["raw_signals"],
                "passed_volume": fstats["passed_volume"],
                "passed_all": fstats["passed_all"],
                "exits_sl": fstats["exits_sl"],
                "exits_tp": fstats["exits_tp"],
                "exits_signal": fstats["exits_signal"],
                "exits_max_hold": fstats["exits_max_hold"],
            })

            print(f" {ticker}", end="", flush=True)
        print()

    signals_df = pd.DataFrame(all_records)
    signals_df["date"] = pd.to_datetime(signals_df["date"])
    filter_df = pd.DataFrame(all_filter_stats)

    print(f"\n  Total records: {len(signals_df):,}")
    return signals_df, filter_df


# ============================================================
# Metrics computation
# ============================================================

def compute_metrics(signals_df):
    """Compute performance metrics per strategy × ticker."""
    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)

    rows = []
    for sname in STRATEGIES:
        for ticker in TICKERS:
            mask = (signals_df["strategy"] == sname) & (signals_df["ticker"] == ticker)
            sdf = signals_df[mask].sort_values("date")

            dr = sdf["daily_return"].values
            pos = sdf["position"].values
            n_days = len(dr)

            if n_days == 0:
                continue

            # Active days
            active = pos != 0
            n_active = active.sum()
            exposure = n_active / n_days if n_days > 0 else 0

            # Returns
            mean_ret = np.mean(dr)
            std_ret = np.std(dr, ddof=1) if n_days > 1 else 1e-10

            ann_return = mean_ret * 252
            ann_vol = std_ret * np.sqrt(252)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-12 else 0.0

            # Max drawdown
            cumret = np.cumsum(dr)
            running_max = np.maximum.accumulate(cumret)
            drawdown = cumret - running_max
            max_dd = drawdown.min()

            calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

            # Win rate (only among active days)
            if n_active > 0:
                active_returns = dr[active]
                win_rate = (active_returns > 0).sum() / n_active
                gross_profit = active_returns[active_returns > 0].sum()
                gross_loss = abs(active_returns[active_returns < 0].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else np.inf
            else:
                win_rate = 0.0
                profit_factor = 0.0

            # Turnover
            pos_changes = np.sum(np.abs(np.diff(pos)) > 0)
            n_years = n_days / 252
            turnover = pos_changes / n_years if n_years > 0 else 0

            # Average trade duration
            dip = sdf["days_in_position"].values
            # Count trades: transitions from 0 to non-zero
            trade_starts = 0
            total_trade_days = 0
            in_trade = False
            for i in range(len(pos)):
                if pos[i] != 0 and not in_trade:
                    trade_starts += 1
                    in_trade = True
                elif pos[i] == 0:
                    in_trade = False
                if pos[i] != 0:
                    total_trade_days += 1
            avg_trade_days = total_trade_days / trade_starts if trade_starts > 0 else 0

            rows.append({
                "strategy": sname,
                "category": STRATEGIES[sname]["category"],
                "ticker": ticker,
                "sharpe": sharpe,
                "annual_return": ann_return,
                "annual_vol": ann_vol,
                "max_drawdown": max_dd,
                "calmar": calmar,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "exposure": exposure,
                "turnover": turnover,
                "avg_trade_days": avg_trade_days,
                "n_trades": trade_starts,
                "n_days": n_days,
                "n_active": n_active,
            })

    metrics_df = pd.DataFrame(rows)
    print(f"  {len(metrics_df)} strategy×ticker combinations")
    return metrics_df


def aggregate_metrics(metrics_df):
    """Aggregate metrics across tickers."""
    metric_cols = ["sharpe", "annual_return", "annual_vol", "max_drawdown",
                   "calmar", "win_rate", "profit_factor", "exposure",
                   "turnover", "avg_trade_days"]

    summary_rows = []
    for sname in STRATEGIES:
        sdf = metrics_df[metrics_df["strategy"] == sname]
        row = {
            "strategy": sname,
            "category": STRATEGIES[sname]["category"],
        }
        for col in metric_cols:
            row[f"{col}_median"] = sdf[col].median()
            row[f"{col}_mean"] = sdf[col].mean()
            row[f"{col}_std"] = sdf[col].std()
            row[f"{col}_q25"] = sdf[col].quantile(0.25)
            row[f"{col}_q75"] = sdf[col].quantile(0.75)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def aggregate_filter_stats(filter_df):
    """Aggregate filter statistics across tickers per strategy."""
    agg = filter_df.groupby("strategy").agg({
        "raw_signals": "sum",
        "passed_volume": "sum",
        "passed_all": "sum",
        "exits_sl": "sum",
        "exits_tp": "sum",
        "exits_signal": "sum",
        "exits_max_hold": "sum",
    }).reset_index()

    agg["volume_filter_rate"] = np.where(
        agg["raw_signals"] > 0,
        agg["passed_volume"] / agg["raw_signals"] * 100,
        0.0
    )
    return agg


# ============================================================
# Output
# ============================================================

def save_outputs(signals_df, metrics_df, summary_df, filter_df):
    """Save all outputs."""
    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # 1. Signals parquet
    path = OUT_DATA / "signals_baseline.parquet"
    signals_df.to_parquet(path, index=False)
    print(f"  {path.relative_to(BASE)}: {len(signals_df):,} rows")

    # 2. Summary CSV
    summary_clean = summary_df[[
        "strategy", "category",
        "sharpe_median", "sharpe_mean",
        "annual_return_median", "annual_vol_median",
        "max_drawdown_median", "calmar_median",
        "win_rate_median", "profit_factor_median",
        "exposure_median", "turnover_median", "avg_trade_days_median"
    ]].copy()
    summary_clean.columns = [
        "strategy", "category", "sharpe_median", "sharpe_mean",
        "return_annual", "vol_annual", "maxdd",
        "calmar", "winrate", "profit_factor",
        "exposure", "turnover", "avg_trade_days"
    ]
    path = OUT_TABLES / "baseline_summary.csv"
    summary_clean.to_csv(path, index=False)
    print(f"  {path.relative_to(BASE)}: {len(summary_clean)} strategies")

    # 3. By ticker
    path = OUT_TABLES / "baseline_by_ticker.csv"
    metrics_df.to_csv(path, index=False)
    print(f"  {path.relative_to(BASE)}: {len(metrics_df)} rows")

    # 4. Filter stats
    fstats_agg = aggregate_filter_stats(filter_df)
    path = OUT_TABLES / "baseline_filter_stats.csv"
    fstats_agg.to_csv(path, index=False)
    print(f"  {path.relative_to(BASE)}: {len(fstats_agg)} rows")

    # 5. Full summary
    path = OUT_TABLES / "baseline_summary_full.csv"
    summary_df.to_csv(path, index=False)
    print(f"  {path.relative_to(BASE)}: {len(summary_df)} rows")


def print_summary(summary_df, filter_df):
    """Pretty-print summary tables."""
    print("\n" + "=" * 90)
    print("BASELINE STRATEGIES WITH FILTERS — SUMMARY (median across 17 tickers)")
    print("=" * 90)

    header = (f"{'Strategy':<18} {'Cat':<12} {'Sharpe':>7} {'AnnRet%':>8} "
              f"{'MaxDD%':>8} {'Calmar':>7} {'WinR%':>6} {'PF':>6} "
              f"{'Exp%':>6} {'Turn':>5} {'AvgD':>5}")
    print(header)
    print("-" * 90)

    for _, row in summary_df.iterrows():
        s = row["strategy"]
        cat = row["category"]
        sharpe = row["sharpe_median"]
        ann_ret = row["annual_return_median"] * 100
        max_dd = row["max_drawdown_median"] * 100
        calmar = row["calmar_median"]
        wr = row["win_rate_median"] * 100
        pf = row["profit_factor_median"]
        exp = row["exposure_median"] * 100
        turn = row["turnover_median"]
        avg_d = row["avg_trade_days_median"]

        pf_str = f"{pf:>6.2f}" if pf < 100 else f"{'inf':>6}"

        print(f"{s:<18} {cat:<12} {sharpe:>7.3f} {ann_ret:>+7.2f}% "
              f"{max_dd:>+7.2f}% {calmar:>7.3f} {wr:>5.1f}% {pf_str} "
              f"{exp:>5.1f}% {turn:>5.0f} {avg_d:>5.1f}")

    print("-" * 90)

    # Best/worst
    best = summary_df.loc[summary_df["sharpe_median"].idxmax()]
    worst = summary_df.loc[summary_df["sharpe_median"].idxmin()]
    print(f"\nBest Sharpe:  {best['strategy']} ({best['sharpe_median']:.3f})")
    print(f"Worst Sharpe: {worst['strategy']} ({worst['sharpe_median']:.3f})")

    # Filter stats
    fstats = aggregate_filter_stats(filter_df)
    print("\n" + "=" * 90)
    print("FILTER STATISTICS (summed across 17 tickers)")
    print("=" * 90)

    header2 = (f"{'Strategy':<18} {'RawSig':>8} {'PassVol':>8} {'PassAll':>8} "
               f"{'VolFilt%':>8} {'ExSL':>6} {'ExTP':>6} {'ExSig':>6} {'ExHold':>6}")
    print(header2)
    print("-" * 90)

    for _, row in fstats.iterrows():
        s = row["strategy"]
        raw = int(row["raw_signals"])
        pv = int(row["passed_volume"])
        pa = int(row["passed_all"])
        vfr = row["volume_filter_rate"]
        esl = int(row["exits_sl"])
        etp = int(row["exits_tp"])
        esig = int(row["exits_signal"])
        ehold = int(row["exits_max_hold"])

        print(f"{s:<18} {raw:>8} {pv:>8} {pa:>8} "
              f"{vfr:>7.1f}% {esl:>6} {etp:>6} {esig:>6} {ehold:>6}")

    print("-" * 90)

    # Low exposure warning
    for _, row in summary_df.iterrows():
        exp = row["exposure_median"] * 100
        if exp < 5:
            print(f"\n  WARNING: {row['strategy']} has very low exposure ({exp:.1f}%) — "
                  "filters may be too restrictive!")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Baseline Trading Strategies WITH FILTERS (Approach A)")
    print(f"Period: 2020-01-01 — end of data")
    print(f"Tickers: {len(TICKERS)}")
    print(f"Risk Management: ATR stops, volume filter, max hold")
    print("=" * 60)

    ohlcv = load_daily_ohlcv()

    signals_df, filter_df = run_backtest(ohlcv)

    metrics_df = compute_metrics(signals_df)

    summary_df = aggregate_metrics(metrics_df)

    save_outputs(signals_df, metrics_df, summary_df, filter_df)

    print_summary(summary_df, filter_df)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
