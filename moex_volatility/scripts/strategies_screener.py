#!/usr/bin/env python3
"""
strategies_screener.py — 8 strategies × 2 timeframes screener (no vol forecasts).
Baseline benchmark: 272 backtests (8 × 2 × 17 tickers).
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
    ind["bw_median"] = pd.Series(ind["bb_width"]).rolling(bw_window, min_periods=50).median().values

    # RSI, Stochastic
    ind["rsi14"] = calc_rsi(close, 14)
    ind["stoch_k"] = calc_stochastic_k(high, low, close, 14, 3)

    # ADX, ATR
    ind["adx14"] = calc_adx(high, low, close, 14)
    ind["atr14"] = calc_atr(high, low, close, 14)

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
# Strategy entry/exit functions
# ════════════════════════════════════════════════════════════

def s1_entry(t, c, ind):
    z, rsi = ind["z"][t], ind["rsi14"][t]
    if np.isnan(z) or np.isnan(rsi): return 0
    if z < -2.0 and rsi < 40: return 1
    if z > 2.0 and rsi > 60: return -1
    return 0

def s1_exit(t, c, ind, pos):
    z = ind["z"][t]
    if np.isnan(z): return False
    return abs(z) < 0.5

def s2_entry(t, c, ind):
    cl = c[t]
    bw, bw_med = ind["bb_width"][t], ind["bw_median"][t]
    lo, up = ind["bb_lower"][t], ind["bb_upper"][t]
    if np.isnan(bw_med): return 0
    if cl <= lo and bw > bw_med: return 1
    if cl >= up and bw > bw_med: return -1
    return 0

def s2_exit(t, c, ind, pos):
    ma = ind["sma20"][t]
    if np.isnan(ma): return False
    if pos == 1.0 and c[t] >= ma: return True
    if pos == -1.0 and c[t] <= ma: return True
    return False

def s3_entry(t, c, ind):
    rsi = ind["rsi14"][t]
    if np.isnan(rsi): return 0
    if rsi < 25: return 1
    if rsi > 75: return -1
    return 0

def s3_exit(t, c, ind, pos):
    rsi = ind["rsi14"][t]
    if np.isnan(rsi): return False
    if pos == 1.0 and rsi >= 50: return True
    if pos == -1.0 and rsi <= 50: return True
    return False

def s4_entry(t, c, ind):
    if t < 1: return 0
    cl, adx = c[t], ind["adx14"][t]
    hc, lc = ind["high_ch20"][t-1], ind["low_ch20"][t-1]
    vs, v = ind["vol_sma20"][t], 0.0
    if np.isnan(adx) or np.isnan(hc): return 0
    # vol ratio check is in RM — here check vol > 1.0×SMA
    if not np.isnan(vs) and vs > 0:
        v = ind["_volume"][t] / vs
    else:
        v = 1.0
    if cl > hc and adx > 20 and v > 1.0: return 1
    if cl < lc and adx > 20 and v > 1.0: return -1
    return 0

def s4_exit(t, c, ind, pos):
    if t < 1: return False
    cl = c[t]
    hc, lc = ind["high_ch20"][t-1], ind["low_ch20"][t-1]
    if np.isnan(hc): return False
    if pos == 1.0 and cl < lc: return True
    if pos == -1.0 and cl > hc: return True
    return False

def s5_entry(t, c, ind):
    if t < 1: return 0
    cl, adx = c[t], ind["adx14"][t]
    st, st_p = ind["supertrend"][t], ind["supertrend"][t-1]
    cp = c[t-1]
    if np.isnan(st) or np.isnan(st_p) or np.isnan(adx): return 0
    if cl > st and cp <= st_p and adx > 20: return 1
    if cl < st and cp >= st_p and adx > 20: return -1
    return 0

def s5_exit(t, c, ind, pos):
    if t < 1: return False
    cl, st = c[t], ind["supertrend"][t]
    cp, st_p = c[t-1], ind["supertrend"][t-1]
    if np.isnan(st) or np.isnan(st_p): return False
    if pos == 1.0 and cl < st and cp >= st_p: return True
    if pos == -1.0 and cl > st and cp <= st_p: return True
    return False

def s6_entry(t, c, ind):
    if t < 1: return 0
    f, s = ind["ma_fast"][t], ind["ma_slow"][t]
    fp, sp = ind["ma_fast"][t-1], ind["ma_slow"][t-1]
    adx = ind["adx14"][t]
    if np.isnan(f) or np.isnan(s) or np.isnan(fp) or np.isnan(sp) or np.isnan(adx):
        return 0
    if f > s and fp <= sp and adx > 15: return 1
    if f < s and fp >= sp and adx > 15: return -1
    return 0

def s6_exit(t, c, ind, pos):
    if t < 1: return False
    f, s = ind["ma_fast"][t], ind["ma_slow"][t]
    fp, sp = ind["ma_fast"][t-1], ind["ma_slow"][t-1]
    if np.isnan(f) or np.isnan(s) or np.isnan(fp) or np.isnan(sp): return False
    if pos == 1.0 and f < s and fp >= sp: return True
    if pos == -1.0 and f > s and fp <= sp: return True
    return False

def s7_entry(t, c, ind):
    cl, adx = c[t], ind["adx14"][t]
    lo, up = ind["kc_lower"][t], ind["kc_upper"][t]
    if np.isnan(adx) or np.isnan(lo): return 0
    if cl < lo and adx < 25: return 1
    if cl > up and adx < 25: return -1
    return 0

def s7_exit(t, c, ind, pos):
    ema = ind["ema20"][t]
    if np.isnan(ema): return False
    if pos == 1.0 and c[t] >= ema: return True
    if pos == -1.0 and c[t] <= ema: return True
    return False

def s8_entry(t, c, ind):
    rsi, sk, adx = ind["rsi14"][t], ind["stoch_k"][t], ind["adx14"][t]
    if np.isnan(rsi) or np.isnan(sk) or np.isnan(adx): return 0
    if rsi < 30 and sk < 20 and adx < 30: return 1
    if rsi > 70 and sk > 80 and adx < 30: return -1
    return 0

def s8_exit(t, c, ind, pos):
    rsi = ind["rsi14"][t]
    if np.isnan(rsi): return False
    if pos == 1.0 and rsi >= 50: return True
    if pos == -1.0 and rsi <= 50: return True
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
# Backtest engine
# ════════════════════════════════════════════════════════════

def backtest_one(sinfo, close, high, low, volume, ind, is_hourly):
    """Run one strategy on one ticker. Returns positions array."""
    n = len(close)
    entry_fn = sinfo["entry"]
    exit_fn = sinfo["exit"]
    max_hold = sinfo["max_hold_h"] if is_hourly else sinfo["max_hold_d"]

    atr14 = ind["atr14"]
    vol_sma20 = ind["vol_sma20"]

    positions = np.zeros(n, dtype=np.float64)
    cur_pos = 0.0
    cur_sl = np.nan
    cur_tp = np.nan
    held = 0

    # Store volume in ind for S4
    ind["_volume"] = volume

    for t in range(WARMUP, n):
        # Check exits if in position
        if cur_pos != 0:
            held += 1
            closed = False

            # RM1: ATR SL/TP check via High/Low
            if cur_pos == 1.0:
                sl_hit = low[t] <= cur_sl
                tp_hit = high[t] >= cur_tp
            else:
                sl_hit = high[t] >= cur_sl
                tp_hit = low[t] <= cur_tp

            if sl_hit:
                closed = True
            elif tp_hit:
                closed = True

            # Strategy exit
            if not closed and exit_fn(t, close, ind, cur_pos):
                closed = True

            # Max hold
            if not closed and max_hold is not None and held >= max_hold:
                closed = True

            if closed:
                cur_pos = 0.0
                cur_sl = np.nan
                cur_tp = np.nan
                held = 0

        # Check entry if flat
        if cur_pos == 0:
            sig = entry_fn(t, close, ind)
            if sig != 0:
                # RM2: volume filter
                vs = vol_sma20[t]
                if not np.isnan(vs) and vs > 0 and volume[t] < 0.5 * vs:
                    sig = 0

                if sig != 0:
                    cur_pos = float(sig)
                    atr_val = atr14[t] if not np.isnan(atr14[t]) else 0.0
                    if sig == 1:
                        cur_sl = close[t] - 2.0 * atr_val
                        cur_tp = close[t] + 3.0 * atr_val
                    else:
                        cur_sl = close[t] + 2.0 * atr_val
                        cur_tp = close[t] - 3.0 * atr_val
                    held = 0

        positions[t] = cur_pos

    return positions


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
    print(f"Running 8 strategies × 2 timeframes × 17 tickers = 272 backtests")
    print("=" * 70)

    records = []

    for tf_name, df, is_hourly, dt_col in [
        ("daily", daily, False, "date"),
        ("hourly", hourly, True, "datetime"),
    ]:
        ann_factor = np.sqrt(252 * 9) if is_hourly else np.sqrt(252)
        bars_per_year = 252 * 9 if is_hourly else 252

        print(f"\n  [{tf_name.upper()}]")

        for sname, sinfo in STRATEGIES.items():
            print(f"    {sname}:", end="", flush=True)

            for ticker in TICKERS:
                tdf = df[df["ticker"] == ticker].sort_values(dt_col).reset_index(drop=True)
                close = tdf["close"].values
                high_a = tdf["high"].values
                low_a = tdf["low"].values
                vol_a = tdf["volume"].values.astype(np.float64)
                dts = tdf[dt_col].values

                ind = precompute(close, high_a, low_a, vol_a, is_hourly)
                positions = backtest_one(sinfo, close, high_a, low_a, vol_a, ind, is_hourly)

                # Returns
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
    print(f"\n  Total records: {len(signals):,}")
    return signals


# ════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════

def compute_metrics(signals):
    print("\nComputing metrics...")
    rows = []

    for tf_name in ["daily", "hourly"]:
        ann_factor = np.sqrt(252 * 9) if tf_name == "hourly" else np.sqrt(252)
        bars_per_year = 252 * 9 if tf_name == "hourly" else 252

        for sname in STRATEGIES:
            for ticker in TICKERS:
                m = (signals["strategy"] == sname) & \
                    (signals["ticker"] == ticker) & \
                    (signals["timeframe"] == tf_name)
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
                else:
                    win_rate = 0.0
                    pf = 0.0

                # Turnover & avg trade
                changes = np.sum(np.abs(np.diff(pos)) > 0)
                n_years = n / bars_per_year
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

                rows.append({
                    "strategy": sname, "timeframe": tf_name,
                    "category": STRATEGIES[sname]["cat"], "ticker": ticker,
                    "sharpe": sharpe, "annual_return": ann_ret,
                    "annual_vol": ann_vol, "max_drawdown": max_dd,
                    "calmar": calmar, "win_rate": win_rate,
                    "profit_factor": pf, "exposure": exposure,
                    "turnover": turnover, "avg_trade_bars": avg_trade,
                    "n_trades": trades,
                })

    return pd.DataFrame(rows)


def aggregate(metrics):
    cols = ["sharpe", "annual_return", "annual_vol", "max_drawdown",
            "calmar", "win_rate", "profit_factor", "exposure",
            "turnover", "avg_trade_bars"]
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
# Selection & output
# ════════════════════════════════════════════════════════════

def select_and_rank(summary):
    """Apply selection criteria and rank."""
    summary["selected"] = (
        (summary["sharpe_median"] > 0.0) &
        (summary["exposure_median"] > 0.05) &
        (summary["max_drawdown_median"] > -0.40)
    )
    summary = summary.sort_values("sharpe_median", ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    return summary


def save_all(signals, metrics, summary):
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    p = OUT_DATA / "signals_screener.parquet"
    signals.to_parquet(p, index=False)
    print(f"  {p.name}: {len(signals):,} rows")

    p = OUT_TABLES / "screener_summary.csv"
    summary.to_csv(p, index=False)
    print(f"  {p.name}: {len(summary)} rows")

    p = OUT_TABLES / "screener_by_ticker.csv"
    metrics.to_csv(p, index=False)
    print(f"  {p.name}: {len(metrics)} rows")

    sel = summary[summary["selected"]].copy()
    p = OUT_TABLES / "screener_selected.csv"
    sel.to_csv(p, index=False)
    print(f"  {p.name}: {len(sel)} rows")


def print_results(summary):
    print("\n" + "=" * 105)
    print("SCREENER RESULTS: 8 strategies × 2 timeframes (sorted by Sharpe median)")
    print("=" * 105)

    hdr = (f"{'#':>2} {'Strategy':<18} {'TF':<7} {'Cat':<11} "
           f"{'Sharpe':>7} {'AnnR%':>7} {'MaxDD%':>7} {'Calmar':>7} "
           f"{'WinR%':>6} {'PF':>6} {'Exp%':>5} {'Turn':>5} {'AvgT':>5} {'SEL':>4}")
    print(hdr)
    print("-" * 105)

    printed_sep = False
    for _, r in summary.iterrows():
        if not printed_sep and not r["selected"]:
            print("-" * 40 + " BELOW CUTOFF " + "-" * 51)
            printed_sep = True

        sel_mark = " *" if r["selected"] else ""
        pf = r["profit_factor_median"]
        pf_s = f"{pf:>6.2f}" if pf < 90 else f"{'inf':>6}"

        print(f"{r['rank']:>2} {r['strategy']:<18} {r['timeframe']:<7} "
              f"{r['category']:<11} "
              f"{r['sharpe_median']:>7.3f} "
              f"{r['annual_return_median']*100:>+6.1f}% "
              f"{r['max_drawdown_median']*100:>+6.1f}% "
              f"{r['calmar_median']:>7.3f} "
              f"{r['win_rate_median']*100:>5.1f}% "
              f"{pf_s} "
              f"{r['exposure_median']*100:>4.1f}% "
              f"{r['turnover_median']:>5.0f} "
              f"{r['avg_trade_bars_median']:>5.1f}"
              f"{sel_mark}")

    if not printed_sep:
        print("-" * 40 + " BELOW CUTOFF " + "-" * 51)

    print("-" * 105)

    n_sel = summary["selected"].sum()
    print(f"\nSelected: {n_sel} / {len(summary)} strategy×timeframe combinations")

    sel = summary[summary["selected"]]
    if len(sel) > 0:
        print(f"\nRECOMMENDATION — take forward to vol-forecast stage:")
        for _, r in sel.iterrows():
            print(f"  {r['rank']:>2}. {r['strategy']:<18} [{r['timeframe']}]  "
                  f"Sharpe={r['sharpe_median']:.3f}  "
                  f"Return={r['annual_return_median']*100:+.1f}%  "
                  f"MaxDD={r['max_drawdown_median']*100:+.1f}%")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Strategy Screener: 8 strategies × 2 timeframes × 17 tickers")
    print("=" * 70)

    daily, hourly = load_data()
    signals = run_all(daily, hourly)
    metrics = compute_metrics(signals)
    summary = aggregate(metrics)
    summary = select_and_rank(summary)

    print("\nSaving...")
    save_all(signals, metrics, summary)
    print_results(summary)
    print("\nDONE")


if __name__ == "__main__":
    main()
