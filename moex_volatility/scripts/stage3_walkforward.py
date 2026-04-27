#!/usr/bin/env python3
"""
stage3_walkforward.py — Walk-Forward Backtesting for Approaches A/B/C/D

Walk-forward with expanding window:
  For each test_year in [2022, 2023, 2024, 2025]:
    Train on all data before test_year, test on test_year.

Train metric: trading_sharpe (no FDR, no RF, with commission)
Test metric:  calc_excess_sharpe (with FDR, with RF, with commission)

Two commission levels: 0.0005 and 0.0006
Approach A: grid search over signal x RM params, top-10 ensemble majority vote
Approaches B, C, D: overlay on A positions using vpred (sigma_h1/h5/h22)
Each of B, C, D must SEPARATELY beat A.

w = 1.0 ALWAYS. Per-year Sharpe -> np.nanmean. Years [2022,2023,2024,2025].
"""
import os, sys, warnings, time, json, datetime, gc
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from itertools import product

np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from archivist import (archive, check_stage_inputs, archive_grid_version,
                       agent_sanity_check, TimeManager, quick_check, save_configs_full)

from strategies_walkforward import (
    TICKERS, WARMUP, load_data,
    calc_rsi, calc_sma, calc_std, calc_atr, calc_adx,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily,
    _year_bounds, warmup_numba,
    SIGNAL_GRIDS, RM_GRIDS_V3, TREND_RM_GRID, CATEGORY,
    expand_grid, compute_base, precompute_sma_cache,
    dispatch_signals, dispatch_bt_atr_v3, calc_sharpe_comm,
    _pctile_cache,
)
from daily_upper_bound import bt_universal

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

# ═══════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

BASE = SCRIPT_DIR.parent
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
VPRED_PATH = BASE / "results/final/strategies/data/vpred_aligned.parquet"
GRID_PATH = BASE / "results/final/strategies/walkforward_v4/narrowed_grid_v2.json"
BEST_CONFIGS_PATH = BASE / "results/final/strategies/test_phases/FINAL_best_configs.json"
OUT_DIR = BASE / "results/final/strategies/walkforward"
GRIDS_DIR = BASE / "results/final/strategies/grids"
LOG_PATH = BASE / "results/final/strategies/MASTER_PROGRESS_LOG.txt"

BPY = 252
TEST_YEARS = [2022, 2023, 2024, 2025]
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
IS_TREND = {"S1": False, "S2": False, "S3": True, "S4": True, "S5": False, "S6": False}
STRATEGY_TYPE = {"S1": "contrarian", "S2": "contrarian", "S3": "trend", "S4": "trend",
                 "S5": "range", "S6": "range"}
COMM_LEVELS = [0.0005, 0.0006]
TRAIN_SHARPE_THRESHOLD = 0.3
ENSEMBLE_TOP_N = 10  # Match v4 TOP_K
WF_TARGET_SHARPE = 0.3  # Target per-strategy WF mean

MA_TYPES = ["sma", "ema", "dema", "hull", "kama"]
DC_WINDOWS = [3, 5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120]
ST_PERIODS = [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 35, 40, 50, 60, 80]
ST_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
PIVOT_TYPES = ["classic", "fibonacci", "woodie", "camarilla"]
VWAP_WINDOWS = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100]
WINDOWS_D = [5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]

# BCD coarse grids
B_GRID = dict(k_sl=[0.5, 1.0, 2.0, 3.0], ratio=[0.5, 1.5, 3.0],
              base_hold=[5, 15, 50], hold_gamma=[0.25, 1.0], be_trigger=[0, 1.5])
C_GRID = dict(threshold=[0.25, 0.55, 0.85], lookback=[63, 252],
              direction=["off", "rise", "fall"], term_str=[False, True], hysteresis=[0, 10])
D_GRID = dict(sigma_target=[0.05, 0.10, 0.15, 0.20, 0.30],
              gamma=[0.25, 0.5, 1.0, 2.0], smooth=[0, 10],
              vol_floor=[0.1, 0.3], vol_cap=[1.5, 2.0, 3.0])
HORIZONS = ["h1", "h5", "h22"]

# Range bands for C (range strategies)
C_RANGE_BANDS = [(0.25, 0.75), (0.3, 0.7), (0.2, 0.8)]


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage3_wf] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# FORMULAS (IMMUTABLE)
# ═══════════════════════════════════════════════════════════════

def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm):
    """OOS metric: Excess Sharpe with FDR, RF, commission."""
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    n = len(pm)
    if n < 20:
        return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exposure = np.abs(pm)
    fdr = (1.0 - exposure) * rf * cd
    daily_total = net + fdr
    excess = daily_total - rf * cd
    s = np.std(daily_total, ddof=1)
    if s < 1e-12:
        return 0.0, {}
    sharpe = np.mean(excess) * np.sqrt(252) / s
    exp_pct = np.mean(exposure) * 100
    # Trades count
    nt = 0; nw = 0; in_t = False; tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t:
            in_t = True; tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1
                if tr > 0:
                    nw += 1
                in_t = False
    cs = max((n / BPY), 0.01)
    return sharpe, dict(
        exposure=round(exp_pct, 1),
        tpy=round(nt / cs, 1),
        wr=round(nw / nt * 100 if nt > 0 else 0, 1),
        n_trades=nt,
    )


def trading_sharpe(pos, log_ret, mask, comm):
    """TRAIN metric: simple Sharpe, no FDR, no RF, with commission."""
    pm = pos[mask]; lr = log_ret[mask]
    n = len(pm)
    if n < 20:
        return -999.0
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    s = np.std(net, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(net) * np.sqrt(252) / s


# ═══════════════════════════════════════════════════════════════
# MA HELPERS
# ═══════════════════════════════════════════════════════════════

def _ema(arr, w):
    a = 2.0 / (w + 1); out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def _dema(arr, w):
    e1 = _ema(arr, w); e2 = _ema(e1, w)
    return 2 * e1 - e2

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
# PRECOMPUTE INDICATORS
# ═══════════════════════════════════════════════════════════════

def precompute_indicators(close, high, low, volume, open_arr, windows):
    n = len(close)
    ind = {}
    ind["atr14"] = calc_atr(high, low, close, 14)
    ind["rsi14"] = calc_rsi(close, 14)
    ind["adx14"] = calc_adx(high, low, close, 14)
    ind["vol_sma20"] = calc_sma(volume, 20)
    ind["rh"] = pd.Series(high).rolling(11, min_periods=1).max().values
    ind["rl"] = pd.Series(low).rolling(11, min_periods=1).min().values
    ind["ma"] = {}; ind["std"] = {}
    for mt in MA_TYPES:
        for w in windows:
            if w >= n - 10:
                continue
            try:
                ind["ma"][(mt, w)] = get_ma(close, w, mt)
                ind["std"][(mt, w)] = calc_std(close, w)
            except Exception:
                pass
    ind["z_atr"] = {}
    atr = ind["atr14"]
    for key, ma in ind["ma"].items():
        z = (close - ma) / np.where(atr > 1e-12, atr, 1.0)
        z[np.isnan(ma)] = np.nan
        ind["z_atr"][key] = z
    ind["z_std"] = {}
    for key in ind["ma"]:
        ma = ind["ma"][key]; std = ind["std"].get(key)
        if std is not None:
            z = (close - ma) / np.where((std > 1e-12) & ~np.isnan(std), std, 1.0)
            z[np.isnan(ma)] = np.nan
            ind["z_std"][key] = z
    ind["volume"] = volume
    return ind


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════

def gen_s1(z, threshold, warmup):
    sig = np.zeros(len(z), dtype=np.int8)
    v = ~np.isnan(z)
    sig[v & (z < -threshold)] = 1; sig[v & (z > threshold)] = -1
    sig[:warmup] = 0
    return sig

def gen_s2(z, bs, warmup):
    sig = np.zeros(len(z), dtype=np.int8)
    v = ~np.isnan(z)
    sig[v & (z < -bs)] = 1; sig[v & (z > bs)] = -1
    sig[:warmup] = 0
    return sig

def gen_s3(close, dc_cache, w, warmup):
    if w not in dc_cache:
        return None
    hc, lc = dc_cache[w]
    ph = np.empty_like(hc); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty_like(lc); pl[0] = np.nan; pl[1:] = lc[:-1]
    v = ~np.isnan(ph); n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close > ph)] = 1; sig[v & (close < pl)] = -1
    sig[:warmup] = 0
    return sig

def gen_s4(close, st_cache, ap, mult, warmup):
    key = (ap, mult)
    if key not in st_cache:
        return None
    st, _ = st_cache[key]; n = len(close)
    sp = np.empty_like(st); sp[0] = np.nan; sp[1:] = st[:-1]
    cp = np.empty_like(close); cp[0] = close[0]; cp[1:] = close[:-1]
    v = ~np.isnan(st) & ~np.isnan(sp)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close > st) & (cp <= sp)] = 1; sig[v & (close < st) & (cp >= sp)] = -1
    sig[:warmup] = 0
    return sig

def gen_s5(close, pivot_data, pt, el, warmup):
    if pt not in pivot_data:
        return None
    P, S1r, R1r, S2r, R2r, S3r, R3r = pivot_data[pt]
    if el == 2:
        S, R = S2r, R2r
    elif el == 3:
        S, R = S3r, R3r
    else:
        S, R = S1r, R1r
    n = len(close); sig = np.zeros(n, dtype=np.int8)
    sig[~np.isnan(S) & (close < S)] = 1; sig[~np.isnan(R) & (close > R)] = -1
    sig[:warmup] = 0
    return sig

def gen_s6(close, vwap_cache, vw, dm, warmup):
    if vw not in vwap_cache:
        return None
    vwap, dev = vwap_cache[vw]
    v = ~np.isnan(vwap); n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close < vwap - dm * dev)] = 1; sig[v & (close > vwap + dm * dev)] = -1
    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# CONFIRMATIONS & FILTERS
# ═══════════════════════════════════════════════════════════════

def apply_confirmation(sig, conf_type, rsi, volume, vol_sma20, is_trend):
    if conf_type == "none":
        return sig
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


def apply_filters(sig, close, volume, ind, atr_arr, day_of_week, fp, is_trend):
    if not fp:
        return sig
    n = len(sig); mask = np.ones(n, dtype=bool)
    adx = ind["adx14"]; rsi = ind["rsi14"]
    at = fp.get("adx_th", 0)
    if at and at > 0:
        if is_trend:
            mask &= (~np.isnan(adx)) & (adx > at)
        else:
            mask &= (~np.isnan(adx)) & (adx < at)
    al = fp.get("atr_lo", 0)
    if al and al > 0:
        va = atr_arr[~np.isnan(atr_arr)]
        if len(va) > 50:
            mask &= (~np.isnan(atr_arr)) & (atr_arr > np.percentile(va, al))
    vt = fp.get("vol_th", 0)
    if vt and vt > 0:
        vs = ind["vol_sma20"]
        vs_s = np.where(np.isnan(vs) | (vs < 1), 1.0, vs)
        mask &= volume / vs_s > vt
    rt = fp.get("rsi_th", 0)
    if rt and rt > 0:
        if not is_trend:
            sig[(sig == 1) & ~((~np.isnan(rsi)) & (rsi < rt))] = 0
            sig[(sig == -1) & ~((~np.isnan(rsi)) & (rsi > (100 - rt)))] = 0
        else:
            sig[(sig == 1) & ~((~np.isnan(rsi)) & (rsi > 50))] = 0
            sig[(sig == -1) & ~((~np.isnan(rsi)) & (rsi < 50))] = 0
    mn = fp.get("mom_n", 0)
    if mn and mn > 0:
        mom = np.zeros(n); mom[mn:] = close[mn:] - close[:-mn]
        if is_trend:
            sig[(sig == 1) & (mom <= 0)] = 0; sig[(sig == -1) & (mom >= 0)] = 0
        else:
            sig[(sig == 1) & (mom >= 0)] = 0; sig[(sig == -1) & (mom <= 0)] = 0
    dw = fp.get("dow", 0)
    if dw and dw > 0 and day_of_week is not None:
        if dw == 1:
            mask &= day_of_week != 0
        elif dw == 2:
            mask &= day_of_week != 4
        elif dw == 3:
            mask &= (day_of_week != 0) & (day_of_week != 4)
    sig[~mask] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# BACKTEST WRAPPER
# ═══════════════════════════════════════════════════════════════

def run_bt(sig, close, high, low, atr, rh, rl, rm, warmup, end_idx):
    return bt_universal(sig, close, high, low, atr, rh, rl,
                        rm.get("sl_p", 0), rm.get("tp_p", 0),
                        rm.get("sl_a", 0), rm.get("tp_a", 0),
                        rm.get("trail", 0), rm.get("mh", 0), rm.get("be", 0),
                        warmup, end_idx,
                        chandelier_atr=rm.get("chand", 0.0),
                        rr_ratio=rm.get("rr", 0.0))


# ═══════════════════════════════════════════════════════════════
# GRID EXPANSION (Cartesian product)
# ═══════════════════════════════════════════════════════════════

def expand_grid(d):
    if not d:
        return [{}]
    keys = sorted(d.keys())
    vals = [d[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ═══════════════════════════════════════════════════════════════
# SIGNAL PARAMETER ITERATION FROM NARROWED GRID
# ═══════════════════════════════════════════════════════════════

def iterate_signal_grid(sid, grid_sig):
    if sid == "S1":
        for ma in grid_sig.get("ma_type", ["sma"]):
            for w in grid_sig.get("window", [20]):
                for th in grid_sig.get("threshold", [1.0]):
                    yield {"ma": ma, "w": w, "thresh": th}
    elif sid == "S2":
        for ma in grid_sig.get("ma_type", ["sma"]):
            for w in grid_sig.get("window", [20]):
                for bs in grid_sig.get("bs", [2.0]):
                    yield {"ma": ma, "w": w, "bs": bs}
    elif sid == "S3":
        for w in grid_sig.get("window", [20]):
            yield {"w": w}
    elif sid == "S4":
        for ap in grid_sig.get("atr_period", [14]):
            for mult in grid_sig.get("multiplier", [2.0]):
                yield {"ap": ap, "mult": mult}
    elif sid == "S5":
        for pt in grid_sig.get("pivot_type", ["classic"]):
            for el in grid_sig.get("level", [1]):
                yield {"pt": pt, "el": el}
    elif sid == "S6":
        for vw in grid_sig.get("vwap_window", [20]):
            for dm in grid_sig.get("dev_mult", [1.0]):
                yield {"vw": vw, "dm": dm}


def gen_signal(sid, td, params):
    close = td["close"]; ind = td["ind"]
    if sid == "S1":
        z = ind["z_atr"].get((params["ma"], params["w"]))
        if z is None:
            return None
        return gen_s1(z, params["thresh"], WARMUP)
    elif sid == "S2":
        z = ind["z_std"].get((params["ma"], params["w"]))
        if z is None:
            return None
        return gen_s2(z, params["bs"], WARMUP)
    elif sid == "S3":
        return gen_s3(close, td["dc_cache"], params["w"], WARMUP)
    elif sid == "S4":
        return gen_s4(close, td["st_cache"], params["ap"], params["mult"], WARMUP)
    elif sid == "S5":
        return gen_s5(close, td["pivot_data"], params["pt"], params["el"], WARMUP)
    elif sid == "S6":
        return gen_s6(close, td["vwap_cache"], params["vw"], params["dm"], WARMUP)
    return None


def iterate_rm_grid(rm_dict):
    """Expand RM dict from narrowed grid to list of RM param dicts."""
    keys = sorted(rm_dict.keys())
    vals = [rm_dict[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))


def iterate_filter_grid(filter_dict, n_random=5):
    """Generate deterministic filter combinations from narrowed grid.
    If grid is empty, yield empty dict."""
    if not filter_dict:
        yield {}
        return
    keys = sorted(filter_dict.keys())
    vals = [filter_dict[k] for k in keys]
    combos = list(product(*vals))
    # Limit to avoid explosion - take every nth to get ~n_random+1
    step = max(1, len(combos) // (n_random + 1))
    yield {}  # always include no-filter
    for i in range(0, len(combos), step):
        fp = {}
        for k, v in zip(keys, combos[i]):
            if v is not None:
                fp[k] = v
        if fp:
            yield fp


# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION PER TICKER
# ═══════════════════════════════════════════════════════════════

def prepare_ticker_data(ticker, daily_df, kr_s, vpred_df):
    """Prepare all data structures for one ticker."""
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP + 200:
        return None
    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)

    # Log returns: lr[t] = log(close[t+1] / close[t]) for position held at t
    lr = np.zeros(n)
    lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    cal_days = np.ones(n)
    for i in range(1, n):
        cal_days[i] = (dates[i] - dates[i - 1]).days

    rf_raw = kr_s.reindex(dates, method="ffill").values
    rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0

    ind = compute_base(close, high, low, open_arr, volume, is_hourly=False)
    # Add RSI (not in compute_base)
    if "rsi14" not in ind:
        ind["rsi14"] = calc_rsi(close, 14)
    atr = ind["atr14"]; rh = ind.get("rh", ind.get("rh_10", np.zeros(n))); rl = ind.get("rl", ind.get("rl_10", np.zeros(n)))
    day_of_week = dates.dayofweek.values

    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS)
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}

    # Vpred sigma
    sigma_dict = {}
    if vpred_df is not None:
        vp = vpred_df[vpred_df["ticker"] == ticker].sort_values("date")
        if len(vp) > 0:
            vp_dates = pd.to_datetime(vp["date"].values)
            for hname in HORIZONS:
                col = f"sigma_{hname}"
                if col not in vp.columns:
                    continue
                vp_vals = vp[col].values
                sigma_arr = np.full(n, np.nan)
                bar_dates = pd.to_datetime(tdf["date"].values)
                idx = np.searchsorted(vp_dates, bar_dates, side="right") - 1
                valid = idx >= 0
                sigma_arr[valid] = vp_vals[idx[valid]]
                sigma_dict[hname] = sigma_arr

    return dict(
        ticker=ticker, close=close, high=high, low=low, open_arr=open_arr,
        volume=volume, dates=dates, n=n, lr=lr, cal_days=cal_days, rf_d=rf_d,
        ind=ind, atr=atr, rh=rh, rl=rl, day_of_week=day_of_week,
        dc_cache=dc_cache, st_cache=st_cache, vwap_cache=vwap_cache,
        pivot_data=pivot_data, sigma_dict=sigma_dict,
    )


# ═══════════════════════════════════════════════════════════════
# APPROACH A: WALK-FORWARD GRID SEARCH + TOP-10 ENSEMBLE
# ═══════════════════════════════════════════════════════════════

def _build_narrow_grid(best_cfg):
    """Build narrow grid: best config ± 1-2 steps per parameter."""
    sig = json.loads(best_cfg["signal"])
    rm = json.loads(best_cfg["rm"])
    conf = best_cfg.get("conf", "none")

    # Signal variations: ±3 steps for numeric params (wider grid)
    sig_variants = [sig]  # always include best
    for key, val in sig.items():
        if isinstance(val, (int, float)) and key not in ("ma",):
            for mult in [0.5, 0.7, 0.85, 1.15, 1.3, 1.5, 2.0]:
                v2 = sig.copy()
                v2[key] = round(val * mult, 2) if isinstance(val, float) else max(1, int(val * mult))
                if v2 != sig:
                    sig_variants.append(v2)
    # Deduplicate
    sig_variants = [dict(t) for t in {tuple(sorted(d.items())) for d in sig_variants}]

    # RM variations: wider
    rm_variants = [rm]
    for key, val in rm.items():
        if val != 0 and isinstance(val, (int, float)):
            for mult in [0.5, 0.7, 0.85, 1.15, 1.3, 1.5, 2.0]:
                r2 = rm.copy()
                r2[key] = round(val * mult, 2) if isinstance(val, float) else max(1, int(val * mult))
                if r2 != rm:
                    rm_variants.append(r2)
    rm_variants = [dict(t) for t in {tuple(sorted(d.items())) for d in rm_variants}]
    # Also add default RM
    rm_variants.append({"sl_a": 2.0, "tp_a": 3.0, "mh": 15})
    rm_variants.append({"sl_a": 2.0, "trail": 2.0, "mh": 0, "be": 1.0})
    rm_variants.append({"sl_a": 3.0, "tp_a": 4.5, "mh": 20})
    rm_variants.append({"sl_a": 1.5, "trail": 3.0, "mh": 0, "be": 0.5})

    confs = list(set(["none", conf, "rsi30", "volume"]))  # wider confirmation set
    return sig_variants, rm_variants, confs


def walkforward_A(ticker_data_dict, grid, comm):
    """Walk-forward A: full grid + 2-pass search (v4 approach).
    Pass 1: coarse RM sample -> find top-30 signal combos
    Pass 2: full RM grid x top signals -> top-N ensemble
    """
    log(f"=== Approach A (comm={comm}) — FULL GRID 2-PASS ===")
    results_rows = []
    positions_dict = {}
    grid_log = {}
    n_bt_total = 0
    ann = np.sqrt(252)

    n_total_cells = len(STRATEGIES) * len(TICKERS) * len(TEST_YEARS)
    tm = TimeManager("stage3_A", n_total_cells, max_seconds=3600)

    for sid in STRATEGIES:
        cat = CATEGORY[sid]
        sig_grid = expand_grid(SIGNAL_GRIDS[sid])
        if cat == "Trend":
            rm_grid_v4 = TREND_RM_GRID
        else:
            rm_grid_v4 = expand_grid(RM_GRIDS_V3[(cat, "daily")])
        coarse_step = max(1, len(rm_grid_v4) // 72)
        n_total = len(sig_grid) * len(rm_grid_v4)
        log(f"  {sid} ({cat}): {len(sig_grid)} sig x {len(rm_grid_v4)} rm = {n_total:,}")

        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                continue
            close = td["close"]; high = td["high"]; low = td["low"]
            lr = td["lr"]; rf_d = td["rf_d"]; cal_days = td["cal_days"]
            dates = td["dates"]; n = td["n"]
            volume = td["volume"]
            open_arr = td.get("open", close)
            ind_td = td["ind"]

            # Precompute caches for this ticker
            _sg = SIGNAL_GRIDS
            sma_windows = sorted(set(
                _sg["S1"].get("ma_window", []) + _sg["S2"].get("bb_window", [])))
            sma_cache = precompute_sma_cache(close, sma_windows)
            dc_cache = precompute_donchian_cache(
                high, low, _sg["S3"].get("dc_window", []))
            st_cache = precompute_supertrend_cache(
                high, low, close,
                _sg["S4"].get("atr_period", []),
                _sg["S4"].get("multiplier", []))
            vwap_cache = precompute_vwap_cache(
                close, high, low, volume,
                _sg["S6"].get("vwap_window", []))
            pivot_data = {}
            for pv in ["classic", "camarilla", "fibonacci"]:
                pivot_data[pv] = calc_pivot_daily(high, low, close, pv)
            _pctile_cache.clear()

            # Cache all signals for this strategy
            cached_sigs = {}
            for sp in sig_grid:
                key = tuple(sorted(sp.items()))
                sig_arr, exit_info = dispatch_signals(
                    sid, close, high, low, volume, ind_td,
                    sma_cache, dc_cache, st_cache, vwap_cache,
                    pivot_data, None, sp, is_hourly=False)
                cached_sigs[key] = (sig_arr, exit_info)

            for test_year in TEST_YEARS:
                t0_y = time.time()
                train_end = np.searchsorted(
                    dates, pd.Timestamp(f"{test_year}-01-01"))
                test_start = train_end
                test_end = np.searchsorted(
                    dates, pd.Timestamp(f"{test_year + 1}-01-01"))
                if train_end <= WARMUP or test_end <= test_start:
                    continue

                train_mask = np.zeros(n, dtype=bool)
                train_mask[WARMUP:train_end] = True
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True

                # Pass 1: coarse RM
                top_combos = []
                n_bt_year = 0
                for sp in sig_grid:
                    sp_key = tuple(sorted(sp.items()))
                    sig_arr, exit_info = cached_sigs[sp_key]
                    for rm_idx in range(0, len(rm_grid_v4), coarse_step):
                        rm_c = rm_grid_v4[rm_idx]
                        pos, _ = dispatch_bt_atr_v3(
                            sid, sig_arr, exit_info, close, high, low,
                            ind_td, rm_c, WARMUP, train_end, 0)
                        sh = calc_sharpe_comm(pos, lr, train_mask, ann)
                        n_bt_year += 1
                        top_combos.append((sh, sp_key, rm_idx))

                top_combos.sort(key=lambda x: x[0], reverse=True)
                seen_sp = set()
                top_sp_keys = []
                for sh, sp_key, _ in top_combos:
                    if sp_key not in seen_sp:
                        seen_sp.add(sp_key)
                        top_sp_keys.append(sp_key)
                        if len(top_sp_keys) >= 30:
                            break

                # Pass 2: fine RM for top signals
                all_results = []
                for sp_key in top_sp_keys:
                    sig_arr, exit_info = cached_sigs[sp_key]
                    for rm_idx, rm_c in enumerate(rm_grid_v4):
                        pos, _ = dispatch_bt_atr_v3(
                            sid, sig_arr, exit_info, close, high, low,
                            ind_td, rm_c, WARMUP, train_end, 0)
                        sh = calc_sharpe_comm(pos, lr, train_mask, ann)
                        n_bt_year += 1
                        all_results.append((sh, sp_key, rm_idx))

                all_results.sort(key=lambda x: x[0], reverse=True)
                top_n = all_results[:ENSEMBLE_TOP_N]
                n_bt_total += n_bt_year

                if not top_n or top_n[0][0] < TRAIN_SHARPE_THRESHOLD:
                    pos_ensemble = np.zeros(n)
                    positions_dict[(sid, ticker, test_year)] = pos_ensemble
                    sh_test, info = calc_excess_sharpe(
                        pos_ensemble, lr, rf_d, cal_days, test_mask, comm)
                    results_rows.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe=round(sh_test, 4), approach="A",
                        train_sharpe=round(
                            top_n[0][0] if top_n else -999, 4),
                        exposure=0, tpy=0, wr=0, n_trades=0,
                        comm=comm, skipped=True))
                    continue

                # Ensemble: continuous averaging (like v4)
                pos_ensemble = np.zeros(n, dtype=np.float64)
                for k, (_, sp_key_c, rm_idx_c) in enumerate(top_n):
                    sig_arr, exit_info = cached_sigs[sp_key_c]
                    rm_c = rm_grid_v4[rm_idx_c]
                    pos, _ = dispatch_bt_atr_v3(
                        sid, sig_arr, exit_info, close, high, low,
                        ind_td, rm_c, WARMUP, test_end, 0)
                    pos_ensemble += np.sign(pos)
                pos_ensemble = pos_ensemble / float(len(top_n))

                positions_dict[(sid, ticker, test_year)] = pos_ensemble
                sh_test, info = calc_excess_sharpe(
                    pos_ensemble, lr, rf_d, cal_days, test_mask, comm)
                results_rows.append(dict(
                    strategy=sid, ticker=ticker, year=test_year,
                    sharpe=round(sh_test, 4), approach="A",
                    train_sharpe=round(top_n[0][0], 4),
                    exposure=info.get("exposure", 0),
                    tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                    n_trades=info.get("n_trades", 0),
                    comm=comm, skipped=False))

                elapsed = time.time() - t0_y
                log(f"    {sid} {ticker} {test_year}: "
                    f"trainSh={top_n[0][0]:+.2f} testSh={sh_test:+.2f} "
                    f"exp={info.get('exposure',0):.0f}% "
                    f"bt={n_bt_year:,} {elapsed:.1f}s")

                quick_check(sid, ticker, test_year, sh_test,
                            info.get("exposure", 0))
                if not tm.tick():
                    log("  ABORT: TimeManager limit reached")
                    break
            else:
                continue
            break

        grid_log[sid] = {"n_sig": len(sig_grid),
                         "n_rm": len(rm_grid_v4)}

    log(f"  A total BT calls: {n_bt_total:,}")
    return results_rows, positions_dict, grid_log


def walkforward_A_DISABLED(ticker_data_dict, grid, comm):
    """DISABLED: old narrow grid approach.
    ~20-50 combos per ticker (NOT millions).
    """
    log(f"=== Approach A (comm={comm}) — NARROW grid from best configs ===")
    results_rows = []
    positions_dict = {}
    grid_log = {}
    n_bt_total = 0

    # Load best configs from test phase (comm-specific)
    comm_tag = f"_{round(comm * 10000):03d}" if comm != 0.0005 else ""
    best_configs_path = BASE / f"results/final/strategies/test_phases/FINAL_best_configs{comm_tag}.json"
    if not best_configs_path.exists():
        best_configs_path = BASE / "results/final/strategies/test_phases/FINAL_best_configs.json"
        log(f"  Using default best_configs (no {comm_tag} file)")
    with open(best_configs_path) as f:
        best_configs = json.load(f)
    log(f"  Loaded {best_configs_path.name}")

    # TimeManager: abort if too slow
    n_total_cells = len(STRATEGIES) * len(TICKERS) * len(TEST_YEARS)
    tm = TimeManager("stage3_A", n_total_cells, max_seconds=3600)

    for sid in STRATEGIES:
        is_trend = IS_TREND[sid]

        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                continue
            close = td["close"]; high = td["high"]; low = td["low"]
            atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
            ind = td["ind"]; volume = td["volume"]; dates = td["dates"]
            lr = td["lr"]; rf_d = td["rf_d"]; cal_days = td["cal_days"]
            n = td["n"]; day_of_week = td["day_of_week"]
            rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]

            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None:
                continue

            sig_variants, rm_variants, confs = _build_narrow_grid(cfg)
            n_combos = len(sig_variants) * len(confs) * len(rm_variants)

            # Sanity check on first ticker
            if ticker == TICKERS[0]:
                log(f"  {sid}: {len(sig_variants)} sig × {len(confs)} conf × {len(rm_variants)} rm = {n_combos} combos/ticker")
                if not agent_sanity_check(f"stage3_A_{sid}", {
                    'n_combos_per_ticker': n_combos, 'n_signal': len(sig_variants),
                    'n_rm': len(rm_variants), 'n_conf': len(confs), 'w': 1.0}):
                    # Auto-shrink: reduce to ×2 variations instead of ×4
                    sig_variants = sig_variants[:max(5, len(sig_variants) // 2)]
                    rm_variants = rm_variants[:max(5, len(rm_variants) // 2)]
                    n_combos = len(sig_variants) * len(confs) * len(rm_variants)
                    log(f"  Auto-shrunk to {n_combos} combos/ticker")

            for test_year in TEST_YEARS:
                t0_y = time.time()
                train_end = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
                test_start = train_end
                test_end_ts = pd.Timestamp(f"{test_year + 1}-01-01")
                test_end = np.searchsorted(dates, test_end_ts)
                if train_end <= WARMUP or test_end <= test_start:
                    continue

                train_mask = np.zeros(n, dtype=bool)
                train_mask[WARMUP:train_end] = True
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True

                all_results_train = []
                n_bt_year = 0

                for sig_params in sig_variants:
                    sig_raw = gen_signal(sid, td, sig_params)
                    if sig_raw is None:
                        continue

                    for conf in confs:
                        sig_conf = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
                        if np.sum(np.abs(sig_conf[train_mask]) > 0) < 3:
                            continue

                        for rm in rm_variants:
                            pos = run_bt(sig_conf, close, high, low, atr, rh, rl, rm, WARMUP, train_end)
                            sh = trading_sharpe(pos, lr, train_mask, comm)
                            n_bt_year += 1
                            if sh > -900:
                                all_results_train.append((sh, sig_params, conf, rm))

                n_bt_total += n_bt_year

                # Rank by training sharpe, take top-N
                all_results_train.sort(key=lambda x: x[0], reverse=True)
                top10 = all_results_train[:ENSEMBLE_TOP_N]

                if not top10 or top10[0][0] < TRAIN_SHARPE_THRESHOLD:
                    pos_ensemble = np.zeros(n)
                    positions_dict[(sid, ticker, test_year)] = pos_ensemble
                    sh_test, info = calc_excess_sharpe(pos_ensemble, lr, rf_d, cal_days, test_mask, comm)
                    results_rows.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe=round(sh_test, 4), approach="A",
                        train_sharpe=round(top10[0][0] if top10 else -999, 4),
                        exposure=info.get("exposure", 0),
                        tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                        n_trades=info.get("n_trades", 0),
                        comm=comm, skipped=True,
                    ))
                    continue

                # Ensemble: majority vote of top-10 positions on TEST
                test_positions = []
                for sh_tr, sig_params, conf, rm in top10:
                    sig_raw = gen_signal(sid, td, sig_params)
                    if sig_raw is None:
                        continue
                    sig_conf = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
                    pos = run_bt(sig_conf, close, high, low, atr, rh, rl, rm, WARMUP, test_end)
                    test_positions.append(pos)

                if not test_positions:
                    pos_ensemble = np.zeros(n)
                else:
                    # Majority vote: sign(mean(positions))
                    pos_stack = np.array(test_positions)
                    pos_mean = np.mean(pos_stack, axis=0)
                    # Majority vote: if >50% agree on direction, take that direction
                    n_top = len(test_positions)
                    long_votes = np.sum(pos_stack > 0.01, axis=0)
                    short_votes = np.sum(pos_stack < -0.01, axis=0)
                    threshold = n_top / 2.0
                    pos_ensemble = np.zeros(n)
                    pos_ensemble[long_votes > threshold] = 1.0
                    pos_ensemble[short_votes > threshold] = -1.0

                positions_dict[(sid, ticker, test_year)] = pos_ensemble

                # Evaluate on test
                sh_test, info = calc_excess_sharpe(pos_ensemble, lr, rf_d, cal_days, test_mask, comm)
                results_rows.append(dict(
                    strategy=sid, ticker=ticker, year=test_year,
                    sharpe=round(sh_test, 4), approach="A",
                    train_sharpe=round(top10[0][0], 4),
                    exposure=info.get("exposure", 0),
                    tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                    n_trades=info.get("n_trades", 0),
                    comm=comm, skipped=False,
                ))

                elapsed = time.time() - t0_y
                n_skip = "SKIP" if (top10 and top10[0][0] < TRAIN_SHARPE_THRESHOLD) else ""
                log(f"    {sid} {ticker} {test_year}: trainSh={top10[0][0]:+.2f} testSh={sh_test:+.2f} "
                    f"exp={info.get('exposure',0):.0f}% bt={n_bt_year:,} {elapsed:.1f}s {n_skip}")

                # Quick check + TimeManager
                quick_check(sid, ticker, test_year, sh_test, info.get("exposure", 0))
                if not tm.tick():
                    log(f"  ABORT: TimeManager limit reached")
                    break
            else:
                continue
            break  # break ticker loop if test_year loop was aborted

        grid_log[sid] = {"n_combos": n_combos}

    log(f"  A total BT calls: {n_bt_total:,}")
    return results_rows, positions_dict, grid_log


# ═══════════════════════════════════════════════════════════════
# APPROACH B: ADAPTIVE SIGMA STOPS (OVERLAY ON A)
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _apply_B_numba(a_pos, close, vol_pred, vol_median, k_sl, ratio, base_hold, hold_gamma, be_trigger):
    """Adaptive stops using vpred. Overlay on A positions.
    When A enters: set vpred-based SL/TP.
    SL = entry * (1 - k_sl * vpred), TP = entry * (1 + k_sl * ratio * vpred)
    Dynamic hold = base_hold * (vol_median / vol_pred)^hold_gamma
    Breakeven trigger at be_trigger * vpred * entry_price
    """
    n = len(a_pos)
    out = np.zeros(n)
    in_trade = False
    entry_price = 0.0
    direction = 0.0
    sl = 0.0; tp = 0.0
    held = 0; max_hold = base_hold
    be_active = False
    b_exited = False  # True when B closed this A-trade via SL/TP/maxhold

    for t in range(n):
        # Reset b_exited when A exits (new A-trade can re-enter)
        if abs(a_pos[t]) < 0.01:
            b_exited = False

        if in_trade:
            held += 1
            # Breakeven
            if be_trigger > 0 and not be_active:
                sv = vol_pred[t]
                if sv != sv:
                    sv = 0.02
                if direction > 0 and (close[t] - entry_price) > be_trigger * sv * entry_price:
                    be_active = True
                    sl = entry_price
                elif direction < 0 and (entry_price - close[t]) > be_trigger * sv * entry_price:
                    be_active = True
                    sl = entry_price

            closed = False
            if direction > 0:
                if close[t] <= sl:
                    closed = True
                elif tp > 0 and close[t] >= tp:
                    closed = True
            else:
                if close[t] >= sl:
                    closed = True
                elif tp > 0 and close[t] <= tp:
                    closed = True

            if max_hold > 0 and held >= max_hold:
                closed = True

            # Check if A exits
            if abs(a_pos[t]) < 0.01:
                closed = True

            if closed:
                in_trade = False
                held = 0
                be_active = False
                # Mark that B exited this A-trade (don't re-enter until A starts new trade)
                if abs(a_pos[t]) > 0.01:
                    b_exited = True
            else:
                out[t] = direction

        if not in_trade and not b_exited and abs(a_pos[t]) > 0.01:
            # A enters a new trade
            sv = vol_pred[t]
            if sv != sv or sv < 1e-8:
                sv = 0.02
            direction = 1.0 if a_pos[t] > 0 else -1.0
            entry_price = close[t]
            in_trade = True
            held = 0
            be_active = False
            if direction > 0:
                sl = entry_price * (1.0 - k_sl * sv)
                tp = entry_price * (1.0 + k_sl * ratio * sv)
            else:
                sl = entry_price * (1.0 + k_sl * sv)
                tp = entry_price * (1.0 - k_sl * ratio * sv)
            # Dynamic hold
            if hold_gamma > 0 and sv > 1e-8 and vol_median > 1e-8:
                mh = base_hold * (vol_median / sv) ** hold_gamma
                max_hold = int(min(max(mh, 3.0), base_hold * 3.0))
            else:
                max_hold = base_hold
            out[t] = direction

    return out


def walkforward_B(ticker_data_dict, a_positions, comm):
    """Walk-forward B: for each test_year, tune B params on train using A positions."""
    log(f"=== Approach B (comm={comm}) ===")
    results_rows = []
    positions_dict = {}
    b_grid_combos = expand_grid(B_GRID)
    n_combos = len(b_grid_combos) * len(HORIZONS)
    log(f"  B grid: {len(b_grid_combos)} base x {len(HORIZONS)} horizons = {n_combos} combos")

    for sid in STRATEGIES:
        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                continue
            close = td["close"]; lr = td["lr"]; rf_d = td["rf_d"]
            cal_days = td["cal_days"]; dates = td["dates"]; n = td["n"]
            sigma_dict = td["sigma_dict"]

            for test_year in TEST_YEARS:
                t0 = time.time()
                # Get A position for this test_year
                a_pos = a_positions.get((sid, ticker, test_year))
                if a_pos is None:
                    continue

                train_end = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
                test_start = train_end
                test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
                if train_end <= WARMUP or test_end <= test_start:
                    continue

                train_mask = np.zeros(n, dtype=bool)
                train_mask[WARMUP:train_end] = True
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True

                best_sh_train = -999.0
                best_params = None

                for horizon in HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None:
                        continue
                    sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                    vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02

                    for bp in b_grid_combos:
                        b_pos = _apply_B_numba(
                            a_pos, close, sigma, vol_median,
                            bp["k_sl"], bp["ratio"], bp["base_hold"],
                            bp["hold_gamma"], bp["be_trigger"]
                        )
                        sh = trading_sharpe(b_pos, lr, train_mask, comm)
                        if sh > best_sh_train:
                            best_sh_train = sh
                            best_params = dict(bp, horizon=horizon)

                if best_params is None:
                    positions_dict[(sid, ticker, test_year)] = a_pos.copy()
                    sh_test, info = calc_excess_sharpe(a_pos, lr, rf_d, cal_days, test_mask, comm)
                    results_rows.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe=round(sh_test, 4), approach="B",
                        train_sharpe=round(best_sh_train, 4),
                        exposure=info.get("exposure", 0),
                        tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                        n_trades=info.get("n_trades", 0),
                        comm=comm, params=str(best_params),
                    ))
                    continue

                # Apply best B params on test
                horizon = best_params["horizon"]
                sigma = sigma_dict.get(horizon, np.full(n, 0.02))
                sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02
                b_pos_test = _apply_B_numba(
                    a_pos, close, sigma, vol_median,
                    best_params["k_sl"], best_params["ratio"],
                    best_params["base_hold"], best_params["hold_gamma"],
                    best_params["be_trigger"]
                )
                positions_dict[(sid, ticker, test_year)] = b_pos_test

                sh_test, info = calc_excess_sharpe(b_pos_test, lr, rf_d, cal_days, test_mask, comm)
                results_rows.append(dict(
                    strategy=sid, ticker=ticker, year=test_year,
                    sharpe=round(sh_test, 4), approach="B",
                    train_sharpe=round(best_sh_train, 4),
                    exposure=info.get("exposure", 0),
                    tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                    n_trades=info.get("n_trades", 0),
                    comm=comm, params=str(best_params),
                ))
                elapsed = time.time() - t0
                log(f"    B {sid} {ticker} {test_year}: trainSh={best_sh_train:+.2f} testSh={sh_test:+.2f} "
                    f"h={best_params['horizon']} {elapsed:.1f}s")

    return results_rows, positions_dict


# ═══════════════════════════════════════════════════════════════
# APPROACH C: REGIME FILTER (OVERLAY ON A)
# ═══════════════════════════════════════════════════════════════

def apply_C(a_pos, vol_pred, threshold, lookback, strategy_type, direction,
            sigma_h22, sigma_h1, term_str, hysteresis):
    """Regime filter: mask A positions based on vol regime.
    - trend: trade only in HIGH vol regime (pctrank > threshold)
    - contrarian: trade only in LOW vol regime (pctrank < 1-threshold)
    - range: trade in MID vol regime (band)
    """
    n = len(a_pos)
    pctrank = pd.Series(vol_pred).rolling(lookback, min_periods=max(50, lookback // 5)).rank(pct=True).values

    if strategy_type == "trend":
        if hysteresis > 0:
            exit_thresh = threshold - hysteresis / 100.0
            mask = np.zeros(n, dtype=bool)
            active = False
            for i in range(n):
                v = pctrank[i]
                if np.isnan(v):
                    mask[i] = False
                    continue
                if not active and v > threshold:
                    active = True
                elif active and v < exit_thresh:
                    active = False
                mask[i] = active
        else:
            mask = ~np.isnan(pctrank) & (pctrank > threshold)
    elif strategy_type == "contrarian":
        if hysteresis > 0:
            exit_thresh = threshold - hysteresis / 100.0
            mask = np.zeros(n, dtype=bool)
            active = False
            for i in range(n):
                v = pctrank[i]
                if np.isnan(v):
                    mask[i] = False
                    continue
                if not active and v < (1.0 - threshold):
                    active = True
                elif active and v > (1.0 - exit_thresh):
                    active = False
                mask[i] = active
        else:
            mask = ~np.isnan(pctrank) & (pctrank < (1 - threshold))
    else:
        # range: use bands
        lo_band = max(0.5 - threshold / 2, 0.1)
        hi_band = min(0.5 + threshold / 2, 0.9)
        mask = ~np.isnan(pctrank) & (pctrank >= lo_band) & (pctrank <= hi_band)

    # Direction filter
    if direction == "rise":
        dw = 5
        delta = np.full(n, np.nan)
        delta[dw:] = vol_pred[dw:] - vol_pred[:-dw]
        mask = mask & (~np.isnan(delta)) & (delta > 0)
    elif direction == "fall":
        dw = 5
        delta = np.full(n, np.nan)
        delta[dw:] = vol_pred[dw:] - vol_pred[:-dw]
        mask = mask & (~np.isnan(delta)) & (delta < 0)

    # Term structure filter
    if term_str and sigma_h22 is not None and sigma_h1 is not None:
        s_h1_safe = np.maximum(sigma_h1, 1e-8)
        ratio = sigma_h22 / s_h1_safe
        valid_r = ~np.isnan(ratio)
        if strategy_type == "trend":
            mask = mask & valid_r & (ratio > 1.0)
        elif strategy_type == "contrarian":
            mask = mask & valid_r & (ratio < 1.0)

    out = a_pos.copy()
    out[~mask] = 0.0
    return out


def walkforward_C(ticker_data_dict, a_positions, comm):
    """Walk-forward C: for each test_year, tune C params on train."""
    log(f"=== Approach C (comm={comm}) ===")
    results_rows = []
    positions_dict = {}
    c_grid_combos = expand_grid(C_GRID)
    n_combos = len(c_grid_combos) * len(HORIZONS)
    log(f"  C grid: {len(c_grid_combos)} base x {len(HORIZONS)} horizons = {n_combos} combos")

    for sid in STRATEGIES:
        stype = STRATEGY_TYPE[sid]
        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                continue
            close = td["close"]; lr = td["lr"]; rf_d = td["rf_d"]
            cal_days = td["cal_days"]; dates = td["dates"]; n = td["n"]
            sigma_dict = td["sigma_dict"]

            for test_year in TEST_YEARS:
                t0 = time.time()
                a_pos = a_positions.get((sid, ticker, test_year))
                if a_pos is None:
                    continue

                train_end = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
                test_start = train_end
                test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
                if train_end <= WARMUP or test_end <= test_start:
                    continue

                train_mask = np.zeros(n, dtype=bool)
                train_mask[WARMUP:train_end] = True
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True

                best_sh_train = -999.0
                best_params = None

                for horizon in HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None:
                        continue
                    sigma_h22 = sigma_dict.get("h22")
                    sigma_h1 = sigma_dict.get("h1")

                    for cp in c_grid_combos:
                        c_pos = apply_C(
                            a_pos, sigma, cp["threshold"], cp["lookback"],
                            stype, cp["direction"],
                            sigma_h22, sigma_h1, cp["term_str"], cp["hysteresis"]
                        )
                        sh = trading_sharpe(c_pos, lr, train_mask, comm)
                        if sh > best_sh_train:
                            # Check min exposure
                            pm_train = c_pos[train_mask]
                            exp_pct = np.mean(np.abs(pm_train)) * 100
                            if exp_pct >= 3.0:
                                best_sh_train = sh
                                best_params = dict(cp, horizon=horizon)

                if best_params is None:
                    # Fallback: use A positions
                    positions_dict[(sid, ticker, test_year)] = a_pos.copy()
                    sh_test, info = calc_excess_sharpe(a_pos, lr, rf_d, cal_days, test_mask, comm)
                    results_rows.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe=round(sh_test, 4), approach="C",
                        train_sharpe=round(best_sh_train, 4),
                        exposure=info.get("exposure", 0),
                        tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                        n_trades=info.get("n_trades", 0),
                        comm=comm, params="fallback_to_A",
                    ))
                    continue

                # Apply best C params on test
                horizon = best_params["horizon"]
                sigma = sigma_dict.get(horizon, np.full(n, 0.02))
                sigma_h22 = sigma_dict.get("h22")
                sigma_h1 = sigma_dict.get("h1")
                c_pos_test = apply_C(
                    a_pos, sigma, best_params["threshold"], best_params["lookback"],
                    stype, best_params["direction"],
                    sigma_h22, sigma_h1, best_params["term_str"], best_params["hysteresis"]
                )
                positions_dict[(sid, ticker, test_year)] = c_pos_test

                sh_test, info = calc_excess_sharpe(c_pos_test, lr, rf_d, cal_days, test_mask, comm)
                results_rows.append(dict(
                    strategy=sid, ticker=ticker, year=test_year,
                    sharpe=round(sh_test, 4), approach="C",
                    train_sharpe=round(best_sh_train, 4),
                    exposure=info.get("exposure", 0),
                    tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                    n_trades=info.get("n_trades", 0),
                    comm=comm, params=str(best_params),
                ))
                elapsed = time.time() - t0
                log(f"    C {sid} {ticker} {test_year}: trainSh={best_sh_train:+.2f} testSh={sh_test:+.2f} "
                    f"h={best_params['horizon']} {elapsed:.1f}s")

    return results_rows, positions_dict


# ═══════════════════════════════════════════════════════════════
# APPROACH D: VOL-TARGETING (OVERLAY ON A)
# ═══════════════════════════════════════════════════════════════

def apply_D(a_pos, vol_pred, sigma_target, gamma, smooth_span, vol_floor, vol_cap):
    """Vol-targeting: scale A positions by (sigma_target / vpred)^gamma.
    Clipped to [-1, 1] since w=1.0 always.
    """
    sigma = vol_pred.copy()
    if smooth_span > 0:
        sigma = pd.Series(sigma).ewm(span=smooth_span, min_periods=1).mean().values
    sigma = np.where(np.isnan(sigma) | (sigma < 1e-6), 1e-6, sigma)
    if vol_floor > 0:
        sigma = np.maximum(sigma, vol_floor)
    if vol_cap > 0:
        sigma = np.minimum(sigma, vol_cap)

    scale = (sigma_target / sigma) ** gamma
    scale = np.minimum(scale, 1.0)  # w=1.0 always, no leverage
    scale = np.maximum(scale, 0.0)  # no negative scaling

    out = a_pos * scale
    return out


def walkforward_D(ticker_data_dict, a_positions, comm):
    """Walk-forward D: for each test_year, tune D params on train."""
    log(f"=== Approach D (comm={comm}) ===")
    results_rows = []
    positions_dict = {}
    d_grid_combos = expand_grid(D_GRID)
    n_combos = len(d_grid_combos) * len(HORIZONS)
    log(f"  D grid: {len(d_grid_combos)} base x {len(HORIZONS)} horizons = {n_combos} combos")

    for sid in STRATEGIES:
        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                continue
            close = td["close"]; lr = td["lr"]; rf_d = td["rf_d"]
            cal_days = td["cal_days"]; dates = td["dates"]; n = td["n"]
            sigma_dict = td["sigma_dict"]

            for test_year in TEST_YEARS:
                t0 = time.time()
                a_pos = a_positions.get((sid, ticker, test_year))
                if a_pos is None:
                    continue

                train_end = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
                test_start = train_end
                test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
                if train_end <= WARMUP or test_end <= test_start:
                    continue

                train_mask = np.zeros(n, dtype=bool)
                train_mask[WARMUP:train_end] = True
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True

                best_sh_train = -999.0
                best_params = None

                for horizon in HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None:
                        continue

                    for dp in d_grid_combos:
                        d_pos = apply_D(
                            a_pos, sigma, dp["sigma_target"], dp["gamma"],
                            dp["smooth"], dp["vol_floor"], dp["vol_cap"]
                        )
                        sh = trading_sharpe(d_pos, lr, train_mask, comm)
                        if sh > best_sh_train:
                            best_sh_train = sh
                            best_params = dict(dp, horizon=horizon)

                if best_params is None:
                    positions_dict[(sid, ticker, test_year)] = a_pos.copy()
                    sh_test, info = calc_excess_sharpe(a_pos, lr, rf_d, cal_days, test_mask, comm)
                    results_rows.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe=round(sh_test, 4), approach="D",
                        train_sharpe=round(best_sh_train, 4),
                        exposure=info.get("exposure", 0),
                        tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                        n_trades=info.get("n_trades", 0),
                        comm=comm, params="fallback_to_A",
                    ))
                    continue

                # Apply best D params on test
                horizon = best_params["horizon"]
                sigma = sigma_dict.get(horizon, np.full(n, 0.02))
                d_pos_test = apply_D(
                    a_pos, sigma, best_params["sigma_target"], best_params["gamma"],
                    best_params["smooth"], best_params["vol_floor"], best_params["vol_cap"]
                )
                positions_dict[(sid, ticker, test_year)] = d_pos_test

                sh_test, info = calc_excess_sharpe(d_pos_test, lr, rf_d, cal_days, test_mask, comm)
                results_rows.append(dict(
                    strategy=sid, ticker=ticker, year=test_year,
                    sharpe=round(sh_test, 4), approach="D",
                    train_sharpe=round(best_sh_train, 4),
                    exposure=info.get("exposure", 0),
                    tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                    n_trades=info.get("n_trades", 0),
                    comm=comm, params=str(best_params),
                ))
                elapsed = time.time() - t0
                log(f"    D {sid} {ticker} {test_year}: trainSh={best_sh_train:+.2f} testSh={sh_test:+.2f} "
                    f"h={best_params['horizon']} {elapsed:.1f}s")

    return results_rows, positions_dict


# ═══════════════════════════════════════════════════════════════
# SAVE POSITIONS AS PARQUET
# ═══════════════════════════════════════════════════════════════

def save_positions_parquet(positions_dict, ticker_data_dict, approach, suffix=""):
    """Save positions as parquet. Columns: date, ticker, strategy, year, position."""
    rows = []
    for (sid, ticker, year), pos_arr in positions_dict.items():
        td = ticker_data_dict.get(ticker)
        if td is None:
            continue
        dates = td["dates"]; n = td["n"]
        lr_t = td["lr"]
        test_start = np.searchsorted(dates, pd.Timestamp(f"{year}-01-01"))
        test_end = np.searchsorted(dates, pd.Timestamp(f"{year + 1}-01-01"))
        for t in range(test_start, min(test_end, n)):
            if abs(pos_arr[t]) > 1e-8:
                rows.append(dict(
                    date=dates[t], ticker=ticker, strategy=sid,
                    year=year, position=round(float(pos_arr[t]), 6),
                    daily_gross_return=round(float(pos_arr[t] * lr_t[t]), 8),
                ))
    if rows:
        df = pd.DataFrame(rows)
        fname = f"wf_positions_{approach}{suffix}.parquet"
        fpath = OUT_DIR / fname
        df.to_parquet(fpath, index=False)
        log(f"  Saved {fpath} ({len(df):,} rows)")
    return len(rows)


# ═══════════════════════════════════════════════════════════════
# SUMMARY & VALIDATION
# ═══════════════════════════════════════════════════════════════

def compute_summary(results_rows, approach_name):
    """Compute per-strategy mean Sharpe across years (np.nanmean of per-year means)."""
    df = pd.DataFrame(results_rows)
    if df.empty:
        return {}
    summary = {}
    for sid in STRATEGIES:
        sub = df[df["strategy"] == sid]
        if sub.empty:
            summary[sid] = np.nan
            continue
        # Per-year: mean across tickers
        year_means = []
        for y in TEST_YEARS:
            ysub = sub[sub["year"] == y]
            if len(ysub) > 0:
                year_means.append(ysub["sharpe"].mean())
            else:
                year_means.append(np.nan)
        summary[sid] = np.nanmean(year_means)
    return summary


def print_summary(a_summary, b_summary, c_summary, d_summary, comm):
    """Print validation summary."""
    log(f"\n{'='*80}")
    log(f"VALIDATION SUMMARY — comm={comm}")
    log(f"{'='*80}")
    log(f"{'Strategy':<10} {'A':>8} {'B':>8} {'C':>8} {'D':>8} {'B>A?':>6} {'C>A?':>6} {'D>A?':>6}")
    log(f"{'-'*68}")

    for sid in STRATEGIES:
        a = a_summary.get(sid, np.nan)
        b = b_summary.get(sid, np.nan)
        c = c_summary.get(sid, np.nan)
        d = d_summary.get(sid, np.nan)
        b_win = "YES" if (not np.isnan(b) and not np.isnan(a) and b > a) else "no"
        c_win = "YES" if (not np.isnan(c) and not np.isnan(a) and c > a) else "no"
        d_win = "YES" if (not np.isnan(d) and not np.isnan(a) and d > a) else "no"
        log(f"{sid:<10} {a:>+8.3f} {b:>+8.3f} {c:>+8.3f} {d:>+8.3f} {b_win:>6} {c_win:>6} {d_win:>6}")

    overall_a = np.nanmean(list(a_summary.values()))
    overall_b = np.nanmean(list(b_summary.values()))
    overall_c = np.nanmean(list(c_summary.values()))
    overall_d = np.nanmean(list(d_summary.values()))
    log(f"{'-'*68}")
    log(f"{'Overall':<10} {overall_a:>+8.3f} {overall_b:>+8.3f} {overall_c:>+8.3f} {overall_d:>+8.3f}")
    log(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_global = time.time()

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("stage3_walkforward.py — Walk-Forward A/B/C/D")
    log(f"Test years: {TEST_YEARS}")
    log(f"Commission levels: {COMM_LEVELS}")
    log(f"Tickers: {len(TICKERS)}, Strategies: {len(STRATEGIES)}")
    log("=" * 80)

    # ── Load data ──
    log("Loading data...")
    daily_df, _, _ = load_data()

    kr = pd.read_parquet(KR_PATH)
    kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    log(f"  Key rate: {len(kr_s)} dates, range {kr_s.index.min().date()} - {kr_s.index.max().date()}")

    vpred_df = pd.read_parquet(VPRED_PATH)
    vpred_df["date"] = pd.to_datetime(vpred_df["date"])
    log(f"  Vpred: {len(vpred_df):,} rows, {vpred_df['date'].min().date()} - {vpred_df['date'].max().date()}")

    # Load narrowed grid
    with open(GRID_PATH) as f:
        grid = json.load(f)
    log(f"  Grid loaded from {GRID_PATH}")

    # Load best configs (for reference)
    best_configs = {}
    if BEST_CONFIGS_PATH.exists():
        with open(BEST_CONFIGS_PATH) as f:
            best_configs = json.load(f)
        log(f"  Best configs loaded: {len(best_configs)} entries")

    # ── Warmup numba ──
    log("Warming up numba...")
    warmup_numba()
    # Also warmup _apply_B_numba
    _test_pos = np.zeros(50)
    _test_pos[10:15] = 1.0
    _test_close = np.random.randn(50).cumsum() + 100
    _test_sigma = np.ones(50) * 0.02
    _ = _apply_B_numba(_test_pos, _test_close, _test_sigma, 0.02, 1.0, 1.5, 15, 0.5, 1.0)
    log("  Numba warmup done")

    # ── Prepare per-ticker data ──
    log("Preparing per-ticker data...")
    ticker_data_dict = {}
    for ticker in TICKERS:
        td = prepare_ticker_data(ticker, daily_df, kr_s, vpred_df)
        if td is not None:
            ticker_data_dict[ticker] = td
            log(f"  {ticker}: {td['n']} bars, {td['dates'].min().date()} - {td['dates'].max().date()}")
        else:
            log(f"  {ticker}: SKIPPED (too few data)")
    log(f"  {len(ticker_data_dict)} tickers prepared")

    # ── Run for each commission level ──
    for comm in COMM_LEVELS:
        t_comm = time.time()
        comm_suffix = "" if comm == 0.0005 else "_006"
        log(f"\n{'#'*80}")
        log(f"COMMISSION = {comm}")
        log(f"{'#'*80}")

        # ── Approach A ──
        a_rows, a_positions, grid_log = walkforward_A(ticker_data_dict, grid, comm)
        a_df = pd.DataFrame(a_rows)
        a_fname = f"wf_A_results{comm_suffix}.csv"
        a_df.to_csv(OUT_DIR / a_fname, index=False)
        log(f"  Saved {OUT_DIR / a_fname} ({len(a_df)} rows)")
        save_positions_parquet(a_positions, ticker_data_dict, "A", comm_suffix)

        # Save grid info
        if comm == 0.0005:
            grid_out = {sid: {"signal": grid[sid].get("signal", {}),
                              "rm": grid[sid].get("rm", {}),
                              "confirmations": grid[sid].get("confirmations", []),
                              "filters": grid[sid].get("filters", {}),
                              "grid_sizes": grid_log.get(sid, {})}
                        for sid in STRATEGIES}
            with open(GRIDS_DIR / "walkforward_grid_A.json", "w") as f:
                json.dump(grid_out, f, indent=2, default=str)
            log(f"  Saved {GRIDS_DIR / 'walkforward_grid_A.json'}")

        a_summary = compute_summary(a_rows, "A")

        # ── Approach B ──
        b_rows, b_positions = walkforward_B(ticker_data_dict, a_positions, comm)
        b_df = pd.DataFrame(b_rows)
        b_fname = f"wf_B_results{comm_suffix}.csv"
        b_df.to_csv(OUT_DIR / b_fname, index=False)
        log(f"  Saved {OUT_DIR / b_fname} ({len(b_df)} rows)")
        save_positions_parquet(b_positions, ticker_data_dict, "B", comm_suffix)
        b_summary = compute_summary(b_rows, "B")

        # ── Approach C ──
        c_rows, c_positions = walkforward_C(ticker_data_dict, a_positions, comm)
        c_df = pd.DataFrame(c_rows)
        c_fname = f"wf_C_results{comm_suffix}.csv"
        c_df.to_csv(OUT_DIR / c_fname, index=False)
        log(f"  Saved {OUT_DIR / c_fname} ({len(c_df)} rows)")
        save_positions_parquet(c_positions, ticker_data_dict, "C", comm_suffix)
        c_summary = compute_summary(c_rows, "C")

        # ── Approach D ──
        d_rows, d_positions = walkforward_D(ticker_data_dict, a_positions, comm)
        d_df = pd.DataFrame(d_rows)
        d_fname = f"wf_D_results{comm_suffix}.csv"
        d_df.to_csv(OUT_DIR / d_fname, index=False)
        log(f"  Saved {OUT_DIR / d_fname} ({len(d_df)} rows)")
        save_positions_parquet(d_positions, ticker_data_dict, "D", comm_suffix)
        d_summary = compute_summary(d_rows, "D")

        # ── Print summary ──
        print_summary(a_summary, b_summary, c_summary, d_summary, comm)

        # ── Archivist: save grids and configs ──
        comm_label = f"{round(comm*10000):03d}"
        for approach, df_app, summary in [("A", a_df, a_summary),
                                           ("B", b_df, b_summary),
                                           ("C", c_df, c_summary),
                                           ("D", d_df, d_summary)]:
            configs_app = {}
            for _, r in df_app.iterrows():
                key = f"{r.get('strategy','?')}_{r.get('ticker','?')}_{r.get('year','?')}"
                configs_app[key] = {c: r[c] for c in r.index if c not in ['sharpe']}
            archive(
                stage_name=f"stage3_wf_{approach}_{comm_label}",
                results_df=df_app,
                configs=configs_app,
                grid_used=grid if approach == "A" else {"approach": approach, "comm": comm},
                out_dir=OUT_DIR
            )

        elapsed_comm = time.time() - t_comm
        log(f"  Commission {comm} done in {elapsed_comm / 60:.1f} min")

    # ── Final cross-commission comparison ──
    log("\n" + "=" * 80)
    log("ALL DONE")
    elapsed_total = time.time() - t_global
    log(f"Total time: {elapsed_total / 60:.1f} min ({elapsed_total / 3600:.2f} hours)")
    log("=" * 80)


if __name__ == "__main__":
    main()
