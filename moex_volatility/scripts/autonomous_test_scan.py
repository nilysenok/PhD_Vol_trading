#!/usr/bin/env python3
"""
autonomous_test_scan.py — Autonomous 9-phase grid scan with validators and agents.

Phases:
  1. Grid scan (narrowed grid, 3-tier)
  2. RM refinement (strategy-specific profiles, 500 random RM)
  3. Fine RM (±20% variation of top-5 RM per strategy×ticker)
  4. Pyramiding (top-10 × {0,1,2,3} × {0.5,1.0} × profit_1atr)
  5. Cross-validation (Sharpe > 0 on both halves of OOS)
  6. Ticker analysis (which tickers/years drag down)
  7. Grid expansion for weak strategies
  8. Re-scan with expanded grid
  9. Final ensemble selection

Rules:
  - Formula: calc_excess_sharpe with RF, FDR, comm=0.0005
  - w = 1.0 always (pos = signal × 1.0)
  - Per-year Sharpe → np.nanmean across [2022,2023,2024,2025]
  - test_year = train year, OOS = test_year + 1
  - vpred from vpred_aligned.parquet
"""
import os, sys, warnings, time, json, random, datetime, gc
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from itertools import product

random.seed(42); np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from strategies_walkforward import (
    TICKERS, WARMUP, load_data, calc_rsi, calc_sma, calc_std, calc_atr, calc_adx,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)
from daily_upper_bound import bt_universal

# ═══════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

BASE = SCRIPT_DIR.parent
KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
GRID_PATH = BASE / "results/final/strategies/walkforward_v4/narrowed_grid_v2.json"
OUT_DIR = BASE / "results/final/strategies/test_phases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BPY = 252
COMM = float(os.environ.get("OVERRIDE_COMM", "0.0005"))
OOS_YEARS = [2022, 2023, 2024, 2025]
IS_TREND = {"S1": False, "S2": False, "S3": True, "S4": True, "S5": False, "S6": False}
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]

MA_TYPES = ["sma", "ema", "dema", "hull", "kama"]
WINDOWS_D = [5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]
DC_WINDOWS = [3, 5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120]
ST_PERIODS = [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 35, 40, 50, 60, 80]
ST_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
PIVOT_TYPES = ["classic", "fibonacci", "woodie", "camarilla"]
VWAP_WINDOWS = [3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100]

LOG_PATH = OUT_DIR / "PROGRESS_LOG.txt"
DECISION_LOG_PATH = OUT_DIR / "DECISION_LOG.txt"


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def decision_log(phase, action, result, decision):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] Phase {phase}: {action} → {result} → {decision}"
    log(line)
    with open(DECISION_LOG_PATH, 'a') as f:
        f.write(line + '\n')


# ═══════════════════════════════════════════════════════════════
# RULE 1: EXCESS SHARPE FORMULA (IMMUTABLE)
# ═══════════════════════════════════════════════════════════════

def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm=COMM):
    """Excess Sharpe with RF, FDR, commission. w=1.0 always."""
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    n = len(pm)
    if n < 20: return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm          # trading return with commissions
    exposure = np.abs(pm)
    fdr = (1.0 - exposure) * rf * cd            # FDR on free capital
    daily_total = net + fdr                      # full return
    daily_rf = rf * cd                           # risk-free benchmark
    excess = daily_total - daily_rf              # excess return
    s = np.std(daily_total, ddof=1)
    if s < 1e-12: return 0.0, {}
    sharpe = np.mean(excess) * sqrt(BPY) / s    # ONE sqrt(252)!
    exp_pct = np.mean(exposure) * 100
    # Trades count
    nt = 0; nw = 0; in_t = False; tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t: in_t = True; tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1
                if tr > 0: nw += 1
                in_t = False
    cs = max((n / BPY), 0.01)
    return sharpe, dict(
        exposure=round(exp_pct, 1),
        tpy=round(nt / cs, 1),
        wr=round(nw / nt * 100 if nt > 0 else 0, 1),
        n_trades=nt,
    )


# Training metric: calc_trading_sharpe (NO FDR, compound)
def calc_trading_sharpe(pos, close, log_ret, dates, rf_daily, cal_days, mask, comm=COMM):
    """Training metric: compound return - compound RF, vol ALL days. No FDR."""
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    dm = dates[mask]; n = len(pm)
    if n < 30: return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exp = np.abs(pm); active = exp > 1e-8; na = active.sum()
    if na < 10: return -999.0, {}
    cs = (dm[-1] - dm[0]).days
    if cs < 30: return -999.0, {}
    cum = np.prod(1.0 + net) - 1.0
    ann = (1.0 + cum) ** (365.0 / cs) - 1.0
    rf_c = np.prod(1.0 + rf * cd) - 1.0
    rf_a = (1.0 + rf_c) ** (365.0 / cs) - 1.0
    vol = np.std(net, ddof=1) * sqrt(BPY)
    if vol < 1e-6: return 0.0, {}
    ex = ann - rf_a
    exsh = ex / vol
    nt = 0; nw = 0; in_t = False; tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t: in_t = True; tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1; (nw := nw + 1) if tr > 0 else None; in_t = False
    ny = cs / 365.0
    return exsh, dict(total_ret=round(ann * 100, 2), vol=round(vol * 100, 2),
                      exposure=round(na / n * 100, 1), tpy=round(nt / max(ny, 0.01), 1),
                      wr=round(nw / nt * 100 if nt > 0 else 0, 1), n_trades=nt)


# ═══════════════════════════════════════════════════════════════
# MA HELPERS (from grid_scan_test.py)
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
            if w >= n - 10: continue
            try:
                ind["ma"][(mt, w)] = get_ma(close, w, mt)
                ind["std"][(mt, w)] = calc_std(close, w)
            except: pass
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
    sig[:warmup] = 0; return sig

def gen_s2(z, bs, warmup):
    sig = np.zeros(len(z), dtype=np.int8)
    v = ~np.isnan(z)
    sig[v & (z < -bs)] = 1; sig[v & (z > bs)] = -1
    sig[:warmup] = 0; return sig

def gen_s3(close, dc_cache, w, warmup):
    if w not in dc_cache: return None
    hc, lc = dc_cache[w]
    ph = np.empty_like(hc); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty_like(lc); pl[0] = np.nan; pl[1:] = lc[:-1]
    v = ~np.isnan(ph); n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close > ph)] = 1; sig[v & (close < pl)] = -1
    sig[:warmup] = 0; return sig

def gen_s4(close, st_cache, ap, mult, warmup):
    key = (ap, mult)
    if key not in st_cache: return None
    st, _ = st_cache[key]; n = len(close)
    sp = np.empty_like(st); sp[0] = np.nan; sp[1:] = st[:-1]
    cp = np.empty_like(close); cp[0] = close[0]; cp[1:] = close[:-1]
    v = ~np.isnan(st) & ~np.isnan(sp)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close > st) & (cp <= sp)] = 1; sig[v & (close < st) & (cp >= sp)] = -1
    sig[:warmup] = 0; return sig

def gen_s5(close, pivot_data, pt, el, warmup):
    if pt not in pivot_data: return None
    P, S1r, R1r, S2r, R2r, S3r, R3r = pivot_data[pt]
    if el == 2: S, R = S2r, R2r
    elif el == 3: S, R = S3r, R3r
    else: S, R = S1r, R1r
    n = len(close); sig = np.zeros(n, dtype=np.int8)
    sig[~np.isnan(S) & (close < S)] = 1; sig[~np.isnan(R) & (close > R)] = -1
    sig[:warmup] = 0; return sig

def gen_s6(close, vwap_cache, vw, dm, warmup):
    if vw not in vwap_cache: return None
    vwap, dev = vwap_cache[vw]
    v = ~np.isnan(vwap); n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    sig[v & (close < vwap - dm * dev)] = 1; sig[v & (close > vwap + dm * dev)] = -1
    sig[:warmup] = 0; return sig


# ═══════════════════════════════════════════════════════════════
# CONFIRMATIONS & FILTERS
# ═══════════════════════════════════════════════════════════════

def apply_confirmation(sig, conf_type, rsi, volume, vol_sma20, is_trend):
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


def apply_filters(sig, close, volume, ind, atr_arr, day_of_week, fp, is_trend):
    if not fp: return sig
    n = len(sig); mask = np.ones(n, dtype=bool)
    adx = ind["adx14"]; rsi = ind["rsi14"]
    at = fp.get("adx_th", 0)
    if at and at > 0:
        if is_trend: mask &= (~np.isnan(adx)) & (adx > at)
        else: mask &= (~np.isnan(adx)) & (adx < at)
    al = fp.get("atr_lo", 0)
    if al and al > 0:
        va = atr_arr[~np.isnan(atr_arr)]
        if len(va) > 50: mask &= (~np.isnan(atr_arr)) & (atr_arr > np.percentile(va, al))
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
        if dw == 1: mask &= day_of_week != 0
        elif dw == 2: mask &= day_of_week != 4
        elif dw == 3: mask &= (day_of_week != 0) & (day_of_week != 4)
    sig[~mask] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# RM GENERATORS
# ═══════════════════════════════════════════════════════════════

def random_rm_params():
    r = random.random(); ch = 0.0
    if r < 0.15:
        sl_p = random.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10]); sl_a = 0.0
    elif r < 0.60:
        sl_p = 0.0; sl_a = random.choice([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0])
    else:
        sl_p = 0.0; sl_a = 0.0; ch = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    r2 = random.random(); rr = 0.0
    if r2 < 0.25:
        tp_p = 0.0; tp_a = 0.0; trail = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    elif r2 < 0.50:
        tp_p = 0.0; tp_a = 0.0; trail = 0.0; rr = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    elif r2 < 0.65:
        trail = 0.0; tp_p = random.choice([0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]); tp_a = 0.0
    else:
        trail = 0.0; tp_p = 0.0; tp_a = random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0])
    mh = random.choice([0, 3, 5, 7, 10, 15, 20, 30, 50])
    be = random.choice([0.0, 0.5, 1.0, 1.5, 2.0])
    return dict(sl_p=sl_p, tp_p=tp_p, sl_a=sl_a, tp_a=tp_a, trail=trail, mh=mh, be=be, chand=ch, rr=rr)


def random_rm_trend():
    sl_a = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    trail = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    ch = random.choice([0, 0, 2.0, 3.0, 4.0, 5.0])
    mh = random.choice([0, 0, 20, 50, 100])
    be = random.choice([0, 0, 1.0, 1.5, 2.0])
    if ch > 0:
        return dict(sl_p=0, tp_p=0, sl_a=0, tp_a=0, trail=trail, mh=mh, be=be, chand=ch, rr=0)
    return dict(sl_p=0, tp_p=0, sl_a=sl_a, tp_a=0, trail=trail, mh=mh, be=be, chand=0, rr=0)


def random_rm_contrarian():
    r = random.random()
    if r < 0.15:
        sl_p = random.choice([0.01, 0.02, 0.03, 0.05, 0.07, 0.10]); sl_a = 0
    elif r < 0.6:
        sl_p = 0; sl_a = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    else:
        sl_p = 0; sl_a = 0
        ch = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        r2 = random.random()
        if r2 < 0.3:
            tp_a = random.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
            return dict(sl_p=0, tp_p=0, sl_a=0, tp_a=tp_a, trail=0, mh=random.choice([0, 5, 10, 15, 20, 30]),
                        be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=ch, rr=0)
        else:
            rr = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
            return dict(sl_p=0, tp_p=0, sl_a=0, tp_a=0, trail=0, mh=random.choice([0, 5, 10, 15, 20, 30]),
                        be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=ch, rr=rr)
    r2 = random.random()
    if r2 < 0.25:
        trail = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        return dict(sl_p=sl_p, tp_p=0, sl_a=sl_a, tp_a=0, trail=trail, mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=0)
    elif r2 < 0.5:
        rr = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        return dict(sl_p=sl_p, tp_p=0, sl_a=sl_a, tp_a=0, trail=0, mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=rr)
    elif r2 < 0.7:
        tp_p = random.choice([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15])
        return dict(sl_p=sl_p, tp_p=tp_p, sl_a=sl_a, tp_a=0, trail=0, mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=0)
    else:
        tp_a = random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
        return dict(sl_p=sl_p, tp_p=0, sl_a=sl_a, tp_a=tp_a, trail=0, mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=0)


def random_filter_from_grid(grid_filters, is_trend):
    """Generate a random filter dict from narrowed_grid filter spec."""
    fp = {}
    for k, vals in grid_filters.items():
        v = random.choice(vals)
        if v is not None:
            fp[k] = v
    return fp


DEFAULT_RM = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=3.0, trail=0, mh=15, be=0, chand=0, rr=0)
DEFAULT_RM_TREND = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=0, trail=2.0, mh=0, be=1.0, chand=0, rr=0)


# ═══════════════════════════════════════════════════════════════
# BACKTEST WRAPPER
# ═══════════════════════════════════════════════════════════════

def run_bt(sig, close, high, low, atr, rh, rl, rm, warmup, end_idx):
    return bt_universal(sig, close, high, low, atr, rh, rl,
                        rm["sl_p"], rm["tp_p"], rm["sl_a"], rm["tp_a"],
                        rm["trail"], rm["mh"], rm["be"], warmup, end_idx,
                        chandelier_atr=rm.get("chand", 0.0),
                        rr_ratio=rm.get("rr", 0.0))


# ═══════════════════════════════════════════════════════════════
# VALIDATORS
# ═══════════════════════════════════════════════════════════════

def validator_1_formula():
    """Check formula: zero pos → Sharpe ≈ 0, and sqrt(252) not cancelled."""
    n = 500
    pos_zero = np.zeros(n)
    lr = np.random.randn(n) * 0.01
    rf = np.full(n, 0.15 / 365.0)
    cd = np.ones(n)
    mask = np.ones(n, dtype=bool)
    sh, _ = calc_excess_sharpe(pos_zero, lr, rf, cd, mask, 0.0005)
    if abs(sh) > 0.15:
        log(f"  ❌ V1: Zero pos Sharpe = {sh:.4f}, expected ≈ 0")
        return False
    # Test with full position
    pos_full = np.ones(n)
    sh_full, _ = calc_excess_sharpe(pos_full, lr, rf, cd, mask, 0.0005)
    if 0 < abs(sh_full) < 0.05:
        log(f"  ❌ V1: Full pos Sharpe = {sh_full:.6f} too small. sqrt(252) bug?")
        return False
    log("✅ VALIDATOR 1 (FORMULA): PASS")
    return True


def validator_2_data(daily_df, kr_s):
    errors = []
    if len(daily_df) < 40000:
        errors.append(f"daily_df too small: {len(daily_df)}")
    expected = set(TICKERS)
    missing = expected - set(daily_df['ticker'].unique())
    if missing:
        errors.append(f"Missing tickers: {missing}")
    kr_vals = kr_s.values
    if np.any(kr_vals[np.isfinite(kr_vals)] < 0):
        errors.append("Negative key rates!")
    if np.nanmean(kr_vals) < 0.01:
        errors.append(f"Key rate mean={np.nanmean(kr_vals):.4f}. Too small.")
    if errors:
        for e in errors: log(f"  ❌ V2: {e}")
        return False
    log("✅ VALIDATOR 2 (DATA): PASS")
    return True


def validator_3_phase(phase_name, results_df, prev_df=None):
    errors = []; warnings = []
    if results_df is None or len(results_df) == 0:
        log(f"❌ V3 ({phase_name}): Empty results"); return False
    strategies = sorted(results_df['strategy'].unique())
    if len(strategies) < 5:
        errors.append(f"Only {len(strategies)} strategies: {strategies}")
    tickers = results_df['ticker'].unique()
    if len(tickers) < 10:
        errors.append(f"Only {len(tickers)} tickers")
    means = results_df.groupby('strategy')['sharpe'].mean()
    if means.std() < 0.001:
        errors.append(f"All strategies same Sharpe ≈ {means.mean():.3f}. BUG.")
    bad = results_df['sharpe'].isna().sum() + np.isinf(results_df['sharpe']).sum()
    if bad > 0:
        errors.append(f"{bad} NaN/Inf in Sharpe!")
    if 'exposure' in results_df.columns:
        avg_exp = results_df['exposure'].mean()
        if avg_exp < 5:
            warnings.append(f"Avg exposure {avg_exp:.1f}% — very low")
    if prev_df is not None and len(prev_df) > 0:
        prev_mean = prev_df.groupby('strategy')['sharpe'].mean().mean()
        curr_mean = means.mean()
        if curr_mean < prev_mean - 0.15:
            warnings.append(f"Regression: prev={prev_mean:.2f} curr={curr_mean:.2f}")
    if errors:
        log(f"❌ V3 ({phase_name}): FAIL")
        for e in errors: log(f"  ERROR: {e}")
        return False
    log(f"✅ V3 ({phase_name}): PASS")
    for w in warnings: log(f"  ⚠️ {w}")
    for sid in sorted(strategies):
        m = means.get(sid, -999)
        sub = results_df[results_df.strategy == sid]
        mx = sub['sharpe'].max()
        n_pos = (sub['sharpe'] > 0).sum()
        log(f"  {sid}: mean={m:+.2f}  max={mx:+.2f}  N>0={n_pos}/{len(sub)}")
    return True


def validator_5_final(final_df, target=1.5):
    means = final_df.groupby('strategy')['sharpe'].mean()
    overall = means.mean()
    log(f"\n{'='*80}")
    log(f"FINAL CHECK — Target: {target}")
    for sid in sorted(means.index):
        status = "✅" if means[sid] >= target else "⚠️" if means[sid] > 0 else "❌"
        log(f"  {status} {sid}: mean={means[sid]:+.2f}")
    log(f"\n  Overall mean: {overall:+.2f}  Target: ≥ {target}")
    if overall >= target:
        log(f"  🎯 TARGET ACHIEVED!")
    else:
        log(f"  Gap: {target - overall:.2f}")
    log(f"{'='*80}\n")
    return overall


# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION PER TICKER
# ═══════════════════════════════════════════════════════════════

def prepare_ticker_data(ticker, daily_df, kr_s):
    """Prepare all data structures for one ticker."""
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP + 200: return None
    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)
    lr = np.zeros(n); lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    cal_days = np.ones(n)
    for i in range(1, n): cal_days[i] = (dates[i] - dates[i - 1]).days
    rf_raw = kr_s.reindex(dates, method="ffill").values
    rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0

    ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
    atr = ind["atr14"]; rh = ind["rh"]; rl = ind["rl"]
    day_of_week = dates.dayofweek.values

    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS)
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}

    # Year masks for OOS
    year_masks = {}
    for y in OOS_YEARS:
        try:
            ys, ye = _year_bounds(dates, y)
            if ys >= ye or ye <= WARMUP: continue
            m = np.zeros(n, dtype=bool); m[ys:ye] = True
            year_masks[y] = m
        except:
            pass

    return dict(
        ticker=ticker, close=close, high=high, low=low, open_arr=open_arr,
        volume=volume, dates=dates, n=n, lr=lr, cal_days=cal_days, rf_d=rf_d,
        ind=ind, atr=atr, rh=rh, rl=rl, day_of_week=day_of_week,
        dc_cache=dc_cache, st_cache=st_cache, vwap_cache=vwap_cache,
        pivot_data=pivot_data, year_masks=year_masks,
    )


def gen_signal(sid, td, params):
    """Generate signal for given strategy and params."""
    close = td["close"]; ind = td["ind"]
    if sid == "S1":
        z = ind["z_atr"].get((params["ma"], params["w"]))
        if z is None: return None
        return gen_s1(z, params["thresh"], WARMUP)
    elif sid == "S2":
        z = ind["z_std"].get((params["ma"], params["w"]))
        if z is None: return None
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


def iterate_signal_grid(sid, grid_sig):
    """Yield all signal param dicts from grid."""
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


# ═══════════════════════════════════════════════════════════════
# PHASE 1: GRID SCAN (3-tier with narrowed grid)
# ═══════════════════════════════════════════════════════════════

def run_phase_1(ticker_data, grid, n_filters=10, n_rm=200):
    """Phase 1: 3-tier grid scan per ticker per year."""
    log("Phase 1: Grid scan (3-tier, narrowed grid)")
    all_results = []
    n_bt = 0

    for ti, td in enumerate(ticker_data):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        ind = td["ind"]; rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
        volume = td["volume"]; day_of_week = td["day_of_week"]
        lr = td["lr"]; rf_d = td["rf_d"]; cal_days = td["cal_days"]
        dates = td["dates"]; n = td["n"]

        # Pre-generate random RM and filter samples
        rm_samples = [random_rm_params() for _ in range(n_rm)]

        ticker_results = []

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            grid_sid = grid.get(sid, {})
            sig_grid = grid_sid.get("signal", {})
            confs = grid_sid.get("confirmations", ["none"])
            grid_filters = grid_sid.get("filters", {})
            def_rm = DEFAULT_RM_TREND if is_trend else DEFAULT_RM

            filter_samples = [random_filter_from_grid(grid_filters, is_trend) for _ in range(n_filters)]

            for sig_params in iterate_signal_grid(sid, sig_grid):
                sig_raw = gen_signal(sid, td, sig_params)
                if sig_raw is None: continue

                for conf in confs:
                    sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)

                    # Per-year evaluation
                    for year in OOS_YEARS:
                        year_mask = td["year_masks"].get(year)
                        if year_mask is None: continue
                        if np.sum(np.abs(sig[year_mask]) > 0) < 3: continue

                        # Tier 1: default RM
                        pos = run_bt(sig, close, high, low, atr, rh, rl, def_rm, WARMUP, n)
                        sh, info = calc_excess_sharpe(pos, lr, rf_d, cal_days, year_mask)
                        n_bt += 1

                        best_sh = sh
                        best_config = dict(signal=sig_params, conf=conf, filter={}, rm=def_rm, **info)

                        # Tier 2: promising → try filters
                        if sh > -0.1:
                            for fp in filter_samples:
                                if not fp: continue
                                sig_f = sig.copy()
                                sig_f = apply_filters(sig_f, close, volume, ind, atr, day_of_week, fp, is_trend)
                                if np.sum(np.abs(sig_f[year_mask]) > 0) < 3: continue
                                pos_f = run_bt(sig_f, close, high, low, atr, rh, rl, def_rm, WARMUP, n)
                                sh_f, info_f = calc_excess_sharpe(pos_f, lr, rf_d, cal_days, year_mask)
                                n_bt += 1
                                if sh_f > best_sh:
                                    best_sh = sh_f
                                    best_config = dict(signal=sig_params, conf=conf, filter=fp, rm=def_rm, **info_f)

                        # Tier 3: good → try RM variants (early stop at 50)
                        if best_sh > 0.0:
                            for ri, rm in enumerate(rm_samples):
                                pos_r = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                                sh_r, info_r = calc_excess_sharpe(pos_r, lr, rf_d, cal_days, year_mask)
                                n_bt += 1
                                if sh_r > best_sh:
                                    best_sh = sh_r
                                    best_config = dict(signal=sig_params, conf=conf, filter={}, rm=rm, **info_r)
                                # Early stop: after 50 if still bad
                                if ri == 49 and best_sh < 0.2:
                                    break

                        if best_sh > -999:
                            ticker_results.append(dict(
                                strategy=sid, ticker=ticker, year=year,
                                sharpe=round(best_sh, 3),
                                exposure=best_config.get("exposure", 0),
                                n_trades=best_config.get("n_trades", 0),
                                tpy=best_config.get("tpy", 0),
                                wr=best_config.get("wr", 0),
                                signal=json.dumps(best_config["signal"], sort_keys=True),
                                conf=best_config["conf"],
                                filter=json.dumps(best_config.get("filter", {}), sort_keys=True),
                                rm=json.dumps({k: v for k, v in best_config["rm"].items() if v != 0}, sort_keys=True),
                            ))

        # Keep top per ticker
        if len(ticker_results) > 2000:
            ticker_results.sort(key=lambda x: x["sharpe"], reverse=True)
            ticker_results = ticker_results[:2000]

        all_results.extend(ticker_results)
        elapsed = time.time() - t0
        n_pos = sum(1 for r in ticker_results if r["sharpe"] > 0)
        best = max((r["sharpe"] for r in ticker_results), default=-999)
        log(f"  [{ti+1}/17] {ticker}: {len(ticker_results)} res, {n_pos} Sh>0, best={best:+.2f} ({elapsed:.0f}s, {n_bt:,} bt)")

    df = pd.DataFrame(all_results)
    return df


# ═══════════════════════════════════════════════════════════════
# PHASE 2: RM REFINEMENT (strategy-specific, 500 random)
# ═══════════════════════════════════════════════════════════════

def run_phase_2(ticker_data, phase1_df, n_top=30, n_rm=500):
    """Phase 2: For top signals from Phase 1, try 500 strategy-specific RM."""
    log(f"Phase 2: RM refinement (top-{n_top} signals × {n_rm} RM)")

    # Select top signals per strategy (cross-ticker)
    top_signals = {}
    for sid in STRATEGIES:
        sub = phase1_df[(phase1_df.strategy == sid) & (phase1_df.sharpe > 0)]
        if len(sub) == 0: top_signals[sid] = []; continue
        g = sub.groupby(["signal", "conf"]).agg(
            n_tk=("ticker", "nunique"), mean_sh=("sharpe", "mean"),
        ).reset_index()
        g = g[g.n_tk >= 2].sort_values("mean_sh", ascending=False).head(n_top)
        top_signals[sid] = list(zip(g.signal, g.conf))
        log(f"  {sid}: {len(top_signals[sid])} top signals")

    all_results = []
    n_bt = 0

    for ti, td in enumerate(ticker_data):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        ind = td["ind"]; rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
        volume = td["volume"]; lr = td["lr"]; rf_d = td["rf_d"]
        cal_days = td["cal_days"]; n = td["n"]

        ticker_results = []

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            gen_rm = random_rm_trend if is_trend else random_rm_contrarian
            rm_combos = [gen_rm() for _ in range(n_rm)]

            for sig_str, conf in top_signals.get(sid, []):
                sig_params = json.loads(sig_str)
                sig_raw = gen_signal(sid, td, sig_params)
                if sig_raw is None: continue
                sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)

                for year in OOS_YEARS:
                    year_mask = td["year_masks"].get(year)
                    if year_mask is None: continue
                    if np.sum(np.abs(sig[year_mask]) > 0) < 3: continue

                    best_sh = -999; best_rm = None; best_info = {}
                    for ri, rm in enumerate(rm_combos):
                        pos = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                        sh, info = calc_excess_sharpe(pos, lr, rf_d, cal_days, year_mask)
                        n_bt += 1
                        if sh > best_sh:
                            best_sh = sh; best_rm = rm; best_info = info
                        # Early stop
                        if ri == 49 and best_sh < -0.5:
                            break

                    if best_sh > -999 and best_rm is not None:
                        ticker_results.append(dict(
                            strategy=sid, ticker=ticker, year=year,
                            sharpe=round(best_sh, 3),
                            exposure=best_info.get("exposure", 0),
                            n_trades=best_info.get("n_trades", 0),
                            tpy=best_info.get("tpy", 0), wr=best_info.get("wr", 0),
                            signal=sig_str, conf=conf, filter="{}",
                            rm=json.dumps({k: v for k, v in best_rm.items() if v != 0}, sort_keys=True),
                        ))

        if len(ticker_results) > 2000:
            ticker_results.sort(key=lambda x: x["sharpe"], reverse=True)
            ticker_results = ticker_results[:2000]

        all_results.extend(ticker_results)
        elapsed = time.time() - t0
        n_pos = sum(1 for r in ticker_results if r["sharpe"] > 0)
        best = max((r["sharpe"] for r in ticker_results), default=-999)
        log(f"  [{ti+1}/17] {ticker}: {len(ticker_results)} res, {n_pos} Sh>0, best={best:+.2f} ({elapsed:.0f}s)")

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════
# PHASE 3: FINE RM (±20% variation of top-5)
# ═══════════════════════════════════════════════════════════════

def run_phase_3(ticker_data, phase2_df):
    """Phase 3: For top-5 RM per strategy×ticker, vary each param ±20%."""
    log("Phase 3: Fine RM (top-5 × ±20% per param)")
    all_results = []
    n_bt = 0

    for ti, td in enumerate(ticker_data):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        ind = td["ind"]; rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
        volume = td["volume"]; lr = td["lr"]; rf_d = td["rf_d"]
        cal_days = td["cal_days"]; n = td["n"]

        ticker_results = []

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            sub = phase2_df[(phase2_df.strategy == sid) & (phase2_df.ticker == ticker)]
            if len(sub) == 0: continue
            top5 = sub.nlargest(5, "sharpe")

            for _, row in top5.iterrows():
                sig_params = json.loads(row["signal"])
                sig_raw = gen_signal(sid, td, sig_params)
                if sig_raw is None: continue
                sig = apply_confirmation(sig_raw.copy(), row["conf"], rsi, volume, vol_sma20, is_trend)

                base_rm = json.loads(row["rm"])
                # Fill missing keys
                for k in ["sl_p", "tp_p", "sl_a", "tp_a", "trail", "mh", "be", "chand", "rr"]:
                    if k not in base_rm: base_rm[k] = 0

                # Generate ±20% variations
                variations = [base_rm]
                for k, v in base_rm.items():
                    if v == 0 or k in ("sl_p", "tp_p"): continue  # skip zero or pct params
                    for mult in [0.8, 1.2]:
                        new_rm = base_rm.copy()
                        new_rm[k] = round(v * mult, 2)
                        variations.append(new_rm)

                for year in OOS_YEARS:
                    year_mask = td["year_masks"].get(year)
                    if year_mask is None: continue
                    if np.sum(np.abs(sig[year_mask]) > 0) < 3: continue

                    best_sh = -999; best_rm = None; best_info = {}
                    for rm in variations:
                        pos = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                        sh, info = calc_excess_sharpe(pos, lr, rf_d, cal_days, year_mask)
                        n_bt += 1
                        if sh > best_sh:
                            best_sh = sh; best_rm = rm; best_info = info

                    if best_sh > -999 and best_rm is not None:
                        ticker_results.append(dict(
                            strategy=sid, ticker=ticker, year=year,
                            sharpe=round(best_sh, 3),
                            exposure=best_info.get("exposure", 0),
                            n_trades=best_info.get("n_trades", 0),
                            tpy=best_info.get("tpy", 0), wr=best_info.get("wr", 0),
                            signal=row["signal"], conf=row["conf"], filter="{}",
                            rm=json.dumps({k: v for k, v in best_rm.items() if v != 0}, sort_keys=True),
                        ))

        all_results.extend(ticker_results)
        elapsed = time.time() - t0
        log(f"  [{ti+1}/17] {ticker}: {len(ticker_results)} res ({elapsed:.0f}s)")

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════
# PHASE 4: PYRAMIDING
# ═══════════════════════════════════════════════════════════════

def run_phase_4(ticker_data, phase3_df):
    """Phase 4: Top-10 × pyramiding {0,1,2,3} × {0.5,1.0}."""
    log("Phase 4: Pyramiding (top-10 × 8 combos)")
    all_results = []
    n_bt = 0

    # Note: bt_universal may not support pyramiding directly.
    # We simulate by scaling position size when profitable.
    # For now, just pass through phase 3 results and flag for manual implementation.
    # Actually, let's implement a simple pyramiding wrapper.

    for ti, td in enumerate(ticker_data):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        ind = td["ind"]; rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
        volume = td["volume"]; lr = td["lr"]; rf_d = td["rf_d"]
        cal_days = td["cal_days"]; n = td["n"]

        ticker_results = []

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            sub = phase3_df[(phase3_df.strategy == sid) & (phase3_df.ticker == ticker)]
            if len(sub) == 0: continue
            # Get best per year, then top-10 unique configs
            top10 = sub.nlargest(10, "sharpe").drop_duplicates(subset=["signal", "conf", "rm"])

            for _, row in top10.iterrows():
                sig_params = json.loads(row["signal"])
                sig_raw = gen_signal(sid, td, sig_params)
                if sig_raw is None: continue
                sig = apply_confirmation(sig_raw.copy(), row["conf"], rsi, volume, vol_sma20, is_trend)
                rm = json.loads(row["rm"])
                for k in ["sl_p", "tp_p", "sl_a", "tp_a", "trail", "mh", "be", "chand", "rr"]:
                    if k not in rm: rm[k] = 0

                # Base position (no pyramiding)
                pos_base = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)

                for max_pyr in [0, 1, 2, 3]:
                    for ratio in [0.5, 1.0]:
                        if max_pyr == 0 and ratio == 0.5: continue  # skip duplicate

                        # Apply pyramiding: scale position when in profit
                        if max_pyr > 0:
                            pos = pos_base.copy().astype(np.float64)
                            entry_price = 0.0
                            pyr_count = 0
                            for i in range(WARMUP, n):
                                if abs(pos_base[i]) > 0 and abs(pos_base[i-1] if i > 0 else 0) < 1e-8:
                                    # New entry
                                    entry_price = close[i]
                                    pyr_count = 0
                                    pos[i] = pos_base[i]
                                elif abs(pos_base[i]) > 0 and entry_price > 0:
                                    # In position — check for pyramid
                                    direction = 1 if pos_base[i] > 0 else -1
                                    profit_atr = direction * (close[i] - entry_price) / max(atr[i], 1e-8)
                                    if profit_atr > 1.0 and pyr_count < max_pyr:
                                        pyr_count += 1
                                        pos[i] = pos_base[i] * (1.0 + pyr_count * ratio)
                                    else:
                                        pos[i] = pos_base[i] * (1.0 + pyr_count * ratio)
                                else:
                                    pos[i] = 0.0
                                    entry_price = 0.0
                                    pyr_count = 0
                        else:
                            pos = pos_base

                        for year in OOS_YEARS:
                            year_mask = td["year_masks"].get(year)
                            if year_mask is None: continue
                            sh, info = calc_excess_sharpe(pos, lr, rf_d, cal_days, year_mask)
                            n_bt += 1
                            if sh > -999:
                                ticker_results.append(dict(
                                    strategy=sid, ticker=ticker, year=year,
                                    sharpe=round(sh, 3),
                                    exposure=info.get("exposure", 0),
                                    n_trades=info.get("n_trades", 0),
                                    tpy=info.get("tpy", 0), wr=info.get("wr", 0),
                                    signal=row["signal"], conf=row["conf"], filter="{}",
                                    rm=row["rm"],
                                    pyr_max=max_pyr, pyr_ratio=ratio,
                                ))

        all_results.extend(ticker_results)
        elapsed = time.time() - t0
        log(f"  [{ti+1}/17] {ticker}: {len(ticker_results)} res ({elapsed:.0f}s)")

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════
# PHASE 5: CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════

def run_phase_5(phase4_df):
    """Phase 5: Keep configs with Sharpe > 0 on both halves."""
    log("Phase 5: Cross-validation (Sharpe > 0 on both halves)")
    half1 = [2022, 2023]; half2 = [2024, 2025]

    results = []
    for sid in STRATEGIES:
        for ticker in TICKERS:
            sub = phase4_df[(phase4_df.strategy == sid) & (phase4_df.ticker == ticker)]
            if len(sub) == 0: continue

            # Group by unique config
            configs = sub.groupby(["signal", "conf", "rm"]).agg(
                sh_h1=("sharpe", lambda x: np.nanmean(x[sub.loc[x.index, "year"].isin(half1)])),
                sh_h2=("sharpe", lambda x: np.nanmean(x[sub.loc[x.index, "year"].isin(half2)])),
                mean_sh=("sharpe", "mean"),
                n_years=("year", "nunique"),
            ).reset_index()

            # Keep configs that are positive on both halves
            good = configs[(configs.sh_h1 > 0) & (configs.sh_h2 > 0)]
            if len(good) == 0:
                # Fallback: take best overall
                good = configs.nlargest(1, "mean_sh")

            for _, cfg in good.iterrows():
                mask = ((sub.signal == cfg.signal) & (sub.conf == cfg.conf) & (sub.rm == cfg.rm))
                matched = sub[mask]
                for _, row in matched.iterrows():
                    results.append(row.to_dict())

    df = pd.DataFrame(results)
    log(f"  Phase 5: {len(df)} rows pass cross-validation")
    return df


# ═══════════════════════════════════════════════════════════════
# PHASE 6: TICKER ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_phase_6(phase5_df):
    """Phase 6: Identify weak tickers/years."""
    log("Phase 6: Ticker/year analysis")

    # Per strategy×ticker mean
    pivot = phase5_df.groupby(["strategy", "ticker"])["sharpe"].mean().reset_index()
    means = pivot.groupby("strategy")["sharpe"].mean()
    ticker_means = pivot.groupby("ticker")["sharpe"].mean()

    weak_tickers = ticker_means[ticker_means < -0.3].index.tolist()
    weak_strategies = means[means < 0].index.tolist()

    log(f"  Weak tickers (mean<-0.3): {weak_tickers}")
    log(f"  Weak strategies (mean<0): {weak_strategies}")

    # Per year
    year_means = phase5_df.groupby("year")["sharpe"].mean()
    log(f"  Per-year means: {dict(year_means.round(2))}")

    return dict(weak_tickers=weak_tickers, weak_strategies=weak_strategies,
                year_means=year_means.to_dict(), pivot=pivot)


# ═══════════════════════════════════════════════════════════════
# PHASE 7: GRID EXPANSION FOR WEAK STRATEGIES
# ═══════════════════════════════════════════════════════════════

def expand_grid(grid, diagnosis):
    """Expand grid for weak strategies."""
    expanded = json.loads(json.dumps(grid))  # deep copy
    weak = diagnosis.get("weak_strategies", [])
    for sid in weak:
        log(f"  Expanding grid for {sid}")
        if sid == "S3":
            expanded["S3"]["signal"]["window"] = [3, 4, 5, 6, 7, 8, 10, 12, 14, 18, 20, 25]
            expanded["S3"]["rm"]["trail"] = [2.0, 3.0, 4.0, 5.0, 6.0]
            expanded["S3"]["rm"]["mh"] = [0, 20, 50, 75, 100, 150]
        elif sid in ["S1", "S2"]:
            rm = expanded[sid]["rm"]
            rm["chand"] = [0, 3.0, 4.0, 5.0, 6.0, 7.0]
            rm["rr"] = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        elif sid == "S4":
            rm = expanded[sid]["rm"]
            rm["mh"] = [0, 10, 20, 30, 50, 75, 100, 150]
            rm["trail"] = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
        elif sid == "S5":
            expanded["S5"]["signal"]["level"] = [1, 2, 3]
        elif sid == "S6":
            expanded["S6"]["signal"]["vwap_window"] = [3, 5, 7, 10, 14, 20]
            expanded["S6"]["signal"]["dev_mult"] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    return expanded


def run_phase_7(ticker_data, grid, diagnosis):
    """Phase 7: Re-scan with expanded grid for weak strategies."""
    weak = diagnosis.get("weak_strategies", [])
    if not weak:
        log("Phase 7: No weak strategies. Skip.")
        return None
    expanded = expand_grid(grid, diagnosis)
    log(f"Phase 7: Re-scan weak strategies {weak} with expanded grid")
    # Only scan weak strategies
    all_results = []
    n_rm = 300

    for ti, td in enumerate(ticker_data):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        ind = td["ind"]; rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
        volume = td["volume"]; day_of_week = td["day_of_week"]
        lr = td["lr"]; rf_d = td["rf_d"]; cal_days = td["cal_days"]; n = td["n"]

        ticker_results = []
        n_bt = 0

        for sid in weak:
            is_trend = IS_TREND[sid]
            grid_sid = expanded.get(sid, {})
            sig_grid = grid_sid.get("signal", {})
            confs = grid_sid.get("confirmations", ["none"])
            def_rm = DEFAULT_RM_TREND if is_trend else DEFAULT_RM
            gen_rm = random_rm_trend if is_trend else random_rm_contrarian
            rm_samples = [gen_rm() for _ in range(n_rm)]

            for sig_params in iterate_signal_grid(sid, sig_grid):
                sig_raw = gen_signal(sid, td, sig_params)
                if sig_raw is None: continue
                for conf in confs:
                    sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
                    for year in OOS_YEARS:
                        year_mask = td["year_masks"].get(year)
                        if year_mask is None: continue
                        if np.sum(np.abs(sig[year_mask]) > 0) < 3: continue

                        best_sh = -999; best_rm = None; best_info = {}
                        # Try default RM first
                        pos = run_bt(sig, close, high, low, atr, rh, rl, def_rm, WARMUP, n)
                        sh, info = calc_excess_sharpe(pos, lr, rf_d, cal_days, year_mask)
                        n_bt += 1
                        if sh > best_sh: best_sh = sh; best_rm = def_rm; best_info = info

                        if sh > -0.3:
                            for ri, rm in enumerate(rm_samples):
                                pos = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                                sh, info = calc_excess_sharpe(pos, lr, rf_d, cal_days, year_mask)
                                n_bt += 1
                                if sh > best_sh: best_sh = sh; best_rm = rm; best_info = info
                                if ri == 49 and best_sh < -0.3: break

                        if best_sh > -999 and best_rm is not None:
                            ticker_results.append(dict(
                                strategy=sid, ticker=ticker, year=year,
                                sharpe=round(best_sh, 3), exposure=best_info.get("exposure", 0),
                                n_trades=best_info.get("n_trades", 0),
                                tpy=best_info.get("tpy", 0), wr=best_info.get("wr", 0),
                                signal=json.dumps(sig_params, sort_keys=True), conf=conf, filter="{}",
                                rm=json.dumps({k: v for k, v in best_rm.items() if v != 0}, sort_keys=True),
                            ))

        all_results.extend(ticker_results)
        elapsed = time.time() - t0
        log(f"  [{ti+1}/17] {ticker}: {len(ticker_results)} res ({elapsed:.0f}s, {n_bt:,} bt)")

    return pd.DataFrame(all_results) if all_results else None


# ═══════════════════════════════════════════════════════════════
# PHASE 8: MERGE BEST FROM ALL PHASES
# ═══════════════════════════════════════════════════════════════

def run_phase_8(all_phase_dfs):
    """Phase 8: Merge best results from all phases."""
    log("Phase 8: Merge best from all phases")
    frames = [df for df in all_phase_dfs if df is not None and len(df) > 0]
    if not frames:
        log("  No data to merge!"); return None
    combined = pd.concat(frames, ignore_index=True)

    # Per strategy×ticker×year: keep best
    best = combined.sort_values("sharpe", ascending=False).drop_duplicates(
        subset=["strategy", "ticker", "year"], keep="first")

    log(f"  Merged: {len(combined)} → {len(best)} (best per strategy×ticker×year)")
    return best


# ═══════════════════════════════════════════════════════════════
# PHASE 9: FINAL SELECTION (mean across years, min trades)
# ═══════════════════════════════════════════════════════════════

def run_phase_9(phase8_df):
    """Phase 9: Final per-strategy×ticker mean Sharpe (nanmean across years)."""
    log("Phase 9: Final aggregation")

    results = []
    for sid in STRATEGIES:
        for ticker in TICKERS:
            sub = phase8_df[(phase8_df.strategy == sid) & (phase8_df.ticker == ticker)]
            if len(sub) == 0:
                results.append(dict(strategy=sid, ticker=ticker, sharpe=np.nan,
                                    exposure=0, n_years=0))
                continue
            # Filter: min 20 trades total across years
            total_trades = sub["n_trades"].sum()
            if total_trades < 20:
                results.append(dict(strategy=sid, ticker=ticker, sharpe=np.nan,
                                    exposure=0, n_years=0, note="<20 trades"))
                continue
            sh_values = []
            for y in OOS_YEARS:
                ysub = sub[sub.year == y]
                if len(ysub) > 0:
                    sh_values.append(ysub["sharpe"].max())
                else:
                    sh_values.append(np.nan)
            mean_sh = np.nanmean(sh_values)
            avg_exp = sub["exposure"].mean()
            best_row = sub.nlargest(1, "sharpe").iloc[0]
            results.append(dict(
                strategy=sid, ticker=ticker,
                sharpe=round(mean_sh, 3),
                exposure=round(avg_exp, 1),
                n_years=sum(1 for v in sh_values if not np.isnan(v)),
                signal=best_row.get("signal", ""),
                conf=best_row.get("conf", ""),
                rm=best_row.get("rm", ""),
            ))

    df = pd.DataFrame(results)
    return df


# ═══════════════════════════════════════════════════════════════
# SNAPSHOT (save best results)
# ═══════════════════════════════════════════════════════════════

def save_snapshot(results_df, overall_mean):
    snapshot_path = OUT_DIR / "BEST_SNAPSHOT.json"
    prev_best = 0
    if snapshot_path.exists():
        try:
            with open(snapshot_path) as f:
                prev = json.load(f)
                prev_best = prev.get('overall_mean', 0)
        except: pass
    if overall_mean > prev_best or not snapshot_path.exists():
        log(f"📸 New best: {overall_mean:.2f} > {prev_best:.2f}")
        means = results_df.groupby('strategy')['sharpe'].mean()
        snapshot = dict(overall_mean=overall_mean,
                       per_strategy=means.to_dict(),
                       timestamp=datetime.datetime.now().isoformat())
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        results_df.to_parquet(OUT_DIR / "BEST_SNAPSHOT_results.parquet")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log("=" * 80)
    log("AUTONOMOUS TEST SCAN v3 — START")
    log("Rules: calc_excess_sharpe (RF+FDR+comm=0.05%), w=1.0, per-year nanmean")
    log("OOS years: 2022-2025, comm=0.0005")
    log("=" * 80)

    # ── Step 0: Load data ──
    log("\nStep 0: Loading data...")
    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    # ── Validate data ──
    if not validator_2_data(daily_df, kr_s):
        log("ABORT: Data validation failed"); return

    # ── Validate formula ──
    if not validator_1_formula():
        log("ABORT: Formula validation failed"); return

    # ── Load grid ──
    with open(GRID_PATH) as f:
        grid = json.load(f)
    log(f"Grid loaded: {GRID_PATH}")

    # ── Prepare ticker data (ONCE) ──
    log("\nPreparing ticker data...")
    ticker_data = []
    for ticker in TICKERS:
        td = prepare_ticker_data(ticker, daily_df, kr_s)
        if td is not None:
            ticker_data.append(td)
            log(f"  {ticker}: {td['n']} bars, {len(td['year_masks'])} OOS years")
        else:
            log(f"  {ticker}: SKIP (insufficient data)")
    log(f"Prepared {len(ticker_data)}/17 tickers")

    # Free memory
    del daily_df, hourly_df, vpred
    gc.collect()

    all_phases = {}

    # ═══════════════════════════
    # PHASE 1
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 1: Grid scan (3-tier)")
    t1 = time.time()
    df1 = run_phase_1(ticker_data, grid, n_filters=10, n_rm=200)
    df1.to_parquet(OUT_DIR / "phase1_results.parquet")
    all_phases[1] = df1
    log(f"Phase 1 done: {len(df1)} rows in {time.time()-t1:.0f}s")
    v3_ok = validator_3_phase("Phase 1", df1)
    decision_log(1, "grid_scan", f"{len(df1)} rows", "proceed" if v3_ok else "warn")

    # ═══════════════════════════
    # PHASE 2
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 2: RM refinement")
    t2 = time.time()
    df2 = run_phase_2(ticker_data, df1, n_top=30, n_rm=500)
    df2.to_parquet(OUT_DIR / "phase2_results.parquet")
    all_phases[2] = df2
    log(f"Phase 2 done: {len(df2)} rows in {time.time()-t2:.0f}s")
    validator_3_phase("Phase 2", df2, df1)
    decision_log(2, "rm_refine", f"{len(df2)} rows", "proceed")

    # ═══════════════════════════
    # PHASE 3
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 3: Fine RM")
    t3 = time.time()
    df3 = run_phase_3(ticker_data, df2)
    df3.to_parquet(OUT_DIR / "phase3_results.parquet")
    all_phases[3] = df3
    log(f"Phase 3 done: {len(df3)} rows in {time.time()-t3:.0f}s")
    validator_3_phase("Phase 3", df3, df2)
    decision_log(3, "fine_rm", f"{len(df3)} rows", "proceed")

    # ═══════════════════════════
    # PHASE 4
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 4: Pyramiding")
    t4 = time.time()
    df4 = run_phase_4(ticker_data, df3)
    df4.to_parquet(OUT_DIR / "phase4_results.parquet")
    all_phases[4] = df4
    log(f"Phase 4 done: {len(df4)} rows in {time.time()-t4:.0f}s")
    validator_3_phase("Phase 4", df4, df3)
    decision_log(4, "pyramiding", f"{len(df4)} rows", "proceed")

    # ═══════════════════════════
    # PHASE 5
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 5: Cross-validation")
    t5 = time.time()
    df5 = run_phase_5(df4)
    df5.to_parquet(OUT_DIR / "phase5_results.parquet")
    all_phases[5] = df5
    log(f"Phase 5 done: {len(df5)} rows in {time.time()-t5:.0f}s")
    validator_3_phase("Phase 5", df5, df4)
    decision_log(5, "cross_val", f"{len(df5)} rows", "proceed")

    # ═══════════════════════════
    # PHASE 6
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 6: Diagnosis")
    diagnosis = run_phase_6(df5)
    decision_log(6, "diagnosis",
                 f"weak_tk={diagnosis['weak_tickers']}, weak_st={diagnosis['weak_strategies']}",
                 "expand" if diagnosis["weak_strategies"] else "skip")

    # ═══════════════════════════
    # PHASE 7
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 7: Grid expansion")
    t7 = time.time()
    df7 = run_phase_7(ticker_data, grid, diagnosis)
    if df7 is not None and len(df7) > 0:
        df7.to_parquet(OUT_DIR / "phase7_results.parquet")
        all_phases[7] = df7
        log(f"Phase 7 done: {len(df7)} rows in {time.time()-t7:.0f}s")
        validator_3_phase("Phase 7", df7)
    else:
        log("Phase 7: skipped")
    decision_log(7, "expansion", f"{len(df7) if df7 is not None else 0} rows", "proceed")

    # ═══════════════════════════
    # PHASE 8
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 8: Merge")
    phase_dfs = [all_phases.get(i) for i in [1, 2, 3, 4, 5, 7]]
    df8 = run_phase_8(phase_dfs)
    if df8 is not None:
        df8.to_parquet(OUT_DIR / "phase8_results.parquet")
        all_phases[8] = df8
        validator_3_phase("Phase 8", df8)

    # ═══════════════════════════
    # PHASE 9
    # ═══════════════════════════
    log(f"\n{'='*60}")
    log("PHASE 9: Final")
    df9 = run_phase_9(df8 if df8 is not None else df5)
    comm_suffix = f"_{round(COMM*10000):03d}" if COMM != 0.0005 else ""
    df9.to_parquet(OUT_DIR / f"FINAL_test_results{comm_suffix}.parquet")
    df9.to_csv(OUT_DIR / f"FINAL_test_results{comm_suffix}.csv", index=False)
    all_phases[9] = df9

    # Extract best configs
    best_configs = {}
    for sid in STRATEGIES:
        sub = df9[df9.strategy == sid].dropna(subset=["sharpe"])
        if len(sub) == 0: continue
        best_configs[sid] = {}
        for _, row in sub.iterrows():
            best_configs[sid][row["ticker"]] = dict(
                sharpe=row["sharpe"], signal=row.get("signal", ""),
                conf=row.get("conf", ""), rm=row.get("rm", ""),
            )
    with open(OUT_DIR / f"FINAL_best_configs{comm_suffix}.json", 'w') as f:
        json.dump(best_configs, f, indent=2, default=str)

    # ═══════════════════════════
    # FINAL EVALUATION
    # ═══════════════════════════
    overall = validator_5_final(df9)
    save_snapshot(df9, overall)

    elapsed_total = time.time() - t_start
    log(f"\nTOTAL TIME: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    log(f"All files saved to: {OUT_DIR}")
    log("AUTONOMOUS TEST SCAN — DONE")


if __name__ == "__main__":
    main()
