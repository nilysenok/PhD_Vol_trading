#!/usr/bin/env python3
"""
autonomous_bcd_scan.py — Phase 10-12: BCD overlay on top of A positions.

Phase 10: BCD grid scan (B adaptive stops, C regime filter, D vol targeting)
Phase 11: BCD analysis (best approach per strategy)
Phase 12: BCD cross-validation (half1 vs half2)

Uses vpred_aligned.parquet (sigma_h1/h5/h22) from hybrid model.
"""
import os, sys, warnings, time, json, random, datetime, gc
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

random.seed(42); np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from strategies_walkforward import (
    TICKERS, WARMUP, load_data, calc_rsi, calc_sma, calc_std, calc_atr, calc_adx,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)
from daily_upper_bound import bt_universal

BASE = SCRIPT_DIR.parent
KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
VPRED_PATH = BASE / "results/final/strategies/data/vpred_aligned.parquet"
PHASE_DIR = BASE / "results/final/strategies/test_phases"
OUT_DIR = PHASE_DIR  # same dir

BPY = 252
COMM = 0.0005
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

LOG_PATH = OUT_DIR / "BCD_PROGRESS_LOG.txt"


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


# ═══════════════════════════════════════════════════════════════
# EXCESS SHARPE (same as Phase 1-9)
# ═══════════════════════════════════════════════════════════════

def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm=COMM):
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    n = len(pm)
    if n < 20: return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exposure = np.abs(pm)
    fdr = (1.0 - exposure) * rf * cd
    daily_total = net + fdr
    daily_rf = rf * cd
    excess = daily_total - daily_rf
    s = np.std(daily_total, ddof=1)
    if s < 1e-12: return 0.0, {}
    sharpe = np.mean(excess) * sqrt(BPY) / s
    exp_pct = np.mean(exposure) * 100
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
    return sharpe, dict(exposure=round(exp_pct, 1), tpy=round(nt / cs, 1),
                        wr=round(nw / nt * 100 if nt > 0 else 0, 1), n_trades=nt)


# ═══════════════════════════════════════════════════════════════
# SIGNAL/BT HELPERS (copied from autonomous_test_scan.py)
# ═══════════════════════════════════════════════════════════════

def _ema(arr, w):
    a = 2.0 / (w + 1); out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def _dema(arr, w):
    e1 = _ema(arr, w); return 2 * e1 - _ema(e1, w)

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

def precompute_indicators(close, high, low, volume, open_arr, windows):
    n = len(close); ind = {}
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
        z[np.isnan(ma)] = np.nan; ind["z_atr"][key] = z
    ind["z_std"] = {}
    for key in ind["ma"]:
        ma = ind["ma"][key]; std = ind["std"].get(key)
        if std is not None:
            z = (close - ma) / np.where((std > 1e-12) & ~np.isnan(std), std, 1.0)
            z[np.isnan(ma)] = np.nan; ind["z_std"][key] = z
    ind["volume"] = volume
    return ind

def gen_s1(z, threshold, warmup):
    sig = np.zeros(len(z), dtype=np.int8); v = ~np.isnan(z)
    sig[v & (z < -threshold)] = 1; sig[v & (z > threshold)] = -1; sig[:warmup] = 0; return sig
def gen_s2(z, bs, warmup):
    sig = np.zeros(len(z), dtype=np.int8); v = ~np.isnan(z)
    sig[v & (z < -bs)] = 1; sig[v & (z > bs)] = -1; sig[:warmup] = 0; return sig
def gen_s3(close, dc_cache, w, warmup):
    if w not in dc_cache: return None
    hc, lc = dc_cache[w]
    ph = np.empty_like(hc); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty_like(lc); pl[0] = np.nan; pl[1:] = lc[:-1]
    v = ~np.isnan(ph); sig = np.zeros(len(close), dtype=np.int8)
    sig[v & (close > ph)] = 1; sig[v & (close < pl)] = -1; sig[:warmup] = 0; return sig
def gen_s4(close, st_cache, ap, mult, warmup):
    key = (ap, mult)
    if key not in st_cache: return None
    st, _ = st_cache[key]; n = len(close)
    sp = np.empty_like(st); sp[0] = np.nan; sp[1:] = st[:-1]
    cp = np.empty_like(close); cp[0] = close[0]; cp[1:] = close[:-1]
    v = ~np.isnan(st) & ~np.isnan(sp); sig = np.zeros(n, dtype=np.int8)
    sig[v & (close > st) & (cp <= sp)] = 1; sig[v & (close < st) & (cp >= sp)] = -1
    sig[:warmup] = 0; return sig
def gen_s5(close, pivot_data, pt, el, warmup):
    if pt not in pivot_data: return None
    P, S1r, R1r, S2r, R2r, S3r, R3r = pivot_data[pt]
    if el == 2: S, R = S2r, R2r
    elif el == 3: S, R = S3r, R3r
    else: S, R = S1r, R1r
    sig = np.zeros(len(close), dtype=np.int8)
    sig[~np.isnan(S) & (close < S)] = 1; sig[~np.isnan(R) & (close > R)] = -1
    sig[:warmup] = 0; return sig
def gen_s6(close, vwap_cache, vw, dm, warmup):
    if vw not in vwap_cache: return None
    vwap, dev = vwap_cache[vw]; v = ~np.isnan(vwap)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[v & (close < vwap - dm * dev)] = 1; sig[v & (close > vwap + dm * dev)] = -1
    sig[:warmup] = 0; return sig

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

def run_bt(sig, close, high, low, atr, rh, rl, rm, warmup, end_idx):
    return bt_universal(sig, close, high, low, atr, rh, rl,
                        rm.get("sl_p", 0), rm.get("tp_p", 0), rm.get("sl_a", 0),
                        rm.get("tp_a", 0), rm.get("trail", 0), rm.get("mh", 0),
                        rm.get("be", 0), warmup, end_idx,
                        chandelier_atr=rm.get("chand", 0.0),
                        rr_ratio=rm.get("rr", 0.0))

def gen_signal(sid, td, params):
    close = td["close"]; ind = td["ind"]
    if sid == "S1":
        z = ind["z_atr"].get((params["ma"], params["w"]))
        return gen_s1(z, params["thresh"], WARMUP) if z is not None else None
    elif sid == "S2":
        z = ind["z_std"].get((params["ma"], params["w"]))
        return gen_s2(z, params["bs"], WARMUP) if z is not None else None
    elif sid == "S3": return gen_s3(close, td["dc_cache"], params["w"], WARMUP)
    elif sid == "S4": return gen_s4(close, td["st_cache"], params["ap"], params["mult"], WARMUP)
    elif sid == "S5": return gen_s5(close, td["pivot_data"], params["pt"], params["el"], WARMUP)
    elif sid == "S6": return gen_s6(close, td["vwap_cache"], params["vw"], params["dm"], WARMUP)
    return None


# ═══════════════════════════════════════════════════════════════
# BCD IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

def apply_B(a_pos, close, atr, vol_pred, vol_median,
            k_sl, ratio, base_hold, hold_gamma, be_trigger):
    """Approach B: Adaptive stops using vpred."""
    n = len(close)
    b_pos = np.zeros(n, dtype=np.float64)
    in_trade = False; entry_price = 0.0; entry_bar = 0
    direction = 0; current_sl = 0; best_pnl = 0; max_hold = 0

    for i in range(n):
        if not in_trade:
            if abs(a_pos[i]) > 1e-8:
                in_trade = True
                direction = 1 if a_pos[i] > 0 else -1
                entry_price = close[i]
                entry_bar = i
                best_pnl = 0.0
                b_pos[i] = float(direction)
                vp = vol_pred[i] if not np.isnan(vol_pred[i]) else vol_median
                current_sl = k_sl * vp * entry_price
                max_hold = max(1, int(base_hold * (vol_median / max(vp, 1e-8)) ** hold_gamma))
            else:
                b_pos[i] = 0.0
        else:
            pnl = (close[i] - entry_price) * direction
            best_pnl = max(best_pnl, pnl)
            bars_held = i - entry_bar
            exit_now = False

            if pnl < -current_sl: exit_now = True
            if ratio > 0 and pnl > current_sl * ratio: exit_now = True
            if max_hold > 0 and bars_held >= max_hold: exit_now = True
            if be_trigger > 0:
                vp = vol_pred[i] if not np.isnan(vol_pred[i]) else vol_median
                if best_pnl > be_trigger * vp * entry_price:
                    current_sl = max(0, current_sl * 0.1)
            if abs(a_pos[i]) < 1e-8: exit_now = True
            if abs(a_pos[i]) > 1e-8 and (1 if a_pos[i] > 0 else -1) != direction: exit_now = True

            if exit_now:
                in_trade = False; b_pos[i] = 0.0
            else:
                b_pos[i] = float(direction)
    return b_pos


def apply_C(a_pos, vol_pred, threshold, lookback, strategy_type,
            direction_filter, sigma_h22, sigma_h1, term_structure, hysteresis):
    """Approach C: Regime filter using vpred percentile rank."""
    n = len(a_pos)
    c_pos = a_pos.copy().astype(np.float64)

    # Expanding percentile rank
    pct_rank = np.full(n, 0.5)
    for i in range(lookback, n):
        window = vol_pred[max(0, i - lookback):i]
        valid = window[~np.isnan(window)]
        if len(valid) > 10:
            pct_rank[i] = np.mean(valid < vol_pred[i])

    vol_mask = np.ones(n, dtype=bool)
    if strategy_type == 'trend':
        vol_mask = pct_rank > threshold
    elif strategy_type == 'contrarian':
        vol_mask = pct_rank < (1.0 - threshold)
    elif strategy_type == 'range':
        vol_mask = (pct_rank > (0.5 - threshold / 2)) & (pct_rank < (0.5 + threshold / 2))

    if direction_filter == 'rise':
        vc = np.diff(vol_pred, prepend=vol_pred[0]); vol_mask &= vc > 0
    elif direction_filter == 'fall':
        vc = np.diff(vol_pred, prepend=vol_pred[0]); vol_mask &= vc < 0

    if term_structure and sigma_h22 is not None and sigma_h1 is not None:
        ts = sigma_h22 / np.maximum(sigma_h1, 1e-8)
        if strategy_type == 'trend': vol_mask &= ts > 1.0
        elif strategy_type == 'contrarian': vol_mask &= ts < 1.0

    if hysteresis > 0:
        last_sw = -hysteresis; state = True
        for i in range(n):
            if vol_mask[i] != state and (i - last_sw) >= hysteresis:
                state = vol_mask[i]; last_sw = i
            vol_mask[i] = state

    c_pos[~vol_mask] = 0.0
    return c_pos


def apply_D(a_pos, vol_pred, sigma_target, gamma, smooth_span, vol_floor, vol_cap):
    """Approach D: Continuous vol targeting."""
    n = len(a_pos)
    vp = vol_pred.copy()
    if smooth_span > 0:
        alpha = 2.0 / (smooth_span + 1)
        for i in range(1, n):
            if not np.isnan(vp[i]) and not np.isnan(vp[i - 1]):
                vp[i] = alpha * vp[i] + (1 - alpha) * vp[i - 1]
    scale = (sigma_target / np.maximum(vp, 1e-8)) ** gamma
    scale = np.clip(scale, vol_floor, vol_cap)
    scale[np.isnan(vp)] = 1.0
    d_pos = a_pos.astype(np.float64) * scale
    return d_pos


# ═══════════════════════════════════════════════════════════════
# PREPARE DATA
# ═══════════════════════════════════════════════════════════════

def prepare_ticker_data(ticker, daily_df, kr_s, vpred_df):
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

    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS)
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}

    # Align vpred to daily dates
    vp_tk = vpred_df[vpred_df.ticker == ticker].set_index("date").sort_index()
    sigma = {}
    for h in ["h1", "h5", "h22"]:
        col = f"sigma_{h}"
        s = vp_tk[col].reindex(dates, method="ffill").values
        sigma[h] = s

    year_masks = {}
    for y in OOS_YEARS:
        try:
            ys, ye = _year_bounds(dates, y)
            if ys >= ye or ye <= WARMUP: continue
            m = np.zeros(n, dtype=bool); m[ys:ye] = True
            year_masks[y] = m
        except: pass

    return dict(
        ticker=ticker, close=close, high=high, low=low, open_arr=open_arr,
        volume=volume, dates=dates, n=n, lr=lr, cal_days=cal_days, rf_d=rf_d,
        ind=ind, atr=atr, rh=rh, rl=rl,
        dc_cache=dc_cache, st_cache=st_cache, vwap_cache=vwap_cache,
        pivot_data=pivot_data, year_masks=year_masks, sigma=sigma,
    )


# ═══════════════════════════════════════════════════════════════
# PHASE 10: BCD GRID SCAN
# ═══════════════════════════════════════════════════════════════

def run_phase_10(ticker_data, best_configs):
    log("Phase 10: BCD grid scan")
    all_results = []
    n_bt = 0

    STRATEGY_TYPE = {"S1": "contrarian", "S2": "contrarian", "S3": "trend",
                     "S4": "trend", "S5": "range", "S6": "range"}

    for ti, td in enumerate(ticker_data):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        ind = td["ind"]; rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
        volume = td["volume"]; lr = td["lr"]; rf_d = td["rf_d"]
        cal_days = td["cal_days"]; n = td["n"]
        sigma = td["sigma"]

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            stype = STRATEGY_TYPE[sid]

            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None: continue

            sig_params = json.loads(cfg["signal"])
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"])

            sig_raw = gen_signal(sid, td, sig_params)
            if sig_raw is None: continue
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
            a_pos = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)

            for year in OOS_YEARS:
                year_mask = td["year_masks"].get(year)
                if year_mask is None: continue

                a_sh, a_info = calc_excess_sharpe(a_pos, lr, rf_d, cal_days, year_mask)
                n_bt += 1

                # ═══ APPROACH B ═══
                for horizon in ["h1", "h5", "h22"]:
                    vol_pred = sigma[horizon]
                    vol_med = np.nanmedian(vol_pred[~np.isnan(vol_pred)])
                    best_b_sh = -999; best_b_params = {}

                    for _ in range(300):
                        k_sl = random.choice([0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
                        ratio = random.choice([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
                        bh = random.choice([3, 5, 7, 10, 15, 20, 30, 50])
                        hg = random.choice([0.25, 0.5, 1.0, 1.5])
                        be = random.choice([0, 0.5, 1.0, 1.5, 2.0, 3.0])

                        b_pos = apply_B(a_pos, close, atr, vol_pred, vol_med,
                                        k_sl, ratio, bh, hg, be)
                        b_sh, b_info = calc_excess_sharpe(b_pos, lr, rf_d, cal_days, year_mask)
                        n_bt += 1
                        if b_sh > best_b_sh:
                            best_b_sh = b_sh
                            best_b_params = dict(k_sl=k_sl, ratio=ratio, base_hold=bh,
                                                 hold_gamma=hg, be=be, horizon=horizon)

                    all_results.append(dict(
                        strategy=sid, ticker=ticker, year=year,
                        approach=f"B_{horizon}", a_sharpe=round(a_sh, 3),
                        bcd_sharpe=round(best_b_sh, 3),
                        delta=round(best_b_sh - a_sh, 3),
                        params=json.dumps(best_b_params, sort_keys=True),
                    ))

                # ═══ APPROACH C ═══
                for horizon in ["h1", "h5", "h22"]:
                    vol_pred = sigma[horizon]
                    best_c_sh = -999; best_c_params = {}

                    for _ in range(300):
                        th = random.choice([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
                        lb = random.choice([63, 126, 252, 504])
                        dirn = random.choice(["off", "rise", "fall"])
                        ts = random.choice([False, True])
                        hyst = random.choice([0, 5, 10, 15])

                        c_pos = apply_C(a_pos, vol_pred, th, lb, stype,
                                        dirn, sigma["h22"], sigma["h1"], ts, hyst)
                        c_sh, _ = calc_excess_sharpe(c_pos, lr, rf_d, cal_days, year_mask)
                        n_bt += 1
                        if c_sh > best_c_sh:
                            best_c_sh = c_sh
                            best_c_params = dict(threshold=th, lookback=lb, direction=dirn,
                                                 term_str=ts, hysteresis=hyst, horizon=horizon)

                    all_results.append(dict(
                        strategy=sid, ticker=ticker, year=year,
                        approach=f"C_{horizon}", a_sharpe=round(a_sh, 3),
                        bcd_sharpe=round(best_c_sh, 3),
                        delta=round(best_c_sh - a_sh, 3),
                        params=json.dumps(best_c_params, sort_keys=True, default=str),
                    ))

                # ═══ APPROACH D ═══
                for horizon in ["h1", "h5", "h22"]:
                    vol_pred = sigma[horizon]
                    best_d_sh = -999; best_d_params = {}

                    for _ in range(300):
                        st = random.choice([0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30])
                        gm = random.choice([0.25, 0.5, 1.0, 1.5, 2.0])
                        sm = random.choice([0, 5, 10, 20])
                        fl = random.choice([0.1, 0.2, 0.3])
                        cp = random.choice([1.5, 2.0, 3.0])

                        d_pos = apply_D(a_pos, vol_pred, st, gm, sm, fl, cp)
                        d_sh, _ = calc_excess_sharpe(d_pos, lr, rf_d, cal_days, year_mask)
                        n_bt += 1
                        if d_sh > best_d_sh:
                            best_d_sh = d_sh
                            best_d_params = dict(sigma_target=st, gamma=gm, smooth=sm,
                                                 floor=fl, cap=cp, horizon=horizon)

                    all_results.append(dict(
                        strategy=sid, ticker=ticker, year=year,
                        approach=f"D_{horizon}", a_sharpe=round(a_sh, 3),
                        bcd_sharpe=round(best_d_sh, 3),
                        delta=round(best_d_sh - a_sh, 3),
                        params=json.dumps(best_d_params, sort_keys=True),
                    ))

        elapsed = time.time() - t0
        log(f"  [{ti+1}/17] {ticker}: {elapsed:.0f}s, {n_bt:,} bt")

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════
# PHASE 11: BCD ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_phase_11(phase10_df):
    log("Phase 11: BCD analysis")
    summary = []
    for sid in STRATEGIES:
        sub = phase10_df[phase10_df.strategy == sid]
        for approach in sorted(sub["approach"].unique()):
            asub = sub[sub.approach == approach]
            summary.append(dict(
                strategy=sid, approach=approach,
                mean_delta=round(asub["delta"].mean(), 3),
                median_delta=round(asub["delta"].median(), 3),
                pct_improved=round((asub["delta"] > 0).mean() * 100, 1),
                mean_a=round(asub["a_sharpe"].mean(), 3),
                mean_bcd=round(asub["bcd_sharpe"].mean(), 3),
            ))
    df = pd.DataFrame(summary)

    best_bcd = {}
    for sid in STRATEGIES:
        sub = df[df.strategy == sid]
        if len(sub) == 0: continue
        best = sub.loc[sub["mean_delta"].idxmax()]
        best_bcd[sid] = dict(
            best_approach=best["approach"],
            mean_delta=best["mean_delta"],
            pct_improved=best["pct_improved"],
            mean_bcd=best["mean_bcd"],
        )
        log(f"  {sid}: best={best['approach']}, Δ={best['mean_delta']:+.3f}, "
            f"improved={best['pct_improved']:.0f}%")

    return df, best_bcd


# ═══════════════════════════════════════════════════════════════
# PHASE 12: BCD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════

def run_phase_12(phase10_df, best_bcd):
    log("Phase 12: BCD cross-validation")
    results = []
    for sid in STRATEGIES:
        ba = best_bcd.get(sid, {}).get("best_approach")
        if ba is None: continue
        sub = phase10_df[(phase10_df.strategy == sid) & (phase10_df.approach == ba)]
        for ticker in sub["ticker"].unique():
            tk = sub[sub.ticker == ticker]
            h1 = tk[tk.year.isin([2022, 2023])]["delta"].mean()
            h2 = tk[tk.year.isin([2024, 2025])]["delta"].mean()
            robust = (h1 > 0) and (h2 > 0)
            results.append(dict(strategy=sid, ticker=ticker, approach=ba,
                                delta_half1=round(h1, 3), delta_half2=round(h2, 3),
                                robust=robust))

    df = pd.DataFrame(results)
    for sid in STRATEGIES:
        sub = df[df.strategy == sid]
        n_robust = sub["robust"].sum()
        total = len(sub)
        md = sub[["delta_half1", "delta_half2"]].mean().mean()
        log(f"  {sid}: {n_robust}/{total} robust, mean Δ={md:+.3f}")

    return df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log("=" * 80)
    log("AUTONOMOUS BCD SCAN — Phase 10-12")
    log("Uses vpred_aligned.parquet for B/C/D overlays")
    log("=" * 80)

    # Load data
    log("Loading data...")
    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    vpred_df = pd.read_parquet(VPRED_PATH)
    vpred_df["date"] = pd.to_datetime(vpred_df["date"])
    warmup_numba()

    # Load best A configs
    with open(PHASE_DIR / "FINAL_best_configs.json") as f:
        best_configs = json.load(f)
    log(f"Loaded {sum(len(v) for v in best_configs.values())} A configs")

    # Prepare ticker data
    log("Preparing ticker data...")
    ticker_data = []
    for ticker in TICKERS:
        td = prepare_ticker_data(ticker, daily_df, kr_s, vpred_df)
        if td is not None:
            ticker_data.append(td)
    log(f"Prepared {len(ticker_data)}/17 tickers")

    del daily_df, hourly_df, vpred
    gc.collect()

    # ── Phase 10 ──
    log(f"\n{'='*60}")
    t10 = time.time()
    df10 = run_phase_10(ticker_data, best_configs)
    df10.to_parquet(OUT_DIR / "phase10_bcd_results.parquet")
    log(f"Phase 10 done: {len(df10)} rows in {time.time()-t10:.0f}s")

    # ── Phase 11 ──
    log(f"\n{'='*60}")
    df11, best_bcd = run_phase_11(df10)
    df11.to_csv(OUT_DIR / "phase11_bcd_analysis.csv", index=False)

    # ── Phase 12 ──
    log(f"\n{'='*60}")
    df12 = run_phase_12(df10, best_bcd)
    df12.to_csv(OUT_DIR / "phase12_bcd_crossval.csv", index=False)

    # Save BCD configs
    with open(OUT_DIR / "BCD_best_configs.json", 'w') as f:
        json.dump(best_bcd, f, indent=2, default=str)

    # ── Final report ──
    log(f"\n{'='*80}")
    log("BCD FINAL REPORT")
    log(f"{'='*80}")
    any_positive = False
    for sid in STRATEGIES:
        v = best_bcd.get(sid, {})
        ba = v.get("best_approach", "N/A")
        md = v.get("mean_delta", 0)
        pi = v.get("pct_improved", 0)
        status = "✅" if md > 0 else "❌"
        if md > 0: any_positive = True
        log(f"  {status} {sid}: {ba}, Δ={md:+.3f}, improved={pi:.0f}%")

    if any_positive:
        n_pos = sum(1 for v in best_bcd.values() if v.get("mean_delta", 0) > 0)
        log(f"\n  BCD > A for {n_pos}/6 strategies")
    else:
        log("\n  ⚠️ BCD ≤ A for all strategies")

    elapsed = time.time() - t_start
    log(f"\nTOTAL TIME: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("BCD SCAN — DONE")


if __name__ == "__main__":
    main()
