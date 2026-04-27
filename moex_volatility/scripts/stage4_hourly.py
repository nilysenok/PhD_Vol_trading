#!/usr/bin/env python3
"""
stage4_hourly.py — COMPLETE Hourly Pipeline: Test A -> Test B/C/D -> Walk-Forward A/B/C/D.

Hourly data: ~457K rows, 17 tickers, 2014-2026, ~7-10 bars/day (MOEX 10:00-18:50).
Signals generated on hourly bars. Positions = hourly (per-bar).
For Sharpe: aggregate to DAILY returns first, then calc_excess_sharpe.

Commission levels: 0.0004 (institutional), 0.0005 (standard) — PER SIDE.
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
from numba import njit

random.seed(42); np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from archivist import archive, check_stage_inputs, agent_hourly_adapter

from strategies_walkforward import (
    TICKERS, load_data, calc_rsi, calc_sma, calc_std, calc_atr, calc_adx,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, warmup_numba,
    compute_base, precompute_sma_cache, precompute_session_vwap,
    compute_daily_pivots_for_hourly, build_daily_trend, align_daily_to_hourly,
    _year_bounds, _compute_nhours_per_day,
    SIGNAL_GRIDS_HOURLY, RM_GRIDS_V3, TREND_RM_GRID,
    CATEGORY, WARMUP_HOURLY,
    gen_s1_signals, gen_s2_signals, gen_s3_signals, gen_s4_signals,
    gen_s5_signals, gen_s6_signals,
    dispatch_signals, dispatch_bt_atr_v3, dispatch_bt_vpred_v3,
    expand_grid, calc_sharpe_comm, _quick_exposure_trades,
    bt_contrarian_v3, bt_trend_v3, bt_range_v3, bt_range_split_v3,
    bt_contrarian_vpred_v3, bt_trend_vpred_v3, bt_range_vpred_v3,
    bt_range_split_vpred_v3,
    _apply_hysteresis, _build_c_mask_v3, _compute_d_scale_v3,
    B_GRIDS_V3, C_THRESHOLDS, C_RANGE_BANDS, C_LOOKBACKS, C_HORIZONS,
    C_DIRECTION, C_TERM_FILTER, C_HYSTERESIS,
    C_MIN_EXPOSURE, C_MIN_TRADES_YR, C_FALLBACK_EXPOSURE, C_FALLBACK_TRADES_YR,
    D_TARGET_VOLS, D_MAX_LEVS, D_GAMMA, D_VOL_FLOORS, D_VOL_CAPS,
    D_SMOOTH, D_HORIZONS, D_INV_LOOKBACKS,
    _extract_trades_from_pos,
)
from daily_upper_bound import bt_universal

# ═══════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

BASE = SCRIPT_DIR.parent
KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
VPRED_PATH = BASE / "results/final/strategies/data/vpred_aligned.parquet"
GRID_PATH = BASE / "results/final/strategies/walkforward_v4/narrowed_grid_v2.json"
CONFIGS_PATH = BASE / "results/final/strategies/test_phases/FINAL_best_configs.json"
OUT_DIR = BASE / "results/final/strategies/hourly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BPY = 252
WARMUP_H = WARMUP_HOURLY  # 1800 = 200 days * 9 bars/day
OOS_YEARS = [2022, 2023, 2024, 2025]
A_TEST_YEARS = list(range(2020, 2027))
BCD_TEST_YEARS = list(range(2022, 2027))
IS_TREND = {"S1": False, "S2": False, "S3": True, "S4": True, "S5": False, "S6": False}
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
STRATEGY_TYPE = {"S1": "contrarian", "S2": "contrarian", "S3": "trend",
                 "S4": "trend", "S5": "range", "S6": "range"}
COMM_HOURLY = {"inst": 0.0004, "std": 0.0005}

# BCD grids (coarse — from chapter4_master.py)
B_GRID = dict(k_sl=[0.5, 1.0, 2.0, 3.0], ratio=[0.5, 1.5, 3.0],
              base_hold=[5, 15, 50], hold_gamma=[0.25, 1.0], be_trigger=[0, 1.5])
C_GRID = dict(threshold=[0.25, 0.55, 0.85], lookback=[63, 252],
              direction=['off', 'rise', 'fall'], term_str=[False, True], hysteresis=[0, 10])
D_GRID = dict(sigma_target=[0.05, 0.10, 0.15, 0.20, 0.30], gamma=[0.25, 0.5, 1.0, 2.0],
              smooth=[0, 10], vol_floor=[0.1, 0.3], vol_cap=[1.5, 2.0, 3.0])
HORIZONS = ['h1', 'h5', 'h22']

LOG_PATH = BASE / "results/final/strategies/MASTER_PROGRESS_LOG.txt"


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] HOURLY: {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


# ═══════════════════════════════════════════════════════════════
# FORMULAS (IMMUTABLE)
# ═══════════════════════════════════════════════════════════════

def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm):
    """Excess Sharpe on daily-aggregated returns."""
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
    sharpe = np.mean(excess) * sqrt(BPY) / s
    exp_pct = np.mean(exposure) * 100
    nt = 0; nw = 0; in_t = False; tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t:
            in_t = True; tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1
                if tr > 0: nw += 1
                in_t = False
    cs = max(n / BPY, 0.01)
    return sharpe, dict(exposure=round(exp_pct, 1), tpy=round(nt / cs, 1),
                        wr=round(nw / nt * 100 if nt > 0 else 0, 1), n_trades=nt)


def trading_sharpe(pos, log_ret, mask, comm):
    pm = pos[mask]; lr = log_ret[mask]
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    s = np.std(net, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(net) * sqrt(BPY) / s


# ═══════════════════════════════════════════════════════════════
# HOURLY → DAILY AGGREGATION
# ═══════════════════════════════════════════════════════════════

def hourly_to_daily_agg(hourly_pos, hourly_log_ret, hourly_dates, comm):
    """Aggregate hourly positions to daily returns + exposure."""
    dates_day = hourly_dates.normalize()
    unique_days = sorted(dates_day.unique())
    daily_dates = []
    daily_net = []
    daily_exp = []
    daily_pos_mean = []
    daily_gross_ret = []
    for day in unique_days:
        mask = dates_day == day
        pos_d = hourly_pos[mask]
        lr_d = hourly_log_ret[mask]
        if len(pos_d) == 0:
            continue
        dp = np.diff(pos_d, prepend=0.0)
        net = np.sum(pos_d * lr_d - np.abs(dp) * comm)
        gross = np.sum(pos_d * lr_d)
        exp = np.mean(np.abs(pos_d))
        pm = np.mean(pos_d)
        daily_dates.append(day)
        daily_net.append(net)
        daily_exp.append(exp)
        daily_pos_mean.append(pm)
        daily_gross_ret.append(gross)
    return (pd.DatetimeIndex(daily_dates),
            np.array(daily_net, dtype=np.float64),
            np.array(daily_exp, dtype=np.float64),
            np.array(daily_pos_mean, dtype=np.float64),
            np.array(daily_gross_ret, dtype=np.float64))


def calc_excess_sharpe_hourly(hourly_pos, hourly_log_ret, hourly_dates,
                               kr_s, test_years, comm):
    """Excess Sharpe for hourly strategy: aggregate to daily, then calc."""
    dates, daily_net, daily_exp, _, _ = hourly_to_daily_agg(
        hourly_pos, hourly_log_ret, hourly_dates, comm)
    if len(dates) < 20:
        return -999.0, {}
    # Align RF and cal_days
    rf_aligned = kr_s.reindex(dates, method="ffill").values / 365.0
    rf_aligned = np.where(np.isfinite(rf_aligned), rf_aligned, 0.15 / 365.0)
    cal_days = np.ones(len(dates))
    for i in range(1, len(dates)):
        cal_days[i] = (dates[i] - dates[i - 1]).days
    # Year mask
    years_arr = dates.year
    mask = np.zeros(len(dates), dtype=bool)
    for y in test_years:
        mask |= (years_arr == y)
    if mask.sum() < 20:
        return -999.0, {}
    # FDR
    fdr = (1.0 - daily_exp) * rf_aligned * cal_days
    daily_total = daily_net + fdr
    excess = daily_total - rf_aligned * cal_days
    dt_m = daily_total[mask]; ex_m = excess[mask]
    s = np.std(dt_m, ddof=1)
    if s < 1e-12:
        return 0.0, {}
    sharpe = np.mean(ex_m) * sqrt(BPY) / s
    exp_pct = np.mean(daily_exp[mask]) * 100
    return sharpe, {"exposure": round(exp_pct, 1)}


def calc_excess_sharpe_hourly_year(hourly_pos, hourly_log_ret, hourly_dates,
                                    kr_s, year, comm):
    """Excess Sharpe for a single OOS year."""
    return calc_excess_sharpe_hourly(hourly_pos, hourly_log_ret, hourly_dates,
                                      kr_s, [year], comm)


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION (from autonomous_test_scan.py / chapter4_master.py)
# ═══════════════════════════════════════════════════════════════

MA_TYPES = ["sma", "ema", "dema", "hull", "kama"]
DC_WINDOWS = [3, 5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120]
ST_PERIODS = [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 35, 40, 50, 60, 80]
ST_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
PIVOT_TYPES = ["classic", "fibonacci", "camarilla"]
VWAP_WINDOWS_ALL = [3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100]
WINDOWS_D = [5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]


def _ema(arr, w):
    a = 2.0 / (w + 1); out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


def _dema(arr, w):
    e1 = _ema(arr, w); return 2 * e1 - _ema(e1, w)


def _hull(arr, w):
    h = max(w // 2, 1)
    wh = np.arange(1, h + 1, dtype=np.float64)
    wf = np.arange(1, w + 1, dtype=np.float64)
    wma_h = np.full_like(arr, np.nan); wma_f = np.full_like(arr, np.nan)
    for i in range(h - 1, len(arr)):
        wma_h[i] = np.dot(arr[i - h + 1:i + 1], wh) / wh.sum()
    for i in range(w - 1, len(arr)):
        wma_f[i] = np.dot(arr[i - w + 1:i + 1], wf) / wf.sum()
    d = 2 * wma_h - wma_f
    sw = max(int(sqrt(w)), 1); ws = np.arange(1, sw + 1, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    for i in range(sw - 1, len(d)):
        if not np.isnan(d[i - sw + 1]):
            out[i] = np.dot(d[i - sw + 1:i + 1], ws) / ws.sum()
    return out


def _kama(arr, w):
    fc = 2.0 / 3; sc = 2.0 / 31
    out = np.full_like(arr, np.nan); out[w - 1] = arr[w - 1]
    for i in range(w, len(arr)):
        d = abs(arr[i] - arr[i - w])
        v = np.sum(np.abs(np.diff(arr[i - w:i + 1])))
        er = d / v if v > 1e-12 else 0
        s2 = (er * (fc - sc) + sc) ** 2
        out[i] = out[i - 1] + s2 * (arr[i] - out[i - 1])
    return out


def get_ma(arr, w, mt):
    if mt == "sma": return calc_sma(arr, w)
    elif mt == "ema": return _ema(arr, w)
    elif mt == "dema": return _dema(arr, w)
    elif mt == "hull": return _hull(arr, w)
    elif mt == "kama": return _kama(arr, w)
    return calc_sma(arr, w)


def precompute_indicators(close, high, low, volume, open_arr, windows):
    """Precompute indicators for signal generation (chapter4_master style)."""
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
            except:
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


def gen_s1(z, th, wu):
    s = np.zeros(len(z), dtype=np.int8)
    v = ~np.isnan(z); s[v & (z < -th)] = 1; s[v & (z > th)] = -1
    s[:wu] = 0; return s


def gen_s2(z, bs, wu):
    s = np.zeros(len(z), dtype=np.int8)
    v = ~np.isnan(z); s[v & (z < -bs)] = 1; s[v & (z > bs)] = -1
    s[:wu] = 0; return s


def gen_s3(close, dc, w, wu):
    if w not in dc: return None
    hc, lc = dc[w]
    ph = np.empty_like(hc); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty_like(lc); pl[0] = np.nan; pl[1:] = lc[:-1]
    v = ~np.isnan(ph)
    s = np.zeros(len(close), dtype=np.int8)
    s[v & (close > ph)] = 1; s[v & (close < pl)] = -1
    s[:wu] = 0; return s


def gen_s4(close, stc, ap, mult, wu):
    k = (ap, mult)
    if k not in stc: return None
    st, _ = stc[k]; n = len(close)
    sp = np.empty_like(st); sp[0] = np.nan; sp[1:] = st[:-1]
    cp = np.empty_like(close); cp[0] = close[0]; cp[1:] = close[:-1]
    v = ~np.isnan(st) & ~np.isnan(sp)
    s = np.zeros(n, dtype=np.int8)
    s[v & (close > st) & (cp <= sp)] = 1; s[v & (close < st) & (cp >= sp)] = -1
    s[:wu] = 0; return s


def gen_s5(close, piv, pt, el, wu):
    if pt not in piv: return None
    P, S1r, R1r, S2r, R2r, S3r, R3r = piv[pt]
    if el == 2: S, R = S2r, R2r
    elif el == 3: S, R = S3r, R3r
    else: S, R = S1r, R1r
    s = np.zeros(len(close), dtype=np.int8)
    s[~np.isnan(S) & (close < S)] = 1; s[~np.isnan(R) & (close > R)] = -1
    s[:wu] = 0; return s


def gen_s6(close, vc, vw, dm, wu):
    if vw not in vc: return None
    vwap, dev = vc[vw]; v = ~np.isnan(vwap)
    s = np.zeros(len(close), dtype=np.int8)
    s[v & (close < vwap - dm * dev)] = 1; s[v & (close > vwap + dm * dev)] = -1
    s[:wu] = 0; return s


def apply_confirmation(sig, conf, rsi, volume, vol_sma20, is_trend):
    if conf == "none": return sig
    out = sig.copy()
    if conf == "rsi30":
        if is_trend:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi < 50))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi > 50))] = 0
        else:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi > 30))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi < 70))] = 0
    elif conf == "rsi20":
        if is_trend:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi < 55))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi > 45))] = 0
        else:
            out[(sig == 1) & ((np.isnan(rsi)) | (rsi > 20))] = 0
            out[(sig == -1) & ((np.isnan(rsi)) | (rsi < 80))] = 0
    elif conf == "volume":
        vs = np.where(np.isnan(vol_sma20) | (vol_sma20 < 1), 1.0, vol_sma20)
        out[volume / vs < 1.0] = 0
    return out


def apply_filters(sig, close, volume, ind, atr_arr, day_of_week, fp, is_trend):
    if not fp: return sig
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
        if dw == 1: mask &= day_of_week != 0
        elif dw == 2: mask &= day_of_week != 4
        elif dw == 3: mask &= (day_of_week != 0) & (day_of_week != 4)
    sig[~mask] = 0
    return sig


def gen_signal_hourly(sid, td, params, warmup):
    """Generate signal for given strategy on hourly data (simple signals)."""
    close = td["close"]; ind = td["ind"]
    if sid == "S1":
        z = ind["z_atr"].get((params["ma"], params["w"]))
        if z is None: return None
        return gen_s1(z, params["thresh"], warmup)
    elif sid == "S2":
        z = ind["z_std"].get((params["ma"], params["w"]))
        if z is None: return None
        return gen_s2(z, params["bs"], warmup)
    elif sid == "S3":
        return gen_s3(close, td["dc_cache"], params["w"], warmup)
    elif sid == "S4":
        return gen_s4(close, td["st_cache"], params["ap"], params["mult"], warmup)
    elif sid == "S5":
        return gen_s5(close, td["pivot_data"], params["pt"], params["el"], warmup)
    elif sid == "S6":
        return gen_s6(close, td["vwap_cache"], params["vw"], params["dm"], warmup)
    return None


def run_bt(sig, close, high, low, atr, rh, rl, rm, warmup, end_idx):
    return bt_universal(sig, close, high, low, atr, rh, rl,
                        rm.get("sl_p", 0), rm.get("tp_p", 0), rm.get("sl_a", 0),
                        rm.get("tp_a", 0), rm.get("trail", 0), rm.get("mh", 0),
                        rm.get("be", 0), warmup, end_idx,
                        chandelier_atr=rm.get("chand", 0.0),
                        rr_ratio=rm.get("rr", 0.0))


def iterate_signal_grid(sid, grid_sig):
    """Yield signal param dicts from narrowed grid."""
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


DEFAULT_RM = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=3.0, trail=0, mh=15, be=0, chand=0, rr=0)
DEFAULT_RM_TREND = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=0, trail=2.0, mh=0, be=1.0, chand=0, rr=0)


# ═══════════════════════════════════════════════════════════════
# BCD IMPLEMENTATIONS (from chapter4_master.py)
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _apply_B_numba(a_pos, close, vol_pred, vol_median,
                   k_sl, ratio, base_hold, hold_gamma, be_trigger):
    n = len(close)
    b_pos = np.zeros(n, dtype=np.float64)
    in_trade = False; entry_price = 0.0; entry_bar = 0
    direction = 0; current_sl = 0.0; best_pnl = 0.0; max_hold = 0
    b_exited = False  # prevents re-entry within same A-trade
    for i in range(n):
        # Reset b_exited when A exits (new A-trade can start)
        if abs(a_pos[i]) < 1e-8:
            b_exited = False
        if not in_trade:
            if not b_exited and abs(a_pos[i]) > 1e-8:
                in_trade = True
                direction = 1 if a_pos[i] > 0 else -1
                entry_price = close[i]; entry_bar = i; best_pnl = 0.0
                b_pos[i] = float(direction)
                vp = vol_pred[i] if not np.isnan(vol_pred[i]) else vol_median
                current_sl = k_sl * vp * entry_price
                max_hold = max(1, int(base_hold * (vol_median / max(vp, 1e-8)) ** hold_gamma))
            else:
                b_pos[i] = 0.0
        else:
            pnl = (close[i] - entry_price) * direction
            if pnl > best_pnl:
                best_pnl = pnl
            exit_now = False
            if pnl < -current_sl:
                exit_now = True
            if ratio > 0 and pnl > current_sl * ratio:
                exit_now = True
            if max_hold > 0 and (i - entry_bar) >= max_hold:
                exit_now = True
            if be_trigger > 0:
                vp = vol_pred[i] if not np.isnan(vol_pred[i]) else vol_median
                if best_pnl > be_trigger * vp * entry_price:
                    current_sl = current_sl * 0.1
            if abs(a_pos[i]) < 1e-8:
                exit_now = True
            elif (1 if a_pos[i] > 0 else -1) != direction:
                exit_now = True
            if exit_now:
                in_trade = False; b_pos[i] = 0.0
                # Mark B-exited if A is still active (don't re-enter this A-trade)
                if abs(a_pos[i]) > 1e-8:
                    b_exited = True
            else:
                b_pos[i] = float(direction)
    return b_pos


def apply_B(a_pos, close, atr, vol_pred, vol_median,
            k_sl, ratio, base_hold, hold_gamma, be_trigger):
    return _apply_B_numba(a_pos.astype(np.float64), close, vol_pred, vol_median,
                          k_sl, ratio, float(base_hold), hold_gamma, be_trigger)


def apply_C(a_pos, vol_pred, threshold, lookback, strategy_type,
            direction_filter, sigma_h22, sigma_h1, term_structure, hysteresis):
    """C: Regime filter."""
    n = len(a_pos); c_pos = a_pos.copy().astype(np.float64)
    pct_rank = np.full(n, 0.5)
    for i in range(lookback, n):
        w = vol_pred[max(0, i - lookback):i]; valid = w[~np.isnan(w)]
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
        if strategy_type == 'trend':
            vol_mask &= ts > 1.0
        elif strategy_type == 'contrarian':
            vol_mask &= ts < 1.0
    if hysteresis > 0:
        last_sw = -hysteresis; state = True
        for i in range(n):
            if vol_mask[i] != state and (i - last_sw) >= hysteresis:
                state = vol_mask[i]; last_sw = i
            vol_mask[i] = state
    c_pos[~vol_mask] = 0.0
    return c_pos


def apply_D(a_pos, vol_pred, sigma_target, gamma, smooth_span, vol_floor, vol_cap):
    """D: Continuous vol targeting."""
    n = len(a_pos); vp = vol_pred.copy()
    if smooth_span > 0:
        alpha = 2.0 / (smooth_span + 1)
        for i in range(1, n):
            if not np.isnan(vp[i]) and not np.isnan(vp[i - 1]):
                vp[i] = alpha * vp[i] + (1 - alpha) * vp[i - 1]
    scale = (sigma_target / np.maximum(vp, 1e-8)) ** gamma
    scale = np.clip(scale, vol_floor, vol_cap)
    scale[np.isnan(vp)] = 1.0
    return a_pos.astype(np.float64) * scale


def grid_combos(grid_dict):
    """Generate all combos from a grid dict."""
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))


# ═══════════════════════════════════════════════════════════════
# PREPARE HOURLY TICKER DATA
# ═══════════════════════════════════════════════════════════════

def prepare_hourly_ticker(ticker, hourly_df, daily_df, kr_s, vpred_df):
    """Prepare all data structures for one ticker on hourly timeframe."""
    hdf = hourly_df[hourly_df["ticker"] == ticker].sort_values("datetime").reset_index(drop=True)
    if len(hdf) < WARMUP_H + 200:
        return None
    close = hdf["close"].values.astype(np.float64)
    high = hdf["high"].values.astype(np.float64)
    low = hdf["low"].values.astype(np.float64)
    open_arr = hdf["open"].values.astype(np.float64)
    volume = hdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(hdf["datetime"].values)
    n = len(close)

    # Log returns
    lr = np.zeros(n)
    lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    # Indicators (chapter4_master style for simple signals)
    ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
    atr = ind["atr14"]; rh = ind["rh"]; rl = ind["rl"]

    # Caches
    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    from strategies_walkforward import calc_supertrend
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS_ALL)

    # Pivot data (daily pivots aligned to hourly)
    pivot_data = {}
    for pv in PIVOT_TYPES:
        P, S1p, R1p, S2p, R2p, S3p, R3p = compute_daily_pivots_for_hourly(
            daily_df, ticker, hdf["datetime"].values, pv)
        pivot_data[pv] = (P, S1p, R1p, S2p, R2p, S3p, R3p)

    # Vpred alignment (daily -> hourly via forward-fill)
    vp_tk = vpred_df[vpred_df.ticker == ticker].set_index("date").sort_index()
    sigma = {}
    hourly_bar_dates = dates.normalize()
    vp_dates = pd.to_datetime(vp_tk.index)
    # Compute nhours per day for v2_sqrtN scaling
    nhours = _compute_nhours_per_day(hdf["datetime"].values)
    nhours_safe = np.maximum(nhours, 1.0)

    for h in ["h1", "h5", "h22"]:
        col = f"sigma_{h}"
        if col in vp_tk.columns:
            vp_vals = vp_tk[col].values
            sigma_arr = np.full(n, np.nan)
            idx = np.searchsorted(vp_dates, hourly_bar_dates, side="right") - 1
            valid = idx >= 0
            sigma_arr[valid] = vp_vals[idx[valid]]
            # v2_sqrtN for hourly
            sigma_arr = sigma_arr / np.sqrt(nhours_safe)
            sigma[h] = sigma_arr
        else:
            sigma[h] = np.full(n, 0.02)

    # Daily-level sigma for BCD (NOT scaled)
    sigma_daily = {}
    for h in ["h1", "h5", "h22"]:
        col = f"sigma_{h}"
        if col in vp_tk.columns:
            vp_vals = vp_tk[col].values
            sigma_arr = np.full(n, np.nan)
            idx = np.searchsorted(vp_dates, hourly_bar_dates, side="right") - 1
            valid = idx >= 0
            sigma_arr[valid] = vp_vals[idx[valid]]
            sigma_daily[h] = sigma_arr
        else:
            sigma_daily[h] = np.full(n, 0.02)

    # Year bounds
    year_masks = {}
    for y in OOS_YEARS:
        try:
            ys, ye = _year_bounds(dates, y)
            if ys >= ye or ye <= WARMUP_H:
                continue
            m = np.zeros(n, dtype=bool); m[ys:ye] = True
            year_masks[y] = m
        except:
            pass

    return dict(
        ticker=ticker, close=close, high=high, low=low, open_arr=open_arr,
        volume=volume, dates=dates, n=n, lr=lr,
        ind=ind, atr=atr, rh=rh, rl=rl,
        dc_cache=dc_cache, st_cache=st_cache, vwap_cache=vwap_cache,
        pivot_data=pivot_data, year_masks=year_masks,
        sigma=sigma, sigma_daily=sigma_daily,
    )


# ═══════════════════════════════════════════════════════════════
# TEST A: Evaluate best configs from autonomous_test_scan on hourly
# ═══════════════════════════════════════════════════════════════

def run_test_A(ticker_data_list, best_configs, kr_s, comm):
    """Test A: Apply daily-optimized configs to hourly data, evaluate per year."""
    log(f"  Test A with comm={comm}")
    results = []
    positions_all = []

    for td in ticker_data_list:
        ticker = td["ticker"]
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        lr = td["lr"]; dates = td["dates"]; n = td["n"]

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None:
                continue

            sig_params = agent_hourly_adapter(cfg)  # Scale windows for hourly
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"])

            # Generate signal on hourly data (with adapted params)
            sig_raw = gen_signal_hourly(sid, td, sig_params, WARMUP_H)
            if sig_raw is None:
                continue

            rsi = td["ind"]["rsi14"]; vol_sma20 = td["ind"]["vol_sma20"]
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, td["volume"], vol_sma20, is_trend)

            # Run backtest on hourly
            pos_h = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP_H, n)

            # Aggregate to daily and evaluate per year
            daily_dates, daily_net, daily_exp, daily_pos, daily_gross = hourly_to_daily_agg(
                pos_h, lr, dates, comm)

            if len(daily_dates) < 20:
                continue

            # Build daily arrays for calc_excess_sharpe
            rf_aligned = kr_s.reindex(daily_dates, method="ffill").values / 365.0
            rf_aligned = np.where(np.isfinite(rf_aligned), rf_aligned, 0.15 / 365.0)
            cal_days = np.ones(len(daily_dates))
            for i in range(1, len(daily_dates)):
                cal_days[i] = (daily_dates[i] - daily_dates[i - 1]).days

            for year in OOS_YEARS:
                year_mask = daily_dates.year == year
                if year_mask.sum() < 20:
                    continue
                sh, info = calc_excess_sharpe(daily_pos, daily_net / np.where(np.abs(daily_pos) > 1e-8, daily_pos, 1.0),
                                              rf_aligned, cal_days, year_mask, 0.0)
                # Recompute properly: use daily_net directly
                pm = daily_pos[year_mask]
                dn = daily_net[year_mask]
                de = daily_exp[year_mask]
                rf_y = rf_aligned[year_mask]
                cd_y = cal_days[year_mask]
                ny = len(pm)
                if ny < 20:
                    continue
                fdr = (1.0 - de) * rf_y * cd_y
                daily_total = dn + fdr
                excess = daily_total - rf_y * cd_y
                s = np.std(daily_total, ddof=1)
                if s < 1e-12:
                    sh = 0.0
                else:
                    sh = np.mean(excess) * sqrt(BPY) / s
                exp_pct = np.mean(de) * 100

                results.append(dict(
                    strategy=sid, ticker=ticker, year=year,
                    sharpe=round(sh, 3), exposure=round(exp_pct, 1)))

                # Collect positions for parquet
                dg = daily_gross[year_mask]
                dp_mask = daily_pos[year_mask]
                dd_mask = daily_dates[year_mask]
                for j in range(len(dd_mask)):
                    positions_all.append(dict(
                        date=dd_mask[j], ticker=ticker, strategy=sid, year=year,
                        position=round(float(dp_mask[j]), 6),
                        daily_gross_return=round(float(dg[j]), 8)))

    return pd.DataFrame(results), positions_all


# ═══════════════════════════════════════════════════════════════
# TEST BCD: Apply BCD overlays to hourly A positions
# ═══════════════════════════════════════════════════════════════

def run_test_BCD(ticker_data_list, best_configs, kr_s, comm, comm_label):
    """Test BCD: deterministic grid scan on hourly data."""
    log(f"  Test BCD with comm={comm} ({comm_label})")
    all_results = {"B": [], "C": [], "D": []}
    positions_all = {"B": [], "C": [], "D": []}
    n_eval = 0

    for ti, td in enumerate(ticker_data_list):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; lr = td["lr"]; dates = td["dates"]; n = td["n"]
        sigma_daily = td["sigma_daily"]  # daily sigma forward-filled to hourly

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            stype = STRATEGY_TYPE[sid]
            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None:
                continue

            sig_params = json.loads(cfg["signal"])
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"])

            sig_raw = gen_signal_hourly(sid, td, sig_params, WARMUP_H)
            if sig_raw is None:
                continue
            rsi = td["ind"]["rsi14"]; vol_sma20 = td["ind"]["vol_sma20"]
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, td["volume"], vol_sma20, is_trend)
            a_pos = run_bt(sig, close, high, low, atr, td["rh"], td["rl"], rm, WARMUP_H, n)

            # Aggregate A to daily for baseline
            daily_dates_a, daily_net_a, daily_exp_a, daily_pos_a, _ = hourly_to_daily_agg(
                a_pos, lr, dates, comm)
            if len(daily_dates_a) < 20:
                continue
            rf_aligned_a = kr_s.reindex(daily_dates_a, method="ffill").values / 365.0
            rf_aligned_a = np.where(np.isfinite(rf_aligned_a), rf_aligned_a, 0.15 / 365.0)
            cal_days_a = np.ones(len(daily_dates_a))
            for i in range(1, len(daily_dates_a)):
                cal_days_a[i] = (daily_dates_a[i] - daily_dates_a[i - 1]).days

            for year in OOS_YEARS:
                year_mask_d = daily_dates_a.year == year
                if year_mask_d.sum() < 20:
                    continue

                # Baseline A sharpe for this year
                _compute_year_sharpe = lambda dn, de: _year_sharpe(dn, de, rf_aligned_a, cal_days_a, year_mask_d)
                a_sh = _compute_year_sharpe(daily_net_a, daily_exp_a)

                # ═══ B ═══
                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    vm = np.nanmedian(vp[~np.isnan(vp)]) if np.any(~np.isnan(vp)) else 0.02
                    best_b_sh = -999; best_b_p = {}
                    for bp in grid_combos(B_GRID):
                        b_pos = apply_B(a_pos, close, atr, vp, vm,
                                        bp["k_sl"], bp["ratio"], bp["base_hold"],
                                        bp["hold_gamma"], bp["be_trigger"])
                        dd, dn, de, dp_m, dg = hourly_to_daily_agg(b_pos, lr, dates, comm)
                        if len(dd) < 20:
                            continue
                        rf_b = kr_s.reindex(dd, method="ffill").values / 365.0
                        rf_b = np.where(np.isfinite(rf_b), rf_b, 0.15 / 365.0)
                        cd_b = np.ones(len(dd))
                        for ib in range(1, len(dd)):
                            cd_b[ib] = (dd[ib] - dd[ib - 1]).days
                        ym_b = dd.year == year
                        if ym_b.sum() < 20:
                            continue
                        b_sh = _year_sharpe(dn, de, rf_b, cd_b, ym_b)
                        n_eval += 1
                        if b_sh > best_b_sh:
                            best_b_sh = b_sh; best_b_p = {**bp, "horizon": horizon}
                    if best_b_sh > -900:
                        all_results["B"].append(dict(
                            strategy=sid, ticker=ticker, year=year,
                            a_sharpe=round(a_sh, 3), bcd_sharpe=round(best_b_sh, 3),
                            delta=round(best_b_sh - a_sh, 3),
                            params=json.dumps(best_b_p, sort_keys=True)))

                # ═══ C ═══
                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    best_c_sh = -999; best_c_p = {}
                    for cp in grid_combos(C_GRID):
                        c_pos = apply_C(a_pos, vp, cp["threshold"], cp["lookback"],
                                        stype, cp["direction"],
                                        sigma_daily.get("h22"), sigma_daily.get("h1"),
                                        cp["term_str"], cp["hysteresis"])
                        dd, dn, de, dp_m, dg = hourly_to_daily_agg(c_pos, lr, dates, comm)
                        if len(dd) < 20:
                            continue
                        rf_c = kr_s.reindex(dd, method="ffill").values / 365.0
                        rf_c = np.where(np.isfinite(rf_c), rf_c, 0.15 / 365.0)
                        cd_c = np.ones(len(dd))
                        for ic in range(1, len(dd)):
                            cd_c[ic] = (dd[ic] - dd[ic - 1]).days
                        ym_c = dd.year == year
                        if ym_c.sum() < 20:
                            continue
                        c_sh = _year_sharpe(dn, de, rf_c, cd_c, ym_c)
                        n_eval += 1
                        if c_sh > best_c_sh:
                            best_c_sh = c_sh; best_c_p = {**cp, "horizon": horizon}
                    if best_c_sh > -900:
                        all_results["C"].append(dict(
                            strategy=sid, ticker=ticker, year=year,
                            a_sharpe=round(a_sh, 3), bcd_sharpe=round(best_c_sh, 3),
                            delta=round(best_c_sh - a_sh, 3),
                            params=json.dumps(best_c_p, sort_keys=True, default=str)))

                # ═══ D ═══
                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    best_d_sh = -999; best_d_p = {}
                    for dp in grid_combos(D_GRID):
                        d_pos = apply_D(a_pos, vp, dp["sigma_target"], dp["gamma"],
                                        dp["smooth"], dp["vol_floor"], dp["vol_cap"])
                        dd, dn, de, dp_m, dg = hourly_to_daily_agg(d_pos, lr, dates, comm)
                        if len(dd) < 20:
                            continue
                        rf_d = kr_s.reindex(dd, method="ffill").values / 365.0
                        rf_d = np.where(np.isfinite(rf_d), rf_d, 0.15 / 365.0)
                        cd_d = np.ones(len(dd))
                        for id_ in range(1, len(dd)):
                            cd_d[id_] = (dd[id_] - dd[id_ - 1]).days
                        ym_d = dd.year == year
                        if ym_d.sum() < 20:
                            continue
                        d_sh = _year_sharpe(dn, de, rf_d, cd_d, ym_d)
                        n_eval += 1
                        if d_sh > best_d_sh:
                            best_d_sh = d_sh; best_d_p = {**dp, "horizon": horizon}
                    if best_d_sh > -900:
                        all_results["D"].append(dict(
                            strategy=sid, ticker=ticker, year=year,
                            a_sharpe=round(a_sh, 3), bcd_sharpe=round(best_d_sh, 3),
                            delta=round(best_d_sh - a_sh, 3),
                            params=json.dumps(best_d_p, sort_keys=True)))

        elapsed = time.time() - t0
        log(f"    [{ti + 1}/17] {ticker}: {elapsed:.0f}s, {n_eval:,} evals")

    return {k: pd.DataFrame(v) for k, v in all_results.items()}


def _year_sharpe(daily_net, daily_exp, rf_aligned, cal_days, year_mask):
    """Compute excess sharpe for one year from daily aggregated data."""
    dn = daily_net[year_mask]
    de = daily_exp[year_mask]
    rf = rf_aligned[year_mask]
    cd = cal_days[year_mask]
    ny = len(dn)
    if ny < 20:
        return -999.0
    fdr = (1.0 - de) * rf * cd
    daily_total = dn + fdr
    excess = daily_total - rf * cd
    s = np.std(daily_total, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(excess) * sqrt(BPY) / s


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD A: Expanding window on hourly
# ═══════════════════════════════════════════════════════════════

def walk_forward_A(ticker_data_list, grid, kr_s, comm, comm_label):
    """Walk-forward A on hourly: expanding window, grid search on train, eval on test."""
    log(f"  Walk-Forward A hourly (comm={comm})")
    results = []
    positions_all = []

    for ti, td in enumerate(ticker_data_list):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; rh = td["rh"]; rl = td["rl"]
        lr = td["lr"]; dates = td["dates"]; n = td["n"]
        volume = td["volume"]

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            grid_sid = grid.get(sid, {})
            sig_grid = grid_sid.get("signal", {})
            confs = grid_sid.get("confirmations", ["none"])
            grid_filters = grid_sid.get("filters", {})

            rsi = td["ind"]["rsi14"]; vol_sma20 = td["ind"]["vol_sma20"]
            day_of_week = dates.dayofweek.values

            for test_year in OOS_YEARS:
                # Train: everything before test_year
                _, train_end = _year_bounds(dates, test_year)
                if train_end <= WARMUP_H or train_end >= n:
                    continue
                # Test: test_year only
                test_start, test_end = _year_bounds(dates, test_year)
                if test_year < 2026:
                    _, test_end = _year_bounds(dates, test_year + 1)
                else:
                    test_end = n
                if test_start >= test_end:
                    continue

                train_mask_h = np.zeros(n, dtype=bool)
                train_mask_h[WARMUP_H:train_end] = True

                # NARROW grid: use best config from daily ± variations
                cfg = best_configs.get(sid, {}).get(ticker)
                sig_variants_narrow = []
                rm_variants_narrow = []
                confs_narrow = ["none"]
                if cfg:
                    best_sig_p = json.loads(cfg["signal"])
                    best_rm_p = json.loads(cfg["rm"])
                    confs_narrow = list(set(["none", cfg.get("conf", "none")]))
                    sig_variants_narrow = [best_sig_p]
                    for key, val in best_sig_p.items():
                        if isinstance(val, (int, float)) and key != "ma":
                            for m in [0.7, 0.85, 1.15, 1.3]:
                                v2 = best_sig_p.copy()
                                v2[key] = round(val * m, 2) if isinstance(val, float) else max(1, int(val * m))
                                sig_variants_narrow.append(v2)
                    rm_variants_narrow = [best_rm_p]
                    for key, val in best_rm_p.items():
                        if val != 0 and isinstance(val, (int, float)):
                            for m in [0.7, 1.3]:
                                r2 = best_rm_p.copy()
                                r2[key] = round(val * m, 2) if isinstance(val, float) else max(1, int(val * m))
                                rm_variants_narrow.append(r2)
                    rm_variants_narrow.append(DEFAULT_RM_TREND if is_trend else DEFAULT_RM)
                else:
                    # Fallback: use default RM only with full signal grid (limited to first 20)
                    sig_variants_narrow = list(iterate_signal_grid(sid, sig_grid))[:20]
                    rm_variants_narrow = [DEFAULT_RM_TREND if is_trend else DEFAULT_RM]

                best_sh_train = -999; best_sig = None; best_rm = None; best_conf = None
                n_tried = 0

                for sig_params in sig_variants_narrow:
                    sig_raw = gen_signal_hourly(sid, td, sig_params, WARMUP_H)
                    if sig_raw is None:
                        continue

                    for conf in confs_narrow:
                        sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
                        if np.sum(np.abs(sig[train_mask_h]) > 0) < 5:
                            continue

                        for rm in rm_variants_narrow:
                            pos = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP_H, train_end)

                            # Aggregate to daily for train evaluation
                            dd, dn, de, dp_m, _ = hourly_to_daily_agg(
                                pos[:train_end], lr[:train_end], dates[:train_end], comm)
                            if len(dd) < 30:
                                continue
                            rf_t = kr_s.reindex(dd, method="ffill").values / 365.0
                            rf_t = np.where(np.isfinite(rf_t), rf_t, 0.15 / 365.0)
                            cd_t = np.ones(len(dd))
                            for i in range(1, len(dd)):
                                cd_t[i] = (dd[i] - dd[i - 1]).days
                            full_mask = np.ones(len(dd), dtype=bool)
                            sh_t = _year_sharpe(dn, de, rf_t, cd_t, full_mask)
                            n_tried += 1

                            if sh_t > best_sh_train:
                                best_sh_train = sh_t
                                best_sig = sig_params; best_rm = rm; best_conf = conf

                # Apply best to full data, extract test year
                if best_sig is not None and best_rm is not None:
                    sig_raw = gen_signal_hourly(sid, td, best_sig, WARMUP_H)
                    if sig_raw is not None:
                        sig = apply_confirmation(sig_raw.copy(), best_conf, rsi, volume, vol_sma20, is_trend)
                        pos_full = run_bt(sig, close, high, low, atr, rh, rl, best_rm, WARMUP_H, test_end)

                        # Aggregate full, evaluate test year
                        dd_f, dn_f, de_f, dp_f, dg_f = hourly_to_daily_agg(
                            pos_full, lr, dates, comm)
                        if len(dd_f) > 20:
                            rf_f = kr_s.reindex(dd_f, method="ffill").values / 365.0
                            rf_f = np.where(np.isfinite(rf_f), rf_f, 0.15 / 365.0)
                            cd_f = np.ones(len(dd_f))
                            for i in range(1, len(dd_f)):
                                cd_f[i] = (dd_f[i] - dd_f[i - 1]).days
                            ym = dd_f.year == test_year
                            if ym.sum() >= 20:
                                sh_test = _year_sharpe(dn_f, de_f, rf_f, cd_f, ym)
                                exp_pct = np.mean(de_f[ym]) * 100
                                results.append(dict(
                                    strategy=sid, ticker=ticker, year=test_year,
                                    sharpe=round(sh_test, 3), exposure=round(exp_pct, 1),
                                    train_sharpe=round(best_sh_train, 3), n_tried=n_tried))

                                # Positions
                                dg_y = dg_f[ym]; dp_y = dp_f[ym]; dd_y = dd_f[ym]
                                for j in range(len(dd_y)):
                                    positions_all.append(dict(
                                        date=dd_y[j], ticker=ticker, strategy=sid,
                                        year=test_year,
                                        position=round(float(dp_y[j]), 6),
                                        daily_gross_return=round(float(dg_y[j]), 8)))

        elapsed = time.time() - t0
        log(f"    [{ti + 1}/{len(ticker_data_list)}] {ticker}: {elapsed:.0f}s")

    return pd.DataFrame(results), positions_all


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD BCD: On hourly positions from WF-A
# ═══════════════════════════════════════════════════════════════

def walk_forward_BCD(ticker_data_list, best_configs, kr_s, comm, comm_label,
                     wf_a_positions_df):
    """Walk-forward B/C/D on hourly. Uses WF-A positions as base."""
    log(f"  Walk-Forward BCD hourly (comm={comm})")
    results_all = {"B": [], "C": [], "D": []}
    positions_all = {"B": [], "C": [], "D": []}

    for ti, td in enumerate(ticker_data_list):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; high = td["high"]; low = td["low"]
        atr = td["atr"]; lr = td["lr"]; dates = td["dates"]; n = td["n"]
        sigma_daily = td["sigma_daily"]

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            stype = STRATEGY_TYPE[sid]
            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None:
                continue

            sig_params = json.loads(cfg["signal"])
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"])
            rsi = td["ind"]["rsi14"]; vol_sma20 = td["ind"]["vol_sma20"]

            sig_raw = gen_signal_hourly(sid, td, sig_params, WARMUP_H)
            if sig_raw is None:
                continue
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, td["volume"], vol_sma20, is_trend)
            a_pos = run_bt(sig, close, high, low, atr, td["rh"], td["rl"], rm, WARMUP_H, n)

            for test_year in OOS_YEARS:
                # Train: all before test_year
                _, train_end = _year_bounds(dates, test_year)
                test_start, test_end = _year_bounds(dates, test_year)
                if test_year < 2026:
                    _, test_end = _year_bounds(dates, test_year + 1)
                else:
                    test_end = n
                if train_end <= WARMUP_H or test_start >= test_end:
                    continue

                # Aggregate train A to daily
                dd_train, dn_train, de_train, _, _ = hourly_to_daily_agg(
                    a_pos[:train_end], lr[:train_end], dates[:train_end], comm)
                if len(dd_train) < 30:
                    continue
                rf_train = kr_s.reindex(dd_train, method="ffill").values / 365.0
                rf_train = np.where(np.isfinite(rf_train), rf_train, 0.15 / 365.0)
                cd_train = np.ones(len(dd_train))
                for i in range(1, len(dd_train)):
                    cd_train[i] = (dd_train[i] - dd_train[i - 1]).days
                fm_train = np.ones(len(dd_train), dtype=bool)

                # ═══ B: grid search on train, apply to test ═══
                best_b_sh = -999; best_b_p = {}
                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    vm = np.nanmedian(vp[~np.isnan(vp)]) if np.any(~np.isnan(vp)) else 0.02
                    for bp in grid_combos(B_GRID):
                        b_pos_train = apply_B(a_pos[:train_end], close[:train_end], atr[:train_end],
                                              vp[:train_end], vm,
                                              bp["k_sl"], bp["ratio"], bp["base_hold"],
                                              bp["hold_gamma"], bp["be_trigger"])
                        dd_b, dn_b, de_b, _, _ = hourly_to_daily_agg(
                            b_pos_train, lr[:train_end], dates[:train_end], comm)
                        if len(dd_b) < 30:
                            continue
                        rf_b = kr_s.reindex(dd_b, method="ffill").values / 365.0
                        rf_b = np.where(np.isfinite(rf_b), rf_b, 0.15 / 365.0)
                        cd_b = np.ones(len(dd_b))
                        for i in range(1, len(dd_b)):
                            cd_b[i] = (dd_b[i] - dd_b[i - 1]).days
                        fm_b = np.ones(len(dd_b), dtype=bool)
                        b_sh = _year_sharpe(dn_b, de_b, rf_b, cd_b, fm_b)
                        if b_sh > best_b_sh:
                            best_b_sh = b_sh; best_b_p = {**bp, "horizon": horizon}

                # Apply best B to full data, eval test year
                if best_b_p:
                    horizon = best_b_p["horizon"]
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    vm = np.nanmedian(vp[~np.isnan(vp)]) if np.any(~np.isnan(vp)) else 0.02
                    b_pos_full = apply_B(a_pos, close, atr, vp, vm,
                                         best_b_p["k_sl"], best_b_p["ratio"],
                                         best_b_p["base_hold"], best_b_p["hold_gamma"],
                                         best_b_p["be_trigger"])
                    dd_bf, dn_bf, de_bf, dp_bf, dg_bf = hourly_to_daily_agg(
                        b_pos_full, lr, dates, comm)
                    if len(dd_bf) > 20:
                        rf_bf = kr_s.reindex(dd_bf, method="ffill").values / 365.0
                        rf_bf = np.where(np.isfinite(rf_bf), rf_bf, 0.15 / 365.0)
                        cd_bf = np.ones(len(dd_bf))
                        for i in range(1, len(dd_bf)):
                            cd_bf[i] = (dd_bf[i] - dd_bf[i - 1]).days
                        ym_bf = dd_bf.year == test_year
                        if ym_bf.sum() >= 20:
                            b_test_sh = _year_sharpe(dn_bf, de_bf, rf_bf, cd_bf, ym_bf)
                            results_all["B"].append(dict(
                                strategy=sid, ticker=ticker, year=test_year,
                                sharpe=round(b_test_sh, 3),
                                exposure=round(np.mean(de_bf[ym_bf]) * 100, 1)))
                            dg_y = dg_bf[ym_bf]; dp_y = dp_bf[ym_bf]; dd_y = dd_bf[ym_bf]
                            for j in range(len(dd_y)):
                                positions_all["B"].append(dict(
                                    date=dd_y[j], ticker=ticker, strategy=sid,
                                    year=test_year,
                                    position=round(float(dp_y[j]), 6),
                                    daily_gross_return=round(float(dg_y[j]), 8)))

                # ═══ C: grid search on train, apply to test ═══
                best_c_sh = -999; best_c_p = {}
                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    for cp in grid_combos(C_GRID):
                        c_pos_train = apply_C(a_pos[:train_end], vp[:train_end],
                                              cp["threshold"], cp["lookback"],
                                              stype, cp["direction"],
                                              sigma_daily.get("h22", np.full(n, 0.02))[:train_end],
                                              sigma_daily.get("h1", np.full(n, 0.02))[:train_end],
                                              cp["term_str"], cp["hysteresis"])
                        dd_c, dn_c, de_c, _, _ = hourly_to_daily_agg(
                            c_pos_train, lr[:train_end], dates[:train_end], comm)
                        if len(dd_c) < 30:
                            continue
                        rf_c = kr_s.reindex(dd_c, method="ffill").values / 365.0
                        rf_c = np.where(np.isfinite(rf_c), rf_c, 0.15 / 365.0)
                        cd_c = np.ones(len(dd_c))
                        for i in range(1, len(dd_c)):
                            cd_c[i] = (dd_c[i] - dd_c[i - 1]).days
                        fm_c = np.ones(len(dd_c), dtype=bool)
                        c_sh = _year_sharpe(dn_c, de_c, rf_c, cd_c, fm_c)
                        if c_sh > best_c_sh:
                            best_c_sh = c_sh; best_c_p = {**cp, "horizon": horizon}

                if best_c_p:
                    horizon = best_c_p["horizon"]
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    c_pos_full = apply_C(a_pos, vp, best_c_p["threshold"], best_c_p["lookback"],
                                         stype, best_c_p["direction"],
                                         sigma_daily.get("h22", np.full(n, 0.02)),
                                         sigma_daily.get("h1", np.full(n, 0.02)),
                                         best_c_p["term_str"], best_c_p["hysteresis"])
                    dd_cf, dn_cf, de_cf, dp_cf, dg_cf = hourly_to_daily_agg(
                        c_pos_full, lr, dates, comm)
                    if len(dd_cf) > 20:
                        rf_cf = kr_s.reindex(dd_cf, method="ffill").values / 365.0
                        rf_cf = np.where(np.isfinite(rf_cf), rf_cf, 0.15 / 365.0)
                        cd_cf = np.ones(len(dd_cf))
                        for i in range(1, len(dd_cf)):
                            cd_cf[i] = (dd_cf[i] - dd_cf[i - 1]).days
                        ym_cf = dd_cf.year == test_year
                        if ym_cf.sum() >= 20:
                            c_test_sh = _year_sharpe(dn_cf, de_cf, rf_cf, cd_cf, ym_cf)
                            results_all["C"].append(dict(
                                strategy=sid, ticker=ticker, year=test_year,
                                sharpe=round(c_test_sh, 3),
                                exposure=round(np.mean(de_cf[ym_cf]) * 100, 1)))
                            dg_y = dg_cf[ym_cf]; dp_y = dp_cf[ym_cf]; dd_y = dd_cf[ym_cf]
                            for j in range(len(dd_y)):
                                positions_all["C"].append(dict(
                                    date=dd_y[j], ticker=ticker, strategy=sid,
                                    year=test_year,
                                    position=round(float(dp_y[j]), 6),
                                    daily_gross_return=round(float(dg_y[j]), 8)))

                # ═══ D: grid search on train, apply to test ═══
                best_d_sh = -999; best_d_p = {}
                for horizon in HORIZONS:
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    for dp in grid_combos(D_GRID):
                        d_pos_train = apply_D(a_pos[:train_end], vp[:train_end],
                                              dp["sigma_target"], dp["gamma"],
                                              dp["smooth"], dp["vol_floor"], dp["vol_cap"])
                        dd_d, dn_d, de_d, _, _ = hourly_to_daily_agg(
                            d_pos_train, lr[:train_end], dates[:train_end], comm)
                        if len(dd_d) < 30:
                            continue
                        rf_d = kr_s.reindex(dd_d, method="ffill").values / 365.0
                        rf_d = np.where(np.isfinite(rf_d), rf_d, 0.15 / 365.0)
                        cd_d = np.ones(len(dd_d))
                        for i in range(1, len(dd_d)):
                            cd_d[i] = (dd_d[i] - dd_d[i - 1]).days
                        fm_d = np.ones(len(dd_d), dtype=bool)
                        d_sh = _year_sharpe(dn_d, de_d, rf_d, cd_d, fm_d)
                        if d_sh > best_d_sh:
                            best_d_sh = d_sh; best_d_p = {**dp, "horizon": horizon}

                if best_d_p:
                    horizon = best_d_p["horizon"]
                    vp = sigma_daily.get(horizon, np.full(n, 0.02))
                    d_pos_full = apply_D(a_pos, vp, best_d_p["sigma_target"],
                                         best_d_p["gamma"], best_d_p["smooth"],
                                         best_d_p["vol_floor"], best_d_p["vol_cap"])
                    dd_df, dn_df, de_df, dp_df, dg_df = hourly_to_daily_agg(
                        d_pos_full, lr, dates, comm)
                    if len(dd_df) > 20:
                        rf_df = kr_s.reindex(dd_df, method="ffill").values / 365.0
                        rf_df = np.where(np.isfinite(rf_df), rf_df, 0.15 / 365.0)
                        cd_df = np.ones(len(dd_df))
                        for i in range(1, len(dd_df)):
                            cd_df[i] = (dd_df[i] - dd_df[i - 1]).days
                        ym_df = dd_df.year == test_year
                        if ym_df.sum() >= 20:
                            d_test_sh = _year_sharpe(dn_df, de_df, rf_df, cd_df, ym_df)
                            results_all["D"].append(dict(
                                strategy=sid, ticker=ticker, year=test_year,
                                sharpe=round(d_test_sh, 3),
                                exposure=round(np.mean(de_df[ym_df]) * 100, 1)))
                            dg_y = dg_df[ym_df]; dp_y = dp_df[ym_df]; dd_y = dd_df[ym_df]
                            for j in range(len(dd_y)):
                                positions_all["D"].append(dict(
                                    date=dd_y[j], ticker=ticker, strategy=sid,
                                    year=test_year,
                                    position=round(float(dp_y[j]), 6),
                                    daily_gross_return=round(float(dg_y[j]), 8)))

        elapsed = time.time() - t0
        log(f"    [{ti + 1}/{len(ticker_data_list)}] {ticker}: {elapsed:.0f}s")

    return {k: pd.DataFrame(v) for k, v in results_all.items()}, positions_all


# ═══════════════════════════════════════════════════════════════
# SAVE HELPERS
# ═══════════════════════════════════════════════════════════════

def save_results_csv(df, path, label):
    df.to_csv(path, index=False)
    log(f"  Saved {label}: {path} ({len(df)} rows)")


def save_positions_parquet(positions_list, path, label):
    if not positions_list:
        log(f"  SKIP {label}: no positions")
        return
    df = pd.DataFrame(positions_list)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(path, index=False)
    log(f"  Saved {label}: {path} ({len(df)} rows)")


def print_summary(df, label):
    if df is None or len(df) == 0:
        log(f"  {label}: no results")
        return
    log(f"\n  {label} Summary:")
    for sid in STRATEGIES:
        sub = df[df.strategy == sid]
        if len(sub) == 0:
            continue
        mean_sh = sub["sharpe"].mean()
        n_pos = (sub["sharpe"] > 0).sum()
        log(f"    {sid}: mean={mean_sh:+.2f}, N>0={n_pos}/{len(sub)}")
    overall = df.groupby("strategy")["sharpe"].mean().mean()
    log(f"    Overall: {overall:+.2f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log("=" * 80)
    log("STAGE 4: HOURLY PIPELINE (COMPLETE)")
    log(f"  Commission: inst={COMM_HOURLY['inst']}, std={COMM_HOURLY['std']}")
    log(f"  OOS years: {OOS_YEARS}")
    log(f"  Warmup: {WARMUP_H} hourly bars")
    log("=" * 80)

    # ═══════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════
    log("\nLoading data...")
    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    vpred_df = pd.read_parquet(VPRED_PATH)
    vpred_df["date"] = pd.to_datetime(vpred_df["date"])
    warmup_numba()

    log(f"  Hourly data: {len(hourly_df):,} rows, {hourly_df['ticker'].nunique()} tickers")
    log(f"  Vpred: {len(vpred_df):,} rows")

    if len(hourly_df) < 1000:
        log("ERROR: Hourly data too small. Aborting.")
        return

    # Load best configs (from autonomous_test_scan phase 1-9)
    if not CONFIGS_PATH.exists():
        log(f"ERROR: {CONFIGS_PATH} not found. Cannot proceed.")
        return
    with open(CONFIGS_PATH) as f:
        best_configs = json.load(f)
    log(f"  Loaded {sum(len(v) for v in best_configs.values())} best A configs")

    # Load narrowed grid
    if GRID_PATH.exists():
        with open(GRID_PATH) as f:
            narrowed_grid = json.load(f)
        log(f"  Loaded narrowed grid from {GRID_PATH}")
    else:
        log("WARNING: narrowed_grid_v2.json not found, using empty grid")
        narrowed_grid = {}

    # Warmup numba for apply_B
    log("  Warming up apply_B numba...")
    _dummy = np.zeros(10, dtype=np.float64)
    _apply_B_numba(_dummy, np.ones(10), np.ones(10), 0.02, 1.0, 1.0, 10.0, 0.5, 0.0)

    # ═══════════════════════════════════════
    # PREPARE TICKER DATA
    # ═══════════════════════════════════════
    log("\nPreparing hourly ticker data...")
    ticker_data_list = []
    for ticker in TICKERS:
        td = prepare_hourly_ticker(ticker, hourly_df, daily_df, kr_s, vpred_df)
        if td is not None:
            ticker_data_list.append(td)
            log(f"  {ticker}: {td['n']:,} hourly bars")
        else:
            log(f"  {ticker}: SKIP (too few bars)")
    log(f"  Prepared {len(ticker_data_list)}/17 tickers")

    del hourly_df; gc.collect()

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: TEST A (both commissions)
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PHASE 1: TEST A on HOURLY")
    t1 = time.time()

    for comm_label, comm_val in COMM_HOURLY.items():
        suffix = f"_{int(comm_val * 10000):03d}"
        df_a, pos_a = run_test_A(ticker_data_list, best_configs, kr_s, comm_val)
        save_results_csv(df_a, OUT_DIR / f"hourly_test_A_results{suffix}.csv", f"Test A ({comm_label})")
        print_summary(df_a, f"Test A ({comm_label})")

    log(f"Phase 1 done in {time.time() - t1:.0f}s")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: TEST BCD (both commissions)
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PHASE 2: TEST BCD on HOURLY")
    t2 = time.time()

    for comm_label, comm_val in COMM_HOURLY.items():
        bcd_results = run_test_BCD(ticker_data_list, best_configs, kr_s, comm_val, comm_label)
        for approach in ["B", "C", "D"]:
            df_bcd = bcd_results[approach]
            if len(df_bcd) > 0:
                save_results_csv(df_bcd, OUT_DIR / f"hourly_bcd_{approach}_results.csv",
                                 f"BCD {approach} ({comm_label})")
                mean_delta = df_bcd["delta"].mean() if "delta" in df_bcd.columns else 0
                log(f"    {approach}: mean delta={mean_delta:+.3f}")

    log(f"Phase 2 done in {time.time() - t2:.0f}s")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: WALK-FORWARD A (both commissions)
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PHASE 3: WALK-FORWARD A on HOURLY")
    t3 = time.time()

    wf_a_positions = {}
    for comm_label, comm_val in COMM_HOURLY.items():
        suffix = f"_{int(comm_val * 10000):03d}"
        df_wf_a, pos_wf_a = walk_forward_A(ticker_data_list, narrowed_grid, kr_s, comm_val, comm_label)
        save_results_csv(df_wf_a, OUT_DIR / f"hourly_wf_A_results{suffix}.csv", f"WF A ({comm_label})")
        save_positions_parquet(pos_wf_a, OUT_DIR / "hourly_wf_positions_A.parquet", f"WF A positions")
        print_summary(df_wf_a, f"WF A ({comm_label})")
        wf_a_positions[comm_label] = pos_wf_a
        # Also save combined csv
        save_results_csv(df_wf_a, OUT_DIR / f"hourly_wf_A_results.csv", f"WF A combined")

    log(f"Phase 3 done in {time.time() - t3:.0f}s")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: WALK-FORWARD BCD (both commissions)
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PHASE 4: WALK-FORWARD BCD on HOURLY")
    t4 = time.time()

    # Use institutional comm positions for WF-BCD base
    wf_a_pos_df = pd.DataFrame(wf_a_positions.get("inst", []))

    for comm_label, comm_val in COMM_HOURLY.items():
        suffix = f"_{int(comm_val * 10000):03d}"
        bcd_wf_results, bcd_wf_positions = walk_forward_BCD(
            ticker_data_list, best_configs, kr_s, comm_val, comm_label, wf_a_pos_df)

        for approach in ["B", "C", "D"]:
            df_bcd_wf = bcd_wf_results[approach]
            if len(df_bcd_wf) > 0:
                save_results_csv(df_bcd_wf,
                                 OUT_DIR / f"hourly_wf_{approach}_results{suffix}.csv",
                                 f"WF {approach} ({comm_label})")
                # Also save combined
                save_results_csv(df_bcd_wf,
                                 OUT_DIR / f"hourly_wf_{approach}_results.csv",
                                 f"WF {approach} combined")
                print_summary(df_bcd_wf, f"WF {approach} ({comm_label})")

            pos_list = bcd_wf_positions[approach]
            if pos_list:
                save_positions_parquet(pos_list,
                                       OUT_DIR / f"hourly_wf_positions_{approach}.parquet",
                                       f"WF {approach} positions")

    log(f"Phase 4 done in {time.time() - t4:.0f}s")

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    log(f"\n{'=' * 80}")
    log(f"STAGE 4 HOURLY PIPELINE COMPLETE")
    log(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"{'=' * 80}")

    # List all output files
    log("\nOutput files:")
    for f in sorted(OUT_DIR.glob("hourly_*")):
        size = f.stat().st_size / 1024
        log(f"  {f.name} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
