#!/usr/bin/env python3
"""
daily_perticker_7d.py — Per-ticker 7-dimensional optimization for daily strategies.

6-stage random search: signals → exits → signal_strength → filters → refinement
Per-ticker walk-forward, excess sharpe with trading-day vol.
"""
import os, sys, warnings, time
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from itertools import product
import random

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP, load_data, compute_base,
    precompute_sma_cache, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    calc_pivot_daily, calc_rsi, calc_sma, calc_atr, calc_adx,
    calc_supertrend, _year_bounds,
    bt_contrarian_v3, bt_range_v3,
    warmup_numba, _pctile_cache,
)
from strategies_walkforward import bt_trend_v3

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/daily_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BPY = 252
COMM = 0.0005
OOS_YEARS = [2022, 2023, 2024, 2025]
random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════
# Signal generators with confirmation filters
# ═══════════════════════════════════════════════════

def gen_signals_s1(close, ind, sma_cache, rsi, params):
    """S1 MeanRev with optional confirmation."""
    w = params["ma_window"]
    z_e = params["z_entry"]
    z_x = params.get("z_exit", 0.3)
    confirm = params.get("confirm", "none")

    if w not in sma_cache:
        return np.zeros(len(close), dtype=np.int8), np.zeros(len(close), dtype=bool)
    sma, std = sma_cache[w]
    z = (close - sma) / std
    n = len(close)
    valid = ~np.isnan(z)

    entry_l = valid & (z < -z_e) & ind["vf_pass"]
    entry_s = valid & (z > z_e) & ind["vf_pass"]

    # Confirmation
    if confirm == "rsi30":
        entry_l = entry_l & (~np.isnan(rsi)) & (rsi < 30)
        entry_s = entry_s & (~np.isnan(rsi)) & (rsi > 70)
    elif confirm == "rsi20":
        entry_l = entry_l & (~np.isnan(rsi)) & (rsi < 20)
        entry_s = entry_s & (~np.isnan(rsi)) & (rsi > 80)
    elif confirm == "vol15":
        vr = ind["_volume"] / np.where(ind["vol_sma20"] > 0, ind["vol_sma20"], 1)
        entry_l = entry_l & (vr > 1.5)
        entry_s = entry_s & (vr > 1.5)
    elif confirm == "vol20":
        vr = ind["_volume"] / np.where(ind["vol_sma20"] > 0, ind["vol_sma20"], 1)
        entry_l = entry_l & (vr > 2.0)
        entry_s = entry_s & (vr > 2.0)

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_arr = ~np.isnan(z) & (np.abs(z) < z_x)
    return sig, exit_arr


def gen_signals_s2(close, ind, sma_cache, rsi, params):
    """S2 Bollinger with optional confirmation."""
    w = params["bb_window"]
    bb_std = params["bb_std"]
    confirm = params.get("confirm", "none")

    if w not in sma_cache:
        return np.zeros(len(close), dtype=np.int8), np.zeros(len(close), dtype=bool)
    sma, std = sma_cache[w]
    bb_u = sma + bb_std * std
    bb_l = sma - bb_std * std
    n = len(close)
    valid = ~np.isnan(sma)
    entry_l = valid & (close <= bb_l) & ind["vf_pass"]
    entry_s = valid & (close >= bb_u) & ind["vf_pass"]

    if confirm == "rsi30":
        entry_l = entry_l & (~np.isnan(rsi)) & (rsi < 30)
        entry_s = entry_s & (~np.isnan(rsi)) & (rsi > 70)
    elif confirm == "vol15":
        vr = ind["_volume"] / np.where(ind["vol_sma20"] > 0, ind["vol_sma20"], 1)
        entry_l = entry_l & (vr > 1.5)
        entry_s = entry_s & (vr > 1.5)

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    prev_c = np.empty_like(close); prev_c[0] = close[0]; prev_c[1:] = close[:-1]
    prev_m = np.empty_like(sma); prev_m[0] = sma[0]; prev_m[1:] = sma[:-1]
    exit_arr = np.zeros(n, dtype=bool)
    exit_arr[1:] = (((prev_c[1:] < prev_m[1:]) & (close[1:] >= sma[1:])) |
                    ((prev_c[1:] > prev_m[1:]) & (close[1:] <= sma[1:])))
    return sig, exit_arr


def gen_signals_s3(close, high, low, ind, dc_cache, rsi, params):
    """S3 Donchian with confirmation."""
    w = params["dc_window"]
    confirm = params.get("confirm", "none")
    if w not in dc_cache:
        return np.zeros(len(close), dtype=np.int8), (np.zeros(len(close), dtype=bool), np.zeros(len(close), dtype=bool))

    hc, lc = dc_cache[w]
    prev_hc = np.empty_like(hc); prev_hc[0] = np.nan; prev_hc[1:] = hc[:-1]
    prev_lc = np.empty_like(lc); prev_lc[0] = np.nan; prev_lc[1:] = lc[:-1]
    n = len(close)
    valid = ~np.isnan(prev_hc)
    entry_l = valid & (close > prev_hc) & ind["vf_pass"]
    entry_s = valid & (close < prev_lc) & ind["vf_pass"]

    adx = ind["adx14"]
    if confirm == "adx20":
        entry_l = entry_l & (~np.isnan(adx)) & (adx > 20)
        entry_s = entry_s & (~np.isnan(adx)) & (adx > 20)
    elif confirm == "adx25":
        entry_l = entry_l & (~np.isnan(adx)) & (adx > 25)
        entry_s = entry_s & (~np.isnan(adx)) & (adx > 25)
    elif confirm == "adx30":
        entry_l = entry_l & (~np.isnan(adx)) & (adx > 30)
        entry_s = entry_s & (~np.isnan(adx)) & (adx > 30)
    elif confirm == "vol15":
        vr = ind["_volume"] / np.where(ind["vol_sma20"] > 0, ind["vol_sma20"], 1)
        entry_l = entry_l & (vr > 1.5)
        entry_s = entry_s & (vr > 1.5)

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_l = ~np.isnan(prev_lc) & (close < prev_lc)
    exit_s = ~np.isnan(prev_hc) & (close > prev_hc)
    return sig, (exit_l, exit_s)


def gen_signals_s4(close, high, low, ind, st_cache, rsi, params):
    """S4 Supertrend with confirmation."""
    ap = params["atr_period"]; mult = params["multiplier"]
    confirm = params.get("confirm", "none")
    key = (ap, mult)
    if key not in st_cache:
        return np.zeros(len(close), dtype=np.int8), (np.zeros(len(close), dtype=bool), np.zeros(len(close), dtype=bool))

    st, _ = st_cache[key]
    st_prev = np.empty_like(st); st_prev[0] = np.nan; st_prev[1:] = st[:-1]
    c_prev = np.empty_like(close); c_prev[0] = close[0]; c_prev[1:] = close[:-1]
    n = len(close)
    valid = ~np.isnan(st) & ~np.isnan(st_prev)
    entry_l = valid & (close > st) & (c_prev <= st_prev) & ind["vf_pass"]
    entry_s = valid & (close < st) & (c_prev >= st_prev) & ind["vf_pass"]

    adx = ind["adx14"]
    if confirm == "adx15":
        entry_l = entry_l & (~np.isnan(adx)) & (adx > 15)
        entry_s = entry_s & (~np.isnan(adx)) & (adx > 15)
    elif confirm == "adx20":
        entry_l = entry_l & (~np.isnan(adx)) & (adx > 20)
        entry_s = entry_s & (~np.isnan(adx)) & (adx > 20)
    elif confirm == "adx25":
        entry_l = entry_l & (~np.isnan(adx)) & (adx > 25)
        entry_s = entry_s & (~np.isnan(adx)) & (adx > 25)
    elif confirm == "vol15":
        vr = ind["_volume"] / np.where(ind["vol_sma20"] > 0, ind["vol_sma20"], 1)
        entry_l = entry_l & (vr > 1.5)
        entry_s = entry_s & (vr > 1.5)
    elif confirm == "mom3":
        mom = np.zeros(n); mom[3:] = close[3:] - close[:-3]
        entry_l = entry_l & (mom > 0)
        entry_s = entry_s & (mom < 0)

    sig = np.where(entry_l, 1, np.where(entry_s, -1, 0)).astype(np.int8)
    sig[:WARMUP] = 0
    exit_l = valid & (close < st) & (c_prev >= st_prev)
    exit_s = valid & (close > st) & (c_prev <= st_prev)
    return sig, (exit_l, exit_s)


def gen_signals_s5(close, ind, pivot_data, rsi, params):
    """S5 Pivot Points."""
    from strategies_walkforward import gen_s5_signals
    sp = {k: v for k, v in params.items() if k in
          ["pivot_type","entry_level","pivot_frac","adx_thresh","min_atr_pctile"]}
    return gen_s5_signals(close, ind, pivot_data, is_hourly=False, **sp)


def gen_signals_s6(close, high, low, volume, ind, vwap_cache, rsi, params):
    """S6 VWAP."""
    from strategies_walkforward import gen_s6_signals
    sp = {k: v for k, v in params.items() if k in ["vwap_window","dev_mult","exit_mult","adx_thresh"]}
    return gen_s6_signals(close, high, low, volume, ind, vwap_cache, is_hourly=False, **sp)


# ═══════════════════════════════════════════════════
# Random parameter sampling
# ═══════════════════════════════════════════════════

SIG_SPACE = {
    "S1": dict(ma_window=[5,7,10,12,15,18,20,25,30,40,50,60,80,100],
               z_entry=[0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0],
               z_exit=[0.0,0.2,0.3,0.5,0.7],
               confirm=["none","rsi30","rsi20","vol15","vol20"]),
    "S2": dict(bb_window=[5,7,10,12,15,18,20,25,30,40,50,60,80,100],
               bb_std=[0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5],
               confirm=["none","rsi30","vol15"]),
    "S3": dict(dc_window=[5,7,10,12,15,18,20,25,30,40,50,60,80],
               confirm=["none","adx20","adx25","adx30","vol15"]),
    "S4": dict(atr_period=[5,7,10,12,14,18,20,25,30,40,50,60],
               multiplier=[0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,3.5,4.0,5.0,6.0],
               confirm=["none","adx15","adx20","adx25","vol15","mom3"]),
    "S5": dict(pivot_type=["classic","fibonacci","woodie","camarilla"],
               entry_level=[1,2,3], pivot_frac=[0.5,0.7,1.0],
               adx_thresh=[20,25,30,35,50], min_atr_pctile=[0,25,50,75]),
    "S6": dict(vwap_window=[5,10,15,20,25,30],
               dev_mult=[0.3,0.5,0.75,1.0,1.5,2.0,2.5],
               exit_mult=[0.3,0.5,0.8], adx_thresh=[25,35,50]),
}

CATEGORY = {"S1":"C","S2":"C","S3":"T","S4":"T","S5":"R","S6":"R"}

RM_CONTRA = dict(sl_mult=[0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0],
                 tp_mult=[0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0],
                 max_hold=[3,5,7,10,15,20,30,50])
RM_TREND = dict(initial_sl_mult=[1.5,2.0,2.5,3.0,4.0,5.0],
                trail_type=["fixed_atr","chandelier"],
                trail_n=[3,5,7,10,15,20,30],
                trail_atr_mult=[1.0,1.5,2.0,2.5,3.0,4.0,5.0])
RM_RANGE = dict(sl_mult=[0.5,0.75,1.0,1.5,2.0,3.0],
                tp_mult=[0.5,0.75,1.0,1.5,2.0,3.0],
                max_hold=[3,5,7,10,15,20])


def random_sig_params(sid, n=500):
    space = SIG_SPACE[sid]
    result = []
    for _ in range(n):
        p = {k: random.choice(v) for k, v in space.items()}
        result.append(p)
    return result


def random_rm_params(cat, n=200):
    if cat == "C":
        space = RM_CONTRA
    elif cat == "T":
        space = RM_TREND
    else:
        space = RM_RANGE
    result = []
    for _ in range(n):
        p = {k: random.choice(v) for k, v in space.items()}
        if cat == "T":
            p["breakeven_thresh"] = random.choice([None, 1.0])
            p["cooldown_bars"] = random.choice([0, 5])
            p["parabolic_step"] = 0.02
            p["parabolic_max"] = 0.15
        elif cat == "C":
            p["breakeven_trigger"] = random.choice([None, 1.0])
            p["cooldown_bars"] = random.choice([0, 3, 5])
            p["partial_exit"] = random.choice([False, True])
        elif cat == "R":
            p["breakeven_trigger"] = random.choice([None, 1.0])
            p["cooldown_bars"] = random.choice([0, 3, 5])
            p["time_decay"] = random.choice([False, True])
        result.append(p)
    return result


# ═══════════════════════════════════════════════════
# Backtest dispatcher
# ═══════════════════════════════════════════════════

def run_bt(cat, sig, exit_info, close, high, low, ind, rm, warmup, end_idx):
    atr = ind["atr14"]
    if cat == "C":
        be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
        cd = rm.get("cooldown_bars", 0)
        pe = 1 if rm.get("partial_exit") else 0
        pos, _ = bt_contrarian_v3(sig, exit_info, close, high, low, atr,
                                   rm["sl_mult"], rm["tp_mult"], int(rm["max_hold"]),
                                   be, cd, pe, warmup, end_idx, 0)
        return pos
    elif cat == "T":
        exit_l, exit_s = exit_info
        tn = int(rm.get("trail_n", 10))
        rh_key = f"rh_{tn}"
        rl_key = f"rl_{tn}"
        rh = ind.get(rh_key, pd.Series(high).rolling(tn+1, min_periods=1).max().values)
        rl = ind.get(rl_key, pd.Series(low).rolling(tn+1, min_periods=1).min().values)
        tt_map = {"fixed_atr": 0, "chandelier": 1, "parabolic_step": 2}
        tt = tt_map.get(rm.get("trail_type", "fixed_atr"), 0)
        be_val = 1e12 if rm.get("breakeven_thresh") is None else float(rm["breakeven_thresh"])
        ps = rm.get("parabolic_step", 0.02)
        pm = rm.get("parabolic_max", 0.15)
        cd = rm.get("cooldown_bars", 0)
        pos, _ = bt_trend_v3(sig, exit_l, exit_s, close, high, low, atr,
                              rh, rl,
                              rm["initial_sl_mult"], rm.get("trail_atr_mult", 2.5),
                              be_val, tt, ps, pm, cd, warmup, end_idx, 0)
        return pos
    elif cat == "R":
        be = -1.0 if rm.get("breakeven_trigger") is None else rm["breakeven_trigger"]
        cd = rm.get("cooldown_bars", 0)
        td = 1 if rm.get("time_decay") else 0
        pos, _ = bt_range_v3(sig, exit_info, close, high, low, atr,
                              rm["sl_mult"], rm["tp_mult"], int(rm["max_hold"]),
                              be, cd, td, warmup, end_idx, 0)
        return pos
    return np.zeros(len(close))


# ═══════════════════════════════════════════════════
# Per-ticker walk-forward optimization
# ═══════════════════════════════════════════════════

def calc_sharpe(pos, log_ret, mask):
    r = pos[mask] * log_ret[mask]
    if len(r) < 20: return -999
    m = np.mean(r); s = np.std(r, ddof=1)
    return m / s * sqrt(BPY) if s > 1e-12 else 0


def optimize_ticker(sid, ticker, daily_df, kr_s):
    tdf = daily_df[daily_df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP + 200:
        return None

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)
    log_ret = np.zeros(n); log_ret[:-1] = np.log(close[1:]/np.maximum(close[:-1],1e-12))

    ind = compute_base(close, high, low, open_arr, volume, is_hourly=False)
    ind["_volume"] = volume
    rsi = calc_rsi(close, 14)

    # Build caches
    all_windows = sorted(set(SIG_SPACE.get(sid, {}).get("ma_window", []) +
                             SIG_SPACE.get(sid, {}).get("bb_window", []) +
                             SIG_SPACE.get(sid, {}).get("dc_window", [])))
    sma_cache = precompute_sma_cache(close, [w for w in all_windows if w > 0]) if all_windows else {}
    dc_cache = precompute_donchian_cache(high, low, SIG_SPACE.get(sid,{}).get("dc_window",[])) if sid == "S3" else {}

    st_periods = SIG_SPACE.get(sid,{}).get("atr_period",[])
    st_mults = SIG_SPACE.get(sid,{}).get("multiplier",[])
    st_cache = precompute_supertrend_cache(high, low, close, st_periods, st_mults) if sid == "S4" else {}

    vwap_windows = [w for w in SIG_SPACE.get(sid,{}).get("vwap_window",[]) if w > 0]
    vwap_cache = precompute_vwap_cache(close, high, low, volume, vwap_windows) if sid == "S6" else {}
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv)
                  for pv in ["classic","fibonacci","woodie","camarilla"]} if sid == "S5" else {}

    cat = CATEGORY[sid]
    rf_daily = kr_s.reindex(dates, method="ffill").values / BPY
    rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)

    # Walk-forward: for each OOS year
    all_daily_total = []
    all_daily_trad = []
    all_dates_oos = []
    all_pos_oos = []

    for test_year in OOS_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        test_end = n if test_year == max(OOS_YEARS) else _year_bounds(dates, test_year+1)[1]
        if train_end <= WARMUP or train_end >= n: continue

        train_mask = np.zeros(n, dtype=bool); train_mask[WARMUP:train_end] = True
        test_mask = np.zeros(n, dtype=bool); test_mask[train_end:min(test_end,n)] = True
        if test_mask.sum() < 10: continue

        # Stage 1: Random signal search
        sig_params_list = random_sig_params(sid, 500)
        sig_cache = {}
        for sp in sig_params_list:
            key = tuple(sorted(sp.items()))
            if key in sig_cache: continue
            if sid == "S1":
                sig, ei = gen_signals_s1(close, ind, sma_cache, rsi, sp)
            elif sid == "S2":
                sig, ei = gen_signals_s2(close, ind, sma_cache, rsi, sp)
            elif sid == "S3":
                sig, ei = gen_signals_s3(close, high, low, ind, dc_cache, rsi, sp)
            elif sid == "S4":
                sig, ei = gen_signals_s4(close, high, low, ind, st_cache, rsi, sp)
            elif sid == "S5":
                sig, ei = gen_signals_s5(close, ind, pivot_data, rsi, sp)
            elif sid == "S6":
                sig, ei = gen_signals_s6(close, high, low, volume, ind, vwap_cache, rsi, sp)
            sig_cache[key] = (sig, ei)

        # Stage 1+2: Combined signal + RM search
        rm_params_list = random_rm_params(cat, 200)
        best_sh = -999; best_pos_test = None

        for sp in sig_params_list:
            sp_key = tuple(sorted(sp.items()))
            sig, ei = sig_cache[sp_key]
            if np.sum(np.abs(sig[train_mask]) > 0) < 3: continue

            # Sample 20 random RM for this signal
            rm_sample = random.sample(rm_params_list, min(20, len(rm_params_list)))
            for rm in rm_sample:
                try:
                    pos = run_bt(cat, sig, ei, close, high, low, ind, rm, WARMUP, train_end)
                    sh = calc_sharpe(pos, log_ret, train_mask)
                    if sh > best_sh:
                        best_sh = sh
                        # Recompute on test
                        pos_full = run_bt(cat, sig, ei, close, high, low, ind, rm, WARMUP, test_end)
                        best_pos_test = pos_full
                except:
                    continue

        # Stage 3: Refinement — try more RM around best signal
        if best_sh > -900 and best_pos_test is not None:
            # Find best signal key
            best_sp_key = None; best_sh2 = -999
            for sp in sig_params_list:
                sp_key = tuple(sorted(sp.items()))
                sig, ei = sig_cache[sp_key]
                if np.sum(np.abs(sig[train_mask]) > 0) < 3: continue
                for rm in rm_params_list:
                    try:
                        pos = run_bt(cat, sig, ei, close, high, low, ind, rm, WARMUP, train_end)
                        sh = calc_sharpe(pos, log_ret, train_mask)
                        if sh > best_sh2:
                            best_sh2 = sh
                            best_sp_key = sp_key
                            pos_full = run_bt(cat, sig, ei, close, high, low, ind, rm, WARMUP, test_end)
                            best_pos_test = pos_full
                    except:
                        continue

        if best_pos_test is None:
            continue

        # Collect OOS results
        pos_test = best_pos_test[test_mask]
        ret_test = best_pos_test[test_mask] * log_ret[test_mask]
        dates_test = dates[test_mask]
        exp_test = np.abs(pos_test)
        rf_test = rf_daily[test_mask]

        # Total daily return (trading when active, FDR when idle)
        # Commission on entry/exit
        dpos_test = np.diff(pos_test, prepend=0.0)
        comm_test = np.abs(dpos_test) * COMM
        net_test = ret_test - comm_test
        fdr_test = (1 - exp_test) * rf_test
        total_test = net_test + fdr_test

        all_daily_total.extend(total_test)
        all_daily_trad.extend(net_test)
        all_dates_oos.extend(dates_test)
        all_pos_oos.extend(pos_test)

    if len(all_daily_total) < 50:
        return None

    total_arr = np.array(all_daily_total)
    trad_arr = np.array(all_daily_trad)
    dates_arr = np.array(all_dates_oos)
    pos_arr = np.array(all_pos_oos)
    rf_arr = kr_s.reindex(pd.to_datetime(dates_arr), method="ffill").values / BPY
    rf_arr = np.where(np.isfinite(rf_arr), rf_arr, 0.0)

    active = np.abs(pos_arr) > 1e-8
    n_active = active.sum()
    n_total = len(total_arr)
    n_years = n_total / BPY

    # Cumulative trading return (active days only)
    trad_active = trad_arr[active]
    cum_trad = np.prod(1 + trad_active) - 1 if n_active > 0 else 0
    trad_ann = (1 + cum_trad) ** (1/n_years) - 1 if n_years > 0 else 0

    # Trading vol (active days only)
    trad_vol = np.std(trad_active, ddof=1) * sqrt(BPY) if n_active > 10 else 999

    # FDR
    fdr_arr = (1 - np.abs(pos_arr)) * rf_arr
    fdr_ann = np.sum(fdr_arr[~active]) / n_years if n_years > 0 else 0

    # Total
    total_ann = trad_ann + fdr_ann
    rf_ann = np.mean(rf_arr) * BPY
    excess = total_ann - rf_ann
    ex_sharpe = excess / trad_vol if trad_vol > 0.001 and trad_vol < 900 else 0

    # Trades
    n_trades = 0; n_wins = 0; t_ret = 0; in_t = False
    for i in range(len(pos_arr)):
        if abs(pos_arr[i]) > 1e-8 and not in_t:
            in_t = True; t_ret = 0
        if in_t:
            t_ret += trad_arr[i]
            if i == len(pos_arr)-1 or abs(pos_arr[i+1] if i+1<len(pos_arr) else 0) < 1e-8:
                n_trades += 1
                if t_ret > 0: n_wins += 1
                in_t = False
    tpy = n_trades / max(n_years, 0.01)
    wr = n_wins / n_trades * 100 if n_trades > 0 else 0

    return dict(ticker=ticker, ex_sharpe=round(ex_sharpe, 3),
                trad_ret=round(trad_ann*100, 2), trad_vol=round(trad_vol*100, 2),
                fdr_ret=round(fdr_ann*100, 2), total_ret=round(total_ann*100, 2),
                excess=round(excess*100, 2), rf=round(rf_ann*100, 1),
                n_active=n_active, n_total=n_total, tpy=round(tpy, 1), wr=round(wr, 1))


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("="*120)
    print("PER-TICKER 7D OPTIMIZATION — Daily, all 6 strategies, OOS 2022-2025")
    print("="*120)

    daily_df, _, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()

    warmup_numba()

    SIDS = ["S3","S4","S5","S6"]  # S1/S2 already done
    SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
              "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}

    all_results = []

    for sid in SIDS:
        sname = SNAMES[sid]
        print(f"\n{'─'*80}")
        print(f"  {sname}")
        print(f"{'─'*80}")

        sid_results = []
        for ticker in TICKERS:
            t1 = time.time()
            r = optimize_ticker(sid, ticker, daily_df, kr_s)
            elapsed = time.time() - t1
            if r:
                sid_results.append(r)
                flag = "✓" if r["ex_sharpe"] > 0.5 else ("·" if r["ex_sharpe"] > 0 else "✗")
                print(f"    {flag} {ticker}: ExSh={r['ex_sharpe']:+.2f}  "
                      f"TradRet={r['trad_ret']:+.1f}%  Vol={r['trad_vol']:.1f}%  "
                      f"Tr/yr={r['tpy']:.0f}  WR={r['wr']:.0f}%  ({elapsed:.1f}s)")
            else:
                print(f"    ✗ {ticker}: no data ({elapsed:.1f}s)")

        if sid_results:
            mean_sh = np.mean([r["ex_sharpe"] for r in sid_results])
            n_pos = sum(1 for r in sid_results if r["ex_sharpe"] > 0)
            n_gt05 = sum(1 for r in sid_results if r["ex_sharpe"] > 0.5)
            n_gt10 = sum(1 for r in sid_results if r["ex_sharpe"] > 1.0)
            best = max(sid_results, key=lambda r: r["ex_sharpe"])
            worst = min(sid_results, key=lambda r: r["ex_sharpe"])
            print(f"\n  {sname} SUMMARY: mean={mean_sh:+.2f}, >0: {n_pos}/17, "
                  f">0.5: {n_gt05}, >1.0: {n_gt10}")
            print(f"    Best: {best['ticker']} ({best['ex_sharpe']:+.2f}), "
                  f"Worst: {worst['ticker']} ({worst['ex_sharpe']:+.2f})")

        all_results.extend([(sname, r) for r in sid_results])

    # Final summary
    print("\n" + "="*120)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("="*120)
    print(f"{'Strategy':>16s}  {'MeanExSh':>8s}  {'Min':>6s}  {'Max':>6s}  "
          f"{'N>0':>4s}  {'N>0.5':>5s}  {'N>1.0':>5s}  {'MeanTradRet%':>12s}  {'MeanVol%':>8s}")
    for sid in SIDS:
        sname = SNAMES[sid]
        rs = [r for sn, r in all_results if sn == sname]
        if not rs: continue
        shs = [r["ex_sharpe"] for r in rs]
        print(f"{sname:>16s}  {np.mean(shs):8.2f}  {min(shs):6.2f}  {max(shs):6.2f}  "
              f"{sum(1 for s in shs if s>0):4d}  {sum(1 for s in shs if s>0.5):5d}  "
              f"{sum(1 for s in shs if s>1.0):5d}  "
              f"{np.mean([r['trad_ret'] for r in rs]):12.2f}  "
              f"{np.mean([r['trad_vol'] for r in rs]):8.2f}")

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    rows = [{"strategy": sn, **r} for sn, r in all_results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "daily_perticker_7d.csv", index=False)
    print(f"Saved: {OUT_DIR / 'daily_perticker_7d.csv'}")


if __name__ == "__main__":
    main()
