#!/usr/bin/env python3
"""
daily_upper_bound.py — In-sample optimization on OOS (2022-2026).

Finds the UPPER BOUND of Excess Sharpe per ticker × strategy.
This is NOT a valid backtest — it optimizes on test data.
Purpose: determine if alpha EXISTS and find parameter patterns.

2-stage: Stage1=5000 signal combos × default RM → top-10.
         Stage2=top-10 × 3000 RM combos = 30,000.
Total per task: ~35,000 bt_universal calls.
102 tasks × 35K = 3.57M calls. At 0.08ms = ~5 min.

Extended MA types for S1/S2: SMA, EMA, WMA, DEMA, Hull, KAMA.
Strategy-specific filters: counter-trend vs trend vs range.
"""
import os, sys, warnings, time, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP, load_data, compute_base, calc_rsi, calc_sma, calc_std,
    calc_atr, calc_adx, calc_supertrend, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache, calc_pivot_daily,
    _year_bounds, warmup_numba,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/daily_a"
BPY = 252; COMM = 0.0005
random.seed(42); np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# EXTENDED MA TYPES
# ═══════════════════════════════════════════════════════════════

def calc_ema(arr, w):
    alpha = 2.0 / (w + 1)
    out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def calc_wma(arr, w):
    weights = np.arange(1, w+1, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    for i in range(w-1, len(arr)):
        out[i] = np.dot(arr[i-w+1:i+1], weights) / weights.sum()
    return out

def calc_dema(arr, w):
    e1 = calc_ema(arr, w); e2 = calc_ema(e1, w)
    return 2*e1 - e2

def calc_hull(arr, w):
    half = max(w//2, 1)
    wma_half = calc_wma(arr, half)
    wma_full = calc_wma(arr, w)
    diff = 2*wma_half - wma_full
    sw = max(int(sqrt(w)), 1)
    return calc_wma(diff, sw)

def calc_kama(arr, w, fast=2, slow=30):
    fc = 2.0/(fast+1); sc = 2.0/(slow+1)
    out = np.full_like(arr, np.nan); out[w-1] = arr[w-1]
    for i in range(w, len(arr)):
        direction = abs(arr[i] - arr[i-w])
        volatility = np.sum(np.abs(np.diff(arr[i-w:i+1])))
        if volatility < 1e-12: er = 0
        else: er = direction / volatility
        sc_val = (er * (fc - sc) + sc) ** 2
        out[i] = out[i-1] + sc_val * (arr[i] - out[i-1])
    return out

def get_ma(arr, w, ma_type):
    if ma_type == "sma": return calc_sma(arr, w)
    elif ma_type == "ema": return calc_ema(arr, w)
    elif ma_type == "wma": return calc_wma(arr, w)
    elif ma_type == "dema": return calc_dema(arr, w)
    elif ma_type == "hull": return calc_hull(arr, w)
    elif ma_type == "kama": return calc_kama(arr, w)
    else: return calc_sma(arr, w)


# ═══════════════════════════════════════════════════════════════
# BT ENGINE (same as daily_full_opt)
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def bt_universal(sig, close, high, low, atr, rh, rl,
                 sl_pct, tp_pct, sl_atr, tp_atr, trail,
                 max_hold, be_atr, warmup, end_idx,
                 chandelier_atr=0.0, rr_ratio=0.0):
    n = min(end_idx, len(close))
    pos = np.zeros(len(close))
    cp=0.0; ep=0.0; sl_p=0.0; tp_p=0.0; held=0; be_on=False; tr_p=0.0
    for t in range(warmup, n):
        if cp != 0.0:
            held += 1; closed = False
            ea = atr[t] if not np.isnan(atr[t]) else atr[t-1] if t>0 else 1.0
            if chandelier_atr>0 and not be_on:
                if cp==1.0:
                    ns = rh[t]-chandelier_atr*ea
                    if ns>sl_p: sl_p=ns
                else:
                    ns = rl[t]+chandelier_atr*ea
                    if ns<sl_p: sl_p=ns
            if trail>0:
                if cp==1.0:
                    nt = high[t]-trail*ea
                    if nt>tr_p: tr_p=nt
                else:
                    nt = low[t]+trail*ea
                    if nt<tr_p: tr_p=nt
            if be_atr>0 and not be_on:
                if cp==1.0 and (close[t]-ep)>be_atr*ea: be_on=True; sl_p=ep
                elif cp==-1.0 and (ep-close[t])>be_atr*ea: be_on=True; sl_p=ep
            if cp==1.0:
                es = max(sl_p,tr_p) if trail>0 else sl_p
                if low[t]<=es: closed=True
                elif tp_p>0 and trail<=0 and high[t]>=tp_p: closed=True
            else:
                es = min(sl_p,tr_p) if trail>0 else sl_p
                if high[t]>=es: closed=True
                elif tp_p>0 and trail<=0 and low[t]<=tp_p: closed=True
            if max_hold>0 and held>=max_hold: closed=True
            if closed: cp=0.0; held=0; be_on=False
            pos[t]=cp
            if closed: continue
        if cp==0.0 and sig[t]!=0:
            cp=float(sig[t]); ep=close[t]; held=0; be_on=False
            ea = atr[t] if not np.isnan(atr[t]) else 1.0
            if sl_pct>0: sd=ep*sl_pct
            elif sl_atr>0: sd=sl_atr*ea
            elif chandelier_atr>0:
                if cp==1.0: sd=max(rh[t]-ep+chandelier_atr*ea, ea)
                else: sd=max(ep-rl[t]+chandelier_atr*ea, ea)
            else: sd=3.0*ea
            if cp==1.0: sl_p=ep-sd
            else: sl_p=ep+sd
            if trail>0:
                tp_p=0.0
                if cp==1.0: tr_p=ep-trail*ea
                else: tr_p=ep+trail*ea
            elif rr_ratio>0:
                td=sd*rr_ratio
                if cp==1.0: tp_p=ep+td
                else: tp_p=ep-td
                tr_p=0.0
            else:
                if tp_pct>0: td=ep*tp_pct
                elif tp_atr>0: td=tp_atr*ea
                else: td=3.0*ea
                if cp==1.0: tp_p=ep+td
                else: tp_p=ep-td
                tr_p=0.0
            pos[t]=cp
    return pos

def run_bt(s, close, high, low, atr, rh, rl, rm, warmup, end_idx):
    return bt_universal(s, close, high, low, atr, rh, rl,
                        rm.get("sl_p",0), rm.get("tp_p",0),
                        rm.get("sl_a",0), rm.get("tp_a",0),
                        rm.get("trail",0), rm.get("mh",0), rm.get("be",0),
                        warmup, end_idx,
                        chandelier_atr=rm.get("chand",0.0),
                        rr_ratio=rm.get("rr",0.0))


# ═══════════════════════════════════════════════════════════════
# METRICS — same proven formula
# ═══════════════════════════════════════════════════════════════

def calc_excess_sharpe(pos, close, log_ret, dates, rf_daily, cal_days, mask, comm):
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    dm = dates[mask]; n = len(pm)
    if n < 30: return -999, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm*lr - np.abs(dp)*comm
    exp = np.abs(pm)
    fdr = (1.0-exp)*rf*cd
    total = net + fdr
    active = exp > 1e-8; na = active.sum()
    if na < 10: return -999, {}
    cs = (dm[-1]-dm[0]).days
    if cs < 30: return -999, {}
    cum = np.prod(1.0+total)-1.0
    ann = (1.0+cum)**(365.0/cs)-1.0
    rf_c = np.prod(1.0+rf*cd)-1.0
    rf_a = (1.0+rf_c)**(365.0/cs)-1.0
    vol = np.std(total[active], ddof=1)*sqrt(BPY)
    ex = ann - rf_a
    exsh = ex/vol if vol>1e-6 else 0.0
    # trades
    nt=0; nw=0; in_t=False; tr=0.0
    for i in range(n):
        if abs(pm[i])>1e-8 and not in_t: in_t=True; tr=0.0
        if in_t:
            tr += net[i]
            if i==n-1 or abs(pm[i+1] if i+1<n else 0)<1e-8:
                nt+=1
                if tr>0: nw+=1
                in_t=False
    ny = cs/365.0
    info = dict(total_ret=round(ann*100,2), excess=round(ex*100,2),
                vol=round(vol*100,2), exposure=round(na/n*100,1),
                tpy=round(nt/max(ny,0.01),1), wr=round(nw/nt*100 if nt>0 else 0,1),
                n_trades=nt)
    return exsh, info


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATORS — EXTENDED
# ═══════════════════════════════════════════════════════════════

def gen_sig_s1_ext(close, std_arr, ma_arr, params):
    """Mean reversion: buy when price < MA - dev, sell when > MA + dev."""
    dev_type = params.get("dev_type", "std")  # "std" or "atr_mult"
    dev_val = params.get("dev_val", 2.0)
    if dev_type == "std":
        dev = std_arr * dev_val
    else:
        dev = params.get("atr_arr", std_arr) * dev_val
    valid = ~np.isnan(ma_arr) & ~np.isnan(dev) & (dev > 1e-12)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close < ma_arr - dev)] = 1
    sig[valid & (close > ma_arr + dev)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_sig_s2_ext(close, std_arr, ma_arr, params):
    """Bollinger: same as S1 but with band std as deviation."""
    bs = params.get("bs", 2.0)
    bb_u = ma_arr + bs * std_arr; bb_l = ma_arr - bs * std_arr
    valid = ~np.isnan(ma_arr)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close <= bb_l)] = 1; sig[valid & (close >= bb_u)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_sig_s3(close, high, low, dc_cache, params):
    w = params["w"]
    if w not in dc_cache: return None
    hc, lc = dc_cache[w]
    ph = np.empty_like(hc); ph[0]=np.nan; ph[1:]=hc[:-1]
    pl = np.empty_like(lc); pl[0]=np.nan; pl[1:]=lc[:-1]
    valid = ~np.isnan(ph)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close > ph)] = 1; sig[valid & (close < pl)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_sig_s4(close, high, low, st_cache, params):
    key = (params["ap"], params["mult"])
    if key not in st_cache: return None
    st, _ = st_cache[key]
    sp = np.empty_like(st); sp[0]=np.nan; sp[1:]=st[:-1]
    cp = np.empty_like(close); cp[0]=close[0]; cp[1:]=close[:-1]
    valid = ~np.isnan(st) & ~np.isnan(sp)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close>st) & (cp<=sp)] = 1
    sig[valid & (close<st) & (cp>=sp)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_sig_s5(close, high, low, pivot_data, params):
    pt = params.get("pt","classic"); el = params.get("el",1)
    if pt not in pivot_data: return None
    P,S1,R1,S2,R2,S3,R3 = pivot_data[pt]
    if el==2: S,R=S2,R2
    elif el==3: S,R=S3,R3
    else: S,R=S1,R1
    sig = np.zeros(len(close), dtype=np.int8)
    sig[~np.isnan(S) & (close<S)] = 1; sig[~np.isnan(R) & (close>R)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_sig_s6(close, high, low, volume, vwap_cache, params):
    w = params.get("vw",20); dm = params.get("dm",1.5)
    if w not in vwap_cache: return None
    vwap, dev = vwap_cache[w]
    valid = ~np.isnan(vwap)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close<vwap-dm*dev)] = 1; sig[valid & (close>vwap+dm*dev)] = -1
    sig[:WARMUP] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# STRATEGY-SPECIFIC FILTERS
# ═══════════════════════════════════════════════════════════════

def apply_filters_generic(sig, close, volume, ind, rsi, atr_arr, day_of_week, params, is_trend):
    n = len(sig); mask = np.ones(n, dtype=bool)
    adx = ind["adx14"]
    # ADX
    at = params.get("adx_th", 0)
    if at > 0:
        if is_trend: mask &= (~np.isnan(adx)) & (adx > at)
        else: mask &= (~np.isnan(adx)) & (adx < at)
    # ATR percentile
    al = params.get("atr_lo", 0)
    if al > 0:
        va = atr_arr[~np.isnan(atr_arr)]
        if len(va)>50: mask &= (~np.isnan(atr_arr)) & (atr_arr > np.percentile(va, al))
    # Volume
    vt = params.get("vol_th", 0)
    if vt > 0:
        vs = ind["vol_sma20"]; vs_s = np.where(np.isnan(vs)|(vs<1), 1.0, vs)
        mask &= volume/vs_s > vt
    # RSI
    rt = params.get("rsi_th", 0)
    if rt > 0:
        if not is_trend:
            sig[(sig==1) & ~((~np.isnan(rsi))&(rsi<rt))] = 0
            sig[(sig==-1) & ~((~np.isnan(rsi))&(rsi>(100-rt)))] = 0
        else:
            sig[(sig==1) & ~((~np.isnan(rsi))&(rsi>50))] = 0
            sig[(sig==-1) & ~((~np.isnan(rsi))&(rsi<50))] = 0
    # Momentum
    mn = params.get("mom_n", 0)
    if mn > 0:
        mom = np.zeros(n); mom[mn:] = close[mn:]-close[:-mn]
        if is_trend: sig[(sig==1)&(mom<=0)]=0; sig[(sig==-1)&(mom>=0)]=0
        else: sig[(sig==1)&(mom>=0)]=0; sig[(sig==-1)&(mom<=0)]=0
    # DOW
    dw = params.get("dow", 0)
    if dw > 0 and day_of_week is not None:
        if dw==1: mask &= day_of_week!=0
        elif dw==2: mask &= day_of_week!=4
        elif dw==3: mask &= (day_of_week!=0)&(day_of_week!=4)
    sig[~mask] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# RANDOM GENERATORS
# ═══════════════════════════════════════════════════════════════

MA_TYPES = ["sma","ema","wma","dema","hull","kama"]
WINDOWS = [3,5,7,10,12,15,18,20,25,30,35,40,50,60,80,100,120,150,200]
STD_DEVS = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0]
ATR_DEVS = [0.3,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0,5.0]
BB_STDS = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0]

def random_s1_params():
    return {"ma_type": random.choice(MA_TYPES),
            "w": random.choice(WINDOWS),
            "dev_type": random.choice(["std","atr"]),
            "dev_val": random.choice(STD_DEVS if random.random()<0.5 else ATR_DEVS)}

def random_s2_params():
    return {"ma_type": random.choice(MA_TYPES),
            "w": random.choice(WINDOWS),
            "bs": random.choice(BB_STDS)}

def random_s3_params():
    return {"w": random.choice([3,5,7,10,12,15,18,20,25,30,35,40,50,60,80,100,120])}

def random_s4_params():
    return {"ap": random.choice([3,5,7,10,12,14,18,20,25,30,35,40,50,60,80]),
            "mult": random.choice([0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0,5.0,6.0])}

def random_s5_params():
    return {"pt": random.choice(["classic","fibonacci","woodie","camarilla"]),
            "el": random.choice([1,2,3])}

def random_s6_params():
    return {"vw": random.choice([3,5,7,10,12,15,18,20,25,30,40,50,60,80,100]),
            "dm": random.choice([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0])}

def random_filter_params(is_trend):
    p = {}
    if is_trend:
        p["adx_th"] = random.choice([0,15,20,25,30])
    else:
        p["adx_th"] = random.choice([0,15,20,25,30,35,40])
    p["atr_lo"] = random.choice([0,5,10,15,20,25])
    p["vol_th"] = random.choice([0,0.5,0.7,1.0,1.2,1.5])
    p["rsi_th"] = random.choice([0,20,25,30,35,40])
    p["mom_n"] = random.choice([0,1,2,3,5,7,10])
    p["dow"] = random.choice([0,1,2,3])
    return p

def random_rm():
    r = random.random(); chand=0.0
    if r<0.15: sl_p=random.choice([0.005,0.01,0.015,0.02,0.03,0.05,0.07,0.10]); sl_a=0.0
    elif r<0.65: sl_p=0.0; sl_a=random.choice([0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,8.0])
    else: sl_p=0.0; sl_a=0.0; chand=random.choice([1.5,2.0,2.5,3.0,4.0,5.0])
    r2 = random.random(); rr=0.0
    if r2<0.25: tp_p=0.0; tp_a=0.0; trail=random.choice([0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0])
    elif r2<0.5: tp_p=0.0; tp_a=0.0; trail=0.0; rr=random.choice([0.5,1.0,1.5,2.0,2.5,3.0,4.0])
    elif r2<0.65: trail=0.0; tp_p=random.choice([0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15]); tp_a=0.0
    else: trail=0.0; tp_p=0.0; tp_a=random.choice([0.5,1.0,1.5,2.0,3.0,4.0,5.0,8.0,10.0])
    return dict(sl_p=sl_p, tp_p=tp_p, sl_a=sl_a, tp_a=tp_a, trail=trail,
                mh=random.choice([0,3,5,7,10,15,20,30,50]),
                be=random.choice([0.0,0.5,1.0,1.5,2.0]), chand=chand, rr=rr)

IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}

N_SIG = 5000  # signal combos
N_RM = 3000   # RM combos per top signal
N_TOP = 10    # top signals to search RM


# ═══════════════════════════════════════════════════════════════
# PER-TICKER UPPER BOUND
# ═══════════════════════════════════════════════════════════════

def find_upper_bound(args):
    sid, ticker, daily_data, kr_data = args
    tdf = daily_data[daily_data["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP+200: return sid, ticker, None, 0

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)
    log_ret = np.zeros(n); log_ret[:-1] = np.log(close[1:]/np.maximum(close[:-1],1e-12))

    ind = compute_base(close, high, low, tdf["open"].values.astype(np.float64), volume, is_hourly=False)
    rsi = calc_rsi(close, 14); atr = ind["atr14"]
    cal_days = np.ones(n)
    for i in range(1, n): cal_days[i] = (dates[i]-dates[i-1]).days
    rf_raw = kr_data.reindex(dates, method="ffill").values
    rf_daily = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0
    dow = dates.dayofweek.values.astype(np.int32)
    rh = pd.Series(high).rolling(11, min_periods=1).max().values
    rl = pd.Series(low).rolling(11, min_periods=1).min().values

    is_trend = IS_TREND[sid]

    # OOS mask: 2022-2026
    oos_start = _year_bounds(dates, 2022)[0]
    if oos_start <= WARMUP: return sid, ticker, None, 0
    oos_mask = np.zeros(n, dtype=bool); oos_mask[oos_start:] = True

    # Precompute caches
    dc_windows = [3,5,7,10,12,15,18,20,25,30,35,40,50,60,80,100,120]
    dc_cache = precompute_donchian_cache(high, low, dc_windows) if sid=="S3" else {}
    st_ap = [3,5,7,10,12,14,18,20,25,30,35,40,50,60,80]
    st_m = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0,5.0,6.0]
    st_cache = precompute_supertrend_cache(high,low,close,st_ap,st_m) if sid=="S4" else {}
    vw = [3,5,7,10,12,15,18,20,25,30,40,50,60,80,100]
    vwap_cache = precompute_vwap_cache(close,high,low,volume,vw) if sid=="S6" else {}
    pivot_data = {p: calc_pivot_daily(high,low,close,p)
                  for p in ["classic","fibonacci","woodie","camarilla"]} if sid=="S5" else {}

    # Precompute MA caches for S1/S2
    ma_cache = {}
    if sid in ("S1","S2"):
        for mt in MA_TYPES:
            for w in WINDOWS:
                try:
                    ma = get_ma(close, w, mt)
                    std = calc_std(close, w)
                    ma_cache[(mt,w)] = (ma, std)
                except: pass

    total_bt = 0
    default_rm = dict(sl_p=0,tp_p=0,sl_a=2.0,tp_a=3.0,trail=0,mh=15,be=0,chand=0,rr=0)

    # ── STAGE 1: Signal search (5000 random combos) ──
    sig_results = []  # (sharpe, sig_array, params_str)
    for _ in range(N_SIG):
        if sid == "S1":
            sp = random_s1_params()
            key = (sp["ma_type"], sp["w"])
            if key not in ma_cache: continue
            ma, std = ma_cache[key]
            sp["atr_arr"] = atr
            s = gen_sig_s1_ext(close, std, ma, sp)
        elif sid == "S2":
            sp = random_s2_params()
            key = (sp["ma_type"], sp["w"])
            if key not in ma_cache: continue
            ma, std = ma_cache[key]
            s = gen_sig_s2_ext(close, std, ma, sp)
        elif sid == "S3":
            sp = random_s3_params()
            s = gen_sig_s3(close, high, low, dc_cache, sp)
        elif sid == "S4":
            sp = random_s4_params()
            s = gen_sig_s4(close, high, low, st_cache, sp)
        elif sid == "S5":
            sp = random_s5_params()
            s = gen_sig_s5(close, high, low, pivot_data, sp)
        elif sid == "S6":
            sp = random_s6_params()
            s = gen_sig_s6(close, high, low, volume, vwap_cache, sp)
        else: continue

        if s is None: continue

        # Apply random filter
        fp = random_filter_params(is_trend)
        s = apply_filters_generic(s, close, volume, ind, rsi, atr, dow, fp, is_trend)
        if np.sum(np.abs(s[oos_mask]) > 0) < 5: continue

        pos = run_bt(s, close, high, low, atr, rh, rl, default_rm, WARMUP, n)
        total_bt += 1
        exsh, _ = calc_excess_sharpe(pos, close, log_ret, dates, rf_daily, cal_days, oos_mask, COMM)
        if exsh > -900:
            sig_results.append((exsh, s.copy(), str(sp), str(fp)))

    sig_results.sort(key=lambda x: x[0], reverse=True)
    top_sigs = sig_results[:N_TOP]
    if not top_sigs:
        return sid, ticker, (-999, {}), total_bt

    # ── STAGE 2: RM search for top signals ──
    best_exsh = -999; best_info = {}; best_params = ""
    for exsh_s1, s, sp_str, fp_str in top_sigs:
        for _ in range(N_RM):
            rm = random_rm()
            pos = run_bt(s, close, high, low, atr, rh, rl, rm, WARMUP, n)
            total_bt += 1
            exsh, info = calc_excess_sharpe(pos, close, log_ret, dates, rf_daily, cal_days, oos_mask, COMM)
            if exsh > best_exsh:
                best_exsh = exsh; best_info = info
                # Determine SL/TP type
                if rm.get("sl_p",0)>0: sl_t="pct"
                elif rm.get("sl_a",0)>0: sl_t="atr"
                elif rm.get("chand",0)>0: sl_t="chand"
                else: sl_t="def"
                if rm.get("trail",0)>0: tp_t="trail"
                elif rm.get("rr",0)>0: tp_t="rr"
                elif rm.get("tp_p",0)>0: tp_t="pct"
                elif rm.get("tp_a",0)>0: tp_t="atr"
                else: tp_t="def"
                best_info["sl_type"] = sl_t
                best_info["tp_type"] = tp_t
                best_info["rm"] = rm.copy()
                best_info["sig_params"] = sp_str
                best_info["filt_params"] = fp_str

    # ── Position sizing sweep ──
    if best_exsh > -900 and best_info:
        # Re-run best with w sweep
        # (skip for speed — w=1.0 is the upper bound)
        pass

    return sid, ticker, (round(best_exsh, 3), best_info), total_bt


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 130)
    print("UPPER BOUND SEARCH — In-sample optimization on OOS (2022-2026)")
    print(f"Stage 1: {N_SIG} signal combos (extended MA types). Stage 2: top-{N_TOP} × {N_RM} RM combos.")
    print("This is NOT a valid backtest. Purpose: find maximum possible ExSh.")
    print("=" * 130)

    daily_df, _, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS = ["S1","S2","S3","S4","S5","S6"]
    SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
              "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}

    tasks = [(sid, ticker, daily_df, kr_s) for sid in SIDS for ticker in TICKERS]
    print(f"\nTasks: {len(tasks)}")
    total_bt = 0; results = {}

    for i, task in enumerate(tasks):
        sid, ticker = task[0], task[1]
        t1 = time.time()
        _, _, result, nbt = find_upper_bound(task)
        elapsed = time.time() - t1
        results[(sid,ticker)] = result
        total_bt += nbt
        if result and result[0] > -900:
            exsh, info = result
            flag = "✓" if exsh>1.0 else ("·" if exsh>0 else "✗")
            print(f"  [{i+1:3d}/{len(tasks)}] {flag} {SNAMES[sid]:>16s} {ticker:>6s}: "
                  f"ExSh={exsh:+.2f}  Ret={info.get('total_ret',0):+.1f}%  "
                  f"Vol={info.get('vol',0):.0f}%  Exp={info.get('exposure',0):.0f}%  "
                  f"WR={info.get('wr',0):.0f}%  Tr/yr={info.get('tpy',0):.0f}  "
                  f"SL={info.get('sl_type','?')}  TP={info.get('tp_type','?')}  "
                  f"BT={nbt}  ({elapsed:.1f}s)")
        else:
            print(f"  [{i+1:3d}/{len(tasks)}] ✗ {SNAMES[sid]:>16s} {ticker:>6s}: no result  BT={nbt}  ({elapsed:.1f}s)")

    # ══════════════════════════════════════════════════════════
    # FULL PARAMETER TABLE — 102 rows with all params
    # ══════════════════════════════════════════════════════════
    import ast
    print(f"\n{'='*200}")
    print("FULL PARAMETER TABLE — all 102 upper-bound configs")
    print(f"{'='*200}")
    print(f"{'#':>3s} {'Strat':>5s} {'Tkr':>5s} {'ExSh':>6s} {'Ret%':>5s} {'Vol%':>4s} "
          f"{'Exp%':>4s} {'WR%':>4s} {'T/yr':>4s} "
          f"{'SL_t':>5s} {'SL_v':>6s} {'TP_t':>5s} {'TP_v':>6s} {'MH':>3s} {'BE':>4s} "
          f"{'SigParams':>40s} {'Filters':>50s}")
    print("-"*200)

    csv_rows = []
    for idx, (sid, ticker) in enumerate([(s,t) for s in SIDS for t in TICKERS]):
        r = results.get((sid, ticker))
        if not r or r[0] <= -900: continue
        exsh, info = r
        rm = info.get("rm", {})
        sp_str = info.get("sig_params", "")
        fp_str = info.get("filt_params", "")
        sl_t = info.get("sl_type", "?"); tp_t = info.get("tp_type", "?")

        # SL value
        if sl_t=="pct": sl_v = rm.get("sl_p",0)
        elif sl_t=="atr": sl_v = rm.get("sl_a",0)
        elif sl_t=="chand": sl_v = rm.get("chand",0)
        else: sl_v = 0
        # TP value
        if tp_t=="trail": tp_v = rm.get("trail",0)
        elif tp_t=="rr": tp_v = rm.get("rr",0)
        elif tp_t=="pct": tp_v = rm.get("tp_p",0)
        elif tp_t=="atr": tp_v = rm.get("tp_a",0)
        else: tp_v = 0

        # Parse sig_params from string
        try: sp_dict = ast.literal_eval(sp_str)
        except: sp_dict = {}
        # Compact sig params
        sp_compact = " ".join(f"{k}={v}" for k,v in sp_dict.items()
                              if k not in ("atr_arr",))

        # Parse filter params
        try: fp_dict = ast.literal_eval(fp_str)
        except: fp_dict = {}
        fp_compact = " ".join(f"{k}={v}" for k,v in fp_dict.items() if v)

        print(f"{idx+1:3d} {sid:>5s} {ticker:>5s} {exsh:+6.2f} "
              f"{info.get('total_ret',0):5.1f} {info.get('vol',0):4.0f} "
              f"{info.get('exposure',0):4.0f} {info.get('wr',0):4.0f} "
              f"{info.get('tpy',0):4.0f} "
              f"{sl_t:>5s} {sl_v:6.2f} {tp_t:>5s} {tp_v:6.2f} "
              f"{rm.get('mh',0):3.0f} {rm.get('be',0):4.1f} "
              f"{sp_compact:>40s} {fp_compact:>50s}")

        csv_rows.append({
            "strategy": SNAMES[sid], "ticker": ticker, "ex_sharpe": exsh,
            "total_ret": info.get("total_ret",0), "vol": info.get("vol",0),
            "exposure": info.get("exposure",0), "wr": info.get("wr",0),
            "tpy": info.get("tpy",0),
            "sl_type": sl_t, "sl_val": round(sl_v,4),
            "tp_type": tp_t, "tp_val": round(tp_v,4),
            "max_hold": rm.get("mh",0), "be_atr": rm.get("be",0),
            "sig_params": sp_compact, "filt_params": fp_compact,
            **{f"rm_{k}": v for k,v in rm.items()},
        })

    # ══════════════════════════════════════════════════════════
    # PATTERN ANALYSIS per strategy
    # ══════════════════════════════════════════════════════════
    from collections import Counter
    print(f"\n{'='*130}")
    print("PATTERN ANALYSIS — dominant params per strategy")
    print(f"{'='*130}")

    for sid in SIDS:
        sname = SNAMES[sid]
        configs = []
        for ticker in TICKERS:
            r = results.get((sid, ticker))
            if r and r[0] > -900: configs.append(r[1])
        if not configs: continue

        # SL/TP
        sl_c = Counter(c.get("sl_type","?") for c in configs)
        tp_c = Counter(c.get("tp_type","?") for c in configs)

        # RM values
        rms = [c.get("rm",{}) for c in configs if c.get("rm")]
        sl_vals = [rm.get("sl_a",0) or rm.get("sl_p",0) or rm.get("chand",0) for rm in rms]
        tp_vals = [rm.get("tp_a",0) or rm.get("tp_p",0) or rm.get("trail",0) or rm.get("rr",0) for rm in rms]
        mh_vals = [rm.get("mh",0) for rm in rms]
        be_vals = [rm.get("be",0) for rm in rms]

        # Signal params
        sig_params = []
        for c in configs:
            try: sig_params.append(ast.literal_eval(c.get("sig_params","{}")))
            except: pass

        # MA type distribution (S1/S2)
        ma_str = ""
        if sid in ("S1","S2") and sig_params:
            ma_c = Counter(sp.get("ma_type","?") for sp in sig_params)
            w_vals = [sp.get("w",0) for sp in sig_params if sp.get("w",0)>0]
            ma_str = f"MA={ma_c.most_common()}  AvgWindow={np.mean(w_vals):.0f}"
        elif sid == "S3" and sig_params:
            w_vals = [sp.get("w",0) for sp in sig_params if sp.get("w",0)>0]
            ma_str = f"AvgWindow={np.mean(w_vals):.0f}  Median={np.median(w_vals):.0f}"
        elif sid == "S4" and sig_params:
            ap_vals = [sp.get("ap",0) for sp in sig_params if sp.get("ap",0)>0]
            m_vals = [sp.get("mult",0) for sp in sig_params if sp.get("mult",0)>0]
            ma_str = f"AvgATRp={np.mean(ap_vals):.0f}  AvgMult={np.mean(m_vals):.1f}"
        elif sid == "S5" and sig_params:
            pt_c = Counter(sp.get("pt","?") for sp in sig_params)
            el_c = Counter(sp.get("el",0) for sp in sig_params)
            ma_str = f"PivotType={pt_c.most_common()}  Level={el_c.most_common()}"
        elif sid == "S6" and sig_params:
            vw_vals = [sp.get("vw",0) for sp in sig_params if sp.get("vw",0)>0]
            dm_vals = [sp.get("dm",0) for sp in sig_params if sp.get("dm",0)>0]
            ma_str = f"AvgVWAPw={np.mean(vw_vals):.0f}  AvgDevMult={np.mean(dm_vals):.1f}"

        # Filter analysis
        filt_params = []
        for c in configs:
            try: filt_params.append(ast.literal_eval(c.get("filt_params","{}")))
            except: pass
        filt_str = ""
        if filt_params:
            for fk in ["adx_th","atr_lo","vol_th","rsi_th","mom_n"]:
                vals = [fp.get(fk,0) for fp in filt_params if fp.get(fk,0)>0]
                if vals:
                    filt_str += f" {fk}:avg={np.mean(vals):.0f}({len(vals)}/17)"

        avg_exp = np.mean([c.get("exposure",0) for c in configs])
        avg_wr = np.mean([c.get("wr",0) for c in configs])
        avg_tpy = np.mean([c.get("tpy",0) for c in configs])
        avg_exsh = np.mean([r[0] for r in [results.get((sid,t)) for t in TICKERS]
                           if r and r[0]>-900])

        print(f"\n  {sname} (mean ExSh={avg_exsh:+.2f}, avgExp={avg_exp:.0f}%, "
              f"avgWR={avg_wr:.0f}%, avgTr/yr={avg_tpy:.0f}):")
        print(f"    SL: {sl_c.most_common()}  avgVal={np.mean(sl_vals):.2f}")
        print(f"    TP: {tp_c.most_common()}  avgVal={np.mean(tp_vals):.2f}")
        print(f"    MaxHold: avg={np.mean(mh_vals):.0f}  BE: avg={np.mean(be_vals):.1f}")
        if ma_str: print(f"    Signal: {ma_str}")
        if filt_str: print(f"    Filters:{filt_str}")

    # ══════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("SUMMARY — Mean ExSh per strategy")
    print(f"{'='*130}")
    print(f"  {'Strategy':>16s} {'MeanExSh':>8s} {'N>0':>4s} {'N>0.5':>5s} {'N>1.0':>5s} {'Best':>8s} {'Worst':>8s}")
    for sid in SIDS:
        vals = [r[0] for r in [results.get((sid,t)) for t in TICKERS] if r and r[0]>-900]
        if vals:
            print(f"  {SNAMES[sid]:>16s} {np.mean(vals):+8.2f} {sum(1 for x in vals if x>0):4d} "
                  f"{sum(1 for x in vals if x>0.5):5d} {sum(1 for x in vals if x>1.0):5d} "
                  f"{max(vals):+8.2f} {min(vals):+8.2f}")

    # Save CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "upper_bound_params.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*130}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total bt_universal calls: {total_bt:,}")
    print(f"Saved: {OUT_DIR / 'upper_bound_params.csv'}")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
