#!/usr/bin/env python3
"""
daily_full_opt_v5.py — Walk-forward with NARROWED grid from Phase 1-4.

Changes vs v4:
  - Signal params from narrowed_grid.json (deterministic, not random)
  - All params exhaustively enumerated instead of random sampling
  - Same walk-forward, filters, RM search, robustness
"""
import os, sys, warnings, time, random, json
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
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
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/daily_a_v5"
BPY = 252; COMM = 0.0005
OOS_YEARS = [2022, 2023, 2024, 2025]
TRAIN_WINDOW = 2  # 0=expanding (all history), 2=rolling 2yr, 3=rolling 3yr
random.seed(42); np.random.seed(42)

# ── Narrowed grid from Phase 1-4 ──
GRID_PATH = BASE / "results/final/strategies/walkforward_v4/narrowed_grid.json"
with open(GRID_PATH) as f:
    NARROWED_GRID = json.load(f)
print(f"Loaded narrowed grid: " + ", ".join(f"{k}={len(v)}" for k,v in NARROWED_GRID.items() if k!="pyramiding"), flush=True)

# ── Search sizes ──
N_SIG = 0      # NOT USED — exhaustive from grid instead
N_FILT = 3000  # filter combos (top-10 sigs × 300)
N_RM = 5000    # RM combos
N_SS = 9       # signal strength (exhaustive)
N_W = 12       # position sizing (exhaustive)
N_REFINE = 1500
IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
W_SPACE = [0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50]
SS_SPACE = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]


# ═══════════════════════════════════════════════════════════════
# EXTENDED MA TYPES
# ═══════════════════════════════════════════════════════════════
MA_TYPES = ["sma","ema","wma","dema","hull","kama"]
WINDOWS = [5,7,10,15,20,25,30,40,50,60,80,100,150,200]

def _ema(arr, w):
    a = 2.0/(w+1); out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = a*arr[i]+(1-a)*out[i-1]
    return out

def _wma(arr, w):
    wts = np.arange(1,w+1,dtype=np.float64); out = np.full_like(arr, np.nan)
    for i in range(w-1, len(arr)): out[i] = np.dot(arr[i-w+1:i+1], wts)/wts.sum()
    return out

def _hull(arr, w):
    h = max(w//2,1); wh = _wma(arr,h); wf = _wma(arr,w)
    d = 2*wh-wf; sw = max(int(sqrt(w)),1); return _wma(d, sw)

def _kama(arr, w):
    fc=2.0/(2+1); sc=2.0/(30+1); out=np.full_like(arr,np.nan); out[w-1]=arr[w-1]
    for i in range(w, len(arr)):
        dr = abs(arr[i]-arr[i-w])
        vl = np.sum(np.abs(np.diff(arr[i-w:i+1])))
        er = dr/vl if vl>1e-12 else 0
        s = (er*(fc-sc)+sc)**2
        out[i] = out[i-1]+s*(arr[i]-out[i-1])
    return out

def get_ma(arr, w, mt):
    if mt=="sma": return calc_sma(arr, w)
    elif mt=="ema": return _ema(arr, w)
    elif mt=="wma": return _wma(arr, w)
    elif mt=="dema": e1=_ema(arr,w); return 2*e1-_ema(e1,w)
    elif mt=="hull": return _hull(arr, w)
    elif mt=="kama": return _kama(arr, w)
    return calc_sma(arr, w)

def precompute_ma_cache(close, windows=WINDOWS, ma_types=MA_TYPES):
    """Precompute (ma, std) for each (ma_type, window)."""
    cache = {}
    for mt in ma_types:
        for w in windows:
            if w >= len(close) - 10: continue
            try:
                ma = get_ma(close, w, mt)
                std = calc_std(close, w)
                std_safe = np.where((std<1e-12)|np.isnan(std), 1e-12, std)
                cache[(mt, w)] = (ma, std_safe)
            except: pass
    return cache


# ═══════════════════════════════════════════════════════════════
# STRATEGY-SPECIFIC INDICATORS
# ═══════════════════════════════════════════════════════════════

def calc_hurst(close, w=100):
    """Simplified Hurst exponent via rescaled range."""
    n = len(close); h = np.full(n, np.nan)
    log_ret = np.diff(np.log(np.maximum(close, 1e-12)))
    for i in range(w, n):
        r = log_ret[i-w:i]
        m = np.mean(r); y = np.cumsum(r - m)
        R = np.max(y) - np.min(y)
        S = np.std(r, ddof=1)
        if S > 1e-12 and R > 1e-12:
            h[i] = np.log(R/S) / np.log(w)
    return h

def calc_bb_width(close, w=20):
    """Bollinger bandwidth = (upper - lower) / middle."""
    sma = calc_sma(close, w); std = calc_std(close, w)
    return np.where(np.abs(sma) > 1e-12, 2*std/sma, np.nan)

def calc_slope(ma, n=10):
    """Slope of MA over n bars, as % per bar."""
    sl = np.full_like(ma, np.nan)
    for i in range(n, len(ma)):
        if not np.isnan(ma[i]) and not np.isnan(ma[i-n]) and abs(ma[i-n]) > 1e-12:
            sl[i] = (ma[i] - ma[i-n]) / ma[i-n] / n * 100
    return sl


# ═══════════════════════════════════════════════════════════════
# BT ENGINE (unchanged from v3)
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
                    ns=rh[t]-chandelier_atr*ea
                    if ns>sl_p: sl_p=ns
                else:
                    ns=rl[t]+chandelier_atr*ea
                    if ns<sl_p: sl_p=ns
            if trail>0:
                if cp==1.0:
                    nt=high[t]-trail*ea
                    if nt>tr_p: tr_p=nt
                else:
                    nt=low[t]+trail*ea
                    if nt<tr_p: tr_p=nt
            if be_atr>0 and not be_on:
                if cp==1.0 and (close[t]-ep)>be_atr*ea: be_on=True; sl_p=ep
                elif cp==-1.0 and (ep-close[t])>be_atr*ea: be_on=True; sl_p=ep
            if cp==1.0:
                es=max(sl_p,tr_p) if trail>0 else sl_p
                if low[t]<=es: closed=True
                elif tp_p>0 and trail<=0 and high[t]>=tp_p: closed=True
            else:
                es=min(sl_p,tr_p) if trail>0 else sl_p
                if high[t]>=es: closed=True
                elif tp_p>0 and trail<=0 and low[t]<=tp_p: closed=True
            if max_hold>0 and held>=max_hold: closed=True
            if closed: cp=0.0; held=0; be_on=False
            pos[t]=cp
            if closed: continue
        if cp==0.0 and sig[t]!=0:
            cp=float(sig[t]); ep=close[t]; held=0; be_on=False
            ea=atr[t] if not np.isnan(atr[t]) else 1.0
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
# SIGNAL GENERATORS — EXTENDED
# ═══════════════════════════════════════════════════════════════

def gen_sig_s1(close, ma_cache, atr_arr, params):
    mt = params.get("ma_type","sma"); w = params["w"]
    key = (mt, w)
    if key not in ma_cache: return None
    ma, std = ma_cache[key]
    dev_type = params.get("dev_type","std"); dv = params.get("ze", params.get("dev_val",2.0))
    if dev_type == "atr":
        dev = atr_arr * dv
    else:
        dev = std * dv
    valid = ~np.isnan(ma) & ~np.isnan(dev) & (dev > 1e-12)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close < ma - dev)] = 1; sig[valid & (close > ma + dev)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_sig_s2(close, ma_cache, params):
    mt = params.get("ma_type","sma"); w = params["w"]
    key = (mt, w)
    if key not in ma_cache: return None
    ma, std = ma_cache[key]
    bs = params.get("bs", 2.0)
    bb_u = ma + bs*std; bb_l = ma - bs*std
    valid = ~np.isnan(ma)
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
    sig[valid & (close>ph)] = 1; sig[valid & (close<pl)] = -1
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
    sig[valid & (close>st) & (cp<=sp)] = 1; sig[valid & (close<st) & (cp>=sp)] = -1
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

def apply_filters(sig, close, volume, ind, rsi, atr_arr, dow, params, is_trend,
                  hurst=None, bb_width=None, ma_slope=None):
    n = len(sig); mask = np.ones(n, dtype=bool)
    adx = ind["adx14"]

    # ADX filter (direction depends on strategy type)
    at = params.get("adx_th", 0)
    if at > 0:
        if is_trend: mask &= (~np.isnan(adx)) & (adx > at)
        else: mask &= (~np.isnan(adx)) & (adx < at)

    # ATR percentile
    al = params.get("atr_lo", 0)
    if al > 0:
        va = atr_arr[~np.isnan(atr_arr)]
        if len(va) > 50: mask &= (~np.isnan(atr_arr)) & (atr_arr > np.percentile(va, al))

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
    if dw > 0:
        if dw==1: mask &= dow!=0
        elif dw==2: mask &= dow!=4
        elif dw==3: mask &= (dow!=0)&(dow!=4)

    # ── STRATEGY-SPECIFIC ──

    # Hurst filter (counter-trend only): trade when Hurst < threshold
    ht = params.get("hurst_th", 0)
    if ht > 0 and hurst is not None and not is_trend:
        mask &= (~np.isnan(hurst)) & (hurst < ht)

    # BB squeeze filter (counter-trend only): trade when bandwidth < percentile
    sq = params.get("squeeze_pct", 0)
    if sq > 0 and bb_width is not None and not is_trend:
        vw = bb_width[~np.isnan(bb_width)]
        if len(vw) > 50:
            thresh = np.percentile(vw, sq)
            mask &= (~np.isnan(bb_width)) & (bb_width < thresh)

    # Slope filter (counter-trend): don't enter if MA slope too strong
    sl_max = params.get("slope_max", 0)
    if sl_max > 0 and ma_slope is not None and not is_trend:
        mask &= (~np.isnan(ma_slope)) & (np.abs(ma_slope) < sl_max)

    # Trend confirmation filter (trend only): slope must be in direction
    sl_min = params.get("slope_min", 0)
    if sl_min > 0 and ma_slope is not None and is_trend:
        sig[(sig==1) & ((np.isnan(ma_slope))|(ma_slope<sl_min))] = 0
        sig[(sig==-1) & ((np.isnan(ma_slope))|(ma_slope>-sl_min))] = 0

    sig[~mask] = 0
    return sig

# Filter parameter spaces
FILT_COUNTER = {  # for S1, S2, S5, S6
    "adx_th": [0,15,20,25,30,35,40],
    "atr_lo": [0,5,10,15,20,25],
    "vol_th": [0,0.5,0.7,1.0,1.5],
    "rsi_th": [0,20,25,30,35,40],
    "mom_n": [0,1,3,5,7],
    "dow": [0,1,2,3],
    "hurst_th": [0,0.4,0.45,0.5],
    "squeeze_pct": [0,25,40,50,60],
    "slope_max": [0,0.05,0.1,0.2],
}
FILT_TREND = {  # for S3, S4
    "adx_th": [0,15,20,25,30],
    "atr_lo": [0,5,10,15,20,25],
    "vol_th": [0,0.5,0.7,1.0,1.5],
    "rsi_th": [0,25,30,35],
    "mom_n": [0,1,3,5,7,10],
    "dow": [0,1,2,3],
    "slope_min": [0,0.02,0.05,0.1],
}


# ═══════════════════════════════════════════════════════════════
# RANDOM GENERATORS
# ═══════════════════════════════════════════════════════════════

def random_sig_params(sid):
    if sid == "S1":
        return {"ma_type":random.choice(MA_TYPES), "w":random.choice(WINDOWS),
                "dev_type":random.choice(["std","atr"]),
                "ze":random.choice([0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0,5.0])}
    elif sid == "S2":
        return {"ma_type":random.choice(MA_TYPES), "w":random.choice(WINDOWS),
                "bs":random.choice([0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0])}
    elif sid == "S3":
        return {"w":random.choice([3,5,7,10,12,15,18,20,25,30,35,40,50,60,80,100,120])}
    elif sid == "S4":
        return {"ap":random.choice([3,5,7,10,12,14,18,20,25,30,35,40,50,60,80]),
                "mult":random.choice([0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0])}
    elif sid == "S5":
        return {"pt":random.choice(["classic","fibonacci","woodie","camarilla"]), "el":random.choice([1,2,3])}
    elif sid == "S6":
        return {"vw":random.choice([3,5,7,10,15,20,25,30,40,50,60,80,100]),
                "dm":random.choice([0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0])}

def random_filt(is_trend):
    space = FILT_TREND if is_trend else FILT_COUNTER
    return {k: random.choice(v) for k,v in space.items()}

def random_rm():
    r=random.random(); ch=0.0
    if r<0.15: sp=random.choice([0.005,0.01,0.015,0.02,0.03,0.05,0.07,0.10]); sa=0.0
    elif r<0.65: sp=0.0; sa=random.choice([0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,8.0])
    else: sp=0.0; sa=0.0; ch=random.choice([1.5,2.0,2.5,3.0,4.0,5.0])
    r2=random.random(); rr=0.0
    if r2<0.25: tp=0.0; ta=0.0; tr=random.choice([0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0])
    elif r2<0.5: tp=0.0; ta=0.0; tr=0.0; rr=random.choice([0.5,1.0,1.5,2.0,2.5,3.0,4.0])
    elif r2<0.65: tr=0.0; tp=random.choice([0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15]); ta=0.0
    else: tr=0.0; tp=0.0; ta=random.choice([0.5,1.0,1.5,2.0,3.0,4.0,5.0,8.0,10.0])
    return dict(sl_p=sp,tp_p=tp,sl_a=sa,tp_a=ta,trail=tr,
                mh=random.choice([0,3,5,7,10,15,20,30,50]),
                be=random.choice([0.0,0.5,1.0,1.5,2.0]),chand=ch,rr=rr)


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def calc_metrics(pos, close, log_ret, dates, rf_daily, cal_days, mask, comm):
    pm=pos[mask]; lr=log_ret[mask]; rf=rf_daily[mask]; cd=cal_days[mask]
    dm=dates[mask]; n=len(pm)
    if n<30: return -999, {}
    dp=np.diff(pm, prepend=0.0); net=pm*lr-np.abs(dp)*comm
    exp=np.abs(pm); fdr=(1.0-exp)*rf*cd; total=net+fdr
    active=exp>1e-8; na=active.sum()
    if na<10: return -999, {}
    cs=(dm[-1]-dm[0]).days
    if cs<30: return -999, {}
    cum=np.prod(1.0+total)-1.0; ann=(1.0+cum)**(365.0/cs)-1.0
    rf_c=np.prod(1.0+rf*cd)-1.0; rf_a=(1.0+rf_c)**(365.0/cs)-1.0
    vol=np.std(total[active], ddof=1)*sqrt(BPY)
    ex=ann-rf_a; exsh=ex/vol if vol>1e-6 else 0.0
    nt=0; nw=0; in_t=False; tr=0.0
    for i in range(n):
        if abs(pm[i])>1e-8 and not in_t: in_t=True; tr=0.0
        if in_t:
            tr+=net[i]
            if i==n-1 or abs(pm[i+1] if i+1<n else 0)<1e-8:
                nt+=1
                if tr>0: nw+=1
                in_t=False
    ny=cs/365.0
    return exsh, dict(total_ret=round(ann*100,2), excess=round(ex*100,2),
                      trad_vol=round(vol*100,2), exposure=round(na/n*100,1),
                      tpy=round(nt/max(ny,0.01),1), wr=round(nw/nt*100 if nt>0 else 0,1),
                      n_trades=nt, rf=round(rf_a*100,1))

def calc_train_sharpe(pos, log_ret, mask):
    r=pos[mask]*log_ret[mask]
    if len(r)<20: return -999
    m=np.mean(r); s=np.std(r, ddof=1)
    return m/s*sqrt(BPY) if s>1e-12 else 0

def count_trades(pa):
    nt=0; it=False
    for v in pa:
        if abs(v)>1e-8 and not it: it=True
        elif abs(v)<1e-8 and it: nt+=1; it=False
    if it: nt+=1
    return nt

def max_drawdown(pos, log_ret, mask):
    r=pos[mask]*log_ret[mask]; cum=np.cumsum(r)
    pk=np.maximum.accumulate(cum); dd=pk-cum
    return np.max(dd) if len(dd)>0 else 0


# ═══════════════════════════════════════════════════════════════
# PER-TICKER OPTIMIZER
# ═══════════════════════════════════════════════════════════════

def optimize_one(args):
    sid, ticker, daily_data, kr_data = args
    tdf = daily_data[daily_data["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP+200: return sid, ticker, None, 0, {}

    close=tdf["close"].values.astype(np.float64)
    high=tdf["high"].values.astype(np.float64)
    low=tdf["low"].values.astype(np.float64)
    open_arr=tdf["open"].values.astype(np.float64)
    volume=tdf["volume"].values.astype(np.float64)
    dates=pd.to_datetime(tdf["date"].values); n=len(close)
    log_ret=np.zeros(n); log_ret[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))

    ind=compute_base(close,high,low,open_arr,volume,is_hourly=False)
    rsi=calc_rsi(close,14); atr=ind["atr14"]
    cal_days=np.ones(n)
    for i in range(1,n): cal_days[i]=(dates[i]-dates[i-1]).days
    rf_raw=kr_data.reindex(dates,method="ffill").values
    rf_daily=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
    dow=dates.dayofweek.values.astype(np.int32)
    rh=pd.Series(high).rolling(11,min_periods=1).max().values
    rl=pd.Series(low).rolling(11,min_periods=1).min().values

    is_trend=IS_TREND[sid]

    # Precompute caches
    ma_cache = precompute_ma_cache(close) if sid in ("S1","S2") else {}
    dc_w = [3,5,7,10,12,15,18,20,25,30,35,40,50,60,80,100,120]
    dc_cache = precompute_donchian_cache(high,low,dc_w) if sid=="S3" else {}
    st_ap=[3,5,7,10,12,14,18,20,25,30,35,40,50,60,80]
    st_m=[0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0]
    st_cache=precompute_supertrend_cache(high,low,close,st_ap,st_m) if sid=="S4" else {}
    vw=[3,5,7,10,15,20,25,30,40,50,60,80,100]
    vwap_cache=precompute_vwap_cache(close,high,low,volume,vw) if sid=="S6" else {}
    pivot_data={p:calc_pivot_daily(high,low,close,p)
                for p in ["classic","fibonacci","woodie","camarilla"]} if sid=="S5" else {}

    # Strategy-specific indicators
    hurst = calc_hurst(close, 100) if not is_trend else None
    bb_width = calc_bb_width(close, 20) if not is_trend else None
    sma50 = calc_sma(close, 50)
    ma_slope = calc_slope(sma50, 10)
    strength_std = calc_std(close, 50)
    strength_std_safe = np.where((strength_std<1e-12)|np.isnan(strength_std), 1.0, strength_std)
    strength = np.abs(close - sma50) / strength_std_safe
    strength[np.isnan(sma50)] = 0.0

    all_oos_pos=np.zeros(n); all_oos_mask=np.zeros(n,dtype=bool)
    best_train_sharpes=[]; total_bt=0
    sc={"s1":0,"s2":0,"s3":0,"s4":0,"s5":0,"s6":0,"s7":0}

    def _gen_sig(sp):
        if sid=="S1": return gen_sig_s1(close, ma_cache, atr, sp)
        elif sid=="S2": return gen_sig_s2(close, ma_cache, sp)
        elif sid=="S3": return gen_sig_s3(close, high, low, dc_cache, sp)
        elif sid=="S4": return gen_sig_s4(close, high, low, st_cache, sp)
        elif sid=="S5": return gen_sig_s5(close, high, low, pivot_data, sp)
        elif sid=="S6": return gen_sig_s6(close, high, low, volume, vwap_cache, sp)
        return None

    default_rm=dict(sl_p=0,tp_p=0,sl_a=2.0,tp_a=3.0,trail=0,mh=15,be=0,chand=0,rr=0)

    for test_year in OOS_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        test_end = n if test_year==max(OOS_YEARS) else _year_bounds(dates, test_year+1)[1]
        if train_end<=WARMUP or train_end>=n: continue
        # Rolling vs expanding train window
        if TRAIN_WINDOW > 0:
            roll_start_year = test_year - TRAIN_WINDOW
            try:
                train_start_idx = _year_bounds(dates, roll_start_year)[0]
            except:
                train_start_idx = WARMUP
            train_start = max(train_start_idx, WARMUP)
        else:
            train_start = WARMUP
        train_mask=np.zeros(n,dtype=bool); train_mask[train_start:train_end]=True
        test_mask=np.zeros(n,dtype=bool); test_mask[train_end:min(test_end,n)]=True
        if test_mask.sum()<10: continue
        ti=np.where(train_mask)[0]; half=len(ti)//2
        t_h1=np.zeros(n,dtype=bool); t_h1[ti[:half]]=True
        t_h2=np.zeros(n,dtype=bool); t_h2[ti[half:]]=True

        # ── S1: Signal search (deterministic from narrowed grid) ──
        sig_cache={}
        grid_params = NARROWED_GRID.get(sid, [])
        for gp in grid_params:
            # Convert narrowed grid format to sig_params format
            if sid == "S1":
                sp = {"ma_type": gp["ma"], "w": gp["w"],
                      "dev_type": "atr", "dev_val": gp["thresh"]}
            elif sid == "S2":
                sp = {"ma_type": gp["ma"], "w": gp["w"], "bs": gp["bs"]}
            elif sid == "S3":
                sp = {"w": gp["w"]}
            elif sid == "S4":
                sp = {"ap": gp["ap"], "mult": gp["mult"]}
            elif sid == "S5":
                sp = {"pt": gp["pt"], "el": gp["el"]}
            elif sid == "S6":
                sp = {"vw": gp["vw"], "dm": gp["dm"]}
            else:
                continue
            key=str(sp)
            if key in sig_cache: continue
            s=_gen_sig(sp)
            if s is None: continue
            if np.sum(np.abs(s[train_mask])>0)<3: continue
            sig_cache[key]=(s, sp)
        # Also add some random exploration (500 random from original wide grid)
        for _ in range(500):
            sp=random_sig_params(sid)
            key=str(sp)
            if key in sig_cache: continue
            s=_gen_sig(sp)
            if s is None: continue
            if np.sum(np.abs(s[train_mask])>0)<3: continue
            sig_cache[key]=(s, sp)

        top_sigs=[]
        for key,(s,sp) in sig_cache.items():
            pos=run_bt(s,close,high,low,atr,rh,rl,default_rm,WARMUP,train_end)
            total_bt+=1; sc["s1"]+=1
            sh=calc_train_sharpe(pos,log_ret,train_mask)
            top_sigs.append((sh,key))
        top_sigs.sort(key=lambda x:x[0], reverse=True)
        top30=[k for _,k in top_sigs[:30]]
        if not top30: all_oos_mask|=test_mask; continue

        # ── S2: Filter search (top-10 × 300 = 3000) ──
        best_sh=-999; best_cfg=None
        for sig_key in top30[:10]:
            s_base,sp=sig_cache[sig_key]
            for _ in range(300):
                fp=random_filt(is_trend)
                s=s_base.copy()
                s=apply_filters(s,close,volume,ind,rsi,atr,dow,fp,is_trend,
                                hurst,bb_width,ma_slope)
                if np.sum(np.abs(s[train_mask])>0)<20: continue
                pos=run_bt(s,close,high,low,atr,rh,rl,default_rm,WARMUP,train_end)
                total_bt+=1; sc["s2"]+=1
                sh=calc_train_sharpe(pos,log_ret,train_mask)
                if sh>best_sh:
                    best_sh=sh; best_cfg={"key":sig_key,"filt":fp,"sig_arr":s.copy()}
        if best_cfg is None:
            best_cfg={"key":top30[0],"filt":{},"sig_arr":sig_cache[top30[0]][0].copy()}

        # ── S3: RM search (5000) ──
        best_sh3=-999; best_rm={}; best_pos3=None
        s=best_cfg["sig_arr"]
        for _ in range(N_RM):
            rm=random_rm()
            pos=run_bt(s,close,high,low,atr,rh,rl,rm,WARMUP,train_end)
            total_bt+=1; sc["s3"]+=1
            sh=calc_train_sharpe(pos,log_ret,train_mask)
            if sh>best_sh3: best_sh3=sh; best_rm=rm.copy(); best_pos3=pos.copy()

        # ── S4: Signal strength (9 exhaustive) ──
        best_sh4=best_sh3; best_sig4=s.copy()
        for ss in SS_SPACE:
            if ss==0.0: continue
            s_t=s.copy(); act=np.abs(s_t)>0
            if act.sum()<5: continue
            cutoff=np.percentile(strength[act], ss*100)
            s_t[act & (strength<cutoff)]=0
            if np.sum(np.abs(s_t[train_mask])>0)<10: continue
            pos=run_bt(s_t,close,high,low,atr,rh,rl,best_rm,WARMUP,train_end)
            total_bt+=1; sc["s4"]+=1
            sh=calc_train_sharpe(pos,log_ret,train_mask)
            if sh>best_sh4: best_sh4=sh; best_sig4=s_t.copy(); best_pos3=pos.copy()

        # ── S5: Position sizing (12 exhaustive) ──
        best_w=1.0; best_sh5=best_sh4
        if best_pos3 is not None:
            for w in W_SPACE:
                pw=best_pos3*w
                sw=calc_train_sharpe(pw,log_ret,train_mask)
                sc["s5"]+=1
                if sw>best_sh5: best_sh5=sw; best_w=w

        # ── S6: Refinement (1500) ──
        s_final=best_sig4
        for _ in range(N_REFINE):
            rm=best_rm.copy()
            kp=random.choice(list(rm.keys())); v=rm[kp]
            if isinstance(v,float) and v>0: rm[kp]=v*random.uniform(0.7,1.4)
            elif isinstance(v,int) and v>0: rm[kp]=max(1,int(v*random.uniform(0.7,1.4)))
            pos=run_bt(s_final,close,high,low,atr,rh,rl,rm,WARMUP,train_end)
            total_bt+=1; sc["s6"]+=1
            sh=calc_train_sharpe(pos,log_ret,train_mask)
            if sh>best_sh4: best_sh4=sh; best_rm=rm.copy(); best_pos3=pos.copy()

        # ── S7: Robustness ──
        sc["s7"]+=1; skip=None
        if best_sh4<0.5: skip=f"low_sh({best_sh4:.2f})"
        if skip is None and best_pos3 is not None:
            nt=count_trades(best_pos3[:train_end])
            if nt<20: skip=f"few_tr({nt})"
        if skip is None and best_pos3 is not None:
            sh1=calc_train_sharpe(best_pos3,log_ret,t_h1)
            sh2=calc_train_sharpe(best_pos3,log_ret,t_h2)
            if sh1<0.2 or sh2<0.2: skip=f"inconsist"
        if skip is None and best_pos3 is not None:
            mdd=max_drawdown(best_pos3,log_ret,train_mask)
            if mdd>0.30: skip=f"mdd({mdd:.0%})"

        if skip:
            all_oos_mask|=test_mask; best_train_sharpes.append(0.0); continue

        # Apply on test
        pos_full=run_bt(s_final,close,high,low,atr,rh,rl,best_rm,WARMUP,test_end)
        all_oos_pos[test_mask]=pos_full[test_mask]*best_w
        all_oos_mask|=test_mask; best_train_sharpes.append(best_sh4)

    if all_oos_mask.sum()<50: return sid, ticker, None, total_bt, sc

    exsh, info = calc_metrics(all_oos_pos,close,log_ret,dates,rf_daily,cal_days,all_oos_mask,COMM)
    if not info: info={}
    info["train_sh"]=round(np.mean(best_train_sharpes),2) if best_train_sharpes else 0
    va=atr[~np.isnan(atr)]; vc=close[close>0]
    info["atr_pct"]=round(np.mean(va)/np.mean(vc)*100,2) if len(va)>0 else 0
    n_traded=sum(1 for sh in best_train_sharpes if sh>0)
    info["n_traded_yrs"]=n_traded; info["n_skipped_yrs"]=len(OOS_YEARS)-n_traded
    return sid, ticker, (round(exsh,3), info), total_bt, sc


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0=time.time()
    print("="*130)
    tw_str = f"rolling {TRAIN_WINDOW}yr" if TRAIN_WINDOW > 0 else "expanding (all history)"
    print(f"FULL DAILY OPTIMIZATION v5 — NARROWED GRID + Walk-Forward")
    print(f"Train window: {tw_str}")
    print(f"Signal: narrowed grid (deterministic) + 500 random exploration")
    print(f"Stages: S1=grid+500rand S2={N_FILT} S3={N_RM} S4={N_SS} S5={N_W} S6={N_REFINE} S7=robustness")
    print(f"MA types: {MA_TYPES}  |  Counter-trend filters: Hurst, BB squeeze, slope")
    print("="*130)

    daily_df,_,_=load_data()
    kr=pd.read_parquet(KR_PATH); kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS=["S1","S2","S3","S4","S5","S6"]
    SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
            "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}

    tasks=[(sid,ticker,daily_df,kr_s) for sid in SIDS for ticker in TICKERS]
    print(f"\nTasks: {len(tasks)}")
    exp_bt=len(tasks)*4*(N_SIG+N_FILT+N_RM+N_SS+N_W+N_REFINE)
    print(f"Expected ~{exp_bt:,} bt_universal calls\n")

    results={}; all_sc={}; total_bt=0
    for i,task in enumerate(tasks):
        sid,ticker=task[0],task[1]
        t1=time.time()
        _,_,result,nbt,sc=optimize_one(task)
        elapsed=time.time()-t1
        results[(sid,ticker)]=result; all_sc[(sid,ticker)]=sc; total_bt+=nbt
        if result:
            exsh,info=result
            tr=info.get("n_traded_yrs",0); sk=info.get("n_skipped_yrs",0)
            flag="✓" if exsh>0.5 else ("·" if exsh>0 else ("○" if exsh>-0.05 else "✗"))
            print(f"  [{i+1:3d}/{len(tasks)}] {flag} {SNAMES[sid]:>16s} {ticker:>6s}: "
                  f"ExSh={exsh:+.2f}  TrSh={info.get('train_sh',0):.2f}  "
                  f"Ret={info.get('total_ret',0):+.1f}%  Vol={info.get('trad_vol',0):.0f}%  "
                  f"Exp={info.get('exposure',0):.0f}%  WR={info.get('wr',0):.0f}%  "
                  f"T/S={tr}/{sk}  BT={nbt}  ({elapsed:.1f}s)")
        else:
            print(f"  [{i+1:3d}/{len(tasks)}] ✗ {SNAMES[sid]:>16s} {ticker:>6s}: no data  "
                  f"BT={nbt}  ({elapsed:.1f}s)")

    # ══════════════════════════════════════════════════════════
    # Per-strategy tables
    # ══════════════════════════════════════════════════════════
    for sid in SIDS:
        sname=SNAMES[sid]
        print(f"\n{'='*130}")
        print(f"TABLE: {sname}")
        print(f"{'='*130}")
        print(f"  {'Ticker':>6s} {'ExSh':>7s} {'TotRet%':>7s} {'Vol%':>5s} {'Exp%':>5s} "
              f"{'WR%':>5s} {'Tr/yr':>5s} {'TrSh':>5s} {'T/S':>5s}")
        print("  "+"-"*80)
        exsh_fb0=[]
        for ticker in TICKERS:
            r=results.get((sid,ticker))
            if not r or r[0]<=-900:
                print(f"  {ticker:>6s} {'FBK':>7s}  → ExSh=0.0"); exsh_fb0.append(0.0); continue
            exsh,info=r
            tr=info.get("n_traded_yrs",0); sk=info.get("n_skipped_yrs",0)
            flag=">>>" if exsh>0 else "   "
            print(f"{flag}{ticker:>6s} {exsh:+7.2f} {info.get('total_ret',0):7.1f} "
                  f"{info.get('trad_vol',0):5.0f} {info.get('exposure',0):5.0f} "
                  f"{info.get('wr',0):5.0f} {info.get('tpy',0):5.1f} "
                  f"{info.get('train_sh',0):5.2f} {tr}T/{sk}S")
            exsh_fb0.append(max(exsh,0.0))
        m_raw=np.mean([r[0] if (r:=results.get((sid,t))) and r[0]>-900 else 0.0 for t in TICKERS])
        m_fb=np.mean(exsh_fb0)
        n_pos=sum(1 for v in exsh_fb0 if v>0)
        print(f"  {'MEAN':>6s} {m_raw:+7.2f} (raw)  {m_fb:+7.2f} (fb=0)  N_trade={n_pos}/{17}")

    # Summary
    print(f"\n{'='*130}")
    print("SUMMARY")
    print(f"{'='*130}")
    print(f"  {'Strategy':>16s} {'MeanExSh':>8s} {'Mean(fb=0)':>10s} {'N>0':>4s} "
          f"{'Best':>8s} {'BestTkr':>8s} {'Worst':>8s}")
    for sid in SIDS:
        sname=SNAMES[sid]; vals=[]; fb=[]
        best_v=-999; best_t=""; worst_v=999; worst_t=""
        for ticker in TICKERS:
            r=results.get((sid,ticker))
            if r and r[0]>-900:
                v=r[0]; vals.append(v); fb.append(max(v,0.0))
                if v>best_v: best_v=v; best_t=ticker
                if v<worst_v: worst_v=v; worst_t=ticker
            else: fb.append(0.0)
        mr=np.mean(vals) if vals else 0; mf=np.mean(fb)
        np_=sum(1 for v in vals if v>0)
        print(f"  {sname:>16s} {mr:+8.2f} {mf:+10.2f} {np_:4d} "
              f"{best_v:+8.2f} {best_t:>8s} {worst_v:+8.2f}")

    # Stage counters
    print(f"\n  Stage counters (avg per task):")
    for sn in ["s1","s2","s3","s4","s5","s6","s7"]:
        vals=[all_sc.get(k,{}).get(sn,0) for k in all_sc]; vals=[v for v in vals if v>0]
        if vals: print(f"    {sn}: avg={np.mean(vals):.0f}  total={sum(vals):,}")

    # Save
    save_rows=[]
    for sid in SIDS:
        for ticker in TICKERS:
            r=results.get((sid,ticker))
            if r and r[0]>-900:
                exsh,info=r
                save_rows.append({"strategy":SNAMES[sid],"ticker":ticker,"ex_sharpe":exsh,**info})
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(save_rows).to_csv(OUT_DIR/"daily_full_opt_v5_results.csv",index=False)

    elapsed=time.time()-t0
    print(f"\n{'='*130}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total bt_universal calls: {total_bt:,}")
    print(f"Avg per task: {total_bt//len(tasks):,} calls, {elapsed/len(tasks):.1f}s")
    print(f"Saved: {OUT_DIR/'daily_full_opt_v5_results.csv'}")
    print(f"{'='*130}")

if __name__=="__main__":
    main()
