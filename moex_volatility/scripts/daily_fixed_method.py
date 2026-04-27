#!/usr/bin/env python3
"""
daily_fixed_method.py — Walk-forward with FIXED methodology from upper-bound patterns.

Narrow exhaustive grid around dominant patterns found in upper bound:
  S1: KAMA w=[20-35], SL=ATR[2.5-4], TP=ATR[4-6], ADX<[25-35]
  S2: KAMA/DEMA w=[20-35], SL=ATR[3-4.5], TP=RR[4-6], ADX<[25-35]
  S3: Donchian w=[15-30], SL=ATR[2-3], TP=trail[2-4], ADX>[20,25]
  S4: Supertrend ATRp=[14-25] mult=[1.25-2], SL=ATR[2-3], TP=RR[1.5-2.5], ADX>[20,25]
  S5: Fibonacci pivots, el=[1,2,3], SL=ATR[2.5-3.5], TP=ATR[4,5]
  S6: VWAP w=[15-25], band=[1.5-2.5], SL=ATR[2.5-3.5], TP=ATR[4-6]

Common filters: ATR pctile>[10,15], volume>1.0×, momentum=[3,5] bars.
Exhaustive search: 100-500 combos per strategy. Walk-forward expanding window.
"""
import os, sys, warnings, time, itertools
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
OOS_YEARS = [2022, 2023, 2024, 2025]

# ═══════════════════════════════════════════════════════════════
# MA helpers
# ═══════════════════════════════════════════════════════════════
def _kama(arr, w, fast=2, slow=30):
    fc=2.0/(fast+1); sc=2.0/(slow+1)
    out=np.full_like(arr,np.nan); out[w-1]=arr[w-1]
    for i in range(w, len(arr)):
        d=abs(arr[i]-arr[i-w])
        v=np.sum(np.abs(np.diff(arr[i-w:i+1])))
        er=d/v if v>1e-12 else 0
        s=(er*(fc-sc)+sc)**2
        out[i]=out[i-1]+s*(arr[i]-out[i-1])
    return out

def _dema(arr, w):
    a=2.0/(w+1)
    e1=np.empty_like(arr); e1[0]=arr[0]
    for i in range(1,len(arr)): e1[i]=a*arr[i]+(1-a)*e1[i-1]
    e2=np.empty_like(e1); e2[0]=e1[0]
    for i in range(1,len(e1)): e2[i]=a*e1[i]+(1-a)*e2[i-1]
    return 2*e1-e2


# ═══════════════════════════════════════════════════════════════
# BT engine
# ═══════════════════════════════════════════════════════════════
@njit(cache=True)
def bt_universal(sig, close, high, low, atr, rh, rl,
                 sl_atr, tp_atr, trail, max_hold, be_atr,
                 rr_ratio, warmup, end_idx):
    n=min(end_idx,len(close)); pos=np.zeros(len(close))
    cp=0.0;ep=0.0;sl_p=0.0;tp_p=0.0;held=0;be_on=False;tr_p=0.0
    for t in range(warmup,n):
        if cp!=0.0:
            held+=1; closed=False
            ea=atr[t] if not np.isnan(atr[t]) else atr[t-1] if t>0 else 1.0
            if trail>0:
                if cp==1.0:
                    nt=high[t]-trail*ea
                    if nt>tr_p: tr_p=nt
                else:
                    nt=low[t]+trail*ea
                    if nt<tr_p: tr_p=nt
            if be_atr>0 and not be_on:
                if cp==1.0 and (close[t]-ep)>be_atr*ea: be_on=True;sl_p=ep
                elif cp==-1.0 and (ep-close[t])>be_atr*ea: be_on=True;sl_p=ep
            if cp==1.0:
                es=max(sl_p,tr_p) if trail>0 else sl_p
                if low[t]<=es: closed=True
                elif tp_p>0 and trail<=0 and high[t]>=tp_p: closed=True
            else:
                es=min(sl_p,tr_p) if trail>0 else sl_p
                if high[t]>=es: closed=True
                elif tp_p>0 and trail<=0 and low[t]<=tp_p: closed=True
            if max_hold>0 and held>=max_hold: closed=True
            if closed: cp=0.0;held=0;be_on=False
            pos[t]=cp
            if closed: continue
        if cp==0.0 and sig[t]!=0:
            cp=float(sig[t]);ep=close[t];held=0;be_on=False
            ea=atr[t] if not np.isnan(atr[t]) else 1.0
            sd=sl_atr*ea if sl_atr>0 else 3.0*ea
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
                td=tp_atr*ea if tp_atr>0 else 3.0*ea
                if cp==1.0: tp_p=ep+td
                else: tp_p=ep-td
                tr_p=0.0
            pos[t]=cp
    return pos


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════
def calc_metrics(pos, close, log_ret, dates, rf_daily, cal_days, mask, comm):
    pm=pos[mask];lr=log_ret[mask];rf=rf_daily[mask];cd=cal_days[mask]
    dm=dates[mask];n=len(pm)
    if n<30: return -999,{}
    dp=np.diff(pm,prepend=0.0);net=pm*lr-np.abs(dp)*comm
    exp=np.abs(pm);fdr=(1.0-exp)*rf*cd;total=net+fdr
    active=exp>1e-8;na=active.sum()
    if na<10: return -999,{}
    cs=(dm[-1]-dm[0]).days
    if cs<30: return -999,{}
    cum=np.prod(1.0+total)-1.0;ann=(1.0+cum)**(365.0/cs)-1.0
    rf_c=np.prod(1.0+rf*cd)-1.0;rf_a=(1.0+rf_c)**(365.0/cs)-1.0
    vol=np.std(total[active],ddof=1)*sqrt(BPY)
    ex=ann-rf_a;exsh=ex/vol if vol>1e-6 else 0.0
    nt=0;nw=0;it=False;tr=0.0
    for i in range(n):
        if abs(pm[i])>1e-8 and not it: it=True;tr=0.0
        if it:
            tr+=net[i]
            if i==n-1 or abs(pm[i+1] if i+1<n else 0)<1e-8:
                nt+=1
                if tr>0: nw+=1
                it=False
    ny=cs/365.0
    return exsh,dict(total_ret=round(ann*100,2),trad_vol=round(vol*100,2),
                     exposure=round(na/n*100,1),tpy=round(nt/max(ny,0.01),1),
                     wr=round(nw/nt*100 if nt>0 else 0,1),n_trades=nt,rf=round(rf_a*100,1))

def train_sharpe(pos,log_ret,mask):
    r=pos[mask]*log_ret[mask]
    if len(r)<20: return -999
    m=np.mean(r);s=np.std(r,ddof=1)
    return m/s*sqrt(BPY) if s>1e-12 else 0

def count_trades(pa):
    nt=0;it=False
    for v in pa:
        if abs(v)>1e-8 and not it: it=True
        elif abs(v)<1e-8 and it: nt+=1;it=False
    if it: nt+=1
    return nt


# ═══════════════════════════════════════════════════════════════
# FIXED GRIDS per strategy
# ═══════════════════════════════════════════════════════════════

def build_grid_s1():
    """KAMA mean reversion. ~4×4×3×3×2×2=576 combos."""
    combos = []
    for w in [20,25,30,35]:
        for ze in [1.5,2.0,2.5,3.0]:
            for sl in [2.5,3.0,3.5,4.0]:
                for tp in [4.0,5.0,6.0]:
                    for adx in [25,30,35]:
                        for atr_lo in [10,15]:
                            for mom in [3,5]:
                                combos.append(dict(w=w,ze=ze,sl=sl,tp=tp,adx=adx,
                                                   atr_lo=atr_lo,mom=mom))
    return combos

def build_grid_s2():
    """KAMA/DEMA Bollinger. ~2×4×4×4×3×3×2×2=4608→sample 500."""
    combos = []
    for mt in ["kama","dema"]:
        for w in [20,25,30,35]:
            for bs in [1.0,1.5,2.0,2.5]:
                for sl in [3.0,3.5,4.0,4.5]:
                    for rr in [4.0,5.0,6.0]:
                        for adx in [25,30,35]:
                            for atr_lo in [10,15]:
                                for mom in [3,5]:
                                    combos.append(dict(mt=mt,w=w,bs=bs,sl=sl,rr=rr,
                                                       adx=adx,atr_lo=atr_lo,mom=mom))
    return combos

def build_grid_s3():
    """Donchian breakout. ~4×3×3×2×2×2=288 combos."""
    combos = []
    for w in [15,20,25,30]:
        for sl in [2.0,2.5,3.0]:
            for trail in [2.0,3.0,4.0]:
                for adx in [20,25]:
                    for atr_lo in [10,15]:
                        for mom in [3,5]:
                            combos.append(dict(w=w,sl=sl,trail=trail,adx=adx,
                                               atr_lo=atr_lo,mom=mom))
    return combos

def build_grid_s4():
    """Supertrend. ~4×4×3×3×2×2×2=1152→sample 500."""
    combos = []
    for ap in [14,18,20,25]:
        for mult in [1.25,1.5,1.75,2.0]:
            for sl in [2.0,2.5,3.0]:
                for rr in [1.5,2.0,2.5]:
                    for adx in [20,25]:
                        for atr_lo in [10,15]:
                            for mom in [3,5]:
                                combos.append(dict(ap=ap,mult=mult,sl=sl,rr=rr,
                                                   adx=adx,atr_lo=atr_lo,mom=mom))
    return combos

def build_grid_s5():
    """Fibonacci pivots. ~3×4×2×2×2=96 combos."""
    combos = []
    for el in [1,2,3]:
        for sl in [2.5,3.0,3.5,4.0]:
            for tp in [4.0,5.0]:
                for atr_lo in [10,15]:
                    for mom in [3,5]:
                        combos.append(dict(el=el,sl=sl,tp=tp,atr_lo=atr_lo,mom=mom))
    return combos

def build_grid_s6():
    """VWAP deviation. ~3×3×3×3×2×2=324 combos."""
    combos = []
    for vw in [15,20,25]:
        for dm in [1.5,2.0,2.5]:
            for sl in [2.5,3.0,3.5]:
                for tp in [4.0,5.0,6.0]:
                    for atr_lo in [10,15]:
                        for mom in [3,5]:
                            combos.append(dict(vw=vw,dm=dm,sl=sl,tp=tp,
                                               atr_lo=atr_lo,mom=mom))
    return combos


# ═══════════════════════════════════════════════════════════════
# Signal generators
# ═══════════════════════════════════════════════════════════════

def gen_s1(close, w, ze, ma_cache, atr):
    if w not in ma_cache: return None
    ma, std = ma_cache[w]
    dev = std * ze
    valid = ~np.isnan(ma) & (dev > 1e-12)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close < ma - dev)] = 1
    sig[valid & (close > ma + dev)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_s2(close, w, bs, ma_cache):
    if w not in ma_cache: return None
    ma, std = ma_cache[w]
    bb_u = ma + bs*std; bb_l = ma - bs*std
    valid = ~np.isnan(ma)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close <= bb_l)] = 1; sig[valid & (close >= bb_u)] = -1
    sig[:WARMUP] = 0
    return sig

def gen_s3(close, high, low, dc_cache, w):
    if w not in dc_cache: return None
    hc,lc = dc_cache[w]
    ph=np.empty_like(hc);ph[0]=np.nan;ph[1:]=hc[:-1]
    pl=np.empty_like(lc);pl[0]=np.nan;pl[1:]=lc[:-1]
    valid=~np.isnan(ph)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close>ph)]=1;sig[valid&(close<pl)]=-1
    sig[:WARMUP]=0
    return sig

def gen_s4(close, high, low, st_cache, ap, mult):
    key=(ap,mult)
    if key not in st_cache: return None
    st,_=st_cache[key]
    sp=np.empty_like(st);sp[0]=np.nan;sp[1:]=st[:-1]
    cp=np.empty_like(close);cp[0]=close[0];cp[1:]=close[:-1]
    valid=~np.isnan(st)&~np.isnan(sp)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close>st)&(cp<=sp)]=1;sig[valid&(close<st)&(cp>=sp)]=-1
    sig[:WARMUP]=0
    return sig

def gen_s5(close, high, low, pivot_data, el):
    if "fibonacci" not in pivot_data: return None
    P,S1,R1,S2,R2,S3,R3=pivot_data["fibonacci"]
    if el==2: S,R=S2,R2
    elif el==3: S,R=S3,R3
    else: S,R=S1,R1
    sig=np.zeros(len(close),dtype=np.int8)
    sig[~np.isnan(S)&(close<S)]=1;sig[~np.isnan(R)&(close>R)]=-1
    sig[:WARMUP]=0
    return sig

def gen_s6(close, high, low, volume, vwap_cache, vw, dm):
    if vw not in vwap_cache: return None
    vwap,dev=vwap_cache[vw]
    valid=~np.isnan(vwap)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close<vwap-dm*dev)]=1;sig[valid&(close>vwap+dm*dev)]=-1
    sig[:WARMUP]=0
    return sig


# ═══════════════════════════════════════════════════════════════
# Filter application
# ═══════════════════════════════════════════════════════════════

def apply_filt(sig, close, ind, atr, volume, is_trend, adx_th, atr_lo, mom_n):
    n=len(sig);mask=np.ones(n,dtype=bool)
    adx=ind["adx14"]
    if adx_th>0:
        if is_trend: mask&=(~np.isnan(adx))&(adx>adx_th)
        else: mask&=(~np.isnan(adx))&(adx<adx_th)
    if atr_lo>0:
        va=atr[~np.isnan(atr)]
        if len(va)>50: mask&=(~np.isnan(atr))&(atr>np.percentile(va,atr_lo))
    vs=ind["vol_sma20"];vs_s=np.where(np.isnan(vs)|(vs<1),1.0,vs)
    mask&=volume/vs_s>1.0  # fixed: volume > 1.0×
    if mom_n>0:
        mom=np.zeros(n);mom[mom_n:]=close[mom_n:]-close[:-mom_n]
        if is_trend: sig[(sig==1)&(mom<=0)]=0;sig[(sig==-1)&(mom>=0)]=0
        else: sig[(sig==1)&(mom>=0)]=0;sig[(sig==-1)&(mom<=0)]=0
    sig[~mask]=0
    return sig


# ═══════════════════════════════════════════════════════════════
# Per-ticker optimizer
# ═══════════════════════════════════════════════════════════════

IS_TREND={"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}

def optimize_one(args):
    sid,ticker,daily_data,kr_data=args
    tdf=daily_data[daily_data["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    if len(tdf)<WARMUP+200: return sid,ticker,None,0

    close=tdf["close"].values.astype(np.float64)
    high=tdf["high"].values.astype(np.float64)
    low=tdf["low"].values.astype(np.float64)
    volume=tdf["volume"].values.astype(np.float64)
    dates=pd.to_datetime(tdf["date"].values);n=len(close)
    log_ret=np.zeros(n);log_ret[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
    ind=compute_base(close,high,low,tdf["open"].values.astype(np.float64),volume,is_hourly=False)
    atr=ind["atr14"]
    cal_days=np.ones(n)
    for i in range(1,n): cal_days[i]=(dates[i]-dates[i-1]).days
    rf_raw=kr_data.reindex(dates,method="ffill").values
    rf_daily=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
    rh=pd.Series(high).rolling(11,min_periods=1).max().values
    rl=pd.Series(low).rolling(11,min_periods=1).min().values
    is_trend=IS_TREND[sid]

    # Precompute caches
    if sid=="S1":
        ma_cache={w:(_kama(close,w),calc_std(close,w)) for w in [20,25,30,35]}
    elif sid=="S2":
        ma_cache_k={w:(_kama(close,w),calc_std(close,w)) for w in [20,25,30,35]}
        ma_cache_d={w:(_dema(close,w),calc_std(close,w)) for w in [20,25,30,35]}
    dc_cache=precompute_donchian_cache(high,low,[15,20,25,30]) if sid=="S3" else {}
    st_cache=precompute_supertrend_cache(high,low,close,[14,18,20,25],
                [1.25,1.5,1.75,2.0]) if sid=="S4" else {}
    vwap_cache=precompute_vwap_cache(close,high,low,volume,[15,20,25]) if sid=="S6" else {}
    pivot_data={"fibonacci":calc_pivot_daily(high,low,close,"fibonacci")} if sid=="S5" else {}

    # Build grid
    if sid=="S1": grid=build_grid_s1()
    elif sid=="S2": grid=build_grid_s2()
    elif sid=="S3": grid=build_grid_s3()
    elif sid=="S4": grid=build_grid_s4()
    elif sid=="S5": grid=build_grid_s5()
    elif sid=="S6": grid=build_grid_s6()
    else: return sid,ticker,None,0

    # Cap grid at 500 (random sample if larger)
    import random
    if len(grid)>500: grid=random.sample(grid,500)

    all_oos_pos=np.zeros(n);all_oos_mask=np.zeros(n,dtype=bool)
    best_train_sharpes=[];total_bt=0

    for test_year in OOS_YEARS:
        _,train_end=_year_bounds(dates,test_year)
        test_end=n if test_year==max(OOS_YEARS) else _year_bounds(dates,test_year+1)[1]
        if train_end<=WARMUP or train_end>=n: continue
        train_mask=np.zeros(n,dtype=bool);train_mask[WARMUP:train_end]=True
        test_mask=np.zeros(n,dtype=bool);test_mask[train_end:min(test_end,n)]=True
        if test_mask.sum()<10: continue

        best_sh=-999;best_pos=None
        for c in grid:
            # Generate signal
            if sid=="S1":
                sig=gen_s1(close,c["w"],c["ze"],ma_cache,atr)
            elif sid=="S2":
                mc=ma_cache_k if c["mt"]=="kama" else ma_cache_d
                sig=gen_s2(close,c["w"],c["bs"],mc)
            elif sid=="S3":
                sig=gen_s3(close,high,low,dc_cache,c["w"])
            elif sid=="S4":
                sig=gen_s4(close,high,low,st_cache,c["ap"],c["mult"])
            elif sid=="S5":
                sig=gen_s5(close,high,low,pivot_data,c["el"])
            elif sid=="S6":
                sig=gen_s6(close,high,low,volume,vwap_cache,c["vw"],c["dm"])
            else: continue
            if sig is None: continue

            sig=sig.copy()
            sig=apply_filt(sig,close,ind,atr,volume,is_trend,
                          c.get("adx",0),c.get("atr_lo",0),c.get("mom",0))
            if np.sum(np.abs(sig[train_mask])>0)<10: continue

            # BT with fixed RM from grid
            sl_a=c.get("sl",3.0); tp_a=c.get("tp",0.0)
            trail_v=c.get("trail",0.0); rr_v=c.get("rr",0.0)
            pos=bt_universal(sig,close,high,low,atr,rh,rl,
                            sl_a,tp_a,trail_v,0,0.0,rr_v,WARMUP,train_end)
            total_bt+=1
            sh=train_sharpe(pos,log_ret,train_mask)
            if sh>best_sh:
                best_sh=sh;best_pos=pos.copy()
                best_sig=sig.copy();best_rm=c.copy()

        # Robustness: train_sh >= 0.5, >= 15 trades
        if best_sh<0.5 or best_pos is None:
            all_oos_mask|=test_mask;best_train_sharpes.append(0.0);continue
        nt=count_trades(best_pos[:train_end])
        if nt<15:
            all_oos_mask|=test_mask;best_train_sharpes.append(0.0);continue

        # Apply on test
        c=best_rm
        pos_full=bt_universal(best_sig,close,high,low,atr,rh,rl,
                             c.get("sl",3.0),c.get("tp",0.0),c.get("trail",0.0),
                             0,0.0,c.get("rr",0.0),WARMUP,test_end)
        all_oos_pos[test_mask]=pos_full[test_mask]
        all_oos_mask|=test_mask;best_train_sharpes.append(best_sh)

    if all_oos_mask.sum()<50: return sid,ticker,None,total_bt
    exsh,info=calc_metrics(all_oos_pos,close,log_ret,dates,rf_daily,cal_days,all_oos_mask,COMM)
    if not info: info={}
    info["train_sh"]=round(np.mean(best_train_sharpes),2) if best_train_sharpes else 0
    n_traded=sum(1 for s in best_train_sharpes if s>0)
    info["n_traded"]=n_traded;info["n_skipped"]=len(OOS_YEARS)-n_traded
    return sid,ticker,(round(exsh,3),info),total_bt


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0=time.time()
    print("="*120)
    print("FIXED METHODOLOGY — Narrow exhaustive grid from upper-bound patterns")
    grids={"S1":len(build_grid_s1()),"S2":len(build_grid_s2()),
           "S3":len(build_grid_s3()),"S4":len(build_grid_s4()),
           "S5":len(build_grid_s5()),"S6":len(build_grid_s6())}
    for s,n in grids.items(): print(f"  {s}: {n} combos" + (" (→500 sampled)" if n>500 else ""))
    print("Walk-forward: expanding window, robustness: train_sh≥0.5, ≥15 trades")
    print("="*120)

    daily_df,_,_=load_data()
    kr=pd.read_parquet(KR_PATH);kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS=["S1","S2","S3","S4","S5","S6"]
    SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
            "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}

    tasks=[(sid,ticker,daily_df,kr_s) for sid in SIDS for ticker in TICKERS]
    results={};total_bt=0

    for i,task in enumerate(tasks):
        sid,ticker=task[0],task[1]
        t1=time.time()
        _,_,result,nbt=optimize_one(task)
        elapsed=time.time()-t1
        results[(sid,ticker)]=result;total_bt+=nbt
        if result:
            exsh,info=result
            tr=info.get("n_traded",0);sk=info.get("n_skipped",0)
            flag="✓" if exsh>0.5 else ("·" if exsh>0 else ("○" if exsh>-0.05 else "✗"))
            print(f"  [{i+1:3d}/102] {flag} {SNAMES[sid]:>16s} {ticker:>6s}: "
                  f"ExSh={exsh:+.2f}  TrSh={info.get('train_sh',0):.2f}  "
                  f"Ret={info.get('total_ret',0):+.1f}%  Vol={info.get('trad_vol',0):.0f}%  "
                  f"Exp={info.get('exposure',0):.0f}%  WR={info.get('wr',0):.0f}%  "
                  f"T/S={tr}/{sk}  BT={nbt}  ({elapsed:.1f}s)")
        else:
            print(f"  [{i+1:3d}/102] ✗ {SNAMES[sid]:>16s} {ticker:>6s}: no data  ({elapsed:.1f}s)")

    # Summary per strategy
    print(f"\n{'='*120}")
    print("SUMMARY: Mean per-ticker ExSh (arithmetic, fallback=0 for skipped)")
    print(f"{'='*120}")
    print(f"  {'Strategy':>16s} {'MeanExSh':>8s} {'Mean(fb=0)':>10s} {'N>0':>4s} "
          f"{'Best':>8s} {'BestTkr':>8s} {'Worst':>8s}")
    for sid in SIDS:
        sname=SNAMES[sid];vals=[];fb=[]
        best_v=-999;best_t="";worst_v=999
        for ticker in TICKERS:
            r=results.get((sid,ticker))
            if r and r[0]>-900:
                v=r[0];vals.append(v);fb.append(max(v,0.0))
                if v>best_v: best_v=v;best_t=ticker
                if v<worst_v: worst_v=v
            else: fb.append(0.0)
        mr=np.mean(vals) if vals else 0;mf=np.mean(fb)
        np_=sum(1 for v in vals if v>0)
        print(f"  {sname:>16s} {mr:+8.2f} {mf:+10.2f} {np_:4d} "
              f"{best_v:+8.2f} {best_t:>8s} {worst_v:+8.2f}")

    elapsed=time.time()-t0
    print(f"\n  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Total bt_universal calls: {total_bt:,}")

if __name__=="__main__":
    main()
