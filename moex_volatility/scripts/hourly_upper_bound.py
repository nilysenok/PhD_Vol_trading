#!/usr/bin/env python3
"""
hourly_upper_bound.py — In-sample optimization on OOS (2022-2026) for HOURLY strategies.

Analogous to daily_upper_bound.py but with:
  - 9 bars/day (MOEX), windows scaled ×9
  - WARMUP = 1800 bars
  - Vol = std(ddof=1) × sqrt(252×9) = sqrt(2268)
  - Commission = 0.04% per side (institutional)
  - FDR: rf/365 per calendar day, distributed across first bar of each day
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
    TICKERS, WARMUP_HOURLY as WARMUP, load_data,
    compute_base, calc_rsi, calc_sma, calc_std, calc_atr, calc_adx,
    calc_supertrend, precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, compute_daily_pivots_for_hourly,
    _year_bounds, warmup_numba,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/hourly_a"
BPH = 2268  # 252 days × 9 bars
COMM = 0.0004  # 0.04% per side
random.seed(42); np.random.seed(42)

N_SIG = 3000; N_RM = 2000; N_TOP = 10; N_REFINE = 500
IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}

# ═══════════════════════════════════════════════════════════════
# MA helpers
# ═══════════════════════════════════════════════════════════════
def _ema(arr, w):
    a=2.0/(w+1);out=np.empty_like(arr);out[0]=arr[0]
    for i in range(1,len(arr)): out[i]=a*arr[i]+(1-a)*out[i-1]
    return out
def _kama(arr, w):
    fc=2.0/3;sc=2.0/31;out=np.full_like(arr,np.nan);out[w-1]=arr[w-1]
    for i in range(w,len(arr)):
        d=abs(arr[i]-arr[i-w]);v=np.sum(np.abs(np.diff(arr[i-w:i+1])))
        er=d/v if v>1e-12 else 0;s=(er*(fc-sc)+sc)**2
        out[i]=out[i-1]+s*(arr[i]-out[i-1])
    return out
def _dema(arr, w):
    e1=_ema(arr,w);e2=_ema(e1,w);return 2*e1-e2
def _hull(arr, w):
    from daily_upper_bound import _wma
    h=max(w//2,1);wh=_wma(arr,h);wf=_wma(arr,w)
    d=2*wh-wf;sw=max(int(sqrt(w)),1);return _wma(d,sw)

MA_TYPES = ["sma","ema","kama","dema"]
# Hourly windows (scaled from daily)
H_WINDOWS = [27,45,63,90,108,135,162,180,225,270,360,450,540,720]

def get_ma(arr, w, mt):
    if mt=="sma": return calc_sma(arr,w)
    elif mt=="ema": return _ema(arr,w)
    elif mt=="kama": return _kama(arr,w)
    elif mt=="dema": return _dema(arr,w)
    return calc_sma(arr,w)


# ═══════════════════════════════════════════════════════════════
# BT engine (same numba core)
# ═══════════════════════════════════════════════════════════════
@njit(cache=True)
def bt_universal(sig, close, high, low, atr, rh, rl,
                 sl_pct, tp_pct, sl_atr, tp_atr, trail,
                 max_hold, be_atr, warmup, end_idx,
                 chandelier_atr=0.0, rr_ratio=0.0):
    n=min(end_idx,len(close));pos=np.zeros(len(close))
    cp=0.0;ep=0.0;sl_p=0.0;tp_p=0.0;held=0;be_on=False;tr_p=0.0
    for t in range(warmup,n):
        if cp!=0.0:
            held+=1;closed=False
            ea=atr[t] if not np.isnan(atr[t]) else atr[t-1] if t>0 else 1.0
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
            if sl_pct>0: sd=ep*sl_pct
            elif sl_atr>0: sd=sl_atr*ea
            elif chandelier_atr>0:
                if cp==1.0: sd=max(rh[t]-ep+chandelier_atr*ea,ea)
                else: sd=max(ep-rl[t]+chandelier_atr*ea,ea)
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

def run_bt(s,close,high,low,atr,rh,rl,rm,warmup,end_idx):
    return bt_universal(s,close,high,low,atr,rh,rl,
        rm.get("sl_p",0),rm.get("tp_p",0),rm.get("sl_a",0),rm.get("tp_a",0),
        rm.get("trail",0),rm.get("mh",0),rm.get("be",0),warmup,end_idx,
        chandelier_atr=rm.get("chand",0.0),rr_ratio=rm.get("rr",0.0))


# ═══════════════════════════════════════════════════════════════
# Metrics — hourly version
# ═══════════════════════════════════════════════════════════════
def calc_excess_sharpe(pos, close, log_ret, h_dates, rf_hourly, cal_factor, mask, comm):
    """Sharpe matching walk-forward formula:
    - net_r = pos * log_ret - |dpos| * comm  (ALL bars, including inactive)
    - sharpe = mean(net_r) / std(net_r, ddof=1) * sqrt(2268)
    - No rf subtraction, no FDR, vol on ALL bars
    """
    pm=pos[mask];lr=log_ret[mask];n=len(pm)
    if n<200: return -999,{}
    dp=np.diff(pm,prepend=0.0)
    net_r=pm*lr-np.abs(dp)*comm  # ALL bars
    active=np.abs(pm)>1e-8;na=active.sum()
    if na<50: return -999,{}
    mean_r=np.mean(net_r);std_r=np.std(net_r,ddof=1)
    if std_r<1e-12: return -999,{}
    # Sharpe = mean/std × sqrt(BPH)  — same as walk-forward
    sharpe=mean_r/std_r*sqrt(BPH)
    # Annual return for display
    ann_ret=mean_r*BPH
    ann_vol=std_r*sqrt(BPH)
    # Trades
    nt=0;nw=0;it=False;tr=0.0
    for i in range(n):
        if abs(pm[i])>1e-8 and not it: it=True;tr=0.0
        if it:
            tr+=net_r[i]
            if i==n-1 or abs(pm[i+1] if i+1<n else 0)<1e-8:
                nt+=1
                if tr>0: nw+=1
                it=False
    d0=pd.Timestamp(h_dates[mask][0]);d1=pd.Timestamp(h_dates[mask][-1])
    cs=max((d1-d0).days,1);ny=cs/365.0
    return sharpe,dict(total_ret=round(ann_ret*100,2),vol=round(ann_vol*100,2),
                       exposure=round(na/n*100,1),tpy=round(nt/max(ny,0.01),1),
                       wr=round(nw/nt*100 if nt>0 else 0,1),n_trades=nt)


def calc_excess_sharpe_fdr(pos, close, log_ret, h_dates, cal_factor, mask, comm):
    """Excess Sharpe WITH FDR — compound annual return, compound rf, vol on active hours.
    cal_factor[t] = rf_daily * calendar_days for first bar of day, 0 otherwise.
    """
    pm=pos[mask];lr=log_ret[mask];cf=cal_factor[mask];n=len(pm)
    dm=h_dates[mask]
    if n<200: return -999,{}
    dp=np.diff(pm,prepend=0.0)
    net=pm*lr-np.abs(dp)*comm
    exp=np.abs(pm)
    fdr=(1.0-exp)*cf  # FDR on free capital
    total=net+fdr
    active=exp>1e-8;na=active.sum()
    if na<50: return -999,{}
    d0=pd.Timestamp(dm[0]);d1=pd.Timestamp(dm[-1])
    cs=max((d1-d0).days,1)
    if cs<30: return -999,{}
    # Compound annual return
    cum=np.prod(1.0+total)-1.0;ann=(1.0+cum)**(365.0/cs)-1.0
    # Compound rf
    rf_cum=np.prod(1.0+cf)-1.0;rf_a=(1.0+rf_cum)**(365.0/cs)-1.0
    # Vol: std of total returns on ACTIVE hours, ddof=1, ×sqrt(2268)
    vol=np.std(total[active],ddof=1)*sqrt(BPH)
    ex=ann-rf_a
    exsh=ex/vol if vol>1e-6 else 0.0
    # Trades
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
    return exsh,dict(total_ret=round(ann*100,2),vol=round(vol*100,2),
                     exposure=round(na/n*100,1),tpy=round(nt/max(ny,0.01),1),
                     wr=round(nw/nt*100 if nt>0 else 0,1),n_trades=nt)


# ═══════════════════════════════════════════════════════════════
# Signal generators
# ═══════════════════════════════════════════════════════════════
def gen_s1(close, ma_cache, atr, params):
    mt=params.get("ma_type","sma");w=params["w"]
    if (mt,w) not in ma_cache: return None
    ma,std=ma_cache[(mt,w)]
    dv=params.get("dev_val",2.0);dt=params.get("dev_type","std")
    dev=atr*dv if dt=="atr" else std*dv
    valid=~np.isnan(ma)&(dev>1e-12)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close<ma-dev)]=1;sig[valid&(close>ma+dev)]=-1
    sig[:WARMUP]=0;return sig

def gen_s2(close, ma_cache, params):
    mt=params.get("ma_type","sma");w=params["w"];bs=params.get("bs",2.0)
    if (mt,w) not in ma_cache: return None
    ma,std=ma_cache[(mt,w)]
    valid=~np.isnan(ma)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close<=ma-bs*std)]=1;sig[valid&(close>=ma+bs*std)]=-1
    sig[:WARMUP]=0;return sig

def gen_s3(close, high, low, dc_cache, params):
    w=params["w"]
    if w not in dc_cache: return None
    hc,lc=dc_cache[w]
    ph=np.empty_like(hc);ph[0]=np.nan;ph[1:]=hc[:-1]
    pl=np.empty_like(lc);pl[0]=np.nan;pl[1:]=lc[:-1]
    valid=~np.isnan(ph)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close>ph)]=1;sig[valid&(close<pl)]=-1
    sig[:WARMUP]=0;return sig

def gen_s4(close, high, low, st_cache, params):
    key=(params["ap"],params["mult"])
    if key not in st_cache: return None
    st,_=st_cache[key]
    sp=np.empty_like(st);sp[0]=np.nan;sp[1:]=st[:-1]
    cp=np.empty_like(close);cp[0]=close[0];cp[1:]=close[:-1]
    valid=~np.isnan(st)&~np.isnan(sp)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close>st)&(cp<=sp)]=1;sig[valid&(close<st)&(cp>=sp)]=-1
    sig[:WARMUP]=0;return sig

def gen_s5(close, pivot_data, params):
    pt=params.get("pt","fibonacci");el=params.get("el",1)
    if pt not in pivot_data: return None
    P,S1,R1,S2,R2,S3,R3=pivot_data[pt]
    if el==2: S,R=S2,R2
    elif el==3: S,R=S3,R3
    else: S,R=S1,R1
    sig=np.zeros(len(close),dtype=np.int8)
    sig[~np.isnan(S)&(close<S)]=1;sig[~np.isnan(R)&(close>R)]=-1
    sig[:WARMUP]=0;return sig

def gen_s6(close, high, low, volume, vwap_cache, params):
    w=params.get("vw",90);dm=params.get("dm",1.5)
    if w not in vwap_cache: return None
    vwap,dev=vwap_cache[w]
    valid=~np.isnan(vwap)
    sig=np.zeros(len(close),dtype=np.int8)
    sig[valid&(close<vwap-dm*dev)]=1;sig[valid&(close>vwap+dm*dev)]=-1
    sig[:WARMUP]=0;return sig


# ═══════════════════════════════════════════════════════════════
# Filters + Random generators
# ═══════════════════════════════════════════════════════════════
def apply_filt(sig, close, volume, ind, rsi, atr, params, is_trend):
    n=len(sig);mask=np.ones(n,dtype=bool);adx=ind["adx14"]
    at=params.get("adx_th",0)
    if at>0:
        if is_trend: mask&=(~np.isnan(adx))&(adx>at)
        else: mask&=(~np.isnan(adx))&(adx<at)
    al=params.get("atr_lo",0)
    if al>0:
        va=atr[~np.isnan(atr)]
        if len(va)>50: mask&=(~np.isnan(atr))&(atr>np.percentile(va,al))
    vt=params.get("vol_th",0)
    if vt>0:
        vs=ind["vol_sma20"];vs_s=np.where(np.isnan(vs)|(vs<1),1.0,vs)
        mask&=volume/vs_s>vt
    rt=params.get("rsi_th",0)
    if rt>0:
        if not is_trend:
            sig[(sig==1)&~((~np.isnan(rsi))&(rsi<rt))]=0
            sig[(sig==-1)&~((~np.isnan(rsi))&(rsi>(100-rt)))]=0
        else:
            sig[(sig==1)&~((~np.isnan(rsi))&(rsi>50))]=0
            sig[(sig==-1)&~((~np.isnan(rsi))&(rsi<50))]=0
    mn=params.get("mom_n",0)
    if mn>0:
        mom=np.zeros(n);mom[mn:]=close[mn:]-close[:-mn]
        if is_trend: sig[(sig==1)&(mom<=0)]=0;sig[(sig==-1)&(mom>=0)]=0
        else: sig[(sig==1)&(mom>=0)]=0;sig[(sig==-1)&(mom<=0)]=0
    sig[~mask]=0;return sig

def random_sig(sid):
    if sid=="S1":
        return {"ma_type":random.choice(MA_TYPES),"w":random.choice(H_WINDOWS),
                "dev_type":random.choice(["std","atr"]),
                "dev_val":random.choice([0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0])}
    elif sid=="S2":
        return {"ma_type":random.choice(MA_TYPES),"w":random.choice(H_WINDOWS),
                "bs":random.choice([0.5,0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0])}
    elif sid=="S3":
        return {"w":random.choice(H_WINDOWS)}
    elif sid=="S4":
        return {"ap":random.choice(H_WINDOWS),
                "mult":random.choice([0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0])}
    elif sid=="S5":
        return {"pt":random.choice(["classic","fibonacci","woodie","camarilla"]),
                "el":random.choice([1,2,3])}
    elif sid=="S6":
        return {"vw":random.choice(H_WINDOWS),
                "dm":random.choice([0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0])}

def random_filt(is_trend):
    if is_trend:
        return {"adx_th":random.choice([0,15,20,25,30]),
                "atr_lo":random.choice([0,5,10,15,20]),
                "vol_th":random.choice([0,0.5,1.0,1.5]),
                "mom_n":random.choice([0,3,5,7,10])}
    else:
        return {"adx_th":random.choice([0,15,20,25,30,35,40]),
                "atr_lo":random.choice([0,5,10,15,20,25]),
                "vol_th":random.choice([0,0.5,0.7,1.0,1.2,1.5]),
                "rsi_th":random.choice([0,20,25,30,35,40]),
                "mom_n":random.choice([0,1,3,5,7,10])}

def random_rm():
    r=random.random();ch=0.0
    if r<0.15: sp=random.choice([0.003,0.005,0.01,0.015,0.02,0.03,0.05]); sa=0.0
    elif r<0.65: sp=0.0;sa=random.choice([0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,8.0])
    else: sp=0.0;sa=0.0;ch=random.choice([1.5,2.0,2.5,3.0,4.0,5.0])
    r2=random.random();rr=0.0
    if r2<0.25: tp=0.0;ta=0.0;tr=random.choice([0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0])
    elif r2<0.5: tp=0.0;ta=0.0;tr=0.0;rr=random.choice([0.5,1.0,1.5,2.0,2.5,3.0,4.0])
    elif r2<0.65: tr=0.0;tp=random.choice([0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10]);ta=0.0
    else: tr=0.0;tp=0.0;ta=random.choice([0.5,1.0,1.5,2.0,3.0,4.0,5.0,8.0,10.0])
    return dict(sl_p=sp,tp_p=tp,sl_a=sa,tp_a=ta,trail=tr,
                mh=random.choice([0,9,18,27,45,90,135,180]),
                be=random.choice([0.0,0.5,1.0,1.5,2.0]),chand=ch,rr=rr)


# ═══════════════════════════════════════════════════════════════
# Per-ticker upper bound
# ═══════════════════════════════════════════════════════════════
def find_upper_bound(args):
    sid,ticker,hourly_data,daily_data,kr_data=args
    hdf=hourly_data[hourly_data["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
    if len(hdf)<WARMUP+500: return sid,ticker,(-999,{}),0

    close=hdf["close"].values.astype(np.float64)
    high=hdf["high"].values.astype(np.float64)
    low=hdf["low"].values.astype(np.float64)
    volume=hdf["volume"].values.astype(np.float64)
    h_dts=pd.to_datetime(hdf["datetime"].values);n=len(close)
    log_ret=np.zeros(n);log_ret[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))

    ind=compute_base(close,high,low,hdf["open"].values.astype(np.float64),volume,is_hourly=True)
    rsi=calc_rsi(close,14);atr=ind["atr14"]
    rh=pd.Series(high).rolling(99,min_periods=1).max().values  # ~11 days
    rl=pd.Series(low).rolling(99,min_periods=1).min().values

    # Calendar factor for FDR: rf per bar
    h_dates=h_dts.normalize()  # date part only
    cal_factor=np.zeros(n)
    rf_raw=kr_data.reindex(h_dates,method="ffill").values
    rf_vals=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
    # Assign daily rf to first bar of each day
    prev_date=None
    for i in range(n):
        d=h_dates[i]
        if d!=prev_date:
            cd=1
            if prev_date is not None:
                cd=(d-prev_date).days
            cal_factor[i]=rf_vals[i]*cd  # full day's rf on first bar
            prev_date=d

    is_trend=IS_TREND[sid]

    # OOS mask: 2022+
    oos_idx=0
    for i in range(n):
        if h_dts[i].year>=2022: oos_idx=i; break
    if oos_idx<=WARMUP: return sid,ticker,(-999,{}),0
    oos_mask=np.zeros(n,dtype=bool);oos_mask[oos_idx:]=True

    # Precompute caches
    ma_cache={}
    if sid in ("S1","S2"):
        for mt in MA_TYPES:
            for w in H_WINDOWS:
                if w>=n-10: continue
                try:
                    ma=get_ma(close,w,mt);std=calc_std(close,w)
                    ma_cache[(mt,w)]=(ma,std)
                except: pass
    dc_cache=precompute_donchian_cache(high,low,H_WINDOWS) if sid=="S3" else {}
    st_cache=precompute_supertrend_cache(high,low,close,H_WINDOWS,
        [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0]) if sid=="S4" else {}
    vwap_cache=precompute_vwap_cache(close,high,low,volume,H_WINDOWS) if sid=="S6" else {}
    pivot_data={}
    if sid=="S5":
        for pv in ["classic","fibonacci","woodie","camarilla"]:
            pivot_data[pv]=compute_daily_pivots_for_hourly(daily_data,ticker,h_dts,pv)

    total_bt=0
    default_rm=dict(sl_p=0,tp_p=0,sl_a=2.0,tp_a=3.0,trail=0,mh=90,be=0,chand=0,rr=0)

    def _gen(sp):
        if sid=="S1": return gen_s1(close,ma_cache,atr,sp)
        elif sid=="S2": return gen_s2(close,ma_cache,sp)
        elif sid=="S3": return gen_s3(close,high,low,dc_cache,sp)
        elif sid=="S4": return gen_s4(close,high,low,st_cache,sp)
        elif sid=="S5": return gen_s5(close,pivot_data,sp)
        elif sid=="S6": return gen_s6(close,high,low,volume,vwap_cache,sp)
        return None

    # Stage 1: Signal search
    sig_results=[]
    for _ in range(N_SIG):
        sp=random_sig(sid);s=_gen(sp)
        if s is None: continue
        fp=random_filt(is_trend)
        s=apply_filt(s,close,volume,ind,rsi,atr,fp,is_trend)
        if np.sum(np.abs(s[oos_mask])>0)<5: continue
        pos=run_bt(s,close,high,low,atr,rh,rl,default_rm,WARMUP,n)
        total_bt+=1
        exsh,_=calc_excess_sharpe(pos,close,log_ret,h_dts,np.zeros(n),cal_factor,oos_mask,COMM)
        if exsh>-900: sig_results.append((exsh,s.copy(),str(sp),str(fp)))
    sig_results.sort(key=lambda x:x[0],reverse=True)
    top_sigs=sig_results[:N_TOP]
    if not top_sigs: return sid,ticker,(-999,{}),total_bt

    # Stage 2: RM search
    best_exsh=-999;best_info={}
    for _,s,sp_str,fp_str in top_sigs:
        for _ in range(N_RM):
            rm=random_rm()
            pos=run_bt(s,close,high,low,atr,rh,rl,rm,WARMUP,n)
            total_bt+=1
            exsh,info=calc_excess_sharpe(pos,close,log_ret,h_dts,np.zeros(n),cal_factor,oos_mask,COMM)
            if exsh>best_exsh:
                best_exsh=exsh;best_info=info
                if rm.get("sl_p",0)>0: sl_t="pct"
                elif rm.get("sl_a",0)>0: sl_t="atr"
                elif rm.get("chand",0)>0: sl_t="chand"
                else: sl_t="def"
                if rm.get("trail",0)>0: tp_t="trail"
                elif rm.get("rr",0)>0: tp_t="rr"
                elif rm.get("tp_p",0)>0: tp_t="pct"
                elif rm.get("tp_a",0)>0: tp_t="atr"
                else: tp_t="def"
                best_info["sl_type"]=sl_t;best_info["tp_type"]=tp_t
                best_info["sig_params"]=sp_str;best_info["filt_params"]=fp_str
                best_info["rm"]=rm.copy()

    # Stage 3: Refinement
    if best_exsh>-900 and best_info.get("rm"):
        best_rm=best_info["rm"]
        for _ in range(N_REFINE):
            rm=best_rm.copy()
            kp=random.choice(list(rm.keys()));v=rm[kp]
            if isinstance(v,float) and v>0: rm[kp]=v*random.uniform(0.7,1.4)
            elif isinstance(v,int) and v>0: rm[kp]=max(1,int(v*random.uniform(0.7,1.4)))
            # need to re-run with best signal
            # use the top signal from stage 1
            s=top_sigs[0][1]
            pos=run_bt(s,close,high,low,atr,rh,rl,rm,WARMUP,n)
            total_bt+=1
            exsh,info=calc_excess_sharpe(pos,close,log_ret,h_dts,np.zeros(n),cal_factor,oos_mask,COMM)
            if exsh>best_exsh:
                best_exsh=exsh;best_info.update(info);best_info["rm"]=rm.copy()

    return sid,ticker,(round(best_exsh,3),best_info),total_bt


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0=time.time()
    print("="*130)
    print("HOURLY UPPER BOUND — In-sample on OOS (2022-2026)")
    print(f"Stages: S1={N_SIG} sigs, S2=top-{N_TOP}×{N_RM} RM, S3={N_REFINE} refine")
    print(f"WARMUP={WARMUP}, BPH={BPH}, COMM={COMM}")
    print("="*130)

    daily_df,hourly_df,_=load_data()
    kr=pd.read_parquet(KR_PATH);kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS=["S1","S2","S3","S4","S5","S6"]
    tasks=[(sid,ticker,hourly_df,daily_df,kr_s) for sid in SIDS for ticker in TICKERS]
    print(f"\nTasks: {len(tasks)}\n")
    results={};total_bt=0

    for i,task in enumerate(tasks):
        sid,ticker=task[0],task[1]
        t1=time.time()
        _,_,result,nbt=find_upper_bound(task)
        elapsed=time.time()-t1
        results[(sid,ticker)]=result;total_bt+=nbt
        if result[0]>-900:
            exsh,info=result
            print(f"  [{i+1:3d}/102] {'✓' if exsh>1.0 else '·'} {SNAMES[sid]:>16s} {ticker:>6s}: "
                  f"ExSh={exsh:+.2f}  Ret={info.get('total_ret',0):+.1f}%  "
                  f"Vol={info.get('vol',0):.0f}%  Exp={info.get('exposure',0):.0f}%  "
                  f"WR={info.get('wr',0):.0f}%  Tr/yr={info.get('tpy',0):.0f}  "
                  f"SL={info.get('sl_type','?')}  TP={info.get('tp_type','?')}  "
                  f"BT={nbt}  ({elapsed:.1f}s)")
        else:
            print(f"  [{i+1:3d}/102] ✗ {SNAMES[sid]:>16s} {ticker:>6s}: no result  ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*130}")
    print("TABLE: Per-ticker ExSh (hourly, in-sample OOS)")
    print(f"{'='*130}")
    header=f"  {'Ticker':>6s}"
    for sid in SIDS: header+=f"  {SNAMES[sid]:>14s}"
    header+=f"  {'BEST':>6s}"
    print(header);print("  "+"-"*110)

    for ticker in TICKERS:
        row=f"  {ticker:>6s}"
        best_v=-999
        for sid in SIDS:
            r=results.get((sid,ticker))
            if r and r[0]>-900: row+=f"  {r[0]:+14.2f}"; best_v=max(best_v,r[0])
            else: row+=f"  {'N/A':>14s}"
        row+=f"  {best_v:+6.2f}" if best_v>-900 else f"  {'N/A':>6s}"
        print((">>>" if best_v>1.0 else "   ")+row[3:])

    # Strategy means
    print(f"\n{'='*130}")
    print("SUMMARY — Hourly strategies")
    print(f"{'='*130}")
    print(f"  {'Strategy':>16s} {'MeanExSh':>8s} {'N>0':>4s} {'N>1.0':>5s} {'N>2.0':>5s} {'Best':>8s} {'Worst':>8s}")
    for sid in SIDS:
        vals=[r[0] for r in [results.get((sid,t)) for t in TICKERS] if r and r[0]>-900]
        if vals:
            print(f"  {SNAMES[sid]:>16s} {np.mean(vals):+8.2f} {sum(1 for x in vals if x>0):4d} "
                  f"{sum(1 for x in vals if x>1.0):5d} {sum(1 for x in vals if x>2.0):5d} "
                  f"{max(vals):+8.2f} {min(vals):+8.2f}")

    # Comparison with daily
    print(f"\n  Comparison: Hourly vs Daily upper bound")
    daily_ub={"S1":1.21,"S2":1.25,"S3":1.03,"S4":1.26,"S5":1.43,"S6":1.22}
    for sid in SIDS:
        vals=[r[0] for r in [results.get((sid,t)) for t in TICKERS] if r and r[0]>-900]
        h_mean=np.mean(vals) if vals else 0
        d_mean=daily_ub.get(sid,0)
        print(f"    {SNAMES[sid]:>16s}: hourly={h_mean:+.2f}  daily={d_mean:+.2f}  diff={h_mean-d_mean:+.2f}")

    # Save
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sid in SIDS:
        for ticker in TICKERS:
            r=results.get((sid,ticker))
            if r and r[0]>-900:
                exsh,info=r
                rows.append({"strategy":SNAMES[sid],"ticker":ticker,"ex_sharpe":exsh,**{k:v for k,v in info.items() if k!="rm"}})
    pd.DataFrame(rows).to_csv(OUT_DIR/"hourly_upper_bound.csv",index=False)

    elapsed=time.time()-t0
    print(f"\n{'='*130}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total bt_universal calls: {total_bt:,}")
    print(f"Saved: {OUT_DIR/'hourly_upper_bound.csv'}")
    print(f"{'='*130}")

if __name__=="__main__":
    main()
