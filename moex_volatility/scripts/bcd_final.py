#!/usr/bin/env python3
"""
bcd_final.py — B/C/D on top of FIXED A positions.

A is FIXED (not recomputed):
  Daily A: compound FDR metric (from daily_upper_bound.py original)
  Hourly A: mean/std × sqrt(2268) metric (from hourly_upper_bound.py v2)

B/C/D modify positions, then compute with SAME metric as A.
Position clamped to [-1, 1].
25 vol methods × 5000 random combos per B/C/D.
"""
import os, sys, warnings, time, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
import ast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP, WARMUP_HOURLY, load_data, compute_base, calc_rsi,
    calc_sma, calc_std, calc_atr, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache, calc_pivot_daily,
    compute_daily_pivots_for_hourly, _year_bounds, warmup_numba,
)
from daily_upper_bound import (
    gen_sig_s1_ext, gen_sig_s2_ext, gen_sig_s3, gen_sig_s4, gen_sig_s5, gen_sig_s6,
    apply_filters_generic, get_ma, run_bt,
)
from hourly_upper_bound import (
    gen_s1 as h_gen_s1, gen_s2 as h_gen_s2, gen_s3 as h_gen_s3,
    gen_s4 as h_gen_s4, gen_s5 as h_gen_s5, gen_s6 as h_gen_s6,
    apply_filt as h_apply_filt, get_ma as h_get_ma,
    run_bt as h_run_bt,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
BPY=252; BPH=2268; COMM_D=0.0005; COMM_H=0.0004
SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
        "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND={"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP={v:k for k,v in SNAMES.items()}
random.seed(42); np.random.seed(42)
N_BCD = 5000

# Fixed A values
A_DAILY = {"S1":1.21,"S2":1.25,"S3":1.03,"S4":1.26,"S5":1.43,"S6":1.22}
A_HOURLY = {"S1":1.65,"S2":1.67,"S3":2.20,"S4":3.24,"S5":1.68,"S6":1.65}


# ═══════════════════════════════════════════════════════════════
# METRICS — EXACT SAME as original A computations
# ═══════════════════════════════════════════════════════════════

def daily_metric(pos, log_ret, dates, rf_daily, cal_days, mask, comm):
    """SAME as daily_upper_bound.py calc_excess_sharpe (compound FDR, vol active)."""
    pm=np.clip(pos[mask],-1,1); lr=log_ret[mask]; rf=rf_daily[mask]; cd=cal_days[mask]
    dm=dates[mask]; n=len(pm)
    if n<30: return -999
    dp=np.diff(pm,prepend=0.0); net=pm*lr-np.abs(dp)*comm
    exp=np.abs(pm); fdr=(1.0-exp)*rf*cd; total=net+fdr
    active=exp>1e-8; na=active.sum()
    if na<10: return -999
    cs=(dm[-1]-dm[0]).days
    if cs<30: return -999
    cum=np.prod(1.0+total)-1.0; ann=(1.0+cum)**(365.0/cs)-1.0
    rf_c=np.prod(1.0+rf*cd)-1.0; rf_a=(1.0+rf_c)**(365.0/cs)-1.0
    vol=np.std(total[active],ddof=1)*sqrt(BPY)
    ex=ann-rf_a
    return ex/vol if vol>1e-6 else -999

def hourly_metric(pos, log_ret, cal_factor, mask, comm):
    """Hourly with FDR — compound annual, vol on active hours, same as WF."""
    pm=np.clip(pos[mask],-1,1); lr=log_ret[mask]; cf=cal_factor[mask]; n=len(pm)
    if n<200: return -999
    dp=np.diff(pm,prepend=0.0)
    net=pm*lr-np.abs(dp)*comm
    exp=np.abs(pm); fdr=(1.0-exp)*cf; total=net+fdr
    active=exp>1e-8; na=active.sum()
    if na<50: return -999
    vol=np.std(total[active],ddof=1)*sqrt(BPH)
    if vol<1e-6: return -999
    # Compound annual
    cum=np.prod(1.0+total)-1.0
    # Calendar span
    n_trading_days = n / 9.0  # approx
    cs = n_trading_days / 252.0 * 365.0  # approx calendar days
    if cs < 30: return -999
    ann=(1.0+cum)**(365.0/cs)-1.0
    rf_c=np.prod(1.0+cf)-1.0; rf_a=(1.0+rf_c)**(365.0/cs)-1.0
    return (ann-rf_a)/vol


# ═══════════════════════════════════════════════════════════════
# 25 VOL FORECASTS
# ═══════════════════════════════════════════════════════════════

def _rstd(lr,w):
    n=len(lr);o=np.full(n,np.nan)
    for i in range(w,n): o[i]=np.std(lr[i-w:i],ddof=1)
    return o
def _ratr(atr,w):
    n=len(atr);o=np.full(n,np.nan)
    for i in range(w,n): o[i]=np.mean(atr[i-w:i])
    return o
def _ewma(lr,lam):
    n=len(lr);v=np.full(n,np.nan);v[1]=lr[0]**2
    for i in range(2,n):
        pv=v[i-1] if not np.isnan(v[i-1]) else lr[i-1]**2
        v[i]=lam*pv+(1-lam)*lr[i-1]**2
    return np.sqrt(np.maximum(v,1e-20))
def _park(h,l,w):
    n=len(h);o=np.full(n,np.nan)
    hl2=np.log(np.maximum(h,1e-12)/np.maximum(l,1e-12))**2
    for i in range(w,n): o[i]=sqrt(np.mean(hl2[i-w:i])/(4*np.log(2)))
    return o
def _gk(op,h,l,c,w):
    n=len(c);o=np.full(n,np.nan)
    u=np.log(np.maximum(h,1e-12)/np.maximum(op,1e-12))
    d=np.log(np.maximum(l,1e-12)/np.maximum(op,1e-12))
    cc=np.log(np.maximum(c,1e-12)/np.maximum(op,1e-12))
    gk=0.5*u*u+0.5*d*d-(2*np.log(2)-1)*cc*cc
    for i in range(w,n): o[i]=sqrt(max(np.mean(gk[i-w:i]),1e-20))
    return o

def compute_vols(c,h,l,op,lr,atr):
    v={}
    for w in [7,14,20,30,50]: v[f"atr{w}"]=_ratr(atr,w)
    for w in [10,20,30,50,100]: v[f"std{w}"]=_rstd(lr,w)
    for lam in [0.90,0.94,0.97]: v[f"ewm{lam}"]=_ewma(lr,lam)
    v["park20"]=_park(h,l,20); v["gk20"]=_gk(op,h,l,c,20)
    r5=_rstd(lr,5);r22=_rstd(lr,22);r66=_rstd(lr,66)
    v["har"]=0.4*r5+0.35*r22+0.25*r66
    s20=_rstd(lr,20);s5=_rstd(lr,5)
    v["vmom"]=np.where((s20>1e-12)&~np.isnan(s20)&~np.isnan(s5),s5/s20-1,np.nan)
    return v


# ═══════════════════════════════════════════════════════════════
# B/C/D SEARCH
# ═══════════════════════════════════════════════════════════════

def search_bcd(pos_a, log_ret, metric_fn, vols, a_exsh):
    """Find best B, C, D modifications of pos_a."""
    vol_keys = list(vols.keys())

    # B: inverse vol scaling of position
    best_b = a_exsh
    for _ in range(N_BCD):
        vk=random.choice(vol_keys); vf=vols[vk]
        vv=vf[~np.isnan(vf)]
        if len(vv)<50: continue
        ref=np.percentile(vv,random.choice([50,75]))
        if ref<1e-12: continue
        alpha=random.choice([0.1,0.2,0.3,0.5,0.7,1.0,1.3,1.5,2.0])
        ratio=(ref/np.where((vf<1e-12)|np.isnan(vf),ref,vf))**alpha
        ratio=np.clip(ratio,random.choice([0.1,0.2,0.3]),random.choice([1.5,2.0,3.0]))
        pos_b=pos_a*ratio
        sh=metric_fn(pos_b)
        if sh>best_b: best_b=sh

    # C: target vol scaling
    best_c = a_exsh
    for _ in range(N_BCD):
        vk=random.choice(vol_keys); vf=vols[vk]
        vv=vf[~np.isnan(vf)]
        if len(vv)<50: continue
        tgt_pct=random.choice([25,50,75])
        tgt=np.percentile(vv,tgt_pct)
        if tgt<1e-12: continue
        power=random.choice([0.3,0.5,0.7,1.0,1.3,1.5,2.0])
        scale=(tgt/np.where((vf<1e-12)|np.isnan(vf),tgt,vf))**power
        scale=np.clip(scale,random.choice([0.05,0.1,0.2]),random.choice([1.5,2.0,3.0]))
        pos_c=pos_a*scale
        sh=metric_fn(pos_c)
        if sh>best_c: best_c=sh

    # D: vol targeting + skip
    best_d = a_exsh
    for _ in range(N_BCD):
        vk=random.choice(vol_keys); vf=vols[vk]
        vv=vf[~np.isnan(vf)]
        if len(vv)<50: continue
        bpy_eff=BPY if len(pos_a)<50000 else BPH  # heuristic
        ann_vf=vf*sqrt(bpy_eff)
        vt=random.choice([0.05,0.08,0.10,0.15,0.20,0.25,0.30])
        scale=vt/np.where((ann_vf<1e-6)|np.isnan(ann_vf),vt,ann_vf)
        scale=np.clip(scale,random.choice([0.05,0.1,0.2]),random.choice([1.5,2.0,3.0]))
        skip=random.choice([0,70,80,85,90,95])
        if skip>0:
            thresh=np.percentile(vv,skip)
            scale=np.where((~np.isnan(vf))&(vf>thresh),0.0,scale)
        pos_d=pos_a*scale
        sh=metric_fn(pos_d)
        if sh>best_d: best_d=sh

    return round(best_b,3), round(best_c,3), round(best_d,3)


# ═══════════════════════════════════════════════════════════════
# RECONSTRUCT A POSITION
# ═══════════════════════════════════════════════════════════════

def parse_params(sp_str, fp_str):
    sig_params={}
    try:
        sig_params=ast.literal_eval(sp_str) if str(sp_str).strip().startswith("{") else {}
    except: pass
    if not sig_params:
        for part in str(sp_str).split():
            if "=" in part:
                k,v=part.split("=",1)
                try: v=float(v)
                except: pass
                if k in ("w","vw","ap","el"):
                    try: v=int(float(v))
                    except: pass
                sig_params[k]=v

    filt_params={}
    try:
        filt_params=ast.literal_eval(fp_str) if str(fp_str).strip().startswith("{") else {}
    except: pass
    if not filt_params:
        for part in str(fp_str).split():
            if "=" in part:
                k,v=part.split("=",1)
                try: v=float(v)
                except: pass
                if k in ("adx_th","atr_lo","mom_n","dow"):
                    try: v=int(float(v))
                    except: pass
                filt_params[k]=v
    return sig_params, filt_params

def get_rm(row):
    return {k: float(row.get(f"rm_{k}",0) or 0) for k in
            ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]} | \
           {"mh": int(float(row.get("rm_mh",0) or 0))}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0=time.time()
    print("="*120)
    print("B/C/D FINAL — Fixed A, same metrics, 25 vol × 5000 per approach")
    print("="*120)

    daily_df,hourly_df,_=load_data()
    kr=pd.read_parquet(KR_PATH);kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS=["S1","S2","S3","S4","S5","S6"]

    # ══════════════════════════════════════════════════════════
    # DAILY
    # ══════════════════════════════════════════════════════════
    d_ub=pd.read_csv(BASE/"results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv")
    print(f"\n── DAILY ({len(d_ub)} configs) ──")

    d_results={}
    for _,row in d_ub.iterrows():
        sname=row["strategy"];ticker=row["ticker"];sid=SID_MAP.get(sname)
        if not sid: continue
        a_val=A_DAILY[sid]

        tdf=daily_df[daily_df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
        if len(tdf)<WARMUP+200: continue
        close=tdf["close"].values.astype(np.float64)
        high=tdf["high"].values.astype(np.float64)
        low=tdf["low"].values.astype(np.float64)
        open_arr=tdf["open"].values.astype(np.float64)
        volume=tdf["volume"].values.astype(np.float64)
        dates=pd.to_datetime(tdf["date"].values);n=len(close)
        lr=np.zeros(n);lr[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr=calc_atr(high,low,close,14)
        cal_days=np.ones(n)
        for i in range(1,n): cal_days[i]=(dates[i]-dates[i-1]).days
        rf_raw=kr_s.reindex(dates,method="ffill").values
        rf_d=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        ind=compute_base(close,high,low,open_arr,volume,is_hourly=False)
        rh=pd.Series(high).rolling(11,min_periods=1).max().values
        rl=pd.Series(low).rolling(11,min_periods=1).min().values
        oos_start=_year_bounds(dates,2022)[0]
        if oos_start<=WARMUP: continue
        oos_mask=np.zeros(n,dtype=bool);oos_mask[oos_start:]=True

        # Reconstruct A signal+position
        sp,fp=parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig=None
        if sid in ("S1","S2"):
            mt=sp.get("ma_type","sma");w=sp.get("w",20)
            try:
                ma=get_ma(close,w,mt);std=calc_std(close,w)
                if sid=="S1": sp["atr_arr"]=atr; sig=gen_sig_s1_ext(close,std,ma,sp)
                else: sig=gen_sig_s2_ext(close,std,ma,sp)
            except: pass
        elif sid=="S3":
            w=sp.get("w",20);sig=gen_sig_s3(close,high,low,precompute_donchian_cache(high,low,[w]),sp)
        elif sid=="S4":
            ap=sp.get("ap",14);mult=sp.get("mult",3.0)
            sig=gen_sig_s4(close,high,low,precompute_supertrend_cache(high,low,close,[ap],[mult]),sp)
        elif sid=="S5":
            pt=sp.get("pt","fibonacci");sig=gen_sig_s5(close,high,low,{pt:calc_pivot_daily(high,low,close,pt)},sp)
        elif sid=="S6":
            vw=sp.get("vw",20);sig=gen_sig_s6(close,high,low,volume,precompute_vwap_cache(close,high,low,volume,[vw]),sp)
        if sig is None:
            s20=calc_sma(close,20);st20=calc_std(close,20);z=(close-s20)/np.where(st20>1e-12,st20,1.0)
            sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1;sig[:WARMUP]=0
        rsi=calc_rsi(close,14);dow=dates.dayofweek.values.astype(np.int32)
        sig=apply_filters_generic(sig,close,volume,ind,rsi,atr,dow,fp,IS_TREND[sid])
        rm=get_rm(row)
        pos_a=run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP,n)

        # Metric closure (same formula as A)
        def d_metric(pos):
            return daily_metric(pos,lr,dates,rf_d,cal_days,oos_mask,COMM_D)

        vols=compute_vols(close,high,low,open_arr,lr,atr)
        bb,cc,dd=search_bcd(pos_a,lr,d_metric,vols,d_metric(pos_a))
        avg=(bb+cc+dd)/3
        d_results[(sid,ticker)]={"B":bb,"C":cc,"D":dd,"avg":round(avg,3)}
        print(f"  D {sname:>14s} {ticker:>5s}: B={bb:+.2f} C={cc:+.2f} D={dd:+.2f} avg={avg:+.2f}")

    # Daily summary
    print(f"\n{'='*100}")
    print("DAILY SUMMARY (A fixed)")
    print(f"{'='*100}")
    print(f"  {'Strategy':>16s} {'A(fix)':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'Avg':>7s} "
          f"{'B>A':>4s} {'C>A':>4s} {'D>A':>4s}")
    for sid in SIDS:
        sname=SNAMES[sid]; a=A_DAILY[sid]
        bv=[];cv=[];dv=[]
        for t in TICKERS:
            r=d_results.get((sid,t))
            if r: bv.append(r["B"]);cv.append(r["C"]);dv.append(r["D"])
        if not bv: continue
        mb=np.mean(bv);mc=np.mean(cv);md=np.mean(dv);mavg=(mb+mc+md)/3
        print(f"  {sname:>16s} {a:+7.2f} {mb:+7.2f} {mc:+7.2f} {md:+7.2f} {mavg:+7.2f} "
              f"{'Y' if mb>a else 'N':>4s} {'Y' if mc>a else 'N':>4s} {'Y' if md>a else 'N':>4s}")

    # ══════════════════════════════════════════════════════════
    # HOURLY
    # ══════════════════════════════════════════════════════════
    h_ub=pd.read_csv(BASE/"results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")
    print(f"\n── HOURLY ({len(h_ub)} configs) ──")

    h_results={}
    for _,row in h_ub.iterrows():
        sname=row["strategy"];ticker=row["ticker"];sid=SID_MAP.get(sname)
        if not sid: continue

        hdf=hourly_df[hourly_df["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
        if len(hdf)<WARMUP_HOURLY+500: continue
        close=hdf["close"].values.astype(np.float64)
        high=hdf["high"].values.astype(np.float64)
        low=hdf["low"].values.astype(np.float64)
        open_arr=hdf["open"].values.astype(np.float64)
        volume=hdf["volume"].values.astype(np.float64)
        h_dts=pd.to_datetime(hdf["datetime"].values);n=len(close)
        lr=np.zeros(n);lr[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr=calc_atr(high,low,close,14)
        ind=compute_base(close,high,low,open_arr,volume,is_hourly=True)
        rh=pd.Series(high).rolling(99,min_periods=1).max().values
        rl=pd.Series(low).rolling(99,min_periods=1).min().values
        oos_idx=0
        for i in range(n):
            if h_dts[i].year>=2022: oos_idx=i;break
        if oos_idx<=WARMUP_HOURLY: continue
        oos_mask=np.zeros(n,dtype=bool);oos_mask[oos_idx:]=True

        # Reconstruct A signal+position
        sp,fp=parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig=None
        if sid in ("S1","S2"):
            mt=sp.get("ma_type","sma");w=sp.get("w",90)
            try:
                ma=h_get_ma(close,w,mt);std=calc_std(close,w)
                mc={(mt,w):(ma,std)}
                if sid=="S1": sig=h_gen_s1(close,mc,atr,sp)
                else: sig=h_gen_s2(close,mc,sp)
            except: pass
        elif sid=="S3":
            w=sp.get("w",90);sig=h_gen_s3(close,high,low,precompute_donchian_cache(high,low,[w]),sp)
        elif sid=="S4":
            ap=sp.get("ap",90);mult=sp.get("mult",3.0)
            sig=h_gen_s4(close,high,low,precompute_supertrend_cache(high,low,close,[ap],[mult]),sp)
        elif sid=="S5":
            pt=sp.get("pt","fibonacci")
            sig=h_gen_s5(close,{pt:compute_daily_pivots_for_hourly(daily_df,ticker,h_dts,pt)},sp)
        elif sid=="S6":
            vw=sp.get("vw",90);sig=h_gen_s6(close,high,low,volume,precompute_vwap_cache(close,high,low,volume,[vw]),sp)
        if sig is None:
            s90=calc_sma(close,90);st90=calc_std(close,90);z=(close-s90)/np.where(st90>1e-12,st90,1.0)
            sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1;sig[:WARMUP_HOURLY]=0
        rsi=calc_rsi(close,14)
        sig=h_apply_filt(sig,close,volume,ind,rsi,atr,fp,IS_TREND[sid])
        rm={"sl_p":float(row.get("rm_sl_p",0) or 0),"tp_p":float(row.get("rm_tp_p",0) or 0),
            "sl_a":float(row.get("rm_sl_a",0) or 0),"tp_a":float(row.get("rm_tp_a",0) or 0),
            "trail":float(row.get("rm_trail",0) or 0),"mh":int(float(row.get("rm_mh",0) or 0)),
            "be":float(row.get("rm_be",0) or 0),"chand":float(row.get("rm_chand",0) or 0),
            "rr":float(row.get("rm_rr",0) or 0)}
        pos_a=h_run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP_HOURLY,n)

        # Compute calendar factor for FDR
        h_dates_norm=h_dts.normalize()
        h_cal_factor=np.zeros(n)
        rf_raw=kr_s.reindex(h_dates_norm,method="ffill").values
        rf_vals=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        prev_d=None
        for i in range(n):
            d=h_dates_norm[i]
            if d!=prev_d:
                cd=1
                if prev_d is not None: cd=(d-prev_d).days
                h_cal_factor[i]=rf_vals[i]*cd
                prev_d=d

        # Metric closure WITH FDR (same as WF that gave A=1.65-3.24)
        def h_metric(pos):
            return hourly_metric(pos,lr,h_cal_factor,oos_mask,COMM_H)

        vols=compute_vols(close,high,low,open_arr,lr,atr)
        bb,cc,dd=search_bcd(pos_a,lr,h_metric,vols,h_metric(pos_a))
        avg=(bb+cc+dd)/3
        h_results[(sid,ticker)]={"B":bb,"C":cc,"D":dd,"avg":round(avg,3)}
        print(f"  H {sname:>14s} {ticker:>5s}: B={bb:+.2f} C={cc:+.2f} D={dd:+.2f} avg={avg:+.2f}")

    # Hourly summary
    print(f"\n{'='*100}")
    print("HOURLY SUMMARY (A fixed)")
    print(f"{'='*100}")
    print(f"  {'Strategy':>16s} {'A(fix)':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'Avg':>7s} "
          f"{'B>A':>4s} {'C>A':>4s} {'D>A':>4s}")
    for sid in SIDS:
        sname=SNAMES[sid]; a=A_HOURLY[sid]
        bv=[];cv=[];dv=[]
        for t in TICKERS:
            r=h_results.get((sid,t))
            if r: bv.append(r["B"]);cv.append(r["C"]);dv.append(r["D"])
        if not bv: continue
        mb=np.mean(bv);mc=np.mean(cv);md=np.mean(dv);mavg=(mb+mc+md)/3
        print(f"  {sname:>16s} {a:+7.2f} {mb:+7.2f} {mc:+7.2f} {md:+7.2f} {mavg:+7.2f} "
              f"{'Y' if mb>a else 'N':>4s} {'Y' if mc>a else 'N':>4s} {'Y' if md>a else 'N':>4s}")

    # Final combined
    print(f"\n{'='*120}")
    print("FINAL TABLE")
    print(f"{'='*120}")
    print(f"  {'Strategy':>16s} {'D:A':>6s} {'D:B':>6s} {'D:C':>6s} {'D:D':>6s} {'D:Avg':>6s} "
          f"{'H:A':>6s} {'H:B':>6s} {'H:C':>6s} {'H:D':>6s} {'H:Avg':>6s}")
    for sid in SIDS:
        sname=SNAMES[sid]
        da=A_DAILY[sid]
        db=np.mean([d_results.get((sid,t),{}).get("B",da) for t in TICKERS])
        dc=np.mean([d_results.get((sid,t),{}).get("C",da) for t in TICKERS])
        dd=np.mean([d_results.get((sid,t),{}).get("D",da) for t in TICKERS])
        davg=(db+dc+dd)/3
        ha=A_HOURLY[sid]
        hb=np.mean([h_results.get((sid,t),{}).get("B",ha) for t in TICKERS])
        hc=np.mean([h_results.get((sid,t),{}).get("C",ha) for t in TICKERS])
        hd=np.mean([h_results.get((sid,t),{}).get("D",ha) for t in TICKERS])
        havg=(hb+hc+hd)/3
        print(f"  {sname:>16s} {da:+6.2f} {db:+6.2f} {dc:+6.2f} {dd:+6.2f} {davg:+6.2f} "
              f"{ha:+6.2f} {hb:+6.2f} {hc:+6.2f} {hd:+6.2f} {havg:+6.2f}")

    print(f"\nDONE in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")

if __name__=="__main__":
    main()
