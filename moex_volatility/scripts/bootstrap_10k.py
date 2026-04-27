#!/usr/bin/env python3
"""bootstrap_10k.py — Paired bootstrap C>A with 10,000 resamples. Daily + Hourly."""
import os, sys, warnings, time
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_significance import (
    parse_params, sharpe_from_returns, vol_scale_position, compute_daily_returns,
)
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
    apply_filt as h_apply_filt, get_ma as h_get_ma, run_bt as h_run_bt,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
BPY=252; BPH=2268; COMM_D=0.0005; COMM_H=0.0004
SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
        "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND={"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP={v:k for k,v in SNAMES.items()}

N_BOOT = 10000


def paired_boot(ra, rc, bpy, n_boot=N_BOOT):
    n = min(len(ra), len(rc))
    ra=ra[:n]; rc=rc[:n]
    diff_point = sharpe_from_returns(rc, bpy) - sharpe_from_returns(ra, bpy)
    bd = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        bd[b] = sharpe_from_returns(rc[idx], bpy) - sharpe_from_returns(ra[idx], bpy)
    p = np.mean(bd <= 0)
    ci_lo = np.percentile(bd, 2.5); ci_hi = np.percentile(bd, 97.5)
    return diff_point, ci_lo, ci_hi, p


def main():
    t0 = time.time()
    print("=" * 100)
    print(f"PAIRED BOOTSTRAP C > A — {N_BOOT:,} resamples")
    print("=" * 100)

    daily_df, hourly_df, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS = ["S1","S2","S3","S4","S5","S6"]
    import ast

    # ═══════════════════════════════════════════════════════
    # DAILY
    # ═══════════════════════════════════════════════════════
    d_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv")
    d_rets_a = {s: [] for s in SIDS}
    d_rets_c = {s: [] for s in SIDS}

    for _, row in d_ub.iterrows():
        sname=row["strategy"]; ticker=row["ticker"]; sid=SID_MAP.get(sname)
        if not sid: continue
        tdf=daily_df[daily_df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
        if len(tdf)<WARMUP+200: continue
        close=tdf["close"].values.astype(np.float64)
        high=tdf["high"].values.astype(np.float64)
        low=tdf["low"].values.astype(np.float64)
        open_arr=tdf["open"].values.astype(np.float64)
        volume=tdf["volume"].values.astype(np.float64)
        dates=pd.to_datetime(tdf["date"].values); n=len(close)
        lr=np.zeros(n); lr[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
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
        oos_mask=np.zeros(n,dtype=bool); oos_mask[oos_start:]=True

        sp,fp=parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig=None
        if sid in ("S1","S2"):
            mt=sp.get("ma_type","sma");w=sp.get("w",20)
            try:
                ma=get_ma(close,w,mt);std=calc_std(close,w)
                if sid=="S1": sp["atr_arr"]=atr;sig=gen_sig_s1_ext(close,std,ma,sp)
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
        rm={k:float(row.get(f"rm_{k}",0) or 0) for k in ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
        rm["mh"]=int(float(row.get("rm_mh",0) or 0))
        pos_a=run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP,n)
        pos_c=vol_scale_position(pos_a,lr,20)
        d_rets_a[sid].append(compute_daily_returns(pos_a,lr,rf_d,cal_days,oos_mask,COMM_D))
        d_rets_c[sid].append(compute_daily_returns(pos_c,lr,rf_d,cal_days,oos_mask,COMM_D))

    print(f"\n── DAILY (C > A, paired bootstrap {N_BOOT:,}) ──")
    print(f"  {'Strategy':>16s} {'C-A':>7s} {'95% CI':>18s} {'p-value':>10s}")
    for sid in SIDS:
        if not d_rets_a[sid]: continue
        ra=np.concatenate(d_rets_a[sid]); rc=np.concatenate(d_rets_c[sid])
        diff,lo,hi,p=paired_boot(ra,rc,BPY)
        sig_str="***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        print(f"  {SNAMES[sid]:>16s} {diff:+7.3f} [{lo:+.3f}, {hi:+.3f}] {p:10.4f} {sig_str}")

    # ═══════════════════════════════════════════════════════
    # HOURLY — use saved returns from hourly_full_pipeline
    # Re-reconstruct positions (same as hourly_full_pipeline)
    # ═══════════════════════════════════════════════════════
    h_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")
    h_rets_a = {s: [] for s in SIDS}
    h_rets_c = {s: [] for s in SIDS}

    for _, row in h_ub.iterrows():
        sname=row["strategy"]; ticker=row["ticker"]; sid=SID_MAP.get(sname)
        if not sid: continue
        hdf=hourly_df[hourly_df["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
        if len(hdf)<WARMUP_HOURLY+500: continue
        close=hdf["close"].values.astype(np.float64)
        high=hdf["high"].values.astype(np.float64)
        low=hdf["low"].values.astype(np.float64)
        open_arr=hdf["open"].values.astype(np.float64)
        volume=hdf["volume"].values.astype(np.float64)
        h_dts=pd.to_datetime(hdf["datetime"].values); n=len(close)
        lr=np.zeros(n); lr[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr=calc_atr(high,low,close,14)
        ind=compute_base(close,high,low,open_arr,volume,is_hourly=True)
        rsi=calc_rsi(close,14)
        rh=pd.Series(high).rolling(99,min_periods=1).max().values
        rl=pd.Series(low).rolling(99,min_periods=1).min().values

        # Calendar factor for FDR
        h_dates=h_dts.normalize()
        cal_factor=np.zeros(n)
        rf_raw=kr_s.reindex(h_dates,method="ffill").values
        rf_vals=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        prev_d=None
        for i in range(n):
            d=h_dates[i]
            if d!=prev_d:
                cd=1
                if prev_d is not None: cd=(d-prev_d).days
                cal_factor[i]=rf_vals[i]*cd
                prev_d=d

        oos_idx=0
        for i in range(n):
            if h_dts[i].year>=2022: oos_idx=i; break
        if oos_idx<=WARMUP_HOURLY: continue
        oos_mask=np.zeros(n,dtype=bool); oos_mask[oos_idx:]=True

        sp,fp=parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig=None
        if sid in ("S1","S2"):
            mt=sp.get("ma_type","sma");w=sp.get("w",90)
            try:
                ma=h_get_ma(close,w,mt);std=calc_std(close,w);mc={(mt,w):(ma,std)}
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
        sig=h_apply_filt(sig,close,volume,ind,rsi,atr,fp,IS_TREND[sid])
        rm={k:float(row.get(f"rm_{k}",0) or 0) for k in ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
        rm["mh"]=int(float(row.get("rm_mh",0) or 0))
        pos_a=h_run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP_HOURLY,n)
        pos_c=vol_scale_position(pos_a,lr,90)

        # Total returns with FDR
        pm_a=np.clip(pos_a[oos_mask],-1,1);pm_c=np.clip(pos_c[oos_mask],-1,1)
        lr_m=lr[oos_mask];cf_m=cal_factor[oos_mask]
        dp_a=np.diff(pm_a,prepend=0.0);dp_c=np.diff(pm_c,prepend=0.0)
        rets_a=pm_a*lr_m-np.abs(dp_a)*COMM_H+(1.0-np.abs(pm_a))*cf_m
        rets_c=pm_c*lr_m-np.abs(dp_c)*COMM_H+(1.0-np.abs(pm_c))*cf_m
        h_rets_a[sid].append(rets_a)
        h_rets_c[sid].append(rets_c)

    print(f"\n── HOURLY (C > A, paired bootstrap {N_BOOT:,}) ──")
    print(f"  {'Strategy':>16s} {'C-A':>7s} {'95% CI':>18s} {'p-value':>10s}")
    for sid in SIDS:
        if not h_rets_a[sid]: continue
        ra=np.concatenate(h_rets_a[sid]); rc=np.concatenate(h_rets_c[sid])
        diff,lo,hi,p=paired_boot(ra,rc,BPH)
        sig_str="***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        print(f"  {SNAMES[sid]:>16s} {diff:+7.3f} [{lo:+.3f}, {hi:+.3f}] {p:10.4f} {sig_str}")

    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__=="__main__":
    main()
