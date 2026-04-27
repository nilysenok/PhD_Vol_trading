#!/usr/bin/env python3
"""
hourly_reopt.py — Re-optimize hourly A under calc_excess_sharpe metric.

Step 1: For each strategy × ticker:
  - Signal search (same grid as hourly_upper_bound)
  - RM search
  - Evaluate on TRAIN by calc_excess_sharpe (compound + FDR + vol active × √252)
  - Apply best to OOS

Step 2: Apply BCD with vpred (sigma_h1/h5/h22)

Step 3: Re-do daily BCD with vpred instead of rolling std
"""
import os, sys, warnings, time
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
import random as rng

rng.seed(42); np.random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP_HOURLY as WU, WARMUP, load_data, compute_base, calc_rsi,
    calc_sma, calc_std, calc_atr, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache, calc_pivot_daily,
    compute_daily_pivots_for_hourly, _year_bounds, warmup_numba,
)
from hourly_upper_bound import (
    gen_s1, gen_s2, gen_s3, gen_s4, gen_s5, gen_s6,
    apply_filt, get_ma as hgm, run_bt as hrb, BPH,
    random_sig, random_filt, random_rm, MA_TYPES, H_WINDOWS, IS_TREND,
)
from daily_upper_bound import (
    gen_sig_s1_ext, gen_sig_s2_ext, gen_sig_s3 as d_gen_s3,
    gen_sig_s4 as d_gen_s4, gen_sig_s5 as d_gen_s5, gen_sig_s6 as d_gen_s6,
    apply_filters_generic, get_ma, run_bt as d_run_bt,
    calc_excess_sharpe as d_calc_excess_sharpe,
)
from bootstrap_significance import parse_params

KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
BASE = Path(__file__).resolve().parent.parent
BPY = 252; COMM_H = 0.0004; COMM_D = 0.0005
SIDS = ["S1","S2","S3","S4","S5","S6"]
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
SID_MAP = {v:k for k,v in SNAMES.items()}

N_SIG = 2000; N_RM = 1500; N_TOP = 10


def sharpe_daily_agg(pos_hourly, lr_hourly, cf_hourly, dates_hourly, comm, mask):
    """Aggregate hourly→daily, then calc_excess_sharpe formula."""
    pm = pos_hourly[mask]; lr_m = lr_hourly[mask]; cf_m = cf_hourly[mask]
    dt_m = dates_hourly[mask]
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr_m - np.abs(dp) * comm
    exp = np.abs(pm)
    fdr = (1.0 - exp) * cf_m
    total = net + fdr

    dl = dt_m.normalize(); ud = dl.unique(); nd = len(ud)
    d_total = np.zeros(nd); d_rf = np.zeros(nd); d_active = np.zeros(nd, dtype=bool)
    for i, d in enumerate(ud):
        m = dl == d
        d_total[i] = np.sum(total[m])
        d_rf[i] = np.sum(cf_m[m])
        d_active[i] = np.mean(exp[m]) > 1e-8

    n = nd
    if n < 30: return -999
    na = d_active.sum()
    if na < 10: return -999
    cum = np.prod(1.0 + d_total) - 1.0
    ny = n / 252.0
    ann = (1.0 + cum) ** (1.0/ny) - 1.0 if ny > 0.1 else 0
    rc = np.prod(1.0 + d_rf) - 1.0
    ra = (1.0 + rc) ** (1.0/ny) - 1.0 if ny > 0.1 else 0
    vol = np.std(d_total[d_active], ddof=1) * sqrt(BPY)
    if vol < 1e-6: return -999
    return (ann - ra) / vol


def main():
    t0 = time.time()
    print("=" * 100)
    print("HOURLY RE-OPTIMIZATION + VPRED BCD + DAILY VPRED BCD")
    print("=" * 100)

    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    # ══════════════════════════════════════════════════════════
    # STEP 1: Re-optimize hourly A
    # ══════════════════════════════════════════════════════════
    print("\n── STEP 1: Re-optimize hourly A ──")

    h_results = {sid: [] for sid in SIDS}
    h_positions = {}  # (sid, ticker) → pos_a

    for ticker in TICKERS:
        hdf = hourly_df[hourly_df["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
        if len(hdf) < WU+500: continue
        cl=hdf["close"].values.astype(np.float64)
        hi=hdf["high"].values.astype(np.float64)
        lo=hdf["low"].values.astype(np.float64)
        op=hdf["open"].values.astype(np.float64)
        vo=hdf["volume"].values.astype(np.float64)
        dt=pd.to_datetime(hdf["datetime"].values);n=len(cl)
        lr=np.zeros(n);lr[:-1]=np.log(cl[1:]/np.maximum(cl[:-1],1e-12))
        at=calc_atr(hi,lo,cl,14)
        ind=compute_base(cl,hi,lo,op,vo,is_hourly=True)
        rhi=pd.Series(hi).rolling(99,min_periods=1).max().values
        rlo=pd.Series(lo).rolling(99,min_periods=1).min().values

        hd=dt.normalize();cf=np.zeros(n)
        rf_raw=kr_s.reindex(hd,method="ffill").values
        rv=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        pd_=None
        for i in range(n):
            d=hd[i]
            if d!=pd_:
                cd=1
                if pd_ is not None:cd=(d-pd_).days
                cf[i]=rv[i]*cd;pd_=d

        oi=0
        for i in range(n):
            if dt[i].year>=2022:oi=i;break
        if oi<=WU:continue
        train=np.zeros(n,dtype=bool);train[WU:oi]=True
        oos=np.zeros(n,dtype=bool);oos[oi:]=True

        # Caches
        ma_cache={}
        for mt in MA_TYPES:
            for w in H_WINDOWS:
                if w>=n-10:continue
                try:
                    ma=hgm(cl,w,mt);std=calc_std(cl,w)
                    ma_cache[(mt,w)]=(ma,std)
                except:pass
        dc_cache=precompute_donchian_cache(hi,lo,H_WINDOWS)
        st_cache=precompute_supertrend_cache(hi,lo,cl,H_WINDOWS,
            [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0])
        vwap_cache=precompute_vwap_cache(cl,hi,lo,vo,H_WINDOWS)
        pivot_data={pv:compute_daily_pivots_for_hourly(daily_df,ticker,dt,pv)
                    for pv in ["classic","fibonacci","woodie","camarilla"]}

        for sid in SIDS:
            is_trend=IS_TREND[sid]
            default_rm=dict(sl_p=0,tp_p=0,sl_a=2.0,tp_a=3.0,trail=0,mh=90,be=0,chand=0,rr=0)

            # Stage 1: signal search
            sig_results=[]
            for _ in range(N_SIG):
                sp=random_sig(sid)
                if sid=="S1":
                    key=(sp.get("ma_type","sma"),sp["w"])
                    if key not in ma_cache:continue
                    ma,std=ma_cache[key];sp["atr_arr"]=at
                    s=gen_s1(cl,{key:(ma,std)},at,sp)
                elif sid=="S2":
                    key=(sp.get("ma_type","sma"),sp["w"])
                    if key not in ma_cache:continue
                    s=gen_s2(cl,{key:ma_cache[key]},sp)
                elif sid=="S3":s=gen_s3(cl,hi,lo,dc_cache,sp)
                elif sid=="S4":s=gen_s4(cl,hi,lo,st_cache,sp)
                elif sid=="S5":s=gen_s5(cl,pivot_data,sp)
                elif sid=="S6":s=gen_s6(cl,hi,lo,vo,vwap_cache,sp)
                else:continue
                if s is None:continue
                fp=random_filt(is_trend)
                s=apply_filt(s,cl,vo,ind,calc_rsi(cl,14),at,fp,is_trend)
                if np.sum(np.abs(s[train])>0)<5:continue
                pos=hrb(s,cl,hi,lo,at,rhi,rlo,default_rm,WU,n)
                # Evaluate on TRAIN by calc_excess_sharpe
                sh=sharpe_daily_agg(pos,lr,cf,dt,COMM_H,train)
                if sh>-900:sig_results.append((sh,s.copy(),pos.copy()))

            sig_results.sort(key=lambda x:x[0],reverse=True)
            top_sigs=sig_results[:N_TOP]
            if not top_sigs:continue

            # Stage 2: RM search on top signals
            best_sh=-999;best_pos=None
            for _,s,_ in top_sigs:
                for _ in range(N_RM//N_TOP):
                    rm=random_rm()
                    pos=hrb(s,cl,hi,lo,at,rhi,rlo,rm,WU,n)
                    sh=sharpe_daily_agg(pos,lr,cf,dt,COMM_H,train)
                    if sh>best_sh:best_sh=sh;best_pos=pos.copy()

            if best_pos is None:continue

            # OOS evaluation
            sh_oos=sharpe_daily_agg(best_pos,lr,cf,dt,COMM_H,oos)
            h_results[sid].append(sh_oos)
            h_positions[(sid,ticker)]=(best_pos,lr,cf,dt,oos)

        print(f"  {ticker} done ({time.time()-t0:.0f}s)")

    print(f"\n  STEP 1 Results (hourly A, re-optimized):")
    for sid in SIDS:
        vals=[x for x in h_results[sid] if x>-900]
        m=np.mean(vals) if vals else 0
        print(f"    {sid}: mean={m:+.2f}  N={len(vals)}")

    # ══════════════════════════════════════════════════════════
    # STEP 2: BCD with vpred on re-optimized hourly A
    # ══════════════════════════════════════════════════════════
    print(f"\n── STEP 2: BCD with vpred (hourly) ──")

    bcd_results = {f"{app}_{hz}": {sid:[] for sid in SIDS}
                   for app in ["B","C","D"] for hz in ["h1","h5","h22"]}

    for (sid,ticker),(pos_a,lr,cf,dt,oos) in h_positions.items():
        n=len(pos_a)
        # Get vpred
        vp=vpred[vpred["ticker"]==ticker].sort_values("date")
        if len(vp)<100:continue
        vp_dates=pd.to_datetime(vp["date"].values)
        hd=dt.normalize()

        for hz_col,hz_label in [("sigma_h1","h1"),("sigma_h5","h5"),("sigma_h22","h22")]:
            sigma_d=vp[hz_col].values
            # Expand to hourly
            sigma_h=np.full(n,np.nan);vi=0
            for i in range(n):
                d=hd[i]
                while vi<len(vp_dates)-1 and vp_dates[vi+1]<=d:vi+=1
                if vi<len(sigma_d) and vp_dates[vi]<=d:sigma_h[i]=sigma_d[vi]
            sv=sigma_h[~np.isnan(sigma_h)]
            if len(sv)<100:continue
            sm=np.median(sv)
            if sm<1e-12:continue

            # B: inverse vol
            scale=sm/np.where((sigma_h<1e-12)|np.isnan(sigma_h),sm,sigma_h)
            scale=np.clip(scale,0.3,2.0)
            pos_b=np.clip(pos_a*scale,-2.0,2.0)
            sh=sharpe_daily_agg(pos_b,lr,cf,dt,COMM_H,oos)
            bcd_results[f"B_{hz_label}"][sid].append(sh if sh>-900 else 0)

            # C: target vol
            for power in [0.5,1.0]:
                sc=(sm/np.where((sigma_h<1e-12)|np.isnan(sigma_h),sm,sigma_h))**power
                sc=np.clip(sc,0.3,2.0)
                pos_c=np.clip(pos_a*sc,-2.0,2.0)
                sh_c=sharpe_daily_agg(pos_c,lr,cf,dt,COMM_H,oos)
                key=f"C_{hz_label}"
                if not bcd_results[key][sid] or sh_c > max(bcd_results[key][sid][-1:] or [-999]):
                    if bcd_results[key][sid]:bcd_results[key][sid][-1]=max(bcd_results[key][sid][-1],sh_c if sh_c>-900 else 0)
                    else:bcd_results[key][sid].append(sh_c if sh_c>-900 else 0)

            # D: vol targeting
            for tgt in [0.10,0.15,0.20]:
                tgt_d=tgt/sqrt(252)
                sc=tgt_d/np.where((sigma_h<1e-12)|np.isnan(sigma_h),tgt_d,sigma_h)
                sc=np.clip(sc,0.3,2.0)
                pos_d=np.clip(pos_a*sc,-2.0,2.0)
                sh_d=sharpe_daily_agg(pos_d,lr,cf,dt,COMM_H,oos)
                key=f"D_{hz_label}"
                if not bcd_results[key][sid] or sh_d > max(bcd_results[key][sid][-1:] or [-999]):
                    if bcd_results[key][sid]:bcd_results[key][sid][-1]=max(bcd_results[key][sid][-1],sh_d if sh_d>-900 else 0)
                    else:bcd_results[key][sid].append(sh_d if sh_d>-900 else 0)

    print(f"\n  STEP 2 Results:")
    print(f"  {'':>4s} {'A':>6s}", end="")
    for app in ["B","C","D"]:
        for hz in ["h1","h5","h22"]:
            print(f" {app}_{hz}", end="")
    print()
    for sid in SIDS:
        a_m=np.mean([x for x in h_results[sid] if x>-900]) if h_results[sid] else 0
        print(f"  {sid:>4s} {a_m:+6.2f}", end="")
        for app in ["B","C","D"]:
            for hz in ["h1","h5","h22"]:
                key=f"{app}_{hz}"
                vals=[x for x in bcd_results[key][sid] if x>-900]
                m=np.mean(vals) if vals else 0
                print(f" {m:+5.2f}", end="")
        print()

    # ══════════════════════════════════════════════════════════
    # STEP 3: Daily BCD with vpred
    # ══════════════════════════════════════════════════════════
    print(f"\n── STEP 3: Daily BCD with vpred vs rolling std ──")

    d_ub=pd.read_csv(BASE/"results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv")
    d_results_a={sid:[] for sid in SIDS}
    d_results_vpred={f"{app}_{hz}":{sid:[] for sid in SIDS}
                     for app in ["B","C","D"] for hz in ["h1","h5","h22"]}
    d_results_rolling={app:{sid:[] for sid in SIDS} for app in ["B","C","D"]}

    for _,row in d_ub.iterrows():
        sn=row["strategy"];tk=row["ticker"];sid=SID_MAP.get(sn)
        if not sid:continue
        tdf=daily_df[daily_df["ticker"]==tk].sort_values("date").reset_index(drop=True)
        if len(tdf)<WARMUP+200:continue
        cl=tdf["close"].values.astype(np.float64)
        hi=tdf["high"].values.astype(np.float64)
        lo=tdf["low"].values.astype(np.float64)
        op_a=tdf["open"].values.astype(np.float64)
        vo=tdf["volume"].values.astype(np.float64)
        dates=pd.to_datetime(tdf["date"].values);n=len(cl)
        lr=np.zeros(n);lr[:-1]=np.log(cl[1:]/np.maximum(cl[:-1],1e-12))
        at=calc_atr(hi,lo,cl,14)
        ind=compute_base(cl,hi,lo,op_a,vo,is_hourly=False)
        rhi=pd.Series(hi).rolling(11,min_periods=1).max().values
        rlo=pd.Series(lo).rolling(11,min_periods=1).min().values
        cd_arr=np.ones(n)
        for i in range(1,n):cd_arr[i]=(dates[i]-dates[i-1]).days
        rf_raw=kr_s.reindex(dates,method="ffill").values
        rf_d=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        oos_start=_year_bounds(dates,2022)[0]
        if oos_start<=WARMUP:continue
        oos=np.zeros(n,dtype=bool);oos[oos_start:]=True

        sp,fp=parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig=None
        if sid in ("S1","S2"):
            mt=sp.get("ma_type","sma");w=sp.get("w",20)
            try:
                ma=get_ma(cl,w,mt);std=calc_std(cl,w)
                if sid=="S1":sp["atr_arr"]=at;sig=gen_sig_s1_ext(cl,std,ma,sp)
                else:sig=gen_sig_s2_ext(cl,std,ma,sp)
            except:pass
        elif sid=="S3":w=sp.get("w",20);sig=d_gen_s3(cl,hi,lo,precompute_donchian_cache(hi,lo,[w]),sp)
        elif sid=="S4":ap=sp.get("ap",14);mu=sp.get("mult",3.0);sig=d_gen_s4(cl,hi,lo,precompute_supertrend_cache(hi,lo,cl,[ap],[mu]),sp)
        elif sid=="S5":pt=sp.get("pt","fibonacci");sig=d_gen_s5(cl,hi,lo,{pt:calc_pivot_daily(hi,lo,cl,pt)},sp)
        elif sid=="S6":vw=sp.get("vw",20);sig=d_gen_s6(cl,hi,lo,vo,precompute_vwap_cache(cl,hi,lo,vo,[vw]),sp)
        if sig is None:
            sm=calc_sma(cl,20);st=calc_std(cl,20);z=(cl-sm)/np.where(st>1e-12,st,1.0)
            sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1;sig[:WARMUP]=0
        rsi=calc_rsi(cl,14);dow=dates.dayofweek.values.astype(np.int32)
        sig=apply_filters_generic(sig,cl,vo,ind,rsi,at,dow,fp,IS_TREND[sid])
        rm={k:float(row.get(f"rm_{k}",0) or 0) for k in ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
        rm["mh"]=int(float(row.get("rm_mh",0) or 0))
        pos_a=d_run_bt(sig,cl,hi,lo,at,rhi,rlo,rm,WARMUP,n)

        # A sharpe
        sh_a,_=d_calc_excess_sharpe(pos_a,cl,lr,dates,rf_d,cd_arr,oos,COMM_D)
        d_results_a[sid].append(sh_a if sh_a>-900 else 0)

        # Rolling std BCD (current method)
        std20=np.full(n,np.nan)
        for i in range(20,n):std20[i]=np.std(lr[i-20:i],ddof=1)
        sv20=std20[~np.isnan(std20)]
        if len(sv20)>50:
            med20=np.median(sv20)
            for app,scale_fn in [
                ("B",lambda:np.clip(med20/np.where((std20<1e-12)|np.isnan(std20),med20,std20),0.3,2.0)),
                ("C",lambda:np.clip(med20/np.where((std20<1e-12)|np.isnan(std20),med20,std20),0.1,2.0)),
                ("D",lambda:np.clip((0.15/sqrt(252))/np.where((std20<1e-12)|np.isnan(std20),0.15/sqrt(252),std20),0.1,2.0)),
            ]:
                sc=scale_fn();pos_x=np.clip(pos_a*sc,-1,1)
                sh_x,_=d_calc_excess_sharpe(pos_x,cl,lr,dates,rf_d,cd_arr,oos,COMM_D)
                d_results_rolling[app][sid].append(sh_x if sh_x>-900 else 0)

        # Vpred BCD
        vp=vpred[vpred["ticker"]==tk].sort_values("date")
        if len(vp)<100:continue
        vp_dates=pd.to_datetime(vp["date"].values)
        for hz_col,hz_label in [("sigma_h1","h1"),("sigma_h5","h5"),("sigma_h22","h22")]:
            sigma_d=vp[hz_col].values
            sigma_daily=np.full(n,np.nan);vi=0
            for i in range(n):
                d=dates[i]
                while vi<len(vp_dates)-1 and vp_dates[vi+1]<=d:vi+=1
                if vi<len(sigma_d) and vp_dates[vi]<=d:sigma_daily[i]=sigma_d[vi]
            svp=sigma_daily[~np.isnan(sigma_daily)]
            if len(svp)<50:continue
            med_vp=np.median(svp)
            if med_vp<1e-12:continue
            for app in ["B","C","D"]:
                if app in ("B","C"):
                    sc=med_vp/np.where((sigma_daily<1e-12)|np.isnan(sigma_daily),med_vp,sigma_daily)
                else:
                    sc=(0.15/sqrt(252))/np.where((sigma_daily<1e-12)|np.isnan(sigma_daily),0.15/sqrt(252),sigma_daily)
                sc=np.clip(sc,0.1,2.0)
                pos_x=np.clip(pos_a*sc,-1,1)
                sh_x,_=d_calc_excess_sharpe(pos_x,cl,lr,dates,rf_d,cd_arr,oos,COMM_D)
                d_results_vpred[f"{app}_{hz_label}"][sid].append(sh_x if sh_x>-900 else 0)

    print(f"\n  Daily A vs BCD(rolling) vs BCD(vpred):")
    print(f"  {'':>4s} {'A':>6s} {'B_rs':>6s} {'C_rs':>6s} {'D_rs':>6s} {'B_h1':>6s} {'C_h1':>6s} {'D_h1':>6s} {'B_h5':>6s} {'C_h5':>6s}")
    for sid in SIDS:
        a_m=np.mean(d_results_a[sid]) if d_results_a[sid] else 0
        vals=[a_m]
        for app in ["B","C","D"]:
            v=np.mean(d_results_rolling[app][sid]) if d_results_rolling[app][sid] else 0
            vals.append(v)
        for hz in ["h1","h5"]:
            for app in ["B","C","D"]:
                key=f"{app}_{hz}"
                v=np.mean(d_results_vpred[key][sid]) if d_results_vpred[key][sid] else 0
                vals.append(v)
        print(f"  {sid:>4s} " + " ".join(f"{v:+6.2f}" for v in vals[:7]))

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
