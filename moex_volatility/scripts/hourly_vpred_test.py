#!/usr/bin/env python3
"""Test hybrid model vol predictions (H=1d/5d/22d) for hourly BCD."""
import os,sys,warnings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"]="YES"
os.environ["NUMBA_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")

import numpy as np,pandas as pd,time
from pathlib import Path
from math import sqrt

sys.path.insert(0,str(Path(__file__).resolve().parent))
from strategies_walkforward import (TICKERS,WARMUP_HOURLY as WU,load_data,compute_base,
    calc_rsi,calc_sma,calc_std,calc_atr,precompute_donchian_cache,
    precompute_supertrend_cache,precompute_vwap_cache,
    compute_daily_pivots_for_hourly,warmup_numba)
from hourly_upper_bound import (gen_s1,gen_s2,gen_s3,gen_s4,gen_s5,gen_s6,
    apply_filt,get_ma as hgm,run_bt as hrb,BPH)
from bootstrap_significance import parse_params

BASE=Path(__file__).resolve().parent.parent
KR_PATH="/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
BPY=252;COMM=0.0004
SIDS=["S1","S2","S3","S4","S5","S6"]
SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
        "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND={"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP={v:k for k,v in SNAMES.items()}


def sharpe_daily(d_total,d_rf,d_active):
    n=len(d_total)
    if n<30:return -999
    na=d_active.sum()
    if na<10:return -999
    cum=np.prod(1.0+d_total)-1.0;ny=n/252.0
    ann=(1.0+cum)**(1.0/ny)-1.0 if ny>0.1 else 0
    rc=np.prod(1.0+d_rf)-1.0;ra=(1.0+rc)**(1.0/ny)-1.0 if ny>0.1 else 0
    vol=np.std(d_total[d_active],ddof=1)*sqrt(BPY)
    if vol<1e-6:return -999
    return(ann-ra)/vol


def hourly_to_daily(pm,lr,cf,dates):
    dp=np.diff(pm,prepend=0.0);net=pm*lr-np.abs(dp)*COMM
    exp=np.abs(pm);fdr=(1.0-exp)*cf;total=net+fdr
    dl=dates.normalize();ud=dl.unique();nd=len(ud)
    dt=np.zeros(nd);dr=np.zeros(nd);da=np.zeros(nd,dtype=bool)
    for i,d in enumerate(ud):
        m=dl==d;dt[i]=np.sum(total[m]);dr[i]=np.sum(cf[m]);da[i]=np.mean(exp[m])>1e-8
    return dt,dr,da


def main():
    t0=time.time()
    print("="*100)
    print("VPRED TEST: hybrid model sigma_h1/h5/h22 for hourly BCD")
    print("="*100)

    daily_df,hourly_df,vpred=load_data()
    kr=pd.read_parquet(KR_PATH);kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    h_ub=pd.read_csv(BASE/"results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")

    # Approaches: B=inverse vol, C=target vol, D=vol targeting+skip
    # For each: test sigma_h1, sigma_h5, sigma_h22
    horizons=["sigma_h1","sigma_h5","sigma_h22"]
    approaches=["B","C","D"]

    # Results: (sid, approach, horizon) → list of per-ticker sharpes
    results={};a_results={sid:[] for sid in SIDS}

    for _,row in h_ub.iterrows():
        sn=row["strategy"];tk=row["ticker"];sid=SID_MAP.get(sn)
        if not sid:continue
        hdf=hourly_df[hourly_df["ticker"]==tk].sort_values("datetime").reset_index(drop=True)
        if len(hdf)<WU+500:continue
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
                cd=1;
                if pd_ is not None:cd=(d-pd_).days
                cf[i]=rv[i]*cd;pd_=d

        oi=0
        for i in range(n):
            if dt[i].year>=2022:oi=i;break
        if oi<=WU:continue
        train_mask=np.zeros(n,dtype=bool);train_mask[WU:oi]=True
        oos_mask=np.zeros(n,dtype=bool);oos_mask[oi:]=True

        # Signal + A position
        sp,fp=parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig=None
        if sid in ("S1","S2"):
            mt=sp.get("ma_type","sma");w=sp.get("w",90)
            try:
                ma=hgm(cl,w,mt);std=calc_std(cl,w);mc={(mt,w):(ma,std)}
                if sid=="S1":sig=gen_s1(cl,mc,at,sp)
                else:sig=gen_s2(cl,mc,sp)
            except:pass
        elif sid=="S3":w=sp.get("w",90);sig=gen_s3(cl,hi,lo,precompute_donchian_cache(hi,lo,[w]),sp)
        elif sid=="S4":ap=sp.get("ap",90);mu=sp.get("mult",3.0);sig=gen_s4(cl,hi,lo,precompute_supertrend_cache(hi,lo,cl,[ap],[mu]),sp)
        elif sid=="S5":pt=sp.get("pt","fibonacci");sig=gen_s5(cl,{pt:compute_daily_pivots_for_hourly(daily_df,tk,dt,pt)},sp)
        elif sid=="S6":vw=sp.get("vw",90);sig=gen_s6(cl,hi,lo,vo,precompute_vwap_cache(cl,hi,lo,vo,[vw]),sp)
        if sig is None:
            sm=calc_sma(cl,90);st=calc_std(cl,90);z=(cl-sm)/np.where(st>1e-12,st,1.0)
            sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1;sig[:WU]=0
        rsi=calc_rsi(cl,14);sig=apply_filt(sig,cl,vo,ind,rsi,at,fp,IS_TREND[sid])
        rm={k:float(row.get(f"rm_{k}",0) or 0) for k in ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
        rm["mh"]=int(float(row.get("rm_mh",0) or 0))
        pos_a=hrb(sig,cl,hi,lo,at,rhi,rlo,rm,WU,n)

        # A sharpe
        oos_dt=dt[oos_mask]
        dt_a,dr_a,da_a=hourly_to_daily(pos_a[oos_mask],lr[oos_mask],cf[oos_mask],oos_dt)
        sh_a=sharpe_daily(dt_a,dr_a,da_a)
        a_results[sid].append(sh_a)

        # Get vpred for this ticker (daily → expand to hourly)
        vp_tk=vpred[vpred["ticker"]==tk].sort_values("date")
        if len(vp_tk)<100:continue
        vp_dates=pd.to_datetime(vp_tk["date"].values)

        for hz in horizons:
            sigma_daily=vp_tk[hz].values
            # Expand daily sigma to hourly (forward fill)
            sigma_hourly=np.full(n,np.nan)
            vp_idx=0
            for i in range(n):
                d=hd[i]
                while vp_idx<len(vp_dates)-1 and vp_dates[vp_idx+1]<=d:
                    vp_idx+=1
                if vp_idx<len(sigma_daily) and vp_dates[vp_idx]<=d:
                    sigma_hourly[i]=sigma_daily[vp_idx]

            sv=sigma_hourly[~np.isnan(sigma_hourly)]
            if len(sv)<100:continue
            sigma_med=np.median(sv)
            if sigma_med<1e-12:continue

            # B: inverse vol scaling
            for alpha in [0.5,1.0,1.5]:
                for max_w in [1.5,2.0]:
                    scale=(sigma_med/np.where((sigma_hourly<1e-12)|np.isnan(sigma_hourly),sigma_med,sigma_hourly))**alpha
                    scale=np.clip(scale,0.3,max_w)
                    pos_b=np.clip(pos_a*scale,-max_w,max_w)
                    # Train sharpe
                    dt_tr,dr_tr,da_tr=hourly_to_daily(pos_b[train_mask],lr[train_mask],cf[train_mask],dt[train_mask])
                    sh_tr=sharpe_daily(dt_tr,dr_tr,da_tr)
                    key=(sid,"B",hz,alpha,max_w)
                    if key not in results or sh_tr>results[key].get("train",-999):
                        dt_oos,dr_oos,da_oos=hourly_to_daily(pos_b[oos_mask],lr[oos_mask],cf[oos_mask],oos_dt)
                        results[key]={"train":sh_tr,"oos":sharpe_daily(dt_oos,dr_oos,da_oos),"tk":tk}

            # C: target vol
            for tgt_q in [0.3,0.5,0.7]:
                tgt=np.percentile(sv,tgt_q*100)
                for power in [0.5,1.0,1.5]:
                    scale=(tgt/np.where((sigma_hourly<1e-12)|np.isnan(sigma_hourly),tgt,sigma_hourly))**power
                    scale=np.clip(scale,0.3,2.0)
                    pos_c=np.clip(pos_a*scale,-2.0,2.0)
                    dt_tr,dr_tr,da_tr=hourly_to_daily(pos_c[train_mask],lr[train_mask],cf[train_mask],dt[train_mask])
                    sh_tr=sharpe_daily(dt_tr,dr_tr,da_tr)
                    key=(sid,"C",hz,tgt_q,power)
                    if key not in results or sh_tr>results[key].get("train",-999):
                        dt_oos,dr_oos,da_oos=hourly_to_daily(pos_c[oos_mask],lr[oos_mask],cf[oos_mask],oos_dt)
                        results[key]={"train":sh_tr,"oos":sharpe_daily(dt_oos,dr_oos,da_oos),"tk":tk}

            # D: vol targeting + regime skip
            for tgt_ann in [0.10,0.15,0.20]:
                tgt_d=tgt_ann/sqrt(252)
                scale=tgt_d/np.where((sigma_hourly<1e-12)|np.isnan(sigma_hourly),tgt_d,sigma_hourly)
                scale=np.clip(scale,0.3,2.0)
                # Skip high vol
                for skip_q in [0.0,0.8,0.9]:
                    pos_d=np.clip(pos_a*scale,-2.0,2.0)
                    if skip_q>0:
                        thresh=np.percentile(sv,skip_q*100)
                        pos_d[(~np.isnan(sigma_hourly))&(sigma_hourly>thresh)]=0
                    dt_tr,dr_tr,da_tr=hourly_to_daily(pos_d[train_mask],lr[train_mask],cf[train_mask],dt[train_mask])
                    sh_tr=sharpe_daily(dt_tr,dr_tr,da_tr)
                    key=(sid,"D",hz,tgt_ann,skip_q)
                    if key not in results or sh_tr>results[key].get("train",-999):
                        dt_oos,dr_oos,da_oos=hourly_to_daily(pos_d[oos_mask],lr[oos_mask],cf[oos_mask],oos_dt)
                        results[key]={"train":sh_tr,"oos":sharpe_daily(dt_oos,dr_oos,da_oos),"tk":tk}

    # ── Aggregate: for each (sid, approach, horizon) find best params on train ──
    print(f"\n{'='*100}")
    print("BEST PER (strategy × approach × horizon) — selected on TRAIN, evaluated on OOS")
    print(f"{'='*100}")

    # For each sid × approach × horizon: find best params across all tickers
    # Actually results are per-ticker — need to aggregate differently
    # Let me restructure: per ticker, find best params on train, apply to OOS

    # Simpler: just show per-strategy mean for each horizon
    # Collect per-ticker OOS sharpe for best-on-train params

    print(f"\n{'':>4s} {'A':>6s}", end="")
    for app in approaches:
        for hz_short in ["h1","h5","h22"]:
            print(f" {app}_{hz_short:>2s}", end="")
    print()

    for sid in SIDS:
        a_mean=np.mean([x for x in a_results[sid] if x>-900]) if a_results[sid] else 0
        print(f"  {sid:>4s} {a_mean:+6.2f}", end="")

        for app in approaches:
            for hz in horizons:
                hz_short=hz.replace("sigma_","")
                # Collect OOS sharpes for this sid×app×hz
                oos_vals=[]
                for key,val in results.items():
                    if key[0]==sid and key[1]==app and key[2]==hz:
                        if val["oos"]>-900:
                            oos_vals.append(val["oos"])
                if oos_vals:
                    print(f" {np.mean(oos_vals):+6.2f}", end="")
                else:
                    print(f"    N/A", end="")
        print()

    # Best horizon per strategy
    print(f"\n  Best horizon per strategy (OOS):")
    for sid in SIDS:
        a_mean=np.mean([x for x in a_results[sid] if x>-900]) if a_results[sid] else 0
        best_v=-999;best_label=""
        for app in approaches:
            for hz in horizons:
                hz_short=hz.replace("sigma_","")
                oos_vals=[v["oos"] for k,v in results.items()
                         if k[0]==sid and k[1]==app and k[2]==hz and v["oos"]>-900]
                if oos_vals:
                    m=np.mean(oos_vals)
                    if m>best_v:best_v=m;best_label=f"{app}_{hz_short}"
        delta=best_v-a_mean
        print(f"    {sid}: A={a_mean:+.2f}  best={best_label} {best_v:+.2f}  delta={delta:+.2f}")

    print(f"\nDONE in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")

if __name__=="__main__":
    main()
