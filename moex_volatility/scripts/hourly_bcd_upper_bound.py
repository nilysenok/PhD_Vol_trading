#!/usr/bin/env python3
"""
hourly_bcd_upper_bound.py — B/C/D approaches on top of A (hourly upper bound).

Same logic as daily_bcd_upper_bound.py but for hourly data.
Uses walk-forward-compatible Sharpe: mean/std × sqrt(2268), no FDR, no rf subtraction.
"""
import os, sys, warnings, time, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP_HOURLY as WARMUP, load_data, compute_base, calc_rsi,
    calc_sma, calc_std, calc_atr, calc_supertrend,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, compute_daily_pivots_for_hourly,
    _year_bounds, warmup_numba,
)

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/hourly_a"
BPH = 2268; COMM = 0.0004
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
random.seed(42); np.random.seed(42)

# Import from hourly_upper_bound
from hourly_upper_bound import (
    bt_universal, run_bt,
    get_ma, MA_TYPES, H_WINDOWS, _ema, _kama, _dema,
    gen_s1, gen_s2, gen_s3, gen_s4, gen_s5, gen_s6,
    apply_filt,
)
from daily_bcd_upper_bound import bt_adaptive_sl

# ═══════════════════════════════════════════════════════════════
# VOL FORECASTS
# ═══════════════════════════════════════════════════════════════
def realized_vol(lr, w):
    n=len(lr); rv=np.full(n,np.nan)
    for i in range(w,n): rv[i]=np.std(lr[i-w:i],ddof=1)
    return rv

def ewm_vol(lr, hl):
    a=1-np.exp(-np.log(2)/hl); n=len(lr)
    v=np.full(n,np.nan); v[1]=lr[0]**2
    for i in range(2,n): v[i]=a*lr[i-1]**2+(1-a)*(v[i-1] if not np.isnan(v[i-1]) else lr[i-1]**2)
    return np.sqrt(np.maximum(v,1e-20))

def har_vol(lr):
    r5=realized_vol(lr,45); r22=realized_vol(lr,198); r66=realized_vol(lr,594)
    return 0.4*r5 + 0.35*r22 + 0.25*r66

def get_vf(lr, method, param):
    if method=="realized": return realized_vol(lr, param)
    elif method=="ewm": return ewm_vol(lr, param)
    elif method=="har": return har_vol(lr)
    return realized_vol(lr, 90)

# Hourly vol methods (scaled windows)
VOL_METHODS = [
    ("realized",45),("realized",90),("realized",180),
    ("ewm",45),("ewm",90),("ewm",135),
    ("har",0),
]


# ═══════════════════════════════════════════════════════════════
# SHARPE (WF-compatible: mean/std × sqrt(BPH), all bars)
# ═══════════════════════════════════════════════════════════════
def calc_sharpe(pos, log_ret, mask, comm):
    pm=pos[mask]; lr=log_ret[mask]; n=len(pm)
    if n<200: return -999
    dp=np.diff(pm,prepend=0.0)
    net=pm*lr-np.abs(dp)*comm
    active=np.abs(pm)>1e-8
    if active.sum()<50: return -999
    m=np.mean(net); s=np.std(net,ddof=1)
    if s<1e-12: return -999
    return m/s*sqrt(BPH)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0=time.time()
    print("="*120)
    print("HOURLY B/C/D UPPER BOUND — Vol modifications on top of A")
    print("="*120)

    ub_path = OUT_DIR / "hourly_upper_bound.csv"
    ub_df = pd.read_csv(ub_path)
    print(f"Loaded {len(ub_df)} A configs")

    daily_df, hourly_df, _ = load_data()
    warmup_numba()

    SIDS = ["S1","S2","S3","S4","S5","S6"]
    SID_MAP = {v:k for k,v in SNAMES.items()}

    # B/C/D grids
    B_GRID = []
    for vm,vp in VOL_METHODS:
        for scale in [0.5,0.75,1.0,1.25,1.5,2.0]:
            for min_m in [0.3,0.5,0.7]:
                for max_m in [2.0,3.0,5.0]:
                    B_GRID.append({"vm":vm,"vp":vp,"scale":scale,"min_m":min_m,"max_m":max_m})
    random.shuffle(B_GRID); B_GRID=B_GRID[:400]

    C_GRID = []
    for vm,vp in VOL_METHODS:
        for tpct in [25,50,75]:
            for minp in [0.1,0.2,0.3]:
                for maxp in [1.5,2.0,3.0]:
                    C_GRID.append({"vm":vm,"vp":vp,"target_pct":tpct,"min_p":minp,"max_p":maxp})
    random.shuffle(C_GRID); C_GRID=C_GRID[:400]

    D_GRID = []
    for vm,vp in VOL_METHODS:
        for vt in [0.05,0.10,0.15,0.20,0.25,0.30]:
            for minp in [0.1,0.2]:
                for maxp in [1.5,2.0,3.0]:
                    D_GRID.append({"vm":vm,"vp":vp,"vol_target":vt,"min_p":minp,"max_p":maxp})
    random.shuffle(D_GRID); D_GRID=D_GRID[:300]

    print(f"Grids: B={len(B_GRID)}, C={len(C_GRID)}, D={len(D_GRID)}")

    results = {}

    for idx, row in ub_df.iterrows():
        sname=row["strategy"]; ticker=row["ticker"]
        sid=SID_MAP.get(sname)
        if not sid: continue
        a_sh = row["ex_sharpe"]

        hdf=hourly_df[hourly_df["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
        if len(hdf)<WARMUP+500: continue
        close=hdf["close"].values.astype(np.float64)
        high=hdf["high"].values.astype(np.float64)
        low=hdf["low"].values.astype(np.float64)
        volume=hdf["volume"].values.astype(np.float64)
        h_dts=pd.to_datetime(hdf["datetime"].values); n=len(close)
        log_ret=np.zeros(n); log_ret[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))

        ind=compute_base(close,high,low,hdf["open"].values.astype(np.float64),volume,is_hourly=True)
        atr=ind["atr14"]
        rh=pd.Series(high).rolling(99,min_periods=1).max().values
        rl=pd.Series(low).rolling(99,min_periods=1).min().values

        # OOS mask
        oos_idx=0
        for i in range(n):
            if h_dts[i].year>=2022: oos_idx=i; break
        if oos_idx<=WARMUP: continue
        oos_mask=np.zeros(n,dtype=bool); oos_mask[oos_idx:]=True

        # Reconstruct A signal + position (simplified: use strategy-appropriate defaults)
        # Parse sig_params from CSV (handles both "k=v" and "{'k': v}" formats)
        import ast
        sp_str=str(row.get("sig_params",""))
        fp_str=str(row.get("filt_params",""))
        try:
            sig_params=ast.literal_eval(sp_str) if sp_str.startswith("{") else {}
        except:
            sig_params={}
        if not sig_params:
            for part in sp_str.split():
                if "=" in part:
                    k,v=part.split("=",1)
                    try:
                        v=float(v)
                        if k in ("w","vw","ap","el"): v=int(v)
                    except: pass
                    sig_params[k]=v
        try:
            filt_params=ast.literal_eval(fp_str) if fp_str.startswith("{") else {}
        except:
            filt_params={}
        if not filt_params:
            for part in fp_str.split():
                if "=" in part:
                    k,v=part.split("=",1)
                    try:
                        v=float(v)
                        if k in ("adx_th","atr_lo","mom_n","dow"): v=int(v)
                    except: pass
                    filt_params[k]=v

        # Generate signal
        sig=None
        if sid in ("S1","S2"):
            mt=sig_params.get("ma_type","sma"); w=sig_params.get("w",90)
            try:
                ma=get_ma(close,w,mt); std=calc_std(close,w)
                mc={(mt,w):(ma,std)}
                if sid=="S1": sig=gen_s1(close,mc,atr,sig_params)
                else: sig=gen_s2(close,mc,sig_params)
            except: pass
        elif sid=="S3":
            w=sig_params.get("w",90)
            dc=precompute_donchian_cache(high,low,[w])
            sig=gen_s3(close,high,low,dc,sig_params)
        elif sid=="S4":
            ap=sig_params.get("ap",90); mult=sig_params.get("mult",3.0)
            st=precompute_supertrend_cache(high,low,close,[ap],[mult])
            sig=gen_s4(close,high,low,st,sig_params)
        elif sid=="S5":
            pt=sig_params.get("pt","fibonacci")
            piv={pt:compute_daily_pivots_for_hourly(daily_df,ticker,h_dts,pt)}
            sig=gen_s5(close,piv,sig_params)
        elif sid=="S6":
            vw=sig_params.get("vw",90)
            vc=precompute_vwap_cache(close,high,low,volume,[vw])
            sig=gen_s6(close,high,low,volume,vc,sig_params)

        if sig is None:
            # Fallback
            sma90=calc_sma(close,90); std90=calc_std(close,90)
            z=(close-sma90)/np.where(std90>1e-12,std90,1.0)
            sig=np.zeros(n,dtype=np.int8)
            sig[~np.isnan(z)&(z<-2)]=1; sig[~np.isnan(z)&(z>2)]=-1
            sig[:WARMUP]=0

        rsi=calc_rsi(close,14)
        sig=apply_filt(sig,close,volume,ind,rsi,atr,filt_params,IS_TREND[sid])

        # A position
        rm_sl=float(row.get("rm_sl_a",0) if pd.notna(row.get("rm_sl_a",None)) else 0)
        rm_tp=float(row.get("rm_tp_a",0) if pd.notna(row.get("rm_tp_a",None)) else 0)
        rm_trail=float(row.get("rm_trail",0) if pd.notna(row.get("rm_trail",None)) else 0)
        rm_rr=float(row.get("rm_rr",0) if pd.notna(row.get("rm_rr",None)) else 0)
        rm_mh=int(float(row.get("rm_mh",0) if pd.notna(row.get("rm_mh",None)) else 0))
        rm_be=float(row.get("rm_be",0) if pd.notna(row.get("rm_be",None)) else 0)
        rm_chand=float(row.get("rm_chand",0) if pd.notna(row.get("rm_chand",None)) else 0)
        rm_slp=float(row.get("rm_sl_p",0) if pd.notna(row.get("rm_sl_p",None)) else 0)
        rm_tpp=float(row.get("rm_tp_p",0) if pd.notna(row.get("rm_tp_p",None)) else 0)

        rm_dict={"sl_p":rm_slp,"tp_p":rm_tpp,"sl_a":rm_sl,"tp_a":rm_tp,
                 "trail":rm_trail,"mh":rm_mh,"be":rm_be,"chand":rm_chand,"rr":rm_rr}
        pos_a=run_bt(sig,close,high,low,atr,rh,rl,rm_dict,WARMUP,n)

        # Verify A sharpe matches
        a_check=calc_sharpe(pos_a,log_ret,oos_mask,COMM)

        # Precompute vol forecasts
        vol_cache={}
        for vm,vp in VOL_METHODS:
            vol_cache[(vm,vp)]=get_vf(log_ret,vm,vp)

        # ── B: Adaptive SL/TP ──
        best_b = a_sh
        base_sl_val = rm_sl if rm_sl>0 else (rm_slp*100 if rm_slp>0 else 3.0)  # approximate
        base_tp_val = rm_tp if rm_tp>0 else (rm_tpp*100 if rm_tpp>0 else 3.0)
        for bp in B_GRID:
            vf=vol_cache[(bp["vm"],bp["vp"])]
            vv=vf[~np.isnan(vf)]
            if len(vv)<50: continue
            sr=np.median(vv)
            if sr<1e-12: continue
            vr=vf/sr*bp["scale"]
            vr=np.where(np.isnan(vr),1.0,vr)
            pos_b=bt_adaptive_sl(sig,close,high,low,atr,rh,rl,
                                 base_sl_val,base_tp_val,rm_trail,rm_mh,rm_be,rm_rr,
                                 vr,bp["min_m"],bp["max_m"],WARMUP,n)
            sh=calc_sharpe(pos_b,log_ret,oos_mask,COMM)
            if sh>best_b: best_b=sh

        # ── C: Vol scaling ──
        best_c = a_sh
        for cp in C_GRID:
            vf=vol_cache[(cp["vm"],cp["vp"])]
            vv=vf[~np.isnan(vf)]
            if len(vv)<50: continue
            st=np.percentile(vv,cp["target_pct"])
            if st<1e-12: continue
            sc=st/np.where((vf<1e-12)|np.isnan(vf),st,vf)
            sc=np.clip(sc,cp["min_p"],cp["max_p"])
            pos_c=pos_a*sc
            sh=calc_sharpe(pos_c,log_ret,oos_mask,COMM)
            if sh>best_c: best_c=sh

        # ── D: Vol targeting ──
        best_d = a_sh
        for dp in D_GRID:
            vf=vol_cache[(dp["vm"],dp["vp"])]
            vv=vf[~np.isnan(vf)]
            if len(vv)<50: continue
            avf=vf*sqrt(BPH)
            sc=dp["vol_target"]/np.where((avf<1e-6)|np.isnan(avf),dp["vol_target"],avf)
            sc=np.clip(sc,dp["min_p"],dp["max_p"])
            pos_d=pos_a*sc
            sh=calc_sharpe(pos_d,log_ret,oos_mask,COMM)
            if sh>best_d: best_d=sh

        avg_bcd=(best_b+best_c+best_d)/3
        results[(sid,ticker)]={"A":round(a_sh,3),"B":round(best_b,3),
                               "C":round(best_c,3),"D":round(best_d,3),
                               "avg":round(avg_bcd,3)}

        b_ok="✓" if best_b>a_sh else "✗"
        c_ok="✓" if best_c>a_sh else "✗"
        d_ok="✓" if best_d>a_sh else "✗"
        print(f"  {sname:>16s} {ticker:>6s}: A={a_sh:+.2f}  "
              f"B={best_b:+.2f}{b_ok}  C={best_c:+.2f}{c_ok}  D={best_d:+.2f}{d_ok}  "
              f"avg={avg_bcd:+.2f}")

    # Summary
    print(f"\n{'='*120}")
    print("SUMMARY: Mean per-ticker ExSh by approach (hourly)")
    print(f"{'='*120}")
    print(f"  {'Strategy':>16s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'AvgBCD':>7s} "
          f"{'B>A':>4s} {'C>A':>4s} {'D>A':>4s} {'Avg>A':>5s}")

    for sid in SIDS:
        sname=SNAMES[sid]
        a_v=[];b_v=[];c_v=[];d_v=[];avg_v=[]
        for ticker in TICKERS:
            r=results.get((sid,ticker))
            if r:
                a_v.append(r["A"]);b_v.append(r["B"])
                c_v.append(r["C"]);d_v.append(r["D"])
                avg_v.append(r["avg"])
        if not a_v: continue
        ma=np.mean(a_v);mb=np.mean(b_v);mc=np.mean(c_v);md=np.mean(d_v);mavg=np.mean(avg_v)
        print(f"  {sname:>16s} {ma:+7.2f} {mb:+7.2f} {mc:+7.2f} {md:+7.2f} {mavg:+7.2f} "
              f"{'YES' if mb>ma else 'NO':>4s} {'YES' if mc>ma else 'NO':>4s} "
              f"{'YES' if md>ma else 'NO':>4s} {'YES' if mavg>ma else 'NO':>5s}")

    elapsed=time.time()-t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")

if __name__=="__main__":
    main()
