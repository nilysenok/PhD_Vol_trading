#!/usr/bin/env python3
"""
bcd_ultra.py — B/C/D ULTRA for daily AND hourly, with FDR-corrected metrics.

25 vol forecast methods × expanded B/C/D grids (5000 each).
Daily: compound FDR formula. Hourly: compound FDR formula (corrected).
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
    apply_filters_generic, get_ma, bt_universal, run_bt,
)
from hourly_upper_bound import (
    gen_s1 as h_gen_s1, gen_s2 as h_gen_s2, gen_s3 as h_gen_s3,
    gen_s4 as h_gen_s4, gen_s5 as h_gen_s5, gen_s6 as h_gen_s6,
    apply_filt as h_apply_filt, get_ma as h_get_ma,
    bt_universal as h_bt_universal, run_bt as h_run_bt,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
BPY=252; BPH=2268; COMM_D=0.0005; COMM_H=0.0004
SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
        "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND={"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP={v:k for k,v in SNAMES.items()}
random.seed(42); np.random.seed(42)

N_BCD = 5000  # per approach


# ═══════════════════════════════════════════════════════════════
# 25 VOL FORECAST METHODS
# ═══════════════════════════════════════════════════════════════

def _rolling_std(lr, w):
    n=len(lr); out=np.full(n,np.nan)
    for i in range(w,n): out[i]=np.std(lr[i-w:i],ddof=1)
    return out

def _rolling_atr(atr, w):
    n=len(atr); out=np.full(n,np.nan)
    for i in range(w,n): out[i]=np.mean(atr[i-w:i])
    return out

def _ewma(lr, lam):
    n=len(lr); v=np.full(n,np.nan); v[1]=lr[0]**2
    for i in range(2,n): v[i]=lam*v[i-1]+(1-lam)*lr[i-1]**2 if not np.isnan(v[i-1]) else lr[i-1]**2
    return np.sqrt(np.maximum(v,1e-20))

def _parkinson(high, low, w):
    n=len(high); out=np.full(n,np.nan)
    hl2 = np.log(np.maximum(high,1e-12)/np.maximum(low,1e-12))**2
    for i in range(w,n): out[i]=sqrt(np.mean(hl2[i-w:i])/(4*np.log(2)))
    return out

def _garman_klass(open_arr, high, low, close, w):
    n=len(close); out=np.full(n,np.nan)
    u=np.log(np.maximum(high,1e-12)/np.maximum(open_arr,1e-12))
    d=np.log(np.maximum(low,1e-12)/np.maximum(open_arr,1e-12))
    c=np.log(np.maximum(close,1e-12)/np.maximum(open_arr,1e-12))
    gk=0.5*u*u+0.5*d*d-(2*np.log(2)-1)*c*c
    for i in range(w,n): out[i]=sqrt(max(np.mean(gk[i-w:i]),1e-20))
    return out

def _har(lr):
    r5=_rolling_std(lr,5); r22=_rolling_std(lr,22); r66=_rolling_std(lr,66)
    return 0.4*r5+0.35*r22+0.25*r66

def _vol_pctile(lr, w, lookback):
    std=_rolling_std(lr,w); n=len(lr); out=np.full(n,np.nan)
    for i in range(lookback,n):
        v=std[i-lookback:i]; v=v[~np.isnan(v)]
        if len(v)>10: out[i]=np.searchsorted(np.sort(v),std[i])/len(v) if not np.isnan(std[i]) else np.nan
    return out

def compute_25_vols(close, high, low, open_arr, log_ret, atr):
    """Compute 25 vol forecast methods. Returns dict of arrays."""
    vols = {}
    for w in [7,14,20,30,50]: vols[f"atr_{w}"] = _rolling_atr(atr, w)
    for w in [10,20,30,50,100]: vols[f"std_{w}"] = _rolling_std(log_ret, w)
    for lam in [0.90,0.94,0.97]: vols[f"ewma_{lam}"] = _ewma(log_ret, lam)
    vols["parkinson_20"] = _parkinson(high, low, 20)
    vols["gk_20"] = _garman_klass(open_arr, high, low, close, 20)
    vols["har"] = _har(log_ret)
    vols["vpct_50"] = _vol_pctile(log_ret, 20, 50)
    vols["vpct_100"] = _vol_pctile(log_ret, 20, 100)
    # Derived
    s20 = _rolling_std(log_ret, 20); s5 = _rolling_std(log_ret, 5)
    vols["vol_mom"] = np.where((s20>1e-12)&~np.isnan(s20)&~np.isnan(s5), s5/s20-1, np.nan)
    # vol_of_vol: rolling std of s20 (keep same length)
    n = len(log_ret)
    vov = np.full(n, np.nan)
    for i in range(40, n):
        chunk = s20[i-20:i]; chunk = chunk[~np.isnan(chunk)]
        if len(chunk) > 5: vov[i] = np.std(chunk, ddof=1)
    vols["vol_of_vol"] = vov
    s50 = _rolling_std(log_ret, 50)
    vols["vol_zscore"] = np.where((vov>1e-12)&~np.isnan(vov)&~np.isnan(s20)&~np.isnan(s50),
                                  (s20-s50)/vov, np.nan)
    return vols

VOL_KEYS = None  # set after first computation


# ═══════════════════════════════════════════════════════════════
# METRICS WITH FDR (universal for daily & hourly)
# ═══════════════════════════════════════════════════════════════

def calc_exsh_fdr(pos, log_ret, cal_factor, mask, comm, bpy):
    """Excess Sharpe with FDR. Position CLAMPED to [-1,1]."""
    pm=np.clip(pos[mask], -1.0, 1.0)  # clamp — no leverage
    lr=log_ret[mask]; cf=cal_factor[mask]; n=len(pm)
    if n<30: return -999
    dp=np.diff(pm,prepend=0.0)
    net=pm*lr-np.abs(dp)*comm
    exp=np.abs(pm); fdr=(1.0-exp)*cf; total=net+fdr
    active=exp>1e-8; na=active.sum()
    if na<10: return -999
    vol=np.std(total[active],ddof=1)*sqrt(bpy)
    if vol<1e-6: return -999
    # Mean-based annualization (stable)
    ann_ret=np.mean(total)*bpy
    rf_ann=np.mean(cf)*bpy
    ex=ann_ret-rf_ann
    return ex/vol


# ═══════════════════════════════════════════════════════════════
# B/C/D RANDOM GENERATORS
# ═══════════════════════════════════════════════════════════════

def random_b(vol_keys):
    vk=random.choice(vol_keys)
    return {"vk":vk,
            "alpha":random.choice([0.1,0.2,0.3,0.5,0.7,1.0,1.3,1.5,2.0,3.0]),
            "min_m":random.choice([0.3,0.5,0.7,1.0]),
            "max_m":random.choice([2.0,3.0,5.0,8.0,10.0]),
            "ref_pct":random.choice([50,75])}

def random_c(vol_keys):
    vk=random.choice(vol_keys)
    return {"vk":vk,
            "target_ann":random.choice([0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40]),
            "power":random.choice([0.3,0.5,0.7,1.0,1.3,1.5,2.0]),
            "min_p":random.choice([0.05,0.1,0.2,0.3]),
            "max_p":random.choice([1.0,1.5,2.0])}

def random_d(vol_keys):
    vk=random.choice(vol_keys)
    return {"vk":vk,
            "vol_target":random.choice([0.03,0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.25,0.30]),
            "min_p":random.choice([0.05,0.1,0.2]),
            "max_p":random.choice([1.0,1.5,2.0]),
            "skip_pct":random.choice([0,70,75,80,85,90,95])}


# ═══════════════════════════════════════════════════════════════
# PROCESS ONE TICKER×STRATEGY
# ═══════════════════════════════════════════════════════════════

def process_one(pos_a, log_ret, cal_factor, mask, comm, bpy, vols, a_exsh):
    vol_keys = list(vols.keys())

    # ── B: Adaptive SL (scale position by vol ratio as proxy) ──
    best_b = a_exsh
    for _ in range(N_BCD):
        bp = random_b(vol_keys)
        vf = vols[bp["vk"]]
        vv = vf[~np.isnan(vf)]
        if len(vv) < 50: continue
        ref = np.percentile(vv, bp["ref_pct"])
        if ref < 1e-12: continue
        ratio = (vf / ref) ** bp["alpha"]
        ratio = np.where(np.isnan(ratio), 1.0, ratio)
        ratio = np.clip(ratio, bp["min_m"], bp["max_m"])
        # B modifies position magnitude by inverse vol ratio (wider SL → smaller effective pos)
        pos_b = pos_a / ratio  # high vol → smaller pos (proxy for wider SL)
        sh = calc_exsh_fdr(pos_b, log_ret, cal_factor, mask, comm, bpy)
        if sh > best_b: best_b = sh

    # ── C: Vol scaling ──
    best_c = a_exsh
    for _ in range(N_BCD):
        cp = random_c(vol_keys)
        vf = vols[cp["vk"]]
        vv = vf[~np.isnan(vf)]
        if len(vv) < 50: continue
        ann_vf = vf * sqrt(bpy)
        target = cp["target_ann"]
        scale = (target / np.where((ann_vf<1e-6)|np.isnan(ann_vf), target, ann_vf)) ** cp["power"]
        scale = np.clip(scale, cp["min_p"], cp["max_p"])
        pos_c = pos_a * scale
        sh = calc_exsh_fdr(pos_c, log_ret, cal_factor, mask, comm, bpy)
        if sh > best_c: best_c = sh

    # ── D: Vol targeting + timing ──
    best_d = a_exsh
    for _ in range(N_BCD):
        dp = random_d(vol_keys)
        vf = vols[dp["vk"]]
        vv = vf[~np.isnan(vf)]
        if len(vv) < 50: continue
        ann_vf = vf * sqrt(bpy)
        scale = dp["vol_target"] / np.where((ann_vf<1e-6)|np.isnan(ann_vf), dp["vol_target"], ann_vf)
        scale = np.clip(scale, dp["min_p"], dp["max_p"])
        # Skip high vol periods
        if dp["skip_pct"] > 0:
            thresh = np.percentile(vv, dp["skip_pct"])
            scale = np.where((~np.isnan(vf)) & (vf > thresh), 0.0, scale)
        pos_d = pos_a * scale
        sh = calc_exsh_fdr(pos_d, log_ret, cal_factor, mask, comm, bpy)
        if sh > best_d: best_d = sh

    return round(best_b,3), round(best_c,3), round(best_d,3)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0=time.time()
    print("="*130)
    print("B/C/D ULTRA — 25 vol methods × 5000 per approach, FDR-corrected metrics")
    print(f"N_BCD={N_BCD} per approach. Daily + Hourly.")
    print("="*130)

    daily_df,hourly_df,_=load_data()
    kr=pd.read_parquet(KR_PATH);kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS=["S1","S2","S3","S4","S5","S6"]

    # ══════════════════════════════════════════════════════════
    # DAILY
    # ══════════════════════════════════════════════════════════
    d_ub_path = BASE/"results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv"
    d_ub = pd.read_csv(d_ub_path)
    print(f"\n── DAILY ({len(d_ub)} configs) ──")

    d_results = {}
    for _, row in d_ub.iterrows():
        sname=row["strategy"]; ticker=row["ticker"]; sid=SID_MAP.get(sname)
        if not sid: continue
        a_exsh_orig = row["ex_sharpe"]

        tdf=daily_df[daily_df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
        if len(tdf)<WARMUP+200: continue
        close=tdf["close"].values.astype(np.float64)
        high=tdf["high"].values.astype(np.float64)
        low=tdf["low"].values.astype(np.float64)
        open_arr=tdf["open"].values.astype(np.float64)
        volume=tdf["volume"].values.astype(np.float64)
        dates=pd.to_datetime(tdf["date"].values);n=len(close)
        log_ret=np.zeros(n);log_ret[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr=calc_atr(high,low,close,14)

        cal_days=np.ones(n)
        for i in range(1,n): cal_days[i]=(dates[i]-dates[i-1]).days
        rf_raw=kr_s.reindex(dates,method="ffill").values
        rf_daily=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        cal_factor=rf_daily*cal_days  # daily FDR factor

        oos_start=_year_bounds(dates,2022)[0]
        if oos_start<=WARMUP: continue
        oos_mask=np.zeros(n,dtype=bool);oos_mask[oos_start:]=True

        # Reconstruct A position
        ind=compute_base(close,high,low,open_arr,volume,is_hourly=False)
        rh=pd.Series(high).rolling(11,min_periods=1).max().values
        rl=pd.Series(low).rolling(11,min_periods=1).min().values

        sp_str=str(row.get("sig_params",""));fp_str=str(row.get("filt_params",""))
        try: sig_params=ast.literal_eval(sp_str) if sp_str.startswith("{") else {}
        except: sig_params={}
        if not sig_params:
            for part in sp_str.split():
                if "=" in part:
                    k,v=part.split("=",1)
                    try: v=float(v);
                    except: pass
                    if k in ("w","vw","ap","el"):
                        try: v=int(float(v))
                        except: pass
                    sig_params[k]=v
        try: filt_params=ast.literal_eval(fp_str) if fp_str.startswith("{") else {}
        except: filt_params={}
        if not filt_params:
            for part in fp_str.split():
                if "=" in part:
                    k,v=part.split("=",1)
                    try: v=float(v)
                    except: pass
                    if k in ("adx_th","atr_lo","mom_n","dow"):
                        try: v=int(float(v))
                        except: pass
                    filt_params[k]=v

        # Generate signal
        sig=None
        if sid in ("S1","S2"):
            mt=sig_params.get("ma_type","sma");w=sig_params.get("w",20)
            try:
                ma=get_ma(close,w,mt);std=calc_std(close,w)
                if sid=="S1":
                    sig_params["atr_arr"]=atr
                    sig=gen_sig_s1_ext(close,std,ma,sig_params)
                else: sig=gen_sig_s2_ext(close,std,ma,sig_params)
            except: pass
        elif sid=="S3":
            w=sig_params.get("w",20);dc=precompute_donchian_cache(high,low,[w])
            sig=gen_sig_s3(close,high,low,dc,sig_params)
        elif sid=="S4":
            ap=sig_params.get("ap",14);mult=sig_params.get("mult",3.0)
            st=precompute_supertrend_cache(high,low,close,[ap],[mult])
            sig=gen_sig_s4(close,high,low,st,sig_params)
        elif sid=="S5":
            pt=sig_params.get("pt","fibonacci")
            piv={pt:calc_pivot_daily(high,low,close,pt)}
            sig=gen_sig_s5(close,high,low,piv,sig_params)
        elif sid=="S6":
            vw=sig_params.get("vw",20);vc=precompute_vwap_cache(close,high,low,volume,[vw])
            sig=gen_sig_s6(close,high,low,volume,vc,sig_params)
        if sig is None:
            sma20=calc_sma(close,20);std20=calc_std(close,20)
            z=(close-sma20)/np.where(std20>1e-12,std20,1.0)
            sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1
            sig[:WARMUP]=0
        rsi=calc_rsi(close,14);dow=dates.dayofweek.values.astype(np.int32)
        sig=apply_filters_generic(sig,close,volume,ind,rsi,atr,dow,filt_params,IS_TREND[sid])

        rm={"sl_p":float(row.get("rm_sl_p",0) or 0),"tp_p":float(row.get("rm_tp_p",0) or 0),
            "sl_a":float(row.get("rm_sl_a",0) or 0),"tp_a":float(row.get("rm_tp_a",0) or 0),
            "trail":float(row.get("rm_trail",0) or 0),"mh":int(float(row.get("rm_mh",0) or 0)),
            "be":float(row.get("rm_be",0) or 0),"chand":float(row.get("rm_chand",0) or 0),
            "rr":float(row.get("rm_rr",0) or 0)}
        pos_a=run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP,n)

        # Compute A with FDR metric
        a_fdr=calc_exsh_fdr(pos_a,log_ret,cal_factor,oos_mask,COMM_D,BPY)

        # 25 vol forecasts
        vols=compute_25_vols(close,high,low,open_arr,log_ret,atr)

        # B/C/D
        bb,cc,dd = process_one(pos_a,log_ret,cal_factor,oos_mask,COMM_D,BPY,vols,a_fdr)
        avg=(bb+cc+dd)/3
        d_results[(sid,ticker)]={"A":round(a_fdr,3),"B":bb,"C":cc,"D":dd,"avg":round(avg,3)}

        b_ok="✓" if bb>a_fdr+0.005 else "="
        c_ok="✓" if cc>a_fdr+0.005 else "="
        d_ok="✓" if dd>a_fdr+0.005 else "="
        print(f"  D {sname:>14s} {ticker:>5s}: A={a_fdr:+.2f} B={bb:+.2f}{b_ok} C={cc:+.2f}{c_ok} D={dd:+.2f}{d_ok} avg={avg:+.2f}")

    # Daily summary
    print(f"\n{'='*100}")
    print("DAILY SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Strategy':>16s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'Avg':>7s} {'B>A':>4s} {'C>A':>4s} {'D>A':>4s}")
    for sid in SIDS:
        sname=SNAMES[sid];av=[];bv=[];cv=[];dv=[];avgv=[]
        for t in TICKERS:
            r=d_results.get((sid,t))
            if r: av.append(r["A"]);bv.append(r["B"]);cv.append(r["C"]);dv.append(r["D"]);avgv.append(r["avg"])
        if not av: continue
        ma=np.mean(av);mb=np.mean(bv);mc=np.mean(cv);md=np.mean(dv);mavg=np.mean(avgv)
        print(f"  {sname:>16s} {ma:+7.2f} {mb:+7.2f} {mc:+7.2f} {md:+7.2f} {mavg:+7.2f} "
              f"{'Y' if mb>ma else 'N':>4s} {'Y' if mc>ma else 'N':>4s} {'Y' if md>ma else 'N':>4s}")

    # ══════════════════════════════════════════════════════════
    # HOURLY
    # ══════════════════════════════════════════════════════════
    h_ub_path = BASE/"results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv"
    h_ub = pd.read_csv(h_ub_path)
    print(f"\n── HOURLY ({len(h_ub)} configs, with FDR) ──")

    h_results = {}
    for _, row in h_ub.iterrows():
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
        log_ret=np.zeros(n);log_ret[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr=calc_atr(high,low,close,14)

        # Calendar factor for FDR
        h_dates=h_dts.normalize()
        cal_factor=np.zeros(n)
        rf_raw=kr_s.reindex(h_dates,method="ffill").values
        rf_vals=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        prev_date=None
        for i in range(n):
            d=h_dates[i]
            if d!=prev_date:
                cd=1
                if prev_date is not None: cd=(d-prev_date).days
                cal_factor[i]=rf_vals[i]*cd
                prev_date=d

        oos_idx=0
        for i in range(n):
            if h_dts[i].year>=2022: oos_idx=i;break
        if oos_idx<=WARMUP_HOURLY: continue
        oos_mask=np.zeros(n,dtype=bool);oos_mask[oos_idx:]=True

        # Reconstruct A position
        ind=compute_base(close,high,low,open_arr,volume,is_hourly=True)
        rh=pd.Series(high).rolling(99,min_periods=1).max().values
        rl=pd.Series(low).rolling(99,min_periods=1).min().values

        sp_str=str(row.get("sig_params",""));fp_str=str(row.get("filt_params",""))
        try: sig_params=ast.literal_eval(sp_str) if sp_str.strip().startswith("{") else {}
        except: sig_params={}
        if not sig_params:
            for part in sp_str.split():
                if "=" in part:
                    k,v=part.split("=",1)
                    try: v=float(v)
                    except: pass
                    if k in ("w","vw","ap","el"):
                        try: v=int(float(v))
                        except: pass
                    sig_params[k]=v
        try: filt_params=ast.literal_eval(fp_str) if fp_str.strip().startswith("{") else {}
        except: filt_params={}

        sig=None
        if sid in ("S1","S2"):
            mt=sig_params.get("ma_type","sma");w=sig_params.get("w",90)
            try:
                ma=h_get_ma(close,w,mt);std=calc_std(close,w)
                mc={(mt,w):(ma,std)}
                if sid=="S1": sig=h_gen_s1(close,mc,atr,sig_params)
                else: sig=h_gen_s2(close,mc,sig_params)
            except: pass
        elif sid=="S3":
            w=sig_params.get("w",90);dc=precompute_donchian_cache(high,low,[w])
            sig=h_gen_s3(close,high,low,dc,sig_params)
        elif sid=="S4":
            ap=sig_params.get("ap",90);mult=sig_params.get("mult",3.0)
            st=precompute_supertrend_cache(high,low,close,[ap],[mult])
            sig=h_gen_s4(close,high,low,st,sig_params)
        elif sid=="S5":
            pt=sig_params.get("pt","fibonacci")
            piv={pt:compute_daily_pivots_for_hourly(daily_df,ticker,h_dts,pt)}
            sig=h_gen_s5(close,piv,sig_params)
        elif sid=="S6":
            vw=sig_params.get("vw",90);vc=precompute_vwap_cache(close,high,low,volume,[vw])
            sig=h_gen_s6(close,high,low,volume,vc,sig_params)
        if sig is None:
            sma90=calc_sma(close,90);std90=calc_std(close,90)
            z=(close-sma90)/np.where(std90>1e-12,std90,1.0)
            sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1
            sig[:WARMUP_HOURLY]=0
        rsi=calc_rsi(close,14)
        sig=h_apply_filt(sig,close,volume,ind,rsi,atr,filt_params,IS_TREND[sid])

        rm={"sl_p":float(row.get("rm_sl_p",0) or 0),"tp_p":float(row.get("rm_tp_p",0) or 0),
            "sl_a":float(row.get("rm_sl_a",0) or 0),"tp_a":float(row.get("rm_tp_a",0) or 0),
            "trail":float(row.get("rm_trail",0) or 0),"mh":int(float(row.get("rm_mh",0) or 0)),
            "be":float(row.get("rm_be",0) or 0),"chand":float(row.get("rm_chand",0) or 0),
            "rr":float(row.get("rm_rr",0) or 0)}
        pos_a=h_run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP_HOURLY,n)

        a_fdr=calc_exsh_fdr(pos_a,log_ret,cal_factor,oos_mask,COMM_H,BPH)

        vols=compute_25_vols(close,high,low,open_arr,log_ret,atr)
        bb,cc,dd=process_one(pos_a,log_ret,cal_factor,oos_mask,COMM_H,BPH,vols,a_fdr)
        avg=(bb+cc+dd)/3
        h_results[(sid,ticker)]={"A":round(a_fdr,3),"B":bb,"C":cc,"D":dd,"avg":round(avg,3)}

        b_ok="✓" if bb>a_fdr+0.005 else "="
        c_ok="✓" if cc>a_fdr+0.005 else "="
        d_ok="✓" if dd>a_fdr+0.005 else "="
        print(f"  H {sname:>14s} {ticker:>5s}: A={a_fdr:+.2f} B={bb:+.2f}{b_ok} C={cc:+.2f}{c_ok} D={dd:+.2f}{d_ok} avg={avg:+.2f}")

    # Hourly summary
    print(f"\n{'='*100}")
    print("HOURLY SUMMARY (with FDR)")
    print(f"{'='*100}")
    print(f"  {'Strategy':>16s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'Avg':>7s} {'B>A':>4s} {'C>A':>4s} {'D>A':>4s}")
    for sid in SIDS:
        sname=SNAMES[sid];av=[];bv=[];cv=[];dv=[];avgv=[]
        for t in TICKERS:
            r=h_results.get((sid,t))
            if r: av.append(r["A"]);bv.append(r["B"]);cv.append(r["C"]);dv.append(r["D"]);avgv.append(r["avg"])
        if not av: continue
        ma=np.mean(av);mb=np.mean(bv);mc=np.mean(cv);md=np.mean(dv);mavg=np.mean(avgv)
        print(f"  {sname:>16s} {ma:+7.2f} {mb:+7.2f} {mc:+7.2f} {md:+7.2f} {mavg:+7.2f} "
              f"{'Y' if mb>ma else 'N':>4s} {'Y' if mc>ma else 'N':>4s} {'Y' if md>ma else 'N':>4s}")

    # ══════════════════════════════════════════════════════════
    # COMBINED FINAL TABLE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("FINAL COMBINED TABLE")
    print(f"{'='*130}")
    print(f"  {'Strategy':>16s} {'D:A':>6s} {'D:B':>6s} {'D:C':>6s} {'D:D':>6s} {'D:Avg':>6s} "
          f"{'H:A':>6s} {'H:B':>6s} {'H:C':>6s} {'H:D':>6s} {'H:Avg':>6s}")
    for sid in SIDS:
        sname=SNAMES[sid]
        da=[d_results.get((sid,t),{}).get("A",0) for t in TICKERS]
        db=[d_results.get((sid,t),{}).get("B",0) for t in TICKERS]
        dc=[d_results.get((sid,t),{}).get("C",0) for t in TICKERS]
        dd=[d_results.get((sid,t),{}).get("D",0) for t in TICKERS]
        davg=[d_results.get((sid,t),{}).get("avg",0) for t in TICKERS]
        ha=[h_results.get((sid,t),{}).get("A",0) for t in TICKERS]
        hb=[h_results.get((sid,t),{}).get("B",0) for t in TICKERS]
        hc=[h_results.get((sid,t),{}).get("C",0) for t in TICKERS]
        hd=[h_results.get((sid,t),{}).get("D",0) for t in TICKERS]
        havg=[h_results.get((sid,t),{}).get("avg",0) for t in TICKERS]
        print(f"  {sname:>16s} {np.mean(da):+6.2f} {np.mean(db):+6.2f} {np.mean(dc):+6.2f} "
              f"{np.mean(dd):+6.2f} {np.mean(davg):+6.2f} "
              f"{np.mean(ha):+6.2f} {np.mean(hb):+6.2f} {np.mean(hc):+6.2f} "
              f"{np.mean(hd):+6.2f} {np.mean(havg):+6.2f}")

    elapsed=time.time()-t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__=="__main__":
    main()
