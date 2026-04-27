#!/usr/bin/env python3
"""
hourly_full_pipeline.py — Complete hourly pipeline:
  Step 1: A upper bound with FDR metric (in-sample on OOS 2022-2026)
  Step 2: B/C/D upper bound on top of A (25 vol methods × 5000)
  Step 3: Bootstrap significance (A, C, C>A)

FDR metric: compound annual return, calendar FDR 365, vol on active hours ddof=1 × √2268.
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
    calc_sma, calc_std, calc_atr, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    compute_daily_pivots_for_hourly, _year_bounds, warmup_numba,
)
from hourly_upper_bound import (
    bt_universal, run_bt, get_ma, MA_TYPES, H_WINDOWS,
    gen_s1, gen_s2, gen_s3, gen_s4, gen_s5, gen_s6, apply_filt,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/hourly_a"
BPH = 2268; COMM = 0.0004
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
random.seed(42); np.random.seed(42)

N_SIG = 3000; N_RM = 2000; N_TOP = 10; N_BCD = 5000; N_BOOT = 1000


# ═══════════════════════════════════════════════════════════════
# FDR METRIC (the one we optimize for)
# ═══════════════════════════════════════════════════════════════

def calc_fdr_sharpe(pos, log_ret, cal_factor, mask, comm):
    """Compound annual with FDR, vol on active hours, √2268."""
    pm = np.clip(pos[mask], -1, 1); lr = log_ret[mask]; cf = cal_factor[mask]
    n = len(pm)
    if n < 200: return -999
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exp = np.abs(pm); fdr = (1.0 - exp) * cf; total = net + fdr
    active = exp > 1e-8; na = active.sum()
    if na < 50: return -999
    vol = np.std(total[active], ddof=1) * sqrt(BPH)
    if vol < 1e-6: return -999
    cum = np.prod(1.0 + total) - 1.0
    # Calendar span approximation
    n_days = n / 9.0; cs = n_days / 252.0 * 365.0
    if cs < 30: return -999
    ann = (1.0 + cum) ** (365.0 / cs) - 1.0
    rf_c = np.prod(1.0 + cf) - 1.0; rf_a = (1.0 + rf_c) ** (365.0 / cs) - 1.0
    return (ann - rf_a) / vol


def get_total_returns(pos, log_ret, cal_factor, mask, comm):
    """Get per-bar total returns for bootstrap."""
    pm = np.clip(pos[mask], -1, 1); lr = log_ret[mask]; cf = cal_factor[mask]
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exp = np.abs(pm); fdr = (1.0 - exp) * cf
    return net + fdr


# ═══════════════════════════════════════════════════════════════
# RANDOM GENERATORS (same as hourly_upper_bound)
# ═══════════════════════════════════════════════════════════════

def random_sig(sid):
    if sid=="S1":
        return {"ma_type":random.choice(MA_TYPES),"w":random.choice(H_WINDOWS),
                "dev_type":random.choice(["std","atr"]),
                "dev_val":random.choice([0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0])}
    elif sid=="S2":
        return {"ma_type":random.choice(MA_TYPES),"w":random.choice(H_WINDOWS),
                "bs":random.choice([0.5,0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0])}
    elif sid=="S3": return {"w":random.choice(H_WINDOWS)}
    elif sid=="S4":
        return {"ap":random.choice(H_WINDOWS),
                "mult":random.choice([0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0])}
    elif sid=="S5":
        return {"pt":random.choice(["classic","fibonacci","woodie","camarilla"]),"el":random.choice([1,2,3])}
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
    if r<0.15: sp=random.choice([0.003,0.005,0.01,0.015,0.02,0.03,0.05]);sa=0.0
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
# VOL FORECASTS (simplified 18 methods)
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

def compute_vols(lr, atr):
    v={}
    for w in [45,90,180,360]: v[f"std{w}"]=_rstd(lr,w)
    for w in [45,90,180]: v[f"atr{w}"]=_ratr(atr,w)
    for lam in [0.90,0.94,0.97]: v[f"ewm{lam}"]=_ewma(lr,lam)
    r45=_rstd(lr,45);r198=_rstd(lr,198);r594=_rstd(lr,594)
    v["har"]=np.where(~np.isnan(r45)&~np.isnan(r198),
                      0.5*r45+0.3*r198+0.2*np.where(np.isnan(r594),r198,r594), r45)
    return v


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

def sharpe_from_rets(rets, bpy):
    if len(rets)<10: return 0
    m=np.mean(rets);s=np.std(rets,ddof=1)
    return m/s*sqrt(bpy) if s>1e-12 else 0

def bootstrap_ci(rets, bpy, n_boot=N_BOOT):
    point=sharpe_from_rets(rets,bpy); n=len(rets)
    if n<20: return point,point,point,1.0
    bs=np.zeros(n_boot)
    for b in range(n_boot):
        idx=np.random.randint(0,n,size=n); bs[b]=sharpe_from_rets(rets[idx],bpy)
    return point, np.percentile(bs,2.5), np.percentile(bs,97.5), np.mean(bs<=0)

def paired_bootstrap(ra, rb, bpy, n_boot=N_BOOT):
    n=min(len(ra),len(rb))
    if n<20: return 0,1.0
    ra=ra[:n];rb=rb[:n]
    bd=np.zeros(n_boot)
    for b in range(n_boot):
        idx=np.random.randint(0,n,size=n)
        bd[b]=sharpe_from_rets(rb[idx],bpy)-sharpe_from_rets(ra[idx],bpy)
    return sharpe_from_rets(rb,bpy)-sharpe_from_rets(ra,bpy), np.mean(bd<=0)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0=time.time()
    print("="*120)
    print("HOURLY FULL PIPELINE — A(FDR) upper bound → B/C/D → Bootstrap")
    print(f"A: {N_SIG} sigs × {N_RM} RM. B/C/D: {N_BCD} each. Bootstrap: {N_BOOT}")
    print("="*120)

    daily_df,hourly_df,_=load_data()
    kr=pd.read_parquet(KR_PATH);kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS=["S1","S2","S3","S4","S5","S6"]

    # Collect per-strategy pooled returns for bootstrap
    strat_rets_a={s:[] for s in SIDS}
    strat_rets_c={s:[] for s in SIDS}
    strat_exsh_a={s:[] for s in SIDS}
    strat_exsh_c={s:[] for s in SIDS}

    for sid in SIDS:
        sname=SNAMES[sid]; is_trend=IS_TREND[sid]
        print(f"\n── {sname} ──")

        for ticker in TICKERS:
            hdf=hourly_df[hourly_df["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
            if len(hdf)<WARMUP+500: continue
            close=hdf["close"].values.astype(np.float64)
            high=hdf["high"].values.astype(np.float64)
            low=hdf["low"].values.astype(np.float64)
            open_arr=hdf["open"].values.astype(np.float64)
            volume=hdf["volume"].values.astype(np.float64)
            h_dts=pd.to_datetime(hdf["datetime"].values);n=len(close)
            lr=np.zeros(n);lr[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
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
                if h_dts[i].year>=2022: oos_idx=i;break
            if oos_idx<=WARMUP: continue
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
                    pivot_data[pv]=compute_daily_pivots_for_hourly(daily_df,ticker,h_dts,pv)

            def _gen(sp):
                if sid=="S1": return gen_s1(close,ma_cache,atr,sp)
                elif sid=="S2": return gen_s2(close,ma_cache,sp)
                elif sid=="S3": return gen_s3(close,high,low,dc_cache,sp)
                elif sid=="S4": return gen_s4(close,high,low,st_cache,sp)
                elif sid=="S5": return gen_s5(close,pivot_data,sp)
                elif sid=="S6": return gen_s6(close,high,low,volume,vwap_cache,sp)
                return None

            default_rm=dict(sl_p=0,tp_p=0,sl_a=2.0,tp_a=3.0,trail=0,mh=90,be=0,chand=0,rr=0)

            # ── STEP 1: A upper bound with FDR ──
            sig_results=[]
            for _ in range(N_SIG):
                sp=random_sig(sid);s=_gen(sp)
                if s is None: continue
                fp=random_filt(is_trend)
                s=apply_filt(s,close,volume,ind,rsi,atr,fp,is_trend)
                if np.sum(np.abs(s[oos_mask])>0)<5: continue
                pos=run_bt(s,close,high,low,atr,rh,rl,default_rm,WARMUP,n)
                exsh=calc_fdr_sharpe(pos,lr,cal_factor,oos_mask,COMM)
                if exsh>-900: sig_results.append((exsh,s.copy()))
            sig_results.sort(key=lambda x:x[0],reverse=True)
            top_sigs=sig_results[:N_TOP]

            best_a=-999; best_pos_a=None
            if top_sigs:
                for _,s in top_sigs:
                    for _ in range(N_RM//N_TOP):
                        rm=random_rm()
                        pos=run_bt(s,close,high,low,atr,rh,rl,rm,WARMUP,n)
                        exsh=calc_fdr_sharpe(pos,lr,cal_factor,oos_mask,COMM)
                        if exsh>best_a: best_a=exsh; best_pos_a=pos.copy()

            if best_pos_a is None or best_a<-900:
                print(f"  {ticker:>6s}: no A result"); continue

            # ── STEP 2: C (vol scaling) upper bound ──
            vols=compute_vols(lr,atr)
            vol_keys=list(vols.keys())
            best_c=best_a; best_pos_c=best_pos_a.copy()
            for _ in range(N_BCD):
                vk=random.choice(vol_keys); vf=vols[vk]
                vv=vf[~np.isnan(vf)]
                if len(vv)<50: continue
                tgt=np.percentile(vv,random.choice([25,50,75]))
                if tgt<1e-12: continue
                power=random.choice([0.3,0.5,0.7,1.0,1.3,1.5,2.0])
                scale=(tgt/np.where((vf<1e-12)|np.isnan(vf),tgt,vf))**power
                scale=np.clip(scale,random.choice([0.05,0.1,0.2]),random.choice([1.5,2.0,3.0]))
                pos_c=best_pos_a*scale
                exsh=calc_fdr_sharpe(pos_c,lr,cal_factor,oos_mask,COMM)
                if exsh>best_c: best_c=exsh; best_pos_c=pos_c.copy()

            strat_exsh_a[sid].append(best_a)
            strat_exsh_c[sid].append(best_c)

            # Get returns for bootstrap
            rets_a=get_total_returns(best_pos_a,lr,cal_factor,oos_mask,COMM)
            rets_c=get_total_returns(best_pos_c,lr,cal_factor,oos_mask,COMM)
            strat_rets_a[sid].append(rets_a)
            strat_rets_c[sid].append(rets_c)

            c_ok="✓" if best_c>best_a+0.005 else "="
            print(f"  {ticker:>6s}: A={best_a:+.2f}  C={best_c:+.2f}{c_ok}  diff={best_c-best_a:+.2f}")

    # ── STEP 3: Summary + Bootstrap ──
    print(f"\n{'='*120}")
    print("SUMMARY: Mean per-ticker ExSh")
    print(f"{'='*120}")
    print(f"  {'Strategy':>16s} {'A':>7s} {'C':>7s} {'C-A':>6s} {'C>A':>4s}")
    for sid in SIDS:
        if not strat_exsh_a[sid]: continue
        ma=np.mean(strat_exsh_a[sid]); mc=np.mean(strat_exsh_c[sid])
        print(f"  {SNAMES[sid]:>16s} {ma:+7.2f} {mc:+7.2f} {mc-ma:+6.2f} "
              f"{'Y' if mc>ma else 'N':>4s}")

    print(f"\n{'='*120}")
    print(f"BOOTSTRAP SIGNIFICANCE ({N_BOOT} resamples)")
    print(f"{'='*120}")
    print(f"  {'Strategy':>16s} {'A_ExSh':>7s} {'A_CI':>16s} {'A_p':>6s} "
          f"{'C_ExSh':>7s} {'C_CI':>16s} {'C_p':>6s} "
          f"{'C-A':>6s} {'p(C>A)':>7s}")

    for sid in SIDS:
        sname=SNAMES[sid]
        if not strat_rets_a[sid]: continue
        all_a=np.concatenate(strat_rets_a[sid])
        all_c=np.concatenate(strat_rets_c[sid])
        a_sh,a_lo,a_hi,a_pv=bootstrap_ci(all_a,BPH)
        c_sh,c_lo,c_hi,c_pv=bootstrap_ci(all_c,BPH)
        diff,diff_pv=paired_bootstrap(all_a,all_c,BPH)
        print(f"  {sname:>16s} {a_sh:+7.2f} [{a_lo:+.2f},{a_hi:+.2f}] {a_pv:6.3f} "
              f"{c_sh:+7.2f} [{c_lo:+.2f},{c_hi:+.2f}] {c_pv:6.3f} "
              f"{diff:+6.2f} {diff_pv:7.3f}")

    elapsed=time.time()-t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__=="__main__":
    main()
