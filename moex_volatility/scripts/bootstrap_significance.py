#!/usr/bin/env python3
"""
bootstrap_significance.py — Bootstrap CI and p-values for Excess Sharpe.

For each strategy × timeframe:
  1. Reconstruct A positions from saved params
  2. Compute daily/yearly block returns
  3. Bootstrap 1000 resamples → 95% CI, p-value(H₀: ExSh=0)
  4. For C (vol scaling): paired bootstrap C-A → p-value(H₀: C=A)

Uses DAILY block bootstrap (resample years for daily, resample days for hourly).
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
import ast

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
BPY=252; BPH=2268; COMM_D=0.0005; COMM_H=0.0004
SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
        "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND={"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP={v:k for k,v in SNAMES.items()}

N_BOOT = 1000
random.seed(42); np.random.seed(42)


def parse_params(sp_str, fp_str):
    sig_params={}
    try: sig_params=ast.literal_eval(sp_str) if str(sp_str).strip().startswith("{") else {}
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
    try: filt_params=ast.literal_eval(fp_str) if str(fp_str).strip().startswith("{") else {}
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


def sharpe_from_returns(rets, bpy):
    """Sharpe from array of period returns."""
    if len(rets) < 10: return 0.0
    m = np.mean(rets); s = np.std(rets, ddof=1)
    return m / s * sqrt(bpy) if s > 1e-12 else 0.0


def bootstrap_sharpe(rets, bpy, n_boot=N_BOOT):
    """Bootstrap CI for Sharpe ratio. Returns (point, ci_lo, ci_hi, p_value)."""
    point = sharpe_from_returns(rets, bpy)
    n = len(rets)
    if n < 20: return point, point, point, 1.0

    boot_sharpes = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_sharpes[b] = sharpe_from_returns(rets[idx], bpy)

    ci_lo = np.percentile(boot_sharpes, 2.5)
    ci_hi = np.percentile(boot_sharpes, 97.5)
    # p-value: fraction of bootstrap samples <= 0
    p_val = np.mean(boot_sharpes <= 0)
    return point, ci_lo, ci_hi, p_val


def paired_bootstrap_diff(rets_a, rets_b, bpy, n_boot=N_BOOT):
    """Paired bootstrap for difference Sharpe(B) - Sharpe(A). H₀: diff=0."""
    n = min(len(rets_a), len(rets_b))
    if n < 20: return 0, 1.0
    ra = rets_a[:n]; rb = rets_b[:n]
    diff_point = sharpe_from_returns(rb, bpy) - sharpe_from_returns(ra, bpy)

    boot_diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_diffs[b] = sharpe_from_returns(rb[idx], bpy) - sharpe_from_returns(ra[idx], bpy)

    p_val = np.mean(boot_diffs <= 0)  # one-sided: H₁: C > A
    return diff_point, p_val


def compute_daily_returns(pos, log_ret, rf_daily, cal_days, mask, comm):
    """Compute per-bar total returns (trading + FDR) for bootstrap."""
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exp = np.abs(pm)
    fdr = (1.0 - exp) * rf * cd
    return net + fdr  # total return per bar


def vol_scale_position(pos_a, log_ret, window=20):
    """Simple vol scaling: pos_C = pos_A × (σ_target / σ_current)."""
    n = len(pos_a)
    std_arr = np.full(n, np.nan)
    for i in range(window, n):
        std_arr[i] = np.std(log_ret[i-window:i], ddof=1)
    valid = std_arr[~np.isnan(std_arr)]
    if len(valid) < 50: return pos_a.copy()
    target = np.median(valid)
    scale = target / np.where((std_arr < 1e-12) | np.isnan(std_arr), target, std_arr)
    scale = np.clip(scale, 0.1, 2.0)
    return np.clip(pos_a * scale, -1, 1)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 130)
    print(f"BOOTSTRAP SIGNIFICANCE — {N_BOOT} resamples, 95% CI, p-values")
    print("=" * 130)

    daily_df, hourly_df, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS = ["S1","S2","S3","S4","S5","S6"]

    # Fixed A values
    A_D = {"S1":1.21,"S2":1.25,"S3":1.03,"S4":1.26,"S5":1.43,"S6":1.22}
    # WF hourly A/C values
    A_H = {"S1":0.51,"S2":0.47,"S3":0.87,"S4":0.78,"S5":0.32,"S6":0.33}
    C_H = {"S1":0.96,"S2":1.02,"S3":1.05,"S4":1.04,"S5":1.08,"S6":0.89}

    all_results = []

    # ══════════════════════════════════════════════════════════
    # DAILY — bootstrap from upper_bound_params.csv
    # ══════════════════════════════════════════════════════════
    d_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv")
    print(f"\n── DAILY ({len(d_ub)} configs) ──")

    # Aggregate returns per strategy (pool across tickers)
    d_strat_rets_a = {sid: [] for sid in SIDS}
    d_strat_rets_c = {sid: [] for sid in SIDS}

    for _, row in d_ub.iterrows():
        sname = row["strategy"]; ticker = row["ticker"]; sid = SID_MAP.get(sname)
        if not sid: continue

        tdf = daily_df[daily_df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
        if len(tdf) < WARMUP+200: continue
        close = tdf["close"].values.astype(np.float64)
        high = tdf["high"].values.astype(np.float64)
        low = tdf["low"].values.astype(np.float64)
        open_arr = tdf["open"].values.astype(np.float64)
        volume = tdf["volume"].values.astype(np.float64)
        dates = pd.to_datetime(tdf["date"].values); n = len(close)
        lr = np.zeros(n); lr[:-1] = np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr = calc_atr(high,low,close,14)
        cal_days = np.ones(n)
        for i in range(1,n): cal_days[i] = (dates[i]-dates[i-1]).days
        rf_raw = kr_s.reindex(dates,method="ffill").values
        rf_d = np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
        ind = compute_base(close,high,low,open_arr,volume,is_hourly=False)
        rh = pd.Series(high).rolling(11,min_periods=1).max().values
        rl = pd.Series(low).rolling(11,min_periods=1).min().values
        oos_start = _year_bounds(dates,2022)[0]
        if oos_start <= WARMUP: continue
        oos_mask = np.zeros(n,dtype=bool); oos_mask[oos_start:] = True

        # Reconstruct A position
        sp,fp = parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig = None
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
        rm={k:float(row.get(f"rm_{k}",0) or 0) for k in ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
        rm["mh"]=int(float(row.get("rm_mh",0) or 0))
        pos_a = run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP,n)

        # A returns
        rets_a = compute_daily_returns(pos_a, lr, rf_d, cal_days, oos_mask, COMM_D)
        d_strat_rets_a[sid].append(rets_a)

        # C (vol scaling) position
        pos_c = vol_scale_position(pos_a, lr, 20)
        rets_c = compute_daily_returns(pos_c, lr, rf_d, cal_days, oos_mask, COMM_D)
        d_strat_rets_c[sid].append(rets_c)

    # Bootstrap per strategy (pooled across tickers)
    print(f"\n{'Strategy':>16s} {'A_ExSh':>7s} {'A_CI':>16s} {'A_pval':>7s} "
          f"{'C_ExSh':>7s} {'C_CI':>16s} {'C_pval':>7s} "
          f"{'C-A':>6s} {'p(C>A)':>7s}")
    print("-" * 100)

    for sid in SIDS:
        sname = SNAMES[sid]
        # Pool all ticker returns for this strategy
        if not d_strat_rets_a[sid]: continue
        all_a = np.concatenate(d_strat_rets_a[sid])
        all_c = np.concatenate(d_strat_rets_c[sid])

        a_sh, a_lo, a_hi, a_pv = bootstrap_sharpe(all_a, BPY)
        c_sh, c_lo, c_hi, c_pv = bootstrap_sharpe(all_c, BPY)
        diff, diff_pv = paired_bootstrap_diff(all_a, all_c, BPY)

        print(f"  {sname:>16s} {a_sh:+7.2f} [{a_lo:+.2f},{a_hi:+.2f}] {a_pv:7.3f} "
              f"{c_sh:+7.2f} [{c_lo:+.2f},{c_hi:+.2f}] {c_pv:7.3f} "
              f"{diff:+6.2f} {diff_pv:7.3f}")
        all_results.append({"tf":"daily","strategy":sname,"approach":"A",
                           "sharpe":round(a_sh,3),"ci_lo":round(a_lo,3),"ci_hi":round(a_hi,3),"pval":round(a_pv,4)})
        all_results.append({"tf":"daily","strategy":sname,"approach":"C",
                           "sharpe":round(c_sh,3),"ci_lo":round(c_lo,3),"ci_hi":round(c_hi,3),"pval":round(c_pv,4),
                           "diff_C_A":round(diff,3),"pval_C_gt_A":round(diff_pv,4)})

    # ══════════════════════════════════════════════════════════
    # HOURLY — bootstrap per strategy (pooled tickers)
    # ══════════════════════════════════════════════════════════
    h_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")
    print(f"\n── HOURLY ({len(h_ub)} configs) ──")

    h_strat_rets_a = {sid: [] for sid in SIDS}
    h_strat_rets_c = {sid: [] for sid in SIDS}

    for _, row in h_ub.iterrows():
        sname = row["strategy"]; ticker = row["ticker"]; sid = SID_MAP.get(sname)
        if not sid: continue

        hdf = hourly_df[hourly_df["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
        if len(hdf) < WARMUP_HOURLY+500: continue
        close = hdf["close"].values.astype(np.float64)
        high = hdf["high"].values.astype(np.float64)
        low = hdf["low"].values.astype(np.float64)
        open_arr = hdf["open"].values.astype(np.float64)
        volume = hdf["volume"].values.astype(np.float64)
        h_dts = pd.to_datetime(hdf["datetime"].values); n = len(close)
        lr = np.zeros(n); lr[:-1] = np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr = calc_atr(high,low,close,14)
        ind = compute_base(close,high,low,open_arr,volume,is_hourly=True)
        rh = pd.Series(high).rolling(99,min_periods=1).max().values
        rl = pd.Series(low).rolling(99,min_periods=1).min().values
        oos_idx = 0
        for i in range(n):
            if h_dts[i].year >= 2022: oos_idx = i; break
        if oos_idx <= WARMUP_HOURLY: continue
        oos_mask = np.zeros(n,dtype=bool); oos_mask[oos_idx:] = True

        # Reconstruct A position
        sp, fp = parse_params(row.get("sig_params",""), row.get("filt_params",""))
        sig = None
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
        rsi=calc_rsi(close,14)
        sig=h_apply_filt(sig,close,volume,ind,rsi,atr,fp,IS_TREND[sid])
        rm={k:float(row.get(f"rm_{k}",0) or 0) for k in ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
        rm["mh"]=int(float(row.get("rm_mh",0) or 0))
        pos_a = h_run_bt(sig,close,high,low,atr,rh,rl,rm,WARMUP_HOURLY,n)

        # Hourly returns (no FDR — same as WF metric)
        pm = pos_a[oos_mask]; lr_m = lr[oos_mask]
        dp = np.diff(pm,prepend=0.0)
        rets_a = pm * lr_m - np.abs(dp) * COMM_H
        h_strat_rets_a[sid].append(rets_a)

        # C (vol scaling)
        pos_c = vol_scale_position(pos_a, lr, 90)  # hourly: longer window
        pm_c = pos_c[oos_mask]; dp_c = np.diff(pm_c,prepend=0.0)
        rets_c = pm_c * lr_m - np.abs(dp_c) * COMM_H
        h_strat_rets_c[sid].append(rets_c)

    print(f"\n{'Strategy':>16s} {'A_ExSh':>7s} {'A_CI':>16s} {'A_pval':>7s} "
          f"{'C_ExSh':>7s} {'C_CI':>16s} {'C_pval':>7s} "
          f"{'C-A':>6s} {'p(C>A)':>7s}")
    print("-" * 100)

    for sid in SIDS:
        sname = SNAMES[sid]
        if not h_strat_rets_a[sid]: continue
        all_a = np.concatenate(h_strat_rets_a[sid])
        all_c = np.concatenate(h_strat_rets_c[sid])

        a_sh, a_lo, a_hi, a_pv = bootstrap_sharpe(all_a, BPH)
        c_sh, c_lo, c_hi, c_pv = bootstrap_sharpe(all_c, BPH)
        diff, diff_pv = paired_bootstrap_diff(all_a, all_c, BPH)

        print(f"  {sname:>16s} {a_sh:+7.2f} [{a_lo:+.2f},{a_hi:+.2f}] {a_pv:7.3f} "
              f"{c_sh:+7.2f} [{c_lo:+.2f},{c_hi:+.2f}] {c_pv:7.3f} "
              f"{diff:+6.2f} {diff_pv:7.3f}")
        all_results.append({"tf":"hourly","strategy":sname,"approach":"A",
                           "sharpe":round(a_sh,3),"ci_lo":round(a_lo,3),"ci_hi":round(a_hi,3),"pval":round(a_pv,4)})
        all_results.append({"tf":"hourly","strategy":sname,"approach":"C",
                           "sharpe":round(c_sh,3),"ci_lo":round(c_lo,3),"ci_hi":round(c_hi,3),"pval":round(c_pv,4),
                           "diff_C_A":round(diff,3),"pval_C_gt_A":round(diff_pv,4)})

    # Save
    out = BASE / "results/final/strategies/walkforward_v4"
    pd.DataFrame(all_results).to_csv(out / "bootstrap_significance.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved: {out / 'bootstrap_significance.csv'}")


if __name__ == "__main__":
    main()
