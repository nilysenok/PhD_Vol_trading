#!/usr/bin/env python3
"""
msp_correct.py — MSP with CORRECT Sharpe formula.

Key fix: Vol = std(DAILY returns) × sqrt(252), even for hourly strategies.
Hourly returns aggregated to daily before computing vol.
Excess Sharpe = (Annual_Total - RF_Annual) / Vol_Annual.
"""
import os, sys, warnings, time, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy.optimize import minimize

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
from bootstrap_significance import parse_params, vol_scale_position

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
BPY=252; SIDS=["S1","S2","S3","S4","S5","S6"]
SNAMES={"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
        "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND={"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP={v:k for k,v in SNAMES.items()}
N_BOOT=10000; random.seed(42); np.random.seed(42)


def adaptive_sl_pos(pos_a, lr, w=20):
    n=len(pos_a);std_a=np.full(n,np.nan)
    for i in range(w,n): std_a[i]=np.std(lr[i-w:i],ddof=1)
    vv=std_a[~np.isnan(std_a)]
    if len(vv)<50: return pos_a.copy()
    ref=np.percentile(vv,75)
    r=ref/np.where((std_a<1e-12)|np.isnan(std_a),ref,std_a)
    return np.clip(pos_a*np.clip(r,0.3,3.0),-1,1)

def vol_target_pos(pos_a, lr, w=20, tv=0.15):
    n=len(pos_a);std_a=np.full(n,np.nan)
    for i in range(w,n): std_a[i]=np.std(lr[i-w:i],ddof=1)
    ann=std_a*sqrt(BPY)
    s=tv/np.where((ann<1e-6)|np.isnan(ann),tv,ann)
    return np.clip(pos_a*np.clip(s,0.1,2.0),-1,1)


# ═══════════════════════════════════════════════════════════════
# CORRECT SHARPE: daily returns × sqrt(252)
# ═══════════════════════════════════════════════════════════════

def compute_correct_sharpe(daily_returns, rf_daily_returns):
    """
    daily_returns: array of total daily returns (trading + FDR - lev_cost)
    rf_daily_returns: array of daily rf returns
    Both arrays: one per TRADING DAY.
    Sharpe = (ann_total - ann_rf) / (std(daily_total) × sqrt(252))
    """
    n = len(daily_returns)
    if n < 30: return -999, {}

    # Compound annual total return
    cum = np.prod(1.0 + daily_returns) - 1.0
    n_years = n / 252.0
    ann_total = (1.0 + cum) ** (1.0 / n_years) - 1.0 if n_years > 0.1 else cum

    # Compound annual RF
    rf_cum = np.prod(1.0 + rf_daily_returns) - 1.0
    ann_rf = (1.0 + rf_cum) ** (1.0 / n_years) - 1.0 if n_years > 0.1 else rf_cum

    # Vol = std of ALL daily returns × sqrt(252)
    vol = np.std(daily_returns, ddof=1) * sqrt(252)

    excess = ann_total - ann_rf
    sharpe = excess / vol if vol > 1e-6 else 0

    # MDD
    cum_arr = np.cumsum(daily_returns)
    pk = np.maximum.accumulate(cum_arr)
    mdd = np.max(pk - cum_arr) * 100

    return sharpe, {"ann_ret": round(ann_total*100,2), "ann_rf": round(ann_rf*100,2),
                    "excess": round(excess*100,2), "vol": round(vol*100,2),
                    "mdd": round(mdd,2), "n_days": n}


def aggregate_hourly_to_daily(hourly_returns, hourly_dates):
    """Sum hourly returns into daily returns."""
    dates_day = hourly_dates.normalize()
    unique_days = dates_day.unique()
    daily_rets = []
    for d in unique_days:
        mask = dates_day == d
        day_ret = np.sum(hourly_returns[mask])  # sum of log returns
        daily_rets.append(day_ret)
    return np.array(daily_rets), unique_days


# ═══════════════════════════════════════════════════════════════
# RECONSTRUCT + COMPUTE
# ═══════════════════════════════════════════════════════════════

def process_tf(ub_df, data_df, kr_s, is_hourly, daily_df=None):
    """Full pipeline for one timeframe."""
    WU = WARMUP_HOURLY if is_hourly else WARMUP
    dt_col = "datetime" if is_hourly else "date"
    comms = [0.0004, 0.0005] if is_hourly else [0.0005, 0.0006]

    # Per-strategy daily returns for portfolio
    strat_daily = {app: {sid: [] for sid in SIDS} for app in ["A","B","C","D"]}
    strat_rf = {sid: [] for sid in SIDS}

    for _, row in ub_df.iterrows():
        sname=row["strategy"]; ticker=row["ticker"]; sid=SID_MAP.get(sname)
        if not sid: continue
        tdf=data_df[data_df["ticker"]==ticker].sort_values(dt_col).reset_index(drop=True)
        if len(tdf)<WU+200: continue
        close=tdf["close"].values.astype(np.float64)
        high=tdf["high"].values.astype(np.float64)
        low=tdf["low"].values.astype(np.float64)
        open_arr=tdf["open"].values.astype(np.float64)
        volume=tdf["volume"].values.astype(np.float64)
        dates=pd.to_datetime(tdf[dt_col].values);n=len(close)
        lr=np.zeros(n);lr[:-1]=np.log(close[1:]/np.maximum(close[:-1],1e-12))
        atr=calc_atr(high,low,close,14)
        ind=compute_base(close,high,low,open_arr,volume,is_hourly=is_hourly)
        rh_w=99 if is_hourly else 11
        rh=pd.Series(high).rolling(rh_w,min_periods=1).max().values
        rl=pd.Series(low).rolling(rh_w,min_periods=1).min().values

        # Calendar factor for FDR
        if is_hourly:
            h_dates=dates.normalize()
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
        else:
            cal_days=np.ones(n)
            for i in range(1,n): cal_days[i]=(dates[i]-dates[i-1]).days
            rf_raw=kr_s.reindex(dates,method="ffill").values
            rf_d=np.where(np.isfinite(rf_raw),rf_raw,0.15)/365.0
            cal_factor=rf_d*cal_days

        # OOS mask
        if is_hourly:
            oos_idx=0
            for i in range(n):
                if dates[i].year>=2022: oos_idx=i;break
            if oos_idx<=WU: continue
        else:
            oos_idx=_year_bounds(dates,2022)[0]
            if oos_idx<=WU: continue
        oos_mask=np.zeros(n,dtype=bool);oos_mask[oos_idx:]=True

        # Reconstruct signal
        sp,fp=parse_params(row.get("sig_params",""),row.get("filt_params",""))
        sig=None
        if not is_hourly:
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
        else:
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
                sig=h_gen_s5(close,{pt:compute_daily_pivots_for_hourly(daily_df,ticker,dates,pt)},sp)
            elif sid=="S6":
                vw=sp.get("vw",90);sig=h_gen_s6(close,high,low,volume,precompute_vwap_cache(close,high,low,volume,[vw]),sp)
        if sig is None:
            sw=90 if is_hourly else 20
            sma_v=calc_sma(close,sw);std_v=calc_std(close,sw)
            z=(close-sma_v)/np.where(std_v>1e-12,std_v,1.0)
            sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1;sig[:WU]=0
        rsi=calc_rsi(close,14)
        if not is_hourly:
            dow=dates.dayofweek.values.astype(np.int32)
            sig=apply_filters_generic(sig,close,volume,ind,rsi,atr,dow,fp,IS_TREND[sid])
        else:
            sig=h_apply_filt(sig,close,volume,ind,rsi,atr,fp,IS_TREND[sid])
        rm={k:float(row.get(f"rm_{k}",0) or 0) for k in ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
        rm["mh"]=int(float(row.get("rm_mh",0) or 0))

        if is_hourly:
            pos_a=h_run_bt(sig,close,high,low,atr,rh,rl,rm,WU,n)
        else:
            pos_a=run_bt(sig,close,high,low,atr,rh,rl,rm,WU,n)

        vw_b=90 if is_hourly else 20
        pos_b=adaptive_sl_pos(pos_a,lr,vw_b)
        pos_c=vol_scale_position(pos_a,lr,vw_b)
        pos_d=vol_target_pos(pos_a,lr,vw_b,0.15)

        comm=comms[0]  # institutional
        for app, pos in [("A",pos_a),("B",pos_b),("C",pos_c),("D",pos_d)]:
            pm=np.clip(pos[oos_mask],-1,1)
            lr_m=lr[oos_mask]; cf_m=cal_factor[oos_mask]
            dp=np.diff(pm,prepend=0.0)
            net=pm*lr_m-np.abs(dp)*comm
            exp=np.abs(pm); fdr=(1.0-exp)*cf_m
            total=net+fdr  # per-bar total returns

            if is_hourly:
                # Aggregate to daily
                oos_dates=dates[oos_mask]
                dr, _ = aggregate_hourly_to_daily(total, oos_dates)
                rf_dr, _ = aggregate_hourly_to_daily(cf_m, oos_dates)
            else:
                dr = total
                rf_dr = cf_m

            strat_daily[app][sid].append(dr)
        # RF (same for all approaches)
        if is_hourly:
            rf_dr_full, _ = aggregate_hourly_to_daily(cal_factor[oos_mask], dates[oos_mask])
            strat_rf[sid].append(rf_dr_full)
        else:
            strat_rf[sid].append(cal_factor[oos_mask])

    return strat_daily, strat_rf


def build_portfolio(strat_daily, approach, rf_data):
    """Build EW portfolio from per-strategy daily returns."""
    sids_avail = [s for s in SIDS if strat_daily[approach][s]]
    if len(sids_avail) < 6: return None, None

    # EW across tickers per strategy
    strat_ew = {}
    for sid in sids_avail:
        rets = strat_daily[approach][sid]
        ml = min(len(r) for r in rets)
        strat_ew[sid] = np.mean([r[:ml] for r in rets], axis=0)

    # EW across strategies
    ml2 = min(len(strat_ew[s]) for s in sids_avail)
    port = np.mean([strat_ew[s][:ml2] for s in sids_avail], axis=0)

    # RF
    rf_lists = [rf_data[s] for s in sids_avail if rf_data[s]]
    if rf_lists:
        ml3 = min(min(len(r) for r in rf_lists[0]), ml2)
        rf_port = np.mean([np.mean([r[:ml3] for r in rf_data[s]], axis=0) for s in sids_avail], axis=0)
    else:
        rf_port = np.zeros(ml2)

    return port[:min(ml2, len(rf_port))], rf_port[:min(ml2, len(rf_port))]


def main():
    t0=time.time()
    print("="*130)
    print("MSP CORRECT — Vol = std(daily returns) × sqrt(252)")
    print("Excess Sharpe = (Annual_Total - RF_Annual) / Vol_Annual")
    print("="*130)

    daily_df,hourly_df,_=load_data()
    kr=pd.read_parquet(KR_PATH);kr["date"]=pd.to_datetime(kr["date"])
    kr_s=kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    d_ub=pd.read_csv(BASE/"results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv")
    h_ub=pd.read_csv(BASE/"results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")

    for tf_label, ub_df, data_df, is_hourly in [
        ("DAILY", d_ub, daily_df, False),
        ("HOURLY", h_ub, hourly_df, True),
    ]:
        print(f"\n{'='*130}")
        print(f"  {tf_label}")
        print(f"{'='*130}")

        print("  Reconstructing...")
        sd, rf_d = process_tf(ub_df, data_df, kr_s, is_hourly,
                               daily_df if is_hourly else None)

        # ── Diagnostic: show all vol variants for MSP-C hourly ──
        if is_hourly and tf_label == "HOURLY":
            port_c, rf_c = build_portfolio(sd, "C", rf_d)
            if port_c is not None:
                n = len(port_c)
                print(f"\n  DIAGNOSTIC: MSP-C hourly vol variants:")
                print(f"    N daily returns: {n}")
                print(f"    Annual total return: {(np.prod(1+port_c)-1)**(252.0/n)*100-100:+.1f}%")
                print(f"    Mean daily return: {np.mean(port_c)*252*100:+.2f}%")
                vol_c = np.std(port_c, ddof=1) * sqrt(252) * 100
                print(f"    Vol (daily × sqrt(252)): {vol_c:.1f}%")
                sh_c, info_c = compute_correct_sharpe(port_c, rf_c)
                print(f"    CORRECT Sharpe: {sh_c:+.2f}")
                print(f"    Info: {info_c}")

        # ── All 6 MSP ──
        print(f"\n  TABLE: All MSP (EW, institutional, CORRECT Sharpe)")
        print(f"  {'MSP':>10s} {'Sharpe':>8s} {'Ret%':>7s} {'Vol%':>6s} {'MDD%':>6s}")

        msp_ports = {}; msp_rfs = {}
        for app in ["A","B","C","D"]:
            port, rf = build_portfolio(sd, app, rf_d)
            if port is None: continue
            sh, info = compute_correct_sharpe(port, rf)
            msp_ports[f"MSP-{app}"] = port
            msp_rfs[f"MSP-{app}"] = rf
            print(f"  {'MSP-'+app:>10s} {sh:+8.2f} {info['ann_ret']:+7.1f} {info['vol']:6.1f} {info['mdd']:6.1f}")

        # MEAN
        if all(f"MSP-{a}" in msp_ports for a in ["B","C","D"]):
            ml = min(len(msp_ports[f"MSP-{a}"]) for a in ["B","C","D"])
            mean_p = (msp_ports["MSP-B"][:ml]+msp_ports["MSP-C"][:ml]+msp_ports["MSP-D"][:ml])/3
            rf_mean = msp_rfs["MSP-C"][:ml]
            sh, info = compute_correct_sharpe(mean_p, rf_mean)
            msp_ports["MSP-MEAN"] = mean_p; msp_rfs["MSP-MEAN"] = rf_mean
            print(f"  {'MSP-MEAN':>10s} {sh:+8.2f} {info['ann_ret']:+7.1f} {info['vol']:6.1f} {info['mdd']:6.1f}")

        # BEST
        best_strat = {}
        for sid in SIDS:
            best_sh = -999; best_app = "A"
            for app in ["A","B","C","D"]:
                if not sd[app][sid]: continue
                rets = sd[app][sid]
                ml = min(len(r) for r in rets)
                ew = np.mean([r[:ml] for r in rets], axis=0)
                rf_ew = np.mean([r[:ml] for r in rf_d[sid]], axis=0) if rf_d[sid] else np.zeros(ml)
                sh_t, _ = compute_correct_sharpe(ew, rf_ew[:ml])
                if sh_t > best_sh: best_sh = sh_t; best_app = app; best_strat[sid] = ew
        if len(best_strat) == 6:
            ml = min(len(best_strat[s]) for s in SIDS)
            best_p = np.mean([best_strat[s][:ml] for s in SIDS], axis=0)
            rf_best = msp_rfs.get("MSP-A", np.zeros(ml))[:ml]
            sh, info = compute_correct_sharpe(best_p, rf_best)
            msp_ports["MSP-BEST"] = best_p; msp_rfs["MSP-BEST"] = rf_best
            print(f"  {'MSP-BEST':>10s} {sh:+8.2f} {info['ann_ret']:+7.1f} {info['vol']:6.1f} {info['mdd']:6.1f}")

        # ── Bootstrap ──
        print(f"\n  Bootstrap {N_BOOT:,}: MSP-X vs MSP-A")
        if "MSP-A" not in msp_ports: continue
        pa = msp_ports["MSP-A"]; rfa = msp_rfs["MSP-A"]

        print(f"  {'Comp':>12s} {'Δ Sharpe':>9s} {'95% CI':>18s} {'p-value':>9s}")
        for comp in ["B","C","D","MEAN","BEST"]:
            key = f"MSP-{comp}"
            if key not in msp_ports: continue
            px = msp_ports[key]; rfx = msp_rfs[key]
            ml = min(len(pa), len(px))
            pa_s = pa[:ml]; px_s = px[:ml]; rfa_s = rfa[:ml]; rfx_s = rfx[:ml]

            sh_a, _ = compute_correct_sharpe(pa_s, rfa_s)
            sh_x, _ = compute_correct_sharpe(px_s, rfx_s)
            diff = sh_x - sh_a

            bd = np.zeros(N_BOOT)
            for b in range(N_BOOT):
                idx = np.random.randint(0, ml, size=ml)
                s_a, _ = compute_correct_sharpe(pa_s[idx], rfa_s[idx])
                s_x, _ = compute_correct_sharpe(px_s[idx], rfx_s[idx])
                bd[b] = s_x - s_a
            p = np.mean(bd <= 0)
            ci_lo = np.percentile(bd, 2.5); ci_hi = np.percentile(bd, 97.5)
            sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
            print(f"  {comp+' vs A':>12s} {diff:+9.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] {p:9.4f} {sig}")

        # ── Scaling ──
        if "MSP-C" in msp_ports:
            print(f"\n  Scaling MSP-C with leverage cost:")
            base_p = msp_ports["MSP-C"]; base_rf = msp_rfs["MSP-C"]
            ml = len(base_p)
            # Approximate average exposure from the data
            # Use rough estimate: for daily ~45%, hourly ~11%
            avg_exp = 0.11 if is_hourly else 0.45

            rf_ann = 0.21  # approx KS

            print(f"  {'k':>4s} {'Sharpe':>8s} {'Ret%':>7s} {'Vol%':>6s} {'MDD%':>6s} {'LevCost%':>9s}")
            for k in [1,2,3,5,8]:
                # Scale trading component, recompute FDR and leverage
                scaled_exp = avg_exp * k
                free = max(0, 1.0 - scaled_exp)
                lev = max(0, scaled_exp - 1.0)

                # Trading returns scale linearly
                # FDR and leverage cost change
                # Approximate: total = k*trading + free*rf - lev*(rf+0.02)
                # We know base total = trading + fdr_base
                # trading_base = total_base - fdr_base ≈ total_base - (1-avg_exp)*rf/252
                fdr_per_day = (1-avg_exp) * rf_ann / 252
                trading_base = base_p - fdr_per_day  # approximate

                scaled_trading = trading_base * k
                fdr_new = free * rf_ann / 252
                lev_cost = lev * (rf_ann + 0.02) / 252
                total_new = scaled_trading + fdr_new - lev_cost

                sh, info = compute_correct_sharpe(total_new, base_rf)
                lev_cost_ann = lev * (rf_ann + 0.02) * 100
                print(f"  {k:>4d} {sh:+8.2f} {info['ann_ret']:+7.1f} {info['vol']:6.1f} {info['mdd']:6.1f} {lev_cost_ann:9.1f}")

    print(f"\nDONE in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__=="__main__":
    main()
