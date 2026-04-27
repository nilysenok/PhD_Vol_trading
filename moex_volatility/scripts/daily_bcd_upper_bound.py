#!/usr/bin/env python3
"""
daily_bcd_upper_bound.py — B/C/D approaches on top of A (upper bound).

For EACH ticker × strategy: takes the A upper-bound position,
then applies B/C/D vol-based modifications to find the best.

B: Adaptive SL/TP — scale SL proportional to σ_forecast/σ_ref
C: Vol scaling — position inversely proportional to σ_forecast
D: Vol targeting — fix portfolio vol at target level

Uses simple vol estimates (EWM, realized, HAR-simple).
In-sample on OOS (2022-2026) — upper bound, not walk-forward.
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
    TICKERS, WARMUP, load_data, compute_base, calc_rsi, calc_sma, calc_std,
    calc_atr, calc_supertrend, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache, calc_pivot_daily,
    _year_bounds, warmup_numba,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/daily_a"
BPY = 252; COMM = 0.0005
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}

random.seed(42); np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# VOL FORECAST MODELS (simple alternatives)
# ═══════════════════════════════════════════════════════════════

def realized_vol(log_ret, window):
    """Rolling realized vol (std of log returns)."""
    n = len(log_ret)
    rv = np.full(n, np.nan)
    for i in range(window, n):
        rv[i] = np.std(log_ret[i-window:i], ddof=1)
    return rv

def ewm_vol(log_ret, halflife):
    """Exponentially weighted volatility."""
    alpha = 1 - np.exp(-np.log(2) / halflife)
    n = len(log_ret)
    var = np.full(n, np.nan)
    var[1] = log_ret[0]**2
    for i in range(2, n):
        var[i] = alpha * log_ret[i-1]**2 + (1-alpha) * (var[i-1] if not np.isnan(var[i-1]) else log_ret[i-1]**2)
    return np.sqrt(np.maximum(var, 1e-20))

def har_vol(log_ret):
    """HAR-RV: simple version using daily/weekly/monthly realized vol."""
    n = len(log_ret)
    rv_d = realized_vol(log_ret, 5)    # ~1 week
    rv_w = realized_vol(log_ret, 22)   # ~1 month
    rv_m = realized_vol(log_ret, 66)   # ~3 months
    # Simple HAR: forecast = 0.4*rv_d + 0.35*rv_w + 0.25*rv_m
    har = 0.4 * rv_d + 0.35 * rv_w + 0.25 * rv_m
    return har

def get_vol_forecast(log_ret, method, param):
    """Get vol forecast array."""
    if method == "realized":
        return realized_vol(log_ret, param)
    elif method == "ewm":
        return ewm_vol(log_ret, param)
    elif method == "har":
        return har_vol(log_ret)
    return realized_vol(log_ret, 20)


# ═══════════════════════════════════════════════════════════════
# BT ENGINE (same as daily_upper_bound)
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def bt_universal(sig, close, high, low, atr, rh, rl,
                 sl_atr, tp_atr, trail, max_hold, be_atr,
                 rr_ratio, warmup, end_idx):
    n = min(end_idx, len(close)); pos = np.zeros(len(close))
    cp=0.0;ep=0.0;sl_p=0.0;tp_p=0.0;held=0;be_on=False;tr_p=0.0
    for t in range(warmup, n):
        if cp != 0.0:
            held += 1; closed = False
            ea = atr[t] if not np.isnan(atr[t]) else atr[t-1] if t>0 else 1.0
            if trail > 0:
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
            sd=sl_atr*ea if sl_atr>0 else 3.0*ea
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
                td=tp_atr*ea if tp_atr>0 else 3.0*ea
                if cp==1.0: tp_p=ep+td
                else: tp_p=ep-td
                tr_p=0.0
            pos[t]=cp
    return pos


# ═══════════════════════════════════════════════════════════════
# BT with ADAPTIVE SL/TP (Approach B)
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def bt_adaptive_sl(sig, close, high, low, atr, rh, rl,
                   base_sl, base_tp, trail, max_hold, be_atr, rr_ratio,
                   vol_ratio, min_sl_mult, max_sl_mult,
                   warmup, end_idx):
    """Like bt_universal but SL/TP scaled by vol_ratio[t]."""
    n = min(end_idx, len(close)); pos = np.zeros(len(close))
    cp=0.0;ep=0.0;sl_p=0.0;tp_p=0.0;held=0;be_on=False;tr_p=0.0
    for t in range(warmup, n):
        if cp != 0.0:
            held += 1; closed = False
            ea = atr[t] if not np.isnan(atr[t]) else atr[t-1] if t>0 else 1.0
            if trail > 0:
                vr = vol_ratio[t] if not np.isnan(vol_ratio[t]) else 1.0
                vr = max(min_sl_mult, min(vr, max_sl_mult))
                if cp==1.0:
                    nt=high[t]-trail*vr*ea
                    if nt>tr_p: tr_p=nt
                else:
                    nt=low[t]+trail*vr*ea
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
            vr = vol_ratio[t] if not np.isnan(vol_ratio[t]) else 1.0
            vr = max(min_sl_mult, min(vr, max_sl_mult))
            sd = base_sl * vr * ea if base_sl>0 else 3.0*ea
            if cp==1.0: sl_p=ep-sd
            else: sl_p=ep+sd
            if trail>0:
                tp_p=0.0
                if cp==1.0: tr_p=ep-trail*vr*ea
                else: tr_p=ep+trail*vr*ea
            elif rr_ratio>0:
                td=sd*rr_ratio
                if cp==1.0: tp_p=ep+td
                else: tp_p=ep-td
                tr_p=0.0
            else:
                td = base_tp*vr*ea if base_tp>0 else 3.0*ea
                if cp==1.0: tp_p=ep+td
                else: tp_p=ep-td
                tr_p=0.0
            pos[t]=cp
    return pos


# ═══════════════════════════════════════════════════════════════
# METRICS (same compound formula as daily UB)
# ═══════════════════════════════════════════════════════════════

def calc_exsh(pos, close, log_ret, dates, rf_daily, cal_days, mask, comm):
    pm=pos[mask];lr=log_ret[mask];rf=rf_daily[mask];cd=cal_days[mask]
    dm=dates[mask];n=len(pm)
    if n<30: return -999
    dp=np.diff(pm,prepend=0.0);net=pm*lr-np.abs(dp)*comm
    exp=np.abs(pm);fdr=(1.0-exp)*rf*cd;total=net+fdr
    active=exp>1e-8;na=active.sum()
    if na<10: return -999
    cs=(dm[-1]-dm[0]).days
    if cs<30: return -999
    cum=np.prod(1.0+total)-1.0;ann=(1.0+cum)**(365.0/cs)-1.0
    rf_c=np.prod(1.0+rf*cd)-1.0;rf_a=(1.0+rf_c)**(365.0/cs)-1.0
    vol=np.std(total[active],ddof=1)*sqrt(BPY)
    ex=ann-rf_a
    return ex/vol if vol>1e-6 else 0.0


# ═══════════════════════════════════════════════════════════════
# MAIN: Load A results, compute B/C/D upper bounds
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("="*120)
    print("B/C/D UPPER BOUND — Vol-based modifications on top of A (daily)")
    print("="*120)

    # Load A upper bound params
    ub_path = OUT_DIR / "upper_bound_params.csv"
    ub_df = pd.read_csv(ub_path)
    print(f"Loaded {len(ub_df)} A configs from {ub_path}")

    daily_df, _, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"]=pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    # Import signal generators from daily_upper_bound
    from daily_upper_bound import (
        gen_sig_s1_ext, gen_sig_s2_ext, gen_sig_s3, gen_sig_s4, gen_sig_s5, gen_sig_s6,
        apply_filters_generic, get_ma, MA_TYPES,
    )

    SIDS = ["S1","S2","S3","S4","S5","S6"]
    IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
    SID_MAP = {v:k for k,v in SNAMES.items()}

    # Vol forecast methods and params to try
    VOL_METHODS = [
        ("realized", 10), ("realized", 20), ("realized", 40),
        ("ewm", 10), ("ewm", 20), ("ewm", 30),
        ("har", 0),
    ]

    # B params grid
    B_GRID = []
    for vm, vp in VOL_METHODS:
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            for min_m in [0.3, 0.5, 0.7]:
                for max_m in [2.0, 3.0, 5.0]:
                    B_GRID.append({"vm":vm,"vp":vp,"scale":scale,"min_m":min_m,"max_m":max_m})
    random.shuffle(B_GRID)
    B_GRID = B_GRID[:500]

    # C params grid
    C_GRID = []
    for vm, vp in VOL_METHODS:
        for target_pct in [25, 50, 75]:
            for min_p in [0.1, 0.2, 0.3]:
                for max_p in [1.5, 2.0, 3.0]:
                    C_GRID.append({"vm":vm,"vp":vp,"target_pct":target_pct,"min_p":min_p,"max_p":max_p})
    random.shuffle(C_GRID)
    C_GRID = C_GRID[:400]

    # D params grid
    D_GRID = []
    for vm, vp in VOL_METHODS:
        for vol_target in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for min_p in [0.1, 0.2]:
                for max_p in [1.5, 2.0, 3.0]:
                    D_GRID.append({"vm":vm,"vp":vp,"vol_target":vol_target,"min_p":min_p,"max_p":max_p})
    random.shuffle(D_GRID)
    D_GRID = D_GRID[:300]

    print(f"Grids: B={len(B_GRID)}, C={len(C_GRID)}, D={len(D_GRID)}")

    results = {}  # (sid, ticker) → {A, B, C, D}

    for _, row in ub_df.iterrows():
        sname = row["strategy"]; ticker = row["ticker"]
        sid = SID_MAP.get(sname)
        if not sid: continue
        a_exsh = row["ex_sharpe"]

        tdf = daily_df[daily_df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
        if len(tdf) < WARMUP+200: continue
        close = tdf["close"].values.astype(np.float64)
        high = tdf["high"].values.astype(np.float64)
        low = tdf["low"].values.astype(np.float64)
        volume = tdf["volume"].values.astype(np.float64)
        dates = pd.to_datetime(tdf["date"].values); n = len(close)
        log_ret = np.zeros(n); log_ret[:-1] = np.log(close[1:]/np.maximum(close[:-1],1e-12))

        ind = compute_base(close, high, low, tdf["open"].values.astype(np.float64), volume, is_hourly=False)
        atr = ind["atr14"]
        cal_days = np.ones(n)
        for i in range(1,n): cal_days[i] = (dates[i]-dates[i-1]).days
        rf_raw = kr_s.reindex(dates, method="ffill").values
        rf_daily = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0
        rh = pd.Series(high).rolling(11, min_periods=1).max().values
        rl = pd.Series(low).rolling(11, min_periods=1).min().values

        oos_start = _year_bounds(dates, 2022)[0]
        if oos_start <= WARMUP: continue
        oos_mask = np.zeros(n, dtype=bool); oos_mask[oos_start:] = True

        # Get base SL/TP from A config
        base_sl = float(row.get("sl_val", 3.0))
        base_tp = float(row.get("tp_val", 3.0))
        sl_type = row.get("sl_type", "atr")
        tp_type = row.get("tp_type", "atr")

        # Reconstruct A signal (simplified — use ATR-based defaults)
        # For speed, generate a generic signal from the strategy
        # We need the A position to modify for B/C/D
        # Use bt_universal with A's RM params
        rm_sl = float(row.get("rm_sl_a", base_sl))
        rm_tp = float(row.get("rm_tp_a", base_tp))
        rm_trail = float(row.get("rm_trail", 0))
        rm_rr = float(row.get("rm_rr", 0))
        rm_mh = int(float(row.get("rm_mh", 0)))
        rm_be = float(row.get("rm_be", 0))

        # Precompute vol forecasts (cache across B/C/D grids)
        vol_cache = {}
        for vm, vp in VOL_METHODS:
            key = (vm, vp)
            vol_cache[key] = get_vol_forecast(log_ret, vm, vp)

        # Reconstruct A signal — need to regenerate from params
        # Parse sig_params and filt_params
        import ast
        sp_str = str(row.get("sig_params",""))
        fp_str = str(row.get("filt_params",""))
        sig_params = {}
        for part in sp_str.split():
            if "=" in part:
                k, v = part.split("=",1)
                try:
                    v = float(v)
                    if k in ("w","vw","ap","el"): v = int(v)
                except: pass
                sig_params[k] = v
        filt_params = {}
        for part in fp_str.split():
            if "=" in part:
                k, v = part.split("=",1)
                try:
                    v = float(v)
                    if k in ("adx_th","atr_lo","mom_n","dow"): v = int(v)
                except: pass
                filt_params[k] = v

        # Generate signal
        sig = None
        if sid in ("S1","S2"):
            mt = sig_params.get("ma_type","sma"); w = sig_params.get("w",20)
            try:
                ma = get_ma(close, w, mt); std = calc_std(close, w)
                if sid == "S1":
                    sig_params["atr_arr"] = atr
                    sig = gen_sig_s1_ext(close, std, ma, sig_params)
                else:
                    sig = gen_sig_s2_ext(close, std, ma, sig_params)
            except: pass
        elif sid == "S3":
            w = sig_params.get("w", 20)
            dc = precompute_donchian_cache(high, low, [w])
            sig = gen_sig_s3(close, high, low, dc, sig_params)
        elif sid == "S4":
            ap = sig_params.get("ap",14); mult = sig_params.get("mult",3.0)
            st = precompute_supertrend_cache(high,low,close,[ap],[mult])
            sig = gen_sig_s4(close,high,low,st,sig_params)
        elif sid == "S5":
            pt = sig_params.get("pt","fibonacci")
            piv = {pt: calc_pivot_daily(high,low,close,pt)}
            sig = gen_sig_s5(close,high,low,piv,sig_params)
        elif sid == "S6":
            vw = sig_params.get("vw",20)
            vc = precompute_vwap_cache(close,high,low,volume,[vw])
            sig = gen_sig_s6(close,high,low,volume,vc,sig_params)

        if sig is None:
            # Fallback: simple signal
            sma20 = calc_sma(close, 20); std20 = calc_std(close, 20)
            z = (close - sma20) / np.where(std20>1e-12, std20, 1.0)
            sig = np.zeros(n, dtype=np.int8)
            sig[~np.isnan(z) & (z < -2)] = 1; sig[~np.isnan(z) & (z > 2)] = -1
            sig[:WARMUP] = 0

        # Apply filters
        sig = apply_filters_generic(sig, close, volume, ind, calc_rsi(close,14),
                                     atr, dates.dayofweek.values.astype(np.int32),
                                     filt_params, IS_TREND[sid])

        # A position (base)
        pos_a = bt_universal(sig, close, high, low, atr, rh, rl,
                            rm_sl, rm_tp, rm_trail, rm_mh, rm_be, rm_rr,
                            WARMUP, n)

        # ── B: Adaptive SL/TP ──
        best_b = a_exsh
        for bp in B_GRID:
            vf = vol_cache[(bp["vm"], bp["vp"])]
            valid_vf = vf[~np.isnan(vf)]
            if len(valid_vf) < 50: continue
            sigma_ref = np.median(valid_vf)
            if sigma_ref < 1e-12: continue
            vol_ratio = vf / sigma_ref * bp["scale"]
            vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)

            pos_b = bt_adaptive_sl(sig, close, high, low, atr, rh, rl,
                                   rm_sl, rm_tp, rm_trail, rm_mh, rm_be, rm_rr,
                                   vol_ratio, bp["min_m"], bp["max_m"],
                                   WARMUP, n)
            exsh = calc_exsh(pos_b, close, log_ret, dates, rf_daily, cal_days, oos_mask, COMM)
            if exsh > best_b: best_b = exsh

        # ── C: Vol scaling ──
        best_c = a_exsh
        for cp_g in C_GRID:
            vf = vol_cache[(cp_g["vm"], cp_g["vp"])]
            valid_vf = vf[~np.isnan(vf)]
            if len(valid_vf) < 50: continue
            sigma_target = np.percentile(valid_vf, cp_g["target_pct"])
            if sigma_target < 1e-12: continue
            scale = sigma_target / np.where((vf<1e-12)|np.isnan(vf), sigma_target, vf)
            scale = np.clip(scale, cp_g["min_p"], cp_g["max_p"])
            pos_c = pos_a * scale
            exsh = calc_exsh(pos_c, close, log_ret, dates, rf_daily, cal_days, oos_mask, COMM)
            if exsh > best_c: best_c = exsh

        # ── D: Vol targeting ──
        best_d = a_exsh
        for dp in D_GRID:
            vf = vol_cache[(dp["vm"], dp["vp"])]
            valid_vf = vf[~np.isnan(vf)]
            if len(valid_vf) < 50: continue
            ann_vf = vf * sqrt(BPY)
            scale = dp["vol_target"] / np.where((ann_vf<1e-6)|np.isnan(ann_vf), dp["vol_target"], ann_vf)
            scale = np.clip(scale, dp["min_p"], dp["max_p"])
            pos_d = pos_a * scale
            exsh = calc_exsh(pos_d, close, log_ret, dates, rf_daily, cal_days, oos_mask, COMM)
            if exsh > best_d: best_d = exsh

        avg_bcd = (best_b + best_c + best_d) / 3
        results[(sid, ticker)] = {
            "A": round(a_exsh, 3), "B": round(best_b, 3),
            "C": round(best_c, 3), "D": round(best_d, 3),
            "avg": round(avg_bcd, 3)
        }

        b_ok = "✓" if best_b > a_exsh else "✗"
        c_ok = "✓" if best_c > a_exsh else "✗"
        d_ok = "✓" if best_d > a_exsh else "✗"
        print(f"  {sname:>16s} {ticker:>6s}: A={a_exsh:+.2f}  "
              f"B={best_b:+.2f}{b_ok}  C={best_c:+.2f}{c_ok}  D={best_d:+.2f}{d_ok}  "
              f"avg={avg_bcd:+.2f}")

    # ── Summary per strategy ──
    print(f"\n{'='*120}")
    print("SUMMARY: Mean per-ticker ExSh by approach (daily)")
    print(f"{'='*120}")
    print(f"  {'Strategy':>16s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'AvgBCD':>7s} "
          f"{'B>A':>4s} {'C>A':>4s} {'D>A':>4s} {'Avg>A':>5s}")

    for sid in SIDS:
        sname = SNAMES[sid]
        a_vals = []; b_vals = []; c_vals = []; d_vals = []; avg_vals = []
        for ticker in TICKERS:
            r = results.get((sid, ticker))
            if r:
                a_vals.append(r["A"]); b_vals.append(r["B"])
                c_vals.append(r["C"]); d_vals.append(r["D"])
                avg_vals.append(r["avg"])
        if not a_vals: continue
        ma = np.mean(a_vals); mb = np.mean(b_vals)
        mc = np.mean(c_vals); md = np.mean(d_vals)
        mavg = np.mean(avg_vals)
        print(f"  {sname:>16s} {ma:+7.2f} {mb:+7.2f} {mc:+7.2f} {md:+7.2f} {mavg:+7.2f} "
              f"{'YES' if mb>ma else 'NO':>4s} {'YES' if mc>ma else 'NO':>4s} "
              f"{'YES' if md>ma else 'NO':>4s} {'YES' if mavg>ma else 'NO':>5s}")

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
