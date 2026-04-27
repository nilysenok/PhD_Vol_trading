#!/usr/bin/env python3
"""
daily_apply_ub_params.py — Step 3: Apply upper-bound parameters in walk-forward.

Reads upper_bound_params.csv (optimal params found on OOS 2022-2026).
For each ticker × strategy: applies THOSE EXACT PARAMETERS in walk-forward
(train to generate signals, test for evaluation).

This tests: if we KNEW the right methodology, would it generalize?
"""
import os, sys, warnings, time
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from ast import literal_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP, load_data, compute_base, calc_rsi, calc_sma, calc_std,
    calc_atr, calc_adx, calc_supertrend, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache, calc_pivot_daily,
    _year_bounds, warmup_numba,
)

# Import bt_universal + MA functions from our scripts
from daily_upper_bound import (
    bt_universal, run_bt, calc_excess_sharpe, get_ma,
    gen_sig_s1_ext, gen_sig_s2_ext, gen_sig_s3, gen_sig_s4, gen_sig_s5, gen_sig_s6,
    apply_filters_generic, MA_TYPES, WINDOWS,
)

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/daily_a"
BPY = 252; COMM = 0.0005
OOS_YEARS = [2022, 2023, 2024, 2025]
IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}


def apply_fixed_params(sid, ticker, daily_data, kr_data, sig_params, filt_params, rm_params):
    """Apply FIXED parameters (from upper bound) in walk-forward OOS evaluation."""
    tdf = daily_data[daily_data["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP+200: return -999, {}

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)
    log_ret = np.zeros(n); log_ret[:-1] = np.log(close[1:]/np.maximum(close[:-1],1e-12))

    ind = compute_base(close, high, low, open_arr, volume, is_hourly=False)
    rsi = calc_rsi(close, 14); atr = ind["atr14"]
    cal_days = np.ones(n)
    for i in range(1,n): cal_days[i] = (dates[i]-dates[i-1]).days
    rf_raw = kr_data.reindex(dates, method="ffill").values
    rf_daily = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0
    dow = dates.dayofweek.values.astype(np.int32)
    rh = pd.Series(high).rolling(11, min_periods=1).max().values
    rl = pd.Series(low).rolling(11, min_periods=1).min().values
    is_trend = IS_TREND[sid]

    # Precompute caches
    ma_cache = {}
    if sid in ("S1","S2"):
        mt = sig_params.get("ma_type","sma"); w = sig_params.get("w",20)
        try:
            ma = get_ma(close, w, mt); std = calc_std(close, w)
            ma_cache[(mt,w)] = (ma, std)
        except: pass

    dc_cache = {}
    if sid == "S3":
        w = sig_params.get("w",20)
        dc_cache = precompute_donchian_cache(high, low, [w])

    st_cache = {}
    if sid == "S4":
        ap = sig_params.get("ap",14); mult = sig_params.get("mult",3.0)
        st_cache = precompute_supertrend_cache(high, low, close, [ap], [mult])

    vwap_cache = {}
    if sid == "S6":
        vw = sig_params.get("vw",20)
        vwap_cache = precompute_vwap_cache(close, high, low, volume, [vw])

    pivot_data = {}
    if sid == "S5":
        pt = sig_params.get("pt","classic")
        pivot_data = {pt: calc_pivot_daily(high, low, close, pt)}

    # Generate signal with FIXED params
    if sid == "S1":
        sig_params["atr_arr"] = atr
        key = (sig_params.get("ma_type","sma"), sig_params.get("w",20))
        if key not in ma_cache: return -999, {}
        ma, std = ma_cache[key]
        sig = gen_sig_s1_ext(close, std, ma, sig_params)
    elif sid == "S2":
        key = (sig_params.get("ma_type","sma"), sig_params.get("w",20))
        if key not in ma_cache: return -999, {}
        ma, std = ma_cache[key]
        sig = gen_sig_s2_ext(close, std, ma, sig_params)
    elif sid == "S3":
        sig = gen_sig_s3(close, high, low, dc_cache, sig_params)
    elif sid == "S4":
        sig = gen_sig_s4(close, high, low, st_cache, sig_params)
    elif sid == "S5":
        sig = gen_sig_s5(close, high, low, pivot_data, sig_params)
    elif sid == "S6":
        sig = gen_sig_s6(close, high, low, volume, vwap_cache, sig_params)
    else:
        return -999, {}

    if sig is None: return -999, {}

    # Apply FIXED filters
    sig = apply_filters_generic(sig, close, volume, ind, rsi, atr, dow, filt_params, is_trend)

    # OOS mask: 2022-2026
    oos_start = _year_bounds(dates, 2022)[0]
    if oos_start <= WARMUP: return -999, {}
    oos_mask = np.zeros(n, dtype=bool); oos_mask[oos_start:] = True

    if np.sum(np.abs(sig[oos_mask]) > 0) < 3: return -999, {}

    # Run BT with FIXED RM params
    pos = run_bt(sig, close, high, low, atr, rh, rl, rm_params, WARMUP, n)

    return calc_excess_sharpe(pos, close, log_ret, dates, rf_daily, cal_days, oos_mask, COMM)


def main():
    t0 = time.time()
    print("="*130)
    print("STEP 3: Apply upper-bound params in walk-forward evaluation")
    print("="*130)

    # Load upper bound params
    ub_path = OUT_DIR / "upper_bound_params.csv"
    if not ub_path.exists():
        print(f"ERROR: {ub_path} not found. Run daily_upper_bound.py first.")
        return
    ub_df = pd.read_csv(ub_path)
    print(f"Loaded {len(ub_df)} upper-bound configs from {ub_path}")

    daily_df, _, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    SIDS = ["S1","S2","S3","S4","S5","S6"]
    SID_MAP = {v:k for k,v in SNAMES.items()}

    results = {}
    for _, row in ub_df.iterrows():
        sname = row["strategy"]; ticker = row["ticker"]
        sid = SID_MAP.get(sname)
        if not sid: continue

        # Parse params from CSV
        sp_str = str(row.get("sig_params",""))
        fp_str = str(row.get("filt_params",""))

        # Reconstruct sig_params dict from compact string "key=val key=val"
        sig_params = {}
        int_sig_keys = {"w","vw","ap","el"}
        for part in sp_str.split():
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    v = float(v)
                    if k in int_sig_keys: v = int(v)
                except:
                    pass  # keep as string (ma_type, pt, dev_type)
                sig_params[k] = v

        # Reconstruct filt_params
        filt_params = {}
        int_keys = {"adx_th","atr_lo","mom_n","dow"}
        for part in fp_str.split():
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    v = float(v)
                    if k in int_keys: v = int(v)
                except:
                    try: v = int(v)
                    except: pass
                filt_params[k] = v

        # RM params from rm_* columns
        rm_params = {}
        for col in ub_df.columns:
            if col.startswith("rm_"):
                key = col[3:]  # strip "rm_"
                val = row[col]
                if pd.notna(val):
                    rm_params[key] = float(val) if isinstance(val, (int,float)) else val

        # Map RM keys to what run_bt expects
        rm = {
            "sl_p": rm_params.get("sl_p", 0),
            "tp_p": rm_params.get("tp_p", 0),
            "sl_a": rm_params.get("sl_a", 0),
            "tp_a": rm_params.get("tp_a", 0),
            "trail": rm_params.get("trail", 0),
            "mh": int(rm_params.get("mh", 0)),
            "be": rm_params.get("be", 0),
            "chand": rm_params.get("chand", 0),
            "rr": rm_params.get("rr", 0),
        }

        ub_exsh = row["ex_sharpe"]
        exsh, info = apply_fixed_params(sid, ticker, daily_df, kr_s, sig_params, filt_params, rm)
        results[(sid, ticker)] = (exsh, info, ub_exsh)

        if exsh > -900:
            flag = "✓" if exsh>0.5 else ("·" if exsh>0 else "✗")
            print(f"  {flag} {sname:>16s} {ticker:>6s}: "
                  f"WF_ExSh={exsh:+.2f}  UB_ExSh={ub_exsh:+.2f}  "
                  f"Ret={info.get('total_ret',0):+.1f}%  WR={info.get('wr',0):.0f}%  "
                  f"Exp={info.get('exposure',0):.0f}%")
        else:
            print(f"  ✗ {sname:>16s} {ticker:>6s}: no result (UB={ub_exsh:+.2f})")

    # Summary per strategy
    print(f"\n{'='*130}")
    print("SUMMARY: Walk-forward with fixed UB params vs Upper Bound")
    print(f"{'='*130}")
    print(f"  {'Strategy':>16s} {'WF_Mean':>8s} {'UB_Mean':>8s} {'Gap':>6s} "
          f"{'WF_N>0':>6s} {'WF_Best':>8s} {'WF_Worst':>8s}")

    for sid in SIDS:
        sname = SNAMES[sid]
        wf_vals = []; ub_vals = []
        for ticker in TICKERS:
            r = results.get((sid, ticker))
            if r:
                exsh, info, ub = r
                if exsh > -900:
                    wf_vals.append(exsh)
                else:
                    wf_vals.append(0)
                ub_vals.append(ub)
            else:
                wf_vals.append(0); ub_vals.append(0)

        wf_m = np.mean(wf_vals); ub_m = np.mean(ub_vals)
        wf_pos = sum(1 for v in wf_vals if v > 0)
        wf_best = max(wf_vals) if wf_vals else 0
        wf_worst = min(wf_vals) if wf_vals else 0
        gap = ub_m - wf_m
        print(f"  {sname:>16s} {wf_m:+8.2f} {ub_m:+8.2f} {gap:6.2f} "
              f"{wf_pos:6d} {wf_best:+8.2f} {wf_worst:+8.2f}")

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
