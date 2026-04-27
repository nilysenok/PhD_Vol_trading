#!/usr/bin/env python3
"""
unified_sharpe.py — Compute ALL per-strategy A/B/C/D Sharpe ratios
using ONE formula (msp_correct style):
  1. Hourly returns → aggregate to daily (sum)
  2. FDR = free_capital × КС/365 × calendar_days (daily)
  3. Sharpe = (compound_ann_ret - compound_ann_rf) / (std(ALL daily) × √252)
Per-ticker, then mean across 17 tickers.
"""
import os, sys, warnings
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
    apply_filters_generic, get_ma, run_bt,
)
from hourly_upper_bound import (
    gen_s1 as h_gen_s1, gen_s2 as h_gen_s2, gen_s3 as h_gen_s3,
    gen_s4 as h_gen_s4, gen_s5 as h_gen_s5, gen_s6 as h_gen_s6,
    apply_filt as h_apply_filt, get_ma as h_get_ma, run_bt as h_run_bt,
)
from bootstrap_significance import parse_params, vol_scale_position
from msp_portfolio_opt import adaptive_sl_pos, vol_target_pos

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
SIDS = ["S1","S2","S3","S4","S5","S6"]
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP = {v:k for k,v in SNAMES.items()}


def unified_sharpe(daily_total_returns, daily_rf_returns):
    """THE ONE FORMULA used everywhere.
    Sharpe = (compound_ann_ret - compound_ann_rf) / (std(ALL daily) × √252)
    """
    n = len(daily_total_returns)
    if n < 30: return -999
    cum = np.prod(1.0 + daily_total_returns) - 1.0
    n_years = n / 252.0
    ann_ret = (1.0 + cum) ** (1.0 / n_years) - 1.0 if n_years > 0.1 else 0
    rf_cum = np.prod(1.0 + daily_rf_returns) - 1.0
    ann_rf = (1.0 + rf_cum) ** (1.0 / n_years) - 1.0 if n_years > 0.1 else 0
    vol = np.std(daily_total_returns, ddof=1) * sqrt(252)
    if vol < 1e-8: return -999
    return (ann_ret - ann_rf) / vol


def process_ticker(pos, lr, cal_factor, oos_mask, comm, is_hourly, dates):
    """Get daily total returns (trading + FDR) for one ticker."""
    pm = np.clip(pos[oos_mask], -1, 1)
    lr_m = lr[oos_mask]
    cf_m = cal_factor[oos_mask]

    # Trading returns
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr_m - np.abs(dp) * comm

    # FDR
    exp = np.abs(pm)
    fdr = (1.0 - exp) * cf_m

    total = net + fdr

    if is_hourly:
        # Aggregate hourly → daily by summing
        oos_dates = dates[oos_mask]
        day_labels = oos_dates.normalize()
        unique_days = day_labels.unique()
        daily_total = []
        daily_rf = []
        for d in unique_days:
            mask_d = day_labels == d
            daily_total.append(np.sum(total[mask_d]))
            daily_rf.append(np.sum(cf_m[mask_d]))
        return np.array(daily_total), np.array(daily_rf)
    else:
        return total, cf_m


def main():
    print("=" * 120)
    print("UNIFIED SHARPE — One formula for all")
    print("Sharpe = (compound_ann_ret - compound_ann_rf) / (std(ALL daily) × √252)")
    print("=" * 120)

    daily_df, hourly_df, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    d_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv")
    h_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")

    comms = {
        "DAILY": {"inst": 0.0005, "ret": 0.0006},
        "HOURLY": {"inst": 0.0004, "ret": 0.0005},
    }

    results = {}  # (tf, sid, approach, comm_level) → mean Sharpe

    for tf_label, ub_df, data_df, is_hourly in [
        ("DAILY", d_ub, daily_df, False),
        ("HOURLY", h_ub, hourly_df, True),
    ]:
        WU = WARMUP_HOURLY if is_hourly else WARMUP
        dt_col = "datetime" if is_hourly else "date"

        print(f"\n{'='*80}")
        print(f"  {tf_label}")
        print(f"{'='*80}")

        for _, row in ub_df.iterrows():
            sname = row["strategy"]; ticker = row["ticker"]
            sid = SID_MAP.get(sname)
            if not sid: continue

            tdf = data_df[data_df["ticker"]==ticker].sort_values(dt_col).reset_index(drop=True)
            if len(tdf) < WU + 200: continue
            close = tdf["close"].values.astype(np.float64)
            high = tdf["high"].values.astype(np.float64)
            low = tdf["low"].values.astype(np.float64)
            open_arr = tdf["open"].values.astype(np.float64)
            volume = tdf["volume"].values.astype(np.float64)
            dates = pd.to_datetime(tdf[dt_col].values); n = len(close)
            lr = np.zeros(n); lr[:-1] = np.log(close[1:]/np.maximum(close[:-1], 1e-12))
            atr = calc_atr(high, low, close, 14)
            ind = compute_base(close, high, low, open_arr, volume, is_hourly=is_hourly)
            rh_w = 99 if is_hourly else 11
            rh = pd.Series(high).rolling(rh_w, min_periods=1).max().values
            rl = pd.Series(low).rolling(rh_w, min_periods=1).min().values

            # Calendar factor for FDR (daily granularity)
            if is_hourly:
                h_dates = dates.normalize()
                cal_factor = np.zeros(n)
                rf_raw = kr_s.reindex(h_dates, method="ffill").values
                rf_vals = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0
                prev_d = None
                for i in range(n):
                    d = h_dates[i]
                    if d != prev_d:
                        cd = 1
                        if prev_d is not None: cd = (d - prev_d).days
                        cal_factor[i] = rf_vals[i] * cd
                        prev_d = d
            else:
                cal_days = np.ones(n)
                for i in range(1, n): cal_days[i] = (dates[i] - dates[i-1]).days
                rf_raw = kr_s.reindex(dates, method="ffill").values
                rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0
                cal_factor = rf_d * cal_days

            # OOS mask
            if is_hourly:
                oos_idx = 0
                for i in range(n):
                    if dates[i].year >= 2022: oos_idx = i; break
                if oos_idx <= WU: continue
            else:
                oos_idx = _year_bounds(dates, 2022)[0]
                if oos_idx <= WU: continue
            oos_mask = np.zeros(n, dtype=bool); oos_mask[oos_idx:] = True

            # Reconstruct A signal
            sp, fp = parse_params(row.get("sig_params", ""), row.get("filt_params", ""))
            sig = None
            if not is_hourly:
                if sid in ("S1","S2"):
                    mt=sp.get("ma_type","sma"); w=sp.get("w",20)
                    try:
                        ma=get_ma(close,w,mt); std=calc_std(close,w)
                        if sid=="S1": sp["atr_arr"]=atr; sig=gen_sig_s1_ext(close,std,ma,sp)
                        else: sig=gen_sig_s2_ext(close,std,ma,sp)
                    except: pass
                elif sid=="S3":
                    w=sp.get("w",20); sig=gen_sig_s3(close,high,low,precompute_donchian_cache(high,low,[w]),sp)
                elif sid=="S4":
                    ap=sp.get("ap",14); mult=sp.get("mult",3.0)
                    sig=gen_sig_s4(close,high,low,precompute_supertrend_cache(high,low,close,[ap],[mult]),sp)
                elif sid=="S5":
                    pt=sp.get("pt","fibonacci"); sig=gen_sig_s5(close,high,low,{pt:calc_pivot_daily(high,low,close,pt)},sp)
                elif sid=="S6":
                    vw=sp.get("vw",20); sig=gen_sig_s6(close,high,low,volume,precompute_vwap_cache(close,high,low,volume,[vw]),sp)
            else:
                if sid in ("S1","S2"):
                    mt=sp.get("ma_type","sma"); w=sp.get("w",90)
                    try:
                        ma=h_get_ma(close,w,mt); std=calc_std(close,w); mc={(mt,w):(ma,std)}
                        if sid=="S1": sig=h_gen_s1(close,mc,atr,sp)
                        else: sig=h_gen_s2(close,mc,sp)
                    except: pass
                elif sid=="S3":
                    w=sp.get("w",90); sig=h_gen_s3(close,high,low,precompute_donchian_cache(high,low,[w]),sp)
                elif sid=="S4":
                    ap=sp.get("ap",90); mult=sp.get("mult",3.0)
                    sig=h_gen_s4(close,high,low,precompute_supertrend_cache(high,low,close,[ap],[mult]),sp)
                elif sid=="S5":
                    pt=sp.get("pt","fibonacci")
                    sig=h_gen_s5(close,{pt:compute_daily_pivots_for_hourly(daily_df,ticker,dates,pt)},sp)
                elif sid=="S6":
                    vw=sp.get("vw",90); sig=h_gen_s6(close,high,low,volume,precompute_vwap_cache(close,high,low,volume,[vw]),sp)

            if sig is None:
                sw = 90 if is_hourly else 20
                sma_v=calc_sma(close,sw); std_v=calc_std(close,sw)
                z=(close-sma_v)/np.where(std_v>1e-12,std_v,1.0)
                sig=np.zeros(n,dtype=np.int8); sig[~np.isnan(z)&(z<-2)]=1; sig[~np.isnan(z)&(z>2)]=-1
                sig[:WU]=0

            rsi = calc_rsi(close, 14)
            if not is_hourly:
                dow = dates.dayofweek.values.astype(np.int32)
                sig = apply_filters_generic(sig, close, volume, ind, rsi, atr, dow, fp, IS_TREND[sid])
            else:
                sig = h_apply_filt(sig, close, volume, ind, rsi, atr, fp, IS_TREND[sid])

            rm = {k: float(row.get(f"rm_{k}", 0) or 0) for k in
                  ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
            rm["mh"] = int(float(row.get("rm_mh", 0) or 0))

            if is_hourly:
                pos_a = h_run_bt(sig, close, high, low, atr, rh, rl, rm, WU, n)
            else:
                pos_a = run_bt(sig, close, high, low, atr, rh, rl, rm, WU, n)

            # B/C/D positions
            vw_b = 90 if is_hourly else 20
            pos_b = adaptive_sl_pos(pos_a, lr, vw_b)
            pos_c = vol_scale_position(pos_a, lr, vw_b)
            pos_d = vol_target_pos(pos_a, lr, vw_b, 0.15)

            # Compute Sharpe for each approach × commission
            for app, pos in [("A", pos_a), ("B", pos_b), ("C", pos_c), ("D", pos_d)]:
                for cl, comm in comms[tf_label].items():
                    daily_tot, daily_rf = process_ticker(pos, lr, cal_factor, oos_mask, comm, is_hourly, dates)
                    sh = unified_sharpe(daily_tot, daily_rf)
                    key = (tf_label, sid, app, cl, ticker)
                    results[key] = sh

    # ── AGGREGATE: per-ticker mean ──
    print(f"\n{'='*120}")
    print("СВОДНАЯ ТАБЛИЦА — Per-ticker mean Sharpe (единая формула)")
    print(f"{'='*120}")

    for tf in ["DAILY", "HOURLY"]:
        for cl, cl_label in [("inst", "Инст."), ("ret", "Розн.")]:
            print(f"\n  {tf} ({cl_label}):")
            print(f"  {'Стратегия':>16s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'B-A':>6s} {'C-A':>6s} {'D-A':>6s}")
            means = {}
            for sid in SIDS:
                row_vals = {}
                for app in ["A","B","C","D"]:
                    ticker_sharpes = []
                    for ticker in TICKERS:
                        key = (tf, sid, app, cl, ticker)
                        if key in results and results[key] > -900:
                            ticker_sharpes.append(results[key])
                    if ticker_sharpes:
                        row_vals[app] = np.mean(ticker_sharpes)
                    else:
                        row_vals[app] = 0
                means[sid] = row_vals
                a = row_vals["A"]; b = row_vals["B"]; c = row_vals["C"]; d = row_vals["D"]
                print(f"  {SNAMES[sid]:>16s} {a:+7.2f} {b:+7.2f} {c:+7.2f} {d:+7.2f} "
                      f"{b-a:+6.2f} {c-a:+6.2f} {d-a:+6.2f}")

            # EW average
            ew = {app: np.mean([means[s][app] for s in SIDS]) for app in ["A","B","C","D"]}
            print(f"  {'EW среднее':>16s} {ew['A']:+7.2f} {ew['B']:+7.2f} {ew['C']:+7.2f} {ew['D']:+7.2f} "
                  f"{ew['B']-ew['A']:+6.2f} {ew['C']-ew['A']:+6.2f} {ew['D']-ew['A']:+6.2f}")

    print(f"\nDONE")


if __name__ == "__main__":
    main()
