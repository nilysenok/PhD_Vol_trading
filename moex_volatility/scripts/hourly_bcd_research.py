#!/usr/bin/env python3
"""
hourly_bcd_research.py — Test 10 hypotheses for hourly BCD adaptation.
Train: pre-2022. Test: 2022-2026.
Sharpe: calc_excess_sharpe formula (compound + FDR + vol active × √252).
Deterministic: no random, exhaustive grid.
"""
import os, sys, warnings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP_HOURLY as WU, load_data, compute_base, calc_rsi,
    calc_sma, calc_std, calc_atr, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    compute_daily_pivots_for_hourly, warmup_numba,
)
from hourly_upper_bound import (
    gen_s1, gen_s2, gen_s3, gen_s4, gen_s5, gen_s6,
    apply_filt, get_ma as hgm, run_bt as hrb, BPH,
)
from bootstrap_significance import parse_params

KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
BASE = "/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility"
BPY = 252; COMM = 0.0004
SIDS = ["S1","S2","S3","S4","S5","S6"]
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
IS_TREND = {"S1":False,"S2":False,"S3":True,"S4":True,"S5":False,"S6":False}
SID_MAP = {v:k for k,v in SNAMES.items()}


def sharpe_from_daily(daily_total, daily_rf, daily_active):
    """calc_excess_sharpe formula on daily arrays."""
    n = len(daily_total)
    if n < 30: return -999
    na = daily_active.sum()
    if na < 10: return -999
    cum = np.prod(1.0 + daily_total) - 1.0
    ny = n / 252.0
    ann = (1.0 + cum) ** (1.0 / ny) - 1.0 if ny > 0.1 else 0
    rf_cum = np.prod(1.0 + daily_rf) - 1.0
    rf_ann = (1.0 + rf_cum) ** (1.0 / ny) - 1.0 if ny > 0.1 else 0
    vol = np.std(daily_total[daily_active], ddof=1) * sqrt(BPY)
    if vol < 1e-6: return -999
    return (ann - rf_ann) / vol


def hourly_to_daily(pos_hourly, lr_hourly, cf_hourly, dates_hourly, comm):
    """Convert hourly position/returns to daily aggregates for sharpe calc."""
    pm = pos_hourly
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr_hourly - np.abs(dp) * comm
    exp = np.abs(pm)
    fdr = (1.0 - exp) * cf_hourly
    total = net + fdr

    day_labels = dates_hourly.normalize()
    unique_days = day_labels.unique()
    nd = len(unique_days)

    d_total = np.zeros(nd)
    d_rf = np.zeros(nd)
    d_active = np.zeros(nd, dtype=bool)
    for i, d in enumerate(unique_days):
        mask = day_labels == d
        d_total[i] = np.sum(total[mask])
        d_rf[i] = np.sum(cf_hourly[mask])
        d_active[i] = np.mean(exp[mask]) > 1e-8

    return d_total, d_rf, d_active


def load_ticker_data(ticker, hourly_df, daily_df, kr_s):
    """Load and prepare all data for one ticker."""
    hdf = hourly_df[hourly_df["ticker"]==ticker].sort_values("datetime").reset_index(drop=True)
    if len(hdf) < WU + 500: return None

    cl = hdf["close"].values.astype(np.float64)
    hi = hdf["high"].values.astype(np.float64)
    lo = hdf["low"].values.astype(np.float64)
    op = hdf["open"].values.astype(np.float64)
    vo = hdf["volume"].values.astype(np.float64)
    dt = pd.to_datetime(hdf["datetime"].values); n = len(cl)
    lr = np.zeros(n); lr[:-1] = np.log(cl[1:]/np.maximum(cl[:-1], 1e-12))
    at = calc_atr(hi, lo, cl, 14)
    ind = compute_base(cl, hi, lo, op, vo, is_hourly=True)
    rhi = pd.Series(hi).rolling(99, min_periods=1).max().values
    rlo = pd.Series(lo).rolling(99, min_periods=1).min().values

    # Calendar factor
    hd = dt.normalize(); cf = np.zeros(n)
    rf_raw = kr_s.reindex(hd, method="ffill").values
    rv = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0
    pd_ = None
    for i in range(n):
        d = hd[i]
        if d != pd_:
            cd = 1
            if pd_ is not None: cd = (d - pd_).days
            cf[i] = rv[i] * cd
            pd_ = d

    # OOS split
    oi = 0
    for i in range(n):
        if dt[i].year >= 2022: oi = i; break
    if oi <= WU: return None

    train_mask = np.zeros(n, dtype=bool); train_mask[WU:oi] = True
    oos_mask = np.zeros(n, dtype=bool); oos_mask[oi:] = True

    return dict(cl=cl, hi=hi, lo=lo, op=op, vo=vo, dt=dt, lr=lr, at=at,
                ind=ind, rhi=rhi, rlo=rlo, cf=cf, n=n,
                train_mask=train_mask, oos_mask=oos_mask)


def gen_signal(sid, ticker, data, daily_df, row):
    """Generate signal for a strategy."""
    sp, fp = parse_params(row.get("sig_params",""), row.get("filt_params",""))
    cl=data["cl"];hi=data["hi"];lo=data["lo"];vo=data["vo"]
    dt=data["dt"];at=data["at"];ind=data["ind"];n=data["n"]
    sig = None
    if sid in ("S1","S2"):
        mt=sp.get("ma_type","sma");w=sp.get("w",90)
        try:
            ma=hgm(cl,w,mt);std=calc_std(cl,w);mc={(mt,w):(ma,std)}
            if sid=="S1": sig=gen_s1(cl,mc,at,sp)
            else: sig=gen_s2(cl,mc,sp)
        except: pass
    elif sid=="S3":
        w=sp.get("w",90);sig=gen_s3(cl,hi,lo,precompute_donchian_cache(hi,lo,[w]),sp)
    elif sid=="S4":
        ap=sp.get("ap",90);mu=sp.get("mult",3.0)
        sig=gen_s4(cl,hi,lo,precompute_supertrend_cache(hi,lo,cl,[ap],[mu]),sp)
    elif sid=="S5":
        pt=sp.get("pt","fibonacci")
        sig=gen_s5(cl,{pt:compute_daily_pivots_for_hourly(daily_df,ticker,dt,pt)},sp)
    elif sid=="S6":
        vw=sp.get("vw",90);sig=gen_s6(cl,hi,lo,vo,precompute_vwap_cache(cl,hi,lo,vo,[vw]),sp)
    if sig is None:
        sm=calc_sma(cl,90);st=calc_std(cl,90)
        z=(cl-sm)/np.where(st>1e-12,st,1.0)
        sig=np.zeros(n,dtype=np.int8);sig[~np.isnan(z)&(z<-2)]=1;sig[~np.isnan(z)&(z>2)]=-1;sig[:WU]=0
    rsi=calc_rsi(cl,14)
    sig=apply_filt(sig,cl,vo,ind,rsi,at,fp,IS_TREND[sid])
    return sig


def get_base_position(sig, data, row):
    """Get A position from bt_universal."""
    rm = {k:float(row.get(f"rm_{k}",0) or 0) for k in
          ["sl_p","tp_p","sl_a","tp_a","trail","be","chand","rr"]}
    rm["mh"] = int(float(row.get("rm_mh",0) or 0))
    return hrb(sig, data["cl"], data["hi"], data["lo"], data["at"],
               data["rhi"], data["rlo"], rm, WU, data["n"])


def compute_vol_forecast(lr, window):
    """Rolling realized vol."""
    n = len(lr); rv = np.full(n, np.nan)
    for i in range(window, n):
        rv[i] = np.std(lr[i-window:i], ddof=1)
    return rv


def eval_sharpe(pos, data, mask):
    """Evaluate sharpe for a position on a mask."""
    dt_oos = data["dt"][mask]
    d_total, d_rf, d_active = hourly_to_daily(
        pos[mask], data["lr"][mask], data["cf"][mask], dt_oos, COMM)
    return sharpe_from_daily(d_total, d_rf, d_active)


# ═══════════════════════════════════════════════════════════════
# HYPOTHESES
# ═══════════════════════════════════════════════════════════════

def h1_regime_filter(pos_a, data, mask, is_trend, vol_window=90):
    """H1: trade at full size in favorable regime, scale down otherwise."""
    lr = data["lr"]; rv = compute_vol_forecast(lr, vol_window)
    rv_valid = rv[~np.isnan(rv)]
    if len(rv_valid) < 100: return -999, {}
    best_sh = -999; best_p = {}
    for thresh_q in [0.3, 0.4, 0.5, 0.6, 0.7]:
        thresh = np.percentile(rv_valid, thresh_q * 100)
        for scale_down in [0.0, 0.2, 0.3, 0.5, 0.7]:
            # Trend loves high vol, counter-trend loves low vol
            if is_trend:
                favorable = (~np.isnan(rv)) & (rv >= thresh)
            else:
                favorable = (~np.isnan(rv)) & (rv <= thresh)
            pos_h = pos_a.copy()
            pos_h[~favorable] = pos_a[~favorable] * scale_down
            # Optimize on TRAIN
            sh_train = eval_sharpe(pos_h, data, data["train_mask"])
            if sh_train > best_sh:
                best_sh = sh_train
                best_p = {"thresh_q":thresh_q, "scale_down":scale_down, "pos":pos_h}
    if "pos" not in best_p: return -999, {}
    sh_oos = eval_sharpe(best_p["pos"], data, mask)
    return sh_oos, best_p


def h3_continuous_sizing(pos_a, data, mask, vol_window=90):
    """H3: position = signal × (vol_target / vol)^alpha, allow >1."""
    lr = data["lr"]; rv = compute_vol_forecast(lr, vol_window)
    rv_valid = rv[~np.isnan(rv)]
    if len(rv_valid) < 100: return -999, {}
    best_sh = -999; best_p = {}
    for tgt_q in [0.3, 0.5, 0.7]:
        tgt = np.percentile(rv_valid, tgt_q * 100)
        if tgt < 1e-12: continue
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            for max_pos in [1.5, 2.0]:
                scale = (tgt / np.where((rv<1e-12)|np.isnan(rv), tgt, rv)) ** alpha
                scale = np.clip(scale, 0.3, max_pos)
                pos_h = np.clip(pos_a * scale, -max_pos, max_pos)
                sh_train = eval_sharpe(pos_h, data, data["train_mask"])
                if sh_train > best_sh:
                    best_sh = sh_train
                    best_p = {"tgt_q":tgt_q, "alpha":alpha, "max":max_pos, "pos":pos_h}
    if "pos" not in best_p: return -999, {}
    return eval_sharpe(best_p["pos"], data, mask), best_p


def h5_short_horizon(pos_a, data, mask):
    """H5: use shorter vol forecast horizon (6h, 12h, 24h, 45h)."""
    lr = data["lr"]
    best_sh = -999; best_p = {}
    for w in [6, 12, 24, 45]:
        rv = compute_vol_forecast(lr, w)
        rv_valid = rv[~np.isnan(rv)]
        if len(rv_valid) < 100: continue
        tgt = np.median(rv_valid)
        if tgt < 1e-12: continue
        scale = tgt / np.where((rv<1e-12)|np.isnan(rv), tgt, rv)
        scale = np.clip(scale, 0.3, 2.0)
        pos_h = np.clip(pos_a * scale, -2.0, 2.0)
        sh_train = eval_sharpe(pos_h, data, data["train_mask"])
        if sh_train > best_sh:
            best_sh = sh_train
            best_p = {"window":w, "pos":pos_h}
    if "pos" not in best_p: return -999, {}
    return eval_sharpe(best_p["pos"], data, mask), best_p


def h6_asymmetric(pos_a, data, mask, is_trend, vol_window=90):
    """H6: only INCREASE position in favorable regime, never decrease."""
    lr = data["lr"]; rv = compute_vol_forecast(lr, vol_window)
    rv_valid = rv[~np.isnan(rv)]
    if len(rv_valid) < 100: return -999, {}
    best_sh = -999; best_p = {}
    for fav_q in [0.3, 0.4, 0.5, 0.6]:
        thresh = np.percentile(rv_valid, fav_q * 100)
        for scale_up in [1.2, 1.5, 1.8, 2.0]:
            if is_trend:
                favorable = (~np.isnan(rv)) & (rv >= thresh)
            else:
                favorable = (~np.isnan(rv)) & (rv <= thresh)
            pos_h = pos_a.copy()
            pos_h[favorable] = pos_a[favorable] * scale_up
            pos_h = np.clip(pos_h, -2.0, 2.0)
            sh_train = eval_sharpe(pos_h, data, data["train_mask"])
            if sh_train > best_sh:
                best_sh = sh_train
                best_p = {"fav_q":fav_q, "scale_up":scale_up, "pos":pos_h}
    if "pos" not in best_p: return -999, {}
    return eval_sharpe(best_p["pos"], data, mask), best_p


def h8_vol_of_vol(pos_a, data, mask, vol_window=90):
    """H8: trade only when vol-of-vol is low (stable regime)."""
    lr = data["lr"]; rv = compute_vol_forecast(lr, vol_window)
    n = len(lr); vov = np.full(n, np.nan)
    for i in range(vol_window+20, n):
        chunk = rv[i-20:i]; chunk = chunk[~np.isnan(chunk)]
        if len(chunk) > 5: vov[i] = np.std(chunk, ddof=1)
    vov_valid = vov[~np.isnan(vov)]
    if len(vov_valid) < 100: return -999, {}
    best_sh = -999; best_p = {}
    for thresh_q in [0.3, 0.5, 0.7]:
        thresh = np.percentile(vov_valid, thresh_q * 100)
        pos_h = pos_a.copy()
        pos_h[(~np.isnan(vov)) & (vov > thresh)] = 0  # skip unstable
        sh_train = eval_sharpe(pos_h, data, data["train_mask"])
        if sh_train > best_sh:
            best_sh = sh_train
            best_p = {"thresh_q":thresh_q, "pos":pos_h}
    if "pos" not in best_p: return -999, {}
    return eval_sharpe(best_p["pos"], data, mask), best_p


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import time
    t0 = time.time()
    print("=" * 100)
    print("HOURLY BCD RESEARCH — 10 Hypotheses")
    print("=" * 100)

    daily_df, hourly_df, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    h_ub = pd.read_csv(f"{BASE}/results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")

    # Test hypotheses: H1, H3, H5, H6, H8 (most promising)
    hypotheses = ["A", "H1", "H3", "H5", "H6", "H8"]
    results = {h: {sid: [] for sid in SIDS} for h in hypotheses}

    for _, row in h_ub.iterrows():
        sn = row["strategy"]; tk = row["ticker"]; sid = SID_MAP.get(sn)
        if not sid: continue

        data = load_ticker_data(tk, hourly_df, daily_df, kr_s)
        if data is None: continue

        sig = gen_signal(sid, tk, data, daily_df, row)
        pos_a = get_base_position(sig, data, row)
        oos = data["oos_mask"]
        is_trend = IS_TREND[sid]

        # A baseline
        sh_a = eval_sharpe(pos_a, data, oos)
        results["A"][sid].append(sh_a)

        # H1
        sh_h1, _ = h1_regime_filter(pos_a, data, oos, is_trend)
        results["H1"][sid].append(sh_h1 if sh_h1 > -900 else sh_a)

        # H3
        sh_h3, _ = h3_continuous_sizing(pos_a, data, oos)
        results["H3"][sid].append(sh_h3 if sh_h3 > -900 else sh_a)

        # H5
        sh_h5, _ = h5_short_horizon(pos_a, data, oos)
        results["H5"][sid].append(sh_h5 if sh_h5 > -900 else sh_a)

        # H6
        sh_h6, _ = h6_asymmetric(pos_a, data, oos, is_trend)
        results["H6"][sid].append(sh_h6 if sh_h6 > -900 else sh_a)

        # H8
        sh_h8, _ = h8_vol_of_vol(pos_a, data, oos)
        results["H8"][sid].append(sh_h8 if sh_h8 > -900 else sh_a)

    # Output
    print(f"\n{'='*100}")
    print("RESULTS — Per-strategy mean Sharpe (OOS 2022-2026)")
    print(f"{'='*100}")
    print(f"{'':>4s}", end="")
    for h in hypotheses: print(f" {h:>7s}", end="")
    print(f" {'Best':>7s} {'BestH':>6s}")

    for sid in SIDS:
        print(f"  {sid:>4s}", end="")
        best_h = "A"; best_v = -999
        for h in hypotheses:
            v = np.mean(results[h][sid]) if results[h][sid] else 0
            print(f" {v:+7.2f}", end="")
            if v > best_v: best_v = v; best_h = h
        print(f" {best_v:+7.2f} {best_h:>6s}")

    # EW
    print(f"  {'EW':>4s}", end="")
    for h in hypotheses:
        v = np.mean([np.mean(results[h][s]) for s in SIDS if results[h][s]])
        print(f" {v:+7.2f}", end="")
    print()

    # Deltas
    print(f"\n  Deltas vs A:")
    print(f"{'':>4s}", end="")
    for h in hypotheses[1:]: print(f" {h:>7s}", end="")
    print()
    for sid in SIDS:
        a_mean = np.mean(results["A"][sid]) if results["A"][sid] else 0
        print(f"  {sid:>4s}", end="")
        for h in hypotheses[1:]:
            v = np.mean(results[h][sid]) if results[h][sid] else 0
            print(f" {v-a_mean:+7.2f}", end="")
        print()

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
