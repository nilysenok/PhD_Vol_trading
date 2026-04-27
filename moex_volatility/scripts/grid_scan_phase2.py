#!/usr/bin/env python3
"""
grid_scan_phase2.py — Phase 2: RM refinement with strategy-specific profiles.

- S3/S4 (trend): wide SL, trailing mandatory, long hold, chandelier
- S1/S2/S5/S6 (counter-trend): tight SL/TP, short hold, breakeven
- Pyramiding for all: max_pyr {0,1,2,3}, ratio {0.5,1.0}, cond profit_1atr
- Metric: calc_trading_sharpe (compound - RF, no FDR, vol ALL days)
"""
import os, sys, warnings, time, json, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

random.seed(42); np.random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, WARMUP, load_data, calc_atr,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)
from daily_upper_bound import bt_universal
from grid_scan_test import (
    precompute_indicators, gen_s1, gen_s2, gen_s3, gen_s4, gen_s5, gen_s6,
    apply_confirmation, apply_filters, run_bt, calc_trading_sharpe,
    MA_TYPES, WINDOWS_D, DC_WINDOWS, ST_PERIODS, ST_MULTS,
    PIVOT_TYPES, VWAP_WINDOWS, IS_TREND,
)

KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
BASE = Path(__file__).resolve().parent.parent
PHASE1_PATH = BASE / "results/final/strategies/walkforward_v4/grid_scan_daily_phase1.parquet"
PHASE2_PATH = BASE / "results/final/strategies/walkforward_v4/grid_scan_daily_phase2.parquet"

TOP_N_SIGNALS = 30
MAX_RESULTS_PER_TICKER = 2000
N_RM = 500


# ═══════════════════════════════════════════════════════════════
# STRATEGY-SPECIFIC RM GENERATORS
# ═══════════════════════════════════════════════════════════════

def random_rm_trend():
    """Trend RM: wide SL, trailing mandatory, long hold."""
    sl_a = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    trail = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    ch = random.choice([0, 0, 2.0, 3.0, 4.0, 5.0])  # sometimes chandelier
    mh = random.choice([0, 0, 20, 50, 100])  # long hold or unlimited
    be = random.choice([0, 0, 1.0, 1.5, 2.0])
    # If chandelier, use it as SL instead of fixed ATR
    if ch > 0:
        return dict(sl_p=0, tp_p=0, sl_a=0, tp_a=0, trail=trail,
                    mh=mh, be=be, chand=ch, rr=0)
    return dict(sl_p=0, tp_p=0, sl_a=sl_a, tp_a=0, trail=trail,
                mh=mh, be=be, chand=0, rr=0)


def random_rm_contrarian():
    """Counter-trend RM: tighter SL/TP, shorter hold."""
    r = random.random()
    if r < 0.15:
        sl_p = random.choice([0.01, 0.02, 0.03, 0.05, 0.07, 0.10]); sl_a = 0
    elif r < 0.6:
        sl_p = 0; sl_a = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    else:
        sl_p = 0; sl_a = 0  # chandelier
        ch = random.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        r2 = random.random()
        if r2 < 0.3:
            tp_a = random.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
            return dict(sl_p=0, tp_p=0, sl_a=0, tp_a=tp_a, trail=0,
                        mh=random.choice([0, 5, 10, 15, 20, 30]),
                        be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=ch, rr=0)
        else:
            rr = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
            return dict(sl_p=0, tp_p=0, sl_a=0, tp_a=0, trail=0,
                        mh=random.choice([0, 5, 10, 15, 20, 30]),
                        be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=ch, rr=rr)

    r2 = random.random()
    if r2 < 0.25:
        trail = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        return dict(sl_p=sl_p, tp_p=0, sl_a=sl_a, tp_a=0, trail=trail,
                    mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=0)
    elif r2 < 0.5:
        rr = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        return dict(sl_p=sl_p, tp_p=0, sl_a=sl_a, tp_a=0, trail=0,
                    mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=rr)
    elif r2 < 0.7:
        tp_p = random.choice([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15])
        return dict(sl_p=sl_p, tp_p=tp_p, sl_a=sl_a, tp_a=0, trail=0,
                    mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=0)
    else:
        tp_a = random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
        return dict(sl_p=sl_p, tp_p=0, sl_a=sl_a, tp_a=tp_a, trail=0,
                    mh=random.choice([0, 5, 10, 15, 20, 30]),
                    be=random.choice([0, 0.5, 1.0, 1.5, 2.0]), chand=0, rr=0)


def generate_rm_combos(sid, n_combos=500):
    """Generate strategy-specific RM combos."""
    is_trend = IS_TREND.get(sid, False)
    gen = random_rm_trend if is_trend else random_rm_contrarian
    seen = set(); combos = []
    for _ in range(n_combos * 3):  # oversample to deduplicate
        rm = gen()
        key = tuple(sorted(rm.items()))
        if key not in seen:
            seen.add(key); combos.append(rm)
        if len(combos) >= n_combos:
            break
    return combos


# ═══════════════════════════════════════════════════════════════
# SIGNAL REGENERATION
# ═══════════════════════════════════════════════════════════════

def regenerate_signal(sid, params_str, conf, close, high, low, volume, open_arr,
                      ind, dc_cache, st_cache, vwap_cache, pivot_data,
                      day_of_week, atr_arr):
    p = json.loads(params_str)
    rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
    is_trend = IS_TREND.get(sid, False)
    sig = None

    if sid == "S1":
        z = ind["z_atr"].get((p["ma"], p["w"]))
        if z is not None: sig = gen_s1(z, p["thresh"], WARMUP)
    elif sid == "S2":
        z = ind["z_std"].get((p["ma"], p["w"]))
        if z is not None: sig = gen_s2(z, p["bs"], WARMUP)
    elif sid == "S3":
        sig = gen_s3(close, dc_cache, p["w"], WARMUP)
    elif sid == "S4":
        sig = gen_s4(close, st_cache, p["ap"], p["mult"], WARMUP)
    elif sid == "S5":
        sig = gen_s5(close, pivot_data, p["pt"], p["el"], WARMUP)
    elif sid == "S6":
        sig = gen_s6(close, vwap_cache, p["vw"], p["dm"], WARMUP)

    if sig is None: return None
    sig = apply_confirmation(sig, conf, rsi, volume, vol_sma20, is_trend)
    return sig


# ═══════════════════════════════════════════════════════════════
# SELECT TOP SIGNALS FROM PHASE 1
# ═══════════════════════════════════════════════════════════════

def select_top_signals(df_p1):
    top = {}
    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        sub = df_p1[df_p1.sid == sid]
        if len(sub) == 0: top[sid] = []; continue
        # Group by (params, conf), require 2+ tickers with Sharpe > 0
        pos_sub = sub[sub.sharpe > 0]
        g = pos_sub.groupby(["params", "conf"]).agg(
            n_tk=("ticker", "nunique"),
            mean_sh=("sharpe", "mean"),
        ).reset_index()
        g = g[g.n_tk >= 2].sort_values("mean_sh", ascending=False).head(TOP_N_SIGNALS)
        top[sid] = list(zip(g.params, g.conf))
        print(f"  {sid}: {len(top[sid])} top signals (best mean={g.mean_sh.max():.3f})" if len(g) > 0
              else f"  {sid}: 0 top signals", flush=True)
    return top


# ═══════════════════════════════════════════════════════════════
# SCAN
# ═══════════════════════════════════════════════════════════════

def scan_ticker(ticker, daily_df, kr_s, top_signals):
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP + 200: return [], 0

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values); n = len(close)
    lr = np.zeros(n); lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    cal_days = np.ones(n)
    for i in range(1, n): cal_days[i] = (dates[i] - dates[i-1]).days
    rf_raw = kr_s.reindex(dates, method="ffill").values
    rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0

    oos_start = _year_bounds(dates, 2022)[0]
    if oos_start <= WARMUP: return [], 0
    oos_mask = np.zeros(n, dtype=bool); oos_mask[oos_start:] = True

    ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
    atr = ind["atr14"]; rh = ind["rh"]; rl = ind["rl"]
    day_of_week = dates.dayofweek.values

    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS)
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}

    comm = 0.0005
    results = []
    n_bt = 0

    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        signals_list = top_signals.get(sid, [])
        rm_combos = generate_rm_combos(sid, N_RM)

        for params_str, conf in signals_list:
            sig = regenerate_signal(sid, params_str, conf, close, high, low, volume, open_arr,
                                    ind, dc_cache, st_cache, vwap_cache, pivot_data,
                                    day_of_week, atr)
            if sig is None: continue
            if np.sum(np.abs(sig[oos_mask]) > 0) < 3: continue

            for rm in rm_combos:
                pos = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                sh, info = calc_trading_sharpe(pos, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                n_bt += 1

                if sh > -0.5:
                    rm_str = json.dumps({k: v for k, v in rm.items() if v != 0}, sort_keys=True)
                    results.append({
                        "sid": sid, "ticker": ticker, "params": params_str,
                        "conf": conf, "rm": rm_str,
                        "sharpe": round(sh, 3), "ret": info.get("total_ret", 0),
                        "vol": info.get("vol", 0), "exp": info.get("exposure", 0),
                        "wr": info.get("wr", 0), "tpy": info.get("tpy", 0),
                    })

    if len(results) > MAX_RESULTS_PER_TICKER:
        results.sort(key=lambda x: x["sharpe"], reverse=True)
        results = results[:MAX_RESULTS_PER_TICKER]

    return results, n_bt


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("GRID SCAN — Phase 2: Strategy-specific RM refinement", flush=True)
    print("Metric: calc_trading_sharpe (compound - RF, no FDR, vol ALL days)", flush=True)
    print("S3/S4: trend RM (wide SL, trailing, chandelier)", flush=True)
    print("S1/S2/S5/S6: contrarian RM (tight SL/TP, RR ratio)", flush=True)
    print("=" * 100, flush=True)

    df_p1 = pd.read_parquet(PHASE1_PATH)
    print(f"Phase 1: {len(df_p1)} rows", flush=True)

    print(f"\nSelecting top-{TOP_N_SIGNALS} signals per strategy:", flush=True)
    top_signals = select_top_signals(df_p1)

    daily_df, _, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    all_results = []
    total_bt = 0
    for i, ticker in enumerate(TICKERS):
        t1 = time.time()
        res, n_bt = scan_ticker(ticker, daily_df, kr_s, top_signals)
        elapsed = time.time() - t1
        all_results.extend(res)
        total_bt += n_bt
        n_pos = sum(1 for r in res if r["sharpe"] > 0)
        n_gt1 = sum(1 for r in res if r["sharpe"] > 1.0)
        best_sh = max((r["sharpe"] for r in res), default=-999)
        print(f"  [{i+1}/17] {ticker}: {len(res)} res, {n_bt:,} bt, "
              f"{n_pos} Sh>0, {n_gt1} Sh>1.0, best={best_sh:+.2f} ({elapsed:.1f}s)", flush=True)

    df = pd.DataFrame(all_results)
    PHASE2_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PHASE2_PATH)

    elapsed_total = time.time() - t0
    print(f"\n{'='*100}", flush=True)
    print(f"PHASE 2 SUMMARY — {len(df)} results, {total_bt:,} bt_calls", flush=True)
    print(f"{'='*100}", flush=True)

    # Per-strategy summary
    print(f"\n{'Strat':<6s} {'N':>6s} {'Mean':>7s} {'Max':>7s} {'>0':>5s} {'>1.0':>5s} | P1 max → P2 max", flush=True)
    print("-" * 70, flush=True)
    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        p1 = df_p1[df_p1.sid == sid]
        p2 = df[df.sid == sid] if len(df) > 0 else pd.DataFrame()
        p1_max = p1.sharpe.max() if len(p1) > 0 else 0
        if len(p2) > 0:
            sh = p2.sharpe
            p2_max = sh.max()
            delta = p2_max - p1_max
            print(f"  {sid:<4s} {len(p2):>6d} {sh.mean():>+7.2f} {p2_max:>+7.2f} "
                  f"{(sh>0).sum():>5d} {(sh>1.0).sum():>5d} | "
                  f"{p1_max:+.2f} → {p2_max:+.2f} (Δ={delta:+.2f})", flush=True)
        else:
            print(f"  {sid:<4s}      0       -       - | {p1_max:+.2f} → N/A", flush=True)

    # Top-5 per strategy
    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        sub = df[df.sid == sid].nlargest(5, "sharpe") if len(df) > 0 else pd.DataFrame()
        if len(sub) == 0: continue
        print(f"\n  Top-5 {sid}:", flush=True)
        for _, r in sub.iterrows():
            print(f"    Sh={r['sharpe']:+.2f}  Ret={r['ret']:+.1f}%  Vol={r['vol']:.0f}%  "
                  f"Exp={r['exp']:.0f}%  {r['ticker']:5s}  {r['params']}  RM={r['rm'][:60]}", flush=True)

    print(f"\nDONE in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)", flush=True)
    print(f"Saved: {PHASE2_PATH}", flush=True)


if __name__ == "__main__":
    main()
