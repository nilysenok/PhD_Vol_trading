#!/usr/bin/env python3
"""
grid_scan_phase3.py — Phase 3: Extended signal strategies.

Additions vs Phase 1:
  - S3 Donchian: 6 breakout confirmations (close_confirm, two_close, volume, adx, momentum)
  - S5 PivotPoints: DeMark type, ATR distance filter
  - S1+S2 hybrid: Mean Reversion with Bollinger squeeze filter
  - S3+S4 hybrid: Donchian breakout confirmed by Supertrend
  - S7: RSI Divergence
  - S8: MA Crossover

Architecture: same as Phase 1 (precompute → threshold → bt → sharpe)
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
    TICKERS, WARMUP, load_data, calc_rsi, calc_sma, calc_std, calc_atr, calc_adx,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)
from daily_upper_bound import bt_universal, calc_excess_sharpe
from grid_scan_test import (
    precompute_indicators, run_bt, random_rm_params, apply_confirmation,
    apply_filters, random_filter_params,
    MA_TYPES, WINDOWS_D, DC_WINDOWS, ST_PERIODS, ST_MULTS,
    PIVOT_TYPES, VWAP_WINDOWS, get_ma,
)

KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
BASE = Path(__file__).resolve().parent.parent
BPY = 252

# Thresholds
T1_THRESHOLD = -0.1
T2_THRESHOLD = 0.2
N_RM = 50
MAX_RESULTS_PER_TICKER = 1000

DEFAULT_RM = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=3.0, trail=0, mh=15, be=0, chand=0, rr=0)
DEFAULT_RM_TREND = dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=0, trail=2.0, mh=0, be=1.0, chand=0, rr=0)


# ═══════════════════════════════════════════════════════════════
# S3 EXTENDED: Donchian with breakout confirmations
# ═══════════════════════════════════════════════════════════════

DC_CONFIRMATIONS = ["none", "close_confirm", "two_close", "volume", "adx_above_20", "momentum"]

def gen_s3_extended(close, high, low, dc_cache, w, conf_type, volume, adx, vol_sma20, warmup):
    """S3 Donchian with breakout confirmations."""
    if w not in dc_cache: return None
    hc, lc = dc_cache[w]
    n = len(close)
    # Previous period high/low
    ph = np.empty(n); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty(n); pl[0] = np.nan; pl[1:] = lc[:-1]
    v = ~np.isnan(ph)

    sig = np.zeros(n, dtype=np.int8)

    if conf_type == "none":
        sig[v & (close > ph)] = 1
        sig[v & (close < pl)] = -1

    elif conf_type == "close_confirm":
        # Close above channel high (not just intraday touch)
        sig[v & (close > ph)] = 1
        sig[v & (close < pl)] = -1

    elif conf_type == "two_close":
        # Two consecutive closes above/below
        prev_above = np.zeros(n, dtype=bool)
        prev_below = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(ph[i]):
                prev_above[i] = close[i-1] > ph[i-1] if not np.isnan(ph[i-1]) else False
                prev_below[i] = close[i-1] < pl[i-1] if not np.isnan(pl[i-1]) else False
        sig[v & (close > ph) & prev_above] = 1
        sig[v & (close < pl) & prev_below] = -1

    elif conf_type == "volume":
        vs = np.where(np.isnan(vol_sma20) | (vol_sma20 < 1), 1.0, vol_sma20)
        vol_ok = volume / vs > 1.0
        sig[v & (close > ph) & vol_ok] = 1
        sig[v & (close < pl) & vol_ok] = -1

    elif conf_type == "adx_above_20":
        adx_ok = (~np.isnan(adx)) & (adx > 20)
        sig[v & (close > ph) & adx_ok] = 1
        sig[v & (close < pl) & adx_ok] = -1

    elif conf_type == "momentum":
        mom = np.zeros(n)
        mom[5:] = close[5:] - close[:-5]
        sig[v & (close > ph) & (mom > 0)] = 1
        sig[v & (close < pl) & (mom < 0)] = -1

    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# S5 EXTENDED: PivotPoints with DeMark + ATR filter
# ═══════════════════════════════════════════════════════════════

PIVOT_ATR_FILTERS = [0, 0.5, 1.0, 1.5, 2.0]

def calc_pivot_demark(high, low, close, open_arr):
    """DeMark pivot points."""
    n = len(close)
    P = np.full(n, np.nan)
    S1 = np.full(n, np.nan); R1 = np.full(n, np.nan)
    S2 = np.full(n, np.nan); R2 = np.full(n, np.nan)
    S3 = np.full(n, np.nan); R3 = np.full(n, np.nan)

    for i in range(1, n):
        h, l, c, o = high[i-1], low[i-1], close[i-1], open_arr[i-1]
        if c < o:
            x = h + 2*l + c
        elif c > o:
            x = 2*h + l + c
        else:
            x = h + l + 2*c
        P[i] = x / 4.0
        S1[i] = x / 2.0 - h
        R1[i] = x / 2.0 - l
        # S2/R2/S3/R3: extend using range
        rng = h - l
        S2[i] = P[i] - rng
        R2[i] = P[i] + rng
        S3[i] = P[i] - 2*rng
        R3[i] = P[i] + 2*rng

    return P, S1, R1, S2, R2, S3, R3


def gen_s5_extended(close, pivot_data, pt, el, atr_arr, atr_filter, warmup):
    """S5 PivotPoints with ATR distance filter."""
    if pt not in pivot_data: return None
    P, S1r, R1r, S2r, R2r, S3r, R3r = pivot_data[pt]
    if el == 2: S, R = S2r, R2r
    elif el == 3: S, R = S3r, R3r
    else: S, R = S1r, R1r

    n = len(close)
    sig = np.zeros(n, dtype=np.int8)

    if atr_filter > 0:
        # Only signal when distance to pivot > atr_filter × ATR
        dist_s = np.abs(close - S)
        dist_r = np.abs(close - R)
        atr_ok = (~np.isnan(atr_arr)) & (atr_arr > 1e-12)
        s_ok = atr_ok & (~np.isnan(S)) & (dist_s > atr_filter * atr_arr)
        r_ok = atr_ok & (~np.isnan(R)) & (dist_r > atr_filter * atr_arr)
        sig[s_ok & (close < S)] = 1
        sig[r_ok & (close > R)] = -1
    else:
        sig[~np.isnan(S) & (close < S)] = 1
        sig[~np.isnan(R) & (close > R)] = -1

    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# HYBRID S1+S2: Mean Reversion + Bollinger Squeeze
# ═══════════════════════════════════════════════════════════════

SQUEEZE_PCTILES = [10, 15, 20, 25, 30]

def gen_s1s2_hybrid(close, z_atr, std_arr, ma_arr, squeeze_pctile, threshold, warmup):
    """S1 mean reversion, but only when BB are squeezed (low vol)."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)

    # Bollinger bandwidth = std / ma
    bandwidth = std_arr / np.where((np.abs(ma_arr) > 1e-12) & ~np.isnan(ma_arr), np.abs(ma_arr), 1.0)
    bandwidth[np.isnan(ma_arr)] = np.nan

    # Compute rolling percentile of bandwidth
    bw_valid = bandwidth[~np.isnan(bandwidth)]
    if len(bw_valid) < 50: return None
    thresh_bw = np.percentile(bw_valid, squeeze_pctile)

    # Signal: z-score beyond threshold AND BB squeezed
    squeeze = (~np.isnan(bandwidth)) & (bandwidth < thresh_bw)
    v = ~np.isnan(z_atr)
    sig[v & squeeze & (z_atr < -threshold)] = 1
    sig[v & squeeze & (z_atr > threshold)] = -1
    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# HYBRID S3+S4: Donchian Breakout + Supertrend Confirmation
# ═══════════════════════════════════════════════════════════════

def gen_s3s4_hybrid(close, dc_cache, st_cache, dc_w, st_ap, st_mult, warmup):
    """Donchian breakout confirmed by Supertrend direction."""
    if dc_w not in dc_cache: return None
    st_key = (st_ap, st_mult)
    if st_key not in st_cache: return None

    hc, lc = dc_cache[dc_w]
    n = len(close)
    ph = np.empty(n); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty(n); pl[0] = np.nan; pl[1:] = lc[:-1]

    st, direction = st_cache[st_key]
    dc_v = ~np.isnan(ph)
    st_v = ~np.isnan(st)

    sig = np.zeros(n, dtype=np.int8)
    # Breakout up + Supertrend bullish (direction = +1)
    sig[dc_v & st_v & (close > ph) & (direction > 0)] = 1
    # Breakout down + Supertrend bearish
    sig[dc_v & st_v & (close < pl) & (direction < 0)] = -1
    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# S7: RSI Divergence
# ═══════════════════════════════════════════════════════════════

def gen_s7_rsi_div(close, rsi, lookback, warmup):
    """RSI divergence: price makes new low but RSI doesn't (bullish), and vice versa."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)

    for i in range(lookback + warmup, n):
        if np.isnan(rsi[i]): continue
        # Find local min in price over lookback
        window = slice(i - lookback, i)
        p_min_idx = np.argmin(close[window]) + (i - lookback)
        p_max_idx = np.argmax(close[window]) + (i - lookback)

        # Bullish divergence: current price < previous low, but RSI > previous RSI low
        if close[i] < close[p_min_idx] and not np.isnan(rsi[p_min_idx]):
            if rsi[i] > rsi[p_min_idx]:
                sig[i] = 1

        # Bearish divergence: current price > previous high, but RSI < previous RSI high
        if close[i] > close[p_max_idx] and not np.isnan(rsi[p_max_idx]):
            if rsi[i] < rsi[p_max_idx]:
                sig[i] = -1

    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# S8: Moving Average Crossover
# ═══════════════════════════════════════════════════════════════

def gen_s8_ma_cross(close, fast_ma, slow_ma, warmup):
    """Buy when fast MA crosses above slow MA, sell when crosses below."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    v = ~np.isnan(fast_ma) & ~np.isnan(slow_ma)

    # Find crossover points
    for i in range(1, n):
        if not v[i] or not v[i-1]: continue
        if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            sig[i] = 1  # Golden cross
        elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
            sig[i] = -1  # Death cross

    sig[:warmup] = 0
    return sig


# ═══════════════════════════════════════════════════════════════
# SCAN ONE TICKER
# ═══════════════════════════════════════════════════════════════

def scan_ticker(ticker, daily_df, kr_s):
    """Phase 3: extended strategies scan for one ticker."""
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
    rsi = ind["rsi14"]; adx = ind["adx14"]; vol_sma20 = ind["vol_sma20"]
    day_of_week = dates.dayofweek.values

    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)

    rm_samples = [random_rm_params() for _ in range(N_RM)]
    comm = 0.0005
    results = []
    n_bt = 0

    def eval_sig(sid, sig_raw, sp_str, is_trend):
        nonlocal n_bt
        local = []
        if sig_raw is None: return local
        if np.sum(np.abs(sig_raw[oos_mask]) > 0) < 3: return local

        def_rm = DEFAULT_RM_TREND if is_trend else DEFAULT_RM
        pos = run_bt(sig_raw, close, high, low, atr, rh, rl, def_rm, WARMUP, n)
        sh, info = calc_excess_sharpe(pos, close, lr, dates, rf_d, cal_days, oos_mask, comm)
        n_bt += 1

        if sh > -0.5:
            local.append({"sid": sid, "ticker": ticker, "params": sp_str,
                          "conf": "none", "filter": "{}", "rm": "default",
                          "sharpe": round(sh, 3), "ret": info.get("total_ret", 0),
                          "vol": info.get("vol", 0), "exp": info.get("exposure", 0),
                          "wr": info.get("wr", 0), "tpy": info.get("tpy", 0)})

        if sh > T2_THRESHOLD:
            for rm in rm_samples:
                pos_r = run_bt(sig_raw, close, high, low, atr, rh, rl, rm, WARMUP, n)
                sh_r, info_r = calc_excess_sharpe(pos_r, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                n_bt += 1
                if sh_r > sh:
                    rm_str = json.dumps({k: v for k, v in rm.items() if v != 0}, sort_keys=True)
                    local.append({"sid": sid, "ticker": ticker, "params": sp_str,
                                  "conf": "none", "filter": "{}", "rm": rm_str,
                                  "sharpe": round(sh_r, 3), "ret": info_r.get("total_ret", 0),
                                  "vol": info_r.get("vol", 0), "exp": info_r.get("exposure", 0),
                                  "wr": info_r.get("wr", 0), "tpy": info_r.get("tpy", 0)})
        return local

    # ── S3 Extended: Donchian with confirmations ──
    for w in DC_WINDOWS:
        for conf in DC_CONFIRMATIONS:
            sig = gen_s3_extended(close, high, low, dc_cache, w, conf, volume, adx, vol_sma20, WARMUP)
            if sig is not None:
                sp = json.dumps({"w": w, "dc_conf": conf})
                results.extend(eval_sig("S3x", sig, sp, True))

    # ── S5 Extended: PivotPoints + DeMark + ATR filter ──
    # Add DeMark to pivot data
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}
    pivot_data["demark"] = calc_pivot_demark(high, low, close, open_arr)
    all_pivot_types = PIVOT_TYPES + ["demark"]

    for pt in all_pivot_types:
        for el in [1, 2, 3]:
            for atr_f in PIVOT_ATR_FILTERS:
                sig = gen_s5_extended(close, pivot_data, pt, el, atr, atr_f, WARMUP)
                if sig is not None:
                    sp = json.dumps({"pt": pt, "el": el, "atr_f": atr_f})
                    results.extend(eval_sig("S5x", sig, sp, False))

    # ── S1+S2 Hybrid: Bollinger Squeeze ──
    for (mt, w), z in ind["z_atr"].items():
        std = ind["std"].get((mt, w))
        ma = ind["ma"].get((mt, w))
        if std is None or ma is None: continue
        for sq_pct in SQUEEZE_PCTILES:
            for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                sig = gen_s1s2_hybrid(close, z, std, ma, sq_pct, thresh, WARMUP)
                if sig is not None:
                    sp = json.dumps({"ma": mt, "w": w, "squeeze_pct": sq_pct, "thresh": thresh})
                    results.extend(eval_sig("S1S2", sig, sp, False))

    # ── S3+S4 Hybrid: Donchian + Supertrend ──
    for dc_w in [5, 10, 14, 20, 30, 50]:
        for st_ap in [5, 10, 14, 20]:
            for st_mult in [1.0, 1.5, 2.0, 3.0]:
                sig = gen_s3s4_hybrid(close, dc_cache, st_cache, dc_w, st_ap, st_mult, WARMUP)
                if sig is not None:
                    sp = json.dumps({"dc_w": dc_w, "st_ap": st_ap, "st_mult": st_mult})
                    results.extend(eval_sig("S3S4", sig, sp, True))

    # ── S7: RSI Divergence ──
    for rsi_lookback in [10, 14, 20, 30]:
        sig = gen_s7_rsi_div(close, rsi, rsi_lookback, WARMUP)
        sp = json.dumps({"lookback": rsi_lookback})
        results.extend(eval_sig("S7", sig, sp, False))

    # ── S8: MA Crossover ──
    for fast_w in [5, 10, 20]:
        for slow_w in [50, 100, 200]:
            if fast_w >= slow_w: continue
            for mt in ["sma", "ema"]:
                fast = ind["ma"].get((mt, fast_w))
                slow = ind["ma"].get((mt, slow_w))
                if fast is None or slow is None: continue
                sig = gen_s8_ma_cross(close, fast, slow, WARMUP)
                sp = json.dumps({"fast_w": fast_w, "slow_w": slow_w, "ma_type": mt})
                results.extend(eval_sig("S8", sig, sp, True))

    # Keep top
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
    print("GRID SCAN — Phase 3: Extended strategies (S3x, S5x, S1S2, S3S4, S7, S8)", flush=True)
    print("=" * 100, flush=True)

    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    all_results = []
    total_bt = 0
    for i, ticker in enumerate(TICKERS):
        t1 = time.time()
        res, n_bt = scan_ticker(ticker, daily_df, kr_s)
        elapsed = time.time() - t1
        all_results.extend(res)
        total_bt += n_bt
        n_pos = sum(1 for r in res if r["sharpe"] > 0)
        n_gt1 = sum(1 for r in res if r["sharpe"] > 1.0)
        best_sh = max((r["sharpe"] for r in res), default=-999)
        print(f"  [{i+1}/17] {ticker}: {len(res)} results, {n_bt:,} bt_calls, "
              f"{n_pos} Sh>0, {n_gt1} Sh>1.0, best={best_sh:+.2f} ({elapsed:.1f}s)", flush=True)

    df = pd.DataFrame(all_results)
    out_path = BASE / "results/final/strategies/walkforward_v4/grid_scan_daily_phase3.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    elapsed_total = time.time() - t0
    print(f"\n{'='*100}", flush=True)
    print(f"PHASE 3 SUMMARY — {len(df)} results, {total_bt:,} bt_calls", flush=True)
    print(f"{'='*100}", flush=True)

    for sid in ["S3x", "S5x", "S1S2", "S3S4", "S7", "S8"]:
        sub = df[df["sid"] == sid]
        if len(sub) == 0: continue
        sh = sub["sharpe"]
        print(f"  {sid:5s}: N={len(sub):5d}, mean={sh.mean():+.2f}, median={sh.median():+.2f}, "
              f"max={sh.max():+.2f}, >0={(sh>0).sum()}, >1.0={(sh>1.0).sum()}", flush=True)

    # Top-10 per strategy
    for sid in ["S3x", "S5x", "S1S2", "S3S4", "S7", "S8"]:
        sub = df[df["sid"] == sid].nlargest(10, "sharpe")
        if len(sub) == 0: continue
        print(f"\n  Top-10 {sid}:", flush=True)
        for _, r in sub.iterrows():
            print(f"    Sh={r['sharpe']:+.2f}  Ret={r['ret']:+.1f}%  Vol={r['vol']:.0f}%  "
                  f"Exp={r['exp']:.0f}%  {r['ticker']:5s}  {r['params']}", flush=True)

    # Cross-ticker robustness
    print(f"\n{'='*100}", flush=True)
    print("CROSS-TICKER (3+ tickers, Sh>0):", flush=True)
    print(f"{'='*100}", flush=True)
    for sid in ["S3x", "S5x", "S1S2", "S3S4", "S7", "S8"]:
        sub = df[(df["sid"] == sid) & (df["sharpe"] > 0)]
        if len(sub) == 0: continue
        g = sub.groupby("params").agg(
            n_tickers=("ticker", "nunique"),
            mean_sh=("sharpe", "mean"),
            max_sh=("sharpe", "max"),
        ).sort_values("n_tickers", ascending=False)
        g3 = g[g["n_tickers"] >= 3]
        if len(g3) > 0:
            print(f"\n  {sid} — {len(g3)} combos on 3+ tickers:", flush=True)
            for params, row in g3.head(5).iterrows():
                print(f"    {row['n_tickers']}tk  mean={row['mean_sh']:+.2f}  max={row['max_sh']:+.2f}  {params}", flush=True)

    print(f"\nDONE in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
