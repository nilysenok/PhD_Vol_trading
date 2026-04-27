#!/usr/bin/env python3
"""
grid_scan_phase4.py — Phase 4: Test 8 per-ticker extensions on top Phase 2 combos.

Extensions tested:
  1. Partial Exit (TP1 + trailing remainder)
  2. Asymmetric Long/Short (different SL/TP for shorts)
  3. Market Regime Filter (IMOEX trend + vol regime)
  4. Dynamic Sizing (inverse-vol, signal-weighted)
  5. Cooldown after consecutive losses
  6. Pyramiding (position scaling on repeat signals)
  7. Vol-of-Vol Filter
  8. Seasonality (skip summer, turn-of-month)
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
    TICKERS, WARMUP, load_data, calc_rsi, calc_sma, calc_std, calc_atr,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)
from daily_upper_bound import bt_universal, calc_excess_sharpe
from grid_scan_test import (
    precompute_indicators, run_bt,
    MA_TYPES, WINDOWS_D, DC_WINDOWS, ST_PERIODS, ST_MULTS, PIVOT_TYPES, VWAP_WINDOWS,
    gen_s1, gen_s2, gen_s3, gen_s4, gen_s5, gen_s6,
    apply_confirmation, apply_filters, IS_TREND,
)

KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
IMOEX_PATH = "/Users/nilysenok/Desktop/disser/data/Indexes/D/IMOEX.csv"
BASE = Path(__file__).resolve().parent.parent
PHASE2_PATH = BASE / "results/final/strategies/walkforward_v4/grid_scan_daily_phase2.parquet"
BPY = 252

TOP_N = 5  # top signal params per strategy (cross-ticker)


# ═══════════════════════════════════════════════════════════════
# SIGNAL REGENERATION (from Phase 2 params)
# ═══════════════════════════════════════════════════════════════

def regenerate_signal(sid, params_str, conf, filter_str,
                      close, high, low, volume, open_arr,
                      ind, dc_cache, st_cache, vwap_cache, pivot_data,
                      day_of_week, atr_arr):
    """Regenerate signal array from saved params."""
    p = json.loads(params_str)
    rsi = ind["rsi14"]; vol_sma20 = ind["vol_sma20"]
    is_trend = IS_TREND.get(sid, False)
    sig = None

    if sid == "S1":
        key = (p["ma"], p["w"])
        z = ind["z_atr"].get(key)
        if z is not None:
            sig = gen_s1(z, p["thresh"], WARMUP)
    elif sid == "S2":
        key = (p["ma"], p["w"])
        z = ind["z_std"].get(key)
        if z is not None:
            sig = gen_s2(z, p["bs"], WARMUP)
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

    fp = json.loads(filter_str) if filter_str and filter_str != "{}" else {}
    if fp:
        sig = apply_filters(sig.copy(), close, volume, ind, atr_arr, day_of_week, fp, is_trend)

    return sig


def parse_rm(rm_str):
    """Parse RM string to dict."""
    if rm_str == "default":
        return dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=3.0, trail=0, mh=15, be=0, chand=0, rr=0)
    try:
        d = json.loads(rm_str)
        return dict(sl_p=d.get("sl_p", 0), tp_p=d.get("tp_p", 0),
                    sl_a=d.get("sl_a", 0), tp_a=d.get("tp_a", 0),
                    trail=d.get("trail", 0), mh=d.get("mh", 0),
                    be=d.get("be", 0), chand=d.get("chand", 0), rr=d.get("rr", 0))
    except:
        return dict(sl_p=0, tp_p=0, sl_a=2.0, tp_a=3.0, trail=0, mh=15, be=0, chand=0, rr=0)


# ═══════════════════════════════════════════════════════════════
# EXTENSION IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

def ext_partial_exit(sig, rm, close, high, low, atr, rh, rl, warmup, n,
                     exit_pct, tp1_ratio, trail_atr):
    """Partial exit: close exit_pct at TP1, rest with trailing."""
    # Run tight TP version
    rm_tight = rm.copy()
    if rm["tp_a"] > 0:
        rm_tight["tp_a"] = rm["tp_a"] * tp1_ratio
    elif rm["tp_p"] > 0:
        rm_tight["tp_p"] = rm["tp_p"] * tp1_ratio
    elif rm["rr"] > 0:
        rm_tight["rr"] = rm["rr"] * tp1_ratio
    else:
        rm_tight["tp_a"] = 2.0 * tp1_ratio  # default TP

    # Run trailing version
    rm_trail = rm.copy()
    rm_trail["trail"] = trail_atr
    rm_trail["tp_p"] = 0; rm_trail["tp_a"] = 0; rm_trail["rr"] = 0
    rm_trail["be"] = max(rm.get("be", 0), 1.0)  # breakeven after partial

    pos_tight = run_bt(sig, close, high, low, atr, rh, rl, rm_tight, warmup, n)
    pos_trail = run_bt(sig, close, high, low, atr, rh, rl, rm_trail, warmup, n)

    # Blend: tight portion exits early, trail portion stays longer
    pos = exit_pct * pos_tight + (1 - exit_pct) * pos_trail
    return pos


def ext_asymmetric(sig, rm, close, high, low, atr, rh, rl, warmup, n,
                   sl_mult_short, tp_mult_short, w_ratio_short):
    """Asymmetric: different SL/TP for long vs short."""
    sig_long = sig.copy(); sig_long[sig < 0] = 0
    sig_short = sig.copy(); sig_short[sig > 0] = 0

    pos_long = run_bt(sig_long, close, high, low, atr, rh, rl, rm, warmup, n)

    # Short RM with wider SL, potentially tighter TP
    rm_short = rm.copy()
    if rm["sl_a"] > 0: rm_short["sl_a"] = rm["sl_a"] * sl_mult_short
    if rm["sl_p"] > 0: rm_short["sl_p"] = rm["sl_p"] * sl_mult_short
    if rm["tp_a"] > 0: rm_short["tp_a"] = rm["tp_a"] * tp_mult_short
    if rm["tp_p"] > 0: rm_short["tp_p"] = rm["tp_p"] * tp_mult_short
    if rm["chand"] > 0: rm_short["chand"] = rm["chand"] * sl_mult_short

    pos_short = run_bt(sig_short, close, high, low, atr, rh, rl, rm_short, warmup, n)
    pos = pos_long + pos_short * w_ratio_short
    return pos


def ext_market_regime(sig, imoex_aligned, imoex_sma, atr_imoex, rule):
    """Market regime filter based on IMOEX."""
    out = sig.copy()
    v = ~np.isnan(imoex_sma)

    if rule == "long_only_above":
        out[v & (imoex_aligned < imoex_sma) & (sig > 0)] = 0
    elif rule == "no_short_above":
        out[v & (imoex_aligned > imoex_sma) & (sig < 0)] = 0
    elif rule == "skip_extreme_vol":
        if atr_imoex is not None:
            va = atr_imoex[~np.isnan(atr_imoex)]
            if len(va) > 50:
                p90 = np.percentile(va, 90)
                out[(~np.isnan(atr_imoex)) & (atr_imoex > p90)] = 0
    elif rule == "contrarian_below":
        # Only counter-trend below SMA (mean reversion in downtrend)
        out[v & (imoex_aligned > imoex_sma) & (sig > 0)] = 0
    return out


def ext_dynamic_sizing(pos, close, atr_arr, mode, param):
    """Dynamic position sizing."""
    n = len(pos)
    if mode == "inverse_vol":
        target, window = param
        rv = pd.Series(np.log(close[1:] / close[:-1])).rolling(window).std().values
        rv = np.insert(rv, 0, np.nan)
        rv_safe = np.where((rv < 1e-8) | np.isnan(rv), target, rv)
        scale = np.clip(target / rv_safe, 0.3, 3.0)
        return pos * scale
    elif mode == "signal_weighted":
        # Scale by absolute z-score percentile (rough proxy)
        scaling_type = param
        exp = np.abs(pos)
        if scaling_type == "sqrt":
            return np.sign(pos) * np.sqrt(exp)
        else:  # linear — no change, pos is already {-1, 0, 1}
            return pos
    return pos


def ext_cooldown(pos, lr, trigger_losses, action):
    """Cooldown after consecutive losses."""
    n = len(pos)
    out = pos.copy()
    consec_losses = 0
    in_trade = False
    trade_pnl = 0.0
    cooldown_bars = 0

    for i in range(n):
        if cooldown_bars > 0:
            if action == "pause_10bars":
                out[i] = 0
            elif action == "reduce_50pct":
                out[i] = pos[i] * 0.5
            elif action == "widen_stops":
                pass  # can't modify stops post-hoc, skip
            cooldown_bars -= 1
            continue

        if abs(pos[i]) > 1e-8:
            if not in_trade:
                in_trade = True; trade_pnl = 0.0
            trade_pnl += pos[i] * lr[i]
        else:
            if in_trade:
                if trade_pnl < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0
                in_trade = False

                if consec_losses >= trigger_losses:
                    cooldown_bars = 10
                    consec_losses = 0
    return out


def ext_pyramiding(pos, sig, lr, atr_arr, max_pyr, pyr_ratio, pyr_cond):
    """Pyramiding: scale up position on repeat signals or profit."""
    n = len(pos)
    out = pos.copy()
    n_pyramids = 0
    in_trade = False
    trade_pnl = 0.0
    entry_atr = 1.0

    for i in range(n):
        if abs(pos[i]) > 1e-8:
            if not in_trade:
                in_trade = True; trade_pnl = 0.0; n_pyramids = 0
                entry_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 1.0
            trade_pnl += pos[i] * lr[i]

            # Check pyramid condition
            if n_pyramids < max_pyr:
                add = False
                if pyr_cond == "repeat_signal" and sig[i] == np.sign(pos[i]):
                    add = True
                elif pyr_cond == "profit_1atr" and trade_pnl > entry_atr * 0.01:
                    add = True
                elif pyr_cond == "profit_2atr" and trade_pnl > 2 * entry_atr * 0.01:
                    add = True

                if add:
                    n_pyramids += 1
                    # Scale up remaining position
                    scale = 1.0 + n_pyramids * pyr_ratio
                    out[i] = np.sign(pos[i]) * min(scale, 1.0 + max_pyr * pyr_ratio)
        else:
            in_trade = False; n_pyramids = 0

    return out


def ext_vov_filter(sig, atr_arr, window, pctile):
    """Vol-of-Vol filter: skip when vol is unstable."""
    out = sig.copy()
    vov = pd.Series(atr_arr).rolling(window).std().values
    va = vov[~np.isnan(vov)]
    if len(va) < 50: return out
    thresh = np.percentile(va, pctile)
    out[(~np.isnan(vov)) & (vov > thresh)] = 0
    return out


def ext_seasonality(sig, dates, mode):
    """Seasonality filter."""
    out = sig.copy()
    months = dates.month
    days = dates.day
    # Days in month
    dim = pd.Series(dates).dt.days_in_month.values

    if mode == "skip_summer":
        out[(months >= 6) & (months <= 8)] = 0
    elif mode == "skip_jan":
        out[months == 1] = 0
    elif mode == "turn_of_month":
        # Boost at turn of month: last 2 + first 3 days
        tom = (days <= 3) | (days >= dim - 1)
        # Don't modify signal, but scale position later
        # For simplicity: zero out non-TOM signals (aggressive)
        # Actually just return sig unchanged — we'll handle via position scaling
        pass
    return out


# ═══════════════════════════════════════════════════════════════
# MAIN SCAN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("GRID SCAN — Phase 4: Test 8 extensions on top Phase 2 combos", flush=True)
    print("=" * 100, flush=True)

    # ── Load Phase 2 ──
    df_p2 = pd.read_parquet(PHASE2_PATH)
    print(f"Phase 2: {len(df_p2)} rows", flush=True)

    # ── Select top signal params per strategy (cross-ticker) ──
    top_combos = {}
    for sid in ["S1", "S2", "S3", "S4", "S6"]:
        sub = df_p2[df_p2.sid == sid]
        g = sub.groupby(["params", "conf", "filter"]).agg(
            n_tk=("ticker", "nunique"),
            mean_sh=("sharpe", "mean"),
        ).reset_index()
        g = g[g.n_tk >= 2].sort_values("mean_sh", ascending=False).head(TOP_N)
        top_combos[sid] = []
        for _, row in g.iterrows():
            # For each top signal, get best RM per ticker
            mask = (sub.params == row.params) & (sub.conf == row.conf) & (sub["filter"] == row["filter"])
            best_per_tk = sub[mask].loc[sub[mask].groupby("ticker")["sharpe"].idxmax()]
            top_combos[sid].append({
                "params": row.params, "conf": row.conf, "filter": row["filter"],
                "per_ticker": best_per_tk
            })
        print(f"  {sid}: {len(top_combos[sid])} top signal combos", flush=True)

    # ── Load market data ──
    daily_df, _, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    # ── Load IMOEX ──
    imoex_df = pd.read_csv(IMOEX_PATH, sep=";")
    imoex_df["date"] = pd.to_datetime(imoex_df["begin"])
    imoex_close_s = imoex_df.set_index("date")["close"].sort_index()

    # ── Extension definitions ──
    extensions = {}

    # 1. Partial Exit
    pe_combos = []
    for ep in [0.25, 0.50, 0.75]:
        for tp1r in [0.5, 0.75, 1.0]:
            for ta in [1.5, 2.0, 3.0]:
                pe_combos.append({"exit_pct": ep, "tp1_ratio": tp1r, "trail_atr": ta})
    extensions["partial_exit"] = pe_combos

    # 2. Asymmetric
    asym_combos = []
    for slm in [1.0, 1.25, 1.5, 2.0]:
        for tpm in [0.75, 1.0, 1.25]:
            for wr in [0.5, 0.75, 1.0]:
                if slm == 1.0 and tpm == 1.0 and wr == 1.0: continue  # skip baseline
                asym_combos.append({"sl_mult_short": slm, "tp_mult_short": tpm, "w_ratio_short": wr})
    extensions["asymmetric"] = asym_combos

    # 3. Market Regime
    mr_combos = []
    for sma_p in [50, 100, 200]:
        for rule in ["long_only_above", "no_short_above", "skip_extreme_vol"]:
            mr_combos.append({"sma_period": sma_p, "rule": rule})
    extensions["market_regime"] = mr_combos

    # 4. Dynamic Sizing
    ds_combos = [
        {"mode": "inverse_vol", "param": (0.10, 20)},
        {"mode": "inverse_vol", "param": (0.15, 20)},
        {"mode": "inverse_vol", "param": (0.20, 20)},
        {"mode": "inverse_vol", "param": (0.10, 60)},
        {"mode": "inverse_vol", "param": (0.15, 60)},
        {"mode": "inverse_vol", "param": (0.20, 60)},
        {"mode": "signal_weighted", "param": "sqrt"},
    ]
    extensions["dynamic_sizing"] = ds_combos

    # 5. Cooldown
    cd_combos = []
    for tl in [2, 3, 5]:
        for act in ["pause_10bars", "reduce_50pct"]:
            cd_combos.append({"trigger": tl, "action": act})
    extensions["cooldown"] = cd_combos

    # 6. Pyramiding
    pyr_combos = []
    for mp in [1, 2, 3]:
        for pr in [0.25, 0.5, 1.0]:
            for pc in ["repeat_signal", "profit_1atr", "profit_2atr"]:
                pyr_combos.append({"max_pyr": mp, "pyr_ratio": pr, "pyr_cond": pc})
    extensions["pyramiding"] = pyr_combos

    # 7. Vol-of-Vol
    vov_combos = []
    for w in [20, 40, 60]:
        for p in [75, 85, 90]:
            vov_combos.append({"window": w, "pctile": p})
    extensions["vov_filter"] = vov_combos

    # 8. Seasonality
    seas_combos = [
        {"mode": "skip_summer"},
        {"mode": "skip_jan"},
    ]
    extensions["seasonality"] = seas_combos

    print(f"\nExtensions to test:", flush=True)
    for name, combos in extensions.items():
        print(f"  {name}: {len(combos)} combos", flush=True)

    # ── Scan all tickers ──
    # Results: {ext_name: {sid: [(ticker, baseline_sh, ext_sh, delta, ext_params)]}}
    ext_results = {name: {sid: [] for sid in ["S1", "S2", "S3", "S4", "S6"]}
                   for name in extensions}
    total_bt = 0

    for ti, ticker in enumerate(TICKERS):
        t1 = time.time()
        tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(tdf) < WARMUP + 200: continue

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
        if oos_start <= WARMUP: continue
        oos_mask = np.zeros(n, dtype=bool); oos_mask[oos_start:] = True

        ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
        atr = ind["atr14"]; rh = ind["rh"]; rl = ind["rl"]
        day_of_week = dates.dayofweek.values

        dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
        st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
        vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS)
        pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}

        # Align IMOEX to ticker dates
        imoex_aligned = imoex_close_s.reindex(dates, method="ffill").values.astype(np.float64)
        imoex_smas = {}
        for p in [50, 100, 200]:
            imoex_smas[p] = calc_sma(imoex_aligned, p)
        imoex_atr = calc_atr(
            imoex_aligned, imoex_aligned, imoex_aligned, 14  # rough ATR proxy
        )

        comm = 0.0005
        ticker_bt = 0

        for sid in ["S1", "S2", "S3", "S4", "S6"]:
            if sid not in top_combos: continue
            is_trend = IS_TREND[sid]

            for combo in top_combos[sid]:
                # Regenerate signal
                sig = regenerate_signal(
                    sid, combo["params"], combo["conf"], combo["filter"],
                    close, high, low, volume, open_arr,
                    ind, dc_cache, st_cache, vwap_cache, pivot_data,
                    day_of_week, atr
                )
                if sig is None: continue
                if np.sum(np.abs(sig[oos_mask]) > 0) < 3: continue

                # Get best RM for this ticker from Phase 2
                tk_rows = combo["per_ticker"]
                tk_row = tk_rows[tk_rows.ticker == ticker]
                if len(tk_row) == 0: continue
                rm = parse_rm(tk_row.iloc[0]["rm"])
                baseline_sh = tk_row.iloc[0]["sharpe"]

                # Baseline position
                pos_base = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                ticker_bt += 1

                # ── Test each extension ──

                # 1. Partial Exit
                for ep in extensions["partial_exit"]:
                    pos_ext = ext_partial_exit(sig, rm, close, high, low, atr, rh, rl, WARMUP, n,
                                              ep["exit_pct"], ep["tp1_ratio"], ep["trail_atr"])
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    ticker_bt += 2
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["partial_exit"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps(ep)))

                # 2. Asymmetric
                for ap in extensions["asymmetric"]:
                    pos_ext = ext_asymmetric(sig, rm, close, high, low, atr, rh, rl, WARMUP, n,
                                            ap["sl_mult_short"], ap["tp_mult_short"], ap["w_ratio_short"])
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    ticker_bt += 2
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["asymmetric"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps(ap)))

                # 3. Market Regime
                for mr in extensions["market_regime"]:
                    sma = imoex_smas[mr["sma_period"]]
                    sig_mr = ext_market_regime(sig, imoex_aligned, sma, imoex_atr, mr["rule"])
                    pos_ext = run_bt(sig_mr, close, high, low, atr, rh, rl, rm, WARMUP, n)
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    ticker_bt += 1
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["market_regime"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps(mr)))

                # 4. Dynamic Sizing
                for ds in extensions["dynamic_sizing"]:
                    pos_ext = ext_dynamic_sizing(pos_base.copy(), close, atr, ds["mode"], ds["param"])
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["dynamic_sizing"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps({"mode": ds["mode"]})))

                # 5. Cooldown
                for cd in extensions["cooldown"]:
                    pos_ext = ext_cooldown(pos_base.copy(), lr, cd["trigger"], cd["action"])
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["cooldown"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps(cd)))

                # 6. Pyramiding
                for py in extensions["pyramiding"]:
                    pos_ext = ext_pyramiding(pos_base.copy(), sig, lr, atr,
                                            py["max_pyr"], py["pyr_ratio"], py["pyr_cond"])
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["pyramiding"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps(py)))

                # 7. Vol-of-Vol
                for vv in extensions["vov_filter"]:
                    sig_vv = ext_vov_filter(sig, atr, vv["window"], vv["pctile"])
                    pos_ext = run_bt(sig_vv, close, high, low, atr, rh, rl, rm, WARMUP, n)
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    ticker_bt += 1
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["vov_filter"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps(vv)))

                # 8. Seasonality
                for ss in extensions["seasonality"]:
                    sig_ss = ext_seasonality(sig, dates, ss["mode"])
                    pos_ext = run_bt(sig_ss, close, high, low, atr, rh, rl, rm, WARMUP, n)
                    sh, _ = calc_excess_sharpe(pos_ext, close, lr, dates, rf_d, cal_days, oos_mask, comm)
                    ticker_bt += 1
                    if sh > -900:
                        delta = sh - baseline_sh
                        ext_results["seasonality"][sid].append(
                            (ticker, baseline_sh, sh, delta, json.dumps(ss)))

        total_bt += ticker_bt
        elapsed = time.time() - t1
        print(f"  [{ti+1}/17] {ticker}: {ticker_bt:,} bt_calls ({elapsed:.1f}s)", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════

    elapsed_total = time.time() - t0
    print(f"\n{'='*100}", flush=True)
    print(f"PHASE 4 RESULTS — {total_bt:,} bt_calls, {elapsed_total:.0f}s", flush=True)
    print(f"{'='*100}", flush=True)

    print(f"\n{'Extension':<18s} {'Strategy':<6s} {'N':>5s} {'Mean Δ':>8s} {'Med Δ':>8s} "
          f"{'%Improved':>10s} {'Best Δ':>8s} {'Worst Δ':>8s} Verdict", flush=True)
    print("-" * 100, flush=True)

    summary_rows = []

    for ext_name in ["partial_exit", "asymmetric", "market_regime", "dynamic_sizing",
                     "cooldown", "pyramiding", "vov_filter", "seasonality"]:
        for sid in ["S1", "S2", "S3", "S4", "S6"]:
            data = ext_results[ext_name][sid]
            if len(data) == 0: continue

            deltas = [d[3] for d in data]
            n_total = len(deltas)
            mean_d = np.mean(deltas)
            med_d = np.median(deltas)
            pct_imp = sum(1 for d in deltas if d > 0) / n_total * 100
            best_d = max(deltas)
            worst_d = min(deltas)

            if mean_d > 0.05 and pct_imp > 50:
                verdict = "✅ Include"
            elif mean_d > 0.02 and pct_imp > 40:
                verdict = "⚠️ Marginal"
            elif mean_d > 0 and pct_imp > 30:
                verdict = "⚠️ Weak"
            else:
                verdict = "❌ Skip"

            print(f"{ext_name:<18s} {sid:<6s} {n_total:>5d} {mean_d:>+8.3f} {med_d:>+8.3f} "
                  f"{pct_imp:>9.0f}% {best_d:>+8.3f} {worst_d:>+8.3f} {verdict}", flush=True)

            summary_rows.append({
                "extension": ext_name, "sid": sid, "n": n_total,
                "mean_delta": round(mean_d, 4), "median_delta": round(med_d, 4),
                "pct_improved": round(pct_imp, 1), "best_delta": round(best_d, 3),
                "worst_delta": round(worst_d, 3), "verdict": verdict
            })

    # Save summary
    df_sum = pd.DataFrame(summary_rows)
    out_path = BASE / "results/final/strategies/walkforward_v4/grid_scan_daily_phase4_extensions.parquet"
    df_sum.to_parquet(out_path)

    # ── Best extension combos per strategy ──
    print(f"\n{'='*100}", flush=True)
    print("BEST EXTENSION PARAMS (top-3 per extension×strategy by mean delta)", flush=True)
    print(f"{'='*100}", flush=True)

    for ext_name in ["partial_exit", "asymmetric", "market_regime", "dynamic_sizing",
                     "cooldown", "pyramiding", "vov_filter", "seasonality"]:
        for sid in ["S1", "S2", "S3", "S4", "S6"]:
            data = ext_results[ext_name][sid]
            if len(data) == 0: continue

            # Group by ext_params
            param_groups = {}
            for tk, bsh, esh, delta, ep_str in data:
                if ep_str not in param_groups:
                    param_groups[ep_str] = []
                param_groups[ep_str].append(delta)

            best = sorted(param_groups.items(), key=lambda x: np.mean(x[1]), reverse=True)[:3]
            if best and np.mean(best[0][1]) > 0:
                print(f"\n  {ext_name} × {sid}:", flush=True)
                for ep_str, deltas in best:
                    n_pos = sum(1 for d in deltas if d > 0)
                    print(f"    mean_Δ={np.mean(deltas):+.3f}  {n_pos}/{len(deltas)} improved  {ep_str}",
                          flush=True)

    print(f"\nDONE in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
