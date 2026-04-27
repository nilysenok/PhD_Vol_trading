#!/usr/bin/env python3
"""
grid_scan_train.py — Run narrowed grid on TRAIN period (2014-2021).

Uses narrowed_grid.json from Phase 1 screening.
Evaluates each (sid, params) × ticker on the TRAIN mask (WARMUP to oos_start).
For promising signals (Sharpe > 0.2), also tries N_RM random RM variants.
"""
import os, sys, warnings, time, json, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

random.seed(42); np.random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_scan_test import (
    precompute_indicators, gen_s1, gen_s2, gen_s3, gen_s4, gen_s5, gen_s6,
    apply_confirmation, run_bt, random_rm_params,
    MA_TYPES, WINDOWS_D, DC_WINDOWS, ST_PERIODS, ST_MULTS, PIVOT_TYPES, VWAP_WINDOWS, VWAP_DM,
    IS_TREND, DEFAULT_RM, DEFAULT_RM_TREND,
    CONFIRMATIONS,
)
from daily_upper_bound import calc_excess_sharpe
from strategies_walkforward import (
    TICKERS, WARMUP, load_data, calc_atr,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)

KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
BASE = Path(__file__).resolve().parent.parent
BPY = 252
N_RM = 50
RM_THRESHOLD = 0.2  # try random RM if default Sharpe > this

def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("GRID SCAN TRAIN — Narrowed grid on TRAIN period (2014-2021)", flush=True)
    print("=" * 100, flush=True)

    # Load narrowed grid
    grid_path = BASE / "results/final/strategies/walkforward_v4/narrowed_grid.json"
    with open(grid_path) as f:
        narrowed = json.load(f)

    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        print(f"  {sid}: {len(narrowed.get(sid, []))} param combos", flush=True)

    # Load data
    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH)
    kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    # Pre-generate random RM samples
    rm_samples = [random_rm_params() for _ in range(N_RM)]

    all_results = []
    total_bt = 0
    comm = 0.0005

    for ti, ticker in enumerate(TICKERS):
        t1 = time.time()
        tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(tdf) < WARMUP + 200:
            print(f"  [{ti+1}/17] {ticker}: SKIP (too short)", flush=True)
            continue

        close = tdf["close"].values.astype(np.float64)
        high = tdf["high"].values.astype(np.float64)
        low = tdf["low"].values.astype(np.float64)
        open_arr = tdf["open"].values.astype(np.float64)
        volume = tdf["volume"].values.astype(np.float64)
        dates = pd.to_datetime(tdf["date"].values)
        n = len(close)
        lr = np.zeros(n)
        lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

        # Calendar/RF
        cal_days = np.ones(n)
        for i in range(1, n):
            cal_days[i] = (dates[i] - dates[i - 1]).days
        rf_raw = kr_s.reindex(dates, method="ffill").values
        rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0

        # TRAIN mask: WARMUP to oos_start (before 2022)
        oos_start = _year_bounds(dates, 2022)[0]
        if oos_start <= WARMUP:
            print(f"  [{ti+1}/17] {ticker}: SKIP (no train period)", flush=True)
            continue
        train_mask = np.zeros(n, dtype=bool)
        train_mask[WARMUP:oos_start] = True

        # Precompute indicators
        ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
        atr = ind["atr14"]
        rh = ind["rh"]
        rl = ind["rl"]
        rsi = ind["rsi14"]
        vol_sma20 = ind["vol_sma20"]

        # Precompute strategy-specific caches
        dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
        st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
        vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS)
        pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}

        n_bt_ticker = 0
        ticker_results = []

        def eval_signal(sid, sig_raw, sig_params_str, is_trend):
            nonlocal n_bt_ticker
            local_results = []
            def_rm = DEFAULT_RM_TREND if is_trend else DEFAULT_RM

            for conf in CONFIRMATIONS:
                sig = apply_confirmation(sig_raw.copy(), conf, rsi, volume, vol_sma20, is_trend)
                if np.sum(np.abs(sig[train_mask]) > 0) < 3:
                    continue

                pos = run_bt(sig, close, high, low, atr, rh, rl, def_rm, WARMUP, n)
                sh, info = calc_excess_sharpe(pos, close, lr, dates, rf_d, cal_days, train_mask, comm)
                n_bt_ticker += 1

                if sh > -0.5:
                    local_results.append({
                        "sid": sid, "ticker": ticker, "params": sig_params_str,
                        "conf": conf, "filter": "{}", "rm": "default",
                        "sharpe": round(sh, 3), "ret": info.get("total_ret", 0),
                        "vol": info.get("vol", 0), "exp": info.get("exposure", 0),
                        "wr": info.get("wr", 0), "tpy": info.get("tpy", 0)
                    })

                # Try random RM for promising signals
                if sh > RM_THRESHOLD:
                    for rm in rm_samples:
                        pos_r = run_bt(sig, close, high, low, atr, rh, rl, rm, WARMUP, n)
                        sh_r, info_r = calc_excess_sharpe(pos_r, close, lr, dates, rf_d, cal_days, train_mask, comm)
                        n_bt_ticker += 1
                        if sh_r > sh:
                            rm_str = json.dumps({k: v for k, v in rm.items() if v != 0}, sort_keys=True)
                            local_results.append({
                                "sid": sid, "ticker": ticker, "params": sig_params_str,
                                "conf": conf, "filter": "{}", "rm": rm_str,
                                "sharpe": round(sh_r, 3), "ret": info_r.get("total_ret", 0),
                                "vol": info_r.get("vol", 0), "exp": info_r.get("exposure", 0),
                                "wr": info_r.get("wr", 0), "tpy": info_r.get("tpy", 0)
                            })

            return local_results

        # Run each strategy with narrowed params
        for p in narrowed.get("S1", []):
            key = (p["ma"], p["w"])
            z = ind["z_atr"].get(key)
            if z is None:
                continue
            sig = gen_s1(z, p["thresh"], WARMUP)
            sp = json.dumps({"ma": p["ma"], "w": p["w"], "thresh": p["thresh"]})
            ticker_results.extend(eval_signal("S1", sig, sp, False))

        for p in narrowed.get("S2", []):
            key = (p["ma"], p["w"])
            z = ind["z_std"].get(key)
            if z is None:
                continue
            sig = gen_s2(z, p["bs"], WARMUP)
            sp = json.dumps({"ma": p["ma"], "w": p["w"], "bs": p["bs"]})
            ticker_results.extend(eval_signal("S2", sig, sp, False))

        for p in narrowed.get("S3", []):
            sig = gen_s3(close, dc_cache, p["w"], WARMUP)
            if sig is None:
                continue
            sp = json.dumps({"w": p["w"]})
            ticker_results.extend(eval_signal("S3", sig, sp, True))

        for p in narrowed.get("S4", []):
            sig = gen_s4(close, st_cache, p["ap"], p["mult"], WARMUP)
            if sig is None:
                continue
            sp = json.dumps({"ap": p["ap"], "mult": p["mult"]})
            ticker_results.extend(eval_signal("S4", sig, sp, True))

        for p in narrowed.get("S5", []):
            sig = gen_s5(close, pivot_data, p["pt"], p["el"], WARMUP)
            if sig is None:
                continue
            sp = json.dumps({"pt": p["pt"], "el": p["el"]})
            ticker_results.extend(eval_signal("S5", sig, sp, False))

        for p in narrowed.get("S6", []):
            sig = gen_s6(close, vwap_cache, p["vw"], p["dm"], WARMUP)
            if sig is None:
                continue
            sp = json.dumps({"vw": p["vw"], "dm": p["dm"]})
            ticker_results.extend(eval_signal("S6", sig, sp, False))

        all_results.extend(ticker_results)
        total_bt += n_bt_ticker
        n_pos = sum(1 for r in ticker_results if r["sharpe"] > 0)
        best_sh = max((r["sharpe"] for r in ticker_results), default=-999)
        elapsed = time.time() - t1
        print(f"  [{ti+1}/17] {ticker}: {len(ticker_results)} results, {n_bt_ticker} bt_calls, "
              f"{n_pos} Sh>0, best={best_sh:+.2f} ({elapsed:.1f}s)", flush=True)

    # Save to parquet
    df = pd.DataFrame(all_results)
    out_path = BASE / "results/final/strategies/walkforward_v4/grid_scan_daily_TRAIN.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    elapsed_total = time.time() - t0
    print(f"\n{'='*100}", flush=True)
    print(f"SUMMARY — {len(df)} total results, {total_bt} bt_universal calls", flush=True)
    print(f"{'='*100}", flush=True)

    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        sub = df[df["sid"] == sid]
        if len(sub) == 0:
            continue
        sh = sub["sharpe"]
        print(f"  {sid}: N={len(sub)}, mean={sh.mean():+.2f}, median={sh.median():+.2f}, "
              f"max={sh.max():+.2f}, >0={(sh > 0).sum()}", flush=True)

    print(f"\nDONE in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
