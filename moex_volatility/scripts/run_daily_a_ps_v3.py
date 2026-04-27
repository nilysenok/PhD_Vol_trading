#!/usr/bin/env python3
"""
run_daily_a_ps_v3.py — Daily Approach A + Position Sizing (v3)

Changes from original run_daily_a_ps.py:
- Train metric: trading_sharpe (mean(net)/std(net)×√252, comm=0.0005, no FDR, no RF)
- OOS metric: calc_trading_sharpe (compound - RF, no FDR, vol ALL days)
- Minimum trades filter: ≥20 trades on train
- Sub-period consistency: trading_sharpe > 0 on both halves of train
- Extended PS-2 W_GRID
- Output to daily_a_v3/
"""
import os, sys, warnings, time
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    TICKERS, STRATEGY_IDS, STRATEGY_NAMES, CATEGORY,
    A_TEST_YEARS, WARMUP,
    SIGNAL_GRIDS, RM_GRIDS_V3, TREND_RM_GRID,
    expand_grid, load_data,
    compute_base, precompute_sma_cache, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    calc_pivot_daily,
    dispatch_signals, dispatch_bt_atr_v3,
    warmup_numba,
    _year_bounds, _store_metrics, _extract_trades_from_pos,
    _trades_to_rows, _pctile_cache,
)

BASE = Path(__file__).resolve().parent.parent
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
OUT_DIR = BASE / "results/final/strategies/walkforward_v4/daily_a_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TICKERS = 17
BPY = 252
COMM_LEVELS = {"0.05%": 0.0005, "0.06%": 0.0006}
OOS_YEARS = [2022, 2023, 2024, 2025]


# ═══════════════════════════════════════════════════
# Train metric: trading_sharpe
# ═══════════════════════════════════════════════════

def trading_sharpe(pos, log_ret, mask, comm=0.0005):
    """mean(net)/std(net)×√252, with commission, NO FDR, NO RF."""
    pm = pos[mask]; lr = log_ret[mask]
    if len(pm) < 30:
        return -999.0
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    s = np.std(net, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(net) / s * sqrt(252)


def count_trades(pos_arr):
    """Count number of completed trades in position array."""
    nt = 0; in_t = False
    for v in pos_arr:
        if abs(v) > 1e-8 and not in_t:
            in_t = True
        elif abs(v) < 1e-8 and in_t:
            nt += 1; in_t = False
    if in_t:
        nt += 1
    return nt


# ═══════════════════════════════════════════════════
# OOS metric: calc_trading_sharpe (from grid_scan_test.py)
# ═══════════════════════════════════════════════════

def calc_trading_sharpe(pos, close, log_ret, dates, rf_daily, cal_days, mask, comm):
    """Trading Sharpe: compound return - compound RF, vol on ALL days. No FDR."""
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    dm = dates[mask]; n = len(pm)
    if n < 30:
        return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exp = np.abs(pm); active = exp > 1e-8; na = active.sum()
    if na < 10:
        return -999.0, {}
    cs = (dm[-1] - dm[0]).days
    if cs < 30:
        return -999.0, {}
    cum = np.prod(1.0 + net) - 1.0
    ann = (1.0 + cum) ** (365.0 / cs) - 1.0
    rf_c = np.prod(1.0 + rf * cd) - 1.0
    rf_a = (1.0 + rf_c) ** (365.0 / cs) - 1.0
    vol = np.std(net, ddof=1) * sqrt(BPY)
    if vol < 1e-6:
        return 0.0, {}
    ex = ann - rf_a
    exsh = ex / vol
    ny = cs / 365.0
    nt = 0; nw = 0; in_t = False; tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t:
            in_t = True; tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1
                if tr > 0:
                    nw += 1
                in_t = False
    return exsh, dict(total_ret=round(ann * 100, 2), vol=round(vol * 100, 2),
                      exposure=round(na / n * 100, 1),
                      wr=round(nw / nt * 100, 1) if nt > 0 else 0.0,
                      tpy=round(nt / max(ny, 0.01), 1),
                      rf=round(rf_a * 100, 2))


# ═══════════════════════════════════════════════════
# Part 1: Approach A — daily grid search (v3)
# ═══════════════════════════════════════════════════

TOP_K = 10

def approach_a_daily(sid, close, high, low, volume, open_arr, ind,
                     sma_cache, dc_cache, st_cache, vwap_cache,
                     pivot_data, dates, log_ret):
    """Approach A for daily: 2-pass grid search + top-10 ensemble (v3)."""
    ann = sqrt(252)
    n = len(close)
    train_mask = np.zeros(n, dtype=bool)
    warmup = WARMUP

    sig_grid = expand_grid(SIGNAL_GRIDS[sid])
    cat = CATEGORY[sid]
    if cat == "Trend":
        rm_grid = TREND_RM_GRID
    else:
        rm_grid = expand_grid(RM_GRIDS_V3[(cat, "daily")])

    results_by_year = {}
    params_by_year = {}
    trades_by_year = {}

    for test_year in A_TEST_YEARS:
        _, train_end = _year_bounds(dates, test_year)
        if test_year == 2026:
            test_end = n
        else:
            _, test_end = _year_bounds(dates, test_year + 1)
        if train_end <= warmup or train_end >= n:
            continue

        train_mask[:] = False
        train_mask[warmup:train_end] = True

        # Half-masks for sub-period consistency
        train_len = train_end - warmup
        half = train_len // 2
        first_half_mask = np.zeros(n, dtype=bool)
        first_half_mask[warmup:warmup + half] = True
        second_half_mask = np.zeros(n, dtype=bool)
        second_half_mask[warmup + half:train_end] = True

        # Cache all signals
        cached_sigs = {}
        for sp in sig_grid:
            key = tuple(sorted(sp.items()))
            sig, exit_info = dispatch_signals(sid, close, high, low, volume, ind,
                                              sma_cache, dc_cache, st_cache, vwap_cache,
                                              pivot_data, None, sp, is_hourly=False)
            cached_sigs[key] = (sig, exit_info)

        # Pass 1: coarse — sample every 3rd RM, collect top signal combos
        coarse_step = max(1, len(rm_grid) // 72)
        top_combos = []
        for sp in sig_grid:
            sp_key = tuple(sorted(sp.items()))
            sig, exit_info = cached_sigs[sp_key]
            for rm_idx in range(0, len(rm_grid), coarse_step):
                rm = rm_grid[rm_idx]
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, warmup, train_end, 0)
                sh = trading_sharpe(pos, log_ret, train_mask)
                top_combos.append((sh, sp_key, rm_idx))

        top_combos.sort(key=lambda x: x[0], reverse=True)
        # Collect top-30 unique signal keys
        seen_sp = set()
        top_sp_keys = []
        for sh, sp_key, _ in top_combos:
            if sp_key not in seen_sp:
                seen_sp.add(sp_key)
                top_sp_keys.append(sp_key)
                if len(top_sp_keys) >= 30:
                    break

        # Pass 2: fine — full RM grid for top signal combos
        all_results = []
        for sp_key in top_sp_keys:
            sig, exit_info = cached_sigs[sp_key]
            for rm_idx, rm in enumerate(rm_grid):
                pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                            rm, warmup, train_end, 0)
                sh = trading_sharpe(pos, log_ret, train_mask)
                all_results.append((sh, sp_key, rm_idx))

        all_results.sort(key=lambda x: x[0], reverse=True)
        top10_raw = all_results[:TOP_K * 5]  # take more candidates for filtering

        if not top10_raw or top10_raw[0][0] <= -900:
            continue

        # Filter: min trades + sub-period consistency
        top10 = []
        for sh_c, sp_key_c, rm_idx_c in top10_raw:
            if sh_c <= -900:
                break
            sig, exit_info = cached_sigs[sp_key_c]
            rm_c = rm_grid[rm_idx_c]
            pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                        rm_c, warmup, train_end, 0)
            # Min trades check
            if count_trades(pos[train_mask]) < 20:
                continue
            # Sub-period consistency
            sh_h1 = trading_sharpe(pos, log_ret, first_half_mask)
            sh_h2 = trading_sharpe(pos, log_ret, second_half_mask)
            if sh_h1 <= 0 or sh_h2 <= 0:
                continue
            top10.append((sh_c, sp_key_c, rm_idx_c))
            if len(top10) >= TOP_K:
                break

        if not top10:
            continue

        # Ensemble: top-10 majority vote on test period
        ensemble = np.zeros(n, dtype=np.float64)
        for sh_c, sp_key_c, rm_idx_c in top10:
            sig, exit_info = cached_sigs[sp_key_c]
            rm_c = rm_grid[rm_idx_c]
            pos, _ = dispatch_bt_atr_v3(sid, sig, exit_info, close, high, low, ind,
                                        rm_c, warmup, test_end, 0)
            ensemble += np.sign(pos)
        ensemble = ensemble / float(TOP_K)

        results_by_year[test_year] = ensemble
        params_by_year[test_year] = {"sig": dict(top10[0][1]), "rm_idx": top10[0][2]}
        trades_by_year[test_year] = np.empty((0, 7))

    return results_by_year, params_by_year, trades_by_year


def process_ticker_daily_a(ticker, daily_df, kr_s):
    """Process one ticker for daily approach A (v3)."""
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) == 0:
        return {"results": {}, "positions": []}

    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)

    log_ret = np.zeros(n)
    log_ret[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    # Calendar/RF for OOS calc_trading_sharpe
    cal_days = np.ones(n)
    for i in range(1, n):
        cal_days[i] = (dates[i] - dates[i - 1]).days
    rf_raw = kr_s.reindex(dates, method="ffill").values
    rf_daily = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0

    ind = compute_base(close, high, low, open_arr, volume, is_hourly=False)

    # Caches
    _sg = SIGNAL_GRIDS
    sma_windows = sorted(set(
        _sg["S1"].get("ma_window", []) + _sg["S2"].get("bb_window", [])))
    sma_cache = precompute_sma_cache(close, sma_windows)

    dc_windows = _sg["S3"].get("dc_window", [])
    dc_cache = precompute_donchian_cache(high, low, dc_windows)

    st_periods = _sg["S4"].get("atr_period", [])
    st_mults = _sg["S4"].get("multiplier", [])
    st_cache = precompute_supertrend_cache(high, low, close, st_periods, st_mults)

    vwap_windows = _sg["S6"].get("vwap_window", [])
    vwap_cache = precompute_vwap_cache(close, high, low, volume, vwap_windows)

    # Pivots
    pivot_data = {}
    for pv in ["classic", "camarilla", "fibonacci"]:
        pivot_data[pv] = calc_pivot_daily(high, low, close, pv)

    ann = sqrt(252)
    bpy = 252

    _pctile_cache.clear()

    results = {}
    position_rows = []

    for sid in STRATEGY_IDS:
        sname = STRATEGY_NAMES[sid]

        a_results, a_params, a_trades = approach_a_daily(
            sid, close, high, low, volume, open_arr, ind,
            sma_cache, dc_cache, st_cache, vwap_cache,
            pivot_data, dates, log_ret)

        # OOS evaluation with calc_trading_sharpe
        for year, pos_arr in a_results.items():
            if year == 2026:
                t_end = n
            else:
                _, t_end = _year_bounds(dates, year + 1)
            t_start, _ = _year_bounds(dates, year)
            oos_mask = np.zeros(n, dtype=bool)
            oos_mask[max(t_start, 0):min(t_end, n)] = True

            comm_val = 0.0005
            sh_oos, info = calc_trading_sharpe(pos_arr, close, log_ret, dates,
                                                rf_daily, cal_days, oos_mask, comm_val)

            key = (sid, "daily", "A", year)
            results[key] = {
                "strategy": sname, "tf": "daily", "approach": "A",
                "test_year": year, "ticker": ticker,
                "sharpe": round(sh_oos, 4),
                "total_ret": info.get("total_ret", 0.0),
                "vol": info.get("vol", 0.0),
                "exposure": info.get("exposure", 0.0),
                "wr": info.get("wr", 0.0),
                "tpy": info.get("tpy", 0.0),
                "rf": info.get("rf", 0.0),
            }

        # Store positions
        for year, pos_arr in a_results.items():
            if year == 2026:
                test_end = n
            else:
                _, test_end = _year_bounds(dates, year + 1)
            _, test_start = _year_bounds(dates, year)
            for t in range(max(test_start, 0), min(test_end, n)):
                if abs(pos_arr[t]) > 1e-8 or True:  # store all bars for FDR
                    position_rows.append((
                        dates[t], sname, "daily", ticker, "A", year,
                        round(pos_arr[t], 6),
                        round(pos_arr[t] * log_ret[t], 8)
                    ))

    return {"results": results, "positions": position_rows}


# ═══════════════════════════════════════════════════
# Part 2: Position Sizing + Exact Daily RF
# ═══════════════════════════════════════════════════

def load_key_rate():
    kr = pd.read_parquet(KR_PATH)
    kr["date"] = pd.to_datetime(kr["date"])
    return kr.set_index("date")["key_rate"].sort_index()


def compute_portfolio_returns(signal, ret_per_unit, weights, cap, dates, kr_s):
    T, N = signal.shape
    if weights.ndim == 1:
        w = np.tile(weights, (T, 1))
    else:
        w = weights.copy()
    raw_exp = (signal.values * w).sum(axis=1)
    scale = np.where(raw_exp > cap, cap / np.maximum(raw_exp, 1e-12), 1.0)
    eff_w = signal.values * w * scale[:, None]
    trading_ret = (eff_w * ret_per_unit.values).sum(axis=1)
    exposure = eff_w.sum(axis=1)
    rf_daily = kr_s.reindex(pd.to_datetime(dates), method="ffill").values / BPY
    rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)[:T]
    fdr = np.maximum(0, 1 - exposure) * rf_daily
    lev_cost = np.maximum(0, exposure - 1.0) * (rf_daily * BPY + 0.02) / BPY
    total_ret = trading_ret + fdr - lev_cost
    return total_ret, exposure, trading_ret, fdr


def compute_excess_sharpe_exact(daily_rets, dates, kr_s):
    if len(daily_rets) < 20:
        return 0.0, 0.0, 0.0, 0.0
    rf_daily = kr_s.reindex(pd.to_datetime(dates), method="ffill").values / BPY
    rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)[:len(daily_rets)]
    excess_daily = daily_rets - rf_daily
    ann_excess = np.mean(excess_daily) * BPY
    ann_vol = np.std(daily_rets, ddof=1) * sqrt(BPY)
    cum = np.prod(1 + daily_rets)
    n_years = len(daily_rets) / BPY
    ann_ret = cum ** (1 / max(n_years, 0.01)) - 1
    ann_rf = np.mean(rf_daily) * BPY
    sharpe = ann_excess / ann_vol if ann_vol > 1e-12 else 0.0
    return sharpe, ann_ret, ann_vol, ann_rf


def compute_maxdd(daily_rets):
    cum = np.cumprod(1 + pd.Series(daily_rets))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1)
    return dd.min() * 100


def build_strategy_data(daily_pos, strat, kr_s, comm):
    sub = daily_pos[daily_pos["strategy"] == strat].copy()
    tickers = sorted(sub["ticker"].unique())
    ticker_rets = {}; ticker_pos_dict = {}
    for tkr in tickers:
        g = sub[sub["ticker"] == tkr].sort_values("date")
        pos = g["position"].values
        gross = g["daily_gross_return"].values
        dpos = np.diff(pos, prepend=0.0)
        net = gross - np.abs(dpos) * comm
        ticker_rets[tkr] = pd.Series(net, index=g["date"].values)
        ticker_pos_dict[tkr] = pd.Series(pos, index=g["date"].values)
    ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
    pos_df = pd.DataFrame(ticker_pos_dict).sort_index().fillna(0.0)
    dates = ret_df.index
    signal = (pos_df.abs() > 1e-8).astype(float)
    pos_abs = pos_df.abs().replace(0, np.nan)
    ret_per_unit = (ret_df / pos_abs).fillna(0.0)
    return dates, ret_df, pos_df, signal, ret_per_unit, tickers


def ps1_ew(signal, rpu, dates, kr_s):
    w = np.full(signal.shape[1], 1.0 / N_TICKERS)
    return compute_portfolio_returns(signal, rpu, w, cap=1.0, dates=dates, kr_s=kr_s)


def ps2_best(signal, rpu, dates, kr_s, train_mask):
    W_GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07,
              0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    CAP_GRID = [0.25, 0.50, 0.75, 1.0]
    N = signal.shape[1]; best_sh = -999; best_w = 1/N_TICKERS; best_cap = 1.0
    for wf in W_GRID:
        for cap in CAP_GRID:
            w = np.full(N, wf)
            tr, exp, _, _ = compute_portfolio_returns(signal, rpu, w, cap, dates, kr_s)
            if exp[train_mask].max() > 1.0 + 1e-6:
                continue
            sh, _, _, _ = compute_excess_sharpe_exact(tr[train_mask], dates[train_mask], kr_s)
            if sh > best_sh:
                best_sh = sh; best_w = wf; best_cap = cap
    w = np.full(N, best_w)
    ret = compute_portfolio_returns(signal, rpu, w, best_cap, dates, kr_s)
    return ret, f"w={best_w},cap={best_cap}"


def ps3_kelly(signal, rpu, dates, kr_s, train_mask, ret_df):
    FRAC_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    T, N = signal.shape; best_sh = -999; best_frac = 0.25
    months = pd.Series(dates).dt.to_period("M")
    month_starts = [0]
    prev_m = months.iloc[0]
    for i in range(1, len(months)):
        if months.iloc[i] != prev_m:
            month_starts.append(i); prev_m = months.iloc[i]
    for frac in FRAC_GRID:
        w_matrix = np.full((T, N), 1.0 / N_TICKERS)
        for k, start in enumerate(month_starts):
            end = month_starts[k+1] if k+1 < len(month_starts) else T
            if start < 60:
                continue
            hist = ret_df.iloc[:start].values
            rf_avg = kr_s.reindex(dates[:start], method="ffill").mean()
            mu = np.mean(hist, axis=0) * BPY - rf_avg
            var = np.var(hist, axis=0, ddof=1) * BPY
            var = np.where(var < 1e-12, 1e-12, var)
            w_kelly = np.clip(frac * np.clip(mu / var, 0, 2.0), 0, 0.5)
            for t in range(start, end):
                w_matrix[t] = w_kelly
        tr, exp, _, _ = compute_portfolio_returns(signal, rpu, w_matrix, 1.0, dates, kr_s)
        if exp[train_mask].max() > 1.0 + 1e-6:
            continue
        sh, _, _, _ = compute_excess_sharpe_exact(tr[train_mask], dates[train_mask], kr_s)
        if sh > best_sh:
            best_sh = sh; best_frac = frac
    # Recompute with best
    w_matrix = np.full((T, N), 1.0 / N_TICKERS)
    for k, start in enumerate(month_starts):
        end = month_starts[k+1] if k+1 < len(month_starts) else T
        if start < 60:
            continue
        hist = ret_df.iloc[:start].values
        rf_avg = kr_s.reindex(dates[:start], method="ffill").mean()
        mu = np.mean(hist, axis=0) * BPY - rf_avg
        var = np.var(hist, axis=0, ddof=1) * BPY
        var = np.where(var < 1e-12, 1e-12, var)
        w_kelly = np.clip(best_frac * np.clip(mu / var, 0, 2.0), 0, 0.5)
        for t in range(start, end):
            w_matrix[t] = w_kelly
    ret = compute_portfolio_returns(signal, rpu, w_matrix, 1.0, dates, kr_s)
    return ret, f"frac={best_frac}"


def ps4_sigprop(signal, rpu, dates, kr_s, train_mask, pos_df):
    WBASE_GRID = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    T, N = signal.shape
    strength = pos_df.abs().values
    p99 = np.percentile(strength[strength > 0], 99) if (strength > 0).any() else 1.0
    strength_norm = np.clip(strength / max(p99, 1e-8), 0, 1.0)
    best_sh = -999; best_wb = 0.10
    for wb in WBASE_GRID:
        wm = signal.values * strength_norm * wb
        tr, exp, _, _ = compute_portfolio_returns(signal, rpu, wm, 1.0, dates, kr_s)
        sh, _, _, _ = compute_excess_sharpe_exact(tr[train_mask], dates[train_mask], kr_s)
        if sh > best_sh:
            best_sh = sh; best_wb = wb
    wm = signal.values * strength_norm * best_wb
    ret = compute_portfolio_returns(signal, rpu, wm, 1.0, dates, kr_s)
    return ret, f"wb={best_wb}"


def ps5_mv(signal, rpu, dates, kr_s, train_mask, ret_df):
    WMAX_GRID = [0.10, 0.15, 0.20, 0.25]; CAP_GRID = [0.50, 0.75, 1.0]
    T, N = signal.shape; best_sh = -999; best_wmax = 0.20; best_cap = 1.0
    months = pd.Series(dates).dt.to_period("M")
    month_starts = [0]
    prev_m = months.iloc[0]
    for i in range(1, len(months)):
        if months.iloc[i] != prev_m:
            month_starts.append(i); prev_m = months.iloc[i]
    for wmax in WMAX_GRID:
        for cap in CAP_GRID:
            wm = np.full((T, N), 1.0 / N_TICKERS)
            for k, start in enumerate(month_starts):
                end = month_starts[k+1] if k+1 < len(month_starts) else T
                if start < N + 30:
                    continue
                hist = ret_df.iloc[:start].values
                mu = np.mean(hist, axis=0)
                try:
                    cov = LedoitWolf().fit(hist).covariance_
                except:
                    continue
                def neg_sh(w):
                    return -(w @ mu) / sqrt(w @ cov @ w + 1e-15)
                try:
                    res = minimize(neg_sh, np.ones(N)/N, method="SLSQP",
                                   bounds=[(0, wmax)]*N,
                                   constraints={"type": "ineq", "fun": lambda w: cap - w.sum()},
                                   options={"maxiter": 200})
                    w_opt = res.x if res.success else np.ones(N)/N_TICKERS
                except:
                    w_opt = np.ones(N)/N_TICKERS
                for t in range(start, end):
                    wm[t] = w_opt
            tr, exp, _, _ = compute_portfolio_returns(signal, rpu, wm, cap, dates, kr_s)
            sh, _, _, _ = compute_excess_sharpe_exact(tr[train_mask], dates[train_mask], kr_s)
            if sh > best_sh:
                best_sh = sh; best_wmax = wmax; best_cap = cap
    # Recompute
    wm = np.full((T, N), 1.0 / N_TICKERS)
    for k, start in enumerate(month_starts):
        end = month_starts[k+1] if k+1 < len(month_starts) else T
        if start < N + 30:
            continue
        hist = ret_df.iloc[:start].values
        mu = np.mean(hist, axis=0)
        try:
            cov = LedoitWolf().fit(hist).covariance_
        except:
            continue
        def neg_sh(w):
            return -(w @ mu) / sqrt(w @ cov @ w + 1e-15)
        try:
            res = minimize(neg_sh, np.ones(N)/N, method="SLSQP",
                           bounds=[(0, best_wmax)]*N,
                           constraints={"type": "ineq", "fun": lambda w: best_cap - w.sum()},
                           options={"maxiter": 200})
            w_opt = res.x if res.success else np.ones(N)/N_TICKERS
        except:
            w_opt = np.ones(N)/N_TICKERS
        for t in range(start, end):
            wm[t] = w_opt
    ret = compute_portfolio_returns(signal, rpu, wm, best_cap, dates, kr_s)
    return ret, f"wmax={best_wmax},cap={best_cap}"


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("DAILY APPROACH A v3 + POSITION SIZING", flush=True)
    print("Train: trading_sharpe | OOS: calc_trading_sharpe | MinTrades=20 | SubPeriod", flush=True)
    print("=" * 80, flush=True)

    # Print grid sizes
    for sid in STRATEGY_IDS:
        sg = expand_grid(SIGNAL_GRIDS[sid])
        cat = CATEGORY[sid]
        if cat == "Trend":
            rm = TREND_RM_GRID
        else:
            rm = expand_grid(RM_GRIDS_V3[(cat, "daily")])
        print(f"  {sid}: {len(sg)} sig x {len(rm)} RM", flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    daily_df, hourly_df, vpred_df = load_data()
    kr_s = load_key_rate()

    warmup_numba()

    # Process tickers
    print(f"\nProcessing {len(TICKERS)} tickers...", flush=True)
    all_results = {}
    all_positions = []

    for i, ticker in enumerate(TICKERS):
        t1 = time.time()
        result = process_ticker_daily_a(ticker, daily_df, kr_s)
        all_results[ticker] = result["results"]
        all_positions.extend(result["positions"])
        elapsed = time.time() - t1
        n_met = len(result["results"])
        print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {n_met} metrics ({elapsed:.0f}s)", flush=True)

    # Save positions
    pos_cols = ["date", "strategy", "tf", "ticker", "approach", "test_year",
                "position", "daily_gross_return"]
    pos_df_all = pd.DataFrame(all_positions, columns=pos_cols)
    pos_df_all.to_parquet(OUT_DIR / "daily_a_v3_positions.parquet", index=False)
    print(f"\nPositions: {len(pos_df_all):,} rows -> daily_a_v3_positions.parquet", flush=True)

    # Save raw results
    rows = []
    for ticker, ticker_results in all_results.items():
        for key, met in ticker_results.items():
            rows.append({"ticker": ticker, **met})
    if rows:
        res_df = pd.DataFrame(rows)
        res_df.to_csv(OUT_DIR / "daily_a_v3_results.csv", index=False)
        print(f"Results: {len(res_df)} rows -> daily_a_v3_results.csv", flush=True)

    # ── Position Sizing ──
    print("\n" + "=" * 80, flush=True)
    print("POSITION SIZING + EXACT DAILY RF", flush=True)
    print("=" * 80, flush=True)

    STRATEGIES = [STRATEGY_NAMES[s] for s in STRATEGY_IDS]
    ps_results = []

    for comm_label, comm_val in COMM_LEVELS.items():
        for strat in STRATEGIES:
            dates, ret_df, pos_df, signal, rpu, tickers = \
                build_strategy_data(pos_df_all, strat, kr_s, comm_val)

            if len(dates) < 100:
                continue

            years = pd.to_datetime(dates).year
            train_mask = years < 2022
            oos_mask = np.isin(years, OOS_YEARS)
            if oos_mask.sum() < 20:
                continue

            oos_dates = dates[oos_mask]

            # All 5 PS methods
            ps_funcs = [
                ("PS-1", lambda: (ps1_ew(signal, rpu, dates, kr_s), "1/17")),
                ("PS-2", lambda: ps2_best(signal, rpu, dates, kr_s, train_mask)),
                ("PS-3", lambda: ps3_kelly(signal, rpu, dates, kr_s, train_mask, ret_df)),
                ("PS-4", lambda: ps4_sigprop(signal, rpu, dates, kr_s, train_mask, pos_df)),
                ("PS-5", lambda: ps5_mv(signal, rpu, dates, kr_s, train_mask, ret_df)),
            ]

            for ps_name, func in ps_funcs:
                try:
                    result, params_str = func()
                    total_ret, exp, trading_ret, fdr_ret = result
                    oos_ret = total_ret[oos_mask]
                    oos_exp = exp[oos_mask]
                    sharpe, ann_ret, ann_vol, ann_rf = compute_excess_sharpe_exact(
                        oos_ret, oos_dates, kr_s)
                    mdd = compute_maxdd(oos_ret)

                    # Trades/yr and WinRate
                    n_trades = 0; n_wins = 0
                    for tkr in tickers:
                        if tkr not in pos_df.columns:
                            continue
                        p = pos_df[tkr].values[oos_mask]
                        r = ret_df[tkr].values[oos_mask]
                        in_t = False; t_ret = 0
                        for j in range(len(p)):
                            if abs(p[j]) > 1e-8 and not in_t:
                                in_t = True; t_ret = 0
                            if in_t:
                                t_ret += r[j]
                                if j == len(p)-1 or abs(p[j+1] if j+1 < len(p) else 0) < 1e-8:
                                    n_trades += 1
                                    if t_ret > 0:
                                        n_wins += 1
                                    in_t = False
                    n_yrs = oos_mask.sum() / BPY
                    tpy = n_trades / max(n_yrs, 0.01)
                    wr = (n_wins / n_trades * 100) if n_trades > 0 else 0

                    ps_results.append({
                        "strategy": strat, "comm": comm_label, "ps": ps_name,
                        "params": params_str,
                        "sharpe": round(sharpe, 4),
                        "ann_ret_pct": round(ann_ret * 100, 2),
                        "ann_vol_pct": round(ann_vol * 100, 2),
                        "ann_rf_pct": round(ann_rf * 100, 2),
                        "mdd_pct": round(mdd, 2),
                        "avg_exp_pct": round(np.mean(oos_exp) * 100, 1),
                        "trading_ret_pct": round(np.mean(trading_ret[oos_mask]) * BPY * 100, 2),
                        "fdr_ret_pct": round(np.mean(fdr_ret[oos_mask]) * BPY * 100, 2),
                        "trades_per_yr": round(tpy, 1),
                        "win_rate_pct": round(wr, 1),
                    })

                    if ps_name == "PS-1":
                        print(f"  {strat:>16s} [{comm_label}] {ps_name}: "
                              f"Sharpe={sharpe:.2f}, Ret={ann_ret*100:.1f}%, "
                              f"Exp={np.mean(oos_exp)*100:.1f}%, Tr/yr={tpy:.0f}", flush=True)
                except Exception as e:
                    print(f"  {strat} {ps_name}: ERROR -- {e}", flush=True)

    ps_df = pd.DataFrame(ps_results)
    ps_df.to_csv(OUT_DIR / "daily_a_v3_ps_results.csv", index=False)

    # ── Final Table ──
    print("\n" + "=" * 80, flush=True)
    print("FINAL TABLE: Best PS per strategy (OOS 2022-2025)", flush=True)
    print("=" * 80, flush=True)

    for comm_label in COMM_LEVELS:
        print(f"\n  Commission: {comm_label}", flush=True)
        print(f"  {'Strategy':>16s}  {'ExSharpe':>8s}  {'PS':>4s}  {'Ret%':>6s}  "
              f"{'MDD%':>6s}  {'Exp%':>5s}  {'Tr/yr':>5s}  {'WR%':>5s}  {'TradRet':>7s}  {'FDR':>5s}", flush=True)
        for strat in STRATEGIES:
            sub = ps_df[(ps_df["strategy"] == strat) & (ps_df["comm"] == comm_label)]
            if len(sub) == 0:
                continue
            best = sub.loc[sub["sharpe"].idxmax()]
            print(f"  {strat:>16s}  {best['sharpe']:8.2f}  {best['ps']:>4s}  "
                  f"{best['ann_ret_pct']:6.1f}  {best['mdd_pct']:6.1f}  "
                  f"{best['avg_exp_pct']:5.1f}  {best['trades_per_yr']:5.0f}  "
                  f"{best['win_rate_pct']:5.1f}  {best['trading_ret_pct']:7.2f}  "
                  f"{best['fdr_ret_pct']:5.2f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
