#!/usr/bin/env python3
"""
signal_strength_hourly_a.py — Signal Strength Framework + Exact Daily RF

Part 1: Recompute ALL strategies with exact daily key rate
Part 2: Signal Strength metrics (5 per strategy)
Part 3: Grid search: threshold × sizing_mode × PS
Part 4: Output tables

Usage:  python3 signal_strength_hourly_a.py
"""
import os, sys, warnings, time
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from bisect import insort, bisect_left
from itertools import product

# ── Paths ──
BASE = Path(__file__).resolve().parent.parent
POS_PATH    = BASE / "results/final/strategies/walkforward_v4/hourly_a_scaled/hourly_a_scaled_positions.parquet"
OHLCV_PATH  = BASE / "results/final/strategies/data/ohlcv_hourly.parquet"
KR_PATH     = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
OUT_DIR     = BASE / "results/final/strategies/walkforward_v4/hourly_a_scaled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
N_TICKERS = 17
BPY = 252
OOS_YEARS = [2022, 2023, 2024, 2025]
COMM_LEVELS = [0.0004, 0.0005]

# ── Import shared functions ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies_walkforward import (
    calc_sma, calc_std, calc_rsi, calc_atr, calc_adx,
    calc_supertrend, compute_base,
    precompute_sma_cache, precompute_donchian_cache,
    precompute_supertrend_cache, precompute_vwap_cache,
    precompute_session_vwap,
    calc_pivot_daily, compute_daily_pivots_for_hourly,
    SIGNAL_GRIDS_HOURLY, WARMUP_HOURLY,
)


# ═════════════════════════════════════════════════════════════════
# Part 1: Exact Daily RF utilities
# ═════════════════════════════════════════════════════════════════

def load_key_rate():
    kr = pd.read_parquet(KR_PATH)
    kr["date"] = pd.to_datetime(kr["date"])
    return kr.set_index("date")["key_rate"].sort_index()


def compute_portfolio_returns(signal, ret_per_unit, weights, cap, dates, kr_s):
    """Portfolio returns with exact daily FDR and leverage cost."""
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

    # Exact daily rf
    rf_daily = kr_s.reindex(pd.to_datetime(dates), method="ffill").values / BPY
    rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)[:T]
    fdr = np.maximum(0, 1 - exposure) * rf_daily
    lev_cost = np.maximum(0, exposure - 1.0) * (rf_daily * BPY + 0.02) / BPY
    total_ret = trading_ret + fdr - lev_cost
    return total_ret, exposure, trading_ret, fdr


def compute_excess_sharpe_exact(daily_rets, dates, kr_s):
    """Excess Sharpe with exact daily rf subtraction."""
    if len(daily_rets) < 20:
        return 0.0, 0.0, 0.0, 0.0
    # Daily excess
    rf_daily = kr_s.reindex(pd.to_datetime(dates), method="ffill").values / BPY
    rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)[:len(daily_rets)]
    excess_daily = daily_rets - rf_daily

    ann_excess = np.mean(excess_daily) * BPY
    ann_vol = np.std(daily_rets, ddof=1) * sqrt(BPY)

    # Also compute total metrics
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


# ═════════════════════════════════════════════════════════════════
# Part 2: Signal Strength Metrics
# ═════════════════════════════════════════════════════════════════

def expanding_pctile_rank(arr):
    """Percentile rank on expanding window. O(n log n)."""
    n = len(arr)
    result = np.full(n, np.nan)
    sorted_vals = []
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        val = arr[i]
        insort(sorted_vals, val)
        k = len(sorted_vals)
        if k > 1:
            rank = bisect_left(sorted_vals, val)
            result[i] = rank / (k - 1)
        else:
            result[i] = 0.5
    return result


def normalize_metrics(raw_metrics, mask):
    """Normalize each metric to [0,1] via expanding percentile rank.
    Only compute at bars where mask==True. Returns composite [0,1]."""
    n = len(mask)
    n_metrics = len(raw_metrics)
    normalized = np.zeros((n_metrics, n))

    for j, raw in enumerate(raw_metrics):
        # Compute expanding percentile rank for all bars
        pctile = expanding_pctile_rank(raw)
        normalized[j] = np.where(np.isnan(pctile), 0.5, pctile)

    # Composite = mean of normalized metrics where mask is on
    composite = np.mean(normalized, axis=0)
    # Zero out where no signal
    composite[~mask] = 0.0
    return composite


def compute_strength_s1(close, ind, sma_cache, sig, sig_params):
    """S1 Mean Reversion: 5 metrics at signal bars."""
    n = len(close)
    mask = sig != 0
    w = sig_params.get("ma_window", 180)
    sma, std = sma_cache.get(w, (calc_sma(close, w), calc_std(close, w)))

    # 1. |z-score|
    z = np.abs((close - sma) / np.where(std > 1e-12, std, 1e-12))
    # 2. |deviation / ATR|
    atr = ind["atr14"]
    atr_safe = np.where(np.isnan(atr) | (atr < 1e-12), 1e-12, atr)
    dev_atr = np.abs(close - sma) / atr_safe
    # 3. RSI extremity: |RSI - 50| / 50
    rsi = calc_rsi(close, 14)
    rsi_ext = np.abs(rsi - 50.0) / 50.0
    rsi_ext = np.where(np.isnan(rsi_ext), 0.0, rsi_ext)
    # 4. Volume ratio
    vol = ind["_volume"]
    vol_avg = ind["vol_sma20"]
    vol_avg_safe = np.where(np.isnan(vol_avg) | (vol_avg < 1), 1.0, vol_avg)
    vol_ratio = vol / vol_avg_safe
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    # 5. Speed of deviation: |close-SMA| now vs 9 bars ago
    dev_now = np.abs(close - sma)
    dev_lag = np.full(n, np.nan)
    dev_lag[9:] = dev_now[:-9]
    speed = np.where(np.isnan(dev_lag) | (dev_lag < 1e-12), 1.0,
                     dev_now / np.maximum(dev_lag, 1e-12))

    return normalize_metrics([z, dev_atr, rsi_ext, vol_ratio, speed], mask)


def compute_strength_s2(close, ind, sma_cache, sig, sig_params):
    """S2 Bollinger Bands: 5 metrics at signal bars."""
    n = len(close)
    mask = sig != 0
    w = sig_params.get("bb_window", 180)
    sma, std = sma_cache.get(w, (calc_sma(close, w), calc_std(close, w)))

    # 1. |%B - 0.5| where %B = (close - lower) / (upper - lower)
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    bw = upper - lower
    bw_safe = np.where(np.isnan(bw) | (bw < 1e-12), 1e-12, bw)
    pct_b = (close - lower) / bw_safe
    pct_b_dev = np.abs(pct_b - 0.5)
    # 2. Bandwidth: (upper - lower) / sma
    sma_safe = np.where(np.abs(sma) < 1e-12, 1e-12, sma)
    bandwidth = bw / sma_safe
    bandwidth = np.where(np.isnan(bandwidth), 0.0, bandwidth)
    # 3. Squeeze duration: bars since bandwidth < bw_p30
    bw_p30 = ind.get("bw_p30", np.full(n, 0.01))
    squeeze = np.zeros(n)
    cnt = 0
    for i in range(n):
        if not np.isnan(bandwidth[i]) and not np.isnan(bw_p30[i]) and bandwidth[i] < bw_p30[i]:
            cnt += 1
        else:
            cnt = 0
        squeeze[i] = cnt
    # 4. Volume spike
    vol = ind["_volume"]
    vol_avg = ind["vol_sma20"]
    vol_avg_safe = np.where(np.isnan(vol_avg) | (vol_avg < 1), 1.0, vol_avg)
    vol_ratio = vol / vol_avg_safe
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    # 5. ATR ratio: ATR / avg_ATR_20
    atr = ind["atr14"]
    atr_avg = calc_sma(atr, 20)
    atr_avg_safe = np.where(np.isnan(atr_avg) | (atr_avg < 1e-12), 1e-12, atr_avg)
    atr_ratio = atr / atr_avg_safe
    atr_ratio = np.where(np.isnan(atr_ratio), 1.0, atr_ratio)

    return normalize_metrics([pct_b_dev, bandwidth, squeeze, vol_ratio, atr_ratio], mask)


def compute_strength_s3(close, high, low, ind, dc_cache, sig, sig_params):
    """S3 Donchian Channel: 5 metrics at signal bars."""
    n = len(close)
    mask = sig != 0
    w = sig_params.get("dc_window", 90)
    hc, lc = dc_cache.get(w, (pd.Series(high).rolling(w).max().values,
                               pd.Series(low).rolling(w).min().values))
    atr = ind["atr14"]
    atr_safe = np.where(np.isnan(atr) | (atr < 1e-12), 1e-12, atr)

    # 1. Breakout magnitude: |close - channel_bound| / ATR
    dist_high = np.abs(close - hc)
    dist_low = np.abs(close - lc)
    breakout_mag = np.minimum(dist_high, dist_low) / atr_safe
    breakout_mag = np.where(np.isnan(breakout_mag), 0.0, breakout_mag)
    # 2. Channel width / ATR
    ch_width = (hc - lc) / atr_safe
    ch_width = np.where(np.isnan(ch_width), 0.0, ch_width)
    # 3. ADX
    adx = ind["adx14"]
    adx_val = np.where(np.isnan(adx), 0.0, adx)
    # 4. Volume at breakout
    vol = ind["_volume"]
    vol_avg = ind["vol_sma20"]
    vol_avg_safe = np.where(np.isnan(vol_avg) | (vol_avg < 1), 1.0, vol_avg)
    vol_ratio = vol / vol_avg_safe
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    # 5. Channel contraction: previous width / current width
    ch_w_prev = np.full(n, np.nan)
    ch_w_prev[9:] = ch_width[:-9]
    contraction = np.where(
        np.isnan(ch_w_prev) | (ch_width < 1e-12), 1.0,
        ch_w_prev / np.maximum(ch_width, 1e-12))

    return normalize_metrics([breakout_mag, ch_width, adx_val, vol_ratio, contraction], mask)


def compute_strength_s4(close, high, low, ind, st_cache, sig, sig_params):
    """S4 Supertrend: 5 metrics at signal bars."""
    n = len(close)
    mask = sig != 0
    ap = sig_params.get("atr_period", 90)
    mult = sig_params.get("multiplier", 3.0)
    st_line, st_dir = st_cache.get((ap, mult), calc_supertrend(high, low, close, ap, mult))
    atr = ind["atr14"]
    atr_safe = np.where(np.isnan(atr) | (atr < 1e-12), 1e-12, atr)

    # 1. |price - supertrend_line| / ATR
    dist_st = np.abs(close - st_line) / atr_safe
    dist_st = np.where(np.isnan(dist_st), 0.0, dist_st)
    # 2. ADX
    adx = ind["adx14"]
    adx_val = np.where(np.isnan(adx), 0.0, adx)
    # 3. Trend duration: bars in current trend direction
    trend_dur = np.zeros(n)
    cnt = 0
    prev_d = 0.0
    for i in range(n):
        if st_dir[i] == prev_d:
            cnt += 1
        else:
            cnt = 1
            prev_d = st_dir[i]
        trend_dur[i] = cnt
    # 4. Volume
    vol = ind["_volume"]
    vol_avg = ind["vol_sma20"]
    vol_avg_safe = np.where(np.isnan(vol_avg) | (vol_avg < 1), 1.0, vol_avg)
    vol_ratio = vol / vol_avg_safe
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    # 5. Price momentum: return over 9 bars
    momentum = np.full(n, 0.0)
    momentum[9:] = (close[9:] - close[:-9]) / np.maximum(close[:-9], 1e-8)

    return normalize_metrics([dist_st, adx_val, trend_dur, vol_ratio, momentum], mask)


def compute_strength_s5(close, ind, pivot_data, sig, sig_params):
    """S5 Pivot Points: 5 metrics at signal bars."""
    n = len(close)
    mask = sig != 0
    ptype = sig_params.get("pivot_type", "classic")
    P, S1, R1, S2, R2, S3, R3 = pivot_data.get(ptype, pivot_data["classic"])

    # 1. Level depth: how far from pivot in S/R units
    # Distance to nearest touched level, normalized by pivot range
    range_pr = np.abs(R3 - S3)
    range_safe = np.where(np.isnan(range_pr) | (range_pr < 1e-12), 1e-12, range_pr)
    dist_pivot = np.abs(close - P) / range_safe
    dist_pivot = np.where(np.isnan(dist_pivot), 0.0, dist_pivot)
    # 2. |price - pivot| / (R3 - S3)  [same as above but signed context]
    # Better: which level is touched (S3/R3=1.0, S2/R2=0.6, S1/R1=0.3, P=0.0)
    level_score = np.zeros(n)
    for i in range(n):
        if np.isnan(P[i]):
            continue
        c = close[i]
        if c <= S3[i] if not np.isnan(S3[i]) else False:
            level_score[i] = 1.0
        elif c >= R3[i] if not np.isnan(R3[i]) else False:
            level_score[i] = 1.0
        elif c <= S2[i] if not np.isnan(S2[i]) else False:
            level_score[i] = 0.6
        elif c >= R2[i] if not np.isnan(R2[i]) else False:
            level_score[i] = 0.6
        elif c <= S1[i] if not np.isnan(S1[i]) else False:
            level_score[i] = 0.3
        elif c >= R1[i] if not np.isnan(R1[i]) else False:
            level_score[i] = 0.3
    # 3. Volume ratio
    vol = ind["_volume"]
    vol_avg = ind["vol_sma20"]
    vol_avg_safe = np.where(np.isnan(vol_avg) | (vol_avg < 1), 1.0, vol_avg)
    vol_ratio = vol / vol_avg_safe
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    # 4. ATR ratio: ATR / avg_ATR_20
    atr = ind["atr14"]
    atr_avg = calc_sma(atr, 20)
    atr_avg_safe = np.where(np.isnan(atr_avg) | (atr_avg < 1e-12), 1e-12, atr_avg)
    atr_ratio = atr / atr_avg_safe
    atr_ratio = np.where(np.isnan(atr_ratio), 1.0, atr_ratio)
    # 5. Previous day range / ATR (volatility context)
    atr_safe = np.where(np.isnan(atr) | (atr < 1e-12), 1e-12, atr)
    range_atr = range_pr / atr_safe
    range_atr = np.where(np.isnan(range_atr), 1.0, range_atr)

    return normalize_metrics([dist_pivot, level_score, vol_ratio, atr_ratio, range_atr], mask)


def compute_strength_s6(close, high, low, volume, ind, vwap_cache, sig, sig_params,
                         session_vwap_data=None):
    """S6 VWAP: 5 metrics at signal bars."""
    n = len(close)
    mask = sig != 0
    vw = sig_params.get("vwap_window", 0)
    if vw == 0 and session_vwap_data is not None:
        vwap, dev = session_vwap_data
    else:
        vwap, dev = vwap_cache.get(vw, (np.full(n, np.nan), np.full(n, 1e-8)))

    dev_safe = np.where(np.isnan(dev) | (dev < 1e-12), 1e-12, dev)

    # 1. |price - VWAP| / σ_VWAP
    vwap_zscore = np.abs(close - vwap) / dev_safe
    vwap_zscore = np.where(np.isnan(vwap_zscore), 0.0, vwap_zscore)
    # 2. VWAP slope (9-bar)
    vwap_slope = np.full(n, 0.0)
    vwap_safe = np.where(np.isnan(vwap) | (np.abs(vwap) < 1e-8), 1e-8, vwap)
    vwap_slope[9:] = (vwap[9:] - vwap[:-9]) / np.abs(vwap_safe[:-9])
    vwap_slope = np.abs(vwap_slope)
    # 3. Volume ratio
    vol = ind["_volume"]
    vol_avg = ind["vol_sma20"]
    vol_avg_safe = np.where(np.isnan(vol_avg) | (vol_avg < 1), 1.0, vol_avg)
    vol_ratio = vol / vol_avg_safe
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    # 4. RSI extremity
    rsi = calc_rsi(close, 14)
    rsi_ext = np.abs(rsi - 50.0) / 50.0
    rsi_ext = np.where(np.isnan(rsi_ext), 0.0, rsi_ext)
    # 5. ATR ratio
    atr = ind["atr14"]
    atr_avg = calc_sma(atr, 20)
    atr_avg_safe = np.where(np.isnan(atr_avg) | (atr_avg < 1e-12), 1e-12, atr_avg)
    atr_ratio = atr / atr_avg_safe
    atr_ratio = np.where(np.isnan(atr_ratio), 1.0, atr_ratio)

    return normalize_metrics([vwap_zscore, vwap_slope, vol_ratio, rsi_ext, atr_ratio], mask)


# ═════════════════════════════════════════════════════════════════
# Part 3: Strength-filtered portfolio construction
# ═════════════════════════════════════════════════════════════════

def identify_trades(position):
    """Identify trade boundaries: list of (entry_bar, exit_bar) tuples."""
    trades = []
    in_trade = False
    entry = 0
    for i in range(len(position)):
        if not in_trade and abs(position[i]) > 1e-8:
            entry = i
            in_trade = True
        elif in_trade and abs(position[i]) < 1e-8:
            trades.append((entry, i))
            in_trade = False
    if in_trade:
        trades.append((entry, len(position)))
    return trades


def filter_positions_by_strength(position, gross_ret, strength, threshold, mode="binary"):
    """Filter positions based on signal strength at entry.
    mode='binary': keep/discard trade.
    mode='proportional': scale position by strength.
    Returns: filtered_position, filtered_gross_ret
    """
    pos_out = position.copy()
    gross_out = gross_ret.copy()
    trades = identify_trades(position)

    for entry, exit_bar in trades:
        entry_strength = strength[entry] if entry < len(strength) else 0.0
        if np.isnan(entry_strength):
            entry_strength = 0.0

        if entry_strength < threshold:
            # Skip this trade
            pos_out[entry:exit_bar] = 0.0
            gross_out[entry:exit_bar] = 0.0
        elif mode == "proportional" and entry_strength > 0:
            # Scale position by strength
            scale = entry_strength
            pos_out[entry:exit_bar] *= scale
            gross_out[entry:exit_bar] *= scale

    return pos_out, gross_out


def build_strategy_data_filtered(daily, strat, kr_s, comm,
                                  strength_map=None, threshold=0.0, mode="binary"):
    """Build per-ticker daily return matrix with optional strength filtering.
    strength_map: {ticker: strength_array_hourly} or None.
    """
    sub = daily[daily["strategy"] == strat].copy()
    tickers = sorted(sub["ticker"].unique())

    ticker_rets = {}
    ticker_pos_dict = {}
    for tkr in tickers:
        g = sub[sub["ticker"] == tkr].sort_values("date")
        pos = g["position"].values
        gross = g["daily_gross_return"].values

        # Apply strength filter if provided
        if strength_map is not None and tkr in strength_map and threshold > 0:
            # strength_map has hourly strength; positions are daily-aggregated
            # We need hourly positions. But we have daily. Use the daily positions directly.
            # For daily-aggregated data, apply filter based on trade entry in daily series.
            pos, gross = filter_positions_by_strength(pos, gross, strength_map[tkr],
                                                       threshold, mode)

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


# ═════════════════════════════════════════════════════════════════
# Part 3b: Position sizing approaches
# ═════════════════════════════════════════════════════════════════

def ps1_ew(signal, ret_per_unit, dates, kr_s):
    w = np.full(signal.shape[1], 1.0 / N_TICKERS)
    return compute_portfolio_returns(signal, ret_per_unit, w, cap=1.0, dates=dates, kr_s=kr_s)


def ps2_best(signal, ret_per_unit, dates, kr_s, train_mask):
    """PS-2 Fixed Fraction: grid search on train."""
    W_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    CAP_GRID = [0.25, 0.50, 0.75, 1.0]
    N = signal.shape[1]
    best_sh = -999
    best_w, best_cap = 1.0/N_TICKERS, 1.0

    for w_fixed in W_GRID:
        for cap in CAP_GRID:
            w = np.full(N, w_fixed)
            total_ret, exp, _, _ = compute_portfolio_returns(
                signal, ret_per_unit, w, cap=cap, dates=dates, kr_s=kr_s)
            train_ret = total_ret[train_mask]
            if len(train_ret) < 20:
                continue
            train_exp = exp[train_mask]
            if train_exp.max() > 1.0 + 1e-6:
                continue
            sh, _, _, _ = compute_excess_sharpe_exact(train_ret, dates[train_mask], kr_s)
            if sh > best_sh:
                best_sh = sh
                best_w, best_cap = w_fixed, cap

    w = np.full(N, best_w)
    return compute_portfolio_returns(signal, ret_per_unit, w, cap=best_cap, dates=dates, kr_s=kr_s), best_w, best_cap


def ps4_sigprop(signal, ret_per_unit, dates, kr_s, train_mask, pos_df):
    """PS-4 Signal Proportional."""
    WBASE_GRID = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    T, N = signal.shape
    strength = pos_df.abs().values
    p99 = np.percentile(strength[strength > 0], 99) if (strength > 0).any() else 1.0
    strength_norm = np.clip(strength / max(p99, 1e-8), 0, 1.0)

    best_sh = -999
    best_wb = 0.10
    for w_base in WBASE_GRID:
        w_matrix = signal.values * strength_norm * w_base
        total_ret, exp, _, _ = compute_portfolio_returns(
            signal, ret_per_unit, w_matrix, cap=1.0, dates=dates, kr_s=kr_s)
        train_ret = total_ret[train_mask]
        if len(train_ret) < 20:
            continue
        sh, _, _, _ = compute_excess_sharpe_exact(train_ret, dates[train_mask], kr_s)
        if sh > best_sh:
            best_sh = sh
            best_wb = w_base

    w_matrix = signal.values * strength_norm * best_wb
    return compute_portfolio_returns(signal, ret_per_unit, w_matrix, cap=1.0, dates=dates, kr_s=kr_s), best_wb


# ═════════════════════════════════════════════════════════════════
# Part 4: Compute Hourly Strength per Ticker
# ═════════════════════════════════════════════════════════════════

def compute_all_strengths_hourly(ohlcv_h, daily_df, strat, sig_params_best=None):
    """For each ticker, compute signal strength on hourly data.
    Returns: {ticker: daily_strength_array} aligned to daily positions.
    """
    from strategies_walkforward import (
        dispatch_signals, expand_grid,
        build_daily_trend, align_daily_to_hourly,
    )

    tickers = sorted(ohlcv_h["ticker"].unique())
    result = {}

    # Default sig_params (use mid-grid or best)
    if sig_params_best is None:
        sig_params_best = {}

    for ticker in tickers:
        tdf = ohlcv_h[ohlcv_h["ticker"] == ticker].sort_values("datetime")
        if len(tdf) < WARMUP_HOURLY + 100:
            continue

        close = tdf["close"].values.astype(np.float64)
        high = tdf["high"].values.astype(np.float64)
        low = tdf["low"].values.astype(np.float64)
        volume = tdf["volume"].values.astype(np.float64)
        open_arr = tdf["open"].values.astype(np.float64)
        n = len(close)

        ind = compute_base(close, high, low, open_arr, volume, is_hourly=True)
        ind["_volume"] = volume

        # Build caches
        sma_windows = sorted(set(
            SIGNAL_GRIDS_HOURLY["S1"].get("ma_window", []) +
            SIGNAL_GRIDS_HOURLY["S2"].get("bb_window", [])))
        sma_cache = precompute_sma_cache(close, sma_windows)

        dc_windows = SIGNAL_GRIDS_HOURLY["S3"].get("dc_window", [])
        dc_cache = precompute_donchian_cache(high, low, dc_windows)

        st_periods = SIGNAL_GRIDS_HOURLY["S4"].get("atr_period", [])
        st_mults = SIGNAL_GRIDS_HOURLY["S4"].get("multiplier", [])
        st_cache = precompute_supertrend_cache(high, low, close, st_periods, st_mults)

        vwap_windows = [w for w in SIGNAL_GRIDS_HOURLY["S6"].get("vwap_window", []) if w > 0]
        vwap_cache = precompute_vwap_cache(close, high, low, volume, vwap_windows)

        h_dts = tdf["datetime"].values
        session_vwap_data = precompute_session_vwap(close, high, low, volume, h_dts)

        # Pivot data
        pivot_data = {}
        for pv in ["classic", "camarilla", "fibonacci"]:
            pivot_data[pv] = compute_daily_pivots_for_hourly(daily_df, ticker, h_dts, pv)

        # For each strategy, generate signals and compute strength
        sp = sig_params_best.get(strat, {})

        # Generate signal
        sig, exit_arr = dispatch_signals(
            strat.replace("_MeanRev", "").replace("_Bollinger", "").replace("_Donchian", "")
            .replace("_Supertrend", "").replace("_PivotPoints", "").replace("_VWAP", ""),
            close, high, low, volume, ind,
            sma_cache, dc_cache, st_cache, vwap_cache,
            pivot_data, None, sp, is_hourly=True,
            session_vwap_data=session_vwap_data)

        # Map strategy name to sid
        sid_map = {"S1_MeanRev": "S1", "S2_Bollinger": "S2", "S3_Donchian": "S3",
                   "S4_Supertrend": "S4", "S5_PivotPoints": "S5", "S6_VWAP": "S6"}
        sid = sid_map[strat]

        # Compute strength
        if sid == "S1":
            strength = compute_strength_s1(close, ind, sma_cache, sig, sp)
        elif sid == "S2":
            strength = compute_strength_s2(close, ind, sma_cache, sig, sp)
        elif sid == "S3":
            strength = compute_strength_s3(close, high, low, ind, dc_cache, sig, sp)
        elif sid == "S4":
            strength = compute_strength_s4(close, high, low, ind, st_cache, sig, sp)
        elif sid == "S5":
            strength = compute_strength_s5(close, ind, pivot_data, sig, sp)
        elif sid == "S6":
            strength = compute_strength_s6(close, high, low, volume, ind, vwap_cache, sig, sp,
                                            session_vwap_data)

        # Aggregate hourly strength to daily: take mean per day (for entry detection)
        h_dates_daily = pd.to_datetime(h_dts).normalize()
        daily_str = pd.Series(strength, index=h_dates_daily).groupby(level=0).max()
        result[ticker] = daily_str

    return result


def compute_strength_from_positions(pos_df, ohlcv_h, daily_df, strat):
    """Simpler approach: compute signal strength from daily positions.
    At each trade entry (pos goes from 0 to non-zero), compute strength.
    Propagate entry strength to all bars in the trade.
    """
    # Load hourly data per ticker and compute indicators
    from strategies_walkforward import (
        build_daily_trend, align_daily_to_hourly,
    )

    tickers = sorted(pos_df.columns)
    strength_map = {}

    for ticker in tickers:
        pos_series = pos_df[ticker].values
        n = len(pos_series)
        strength = np.full(n, 0.0)

        # Get hourly data for this ticker
        tdf = ohlcv_h[ohlcv_h["ticker"] == ticker].sort_values("datetime")
        if len(tdf) < 100:
            strength_map[ticker] = strength
            continue

        close_h = tdf["close"].values.astype(np.float64)
        high_h = tdf["high"].values.astype(np.float64)
        low_h = tdf["low"].values.astype(np.float64)
        volume_h = tdf["volume"].values.astype(np.float64)
        open_h = tdf["open"].values.astype(np.float64)
        h_dts = tdf["datetime"].values

        ind = compute_base(close_h, high_h, low_h, open_h, volume_h, is_hourly=True)
        ind["_volume"] = volume_h

        # Compute per-metric arrays on hourly, then aggregate to daily max/mean
        atr = ind["atr14"]
        atr_safe = np.where(np.isnan(atr) | (atr < 1e-12), 1e-12, atr)
        adx = ind["adx14"]
        adx_val = np.where(np.isnan(adx), 0.0, adx / 100.0)  # normalize to [0,1]

        vol_avg = ind["vol_sma20"]
        vol_avg_safe = np.where(np.isnan(vol_avg) | (vol_avg < 1), 1.0, vol_avg)
        vol_ratio = volume_h / vol_avg_safe
        vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)

        rsi = calc_rsi(close_h, 14)
        rsi_ext = np.abs(rsi - 50.0) / 50.0
        rsi_ext = np.where(np.isnan(rsi_ext), 0.0, rsi_ext)

        atr_avg = calc_sma(atr, 20)
        atr_avg_safe = np.where(np.isnan(atr_avg) | (atr_avg < 1e-12), 1e-12, atr_avg)
        atr_ratio = atr / atr_avg_safe
        atr_ratio = np.where(np.isnan(atr_ratio), 1.0, atr_ratio)

        # Aggregate to daily
        h_dates_daily = pd.to_datetime(h_dts).normalize()
        daily_index = pd.Series(np.arange(len(h_dts)), index=h_dates_daily)

        # For each daily bar in position data, get the hourly metrics at that day
        pos_dates = pos_df.index
        pos_dates_norm = pd.to_datetime(pos_dates).normalize()

        # Map hourly metrics to daily (take max vol_ratio, max rsi_ext, etc.)
        metrics_hourly = pd.DataFrame({
            "vol_ratio": vol_ratio,
            "rsi_ext": rsi_ext,
            "atr_ratio": atr_ratio,
            "adx": adx_val,
        }, index=h_dates_daily)
        metrics_daily = metrics_hourly.groupby(level=0).max()

        # Expanding percentile rank for each metric
        metric_names = ["vol_ratio", "rsi_ext", "atr_ratio", "adx"]
        normalized_daily = {}
        for m in metric_names:
            if m in metrics_daily.columns:
                vals = metrics_daily[m].values
                normalized_daily[m] = expanding_pctile_rank(vals)

        # Composite: mean of normalized
        composite_daily = np.nanmean(
            [normalized_daily[m] for m in metric_names], axis=0)
        composite_series = pd.Series(composite_daily, index=metrics_daily.index)

        # Map to position dates
        for i in range(n):
            d = pos_dates_norm[i]
            if d in composite_series.index:
                strength[i] = composite_series.loc[d]

        # Propagate entry strength to all bars in trade
        trades = identify_trades(pos_series)
        for entry, exit_bar in trades:
            entry_str = strength[entry] if entry < n else 0.0
            if np.isnan(entry_str):
                entry_str = 0.0
            strength[entry:min(exit_bar, n)] = entry_str

        strength_map[ticker] = strength

    return strength_map


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("SIGNAL STRENGTH + EXACT DAILY RF — Hourly Approach A")
    print("=" * 80)

    # ── Load data ──
    print("\nLoading data...")
    pos_all = pd.read_parquet(POS_PATH)
    pos_all["date"] = pd.to_datetime(pos_all["date"])
    kr_s = load_key_rate()
    ohlcv_h = pd.read_parquet(OHLCV_PATH)
    ohlcv_h["datetime"] = pd.to_datetime(ohlcv_h["datetime"])

    # Daily data for pivots
    daily_df = ohlcv_h.copy()
    daily_df["date"] = daily_df["datetime"].dt.normalize()
    daily_df = daily_df.groupby(["ticker", "date"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).reset_index()

    # Aggregate hourly positions to daily
    daily_pos = pos_all.groupby(["strategy", "ticker", "test_year",
                                  pd.Grouper(key="date", freq="D")]).agg(
        position=("position", "last"),
        daily_gross_return=("daily_gross_return", "sum"),
    ).reset_index()
    daily_pos = daily_pos[daily_pos["daily_gross_return"] != 0]
    print(f"  Positions: {len(pos_all):,} hourly → {len(daily_pos):,} daily")

    # ══════════════════════════════════════════════════════════════
    # PART 1: Baseline with exact daily rf (no signal strength)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 1: Baseline — exact daily rf")
    print("=" * 80)

    baseline_results = []
    for comm in COMM_LEVELS:
        comm_label = f"{comm*100:.2f}%"
        for strat in STRATEGIES:
            dates, ret_df, pos_df, signal, ret_per_unit, tickers = \
                build_strategy_data_filtered(daily_pos, strat, kr_s, comm)
            years = pd.to_datetime(dates).year
            train_mask = years < 2022
            oos_mask = np.isin(years, OOS_YEARS)
            if oos_mask.sum() < 20:
                continue

            # PS-1: Equal Weight
            total_ret, exposure, trading_ret, fdr_ret = ps1_ew(signal, ret_per_unit, dates, kr_s)
            oos_ret = total_ret[oos_mask]
            oos_dates = dates[oos_mask]
            oos_exp = exposure[oos_mask]
            sharpe, ann_ret, ann_vol, ann_rf = compute_excess_sharpe_exact(oos_ret, oos_dates, kr_s)
            mdd = compute_maxdd(oos_ret)

            # Year-by-year
            yr_sharpes = {}
            for y in OOS_YEARS:
                ym = pd.to_datetime(oos_dates).year == y
                if ym.sum() > 10:
                    s, _, _, _ = compute_excess_sharpe_exact(oos_ret[ym], oos_dates[ym], kr_s)
                    yr_sharpes[y] = s

            baseline_results.append({
                "strategy": strat, "comm": comm_label, "ps": "PS-1",
                "sharpe": round(sharpe, 4),
                "ann_ret_pct": round(ann_ret * 100, 2),
                "ann_vol_pct": round(ann_vol * 100, 2),
                "ann_rf_pct": round(ann_rf * 100, 2),
                "mdd_pct": round(mdd, 2),
                "avg_exp_pct": round(np.mean(oos_exp) * 100, 1),
                "trading_ret_pct": round(np.mean(trading_ret[oos_mask]) * BPY * 100, 2),
                "fdr_ret_pct": round(np.mean(fdr_ret[oos_mask]) * BPY * 100, 2),
                **{f"sh_{y}": round(yr_sharpes.get(y, 0), 2) for y in OOS_YEARS},
            })

    bl_df = pd.DataFrame(baseline_results)
    bl_df.to_csv(OUT_DIR / "ss_baseline_exact_rf.csv", index=False)
    print("\nBaseline (PS-1, exact daily rf):")
    print(bl_df[bl_df["comm"] == "0.04%"][
        ["strategy", "sharpe", "ann_ret_pct", "ann_rf_pct", "ann_vol_pct",
         "avg_exp_pct", "trading_ret_pct", "fdr_ret_pct"]].to_string(index=False))

    # ══════════════════════════════════════════════════════════════
    # PART 2-4: Signal Strength — Grid Search
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PARTS 2-4: Signal Strength Grid Search")
    print("=" * 80)

    THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    MODES = ["binary", "proportional"]
    PS_METHODS = ["PS-1"]  # PS-1 as primary, add PS-2/PS-4 for best

    ss_results = []
    threshold_table = []

    for strat in STRATEGIES:
        print(f"\n{'─'*60}")
        print(f"  {strat}: Computing signal strength...")

        # Build daily data for this strategy
        for comm in COMM_LEVELS:
            comm_label = f"{comm*100:.2f}%"

            dates, ret_df, pos_df, signal, ret_per_unit, tickers = \
                build_strategy_data_filtered(daily_pos, strat, kr_s, comm)
            years = pd.to_datetime(dates).year
            train_mask = years < 2022
            oos_mask = np.isin(years, OOS_YEARS)
            if oos_mask.sum() < 20:
                continue

            # Compute strength for each ticker (simplified: from daily indicators)
            strength_map = compute_strength_from_positions(
                pos_df, ohlcv_h, daily_df, strat)

            # Grid search on train
            best_sh_train = -999
            best_config = {"threshold": 0.0, "mode": "binary"}
            threshold_sharpes_oos = {}

            for threshold in THRESHOLDS:
                for mode in MODES:
                    # Apply strength filter
                    dates_f, ret_df_f, pos_df_f, signal_f, rpu_f, _ = \
                        build_strategy_data_filtered(
                            daily_pos, strat, kr_s, comm,
                            strength_map=strength_map,
                            threshold=threshold, mode=mode)

                    # PS-1
                    total_ret, exp, _, _ = ps1_ew(signal_f, rpu_f, dates_f, kr_s)
                    train_ret = total_ret[train_mask[:len(total_ret)]]
                    train_dates = dates_f[train_mask[:len(total_ret)]]
                    if len(train_ret) < 20:
                        continue
                    sh_train, _, _, _ = compute_excess_sharpe_exact(train_ret, train_dates, kr_s)

                    # OOS metrics
                    oos_m = oos_mask[:len(total_ret)]
                    oos_ret = total_ret[oos_m]
                    oos_dates = dates_f[oos_m]
                    oos_exp = exp[oos_m]
                    if len(oos_ret) < 20:
                        continue
                    sh_oos, ann_ret, ann_vol, ann_rf = compute_excess_sharpe_exact(
                        oos_ret, oos_dates, kr_s)
                    mdd = compute_maxdd(oos_ret)

                    # Count trades
                    n_trades_yr = 0
                    for tkr in tickers:
                        if tkr in pos_df_f.columns:
                            p = pos_df_f[tkr].values[oos_m]
                            n_trades_yr += len(identify_trades(p))
                    n_years_oos = oos_m.sum() / BPY
                    trades_per_yr = n_trades_yr / max(n_years_oos, 0.01)

                    # Track threshold table (OOS)
                    key = (strat, comm_label, threshold, mode)
                    threshold_sharpes_oos[key] = sh_oos

                    ss_results.append({
                        "strategy": strat, "comm": comm_label,
                        "threshold": threshold, "mode": mode,
                        "sharpe_train": round(sh_train, 4),
                        "sharpe_oos": round(sh_oos, 4),
                        "ann_ret_pct": round(ann_ret * 100, 2),
                        "ann_vol_pct": round(ann_vol * 100, 2),
                        "mdd_pct": round(mdd, 2),
                        "avg_exp_pct": round(np.mean(oos_exp) * 100, 1),
                        "trades_per_yr": round(trades_per_yr, 1),
                    })

                    if sh_train > best_sh_train:
                        best_sh_train = sh_train
                        best_config = {"threshold": threshold, "mode": mode,
                                       "sh_oos": sh_oos}

            # Print best for this strategy
            bl_sh = [r["sharpe"] for r in baseline_results
                     if r["strategy"] == strat and r["comm"] == comm_label
                     and r["ps"] == "PS-1"]
            bl_sh = bl_sh[0] if bl_sh else 0
            print(f"  {strat} [{comm_label}]: baseline={bl_sh:.2f} → "
                  f"best SS: th={best_config['threshold']}, "
                  f"mode={best_config['mode']}, "
                  f"OOS Sharpe={best_config.get('sh_oos', 0):.2f}")

    # ── Save results ──
    ss_df = pd.DataFrame(ss_results)
    ss_df.to_csv(OUT_DIR / "ss_grid_results.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    # OUTPUT TABLES
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TABLE 1: Excess Sharpe Before vs After Signal Strength (OOS, PS-1)")
    print("=" * 80)

    for comm_label in ["0.04%", "0.05%"]:
        print(f"\n  Commission: {comm_label}")
        print(f"  {'Strategy':>16s}  {'Baseline':>8s}  {'Best SS':>8s}  "
              f"{'Thresh':>6s}  {'Mode':>12s}  {'Tr/yr old':>9s}  {'Tr/yr new':>9s}")
        for strat in STRATEGIES:
            bl = [r for r in baseline_results
                  if r["strategy"] == strat and r["comm"] == comm_label and r["ps"] == "PS-1"]
            bl_sh = bl[0]["sharpe"] if bl else 0

            # Best SS on train, show OOS
            ss_sub = ss_df[(ss_df["strategy"] == strat) & (ss_df["comm"] == comm_label)]
            if len(ss_sub) == 0:
                continue
            # Select best by train sharpe
            best_idx = ss_sub["sharpe_train"].idxmax()
            best = ss_sub.loc[best_idx]
            print(f"  {strat:>16s}  {bl_sh:8.2f}  {best['sharpe_oos']:8.2f}  "
                  f"{best['threshold']:6.1f}  {best['mode']:>12s}  "
                  f"{'—':>9s}  {best['trades_per_yr']:9.1f}")

    # Table 2: Threshold sweep (OOS Sharpe, binary mode)
    print("\n" + "=" * 80)
    print("TABLE 2: Threshold Sweep — OOS Excess Sharpe (binary mode, comm=0.04%)")
    print("=" * 80)
    header = f"{'Strategy':>16s}" + "".join(f"  th={t:.1f}" for t in THRESHOLDS)
    print(header)
    for strat in STRATEGIES:
        vals = []
        for t in THRESHOLDS:
            sub = ss_df[(ss_df["strategy"] == strat) & (ss_df["comm"] == "0.04%") &
                        (ss_df["threshold"] == t) & (ss_df["mode"] == "binary")]
            if len(sub) > 0:
                vals.append(f"{sub.iloc[0]['sharpe_oos']:7.2f}")
            else:
                vals.append(f"{'—':>7s}")
        print(f"  {strat:>16s}" + "".join(f"  {v}" for v in vals))

    # Table 3: Metric correlation with P&L (computed from strength_map)
    # This requires per-trade P&L data which we'll compute
    print("\n" + "=" * 80)
    print("TABLE 3: Metric Correlation with Next-Trade P&L")
    print("=" * 80)
    print("  (Computed from daily aggregated data — directional correlation)")
    # Simplified: compute for each strategy, correlation of entry strength with trade P&L
    for strat in STRATEGIES:
        dates, ret_df, pos_df, signal, ret_per_unit, tickers = \
            build_strategy_data_filtered(daily_pos, strat, kr_s, 0.0004)
        strength_map = compute_strength_from_positions(pos_df, ohlcv_h, daily_df, strat)

        all_strengths = []
        all_pnls = []
        for tkr in tickers:
            if tkr not in strength_map or tkr not in pos_df.columns:
                continue
            pos = pos_df[tkr].values
            ret = ret_df[tkr].values
            strength = strength_map[tkr]
            trades = identify_trades(pos)
            for entry, exit_bar in trades:
                if entry < len(strength):
                    s = strength[entry]
                    pnl = np.sum(ret[entry:exit_bar])
                    if not np.isnan(s) and not np.isnan(pnl):
                        all_strengths.append(s)
                        all_pnls.append(pnl)

        if len(all_strengths) > 10:
            corr = np.corrcoef(all_strengths, all_pnls)[0, 1]
            print(f"  {strat:>16s}: corr(strength, trade_PnL) = {corr:+.4f}  "
                  f"(n_trades={len(all_strengths)})")
        else:
            print(f"  {strat:>16s}: insufficient data")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved: {OUT_DIR / 'ss_baseline_exact_rf.csv'}")
    print(f"       {OUT_DIR / 'ss_grid_results.csv'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
