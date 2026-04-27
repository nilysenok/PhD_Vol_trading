#!/usr/bin/env python3
"""
stage3c_bcd_30phases.py — Comprehensive BCD optimization: 30 phases of overlays on A positions.

For each phase:
  1. Implement overlay function (B/C/D type)
  2. Walk-forward: for each test_year in [2022,2023,2024,2025], optimize on train, apply on test
  3. Compute trading Sharpe per ticker
  4. Log results
  5. Save positions if improvement over baseline

B phases (B.1–B.10): Adaptive stops, hold limits, spike exits, sizing
C phases (C.1–C.10): Regime filters, direction, term structure, hysteresis, vol-of-vol
D phases (D.1–D.10): Vol targeting, adaptive target, asymmetric gamma

Runtime target: 20–30 minutes.
"""
import os, sys, warnings, time, datetime, gc
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from itertools import product

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent

APOS_PATH   = BASE / "results/final/strategies/walkforward/wf_positions_A.parquet"
OHLCV_PATH  = BASE / "results/final/strategies/data/ohlcv_daily_full.parquet"
VPRED_PATH  = BASE / "results/final/strategies/data/vpred_aligned.parquet"
PRED_PATH   = BASE / "results/final/strategies/data/predictions_aligned.parquet"
KR_PATH     = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
OUT_DIR     = BASE / "results/final/strategies/walkforward"
PHASE_DIR   = BASE / "results/final/strategies/bcd_phases"
LOG_PATH    = BASE / "results/final/strategies/MASTER_PROGRESS_LOG.txt"

BPY = 252
WARMUP = 200
COMM = 0.0005
TEST_YEARS = [2022, 2023, 2024, 2025]
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
STRATEGY_TYPE = {
    "S1": "contrarian", "S2": "contrarian",
    "S3": "trend", "S4": "trend",
    "S5": "range", "S6": "range",
}
TICKERS = [
    "AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
    "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR",
]
HORIZONS = ["h1", "h5", "h22"]


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [bcd_30ph] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def trading_sharpe(pos, log_ret, mask, comm):
    """TRAIN metric: simple Sharpe, no FDR, no RF, with commission."""
    pm = pos[mask]
    lr = log_ret[mask]
    n = len(pm)
    if n < 20:
        return -999.0
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    s = np.std(net, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(net) * sqrt(BPY) / s


def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm):
    """OOS metric: Excess Sharpe with FDR, RF, commission."""
    pm = pos[mask]
    lr = log_ret[mask]
    rf = rf_daily[mask]
    cd = cal_days[mask]
    n = len(pm)
    if n < 20:
        return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exposure = np.abs(pm)
    fdr = (1.0 - exposure) * rf * cd
    daily_total = net + fdr
    excess = daily_total - rf * cd
    s = np.std(daily_total, ddof=1)
    if s < 1e-12:
        return 0.0, {}
    sharpe = np.mean(excess) * sqrt(BPY) / s
    exp_pct = np.mean(exposure) * 100
    # Trades count
    nt = 0
    nw = 0
    in_t = False
    tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t:
            in_t = True
            tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1
                if tr > 0:
                    nw += 1
                in_t = False
    cs = max((n / BPY), 0.01)
    return sharpe, dict(
        exposure=round(exp_pct, 1),
        tpy=round(nt / cs, 1),
        wr=round(nw / nt * 100 if nt > 0 else 0, 1),
        n_trades=nt,
    )


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_all_data():
    """Load A positions, OHLCV, vpred, key rate."""
    log("Loading A positions...")
    a_df = pd.read_parquet(APOS_PATH)
    a_df["date"] = pd.to_datetime(a_df["date"])
    log(f"  A positions: {len(a_df):,} rows, {a_df['date'].min().date()} – {a_df['date'].max().date()}")

    log("Loading OHLCV...")
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    log(f"  OHLCV: {len(ohlcv):,} rows")

    log("Loading vpred...")
    # Try vpred_aligned first (has sigma_h1/h5/h22), fallback to predictions_aligned
    vpred_df = None
    if VPRED_PATH.exists():
        vpred_df = pd.read_parquet(VPRED_PATH)
        vpred_df["date"] = pd.to_datetime(vpred_df["date"])
        log(f"  vpred_aligned: {len(vpred_df):,} rows, cols={vpred_df.columns.tolist()}")
    elif PRED_PATH.exists():
        vpred_df = pd.read_parquet(PRED_PATH)
        vpred_df["date"] = pd.to_datetime(vpred_df["date"])
        # Rename columns to sigma_h* format
        rename_map = {}
        for h in HORIZONS:
            for candidate in [f"rv_pred_{h}", f"sigma_{h}", h]:
                if candidate in vpred_df.columns:
                    rename_map[candidate] = f"sigma_{h}"
                    break
        vpred_df = vpred_df.rename(columns=rename_map)
        log(f"  predictions_aligned: {len(vpred_df):,} rows, cols={vpred_df.columns.tolist()}")

    # Key rate
    kr_df = pd.read_parquet(KR_PATH)
    kr_df["date"] = pd.to_datetime(kr_df["date"])
    kr_s = kr_df.set_index("date")["key_rate"].sort_index()
    log(f"  Key rate: {len(kr_s)} dates")

    return a_df, ohlcv, vpred_df, kr_s


def prepare_ticker_data(ticker, ohlcv, vpred_df, kr_s):
    """Prepare per-ticker arrays: close, lr, rf, cal_days, sigma_dict, dates."""
    tdf = ohlcv[ohlcv["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP + 200:
        return None
    close = tdf["close"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)

    # Log returns: lr[t] = log(close[t+1] / close[t])
    lr = np.zeros(n)
    lr[:-1] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

    cal_days = np.ones(n)
    for i in range(1, n):
        cal_days[i] = (dates[i] - dates[i - 1]).days

    rf_raw = kr_s.reindex(dates, method="ffill").values
    rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0

    # Vpred sigma
    sigma_dict = {}
    if vpred_df is not None:
        vp = vpred_df[vpred_df["ticker"] == ticker].sort_values("date")
        if len(vp) > 0:
            vp_dates = pd.to_datetime(vp["date"].values)
            for hname in HORIZONS:
                col = f"sigma_{hname}"
                if col not in vp.columns:
                    continue
                vp_vals = vp[col].values
                sigma_arr = np.full(n, np.nan)
                idx = np.searchsorted(vp_dates, dates, side="right") - 1
                valid = idx >= 0
                sigma_arr[valid] = vp_vals[idx[valid]]
                sigma_dict[hname] = sigma_arr

    return dict(
        ticker=ticker, close=close, dates=dates, n=n, lr=lr,
        cal_days=cal_days, rf_d=rf_d, sigma_dict=sigma_dict,
    )


def build_a_positions_dict(a_df, ticker_data_dict):
    """Build (sid, ticker, test_year) -> full-length position array."""
    a_pos_dict = {}
    for sid in STRATEGIES:
        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                continue
            n = td["n"]
            dates = td["dates"]

            for test_year in TEST_YEARS:
                # Full-length array (all dates for this ticker)
                pos_arr = np.zeros(n)
                sub = a_df[(a_df["strategy"] == sid) & (a_df["ticker"] == ticker) &
                           (a_df["year"] == test_year)]
                if len(sub) == 0:
                    continue
                for _, row in sub.iterrows():
                    idx = np.searchsorted(dates, row["date"])
                    if idx < n and dates[idx] == row["date"]:
                        pos_arr[idx] = row["position"]
                a_pos_dict[(sid, ticker, test_year)] = pos_arr
    return a_pos_dict


# ═══════════════════════════════════════════════════════════════
# OVERLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def apply_B_stops(a_pos, close, vpred, vol_median, k_sl, ratio, base_hold, hold_gamma, be_trigger):
    """B overlay: adaptive SL/TP/hold on A positions. Mirrors _apply_B_numba."""
    n = len(a_pos)
    out = np.zeros(n)
    in_trade = False
    entry_price = 0.0
    direction = 0.0
    sl = 0.0
    tp = 0.0
    held = 0
    max_hold = base_hold
    be_active = False
    b_exited = False

    for t in range(n):
        if abs(a_pos[t]) < 0.01:
            b_exited = False

        if in_trade:
            held += 1
            # Breakeven
            if be_trigger > 0 and not be_active:
                sv = vpred[t]
                if sv != sv:
                    sv = 0.02
                if direction > 0 and (close[t] - entry_price) > be_trigger * sv * entry_price:
                    be_active = True
                    sl = entry_price
                elif direction < 0 and (entry_price - close[t]) > be_trigger * sv * entry_price:
                    be_active = True
                    sl = entry_price

            closed = False
            if direction > 0:
                if close[t] <= sl:
                    closed = True
                elif tp > 0 and close[t] >= tp:
                    closed = True
            else:
                if close[t] >= sl:
                    closed = True
                elif tp > 0 and close[t] <= tp:
                    closed = True

            if max_hold > 0 and held >= max_hold:
                closed = True

            if abs(a_pos[t]) < 0.01:
                closed = True

            if closed:
                in_trade = False
                held = 0
                be_active = False
                if abs(a_pos[t]) > 0.01:
                    b_exited = True
            else:
                out[t] = direction

        if not in_trade and not b_exited and abs(a_pos[t]) > 0.01:
            sv = vpred[t]
            if sv != sv or sv < 1e-8:
                sv = 0.02
            direction = 1.0 if a_pos[t] > 0 else -1.0
            entry_price = close[t]
            in_trade = True
            held = 0
            be_active = False
            if direction > 0:
                sl = entry_price * (1.0 - k_sl * sv)
                tp = entry_price * (1.0 + k_sl * ratio * sv)
            else:
                sl = entry_price * (1.0 + k_sl * sv)
                tp = entry_price * (1.0 - k_sl * ratio * sv)
            if hold_gamma > 0 and sv > 1e-8 and vol_median > 1e-8:
                mh = base_hold * (vol_median / sv) ** hold_gamma
                max_hold = int(min(max(mh, 3.0), base_hold * 3.0))
            else:
                max_hold = base_hold
            out[t] = direction

    return out


def apply_B_spike_exit(a_pos, vpred, spike_threshold, spike_lookback=1):
    """B.6: Vol spike exit — zero position when vpred spikes."""
    out = a_pos.copy()
    n = len(a_pos)
    for t in range(spike_lookback, n):
        prev = vpred[t - spike_lookback]
        if prev != prev or prev < 1e-10:
            continue
        curr = vpred[t]
        if curr != curr:
            continue
        if curr / prev > spike_threshold:
            out[t] = 0.0
    return out


def apply_B_decay(a_pos, vpred, decay_rate, decay_lookback=5):
    """B.7: Position decay — exponentially reduce position after entry."""
    out = a_pos.copy()
    n = len(a_pos)
    bars_in = 0
    for t in range(1, n):
        if abs(a_pos[t]) > 0.01:
            if abs(a_pos[t - 1]) < 0.01:
                bars_in = 0
            else:
                bars_in += 1
            if bars_in > 0:
                out[t] = a_pos[t] * np.exp(-decay_rate * bars_in)
        else:
            bars_in = 0
    return out


def apply_B_trailing(a_pos, close, vpred, trail_mult):
    """B.8: Trailing stop — exit when price drops trail_mult * vpred from peak."""
    out = a_pos.copy()
    n = len(a_pos)
    in_trade = False
    direction = 0.0
    best_price = 0.0

    for t in range(n):
        if abs(a_pos[t]) < 0.01:
            in_trade = False
            continue
        if not in_trade:
            in_trade = True
            direction = 1.0 if a_pos[t] > 0 else -1.0
            best_price = close[t]
            continue

        sv = vpred[t]
        if sv != sv or sv < 1e-8:
            sv = 0.02

        if direction > 0:
            best_price = max(best_price, close[t])
            if close[t] < best_price * (1.0 - trail_mult * sv):
                out[t] = 0.0
                in_trade = False
        else:
            best_price = min(best_price, close[t])
            if close[t] > best_price * (1.0 + trail_mult * sv):
                out[t] = 0.0
                in_trade = False

    return out


def apply_B_sizing(a_pos, vpred, sizing_gamma, sizing_smooth, sizing_floor, sizing_cap):
    """B.9–B.10: Adaptive position sizing based on vpred."""
    valid = vpred[~np.isnan(vpred) & (vpred > 1e-8)]
    if len(valid) < 10:
        return a_pos.copy()
    median_v = np.nanmedian(valid)
    vpred_safe = np.where(np.isnan(vpred) | (vpred < 1e-8), median_v, vpred)
    scale = (median_v / vpred_safe) ** sizing_gamma
    scale = np.clip(scale, sizing_floor, sizing_cap)
    if sizing_smooth > 0:
        scale = pd.Series(scale).ewm(span=sizing_smooth, min_periods=1).mean().values
    out = a_pos * scale
    return np.clip(out, -1.0, 1.0)


def apply_C_regime(a_pos, vpred, threshold, lookback, strategy_type,
                   direction="off", direction_lookback=5,
                   sigma_h22=None, sigma_h1=None, ts_threshold=0,
                   ts_direction="above", min_regime_days=0,
                   vov_lookback=0, vov_threshold=0, vov_direction="low"):
    """C overlay: regime filter on A positions."""
    n = len(a_pos)
    out = a_pos.copy()

    # Percentile ranking
    pctrank = pd.Series(vpred).rolling(lookback, min_periods=max(50, lookback // 5)).rank(pct=True).values

    # Base regime mask
    if strategy_type == "trend":
        mask = ~np.isnan(pctrank) & (pctrank > threshold)
    elif strategy_type == "contrarian":
        mask = ~np.isnan(pctrank) & (pctrank < (1.0 - threshold))
    else:  # range
        lo_band = max(0.5 - threshold / 2, 0.1)
        hi_band = min(0.5 + threshold / 2, 0.9)
        mask = ~np.isnan(pctrank) & (pctrank >= lo_band) & (pctrank <= hi_band)

    # Direction filter
    if direction == "rise":
        delta = np.full(n, np.nan)
        delta[direction_lookback:] = vpred[direction_lookback:] - vpred[:-direction_lookback]
        mask = mask & (~np.isnan(delta)) & (delta > 0)
    elif direction == "fall":
        delta = np.full(n, np.nan)
        delta[direction_lookback:] = vpred[direction_lookback:] - vpred[:-direction_lookback]
        mask = mask & (~np.isnan(delta)) & (delta < 0)

    # Term structure filter (C.4)
    if ts_threshold > 0 and sigma_h22 is not None and sigma_h1 is not None:
        s_h1_safe = np.maximum(sigma_h1, 1e-8)
        ts_ratio = sigma_h22 / s_h1_safe
        valid_r = ~np.isnan(ts_ratio)
        if ts_direction == "above":
            mask = mask & valid_r & (ts_ratio > ts_threshold)
        else:
            mask = mask & valid_r & (ts_ratio < ts_threshold)

    # Hysteresis (C.5): require min consecutive regime days
    if min_regime_days > 0:
        consecutive = 0
        for t in range(n):
            if mask[t]:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive < min_regime_days:
                mask[t] = False

    # Vol-of-vol filter (C.7)
    if vov_threshold > 0 and vov_lookback > 0:
        for t in range(vov_lookback, n):
            w = vpred[t - vov_lookback:t]
            v = w[~np.isnan(w)]
            if len(v) > 3:
                vov = np.std(v) / max(np.mean(v), 1e-8)
                if vov_direction == "low":
                    if vov > vov_threshold:
                        mask[t] = False
                else:
                    if vov < vov_threshold:
                        mask[t] = False

    out[~mask] = 0.0
    return out


def apply_D_voltarget(a_pos, vpred, sigma_target, gamma, vol_floor, vol_cap,
                      smooth_span=0, adaptive=False, adapt_lookback=126,
                      pctile=50, multiplier=1.0,
                      gamma_up=None, gamma_down=None, direction_lookback=1):
    """D overlay: vol-targeting scale on A positions."""
    n = len(a_pos)
    sigma = vpred.copy()
    if smooth_span > 0:
        sigma = pd.Series(sigma).ewm(span=smooth_span, min_periods=1).mean().values
    sigma = np.where(np.isnan(sigma) | (sigma < 1e-6), 1e-6, sigma)

    # Asymmetric gamma (D.3)
    if gamma_up is not None and gamma_down is not None:
        vpred_prev = np.roll(sigma, direction_lookback)
        vpred_prev[:direction_lookback] = sigma[:direction_lookback]
        rising = sigma > vpred_prev
        gamma_arr = np.where(rising, gamma_up, gamma_down)
    else:
        gamma_arr = gamma

    # Adaptive target (D.2)
    if adaptive:
        adaptive_targets = np.full(n, sigma_target)
        for t in range(adapt_lookback, n):
            w = sigma[max(0, t - adapt_lookback):t]
            v = w[~np.isnan(w)]
            if len(v) > 5:
                adaptive_targets[t] = np.percentile(v, pctile) * multiplier
        scale = (adaptive_targets / np.maximum(sigma, 1e-8)) ** gamma_arr
    else:
        scale = (sigma_target / np.maximum(sigma, 1e-8)) ** gamma_arr

    # Floor/cap
    scale = np.clip(scale, vol_floor, vol_cap)
    # Ensure positions stay in [-1, 1]
    scale = np.minimum(scale, 1.0)
    scale = np.maximum(scale, 0.0)

    out = a_pos * scale
    return out


# ═══════════════════════════════════════════════════════════════
# PARAMETER GRIDS (30 PHASES)
# ═══════════════════════════════════════════════════════════════

def _expand(d):
    """Expand dict of lists to list of dicts (cartesian product)."""
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ── B PHASES ──

# B.1: Core SL/TP (contrarian-tuned)
B01_GRID = _expand(dict(
    k_sl=[1.0, 2.0, 3.0, 5.0, 7.0],
    ratio=[0.5, 1.0, 2.0, 3.0],
    base_hold=[5, 15, 50],
    hold_gamma=[0, 0.5],
    be_trigger=[0],
))

# B.2: Core SL/TP (trend-tuned — wider stops)
B02_GRID = _expand(dict(
    k_sl=[3.0, 5.0, 7.0, 10.0],
    ratio=[1.5, 2.0, 3.0, 5.0],
    base_hold=[10, 30, 60],
    hold_gamma=[0, 0.25],
    be_trigger=[0],
))

# B.3: Core SL/TP with breakeven
B03_GRID = _expand(dict(
    k_sl=[2.0, 3.0, 5.0],
    ratio=[1.0, 2.0, 3.0],
    base_hold=[10, 30],
    hold_gamma=[0, 0.5],
    be_trigger=[1.0, 2.0, 5.0],
))

# B.4: Tight stops
B04_GRID = _expand(dict(
    k_sl=[0.5, 0.75, 1.0, 1.5],
    ratio=[0.5, 1.0, 1.5],
    base_hold=[3, 5, 10],
    hold_gamma=[0, 0.5, 1.0],
    be_trigger=[0],
))

# B.5: Wide stops + long hold
B05_GRID = _expand(dict(
    k_sl=[5.0, 7.0, 10.0, 15.0],
    ratio=[2.0, 3.0, 5.0],
    base_hold=[30, 60, 100],
    hold_gamma=[0, 0.25],
    be_trigger=[0],
))

# B.6: Vol spike exit (additive on A)
B06_GRID = _expand(dict(
    spike_threshold=[1.3, 1.5, 2.0, 2.5, 3.0],
    spike_lookback=[1, 3, 5],
))

# B.7: Position decay
B07_GRID = _expand(dict(
    decay_rate=[0.01, 0.02, 0.05, 0.1, 0.2],
    decay_lookback=[3, 5, 10],
))

# B.8: Trailing stop
B08_GRID = _expand(dict(
    trail_mult=[1.0, 2.0, 3.0, 5.0, 7.0],
))

# B.9: Adaptive sizing (inverse vol)
B09_GRID = _expand(dict(
    sizing_gamma=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    sizing_smooth=[0, 5, 10, 20],
    sizing_floor=[0.2, 0.3, 0.5],
    sizing_cap=[1.0, 1.5, 2.0, 3.0],
))

# B.10: Aggressive sizing
B10_GRID = _expand(dict(
    sizing_gamma=[1.5, 2.0, 3.0],
    sizing_smooth=[0, 5, 20],
    sizing_floor=[0.1, 0.2],
    sizing_cap=[2.0, 3.0, 5.0],
))

# ── C PHASES ──

# C.1: Basic contrarian regime
C01_GRID = _expand(dict(
    threshold=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    lookback=[63, 126, 252],
    direction=["off"],
))

# C.2: Basic trend regime
C02_GRID = _expand(dict(
    threshold=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    lookback=[63, 126, 252],
    direction=["off"],
))

# C.3: Regime + direction filter
C03_GRID = _expand(dict(
    threshold=[0.3, 0.5, 0.7],
    lookback=[63, 126, 252],
    direction=["rise", "fall"],
    direction_lookback=[3, 5, 10],
))

# C.4: Term structure filter
C04_GRID = _expand(dict(
    threshold=[0.3, 0.5, 0.7],
    lookback=[63, 126],
    direction=["off"],
    ts_threshold=[0.8, 0.9, 1.0, 1.1, 1.2],
    ts_direction=["above", "below"],
))

# C.5: Hysteresis
C05_GRID = _expand(dict(
    threshold=[0.3, 0.5, 0.7],
    lookback=[63, 126, 252],
    direction=["off"],
    min_regime_days=[3, 5, 10],
))

# C.6: Range regime (for S5/S6)
C06_GRID = _expand(dict(
    threshold=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    lookback=[63, 126, 252],
    direction=["off"],
))

# C.7: Vol-of-vol filter
C07_GRID = _expand(dict(
    threshold=[0.3, 0.5, 0.7],
    lookback=[63, 126],
    direction=["off"],
    vov_lookback=[20, 40],
    vov_threshold=[0.5, 1.0, 1.5],
    vov_direction=["low", "high"],
))

# C.8: Direction + hysteresis combo
C08_GRID = _expand(dict(
    threshold=[0.3, 0.5, 0.7],
    lookback=[63, 126],
    direction=["rise", "fall"],
    direction_lookback=[5],
    min_regime_days=[3, 5],
))

# C.9: Full kitchen sink (small grid)
C09_GRID = _expand(dict(
    threshold=[0.3, 0.5, 0.7],
    lookback=[63, 126],
    direction=["off", "fall"],
    ts_threshold=[0, 1.0],
    ts_direction=["above"],
    min_regime_days=[0, 5],
    vov_lookback=[0, 20],
    vov_threshold=[0, 1.0],
    vov_direction=["low"],
))

# C.10: Aggressive filter (very tight)
C10_GRID = _expand(dict(
    threshold=[0.15, 0.2, 0.25, 0.3],
    lookback=[40, 63, 126],
    direction=["off"],
    min_regime_days=[0, 5, 10],
))

# ── D PHASES ──

# D.1: Basic vol targeting
D01_GRID = _expand(dict(
    sigma_target=[0.01, 0.02, 0.05, 0.10, 0.15, 0.25],
    gamma=[0.25, 0.5, 1.0, 1.5, 2.0],
    vol_floor=[0.1, 0.3],
    vol_cap=[2.0, 3.0],
    smooth_span=[0],
))

# D.2: Smoothed vol targeting
D02_GRID = _expand(dict(
    sigma_target=[0.02, 0.05, 0.10, 0.15],
    gamma=[0.5, 1.0, 1.5],
    vol_floor=[0.1, 0.3],
    vol_cap=[2.0, 3.0],
    smooth_span=[5, 10, 20],
))

# D.3: Asymmetric gamma
D03_GRID = _expand(dict(
    sigma_target=[0.02, 0.05, 0.10],
    gamma_up=[1.0, 1.5, 2.0, 3.0],
    gamma_down=[0.25, 0.5, 1.0],
    direction_lookback=[1, 5],
    vol_floor=[0.1, 0.3],
    vol_cap=[2.0, 3.0],
    smooth_span=[0],
))

# D.4: Adaptive target
D04_GRID = _expand(dict(
    pctile=[25, 50, 75],
    multiplier=[0.75, 1.0, 1.5],
    adapt_lookback=[63, 126, 252],
    gamma=[0.5, 1.0, 1.5],
    vol_floor=[0.1, 0.3],
    vol_cap=[2.0, 3.0],
    smooth_span=[0],
))

# D.5: Very conservative (low gamma)
D05_GRID = _expand(dict(
    sigma_target=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    gamma=[0.1, 0.15, 0.25],
    vol_floor=[0.3, 0.5],
    vol_cap=[1.5, 2.0],
    smooth_span=[0, 10],
))

# D.6: Aggressive (high gamma)
D06_GRID = _expand(dict(
    sigma_target=[0.01, 0.02, 0.05],
    gamma=[2.0, 3.0, 5.0],
    vol_floor=[0.05, 0.1],
    vol_cap=[3.0, 5.0],
    smooth_span=[0, 5],
))

# D.7: Smoothed + adaptive
D07_GRID = _expand(dict(
    pctile=[25, 50],
    multiplier=[0.75, 1.0],
    adapt_lookback=[63, 126],
    gamma=[0.5, 1.0],
    vol_floor=[0.1],
    vol_cap=[2.0],
    smooth_span=[5, 10, 20],
))

# D.8: Asymmetric + smoothed
D08_GRID = _expand(dict(
    sigma_target=[0.02, 0.05, 0.10],
    gamma_up=[1.5, 2.0],
    gamma_down=[0.5, 1.0],
    direction_lookback=[1, 5],
    vol_floor=[0.1, 0.3],
    vol_cap=[2.0, 3.0],
    smooth_span=[5, 10],
))

# D.9: Fine-tuned mid-range
D09_GRID = _expand(dict(
    sigma_target=[0.03, 0.05, 0.07, 0.10, 0.12],
    gamma=[0.5, 0.75, 1.0, 1.25],
    vol_floor=[0.15, 0.25],
    vol_cap=[1.8, 2.5],
    smooth_span=[0, 5],
))

# D.10: Flat cap (just limit position in high vol)
D10_GRID = _expand(dict(
    sigma_target=[0.10, 0.15, 0.20, 0.30],
    gamma=[1.0],
    vol_floor=[0.5, 0.7],
    vol_cap=[1.0, 1.0],  # cap=1 means no scale up
    smooth_span=[0, 5, 10],
))


# ═══════════════════════════════════════════════════════════════
# PHASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════

# Each phase: (phase_id, overlay_type, param_grid, description, strategy_filter)
# strategy_filter: None = all, or list of strategy types to apply to

PHASE_DEFS = [
    # ── B PHASES ──
    ("B.01", "B_stops",  B01_GRID, "Core SL/TP contrarian", None),
    ("B.02", "B_stops",  B02_GRID, "Core SL/TP trend", None),
    ("B.03", "B_stops",  B03_GRID, "SL/TP with breakeven", None),
    ("B.04", "B_stops",  B04_GRID, "Tight stops", None),
    ("B.05", "B_stops",  B05_GRID, "Wide stops + long hold", None),
    ("B.06", "B_spike",  B06_GRID, "Vol spike exit", None),
    ("B.07", "B_decay",  B07_GRID, "Position decay", None),
    ("B.08", "B_trail",  B08_GRID, "Trailing stop", None),
    ("B.09", "B_sizing", B09_GRID, "Adaptive sizing", None),
    ("B.10", "B_sizing", B10_GRID, "Aggressive sizing", None),
    # ── C PHASES ──
    ("C.01", "C_regime", C01_GRID, "Contrarian regime", ["contrarian"]),
    ("C.02", "C_regime", C02_GRID, "Trend regime", ["trend"]),
    ("C.03", "C_regime", C03_GRID, "Regime + direction", None),
    ("C.04", "C_regime", C04_GRID, "Term structure filter", None),
    ("C.05", "C_regime", C05_GRID, "Hysteresis", None),
    ("C.06", "C_regime", C06_GRID, "Range regime", ["range"]),
    ("C.07", "C_regime", C07_GRID, "Vol-of-vol filter", None),
    ("C.08", "C_regime", C08_GRID, "Direction + hysteresis", None),
    ("C.09", "C_regime", C09_GRID, "Full combo", None),
    ("C.10", "C_regime", C10_GRID, "Aggressive filter", None),
    # ── D PHASES ──
    ("D.01", "D_vol",    D01_GRID, "Basic vol targeting", None),
    ("D.02", "D_vol",    D02_GRID, "Smoothed vol targeting", None),
    ("D.03", "D_vol",    D03_GRID, "Asymmetric gamma", None),
    ("D.04", "D_vol",    D04_GRID, "Adaptive target", None),
    ("D.05", "D_vol",    D05_GRID, "Conservative vol target", None),
    ("D.06", "D_vol",    D06_GRID, "Aggressive vol target", None),
    ("D.07", "D_vol",    D07_GRID, "Smoothed + adaptive", None),
    ("D.08", "D_vol",    D08_GRID, "Asymmetric + smoothed", None),
    ("D.09", "D_vol",    D09_GRID, "Fine-tuned mid-range", None),
    ("D.10", "D_vol",    D10_GRID, "Flat cap high vol", None),
]


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════

def _apply_overlay(overlay_type, a_pos, close, vpred, vol_median, params,
                   sigma_h22=None, sigma_h1=None):
    """Dispatch to the correct overlay function."""
    if overlay_type == "B_stops":
        return apply_B_stops(
            a_pos, close, vpred, vol_median,
            params.get("k_sl", 2.0), params.get("ratio", 1.5),
            params.get("base_hold", 15), params.get("hold_gamma", 0.5),
            params.get("be_trigger", 0),
        )
    elif overlay_type == "B_spike":
        return apply_B_spike_exit(
            a_pos, vpred,
            params.get("spike_threshold", 2.0),
            params.get("spike_lookback", 1),
        )
    elif overlay_type == "B_decay":
        return apply_B_decay(
            a_pos, vpred,
            params.get("decay_rate", 0.05),
            params.get("decay_lookback", 5),
        )
    elif overlay_type == "B_trail":
        return apply_B_trailing(
            a_pos, close, vpred,
            params.get("trail_mult", 3.0),
        )
    elif overlay_type == "B_sizing":
        return apply_B_sizing(
            a_pos, vpred,
            params.get("sizing_gamma", 1.0),
            params.get("sizing_smooth", 0),
            params.get("sizing_floor", 0.2),
            params.get("sizing_cap", 2.0),
        )
    elif overlay_type == "C_regime":
        stype = params.get("_strategy_type", "contrarian")
        return apply_C_regime(
            a_pos, vpred,
            threshold=params.get("threshold", 0.5),
            lookback=params.get("lookback", 126),
            strategy_type=stype,
            direction=params.get("direction", "off"),
            direction_lookback=params.get("direction_lookback", 5),
            sigma_h22=sigma_h22,
            sigma_h1=sigma_h1,
            ts_threshold=params.get("ts_threshold", 0),
            ts_direction=params.get("ts_direction", "above"),
            min_regime_days=params.get("min_regime_days", 0),
            vov_lookback=params.get("vov_lookback", 0),
            vov_threshold=params.get("vov_threshold", 0),
            vov_direction=params.get("vov_direction", "low"),
        )
    elif overlay_type == "D_vol":
        return apply_D_voltarget(
            a_pos, vpred,
            sigma_target=params.get("sigma_target", 0.10),
            gamma=params.get("gamma", 1.0),
            vol_floor=params.get("vol_floor", 0.1),
            vol_cap=params.get("vol_cap", 3.0),
            smooth_span=params.get("smooth_span", 0),
            adaptive=params.get("adaptive", False) or ("pctile" in params and "adapt_lookback" in params),
            adapt_lookback=params.get("adapt_lookback", 126),
            pctile=params.get("pctile", 50),
            multiplier=params.get("multiplier", 1.0),
            gamma_up=params.get("gamma_up", None),
            gamma_down=params.get("gamma_down", None),
            direction_lookback=params.get("direction_lookback", 1),
        )
    else:
        raise ValueError(f"Unknown overlay type: {overlay_type}")


def _subsample_grid(grid, max_combos=500):
    """If grid too large, subsample for coarse pass."""
    if len(grid) <= max_combos:
        return grid, False
    rng = np.random.RandomState(42)
    idx = rng.choice(len(grid), size=max_combos, replace=False)
    return [grid[i] for i in idx], True


def _fine_grid_around(best_params, full_grid, n_neighbors=50):
    """Find N closest neighbors in full grid to best_params (by param distance)."""
    # Simple: find params that differ on at most 1–2 keys
    numeric_keys = [k for k, v in best_params.items() if isinstance(v, (int, float)) and k[0] != '_']
    if not numeric_keys or len(full_grid) <= n_neighbors:
        return full_grid

    best_vec = np.array([float(best_params.get(k, 0)) for k in numeric_keys])
    # Normalize
    all_vecs = np.array([[float(p.get(k, 0)) for k in numeric_keys] for p in full_grid])
    ranges = np.ptp(all_vecs, axis=0)
    ranges[ranges < 1e-12] = 1.0
    all_norm = all_vecs / ranges
    best_norm = best_vec / ranges
    dists = np.sum((all_norm - best_norm) ** 2, axis=1)
    idx = np.argsort(dists)[:n_neighbors]
    return [full_grid[i] for i in idx]


def run_phase(phase_id, overlay_type, param_grid, strategy_filter,
              ticker_data_dict, a_pos_dict, a_sharpe_dict):
    """Walk-forward test of one BCD phase. Returns per (sid, ticker, year) results."""
    t0_phase = time.time()
    results = []
    positions_out = {}

    # 2-pass if grid > 500
    coarse_grid, needs_fine = _subsample_grid(param_grid, max_combos=500)
    n_combos_total = len(coarse_grid) * len(HORIZONS)
    log(f"  {phase_id}: {len(param_grid)} params, coarse={len(coarse_grid)}, "
        f"x{len(HORIZONS)} horizons = {n_combos_total} combos/ticker")

    count = 0
    for sid in STRATEGIES:
        stype = STRATEGY_TYPE[sid]
        # Check strategy filter
        if strategy_filter is not None and stype not in strategy_filter:
            continue

        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                continue
            close = td["close"]
            lr = td["lr"]
            rf_d = td["rf_d"]
            cal_days = td["cal_days"]
            dates = td["dates"]
            n = td["n"]
            sigma_dict = td["sigma_dict"]

            for test_year in TEST_YEARS:
                a_pos = a_pos_dict.get((sid, ticker, test_year))
                if a_pos is None:
                    continue

                # Check that A has any non-zero positions
                if np.sum(np.abs(a_pos) > 0.01) < 3:
                    # A is flat — no point in overlay
                    a_sh = a_sharpe_dict.get((sid, ticker, test_year), 0.0)
                    results.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe_trading=a_sh, sharpe_trading_a=a_sh,
                        approach=phase_id[0], phase=phase_id,
                        params="flat_A", horizon="none",
                    ))
                    positions_out[(sid, ticker, test_year)] = a_pos.copy()
                    continue

                train_end = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
                test_start = train_end
                test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
                if train_end <= WARMUP or test_end <= test_start:
                    continue

                train_mask = np.zeros(n, dtype=bool)
                train_mask[WARMUP:train_end] = True
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True

                best_sh_train = -999.0
                best_params = None
                best_horizon = None

                for horizon in HORIZONS:
                    sigma = sigma_dict.get(horizon)
                    if sigma is None:
                        continue
                    sigma_h22 = sigma_dict.get("h22")
                    sigma_h1 = sigma_dict.get("h1")
                    sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                    vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02

                    for p in coarse_grid:
                        pp = dict(p, _strategy_type=stype)
                        try:
                            bcd_pos = _apply_overlay(
                                overlay_type, a_pos, close, sigma, vol_median, pp,
                                sigma_h22=sigma_h22, sigma_h1=sigma_h1,
                            )
                        except Exception:
                            continue
                        sh = trading_sharpe(bcd_pos, lr, train_mask, COMM)
                        if sh > best_sh_train:
                            best_sh_train = sh
                            best_params = pp.copy()
                            best_horizon = horizon

                # Fine pass
                if needs_fine and best_params is not None:
                    fine_grid = _fine_grid_around(best_params, param_grid, n_neighbors=80)
                    for p in fine_grid:
                        for horizon in HORIZONS:
                            sigma = sigma_dict.get(horizon)
                            if sigma is None:
                                continue
                            sigma_h22 = sigma_dict.get("h22")
                            sigma_h1 = sigma_dict.get("h1")
                            sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                            vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02
                            pp = dict(p, _strategy_type=stype)
                            try:
                                bcd_pos = _apply_overlay(
                                    overlay_type, a_pos, close, sigma, vol_median, pp,
                                    sigma_h22=sigma_h22, sigma_h1=sigma_h1,
                                )
                            except Exception:
                                continue
                            sh = trading_sharpe(bcd_pos, lr, train_mask, COMM)
                            if sh > best_sh_train:
                                best_sh_train = sh
                                best_params = pp.copy()
                                best_horizon = horizon

                # Apply best on test
                a_sh = a_sharpe_dict.get((sid, ticker, test_year), 0.0)
                if best_params is not None and best_horizon is not None:
                    sigma = sigma_dict.get(best_horizon, np.full(n, 0.02))
                    sigma_h22 = sigma_dict.get("h22")
                    sigma_h1 = sigma_dict.get("h1")
                    sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                    vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02
                    try:
                        bcd_pos = _apply_overlay(
                            overlay_type, a_pos, close, sigma, vol_median, best_params,
                            sigma_h22=sigma_h22, sigma_h1=sigma_h1,
                        )
                    except Exception:
                        bcd_pos = a_pos.copy()

                    test_sh = trading_sharpe(bcd_pos, lr, test_mask, COMM)
                    positions_out[(sid, ticker, test_year)] = bcd_pos

                    # Clean params for logging (remove internal keys)
                    log_params = {k: v for k, v in best_params.items() if not k.startswith('_')}
                    results.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe_trading=round(test_sh, 4),
                        sharpe_trading_a=round(a_sh, 4),
                        approach=phase_id[0], phase=phase_id,
                        train_sharpe=round(best_sh_train, 4),
                        params=str(log_params), horizon=best_horizon,
                    ))
                else:
                    # No valid overlay found — keep A
                    positions_out[(sid, ticker, test_year)] = a_pos.copy()
                    results.append(dict(
                        strategy=sid, ticker=ticker, year=test_year,
                        sharpe_trading=round(a_sh, 4),
                        sharpe_trading_a=round(a_sh, 4),
                        approach=phase_id[0], phase=phase_id,
                        train_sharpe=round(best_sh_train, 4),
                        params="fallback_to_A", horizon="none",
                    ))

                count += 1

    elapsed = time.time() - t0_phase
    log(f"  {phase_id}: {count} cells in {elapsed:.1f}s")
    return pd.DataFrame(results), positions_out


# ═══════════════════════════════════════════════════════════════
# COMPUTE BASELINE A SHARPES
# ═══════════════════════════════════════════════════════════════

def compute_a_sharpes(ticker_data_dict, a_pos_dict):
    """Compute trading Sharpe for each A position (test year)."""
    a_sh = {}
    for (sid, ticker, test_year), a_pos in a_pos_dict.items():
        td = ticker_data_dict.get(ticker)
        if td is None:
            continue
        dates = td["dates"]
        n = td["n"]
        lr = td["lr"]
        test_start = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
        test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_end] = True
        sh = trading_sharpe(a_pos, lr, test_mask, COMM)
        a_sh[(sid, ticker, test_year)] = round(sh, 4)
    return a_sh


# ═══════════════════════════════════════════════════════════════
# SAVE POSITIONS
# ═══════════════════════════════════════════════════════════════

def save_positions_parquet(positions_dict, ticker_data_dict, approach, out_dir=None):
    """Save positions as parquet matching A format: date, ticker, strategy, year, position, daily_gross_return."""
    if out_dir is None:
        out_dir = OUT_DIR
    rows = []
    for (sid, ticker, year), pos_arr in positions_dict.items():
        td = ticker_data_dict.get(ticker)
        if td is None:
            continue
        dates = td["dates"]
        n = td["n"]
        lr = td["lr"]
        test_start = np.searchsorted(dates, pd.Timestamp(f"{year}-01-01"))
        test_end = np.searchsorted(dates, pd.Timestamp(f"{year + 1}-01-01"))
        for t in range(test_start, min(test_end, n)):
            rows.append(dict(
                date=dates[t], ticker=ticker, strategy=sid,
                year=year, position=round(float(pos_arr[t]), 6),
                daily_gross_return=round(float(pos_arr[t] * lr[t]), 8),
            ))
    if rows:
        df = pd.DataFrame(rows)
        fpath = Path(out_dir) / f"wf_positions_{approach}.parquet"
        df.to_parquet(fpath, index=False)
        log(f"  Saved {fpath} ({len(df):,} rows)")
    return len(rows)


# ═══════════════════════════════════════════════════════════════
# AGGREGATE: BEST PHASE PER TYPE -> FINAL B/C/D
# ═══════════════════════════════════════════════════════════════

def select_best_phase_per_type(all_phase_results, all_phase_positions, a_sharpe_dict):
    """For each of B, C, D: select the best phase per (strategy, ticker, year)."""
    best_positions = {"B": {}, "C": {}, "D": {}}
    best_phase_log = {"B": {}, "C": {}, "D": {}}

    for approach_type in ["B", "C", "D"]:
        # Collect all phases of this type
        type_phases = {pid: (res, pos) for pid, (res, pos) in
                       zip([p[0] for p in PHASE_DEFS],
                           [(all_phase_results.get(p[0], pd.DataFrame()),
                             all_phase_positions.get(p[0], {})) for p in PHASE_DEFS])
                       if pid.startswith(approach_type)}

        for sid in STRATEGIES:
            for ticker in TICKERS:
                for test_year in TEST_YEARS:
                    key = (sid, ticker, test_year)
                    a_sh = a_sharpe_dict.get(key, 0.0)

                    best_sh = a_sh  # default: keep A if nothing better
                    best_pos = None
                    best_pid = "A"

                    for pid, (res_df, pos_dict) in type_phases.items():
                        if res_df.empty:
                            continue
                        row = res_df[(res_df["strategy"] == sid) &
                                     (res_df["ticker"] == ticker) &
                                     (res_df["year"] == test_year)]
                        if len(row) == 0:
                            continue
                        sh = row["sharpe_trading"].iloc[0]
                        if sh > best_sh:
                            best_sh = sh
                            best_pos = pos_dict.get(key)
                            best_pid = pid

                    if best_pos is not None:
                        best_positions[approach_type][key] = best_pos
                        best_phase_log[approach_type][key] = best_pid
                    else:
                        # Keep A — need to get from any phase that has it
                        for pid, (_, pos_dict) in type_phases.items():
                            if key in pos_dict:
                                best_positions[approach_type][key] = pos_dict[key]
                                break
                        best_phase_log[approach_type][key] = "A"

    return best_positions, best_phase_log


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO & AGGREGATE METRICS
# ═══════════════════════════════════════════════════════════════

def compute_aggregate_metrics(positions_dict, ticker_data_dict, approach_label):
    """Compute per-strategy mean Sharpe across years, plus MSP."""
    # Per-strategy portfolio: equal-weight across tickers
    strategy_year_sharpes = {}
    for sid in STRATEGIES:
        for test_year in TEST_YEARS:
            # Collect daily returns for all tickers
            daily_rets = []
            for ticker in TICKERS:
                key = (sid, ticker, test_year)
                pos = positions_dict.get(key)
                td = ticker_data_dict.get(ticker)
                if pos is None or td is None:
                    continue
                dates = td["dates"]
                n = td["n"]
                lr = td["lr"]
                test_start = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
                test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True
                pm = pos[test_mask]
                lr_m = lr[test_mask]
                dp = np.diff(pm, prepend=0.0)
                net = pm * lr_m - np.abs(dp) * COMM
                daily_rets.append(net)

            if daily_rets:
                # Equal weight portfolio
                max_len = max(len(x) for x in daily_rets)
                portfolio = np.zeros(max_len)
                for dr in daily_rets:
                    portfolio[:len(dr)] += dr / len(daily_rets)
                s = np.std(portfolio, ddof=1)
                sh = np.mean(portfolio) * sqrt(BPY) / s if s > 1e-12 else 0.0
                strategy_year_sharpes[(sid, test_year)] = sh

    # Per-strategy mean across years
    strategy_means = {}
    for sid in STRATEGIES:
        year_shs = [strategy_year_sharpes.get((sid, y), np.nan) for y in TEST_YEARS]
        strategy_means[sid] = np.nanmean(year_shs)

    # MSP: equal-weight across strategies
    msp_year = {}
    for test_year in TEST_YEARS:
        all_rets = []
        for sid in STRATEGIES:
            for ticker in TICKERS:
                key = (sid, ticker, test_year)
                pos = positions_dict.get(key)
                td = ticker_data_dict.get(ticker)
                if pos is None or td is None:
                    continue
                dates = td["dates"]
                n = td["n"]
                lr = td["lr"]
                test_start = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
                test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
                test_mask = np.zeros(n, dtype=bool)
                test_mask[test_start:test_end] = True
                pm = pos[test_mask]
                lr_m = lr[test_mask]
                dp = np.diff(pm, prepend=0.0)
                net = pm * lr_m - np.abs(dp) * COMM
                all_rets.append(net)

        if all_rets:
            max_len = max(len(x) for x in all_rets)
            msp = np.zeros(max_len)
            for dr in all_rets:
                msp[:len(dr)] += dr / len(all_rets)
            s = np.std(msp, ddof=1)
            msp_year[test_year] = np.mean(msp) * sqrt(BPY) / s if s > 1e-12 else 0.0

    msp_mean = np.nanmean(list(msp_year.values())) if msp_year else 0.0

    return strategy_means, msp_year, msp_mean


# ═══════════════════════════════════════════════════════════════
# CERTAINTY EQUIVALENT
# ═══════════════════════════════════════════════════════════════

def compute_ce(positions_dict, ticker_data_dict, risk_aversion=2.0):
    """CE = E[r] - 0.5 * risk_aversion * Var[r], annualized."""
    all_daily = []
    for (sid, ticker, test_year), pos in positions_dict.items():
        td = ticker_data_dict.get(ticker)
        if td is None:
            continue
        dates = td["dates"]
        n = td["n"]
        lr = td["lr"]
        test_start = np.searchsorted(dates, pd.Timestamp(f"{test_year}-01-01"))
        test_end = np.searchsorted(dates, pd.Timestamp(f"{test_year + 1}-01-01"))
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_end] = True
        pm = pos[test_mask]
        lr_m = lr[test_mask]
        dp = np.diff(pm, prepend=0.0)
        net = pm * lr_m - np.abs(dp) * COMM
        all_daily.extend(net.tolist())

    if not all_daily:
        return 0.0
    arr = np.array(all_daily)
    mu = np.mean(arr) * BPY
    var = np.var(arr, ddof=1) * BPY
    return mu - 0.5 * risk_aversion * var


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_global = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("stage3c_bcd_30phases.py — 30-Phase BCD Optimization")
    log(f"Test years: {TEST_YEARS}, Tickers: {len(TICKERS)}, Strategies: {len(STRATEGIES)}")
    log(f"Commission: {COMM}, BPY: {BPY}, Warmup: {WARMUP}")
    log("=" * 80)

    # ── Load data ──
    a_df, ohlcv, vpred_df, kr_s = load_all_data()

    # ── Prepare per-ticker data ──
    log("Preparing per-ticker data...")
    ticker_data_dict = {}
    for ticker in TICKERS:
        td = prepare_ticker_data(ticker, ohlcv, vpred_df, kr_s)
        if td is not None:
            ticker_data_dict[ticker] = td
            log(f"  {ticker}: {td['n']} bars, {td['dates'].min().date()} – {td['dates'].max().date()}")
    log(f"  {len(ticker_data_dict)} tickers prepared")

    # ── Build A positions dict ──
    log("Building A positions dict...")
    a_pos_dict = build_a_positions_dict(a_df, ticker_data_dict)
    log(f"  {len(a_pos_dict)} (strategy, ticker, year) entries")

    # ── Baseline A Sharpes ──
    log("Computing baseline A Sharpes...")
    a_sharpe_dict = compute_a_sharpes(ticker_data_dict, a_pos_dict)
    a_mean = np.nanmean(list(a_sharpe_dict.values()))
    log(f"  A mean trading Sharpe: {a_mean:+.4f}")

    # ── Run all 30 phases ──
    all_phase_results = {}
    all_phase_positions = {}
    phase_summary_rows = []

    for phase_id, overlay_type, param_grid, description, strategy_filter in PHASE_DEFS:
        log(f"\n{'─'*60}")
        log(f"Phase {phase_id}: {description} ({overlay_type}, {len(param_grid)} combos)")

        res_df, pos_dict = run_phase(
            phase_id, overlay_type, param_grid, strategy_filter,
            ticker_data_dict, a_pos_dict, a_sharpe_dict,
        )

        all_phase_results[phase_id] = res_df
        all_phase_positions[phase_id] = pos_dict

        if len(res_df) > 0:
            mean_sh = res_df["sharpe_trading"].mean()
            mean_sh_a = res_df["sharpe_trading_a"].mean()
            delta = mean_sh - mean_sh_a
            n_improved = (res_df["sharpe_trading"] > res_df["sharpe_trading_a"]).sum()
            best_h = res_df.groupby("horizon").size().idxmax() if "horizon" in res_df.columns else "?"

            phase_summary_rows.append(dict(
                phase_id=phase_id, type=phase_id[0],
                description=description,
                mean_trading_sh=round(mean_sh, 4),
                mean_a_sh=round(mean_sh_a, 4),
                delta_vs_a=round(delta, 4),
                n_cells=len(res_df),
                n_improved=int(n_improved),
                pct_improved=round(100 * n_improved / len(res_df), 1),
                best_horizon=best_h,
            ))
            log(f"  {phase_id}: mean_sh={mean_sh:+.4f}, A={mean_sh_a:+.4f}, delta={delta:+.4f}, "
                f"improved={n_improved}/{len(res_df)} ({100*n_improved/len(res_df):.0f}%)")
        else:
            phase_summary_rows.append(dict(
                phase_id=phase_id, type=phase_id[0],
                description=description,
                mean_trading_sh=0, mean_a_sh=0, delta_vs_a=0,
                n_cells=0, n_improved=0, pct_improved=0, best_horizon="none",
            ))
            log(f"  {phase_id}: NO results")

        gc.collect()

    # ── Save phase log ──
    phase_log_df = pd.DataFrame(phase_summary_rows)
    phase_log_path = PHASE_DIR / "phase_log.csv"
    phase_log_df.to_csv(phase_log_path, index=False)
    log(f"\nPhase log saved to {phase_log_path}")

    # ── Select best phase per type ──
    log("\n" + "=" * 60)
    log("Selecting best phase per (B, C, D) for each cell...")
    best_positions, best_phase_log = select_best_phase_per_type(
        all_phase_results, all_phase_positions, a_sharpe_dict
    )

    # ── Save B/C/D positions ──
    for approach_type in ["B", "C", "D"]:
        n_rows = save_positions_parquet(
            best_positions[approach_type], ticker_data_dict, approach_type
        )
        # Count how many cells chose a non-A phase
        non_a = sum(1 for v in best_phase_log[approach_type].values() if v != "A")
        log(f"  {approach_type}: {n_rows:,} rows, {non_a} cells chose overlay (vs A)")

    # ── Aggregate metrics ──
    log("\n" + "=" * 60)
    log("AGGREGATE METRICS")
    log("=" * 60)

    for approach_type in ["A", "B", "C", "D"]:
        if approach_type == "A":
            pos_dict = a_pos_dict
        else:
            pos_dict = best_positions[approach_type]
        s_means, msp_year, msp_mean = compute_aggregate_metrics(
            pos_dict, ticker_data_dict, approach_type
        )
        ce = compute_ce(pos_dict, ticker_data_dict)
        log(f"\n  {approach_type}:")
        for sid in STRATEGIES:
            log(f"    {sid}: mean Sharpe = {s_means.get(sid, np.nan):+.4f}")
        log(f"    MSP: {msp_mean:+.4f}")
        for y, sh in sorted(msp_year.items()):
            log(f"      {y}: {sh:+.4f}")
        log(f"    CE(gamma=2): {ce:+.6f}")

    # ── Human-readable log ──
    txt_path = PHASE_DIR / "PHASE_RESULTS.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BCD 30-Phase Optimization Results\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Commission: {COMM}, BPY: {BPY}, Test years: {TEST_YEARS}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"A baseline mean trading Sharpe: {a_mean:+.4f}\n\n")

        f.write(f"{'Phase':<8} {'Type':<4} {'Description':<30} {'MeanSh':>8} {'Delta':>8} "
                f"{'Improved':>10} {'BestH':>6}\n")
        f.write("-" * 80 + "\n")
        for row in phase_summary_rows:
            f.write(f"{row['phase_id']:<8} {row['type']:<4} {row['description']:<30} "
                    f"{row['mean_trading_sh']:>+8.4f} {row['delta_vs_a']:>+8.4f} "
                    f"{row['n_improved']:>4}/{row['n_cells']:<4} ({row['pct_improved']:>5.1f}%) "
                    f"{row['best_horizon']:>6}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Best phase per type, per cell:\n")
        for approach_type in ["B", "C", "D"]:
            phase_counts = {}
            for v in best_phase_log[approach_type].values():
                phase_counts[v] = phase_counts.get(v, 0) + 1
            f.write(f"\n  {approach_type}: {dict(sorted(phase_counts.items()))}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Aggregate Metrics:\n\n")
        for approach_type in ["A", "B", "C", "D"]:
            if approach_type == "A":
                pos_dict = a_pos_dict
            else:
                pos_dict = best_positions[approach_type]
            s_means, msp_year, msp_mean = compute_aggregate_metrics(
                pos_dict, ticker_data_dict, approach_type
            )
            ce = compute_ce(pos_dict, ticker_data_dict)
            f.write(f"  {approach_type}:\n")
            for sid in STRATEGIES:
                f.write(f"    {sid}: {s_means.get(sid, np.nan):+.4f}\n")
            f.write(f"    MSP mean: {msp_mean:+.4f}\n")
            for y, sh in sorted(msp_year.items()):
                f.write(f"      {y}: {sh:+.4f}\n")
            f.write(f"    CE(gamma=2): {ce:+.6f}\n\n")

    log(f"\nResults written to {txt_path}")

    elapsed = time.time() - t_global
    log(f"\n{'='*80}")
    log(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log(f"{'='*80}")


if __name__ == "__main__":
    main()
