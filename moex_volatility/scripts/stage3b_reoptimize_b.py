#!/usr/bin/env python3
"""
stage3b_reoptimize_b.py — Re-optimize B overlay with strategy-type-specific grids.

For each strategy x ticker x test_year:
  1. Load A positions (wf_positions_A.parquet — ALL days including zeros)
  2. Load close prices (ohlcv_daily_full.parquet)
  3. Load vpred (sigma_h1, sigma_h5, sigma_h22)
  4. Apply B overlay using strategy-type-specific grid
  5. Walk-forward: optimize B params on TRAIN, apply on TEST

Grids:
  - Contrarian (S1, S2): wide stops, short hold  — 3200 combos
  - Trend (S3, S4): tight stops, long hold         — 3675 combos
  - Range (S5, S6): medium stops                    — 1728 combos

Output:
  - walkforward/wf_positions_B.parquet  (date, ticker, strategy, year, position, daily_gross_return)
  - walkforward/wf_B_results.csv        (per ticker x year metrics)
"""
import os, sys, warnings, time, itertools
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

np.random.seed(42)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator
    print("WARNING: numba not installed, will be very slow")

# ═══════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
DATA_DIR = BASE / "results/final/strategies/data"
WF_DIR = BASE / "results/final/strategies/walkforward"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")

BPY = 252
WARMUP = 200
COMM = 0.0005  # training Sharpe commission
TEST_YEARS = [2022, 2023, 2024, 2025]
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
HORIZONS = ["h1", "h5", "h22"]

TICKERS = sorted([
    "AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
    "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR",
])

STRATEGY_TYPE = {
    "S1": "contrarian", "S2": "contrarian",
    "S3": "trend", "S4": "trend",
    "S5": "range", "S6": "range",
}

# ═══════════════════════════════════════════════════════════════
# B GRIDS by strategy type
# ═══════════════════════════════════════════════════════════════

B_GRID_CONTRARIAN = {  # S1, S2: wide stops, short hold
    "k_sl":       [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0],
    "ratio":      [0, 0.5, 1.0, 1.5, 2.0],
    "base_hold":  [2, 3, 5, 7, 10],
    "hold_gamma": [0, 0.25, 0.5, 1.0],
    "be_trigger": [0, 1.0, 2.0, 3.0, 5.0],
}

B_GRID_TREND = {  # S3, S4: tight stops, long hold
    "k_sl":       [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "ratio":      [0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    "base_hold":  [10, 15, 20, 30, 40, 50, 70],
    "hold_gamma": [0.25, 0.5, 1.0, 1.5, 2.0],
    "be_trigger": [0, 0.3, 0.5, 1.0, 1.5],
}

B_GRID_RANGE = {  # S5, S6: medium stops
    "k_sl":       [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "ratio":      [0.5, 1.0, 1.5, 2.0, 3.0],
    "base_hold":  [3, 5, 7, 10, 15, 20],
    "hold_gamma": [0, 0.5, 1.0, 1.5],
    "be_trigger": [0, 0.5, 1.0, 2.0],
}

B_GRIDS = {
    "contrarian": B_GRID_CONTRARIAN,
    "trend":      B_GRID_TREND,
    "range":      B_GRID_RANGE,
}


def expand_grid(d):
    """Cartesian product of grid dict → list of tuples."""
    keys = sorted(d.keys())
    vals = [d[k] for k in keys]
    return keys, list(itertools.product(*vals))


# ═══════════════════════════════════════════════════════════════
# _apply_B_numba (copied from stage3_walkforward.py with b_exited fix)
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _apply_B_numba(a_pos, close, vol_pred, vol_median, k_sl, ratio, base_hold, hold_gamma, be_trigger):
    """Adaptive stops using vpred. Overlay on A positions.
    When A enters: set vpred-based SL/TP.
    SL = entry * (1 - k_sl * vpred), TP = entry * (1 + k_sl * ratio * vpred)
    Dynamic hold = base_hold * (vol_median / vol_pred)^hold_gamma
    Breakeven trigger at be_trigger * vpred * entry_price
    """
    n = len(a_pos)
    out = np.zeros(n)
    in_trade = False
    entry_price = 0.0
    direction = 0.0
    sl = 0.0; tp = 0.0
    held = 0; max_hold = base_hold
    be_active = False
    b_exited = False  # True when B closed this A-trade via SL/TP/maxhold

    for t in range(n):
        # Reset b_exited when A exits (new A-trade can re-enter)
        if abs(a_pos[t]) < 0.01:
            b_exited = False

        if in_trade:
            held += 1
            # Breakeven
            if be_trigger > 0 and not be_active:
                sv = vol_pred[t]
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

            # Check if A exits
            if abs(a_pos[t]) < 0.01:
                closed = True

            if closed:
                in_trade = False
                held = 0
                be_active = False
                # Mark that B exited this A-trade (don't re-enter until A starts new trade)
                if abs(a_pos[t]) > 0.01:
                    b_exited = True
            else:
                out[t] = direction

        if not in_trade and not b_exited and abs(a_pos[t]) > 0.01:
            # A enters a new trade
            sv = vol_pred[t]
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
            # Dynamic hold
            if hold_gamma > 0 and sv > 1e-8 and vol_median > 1e-8:
                mh = base_hold * (vol_median / sv) ** hold_gamma
                max_hold = int(min(max(mh, 3.0), base_hold * 3.0))
            else:
                max_hold = base_hold
            out[t] = direction

    return out


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def trading_sharpe(pos, log_ret, mask, comm):
    """TRAIN metric: simple Sharpe, no FDR, no RF, with commission."""
    pm = pos[mask]; lr = log_ret[mask]
    n = len(pm)
    if n < 20:
        return -999.0
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    s = np.std(net, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(net) * sqrt(252) / s


def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm):
    """OOS metric: Excess Sharpe with FDR, RF, commission."""
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
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
    sharpe = np.mean(excess) * sqrt(252) / s
    exp_pct = np.mean(exposure) * 100
    # Trades count
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
    cs = max((n / BPY), 0.01)
    return sharpe, dict(
        exposure=round(exp_pct, 1),
        tpy=round(nt / cs, 1),
        wr=round(nw / nt * 100 if nt > 0 else 0, 1),
        n_trades=nt,
    )


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

LOG_PATH = BASE / "results/final/strategies/STAGE3B_LOG.txt"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_all_data():
    """Load daily OHLCV, vpred, key rate, and A positions."""
    log("Loading data...")

    # Daily OHLCV
    daily_df = pd.read_parquet(DATA_DIR / "ohlcv_daily_full.parquet")
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    log(f"  Daily: {len(daily_df):,} rows, "
        f"{daily_df['date'].min().date()} - {daily_df['date'].max().date()}")

    # Vpred
    vpred_df = pd.read_parquet(DATA_DIR / "vpred_aligned.parquet")
    vpred_df["date"] = pd.to_datetime(vpred_df["date"])
    log(f"  Vpred: {len(vpred_df):,} rows, "
        f"{vpred_df['date'].min().date()} - {vpred_df['date'].max().date()}")

    # Key rate
    kr = pd.read_parquet(KR_PATH)
    kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    log(f"  Key rate: {len(kr_s)} dates")

    # A positions
    a_pos_df = pd.read_parquet(WF_DIR / "wf_positions_A.parquet")
    a_pos_df["date"] = pd.to_datetime(a_pos_df["date"])
    log(f"  A positions: {len(a_pos_df):,} rows, "
        f"strategies={sorted(a_pos_df['strategy'].unique())}, "
        f"years={sorted(a_pos_df['year'].unique())}")

    return daily_df, vpred_df, kr_s, a_pos_df


def prepare_ticker_data(ticker, daily_df, kr_s, vpred_df):
    """Prepare close, lr, rf_d, cal_days, sigma_dict for one ticker."""
    tdf = daily_df[daily_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
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
        ticker=ticker, close=close, dates=dates, n=n,
        lr=lr, cal_days=cal_days, rf_d=rf_d,
        sigma_dict=sigma_dict,
    )


def build_a_positions_lookup(a_pos_df, ticker_data_dict):
    """Build full-length A position arrays aligned to ticker dates.

    Returns: dict[(sid, ticker, year)] -> np.ndarray of length n (full date range).
    """
    lookup = {}
    for (sid, ticker, year), grp in a_pos_df.groupby(["strategy", "ticker", "year"]):
        td = ticker_data_dict.get(ticker)
        if td is None:
            continue
        n = td["n"]
        dates = td["dates"]
        a_arr = np.zeros(n, dtype=np.float64)
        grp_dates = pd.to_datetime(grp["date"].values)
        grp_pos = grp["position"].values.astype(np.float64)
        idx = np.searchsorted(dates, grp_dates)
        valid = (idx >= 0) & (idx < n)
        a_arr[idx[valid]] = grp_pos[valid]
        lookup[(sid, ticker, year)] = a_arr
    return lookup


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD B OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

COARSE_THRESHOLD = 5000  # use 2-pass if combos > this
COARSE_STEP = 3          # every Nth combo in coarse pass
COARSE_TOP_K = 50        # top-K from coarse pass to refine


def _neighbors(combo, keys, grid):
    """Generate neighboring combos (+-1 step in each dimension) for fine search."""
    neighbors = set()
    neighbors.add(combo)
    for dim in range(len(keys)):
        key = keys[dim]
        vals = grid[key]
        cur_val = combo[dim]
        if cur_val in vals:
            cur_idx = vals.index(cur_val)
        else:
            # find closest
            cur_idx = min(range(len(vals)), key=lambda i: abs(vals[i] - cur_val))
        for offset in [-1, 0, 1]:
            ni = cur_idx + offset
            if 0 <= ni < len(vals):
                new_combo = list(combo)
                new_combo[dim] = vals[ni]
                neighbors.add(tuple(new_combo))
    return list(neighbors)


def walkforward_B_reoptimize(ticker_data_dict, a_lookup):
    """Walk-forward B with strategy-type-specific grids.
    Uses 2-pass optimization for grids > COARSE_THRESHOLD combos.
    """
    log("=" * 70)
    log("WALK-FORWARD B RE-OPTIMIZATION (strategy-type-specific grids)")
    log("=" * 70)

    # Pre-expand grids
    grids_expanded = {}
    for stype, grid in B_GRIDS.items():
        keys, combos = expand_grid(grid)
        use_2pass = len(combos) > COARSE_THRESHOLD
        grids_expanded[stype] = (keys, combos, use_2pass)
        tag = " [2-PASS]" if use_2pass else ""
        log(f"  Grid {stype}: {len(combos)} combos x {len(HORIZONS)} horizons "
            f"= {len(combos) * len(HORIZONS)} total{tag}")

    results_rows = []
    positions_dict = {}  # (sid, ticker, year) -> full-length b_pos array

    total_cells = len(STRATEGIES) * len(TICKERS) * len(TEST_YEARS)
    done = 0
    t_start = time.time()

    for sid in STRATEGIES:
        stype = STRATEGY_TYPE[sid]
        keys, combos, use_2pass = grids_expanded[stype]
        n_combos = len(combos)
        grid_dict = B_GRIDS[stype]
        # key order from sorted: base_hold, be_trigger, hold_gamma, k_sl, ratio
        idx_base_hold  = keys.index("base_hold")
        idx_be_trigger = keys.index("be_trigger")
        idx_hold_gamma = keys.index("hold_gamma")
        idx_k_sl       = keys.index("k_sl")
        idx_ratio      = keys.index("ratio")

        for ticker in TICKERS:
            td = ticker_data_dict.get(ticker)
            if td is None:
                done += len(TEST_YEARS)
                continue
            close = td["close"]; lr = td["lr"]; rf_d = td["rf_d"]
            cal_days = td["cal_days"]; dates = td["dates"]; n = td["n"]
            sigma_dict = td["sigma_dict"]

            for test_year in TEST_YEARS:
                t0 = time.time()
                done += 1

                # Build full A position by merging all year chunks
                # A positions parquet only has test years (2022-2025)
                # _apply_B_numba needs full-length array
                a_pos_full = np.zeros(n, dtype=np.float64)
                for y in TEST_YEARS:
                    a_y = a_lookup.get((sid, ticker, y))
                    if a_y is not None:
                        mask_nonzero = np.abs(a_y) > 1e-8
                        a_pos_full[mask_nonzero] = a_y[mask_nonzero]

                # Train/test masks
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

                def _eval_combo(combo, sigma, vol_median, horizon):
                    """Evaluate one combo, return (sharpe, params_tuple)."""
                    bh  = combo[idx_base_hold]
                    be  = combo[idx_be_trigger]
                    hg  = combo[idx_hold_gamma]
                    ksl = combo[idx_k_sl]
                    rat = combo[idx_ratio]
                    b_pos = _apply_B_numba(
                        a_pos_full, close, sigma, vol_median,
                        ksl, rat, bh, hg, be
                    )
                    sh = trading_sharpe(b_pos, lr, train_mask, COMM)
                    return sh, (ksl, rat, bh, hg, be, horizon)

                if use_2pass:
                    # PASS 1: coarse (every COARSE_STEP-th combo)
                    coarse_combos = combos[::COARSE_STEP]
                    coarse_results = []  # (sharpe, combo, horizon)

                    for horizon in HORIZONS:
                        sigma = sigma_dict.get(horizon)
                        if sigma is None:
                            continue
                        sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                        vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02

                        for combo in coarse_combos:
                            sh, params = _eval_combo(combo, sigma, vol_median, horizon)
                            coarse_results.append((sh, combo, horizon))

                    # Sort by sharpe, take top-K
                    coarse_results.sort(key=lambda x: -x[0])
                    top_coarse = coarse_results[:COARSE_TOP_K]

                    # PASS 2: fine search around top-K (neighbors)
                    fine_combos_set = set()
                    for _, combo, horizon in top_coarse:
                        for nb in _neighbors(combo, keys, grid_dict):
                            fine_combos_set.add((nb, horizon))

                    for (combo, horizon) in fine_combos_set:
                        sigma = sigma_dict.get(horizon)
                        if sigma is None:
                            continue
                        sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                        vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02

                        sh, params = _eval_combo(combo, sigma, vol_median, horizon)
                        if sh > best_sh_train:
                            best_sh_train = sh
                            best_params = params

                else:
                    # Single pass: evaluate all combos
                    for horizon in HORIZONS:
                        sigma = sigma_dict.get(horizon)
                        if sigma is None:
                            continue
                        sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                        vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02

                        for combo in combos:
                            sh, params = _eval_combo(combo, sigma, vol_median, horizon)
                            if sh > best_sh_train:
                                best_sh_train = sh
                                best_params = params

                # Apply best on full data, extract test
                if best_params is not None:
                    ksl, rat, bh, hg, be, horizon = best_params
                    sigma = sigma_dict.get(horizon, np.full(n, 0.02))
                    sigma_pos = sigma[~np.isnan(sigma) & (sigma > 1e-6)]
                    vol_median = float(np.nanmedian(sigma_pos)) if len(sigma_pos) > 0 else 0.02
                    b_pos = _apply_B_numba(
                        a_pos_full, close, sigma, vol_median,
                        ksl, rat, bh, hg, be
                    )
                else:
                    # Fallback: use A positions
                    b_pos = a_pos_full.copy()

                positions_dict[(sid, ticker, test_year)] = b_pos

                # OOS metrics (excess Sharpe)
                sh_excess, info = calc_excess_sharpe(b_pos, lr, rf_d, cal_days, test_mask, COMM)

                # Also compute trading sharpe on test for comparison
                sh_trading = trading_sharpe(b_pos, lr, test_mask, COMM)

                # Sharpe excess vs 9% benchmark
                pm_test = b_pos[test_mask]; lr_test = lr[test_mask]
                dp_test = np.diff(pm_test, prepend=0.0)
                net_test = pm_test * lr_test - np.abs(dp_test) * COMM
                bench_daily = 0.09 / 252
                excess_9 = net_test - bench_daily
                s9 = np.std(excess_9, ddof=1)
                sh_excess_9pct = float(np.mean(excess_9) * sqrt(252) / s9) if s9 > 1e-12 else 0.0

                results_rows.append(dict(
                    strategy=sid, ticker=ticker, year=test_year,
                    approach="B",
                    sharpe_trading=round(sh_trading, 4),
                    sharpe_excess=round(sh_excess, 4),
                    sharpe_excess_9pct=round(sh_excess_9pct, 4),
                    train_sharpe=round(best_sh_train, 4),
                    exposure=info.get("exposure", 0),
                    tpy=info.get("tpy", 0),
                    wr=info.get("wr", 0),
                    n_trades=info.get("n_trades", 0),
                    comm=COMM,
                    best_k_sl=best_params[0] if best_params else None,
                    best_ratio=best_params[1] if best_params else None,
                    best_base_hold=best_params[2] if best_params else None,
                    best_hold_gamma=best_params[3] if best_params else None,
                    best_be_trigger=best_params[4] if best_params else None,
                    best_horizon=best_params[5] if best_params else None,
                    strategy_type=stype,
                ))

                elapsed = time.time() - t0
                total_elapsed = time.time() - t_start
                eta = total_elapsed / done * (total_cells - done) if done > 0 else 0
                if done % 10 == 0 or elapsed > 5:
                    log(f"  [{done}/{total_cells}] {sid} {ticker} {test_year}: "
                        f"trainSh={best_sh_train:+.2f} testSh={sh_trading:+.2f} "
                        f"exSh={sh_excess:+.2f} h={best_params[5] if best_params else '?'} "
                        f"{elapsed:.1f}s (ETA {eta/60:.0f}min)")

    return results_rows, positions_dict


# ═══════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════

def save_positions_parquet(positions_dict, ticker_data_dict):
    """Save B positions as parquet (same format as wf_positions_A.parquet)."""
    rows = []
    for (sid, ticker, year), pos_arr in positions_dict.items():
        td = ticker_data_dict.get(ticker)
        if td is None:
            continue
        dates = td["dates"]; n = td["n"]; lr_t = td["lr"]
        test_start = np.searchsorted(dates, pd.Timestamp(f"{year}-01-01"))
        test_end = np.searchsorted(dates, pd.Timestamp(f"{year + 1}-01-01"))
        for t in range(test_start, min(test_end, n)):
            rows.append(dict(
                date=dates[t],
                ticker=ticker,
                strategy=sid,
                year=year,
                position=round(float(pos_arr[t]), 6),
                daily_gross_return=round(float(pos_arr[t] * lr_t[t]), 8),
            ))
    if rows:
        df = pd.DataFrame(rows)
        fpath = WF_DIR / "wf_positions_B.parquet"
        df.to_parquet(fpath, index=False)
        log(f"  Saved {fpath} ({len(df):,} rows)")
    return len(rows)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_global = time.time()
    WF_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("stage3b_reoptimize_b.py — B Re-Optimization with Type-Specific Grids")
    log(f"Test years: {TEST_YEARS}")
    log(f"Strategies: {STRATEGIES}")
    log(f"Tickers: {len(TICKERS)}")
    log(f"Commission: {COMM}")
    log("=" * 80)

    # Print grid sizes
    for stype, grid in B_GRIDS.items():
        sizes = [len(v) for v in grid.values()]
        total = 1
        for s in sizes:
            total *= s
        strats = [s for s, t in STRATEGY_TYPE.items() if t == stype]
        log(f"  {stype} ({', '.join(strats)}): {total} combos x {len(HORIZONS)} horizons = {total * len(HORIZONS)}")

    # Load data
    daily_df, vpred_df, kr_s, a_pos_df = load_all_data()

    # Warmup numba
    log("Warming up numba...")
    _test_pos = np.zeros(50)
    _test_pos[10:15] = 1.0
    _test_close = np.random.randn(50).cumsum() + 100
    _test_sigma = np.ones(50) * 0.02
    _ = _apply_B_numba(_test_pos, _test_close, _test_sigma, 0.02, 1.0, 1.5, 15, 0.5, 1.0)
    log("  Numba warmup done")

    # Prepare per-ticker data
    log("Preparing per-ticker data...")
    ticker_data_dict = {}
    for ticker in TICKERS:
        td = prepare_ticker_data(ticker, daily_df, kr_s, vpred_df)
        if td is not None:
            ticker_data_dict[ticker] = td
            log(f"  {ticker}: {td['n']} bars, {td['dates'].min().date()} - {td['dates'].max().date()}")
        else:
            log(f"  {ticker}: SKIPPED (too few data)")
    log(f"  {len(ticker_data_dict)} tickers prepared")

    # Build A positions lookup
    log("Building A positions lookup...")
    a_lookup = build_a_positions_lookup(a_pos_df, ticker_data_dict)
    log(f"  {len(a_lookup)} (strategy, ticker, year) entries")

    # Run walk-forward B
    results_rows, positions_dict = walkforward_B_reoptimize(ticker_data_dict, a_lookup)

    # Save results
    log("\nSaving results...")
    results_df = pd.DataFrame(results_rows)
    results_path = WF_DIR / "wf_B_results.csv"
    results_df.to_csv(results_path, index=False)
    log(f"  Saved {results_path} ({len(results_df)} rows)")

    # Save positions
    n_pos = save_positions_parquet(positions_dict, ticker_data_dict)

    # Print summary
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    log(f"{'Strategy':<8} {'Type':<12} {'Mean Sh_tr':>10} {'Mean Sh_ex':>10} {'Mean Sh_9%':>10}")
    log("-" * 55)
    for sid in STRATEGIES:
        sub = results_df[results_df["strategy"] == sid]
        if sub.empty:
            continue
        # Per-year mean, then mean across years
        yr_sh_tr = [sub[sub["year"] == y]["sharpe_trading"].mean() for y in TEST_YEARS]
        yr_sh_ex = [sub[sub["year"] == y]["sharpe_excess"].mean() for y in TEST_YEARS]
        yr_sh_9  = [sub[sub["year"] == y]["sharpe_excess_9pct"].mean() for y in TEST_YEARS]
        log(f"{sid:<8} {STRATEGY_TYPE[sid]:<12} "
            f"{np.nanmean(yr_sh_tr):>+10.3f} {np.nanmean(yr_sh_ex):>+10.3f} {np.nanmean(yr_sh_9):>+10.3f}")

    elapsed = time.time() - t_global
    log(f"\nTotal time: {elapsed/60:.1f} min")
    log("Done.")


if __name__ == "__main__":
    main()
