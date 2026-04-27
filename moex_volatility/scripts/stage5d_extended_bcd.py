#!/usr/bin/env python3
"""
stage5d_extended_bcd.py — Extended BCD overlays: combos + portfolio + bootstrap.

Extends B/C/D with combo overlays (BC, BD, CD, BCD) using the intersection
approach on pre-computed positions from the v4 walk-forward.

PART 1: Per-ticker BCD combos on existing positions
  For each strategy × ticker × year, compute combo positions:
    BC  = C filter ∩ B stops (trade when both agree, use min size)
    BD  = D scaling ∩ B stops (B direction, D magnitude)
    CD  = C filter + D scaling (trade D only where C allows)
    BCD = C filter ∩ B stops ∩ D scaling (all three agree)

PART 2: Per-strategy portfolio aggregation
  For each of 8 approaches (A, B, C, D, BC, BD, CD, BCD):
    - Build EW portfolio across 17 tickers per strategy
    - Compute 3 Sharpe metrics with FDR
    - Save daily returns for bootstrap

PART 3: MSP aggregation
  Aggregate best per-strategy portfolios into MSP for all 8 approaches.

PART 4: Bootstrap
  Test each overlay vs A:
    - Per-strategy: 7 overlays × 6 strategies × 2 comms = 84 tests
    - MSP: 7 overlays × 2 comms = 14 tests
  Total: 98 bootstrap tests

OUTPUT:
  msp/bcd_extended_perstrat.csv
  msp/bcd_extended_msp.csv
  msp/bcd_extended_daily_returns/  (daily returns for bootstrap)
  bootstrap/bootstrap_bcd_extended.csv
"""
import os, sys, warnings, datetime, time
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# Paths & Constants
# ══════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
STRAT_DIR = BASE / "results/final/strategies"
WF_DIR = STRAT_DIR / "walkforward"
MSP_DIR = STRAT_DIR / "msp"
BOOT_DIR = STRAT_DIR / "bootstrap"
BCD_RET_DIR = MSP_DIR / "bcd_extended_daily_returns"
LOG_PATH = STRAT_DIR / "MASTER_PROGRESS_LOG.txt"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")

for d in [MSP_DIR, BOOT_DIR, BCD_RET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BPY = 252
RF_MEAN_HIST = 0.09
N_BOOT = 10000
SEED = 42

TICKERS = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
           "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"]
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
OOS_YEARS = [2022, 2023, 2024, 2025]
BASE_APPROACHES = ["A", "B", "C", "D"]
COMBO_APPROACHES = ["BC", "BD", "CD", "BCD"]
ALL_APPROACHES = BASE_APPROACHES + COMBO_APPROACHES
COMM_DAILY = {"inst": 0.0005, "std": 0.0006}


# ══════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage5d_bcd] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# Load helpers
# ══════════════════════════════════════════════════════════════════════
def load_key_rate():
    if KR_PATH.exists():
        kr = pd.read_parquet(KR_PATH)
        kr["date"] = pd.to_datetime(kr["date"])
        return kr.set_index("date")["key_rate"].sort_index()
    log("WARNING: key rate file not found, using 0.16 flat")
    dates = pd.bdate_range("2019-01-01", "2026-06-01")
    return pd.Series(0.16, index=dates, name="key_rate")


def load_positions():
    """Load daily WF positions for A/B/C/D."""
    pos_data = {}
    for app in BASE_APPROACHES:
        fp = WF_DIR / f"wf_positions_{app}.parquet"
        if fp.exists():
            tmp = pd.read_parquet(fp)
            tmp["date"] = pd.to_datetime(tmp["date"])
            pos_data[app] = tmp
            log(f"  Loaded {fp.name}: {len(tmp):,} rows")
        else:
            log(f"  WARNING: {fp} not found")
    if not pos_data:
        raise FileNotFoundError("No position files found.")
    return pos_data


# ══════════════════════════════════════════════════════════════════════
# Combo overlay functions
# ══════════════════════════════════════════════════════════════════════
EPS = 1e-8


def combo_bc(pos_b, pos_c):
    """BC: trade when both B and C agree. Use C direction, min of abs sizes."""
    b_active = np.abs(pos_b) > EPS
    c_active = np.abs(pos_c) > EPS
    both = b_active & c_active
    return np.where(both,
                    np.sign(pos_c) * np.minimum(np.abs(pos_b), np.abs(pos_c)),
                    0.0)


def combo_bd(pos_b, pos_d):
    """BD: D scaling with B stops. B direction/timing, D magnitude."""
    b_active = np.abs(pos_b) > EPS
    return np.where(b_active,
                    np.sign(pos_b) * np.abs(pos_d),
                    0.0)


def combo_cd(pos_c, pos_d):
    """CD: C regime filter with D vol scaling. Trade D only where C allows."""
    c_active = np.abs(pos_c) > EPS
    return np.where(c_active, pos_d, 0.0)


def combo_bcd(pos_b, pos_c, pos_d):
    """BCD: C filter, D scaling, B stops. All three must agree."""
    b_active = np.abs(pos_b) > EPS
    c_active = np.abs(pos_c) > EPS
    both = b_active & c_active
    return np.where(both,
                    np.sign(pos_c) * np.abs(pos_d),
                    0.0)


# ══════════════════════════════════════════════════════════════════════
# Calendar-days helper
# ══════════════════════════════════════════════════════════════════════
def compute_cal_days(dates_index):
    cal_days = np.ones(len(dates_index))
    for i in range(1, len(dates_index)):
        cal_days[i] = (dates_index[i] - dates_index[i - 1]).days
    return cal_days


# ══════════════════════════════════════════════════════════════════════
# Metrics (same as stage5c)
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(total, excess, trading_ret, exp_arr, cal_days):
    """Compute performance metrics from arrays."""
    n = len(total)
    if n < 30:
        return {k: np.nan for k in [
            "sharpe_excess", "sharpe_trading", "sharpe_excess_9pct",
            "sortino", "calmar", "max_dd", "ann_ret", "ann_vol", "exposure"]}

    n_years = max(n / BPY, 0.01)
    cum = np.sum(total)
    ann_ret = cum / n_years
    ann_vol = np.std(total, ddof=1) * sqrt(BPY)

    s_total = np.std(total, ddof=1)
    sharpe_excess = np.mean(excess) * sqrt(BPY) / s_total if s_total > EPS else 0.0

    s_trad = np.std(trading_ret, ddof=1)
    sharpe_trading = np.mean(trading_ret) * sqrt(BPY) / s_trad if s_trad > EPS else 0.0

    # Counterfactual excess (RF = 9%)
    rf_cf = RF_MEAN_HIST / 365.0
    fdr_cf = np.maximum(0.0, 1.0 - exp_arr) * rf_cf * cal_days
    margin_cf = np.maximum(0.0, exp_arr - 1.0) * (RF_MEAN_HIST + 0.02) / 365.0
    total_cf = trading_ret + fdr_cf - margin_cf
    excess_cf = total_cf - rf_cf * cal_days
    s_cf = np.std(total_cf, ddof=1)
    sharpe_excess_9pct = np.mean(excess_cf) * sqrt(BPY) / s_cf if s_cf > EPS else 0.0

    # Sortino
    downside = total[total < 0]
    ds_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-12
    sortino = np.mean(excess) * sqrt(BPY) / ds_std if ds_std > EPS else 0.0

    # Max drawdown
    cum_arr = np.cumsum(total)
    peak = np.maximum.accumulate(cum_arr)
    dd = peak - cum_arr
    max_dd = np.max(dd) * 100 if len(dd) > 0 else 0.0

    calmar = ann_ret / (max_dd / 100) if max_dd > 1e-6 else 0.0
    avg_exp = np.mean(exp_arr) * 100

    return {
        "sharpe_trading": round(sharpe_trading, 3),
        "sharpe_excess": round(sharpe_excess, 3),
        "sharpe_excess_9pct": round(sharpe_excess_9pct, 3),
        "ann_ret": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "max_dd": round(max_dd, 2),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "exposure": round(avg_exp, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# Sharpe helper for bootstrap
# ══════════════════════════════════════════════════════════════════════
def _sharpe(ret):
    s = np.std(ret, ddof=1)
    return np.mean(ret) * sqrt(BPY) / s if s > EPS else 0.0


def bootstrap_sharpe_diff(ret_new, ret_base, n_boot=N_BOOT, seed=SEED):
    """Bootstrap test: H0 = Sharpe(new) <= Sharpe(base).
    Returns (obs_delta, p_value, ci_lo, ci_hi).
    """
    n = min(len(ret_new), len(ret_base))
    if n < 30:
        return 0.0, 1.0, 0.0, 0.0
    ret_new = ret_new[:n]
    ret_base = ret_base[:n]
    obs = _sharpe(ret_new) - _sharpe(ret_base)
    rng = np.random.RandomState(seed)
    boot_deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_deltas[b] = _sharpe(ret_new[idx]) - _sharpe(ret_base[idx])
    p_value = np.mean(boot_deltas <= 0)
    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)
    return obs, p_value, ci_lo, ci_hi


# ══════════════════════════════════════════════════════════════════════
# PART 1: Compute combo positions from existing A/B/C/D
# ══════════════════════════════════════════════════════════════════════
def build_combo_positions(pos_data):
    """Build BC, BD, CD, BCD positions from existing A/B/C/D positions.
    Returns pos_data dict augmented with combo positions (same parquet format).
    """
    log("=" * 70)
    log("PART 1: Building combo positions (BC, BD, CD, BCD)")
    log("=" * 70)
    t0 = time.time()

    # For each combo, we need to align B/C/D positions by (strategy, ticker, date)
    # Build lookup: (strategy, ticker, date) -> position value for each approach
    required = {"B", "C", "D"}
    if not required.issubset(pos_data.keys()):
        missing = required - set(pos_data.keys())
        raise ValueError(f"Missing required approach positions: {missing}")

    # We use A's structure as reference (since B/C/D are overlays on A,
    # they share the same date/ticker/strategy/year grid — mostly)
    ref_df = pos_data["A"].copy()

    # Index all approach DataFrames by (strategy, ticker, date) for fast lookup
    indexed = {}
    for app in ["A", "B", "C", "D"]:
        df = pos_data[app].copy()
        df = df.set_index(["strategy", "ticker", "date"])
        indexed[app] = df

    combo_dfs = {}
    for combo_name in COMBO_APPROACHES:
        log(f"  Computing {combo_name} positions...")
        rows = []

        for sid in STRATEGIES:
            for tk in TICKERS:
                # Get positions for each base approach
                try:
                    b_sub = indexed["B"].loc[(sid, tk)].sort_index()
                    c_sub = indexed["C"].loc[(sid, tk)].sort_index()
                    d_sub = indexed["D"].loc[(sid, tk)].sort_index()
                except KeyError:
                    continue

                # Align on common dates
                common_dates = b_sub.index.intersection(c_sub.index).intersection(d_sub.index)
                if len(common_dates) == 0:
                    continue

                pos_b = b_sub.loc[common_dates, "position"].values
                pos_c = c_sub.loc[common_dates, "position"].values
                pos_d = d_sub.loc[common_dates, "position"].values

                # Also need gross returns from A (base) for return attribution
                try:
                    a_sub = indexed["A"].loc[(sid, tk)].sort_index()
                    # Align A to common dates
                    a_common = a_sub.reindex(common_dates)
                    gross_ret = a_common["daily_gross_return"].values
                    year_arr = a_common["year"].values
                except KeyError:
                    gross_ret = np.zeros(len(common_dates))
                    year_arr = np.array([d.year for d in common_dates])

                # Compute combo position
                if combo_name == "BC":
                    combo_pos = combo_bc(pos_b, pos_c)
                elif combo_name == "BD":
                    combo_pos = combo_bd(pos_b, pos_d)
                elif combo_name == "CD":
                    combo_pos = combo_cd(pos_c, pos_d)
                elif combo_name == "BCD":
                    combo_pos = combo_bcd(pos_b, pos_c, pos_d)
                else:
                    combo_pos = np.zeros(len(common_dates))

                # Compute daily_gross_return for combo:
                # gross_return = position * log_return_of_underlying
                # Since we have A's gross_return = A_pos * log_ret,
                # and log_ret = A_gross_return / A_pos (where A_pos != 0),
                # combo_gross_return = combo_pos * log_ret = combo_pos * (A_gross / A_pos)
                # Handle A_pos = 0 case (log_ret unavailable from A, use 0)
                try:
                    a_pos = a_common["position"].values
                except Exception:
                    a_pos = np.zeros(len(common_dates))

                with np.errstate(divide='ignore', invalid='ignore'):
                    # log_return per unit = gross_ret / a_pos where a_pos != 0
                    a_abs = np.abs(a_pos)
                    log_ret = np.where(a_abs > EPS,
                                       gross_ret / a_pos,
                                       0.0)
                    # Handle NaN from 0/0
                    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

                combo_gross = combo_pos * log_ret

                for i, dt in enumerate(common_dates):
                    rows.append({
                        "date": dt,
                        "ticker": tk,
                        "strategy": sid,
                        "year": int(year_arr[i]) if not np.isnan(year_arr[i]) else dt.year,
                        "position": combo_pos[i],
                        "daily_gross_return": combo_gross[i],
                    })

        combo_df = pd.DataFrame(rows)
        combo_df["date"] = pd.to_datetime(combo_df["date"])
        combo_dfs[combo_name] = combo_df
        log(f"    {combo_name}: {len(combo_df):,} rows, "
            f"exposure={np.mean(np.abs(combo_df['position'].values))*100:.1f}%")

    elapsed = time.time() - t0
    log(f"PART 1 complete in {elapsed:.1f}s")

    return combo_dfs


# ══════════════════════════════════════════════════════════════════════
# PART 2: Per-strategy EW portfolio for all 8 approaches
# ══════════════════════════════════════════════════════════════════════
def run_part2(pos_data, combo_dfs, kr_series):
    """Build EW portfolio per strategy × approach × commission.
    Returns (results_df, daily_returns_cache).
    """
    log("")
    log("=" * 70)
    log("PART 2: Per-Strategy EW Portfolio (8 approaches)")
    log("=" * 70)
    t0 = time.time()

    # Merge all position sources
    all_pos = {}
    for app in BASE_APPROACHES:
        all_pos[app] = pos_data[app]
    for app in COMBO_APPROACHES:
        all_pos[app] = combo_dfs[app]

    all_rows = []
    daily_returns_cache = {}  # key = (sid, app, comm_label) -> DataFrame

    total_combos = len(STRATEGIES) * len(ALL_APPROACHES) * len(COMM_DAILY)
    done = 0

    for app in ALL_APPROACHES:
        app_df = all_pos[app]

        for sid in STRATEGIES:
            strat_df = app_df[app_df["strategy"] == sid].copy()
            if strat_df.empty:
                done += len(COMM_DAILY)
                continue

            # Build wide DataFrames: date × ticker for gross_return and position
            ticker_frames = {}
            for tk in TICKERS:
                sub = strat_df[strat_df["ticker"] == tk].sort_values("date").reset_index(drop=True)
                if len(sub) > 0:
                    ticker_frames[tk] = sub

            all_dates = sorted(strat_df["date"].unique())
            ret_wide = pd.DataFrame(0.0, index=all_dates, columns=TICKERS)
            pos_wide = pd.DataFrame(0.0, index=all_dates, columns=TICKERS)

            for tk, sub in ticker_frames.items():
                sub_idx = sub.set_index("date")
                common_idx = ret_wide.index.intersection(sub_idx.index)
                ret_wide.loc[common_idx, tk] = sub_idx.loc[common_idx, "daily_gross_return"].values
                pos_wide.loc[common_idx, tk] = sub_idx.loc[common_idx, "position"].values

            ret_wide.index = pd.DatetimeIndex(ret_wide.index)
            pos_wide.index = pd.DatetimeIndex(pos_wide.index)

            for comm_label, comm_val in COMM_DAILY.items():
                done += 1

                # Net returns: gross - |delta_pos| * comm
                net_wide = ret_wide.copy()
                for tk in TICKERS:
                    p = pos_wide[tk].values
                    dp = np.abs(np.diff(p, prepend=0.0))
                    net_wide[tk] = ret_wide[tk].values - dp * comm_val

                # EW portfolio
                n_tk = len(TICKERS)
                w = np.ones(n_tk) / n_tk

                port_net = net_wide.values @ w
                port_gross = ret_wide.values @ w
                port_exp = pos_wide.abs().values @ w

                dates_idx = ret_wide.index

                # FDR + margin
                rf_daily = kr_series.reindex(dates_idx).ffill().bfill().values / 365.0
                cal_days = compute_cal_days(dates_idx)
                kr_ann = kr_series.reindex(dates_idx).ffill().bfill().values

                fdr = np.maximum(0.0, 1.0 - port_exp) * rf_daily * cal_days
                margin = np.maximum(0.0, port_exp - 1.0) * (kr_ann + 0.02) / 365.0

                total_ret = port_net + fdr - margin
                excess_ret = total_ret - rf_daily * cal_days

                metrics = compute_metrics(total_ret, excess_ret,
                                          port_gross, port_exp, cal_days)

                row = {
                    "strategy": sid,
                    "approach": app,
                    "comm_label": comm_label,
                    "comm": comm_val,
                }
                row.update(metrics)
                all_rows.append(row)

                # Cache daily returns
                cache_key = (sid, app, comm_label)
                daily_returns_cache[cache_key] = pd.DataFrame({
                    "date": dates_idx,
                    "trading_ret": port_gross,
                    "total_ret": total_ret,
                    "excess_ret": excess_ret,
                    "exposure": port_exp,
                }).set_index("date")

                # Save daily returns
                ret_path = BCD_RET_DIR / f"{sid}_{app}_{comm_label}.csv"
                daily_returns_cache[cache_key].to_csv(ret_path)

                if done % 16 == 0 or done == total_combos:
                    elapsed = time.time() - t0
                    log(f"  [{done}/{total_combos}] {sid}_{app}_{comm_label}: "
                        f"ShTr={metrics['sharpe_trading']:.3f}  "
                        f"ShEx={metrics['sharpe_excess']:.3f}  "
                        f"Sh9={metrics['sharpe_excess_9pct']:.3f}  "
                        f"Exp={metrics['exposure']:.1f}%  "
                        f"({elapsed:.0f}s)")

    results_df = pd.DataFrame(all_rows)

    # Mark best approach per (strategy, comm_label)
    results_df["best_approach"] = False
    for (sid, cl), grp in results_df.groupby(["strategy", "comm_label"]):
        idx_best = grp["sharpe_trading"].idxmax()
        if pd.notna(grp.loc[idx_best, "sharpe_trading"]):
            results_df.loc[idx_best, "best_approach"] = True

    out_path = MSP_DIR / "bcd_extended_perstrat.csv"
    results_df.to_csv(out_path, index=False)
    log(f"Saved bcd_extended_perstrat.csv: {len(results_df)} rows")

    elapsed = time.time() - t0
    log(f"PART 2 complete in {elapsed:.1f}s")

    return results_df, daily_returns_cache


# ══════════════════════════════════════════════════════════════════════
# PART 3: MSP aggregation for all 8 approaches
# ══════════════════════════════════════════════════════════════════════
def run_part3(perstrat_df, daily_returns_cache, kr_series):
    """MSP = EW across best per-strategy portfolios, for each approach × comm."""
    log("")
    log("=" * 70)
    log("PART 3: MSP Aggregation (8 approaches)")
    log("=" * 70)
    t0 = time.time()

    all_rows = []
    msp_returns_cache = {}  # key = (app, comm_label) -> DataFrame

    for app in ALL_APPROACHES:
        for comm_label in COMM_DAILY:
            # Collect daily returns for all strategies under this approach+comm
            strat_rets = {}
            for sid in STRATEGIES:
                key = (sid, app, comm_label)
                if key in daily_returns_cache:
                    strat_rets[sid] = daily_returns_cache[key]

            if len(strat_rets) < 2:
                log(f"  MSP_{app}_{comm_label}: skipped (only {len(strat_rets)} strategies)")
                continue

            # Find common dates
            common_dates = None
            for sid, df in strat_rets.items():
                idx = df.index
                common_dates = idx if common_dates is None else common_dates.intersection(idx)
            if common_dates is None or len(common_dates) < 30:
                continue
            common_dates = common_dates.sort_values()

            # Build matrices
            n_s = len(strat_rets)
            strat_list = sorted(strat_rets.keys())
            w = np.ones(n_s) / n_s  # EW across strategies

            total_mat = np.column_stack([
                strat_rets[sid].loc[common_dates, "total_ret"].values
                for sid in strat_list])
            trading_mat = np.column_stack([
                strat_rets[sid].loc[common_dates, "trading_ret"].values
                for sid in strat_list])
            exp_mat = np.column_stack([
                strat_rets[sid].loc[common_dates, "exposure"].values
                for sid in strat_list])

            msp_total = total_mat @ w
            msp_trading = trading_mat @ w
            msp_exp = exp_mat @ w

            # For MSP, total_ret already includes FDR/margin from per-strategy level
            # excess = total - rf_daily * cal_days
            rf_daily = kr_series.reindex(common_dates).ffill().bfill().values / 365.0
            cal_days = compute_cal_days(common_dates)
            msp_excess = msp_total - rf_daily * cal_days

            metrics = compute_metrics(msp_total, msp_excess,
                                      msp_trading, msp_exp, cal_days)

            row = {
                "approach": app,
                "n_strategies": n_s,
                "strategies": "+".join(strat_list),
                "comm_label": comm_label,
                "comm": COMM_DAILY[comm_label],
            }
            row.update(metrics)
            all_rows.append(row)

            # Cache
            msp_key = (app, comm_label)
            msp_returns_cache[msp_key] = pd.DataFrame({
                "trading_ret": msp_trading,
                "total_ret": msp_total,
                "excess_ret": msp_excess,
                "exposure": msp_exp,
            }, index=common_dates)

            # Save
            ret_path = BCD_RET_DIR / f"MSP_{app}_{comm_label}.csv"
            msp_returns_cache[msp_key].to_csv(ret_path)

            log(f"  MSP_{app}_{comm_label}: "
                f"ShTr={metrics['sharpe_trading']:.3f}  "
                f"ShEx={metrics['sharpe_excess']:.3f}  "
                f"Sh9={metrics['sharpe_excess_9pct']:.3f}  "
                f"Exp={metrics['exposure']:.1f}%")

    msp_df = pd.DataFrame(all_rows)

    # Mark best approach per comm_label
    msp_df["best_approach"] = False
    for cl, grp in msp_df.groupby("comm_label"):
        idx_best = grp["sharpe_trading"].idxmax()
        if pd.notna(grp.loc[idx_best, "sharpe_trading"]):
            msp_df.loc[idx_best, "best_approach"] = True

    out_path = MSP_DIR / "bcd_extended_msp.csv"
    msp_df.to_csv(out_path, index=False)
    log(f"Saved bcd_extended_msp.csv: {len(msp_df)} rows")

    elapsed = time.time() - t0
    log(f"PART 3 complete in {elapsed:.1f}s")

    return msp_df, msp_returns_cache


# ══════════════════════════════════════════════════════════════════════
# PART 4: Bootstrap tests — each overlay vs A
# ══════════════════════════════════════════════════════════════════════
def run_part4(daily_returns_cache, msp_returns_cache):
    """Bootstrap: compare each overlay approach (B/C/D/BC/BD/CD/BCD) vs A.
    Per-strategy: 7 overlays × 6 strategies × 2 comms = 84 tests.
    MSP: 7 overlays × 2 comms = 14 tests.
    Total: 98 tests.
    """
    log("")
    log("=" * 70)
    log("PART 4: Bootstrap Tests (overlays vs A)")
    log("=" * 70)
    t0 = time.time()

    overlay_approaches = ["B", "C", "D", "BC", "BD", "CD", "BCD"]
    boot_rows = []

    # 4a: Per-strategy tests
    n_tests = 0
    for sid in STRATEGIES:
        for comm_label in COMM_DAILY:
            key_a = (sid, "A", comm_label)
            if key_a not in daily_returns_cache:
                continue
            ret_a = daily_returns_cache[key_a]["excess_ret"].values

            for ov in overlay_approaches:
                key_ov = (sid, ov, comm_label)
                if key_ov not in daily_returns_cache:
                    continue
                ret_ov = daily_returns_cache[key_ov]["excess_ret"].values

                obs, p_val, ci_lo, ci_hi = bootstrap_sharpe_diff(ret_ov, ret_a)
                n_tests += 1

                boot_rows.append({
                    "level": "per_strategy",
                    "test": f"{sid}_{ov}>A",
                    "strategy": sid,
                    "comm_label": comm_label,
                    "delta_sharpe": round(obs, 4),
                    "p_value": round(p_val, 4),
                    "ci_lo": round(ci_lo, 4),
                    "ci_hi": round(ci_hi, 4),
                    "significant": p_val < 0.05,
                })

    log(f"  Per-strategy bootstrap: {n_tests} tests")

    # 4b: MSP tests
    n_msp_tests = 0
    for comm_label in COMM_DAILY:
        msp_key_a = ("A", comm_label)
        if msp_key_a not in msp_returns_cache:
            continue
        ret_a_msp = msp_returns_cache[msp_key_a]["excess_ret"].values

        for ov in overlay_approaches:
            msp_key_ov = (ov, comm_label)
            if msp_key_ov not in msp_returns_cache:
                continue
            ret_ov_msp = msp_returns_cache[msp_key_ov]["excess_ret"].values

            obs, p_val, ci_lo, ci_hi = bootstrap_sharpe_diff(ret_ov_msp, ret_a_msp)
            n_msp_tests += 1

            boot_rows.append({
                "level": "msp",
                "test": f"MSP_{ov}>A",
                "strategy": "MSP",
                "comm_label": comm_label,
                "delta_sharpe": round(obs, 4),
                "p_value": round(p_val, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "significant": p_val < 0.05,
            })

    log(f"  MSP bootstrap: {n_msp_tests} tests")

    boot_df = pd.DataFrame(boot_rows)
    out_path = BOOT_DIR / "bootstrap_bcd_extended.csv"
    boot_df.to_csv(out_path, index=False)
    log(f"Saved bootstrap_bcd_extended.csv: {len(boot_df)} rows")

    elapsed = time.time() - t0
    log(f"PART 4 complete in {elapsed:.1f}s")

    return boot_df


# ══════════════════════════════════════════════════════════════════════
# Summary printer
# ══════════════════════════════════════════════════════════════════════
def print_summary(perstrat_df, msp_df, boot_df):
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)

    # Per-strategy mean Sharpe by approach
    log("")
    log("--- Mean Trading Sharpe by approach (per-strategy, inst comm) ---")
    sub = perstrat_df[perstrat_df["comm_label"] == "inst"]
    for app in ALL_APPROACHES:
        app_sub = sub[sub["approach"] == app]
        if not app_sub.empty:
            mean_sh = app_sub["sharpe_trading"].mean()
            mean_ex = app_sub["sharpe_excess"].mean()
            log(f"  {app:>4s}: ShTr={mean_sh:+.3f}  ShEx={mean_ex:+.3f}  "
                f"n={len(app_sub)}")

    # MSP Sharpe
    log("")
    log("--- MSP Sharpe by approach ---")
    for _, r in msp_df.iterrows():
        log(f"  MSP-{r['approach']:>4s} ({r['comm_label']}): "
            f"ShTr={r['sharpe_trading']:+.3f}  "
            f"ShEx={r['sharpe_excess']:+.3f}  "
            f"Sh9={r['sharpe_excess_9pct']:+.3f}  "
            f"Exp={r['exposure']:.1f}%")

    # Bootstrap significance
    if not boot_df.empty:
        log("")
        log("--- Bootstrap: significant improvements over A ---")
        sig = boot_df[boot_df["significant"]]
        if sig.empty:
            log("  None significant at p<0.05")
        else:
            for _, r in sig.iterrows():
                log(f"  {r['test']:>15s} ({r['comm_label']}): "
                    f"dSh={r['delta_sharpe']:+.4f}  "
                    f"p={r['p_value']:.4f}  "
                    f"CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

        log("")
        log("--- Bootstrap: combo approaches vs single overlays (inst comm) ---")
        inst_boot = boot_df[boot_df["comm_label"] == "inst"]
        for ov in ["BC", "BD", "CD", "BCD"]:
            ov_rows = inst_boot[inst_boot["test"].str.contains(f"_{ov}>A")]
            if not ov_rows.empty:
                mean_delta = ov_rows["delta_sharpe"].mean()
                n_sig = ov_rows["significant"].sum()
                log(f"  {ov:>4s}: mean dSh={mean_delta:+.4f}, "
                    f"significant in {n_sig}/{len(ov_rows)} tests")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("STAGE 5d: Extended BCD Combos + Portfolio + Bootstrap")
    log("=" * 80)
    t_start = time.time()

    # Load data
    kr_series = load_key_rate()
    pos_data = load_positions()

    # PART 1: Build combo positions
    combo_dfs = build_combo_positions(pos_data)

    # PART 2: Per-strategy EW portfolio
    perstrat_df, daily_returns_cache = run_part2(pos_data, combo_dfs, kr_series)

    # PART 3: MSP aggregation
    msp_df, msp_returns_cache = run_part3(perstrat_df, daily_returns_cache, kr_series)

    # PART 4: Bootstrap
    boot_df = run_part4(daily_returns_cache, msp_returns_cache)

    # Summary
    print_summary(perstrat_df, msp_df, boot_df)

    elapsed = time.time() - t_start
    log("")
    log(f"STAGE 5d COMPLETE in {elapsed:.1f}s")
    log("=" * 80)


if __name__ == "__main__":
    main()
