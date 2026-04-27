#!/usr/bin/env python3
"""
stage5_msp.py — 3-level portfolio structure.

Level 1: Per-ticker descriptive stats (strategy × ticker × approach × comm).
Level 2: Per-strategy portfolio (EW across 17 tickers, one per strategy).
Level 3: MSP (EW across all 102 = 6 strategies × 17 tickers).

Daily positions only. Commissions: 0.0005 / 0.0006.

Saves:
  descriptive/per_ticker_stats.csv
  msp/msp_results.csv
  msp/msp_positions.parquet
  msp/perstrat_daily_returns/  (one CSV per variant)
"""
import os, sys, warnings, datetime
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
STRAT_DIR = BASE / "results/final/strategies"
MSP_DIR = STRAT_DIR / "msp"
MSP_DIR.mkdir(parents=True, exist_ok=True)
DESC_DIR = STRAT_DIR / "descriptive"
DESC_DIR.mkdir(parents=True, exist_ok=True)
PERSTRAT_RET_DIR = MSP_DIR / "perstrat_daily_returns"
PERSTRAT_RET_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = STRAT_DIR / "MASTER_PROGRESS_LOG.txt"

WF_DIR = STRAT_DIR / "walkforward"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")

BPY = 252
RF_MEAN_HIST = 0.09
TICKERS = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
           "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"]
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
APPROACHES = ["A", "B", "C", "D"]
OOS_YEARS = [2022, 2023, 2024, 2025]

COMM_DAILY = {"inst": 0.0005, "std": 0.0006}


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage5_msp] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════
# Load key rate
# ══════════════════════════════════════════════════════════════════════
def load_key_rate():
    if KR_PATH.exists():
        kr = pd.read_parquet(KR_PATH)
        kr["date"] = pd.to_datetime(kr["date"])
        return kr.set_index("date")["key_rate"].sort_index()
    log("WARNING: key rate file not found, using 0.16 flat")
    dates = pd.bdate_range("2019-01-01", "2026-06-01")
    return pd.Series(0.16, index=dates, name="key_rate")


# ══════════════════════════════════════════════════════════════════════
# Load positions (daily only)
# ══════════════════════════════════════════════════════════════════════
def load_positions():
    """Load daily WF positions for all approaches.
    Returns DataFrame: date, ticker, strategy, year, position, daily_gross_return, approach.
    """
    frames = []
    for app in APPROACHES:
        fp = WF_DIR / f"wf_positions_{app}.parquet"
        if fp.exists():
            tmp = pd.read_parquet(fp)
            tmp["approach"] = app
            tmp["date"] = pd.to_datetime(tmp["date"])
            frames.append(tmp)
            log(f"  Loaded {fp.name}: {len(tmp)} rows")
        else:
            log(f"  WARNING: {fp} not found")

    if not frames:
        raise FileNotFoundError("No position files found. Stage 3/4 must run first.")

    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# Level 1: Per-ticker descriptive stats
# ══════════════════════════════════════════════════════════════════════
def compute_per_ticker_stats(pos_df):
    """For each strategy × ticker × approach × comm, compute descriptive stats."""
    log("Level 1: Per-ticker descriptive stats")
    rows = []
    for app in APPROACHES:
        app_data = pos_df[pos_df["approach"] == app]
        if app_data.empty:
            continue
        for comm_label, comm_val in COMM_DAILY.items():
            for strat in STRATEGIES:
                strat_data = app_data[app_data["strategy"] == strat]
                if strat_data.empty:
                    continue
                for tick in TICKERS:
                    sub = strat_data[strat_data["ticker"] == tick].sort_values("date").reset_index(drop=True)
                    if len(sub) < 30:
                        continue

                    pos = sub["position"].values
                    gross = sub["daily_gross_return"].values  # = pos * log_return
                    dp = np.diff(pos, prepend=0.0)
                    net = gross - np.abs(dp) * comm_val
                    n = len(net)
                    n_years = max(n / BPY, 0.01)

                    # Trading Sharpe
                    s_net = np.std(net, ddof=1)
                    sharpe_trading = np.mean(net) * sqrt(BPY) / s_net if s_net > 1e-12 else 0.0

                    # Exposure
                    exposure = np.mean(np.abs(pos))

                    # Trades per year
                    entries = np.sum((np.abs(pos[1:]) > 0) & (np.abs(pos[:-1]) == 0))
                    trades_yr = entries / n_years

                    # Max drawdown
                    cum = np.cumsum(net)
                    peak = np.maximum.accumulate(cum)
                    dd = peak - cum
                    max_dd = np.max(dd) * 100 if len(dd) > 0 else 0.0

                    # Win rate
                    active = net[np.abs(pos) > 0]
                    win_rate = np.mean(active > 0) * 100 if len(active) > 0 else 0.0

                    # Ann return
                    cum_total = np.sum(net)
                    ann_ret = cum_total / n_years * 100

                    rows.append({
                        "strategy": strat,
                        "ticker": tick,
                        "approach": app,
                        "comm_label": comm_label,
                        "comm": comm_val,
                        "sharpe_trading": round(sharpe_trading, 3),
                        "ann_ret_pct": round(ann_ret, 2),
                        "exposure": round(exposure * 100, 1),
                        "trades_yr": round(trades_yr, 1),
                        "max_dd_pct": round(max_dd, 2),
                        "win_rate": round(win_rate, 1),
                        "n_days": n,
                    })

    df = pd.DataFrame(rows)
    out_path = DESC_DIR / "per_ticker_stats.csv"
    df.to_csv(out_path, index=False)
    log(f"  Saved per_ticker_stats.csv: {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════════════════
# Portfolio construction (shared for levels 2 and 3)
# ══════════════════════════════════════════════════════════════════════
def build_portfolio_returns(pos_df, kr_series, comm):
    """
    Build EW portfolio from position data.
    pos_df must have: date, ticker, position, daily_gross_return.
    Each (date, ticker) should have exactly one row (pre-aggregated if needed).
    Returns portfolio DataFrame indexed by date.
    """
    # Per-ticker net return
    ticker_frames = []
    for tick in pos_df["ticker"].unique():
        sub = pos_df[pos_df["ticker"] == tick].sort_values("date").reset_index(drop=True)
        pos = sub["position"].values
        gross = sub["daily_gross_return"].values  # already = pos * log_return
        dp = np.diff(pos, prepend=0.0)
        net = gross - np.abs(dp) * comm
        sub = sub.copy()
        sub["net_ret"] = net
        sub["abs_pos"] = np.abs(pos)
        sub["turnover"] = np.abs(dp)
        ticker_frames.append(sub[["date", "ticker", "net_ret", "abs_pos", "turnover"]])

    if not ticker_frames:
        return None

    all_tk = pd.concat(ticker_frames, ignore_index=True)

    # EW portfolio: mean across tickers per date
    port = all_tk.groupby("date").agg(
        portfolio_return=("net_ret", "mean"),
        portfolio_exposure=("abs_pos", "mean"),
        turnover=("turnover", "mean"),
    ).sort_index()

    # Align key rate
    rf_daily = kr_series.reindex(port.index).ffill().bfill() / 365.0

    # Calendar days between trading days
    idx = port.index
    cal_days = np.ones(len(idx))
    for i in range(1, len(idx)):
        cal_days[i] = (idx[i] - idx[i - 1]).days

    port["rf_daily"] = rf_daily.values
    port["cal_days"] = cal_days

    exp_arr = port["portfolio_exposure"].values

    # FDR: cash earns RF
    port["fdr"] = np.maximum(0.0, 1.0 - exp_arr) * port["rf_daily"].values * cal_days
    # Margin cost for leverage > 1
    kr_ann = kr_series.reindex(port.index).ffill().bfill().values
    port["margin_cost"] = np.maximum(0.0, exp_arr - 1.0) * (kr_ann + 0.02) / 365.0

    port["portfolio_total"] = port["portfolio_return"] + port["fdr"] - port["margin_cost"]
    port["excess"] = port["portfolio_total"] - port["rf_daily"] * port["cal_days"]

    return port


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(port):
    """Compute all performance metrics from portfolio DataFrame."""
    total = port["portfolio_total"].values
    excess = port["excess"].values
    exp_arr = port["portfolio_exposure"].values
    turn_arr = port["turnover"].values
    trading_ret = port["portfolio_return"].values
    cd = port["cal_days"].values
    n = len(total)

    if n < 30:
        return {k: np.nan for k in [
            "sharpe", "sharpe_trading", "sharpe_excess_9pct",
            "sortino", "calmar", "max_dd", "max_dd_dur",
            "ann_ret", "ann_vol", "win_rate", "profit_factor",
            "exposure", "turnover"]}

    # Annual return (simple sum-based for log returns)
    n_years = max(n / BPY, 0.01)
    cum = np.sum(total)
    ann_ret = cum / n_years

    # Annual vol
    ann_vol = np.std(total, ddof=1) * sqrt(BPY)

    # Excess Sharpe (real RF)
    s_total = np.std(total, ddof=1)
    sharpe = np.mean(excess) * sqrt(BPY) / s_total if s_total > 1e-12 else 0.0

    # Trading Sharpe (no RF, no FDR — just portfolio_return)
    s_trad = np.std(trading_ret, ddof=1)
    sharpe_trading = np.mean(trading_ret) * sqrt(BPY) / s_trad if s_trad > 1e-12 else 0.0

    # Counterfactual excess (RF = 9%)
    rf_cf = RF_MEAN_HIST / 365.0
    fdr_cf = np.maximum(0.0, 1.0 - exp_arr) * rf_cf * cd
    margin_cf = np.maximum(0.0, exp_arr - 1.0) * (RF_MEAN_HIST + 0.02) / 365.0
    total_cf = trading_ret + fdr_cf - margin_cf
    excess_cf = total_cf - rf_cf * cd
    s_cf = np.std(total_cf, ddof=1)
    sharpe_excess_9pct = np.mean(excess_cf) * sqrt(BPY) / s_cf if s_cf > 1e-12 else 0.0

    # Sortino
    downside = total[total < 0]
    ds_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-12
    sortino = np.mean(excess) * sqrt(BPY) / ds_std if ds_std > 1e-12 else 0.0

    # Max drawdown
    cum_arr = np.cumsum(total)
    peak = np.maximum.accumulate(cum_arr)
    dd = peak - cum_arr
    max_dd = np.max(dd) * 100  # percent

    # Max DD duration (trading days)
    in_dd = dd > 1e-10
    max_dd_dur = 0
    cur_dur = 0
    for i in range(n):
        if in_dd[i]:
            cur_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dur)
        else:
            cur_dur = 0

    # Calmar
    calmar = (ann_ret) / (max_dd / 100) if max_dd > 1e-6 else 0.0

    # Win rate
    win_rate = np.mean(total > 0) * 100

    # Profit factor
    gross_profit = np.sum(total[total > 0])
    gross_loss = np.abs(np.sum(total[total < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else np.inf

    # Exposure & turnover
    avg_exp = np.mean(exp_arr) * 100
    avg_turn = np.mean(turn_arr) * BPY  # annualized

    return {
        "sharpe": round(sharpe, 3),
        "sharpe_trading": round(sharpe_trading, 3),
        "sharpe_excess_9pct": round(sharpe_excess_9pct, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd": round(max_dd, 2),
        "max_dd_dur": int(max_dd_dur),
        "ann_ret": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3),
        "exposure": round(avg_exp, 1),
        "turnover": round(avg_turn, 1),
    }


def per_year_sharpe(port, years=OOS_YEARS):
    """Compute excess Sharpe per year."""
    res = {}
    for y in years:
        mask = port.index.year == y
        sub = port[mask]
        if len(sub) < 20:
            res[f"sharpe_{y}"] = np.nan
            continue
        tot = sub["portfolio_total"].values
        exc = sub["excess"].values
        s = np.std(tot, ddof=1)
        res[f"sharpe_{y}"] = round(np.mean(exc) * sqrt(BPY) / s, 3) if s > 1e-12 else 0.0
    yr_vals = [v for v in res.values() if not np.isnan(v)]
    res["sharpe_avg_yearly"] = round(np.nanmean(yr_vals), 3) if yr_vals else np.nan
    return res


# ══════════════════════════════════════════════════════════════════════
# Level 2: Per-strategy portfolio
# ══════════════════════════════════════════════════════════════════════
def build_per_strategy_portfolios(pos_df, kr_series):
    """Build EW portfolio per strategy (across 17 tickers)."""
    log("Level 2: Per-strategy portfolios")
    all_rows = []
    all_positions = []

    for app in APPROACHES:
        app_data = pos_df[pos_df["approach"] == app]
        if app_data.empty:
            continue
        for comm_label, comm_val in COMM_DAILY.items():
            for strat in STRATEGIES:
                strat_data = app_data[app_data["strategy"] == strat]
                if strat_data.empty:
                    continue

                variant = f"{strat}_{app}_{comm_label}"
                log(f"  Per-strategy portfolio: {variant}")

                port = build_portfolio_returns(strat_data, kr_series, comm_val)
                if port is None or len(port) < 30:
                    log(f"    Skipped {variant}: insufficient data")
                    continue

                metrics = compute_metrics(port)
                yr_sh = per_year_sharpe(port)

                row = {
                    "level": "per_strategy",
                    "strategy": strat,
                    "approach": app,
                    "comm_label": comm_label,
                    "comm": comm_val,
                }
                row.update(metrics)
                row.update(yr_sh)
                all_rows.append(row)

                # Save daily returns for bootstrap
                ret_df = port[["portfolio_return", "portfolio_total", "excess"]].copy()
                ret_df.columns = ["trading_ret", "total_ret", "excess_ret"]
                ret_path = PERSTRAT_RET_DIR / f"{variant}.csv"
                ret_df.to_csv(ret_path)

                # Collect for parquet
                port_out = port.copy()
                port_out["variant"] = variant
                port_out["level"] = "per_strategy"
                port_out["strategy"] = strat
                port_out["approach"] = app
                port_out["comm_label"] = comm_label
                all_positions.append(port_out)

                log(f"    Sh={metrics['sharpe']:.3f}  ShTr={metrics['sharpe_trading']:.3f}  "
                    f"Sh9={metrics['sharpe_excess_9pct']:.3f}  Exp={metrics['exposure']:.1f}%")

    log(f"  Per-strategy portfolios: {len(all_rows)} variants computed")
    return all_rows, all_positions


# ══════════════════════════════════════════════════════════════════════
# Level 3: MSP (EW across all 102 = 6 strategies × 17 tickers)
# ══════════════════════════════════════════════════════════════════════
def build_msp(pos_df, kr_series):
    """Build MSP: EW across all strategy×ticker pairs."""
    log("Level 3: MSP (all 102 pairs)")
    all_rows = []
    all_positions = []

    for app in APPROACHES:
        app_data = pos_df[pos_df["approach"] == app]
        if app_data.empty:
            continue
        for comm_label, comm_val in COMM_DAILY.items():
            variant = f"MSP_{app}_{comm_label}"
            log(f"  MSP: {variant}")

            # Aggregate: for each (date, ticker), average across strategies
            # so that we get 17 tickers each with averaged position/gross_ret
            grp = app_data.groupby(["date", "ticker"]).agg(
                position=("position", "mean"),
                daily_gross_return=("daily_gross_return", "mean"),
            ).reset_index()

            port = build_portfolio_returns(grp, kr_series, comm_val)
            if port is None or len(port) < 30:
                log(f"    Skipped {variant}: insufficient data")
                continue

            metrics = compute_metrics(port)
            yr_sh = per_year_sharpe(port)

            row = {
                "level": "msp",
                "strategy": "ALL",
                "approach": app,
                "comm_label": comm_label,
                "comm": comm_val,
            }
            row.update(metrics)
            row.update(yr_sh)
            all_rows.append(row)

            # Save daily returns for bootstrap
            ret_df = port[["portfolio_return", "portfolio_total", "excess"]].copy()
            ret_df.columns = ["trading_ret", "total_ret", "excess_ret"]
            ret_path = PERSTRAT_RET_DIR / f"{variant}.csv"
            ret_df.to_csv(ret_path)

            # Collect for parquet
            port_out = port.copy()
            port_out["variant"] = variant
            port_out["level"] = "msp"
            port_out["strategy"] = "ALL"
            port_out["approach"] = app
            port_out["comm_label"] = comm_label
            all_positions.append(port_out)

            log(f"    Sh={metrics['sharpe']:.3f}  ShTr={metrics['sharpe_trading']:.3f}  "
                f"Sh9={metrics['sharpe_excess_9pct']:.3f}  Exp={metrics['exposure']:.1f}%  "
                f"AvgYrSh={yr_sh['sharpe_avg_yearly']}")

    log(f"  MSP: {len(all_rows)} variants computed")
    return all_rows, all_positions


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("STAGE 5: 3-Level Portfolio (per-ticker / per-strategy / MSP)")
    log("=" * 80)

    kr_series = load_key_rate()
    pos_df = load_positions()

    # ── Level 1: Per-ticker descriptive stats ──────────────────────
    ticker_stats = compute_per_ticker_stats(pos_df)

    # ── Level 2: Per-strategy portfolios ───────────────────────────
    perstrat_rows, perstrat_positions = build_per_strategy_portfolios(pos_df, kr_series)

    # ── Level 3: MSP ──────────────────────────────────────────────
    msp_rows, msp_positions = build_msp(pos_df, kr_series)

    # ── Save combined results ─────────────────────────────────────
    all_rows = perstrat_rows + msp_rows
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(MSP_DIR / "msp_results.csv", index=False)
    log(f"Saved msp_results.csv: {len(results_df)} rows "
        f"({len(perstrat_rows)} per-strategy + {len(msp_rows)} MSP)")

    # Save all portfolio positions to parquet
    all_pos = perstrat_positions + msp_positions
    if all_pos:
        pos_all = pd.concat(all_pos, ignore_index=False)
        pos_all.to_parquet(MSP_DIR / "msp_positions.parquet", index=True)
        log(f"Saved msp_positions.parquet: {len(pos_all)} rows")
    else:
        log("WARNING: no positions to save")

    log("STAGE 5 COMPLETE")
    print(f"\n{'=' * 80}")
    print(results_df.to_string(index=False))
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
