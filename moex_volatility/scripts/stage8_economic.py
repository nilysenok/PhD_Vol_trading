#!/usr/bin/env python3
"""
stage8_economic.py — Economic value of volatility-forecast-enhanced strategies.

Computes:
  1. Certainty Equivalent (CE) for gamma = {3, 5, 10}
  2. Delta CE: max annual fee for switching from A to B/C/D
  3. K_min: minimum capital = annual_commission_cost / annual_excess_return

Loads MSP returns from msp/msp_positions.parquet.

Saves: economic/economic_value.csv
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
ECON_DIR = STRAT_DIR / "economic"
ECON_DIR.mkdir(parents=True, exist_ok=True)
MSP_DIR = STRAT_DIR / "msp"
LOG_PATH = STRAT_DIR / "MASTER_PROGRESS_LOG.txt"

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")

BPY = 252
APPROACHES = ["A", "B", "C", "D"]
GAMMAS = [3, 5, 10]

COMM_DAILY  = {"inst": 0.0005, "std": 0.0006}
COMM_HOURLY = {"inst": 0.0004, "std": 0.0005}

# Real implementation costs (thousands RUB)
COSTS_YEAR1 = 16800   # Year 1: one-time 5200 + annual 11600
COSTS_YEAR2 = 13100   # Year 2+: annual recurring
KMIN_FEASIBLE = 1000   # K_min < 1000 mln RUB = feasible for institutional


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage8_econ] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════
# Immutable excess-Sharpe formula
# ══════════════════════════════════════════════════════════════════════
def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm):
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
    sharpe = np.mean(excess) * np.sqrt(252) / s
    return sharpe, {"exposure": round(np.mean(exposure) * 100, 1)}


# ══════════════════════════════════════════════════════════════════════
# Certainty Equivalent
# ══════════════════════════════════════════════════════════════════════
def certainty_equivalent(returns, gamma):
    """CE = E[r] * 252 - (gamma/2) * Var[r] * 252"""
    return np.mean(returns) * 252 - (gamma / 2) * np.var(returns) * 252


# ══════════════════════════════════════════════════════════════════════
# K_min: minimum capital
# ══════════════════════════════════════════════════════════════════════
def compute_kmin(ann_excess_return, comm_rate, avg_turnover_daily, n_tickers=17):
    """
    K_min = annual_commission_cost / annual_excess_return.
    annual_commission_cost per 1 ruble = comm_rate × avg_turnover_daily × 252
    If excess return <= 0, return inf.
    """
    annual_comm_per_ruble = comm_rate * avg_turnover_daily * BPY
    if ann_excess_return <= 1e-10:
        return np.inf
    return annual_comm_per_ruble / ann_excess_return


# ══════════════════════════════════════════════════════════════════════
# Load MSP positions
# ══════════════════════════════════════════════════════════════════════
def load_msp():
    fp = MSP_DIR / "msp_positions.parquet"
    if fp.exists():
        log(f"Loading MSP positions from {fp}")
        return pd.read_parquet(fp), False
    log("WARNING: MSP positions not found — creating DUMMY")
    return _create_dummy_msp(), True


def _create_dummy_msp():
    """Create dummy MSP data matching expected structure."""
    dates = pd.bdate_range("2022-01-03", "2025-12-31")
    frames = []
    for app in APPROACHES:
        for tf in ["daily", "hourly", "combined"]:
            n = len(dates)
            total = np.random.normal(0.0004, 0.01, size=n)
            excess = total - 0.16 / 365
            port = pd.DataFrame({
                "portfolio_return": total * 0.9,
                "portfolio_exposure": np.random.uniform(0.2, 0.6, size=n),
                "turnover": np.random.uniform(0.01, 0.05, size=n),
                "rf_daily": np.full(n, 0.16 / 365),
                "cal_days": np.ones(n),
                "fdr": np.random.uniform(0.0, 0.001, size=n),
                "margin_cost": np.zeros(n),
                "portfolio_total": total,
                "excess": excess,
                "variant": f"{app}_{tf}_inst",
                "approach": app,
                "tf_label": tf,
                "comm_label": "inst",
            }, index=dates)
            frames.append(port)
    return pd.concat(frames)


def load_key_rate():
    if KR_PATH.exists():
        kr = pd.read_parquet(KR_PATH)
        kr["date"] = pd.to_datetime(kr["date"])
        return kr.set_index("date")["key_rate"].sort_index()
    return pd.Series(0.16, index=pd.bdate_range("2019-01-01", "2026-06-01"), name="key_rate")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("STAGE 8: Economic Value Analysis")
    log("=" * 80)

    msp_df, is_dummy = load_msp()
    kr_series = load_key_rate()

    if is_dummy:
        log("*** RUNNING WITH DUMMY DATA — results are placeholders ***")

    # Get unique variants
    variants = msp_df["variant"].unique() if "variant" in msp_df.columns else []
    log(f"  Found {len(variants)} MSP variants: {list(variants)}")

    all_rows = []

    # ──────────────────────────────────────────────────────────────
    # 1. CE and Delta-CE for each timeframe
    # ──────────────────────────────────────────────────────────────
    for tf_label in ["daily", "hourly", "combined"]:
        log(f"\n  Timeframe: {tf_label}")

        # Extract returns for each approach
        approach_rets = {}
        approach_meta = {}
        for app in APPROACHES:
            vname = f"{app}_{tf_label}_inst"
            sub = msp_df[msp_df["variant"] == vname]
            if sub.empty:
                log(f"    {vname}: no data")
                continue
            approach_rets[app] = sub["portfolio_total"].values
            # Also compute metrics
            turnover = sub["turnover"].values if "turnover" in sub.columns else np.zeros(len(sub))
            exposure = sub["portfolio_exposure"].values if "portfolio_exposure" in sub.columns else np.ones(len(sub)) * 0.3
            excess = sub["excess"].values if "excess" in sub.columns else sub["portfolio_total"].values - 0.16 / 365
            approach_meta[app] = {
                "ann_excess": np.mean(excess) * BPY,
                "avg_turnover": np.mean(turnover),
                "avg_exposure": np.mean(exposure),
            }

        if "A" not in approach_rets:
            log(f"    No A data for {tf_label}, skipping")
            continue

        # CE for each gamma
        for gamma in GAMMAS:
            ce_vals = {}
            for app in APPROACHES:
                if app not in approach_rets:
                    continue
                ce_vals[app] = certainty_equivalent(approach_rets[app], gamma)

            ce_a = ce_vals.get("A", np.nan)
            log(f"    gamma={gamma}: CE_A={ce_a*100:.2f}%")

            for app in APPROACHES:
                if app not in ce_vals:
                    continue
                ce = ce_vals[app]
                delta_ce = ce - ce_a if app != "A" else 0.0

                row = {
                    "tf": tf_label,
                    "approach": app,
                    "gamma": gamma,
                    "ce_annual_pct": round(ce * 100, 4),
                    "delta_ce_pct": round(delta_ce * 100, 4),
                    "delta_ce_bps": round(delta_ce * 10000, 2),
                }
                all_rows.append(row)

                if app != "A":
                    log(f"      CE_{app}={ce*100:.2f}%  Delta={delta_ce*100:.4f}% ({delta_ce*10000:.2f} bps)")

        # K_min for each approach
        for app in APPROACHES:
            if app not in approach_meta:
                continue
            meta = approach_meta[app]

            # Determine commission rate
            if tf_label == "daily":
                comm_rate = COMM_DAILY["inst"]
            elif tf_label == "hourly":
                comm_rate = COMM_HOURLY["inst"]
            else:
                comm_rate = (COMM_DAILY["inst"] + COMM_HOURLY["inst"]) / 2

            kmin = compute_kmin(meta["ann_excess"], comm_rate, meta["avg_turnover"])

            # Add K_min rows (gamma=0 convention)
            kmin_row = {
                "tf": tf_label,
                "approach": app,
                "gamma": 0,
                "ce_annual_pct": round(meta["ann_excess"] * 100, 4),
                "delta_ce_pct": np.nan,
                "delta_ce_bps": np.nan,
                "kmin": round(kmin, 6) if not np.isinf(kmin) else np.inf,
                "ann_excess_pct": round(meta["ann_excess"] * 100, 4),
                "avg_turnover": round(meta["avg_turnover"], 6),
                "avg_exposure_pct": round(meta["avg_exposure"] * 100, 1),
                "comm_rate": comm_rate,
            }
            all_rows.append(kmin_row)
            log(f"    K_min({app}, {tf_label}): {kmin:.6f} (ann_excess={meta['ann_excess']*100:.2f}%, turnover={meta['avg_turnover']:.4f})")

    # ──────────────────────────────────────────────────────────────
    # 2. K_min with REAL implementation costs (Table 4.25)
    # ──────────────────────────────────────────────────────────────
    log(f"\n  K_min with real implementation costs:")
    log(f"  Year 1 total: {COSTS_YEAR1} thousand RUB")
    log(f"  Year 2+ annual: {COSTS_YEAR2} thousand RUB")

    kmin_rows = []
    for tf_label in ["daily", "hourly", "combined"]:
        for app in ["B", "C", "D"]:
            for gamma in GAMMAS:
                # Find delta CE for this combo
                ce_a_val = None; ce_bcd_val = None
                for row in all_rows:
                    if row.get("tf") == tf_label and row.get("gamma") == gamma:
                        if row.get("approach") == "A":
                            ce_a_val = row.get("ce_annual_pct", 0) / 100
                        elif row.get("approach") == app:
                            ce_bcd_val = row.get("ce_annual_pct", 0) / 100

                if ce_a_val is not None and ce_bcd_val is not None:
                    delta_ce = ce_bcd_val - ce_a_val  # as fraction
                else:
                    delta_ce = 0.0

                delta_ce_pct = delta_ce * 100

                # K_min = annual_cost / delta_CE
                if delta_ce > 1e-8:
                    kmin_y1_thousands = COSTS_YEAR1 / delta_ce
                    kmin_y2_thousands = COSTS_YEAR2 / delta_ce
                    kmin_y1_mln = kmin_y1_thousands / 1000
                    kmin_y2_mln = kmin_y2_thousands / 1000
                else:
                    kmin_y1_mln = float('inf')
                    kmin_y2_mln = float('inf')

                feasible = "Да" if kmin_y2_mln < KMIN_FEASIBLE else "Нет"

                kmin_rows.append({
                    "tf": tf_label,
                    "approach": app,
                    "gamma": gamma,
                    "delta_ce_pct": round(delta_ce_pct, 4),
                    "kmin_y1_mln_rub": round(kmin_y1_mln, 1) if not np.isinf(kmin_y1_mln) else np.inf,
                    "kmin_y2_mln_rub": round(kmin_y2_mln, 1) if not np.isinf(kmin_y2_mln) else np.inf,
                    "feasible": feasible,
                })

                log(f"    {app} γ={gamma} {tf_label}: ΔCE={delta_ce_pct:+.2f}%, "
                    f"K_min_y1={kmin_y1_mln:.1f}M, K_min_y2={kmin_y2_mln:.1f}M, {feasible}")

    kmin_df = pd.DataFrame(kmin_rows)
    kmin_df.to_csv(ECON_DIR / "kmin_real_costs.csv", index=False)
    log(f"  Saved kmin_real_costs.csv ({len(kmin_df)} rows)")

    # Also append to all_rows for unified output
    for kr in kmin_rows:
        all_rows.append({
            "tf": kr["tf"], "approach": kr["approach"], "gamma": kr["gamma"],
            "ce_annual_pct": np.nan, "delta_ce_pct": kr["delta_ce_pct"],
            "delta_ce_bps": round(kr["delta_ce_pct"] * 100, 2),
            "kmin_y1_mln": kr["kmin_y1_mln_rub"],
            "kmin_y2_mln": kr["kmin_y2_mln_rub"],
            "feasible": kr["feasible"],
        })

    # ──────────────────────────────────────────────────────────────
    # 3. Commission sensitivity: how much can comm increase before Sharpe < 0?
    # ──────────────────────────────────────────────────────────────
    log(f"\n  Commission sensitivity:")
    for tf_label in ["daily", "hourly"]:
        for app in APPROACHES:
            vname = f"{app}_{tf_label}_inst"
            sub = msp_df[msp_df["variant"] == vname]
            if sub.empty:
                continue
            base_excess = np.mean(sub["excess"].values) * BPY
            turnover = np.mean(sub["turnover"].values) if "turnover" in sub.columns else 0.02
            # At what commission does excess = 0?
            # excess ~ base_excess - delta_comm * turnover * 252
            if turnover > 1e-8:
                max_comm_increase = base_excess / (turnover * BPY)
            else:
                max_comm_increase = np.inf

            comm_row = {
                "tf": tf_label,
                "approach": app,
                "gamma": -1,  # convention: -1 = commission sensitivity
                "ce_annual_pct": round(base_excess * 100, 4),
                "delta_ce_pct": np.nan,
                "delta_ce_bps": np.nan,
                "max_comm_increase": round(max_comm_increase * 10000, 2),  # in bps
            }
            all_rows.append(comm_row)
            log(f"    {app}_{tf_label}: max comm increase = {max_comm_increase*10000:.2f} bps before Sharpe<0")

    # ──────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(ECON_DIR / "economic_value.csv", index=False)
    log(f"\nSaved economic_value.csv ({len(results_df)} rows)")

    log("STAGE 8 COMPLETE")

    print(f"\n{'='*80}")
    print("CERTAINTY EQUIVALENTS (gamma > 0):")
    ce_rows = results_df[results_df["gamma"].isin(GAMMAS)]
    if not ce_rows.empty:
        print(ce_rows[["tf", "approach", "gamma", "ce_annual_pct", "delta_ce_pct", "delta_ce_bps"]].to_string(index=False))

    print(f"\nK_MIN (gamma=0 rows):")
    kmin_rows = results_df[results_df["gamma"] == 0]
    if not kmin_rows.empty:
        cols = ["tf", "approach", "ann_excess_pct", "avg_turnover", "kmin"]
        valid_cols = [c for c in cols if c in kmin_rows.columns]
        print(kmin_rows[valid_cols].to_string(index=False))

    print(f"\nCOMMISSION SENSITIVITY (gamma=-1 rows):")
    comm_rows = results_df[results_df["gamma"] == -1]
    if not comm_rows.empty:
        cols = ["tf", "approach", "ce_annual_pct", "max_comm_increase"]
        valid_cols = [c for c in cols if c in comm_rows.columns]
        print(comm_rows[valid_cols].to_string(index=False))
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
