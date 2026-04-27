#!/usr/bin/env python3
"""
stage7_bootstrap.py — 2-level bootstrap significance tests (10,000 replications).

42 tests total:
  Level 1 — Per-strategy portfolio (36 tests):
      6 strategies (S1-S6) × 3 comparisons (B>A, C>A, D>A) × 2 commissions (inst, std)
  Level 2 — MSP level (6 tests):
      3 comparisons (B>A, C>A, D>A) × 2 commissions (inst, std)

One-sided test: H1 = BCD daily returns > A daily returns.

Input:  msp/perstrat_daily_returns/{S1..S6,MSP}_{A,B,C,D}_{inst,std}.csv
Output: bootstrap/bootstrap_results.csv  (42 rows)
"""
import os, sys, warnings, datetime
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
STRAT_DIR = BASE / "results/final/strategies"
BOOT_DIR = STRAT_DIR / "bootstrap"
BOOT_DIR.mkdir(parents=True, exist_ok=True)
MSP_DIR = STRAT_DIR / "msp"
DAILY_RET_DIR = MSP_DIR / "perstrat_daily_returns"
LOG_PATH = STRAT_DIR / "MASTER_PROGRESS_LOG.txt"

N_BOOT = 10_000
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
BCD_APPROACHES = ["B", "C", "D"]
COMM_LABELS = ["inst", "std"]


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [stage7_boot] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════
# Bootstrap test
# ══════════════════════════════════════════════════════════════════════
def _sharpe(ret):
    """Annualized Sharpe of a daily return series."""
    s = np.std(ret, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(ret) * np.sqrt(252) / s


def bootstrap_test(ret_bcd, ret_a, n_boot=N_BOOT):
    """One-sided test: Sharpe(BCD) > Sharpe(A).
    Bootstrap the SHARPE DIFFERENCE (not raw return difference).
    This handles the case where BCD has lower mean but also lower vol."""
    n = len(ret_bcd)
    observed_delta = _sharpe(ret_bcd) - _sharpe(ret_a)
    np.random.seed(42)
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot_deltas[i] = _sharpe(ret_bcd[idx]) - _sharpe(ret_a[idx])
    p_value = np.mean(boot_deltas <= 0)
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    return {
        'delta_sharpe': round(float(observed_delta), 4),
        'delta_mean': float(np.mean(ret_bcd) - np.mean(ret_a)),
        'p_value': float(p_value),
        'ci_lo': round(float(ci_lo), 4),
        'ci_hi': round(float(ci_hi), 4),
        'significant': bool(p_value < 0.05),
    }


# ══════════════════════════════════════════════════════════════════════
# Load daily returns
# ══════════════════════════════════════════════════════════════════════
def load_daily_returns(prefix, approach, comm_label):
    """Load a CSV like S1_A_inst.csv or MSP_B_std.csv and return excess_ret Series."""
    fname = f"{prefix}_{approach}_{comm_label}.csv"
    fpath = DAILY_RET_DIR / fname
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df.set_index("date")["excess_ret"]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log(f"STAGE 7: 2-Level Bootstrap Significance Tests ({N_BOOT:,} replications, 42 tests)")
    log("=" * 80)

    all_results = []

    # ──────────────────────────────────────────────────────────────
    # Level 1: Per-strategy portfolio (36 tests)
    # ──────────────────────────────────────────────────────────────
    log("\n  Level 1: Per-strategy portfolio bootstrap (36 tests)")

    for strat in STRATEGIES:
        for bcd in BCD_APPROACHES:
            for cl in COMM_LABELS:
                test_name = f"{strat}_{bcd}>A_{cl}"
                ret_a = load_daily_returns(strat, "A", cl)
                ret_bcd = load_daily_returns(strat, bcd, cl)

                if ret_a is None or ret_bcd is None:
                    log(f"    {test_name}: SKIPPED (file missing)")
                    all_results.append({
                        "level": "per_strategy", "test": test_name,
                        "strategy": strat, "comm_label": cl, "comm": bcd,
                        "delta_mean": np.nan, "delta_sharpe": np.nan,
                        "p_value": 1.0, "ci_lo": np.nan, "ci_hi": np.nan,
                        "significant": False,
                    })
                    continue

                # Align dates
                common = ret_a.index.intersection(ret_bcd.index)
                if len(common) < 30:
                    log(f"    {test_name}: SKIPPED (only {len(common)} common dates)")
                    all_results.append({
                        "level": "per_strategy", "test": test_name,
                        "strategy": strat, "comm_label": cl, "comm": bcd,
                        "delta_mean": np.nan, "delta_sharpe": np.nan,
                        "p_value": 1.0, "ci_lo": np.nan, "ci_hi": np.nan,
                        "significant": False,
                    })
                    continue

                a_vals = ret_a.loc[common].values
                bcd_vals = ret_bcd.loc[common].values

                res = bootstrap_test(bcd_vals, a_vals)

                stars = "***" if res['p_value'] < 0.01 else (
                    "**" if res['p_value'] < 0.05 else (
                        "*" if res['p_value'] < 0.10 else ""))
                log(f"    {test_name}: delta={res['delta_mean']:+.6f} "
                    f"ann={res['delta_sharpe']:+.4f} p={res['p_value']:.4f} "
                    f"CI=[{res['ci_lo']:.6f},{res['ci_hi']:.6f}] {stars}")

                all_results.append({
                    "level": "per_strategy",
                    "test": test_name,
                    "strategy": strat,
                    "comm_label": cl,
                    "comm": bcd,
                    "delta_mean": round(res['delta_mean'], 8),
                    "delta_sharpe": round(res['delta_sharpe'], 6),
                    "p_value": round(res['p_value'], 6),
                    "ci_lo": round(res['ci_lo'], 8),
                    "ci_hi": round(res['ci_hi'], 8),
                    "significant": res['significant'],
                })

    # ──────────────────────────────────────────────────────────────
    # Level 2: MSP level (6 tests)
    # ──────────────────────────────────────────────────────────────
    log("\n  Level 2: MSP-level bootstrap (6 tests)")

    for bcd in BCD_APPROACHES:
        for cl in COMM_LABELS:
            test_name = f"MSP_{bcd}>A_{cl}"
            ret_a = load_daily_returns("MSP", "A", cl)
            ret_bcd = load_daily_returns("MSP", bcd, cl)

            if ret_a is None or ret_bcd is None:
                log(f"    {test_name}: SKIPPED (file missing)")
                all_results.append({
                    "level": "msp", "test": test_name,
                    "strategy": "MSP", "comm_label": cl, "comm": bcd,
                    "delta_mean": np.nan, "delta_sharpe": np.nan,
                    "p_value": 1.0, "ci_lo": np.nan, "ci_hi": np.nan,
                    "significant": False,
                })
                continue

            common = ret_a.index.intersection(ret_bcd.index)
            if len(common) < 30:
                log(f"    {test_name}: SKIPPED (only {len(common)} common dates)")
                all_results.append({
                    "level": "msp", "test": test_name,
                    "strategy": "MSP", "comm_label": cl, "comm": bcd,
                    "delta_mean": np.nan, "delta_sharpe": np.nan,
                    "p_value": 1.0, "ci_lo": np.nan, "ci_hi": np.nan,
                    "significant": False,
                })
                continue

            a_vals = ret_a.loc[common].values
            bcd_vals = ret_bcd.loc[common].values

            res = bootstrap_test(bcd_vals, a_vals)

            stars = "***" if res['p_value'] < 0.01 else (
                "**" if res['p_value'] < 0.05 else (
                    "*" if res['p_value'] < 0.10 else ""))
            log(f"    {test_name}: delta={res['delta_mean']:+.6f} "
                f"ann={res['delta_sharpe']:+.4f} p={res['p_value']:.4f} "
                f"CI=[{res['ci_lo']:.6f},{res['ci_hi']:.6f}] {stars}")

            all_results.append({
                "level": "msp",
                "test": test_name,
                "strategy": "MSP",
                "comm_label": cl,
                "comm": bcd,
                "delta_mean": round(res['delta_mean'], 8),
                "delta_sharpe": round(res['delta_sharpe'], 6),
                "p_value": round(res['p_value'], 6),
                "ci_lo": round(res['ci_lo'], 8),
                "ci_hi": round(res['ci_hi'], 8),
                "significant": res['significant'],
            })

    # ──────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────
    all_df = pd.DataFrame(all_results)
    cols = ["level", "test", "strategy", "comm_label", "comm",
            "delta_mean", "delta_sharpe", "p_value", "ci_lo", "ci_hi", "significant"]
    all_df = all_df[cols]
    all_df.to_csv(BOOT_DIR / "bootstrap_results.csv", index=False)
    log(f"\nSaved bootstrap_results.csv ({len(all_df)} rows)")

    # Summary
    n_sig = all_df["significant"].sum()
    n_total = len(all_df)
    perstrat_sig = all_df[all_df["level"] == "per_strategy"]["significant"].sum()
    msp_sig = all_df[all_df["level"] == "msp"]["significant"].sum()

    log(f"\nSUMMARY:")
    log(f"  Per-strategy: {perstrat_sig}/36 significant at p<0.05")
    log(f"  MSP level:    {msp_sig}/6 significant at p<0.05")
    log(f"  Total:        {n_sig}/{n_total} significant at p<0.05")
    log("STAGE 7 COMPLETE")

    print(f"\n{'='*100}")
    print(all_df.to_string(index=False))
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
