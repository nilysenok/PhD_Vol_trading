#!/usr/bin/env python3
"""
commission_rerun.py — Run full walk-forward pipeline with commission variants.

Reruns: strategies_walkforward.py → s5_rerun.py → s5s6_rerun.py
for each COMMISSION value (0.0005 and 0.001).

Commission is set via WF_COMMISSION environment variable.
Results saved to commission-specific CSV files.
Original results (COMMISSION=0) preserved untouched.
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v3"
OUT_DATA = OUT_DIR / "data"
OUT_TABLES = OUT_DIR / "tables"

MAIN_CSV = OUT_DATA / "wf_v3_all_results.csv"
POSITIONS_PARQUET = OUT_DIR / "daily_positions.parquet"
TRADE_LOG = OUT_DIR / "trade_log.parquet"

COMMISSION_VALUES = [0.0005, 0.001]

BCD_TEST_YEARS = [2022, 2023, 2024, 2025]
A_TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def comm_label(c):
    """Human-readable label: 0.0005 -> '5bps', 0.001 -> '10bps'."""
    return f"{int(c * 10000)}bps"


def comm_suffix(c):
    """File suffix: 0.0005 -> '_comm5', 0.001 -> '_comm10'."""
    return f"_comm{int(c * 10000)}"


def run_script(script_name, commission, timeout_min=120):
    """Run a pipeline script with WF_COMMISSION env var."""
    script_path = SCRIPTS / script_name
    env = os.environ.copy()
    env["WF_COMMISSION"] = str(commission)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n{'─' * 60}", flush=True)
    print(f"  Running {script_name} with COMMISSION={commission} ({comm_label(commission)})", flush=True)
    print(f"{'─' * 60}", flush=True)

    # Save full log to file
    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script_name.replace('.py', '')}{comm_suffix(commission)}.log"

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            stripped = line.strip()
            # Print key progress lines to console
            if any(k in stripped for k in [
                "DONE in", "MeanSharpe", "Updated", "replaced",
                "Commission:", "Walk-Forward", "RERUN", "BEFORE", "AFTER",
                "delta=", "ΔSharpe", "tickers", "done (",
                "Warming", "Processing", "Loading", "═",
            ]):
                print(f"    {stripped}", flush=True)
        proc.wait()

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"  Full log: {log_path}", flush=True)

    if proc.returncode != 0:
        print(f"  ERROR (return code {proc.returncode}):", flush=True)
        # Print last 30 lines of log
        with open(log_path) as f:
            lines = f.readlines()
        for line in lines[-30:]:
            print(f"    {line.rstrip()}", flush=True)
        return False

    return True


def backup_and_run(commission):
    """Backup originals, run all 3 pipelines, save commission-specific results."""
    suffix = comm_suffix(commission)
    label = comm_label(commission)

    print(f"\n{'═' * 70}")
    print(f"  COMMISSION = {commission} ({label})")
    print(f"{'═' * 70}")

    # Backup original files
    backup_csv = MAIN_CSV.with_suffix(".csv.bak")
    backup_pos = POSITIONS_PARQUET.with_suffix(".parquet.bak")
    backup_trade = TRADE_LOG.with_suffix(".parquet.bak")

    if MAIN_CSV.exists():
        shutil.copy2(MAIN_CSV, backup_csv)
    if POSITIONS_PARQUET.exists():
        shutil.copy2(POSITIONS_PARQUET, backup_pos)
    if TRADE_LOG.exists():
        shutil.copy2(TRADE_LOG, backup_trade)

    try:
        # 1. Main pipeline (all 6 strategies)
        ok = run_script("strategies_walkforward.py", commission, timeout_min=120)
        if not ok:
            print(f"  FAILED: main pipeline with {label}")
            return False

        # 2. S5 rerun (improved pivot logic)
        ok = run_script("s5_rerun.py", commission, timeout_min=30)
        if not ok:
            print(f"  FAILED: s5_rerun with {label}")
            return False

        # 3. S5h + S6d + S6h rerun
        ok = run_script("s5s6_rerun.py", commission, timeout_min=30)
        if not ok:
            print(f"  FAILED: s5s6_rerun with {label}")
            return False

        # Save commission-specific results
        dest_csv = OUT_DATA / f"wf_v3_all_results{suffix}.csv"
        shutil.copy2(MAIN_CSV, dest_csv)
        print(f"\n  Results saved → {dest_csv.name}")

        # Save commission-specific positions
        if POSITIONS_PARQUET.exists():
            dest_pos = OUT_DIR / f"daily_positions{suffix}.parquet"
            shutil.copy2(POSITIONS_PARQUET, dest_pos)

        return True

    finally:
        # Restore originals
        if backup_csv.exists():
            shutil.move(str(backup_csv), str(MAIN_CSV))
        if backup_pos.exists():
            shutil.move(str(backup_pos), str(POSITIONS_PARQUET))
        if backup_trade.exists():
            shutil.move(str(backup_trade), str(TRADE_LOG))


def generate_comparison():
    """Load all commission CSVs and generate comparison report."""
    print(f"\n{'═' * 70}")
    print("  COMMISSION COMPARISON REPORT")
    print(f"{'═' * 70}")

    # Load all CSVs
    dfs = {}
    # Original (comm=0)
    if MAIN_CSV.exists():
        dfs["0bps"] = pd.read_csv(MAIN_CSV)
    for c in COMMISSION_VALUES:
        path = OUT_DATA / f"wf_v3_all_results{comm_suffix(c)}.csv"
        if path.exists():
            dfs[comm_label(c)] = pd.read_csv(path)

    if len(dfs) < 2:
        print("  Not enough results to compare!")
        return

    strategies = sorted(dfs["0bps"]["strategy"].unique())
    timeframes = ["daily", "hourly"]
    approaches = ["A", "B", "C", "D"]

    # ── 1. Summary table: Mean Sharpe per strategy×TF×approach ──
    print(f"\n{'─' * 70}")
    print("  1. MEAN SHARPE (across tickers, BCD years 2022-2025)")
    print(f"{'─' * 70}")

    comm_labels = sorted(dfs.keys(), key=lambda x: int(x.replace("bps", "")))
    header = f"  {'Strategy':<18s} {'TF':>6s} {'Appr':>4s}"
    for cl in comm_labels:
        header += f"  {cl:>8s}"
    header += f"  {'Δ(5bps)':>8s}  {'Δ(10bps)':>9s}"
    print(header)
    print("  " + "─" * len(header))

    summary_rows = []
    for strat in strategies:
        for tf in timeframes:
            for appr in approaches:
                vals = {}
                for cl, df in dfs.items():
                    sub = df[(df["strategy"] == strat) &
                             (df["timeframe"] == tf) &
                             (df["approach"] == appr) &
                             (df["year"].isin(BCD_TEST_YEARS))]
                    vals[cl] = sub["sharpe"].mean() if len(sub) > 0 else np.nan

                line = f"  {strat:<18s} {tf:>6s} {appr:>4s}"
                for cl in comm_labels:
                    v = vals.get(cl, np.nan)
                    line += f"  {v:8.4f}" if not np.isnan(v) else f"  {'N/A':>8s}"

                # Deltas
                base = vals.get("0bps", np.nan)
                for delta_cl in ["5bps", "10bps"]:
                    v = vals.get(delta_cl, np.nan)
                    if not np.isnan(base) and not np.isnan(v):
                        d = v - base
                        line += f"  {d:+8.4f}"
                    else:
                        line += f"  {'N/A':>8s}"

                print(line)
                summary_rows.append({
                    "strategy": strat, "timeframe": tf, "approach": appr,
                    **{f"sharpe_{cl}": vals.get(cl, np.nan) for cl in comm_labels}
                })

    # ── 2. Aggregated by approach (all strategies pooled) ──
    print(f"\n{'─' * 70}")
    print("  2. AGGREGATE (mean across all strategies, BCD years)")
    print(f"{'─' * 70}")
    header2 = f"  {'TF':>6s} {'Appr':>4s}"
    for cl in comm_labels:
        header2 += f"  {cl:>8s}"
    header2 += f"  {'Δ(5bps)':>8s}  {'Δ(10bps)':>9s}"
    print(header2)
    print("  " + "─" * len(header2))

    for tf in timeframes:
        for appr in approaches:
            vals = {}
            for cl, df in dfs.items():
                sub = df[(df["timeframe"] == tf) &
                         (df["approach"] == appr) &
                         (df["year"].isin(BCD_TEST_YEARS))]
                vals[cl] = sub["sharpe"].mean() if len(sub) > 0 else np.nan

            line = f"  {tf:>6s} {appr:>4s}"
            for cl in comm_labels:
                v = vals.get(cl, np.nan)
                line += f"  {v:8.4f}" if not np.isnan(v) else f"  {'N/A':>8s}"
            base = vals.get("0bps", np.nan)
            for delta_cl in ["5bps", "10bps"]:
                v = vals.get(delta_cl, np.nan)
                if not np.isnan(base) and not np.isnan(v):
                    line += f"  {v - base:+8.4f}"
                else:
                    line += f"  {'N/A':>8s}"
            print(line)

    # ── 3. Impact on strategy rankings ──
    print(f"\n{'─' * 70}")
    print("  3. STRATEGY RANKING BY MEAN SHARPE (Approach A, BCD years)")
    print(f"{'─' * 70}")
    for tf in timeframes:
        print(f"\n  {tf}:")
        for cl in comm_labels:
            df = dfs[cl]
            sub = df[(df["timeframe"] == tf) &
                     (df["approach"] == "A") &
                     (df["year"].isin(BCD_TEST_YEARS))]
            ranking = sub.groupby("strategy")["sharpe"].mean().sort_values(ascending=False)
            rank_str = " > ".join(f"{s.split('_')[0]}({v:.3f})" for s, v in ranking.items())
            print(f"    {cl:>6s}: {rank_str}")

    # ── 4. Strategies that flip sign ──
    print(f"\n{'─' * 70}")
    print("  4. STRATEGIES THAT FLIP SIGN (gross → negative with commission)")
    print(f"{'─' * 70}")
    flips = []
    for strat in strategies:
        for tf in timeframes:
            for appr in approaches:
                gross_sub = dfs["0bps"]
                gross_sh = gross_sub[
                    (gross_sub["strategy"] == strat) &
                    (gross_sub["timeframe"] == tf) &
                    (gross_sub["approach"] == appr) &
                    (gross_sub["year"].isin(BCD_TEST_YEARS))
                ]["sharpe"].mean()

                for cl in comm_labels[1:]:  # skip 0bps
                    if cl not in dfs:
                        continue
                    net_sub = dfs[cl]
                    net_sh = net_sub[
                        (net_sub["strategy"] == strat) &
                        (net_sub["timeframe"] == tf) &
                        (net_sub["approach"] == appr) &
                        (net_sub["year"].isin(BCD_TEST_YEARS))
                    ]["sharpe"].mean()
                    if gross_sh > 0 and net_sh < 0:
                        flips.append(f"  {strat} {tf} {appr}: {gross_sh:.4f} → {net_sh:.4f} ({cl})")

    if flips:
        for f in flips:
            print(f)
    else:
        print("  None — all strategies remain positive with commission.")

    # ── 5. Turnover analysis ──
    print(f"\n{'─' * 70}")
    print("  5. AVERAGE N_TRADES (proxy for turnover)")
    print(f"{'─' * 70}")
    header5 = f"  {'Strategy':<18s} {'TF':>6s} {'Appr':>4s}"
    for cl in comm_labels:
        header5 += f"  {cl:>8s}"
    print(header5)
    print("  " + "─" * len(header5))

    for strat in strategies:
        for tf in timeframes:
            for appr in ["A"]:  # Only show A for brevity
                line = f"  {strat:<18s} {tf:>6s} {appr:>4s}"
                for cl in comm_labels:
                    df = dfs[cl]
                    sub = df[(df["strategy"] == strat) &
                             (df["timeframe"] == tf) &
                             (df["approach"] == appr) &
                             (df["year"].isin(BCD_TEST_YEARS))]
                    n_tr = sub["n_trades"].mean() if len(sub) > 0 else np.nan
                    line += f"  {n_tr:8.1f}" if not np.isnan(n_tr) else f"  {'N/A':>8s}"
                print(line)

    # ── 6. Save comparison CSV ──
    all_rows = []
    for cl, df in dfs.items():
        dfcopy = df.copy()
        dfcopy["commission"] = cl
        all_rows.append(dfcopy)
    comparison_df = pd.concat(all_rows, ignore_index=True)
    comp_path = OUT_DATA / "wf_v3_commission_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"\n  Full comparison saved → {comp_path.name}")

    # ── 7. Summary stats ──
    print(f"\n{'─' * 70}")
    print("  SUMMARY")
    print(f"{'─' * 70}")
    for cl in comm_labels:
        df = dfs[cl]
        sub = df[(df["approach"] == "A") & (df["year"].isin(BCD_TEST_YEARS))]
        mean_sh = sub["sharpe"].mean()
        pos_pct = (sub.groupby(["strategy", "timeframe"])["sharpe"]
                   .mean().gt(0).mean() * 100)
        print(f"  {cl:>6s}: Mean Sharpe(A) = {mean_sh:.4f}, "
              f"strategies with Sharpe>0 = {pos_pct:.0f}%")


def main():
    t0 = time.time()
    print("═" * 70)
    print("COMMISSION COMPARISON — Full Walk-Forward Rerun")
    print(f"Commission values: {', '.join(comm_label(c) for c in COMMISSION_VALUES)}")
    print(f"Scripts: strategies_walkforward.py → s5_rerun.py → s5s6_rerun.py")
    print("═" * 70)

    # Verify original results exist
    if not MAIN_CSV.exists():
        print(f"ERROR: Original results not found at {MAIN_CSV}")
        print("Run strategies_walkforward.py first with COMMISSION=0")
        sys.exit(1)

    # Check if commission results already exist
    for c in COMMISSION_VALUES:
        dest = OUT_DATA / f"wf_v3_all_results{comm_suffix(c)}.csv"
        if dest.exists():
            print(f"  Found existing: {dest.name}")

    # Run for each commission value
    for c in COMMISSION_VALUES:
        dest = OUT_DATA / f"wf_v3_all_results{comm_suffix(c)}.csv"
        if dest.exists():
            print(f"\n  Skipping {comm_label(c)} — results already exist at {dest.name}")
            print(f"  Delete the file to force rerun.")
            continue

        ok = backup_and_run(c)
        if not ok:
            print(f"\n  FAILED for {comm_label(c)}, skipping comparison")
            continue

    # Generate comparison
    generate_comparison()

    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print(f"TOTAL TIME: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
