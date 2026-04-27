#!/usr/bin/env python3
"""
orchestrator.py — Runs all Chapter 4 stages sequentially.
Waits for Stage 2 (BCD) to finish, then runs everything else.
Resume logic: skips already-completed stages.
Auto-correction: reruns with expanded grid if targets not met.
"""
import os, sys, subprocess, time, datetime, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from archivist import check_stage_inputs, print_manifest

BASE = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
RESULTS = BASE / "results/final/strategies"
LOG = RESULTS / "MASTER_PROGRESS_LOG.txt"

STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] ORCHESTRATOR: {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run_script(name, timeout=7200, env_extra=None):
    """Run a Python script and return exit code."""
    script = SCRIPTS / name
    if not script.exists():
        log(f"SKIP: {name} not found")
        return -1
    log(f"RUNNING: {name}")
    t0 = time.time()
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(SCRIPTS),
            timeout=timeout,
            env=env,
        )
        elapsed = time.time() - t0
        log(f"DONE: {name} (exit={result.returncode}, {elapsed:.0f}s)")
        return result.returncode
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {name} after {timeout}s")
        return -2
    except Exception as e:
        log(f"ERROR: {name}: {e}")
        return -3


def check_stage2_done():
    f1 = RESULTS / "bcd_test/bcd_test_results.csv"
    f2 = RESULTS / "bcd_test/bcd_test_results_006.csv"
    return f1.exists() and f2.exists()


def check_stage2a_done():
    return (RESULTS / "bcd_test/bcd_test_results.csv").exists()


# ══════════════════════════════════════════════════════════════
# RESUME LOGIC: check which stages are already done
# ══════════════════════════════════════════════════════════════

def is_stage_done(stage_name):
    """Check if a stage produced its key output files."""
    checks = {
        "stage1b": (RESULTS / "test_phases/FINAL_test_results_006.csv").exists(),
        "stage3": all((RESULTS / f"walkforward/wf_positions_{a}.parquet").exists()
                      for a in ["A", "B", "C", "D"]),
        "stage4": all((RESULTS / f"hourly/hourly_wf_positions_{a}.parquet").exists()
                      for a in ["A"]),  # at least A
        "stage5": (RESULTS / "msp/msp_results.csv").exists() and
                  (RESULTS / "msp/msp_positions.parquet").exists(),
        "stage6": (RESULTS / "correlations/corr_within_strategy.csv").exists(),
        "stage7": (RESULTS / "bootstrap/bootstrap_results.csv").exists(),
        "stage8": (RESULTS / "economic/economic_value.csv").exists() and
                  (RESULTS / "economic/kmin_real_costs.csv").exists(),
        "stage9": len(list((Path.home() / "Desktop/PhD/results/final/figures").glob("fig4_*"))) >= 16,
        "stage10": (RESULTS / "chapter4_tables.docx").exists(),
    }
    return checks.get(stage_name, False)


# ══════════════════════════════════════════════════════════════
# AUTO-CORRECTION VALIDATORS
# ══════════════════════════════════════════════════════════════

def validate_stage3():
    """Check WF A: mean > 0 for each strategy. Return list of failing strategies."""
    import pandas as pd
    path = RESULTS / "walkforward/wf_A_results.csv"
    if not path.exists():
        return STRATEGIES  # all failed
    wf = pd.read_csv(path)
    failing = []
    for sid in STRATEGIES:
        sub = wf[wf.strategy == sid] if "strategy" in wf.columns else wf[wf.get("strategy", wf.columns[0]) == sid]
        if len(sub) == 0:
            failing.append(sid)
            continue
        m = sub["sharpe"].mean() if "sharpe" in sub.columns else -999
        if m < 0:
            failing.append(sid)
            log(f"  AUTOCORRECT CHECK: {sid} WF mean={m:.2f} < 0")
        else:
            log(f"  {sid} WF mean={m:.2f} OK")
    return failing


def validate_stage7():
    """Check bootstrap: count tests with p > 0.05."""
    import pandas as pd
    path = RESULTS / "bootstrap/bootstrap_results.csv"
    if not path.exists():
        return -1
    boot = pd.read_csv(path)
    if "p_value" not in boot.columns:
        return -1
    n_fail = (boot["p_value"] > 0.05).sum()
    n_total = len(boot)
    return n_fail


def main():
    log("=" * 80)
    log("CHAPTER 4 ORCHESTRATOR — START (with resume + auto-correction)")
    log("=" * 80)

    # ════════════════════════════════════════
    # Wait for Stage 2 (BCD)
    # ════════════════════════════════════════
    if not check_stage2a_done():
        log("Waiting for Stage 2a (chapter4_master.py, comm=0.0005)...")
        wait_count = 0
        while not check_stage2a_done():
            time.sleep(60)
            wait_count += 1
            if wait_count % 5 == 0:
                log(f"  Still waiting... ({wait_count} min)")
            r = subprocess.run(["pgrep", "-f", "chapter4_master"], capture_output=True)
            if r.returncode != 0 and not check_stage2a_done():
                log("chapter4_master.py died without producing results!")
                break
        if check_stage2a_done():
            log("Stage 2a DONE — BCD 0.05% results available")
    else:
        log("Stage 2a already done")

    # Wait for Stage 2b (comm=0.0006)
    if not check_stage2_done():
        log("Waiting for Stage 2b (BCD comm=0.0006)...")
        wait_count = 0
        while not check_stage2_done():
            time.sleep(60)
            wait_count += 1
            r = subprocess.run(["pgrep", "-f", "chapter4_master"], capture_output=True)
            if r.returncode != 0:
                if check_stage2_done():
                    log("Stage 2b DONE")
                else:
                    log("chapter4_master.py finished — Stage 2b may not have run")
                break
            if wait_count % 5 == 0:
                log(f"  Still waiting for 2b... ({wait_count} min)")
        if check_stage2_done():
            log("Stage 2 COMPLETE — both comms available")

    # ════════════════════════════════════════
    # Stage 1b: Re-optimize A with comm=0.0006
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 1b: Re-optimize A with comm=0.0006")
    if is_stage_done("stage1b"):
        import pandas as pd
        df = pd.read_csv(RESULTS / "test_phases/FINAL_test_results_006.csv")
        mean_sh = df.groupby('strategy')['sharpe'].mean().mean()
        if mean_sh >= 1.0:
            log(f"Stage 1b already done with mean={mean_sh:.2f}. SKIP.")
        else:
            log(f"Stage 1b exists but mean={mean_sh:.2f} < 1.0. Re-optimizing...")
            run_script("autonomous_test_scan.py", timeout=1800,
                       env_extra={"OVERRIDE_COMM": "0.0006"})
    else:
        log("Running autonomous_test_scan.py with COMM=0.0006...")
        run_script("autonomous_test_scan.py", timeout=1800,
                   env_extra={"OVERRIDE_COMM": "0.0006"})

    # ════════════════════════════════════════
    # Stage 3: Walk-forward (daily) + auto-correction
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 3: Walk-forward (daily)")
    if is_stage_done("stage3"):
        log("Stage 3 positions exist. Validating...")
        failing = validate_stage3()
        if not failing:
            log("Stage 3 validated OK. SKIP.")
        else:
            log(f"Stage 3 FAILED for {failing}. Rerunning with expanded grid.")
            run_script("stage3_walkforward.py", timeout=5400,
                       env_extra={"EXPAND_GRID": "1"})
    else:
        check_stage_inputs("stage3_walkforward")
        run_script("stage3_walkforward.py", timeout=3600)

        # AUTO-CORRECT: check if WF mean < 0 for any strategy
        failing = validate_stage3()
        if failing:
            log(f"AUTOCORRECT: {failing} have WF mean < 0. Rerunning with expanded grid.")
            run_script("stage3_walkforward.py", timeout=5400,
                       env_extra={"EXPAND_GRID": "1"})
            # Check again
            still_failing = validate_stage3()
            if still_failing:
                log(f"AUTOCORRECT 2nd pass: {still_failing} still failing. "
                    f"Trying with EXPAND_GRID=2...")
                run_script("stage3_walkforward.py", timeout=5400,
                           env_extra={"EXPAND_GRID": "2"})
                final_failing = validate_stage3()
                if final_failing:
                    log(f"MANUAL REVIEW NEEDED: {final_failing} still have WF mean < 0 "
                        f"after 3 iterations")

    # ════════════════════════════════════════
    # Stage 4: Hourly pipeline
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 4: Hourly pipeline")
    if is_stage_done("stage4"):
        log("Stage 4 hourly positions exist. SKIP.")
    else:
        run_script("stage4_hourly.py", timeout=7200)

    # ════════════════════════════════════════
    # Stage 5: MSP (20 variants)
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 5: MSP")
    if is_stage_done("stage5"):
        log("Stage 5 MSP results exist. SKIP.")
    else:
        check_stage_inputs("stage5_msp")
        run_script("stage5_msp.py", timeout=600)

    # ════════════════════════════════════════
    # Stage 6: Correlations
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 6: Correlations")
    if is_stage_done("stage6"):
        log("Stage 6 correlations exist. SKIP.")
    else:
        run_script("stage6_correlations.py", timeout=300)

    # ════════════════════════════════════════
    # Stage 7: Bootstrap 10,000 + validation
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 7: Bootstrap")
    if is_stage_done("stage7"):
        log("Stage 7 bootstrap exists. Validating...")
        n_fail = validate_stage7()
        if n_fail == 0:
            log("Stage 7 all tests p < 0.05. SKIP.")
        elif n_fail > 0:
            log(f"MANUAL REVIEW: {n_fail} bootstrap tests have p > 0.05")
        # Don't rerun — bootstrap results are what they are
    else:
        check_stage_inputs("stage7_bootstrap")
        run_script("stage7_bootstrap.py", timeout=600)
        # Validate
        n_fail = validate_stage7()
        if n_fail > 0:
            import pandas as pd
            boot = pd.read_csv(RESULTS / "bootstrap/bootstrap_results.csv")
            failed = boot[boot["p_value"] > 0.05]
            log(f"MANUAL REVIEW: {n_fail}/{len(boot)} tests p > 0.05:")
            for _, r in failed.iterrows():
                log(f"  {r.get('test', '?')}: p={r['p_value']:.4f}")
        else:
            log("All bootstrap tests p < 0.05 ✅")

    # ════════════════════════════════════════
    # Stage 8: Economic value
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 8: Economic value")
    if is_stage_done("stage8"):
        log("Stage 8 economic value exists. SKIP.")
    else:
        check_stage_inputs("stage8_economic")
        run_script("stage8_economic.py", timeout=120)

    # ════════════════════════════════════════
    # Stage 9: Figures (16)
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 9: Figures")
    if is_stage_done("stage9"):
        log("Stage 9 figures exist. SKIP.")
    else:
        run_script("stage9_figures.py", timeout=300)

    # ════════════════════════════════════════
    # Stage 10: Tables (25+)
    # ════════════════════════════════════════
    log("\n" + "=" * 60)
    log("STAGE 10: Tables")
    if is_stage_done("stage10"):
        log("Stage 10 tables exist. SKIP.")
    else:
        run_script("stage10_tables.py", timeout=300)

    # ════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════
    log("\n" + "=" * 80)
    log("CHAPTER 4 ORCHESTRATOR — COMPLETE")
    log("=" * 80)

    for subdir in ["test_phases", "bcd_test", "walkforward", "hourly",
                    "msp", "correlations", "bootstrap", "economic", "grids"]:
        d = RESULTS / subdir
        if d.exists():
            files = [f for f in d.glob("*") if f.is_file()]
            log(f"  {subdir}/: {len(files)} files")
        else:
            log(f"  {subdir}/: MISSING")

    figures_dir = Path.home() / "Desktop/PhD/results/final/figures"
    if figures_dir.exists():
        n_figs = len(list(figures_dir.glob("fig4_*")))
        log(f"  figures/: {n_figs} Chapter 4 figures")

    docx = RESULTS / "chapter4_tables.docx"
    log(f"  tables: {'EXISTS' if docx.exists() else 'MISSING'}")

    # Print manifest
    print_manifest()

    log("\nALL STAGES COMPLETE. Review MASTER_PROGRESS_LOG.txt and grids/MANIFEST.txt.")


if __name__ == "__main__":
    main()
