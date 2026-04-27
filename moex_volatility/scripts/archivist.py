#!/usr/bin/env python3
"""
archivist.py — Agent 10: Reproducibility & grid archival.

Called after each successful stage validation.
Saves grids, best params, results, and manifest for full reproducibility.

Also provides pre-stage input checks.
"""
import os, json, datetime
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
STRAT_DIR = BASE / "results/final/strategies"
GRIDS_DIR = STRAT_DIR / "grids"
GRIDS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = GRIDS_DIR / "MANIFEST.txt"
LOG_PATH = STRAT_DIR / "MASTER_PROGRESS_LOG.txt"


def _log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] ARCHIVIST: {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def archive(stage_name, results_df, configs, grid_used, out_dir=None):
    """
    Called AFTER each successful stage (validator PASS).
    Saves everything needed to reproduce results.

    Args:
        stage_name: e.g. "stage1_test_A_005", "stage2_bcd_B_005"
        results_df: pd.DataFrame with at least 'sharpe' column
        configs: dict of best params per strategy×ticker (or any serializable)
        grid_used: dict of parameter grid that was searched
        out_dir: override output directory (default: STRAT_DIR)
    """
    if out_dir is None:
        out_dir = STRAT_DIR

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Grid used
    grid_path = GRIDS_DIR / f"{stage_name}_grid.json"
    with open(grid_path, 'w') as f:
        json.dump(grid_used, f, indent=2, default=str)
    _log(f"saved grid → {grid_path.name}")

    # 2. Best params
    params_path = GRIDS_DIR / f"{stage_name}_best_params.json"
    with open(params_path, 'w') as f:
        json.dump(configs, f, indent=2, default=str)
    _log(f"saved best params → {params_path.name}")

    # 3. Results (CSV for human, parquet for code)
    csv_path = Path(out_dir) / f"{stage_name}_results.csv"
    pq_path = Path(out_dir) / f"{stage_name}_results.parquet"
    if results_df is not None and len(results_df) > 0:
        results_df.to_csv(csv_path, index=False)
        results_df.to_parquet(pq_path, index=False)
    _log(f"saved results → {csv_path.name} ({len(results_df) if results_df is not None else 0} rows)")

    # 4. Versioned copy (for auto-correction iterations)
    version_path = GRIDS_DIR / f"{stage_name}_grid_{ts}.json"
    with open(version_path, 'w') as f:
        json.dump(grid_used, f, indent=2, default=str)

    # 5. Manifest entry
    mean_sh = "N/A"
    if results_df is not None and 'sharpe' in results_df.columns:
        valid = results_df['sharpe'].dropna()
        valid = valid[valid > -900]
        mean_sh = f"{valid.mean():.2f}" if len(valid) > 0 else "N/A"

    n_configs = 0
    if isinstance(configs, dict):
        # Count leaf entries
        n_configs = sum(len(v) if isinstance(v, dict) else 1 for v in configs.values())
    elif isinstance(configs, list):
        n_configs = len(configs)

    with open(MANIFEST_PATH, 'a') as f:
        f.write(f"[{ts}] {stage_name}: grid={grid_path.name}, "
                f"params={params_path.name}, mean_sharpe={mean_sh}, "
                f"n_configs={n_configs}\n")

    # 6. Verify files written
    for p in [grid_path, params_path]:
        if not p.exists() or p.stat().st_size < 10:
            _log(f"ERROR: {p.name} failed to write!")
            return False

    _log(f"{stage_name} archived successfully")
    return True


def check_inputs(stage_name, required_files):
    """
    Check that all input files exist before running a stage.
    required_files: list of paths (str or Path), relative to STRAT_DIR or absolute.
    """
    missing = []
    for f in required_files:
        fp = Path(f) if Path(f).is_absolute() else STRAT_DIR / f
        if not fp.exists():
            missing.append(str(fp))

    if missing:
        _log(f"CHECK FAILED for {stage_name}: missing {len(missing)} files:")
        for m in missing:
            _log(f"  ✗ {m}")
        return False

    _log(f"CHECK OK for {stage_name}: all {len(required_files)} inputs present")
    return True


def archive_grid_version(stage_name, grid, version_label):
    """Save a specific version of a grid (e.g. after auto-correction)."""
    path = GRIDS_DIR / f"{stage_name}_grid_{version_label}.json"
    with open(path, 'w') as f:
        json.dump(grid, f, indent=2, default=str)
    _log(f"saved grid version → {path.name}")

    # Log decision
    decision_log = STRAT_DIR / "DECISION_LOG.txt"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(decision_log, 'a') as f:
        f.write(f"[{ts}] {stage_name} used {path.name}\n")

    return path


def print_manifest():
    """Print the manifest for review."""
    if MANIFEST_PATH.exists():
        print(f"\n{'='*80}")
        print("MANIFEST (all archived stages):")
        print(f"{'='*80}")
        print(MANIFEST_PATH.read_text())
        print(f"{'='*80}\n")
    else:
        print("No manifest yet.")


# ══════════════════════════════════════════════════════════════
# PRE-DEFINED INPUT CHECKS per stage
# ══════════════════════════════════════════════════════════════

STAGE_INPUTS = {
    "stage1_test_A": [],  # no dependencies
    "stage2_bcd": [
        "test_phases/FINAL_best_configs.json",
    ],
    "stage3_walkforward": [
        "test_phases/FINAL_best_configs.json",
        "walkforward_v4/narrowed_grid_v2.json",
    ],
    "stage4_hourly": [
        "test_phases/FINAL_best_configs.json",
    ],
    "stage5_msp": [
        "walkforward/wf_positions_A.parquet",
        "walkforward/wf_positions_B.parquet",
        "walkforward/wf_positions_C.parquet",
        "walkforward/wf_positions_D.parquet",
    ],
    "stage6_correlations": [
        "walkforward/wf_positions_A.parquet",
    ],
    "stage7_bootstrap": [
        "msp/msp_positions.parquet",
        "walkforward/wf_positions_A.parquet",
    ],
    "stage8_economic": [
        "msp/msp_positions.parquet",
        "msp/msp_results.csv",
    ],
    "stage9_figures": [
        "msp/msp_positions.parquet",
    ],
    "stage10_tables": [
        "msp/msp_results.csv",
    ],
}


def check_stage_inputs(stage_name):
    """Check inputs for a named stage using pre-defined requirements."""
    required = STAGE_INPUTS.get(stage_name, [])
    if not required:
        _log(f"CHECK OK for {stage_name}: no required inputs")
        return True
    return check_inputs(stage_name, required)


# ══════════════════════════════════════════════════════════════
# AGENT 11: САНИТАР (sanity check BEFORE launch)
# ══════════════════════════════════════════════════════════════

def agent_sanity_check(stage_name, params):
    """Check for absurd parameters BEFORE burning CPU."""
    errors = []
    warnings = []

    n = params.get('n_combos_per_ticker', 0)
    if n > 50000:
        errors.append(f"АБСУРД: {n:,} combos/ticker. Max 50,000.")
    elif n > 10000:
        warnings.append(f"{n:,} combos/ticker — ETA ~{n * 17 * 4 * 0.001 / 60:.0f} мин")

    eta = params.get('estimated_seconds', 0)
    if eta > 7200:
        errors.append(f"ETA = {eta / 3600:.1f}ч. Max 2ч.")

    n_sig = params.get('n_signal', 0)
    n_rm = params.get('n_rm', 0)
    n_filt = params.get('n_filter', 1)
    n_conf = params.get('n_conf', 1)
    if n_sig > 0 and n_rm > 0:
        total = n_sig * n_rm * n_filt * n_conf
        if total > 50000:
            errors.append(f"sig({n_sig})×rm({n_rm})×filt({n_filt})×conf({n_conf})={total:,}")

    if params.get('w', 1.0) != 1.0:
        errors.append(f"w = {params['w']}. ДОЛЖНО быть 1.0.")

    for f in params.get('required_files', []):
        if not Path(f).exists() and not (STRAT_DIR / f).exists():
            errors.append(f"File missing: {f}")

    if errors:
        _log(f"❌ САНИТАР {stage_name}: BLOCKED")
        for e in errors:
            _log(f"  {e}")
        return False

    _log(f"✅ САНИТАР {stage_name}: OK ({n} combos/ticker)")
    for w in warnings:
        _log(f"  ⚠️ {w}")
    return True


# ══════════════════════════════════════════════════════════════
# AGENT 12: ТАЙМ-МЕНЕДЖЕР (ETA monitoring inside stage)
# ══════════════════════════════════════════════════════════════

import time as _time

class TimeManager:
    def __init__(self, stage_name, n_total, max_seconds):
        self.stage_name = stage_name
        self.n_total = n_total
        self.max_seconds = max_seconds
        self.start = _time.time()
        self.n_done = 0

    def tick(self, n=1):
        self.n_done += n
        elapsed = _time.time() - self.start
        if self.n_done >= 3:
            eta_total = elapsed / self.n_done * self.n_total
            remaining = eta_total - elapsed
            step = max(1, self.n_total // 10)
            if self.n_done % step == 0:
                _log(f"  {self.stage_name}: {self.n_done}/{self.n_total} "
                     f"({self.n_done / self.n_total * 100:.0f}%), "
                     f"ETA {remaining / 60:.1f}min")
            if eta_total > self.max_seconds:
                _log(f"  ⚠️ ABORT {self.stage_name}: ETA {eta_total / 60:.0f}min > "
                     f"limit {self.max_seconds / 60:.0f}min")
                return False
        return True

    def summary(self):
        elapsed = _time.time() - self.start
        _log(f"  {self.stage_name}: {self.n_done}/{self.n_total} in {elapsed / 60:.1f}min")


# ══════════════════════════════════════════════════════════════
# AGENT 13: QUICK-CHECK (per-ticker sanity)
# ══════════════════════════════════════════════════════════════

def quick_check(sid, ticker, year, sharpe, exposure):
    if sharpe < -10 or sharpe > 20:
        _log(f"  ⚠️ {sid}/{ticker}/{year}: Sharpe={sharpe:.2f} вне [-10,20]")
        return False
    if exposure < 0 or exposure > 100:
        _log(f"  ⚠️ {sid}/{ticker}/{year}: Exposure={exposure:.1f}% вне [0,100]")
        return False
    return True


# ══════════════════════════════════════════════════════════════
# CONFIGS WITH FULL PRECISION (json + pickle)
# ══════════════════════════════════════════════════════════════

def save_configs_full(configs, stage_name):
    """Save configs as json + pickle for full reproducibility."""
    import pickle
    json_path = GRIDS_DIR / f"{stage_name}_configs.json"
    pkl_path = GRIDS_DIR / f"{stage_name}_configs.pkl"
    with open(json_path, 'w') as f:
        json.dump(configs, f, indent=2, default=str)
    with open(pkl_path, 'wb') as f:
        pickle.dump(configs, f)
    _log(f"Configs saved: {json_path.name} + {pkl_path.name}")


# ══════════════════════════════════════════════════════════════
# AGENT: HOURLY ADAPTER (scale daily configs for hourly bars)
# ══════════════════════════════════════════════════════════════

HOURLY_SCALE = 7  # bars per day on MOEX

WINDOW_PARAMS = ['w', 'ap', 'atr_period', 'vwap_window', 'bb_window',
                 'dc_window', 'vw', 'dc']
NO_SCALE_PARAMS = ['thresh', 'mult', 'bs', 'dev_mult', 'multiplier',
                   'level', 'dm', 'el', 'pt', 'ma']


def agent_hourly_adapter(daily_config, scale=None):
    """Adapt daily config for hourly bars. Window params × scale."""
    if scale is None:
        scale = HOURLY_SCALE
    sig = json.loads(daily_config['signal']) if isinstance(
        daily_config.get('signal'), str) else daily_config.get('signal', {})
    adapted = sig.copy()
    for key, val in adapted.items():
        if key in WINDOW_PARAMS and isinstance(val, (int, float)):
            adapted[key] = (int(val * scale) if isinstance(val, int)
                            else round(val * scale, 1))
    return adapted


if __name__ == "__main__":
    print("Archivist module — import and use archive() / check_inputs()")
    print(f"GRIDS_DIR: {GRIDS_DIR}")
    print(f"MANIFEST: {MANIFEST_PATH}")
    print_manifest()
