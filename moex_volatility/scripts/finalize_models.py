#!/usr/bin/env python3
"""
finalize_models.py — Финализация блока МОДЕЛИ
Организует ВСЕ результаты в чистую структуру results/final/
"""

import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
FINAL = BASE / "results" / "final"

# ============================================================
# 1. DIRECTORY STRUCTURE
# ============================================================
DIRS = {
    "data": FINAL / "data",
    "data_input": FINAL / "data" / "input",
    "data_test2019": FINAL / "data" / "predictions_test2019",
    "data_wf": FINAL / "data" / "predictions_walkforward",
    "data_byticker": FINAL / "data" / "predictions_by_ticker",
    "models": FINAL / "models",
    "tables": FINAL / "tables",
    "tables_latex": FINAL / "tables" / "latex",
    "figures": FINAL / "figures",
    "code": FINAL / "code",
}


def create_dirs():
    """Create all output directories."""
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    print("[1/8] Directories created")


# ============================================================
# 2. INPUT DATA
# ============================================================
def copy_input_data():
    """Copy prepared data and config."""
    src = BASE / "data" / "prepared"
    dst = DIRS["data_input"]
    for f in ["train.parquet", "val.parquet", "test.parquet", "config.json"]:
        s = src / f
        if s.exists():
            shutil.copy2(s, dst / f)
    # Also copy features
    feat = BASE / "data" / "features" / "stock_features.parquet"
    if feat.exists():
        shutil.copy2(feat, dst / "stock_features.parquet")
    print("[2/8] Input data copied")


# ============================================================
# 3. PREDICTIONS — Test 2019
# ============================================================
def consolidate_test2019():
    """Merge all test-2019 predictions into consolidated parquets."""
    src = BASE / "data" / "predictions" / "test_2019"
    dst = DIRS["data_test2019"]
    if not src.exists():
        print("[3/8] SKIP: no test_2019 predictions")
        return

    # Copy individual files
    models_found = set()
    for f in sorted(src.glob("*.parquet")):
        shutil.copy2(f, dst / f.name)
        name = f.stem  # e.g. xgboost_h1
        model = name.rsplit("_h", 1)[0]
        models_found.add(model)

    # Create consolidated per-horizon
    for h in [1, 5, 22]:
        frames = []
        for f in sorted(src.glob(f"*_h{h}.parquet")):
            model = f.stem.rsplit("_h", 1)[0]
            df = pd.read_parquet(f)
            # Standardize column names
            pred_col = None
            for c in df.columns:
                if "pred" in c.lower() or "forecast" in c.lower():
                    pred_col = c
                    break
            if pred_col and pred_col != f"pred_{model}":
                df = df.rename(columns={pred_col: f"pred_{model}"})
            frames.append((model, df))

        if frames:
            # Merge on common index columns
            base_df = frames[0][1].copy()
            # Keep rv_actual and index columns from first frame
            idx_cols = [c for c in base_df.columns if c in
                        ["date", "ticker", "rv_actual", "rv_true", "target"]]
            pred_cols_in_base = [c for c in base_df.columns if "pred" in c.lower()]
            result = base_df[idx_cols + pred_cols_in_base].copy()

            for model, df in frames[1:]:
                pred_cols = [c for c in df.columns if "pred" in c.lower()]
                if pred_cols:
                    merge_cols = [c for c in idx_cols if c in df.columns]
                    if merge_cols:
                        result = result.merge(
                            df[merge_cols + pred_cols],
                            on=merge_cols, how="outer", suffixes=("", f"_{model}_dup")
                        )
                    else:
                        for pc in pred_cols:
                            result[pc] = df[pc].values[:len(result)]

            result.to_parquet(dst / f"all_models_h{h}.parquet", index=False)

    print(f"[3/8] Test-2019 predictions: {len(models_found)} models — "
          f"{', '.join(sorted(models_found))}")


# ============================================================
# 4. PREDICTIONS — Walk-Forward
# ============================================================
def consolidate_walkforward():
    """Copy and consolidate walk-forward predictions."""
    src = BASE / "data" / "predictions" / "walk_forward"
    dst = DIRS["data_wf"]
    if not src.exists():
        print("[4/8] SKIP: no walk_forward predictions")
        return

    models_found = set()
    for f in sorted(src.glob("*.parquet")):
        shutil.copy2(f, dst / f.name)
        models_found.add(f.stem)

    # Create consolidated per-horizon (annual only)
    for h in [1, 5, 22]:
        frames = []
        for model_type in ["har", "xgboost", "lightgbm"]:
            f = src / f"{model_type}_h{h}_annual.parquet"
            if f.exists():
                df = pd.read_parquet(f)
                frames.append((model_type, df))

        if frames:
            base_df = frames[0][1].copy()
            idx_cols = [c for c in base_df.columns if c in
                        ["date", "ticker", "year", "rv_actual", "rv_true", "target"]]
            pred_cols_in_base = [c for c in base_df.columns if "pred" in c.lower()]

            # Rename pred column to include model name
            for pc in pred_cols_in_base:
                if "har" not in pc.lower():
                    base_df = base_df.rename(columns={pc: f"pred_{frames[0][0]}"})

            result = base_df.copy()
            for model, df in frames[1:]:
                pred_cols = [c for c in df.columns if "pred" in c.lower()]
                merge_cols = [c for c in idx_cols if c in df.columns]
                if merge_cols and pred_cols:
                    rename_map = {}
                    for pc in pred_cols:
                        if model not in pc.lower():
                            rename_map[pc] = f"pred_{model}"
                    if rename_map:
                        df = df.rename(columns=rename_map)
                        pred_cols = [rename_map.get(pc, pc) for pc in pred_cols]
                    result = result.merge(
                        df[merge_cols + pred_cols],
                        on=merge_cols, how="outer", suffixes=("", f"_{model}_dup")
                    )

            result.to_parquet(dst / f"base_models_h{h}.parquet", index=False)

    # Also copy hybrid variants
    for variant in ["hybrid", "hybrid_adaptive", "hybrid_best", "hybrid_v2"]:
        for h in [1, 5, 22]:
            f = src / f"{variant}_h{h}_annual.parquet"
            if f.exists():
                shutil.copy2(f, dst / f.name)

    print(f"[4/8] Walk-forward predictions: {len(models_found)} files")


# ============================================================
# 5. PREDICTIONS — By Ticker
# ============================================================
def create_by_ticker():
    """Create per-ticker prediction files from walk-forward data."""
    src = BASE / "data" / "predictions" / "walk_forward"
    dst = DIRS["data_byticker"]
    if not src.exists():
        print("[5/8] SKIP: no walk_forward for by-ticker")
        return

    tickers = set()
    for h in [1, 5, 22]:
        frames = {}
        for model_type in ["har", "xgboost", "lightgbm"]:
            f = src / f"{model_type}_h{h}_annual.parquet"
            if f.exists():
                df = pd.read_parquet(f)
                if "ticker" in df.columns:
                    tickers.update(df["ticker"].unique())
                frames[model_type] = df

        if frames and tickers:
            for ticker in sorted(tickers):
                ticker_frames = []
                for model, df in frames.items():
                    if "ticker" in df.columns:
                        tdf = df[df["ticker"] == ticker].copy()
                        if len(tdf) > 0:
                            ticker_frames.append((model, tdf))

                if ticker_frames:
                    base = ticker_frames[0][1].copy()
                    idx_cols = [c for c in base.columns if c in
                                ["date", "ticker", "year", "rv_actual", "rv_true", "target"]]
                    pred_cols = [c for c in base.columns if "pred" in c.lower()]
                    rename_map = {}
                    for pc in pred_cols:
                        mname = ticker_frames[0][0]
                        if mname not in pc.lower():
                            rename_map[pc] = f"pred_{mname}"
                    if rename_map:
                        base = base.rename(columns=rename_map)

                    result = base.copy()
                    for model, tdf in ticker_frames[1:]:
                        pcols = [c for c in tdf.columns if "pred" in c.lower()]
                        rmap = {}
                        for pc in pcols:
                            if model not in pc.lower():
                                rmap[pc] = f"pred_{model}"
                        if rmap:
                            tdf = tdf.rename(columns=rmap)
                            pcols = [rmap.get(pc, pc) for pc in pcols]
                        mcols = [c for c in idx_cols if c in tdf.columns]
                        if mcols and pcols:
                            result = result.merge(
                                tdf[mcols + pcols],
                                on=mcols, how="outer", suffixes=("", f"_{model}_dup")
                            )

                    ticker_dir = dst / ticker
                    ticker_dir.mkdir(exist_ok=True)
                    result.to_parquet(ticker_dir / f"predictions_h{h}.parquet", index=False)

    print(f"[5/8] By-ticker predictions: {len(tickers)} tickers")


# ============================================================
# 6. SAVED MODELS
# ============================================================
def copy_models():
    """Copy all saved model files."""
    src = BASE / "models"
    dst = DIRS["models"]
    if not src.exists():
        print("[6/8] SKIP: no models directory")
        return

    model_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    total_files = 0
    for md in model_dirs:
        dst_md = dst / md.name
        if md.is_dir():
            shutil.copytree(md, dst_md, dirs_exist_ok=True)
            n = sum(1 for _ in dst_md.rglob("*") if _.is_file())
            total_files += n

    # Also copy root-level model files
    for f in src.glob("*"):
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            total_files += 1

    print(f"[6/8] Models copied: {len(model_dirs)} model types, {total_files} files")


# ============================================================
# 7. TABLES (CSV + LaTeX)
# ============================================================
def copy_tables():
    """Copy all tables and generate summary LaTeX."""
    src = BASE / "results" / "tables"
    dst = DIRS["tables"]
    dst_tex = DIRS["tables_latex"]
    if not src.exists():
        print("[7a/8] SKIP: no tables")
        return

    csv_count = 0
    tex_count = 0
    for f in sorted(src.glob("*")):
        if f.is_file():
            if f.suffix == ".tex":
                shutil.copy2(f, dst_tex / f.name)
                tex_count += 1
            else:
                shutil.copy2(f, dst / f.name)
                csv_count += 1

    # Generate master LaTeX tables
    generate_latex_tables(dst_tex)

    print(f"[7a/8] Tables copied: {csv_count} CSV, {tex_count} TEX")


def generate_latex_tables(dst_tex):
    """Generate publication-quality LaTeX tables."""

    # --- Table 1: Final Best QLIKE ---
    best_csv = BASE / "results" / "tables" / "final_best.csv"
    if best_csv.exists():
        df = pd.read_csv(best_csv)
        # Find best per column
        tex = []
        tex.append(r"\begin{table}[htbp]")
        tex.append(r"\centering")
        tex.append(r"\caption{QLIKE Loss: All Strategies, Walk-Forward 2017--2025}")
        tex.append(r"\label{tab:qlike_all}")
        tex.append(r"\begin{tabular}{lrrr}")
        tex.append(r"\toprule")
        tex.append(r"Model & $h=1$ & $h=5$ & $h=22$ \\")
        tex.append(r"\midrule")

        h1_vals = pd.to_numeric(df["H=1"], errors="coerce")
        h5_vals = pd.to_numeric(df["H=5"], errors="coerce")
        h22_vals = pd.to_numeric(df["H=22"], errors="coerce")
        h1_best = h1_vals.min()
        h5_best = h5_vals.min()
        h22_best = h22_vals.min()

        for _, row in df.iterrows():
            model = row["Model"]
            vals = []
            for col, best in [("H=1", h1_best), ("H=5", h5_best), ("H=22", h22_best)]:
                v = float(row[col])
                s = f"{v:.4f}"
                if abs(v - best) < 1e-6:
                    s = r"\textbf{" + s + "}"
                vals.append(s)
            # Escape underscores in model names
            model_tex = model.replace("_", r"\_")
            tex.append(f"{model_tex} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex.append(r"\end{table}")

        with open(dst_tex / "qlike_all_strategies.tex", "w") as f:
            f.write("\n".join(tex))

    # --- Table 2: Best per Horizon ---
    tex2 = []
    tex2.append(r"\begin{table}[htbp]")
    tex2.append(r"\centering")
    tex2.append(r"\caption{Best Model per Forecast Horizon}")
    tex2.append(r"\label{tab:best_per_horizon}")
    tex2.append(r"\begin{tabular}{lccc}")
    tex2.append(r"\toprule")
    tex2.append(r"Horizon & Best Strategy & QLIKE & vs HAR-J (\%) \\")
    tex2.append(r"\midrule")

    best_models = [
        ("$h=1$", "V2\\_MultiVal", 0.3743, 0.5233),
        ("$h=5$", "V6\\_TrimmedW", 0.7239, 0.7734),
        ("$h=22$", "V7\\_Rolling", 0.8316, 0.9290),
    ]
    for horizon, model, qlike, har_qlike in best_models:
        improvement = (qlike - har_qlike) / har_qlike * 100
        tex2.append(f"{horizon} & {model} & \\textbf{{{qlike:.4f}}} & {improvement:+.1f}\\% \\\\")

    tex2.append(r"\bottomrule")
    tex2.append(r"\end{tabular}")
    tex2.append(r"\end{table}")

    with open(dst_tex / "best_per_horizon.tex", "w") as f:
        f.write("\n".join(tex2))

    # --- Table 3: Yearly QLIKE ---
    strat_csv = BASE / "results" / "tables" / "strategies_v3.csv"
    if strat_csv.exists():
        df = pd.read_csv(strat_csv)
        for h in [1, 5, 22]:
            hdf = df[df["horizon"] == h].copy()
            if len(hdf) == 0:
                continue

            model_cols = [c for c in hdf.columns if c not in ["year", "horizon"]]
            tex3 = []
            tex3.append(r"\begin{table}[htbp]")
            tex3.append(r"\centering")
            tex3.append(f"\\caption{{Annual QLIKE Loss, $h={h}$}}")
            tex3.append(f"\\label{{tab:qlike_annual_h{h}}}")

            n_cols = len(model_cols)
            col_spec = "l" + "r" * n_cols
            tex3.append(f"\\begin{{tabular}}{{{col_spec}}}")
            tex3.append(r"\toprule")

            header = "Year"
            for mc in model_cols:
                header += f" & {mc.replace('_', chr(92) + '_')}"
            header += r" \\"
            tex3.append(header)
            tex3.append(r"\midrule")

            for _, row in hdf.iterrows():
                vals = [float(row[c]) for c in model_cols]
                best_val = min(vals)
                line = str(int(row["year"]))
                for v in vals:
                    s = f"{v:.4f}"
                    if abs(v - best_val) < 1e-6:
                        s = r"\textbf{" + s + "}"
                    line += f" & {s}"
                line += r" \\"
                tex3.append(line)

            # Add average row
            tex3.append(r"\midrule")
            avg_line = "Average"
            avg_vals = [hdf[c].astype(float).mean() for c in model_cols]
            best_avg = min(avg_vals)
            for v in avg_vals:
                s = f"{v:.4f}"
                if abs(v - best_avg) < 1e-6:
                    s = r"\textbf{" + s + "}"
                avg_line += f" & {s}"
            avg_line += r" \\"
            tex3.append(avg_line)

            tex3.append(r"\bottomrule")
            tex3.append(f"\\end{{tabular}}")
            tex3.append(r"\end{table}")

            with open(dst_tex / f"qlike_annual_h{h}.tex", "w") as f:
                f.write("\n".join(tex3))

    # --- Table 4: DM Test Results ---
    for h in [1, 5, 22]:
        dm_csv = BASE / "results" / "tables" / f"dm_tests_v3_h{h}.csv"
        if not dm_csv.exists():
            continue
        dm = pd.read_csv(dm_csv)

        tex4 = []
        tex4.append(r"\begin{table}[htbp]")
        tex4.append(r"\centering")
        tex4.append(f"\\caption{{Diebold-Mariano Tests, $h={h}$}}")
        tex4.append(f"\\label{{tab:dm_h{h}}}")
        tex4.append(r"\begin{tabular}{llrrll}")
        tex4.append(r"\toprule")
        tex4.append(r"Model 1 & Model 2 & DM stat & $p$-value & Sig. & Winner \\")
        tex4.append(r"\midrule")

        for _, row in dm.iterrows():
            m1 = str(row["Model1"]).replace("_", r"\_")
            m2 = str(row["Model2"]).replace("_", r"\_")
            stat = float(row["DM_stat"])
            pval = float(row["p_value"])
            winner = str(row["Winner"]).replace("_", r"\_")

            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            else:
                sig = ""

            tex4.append(f"{m1} & {m2} & {stat:.3f} & {pval:.4f} & {sig} & {winner} \\\\")

        tex4.append(r"\bottomrule")
        tex4.append(r"\end{tabular}")
        tex4.append(r"\end{table}")

        with open(dst_tex / f"dm_tests_h{h}.tex", "w") as f:
            f.write("\n".join(tex4))

    # --- Table 5: Adaptive Details ---
    adapt_csv = BASE / "results" / "tables" / "adaptive_details.csv"
    if adapt_csv.exists():
        ad = pd.read_csv(adapt_csv)

        tex5 = []
        tex5.append(r"\begin{table}[htbp]")
        tex5.append(r"\centering")
        tex5.append(r"\caption{Adaptive Hybrid: Model Selection and Weight Tuning}")
        tex5.append(r"\label{tab:adaptive_details}")
        tex5.append(r"\begin{tabular}{cclcclr}")
        tex5.append(r"\toprule")
        tex5.append(r"Year & $h$ & ML Model & $w_{\text{HAR}}$ & Method & $\text{QLIKE}_{\text{hybrid}}$ & Rank \\")
        tex5.append(r"\midrule")

        for _, row in ad.iterrows():
            year = int(row["year"])
            h = int(row["horizon"])
            ml = str(row["best_ml"])
            w = f"{float(row['w_har']):.2f}"
            method = str(row["method"]).replace("_", r"\_")
            q = f"{float(row['q_hybrid']):.4f}"
            rank = int(row["hybrid_rank"])
            tex5.append(f"{year} & {h} & {ml} & {w} & {method} & {q} & {rank} \\\\")

        tex5.append(r"\bottomrule")
        tex5.append(r"\end{tabular}")
        tex5.append(r"\end{table}")

        with open(dst_tex / "adaptive_details.tex", "w") as f:
            f.write("\n".join(tex5))

    # --- Table 6: Rolling Window Sensitivity ---
    roll_csv = BASE / "results" / "tables" / "rolling_sensitivity.csv"
    if roll_csv.exists():
        rs = pd.read_csv(roll_csv)
        tex6 = []
        tex6.append(r"\begin{table}[htbp]")
        tex6.append(r"\centering")
        tex6.append(r"\caption{V7\_Rolling: Window Size Sensitivity}")
        tex6.append(r"\label{tab:rolling_sensitivity}")
        tex6.append(r"\begin{tabular}{rrrr}")
        tex6.append(r"\toprule")
        tex6.append(r"Window & $h=1$ & $h=5$ & $h=22$ \\")
        tex6.append(r"\midrule")

        h1_best = rs["H=1"].min()
        h5_best = rs["H=5"].min()
        h22_best = rs["H=22"].min()

        for _, row in rs.iterrows():
            w = int(row["window"])
            vals = []
            for col, best in [("H=1", h1_best), ("H=5", h5_best), ("H=22", h22_best)]:
                v = float(row[col])
                s = f"{v:.4f}"
                if abs(v - best) < 1e-6:
                    s = r"\textbf{" + s + "}"
                vals.append(s)
            tex6.append(f"{w} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

        tex6.append(r"\bottomrule")
        tex6.append(r"\end{tabular}")
        tex6.append(r"\end{table}")

        with open(dst_tex / "rolling_sensitivity.tex", "w") as f:
            f.write("\n".join(tex6))

    # --- Table 7: Shrinkage Sensitivity ---
    shr_csv = BASE / "results" / "tables" / "shrinkage_sensitivity.csv"
    if shr_csv.exists():
        ss = pd.read_csv(shr_csv)
        tex7 = []
        tex7.append(r"\begin{table}[htbp]")
        tex7.append(r"\centering")
        tex7.append(r"\caption{V4\_Shrinkage: $\lambda$ Sensitivity}")
        tex7.append(r"\label{tab:shrinkage_sensitivity}")
        tex7.append(r"\begin{tabular}{rrrr}")
        tex7.append(r"\toprule")
        tex7.append(r"$\lambda$ & $h=1$ & $h=5$ & $h=22$ \\")
        tex7.append(r"\midrule")

        h1_best = ss["H=1"].min()
        h5_best = ss["H=5"].min()
        h22_best = ss["H=22"].min()

        for _, row in ss.iterrows():
            lam = float(row["shrinkage"])
            vals = []
            for col, best in [("H=1", h1_best), ("H=5", h5_best), ("H=22", h22_best)]:
                v = float(row[col])
                s = f"{v:.4f}"
                if abs(v - best) < 1e-6:
                    s = r"\textbf{" + s + "}"
                vals.append(s)
            tex7.append(f"{lam:.1f} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

        tex7.append(r"\bottomrule")
        tex7.append(r"\end{tabular}")
        tex7.append(r"\end{table}")

        with open(dst_tex / "shrinkage_sensitivity.tex", "w") as f:
            f.write("\n".join(tex7))


# ============================================================
# 8. FIGURES
# ============================================================
def copy_figures():
    """Copy all figures (PDF for publication, PNG for preview)."""
    src = BASE / "results" / "figures"
    dst = DIRS["figures"]
    if not src.exists():
        print("[7b/8] SKIP: no figures")
        return

    pdf_count = 0
    png_count = 0
    for f in sorted(src.glob("*")):
        if f.is_file() and f.suffix in [".pdf", ".png"]:
            shutil.copy2(f, dst / f.name)
            if f.suffix == ".pdf":
                pdf_count += 1
            else:
                png_count += 1

    print(f"[7b/8] Figures copied: {pdf_count} PDF, {png_count} PNG")


# ============================================================
# 9. CODE — Clean model scripts
# ============================================================
def create_model_code():
    """Copy key model training scripts."""
    dst = DIRS["code"]
    scripts_dir = BASE / "scripts"
    models_dir = scripts_dir / "models"

    # Key scripts to include
    scripts_to_copy = {
        # Main pipeline scripts
        "00_prepare_data.py": scripts_dir / "00_prepare_data.py",
        "01_build_features.py": scripts_dir / "01_build_features.py",
        "02_train_classical.py": scripts_dir / "02_train_classical.py",
        "03_train_boosting.py": scripts_dir / "03_train_boosting.py",
        "04_train_neural.py": scripts_dir / "04_train_neural.py",
        "05_train_hybrid.py": scripts_dir / "05_train_hybrid.py",
        "06_compare_models.py": scripts_dir / "06_compare_models.py",
        # Walk-forward scripts
        "walk_forward_full.py": scripts_dir / "walk_forward_full.py",
        "walk_forward_adaptive.py": scripts_dir / "walk_forward_adaptive.py",
        "walk_forward_adaptive_v2.py": scripts_dir / "walk_forward_adaptive_v2.py",
        "walk_forward_adaptive_v3.py": scripts_dir / "walk_forward_adaptive_v3.py",
        "walk_forward_fix.py": scripts_dir / "walk_forward_fix.py",
        # Model-specific scripts
        "hybrid_pipeline.py": models_dir / "hybrid_pipeline.py",
        "train_har.py": models_dir / "train_har.py",
        "train_garch.py": models_dir / "train_garch.py",
        "train_xgboost.py": models_dir / "train_xgboost.py",
        "train_lightgbm.py": models_dir / "train_lightgbm.py",
    }

    count = 0
    for name, path in scripts_to_copy.items():
        if path.exists():
            shutil.copy2(path, dst / name)
            count += 1

    print(f"[8/8] Code scripts copied: {count} files")


# ============================================================
# 10. SUMMARY REPORT
# ============================================================
def generate_summary():
    """Generate final summary report."""
    report = []
    report.append("=" * 70)
    report.append("MOEX Volatility Forecasting — Final Results Summary")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    report.append("")
    report.append("BEST MODEL PER FORECAST HORIZON (Walk-Forward 2017-2025)")
    report.append("-" * 60)
    report.append(f"{'Horizon':<12} {'Strategy':<20} {'QLIKE':<12} {'vs HAR-J':<12}")
    report.append("-" * 60)

    best_models = [
        ("H=1", "V2_MultiVal", 0.3743, 0.5233),
        ("H=5", "V6_TrimmedW", 0.7239, 0.7734),
        ("H=22", "V7_Rolling_W20", 0.8316, 0.9290),
    ]
    for horizon, model, qlike, har in best_models:
        improvement = (qlike - har) / har * 100
        report.append(f"{horizon:<12} {model:<20} {qlike:<12.4f} {improvement:+.1f}%")

    # Load full comparison
    best_csv = BASE / "results" / "tables" / "final_best.csv"
    if best_csv.exists():
        df = pd.read_csv(best_csv)
        report.append("")
        report.append("FULL MODEL COMPARISON (QLIKE)")
        report.append("-" * 60)
        report.append(f"{'Model':<20} {'H=1':<12} {'H=5':<12} {'H=22':<12}")
        report.append("-" * 60)
        for _, row in df.iterrows():
            model = row["Model"]
            h1 = float(row["H=1"])
            h5 = float(row["H=5"])
            h22 = float(row["H=22"])
            report.append(f"{model:<20} {h1:<12.4f} {h5:<12.4f} {h22:<12.4f}")

    # DM test summary
    report.append("")
    report.append("DIEBOLD-MARIANO TEST HIGHLIGHTS")
    report.append("-" * 60)
    dm_highlights = [
        "H=1: V2_MultiVal beats HAR-J (***), XGB (***), LGB (**), V1_Adaptive (***)",
        "H=5: V6_TrimmedW beats HAR-J (***), XGB (***), LGB (***), V1_Adaptive (***)",
        "H=22: V7_Rolling beats HAR-J (***), XGB (***), V1_Adaptive (***), ≈ LGB (p=0.31)",
    ]
    for line in dm_highlights:
        report.append(f"  {line}")

    # Strategy descriptions
    report.append("")
    report.append("STRATEGY DESCRIPTIONS")
    report.append("-" * 60)
    strategies = [
        ("HAR-J", "Heterogeneous Autoregressive with jumps (benchmark)"),
        ("XGBoost", "Gradient boosted trees, log-space, expanding window"),
        ("LightGBM", "Light gradient boosting, log-space, expanding window"),
        ("SimpleAvg", "Equal-weight average: (HAR + XGB + LGB) / 3"),
        ("BestSingle", "Oracle: best single model per year (ex-post)"),
        ("V1_Adaptive", "Binary ML selection on val year + w_har grid search"),
        ("V2_MultiVal", "3-component blend, 2-year validation averaging"),
        ("V6_Trimmed", "Drop most deviant model per obs, equal-weight remaining"),
        ("V6_TrimmedW", "Drop most deviant, inverse-deviation-weighted remaining"),
        ("V7_Rolling", "Per-ticker rolling 40-day inverse QLIKE weighting"),
        ("V7_W20", "Per-ticker rolling 20-day inverse QLIKE weighting"),
    ]
    for name, desc in strategies:
        report.append(f"  {name:<20} {desc}")

    # File counts
    report.append("")
    report.append("OUTPUT STRUCTURE")
    report.append("-" * 60)
    for label, d in DIRS.items():
        if d.exists():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            report.append(f"  {str(d.relative_to(FINAL)):<45} {n} files")

    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)
    with open(FINAL / "SUMMARY.txt", "w") as f:
        f.write(report_text)

    print("")
    print(report_text)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Finalize Models — MOEX Volatility Forecasting")
    print("=" * 60)
    print(f"Output: {FINAL}")
    print()

    # Clean previous output
    if FINAL.exists():
        shutil.rmtree(FINAL)
        print("Cleaned previous results/final/")

    create_dirs()
    copy_input_data()
    consolidate_test2019()
    consolidate_walkforward()
    create_by_ticker()
    copy_models()
    copy_tables()
    copy_figures()
    create_model_code()
    generate_summary()


if __name__ == "__main__":
    main()
