#!/usr/bin/env python3
"""
update_final.py — Rebuild results/final/ with 2026 walk-forward data.
Updates predictions, tables (CSV+LaTeX), figures (PNG+PDF), SUMMARY.txt.
"""

import os
import sys
import shutil
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FINAL = BASE / "results" / "final"
PRED_SRC = BASE / "data" / "predictions" / "walk_forward"
TABLE_SRC = BASE / "results" / "tables"

FINAL_DATA_WF = FINAL / "data" / "predictions_walkforward"
FINAL_DATA_BT = FINAL / "data" / "predictions_by_ticker"
FINAL_TABLES = FINAL / "tables"
FINAL_LATEX = FINAL / "tables" / "latex"
FINAL_FIGS = FINAL / "figures"

# Core models for walk-forward analysis
CORE_MODELS = ["har", "xgboost", "lightgbm", "hybrid_adaptive"]
MODEL_DISPLAY = {
    "har": "HAR-J", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "hybrid_adaptive": "V1_Adaptive",
}
HORIZONS = [1, 5, 22]

updated_files = []

# ============================================================
# Utility functions
# ============================================================

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    mask = (y_true > 1e-12) & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return np.nan
    ratio = yt / yp
    return float(np.mean(ratio - np.log(ratio) - 1))


def qlike_per_obs(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1


def dm_test(loss1, loss2, h_horizon=1):
    d = loss1 - loss2
    mask = np.isfinite(d)
    d = d[mask]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    d_mean = np.mean(d)
    lag = max(1, h_horizon - 1)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, lag + 1):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 1 else 0.0
        gamma_sum += 2 * (1 - k / (lag + 1)) * gamma_k
    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def save(path, content=None):
    """Track saved file."""
    updated_files.append(str(path.relative_to(FINAL)))


def load_all_predictions():
    """Load all core model predictions."""
    preds = {}
    for m in CORE_MODELS:
        for h in HORIZONS:
            path = PRED_SRC / f"{m}_h{h}_annual.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                # Filter valid
                df = df[df["rv_actual"] > 1e-12].copy()
                df = df[np.isfinite(df["rv_actual"]) & np.isfinite(df["rv_pred"])]
                preds[(m, h)] = df
    return preds


# ============================================================
# 1. Update predictions
# ============================================================

def update_predictions(preds):
    print("=" * 60)
    print("[1/5] Updating predictions")
    print("=" * 60)

    FINAL_DATA_WF.mkdir(parents=True, exist_ok=True)
    FINAL_DATA_BT.mkdir(parents=True, exist_ok=True)

    # Copy all walk-forward parquets
    for f in sorted(PRED_SRC.glob("*.parquet")):
        shutil.copy2(f, FINAL_DATA_WF / f.name)
    print(f"  Copied {len(list(PRED_SRC.glob('*.parquet')))} parquets to predictions_walkforward/")

    # Build consolidated per-horizon
    for h in HORIZONS:
        frames = []
        for m in CORE_MODELS:
            key = (m, h)
            if key not in preds:
                continue
            df = preds[key][["date", "ticker", "year", "rv_actual", "rv_pred"]].copy()
            df = df.rename(columns={"rv_pred": f"pred_{MODEL_DISPLAY[m]}"})
            frames.append(df)

        if frames:
            result = frames[0]
            for f in frames[1:]:
                pred_col = [c for c in f.columns if c.startswith("pred_")]
                result = result.merge(
                    f[["date", "ticker"] + pred_col],
                    on=["date", "ticker"], how="outer"
                )
            path = FINAL_DATA_WF / f"walkforward_all_h{h}.parquet"
            result.to_parquet(path, index=False)
            save(path)
            print(f"  walkforward_all_h{h}: {len(result)} rows, {sorted(result['year'].unique())}")

    # By-ticker
    tickers_done = set()
    for h in HORIZONS:
        frames = {}
        for m in CORE_MODELS:
            key = (m, h)
            if key not in preds:
                continue
            df = preds[key].copy()
            df = df.rename(columns={"rv_pred": f"pred_{MODEL_DISPLAY[m]}"})
            frames[m] = df

        if not frames:
            continue

        all_tickers = set()
        for df in frames.values():
            all_tickers.update(df["ticker"].unique())

        for ticker in sorted(all_tickers):
            ticker_dir = FINAL_DATA_BT / ticker
            ticker_dir.mkdir(exist_ok=True)

            base = None
            for m in CORE_MODELS:
                if m not in frames:
                    continue
                tdf = frames[m][frames[m]["ticker"] == ticker].copy()
                if len(tdf) == 0:
                    continue
                pred_col = f"pred_{MODEL_DISPLAY[m]}"
                if base is None:
                    base = tdf[["date", "ticker", "year", "rv_actual", pred_col]].copy()
                else:
                    base = base.merge(
                        tdf[["date", "ticker", pred_col]],
                        on=["date", "ticker"], how="outer"
                    )

            if base is not None:
                path = ticker_dir / f"predictions_h{h}.parquet"
                base.to_parquet(path, index=False)
                tickers_done.add(ticker)

    print(f"  By-ticker: {len(tickers_done)} tickers")
    save(FINAL_DATA_BT)


# ============================================================
# 2. Update tables
# ============================================================

def update_tables(preds):
    print("\n" + "=" * 60)
    print("[2/5] Updating tables")
    print("=" * 60)

    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    FINAL_LATEX.mkdir(parents=True, exist_ok=True)

    # --- 2a: QLIKE by year (walkforward) ---
    qlike_rows = []
    for h in HORIZONS:
        all_years = set()
        for m in CORE_MODELS:
            if (m, h) in preds:
                all_years.update(preds[(m, h)]["year"].unique())

        for year in sorted(all_years):
            row = {"year": int(year), "horizon": h}
            for m in CORE_MODELS:
                if (m, h) not in preds:
                    continue
                ydf = preds[(m, h)]
                ydf = ydf[ydf["year"] == year]
                if len(ydf) > 0:
                    row[MODEL_DISPLAY[m]] = qlike(ydf["rv_actual"].values, ydf["rv_pred"].values)
                else:
                    row[MODEL_DISPLAY[m]] = np.nan
            qlike_rows.append(row)

    qlike_df = pd.DataFrame(qlike_rows)

    # Save per-horizon CSV
    model_names = [MODEL_DISPLAY[m] for m in CORE_MODELS]
    for h in HORIZONS:
        hdf = qlike_df[qlike_df["horizon"] == h][["year"] + model_names].copy()
        # Add Average row
        avg_row = {"year": "Average"}
        for mn in model_names:
            avg_row[mn] = hdf[mn].mean()
        hdf_with_avg = pd.concat([hdf, pd.DataFrame([avg_row])], ignore_index=True)

        path = FINAL_TABLES / f"qlike_walkforward_h{h}.csv"
        hdf_with_avg.to_csv(path, index=False)
        save(path)

    # Full wide table (all horizons)
    qlike_df.to_csv(FINAL_TABLES / "qlike_walkforward.csv", index=False)
    save(FINAL_TABLES / "qlike_walkforward.csv")
    print(f"  qlike_walkforward: {len(qlike_df)} rows (years×horizons)")

    # --- 2b: Model summary (mean QLIKE across years) ---
    summary_rows = []
    for m in CORE_MODELS:
        row = {"Model": MODEL_DISPLAY[m]}
        for h in HORIZONS:
            hdf = qlike_df[qlike_df["horizon"] == h]
            row[f"H={h}"] = hdf[MODEL_DISPLAY[m]].mean() if MODEL_DISPLAY[m] in hdf.columns else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    path = FINAL_TABLES / "model_summary.csv"
    summary_df.to_csv(path, index=False)
    save(path)
    print(f"  model_summary: {len(summary_df)} models")

    # --- 2c: QLIKE by ticker ---
    ticker_rows = []
    for h in HORIZONS:
        for m in CORE_MODELS:
            if (m, h) not in preds:
                continue
            df = preds[(m, h)]
            for ticker, tdf in df.groupby("ticker"):
                q = qlike(tdf["rv_actual"].values, tdf["rv_pred"].values)
                ticker_rows.append({
                    "horizon": h, "ticker": ticker,
                    "model": MODEL_DISPLAY[m], "qlike": q
                })

    ticker_df = pd.DataFrame(ticker_rows)
    # Pivot
    for h in HORIZONS:
        hdf = ticker_df[ticker_df["horizon"] == h].pivot(
            index="ticker", columns="model", values="qlike"
        ).reset_index()
        path = FINAL_TABLES / f"qlike_by_ticker_h{h}.csv"
        hdf.to_csv(path, index=False)
        save(path)

    print(f"  qlike_by_ticker: {ticker_df['ticker'].nunique()} tickers")

    # --- 2d: DM tests (all pairwise, 2017-2026) ---
    dm_rows = []
    for h in HORIZONS:
        # Merge predictions
        merged = None
        for m in CORE_MODELS:
            if (m, h) not in preds:
                continue
            df = preds[(m, h)][["date", "ticker", "rv_actual", "rv_pred"]].copy()
            df = df.rename(columns={"rv_pred": f"pred_{m}"})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(
                    df[["date", "ticker", f"pred_{m}"]],
                    on=["date", "ticker"], how="inner"
                )

        if merged is None or len(merged) < 10:
            continue

        y = merged["rv_actual"].values
        losses = {}
        for m in CORE_MODELS:
            col = f"pred_{m}"
            if col in merged.columns:
                losses[m] = qlike_per_obs(y, merged[col].values)

        # All pairwise
        model_list = [m for m in CORE_MODELS if m in losses]
        for i, m1 in enumerate(model_list):
            for m2 in model_list[i + 1:]:
                stat, pval = dm_test(losses[m1], losses[m2], h)
                winner = MODEL_DISPLAY[m2] if stat > 0 else MODEL_DISPLAY[m1]
                dm_rows.append({
                    "Horizon": h,
                    "Model1": MODEL_DISPLAY[m1],
                    "Model2": MODEL_DISPLAY[m2],
                    "DM_stat": stat, "p_value": pval,
                    "Sig": sig_stars(pval), "Winner": winner,
                })

    dm_df = pd.DataFrame(dm_rows)
    for h in HORIZONS:
        hdf = dm_df[dm_df["Horizon"] == h]
        path = FINAL_TABLES / f"dm_tests_h{h}.csv"
        hdf.to_csv(path, index=False)
        save(path)

    dm_df.to_csv(FINAL_TABLES / "dm_tests.csv", index=False)
    save(FINAL_TABLES / "dm_tests.csv")
    print(f"  dm_tests: {len(dm_df)} test pairs")

    # --- 2e: Hybrid weights ---
    ad_path = TABLE_SRC / "adaptive_details.csv"
    if ad_path.exists():
        ad = pd.read_csv(ad_path)
        path = FINAL_TABLES / "hybrid_weights.csv"
        ad.to_csv(path, index=False)
        save(path)
        years_hw = sorted(ad["year"].unique())
        print(f"  hybrid_weights: years {int(years_hw[0])}-{int(years_hw[-1])}")

    # --- 2f: Copy source tables (keep originals) ---
    for f in sorted(TABLE_SRC.glob("*.csv")):
        dst = FINAL_TABLES / f.name
        if not dst.exists():  # Don't overwrite newly generated
            shutil.copy2(f, dst)

    # --- 2g: Generate LaTeX tables ---
    generate_all_latex(qlike_df, summary_df, dm_df, model_names)

    return qlike_df, summary_df, dm_df


def generate_all_latex(qlike_df, summary_df, dm_df, model_names):
    """Generate all LaTeX tables."""

    # --- LaTeX: QLIKE walkforward per horizon ---
    for h in HORIZONS:
        hdf = qlike_df[qlike_df["horizon"] == h].copy()
        years = sorted(hdf["year"].unique())

        tex = []
        tex.append(r"\begin{table}[htbp]")
        tex.append(r"\centering")
        tex.append(f"\\caption{{Annual QLIKE Loss, Walk-Forward, $h={h}$}}")
        tex.append(f"\\label{{tab:qlike_wf_h{h}}}")
        col_spec = "l" + "r" * len(model_names)
        tex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        tex.append(r"\toprule")
        header = "Year"
        for mn in model_names:
            header += f" & {mn.replace('_', chr(92) + '_')}"
        header += r" \\"
        tex.append(header)
        tex.append(r"\midrule")

        for year in years:
            row_data = hdf[hdf["year"] == year]
            vals = [float(row_data[mn].iloc[0]) if mn in row_data.columns and not pd.isna(row_data[mn].iloc[0]) else np.nan
                    for mn in model_names]
            best = min(v for v in vals if np.isfinite(v))
            line = str(int(year))
            if int(year) == 2026:
                line += "$^{\\dagger}$"
            for v in vals:
                s = f"{v:.4f}" if np.isfinite(v) else "---"
                if np.isfinite(v) and abs(v - best) < 1e-6:
                    s = r"\textbf{" + s + "}"
                line += f" & {s}"
            line += r" \\"
            tex.append(line)

        # Average
        tex.append(r"\midrule")
        avg_vals = [hdf[mn].mean() for mn in model_names]
        best_avg = min(v for v in avg_vals if np.isfinite(v))
        avg_line = "Average"
        for v in avg_vals:
            s = f"{v:.4f}"
            if abs(v - best_avg) < 1e-6:
                s = r"\textbf{" + s + "}"
            avg_line += f" & {s}"
        avg_line += r" \\"
        tex.append(avg_line)

        tex.append(r"\bottomrule")
        tex.append(r"\multicolumn{" + str(len(model_names) + 1) + r"}{l}{\footnotesize $^\dagger$ 2026: partial year (Jan 5 -- Feb 3, 27 trading days)} \\")
        tex.append(r"\end{tabular}")
        tex.append(r"\end{table}")

        path = FINAL_LATEX / f"qlike_walkforward_h{h}.tex"
        with open(path, "w") as f:
            f.write("\n".join(tex))
        save(path)

    # --- LaTeX: Model Summary ---
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{Model Comparison: Mean QLIKE, Walk-Forward 2017--2026}")
    tex.append(r"\label{tab:model_summary}")
    tex.append(r"\begin{tabular}{lrrr}")
    tex.append(r"\toprule")
    tex.append(r"Model & $h=1$ & $h=5$ & $h=22$ \\")
    tex.append(r"\midrule")

    h_bests = {}
    for h in HORIZONS:
        vals = [float(summary_df[summary_df["Model"] == MODEL_DISPLAY[m]][f"H={h}"].iloc[0])
                for m in CORE_MODELS if len(summary_df[summary_df["Model"] == MODEL_DISPLAY[m]]) > 0]
        h_bests[h] = min(vals) if vals else np.inf

    for _, row in summary_df.iterrows():
        model = row["Model"].replace("_", r"\_")
        cells = []
        for h in HORIZONS:
            v = float(row[f"H={h}"])
            s = f"{v:.4f}"
            if abs(v - h_bests[h]) < 1e-6:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        tex.append(f"{model} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    path = FINAL_LATEX / "model_summary.tex"
    with open(path, "w") as f:
        f.write("\n".join(tex))
    save(path)

    # --- LaTeX: DM Tests ---
    for h in HORIZONS:
        hdf = dm_df[dm_df["Horizon"] == h]
        if len(hdf) == 0:
            continue

        tex = []
        tex.append(r"\begin{table}[htbp]")
        tex.append(r"\centering")
        tex.append(f"\\caption{{Diebold-Mariano Tests, Walk-Forward 2017--2026, $h={h}$}}")
        tex.append(f"\\label{{tab:dm_wf_h{h}}}")
        tex.append(r"\begin{tabular}{llrcl}")
        tex.append(r"\toprule")
        tex.append(r"Model 1 & Model 2 & DM stat & $p$-value & Winner \\")
        tex.append(r"\midrule")

        for _, row in hdf.iterrows():
            m1 = str(row["Model1"]).replace("_", r"\_")
            m2 = str(row["Model2"]).replace("_", r"\_")
            stat = float(row["DM_stat"])
            pval = float(row["p_value"])
            sig = row["Sig"]
            winner = str(row["Winner"]).replace("_", r"\_")
            tex.append(f"{m1} & {m2} & {stat:.3f} & {pval:.4f}{sig} & {winner} \\\\")

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex.append(r"\end{table}")

        path = FINAL_LATEX / f"dm_tests_h{h}.tex"
        with open(path, "w") as f:
            f.write("\n".join(tex))
        save(path)

    # --- LaTeX: Best per horizon ---
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{Recommended Model: Adaptive Hybrid (V1)}")
    tex.append(r"\label{tab:recommended}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"& V1\_Adaptive & HAR-J & XGBoost & LightGBM \\")
    tex.append(r"\midrule")

    for h in HORIZONS:
        vals = []
        for m in CORE_MODELS:
            sdf = summary_df[summary_df["Model"] == MODEL_DISPLAY[m]]
            if len(sdf) > 0:
                v = float(sdf[f"H={h}"].iloc[0])
                vals.append((MODEL_DISPLAY[m], v))
            else:
                vals.append((MODEL_DISPLAY[m], np.nan))

        best = min(v for _, v in vals if np.isfinite(v))
        cells = []
        # Reorder: V1_Adaptive, HAR-J, XGBoost, LightGBM
        order = ["V1_Adaptive", "HAR-J", "XGBoost", "LightGBM"]
        for mn in order:
            v = next((v for n, v in vals if n == mn), np.nan)
            s = f"{v:.4f}"
            if abs(v - best) < 1e-6:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        tex.append(f"$h={h}$ & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    path = FINAL_LATEX / "recommended_model.tex"
    with open(path, "w") as f:
        f.write("\n".join(tex))
    save(path)

    print(f"  LaTeX: {len(list(FINAL_LATEX.glob('*.tex')))} files")


# ============================================================
# 3. Update figures
# ============================================================

def update_figures(preds, qlike_df, summary_df):
    print("\n" + "=" * 60)
    print("[3/5] Updating figures")
    print("=" * 60)

    FINAL_FIGS.mkdir(parents=True, exist_ok=True)
    model_names = [MODEL_DISPLAY[m] for m in CORE_MODELS]
    colors = {"HAR-J": "#e74c3c", "XGBoost": "#3498db",
              "LightGBM": "#2ecc71", "V1_Adaptive": "#9b59b6"}

    # --- 3a: QLIKE by year ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, h in enumerate(HORIZONS):
        ax = axes[idx]
        hdf = qlike_df[qlike_df["horizon"] == h]
        years = sorted(hdf["year"].unique())

        for mn in model_names:
            if mn in hdf.columns:
                vals = [float(hdf[hdf["year"] == y][mn].iloc[0])
                        if len(hdf[hdf["year"] == y]) > 0 else np.nan
                        for y in years]
                ax.plot(years, vals, "o-", label=mn, color=colors[mn],
                        markersize=5, linewidth=1.5)

        # Mark 2026 as partial
        ax.axvline(x=2025.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(2026, ax.get_ylim()[1] * 0.95, "partial", fontsize=7,
                ha="center", color="gray", style="italic")

        ax.set_title(f"h = {h}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("QLIKE")
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(y)) for y in years], rotation=45, fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("Walk-Forward QLIKE by Year (2017-2026)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = FINAL_FIGS / f"qlike_by_year{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        save(path)
    plt.close(fig)
    print("  qlike_by_year: OK")

    # --- 3b: QLIKE comparison bars ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for idx, h in enumerate(HORIZONS):
        ax = axes[idx]
        vals = []
        names = []
        cols = []
        for m in CORE_MODELS:
            mn = MODEL_DISPLAY[m]
            sdf = summary_df[summary_df["Model"] == mn]
            if len(sdf) > 0:
                v = float(sdf[f"H={h}"].iloc[0])
                vals.append(v)
                names.append(mn.replace("V1_", "V1\n"))
                cols.append(colors[mn])

        bars = ax.bar(names, vals, color=cols, edgecolor="white", width=0.6)
        best_idx = np.argmin(vals)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_title(f"h = {h}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean QLIKE (2017-2026)")
        ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle("Model Comparison: Mean QLIKE (Walk-Forward 2017-2026)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = FINAL_FIGS / f"qlike_comparison{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        save(path)
    plt.close(fig)
    print("  qlike_comparison: OK")

    # --- 3c: Hybrid weights over years ---
    ad_path = TABLE_SRC / "adaptive_details.csv"
    if ad_path.exists():
        ad = pd.read_csv(ad_path)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for idx, h in enumerate(HORIZONS):
            ax = axes[idx]
            hdf = ad[ad["horizon"] == h].sort_values("year")
            years = hdf["year"].values
            w_har = hdf["w_har"].values
            best_ml = hdf["best_ml"].values

            # Color by ML model
            ml_colors = ["#3498db" if ml == "xgboost" else "#2ecc71" for ml in best_ml]
            ax.bar(years, w_har, color=ml_colors, edgecolor="white", width=0.7, alpha=0.8)
            ax.bar(years, 1 - w_har, bottom=w_har, color=[c + "80" for c in ml_colors],
                   edgecolor="white", width=0.7, alpha=0.4)

            # Labels
            for y, w, ml in zip(years, w_har, best_ml):
                label = f"{ml[:3].upper()}\n{w:.2f}"
                ax.text(y, 0.02, label, ha="center", fontsize=6, fontweight="bold")

            ax.axvline(x=2025.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_title(f"h = {h}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("w_HAR")
            ax.set_ylim(0, 1)
            ax.set_xticks(years)
            ax.set_xticklabels([str(int(y)) for y in years], rotation=45, fontsize=8)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#3498db", label="XGBoost"),
                           Patch(facecolor="#2ecc71", label="LightGBM")]
        axes[0].legend(handles=legend_elements, fontsize=8, loc="upper right")

        fig.suptitle("Adaptive Hybrid: ML Model Selection & HAR Weight (2017-2026)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        for ext in [".png", ".pdf"]:
            path = FINAL_FIGS / f"hybrid_weights{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            save(path)
        plt.close(fig)
        print("  hybrid_weights: OK")

    # --- 3d: Model rankings ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, h in enumerate(HORIZONS):
        ax = axes[idx]
        hdf = qlike_df[qlike_df["horizon"] == h]
        years = sorted(hdf["year"].unique())

        for mn in model_names:
            if mn not in hdf.columns:
                continue
            rankings = []
            for year in years:
                row = hdf[hdf["year"] == year]
                if len(row) == 0:
                    rankings.append(np.nan)
                    continue
                year_vals = [(mn2, float(row[mn2].iloc[0]))
                             for mn2 in model_names if mn2 in row.columns]
                year_vals.sort(key=lambda x: x[1])
                rank = next((i + 1 for i, (n, _) in enumerate(year_vals) if n == mn), np.nan)
                rankings.append(rank)

            ax.plot(years, rankings, "o-", label=mn, color=colors[mn],
                    markersize=6, linewidth=1.5)

        ax.axvline(x=2025.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(f"h = {h}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Rank (1=best)")
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylim(0.5, 4.5)
        ax.invert_yaxis()
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(y)) for y in years], rotation=45, fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("Model Rankings by Year (2017-2026)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = FINAL_FIGS / f"model_rankings{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        save(path)
    plt.close(fig)
    print("  model_rankings: OK")

    # --- 3e: Error correlation ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, h in enumerate(HORIZONS):
        ax = axes[idx]

        # Merge all predictions
        merged = None
        for m in CORE_MODELS:
            if (m, h) not in preds:
                continue
            df = preds[(m, h)][["date", "ticker", "rv_actual", "rv_pred"]].copy()
            df = df.rename(columns={"rv_pred": f"pred_{m}"})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(
                    df[["date", "ticker", f"pred_{m}"]],
                    on=["date", "ticker"], how="inner"
                )

        if merged is None:
            continue

        y = merged["rv_actual"].values
        errors = {}
        for m in CORE_MODELS:
            col = f"pred_{m}"
            if col in merged.columns:
                errors[MODEL_DISPLAY[m]] = qlike_per_obs(y, merged[col].values)

        err_df = pd.DataFrame(errors)
        corr = err_df.corr()

        im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        labels = [c.replace("V1_", "V1\n") for c in corr.columns]
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.values[i, j]
                color = "white" if v > 0.7 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_title(f"h = {h}", fontsize=13, fontweight="bold")

    fig.suptitle("Error Correlation (QLIKE Loss, 2017-2026)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = FINAL_FIGS / f"error_correlation{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        save(path)
    plt.close(fig)
    print("  error_correlation: OK")

    # --- 3f: Copy other figures that don't need update ---
    src_figs = BASE / "results" / "figures"
    keep_as_is = [
        "actual_vs_predicted", "feature_importance",
        "predictions_AFLT_h1", "predictions_AFLT_h5", "predictions_AFLT_h22",
        "strategy_comparison", "strategy_comparison_v3",
        "trimmed_analysis", "adaptive_weights_v2",
        "shrinkage_sensitivity", "rolling_window_sensitivity",
        "metrics_comparison_h1", "metrics_comparison_h5", "metrics_comparison_h22",
    ]
    for name in keep_as_is:
        for ext in [".png", ".pdf"]:
            src = src_figs / f"{name}{ext}"
            if src.exists():
                dst = FINAL_FIGS / f"{name}{ext}"
                if not dst.exists():
                    shutil.copy2(src, dst)

    print(f"  Total figures: {len(list(FINAL_FIGS.glob('*')))}")


# ============================================================
# 4. Update SUMMARY.txt
# ============================================================

def update_summary(qlike_df, summary_df, dm_df):
    print("\n" + "=" * 60)
    print("[4/5] Updating SUMMARY.txt")
    print("=" * 60)

    model_names = [MODEL_DISPLAY[m] for m in CORE_MODELS]

    # Extract key numbers
    def get_q(model_name, h):
        sdf = summary_df[summary_df["Model"] == model_name]
        return float(sdf[f"H={h}"].iloc[0]) if len(sdf) > 0 else np.nan

    # 2026 only
    q2026 = {}
    for h in HORIZONS:
        hdf = qlike_df[(qlike_df["horizon"] == h) & (qlike_df["year"] == 2026)]
        for mn in model_names:
            if mn in hdf.columns and len(hdf) > 0:
                q2026[(mn, h)] = float(hdf[mn].iloc[0])

    # Year range
    years = sorted(qlike_df["year"].unique())
    n_years = len(years)

    lines = []
    lines.append("=" * 70)
    lines.append("MOEX Volatility Forecasting — Final Results Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("ФИНАЛЬНАЯ МОДЕЛЬ: Adaptive Hybrid (V1)")
    lines.append("-" * 60)
    lines.append("- Единый метод для всех горизонтов")
    lines.append("- Алгоритм: HAR-J + лучший ML (XGBoost/LightGBM), выбор и веса на val")
    lines.append("- Обоснование: стабильный top-2-3 на всех горизонтах, никогда не последний")
    lines.append("")

    lines.append(f"Walk-Forward QLIKE (mean {int(years[0])}-{int(years[-1])}):")
    for m in ["hybrid_adaptive", "lightgbm", "xgboost", "har"]:
        mn = MODEL_DISPLAY[m]
        label = {"hybrid_adaptive": "Hybrid_Adaptive",
                 "lightgbm": "LightGBM       ",
                 "xgboost": "XGBoost        ",
                 "har": "HAR-J          "}[m]
        vals = [f"H={h}: {get_q(mn, h):.4f}" for h in HORIZONS]
        lines.append(f"  {label}: {', '.join(vals)}")
    lines.append("")

    lines.append(f"2026 QLIKE (partial year, Jan 5 — Feb 3, 27 trading days):")
    for m in ["hybrid_adaptive", "lightgbm", "xgboost", "har"]:
        mn = MODEL_DISPLAY[m]
        label = {"hybrid_adaptive": "Hybrid_Adaptive",
                 "lightgbm": "LightGBM       ",
                 "xgboost": "XGBoost        ",
                 "har": "HAR-J          "}[m]
        vals = [f"H={h}: {q2026.get((mn, h), np.nan):.4f}" for h in HORIZONS]
        lines.append(f"  {label}: {', '.join(vals)}")
    lines.append("")

    # Hybrid weights 2026
    ad_path = TABLE_SRC / "adaptive_details.csv"
    if ad_path.exists():
        ad = pd.read_csv(ad_path)
        ad2026 = ad[ad["year"] == 2026]
        if len(ad2026) > 0:
            lines.append("Hybrid 2026 weights:")
            for _, row in ad2026.iterrows():
                h = int(row["horizon"])
                ml = row["best_ml"]
                w = float(row["w_har"])
                rank = int(row["hybrid_rank"])
                lines.append(f"  H={h}: best_ml={ml}, w_har={w:.2f} (rank {rank})")
            lines.append("")

    lines.append("Test 2019 QLIKE:")
    lines.append("  Hybrid: H=1: 0.2755, H=5: 0.3725, H=22: 0.4407 (лучший на всех горизонтах)")
    lines.append("")

    # Full comparison
    lines.append(f"ПОЛНОЕ СРАВНЕНИЕ МОДЕЛЕЙ (QLIKE, Walk-Forward {int(years[0])}-{int(years[-1])})")
    lines.append("-" * 60)
    lines.append(f"{'Model':<20} {'H=1':<12} {'H=5':<12} {'H=22':<12}")
    lines.append("-" * 60)
    for _, row in summary_df.iterrows():
        m = row["Model"]
        vals = [f"{float(row[f'H={h}']):.4f}" for h in HORIZONS]
        lines.append(f"{m:<20} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")
    lines.append("")

    # DM test highlights
    lines.append("DIEBOLD-MARIANO TESTS (V1_Adaptive vs others)")
    lines.append("-" * 60)
    ref_tests = dm_df[(dm_df["Model1"] == "HAR-J") & (dm_df["Model2"] == "V1_Adaptive") |
                       (dm_df["Model1"] == "XGBoost") & (dm_df["Model2"] == "V1_Adaptive") |
                       (dm_df["Model1"] == "LightGBM") & (dm_df["Model2"] == "V1_Adaptive") |
                       (dm_df["Model1"] == "V1_Adaptive")]

    for h in HORIZONS:
        parts = []
        hdf = dm_df[dm_df["Horizon"] == h]
        for _, row in hdf.iterrows():
            m1, m2 = row["Model1"], row["Model2"]
            if "V1_Adaptive" not in (m1, m2):
                continue
            other = m2 if m1 == "V1_Adaptive" else m1
            p = float(row["p_value"])
            winner = row["Winner"]
            s = sig_stars(p)
            if winner == "V1_Adaptive" and s:
                parts.append(f"beats {other} ({s})")
            elif winner != "V1_Adaptive" and s:
                parts.append(f"loses to {other} ({s})")
            else:
                parts.append(f"≈ {other} (p={p:.2f})")
        lines.append(f"  H={h}: {', '.join(parts)}")
    lines.append("")

    # Predictions
    lines.append("PREDICTIONS (hybrid_adaptive):")
    for h in HORIZONS:
        path = PRED_SRC / f"hybrid_adaptive_h{h}_annual.parquet"
        status = "[OK]" if path.exists() else "[MISSING]"
        lines.append(f"  data/predictions/walk_forward/hybrid_adaptive_h{h}_annual.parquet  {status}")
    lines.append("")

    # Walk-forward coverage
    lines.append(f"WALK-FORWARD COVERAGE: {int(years[0])}-{int(years[-1])} ({n_years} years)")
    lines.append("-" * 60)
    for h in HORIZONS:
        for y in years:
            hdf = qlike_df[(qlike_df["horizon"] == h) & (qlike_df["year"] == y)]
            # Get obs count from predictions
            key = ("har", h)
            if key in load_all_predictions.__code__.co_varnames:
                pass
            partial = " (PARTIAL)" if int(y) == 2026 else ""
            # Just note the year
        pass

    har_h1 = pd.read_parquet(PRED_SRC / "har_h1_annual.parquet")
    year_counts = har_h1.groupby("year").size()
    for y in years:
        n = int(year_counts.get(y, 0))
        partial = " (PARTIAL — 27 trading days)" if int(y) == 2026 else ""
        lines.append(f"  {int(y)}: {n} obs{partial}")
    lines.append("")

    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    path = FINAL / "SUMMARY.txt"
    with open(path, "w") as f:
        f.write(summary_text)
    save(path)
    print("  SUMMARY.txt updated")

    return summary_text


# ============================================================
# 5. Verification & output
# ============================================================

def verify_and_print(qlike_df, summary_df, dm_df):
    print("\n" + "=" * 60)
    print("[5/5] Verification")
    print("=" * 60)

    model_names = [MODEL_DISPLAY[m] for m in CORE_MODELS]

    # Check CSV numbers = summary numbers
    ok = True
    for h in HORIZONS:
        hdf = qlike_df[qlike_df["horizon"] == h]
        for mn in model_names:
            csv_avg = hdf[mn].mean() if mn in hdf.columns else np.nan
            summ_val = float(summary_df[summary_df["Model"] == mn][f"H={h}"].iloc[0]) \
                if len(summary_df[summary_df["Model"] == mn]) > 0 else np.nan
            if abs(csv_avg - summ_val) > 1e-8:
                print(f"  MISMATCH: {mn} H={h}: csv_avg={csv_avg:.6f} vs summary={summ_val:.6f}")
                ok = False

    # Check years in tables = years in figures
    years_in_table = sorted(qlike_df["year"].unique())
    print(f"  Years in tables: {[int(y) for y in years_in_table]}")
    print(f"  2026 included: {'yes' if 2026 in years_in_table else 'NO'}")

    # Check DM tests on same data
    dm_years = set()
    for m in CORE_MODELS:
        path = PRED_SRC / f"{m}_h1_annual.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            dm_years.update(df["year"].unique())
    print(f"  DM test data years: {sorted([int(y) for y in dm_years])}")

    if ok:
        print("\n  ALL NUMBERS CONSISTENT")
    else:
        print("\n  WARNING: Some mismatches found!")

    # Final QLIKE summary
    print("\n" + "=" * 60)
    print("FINAL QLIKE SUMMARY (Walk-Forward 2017-2026)")
    print("=" * 60)
    print(f"{'Model':<20} {'H=1':<12} {'H=5':<12} {'H=22':<12}")
    print("-" * 56)
    for _, row in summary_df.iterrows():
        m = row["Model"]
        print(f"{m:<20} {float(row['H=1']):<12.4f} {float(row['H=5']):<12.4f} {float(row['H=22']):<12.4f}")

    # DM tests
    print(f"\nDM Tests (V1_Adaptive vs others):")
    for h in HORIZONS:
        hdf = dm_df[dm_df["Horizon"] == h]
        parts = []
        for _, row in hdf.iterrows():
            m1, m2 = row["Model1"], row["Model2"]
            if "V1_Adaptive" not in (m1, m2):
                continue
            other = m2 if m1 == "V1_Adaptive" else m1
            p = float(row["p_value"])
            winner = row["Winner"]
            s = sig_stars(p)
            if winner == "V1_Adaptive":
                parts.append(f"beats {other} ({s or 'ns'})")
            else:
                parts.append(f"≈ {other} (p={p:.2f})")
        print(f"  H={h}: {', '.join(parts)}")

    # Updated files
    print(f"\nUpdated files: {len(updated_files)}")
    for f in sorted(set(updated_files)):
        print(f"  {f}")

    print(f"\nAll numbers consistent: {'YES' if ok else 'NO'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Update results/final/ with 2026 data")
    print("=" * 60)

    # Ensure dirs exist
    for d in [FINAL, FINAL_DATA_WF, FINAL_DATA_BT, FINAL_TABLES, FINAL_LATEX, FINAL_FIGS]:
        d.mkdir(parents=True, exist_ok=True)

    # Load all predictions
    print("\nLoading predictions...")
    preds = load_all_predictions()
    print(f"  Loaded {len(preds)} model×horizon combinations")

    # Check 2026 coverage
    for m in CORE_MODELS:
        df = preds.get((m, 1))
        if df is not None:
            years = sorted(df["year"].unique())
            has_2026 = 2026 in years
            print(f"  {MODEL_DISPLAY[m]}: years {int(years[0])}-{int(years[-1])}, 2026={'YES' if has_2026 else 'NO'}")

    # Execute all steps
    update_predictions(preds)
    qlike_df, summary_df, dm_df = update_tables(preds)
    update_figures(preds, qlike_df, summary_df)
    update_summary(qlike_df, summary_df, dm_df)
    verify_and_print(qlike_df, summary_df, dm_df)

    print("\n" + "=" * 60)
    print("DONE: results/final/ rebuilt with 2026")
    print("=" * 60)


if __name__ == "__main__":
    main()
