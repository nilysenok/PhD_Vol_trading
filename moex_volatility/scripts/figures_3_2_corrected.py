#!/usr/bin/env python3
"""
Section 3.2 Figures — corrected data, 7 figures for dissertation.

Usage: python3 scripts/figures_3_2_corrected.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "figures_3_2_corrected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WF_DIR = BASE / "results" / "final" / "data" / "predictions_walkforward"

# ── Style ──────────────────────────────────────────────────────────────
def setup_rc():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.unicode_minus": False,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
    })


def hgrid(ax):
    ax.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _darker(hex_color, factor=0.6):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def fmt(v, decimals=3):
    """Format number with comma as decimal separator."""
    return f"{v:.{decimals}f}".replace(".", ",")


# ── Colours ────────────────────────────────────────────────────────────
C_HYBRID  = "#2E7D32"   # dark green — best
C_LGB     = "#43A047"   # green
C_XGB     = "#66BB6A"   # light green
C_HAR     = "#1565C0"   # blue (classical)
C_LSTM    = "#E65100"   # dark orange
C_GRU     = "#EF6C00"   # orange
C_GARCH   = "#B71C1C"   # dark red — worst
C_GREY    = "#757575"


# ══════════════════════════════════════════════════════════════════════
#  Figure 3.5 — All 7 models × 3 horizons, 3D bars (test 2019)
# ══════════════════════════════════════════════════════════════════════
def fig_3_5():
    models = ["Гибридная", "LightGBM", "XGBoost", "HAR-J", "LSTM", "GRU", "GJR-GARCH"]
    colors = [C_HYBRID, C_LGB, C_XGB, C_HAR, C_LSTM, C_GRU, C_GARCH]
    horizons = ["H=1", "H=5", "H=22"]

    # Test 2019 data (from user's corrected table)
    data = {
        "H=1":  [0.276, 0.296, 0.277, 0.305, 0.372, 0.399, 0.501],
        "H=5":  [0.373, None,  None,  0.424, None,  None,  0.567],
        "H=22": [0.441, None,  None,  0.467, None,  None,  0.601],
    }

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Pane styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
    ax.zaxis.gridlines.set_alpha(0.25)
    ax.xaxis.gridlines.set_alpha(0.1)
    ax.yaxis.gridlines.set_alpha(0.1)

    n_models = len(models)
    n_horiz = len(horizons)
    dx, dy = 0.6, 0.35

    xs = np.arange(n_horiz, dtype=float)
    ys = np.arange(n_models, dtype=float)

    for j in reversed(range(n_models)):
        col = colors[j]
        edge = _darker(col, 0.5)
        for i, h_key in enumerate(horizons):
            val = data[h_key][j]
            if val is None:
                continue
            ax.bar3d(xs[i] - dx/2, ys[j] - dy/2, 0,
                     dx, dy, val,
                     color=col, edgecolor=edge, lw=0.5, alpha=0.92)
            ax.text(xs[i], ys[j], val + 0.015,
                    fmt(val), ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="black", zorder=100)

    ax.set_xticks(xs)
    ax.set_xticklabels(horizons, fontsize=10)
    ax.set_xlabel("Горизонт прогнозирования", labelpad=10)
    ax.set_yticks(ys)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_ylabel("Модель", labelpad=10)
    ax.set_zlabel("")
    ax.set_zlim(0, 0.75)

    ax.view_init(elev=25, azim=-55)
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='z', labelsize=9)

    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.0, top=0.88)
    fig.text(0.92, 0.45, "QLIKE", rotation=90, va="center", ha="center", fontsize=11)

    fig.text(0.5, 0.92,
             "Рисунок 3.5 — Сравнение QLIKE моделей\nпо горизонтам прогнозирования",
             ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.02,
             "Тестовый период 2019 (N=3 910–4 284)",
             ha="center", fontsize=9, color="0.4")

    fig.savefig(OUT_DIR / "fig_3_5_all_models_3d.png")
    plt.close(fig)
    print("  fig_3_5 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3.6 — Boosting vs HAR-J (test 2019, H=1)
# ══════════════════════════════════════════════════════════════════════
def fig_3_6():
    models = ["XGBoost", "LightGBM", "HAR-J"]
    values = [0.277, 0.296, 0.305]
    colors = [C_XGB, C_LGB, C_HAR]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, values, width=0.55, color=colors,
                  edgecolor=[_darker(c) for c in colors], lw=0.7, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
                fmt(val), ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("QLIKE")
    ax.set_ylim(0, max(values) * 1.18)
    hgrid(ax)

    fig.text(0.5, 0.95,
             "Рисунок 3.6 — QLIKE моделей бустинга и HAR-J\n(тест 2019, H=1)",
             ha="center", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(OUT_DIR / "fig_3_6_boosting_vs_har.png")
    plt.close(fig)
    print("  fig_3_6 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3.7 — RNN vs Boosting vs HAR-J (test 2019, H=1)
# ══════════════════════════════════════════════════════════════════════
def fig_3_7():
    models = ["XGBoost", "LightGBM", "HAR-J", "LSTM", "GRU"]
    values = [0.277, 0.296, 0.305, 0.372, 0.399]
    colors = [C_XGB, C_LGB, C_HAR, C_LSTM, C_GRU]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    bars = ax.bar(x, values, width=0.55, color=colors,
                  edgecolor=[_darker(c) for c in colors], lw=0.7, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                fmt(val), ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Annotation: % worse than XGBoost for LSTM, GRU
    for idx, (model, val) in enumerate(zip(models, values)):
        if model in ("LSTM", "GRU"):
            pct = (val / 0.277 - 1) * 100
            ax.text(x[idx], val + 0.022, f"+{pct:.0f}%",
                    ha="center", va="bottom", fontsize=9, color="0.35",
                    style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("QLIKE")
    ax.set_ylim(0, max(values) * 1.22)
    hgrid(ax)

    fig.text(0.5, 0.95,
             "Рисунок 3.7 — QLIKE рекуррентных сетей, бустинга и HAR-J\n(тест 2019, H=1)",
             ha="center", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(OUT_DIR / "fig_3_7_rnn_vs_boosting.png")
    plt.close(fig)
    print("  fig_3_7 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3.8 — Average QLIKE by model class (test 2019, H=1)
# ══════════════════════════════════════════════════════════════════════
def fig_3_8():
    classes = ["Бустинг", "HAR-J", "RNN", "GJR-GARCH"]
    values = [
        np.mean([0.277, 0.296]),  # (XGB + LGB) / 2
        0.305,
        np.mean([0.372, 0.399]),  # (LSTM + GRU) / 2
        0.501,
    ]
    colors = [C_XGB, C_HAR, C_LSTM, C_GARCH]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(classes))
    bars = ax.bar(x, values, width=0.55, color=colors,
                  edgecolor=[_darker(c) for c in colors], lw=0.7, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                fmt(val), ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("QLIKE")
    ax.set_ylim(0, max(values) * 1.18)
    hgrid(ax)

    fig.text(0.5, 0.95,
             "Рисунок 3.8 — Средний QLIKE по категориям моделей\n(тест 2019, H=1)",
             ha="center", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(OUT_DIR / "fig_3_8_model_classes.png")
    plt.close(fig)
    print("  fig_3_8 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3.9 — Walk-forward QLIKE by horizon (corrected data)
# ══════════════════════════════════════════════════════════════════════
def fig_3_9():
    horizons = ["H=1", "H=5", "H=22"]
    models = ["HAR-J", "XGBoost", "LightGBM", "Гибрид"]
    colors_m = [C_HAR, C_XGB, C_LGB, C_HYBRID]

    data = {
        "HAR-J":    [0.524, 0.796, 0.961],
        "XGBoost":  [0.400, 0.956, 1.127],
        "LightGBM": [0.378, 0.821, 0.870],
        "Гибрид":   [0.380, 0.762, 0.888],
    }

    n_h = len(horizons)
    n_m = len(models)
    x = np.arange(n_h)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model, col) in enumerate(zip(models, colors_m)):
        vals = data[model]
        bars = ax.bar(x + offsets[i], vals, width,
                      color=col, edgecolor=_darker(col), lw=0.6,
                      label=model, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                    fmt(val), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=12)
    ax.set_xlabel("Горизонт прогнозирования")
    ax.set_ylabel("QLIKE (среднее по годам)")
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    hgrid(ax)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, fontsize=10)

    fig.text(0.5, 0.96,
             "Рисунок 3.9 — Walk-forward QLIKE по горизонтам прогнозирования\n(среднее 2017–2025, 17 тикеров)",
             ha="center", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(OUT_DIR / "fig_3_9_walkforward_qlike.png")
    plt.close(fig)
    print("  fig_3_9 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3.10 — DM-test heatmap
# ══════════════════════════════════════════════════════════════════════
def fig_3_10():
    pairs = [
        "HAR-J vs Гибрид",
        "HAR-J vs LightGBM",
        "HAR-J vs XGBoost",
    ]
    horizons = ["H=1", "H=5", "H=22"]

    t_stats = np.array([
        [13.55,   3.41,   7.03],   # HAR-J vs Hybrid
        [12.11,  -1.48,   4.61],   # HAR-J vs LightGBM
        [11.95,  -5.49,  -4.63],   # HAR-J vs XGBoost
    ])

    p_vals = np.array([
        [0.0001, 0.0001, 0.0001],
        [0.0001, 0.139,  0.0001],
        [0.0001, 0.0001, 0.0001],
    ])

    fig, ax = plt.subplots(figsize=(9, 5))

    # Color mapping: green = ML better (t>0), red = HAR better (t<0), grey = insignificant
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dm",
        [(0.0, "#C62828"),     # strong red (HAR-J better)
         (0.35, "#FFCDD2"),    # light red
         (0.45, "#E0E0E0"),    # grey (neutral)
         (0.55, "#E0E0E0"),    # grey
         (0.65, "#C8E6C9"),    # light green
         (1.0, "#2E7D32")],    # strong green (ML better)
    )

    vmax = 16
    im = ax.imshow(t_stats, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(horizons)))
    ax.set_yticks(range(len(pairs)))
    ax.set_xticklabels(horizons, fontsize=11)
    ax.set_yticklabels(pairs, fontsize=11)

    for i in range(len(pairs)):
        for j in range(len(horizons)):
            t = t_stats[i, j]
            p = p_vals[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            # Text color: white on dark backgrounds
            text_color = "white" if abs(t) > 8 else "black"
            ax.text(j, i, f"{fmt(t, 2)}{sig}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("t-статистика DM-теста", fontsize=10)

    # Custom annotations
    ax.text(len(horizons) + 0.85, -0.6, "ML лучше →",
            fontsize=8, color=C_HYBRID, ha="center", fontweight="bold")
    ax.text(len(horizons) + 0.85, len(pairs) - 0.4, "← HAR-J лучше",
            fontsize=8, color="#C62828", ha="center", fontweight="bold")

    ax.spines[:].set_visible(True)
    ax.spines[:].set_color("0.7")

    fig.text(0.45, 0.96,
             "Рисунок 3.10 — DM-тест: HAR-J против ML-моделей\n(walk-forward 2017–2025)",
             ha="center", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(OUT_DIR / "fig_3_10_dm_test_heatmap.png")
    plt.close(fig)
    print("  fig_3_10 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3.11 — Hybrid QLIKE by year (walk-forward, line chart)
# ══════════════════════════════════════════════════════════════════════
def fig_3_11():
    """Load walk-forward predictions and compute hybrid QLIKE by year."""
    def qlike(y_true, y_pred, eps=1e-10):
        y_pred = np.maximum(y_pred, eps)
        ratio = y_true / y_pred
        return np.mean(ratio - np.log(ratio) - 1)

    years_all = list(range(2017, 2026))
    results = {}

    for h in [1, 5, 22]:
        path = WF_DIR / f"walkforward_all_h{h}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping fig_3_11")
            return
        df = pd.read_parquet(path)
        df = df[df['rv_actual'] > 0]

        yearly = []
        for y in years_all:
            sub = df[df['year'] == y]
            if len(sub) > 0:
                yearly.append(qlike(sub['rv_actual'].values,
                                    sub['pred_V1_Adaptive'].values))
            else:
                yearly.append(np.nan)
        results[h] = yearly

    fig, ax = plt.subplots(figsize=(12, 6))

    line_styles = {
        1:  (C_HYBRID,  "-",  "o",  2.2, "H=1"),
        5:  (C_HAR,     "--", "s",  2.0, "H=5"),
        22: (C_GARCH,   "-.", "D",  2.0, "H=22"),
    }

    for h, (color, ls, marker, lw, label) in line_styles.items():
        vals = results[h]
        ax.plot(years_all, vals, color=color, ls=ls, lw=lw,
                marker=marker, markersize=7, markeredgecolor=_darker(color),
                markerfacecolor=color, label=label, zorder=3)
        for yr, val in zip(years_all, vals):
            if not np.isnan(val):
                ax.text(yr, val + 0.03, fmt(val),
                        ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(years_all)
    ax.set_xticklabels([str(y) for y in years_all], fontsize=10)
    ax.set_xlabel("Год (тестовый)")
    ax.set_ylabel("QLIKE")
    hgrid(ax)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, fontsize=10)

    fig.text(0.5, 0.96,
             "Рисунок 3.11 — QLIKE гибридной модели по годам\n(walk-forward, 17 тикеров)",
             ha="center", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(OUT_DIR / "fig_3_11_hybrid_by_year.png")
    plt.close(fig)
    print("  fig_3_11 saved")


# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    print("Generating Section 3.2 figures (corrected) ...")
    fig_3_5()
    fig_3_6()
    fig_3_7()
    fig_3_8()
    fig_3_9()
    fig_3_10()
    fig_3_11()
    print(f"\nAll saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
