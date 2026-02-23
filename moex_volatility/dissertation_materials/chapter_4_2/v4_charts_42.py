#!/usr/bin/env python3
"""
Section 4.2 Charts — 4 academic figures for dissertation.

Generates:
  fig_4_6  — Grouped bar: EW portfolio Net Sharpe by strategy × approach
  fig_4_7  — Horizontal bars: Exposure & Turnover by approach (1×2)
  fig_4_8  — 3D bar: Δ% Sharpe gain by category × approach
  fig_4_9  — A vs D on daily & hourly timeframes (1×2)

Usage: python3 scripts/v4_charts_42.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
FIG_DIR = BASE / "output_4_2_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
def setup_rc():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.unicode_minus": False,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
    })


def hgrid(ax):
    ax.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ── Colour palette ─────────────────────────────────────────────────────
COL_A = "#808080"   # grey
COL_B = "#4472C4"   # blue
COL_C = "#ED7D31"   # orange
COL_D = "#548235"   # green
COLORS_ABCD = [COL_A, COL_B, COL_C, COL_D]

LABELS = [
    "A (без прогнозов)",
    "B (адаптивные стопы)",
    "C (режимная фильтрация)",
    "D (таргетирование волатильности)",
]


def _darker(hex_color, factor=0.65):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.6 — Grouped bar: EW Net Sharpe by strategy × approach
# ══════════════════════════════════════════════════════════════════════
def fig_4_6():
    strategies = [
        "S1 Mean\nReversion", "S2 Bollinger\nBands", "S3 Donchian\nChannel",
        "S4 Super-\ntrend", "S5 Pivot\nPoints", "S6 VWAP",
    ]
    data = {
        "A": [1.682, 1.937, 0.855, 1.075, 1.218, 1.728],
        "B": [1.821, 1.818, 1.469, 1.507, 1.584, 1.895],
        "C": [2.049, 2.614, 0.899, 1.088, 1.411, 1.988],
        "D": [2.431, 2.450, 1.530, 2.125, 1.352, 2.407],
    }

    n_strat = len(strategies)
    n_app = 4
    x = np.arange(n_strat)
    width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (app, col, lbl) in enumerate(zip(
            ["A", "B", "C", "D"], COLORS_ABCD, LABELS)):
        bars = ax.bar(x + offsets[i], data[app], width,
                      color=col, edgecolor=_darker(col, 0.6),
                      linewidth=0.5, label=lbl, zorder=3)

    ax.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("Коэффициент Шарпа (net)")
    hgrid(ax)

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, fontsize=8.5)

    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_4_6_net_sharpe_comparison.png")
    plt.close(fig)
    print("  fig_4_6 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.7 — Exposure & Turnover by approach (horizontal bars)
# ══════════════════════════════════════════════════════════════════════
def fig_4_7():
    approaches = ["A", "B", "C", "D"]
    exposure   = [15.8, 4.1, 7.3, 12.2]
    turnover   = [2.3, 2.2, 4.5, 2.1]
    colors     = COLORS_ABCD

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # --- Exposure ---
    y = np.arange(len(approaches))
    bars1 = ax1.barh(y, exposure, height=0.55, color=colors,
                     edgecolor=[_darker(c, 0.6) for c in colors], lw=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(approaches, fontsize=10)
    ax1.set_xlabel("Средняя экспозиция, %")
    ax1.invert_yaxis()
    ax1.xaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, exposure):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left", fontsize=9,
                 fontweight="bold")
    ax1.set_xlim(0, max(exposure) * 1.25)

    # --- Turnover ---
    bars2 = ax2.barh(y, turnover, height=0.55, color=colors,
                     edgecolor=[_darker(c, 0.6) for c in colors], lw=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(approaches, fontsize=10)
    ax2.set_xlabel("Средняя оборачиваемость, сделок/год")
    ax2.invert_yaxis()
    ax2.xaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars2, turnover):
        ax2.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", ha="left", fontsize=9,
                 fontweight="bold")
    ax2.set_xlim(0, max(turnover) * 1.25)

    fig.tight_layout(w_pad=3)
    fig.savefig(FIG_DIR / "fig_4_7_exposure_turnover.png")
    plt.close(fig)
    print("  fig_4_7 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.8 — 3D bar: Δ% Sharpe gain by category × approach
# ══════════════════════════════════════════════════════════════════════
def _setup_3d_ax(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
    ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
    ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
    ax.zaxis.gridlines.set_alpha(0.25)
    ax.xaxis.gridlines.set_alpha(0.1)
    ax.yaxis.gridlines.set_alpha(0.1)


def fig_4_8():
    categories = ["Контртренд\n(S1, S2)", "Тренд\n(S3, S4)", "Диапазон\n(S5, S6)"]
    approaches = ["B", "C", "D"]
    app_colors = [COL_B, COL_C, COL_D]

    # Δ% data: rows = categories, cols = B, C, D
    delta = np.array([
        [0.5,  28.8, 34.8],   # Контртренд
        [54.1,  3.0, 89.3],   # Тренд
        [18.1, 15.4, 27.6],   # Диапазон
    ])

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    _setup_3d_ax(ax)

    dx, dy = 0.55, 0.45
    n_cat = len(categories)
    n_app = len(approaches)

    xs = np.arange(n_cat, dtype=float)
    ys = np.arange(n_app, dtype=float)

    zmax = delta.max()

    # Draw back-to-front for correct occlusion (D first, then C, then B)
    for j in reversed(range(n_app)):
        col = app_colors[j]
        edge = _darker(col, 0.5)
        for i in range(n_cat):
            val = delta[i, j]
            ax.bar3d(xs[i] - dx / 2, ys[j] - dy / 2, 0,
                     dx, dy, val,
                     color=col, edgecolor=edge, lw=0.6, alpha=0.92)
            # Value label above bar
            ax.text(xs[i], ys[j], val + zmax * 0.03,
                    f"+{val:.1f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="black", zorder=100)

    # Zero plane
    xx, yy = np.meshgrid(
        [xs[0] - 0.5, xs[-1] + 0.5],
        [ys[0] - 0.5, ys[-1] + 0.5],
    )
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.04, color="grey")

    ax.set_xticks(xs)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(approaches, fontsize=9.5, fontweight="bold")
    ax.set_zlabel("")
    ax.set_zlim(0, zmax * 1.35)

    ax.view_init(elev=30, azim=-55)
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='z', labelsize=8)
    ax.tick_params(axis='x', pad=3)
    ax.tick_params(axis='y', pad=2)

    # Z-axis label on the right side
    fig.subplots_adjust(left=0.0, right=0.87, bottom=0.0, top=1.0)
    fig.text(0.92, 0.50, "Δ% прироста Шарпа vs A", rotation=90,
             va="center", ha="center", fontsize=10)

    # Legend
    handles = [mpatches.Patch(facecolor=c, label=f"Подход {a}")
               for a, c in zip(approaches, app_colors)]
    ax.legend(handles=handles, frameon=True, fancybox=False,
              edgecolor="0.8", loc="upper left",
              bbox_to_anchor=(0.0, 0.95), fontsize=9)

    fig.savefig(FIG_DIR / "fig_4_8_category_3d.png")
    plt.close(fig)
    print("  fig_4_8 saved")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.9 — A vs D: daily & hourly (1×2 subplots)
# ══════════════════════════════════════════════════════════════════════
def fig_4_9():
    strategies = ["S1", "S2", "S3", "S4", "S5", "S6"]
    strat_full = [
        "S1 Mean\nReversion", "S2 Bollinger\nBands", "S3 Donchian\nChannel",
        "S4 Super-\ntrend", "S5 Pivot\nPoints", "S6 VWAP",
    ]

    daily_A = [0.309, 0.354, 0.381, 0.351, 0.178, 0.363]
    daily_D = [0.490, 0.427, 0.702, 0.584, 0.221, 0.499]

    hourly_A = [-0.459, -0.397, -0.179, -0.155, -0.134, -1.028]
    hourly_D = [0.105, 0.086, 0.099, 0.166, 0.219, 0.046]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    n = len(strategies)
    x = np.arange(n)
    width = 0.35

    # ─── Subplot 1: Daily ───
    ax1.bar(x - width / 2, daily_A, width, color=COL_A,
            edgecolor=_darker(COL_A, 0.6), lw=0.5, label="A (без прогнозов)",
            zorder=3)
    ax1.bar(x + width / 2, daily_D, width, color=COL_D,
            edgecolor=_darker(COL_D, 0.6), lw=0.5,
            label="D (таргетирование волатильности)", zorder=3)

    ax1.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strat_full, fontsize=8)
    ax1.set_ylabel("Коэффициент Шарпа (net)")
    ax1.set_title("Дневной таймфрейм (net 0,40%)", fontsize=10, pad=8)
    hgrid(ax1)
    ax1.legend(loc="upper right", frameon=True, fancybox=False,
               edgecolor="0.8", fontsize=8)
    ax1.set_ylim(0, max(daily_D) * 1.15)

    # ─── Subplot 2: Hourly ───
    ax2.bar(x - width / 2, hourly_A, width, color=COL_A,
            edgecolor=_darker(COL_A, 0.6), lw=0.5, label="A (без прогнозов)",
            zorder=3)
    ax2.bar(x + width / 2, hourly_D, width, color=COL_D,
            edgecolor=_darker(COL_D, 0.6), lw=0.5,
            label="D (таргетирование волатильности)", zorder=3)

    ax2.axhline(0, color="black", lw=0.8, ls="-", zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strat_full, fontsize=8)
    ax2.set_ylabel("Коэффициент Шарпа (net)")
    ax2.set_title("Часовой таймфрейм (net 0,35%)", fontsize=10, pad=8)
    hgrid(ax2)
    ax2.legend(loc="lower right", frameon=True, fancybox=False,
               edgecolor="0.8", fontsize=8)

    # Symmetric y-limits for hourly to emphasize the zero crossing
    yabs = max(abs(min(hourly_A)), max(hourly_D)) * 1.2
    ax2.set_ylim(-yabs, yabs)

    fig.tight_layout(w_pad=3)
    fig.savefig(FIG_DIR / "fig_4_9_daily_vs_hourly.png")
    plt.close(fig)
    print("  fig_4_9 saved")


# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    print("Generating Section 4.2 figures ...")
    fig_4_6()
    fig_4_7()
    fig_4_8()
    fig_4_9()
    print(f"\nAll saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
