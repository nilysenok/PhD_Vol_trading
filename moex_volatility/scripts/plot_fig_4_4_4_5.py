#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Рис. 4.4 — Шарп равновзвешенных портфелей по стратегиям и подходам (ком. 0,05%)
Рис. 4.5 — Сравнение подходов A и лучшего BCD на дневном и часовом таймфрейме

Usage: python3 scripts/plot_fig_4_4_4_5.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
FIG_DIR = BASE / "dissertation_materials" / "chapter_4_2"
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

# Best BCD colour — dark teal
COL_BEST = "#1B5E20"

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


# ── Data (from v4_portfolios_{daily,hourly}.csv, EW, net_0.05/net_0.04) ──

STRATEGIES = [
    "S1 Mean\nReversion", "S2 Bollinger\nBands", "S3 Donchian\nChannel",
    "S4 Super-\ntrend", "S5 Pivot\nPoints", "S6 VWAP",
]

# Daily EW net_0.05
DAILY = {
    "A": [2.556, 2.541, 1.051, 1.262, 1.662, 2.729],
    "B": [2.682, 2.521, 2.156, 2.045, 1.958, 2.902],
    "C": [3.221, 3.500, 1.649, 1.885, 1.960, 3.074],
    "D": [3.213, 3.042, 1.843, 2.448, 1.666, 3.303],
}

# Hourly EW net_0.04
HOURLY = {
    "A": [0.305, 1.791, 0.741, 1.268, 1.553, 0.477],
    "B": [1.902, 2.099, 2.832, 2.204, 2.238, 0.046],
    "C": [0.285, 1.792, 0.893, 1.471, 2.037, 1.086],
    "D": [2.212, 2.622, 1.568, 2.030, 2.409, 1.173],
}


def _best_bcd(data):
    """For each strategy, pick max(B, C, D)."""
    n = len(data["B"])
    best_vals = []
    best_apps = []
    for i in range(n):
        candidates = [(data[a][i], a) for a in ("B", "C", "D")]
        val, app = max(candidates, key=lambda x: x[0])
        best_vals.append(val)
        best_apps.append(app)
    return best_vals, best_apps


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.4 — Grouped bar: EW Net Sharpe by strategy × approach
# ══════════════════════════════════════════════════════════════════════
def fig_4_4():
    n_strat = len(STRATEGIES)
    x = np.arange(n_strat)
    width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (app, col, lbl) in enumerate(zip(
            ["A", "B", "C", "D"], COLORS_ABCD, LABELS)):
        ax.bar(x + offsets[i], DAILY[app], width,
               color=col, edgecolor=_darker(col, 0.6),
               linewidth=0.5, label=lbl, zorder=3)

    ax.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES, fontsize=9)
    ax.set_ylabel("Коэффициент Шарпа (net)")
    hgrid(ax)

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, fontsize=8.5)

    ax.set_ylim(0, max(max(v) for v in DAILY.values()) * 1.12)
    fig.tight_layout()
    p = FIG_DIR / "fig_4_4_ew_sharpe_by_approach.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.5 — A vs Best BCD: daily & hourly (1×2 subplots)
# ══════════════════════════════════════════════════════════════════════
def fig_4_5():
    strat_full = [
        "S1 Mean\nReversion", "S2 Bollinger\nBands", "S3 Donchian\nChannel",
        "S4 Super-\ntrend", "S5 Pivot\nPoints", "S6 VWAP",
    ]

    daily_best, daily_best_apps = _best_bcd(DAILY)
    hourly_best, hourly_best_apps = _best_bcd(HOURLY)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    n = len(strat_full)
    x = np.arange(n)
    width = 0.35

    # Map approach → colour for colouring best bars
    app_col = {"B": COL_B, "C": COL_C, "D": COL_D}

    # ─── Subplot 1: Daily ───
    ax1.bar(x - width / 2, DAILY["A"], width, color=COL_A,
            edgecolor=_darker(COL_A, 0.6), lw=0.5, label="A (без прогнозов)",
            zorder=3)

    # Best BCD bars coloured by which approach won
    for i in range(n):
        col = app_col[daily_best_apps[i]]
        ax1.bar(x[i] + width / 2, daily_best[i], width, color=col,
                edgecolor=_darker(col, 0.6), lw=0.5, zorder=3)
        # Label which approach
        ax1.text(x[i] + width / 2, daily_best[i] + 0.04,
                 daily_best_apps[i], ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=_darker(col, 0.5))

    # Dummy handles for legend
    import matplotlib.patches as mpatches
    h_a = mpatches.Patch(facecolor=COL_A, label="A (без прогнозов)")
    h_b = mpatches.Patch(facecolor=COL_B, label="B (адаптивные стопы)")
    h_c = mpatches.Patch(facecolor=COL_C, label="C (режимная фильтрация)")
    h_d = mpatches.Patch(facecolor=COL_D, label="D (vol-targeting)")
    ax1.legend(handles=[h_a, h_b, h_c, h_d], loc="upper right",
               frameon=True, fancybox=False, edgecolor="0.8", fontsize=7.5,
               title="Лучший из BCD:", title_fontsize=8)

    ax1.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strat_full, fontsize=8)
    ax1.set_ylabel("Коэффициент Шарпа (net)")
    ax1.set_title("Дневной таймфрейм (net 0,05%)", fontsize=10, pad=8)
    hgrid(ax1)
    ax1.set_ylim(0, max(max(DAILY["A"]), max(daily_best)) * 1.18)

    # ─── Subplot 2: Hourly ───
    ax2.bar(x - width / 2, HOURLY["A"], width, color=COL_A,
            edgecolor=_darker(COL_A, 0.6), lw=0.5, label="A (без прогнозов)",
            zorder=3)

    for i in range(n):
        col = app_col[hourly_best_apps[i]]
        ax2.bar(x[i] + width / 2, hourly_best[i], width, color=col,
                edgecolor=_darker(col, 0.6), lw=0.5, zorder=3)
        ax2.text(x[i] + width / 2, hourly_best[i] + 0.04,
                 hourly_best_apps[i], ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=_darker(col, 0.5))

    ax2.legend(handles=[h_a, h_b, h_c, h_d], loc="upper right",
               frameon=True, fancybox=False, edgecolor="0.8", fontsize=7.5,
               title="Лучший из BCD:", title_fontsize=8)

    ax2.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strat_full, fontsize=8)
    ax2.set_ylabel("Коэффициент Шарпа (net)")
    ax2.set_title("Часовой таймфрейм (net 0,04%)", fontsize=10, pad=8)
    hgrid(ax2)
    ax2.set_ylim(0, max(max(HOURLY["A"]), max(hourly_best)) * 1.18)

    fig.tight_layout(w_pad=3)
    p = FIG_DIR / "fig_4_5_a_vs_best_bcd.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    print("Generating fig 4.4–4.5 ...")
    fig_4_4()
    fig_4_5()
    print(f"\nAll saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
