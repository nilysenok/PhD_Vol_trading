#!/usr/bin/env python3
"""
Section 4.1 Charts — Baseline strategy analysis (Approach A).

Generates 3 academic figures (4.1–4.3) from walk-forward V4 results.
All use approach A (no volatility forecast).

Usage: python3 dissertation_materials/chapter_4_1/plot_section_4_1.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = (
    Path(__file__).resolve().parent.parent.parent
    / "results" / "final" / "strategies" / "walkforward_v4"
)
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
STRATEGIES = [
    "S1_MeanRev", "S2_Bollinger", "S3_Donchian",
    "S4_Supertrend", "S5_PivotPoints", "S6_VWAP",
]

ROW_LABELS = [
    "S1 Mean Reversion",
    "S2 Bollinger Bands",
    "S3 Donchian Channel",
    "S4 Supertrend",
    "S5 Pivot Points",
    "S6 VWAP",
]

CATEGORY = {
    "S1_MeanRev":     "Контртренд",
    "S2_Bollinger":   "Контртренд",
    "S3_Donchian":    "Тренд",
    "S4_Supertrend":  "Тренд",
    "S5_PivotPoints": "Диапазон",
    "S6_VWAP":        "Диапазон",
}

CAT_COLORS = {
    "Контртренд": "#4472C4",
    "Тренд":      "#548235",
    "Диапазон":   "#BF8F00",
}

APP = "A"


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
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
    })


def v(df, strat, col):
    """Extract one value from the results table for approach A."""
    row = df[(df["Strategy"] == strat) & (df["App"] == APP)]
    return row.iloc[0][col] if not row.empty else np.nan


def load():
    daily  = pd.read_csv(BASE / "tables" / "v4_full_daily.csv")
    hourly = pd.read_csv(BASE / "tables" / "v4_full_hourly.csv")
    return daily, hourly


# ── Heatmap colormap: red → yellow → green ────────────────────────────
CMAP_RYG = LinearSegmentedColormap.from_list(
    "ryg", ["#C0504D", "#FADC68", "#548235"], N=256
)


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.1 — Heatmap: Daily timeframe
# ══════════════════════════════════════════════════════════════════════
def fig_4_1(daily):
    cols = ["GrossSharpe", "Net0.05Sharpe", "Net0.06Sharpe"]
    col_labels = ["Без\nкомиссий", "Ком.\n0,05 %", "Ком.\n0,06 %"]

    data = np.array([[v(daily, s, c) for c in cols] for s in STRATEGIES])

    fig, ax = plt.subplots(figsize=(10 / 2.54, 10 / 2.54))
    im = ax.imshow(data, cmap=CMAP_RYG, aspect="auto",
                   vmin=min(0, np.nanmin(data)), vmax=np.nanmax(data))

    # Value annotations
    for i in range(len(STRATEGIES)):
        for j in range(len(cols)):
            val = data[i, j]
            # Dark text on light cells, white on dark
            lum = 0.299 * im.cmap(im.norm(val))[0] + \
                  0.587 * im.cmap(im.norm(val))[1] + \
                  0.114 * im.cmap(im.norm(val))[2]
            color = "black" if lum > 0.45 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=9)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # Remove all spines for clean heatmap look
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Дневной таймфрейм: коэффициент Шарпа\n(подход A)",
                 fontsize=11, pad=10)

    fig.tight_layout()
    p = FIG_DIR / "fig_4_1_heatmap_daily.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.2 — Heatmap: Hourly timeframe
# ══════════════════════════════════════════════════════════════════════
def fig_4_2(hourly):
    cols = ["GrossSharpe", "Net0.04Sharpe", "Net0.05Sharpe"]
    col_labels = ["Без\nкомиссий", "Ком.\n0,04 %", "Ком.\n0,05 %"]

    data = np.array([[v(hourly, s, c) for c in cols] for s in STRATEGIES])

    fig, ax = plt.subplots(figsize=(10 / 2.54, 10 / 2.54))
    im = ax.imshow(data, cmap=CMAP_RYG, aspect="auto",
                   vmin=min(0, np.nanmin(data)), vmax=np.nanmax(data))

    for i in range(len(STRATEGIES)):
        for j in range(len(cols)):
            val = data[i, j]
            lum = 0.299 * im.cmap(im.norm(val))[0] + \
                  0.587 * im.cmap(im.norm(val))[1] + \
                  0.114 * im.cmap(im.norm(val))[2]
            color = "black" if lum > 0.45 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=9)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Часовой таймфрейм: коэффициент Шарпа\n(подход A)",
                 fontsize=11, pad=10)

    fig.tight_layout()
    p = FIG_DIR / "fig_4_2_heatmap_hourly.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.3 — Grouped bar: avg Sharpe by category (Gross + 2 comms)
# ══════════════════════════════════════════════════════════════════════
def fig_4_3(daily, hourly):
    cats = [
        ("Контртренд", ["S1_MeanRev", "S2_Bollinger"]),
        ("Тренд",      ["S3_Donchian", "S4_Supertrend"]),
        ("Диапазон",   ["S5_PivotPoints", "S6_VWAP"]),
    ]
    cat_names = [c[0] for c in cats]

    # Daily: Gross, 0.05%, 0.06%
    d_cols = ["GrossSharpe", "Net0.05Sharpe", "Net0.06Sharpe"]
    d_labels = ["Gross", "0,05 %", "0,06 %"]
    # Hourly: Gross, 0.04%, 0.05%
    h_cols = ["GrossSharpe", "Net0.04Sharpe", "Net0.05Sharpe"]
    h_labels = ["Gross", "0,04 %", "0,05 %"]

    d_data = [[np.nanmean([v(daily, s, c) for s in st]) for c in d_cols]
              for _, st in cats]
    h_data = [[np.nanmean([v(hourly, s, c) for s in st]) for c in h_cols]
              for _, st in cats]

    fig, axes = plt.subplots(1, 2, figsize=(28 / 2.54, 10 / 2.54), sharey=True)

    hatches = ["", "///", "xxx"]
    alphas = [1.0, 0.75, 0.55]

    for ax_idx, (ax, data, labels, title) in enumerate(zip(
        axes,
        [d_data, h_data],
        [d_labels, h_labels],
        ["Дневной таймфрейм", "Часовой таймфрейм"],
    )):
        x = np.arange(len(cats))
        n_bars = len(labels)
        w = 0.24
        offsets = [-(n_bars - 1) / 2 * w + k * w for k in range(n_bars)]

        for j, (off, lbl, hatch, alpha) in enumerate(
            zip(offsets, labels, hatches, alphas)
        ):
            for i, (name, _) in enumerate(cats):
                c = CAT_COLORS[name]
                val = data[i][j]
                ax.bar(x[i] + off, val, w * 0.9,
                       color=c, edgecolor="white", lw=0.5,
                       hatch=hatch, alpha=alpha)
                ax.text(x[i] + off, val + 0.012, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7)

        # Legend from hatches
        leg = [mpatches.Patch(facecolor="grey", hatch=h, alpha=a, label=l)
               for l, h, a in zip(labels, hatches, alphas)]
        ax.legend(handles=leg, frameon=False, fontsize=7.5, ncol=n_bars,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0))

        ax.set_xticks(x)
        ax.set_xticklabels(cat_names, fontsize=9)
        ax.set_title(title, fontsize=11, pad=22)
        ax.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Коэффициент Шарпа (подход A)")
    all_vals = [v for row in d_data + h_data for v in row]
    y_max = max(all_vals) * 1.15
    axes[0].set_ylim(0, y_max)

    fig.tight_layout()
    p = FIG_DIR / "fig_4_3_categories.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    daily, hourly = load()
    print("Generating Section 4.1 figures ...")
    fig_4_1(daily)
    fig_4_2(hourly)
    fig_4_3(daily, hourly)
    print(f"\nAll saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
