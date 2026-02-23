#!/usr/bin/env python3
"""
Section 4.1 Charts — V4 Strategy x Approach Analysis.

Generates 5 academic figures (Figures 4.1-4.5) from walk-forward V4 results.
All use approach D (forecast-enhanced), the main contribution.

Usage: python3 scripts/v4_charts_41.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = (
    Path(__file__).resolve().parent.parent
    / "results" / "final" / "strategies" / "walkforward_v4"
)
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
STRATEGIES = [
    "S1_MeanRev", "S2_Bollinger", "S3_Donchian",
    "S4_Supertrend", "S5_PivotPoints", "S6_VWAP",
]

SHORT_NAMES = {
    "S1_MeanRev":     "S1 Mean\nReversion",
    "S2_Bollinger":   "S2 Bollinger\nBands",
    "S3_Donchian":    "S3 Donchian\nChannel",
    "S4_Supertrend":  "S4 Super-\ntrend",
    "S5_PivotPoints": "S5 Pivot\nPoints",
    "S6_VWAP":        "S6 VWAP",
}

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

APP = "D"
RED = "#C0504D"
FIGSIZE = (18 / 2.54, 12 / 2.54)      # 18 x 12 cm


def _c(s):
    """Strategy color by category."""
    return CAT_COLORS[CATEGORY[s]]


# ── Style ──────────────────────────────────────────────────────────────
def setup_rc():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.15,
    })


def hgrid(ax):
    """Subtle horizontal grid only."""
    ax.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def cat_legend(ax, **kw):
    """Category-colour legend (no duplicates)."""
    seen = {}
    for s in STRATEGIES:
        cat = CATEGORY[s]
        if cat not in seen:
            seen[cat] = CAT_COLORS[cat]
    handles = [mpatches.Patch(facecolor=c, label=n) for n, c in seen.items()]
    return ax.legend(handles=handles, frameon=False, **kw)


def v(df, strat, col):
    """Extract one value from the full-results table for approach D."""
    row = df[(df["Strategy"] == strat) & (df["App"] == APP)]
    return row.iloc[0][col] if not row.empty else np.nan


# ── Data ───────────────────────────────────────────────────────────────
def load():
    daily  = pd.read_csv(BASE / "tables" / "v4_full_daily.csv")
    hourly = pd.read_csv(BASE / "tables" / "v4_full_hourly.csv")
    return daily, hourly


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.1  —  Grouped bar: Gross Sharpe, daily vs hourly
# ══════════════════════════════════════════════════════════════════════
def fig_4_1(daily, hourly):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(STRATEGIES))
    w = 0.33

    d_vals = [v(daily,  s, "GrossSharpe") for s in STRATEGIES]
    h_vals = [v(hourly, s, "GrossSharpe") for s in STRATEGIES]

    for i, s in enumerate(STRATEGIES):
        c = _c(s)
        ax.bar(x[i] - w/2, d_vals[i], w,
               color=c, edgecolor="white", lw=0.5)
        ax.bar(x[i] + w/2, h_vals[i], w,
               color=c, edgecolor="white", lw=0.5, hatch="///", alpha=0.7)

    # Timeframe legend (upper-left)
    tf = [
        mpatches.Patch(facecolor="grey",              label="Дневные"),
        mpatches.Patch(facecolor="grey", hatch="///",
                       alpha=0.7,                     label="Часовые"),
    ]
    leg1 = ax.legend(handles=tf, frameon=False, loc="upper left")
    ax.add_artist(leg1)
    # Category legend (upper-right)
    cat_legend(ax, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[s] for s in STRATEGIES])
    ax.set_ylabel("Коэффициент Шарпа (gross)")
    ax.set_ylim(0)
    hgrid(ax)

    fig.tight_layout()
    p = FIG_DIR / "fig_4_1_gross_sharpe.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.2  —  Waterfall: Gross → Net
# ══════════════════════════════════════════════════════════════════════
def fig_4_2(daily, hourly):
    fig, (ax_d, ax_h) = plt.subplots(
        2, 1, figsize=(22 / 2.54, 16 / 2.54)
    )
    configs = [
        (ax_d, daily,  "Net0.05Sharpe", "Дневные данные (комиссия 0,05 %)"),
        (ax_h, hourly, "Net0.04Sharpe", "Часовые данные (комиссия 0,04 %)"),
    ]

    bw   = 0.52
    gap  = 4.2        # x-spacing between strategy groups

    for ax, df, net_col, title in configs:
        ymax = 0
        for i, s in enumerate(STRATEGIES):
            g    = v(df, s, "GrossSharpe")
            n    = v(df, s, net_col)
            loss = g - n
            c    = _c(s)
            xb   = i * gap              # base x for this group
            ymax = max(ymax, g)

            # 1) Gross bar
            ax.bar(xb, g, bw, color=c, edgecolor="white", lw=0.5)
            # 2) Commission loss (floating from net up to gross)
            ax.bar(xb + 1, loss, bw, bottom=n,
                   color=RED, edgecolor="white", lw=0.5, alpha=0.8)
            # 3) Net bar
            ax.bar(xb + 2, n, bw, color=c, edgecolor="white", lw=0.5,
                   alpha=0.55)

            # Dashed connectors
            ax.plot([xb + bw/2, xb + 1 - bw/2], [g, g],
                    color="grey", lw=0.4, ls="--")
            ax.plot([xb + 1 + bw/2, xb + 2 - bw/2], [n, n],
                    color="grey", lw=0.4, ls="--")

            # Value labels
            off = ymax * 0.025
            ax.text(xb,     g + off,          f"{g:.2f}",
                    ha="center", va="bottom", fontsize=6.5)
            ax.text(xb + 2, max(n, 0) + off,  f"{n:.2f}",
                    ha="center", va="bottom", fontsize=6.5)

        ax.axhline(0, color="black", lw=0.5)
        centres = [i * gap + 1 for i in range(len(STRATEGIES))]
        ax.set_xticks(centres)
        ax.set_xticklabels([s.split("_")[1] for s in STRATEGIES], fontsize=9)
        ax.set_ylabel("Коэфф. Шарпа")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylim(bottom=-0.02, top=ymax * 1.15)
        hgrid(ax)

    # Legend — top subplot
    leg = [
        mpatches.Patch(facecolor="grey",           label="Gross"),
        mpatches.Patch(facecolor=RED, alpha=0.8,   label="Комиссии"),
        mpatches.Patch(facecolor="grey", alpha=0.55, label="Net"),
    ]
    ax_d.legend(handles=leg, frameon=False, loc="upper right",
                ncol=3, fontsize=8)

    fig.tight_layout(h_pad=1.5)
    p = FIG_DIR / "fig_4_2_waterfall.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.3  —  Net Sharpe at 0.40 % vs 0.50 %
# ══════════════════════════════════════════════════════════════════════
def fig_4_3(daily):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(STRATEGIES))
    w = 0.33

    n40 = [v(daily, s, "Net0.05Sharpe") for s in STRATEGIES]
    n50 = [v(daily, s, "Net0.06Sharpe") for s in STRATEGIES]

    for i, s in enumerate(STRATEGIES):
        c = _c(s)
        ax.bar(x[i] - w/2, n40[i], w,
               color=c, edgecolor="white", lw=0.5)
        ax.bar(x[i] + w/2, n50[i], w,
               color=c, edgecolor="white", lw=0.5, alpha=0.55, hatch="\\\\")

    # Commission-level legend (upper-left)
    tf = [
        mpatches.Patch(facecolor="grey",                       label="Комиссия 0,05 %"),
        mpatches.Patch(facecolor="grey", alpha=0.55, hatch="\\\\", label="Комиссия 0,06 %"),
    ]
    leg1 = ax.legend(handles=tf, frameon=False, loc="upper left")
    ax.add_artist(leg1)
    cat_legend(ax, loc="upper right")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[s] for s in STRATEGIES])
    ax.set_ylabel("Коэффициент Шарпа (net)")
    hgrid(ax)

    fig.tight_layout()
    p = FIG_DIR / "fig_4_3_net_sensitivity.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.4  —  3-D bar: Net Sharpe, all strategies
# ══════════════════════════════════════════════════════════════════════
def _darker(hex_color, factor=0.65):
    """Return a darker shade for bar edges."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def _lighter(hex_color, factor=0.55):
    """Return a lighter/desaturated shade for hourly bars."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (1 - (1 - r) * factor, 1 - (1 - g) * factor, 1 - (1 - b) * factor)


def _setup_3d_ax(ax):
    """Common 3D axes styling."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
    ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
    ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
    ax.zaxis.gridlines.set_alpha(0.2)
    ax.xaxis.gridlines.set_alpha(0.1)
    ax.yaxis.gridlines.set_alpha(0.1)


def _bar3d_single(ax, xs, values, colors, dx=0.7, dy=0.6):
    """Draw one row of 3D bars with value labels."""
    for i in range(len(xs)):
        c     = colors[i]
        cEdge = _darker(c, 0.5)
        ax.bar3d(xs[i] - dx/2, 0, 0, dx, dy, values[i],
                 color=c, edgecolor=cEdge, lw=0.5, alpha=0.95)
        # Value label above bar
        ax.text(xs[i], dy/2, values[i] + max(values) * 0.03,
                f"{values[i]:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # Zero plane
    xx, yy = np.meshgrid([xs[0] - 0.6, xs[-1] + 0.6], [-0.15, dy + 0.15])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color="grey")

    # Hide y-axis (only one row — no need)
    ax.set_yticks([])
    ax.set_ylabel("")


def _finish_3d(fig, ax, zlabel, fname, zmax):
    """Common finishing for single-row 3D charts."""
    ax.set_zlabel("")
    ax.set_zlim(0, zmax * 1.25)
    ax.view_init(elev=28, azim=-35)
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='z', labelsize=8)
    ax.tick_params(axis='x', pad=1)

    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.0, top=1.0)
    fig.text(0.95, 0.50, zlabel, rotation=90,
             va="center", ha="center", fontsize=9)

    p = FIG_DIR / fname
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figures 4.4a/b  —  3-D bars: Net Sharpe per strategy
# ══════════════════════════════════════════════════════════════════════
def fig_4_4(daily, hourly):
    d_net = [v(daily,  s, "Net0.05Sharpe") for s in STRATEGIES]
    h_net = [v(hourly, s, "Net0.04Sharpe") for s in STRATEGIES]

    # Sort by daily Sharpe descending
    d_order = np.argsort(d_net)[::-1]
    d_vals_s  = [d_net[i] for i in d_order]
    d_cols_s  = [_c(STRATEGIES[i]) for i in d_order]
    d_xlabs_s = [STRATEGIES[i].split("_")[1] for i in d_order]

    # Sort by hourly Sharpe descending
    h_order = np.argsort(h_net)[::-1]
    h_vals_s  = [h_net[i] for i in h_order]
    h_cols_s  = [_c(STRATEGIES[i]) for i in h_order]
    h_xlabs_s = [STRATEGIES[i].split("_")[1] for i in h_order]

    # Category legend handles
    seen = {}
    for s in STRATEGIES:
        cat = CATEGORY[s]
        if cat not in seen:
            seen[cat] = _c(s)
    leg_h = [mpatches.Patch(facecolor=c, edgecolor=_darker(c, 0.5),
                            label=n) for n, c in seen.items()]

    # --- 4.4a: Daily (sorted) ---
    xs = np.arange(len(STRATEGIES), dtype=float)
    fig = plt.figure(figsize=(22 / 2.54, 13 / 2.54))
    ax  = fig.add_subplot(111, projection="3d")
    _setup_3d_ax(ax)
    _bar3d_single(ax, xs, d_vals_s, d_cols_s)
    ax.set_xticks(xs)
    ax.set_xticklabels(d_xlabs_s, fontsize=8)
    ax.set_title("Дневные стратегии (комиссия 0,05 %)", fontsize=10, pad=12)
    ax.legend(handles=leg_h, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 0.95), fontsize=8)
    _finish_3d(fig, ax, "Коэфф. Шарпа (net)", "fig_4_4a_3d_daily.png",
               max(d_vals_s))

    # --- 4.4b: Hourly (sorted) ---
    fig = plt.figure(figsize=(22 / 2.54, 13 / 2.54))
    ax  = fig.add_subplot(111, projection="3d")
    _setup_3d_ax(ax)
    _bar3d_single(ax, xs, h_vals_s, h_cols_s)
    ax.set_xticks(xs)
    ax.set_xticklabels(h_xlabs_s, fontsize=8)
    ax.set_title("Часовые стратегии (комиссия 0,04 %)", fontsize=10, pad=12)
    ax.legend(handles=leg_h, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 0.95), fontsize=8)
    _finish_3d(fig, ax, "Коэфф. Шарпа (net)", "fig_4_4b_3d_hourly.png",
               max(h_vals_s))


# ══════════════════════════════════════════════════════════════════════
#  Figures 4.5a/b  —  3-D bars: Net Sharpe per category
# ══════════════════════════════════════════════════════════════════════
def fig_4_5(daily, hourly):
    cats = [
        ("Контртренд", ["S1_MeanRev", "S2_Bollinger"]),
        ("Тренд",      ["S3_Donchian", "S4_Supertrend"]),
        ("Диапазон",   ["S5_PivotPoints", "S6_VWAP"]),
    ]

    d_vals = [np.nanmean([v(daily,  s, "Net0.05Sharpe") for s in st])
              for _, st in cats]
    h_vals = [np.nanmean([v(hourly, s, "Net0.04Sharpe") for s in st])
              for _, st in cats]

    # Sort daily by descending Sharpe
    d_order = np.argsort(d_vals)[::-1]
    d_vals_s  = [d_vals[i] for i in d_order]
    d_cols_s  = [CAT_COLORS[cats[i][0]] for i in d_order]
    d_xlabs_s = [cats[i][0] for i in d_order]

    # Sort hourly by descending Sharpe
    h_order = np.argsort(h_vals)[::-1]
    h_vals_s  = [h_vals[i] for i in h_order]
    h_cols_s  = [CAT_COLORS[cats[i][0]] for i in h_order]
    h_xlabs_s = [cats[i][0] for i in h_order]

    # --- 4.5a: Daily (sorted) ---
    xs = np.arange(len(cats), dtype=float)
    fig = plt.figure(figsize=(20 / 2.54, 13 / 2.54))
    ax  = fig.add_subplot(111, projection="3d")
    _setup_3d_ax(ax)
    _bar3d_single(ax, xs, d_vals_s, d_cols_s, dx=0.8, dy=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(d_xlabs_s, fontsize=9)
    ax.set_title("Дневные: среднее по категориям (0,05 %)", fontsize=10, pad=12)
    _finish_3d(fig, ax, "Коэфф. Шарпа (net, среднее)",
               "fig_4_5a_3d_cat_daily.png", max(d_vals_s))

    # --- 4.5b: Hourly (sorted) ---
    fig = plt.figure(figsize=(20 / 2.54, 13 / 2.54))
    ax  = fig.add_subplot(111, projection="3d")
    _setup_3d_ax(ax)
    _bar3d_single(ax, xs, h_vals_s, h_cols_s, dx=0.8, dy=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(h_xlabs_s, fontsize=9)
    ax.set_title("Часовые: среднее по категориям (0,04 %)", fontsize=10, pad=12)
    _finish_3d(fig, ax, "Коэфф. Шарпа (net, среднее)",
               "fig_4_5b_3d_cat_hourly.png", max(h_vals_s))


# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    daily, hourly = load()
    print("Generating Section 4.1 figures ...")
    fig_4_1(daily, hourly)
    fig_4_2(daily, hourly)
    fig_4_3(daily)
    fig_4_4(daily, hourly)
    fig_4_5(daily, hourly)
    print(f"\nAll saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
