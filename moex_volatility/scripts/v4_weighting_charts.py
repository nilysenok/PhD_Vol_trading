#!/usr/bin/env python3
"""
Dissertation Section 5 — Weighting Method Comparison Charts.

  Fig 5.1 — Grouped bar: mean Sharpe by weighting method (daily + hourly panels)
  Fig 5.2 — Wins out of 24 by weighting method (daily + hourly panels)
  Fig 5.3 — Annual turnover comparison (daily vs hourly, multiplier vs EW)

Data sources:
  - v4_weighting_summary_{daily,hourly}.csv  (aggregated summary)
  - v4_portfolios_{daily,hourly}.csv          (per-portfolio detail)

Usage: python3 scripts/v4_weighting_charts.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
V4_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
TABLE_DIR = V4_DIR / "tables"
FIG_DIR = V4_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style setup ───────────────────────────────────────────────────────
def setup_rc():
    """Set academic matplotlib style matching other dissertation figures."""
    try:
        from matplotlib.font_manager import findSystemFonts
        available = [f.lower() for f in findSystemFonts()]
        has_times = any("times" in f for f in available)
    except Exception:
        has_times = False

    font_family = "serif"
    font_serif = ["Times New Roman", "Times", "DejaVu Serif"]

    plt.rcParams.update({
        "font.family":        font_family,
        "font.serif":         font_serif,
        "font.size":          12,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.unicode_minus": False,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "grid.color":         "#B0B0B0",
    })


def comma_fmt(x, _):
    """Format number with comma as decimal separator (Russian convention)."""
    s = f"{x:g}"
    return s.replace(".", ",")


def comma_fmt_2f(x, _):
    s = f"{x:.2f}"
    return s.replace(".", ",")


# ── Colour palette ────────────────────────────────────────────────────
METHOD_COLORS = {
    "EW":       "#4472C4",   # blue
    "MinVar":   "#ED7D31",   # orange
    "MaxSharpe":"#70AD47",   # green
    "InvVol":   "#C00000",   # red
}
METHOD_ORDER = ["EW", "MinVar", "MaxSharpe", "InvVol"]
METHOD_LABELS_RU = {
    "EW":        "Равн. веса (EW)",
    "MinVar":    "Мин. дисп. (MinVar)",
    "MaxSharpe": "Макс. Шарп (MaxSharpe)",
    "InvVol":    "Обр. волат. (InvVol)",
}

COMM_LABELS_RU = {
    "gross":    "Gross",
    "net_0.05": "Net 0,05%",
    "net_0.06": "Net 0,06%",
    "net_0.04": "Net 0,04%",
}


# ── Load data ─────────────────────────────────────────────────────────
def load_summary():
    """Load summary CSVs for daily and hourly."""
    df_d = pd.read_csv(TABLE_DIR / "v4_weighting_summary_daily.csv")
    df_h = pd.read_csv(TABLE_DIR / "v4_weighting_summary_hourly.csv")
    return df_d, df_h


def load_portfolios():
    """Load per-portfolio CSVs for daily and hourly."""
    df_d = pd.read_csv(TABLE_DIR / "v4_portfolios_daily.csv")
    df_h = pd.read_csv(TABLE_DIR / "v4_portfolios_hourly.csv")
    return df_d, df_h


# ══════════════════════════════════════════════════════════════════════
#  FIG 5.1 — Grouped bar: mean Sharpe by method across commission levels
# ══════════════════════════════════════════════════════════════════════
def fig_5_1_sharpe_by_method(df_daily, df_hourly):
    """Two-panel grouped bar chart: mean Sharpe per method per commission level."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

    datasets = [
        (df_daily,  axes[0], "Дневные данные"),
        (df_hourly, axes[1], "Часовые данные"),
    ]

    for df, ax, title in datasets:
        comm_levels = df["comm_level"].unique()
        n_comm = len(comm_levels)
        n_methods = len(METHOD_ORDER)

        bar_width = 0.18
        x_base = np.arange(n_comm)

        for i, method in enumerate(METHOD_ORDER):
            offsets = x_base + (i - (n_methods - 1) / 2) * bar_width
            values = []
            for cl in comm_levels:
                row = df[(df["comm_level"] == cl) & (df["method"] == method)]
                values.append(row["mean_sharpe"].values[0] if len(row) > 0 else 0)

            bars = ax.bar(
                offsets, values, bar_width,
                label=METHOD_LABELS_RU[method],
                color=METHOD_COLORS[method],
                edgecolor="white", linewidth=0.5,
                zorder=3,
            )

            # Add value labels on bars
            for bar, val in zip(bars, values):
                label_text = f"{val:.2f}".replace(".", ",")
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    label_text,
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        # X-axis labels
        ax.set_xticks(x_base)
        ax.set_xticklabels([COMM_LABELS_RU.get(cl, cl) for cl in comm_levels])
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylabel("Средний коэффициент Шарпа")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt_2f))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        ncol=4, frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Рис. 5.1. Средний коэффициент Шарпа по методам взвешивания",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    out = FIG_DIR / "fig_5_1_sharpe_by_method.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
#  FIG 5.2 — Wins out of 24 by method
# ══════════════════════════════════════════════════════════════════════
def fig_5_2_wins_by_method(df_daily, df_hourly):
    """Two-panel grouped bar chart: wins by method across commission levels."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

    datasets = [
        (df_daily,  axes[0], "Дневные данные"),
        (df_hourly, axes[1], "Часовые данные"),
    ]

    for df, ax, title in datasets:
        comm_levels = df["comm_level"].unique()
        n_comm = len(comm_levels)
        n_methods = len(METHOD_ORDER)

        bar_width = 0.18
        x_base = np.arange(n_comm)

        for i, method in enumerate(METHOD_ORDER):
            offsets = x_base + (i - (n_methods - 1) / 2) * bar_width
            values = []
            for cl in comm_levels:
                row = df[(df["comm_level"] == cl) & (df["method"] == method)]
                values.append(int(row["wins_24"].values[0]) if len(row) > 0 else 0)

            bars = ax.bar(
                offsets, values, bar_width,
                label=METHOD_LABELS_RU[method],
                color=METHOD_COLORS[method],
                edgecolor="white", linewidth=0.5,
                zorder=3,
            )

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(val),
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                    )

        # Highlight EW dominance: add a horizontal line at 24 (max wins)
        ax.axhline(y=24, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
        ax.text(
            n_comm - 0.5, 24.3, "макс = 24",
            ha="right", va="bottom", fontsize=9, color="#999999", fontstyle="italic",
        )

        ax.set_xticks(x_base)
        ax.set_xticklabels([COMM_LABELS_RU.get(cl, cl) for cl in comm_levels])
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylabel("Число побед (из 24 портфелей)")
        ax.set_ylim(0, 28)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(4))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        ncol=4, frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Рис. 5.2. Число побед методов взвешивания (из 24 портфелей)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    out = FIG_DIR / "fig_5_2_wins_by_method.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
#  FIG 5.3 — Annual turnover comparison
# ══════════════════════════════════════════════════════════════════════
# Actual annual turnover computed by v4_turnover_analysis.py (build_portfolio weight drift)
TURNOVER_DAILY = {"EW": 18, "MinVar": 49, "MaxSharpe": 119, "InvVol": 194}
TURNOVER_HOURLY = {"EW": 15, "MinVar": 62, "MaxSharpe": 153, "InvVol": 185}


def fig_5_3_turnover_comparison(df_daily_port, df_hourly_port):
    """Bar chart: actual annual rebalancing turnover by method."""

    turnover_daily = TURNOVER_DAILY
    turnover_hourly = TURNOVER_HOURLY

    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_methods = len(METHOD_ORDER)
    bar_width = 0.32
    x_base = np.arange(n_methods)

    # Daily bars
    vals_d = [turnover_daily[m] for m in METHOD_ORDER]
    bars_d = ax.bar(
        x_base - bar_width / 2, vals_d, bar_width,
        label="Дневные данные",
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor="white", linewidth=0.5, zorder=3,
        alpha=0.85,
    )

    # Hourly bars
    vals_h = [turnover_hourly[m] for m in METHOD_ORDER]
    bars_h = ax.bar(
        x_base + bar_width / 2, vals_h, bar_width,
        label="Часовые данные",
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor="black", linewidth=0.8, zorder=3,
        alpha=0.50,
        hatch="//",
    )

    # Compute multipliers vs EW and annotate
    ew_d = turnover_daily["EW"]
    ew_h = turnover_hourly["EW"]

    for i, method in enumerate(METHOD_ORDER):
        # Daily multiplier
        mult_d = turnover_daily[method] / ew_d if ew_d != 0 else 0
        y_d = vals_d[i]
        ax.text(
            x_base[i] - bar_width / 2, y_d + 0.02,
            f"{mult_d:.1f}x".replace(".", ","),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=METHOD_COLORS[method],
        )

        # Hourly multiplier
        mult_h = turnover_hourly[method] / ew_h if ew_h != 0 else 0
        y_h = vals_h[i]
        ax.text(
            x_base[i] + bar_width / 2, y_h + 0.02,
            f"{mult_h:.1f}x".replace(".", ","),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#444444",
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([METHOD_LABELS_RU[m] for m in METHOD_ORDER], fontsize=10)
    ax.set_ylabel("Среднегодовой turnover, %")
    ax.set_ylim(0, max(max(vals_d), max(vals_h)) * 1.25)

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_title(
        "Рис. 5.3. Среднегодовой turnover по методам взвешивания\n"
        "(множитель относительно EW)",
        fontsize=13, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    out = FIG_DIR / "fig_5_3_turnover_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()

    print("Loading data...")
    df_sum_d, df_sum_h = load_summary()
    df_port_d, df_port_h = load_portfolios()
    print(f"  Summary daily:  {len(df_sum_d)} rows, hourly: {len(df_sum_h)} rows")
    print(f"  Portfolios daily: {len(df_port_d)} rows, hourly: {len(df_port_h)} rows")

    print("\nGenerating figures...")
    fig_5_1_sharpe_by_method(df_sum_d, df_sum_h)
    fig_5_2_wins_by_method(df_sum_d, df_sum_h)
    fig_5_3_turnover_comparison(df_port_d, df_port_h)

    print("\nDone! All figures saved to:")
    print(f"  {FIG_DIR}")


if __name__ == "__main__":
    main()
