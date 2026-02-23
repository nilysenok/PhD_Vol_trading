#!/usr/bin/env python3
"""
Section 4.3 — 5 academic figures for dissertation (final version).

fig_4_10 — Cumulative returns: MCP vs IMOEX
fig_4_11 — Correlation heatmap S1–S6
fig_4_12 — Grouped bar: Net Sharpe strategy × approach
fig_4_13 — Waterfall: value creation levels
fig_4_14 — Grouped bar: yearly returns

Usage: python3 output_4_3/figures/generate_section_43.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
V4_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
OUT_DIR = Path(__file__).resolve().parent
IMOEX_PATH = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/dataset_final"
                   "/02_external/candles_10m/IMOEX.parquet")

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
BCD_YEARS = [2022, 2023, 2024, 2025]
COMM = 0.0040

BEST_APP = {
    "S1_MeanRev": "D", "S2_Bollinger": "C", "S3_Donchian": "D",
    "S4_Supertrend": "D", "S5_PivotPoints": "B", "S6_VWAP": "D",
}

# ── Colours ───────────────────────────────────────────────────
C_HARJ     = "#1f4e79"
C_GARCH    = "#8b1a1a"
C_XGB      = "#2e7d32"
C_LGB      = "#e65100"
C_LSTM     = "#757575"
C_GRU      = "#9e9e9e"
C_HYBRID   = "#4a148c"

# Approach colours
C_A = "#808080"
C_B = "#4472C4"
C_C = "#ED7D31"
C_D = "#548235"

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size":          10,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.unicode_minus": False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})


def hgrid(ax):
    ax.yaxis.grid(True, ls=":", alpha=0.4, lw=0.6, color="grey")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ── Helpers ───────────────────────────────────────────────────
def net_returns(pos, gross_r, comm):
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * comm


def load_meta_daily_returns():
    """Build daily return series for each META portfolio from positions."""
    pos_df = pd.read_parquet(V4_DIR / "daily_positions.parquet")
    pos_df = pos_df[(pos_df["tf"] == "daily") &
                    (pos_df["test_year"].isin(BCD_YEARS))].copy()

    port_cache = {}
    for strat in STRATEGIES:
        for appr in APPROACHES:
            sub = pos_df[(pos_df["strategy"] == strat) & (pos_df["approach"] == appr)]
            tickers = sorted(sub["ticker"].unique())
            ticker_rets = {}
            for tkr in tickers:
                g = sub[sub["ticker"] == tkr].sort_values("date")
                nr = net_returns(g["position"].values,
                                 g["daily_gross_return"].values, COMM)
                ticker_rets[tkr] = pd.Series(nr, index=pd.to_datetime(g["date"].values))
            ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
            port_cache[(strat, appr)] = ret_df.mean(axis=1)

    def build_meta(approach_map):
        rets = []
        for s in STRATEGIES:
            appr = approach_map if isinstance(approach_map, str) else approach_map[s]
            rets.append(port_cache[(s, appr)])
        df = pd.DataFrame({s: r for s, r in zip(STRATEGIES, rets)}).sort_index().fillna(0)
        return df.mean(axis=1)

    return {
        "MCP-A": build_meta("A"),
        "MCP-B": build_meta("B"),
        "MCP-C": build_meta("C"),
        "MCP-D": build_meta("D"),
        "MCP-BEST": build_meta(BEST_APP),
    }


def load_imoex_daily_returns():
    """Load IMOEX daily returns."""
    imoex = pd.read_parquet(IMOEX_PATH)
    imoex["date"] = pd.to_datetime(imoex["begin"]).dt.date
    daily_close = imoex.groupby("date")["close"].last()
    daily_close.index = pd.to_datetime(daily_close.index)
    daily_ret = daily_close.pct_change().dropna()
    daily_ret = daily_ret[(daily_ret.index >= "2022-01-01") &
                          (daily_ret.index <= "2025-12-31")]
    return daily_ret


# ══════════════════════════════════════════════════════════════
#  Figure 4.10 — Cumulative returns: MCP vs IMOEX
# ══════════════════════════════════════════════════════════════
def fig_4_10(meta_rets, imoex_ret):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # IMOEX cumulative
    imoex_cum = (1 + imoex_ret).cumprod()
    ax.plot(imoex_cum.index, imoex_cum.values,
            color="#999999", ls="--", lw=1.5, label="IMOEX B&H", zorder=2)

    # MCP portfolios
    styles = {
        "MCP-A":    (C_A, "-",  1.0, 2),
        "MCP-B":    (C_B, "-",  1.0, 2),
        "MCP-D":    (C_D, "-",  1.0, 2),
        "MCP-BEST": (C_HYBRID, "-", 2.8, 4),
    }

    for name, (color, ls, lw, zo) in styles.items():
        ret_s = meta_rets[name]
        cum = (1 + ret_s).cumprod()
        ax.plot(cum.index, cum.values, color=color, ls=ls, lw=lw,
                label=name, zorder=zo)

    # Trading halt zone: Feb 25 – Mar 24, 2022
    ax.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"),
               color="#d0d0d0", alpha=0.4, zorder=0,
               label="Остановка торгов")

    ax.set_ylabel("Кумулятивная доходность", fontsize=12)
    ax.set_xlabel("Дата", fontsize=12)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    hgrid(ax)
    ax.axhline(1.0, color="black", lw=0.4, ls="-", zorder=1)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_10_cumulative_returns.png", facecolor="white")
    plt.close(fig)
    print("  fig_4_10 saved")


# ══════════════════════════════════════════════════════════════
#  Figure 4.11 — Correlation heatmap S1–S6
# ══════════════════════════════════════════════════════════════
def fig_4_11():
    corr = pd.read_csv(BASE / "output_4_3" / "table_2_correlation_matrix.csv", index_col=0)

    labels = ["S1 MeanRev", "S2 Bollinger", "S3 Donchian",
              "S4 Supertrend", "S5 Pivot", "S6 VWAP"]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Numbers inside cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Корреляция", fontsize=12)

    ax.spines[:].set_visible(True)
    ax.spines[:].set_color("0.8")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_11_correlation_heatmap.png", facecolor="white")
    plt.close(fig)
    print("  fig_4_11 saved")


# ══════════════════════════════════════════════════════════════
#  Figure 4.12 — Grouped bar: Net Sharpe strategy × approach
# ══════════════════════════════════════════════════════════════
def fig_4_12():
    strat_labels = [
        "S1\nMeanRev", "S2\nBollinger", "S3\nDonchian",
        "S4\nSupertrend", "S5\nPivot", "S6\nVWAP",
    ]
    data = {
        "A": [1.682, 1.937, 0.855, 1.075, 1.218, 1.728],
        "B": [1.821, 1.818, 1.469, 1.507, 1.584, 1.895],
        "C": [2.049, 2.614, 0.899, 1.088, 1.411, 1.988],
        "D": [2.431, 2.450, 1.530, 2.125, 1.352, 2.407],
    }

    mean_a = np.mean(data["A"])  # 1.416

    n_strat = len(strat_labels)
    x = np.arange(n_strat)
    width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    colors_abcd = [C_A, C_B, C_C, C_D]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, (app, col) in enumerate(zip(APPROACHES, colors_abcd)):
        ax.bar(x + offsets[i], data[app], width,
               color=col, edgecolor="black", linewidth=0.3,
               label=app, zorder=3)

    # Mean Sharpe approach A dashed line
    ax.axhline(mean_a, color=C_A, ls="--", lw=1.2, alpha=0.7, zorder=2,
               label=f"Среднее A ({mean_a:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylabel("Коэффициент Шарпа (net)", fontsize=12)
    hgrid(ax)

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, fontsize=10, ncol=5)

    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_12_net_sharpe_strategy_approach.png", facecolor="white")
    plt.close(fig)
    print("  fig_4_12 saved")


# ══════════════════════════════════════════════════════════════
#  Figure 4.13 — Waterfall: value creation levels
# ══════════════════════════════════════════════════════════════
def fig_4_13():
    labels = ["IMOEX B&H", "MCP-A", "MCP-BEST", "MCP-BEST\n+ ФДР"]
    values = [-0.12, 1.43, 2.68, 5.44]
    colors = ["#CC4444", C_A, C_D, "#2E75B6"]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    y = np.arange(len(labels))
    bars = ax.barh(y, values, height=0.55, color=colors,
                   edgecolor="black", lw=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Коэффициент Шарпа (net)", fontsize=12)
    ax.xaxis.grid(True, ls=":", alpha=0.4, lw=0.6, color="grey")
    ax.set_axisbelow(True)
    ax.axvline(0, color="black", lw=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        if val >= 0:
            ax.text(x_pos + 0.08, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="left",
                    fontsize=12, fontweight="bold")
        else:
            ax.text(x_pos - 0.08, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="right",
                    fontsize=12, fontweight="bold", color="white")

    ax.set_xlim(-0.5, max(values) + 0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_13_waterfall_value_creation.png", facecolor="white")
    plt.close(fig)
    print("  fig_4_13 saved")


# ══════════════════════════════════════════════════════════════
#  Figure 4.14 — Grouped bar: yearly returns
# ══════════════════════════════════════════════════════════════
def fig_4_14():
    years = [2022, 2023, 2024, 2025]
    mcp_a =    [11.41,  4.75,  7.26,  2.12]
    mcp_best = [13.75,  5.71,  8.32,  3.15]
    imoex =    [-43.12, 43.87, -6.97, -4.04]

    n = len(years)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_a = ax.bar(x - width, mcp_a, width, color=C_A,
                    edgecolor="black", lw=0.3,
                    label="MCP-A", zorder=3)
    bars_best = ax.bar(x, mcp_best, width, color=C_D,
                       edgecolor="black", lw=0.3,
                       label="MCP-BEST", zorder=3)
    bars_imoex = ax.bar(x + width, imoex, width, color="#CC4444",
                        edgecolor="black", lw=0.3,
                        label="IMOEX B&H", zorder=3)

    # Value labels above/below bars
    def add_labels(bars, vals):
        for bar, val in zip(bars, vals):
            offset = 1.0 if val >= 0 else -1.5
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:+.1f}%", ha="center", va=va, fontsize=9,
                    fontweight="bold")

    add_labels(bars_a, mcp_a)
    add_labels(bars_best, mcp_best)
    add_labels(bars_imoex, imoex)

    ax.axhline(0, color="black", lw=0.6, ls="-", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_ylabel("Годовая доходность, %", fontsize=12)
    hgrid(ax)

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, fontsize=10)

    ax.set_ylim(min(imoex) * 1.18, max(max(mcp_best), max(imoex)) * 1.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_14_yearly_returns.png", facecolor="white")
    plt.close(fig)
    print("  fig_4_14 saved")


# ══════════════════════════════════════════════════════════════
def main():
    print("Section 4.3 figures (final) ...")
    print("  Loading data...")

    meta_rets = load_meta_daily_returns()
    imoex_ret = load_imoex_daily_returns()
    print("  Data loaded.")

    fig_4_10(meta_rets, imoex_ret)
    fig_4_11()
    fig_4_12()
    fig_4_13()
    fig_4_14()

    print(f"\nAll saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
