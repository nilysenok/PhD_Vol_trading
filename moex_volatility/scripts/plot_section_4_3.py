#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.3 Charts — 9 figures for dissertation.

fig_4_5  — Heatmap: Sharpe by method × strategy × timeframe
fig_4_6  — Cumulative returns: 6 MSP vs IMOEX (daily)
fig_4_7  — Bootstrap distributions ΔSharpe (daily + hourly)
fig_4_8  — Correlation matrices (daily + hourly)
fig_4_9  — Decomposition: strategy vs MMF (6 MSP × 2 tf)
fig_4_10 — Efficient frontier: Sharpe vs MDD (scaling MSP-BEST)
fig_4_11 — Cumulative: MSP-BEST (3x/5x) vs IMOEX
fig_4_12 — Drawdowns: MSP-BEST (3x daily) vs IMOEX
fig_4_13 — Decomposition at 3x: strategy + MMF + scaling

Usage: python3 scripts/plot_section_4_3.py
"""

import json
import urllib.request
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
TBL = BASE / "results" / "final" / "strategies" / "walkforward_v4" / "tables"
FIG_DIR = BASE / "dissertation_materials" / "chapter_4_3"
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


def _darker(hex_color, factor=0.65):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


CMAP_RYG = LinearSegmentedColormap.from_list(
    "ryg", ["#C0504D", "#FADC68", "#548235"], N=256
)

# MSP colours
MSP_COLORS = {
    "A": "#808080", "B": "#4472C4", "C": "#ED7D31",
    "D": "#548235", "MEAN": "#7030A0", "BEST": "#C00000",
}
MSP_LABELS = {
    "A": "МСП-A", "B": "МСП-B", "C": "МСП-C",
    "D": "МСП-D", "MEAN": "МСП-MEAN(BCD)", "BEST": "МСП-BEST",
}


# ══════════════════════════════════════════════════════════════════════
#  IMOEX fetch
# ══════════════════════════════════════════════════════════════════════
def fetch_imoex(start="2022-01-01", end="2026-02-20"):
    """Fetch IMOEX daily close from MOEX ISS."""
    base_url = (
        f"https://iss.moex.com/iss/history/engines/stock/markets/index/"
        f"securities/IMOEX.json?from={start}&till={end}&iss.meta=off"
    )
    all_rows, offset = [], 0
    while True:
        url = f"{base_url}&start={offset}"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = data["history"]["data"]
        if not rows:
            break
        all_rows.extend(rows)
        offset += 100

    columns = data["history"]["columns"]
    date_idx = columns.index("TRADEDATE")
    close_idx = columns.index("CLOSE")

    dates, closes = [], []
    for row in all_rows:
        if row[close_idx] is not None and row[close_idx] > 0:
            dates.append(row[date_idx])
            closes.append(float(row[close_idx]))

    return pd.Series(closes, index=pd.to_datetime(dates), name="IMOEX")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_5 — Heatmap: Sharpe by method × strategy × timeframe
# ══════════════════════════════════════════════════════════════════════
def fig_4_5():
    strats = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
    row_labels = ["S1 MeanRev", "S2 Bollinger", "S3 Donchian",
                  "S4 Supertrend", "S5 PivotPoints", "S6 VWAP"]
    methods = ["EW", "MinVar", "MaxSharpe", "InvVol"]

    d = pd.read_csv(TBL / "v4_portfolios_daily.csv")
    h = pd.read_csv(TBL / "v4_portfolios_hourly.csv")

    def get_matrix(df, comm):
        net = df[df["comm_level"] == comm]
        mat = np.zeros((len(strats), len(methods)))
        for i, s in enumerate(strats):
            for j, m in enumerate(methods):
                rows = net[(net["strategy"] == s) & (net["method"] == m)]
                mat[i, j] = rows["net_sharpe"].mean()
        return mat

    daily_mat = get_matrix(d, "net_0.05")
    hourly_mat = get_matrix(h, "net_0.04")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24 / 2.54, 10 / 2.54))

    for ax, mat, title in [
        (ax1, daily_mat, "Дневной (net 0,05%)"),
        (ax2, hourly_mat, "Часовой (net 0,04%)"),
    ]:
        vmin = min(0, np.nanmin(mat))
        vmax = np.nanmax(mat)
        im = ax.imshow(mat, cmap=CMAP_RYG, aspect="auto", vmin=vmin, vmax=vmax)

        for i in range(len(strats)):
            for j in range(len(methods)):
                val = mat[i, j]
                rgba = im.cmap(im.norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if lum > 0.45 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels if ax == ax1 else [""] * len(row_labels), fontsize=8)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10, pad=8)

    fig.tight_layout(w_pad=2)
    p = FIG_DIR / "fig_4_5_heatmap_methods.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_6 — Cumulative returns: 6 MSP vs IMOEX (daily)
# ══════════════════════════════════════════════════════════════════════
def fig_4_6(imoex):
    eq = pd.read_csv(TBL / "equity_curves_meta_all.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # MSPs
    col_map = {
        "META_A": "A", "META_B": "B", "META_C": "C",
        "META_D": "D", "META_MEANBCD": "MEAN", "META_BEST": "BEST",
    }
    for col, key in col_map.items():
        ret_pct = (eq[col] / eq[col].iloc[0] - 1) * 100
        ax.plot(eq.index, ret_pct, color=MSP_COLORS[key], lw=1.5,
                label=MSP_LABELS[key], zorder=3)

    # IMOEX
    imoex_aligned = imoex.reindex(eq.index, method="ffill")
    if imoex_aligned.notna().sum() > 0:
        imoex_ret = (imoex_aligned / imoex_aligned.dropna().iloc[0] - 1) * 100
        ax.plot(imoex_ret.index, imoex_ret.values, color="black", lw=1.8,
                ls="--", label="IMOEX", zorder=4)

    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.set_ylabel("Кумулятивная доходность, %")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="0.8",
              fontsize=8, ncol=2)
    hgrid(ax)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    fig.tight_layout()
    p = FIG_DIR / "fig_4_6_cumulative_msp.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_7 — Bootstrap ΔSharpe (CI ranges only, no bars)
# ══════════════════════════════════════════════════════════════════════
def fig_4_7():
    daily_data = [
        ("B vs A",    2.15, 1.30, 3.58),
        ("C vs A",    0.93, 0.51, 1.71),
        ("D vs A",    1.30, 0.91, 1.76),
        ("MEAN vs A", 1.54, 1.06, 2.38),
        ("BEST vs A", 2.15, 1.54, 3.08),
    ]
    hourly_data = [
        ("B vs A",    2.88, 1.79, 3.94),
        ("C vs A",    0.19, 0.04, 0.51),
        ("D vs A",    1.07, 0.40, 2.20),
        ("MEAN vs A", 0.89, 0.54, 1.49),
        ("BEST vs A", 3.15, 2.07, 4.20),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    colors = [MSP_COLORS[k] for k in ["B", "C", "D", "MEAN", "BEST"]]

    for ax, data, title in [
        (ax1, daily_data, "Дневной таймфрейм"),
        (ax2, hourly_data, "Часовой таймфрейм"),
    ]:
        labels = [d[0] for d in data]
        deltas = [d[1] for d in data]
        lo = [d[2] for d in data]
        hi = [d[3] for d in data]
        y = np.arange(len(labels))

        for i in range(len(data)):
            col = colors[i]
            # Shaded CI range
            ax.barh(y[i], hi[i] - lo[i], height=0.45, left=lo[i],
                    color=col, alpha=0.25, edgecolor="none", zorder=2)
            # Thick line for CI
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=col, lw=3,
                    solid_capstyle="round", zorder=3)
            # Diamond for point estimate
            ax.plot(deltas[i], y[i], "D", color=col, ms=7,
                    markeredgecolor=_darker(col, 0.5), markeredgewidth=0.8, zorder=5)
            # Value label
            ax.text(hi[i] + 0.08, y[i], f"+{deltas[i]:.2f}",
                    va="center", ha="left", fontsize=9, fontweight="bold", color=col)

        ax.axvline(0, color="black", lw=0.6, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("ΔSharpe (95% ДИ)")
        ax.set_title(title, fontsize=10, pad=8)
        ax.xaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.1, None)

    fig.tight_layout(w_pad=3)
    p = FIG_DIR / "fig_4_7_bootstrap_delta.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_8 — Correlation matrices (daily + hourly)
# ══════════════════════════════════════════════════════════════════════
def fig_4_8():
    d_ret = pd.read_csv(TBL / "strategy_returns_daily.csv", index_col=0, parse_dates=True)
    h_ret = pd.read_csv(TBL / "strategy_returns_hourly.csv", index_col=0, parse_dates=True)

    short_names = ["S1", "S2", "S3", "S4", "S5", "S6"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22 / 2.54, 10 / 2.54))

    for ax, df, title in [
        (ax1, d_ret, "Дневной таймфрейм"),
        (ax2, h_ret, "Часовой таймфрейм"),
    ]:
        corr = df.corr().values

        im = ax.imshow(corr, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="equal")
        for i in range(6):
            for j in range(6):
                val = corr[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

        ax.set_xticks(range(6))
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_yticks(range(6))
        ax.set_yticklabels(short_names if ax == ax1 else [""] * 6, fontsize=8)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10, pad=8)

    fig.tight_layout(w_pad=2)
    p = FIG_DIR / "fig_4_8_correlation.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_9 — Decomposition: strategy vs MMF (stacked bars)
# ══════════════════════════════════════════════════════════════════════
def fig_4_9():
    # Data from Table 4.16
    labels = ["МСП-A", "МСП-B", "МСП-C", "МСП-D", "МСП-MEAN", "МСП-BEST"]
    d_strat = [6.10, 3.56, 4.78, 7.16, 5.17, 7.32]
    d_mmf   = [7.41, 8.45, 8.15, 7.72, 8.10, 7.71]
    h_strat = [6.09, 3.15, 6.83, 7.14, 5.72, 3.48]
    h_mmf   = [7.91, 8.55, 8.30, 8.13, 8.33, 8.56]

    colors_strat = [MSP_COLORS[k] for k in ["A", "B", "C", "D", "MEAN", "BEST"]]
    col_mmf = "#D9E2F3"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, (ax, strat_vals, mmf_vals, title) in enumerate([
        (ax1, d_strat, d_mmf, "Дневной (1x + ФДР)"),
        (ax2, h_strat, h_mmf, "Часовой (1x + ФДР)"),
    ]):
        x = np.arange(len(labels))
        w = 0.6

        ax.bar(x, strat_vals, w, color=colors_strat,
               edgecolor=[_darker(c, 0.6) for c in colors_strat], lw=0.5,
               label="Стратегия", zorder=3)
        ax.bar(x, mmf_vals, w, bottom=strat_vals, color=col_mmf,
               edgecolor="#A0A0A0", lw=0.5, label="ФДР", zorder=3)

        # Total labels
        for i in range(len(labels)):
            total = strat_vals[i] + mmf_vals[i]
            ax.text(x[i], total + 0.15, f"{total:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=0, ha="center")
        ax.set_ylabel("Годовая доходность, %")
        ax.set_title(title, fontsize=10, pad=8)
        hgrid(ax)
        # Legend only on right panel
        if idx == 1:
            ax.legend(loc="upper right", frameon=True, fancybox=False,
                      edgecolor="0.8", fontsize=8)

    fig.tight_layout(w_pad=3)
    p = FIG_DIR / "fig_4_9_decomposition_1x.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_10 — Efficient frontier: Sharpe vs MDD (scaling MSP-BEST)
# ══════════════════════════════════════════════════════════════════════
def fig_4_10():
    # Data from Tables 4.17 / 4.18 (without 8x)
    daily_k   = [1, 2, 3, 5]
    daily_sh  = [5.44, 3.84, 3.31, 2.89]
    daily_mdd = [1.28, 3.14, 5.08, 9.54]
    daily_ret = [15.03, 21.24, 27.46, 39.93]

    hourly_k   = [1, 2, 3, 5]
    hourly_sh  = [11.52, 7.41, 6.00, 4.87]
    hourly_mdd = [0.17, 0.40, 0.64, 1.10]
    hourly_ret = [12.04, 15.30, 18.56, 25.07]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(daily_mdd, daily_sh, "o-", color="#1A237E", lw=2, ms=7,
            label="Дневной (МСП-BEST)", zorder=5)
    ax.plot(hourly_mdd, hourly_sh, "s-", color="#B71C1C", lw=2, ms=7,
            label="Часовой (МСП-BEST)", zorder=5)

    # Labels
    for k, mdd, sh in zip(daily_k, daily_mdd, daily_sh):
        ax.annotate(f"{k}x", (mdd, sh), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color="#1A237E", fontweight="bold")
    for k, mdd, sh in zip(hourly_k, hourly_mdd, hourly_sh):
        ax.annotate(f"{k}x", (mdd, sh), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color="#B71C1C", fontweight="bold")

    ax.set_xlabel("Максимальная просадка (MDD), %")
    ax.set_ylabel("Коэффициент Шарпа")
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.set_axisbelow(True)

    fig.tight_layout()
    p = FIG_DIR / "fig_4_10_efficient_frontier.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_11 — Cumulative: MSP-BEST (3x daily / 5x hourly) vs IMOEX
# ══════════════════════════════════════════════════════════════════════
def fig_4_11(imoex):
    eq_d = pd.read_csv(TBL / "equity_curves_meta_all.csv", index_col=0, parse_dates=True)
    eq_h = pd.read_csv(TBL / "equity_curves_meta_all_hourly.csv", index_col=0, parse_dates=True)

    # Scale BEST: daily 3x, hourly 5x
    # r_scaled = r_original * k + rfr * (1 - k*exposure)
    # Simplified: cumulative at kx ≈ (equity - 1) * k + 1 + rfr*(1-k*avg_exp)*years
    # Better: use daily returns * k + rfr_daily * (1 - k*avg_exp)
    # For simplicity, scale equity curve: r_i_scaled = k * r_i_original
    best_d = eq_d["META_BEST"]
    ret_d = best_d.pct_change().fillna(0)
    ret_d_3x = ret_d * 3
    eq_d_3x = (1 + ret_d_3x).cumprod()

    # Add MMF for free capital: daily rate ~ 14.5% - 1.5% = 13% / 252
    mmf_daily = 0.13 / 252
    avg_exp_d = 0.124  # from table
    mmf_contrib_d = mmf_daily * (1 - 3 * avg_exp_d)
    eq_d_3x_fdr = (1 + ret_d_3x + mmf_contrib_d).cumprod()

    best_h = eq_h["META_BEST"]
    ret_h = best_h.pct_change().fillna(0)
    ret_h_5x = ret_h * 5
    mmf_hourly = 0.13 / 2268
    avg_exp_h = 0.026
    mmf_contrib_h = mmf_hourly * (1 - 5 * avg_exp_h)
    eq_h_5x_fdr = (1 + ret_h_5x + mmf_contrib_h).cumprod()

    # Convert hourly to daily for plotting (take last value per day)
    eq_h_daily = eq_h_5x_fdr.copy()
    eq_h_daily.index = pd.to_datetime(eq_h_daily.index)
    eq_h_daily = eq_h_daily.groupby(eq_h_daily.index.date).last()
    eq_h_daily.index = pd.to_datetime(eq_h_daily.index)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # IMOEX
    imoex_start = imoex.loc[imoex.index >= eq_d.index[0]]
    imoex_norm = imoex_start / imoex_start.iloc[0]
    imoex_pct = (imoex_norm - 1) * 100
    ax.plot(imoex_pct.index, imoex_pct.values, color="black", lw=1.8,
            ls="--", label="IMOEX", zorder=4)

    # MSP-BEST daily 3x + FDR
    d_pct = (eq_d_3x_fdr / eq_d_3x_fdr.iloc[0] - 1) * 100
    d_pct.index = pd.to_datetime(d_pct.index)
    ax.plot(d_pct.index, d_pct.values, color="#1A237E", lw=1.8,
            label="МСП-BEST 3x + ФДР (дневной)", zorder=5)

    # MSP-BEST hourly 5x + FDR
    h_pct = (eq_h_daily / eq_h_daily.iloc[0] - 1) * 100
    ax.plot(h_pct.index, h_pct.values, color="#B71C1C", lw=1.8,
            label="МСП-BEST 5x + ФДР (часовой)", zorder=5)

    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.set_ylabel("Кумулятивная доходность, %")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="0.8",
              fontsize=9)
    hgrid(ax)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    fig.tight_layout()
    p = FIG_DIR / "fig_4_11_scaled_vs_imoex.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_12 — Drawdowns: MSP-BEST (3x daily) vs IMOEX (two panels)
# ══════════════════════════════════════════════════════════════════════
def fig_4_12(imoex):
    eq_d = pd.read_csv(TBL / "equity_curves_meta_all.csv", index_col=0, parse_dates=True)

    best_d = eq_d["META_BEST"]
    ret_d = best_d.pct_change().fillna(0)
    ret_d_3x = ret_d * 3
    mmf_daily = 0.13 / 252
    avg_exp_d = 0.124
    mmf_contrib_d = mmf_daily * (1 - 3 * avg_exp_d)
    eq_d_3x_fdr = (1 + ret_d_3x + mmf_contrib_d).cumprod()

    def drawdown(eq):
        peak = np.maximum.accumulate(eq)
        return (eq - peak) / peak * 100

    dates_d = pd.to_datetime(eq_d.index)
    dd_best = drawdown(eq_d_3x_fdr.values)

    imoex_start = imoex.loc[imoex.index >= dates_d[0]]
    dd_imoex = drawdown(imoex_start.values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 2.5]})

    # Top: MSP-BEST (small drawdowns, own scale)
    ax1.fill_between(dates_d, dd_best, 0, alpha=0.5, color="#1A237E", zorder=3)
    ax1.plot(dates_d, dd_best, color="#1A237E", lw=0.6, zorder=4)
    ax1.set_ylabel("МСП-BEST\n3x + ФДР, %", fontsize=9)
    ax1.set_ylim(min(dd_best) * 1.3, 0.5)
    ax1.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax1.set_axisbelow(True)
    ax1.text(0.01, 0.15, f"MDD = {min(dd_best):.1f}%", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", color="#1A237E")

    # Bottom: IMOEX (large drawdowns)
    ax2.fill_between(imoex_start.index, dd_imoex, 0, alpha=0.4, color="#C00000",
                     zorder=3)
    ax2.plot(imoex_start.index, dd_imoex, color="#C00000", lw=0.6, zorder=4)
    ax2.set_ylabel("IMOEX, %", fontsize=9)
    ax2.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)
    ax2.text(0.01, 0.08, f"MDD = {min(dd_imoex):.1f}%", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", color="#C00000")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    fig.tight_layout(h_pad=0.5)
    p = FIG_DIR / "fig_4_12_drawdowns.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  fig_4_13 — Decomposition at 3x: strategy + MMF + scaling
# ══════════════════════════════════════════════════════════════════════
def fig_4_13():
    # Data from Tables 4.19 (daily 3x+FDR) and hourly (5x+FDR)
    labels = ["МСП-A", "МСП-B", "МСП-C", "МСП-D", "МСП-MEAN", "МСП-BEST"]

    # Daily 3x+FDR returns = total. Strategy 1x + MMF 1x from table 4.16
    # Scaling effect = 3x_total - 1x_total
    d_1x = [13.51, 12.01, 12.93, 14.88, 13.27, 15.03]
    d_3x = [22.90, 18.41, 21.15, 27.02, 22.19, 27.46]
    d_strat = [6.10, 3.56, 4.78, 7.16, 5.17, 7.32]
    d_mmf_1x = [7.41, 8.45, 8.15, 7.72, 8.10, 7.71]

    h_1x = [14.00, 11.70, 15.13, 15.27, 14.05, 12.04]
    h_5x = [34.85, 23.34, 40.48, 41.22, 35.08, 25.07]
    h_strat = [6.09, 3.15, 6.83, 7.14, 5.72, 3.48]
    h_mmf_1x = [7.91, 8.55, 8.30, 8.13, 8.33, 8.56]

    colors_strat = [MSP_COLORS[k] for k in ["A", "B", "C", "D", "MEAN", "BEST"]]
    col_mmf = "#D9E2F3"
    col_scale = "#FFF2CC"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, (ax, strat, mmf, total_1x, total_kx, k_label, title) in enumerate([
        (ax1, d_strat, d_mmf_1x, d_1x, d_3x, "3x", "Дневной (3x + ФДР)"),
        (ax2, h_strat, h_mmf_1x, h_1x, h_5x, "5x", "Часовой (5x + ФДР)"),
    ]):
        x = np.arange(len(labels))
        w = 0.6
        scale = [total_kx[i] - total_1x[i] for i in range(len(labels))]

        ax.bar(x, strat, w, color=colors_strat,
               edgecolor=[_darker(c, 0.6) for c in colors_strat], lw=0.5,
               label="Стратегия (1x)", zorder=3)
        ax.bar(x, mmf, w, bottom=strat, color=col_mmf,
               edgecolor="#A0A0A0", lw=0.5, label="ФДР (1x)", zorder=3)
        ax.bar(x, scale, w, bottom=[strat[i] + mmf[i] for i in range(len(labels))],
               color=col_scale, edgecolor="#C0A000", lw=0.5,
               label=f"Эффект масштаб. ({k_label})", zorder=3)

        for i in range(len(labels)):
            ax.text(x[i], total_kx[i] + 0.3, f"{total_kx[i]:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=0, ha="center")
        ax.set_ylabel("Годовая доходность, %")
        ax.set_title(title, fontsize=10, pad=8)
        hgrid(ax)
        # Legend only on right panel
        if idx == 1:
            ax.legend(loc="upper right", frameon=True, fancybox=False,
                      edgecolor="0.8", fontsize=7.5)

    fig.tight_layout(w_pad=3)
    p = FIG_DIR / "fig_4_13_decomposition_scaled.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    print("Generating Section 4.3 figures ...")

    # Fetch IMOEX once
    print("  Fetching IMOEX...")
    imoex = fetch_imoex()
    print(f"  IMOEX: {len(imoex)} days")

    fig_4_5()
    fig_4_6(imoex)
    fig_4_7()
    fig_4_8()
    fig_4_9()
    fig_4_10()
    fig_4_11(imoex)
    fig_4_12(imoex)
    fig_4_13()

    print(f"\nAll saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
