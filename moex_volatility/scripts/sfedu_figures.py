#!/usr/bin/env python3
"""
2 figures for SFedU conference "Digital Ecosystem of Economy".
Compact (width 1400-1600 px at 300 dpi ≈ 4.7-5.3 inches), white bg, Russian labels.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
})

BASE = Path("/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility")
OUT = BASE / "sfedu_figures"
DR = BASE / "results/final/strategies/msp/perstrat_daily_returns"

C_A = "#7f7f7f"
C_B = "#1f77b4"
C_C = "#ff7f0e"
C_D = "#2ca02c"
C_IMOEX = "#d62728"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {name}")


# ══════════════════════════════════════════════════════════════════════
#  Fig 1 — Cumulative returns MSP A/B/C/D + IMOEX (real data)
# ══════════════════════════════════════════════════════════════════════
def fig1():
    # ~1500px / 300dpi = 5.0 inches width
    msp = {}
    for app in ["A", "B", "C", "D"]:
        msp[app] = pd.read_csv(DR / f"MSP_{app}_inst.csv", parse_dates=["date"]).set_index("date")

    features = pd.read_parquet(BASE / "data/features/features_all.parquet")
    imoex = features[["date", "close_IMOEX"]].drop_duplicates("date").sort_values("date")
    imoex["date"] = pd.to_datetime(imoex["date"])
    imoex = imoex.set_index("date")
    imoex["log_ret"] = np.log(imoex["close_IMOEX"] / imoex["close_IMOEX"].shift(1))
    imoex = imoex.dropna()

    common = msp["A"].index
    for app in ["B", "C", "D"]:
        common = common.intersection(msp[app].index)
    common = common.intersection(imoex.index)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    styles = {
        "A": (C_A, "-", 1.0, "МСП-A"),
        "B": (C_B, "--", 0.9, "МСП-B"),
        "C": (C_C, "-", 1.4, "МСП-C"),
        "D": (C_D, "-", 1.4, "МСП-D"),
    }
    for app in ["A", "B", "C", "D"]:
        cum = np.cumsum(msp[app].loc[common, "total_ret"].values) * 100
        clr, ls, lw, lbl = styles[app]
        ax.plot(common, cum, color=clr, linewidth=lw, linestyle=ls, label=lbl)

    imoex_cum = np.cumsum(imoex.loc[common, "log_ret"].values) * 100
    ax.plot(common, imoex_cum, color=C_IMOEX, linewidth=1.0, linestyle="--", label="IMOEX")

    ax.axhline(0, color="#cccccc", linewidth=0.4, zorder=0)
    ax.set_ylabel("Кумулятивная доходность (%)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.5)
    ax.grid(True, color="#eeeeee", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#333")

    fig.tight_layout(pad=0.5)
    save(fig, "sfedu_fig1_cumulative.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 2 — Approaches A/B/C/D grouped bar (2×2 panels)
# ══════════════════════════════════════════════════════════════════════
def fig2():
    approaches = ["A", "B", "C", "D"]
    colors = [C_A, C_B, C_C, C_D]
    ret = [7.6, 7.3, 7.9, 8.4]
    exp = [15.8, 15.6, 7.4, 12.2]
    mdd_pos = [13.4, 12.6, 7.9, 9.5]
    wr = [56.5, 56.5, 60.0, 61.2]

    panels = [
        ("Доходность (%)", ret),
        ("Экспозиция (%)", exp),
        ("|MDD| (%)", mdd_pos),
        ("Доля прибыльных (%)", wr),
    ]
    x = np.arange(4)

    fig, axes = plt.subplots(2, 2, figsize=(5.0, 3.5))
    for ax, (title, vals) in zip(axes.flat, panels):
        bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="black",
                      linewidth=0.2, alpha=0.85, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=9, pad=4)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color="#333")
        ax.set_ylim(0, max(vals) * 1.16)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("#555")

    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor=c, edgecolor="black", lw=0.3, label=f"{a}")
                 for a, c in zip(approaches, colors)]
    fig.legend(handles=legend_el, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=1.0, w_pad=1.0)
    save(fig, "sfedu_fig2_approaches.png")


if __name__ == "__main__":
    fig1()
    fig2()
    print("\nAll 2 SFedU figures saved.")
