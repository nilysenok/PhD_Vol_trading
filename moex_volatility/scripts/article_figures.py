#!/usr/bin/env python3
"""
4 figures for journal article. PNG 300 dpi, white bg, Russian labels.
NO "Рисунок N" captions on figures themselves.
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
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
})

BASE = Path("/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility")
OUT = BASE / "article_figures"
DR = BASE / "results/final/strategies/msp/perstrat_daily_returns"

C_A = "#888888"
C_B = "#4a90d9"
C_C = "#e67e22"
C_D = "#27ae60"
C_IMOEX = "#cc3333"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {name}")


# ══════════════════════════════════════════════════════════════════════
#  Fig 3 — Approaches A/B/C/D grouped bar (2×2 panels)
# ══════════════════════════════════════════════════════════════════════
def fig3():
    approaches = ["A", "B", "C", "D"]
    colors = [C_A, C_B, C_C, C_D]
    ret = [7.6, 7.3, 7.9, 8.4]
    exp = [15.8, 15.6, 7.4, 12.2]
    mdd_pos = [13.4, 12.6, 7.9, 9.5]  # positive
    wr = [56.5, 56.5, 60.0, 61.2]

    panels = [
        ("Доходность (%)", ret),
        ("Экспозиция (%)", exp),
        ("|MDD| (%)", mdd_pos),
        ("Доля прибыльных (%)", wr),
    ]
    x = np.arange(4)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (title, vals) in zip(axes.flat, panels):
        bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="black",
                      linewidth=0.3, alpha=0.85, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=12, pad=6)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="#333")
        ax.set_ylim(0, max(vals) * 1.15)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#555")

    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor=c, edgecolor="black", lw=0.4, label=f"Подход {a}")
                 for a, c in zip(approaches, colors)]
    fig.legend(handles=legend_el, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "article_fig3_approaches.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4 — Cumulative returns MSP A/B/C/D + IMOEX (real data)
# ══════════════════════════════════════════════════════════════════════
def fig4():
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

    fig, ax = plt.subplots(figsize=(8, 6))

    styles = {
        "A": (C_A, "-", 1.3, "МСП-A"),
        "B": (C_B, "--", 1.2, "МСП-B"),
        "C": (C_C, "-", 1.8, "МСП-C"),
        "D": (C_D, "-", 1.8, "МСП-D"),
    }
    for app in ["A", "B", "C", "D"]:
        cum = np.cumsum(msp[app].loc[common, "total_ret"].values) * 100
        clr, ls, lw, lbl = styles[app]
        ax.plot(common, cum, color=clr, linewidth=lw, linestyle=ls, label=lbl)

    imoex_cum = np.cumsum(imoex.loc[common, "log_ret"].values) * 100
    ax.plot(common, imoex_cum, color=C_IMOEX, linewidth=1.3, linestyle="--", label="IMOEX")

    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.set_ylabel("Кумулятивная доходность (%)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, color="#eeeeee", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("#333")

    save(fig, "article_fig4_cumulative.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 5 — Correlation heatmap 6×6 (real data)
# ══════════════════════════════════════════════════════════════════════
def fig5():
    # Load real daily returns and compute correlation
    strats = ["S1", "S2", "S3", "S4", "S5", "S6"]
    rets = {}
    import os
    for s in strats:
        f = DR / f"{s}_A_inst.csv"
        if os.path.exists(f):
            df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            rets[s] = df["trading_ret"]
    rdf = pd.DataFrame(rets)
    corr = rdf.corr()

    labels = ["S1 MeanRev", "S2 Bollinger", "S3 Donchian",
              "S4 Supertrend", "S5 Pivot", "S6 VWAP"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(6):
        for j in range(6):
            v = corr.values[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            fw = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight=fw)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Корреляция", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    avg_offdiag = corr.values[np.triu_indices(6, k=1)].mean()
    ax.set_title(f"Средняя попарная корреляция: {avg_offdiag:.2f}", fontsize=11, pad=10)

    fig.tight_layout()
    save(fig, "article_fig5_correlation.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 6 — QLIKE models by horizon (grouped bar)
# ══════════════════════════════════════════════════════════════════════
def fig6():
    models = ["HAR-J", "XGBoost", "LightGBM", "Гибридная"]
    colors = [C_A, C_B, C_C, C_D]
    data = {
        "HAR-J":      [0.524, 0.796, 0.961],
        "XGBoost":    [0.400, 0.956, 1.127],
        "LightGBM":   [0.378, 0.821, 0.870],
        "Гибридная":  [0.376, 0.762, 0.888],
    }
    horizons = ["h=1", "h=5", "h=22"]
    n_models = len(models)
    n_hor = len(horizons)
    x = np.arange(n_hor)
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 6))
    for j, model in enumerate(models):
        offset = (j - 1.5) * width
        vals = data[model]
        bars = ax.bar(x + offset, vals, width, color=colors[j],
                      edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3,
                      label=model)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=12)
    ax.set_ylabel("QLIKE (меньше = лучше)", fontsize=12)
    ax.set_ylim(0, 1.3)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("#333")

    fig.tight_layout()
    save(fig, "article_fig6_qlike.png")


if __name__ == "__main__":
    fig3()
    fig4()
    fig5()
    fig6()
    print("\nAll 4 article figures saved.")
