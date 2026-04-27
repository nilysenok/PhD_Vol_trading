#!/usr/bin/env python3
"""
Figures 4.5–4.9 for dissertation chapter 4.
Academic style, white background, Times New Roman 12pt, PNG 300 DPI, figsize=(8,6).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import sqrt
from pathlib import Path

# ── Style ────────────────────────────────────────────────────────────
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
OUT = BASE / "results/final/strategies/figures"
OUT.mkdir(parents=True, exist_ok=True)
DR = BASE / "results/final/strategies/msp/perstrat_daily_returns"

COLORS = {"A": "#888888", "B": "#1f4e8c", "C": "#2e7d32", "D": "#b22222"}
BPY = 252


def sharpe(xs):
    s = np.std(xs, ddof=1)
    return np.mean(xs) * sqrt(BPY) / s if s > 1e-12 else 0.0


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.5 — Grouped bar: per-strategy excess Sharpe by approach
# ══════════════════════════════════════════════════════════════════════
def fig_4_5():
    strats = ["S1", "S2", "S3", "S4", "S5", "S6"]
    data = {
        "A": [2.00, 1.98, 0.62, 0.92, 1.28, 2.06],
        "B": [1.93, 1.88, 0.65, 1.01, 1.23, 2.03],
        "C": [2.57, 2.79, 1.36, 1.62, 1.39, 2.31],
        "D": [2.47, 2.32, 1.81, 1.96, 1.39, 2.47],
    }
    approaches = ["A", "B", "C", "D"]
    n_strat = len(strats)
    n_app = len(approaches)
    x = np.arange(n_strat)
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 6))
    for j, app in enumerate(approaches):
        offset = (j - 1.5) * width
        bars = ax.bar(x + offset, data[app], width, color=COLORS[app],
                      edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3,
                      label=f"Подход {app}")
        for bar, v in zip(bars, data[app]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(strats, fontsize=11)
    ax.set_ylabel("Шарп", fontsize=12)
    ax.set_ylim(0, 3.2)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333")

    ax.legend(loc="upper center", ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Рис. 4.5. Шарп портфелей стратегий по подходам A, B, C, D",
        fontsize=11, y=0.02, va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(OUT / "fig_4_5.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved fig_4_5.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.6 — Correlation heatmap 6×6
# ══════════════════════════════════════════════════════════════════════
def fig_4_6():
    labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
    corr = np.array([
        [1.00, 0.67, -0.24, -0.16, 0.20, 0.16],
        [0.67, 1.00, -0.22, -0.11, 0.19, 0.20],
        [-0.24, -0.22, 1.00, 0.86, -0.10, -0.16],
        [-0.16, -0.11, 0.86, 1.00, -0.03, -0.15],
        [0.20, 0.19, -0.10, -0.03, 1.00, 0.21],
        [0.16, 0.20, -0.16, -0.15, 0.21, 1.00],
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(6):
        for j in range(6):
            v = corr[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold" if i == j else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Корреляция", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Рис. 4.6. Корреляционная матрица дневных доходностей стратегий (подход A)",
        fontsize=11, y=0.02, va="bottom")
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(OUT / "fig_4_6.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved fig_4_6.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.7 — Bootstrap distribution of ΔSharpe (C>A, D>A)
# ══════════════════════════════════════════════════════════════════════
def fig_4_7():
    a = pd.read_csv(DR / "MSP_A_inst.csv", parse_dates=["date"]).set_index("date")
    c = pd.read_csv(DR / "MSP_C_inst.csv", parse_dates=["date"]).set_index("date")
    d = pd.read_csv(DR / "MSP_D_inst.csv", parse_dates=["date"]).set_index("date")

    common = a.index.intersection(c.index).intersection(d.index)
    ra = a.loc[common, "excess_ret"].values
    rc = c.loc[common, "excess_ret"].values
    rd = d.loc[common, "excess_ret"].values
    n = len(ra)

    N_BOOT = 10_000
    np.random.seed(42)
    deltas_c = np.empty(N_BOOT)
    deltas_d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.random.randint(0, n, n)
        deltas_c[i] = sharpe(rc[idx]) - sharpe(ra[idx])
        deltas_d[i] = sharpe(rd[idx]) - sharpe(ra[idx])

    obs_c = sharpe(rc) - sharpe(ra)
    obs_d = sharpe(rd) - sharpe(ra)
    p_c = np.mean(deltas_c <= 0)
    p_d = np.mean(deltas_d <= 0)

    fig, ax = plt.subplots(figsize=(8, 6))

    bins = np.linspace(-0.5, 3.0, 60)
    ax.hist(deltas_c, bins=bins, alpha=0.5, color=COLORS["C"], edgecolor="none",
            label=f"C>A (ΔSh={obs_c:+.2f}, p<0.001)", density=True)
    ax.hist(deltas_d, bins=bins, alpha=0.5, color=COLORS["D"], edgecolor="none",
            label=f"D>A (ΔSh={obs_d:+.2f}, p<0.001)", density=True)

    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0, zorder=4)
    ax.axvline(obs_c, color=COLORS["C"], linestyle="-", linewidth=1.5, alpha=0.7)
    ax.axvline(obs_d, color=COLORS["D"], linestyle="-", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("ΔSharpe", fontsize=12)
    ax.set_ylabel("Плотность", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333")

    fig.suptitle(
        "Рис. 4.7. Распределение bootstrap-оценок разности\n"
        "коэффициентов Шарпа (N = 10 000)",
        fontsize=11, y=0.02, va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(OUT / "fig_4_7.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved fig_4_7.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.8 — Cumulative return MSP-A vs IMOEX
# ══════════════════════════════════════════════════════════════════════
def fig_4_8():
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

    imoex_cum = np.cumsum(imoex.loc[common, "log_ret"].values) * 100
    dates = common

    fig, ax = plt.subplots(figsize=(8, 6))

    msp_styles = {
        "A": ("#888888", "-", 1.3, "МСП-A"),
        "B": ("#1f4e8c", "--", 1.1, "МСП-B"),
        "C": ("#2e7d32", "-", 1.5, "МСП-C"),
        "D": ("#b22222", "-", 1.5, "МСП-D"),
    }

    finals = {}
    for app in ["A", "B", "C", "D"]:
        cum = np.cumsum(msp[app].loc[common, "total_ret"].values) * 100
        clr, ls, lw, lbl = msp_styles[app]
        ax.plot(dates, cum, color=clr, linewidth=lw, linestyle=ls, label=lbl)
        finals[app] = cum[-1]

    ax.plot(dates, imoex_cum, color="#555555", linewidth=1.4, linestyle="--",
            label="IMOEX")

    # Annotate with just letters, close to lines
    offsets_y = {"C": 5, "D": -3, "A": 0, "B": -10}
    for app in ["C", "D", "A", "B"]:
        clr = msp_styles[app][0]
        ax.annotate(app,
                    xy=(dates[-1], finals[app]),
                    xytext=(6, offsets_y[app]), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=clr)

    # IMOEX label in legend only

    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.set_ylabel("Кумулятивная доходность (%)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, color="#eeeeee", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333")

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Рис. 4.8. Кумулятивная доходность МСП (подходы A, B, C, D) и индекса IMOEX, 2022–2026",
        fontsize=10, y=0.02, va="bottom")
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(OUT / "fig_4_8.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved fig_4_8.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.9 — ΔCE by γ for B/C/D
# ══════════════════════════════════════════════════════════════════════
def fig_4_9():
    gammas = [3, 5, 10]
    ce_b = [-0.09, -0.05, 0.04]
    ce_c = [1.77, 1.86, 2.07]
    ce_d = [1.46, 1.56, 1.79]

    # p-values
    p_b = [0.59, 0.55, 0.45]
    p_c = [0.03, 0.02, 0.01]
    p_d = [0.07, 0.05, 0.02]

    fig, ax = plt.subplots(figsize=(8, 6))

    for vals, pvals, app, ls in [
        (ce_b, p_b, "B", "--"),
        (ce_c, p_c, "C", "-"),
        (ce_d, p_d, "D", "-"),
    ]:
        ax.plot(gammas, vals, color=COLORS[app], linewidth=1.8, linestyle=ls,
                label=f"Подход {app}", zorder=3)
        for g, v, p in zip(gammas, vals, pvals):
            ax.plot(g, v, marker="o", color=COLORS[app], markersize=7,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=4)

    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_xlabel("Коэффициент неприятия риска (γ)", fontsize=12)
    ax.set_ylabel("ΔCE (% годовых)", fontsize=12)
    ax.set_xticks(gammas)
    ax.set_xlim(1.5, 11.5)
    ax.set_ylim(-0.8, 2.8)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, color="#eeeeee", linewidth=0.5)
    ax.set_axisbelow(True)


    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333")

    fig.suptitle(
        "Рис. 4.9. Экономическая ценность подходов B, C, D\n"
        "при различных уровнях неприятия риска",
        fontsize=11, y=0.02, va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(OUT / "fig_4_9.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved fig_4_9.png")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_4_5()
    fig_4_6()
    fig_4_7()
    fig_4_8()
    fig_4_9()
    print("\nAll 5 figures saved.")
