#!/usr/bin/env python3
"""
Figures 4.1 (radar) and 4.2 (scatter) for dissertation section 4.1.
Per-ticker approach A, comm=0.05%, daily, mean across 17 tickers, OOS 2022-2026.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

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

OUT = "/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility/results/final/strategies/figures"

# ── Data ─────────────────────────────────────────────────────────────
strategies = ["S1", "S2", "S3", "S4", "S5", "S6"]
labels_ru = [
    "S1 (контртренд)", "S2 (контртренд)",
    "S3 (тренд)", "S4 (тренд)",
    "S5 (диапазон)", "S6 (диапазон)",
]

ret  = np.array([4.3, 4.4, 16.4, 14.8, 1.8, 3.7])
exp  = np.array([4.1, 3.9, 50.7, 31.6, 1.1, 3.1])
wr   = np.array([57.1, 58.8, 51.6, 52.2, 57.8, 61.4])
mdd  = np.array([-6.0, -4.8, -34.9, -28.0, -1.9, -4.7])
tpy  = np.array([2.1, 1.6, 4.0, 2.8, 0.7, 2.0])

# Categories: contrarian=blue, trend=red, range=green
cat_colors = ["#1f4e8c", "#1f4e8c", "#b22222", "#b22222", "#2e7d32", "#2e7d32"]
cat_lines  = ["-", "--", "-", "--", "-", "--"]

# ══════════════════════════════════════════════════════════════════════
#  Fig 4.1 — 2×2 grouped bar chart (replaces radar)
# ══════════════════════════════════════════════════════════════════════

stability = 100 - np.abs(mdd)  # higher = better

panels = [
    ("Доходность (%)", ret),
    ("Экспозиция (%)", exp),
    ("Устойчивость (100 − |MDD|)", stability),
    ("Частота сделок (сделок/год)", tpy),
]

bar_labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
x = np.arange(6)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for idx, (ax, (title, vals)) in enumerate(zip(axes.flat, panels)):
    bars = ax.bar(x, vals, width=0.65, color=cat_colors, edgecolor="black",
                  linewidth=0.4, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Value labels on top of bars
    for bar, v in zip(bars, vals):
        fmt = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt, ha="center", va="bottom", fontsize=8, color="#333333")

    # Y-axis padding
    ymax = max(vals) * 1.18
    ax.set_ylim(0, ymax)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#555555")

# Legend (one for whole figure)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1f4e8c", edgecolor="black", linewidth=0.4, label="Контртренд (S1, S2)"),
    Patch(facecolor="#b22222", edgecolor="black", linewidth=0.4, label="Тренд (S3, S4)"),
    Patch(facecolor="#2e7d32", edgecolor="black", linewidth=0.4, label="Диапазон (S5, S6)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=9, frameon=True, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Рис. 4.1. Профиль стратегий подхода A (средние по 17 тикерам, 2022–2026)",
    fontsize=11, y=1.01,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.98])
fig.savefig(f"{OUT}/fig_4_1_radar.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {OUT}/fig_4_1_radar.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.2 — Scatter plot
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 6))

# Point size proportional to 1/|MDD|, scaled for visibility
inv_mdd = 1.0 / np.abs(mdd)
sizes = 80 + (inv_mdd - inv_mdd.min()) / (inv_mdd.max() - inv_mdd.min()) * 600

for i in range(6):
    ax.scatter(exp[i], ret[i], s=sizes[i], c=cat_colors[i],
               edgecolors="black", linewidths=0.6, alpha=0.85, zorder=5)

# Labels closer to points
offsets = {
    "S1": (18, 16), "S2": (18, -14), "S3": (12, 8),
    "S4": (-22, -16), "S5": (18, -14), "S6": (-28, 14),
}
for i, s in enumerate(strategies):
    dx, dy = offsets[s]
    ax.annotate(
        s, (exp[i], ret[i]),
        textcoords="offset points", xytext=(dx, dy),
        fontsize=11, fontweight="bold", color=cat_colors[i],
    )

ax.set_xlabel("Экспозиция (%)", fontsize=12)
ax.set_ylabel("Доходность (%)", fontsize=12)
ax.set_xlim(-3, 56)
ax.set_ylim(-1, 20)

# Light grid
ax.grid(True, color="#eeeeee", linewidth=0.5)
ax.set_axisbelow(True)

# Frame
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#333333")

# Legend in bottom-right corner
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f4e8c",
           markeredgecolor="black", markersize=10, label="Контртренд (S1, S2)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#b22222",
           markeredgecolor="black", markersize=10, label="Тренд (S3, S4)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32",
           markeredgecolor="black", markersize=10, label="Диапазон (S5, S6)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
          frameon=True, edgecolor="#cccccc", fancybox=False)

fig.suptitle(
    "Рис. 4.2. Доходность и экспозиция стратегий подхода A\n"
    "(размер точки обратно пропорционален MDD)",
    fontsize=12, y=0.02, va="bottom",
)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(f"{OUT}/fig_4_2_scatter.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {OUT}/fig_4_2_scatter.png")
