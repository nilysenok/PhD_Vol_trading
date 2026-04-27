#!/usr/bin/env python3
"""
Figures 4.3 (scatter) and 4.4 (2×2 bars) for dissertation section 4.2.
Approaches A/B/C/D, mean across 6 strategies × 17 tickers, comm=0.05%, OOS 2022-2026.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
approaches = ["A", "B", "C", "D"]
ret = np.array([7.6, 7.3, 7.9, 8.4])
exp = np.array([15.8, 15.6, 7.4, 12.2])
mdd = np.array([-13.4, -12.6, -7.9, -9.5])
wr  = np.array([56.5, 56.5, 60.0, 61.2])

colors = ["#888888", "#1f4e8c", "#2e7d32", "#b22222"]

# ══════════════════════════════════════════════════════════════════════
#  Fig 4.3 — Scatter: Доходность × |MDD|
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 6))

abs_mdd = np.abs(mdd)

# Size proportional to 1/exposure
inv_exp = 1.0 / exp
sizes = 150 + (inv_exp - inv_exp.min()) / (inv_exp.max() - inv_exp.min()) * 500

for i in range(4):
    ax.scatter(abs_mdd[i], ret[i], s=sizes[i], c=colors[i],
               edgecolors="black", linewidths=0.7, alpha=0.9, zorder=5)

# Labels
offsets = {"A": (14, -12), "B": (10, 6), "C": (-12, 14), "D": (-18, -18)}
for i, app in enumerate(approaches):
    dx, dy = offsets[app]
    ax.annotate(app, (abs_mdd[i], ret[i]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=14, fontweight="bold", color=colors[i])


ax.set_xlabel("|MDD| (%)", fontsize=12)
ax.set_ylabel("Доходность (%)", fontsize=12)
ax.set_xlim(5, 16)
ax.set_ylim(6.5, 9.5)

ax.grid(True, color="#eeeeee", linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#333333")

# Legend
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i],
           markeredgecolor="black", markersize=10, label=f"Подход {app}")
    for i, app in enumerate(approaches)
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

fig.suptitle(
    "Рис. 4.3. Доходность и максимальная просадка подходов A, B, C, D\n"
    "(размер точки обратно пропорционален экспозиции)",
    fontsize=11, y=0.02, va="bottom",
)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(f"{OUT}/fig_4_3_scatter.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {OUT}/fig_4_3_scatter.png")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.4 — 2×2 bar subplots
# ══════════════════════════════════════════════════════════════════════

stability = 100 - np.abs(mdd)

panels = [
    ("Доходность (%)", ret),
    ("Экспозиция (%)", exp),
    ("Устойчивость (100 − |MDD|)", stability),
    ("Доля прибыльных (%)", wr),
]

x = np.arange(4)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for idx, (ax, (title, vals)) in enumerate(zip(axes.flat, panels)):
    bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="black",
                  linewidth=0.4, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Value labels
    for bar, v in zip(bars, vals):
        fmt = f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt, ha="center", va="bottom", fontsize=9, color="#333333")

    ymax = max(vals) * 1.15
    ax.set_ylim(0, ymax)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#555555")

# Legend
legend_elements = [
    Patch(facecolor=colors[i], edgecolor="black", linewidth=0.4,
          label=f"Подход {app}")
    for i, app in enumerate(approaches)
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=9, frameon=True, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Рис. 4.4. Сравнительные метрики подходов A, B, C, D\n"
    "(средние по 17 тикерам, 2022–2026)",
    fontsize=11, y=1.01,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.98])
fig.savefig(f"{OUT}/fig_4_4_bars.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {OUT}/fig_4_4_bars.png")
