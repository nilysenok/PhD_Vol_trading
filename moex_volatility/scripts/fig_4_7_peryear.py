#!/usr/bin/env python3
"""
Fig 4.7 — Per-year bar chart of MSP Sharpe by approach + key rate on secondary axis.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

years = ["2022", "2023", "2024", "2025"]
data = {
    "A": [1.68, 2.99, 1.56, 0.40],
    "B": [1.82, 2.97, 1.78, 0.49],
    "C": [2.86, 5.09, 2.98, 1.87],
    "D": [3.69, 3.89, 2.29, 1.12],
}
kr = [10.6, 9.9, 17.4, 21.0]

colors = {"A": "#888888", "B": "#1f4e8c", "C": "#2e7d32", "D": "#b22222"}
approaches = ["A", "B", "C", "D"]

x = np.arange(len(years))
width = 0.18

fig, ax1 = plt.subplots(figsize=(10, 6))

for j, app in enumerate(approaches):
    offset = (j - 1.5) * width
    bars = ax1.bar(x + offset, data[app], width, color=colors[app],
                   edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3,
                   label=f"Подход {app}")
    for bar, v in zip(bars, data[app]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333")

ax1.set_xticks(x)
ax1.set_xticklabels(years, fontsize=12)
ax1.set_ylabel("Шарп", fontsize=12)
ax1.set_ylim(0, 5.8)
ax1.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#333")

ax1.legend(loc="upper center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.07))

fig.suptitle(
    "Рис. 4.7. Коэффициент Шарпа МСП по годам и подходам",
    fontsize=11, y=0.02, va="bottom")
fig.tight_layout(rect=[0, 0.06, 1, 0.97])
fig.savefig(f"{OUT}/fig_4_7_peryear.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {OUT}/fig_4_7_peryear.png")
