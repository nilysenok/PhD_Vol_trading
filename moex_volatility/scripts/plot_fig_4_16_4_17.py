#!/usr/bin/env python3
"""
Рис. 4.16 — Δ(γ) и пороги безубыточности (2 линии: daily AVG + hourly AVG)
Рис. 4.17 — Waterfall-декомпозиция полезности (1×2: daily AVG / hourly AVG)

Данные из calc_msp_avg_delta.py (3x daily + ФДР, 5x hourly + ФДР, γ = 1..15).
МСП-AVG = MEAN(BCD) — среднее по трём прогнозным подходам (без look-ahead bias).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "charts_4_3")
os.makedirs(OUT, exist_ok=True)

# ── Font setup ───────────────────────────────────────────────────
for font in ["Times New Roman", "DejaVu Serif"]:
    try:
        plt.rcParams.update({"font.family": "serif", "font.serif": [font]})
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, "test")
        plt.close(fig_test)
        break
    except Exception:
        continue

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def comma_fmt(decimals=1):
    def _fmt(x, pos):
        return f"{x:.{decimals}f}".replace(".", ",")
    return mticker.FuncFormatter(_fmt)


# ═══════════════════════════════════════════════════════════════════
# DATA (from calc_msp_avg_delta.py)
# Linear: Δ(γ) = a + (γ/2) * b
# where a = ΔE(r) in п.п., b = Δσ² × 100
# Only AVG (= MEAN(BCD), no look-ahead bias)
# ═══════════════════════════════════════════════════════════════════
DELTA_PARAMS = {
    # (a = ΔE(r) п.п., b = Δσ²×100)
    "daily_avg":   (0.45,  1.2683),
    "hourly_avg":  (0.72,  2.3758),
}

# Waterfall values at γ = 5
WATERFALL = {
    # (U_A, ΔE(r), Δrisk, U_PORT)
    "daily_avg":   (23.76,  +0.45, +3.17, 27.38),
    "hourly_avg":  (24.81,  +0.72, +5.94, 31.46),
}


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.16 — Δ(γ) с порогами безубыточности (только AVG)
# ═══════════════════════════════════════════════════════════════════
def fig_4_16():
    gamma = np.arange(1, 16)

    d_avg = DELTA_PARAMS["daily_avg"]
    h_avg = DELTA_PARAMS["hourly_avg"]

    delta_d_avg = d_avg[0] + (gamma / 2) * d_avg[1]
    delta_h_avg = h_avg[0] + (gamma / 2) * h_avg[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gamma, delta_d_avg, "o-", color="#1A237E", linewidth=2.5,
            markersize=6, label="Дневной таймфрейм (3x + ФДР)", zorder=5)
    ax.plot(gamma, delta_h_avg, "s-", color="#B71C1C", linewidth=2.5,
            markersize=6, label="Часовой таймфрейм (5x + ФДР)", zorder=5)

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    # Threshold lines (3 mln / capital × 100%)
    thresholds = [
        (6.0,  "капитал 50 млн",  12.5),
        (3.0,  "капитал 100 млн", 12.5),
        (1.5,  "капитал 200 млн", 12.5),
        (0.6,  "капитал 500 млн", 12.5),
    ]
    bbox = dict(boxstyle="round,pad=0.2", facecolor="white",
                edgecolor="none", alpha=0.85)
    for level, label, x_pos in thresholds:
        ax.axhline(level, color="#999999", linewidth=0.9, linestyle="--",
                   alpha=0.5, zorder=1)
        ax.text(x_pos, level, label, va="center", ha="center", fontsize=9,
                color="#666666", bbox=bbox, zorder=4)

    ax.set_xlabel("Коэффициент неприятия риска γ")
    ax.set_ylabel("Экономическая ценность Δ, п.п.")
    ax.set_xticks(gamma)
    ax.set_xlim(0.5, 15.5)
    ax.yaxis.set_major_formatter(comma_fmt(0))

    # Light grid
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_16.png"))
    plt.close(fig)
    print("  ✓ fig_4_16.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.17 — Waterfall-декомпозиция (γ = 5), 1×2 (только AVG)
# ═══════════════════════════════════════════════════════════════════
def _draw_waterfall(ax, u_a, delta_er, delta_risk, u_port, title):
    """Draw a single waterfall: U(A) → ΔE(r) → Δrisk → U(AVG)."""

    labels = ["U(МСП-A)", "ΔE(r)", "−Δ(γ/2)σ²", "U(МСП-\nAVG)"]

    # After ΔE(r): cumulative = u_a + delta_er
    after_er = u_a + delta_er

    # Bar heights and bottoms
    if delta_er >= 0:
        er_bottom = u_a
        er_height = delta_er
        er_color = "#4CAF50"  # green
    else:
        er_bottom = after_er
        er_height = abs(delta_er)
        er_color = "#E53935"  # red

    # Risk component (always positive in our data)
    risk_bottom = after_er
    risk_height = delta_risk
    risk_color = "#388E3C"  # dark green

    vals = [u_a, er_height, risk_height, u_port]
    bottoms = [0, er_bottom, risk_bottom, 0]
    colors = ["#9E9E9E", er_color, risk_color, "#1565C0"]
    display_vals = [u_a, delta_er, delta_risk, u_port]

    x = np.arange(4)
    bar_w = 0.55

    bars = ax.bar(x, vals, bar_w, bottom=bottoms, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # Connector lines at cumulative values
    cumulatives = [u_a, after_er, u_port]
    for i in range(3):
        ax.plot([x[i] + bar_w/2, x[i+1] - bar_w/2],
                [cumulatives[i], cumulatives[i]],
                color="#666666", linewidth=0.8, linestyle=":", zorder=2)

    # Value labels
    for i, (bar, dv) in enumerate(zip(bars, display_vals)):
        top = bottoms[i] + vals[i]
        if i == 0 or i == 3:
            txt = f"{dv:.2f}".replace(".", ",")
            ax.text(bar.get_x() + bar.get_width()/2, top + 0.3,
                    txt, ha="center", va="bottom", fontsize=11,
                    fontweight="bold")
        else:
            sign = "+" if dv > 0 else ""
            txt = f"{sign}{dv:.2f}".replace(".", ",")
            ax.text(bar.get_x() + bar.get_width()/2, top + 0.3,
                    txt, ha="center", va="bottom", fontsize=11,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Полезность, п.п.", fontsize=12)

    y_min = min(0, min(bottoms) - 1)
    y_max = max(max(b + v for b, v in zip(bottoms, vals)) + 2, u_port + 3)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(comma_fmt(0))
    ax.set_title(title, fontsize=13, pad=8)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.5)


def fig_4_17():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    panels = [
        (axes[0], "daily_avg",  "а) Дневной таймфрейм (3x + ФДР)"),
        (axes[1], "hourly_avg", "б) Часовой таймфрейм (5x + ФДР)"),
    ]

    for ax, key, title in panels:
        u_a, delta_er, delta_risk, u_port = WATERFALL[key]
        _draw_waterfall(ax, u_a, delta_er, delta_risk, u_port, title)

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(OUT, "fig_4_17.png"))
    plt.close(fig)
    print("  ✓ fig_4_17.png")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating fig 4.16-4.17...")
    fig_4_16()
    fig_4_17()
    print("Done.")
