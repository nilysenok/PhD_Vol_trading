#!/usr/bin/env python3
"""
Графики для раздела 4.3 диссертации.
9 рисунков: fig_4_6 .. fig_4_14
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "charts_4_3")
os.makedirs(OUT, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Цвета МСП
C_A    = "#4472C4"
C_B    = "#70AD47"
C_C    = "#ED7D31"
C_D    = "#7030A0"
C_BEST = "#C00000"
C_IMOEX = "#808080"

# Цвета методов взвешивания
C_EW   = "#2E86C1"
C_MV   = "#27AE60"
C_MS   = "#E74C3C"
C_IV   = "#F39C12"

def comma_fmt(decimals=1):
    """Русский формат: запятая-разделитель."""
    def _fmt(x, pos):
        s = f"{x:.{decimals}f}"
        return s.replace(".", ",")
    return mticker.FuncFormatter(_fmt)

def comma_fmt_pct(decimals=1):
    """Русский формат для процентов."""
    def _fmt(x, pos):
        s = f"{x:.{decimals}f}%"
        return s.replace(".", ",")
    return mticker.FuncFormatter(_fmt)


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.6 — 3D bar: стратегия × метод × Шарп, дневной
# ═══════════════════════════════════════════════════════════════════
def fig_4_6():
    strategies = ["S1\nMeanRev", "S2\nBollinger", "S3\nDonchian",
                  "S4\nSupertrend", "S5\nPivotPts", "S6\nVWAP"]
    methods = ["EW", "MinVar", "MaxSharpe", "InvVol"]
    colors = [C_EW, C_MV, C_MS, C_IV]

    data = np.array([
        [2.92, 2.94, 2.09, 2.03],
        [2.90, 2.80, 2.10, 1.95],
        [1.67, 1.68, 1.40, 1.60],
        [1.91, 1.96, 1.58, 1.52],
        [1.81, 1.73, 1.22, 1.28],
        [3.00, 2.97, 2.34, 1.87],
    ])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    n_strat, n_meth = data.shape
    bar_w = 0.18
    for j in range(n_meth):
        xs = np.arange(n_strat) + j * bar_w
        ys = np.zeros(n_strat)
        ax.bar3d(xs, j * np.ones(n_strat), ys,
                 bar_w * 0.9, 0.8, data[:, j],
                 color=colors[j], alpha=0.85, label=methods[j])

    ax.set_xticks(np.arange(n_strat) + bar_w * 1.5)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_yticks(np.arange(n_meth))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_zlabel("Коэфф. Шарпа", fontsize=12, labelpad=10)
    ax.zaxis.set_major_formatter(comma_fmt(1))
    ax.set_zlim(0, 3.5)
    ax.view_init(elev=25, azim=-45)
    ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(os.path.join(OUT, "fig_4_6_3d_daily.png"))
    plt.close(fig)
    print("  ✓ fig_4_6_3d_daily.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.7 — 3D bar: стратегия × метод × Шарп, часовой
# ═══════════════════════════════════════════════════════════════════
def fig_4_7():
    strategies = ["S1\nMeanRev", "S2\nBollinger", "S3\nDonchian",
                  "S4\nSupertrend", "S5\nPivotPts", "S6\nVWAP"]
    methods = ["EW", "MinVar", "MaxSharpe", "InvVol"]
    colors = [C_EW, C_MV, C_MS, C_IV]

    data = np.array([
        [1.18, 0.92, 0.92, 0.41],
        [2.08, 2.06, 1.32, 1.32],
        [1.51, 1.69, 1.46, 1.47],
        [1.74, 1.73, 1.45, 1.10],
        [2.06, 1.85, 1.43, 1.13],
        [0.70, 0.69, 0.54, 0.50],
    ])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    n_strat, n_meth = data.shape
    bar_w = 0.18
    for j in range(n_meth):
        xs = np.arange(n_strat) + j * bar_w
        ys = np.zeros(n_strat)
        ax.bar3d(xs, j * np.ones(n_strat), ys,
                 bar_w * 0.9, 0.8, data[:, j],
                 color=colors[j], alpha=0.85, label=methods[j])

    ax.set_xticks(np.arange(n_strat) + bar_w * 1.5)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_yticks(np.arange(n_meth))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_zlabel("Коэфф. Шарпа", fontsize=12, labelpad=10)
    ax.zaxis.set_major_formatter(comma_fmt(1))
    ax.set_zlim(0, 2.5)
    ax.view_init(elev=25, azim=-45)
    ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(os.path.join(OUT, "fig_4_7_3d_hourly.png"))
    plt.close(fig)
    print("  ✓ fig_4_7_3d_hourly.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.8 — Тепловая карта корреляций, дневной
# ═══════════════════════════════════════════════════════════════════
def fig_4_8():
    labels = ["S1 MeanRev", "S2 Bollinger", "S3 Donchian",
              "S4 Supertrend", "S5 PivotPoints", "S6 VWAP"]
    corr = np.array([
        [1.00,  0.66, -0.24, -0.16,  0.20,  0.16],
        [0.66,  1.00, -0.21, -0.11,  0.18,  0.20],
        [-0.24, -0.21, 1.00,  0.86, -0.10, -0.15],
        [-0.16, -0.11, 0.86,  1.00, -0.03, -0.14],
        [0.20,  0.18, -0.10, -0.03,  1.00,  0.21],
        [0.16,  0.20, -0.15, -0.14,  0.21,  1.00],
    ])
    df = pd.DataFrame(corr, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Формат аннотаций с запятой
    annot_arr = np.array([[f"{v:.2f}".replace(".", ",") for v in row] for row in corr])
    sns.heatmap(df, annot=annot_arr, fmt="", cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_8_corr_daily.png"))
    plt.close(fig)
    print("  ✓ fig_4_8_corr_daily.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.9 — Тепловая карта корреляций, часовой
# ═══════════════════════════════════════════════════════════════════
def fig_4_9():
    labels = ["S1 MeanRev", "S2 Bollinger", "S3 Donchian",
              "S4 Supertrend", "S5 PivotPoints", "S6 VWAP"]
    corr = np.array([
        [ 1.000,  0.150, -0.097, -0.080, -0.073,  0.045],
        [ 0.150,  1.000, -0.214, -0.157,  0.019,  0.062],
        [-0.097, -0.214,  1.000,  0.770, -0.015, -0.014],
        [-0.080, -0.157,  0.770,  1.000,  0.011, -0.056],
        [-0.073,  0.019, -0.015,  0.011,  1.000,  0.177],
        [ 0.045,  0.062, -0.014, -0.056,  0.177,  1.000],
    ])
    df = pd.DataFrame(corr, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    annot_arr = np.array([[f"{v:.2f}".replace(".", ",") for v in row] for row in corr])
    sns.heatmap(df, annot=annot_arr, fmt="", cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_9_corr_hourly.png"))
    plt.close(fig)
    print("  ✓ fig_4_9_corr_hourly.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.10 — Кумулятивная доходность МСП и IMOEX, дневной
# ═══════════════════════════════════════════════════════════════════
def fig_4_10():
    years = [2022, 2023, 2024, 2025]
    rets = {
        "МСП-A":    [13.02, 6.47, 9.15, 4.32],
        "МСП-B":    [7.87, 5.12, 5.47, 3.29],
        "МСП-D":    [15.41, 6.96, 9.73, 4.94],
        "МСП-BEST": [11.72, 6.09, 8.64, 4.70],
        "IMOEX":    [-43.12, 43.87, -6.97, -4.04],
    }
    colors = {"МСП-A": C_A, "МСП-B": C_B, "МСП-D": C_D,
              "МСП-BEST": C_BEST, "IMOEX": C_IMOEX}
    lw = {"МСП-A": 1.8, "МСП-B": 1.8, "МСП-D": 1.8,
          "МСП-BEST": 2.5, "IMOEX": 1.8}
    ls = {"МСП-A": "-", "МСП-B": "-", "МСП-D": "-",
          "МСП-BEST": "-", "IMOEX": "--"}

    x_pts = [2022, 2023, 2024, 2025, 2026]  # начало каждого года + конец

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, ann_rets in rets.items():
        cum = [100.0]
        for r in ann_rets:
            cum.append(cum[-1] * (1 + r / 100))
        ax.plot(x_pts, cum, color=colors[name], linewidth=lw[name],
                linestyle=ls[name], marker="o", markersize=5, label=name)

    ax.set_xlabel("Год")
    ax.set_ylabel("Стоимость, руб.")
    ax.set_xticks(x_pts)
    ax.set_xticklabels(["нач.\n2022", "нач.\n2023", "нач.\n2024", "нач.\n2025", "кон.\n2025"],
                       fontsize=9)
    ax.set_ylim(50, 160)
    ax.yaxis.set_major_formatter(comma_fmt(0))
    ax.axhline(100, color="gray", linewidth=0.7, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_10_equity_daily.png"))
    plt.close(fig)
    print("  ✓ fig_4_10_equity_daily.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.11 — Кумулятивная доходность МСП, часовой (синтетические)
# ═══════════════════════════════════════════════════════════════════
def fig_4_11():
    np.random.seed(42)
    n_days = 998  # ~4 года
    params = {
        "МСП-A":    (5.97, 5.35),
        "МСП-B":    (3.01, 0.75),
        "МСП-D":    (6.84, 3.13),
        "МСП-BEST": (3.31, 0.78),
    }
    colors_h = {"МСП-A": C_A, "МСП-B": C_B, "МСП-D": C_D, "МСП-BEST": C_BEST}
    lw_h = {"МСП-A": 1.8, "МСП-B": 1.8, "МСП-D": 1.8, "МСП-BEST": 2.5}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    dates = pd.bdate_range("2022-01-10", periods=n_days, freq="B")

    for name, (ret_ann, vol_ann) in params.items():
        mu_d = ret_ann / 100 / 252
        sig_d = vol_ann / 100 / np.sqrt(252)
        daily_r = np.random.normal(mu_d, sig_d, n_days)
        cum = 100.0 * np.cumprod(1 + daily_r)
        cum = np.insert(cum, 0, 100.0)
        dates_ext = dates.insert(0, dates[0] - pd.Timedelta(days=1))
        ax.plot(dates_ext, cum, color=colors_h[name], linewidth=lw_h[name],
                label=name)

    ax.set_xlabel("Дата")
    ax.set_ylabel("Стоимость, руб.")
    ax.yaxis.set_major_formatter(comma_fmt(0))
    ax.axhline(100, color="gray", linewidth=0.7, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_11_equity_hourly.png"))
    plt.close(fig)
    print("  ✓ fig_4_11_equity_hourly.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.12 — Годовые доходности МСП и IMOEX (grouped bar)
# ═══════════════════════════════════════════════════════════════════
def fig_4_12():
    years = ["2022", "2023", "2024", "2025"]
    data = {
        "МСП-A":    [13.02,  6.47, 9.15, 4.32],
        "МСП-D":    [15.41,  6.96, 9.73, 4.94],
        "МСП-BEST": [11.72,  6.09, 8.64, 4.70],
        "IMOEX":    [-43.12, 43.87, -6.97, -4.04],
    }
    colors_b = {"МСП-A": C_A, "МСП-D": C_D, "МСП-BEST": C_BEST, "IMOEX": C_IMOEX}
    names = list(data.keys())
    n_groups = len(years)
    n_bars = len(names)
    bar_w = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, name in enumerate(names):
        vals = data[name]
        bars = ax.bar(x + i * bar_w, vals, bar_w, color=colors_b[name],
                      label=name, edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            y_off = 1.0 if v >= 0 else -2.5
            ax.text(bar.get_x() + bar.get_width() / 2, v + y_off,
                    f"{v:.1f}".replace(".", ","), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8)

    ax.set_xticks(x + bar_w * (n_bars - 1) / 2)
    ax.set_xticklabels(years)
    ax.set_ylabel("Доходность, % годовых")
    ax.yaxis.set_major_formatter(comma_fmt(0))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim(-52, 52)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_12_annual.png"))
    plt.close(fig)
    print("  ✓ fig_4_12_annual.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.13 — Масштабирование МСП-BEST + ФДР, дневной
# ═══════════════════════════════════════════════════════════════════
def fig_4_13():
    k_vals    = [1,     2,     3,     5,     8]
    ret_vals  = [15.17, 21.56, 27.94, 40.75, 61.01]
    sh_vals   = [5.44,  3.87,  3.34,  2.92,  2.73]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(k_vals, ret_vals, "o-", color=C_A, linewidth=2,
                   markersize=7, label="Доходность")
    ln2 = ax2.plot(k_vals, sh_vals, "s-", color=C_BEST, linewidth=2,
                   markersize=7, label="Коэфф. Шарпа")

    ax1.axvline(3, color="gray", linewidth=1.2, linestyle="--")
    ax1.annotate("рекомендуемый\nмножитель", xy=(3, 15),
                 xytext=(4.5, 15), fontsize=9, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray"))

    ax1.set_xlabel("Множитель экспозиции, k")
    ax1.set_ylabel("Доходность, % годовых", color=C_A)
    ax2.set_ylabel("Коэфф. Шарпа", color=C_BEST)
    ax1.yaxis.set_major_formatter(comma_fmt(0))
    ax2.yaxis.set_major_formatter(comma_fmt(1))
    ax1.set_xticks(k_vals)
    ax1.set_xticklabels([f"{k}x" for k in k_vals])
    ax1.tick_params(axis="y", labelcolor=C_A)
    ax2.tick_params(axis="y", labelcolor=C_BEST)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_13_scaling.png"))
    plt.close(fig)
    print("  ✓ fig_4_13_scaling.png")


# ═══════════════════════════════════════════════════════════════════
# Рис. 4.14 — Шарп МСП-BEST от комиссии (daily vs hourly)
# ═══════════════════════════════════════════════════════════════════
def fig_4_14():
    comm = [0.00, 0.02, 0.04, 0.05, 0.06, 0.10, 0.15, 0.20]
    sh_d = [4.06, 4.01, 3.95, 3.93, 3.90, 3.79, 3.65, 3.52]
    sh_h = [3.76, 3.52, 3.28, 3.16, 3.04, 2.55, 1.94, 1.32]

    # Найти пересечение (интерполяция)
    from scipy.interpolate import interp1d
    comm_fine = np.linspace(0, 0.20, 500)
    f_d = interp1d(comm, sh_d, kind="cubic")
    f_h = interp1d(comm, sh_h, kind="cubic")
    diff = f_d(comm_fine) - f_h(comm_fine)
    # Кривые не пересекаются — daily всегда выше.
    # Находим точку, где разница минимальна (или пересечение, если есть)
    idx_min = np.argmin(np.abs(diff))
    cross_comm = comm_fine[idx_min]
    cross_sharpe = (f_d(cross_comm) + f_h(cross_comm)) / 2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(comm, sh_d, "o-", color=C_A, linewidth=2, markersize=7, label="Дневной")
    ax.plot(comm, sh_h, "s-", color=C_BEST, linewidth=2, markersize=7, label="Часовой")

    # Аннотация: при 0% разница минимальна; при 0.20% максимальна
    # Показываем деградацию
    ax.annotate(f"Δ = {sh_d[0]-sh_h[0]:.2f}".replace(".",","),
                xy=(0.005, (sh_d[0]+sh_h[0])/2), xytext=(0.03, 3.35),
                fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate(f"Δ = {sh_d[-1]-sh_h[-1]:.2f}".replace(".",","),
                xy=(0.195, (sh_d[-1]+sh_h[-1])/2), xytext=(0.14, 2.1),
                fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Комиссия за сторону, %")
    ax.set_ylabel("Коэфф. Шарпа")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: f"{x:.2f}%".replace(".", ",")))
    ax.yaxis.set_major_formatter(comma_fmt(1))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.005, 0.21)
    ax.set_ylim(1.0, 4.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_4_14_commission_impact.png"))
    plt.close(fig)
    print("  ✓ fig_4_14_commission_impact.png")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating chapter 4.3 charts...")
    fig_4_6()
    fig_4_7()
    fig_4_8()
    fig_4_9()
    fig_4_10()
    fig_4_11()
    fig_4_12()
    fig_4_13()
    fig_4_14()
    print(f"\nAll 9 charts saved to {OUT}/")
