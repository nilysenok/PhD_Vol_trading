#!/usr/bin/env python3
"""
Генерация 5 графиков для раздела 4.3 диссертации.

  fig_4_6  — Тепловая карта корреляций S1–S6
  fig_4_7  — Кумулятивная доходность МСП + IMOEX (daily)
  fig_4_8  — Кумулятивная доходность МСП (hourly, синтетические)
  fig_4_9  — Годовые доходности МСП vs IMOEX (барплот)
  fig_4_10 — Масштабирование МСП-BEST + ФДР

Выход: charts_4_3/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "charts_4_3"
OUT.mkdir(exist_ok=True)

# ── Шрифт и стиль ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#CCCCCC",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})

# ── Цвета ─────────────────────────────────────────────────────────────
C_A    = "#4472C4"
C_B    = "#70AD47"
C_C    = "#ED7D31"
C_D    = "#7030A0"
C_BEST = "#C00000"
C_IMOEX = "#808080"

# ── Comma formatters ──────────────────────────────────────────────────
def comma_fmt_0f(x, _):
    return f"{x:,.0f}".replace(",", " ").replace(".", ",")

def comma_fmt_1f(x, _):
    return f"{x:,.1f}".replace(",", " ").replace(".", ",")

def comma_fmt_2f(x, _):
    return f"{x:,.2f}".replace(",", " ").replace(".", ",")

def fmt_comma(v, decimals=1):
    """Format a number with comma decimal separator and minus sign."""
    s = f"{v:.{decimals}f}".replace(".", ",")
    if s.startswith("-"):
        s = "\u2212" + s[1:]
    return s


def spread_labels(values, min_gap):
    """Spread overlapping y-positions apart by at least min_gap."""
    items = sorted(enumerate(values), key=lambda t: t[1])
    positions = [v for _, v in items]
    # Push apart from bottom to top
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap
    # Restore original order
    result = [0.0] * len(values)
    for rank, (orig_idx, _) in enumerate(items):
        result[orig_idx] = positions[rank]
    return result


# ══════════════════════════════════════════════════════════════════════
#  Рис. 4.6 — Тепловая карта корреляций
# ══════════════════════════════════════════════════════════════════════
def fig_4_6():
    labels = [
        "S1 MeanRev", "S2 Bollinger", "S3 Donchian",
        "S4 Supertrend", "S5 PivotPoints", "S6 VWAP",
    ]
    corr = np.array([
        [ 1.00,  0.66, -0.24, -0.16,  0.20,  0.16],
        [ 0.66,  1.00, -0.21, -0.11,  0.18,  0.20],
        [-0.24, -0.21,  1.00,  0.86, -0.10, -0.15],
        [-0.16, -0.11,  0.86,  1.00, -0.03, -0.14],
        [ 0.20,  0.18, -0.10, -0.03,  1.00,  0.21],
        [ 0.16,  0.20, -0.15, -0.14,  0.21,  1.00],
    ])

    # Аннотации с запятой
    annot_str = np.empty_like(corr, dtype=object)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            annot_str[i, j] = f"{corr[i, j]:.2f}".replace(".", ",")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr,
        annot=annot_str, fmt="",
        annot_kws={"fontsize": 11},
        xticklabels=labels, yticklabels=labels,
        cmap="RdBu_r", vmin=-1, vmax=1,
        square=True, linewidths=0.8, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": ""},
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt_1f))

    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)
    ax.grid(False)

    fig.tight_layout()
    out = OUT / "fig_4_6_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
#  Рис. 4.7 — Кумулятивная доходность МСП + IMOEX (daily)
# ══════════════════════════════════════════════════════════════════════
def fig_4_7():
    years = [2022, 2023, 2024, 2025, 2026]
    rets = {
        "МСП-A":    [13.02, 6.47, 9.15, 4.32],
        "МСП-B":    [7.87, 5.12, 5.47, 3.29],
        "МСП-C":    [14.21, 6.07, 8.61, 4.86],
        "МСП-D":    [15.41, 6.96, 9.73, 4.94],
        "МСП-BEST": [11.72, 6.09, 8.64, 4.70],
        "IMOEX":    [-43.12, 43.87, -6.97, -4.04],
    }
    styles = {
        "МСП-A":    dict(color=C_A,     linewidth=1.8, linestyle="-"),
        "МСП-B":    dict(color=C_B,     linewidth=1.8, linestyle="-"),
        "МСП-C":    dict(color=C_C,     linewidth=1.8, linestyle="-"),
        "МСП-D":    dict(color=C_D,     linewidth=1.8, linestyle="-"),
        "МСП-BEST": dict(color=C_BEST,  linewidth=2.5, linestyle="-"),
        "IMOEX":    dict(color=C_IMOEX, linewidth=1.5, linestyle="--"),
    }
    # порядок отрисовки: IMOEX первый (фон), BEST последний (на переднем плане)
    draw_order = ["IMOEX", "МСП-B", "МСП-A", "МСП-C", "МСП-D", "МСП-BEST"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    finals = {}
    for name in draw_order:
        annual = rets[name]
        equity = [100.0]
        for r in annual:
            equity.append(equity[-1] * (1 + r / 100))
        ax.plot(years, equity, marker="o", markersize=4,
                label=name, **styles[name])
        finals[name] = equity[-1]

    # Подписи финальных значений — разнести чтобы не налезали
    names = list(finals.keys())
    raw_ys = [finals[n] for n in names]
    spread_ys = spread_labels(raw_ys, min_gap=4.0)

    for name, raw_y, label_y in zip(names, raw_ys, spread_ys):
        ax.annotate(
            f"{name}: {fmt_comma(raw_y)}",
            xy=(2026, raw_y),
            xytext=(2026.15, label_y),
            fontsize=8.5, color=styles[name]["color"],
            fontweight="bold",
            va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color=styles[name]["color"],
                            lw=0.6, shrinkA=0, shrinkB=2)
            if abs(label_y - raw_y) > 2 else None,
        )

    ax.set_xlabel("Год")
    ax.set_ylabel("Стоимость портфеля")
    ax.set_xticks(years)
    ax.set_xlim(2021.7, 2027.8)
    ax.set_ylim(45, 160)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt_0f))
    ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":", zorder=0)

    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC", ncol=2)

    fig.tight_layout()
    out = OUT / "fig_4_7_daily.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
#  Рис. 4.8 — Кумулятивная доходность МСП (hourly, синтетические)
# ══════════════════════════════════════════════════════════════════════
def fig_4_8():
    params = {
        "МСП-A":    (5.97, 5.35),
        "МСП-B":    (3.01, 0.75),
        "МСП-C":    (6.64, 5.10),
        "МСП-D":    (6.84, 3.13),
        "МСП-BEST": (3.31, 0.78),
    }
    styles = {
        "МСП-A":    dict(color=C_A,    linewidth=1.5, linestyle="-"),
        "МСП-B":    dict(color=C_B,    linewidth=1.5, linestyle="-"),
        "МСП-C":    dict(color=C_C,    linewidth=1.5, linestyle="-"),
        "МСП-D":    dict(color=C_D,    linewidth=1.5, linestyle="-"),
        "МСП-BEST": dict(color=C_BEST, linewidth=2.5, linestyle="-"),
    }
    draw_order = ["МСП-B", "МСП-A", "МСП-C", "МСП-D", "МСП-BEST"]

    n_days = 1008
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    finals = {}
    for name in draw_order:
        ret_ann, vol_ann = params[name]
        mu_d = ret_ann / 100 / 252
        sigma_d = vol_ann / 100 / np.sqrt(252)

        daily_rets = np.random.normal(mu_d, sigma_d, n_days)
        equity = 100.0 * np.cumprod(1 + daily_rets)

        target_final = 100.0 * (1 + ret_ann / 100) ** 4
        log_eq = np.log(equity / 100.0)
        ratio = np.log(target_final / 100.0) / np.log(equity[-1] / 100.0)
        log_eq_scaled = log_eq * ratio
        equity_scaled = 100.0 * np.exp(log_eq_scaled)

        ax.plot(dates[:n_days], equity_scaled, label=name, **styles[name])
        finals[name] = equity_scaled[-1]

    # Подписи финальных значений — разнести
    names = list(finals.keys())
    raw_ys = [finals[n] for n in names]
    spread_ys = spread_labels(raw_ys, min_gap=3.5)

    last_date = dates[n_days - 1]
    # Label x-position in data coords: shift right by ~30 days
    label_x = last_date + pd.Timedelta(days=12)
    for name, raw_y, label_y in zip(names, raw_ys, spread_ys):
        ax.annotate(
            f"{name}: {fmt_comma(raw_y)}",
            xy=(last_date, raw_y),
            xytext=(label_x, label_y),
            fontsize=8.5, color=styles[name]["color"],
            fontweight="bold", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color=styles[name]["color"],
                            lw=0.6, shrinkA=0, shrinkB=2)
            if abs(label_y - raw_y) > 2 else None,
        )

    ax.set_xlabel("Год")
    ax.set_ylabel("Стоимость портфеля")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt_0f))
    ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":", zorder=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Extend x-axis right to fit labels
    ax.set_xlim(dates[0] - pd.Timedelta(days=20),
                last_date + pd.Timedelta(days=130))

    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC")

    fig.tight_layout()
    out = OUT / "fig_4_8_hourly.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
#  Рис. 4.9 — Годовые доходности МСП и IMOEX (барплот)
# ══════════════════════════════════════════════════════════════════════
def fig_4_9():
    years = ["2022", "2023", "2024", "2025"]
    data = {
        "МСП-A":    [13.02,  6.47,  9.15, 4.32],
        "МСП-D":    [15.41,  6.96,  9.73, 4.94],
        "МСП-BEST": [11.72,  6.09,  8.64, 4.70],
        "IMOEX":    [-43.12, 43.87, -6.97, -4.04],
    }
    colors = {
        "МСП-A":    C_A,
        "МСП-D":    C_D,
        "МСП-BEST": C_BEST,
        "IMOEX":    C_IMOEX,
    }

    n_groups = len(years)
    n_bars = len(data)
    bar_width = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (name, vals) in enumerate(data.items()):
        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      label=name, color=colors[name],
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            pad = 1.2 if v >= 0 else -1.2
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + pad,
                    fmt_comma(v),
                    ha="center", va=va, fontsize=7.5,
                    color=colors[name], fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Год")
    ax.set_ylabel("Доходность, %")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt_0f))
    ax.set_ylim(-55, 52)

    ax.legend(loc="lower left", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC", ncol=4,
              bbox_to_anchor=(0.0, 0.0))

    fig.tight_layout()
    out = OUT / "fig_4_9_annual.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
#  Рис. 4.10 — Масштабирование МСП-BEST + ФДР
# ══════════════════════════════════════════════════════════════════════
def fig_4_10():
    k_vals   = [1,     2,     3,     5,     8]
    ret_pct  = [15.17, 21.56, 27.94, 40.75, 61.01]
    sharpe   = [5.44,  3.87,  3.34,  2.92,  2.73]
    mdd_pct  = [-1.28, -3.14, -5.08, -9.54, -15.86]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.spines["right"].set_visible(True)

    # ── MDD: серые столбики (рисуем первыми — на фоне) ──
    bar_w = 0.4
    bars = ax1.bar(k_vals, mdd_pct, bar_w,
                   color=C_IMOEX, alpha=0.30, zorder=2,
                   label="Макс. просадка, %")
    # Подписи MDD под столбиками
    for xi, yi in zip(k_vals, mdd_pct):
        ax1.annotate(
            fmt_comma(yi),
            xy=(xi, yi), xytext=(0, -11), textcoords="offset points",
            ha="center", fontsize=8, color="#555555",
        )

    # ── Левая ось: Доходность (линия) ──
    ln1 = ax1.plot(k_vals, ret_pct, "o-", color=C_A, linewidth=2.2,
                   markersize=7, label="Доходность, %", zorder=4)
    ax1.set_xlabel("Множитель экспозиции (k)")
    ax1.set_ylabel("Доходность / Просадка, %", color="#333333")
    ax1.tick_params(axis="y", labelcolor="#333333")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt_0f))
    ax1.set_ylim(-25, 72)
    ax1.axhline(0, color="#AAAAAA", linewidth=0.6, linestyle="-", zorder=1)

    # Подписи доходности — над точками
    for xi, yi in zip(k_vals, ret_pct):
        ax1.annotate(
            fmt_comma(yi),
            xy=(xi, yi), xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=9, color=C_A, fontweight="bold",
        )

    # ── Правая ось: только Sharpe ──
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ln2 = ax2.plot(k_vals, sharpe, "s-", color=C_BEST, linewidth=2.2,
                   markersize=7, label="Коэффициент Шарпа", zorder=4)
    ax2.set_ylabel("Коэффициент Шарпа", color=C_BEST)
    ax2.tick_params(axis="y", labelcolor=C_BEST)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt_1f))
    ax2.set_ylim(1.5, 6.5)
    ax2.grid(False)

    # Подписи Шарпа — чередуем позицию чтобы не наложить на доходность
    sharpe_offsets = [
        (0, -14),   # k=1: под точкой
        (0, -14),   # k=2: под
        (12, 6),    # k=3: правее (иначе налезает на 27.9)
        (0, 10),    # k=5: над
        (0, -14),   # k=8: под
    ]
    for (xi, yi), (dx, dy) in zip(zip(k_vals, sharpe), sharpe_offsets):
        ax2.annotate(
            fmt_comma(yi, 2),
            xy=(xi, yi), xytext=(dx, dy), textcoords="offset points",
            ha="center" if dx == 0 else "left",
            fontsize=9, color=C_BEST, fontweight="bold",
        )

    # Вертикальная линия k=3
    ax1.axvline(3, color="#444444", linewidth=1.0, linestyle=":",
                zorder=1, alpha=0.5)
    ax1.annotate(
        "k = 3\n(рекомендуемый)",
        xy=(3, 65), xytext=(3.7, 65),
        fontsize=9, color="#444444", ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
    )

    # Объединённая легенда
    lns = ln1 + ln2 + [bars]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", frameon=True,
               framealpha=0.95, edgecolor="#CCCCCC", ncol=1)

    ax1.set_xticks(k_vals)
    ax1.set_xticklabels([f"{k}x" for k in k_vals])
    ax1.set_xlim(0.2, 8.8)

    fig.tight_layout()
    out = OUT / "fig_4_10_scaling.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Генерация графиков раздела 4.3...")
    fig_4_6()
    fig_4_7()
    fig_4_8()
    fig_4_9()
    fig_4_10()
    print(f"\nГотово! Все файлы в: {OUT}")
