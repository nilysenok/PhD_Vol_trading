#!/usr/bin/env python3
"""
Dissertation Section 4.3 — 5 academic figures.

  Fig 4.6  — Cumulative returns: MSP vs IMOEX (daily, 2022-2026)
  Fig 4.7  — Correlation heatmap S1-S6 (approach A, daily)
  Fig 4.8  — Grouped bar: yearly returns MSP-A, MSP-D, MSP-BEST, IMOEX
  Fig 4.9  — Scaling MSP-BEST + MMF: return & Sharpe (dual-axis)
  Fig 4.10 — Cumulative returns: MSP (hourly, 2022-2026)

Usage: python3 scripts/v4_dissertation_charts.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
V4_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
OUT_DIR = BASE / "dissertation_materials" / "chapter_4_3" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMOEX_PATH = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/"
                   "dataset_final/02_external/candles_10m/IMOEX.parquet")

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
BCD_YEARS = [2022, 2023, 2024, 2025]

COMM_DAILY = 0.0005
COMM_HOURLY = 0.0004

BEST_APP_DAILY = {
    "S1_MeanRev": "C", "S2_Bollinger": "C", "S3_Donchian": "B",
    "S4_Supertrend": "D", "S5_PivotPoints": "C", "S6_VWAP": "D",
}
BEST_APP_HOURLY = {
    "S1_MeanRev": "D", "S2_Bollinger": "D", "S3_Donchian": "B",
    "S4_Supertrend": "B", "S5_PivotPoints": "D", "S6_VWAP": "D",
}

# ── Academic colour palette ───────────────────────────────────────────
COL_A = "#7F7F7F"      # grey
COL_B = "#4472C4"      # blue
COL_C = "#C55A11"      # dark orange
COL_D = "#548235"      # dark green
COL_BEST = "#1F1F1F"   # near-black
COL_IMOEX = "#C00000"  # dark red


# ── Style setup ───────────────────────────────────────────────────────
def setup_rc():
    # Try Times New Roman, fall back to serif
    try:
        from matplotlib.font_manager import findSystemFonts
        available = [f.lower() for f in findSystemFonts()]
        has_times = any("times" in f for f in available)
    except Exception:
        has_times = False

    font_family = "serif"
    font_serif = ["Times New Roman", "Times", "DejaVu Serif"]

    plt.rcParams.update({
        "font.family":        font_family,
        "font.serif":         font_serif,
        "font.size":          12,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.unicode_minus": False,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "grid.color":         "#B0B0B0",
    })


def comma_fmt(x, _):
    """Format number with comma as decimal separator."""
    s = f"{x:g}"
    return s.replace(".", ",")


def comma_fmt_1f(x, _):
    s = f"{x:.1f}"
    return s.replace(".", ",")


def comma_fmt_2f(x, _):
    s = f"{x:.2f}"
    return s.replace(".", ",")


def hgrid(ax):
    ax.yaxis.grid(True, ls="--", alpha=0.3, lw=0.5, color="#B0B0B0")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ── Data helpers ──────────────────────────────────────────────────────
def net_returns(pos, gross_r, comm):
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * comm


def build_meta_returns(pos_df, comm, best_app, ticker_universe_src="A"):
    """Build EW meta-portfolio return series for A, B, C, D, BEST."""
    # Get ticker universe from approach A for each strategy
    ticker_universes = {}
    for strat in STRATEGIES:
        sub_a = pos_df[(pos_df["strategy"] == strat) &
                       (pos_df["approach"] == "A")]
        ticker_universes[strat] = sorted(sub_a["ticker"].unique())

    port_cache = {}
    for strat in STRATEGIES:
        for appr in APPROACHES:
            sub = pos_df[(pos_df["strategy"] == strat) &
                         (pos_df["approach"] == appr)]
            ticker_rets = {}
            for tkr in sorted(sub["ticker"].unique()):
                g = sub[sub["ticker"] == tkr].sort_values("date")
                nr = net_returns(g["position"].values,
                                 g["daily_gross_return"].values, comm)
                ticker_rets[tkr] = pd.Series(nr, index=pd.to_datetime(g["date"].values))
            ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
            # Fix: ensure full ticker universe
            for tkr in ticker_universes[strat]:
                if tkr not in ret_df.columns:
                    ret_df[tkr] = 0.0
            port_cache[(strat, appr)] = ret_df.mean(axis=1)

    def build_meta(approach_map):
        rets = []
        for s in STRATEGIES:
            appr = approach_map if isinstance(approach_map, str) else approach_map[s]
            rets.append(port_cache[(s, appr)])
        df = pd.DataFrame({s: r for s, r in zip(STRATEGIES, rets)}).sort_index().fillna(0)
        return df.mean(axis=1)

    return {
        "A": build_meta("A"),
        "B": build_meta("B"),
        "C": build_meta("C"),
        "D": build_meta("D"),
        "BEST": build_meta(best_app),
    }


def load_imoex_daily():
    """Load IMOEX daily returns for 2022-2025."""
    imoex = pd.read_parquet(IMOEX_PATH)
    imoex["date"] = pd.to_datetime(imoex["begin"]).dt.date
    daily_close = imoex.groupby("date")["close"].last()
    daily_close.index = pd.to_datetime(daily_close.index)
    daily_ret = daily_close.pct_change().dropna()
    return daily_ret[(daily_ret.index >= "2022-01-01") &
                     (daily_ret.index <= "2025-12-31")]


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.6 — Cumulative returns: MSP vs IMOEX (daily)
# ══════════════════════════════════════════════════════════════════════
def fig_4_6(meta_rets, imoex_ret):
    fig, ax = plt.subplots(figsize=(8, 5))

    # IMOEX
    imoex_cum = (1 + imoex_ret).cumprod() * 100
    ax.plot(imoex_cum.index, imoex_cum.values,
            color=COL_IMOEX, ls="--", lw=1.5, label="IMOEX B&H", zorder=2)

    # MSP portfolios
    styles = {
        "A":    (COL_A,    "-", 1.2, "МСП-A"),
        "B":    (COL_B,    "-", 1.0, "МСП-B"),
        "C":    (COL_C,    "-", 1.0, "МСП-C"),
        "D":    (COL_D,    "-", 1.2, "МСП-D"),
        "BEST": (COL_BEST, "-", 2.2, "МСП-BEST"),
    }
    for key, (color, ls, lw, label) in styles.items():
        cum = (1 + meta_rets[key]).cumprod() * 100
        ax.plot(cum.index, cum.values, color=color, ls=ls, lw=lw,
                label=label, zorder=3 if key == "BEST" else 2)

    # Trading halt
    ax.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"),
               color="#E0E0E0", alpha=0.5, zorder=0)

    ax.axhline(100, color="black", lw=0.4, zorder=1)
    ax.set_ylabel("Стоимость портфеля (начальная = 100)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    hgrid(ax)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, ncol=2)

    # Final values annotation
    for key in ["BEST", "A"]:
        cum = (1 + meta_rets[key]).cumprod() * 100
        final_val = cum.iloc[-1]
        ax.annotate(f"{final_val:.1f}".replace(".", ","),
                    xy=(cum.index[-1], final_val),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=9, color=styles[key][0],
                    fontweight="bold", va="center")
    # IMOEX final
    ax.annotate(f"{imoex_cum.iloc[-1]:.1f}".replace(".", ","),
                xy=(imoex_cum.index[-1], imoex_cum.iloc[-1]),
                xytext=(10, 0), textcoords="offset points",
                fontsize=9, color=COL_IMOEX, fontweight="bold", va="center")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_6_cumulative_daily.png")
    plt.close(fig)
    print("  fig_4_6 saved")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.7 — Correlation heatmap S1-S6
# ══════════════════════════════════════════════════════════════════════
def fig_4_7():
    labels = ["S1\nMeanRev", "S2\nBollinger", "S3\nDonchian",
              "S4\nSupertrend", "S5\nPivotPoints", "S6\nVWAP"]
    data = np.array([
        [1.00,  0.66, -0.24, -0.16,  0.20,  0.16],
        [0.66,  1.00, -0.21, -0.11,  0.18,  0.20],
        [-0.24, -0.21,  1.00,  0.86, -0.10, -0.15],
        [-0.16, -0.11,  0.86,  1.00, -0.03, -0.14],
        [0.20,  0.18, -0.10, -0.03,  1.00,  0.21],
        [0.16,  0.20, -0.15, -0.14,  0.21,  1.00],
    ])
    corr_df = pd.DataFrame(data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Format annotations with comma
    annot_text = np.array([[f"{v:.2f}".replace(".", ",") for v in row] for row in data])

    sns.heatmap(corr_df, annot=annot_text, fmt="", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                linecolor="white", ax=ax,
                cbar_kws={"label": "Корреляция", "shrink": 0.85},
                annot_kws={"size": 11, "fontweight": "bold"})

    # Fix text colors: white on dark cells, black on light
    for text in ax.texts:
        val_str = text.get_text().replace(",", ".")
        try:
            val = float(val_str)
            if abs(val) > 0.55:
                text.set_color("white")
            else:
                text.set_color("black")
        except ValueError:
            pass

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_7_correlation_heatmap.png")
    plt.close(fig)
    print("  fig_4_7 saved")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.8 — Grouped bar: yearly returns MSP vs IMOEX
# ══════════════════════════════════════════════════════════════════════
def fig_4_8():
    years = ["2022", "2023", "2024", "2025"]
    mcp_a =    [13.02,  6.47,  9.15,  4.32]
    mcp_d =    [15.41,  6.96,  9.73,  4.94]
    mcp_best = [11.72,  6.09,  8.64,  4.70]
    imoex =    [-43.12, 43.87, -6.97, -4.04]

    x = np.arange(len(years))
    width = 0.20

    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - 1.5*width, mcp_a, width, color=COL_A,
                edgecolor="#555555", lw=0.5, label="МСП-A", zorder=3)
    b2 = ax.bar(x - 0.5*width, mcp_d, width, color=COL_D,
                edgecolor="#3A5C24", lw=0.5, label="МСП-D", zorder=3)
    b3 = ax.bar(x + 0.5*width, mcp_best, width, color=COL_B,
                edgecolor="#2F5496", lw=0.5, label="МСП-BEST", zorder=3)
    b4 = ax.bar(x + 1.5*width, imoex, width, color=COL_IMOEX,
                edgecolor="#800000", lw=0.5, label="IMOEX B&H", zorder=3)

    # Value labels
    def add_val(bars, vals):
        for bar, val in zip(bars, vals):
            y = val
            offset = 1.0 if val >= 0 else -1.5
            va = "bottom" if val >= 0 else "top"
            txt = f"{val:+.1f}".replace(".", ",")
            ax.text(bar.get_x() + bar.get_width()/2, y + offset,
                    txt, ha="center", va=va, fontsize=7.5, fontweight="bold")

    add_val(b1, mcp_a)
    add_val(b2, mcp_d)
    add_val(b3, mcp_best)
    add_val(b4, imoex)

    ax.axhline(0, color="black", lw=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Годовая доходность, %")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt))
    hgrid(ax)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, ncol=2)

    ax.set_ylim(min(imoex)*1.2, max(imoex)*1.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_8_yearly_returns.png")
    plt.close(fig)
    print("  fig_4_8 saved")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.9 — Scaling MSP-BEST + MMF: return & Sharpe
# ══════════════════════════════════════════════════════════════════════
def fig_4_9():
    k = [1, 2, 3, 5]
    ret_pct = [15.17, 21.56, 27.94, 40.75]
    sharpe = [5.44, 3.87, 3.34, 2.92]
    mdd = [-1.28, -3.14, -5.08, -9.54]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Left axis: Return %
    line1, = ax1.plot(k, ret_pct, "o-", color=COL_B, lw=2, ms=8, label="Доходность, %", zorder=3)
    # Right axis: Sharpe
    line2, = ax2.plot(k, sharpe, "s--", color=COL_D, lw=2, ms=8, label="Коэффициент Шарпа", zorder=3)
    # MDD on right axis
    line3, = ax2.plot(k, mdd, "D:", color=COL_IMOEX, lw=1.5, ms=7, label="MDD, %", zorder=3)

    # Annotate return values
    for xi, yi in zip(k, ret_pct):
        ax1.annotate(f"{yi:.1f}%".replace(".", ","),
                     xy=(xi, yi), xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=10, color=COL_B, fontweight="bold")
    # Annotate Sharpe values
    for xi, yi in zip(k, sharpe):
        ax2.annotate(f"{yi:.2f}".replace(".", ","),
                     xy=(xi, yi), xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=10, color=COL_D, fontweight="bold")
    # Annotate MDD values
    for xi, yi in zip(k, mdd):
        ax2.annotate(f"{yi:.1f}%".replace(".", ","),
                     xy=(xi, yi), xytext=(0, -14), textcoords="offset points",
                     ha="center", fontsize=9, color=COL_IMOEX, fontweight="bold")

    ax1.set_xlabel("Множитель экспозиции")
    ax1.set_ylabel("Доходность, %", color=COL_B)
    ax2.set_ylabel("Шарп / MDD, %", color=COL_D)

    ax1.set_xticks(k)
    ax1.set_xticklabels([f"{ki}x" for ki in k])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt))

    ax1.tick_params(axis="y", colors=COL_B)
    ax2.tick_params(axis="y", colors=COL_D)
    ax1.spines["left"].set_color(COL_B)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COL_D)

    # Highlight 3x as optimal balance
    ax1.axvline(3, color="#B0B0B0", ls=":", lw=1, alpha=0.7, zorder=1)
    ax1.annotate("оптимальный\nбаланс",
                 xy=(3, ret_pct[2]), xytext=(3.5, ret_pct[0]+2),
                 fontsize=9, color="#666666", fontstyle="italic",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=1))

    hgrid(ax1)
    ax2.grid(False)

    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True, fancybox=False,
               edgecolor="0.8", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_9_scaling_msp_best.png")
    plt.close(fig)
    print("  fig_4_9 saved")


# ══════════════════════════════════════════════════════════════════════
#  Fig 4.10 — Cumulative returns: MSP (hourly, 2022-2026)
# ══════════════════════════════════════════════════════════════════════
def fig_4_10(hourly_meta_rets):
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "A":    (COL_A,    "-", 1.2, "МСП-A"),
        "B":    (COL_B,    "-", 1.0, "МСП-B"),
        "C":    (COL_C,    "-", 1.0, "МСП-C"),
        "D":    (COL_D,    "-", 1.2, "МСП-D"),
        "BEST": (COL_BEST, "-", 2.2, "МСП-BEST"),
    }

    for key, (color, ls, lw, label) in styles.items():
        ret_s = hourly_meta_rets[key]
        # Resample hourly to daily (last value per day) for cleaner plot
        cum = (1 + ret_s).cumprod() * 100
        if len(cum) > 2000:
            cum_daily = cum.groupby(cum.index.date).last()
            cum_daily.index = pd.to_datetime(cum_daily.index)
        else:
            cum_daily = cum
        ax.plot(cum_daily.index, cum_daily.values,
                color=color, ls=ls, lw=lw, label=label,
                zorder=3 if key == "BEST" else 2)

    # Trading halt
    ax.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"),
               color="#E0E0E0", alpha=0.5, zorder=0)

    ax.axhline(100, color="black", lw=0.4, zorder=1)
    ax.set_ylabel("Стоимость портфеля (начальная = 100)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(comma_fmt))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    hgrid(ax)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=0.95, ncol=2)

    # Final values annotation
    for key in ["BEST", "A"]:
        ret_s = hourly_meta_rets[key]
        cum = (1 + ret_s).cumprod() * 100
        cum_daily = cum.groupby(cum.index.date).last()
        cum_daily.index = pd.to_datetime(cum_daily.index)
        final_val = cum_daily.iloc[-1]
        ax.annotate(f"{final_val:.1f}".replace(".", ","),
                    xy=(cum_daily.index[-1], final_val),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=9, color=styles[key][0],
                    fontweight="bold", va="center")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_10_cumulative_hourly.png")
    plt.close(fig)
    print("  fig_4_10 saved")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    print("Generating dissertation figures (Section 4.3)...")

    # ── Load positions ──
    print("  Loading daily positions...")
    pos_all = pd.read_parquet(V4_DIR / "daily_positions.parquet")

    # Daily meta-portfolios
    pos_daily = pos_all[(pos_all["tf"] == "daily") &
                        (pos_all["test_year"].isin(BCD_YEARS))].copy()
    daily_rets = build_meta_returns(pos_daily, COMM_DAILY, BEST_APP_DAILY)
    print(f"    Daily: {len(pos_daily):,} rows")

    # Hourly meta-portfolios
    pos_hourly = pos_all[(pos_all["tf"] == "hourly") &
                         (pos_all["test_year"].isin(BCD_YEARS))].copy()
    hourly_rets = build_meta_returns(pos_hourly, COMM_HOURLY, BEST_APP_HOURLY)
    print(f"    Hourly: {len(pos_hourly):,} rows")

    # IMOEX
    print("  Loading IMOEX...")
    imoex_ret = load_imoex_daily()
    print(f"    IMOEX: {len(imoex_ret)} daily returns")

    # ── Generate figures ──
    print("  Generating figures...")
    fig_4_6(daily_rets, imoex_ret)
    fig_4_7()
    fig_4_8()
    fig_4_9()
    fig_4_10(hourly_rets)

    print(f"\nAll 5 figures saved to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
