#!/usr/bin/env python3
"""
Section 4.2 — Data extraction and charts (2022-2026).

Extracts tables 4.3–4.9 and generates figures 4.3–4.4 for dissertation section 4.2.
All output saved to output_4_2/.

Data sources:
  - Net Sharpe 6×4:  v4_A_vs_forecast_comparison.csv (EW portfolios, 2022-2026)
  - Passports B/C/D: v4_S{1-6}_daily.csv (per-strategy, 2022-2025 best available)
  - Turnover:        v4_full_daily.csv (mean across tickers)
  - Exposure:        cross-ticker daily definition (from final_summary_v6, 2022-2026)

Usage: python3 scripts/v4_data_42.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "final" / "strategies" / "walkforward_v4"
OUT  = ROOT / "output_4_2"
OUT.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
STRATEGIES = [
    "S1_MeanRev", "S2_Bollinger", "S3_Donchian",
    "S4_Supertrend", "S5_PivotPoints", "S6_VWAP",
]
SHORT = {
    "S1_MeanRev":     "S1 Mean Reversion",
    "S2_Bollinger":   "S2 Bollinger Bands",
    "S3_Donchian":    "S3 Donchian Channel",
    "S4_Supertrend":  "S4 Supertrend",
    "S5_PivotPoints": "S5 Pivot Points",
    "S6_VWAP":        "S6 VWAP",
}
APPROACHES = ["A", "B", "C", "D"]
APP_LABELS = {
    "A": "Подход A (без прогнозов)",
    "B": "Подход B (адаптивные стопы)",
    "C": "Подход C (режимная фильтрация)",
    "D": "Подход D (vol-gate)",
}
APP_COLORS = {"A": "#808080", "B": "#4472C4", "C": "#ED7D31", "D": "#548235"}

# ── Hardcoded exposure (cross-ticker daily, 2022-2026) ────────────────
# Source: final_summary_v6_complete.md, section 5.4
EXPOSURE = {
    "S1_MeanRev":     {"A": 4.1,  "B": 3.1,  "C": 2.9,  "D": 3.1},
    "S2_Bollinger":   {"A": 3.9,  "B": 3.1,  "C": 2.5,  "D": 2.3},
    "S3_Donchian":    {"A": 50.7, "B": 10.5, "C": 21.0, "D": 41.1},
    "S4_Supertrend":  {"A": 31.6, "B": 4.5,  "C": 14.6, "D": 23.8},
    "S5_PivotPoints": {"A": 1.1,  "B": 1.0,  "C": 0.7,  "D": 0.7},
    "S6_VWAP":        {"A": 3.1,  "B": 2.7,  "C": 2.2,  "D": 2.3},
}

# ── Hardcoded turnover (trades/year, 2022-2026) ──────────────────────
# Source: final_summary_v6_complete.md, section 5.3
TURNOVER = {
    "S1_MeanRev":     {"A": 2.1, "B": 1.8, "C": 2.5, "D": 1.6},
    "S2_Bollinger":   {"A": 1.5, "B": 1.5, "C": 1.9, "D": 1.2},
    "S3_Donchian":    {"A": 4.3, "B": 5.0, "C": 11.1, "D": 4.7},
    "S4_Supertrend":  {"A": 3.1, "B": 2.3, "C": 9.2, "D": 3.1},
    "S5_PivotPoints": {"A": 0.7, "B": 0.6, "C": 0.6, "D": 0.5},
    "S6_VWAP":        {"A": 2.0, "B": 1.9, "C": 1.9, "D": 1.6},
}

LOG = []


def log(msg):
    LOG.append(msg)
    print(msg)


# ── Style ──────────────────────────────────────────────────────────────
def setup_rc():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         12,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   11,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.15,
    })


# ══════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════
def load_all():
    data = {}

    # 1. EW portfolio comparison (Table 4.6)
    p = BASE / "tables" / "v4_A_vs_forecast_comparison.csv"
    data["comparison"] = pd.read_csv(p)
    log(f"[OK] Loaded {p.relative_to(ROOT)}")

    # 2. Per-strategy daily files (Tables 4.3-4.5 passports)
    passports = {}
    for s in STRATEGIES:
        snum = s.split("_")[0]  # e.g. "S1"
        p = BASE / "tables" / f"v4_{snum}_daily.csv"
        if not p.exists():
            p = BASE / "tables" / f"v4_{s}_daily.csv"
        if p.exists():
            passports[s] = pd.read_csv(p)
            log(f"[OK] Loaded {p.relative_to(ROOT)}")
        else:
            log(f"[WARN] Not found: per-strategy file for {s}")
    data["passports"] = passports

    # 3. Full daily (mean-ticker metrics — for passport fallback)
    p = BASE / "tables" / "v4_full_daily.csv"
    data["full_daily"] = pd.read_csv(p)
    log(f"[OK] Loaded {p.relative_to(ROOT)}")

    log(f"[INFO] Exposure: hardcoded cross-ticker daily (2022-2026)")
    log(f"[INFO] Turnover: hardcoded from summary (2022-2026)")

    return data


# ══════════════════════════════════════════════════════════════════════
#  Table 4.6: Net Sharpe 6×4 (EW portfolios)
# ══════════════════════════════════════════════════════════════════════
def make_table_4_6(data):
    log("\n--- Table 4.6: Net Sharpe 6×4 (EW portfolios) ---")
    comp = data["comparison"]
    ew = comp[comp["table"] == "EW_NetSharpe"].copy()

    rows = []
    for s in STRATEGIES:
        r = ew[ew["strategy"] == s].iloc[0]
        a, b, c, d = float(r["A"]), float(r["B"]), float(r["C"]), float(r["D"])
        vals = {"A": a, "B": b, "C": c, "D": d}
        best_key = max(vals, key=vals.get)
        mean_bcd = np.mean([b, c, d])
        best_bcd = max(b, c, d)
        delta_mean = (mean_bcd / a - 1) * 100 if a != 0 else np.nan
        delta_best = (best_bcd / a - 1) * 100 if a != 0 else np.nan
        rows.append({
            "Стратегия": SHORT[s],
            "A": round(a, 4),
            "B": round(b, 4),
            "C": round(c, 4),
            "D": round(d, 4),
            "Best": best_key,
            "mean_BCD": round(mean_bcd, 4),
            "Δ% mean vs A": round(delta_mean, 1),
            "Δ% best vs A": round(delta_best, 1),
        })

    # Mean row
    mean_a = np.mean([r["A"] for r in rows])
    mean_b = np.mean([r["B"] for r in rows])
    mean_c = np.mean([r["C"] for r in rows])
    mean_d = np.mean([r["D"] for r in rows])
    mean_bcd = np.mean([mean_b, mean_c, mean_d])
    best_bcd_avg = np.mean([r["D"] if r["Best"] == "D" else
                            r["C"] if r["Best"] == "C" else r["B"]
                            for r in rows])
    rows.append({
        "Стратегия": "Среднее",
        "A": round(mean_a, 4),
        "B": round(mean_b, 4),
        "C": round(mean_c, 4),
        "D": round(mean_d, 4),
        "Best": "D",
        "mean_BCD": round(mean_bcd, 4),
        "Δ% mean vs A": round((mean_bcd / mean_a - 1) * 100, 1),
        "Δ% best vs A": round((mean_d / mean_a - 1) * 100, 1),
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_4_6_net_sharpe_summary.csv", index=False)
    log(f"  Saved table_4_6_net_sharpe_summary.csv")
    log(f"  Mean: A={mean_a:.4f}, B={mean_b:.4f}, C={mean_c:.4f}, D={mean_d:.4f}")
    log(f"  Δ% mean(BCD) vs A: {(mean_bcd / mean_a - 1) * 100:+.1f}%")
    log(f"  Δ% D vs A: {(mean_d / mean_a - 1) * 100:+.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════════
#  Tables 4.3–4.5: Passport per approach (B, C, D)
# ══════════════════════════════════════════════════════════════════════
def make_passport_tables(data):
    log("\n--- Tables 4.3–4.5: Approach passports ---")
    passports = data["passports"]

    for app, tnum in [("B", "4_3"), ("C", "4_4"), ("D", "4_5")]:
        log(f"\n  Approach {app} (Table {tnum}):")
        rows = []
        for s in STRATEGIES:
            if s not in passports:
                log(f"    [SKIP] {s}: no per-strategy file")
                continue
            pdf = passports[s]
            r = pdf[pdf["approach"] == app]
            if r.empty:
                log(f"    [SKIP] {s}: no approach {app} data")
                rows.append({
                    "Стратегия": SHORT[s],
                    "Net Sharpe": np.nan,
                    "Ann. Return (%)": np.nan,
                    "Ann. Vol (%)": np.nan,
                    "Max Drawdown (%)": np.nan,
                    "Экспозиция (%)": np.nan,
                    "Сделок/год": np.nan,
                    "Win Rate (%)": np.nan,
                })
                continue
            r = r.iloc[0]
            # Use hardcoded exposure (cross-ticker daily, 2022-2026)
            expo = EXPOSURE.get(s, {}).get(app, round(r["Exposure"], 2))
            turn = TURNOVER.get(s, {}).get(app, round(r["TradesPerYr"], 2))
            rows.append({
                "Стратегия": SHORT[s],
                "Net Sharpe": round(r["MeanSharpe"], 4),
                "Ann. Return (%)": round(r["AnnReturn"], 2),
                "Ann. Vol (%)": round(r["AnnVol"], 2),
                "Max Drawdown (%)": round(r["MaxDD"], 2),
                "Экспозиция (%)": expo,
                "Сделок/год": turn,
                "Win Rate (%)": round(r["WinRate"], 2),
            })

        df = pd.DataFrame(rows)
        fname = f"table_{tnum}_approach_{app}.csv"
        df.to_csv(OUT / fname, index=False)
        log(f"  Saved {fname}")

        # Print summary
        valid = df[df["Net Sharpe"].notna()] if "Net Sharpe" in df.columns else df.iloc[0:0]
        if not valid.empty:
            log(f"    Mean Net Sharpe: {valid['Net Sharpe'].mean():.4f}")
            log(f"    Strategies with data: {len(valid)}/{len(STRATEGIES)}")


# ══════════════════════════════════════════════════════════════════════
#  Tables 4.7–4.8: Turnover & Exposure (6×4)
# ══════════════════════════════════════════════════════════════════════
def make_turnover_exposure(data):
    log("\n--- Tables 4.7–4.8: Turnover & Exposure (2022-2026) ---")
    log("  Source: hardcoded from final_summary_v6 (cross-ticker daily)")

    turn_rows = []
    expo_rows = []

    for s in STRATEGIES:
        t_row = {"Стратегия": SHORT[s]}
        e_row = {"Стратегия": SHORT[s]}
        for app in APPROACHES:
            t_row[app] = TURNOVER[s][app]
            e_row[app] = EXPOSURE[s][app]
        turn_rows.append(t_row)
        expo_rows.append(e_row)

    # Mean row
    t_mean = {"Стратегия": "Среднее"}
    e_mean = {"Стратегия": "Среднее"}
    for app in APPROACHES:
        t_mean[app] = round(np.mean([r[app] for r in turn_rows]), 1)
        e_mean[app] = round(np.mean([r[app] for r in expo_rows]), 1)
    turn_rows.append(t_mean)
    expo_rows.append(e_mean)

    turn_df = pd.DataFrame(turn_rows)
    expo_df = pd.DataFrame(expo_rows)

    turn_df.to_csv(OUT / "table_4_7_turnover.csv", index=False)
    expo_df.to_csv(OUT / "table_4_8_exposure.csv", index=False)
    log(f"  Saved table_4_7_turnover.csv")
    log(f"  Saved table_4_8_exposure.csv")

    # Log means
    for app in APPROACHES:
        log(f"  Mean turnover {app}: {t_mean[app]:.1f} trades/yr")
    for app in APPROACHES:
        log(f"  Mean exposure {app}: {e_mean[app]:.1f}%")

    return turn_df, expo_df


# ══════════════════════════════════════════════════════════════════════
#  Table 4.9: Forecast effect summary (A vs BCD)
# ══════════════════════════════════════════════════════════════════════
def make_forecast_effect(data):
    log("\n--- Table 4.9: Forecast effect summary ---")
    comp = data["comparison"]
    ew = comp[comp["table"] == "EW_NetSharpe"].copy()

    rows = []
    for s in STRATEGIES:
        r = ew[ew["strategy"] == s].iloc[0]
        a = float(r["A"])
        b, c, d = float(r["B"]), float(r["C"]), float(r["D"])
        mean_bcd = np.mean([b, c, d])
        best_bcd = max(b, c, d)

        rows.append({
            "Стратегия": SHORT[s],
            "A (baseline)": round(a, 4),
            "Mean(B,C,D)": round(mean_bcd, 4),
            "Best(B,C,D)": round(best_bcd, 4),
            "Δ mean vs A": round(mean_bcd - a, 4),
            "Δ% mean vs A": round((mean_bcd / a - 1) * 100, 1),
            "Δ best vs A": round(best_bcd - a, 4),
            "Δ% best vs A": round((best_bcd / a - 1) * 100, 1),
        })

    # Mean
    vals = {k: np.mean([r[k] for r in rows]) for k in
            ["A (baseline)", "Mean(B,C,D)", "Best(B,C,D)",
             "Δ mean vs A", "Δ best vs A"]}
    rows.append({
        "Стратегия": "Среднее",
        **{k: round(v, 4) for k, v in vals.items()},
        "Δ% mean vs A": round((vals["Mean(B,C,D)"] / vals["A (baseline)"] - 1) * 100, 1),
        "Δ% best vs A": round((vals["Best(B,C,D)"] / vals["A (baseline)"] - 1) * 100, 1),
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_4_9_forecast_effect.csv", index=False)
    log(f"  Saved table_4_9_forecast_effect.csv")
    mean_row = rows[-1]
    log(f"  Overall: mean(BCD) vs A = {mean_row['Δ% mean vs A']:+.1f}%, "
        f"best(BCD) vs A = {mean_row['Δ% best vs A']:+.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.3: Grouped bar — Net Sharpe by strategy × approach
# ══════════════════════════════════════════════════════════════════════
def fig_4_3(data):
    log("\n--- Figure 4.3: Net Sharpe comparison ---")
    comp = data["comparison"]
    ew = comp[comp["table"] == "EW_NetSharpe"].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(STRATEGIES))
    n_app = len(APPROACHES)
    w = 0.18
    offsets = np.array([(i - (n_app - 1) / 2) * w for i in range(n_app)])

    for j, app in enumerate(APPROACHES):
        vals = []
        for s in STRATEGIES:
            r = ew[ew["strategy"] == s]
            vals.append(float(r.iloc[0][app]) if not r.empty else 0)
        bars = ax.bar(x + offsets[j], vals, w,
                      color=APP_COLORS[app], edgecolor="white", lw=0.5,
                      label=APP_LABELS[app])
        # Value labels on top
        for xi, vi in zip(x + offsets[j], vals):
            ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=0)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in STRATEGIES], fontsize=9)
    ax.set_ylabel("Коэффициент Шарпа (net)")
    ax.set_title("Коэффициент Шарпа (net) по стратегиям и подходам, 2022–2026")
    ax.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    p = OUT / "fig_4_3_net_sharpe_comparison.png"
    fig.savefig(p)
    plt.close(fig)
    log(f"  Saved {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4.4: Exposure + Turnover by approach (averages)
# ══════════════════════════════════════════════════════════════════════
def fig_4_4(data):
    log("\n--- Figure 4.4: Exposure & Turnover ---")

    # Means from hardcoded data (cross-ticker daily, 2022-2026)
    mean_expo = {}
    mean_turn = {}
    for app in APPROACHES:
        mean_expo[app] = np.mean([EXPOSURE[s][app] for s in STRATEGIES])
        mean_turn[app] = np.mean([TURNOVER[s][app] for s in STRATEGIES])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    apps = APPROACHES
    x = np.arange(len(apps))
    colors = [APP_COLORS[a] for a in apps]

    # Left: Exposure
    expo_vals = [mean_expo[a] for a in apps]
    bars1 = ax1.bar(x, expo_vals, 0.55, color=colors, edgecolor="white", lw=0.5)
    for xi, vi in zip(x, expo_vals):
        ax1.text(xi, vi + max(expo_vals) * 0.02, f"{vi:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Подход {a}" for a in apps])
    ax1.set_ylabel("Средняя экспозиция (%)")
    ax1.set_title("Экспозиция")
    ax1.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(expo_vals) * 1.2)

    # Right: Turnover
    turn_vals = [mean_turn[a] for a in apps]
    bars2 = ax2.bar(x, turn_vals, 0.55, color=colors, edgecolor="white", lw=0.5)
    for xi, vi in zip(x, turn_vals):
        ax2.text(xi, vi + max(turn_vals) * 0.02, f"{vi:.1f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Подход {a}" for a in apps])
    ax2.set_ylabel("Среднее число сделок / год")
    ax2.set_title("Оборачиваемость")
    ax2.yaxis.grid(True, ls="--", alpha=0.25, lw=0.5)
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(turn_vals) * 1.2)

    fig.suptitle("Средняя экспозиция и оборачиваемость по подходам",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = OUT / "fig_4_4_exposure_turnover.png"
    fig.savefig(p)
    plt.close(fig)
    log(f"  Saved {p.name}")


# ══════════════════════════════════════════════════════════════════════
def main():
    setup_rc()
    log(f"=== Section 4.2 data extraction — {datetime.now():%Y-%m-%d %H:%M} ===\n")

    data = load_all()

    # Tables
    make_table_4_6(data)
    make_passport_tables(data)
    turn_df, expo_df = make_turnover_exposure(data)
    make_forecast_effect(data)

    # Charts
    fig_4_3(data)
    fig_4_4(data)

    # Save log
    log_path = OUT / "data_extraction_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(LOG))
    print(f"\n=== All output saved to {OUT.relative_to(ROOT)}/ ===")


if __name__ == "__main__":
    main()
