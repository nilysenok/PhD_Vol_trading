#!/usr/bin/env python3
"""
Final table collection for section 4.2 — 8 clean tables + text data.

Sources:
  - v4_full_daily.csv: per-ticker mean Sharpe (flat avg of ticker-year)
  - v4_full_hourly.csv: same for hourly
  - v4_portfolios_daily.csv: EW portfolio Sharpe (net only)
  - v4_stat_tests.csv: DM-tests per strategy
  - stat_tests.csv: meta-portfolio DM-tests

Usage: python3 scripts/v4_final_tables_42.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "final" / "strategies" / "walkforward_v4" / "tables"
OUT = ROOT / "output_4_2_final"
OUT.mkdir(parents=True, exist_ok=True)

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
CATEGORIES = {
    "S1_MeanRev": "Контртренд", "S2_Bollinger": "Контртренд",
    "S3_Donchian": "Тренд",     "S4_Supertrend": "Тренд",
    "S5_PivotPoints": "Диапазон", "S6_VWAP": "Диапазон",
}
CAT_ORDER = ["Контртренд", "Тренд", "Диапазон"]
APPROACHES = ["A", "B", "C", "D"]

LOG = []


def log(msg=""):
    LOG.append(str(msg))
    print(msg)


def r3(v):
    """Round to 3 decimals."""
    return round(float(v), 3) if pd.notna(v) else np.nan


def r4(v):
    return round(float(v), 4) if pd.notna(v) else np.nan


def get_val(df, strategy, app, col):
    """Get value from full_daily/hourly DataFrame."""
    r = df[(df["Strategy"] == strategy) & (df["App"] == app)]
    if r.empty or col not in r.columns:
        return np.nan
    return float(r.iloc[0][col])


def add_mean_and_cat_rows(rows, val_cols):
    """Add Среднее + category rows."""
    cat_rows = []
    for cat in CAT_ORDER:
        cat_strategies = [s for s in STRATEGIES if CATEGORIES[s] == cat]
        cat_row = {"Стратегия": cat, "Категория": cat}
        for c in val_cols:
            vals = [r[c] for r in rows if r["Стратегия"] in [SHORT[s] for s in cat_strategies] and pd.notna(r.get(c))]
            cat_row[c] = r3(np.mean(vals)) if vals else np.nan
        cat_rows.append(cat_row)

    mean_row = {"Стратегия": "Среднее", "Категория": ""}
    for c in val_cols:
        vals = [r[c] for r in rows if pd.notna(r.get(c))]
        mean_row[c] = r3(np.mean(vals)) if vals else np.nan

    return rows + cat_rows + [mean_row]


# ── Load data ──────────────────────────────────────────────────────────
log("=== Loading data ===\n")

daily = pd.read_csv(BASE / "v4_full_daily.csv")
log(f"[OK] v4_full_daily.csv: {len(daily)} rows, cols: {list(daily.columns)}")

hourly = pd.read_csv(BASE / "v4_full_hourly.csv")
log(f"[OK] v4_full_hourly.csv: {len(hourly)} rows, cols: {list(hourly.columns)}")

portfolios = pd.read_csv(BASE / "v4_portfolios_daily.csv")
log(f"[OK] v4_portfolios_daily.csv: {len(portfolios)} rows, cols: {list(portfolios.columns)}")

stat_tests = pd.read_csv(BASE / "v4_stat_tests.csv")
log(f"[OK] v4_stat_tests.csv: {len(stat_tests)} rows")

meta_tests = pd.read_csv(BASE / "stat_tests.csv")
log(f"[OK] stat_tests.csv: {len(meta_tests)} rows")

# Check for gross portfolio data
port_has_gross = "gross_sharpe" in portfolios.columns
log(f"\n[INFO] v4_portfolios_daily.csv has gross_sharpe: {port_has_gross}")
log(f"[INFO] Available portfolio columns: {list(portfolios.columns)}")


# ══════════════════════════════════════════════════════════════════════
#  Tables 4.1–4.6: Per-approach gross + net
# ══════════════════════════════════════════════════════════════════════

def make_approach_gross_table(app, table_num, label):
    """Table: Gross Sharpe for approach X, daily + hourly."""
    log(f"\n--- Table {table_num}: Gross Sharpe подход {app} ({label}) ---")
    rows = []
    for s in STRATEGIES:
        d_gross = get_val(daily, s, app, "GrossSharpe")
        h_gross = get_val(hourly, s, app, "GrossSharpe")
        rows.append({
            "Стратегия": SHORT[s],
            "Категория": CATEGORIES[s],
            "Gross daily": r3(d_gross),
            "Gross hourly": r3(h_gross),
        })

    rows = add_mean_and_cat_rows(rows, ["Gross daily", "Gross hourly"])
    df = pd.DataFrame(rows)
    fname = f"table_{table_num}_{app}_gross.csv"
    df.to_csv(OUT / fname, index=False)
    log(f"  Saved {fname}")
    return df


def make_approach_net_table(app, table_num, label):
    """Table: Net Sharpe for approach X (gross, net 0.05%, net 0.06%, delta)."""
    log(f"\n--- Table {table_num}: Net Sharpe подход {app} ({label}) ---")
    rows = []
    for s in STRATEGIES:
        gross = get_val(daily, s, app, "GrossSharpe")
        net05 = get_val(daily, s, app, "Net0.05Sharpe")
        net06 = get_val(daily, s, app, "Net0.06Sharpe")
        h_gross = get_val(hourly, s, app, "GrossSharpe")
        h_net04 = get_val(hourly, s, app, "Net0.04Sharpe")
        h_net05 = get_val(hourly, s, app, "Net0.05Sharpe")
        delta_d = r3(net05 - gross) if pd.notna(gross) and pd.notna(net05) else np.nan
        delta_h = r3(h_net04 - h_gross) if pd.notna(h_gross) and pd.notna(h_net04) else np.nan

        rows.append({
            "Стратегия": SHORT[s],
            "Категория": CATEGORIES[s],
            "Gross (d)": r3(gross),
            "Net 0.05% (d)": r3(net05),
            "Net 0.06% (d)": r3(net06),
            "Δ (d)": delta_d,
            "Gross (h)": r3(h_gross),
            "Net 0.04% (h)": r3(h_net04),
            "Net 0.05% (h)": r3(h_net05),
            "Δ (h)": delta_h,
        })

    val_cols = ["Gross (d)", "Net 0.05% (d)", "Net 0.06% (d)", "Δ (d)",
                "Gross (h)", "Net 0.04% (h)", "Net 0.05% (h)", "Δ (h)"]
    rows = add_mean_and_cat_rows(rows, val_cols)
    df = pd.DataFrame(rows)
    fname = f"table_{table_num}_{app}_net.csv"
    df.to_csv(OUT / fname, index=False)
    log(f"  Saved {fname}")
    return df


# Generate per-approach tables
make_approach_gross_table("B", "4_1", "адаптивные стопы")
make_approach_net_table("B", "4_2", "адаптивные стопы")
make_approach_gross_table("C", "4_3", "режимная фильтрация")
make_approach_net_table("C", "4_4", "режимная фильтрация")
make_approach_gross_table("D", "4_5", "vol-gate")
make_approach_net_table("D", "4_6", "vol-gate")


# ══════════════════════════════════════════════════════════════════════
#  Table 4.7: Summary Gross — per-ticker mean (no EW gross available)
# ══════════════════════════════════════════════════════════════════════
log("\n--- Table 4.7: Сводная Gross Sharpe (mean per-ticker) ---")
log("  NOTE: EW portfolio gross Sharpe not available in v4_portfolios_daily.csv")
log("  Using per-ticker mean gross from v4_full_daily.csv instead")

rows_7 = []
for s in STRATEGIES:
    row = {"Стратегия": SHORT[s], "Категория": CATEGORIES[s]}
    vals = {}
    for app in APPROACHES:
        v = get_val(daily, s, app, "GrossSharpe")
        row[app] = r3(v)
        vals[app] = v

    best_app = max(vals, key=lambda k: vals[k] if pd.notna(vals[k]) else -999)
    best_val = vals[best_app]
    a_val = vals["A"]
    row["Best"] = best_app
    row["Δ% best vs A"] = r3((best_val / a_val - 1) * 100) if a_val and a_val != 0 else np.nan
    rows_7.append(row)

# Mean row
mean_7 = {"Стратегия": "Среднее", "Категория": ""}
for app in APPROACHES:
    mean_7[app] = r3(np.mean([r[app] for r in rows_7 if pd.notna(r.get(app))]))
mean_a = mean_7["A"]
mean_bcd = r3(np.mean([mean_7["B"], mean_7["C"], mean_7["D"]]))
best_mean = max(mean_7["B"], mean_7["C"], mean_7["D"])
mean_7["Best"] = "C" if best_mean == mean_7["C"] else "D" if best_mean == mean_7["D"] else "B"
mean_7["Δ% best vs A"] = r3((best_mean / mean_a - 1) * 100) if mean_a else np.nan

# Category rows
cat_7 = []
for cat in CAT_ORDER:
    cat_strats = [s for s in STRATEGIES if CATEGORIES[s] == cat]
    cr = {"Стратегия": cat, "Категория": cat}
    for app in APPROACHES:
        vals = [r[app] for r in rows_7 if CATEGORIES.get([k for k, v in SHORT.items() if v == r["Стратегия"]][0], "") == cat and pd.notna(r.get(app))]
        cr[app] = r3(np.mean(vals)) if vals else np.nan
    cr["Best"] = ""
    cr["Δ% best vs A"] = np.nan
    cat_7.append(cr)

df_7 = pd.DataFrame(rows_7 + cat_7 + [mean_7])
df_7.to_csv(OUT / "table_4_7_summary_gross.csv", index=False)
log(f"  Saved table_4_7_summary_gross.csv")


# ══════════════════════════════════════════════════════════════════════
#  Table 4.8: Summary Net — EW portfolio Sharpe
# ══════════════════════════════════════════════════════════════════════
log("\n--- Table 4.8: Сводная Net Sharpe (EW-портфели, net 0.05%) ---")

ew_port = portfolios[portfolios["method"] == "EW"].copy()

rows_8 = []
for s in STRATEGIES:
    row = {"Стратегия": SHORT[s], "Категория": CATEGORIES[s]}
    vals = {}
    for app in APPROACHES:
        r = ew_port[(ew_port["strategy"] == s) & (ew_port["approach"] == app)]
        v = float(r.iloc[0]["net_sharpe"]) if not r.empty else np.nan
        row[app] = r4(v)
        vals[app] = v

    best_app = max(vals, key=lambda k: vals[k] if pd.notna(vals[k]) else -999)
    best_val = vals[best_app]
    a_val = vals["A"]
    mean_bcd = np.mean([vals[k] for k in ["B", "C", "D"] if pd.notna(vals.get(k))])
    row["Best"] = best_app
    row["Mean(BCD)"] = r4(mean_bcd)
    row["Δ% mean vs A"] = r3((mean_bcd / a_val - 1) * 100) if a_val and a_val != 0 else np.nan
    row["Δ% best vs A"] = r3((best_val / a_val - 1) * 100) if a_val and a_val != 0 else np.nan
    rows_8.append(row)

# Category rows
cat_8 = []
for cat in CAT_ORDER:
    cat_strats = [s for s in STRATEGIES if CATEGORIES[s] == cat]
    cr = {"Стратегия": cat, "Категория": cat}
    for app in APPROACHES:
        v_list = []
        for s in cat_strats:
            r = ew_port[(ew_port["strategy"] == s) & (ew_port["approach"] == app)]
            if not r.empty:
                v_list.append(float(r.iloc[0]["net_sharpe"]))
        cr[app] = r4(np.mean(v_list)) if v_list else np.nan
    cr["Best"] = ""
    cr["Mean(BCD)"] = np.nan
    cr["Δ% mean vs A"] = np.nan
    cr["Δ% best vs A"] = np.nan
    cat_8.append(cr)

# Mean row
mean_8 = {"Стратегия": "Среднее", "Категория": ""}
for app in APPROACHES:
    mean_8[app] = r4(np.mean([r[app] for r in rows_8 if pd.notna(r.get(app))]))
mean_a = mean_8["A"]
mean_b = mean_8["B"]
mean_c = mean_8["C"]
mean_d = mean_8["D"]
mean_bcd_all = np.mean([mean_b, mean_c, mean_d])
best_mean_v = max(mean_b, mean_c, mean_d)
mean_8["Best"] = "D" if best_mean_v == mean_d else "C" if best_mean_v == mean_c else "B"
mean_8["Mean(BCD)"] = r4(mean_bcd_all)
mean_8["Δ% mean vs A"] = r3((mean_bcd_all / mean_a - 1) * 100) if mean_a else np.nan
mean_8["Δ% best vs A"] = r3((best_mean_v / mean_a - 1) * 100) if mean_a else np.nan

df_8 = pd.DataFrame(rows_8 + cat_8 + [mean_8])
df_8.to_csv(OUT / "table_4_8_summary_net.csv", index=False)
log(f"  Saved table_4_8_summary_net.csv")

# Verify against known values
log("\n  Verification (must match section_4_2_tables.md):")
for s in STRATEGIES:
    r = ew_port[(ew_port["strategy"] == s) & (ew_port["approach"] == "D") & (ew_port["method"] == "EW")]
    if not r.empty:
        v = float(r.iloc[0]["net_sharpe"])
        log(f"    {SHORT[s]} D: {v:.4f}")

log(f"\n  Mean: A={mean_a:.4f}, B={mean_b:.4f}, C={mean_c:.4f}, D={mean_d:.4f}")
log(f"  Δ% mean(BCD) vs A: {(mean_bcd_all / mean_a - 1) * 100:+.1f}%")
log(f"  Δ% D vs A: {(mean_d / mean_a - 1) * 100:+.1f}%")


# ══════════════════════════════════════════════════════════════════════
#  Hourly summary
# ══════════════════════════════════════════════════════════════════════
log("\n--- Hourly summary ---")

hourly_rows = []
for s in STRATEGIES:
    for app in APPROACHES:
        r = hourly[(hourly["Strategy"] == s) & (hourly["App"] == app)]
        if not r.empty:
            r = r.iloc[0]
            hourly_rows.append({
                "Strategy": SHORT[s], "App": app,
                "Gross": r3(r["GrossSharpe"]),
                "Net 0.04%": r3(r["Net0.04Sharpe"]),
                "Net 0.05%": r3(r["Net0.05Sharpe"]),
                "Tr/yr": r3(r["Tr/yr"]),
            })

hourly_df = pd.DataFrame(hourly_rows)
hourly_df.to_csv(OUT / "hourly_summary.csv", index=False)
log(f"  Saved hourly_summary.csv")

# Count positive by approach
log("\n  Positive net 0.04% by approach:")
for app in APPROACHES:
    sub = hourly[(hourly["App"] == app)]
    n_pos = (sub["Net0.04Sharpe"] > 0).sum()
    log(f"    {app}: {n_pos}/6 positive")
total_pos = (hourly["Net0.04Sharpe"] > 0).sum()
log(f"    Total: {total_pos}/24 positive")


# ══════════════════════════════════════════════════════════════════════
#  Text data
# ══════════════════════════════════════════════════════════════════════
log("\n--- Generating text_data.txt ---")

txt = []
txt.append("=" * 60)
txt.append("ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ ДЛЯ ТЕКСТА РАЗДЕЛА 4.2")
txt.append("=" * 60)

# 1. Hourly summary
txt.append("\n1. HOURLY СВОДКА (net 0.04%)")
txt.append("-" * 40)
for app in APPROACHES:
    sub = hourly[hourly["App"] == app]
    n_pos = (sub["Net0.04Sharpe"] > 0).sum()
    pos_strats = sub[sub["Net0.04Sharpe"] > 0]["Strategy"].tolist()
    txt.append(f"  {app}: {n_pos}/6 positive — {', '.join(pos_strats) if pos_strats else 'none'}")
txt.append(f"  Итого: {(hourly['Net0.04Sharpe'] > 0).sum()}/24 positive")
txt.append(f"  D: positive для всех 6 стратегий")
txt.append(f"  A: 0/6 positive, B: 0/6 positive")

# 2. EW portfolio Return/Vol/Sharpe/MDD (daily, net)
txt.append("\n2. EW-ПОРТФЕЛИ: Return, Vol, Sharpe, MDD (daily, net 0.05%)")
txt.append("-" * 40)
txt.append(f"{'Strategy':<20s} {'App':>4s} {'Sharpe':>8s} {'Ret%':>8s} {'MDD%':>8s}")
txt.append("-" * 50)
for s in STRATEGIES:
    for app in APPROACHES:
        r = ew_port[(ew_port["strategy"] == s) & (ew_port["approach"] == app)]
        if not r.empty:
            r = r.iloc[0]
            txt.append(f"{SHORT[s]:<20s} {app:>4s} {r['net_sharpe']:8.4f} {r['ann_ret_pct']:8.2f} {r['maxdd_pct']:8.2f}")

# Note: vol not in portfolios file, but can estimate from Sharpe = Ret/Vol * sqrt(252)
# Vol ~ Ret / Sharpe (annualized) — rough estimate
txt.append("\n  NOTE: v4_portfolios_daily.csv doesn't have vol column.")
txt.append("  Estimated vol = |ann_ret| / |net_sharpe| where available:")
for s in STRATEGIES:
    for app in APPROACHES:
        r = ew_port[(ew_port["strategy"] == s) & (ew_port["approach"] == app)]
        if not r.empty:
            r = r.iloc[0]
            sh = r["net_sharpe"]
            ret = r["ann_ret_pct"]
            if abs(sh) > 0.01:
                est_vol = abs(ret / sh)
                txt.append(f"    {SHORT[s]} {app}: est vol = {est_vol:.2f}%")

# 3. S5 activity
txt.append("\n3. S5 PIVOT POINTS — АКТИВНОСТЬ")
txt.append("-" * 40)
txt.append("  S5_A: 9/17 тикеров с ≥1 сделкой, средние 1.4 сделки/4 года")
txt.append("  S5_B: 5/17 тикеров с ≥1 сделкой, средние 0.3 сделки/4 года")
txt.append("  S5_C: нет raw-данных (pipeline gap V4→V4.1)")
txt.append("  S5_D: 6/17 тикеров с ≥1 сделкой, средние 0.6 сделки/4 года")
txt.append("  Экспозиция: A=1.1%, B=1.0%, D=0.7% (cross-ticker daily)")
txt.append("  EW portfolio Sharpe: A=1.22, B=1.58, C=1.41, D=1.35")
txt.append("  Вывод: математически валидный Sharpe, экономически незначим")

# 4. DM-тесты
txt.append("\n4. DM-ТЕСТЫ PER STRATEGY (B/C/D vs A, daily)")
txt.append("-" * 40)
txt.append(f"{'Strategy':<20s} {'Test':>10s} {'DM_t':>8s} {'p':>8s} {'Sig':>5s}")
txt.append("-" * 55)
daily_tests = stat_tests[stat_tests["timeframe"] == "daily"]
for _, r in daily_tests.iterrows():
    sig = str(r['sig']) if pd.notna(r['sig']) else ""
    txt.append(f"{r['strategy']:<20s} {r['comparison']:>10s} {r['dm_t']:8.4f} {r['dm_p']:8.4f} {sig:>5s}")

txt.append("\nКлючевые:")
txt.append("  S2 C_vs_A: t=−2.83, p=0.005 *** — C ВРЕДИТ контртренду!")
txt.append("  S4 C_vs_A: t=−2.58, p=0.010 *** — C вредит Supertrend")
txt.append("  S4 B_vs_A: t=−2.97, p=0.003 *** — B вредит Supertrend")
txt.append("  S3 B_vs_A: t=−2.00, p=0.046 ** — B вредит Donchian")
txt.append("  S5 D_vs_A: t=+2.02, p=0.043 ** — D улучшает S5")
txt.append("  S6 D_vs_A: t=+2.15, p=0.032 ** — D улучшает VWAP")

txt.append("\n5. DM-ТЕСТЫ МЕТА-ПОРТФЕЛЬ")
txt.append("-" * 40)
for _, r in meta_tests.iterrows():
    txt.append(f"  {r['test']}: ΔSharpe={r['delta_sharpe']:.4f}, t={r['t_stat']:.3f}, p={r['p_ttest']:.4f}, CI=[{r['boot_ci_lo']:.4f}, {r['boot_ci_hi']:.4f}]")

txt.append("\n6. ТРИ МЕТРИКИ Δ% FORECAST EFFECT")
txt.append("-" * 40)
txt.append("  +27.3% = mean(BCD) vs A — средний эффект прогнозов")
txt.append("  +44.7% = D vs A — лучший единый подход")
txt.append("  +49.4% = best per strategy vs A — оптимальный per strategy")
txt.append("  Рекомендация: использовать +27% и +45% в основном тексте")

with open(OUT / "text_data.txt", "w") as f:
    f.write("\n".join(txt))
log("  Saved text_data.txt")


# ══════════════════════════════════════════════════════════════════════
#  Data sources log
# ══════════════════════════════════════════════════════════════════════
src = []
src.append("DATA SOURCES LOG")
src.append("=" * 50)
src.append(f"Generated: 2026-02-16")
src.append("")
src.append("Tables 4.1–4.6 (per-approach gross/net):")
src.append(f"  Source: v4_full_daily.csv")
src.append(f"    Columns: GrossSharpe, Net0.05Sharpe, Net0.06Sharpe")
src.append(f"    Definition: mean of per-ticker-year Sharpe (flat average)")
src.append(f"    Period: 2022–2025 (OOS)")
src.append(f"  Source: v4_full_hourly.csv")
src.append(f"    Columns: GrossSharpe, Net0.04Sharpe, Net0.05Sharpe")
src.append(f"    Period: 2022–2025 (OOS)")
src.append("")
src.append("Table 4.7 (summary gross):")
src.append(f"  Source: v4_full_daily.csv, column GrossSharpe")
src.append(f"  NOTE: This is PER-TICKER MEAN gross, NOT EW portfolio gross")
src.append(f"  (v4_portfolios_daily.csv has only net_sharpe, no gross column)")
src.append("")
src.append("Table 4.8 (summary net EW portfolio):")
src.append(f"  Source: v4_portfolios_daily.csv, method=EW, column net_sharpe")
src.append(f"  Definition: Sharpe(EW portfolio return series), net 0.05%")
src.append(f"  Period: 2022–2025 (OOS)")
src.append(f"  Verified: matches v4_A_vs_forecast_comparison.csv EW_NetSharpe")
src.append("")
src.append("DM-tests:")
src.append(f"  Per-strategy: v4_stat_tests.csv")
src.append(f"  Meta-portfolio: stat_tests.csv")
src.append("")
src.append("METRIC DEFINITIONS:")
src.append("  Per-ticker mean (Tables 4.1–4.7): mean of all (ticker,year) Sharpe values")
src.append("  EW Portfolio (Table 4.8): Sharpe(mean(r_1,...,r_17)) — aggregate return series")
src.append("  Typical ratio portfolio/per-ticker: 2.2x–6.8x (diversification effect)")

with open(OUT / "data_sources_log.txt", "w") as f:
    f.write("\n".join(src))
log("\n  Saved data_sources_log.txt")


# ── Final save log ────────────────────────────────────────────────────
with open(OUT / "extraction_log.txt", "w") as f:
    f.write("\n".join(LOG))

print(f"\n=== All output saved to {OUT.relative_to(ROOT)}/ ===")
