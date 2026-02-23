#!/usr/bin/env python3
"""
Detailed weighting method comparison (EW vs InvVol vs MaxSharpe vs MinVar).
Handles 3 commission levels per timeframe: Gross, Net primary, Net secondary.
"""
import pandas as pd
import numpy as np
import os

BASE = "/Users/nilysenok/Desktop/MOEX_ISS/moex_volatility"
TABLE_DIR = os.path.join(BASE, "results/final/strategies/walkforward_v4/tables")

daily = pd.read_csv(os.path.join(TABLE_DIR, "v4_portfolios_daily.csv"))
hourly = pd.read_csv(os.path.join(TABLE_DIR, "v4_portfolios_hourly.csv"))

METHODS = ["EW", "InvVol", "MaxSharpe", "MinVar"]
STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian", "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]


def count_wins(df):
    """Count wins per method across 24 strategy×approach combos."""
    wins = {m: 0 for m in METHODS}
    for strat in STRATEGIES:
        for appr in APPROACHES:
            sub = df[(df["strategy"] == strat) & (df["approach"] == appr)]
            if len(sub) == 0:
                continue
            best_val, best_m = -999, None
            for m in METHODS:
                r = sub[sub["method"] == m]
                if len(r) > 0 and r.iloc[0]["net_sharpe"] > best_val:
                    best_val = r.iloc[0]["net_sharpe"]
                    best_m = m
            if best_m:
                wins[best_m] += 1
    return wins


def ew_vs_method(df, other):
    """Count how often EW > other method."""
    ew_better = 0
    total = 0
    for strat in STRATEGIES:
        for appr in APPROACHES:
            sub = df[(df["strategy"] == strat) & (df["approach"] == appr)]
            r_ew = sub[sub["method"] == "EW"]
            r_other = sub[sub["method"] == other]
            if len(r_ew) > 0 and len(r_other) > 0:
                total += 1
                if r_ew.iloc[0]["net_sharpe"] > r_other.iloc[0]["net_sharpe"]:
                    ew_better += 1
    return ew_better, total


def ew_in_top2(df):
    """Count how often EW is in top-2."""
    count = 0
    total = 0
    for strat in STRATEGIES:
        for appr in APPROACHES:
            sub = df[(df["strategy"] == strat) & (df["approach"] == appr)]
            if len(sub) < 2:
                continue
            total += 1
            ranked = sub.sort_values("net_sharpe", ascending=False)
            top2_methods = ranked["method"].values[:2]
            if "EW" in top2_methods:
                count += 1
    return count, total


def analyze_scenario(df, label):
    """Analyze one commission scenario."""
    wins = count_wins(df)
    mean_sh = df.groupby("method")["net_sharpe"].mean()
    mean_ret = df.groupby("method")["ann_ret_pct"].mean()
    mean_mdd = df.groupby("method")["maxdd_pct"].mean()
    ew_top2, total = ew_in_top2(df)

    return {
        "label": label,
        "wins": wins,
        "mean_sharpe": {m: mean_sh.get(m, 0) for m in METHODS},
        "mean_ret": {m: mean_ret.get(m, 0) for m in METHODS},
        "mean_mdd": {m: mean_mdd.get(m, 0) for m in METHODS},
        "ew_top2": ew_top2,
        "total": total,
        "ew_vs_minvar": ew_vs_method(df, "MinVar"),
        "ew_vs_invvol": ew_vs_method(df, "InvVol"),
        "ew_vs_maxsharpe": ew_vs_method(df, "MaxSharpe"),
    }


def strategy_method_table(df, label):
    """Strategy × Method average Sharpe table."""
    pivot = df.pivot_table(values="net_sharpe", index="strategy", columns="method", aggfunc="mean")
    pivot = pivot.reindex(columns=METHODS)
    pivot = pivot.reindex(STRATEGIES)
    means = pivot.mean()
    pivot.loc["Среднее"] = means
    return pivot


# ═════════════════════════════════════════════════════════════════
print("=" * 90)
print("  WEIGHTING METHOD COMPARISON — ALL SCENARIOS")
print("=" * 90)

for tf_label, df_all, scenarios in [
    ("DAILY", daily, ["gross", "net_0.05", "net_0.06"]),
    ("HOURLY", hourly, ["gross", "net_0.04", "net_0.05"]),
]:
    print(f"\n{'='*90}")
    print(f"  {tf_label}")
    print(f"{'='*90}")

    results = []
    for sc in scenarios:
        sub = df_all[df_all["comm_level"] == sc]
        if len(sub) == 0:
            continue
        res = analyze_scenario(sub, sc)
        results.append(res)

    # ── Table 1: Summary across all scenarios ──
    print(f"\n### Сводная таблица: Sharpe по методам × сценарий ({tf_label})")
    print()
    sc_labels = [r["label"] for r in results]
    header = "| Метод |"
    for sc in sc_labels:
        header += f" {sc} |"
    print(header)
    sep = "|-------|"
    for _ in sc_labels:
        sep += "---:|"
    print(sep)
    for m in METHODS:
        row = f"| {m} |"
        for r in results:
            row += f" {r['mean_sharpe'][m]:.2f} |"
        print(row)

    # ── Table 2: Wins across scenarios ──
    print(f"\n### Побед из 24 × сценарий ({tf_label})")
    print()
    header = "| Метод |"
    for sc in sc_labels:
        header += f" {sc} |"
    print(header)
    sep = "|-------|"
    for _ in sc_labels:
        sep += ":---:|"
    print(sep)
    for m in METHODS:
        row = f"| {m} |"
        for r in results:
            w = r["wins"][m]
            bold = "**" if w == max(r["wins"].values()) else ""
            row += f" {bold}{w}{bold} |"
        print(row)

    # ── Table 3: EW dominance across scenarios ──
    print(f"\n### Доминирование EW × сценарий ({tf_label})")
    print()
    header = "| Метрика |"
    for sc in sc_labels:
        header += f" {sc} |"
    print(header)
    sep = "|---------|"
    for _ in sc_labels:
        sep += ":---:|"
    print(sep)

    for metric_name, get_val in [
        ("EW побед", lambda r: f"{r['wins']['EW']}/24"),
        ("EW в топ-2", lambda r: f"{r['ew_top2']}/{r['total']}"),
        ("EW > MinVar", lambda r: f"{r['ew_vs_minvar'][0]}/{r['ew_vs_minvar'][1]}"),
        ("EW > InvVol", lambda r: f"{r['ew_vs_invvol'][0]}/{r['ew_vs_invvol'][1]}"),
        ("EW > MaxSharpe", lambda r: f"{r['ew_vs_maxsharpe'][0]}/{r['ew_vs_maxsharpe'][1]}"),
    ]:
        row = f"| {metric_name} |"
        for r in results:
            row += f" {get_val(r)} |"
        print(row)

    # ── Table 4: Strategy × Method for primary net scenario ──
    primary_sc = scenarios[1]  # net_0.05 or net_0.04
    sub = df_all[df_all["comm_level"] == primary_sc]
    pivot = strategy_method_table(sub, primary_sc)

    print(f"\n### Средний Sharpe по стратегии × метод ({tf_label}, {primary_sc})")
    print()
    print(f"| Стратегия | EW | InvVol | MaxSharpe | MinVar | Best |")
    print(f"|-----------|---:|-------:|----------:|-------:|------|")
    for idx in pivot.index:
        row_data = pivot.loc[idx]
        best_m = row_data[METHODS].idxmax()
        parts = []
        for m in METHODS:
            v = row_data[m]
            s = f"{v:.2f}"
            if m == best_m:
                s = f"**{s}**"
            parts.append(s)
        name = f"**{idx}**" if idx == "Среднее" else idx
        print(f"| {name} | {' | '.join(parts)} | {best_m} |")

    # ── Table 5: Full 24-combo for primary net ──
    wins_primary = count_wins(sub)
    mean_sh_primary = sub.groupby("method")["net_sharpe"].mean()
    mean_ret_primary = sub.groupby("method")["ann_ret_pct"].mean()
    mean_mdd_primary = sub.groupby("method")["maxdd_pct"].mean()

    print(f"\n### Средние метрики по методам ({tf_label}, {primary_sc})")
    print()
    print(f"| Метод | Побед из 24 | Ср. Sharpe | Ср. Ret% | Ср. MDD% |")
    print(f"|-------|:----------:|----------:|--------:|---------:|")
    for m in METHODS:
        w = wins_primary[m]
        bold = "**" if w == max(wins_primary.values()) else ""
        print(f"| {bold}{m}{bold} | {bold}{w}{bold} | "
              f"{mean_sh_primary.get(m,0):.2f} | {mean_ret_primary.get(m,0):.2f} | "
              f"{mean_mdd_primary.get(m,0):.2f} |")

# Save summary CSVs
for tf_label, df_all, scenarios in [
    ("daily", daily, ["gross", "net_0.05", "net_0.06"]),
    ("hourly", hourly, ["gross", "net_0.04", "net_0.05"]),
]:
    rows = []
    for sc in scenarios:
        sub = df_all[df_all["comm_level"] == sc]
        if len(sub) == 0:
            continue
        wins = count_wins(sub)
        mean_sh = sub.groupby("method")["net_sharpe"].mean()
        mean_ret = sub.groupby("method")["ann_ret_pct"].mean()
        mean_mdd = sub.groupby("method")["maxdd_pct"].mean()
        for m in METHODS:
            rows.append({
                "comm_level": sc, "method": m,
                "wins_24": wins[m],
                "mean_sharpe": round(mean_sh.get(m, 0), 4),
                "mean_ret_pct": round(mean_ret.get(m, 0), 2),
                "mean_mdd_pct": round(mean_mdd.get(m, 0), 2),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLE_DIR, f"v4_weighting_summary_{tf_label}.csv"), index=False)

print(f"\nSaved: v4_weighting_summary_daily.csv, v4_weighting_summary_hourly.csv")
