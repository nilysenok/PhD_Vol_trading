#!/usr/bin/env python3
"""Build V4 summary tables: Gross + Net Sharpe at two commission levels."""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
BCD_YEARS = [2022, 2023, 2024, 2025]

# Commission levels per side
COMM_DAILY  = [0.0, 0.0005, 0.0006]   # gross, 0.05%, 0.06%
COMM_HOURLY = [0.0, 0.0004, 0.0005]   # gross, 0.04%, 0.05%


def compute_metrics(grp, commission, is_hourly):
    """Compute Sharpe and AnnReturn for a ticker-year group."""
    pos = grp["position"].values
    gross_r = grp["daily_gross_return"].values
    n = len(pos)
    if n == 0:
        return None

    # Net returns
    dpos = np.diff(pos, prepend=0.0)
    comm_cost = np.abs(dpos) * commission
    net_r = gross_r - comm_cost

    # Sharpe
    bpy = 252 * 9 if is_hourly else 252
    ann = np.sqrt(bpy)
    std_r = np.std(net_r, ddof=1) if n > 1 else 1e-10
    sharpe = np.mean(net_r) / std_r * ann if std_r > 1e-12 else 0.0
    ann_ret = np.mean(net_r) * bpy

    # Trades (round-trips)
    trades = 0
    in_t = False
    for i in range(n):
        if pos[i] != 0 and not in_t:
            trades += 1
            in_t = True
        if in_t and pos[i] == 0:
            in_t = False
    years = max(n / bpy, 0.5)
    tpy = trades / years

    return {"sharpe": sharpe, "ann_ret": ann_ret * 100, "tpy": tpy}


def build_table(pos_df, tf, comm_levels):
    """Build full table for one timeframe."""
    is_hourly = tf == "hourly"
    sub = pos_df[(pos_df["tf"] == tf) & (pos_df["test_year"].isin(BCD_YEARS))].copy()

    rows = []
    for strat in STRATEGIES:
        for appr in APPROACHES:
            chunk = sub[(sub["strategy"] == strat) & (sub["approach"] == appr)]
            if len(chunk) == 0:
                rows.append({
                    "Strategy": strat, "App": appr,
                    "GrossSharpe": np.nan,
                    f"Net{comm_levels[1]*100:.2f}Sharpe": np.nan,
                    f"Net{comm_levels[2]*100:.2f}Sharpe": np.nan,
                    "Tr/yr": np.nan,
                    "GrossRet%": np.nan,
                    f"Net{comm_levels[1]*100:.2f}Ret%": np.nan,
                    f"Net{comm_levels[2]*100:.2f}Ret%": np.nan,
                })
                continue

            # Compute per ticker, then mean
            results_by_comm = {c: {"sharpes": [], "rets": [], "tpys": []}
                               for c in comm_levels}

            for ticker in sorted(chunk["ticker"].unique()):
                for year in BCD_YEARS:
                    grp = chunk[(chunk["ticker"] == ticker) & (chunk["test_year"] == year)]
                    if len(grp) == 0:
                        continue
                    for c in comm_levels:
                        m = compute_metrics(grp, c, is_hourly)
                        if m is not None:
                            results_by_comm[c]["sharpes"].append(m["sharpe"])
                            results_by_comm[c]["rets"].append(m["ann_ret"])
                            results_by_comm[c]["tpys"].append(m["tpy"])

            gross_c = comm_levels[0]
            net1_c = comm_levels[1]
            net2_c = comm_levels[2]

            def safe_mean(lst):
                return np.mean(lst) if lst else np.nan

            row = {
                "Strategy": strat,
                "App": appr,
                "GrossSharpe": round(safe_mean(results_by_comm[gross_c]["sharpes"]), 4),
                f"Net{net1_c*100:.2f}Sharpe": round(safe_mean(results_by_comm[net1_c]["sharpes"]), 4),
                f"Net{net2_c*100:.2f}Sharpe": round(safe_mean(results_by_comm[net2_c]["sharpes"]), 4),
                "Tr/yr": round(safe_mean(results_by_comm[gross_c]["tpys"]), 2),
                "GrossRet%": round(safe_mean(results_by_comm[gross_c]["rets"]), 2),
                f"Net{net1_c*100:.2f}Ret%": round(safe_mean(results_by_comm[net1_c]["rets"]), 2),
                f"Net{net2_c*100:.2f}Ret%": round(safe_mean(results_by_comm[net2_c]["rets"]), 2),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def print_table(df, title, comm_labels):
    """Print formatted table."""
    c1, c2 = comm_labels
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    hdr = (f"{'Strategy':>16s} {'App':>3s} | {'Gross':>7s} {c1:>9s} {c2:>9s} | "
           f"{'Tr/yr':>6s} | {'GrossR%':>8s} {c1+'R%':>10s} {c2+'R%':>10s}")
    print(hdr)
    print("-" * len(hdr))

    cols = list(df.columns)
    gross_sh = cols[2]
    net1_sh = cols[3]
    net2_sh = cols[4]
    tpy_col = cols[5]
    gross_ret = cols[6]
    net1_ret = cols[7]
    net2_ret = cols[8]

    prev_strat = None
    for _, r in df.iterrows():
        strat_disp = r["Strategy"] if r["Strategy"] != prev_strat else ""
        prev_strat = r["Strategy"]

        def fmt(v, w=7):
            if pd.isna(v):
                return " " * (w - 1) + "-"
            return f"{v:{w}.4f}" if abs(v) < 100 else f"{v:{w}.2f}"

        def fmt_ret(v, w=8):
            if pd.isna(v):
                return " " * (w - 1) + "-"
            return f"{v:{w}.2f}"

        def fmt_tpy(v, w=6):
            if pd.isna(v):
                return " " * (w - 1) + "-"
            return f"{v:{w}.2f}"

        line = (f"{strat_disp:>16s} {r['App']:>3s} | "
                f"{fmt(r[gross_sh]):>7s} {fmt(r[net1_sh], 9):>9s} {fmt(r[net2_sh], 9):>9s} | "
                f"{fmt_tpy(r[tpy_col]):>6s} | "
                f"{fmt_ret(r[gross_ret]):>8s} {fmt_ret(r[net1_ret], 10):>10s} {fmt_ret(r[net2_ret], 10):>10s}")
        print(line)


def main():
    print("Loading data...")
    pos_df = pd.read_parquet(OUT_DIR / "daily_positions.parquet")
    print(f"  Loaded: {len(pos_df):,} rows")

    # ── DAILY ──
    print("\nComputing daily table...")
    df_daily = build_table(pos_df, "daily", COMM_DAILY)
    print_table(df_daily, "TABLE 1: DAILY (BCD years 2022-2025, mean across 17 tickers)",
                ["Net0.05%", "Net0.06%"])
    df_daily.to_csv(OUT_DIR / "tables" / "v4_full_daily.csv", index=False)

    # ── HOURLY ──
    print("\nComputing hourly table...")
    df_hourly = build_table(pos_df, "hourly", COMM_HOURLY)
    print_table(df_hourly, "TABLE 2: HOURLY (BCD years 2022-2025, mean across 17 tickers)",
                ["Net0.04%", "Net0.05%"])
    df_hourly.to_csv(OUT_DIR / "tables" / "v4_full_hourly.csv", index=False)

    print(f"\nSaved: v4_full_daily.csv, v4_full_hourly.csv")


if __name__ == "__main__":
    main()
