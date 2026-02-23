#!/usr/bin/env python3
"""V4 A vs Forecast comparison tables from precomputed portfolio data."""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4" / "tables"

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
APPROACHES = ["A", "B", "C", "D"]
METHODS = ["EW", "InvVol", "MaxSharpe", "MinVar"]


def get_val(df, strat, appr, method, col):
    row = df[(df["strategy"] == strat) & (df["approach"] == appr) & (df["method"] == method)]
    if len(row) == 0:
        return np.nan
    return row.iloc[0][col]


def print_comparison_table(title, data, col_headers, fmt=".4f", show_best_app=True):
    """Print a comparison table.
    data: list of dicts with keys matching col_headers + 'strategy'
    """
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    # Header
    w_s = 16
    w_v = 8
    hdr = f"{'Strategy':>{w_s}}"
    for c in col_headers:
        if c == "best_app":
            hdr += f" {'best':>6s}"
        else:
            hdr += f" {c:>{w_v}s}"
    print(hdr)
    print("-" * len(hdr))

    for row in data:
        if row.get("_sep"):
            print("-" * len(hdr))
            continue
        s = row.get("strategy", "")
        line = f"{s:>{w_s}}"
        for c in col_headers:
            v = row.get(c)
            if c == "best_app":
                line += f" {str(v):>6s}"
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                line += f" {'—':>{w_v}s}"
            elif isinstance(v, str):
                line += f" {v:>{w_v}s}"
            else:
                line += f" {v:{w_v}{fmt}}"
        print(line)


def build_table(df, metric_col, method_mode="EW"):
    """Build A vs B vs C vs D comparison.
    method_mode: "EW" = use EW only, "best" = best method per cell.
    """
    rows = []
    avg_a, avg_b, avg_c, avg_d = [], [], [], []

    for strat in STRATEGIES:
        if method_mode == "EW":
            a = get_val(df, strat, "A", "EW", metric_col)
            b = get_val(df, strat, "B", "EW", metric_col)
            c = get_val(df, strat, "C", "EW", metric_col)
            d = get_val(df, strat, "D", "EW", metric_col)
        else:
            # Best method per cell
            a = df[(df["strategy"] == strat) & (df["approach"] == "A")][metric_col].max()
            b = df[(df["strategy"] == strat) & (df["approach"] == "B")][metric_col].max()
            c = df[(df["strategy"] == strat) & (df["approach"] == "C")][metric_col].max()
            d = df[(df["strategy"] == strat) & (df["approach"] == "D")][metric_col].max()

        bcd_vals = [v for v in [b, c, d] if not np.isnan(v)]
        mean_bcd = np.mean(bcd_vals) if bcd_vals else np.nan
        best_bcd = np.max(bcd_vals) if bcd_vals else np.nan

        # Which approach is best among B,C,D
        best_app = ""
        if bcd_vals:
            idx = np.argmax([b, c, d])
            best_app = ["B", "C", "D"][idx]

        rows.append({
            "strategy": strat, "A": a, "B": b, "C": c, "D": d,
            "mean_BCD": mean_bcd, "best_BCD": best_bcd, "best_app": best_app,
        })

        if not np.isnan(a): avg_a.append(a)
        if not np.isnan(b): avg_b.append(b)
        if not np.isnan(c): avg_c.append(c)
        if not np.isnan(d): avg_d.append(d)

    # Averages
    ma = np.mean(avg_a) if avg_a else np.nan
    mb = np.mean(avg_b) if avg_b else np.nan
    mc = np.mean(avg_c) if avg_c else np.nan
    md = np.mean(avg_d) if avg_d else np.nan
    bcd_avg = [v for v in [mb, mc, md] if not np.isnan(v)]
    mean_bcd_avg = np.mean(bcd_avg) if bcd_avg else np.nan
    best_bcd_avg = np.max(bcd_avg) if bcd_avg else np.nan
    best_app_avg = ["B", "C", "D"][np.argmax([mb, mc, md])] if bcd_avg else ""

    rows.append({"_sep": True})
    rows.append({
        "strategy": "Mean", "A": ma, "B": mb, "C": mc, "D": md,
        "mean_BCD": mean_bcd_avg, "best_BCD": best_bcd_avg, "best_app": best_app_avg,
    })

    # Delta vs A
    rows.append({"_sep": True})
    rows.append({
        "strategy": "Delta vs A", "A": np.nan,
        "B": mb - ma, "C": mc - ma, "D": md - ma,
        "mean_BCD": mean_bcd_avg - ma, "best_BCD": best_bcd_avg - ma, "best_app": "",
    })

    # Delta % vs A
    rows.append({
        "strategy": "Delta% vs A", "A": np.nan,
        "B": (mb - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "C": (mc - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "D": (md - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "mean_BCD": (mean_bcd_avg - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "best_BCD": (best_bcd_avg - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "best_app": "",
    })

    return rows


def build_table_maxdd(df, method_mode="EW"):
    """For MaxDD: best = least negative (max value), but display as negative."""
    rows = []
    avg_a, avg_b, avg_c, avg_d = [], [], [], []

    for strat in STRATEGIES:
        if method_mode == "EW":
            a = get_val(df, strat, "A", "EW", "maxdd_pct")
            b = get_val(df, strat, "B", "EW", "maxdd_pct")
            c = get_val(df, strat, "C", "EW", "maxdd_pct")
            d = get_val(df, strat, "D", "EW", "maxdd_pct")
        else:
            # Best = least negative = max
            a = df[(df["strategy"] == strat) & (df["approach"] == "A")]["maxdd_pct"].max()
            b = df[(df["strategy"] == strat) & (df["approach"] == "B")]["maxdd_pct"].max()
            c = df[(df["strategy"] == strat) & (df["approach"] == "C")]["maxdd_pct"].max()
            d = df[(df["strategy"] == strat) & (df["approach"] == "D")]["maxdd_pct"].max()

        bcd_vals = [v for v in [b, c, d] if not np.isnan(v)]
        mean_bcd = np.mean(bcd_vals) if bcd_vals else np.nan
        best_bcd = np.max(bcd_vals) if bcd_vals else np.nan  # least negative

        best_app = ""
        if bcd_vals:
            idx = np.argmax([b, c, d])  # least negative
            best_app = ["B", "C", "D"][idx]

        rows.append({
            "strategy": strat, "A": a, "B": b, "C": c, "D": d,
            "mean_BCD": mean_bcd, "best_BCD": best_bcd, "best_app": best_app,
        })

        if not np.isnan(a): avg_a.append(a)
        if not np.isnan(b): avg_b.append(b)
        if not np.isnan(c): avg_c.append(c)
        if not np.isnan(d): avg_d.append(d)

    ma = np.mean(avg_a) if avg_a else np.nan
    mb = np.mean(avg_b) if avg_b else np.nan
    mc = np.mean(avg_c) if avg_c else np.nan
    md = np.mean(avg_d) if avg_d else np.nan
    bcd_avg = [v for v in [mb, mc, md] if not np.isnan(v)]
    mean_bcd_avg = np.mean(bcd_avg) if bcd_avg else np.nan
    best_bcd_avg = np.max(bcd_avg) if bcd_avg else np.nan
    best_app_avg = ["B", "C", "D"][np.argmax([mb, mc, md])] if bcd_avg else ""

    rows.append({"_sep": True})
    rows.append({
        "strategy": "Mean", "A": ma, "B": mb, "C": mc, "D": md,
        "mean_BCD": mean_bcd_avg, "best_BCD": best_bcd_avg, "best_app": best_app_avg,
    })

    rows.append({"_sep": True})
    # Delta: improvement = less negative = positive delta
    rows.append({
        "strategy": "Delta vs A", "A": np.nan,
        "B": mb - ma, "C": mc - ma, "D": md - ma,
        "mean_BCD": mean_bcd_avg - ma, "best_BCD": best_bcd_avg - ma, "best_app": "",
    })
    rows.append({
        "strategy": "Delta% vs A", "A": np.nan,
        "B": (mb - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "C": (mc - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "D": (md - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "mean_BCD": (mean_bcd_avg - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "best_BCD": (best_bcd_avg - ma) / abs(ma) * 100 if abs(ma) > 1e-10 else np.nan,
        "best_app": "",
    })

    return rows


def run_tables(df, tf_label, comm_label):
    """Build and print all 4 comparison tables. Returns list of rows for CSV."""
    cols = ["A", "B", "C", "D", "mean_BCD", "best_BCD", "best_app"]

    t1 = build_table(df, "net_sharpe", method_mode="EW")
    print_comparison_table(
        f"TABLE 1 ({tf_label}): EW Portfolios — Net Sharpe @ {comm_label}  (A vs B vs C vs D)",
        t1, cols, fmt=".4f")

    t2 = build_table(df, "net_sharpe", method_mode="best")
    print_comparison_table(
        f"TABLE 2 ({tf_label}): Best Method — Net Sharpe @ {comm_label}  (A vs B vs C vs D)",
        t2, cols, fmt=".4f")

    t3 = build_table(df, "ann_ret_pct", method_mode="EW")
    print_comparison_table(
        f"TABLE 3 ({tf_label}): EW Portfolios — Ann Return %  (A vs B vs C vs D)",
        t3, cols, fmt=".2f")

    t4 = build_table_maxdd(df, method_mode="EW")
    print_comparison_table(
        f"TABLE 4 ({tf_label}): EW Portfolios — Max Drawdown %  (A vs B vs C vs D)",
        t4, cols, fmt=".2f")

    all_rows = []
    for table_name, table_data in [
        (f"{tf_label}_EW_NetSharpe", t1),
        (f"{tf_label}_BestMethod_NetSharpe", t2),
        (f"{tf_label}_EW_AnnRet", t3),
        (f"{tf_label}_EW_MaxDD", t4),
    ]:
        for row in table_data:
            if row.get("_sep"):
                continue
            r = {"table": table_name}
            r.update(row)
            all_rows.append(r)

    return all_rows


def main():
    # ── Daily ──
    df_daily_all = pd.read_csv(OUT_DIR / "v4_portfolios_daily.csv")
    print(f"Loaded {len(df_daily_all)} rows from v4_portfolios_daily.csv")
    # Filter to primary net scenario (if comm_level column exists)
    if "comm_level" in df_daily_all.columns:
        df_daily = df_daily_all[df_daily_all["comm_level"] == "net_0.05"].copy()
        print(f"  Filtered to net_0.05: {len(df_daily)} rows")
    else:
        df_daily = df_daily_all
    daily_rows = run_tables(df_daily, "daily", "0.05%")

    # ── Hourly ──
    hourly_path = OUT_DIR / "v4_portfolios_hourly.csv"
    hourly_rows = []
    if hourly_path.exists():
        df_hourly_all = pd.read_csv(hourly_path)
        print(f"\nLoaded {len(df_hourly_all)} rows from v4_portfolios_hourly.csv")
        if "comm_level" in df_hourly_all.columns:
            df_hourly = df_hourly_all[df_hourly_all["comm_level"] == "net_0.04"].copy()
            print(f"  Filtered to net_0.04: {len(df_hourly)} rows")
        else:
            df_hourly = df_hourly_all
        hourly_rows = run_tables(df_hourly, "hourly", "0.04%")
    else:
        print("\nNo v4_portfolios_hourly.csv found, skipping hourly.")

    # Save combined CSV
    all_rows = daily_rows + hourly_rows
    out_df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "v4_A_vs_forecast_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
