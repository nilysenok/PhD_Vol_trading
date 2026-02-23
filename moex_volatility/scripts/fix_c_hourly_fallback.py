#!/usr/bin/env python3
"""
Post-hoc fix: fill missing approach C hourly data with approach A positions.

When approach_c_v4 grid search fails for a ticker-year, no rows are generated.
The correct behavior is fallback to A (keep baseline positions).
This script patches the parquet by copying A's rows for missing C entries.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "final" / "strategies" / "walkforward_v4"
PARQUET = OUT_DIR / "daily_positions.parquet"
BCD_YEARS = [2022, 2023, 2024, 2025]

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]


def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET)
    print(f"  Total rows: {len(df):,}")

    hourly = df[df["tf"] == "hourly"].copy()
    print(f"  Hourly rows: {len(hourly):,}")

    # For each strategy, find missing (ticker, year) combos in C vs A
    new_rows = []
    for strat in STRATEGIES:
        a_data = hourly[(hourly["strategy"] == strat) &
                        (hourly["approach"] == "A") &
                        (hourly["test_year"].isin(BCD_YEARS))]
        c_data = hourly[(hourly["strategy"] == strat) &
                        (hourly["approach"] == "C") &
                        (hourly["test_year"].isin(BCD_YEARS))]

        # Get all (ticker, year) combos
        a_combos = set(zip(a_data["ticker"], a_data["test_year"]))
        c_combos = set(zip(c_data["ticker"], c_data["test_year"]))
        missing = a_combos - c_combos

        if not missing:
            print(f"  {strat}: no missing combos in C")
            continue

        print(f"  {strat}: {len(missing)} missing (ticker, year) combos in C")

        for ticker, year in sorted(missing):
            # Get A's rows for this combo
            mask = ((a_data["ticker"] == ticker) &
                    (a_data["test_year"] == year))
            a_rows = a_data[mask].copy()
            a_rows["approach"] = "C"
            new_rows.append(a_rows)
            n_rows = len(a_rows)
            n_active = (a_rows["position"] != 0).sum()
            print(f"    {ticker}/{year}: {n_rows:,} rows copied "
                  f"({n_active:,} active positions)")

        # Also check for partial missing dates within existing tickers
        for ticker, year in sorted(c_combos):
            a_mask = ((a_data["ticker"] == ticker) &
                      (a_data["test_year"] == year))
            c_mask = ((c_data["ticker"] == ticker) &
                      (c_data["test_year"] == year))
            a_dates = set(a_data[a_mask]["date"].values)
            c_dates = set(c_data[c_mask]["date"].values)
            missing_dates = a_dates - c_dates

            if missing_dates:
                # Copy A's rows for missing dates
                a_sub = a_data[a_mask]
                fill = a_sub[a_sub["date"].isin(missing_dates)].copy()
                fill["approach"] = "C"
                new_rows.append(fill)
                n_active = (fill["position"] != 0).sum()
                print(f"    {ticker}/{year}: {len(fill):,} partial dates filled "
                      f"({n_active:,} active)")

    if not new_rows:
        print("\nNo missing data found. Nothing to fix.")
        return

    new_df = pd.concat(new_rows, ignore_index=True)
    print(f"\n  Total new rows to add: {len(new_df):,}")

    # Merge
    merged = pd.concat([df, new_df], ignore_index=True)
    print(f"  After merge: {len(merged):,} rows (was {len(df):,})")

    # Verify: check C row counts now
    h_new = merged[merged["tf"] == "hourly"]
    for strat in STRATEGIES:
        a_n = len(h_new[(h_new["strategy"] == strat) & (h_new["approach"] == "A") &
                        (h_new["test_year"].isin(BCD_YEARS))])
        c_n = len(h_new[(h_new["strategy"] == strat) & (h_new["approach"] == "C") &
                        (h_new["test_year"].isin(BCD_YEARS))])
        ratio = c_n / a_n * 100 if a_n > 0 else 0
        flag = " *** FIXED" if strat == "S2_Bollinger" else ""
        print(f"  {strat}: A={a_n:,} C={c_n:,} ({ratio:.1f}%){flag}")

    # Save
    merged.to_parquet(PARQUET, index=False)
    print(f"\nSaved: {PARQUET}")


if __name__ == "__main__":
    main()
