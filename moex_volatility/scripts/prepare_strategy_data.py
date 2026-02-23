#!/usr/bin/env python3
"""
prepare_strategy_data.py — Prepare Daily + Hourly OHLCV and align with WF predictions.

Input:  10-min candles from moex_discovery/data/raw/candles_10m/
Output:
  1. results/final/strategies/data/ohlcv_daily.parquet
  2. results/final/strategies/data/ohlcv_hourly.parquet
  3. results/final/strategies/data/predictions_aligned.parquet
  4. results/final/strategies/data_inventory.md
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
CANDLE_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/raw/candles_10m")
PRED_DIR = BASE / "data" / "predictions" / "walk_forward"
OUT_DIR = BASE / "results" / "final" / "strategies"
OUT_DATA = OUT_DIR / "data"

# 17 model tickers (all have 10-min candles + WF predictions)
TICKERS = sorted([
    "AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
    "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"
])

# Lookback start (252 trading days before 2020-01-01 for indicator warmup)
LOOKBACK_START = pd.Timestamp("2019-01-01")
STRATEGY_START = pd.Timestamp("2020-01-01")

# MOEX session hours (Moscow time)
SESSION_START_HOUR = 10
SESSION_END_HOUR = 18  # last bar 18:00-18:50


def load_10min(tickers):
    """Load 10-min candles for given tickers, filter to LOOKBACK_START onwards."""
    print("Loading 10-min candles...")
    frames = []
    for ticker in tickers:
        path = CANDLE_DIR / f"{ticker}.parquet"
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] >= LOOKBACK_START].copy()
        df["ticker"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df):,} bars, "
              f"{df['timestamp'].min().date()} — {df['timestamp'].max().date()}")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    print(f"  Total: {len(raw):,} bars, {raw['ticker'].nunique()} tickers")
    return raw


def aggregate_daily(raw):
    """Aggregate 10-min candles to daily OHLCV."""
    print("\nAggregating to daily OHLCV...")
    raw["date"] = raw["timestamp"].dt.normalize()
    daily = raw.groupby(["ticker", "date"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Stats
    for ticker in daily["ticker"].unique():
        tdf = daily[daily["ticker"] == ticker]
        print(f"  {ticker}: {len(tdf):,} days, "
              f"{tdf['date'].min().date()} — {tdf['date'].max().date()}")

    print(f"  Total: {len(daily):,} daily bars")
    return daily


def aggregate_hourly(raw):
    """Aggregate 10-min candles to hourly OHLCV.

    MOEX session: 10:00-18:50 MSK
    Hourly bars: 10:00-10:59, 11:00-11:59, ..., 18:00-18:50
    => 9 bars per day (hours 10,11,12,13,14,15,16,17,18)
    """
    print("\nAggregating to hourly OHLCV...")

    raw["hour"] = raw["timestamp"].dt.hour
    # Filter to session hours only (10-18)
    session = raw[(raw["hour"] >= SESSION_START_HOUR) &
                  (raw["hour"] <= SESSION_END_HOUR)].copy()

    # Create hourly datetime: floor to hour
    session["datetime"] = session["timestamp"].dt.floor("h")

    hourly = session.groupby(["ticker", "datetime"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    hourly = hourly.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    # Bars per day stats
    hourly["date"] = hourly["datetime"].dt.normalize()
    bars_per_day = hourly.groupby(["ticker", "date"]).size()

    print(f"  Total: {len(hourly):,} hourly bars")
    print(f"  Bars/day: mean={bars_per_day.mean():.1f}, "
          f"min={bars_per_day.min()}, max={bars_per_day.max()}, "
          f"median={bars_per_day.median():.0f}")

    # Distribution of bars per day
    bpd_counts = bars_per_day.value_counts().sort_index()
    print(f"  Bars/day distribution:")
    for nbars, count in bpd_counts.items():
        pct = count / len(bars_per_day) * 100
        print(f"    {nbars} bars: {count:,} days ({pct:.1f}%)")

    hourly = hourly.drop(columns=["date"])
    return hourly, bars_per_day


def load_predictions(tickers):
    """Load WF predictions for h=1, h=5, h=22 and filter to available tickers."""
    print("\nLoading walk-forward predictions...")
    preds = {}
    for h in [1, 5, 22]:
        path = PRED_DIR / f"hybrid_adaptive_h{h}_annual.parquet"
        if not path.exists():
            print(f"  WARNING: {path.name} not found!")
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        # Filter to available tickers and period
        df = df[df["ticker"].isin(tickers)].copy()
        df = df[df["date"] >= LOOKBACK_START].copy()
        df = df.rename(columns={"rv_pred": f"rv_pred_h{h}", "rv_actual": f"rv_actual_h{h}"})
        df = df[["date", "ticker", f"rv_pred_h{h}", f"rv_actual_h{h}"]]
        preds[h] = df
        print(f"  h={h}: {len(df):,} rows, {df['ticker'].nunique()} tickers, "
              f"{df['date'].min().date()} — {df['date'].max().date()}")

    # Merge all horizons
    if len(preds) == 0:
        return pd.DataFrame()

    merged = preds[1]
    for h in [5, 22]:
        if h in preds:
            merged = merged.merge(preds[h], on=["date", "ticker"], how="outer")

    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  Merged: {len(merged):,} rows")
    return merged


def check_alignment(daily, predictions, tickers):
    """Check date alignment between daily OHLCV and predictions."""
    print("\nChecking date alignment (OHLCV vs predictions, 2020+)...")
    alignment = []

    for ticker in tickers:
        d_dates = set(daily[(daily["ticker"] == ticker) &
                            (daily["date"] >= STRATEGY_START)]["date"])
        p_dates = set(predictions[(predictions["ticker"] == ticker) &
                                   (predictions["date"] >= STRATEGY_START)]["date"])

        inner = d_dates & p_dates
        ohlcv_only = d_dates - p_dates
        pred_only = p_dates - d_dates

        alignment.append({
            "ticker": ticker,
            "ohlcv_days": len(d_dates),
            "pred_days": len(p_dates),
            "inner": len(inner),
            "ohlcv_only": len(ohlcv_only),
            "pred_only": len(pred_only),
        })

    adf = pd.DataFrame(alignment)
    print(adf.to_string(index=False))
    return adf


def check_gaps(daily, tickers):
    """Find gaps > 5 business days in daily data."""
    print("\nChecking for gaps > 5 business days...")
    gaps = []
    for ticker in tickers:
        tdf = daily[(daily["ticker"] == ticker) &
                     (daily["date"] >= STRATEGY_START)].sort_values("date")
        dates = tdf["date"].values
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
            if delta > 7:  # > 5 business days ~ > 7 calendar days
                gaps.append({
                    "ticker": ticker,
                    "from": pd.Timestamp(dates[i - 1]).date(),
                    "to": pd.Timestamp(dates[i]).date(),
                    "calendar_days": int(delta),
                })
    if gaps:
        gdf = pd.DataFrame(gaps)
        print(gdf.to_string(index=False))
    else:
        print("  No gaps > 5 business days found.")
    return gaps


def save_outputs(daily, hourly, predictions, alignment_df, gaps, tickers,
                 bars_per_day):
    """Save all output files."""
    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    OUT_DATA.mkdir(parents=True, exist_ok=True)

    # 1. Daily OHLCV
    path = OUT_DATA / "ohlcv_daily.parquet"
    daily.to_parquet(path, index=False)
    print(f"  {path.name}: {len(daily):,} rows")

    # 2. Hourly OHLCV
    path = OUT_DATA / "ohlcv_hourly.parquet"
    hourly.to_parquet(path, index=False)
    print(f"  {path.name}: {len(hourly):,} rows")

    # 3. Predictions aligned
    path = OUT_DATA / "predictions_aligned.parquet"
    predictions.to_parquet(path, index=False)
    print(f"  {path.name}: {len(predictions):,} rows")

    # 4. Data inventory report
    write_inventory(daily, hourly, predictions, alignment_df, gaps,
                    tickers, bars_per_day)


def write_inventory(daily, hourly, predictions, alignment_df, gaps,
                    tickers, bars_per_day):
    """Write data_inventory.md report."""
    path = OUT_DIR / "data_inventory.md"

    lines = []
    lines.append("# Data Inventory for Trading Strategies Block")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Ticker status
    lines.append("---")
    lines.append("")
    lines.append("## 1. Tickers")
    lines.append("")
    lines.append(f"**17 model tickers** (all have 10-min candles, features, and WF predictions):")
    lines.append(f"  {', '.join(tickers)}")
    lines.append("")

    # Daily summary
    lines.append("---")
    lines.append("")
    lines.append("## 2. Daily OHLCV")
    lines.append("")
    lines.append(f"**File:** `data/ohlcv_daily.parquet`")
    lines.append(f"**Rows:** {len(daily):,}")
    lines.append(f"**Columns:** date, ticker, open, high, low, close, volume")
    lines.append(f"**Period:** {daily['date'].min().date()} — {daily['date'].max().date()}")
    lines.append(f"**Tickers:** {daily['ticker'].nunique()}")
    lines.append("")
    lines.append("| Ticker | Days | Date Range | Close Range |")
    lines.append("|--------|------|------------|-------------|")
    for ticker in tickers:
        tdf = daily[daily["ticker"] == ticker]
        lines.append(f"| {ticker} | {len(tdf):,} | "
                      f"{tdf['date'].min().date()} — {tdf['date'].max().date()} | "
                      f"{tdf['close'].min():.2f} — {tdf['close'].max():.2f} |")
    lines.append("")

    # Hourly summary
    lines.append("---")
    lines.append("")
    lines.append("## 3. Hourly OHLCV")
    lines.append("")
    lines.append(f"**File:** `data/ohlcv_hourly.parquet`")
    lines.append(f"**Rows:** {len(hourly):,}")
    lines.append(f"**Columns:** datetime, ticker, open, high, low, close, volume")
    lines.append(f"**Period:** {hourly['datetime'].min()} — {hourly['datetime'].max()}")
    lines.append(f"**Tickers:** {hourly['ticker'].nunique()}")
    lines.append(f"**Session:** {SESSION_START_HOUR}:00 — {SESSION_END_HOUR}:50 MSK")
    lines.append(f"**Expected bars/day:** 9 (hours 10-18)")
    lines.append("")

    bpd_mean = bars_per_day.mean()
    bpd_median = bars_per_day.median()
    bpd_min = bars_per_day.min()
    bpd_max = bars_per_day.max()
    lines.append(f"**Bars/day stats:** mean={bpd_mean:.1f}, median={bpd_median:.0f}, "
                  f"min={bpd_min}, max={bpd_max}")
    lines.append("")

    bpd_counts = bars_per_day.value_counts().sort_index()
    lines.append("| Bars/Day | Count | % |")
    lines.append("|----------|-------|---|")
    for nbars, count in bpd_counts.items():
        pct = count / len(bars_per_day) * 100
        lines.append(f"| {nbars} | {count:,} | {pct:.1f}% |")
    lines.append("")

    hourly_per_ticker = hourly.groupby("ticker").size()
    lines.append("| Ticker | Hourly Bars | Days (approx) |")
    lines.append("|--------|-------------|---------------|")
    for ticker in tickers:
        if ticker in hourly_per_ticker.index:
            nb = hourly_per_ticker[ticker]
            lines.append(f"| {ticker} | {nb:,} | ~{nb // 9:,} |")
    lines.append("")

    # Predictions
    lines.append("---")
    lines.append("")
    lines.append("## 4. Walk-Forward Predictions (Hybrid V1_Adaptive)")
    lines.append("")
    lines.append(f"**File:** `data/predictions_aligned.parquet`")
    lines.append(f"**Rows:** {len(predictions):,}")
    lines.append(f"**Columns:** date, ticker, rv_pred_h1, rv_actual_h1, "
                  "rv_pred_h5, rv_actual_h5, rv_pred_h22, rv_actual_h22")
    lines.append(f"**Period:** {predictions['date'].min().date()} — "
                  f"{predictions['date'].max().date()}")
    lines.append(f"**Tickers:** {predictions['ticker'].nunique()}")
    lines.append("")
    lines.append("**Hourly scaling rule:** sigma_hour = sigma_day / sqrt(9)")
    lines.append("")

    for h in [1, 5, 22]:
        col = f"rv_pred_h{h}"
        if col in predictions.columns:
            vals = predictions[col].dropna()
            lines.append(f"**h={h}:** {len(vals):,} predictions, "
                          f"mean={vals.mean():.6f}, std={vals.std():.6f}, "
                          f"min={vals.min():.6f}, max={vals.max():.6f}")
    lines.append("")

    # Alignment
    lines.append("---")
    lines.append("")
    lines.append("## 5. Date Alignment (OHLCV vs Predictions, 2020+)")
    lines.append("")
    lines.append("| Ticker | OHLCV days | Pred days | Inner | OHLCV only | Pred only |")
    lines.append("|--------|-----------|-----------|-------|------------|-----------|")
    for _, row in alignment_df.iterrows():
        lines.append(f"| {row['ticker']} | {row['ohlcv_days']} | {row['pred_days']} | "
                      f"{row['inner']} | {row['ohlcv_only']} | {row['pred_only']} |")
    lines.append("")

    # Gaps
    lines.append("---")
    lines.append("")
    lines.append("## 6. Gaps > 5 Business Days (2020+)")
    lines.append("")
    if gaps:
        lines.append("| Ticker | From | To | Calendar Days | Reason |")
        lines.append("|--------|------|----|---------------|--------|")
        for g in gaps:
            reason = "MOEX trading halt" if g["from"] >= pd.Timestamp("2022-02-20").date() and g["to"] <= pd.Timestamp("2022-04-01").date() else "Unknown"
            lines.append(f"| {g['ticker']} | {g['from']} | {g['to']} | "
                          f"{g['calendar_days']} | {reason} |")
    else:
        lines.append("No gaps > 5 business days found.")
    lines.append("")

    # Quality checks
    lines.append("---")
    lines.append("")
    lines.append("## 7. Quality Checks")
    lines.append("")

    # Check for negative prices
    neg_prices = (daily[["open", "high", "low", "close"]] <= 0).any().any()
    lines.append(f"- Negative/zero prices in daily: {'FAIL' if neg_prices else 'PASS'}")

    # Check for NaN
    nan_daily = daily[["open", "high", "low", "close", "volume"]].isna().any().any()
    lines.append(f"- NaN in daily OHLCV: {'FAIL' if nan_daily else 'PASS'}")

    nan_hourly = hourly[["open", "high", "low", "close", "volume"]].isna().any().any()
    lines.append(f"- NaN in hourly OHLCV: {'FAIL' if nan_hourly else 'PASS'}")

    # Zero volume
    zero_vol_d = (daily["volume"] == 0).sum()
    lines.append(f"- Zero volume days (daily): {zero_vol_d}")

    zero_vol_h = (hourly["volume"] == 0).sum()
    lines.append(f"- Zero volume bars (hourly): {zero_vol_h}")

    # Prediction NaN
    for h in [1, 5, 22]:
        col = f"rv_pred_h{h}"
        if col in predictions.columns:
            nans = predictions[col].isna().sum()
            lines.append(f"- NaN in rv_pred_h{h}: {nans}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  {path.name}: written")


def print_final_summary(daily, hourly, predictions, tickers, bars_per_day):
    """Print final summary to console."""
    print("\n" + "=" * 70)
    print("DATA PREPARATION SUMMARY")
    print("=" * 70)

    print(f"\nTickers: {len(tickers)}")
    print(f"  {', '.join(tickers)}")

    print(f"\n--- DAILY OHLCV ---")
    print(f"  Rows: {len(daily):,}")
    print(f"  Period: {daily['date'].min().date()} — {daily['date'].max().date()}")
    days_per_ticker = daily.groupby("ticker").size()
    print(f"  Days/ticker: {days_per_ticker.min()} — {days_per_ticker.max()} "
          f"(mean {days_per_ticker.mean():.0f})")

    print(f"\n--- HOURLY OHLCV ---")
    print(f"  Rows: {len(hourly):,}")
    print(f"  Period: {hourly['datetime'].min()} — {hourly['datetime'].max()}")
    bars_per_ticker = hourly.groupby("ticker").size()
    print(f"  Bars/ticker: {bars_per_ticker.min():,} — {bars_per_ticker.max():,} "
          f"(mean {bars_per_ticker.mean():,.0f})")
    print(f"  Bars/day: median={bars_per_day.median():.0f}, "
          f"mean={bars_per_day.mean():.1f}")

    print(f"\n--- PREDICTIONS ---")
    print(f"  Rows: {len(predictions):,}")
    print(f"  Period: {predictions['date'].min().date()} — "
          f"{predictions['date'].max().date()}")
    for h in [1, 5, 22]:
        col = f"rv_pred_h{h}"
        if col in predictions.columns:
            n = predictions[col].notna().sum()
            print(f"  h={h}: {n:,} predictions")

    print(f"\n--- HOURLY VOL SCALING ---")
    print(f"  sigma_hour = sigma_day / sqrt(9) = sigma_day / 3.0")
    for h in [1, 5, 22]:
        col = f"rv_pred_h{h}"
        if col in predictions.columns:
            mean_daily = predictions[col].mean()
            mean_hourly = mean_daily / np.sqrt(9)
            print(f"  h={h}: daily mean rv_pred={mean_daily:.6f}, "
                  f"hourly scaled={mean_hourly:.6f}")

    print("\n" + "=" * 70)


def main():
    print("=" * 60)
    print("Data Preparation: Daily + Hourly OHLCV + Predictions")
    print(f"17 model tickers")
    print("=" * 60)

    tickers = TICKERS

    # 1. Load 10-min data
    raw = load_10min(tickers)

    # 2. Aggregate to daily
    daily = aggregate_daily(raw)

    # 3. Aggregate to hourly
    hourly, bars_per_day = aggregate_hourly(raw)

    # 4. Load predictions
    predictions = load_predictions(tickers)

    # 5. Check alignment
    alignment_df = check_alignment(daily, predictions, tickers)

    # 6. Check gaps
    gaps = check_gaps(daily, tickers)

    # 7. Save
    save_outputs(daily, hourly, predictions, alignment_df, gaps,
                 tickers, bars_per_day)

    # 8. Print summary
    print_final_summary(daily, hourly, predictions, tickers, bars_per_day)

    print("\nDONE")


if __name__ == "__main__":
    main()
