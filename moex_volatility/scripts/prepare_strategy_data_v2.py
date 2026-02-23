#!/usr/bin/env python3
"""
prepare_strategy_data_v2.py — Prepare full-history Daily + Hourly OHLCV + sigma_pred.

Input:
  - 10-min candles from moex_discovery/data/raw/candles_10m/ (from 2014)
  - Walk-forward predictions from results/final/data/predictions_walkforward/

Output:
  1. data/ohlcv_daily_full.parquet  (ticker, date, OHLCV, ~2014-2026)
  2. data/ohlcv_hourly_full.parquet (ticker, datetime, OHLCV, ~2014-2026)
  3. data/vpred_aligned.parquet     (date, ticker, sigma_h1, sigma_h5, sigma_h22, 2017-2026)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
CANDLE_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/raw/candles_10m")
WF_DIR = BASE / "results" / "final" / "data" / "predictions_walkforward"
OUT_DIR = BASE / "results" / "final" / "strategies" / "data"

TICKERS = sorted([
    "AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
    "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"
])

# Start from 2014 to have enough history for expanding walk-forward
LOOKBACK_START = pd.Timestamp("2014-06-01")

# MOEX session hours (Moscow time)
SESSION_START_HOUR = 10
SESSION_END_HOUR = 18  # last bar 18:00-18:50


def load_10min(tickers):
    """Load 10-min candles for given tickers, from LOOKBACK_START onwards."""
    print("Loading 10-min candles...")
    frames = []
    for ticker in tickers:
        path = CANDLE_DIR / f"{ticker}.parquet"
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] >= LOOKBACK_START].copy()
        df["ticker"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df):>8,} bars, "
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

    days_per_ticker = daily.groupby("ticker").size()
    print(f"  Total: {len(daily):,} daily bars")
    print(f"  Days/ticker: {days_per_ticker.min()} — {days_per_ticker.max()} "
          f"(mean {days_per_ticker.mean():.0f})")
    print(f"  Period: {daily['date'].min().date()} — {daily['date'].max().date()}")
    return daily


def aggregate_hourly(raw):
    """Aggregate 10-min candles to hourly OHLCV.

    MOEX session: 10:00-18:50 MSK
    Hourly bars: 10:00-10:59, 11:00-11:59, ..., 18:00-18:50
    => 9 bars per day (hours 10,11,12,13,14,15,16,17,18)
    """
    print("\nAggregating to hourly OHLCV...")
    raw["hour"] = raw["timestamp"].dt.hour
    session = raw[(raw["hour"] >= SESSION_START_HOUR) &
                  (raw["hour"] <= SESSION_END_HOUR)].copy()
    session["datetime"] = session["timestamp"].dt.floor("h")

    hourly = session.groupby(["ticker", "datetime"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    hourly = hourly.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    hourly["date"] = hourly["datetime"].dt.normalize()
    bars_per_day = hourly.groupby(["ticker", "date"]).size()
    print(f"  Total: {len(hourly):,} hourly bars")
    print(f"  Bars/day: mean={bars_per_day.mean():.1f}, "
          f"median={bars_per_day.median():.0f}, "
          f"min={bars_per_day.min()}, max={bars_per_day.max()}")
    print(f"  Period: {hourly['datetime'].min()} — {hourly['datetime'].max()}")
    hourly = hourly.drop(columns=["date"])
    return hourly


def load_vpred(tickers):
    """Load walk-forward predictions and compute sigma_pred = sqrt(pred_V1_Adaptive).

    pred_V1_Adaptive is in RV (variance) units.
    sigma = sqrt(RV) gives daily volatility (~0.017 mean).
    """
    print("\nLoading walk-forward predictions (V1_Adaptive)...")
    frames = []
    for h in [1, 5, 22]:
        path = WF_DIR / f"walkforward_all_h{h}.parquet"
        if not path.exists():
            print(f"  WARNING: {path.name} not found!")
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["ticker"].isin(tickers)].copy()

        # pred_V1_Adaptive is RV (variance), take sqrt for volatility
        rv = df["pred_V1_Adaptive"].values
        rv = np.clip(rv, 0, None)  # ensure non-negative
        sigma = np.sqrt(rv)

        sub = pd.DataFrame({
            "date": df["date"].values,
            "ticker": df["ticker"].values,
            f"sigma_h{h}": sigma,
        })
        frames.append(sub)
        print(f"  h={h}: {len(sub):,} rows, "
              f"sigma mean={sigma.mean():.5f}, std={sigma.std():.5f}, "
              f"{sub['date'].min().date()} — {sub['date'].max().date()}")

    # Merge all horizons
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["date", "ticker"], how="outer")

    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  Merged: {len(merged):,} rows, {merged['ticker'].nunique()} tickers")
    print(f"  Period: {merged['date'].min().date()} — {merged['date'].max().date()}")
    return merged


def main():
    print("=" * 60)
    print("Data Preparation V2: Full History + Sigma Pred")
    print(f"17 tickers, from {LOOKBACK_START.date()}")
    print("=" * 60)

    # 1. Load 10-min data
    raw = load_10min(TICKERS)

    # 2. Aggregate to daily
    daily = aggregate_daily(raw)

    # 3. Aggregate to hourly
    hourly = aggregate_hourly(raw)

    # 4. Load sigma predictions
    vpred = load_vpred(TICKERS)

    # 5. Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    path_d = OUT_DIR / "ohlcv_daily_full.parquet"
    daily.to_parquet(path_d, index=False)
    print(f"\nSaved {path_d.name}: {len(daily):,} rows")

    path_h = OUT_DIR / "ohlcv_hourly_full.parquet"
    hourly.to_parquet(path_h, index=False)
    print(f"Saved {path_h.name}: {len(hourly):,} rows")

    path_v = OUT_DIR / "vpred_aligned.parquet"
    vpred.to_parquet(path_v, index=False)
    print(f"Saved {path_v.name}: {len(vpred):,} rows")

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Daily:  {len(daily):>8,} rows, {daily['date'].min().date()} — {daily['date'].max().date()}")
    print(f"  Hourly: {len(hourly):>8,} rows, {hourly['datetime'].min()} — {hourly['datetime'].max()}")
    print(f"  Vpred:  {len(vpred):>8,} rows, {vpred['date'].min().date()} — {vpred['date'].max().date()}")
    for h in [1, 5, 22]:
        col = f"sigma_h{h}"
        if col in vpred.columns:
            s = vpred[col].dropna()
            print(f"  sigma_h{h}: mean={s.mean():.5f}, median={s.median():.5f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
