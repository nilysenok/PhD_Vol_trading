#!/usr/bin/env python3
"""Intraday GARCH on 10-minute returns → daily RV forecast.

Approach:
  1. Load 10-min candles for 17 MOEX stocks
  2. Filter main session (10:00–18:40), remove overnight returns
  3. Fit GJR-GARCH(1,1,1) with Student-t on train period (≤2018-12-31)
  4. Fix parameters and filter through test 2019
  5. Aggregate intraday conditional variances to daily RV forecasts
  6. Evaluate with QLIKE for H=1, H=5, H=22

Usage:
    python scripts/garch_10min.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import time as dtime
from arch import arch_model
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

from src.evaluation.metrics import qlike


def qlike_patton(y_true, y_pred, epsilon=1e-10):
    """Patton (2011) QLIKE: mean(actual/pred - log(actual/pred) - 1).

    Always non-negative; equals 0 when pred == actual.
    This is the variant used for the dissertation figures.
    """
    y_pred_safe = np.maximum(y_pred, epsilon)
    ratio = y_true / y_pred_safe
    return np.mean(ratio - np.log(ratio) - 1)

# ── Configuration ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CANDLES_DIR = BASE_DIR.parent / "moex_discovery" / "data" / "final_v6" / "candles_10m" / "stocks"
FEATURES_PATH = BASE_DIR / "data" / "features" / "features_all.parquet"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions" / "test_2019"
MODELS_DIR = BASE_DIR / "models" / "garch_10min"

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2019-12-31")

HORIZONS = [1, 5, 22]
SCALE = 10000  # basis points for numerical stability

TICKERS = [
    "SBER", "LKOH", "TATN", "NVTK", "VTBR", "ALRS", "AFLT",
    "HYDR", "MGNT", "MOEX", "RTKM", "MTSS", "MTLR", "IRAO",
    "OGKB", "PHOR", "LSRG",
]


# ── Data loading ─────────────────────────────────────────────────────
def load_and_prepare(ticker: str) -> pd.DataFrame:
    """Load 10-min candles, filter main session, compute scaled returns."""
    path = CANDLES_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(path)

    # Filter main session: 10:00 – 18:40
    t = df["timestamp"].dt.time
    df = df[(t >= dtime(10, 0)) & (t <= dtime(18, 40))].copy()

    df["date"] = df["timestamp"].dt.date

    # Log returns (global shift — first bar of each day is overnight)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Remove first bar of each day (contains overnight/pre-open gap)
    df["is_first"] = df.groupby("date").cumcount() == 0
    df = df[~df["is_first"]].copy()
    df = df.dropna(subset=["log_return"])

    # Scale to basis points
    df["r_scaled"] = df["log_return"] * SCALE

    return df


# ── Per-ticker processing ────────────────────────────────────────────
def process_ticker(ticker: str):
    """Fit GJR-GARCH on train 10-min returns, filter through test."""
    try:
        df = load_and_prepare(ticker)
    except Exception as e:
        print(f"  {ticker}: load failed — {e}")
        return None

    dates = pd.Series(df["date"].values)
    train_mask = dates <= TRAIN_END.date()
    test_mask = (dates >= TEST_START.date()) & (dates <= TEST_END.date())

    train_returns = df.loc[train_mask.values, "r_scaled"].values
    if len(train_returns) < 1000:
        print(f"  {ticker}: insufficient train data ({len(train_returns)} bars), skip")
        return None

    # ── Fit GJR-GARCH(1,1,1) t-dist on train ──
    try:
        m_train = arch_model(
            train_returns, mean="Zero", vol="GARCH",
            p=1, o=1, q=1, dist="t",
        )
        res_train = m_train.fit(disp="off")
        params = res_train.params
        aic = res_train.aic
    except Exception as e:
        print(f"  {ticker}: fit failed — {e}")
        return None

    # ── Filter with fixed params through ALL data ──
    all_returns = df["r_scaled"].values
    try:
        m_all = arch_model(
            all_returns, mean="Zero", vol="GARCH",
            p=1, o=1, q=1, dist="t",
        )
        filtered = m_all.fix(params)
        cond_var_scaled = filtered.conditional_volatility ** 2   # σ² in bp²
    except Exception as e:
        print(f"  {ticker}: filter failed — {e}")
        return None

    # Attach to DataFrame and convert back to raw variance units
    df = df.reset_index(drop=True)
    df["cond_var"] = cond_var_scaled / (SCALE ** 2)   # raw units

    # ── Aggregate to daily for test period ──
    test_df = df[test_mask.values]
    daily = (
        test_df
        .groupby("date")
        .agg(rv_pred=("cond_var", "sum"))   # Σ σ²_i per day
        .reset_index()
    )
    daily["ticker"] = ticker
    daily["date"] = pd.to_datetime(daily["date"])

    n_days = len(daily)
    print(f"  {ticker}: OK — {len(train_returns)} train bars, "
          f"{n_days} test days, AIC={aic:.1f}")

    return {
        "ticker": ticker,
        "daily": daily,
        "params": dict(params),
        "aic": aic,
    }


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GARCH on 10-minute data  (Intraday GJR-GARCH)")
    print("=" * 60)

    # Process tickers in parallel
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_ticker)(t) for t in TICKERS
    )
    results = [r for r in results if r is not None]
    print(f"\nProcessed: {len(results)}/{len(TICKERS)} tickers")

    if not results:
        print("ERROR: no tickers processed successfully")
        sys.exit(1)

    # ── Save parameters ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    params_rows = []
    for r in results:
        row = {"ticker": r["ticker"], "aic": r["aic"]}
        row.update(r["params"])
        params_rows.append(row)
    params_df = pd.DataFrame(params_rows)
    params_df.to_csv(MODELS_DIR / "params.csv", index=False)
    print(f"\nSaved params → {MODELS_DIR / 'params.csv'}")

    # ── Combine daily forecasts ──
    all_daily = pd.concat([r["daily"] for r in results], ignore_index=True)

    # ── Load reference targets from HAR prediction files ──
    # The dissertation QLIKE values (0.3052, 1.8017, etc.) are the Patton
    # variant computed on har_h{h}.parquet targets.  We use the same targets
    # so that our GARCH-10min QLIKE is directly comparable.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print("QLIKE results — Patton variant  (test 2019)")
    print(f"{'─'*60}")
    print(f"{'Horizon':<10} {'N':>6}  {'QLIKE':>8}")
    print(f"{'─'*60}")

    qlike_results = {}

    for h in HORIZONS:
        ref_path = PREDICTIONS_DIR / f"har_h{h}.parquet"
        if not ref_path.exists():
            print(f"  H={h}: reference file {ref_path.name} not found, skip")
            continue

        ref = pd.read_parquet(ref_path, columns=["date", "ticker", "rv_actual"])
        ref["date"] = pd.to_datetime(ref["date"])

        # Merge: keep only (date, ticker) pairs present in reference
        merged = ref.merge(all_daily, on=["date", "ticker"], how="inner")

        if len(merged) == 0:
            print(f"  H={h}: no matching rows after merge")
            continue

        # Compute Patton QLIKE (non-negative, matches dissertation figures)
        q = qlike_patton(merged["rv_actual"].values, merged["rv_pred"].values)
        qlike_results[h] = q

        # Save predictions
        out = merged[["date", "ticker", "rv_actual", "rv_pred"]].copy()
        out_path = PREDICTIONS_DIR / f"garch_10min_h{h}.parquet"
        out.to_parquet(out_path, index=False)

        print(f"  H={h:<5} {len(merged):>6}  {q:>8.4f}")
        print(f"         → saved {out_path.name}")

    # ── Summary comparison ──
    print(f"\n{'='*60}")
    print("Comparison table  (Patton QLIKE, lower is better)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'H=1':>8} {'H=5':>8} {'H=22':>8}")
    print(f"{'─'*60}")
    print(f"{'HAR-J':<20} {'0.3052':>8} {'0.4243':>8} {'0.4669':>8}")
    print(f"{'GARCH-GJR (daily)':<20} {'1.8017':>8} {'1.0514':>8} {'0.7320':>8}")

    g10 = "GARCH-10min"
    vals = [f"{qlike_results.get(h, float('nan')):.4f}" for h in HORIZONS]
    print(f"{g10:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")
    print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
