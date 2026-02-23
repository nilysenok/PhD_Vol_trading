#!/usr/bin/env python3
"""Compare all trained models and generate reports.

Usage:
    python scripts/06_compare_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics import evaluate_forecast, compare_models
from src.evaluation.statistical_tests import (
    diebold_mariano_test,
    model_confidence_set,
    mincer_zarnowitz_test
)
from src.utils.logger import setup_logger


# Configuration
PREDICTIONS_DIR = "data/predictions/test_2019"
RESULTS_DIR = "results"
HORIZONS = [1, 5, 22]


def load_all_predictions(pred_dir: Path, horizon: int) -> pd.DataFrame:
    """Load and merge all predictions for a horizon."""
    dfs = []

    # Classical models
    classical_path = pred_dir / f"classical_h{horizon}.parquet"
    if classical_path.exists():
        df = pd.read_parquet(classical_path)
        dfs.append(df)

    # Boosting models
    boosting_path = pred_dir / f"boosting_h{horizon}.parquet"
    if boosting_path.exists():
        df = pd.read_parquet(boosting_path)
        # Remove duplicate columns
        df = df.drop(columns=["y_true"], errors="ignore")
        dfs.append(df)

    # Neural models
    neural_path = pred_dir / f"neural_h{horizon}.parquet"
    if neural_path.exists():
        df = pd.read_parquet(neural_path)
        df = df.drop(columns=["y_true"], errors="ignore")
        dfs.append(df)

    # Hybrid models
    hybrid_path = pred_dir / f"hybrid_h{horizon}.parquet"
    if hybrid_path.exists():
        df = pd.read_parquet(hybrid_path)
        df = df.drop(columns=["y_true"], errors="ignore")
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Merge on date and ticker
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=["date", "ticker"], how="outer")

    return result


def calculate_metrics_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for each ticker and model."""
    # Find prediction columns
    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
    model_names = [c.replace("y_pred_", "") for c in pred_cols]

    results = []

    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].dropna()

        if len(ticker_df) < 10:
            continue

        y_true = ticker_df["y_true"].values

        for pred_col, model_name in zip(pred_cols, model_names):
            if pred_col not in ticker_df.columns:
                continue

            y_pred = ticker_df[pred_col].values
            mask = ~np.isnan(y_pred)

            if mask.sum() < 10:
                continue

            metrics = evaluate_forecast(y_true[mask], y_pred[mask])

            results.append({
                "ticker": ticker,
                "model": model_name,
                **metrics.to_dict()
            })

    return pd.DataFrame(results)


def calculate_aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregate metrics across all tickers."""
    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
    model_names = [c.replace("y_pred_", "") for c in pred_cols]

    results = []

    for pred_col, model_name in zip(pred_cols, model_names):
        if pred_col not in df.columns:
            continue

        # Get valid rows
        valid = df[["y_true", pred_col]].dropna()

        if len(valid) < 10:
            continue

        y_true = valid["y_true"].values
        y_pred = valid[pred_col].values

        metrics = evaluate_forecast(y_true, y_pred)

        results.append({
            "model": model_name,
            **metrics.to_dict()
        })

    result_df = pd.DataFrame(results)

    # Add rank
    if len(result_df) > 0:
        result_df["rank_QLIKE"] = result_df["QLIKE"].rank()
        result_df["rank_RMSE"] = result_df["RMSE"].rank()

    return result_df.sort_values("QLIKE")


def run_dm_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Diebold-Mariano tests between all model pairs."""
    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
    model_names = [c.replace("y_pred_", "") for c in pred_cols]

    results = []
    valid = df[["y_true"] + pred_cols].dropna()

    if len(valid) < 50:
        return pd.DataFrame()

    y_true = valid["y_true"].values

    for i, (col1, name1) in enumerate(zip(pred_cols, model_names)):
        for col2, name2 in zip(pred_cols[i+1:], model_names[i+1:]):
            pred1 = valid[col1].values
            pred2 = valid[col2].values

            try:
                test_result = diebold_mariano_test(y_true, pred1, pred2, loss_func="mse")
                results.append({
                    "model1": name1,
                    "model2": name2,
                    "dm_statistic": test_result.statistic,
                    "p_value": test_result.p_value,
                    "conclusion": test_result.conclusion,
                })
            except Exception as e:
                print(f"DM test failed for {name1} vs {name2}: {e}")

    return pd.DataFrame(results)


def run_mcs(df: pd.DataFrame) -> dict:
    """Run Model Confidence Set procedure."""
    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
    model_names = [c.replace("y_pred_", "") for c in pred_cols]

    valid = df[["y_true"] + pred_cols].dropna()

    if len(valid) < 50:
        return {}

    y_true = valid["y_true"].values

    # Calculate losses for each model
    losses = {}
    for col, name in zip(pred_cols, model_names):
        y_pred = valid[col].values
        losses[name] = (y_true - y_pred) ** 2

    losses_df = pd.DataFrame(losses)

    try:
        mcs_result = model_confidence_set(y_true, {n: valid[c].values for c, n in zip(pred_cols, model_names)})
        return mcs_result
    except Exception as e:
        print(f"MCS failed: {e}")
        return {}


def plot_metrics_comparison(metrics_df: pd.DataFrame, horizon: int, save_dir: Path):
    """Plot metrics comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = ["QLIKE", "RMSE", "R2"]

    for ax, metric in zip(axes, metrics_to_plot):
        data = metrics_df.sort_values(metric, ascending=(metric != "R2"))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))

        if metric == "R2":
            colors = colors[::-1]

        ax.barh(data["model"], data[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} (H={horizon})")

    plt.tight_layout()
    plt.savefig(save_dir / f"metrics_comparison_h{horizon}.png", dpi=150)
    plt.close()


def plot_predictions_vs_actual(df: pd.DataFrame, horizon: int, ticker: str, save_dir: Path):
    """Plot predictions vs actual for a ticker."""
    ticker_df = df[df["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("date")

    pred_cols = [c for c in ticker_df.columns if c.startswith("y_pred_")]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(ticker_df["date"], ticker_df["y_true"], label="Actual", color="black", linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(pred_cols)))
    for col, color in zip(pred_cols[:5], colors):  # Plot top 5 models
        model_name = col.replace("y_pred_", "")
        ax.plot(ticker_df["date"], ticker_df[col], label=model_name, alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("RV")
    ax.set_title(f"{ticker} - Predictions vs Actual (H={horizon})")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_dir / f"predictions_{ticker}_h{horizon}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--horizon", type=int, nargs="+", default=HORIZONS)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    logger = setup_logger("compare_models", log_file=str(base_dir / RESULTS_DIR / "compare_models.log"))

    logger.info("=" * 70)
    logger.info("Model Comparison")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 70)

    pred_dir = base_dir / PREDICTIONS_DIR
    results_dir = base_dir / RESULTS_DIR
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    for d in [figures_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizon:
        logger.info(f"\n{'='*50}")
        logger.info(f"Horizon: {horizon} days")
        logger.info("=" * 50)

        # Load predictions
        df = load_all_predictions(pred_dir, horizon)

        if len(df) == 0:
            logger.warning(f"No predictions found for horizon {horizon}")
            continue

        logger.info(f"Loaded {len(df)} predictions")

        # Calculate aggregate metrics
        agg_metrics = calculate_aggregate_metrics(df)
        logger.info("\nAggregate Metrics:")
        logger.info(agg_metrics.to_string())

        agg_metrics.to_csv(metrics_dir / f"aggregate_metrics_h{horizon}.csv", index=False)

        # Calculate per-ticker metrics
        ticker_metrics = calculate_metrics_per_ticker(df)
        ticker_metrics.to_csv(metrics_dir / f"ticker_metrics_h{horizon}.csv", index=False)

        # Run DM tests
        dm_results = run_dm_tests(df)
        if len(dm_results) > 0:
            dm_results.to_csv(metrics_dir / f"dm_tests_h{horizon}.csv", index=False)
            logger.info(f"\nDM Tests: {len(dm_results)} pairs tested")

        # Run MCS
        mcs_results = run_mcs(df)
        if mcs_results:
            mcs_models = [m for m, in_mcs in mcs_results.items() if in_mcs]
            logger.info(f"\nModel Confidence Set: {mcs_models}")

            with open(metrics_dir / f"mcs_h{horizon}.txt", "w") as f:
                f.write(f"Models in MCS (H={horizon}):\n")
                for m in mcs_models:
                    f.write(f"  - {m}\n")

        # Plots
        if len(agg_metrics) > 0:
            plot_metrics_comparison(agg_metrics, horizon, figures_dir)

        # Plot for top ticker
        top_ticker = df["ticker"].value_counts().index[0]
        plot_predictions_vs_actual(df, horizon, top_ticker, figures_dir)

    logger.info("\n" + "=" * 70)
    logger.info("Model comparison complete!")
    logger.info(f"Results saved to {results_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
