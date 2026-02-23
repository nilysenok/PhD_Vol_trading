#!/usr/bin/env python3
"""Build all features from 10-minute candles and external data.

Usage:
    python scripts/01_build_features.py
    python scripts/01_build_features.py --no-polars
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime

from src.data.external_features import build_all_features


# Configuration
DATASET_PATH = "../moex_discovery/data/dataset_final"
OUTPUT_DIR = "data/features"

STOCK_TICKERS = [
    "SBER", "LKOH", "TATN", "NVTK", "VTBR", "ALRS", "AFLT",
    "HYDR", "MGNT", "MOEX", "RTKM", "MTSS", "MTLR", "IRAO",
    "OGKB", "PHOR", "LSRG"
]


def main():
    parser = argparse.ArgumentParser(description="Build volatility features")
    parser.add_argument("--no-polars", action="store_true", help="Use pandas instead of polars")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH, help="Path to dataset_final")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("MOEX Volatility - Feature Engineering")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / args.dataset_path
    output_dir = base_dir / args.output_dir

    print(f"\nDataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using polars: {not args.no_polars}")
    print(f"Tickers: {len(STOCK_TICKERS)}")

    # Build features
    output_path = build_all_features(
        dataset_path=str(dataset_path),
        stock_tickers=STOCK_TICKERS,
        output_dir=str(output_dir),
        use_polars=not args.no_polars,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("Feature engineering complete!")
    print(f"Output: {output_path}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
