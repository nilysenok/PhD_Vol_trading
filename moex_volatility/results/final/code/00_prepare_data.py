#!/usr/bin/env python3
"""Prepare data for parallel model training.

This script loads features, creates train/val/test/walkforward splits,
fills NaN values with train medians, and saves prepared datasets.

Usage:
    python scripts/00_prepare_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd

# Paths
FEATURES_PATH = Path('data/features/features_all.parquet')
PREPARED_PATH = Path('data/prepared')
PREPARED_PATH.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load data
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])

    # 2. Define columns
    exclude = ['date', 'ticker', 'rv_target_h1', 'rv_target_h5', 'rv_target_h22']

    # All features (exclude meta and target columns)
    all_features = [c for c in df.columns
                    if c not in exclude
                    and 'target' not in c.lower()
                    and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # HAR features (find correct names)
    har_candidates = ['rv_d', 'rv_lag1', 'rv_w', 'rv_week', 'rv_m', 'rv_month']
    har_features = [c for c in har_candidates if c in df.columns][:3]

    # External features
    external_features = [c for c in all_features if c not in har_features]

    # 3. Create target columns
    for h in [1, 5, 22]:
        df[f'rv_target_h{h}'] = df.groupby('ticker')['rv'].shift(-h)

    # 4. Split by date
    train_mask = df['date'] <= '2017-12-31'
    val_mask = (df['date'] >= '2018-01-01') & (df['date'] <= '2018-12-31')
    test_mask = (df['date'] >= '2019-01-01') & (df['date'] <= '2019-12-31')

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # 5. Walk-forward data by year
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    wf_dfs = {}
    wf_total = 0
    for year in wf_years:
        year_mask = df['date'].dt.year == year
        if year_mask.sum() > 0:
            wf_dfs[year] = df[year_mask].copy()
            wf_total += len(wf_dfs[year])

    # 6. Compute train medians and fill NaN
    train_medians = {}
    for col in all_features:
        median_val = train_df[col].median()
        train_medians[col] = float(median_val) if pd.notna(median_val) else 0.0

    # Fill NaN in all splits
    for split_df in [train_df, val_df, test_df] + list(wf_dfs.values()):
        for col in all_features:
            split_df[col] = split_df[col].fillna(train_medians[col])

    # 7. Save datasets
    train_df.to_parquet(PREPARED_PATH / 'train.parquet', index=False)
    val_df.to_parquet(PREPARED_PATH / 'val.parquet', index=False)
    test_df.to_parquet(PREPARED_PATH / 'test.parquet', index=False)

    for year, wf_df in wf_dfs.items():
        wf_df.to_parquet(PREPARED_PATH / f'walkforward_{year}.parquet', index=False)

    # Save config
    config = {
        'feature_cols': all_features,
        'har_features': har_features,
        'external_features': external_features,
        'horizons': [1, 5, 22],
        'train_period': '2014-2017',
        'val_period': '2018',
        'test_period': '2019',
        'wf_years': list(wf_dfs.keys())
    }
    with open(PREPARED_PATH / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save medians
    with open(PREPARED_PATH / 'medians.json', 'w') as f:
        json.dump(train_medians, f, indent=2)

    # Output summary
    print("Data prepared!")
    print(f"Train: {len(train_df)} rows (2014-2017)")
    print(f"Val: {len(val_df)} rows (2018)")
    print(f"Test: {len(test_df)} rows (2019)")
    print(f"WalkForward: {wf_total} rows (2020-2026)")
    print(f"Features: {len(all_features)} total, {len(har_features)} HAR, {len(external_features)} external")


if __name__ == "__main__":
    main()
