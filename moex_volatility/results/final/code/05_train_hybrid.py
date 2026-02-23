#!/usr/bin/env python3
"""Train hybrid models (HAR + ML on residuals).

Usage:
    python scripts/05_train_hybrid.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.models.hybrid import HybridHARML, LIGHTGBM_AVAILABLE

# Try to use LightGBM version if available
if LIGHTGBM_AVAILABLE:
    from src.models.hybrid import HybridHARLGBM
    HybridModel = HybridHARLGBM
    print("✅ Using LightGBM for hybrid model")
else:
    HybridModel = HybridHARML
    print("⚠️ LightGBM not available, using HistGradientBoosting")

# Paths
DATA_PATH = Path('data/features/features_all.parquet')
MODELS_PATH = Path('models/hybrid')
PREDICTIONS_PATH = Path('data/predictions/test_2019')
MODELS_PATH.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)


def qlike(y_true, y_pred):
    """QLIKE metric - lower is better."""
    y_pred = np.clip(y_pred, 1e-10, None)
    y_true = np.clip(y_true, 1e-10, None)
    return np.mean(np.log(y_pred) + y_true / y_pred)


def main():
    # Load data
    print("Loading features...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Shape: {df.shape}")

    # Split
    train_mask = df['date'] <= '2017-12-31'
    val_mask = (df['date'] > '2017-12-31') & (df['date'] <= '2018-12-31')
    test_mask = (df['date'] > '2018-12-31') & (df['date'] <= '2019-12-31')

    # Define HAR features (classic HAR-RV components only)
    # rv_d = 1-day lag, rv_w = 5-day average, rv_m = 22-day average
    har_features_candidates = [
        'rv_d', 'rv_w', 'rv_m',       # Classic HAR components (Corsi 2009)
    ]

    # Get available HAR features
    har_features = [f for f in har_features_candidates if f in df.columns]
    print(f"\nHAR features ({len(har_features)}): {har_features}")

    # External features (everything numeric that's not HAR or meta)
    # Exclude current rv, bv, jump (would be data leak) and HAR features
    exclude_cols = ['date', 'ticker', 'rv_target', 'source', 'rv', 'bv', 'jump',
                    'rsv_pos', 'rsv_neg', 'rskew', 'rkurt'] + har_features
    exclude_prefixes = ['rv_target', 'low_bars', 'source']

    external_features = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if df[col].dtype in ['object', 'bool']:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            external_features.append(col)

    print(f"External features ({len(external_features)}): {external_features[:15]}...")

    results = []

    for h in [1, 5, 22]:
        print(f"\n{'='*60}")
        print(f"Training Hybrid Model for h={h}")
        print('='*60)

        # Create target
        df_h = df.copy()
        df_h['rv_target'] = df_h.groupby('ticker')['rv'].shift(-h)

        # Prepare data splits
        train_df = df_h[train_mask].copy()
        val_df = df_h[val_mask].copy()
        test_df = df_h[test_mask].copy()

        # Fill NaN in features
        all_features = har_features + external_features
        for col in all_features:
            if col in train_df.columns:
                train_df[col] = train_df[col].ffill().fillna(0)
                val_df[col] = val_df[col].ffill().fillna(0)
                test_df[col] = test_df[col].ffill().fillna(0)

        # Drop rows without target
        train_df = train_df.dropna(subset=['rv_target'])
        val_df = val_df.dropna(subset=['rv_target'])
        test_df = test_df.dropna(subset=['rv_target'])

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Combine train + val for final model
        train_val_df = pd.concat([train_df, val_df])

        X_train_val = train_val_df[all_features]
        y_train_val = train_val_df['rv_target'].values
        X_test = test_df[all_features]
        y_test = test_df['rv_target'].values

        # Train hybrid model (uses LightGBM if available)
        model = HybridModel(h, har_features, external_features)
        model.fit(X_train_val, y_train_val)

        # Predictions
        pred_test = model.predict(X_test)

        # QLIKE
        qlike_test = qlike(y_test, pred_test)

        print(f"\nQLIKE Test: {qlike_test:.4f}")

        # HAR coefficients
        print(f"\nHAR coefficients:")
        coefs = model.get_har_coefs()
        for feat, coef in coefs.items():
            print(f"  {feat}: {coef:.6f}")

        # ML feature importance
        importance = model.get_ml_importance()
        if importance is not None:
            print(f"\nTop-10 external features (for residual prediction):")
            print(importance.head(10).to_string(index=False))

        # Save model
        model.save(MODELS_PATH / f'model_h{h}.pkl')

        # Save predictions
        pred_df = test_df[['date', 'ticker']].copy()
        pred_df['rv_actual'] = y_test
        pred_df['rv_pred'] = pred_test
        pred_df.to_parquet(PREDICTIONS_PATH / f'hybrid_h{h}.parquet', index=False)

        results.append({
            'horizon': h,
            'qlike_test': qlike_test
        })

    # Final comparison with all models
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON")
    print("="*70)

    # Reference QLIKE values
    ref_models = {
        'HAR-RV': {1: -7.527, 5: -7.414, 22: -7.340},
        'LSTM': {1: -7.305, 5: -7.357, 22: -6.977},
        'GARCH-GJR': {1: -7.444, 5: -6.145, 22: -4.248},
        'XGBoost': {1: -7.133, 5: -7.131, 22: None},
        'GRU': {1: -6.750, 5: -6.651, 22: -4.609},
        'HistGBoost': {1: -6.617, 5: -6.623, 22: -6.569},
    }

    hybrid_qlike = {r['horizon']: r['qlike_test'] for r in results}

    print(f"\n{'Model':<17} {'H=1':>10} {'H=5':>10} {'H=22':>10}")
    print("-"*50)

    for model_name, scores in ref_models.items():
        s1 = f"{scores[1]:.3f}" if scores[1] else 'N/A'
        s5 = f"{scores[5]:.3f}" if scores[5] else 'N/A'
        s22 = f"{scores[22]:.3f}" if scores[22] else 'N/A'
        print(f"{model_name:<17} {s1:>10} {s5:>10} {s22:>10}")

    print(f"{'HYBRID (HAR+ML)':<17} {hybrid_qlike[1]:>10.3f} {hybrid_qlike[5]:>10.3f} {hybrid_qlike[22]:>10.3f}")

    print("\n" + "-"*50)

    # Check if hybrid is best for each horizon
    for h in [1, 5, 22]:
        all_scores = {name: scores[h] for name, scores in ref_models.items() if scores[h] is not None}
        all_scores['HYBRID'] = hybrid_qlike[h]

        best_model = min(all_scores, key=all_scores.get)
        best_score = all_scores[best_model]

        if best_model == 'HYBRID':
            print(f"✅ Hybrid is BEST for H={h} (QLIKE={best_score:.3f})")
        else:
            diff = hybrid_qlike[h] - best_score
            print(f"❌ Hybrid NOT best for H={h} (best: {best_model} {best_score:.3f}, gap: {diff:+.3f})")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv('results/hybrid_summary.csv', index=False)

    print("\nDone!")
    print(f"Models saved to: {MODELS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
