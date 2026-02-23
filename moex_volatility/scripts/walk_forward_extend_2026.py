#!/usr/bin/env python3
"""
walk_forward_extend_2026.py — Extend walk-forward predictions to 2026
Train on 2014-2025, predict 2026 (partial year: Jan-Feb).
Then recompute ALL tables (QLIKE, DM, hybrid weights) for 2017-2026.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
PRED_DIR = BASE / "data" / "predictions" / "walk_forward"
TABLE_DIR = BASE / "results" / "tables"

# ============================================================
# Utility functions
# ============================================================

def qlike(y_true, y_pred):
    """QLIKE loss with safety checks."""
    y_pred = np.clip(y_pred, 1e-10, None)
    mask = (y_true > 1e-12) & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return np.nan
    ratio = yt / yp
    return np.mean(ratio - np.log(ratio) - 1)


def qlike_loss_per_obs(y_true, y_pred):
    """Per-observation QLIKE loss."""
    y_pred = np.clip(y_pred, 1e-10, None)
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1


def dm_test(loss1, loss2, h_horizon=1):
    """Diebold-Mariano test with Newey-West HAC."""
    d = loss1 - loss2
    mask = np.isfinite(d)
    d = d[mask]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    d_mean = np.mean(d)
    lag = max(1, h_horizon - 1)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, lag + 1):
        if len(d[k:]) > 1:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        else:
            gamma_k = 0.0
        gamma_sum += 2 * (1 - k / (lag + 1)) * gamma_k
    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


# ============================================================
# STEP 0: Check what exists
# ============================================================

def step0_check_data():
    """Check data coverage."""
    print("=" * 60)
    print("STEP 0: Check data coverage")
    print("=" * 60)

    # Load all data
    frames = []
    for name in ["train", "val", "test"]:
        df = pd.read_parquet(BASE / "data" / "prepared" / f"{name}.parquet")
        frames.append(df)
        print(f"  {name}: {len(df)} rows, {df['date'].min().date()} — {df['date'].max().date()}")

    for y in range(2020, 2027):
        path = BASE / "data" / "prepared" / f"walkforward_{y}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df)
            print(f"  walkforward_{y}: {len(df)} rows, {df['date'].min().date()} — {df['date'].max().date()}")

    full = pd.concat(frames, ignore_index=True)
    full["year"] = pd.to_datetime(full["date"]).dt.year
    print(f"\n  Full data: {len(full)} rows, {full['date'].min().date()} — {full['date'].max().date()}")
    print(f"  Years: {sorted(full['year'].unique())}")

    # Check walk-forward prediction coverage
    print("\n  Walk-forward prediction coverage:")
    for model in ["har", "xgboost", "lightgbm", "hybrid_adaptive"]:
        path = PRED_DIR / f"{model}_h1_annual.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            years = sorted(df["year"].unique())
            print(f"    {model}: years {years[0]}-{years[-1]}")

    # Check 2026 data
    data_2026 = full[full["year"] == 2026]
    if len(data_2026) == 0:
        print("\n  WARNING: No 2026 data found!")
        return None, None

    n_tickers = data_2026["ticker"].nunique()
    n_dates = data_2026["date"].nunique()
    print(f"\n  2026 data: {len(data_2026)} rows, {n_tickers} tickers, {n_dates} dates")
    print(f"    Date range: {data_2026['date'].min().date()} — {data_2026['date'].max().date()}")

    for h in [1, 5, 22]:
        col = f"rv_target_h{h}"
        if col in data_2026.columns:
            valid = data_2026[col].notna().sum()
            print(f"    {col}: {valid}/{len(data_2026)} non-null")

    if n_dates < 20:
        print(f"\n  NOTE: Partial year 2026 ({n_dates} trading days)")

    return full, data_2026


# ============================================================
# STEP 1: Prepare train/val/test splits for 2026
# ============================================================

def step1_prepare_splits(full):
    """Prepare data splits for 2026 prediction."""
    print("\n" + "=" * 60)
    print("STEP 1: Prepare data splits")
    print("=" * 60)

    full["year"] = pd.to_datetime(full["date"]).dt.year
    train_2026 = full[full["year"] <= 2025].copy()
    val_2025 = full[full["year"] == 2025].copy()
    test_2026 = full[full["year"] == 2026].copy()

    print(f"  Train (2014-2025): {len(train_2026)} rows")
    print(f"  Val for weights (2025): {len(val_2025)} rows")
    print(f"  Test 2026: {len(test_2026)} rows")

    return train_2026, val_2025, test_2026


# ============================================================
# STEP 2: Train models and predict 2026
# ============================================================

def get_har_features(config):
    """Get HAR-J feature columns."""
    har_base = config.get("har_features", ["rv_d", "rv_w", "rv_m"])
    # Add jump columns
    all_cols = config.get("feature_cols", [])
    jump_cols = [c for c in all_cols if "jump" in c.lower()
                 and "target" not in c.lower()
                 and not c.startswith("idx_")]
    return har_base + jump_cols


def train_har(train_df, val_df, test_df, h, config):
    """Train HAR-J model."""
    target = f"rv_target_h{h}"
    har_features = get_har_features(config)
    available = [c for c in har_features if c in train_df.columns]

    # Train
    tr = train_df.dropna(subset=[target]).copy()
    X_train = np.nan_to_num(np.log(tr[available].clip(1e-10).values), nan=0, posinf=0, neginf=0)
    y_train = np.log(tr[target].values + 1e-10)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict val
    va = val_df.dropna(subset=[target]).copy()
    X_val = np.nan_to_num(np.log(va[available].clip(1e-10).values), nan=0, posinf=0, neginf=0)
    val_pred = np.clip(np.exp(model.predict(X_val)), 1e-10, None)

    # Predict test
    te = test_df.dropna(subset=[target]).copy()
    X_test = np.nan_to_num(np.log(te[available].clip(1e-10).values), nan=0, posinf=0, neginf=0)
    test_pred = np.clip(np.exp(model.predict(X_test)), 1e-10, None)

    val_result = pd.DataFrame({
        "date": va["date"].values,
        "ticker": va["ticker"].values,
        "year": 2025,
        "rv_actual": va[target].values,
        "rv_pred": val_pred,
    })

    test_result = pd.DataFrame({
        "date": te["date"].values,
        "ticker": te["ticker"].values,
        "year": 2026,
        "rv_actual": te[target].values,
        "rv_pred": test_pred,
    })

    return val_result, test_result


def train_xgboost(train_df, val_df, test_df, h, config):
    """Train XGBoost model."""
    from xgboost import XGBRegressor

    target = f"rv_target_h{h}"
    feature_cols = config["feature_cols"]
    available = [c for c in feature_cols if c in train_df.columns]

    # Load params
    params_path = BASE / "models" / "xgboost" / f"params_h{h}.json"
    with open(params_path) as f:
        params = json.load(f)

    # Remove early_stopping_rounds from params (pass separately)
    early_stop = params.pop("early_stopping_rounds", 50)
    params.update({"tree_method": "hist", "verbosity": 0, "random_state": 42, "n_jobs": -1})

    # Train
    tr = train_df.dropna(subset=[target]).copy()
    X_train = np.nan_to_num(tr[available].values, nan=0, posinf=0, neginf=0)
    y_train_log = np.log(tr[target].values + 1e-10)

    # Use last part of training data as eval set for early stopping
    # Split: use 2025 data within train as eval
    tr_years = pd.to_datetime(tr["date"]).dt.year
    eval_mask = tr_years == 2025
    X_eval = X_train[eval_mask]
    y_eval = y_train_log[eval_mask]
    X_tr = X_train[~eval_mask]
    y_tr = y_train_log[~eval_mask]

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_eval, y_eval)],
              verbose=False)

    # Predict val (2025)
    va = val_df.dropna(subset=[target]).copy()
    X_val = np.nan_to_num(va[available].values, nan=0, posinf=0, neginf=0)
    val_pred = np.clip(np.exp(model.predict(X_val)), 1e-10, None)

    # Predict test (2026)
    te = test_df.dropna(subset=[target]).copy()
    X_test = np.nan_to_num(te[available].values, nan=0, posinf=0, neginf=0)
    test_pred = np.clip(np.exp(model.predict(X_test)), 1e-10, None)

    val_result = pd.DataFrame({
        "date": va["date"].values,
        "ticker": va["ticker"].values,
        "year": 2025,
        "rv_actual": va[target].values,
        "rv_pred": val_pred,
    })

    test_result = pd.DataFrame({
        "date": te["date"].values,
        "ticker": te["ticker"].values,
        "year": 2026,
        "rv_actual": te[target].values,
        "rv_pred": test_pred,
    })

    return val_result, test_result


def train_lightgbm(train_df, val_df, test_df, h, config):
    """Train LightGBM model."""
    from lightgbm import LGBMRegressor, early_stopping

    target = f"rv_target_h{h}"
    feature_cols = config["feature_cols"]
    available = [c for c in feature_cols if c in train_df.columns]

    # Load params
    params_path = BASE / "models" / "lightgbm" / f"params_h{h}.json"
    with open(params_path) as f:
        params = json.load(f)

    params.update({"verbosity": -1, "random_state": 42, "n_jobs": -1})

    # Handle GOSS + subsample conflict
    if params.get("boosting_type") == "goss":
        params["subsample"] = 1.0
        params.pop("bagging_freq", None)

    # Train
    tr = train_df.dropna(subset=[target]).copy()
    X_train = np.nan_to_num(tr[available].values, nan=0, posinf=0, neginf=0)
    y_train_log = np.log(tr[target].values + 1e-10)

    # For dart mode, no early stopping (just use n_estimators)
    is_dart = params.get("boosting_type") == "dart"

    # Use 2025 data within train as eval set
    tr_years = pd.to_datetime(tr["date"]).dt.year
    eval_mask = tr_years == 2025
    X_eval = X_train[eval_mask]
    y_eval = y_train_log[eval_mask]
    X_tr = X_train[~eval_mask]
    y_tr = y_train_log[~eval_mask]

    model = LGBMRegressor(**params)
    if is_dart:
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr,
                  eval_set=[(X_eval, y_eval)],
                  callbacks=[early_stopping(50, verbose=False)])

    # Predict val (2025)
    va = val_df.dropna(subset=[target]).copy()
    X_val = np.nan_to_num(va[available].values, nan=0, posinf=0, neginf=0)
    val_pred = np.clip(np.exp(model.predict(X_val)), 1e-10, None)

    # Predict test (2026)
    te = test_df.dropna(subset=[target]).copy()
    X_test = np.nan_to_num(te[available].values, nan=0, posinf=0, neginf=0)
    test_pred = np.clip(np.exp(model.predict(X_test)), 1e-10, None)

    val_result = pd.DataFrame({
        "date": va["date"].values,
        "ticker": va["ticker"].values,
        "year": 2025,
        "rv_actual": va[target].values,
        "rv_pred": val_pred,
    })

    test_result = pd.DataFrame({
        "date": te["date"].values,
        "ticker": te["ticker"].values,
        "year": 2026,
        "rv_actual": te[target].values,
        "rv_pred": test_pred,
    })

    return val_result, test_result


def train_hybrid_adaptive(val_preds, test_preds, h):
    """Build adaptive hybrid for 2026 using 2025 val predictions."""
    # val_preds: dict of model -> DataFrame with rv_actual, rv_pred for 2025
    # test_preds: dict of model -> DataFrame with rv_actual, rv_pred for 2026

    har_val = val_preds["har"]
    xgb_val = val_preds["xgboost"]
    lgb_val = val_preds["lightgbm"]

    # Merge val predictions on date+ticker
    merge_cols = ["date", "ticker"]
    val_merged = har_val[merge_cols + ["rv_actual", "rv_pred"]].rename(columns={"rv_pred": "har_pred"})
    val_merged = val_merged.merge(
        xgb_val[merge_cols + ["rv_pred"]].rename(columns={"rv_pred": "xgb_pred"}),
        on=merge_cols, how="inner"
    )
    val_merged = val_merged.merge(
        lgb_val[merge_cols + ["rv_pred"]].rename(columns={"rv_pred": "lgb_pred"}),
        on=merge_cols, how="inner"
    )

    # Filter valid
    mask = val_merged["rv_actual"] > 1e-12
    vm = val_merged[mask].copy()

    # Select best ML
    q_xgb = qlike(vm["rv_actual"].values, vm["xgb_pred"].values)
    q_lgb = qlike(vm["rv_actual"].values, vm["lgb_pred"].values)
    best_ml = "xgboost" if q_xgb < q_lgb else "lightgbm"
    ml_col = "xgb_pred" if best_ml == "xgboost" else "lgb_pred"

    print(f"    Val QLIKE: XGB={q_xgb:.4f}, LGB={q_lgb:.4f} -> best_ml={best_ml}")

    # Grid search w_har on val
    best_w = 0.0
    best_q = np.inf
    for w in np.arange(0.0, 0.61, 0.01):
        hybrid_pred = w * vm["har_pred"].values + (1 - w) * vm[ml_col].values
        q = qlike(vm["rv_actual"].values, hybrid_pred)
        if q < best_q:
            best_q = q
            best_w = w

    method = "val_grid"

    # Fallback: inverse QLIKE if w_har = 0
    if best_w < 0.01:
        q_har = qlike(vm["rv_actual"].values, vm["har_pred"].values)
        q_ml = qlike(vm["rv_actual"].values, vm[ml_col].values)
        if q_har > 0 and q_ml > 0:
            inv_har = 1.0 / q_har
            inv_ml = 1.0 / q_ml
            best_w = inv_har / (inv_har + inv_ml)
            method = "inverse_qlike"

    # Clip
    best_w = np.clip(best_w, 0.05, 0.55)

    print(f"    w_har={best_w:.2f}, method={method}")

    # Apply to test 2026
    har_test = test_preds["har"]
    ml_test = test_preds[best_ml]

    test_merged = har_test[merge_cols + ["rv_actual", "rv_pred"]].rename(columns={"rv_pred": "har_pred"})
    test_merged = test_merged.merge(
        ml_test[merge_cols + ["rv_pred"]].rename(columns={"rv_pred": "ml_pred"}),
        on=merge_cols, how="inner"
    )

    hybrid_pred = best_w * test_merged["har_pred"].values + (1 - best_w) * test_merged["ml_pred"].values
    hybrid_pred = np.clip(hybrid_pred, 1e-10, None)

    result = pd.DataFrame({
        "date": test_merged["date"].values,
        "ticker": test_merged["ticker"].values,
        "year": 2026,
        "rv_actual": test_merged["rv_actual"].values,
        "rv_pred": hybrid_pred,
    })

    return result, best_ml, best_w, method


def step2_train_and_predict(train_df, val_df, test_df, config):
    """Train all models and predict 2026."""
    print("\n" + "=" * 60)
    print("STEP 2: Train models on 2014-2025, predict 2026")
    print("=" * 60)

    all_results = {}  # model -> {h -> test_df}
    hybrid_details = []

    for h in [1, 5, 22]:
        target = f"rv_target_h{h}"
        n_test = test_df[target].notna().sum()
        print(f"\n  Horizon h={h} ({n_test} test observations with valid target)")

        if n_test == 0:
            print(f"    SKIP: no valid targets for h={h}")
            continue

        val_preds = {}
        test_preds = {}

        # HAR-J
        print(f"    Training HAR-J...")
        har_val, har_test = train_har(train_df, val_df, test_df, h, config)
        val_preds["har"] = har_val
        test_preds["har"] = har_test
        q_har = qlike(har_test["rv_actual"].values, har_test["rv_pred"].values)
        print(f"      HAR-J: {len(har_test)} preds, QLIKE={q_har:.4f}")

        # XGBoost
        print(f"    Training XGBoost...")
        xgb_val, xgb_test = train_xgboost(train_df, val_df, test_df, h, config)
        val_preds["xgboost"] = xgb_val
        test_preds["xgboost"] = xgb_test
        q_xgb = qlike(xgb_test["rv_actual"].values, xgb_test["rv_pred"].values)
        print(f"      XGBoost: {len(xgb_test)} preds, QLIKE={q_xgb:.4f}")

        # LightGBM
        print(f"    Training LightGBM...")
        lgb_val, lgb_test = train_lightgbm(train_df, val_df, test_df, h, config)
        val_preds["lightgbm"] = lgb_val
        test_preds["lightgbm"] = lgb_test
        q_lgb = qlike(lgb_test["rv_actual"].values, lgb_test["rv_pred"].values)
        print(f"      LightGBM: {len(lgb_test)} preds, QLIKE={q_lgb:.4f}")

        # Hybrid Adaptive
        print(f"    Building Hybrid Adaptive...")
        hybrid_test, best_ml, w_har, method = train_hybrid_adaptive(val_preds, test_preds, h)
        test_preds["hybrid_adaptive"] = hybrid_test
        q_hybrid = qlike(hybrid_test["rv_actual"].values, hybrid_test["rv_pred"].values)
        print(f"      Hybrid: {len(hybrid_test)} preds, QLIKE={q_hybrid:.4f}")

        # Store results
        for model_name in ["har", "xgboost", "lightgbm", "hybrid_adaptive"]:
            key = (model_name, h)
            all_results[key] = test_preds[model_name]

        # Compute ranks
        q_vals = {"HAR-J": q_har, "XGBoost": q_xgb, "LightGBM": q_lgb, "Hybrid": q_hybrid}
        ranked = sorted(q_vals.items(), key=lambda x: x[1])
        rank = [i + 1 for i, (m, _) in enumerate(ranked) if m == "Hybrid"][0]

        hybrid_details.append({
            "year": 2026, "horizon": h, "best_ml": best_ml,
            "w_har": w_har, "method": method,
            "q_hybrid": q_hybrid, "q_har": q_har,
            "q_xgb": q_xgb, "q_lgb": q_lgb, "hybrid_rank": rank,
        })

    return all_results, hybrid_details


# ============================================================
# STEP 3: Save predictions (append to existing)
# ============================================================

def step3_save_predictions(all_results):
    """Append 2026 predictions to existing parquets."""
    print("\n" + "=" * 60)
    print("STEP 3: Save predictions")
    print("=" * 60)

    for (model_name, h), test_df in all_results.items():
        fname = f"{model_name}_h{h}_annual.parquet"
        path = PRED_DIR / fname

        if path.exists():
            existing = pd.read_parquet(path)
            # Remove any existing 2026 rows
            existing = existing[existing["year"] != 2026]
            combined = pd.concat([existing, test_df], ignore_index=True)
            combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
            combined.to_parquet(path, index=False)
            print(f"  {fname}: {len(existing)} + {len(test_df)} -> {len(combined)} rows")
        else:
            test_df.to_parquet(path, index=False)
            print(f"  {fname}: NEW {len(test_df)} rows")


# ============================================================
# STEP 4: Recompute ALL tables (2017-2026)
# ============================================================

def step4_recompute_tables(hybrid_details):
    """Recompute QLIKE, DM tests, hybrid weights tables."""
    print("\n" + "=" * 60)
    print("STEP 4: Recompute tables (2017-2026)")
    print("=" * 60)

    models = ["har", "xgboost", "lightgbm", "hybrid_adaptive"]
    model_display = {"har": "HAR-J", "xgboost": "XGBoost",
                     "lightgbm": "LightGBM", "hybrid_adaptive": "V1_Adaptive"}

    # Load all predictions
    all_preds = {}
    for model in models:
        for h in [1, 5, 22]:
            path = PRED_DIR / f"{model}_h{h}_annual.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                all_preds[(model, h)] = df

    # --- QLIKE by year ---
    print("\n  Computing QLIKE by year...")
    qlike_rows = []
    for h in [1, 5, 22]:
        # Get years from HAR predictions
        har_df = all_preds.get(("har", h))
        if har_df is None:
            continue
        years = sorted(har_df["year"].unique())

        for year in years:
            row = {"year": year, "horizon": h}
            for model in models:
                df = all_preds.get((model, h))
                if df is None:
                    continue
                ydf = df[df["year"] == year]
                mask = (ydf["rv_actual"] > 1e-12) & np.isfinite(ydf["rv_actual"]) & np.isfinite(ydf["rv_pred"])
                ydf = ydf[mask]
                if len(ydf) > 0:
                    q = qlike(ydf["rv_actual"].values, ydf["rv_pred"].values)
                else:
                    q = np.nan
                row[model_display[model]] = q
            qlike_rows.append(row)

    qlike_df = pd.DataFrame(qlike_rows)

    # Save per-horizon QLIKE
    for h in [1, 5, 22]:
        hdf = qlike_df[qlike_df["horizon"] == h].drop(columns=["horizon"])
        hdf.to_csv(TABLE_DIR / f"qlike_by_year_h{h}.csv", index=False)

    # Save wide format for strategies_v3 compatibility
    qlike_wide = qlike_df.copy()
    qlike_wide.to_csv(TABLE_DIR / "strategies_v4.csv", index=False)
    print(f"    Saved qlike_by_year and strategies_v4")

    # --- Overall QLIKE ---
    print("\n  Computing overall QLIKE (mean across years)...")
    overall = {}
    for h in [1, 5, 22]:
        hdf = qlike_df[qlike_df["horizon"] == h]
        for model in models:
            dname = model_display[model]
            if dname in hdf.columns:
                overall[(dname, h)] = hdf[dname].mean()

    # Build summary table
    summary_rows = []
    for model in models:
        dname = model_display[model]
        row = {"Model": dname}
        for h in [1, 5, 22]:
            row[f"H={h}"] = overall.get((dname, h), np.nan)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLE_DIR / "final_summary_v3.csv", index=False)
    print(f"    Saved final_summary_v3.csv")

    # --- DM Tests ---
    print("\n  Computing Diebold-Mariano tests...")
    dm_rows = []
    for h in [1, 5, 22]:
        # Merge all predictions on date+ticker
        base = all_preds.get(("har", h))
        if base is None:
            continue

        merged = base[["date", "ticker", "year", "rv_actual"]].copy()
        merged = merged.rename(columns={"rv_pred": "pred_har"}) if "rv_pred" in merged.columns else merged

        for model in models:
            df = all_preds.get((model, h))
            if df is not None:
                tmp = df[["date", "ticker", "rv_pred"]].rename(columns={"rv_pred": f"pred_{model}"})
                merged = merged.merge(tmp, on=["date", "ticker"], how="inner")

        # Filter valid
        mask = (merged["rv_actual"] > 1e-12) & np.isfinite(merged["rv_actual"])
        for model in models:
            col = f"pred_{model}"
            if col in merged.columns:
                mask = mask & np.isfinite(merged[col])
        merged = merged[mask]

        y = merged["rv_actual"].values

        # Compute per-obs QLIKE for each model
        losses = {}
        for model in models:
            col = f"pred_{model}"
            if col in merged.columns:
                losses[model] = qlike_loss_per_obs(y, merged[col].values)

        # Pairwise DM tests: V1_Adaptive vs others
        ref = "hybrid_adaptive"
        for other in ["har", "xgboost", "lightgbm"]:
            if ref in losses and other in losses:
                dm_stat, p_val = dm_test(losses[other], losses[ref], h)
                winner = model_display[ref] if dm_stat > 0 else model_display[other]
                dm_rows.append({
                    "Horizon": h,
                    "Model1": model_display[ref],
                    "Model2": model_display[other],
                    "DM_stat": dm_stat,
                    "p_value": p_val,
                    "Winner": winner,
                })

        # Also: all pairwise among base models
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                if m1 in losses and m2 in losses:
                    dm_stat, p_val = dm_test(losses[m1], losses[m2], h)
                    winner = model_display[m2] if dm_stat > 0 else model_display[m1]
                    dm_rows.append({
                        "Horizon": h,
                        "Model1": model_display[m1],
                        "Model2": model_display[m2],
                        "DM_stat": dm_stat,
                        "p_value": p_val,
                        "Winner": winner,
                    })

    dm_df = pd.DataFrame(dm_rows)
    for h in [1, 5, 22]:
        hdf = dm_df[dm_df["Horizon"] == h]
        hdf.to_csv(TABLE_DIR / f"dm_tests_v4_h{h}.csv", index=False)
    print(f"    Saved dm_tests_v4")

    # --- Hybrid weights (update adaptive_details) ---
    print("\n  Updating adaptive details...")
    ad_path = TABLE_DIR / "adaptive_details.csv"
    if ad_path.exists():
        ad = pd.read_csv(ad_path)
        # Remove existing 2026
        ad = ad[ad["year"] != 2026]
        new_rows = pd.DataFrame(hybrid_details)
        ad = pd.concat([ad, new_rows], ignore_index=True)
        ad = ad.sort_values(["year", "horizon"]).reset_index(drop=True)
        ad.to_csv(ad_path, index=False)
        print(f"    Updated adaptive_details.csv with 2026")

    return qlike_df, summary_df, dm_df


# ============================================================
# STEP 5: Print summary
# ============================================================

def step5_print_summary(qlike_df, summary_df, dm_df, hybrid_details):
    """Print comprehensive summary."""
    print("\n" + "=" * 60)
    print("STEP 5: Summary")
    print("=" * 60)

    # 1. Full date range
    har_h1 = pd.read_parquet(PRED_DIR / "har_h1_annual.parquet")
    print(f"\n  1. Full data range: {har_h1['date'].min().date()} — {har_h1['date'].max().date()}")

    # 2. Walk-forward coverage
    years = sorted(har_h1["year"].unique())
    print(f"  2. Walk-forward coverage: {years[0]}-{years[-1]} ({len(years)} years)")
    year_counts = har_h1.groupby("year").size()
    for y in years:
        n = year_counts.get(y, 0)
        marker = " (PARTIAL)" if n < 500 else ""
        print(f"       {y}: {n} obs{marker}")

    # 3. Updated QLIKE table
    print(f"\n  3. Updated QLIKE table (mean 2017-{years[-1]}):")
    print(f"     {'Model':<20} {'H=1':<12} {'H=5':<12} {'H=22':<12}")
    print(f"     {'-'*56}")
    for _, row in summary_df.iterrows():
        m = row["Model"]
        h1 = f"{row.get('H=1', np.nan):.4f}"
        h5 = f"{row.get('H=5', np.nan):.4f}"
        h22 = f"{row.get('H=22', np.nan):.4f}"
        print(f"     {m:<20} {h1:<12} {h5:<12} {h22:<12}")

    # 4. 2026-only QLIKE
    print(f"\n     2026 QLIKE only:")
    q2026 = qlike_df[qlike_df["year"] == 2026]
    for _, row in q2026.iterrows():
        h = int(row["horizon"])
        vals = []
        for m in ["HAR-J", "XGBoost", "LightGBM", "V1_Adaptive"]:
            if m in row:
                vals.append(f"{m}={row[m]:.4f}")
        print(f"       H={h}: {', '.join(vals)}")

    # 5. Hybrid weights 2026
    print(f"\n  4. Hybrid Adaptive weights (2026):")
    for d in hybrid_details:
        h = d["horizon"]
        print(f"       H={h}: best_ml={d['best_ml']}, w_har={d['w_har']:.2f}, "
              f"method={d['method']}, QLIKE={d['q_hybrid']:.4f}, rank={d['hybrid_rank']}")

    # 6. DM tests
    print(f"\n  5. DM tests (V1_Adaptive vs others, full 2017-{years[-1]}):")
    ref_tests = dm_df[dm_df["Model1"] == "V1_Adaptive"]
    for _, row in ref_tests.iterrows():
        h = int(row["Horizon"])
        m2 = row["Model2"]
        stat = row["DM_stat"]
        p = row["p_value"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        winner = row["Winner"]
        print(f"       H={h}: vs {m2:<12} DM={stat:+.3f}, p={p:.4f} {sig:>3} -> {winner}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Walk-Forward Extension to 2026")
    print("=" * 60)

    # Load config
    with open(BASE / "data" / "prepared" / "config.json") as f:
        config = json.load(f)

    # Step 0
    full, data_2026 = step0_check_data()
    if data_2026 is None or len(data_2026) == 0:
        print("\nWARNING: No 2026 data available. Walk-forward covers 2017-2025 only.")
        return

    # Step 1
    train_df, val_df, test_df = step1_prepare_splits(full)

    # Step 2
    all_results, hybrid_details = step2_train_and_predict(train_df, val_df, test_df, config)

    if not all_results:
        print("\nWARNING: No valid predictions generated for 2026.")
        return

    # Step 3
    step3_save_predictions(all_results)

    # Step 4
    qlike_df, summary_df, dm_df = step4_recompute_tables(hybrid_details)

    # Step 5
    step5_print_summary(qlike_df, summary_df, dm_df, hybrid_details)

    print("\n" + "=" * 60)
    print("DONE: Walk-forward extended to 2026")
    print("=" * 60)


if __name__ == "__main__":
    main()
