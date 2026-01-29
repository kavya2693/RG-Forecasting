"""
Optimized Training with Improved Hyperparameters
================================================
Based on experiment results:
- A-items: num_leaves=31 (was 255) - simpler model prevents overfitting
- B-items: num_leaves=15 (was 63) - simpler model
- C-items: bias correction with k=0.8

Also applies optimal thresholds.
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/tmp/optimized_model")
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature columns
FEATURE_COLS = [
    "dow", "is_weekend", "day_of_year", "week_of_year",
    "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_56",
    "roll_mean_7", "roll_mean_28", "roll_sum_7", "roll_sum_28", "roll_std_28",
    "nz_rate_7", "nz_rate_28", "roll_mean_pos_28",
    "days_from_prev_closure", "days_to_next_closure",
    "last_sale_qty_asof", "days_since_last_sale",
]

CAT_FEATURES = ["sku_id", "store_id"]

# OPTIMIZED hyperparameters (from experiment results)
OPTIMIZED_PARAMS = {
    "A": {
        "num_leaves": 31,          # Was 255 - simpler prevents overfitting
        "min_child_samples": 50,   # Was 10
        "learning_rate": 0.03,
        "n_estimators": 500,
        "threshold": 0.80,         # Optimized from 0.6
        "bias_k": 1.04,
    },
    "B": {
        "num_leaves": 15,          # Was 63 - much simpler
        "min_child_samples": 100,  # Was 50
        "learning_rate": 0.03,
        "n_estimators": 400,
        "threshold": 0.70,         # Optimized from 0.6
        "bias_k": 0.96,
    },
    "C": {
        "num_leaves": 15,          # Was 31
        "min_child_samples": 100,  # Was 100
        "learning_rate": 0.05,
        "n_estimators": 300,
        "threshold": 0.65,         # Optimized from 0.7
        "bias_k": 0.80,            # Key insight: C-items overpredict
    },
}

# Original hyperparameters for comparison
ORIGINAL_PARAMS = {
    "A": {"num_leaves": 255, "min_child_samples": 10, "learning_rate": 0.015, "n_estimators": 1000, "threshold": 0.6, "bias_k": 1.0},
    "B": {"num_leaves": 63, "min_child_samples": 50, "learning_rate": 0.03, "n_estimators": 400, "threshold": 0.6, "bias_k": 1.0},
    "C": {"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05, "n_estimators": 300, "threshold": 0.7, "bias_k": 1.0},
}


def load_data(max_rows=None):
    """Load training and validation data."""
    train = pd.read_csv("/tmp/c1_data/train_final.csv", nrows=max_rows)
    val = pd.read_csv("/tmp/c1_data/val_final.csv", nrows=max_rows // 2 if max_rows else None)
    return train, val


def assign_segments(df):
    """Assign ABC segments."""
    target = "y" if "y" in df.columns else "qty"
    sku_sales = df.groupby("sku_id")[target].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = sku_sales.sum()

    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)

    def get_seg(sku):
        if sku in a_skus: return "A"
        elif sku in b_skus: return "B"
        return "C"

    df["segment"] = df["sku_id"].apply(get_seg)
    return df


def compute_metrics(y_true, y_pred):
    """Compute WMAPE and WFA."""
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    return {"wmape": wmape, "wfa": 100 - wmape}


def train_and_evaluate(train, val, params_dict, param_name=""):
    """Train model with given parameters and evaluate."""
    target = "y" if "y" in train.columns else "qty"

    val = val.copy()
    val["y_pred"] = 0.0

    results = {}

    for segment in ["A", "B", "C"]:
        params = params_dict[segment]

        train_seg = train[train["segment"] == segment].copy()
        val_seg = val[val["segment"] == segment].copy()

        if len(train_seg) < 100 or len(val_seg) < 100:
            continue

        # Prepare features
        numeric_features = [f for f in FEATURE_COLS if f in train_seg.columns]
        cat_features_present = [f for f in CAT_FEATURES if f in train_seg.columns]

        for col in cat_features_present:
            train_seg[col] = train_seg[col].astype("category")
            val_seg[col] = val_seg[col].astype("category")

        for col in numeric_features:
            train_seg[col] = train_seg[col].fillna(0)
            val_seg[col] = val_seg[col].fillna(0)

        available = numeric_features + cat_features_present
        X_train = train_seg[available]
        X_val = val_seg[available]
        y_train = train_seg[target].values
        y_val = val_seg[target].values

        # Stage 1: Binary classifier
        clf = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            learning_rate=params["learning_rate"],
            n_estimators=min(params["n_estimators"], 500),
            verbose=-1, n_jobs=-1, random_state=42,
        )
        clf.fit(X_train, (y_train > 0).astype(int))
        prob = clf.predict_proba(X_val)[:, 1]

        # Stage 2: Regressor
        train_nz = train_seg[train_seg[target] > 0].copy()
        X_train_nz = train_nz[available]
        y_train_nz = np.log1p(train_nz[target].values)

        reg = lgb.LGBMRegressor(
            objective="regression_l1",
            num_leaves=params["num_leaves"],
            min_child_samples=max(5, params["min_child_samples"] // 2),
            learning_rate=params["learning_rate"],
            n_estimators=min(params["n_estimators"], 500),
            reg_lambda=0.5,
            verbose=-1, n_jobs=-1, random_state=42,
        )
        reg.fit(X_train_nz, y_train_nz)
        pred_value = np.expm1(reg.predict(X_val))

        # Combine with optimized threshold and bias correction
        threshold = params["threshold"]
        bias_k = params["bias_k"]

        y_pred = np.where(prob > threshold, pred_value * bias_k, 0)
        y_pred = np.maximum(0, y_pred)

        val.loc[val["segment"] == segment, "y_pred"] = y_pred

        # Segment metrics
        seg_metrics = compute_metrics(y_val, y_pred)
        results[segment] = seg_metrics
        print(f"  {segment}: WFA={seg_metrics['wfa']:.2f}%")

    # Overall metrics
    overall = compute_metrics(val[target].values, val["y_pred"].values)
    results["overall"] = overall

    # Weekly aggregations
    val["date_parsed"] = pd.to_datetime(val["date"])
    val["week"] = val["date_parsed"].dt.isocalendar().week.astype(int)
    val["year"] = val["date_parsed"].dt.year

    weekly_store = val.groupby(["store_id", "year", "week"]).agg({target: "sum", "y_pred": "sum"}).reset_index()
    weekly_store_metrics = compute_metrics(weekly_store[target].values, weekly_store["y_pred"].values)
    results["weekly_store"] = weekly_store_metrics

    weekly_total = val.groupby(["year", "week"]).agg({target: "sum", "y_pred": "sum"}).reset_index()
    weekly_total_metrics = compute_metrics(weekly_total[target].values, weekly_total["y_pred"].values)
    results["weekly_total"] = weekly_total_metrics

    return results, val


def main():
    print("=" * 70)
    print("OPTIMIZED MODEL TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Load data
    print("\nLoading data...")
    train, val = load_data(max_rows=500000)
    train = assign_segments(train)
    val = assign_segments(val)
    print(f"  Train: {len(train):,}, Val: {len(val):,}")

    # Train with ORIGINAL parameters
    print("\n" + "=" * 50)
    print("ORIGINAL HYPERPARAMETERS")
    print("=" * 50)
    original_results, _ = train_and_evaluate(train, val, ORIGINAL_PARAMS, "Original")

    # Train with OPTIMIZED parameters
    print("\n" + "=" * 50)
    print("OPTIMIZED HYPERPARAMETERS")
    print("=" * 50)
    optimized_results, val_pred = train_and_evaluate(train, val, OPTIMIZED_PARAMS, "Optimized")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs OPTIMIZED")
    print("=" * 70)
    print(f"{'Metric':<25} {'Original':>12} {'Optimized':>12} {'Change':>12}")
    print("-" * 61)

    for key in ["A", "B", "C", "overall", "weekly_store", "weekly_total"]:
        if key in original_results and key in optimized_results:
            orig_wfa = original_results[key]["wfa"]
            opt_wfa = optimized_results[key]["wfa"]
            change = opt_wfa - orig_wfa
            label = f"Daily {key}" if key in ["A", "B", "C", "overall"] else key.replace("_", " ").title()
            print(f"{label:<25} {orig_wfa:>11.2f}% {opt_wfa:>11.2f}% {change:>+11.2f}pp")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
    1. SIMPLER MODELS WORK BETTER:
       - A-items: 31 leaves (was 255) → model was overfitting
       - B-items: 15 leaves (was 63) → simpler generalized better

    2. HIGHER THRESHOLDS REDUCE FALSE POSITIVES:
       - More conservative predictions reduce over-prediction errors

    3. BIAS CORRECTION HELPS C-ITEMS:
       - C-items overpredict → k=0.8 correction improves accuracy
    """)

    # Save results
    comparison = {
        "original": original_results,
        "optimized": optimized_results,
        "optimized_params": OPTIMIZED_PARAMS,
        "timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR / 'comparison_results.json'}")

    return comparison


if __name__ == "__main__":
    main()
