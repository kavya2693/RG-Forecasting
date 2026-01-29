"""
Balanced Training - Optimizing for Both Daily and Weekly Accuracy
================================================================
Strategy: Use simpler models (prevent overfitting) but keep calibrated thresholds
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/tmp/balanced_model")
OUTPUT_DIR.mkdir(exist_ok=True)

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

# BALANCED: Simpler models (from experiments) but calibrated thresholds
BALANCED_PARAMS = {
    "A": {
        "num_leaves": 63,          # Reduced from 255 but not too aggressive
        "min_child_samples": 20,
        "learning_rate": 0.02,
        "n_estimators": 600,
        "threshold": 0.55,         # Lower threshold to not miss sales
        "bias_k": 1.0,             # No bias correction - let calibration handle it
    },
    "B": {
        "num_leaves": 31,          # Reduced from 63
        "min_child_samples": 50,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "threshold": 0.55,
        "bias_k": 1.0,
    },
    "C": {
        "num_leaves": 15,          # Reduced from 31
        "min_child_samples": 100,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "threshold": 0.60,         # Slightly higher for sparse items
        "bias_k": 1.0,
    },
}


def load_data(max_rows=None):
    train = pd.read_csv("/tmp/c1_data/train_final.csv", nrows=max_rows)
    val = pd.read_csv("/tmp/c1_data/val_final.csv", nrows=max_rows // 2 if max_rows else None)
    return train, val


def assign_segments(df):
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
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    return {"wmape": wmape, "wfa": 100 - wmape}


def train_with_calibration(train, val, params_dict):
    """Train model with automatic calibration per segment."""
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

        # Stage 1: Classifier
        clf = lgb.LGBMClassifier(
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            learning_rate=params["learning_rate"],
            n_estimators=min(params["n_estimators"], 500),
            verbose=-1, n_jobs=-1, random_state=42,
        )
        clf.fit(X_train, (y_train > 0).astype(int))
        prob_val = clf.predict_proba(X_val)[:, 1]
        prob_train = clf.predict_proba(X_train)[:, 1]

        # Stage 2: Regressor
        train_nz = train_seg[train_seg[target] > 0].copy()
        X_train_nz = train_nz[available]
        y_train_nz = np.log1p(train_nz[target].values)

        reg = lgb.LGBMRegressor(
            num_leaves=params["num_leaves"],
            min_child_samples=max(5, params["min_child_samples"] // 2),
            learning_rate=params["learning_rate"],
            n_estimators=min(params["n_estimators"], 500),
            reg_lambda=0.5,
            verbose=-1, n_jobs=-1, random_state=42,
        )
        reg.fit(X_train_nz, y_train_nz)
        pred_val = np.expm1(reg.predict(X_val))
        pred_train = np.expm1(reg.predict(X_train))

        # Combine with threshold
        threshold = params["threshold"]
        y_pred_val = np.where(prob_val > threshold, pred_val, 0)
        y_pred_val = np.maximum(0, y_pred_val)

        y_pred_train = np.where(prob_train > threshold, pred_train, 0)
        y_pred_train = np.maximum(0, y_pred_train)

        # Calibration: match total volume on training data
        mask = y_pred_train > 0.1
        if np.sum(y_pred_train[mask]) > 0:
            k = np.sum(y_train[mask]) / np.sum(y_pred_train[mask])
            k = np.clip(k, 0.8, 1.3)  # Reasonable bounds
            y_pred_val = y_pred_val * k
            print(f"  {segment}: calibration k={k:.3f}")

        val.loc[val["segment"] == segment, "y_pred"] = y_pred_val

        seg_metrics = compute_metrics(y_val, y_pred_val)
        results[segment] = seg_metrics
        print(f"  {segment}: WFA={seg_metrics['wfa']:.2f}%")

    # Overall and aggregated metrics
    overall = compute_metrics(val[target].values, val["y_pred"].values)
    results["daily_sku_store"] = overall

    val["date_parsed"] = pd.to_datetime(val["date"])
    val["week"] = val["date_parsed"].dt.isocalendar().week.astype(int)
    val["year"] = val["date_parsed"].dt.year

    weekly_sku_store = val.groupby(["store_id", "sku_id", "year", "week"]).agg({target: "sum", "y_pred": "sum"}).reset_index()
    results["weekly_sku_store"] = compute_metrics(weekly_sku_store[target].values, weekly_sku_store["y_pred"].values)

    weekly_store = val.groupby(["store_id", "year", "week"]).agg({target: "sum", "y_pred": "sum"}).reset_index()
    results["weekly_store"] = compute_metrics(weekly_store[target].values, weekly_store["y_pred"].values)

    weekly_total = val.groupby(["year", "week"]).agg({target: "sum", "y_pred": "sum"}).reset_index()
    results["weekly_total"] = compute_metrics(weekly_total[target].values, weekly_total["y_pred"].values)

    return results


def main():
    print("=" * 70)
    print("BALANCED MODEL TRAINING")
    print("=" * 70)

    train, val = load_data(max_rows=500000)
    train = assign_segments(train)
    val = assign_segments(val)
    print(f"Train: {len(train):,}, Val: {len(val):,}")

    print("\nTraining with BALANCED hyperparameters...")
    results = train_with_calibration(train, val, BALANCED_PARAMS)

    print("\n" + "=" * 70)
    print("RESULTS: BALANCED MODEL")
    print("=" * 70)
    print(f"{'Aggregation Level':<25} {'WFA':>10} {'WMAPE':>10}")
    print("-" * 45)
    for level in ["daily_sku_store", "weekly_sku_store", "weekly_store", "weekly_total"]:
        if level in results:
            print(f"{level:<25} {results[level]['wfa']:>9.2f}% {results[level]['wmape']:>9.2f}%")

    with open(OUTPUT_DIR / "balanced_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR / 'balanced_results.json'}")
    return results


if __name__ == "__main__":
    main()
