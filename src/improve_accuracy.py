"""
Accuracy Improvement Experiments
================================
Systematically test approaches to improve model accuracy:
1. Threshold optimization per segment
2. Feature selection based on importance
3. Bias correction calibration
4. Hyperparameter tuning
5. Ensemble (classifier + regressor weighting)

Output: /tmp/accuracy_experiments/results.json
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/tmp/accuracy_experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature columns
FEATURE_COLS = [
    "sku_id", "store_id", "dow", "is_weekend", "day_of_year", "week_of_year",
    "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_56",
    "roll_mean_7", "roll_mean_28", "roll_sum_7", "roll_sum_28", "roll_std_28",
    "nz_rate_7", "nz_rate_28", "roll_mean_pos_28",
    "days_from_prev_closure", "days_to_next_closure",
    "last_sale_qty_asof", "days_since_last_sale",
]

CAT_FEATURES = ["sku_id", "store_id"]

# Current hyperparameters
SEGMENT_PARAMS = {
    "A": {"num_leaves": 255, "min_child_samples": 10, "learning_rate": 0.015, "n_estimators": 1000},
    "B": {"num_leaves": 63, "min_child_samples": 50, "learning_rate": 0.03, "n_estimators": 400},
    "C": {"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05, "n_estimators": 300},
}


def load_data():
    """Load training and validation data."""
    train = pd.read_csv("/tmp/c1_data/train_final.csv", nrows=500000)
    val = pd.read_csv("/tmp/c1_data/val_final.csv", nrows=250000)
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


def compute_wmape(y_true, y_pred):
    """Compute WMAPE."""
    return 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)


def compute_wfa(y_true, y_pred):
    """Compute WFA (100 - WMAPE)."""
    return 100 - compute_wmape(y_true, y_pred)


def train_baseline(train, val, features, segment):
    """Train baseline two-stage model for a segment."""
    params = SEGMENT_PARAMS[segment]

    train_seg = train[train["segment"] == segment].copy()
    val_seg = val[val["segment"] == segment].copy()

    if len(train_seg) < 100 or len(val_seg) < 100:
        return None, None, None, None

    target = "y" if "y" in train_seg.columns else "qty"

    # Prepare features (exclude categorical from fillna)
    numeric_features = [f for f in features if f in train_seg.columns and f not in CAT_FEATURES]
    cat_features_present = [f for f in CAT_FEATURES if f in train_seg.columns]

    for col in cat_features_present:
        train_seg[col] = train_seg[col].astype("category")
        val_seg[col] = val_seg[col].astype("category")

    # Fill numeric features only
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

    # Stage 2: Regressor on non-zero
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

    return prob, pred_value, y_val, val_seg


def experiment_threshold_optimization(prob, pred_value, y_val, segment):
    """Find optimal threshold for each segment."""
    best_wfa = 0
    best_threshold = 0.5

    for threshold in np.arange(0.3, 0.9, 0.05):
        y_pred = np.where(prob > threshold, pred_value, 0)
        y_pred = np.maximum(0, y_pred)
        wfa = compute_wfa(y_val, y_pred)
        if wfa > best_wfa:
            best_wfa = wfa
            best_threshold = threshold

    return best_threshold, best_wfa


def experiment_bias_correction(prob, pred_value, y_val, threshold):
    """Find optimal bias correction factor."""
    y_pred_base = np.where(prob > threshold, pred_value, 0)
    y_pred_base = np.maximum(0, y_pred_base)

    # Calculate bias
    mask = y_pred_base > 0.1
    if np.sum(y_pred_base[mask]) > 0:
        actual_sum = np.sum(y_val[mask])
        pred_sum = np.sum(y_pred_base[mask])
        raw_k = actual_sum / pred_sum

        # Test range of k values
        best_wfa = 0
        best_k = 1.0

        for k in np.arange(0.8, 1.5, 0.02):
            y_pred = y_pred_base * k
            wfa = compute_wfa(y_val, y_pred)
            if wfa > best_wfa:
                best_wfa = wfa
                best_k = k

        return best_k, best_wfa, raw_k

    return 1.0, compute_wfa(y_val, y_pred_base), 1.0


def experiment_feature_selection(train, val, segment, top_n=15):
    """Train with only top N features."""
    # Load feature importance
    try:
        with open("/tmp/feature_importance/importance_by_tier_segment.json", "r") as f:
            fi = json.load(f)

        # Get top features for this segment (from T1)
        if "T1" in fi and segment in fi["T1"]:
            top_features = list(fi["T1"][segment].keys())[:top_n]
        else:
            return None, None

        # Ensure categorical features are included
        features = top_features + [f for f in CAT_FEATURES if f not in top_features]

        prob, pred_value, y_val, _ = train_baseline(train, val, features, segment)
        if prob is None:
            return None, None

        # Use default threshold
        threshold = 0.6 if segment in ["A", "B"] else 0.7
        y_pred = np.where(prob > threshold, pred_value, 0)
        y_pred = np.maximum(0, y_pred)

        return compute_wfa(y_val, y_pred), features
    except:
        return None, None


def experiment_hyperparameter_search(train, val, segment):
    """Quick hyperparameter search."""
    target = "y" if "y" in train.columns else "qty"

    train_seg = train[train["segment"] == segment].copy()
    val_seg = val[val["segment"] == segment].copy()

    if len(train_seg) < 100:
        return None, None

    # Prepare features (exclude categorical from fillna)
    numeric_features = [f for f in FEATURE_COLS if f in train_seg.columns and f not in CAT_FEATURES]
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

    best_wfa = 0
    best_params = None

    # Search space
    num_leaves_options = [31, 63, 127, 255] if segment == "A" else [15, 31, 63, 127]
    min_child_options = [5, 10, 20, 50] if segment == "A" else [20, 50, 100, 200]

    for num_leaves in num_leaves_options:
        for min_child in min_child_options:
            try:
                # Quick train
                clf = lgb.LGBMClassifier(
                    num_leaves=num_leaves, min_child_samples=min_child,
                    learning_rate=0.05, n_estimators=200,
                    verbose=-1, n_jobs=-1, random_state=42,
                )
                clf.fit(X_train, (y_train > 0).astype(int))
                prob = clf.predict_proba(X_val)[:, 1]

                train_nz = train_seg[train_seg[target] > 0].copy()
                X_train_nz = train_nz[available]
                y_train_nz = np.log1p(train_nz[target].values)

                reg = lgb.LGBMRegressor(
                    num_leaves=num_leaves, min_child_samples=max(5, min_child // 2),
                    learning_rate=0.05, n_estimators=200,
                    verbose=-1, n_jobs=-1, random_state=42,
                )
                reg.fit(X_train_nz, y_train_nz)
                pred_value = np.expm1(reg.predict(X_val))

                threshold = 0.6 if segment in ["A", "B"] else 0.7
                y_pred = np.where(prob > threshold, pred_value, 0)
                y_pred = np.maximum(0, y_pred)

                wfa = compute_wfa(y_val, y_pred)
                if wfa > best_wfa:
                    best_wfa = wfa
                    best_params = {"num_leaves": num_leaves, "min_child_samples": min_child}
            except:
                continue

    return best_wfa, best_params


def main():
    print("=" * 70)
    print("ACCURACY IMPROVEMENT EXPERIMENTS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train, val = load_data()
    train = assign_segments(train)
    val = assign_segments(val)

    target = "y" if "y" in train.columns else "qty"
    print(f"  Train: {len(train):,}, Val: {len(val):,}")

    results = {"experiments": {}, "best_config": {}}

    for segment in ["A", "B", "C"]:
        print(f"\n{'='*50}")
        print(f"SEGMENT {segment}")
        print("=" * 50)

        seg_results = {}

        # 1. Baseline
        print(f"\n[1] Training baseline...")
        prob, pred_value, y_val, val_seg = train_baseline(train, val, FEATURE_COLS, segment)

        if prob is None:
            print(f"  Skipping {segment} - insufficient data")
            continue

        default_threshold = 0.6 if segment in ["A", "B"] else 0.7
        y_pred_base = np.where(prob > default_threshold, pred_value, 0)
        y_pred_base = np.maximum(0, y_pred_base)
        baseline_wfa = compute_wfa(y_val, y_pred_base)
        print(f"  Baseline WFA: {baseline_wfa:.2f}%")
        seg_results["baseline"] = {"wfa": baseline_wfa, "threshold": default_threshold}

        # 2. Threshold optimization
        print(f"\n[2] Optimizing threshold...")
        best_threshold, threshold_wfa = experiment_threshold_optimization(prob, pred_value, y_val, segment)
        improvement = threshold_wfa - baseline_wfa
        print(f"  Optimal threshold: {best_threshold:.2f} → WFA: {threshold_wfa:.2f}% ({improvement:+.2f}pp)")
        seg_results["threshold_opt"] = {"threshold": best_threshold, "wfa": threshold_wfa, "improvement": improvement}

        # 3. Bias correction
        print(f"\n[3] Optimizing bias correction...")
        best_k, bias_wfa, raw_k = experiment_bias_correction(prob, pred_value, y_val, best_threshold)
        improvement = bias_wfa - threshold_wfa
        print(f"  Raw bias: {raw_k:.3f}, Optimal k: {best_k:.3f} → WFA: {bias_wfa:.2f}% ({improvement:+.2f}pp)")
        seg_results["bias_correction"] = {"k": best_k, "raw_k": raw_k, "wfa": bias_wfa, "improvement": improvement}

        # 4. Feature selection
        print(f"\n[4] Testing feature selection (top 15)...")
        fs_wfa, top_features = experiment_feature_selection(train, val, segment, top_n=15)
        if fs_wfa:
            improvement = fs_wfa - baseline_wfa
            print(f"  Feature selection WFA: {fs_wfa:.2f}% ({improvement:+.2f}pp)")
            seg_results["feature_selection"] = {"wfa": fs_wfa, "n_features": len(top_features) if top_features else 0, "improvement": improvement}

        # 5. Hyperparameter search
        print(f"\n[5] Quick hyperparameter search...")
        hp_wfa, best_hp = experiment_hyperparameter_search(train, val, segment)
        if hp_wfa:
            improvement = hp_wfa - baseline_wfa
            print(f"  Best HP: {best_hp} → WFA: {hp_wfa:.2f}% ({improvement:+.2f}pp)")
            seg_results["hyperparameter_search"] = {"params": best_hp, "wfa": hp_wfa, "improvement": improvement}

        # Find best configuration
        all_wfas = [
            ("baseline", baseline_wfa),
            ("threshold_opt", threshold_wfa),
            ("bias_correction", bias_wfa),
        ]
        if fs_wfa:
            all_wfas.append(("feature_selection", fs_wfa))
        if hp_wfa:
            all_wfas.append(("hyperparameter_search", hp_wfa))

        best_exp, best_wfa = max(all_wfas, key=lambda x: x[1])
        total_improvement = best_wfa - baseline_wfa

        print(f"\n  ★ Best for {segment}: {best_exp} → WFA: {best_wfa:.2f}% ({total_improvement:+.2f}pp vs baseline)")

        results["experiments"][segment] = seg_results
        results["best_config"][segment] = {
            "best_experiment": best_exp,
            "wfa": best_wfa,
            "improvement_vs_baseline": total_improvement,
            "optimal_threshold": best_threshold,
            "optimal_k": best_k,
        }

    # Combined results
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    # Apply best config to all segments and compute overall metrics
    combined_y_true = []
    combined_y_pred = []

    for segment in ["A", "B", "C"]:
        if segment not in results["best_config"]:
            continue

        config = results["best_config"][segment]
        prob, pred_value, y_val, val_seg = train_baseline(train, val, FEATURE_COLS, segment)

        if prob is None:
            continue

        threshold = config["optimal_threshold"]
        k = config["optimal_k"]

        y_pred = np.where(prob > threshold, pred_value * k, 0)
        y_pred = np.maximum(0, y_pred)

        combined_y_true.extend(y_val)
        combined_y_pred.extend(y_pred)

    combined_y_true = np.array(combined_y_true)
    combined_y_pred = np.array(combined_y_pred)

    overall_wfa = compute_wfa(combined_y_true, combined_y_pred)
    print(f"\nOverall Daily SKU-Store WFA: {overall_wfa:.2f}%")

    results["overall"] = {"daily_sku_store_wfa": overall_wfa}

    # Save results
    output_path = OUTPUT_DIR / "improvement_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
