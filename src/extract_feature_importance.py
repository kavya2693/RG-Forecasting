"""
Extract Feature Importance per Tier and Segment
================================================
Trains LightGBM regressors for each tier (T1, T2) and segment (A, B, C)
and extracts feature importance (gain-based).

Output: /tmp/feature_importance/importance_by_tier_segment.json
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/tmp/feature_importance")
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature columns (baseline)
FEATURE_COLS = [
    "sku_id", "store_id", "dow", "is_weekend", "day_of_year", "week_of_year",
    "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_56",
    "roll_mean_7", "roll_mean_28", "roll_sum_7", "roll_sum_28", "roll_std_28",
    "nz_rate_7", "nz_rate_28", "roll_mean_pos_28",
    "days_from_prev_closure", "days_to_next_closure",
    "last_sale_qty_asof", "days_since_last_sale",
    "sku_first_sale_date", "sku_age_days", "is_new_sku",
    "store_first_date"
]

# Hyperparameters per segment (from pipeline_params.yaml)
SEGMENT_PARAMS = {
    "A": {"num_leaves": 255, "min_child_samples": 50, "learning_rate": 0.05, "n_estimators": 500},
    "B": {"num_leaves": 127, "min_child_samples": 100, "learning_rate": 0.05, "n_estimators": 400},
    "C": {"num_leaves": 63, "min_child_samples": 200, "learning_rate": 0.03, "n_estimators": 300},
}


def load_data(tier: str, max_rows: int = 500000):
    """Load training data for a specific tier."""
    if tier == "T1":
        # Use c1_data for T1 (largest, most mature)
        train_path = "/tmp/c1_data/train_final.csv"
        val_path = "/tmp/c1_data/val_final.csv"
    elif tier == "T2":
        # Use g1_data for T2 (growing)
        train_path = "/tmp/c1_data/g1_train_with_header.csv"
        val_path = "/tmp/c1_data/g1_val_with_header.csv"
    else:
        return None, None

    if not os.path.exists(train_path):
        print(f"  Data not found for {tier}: {train_path}")
        return None, None

    print(f"  Loading {tier} data from {train_path}...")
    train = pd.read_csv(train_path, nrows=max_rows)
    val = pd.read_csv(val_path, nrows=max_rows // 2) if os.path.exists(val_path) else None

    return train, val


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ABC segments based on sales volume."""
    # Target column is 'y' (not 'qty')
    target_col = "y" if "y" in df.columns else "qty"

    # Calculate total sales per SKU
    sku_sales = df.groupby("sku_id")[target_col].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = sku_sales.sum()

    # A = top 80%, B = next 15%, C = bottom 5%
    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)
    c_skus = set(sku_sales[cumsum > total * 0.95].index)

    def get_segment(sku):
        if sku in a_skus:
            return "A"
        elif sku in b_skus:
            return "B"
        else:
            return "C"

    df["segment"] = df["sku_id"].apply(get_segment)
    return df


def train_regressor(df: pd.DataFrame, segment: str) -> dict:
    """Train LightGBM regressor and extract feature importance."""
    # Target column is 'y' (not 'qty')
    target_col = "y" if "y" in df.columns else "qty"

    # Filter to segment
    seg_df = df[df["segment"] == segment].copy()

    if len(seg_df) < 1000:
        print(f"    Segment {segment}: Too few rows ({len(seg_df)}), skipping")
        return {}

    # Only positive sales for regressor
    pos_df = seg_df[seg_df[target_col] > 0].copy()

    if len(pos_df) < 500:
        print(f"    Segment {segment}: Too few positive rows ({len(pos_df)}), skipping")
        return {}

    # Prepare features
    available_features = [f for f in FEATURE_COLS if f in pos_df.columns]
    X = pos_df[available_features].copy()
    y = np.log1p(pos_df[target_col])  # Log-transform target

    # Fill NaN
    X = X.fillna(0)

    # Get hyperparameters
    params = SEGMENT_PARAMS.get(segment, SEGMENT_PARAMS["C"])

    print(f"    Segment {segment}: Training on {len(X)} positive rows, {len(available_features)} features")

    # Train model
    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        verbose=-1,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X, y)

    # Extract feature importance
    importance = dict(zip(available_features, model.feature_importances_.tolist()))

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    # Normalize to percentage
    total = sum(importance.values())
    if total > 0:
        importance = {k: round(v / total * 100, 2) for k, v in importance.items()}

    return importance


def main():
    print("=" * 70)
    print("FEATURE IMPORTANCE EXTRACTION PER TIER AND SEGMENT")
    print("=" * 70)

    results = {}

    for tier in ["T1", "T2"]:
        print(f"\n{'='*50}")
        print(f"Processing {tier}...")
        print("=" * 50)

        train, val = load_data(tier)

        if train is None:
            print(f"  Skipping {tier} - no data")
            continue

        # Assign segments
        train = assign_segments(train)

        seg_counts = train["segment"].value_counts()
        print(f"  Segment distribution: A={seg_counts.get('A', 0)}, B={seg_counts.get('B', 0)}, C={seg_counts.get('C', 0)}")

        results[tier] = {}

        for segment in ["A", "B", "C"]:
            print(f"\n  Training {tier}/{segment}...")
            importance = train_regressor(train, segment)

            if importance:
                results[tier][segment] = importance
                top_5 = list(importance.items())[:5]
                print(f"    Top 5 features: {top_5}")

    # Save results
    output_path = OUTPUT_DIR / "importance_by_tier_segment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved to {output_path}")
    print("=" * 70)

    # Also create a summary for the UI
    summary = {
        "description": "Feature importance (gain %) from LightGBM regressors",
        "tiers": list(results.keys()),
        "segments": ["A", "B", "C"],
        "data": results,
        "top_features_overall": get_top_features_overall(results),
    }

    summary_path = OUTPUT_DIR / "importance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")

    return results


def get_top_features_overall(results: dict) -> list:
    """Get top 10 features across all tiers and segments."""
    feature_scores = {}

    for tier, segments in results.items():
        for segment, importance in segments.items():
            for feature, score in importance.items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(score)

    # Average score per feature
    avg_scores = {f: np.mean(scores) for f, scores in feature_scores.items()}

    # Sort and return top 10
    top = sorted(avg_scores.items(), key=lambda x: -x[1])[:10]
    return [{"feature": f, "avg_importance": round(s, 2)} for f, s in top]


if __name__ == "__main__":
    main()
