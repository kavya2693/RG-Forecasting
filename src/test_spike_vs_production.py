"""
Proper Test: Full Production Baseline vs Production + Spike Features
====================================================================
Compare the ACTUAL production feature set against production + spike features
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

OUTPUT_DIR = Path("/tmp/spike_analysis")

# FULL production feature set (matching the actual pipeline)
PRODUCTION_FEATURES = [
    "dow", "is_weekend", "day_of_year", "week_of_year",
    "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_sum_7", "roll_sum_28", "roll_std_28",
    "nz_rate_7", "nz_rate_28", "roll_mean_pos_28",
    "days_from_prev_closure", "days_to_next_closure",
    "last_sale_qty_asof", "days_since_last_sale",
    "dormancy_capped",
]

CAT_FEATURES = ["sku_id", "store_id"]

# Production hyperparameters
SEGMENT_PARAMS = {
    "A": {"num_leaves": 255, "min_child_samples": 10, "learning_rate": 0.015, "n_estimators": 800},
    "B": {"num_leaves": 63, "min_child_samples": 50, "learning_rate": 0.03, "n_estimators": 400},
    "C": {"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05, "n_estimators": 300},
}


def load_data():
    train = pd.read_csv("/tmp/c1_data/train_final.csv")
    val = pd.read_csv("/tmp/c1_data/val_final.csv")

    for df in [train, val]:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].dt.isocalendar().week.astype(int)

    return train, val


def add_spike_features(df):
    """Add spike detection and features."""
    target = 'y' if 'y' in df.columns else 'qty'
    df = df.copy()

    # Detect spikes
    df['_baseline'] = df['roll_mean_28'].fillna(0) + 0.1
    df['_spike_ratio'] = df[target] / df['_baseline']
    df['is_spike'] = (df['_spike_ratio'] > 2.0) & (df[target] > 1)

    # Store-wide detection
    store_day = df.groupby(['store_id', 'date']).agg({
        'is_spike': 'sum', 'sku_id': 'nunique'
    }).reset_index()
    store_day.columns = ['store_id', 'date', 'spike_count', 'sku_count']
    store_day['spike_pct'] = store_day['spike_count'] / store_day['sku_count']
    store_day['is_store_wide'] = store_day['spike_pct'] > 0.15
    df = df.merge(store_day[['store_id', 'date', 'spike_pct', 'is_store_wide']], on=['store_id', 'date'], how='left')

    # Seasonal lift
    week_avg = df.groupby(['store_id', 'sku_id', 'week'])[target].mean().reset_index()
    week_avg.columns = ['store_id', 'sku_id', 'week', 'week_avg']
    overall_avg = df.groupby(['store_id', 'sku_id'])[target].mean().reset_index()
    overall_avg.columns = ['store_id', 'sku_id', 'overall_avg']
    week_avg = week_avg.merge(overall_avg, on=['store_id', 'sku_id'])
    week_avg['week_lift'] = week_avg['week_avg'] / (week_avg['overall_avg'] + 0.1)
    df = df.merge(week_avg[['store_id', 'sku_id', 'week', 'week_lift']], on=['store_id', 'sku_id', 'week'], how='left')

    # Create features
    df['feat_store_promo_day'] = df['is_store_wide'].fillna(False).astype(int)
    df['feat_seasonal_lift'] = df['week_lift'].fillna(1.0).clip(0.5, 3.0)
    df['feat_store_spike_pct'] = df['spike_pct'].fillna(0).clip(0, 1)

    df = df.sort_values(['store_id', 'sku_id', 'date'])
    df['feat_had_recent_spike'] = df.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()
    ).fillna(0).astype(int)

    spike_prob = df.groupby(['store_id', 'sku_id', 'week'])['is_spike'].mean().reset_index()
    spike_prob.columns = ['store_id', 'sku_id', 'week', 'feat_historical_spike_prob']
    df = df.merge(spike_prob, on=['store_id', 'sku_id', 'week'], how='left')
    df['feat_historical_spike_prob'] = df['feat_historical_spike_prob'].fillna(0)

    new_features = [
        'feat_store_promo_day', 'feat_seasonal_lift', 'feat_store_spike_pct',
        'feat_had_recent_spike', 'feat_historical_spike_prob'
    ]

    return df, new_features


def assign_segments(df):
    target = 'y' if 'y' in df.columns else 'qty'
    sku_sales = df.groupby('sku_id')[target].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = sku_sales.sum()
    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)
    df['segment'] = df['sku_id'].apply(lambda x: 'A' if x in a_skus else ('B' if x in b_skus else 'C'))
    return df


def compute_metrics(y_true, y_pred, val_df):
    target = 'y' if 'y' in val_df.columns else 'qty'
    results = {}

    # Daily
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    results['daily_sku_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Weekly store
    val_df = val_df.copy()
    val_df['y_pred'] = y_pred
    weekly_store = val_df.groupby(['store_id', 'year', 'week'], observed=True).agg(
        {target: 'sum', 'y_pred': 'sum'}
    ).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_store[target] - weekly_store['y_pred'])) / max(np.sum(weekly_store[target]), 1)
    results['weekly_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    return results


def train_segment(train, val, features, segment, params):
    """Train production-style two-stage model."""
    target = 'y' if 'y' in train.columns else 'qty'

    train_seg = train[train['segment'] == segment].copy()
    val_seg = val[val['segment'] == segment].copy()

    if len(train_seg) < 100 or len(val_seg) < 100:
        return None, None

    # Prepare
    available = [f for f in features if f in train_seg.columns]
    for col in CAT_FEATURES:
        train_seg[col] = train_seg[col].astype('category')
        val_seg[col] = val_seg[col].astype('category')

    for col in available:
        train_seg[col] = train_seg[col].fillna(0)
        val_seg[col] = val_seg[col].fillna(0)

    all_features = available + CAT_FEATURES
    X_train = train_seg[all_features]
    X_val = val_seg[all_features]
    y_train = train_seg[target].values
    y_val = val_seg[target].values

    # Classifier
    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'],
        n_estimators=min(params['n_estimators'], 500),
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))
    prob = clf.predict_proba(X_val)[:, 1]

    # Regressor
    train_nz = train_seg[train_seg[target] > 0].copy()
    X_train_nz = train_nz[all_features]

    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'],
        min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'],
        n_estimators=min(params['n_estimators'], 500),
        reg_lambda=0.5,
        verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(X_train_nz, np.log1p(train_nz[target]))
    pred = np.expm1(reg.predict(X_val))

    # Combine
    threshold = 0.6 if segment in ['A', 'B'] else 0.7
    y_pred = np.where(prob > threshold, pred, 0)
    y_pred = np.maximum(0, y_pred)

    return y_pred, val_seg


def main():
    print("=" * 70)
    print("PROPER TEST: PRODUCTION BASELINE vs PRODUCTION + SPIKE FEATURES")
    print("=" * 70)

    # Load
    print("\nLoading data...")
    train, val = load_data()
    print(f"  Train: {len(train):,}, Val: {len(val):,}")

    # Segments
    train = assign_segments(train)
    val = assign_segments(val)

    # Add spike features
    print("\nAdding spike features...")
    train, spike_features = add_spike_features(train)
    val, _ = add_spike_features(val)

    print(f"\nProduction features: {len(PRODUCTION_FEATURES)}")
    print(f"Spike features added: {len(spike_features)}")

    results = {}

    for segment in ['A', 'B', 'C']:
        print(f"\n{'='*50}")
        print(f"SEGMENT {segment}")
        print("=" * 50)

        params = SEGMENT_PARAMS[segment]

        # Test 1: Production baseline
        print("\n  [1] Production baseline...")
        y_pred_base, val_seg = train_segment(train, val, PRODUCTION_FEATURES, segment, params)
        if y_pred_base is None:
            continue

        target = 'y' if 'y' in val_seg.columns else 'qty'
        y_val = val_seg[target].values
        metrics_base = compute_metrics(y_val, y_pred_base, val_seg)
        print(f"      Daily WFA: {metrics_base['daily_sku_store']['wfa']:.2f}%")
        print(f"      Weekly Store WFA: {metrics_base['weekly_store']['wfa']:.2f}%")

        # Test 2: Production + spike features
        print("\n  [2] Production + spike features...")
        all_features = PRODUCTION_FEATURES + spike_features
        y_pred_spike, val_seg = train_segment(train, val, all_features, segment, params)

        metrics_spike = compute_metrics(y_val, y_pred_spike, val_seg)
        print(f"      Daily WFA: {metrics_spike['daily_sku_store']['wfa']:.2f}%")
        print(f"      Weekly Store WFA: {metrics_spike['weekly_store']['wfa']:.2f}%")

        # Compare
        daily_change = metrics_spike['daily_sku_store']['wfa'] - metrics_base['daily_sku_store']['wfa']
        weekly_change = metrics_spike['weekly_store']['wfa'] - metrics_base['weekly_store']['wfa']

        print(f"\n  CHANGE: Daily {daily_change:+.2f}pp, Weekly Store {weekly_change:+.2f}pp")

        results[segment] = {
            'baseline': metrics_base,
            'with_spikes': metrics_spike,
            'daily_change_pp': daily_change,
            'weekly_change_pp': weekly_change
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ACTUAL IMPROVEMENT FROM SPIKE FEATURES")
    print("=" * 70)
    print(f"\n{'Segment':<10} {'Daily WFA Base':>15} {'Daily +Spikes':>15} {'Change':>10}")
    print("-" * 50)
    for seg in ['A', 'B', 'C']:
        if seg in results:
            base = results[seg]['baseline']['daily_sku_store']['wfa']
            spike = results[seg]['with_spikes']['daily_sku_store']['wfa']
            change = results[seg]['daily_change_pp']
            print(f"{seg:<10} {base:>14.2f}% {spike:>14.2f}% {change:>+9.2f}pp")

    with open(OUTPUT_DIR / 'production_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'production_comparison.json'}")


if __name__ == "__main__":
    main()
