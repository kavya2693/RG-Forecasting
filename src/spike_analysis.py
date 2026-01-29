"""
Spike Analysis and Classification
=================================
1. Detect all spikes in the data
2. Classify each spike by type:
   - STORE_WIDE: Many SKUs in same store spike together
   - SEASONAL: Spike occurs at same time each year
   - DOW_PATTERN: Spike aligns with day-of-week patterns
   - ISOLATED: Single SKU spike (could be item promo or random)
3. Create features based on classification
4. Test if these improve model accuracy

Output: /tmp/spike_analysis/
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict
from datetime import datetime

OUTPUT_DIR = Path("/tmp/spike_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1: LOAD DATA AND DETECT SPIKES
# ---------------------------------------------------------------------------

def load_data():
    """Load data with dates parsed."""
    train = pd.read_csv("/tmp/c1_data/train_final.csv")
    val = pd.read_csv("/tmp/c1_data/val_final.csv")

    for df in [train, val]:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['day_of_year'] = df['date'].dt.dayofyear

    return train, val


def detect_spikes(df, threshold=2.0):
    """
    Detect spikes where actual sales > threshold × rolling mean.
    Returns DataFrame with spike information.
    """
    target = 'y' if 'y' in df.columns else 'qty'

    # Calculate spike ratio
    df = df.copy()
    df['baseline'] = df['roll_mean_28'].fillna(0) + 0.1  # Avoid division by zero
    df['spike_ratio'] = df[target] / df['baseline']

    # Flag spikes
    df['is_spike'] = (df['spike_ratio'] > threshold) & (df[target] > 1)

    # Get spike magnitude
    df['spike_magnitude'] = np.where(df['is_spike'], df[target] - df['baseline'], 0)

    return df


# ---------------------------------------------------------------------------
# STEP 2: CLASSIFY SPIKES
# ---------------------------------------------------------------------------

def classify_spikes(df):
    """
    Classify each spike into categories:
    - STORE_WIDE: >20% of SKUs in store spike on same day
    - SEASONAL: Spike occurs in similar week across years
    - DOW_ALIGNED: Spike on a high-traffic day (weekend)
    - ISOLATED: Individual item spike
    """
    target = 'y' if 'y' in df.columns else 'qty'

    print("Classifying spikes...")

    # --- Store-wide spike detection ---
    print("  Detecting store-wide spikes...")
    store_day_stats = df.groupby(['store_id', 'date']).agg({
        'is_spike': ['sum', 'count'],
        'sku_id': 'nunique'
    }).reset_index()
    store_day_stats.columns = ['store_id', 'date', 'spike_count', 'total_count', 'sku_count']
    store_day_stats['spike_pct'] = store_day_stats['spike_count'] / store_day_stats['sku_count']
    store_day_stats['is_store_wide'] = store_day_stats['spike_pct'] > 0.15  # 15% of SKUs spike

    # Merge back
    df = df.merge(
        store_day_stats[['store_id', 'date', 'spike_pct', 'is_store_wide']],
        on=['store_id', 'date'],
        how='left'
    )

    # --- Seasonal spike detection ---
    print("  Detecting seasonal spikes...")
    # Check if this week historically has high sales
    week_avg = df.groupby(['store_id', 'sku_id', 'week'])[target].mean().reset_index()
    week_avg.columns = ['store_id', 'sku_id', 'week', 'week_avg']
    overall_avg = df.groupby(['store_id', 'sku_id'])[target].mean().reset_index()
    overall_avg.columns = ['store_id', 'sku_id', 'overall_avg']

    week_avg = week_avg.merge(overall_avg, on=['store_id', 'sku_id'])
    week_avg['week_lift'] = week_avg['week_avg'] / (week_avg['overall_avg'] + 0.1)
    week_avg['is_high_season_week'] = week_avg['week_lift'] > 1.3  # 30% above average

    df = df.merge(
        week_avg[['store_id', 'sku_id', 'week', 'week_lift', 'is_high_season_week']],
        on=['store_id', 'sku_id', 'week'],
        how='left'
    )

    # --- Day-of-week aligned spike ---
    print("  Detecting DOW-aligned spikes...")
    df['is_dow_aligned'] = df['is_spike'] & (df['is_weekend'] == 1)

    # --- Classify each spike ---
    print("  Final classification...")
    conditions = [
        df['is_spike'] & df['is_store_wide'],
        df['is_spike'] & df['is_high_season_week'] & ~df['is_store_wide'],
        df['is_spike'] & df['is_dow_aligned'] & ~df['is_store_wide'] & ~df['is_high_season_week'],
        df['is_spike'] & ~df['is_store_wide'] & ~df['is_high_season_week'] & ~df['is_dow_aligned'],
    ]
    choices = ['STORE_PROMO', 'SEASONAL', 'DOW_PATTERN', 'ISOLATED']
    df['spike_type'] = np.select(conditions, choices, default='NONE')

    return df


# ---------------------------------------------------------------------------
# STEP 3: ANALYZE SPIKE PATTERNS
# ---------------------------------------------------------------------------

def analyze_spike_patterns(df):
    """Generate statistics about spike patterns."""
    target = 'y' if 'y' in df.columns else 'qty'

    print("\n" + "=" * 60)
    print("SPIKE ANALYSIS RESULTS")
    print("=" * 60)

    total_rows = len(df)
    spike_rows = df['is_spike'].sum()

    print(f"\nTotal observations: {total_rows:,}")
    print(f"Spike observations: {spike_rows:,} ({100*spike_rows/total_rows:.2f}%)")

    # Breakdown by type
    print("\nSpike Classification Breakdown:")
    print("-" * 40)
    type_counts = df[df['is_spike']]['spike_type'].value_counts()
    for spike_type, count in type_counts.items():
        pct = 100 * count / spike_rows
        avg_magnitude = df[df['spike_type'] == spike_type]['spike_magnitude'].mean()
        print(f"  {spike_type:<15}: {count:>8,} ({pct:>5.1f}%) | Avg magnitude: {avg_magnitude:.1f}")

    # Store-wide events
    store_wide_days = df[df['is_store_wide']].groupby(['store_id', 'date']).size().reset_index()
    print(f"\nStore-wide spike events detected: {len(store_wide_days):,}")

    # Seasonal patterns
    high_season_weeks = df[df['is_high_season_week']]['week'].unique()
    print(f"High-season weeks identified: {sorted(high_season_weeks)[:10]}...")

    # Save detailed analysis
    analysis = {
        'total_observations': int(total_rows),
        'spike_observations': int(spike_rows),
        'spike_rate': float(spike_rows / total_rows),
        'breakdown': {k: int(v) for k, v in type_counts.items()},
        'store_wide_events': int(len(store_wide_days)),
        'high_season_weeks': [int(w) for w in sorted(high_season_weeks)],
    }

    return analysis


# ---------------------------------------------------------------------------
# STEP 4: CREATE SPIKE-BASED FEATURES
# ---------------------------------------------------------------------------

def create_spike_features(df):
    """Create features based on spike classification."""
    target = 'y' if 'y' in df.columns else 'qty'

    print("\nCreating spike-based features...")

    # Feature 1: Is this a store-wide promo day?
    df['feat_store_promo_day'] = df['is_store_wide'].astype(int)

    # Feature 2: Seasonal lift for this week
    df['feat_seasonal_lift'] = df['week_lift'].fillna(1.0)

    # Feature 3: Recent spike indicator (was there a spike in last 7 days for this series?)
    df = df.sort_values(['store_id', 'sku_id', 'date'])
    df['feat_had_recent_spike'] = df.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()
    ).fillna(0).astype(int)

    # Feature 4: Store spike intensity (what % of store is spiking)
    df['feat_store_spike_pct'] = df['spike_pct'].fillna(0)

    # Feature 5: Post-spike indicator (7 days after a spike - expect dip)
    df['feat_post_spike'] = df.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(7)
    ).fillna(0).astype(int)

    # Feature 6: Spike type encodings
    df['feat_is_seasonal_period'] = df['is_high_season_week'].astype(int)

    # Feature 7: Expected spike probability based on historical patterns
    spike_prob = df.groupby(['store_id', 'sku_id', 'week'])['is_spike'].mean().reset_index()
    spike_prob.columns = ['store_id', 'sku_id', 'week', 'feat_historical_spike_prob']
    df = df.merge(spike_prob, on=['store_id', 'sku_id', 'week'], how='left')
    df['feat_historical_spike_prob'] = df['feat_historical_spike_prob'].fillna(0)

    new_features = [
        'feat_store_promo_day',
        'feat_seasonal_lift',
        'feat_had_recent_spike',
        'feat_store_spike_pct',
        'feat_post_spike',
        'feat_is_seasonal_period',
        'feat_historical_spike_prob',
    ]

    print(f"  Created {len(new_features)} new features")

    return df, new_features


# ---------------------------------------------------------------------------
# STEP 5: TEST ON A-SEGMENT
# ---------------------------------------------------------------------------

def assign_segments(df):
    """Assign ABC segments."""
    target = 'y' if 'y' in df.columns else 'qty'
    sku_sales = df.groupby('sku_id')[target].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = sku_sales.sum()

    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)

    df['segment'] = df['sku_id'].apply(lambda x: 'A' if x in a_skus else ('B' if x in b_skus else 'C'))
    return df


def test_model_improvement(train, val, new_features, segment='A'):
    """Test if spike features improve accuracy for a segment."""
    target = 'y' if 'y' in train.columns else 'qty'

    print(f"\n{'='*60}")
    print(f"TESTING ON {segment}-SEGMENT")
    print("=" * 60)

    # Filter to segment
    train_seg = train[train['segment'] == segment].copy()
    val_seg = val[val['segment'] == segment].copy()

    print(f"  Train: {len(train_seg):,}, Val: {len(val_seg):,}")

    # Base features
    base_features = [
        'dow', 'is_weekend', 'day_of_year',
        'sin_doy', 'cos_doy',
        'lag_7', 'lag_14',
        'roll_mean_28', 'nz_rate_28', 'roll_mean_pos_28',
    ]

    cat_features = ['sku_id', 'store_id']

    # Prepare data
    for col in cat_features:
        train_seg[col] = train_seg[col].astype('category')
        val_seg[col] = val_seg[col].astype('category')

    available_base = [f for f in base_features if f in train_seg.columns]
    available_new = [f for f in new_features if f in train_seg.columns]

    for col in available_base + available_new:
        train_seg[col] = train_seg[col].fillna(0)
        val_seg[col] = val_seg[col].fillna(0)

    y_train = train_seg[target].values
    y_val = val_seg[target].values

    results = {}

    # --- Test 1: Baseline (no spike features) ---
    print("\n  [1] Training BASELINE model...")
    X_train_base = train_seg[available_base + cat_features]
    X_val_base = val_seg[available_base + cat_features]

    clf_base = lgb.LGBMClassifier(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1)
    clf_base.fit(X_train_base, (y_train > 0).astype(int))
    prob_base = clf_base.predict_proba(X_val_base)[:, 1]

    train_nz = train_seg[train_seg[target] > 0]
    reg_base = lgb.LGBMRegressor(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1)
    reg_base.fit(train_nz[available_base + cat_features], np.log1p(train_nz[target]))
    pred_base = np.expm1(reg_base.predict(X_val_base))

    y_pred_base = np.where(prob_base > 0.6, pred_base, 0)
    y_pred_base = np.maximum(0, y_pred_base)

    wmape_base = 100 * np.sum(np.abs(y_val - y_pred_base)) / np.sum(y_val)
    wfa_base = 100 - wmape_base
    results['baseline'] = {'wfa': wfa_base, 'wmape': wmape_base}
    print(f"      Baseline WFA: {wfa_base:.2f}%")

    # --- Test 2: With spike features ---
    print("\n  [2] Training WITH SPIKE FEATURES...")
    all_features = available_base + available_new + cat_features
    X_train_spike = train_seg[all_features]
    X_val_spike = val_seg[all_features]

    clf_spike = lgb.LGBMClassifier(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1)
    clf_spike.fit(X_train_spike, (y_train > 0).astype(int))
    prob_spike = clf_spike.predict_proba(X_val_spike)[:, 1]

    reg_spike = lgb.LGBMRegressor(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1)
    reg_spike.fit(train_nz[all_features], np.log1p(train_nz[target]))
    pred_spike = np.expm1(reg_spike.predict(X_val_spike))

    y_pred_spike = np.where(prob_spike > 0.6, pred_spike, 0)
    y_pred_spike = np.maximum(0, y_pred_spike)

    wmape_spike = 100 * np.sum(np.abs(y_val - y_pred_spike)) / np.sum(y_val)
    wfa_spike = 100 - wmape_spike
    results['with_spike_features'] = {'wfa': wfa_spike, 'wmape': wmape_spike}
    print(f"      With Spike Features WFA: {wfa_spike:.2f}%")

    # --- Feature importance for new features ---
    print("\n  Feature Importance (new spike features):")
    importance = dict(zip(all_features, reg_spike.feature_importances_))
    for feat in available_new:
        print(f"      {feat}: {importance.get(feat, 0):.0f}")

    # --- Summary ---
    improvement = wfa_spike - wfa_base
    print(f"\n  {'='*40}")
    print(f"  RESULT: {improvement:+.2f}pp change")
    if improvement > 0.1:
        print(f"  ✓ Spike features IMPROVED accuracy!")
    elif improvement < -0.1:
        print(f"  ✗ Spike features HURT accuracy")
    else:
        print(f"  ~ No significant change")

    results['improvement'] = improvement
    results['new_features_used'] = available_new

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SPIKE ANALYSIS AND CLASSIFICATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Load data
    print("\n[1] Loading data...")
    train, val = load_data()
    print(f"    Train: {len(train):,}, Val: {len(val):,}")

    # Detect spikes
    print("\n[2] Detecting spikes (threshold=2.0)...")
    train = detect_spikes(train, threshold=2.0)
    val = detect_spikes(val, threshold=2.0)

    # Classify spikes
    print("\n[3] Classifying spikes...")
    train = classify_spikes(train)
    val = classify_spikes(val)

    # Analyze patterns
    print("\n[4] Analyzing patterns...")
    analysis = analyze_spike_patterns(train)

    # Create features
    print("\n[5] Creating spike-based features...")
    train, new_features = create_spike_features(train)
    val, _ = create_spike_features(val)

    # Assign segments
    train = assign_segments(train)
    val = assign_segments(val)

    # Test on A-segment
    print("\n[6] Testing model improvement...")
    test_results = test_model_improvement(train, val, new_features, segment='A')

    # Save results
    all_results = {
        'spike_analysis': analysis,
        'model_test': test_results,
        'timestamp': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'spike_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Results saved to {OUTPUT_DIR}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
