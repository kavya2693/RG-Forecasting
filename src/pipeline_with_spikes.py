"""
Full Production Pipeline + Spike Features
==========================================
ALL original 32 features + 2 categorical + 5 spike features
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/tmp/pipeline_with_spikes")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# FULL ORIGINAL FEATURE SET (32 numeric + 2 categorical)
# ============================================================================
ORIGINAL_FEATURES = [
    # Calendar/temporal (9)
    "dow", "is_weekend", "day_of_year", "week_of_year", "month",
    "sin_dow", "cos_dow", "sin_doy", "cos_doy",

    # Store closures (4)
    "is_store_closed", "days_to_next_closure", "days_from_prev_closure", "is_closure_week",

    # Lag features (5)
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_56",

    # Rolling statistics (7)
    "roll_mean_7", "roll_mean_28", "roll_sum_7", "roll_sum_28", "roll_std_28",
    "roll_mean_pos_28", "lag_7_log1p", "roll_mean_pos_28_log1p",

    # Sparsity-aware (2)
    "nz_rate_7", "nz_rate_28",

    # Dormancy features (5)
    "days_since_last_sale_asof", "dormancy_capped", "dormancy_bucket",
    "last_sale_qty_asof", "trend_idx",
]

CAT_FEATURES = ["sku_id", "store_id"]

# Production hyperparameters
SEGMENT_PARAMS = {
    "A": {"num_leaves": 255, "min_child_samples": 10, "learning_rate": 0.015, "n_estimators": 1000, "threshold": 0.6},
    "B": {"num_leaves": 63, "min_child_samples": 50, "learning_rate": 0.03, "n_estimators": 400, "threshold": 0.6},
    "C": {"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05, "n_estimators": 300, "threshold": 0.7},
}


def load_data():
    """Load full training and validation data."""
    print("Loading data...")
    train = pd.read_csv("/tmp/c1_data/train_final.csv")
    val = pd.read_csv("/tmp/c1_data/val_final.csv")

    for df in [train, val]:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week.astype(int)

    print(f"  Train: {len(train):,}, Val: {len(val):,}")
    return train, val


def add_spike_features(df, is_train=True, train_stats=None):
    """
    Add spike detection and classification features.

    Spike Classifications:
    - STORE_PROMO: Store-wide event (>15% SKUs spike)
    - SEASONAL: Historical high-season week
    - PANIC_BUY: Isolated spike with high magnitude (>5x baseline)
    - REGULAR_SPIKE: Normal promotional spike (2-5x baseline)
    """
    target = 'y' if 'y' in df.columns else 'qty'
    df = df.copy()

    print("  Adding spike features...")

    # ---- SPIKE DETECTION ----
    df['_baseline'] = df['roll_mean_28'].fillna(0) + 0.1
    df['_spike_ratio'] = df[target] / df['_baseline']

    # Classify spike types
    df['is_spike'] = (df['_spike_ratio'] > 2.0) & (df[target] > 1)
    df['is_panic_spike'] = (df['_spike_ratio'] > 5.0) & (df[target] > 3)  # Extreme spike
    df['is_regular_spike'] = df['is_spike'] & ~df['is_panic_spike']

    # ---- STORE-WIDE EVENT DETECTION ----
    store_day = df.groupby(['store_id', 'date']).agg({
        'is_spike': 'sum',
        'is_panic_spike': 'sum',
        'sku_id': 'nunique'
    }).reset_index()
    store_day.columns = ['store_id', 'date', 'spike_count', 'panic_count', 'sku_count']
    store_day['spike_pct'] = store_day['spike_count'] / store_day['sku_count']
    store_day['panic_pct'] = store_day['panic_count'] / store_day['sku_count']
    store_day['is_store_promo'] = store_day['spike_pct'] > 0.15
    store_day['is_store_panic'] = store_day['panic_pct'] > 0.05  # 5% panic = major event

    df = df.merge(
        store_day[['store_id', 'date', 'spike_pct', 'panic_pct', 'is_store_promo', 'is_store_panic']],
        on=['store_id', 'date'], how='left'
    )

    # ---- SEASONAL PATTERN DETECTION ----
    if is_train:
        # Calculate from training data
        week_avg = df.groupby(['store_id', 'sku_id', 'week'])[target].mean().reset_index()
        week_avg.columns = ['store_id', 'sku_id', 'week', 'week_avg']
        overall_avg = df.groupby(['store_id', 'sku_id'])[target].mean().reset_index()
        overall_avg.columns = ['store_id', 'sku_id', 'overall_avg']
        week_avg = week_avg.merge(overall_avg, on=['store_id', 'sku_id'])
        week_avg['week_lift'] = week_avg['week_avg'] / (week_avg['overall_avg'] + 0.1)

        # Historical spike probability
        spike_prob = df.groupby(['store_id', 'sku_id', 'week'])['is_spike'].mean().reset_index()
        spike_prob.columns = ['store_id', 'sku_id', 'week', 'hist_spike_prob']

        train_stats = {'week_lift': week_avg, 'spike_prob': spike_prob}

    df = df.merge(
        train_stats['week_lift'][['store_id', 'sku_id', 'week', 'week_lift']],
        on=['store_id', 'sku_id', 'week'], how='left'
    )
    df = df.merge(
        train_stats['spike_prob'],
        on=['store_id', 'sku_id', 'week'], how='left'
    )

    # ---- CREATE FINAL FEATURES ----

    # 1. Store promo day (probable promotion)
    df['spike_store_promo'] = df['is_store_promo'].fillna(False).astype(int)

    # 2. Store panic event (probable panic buying / major event)
    df['spike_store_panic'] = df['is_store_panic'].fillna(False).astype(int)

    # 3. Seasonal lift multiplier
    df['spike_seasonal_lift'] = df['week_lift'].fillna(1.0).clip(0.5, 3.0)

    # 4. Store spike intensity (continuous)
    df['spike_store_intensity'] = df['spike_pct'].fillna(0).clip(0, 1)

    # 5. Historical spike probability
    df['spike_hist_prob'] = df['hist_spike_prob'].fillna(0).clip(0, 1)

    # 6. Recent spike memory (had spike in last 7 days)
    df = df.sort_values(['store_id', 'sku_id', 'date'])
    df['spike_recent'] = df.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()
    ).fillna(0).astype(int)

    # 7. Post-spike dip expected (7 days after spike)
    df['spike_post_dip'] = df.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(7)
    ).fillna(0).astype(int)

    # Clean up temp columns
    temp_cols = ['_baseline', '_spike_ratio', 'is_spike', 'is_panic_spike', 'is_regular_spike',
                 'spike_pct', 'panic_pct', 'is_store_promo', 'is_store_panic', 'week_lift', 'hist_spike_prob']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')

    spike_features = [
        'spike_store_promo',      # Probable promotion day
        'spike_store_panic',      # Probable panic/major event
        'spike_seasonal_lift',    # Seasonal multiplier
        'spike_store_intensity',  # Store-wide spike %
        'spike_hist_prob',        # Historical spike probability
        'spike_recent',           # Had recent spike
        'spike_post_dip',         # Expect post-spike dip
    ]

    return df, spike_features, train_stats


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


def compute_all_metrics(y_true, y_pred, val_df):
    """Compute metrics at all aggregation levels."""
    target = 'y' if 'y' in val_df.columns else 'qty'
    results = {}

    # Daily SKU-Store
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    results['daily_sku_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Weekly aggregations
    val_df = val_df.copy()
    val_df['y_pred'] = y_pred

    # Weekly SKU-Store
    weekly = val_df.groupby(['store_id', 'sku_id', 'year', 'week'], observed=True).agg(
        {target: 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly[target] - weekly['y_pred'])) / max(np.sum(weekly[target]), 1)
    results['weekly_sku_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Weekly Store
    weekly_store = val_df.groupby(['store_id', 'year', 'week'], observed=True).agg(
        {target: 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_store[target] - weekly_store['y_pred'])) / max(np.sum(weekly_store[target]), 1)
    results['weekly_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Weekly Total
    weekly_total = val_df.groupby(['year', 'week'], observed=True).agg(
        {target: 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_total[target] - weekly_total['y_pred'])) / max(np.sum(weekly_total[target]), 1)
    results['weekly_total'] = {'wmape': wmape, 'wfa': 100 - wmape}

    return results


def train_two_stage(train, val, features, segment):
    """Train production two-stage model."""
    target = 'y' if 'y' in train.columns else 'qty'
    params = SEGMENT_PARAMS[segment]

    train_seg = train[train['segment'] == segment].copy()
    val_seg = val[val['segment'] == segment].copy()

    if len(train_seg) < 100 or len(val_seg) < 100:
        return None, None

    # Prepare features
    available = [f for f in features if f in train_seg.columns]

    for col in CAT_FEATURES:
        train_seg[col] = train_seg[col].astype('category')
        val_seg[col] = val_seg[col].astype('category')

    for col in available:
        train_seg[col] = pd.to_numeric(train_seg[col], errors='coerce').fillna(0)
        val_seg[col] = pd.to_numeric(val_seg[col], errors='coerce').fillna(0)

    all_features = available + CAT_FEATURES
    X_train = train_seg[all_features]
    X_val = val_seg[all_features]
    y_train = train_seg[target].values
    y_val = val_seg[target].values

    # Stage 1: Classifier
    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'],
        n_estimators=min(params['n_estimators'], 600),
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))
    prob = clf.predict_proba(X_val)[:, 1]

    # Stage 2: Regressor on non-zero
    train_nz = train_seg[train_seg[target] > 0].copy()
    X_train_nz = train_nz[all_features]

    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'],
        min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'],
        n_estimators=min(params['n_estimators'], 600),
        reg_lambda=0.5,
        verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(X_train_nz, np.log1p(train_nz[target]))
    pred = np.expm1(reg.predict(X_val))

    # Combine
    threshold = params['threshold']
    y_pred = np.where(prob > threshold, pred, 0)
    y_pred = np.maximum(0, y_pred)

    # Calibration for A-items
    if segment == 'A':
        prob_tr = clf.predict_proba(X_train)[:, 1]
        pred_tr = np.expm1(reg.predict(X_train))
        y_pred_tr = np.where(prob_tr > threshold, pred_tr, 0)
        mask = y_pred_tr > 0.1
        if np.sum(y_pred_tr[mask]) > 0:
            k = np.clip(np.sum(y_train[mask]) / np.sum(y_pred_tr[mask]), 0.8, 1.3)
            y_pred = y_pred * k

    # Force zero on closure days
    if 'is_store_closed' in val_seg.columns:
        y_pred[val_seg['is_store_closed'].values == 1] = 0

    return y_pred, val_seg


def run_pipeline(train, val, features, name):
    """Run full pipeline and return metrics."""
    target = 'y' if 'y' in train.columns else 'qty'

    all_y_true = []
    all_y_pred = []
    all_val_dfs = []
    segment_metrics = {}

    for segment in ['A', 'B', 'C']:
        y_pred, val_seg = train_two_stage(train, val, features, segment)
        if y_pred is None:
            continue

        y_val = val_seg[target].values
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_val_dfs.append(val_seg.assign(y_pred=y_pred))

        # Segment metrics
        wmape = 100 * np.sum(np.abs(y_val - y_pred)) / max(np.sum(y_val), 1)
        segment_metrics[segment] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Combined metrics
    combined_val = pd.concat(all_val_dfs, ignore_index=True)
    overall = compute_all_metrics(np.array(all_y_true), np.array(all_y_pred), combined_val)

    return overall, segment_metrics


def main():
    print("=" * 70)
    print("FULL PIPELINE TEST: ORIGINAL vs ORIGINAL + SPIKE FEATURES")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Load data
    train, val = load_data()

    # Assign segments
    train = assign_segments(train)
    val = assign_segments(val)

    # ========== TEST 1: ORIGINAL FEATURES ONLY ==========
    print("\n" + "=" * 50)
    print("[1] ORIGINAL FEATURES (32 numeric + 2 categorical)")
    print("=" * 50)

    available_original = [f for f in ORIGINAL_FEATURES if f in train.columns]
    print(f"    Features available: {len(available_original)}")

    original_metrics, original_seg = run_pipeline(train, val, available_original, "Original")

    print(f"\n    Results:")
    print(f"    {'Level':<20} {'WFA':>10}")
    print("    " + "-" * 32)
    for level in ['daily_sku_store', 'weekly_sku_store', 'weekly_store', 'weekly_total']:
        print(f"    {level:<20} {original_metrics[level]['wfa']:>9.2f}%")

    # ========== TEST 2: ORIGINAL + SPIKE FEATURES ==========
    print("\n" + "=" * 50)
    print("[2] ORIGINAL + SPIKE FEATURES")
    print("=" * 50)

    # Add spike features
    train_spike, spike_features, train_stats = add_spike_features(train, is_train=True)
    val_spike, _, _ = add_spike_features(val, is_train=False, train_stats=train_stats)

    all_features = available_original + spike_features
    print(f"    Original features: {len(available_original)}")
    print(f"    Spike features: {len(spike_features)}")
    print(f"    Total features: {len(all_features)}")
    print(f"    Spike features: {spike_features}")

    spike_metrics, spike_seg = run_pipeline(train_spike, val_spike, all_features, "With Spikes")

    print(f"\n    Results:")
    print(f"    {'Level':<20} {'WFA':>10}")
    print("    " + "-" * 32)
    for level in ['daily_sku_store', 'weekly_sku_store', 'weekly_store', 'weekly_total']:
        print(f"    {level:<20} {spike_metrics[level]['wfa']:>9.2f}%")

    # ========== COMPARISON ==========
    print("\n" + "=" * 70)
    print("COMPARISON: IMPROVEMENT FROM SPIKE FEATURES")
    print("=" * 70)
    print(f"\n{'Level':<25} {'Original':>12} {'+ Spikes':>12} {'Change':>12}")
    print("-" * 61)

    for level in ['daily_sku_store', 'weekly_sku_store', 'weekly_store', 'weekly_total']:
        orig = original_metrics[level]['wfa']
        spike = spike_metrics[level]['wfa']
        change = spike - orig
        status = "✓" if change > 0.1 else ("✗" if change < -0.1 else "~")
        print(f"{level:<25} {orig:>11.2f}% {spike:>11.2f}% {change:>+10.2f}pp {status}")

    # Segment breakdown
    print(f"\n{'Segment':<10} {'Original':>12} {'+ Spikes':>12} {'Change':>12}")
    print("-" * 46)
    for seg in ['A', 'B', 'C']:
        if seg in original_seg and seg in spike_seg:
            orig = original_seg[seg]['wfa']
            spike = spike_seg[seg]['wfa']
            change = spike - orig
            print(f"{seg:<10} {orig:>11.2f}% {spike:>11.2f}% {change:>+10.2f}pp")

    # Save results
    results = {
        'original': {
            'features_count': len(available_original),
            'metrics': original_metrics,
            'segment_metrics': original_seg
        },
        'with_spikes': {
            'features_count': len(all_features),
            'spike_features': spike_features,
            'metrics': spike_metrics,
            'segment_metrics': spike_seg
        },
        'improvement': {
            level: spike_metrics[level]['wfa'] - original_metrics[level]['wfa']
            for level in ['daily_sku_store', 'weekly_sku_store', 'weekly_store', 'weekly_total']
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(OUTPUT_DIR / 'pipeline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {OUTPUT_DIR / 'pipeline_comparison.json'}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
