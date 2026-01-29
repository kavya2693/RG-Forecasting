"""
FULL-SCALE Feature Test (Production Matching)
==============================================
Tests on FULL data (51M train, 19M val) matching production settings.

Tests all combinations:
1. BASELINE (30 features) - matches production
2. + SPIKE (5 features)
3. + VELOCITY (4 features)
4. + SPIKE + VELOCITY
5. + ALL (spike + velocity + dormancy + seasonal)
"""

import sys
import json
import argparse
from google.cloud import storage
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import io


def log(msg):
    """Print and flush for Cloud Logging."""
    print(msg, flush=True)
    sys.stdout.flush()


# EXACT PRODUCTION FEATURES (30 numeric)
BASELINE_FEATURES = [
    # Calendar (5)
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    # Cyclical (4)
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    # Closure (4)
    'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
    # Lags (5)
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    # Rolling (5)
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    # Non-zero rates (3)
    'nz_rate_7', 'nz_rate_28', 'roll_mean_pos_28',
    # Dormancy (4)
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
]

SPIKE_FEATURES = [
    'feat_store_spike_pct', 'feat_store_promo_day', 'feat_seasonal_lift',
    'feat_had_recent_spike', 'feat_historical_spike_prob'
]

VELOCITY_FEATURES = [
    'feat_sale_frequency', 'feat_gap_vs_median', 'feat_is_overdue', 'feat_gap_pressure'
]

DORMANCY_FEATURES = ['feat_long_dormancy', 'feat_dormancy_burst_factor']
SEASONAL_FEATURES = ['feat_is_december', 'feat_pre_christmas_week', 'feat_is_ramadan']

CAT_FEATURES = ['store_id', 'sku_id']

# PRODUCTION MODEL PARAMETERS (per segment)
SEGMENT_PARAMS = {
    'A': {'num_leaves': 255, 'min_child_samples': 20, 'learning_rate': 0.03, 'n_estimators': 300, 'reg_lambda': 0.1},
    'B': {'num_leaves': 127, 'min_child_samples': 50, 'learning_rate': 0.03, 'n_estimators': 200, 'reg_lambda': 0.3},
    'C': {'num_leaves': 63, 'min_child_samples': 100, 'learning_rate': 0.05, 'n_estimators': 150, 'reg_lambda': 0.5},
}


def load_parquet_from_gcs(bucket_name, prefix):
    """Load all parquet files from GCS prefix."""
    log(f"Loading from gs://{bucket_name}/{prefix}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    parquet_blobs = [b for b in blobs if b.name.endswith('.parquet')]
    log(f"  Found {len(parquet_blobs)} parquet files")

    dfs = []
    for i, blob in enumerate(parquet_blobs):
        data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(data))
        dfs.append(df)
        if (i + 1) % 50 == 0:
            log(f"    Loaded {i+1}/{len(parquet_blobs)} files...")

    result = pd.concat(dfs, ignore_index=True)
    log(f"  TOTAL: {len(result):,} rows")
    return result


def assign_segments(df, train_df=None):
    """Assign ABC segments based on sales volume."""
    if train_df is None:
        train_df = df
    sku_sales = train_df.groupby('sku_id')['y'].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = sku_sales.sum()
    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)
    df['segment'] = df['sku_id'].apply(lambda x: 'A' if x in a_skus else ('B' if x in b_skus else 'C'))
    return df


def train_segment_model(train_df, val_df, features, segment, params):
    """Train classifier + regressor for one segment."""
    train_seg = train_df[train_df['segment'] == segment].copy()
    val_seg = val_df[val_df['segment'] == segment].copy()

    if len(train_seg) < 100 or len(val_seg) < 100:
        return None, None

    # Filter to available features
    available = [f for f in features if f in train_seg.columns]

    # Prepare categorical features
    for col in CAT_FEATURES:
        if col in train_seg.columns:
            train_seg[col] = train_seg[col].astype('category')
            val_seg[col] = val_seg[col].astype('category')

    # Fill missing numeric
    for col in available:
        train_seg[col] = pd.to_numeric(train_seg[col], errors='coerce').fillna(0)
        val_seg[col] = pd.to_numeric(val_seg[col], errors='coerce').fillna(0)

    all_features = available + CAT_FEATURES
    X_train = train_seg[all_features]
    X_val = val_seg[all_features]
    y_train = train_seg['y'].values

    # Stage 1: Binary Classifier
    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))
    prob = clf.predict_proba(X_val)[:, 1]

    # Stage 2: Log-Transform Regressor (non-zero only)
    train_nz = train_seg[train_seg['y'] > 0]
    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'],
        min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(train_nz[all_features], np.log1p(train_nz['y']))
    pred_qty = np.expm1(reg.predict(X_val))

    # Combine: threshold on probability
    threshold = 0.5 if segment == 'A' else (0.6 if segment == 'B' else 0.7)
    y_pred = np.where(prob > threshold, pred_qty, 0)
    y_pred = np.maximum(0, y_pred)

    # Zero out closed stores
    if 'is_store_closed' in val_seg.columns:
        y_pred[val_seg['is_store_closed'].values == 1] = 0

    return y_pred, val_seg


def compute_metrics(val_df, y_pred):
    """Compute WFA metrics at daily and weekly-store levels."""
    y_true = val_df['y'].values

    # Daily WFA
    wmape_daily = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    wfa_daily = 100 - wmape_daily

    # Weekly Store WFA
    val_df = val_df.copy()
    val_df['y_pred'] = y_pred
    val_df['date'] = pd.to_datetime(val_df['date'])
    val_df['week'] = val_df['date'].dt.isocalendar().week
    val_df['year'] = val_df['date'].dt.year

    weekly = val_df.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape_weekly = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / max(np.sum(weekly['y']), 1)
    wfa_weekly = 100 - wmape_weekly

    return {'daily_wfa': wfa_daily, 'weekly_store_wfa': wfa_weekly}


def run_test(train_df, val_df, feature_sets, label):
    """Run training and evaluation for a feature configuration."""
    log(f"\n{'='*70}")
    log(f"  {label}")
    log(f"{'='*70}")

    # Build feature list
    features = BASELINE_FEATURES.copy()
    if 'spike' in feature_sets:
        features += SPIKE_FEATURES
    if 'velocity' in feature_sets:
        features += VELOCITY_FEATURES
    if 'dormancy' in feature_sets:
        features += DORMANCY_FEATURES
    if 'seasonal' in feature_sets:
        features += SEASONAL_FEATURES

    log(f"  Total features: {len(features)}")

    all_preds, all_vals = [], []
    for seg in ['A', 'B', 'C']:
        log(f"  Training {seg}-segment...")
        y_pred, val_seg = train_segment_model(train_df, val_df, features, seg, SEGMENT_PARAMS[seg])
        if y_pred is not None:
            all_preds.append(y_pred)
            all_vals.append(val_seg)
            log(f"    {seg}: {len(val_seg):,} rows")

    combined_val = pd.concat(all_vals, ignore_index=True)
    combined_pred = np.concatenate(all_preds)
    metrics = compute_metrics(combined_val, combined_pred)

    log(f"  >>> DAILY WFA:        {metrics['daily_wfa']:.2f}%")
    log(f"  >>> WEEKLY STORE WFA: {metrics['weekly_store_wfa']:.2f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='myforecastingsales-data')
    parser.add_argument('--train-prefix', default='full_test/train/')
    parser.add_argument('--val-prefix', default='full_test/val/')
    args = parser.parse_args()

    log("=" * 80)
    log("FULL-SCALE FEATURE TEST (PRODUCTION MATCHING)")
    log("=" * 80)
    log(f"Time: {datetime.now()}")
    log(f"Train: gs://{args.bucket}/{args.train_prefix}")
    log(f"Val:   gs://{args.bucket}/{args.val_prefix}")
    log("=" * 80)

    # Load FULL data
    train_df = load_parquet_from_gcs(args.bucket, args.train_prefix)
    val_df = load_parquet_from_gcs(args.bucket, args.val_prefix)

    # Assign segments based on training data
    log("\nAssigning ABC segments...")
    train_df = assign_segments(train_df)
    val_df = assign_segments(val_df, train_df)

    for seg in ['A', 'B', 'C']:
        train_cnt = len(train_df[train_df['segment'] == seg])
        val_cnt = len(val_df[val_df['segment'] == seg])
        log(f"  {seg}: {train_cnt:,} train, {val_cnt:,} val")

    # Test configurations
    results = {}
    configs = [
        ('baseline', [], "1. BASELINE (30 features)"),
        ('spike', ['spike'], "2. BASELINE + SPIKE (35 features)"),
        ('velocity', ['velocity'], "3. BASELINE + VELOCITY (34 features)"),
        ('spike_velocity', ['spike', 'velocity'], "4. SPIKE + VELOCITY (39 features)"),
        ('all', ['spike', 'velocity', 'dormancy', 'seasonal'], "5. ALL FEATURES (44 features)"),
    ]

    for name, sets, label in configs:
        results[name] = run_test(train_df, val_df, sets, label)

    # Final summary
    log("\n" + "=" * 80)
    log("FINAL RESULTS SUMMARY")
    log("=" * 80)

    baseline_daily = results['baseline']['daily_wfa']
    baseline_weekly = results['baseline']['weekly_store_wfa']

    log(f"\n{'Config':<20} {'Daily WFA':>12} {'Weekly WFA':>14} {'Δ Daily':>10} {'Δ Weekly':>10}")
    log("-" * 80)
    for name, metrics in results.items():
        d_delta = metrics['daily_wfa'] - baseline_daily
        w_delta = metrics['weekly_store_wfa'] - baseline_weekly
        log(f"{name:<20} {metrics['daily_wfa']:>11.2f}% {metrics['weekly_store_wfa']:>13.2f}% {d_delta:>+9.2f}pp {w_delta:>+9.2f}pp")

    # Best config
    best = max(results.items(), key=lambda x: x[1]['daily_wfa'])
    log(f"\n{'=' * 80}")
    log(f"BEST CONFIG: {best[0].upper()}")
    log(f"  Daily WFA: {best[1]['daily_wfa']:.2f}% (improvement: +{best[1]['daily_wfa'] - baseline_daily:.2f}pp)")
    log(f"  Weekly WFA: {best[1]['weekly_store_wfa']:.2f}% (improvement: +{best[1]['weekly_store_wfa'] - baseline_weekly:.2f}pp)")
    log("=" * 80)

    # Feature group analysis
    log("\nFEATURE GROUP ANALYSIS:")
    log("-" * 40)
    for name in ['spike', 'velocity']:
        if name in results:
            d = results[name]['daily_wfa'] - baseline_daily
            verdict = "KEEP ✓" if d > 0.3 else ("DROP ✗" if d < -0.1 else "NEUTRAL")
            log(f"  {name.upper():<15}: {d:>+.2f}pp daily -> {verdict}")

    if 'spike_velocity' in results and 'spike' in results and 'velocity' in results:
        combo = results['spike_velocity']['daily_wfa'] - baseline_daily
        spike_alone = results['spike']['daily_wfa'] - baseline_daily
        vel_alone = results['velocity']['daily_wfa'] - baseline_daily
        synergy = combo - spike_alone - vel_alone
        log(f"\n  COMBINATION ANALYSIS:")
        log(f"    Spike alone:        {spike_alone:>+.2f}pp")
        log(f"    Velocity alone:     {vel_alone:>+.2f}pp")
        log(f"    Spike+Velocity:     {combo:>+.2f}pp")
        log(f"    Synergy:            {synergy:>+.2f}pp")

    # Save results as JSON
    output = {
        'results': results,
        'best_config': best[0],
        'baseline_daily': baseline_daily,
        'baseline_weekly': baseline_weekly,
        'best_daily': best[1]['daily_wfa'],
        'best_weekly': best[1]['weekly_store_wfa'],
        'timestamp': datetime.now().isoformat()
    }

    log(f"\nJSON RESULTS:")
    log(json.dumps(output, indent=2))

    log("\n" + "=" * 80)
    log("TEST COMPLETE")
    log("=" * 80)


if __name__ == '__main__':
    main()
