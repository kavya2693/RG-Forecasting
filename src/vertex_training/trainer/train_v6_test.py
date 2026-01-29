"""
Feature Comparison Test (v6) - WITH PROPER LOGGING
===================================================
Tests key configurations with flush to ensure Cloud Logging captures output.

Tests:
1. BASELINE (31 features)
2. + SPIKE only (36 features)
3. + VELOCITY only (35 features)
4. + SPIKE + VELOCITY (40 features)
5. + ALL (45 features)
"""

import os
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
    """Print and flush to ensure Cloud Logging captures output."""
    print(msg)
    sys.stdout.flush()


# EXACT PRODUCTION FEATURES (31 numeric + 2 categorical = 33 total)
ORIGINAL_NUMERIC = [
    # Calendar features (5)
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    # Cyclical encodings (4)
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    # Store closure features (4)
    'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
    # Lag features (5)
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    # Rolling statistics (5)
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    # Non-zero rates (3)
    'nz_rate_7', 'nz_rate_28', 'roll_mean_pos_28',
    # Dormancy/recency features (4)
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
]
# Note: 'is_local' from SKU attributes added separately in production (total 32 numeric)

SPIKE_FEATURES = [
    'feat_store_spike_pct', 'feat_store_promo_day', 'feat_seasonal_lift',
    'feat_had_recent_spike', 'feat_historical_spike_prob'
]

VELOCITY_FEATURES = [
    'feat_sale_frequency', 'feat_gap_vs_median', 'feat_is_overdue', 'feat_gap_pressure'
]

DORMANCY_FEATURES = [
    'feat_long_dormancy', 'feat_dormancy_burst_factor'
]

SEASONAL_FEATURES = [
    'feat_is_december', 'feat_pre_christmas_week', 'feat_is_ramadan'
]

CAT_FEATURES = ['store_id', 'sku_id']

SEGMENT_PARAMS = {
    'A': {'num_leaves': 127, 'min_child_samples': 20, 'learning_rate': 0.03, 'n_estimators': 300},
    'B': {'num_leaves': 63, 'min_child_samples': 50, 'learning_rate': 0.03, 'n_estimators': 200},
    'C': {'num_leaves': 31, 'min_child_samples': 100, 'learning_rate': 0.05, 'n_estimators': 150},
}


def load_parquet_from_gcs(bucket_name, prefix):
    """Load all parquet files from GCS prefix."""
    log(f"Loading from gs://{bucket_name}/{prefix}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    dfs = []
    for blob in blobs:
        if blob.name.endswith('.parquet'):
            data = blob.download_as_bytes()
            df = pd.read_parquet(io.BytesIO(data))
            dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    log(f"  Loaded {len(result):,} rows from {len(dfs)} files")
    return result


def assign_segments(df, train_df=None):
    if train_df is None:
        train_df = df
    sku_sales = train_df.groupby('sku_id')['y'].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = sku_sales.sum()
    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)
    df['segment'] = df['sku_id'].apply(lambda x: 'A' if x in a_skus else ('B' if x in b_skus else 'C'))
    return df


def train_model(train_df, val_df, features, segment, params):
    train_seg = train_df[train_df['segment'] == segment].copy()
    val_seg = val_df[val_df['segment'] == segment].copy()

    if len(train_seg) < 50 or len(val_seg) < 50:
        return None, None

    available = [f for f in features if f in train_seg.columns]
    cat_cols = CAT_FEATURES.copy()

    if 'feat_velocity_segment' in train_seg.columns and 'feat_sale_frequency' in features:
        cat_cols.append('feat_velocity_segment')

    for col in cat_cols:
        if col in train_seg.columns:
            train_seg[col] = train_seg[col].astype('category')
            val_seg[col] = val_seg[col].astype('category')

    for col in available:
        train_seg[col] = pd.to_numeric(train_seg[col], errors='coerce').fillna(0)
        val_seg[col] = pd.to_numeric(val_seg[col], errors='coerce').fillna(0)

    all_features = available + [c for c in cat_cols if c in train_seg.columns]
    X_train, X_val = train_seg[all_features], val_seg[all_features]
    y_train = train_seg['y'].values

    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'], min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))
    prob = clf.predict_proba(X_val)[:, 1]

    train_nz = train_seg[train_seg['y'] > 0]
    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'], min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
        reg_lambda=0.5, verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(train_nz[all_features], np.log1p(train_nz['y']))
    pred = np.expm1(reg.predict(X_val))

    threshold = 0.6 if segment in ['A', 'B'] else 0.7
    y_pred = np.where(prob > threshold, pred, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_seg['is_store_closed'].values == 1] = 0

    return y_pred, val_seg


def compute_metrics(val_df, y_pred):
    y_true = val_df['y'].values
    wmape_daily = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    wfa_daily = 100 - wmape_daily

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
    log(f"\n{'='*60}")
    log(f"  {label}")
    log(f"{'='*60}")

    features = ORIGINAL_NUMERIC.copy()
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
        y_pred, val_seg = train_model(train_df, val_df, features, seg, SEGMENT_PARAMS[seg])
        if y_pred is not None:
            all_preds.append(y_pred)
            all_vals.append(val_seg)
            log(f"  {seg}-segment: {len(val_seg):,} rows")

    combined_val = pd.concat(all_vals, ignore_index=True)
    combined_pred = np.concatenate(all_preds)
    metrics = compute_metrics(combined_val, combined_pred)

    log(f"  >>> Daily WFA:        {metrics['daily_wfa']:.2f}%")
    log(f"  >>> Weekly Store WFA: {metrics['weekly_store_wfa']:.2f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='myforecastingsales-data')
    parser.add_argument('--train-prefix', default='spike_test_v5/train/')
    parser.add_argument('--val-prefix', default='spike_test_v5/val/')
    args = parser.parse_args()

    log("=" * 70)
    log("FEATURE COMPARISON TEST v6 (WITH PROPER LOGGING)")
    log("=" * 70)
    log(f"Time: {datetime.now()}")
    log(f"Train: gs://{args.bucket}/{args.train_prefix}")
    log(f"Val:   gs://{args.bucket}/{args.val_prefix}")
    log("=" * 70)

    train_df = load_parquet_from_gcs(args.bucket, args.train_prefix)
    val_df = load_parquet_from_gcs(args.bucket, args.val_prefix)

    train_df = assign_segments(train_df)
    val_df = assign_segments(val_df, train_df)

    log(f"\nSegment distribution (val):")
    for seg in ['A', 'B', 'C']:
        cnt = len(val_df[val_df['segment'] == seg])
        log(f"  {seg}: {cnt:,} rows")

    # Key configurations to test
    results = {}
    configs = [
        ('baseline', [], "1. BASELINE (31 features)"),
        ('spike', ['spike'], "2. BASELINE + SPIKE (36 features)"),
        ('velocity', ['velocity'], "3. BASELINE + VELOCITY (35 features)"),
        ('spike_velocity', ['spike', 'velocity'], "4. BASELINE + SPIKE + VELOCITY (40 features)"),
        ('all', ['spike', 'velocity', 'dormancy', 'seasonal'], "5. BASELINE + ALL (45 features)"),
    ]

    for name, sets, label in configs:
        results[name] = run_test(train_df, val_df, sets, label)

    # Summary table
    log("\n" + "=" * 80)
    log("FINAL COMPARISON SUMMARY")
    log("=" * 80)

    baseline_daily = results['baseline']['daily_wfa']
    baseline_weekly = results['baseline']['weekly_store_wfa']

    log(f"\n{'Config':<25} {'Daily WFA':>12} {'Weekly WFA':>14} {'Δ Daily':>10} {'Δ Weekly':>10}")
    log("-" * 80)
    for name, metrics in results.items():
        d_delta = metrics['daily_wfa'] - baseline_daily
        w_delta = metrics['weekly_store_wfa'] - baseline_weekly
        log(f"{name:<25} {metrics['daily_wfa']:>11.2f}% {metrics['weekly_store_wfa']:>13.2f}% {d_delta:>+9.2f}pp {w_delta:>+9.2f}pp")

    # Best config
    best = max(results.items(), key=lambda x: x[1]['daily_wfa'])
    log(f"\n{'=' * 80}")
    log(f"BEST CONFIG: {best[0].upper()}")
    log(f"  Daily WFA improvement: +{best[1]['daily_wfa'] - baseline_daily:.2f}pp")
    log(f"  Weekly WFA improvement: +{best[1]['weekly_store_wfa'] - baseline_weekly:.2f}pp")
    log("=" * 80)

    # Feature group impact
    log("\nFEATURE GROUP IMPACT (vs baseline):")
    log("-" * 40)
    for name in ['spike', 'velocity']:
        if name in results:
            delta = results[name]['daily_wfa'] - baseline_daily
            verdict = "KEEP" if delta > 0.3 else ("MAYBE" if delta > 0 else "DROP")
            log(f"  {name.upper():<15}: {delta:>+.2f}pp daily -> {verdict}")

    # Combination impact
    if 'spike_velocity' in results:
        sv_delta = results['spike_velocity']['daily_wfa'] - baseline_daily
        spike_delta = results['spike']['daily_wfa'] - baseline_daily
        vel_delta = results['velocity']['daily_wfa'] - baseline_daily
        synergy = sv_delta - spike_delta - vel_delta
        log(f"\n  SPIKE+VELOCITY combo:   {sv_delta:>+.2f}pp daily")
        log(f"  Synergy (vs sum):       {synergy:>+.2f}pp")

    # Save results
    output = {
        'results': {k: v for k, v in results.items()},
        'best_config': best[0],
        'baseline_daily': baseline_daily,
        'baseline_weekly': baseline_weekly,
        'best_daily': best[1]['daily_wfa'],
        'best_weekly': best[1]['weekly_store_wfa'],
        'improvement_daily_pp': best[1]['daily_wfa'] - baseline_daily,
        'improvement_weekly_pp': best[1]['weekly_store_wfa'] - baseline_weekly,
        'timestamp': datetime.now().isoformat()
    }

    # Also save to GCS for retrieval
    results_json = json.dumps(output, indent=2)
    log(f"\n{results_json}")

    with open('/tmp/v6_test_results.json', 'w') as f:
        f.write(results_json)
    log("\nResults saved to /tmp/v6_test_results.json")
    log("TEST COMPLETE")


if __name__ == '__main__':
    main()
