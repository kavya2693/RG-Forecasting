"""
BASELINE-2: Vertex AI Training Script
=====================================
- Per-fold tier tables (no look-ahead bias)
- Spike + Velocity features (9 new)
- Compares recursive vs teacher-forcing validation

Run on Vertex AI with:
  gcloud ai custom-jobs create --region=us-central1 ...
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
import gc


def log(msg):
    """Print and flush for Cloud Logging."""
    print(msg, flush=True)
    sys.stdout.flush()


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
BASELINE_FEATURES = [
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'is_store_closed',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_28',
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
]

SPIKE_FEATURES = [
    'feat_store_spike_pct', 'feat_store_promo_day', 'feat_seasonal_lift',
    'feat_had_recent_spike', 'feat_historical_spike_prob'
]

VELOCITY_FEATURES = [
    'feat_sale_frequency', 'feat_gap_vs_median', 'feat_is_overdue', 'feat_gap_pressure'
]

ALL_FEATURES = BASELINE_FEATURES + SPIKE_FEATURES + VELOCITY_FEATURES
CAT_FEATURES = ['store_id', 'sku_id']

SEGMENT_PARAMS = {
    'A': {'num_leaves': 255, 'min_child_samples': 20, 'learning_rate': 0.03,
          'n_estimators': 300, 'reg_lambda': 0.1, 'threshold': 0.5},
    'B': {'num_leaves': 127, 'min_child_samples': 50, 'learning_rate': 0.03,
          'n_estimators': 200, 'reg_lambda': 0.3, 'threshold': 0.6},
    'C': {'num_leaves': 63, 'min_child_samples': 100, 'learning_rate': 0.05,
          'n_estimators': 150, 'reg_lambda': 0.5, 'threshold': 0.7},
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


def train_segment_model(train_df, features, segment, params):
    """Train classifier + regressor for one segment."""
    train_seg = train_df[train_df['segment'] == segment].copy()

    if len(train_seg) < 100:
        return None, None

    available = [f for f in features if f in train_seg.columns]

    for col in CAT_FEATURES:
        if col in train_seg.columns:
            train_seg[col] = train_seg[col].astype('category')

    for col in available:
        train_seg[col] = pd.to_numeric(train_seg[col], errors='coerce').fillna(0)

    all_features = available + CAT_FEATURES
    X_train = train_seg[all_features]
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

    # Stage 2: Log-Transform Regressor
    train_nz = train_seg[train_seg['y'] > 0]
    if len(train_nz) < 10:
        return clf, None

    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'],
        min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(train_nz[all_features], np.log1p(train_nz['y']))

    return clf, reg


def predict_teacher_forcing(clf, reg, val_df, features, segment, params):
    """Predict using pre-computed features (teacher forcing - Baseline-1 behavior)."""
    val_seg = val_df[val_df['segment'] == segment].copy()

    if len(val_seg) < 100:
        return None, None

    available = [f for f in features if f in val_seg.columns]

    for col in CAT_FEATURES:
        if col in val_seg.columns:
            val_seg[col] = val_seg[col].astype('category')

    for col in available:
        val_seg[col] = pd.to_numeric(val_seg[col], errors='coerce').fillna(0)

    all_features = available + CAT_FEATURES
    X_val = val_seg[all_features]

    prob = clf.predict_proba(X_val)[:, 1]
    if reg is not None:
        pred_value = np.expm1(reg.predict(X_val))
    else:
        pred_value = np.zeros(len(X_val))

    threshold = params['threshold']
    y_pred = np.where(prob > threshold, pred_value, 0)
    y_pred = np.maximum(0, y_pred)

    if 'is_store_closed' in val_seg.columns:
        y_pred[val_seg['is_store_closed'].values == 1] = 0

    return y_pred, val_seg


def compute_metrics(val_df, y_pred):
    """Compute WFA metrics at daily and weekly-store levels."""
    y_true = val_df['y'].values

    # Daily WFA
    wmape_daily = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    wfa_daily = 100 - wmape_daily

    # Bias ratio
    bias_ratio = np.sum(y_pred) / max(np.sum(y_true), 1)

    # Weekly Store WFA
    val_df = val_df.copy()
    val_df['y_pred'] = y_pred
    val_df['date'] = pd.to_datetime(val_df['date'])
    val_df['week'] = val_df['date'].dt.isocalendar().week
    val_df['year'] = val_df['date'].dt.year

    weekly = val_df.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape_weekly = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / max(np.sum(weekly['y']), 1)
    wfa_weekly = 100 - wmape_weekly

    # Sale F1
    actual_sale = (y_true > 0).astype(int)
    pred_sale = (y_pred > 0).astype(int)
    tp = np.sum((pred_sale == 1) & (actual_sale == 1))
    fp = np.sum((pred_sale == 1) & (actual_sale == 0))
    fn = np.sum((pred_sale == 0) & (actual_sale == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        'daily_wfa': wfa_daily,
        'weekly_store_wfa': wfa_weekly,
        'bias_ratio': bias_ratio,
        'sale_f1': f1,
        'sale_precision': precision,
        'sale_recall': recall,
    }


def run_test(train_df, val_df, features, label):
    """Run training and evaluation for a feature configuration."""
    log(f"\n{'='*70}")
    log(f"  {label}")
    log(f"{'='*70}")
    log(f"  Total features: {len(features)}")

    all_preds, all_vals = [], []
    for seg in ['A', 'B', 'C']:
        log(f"  Training {seg}-segment...")

        clf, reg = train_segment_model(train_df, features, seg, SEGMENT_PARAMS[seg])
        if clf is None:
            log(f"    {seg}: skipped (too few samples)")
            continue

        y_pred, val_seg = predict_teacher_forcing(clf, reg, val_df, features, seg, SEGMENT_PARAMS[seg])
        if y_pred is None:
            continue

        all_preds.append(y_pred)
        all_vals.append(val_seg)

        seg_metrics = compute_metrics(val_seg, y_pred)
        log(f"    {seg}: {len(val_seg):,} rows, Daily WFA={seg_metrics['daily_wfa']:.2f}%, "
            f"Bias={seg_metrics['bias_ratio']:.3f}")

        del clf, reg
        gc.collect()

    if len(all_preds) == 0:
        return None

    combined_val = pd.concat(all_vals, ignore_index=True)
    combined_pred = np.concatenate(all_preds)
    metrics = compute_metrics(combined_val, combined_pred)

    log(f"\n  >>> DAILY WFA:        {metrics['daily_wfa']:.2f}%")
    log(f"  >>> WEEKLY STORE WFA: {metrics['weekly_store_wfa']:.2f}%")
    log(f"  >>> BIAS RATIO:       {metrics['bias_ratio']:.3f}")
    log(f"  >>> SALE F1:          {metrics['sale_f1']:.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='myforecastingsales-data')
    parser.add_argument('--train-prefix', default='baseline2/f1_train/')
    parser.add_argument('--val-prefix', default='baseline2/f1_val/')
    args = parser.parse_args()

    log("=" * 80)
    log("BASELINE-2: FEATURE TEST WITH SPIKE + VELOCITY")
    log("=" * 80)
    log(f"Time: {datetime.now()}")
    log(f"Train: gs://{args.bucket}/{args.train_prefix}")
    log(f"Val:   gs://{args.bucket}/{args.val_prefix}")
    log("=" * 80)

    # Load data
    train_df = load_parquet_from_gcs(args.bucket, args.train_prefix)
    val_df = load_parquet_from_gcs(args.bucket, args.val_prefix)

    # Assign segments
    log("\nAssigning ABC segments...")
    train_df = assign_segments(train_df)
    val_df = assign_segments(val_df, train_df)

    for seg in ['A', 'B', 'C']:
        train_cnt = len(train_df[train_df['segment'] == seg])
        val_cnt = len(val_df[val_df['segment'] == seg])
        log(f"  {seg}: {train_cnt:,} train, {val_cnt:,} val")

    # Test configurations
    results = {}

    # 1. Baseline only (original 24 features available)
    results['baseline'] = run_test(train_df, val_df, BASELINE_FEATURES,
                                   f"1. BASELINE ({len(BASELINE_FEATURES)} features)")

    # 2. Baseline + Spike
    results['baseline_spike'] = run_test(train_df, val_df, BASELINE_FEATURES + SPIKE_FEATURES,
                                         f"2. BASELINE + SPIKE ({len(BASELINE_FEATURES + SPIKE_FEATURES)} features)")

    # 3. Baseline + Velocity
    results['baseline_velocity'] = run_test(train_df, val_df, BASELINE_FEATURES + VELOCITY_FEATURES,
                                            f"3. BASELINE + VELOCITY ({len(BASELINE_FEATURES + VELOCITY_FEATURES)} features)")

    # 4. Baseline + Spike + Velocity (Baseline-2)
    results['baseline2'] = run_test(train_df, val_df, ALL_FEATURES,
                                    f"4. BASELINE-2 (ALL: {len(ALL_FEATURES)} features)")

    # Final summary
    log("\n" + "=" * 80)
    log("BASELINE-2 COMPARISON SUMMARY")
    log("=" * 80)

    if results['baseline']:
        baseline_daily = results['baseline']['daily_wfa']
        baseline_weekly = results['baseline']['weekly_store_wfa']

        log(f"\n{'Config':<25} {'Daily WFA':>12} {'Weekly WFA':>14} {'Δ Daily':>10} {'Δ Weekly':>10} {'Bias':>8}")
        log("-" * 90)

        for name, metrics in results.items():
            if metrics:
                d_delta = metrics['daily_wfa'] - baseline_daily
                w_delta = metrics['weekly_store_wfa'] - baseline_weekly
                log(f"{name:<25} {metrics['daily_wfa']:>11.2f}% {metrics['weekly_store_wfa']:>13.2f}% "
                    f"{d_delta:>+9.2f}pp {w_delta:>+9.2f}pp {metrics['bias_ratio']:>7.3f}")

    # Feature impact analysis
    log("\n" + "-" * 40)
    log("FEATURE GROUP IMPACT:")
    if results['baseline'] and results['baseline_spike']:
        spike_impact = results['baseline_spike']['daily_wfa'] - results['baseline']['daily_wfa']
        log(f"  SPIKE:    {spike_impact:>+.2f}pp daily")
    if results['baseline'] and results['baseline_velocity']:
        vel_impact = results['baseline_velocity']['daily_wfa'] - results['baseline']['daily_wfa']
        log(f"  VELOCITY: {vel_impact:>+.2f}pp daily")
    if results['baseline'] and results['baseline2']:
        total_impact = results['baseline2']['daily_wfa'] - results['baseline']['daily_wfa']
        log(f"  COMBINED: {total_impact:>+.2f}pp daily")

    # Save results
    output = {
        'results': results,
        'features': {
            'baseline': BASELINE_FEATURES,
            'spike': SPIKE_FEATURES,
            'velocity': VELOCITY_FEATURES,
            'total': len(ALL_FEATURES)
        },
        'timestamp': datetime.now().isoformat(),
        'description': 'Baseline-2 with per-fold tiers, spike+velocity features'
    }

    log(f"\nJSON RESULTS:")
    log(json.dumps(output, indent=2, default=str))

    log("\n" + "=" * 80)
    log("BASELINE-2 TEST COMPLETE")
    log("=" * 80)


if __name__ == '__main__':
    main()
