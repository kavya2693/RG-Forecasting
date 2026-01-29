"""
Vertex AI Training Script - Spike Feature Comparison
=====================================================
Compares production baseline vs production + spike features.
Runs on Vertex AI with BigQuery data access.
"""

import os
import json
import argparse
from google.cloud import bigquery
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

# ==============================================================================
# FEATURE SETS
# ==============================================================================

ORIGINAL_NUMERIC = {
    'T1_MATURE': [
        'trend_idx', 'dow', 'week_of_year', 'month', 'is_weekend', 'day_of_year',
        'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
        'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
        'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
        'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28', 'nz_rate_28',
        'nz_rate_7', 'roll_mean_pos_28',
        'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
        'lag_7_log1p', 'roll_mean_28_log1p', 'roll_mean_pos_28_log1p'
    ],
    'T2_GROWING': [
        'trend_idx', 'dow', 'month', 'is_weekend', 'day_of_year',
        'sin_doy', 'cos_doy',
        'is_store_closed', 'is_closure_week',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'roll_mean_7', 'roll_mean_28', 'nz_rate_28',
        'nz_rate_7', 'roll_mean_pos_28',
        'days_since_last_sale_asof', 'zero_run_length_asof', 'dormancy_capped',
        'lag_7_log1p', 'roll_mean_28_log1p'
    ],
}

SPIKE_FEATURES = [
    'feat_store_spike_pct',
    'feat_store_promo_day',
    'feat_seasonal_lift',
    'feat_had_recent_spike',
    'feat_historical_spike_prob'
]

CAT_FEATURES = ['store_id', 'sku_id']

# Segment-specific hyperparameters (production)
SEGMENT_PARAMS = {
    'A': {'num_leaves': 255, 'min_child_samples': 10, 'learning_rate': 0.015, 'n_estimators': 800},
    'B': {'num_leaves': 63, 'min_child_samples': 50, 'learning_rate': 0.03, 'n_estimators': 400},
    'C': {'num_leaves': 31, 'min_child_samples': 100, 'learning_rate': 0.05, 'n_estimators': 300},
}


def load_data(client, fold_id, tier_name, split_role, include_spike_features=True):
    """Load data from BigQuery."""

    view_name = 'v_trainval_lgbm_v3' if include_spike_features else 'v_trainval_lgbm_v2'

    numeric_features = ORIGINAL_NUMERIC.get(tier_name, ORIGINAL_NUMERIC['T1_MATURE']).copy()
    if include_spike_features:
        numeric_features.extend(SPIKE_FEATURES)

    feature_list = []
    for f in numeric_features:
        if f in ['is_store_closed', 'is_weekend', 'is_closure_week', 'feat_store_promo_day', 'feat_had_recent_spike']:
            feature_list.append(f)
        elif f == 'trend_idx':
            continue
        else:
            feature_list.append(f'COALESCE({f}, 0) AS {f}')

    query = f"""
    SELECT
        store_id, sku_id, date,
        y, is_store_closed,
        DATE_DIFF(date, DATE '2019-01-02', DAY) AS trend_idx,
        {', '.join(feature_list)}
    FROM `myforecastingsales.forecasting.{view_name}`
    WHERE fold_id = '{fold_id}'
      AND tier_name = '{tier_name}'
      AND split_role = '{split_role}'
    """

    print(f"Loading {split_role} for {tier_name}/{fold_id} (spikes={include_spike_features})...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df):,} rows")

    return df


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


def train_two_stage_model(train_df, val_df, features, segment, params):
    """Train production two-stage model (classifier + regressor)."""

    train_seg = train_df[train_df['segment'] == segment].copy()
    val_seg = val_df[val_df['segment'] == segment].copy()

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
    y_train = train_seg['y'].values

    # Stage 1: Classifier
    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'],
        n_estimators=min(params['n_estimators'], 500),
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))
    prob = clf.predict_proba(X_val)[:, 1]

    # Stage 2: Regressor on non-zero
    train_nz = train_seg[train_seg['y'] > 0]
    X_train_nz = train_nz[all_features]

    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'],
        min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'],
        n_estimators=min(params['n_estimators'], 500),
        reg_lambda=0.5,
        verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(X_train_nz, np.log1p(train_nz['y']))
    pred = np.expm1(reg.predict(X_val))

    # Combine with threshold
    threshold = 0.6 if segment in ['A', 'B'] else 0.7
    y_pred = np.where(prob > threshold, pred, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_seg['is_store_closed'].values == 1] = 0

    return y_pred, val_seg


def compute_metrics(val_df, y_pred):
    """Compute WFA metrics at multiple levels."""
    y_true = val_df['y'].values

    # Daily
    wmape_daily = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    wfa_daily = 100 - wmape_daily

    # Weekly store
    val_df = val_df.copy()
    val_df['y_pred'] = y_pred
    val_df['date'] = pd.to_datetime(val_df['date'])
    val_df['week'] = val_df['date'].dt.isocalendar().week
    val_df['year'] = val_df['date'].dt.year

    weekly_store = val_df.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape_weekly = 100 * np.sum(np.abs(weekly_store['y'] - weekly_store['y_pred'])) / max(np.sum(weekly_store['y']), 1)
    wfa_weekly = 100 - wmape_weekly

    return {
        'daily_wfa': wfa_daily,
        'daily_wmape': wmape_daily,
        'weekly_store_wfa': wfa_weekly,
        'weekly_store_wmape': wmape_weekly
    }


def run_experiment(client, fold_id, tier_name, include_spikes):
    """Run full experiment with all segments."""

    label = "WITH SPIKES" if include_spikes else "BASELINE"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Load data
    train_df = load_data(client, fold_id, tier_name, 'TRAIN', include_spikes)
    val_df = load_data(client, fold_id, tier_name, 'VAL', include_spikes)

    # Assign segments
    train_df = assign_segments(train_df)
    val_df = assign_segments(val_df, train_df)

    # Get features
    features = ORIGINAL_NUMERIC.get(tier_name, ORIGINAL_NUMERIC['T1_MATURE']).copy()
    if include_spikes:
        features.extend(SPIKE_FEATURES)

    print(f"  Features: {len(features)}")

    # Train per segment and collect predictions
    all_preds = []
    all_vals = []

    for segment in ['A', 'B', 'C']:
        params = SEGMENT_PARAMS[segment]
        y_pred, val_seg = train_two_stage_model(train_df, val_df, features, segment, params)

        if y_pred is not None:
            all_preds.append(y_pred)
            all_vals.append(val_seg)
            print(f"  {segment}-items: {len(val_seg):,} rows trained")

    # Combine all segments
    combined_val = pd.concat(all_vals, ignore_index=True)
    combined_pred = np.concatenate(all_preds)

    # Compute metrics
    metrics = compute_metrics(combined_val, combined_pred)

    print(f"\n  Results:")
    print(f"    Daily WFA:        {metrics['daily_wfa']:.2f}%")
    print(f"    Weekly Store WFA: {metrics['weekly_store_wfa']:.2f}%")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', default='T1_MATURE', choices=['T1_MATURE', 'T2_GROWING'])
    parser.add_argument('--fold', default='F1')
    parser.add_argument('--project', default='myforecastingsales')
    args = parser.parse_args()

    print("="*70)
    print("SPIKE FEATURE TEST - VERTEX AI")
    print("="*70)
    print(f"Tier: {args.tier}")
    print(f"Fold: {args.fold}")
    print(f"Timestamp: {datetime.now()}")
    print("="*70)

    client = bigquery.Client(project=args.project)

    # Run baseline
    baseline_metrics = run_experiment(client, args.fold, args.tier, include_spikes=False)

    # Run with spikes
    spike_metrics = run_experiment(client, args.fold, args.tier, include_spikes=True)

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    daily_change = spike_metrics['daily_wfa'] - baseline_metrics['daily_wfa']
    weekly_change = spike_metrics['weekly_store_wfa'] - baseline_metrics['weekly_store_wfa']

    print(f"\n{'Metric':<25} {'Baseline':>12} {'With Spikes':>12} {'Change':>10}")
    print("-"*60)
    print(f"{'Daily WFA':<25} {baseline_metrics['daily_wfa']:>11.2f}% {spike_metrics['daily_wfa']:>11.2f}% {daily_change:>+9.2f}pp")
    print(f"{'Weekly Store WFA':<25} {baseline_metrics['weekly_store_wfa']:>11.2f}% {spike_metrics['weekly_store_wfa']:>11.2f}% {weekly_change:>+9.2f}pp")

    print(f"\n{'='*70}")
    if daily_change > 0.5 or weekly_change > 0.5:
        print(f"VERDICT: IMPROVEMENT - Spike features help (+{max(daily_change, weekly_change):.2f}pp)")
        print("RECOMMENDATION: Add spike features to production")
    elif daily_change < -0.5 or weekly_change < -0.5:
        print(f"VERDICT: DEGRADATION - Spike features hurt ({min(daily_change, weekly_change):.2f}pp)")
        print("RECOMMENDATION: Do NOT add spike features")
    else:
        print(f"VERDICT: NO SIGNIFICANT CHANGE (within 0.5pp)")
        print("RECOMMENDATION: Spike features optional")
    print("="*70)

    # Save results
    results = {
        'tier': args.tier,
        'fold': args.fold,
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_metrics,
        'with_spikes': spike_metrics,
        'daily_change_pp': daily_change,
        'weekly_change_pp': weekly_change
    }

    output_path = f'/tmp/spike_test_results_{args.tier}_{args.fold}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
