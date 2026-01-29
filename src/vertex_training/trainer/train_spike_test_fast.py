"""
FAST Spike Feature Test - 10% Sample
=====================================
Quick validation using sampled data. No leakage - sampling is random.
"""

import os
import json
import argparse
from google.cloud import bigquery
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

ORIGINAL_NUMERIC = [
    'trend_idx', 'dow', 'week_of_year', 'month', 'is_weekend', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28', 'nz_rate_28',
    'nz_rate_7', 'roll_mean_pos_28',
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
]

SPIKE_FEATURES = [
    'feat_store_spike_pct', 'feat_store_promo_day', 'feat_seasonal_lift',
    'feat_had_recent_spike', 'feat_historical_spike_prob'
]

CAT_FEATURES = ['store_id', 'sku_id']

SEGMENT_PARAMS = {
    'A': {'num_leaves': 127, 'min_child_samples': 20, 'learning_rate': 0.03, 'n_estimators': 300},
    'B': {'num_leaves': 63, 'min_child_samples': 50, 'learning_rate': 0.03, 'n_estimators': 200},
    'C': {'num_leaves': 31, 'min_child_samples': 100, 'learning_rate': 0.05, 'n_estimators': 150},
}


def load_sampled_data(client, fold_id, tier_name, split_role, include_spikes, sample_pct=10):
    """Load SAMPLED data from BigQuery for speed."""

    view = 'v_trainval_lgbm_v3' if include_spikes else 'v_trainval_lgbm_v2'
    features = ORIGINAL_NUMERIC + (SPIKE_FEATURES if include_spikes else [])

    feature_sql = ", ".join([
        f"COALESCE({f}, 0) AS {f}" if f not in ['is_store_closed', 'is_weekend', 'is_closure_week',
            'feat_store_promo_day', 'feat_had_recent_spike', 'trend_idx'] else f
        for f in features if f != 'trend_idx'
    ])

    query = f"""
    SELECT
        store_id, sku_id, date, y, is_store_closed,
        DATE_DIFF(date, DATE '2019-01-02', DAY) AS trend_idx,
        {feature_sql}
    FROM `myforecastingsales.forecasting.{view}`
    WHERE fold_id = '{fold_id}'
      AND tier_name = '{tier_name}'
      AND split_role = '{split_role}'
      AND MOD(ABS(FARM_FINGERPRINT(CONCAT(CAST(store_id AS STRING), CAST(sku_id AS STRING)))), 100) < {sample_pct}
    """

    print(f"Loading {sample_pct}% sample of {split_role} (spikes={include_spikes})...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df):,} rows")
    return df


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
    for col in CAT_FEATURES:
        train_seg[col] = train_seg[col].astype('category')
        val_seg[col] = val_seg[col].astype('category')
    for col in available:
        train_seg[col] = pd.to_numeric(train_seg[col], errors='coerce').fillna(0)
        val_seg[col] = pd.to_numeric(val_seg[col], errors='coerce').fillna(0)

    all_features = available + CAT_FEATURES
    X_train, X_val = train_seg[all_features], val_seg[all_features]
    y_train = train_seg['y'].values

    # Classifier
    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'], min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))
    prob = clf.predict_proba(X_val)[:, 1]

    # Regressor
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


def run_test(client, fold_id, tier_name, include_spikes, sample_pct):
    label = "WITH SPIKES" if include_spikes else "BASELINE"
    print(f"\n{'='*50}\n  {label}\n{'='*50}")

    train_df = load_sampled_data(client, fold_id, tier_name, 'TRAIN', include_spikes, sample_pct)
    val_df = load_sampled_data(client, fold_id, tier_name, 'VAL', include_spikes, sample_pct)

    train_df = assign_segments(train_df)
    val_df = assign_segments(val_df, train_df)

    features = ORIGINAL_NUMERIC + (SPIKE_FEATURES if include_spikes else [])
    print(f"  Features: {len(features)}")

    all_preds, all_vals = [], []
    for seg in ['A', 'B', 'C']:
        y_pred, val_seg = train_model(train_df, val_df, features, seg, SEGMENT_PARAMS[seg])
        if y_pred is not None:
            all_preds.append(y_pred)
            all_vals.append(val_seg)
            print(f"  {seg}-items: {len(val_seg):,} rows")

    combined_val = pd.concat(all_vals, ignore_index=True)
    combined_pred = np.concatenate(all_preds)
    metrics = compute_metrics(combined_val, combined_pred)

    print(f"\n  Daily WFA:        {metrics['daily_wfa']:.2f}%")
    print(f"  Weekly Store WFA: {metrics['weekly_store_wfa']:.2f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', default='T1_MATURE')
    parser.add_argument('--fold', default='F1')
    parser.add_argument('--project', default='myforecastingsales')
    parser.add_argument('--sample', type=int, default=10, help='Sample percentage (1-100)')
    args = parser.parse_args()

    print("="*60)
    print(f"FAST SPIKE TEST ({args.sample}% SAMPLE)")
    print("="*60)
    print(f"Tier: {args.tier}, Fold: {args.fold}")
    print(f"Time: {datetime.now()}")
    print("="*60)

    client = bigquery.Client(project=args.project)

    baseline = run_test(client, args.fold, args.tier, False, args.sample)
    with_spikes = run_test(client, args.fold, args.tier, True, args.sample)

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    daily_change = with_spikes['daily_wfa'] - baseline['daily_wfa']
    weekly_change = with_spikes['weekly_store_wfa'] - baseline['weekly_store_wfa']

    print(f"\n{'Metric':<20} {'Baseline':>10} {'Spikes':>10} {'Change':>10}")
    print("-"*50)
    print(f"{'Daily WFA':<20} {baseline['daily_wfa']:>9.2f}% {with_spikes['daily_wfa']:>9.2f}% {daily_change:>+9.2f}pp")
    print(f"{'Weekly Store WFA':<20} {baseline['weekly_store_wfa']:>9.2f}% {with_spikes['weekly_store_wfa']:>9.2f}% {weekly_change:>+9.2f}pp")

    print(f"\n{'='*60}")
    if daily_change > 0.5 or weekly_change > 0.5:
        print(f"VERDICT: IMPROVEMENT (+{max(daily_change, weekly_change):.2f}pp)")
    elif daily_change < -0.5 or weekly_change < -0.5:
        print(f"VERDICT: DEGRADATION ({min(daily_change, weekly_change):.2f}pp)")
    else:
        print("VERDICT: NO SIGNIFICANT CHANGE")
    print("="*60)

    # Save results
    results = {
        'tier': args.tier, 'fold': args.fold, 'sample_pct': args.sample,
        'baseline': baseline, 'with_spikes': with_spikes,
        'daily_change_pp': daily_change, 'weekly_change_pp': weekly_change,
        'timestamp': datetime.now().isoformat()
    }
    with open('/tmp/spike_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/spike_test_results.json")


if __name__ == '__main__':
    main()
