"""
Vertex AI Custom Training Script - WITH SPIKE FEATURES
=======================================================
Tests production model + inferred promotional features from spike detection.

Spike Features Added:
  - feat_store_spike_pct: % of SKUs spiking in store today
  - feat_store_promo_day: Binary store-wide promotional event
  - feat_seasonal_lift: Week-level seasonal multiplier
  - feat_had_recent_spike: Spike in last 7 days
  - feat_historical_spike_prob: Historical spike probability
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
# FEATURE SETS - ORIGINAL + SPIKE FEATURES
# ==============================================================================

# Original features (production baseline)
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
    'T3_COLD_START': [
        'trend_idx', 'dow', 'month', 'is_weekend', 'day_of_year',
        'sin_doy', 'cos_doy',
        'is_store_closed', 'is_closure_week',
        'lag_7', 'lag_14',
        'roll_mean_28', 'nz_rate_28', 'roll_mean_pos_28',
        'days_since_last_sale_asof', 'dormancy_capped',
        'lag_7_log1p', 'roll_mean_pos_28_log1p'
    ]
}

# Spike features (inferred promotional signals)
SPIKE_FEATURES = [
    'feat_store_spike_pct',       # % of SKUs spiking in store
    'feat_store_promo_day',       # Binary store-wide event
    'feat_seasonal_lift',         # Week-level seasonality multiplier
    'feat_had_recent_spike',      # Had spike in last 7 days
    'feat_historical_spike_prob'  # Historical spike probability
]

CAT_FEATURES = ['store_id', 'sku_id']

# LightGBM parameters (same as production)
LGB_PARAMS = {
    'T1_MATURE': {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.2,
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'num_iterations': 200,
        'early_stopping_rounds': 20,
        'verbose': -1
    },
    'T2_GROWING': {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.2,
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'num_iterations': 150,
        'early_stopping_rounds': 20,
        'verbose': -1
    },
    'T3_COLD_START': {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 15,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l2': 2.0,
        'num_iterations': 100,
        'early_stopping_rounds': 15,
        'verbose': -1
    }
}


def load_data(client, fold_id, tier_name, split_role, include_spike_features=True):
    """Load data from BigQuery training view with optional spike features."""

    # Use v3 view if spike features, else v2
    view_name = 'v_trainval_lgbm_v3' if include_spike_features else 'v_trainval_lgbm_v2'

    numeric_features = ORIGINAL_NUMERIC[tier_name].copy()
    if include_spike_features:
        numeric_features.extend(SPIKE_FEATURES)

    # Build feature list for query
    feature_list = []
    for f in numeric_features:
        if f in ['is_store_closed', 'is_weekend', 'is_closure_week', 'feat_store_promo_day', 'feat_had_recent_spike']:
            feature_list.append(f)
        elif f == 'trend_idx':
            continue  # Computed separately
        else:
            feature_list.append(f'COALESCE({f}, 0) AS {f}')

    query = f"""
    SELECT
        store_id, sku_id, date,
        y, is_store_closed, dormancy_bucket, days_since_last_sale_asof,
        DATE_DIFF(date, DATE '2019-01-02', DAY) AS trend_idx,
        {', '.join(feature_list)}
    FROM `myforecastingsales.forecasting.{view_name}`
    WHERE fold_id = '{fold_id}'
      AND tier_name = '{tier_name}'
      AND split_role = '{split_role}'
    """

    print(f"Loading {split_role} data for {tier_name}/{fold_id} (spike_features={include_spike_features})...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df):,} rows")

    return df


def prepare_features(df, tier_name, include_spike_features=True):
    """Prepare features for LightGBM."""

    numeric_features = ORIGINAL_NUMERIC[tier_name].copy()
    if include_spike_features:
        numeric_features.extend(SPIKE_FEATURES)

    # Convert categoricals
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    # Get available features
    all_features = CAT_FEATURES + numeric_features
    available_features = [f for f in all_features if f in df.columns]

    return df, available_features


def train_model(train_df, val_df, tier_name, features):
    """Train LightGBM model."""

    params = LGB_PARAMS[tier_name].copy()

    train_data = lgb.Dataset(
        train_df[features],
        label=train_df['y'],
        categorical_feature=CAT_FEATURES
    )
    val_data = lgb.Dataset(
        val_df[features],
        label=val_df['y'],
        categorical_feature=CAT_FEATURES,
        reference=train_data
    )

    print(f"Training {tier_name} model with {len(features)} features...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val']
    )

    return model


def predict_and_apply_rules(model, df, features):
    """Make predictions and apply closure override."""

    yhat_raw = model.predict(df[features])

    yhat = np.where(
        df['is_store_closed'] == 1,
        0,
        np.maximum(yhat_raw, 0)
    )

    return yhat_raw, yhat


def compute_metrics(df, label=''):
    """Compute detailed evaluation metrics."""

    y = df['y'].values
    yhat = df['yhat'].values

    mae = np.mean(np.abs(y - yhat))
    sum_y = np.sum(y)
    wmape = np.sum(np.abs(y - yhat)) / sum_y if sum_y > 0 else None
    wfa = 100 - (wmape * 100) if wmape is not None else None

    # Zero accuracy
    zero_mask = y == 0
    zero_acc = np.mean((yhat < 0.5)[zero_mask]) if zero_mask.sum() > 0 else None

    # MAE on non-zero
    pos_mask = y > 0
    mae_nonzero = np.mean(np.abs(y - yhat)[pos_mask]) if pos_mask.sum() > 0 else None
    bias_nonzero = np.mean((yhat - y)[pos_mask]) if pos_mask.sum() > 0 else None

    # Weekly store aggregation
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year

    weekly_store = df.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'yhat': 'sum'}).reset_index()
    weekly_wmape = np.sum(np.abs(weekly_store['y'] - weekly_store['yhat'])) / np.sum(weekly_store['y'])
    weekly_wfa = 100 - (weekly_wmape * 100)

    return {
        'label': label,
        'mae': mae,
        'wmape': wmape,
        'wfa': wfa,
        'weekly_store_wfa': weekly_wfa,
        'zero_acc': zero_acc,
        'mae_nonzero': mae_nonzero,
        'bias_nonzero': bias_nonzero
    }


def run_comparison(client, fold_id, tier_name, output_table):
    """Run comparison between baseline and spike features."""

    results = {}

    for include_spikes, label in [(False, 'baseline'), (True, 'with_spikes')]:
        print(f"\n{'='*60}")
        print(f"  {label.upper()}: spike_features={include_spikes}")
        print('='*60)

        # Load data
        train_df = load_data(client, fold_id, tier_name, 'TRAIN', include_spikes)
        val_df = load_data(client, fold_id, tier_name, 'VAL', include_spikes)

        # Prepare features
        train_df, features = prepare_features(train_df, tier_name, include_spikes)
        val_df, _ = prepare_features(val_df, tier_name, include_spikes)

        print(f"  Features: {len(features)}")
        if include_spikes:
            spike_feats_present = [f for f in SPIKE_FEATURES if f in features]
            print(f"  Spike features: {spike_feats_present}")

        # Train
        model = train_model(train_df, val_df, tier_name, features)

        # Predict
        val_df['yhat_raw'], val_df['yhat'] = predict_and_apply_rules(model, val_df, features)
        val_df['fold_id'] = fold_id
        val_df['tier_name'] = tier_name

        # Metrics
        metrics = compute_metrics(val_df, label)
        results[label] = metrics

        print(f"\n  Results ({label}):")
        print(f"    Daily WFA:        {metrics['wfa']:.2f}%")
        print(f"    Weekly Store WFA: {metrics['weekly_store_wfa']:.2f}%")
        print(f"    Zero Accuracy:    {metrics['zero_acc']*100:.2f}%")
        print(f"    MAE (non-zero):   {metrics['mae_nonzero']:.2f}")
        print(f"    Bias (non-zero):  {metrics['bias_nonzero']:.2f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        print(f"\n  Top 10 features:")
        print(importance.head(10).to_string(index=False))

        # Write predictions if with_spikes
        if include_spikes:
            output_cols = [
                'fold_id', 'tier_name', 'store_id', 'sku_id', 'date',
                'y', 'yhat_raw', 'yhat', 'is_store_closed', 'dormancy_bucket',
                'days_since_last_sale_asof'
            ]
            output_df = val_df[output_cols].copy()
            job_config = bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')
            job = client.load_table_from_dataframe(output_df, output_table, job_config=job_config)
            job.result()
            print(f"\n  Wrote {len(output_df):,} predictions to {output_table}")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON: BASELINE vs WITH SPIKE FEATURES")
    print('='*70)

    base = results['baseline']
    spike = results['with_spikes']

    daily_change = spike['wfa'] - base['wfa']
    weekly_change = spike['weekly_store_wfa'] - base['weekly_store_wfa']

    print(f"\n  Metric                  Baseline    +Spikes     Change")
    print(f"  " + "-"*55)
    print(f"  Daily WFA              {base['wfa']:>8.2f}%   {spike['wfa']:>8.2f}%   {daily_change:>+7.2f}pp")
    print(f"  Weekly Store WFA       {base['weekly_store_wfa']:>8.2f}%   {spike['weekly_store_wfa']:>8.2f}%   {weekly_change:>+7.2f}pp")
    print(f"  Zero Accuracy          {base['zero_acc']*100:>8.2f}%   {spike['zero_acc']*100:>8.2f}%")
    print(f"  MAE (non-zero)         {base['mae_nonzero']:>8.2f}    {spike['mae_nonzero']:>8.2f}")

    print(f"\n  VERDICT: ", end='')
    if daily_change > 0.5 or weekly_change > 0.5:
        print(f"IMPROVEMENT - spike features help (+{max(daily_change, weekly_change):.2f}pp)")
    elif daily_change < -0.5 or weekly_change < -0.5:
        print(f"DEGRADATION - spike features hurt ({min(daily_change, weekly_change):.2f}pp)")
    else:
        print(f"NO SIGNIFICANT CHANGE (within 0.5pp)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', required=True, choices=['T1_MATURE', 'T2_GROWING', 'T3_COLD_START'])
    parser.add_argument('--fold', required=True)
    parser.add_argument('--project', default='myforecastingsales')
    parser.add_argument('--output_table', default='myforecastingsales.forecasting.val_pred_spike_test')
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)

    print("="*70)
    print("SPIKE FEATURE TEST: Production vs Production + Spike Features")
    print("="*70)
    print(f"Tier: {args.tier}")
    print(f"Fold: {args.fold}")
    print(f"Spike features: {SPIKE_FEATURES}")
    print("="*70)

    results = run_comparison(client, args.fold, args.tier, args.output_table)

    # Save results
    output_path = f'/tmp/spike_test_{args.tier}_{args.fold}.json'
    with open(output_path, 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
