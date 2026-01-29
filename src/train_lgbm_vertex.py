"""
Vertex AI Custom Training Script for LightGBM
Trains tier-specific models and writes predictions back to BigQuery
"""

import os
import json
import argparse
from google.cloud import bigquery
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

# Feature sets per tier
FEATURE_SETS = {
    'T1_MATURE': {
        'categorical': ['store_id', 'sku_id'],
        'numeric': [
            'trend_idx', 'dow', 'week_of_year', 'month', 'is_weekend', 'day_of_year',
            'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
            'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
            'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
            'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28', 'nz_rate_28',
            'nz_rate_7', 'roll_mean_pos_28',
            'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
            'lag_7_log1p', 'roll_mean_28_log1p', 'roll_mean_pos_28_log1p'
        ]
    },
    'T2_GROWING': {
        'categorical': ['store_id', 'sku_id'],
        'numeric': [
            'trend_idx', 'dow', 'month', 'is_weekend', 'day_of_year',
            'sin_doy', 'cos_doy',
            'is_store_closed', 'is_closure_week',
            'lag_1', 'lag_7', 'lag_14', 'lag_28',
            'roll_mean_7', 'roll_mean_28', 'nz_rate_28',
            'nz_rate_7', 'roll_mean_pos_28',
            'days_since_last_sale_asof', 'zero_run_length_asof', 'dormancy_capped',
            'lag_7_log1p', 'roll_mean_28_log1p'
        ]
    },
    'T3_COLD_START': {
        'categorical': ['store_id', 'sku_id'],
        'numeric': [
            'trend_idx', 'dow', 'month', 'is_weekend', 'day_of_year',
            'sin_doy', 'cos_doy',
            'is_store_closed', 'is_closure_week',
            'lag_7', 'lag_14',
            'roll_mean_28', 'nz_rate_28', 'roll_mean_pos_28',
            'days_since_last_sale_asof', 'dormancy_capped',
            'lag_7_log1p', 'roll_mean_pos_28_log1p'
        ]
    }
}

# LightGBM parameters per tier
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


def load_data(client, fold_id, tier_name, split_role):
    """Load data from BigQuery training view"""

    feature_set = FEATURE_SETS[tier_name]
    all_features = feature_set['categorical'] + feature_set['numeric']

    # Build query with trend_idx
    query = f"""
    SELECT
        store_id, sku_id, date,
        y, is_store_closed, dormancy_bucket, days_since_last_sale_asof,
        DATE_DIFF(date, DATE '2019-01-02', DAY) AS trend_idx,
        {', '.join([f'COALESCE({f}, 0) AS {f}' if f not in ['store_id', 'sku_id', 'trend_idx', 'is_store_closed', 'is_weekend', 'is_closure_week'] else f for f in feature_set['numeric'] if f != 'trend_idx'])}
    FROM `myforecastingsales.forecasting.v_trainval_lgbm_v2`
    WHERE fold_id = '{fold_id}'
      AND tier_name = '{tier_name}'
      AND split_role = '{split_role}'
    """

    print(f"Loading {split_role} data for {tier_name}/{fold_id}...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df):,} rows")

    return df


def prepare_features(df, tier_name):
    """Prepare features for LightGBM"""

    feature_set = FEATURE_SETS[tier_name]

    # Convert categoricals
    for col in feature_set['categorical']:
        df[col] = df[col].astype('category')

    # Get feature columns that exist in dataframe
    all_features = feature_set['categorical'] + feature_set['numeric']
    available_features = [f for f in all_features if f in df.columns]

    return df, available_features


def train_model(train_df, val_df, tier_name, features):
    """Train LightGBM model"""

    params = LGB_PARAMS[tier_name].copy()
    categorical = FEATURE_SETS[tier_name]['categorical']

    # Create datasets
    train_data = lgb.Dataset(
        train_df[features],
        label=train_df['y'],
        categorical_feature=categorical
    )
    val_data = lgb.Dataset(
        val_df[features],
        label=val_df['y'],
        categorical_feature=categorical,
        reference=train_data
    )

    # Train
    print(f"Training {tier_name} model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val']
    )

    return model


def predict_and_apply_rules(model, df, features):
    """Make predictions and apply closure override"""

    yhat_raw = model.predict(df[features])

    # Apply closure override and clamp
    yhat = np.where(
        df['is_store_closed'] == 1,
        0,
        np.maximum(yhat_raw, 0)
    )

    return yhat_raw, yhat


def compute_metrics(df):
    """Compute evaluation metrics"""

    y = df['y'].values
    yhat = df['yhat'].values

    mae = np.mean(np.abs(y - yhat))

    sum_y = np.sum(y)
    wmape = np.sum(np.abs(y - yhat)) / sum_y if sum_y > 0 else None

    zero_mask = y == 0
    zero_acc = np.mean((yhat < 0.5)[zero_mask]) if zero_mask.sum() > 0 else None

    pos_mask = y > 0
    mae_nonzero = np.mean(np.abs(y - yhat)[pos_mask]) if pos_mask.sum() > 0 else None
    bias_nonzero = np.mean((yhat - y)[pos_mask]) if pos_mask.sum() > 0 else None

    return {
        'mae': mae,
        'wmape': wmape,
        'zero_acc': zero_acc,
        'mae_nonzero': mae_nonzero,
        'bias_nonzero': bias_nonzero
    }


def write_predictions(client, df, table_id, write_disposition='WRITE_APPEND'):
    """Write predictions to BigQuery"""

    # Select only required columns
    output_cols = [
        'fold_id', 'tier_name', 'store_id', 'sku_id', 'date',
        'y', 'yhat_raw', 'yhat', 'is_store_closed', 'dormancy_bucket',
        'days_since_last_sale_asof'
    ]
    output_df = df[output_cols].copy()

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition
    )

    job = client.load_table_from_dataframe(output_df, table_id, job_config=job_config)
    job.result()
    print(f"  Wrote {len(output_df):,} rows to {table_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', required=True, choices=['T1_MATURE', 'T2_GROWING', 'T3_COLD_START'])
    parser.add_argument('--fold', required=True)
    parser.add_argument('--project', default='myforecastingsales')
    parser.add_argument('--output_table', default='myforecastingsales.forecasting.val_pred_gbdt_v2')
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)

    # Load data
    train_df = load_data(client, args.fold, args.tier, 'TRAIN')
    val_df = load_data(client, args.fold, args.tier, 'VAL')

    # Prepare features
    train_df, features = prepare_features(train_df, args.tier)
    val_df, _ = prepare_features(val_df, args.tier)

    # Train model
    model = train_model(train_df, val_df, args.tier, features)

    # Predict on validation
    val_df['yhat_raw'], val_df['yhat'] = predict_and_apply_rules(model, val_df, features)
    val_df['fold_id'] = args.fold
    val_df['tier_name'] = args.tier

    # Compute metrics
    metrics = compute_metrics(val_df)
    print(f"\nMetrics for {args.tier}/{args.fold}:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    # Write predictions
    write_predictions(client, val_df, args.output_table)

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    print(f"\nDone! Model trained and predictions saved.")


if __name__ == '__main__':
    main()
