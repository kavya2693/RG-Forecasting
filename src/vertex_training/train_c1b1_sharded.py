"""
Vertex AI Training Script: C1+B1 Two-Stage Model with Log Transform
===================================================================
Handles sharded CSV input from GCS
"""

import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from google.cloud import storage
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Feature configuration
FEATURES = [
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_7', 'nz_rate_28', 'roll_mean_pos_28',
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
    'is_local'
]
CAT_FEATURES = ['store_id', 'sku_id']


def load_sharded_data(bucket_name, prefix):
    """Load all CSV shards from a GCS prefix."""
    print(f"Loading sharded data from gs://{bucket_name}/{prefix}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    csv_blobs = [b for b in blobs if b.name.endswith('.csv')]

    print(f"  Found {len(csv_blobs)} CSV files")

    os.makedirs('/tmp/data', exist_ok=True)

    dfs = []
    for i, blob in enumerate(csv_blobs):
        local_path = f'/tmp/data/shard_{i:03d}.csv'
        blob.download_to_filename(local_path)
        df = pd.read_csv(local_path)
        dfs.append(df)

        if (i + 1) % 10 == 0:
            print(f"    Loaded {i + 1}/{len(csv_blobs)} files...")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(full_df):,}")

    return full_df


def prepare_data(train, val, sku_attr):
    """Prepare data with SKU attributes."""
    print("Preparing data...")

    # Create is_local feature
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(
        lambda x: 1 if x in ['L', 'LI'] else 0
    )

    train['sku_id'] = train['sku_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)

    train = train.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    val = val.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')

    train['is_local'] = train['is_local'].fillna(0).astype(int)
    val['is_local'] = val['is_local'].fillna(0).astype(int)

    for col in FEATURES:
        if col in train.columns:
            train[col] = train[col].fillna(0)
            val[col] = val[col].fillna(0)

    print(f"  Data prepared. Train: {len(train):,}, Val: {len(val):,}")

    return train, val


def train_c1b1_model(train, val, threshold=0.7):
    """Train C1+B1 Two-Stage model."""
    print("\n" + "="*60)
    print("TRAINING C1+B1 TWO-STAGE MODEL")
    print("="*60)

    train = train.copy()
    val_copy = val.copy()

    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val_copy[FEATURES + CAT_FEATURES]
    y_val = val_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    # Stage 1: Binary Classifier
    print("\nStage 1: Training Binary Classifier...")
    params_clf = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 100,
        'verbose': -1,
        'n_jobs': -1
    }

    train_clf = lgb.Dataset(X_train, label=train['y_binary'], categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(X_val)
    print(f"  Classifier trained.")

    # Stage 2: Log-Transform Regressor
    print("\nStage 2: Training Log-Transform Regressor...")
    train_nz = train[train['y'] > 0].copy()
    train_nz['y_log'] = np.log1p(train_nz['y'])

    params_reg = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 30,
        'lambda_l2': 1.5,
        'verbose': -1,
        'n_jobs': -1
    }

    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    train_reg = lgb.Dataset(X_train_nz, label=train_nz['y_log'], categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)
    print(f"  Regressor trained on {len(train_nz):,} non-zero rows.")

    # Combine Predictions
    print(f"\nCombining predictions with threshold={threshold}...")
    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)

    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # Calculate Metrics
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

    actual_zero = (y_val == 0).astype(int)
    pred_zero = (y_pred == 0).astype(int)
    zero_accuracy = np.mean(actual_zero == pred_zero)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  WMAPE: {wmape:.2f}%")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Zero Accuracy: {100*zero_accuracy:.2f}%")

    metrics = {
        'wmape': wmape,
        'mae': mae,
        'rmse': rmse,
        'zero_accuracy': zero_accuracy,
        'threshold': threshold,
        'n_train': len(train),
        'n_train_nonzero': len(train_nz),
        'n_val': len(val_copy),
        'timestamp': datetime.now().isoformat()
    }

    return clf_model, reg_model, metrics


def save_models(clf_model, reg_model, metrics, bucket_name, output_path):
    """Save models to GCS."""
    print(f"\nSaving models to gs://{bucket_name}/{output_path}/")

    os.makedirs('/tmp/models', exist_ok=True)

    clf_model.save_model('/tmp/models/clf_model.txt')
    reg_model.save_model('/tmp/models/reg_model.txt')

    with open('/tmp/models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    bucket.blob(f"{output_path}/clf_model.txt").upload_from_filename('/tmp/models/clf_model.txt')
    bucket.blob(f"{output_path}/reg_model.txt").upload_from_filename('/tmp/models/reg_model.txt')
    bucket.blob(f"{output_path}/metrics.json").upload_from_filename('/tmp/models/metrics.json')

    print("  Models saved!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, default='myforecastingsales-data')
    parser.add_argument('--train-prefix', type=str, default='training_data/f1_train/')
    parser.add_argument('--val-prefix', type=str, default='training_data/f1_val/')
    parser.add_argument('--sku-attr-path', type=str, default='training_data/sku_list_attribute.csv')
    parser.add_argument('--output-path', type=str, default='models/c1b1_f1')
    parser.add_argument('--threshold', type=float, default=0.7)

    args = parser.parse_args()

    print("="*60)
    print("VERTEX AI TRAINING: C1+B1 TWO-STAGE MODEL")
    print("="*60)
    print(f"Bucket: {args.bucket}")
    print(f"Train prefix: {args.train_prefix}")
    print(f"Val prefix: {args.val_prefix}")
    print(f"Output: {args.output_path}")

    # Load data
    train = load_sharded_data(args.bucket, args.train_prefix)
    val = load_sharded_data(args.bucket, args.val_prefix)

    # Load SKU attributes
    client = storage.Client()
    bucket = client.bucket(args.bucket)
    bucket.blob(args.sku_attr_path).download_to_filename('/tmp/data/sku_attr.csv')
    sku_attr = pd.read_csv('/tmp/data/sku_attr.csv')

    # Prepare data
    train, val = prepare_data(train, val, sku_attr)

    # Train model
    clf_model, reg_model, metrics = train_c1b1_model(train, val, threshold=args.threshold)

    # Save models
    save_models(clf_model, reg_model, metrics, args.bucket, args.output_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
