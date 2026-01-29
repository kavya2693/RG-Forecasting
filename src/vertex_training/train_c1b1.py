"""
Vertex AI Training Script: C1+B1 Two-Stage Model with Log Transform
===================================================================
- Stage 1: Binary classifier (zero vs non-zero)
- Stage 2: Log-transform regressor (trained on log(1+y) for y>0)
- Includes is_local SKU attribute
"""

import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from google.cloud import storage, bigquery
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
    'is_local'  # SKU attribute
]
CAT_FEATURES = ['store_id', 'sku_id']


def load_data_from_gcs(bucket_name, train_path, val_path, sku_attr_path):
    """Load training and validation data from GCS."""
    print(f"Loading data from GCS bucket: {bucket_name}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download files locally
    os.makedirs('/tmp/data', exist_ok=True)

    print(f"  Downloading {train_path}...")
    bucket.blob(train_path).download_to_filename('/tmp/data/train.csv')

    print(f"  Downloading {val_path}...")
    bucket.blob(val_path).download_to_filename('/tmp/data/val.csv')

    print(f"  Downloading {sku_attr_path}...")
    bucket.blob(sku_attr_path).download_to_filename('/tmp/data/sku_attr.csv')

    # Load CSVs
    train = pd.read_csv('/tmp/data/train.csv')
    val = pd.read_csv('/tmp/data/val.csv')
    sku_attr = pd.read_csv('/tmp/data/sku_attr.csv')

    print(f"  Train: {len(train):,} rows")
    print(f"  Val: {len(val):,} rows")

    return train, val, sku_attr


def prepare_data(train, val, sku_attr):
    """Prepare data with SKU attributes and proper types."""
    print("Preparing data...")

    # Create is_local feature from SKU attributes
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(
        lambda x: 1 if x in ['L', 'LI'] else 0
    )

    # Merge with train and val
    train['sku_id'] = train['sku_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)

    train = train.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    val = val.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')

    # Fill missing
    train['is_local'] = train['is_local'].fillna(0).astype(int)
    val['is_local'] = val['is_local'].fillna(0).astype(int)

    # Fill NaN in features
    for col in FEATURES:
        if col in train.columns:
            train[col] = train[col].fillna(0)
            val[col] = val[col].fillna(0)

    print(f"  Train with is_local: {train['is_local'].sum():,} local SKUs")

    return train, val


def train_c1b1_model(train, val, threshold=0.7):
    """Train C1+B1 Two-Stage model."""
    print("\n" + "="*60)
    print("TRAINING C1+B1 TWO-STAGE MODEL")
    print("="*60)

    train = train.copy()
    val_copy = val.copy()

    # Create binary target
    train['y_binary'] = (train['y'] > 0).astype(int)

    # Convert categoricals
    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val_copy[FEATURES + CAT_FEATURES]
    y_val = val_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    # =========================================
    # Stage 1: Binary Classifier
    # =========================================
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

    train_clf = lgb.Dataset(
        X_train,
        label=train['y_binary'],
        categorical_feature=CAT_FEATURES
    )

    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(X_val)

    print(f"  Classifier trained. AUC on validation will be computed.")

    # =========================================
    # Stage 2: Log-Transform Regressor
    # =========================================
    print("\nStage 2: Training Log-Transform Regressor on non-zeros...")

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

    train_reg = lgb.Dataset(
        X_train_nz,
        label=train_nz['y_log'],
        categorical_feature=CAT_FEATURES
    )

    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)

    print(f"  Regressor trained on {len(train_nz):,} non-zero rows.")

    # =========================================
    # Combine Predictions
    # =========================================
    print(f"\nCombining predictions with threshold={threshold}...")

    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)

    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # =========================================
    # Calculate Metrics
    # =========================================
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

    # Zero classification metrics
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

    return clf_model, reg_model, metrics, y_pred


def save_models_to_gcs(clf_model, reg_model, metrics, bucket_name, output_path):
    """Save trained models and metrics to GCS."""
    print(f"\nSaving models to GCS: gs://{bucket_name}/{output_path}/")

    os.makedirs('/tmp/models', exist_ok=True)

    # Save models locally first
    clf_model.save_model('/tmp/models/clf_model.txt')
    reg_model.save_model('/tmp/models/reg_model.txt')

    with open('/tmp/models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    bucket.blob(f"{output_path}/clf_model.txt").upload_from_filename('/tmp/models/clf_model.txt')
    bucket.blob(f"{output_path}/reg_model.txt").upload_from_filename('/tmp/models/reg_model.txt')
    bucket.blob(f"{output_path}/metrics.json").upload_from_filename('/tmp/models/metrics.json')

    print("  Models saved successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='GCS bucket name')
    parser.add_argument('--train-path', type=str, required=True, help='Path to training data in GCS')
    parser.add_argument('--val-path', type=str, required=True, help='Path to validation data in GCS')
    parser.add_argument('--sku-attr-path', type=str, required=True, help='Path to SKU attributes in GCS')
    parser.add_argument('--output-path', type=str, required=True, help='Output path in GCS for models')
    parser.add_argument('--threshold', type=float, default=0.7, help='Probability threshold for zero classification')

    args = parser.parse_args()

    print("="*60)
    print("VERTEX AI TRAINING: C1+B1 TWO-STAGE MODEL")
    print("="*60)
    print(f"Bucket: {args.bucket}")
    print(f"Train: {args.train_path}")
    print(f"Val: {args.val_path}")
    print(f"SKU Attr: {args.sku_attr_path}")
    print(f"Output: {args.output_path}")
    print(f"Threshold: {args.threshold}")

    # Load data
    train, val, sku_attr = load_data_from_gcs(
        args.bucket,
        args.train_path,
        args.val_path,
        args.sku_attr_path
    )

    # Prepare data
    train, val = prepare_data(train, val, sku_attr)

    # Train model
    clf_model, reg_model, metrics, predictions = train_c1b1_model(
        train, val, threshold=args.threshold
    )

    # Save models
    save_models_to_gcs(clf_model, reg_model, metrics, args.bucket, args.output_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
