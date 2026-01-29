"""
Train C1+B1 Model on Full Data (10M+ rows)
==========================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

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


def load_sharded_csvs(folder):
    """Load all CSVs from a folder and concatenate."""
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    print(f"Loading {len(files)} files from {folder}...")

    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        dfs.append(df)
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files...")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(full_df):,}")
    return full_df


def main():
    print("="*60)
    print("FULL DATA TRAINING: C1+B1 TWO-STAGE MODEL")
    print("="*60)
    print(f"Started: {datetime.now()}")

    # Load data
    train = load_sharded_csvs('/tmp/full_data/train')
    val = load_sharded_csvs('/tmp/full_data/val')

    # Load and merge SKU attributes
    print("\nMerging SKU attributes...")
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(lambda x: 1 if x in ['L', 'LI'] else 0)

    train['sku_id'] = train['sku_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)

    train = train.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    val = val.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')

    train['is_local'] = train['is_local'].fillna(0).astype(int)
    val['is_local'] = val['is_local'].fillna(0).astype(int)

    # Fill missing features
    for col in FEATURES:
        if col in train.columns:
            train[col] = train[col].fillna(0)
            val[col] = val[col].fillna(0)

    print(f"\nTrain: {len(train):,} rows")
    print(f"Val: {len(val):,} rows")
    print(f"Train is_local=1: {train['is_local'].sum():,}")

    # Prepare for training
    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val[col] = val[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val[FEATURES + CAT_FEATURES]
    y_val = val['y'].values
    is_closed = val['is_store_closed'].values

    # =========================================
    # Stage 1: Binary Classifier
    # =========================================
    print("\n" + "="*60)
    print("STAGE 1: Binary Classifier")
    print("="*60)

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
    print("Training classifier...")
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    print("Predicting probabilities...")
    prob_nonzero = clf_model.predict(X_val)
    print(f"Classifier done. Mean prob: {prob_nonzero.mean():.4f}")

    # =========================================
    # Stage 2: Log-Transform Regressor
    # =========================================
    print("\n" + "="*60)
    print("STAGE 2: Log-Transform Regressor")
    print("="*60)

    train_nz = train[train['y'] > 0].copy()
    train_nz['y_log'] = np.log1p(train_nz['y'])
    print(f"Non-zero training rows: {len(train_nz):,}")

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
    print("Training regressor...")
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)
    print("Regressor done.")

    # =========================================
    # Combine Predictions
    # =========================================
    print("\n" + "="*60)
    print("COMBINING PREDICTIONS (threshold=0.7)")
    print("="*60)

    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)

    y_pred = np.where(prob_nonzero > 0.7, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # =========================================
    # Metrics
    # =========================================
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

    actual_zero = (y_val == 0).astype(int)
    pred_zero = (y_pred == 0).astype(int)
    zero_accuracy = np.mean(actual_zero == pred_zero)
    zero_precision = np.sum((pred_zero == 1) & (actual_zero == 1)) / max(np.sum(pred_zero == 1), 1)
    zero_recall = np.sum((pred_zero == 1) & (actual_zero == 1)) / max(np.sum(actual_zero == 1), 1)

    print("\n" + "="*60)
    print("FINAL RESULTS - FULL DATA")
    print("="*60)
    print(f"  WMAPE: {wmape:.2f}%")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Zero Accuracy: {100*zero_accuracy:.2f}%")
    print(f"  Zero Precision: {100*zero_precision:.2f}%")
    print(f"  Zero Recall: {100*zero_recall:.2f}%")

    # Save models
    print("\nSaving models...")
    os.makedirs('/tmp/full_models', exist_ok=True)
    clf_model.save_model('/tmp/full_models/clf_model_full.txt')
    reg_model.save_model('/tmp/full_models/reg_model_full.txt')

    metrics = {
        'wmape': wmape,
        'mae': mae,
        'rmse': rmse,
        'zero_accuracy': zero_accuracy,
        'zero_precision': zero_precision,
        'zero_recall': zero_recall,
        'n_train': len(train),
        'n_train_nonzero': len(train_nz),
        'n_val': len(val),
        'threshold': 0.7,
        'timestamp': datetime.now().isoformat()
    }

    with open('/tmp/full_models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels saved to /tmp/full_models/")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
