"""
Validate C1+B1 Model on All Folds (F1, F2, F3)
==============================================
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
    print(f"  Loading {len(files)} files from {folder}...")

    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        dfs.append(df)
        if (i + 1) % 20 == 0:
            print(f"    Loaded {i + 1}/{len(files)} files...")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"    Total rows: {len(full_df):,}")
    return full_df


def train_and_evaluate(train, val, sku_attr, threshold=0.7):
    """Train C1+B1 and return metrics."""

    # Merge SKU attributes
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

    # Prepare for training
    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val[col] = val[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val[FEATURES + CAT_FEATURES]
    y_val = val['y'].values
    is_closed = val['is_store_closed'].values

    # Stage 1: Binary Classifier
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

    # Stage 2: Log-Transform Regressor
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

    # Combine Predictions
    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)

    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # Calculate Metrics
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

    # Bias
    bias = np.mean(y_pred - y_val)
    bias_pct = 100 * bias / np.mean(y_val)

    # R-squared
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Zero classification metrics
    actual_zero = (y_val == 0).astype(int)
    pred_zero = (y_pred == 0).astype(int)
    zero_accuracy = np.mean(actual_zero == pred_zero)
    zero_precision = np.sum((pred_zero == 1) & (actual_zero == 1)) / max(np.sum(pred_zero == 1), 1)
    zero_recall = np.sum((pred_zero == 1) & (actual_zero == 1)) / max(np.sum(actual_zero == 1), 1)
    zero_f1 = 2 * zero_precision * zero_recall / max(zero_precision + zero_recall, 1e-10)

    metrics = {
        'wmape': wmape,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'bias': bias,
        'bias_pct': bias_pct,
        'zero_accuracy': zero_accuracy,
        'zero_precision': zero_precision,
        'zero_recall': zero_recall,
        'zero_f1': zero_f1,
        'n_train': len(train),
        'n_train_nonzero': len(train_nz),
        'n_val': len(val),
        'pct_zeros_actual': 100 * np.sum(actual_zero) / len(y_val),
        'pct_zeros_pred': 100 * np.sum(pred_zero) / len(y_pred)
    }

    return metrics, clf_model, reg_model


def main():
    print("=" * 70)
    print("CROSS-FOLD VALIDATION: C1+B1 TWO-STAGE MODEL")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load SKU attributes
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    folds = ['f1', 'f2', 'f3']
    all_metrics = {}

    for fold in folds:
        print(f"\n{'=' * 70}")
        print(f"FOLD: {fold.upper()}")
        print("=" * 70)

        train_folder = f'/tmp/full_data/{fold}_train' if fold == 'f1' else f'/tmp/{fold}_data/train'
        val_folder = f'/tmp/full_data/{fold}_val' if fold == 'f1' else f'/tmp/{fold}_data/val'

        # Check if data exists locally, if not download
        if fold != 'f1':
            os.makedirs(f'/tmp/{fold}_data/train', exist_ok=True)
            os.makedirs(f'/tmp/{fold}_data/val', exist_ok=True)

            print(f"  Downloading {fold} data from GCS...")
            os.system(f"gsutil -m cp 'gs://myforecastingsales-data/training_data/{fold}_train/*.csv' /tmp/{fold}_data/train/")
            os.system(f"gsutil -m cp 'gs://myforecastingsales-data/training_data/{fold}_val/*.csv' /tmp/{fold}_data/val/")

            train_folder = f'/tmp/{fold}_data/train'
            val_folder = f'/tmp/{fold}_data/val'
        else:
            # F1 already in /tmp/full_data
            train_folder = '/tmp/full_data/train'
            val_folder = '/tmp/full_data/val'

        # Load data
        train = load_sharded_csvs(train_folder)
        val = load_sharded_csvs(val_folder)

        # Train and evaluate
        print(f"  Training {fold.upper()} model...")
        metrics, clf_model, reg_model = train_and_evaluate(train.copy(), val.copy(), sku_attr.copy())

        all_metrics[fold] = metrics

        print(f"\n  {fold.upper()} Results:")
        print(f"    WMAPE: {metrics['wmape']:.2f}%")
        print(f"    MAE: {metrics['mae']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    R²: {metrics['r2']:.4f}")
        print(f"    Bias: {metrics['bias']:.4f} ({metrics['bias_pct']:.2f}%)")
        print(f"    Zero Accuracy: {100*metrics['zero_accuracy']:.2f}%")
        print(f"    Zero F1: {100*metrics['zero_f1']:.2f}%")

        # Save fold model
        os.makedirs(f'/tmp/fold_models/{fold}', exist_ok=True)
        clf_model.save_model(f'/tmp/fold_models/{fold}/clf_model.txt')
        reg_model.save_model(f'/tmp/fold_models/{fold}/reg_model.txt')

    # Summary Table
    print("\n" + "=" * 70)
    print("CROSS-FOLD VALIDATION SUMMARY")
    print("=" * 70)

    print("\n" + "-" * 90)
    print(f"{'Metric':<20} {'F1':>15} {'F2':>15} {'F3':>15} {'Mean':>15}")
    print("-" * 90)

    for metric in ['wmape', 'mae', 'rmse', 'r2', 'bias_pct', 'zero_accuracy', 'zero_f1']:
        vals = [all_metrics[f][metric] for f in folds]
        mean_val = np.mean(vals)

        if metric in ['wmape', 'bias_pct']:
            print(f"{metric.upper():<20} {vals[0]:>14.2f}% {vals[1]:>14.2f}% {vals[2]:>14.2f}% {mean_val:>14.2f}%")
        elif metric in ['zero_accuracy', 'zero_f1']:
            print(f"{metric:<20} {100*vals[0]:>14.2f}% {100*vals[1]:>14.2f}% {100*vals[2]:>14.2f}% {100*mean_val:>14.2f}%")
        else:
            print(f"{metric.upper():<20} {vals[0]:>15.4f} {vals[1]:>15.4f} {vals[2]:>15.4f} {mean_val:>15.4f}")

    print("-" * 90)

    # Save all metrics
    with open('/tmp/fold_models/cross_fold_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to /tmp/fold_models/cross_fold_metrics.json")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
