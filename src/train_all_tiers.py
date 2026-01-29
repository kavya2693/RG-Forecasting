"""
Train C1+B1 Model on All Tiers (T1, T2, T3)
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

    # Handle WMAPE for cases where sum(y_val) might be 0
    if np.sum(y_val) > 0:
        wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
    else:
        wmape = np.nan

    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

    # Bias
    bias = np.mean(y_pred - y_val)
    if np.mean(y_val) > 0:
        bias_pct = 100 * bias / np.mean(y_val)
    else:
        bias_pct = np.nan

    # R-squared
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

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
    print("TRAINING C1+B1 MODEL ON ALL TIERS")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load SKU attributes
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    tiers = [
        {'name': 'T1_MATURE', 'train_folder': '/tmp/full_data/train', 'val_folder': '/tmp/full_data/val'},
        {'name': 'T2_GROWING', 'train_folder': '/tmp/t2_data/train', 'val_folder': '/tmp/t2_data/val'},
        {'name': 'T3_COLD_START', 'train_folder': '/tmp/t3_data/train', 'val_folder': '/tmp/t3_data/val'},
    ]

    all_metrics = {}

    for tier in tiers:
        print(f"\n{'=' * 70}")
        print(f"TIER: {tier['name']}")
        print("=" * 70)

        # Check if data exists locally, if not download
        if not os.path.exists(tier['train_folder']) or len(glob.glob(f"{tier['train_folder']}/*.csv")) == 0:
            if tier['name'] == 'T2_GROWING':
                os.makedirs('/tmp/t2_data/train', exist_ok=True)
                os.makedirs('/tmp/t2_data/val', exist_ok=True)
                print(f"  Downloading T2 data from GCS...")
                os.system("gsutil -m cp 'gs://myforecastingsales-data/training_data/t2_g1_train/*.csv' /tmp/t2_data/train/")
                os.system("gsutil -m cp 'gs://myforecastingsales-data/training_data/t2_g1_val/*.csv' /tmp/t2_data/val/")
            elif tier['name'] == 'T3_COLD_START':
                os.makedirs('/tmp/t3_data/train', exist_ok=True)
                os.makedirs('/tmp/t3_data/val', exist_ok=True)
                print(f"  Downloading T3 data from GCS...")
                os.system("gsutil -m cp 'gs://myforecastingsales-data/training_data/t3_c1_train/*.csv' /tmp/t3_data/train/")
                os.system("gsutil -m cp 'gs://myforecastingsales-data/training_data/t3_c1_val/*.csv' /tmp/t3_data/val/")

        # Load data
        train = load_sharded_csvs(tier['train_folder'])
        val = load_sharded_csvs(tier['val_folder'])

        # Train and evaluate
        print(f"  Training {tier['name']} model...")
        metrics, clf_model, reg_model = train_and_evaluate(train.copy(), val.copy(), sku_attr.copy())

        all_metrics[tier['name']] = metrics

        print(f"\n  {tier['name']} Results:")
        print(f"    WMAPE: {metrics['wmape']:.2f}%" if not np.isnan(metrics['wmape']) else "    WMAPE: N/A")
        print(f"    MAE: {metrics['mae']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    R²: {metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "    R²: N/A")
        print(f"    Bias: {metrics['bias']:.4f} ({metrics['bias_pct']:.2f}%)" if not np.isnan(metrics['bias_pct']) else f"    Bias: {metrics['bias']:.4f}")
        print(f"    Zero Accuracy: {100*metrics['zero_accuracy']:.2f}%")
        print(f"    Zero F1: {100*metrics['zero_f1']:.2f}%")
        print(f"    Actual Zeros: {metrics['pct_zeros_actual']:.1f}%")
        print(f"    Predicted Zeros: {metrics['pct_zeros_pred']:.1f}%")

        # Save tier model
        os.makedirs(f'/tmp/tier_models/{tier["name"].lower()}', exist_ok=True)
        clf_model.save_model(f'/tmp/tier_models/{tier["name"].lower()}/clf_model.txt')
        reg_model.save_model(f'/tmp/tier_models/{tier["name"].lower()}/reg_model.txt')

    # Summary Table
    print("\n" + "=" * 70)
    print("ALL TIERS SUMMARY")
    print("=" * 70)

    print("\n" + "-" * 100)
    print(f"{'Metric':<20} {'T1_MATURE':>20} {'T2_GROWING':>20} {'T3_COLD_START':>20}")
    print("-" * 100)

    for metric in ['wmape', 'mae', 'rmse', 'r2', 'bias_pct', 'zero_accuracy', 'zero_f1', 'pct_zeros_actual']:
        vals = []
        for tier_name in ['T1_MATURE', 'T2_GROWING', 'T3_COLD_START']:
            v = all_metrics[tier_name][metric]
            if np.isnan(v):
                vals.append("N/A")
            elif metric in ['wmape', 'bias_pct', 'pct_zeros_actual']:
                vals.append(f"{v:.2f}%")
            elif metric in ['zero_accuracy', 'zero_f1']:
                vals.append(f"{100*v:.2f}%")
            else:
                vals.append(f"{v:.4f}")

        print(f"{metric:<20} {vals[0]:>20} {vals[1]:>20} {vals[2]:>20}")

    print("-" * 100)

    # Save all metrics
    with open('/tmp/tier_models/all_tiers_metrics.json', 'w') as f:
        # Convert nan to None for JSON serialization
        serializable_metrics = {}
        for tier, metrics in all_metrics.items():
            serializable_metrics[tier] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)

    print(f"\nMetrics saved to /tmp/tier_models/all_tiers_metrics.json")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
