"""
Run All 4 Improvement Approaches Locally
=========================================
D3: Ensemble Folds (baseline from BigQuery)
C1: Log Transform Target
B1: Two-Stage Model
B3: Per-Store Models
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# File paths
TRAIN_FILE = '/tmp/c1_data/f1_train_500k.csv'
VAL_FILE = '/tmp/c1_data/f1_val_500k.csv'

FEATURES = [
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_7', 'nz_rate_28', 'roll_mean_pos_28',
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof'
]
CAT_FEATURES = ['store_id', 'sku_id']


def load_data():
    """Load train and validation data."""
    print("Loading data...")
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)

    # Fill NaN with 0 for numeric columns
    for col in FEATURES:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)

    print(f"  Train: {len(train):,} rows")
    print(f"  Val: {len(val):,} rows")
    return train, val


def approach_c1_log_transform(train, val):
    """C1: Log Transform Target - train with log(1+y)."""
    print("\n" + "="*60)
    print("APPROACH C1: LOG TRANSFORM TARGET")
    print("="*60)

    # Log transform target
    train = train.copy()
    train['y_log'] = np.log1p(train['y'])

    # Prepare features
    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val[col] = val[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    y_train = train['y_log']
    X_val = val[FEATURES + CAT_FEATURES]
    y_val = val['y'].values
    is_closed = val['is_store_closed'].values

    # Train LightGBM
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l2': 2.0,
        'verbose': -1,
        'n_jobs': -1
    }

    print("Training LightGBM with log transform...")
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_data, num_boost_round=300)

    # Predict and inverse transform
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # Evaluate
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    print(f"C1 Log Transform MAE: {mae:.4f}")
    print(f"C1 Log Transform WMAPE: {wmape:.2f}%")

    return {'approach': 'C1_log_transform', 'wmape': wmape, 'mae': mae}


def approach_b1_two_stage(train, val):
    """B1: Two-Stage Model - classify zero/non-zero, then predict value."""
    print("\n" + "="*60)
    print("APPROACH B1: TWO-STAGE MODEL")
    print("="*60)

    train = train.copy()
    val_copy = val.copy()

    # Stage 1: Binary classification (y > 0)
    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val_copy[FEATURES + CAT_FEATURES]

    # Train classifier
    print("Training Stage 1: Zero classifier...")
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

    # Predict probability of non-zero
    prob_nonzero = clf_model.predict(X_val)

    # Stage 2: Regression on non-zeros only
    print("Training Stage 2: Value regressor...")
    train_nonzero = train[train['y'] > 0].copy()

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

    X_train_nz = train_nonzero[FEATURES + CAT_FEATURES]
    y_train_nz = train_nonzero['y']

    train_reg = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)

    # Predict values
    y_pred_value = reg_model.predict(X_val)

    # Combine: use threshold on probability
    threshold = 0.5
    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_copy['is_store_closed'].values == 1] = 0

    # Evaluate
    y_val = val_copy['y'].values
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    print(f"B1 Two-Stage MAE: {mae:.4f}")
    print(f"B1 Two-Stage WMAPE: {wmape:.2f}%")

    # Also try different thresholds
    print("\nTrying different thresholds:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_t = np.where(prob_nonzero > thresh, y_pred_value, 0)
        y_pred_t = np.maximum(0, y_pred_t)
        y_pred_t[val_copy['is_store_closed'].values == 1] = 0
        wmape_t = 100 * np.sum(np.abs(y_val - y_pred_t)) / np.sum(y_val)
        print(f"  Threshold {thresh}: WMAPE = {wmape_t:.2f}%")

    return {'approach': 'B1_two_stage', 'wmape': wmape, 'mae': mae}


def train_single_store(store_id, train, val, features):
    """Train model for a single store."""
    train_store = train[train['store_id'] == store_id].copy()
    val_store = val[val['store_id'] == store_id].copy()

    if len(train_store) < 100 or len(val_store) < 10:
        return {'store_id': store_id, 'wmape': None, 'error': 'Insufficient data'}

    # Use SKU as only categorical (store is fixed)
    features_no_store = [f for f in features if f not in CAT_FEATURES] + ['sku_id']

    train_store['sku_id'] = train_store['sku_id'].astype('category')
    val_store['sku_id'] = val_store['sku_id'].astype('category')

    X_train = train_store[features_no_store]
    y_train = train_store['y']
    X_val = val_store[features_no_store]
    y_val = val_store['y'].values

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 20,
        'lambda_l2': 1.5,
        'verbose': -1,
        'n_jobs': 1  # Use 1 for parallel training
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['sku_id'])
    model = lgb.train(params, train_data, num_boost_round=150)

    y_pred = model.predict(X_val)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_store['is_store_closed'].values == 1] = 0

    total_y = np.sum(y_val)
    if total_y == 0:
        wmape = None
    else:
        wmape = 100 * np.sum(np.abs(y_val - y_pred)) / total_y

    return {
        'store_id': store_id,
        'wmape': wmape,
        'n_train': len(train_store),
        'n_val': len(val_store),
        'predictions': y_pred,
        'actuals': y_val
    }


def approach_b3_per_store(train, val):
    """B3: Train separate model for each store."""
    print("\n" + "="*60)
    print("APPROACH B3: PER-STORE MODELS")
    print("="*60)

    stores = train['store_id'].unique()
    print(f"Training {len(stores)} store models in parallel...")

    store_results = []
    all_preds = []
    all_actuals = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(train_single_store, s, train, val, FEATURES): s for s in stores}
        for future in as_completed(futures):
            store_id = futures[future]
            try:
                r = future.result()
                store_results.append(r)
                if r['wmape'] is not None:
                    print(f"  Store {r['store_id']}: WMAPE={r['wmape']:.2f}%, n_val={r['n_val']:,}")
                    all_preds.extend(r['predictions'])
                    all_actuals.extend(r['actuals'])
            except Exception as e:
                print(f"  Store {store_id} Error: {e}")

    # Calculate overall WMAPE
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    overall_wmape = 100 * np.sum(np.abs(all_actuals - all_preds)) / np.sum(all_actuals)

    # Also calculate average store WMAPE
    valid_stores = [r for r in store_results if r.get('wmape') is not None]
    avg_wmape = np.mean([r['wmape'] for r in valid_stores]) if valid_stores else None

    print(f"\nB3 Overall WMAPE (combined predictions): {overall_wmape:.2f}%")
    print(f"B3 Average Store WMAPE: {avg_wmape:.2f}%")

    return {'approach': 'B3_per_store', 'wmape': overall_wmape, 'avg_store_wmape': avg_wmape}


def approach_baseline(train, val):
    """Train baseline LightGBM without any special treatment."""
    print("\n" + "="*60)
    print("BASELINE: Standard LightGBM")
    print("="*60)

    train = train.copy()
    val_copy = val.copy()

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    y_train = train['y']
    X_val = val_copy[FEATURES + CAT_FEATURES]
    y_val = val_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l2': 2.0,
        'verbose': -1,
        'n_jobs': -1
    }

    print("Training baseline LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_data, num_boost_round=300)

    y_pred = model.predict(X_val)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    print(f"Baseline MAE: {mae:.4f}")
    print(f"Baseline WMAPE: {wmape:.2f}%")

    return {'approach': 'Baseline_LightGBM', 'wmape': wmape, 'mae': mae}


def main():
    print("="*60)
    print("RUNNING ALL 4 IMPROVEMENT APPROACHES")
    print("="*60)

    # Load data
    train, val = load_data()

    results = []

    # Baseline
    r = approach_baseline(train, val)
    results.append(r)

    # C1: Log Transform
    r = approach_c1_log_transform(train, val)
    results.append(r)

    # B1: Two-Stage
    r = approach_b1_two_stage(train, val)
    results.append(r)

    # B3: Per-Store
    r = approach_b3_per_store(train, val)
    results.append(r)

    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print("\nApproach                    WMAPE")
    print("-" * 40)
    for r in sorted(results, key=lambda x: x.get('wmape', 999)):
        wmape = r.get('wmape', 'N/A')
        if isinstance(wmape, float):
            print(f"{r['approach']:<25} {wmape:.2f}%")
        else:
            print(f"{r['approach']:<25} {wmape}")

    # Reference: D3 baseline from BigQuery
    print("\nReference: D3 (F1 BQML Baseline) = 53.76%")

    return results


if __name__ == "__main__":
    main()
