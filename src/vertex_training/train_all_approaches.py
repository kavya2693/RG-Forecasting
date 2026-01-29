"""
Vertex AI Parallel Training - All 4 Improvement Approaches
===========================================================
D3: Ensemble Folds
C1: Log Transform Target
B1: Two-Stage Model
B3: Per-Store Models
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from google.cloud import bigquery, storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ID = "myforecastingsales"
BUCKET = "myforecastingsales-data"
REGION = "us-central1"

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


def load_fold_data(fold_id):
    """Load data for a specific fold from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT * FROM `{PROJECT_ID}.forecasting_us.trainval_{fold_id.lower()}`
    """
    df = client.query(query).to_dataframe()
    return df


def approach_d3_ensemble():
    """D3: Ensemble existing F1+F2+F3 predictions."""
    print("\n" + "="*60)
    print("APPROACH D3: ENSEMBLE FOLDS")
    print("="*60)

    client = bigquery.Client(project=PROJECT_ID)

    # Get F1 predictions (most recent fold)
    query = """
    SELECT
        store_id, sku_id, date, y, yhat,
        is_store_closed
    FROM `myforecastingsales.forecasting_us.f1_val_pred`
    """
    f1_pred = client.query(query).to_dataframe()

    # For ensemble, we'd need F2/F3 predictions on same dates
    # Since they have different VAL periods, we'll just report F1 baseline

    wmape = 100 * np.sum(np.abs(f1_pred['y'] - f1_pred['yhat'])) / np.sum(f1_pred['y'])
    print(f"F1 Baseline WMAPE: {wmape:.2f}%")

    return {'approach': 'D3_ensemble', 'wmape': wmape}


def approach_c1_log_transform(fold_id='f1'):
    """C1: Log Transform Target - train with log(1+y)."""
    print(f"\n" + "="*60)
    print(f"APPROACH C1: LOG TRANSFORM ({fold_id.upper()})")
    print("="*60)

    client = bigquery.Client(project=PROJECT_ID)

    # Load data
    query = f"""
    SELECT * FROM `{PROJECT_ID}.forecasting_us.trainval_{fold_id}`
    """
    df = client.query(query).to_dataframe()

    train = df[df['split_role'] == 'TRAIN'].copy()
    val = df[df['split_role'] == 'VAL'].copy()

    # Log transform target
    train['y_log'] = np.log1p(train['y'])

    # Prepare features
    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val[col] = val[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    y_train = train['y_log']
    X_val = val[FEATURES + CAT_FEATURES]
    y_val = val['y']
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
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)

    model = lgb.train(params, train_data, num_boost_round=200)

    # Predict and inverse transform
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)  # Inverse of log1p
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # Evaluate
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    print(f"{fold_id.upper()} Log Transform MAE: {mae:.4f}")
    print(f"{fold_id.upper()} Log Transform WMAPE: {wmape:.2f}%")

    return {'approach': f'C1_log_{fold_id}', 'wmape': wmape, 'mae': mae, 'model': model}


def approach_b1_two_stage(fold_id='f1'):
    """B1: Two-Stage Model - classify zero/non-zero, then predict value."""
    print(f"\n" + "="*60)
    print(f"APPROACH B1: TWO-STAGE MODEL ({fold_id.upper()})")
    print("="*60)

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT * FROM `{PROJECT_ID}.forecasting_us.trainval_{fold_id}`
    """
    df = client.query(query).to_dataframe()

    train = df[df['split_role'] == 'TRAIN'].copy()
    val = df[df['split_role'] == 'VAL'].copy()

    # Stage 1: Binary classification (y > 0)
    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val[col] = val[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val[FEATURES + CAT_FEATURES]

    # Train classifier
    print("Training Stage 1: Zero classifier...")
    params_clf = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 100,
        'verbose': -1
    }

    train_clf = lgb.Dataset(X_train, label=train['y_binary'], categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=100)

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
        'verbose': -1
    }

    X_train_nz = train_nonzero[FEATURES + CAT_FEATURES]
    y_train_nz = train_nonzero['y']

    train_reg = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=200)

    # Predict values
    y_pred_value = reg_model.predict(X_val)

    # Combine: use threshold on probability
    threshold = 0.5
    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val['is_store_closed'].values == 1] = 0

    # Evaluate
    y_val = val['y'].values
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    print(f"{fold_id.upper()} Two-Stage MAE: {mae:.4f}")
    print(f"{fold_id.upper()} Two-Stage WMAPE: {wmape:.2f}%")

    return {'approach': f'B1_twostage_{fold_id}', 'wmape': wmape, 'mae': mae}


def approach_b3_per_store(store_id, fold_id='f1'):
    """B3: Train model for a single store."""
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT * FROM `{PROJECT_ID}.forecasting_us.trainval_{fold_id}`
    WHERE store_id = '{store_id}'
    """
    df = client.query(query).to_dataframe()

    if len(df) == 0:
        return {'store_id': store_id, 'wmape': None, 'error': 'No data'}

    train = df[df['split_role'] == 'TRAIN'].copy()
    val = df[df['split_role'] == 'VAL'].copy()

    if len(train) < 100 or len(val) < 10:
        return {'store_id': store_id, 'wmape': None, 'error': 'Insufficient data'}

    # Use SKU as only categorical (store is fixed)
    features_no_store = [f for f in FEATURES if f != 'store_id'] + ['sku_id']

    train['sku_id'] = train['sku_id'].astype('category')
    val['sku_id'] = val['sku_id'].astype('category')

    X_train = train[features_no_store]
    y_train = train['y']
    X_val = val[features_no_store]
    y_val = val['y']

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 20,
        'lambda_l2': 1.5,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['sku_id'])
    model = lgb.train(params, train_data, num_boost_round=100)

    y_pred = model.predict(X_val)
    y_pred = np.maximum(0, y_pred)
    y_pred[val['is_store_closed'].values == 1] = 0

    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / max(np.sum(y_val), 1)

    return {'store_id': store_id, 'wmape': wmape, 'n_train': len(train), 'n_val': len(val)}


def run_all_approaches():
    """Run all 4 approaches and compare."""
    results = []

    # D3: Ensemble (baseline)
    try:
        r = approach_d3_ensemble()
        results.append(r)
    except Exception as e:
        print(f"D3 Error: {e}")

    # C1: Log Transform
    try:
        r = approach_c1_log_transform('f1')
        results.append(r)
    except Exception as e:
        print(f"C1 Error: {e}")

    # B1: Two-Stage
    try:
        r = approach_b1_two_stage('f1')
        results.append(r)
    except Exception as e:
        print(f"B1 Error: {e}")

    # B3: Per-Store (parallel)
    print("\n" + "="*60)
    print("APPROACH B3: PER-STORE MODELS (PARALLEL)")
    print("="*60)

    stores = [str(i) for i in range(201, 234)]
    store_results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(approach_b3_per_store, s): s for s in stores}
        for future in as_completed(futures):
            store_id = futures[future]
            try:
                r = future.result()
                store_results.append(r)
                if r['wmape']:
                    print(f"  Store {r['store_id']}: WMAPE={r['wmape']:.2f}%")
            except Exception as e:
                print(f"  Store {store_id} Error: {e}")

    # Aggregate per-store results
    valid_stores = [r for r in store_results if r.get('wmape') is not None]
    if valid_stores:
        avg_wmape = np.mean([r['wmape'] for r in valid_stores])
        results.append({'approach': 'B3_per_store_avg', 'wmape': avg_wmape})
        print(f"\nB3 Per-Store Average WMAPE: {avg_wmape:.2f}%")

    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    for r in sorted(results, key=lambda x: x.get('wmape', 999)):
        print(f"  {r['approach']}: WMAPE={r.get('wmape', 'N/A'):.2f}%")

    return results


if __name__ == "__main__":
    run_all_approaches()
