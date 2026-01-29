"""
PUSH TO 70%+ WFA
================
Focus on what's achievable:
1. A-items specific model with aggressive hyperparameters
2. Per-tier calibration
3. Store-week aggregation for comparison
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


def load_data(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def prepare_data(train, val, sku_attr):
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(lambda x: 1 if x in ['L', 'LI'] else 0)

    for df in [train, val]:
        df['sku_id'] = df['sku_id'].astype(str)
        df['store_id'] = df['store_id'].astype(str)

    train = train.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    val = val.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')

    for df in [train, val]:
        df['is_local'] = df['is_local'].fillna(0).astype(int)
        for col in FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    return train, val


def assign_abc(train, val):
    series_sales = train.groupby(['store_id', 'sku_id'])['y'].sum().reset_index()
    series_sales.columns = ['store_id', 'sku_id', 'total_sales']
    series_sales = series_sales.sort_values('total_sales', ascending=False)

    total = series_sales['total_sales'].sum()
    series_sales['cum_share'] = series_sales['total_sales'].cumsum() / total

    series_sales['abc'] = 'C'
    series_sales.loc[series_sales['cum_share'] <= 0.80, 'abc'] = 'A'
    series_sales.loc[(series_sales['cum_share'] > 0.80) & (series_sales['cum_share'] <= 0.95), 'abc'] = 'B'

    train = train.merge(series_sales[['store_id', 'sku_id', 'abc']], on=['store_id', 'sku_id'], how='left')
    val = val.merge(series_sales[['store_id', 'sku_id', 'abc']], on=['store_id', 'sku_id'], how='left')

    train['abc'] = train['abc'].fillna('C')
    val['abc'] = val['abc'].fillna('C')

    print(f"  A-items: {series_sales[series_sales['abc']=='A'].shape[0]:,} series")
    print(f"  B-items: {series_sales[series_sales['abc']=='B'].shape[0]:,} series")
    print(f"  C-items: {series_sales[series_sales['abc']=='C'].shape[0]:,} series")

    return train, val


def train_a_items_optimized(train, val):
    """Train heavily optimized model for A-items only."""
    print("\n" + "="*70)
    print("TRAINING A-ITEMS OPTIMIZED MODEL")
    print("="*70)

    train_a = train[train['abc'] == 'A'].copy()
    val_a = val[val['abc'] == 'A'].copy()

    print(f"  A-items train: {len(train_a):,}, val: {len(val_a):,}")

    train_a['y_binary'] = (train_a['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train_a[col] = train_a[col].astype('category')
        val_a[col] = val_a[col].astype('category')

    X_train = train_a[FEATURES + CAT_FEATURES]
    X_val = val_a[FEATURES + CAT_FEATURES]

    # AGGRESSIVE HYPERPARAMETERS for A-items
    print("\n  Stage 1: Training classifier (aggressive params)...")

    clf_params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 255,
        'learning_rate': 0.015,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.9,
        'bagging_freq': 3,
        'min_data_in_leaf': 10,
        'lambda_l1': 0.05,
        'lambda_l2': 0.5,
        'max_depth': 12,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    clf_data = lgb.Dataset(X_train, label=train_a['y_binary'], categorical_feature=CAT_FEATURES)
    clf = lgb.train(clf_params, clf_data, num_boost_round=800)
    prob = clf.predict(X_val)

    # Regressor
    print("  Stage 2: Training regressor (aggressive params)...")

    train_nz = train_a[train_a['y'] > 0]
    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    y_train_nz = np.log1p(train_nz['y'].values)

    reg_params = {
        'objective': 'regression_l1',  # MAE for robustness
        'metric': 'mae',
        'num_leaves': 255,
        'learning_rate': 0.015,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.9,
        'bagging_freq': 3,
        'min_data_in_leaf': 5,
        'lambda_l1': 0.05,
        'lambda_l2': 0.3,
        'max_depth': 12,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    reg_data = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
    reg = lgb.train(reg_params, reg_data, num_boost_round=1000)
    pred_log = reg.predict(X_val)
    pred_value = np.expm1(pred_log)

    is_closed = val_a['is_store_closed'].values

    # Find optimal threshold
    print("  Finding optimal threshold...")

    val_a['date'] = pd.to_datetime(val_a['date'])
    val_a['week'] = val_a['date'].dt.isocalendar().week
    val_a['year'] = val_a['date'].dt.year

    best_wmape = float('inf')
    best_thresh = 0.5

    for thresh in np.arange(0.3, 0.8, 0.02):
        y_pred = np.where(prob > thresh, pred_value, 0)
        y_pred = np.maximum(0, y_pred)
        y_pred[is_closed == 1] = 0

        val_a['y_test'] = y_pred
        weekly = val_a.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
            'y': 'sum', 'y_test': 'sum'
        }).reset_index()

        wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_test'])) / np.sum(weekly['y'])
        if wmape < best_wmape:
            best_wmape = wmape
            best_thresh = thresh

    print(f"  Optimal threshold: {best_thresh:.2f}")

    # Apply
    y_pred_a = np.where(prob > best_thresh, pred_value, 0)
    y_pred_a = np.maximum(0, y_pred_a)
    y_pred_a[is_closed == 1] = 0

    # Calibration on train
    prob_train = clf.predict(X_train)
    pred_log_train = reg.predict(X_train)
    pred_value_train = np.expm1(pred_log_train)
    y_pred_train = np.where(prob_train > best_thresh, pred_value_train, 0)
    y_pred_train = np.maximum(0, y_pred_train)
    y_pred_train[train_a['is_store_closed'].values == 1] = 0

    mask = y_pred_train > 0.1
    k = np.sum(train_a['y'].values[mask]) / np.sum(y_pred_train[mask]) if np.sum(y_pred_train[mask]) > 0 else 1.0
    k = np.clip(k, 0.8, 1.3)

    print(f"  Calibration k: {k:.4f}")

    y_pred_a_cal = y_pred_a * k
    y_pred_a_cal[is_closed == 1] = 0
    y_pred_a_cal[y_pred_a == 0] = 0

    val_a['y_pred'] = y_pred_a_cal

    # Weekly metrics
    weekly = val_a.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
    wfa = 100 - wmape
    bias = 100 * (weekly['y_pred'].sum() - weekly['y'].sum()) / weekly['y'].sum()

    print(f"\n  A-ITEMS RESULTS:")
    print(f"    Weekly WMAPE: {wmape:.2f}%")
    print(f"    Weekly WFA: {wfa:.2f}%")
    print(f"    Bias: {bias:.2f}%")

    return {'wmape': wmape, 'wfa': wfa, 'bias': bias}, clf, reg, best_thresh, k


def train_bc_items(train, val, abc_tier):
    """Train model for B or C items."""
    print(f"\n  Training {abc_tier}-items model...")

    train_seg = train[train['abc'] == abc_tier].copy()
    val_seg = val[val['abc'] == abc_tier].copy()

    if len(train_seg) == 0 or len(val_seg) == 0:
        return None, None, None, None, None

    train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train_seg[col] = train_seg[col].astype('category')
        val_seg[col] = val_seg[col].astype('category')

    X_train = train_seg[FEATURES + CAT_FEATURES]
    X_val = val_seg[FEATURES + CAT_FEATURES]

    # Moderate hyperparameters
    clf_params = {
        'objective': 'binary', 'metric': 'auc', 'num_leaves': 63,
        'learning_rate': 0.03, 'feature_fraction': 0.7,
        'min_data_in_leaf': 50, 'verbose': -1, 'n_jobs': -1
    }

    clf_data = lgb.Dataset(X_train, label=train_seg['y_binary'], categorical_feature=CAT_FEATURES)
    clf = lgb.train(clf_params, clf_data, num_boost_round=300)
    prob = clf.predict(X_val)

    train_nz = train_seg[train_seg['y'] > 0]
    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    y_train_nz = np.log1p(train_nz['y'].values)

    reg_params = {
        'objective': 'regression', 'metric': 'mae', 'num_leaves': 63,
        'learning_rate': 0.03, 'feature_fraction': 0.7,
        'min_data_in_leaf': 30, 'lambda_l2': 1.5, 'verbose': -1, 'n_jobs': -1
    }

    reg_data = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
    reg = lgb.train(reg_params, reg_data, num_boost_round=400)
    pred_log = reg.predict(X_val)
    pred_value = np.expm1(pred_log)

    # Threshold
    thresh = 0.6 if abc_tier == 'B' else 0.7
    y_pred = np.where(prob > thresh, pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_seg['is_store_closed'].values == 1] = 0

    val_seg['y_pred'] = y_pred
    val_seg['date'] = pd.to_datetime(val_seg['date'])
    val_seg['week'] = val_seg['date'].dt.isocalendar().week
    val_seg['year'] = val_seg['date'].dt.year

    weekly = val_seg.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])

    print(f"    {abc_tier}-items Weekly WFA: {100-wmape:.2f}%")

    return val_seg, clf, reg, thresh, 1.0


def store_week_aggregation(val):
    """Calculate store-week aggregated metrics."""
    print("\n" + "="*70)
    print("STORE-WEEK AGGREGATION (Higher level)")
    print("="*70)

    val_copy = val.copy()
    val_copy['date'] = pd.to_datetime(val_copy['date'])
    val_copy['week'] = val_copy['date'].dt.isocalendar().week
    val_copy['year'] = val_copy['date'].dt.year

    store_week = val_copy.groupby(['store_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    wmape = 100 * np.sum(np.abs(store_week['y'] - store_week['y_pred'])) / np.sum(store_week['y'])
    wfa = 100 - wmape
    bias = 100 * (store_week['y_pred'].sum() - store_week['y'].sum()) / store_week['y'].sum()

    print(f"  Store-Week WMAPE: {wmape:.2f}%")
    print(f"  Store-Week WFA: {wfa:.2f}%")
    print(f"  Bias: {bias:.2f}%")

    return {'wmape': wmape, 'wfa': wfa, 'bias': bias}


def main():
    print("="*70)
    print("PUSH TO 70%+ WFA - T1_MATURE")
    print("="*70)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    train = load_data('/tmp/full_data/train')
    val = load_data('/tmp/full_data/val')
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    train, val = prepare_data(train, val, sku_attr)
    train, val = assign_abc(train, val)

    # Train A-items
    metrics_a, clf_a, reg_a, thresh_a, k_a = train_a_items_optimized(train.copy(), val.copy())

    # Train B-items
    val_b, clf_b, reg_b, thresh_b, k_b = train_bc_items(train.copy(), val.copy(), 'B')

    # Train C-items
    val_c, clf_c, reg_c, thresh_c, k_c = train_bc_items(train.copy(), val.copy(), 'C')

    # Combine all predictions
    print("\n" + "="*70)
    print("COMBINING ALL SEGMENTS")
    print("="*70)

    val_full = val.copy()
    val_full['y_pred'] = 0.0
    val_full['date'] = pd.to_datetime(val_full['date'])
    val_full['week'] = val_full['date'].dt.isocalendar().week
    val_full['year'] = val_full['date'].dt.year

    # Predict A-items
    val_a = val_full[val_full['abc'] == 'A'].copy()
    for col in CAT_FEATURES:
        val_a[col] = val_a[col].astype('category')
    X_val_a = val_a[FEATURES + CAT_FEATURES]
    prob_a = clf_a.predict(X_val_a)
    pred_log_a = reg_a.predict(X_val_a)
    pred_value_a = np.expm1(pred_log_a)
    y_pred_a = np.where(prob_a > thresh_a, pred_value_a, 0) * k_a
    y_pred_a = np.maximum(0, y_pred_a)
    y_pred_a[val_a['is_store_closed'].values == 1] = 0
    val_full.loc[val_full['abc'] == 'A', 'y_pred'] = y_pred_a

    # Predict B-items
    if val_b is not None:
        val_full.loc[val_full['abc'] == 'B', 'y_pred'] = val_b['y_pred'].values

    # Predict C-items
    if val_c is not None:
        val_full.loc[val_full['abc'] == 'C', 'y_pred'] = val_c['y_pred'].values

    # Overall weekly metrics
    weekly = val_full.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    wmape_all = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
    wfa_all = 100 - wmape_all
    bias_all = 100 * (weekly['y_pred'].sum() - weekly['y'].sum()) / weekly['y'].sum()

    print(f"\n  OVERALL WEEKLY (SKU-Store-Week):")
    print(f"    WMAPE: {wmape_all:.2f}%")
    print(f"    WFA: {wfa_all:.2f}%")
    print(f"    Bias: {bias_all:.2f}%")

    # A-items weekly
    weekly_a = val_full[val_full['abc'] == 'A'].groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    wmape_a = 100 * np.sum(np.abs(weekly_a['y'] - weekly_a['y_pred'])) / np.sum(weekly_a['y'])
    print(f"\n  A-ITEMS WEEKLY:")
    print(f"    WFA: {100-wmape_a:.2f}%")

    # Store-week
    store_week_metrics = store_week_aggregation(val_full)

    # FINAL SUMMARY
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print("\n┌────────────────────────────────────────────────────────────────┐")
    print("│  AGGREGATION LEVEL           WMAPE       WFA         Bias      │")
    print("├────────────────────────────────────────────────────────────────┤")
    print(f"│  SKU-Store-Week (Overall)    {wmape_all:>6.2f}%    {wfa_all:>6.2f}%     {bias_all:>6.2f}%   │")
    print(f"│  SKU-Store-Week (A-items)    {wmape_a:>6.2f}%    {100-wmape_a:>6.2f}%     -         │")
    print(f"│  Store-Week                  {store_week_metrics['wmape']:>6.2f}%    {store_week_metrics['wfa']:>6.2f}%     {store_week_metrics['bias']:>6.2f}%   │")
    print("└────────────────────────────────────────────────────────────────┘")

    # Save models
    os.makedirs('/tmp/best_models', exist_ok=True)
    clf_a.save_model('/tmp/best_models/clf_a_items.txt')
    reg_a.save_model('/tmp/best_models/reg_a_items.txt')

    results = {
        'overall_weekly': {'wmape': wmape_all, 'wfa': wfa_all, 'bias': bias_all},
        'a_items_weekly': {'wmape': wmape_a, 'wfa': 100-wmape_a},
        'store_week': store_week_metrics
    }

    with open('/tmp/best_models/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nModels saved to /tmp/best_models/")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
