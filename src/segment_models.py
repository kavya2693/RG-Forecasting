"""
PER-SEGMENT MODELS
==================
Train separate optimized models for A, B, C items
Target: 78%+ Weekly WFA
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
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def assign_abc_tiers(train, val):
    """Assign ABC tiers based on training sales volume."""
    train['store_id'] = train['store_id'].astype(str)
    train['sku_id'] = train['sku_id'].astype(str)
    val['store_id'] = val['store_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)

    # Calculate sales per series
    series_sales = train.groupby(['store_id', 'sku_id'])['y'].sum().reset_index()
    series_sales.columns = ['store_id', 'sku_id', 'total_sales']
    series_sales = series_sales.sort_values('total_sales', ascending=False)

    # Cumulative share
    total = series_sales['total_sales'].sum()
    series_sales['cum_share'] = series_sales['total_sales'].cumsum() / total

    # Assign tiers
    series_sales['abc'] = 'C'
    series_sales.loc[series_sales['cum_share'] <= 0.80, 'abc'] = 'A'
    series_sales.loc[(series_sales['cum_share'] > 0.80) & (series_sales['cum_share'] <= 0.95), 'abc'] = 'B'

    # Merge
    train = train.merge(series_sales[['store_id', 'sku_id', 'abc']], on=['store_id', 'sku_id'], how='left')
    val = val.merge(series_sales[['store_id', 'sku_id', 'abc']], on=['store_id', 'sku_id'], how='left')

    train['abc'] = train['abc'].fillna('C')
    val['abc'] = val['abc'].fillna('C')

    counts = series_sales['abc'].value_counts()
    print(f"  A-items: {counts.get('A', 0):,}, B-items: {counts.get('B', 0):,}, C-items: {counts.get('C', 0):,}")

    return train, val


def prepare_features(df, sku_attr):
    """Prepare features."""
    df = df.copy()
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(lambda x: 1 if x in ['L', 'LI'] else 0)

    df['sku_id'] = df['sku_id'].astype(str)
    df = df.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    df['is_local'] = df['is_local'].fillna(0).astype(int)

    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def train_segment_model(train_seg, val_seg, segment_name, threshold_range=(0.3, 0.8)):
    """Train optimized model for a segment."""

    if len(train_seg) == 0 or len(val_seg) == 0:
        return None, None

    # Prepare
    train_seg = train_seg.copy()
    val_seg = val_seg.copy()
    train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train_seg[col] = train_seg[col].astype('category')
        val_seg[col] = val_seg[col].astype('category')

    X_train = train_seg[FEATURES + CAT_FEATURES]
    X_val = val_seg[FEATURES + CAT_FEATURES]
    y_train_binary = train_seg['y_binary'].values
    y_val = val_seg['y'].values
    is_closed = val_seg['is_store_closed'].values

    # Non-zero for regressor
    train_nz = train_seg[train_seg['y'] > 0]
    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    y_train_nz_log = np.log1p(train_nz['y'].values)

    # Segment-specific hyperparameters
    if segment_name == 'A':
        # A-items: High volume, more aggressive
        clf_params = {
            'objective': 'binary', 'metric': 'auc', 'num_leaves': 127,
            'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.9,
            'bagging_freq': 5, 'min_data_in_leaf': 20, 'verbose': -1, 'n_jobs': -1
        }
        reg_params = {
            'objective': 'regression', 'metric': 'mae', 'num_leaves': 255,
            'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.9,
            'bagging_freq': 5, 'min_data_in_leaf': 10, 'lambda_l2': 0.5, 'verbose': -1, 'n_jobs': -1
        }
        clf_rounds, reg_rounds = 500, 800
    elif segment_name == 'B':
        # B-items: Medium volume
        clf_params = {
            'objective': 'binary', 'metric': 'auc', 'num_leaves': 63,
            'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'min_data_in_leaf': 50, 'verbose': -1, 'n_jobs': -1
        }
        reg_params = {
            'objective': 'regression', 'metric': 'mae', 'num_leaves': 127,
            'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'min_data_in_leaf': 20, 'lambda_l2': 1.0, 'verbose': -1, 'n_jobs': -1
        }
        clf_rounds, reg_rounds = 300, 500
    else:
        # C-items: Low volume, more regularization
        clf_params = {
            'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
            'learning_rate': 0.05, 'feature_fraction': 0.6, 'bagging_fraction': 0.7,
            'bagging_freq': 5, 'min_data_in_leaf': 100, 'verbose': -1, 'n_jobs': -1
        }
        reg_params = {
            'objective': 'regression', 'metric': 'mae', 'num_leaves': 63,
            'learning_rate': 0.05, 'feature_fraction': 0.6, 'bagging_fraction': 0.7,
            'bagging_freq': 5, 'min_data_in_leaf': 50, 'lambda_l2': 2.0, 'verbose': -1, 'n_jobs': -1
        }
        clf_rounds, reg_rounds = 200, 300

    # Train classifier
    train_clf_data = lgb.Dataset(X_train, label=y_train_binary, categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(clf_params, train_clf_data, num_boost_round=clf_rounds)
    prob = clf_model.predict(X_val)

    # Train regressor
    train_reg_data = lgb.Dataset(X_train_nz, label=y_train_nz_log, categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(reg_params, train_reg_data, num_boost_round=reg_rounds)
    pred_log = reg_model.predict(X_val)
    pred_value = np.expm1(pred_log)

    # Find optimal threshold
    best_wmape = float('inf')
    best_thresh = 0.5
    best_pred = None

    for thresh in np.arange(threshold_range[0], threshold_range[1], 0.05):
        y_pred = np.where(prob > thresh, pred_value, 0)
        y_pred = np.maximum(0, y_pred)
        y_pred_closed = y_pred.copy()
        y_pred_closed[is_closed == 1] = 0

        if np.sum(y_val) > 0:
            wmape = 100 * np.sum(np.abs(y_val - y_pred_closed)) / np.sum(y_val)
            if wmape < best_wmape:
                best_wmape = wmape
                best_thresh = thresh
                best_pred = y_pred_closed

    return best_pred, {'wmape': best_wmape, 'wfa': 100 - best_wmape, 'threshold': best_thresh, 'n_rows': len(val_seg)}


def train_all_segments(train, val, tier_name):
    """Train models for all segments and combine."""

    print(f"\n{'='*60}")
    print(f"TRAINING PER-SEGMENT MODELS: {tier_name}")
    print("="*60)

    # Assign ABC
    print("\n1. Assigning ABC tiers...")
    train, val = assign_abc_tiers(train, val)

    all_preds = np.zeros(len(val))
    segment_metrics = {}

    for segment in ['A', 'B', 'C']:
        train_seg = train[train['abc'] == segment]
        val_seg = val[val['abc'] == segment]

        print(f"\n2. Training {segment}-segment model...")
        print(f"   Train: {len(train_seg):,}, Val: {len(val_seg):,}")

        if len(train_seg) > 0 and len(val_seg) > 0:
            preds, metrics = train_segment_model(train_seg, val_seg, segment)

            if preds is not None:
                # Store predictions at correct indices
                val_indices = val[val['abc'] == segment].index
                all_preds[val_indices] = preds

                segment_metrics[segment] = metrics
                print(f"   {segment}-segment WMAPE: {metrics['wmape']:.2f}%, WFA: {metrics['wfa']:.2f}%")
                print(f"   Optimal threshold: {metrics['threshold']:.2f}")

    # Calculate overall metrics
    y_val = val['y'].values

    wmape_daily = 100 * np.sum(np.abs(y_val - all_preds)) / np.sum(y_val)
    wfa_daily = 100 - wmape_daily

    # Weekly
    val_result = val.copy()
    val_result['y_pred'] = all_preds
    val_result['date'] = pd.to_datetime(val_result['date'])
    val_result['week'] = val_result['date'].dt.isocalendar().week
    val_result['year'] = val_result['date'].dt.year

    weekly = val_result.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    wmape_weekly = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
    wfa_weekly = 100 - wmape_weekly

    print(f"\n{'='*60}")
    print(f"OVERALL METRICS - {tier_name}")
    print("="*60)
    print(f"  Daily WMAPE: {wmape_daily:.2f}%, WFA: {wfa_daily:.2f}%")
    print(f"  Weekly WMAPE: {wmape_weekly:.2f}%, WFA: {wfa_weekly:.2f}%")

    return {
        'daily_wmape': wmape_daily,
        'daily_wfa': wfa_daily,
        'weekly_wmape': wmape_weekly,
        'weekly_wfa': wfa_weekly,
        'segments': segment_metrics
    }


def main():
    print("="*70)
    print("PER-SEGMENT MODELS FOR MAXIMUM ACCURACY")
    print("Target: 78%+ Weekly WFA")
    print("="*70)
    print(f"Started: {datetime.now()}")

    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    results = {}

    # T1_MATURE
    print("\n" + "#"*70)
    print("TIER: T1_MATURE")
    print("#"*70)

    train = load_data('/tmp/full_data/train')
    val = load_data('/tmp/full_data/val')
    train = prepare_features(train, sku_attr.copy())
    val = prepare_features(val, sku_attr.copy())

    results['T1_MATURE'] = train_all_segments(train, val, 'T1_MATURE')

    # T2_GROWING
    print("\n" + "#"*70)
    print("TIER: T2_GROWING")
    print("#"*70)

    train = load_data('/tmp/t2_data/train')
    val = load_data('/tmp/t2_data/val')
    train = prepare_features(train, sku_attr.copy())
    val = prepare_features(val, sku_attr.copy())

    results['T2_GROWING'] = train_all_segments(train, val, 'T2_GROWING')

    # FINAL SUMMARY
    print("\n" + "="*70)
    print("FINAL RESULTS - PER-SEGMENT MODELS")
    print("="*70)

    print("\n┌───────────────────────────────────────────────────────────────────┐")
    print("│  METRIC                    T1_MATURE         T2_GROWING          │")
    print("├───────────────────────────────────────────────────────────────────┤")
    print(f"│  Daily WMAPE               {results['T1_MATURE']['daily_wmape']:>6.2f}%            {results['T2_GROWING']['daily_wmape']:>6.2f}%           │")
    print(f"│  Daily WFA                 {results['T1_MATURE']['daily_wfa']:>6.2f}%            {results['T2_GROWING']['daily_wfa']:>6.2f}%           │")
    print(f"│  Weekly WMAPE              {results['T1_MATURE']['weekly_wmape']:>6.2f}%            {results['T2_GROWING']['weekly_wmape']:>6.2f}%           │")
    print(f"│  Weekly WFA                {results['T1_MATURE']['weekly_wfa']:>6.2f}%            {results['T2_GROWING']['weekly_wfa']:>6.2f}%           │")
    print("└───────────────────────────────────────────────────────────────────┘")

    # A-items breakdown
    print("\n  A-ITEMS (Top 80% sales):")
    print(f"    T1: WFA {results['T1_MATURE']['segments']['A']['wfa']:.2f}%")
    print(f"    T2: WFA {results['T2_GROWING']['segments']['A']['wfa']:.2f}%")

    # Save
    os.makedirs('/tmp/segment_models', exist_ok=True)
    with open('/tmp/segment_models/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to /tmp/segment_models/results.json")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
