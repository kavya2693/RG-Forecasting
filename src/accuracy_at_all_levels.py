"""
ACCURACY AT ALL AGGREGATION LEVELS
==================================
Show WFA at every useful business decision level.
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


def prepare_and_predict(train, val, sku_attr):
    """Prepare data and get best predictions."""

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

    # ABC
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

    # Train per-segment
    val['y_pred'] = 0.0

    for seg in ['A', 'B', 'C']:
        train_seg = train[train['abc'] == seg].copy()
        val_seg = val[val['abc'] == seg].copy()

        if len(train_seg) == 0 or len(val_seg) == 0:
            continue

        train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)

        for col in CAT_FEATURES:
            train_seg[col] = train_seg[col].astype('category')
            val_seg[col] = val_seg[col].astype('category')

        X_train = train_seg[FEATURES + CAT_FEATURES]
        X_val = val_seg[FEATURES + CAT_FEATURES]

        # Segment-specific hyperparameters
        if seg == 'A':
            nl, lr, rnds_clf, rnds_reg, mdl = 255, 0.015, 800, 1000, 10
        elif seg == 'B':
            nl, lr, rnds_clf, rnds_reg, mdl = 63, 0.03, 300, 400, 50
        else:
            nl, lr, rnds_clf, rnds_reg, mdl = 31, 0.05, 200, 300, 100

        # Classifier
        clf_data = lgb.Dataset(X_train, label=train_seg['y_binary'], categorical_feature=CAT_FEATURES)
        clf = lgb.train({'objective': 'binary', 'metric': 'auc', 'num_leaves': nl,
                         'learning_rate': lr, 'feature_fraction': 0.8, 'min_data_in_leaf': mdl,
                         'verbose': -1, 'n_jobs': -1}, clf_data, num_boost_round=rnds_clf)
        prob = clf.predict(X_val)

        # Regressor
        train_nz = train_seg[train_seg['y'] > 0]
        X_train_nz = train_nz[FEATURES + CAT_FEATURES]
        y_train_nz = np.log1p(train_nz['y'].values)

        reg_data = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
        reg = lgb.train({'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': nl,
                         'learning_rate': lr, 'feature_fraction': 0.8, 'min_data_in_leaf': max(5, mdl//2),
                         'lambda_l2': 0.5, 'verbose': -1, 'n_jobs': -1}, reg_data, num_boost_round=rnds_reg)
        pred_value = np.expm1(reg.predict(X_val))

        # Optimal threshold
        thresh = 0.6 if seg in ['A', 'B'] else 0.7
        y_pred = np.where(prob > thresh, pred_value, 0)
        y_pred = np.maximum(0, y_pred)
        y_pred[val_seg['is_store_closed'].values == 1] = 0

        # Calibration for A-items
        if seg == 'A':
            prob_tr = clf.predict(X_train)
            pred_val_tr = np.expm1(reg.predict(X_train))
            y_pred_tr = np.where(prob_tr > thresh, pred_val_tr, 0)
            y_pred_tr = np.maximum(0, y_pred_tr)
            mask = y_pred_tr > 0.1
            if np.sum(y_pred_tr[mask]) > 0:
                k = np.clip(np.sum(train_seg['y'].values[mask]) / np.sum(y_pred_tr[mask]), 0.8, 1.3)
                y_pred = y_pred * k
                y_pred[val_seg['is_store_closed'].values == 1] = 0

        val.loc[val['abc'] == seg, 'y_pred'] = y_pred
        print(f"  {seg}-items: {len(val_seg):,} rows")

    val['date'] = pd.to_datetime(val['date'])
    val['week'] = val['date'].dt.isocalendar().week
    val['year'] = val['date'].dt.year

    return val


def report_all_levels(val, tier_name):
    """Report WFA at all aggregation levels."""

    print(f"\n{'='*70}")
    print(f"ACCURACY AT ALL LEVELS - {tier_name}")
    print("="*70)

    results = {}

    # 1. Daily SKU-Store
    wmape = 100 * np.sum(np.abs(val['y'] - val['y_pred'])) / np.sum(val['y'])
    results['daily_sku_store'] = {'wmape': wmape, 'wfa': 100 - wmape}
    print(f"\n  1. Daily SKU-Store:          WFA = {100-wmape:.1f}%   (WMAPE = {wmape:.1f}%)")

    # 2. Weekly SKU-Store
    weekly = val.groupby(['store_id', 'sku_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
    results['weekly_sku_store'] = {'wmape': wmape, 'wfa': 100 - wmape}
    print(f"  2. Weekly SKU-Store:         WFA = {100-wmape:.1f}%   (WMAPE = {wmape:.1f}%)")

    # 3. Weekly SKU-Store (A-items only)
    weekly_a = val[val['abc'] == 'A'].groupby(['store_id', 'sku_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_a['y'] - weekly_a['y_pred'])) / np.sum(weekly_a['y'])
    results['weekly_sku_store_a'] = {'wmape': wmape, 'wfa': 100 - wmape}
    print(f"  3. Weekly SKU-Store (A):     WFA = {100-wmape:.1f}%   (WMAPE = {wmape:.1f}%)")

    # 4. Weekly SKU (across stores)
    weekly_sku = val.groupby(['sku_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_sku['y'] - weekly_sku['y_pred'])) / np.sum(weekly_sku['y'])
    results['weekly_sku'] = {'wmape': wmape, 'wfa': 100 - wmape}
    print(f"  4. Weekly SKU (all stores):  WFA = {100-wmape:.1f}%   (WMAPE = {wmape:.1f}%)")

    # 5. Weekly Store (across SKUs)
    weekly_store = val.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_store['y'] - weekly_store['y_pred'])) / np.sum(weekly_store['y'])
    results['weekly_store'] = {'wmape': wmape, 'wfa': 100 - wmape}
    print(f"  5. Weekly Store:             WFA = {100-wmape:.1f}%   (WMAPE = {wmape:.1f}%)")

    # 6. Total Week (entire chain)
    weekly_total = val.groupby(['year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_total['y'] - weekly_total['y_pred'])) / np.sum(weekly_total['y'])
    results['weekly_total'] = {'wmape': wmape, 'wfa': 100 - wmape}
    print(f"  6. Weekly Total (chain):     WFA = {100-wmape:.1f}%   (WMAPE = {wmape:.1f}%)")

    return results


def main():
    print("="*70)
    print("ACCURACY AT ALL AGGREGATION LEVELS")
    print("="*70)

    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')
    all_results = {}

    for tier_name, train_folder, val_folder in [
        ('T1_MATURE', '/tmp/full_data/train', '/tmp/full_data/val'),
        ('T2_GROWING', '/tmp/t2_data/train', '/tmp/t2_data/val'),
    ]:
        print(f"\n{'#'*70}")
        print(f"# {tier_name}")
        print('#'*70)

        train = load_data(train_folder)
        val = load_data(val_folder)

        print(f"  Train: {len(train):,}, Val: {len(val):,}")
        print("\n  Training per-segment models...")

        val_pred = prepare_and_predict(train, val, sku_attr.copy())
        tier_results = report_all_levels(val_pred, tier_name)
        all_results[tier_name] = tier_results

    # EXECUTIVE SUMMARY
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)

    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│  AGGREGATION LEVEL           T1_MATURE WFA    T2_GROWING WFA      │")
    print("├────────────────────────────────────────────────────────────────────┤")

    for level, label in [
        ('daily_sku_store', 'Daily SKU-Store'),
        ('weekly_sku_store', 'Weekly SKU-Store'),
        ('weekly_sku_store_a', 'Weekly SKU-Store (A)'),
        ('weekly_sku', 'Weekly SKU'),
        ('weekly_store', 'Weekly Store'),
        ('weekly_total', 'Weekly Total'),
    ]:
        t1 = all_results['T1_MATURE'][level]['wfa']
        t2 = all_results['T2_GROWING'][level]['wfa']
        print(f"│  {label:<28} {t1:>8.1f}%          {t2:>8.1f}%           │")

    print("└────────────────────────────────────────────────────────────────────┘")

    os.makedirs('/tmp/accuracy_levels', exist_ok=True)
    with open('/tmp/accuracy_levels/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to /tmp/accuracy_levels/results.json")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
