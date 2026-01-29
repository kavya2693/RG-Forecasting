"""
Business-Focused Accuracy Metrics for All Tiers
================================================
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
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def get_predictions(train, val, sku_attr, threshold=0.7):
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(lambda x: 1 if x in ['L', 'LI'] else 0)

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

    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val[col] = val[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val[FEATURES + CAT_FEATURES]

    params_clf = {
        'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
        'learning_rate': 0.05, 'feature_fraction': 0.8,
        'min_data_in_leaf': 100, 'verbose': -1, 'n_jobs': -1
    }
    train_clf = lgb.Dataset(X_train, label=train['y_binary'], categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(X_val)

    train_nz = train[train['y'] > 0].copy()
    train_nz['y_log'] = np.log1p(train_nz['y'])

    params_reg = {
        'objective': 'regression', 'metric': 'mae', 'num_leaves': 63,
        'learning_rate': 0.05, 'feature_fraction': 0.8,
        'min_data_in_leaf': 30, 'lambda_l2': 1.5, 'verbose': -1, 'n_jobs': -1
    }
    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    train_reg = lgb.Dataset(X_train_nz, label=train_nz['y_log'], categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)

    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)
    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val['is_store_closed'].values == 1] = 0

    val['y_pred'] = y_pred
    return val, train


def calculate_business_metrics(val, train, tier_name):
    results = {}

    # Daily
    daily_wmape = 100 * np.sum(np.abs(val['y'] - val['y_pred'])) / np.sum(val['y']) if np.sum(val['y']) > 0 else np.nan
    results['daily'] = {'wmape': daily_wmape, 'wfa': 100 - daily_wmape if not np.isnan(daily_wmape) else np.nan}

    # Weekly
    val['date'] = pd.to_datetime(val['date'])
    val['week'] = val['date'].dt.isocalendar().week
    val['year'] = val['date'].dt.year

    weekly = val.groupby(['store_id', 'sku_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape_weekly = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y']) if np.sum(weekly['y']) > 0 else np.nan
    results['weekly'] = {'wmape': wmape_weekly, 'wfa': 100 - wmape_weekly if not np.isnan(wmape_weekly) else np.nan}

    # ABC
    train['sku_id'] = train['sku_id'].astype(str)
    train['store_id'] = train['store_id'].astype(str)
    series_sales = train.groupby(['store_id', 'sku_id'])['y'].sum().reset_index()
    series_sales.columns = ['store_id', 'sku_id', 'total_train_sales']
    series_sales = series_sales.sort_values('total_train_sales', ascending=False)

    total_sales = series_sales['total_train_sales'].sum()
    series_sales['cumulative_share'] = series_sales['total_train_sales'].cumsum() / total_sales

    series_sales['abc_tier'] = 'C'
    series_sales.loc[series_sales['cumulative_share'] <= 0.80, 'abc_tier'] = 'A'
    series_sales.loc[(series_sales['cumulative_share'] > 0.80) & (series_sales['cumulative_share'] <= 0.95), 'abc_tier'] = 'B'

    val['store_id'] = val['store_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)
    val_abc = val.merge(series_sales[['store_id', 'sku_id', 'abc_tier']], on=['store_id', 'sku_id'], how='left')
    val_abc['abc_tier'] = val_abc['abc_tier'].fillna('C')

    results['abc'] = {}
    for tier in ['A', 'B', 'C']:
        tier_data = val_abc[val_abc['abc_tier'] == tier]
        if len(tier_data) > 0 and tier_data['y'].sum() > 0:
            wmape_tier = 100 * np.sum(np.abs(tier_data['y'] - tier_data['y_pred'])) / np.sum(tier_data['y'])
            results['abc'][tier] = {'wmape': wmape_tier, 'wfa': 100 - wmape_tier, 'n_rows': len(tier_data)}

    # Non-zero
    nz_days = val[val['y'] > 0]
    if len(nz_days) > 0 and nz_days['y'].sum() > 0:
        wmape_nz = 100 * np.sum(np.abs(nz_days['y'] - nz_days['y_pred'])) / np.sum(nz_days['y'])
        nz_days_gt2 = nz_days[nz_days['y'] >= 2]
        if len(nz_days_gt2) > 0:
            pct_error = np.abs(nz_days_gt2['y_pred'] - nz_days_gt2['y']) / nz_days_gt2['y']
            within_50pct = 100 * np.mean(pct_error <= 0.50)
        else:
            within_50pct = np.nan
        results['nonzero'] = {'wmape': wmape_nz, 'wfa': 100 - wmape_nz, 'pct_within_50': within_50pct}

    return results


def main():
    print("="*70)
    print("BUSINESS METRICS - ALL TIERS SUMMARY")
    print("="*70)

    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    tiers = [
        {'name': 'T1_MATURE', 'train': '/tmp/full_data/train', 'val': '/tmp/full_data/val'},
        {'name': 'T2_GROWING', 'train': '/tmp/t2_data/train', 'val': '/tmp/t2_data/val'},
        {'name': 'T3_COLD_START', 'train': '/tmp/t3_data/train', 'val': '/tmp/t3_data/val'},
    ]

    all_results = {}

    for tier in tiers:
        print(f"\n{'='*60}")
        print(f"Processing {tier['name']}...")
        print("="*60)

        train = load_sharded_csvs(tier['train'])
        val = load_sharded_csvs(tier['val'])

        val_pred, train_data = get_predictions(train, val, sku_attr.copy())
        results = calculate_business_metrics(val_pred, train_data, tier['name'])
        all_results[tier['name']] = results

        print(f"  Daily WFA: {results['daily']['wfa']:.1f}%")
        print(f"  Weekly WFA: {results['weekly']['wfa']:.1f}%")
        if 'A' in results['abc']:
            print(f"  A-Items WFA: {results['abc']['A']['wfa']:.1f}%")

    # Summary table
    print("\n" + "="*80)
    print("BUSINESS METRICS SUMMARY - ALL TIERS")
    print("="*80)

    print("\n┌────────────────────────────────────────────────────────────────────────┐")
    print("│  METRIC                   T1_MATURE    T2_GROWING   T3_COLD_START     │")
    print("├────────────────────────────────────────────────────────────────────────┤")

    # Daily WFA
    t1_daily = all_results['T1_MATURE']['daily']['wfa']
    t2_daily = all_results['T2_GROWING']['daily']['wfa']
    t3_daily = all_results['T3_COLD_START']['daily']['wfa']
    print(f"│  Daily WFA (1-WMAPE)      {t1_daily:>6.1f}%       {t2_daily:>6.1f}%       {t3_daily:>6.1f}%        │")

    # Weekly WFA
    t1_weekly = all_results['T1_MATURE']['weekly']['wfa']
    t2_weekly = all_results['T2_GROWING']['weekly']['wfa']
    t3_weekly = all_results['T3_COLD_START']['weekly']['wfa']
    print(f"│  Weekly WFA               {t1_weekly:>6.1f}%       {t2_weekly:>6.1f}%       {t3_weekly:>6.1f}%        │")

    # A-Items WFA
    t1_a = all_results['T1_MATURE']['abc'].get('A', {}).get('wfa', np.nan)
    t2_a = all_results['T2_GROWING']['abc'].get('A', {}).get('wfa', np.nan)
    t3_a = all_results['T3_COLD_START']['abc'].get('A', {}).get('wfa', np.nan)
    print(f"│  A-Items WFA (Top 80%)    {t1_a:>6.1f}%       {t2_a:>6.1f}%       {t3_a:>6.1f}%        │")

    # Non-zero WFA
    t1_nz = all_results['T1_MATURE']['nonzero']['wfa']
    t2_nz = all_results['T2_GROWING']['nonzero']['wfa']
    t3_nz = all_results['T3_COLD_START']['nonzero']['wfa']
    print(f"│  Non-Zero Day WFA         {t1_nz:>6.1f}%       {t2_nz:>6.1f}%       {t3_nz:>6.1f}%        │")

    # Within 50%
    t1_50 = all_results['T1_MATURE']['nonzero']['pct_within_50']
    t2_50 = all_results['T2_GROWING']['nonzero']['pct_within_50']
    t3_50 = all_results['T3_COLD_START']['nonzero']['pct_within_50']
    print(f"│  % within ±50% (y≥2)      {t1_50:>6.1f}%       {t2_50:>6.1f}%       {t3_50:>6.1f}%        │")

    print("└────────────────────────────────────────────────────────────────────────┘")

    # Save
    os.makedirs('/tmp/business_metrics', exist_ok=True)
    with open('/tmp/business_metrics/all_tiers_business_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    print(f"\nSaved to /tmp/business_metrics/all_tiers_business_metrics.json")


if __name__ == "__main__":
    main()
