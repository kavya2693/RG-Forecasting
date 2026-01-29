"""
Business-Focused Accuracy Metrics
=================================
Calculates metrics at the planning level where decisions are made.
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
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_predictions(train, val, sku_attr, threshold=0.7):
    """Train C1+B1 and return predictions."""

    # Merge SKU attributes
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

    # Stage 1: Binary Classifier
    params_clf = {
        'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
        'learning_rate': 0.05, 'feature_fraction': 0.8,
        'min_data_in_leaf': 100, 'verbose': -1, 'n_jobs': -1
    }
    train_clf = lgb.Dataset(X_train, label=train['y_binary'], categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(X_val)

    # Stage 2: Log-Transform Regressor
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

    # Combine
    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)
    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val['is_store_closed'].values == 1] = 0

    # Return val with predictions
    val['y_pred'] = y_pred
    return val, train


def calculate_business_metrics(val, train):
    """Calculate business-focused accuracy metrics."""

    results = {}

    # ==================================================================
    # A) WEEKLY AGGREGATION
    # ==================================================================
    print("\n" + "="*60)
    print("A) WEEKLY AGGREGATED METRICS")
    print("="*60)

    val['date'] = pd.to_datetime(val['date'])
    val['week'] = val['date'].dt.isocalendar().week
    val['year'] = val['date'].dt.year

    weekly = val.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum',
        'y_pred': 'sum'
    }).reset_index()

    wmape_weekly = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
    wfa_weekly = 100 - wmape_weekly
    mae_weekly = np.mean(np.abs(weekly['y'] - weekly['y_pred']))

    # RMSLE
    rmsle_weekly = np.sqrt(np.mean((np.log1p(weekly['y']) - np.log1p(weekly['y_pred']))**2))

    bias_weekly = np.mean(weekly['y_pred'] - weekly['y'])
    bias_pct_weekly = 100 * bias_weekly / np.mean(weekly['y'])

    print(f"  Weekly WMAPE: {wmape_weekly:.2f}%")
    print(f"  Weekly WFA (1-WMAPE): {wfa_weekly:.2f}%")
    print(f"  Weekly MAE: {mae_weekly:.2f}")
    print(f"  Weekly RMSLE: {rmsle_weekly:.4f}")
    print(f"  Weekly Bias: {bias_pct_weekly:.2f}%")

    results['weekly'] = {
        'wmape': wmape_weekly,
        'wfa': wfa_weekly,
        'mae': mae_weekly,
        'rmsle': rmsle_weekly,
        'bias_pct': bias_pct_weekly
    }

    # ==================================================================
    # B) ABC SEGMENTATION (by sales share)
    # ==================================================================
    print("\n" + "="*60)
    print("B) ABC ITEM SEGMENTATION (Top Sellers Accuracy)")
    print("="*60)

    # Calculate total sales per series in training data
    train['sku_id'] = train['sku_id'].astype(str)
    train['store_id'] = train['store_id'].astype(str)
    series_sales = train.groupby(['store_id', 'sku_id'])['y'].sum().reset_index()
    series_sales.columns = ['store_id', 'sku_id', 'total_train_sales']
    series_sales = series_sales.sort_values('total_train_sales', ascending=False)

    # Calculate cumulative share
    total_sales = series_sales['total_train_sales'].sum()
    series_sales['cumulative_share'] = series_sales['total_train_sales'].cumsum() / total_sales

    # Assign ABC categories
    series_sales['abc_tier'] = 'C'  # Default
    series_sales.loc[series_sales['cumulative_share'] <= 0.80, 'abc_tier'] = 'A'
    series_sales.loc[(series_sales['cumulative_share'] > 0.80) &
                     (series_sales['cumulative_share'] <= 0.95), 'abc_tier'] = 'B'

    # Count series in each tier
    tier_counts = series_sales['abc_tier'].value_counts()
    print(f"  A-items (top 80% sales): {tier_counts.get('A', 0):,} series")
    print(f"  B-items (next 15% sales): {tier_counts.get('B', 0):,} series")
    print(f"  C-items (last 5% sales): {tier_counts.get('C', 0):,} series")

    # Merge ABC tier to validation data
    val['store_id'] = val['store_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)
    val_abc = val.merge(series_sales[['store_id', 'sku_id', 'abc_tier']],
                        on=['store_id', 'sku_id'], how='left')
    val_abc['abc_tier'] = val_abc['abc_tier'].fillna('C')

    results['abc'] = {}

    for tier in ['A', 'B', 'C']:
        tier_data = val_abc[val_abc['abc_tier'] == tier]
        if len(tier_data) > 0 and tier_data['y'].sum() > 0:
            wmape_tier = 100 * np.sum(np.abs(tier_data['y'] - tier_data['y_pred'])) / np.sum(tier_data['y'])
            wfa_tier = 100 - wmape_tier
            mae_tier = np.mean(np.abs(tier_data['y'] - tier_data['y_pred']))

            print(f"\n  {tier}-items:")
            print(f"    WMAPE: {wmape_tier:.2f}%")
            print(f"    WFA (1-WMAPE): {wfa_tier:.2f}%")
            print(f"    MAE: {mae_tier:.4f}")
            print(f"    Val rows: {len(tier_data):,}")

            results['abc'][tier] = {
                'wmape': wmape_tier,
                'wfa': wfa_tier,
                'mae': mae_tier,
                'n_rows': len(tier_data)
            }

    # ==================================================================
    # C) NON-ZERO DAY ACCURACY
    # ==================================================================
    print("\n" + "="*60)
    print("C) NON-ZERO DAY ACCURACY (When demand happens)")
    print("="*60)

    # Filter to non-zero actual days
    nz_days = val[val['y'] > 0]

    mae_nonzero = np.mean(np.abs(nz_days['y'] - nz_days['y_pred']))
    bias_nonzero = np.mean(nz_days['y_pred'] - nz_days['y'])
    bias_pct_nonzero = 100 * bias_nonzero / np.mean(nz_days['y'])

    # Band metric: % within ±50% for y >= 2
    nz_days_gt2 = nz_days[nz_days['y'] >= 2]
    if len(nz_days_gt2) > 0:
        pct_error = np.abs(nz_days_gt2['y_pred'] - nz_days_gt2['y']) / nz_days_gt2['y']
        within_50pct = 100 * np.mean(pct_error <= 0.50)
        within_30pct = 100 * np.mean(pct_error <= 0.30)
    else:
        within_50pct = np.nan
        within_30pct = np.nan

    # WMAPE for non-zero days
    wmape_nonzero = 100 * np.sum(np.abs(nz_days['y'] - nz_days['y_pred'])) / np.sum(nz_days['y'])
    wfa_nonzero = 100 - wmape_nonzero

    print(f"  Non-zero days: {len(nz_days):,}")
    print(f"  WMAPE (non-zero days): {wmape_nonzero:.2f}%")
    print(f"  WFA (non-zero days): {wfa_nonzero:.2f}%")
    print(f"  MAE (non-zero days): {mae_nonzero:.4f}")
    print(f"  Bias (non-zero days): {bias_pct_nonzero:.2f}%")
    print(f"  % within ±50% (y≥2): {within_50pct:.1f}%")
    print(f"  % within ±30% (y≥2): {within_30pct:.1f}%")

    results['nonzero'] = {
        'wmape': wmape_nonzero,
        'wfa': wfa_nonzero,
        'mae': mae_nonzero,
        'bias_pct': bias_pct_nonzero,
        'pct_within_50': within_50pct,
        'pct_within_30': within_30pct,
        'n_days': len(nz_days)
    }

    # ==================================================================
    # D) LIFT VS BASELINE
    # ==================================================================
    print("\n" + "="*60)
    print("D) LIFT VS BQML BASELINE")
    print("="*60)

    baseline_wmape = 53.76  # BQML XGBoost baseline
    current_wmape = 100 * np.sum(np.abs(val['y'] - val['y_pred'])) / np.sum(val['y'])

    lift = (baseline_wmape - current_wmape) / baseline_wmape * 100

    print(f"  BQML Baseline WMAPE: {baseline_wmape:.2f}%")
    print(f"  C1+B1 Model WMAPE: {current_wmape:.2f}%")
    print(f"  Absolute Improvement: {baseline_wmape - current_wmape:.2f} pp")
    print(f"  Relative Lift: {lift:.1f}%")

    results['lift'] = {
        'baseline_wmape': baseline_wmape,
        'model_wmape': current_wmape,
        'absolute_improvement': baseline_wmape - current_wmape,
        'relative_lift_pct': lift
    }

    # ==================================================================
    # DAILY (original) for reference
    # ==================================================================
    results['daily'] = {
        'wmape': current_wmape,
        'wfa': 100 - current_wmape,
        'mae': np.mean(np.abs(val['y'] - val['y_pred'])),
    }

    return results


def main():
    print("="*70)
    print("BUSINESS-FOCUSED ACCURACY METRICS")
    print("="*70)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading T1_MATURE F1 data...")
    train = load_sharded_csvs('/tmp/full_data/train')
    val = load_sharded_csvs('/tmp/full_data/val')
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    print(f"Train: {len(train):,}, Val: {len(val):,}")

    # Get predictions
    print("\nTraining model and getting predictions...")
    val_pred, train_data = get_predictions(train, val, sku_attr)

    # Calculate business metrics
    results = calculate_business_metrics(val_pred, train_data)

    # ==================================================================
    # EXECUTIVE SUMMARY
    # ==================================================================
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY - T1_MATURE")
    print("="*70)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  HEADLINE METRICS (Recommended for Presentation)                │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Weekly Forecast Accuracy (WFA):      {results['weekly']['wfa']:>6.1f}%                 │")
    print(f"│  A-Items Forecast Accuracy (Top 80%): {results['abc']['A']['wfa']:>6.1f}%                 │")
    print(f"│  Non-Zero Day Accuracy:               {results['nonzero']['wfa']:>6.1f}%                 │")
    print(f"│  Lift vs Baseline:                    {results['lift']['relative_lift_pct']:>6.1f}%                 │")
    print("└─────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  DETAILED METRICS                                               │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Daily WMAPE:             {results['daily']['wmape']:>6.2f}%                            │")
    print(f"│  Weekly WMAPE:            {results['weekly']['wmape']:>6.2f}%                            │")
    print(f"│  Weekly RMSLE:            {results['weekly']['rmsle']:>6.4f}                            │")
    print(f"│  A-Items WMAPE:           {results['abc']['A']['wmape']:>6.2f}%                            │")
    print(f"│  B-Items WMAPE:           {results['abc']['B']['wmape']:>6.2f}%                            │")
    print(f"│  C-Items WMAPE:           {results['abc']['C']['wmape']:>6.2f}%                            │")
    print(f"│  % within ±50% (y≥2):     {results['nonzero']['pct_within_50']:>6.1f}%                            │")
    print(f"│  % within ±30% (y≥2):     {results['nonzero']['pct_within_30']:>6.1f}%                            │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Save results
    os.makedirs('/tmp/business_metrics', exist_ok=True)
    with open('/tmp/business_metrics/t1_mature_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to /tmp/business_metrics/t1_mature_metrics.json")
    print(f"Finished: {datetime.now()}")

    return results


if __name__ == "__main__":
    main()
