"""
PRODUCTION FORECAST: 168-Day Daily Predictions
===============================================
Generate daily SKU-Store level forecasts for all tiers (T1, T2, T3)
using per-segment (A/B/C) two-stage LightGBM models.

Model: Binary Classifier (P(y>0)) + Log-transform Regressor (log1p(y) for y>0)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
import os
from datetime import datetime
import json
import gc
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

# Best hyperparameters per segment (from accuracy_at_all_levels.py)
SEGMENT_PARAMS = {
    'A': {'nl': 255, 'lr': 0.015, 'rnds_clf': 800, 'rnds_reg': 1000, 'mdl': 10, 'thresh': 0.6},
    'B': {'nl': 63, 'lr': 0.03, 'rnds_clf': 300, 'rnds_reg': 400, 'mdl': 50, 'thresh': 0.6},
    'C': {'nl': 31, 'lr': 0.05, 'rnds_clf': 200, 'rnds_reg': 300, 'mdl': 100, 'thresh': 0.7},
}


def load_data(folder):
    """Load all CSV files from a folder."""
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    print(f"    Loading {len(files)} files from {folder}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"    Loaded {len(df):,} rows")
    return df


def prepare_data(df, sku_attr):
    """Prepare features for model."""
    df = df.copy()
    df['sku_id'] = df['sku_id'].astype(str)
    df['store_id'] = df['store_id'].astype(str)

    # Merge is_local
    if 'is_local' not in df.columns:
        sku_attr_local = sku_attr[['sku_id', 'is_local']].copy()
        sku_attr_local['sku_id'] = sku_attr_local['sku_id'].astype(str)
        df = df.merge(sku_attr_local, on='sku_id', how='left')

    df['is_local'] = df['is_local'].fillna(0).astype(int)

    # Handle boolean columns (from BQ export as string)
    for col in ['is_weekend', 'is_store_closed', 'is_closure_week']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({'true': 1, 'false': 0, True: 1, False: 0}).fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(0).astype(int)

    # Fill missing features
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    return df


def assign_abc(train, forecast):
    """Assign ABC tiers based on training sales volume."""
    series_sales = train.groupby(['store_id', 'sku_id'])['y'].sum().reset_index()
    series_sales.columns = ['store_id', 'sku_id', 'total_sales']
    series_sales = series_sales.sort_values('total_sales', ascending=False)
    total = series_sales['total_sales'].sum()
    series_sales['cum_share'] = series_sales['total_sales'].cumsum() / total
    series_sales['abc'] = 'C'
    series_sales.loc[series_sales['cum_share'] <= 0.80, 'abc'] = 'A'
    series_sales.loc[(series_sales['cum_share'] > 0.80) & (series_sales['cum_share'] <= 0.95), 'abc'] = 'B'

    train = train.merge(series_sales[['store_id', 'sku_id', 'abc']], on=['store_id', 'sku_id'], how='left')
    forecast = forecast.merge(series_sales[['store_id', 'sku_id', 'abc']], on=['store_id', 'sku_id'], how='left')
    train['abc'] = train['abc'].fillna('C')
    forecast['abc'] = forecast['abc'].fillna('C')

    counts = series_sales['abc'].value_counts()
    print(f"    ABC: A={counts.get('A', 0):,}, B={counts.get('B', 0):,}, C={counts.get('C', 0):,}")

    return train, forecast


def train_and_predict_segment(train_seg, forecast_seg, seg_name):
    """Train per-segment model and predict."""
    if len(train_seg) == 0 or len(forecast_seg) == 0:
        return np.zeros(len(forecast_seg)), None

    params = SEGMENT_PARAMS[seg_name]
    nl, lr, rnds_clf, rnds_reg, mdl, thresh = (
        params['nl'], params['lr'], params['rnds_clf'],
        params['rnds_reg'], params['mdl'], params['thresh']
    )

    train_seg = train_seg.copy()
    forecast_seg = forecast_seg.copy()
    train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train_seg[col] = train_seg[col].astype('category')
        forecast_seg[col] = forecast_seg[col].astype('category')

    X_train = train_seg[FEATURES + CAT_FEATURES]
    X_forecast = forecast_seg[FEATURES + CAT_FEATURES]

    # --- Classifier ---
    clf_data = lgb.Dataset(X_train, label=train_seg['y_binary'], categorical_feature=CAT_FEATURES)
    clf = lgb.train({
        'objective': 'binary', 'metric': 'auc', 'num_leaves': nl,
        'learning_rate': lr, 'feature_fraction': 0.8, 'min_data_in_leaf': mdl,
        'verbose': -1, 'n_jobs': -1
    }, clf_data, num_boost_round=rnds_clf)
    prob = clf.predict(X_forecast)

    # --- Regressor (on non-zero training data only) ---
    train_nz = train_seg[train_seg['y'] > 0]
    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    y_train_nz = np.log1p(train_nz['y'].values)

    reg_data = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
    reg = lgb.train({
        'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': nl,
        'learning_rate': lr, 'feature_fraction': 0.8, 'min_data_in_leaf': max(5, mdl // 2),
        'lambda_l2': 0.5, 'verbose': -1, 'n_jobs': -1
    }, reg_data, num_boost_round=rnds_reg)
    pred_value = np.expm1(reg.predict(X_forecast))

    # --- Combine ---
    y_pred = np.where(prob > thresh, pred_value, 0)
    y_pred = np.maximum(0, y_pred)

    # Zero out closed stores
    if 'is_store_closed' in forecast_seg.columns:
        y_pred[forecast_seg['is_store_closed'].values == 1] = 0

    # --- Calibration for A-items ---
    if seg_name == 'A':
        prob_tr = clf.predict(X_train)
        pred_val_tr = np.expm1(reg.predict(X_train))
        y_pred_tr = np.where(prob_tr > thresh, pred_val_tr, 0)
        y_pred_tr = np.maximum(0, y_pred_tr)
        mask = y_pred_tr > 0.1
        if np.sum(y_pred_tr[mask]) > 0:
            k = np.clip(
                np.sum(train_seg['y'].values[mask]) / np.sum(y_pred_tr[mask]),
                0.8, 1.3
            )
            y_pred = y_pred * k
            if 'is_store_closed' in forecast_seg.columns:
                y_pred[forecast_seg['is_store_closed'].values == 1] = 0
            print(f"      Calibration factor k = {k:.4f}")

    # Metrics (if actuals available)
    metrics = None
    if 'y' in forecast_seg.columns and forecast_seg['y'].sum() > 0:
        wmape = 100 * np.sum(np.abs(forecast_seg['y'].values - y_pred)) / np.sum(forecast_seg['y'].values)
        metrics = {'wmape': wmape, 'wfa': 100 - wmape, 'threshold': thresh}

    del clf, reg, clf_data, reg_data
    gc.collect()

    return y_pred, metrics


def process_tier(tier_name, train_folder, forecast_folder, sku_attr):
    """Process one tier: train, predict, return forecasts."""
    print(f"\n{'#'*70}")
    print(f"# TIER: {tier_name}")
    print('#' * 70)

    # Load training data
    print("\n  Loading training data...")
    train = load_data(train_folder)

    # Load forecast data
    print("  Loading forecast period data...")
    forecast = load_data(forecast_folder)

    # Prepare features
    print("  Preparing features...")
    train = prepare_data(train, sku_attr)
    forecast = prepare_data(forecast, sku_attr)

    # Assign ABC
    print("  Assigning ABC tiers...")
    train, forecast = assign_abc(train, forecast)

    # Train and predict per segment
    forecast['y_pred'] = 0.0

    for seg in ['A', 'B', 'C']:
        train_seg = train[train['abc'] == seg]
        forecast_seg = forecast[forecast['abc'] == seg]

        print(f"\n  Training {seg}-segment...")
        print(f"    Train: {len(train_seg):,}, Forecast: {len(forecast_seg):,}")

        if len(train_seg) > 0 and len(forecast_seg) > 0:
            preds, metrics = train_and_predict_segment(train_seg, forecast_seg, seg)
            forecast.loc[forecast['abc'] == seg, 'y_pred'] = preds

            if metrics:
                print(f"    {seg}-items: WMAPE={metrics['wmape']:.2f}%, WFA={metrics['wfa']:.2f}%")

    # Overall metrics
    if forecast['y'].sum() > 0:
        wmape_daily = 100 * np.sum(np.abs(forecast['y'] - forecast['y_pred'])) / np.sum(forecast['y'])
        print(f"\n  Overall Daily WMAPE: {wmape_daily:.2f}%, WFA: {100-wmape_daily:.2f}%")

        # Weekly metrics
        forecast['date'] = pd.to_datetime(forecast['date'])
        forecast['week'] = forecast['date'].dt.isocalendar().week.astype(int)
        forecast['year'] = forecast['date'].dt.year

        weekly = forecast.groupby(['store_id', 'sku_id', 'year', 'week']).agg(
            {'y': 'sum', 'y_pred': 'sum'}).reset_index()
        wmape_weekly = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
        print(f"  Weekly SKU-Store WMAPE: {wmape_weekly:.2f}%, WFA: {100-wmape_weekly:.2f}%")

        weekly_store = forecast.groupby(['store_id', 'year', 'week']).agg(
            {'y': 'sum', 'y_pred': 'sum'}).reset_index()
        wmape_store = 100 * np.sum(np.abs(weekly_store['y'] - weekly_store['y_pred'])) / np.sum(weekly_store['y'])
        print(f"  Weekly Store WMAPE: {wmape_store:.2f}%, WFA: {100-wmape_store:.2f}%")

        weekly_total = forecast.groupby(['year', 'week']).agg(
            {'y': 'sum', 'y_pred': 'sum'}).reset_index()
        wmape_total = 100 * np.sum(np.abs(weekly_total['y'] - weekly_total['y_pred'])) / np.sum(weekly_total['y'])
        print(f"  Weekly Total WMAPE: {wmape_total:.2f}%, WFA: {100-wmape_total:.2f}%")

    # Clean up training data
    del train
    gc.collect()

    # Return forecast output
    output = forecast[['store_id', 'sku_id', 'date', 'y', 'y_pred', 'abc']].copy()
    output['tier_name'] = tier_name
    output['date'] = pd.to_datetime(output['date']).dt.strftime('%Y-%m-%d')

    return output


def main():
    print("=" * 70)
    print("PRODUCTION FORECAST: 168-Day Daily Predictions")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load SKU attributes
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(lambda x: 1 if x in ['L', 'LI'] else 0)

    # Configuration per tier
    tiers = [
        {
            'name': 'T1_MATURE',
            'train_folder': '/tmp/full_data/train',
            'forecast_folder': '/tmp/forecast_data/t1',
        },
        {
            'name': 'T2_GROWING',
            'train_folder': '/tmp/t2_data/train',
            'forecast_folder': '/tmp/forecast_data/t2',
        },
        {
            'name': 'T3_COLD_START',
            'train_folder': '/tmp/t3_data/train',
            'forecast_folder': '/tmp/forecast_data/t3',
        },
    ]

    all_forecasts = []

    for tier in tiers:
        tier_forecast = process_tier(
            tier['name'], tier['train_folder'], tier['forecast_folder'], sku_attr
        )
        all_forecasts.append(tier_forecast)
        gc.collect()

    # Combine all forecasts
    print(f"\n{'='*70}")
    print("COMBINING ALL FORECASTS")
    print("=" * 70)

    combined = pd.concat(all_forecasts, ignore_index=True)
    print(f"Total forecast rows: {len(combined):,}")
    print(f"Unique series: {combined.groupby(['store_id', 'sku_id']).ngroups:,}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Tiers: {combined['tier_name'].value_counts().to_dict()}")

    # Save to CSV (split into chunks for BQ upload)
    output_dir = '/tmp/forecast_output'
    os.makedirs(output_dir, exist_ok=True)

    # Save full combined forecast
    output_file = os.path.join(output_dir, 'forecast_168day.csv')
    combined.to_csv(output_file, index=False)
    print(f"\nSaved full forecast to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1e9:.2f} GB")

    # Also save per-tier for easier upload
    for tier_name in combined['tier_name'].unique():
        tier_df = combined[combined['tier_name'] == tier_name]
        tier_file = os.path.join(output_dir, f'forecast_{tier_name.lower()}.csv')
        tier_df.to_csv(tier_file, index=False)
        print(f"  {tier_name}: {len(tier_df):,} rows -> {tier_file}")

    # FINAL SUMMARY
    print(f"\n{'='*70}")
    print("FINAL PRODUCTION FORECAST SUMMARY")
    print("=" * 70)

    print(f"\n  Forecast horizon: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Unique SKU-Store series: {combined.groupby(['store_id', 'sku_id']).ngroups:,}")

    for tier_name in ['T1_MATURE', 'T2_GROWING', 'T3_COLD_START']:
        tier_df = combined[combined['tier_name'] == tier_name]
        if len(tier_df) > 0 and tier_df['y'].sum() > 0:
            wmape = 100 * np.sum(np.abs(tier_df['y'] - tier_df['y_pred'])) / np.sum(tier_df['y'])
            print(f"\n  {tier_name}:")
            print(f"    Rows: {len(tier_df):,}")
            print(f"    Daily WMAPE: {wmape:.2f}%, WFA: {100-wmape:.2f}%")

            tier_df_copy = tier_df.copy()
            tier_df_copy['date'] = pd.to_datetime(tier_df_copy['date'])
            tier_df_copy['week'] = tier_df_copy['date'].dt.isocalendar().week.astype(int)
            tier_df_copy['year'] = tier_df_copy['date'].dt.year

            weekly = tier_df_copy.groupby(['store_id', 'sku_id', 'year', 'week']).agg(
                {'y': 'sum', 'y_pred': 'sum'}).reset_index()
            wmape_w = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
            print(f"    Weekly SKU-Store WMAPE: {wmape_w:.2f}%, WFA: {100-wmape_w:.2f}%")

            weekly_store = tier_df_copy.groupby(['store_id', 'year', 'week']).agg(
                {'y': 'sum', 'y_pred': 'sum'}).reset_index()
            wmape_s = 100 * np.sum(np.abs(weekly_store['y'] - weekly_store['y_pred'])) / np.sum(weekly_store['y'])
            print(f"    Weekly Store WMAPE: {wmape_s:.2f}%, WFA: {100-wmape_s:.2f}%")

    print(f"\nFinished: {datetime.now()}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
