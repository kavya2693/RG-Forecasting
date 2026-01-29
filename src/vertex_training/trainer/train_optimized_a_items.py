"""
OPTIMIZED A-ITEMS TRAINING
==========================
Target: 62% A-items daily accuracy (up from 58.5%)

Optimization parameters:
- calibration_factor = 1.25 (up from 1.15)
- num_leaves = 511 (up from 255) for A-items
- threshold = 0.45 (down from 0.50) for A-items
- learning_rate = 0.012 for A-items
- n_estimators = 1200 for A-items
- min_data_in_leaf = 5 for A-items

Run on Vertex AI with highmem-32 for full data.
"""

import sys
import json
import argparse
from google.cloud import storage
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import io
import gc


def log(msg):
    """Print and flush for Cloud Logging."""
    print(msg, flush=True)
    sys.stdout.flush()


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
FEATURES = [
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'is_store_closed',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_28',
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
    'feat_store_spike_pct', 'feat_store_promo_day', 'feat_seasonal_lift',
    'feat_had_recent_spike', 'feat_historical_spike_prob'
]
CAT_FEATURES = ['store_id', 'sku_id']

# OPTIMIZED SEGMENT PARAMETERS
# A-items: Significantly increased complexity for better accuracy
SEGMENT_PARAMS = {
    'A': {
        'num_leaves': 511,           # Up from 127/255 - more complex trees
        'learning_rate': 0.012,      # Lower for better generalization with more trees
        'n_clf': 1200,               # More boosting rounds
        'n_reg': 1200,               # More boosting rounds
        'min_data': 5,               # Lower to capture fine patterns
        'reg_lambda': 0.05           # Reduced regularization for A-items
    },
    'B': {
        'num_leaves': 127,
        'learning_rate': 0.02,
        'n_clf': 400,
        'n_reg': 500,
        'min_data': 15,
        'reg_lambda': 0.2
    },
    'C': {
        'num_leaves': 63,
        'learning_rate': 0.03,
        'n_clf': 200,
        'n_reg': 300,
        'min_data': 30,
        'reg_lambda': 0.3
    },
}

# OPTIMIZATION PARAMETERS
CALIBRATION_FACTOR = 1.25  # Up from 1.15 - boost predictions
A_ITEMS_THRESHOLD = 0.45   # Down from 0.50 - more aggressive predictions


def load_parquet_from_gcs(bucket_name, prefix):
    """Load all parquet files from GCS prefix."""
    log(f"Loading from gs://{bucket_name}/{prefix}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    parquet_blobs = [b for b in blobs if b.name.endswith('.parquet')]
    log(f"  Found {len(parquet_blobs)} parquet files")

    dfs = []
    for i, blob in enumerate(parquet_blobs):
        data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(data))
        dfs.append(df)
        if (i + 1) % 50 == 0:
            log(f"    Loaded {i+1}/{len(parquet_blobs)} files...")

    result = pd.concat(dfs, ignore_index=True)
    log(f"  TOTAL: {len(result):,} rows")
    return result


def assign_segments(df, train_df=None):
    """Assign ABC segments based on sales volume."""
    if train_df is None:
        train_df = df
    sku_sales = train_df.groupby('sku_id')['y'].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = max(sku_sales.sum(), 1)
    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)
    df['segment'] = df['sku_id'].apply(lambda x: 'A' if x in a_skus else ('B' if x in b_skus else 'C'))
    return df


def get_available_features(df, requested_features):
    """Get features that exist in the dataframe."""
    available = [f for f in requested_features if f in df.columns]
    missing = [f for f in requested_features if f not in df.columns]
    if missing:
        log(f"  Missing features (will be filled with 0): {missing[:5]}...")
        for f in missing:
            df[f] = 0
    return requested_features


def train_two_stage_models(train_df, features, cat_features):
    """Train two-stage LightGBM models with optimized A-items parameters."""
    log("\nTraining two-stage models (OPTIMIZED A-ITEMS)...")
    log(f"  Calibration factor: {CALIBRATION_FACTOR}")
    log(f"  A-items threshold: {A_ITEMS_THRESHOLD}")
    models = {}

    for seg in ['A', 'B', 'C']:
        params = SEGMENT_PARAMS[seg]
        train_seg = train_df[train_df['segment'] == seg].copy()

        if len(train_seg) < 100:
            log(f"  {seg}: Skip (only {len(train_seg)} rows)")
            models[seg] = {'clf': None, 'reg': None, 'smear': 1.0}
            continue

        log(f"\n  Training segment {seg}:")
        log(f"    num_leaves={params['num_leaves']}, lr={params['learning_rate']}")
        log(f"    n_clf={params['n_clf']}, n_reg={params['n_reg']}")
        log(f"    min_data={params['min_data']}, reg_lambda={params['reg_lambda']}")

        # Prepare features
        train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)
        for col in cat_features:
            train_seg[col] = train_seg[col].astype('category')
        X = train_seg[features + cat_features]

        # Stage 1: Binary Classifier
        clf_params = {
            'objective': 'binary',
            'num_leaves': params['num_leaves'],
            'learning_rate': params['learning_rate'],
            'min_data_in_leaf': params['min_data'],
            'feature_fraction': 0.8,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'lambda_l2': params['reg_lambda'],
            'verbose': -1,
            'seed': 42,
            'num_threads': -1
        }
        clf_data = lgb.Dataset(X, label=train_seg['y_binary'], categorical_feature=cat_features)
        clf = lgb.train(clf_params, clf_data, num_boost_round=params['n_clf'])

        # Stage 2: Regression on non-zero sales (log-transformed)
        train_nz = train_seg[train_seg['y'] > 0]
        if len(train_nz) < 10:
            log(f"  {seg}: No regressor (only {len(train_nz)} non-zero rows)")
            models[seg] = {'clf': clf, 'reg': None, 'smear': 1.0}
            continue

        X_nz = train_nz[features + cat_features]
        y_log = np.log1p(train_nz['y'].values)

        reg_params = {
            'objective': 'regression_l1',  # MAE for robustness
            'num_leaves': params['num_leaves'],
            'learning_rate': params['learning_rate'],
            'min_data_in_leaf': max(3, params['min_data'] // 2),
            'feature_fraction': 0.8,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'lambda_l2': params['reg_lambda'],
            'verbose': -1,
            'seed': 42,
            'num_threads': -1
        }
        reg_data = lgb.Dataset(X_nz, label=y_log, categorical_feature=cat_features)
        reg = lgb.train(reg_params, reg_data, num_boost_round=params['n_reg'])

        # Compute Duan's smearing factor to correct log-transform bias
        pred_log = reg.predict(X_nz)
        residuals = y_log - pred_log
        smear = float(np.mean(np.exp(residuals)))

        models[seg] = {'clf': clf, 'reg': reg, 'smear': smear}
        log(f"    {seg}: {len(train_seg):,} rows, {len(train_nz):,} non-zero, smear={smear:.4f}")

    return models


def predict_expected_value(models, df, features, cat_features):
    """
    Predict using expected value formula with OPTIMIZATIONS:

    For A-items:
    - threshold = 0.45 (more aggressive)
    - calibration_factor = 1.25

    E[y] = p × mu × smear × calibration (if p > threshold)
    """
    df = df.copy()
    df['y_pred'] = 0.0
    df['prob'] = 0.0

    for seg in ['A', 'B', 'C']:
        seg_mask = df['segment'] == seg
        if seg_mask.sum() == 0:
            continue

        seg_data = df[seg_mask].copy()

        if models[seg]['clf'] is None:
            continue

        for col in cat_features:
            seg_data[col] = seg_data[col].astype('category')
        X = seg_data[features + cat_features]

        # Get probability of sale
        prob = models[seg]['clf'].predict(X)
        df.loc[seg_mask, 'prob'] = prob

        # Get expected quantity given sale
        if models[seg]['reg'] is not None:
            pred_log = models[seg]['reg'].predict(X)
            smear = models[seg]['smear']
            mu = smear * np.expm1(pred_log)
        else:
            mu = np.ones(len(X))

        # OPTIMIZED PREDICTION LOGIC
        if seg == 'A':
            # A-items: Lower threshold + higher calibration
            threshold = A_ITEMS_THRESHOLD
            calibration = CALIBRATION_FACTOR

            # Expected value with threshold
            y_pred = np.where(
                prob > threshold,
                prob * mu * calibration,
                prob * mu * calibration * 0.5  # Scaled down below threshold
            )
        else:
            # B/C items: Standard expected value
            threshold = 0.50
            calibration = 1.15
            y_pred = np.where(
                prob > threshold,
                prob * mu * calibration,
                prob * mu * calibration * 0.3
            )

        y_pred = np.maximum(0, y_pred)  # Floor at 0
        df.loc[seg_mask, 'y_pred'] = y_pred

    return df


def compute_metrics(df, level='overall'):
    """Compute WMAPE/WFA at different aggregation levels."""
    y_true = df['y'].values
    y_pred = df['y_pred'].values

    results = {}

    # Daily SKU-Store
    wmape_daily = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    results['daily_wmape'] = wmape_daily
    results['daily_wfa'] = 100 - wmape_daily

    # Bias
    results['bias'] = float(np.sum(y_pred) / max(np.sum(y_true), 1))

    # Weekly Store
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['year'] = df['date'].dt.year

    weekly_store = df.groupby(['store_id', 'year', 'week']).agg({
        'y': 'sum',
        'y_pred': 'sum'
    }).reset_index()
    wmape_weekly_store = 100 * np.sum(np.abs(weekly_store['y'] - weekly_store['y_pred'])) / max(np.sum(weekly_store['y']), 1)
    results['weekly_store_wmape'] = wmape_weekly_store
    results['weekly_store_wfa'] = 100 - wmape_weekly_store

    # Weekly Total
    weekly_total = df.groupby(['year', 'week']).agg({
        'y': 'sum',
        'y_pred': 'sum'
    }).reset_index()
    wmape_weekly_total = 100 * np.sum(np.abs(weekly_total['y'] - weekly_total['y_pred'])) / max(np.sum(weekly_total['y']), 1)
    results['weekly_total_wmape'] = wmape_weekly_total
    results['weekly_total_wfa'] = 100 - wmape_weekly_total

    return results


def save_to_gcs(df, bucket_name, path, filename):
    """Save DataFrame to GCS as CSV."""
    log(f"Saving to gs://{bucket_name}/{path}/{filename}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    csv_buffer = df.to_csv(index=False)
    blob = bucket.blob(f"{path}/{filename}")
    blob.upload_from_string(csv_buffer, content_type='text/csv')
    log(f"  Saved {len(df):,} rows")


def save_json_to_gcs(data, bucket_name, path, filename):
    """Save JSON to GCS."""
    log(f"Saving to gs://{bucket_name}/{path}/{filename}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    json_str = json.dumps(data, indent=2)
    blob = bucket.blob(f"{path}/{filename}")
    blob.upload_from_string(json_str, content_type='application/json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, default='myforecastingsales-data')
    parser.add_argument('--train-prefix', type=str, default='baseline2/f1_train/')
    parser.add_argument('--val-prefix', type=str, default='baseline2/f1_val/')
    parser.add_argument('--output-path', type=str, default='optimized_a_items')
    args = parser.parse_args()

    log("=" * 80)
    log("OPTIMIZED A-ITEMS TRAINING")
    log("Target: 62% A-items daily accuracy")
    log("=" * 80)
    log(f"Started: {datetime.now()}")
    log(f"\nOPTIMIZATION PARAMETERS:")
    log(f"  A-items num_leaves: {SEGMENT_PARAMS['A']['num_leaves']}")
    log(f"  A-items learning_rate: {SEGMENT_PARAMS['A']['learning_rate']}")
    log(f"  A-items n_estimators: {SEGMENT_PARAMS['A']['n_clf']}/{SEGMENT_PARAMS['A']['n_reg']}")
    log(f"  A-items min_data_in_leaf: {SEGMENT_PARAMS['A']['min_data']}")
    log(f"  Calibration factor: {CALIBRATION_FACTOR}")
    log(f"  A-items threshold: {A_ITEMS_THRESHOLD}")

    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    log("\n" + "=" * 60)
    log("STEP 1: LOAD DATA")
    log("=" * 60)

    train_df = load_parquet_from_gcs(args.bucket, args.train_prefix)
    val_df = load_parquet_from_gcs(args.bucket, args.val_prefix)

    log(f"\nTrain: {len(train_df):,} rows, {train_df['date'].min()} to {train_df['date'].max()}")
    log(f"Val:   {len(val_df):,} rows, {val_df['date'].min()} to {val_df['date'].max()}")

    # Ensure features exist
    features = get_available_features(train_df, FEATURES)
    get_available_features(val_df, FEATURES)

    # ==========================================================================
    # STEP 2: ASSIGN SEGMENTS (from training data)
    # ==========================================================================
    log("\n" + "=" * 60)
    log("STEP 2: ASSIGN SEGMENTS")
    log("=" * 60)

    train_df = assign_segments(train_df)
    val_df = assign_segments(val_df, train_df)  # Use train for segment assignment

    for seg in ['A', 'B', 'C']:
        train_n = (train_df['segment'] == seg).sum()
        val_n = (val_df['segment'] == seg).sum()
        log(f"  {seg}: Train={train_n:,}, Val={val_n:,}")

    # ==========================================================================
    # STEP 3: TRAIN MODELS
    # ==========================================================================
    log("\n" + "=" * 60)
    log("STEP 3: TRAIN TWO-STAGE MODELS (OPTIMIZED)")
    log("=" * 60)

    models = train_two_stage_models(train_df, features, CAT_FEATURES)

    # ==========================================================================
    # STEP 4: HOLDOUT VALIDATION
    # ==========================================================================
    log("\n" + "=" * 60)
    log("STEP 4: HOLDOUT VALIDATION")
    log("=" * 60)

    val_df = predict_expected_value(models, val_df, features, CAT_FEATURES)

    # Overall metrics
    overall_metrics = compute_metrics(val_df)
    log("\n" + "-" * 40)
    log("OVERALL HOLDOUT RESULTS:")
    log("-" * 40)
    log(f"  Daily WFA:         {overall_metrics['daily_wfa']:.2f}%")
    log(f"  Weekly Store WFA:  {overall_metrics['weekly_store_wfa']:.2f}%")
    log(f"  Weekly Total WFA:  {overall_metrics['weekly_total_wfa']:.2f}%")
    log(f"  Bias:              {overall_metrics['bias']:.3f}")

    # By segment - FOCUS ON A-ITEMS
    log("\n" + "-" * 40)
    log("BY SEGMENT (TARGET: A-items >= 62%):")
    log("-" * 40)
    segment_metrics = {}
    for seg in ['A', 'B', 'C']:
        seg_df = val_df[val_df['segment'] == seg]
        if len(seg_df) > 0 and seg_df['y'].sum() > 0:
            seg_m = compute_metrics(seg_df)
            segment_metrics[seg] = seg_m
            n_series = seg_df.groupby(['store_id', 'sku_id']).ngroups

            # Highlight A-items result
            if seg == 'A':
                target_met = "TARGET MET!" if seg_m['daily_wfa'] >= 62.0 else "TARGET NOT MET"
                log(f"  *** {seg}: Daily WFA={seg_m['daily_wfa']:.2f}% ({target_met}) ***")
                log(f"       Weekly Store={seg_m['weekly_store_wfa']:.2f}%, "
                    f"Bias={seg_m['bias']:.3f}, Series={n_series:,}")
            else:
                log(f"  {seg}: Daily WFA={seg_m['daily_wfa']:.2f}%, "
                    f"Weekly Store={seg_m['weekly_store_wfa']:.2f}%, "
                    f"Bias={seg_m['bias']:.3f}, Series={n_series:,}")

    # Save validation results
    all_results = {
        'optimization_params': {
            'calibration_factor': CALIBRATION_FACTOR,
            'a_items_threshold': A_ITEMS_THRESHOLD,
            'a_items_num_leaves': SEGMENT_PARAMS['A']['num_leaves'],
            'a_items_learning_rate': SEGMENT_PARAMS['A']['learning_rate'],
            'a_items_n_estimators': SEGMENT_PARAMS['A']['n_clf']
        },
        'overall': overall_metrics,
        'by_segment': segment_metrics,
        'target_a_items_wfa': 62.0,
        'achieved_a_items_wfa': segment_metrics.get('A', {}).get('daily_wfa', 0),
        'target_met': segment_metrics.get('A', {}).get('daily_wfa', 0) >= 62.0,
        'timestamp': datetime.now().isoformat()
    }
    save_json_to_gcs(all_results, args.bucket, args.output_path, 'optimized_a_items_metrics.json')

    # Clean up validation data
    del val_df
    gc.collect()

    # ==========================================================================
    # STEP 5: RETRAIN ON FULL DATA FOR PRODUCTION
    # ==========================================================================
    log("\n" + "=" * 60)
    log("STEP 5: RETRAIN ON FULL DATA")
    log("=" * 60)

    # Reload and combine train + val
    full_train = load_parquet_from_gcs(args.bucket, args.train_prefix)
    full_val = load_parquet_from_gcs(args.bucket, args.val_prefix)
    full_df = pd.concat([full_train, full_val], ignore_index=True)
    del full_train, full_val
    gc.collect()

    # Ensure features
    get_available_features(full_df, FEATURES)

    # Assign segments on full data
    full_df = assign_segments(full_df)

    # Train final models
    final_models = train_two_stage_models(full_df, features, CAT_FEATURES)

    # ==========================================================================
    # STEP 6: GENERATE PRODUCTION FORECAST
    # ==========================================================================
    log("\n" + "=" * 60)
    log("STEP 6: GENERATE 168-DAY PRODUCTION FORECAST")
    log("=" * 60)

    # Get last known features for each series
    full_df['date'] = pd.to_datetime(full_df['date'])
    cutoff_date = full_df['date'].max()
    cutoff_60 = cutoff_date - pd.Timedelta(days=60)
    log(f"Cutoff date: {cutoff_date.date()}")

    # Get unique series
    series = full_df[['store_id', 'sku_id', 'segment']].drop_duplicates()
    log(f"Series count: {len(series):,}")

    # Get last known feature values
    recent = full_df[full_df['date'] > cutoff_60]
    last_features = recent.groupby(['store_id', 'sku_id']).agg({
        'roll_mean_28': 'last',
        'nz_rate_28': 'last',
        'days_since_last_sale_asof': 'last',
        'dormancy_capped': 'last',
        'zero_run_length_asof': 'last',
        'last_sale_qty_asof': 'last',
        'feat_store_spike_pct': 'mean',
        'feat_historical_spike_prob': 'last',
        'feat_had_recent_spike': 'last'
    }).reset_index()

    # Create forecast dates
    forecast_start = cutoff_date + pd.Timedelta(days=1)
    forecast_dates = pd.date_range(forecast_start, periods=168, freq='D')
    log(f"Forecast range: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

    # Build forecast panel
    series['_key'] = 1
    dates_df = pd.DataFrame({'date': forecast_dates, '_key': 1})
    forecast_df = series.merge(dates_df, on='_key').drop('_key', axis=1)

    # Merge last known features
    forecast_df = forecast_df.merge(last_features, on=['store_id', 'sku_id'], how='left')

    # Fill lag features with roll_mean_28
    for lag in [1, 7, 14, 28, 56]:
        forecast_df[f'lag_{lag}'] = forecast_df['roll_mean_28'].fillna(0)
    forecast_df['roll_mean_7'] = forecast_df['roll_mean_28'].fillna(0)
    forecast_df['roll_sum_7'] = forecast_df['roll_mean_28'].fillna(0) * 7
    forecast_df['roll_sum_28'] = forecast_df['roll_mean_28'].fillna(0) * 28
    forecast_df['roll_std_28'] = 0

    # Calendar features
    forecast_df['dow'] = forecast_df['date'].dt.dayofweek
    forecast_df['is_weekend'] = forecast_df['dow'].isin([5, 6]).astype(int)
    forecast_df['week_of_year'] = forecast_df['date'].dt.isocalendar().week.astype(int)
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['day_of_year'] = forecast_df['date'].dt.dayofyear
    forecast_df['sin_doy'] = np.sin(2 * np.pi * forecast_df['day_of_year'] / 365)
    forecast_df['cos_doy'] = np.cos(2 * np.pi * forecast_df['day_of_year'] / 365)
    forecast_df['sin_dow'] = np.sin(2 * np.pi * forecast_df['dow'] / 7)
    forecast_df['cos_dow'] = np.cos(2 * np.pi * forecast_df['dow'] / 7)
    forecast_df['is_store_closed'] = 0

    # Fill spike features
    forecast_df['feat_store_promo_day'] = 0
    forecast_df['feat_seasonal_lift'] = 1.0

    # Fill any remaining NaN
    for col in features:
        if col not in forecast_df.columns:
            forecast_df[col] = 0
        forecast_df[col] = forecast_df[col].fillna(0)

    log(f"Forecast panel: {len(forecast_df):,} rows")

    # Generate predictions
    forecast_df = predict_expected_value(final_models, forecast_df, features, CAT_FEATURES)

    # Prepare output
    output = forecast_df[['sku_id', 'store_id', 'date', 'y_pred', 'segment']].copy()
    output = output.rename(columns={'sku_id': 'item_id', 'y_pred': 'predicted_sales'})
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output['predicted_sales'] = output['predicted_sales'].round(2)

    # Save forecast
    save_to_gcs(output, args.bucket, args.output_path, 'forecast_168day_optimized.csv')

    # ==========================================================================
    # FINAL REPORT
    # ==========================================================================
    log("\n" + "=" * 80)
    log("FINAL REPORT - OPTIMIZED A-ITEMS")
    log("=" * 80)

    log("\nOPTIMIZATION PARAMETERS USED:")
    log(f"  A-items num_leaves: {SEGMENT_PARAMS['A']['num_leaves']}")
    log(f"  A-items learning_rate: {SEGMENT_PARAMS['A']['learning_rate']}")
    log(f"  A-items n_estimators: {SEGMENT_PARAMS['A']['n_clf']}")
    log(f"  Calibration factor: {CALIBRATION_FACTOR}")
    log(f"  A-items threshold: {A_ITEMS_THRESHOLD}")

    log("\nHOLDOUT VALIDATION METRICS:")
    log(f"  Overall Daily WFA:  {overall_metrics['daily_wfa']:.2f}%")
    log(f"  A-items Daily WFA:  {segment_metrics.get('A', {}).get('daily_wfa', 0):.2f}%")
    log(f"  Target:             62.00%")

    a_wfa = segment_metrics.get('A', {}).get('daily_wfa', 0)
    if a_wfa >= 62.0:
        log(f"\n  *** TARGET MET! A-items WFA = {a_wfa:.2f}% >= 62.00% ***")
    else:
        log(f"\n  Target not met: {a_wfa:.2f}% < 62.00% (gap: {62.0 - a_wfa:.2f}%)")

    log("\nPRODUCTION FORECAST:")
    log(f"  Rows:       {len(output):,}")
    log(f"  Series:     {output.groupby(['item_id', 'store_id']).ngroups:,}")
    log(f"  Date range: {output['date'].min()} to {output['date'].max()}")

    log("\nSANITY CHECKS:")
    log(f"  [{'PASS' if (output['predicted_sales'] >= 0).all() else 'FAIL'}] No negative predictions")
    log(f"  [{'PASS' if output['predicted_sales'].notna().all() else 'FAIL'}] No NaN predictions")
    log(f"  [{'PASS' if output['date'].nunique() == 168 else 'FAIL'}] Correct date range")

    log("\n" + "=" * 80)
    log(f"Completed: {datetime.now()}")
    log("=" * 80)


if __name__ == "__main__":
    main()
