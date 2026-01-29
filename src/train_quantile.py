"""
Strategy 4: Quantile Regression for High-Volume SKUs
=====================================================
Train LightGBM with quantile loss to predict different percentiles.
This helps capture demand uncertainty and spikes.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from google.cloud import bigquery
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ID = "myforecastingsales"
DATASET = "forecasting_us"
QUANTILES = [0.5, 0.75, 0.9, 0.95]  # Median, 75th, 90th, 95th percentile

# Features for high-volume SKUs
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

def load_data_from_bq():
    """Load high-volume SKU data from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT
        store_id, sku_id, date, y, split_role,
        {', '.join(FEATURES)}
    FROM `{PROJECT_ID}.{DATASET}.trainval_f1_highvol`
    """

    print("Loading data from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df):,} rows")

    return df


def prepare_features(df):
    """Prepare features for training."""
    # Convert categoricals
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    # Fill nulls
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def train_quantile_model(X_train, y_train, X_val, y_val, quantile):
    """Train a single quantile model."""
    params = {
        'objective': 'quantile',
        'alpha': quantile,
        'metric': 'quantile',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l2': 1.0,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=CAT_FEATURES)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )

    return model


def evaluate_quantile_predictions(y_true, predictions_dict):
    """Evaluate quantile predictions."""
    results = {}

    for q, y_pred in predictions_dict.items():
        # Clip negatives and apply closure override
        y_pred = np.maximum(0, y_pred)

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        coverage = np.mean(y_true <= y_pred)  # Should be close to quantile

        results[q] = {
            'MAE': mae,
            'Coverage': coverage,
            'Expected_Coverage': q
        }

    return results


def smart_ensemble(predictions_dict, y_true):
    """
    Create smart ensemble: use higher quantile when variance is high.
    This helps capture spikes better.
    """
    p50 = predictions_dict[0.5]
    p90 = predictions_dict[0.9]

    # Calculate uncertainty as difference between p90 and p50
    uncertainty = p90 - p50

    # When uncertainty is high, lean towards higher quantile
    # uncertainty_ratio = uncertainty / (p50 + 1)
    # weight_90 = np.clip(uncertainty_ratio / 2, 0, 0.5)  # Max 50% weight to p90

    # Simple approach: use p75 as base, add uncertainty adjustment
    p75 = predictions_dict[0.75]

    # Adaptive blend based on uncertainty
    blend = np.where(
        uncertainty > p50 * 0.5,  # High uncertainty
        0.7 * p75 + 0.3 * p90,    # Lean higher
        0.8 * p50 + 0.2 * p75     # Stay conservative
    )

    return blend


def main():
    print("=" * 60)
    print("STRATEGY 4: QUANTILE REGRESSION FOR HIGH-VOLUME SKUs")
    print("=" * 60)

    # Load data
    df = load_data_from_bq()
    df = prepare_features(df)

    # Split data
    train_df = df[df['split_role'] == 'TRAIN'].copy()
    val_df = df[df['split_role'] == 'VAL'].copy()

    all_features = FEATURES + CAT_FEATURES
    X_train = train_df[all_features]
    y_train = train_df['y']
    X_val = val_df[all_features]
    y_val = val_df['y']

    print(f"\nTrain: {len(X_train):,} rows")
    print(f"Val: {len(X_val):,} rows")

    # Train quantile models
    print("\n" + "-" * 40)
    print("Training quantile models...")
    print("-" * 40)

    models = {}
    predictions = {}

    for q in QUANTILES:
        print(f"\nTraining Q{int(q*100)} model...")
        model = train_quantile_model(X_train, y_train, X_val, y_val, q)
        models[q] = model

        # Predict
        y_pred = model.predict(X_val)
        y_pred = np.maximum(0, y_pred)  # Clip negatives

        # Apply closure override
        y_pred[val_df['is_store_closed'].values == 1] = 0

        predictions[q] = y_pred

    # Evaluate individual quantiles
    print("\n" + "-" * 40)
    print("QUANTILE MODEL RESULTS")
    print("-" * 40)

    results = evaluate_quantile_predictions(y_val.values, predictions)

    for q, metrics in results.items():
        print(f"\nQ{int(q*100)}:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  Coverage: {metrics['Coverage']*100:.1f}% (expected: {metrics['Expected_Coverage']*100:.0f}%)")

    # Smart ensemble
    print("\n" + "-" * 40)
    print("SMART ENSEMBLE (Adaptive Blend)")
    print("-" * 40)

    y_ensemble = smart_ensemble(predictions, y_val.values)

    mae_ensemble = np.mean(np.abs(y_val.values - y_ensemble))
    wmape_ensemble = 100 * np.sum(np.abs(y_val.values - y_ensemble)) / np.sum(y_val.values)

    print(f"\nEnsemble MAE: {mae_ensemble:.4f}")
    print(f"Ensemble WMAPE: {wmape_ensemble:.2f}%")

    # Compare with original model
    print("\n" + "-" * 40)
    print("COMPARISON WITH ORIGINAL MODEL")
    print("-" * 40)

    # For high-volume SKUs only
    original_wmape = 53.76  # From original F1 results
    improvement = original_wmape - wmape_ensemble

    print(f"\nOriginal WMAPE (all SKUs): {original_wmape:.2f}%")
    print(f"Quantile Ensemble WMAPE (high-vol only): {wmape_ensemble:.2f}%")
    print(f"Improvement on high-vol SKUs: {improvement:+.2f}%")

    # Save predictions
    print("\n" + "-" * 40)
    print("Saving predictions...")
    print("-" * 40)

    val_df['yhat_q50'] = predictions[0.5]
    val_df['yhat_q75'] = predictions[0.75]
    val_df['yhat_q90'] = predictions[0.9]
    val_df['yhat_q95'] = predictions[0.95]
    val_df['yhat_ensemble'] = y_ensemble

    # Upload to BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET}.f1_val_pred_quantile"

    upload_df = val_df[['store_id', 'sku_id', 'date', 'y', 'is_store_closed',
                        'yhat_q50', 'yhat_q75', 'yhat_q90', 'yhat_q95', 'yhat_ensemble']].copy()

    job = client.load_table_from_dataframe(upload_df, table_id)
    job.result()

    print(f"Uploaded {len(upload_df):,} predictions to {table_id}")

    print("\n" + "=" * 60)
    print("QUANTILE REGRESSION COMPLETE")
    print("=" * 60)

    return models, predictions, results


if __name__ == "__main__":
    main()
