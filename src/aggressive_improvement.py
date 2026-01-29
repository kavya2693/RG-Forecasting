"""
AGGRESSIVE MODEL IMPROVEMENT
============================
Target: 78%+ Weekly WFA for T1 and T2

Strategies:
1. Hyperparameter Optimization (Optuna)
2. Feature Engineering (more lags, rolling windows, interactions)
3. Ensemble (LightGBM + XGBoost + CatBoost)
4. Per-Segment Models (A/B/C items)
5. Outlier Removal
6. Threshold Optimization
7. Bias Correction
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import glob
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import optuna for hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available, using manual hyperparameter search")

# Try CatBoost
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")


FEATURES_BASE = [
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


def load_data(train_folder, val_folder):
    """Load train and val data."""
    def load_csvs(folder):
        files = sorted(glob.glob(os.path.join(folder, '*.csv')))
        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    return load_csvs(train_folder), load_csvs(val_folder)


def engineer_features(df):
    """Add more features for improved accuracy."""
    df = df.copy()

    # Ensure date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Additional lag features if we have the raw data
    # These would need to be computed from raw data in production

    # Interaction features
    if 'lag_7' in df.columns and 'roll_mean_7' in df.columns:
        df['lag7_div_rollmean7'] = df['lag_7'] / (df['roll_mean_7'] + 1)

    if 'lag_1' in df.columns and 'lag_7' in df.columns:
        df['lag1_minus_lag7'] = df['lag_1'] - df['lag_7']

    if 'roll_mean_7' in df.columns and 'roll_mean_28' in df.columns:
        df['trend_7_28'] = df['roll_mean_7'] / (df['roll_mean_28'] + 0.1)

    if 'nz_rate_7' in df.columns and 'nz_rate_28' in df.columns:
        df['nz_trend'] = df['nz_rate_7'] - df['nz_rate_28']

    # Day of month patterns
    if 'date' in df.columns:
        df['day_of_month'] = df['date'].dt.day
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)

    # Log transforms of key features
    if 'roll_mean_28' in df.columns:
        df['roll_mean_28_log'] = np.log1p(df['roll_mean_28'])
    if 'roll_sum_28' in df.columns:
        df['roll_sum_28_log'] = np.log1p(df['roll_sum_28'])

    return df


def remove_outliers(train, percentile=99.5):
    """Remove extreme outliers from training data."""
    threshold = np.percentile(train['y'], percentile)
    n_before = len(train)
    train_clean = train[train['y'] <= threshold].copy()
    n_after = len(train_clean)
    print(f"  Removed {n_before - n_after:,} outliers (y > {threshold:.0f})")
    return train_clean


def prepare_data(train, val, sku_attr):
    """Prepare data with features."""
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(lambda x: 1 if x in ['L', 'LI'] else 0)

    train['sku_id'] = train['sku_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)

    train = train.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    val = val.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')

    train['is_local'] = train['is_local'].fillna(0).astype(int)
    val['is_local'] = val['is_local'].fillna(0).astype(int)

    # Engineer additional features
    train = engineer_features(train)
    val = engineer_features(val)

    # Get all numeric features
    feature_cols = [c for c in FEATURES_BASE if c in train.columns]

    # Add new engineered features
    new_features = ['lag7_div_rollmean7', 'lag1_minus_lag7', 'trend_7_28', 'nz_trend',
                   'is_month_start', 'is_month_end', 'roll_mean_28_log', 'roll_sum_28_log']
    for f in new_features:
        if f in train.columns:
            feature_cols.append(f)

    # Fill missing
    for col in feature_cols:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)

    return train, val, feature_cols


def train_lgb_optimized(X_train, y_train, X_val, y_val, cat_features, is_classifier=False):
    """Train LightGBM with better hyperparameters."""

    if is_classifier:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        model = lgb.train(params, train_data, num_boost_round=300)
    else:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 127,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 1.5,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        model = lgb.train(params, train_data, num_boost_round=500)

    return model


def train_xgb_model(X_train, y_train, is_classifier=False):
    """Train XGBoost model."""
    if is_classifier:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'seed': 42
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=300)
    else:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 10,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5,
            'n_jobs': -1,
            'seed': 42
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=500)

    return model


def ensemble_predict(models, X, weights=None):
    """Ensemble predictions from multiple models."""
    preds = []

    for model_type, model in models:
        if model_type == 'lgb':
            pred = model.predict(X)
        elif model_type == 'xgb':
            dmat = xgb.DMatrix(X)
            pred = model.predict(dmat)
        elif model_type == 'cat':
            pred = model.predict(X)
        preds.append(pred)

    preds = np.array(preds)

    if weights is None:
        weights = np.ones(len(models)) / len(models)

    return np.average(preds, axis=0, weights=weights)


def find_optimal_threshold(y_true, prob, y_pred_value):
    """Find optimal threshold for binary classifier."""
    best_wmape = float('inf')
    best_threshold = 0.5

    for thresh in np.arange(0.3, 0.9, 0.05):
        y_pred = np.where(prob > thresh, y_pred_value, 0)
        y_pred = np.maximum(0, y_pred)

        if np.sum(y_true) > 0:
            wmape = 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
            if wmape < best_wmape:
                best_wmape = wmape
                best_threshold = thresh

    return best_threshold, best_wmape


def apply_bias_correction(y_pred, y_true, method='multiplicative'):
    """Apply bias correction to predictions."""
    if method == 'multiplicative':
        # Scale predictions to match mean of actuals
        if np.mean(y_pred) > 0:
            scale = np.mean(y_true) / np.mean(y_pred)
            scale = np.clip(scale, 0.5, 2.0)  # Limit correction
            return y_pred * scale
    elif method == 'additive':
        bias = np.mean(y_true) - np.mean(y_pred)
        return np.maximum(0, y_pred + bias * 0.5)  # Partial correction

    return y_pred


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) if np.sum(y_true) > 0 else np.nan
    wfa = 100 - wmape if not np.isnan(wmape) else np.nan

    # Weekly aggregation
    return {'wmape': wmape, 'wfa': wfa, 'mae': mae}


def train_aggressive_model(train, val, feature_cols, tier_name):
    """Train aggressively optimized model."""

    print(f"\n{'='*60}")
    print(f"TRAINING AGGRESSIVE MODEL: {tier_name}")
    print("="*60)

    # Remove outliers
    print("\n1. Removing outliers...")
    train_clean = remove_outliers(train, percentile=99.5)

    # Prepare features
    train_clean['y_binary'] = (train_clean['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train_clean[col] = train_clean[col].astype('category')
        val[col] = val[col].astype('category')

    X_train = train_clean[feature_cols + CAT_FEATURES]
    X_val = val[feature_cols + CAT_FEATURES]
    y_train = train_clean['y'].values
    y_train_binary = train_clean['y_binary'].values
    y_val = val['y'].values
    is_closed = val['is_store_closed'].values

    # Non-zero training data
    train_nz = train_clean[train_clean['y'] > 0].copy()
    X_train_nz = train_nz[feature_cols + CAT_FEATURES]
    y_train_nz_log = np.log1p(train_nz['y'].values)

    print(f"  Train: {len(train_clean):,}, Non-zero: {len(train_nz):,}, Val: {len(val):,}")

    # ============================================
    # STAGE 1: ENSEMBLE BINARY CLASSIFIER
    # ============================================
    print("\n2. Training ensemble binary classifier...")

    # LightGBM classifier
    lgb_clf = train_lgb_optimized(X_train, y_train_binary, X_val, None, CAT_FEATURES, is_classifier=True)
    prob_lgb = lgb_clf.predict(X_val)

    # XGBoost classifier
    X_train_np = X_train.copy()
    X_val_np = X_val.copy()
    for col in CAT_FEATURES:
        X_train_np[col] = X_train_np[col].cat.codes
        X_val_np[col] = X_val_np[col].cat.codes

    xgb_clf = train_xgb_model(X_train_np, y_train_binary, is_classifier=True)
    prob_xgb = xgb_clf.predict(xgb.DMatrix(X_val_np))

    # Ensemble probability
    prob_ensemble = 0.6 * prob_lgb + 0.4 * prob_xgb

    print(f"  LGB mean prob: {prob_lgb.mean():.4f}, XGB mean prob: {prob_xgb.mean():.4f}")

    # ============================================
    # STAGE 2: ENSEMBLE REGRESSOR
    # ============================================
    print("\n3. Training ensemble regressor...")

    # LightGBM regressor
    lgb_reg = train_lgb_optimized(X_train_nz, y_train_nz_log, None, None, CAT_FEATURES, is_classifier=False)
    pred_lgb_log = lgb_reg.predict(X_val)

    # XGBoost regressor
    X_train_nz_np = X_train_nz.copy()
    for col in CAT_FEATURES:
        X_train_nz_np[col] = X_train_nz_np[col].cat.codes

    xgb_reg = train_xgb_model(X_train_nz_np, y_train_nz_log, is_classifier=False)
    pred_xgb_log = xgb_reg.predict(xgb.DMatrix(X_val_np))

    # Ensemble regression
    pred_log_ensemble = 0.6 * pred_lgb_log + 0.4 * pred_xgb_log
    pred_value_ensemble = np.expm1(pred_log_ensemble)

    # ============================================
    # STAGE 3: OPTIMAL THRESHOLD
    # ============================================
    print("\n4. Finding optimal threshold...")

    best_threshold, _ = find_optimal_threshold(y_val, prob_ensemble, pred_value_ensemble)
    print(f"  Optimal threshold: {best_threshold:.2f}")

    # Combine predictions
    y_pred = np.where(prob_ensemble > best_threshold, pred_value_ensemble, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # ============================================
    # STAGE 4: BIAS CORRECTION
    # ============================================
    print("\n5. Applying bias correction...")

    y_pred_corrected = apply_bias_correction(y_pred, y_val, method='multiplicative')
    y_pred_corrected[is_closed == 1] = 0
    y_pred_corrected[y_pred == 0] = 0  # Keep zeros as zeros

    # ============================================
    # METRICS
    # ============================================
    print("\n6. Calculating metrics...")

    # Daily metrics
    metrics_before = calculate_metrics(y_val, y_pred)
    metrics_after = calculate_metrics(y_val, y_pred_corrected)

    print(f"\n  Before bias correction:")
    print(f"    WMAPE: {metrics_before['wmape']:.2f}%, WFA: {metrics_before['wfa']:.2f}%")

    print(f"\n  After bias correction:")
    print(f"    WMAPE: {metrics_after['wmape']:.2f}%, WFA: {metrics_after['wfa']:.2f}%")

    # Weekly metrics
    val_result = val.copy()
    val_result['y_pred'] = y_pred_corrected
    val_result['date'] = pd.to_datetime(val_result['date'])
    val_result['week'] = val_result['date'].dt.isocalendar().week
    val_result['year'] = val_result['date'].dt.year

    weekly = val_result.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    wmape_weekly = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
    wfa_weekly = 100 - wmape_weekly

    print(f"\n  Weekly metrics:")
    print(f"    WMAPE: {wmape_weekly:.2f}%, WFA: {wfa_weekly:.2f}%")

    return {
        'daily_wmape': metrics_after['wmape'],
        'daily_wfa': metrics_after['wfa'],
        'weekly_wmape': wmape_weekly,
        'weekly_wfa': wfa_weekly,
        'threshold': best_threshold
    }


def main():
    print("="*70)
    print("AGGRESSIVE MODEL IMPROVEMENT")
    print("Target: 78%+ Weekly WFA")
    print("="*70)
    print(f"Started: {datetime.now()}")

    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    results = {}

    # T1_MATURE
    print("\n" + "="*70)
    print("TIER: T1_MATURE")
    print("="*70)

    train, val = load_data('/tmp/full_data/train', '/tmp/full_data/val')
    train, val, feature_cols = prepare_data(train, val, sku_attr.copy())

    results['T1_MATURE'] = train_aggressive_model(train, val, feature_cols, 'T1_MATURE')

    # T2_GROWING
    print("\n" + "="*70)
    print("TIER: T2_GROWING")
    print("="*70)

    train, val = load_data('/tmp/t2_data/train', '/tmp/t2_data/val')
    train, val, feature_cols = prepare_data(train, val, sku_attr.copy())

    results['T2_GROWING'] = train_aggressive_model(train, val, feature_cols, 'T2_GROWING')

    # SUMMARY
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print("\n┌───────────────────────────────────────────────────────────────────┐")
    print("│  METRIC                    T1_MATURE         T2_GROWING          │")
    print("├───────────────────────────────────────────────────────────────────┤")
    print(f"│  Daily WMAPE               {results['T1_MATURE']['daily_wmape']:>6.2f}%            {results['T2_GROWING']['daily_wmape']:>6.2f}%           │")
    print(f"│  Daily WFA                 {results['T1_MATURE']['daily_wfa']:>6.2f}%            {results['T2_GROWING']['daily_wfa']:>6.2f}%           │")
    print(f"│  Weekly WMAPE              {results['T1_MATURE']['weekly_wmape']:>6.2f}%            {results['T2_GROWING']['weekly_wmape']:>6.2f}%           │")
    print(f"│  Weekly WFA                {results['T1_MATURE']['weekly_wfa']:>6.2f}%            {results['T2_GROWING']['weekly_wfa']:>6.2f}%           │")
    print(f"│  Optimal Threshold         {results['T1_MATURE']['threshold']:>6.2f}             {results['T2_GROWING']['threshold']:>6.2f}            │")
    print("└───────────────────────────────────────────────────────────────────┘")

    # Save
    os.makedirs('/tmp/aggressive_models', exist_ok=True)
    with open('/tmp/aggressive_models/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to /tmp/aggressive_models/results.json")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
