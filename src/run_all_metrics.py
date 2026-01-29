"""
Calculate comprehensive metrics for all approaches
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

TRAIN_FILE = '/tmp/c1_data/f1_train_500k.csv'
VAL_FILE = '/tmp/c1_data/f1_val_500k.csv'

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


def calculate_metrics(y_true, y_pred, approach_name):
    """Calculate comprehensive metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)

    # WMAPE (Weighted Mean Absolute Percentage Error)
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

    # MAPE (only on non-zero actuals to avoid division by zero)
    nonzero_mask = y_true > 0
    if np.sum(nonzero_mask) > 0:
        mape = 100 * np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
    else:
        mape = np.nan

    # R-squared
    r2 = r2_score(y_true, y_pred)

    # Bias (mean error - positive means over-prediction)
    bias = np.mean(y_pred - y_true)
    bias_pct = 100 * bias / np.mean(y_true)

    # Zero classification metrics
    actual_zero = (y_true == 0).astype(int)
    pred_zero = (y_pred == 0).astype(int)

    zero_accuracy = accuracy_score(actual_zero, pred_zero)
    zero_precision = precision_score(actual_zero, pred_zero, zero_division=0)
    zero_recall = recall_score(actual_zero, pred_zero, zero_division=0)
    zero_f1 = f1_score(actual_zero, pred_zero, zero_division=0)

    # Metrics on non-zero actuals only
    if np.sum(nonzero_mask) > 0:
        mae_nonzero = mean_absolute_error(y_true[nonzero_mask], y_pred[nonzero_mask])
        rmse_nonzero = np.sqrt(mean_squared_error(y_true[nonzero_mask], y_pred[nonzero_mask]))
    else:
        mae_nonzero = np.nan
        rmse_nonzero = np.nan

    # Percentile errors
    abs_errors = np.abs(y_true - y_pred)
    p50_error = np.percentile(abs_errors, 50)
    p90_error = np.percentile(abs_errors, 90)
    p99_error = np.percentile(abs_errors, 99)

    return {
        'Approach': approach_name,
        'WMAPE': wmape,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'Bias': bias,
        'Bias%': bias_pct,
        'Zero_Acc': zero_accuracy,
        'Zero_Prec': zero_precision,
        'Zero_Rec': zero_recall,
        'Zero_F1': zero_f1,
        'MAE_NonZero': mae_nonzero,
        'RMSE_NonZero': rmse_nonzero,
        'P50_Err': p50_error,
        'P90_Err': p90_error,
        'P99_Err': p99_error
    }


def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    for col in FEATURES:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)
    return train, val


def run_baseline(train, val):
    """Standard LightGBM baseline."""
    train = train.copy()
    val_copy = val.copy()

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

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
        'verbose': -1,
        'n_jobs': -1
    }

    train_data = lgb.Dataset(train[FEATURES + CAT_FEATURES], label=train['y'], categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_data, num_boost_round=300)

    y_pred = model.predict(val_copy[FEATURES + CAT_FEATURES])
    y_pred = np.maximum(0, y_pred)
    y_pred[val_copy['is_store_closed'].values == 1] = 0

    return val_copy['y'].values, y_pred


def run_c1_log(train, val):
    """C1: Log Transform."""
    train = train.copy()
    val_copy = val.copy()
    train['y_log'] = np.log1p(train['y'])

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

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
        'verbose': -1,
        'n_jobs': -1
    }

    train_data = lgb.Dataset(train[FEATURES + CAT_FEATURES], label=train['y_log'], categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_data, num_boost_round=300)

    y_pred_log = model.predict(val_copy[FEATURES + CAT_FEATURES])
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_copy['is_store_closed'].values == 1] = 0

    return val_copy['y'].values, y_pred


def run_b1_twostage(train, val, threshold=0.5):
    """B1: Two-Stage Model."""
    train = train.copy()
    val_copy = val.copy()
    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    # Stage 1: Classifier
    params_clf = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 100,
        'verbose': -1,
        'n_jobs': -1
    }

    train_clf = lgb.Dataset(train[FEATURES + CAT_FEATURES], label=train['y_binary'], categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(val_copy[FEATURES + CAT_FEATURES])

    # Stage 2: Regressor on non-zeros
    train_nz = train[train['y'] > 0].copy()
    params_reg = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 30,
        'lambda_l2': 1.5,
        'verbose': -1,
        'n_jobs': -1
    }

    train_reg = lgb.Dataset(train_nz[FEATURES + CAT_FEATURES], label=train_nz['y'], categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)
    y_pred_value = reg_model.predict(val_copy[FEATURES + CAT_FEATURES])

    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_copy['is_store_closed'].values == 1] = 0

    return val_copy['y'].values, y_pred


def run_c1_b1_combined(train, val, threshold=0.7):
    """C1+B1: Two-Stage with Log Transform."""
    train = train.copy()
    val_copy = val.copy()
    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    # Stage 1: Classifier
    params_clf = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 100,
        'verbose': -1,
        'n_jobs': -1
    }

    train_clf = lgb.Dataset(train[FEATURES + CAT_FEATURES], label=train['y_binary'], categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(val_copy[FEATURES + CAT_FEATURES])

    # Stage 2: Log-transform regressor on non-zeros
    train_nz = train[train['y'] > 0].copy()
    train_nz['y_log'] = np.log1p(train_nz['y'])

    params_reg = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 30,
        'lambda_l2': 1.5,
        'verbose': -1,
        'n_jobs': -1
    }

    train_reg = lgb.Dataset(train_nz[FEATURES + CAT_FEATURES], label=train_nz['y_log'], categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)
    y_pred_log = reg_model.predict(val_copy[FEATURES + CAT_FEATURES])
    y_pred_value = np.expm1(y_pred_log)

    y_pred = np.where(prob_nonzero > threshold, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_copy['is_store_closed'].values == 1] = 0

    return val_copy['y'].values, y_pred


def run_b3_perstore(train, val):
    """B3: Per-Store Models."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def train_store(store_id):
        train_s = train[train['store_id'] == store_id].copy()
        val_s = val[val['store_id'] == store_id].copy()

        if len(train_s) < 100 or len(val_s) < 10:
            return None, None, None

        features_no_store = [f for f in FEATURES if f not in CAT_FEATURES] + ['sku_id']
        train_s['sku_id'] = train_s['sku_id'].astype('category')
        val_s['sku_id'] = val_s['sku_id'].astype('category')

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 20,
            'lambda_l2': 1.5,
            'verbose': -1,
            'n_jobs': 1
        }

        train_data = lgb.Dataset(train_s[features_no_store], label=train_s['y'], categorical_feature=['sku_id'])
        model = lgb.train(params, train_data, num_boost_round=150)

        y_pred = model.predict(val_s[features_no_store])
        y_pred = np.maximum(0, y_pred)
        y_pred[val_s['is_store_closed'].values == 1] = 0

        return val_s.index.tolist(), val_s['y'].values, y_pred

    stores = train['store_id'].unique()
    all_idx, all_actual, all_pred = [], [], []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(train_store, s): s for s in stores}
        for future in as_completed(futures):
            idx, actual, pred = future.result()
            if idx is not None:
                all_idx.extend(idx)
                all_actual.extend(actual)
                all_pred.extend(pred)

    return np.array(all_actual), np.array(all_pred)


def main():
    train, val = load_data()

    results = []

    print("\n1/5 Running Baseline...")
    y_true, y_pred = run_baseline(train, val)
    results.append(calculate_metrics(y_true, y_pred, "Baseline_LightGBM"))

    print("2/5 Running C1 Log Transform...")
    y_true, y_pred = run_c1_log(train, val)
    results.append(calculate_metrics(y_true, y_pred, "C1_Log_Transform"))

    print("3/5 Running B1 Two-Stage (0.5)...")
    y_true, y_pred = run_b1_twostage(train, val, threshold=0.5)
    results.append(calculate_metrics(y_true, y_pred, "B1_TwoStage_0.5"))

    print("4/5 Running C1+B1 Combined (0.7)...")
    y_true, y_pred = run_c1_b1_combined(train, val, threshold=0.7)
    results.append(calculate_metrics(y_true, y_pred, "C1+B1_Combined_0.7"))

    print("5/5 Running B3 Per-Store...")
    y_true, y_pred = run_b3_perstore(train, val)
    results.append(calculate_metrics(y_true, y_pred, "B3_Per_Store"))

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder and format
    df = df.sort_values('WMAPE')

    print("\n" + "="*120)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*120)

    # Primary metrics
    print("\n### PRIMARY METRICS ###")
    cols1 = ['Approach', 'WMAPE', 'MAE', 'RMSE', 'MAPE', 'R²']
    print(df[cols1].to_string(index=False, float_format=lambda x: f'{x:.4f}' if abs(x) < 100 else f'{x:.2f}'))

    # Bias metrics
    print("\n### BIAS METRICS ###")
    cols2 = ['Approach', 'Bias', 'Bias%']
    print(df[cols2].to_string(index=False, float_format=lambda x: f'{x:.4f}' if abs(x) < 10 else f'{x:.2f}'))

    # Zero classification metrics
    print("\n### ZERO CLASSIFICATION METRICS ###")
    cols3 = ['Approach', 'Zero_Acc', 'Zero_Prec', 'Zero_Rec', 'Zero_F1']
    print(df[cols3].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Non-zero performance
    print("\n### NON-ZERO PERFORMANCE ###")
    cols4 = ['Approach', 'MAE_NonZero', 'RMSE_NonZero']
    print(df[cols4].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Error percentiles
    print("\n### ERROR PERCENTILES ###")
    cols5 = ['Approach', 'P50_Err', 'P90_Err', 'P99_Err']
    print(df[cols5].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Save to CSV
    df.to_csv('/tmp/c1_data/all_metrics.csv', index=False)
    print("\n\nFull metrics saved to /tmp/c1_data/all_metrics.csv")

    return df


if __name__ == "__main__":
    main()
