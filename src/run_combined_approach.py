"""
Combined Approach: C1 + B1 (Log Transform + Two-Stage)
=====================================================
Classifier for zero/non-zero, then log-transform regression
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
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


def load_data():
    """Load train and validation data."""
    print("Loading data...")
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)

    for col in FEATURES:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)

    print(f"  Train: {len(train):,} rows")
    print(f"  Val: {len(val):,} rows")
    return train, val


def combined_c1_b1(train, val):
    """Combined C1 + B1: Two-stage with log transform on Stage 2."""
    print("\n" + "="*60)
    print("COMBINED APPROACH: C1 + B1 (Two-Stage + Log Transform)")
    print("="*60)

    train = train.copy()
    val_copy = val.copy()

    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    X_val = val_copy[FEATURES + CAT_FEATURES]
    y_val = val_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    # Stage 1: Binary classifier
    print("Training Stage 1: Zero classifier...")
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

    train_clf = lgb.Dataset(X_train, label=train['y_binary'], categorical_feature=CAT_FEATURES)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(X_val)

    # Stage 2: Log-transform regression on non-zeros
    print("Training Stage 2: Log-transform regressor on non-zeros...")
    train_nonzero = train[train['y'] > 0].copy()
    train_nonzero['y_log'] = np.log1p(train_nonzero['y'])

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

    X_train_nz = train_nonzero[FEATURES + CAT_FEATURES]
    y_train_nz = train_nonzero['y_log']

    train_reg = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)

    # Predict
    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)

    # Test different thresholds
    print("\nResults by threshold:")
    print("-" * 40)
    best_wmape = 999
    best_thresh = 0.5

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = np.where(prob_nonzero > thresh, y_pred_value, 0)
        y_pred = np.maximum(0, y_pred)
        y_pred[is_closed == 1] = 0
        wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
        print(f"  Threshold {thresh}: WMAPE = {wmape:.2f}%")
        if wmape < best_wmape:
            best_wmape = wmape
            best_thresh = thresh

    print(f"\nBest: Threshold {best_thresh} with WMAPE = {best_wmape:.2f}%")

    # Also try multiplying by probability (soft gating)
    print("\nSoft gating (multiply by probability):")
    y_pred_soft = prob_nonzero * y_pred_value
    y_pred_soft = np.maximum(0, y_pred_soft)
    y_pred_soft[is_closed == 1] = 0
    wmape_soft = 100 * np.sum(np.abs(y_val - y_pred_soft)) / np.sum(y_val)
    print(f"  Soft gating WMAPE = {wmape_soft:.2f}%")

    return {'approach': 'C1+B1_combined', 'best_wmape': best_wmape, 'soft_wmape': wmape_soft}


def c1_with_more_trees(train, val):
    """C1 with more boosting rounds to see if we can improve further."""
    print("\n" + "="*60)
    print("C1 LOG TRANSFORM WITH MORE TREES")
    print("="*60)

    train = train.copy()
    val_copy = val.copy()

    train['y_log'] = np.log1p(train['y'])

    for col in CAT_FEATURES:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train[FEATURES + CAT_FEATURES]
    y_train = train['y_log']
    X_val = val_copy[FEATURES + CAT_FEATURES]
    y_val = val_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 127,  # More leaves
        'learning_rate': 0.03,  # Lower LR
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': -1
    }

    print("Training with 500 rounds, 127 leaves...")
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_data, num_boost_round=500)

    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
    print(f"C1 Extended WMAPE: {wmape:.2f}%")

    return {'approach': 'C1_extended', 'wmape': wmape}


def main():
    train, val = load_data()

    # Combined approach
    r1 = combined_c1_b1(train, val)

    # C1 with more trees
    r2 = c1_with_more_trees(train, val)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"C1+B1 Combined Best:  {r1['best_wmape']:.2f}%")
    print(f"C1+B1 Soft Gating:    {r1['soft_wmape']:.2f}%")
    print(f"C1 Extended:          {r2['wmape']:.2f}%")
    print(f"\nPrevious best (C1):   49.59%")
    print(f"BQML Baseline:        53.76%")


if __name__ == "__main__":
    main()
