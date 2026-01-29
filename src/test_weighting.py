"""
Test Recency Weighting and COVID Downweighting
===============================================
Compare C1+B1 approach with different weighting schemes
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
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

# COVID panic period
COVID_PANIC_START = '2020-03-10'
COVID_PANIC_END = '2020-03-25'


def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)

    # Parse dates
    train['date'] = pd.to_datetime(train['date'])
    val['date'] = pd.to_datetime(val['date'])

    for col in FEATURES:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)

    print(f"  Train: {len(train):,} rows, date range: {train['date'].min()} to {train['date'].max()}")
    print(f"  Val: {len(val):,} rows")
    return train, val


def calculate_weights(train, recency=False, covid_downweight=False, recency_lambda=0.001):
    """Calculate sample weights based on recency and/or COVID period."""
    weights = np.ones(len(train))

    if recency:
        # Exponential decay based on days ago from max date
        max_date = train['date'].max()
        days_ago = (max_date - train['date']).dt.days
        recency_weights = np.exp(-recency_lambda * days_ago)
        weights = weights * recency_weights
        print(f"    Recency weights: min={recency_weights.min():.3f}, max={recency_weights.max():.3f}, mean={recency_weights.mean():.3f}")

    if covid_downweight:
        # Downweight COVID panic period to 0.25
        covid_mask = (train['date'] >= COVID_PANIC_START) & (train['date'] <= COVID_PANIC_END)
        covid_count = covid_mask.sum()
        weights[covid_mask] = weights[covid_mask] * 0.25
        print(f"    COVID downweight: {covid_count:,} rows affected ({100*covid_count/len(train):.2f}%)")

    return weights


def run_c1_b1_weighted(train, val, weights=None, label=""):
    """Run C1+B1 approach with optional sample weights."""
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

    train_clf = lgb.Dataset(
        X_train,
        label=train['y_binary'],
        weight=weights,
        categorical_feature=CAT_FEATURES
    )
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(X_val)

    # Stage 2: Log-transform regressor on non-zeros
    train_nz = train[train['y'] > 0].copy()
    train_nz['y_log'] = np.log1p(train_nz['y'])

    # Get weights for non-zero rows
    if weights is not None:
        nz_mask = train['y'] > 0
        weights_nz = weights[nz_mask]
    else:
        weights_nz = None

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

    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    train_reg = lgb.Dataset(
        X_train_nz,
        label=train_nz['y_log'],
        weight=weights_nz,
        categorical_feature=CAT_FEATURES
    )
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)

    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)

    # Combine with threshold 0.7
    y_pred = np.where(prob_nonzero > 0.7, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # Calculate metrics
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    return wmape, mae


def main():
    train, val = load_data()

    results = []

    # A: No weighting (baseline)
    print("\n" + "="*60)
    print("A: C1+B1 NO WEIGHTING (baseline)")
    print("="*60)
    wmape, mae = run_c1_b1_weighted(train, val, weights=None)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'A: No weighting', 'WMAPE': wmape, 'MAE': mae})

    # B: Recency weighting only
    print("\n" + "="*60)
    print("B: C1+B1 + RECENCY WEIGHTING")
    print("="*60)
    weights_b = calculate_weights(train, recency=True, covid_downweight=False, recency_lambda=0.001)
    wmape, mae = run_c1_b1_weighted(train, val, weights=weights_b)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'B: Recency only', 'WMAPE': wmape, 'MAE': mae})

    # B2: Recency with stronger decay
    print("\n" + "="*60)
    print("B2: C1+B1 + STRONGER RECENCY (lambda=0.002)")
    print("="*60)
    weights_b2 = calculate_weights(train, recency=True, covid_downweight=False, recency_lambda=0.002)
    wmape, mae = run_c1_b1_weighted(train, val, weights=weights_b2)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'B2: Stronger recency', 'WMAPE': wmape, 'MAE': mae})

    # C: COVID downweight only
    print("\n" + "="*60)
    print("C: C1+B1 + COVID DOWNWEIGHT")
    print("="*60)
    weights_c = calculate_weights(train, recency=False, covid_downweight=True)
    wmape, mae = run_c1_b1_weighted(train, val, weights=weights_c)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'C: COVID downweight', 'WMAPE': wmape, 'MAE': mae})

    # D: Both recency + COVID
    print("\n" + "="*60)
    print("D: C1+B1 + RECENCY + COVID")
    print("="*60)
    weights_d = calculate_weights(train, recency=True, covid_downweight=True, recency_lambda=0.001)
    wmape, mae = run_c1_b1_weighted(train, val, weights=weights_d)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'D: Recency + COVID', 'WMAPE': wmape, 'MAE': mae})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: WEIGHTING COMPARISON")
    print("="*60)
    print(f"\n{'Approach':<25} {'WMAPE':>10} {'MAE':>10} {'vs Baseline':>12}")
    print("-" * 60)

    baseline_wmape = results[0]['WMAPE']
    for r in sorted(results, key=lambda x: x['WMAPE']):
        diff = r['WMAPE'] - baseline_wmape
        diff_str = f"{diff:+.2f}pp" if diff != 0 else "baseline"
        print(f"{r['Approach']:<25} {r['WMAPE']:>9.2f}% {r['MAE']:>10.4f} {diff_str:>12}")

    return results


if __name__ == "__main__":
    main()
