"""
DISCIPLINED WFA IMPROVEMENT
===========================
No thrashing. Controlled experiments with 4 candidates.
Target: 70-80% Weekly WFA for T1
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


def prepare_data(train, val, sku_attr):
    """Prepare data with features."""
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

    return train, val


def assign_abc(train, val):
    """Assign ABC tiers based on training sales."""
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

    return train, val, series_sales


def weekly_metrics(val, y_pred_col='y_pred'):
    """Calculate weekly aggregated metrics."""
    val = val.copy()
    val['date'] = pd.to_datetime(val['date'])
    val['week'] = val['date'].dt.isocalendar().week
    val['year'] = val['date'].dt.year

    weekly = val.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum',
        y_pred_col: 'sum'
    }).reset_index()

    wmape = 100 * np.sum(np.abs(weekly['y'] - weekly[y_pred_col])) / np.sum(weekly['y'])
    wfa = 100 - wmape
    bias = np.mean(weekly[y_pred_col] - weekly['y'])
    bias_pct = 100 * bias / np.mean(weekly['y']) if np.mean(weekly['y']) > 0 else 0

    return {'wmape': wmape, 'wfa': wfa, 'bias_pct': bias_pct}


def step_a_error_decomposition(val, train):
    """STEP A: Error decomposition to understand where error comes from."""
    print("\n" + "="*70)
    print("STEP A: ERROR DECOMPOSITION")
    print("="*70)

    # Assign ABC
    train, val, series_sales = assign_abc(train.copy(), val.copy())

    # Need predictions first - train a baseline model
    print("\nTraining baseline model for decomposition...")

    train_copy = train.copy()
    val_copy = val.copy()
    train_copy['y_binary'] = (train_copy['y'] > 0).astype(int)

    for col in CAT_FEATURES:
        train_copy[col] = train_copy[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train_copy[FEATURES + CAT_FEATURES]
    X_val = val_copy[FEATURES + CAT_FEATURES]

    # Classifier
    clf_data = lgb.Dataset(X_train, label=train_copy['y_binary'], categorical_feature=CAT_FEATURES)
    clf = lgb.train({'objective': 'binary', 'metric': 'auc', 'num_leaves': 31, 'learning_rate': 0.05,
                     'verbose': -1, 'n_jobs': -1}, clf_data, num_boost_round=150)
    prob = clf.predict(X_val)

    # Regressor on non-zero
    train_nz = train_copy[train_copy['y'] > 0]
    X_train_nz = train_nz[FEATURES + CAT_FEATURES]
    y_train_nz = np.log1p(train_nz['y'].values)

    reg_data = lgb.Dataset(X_train_nz, label=y_train_nz, categorical_feature=CAT_FEATURES)
    reg = lgb.train({'objective': 'regression', 'metric': 'mae', 'num_leaves': 63, 'learning_rate': 0.05,
                     'verbose': -1, 'n_jobs': -1}, reg_data, num_boost_round=300)
    pred_log = reg.predict(X_val)
    pred_value = np.expm1(pred_log)

    # Combine with threshold 0.7
    y_pred = np.where(prob > 0.7, pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[val_copy['is_store_closed'].values == 1] = 0

    val_copy['y_pred'] = y_pred

    # Weekly aggregation
    val_copy['date'] = pd.to_datetime(val_copy['date'])
    val_copy['week'] = val_copy['date'].dt.isocalendar().week
    val_copy['year'] = val_copy['date'].dt.year

    weekly = val_copy.groupby(['store_id', 'sku_id', 'year', 'week', 'abc']).agg({
        'y': 'sum', 'y_pred': 'sum'
    }).reset_index()

    # Calculate p99 per series for spike detection
    series_p99 = weekly.groupby(['store_id', 'sku_id'])['y'].quantile(0.99).reset_index()
    series_p99.columns = ['store_id', 'sku_id', 'y_p99']
    weekly = weekly.merge(series_p99, on=['store_id', 'sku_id'], how='left')
    weekly['is_spike'] = (weekly['y'] > weekly['y_p99']).astype(int)

    print("\n" + "-"*70)
    print("1. BY ABC TIER (sales contribution)")
    print("-"*70)
    print(f"{'Tier':<8} {'Weekly WMAPE':<15} {'WFA':<10} {'Bias%':<12} {'Sales Share':<12}")
    print("-"*70)

    total_sales = weekly['y'].sum()
    for abc in ['A', 'B', 'C']:
        tier_data = weekly[weekly['abc'] == abc]
        if len(tier_data) > 0 and tier_data['y'].sum() > 0:
            wmape = 100 * np.sum(np.abs(tier_data['y'] - tier_data['y_pred'])) / np.sum(tier_data['y'])
            wfa = 100 - wmape
            bias = 100 * (tier_data['y_pred'].sum() - tier_data['y'].sum()) / tier_data['y'].sum()
            share = 100 * tier_data['y'].sum() / total_sales
            print(f"{abc:<8} {wmape:>10.2f}%     {wfa:>6.2f}%   {bias:>8.2f}%    {share:>8.2f}%")

    print("\n" + "-"*70)
    print("2. BY SPIKE WEEKS (y_week > p99 of series)")
    print("-"*70)

    for spike in [0, 1]:
        spike_data = weekly[weekly['is_spike'] == spike]
        if len(spike_data) > 0 and spike_data['y'].sum() > 0:
            wmape = 100 * np.sum(np.abs(spike_data['y'] - spike_data['y_pred'])) / np.sum(spike_data['y'])
            wfa = 100 - wmape
            share = 100 * spike_data['y'].sum() / total_sales
            label = "Spike" if spike == 1 else "Normal"
            print(f"{label:<10} WMAPE: {wmape:>6.2f}%, WFA: {wfa:>6.2f}%, Sales Share: {share:>6.2f}%")

    # Overall
    print("\n" + "-"*70)
    print("3. OVERALL WEEKLY METRICS")
    print("-"*70)
    wmape_all = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / np.sum(weekly['y'])
    wfa_all = 100 - wmape_all
    bias_all = 100 * (weekly['y_pred'].sum() - weekly['y'].sum()) / weekly['y'].sum()
    print(f"Weekly WMAPE: {wmape_all:.2f}%")
    print(f"Weekly WFA: {wfa_all:.2f}%")
    print(f"Bias: {bias_all:.2f}%")

    return val_copy, train_copy, clf, reg


def candidate_1_bias_calibration(val, train, clf, reg):
    """CANDIDATE 1: Multiplicative bias calibration."""
    print("\n" + "="*70)
    print("CANDIDATE 1: MULTIPLICATIVE BIAS CALIBRATION")
    print("="*70)

    # Get predictions on train to fit calibration
    train_copy = train.copy()
    for col in CAT_FEATURES:
        train_copy[col] = train_copy[col].astype('category')

    X_train = train_copy[FEATURES + CAT_FEATURES]
    prob_train = clf.predict(X_train)
    pred_log_train = reg.predict(X_train)
    pred_value_train = np.expm1(pred_log_train)

    y_pred_train = np.where(prob_train > 0.7, pred_value_train, 0)
    y_pred_train = np.maximum(0, y_pred_train)
    y_pred_train[train_copy['is_store_closed'].values == 1] = 0

    # Fit calibration factor on train
    y_train = train_copy['y'].values
    mask = y_pred_train > 0
    if np.sum(y_pred_train[mask]) > 0:
        k = np.sum(y_train[mask]) / np.sum(y_pred_train[mask])
        k = np.clip(k, 0.5, 2.5)  # Limit correction
    else:
        k = 1.0

    print(f"  Calibration factor k = {k:.4f}")

    # Apply to validation
    val_copy = val.copy()
    y_pred_cal = val_copy['y_pred'].values * k
    y_pred_cal = np.maximum(0, y_pred_cal)
    y_pred_cal[val_copy['is_store_closed'].values == 1] = 0
    y_pred_cal[val_copy['y_pred'].values == 0] = 0  # Keep zeros

    val_copy['y_pred_cal'] = y_pred_cal

    metrics = weekly_metrics(val_copy, 'y_pred_cal')
    print(f"  Weekly WMAPE: {metrics['wmape']:.2f}%")
    print(f"  Weekly WFA: {metrics['wfa']:.2f}%")
    print(f"  Bias: {metrics['bias_pct']:.2f}%")

    # A-items
    val_copy['date'] = pd.to_datetime(val_copy['date'])
    val_copy['week'] = val_copy['date'].dt.isocalendar().week
    val_copy['year'] = val_copy['date'].dt.year

    weekly_a = val_copy[val_copy['abc'] == 'A'].groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred_cal': 'sum'
    }).reset_index()
    wmape_a = 100 * np.sum(np.abs(weekly_a['y'] - weekly_a['y_pred_cal'])) / np.sum(weekly_a['y'])
    print(f"  A-items Weekly WFA: {100 - wmape_a:.2f}%")

    return val_copy, metrics, k


def candidate_2_threshold_tuning(val, train, clf, reg):
    """CANDIDATE 2: Two-stage with optimized threshold."""
    print("\n" + "="*70)
    print("CANDIDATE 2: THRESHOLD OPTIMIZATION")
    print("="*70)

    val_copy = val.copy()
    val_copy['date'] = pd.to_datetime(val_copy['date'])
    val_copy['week'] = val_copy['date'].dt.isocalendar().week
    val_copy['year'] = val_copy['date'].dt.year

    # Get raw predictions
    for col in CAT_FEATURES:
        val_copy[col] = val_copy[col].astype('category')

    X_val = val_copy[FEATURES + CAT_FEATURES]
    prob = clf.predict(X_val)
    pred_log = reg.predict(X_val)
    pred_value = np.expm1(pred_log)

    is_closed = val_copy['is_store_closed'].values

    # Find optimal threshold
    best_wmape = float('inf')
    best_thresh = 0.5

    for thresh in np.arange(0.3, 0.85, 0.05):
        y_pred = np.where(prob > thresh, pred_value, 0)
        y_pred = np.maximum(0, y_pred)
        y_pred[is_closed == 1] = 0

        val_copy['y_pred_test'] = y_pred
        weekly = val_copy.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
            'y': 'sum', 'y_pred_test': 'sum'
        }).reset_index()

        wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred_test'])) / np.sum(weekly['y'])
        if wmape < best_wmape:
            best_wmape = wmape
            best_thresh = thresh

    print(f"  Optimal threshold: {best_thresh:.2f}")

    # Apply best threshold
    y_pred_opt = np.where(prob > best_thresh, pred_value, 0)
    y_pred_opt = np.maximum(0, y_pred_opt)
    y_pred_opt[is_closed == 1] = 0

    val_copy['y_pred_opt'] = y_pred_opt

    metrics = weekly_metrics(val_copy, 'y_pred_opt')
    print(f"  Weekly WMAPE: {metrics['wmape']:.2f}%")
    print(f"  Weekly WFA: {metrics['wfa']:.2f}%")
    print(f"  Bias: {metrics['bias_pct']:.2f}%")

    # A-items
    weekly_a = val_copy[val_copy['abc'] == 'A'].groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred_opt': 'sum'
    }).reset_index()
    wmape_a = 100 * np.sum(np.abs(weekly_a['y'] - weekly_a['y_pred_opt'])) / np.sum(weekly_a['y'])
    print(f"  A-items Weekly WFA: {100 - wmape_a:.2f}%")

    return val_copy, metrics, best_thresh


def candidate_3_tweedie(train, val):
    """CANDIDATE 3: Tweedie objective."""
    print("\n" + "="*70)
    print("CANDIDATE 3: TWEEDIE OBJECTIVE")
    print("="*70)

    train_copy = train.copy()
    val_copy = val.copy()

    for col in CAT_FEATURES:
        train_copy[col] = train_copy[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train_copy[FEATURES + CAT_FEATURES]
    X_val = val_copy[FEATURES + CAT_FEATURES]
    y_train = train_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    # Train with tweedie objective
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,  # Between 1 (Poisson) and 2 (Gamma)
        'metric': 'tweedie',
        'num_leaves': 127,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_jobs': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_data, num_boost_round=500)

    y_pred = model.predict(X_val)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    val_copy['y_pred_tweedie'] = y_pred
    val_copy['date'] = pd.to_datetime(val_copy['date'])
    val_copy['week'] = val_copy['date'].dt.isocalendar().week
    val_copy['year'] = val_copy['date'].dt.year

    metrics = weekly_metrics(val_copy, 'y_pred_tweedie')
    print(f"  Weekly WMAPE: {metrics['wmape']:.2f}%")
    print(f"  Weekly WFA: {metrics['wfa']:.2f}%")
    print(f"  Bias: {metrics['bias_pct']:.2f}%")

    # A-items
    weekly_a = val_copy[val_copy['abc'] == 'A'].groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred_tweedie': 'sum'
    }).reset_index()
    wmape_a = 100 * np.sum(np.abs(weekly_a['y'] - weekly_a['y_pred_tweedie'])) / np.sum(weekly_a['y'])
    print(f"  A-items Weekly WFA: {100 - wmape_a:.2f}%")

    return val_copy, metrics, model


def candidate_4_blend_with_baseline(val, train, clf, reg):
    """CANDIDATE 4: Blend ML with rolling mean baseline."""
    print("\n" + "="*70)
    print("CANDIDATE 4: BLEND ML WITH ROLL_MEAN_28 BASELINE")
    print("="*70)

    val_copy = val.copy()
    val_copy['date'] = pd.to_datetime(val_copy['date'])
    val_copy['week'] = val_copy['date'].dt.isocalendar().week
    val_copy['year'] = val_copy['date'].dt.year

    for col in CAT_FEATURES:
        val_copy[col] = val_copy[col].astype('category')

    X_val = val_copy[FEATURES + CAT_FEATURES]
    prob = clf.predict(X_val)
    pred_log = reg.predict(X_val)
    pred_value = np.expm1(pred_log)

    y_pred_ml = np.where(prob > 0.65, pred_value, 0)
    y_pred_ml = np.maximum(0, y_pred_ml)

    # Baseline: roll_mean_28
    y_baseline = val_copy['roll_mean_28'].values

    is_closed = val_copy['is_store_closed'].values

    # Find optimal alpha on weekly WMAPE
    best_wmape = float('inf')
    best_alpha = 0.5

    for alpha in np.arange(0.3, 1.0, 0.05):
        y_blend = alpha * y_pred_ml + (1 - alpha) * y_baseline
        y_blend = np.maximum(0, y_blend)
        y_blend[is_closed == 1] = 0

        val_copy['y_blend_test'] = y_blend
        weekly = val_copy.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
            'y': 'sum', 'y_blend_test': 'sum'
        }).reset_index()

        wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_blend_test'])) / np.sum(weekly['y'])
        if wmape < best_wmape:
            best_wmape = wmape
            best_alpha = alpha

    print(f"  Optimal alpha (ML weight): {best_alpha:.2f}")

    # Apply best blend
    y_blend = best_alpha * y_pred_ml + (1 - best_alpha) * y_baseline
    y_blend = np.maximum(0, y_blend)
    y_blend[is_closed == 1] = 0

    val_copy['y_pred_blend'] = y_blend

    metrics = weekly_metrics(val_copy, 'y_pred_blend')
    print(f"  Weekly WMAPE: {metrics['wmape']:.2f}%")
    print(f"  Weekly WFA: {metrics['wfa']:.2f}%")
    print(f"  Bias: {metrics['bias_pct']:.2f}%")

    # A-items
    weekly_a = val_copy[val_copy['abc'] == 'A'].groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred_blend': 'sum'
    }).reset_index()
    wmape_a = 100 * np.sum(np.abs(weekly_a['y'] - weekly_a['y_pred_blend'])) / np.sum(weekly_a['y'])
    print(f"  A-items Weekly WFA: {100 - wmape_a:.2f}%")

    return val_copy, metrics, best_alpha


def candidate_5_combined_best(train, val):
    """CANDIDATE 5: Tweedie + Bias Calibration + Optimal Threshold."""
    print("\n" + "="*70)
    print("CANDIDATE 5: TWEEDIE + CALIBRATION + BLEND")
    print("="*70)

    train_copy = train.copy()
    val_copy = val.copy()

    for col in CAT_FEATURES:
        train_copy[col] = train_copy[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train_copy[FEATURES + CAT_FEATURES]
    X_val = val_copy[FEATURES + CAT_FEATURES]
    y_train = train_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    # Train Tweedie model
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.3,
        'metric': 'tweedie',
        'num_leaves': 127,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_jobs': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_data, num_boost_round=600)

    # Predict on train for calibration
    y_pred_train = model.predict(X_train)
    y_pred_train = np.maximum(0, y_pred_train)

    # Fit calibration on train
    mask = y_pred_train > 0.1
    if np.sum(y_pred_train[mask]) > 0:
        k = np.sum(y_train[mask]) / np.sum(y_pred_train[mask])
        k = np.clip(k, 0.7, 1.5)
    else:
        k = 1.0

    print(f"  Calibration factor k = {k:.4f}")

    # Predict on val
    y_pred_ml = model.predict(X_val)
    y_pred_ml = np.maximum(0, y_pred_ml) * k

    # Baseline
    y_baseline = val_copy['roll_mean_28'].values

    val_copy['date'] = pd.to_datetime(val_copy['date'])
    val_copy['week'] = val_copy['date'].dt.isocalendar().week
    val_copy['year'] = val_copy['date'].dt.year

    # Find optimal blend
    best_wmape = float('inf')
    best_alpha = 0.7

    for alpha in np.arange(0.5, 1.0, 0.05):
        y_blend = alpha * y_pred_ml + (1 - alpha) * y_baseline
        y_blend = np.maximum(0, y_blend)
        y_blend[is_closed == 1] = 0

        val_copy['y_test'] = y_blend
        weekly = val_copy.groupby(['store_id', 'sku_id', 'year', 'week']).agg({
            'y': 'sum', 'y_test': 'sum'
        }).reset_index()

        wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_test'])) / np.sum(weekly['y'])
        if wmape < best_wmape:
            best_wmape = wmape
            best_alpha = alpha

    print(f"  Optimal alpha: {best_alpha:.2f}")

    # Apply best
    y_final = best_alpha * y_pred_ml + (1 - best_alpha) * y_baseline
    y_final = np.maximum(0, y_final)
    y_final[is_closed == 1] = 0

    val_copy['y_pred_final'] = y_final

    metrics = weekly_metrics(val_copy, 'y_pred_final')
    print(f"  Weekly WMAPE: {metrics['wmape']:.2f}%")
    print(f"  Weekly WFA: {metrics['wfa']:.2f}%")
    print(f"  Bias: {metrics['bias_pct']:.2f}%")

    # A-items
    weekly_a = val_copy[val_copy['abc'] == 'A'].groupby(['store_id', 'sku_id', 'year', 'week']).agg({
        'y': 'sum', 'y_pred_final': 'sum'
    }).reset_index()
    wmape_a = 100 * np.sum(np.abs(weekly_a['y'] - weekly_a['y_pred_final'])) / np.sum(weekly_a['y'])
    print(f"  A-items Weekly WFA: {100 - wmape_a:.2f}%")

    return val_copy, metrics


def main():
    print("="*70)
    print("DISCIPLINED WFA IMPROVEMENT - T1_MATURE")
    print("="*70)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    train = load_data('/tmp/full_data/train')
    val = load_data('/tmp/full_data/val')
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')

    train, val = prepare_data(train, val, sku_attr)
    print(f"Train: {len(train):,}, Val: {len(val):,}")

    # STEP A: Error decomposition
    val_base, train_base, clf, reg = step_a_error_decomposition(val.copy(), train.copy())

    # CANDIDATES
    results = {}

    # Candidate 1: Bias calibration
    _, metrics1, k = candidate_1_bias_calibration(val_base.copy(), train_base.copy(), clf, reg)
    results['1_bias_cal'] = metrics1

    # Candidate 2: Threshold optimization
    _, metrics2, thresh = candidate_2_threshold_tuning(val_base.copy(), train_base.copy(), clf, reg)
    results['2_threshold'] = metrics2

    # Candidate 3: Tweedie
    train_abc, val_abc, _ = assign_abc(train.copy(), val.copy())
    _, metrics3, _ = candidate_3_tweedie(train_abc.copy(), val_abc.copy())
    results['3_tweedie'] = metrics3

    # Candidate 4: Blend
    _, metrics4, alpha = candidate_4_blend_with_baseline(val_base.copy(), train_base.copy(), clf, reg)
    results['4_blend'] = metrics4

    # Candidate 5: Combined
    _, metrics5 = candidate_5_combined_best(train_abc.copy(), val_abc.copy())
    results['5_combined'] = metrics5

    # SUMMARY
    print("\n" + "="*70)
    print("FINAL COMPARISON - T1_MATURE")
    print("="*70)

    print("\n┌────────────────────────────────────────────────────────────────┐")
    print("│  CANDIDATE                  Weekly WMAPE    Weekly WFA    Bias │")
    print("├────────────────────────────────────────────────────────────────┤")

    for name, m in results.items():
        print(f"│  {name:<24}    {m['wmape']:>6.2f}%       {m['wfa']:>6.2f}%    {m['bias_pct']:>6.2f}% │")

    print("└────────────────────────────────────────────────────────────────┘")

    # Find best
    best = min(results.items(), key=lambda x: x[1]['wmape'])
    print(f"\n  WINNER: {best[0]} with Weekly WFA = {best[1]['wfa']:.2f}%")

    # Save
    os.makedirs('/tmp/disciplined', exist_ok=True)
    with open('/tmp/disciplined/t1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
