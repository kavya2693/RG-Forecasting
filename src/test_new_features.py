"""
TEST NEW FEATURES
=================
Compare baseline model (original features) vs enhanced model (with new calculated features).
New features include calendar signals, store-level aggregates, SKU-level aggregates,
and store-SKU interaction features.
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
import os
import json
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
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

NEW_FEATURE_NAMES = [
    # Calendar-based
    'is_december', 'is_ramadan_approx', 'is_week_49_52',
    # Store-level pre-computed
    'store_avg_daily', 'store_zero_rate', 'store_weekend_ratio',
    # SKU-level pre-computed
    'sku_avg_daily', 'sku_nz_rate_global', 'sku_trend',
    # Store-SKU interaction pre-computed
    'store_sku_avg', 'store_sku_nz_rate',
]

FEATURES_NEW = FEATURES + NEW_FEATURE_NAMES

# Approximate Ramadan months by year (Middle Eastern retail dataset)
# Stored as set of (year, month) tuples for fast vectorized lookup
RAMADAN_SET = {
    (2019, 5), (2020, 4), (2021, 4), (2022, 4),
    (2023, 3), (2024, 3), (2025, 3),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(folder, max_files=10):
    """Load CSVs from folder (limited to max_files for memory efficiency)."""
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))[:max_files]
    log(f"  Loading {len(files)} files from {folder}...")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    log(f"    -> {len(df):,} rows loaded")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_calendar_features(df):
    """Add calendar-based features that don't require training data. Fully vectorized."""
    # is_december
    df['is_december'] = (df['month'] == 12).astype(np.int8)

    # is_ramadan_approx - vectorized via year*100+month key
    ramadan_keys = set(y * 100 + m for y, m in RAMADAN_SET)
    ym_key = df['year'].values * 100 + df['month'].values
    df['is_ramadan_approx'] = np.isin(ym_key, list(ramadan_keys)).astype(np.int8)

    # is_week_49_52
    df['is_week_49_52'] = (df['week_of_year'] >= 49).astype(np.int8)

    return df


def compute_store_features(train):
    """Compute store-level features from training data only."""
    log("  Computing store-level features...")

    # store_avg_daily
    store_avg = train.groupby('store_id')['y'].mean().rename('store_avg_daily')

    # store_zero_rate
    store_zero = train.groupby('store_id')['y'].apply(lambda s: (s == 0).mean()).rename('store_zero_rate')

    # store_weekend_ratio
    is_wknd = train['is_weekend'].astype(bool)
    weekend_avg = train.loc[is_wknd].groupby('store_id')['y'].mean().rename('weekend_avg')
    weekday_avg = train.loc[~is_wknd].groupby('store_id')['y'].mean().rename('weekday_avg')
    wknd_ratio = (weekend_avg / (weekday_avg + 0.01)).rename('store_weekend_ratio')

    store_feats = pd.concat([store_avg, store_zero, wknd_ratio], axis=1).reset_index()
    return store_feats


def compute_sku_features(train):
    """Compute SKU-level features from training data only."""
    log("  Computing SKU-level features...")

    # sku_avg_daily
    sku_avg = train.groupby('sku_id')['y'].mean().rename('sku_avg_daily')

    # sku_nz_rate_global
    sku_nz = train.groupby('sku_id')['y'].apply(lambda s: (s > 0).mean()).rename('sku_nz_rate_global')

    # sku_trend: ratio of second-half avg to first-half avg
    log("    Computing SKU trend (may take a moment)...")
    dates = pd.to_datetime(train['date'])
    date_num = dates.astype(np.int64)  # nanoseconds since epoch

    # Per-SKU min/max dates
    grp_dates = pd.DataFrame({'sku_id': train['sku_id'], 'date_num': date_num})
    sku_date_stats = grp_dates.groupby('sku_id')['date_num'].agg(['min', 'max'])
    sku_date_stats['midpoint'] = (sku_date_stats['min'] + sku_date_stats['max']) / 2

    # Map midpoint back to each row
    midpoint_map = sku_date_stats['midpoint']
    row_midpoint = train['sku_id'].map(midpoint_map).values
    is_second_half = date_num.values >= row_midpoint

    train_tmp = pd.DataFrame({
        'sku_id': train['sku_id'].values,
        'y': train['y'].values,
        'is_second_half': is_second_half
    })

    first_half_avg = train_tmp.loc[~train_tmp['is_second_half']].groupby('sku_id')['y'].mean().rename('first_half_avg')
    second_half_avg = train_tmp.loc[train_tmp['is_second_half']].groupby('sku_id')['y'].mean().rename('second_half_avg')

    sku_trend = (second_half_avg / (first_half_avg + 0.01)).rename('sku_trend')

    sku_feats = pd.concat([sku_avg, sku_nz, sku_trend], axis=1).reset_index()
    return sku_feats


def compute_store_sku_features(train):
    """Compute store-SKU interaction features from training data only."""
    log("  Computing store-SKU interaction features...")

    ss_avg = train.groupby(['store_id', 'sku_id'])['y'].mean().rename('store_sku_avg')
    ss_nz = train.groupby(['store_id', 'sku_id'])['y'].apply(lambda s: (s > 0).mean()).rename('store_sku_nz_rate')

    ss_feats = pd.concat([ss_avg, ss_nz], axis=1).reset_index()
    return ss_feats


def add_precomputed_features(train, val):
    """
    Compute features from training data and merge to both train and val.
    This avoids data leakage - val features are computed from train only.
    """
    store_feats = compute_store_features(train)
    sku_feats = compute_sku_features(train)
    ss_feats = compute_store_sku_features(train)

    log("  Merging pre-computed features to train and val...")

    # Merge to train
    train = train.merge(store_feats, on='store_id', how='left')
    train = train.merge(sku_feats, on='sku_id', how='left')
    train = train.merge(ss_feats, on=['store_id', 'sku_id'], how='left')

    # Merge to val
    val = val.merge(store_feats, on='store_id', how='left')
    val = val.merge(sku_feats, on='sku_id', how='left')
    val = val.merge(ss_feats, on=['store_id', 'sku_id'], how='left')

    # Fill NaN in new features with 0
    for col in NEW_FEATURE_NAMES:
        for df in [train, val]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(np.float32)

    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# ABC SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def abc_segment(train, val):
    """Assign ABC segments based on training data sales."""
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

    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def train_and_predict(train, val, feature_list, cat_features, model_label):
    """
    Train two-stage ABC-segmented model and predict on validation set.
    Returns val DataFrame with y_pred column and feature importance dict.
    """
    log(f"\n  Training [{model_label}] with {len(feature_list)} features + {len(cat_features)} cat features")

    # Ensure feature columns are filled
    for df in [train, val]:
        for col in feature_list:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    val = val.copy()
    val['y_pred'] = 0.0

    feat_importance = {}

    for seg in ['A', 'B', 'C']:
        t_seg = time.time()
        train_seg = train[train['abc'] == seg].copy()
        val_seg = val[val['abc'] == seg].copy()

        if len(train_seg) == 0 or len(val_seg) == 0:
            log(f"    {seg}-items: SKIPPED (no data)")
            continue

        train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)

        for col in cat_features:
            train_seg[col] = train_seg[col].astype('category')
            val_seg[col] = val_seg[col].astype('category')

        X_train = train_seg[feature_list + cat_features]
        X_val = val_seg[feature_list + cat_features]

        # Segment-specific hyperparameters
        if seg == 'A':
            nl, lr, rnds_clf, rnds_reg, mdl = 255, 0.015, 800, 1000, 10
        elif seg == 'B':
            nl, lr, rnds_clf, rnds_reg, mdl = 63, 0.03, 300, 400, 50
        else:
            nl, lr, rnds_clf, rnds_reg, mdl = 31, 0.05, 200, 300, 100

        # STAGE 1: Binary classifier
        log(f"    {seg}-items: training classifier ({len(train_seg):,} rows)...")
        clf_data = lgb.Dataset(X_train, label=train_seg['y_binary'],
                               categorical_feature=cat_features)
        clf = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': nl,
             'learning_rate': lr, 'feature_fraction': 0.8, 'min_data_in_leaf': mdl,
             'verbose': -1, 'n_jobs': -1},
            clf_data, num_boost_round=rnds_clf
        )
        prob = clf.predict(X_val)

        # STAGE 2: Log-transform regressor (non-zero rows only)
        train_nz = train_seg[train_seg['y'] > 0]
        X_train_nz = train_nz[feature_list + cat_features]
        y_train_nz = np.log1p(train_nz['y'].values)

        log(f"    {seg}-items: training regressor ({len(train_nz):,} non-zero rows)...")
        reg_data = lgb.Dataset(X_train_nz, label=y_train_nz,
                               categorical_feature=cat_features)
        reg = lgb.train(
            {'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': nl,
             'learning_rate': lr, 'feature_fraction': 0.8,
             'min_data_in_leaf': max(5, mdl // 2),
             'lambda_l2': 0.5, 'verbose': -1, 'n_jobs': -1},
            reg_data, num_boost_round=rnds_reg
        )
        pred_value = np.expm1(reg.predict(X_val))

        # Threshold
        thresh = 0.6 if seg in ['A', 'B'] else 0.7
        y_pred = np.where(prob > thresh, pred_value, 0)
        y_pred = np.maximum(0, y_pred)
        y_pred[val_seg['is_store_closed'].values == 1] = 0

        # Calibration for A-items
        if seg == 'A':
            prob_tr = clf.predict(X_train)
            pred_val_tr = np.expm1(reg.predict(X_train))
            y_pred_tr = np.where(prob_tr > thresh, pred_val_tr, 0)
            y_pred_tr = np.maximum(0, y_pred_tr)
            mask = y_pred_tr > 0.1
            if np.sum(y_pred_tr[mask]) > 0:
                k = np.clip(np.sum(train_seg['y'].values[mask]) / np.sum(y_pred_tr[mask]), 0.8, 1.3)
                y_pred = y_pred * k
                y_pred[val_seg['is_store_closed'].values == 1] = 0
                log(f"    {seg}-items: {len(val_seg):,} val rows, k={k:.4f}, took {time.time()-t_seg:.1f}s")
            else:
                log(f"    {seg}-items: {len(val_seg):,} val rows, k=N/A, took {time.time()-t_seg:.1f}s")

            # Capture feature importance from A-segment regressor
            importance = reg.feature_importance(importance_type='gain')
            feat_names = reg.feature_name()
            for fn, fi in zip(feat_names, importance):
                feat_importance[fn] = float(fi)
        else:
            log(f"    {seg}-items: {len(val_seg):,} val rows, took {time.time()-t_seg:.1f}s")

        val.loc[val['abc'] == seg, 'y_pred'] = y_pred

    # Prepare date columns for aggregation
    val['date'] = pd.to_datetime(val['date'])
    val['week'] = val['date'].dt.isocalendar().week.astype(int)
    val['year_val'] = val['date'].dt.year

    return val, feat_importance


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(val):
    """Compute WMAPE and WFA at four aggregation levels."""
    results = {}

    # 1. Daily SKU-Store
    wmape = 100 * np.sum(np.abs(val['y'] - val['y_pred'])) / max(np.sum(val['y']), 1)
    results['daily_sku_store'] = {'wmape': round(float(wmape), 2), 'wfa': round(float(100 - wmape), 2)}

    # 2. Weekly SKU-Store
    weekly = val.groupby(['store_id', 'sku_id', 'year_val', 'week']).agg(
        {'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / max(np.sum(weekly['y']), 1)
    results['weekly_sku_store'] = {'wmape': round(float(wmape), 2), 'wfa': round(float(100 - wmape), 2)}

    # 3. Weekly Store
    weekly_store = val.groupby(['store_id', 'year_val', 'week']).agg(
        {'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_store['y'] - weekly_store['y_pred'])) / max(np.sum(weekly_store['y']), 1)
    results['weekly_store'] = {'wmape': round(float(wmape), 2), 'wfa': round(float(100 - wmape), 2)}

    # 4. Weekly Total
    weekly_total = val.groupby(['year_val', 'week']).agg(
        {'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_total['y'] - weekly_total['y_pred'])) / max(np.sum(weekly_total['y']), 1)
    results['weekly_total'] = {'wmape': round(float(wmape), 2), 'wfa': round(float(100 - wmape), 2)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    log("=" * 80)
    log("  NEW FEATURES TEST: BASELINE vs ENHANCED")
    log("  " + "=" * 76)
    log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    # ── Load data ────────────────────────────────────────────────────────
    log("\n[1/6] LOADING DATA")
    log("-" * 40)
    train_raw = load_data('/tmp/full_data/train')
    val_raw = load_data('/tmp/full_data/val')

    log(f"\n  Loading SKU attributes...")
    sku_attr = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv')
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(lambda x: 1 if x in ['L', 'LI'] else 0)
    log(f"    -> {len(sku_attr):,} SKUs loaded")

    # ── Prepare base data ────────────────────────────────────────────────
    log("\n[2/6] PREPARING BASE DATA")
    log("-" * 40)

    for df in [train_raw, val_raw]:
        df['sku_id'] = df['sku_id'].astype(str)
        df['store_id'] = df['store_id'].astype(str)

    train_raw = train_raw.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    val_raw = val_raw.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')

    for df in [train_raw, val_raw]:
        df['is_local'] = df['is_local'].fillna(0).astype(int)

    log(f"  Base data prepared. Train: {len(train_raw):,}, Val: {len(val_raw):,}")

    # ── Add calendar features (no leakage, purely date-based) ──────────
    log("\n[3/6] COMPUTING NEW FEATURES")
    log("-" * 40)
    log("  Adding calendar features (vectorized)...")
    train_enhanced = add_calendar_features(train_raw.copy())
    val_enhanced = add_calendar_features(val_raw.copy())
    log("    Calendar features done.")

    # ── Add pre-computed features (from train only) ──────────────────────
    train_enhanced, val_enhanced = add_precomputed_features(train_enhanced, val_enhanced)

    # Verify new features exist
    for feat in NEW_FEATURE_NAMES:
        assert feat in train_enhanced.columns, f"Missing feature in train: {feat}"
        assert feat in val_enhanced.columns, f"Missing feature in val: {feat}"
    log(f"  All {len(NEW_FEATURE_NAMES)} new features verified in both train and val.")

    # Print feature stats
    log("\n  New feature stats (train sample):")
    for feat in NEW_FEATURE_NAMES:
        mn = train_enhanced[feat].mean()
        std = train_enhanced[feat].std()
        nz = (train_enhanced[feat] != 0).mean() * 100
        log(f"    {feat:<25} mean={mn:.4f}  std={std:.4f}  non-zero={nz:.1f}%")

    # ── ABC segmentation ────────────────────────────────────────────────
    log("\n[4/6] ABC SEGMENTATION")
    log("-" * 40)

    # For baseline, use copies of the raw data (which already has is_local merged)
    train_base = train_raw.copy()
    val_base = val_raw.copy()
    train_base, val_base = abc_segment(train_base, val_base)

    # For enhanced, use the enhanced data
    train_enhanced, val_enhanced = abc_segment(train_enhanced, val_enhanced)

    for seg in ['A', 'B', 'C']:
        n_train = (train_base['abc'] == seg).sum()
        n_val = (val_base['abc'] == seg).sum()
        log(f"  {seg}: train={n_train:,}  val={n_val:,}")

    # ── Train & Predict: BASELINE ────────────────────────────────────────
    log("\n[5/6] TRAINING MODELS")
    log("-" * 40)

    t_baseline_start = time.time()
    val_baseline, _ = train_and_predict(
        train_base, val_base, FEATURES, CAT_FEATURES, "BASELINE"
    )
    t_baseline = time.time() - t_baseline_start
    log(f"  >>> Baseline total: {t_baseline:.1f}s")

    # ── Train & Predict: ENHANCED ────────────────────────────────────────
    t_enhanced_start = time.time()
    val_enhanced_pred, feat_importance = train_and_predict(
        train_enhanced, val_enhanced, FEATURES_NEW, CAT_FEATURES, "ENHANCED"
    )
    t_enhanced = time.time() - t_enhanced_start
    log(f"  >>> Enhanced total: {t_enhanced:.1f}s")

    # ── Compute Metrics ──────────────────────────────────────────────────
    log("\n[6/6] COMPUTING METRICS")
    log("-" * 40)

    baseline_results = compute_metrics(val_baseline)
    enhanced_results = compute_metrics(val_enhanced_pred)

    # ── Print Comparison ─────────────────────────────────────────────────
    log("\n")
    log("=" * 90)
    log("  COMPARISON: BASELINE vs ENHANCED")
    log("=" * 90)

    header = f"  {'Level':<25} {'Base WMAPE':>12} {'Enh WMAPE':>12} {'Delta':>9} {'Base WFA':>11} {'Enh WFA':>11} {'Delta':>9}"
    log(header)
    log("  " + "-" * 88)

    levels = ['daily_sku_store', 'weekly_sku_store', 'weekly_store', 'weekly_total']
    level_labels = ['Daily SKU-Store', 'Weekly SKU-Store', 'Weekly Store', 'Weekly Total']

    for level, label in zip(levels, level_labels):
        bw = baseline_results[level]['wmape']
        ew = enhanced_results[level]['wmape']
        dw = ew - bw
        bf = baseline_results[level]['wfa']
        ef = enhanced_results[level]['wfa']
        df_val = ef - bf
        sign_w = "+" if dw > 0 else ""
        sign_f = "+" if df_val > 0 else ""
        log(f"  {label:<25} {bw:>11.2f}% {ew:>11.2f}% {sign_w}{dw:>8.2f}% {bf:>10.2f}% {ef:>10.2f}% {sign_f}{df_val:>8.2f}%")

    # ── Feature Importance ───────────────────────────────────────────────
    log("\n")
    log("=" * 60)
    log("  TOP 30 FEATURES (A-segment regressor, enhanced model)")
    log("=" * 60)

    sorted_imp = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:30]
    for i, (fn, fi) in enumerate(sorted_imp, 1):
        marker = " <-- NEW" if fn in NEW_FEATURE_NAMES else ""
        log(f"  {i:>2}. {fn:<30} {fi:>12.1f}{marker}")

    # Highlight new features specifically
    log("\n  New features ranking:")
    all_sorted = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (fn, fi) in enumerate(all_sorted, 1):
        if fn in NEW_FEATURE_NAMES:
            log(f"    #{rank}: {fn} = {fi:.1f}")

    # ── Save Results ─────────────────────────────────────────────────────
    os.makedirs('/tmp/new_features_test', exist_ok=True)

    # Top 30 feature importance as dict
    feat_imp_top30 = {fn: round(fi, 2) for fn, fi in sorted_imp}

    output = {
        "baseline": baseline_results,
        "enhanced": enhanced_results,
        "new_features_added": NEW_FEATURE_NAMES,
        "feature_importance_new": feat_imp_top30,
        "timing": {
            "baseline_seconds": round(t_baseline, 1),
            "enhanced_seconds": round(t_enhanced, 1),
            "total_seconds": round(time.time() - t_start, 1),
        }
    }

    with open('/tmp/new_features_test/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    log(f"\n  Results saved to /tmp/new_features_test/results.json")

    t_total = time.time() - t_start
    log(f"\n  Total runtime: {t_total:.1f}s ({t_total/60:.1f} minutes)")
    log(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)


if __name__ == "__main__":
    main()
