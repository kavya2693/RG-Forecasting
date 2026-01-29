"""
BASELINE-2: Production-Parity Training with Recursive Validation
================================================================
Fixes from Baseline-1:
1. Per-fold tier tables (no look-ahead bias)
2. Recursive validation (no teacher forcing)
3. Includes spike + velocity features

Features: 39 numeric + 2 categorical
- Original 30 baseline features
- 5 spike features
- 4 velocity features
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
import os
from datetime import datetime, timedelta
import json
import gc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Original baseline features (30)
BASELINE_FEATURES = [
    # Calendar (5)
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    # Cyclical (4)
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    # Closure (4)
    'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
    # Lags (5)
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    # Rolling (5)
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    # Non-zero rates (3)
    'nz_rate_7', 'nz_rate_28', 'roll_mean_pos_28',
    # Dormancy (4)
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
]

# New spike features (5)
SPIKE_FEATURES = [
    'feat_store_spike_pct', 'feat_store_promo_day', 'feat_seasonal_lift',
    'feat_had_recent_spike', 'feat_historical_spike_prob'
]

# New velocity features (4)
VELOCITY_FEATURES = [
    'feat_sale_frequency', 'feat_gap_vs_median', 'feat_is_overdue', 'feat_gap_pressure'
]

# All numeric features for Baseline-2
ALL_FEATURES = BASELINE_FEATURES + SPIKE_FEATURES + VELOCITY_FEATURES

# Categorical features
CAT_FEATURES = ['store_id', 'sku_id']

# Per-segment hyperparameters (production-tuned)
SEGMENT_PARAMS = {
    'A': {'num_leaves': 255, 'min_child_samples': 20, 'learning_rate': 0.03,
          'n_estimators': 300, 'reg_lambda': 0.1, 'threshold': 0.5},
    'B': {'num_leaves': 127, 'min_child_samples': 50, 'learning_rate': 0.03,
          'n_estimators': 200, 'reg_lambda': 0.3, 'threshold': 0.6},
    'C': {'num_leaves': 63, 'min_child_samples': 100, 'learning_rate': 0.05,
          'n_estimators': 150, 'reg_lambda': 0.5, 'threshold': 0.7},
}


def load_data(folder):
    """Load all CSV files from a folder."""
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(folder, '*.parquet')))
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV or Parquet files in {folder}")
        dfs = [pd.read_parquet(f) for f in files]
    else:
        dfs = [pd.read_csv(f) for f in files]

    print(f"  Loaded {len(files)} files from {folder}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(df):,} rows")
    return df


def prepare_features(df, sku_attr=None):
    """Prepare features for model."""
    df = df.copy()
    df['sku_id'] = df['sku_id'].astype(str)
    df['store_id'] = df['store_id'].astype(str)

    # Merge is_local if available
    if sku_attr is not None and 'is_local' not in df.columns:
        sku_attr = sku_attr.copy()
        sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
        if 'local_imported_attribute' in sku_attr.columns:
            sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(
                lambda x: 1 if x in ['L', 'LI'] else 0
            )
        if 'is_local' in sku_attr.columns:
            df = df.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
            df['is_local'] = df['is_local'].fillna(0).astype(int)

    # Fill missing features
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    return df


def assign_abc_segments(train_df, val_df):
    """Assign ABC segments based on training sales volume."""
    # Compute ABC from training data only
    sku_sales = train_df.groupby('sku_id')['y'].sum().sort_values(ascending=False)
    cumsum = sku_sales.cumsum()
    total = sku_sales.sum()

    a_skus = set(sku_sales[cumsum <= total * 0.80].index)
    b_skus = set(sku_sales[(cumsum > total * 0.80) & (cumsum <= total * 0.95)].index)

    train_df['segment'] = train_df['sku_id'].apply(
        lambda x: 'A' if x in a_skus else ('B' if x in b_skus else 'C')
    )
    val_df['segment'] = val_df['sku_id'].apply(
        lambda x: 'A' if x in a_skus else ('B' if x in b_skus else 'C')
    )

    print(f"  ABC segments: A={len(a_skus)}, B={len(b_skus)}, C={len(sku_sales) - len(a_skus) - len(b_skus)}")
    return train_df, val_df


def train_segment_model(train_seg, segment, params):
    """Train classifier + regressor for one segment."""
    if len(train_seg) < 100:
        return None, None

    # Prepare features
    available = [f for f in ALL_FEATURES if f in train_seg.columns]

    for col in CAT_FEATURES:
        if col in train_seg.columns:
            train_seg[col] = train_seg[col].astype('category')

    all_features = available + CAT_FEATURES
    X_train = train_seg[all_features]
    y_train = train_seg['y'].values

    # Stage 1: Binary Classifier
    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))

    # Stage 2: Log-Transform Regressor (non-zero only)
    train_nz = train_seg[train_seg['y'] > 0]
    if len(train_nz) < 10:
        return clf, None

    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'],
        min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(train_nz[all_features], np.log1p(train_nz['y']))

    return clf, reg


def update_features_recursive(row, predictions_history, series_stats):
    """
    Update lag and rolling features using predictions (not actuals).
    This simulates true production behavior.
    """
    series_key = (row['store_id'], row['sku_id'])
    history = predictions_history.get(series_key, [])

    # Update lag features
    row['lag_1'] = history[-1] if len(history) >= 1 else row.get('lag_1', 0)
    row['lag_7'] = history[-7] if len(history) >= 7 else row.get('lag_7', 0)
    row['lag_14'] = history[-14] if len(history) >= 14 else row.get('lag_14', 0)
    row['lag_28'] = history[-28] if len(history) >= 28 else row.get('lag_28', 0)
    row['lag_56'] = history[-56] if len(history) >= 56 else row.get('lag_56', 0)

    # Update rolling features (last 7 and 28 days)
    if len(history) >= 7:
        recent_7 = history[-7:]
        row['roll_mean_7'] = np.mean(recent_7)
        row['roll_sum_7'] = np.sum(recent_7)
        row['nz_rate_7'] = np.mean([1 if x > 0 else 0 for x in recent_7])

    if len(history) >= 28:
        recent_28 = history[-28:]
        row['roll_mean_28'] = np.mean(recent_28)
        row['roll_sum_28'] = np.sum(recent_28)
        row['roll_std_28'] = np.std(recent_28)
        row['nz_rate_28'] = np.mean([1 if x > 0 else 0 for x in recent_28])
        pos_vals = [x for x in recent_28 if x > 0]
        row['roll_mean_pos_28'] = np.mean(pos_vals) if pos_vals else 0

    # Update dormancy features
    last_sale_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i] > 0:
            last_sale_idx = len(history) - 1 - i
            row['last_sale_qty_asof'] = history[i]
            break

    if last_sale_idx is not None:
        row['days_since_last_sale_asof'] = last_sale_idx + 1
        row['dormancy_capped'] = min(last_sale_idx + 1, 365)
        row['zero_run_length_asof'] = last_sale_idx
    else:
        # No sale in prediction history, use original value + horizon
        base_days = series_stats.get(series_key, {}).get('days_since_last_sale', 0)
        row['days_since_last_sale_asof'] = base_days + len(history)
        row['dormancy_capped'] = min(base_days + len(history), 365)
        row['zero_run_length_asof'] = len(history)

    # Update velocity features
    if len(history) >= 7:
        median_gap = series_stats.get(series_key, {}).get('median_gap', 7)
        row['feat_gap_vs_median'] = row['days_since_last_sale_asof'] / max(median_gap, 1)
        row['feat_is_overdue'] = 1 if row['days_since_last_sale_asof'] > 1.5 * median_gap else 0
        row['feat_gap_pressure'] = 1 - np.exp(-0.5 * row['days_since_last_sale_asof'] / max(median_gap, 1))

    return row


def recursive_predict_segment(clf, reg, val_seg, segment, params, series_stats):
    """
    Predict recursively, updating features day-by-day.
    This fixes the teacher forcing issue.
    """
    threshold = params['threshold']
    available = [f for f in ALL_FEATURES if f in val_seg.columns]
    all_features = available + CAT_FEATURES

    # Sort by series and date
    val_seg = val_seg.sort_values(['store_id', 'sku_id', 'date']).copy()

    # Initialize predictions history per series
    predictions_history = {}
    predictions = []

    # Group by series for efficient processing
    for (store_id, sku_id), group in val_seg.groupby(['store_id', 'sku_id']):
        series_key = (store_id, sku_id)
        predictions_history[series_key] = []

        for idx, row in group.iterrows():
            # Update features using prediction history
            if len(predictions_history[series_key]) > 0:
                row = update_features_recursive(
                    row.copy(), predictions_history, series_stats
                )

            # Prepare input for prediction
            X = pd.DataFrame([row[all_features]])
            for col in CAT_FEATURES:
                X[col] = X[col].astype('category')

            # Predict
            prob = clf.predict_proba(X)[:, 1][0]
            if reg is not None and prob > threshold:
                pred_log = reg.predict(X)[0]
                pred = max(0, np.expm1(pred_log))
            else:
                pred = 0

            # Override closed stores
            if row.get('is_store_closed', 0) == 1:
                pred = 0

            predictions.append({'idx': idx, 'y_pred': pred})
            predictions_history[series_key].append(pred)

    # Map predictions back
    pred_df = pd.DataFrame(predictions).set_index('idx')
    val_seg['y_pred'] = pred_df['y_pred']

    return val_seg['y_pred'].values


def compute_series_stats(train_df):
    """Compute per-series statistics needed for recursive updates."""
    stats = {}

    for (store_id, sku_id), group in train_df.groupby(['store_id', 'sku_id']):
        series_key = (store_id, sku_id)

        # Days since last sale (at end of training)
        last_sale_date = group[group['y'] > 0]['date'].max() if (group['y'] > 0).any() else None
        last_date = group['date'].max()
        if last_sale_date is not None:
            days_since = (pd.to_datetime(last_date) - pd.to_datetime(last_sale_date)).days
        else:
            days_since = len(group)

        # Median gap between sales
        sale_dates = group[group['y'] > 0]['date'].sort_values()
        if len(sale_dates) > 1:
            gaps = sale_dates.diff().dt.days.dropna()
            median_gap = gaps.median() if len(gaps) > 0 else 7
        else:
            median_gap = 7

        stats[series_key] = {
            'days_since_last_sale': days_since,
            'median_gap': median_gap,
            'sale_frequency': (group['y'] > 0).mean()
        }

    return stats


def compute_metrics(y_true, y_pred, dates=None, store_ids=None):
    """Compute comprehensive metrics."""
    # Daily metrics
    wmape_daily = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    wfa_daily = 100 - wmape_daily

    # Bias
    bias_ratio = np.sum(y_pred) / max(np.sum(y_true), 1)

    # Zero classification
    actual_sale = (y_true > 0).astype(int)
    pred_sale = (y_pred > 0).astype(int)

    tp = np.sum((pred_sale == 1) & (actual_sale == 1))
    fp = np.sum((pred_sale == 1) & (actual_sale == 0))
    fn = np.sum((pred_sale == 0) & (actual_sale == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    metrics = {
        'daily_wmape': wmape_daily,
        'daily_wfa': wfa_daily,
        'bias_ratio': bias_ratio,
        'sale_precision': precision,
        'sale_recall': recall,
        'sale_f1': f1,
    }

    # Weekly metrics if dates and store_ids provided
    if dates is not None and store_ids is not None:
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'store_id': store_ids,
            'y_true': y_true,
            'y_pred': y_pred
        })
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year

        # Weekly store
        weekly_store = df.groupby(['store_id', 'year', 'week']).agg({
            'y_true': 'sum', 'y_pred': 'sum'
        }).reset_index()
        wmape_ws = 100 * np.sum(np.abs(weekly_store['y_true'] - weekly_store['y_pred'])) / max(np.sum(weekly_store['y_true']), 1)
        metrics['weekly_store_wfa'] = 100 - wmape_ws

        # Weekly total
        weekly_total = df.groupby(['year', 'week']).agg({
            'y_true': 'sum', 'y_pred': 'sum'
        }).reset_index()
        wmape_wt = 100 * np.sum(np.abs(weekly_total['y_true'] - weekly_total['y_pred'])) / max(np.sum(weekly_total['y_true']), 1)
        metrics['weekly_total_wfa'] = 100 - wmape_wt

    return metrics


def train_and_evaluate_fold(train_df, val_df, fold_name, use_recursive=True):
    """Train and evaluate one fold with optional recursive validation."""
    print(f"\n{'='*70}")
    print(f"FOLD: {fold_name} (Recursive={use_recursive})")
    print("="*70)

    # Assign ABC segments
    train_df, val_df = assign_abc_segments(train_df.copy(), val_df.copy())

    # Compute series stats for recursive updates
    series_stats = compute_series_stats(train_df) if use_recursive else {}

    all_preds = []
    all_y_true = []
    all_dates = []
    all_stores = []

    for seg in ['A', 'B', 'C']:
        train_seg = train_df[train_df['segment'] == seg].copy()
        val_seg = val_df[val_df['segment'] == seg].copy()

        if len(train_seg) < 100 or len(val_seg) < 100:
            print(f"  {seg}-segment: skipped (too few samples)")
            continue

        print(f"\n  Training {seg}-segment...")
        print(f"    Train: {len(train_seg):,}, Val: {len(val_seg):,}")

        # Prepare features
        for col in CAT_FEATURES:
            train_seg[col] = train_seg[col].astype('category')
            val_seg[col] = val_seg[col].astype('category')

        # Train models
        clf, reg = train_segment_model(train_seg, seg, SEGMENT_PARAMS[seg])

        if clf is None:
            print(f"    {seg}-segment: training failed")
            continue

        # Predict
        if use_recursive:
            # RECURSIVE: Update features day-by-day using predictions
            y_pred = recursive_predict_segment(
                clf, reg, val_seg, seg, SEGMENT_PARAMS[seg], series_stats
            )
        else:
            # TEACHER FORCING: Use pre-computed features (baseline-1 behavior)
            available = [f for f in ALL_FEATURES if f in val_seg.columns]
            all_features = available + CAT_FEATURES
            X_val = val_seg[all_features]

            prob = clf.predict_proba(X_val)[:, 1]
            if reg is not None:
                pred_value = np.expm1(reg.predict(X_val))
            else:
                pred_value = np.zeros(len(X_val))

            threshold = SEGMENT_PARAMS[seg]['threshold']
            y_pred = np.where(prob > threshold, pred_value, 0)
            y_pred = np.maximum(0, y_pred)
            y_pred[val_seg['is_store_closed'].values == 1] = 0

        # Segment metrics
        seg_metrics = compute_metrics(
            val_seg['y'].values, y_pred,
            val_seg['date'].values, val_seg['store_id'].values
        )
        print(f"    {seg}: Daily WFA={seg_metrics['daily_wfa']:.2f}%, "
              f"Weekly Store WFA={seg_metrics.get('weekly_store_wfa', 0):.2f}%, "
              f"Bias={seg_metrics['bias_ratio']:.3f}")

        all_preds.extend(y_pred)
        all_y_true.extend(val_seg['y'].values)
        all_dates.extend(val_seg['date'].values)
        all_stores.extend(val_seg['store_id'].values)

        del clf, reg
        gc.collect()

    # Overall metrics
    if len(all_preds) > 0:
        metrics = compute_metrics(
            np.array(all_y_true), np.array(all_preds),
            all_dates, all_stores
        )
        metrics['n_val'] = len(all_preds)
        print(f"\n  FOLD {fold_name} OVERALL:")
        print(f"    Daily WFA:        {metrics['daily_wfa']:.2f}%")
        print(f"    Weekly Store WFA: {metrics.get('weekly_store_wfa', 0):.2f}%")
        print(f"    Weekly Total WFA: {metrics.get('weekly_total_wfa', 0):.2f}%")
        print(f"    Bias Ratio:       {metrics['bias_ratio']:.3f}")
        print(f"    Sale F1:          {metrics['sale_f1']:.3f}")
        return metrics

    return None


def main():
    print("=" * 80)
    print("BASELINE-2: PRODUCTION-PARITY TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print(f"\nFeatures: {len(ALL_FEATURES)} numeric + {len(CAT_FEATURES)} categorical")
    print(f"  Baseline: {len(BASELINE_FEATURES)}")
    print(f"  Spike: {len(SPIKE_FEATURES)}")
    print(f"  Velocity: {len(VELOCITY_FEATURES)}")
    print("=" * 80)

    # Load SKU attributes
    sku_attr_path = '/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv'
    if os.path.exists(sku_attr_path):
        sku_attr = pd.read_csv(sku_attr_path)
    else:
        sku_attr = None

    # Define folds
    folds = [
        {
            'name': 'F1',
            'train_folder': '/tmp/baseline2/f1_train',
            'val_folder': '/tmp/baseline2/f1_val',
            'train_dates': ('2019-01-02', '2025-06-26'),
            'val_dates': ('2025-07-03', '2025-12-17'),
        },
    ]

    all_results = {
        'baseline2_recursive': {},
        'baseline1_teacher_forcing': {},
    }

    for fold in folds:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING FOLD: {fold['name']}")
        print('#' * 80)

        # Check if data exists
        if not os.path.exists(fold['train_folder']):
            print(f"  Data not found at {fold['train_folder']}")
            print(f"  Please export data first using:")
            print(f"  bq extract --destination_format=CSV \\")
            print(f"    'myforecastingsales.forecasting.v_trainval_baseline2_{fold[\"name\"].lower()}' \\")
            print(f"    'gs://myforecastingsales-data/baseline2/{fold[\"name\"].lower()}_*.csv'")
            continue

        # Load data
        print("\n  Loading training data...")
        train_df = load_data(fold['train_folder'])
        train_df = prepare_features(train_df, sku_attr)

        print("  Loading validation data...")
        val_df = load_data(fold['val_folder'])
        val_df = prepare_features(val_df, sku_attr)

        # Run with recursive validation (Baseline-2)
        print("\n" + "="*70)
        print("BASELINE-2: RECURSIVE VALIDATION (Production Parity)")
        print("="*70)
        metrics_recursive = train_and_evaluate_fold(
            train_df.copy(), val_df.copy(), fold['name'], use_recursive=True
        )
        if metrics_recursive:
            all_results['baseline2_recursive'][fold['name']] = metrics_recursive

        # Run with teacher forcing (Baseline-1 behavior) for comparison
        print("\n" + "="*70)
        print("BASELINE-1 (COMPARISON): TEACHER FORCING")
        print("="*70)
        metrics_teacher = train_and_evaluate_fold(
            train_df.copy(), val_df.copy(), fold['name'], use_recursive=False
        )
        if metrics_teacher:
            all_results['baseline1_teacher_forcing'][fold['name']] = metrics_teacher

        del train_df, val_df
        gc.collect()

    # Final comparison
    print("\n" + "=" * 80)
    print("BASELINE-1 vs BASELINE-2 COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Baseline-1 (Teacher)':<20} {'Baseline-2 (Recursive)':<20} {'Delta':<10}")
    print("-" * 80)

    for fold_name in all_results['baseline2_recursive']:
        b1 = all_results['baseline1_teacher_forcing'].get(fold_name, {})
        b2 = all_results['baseline2_recursive'].get(fold_name, {})

        for metric in ['daily_wfa', 'weekly_store_wfa', 'bias_ratio', 'sale_f1']:
            v1 = b1.get(metric, 0)
            v2 = b2.get(metric, 0)
            delta = v2 - v1

            if metric in ['daily_wfa', 'weekly_store_wfa']:
                print(f"{fold_name} {metric:<20} {v1:>18.2f}% {v2:>18.2f}% {delta:>+9.2f}pp")
            else:
                print(f"{fold_name} {metric:<20} {v1:>19.3f} {v2:>19.3f} {delta:>+10.3f}")

    # Save results
    output = {
        'baseline1_metrics': all_results['baseline1_teacher_forcing'],
        'baseline2_metrics': all_results['baseline2_recursive'],
        'features': {
            'baseline': BASELINE_FEATURES,
            'spike': SPIKE_FEATURES,
            'velocity': VELOCITY_FEATURES,
            'total_count': len(ALL_FEATURES)
        },
        'timestamp': datetime.now().isoformat(),
        'description': 'Baseline-2 with per-fold tiers, recursive validation, spike+velocity features'
    }

    os.makedirs('/tmp/baseline2', exist_ok=True)
    with open('/tmp/baseline2/comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to /tmp/baseline2/comparison_results.json")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
