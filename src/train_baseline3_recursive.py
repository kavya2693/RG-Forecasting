"""
BASELINE-3: Production-Parity Recursive Validation
===================================================
Implements true production parity with:
1. Vectorized ring buffer for lag/roll features
2. Recursive rollout (predictions feed into features)
3. Expected value: yhat = p * mu (no hard threshold)
4. Smearing correction for log-transform bias

No future actuals inside validation window.
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
# CONFIGURATION
# =============================================================================
L = 56  # Ring buffer length (max lag = 56, can extend to 365 for annual)
EPS = 0.5  # Threshold for "sale occurred" in predictions

SEGMENT_PARAMS = {
    'A': {'num_leaves': 255, 'min_child_samples': 20, 'learning_rate': 0.03,
          'n_estimators': 300, 'reg_lambda': 0.1},
    'B': {'num_leaves': 127, 'min_child_samples': 50, 'learning_rate': 0.03,
          'n_estimators': 200, 'reg_lambda': 0.3},
    'C': {'num_leaves': 63, 'min_child_samples': 100, 'learning_rate': 0.05,
          'n_estimators': 150, 'reg_lambda': 0.5},
}

# Features that need recursive update
LAG_OFFSETS = [1, 7, 14, 28, 56]
ROLL_WINDOWS = [7, 28]

# Static features (don't change during rollout)
STATIC_FEATURES = [
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'is_store_closed',
]

# Dynamic features (computed from ring buffer)
DYNAMIC_FEATURES = [
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_28',
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof',
]

ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES
CAT_FEATURES = ['store_id', 'sku_id']


class RingBufferState:
    """
    Vectorized ring buffer for efficient recursive feature computation.
    Maintains per-series state without Python loops.
    """

    def __init__(self, n_series: int, L: int = 56):
        self.n = n_series
        self.L = L
        self.pos = 0  # Current position in ring buffer

        # Ring buffer: sales history (actual up to cutoff, then predictions)
        self.hist = np.zeros((n_series, L), dtype=np.float32)

        # Rolling sums (for efficient mean/std computation)
        self.sum_7 = np.zeros(n_series, dtype=np.float32)
        self.sum_28 = np.zeros(n_series, dtype=np.float32)
        self.sumsq_28 = np.zeros(n_series, dtype=np.float32)

        # Non-zero counts
        self.nz_7 = np.zeros(n_series, dtype=np.int16)
        self.nz_28 = np.zeros(n_series, dtype=np.int16)

        # Recency state
        self.days_since_last_sale = np.zeros(n_series, dtype=np.int16)
        self.zero_run_length = np.zeros(n_series, dtype=np.int16)
        self.last_sale_qty = np.zeros(n_series, dtype=np.float32)

    def initialize_from_history(self, sales_history: np.ndarray):
        """
        Initialize ring buffer from last L days of actual sales at cutoff.
        sales_history: shape (n_series, L)
        """
        assert sales_history.shape == (self.n, self.L)
        self.hist = sales_history.astype(np.float32)
        self.pos = 0

        # Compute initial rolling sums
        self.sum_7 = self.hist[:, -7:].sum(axis=1)
        self.sum_28 = self.hist[:, -28:].sum(axis=1)
        self.sumsq_28 = (self.hist[:, -28:] ** 2).sum(axis=1)

        # Compute initial non-zero counts
        self.nz_7 = (self.hist[:, -7:] > EPS).sum(axis=1).astype(np.int16)
        self.nz_28 = (self.hist[:, -28:] > EPS).sum(axis=1).astype(np.int16)

        # Compute initial recency state
        for i in range(self.n):
            # Find last sale
            for j in range(self.L - 1, -1, -1):
                if self.hist[i, j] > EPS:
                    self.days_since_last_sale[i] = self.L - 1 - j
                    self.last_sale_qty[i] = self.hist[i, j]
                    break
            else:
                self.days_since_last_sale[i] = self.L
                self.last_sale_qty[i] = 0

            # Zero run length
            self.zero_run_length[i] = 0
            for j in range(self.L - 1, -1, -1):
                if self.hist[i, j] > EPS:
                    break
                self.zero_run_length[i] += 1

    def get_lags(self) -> dict:
        """Get lag features from ring buffer (vectorized)."""
        lags = {}
        for offset in LAG_OFFSETS:
            idx = (self.pos - offset) % self.L
            lags[f'lag_{offset}'] = self.hist[:, idx]
        return lags

    def get_rolling_features(self) -> dict:
        """Get rolling mean/sum/std features (from maintained sums)."""
        features = {}

        # 7-day window
        features['roll_mean_7'] = self.sum_7 / 7.0
        features['roll_sum_7'] = self.sum_7

        # 28-day window
        features['roll_mean_28'] = self.sum_28 / 28.0
        features['roll_sum_28'] = self.sum_28

        # Std requires sumsq
        var_28 = (self.sumsq_28 / 28.0) - (features['roll_mean_28'] ** 2)
        features['roll_std_28'] = np.sqrt(np.maximum(var_28, 0))

        # Non-zero rates
        features['nz_rate_28'] = self.nz_28.astype(np.float32) / 28.0

        return features

    def get_recency_features(self) -> dict:
        """Get recency/dormancy features."""
        return {
            'days_since_last_sale_asof': self.days_since_last_sale.astype(np.float32),
            'dormancy_capped': np.minimum(self.days_since_last_sale, 90).astype(np.float32),
            'zero_run_length_asof': self.zero_run_length.astype(np.float32),
            'last_sale_qty_asof': self.last_sale_qty,
        }

    def update(self, yhat: np.ndarray):
        """
        Update ring buffer and rolling state with new predictions.
        This is the core of recursive rollout.
        """
        # Values leaving the windows
        leave_7_idx = (self.pos - 7) % self.L
        leave_28_idx = (self.pos - 28) % self.L

        leave_7 = self.hist[:, leave_7_idx]
        leave_28 = self.hist[:, leave_28_idx]

        # Update rolling sums (add new, subtract old)
        self.sum_7 += yhat - leave_7
        self.sum_28 += yhat - leave_28
        self.sumsq_28 += (yhat ** 2) - (leave_28 ** 2)

        # Update non-zero counts
        leave_7_nz = (leave_7 > EPS).astype(np.int16)
        leave_28_nz = (leave_28 > EPS).astype(np.int16)
        enter_nz = (yhat > EPS).astype(np.int16)

        self.nz_7 += enter_nz - leave_7_nz
        self.nz_28 += enter_nz - leave_28_nz

        # Update recency state
        is_sale = yhat > EPS
        self.days_since_last_sale = np.where(is_sale, 0, self.days_since_last_sale + 1).astype(np.int16)
        self.zero_run_length = np.where(is_sale, 0, self.zero_run_length + 1).astype(np.int16)
        self.last_sale_qty = np.where(is_sale, yhat, self.last_sale_qty)

        # Write new value to ring buffer and advance position
        self.hist[:, self.pos] = yhat
        self.pos = (self.pos + 1) % self.L


def compute_calendar_features(date: datetime) -> dict:
    """Compute static calendar features for a date."""
    dow = date.weekday() + 1  # 1=Monday, 7=Sunday
    doy = date.timetuple().tm_yday

    return {
        'dow': dow,
        'is_weekend': 1 if dow >= 6 else 0,
        'week_of_year': date.isocalendar()[1],
        'month': date.month,
        'day_of_year': doy,
        'sin_doy': np.sin(2 * np.pi * doy / 365),
        'cos_doy': np.cos(2 * np.pi * doy / 365),
        'sin_dow': np.sin(2 * np.pi * dow / 7),
        'cos_dow': np.cos(2 * np.pi * dow / 7),
    }


def compute_smearing_factor(train_df, reg_model, features):
    """
    Compute smearing correction factor for log-transform bias.
    smear = mean(exp(residuals)) where residuals = log1p(y) - pred_log
    """
    train_nz = train_df[train_df['y'] > 0].copy()
    if len(train_nz) < 100:
        return 1.0

    X_train = train_nz[features]
    pred_log = reg_model.predict(X_train)
    actual_log = np.log1p(train_nz['y'].values)

    residuals = actual_log - pred_log
    smear = np.mean(np.exp(residuals))

    # Clip to reasonable range
    smear = np.clip(smear, 0.8, 1.5)
    return smear


def train_segment_models(train_df, segment, features):
    """Train classifier and regressor for a segment."""
    train_seg = train_df[train_df['segment'] == segment].copy()

    if len(train_seg) < 100:
        return None, None, 1.0

    for col in CAT_FEATURES:
        if col in train_seg.columns:
            train_seg[col] = train_seg[col].astype('category')

    X_train = train_seg[features + CAT_FEATURES]
    y_train = train_seg['y'].values

    params = SEGMENT_PARAMS[segment]

    # Classifier
    clf = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, (y_train > 0).astype(int))

    # Regressor (on non-zero only)
    train_nz = train_seg[train_seg['y'] > 0]
    if len(train_nz) < 10:
        return clf, None, 1.0

    reg = lgb.LGBMRegressor(
        num_leaves=params['num_leaves'],
        min_child_samples=max(5, params['min_child_samples'] // 2),
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        verbose=-1, n_jobs=-1, random_state=42
    )
    reg.fit(train_nz[features + CAT_FEATURES], np.log1p(train_nz['y']))

    # Compute smearing factor
    smear = compute_smearing_factor(train_seg, reg, features + CAT_FEATURES)

    return clf, reg, smear


def recursive_forecast(
    clf, reg, smear,
    series_ids: pd.DataFrame,  # store_id, sku_id
    initial_history: np.ndarray,  # (n_series, L) actual sales at cutoff
    closure_calendar: dict,  # {date: set of closed store_ids}
    forecast_dates: list,  # List of dates to forecast
    features: list,
):
    """
    Run recursive rollout forecast for a batch of series.
    Returns predictions for all series × all dates.
    """
    n_series = len(series_ids)
    n_days = len(forecast_dates)

    # Initialize ring buffer state
    state = RingBufferState(n_series, L)
    state.initialize_from_history(initial_history)

    # Output array
    predictions = np.zeros((n_series, n_days), dtype=np.float32)

    # Prepare static ID features
    store_ids = series_ids['store_id'].values
    sku_ids = series_ids['sku_id'].values

    for h, date in enumerate(forecast_dates):
        # 1. Compute calendar features (same for all series)
        cal = compute_calendar_features(date)

        # 2. Get dynamic features from ring buffer state
        lags = state.get_lags()
        rolls = state.get_rolling_features()
        recency = state.get_recency_features()

        # 3. Build feature matrix
        X = pd.DataFrame({
            'store_id': store_ids,
            'sku_id': sku_ids,
            **{k: np.full(n_series, v) for k, v in cal.items()},
            **lags,
            **rolls,
            **recency,
        })

        # Get closure flags for this date
        closed_stores = closure_calendar.get(date, set())
        X['is_store_closed'] = X['store_id'].isin(closed_stores).astype(int)

        # Convert categorical
        for col in CAT_FEATURES:
            X[col] = X[col].astype('category')

        # 4. Predict
        X_features = X[features + CAT_FEATURES]

        # Probability of sale
        p = clf.predict_proba(X_features)[:, 1]

        # Expected quantity (with smearing correction)
        if reg is not None:
            pred_log = reg.predict(X_features)
            mu = smear * np.expm1(pred_log)  # smear * (exp(pred_log) - 1)
        else:
            mu = np.zeros(n_series)

        # 5. Expected value: yhat = p * mu (NO hard threshold!)
        yhat = p * mu

        # 6. Apply constraints
        yhat = np.maximum(yhat, 0)
        yhat[X['is_store_closed'].values == 1] = 0

        # 7. Store prediction and update state
        predictions[:, h] = yhat
        state.update(yhat)

        if (h + 1) % 28 == 0:
            print(f"    Day {h+1}/{n_days}: mean_pred={yhat.mean():.4f}, "
                  f"pct_nonzero={(yhat > EPS).mean()*100:.1f}%")

    return predictions


def prepare_initial_history(train_df, series_ids, cutoff_date):
    """
    Prepare initial sales history for ring buffer.
    Returns array of shape (n_series, L) with last L days of sales.
    """
    n_series = len(series_ids)
    history = np.zeros((n_series, L), dtype=np.float32)

    # Create series index mapping
    series_idx = {
        (row['store_id'], row['sku_id']): i
        for i, row in series_ids.iterrows()
    }

    # Get last L days before cutoff
    start_date = cutoff_date - timedelta(days=L)
    recent_data = train_df[
        (train_df['date'] >= start_date.strftime('%Y-%m-%d')) &
        (train_df['date'] <= cutoff_date.strftime('%Y-%m-%d'))
    ].copy()

    recent_data['date'] = pd.to_datetime(recent_data['date'])

    # Fill history array
    for _, row in recent_data.iterrows():
        key = (row['store_id'], row['sku_id'])
        if key in series_idx:
            i = series_idx[key]
            day_offset = (row['date'].date() - start_date.date()).days
            if 0 <= day_offset < L:
                history[i, day_offset] = row['y']

    return history


def prepare_closure_calendar(train_df, val_dates):
    """
    Build closure calendar: {date: set of closed store_ids}
    """
    closure_cal = {}

    # Get closure info from data
    if 'is_store_closed' in train_df.columns:
        closed_data = train_df[train_df['is_store_closed'] == 1][['store_id', 'date']].drop_duplicates()
        for _, row in closed_data.iterrows():
            date = pd.to_datetime(row['date']).date()
            if date not in closure_cal:
                closure_cal[date] = set()
            closure_cal[date].add(row['store_id'])

    return closure_cal


def run_recursive_validation(train_df, val_df, segment, features, cutoff_date, val_dates):
    """
    Run full recursive validation for a segment.
    """
    print(f"\n  Training {segment}-segment models...")

    # Train models
    clf, reg, smear = train_segment_models(train_df, segment, features)
    if clf is None:
        return None

    print(f"    Smearing factor: {smear:.4f}")

    # Get series IDs for this segment
    train_seg = train_df[train_df['segment'] == segment]
    val_seg = val_df[val_df['segment'] == segment]

    series_ids = train_seg[['store_id', 'sku_id']].drop_duplicates().reset_index(drop=True)
    n_series = len(series_ids)
    print(f"    Series: {n_series:,}")

    # Prepare initial history
    print(f"    Preparing initial history (last {L} days)...")
    initial_history = prepare_initial_history(train_seg, series_ids, cutoff_date)

    # Prepare closure calendar
    closure_cal = prepare_closure_calendar(train_df, val_dates)

    # Run recursive forecast
    print(f"    Running recursive forecast ({len(val_dates)} days)...")
    predictions = recursive_forecast(
        clf, reg, smear,
        series_ids, initial_history, closure_cal, val_dates, features
    )

    # Map predictions back to validation dataframe
    val_seg = val_seg.copy()
    val_seg['date'] = pd.to_datetime(val_seg['date'])

    # Create prediction lookup
    series_idx = {
        (row['store_id'], row['sku_id']): i
        for i, row in series_ids.iterrows()
    }
    date_idx = {d: i for i, d in enumerate(val_dates)}

    y_pred = np.zeros(len(val_seg))
    for j, row in val_seg.iterrows():
        key = (row['store_id'], row['sku_id'])
        date = row['date'].date()
        if key in series_idx and date in date_idx:
            y_pred[j] = predictions[series_idx[key], date_idx[date]]

    val_seg['y_pred'] = y_pred

    # Compute metrics
    y_true = val_seg['y'].values
    y_pred = val_seg['y_pred'].values

    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    wfa = 100 - wmape
    bias_ratio = np.sum(y_pred) / max(np.sum(y_true), 1)

    # Weekly store metrics
    val_seg['week'] = val_seg['date'].dt.isocalendar().week
    val_seg['year'] = val_seg['date'].dt.year
    weekly = val_seg.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape_ws = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / max(np.sum(weekly['y']), 1)
    wfa_ws = 100 - wmape_ws
    bias_ws = np.sum(weekly['y_pred']) / max(np.sum(weekly['y']), 1)

    print(f"\n    {segment}-segment Results (RECURSIVE):")
    print(f"      Daily WFA:        {wfa:.2f}%")
    print(f"      Weekly Store WFA: {wfa_ws:.2f}%")
    print(f"      Bias (daily):     {bias_ratio:.3f}")
    print(f"      Bias (weekly):    {bias_ws:.3f}")

    del clf, reg
    gc.collect()

    return {
        'segment': segment,
        'n_series': n_series,
        'n_val': len(val_seg),
        'daily_wfa': wfa,
        'weekly_store_wfa': wfa_ws,
        'bias_daily': bias_ratio,
        'bias_weekly': bias_ws,
        'smear': smear,
        'val_df': val_seg,
    }


def main():
    print("=" * 80)
    print("BASELINE-3: PRODUCTION-PARITY RECURSIVE VALIDATION")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print(f"\nKey Changes from Baseline-1/2:")
    print(f"  1. Recursive rollout (predictions feed into features)")
    print(f"  2. Expected value: yhat = p * mu (no hard threshold)")
    print(f"  3. Smearing correction for log-transform bias")
    print(f"  4. Ring buffer length: L={L}")
    print("=" * 80)

    # Load data
    train_folder = '/tmp/baseline2/f1_train'
    val_folder = '/tmp/baseline2/f1_val'

    if not os.path.exists(train_folder):
        print(f"ERROR: Data not found at {train_folder}")
        print("Please download data first.")
        return

    # Load training data
    print("\nLoading training data...")
    files = sorted(glob.glob(os.path.join(train_folder, '*.parquet')))
    if not files:
        files = sorted(glob.glob(os.path.join(train_folder, '*.csv')))
        train_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        train_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Train: {len(train_df):,} rows")

    # Load validation data
    print("Loading validation data...")
    files = sorted(glob.glob(os.path.join(val_folder, '*.parquet')))
    if not files:
        files = sorted(glob.glob(os.path.join(val_folder, '*.csv')))
        val_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        val_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Val: {len(val_df):,} rows")

    # Assign ABC segments
    print("\nAssigning ABC segments...")
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

    # Dates
    cutoff_date = datetime(2025, 6, 26)
    val_start = datetime(2025, 7, 3)
    val_end = datetime(2025, 12, 17)
    val_dates = [val_start.date() + timedelta(days=i) for i in range((val_end - val_start).days + 1)]
    print(f"\nCutoff: {cutoff_date.date()}")
    print(f"Validation: {val_start.date()} to {val_end.date()} ({len(val_dates)} days)")

    # Features to use (only those available at cutoff)
    features = [f for f in ALL_FEATURES if f in train_df.columns]
    print(f"\nFeatures: {len(features)}")

    # Run recursive validation per segment
    all_results = []
    for seg in ['A', 'B', 'C']:
        result = run_recursive_validation(
            train_df, val_df, seg, features, cutoff_date, val_dates
        )
        if result:
            all_results.append(result)

    # Combine results
    print("\n" + "=" * 80)
    print("BASELINE-3 FINAL RESULTS (RECURSIVE VALIDATION)")
    print("=" * 80)

    combined_val = pd.concat([r['val_df'] for r in all_results], ignore_index=True)
    y_true = combined_val['y'].values
    y_pred = combined_val['y_pred'].values

    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    wfa = 100 - wmape
    bias = np.sum(y_pred) / max(np.sum(y_true), 1)

    combined_val['week'] = pd.to_datetime(combined_val['date']).dt.isocalendar().week
    combined_val['year'] = pd.to_datetime(combined_val['date']).dt.year
    weekly = combined_val.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape_ws = 100 * np.sum(np.abs(weekly['y'] - weekly['y_pred'])) / max(np.sum(weekly['y']), 1)
    wfa_ws = 100 - wmape_ws
    bias_ws = np.sum(weekly['y_pred']) / max(np.sum(weekly['y']), 1)

    print(f"\n  OVERALL (Production Parity):")
    print(f"    Daily WFA:        {wfa:.2f}%")
    print(f"    Weekly Store WFA: {wfa_ws:.2f}%")
    print(f"    Bias (daily):     {bias:.3f}")
    print(f"    Bias (weekly):    {bias_ws:.3f}")

    print(f"\n  PER SEGMENT:")
    print(f"  {'Segment':<10} {'Daily WFA':>12} {'Weekly WFA':>12} {'Bias':>8} {'Smear':>8}")
    print(f"  {'-'*50}")
    for r in all_results:
        print(f"  {r['segment']:<10} {r['daily_wfa']:>11.2f}% {r['weekly_store_wfa']:>11.2f}% "
              f"{r['bias_daily']:>7.3f} {r['smear']:>7.3f}")

    # Save results
    output = {
        'overall': {
            'daily_wfa': wfa,
            'weekly_store_wfa': wfa_ws,
            'bias_daily': bias,
            'bias_weekly': bias_ws,
        },
        'per_segment': [{k: v for k, v in r.items() if k != 'val_df'} for r in all_results],
        'config': {
            'ring_buffer_L': L,
            'eps': EPS,
            'method': 'recursive_rollout',
            'expected_value': 'p * mu (no hard threshold)',
            'smearing': True,
        },
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs('/tmp/baseline3', exist_ok=True)
    with open('/tmp/baseline3/recursive_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to /tmp/baseline3/recursive_results.json")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
