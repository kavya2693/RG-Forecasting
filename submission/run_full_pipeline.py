"""
COMPLETE PIPELINE: Holdout Validation + Production Forecast
- Holdout validation with proper train/test split
- Expected value formula (p × μ) instead of hard threshold
- Smearing correction for log-transform bias
- Metrics by segment (A/B/C) and behavior bucket
- Final production forecast
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 80)
print("COMPLETE FORECASTING PIPELINE")
print("=" * 80)
print(f"Started: {datetime.now()}")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "../final_data 2.csv"
SKU_ATTR_PATH = "../sku_list_attribute.csv"
OUTPUT_PATH = "outputs/forecast_168day.csv"

# Dates
HOLDOUT_CUTOFF = "2025-06-26"      # Train up to this date
HOLDOUT_END = "2025-12-17"         # Holdout ends here (168 days)
PRODUCTION_CUTOFF = "2025-12-17"   # Final training cutoff
FORECAST_START = "2025-12-18"      # Production forecast start
HORIZON_DAYS = 168

# Use sample for faster testing (set to None for full data)
SAMPLE_SKUS = 200  # Set to None for full run

print(f"\nConfiguration:")
print(f"  Holdout: Train → {HOLDOUT_CUTOFF}, Test → {HOLDOUT_END}")
print(f"  Production: Train → {PRODUCTION_CUTOFF}, Forecast → {FORECAST_START}+{HORIZON_DAYS}d")
print(f"  Sample SKUs: {SAMPLE_SKUS if SAMPLE_SKUS else 'Full data'}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD DATA")
print("=" * 80)

df_raw = pd.read_csv(DATA_PATH)
df_raw.columns = df_raw.columns.str.lower().str.strip()
if 'item_id' in df_raw.columns:
    df_raw = df_raw.rename(columns={'item_id': 'sku_id'})
df_raw['date'] = pd.to_datetime(df_raw['date'])
print(f"Loaded {len(df_raw):,} rows")

# Load SKU attributes
sku_attr = pd.read_csv(SKU_ATTR_PATH)
sku_attr.columns = sku_attr.columns.str.lower().str.strip()
sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
attr_col = [c for c in sku_attr.columns if 'attribute' in c.lower()][0]
sku_attr['is_local'] = sku_attr[attr_col].apply(lambda x: 1 if str(x).upper() in ['L', 'LI'] else 0)

# Sample if needed
if SAMPLE_SKUS:
    top_skus = df_raw.groupby('sku_id').size().nlargest(SAMPLE_SKUS).index.tolist()
    df_raw = df_raw[df_raw['sku_id'].isin(top_skus)]
    print(f"Sampled to {len(df_raw):,} rows ({SAMPLE_SKUS} SKUs)")

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================
FEATURES = [
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_28', 'days_since_last_sale', 'zero_run_length', 'last_sale_qty',
    'store_spike_pct', 'hist_spike_prob', 'had_recent_spike',
    'is_local'
]
CAT_FEATURES = ['store_id', 'sku_id']

SEGMENT_PARAMS = {
    'A': {'num_leaves': 127, 'learning_rate': 0.03, 'n_clf': 300, 'n_reg': 400, 'min_data': 10},
    'B': {'num_leaves': 63, 'learning_rate': 0.03, 'n_clf': 200, 'n_reg': 300, 'min_data': 30},
    'C': {'num_leaves': 31, 'learning_rate': 0.05, 'n_clf': 150, 'n_reg': 200, 'min_data': 50},
}

def build_panel(df, cutoff_date, sku_attr):
    """Build complete panel with features up to cutoff date."""
    cutoff = pd.to_datetime(cutoff_date)
    df = df[df['date'] <= cutoff].copy()

    # Get series
    series = df[['store_id', 'sku_id']].drop_duplicates()
    min_date = df['date'].min()
    date_range = pd.date_range(min_date, cutoff, freq='D')

    # Build spine
    series['_key'] = 1
    dates_df = pd.DataFrame({'date': date_range, '_key': 1})
    panel = series.merge(dates_df, on='_key').drop('_key', axis=1)

    # Merge sales
    sales = df[['store_id', 'sku_id', 'date', 'sales']].rename(columns={'sales': 'y'})
    panel = panel.merge(sales, on=['store_id', 'sku_id', 'date'], how='left')
    panel['y'] = panel['y'].fillna(0).clip(lower=0)

    # Convert IDs
    panel['store_id'] = panel['store_id'].astype(str)
    panel['sku_id'] = panel['sku_id'].astype(str)

    # Merge attributes
    panel = panel.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
    panel['is_local'] = panel['is_local'].fillna(0).astype(int)

    # Sort
    panel = panel.sort_values(['store_id', 'sku_id', 'date']).reset_index(drop=True)

    # Calendar features
    panel['dow'] = panel['date'].dt.dayofweek
    panel['is_weekend'] = panel['dow'].isin([5, 6]).astype(int)
    panel['week_of_year'] = panel['date'].dt.isocalendar().week.astype(int)
    panel['month'] = panel['date'].dt.month
    panel['day_of_year'] = panel['date'].dt.dayofyear
    panel['sin_doy'] = np.sin(2 * np.pi * panel['day_of_year'] / 365)
    panel['cos_doy'] = np.cos(2 * np.pi * panel['day_of_year'] / 365)
    panel['sin_dow'] = np.sin(2 * np.pi * panel['dow'] / 7)
    panel['cos_dow'] = np.cos(2 * np.pi * panel['dow'] / 7)

    # Lag features
    for lag in [1, 7, 14, 28, 56]:
        panel[f'lag_{lag}'] = panel.groupby(['store_id', 'sku_id'])['y'].shift(lag)

    # Rolling features
    for window in [7, 28]:
        panel[f'roll_mean_{window}'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        panel[f'roll_sum_{window}'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    panel['roll_std_28'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
        lambda x: x.shift(1).rolling(28, min_periods=7).std()).fillna(0)

    # Dormancy features
    panel['nz_rate_28'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).apply(lambda w: (w > 0).mean())).fillna(0)

    def calc_days_since_sale(series):
        result = np.zeros(len(series))
        last_sale_idx = -1
        for i in range(len(series)):
            if i > 0 and series.iloc[i-1] > 0:
                last_sale_idx = i - 1
            result[i] = min(i - last_sale_idx, 90) if last_sale_idx >= 0 else 90
        return pd.Series(result, index=series.index)

    panel['days_since_last_sale'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(calc_days_since_sale)

    def calc_zero_run(series):
        result = np.zeros(len(series))
        run = 0
        for i in range(len(series)):
            if i > 0:
                run = run + 1 if series.iloc[i-1] == 0 else 0
            result[i] = min(run, 60)
        return pd.Series(result, index=series.index)

    panel['zero_run_length'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(calc_zero_run)
    panel['last_sale_qty'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).ffill()).fillna(0).clip(upper=50)

    # Spike features
    series_mean = panel[panel['y'] > 0].groupby(['store_id', 'sku_id'])['y'].mean().reset_index()
    series_mean.columns = ['store_id', 'sku_id', 'hist_mean']
    panel = panel.merge(series_mean, on=['store_id', 'sku_id'], how='left')
    panel['hist_mean'] = panel['hist_mean'].fillna(1)
    panel['is_spike'] = ((panel['y'] > 3 * panel['hist_mean']) & (panel['y'] > 5)).astype(int)
    panel['store_spike_pct'] = panel.groupby(['store_id', 'date'])['is_spike'].transform('mean')
    panel['hist_spike_prob'] = panel.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0)
    panel['had_recent_spike'] = panel.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()).fillna(0)
    panel = panel.drop(columns=['hist_mean', 'is_spike'])

    # Fill NaNs
    for col in FEATURES:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    return panel

def assign_abc(panel):
    """Assign ABC segments based on sales volume."""
    series_sales = panel.groupby(['store_id', 'sku_id'])['y'].sum().reset_index()
    series_sales.columns = ['store_id', 'sku_id', 'total_sales']
    series_sales = series_sales.sort_values('total_sales', ascending=False)
    total = max(series_sales['total_sales'].sum(), 1)
    series_sales['cum_share'] = series_sales['total_sales'].cumsum() / total
    series_sales['abc'] = 'C'
    series_sales.loc[series_sales['cum_share'] <= 0.80, 'abc'] = 'A'
    series_sales.loc[(series_sales['cum_share'] > 0.80) & (series_sales['cum_share'] <= 0.95), 'abc'] = 'B'

    panel = panel.merge(series_sales[['store_id', 'sku_id', 'abc', 'total_sales']],
                        on=['store_id', 'sku_id'], how='left')
    panel['abc'] = panel['abc'].fillna('C')
    return panel, series_sales

def train_models(panel):
    """Train two-stage models for each segment."""
    models = {}
    for seg in ['A', 'B', 'C']:
        params = SEGMENT_PARAMS[seg]
        train_seg = panel[panel['abc'] == seg].copy()

        if len(train_seg) < 100:
            models[seg] = {'clf': None, 'reg': None, 'smear': 1.0}
            continue

        train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)
        for col in CAT_FEATURES:
            train_seg[col] = train_seg[col].astype('category')
        X = train_seg[FEATURES + CAT_FEATURES]

        # Classifier
        clf_data = lgb.Dataset(X, label=train_seg['y_binary'], categorical_feature=CAT_FEATURES)
        clf = lgb.train({
            'objective': 'binary', 'num_leaves': params['num_leaves'],
            'learning_rate': params['learning_rate'], 'min_data_in_leaf': params['min_data'],
            'feature_fraction': 0.8, 'verbose': -1, 'seed': 42
        }, clf_data, num_boost_round=params['n_clf'])

        # Regressor (non-zero only)
        train_nz = train_seg[train_seg['y'] > 0]
        if len(train_nz) < 10:
            models[seg] = {'clf': clf, 'reg': None, 'smear': 1.0}
            continue

        X_nz = train_nz[FEATURES + CAT_FEATURES]
        y_log = np.log1p(train_nz['y'].values)

        reg_data = lgb.Dataset(X_nz, label=y_log, categorical_feature=CAT_FEATURES)
        reg = lgb.train({
            'objective': 'regression_l1', 'num_leaves': params['num_leaves'],
            'learning_rate': params['learning_rate'], 'min_data_in_leaf': max(5, params['min_data']//2),
            'feature_fraction': 0.8, 'lambda_l2': 0.5, 'verbose': -1, 'seed': 42
        }, reg_data, num_boost_round=params['n_reg'])

        # Compute smearing factor (Duan's smearing)
        pred_log = reg.predict(X_nz)
        residuals = y_log - pred_log
        smear = np.mean(np.exp(residuals))

        models[seg] = {'clf': clf, 'reg': reg, 'smear': smear}
        print(f"  {seg}: trained on {len(train_seg):,} rows, smear={smear:.3f}")

    return models

def predict_expected_value(models, data, features, cat_features):
    """Predict using expected value formula: E[y] = p × μ (with smearing)."""
    data = data.copy()
    data['y_pred'] = 0.0

    for seg in ['A', 'B', 'C']:
        seg_mask = data['abc'] == seg
        seg_data = data[seg_mask].copy()

        if len(seg_data) == 0 or models[seg]['clf'] is None:
            continue

        for col in cat_features:
            seg_data[col] = seg_data[col].astype('category')
        X = seg_data[features + cat_features]

        # Get probability
        prob = models[seg]['clf'].predict(X)

        # Get expected value given sale
        if models[seg]['reg'] is not None:
            pred_log = models[seg]['reg'].predict(X)
            smear = models[seg]['smear']
            mu = smear * np.expm1(pred_log)  # Smearing correction
        else:
            mu = np.ones(len(X))

        # Expected value: E[y] = p × μ (NO hard threshold!)
        y_pred = prob * mu
        y_pred = np.maximum(0, y_pred)

        data.loc[seg_mask, 'y_pred'] = y_pred

    return data

def compute_metrics(data):
    """Compute WMAPE/WFA metrics at multiple levels."""
    y_true = data['y'].values
    y_pred = data['y_pred'].values

    # Daily
    wmape_daily = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)

    # Bias
    bias = np.sum(y_pred) / max(np.sum(y_true), 1)

    # Weekly Store
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['week'] = data['date'].dt.isocalendar().week.astype(int)
    data['year'] = data['date'].dt.year

    weekly_store = data.groupby(['store_id', 'year', 'week']).agg({'y': 'sum', 'y_pred': 'sum'}).reset_index()
    wmape_weekly_store = 100 * np.sum(np.abs(weekly_store['y'] - weekly_store['y_pred'])) / max(np.sum(weekly_store['y']), 1)

    return {
        'daily_wfa': 100 - wmape_daily,
        'weekly_store_wfa': 100 - wmape_weekly_store,
        'bias': bias
    }

# =============================================================================
# STEP 2: HOLDOUT VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: HOLDOUT VALIDATION")
print("=" * 80)
print(f"Train: → {HOLDOUT_CUTOFF}")
print(f"Test:  {HOLDOUT_CUTOFF} → {HOLDOUT_END} (168 days)")

# Build training panel
print("\nBuilding training panel...")
train_panel = build_panel(df_raw, HOLDOUT_CUTOFF, sku_attr)
print(f"  Training panel: {len(train_panel):,} rows")

# Assign ABC
train_panel, series_info = assign_abc(train_panel)
print(f"  ABC: A={sum(series_info['abc']=='A')}, B={sum(series_info['abc']=='B')}, C={sum(series_info['abc']=='C')}")

# Train models
print("\nTraining models...")
models = train_models(train_panel)

# Build holdout panel (with actuals for evaluation)
print("\nBuilding holdout panel...")
holdout_start = pd.to_datetime(HOLDOUT_CUTOFF) + timedelta(days=1)
holdout_end = pd.to_datetime(HOLDOUT_END)
holdout_dates = pd.date_range(holdout_start, holdout_end, freq='D')
print(f"  Holdout dates: {holdout_start.date()} to {holdout_end.date()} ({len(holdout_dates)} days)")

# Get series and their ABC
series_list = train_panel[['store_id', 'sku_id', 'abc', 'is_local']].drop_duplicates()

# Create holdout panel
series_list['_key'] = 1
dates_df = pd.DataFrame({'date': holdout_dates, '_key': 1})
holdout_panel = series_list.merge(dates_df, on='_key').drop('_key', axis=1)

# Add actuals
holdout_sales = df_raw[(df_raw['date'] >= holdout_start) & (df_raw['date'] <= holdout_end)].copy()
holdout_sales['store_id'] = holdout_sales['store_id'].astype(str)
holdout_sales['sku_id'] = holdout_sales['sku_id'].astype(str)
holdout_sales = holdout_sales.rename(columns={'sales': 'y'})
holdout_panel = holdout_panel.merge(holdout_sales[['store_id', 'sku_id', 'date', 'y']],
                                     on=['store_id', 'sku_id', 'date'], how='left')
holdout_panel['y'] = holdout_panel['y'].fillna(0).clip(lower=0)

# Add features from last known training values
print("  Adding features from training cutoff...")
last_train = train_panel[train_panel['date'] > pd.to_datetime(HOLDOUT_CUTOFF) - timedelta(days=60)]
last_stats = last_train.groupby(['store_id', 'sku_id']).agg({
    'y': 'mean', 'nz_rate_28': 'last', 'days_since_last_sale': 'last',
    'zero_run_length': 'last', 'last_sale_qty': 'last',
    'store_spike_pct': 'mean', 'hist_spike_prob': 'last', 'had_recent_spike': 'last'
}).reset_index()
last_stats.columns = ['store_id', 'sku_id', 'roll_mean_28', 'nz_rate_28', 'days_since_last_sale',
                      'zero_run_length', 'last_sale_qty', 'store_spike_pct', 'hist_spike_prob', 'had_recent_spike']
for lag in [1, 7, 14, 28, 56]:
    last_stats[f'lag_{lag}'] = last_stats['roll_mean_28']
last_stats['roll_mean_7'] = last_stats['roll_mean_28']
last_stats['roll_sum_7'] = last_stats['roll_mean_28'] * 7
last_stats['roll_sum_28'] = last_stats['roll_mean_28'] * 28
last_stats['roll_std_28'] = 0

holdout_panel = holdout_panel.merge(last_stats, on=['store_id', 'sku_id'], how='left')

# Calendar features
holdout_panel['dow'] = holdout_panel['date'].dt.dayofweek
holdout_panel['is_weekend'] = holdout_panel['dow'].isin([5, 6]).astype(int)
holdout_panel['week_of_year'] = holdout_panel['date'].dt.isocalendar().week.astype(int)
holdout_panel['month'] = holdout_panel['date'].dt.month
holdout_panel['day_of_year'] = holdout_panel['date'].dt.dayofyear
holdout_panel['sin_doy'] = np.sin(2 * np.pi * holdout_panel['day_of_year'] / 365)
holdout_panel['cos_doy'] = np.cos(2 * np.pi * holdout_panel['day_of_year'] / 365)
holdout_panel['sin_dow'] = np.sin(2 * np.pi * holdout_panel['dow'] / 7)
holdout_panel['cos_dow'] = np.cos(2 * np.pi * holdout_panel['dow'] / 7)

for col in FEATURES:
    if col not in holdout_panel.columns:
        holdout_panel[col] = 0
    holdout_panel[col] = holdout_panel[col].fillna(0)

print(f"  Holdout panel: {len(holdout_panel):,} rows")

# Predict with expected value formula
print("\nGenerating predictions (expected value: p × μ)...")
holdout_panel = predict_expected_value(models, holdout_panel, FEATURES, CAT_FEATURES)

# Compute metrics
print("\n" + "-" * 60)
print("HOLDOUT VALIDATION RESULTS")
print("-" * 60)

overall_metrics = compute_metrics(holdout_panel)
print(f"\nOVERALL:")
print(f"  Daily WFA:        {overall_metrics['daily_wfa']:.2f}%")
print(f"  Weekly Store WFA: {overall_metrics['weekly_store_wfa']:.2f}%")
print(f"  Bias:             {overall_metrics['bias']:.3f}")

# By segment
print(f"\nBY SEGMENT:")
for seg in ['A', 'B', 'C']:
    seg_data = holdout_panel[holdout_panel['abc'] == seg]
    if len(seg_data) > 0 and seg_data['y'].sum() > 0:
        seg_metrics = compute_metrics(seg_data)
        n_series = seg_data.groupby(['store_id', 'sku_id']).ngroups
        print(f"  {seg}: Daily WFA={seg_metrics['daily_wfa']:.2f}%, "
              f"Weekly Store={seg_metrics['weekly_store_wfa']:.2f}%, "
              f"Bias={seg_metrics['bias']:.3f}, Series={n_series:,}")

# Clean up
del train_panel, holdout_panel
gc.collect()

# =============================================================================
# STEP 3: PRODUCTION FORECAST
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: PRODUCTION FORECAST")
print("=" * 80)
print(f"Train: → {PRODUCTION_CUTOFF}")
print(f"Forecast: {FORECAST_START} → +{HORIZON_DAYS} days")

# Build full training panel
print("\nBuilding full training panel...")
full_panel = build_panel(df_raw, PRODUCTION_CUTOFF, sku_attr)
full_panel, series_info = assign_abc(full_panel)
print(f"  Full panel: {len(full_panel):,} rows")
print(f"  Series: {len(series_info):,}")

# Train final models
print("\nTraining final models...")
final_models = train_models(full_panel)

# Build forecast panel
print("\nBuilding forecast panel...")
forecast_start = pd.to_datetime(FORECAST_START)
forecast_dates = pd.date_range(forecast_start, periods=HORIZON_DAYS, freq='D')

series_list = full_panel[['store_id', 'sku_id', 'abc', 'is_local']].drop_duplicates()
series_list['_key'] = 1
dates_df = pd.DataFrame({'date': forecast_dates, '_key': 1})
forecast_panel = series_list.merge(dates_df, on='_key').drop('_key', axis=1)

# Add features
last_full = full_panel[full_panel['date'] > pd.to_datetime(PRODUCTION_CUTOFF) - timedelta(days=60)]
last_full_stats = last_full.groupby(['store_id', 'sku_id']).agg({
    'y': 'mean', 'nz_rate_28': 'last', 'days_since_last_sale': 'last',
    'zero_run_length': 'last', 'last_sale_qty': 'last',
    'store_spike_pct': 'mean', 'hist_spike_prob': 'last', 'had_recent_spike': 'last'
}).reset_index()
last_full_stats.columns = ['store_id', 'sku_id', 'roll_mean_28', 'nz_rate_28', 'days_since_last_sale',
                           'zero_run_length', 'last_sale_qty', 'store_spike_pct', 'hist_spike_prob', 'had_recent_spike']
for lag in [1, 7, 14, 28, 56]:
    last_full_stats[f'lag_{lag}'] = last_full_stats['roll_mean_28']
last_full_stats['roll_mean_7'] = last_full_stats['roll_mean_28']
last_full_stats['roll_sum_7'] = last_full_stats['roll_mean_28'] * 7
last_full_stats['roll_sum_28'] = last_full_stats['roll_mean_28'] * 28
last_full_stats['roll_std_28'] = 0

forecast_panel = forecast_panel.merge(last_full_stats, on=['store_id', 'sku_id'], how='left')

# Calendar features
forecast_panel['dow'] = forecast_panel['date'].dt.dayofweek
forecast_panel['is_weekend'] = forecast_panel['dow'].isin([5, 6]).astype(int)
forecast_panel['week_of_year'] = forecast_panel['date'].dt.isocalendar().week.astype(int)
forecast_panel['month'] = forecast_panel['date'].dt.month
forecast_panel['day_of_year'] = forecast_panel['date'].dt.dayofyear
forecast_panel['sin_doy'] = np.sin(2 * np.pi * forecast_panel['day_of_year'] / 365)
forecast_panel['cos_doy'] = np.cos(2 * np.pi * forecast_panel['day_of_year'] / 365)
forecast_panel['sin_dow'] = np.sin(2 * np.pi * forecast_panel['dow'] / 7)
forecast_panel['cos_dow'] = np.cos(2 * np.pi * forecast_panel['dow'] / 7)

for col in FEATURES:
    if col not in forecast_panel.columns:
        forecast_panel[col] = 0
    forecast_panel[col] = forecast_panel[col].fillna(0)

print(f"  Forecast panel: {len(forecast_panel):,} rows")

# Predict
print("\nGenerating production forecasts...")
forecast_panel = predict_expected_value(final_models, forecast_panel, FEATURES, CAT_FEATURES)

# Save output
print("\nSaving output...")
output = forecast_panel[['sku_id', 'store_id', 'date', 'y_pred']].copy()
output = output.rename(columns={'sku_id': 'item_id', 'y_pred': 'predicted_sales'})
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output['predicted_sales'] = output['predicted_sales'].round(2)

os.makedirs('outputs', exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

print(f"\nHOLDOUT VALIDATION METRICS:")
print(f"  Daily WFA:        {overall_metrics['daily_wfa']:.2f}%")
print(f"  Weekly Store WFA: {overall_metrics['weekly_store_wfa']:.2f}%")
print(f"  Bias:             {overall_metrics['bias']:.3f}")

print(f"\nPRODUCTION FORECAST:")
print(f"  Output file: {OUTPUT_PATH}")
print(f"  Rows: {len(output):,}")
print(f"  Series: {output.groupby(['item_id', 'store_id']).ngroups:,}")
print(f"  Date range: {output['date'].min()} to {output['date'].max()}")
print(f"  Days: {output['date'].nunique()}")

# Sanity checks
print(f"\nSANITY CHECKS:")
print(f"  [{'PASS' if (output['predicted_sales'] >= 0).all() else 'FAIL'}] No negative predictions")
print(f"  [{'PASS' if output['predicted_sales'].notna().all() else 'FAIL'}] No NaN predictions")
print(f"  [{'PASS' if output['date'].nunique() == HORIZON_DAYS else 'FAIL'}] Correct date range ({output['date'].nunique()} days)")

print("\n" + "=" * 80)
print(f"Completed: {datetime.now()}")
print("=" * 80)
