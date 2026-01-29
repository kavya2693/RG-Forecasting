"""
Smoke Test: Run notebook logic end-to-end on sample data
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

print("=" * 70)
print("SMOKE TEST: RG-Forecasting Notebook")
print("=" * 70)
print(f"Started: {datetime.now()}")

# === CONFIG ===
DATA_PATH = "test_sample.csv"
SKU_ATTR_PATH = "../sku_list_attribute.csv"
OUTPUT_PATH = "outputs/forecast_168day_test.csv"
CUTOFF_DATE = "2025-12-17"
FORECAST_START = "2025-12-18"
HORIZON_DAYS = 168

# === 1. LOAD DATA ===
print("\n[1/10] Loading data...")
df_raw = pd.read_csv(DATA_PATH)
df_raw.columns = df_raw.columns.str.lower().str.strip()
if 'item_id' in df_raw.columns:
    df_raw = df_raw.rename(columns={'item_id': 'sku_id'})
df_raw['date'] = pd.to_datetime(df_raw['date'])
print(f"  Loaded {len(df_raw):,} rows")

# Load SKU attributes
sku_attr = pd.read_csv(SKU_ATTR_PATH)
sku_attr.columns = sku_attr.columns.str.lower().str.strip()
sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)
attr_col = [c for c in sku_attr.columns if 'attribute' in c.lower()][0]
sku_attr['is_local'] = sku_attr[attr_col].apply(lambda x: 1 if str(x).upper() in ['L', 'LI'] else 0)
print(f"  SKU attributes: {len(sku_attr):,}")

# === 2. BUILD SPINE ===
print("\n[2/10] Building spine (complete panel)...")
stores = df_raw['store_id'].unique()
skus = df_raw['sku_id'].unique()
cutoff = pd.to_datetime(CUTOFF_DATE)
min_date = df_raw['date'].min()
date_range = pd.date_range(min_date, cutoff, freq='D')

series = df_raw[['store_id', 'sku_id']].drop_duplicates()
n_series = len(series)
print(f"  Series: {n_series}")
print(f"  Date range: {min_date.date()} to {cutoff.date()} ({len(date_range)} days)")

# Cross join
dates_df = pd.DataFrame({'date': date_range})
series['_key'] = 1
dates_df['_key'] = 1
spine = series.merge(dates_df, on='_key').drop('_key', axis=1)
print(f"  Spine size: {len(spine):,}")

# Merge sales
df_train = df_raw[df_raw['date'] <= cutoff][['store_id', 'sku_id', 'date', 'sales']].copy()
df_train = df_train.rename(columns={'sales': 'y'})
panel = spine.merge(df_train, on=['store_id', 'sku_id', 'date'], how='left')
panel['y'] = panel['y'].fillna(0)

zero_rate = (panel['y'] == 0).mean() * 100
print(f"  Zero-sales rate: {zero_rate:.1f}%")

# === 3. CLEANING ===
print("\n[3/10] Cleaning...")
panel['y'] = panel['y'].clip(lower=0)
panel['store_id'] = panel['store_id'].astype(str)
panel['sku_id'] = panel['sku_id'].astype(str)
panel = panel.merge(sku_attr[['sku_id', 'is_local']], on='sku_id', how='left')
panel['is_local'] = panel['is_local'].fillna(0).astype(int)
print("  ✓ Cleaned")

# === 4. FEATURE ENGINEERING ===
print("\n[4/10] Engineering features...")
panel = panel.sort_values(['store_id', 'sku_id', 'date']).reset_index(drop=True)

# Calendar
panel['dow'] = panel['date'].dt.dayofweek
panel['is_weekend'] = panel['dow'].isin([5, 6]).astype(int)
panel['week_of_year'] = panel['date'].dt.isocalendar().week.astype(int)
panel['month'] = panel['date'].dt.month
panel['day_of_year'] = panel['date'].dt.dayofyear
panel['sin_doy'] = np.sin(2 * np.pi * panel['day_of_year'] / 365)
panel['cos_doy'] = np.cos(2 * np.pi * panel['day_of_year'] / 365)
panel['sin_dow'] = np.sin(2 * np.pi * panel['dow'] / 7)
panel['cos_dow'] = np.cos(2 * np.pi * panel['dow'] / 7)

# Lags
for lag in [1, 7, 14, 28, 56]:
    panel[f'lag_{lag}'] = panel.groupby(['store_id', 'sku_id'])['y'].shift(lag)

# Rolling
for window in [7, 28]:
    panel[f'roll_mean_{window}'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    panel[f'roll_sum_{window}'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum())
panel['roll_std_28'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
    lambda x: x.shift(1).rolling(28, min_periods=7).std()).fillna(0)

# Dormancy
panel['nz_rate_28'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(
    lambda x: x.shift(1).rolling(28, min_periods=1).apply(lambda w: (w > 0).mean())).fillna(0)
# Days since last sale - simpler approach
def calc_days_since_sale(series):
    result = np.zeros(len(series))
    last_sale_idx = -1
    for i in range(len(series)):
        if i > 0 and series.iloc[i-1] > 0:
            last_sale_idx = i - 1
        if last_sale_idx >= 0:
            result[i] = min(i - last_sale_idx, 90)
        else:
            result[i] = 90
    return pd.Series(result, index=series.index)

panel['days_since_last_sale'] = panel.groupby(['store_id', 'sku_id'])['y'].transform(calc_days_since_sale)
# Zero run length - simpler approach
def calc_zero_run(series):
    result = np.zeros(len(series))
    run = 0
    for i in range(len(series)):
        if i > 0:
            if series.iloc[i-1] == 0:
                run += 1
            else:
                run = 0
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
feature_cols = ['dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_28', 'days_since_last_sale', 'zero_run_length', 'last_sale_qty',
    'store_spike_pct', 'hist_spike_prob', 'had_recent_spike', 'is_local']
for col in feature_cols:
    if col in panel.columns:
        panel[col] = panel[col].fillna(0)
print(f"  ✓ {len(feature_cols)} features created")

# === 5. ABC SEGMENTATION ===
print("\n[5/10] ABC segmentation...")
series_sales = panel.groupby(['store_id', 'sku_id'])['y'].sum().reset_index()
series_sales.columns = ['store_id', 'sku_id', 'total_sales']
series_sales = series_sales.sort_values('total_sales', ascending=False)
total = series_sales['total_sales'].sum()
series_sales['cum_share'] = series_sales['total_sales'].cumsum() / max(total, 1)
series_sales['abc'] = 'C'
series_sales.loc[series_sales['cum_share'] <= 0.80, 'abc'] = 'A'
series_sales.loc[(series_sales['cum_share'] > 0.80) & (series_sales['cum_share'] <= 0.95), 'abc'] = 'B'
panel = panel.merge(series_sales[['store_id', 'sku_id', 'abc', 'total_sales']], on=['store_id', 'sku_id'], how='left')
panel['abc'] = panel['abc'].fillna('C')
print(f"  A: {(series_sales['abc']=='A').sum()}, B: {(series_sales['abc']=='B').sum()}, C: {(series_sales['abc']=='C').sum()}")

# === 6. MODEL TRAINING ===
print("\n[6/10] Training models...")
FEATURES = ['dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_28', 'days_since_last_sale', 'zero_run_length', 'last_sale_qty',
    'store_spike_pct', 'hist_spike_prob', 'had_recent_spike', 'is_local']
CAT_FEATURES = ['store_id', 'sku_id']

SEGMENT_PARAMS = {
    'A': {'num_leaves': 63, 'learning_rate': 0.05, 'n_clf': 100, 'n_reg': 100, 'min_data': 10, 'threshold': 0.6},
    'B': {'num_leaves': 31, 'learning_rate': 0.05, 'n_clf': 50, 'n_reg': 50, 'min_data': 20, 'threshold': 0.6},
    'C': {'num_leaves': 15, 'learning_rate': 0.05, 'n_clf': 30, 'n_reg': 30, 'min_data': 50, 'threshold': 0.7},
}

models = {}
for seg in ['A', 'B', 'C']:
    params = SEGMENT_PARAMS[seg]
    train_seg = panel[panel['abc'] == seg].copy()
    if len(train_seg) < 100:
        models[seg] = {'clf': None, 'reg': None, 'params': params}
        continue

    train_seg['y_binary'] = (train_seg['y'] > 0).astype(int)
    for col in CAT_FEATURES:
        train_seg[col] = train_seg[col].astype('category')
    X_train = train_seg[FEATURES + CAT_FEATURES]

    # Classifier
    clf_data = lgb.Dataset(X_train, label=train_seg['y_binary'], categorical_feature=CAT_FEATURES)
    clf = lgb.train({'objective': 'binary', 'num_leaves': params['num_leaves'],
                     'learning_rate': params['learning_rate'], 'verbose': -1},
                    clf_data, num_boost_round=params['n_clf'])

    # Regressor
    train_nz = train_seg[train_seg['y'] > 0]
    reg = None
    if len(train_nz) >= 10:
        X_nz = train_nz[FEATURES + CAT_FEATURES]
        reg_data = lgb.Dataset(X_nz, label=np.log1p(train_nz['y'].values), categorical_feature=CAT_FEATURES)
        reg = lgb.train({'objective': 'regression_l1', 'num_leaves': params['num_leaves'],
                         'learning_rate': params['learning_rate'], 'verbose': -1},
                        reg_data, num_boost_round=params['n_reg'])

    models[seg] = {'clf': clf, 'reg': reg, 'params': params}
    print(f"  {seg}: trained on {len(train_seg):,} rows")

# === 7. BUILD FORECAST PANEL ===
print("\n[7/10] Building forecast panel...")
forecast_start = pd.to_datetime(FORECAST_START)
forecast_dates = pd.date_range(forecast_start, periods=HORIZON_DAYS, freq='D')
print(f"  Forecast: {forecast_dates[0].date()} to {forecast_dates[-1].date()} ({len(forecast_dates)} days)")

series_list = panel[['store_id', 'sku_id', 'abc', 'is_local']].drop_duplicates()
series_list['_key'] = 1
dates_df = pd.DataFrame({'date': forecast_dates})
dates_df['_key'] = 1
forecast_panel = series_list.merge(dates_df, on='_key').drop('_key', axis=1)
print(f"  Forecast panel: {len(forecast_panel):,} rows")

# Add features
forecast_panel['dow'] = forecast_panel['date'].dt.dayofweek
forecast_panel['is_weekend'] = forecast_panel['dow'].isin([5, 6]).astype(int)
forecast_panel['week_of_year'] = forecast_panel['date'].dt.isocalendar().week.astype(int)
forecast_panel['month'] = forecast_panel['date'].dt.month
forecast_panel['day_of_year'] = forecast_panel['date'].dt.dayofyear
forecast_panel['sin_doy'] = np.sin(2 * np.pi * forecast_panel['day_of_year'] / 365)
forecast_panel['cos_doy'] = np.cos(2 * np.pi * forecast_panel['day_of_year'] / 365)
forecast_panel['sin_dow'] = np.sin(2 * np.pi * forecast_panel['dow'] / 7)
forecast_panel['cos_dow'] = np.cos(2 * np.pi * forecast_panel['dow'] / 7)

# Get last values
lookback = panel[panel['date'] > cutoff - timedelta(days=60)]
last_stats = lookback.groupby(['store_id', 'sku_id']).agg({
    'y': 'mean', 'nz_rate_28': 'last', 'days_since_last_sale': 'last',
    'zero_run_length': 'last', 'last_sale_qty': 'last',
    'store_spike_pct': 'mean', 'hist_spike_prob': 'last', 'had_recent_spike': 'last'
}).reset_index()
last_stats.columns = ['store_id', 'sku_id', 'roll_mean_28', 'nz_rate_28', 'days_since_last_sale',
                      'zero_run_length', 'last_sale_qty', 'store_spike_pct', 'hist_spike_prob', 'had_recent_spike']
last_stats['lag_1'] = last_stats['roll_mean_28']
last_stats['lag_7'] = last_stats['roll_mean_28']
last_stats['lag_14'] = last_stats['roll_mean_28']
last_stats['lag_28'] = last_stats['roll_mean_28']
last_stats['lag_56'] = last_stats['roll_mean_28']
last_stats['roll_mean_7'] = last_stats['roll_mean_28']
last_stats['roll_sum_7'] = last_stats['roll_mean_28'] * 7
last_stats['roll_sum_28'] = last_stats['roll_mean_28'] * 28
last_stats['roll_std_28'] = 0

forecast_panel = forecast_panel.merge(last_stats, on=['store_id', 'sku_id'], how='left')
for col in FEATURES:
    if col not in forecast_panel.columns:
        forecast_panel[col] = 0
    forecast_panel[col] = forecast_panel[col].fillna(0)

# === 8. GENERATE PREDICTIONS ===
print("\n[8/10] Generating predictions...")
forecast_panel['predicted_sales'] = 0.0

for seg in ['A', 'B', 'C']:
    seg_mask = forecast_panel['abc'] == seg
    seg_data = forecast_panel[seg_mask].copy()
    if len(seg_data) == 0 or models[seg]['clf'] is None:
        continue

    for col in CAT_FEATURES:
        seg_data[col] = seg_data[col].astype('category')
    X = seg_data[FEATURES + CAT_FEATURES]

    prob = models[seg]['clf'].predict(X)
    if models[seg]['reg'] is not None:
        pred_value = np.expm1(models[seg]['reg'].predict(X))
    else:
        pred_value = np.ones(len(X))

    threshold = models[seg]['params']['threshold']
    y_pred = np.where(prob > threshold, pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    forecast_panel.loc[seg_mask, 'predicted_sales'] = y_pred
    print(f"  {seg}: {len(seg_data):,} rows, avg pred = {y_pred.mean():.3f}")

# === 9. SAVE OUTPUT ===
print("\n[9/10] Saving output...")
output = forecast_panel[['sku_id', 'store_id', 'date', 'predicted_sales']].copy()
output = output.rename(columns={'sku_id': 'item_id'})
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output['predicted_sales'] = output['predicted_sales'].round(2)

os.makedirs('outputs', exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)
print(f"  Saved to {OUTPUT_PATH}")

# === 10. VALIDATION ===
print("\n[10/10] VALIDATION REPORT")
print("=" * 70)

checks = []

# 1. File exists
file_exists = os.path.exists(OUTPUT_PATH)
checks.append(("Output file exists", file_exists))
print(f"[{'PASS' if file_exists else 'FAIL'}] Output file exists: {OUTPUT_PATH}")

# 2. Schema
expected_cols = ['item_id', 'store_id', 'date', 'predicted_sales']
actual_cols = list(output.columns)
schema_ok = actual_cols == expected_cols
checks.append(("Schema correct", schema_ok))
print(f"[{'PASS' if schema_ok else 'FAIL'}] Schema: {actual_cols}")

# 3. Row count
n_series_forecast = forecast_panel.groupby(['store_id', 'sku_id']).ngroups
expected_rows = n_series_forecast * HORIZON_DAYS
actual_rows = len(output)
rows_ok = expected_rows == actual_rows
checks.append(("Row count correct", rows_ok))
print(f"[{'PASS' if rows_ok else 'FAIL'}] Rows: {actual_rows:,} (expected {n_series_forecast} × {HORIZON_DAYS} = {expected_rows:,})")

# 4. Date range
min_date_out = output['date'].min()
max_date_out = output['date'].max()
n_dates = output['date'].nunique()
dates_ok = (min_date_out == FORECAST_START and n_dates == HORIZON_DAYS)
checks.append(("Date range correct", dates_ok))
print(f"[{'PASS' if dates_ok else 'FAIL'}] Dates: {min_date_out} to {max_date_out} ({n_dates} days)")

# 5. No NaNs
nan_count = output['predicted_sales'].isna().sum()
no_nans = nan_count == 0
checks.append(("No NaN predictions", no_nans))
print(f"[{'PASS' if no_nans else 'FAIL'}] NaN predictions: {nan_count}")

# 6. No negatives
neg_count = (output['predicted_sales'] < 0).sum()
no_negs = neg_count == 0
checks.append(("No negative predictions", no_negs))
print(f"[{'PASS' if no_negs else 'FAIL'}] Negative predictions: {neg_count}")

# Extra checks
print("\n--- EXTRA SANITY CHECKS ---")
zero_history_series = (series_sales['total_sales'] == 0).sum()
print(f"Series with all-zeros history: {zero_history_series} ({zero_history_series/len(series_sales)*100:.1f}%)")

top_items = output.groupby('item_id')['predicted_sales'].sum().nlargest(5)
print(f"Top 5 items by predicted total:")
for item, total in top_items.items():
    print(f"  {item}: {total:.1f}")

print("\n" + "=" * 70)
all_pass = all(c[1] for c in checks)
print(f"{'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")
print("=" * 70)
print(f"Finished: {datetime.now()}")
