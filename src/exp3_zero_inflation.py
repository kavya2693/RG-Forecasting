#!/usr/bin/env python3
"""
Experiment 3: Zero-Inflated Approach for Handling 75% Zero-Rate
===============================================================
Key insight: Most error comes from predicting small values when actual is 0.
By being more conservative (higher threshold), we might reduce overall error.

Approach:
- Stage 1: Classifier predicts P(sale > 0)
- Stage 2: Regressor predicts E[sales | sales > 0]

Two prediction strategies:
1. Baseline (Expected Value): prediction = P(sale>0) * E[sales|sales>0]
2. Zero-Inflated: Only predict non-zero when classifier confidence >= threshold
   - If P(sale>0) < threshold: predict 0
   - If P(sale>0) >= threshold: predict E[sales|sales>0]
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("EXPERIMENT 3: ZERO-INFLATED APPROACH WITH HIGH CONFIDENCE THRESHOLD")
print("=" * 70)

print("\n[1] Loading and preparing data...")
df = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/final_data 2.csv')
# Rename columns to match expected names
df = df.rename(columns={'item_id': 'sku_id', 'sales': 'net_sales_qty'})
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['year'] = df['date'].dt.year

print("[2] Engineering features...")
df = df.sort_values(['sku_id', 'store_id', 'date'])
hist_stats = df.groupby(['sku_id', 'store_id']).agg({'net_sales_qty': ['mean', 'std', 'sum'], 'date': 'count'}).reset_index()
hist_stats.columns = ['sku_id', 'store_id', 'hist_mean', 'hist_std', 'hist_sum', 'n_days']
hist_stats['hist_std'] = hist_stats['hist_std'].fillna(0)
df = df.merge(hist_stats, on=['sku_id', 'store_id'], how='left')

df['lag_7_mean'] = df.groupby(['sku_id', 'store_id'])['net_sales_qty'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df['lag_7_max'] = df.groupby(['sku_id', 'store_id'])['net_sales_qty'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
df['lag_7_nonzero_rate'] = df.groupby(['sku_id', 'store_id'])['net_sales_qty'].transform(lambda x: (x.shift(1).rolling(7, min_periods=1).apply(lambda y: (y > 0).mean())))
df['lag_7_mean'] = df['lag_7_mean'].fillna(df['hist_mean'])
df['lag_7_max'] = df['lag_7_max'].fillna(df['hist_mean'])
df['lag_7_nonzero_rate'] = df['lag_7_nonzero_rate'].fillna(0.25)

le_sku = LabelEncoder()
le_store = LabelEncoder()
df['sku_encoded'] = le_sku.fit_transform(df['sku_id'])
df['store_encoded'] = le_store.fit_transform(df['store_id'])
df['is_nonzero'] = (df['net_sales_qty'] > 0).astype(int)

feature_cols = ['sku_encoded', 'store_encoded', 'month', 'day_of_week', 'week_of_year', 'hist_mean', 'hist_std', 'lag_7_mean', 'lag_7_max', 'lag_7_nonzero_rate']
df_clean = df.dropna(subset=feature_cols + ['net_sales_qty'])
print(f"   Clean dataset: {len(df_clean):,} rows")
actual_zero_rate = (df_clean['net_sales_qty'] == 0).mean()
print(f"   Actual zero rate: {actual_zero_rate:.1%}")

df_clean = df_clean.sort_values('date')
split_idx = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]
print(f"   Training: {len(train_df):,}, Test: {len(test_df):,}")

X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_test = test_df['net_sales_qty']

print("\n[3] Training classifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train, train_df['is_nonzero'])
proba_nonzero = clf.predict_proba(X_test)[:, 1]

print("[4] Training regressor...")
train_nonzero = train_df[train_df['net_sales_qty'] > 0]
reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
reg.fit(train_nonzero[feature_cols], train_nonzero['net_sales_qty'])
pred_if_nonzero = reg.predict(X_test)

def calc_wmape(actual, predicted):
    total = np.sum(np.abs(actual))
    return np.sum(np.abs(actual - predicted)) / total if total > 0 else np.nan

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Baseline: Expected value = P(nonzero) * E[sales | nonzero]
pred_baseline = proba_nonzero * pred_if_nonzero
wmape_baseline = calc_wmape(y_test.values, pred_baseline)
print(f"\n[Baseline] Expected Value Approach")
print(f"   WMAPE: {wmape_baseline:.4f} ({wmape_baseline*100:.2f}%)")

# Analyze error sources
test_zeros = y_test.values == 0
test_nonzeros = y_test.values > 0
error_from_zeros = np.sum(np.abs(pred_baseline[test_zeros]))
error_from_nonzeros = np.sum(np.abs(y_test.values[test_nonzeros] - pred_baseline[test_nonzeros]))
total_error = np.sum(np.abs(y_test.values - pred_baseline))
print(f"\n[Error Analysis]")
print(f"   Error from predicting on true zeros: {error_from_zeros:.2f} ({error_from_zeros/total_error*100:.1f}%)")
print(f"   Error from predicting on true non-zeros: {error_from_nonzeros:.2f} ({error_from_nonzeros/total_error*100:.1f}%)")

print(f"\n[Zero-Inflated Approach - Testing Thresholds]")
print("-" * 50)

best_thresh = 0
best_wmape = wmape_baseline

for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    # Zero-inflated: predict 0 if probability < threshold, else use regressor prediction
    pred_zi = np.where(proba_nonzero >= thresh, pred_if_nonzero, 0)
    wmape_zi = calc_wmape(y_test.values, pred_zi)

    # Calculate how many predictions become zero
    n_pred_zero = np.sum(proba_nonzero < thresh)
    pct_pred_zero = n_pred_zero / len(proba_nonzero) * 100

    # Improvement vs baseline
    improvement = (wmape_baseline - wmape_zi) / wmape_baseline * 100

    indicator = " <-- BEST" if wmape_zi < best_wmape else ""
    print(f"   Threshold {thresh}: WMAPE={wmape_zi:.4f} ({wmape_zi*100:.2f}%) | Pred zeros: {pct_pred_zero:.1f}% | vs Baseline: {improvement:+.2f}%{indicator}")

    if wmape_zi < best_wmape:
        best_wmape = wmape_zi
        best_thresh = thresh

print("-" * 50)
print(f"\n[SUMMARY]")
print(f"   Baseline WMAPE: {wmape_baseline:.4f} ({wmape_baseline*100:.2f}%)")
print(f"   Best Threshold: {best_thresh}")
print(f"   Best WMAPE: {best_wmape:.4f} ({best_wmape*100:.2f}%)")
improvement = (wmape_baseline - best_wmape) / wmape_baseline * 100
print(f"   Improvement: {improvement:.2f}%")

print("\nDone!")
