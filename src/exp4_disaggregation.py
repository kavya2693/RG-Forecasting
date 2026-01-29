#!/usr/bin/env python3
"""
Experiment 4: Temporal Disaggregation
Forecast at weekly level and disaggregate to daily using DOW patterns.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("EXPERIMENT 4: TEMPORAL DISAGGREGATION")
print("Weekly Forecast -> Daily via DOW Patterns")
print("="*60)

# Load data
print("\n[1] Loading data...")
train = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/data/highvol/train.csv', parse_dates=['date'])
val = pd.read_csv('/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/data/highvol/val.csv', parse_dates=['date'])

print(f"    Train: {len(train):,} rows, {train['date'].min()} to {train['date'].max()}")
print(f"    Val:   {len(val):,} rows, {val['date'].min()} to {val['date'].max()}")

# Calculate DOW weights per SKU-Store
print("\n[2] Learning DOW patterns from training data...")
dow_sales = train.groupby(['store_id', 'sku_id', 'dow'])['y'].sum().reset_index()
dow_sales.columns = ['store_id', 'sku_id', 'dow', 'total_sales']

weekly_total = dow_sales.groupby(['store_id', 'sku_id'])['total_sales'].sum().reset_index()
weekly_total.columns = ['store_id', 'sku_id', 'weekly_total']

dow_sales = dow_sales.merge(weekly_total, on=['store_id', 'sku_id'])
dow_sales['dow_weight'] = dow_sales['total_sales'] / dow_sales['weekly_total'].replace(0, 1) * 7
dow_weights = dow_sales[['store_id', 'sku_id', 'dow', 'dow_weight']]

# Show average DOW weights across all SKU-stores
avg_dow = dow_weights.groupby('dow')['dow_weight'].mean()
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print("\n    Average DOW Weights (1.0 = uniform):")
for i, name in enumerate(dow_names):
    print(f"      {name}: {avg_dow.get(i, 1.0):.3f}")

# Create weekly forecasts from last 4 weeks of training
print("\n[3] Creating weekly forecasts from recent history...")
last_date = train['date'].max()
recent = train[train['date'] > last_date - timedelta(weeks=4)]

recent['year_week'] = recent['date'].dt.isocalendar().year.astype(str) + '-W' + \
                       recent['date'].dt.isocalendar().week.astype(str).str.zfill(2)

# Weekly sums
weekly_sums = recent.groupby(['store_id', 'sku_id', 'year_week'])['y'].sum().reset_index()
# Average weekly forecast
weekly_avg = weekly_sums.groupby(['store_id', 'sku_id'])['y'].mean().reset_index()
weekly_avg.columns = ['store_id', 'sku_id', 'weekly_forecast']

print(f"    Created weekly forecasts for {len(weekly_avg):,} SKU-Store combinations")

# Merge forecasts and DOW weights to validation set
print("\n[4] Disaggregating weekly forecasts to daily...")
val_with_forecast = val.merge(weekly_avg, on=['store_id', 'sku_id'], how='left')
val_with_forecast = val_with_forecast.merge(dow_weights, on=['store_id', 'sku_id', 'dow'], how='left')

# Fill missing values
val_with_forecast['weekly_forecast'] = val_with_forecast['weekly_forecast'].fillna(0)
val_with_forecast['dow_weight'] = val_with_forecast['dow_weight'].fillna(1.0)

# Disaggregated predictions
val_with_forecast['daily_pred_disagg'] = (val_with_forecast['weekly_forecast'] / 7) * val_with_forecast['dow_weight']
val_with_forecast['daily_pred_uniform'] = val_with_forecast['weekly_forecast'] / 7

# Baseline predictions (from features if available)
if 'lag_7' in val_with_forecast.columns:
    val_with_forecast['pred_lag7'] = val_with_forecast['lag_7'].fillna(0)
if 'roll_mean_7' in val_with_forecast.columns:
    val_with_forecast['pred_rollmean7'] = val_with_forecast['roll_mean_7'].fillna(0)

# WMAPE calculation function
def wmape(actual, predicted):
    """Weighted Mean Absolute Percentage Error"""
    total = np.sum(actual)
    if total == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / total * 100

# Calculate metrics
print("\n[5] Calculating WMAPE metrics...")
print("\n" + "="*60)
print("RESULTS: DAILY WMAPE")
print("="*60)

actual = val_with_forecast['y'].values

# Main comparison
disagg_wmape = wmape(actual, val_with_forecast['daily_pred_disagg'].values)
uniform_wmape = wmape(actual, val_with_forecast['daily_pred_uniform'].values)

print(f"\n  DOW-Weighted Disaggregation:  {disagg_wmape:.2f}%")
print(f"  Uniform Disaggregation:        {uniform_wmape:.2f}%")
print(f"  Improvement from DOW:          {uniform_wmape - disagg_wmape:.2f}pp")

# Baselines
if 'pred_lag7' in val_with_forecast.columns:
    lag7_wmape = wmape(actual, val_with_forecast['pred_lag7'].values)
    print(f"\n  Lag-7 Baseline:                {lag7_wmape:.2f}%")

if 'pred_rollmean7' in val_with_forecast.columns:
    rollmean_wmape = wmape(actual, val_with_forecast['pred_rollmean7'].values)
    print(f"  Rolling Mean-7 Baseline:       {rollmean_wmape:.2f}%")

# Weekly aggregated accuracy (for reference)
print("\n" + "="*60)
print("RESULTS: WEEKLY AGGREGATED WMAPE")
print("="*60)

val_with_forecast['year_week'] = val_with_forecast['date'].dt.isocalendar().year.astype(str) + '-W' + \
                                  val_with_forecast['date'].dt.isocalendar().week.astype(str).str.zfill(2)

weekly_actual = val_with_forecast.groupby(['store_id', 'sku_id', 'year_week'])['y'].sum().reset_index()
weekly_disagg = val_with_forecast.groupby(['store_id', 'sku_id', 'year_week'])['daily_pred_disagg'].sum().reset_index()
weekly_uniform = val_with_forecast.groupby(['store_id', 'sku_id', 'year_week'])['daily_pred_uniform'].sum().reset_index()

weekly_merged = weekly_actual.merge(weekly_disagg, on=['store_id', 'sku_id', 'year_week'])
weekly_merged = weekly_merged.merge(weekly_uniform, on=['store_id', 'sku_id', 'year_week'])

weekly_disagg_wmape = wmape(weekly_merged['y'].values, weekly_merged['daily_pred_disagg'].values)
weekly_uniform_wmape = wmape(weekly_merged['y'].values, weekly_merged['daily_pred_uniform'].values)

print(f"\n  DOW-Weighted (aggregated):     {weekly_disagg_wmape:.2f}%")
print(f"  Uniform (aggregated):          {weekly_uniform_wmape:.2f}%")

# Analysis by DOW
print("\n" + "="*60)
print("RESULTS: WMAPE BY DAY OF WEEK")
print("="*60)
print("\n  Day     DOW-Weight  Uniform   Improvement")
print("  " + "-"*45)

for i, name in enumerate(dow_names):
    mask = val_with_forecast['dow'] == i
    if mask.sum() > 0:
        dow_actual = val_with_forecast.loc[mask, 'y'].values
        dow_disagg = val_with_forecast.loc[mask, 'daily_pred_disagg'].values
        dow_uniform = val_with_forecast.loc[mask, 'daily_pred_uniform'].values

        dow_disagg_wmape = wmape(dow_actual, dow_disagg)
        dow_uniform_wmape = wmape(dow_actual, dow_uniform)
        improvement = dow_uniform_wmape - dow_disagg_wmape

        print(f"  {name}:     {dow_disagg_wmape:6.2f}%    {dow_uniform_wmape:6.2f}%    {improvement:+.2f}pp")

# SKU-Store level breakdown
print("\n" + "="*60)
print("RESULTS: SKU-STORE LEVEL ACCURACY")
print("="*60)

sku_store_results = []
for (store_id, sku_id), group in val_with_forecast.groupby(['store_id', 'sku_id']):
    actual_grp = group['y'].values
    disagg_grp = group['daily_pred_disagg'].values
    uniform_grp = group['daily_pred_uniform'].values

    sku_store_results.append({
        'store_id': store_id,
        'sku_id': sku_id,
        'actual_sum': actual_grp.sum(),
        'wmape_disagg': wmape(actual_grp, disagg_grp),
        'wmape_uniform': wmape(actual_grp, uniform_grp)
    })

sku_store_df = pd.DataFrame(sku_store_results)
sku_store_df['improvement'] = sku_store_df['wmape_uniform'] - sku_store_df['wmape_disagg']

print(f"\n  Total SKU-Store combinations: {len(sku_store_df)}")
print(f"  SKUs improved by DOW weights: {(sku_store_df['improvement'] > 0).sum()} ({(sku_store_df['improvement'] > 0).mean()*100:.1f}%)")
print(f"  Mean improvement:             {sku_store_df['improvement'].mean():.2f}pp")
print(f"  Median improvement:           {sku_store_df['improvement'].median():.2f}pp")

# Volume-weighted metrics
total_volume = sku_store_df['actual_sum'].sum()
weighted_disagg = (sku_store_df['wmape_disagg'] * sku_store_df['actual_sum']).sum() / total_volume
weighted_uniform = (sku_store_df['wmape_uniform'] * sku_store_df['actual_sum']).sum() / total_volume

print(f"\n  Volume-weighted WMAPE (DOW):     {weighted_disagg:.2f}%")
print(f"  Volume-weighted WMAPE (Uniform): {weighted_uniform:.2f}%")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if disagg_wmape < uniform_wmape:
    print(f"\n  DOW-weighted disaggregation IMPROVES daily accuracy")
    print(f"  by {uniform_wmape - disagg_wmape:.2f} percentage points over uniform split.")
else:
    print(f"\n  DOW-weighted disaggregation does NOT improve over uniform.")
    print(f"  Difference: {disagg_wmape - uniform_wmape:.2f}pp worse.")

print("\n" + "="*60)
