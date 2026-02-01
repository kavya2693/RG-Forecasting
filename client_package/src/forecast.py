#!/usr/bin/env python3
"""
RG-Forecasting Production Inference
====================================
Generate forecasts using trained models.
"""

import os
import pickle
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

UAE_HOLIDAYS = [
    '2024-03-10', '2024-04-09', '2025-02-28', '2025-03-30',
    '2024-06-16', '2025-06-06', '2024-12-02', '2025-12-02',
    '2024-01-01', '2025-01-01', '2026-01-01',
]


def load_model(path):
    """Load trained model."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add required features for inference."""
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])

    # Calendar features
    df['dow'] = df['date_dt'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['week_of_year'] = df['date_dt'].dt.isocalendar().week.astype(int)
    df['month'] = df['date_dt'].dt.month
    df['day_of_year'] = df['date_dt'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['sin_dow'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dow'] / 7)

    # Holiday
    holiday_dates = pd.to_datetime(UAE_HOLIDAYS)
    df['is_holiday'] = df['date_dt'].isin(holiday_dates).astype(int)

    df.drop(columns=['date_dt'], inplace=True)
    return df


def assign_abc_from_history(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Assign ABC based on historical sales."""
    series = history_df.groupby(['store_id', 'sku_id'])['sales'].sum().reset_index()
    series.columns = ['store_id', 'sku_id', 'total']
    series = series.sort_values('total', ascending=False)
    series['cum'] = series['total'].cumsum() / series['total'].sum()
    series['abc'] = series['cum'].apply(lambda x: 'A' if x <= 0.80 else ('B' if x <= 0.95 else 'C'))

    df = df.merge(series[['store_id', 'sku_id', 'abc']], on=['store_id', 'sku_id'], how='left')
    df['abc'] = df['abc'].fillna('C')
    return df


def generate_forecast(model_dir: str, input_df: pd.DataFrame, horizon_days: int = 168) -> pd.DataFrame:
    """Generate forecasts for all store-SKU combinations."""

    # Load models
    models = {}
    for seg in ['A', 'B', 'C']:
        path = os.path.join(model_dir, f'model_{seg}.pkl')
        if os.path.exists(path):
            models[seg] = load_model(path)

    # Add features
    input_df = add_features(input_df)

    # Predict
    input_df['predicted_sales'] = 0.0

    for seg, model_data in models.items():
        mask = input_df['abc'] == seg
        if mask.sum() == 0:
            continue

        clf = model_data['clf']
        reg = model_data['reg']
        smear = model_data['smear']
        features = model_data['features']
        params = model_data['params']

        # Ensure all features exist
        missing = [f for f in features if f not in input_df.columns]
        for f in missing:
            input_df[f] = 0

        p_sale = clf.predict_proba(input_df[mask][features])[:, 1]
        log_qty = reg.predict(input_df[mask][features])
        qty = np.expm1(log_qty) * smear

        y_pred = np.where(p_sale >= params['threshold'], qty * params['calibration'], 0)
        input_df.loc[mask, 'predicted_sales'] = np.maximum(y_pred, 0)

    # Store closure override
    if 'is_store_closed' in input_df.columns:
        input_df.loc[input_df['is_store_closed'] == 1, 'predicted_sales'] = 0

    return input_df[['store_id', 'sku_id', 'date', 'predicted_sales', 'abc']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Forecasts')
    parser.add_argument('--model-dir', required=True, help='Directory with trained models')
    parser.add_argument('--input', required=True, help='Input CSV with features')
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    forecast_df = generate_forecast(args.model_dir, input_df)
    forecast_df.to_csv(args.output, index=False)
    print(f"Forecast saved to {args.output}")
    print(f"Rows: {len(forecast_df):,}")
