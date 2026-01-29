"""
Spike-Based Feature Engineering
===============================
Inferred promotional/event signals from sales patterns.

Features:
- feat_store_promo_day: Is this a store-wide spike day?
- feat_seasonal_lift: Week-level seasonal multiplier
- feat_had_recent_spike: Was there a spike in last 7 days?
- feat_store_spike_pct: % of SKUs spiking in this store today
- feat_post_spike: Is this 7 days after a spike (expect dip)?
- feat_historical_spike_prob: Historical probability of spike for this series/week
"""

import pandas as pd
import numpy as np


def detect_spikes(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect spikes where actual sales > threshold × rolling mean.

    Args:
        df: DataFrame with 'y' (target) and 'roll_mean_28' columns
        threshold: Multiplier for spike detection (default 2.0)

    Returns:
        DataFrame with spike columns added
    """
    df = df.copy()
    target = 'y' if 'y' in df.columns else 'qty'

    # Baseline and spike ratio
    df['_baseline'] = df['roll_mean_28'].fillna(0) + 0.1
    df['_spike_ratio'] = df[target] / df['_baseline']
    df['is_spike'] = (df['_spike_ratio'] > threshold) & (df[target] > 1)

    return df


def classify_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify spikes into categories:
    - STORE_WIDE: >15% of SKUs in store spike on same day
    - SEASONAL: Spike occurs in high-season week
    - DOW_ALIGNED: Spike on weekend
    - ISOLATED: Individual item spike
    """
    df = df.copy()
    target = 'y' if 'y' in df.columns else 'qty'

    # Ensure date columns exist
    if 'year' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].dt.isocalendar().week.astype(int)

    # Store-wide spike detection
    store_day_stats = df.groupby(['store_id', 'date']).agg({
        'is_spike': 'sum',
        'sku_id': 'nunique'
    }).reset_index()
    store_day_stats.columns = ['store_id', 'date', 'spike_count', 'sku_count']
    store_day_stats['spike_pct'] = store_day_stats['spike_count'] / store_day_stats['sku_count']
    store_day_stats['is_store_wide'] = store_day_stats['spike_pct'] > 0.15

    df = df.merge(
        store_day_stats[['store_id', 'date', 'spike_pct', 'is_store_wide']],
        on=['store_id', 'date'],
        how='left'
    )

    # Seasonal detection (week-level lift)
    week_avg = df.groupby(['store_id', 'sku_id', 'week'])[target].mean().reset_index()
    week_avg.columns = ['store_id', 'sku_id', 'week', 'week_avg']
    overall_avg = df.groupby(['store_id', 'sku_id'])[target].mean().reset_index()
    overall_avg.columns = ['store_id', 'sku_id', 'overall_avg']

    week_avg = week_avg.merge(overall_avg, on=['store_id', 'sku_id'])
    week_avg['week_lift'] = week_avg['week_avg'] / (week_avg['overall_avg'] + 0.1)
    week_avg['is_high_season_week'] = week_avg['week_lift'] > 1.3

    df = df.merge(
        week_avg[['store_id', 'sku_id', 'week', 'week_lift', 'is_high_season_week']],
        on=['store_id', 'sku_id', 'week'],
        how='left'
    )

    return df


def create_spike_features(df: pd.DataFrame) -> tuple:
    """
    Create spike-based features for model training.

    Returns:
        (DataFrame with features, list of new feature names)
    """
    df = df.copy()
    target = 'y' if 'y' in df.columns else 'qty'

    # Feature 1: Store promo day indicator
    df['feat_store_promo_day'] = df['is_store_wide'].fillna(False).astype(int)

    # Feature 2: Seasonal lift
    df['feat_seasonal_lift'] = df['week_lift'].fillna(1.0).clip(0.5, 3.0)

    # Feature 3: Recent spike indicator
    df = df.sort_values(['store_id', 'sku_id', 'date'])
    df['feat_had_recent_spike'] = df.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()
    ).fillna(0).astype(int)

    # Feature 4: Store spike intensity
    df['feat_store_spike_pct'] = df['spike_pct'].fillna(0).clip(0, 1)

    # Feature 5: Post-spike indicator
    df['feat_post_spike'] = df.groupby(['store_id', 'sku_id'])['is_spike'].transform(
        lambda x: x.shift(7)
    ).fillna(0).astype(int)

    # Feature 6: Seasonal period flag
    df['feat_is_seasonal_period'] = df['is_high_season_week'].fillna(False).astype(int)

    # Feature 7: Historical spike probability
    spike_prob = df.groupby(['store_id', 'sku_id', 'week'])['is_spike'].mean().reset_index()
    spike_prob.columns = ['store_id', 'sku_id', 'week', 'feat_historical_spike_prob']
    df = df.merge(spike_prob, on=['store_id', 'sku_id', 'week'], how='left')
    df['feat_historical_spike_prob'] = df['feat_historical_spike_prob'].fillna(0).clip(0, 1)

    new_features = [
        'feat_store_promo_day',
        'feat_seasonal_lift',
        'feat_had_recent_spike',
        'feat_store_spike_pct',
        'feat_post_spike',
        'feat_is_seasonal_period',
        'feat_historical_spike_prob',
    ]

    # Clean up temp columns
    temp_cols = ['_baseline', '_spike_ratio', 'is_spike', 'spike_pct', 'is_store_wide',
                 'week_lift', 'is_high_season_week']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')

    return df, new_features


def add_spike_features(df: pd.DataFrame, threshold: float = 2.0) -> tuple:
    """
    Main entry point: Add all spike-based features to a DataFrame.

    Args:
        df: DataFrame with sales data
        threshold: Spike detection threshold (default 2.0)

    Returns:
        (DataFrame with features, list of new feature names)
    """
    df = detect_spikes(df, threshold)
    df = classify_spikes(df)
    df, new_features = create_spike_features(df)
    return df, new_features
