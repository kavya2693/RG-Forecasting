#!/usr/bin/env python3
"""
RG-Forecasting V2 + Croston Hybrid
==================================
V2 Two-Stage LightGBM for A/B items + Croston for C items only.

This is the production training script:
- A-items: Two-Stage LightGBM (classifier + regressor)
- B-items: Two-Stage LightGBM (classifier + regressor)
- C-items: Croston's method for intermittent demand

Usage:
    python src/train_v2_croston.py --train train.csv --val val.csv
"""

import os
import sys
import json
import pickle
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - V2 BEST HYPERPARAMETERS
# =============================================================================

SEGMENT_PARAMS = {
    'A': {
        'classifier': {
            'objective': 'binary',
            'num_leaves': 1023,
            'learning_rate': 0.008,
            'n_estimators': 1500,
            'min_child_samples': 3,
            'reg_lambda': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 3,
            'verbose': -1,
        },
        'regressor': {
            'objective': 'regression',
            'num_leaves': 1023,
            'learning_rate': 0.008,
            'n_estimators': 1500,
            'min_child_samples': 3,
            'reg_lambda': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 3,
            'verbose': -1,
        },
        'threshold': 0.45,
        'calibration': 1.15,
    },
    'B': {
        'classifier': {
            'objective': 'binary',
            'num_leaves': 255,
            'learning_rate': 0.015,
            'n_estimators': 800,
            'min_child_samples': 10,
            'reg_lambda': 0.1,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'verbose': -1,
        },
        'regressor': {
            'objective': 'regression',
            'num_leaves': 255,
            'learning_rate': 0.015,
            'n_estimators': 800,
            'min_child_samples': 10,
            'reg_lambda': 0.1,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'verbose': -1,
        },
        'threshold': 0.50,
        'calibration': 1.10,
    },
}

# Croston parameters for C-items
CROSTON_PARAMS = {
    'alpha_demand': 0.1,
    'alpha_interval': 0.1,
    'min_nz_count': 3,
}

# Features to use
FEATURES = [
    'dow', 'week_of_year', 'month', 'day_of_month',
    'is_weekend', 'sin_doy', 'cos_doy',
    'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_std_28',
    'nz_rate_7', 'nz_rate_28', 'days_since_last_sale',
    'dormancy_capped', 'is_store_closed',
]


def setup_logging(output_dir: str):
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# CROSTON'S METHOD FOR C-ITEMS
# =============================================================================

class CrostonModel:
    """Croston's method for intermittent demand forecasting."""

    def __init__(self, alpha_demand: float = 0.1, alpha_interval: float = 0.1):
        self.alpha_demand = alpha_demand
        self.alpha_interval = alpha_interval
        self.series_params = {}  # store-sku -> (demand_est, interval_est)

    def fit(self, df: pd.DataFrame):
        """Fit Croston parameters per store-sku series."""
        for (store_id, sku_id), group in df.groupby(['store_id', 'sku_id']):
            sales = group.sort_values('date')['y'].values

            # Find non-zero demands
            nz_idx = np.where(sales > 0)[0]

            if len(nz_idx) < CROSTON_PARAMS['min_nz_count']:
                # Too few demands - use simple average
                self.series_params[(store_id, sku_id)] = (np.mean(sales), len(sales))
                continue

            # Extract demand sizes and intervals
            demand_sizes = sales[nz_idx]
            intervals = np.diff(nz_idx)

            if len(intervals) == 0:
                self.series_params[(store_id, sku_id)] = (demand_sizes[0], 1.0)
                continue

            # Initialize
            z = demand_sizes[0]
            p = intervals[0] if len(intervals) > 0 else 1.0

            # Exponential smoothing
            for i in range(1, len(demand_sizes)):
                z = self.alpha_demand * demand_sizes[i] + (1 - self.alpha_demand) * z
                if i <= len(intervals):
                    p = self.alpha_interval * intervals[i-1] + (1 - self.alpha_interval) * p

            self.series_params[(store_id, sku_id)] = (z, max(p, 1.0))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using Croston rate = demand / interval."""
        preds = np.zeros(len(df))

        for i, row in df.iterrows():
            key = (row['store_id'], row['sku_id'])
            if key in self.series_params:
                z, p = self.series_params[key]
                preds[i] = z / p if p > 0 else 0
            else:
                preds[i] = 0

        return preds

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# TWO-STAGE MODEL FOR A/B ITEMS
# =============================================================================

class TwoStageModel:
    """Two-stage LightGBM: classifier + regressor with smearing."""

    def __init__(self, segment: str, logger=None):
        self.segment = segment
        self.logger = logger
        self.params = SEGMENT_PARAMS[segment]
        self.classifier = None
        self.regressor = None
        self.smear_factor = 1.0
        self.features = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, features: List[str], weights: np.ndarray = None):
        self.features = features

        # Stage 1: Binary classifier
        y_binary = (y > 0).astype(int)

        if self.logger:
            self.logger.info(f"  Training {self.segment} classifier: {len(X):,} samples, {y_binary.sum():,} positive")

        self.classifier = lgb.LGBMClassifier(**self.params['classifier'])
        self.classifier.fit(X[features], y_binary, sample_weight=weights)

        # Stage 2: Log-transform regressor on positive samples
        pos_mask = y > 0
        X_pos = X[pos_mask]
        y_pos = np.log1p(y[pos_mask])
        w_pos = weights[pos_mask] if weights is not None else None

        if self.logger:
            self.logger.info(f"  Training {self.segment} regressor: {len(X_pos):,} samples")

        self.regressor = lgb.LGBMRegressor(**self.params['regressor'])
        self.regressor.fit(X_pos[features], y_pos, sample_weight=w_pos)

        # Compute smearing factor (Duan's method)
        y_pred_log = self.regressor.predict(X_pos[features])
        residuals = y_pos - y_pred_log
        self.smear_factor = np.mean(np.exp(residuals))

        if self.logger:
            self.logger.info(f"  Smearing factor: {self.smear_factor:.4f}")

        return self

    def predict(self, X: pd.DataFrame):
        p_sale = self.classifier.predict_proba(X[self.features])[:, 1]
        log_qty = self.regressor.predict(X[self.features])

        # Apply threshold
        threshold = self.params['threshold']
        calibration = self.params['calibration']

        y_pred = np.where(
            p_sale >= threshold,
            np.expm1(log_qty) * self.smear_factor * calibration,
            0
        )

        return np.maximum(y_pred, 0), p_sale

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    total = np.sum(y_true)
    if total == 0:
        return {'wmape': np.nan, 'wfa': np.nan, 'bias_pct': np.nan, 'n': len(y_true)}

    wmape = np.sum(np.abs(y_true - y_pred)) / total * 100
    bias_pct = (np.sum(y_pred) - total) / total * 100

    return {
        'wmape': wmape,
        'wfa': 100 - wmape,
        'bias_pct': bias_pct,
        'n': len(y_true)
    }


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train(
    train_path: str,
    val_path: str,
    output_dir: str = 'models',
) -> Dict:
    """Train V2 + Croston hybrid model."""

    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("V2 + CROSTON HYBRID TRAINING")
    logger.info("=" * 60)
    logger.info("A/B items: Two-Stage LightGBM")
    logger.info("C items: Croston's method")

    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    logger.info(f"Train: {len(train_df):,} rows")
    logger.info(f"Val: {len(val_df):,} rows")

    # Get available features
    features = [f for f in FEATURES if f in train_df.columns]
    logger.info(f"Features: {len(features)}")

    # Sample weights (optional COVID downweight)
    weights = np.ones(len(train_df))
    if 'is_covid_period' in train_df.columns:
        weights = np.where(train_df['is_covid_period'] == 1, 0.25, 1.0)

    models = {}
    val_df['y_pred'] = 0.0

    # Train A and B with Two-Stage LightGBM
    for seg in ['A', 'B']:
        logger.info(f"\n--- Segment {seg}: Two-Stage LightGBM ---")

        mask_t = train_df['abc'] == seg
        mask_v = val_df['abc'] == seg

        if mask_t.sum() < 100:
            logger.warning(f"Skipping {seg}: only {mask_t.sum()} samples")
            continue

        model = TwoStageModel(seg, logger)
        model.fit(train_df[mask_t], train_df.loc[mask_t, 'y'].values, features, weights[mask_t])
        models[seg] = model
        model.save(os.path.join(output_dir, f'model_{seg}.pkl'))

        # Predict on validation
        y_pred, _ = model.predict(val_df[mask_v])
        val_df.loc[mask_v, 'y_pred'] = y_pred

    # Train C with Croston
    logger.info("\n--- Segment C: Croston's Method ---")
    mask_t = train_df['abc'] == 'C'
    mask_v = val_df['abc'] == 'C'

    if mask_t.sum() >= 100:
        croston = CrostonModel(
            alpha_demand=CROSTON_PARAMS['alpha_demand'],
            alpha_interval=CROSTON_PARAMS['alpha_interval']
        )
        croston.fit(train_df[mask_t])
        models['C'] = croston
        croston.save(os.path.join(output_dir, 'model_C_croston.pkl'))

        # Predict on validation
        y_pred = croston.predict(val_df[mask_v])
        val_df.loc[mask_v, 'y_pred'] = y_pred

        logger.info(f"  Fitted {len(croston.series_params):,} series")

    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    # Overall
    overall = calc_metrics(val_df['y'].values, val_df['y_pred'].values)
    logger.info(f"Overall Daily WFA: {overall['wfa']:.1f}%")

    # Per segment
    segment_metrics = {}
    for seg in ['A', 'B', 'C']:
        mask = val_df['abc'] == seg
        if mask.sum() > 0:
            m = calc_metrics(val_df.loc[mask, 'y'].values, val_df.loc[mask, 'y_pred'].values)
            segment_metrics[seg] = m
            logger.info(f"Segment {seg}: WFA={m['wfa']:.1f}%, Bias={m['bias_pct']:.1f}%, n={m['n']:,}")

    # Save results
    results = {
        'overall': overall,
        'segments': segment_metrics,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nModels saved to: {output_dir}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='V2 + Croston Hybrid Training')
    parser.add_argument('--train', required=True, help='Training CSV path')
    parser.add_argument('--val', required=True, help='Validation CSV path')
    parser.add_argument('--output-dir', default='models', help='Output directory')

    args = parser.parse_args()

    results = train(args.train, args.val, args.output_dir)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Overall WFA: {results['overall']['wfa']:.1f}%")
