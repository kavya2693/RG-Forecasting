# RG-FORECASTING PIPELINE V2 - COMPLETE REFERENCE

## PROVEN BEST APPROACH (49.02% WMAPE)
- **C1+B1 Combined**: Two-stage model + log transform
- Stage 1: Binary classifier P(sale > 0)
- Stage 2: Log-transform regression E[qty | sale > 0]
- Smearing correction for log-bias
- Per-segment calibration factors

## 52 FEATURES (8 CATEGORIES)

| Category | Count | Features |
|----------|-------|----------|
| Calendar | 10 | dow, is_weekend, week_of_year, month, day_of_year, sin_doy, cos_doy, sin_dow, cos_dow, is_month_start, is_month_end |
| Lag | 5 | lag_1, lag_7, lag_14, lag_28, lag_56 |
| Rolling | 10 | roll_mean_7, roll_sum_7, roll_std_7, roll_mean_28, roll_sum_28, roll_std_28, roll_max_7, roll_max_28, roll_min_7, roll_min_28 |
| Sparse-Aware | 6 | nz_rate_7, nz_rate_28, roll_mean_pos_7, roll_mean_pos_28, days_since_last_sale, zero_run_length |
| Spike | 7 | is_recent_spike_7d, is_recent_spike_14d, spike_magnitude, store_spike_pct, feat_store_promo_day, feat_seasonal_lift, feat_historical_spike_prob |
| Hierarchy | 6 | store_avg_daily_sales, store_total_weekly, sku_avg_across_stores, sku_total_weekly, dow_store_effect, month_sku_effect |
| Dormancy | 4 | dormancy_capped, dormancy_bucket, last_sale_qty, days_to_reactivation_estimate |
| Closure | 4 | is_store_closed, is_closure_week, days_to_next_closure, days_from_prev_closure |
| **TOTAL** | **52** | + 2 categorical (store_id, sku_id) = **54** |

## OPTIMIZED HYPERPARAMETERS

| Segment | num_leaves | lr | n_estimators | min_child | reg_lambda | threshold | calibration |
|---------|------------|-----|--------------|-----------|------------|-----------|-------------|
| A-items | 511 | 0.012 | 1200 | 5 | 0.05 | 0.45 | 1.25 |
| B-items | 127 | 0.02 | 400-500 | 15 | 0.2 | 0.50 | 1.15 |
| C-items | 63 | 0.03 | 200-300 | 30 | 0.3 | 0.55 | 1.10 |

## CROSS-VALIDATION FOLDS

| Fold | Tier | Val Start | Val End | Train End |
|------|------|-----------|---------|-----------|
| F1 | T1_MATURE | 2025-06-03 | 2025-12-17 | 2025-06-02 |
| F2 | T1_MATURE | 2024-12-05 | 2025-06-02 | 2024-12-04 |
| F3 | T1_MATURE | 2024-06-08 | 2024-12-04 | 2024-06-07 |
| G1 | T2_GROWING | 2025-06-03 | 2025-12-17 | 2025-06-02 |
| G2 | T2_GROWING | 2024-12-05 | 2025-06-02 | 2024-12-04 |
| C1 | T3_COLD_START | Per-series last 28 days | | |

## ABC SEGMENTATION
- **A**: Top 80% of sales volume
- **B**: Next 15%
- **C**: Bottom 5%

## TIERING CRITERIA

| Tier | History Days | NZ Rate | Features |
|------|--------------|---------|----------|
| T1_MATURE | >= 365 | >= 10% | 54 |
| T2_GROWING | 90-364 | >= 5% | 37 |
| T3_COLD_START | 28-89 | < 5% | 23 |

## DATA CLEANING
- Negatives clipped to 0
- COVID downweight (2020-03-15 to 2021-06-30): weight=0.25
- Extreme spikes flagged (>10,000)
- Store closure override: yhat=0 when closed

## CURRENT ACCURACY

| Tier | Daily WFA | Weekly Store WFA | Weekly Total WFA |
|------|-----------|------------------|------------------|
| T1_MATURE | 51.5% | 87.9% | 88.3% |
| T2_GROWING | 45.6% | 80.1% | 80.1% |
| T3_COLD_START | 44.0% | 60.0% | 59.8% |

## KEY FILES
- `src/vertex_training/train_v2_DEFINITIVE.py`: Complete training with all improvements
- `src/vertex_training/submit_v2_definitive.py`: Vertex AI submission
- `params/pipeline_params.yaml`: Full configuration
- `IMPROVEMENT_RESULTS.md`: Experiment results

## RUN TRAINING

```bash
# Local test (sampled)
python src/vertex_training/train_v2_DEFINITIVE.py --sample-frac 0.01 --no-gcs

# Full training on Vertex AI
python src/vertex_training/submit_v2_definitive.py --tiers all
```
