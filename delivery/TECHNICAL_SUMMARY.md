# Technical Summary: RG-Forecast

## The Problem

### Data Characteristics
- **75% sparse data**: Most daily observations are zeros (slow movers)
- **No promotional data**: Cannot predict demand spikes from promotions
- **7 years of history**: Jan 2019 to Dec 2025 (60.9M transaction rows)
- **Scale**: 33 stores x 3,650 SKUs = 114,501 unique series

### Business Requirements
- 168-day (24-week) forecast horizon for inventory planning
- Daily granularity at store-SKU level
- Must handle new products (cold start), slow movers, and fast movers

## The Solution

### Two-Stage LightGBM Model

Standard regression struggles with 75% zeros because it must simultaneously learn:
1. When will a sale occur?
2. How much will sell?

Our two-stage approach separates these problems:

```
Stage 1: Classifier (Binary)
    Input: Features
    Output: P(sale > 0)

Stage 2: Regressor (Continuous)
    Input: Features
    Output: log(quantity | sale > 0)

Final Prediction:
    E[sales] = P(sale) x exp(log_pred) x smearing_factor
```

### 18 Models Total

We train separate models for each combination of:
- **Tier** (T1, T2, T3) - based on history length
- **ABC Segment** (A, B, C) - based on sales volume
- **Stage** (Classifier, Regressor)

This allows appropriate regularization per segment:
- A-items: Complex models (255 leaves, 1000 rounds)
- C-items: Heavily regularized (31 leaves, 300 rounds, min_data_in_leaf=100)

## Architecture

### Medallion Pattern (Bronze-Silver-Gold)

```
Bronze Layer (Raw Data)
    |
    | 60.9M transaction rows
    | Stored as-is for audit trail
    |
    v
Silver Layer (Complete Panel)
    |
    | 134.9M rows after panel expansion
    | Explicit zeros for missing dates
    | Enables rolling window calculations
    |
    v
Gold Layer (Feature Engineering)
    |
    | 32 production features
    | All features are causal (no future leakage)
    |
    v
Model Training & Inference
```

### Data Tiering

| Tier | Criterion | Series | Volume | Regularization |
|------|-----------|--------|--------|----------------|
| T1 Mature | 6+ years history | 65,000 | 93% | Standard |
| T2 Growing | 1-6 years history | 35,000 | 7% | Moderate |
| T3 Cold Start | <90 days history | 14,000 | <1% | Heavy |

### ABC Segmentation

Within each tier:
- **A-Items**: Top products = 80% of sales
- **B-Items**: Next tier = 15% of sales
- **C-Items**: Long tail = 5% of sales

## Feature Engineering

### 52 Production Features (8 Categories)

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
| **Total** | **52** | + 2 categorical (store_id, sku_id) = **54** |

### Causality Guarantee

All features use only past information. Window functions end at "1 preceding":
```sql
AVG(sales) OVER (
    PARTITION BY store_id, item_id
    ORDER BY date
    ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
)
```

## Validation Strategy

### 3-Fold Cross-Validation (T1 Mature)

| Fold | Training Cutoff | Validation Period |
|------|-----------------|-------------------|
| F1 | June 2025 | Jul-Dec 2025 (168 days) |
| F2 | December 2024 | Jan-Jun 2025 (168 days) |
| F3 | June 2024 | Jul-Dec 2024 (168 days) |

**Validation horizon matches production (168 days)** - critical for realistic accuracy estimates.

### Cross-Fold Stability

| Fold | Accuracy |
|------|----------|
| F1 | 51.9% |
| F2 | 52.2% |
| F3 | 51.0% |
| **Std Dev** | **<1pp** |

Stability under 1 percentage point proves no overfitting.

## Results

### Accuracy by Level and Tier

| Level | T1 Mature | T2 Growing | T3 Cold Start |
|-------|-----------|------------|---------------|
| Daily SKU-Store (all) | 52% | 46% | 44% |
| Daily SKU-Store (A-Items) | 58% | 54% | 49% |
| Daily SKU-Store (B-Items) | 40% | 38% | 35% |
| Daily SKU-Store (C-Items) | 15% | 14% | 12% |
| Weekly SKU-Store | 57% | 52% | 48% |
| Weekly SKU-Store (A-Items) | 63% | 58% | 52% |
| **Weekly Store** | **88%** | **80%** | **60%** |

### Key Metric: Weekly Store = 88%

This is the actionable metric:
- Drives store-level replenishment decisions
- Errors cancel across thousands of SKUs
- Matches operational decision granularity

### Improvement Journey

| Change | Impact |
|--------|--------|
| Log transform on target | +4.6pp |
| ABC segmentation | +0.9pp |
| Spike features | +1.0pp |
| Two-stage model | +0.6pp |
| **Total improvement** | **~7pp over baseline** |

### What Failed

| Approach | Result | Reason |
|----------|--------|--------|
| Per-store models | 44% (-8pp) | Data fragmentation, overfitting |
| Holiday indicator flags | 36% (-16pp) | Only 6-7 examples per holiday, memorization |
| Local/Imported attribute | No change | Near-zero feature importance |

**Lesson**: Keep data pooled, use smooth features instead of sparse indicators.

## Improvement Scope

### High-Value Opportunities

1. **Promotional Data** (estimated +10-15pp daily)
   - Promotions are the primary driver of unpredictable spikes
   - Currently the single biggest limitation

2. **Stock-out Detection**
   - Cannot distinguish "no demand" from "no stock"
   - Would improve training data quality

3. **Event Calendars**
   - Ramadan, Eid, local events
   - Must be encoded smoothly (not binary indicators)

### Lower-Priority Improvements

4. **Temporal Fusion Transformer** for A-items
   - Deep learning for time series
   - Requires GPU infrastructure
   - Harder to debug and explain

5. **External Data**
   - Weather (limited impact in retail)
   - Economic indicators (too coarse)

## Production Deployment

### Infrastructure
- **Training**: Vertex AI (n2-standard-8)
- **Storage**: BigQuery (Bronze/Silver/Gold tables)
- **Models**: Cloud Storage (serialized LightGBM)
- **Dashboard**: Streamlit via Cloud Run

### Refresh Cadence
- Weekly model retraining with latest actuals
- Daily forecast generation (168-day rolling window)

### Output Format
```
| Column | Type | Description |
|--------|------|-------------|
| store_id | INT | Store identifier |
| item_id | INT | SKU identifier |
| date | DATE | Forecast date |
| predicted_sales | FLOAT | Point forecast |
| lower_bound | FLOAT | 10th percentile |
| upper_bound | FLOAT | 90th percentile |
```

## Summary

| Aspect | Decision |
|--------|----------|
| **Model** | Two-stage LightGBM (classifier + regressor) |
| **Architecture** | Bronze-Silver-Gold medallion |
| **Tiering** | T1/T2/T3 by history length |
| **Segmentation** | ABC by sales volume |
| **Features** | 32 causal features |
| **Validation** | 3-fold CV, 168-day horizon |
| **Best Accuracy** | 88% weekly store (T1) |
| **Limitation** | No promotional data |
