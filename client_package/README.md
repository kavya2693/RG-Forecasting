# RG-Forecasting: Retail Demand Forecasting System

## Production-Ready Demand Forecasting for Retail

A complete, end-to-end demand forecasting solution for 33 stores and 3,650 SKUs.

**Key Metrics:**
- **Weekly Store WFA**: 88% (T1 Mature), 80% (T2 Growing)
- **Daily WFA**: 52-58% depending on tier and segment
- **168-day forecast horizon** (24 weeks)

---

## Architecture: V2 + Croston Hybrid

| Segment | Model | Description |
|---------|-------|-------------|
| **A-Items** | Two-Stage LightGBM | Top 80% sales volume |
| **B-Items** | Two-Stage LightGBM | Next 15% sales volume |
| **C-Items** | Croston's Method | Bottom 5% (sparse/intermittent) |

### Two-Stage LightGBM (A/B Items)
1. **Stage 1**: Binary classifier predicts P(sale > 0)
2. **Stage 2**: Log-transform regressor predicts E[log(qty) | sale > 0]
3. **Smearing correction** using Duan's method
4. **Per-segment calibration** for bias adjustment

### Croston's Method (C Items)
- Designed for intermittent demand patterns
- Separately estimates demand size and inter-arrival time
- Forecast = demand_size / inter_arrival_time
- Works well when 75%+ of observations are zeros

---

## Accuracy Results by Tier and Segment

### Daily WFA by ABC Segment

| Tier | A-Items | B-Items | C-Items | Overall |
|------|---------|---------|---------|---------|
| **T1 MATURE** | 58.5% | 40.1% | 15.4% | 51.9% |
| **T2 GROWING** | 53.6% | 32.6% | 14.3% | 45.8% |
| **T3 COLD START** | 49.4% | 38.9% | 18.2% | 44.0% |

### WFA by Aggregation Level

| Aggregation | T1 MATURE | T2 GROWING |
|-------------|-----------|------------|
| Daily SKU-Store | 51.5% | 45.6% |
| Weekly SKU-Store | 56.8% | 67.5% |
| Weekly Store | **87.9%** | **80.1%** |
| Weekly Total | **88.3%** | **80.1%** |

### Data Volume

| Tier | A-Items | B-Items | C-Items | Total | % |
|------|---------|---------|---------|-------|---|
| T1 MATURE | 188,630 | 269,906 | 644,172 | 1,102,708 | 15.6% |
| T2 GROWING | 910,560 | 1,240,680 | 3,484,824 | 5,636,064 | 80.0% |
| T3 COLD START | 47,096 | 71,232 | 190,456 | 308,784 | 4.4% |
| **Total** | **1,146,286** | **1,581,818** | **4,319,452** | **7,047,556** | 100% |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python src/train.py --train data/train.csv --val data/val.csv --output-dir models/
```

### 3. Generate Forecasts
```bash
python src/forecast.py \
    --model-dir models/ \
    --input data/future_features.csv \
    --output forecasts.csv
```

---

## Project Structure

```
client_package/
|-- README.md                 # This file
|-- requirements.txt          # Python dependencies
|-- src/
|   |-- train.py             # Model training (V2 + Croston)
|   |-- forecast.py          # Forecast generation
|   |-- croston.py           # Croston's method implementation
|-- models/
|   |-- model_A.pkl          # A-items two-stage model
|   |-- model_B.pkl          # B-items two-stage model
|   +-- model_C_croston.pkl  # C-items Croston model
|-- configs/
|   +-- config.yaml          # Hyperparameters
+-- docs/
    +-- METHODOLOGY.md       # Technical documentation
```

---

## Output Format

| Column | Type | Description |
|--------|------|-------------|
| store_id | int | Store identifier (1-33) |
| sku_id | int | SKU identifier |
| date | date | Forecast date |
| predicted_sales | float | Point forecast |
| abc_segment | str | A, B, or C |

### Example
```csv
store_id,sku_id,date,predicted_sales,abc_segment
1,1001,2025-12-18,12.5,A
1,2001,2025-12-18,0.8,C
```

---

## Model Configuration

### A-Items (High Volume)
```python
{
    'num_leaves': 1023,
    'learning_rate': 0.008,
    'n_estimators': 1500,
    'threshold': 0.45,
    'calibration': 1.15
}
```

### B-Items (Medium Volume)
```python
{
    'num_leaves': 255,
    'learning_rate': 0.015,
    'n_estimators': 800,
    'threshold': 0.50,
    'calibration': 1.10
}
```

### C-Items (Sparse/Intermittent)
```python
{
    'method': 'croston',
    'alpha_demand': 0.1,
    'alpha_interval': 0.1,
    'min_observations': 3
}
```

---

## Features (20 total)

| Category | Features |
|----------|----------|
| **Calendar** | dow, week_of_year, month, day_of_month, is_weekend, sin_doy, cos_doy |
| **Lag** | lag_1, lag_7, lag_14, lag_28 |
| **Rolling** | roll_mean_7, roll_sum_7, roll_mean_28, roll_std_28 |
| **Sparse** | nz_rate_7, nz_rate_28, days_since_last_sale |
| **Other** | dormancy_capped, is_store_closed |

---

## Tiering Strategy

| Tier | Criteria | Series Count | Holdout |
|------|----------|--------------|---------|
| T1 MATURE | 6+ years history | 65,000 | 168-day, 3-fold CV |
| T2 GROWING | 1-6 years history | 35,000 | 168-day, 2-fold CV |
| T3 COLD START | <90 days history | 14,000 | 28-day per-series |

---

## Key Decisions

1. **Why Two-Stage?** 75% of observations are zeros - classifier handles sparsity
2. **Why Croston for C-items?** Intermittent demand patterns need specialized method
3. **Why per-segment params?** A-items can handle complexity (1023 leaves), C-items need simplicity
4. **Why smearing?** Log-transform creates bias that Duan's method corrects

---

## What Didn't Work

| Approach | Result |
|----------|--------|
| XGBoost ensemble | No improvement over LightGBM |
| Per-store models | Insufficient data per store |
| Single model for all segments | 5pp worse than per-segment |
| Heavy holiday features | Minimal impact in UAE |

---

## Future Improvements

1. **Promotional data** (+10-15pp expected when available)
2. **Stock-out detection** (distinguish no-demand from no-stock)
3. **Event calendars** (Ramadan timing, local events)
4. **Hierarchical forecasting** (reconcile SKU-store-total)

---

## Requirements

- Python 3.8+
- pandas, numpy
- lightgbm
- scikit-learn
- scipy

---

## License

Proprietary - All rights reserved.
