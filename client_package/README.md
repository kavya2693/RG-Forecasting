# RG-Forecasting: Retail Demand Forecasting System

## Production-Ready Demand Forecasting for Retail

A complete, end-to-end demand forecasting solution achieving **57.6% daily accuracy** and **83.6% weekly store accuracy**.

---

## Final Accuracy Results

| Segment | Daily WFA | Description |
|---------|-----------|-------------|
| **A-Items** | **57.3%** | Top 80% sales volume |
| **B-Items** | **61.6%** | Next 15% sales volume |
| **C-Items** | **50.3%** | Bottom 5% (sparse) |
| **Overall** | **57.6%** | Weighted across all |

| Aggregation Level | WFA |
|-------------------|-----|
| Daily SKU-Store | 57.6% |
| Weekly SKU-Store | 62.1% |
| **Weekly Store** | **83.6%** |
| Weekly Total | 94.2% |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python src/train.py --train data/train.csv --val data/val.csv
```

### 3. Generate Forecasts
```bash
python src/forecast.py --model-dir models/ --output forecasts.csv
```

### 4. Run Dashboard
```bash
streamlit run src/dashboard.py
```

---

## Project Structure

```
client_package/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/
│   ├── train.py             # Model training
│   ├── forecast.py          # Forecast generation
│   ├── evaluate.py          # Accuracy evaluation
│   ├── features.py          # Feature engineering
│   └── dashboard.py         # Streamlit dashboard
├── models/
│   ├── model_A.pkl          # A-items model
│   ├── model_B.pkl          # B-items model
│   └── model_C.pkl          # C-items model
├── configs/
│   └── config.yaml          # Hyperparameters
├── notebooks/
│   └── analysis.ipynb       # Exploration notebook
└── docs/
    ├── TECHNICAL.md         # Technical documentation
    └── METHODOLOGY.md       # Approach explanation
```

---

## Model Architecture

### Two-Stage LightGBM

```
Stage 1: Classifier
    Input: 34 features
    Output: P(sale > 0)

Stage 2: Log-Transform Regressor
    Input: 34 features (trained on positive sales only)
    Output: log(quantity)

Final Prediction:
    if P(sale) >= threshold:
        prediction = exp(log_pred) × smearing × calibration
    else:
        prediction = 0
```

### Per-Segment Optimization

| Segment | num_leaves | threshold | calibration | smearing |
|---------|------------|-----------|-------------|----------|
| A-items | 511 | 0.45 | 1.15 | 1.053 |
| B-items | 255 | 0.55 | 1.10 | 1.072 |
| C-items | 63 | 0.55 | 1.00 | 1.135 |

---

## Features (34 total)

| Category | Features |
|----------|----------|
| **Calendar** | dow, week_of_year, month, sin_doy, cos_doy, is_weekend, is_holiday |
| **Lag** | lag_1, lag_7, lag_14, lag_28, lag_56 |
| **Rolling** | roll_mean_7, roll_sum_7, roll_mean_28, roll_sum_28, roll_std_28 |
| **Sparse** | nz_rate_7, nz_rate_28, days_since_last_sale |
| **Closure** | is_store_closed, days_to_next_closure, days_from_prev_closure |
| **Dormancy** | dormancy_capped, last_sale_qty, zero_run_length |

### UAE Holiday Features
- is_holiday (Ramadan, Eid al-Fitr, Eid al-Adha, National Day)
- days_to_holiday (smoothed proximity feature)

---

## Zero Handling

### Problem: 75% of observations are zeros

### Solution: Two-Stage Model
1. **Classifier predicts P(sale > 0)** - learns when sales will occur
2. **Regressor predicts quantity** - only activated when P(sale) > threshold
3. **Threshold tuning** - segment-specific (A: 0.45, B: 0.55, C: 0.55)

### For Sparse C-Items: Croston's Method
- Exponential smoothing on inter-arrival times
- Exponential smoothing on demand sizes
- Forecast = demand_size / inter_arrival_time

---

## Data Requirements

### Input Format
| Column | Type | Description |
|--------|------|-------------|
| store_id | int | Store identifier |
| sku_id | int | SKU identifier |
| date | date | Transaction date |
| sales | float | Quantity sold |

### Output Format
| Column | Type | Description |
|--------|------|-------------|
| store_id | int | Store identifier |
| sku_id | int | SKU identifier |
| date | date | Forecast date |
| predicted_sales | float | Point forecast |

---

## Infrastructure

- **Training**: 16GB RAM minimum, GPU optional
- **Inference**: 8GB RAM sufficient
- **Storage**: ~500MB for models
- **Dashboard**: Streamlit (cloud-deployable)

---

## Improvement Roadmap

1. **Promotional Data** (+10-15pp expected)
2. **Stock-out Detection** (distinguish no-demand from no-stock)
3. **Event Calendars** (Ramadan timing, local events)
4. **Deep Learning** (Temporal Fusion Transformer for A-items)

---

## License

Proprietary - All rights reserved.

## Contact

For support, contact the development team.
