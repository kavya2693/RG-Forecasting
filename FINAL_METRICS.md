# RG-Forecasting V2 - Final Metrics Summary

## Training Completed: January 31, 2026

### Model Configuration
- **Architecture**: Two-Stage LightGBM (Classifier + Log-Transform Regressor)
- **Smearing Correction**: Duan's method for log-transform bias
- **Per-Segment Calibration**: A=1.20, B=1.15, C=1.10

### Features Used (34 total)
| Category | Count | Examples |
|----------|-------|----------|
| Calendar | 12 | dow, week_of_year, month, sin_doy, cos_doy, is_holiday |
| Lag | 5 | lag_1, lag_7, lag_14, lag_28, lag_56 |
| Rolling | 5 | roll_mean_7, roll_sum_7, roll_mean_28, roll_sum_28, roll_std_28 |
| Sparse | 5 | nz_rate_7, nz_rate_28, days_since_last_sale |
| Dormancy | 3 | dormancy_capped, last_sale_qty, zero_run_length |
| Closure | 4 | is_store_closed, days_to_next_closure |

### UAE Holiday Features
- is_uae_holiday (binary)
- days_to_holiday (capped at 30)
- Covers: Ramadan, Eid al-Fitr, Eid al-Adha, National Day, New Year

### Croston's Method for Sparse Items
- Applied to series with nz_rate < 5%
- Exponential smoothing on inter-arrival times and demand sizes

---

## Final Accuracy Results

### Overall Metrics (High-Volume Subset)

| Level | WFA | WMAPE | Bias |
|-------|-----|-------|------|
| **Daily SKU-Store** | **54.3%** | 45.7% | +5.1% |
| Weekly SKU-Store | 58.5% | 41.5% | +5.1% |
| **Weekly Store** | **81.0%** | 19.0% | +5.1% |
| Weekly Total | 92.7% | 7.3% | +5.1% |

### Per-Segment Performance

| Segment | Daily WFA | Weekly Store WFA | Bias |
|---------|-----------|------------------|------|
| A-Items | 54.0% | ~82% | +3.8% |
| B-Items | 58.1% | ~79% | +11.3% |
| C-Items | 47.5% | ~75% | +6.1% |

### Key Achievements
1. **Daily accuracy improved from 52% to 54.3%** (+2.3pp)
2. **Weekly store accuracy at 81%** for inventory planning
3. **Positive bias of 5%** - safer for inventory (slight over-forecast)
4. **Smearing factor ~1.1** indicates well-calibrated log-transform

---

## Model Hyperparameters

### A-Items (80% of sales volume)
```python
{
    'num_leaves': 255,
    'learning_rate': 0.015,
    'n_estimators': 800,
    'min_child_samples': 10,
    'reg_lambda': 0.1,
    'threshold': 0.45,
    'calibration': 1.20
}
```

### B-Items (15% of sales volume)
```python
{
    'num_leaves': 127,
    'learning_rate': 0.02,
    'n_estimators': 500,
    'min_child_samples': 15,
    'reg_lambda': 0.2,
    'threshold': 0.50,
    'calibration': 1.15
}
```

### C-Items (5% of sales volume)
```python
{
    'num_leaves': 63,
    'learning_rate': 0.03,
    'n_estimators': 300,
    'min_child_samples': 30,
    'reg_lambda': 0.3,
    'threshold': 0.55,
    'calibration': 1.10
}
```

---

## What's Working

1. **Two-Stage Model**: Separating P(sale) from E[qty|sale] handles 75% zeros effectively
2. **Log Transform**: Reduces outlier impact, smearing corrects bias
3. **ABC Segmentation**: Different complexity levels for different value items
4. **UAE Holidays**: Captures local demand patterns
5. **Closure Features**: Correctly predicts zero when store closed

## Remaining Limitations

1. **No Promotional Data**: Primary driver of unpredictable spikes
2. **Daily accuracy ~54%**: Inherent limit without promo signals
3. **B-Item bias +11%**: Over-forecasting on medium movers

## Files

| File | Purpose |
|------|---------|
| `src/vertex_training/train_v2_local.py` | Local training script |
| `src/vertex_training/train_v2_DEFINITIVE.py` | Full BigQuery training |
| `/tmp/v2_local/model_A.pkl` | Trained A-items model |
| `/tmp/v2_local/model_B.pkl` | Trained B-items model |
| `/tmp/v2_local/model_C.pkl` | Trained C-items model |
| `delivery/` | Client handover package |
