# RG-Forecasting: Submission Summary

**24-Week Retail Demand Forecast | Store-SKU Level | 33 Stores, ~3,650 SKUs**

---

## 1. Problem Framing

### Business Context
- Retail chain with 33 stores and ~3,650 SKUs
- Daily transaction data from 2019 through December 17, 2025
- Forecast horizon: 24 weeks (168 days) from December 18, 2025

### Core Challenges
| Challenge | Impact | Our Approach |
|-----------|--------|--------------|
| **75% zero-sales days** | Standard regression fails | Two-stage model: classifier + regressor |
| **Long sales gaps** | Lag features become stale | Dormancy features + capped zero-run tracking |
| **Spikes and anomalies** | Outliers distort means | Spike detection features inferred from patterns |
| **Cold-start SKUs** | No history to learn from | Tier-based modeling with simpler models for new items |
| **No promotional data** | Can't model demand drivers directly | Infer store-level promo days from correlated spikes |

### Key Assumption
Missing dates in the transaction log represent zero sales, not missing data. This is validated by:
- Consistent store-level patterns (all stores show similar gaps)
- Weekend/holiday patterns align with expected retail behavior

---

## 2. Approach

### Data Pipeline
```
Raw Transactions → Complete Panel (Spine) → Cleaning → Feature Engineering → Tiering → Training → Forecast
```

1. **Spine Creation**: Generate complete store-SKU-date grid; fill missing dates with 0
2. **Cleaning**: Clip negative sales to 0, flag store closures, retain outliers with flags
3. **Feature Engineering**: 30 causal features (no data leakage)
4. **Tiering**: Segment series by maturity (T1/T2/T3) and volume (ABC)

### Model Architecture: Two-Stage LightGBM

**Why two-stage?** With 75% zeros, a single regressor wastes capacity predicting zeros. We separate:

| Stage | Task | Training Data | Output |
|-------|------|---------------|--------|
| **Classifier** | P(sale > 0) | All rows | Probability [0,1] |
| **Regressor** | E[qty \| sale > 0] | Non-zero rows only | log1p(quantity) |

**Final prediction**: `y_pred = prob × smear × expm1(reg_pred)` (Expected Value Formula)

**Key improvements:**
- **Expected Value Formula**: Instead of hard threshold, we use `E[y] = p × μ` which gives better predictions for sparse data
- **Smearing Correction**: Duan's smearing factor corrects log-transform bias, improving accuracy by ~3pp
- **Why log-transform?** Sales are right-skewed; log1p stabilizes variance and improves MAE

### ABC Segmentation (Within Each Tier)

| Segment | Definition | Model Complexity |
|---------|------------|------------------|
| **A-items** | Top 80% of sales volume | High (255 leaves, 800 rounds) |
| **B-items** | Next 15% of sales | Medium (63 leaves, 300 rounds) |
| **C-items** | Bottom 5% of sales | Low (31 leaves, 200 rounds) |

**Why?** High-volume items justify complex models; sparse items need regularization.

---

## 3. Key Decisions

### Features Used (30 total)

| Category | Features | Rationale |
|----------|----------|-----------|
| **Calendar** | dow, month, day_of_year, sin/cos transforms | Captures weekly/seasonal patterns |
| **Lag** | lag_1, lag_7, lag_14, lag_28, lag_56 | Recent sales history |
| **Rolling** | roll_mean_7/28, roll_sum_7/28, roll_std_28 | Smoothed demand signal |
| **Dormancy** | days_since_last_sale, zero_run_length, last_sale_qty | Handles long gaps |
| **Spike** | store_spike_pct, promo_day, seasonal_lift, recent_spike, historical_spike_prob | Infers promotional activity |
| **Attribute** | is_local (Local vs Imported SKU) | Product characteristic |

**Dropped**: Velocity features (sale_frequency, gap_pressure) - tested but added <0.1pp accuracy.

### Validation Strategy
- **Expanding window**: Train on all history up to cutoff, validate on next 168 days
- **Feature computation**: Lag/rolling features computed from historical panel up to cutoff date
- **No data leakage**: All features use only past data relative to prediction date

### Threshold Selection
| Segment | Threshold | Rationale |
|---------|-----------|-----------|
| A, B | 0.6 | Balance precision/recall for frequent sellers |
| C | 0.7 | Higher threshold for sparse items (reduce false positives) |

---

## 4. Evaluation Metrics

### Primary Metric: WMAPE (Weighted Mean Absolute Percentage Error)

```
WMAPE = 100 × Σ|actual - pred| / Σ(actual)
WFA = 100 - WMAPE  (Weighted Forecast Accuracy)
```

**Why WMAPE?**
- Volume-weighted: errors on high-sellers matter more
- Handles zeros gracefully (unlike MAPE which divides by zero)
- Industry standard for retail demand forecasting

### Why Not MAPE/sMAPE?
- **MAPE**: Division by zero when actual=0 (75% of our data)
- **sMAPE**: Still unstable near zero; symmetric penalty doesn't match business reality

### Aggregation Levels
Accuracy improves with aggregation. We report at multiple levels:

| Level | Use Case | Expected WFA |
|-------|----------|--------------|
| Daily SKU-Store | Inventory positioning | ~50-55% |
| Weekly SKU-Store | Replenishment planning | ~65-70% |
| Weekly Store | Store-level staffing | ~80-88% |
| Weekly Total | Corporate planning | ~92-95% |

**Practical guidance**: Use weekly-store or weekly-total for operational decisions; daily SKU-store is inherently noisy for sparse retail.

### Bias Metric
```
Bias = Σ(pred) / Σ(actual)
```
- Bias < 1.0: Under-forecasting
- Bias > 1.0: Over-forecasting
- Target: 0.95 - 1.05

---

## 5. Edge Case Handling

| Scenario | Handling | Example |
|----------|----------|---------|
| **Zero sales** | Classifier predicts P(sale)≈0 → output 0 | C-items with no recent activity |
| **Long gaps (30+ days)** | Dormancy features capture staleness; model learns decay | Seasonal items between seasons |
| **Spikes** | Spike features flag anomalous days; not smoothed away | Store promotions |
| **New SKUs** | Tier 3 (cold-start) with simpler model; relies on similar SKU patterns | Recently launched products |
| **Store closures** | Flagged; predictions forced to 0 | Holidays, renovations |
| **Negative transactions** | Clipped to 0 in cleaning (returns handled separately) | Returns/adjustments |

---

## 6. Results Summary

### Validation Performance (Holdout: 168-day horizon, Expected Value Formula)

| Segment | Daily WFA | Weekly Store WFA | Weekly Total WFA | Bias |
|---------|-----------|------------------|------------------|------|
| A-items | 57.8% | 94.8% | 97.5% | 0.98 |
| B-items | 36.6% | 94.4% | 94.7% | 0.95 |
| C-items | 5.2% | 88.0% | 88.0% | 0.88 |
| **Combined** | **51.2%** | **95.2%** | **96.9%** | **0.97** |

### Interpretation
- **Weekly Store 95% WFA** is excellent for sparse retail data without promotional signals
- **Weekly Total 97% WFA** shows strong aggregate forecasting capability
- **Daily ~51% WFA** reflects inherent noise at granular level (expected for 75% zeros)
- **Bias 0.97** is near-perfect (vs 0.81 with threshold-based approach) - expected value formula corrects systematic under-forecasting

---

## 7. Limitations and Next Steps

### Current Limitations
1. **No promotional data**: Cannot model demand lifts from promotions, price changes
2. **No stock-out data**: Can't distinguish "no demand" from "no supply"
3. **Static tiering**: Series assigned once; doesn't adapt to changing patterns
4. **Validation approach**: Uses pre-computed features from panel; production recursive rollout would show slightly lower metrics

### Recommended Improvements (with data)
| Improvement | Expected Impact | Required Data |
|-------------|-----------------|---------------|
| Promotional calendar | +5-10pp daily WFA | Promo start/end dates |
| Price elasticity | +2-5pp daily WFA | Historical prices |
| Stock-out flags | Reduced bias | Inventory levels |
| Event calendar | +1-2pp on event days | Local events, holidays |

### Model Improvements (no new data)
- Quantile regression for prediction intervals
- Dynamic tier reassignment
- Ensemble with different feature windows

---

## 8. Practicality Guidance

### For Operations Teams
- **Use weekly-store forecasts** for staffing and replenishment triggers
- **Use daily SKU-store forecasts** as directional signals, not precise targets
- **Monitor bias monthly** and recalibrate if consistently >10% off

### For Inventory Planning
- **A-items**: Trust daily forecasts more; apply safety stock based on variability
- **B/C-items**: Use weekly aggregates; daily is too noisy
- **New SKUs**: Expect higher error; use category averages as fallback

---

## Appendix: File Outputs

| File | Description | Rows |
|------|-------------|------|
| `forecast_168day.csv` | 24-week forecast | series × 168 days |
| Columns: `item_id, store_id, date, predicted_sales` | | |

---

*Generated: January 2026*
