# RG-Forecasting: Methodology Documentation

## Complete Justification of All Assumptions, Thresholds, and Classifications

---

## 1. ABC Segmentation

### Thresholds
| Segment | Sales Share | Justification |
|---------|-------------|---------------|
| A-items | Top 80% of cumulative sales | Industry standard Pareto principle - top performers drive most revenue |
| B-items | Next 15% (80-95%) | Middle tier with moderate volume |
| C-items | Bottom 5% (95-100%) | Long-tail items with sparse sales |

### Why ABC Segmentation?
1. **Different data densities**: A-items have ~34% non-zero rate vs C-items ~15%
2. **Optimal hyperparameters differ**: A-items need complex models (255 leaves), C-items need regularization (31 leaves)
3. **Error impact varies**: A-items errors matter more for inventory planning
4. **Research support**: "Machine learning methods effectively manage demand volatility when segmented by sales velocity" - Journal of Forecasting, 2024

---

## 2. Spike Detection

### Threshold: 2.0× Rolling Mean

**Definition**: A spike is detected when:
```
actual_sales > 2.0 × roll_mean_28 AND actual_sales > 1
```

**Justification**:
- **Why 2.0×?** Captures significant deviations while avoiding noise. Tested range 1.5-3.0:
  - 1.5×: Too sensitive, captures 15%+ of data (too many false positives)
  - 2.0×: Captures ~8% of data (meaningful anomalies)
  - 3.0×: Too strict, misses moderate promotions
- **Why roll_mean_28?** 28-day window captures monthly patterns while being robust to weekly variation
- **Why min 1 unit?** Excludes zero-to-small transitions which aren't meaningful spikes

### Spike Classification Thresholds

| Classification | Threshold | Justification |
|----------------|-----------|---------------|
| STORE_PROMO | >15% of SKUs spike | If >15% of a store's products spike on same day, it's likely a store-wide event (promotion, holiday, special event). Tested 10-25%, 15% balances sensitivity vs specificity |
| SEASONAL | Week lift > 1.3× | Week's average sales > 30% above series average indicates seasonal pattern. Based on December lift (~35%) as benchmark |
| DOW_PATTERN | Spike + is_weekend | Weekend spikes capture regular traffic patterns |
| ISOLATED | None of above | Item-specific events (item promo, viral, stockout recovery) |

### Evidence for Thresholds
From our data analysis:
- STORE_PROMO events: 550 unique store-days detected, avg magnitude **25 units** (highest)
- SEASONAL: Weeks 25-47 identified (summer through fall season)
- Store-wide spikes account for 21% of all spikes but have 2× the magnitude of isolated spikes

---

## 3. Inferred Promotional Features

### Feature: `feat_store_promo_day`
- **Definition**: Binary flag indicating store-wide promotional event
- **Calculation**: 1 if >15% of SKUs in store spike on this day
- **Justification**: Store-wide events (sales, holidays) affect all SKUs. This captures "rising tide lifts all boats" effect
- **Impact**: +1936 feature importance (2nd highest among new features)

### Feature: `feat_seasonal_lift`
- **Definition**: Week-level seasonal multiplier for this series
- **Calculation**: `avg_sales_this_week / avg_sales_all_weeks` (clipped 0.5-3.0)
- **Justification**: Captures recurring yearly patterns (Ramadan, summer, back-to-school)
- **Impact**: +1714 feature importance

### Feature: `feat_had_recent_spike`
- **Definition**: Was there a spike in the last 7 days for this series?
- **Calculation**: Rolling max of spike flag over 7-day window (shifted by 1)
- **Justification**: Spike events have aftereffects (restocking, customer awareness)
- **Impact**: +164 feature importance

### Feature: `feat_store_spike_pct`
- **Definition**: What % of SKUs are spiking in this store today?
- **Calculation**: Count of spiking SKUs / Total SKUs in store
- **Justification**: Higher percentages indicate store-wide events vs isolated item events
- **Impact**: +1936 feature importance (highest among new features)

### Feature: `feat_post_spike`
- **Definition**: Is this 7 days after a spike?
- **Calculation**: Spike flag shifted by 7 days
- **Justification**: Post-promotional dips are documented in literature - consumers stockpile during promotions
- **Research**: "The postpromotional phase shows demand decreases due to consumer stockpiling" - Journal of Forecasting, Hewage 2024
- **Impact**: +56 feature importance

### Feature: `feat_is_seasonal_period`
- **Definition**: Is this week historically high-sales?
- **Calculation**: 1 if week's average > 1.3× series average
- **Justification**: Binary flag for model to distinguish seasonal vs non-seasonal periods
- **Impact**: +0 feature importance (redundant with feat_seasonal_lift)

### Feature: `feat_historical_spike_prob`
- **Definition**: Historical probability of spike for this series in this week
- **Calculation**: Mean of spike flag grouped by (store, sku, week)
- **Justification**: Some series have predictable promotional calendars
- **Impact**: +809 feature importance

---

## 4. Model Hyperparameters

### Classifier Thresholds (Binary Classification)

| Segment | Threshold | Justification |
|---------|-----------|---------------|
| A-items | 0.6 | Lower threshold → more non-zero predictions. A-items have higher non-zero rate (~34%), so we can be less conservative |
| B-items | 0.6 | Same reasoning as A-items |
| C-items | 0.7 | Higher threshold → fewer non-zero predictions. C-items are sparse (~15% non-zero), so we need to be more conservative to avoid over-prediction |

### LightGBM Hyperparameters by Segment

| Parameter | A-items | B-items | C-items | Justification |
|-----------|---------|---------|---------|---------------|
| num_leaves | 255 | 63 | 31 | More complex models for data-rich segments |
| min_child_samples | 10 | 50 | 100 | More regularization for sparse segments |
| learning_rate | 0.015 | 0.03 | 0.05 | Slower learning for A-items (more boosting rounds) |
| n_estimators (clf) | 800 | 300 | 200 | More rounds for A-items |
| n_estimators (reg) | 1000 | 400 | 300 | Same reasoning |

**Why these values?**
- Grid search over: num_leaves ∈ {15, 31, 63, 127, 255}, min_child ∈ {5, 10, 20, 50, 100, 200}
- Cross-validated on 3 folds (F1, F2, F3)
- A-items: More data per series supports complex models
- C-items: Limited data requires strong regularization to prevent overfitting

---

## 5. Tiering Strategy

### Tier Definitions

| Tier | Criteria | % of Sales | Justification |
|------|----------|------------|---------------|
| T1_MATURE | >180 days history, >52 weeks data | 93% | Long history enables robust lag features |
| T2_GROWING | 90-180 days history | 7% | Moderate history, need shorter lags |
| T3_COLD_START | <90 days history | <1% | Insufficient history, use item/store averages |

### Why 180/90 Day Cutoffs?
- **180 days**: Allows 28-day and 56-day lag features to be populated for most dates
- **90 days**: Minimum for 7-day and 14-day lags to be meaningful
- Based on feature importance: lag_7, lag_14, lag_28 are in top 20 features

---

## 6. Feature Engineering Decisions

### Lag Features
| Feature | Window | Justification |
|---------|--------|---------------|
| lag_1 | t-1 | Yesterday's sales (most recent signal) |
| lag_7 | t-7 | Same day last week (weekly pattern) |
| lag_14 | t-14 | Two weeks ago |
| lag_28 | t-28 | Same day ~4 weeks ago (monthly pattern) |
| lag_56 | t-56 | Same day ~8 weeks ago |

### Rolling Statistics
| Feature | Window | Justification |
|---------|--------|---------------|
| roll_mean_7 | 7-day | Short-term trend |
| roll_mean_28 | 28-day | Medium-term baseline |
| roll_mean_pos_28 | 28-day, non-zero only | Baseline excluding zeros (key for sparse data) |
| roll_std_28 | 28-day | Volatility measure |
| nz_rate_7/28 | 7/28-day | Non-zero rate (sparsity indicator) |

### Why roll_mean_pos_28 is Important
- In 75% zero data, regular mean is heavily biased toward zero
- roll_mean_pos_28 captures "when it sells, how much does it sell?"
- **Feature importance**: #3 overall (after sku_id and roll_mean_7)

---

## 7. Evaluation Metrics

### Why WMAPE (Weighted Mean Absolute Percentage Error)?
```
WMAPE = Σ|actual - predicted| / Σ(actual)
```

**Justification**:
1. **Handles zeros**: Unlike MAPE, doesn't divide by actuals
2. **Sales-weighted**: High-volume items contribute more to error
3. **Interpretable**: "What % of total sales did we mis-predict?"
4. **Industry standard**: Used by Amazon, Walmart, and major retailers

### Why WFA (Weighted Forecast Accuracy)?
```
WFA = 100 - WMAPE
```

**Justification**:
- More intuitive: "88% accurate" is easier to communicate than "12% error"
- Positive framing for stakeholder reporting

### Why Multiple Aggregation Levels?
| Level | Use Case |
|-------|----------|
| Daily SKU-Store | Operational: daily replenishment decisions |
| Weekly SKU-Store | Tactical: weekly ordering |
| Weekly Store | Store manager planning |
| Weekly Total | Executive dashboard, budget planning |

**Key insight**: Errors cancel out at higher aggregations, so weekly store accuracy (88%) is much higher than daily (52%).

---

## 8. Data Quality Decisions

### Negative Sales Handling
- **Decision**: Clip to 0
- **Justification**: Negative values are returns, not demand. For demand forecasting, we predict gross sales, not net.

### Store Closure Handling
- **Decision**: Force predictions to 0 on closure days
- **Justification**: No sales possible when store is closed
- **Implementation**: `y_pred[is_store_closed == 1] = 0`

### Outlier Handling
- **Decision**: Retain outliers (no capping)
- **Justification**: Spikes may be real promotional events. Capping would remove valuable signal.
- **Alternative considered**: Log-transform (used in regressor) naturally compresses outliers

---

## 9. Spike Feature Validation

### Why Spike Features Work for A/B but Not C

| Segment | Spike Feature Impact | Reason |
|---------|---------------------|--------|
| A-items | +11pp daily, +12pp weekly | Enough data to learn spike patterns |
| B-items | +6pp daily, +1pp weekly | Moderate impact |
| C-items | -5pp daily, +7pp weekly | Too sparse at item level; only helps aggregated |

**Recommendation**: Apply spike features only to A and B segments for daily-level forecasting.

---

## 10. Limitations and Caveats

1. **Spike inference is backward-looking**: We detect spikes from historical patterns. Future novel promotions won't be captured.

2. **Store-wide threshold (15%) is empirical**: May need adjustment for different retail formats.

3. **No true promotion data**: All "promotional" features are inferred, not ground truth.

4. **Seasonal patterns assume stationarity**: If business changes seasonality (e.g., new product launches), historical patterns may not hold.

5. **C-item daily accuracy remains low**: Sparse data fundamentally limits daily-level predictions.

---

## References

1. Hewage et al. (2024). "Enhancing Demand Forecasting in Retail: Promotional Effects on the Entire Demand Life Cycle." *Journal of Forecasting*.

2. Artefact (2023). "Counterfactual Forecasting for Promotion Profitability." *Artefact Engineering Blog*.

3. RELEX Solutions (2024). "Machine Learning in Retail Demand Forecasting." *Industry Whitepaper*.

4. Microsoft (2024). "Sales Anomaly Detection with ML.NET." *Microsoft Learn*.

---

*Document Version: 1.0 | Last Updated: January 2026*
