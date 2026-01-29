# Retail Sales Forecasting Project

## Business Context

This project is for a retail business with:
- **33 stores**
- **~3,500 SKUs** (actual dataset contains 3,650 SKUs)
- Complex sales behavior including:
  - Panic buying periods
  - Sudden spikes
  - Seasonality
  - Low and high moving items
  - Long gaps in sales
- Live retail environment where outputs support **operational decision making**

## Scope of Work

### Primary Deliverables
1. **Sales forecasting model** at store and SKU level for a **24-week horizon**
2. **Full implementation in notebook form** with documentation
3. **Written summary** explaining approach, assumptions, and key decisions

### Forecast Details
- **Training data**: 2019-01-02 to 2025-12-17
- **Forecast horizon**: 24 weeks from 2025-12-18 (ending ~2026-06-03)
- **Granularity**: Store-SKU-Day level (daily forecasts)
- **Total forecast points**: ~3,650 SKUs × 33 stores × 168 days = ~20.2 million rows

## Dataset Information

### Main Dataset: `final_data 2.csv`
| Column | Description |
|--------|-------------|
| item_id | Internal SKU identifier |
| store_id | Store identifier |
| date | Transaction date |
| sales | Quantity sold for that item on that date |

**Important Notes:**
- ~61 million rows
- Missing dates imply **no sales** (not missing data)
- Zero sales are **valid observations**
- Includes fast movers, slow movers, intermittent SKUs, and SKUs with long sales gaps

### Attribute Dataset: `sku_list_attribute.csv`
| Column | Description |
|--------|-------------|
| sku_id | SKU identifier (maps to item_id) |
| local_imported_attribute | L = Local, I = Imported, LI = Both |

**Requirement**: Append this attribute to the dataset and consider treating Local vs Imported SKUs separately, as their sales behavior can differ significantly.

## Performance Expectations

The solution must demonstrate:
1. **Stable behavior** across all SKUs
2. **Sensible forecasts** across stores with different sales patterns
3. **Ability to handle long gaps and zero sales**
4. **Ability to handle spikes and unusual behavior**
5. **Reasonable handling of newly introduced SKUs** with limited history
6. **Ability to recover sensibly from missed predictions**

## Evaluation Checklist

| Criterion | Description |
|-----------|-------------|
| Problem Approach | How you frame the forecasting problem and assumptions |
| Data Handling | Cleaning, imputation, and treatment of anomalies |
| Model Design | Suitability of methods and structure |
| Code Quality | Clarity, structure, and reproducibility |
| Reasoning | Explanation of trade-offs and decisions |
| Practicality | How usable and stable the outputs are in a real business setting |

## Metrics Selection

**No single mandated metric.** Choose metric(s) that best reflect forecast quality in a real retail setting.

Key considerations:
- Explain **why** you chose those metrics
- Show how they behave across **fast movers, slow movers, and intermittent SKUs**
- Explain handling of **edge cases** (zero sales, long gaps)

Suggested metrics to consider:
- **WAPE** (Weighted Absolute Percentage Error) - good for varying volume items
- **MAPE** (Mean Absolute Percentage Error) - interpretable but problematic with zeros
- **sMAPE** (Symmetric MAPE) - handles zeros better
- **RMSE** (Root Mean Square Error) - penalizes large errors
- **MAE** (Mean Absolute Error) - robust to outliers
- **Bias** - directional accuracy

## Expected Output Format

### Forecast CSV
| Column | Description |
|--------|-------------|
| item_id | SKU identifier |
| store_id | Store identifier |
| date | Future date (daily) |
| predicted_sales | Forecasted quantity |
| *(optional)* lower_bound | Lower confidence interval |
| *(optional)* upper_bound | Upper confidence interval |

## Acceptance Criteria

### Must Have
- [ ] Working forecasting model producing 24-week daily forecasts (168 days) for all store-SKU combinations
- [ ] Train/test split validation demonstrating model performance before final training
- [ ] Forecast output in specified CSV format
- [ ] Jupyter notebook with full implementation
- [ ] Written summary document explaining approach and decisions
- [ ] Handling of Local vs Imported SKU classification

### Should Have
- [ ] Multiple metrics showing performance across different SKU types
- [ ] Visualization of forecasts vs actuals for sample SKUs
- [ ] Documentation of data cleaning and imputation decisions
- [ ] Confidence intervals on forecasts

### Nice to Have
- [ ] Comparison of multiple modeling approaches
- [ ] Anomaly detection and treatment explanation
- [ ] Performance breakdown by store and SKU category
- [ ] Scalability considerations for production deployment

## Expected Artifacts

1. **`forecast_output.csv`** - Final 24-week daily forecast (168 days) for all store-SKU combinations
2. **`forecasting_notebook.ipynb`** - Complete Jupyter notebook with:
   - Data exploration and cleaning
   - Feature engineering
   - Model training and validation
   - Forecast generation
3. **`summary_report.md`** - Written summary including:
   - Problem framing and assumptions
   - Data handling decisions
   - Model selection rationale
   - Metrics and results
   - Observations on Local vs Imported SKUs
   - Key trade-offs and decisions
4. **`validation_results.csv`** - Train/test split performance metrics

## Technical Notes

### Data Characteristics
- Date range: 2019-01-02 to 2025-12-17 (~7 years)
- Rows: ~61 million
- SKUs: 3,650
- Stores: 33
- Data is sparse (not all SKU-store-date combinations exist)

### Environment
- GCP VM: n2-standard-8 (8 vCPUs, 32 GB RAM) in me-central1-a
- Python-based implementation preferred

### Key Challenges
1. Large dataset requiring efficient processing
2. Intermittent demand patterns
3. Zero-inflated sales data
4. Cold start problem for new SKUs
5. Varying seasonality across products
6. Panic buying and spike detection
