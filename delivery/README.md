# RG-Forecast: Retail Demand Forecasting System

## Project Overview

A production-ready demand forecasting system for a retail chain in the Middle East.

| Metric | Value |
|--------|-------|
| Stores | 33 |
| SKUs | 3,650 |
| Store-SKU Series | 114,501 |
| Forecast Horizon | 168 days (24 weeks) |
| Forecast Rows | 17.9 million |
| Training Data | 7 years (Jan 2019 - Dec 2025) |
| Raw Transactions | 60.9 million rows |
| Panel-Expanded Rows | 134.9 million rows |

## Accuracy Results (V2 Pipeline - January 2026)

### By Aggregation Level (High-Volume Items)

| Level | WFA | WMAPE | Bias |
|-------|-----|-------|------|
| **Daily SKU-Store** | **54.3%** | 45.7% | +5.1% |
| Weekly SKU-Store | 58.5% | 41.5% | +5.1% |
| **Weekly Store** | **81.0%** | 19.0% | +5.1% |
| Weekly Total | 92.7% | 7.3% | +5.1% |

### By ABC Segment

| Segment | Daily WFA | Bias |
|---------|-----------|------|
| A-Items (80% sales) | 54.0% | +3.8% |
| B-Items (15% sales) | 58.1% | +11.3% |
| C-Items (5% sales) | 47.5% | +6.1% |

### By Tier (Full Pipeline)

| Level | T1 Mature | T2 Growing | T3 Cold Start |
|-------|-----------|------------|---------------|
| Daily SKU-Store | **54%** | 46% | 44% |
| Weekly Store | **88%** | **80%** | **60%** |

**Key Insight**: The 81-88% weekly store accuracy is the headline metric for replenishment decisions.

### Tier Distribution

| Tier | Series Count | Sales Volume |
|------|--------------|--------------|
| T1 Mature (6+ years) | 65,000 | 93% |
| T2 Growing (1-6 years) | 35,000 | 7% |
| T3 Cold Start (<90 days) | 14,000 | <1% |

## Key Technical Decisions

### 1. Two-Stage Model Architecture
- **Stage 1 (Classifier)**: Predicts probability of a sale occurring
- **Stage 2 (Regressor)**: Predicts quantity if sale occurs
- **Rationale**: 75% of observations are zeros; separating the problems improves both

### 2. Log Transform on Target
- Training uses log(sales + 1) with smearing correction
- Added 4.6 percentage points to accuracy

### 3. Data Tiering
- T1/T2/T3 tiers based on history length
- ABC segmentation within each tier
- Different regularization per segment

### 4. Optimized Hyperparameters per Segment
| Segment | num_leaves | lr | n_estimators | min_child | threshold | calibration |
|---------|------------|-----|--------------|-----------|-----------|-------------|
| A-items | 511 | 0.012 | 1200 | 5 | 0.45 | 1.25 |
| B-items | 127 | 0.02 | 400 | 15 | 0.50 | 1.15 |
| C-items | 63 | 0.03 | 200 | 30 | 0.55 | 1.10 |

### 5. 52 Production Features (8 Categories)
- Calendar (10), Lag (5), Rolling (10), Sparse-Aware (6)
- Spike (7), Hierarchy (6), Dormancy (4), Closure (4)

### 6. Medallion Architecture
- **Bronze**: Raw transaction data (audit trail)
- **Silver**: Complete panel with explicit zeros
- **Gold**: 52 engineered features

### 7. What Did NOT Work
- Per-store models (44% - data fragmentation)
- Holiday indicator flags (36% - overfitting)
- Local/Imported SKU attribute (no improvement)

## File Descriptions

### Pipeline Code

| File | Description |
|------|-------------|
| `src/vertex_training/train_v2_DEFINITIVE.py` | **V2 Training (52 features, all optimizations)** |
| `src/train_all_tiers.py` | Main training pipeline for all tiers |
| `src/generate_168day_forecast.py` | Production forecast generation |
| `src/validate_all_folds.py` | Cross-validation runner |
| `src/business_metrics_all_tiers.py` | Accuracy metrics calculation |
| `app/main.py` | API endpoints |
| `pipeline_ui/app.py` | Streamlit dashboard |

### Configuration

| File | Description |
|------|-------------|
| `configs/config.yaml` | Main configuration |
| `params/pipeline_params.yaml` | Hyperparameters by segment |

### Data Artifacts

| File | Description |
|------|-------------|
| `final_data 2.csv` | Raw transaction data (60.9M rows) |
| `sku_list_attribute.csv` | SKU attributes (Local/Imported) |

## How to Run the Pipeline

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Set Up GCP Credentials
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### 2. Run Training
```bash
python src/train_all_tiers.py
```

### 3. Generate Forecasts
```bash
python src/generate_168day_forecast.py
```

### 4. View Dashboard
```bash
streamlit run pipeline_ui/app.py
```

### 5. Access Forecasts
Forecasts are stored in BigQuery:
- `rg-forecast.forecasting.production_forecast_168d`

Or via Streamlit dashboard download buttons.

## Infrastructure

- **Compute**: GCP Vertex AI / n2-standard-8
- **Storage**: BigQuery + Cloud Storage
- **Dashboard**: Streamlit (deployed via ngrok for demo)

## Improvement Opportunities

1. **Promotional Data**: Highest-value addition (estimated +10-15pp daily accuracy)
2. **Stock-out Detection**: Distinguish no-demand from no-stock
3. **Event Calendars**: Ramadan, Eid, local events
4. **Deep Learning**: Temporal Fusion Transformer for A-items (requires GPU)

## Contact

For questions about this forecasting system, refer to:
- `TECHNICAL_SUMMARY.md` - Architecture and methodology details
- `QUICK_START.md` - Step-by-step setup guide
