# RG-Forecasting Pipeline

**Daily SKU×Store Forecasting for 24 Weeks**

## Business Goal
Produce daily forecasts for **33 stores** (201-233), **~3,650 SKUs**, for the next **168 days** (24 weeks) starting **2025-12-18**.

## Project Structure
```
RG-Forecasting/
├── configs/
│   └── config.yaml          # Single source of truth
├── notebooks/
│   └── 01_build_spine.ipynb  # Step-by-step notebook
├── sql/
│   ├── 01_ingest_bronze.sql
│   ├── 02_deduplicate_silver.sql
│   ├── 03_cleaning_silver_plus.sql
│   ├── 04_build_spine_gold.sql
│   ├── 05_pre_tiering.sql
│   ├── 06_tiering.sql
│   ├── 07_feature_engineering.sql
│   └── 08_time_splits.sql
├── src/                      # Python modules (to be added)
├── artifacts/                # Reports/plots (to be added)
├── app/
│   ├── main.py               # Streamlit dashboard
│   ├── Dockerfile
│   └── requirements.txt
└── README.md
```

## How to Run

### Prerequisites
- GCP Project: `myforecastingsales`
- BigQuery Dataset: `forecasting`
- GCS Bucket: `gs://myforecastingsales-data/`

### Step-by-Step

1. **Step 0: Setup**
   - Verify project structure
   - Review `configs/config.yaml`

2. **Step 1: Ingest Bronze**
   ```bash
   bq load --source_format=CSV --autodetect \
     myforecastingsales:forecasting.sales_raw \
     gs://myforecastingsales-data/raw/sales/final_data.csv
   ```

3. **Step 2: Deduplicate (Silver)**
   ```bash
   bq query --use_legacy_sql=false < sql/02_deduplicate_silver.sql
   ```

4. **Step 3: Apply Cleaning (Silver+)**
   ```bash
   bq query --use_legacy_sql=false < sql/03_cleaning_silver_plus.sql
   ```

5. **Step 4: Build Spine (Gold)**
   ```bash
   bq query --use_legacy_sql=false < sql/04_build_spine_gold.sql
   ```

6. **Step 5: Pre-Tiering (Calendar + Closures + Stats)**
   ```bash
   bq query --use_legacy_sql=false < sql/05_pre_tiering.sql
   ```

7. **Step 6: Tiering**
   ```bash
   bq query --use_legacy_sql=false < sql/06_tiering.sql
   ```

8. **Step 7: Feature Engineering**
   ```bash
   bq query --use_legacy_sql=false < sql/07_feature_engineering.sql
   ```

9. **Step 8: Time Splits**
   ```bash
   bq query --use_legacy_sql=false < sql/08_time_splits.sql
   ```

## Data Lineage
```
GCS: final_data.csv
    ↓
sales_raw (Bronze - original)
    ↓
sales_daily (Silver - deduplicated)
    ↓
sales_daily_clean (Silver+ - with flags)
    ↓
gold_panel_spine (Gold - complete grid, 134.9M rows)
    ↓
gold_panel_spine_enriched (Gold+ - with calendar + closures)
    ↓
series_stats_pre_tiering (Series-level stats for tiering)
    ↓
[Tiering] → tier assignments
    ↓
features_daily (Features - ready for modeling)
    ↓
train_set / val_set (Splits)
```

## Cleaning Rules (LOCKED)
| Rule | Implementation |
|------|----------------|
| Negatives | `sales_clean = max(saleqty, 0)` |
| COVID period | `is_covid_period = 1` for 2020-03-15 to 2021-06-30 (diagnostic) |
| COVID panic | `is_covid_panic_spike = 1` for COVID + sales > p99.9 (downweight) |
| Extreme spikes | `is_extreme_spike = 1` for sales > 10,000 (keep as valid) |

## Closure Days (Forced - No Algorithm)
| Holiday | Dates |
|---------|-------|
| New Year | Jan 1 (2019-2026) |
| Christmas | Dec 25 (2019-2026) |
| Good Friday | 2019-04-19, 2020-04-10, 2021-04-02, 2022-04-15, 2023-04-07, 2024-03-29, 2025-04-18, 2026-04-03 |

**Total closure days:** 24 (3 holidays × 8 years)
**In forecast horizon:** 3 (2025-12-25, 2026-01-01, 2026-04-03)

## Dashboard
**URL:** https://forecasting-sales-pipeline-130714632895.me-central1.run.app

## Key Dates
| Date | Description |
|------|-------------|
| 2019-01-02 | Data start |
| 2025-12-17 | Data end |
| 2025-12-18 | Forecast start |
| 2026-06-03 | Forecast end (24 weeks) |

## Tiering (Mutually Exclusive)

### Thresholds
| Constant | Value |
|----------|-------|
| T1_MIN_HISTORY_DAYS | 728 (2 years) |
| T2_MIN_HISTORY_DAYS | 182 (6 months) |
| RECENCY_THRESHOLD_DAYS | 182 (6 months) |
| ADI_INTERMITTENT_THRESHOLD | 1.32 |

### Tier Definitions (strict order)
| Tier | Rule | Description |
|------|------|-------------|
| T0_EXCLUDED | n_pos_days=0 OR days_since_last_sale>182 | Dead/inactive series |
| T1_MATURE | history_days >= 728 | 2+ years of history |
| T2_GROWING | 182 <= history_days < 728 | 6mo-2yr of history |
| T3_COLD_START | history_days < 182 | Less than 6mo |

**Note:** Intermittency (ADI >= 1.32) is a FLAG, not a tier.

### Tier Summary (as-of 2025-12-17)
| Tier | Series Count | % | Sales Share |
|------|--------------|---|-------------|
| T0_EXCLUDED | 2,474 | 2.11% | 0.14% |
| T1_MATURE | 65,724 | 56.19% | 92.71% |
| T2_GROWING | 34,639 | 29.61% | 6.55% |
| T3_COLD_START | 14,138 | 12.09% | 0.60% |

### Tier × Intermittency
| Tier | Smooth | Intermittent |
|------|--------|--------------|
| T0_EXCLUDED | 26 | 2,448 |
| T1_MATURE | 13,378 | 52,346 |
| T2_GROWING | 5,867 | 28,772 |
| T3_COLD_START | 3,287 | 10,851 |

### Inactivity Buckets
| Bucket | Series | % |
|--------|--------|---|
| Never sold | 10 | 0.01% |
| 0-28 days | 104,858 | 89.64% |
| 29-56 days | 4,510 | 3.86% |
| 57-182 days | 5,133 | 4.39% |
| >182 days | 2,464 | 2.11% |

## Time Splits (168-Day Horizon)

### Fold Definitions
| Fold | Tier | Train Start | Train End | Val Start | Val End | Embargo |
|------|------|-------------|-----------|-----------|---------|---------|
| F1 | T1_MATURE | 2019-01-02 | 2025-06-26 | 2025-07-03 | 2025-12-17 | 7 days |
| F2 | T1_MATURE | 2019-01-02 | 2025-03-10 | 2025-03-17 | 2025-08-31 | 7 days |
| F3 | T1_MATURE | 2019-01-02 | 2024-11-07 | 2024-11-14 | 2025-04-30 | 7 days |
| G1 | T2_GROWING | 2023-07-06 | 2025-06-26 | 2025-07-03 | 2025-12-17 | 7 days |
| G2 | T2_GROWING | 2023-03-20 | 2025-03-10 | 2025-03-17 | 2025-08-31 | 7 days |
| C1 | T3_COLD_START | 2025-01-02 | 2025-06-26 | 2025-07-03 | 2025-12-17 | 7 days |

## T0 Breakdown (Exclusion Reasons)
| Reason | Count | % of T0 |
|--------|-------|---------|
| DORMANT_182D | 2,464 | 99.6% |
| NEVER_SOLD | 10 | 0.4% |

## Historical Revival Rates (20,206 Dormancy Events)
| Horizon | Revived | Rate |
|---------|---------|------|
| 30 days | 4,009 | 19.84% |
| 90 days | 8,613 | 42.63% |
| 180 days | 11,971 | 59.24% |

## Feature Engineering (Step 7)

### V1 Causal Features (gold_panel_features_v1)
All features use only information up to date-1 (no leakage).

| Feature Category | Features |
|------------------|----------|
| Lags | lag_1, lag_7, lag_14, lag_28, lag_56 |
| Rolling (7d) | roll_mean_7, roll_sum_7 |
| Rolling (28d) | roll_mean_28, roll_sum_28, roll_std_28, nz_rate_28 |
| Recency | days_since_last_sale_asof, zero_run_length_asof, last_sale_qty_asof |

### V2 Enhanced Features (gold_panel_features_v2)
Improvements for better seasonality, sparse handling, and closure proximity.

| Feature Category | New Features | Purpose |
|------------------|--------------|---------|
| Yearly Seasonality | sin_doy, cos_doy | Smooth yearly cycle (365.25) |
| Weekly Seasonality | sin_dow, cos_dow | Smooth weekly cycle |
| Closure Proximity | days_to_next_closure, days_from_prev_closure, is_closure_week | Holiday effects |
| Dormancy | dormancy_bucket (D0_7, D8_28, D29_56, D57_182, D183_PLUS), dormancy_capped | Sparse handling |
| Intermittency | nz_rate_7, roll_mean_pos_28 | Non-zero rate + positive-only rolling |
| Safe Transforms | y_log1p, lag_7_log1p, roll_mean_28_log1p | Model stability |

### Validation Results
- Total rows: 134,887,953
- sin/cos ranges: **PASS** (all within [-1, 1])
- Closure proximity: **PASS** (0-60 days, capped)
- Closure week flag: **PASS** (100% of closure days flagged)
- Causality check: **PASS** (lag_1 matches previous day sales 100%)

## Baseline Backtest Results (F1 Fold: 2025-07-03 to 2025-12-17)

### Overall Metrics (17.9M observations)
| Model | MAE | WMAPE | RMSLE | ZeroAcc | MAE_nonzero |
|-------|-----|-------|-------|---------|-------------|
| pred_lag7 | 3.41 | 72.48% | 0.770 | 0.431 | 6.49 |
| pred_roll28 | **2.93** | **62.38%** | **0.642** | 0.379 | **5.31** |
| pred_smart | **2.93** | **62.38%** | **0.642** | 0.379 | **5.31** |

### By Tier
| Tier | Model | MAE | WMAPE | ZeroAcc | MAE_nonzero |
|------|-------|-----|-------|---------|-------------|
| T1_MATURE | pred_lag7 | 3.88 | 69.26% | 0.394 | 6.93 |
| T1_MATURE | **pred_roll28** | **3.29** | **58.82%** | 0.336 | **5.64** |
| T2_GROWING | pred_lag7 | 2.49 | 79.86% | 0.496 | 5.27 |
| T2_GROWING | **pred_roll28** | **2.16** | **69.53%** | 0.457 | **4.33** |
| T3_COLD_START | pred_lag7 | 3.56 | 88.47% | 0.467 | 7.39 |
| T3_COLD_START | **pred_roll28** | **3.41** | **84.73%** | 0.395 | **6.52** |

### By Dormancy Bucket (with Metric Guards)
| Bucket | % Data | MAE | WMAPE | ZeroAcc | MAE_nonzero | Best Model |
|--------|--------|-----|-------|---------|-------------|------------|
| D0_7 | 73.3% | 3.79 | 59.89% | 0.204 | 5.39 | Roll28 |
| D8_28 | 14.8% | 0.28 | 100% | 0.912 | 3.11 | **Lag7** |
| D29_56 | 5.6% | 0.12 | 100% | 0.965 | 3.47 | Tie |
| D57_182 | 5.2% | 0.06 | 100% | 0.978 | 2.99 | Tie |
| D183_PLUS | 1.2% | 0.97 | 100% | 0.924 | 12.70 | Tie |

**Key Insights:**
1. **Roll28 wins for very active series** (D0_7): WMAPE 59.89%
2. **Lag7 wins for moderately sparse** (D8_28): WMAPE 100% vs Roll28's 327%
3. **High ZeroAcc for sparse buckets** (>91%): Models correctly predict zeros
4. **MAE_nonzero** reveals true error on actual demand: ~3-13 units
5. **WMAPE is misleading for sparse buckets** where sum(actual)≈0

## Pipeline Steps Status
| Step | Description | Status |
|------|-------------|--------|
| 0 | Project Setup | ✅ DONE |
| 1 | Ingest Bronze | ✅ DONE |
| 2 | Deduplicate Silver | ✅ DONE |
| 3 | Cleaning Silver+ | ✅ LOCKED |
| 4 | Build Spine (Gold) | ✅ DONE |
| 5 | Pre-Tiering (Calendar + Closures + Stats) | ✅ DONE |
| 6 | Tiering | ✅ DONE |
| 7 | Feature Engineering | ✅ DONE |
| 8 | Time Splits | ✅ DONE |
| 9 | Baselines | ✅ DONE |
| 10 | Model Training | ✅ DONE (59.4% A-items, 84% weekly store) |
| 11 | Evaluation | ✅ DONE |
| 12 | Forecast Generation | ✅ DONE (168-day forecast live) |
