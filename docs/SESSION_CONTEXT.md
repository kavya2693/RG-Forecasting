# RG-Forecasting Project Context (for new Claude session)

## Project Overview
- **Retail demand forecasting** for 114K SKU-store series, 33 stores
- **134.9M rows** of data, 75% zeros (sparse intermittent demand)
- **Two-stage LightGBM model**: binary classifier + log-transform regressor
- **ABC segmentation**: A-items (80% sales), B-items (15%), C-items (5%)
- **Tiered architecture**: T1_MATURE, T2_GROWING, T3_COLD_START

## Current Production Accuracy (from dashboard)
| Tier | Daily WFA | Weekly Store WFA |
|------|-----------|------------------|
| T1_MATURE | 51.5% | 87.9% |
| T2_GROWING | 45.6% | 80.1% |

## Current Work: Spike Feature Testing
Testing whether inferred promotional features (spike detection) improve accuracy.

### Spike Features Created (in BigQuery)
1. `feat_store_spike_pct` - % of SKUs spiking in store today
2. `feat_store_promo_day` - Binary store-wide promotional event (>15% SKUs spike)
3. `feat_seasonal_lift` - Week-level seasonal multiplier (0.5-3.0)
4. `feat_had_recent_spike` - Spike in last 7 days
5. `feat_historical_spike_prob` - Historical spike probability by week

### Validation Shows Signal
- Promo days: 8.79 avg sales (64% higher than non-promo 5.35)
- Recent spike: 6.33 avg sales (24% higher than no recent spike 5.09)

### Vertex AI Job Running
- Job ID: `2009416512611287040`
- Region: us-central1
- Testing T1_MATURE with F1 fold
- Comparing baseline vs baseline + spike features

Check status:
```bash
gcloud ai custom-jobs describe projects/130714632895/locations/us-central1/customJobs/2009416512611287040 --project=myforecastingsales
```

## Key Files
- `/sql/07c_spike_features.sql` - BigQuery spike feature creation
- `/src/vertex_training/trainer/train_spike_test.py` - Vertex AI test script
- `/docs/methodology_documentation.md` - All thresholds justified
- `/tmp/spike_analysis/spike_features_summary.json` - Feature documentation

## Data Locations
- BigQuery: `myforecastingsales.forecasting.v_trainval_lgbm_v3` (with spike features)
- GCS: `gs://myforecastingsales-data/training_code/trainer-0.2.tar.gz`

## Key Decisions Made
1. Spike threshold: 2.0x baseline (captures ~8% meaningful anomalies)
2. Store-wide promo: >15% of SKUs spike (balanced sensitivity/specificity)
3. Seasonal lift: Week avg / overall avg, clipped to [0.5, 3.0]
4. Apply spike features to A and B items only (C too sparse)
