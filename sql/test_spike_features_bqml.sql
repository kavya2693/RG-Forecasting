-- =============================================================================
-- TEST: SPIKE FEATURES IMPACT ON MODEL ACCURACY (BQML)
-- =============================================================================
-- Compares baseline model vs model with spike features using BigQuery ML
-- Uses BOOSTED_TREE_REGRESSOR with same parameters as production
-- =============================================================================

-- =============================================================================
-- STEP 1: TRAIN BASELINE MODEL (WITHOUT SPIKE FEATURES)
-- =============================================================================
CREATE OR REPLACE MODEL `myforecastingsales.forecasting.m_spike_test_baseline`
OPTIONS(
  MODEL_TYPE = 'BOOSTED_TREE_REGRESSOR',
  DATA_SPLIT_METHOD = 'NO_SPLIT',
  NUM_PARALLEL_TREE = 1,
  MAX_ITERATIONS = 100,
  TREE_METHOD = 'HIST',
  SUBSAMPLE = 0.8,
  LEARN_RATE = 0.05,
  MAX_TREE_DEPTH = 6,
  MIN_SPLIT_LOSS = 0,
  L2_REG = 0.5
) AS
SELECT
  y AS label,
  -- Original features only (no spike features)
  CAST(store_id AS STRING) AS store_id,
  CAST(sku_id AS STRING) AS sku_id,
  dow, week_of_year, month, is_weekend, day_of_year,
  sin_doy, cos_doy, sin_dow, cos_dow,
  is_store_closed, days_to_next_closure, days_from_prev_closure, is_closure_week,
  COALESCE(lag_1, 0) AS lag_1,
  COALESCE(lag_7, 0) AS lag_7,
  COALESCE(lag_14, 0) AS lag_14,
  COALESCE(lag_28, 0) AS lag_28,
  COALESCE(roll_mean_7, 0) AS roll_mean_7,
  COALESCE(roll_mean_28, 0) AS roll_mean_28,
  COALESCE(roll_sum_7, 0) AS roll_sum_7,
  COALESCE(roll_sum_28, 0) AS roll_sum_28,
  COALESCE(roll_std_28, 0) AS roll_std_28,
  COALESCE(nz_rate_7, 0) AS nz_rate_7,
  COALESCE(nz_rate_28, 0) AS nz_rate_28,
  COALESCE(roll_mean_pos_28, 0) AS roll_mean_pos_28,
  COALESCE(days_since_last_sale_asof, 0) AS days_since_last_sale,
  COALESCE(dormancy_capped, 0) AS dormancy_capped
FROM `myforecastingsales.forecasting.v_trainval_lgbm_v3`
WHERE fold_id = 'F1'
  AND split_role = 'TRAIN'
  AND tier_name = 'T1_MATURE'
  -- Sample for faster training (10%)
  AND MOD(ABS(FARM_FINGERPRINT(CONCAT(CAST(store_id AS STRING), CAST(sku_id AS STRING), CAST(date AS STRING)))), 10) = 0;


-- =============================================================================
-- STEP 2: TRAIN MODEL WITH SPIKE FEATURES
-- =============================================================================
CREATE OR REPLACE MODEL `myforecastingsales.forecasting.m_spike_test_with_spikes`
OPTIONS(
  MODEL_TYPE = 'BOOSTED_TREE_REGRESSOR',
  DATA_SPLIT_METHOD = 'NO_SPLIT',
  NUM_PARALLEL_TREE = 1,
  MAX_ITERATIONS = 100,
  TREE_METHOD = 'HIST',
  SUBSAMPLE = 0.8,
  LEARN_RATE = 0.05,
  MAX_TREE_DEPTH = 6,
  MIN_SPLIT_LOSS = 0,
  L2_REG = 0.5
) AS
SELECT
  y AS label,
  -- Original features
  CAST(store_id AS STRING) AS store_id,
  CAST(sku_id AS STRING) AS sku_id,
  dow, week_of_year, month, is_weekend, day_of_year,
  sin_doy, cos_doy, sin_dow, cos_dow,
  is_store_closed, days_to_next_closure, days_from_prev_closure, is_closure_week,
  COALESCE(lag_1, 0) AS lag_1,
  COALESCE(lag_7, 0) AS lag_7,
  COALESCE(lag_14, 0) AS lag_14,
  COALESCE(lag_28, 0) AS lag_28,
  COALESCE(roll_mean_7, 0) AS roll_mean_7,
  COALESCE(roll_mean_28, 0) AS roll_mean_28,
  COALESCE(roll_sum_7, 0) AS roll_sum_7,
  COALESCE(roll_sum_28, 0) AS roll_sum_28,
  COALESCE(roll_std_28, 0) AS roll_std_28,
  COALESCE(nz_rate_7, 0) AS nz_rate_7,
  COALESCE(nz_rate_28, 0) AS nz_rate_28,
  COALESCE(roll_mean_pos_28, 0) AS roll_mean_pos_28,
  COALESCE(days_since_last_sale_asof, 0) AS days_since_last_sale,
  COALESCE(dormancy_capped, 0) AS dormancy_capped,
  -- SPIKE FEATURES (the new features we're testing)
  COALESCE(feat_store_spike_pct, 0) AS feat_store_spike_pct,
  COALESCE(feat_store_promo_day, 0) AS feat_store_promo_day,
  COALESCE(feat_seasonal_lift, 1.0) AS feat_seasonal_lift,
  COALESCE(feat_had_recent_spike, 0) AS feat_had_recent_spike,
  COALESCE(feat_historical_spike_prob, 0) AS feat_historical_spike_prob
FROM `myforecastingsales.forecasting.v_trainval_lgbm_v3`
WHERE fold_id = 'F1'
  AND split_role = 'TRAIN'
  AND tier_name = 'T1_MATURE'
  -- Same sample
  AND MOD(ABS(FARM_FINGERPRINT(CONCAT(CAST(store_id AS STRING), CAST(sku_id AS STRING), CAST(date AS STRING)))), 10) = 0;


-- =============================================================================
-- STEP 3: EVALUATE BASELINE MODEL
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.spike_test_baseline_predictions` AS
SELECT
  v.*,
  GREATEST(pred.predicted_label, 0) AS y_pred
FROM `myforecastingsales.forecasting.v_trainval_lgbm_v3` v
JOIN ML.PREDICT(
  MODEL `myforecastingsales.forecasting.m_spike_test_baseline`,
  (
    SELECT
      store_id, sku_id, date, y,
      CAST(store_id AS STRING) AS store_id_str,
      CAST(sku_id AS STRING) AS sku_id_str,
      dow, week_of_year, month, is_weekend, day_of_year,
      sin_doy, cos_doy, sin_dow, cos_dow,
      is_store_closed, days_to_next_closure, days_from_prev_closure, is_closure_week,
      COALESCE(lag_1, 0) AS lag_1,
      COALESCE(lag_7, 0) AS lag_7,
      COALESCE(lag_14, 0) AS lag_14,
      COALESCE(lag_28, 0) AS lag_28,
      COALESCE(roll_mean_7, 0) AS roll_mean_7,
      COALESCE(roll_mean_28, 0) AS roll_mean_28,
      COALESCE(roll_sum_7, 0) AS roll_sum_7,
      COALESCE(roll_sum_28, 0) AS roll_sum_28,
      COALESCE(roll_std_28, 0) AS roll_std_28,
      COALESCE(nz_rate_7, 0) AS nz_rate_7,
      COALESCE(nz_rate_28, 0) AS nz_rate_28,
      COALESCE(roll_mean_pos_28, 0) AS roll_mean_pos_28,
      COALESCE(days_since_last_sale_asof, 0) AS days_since_last_sale,
      COALESCE(dormancy_capped, 0) AS dormancy_capped
    FROM `myforecastingsales.forecasting.v_trainval_lgbm_v3`
    WHERE fold_id = 'F1'
      AND split_role = 'VAL'
      AND tier_name = 'T1_MATURE'
  )
) pred
ON v.store_id = pred.store_id AND v.sku_id = pred.sku_id AND v.date = pred.date
WHERE v.fold_id = 'F1' AND v.split_role = 'VAL' AND v.tier_name = 'T1_MATURE';


-- =============================================================================
-- STEP 4: EVALUATE MODEL WITH SPIKE FEATURES
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.spike_test_with_spikes_predictions` AS
SELECT
  v.*,
  GREATEST(pred.predicted_label, 0) AS y_pred
FROM `myforecastingsales.forecasting.v_trainval_lgbm_v3` v
JOIN ML.PREDICT(
  MODEL `myforecastingsales.forecasting.m_spike_test_with_spikes`,
  (
    SELECT
      store_id, sku_id, date, y,
      CAST(store_id AS STRING) AS store_id_str,
      CAST(sku_id AS STRING) AS sku_id_str,
      dow, week_of_year, month, is_weekend, day_of_year,
      sin_doy, cos_doy, sin_dow, cos_dow,
      is_store_closed, days_to_next_closure, days_from_prev_closure, is_closure_week,
      COALESCE(lag_1, 0) AS lag_1,
      COALESCE(lag_7, 0) AS lag_7,
      COALESCE(lag_14, 0) AS lag_14,
      COALESCE(lag_28, 0) AS lag_28,
      COALESCE(roll_mean_7, 0) AS roll_mean_7,
      COALESCE(roll_mean_28, 0) AS roll_mean_28,
      COALESCE(roll_sum_7, 0) AS roll_sum_7,
      COALESCE(roll_sum_28, 0) AS roll_sum_28,
      COALESCE(roll_std_28, 0) AS roll_std_28,
      COALESCE(nz_rate_7, 0) AS nz_rate_7,
      COALESCE(nz_rate_28, 0) AS nz_rate_28,
      COALESCE(roll_mean_pos_28, 0) AS roll_mean_pos_28,
      COALESCE(days_since_last_sale_asof, 0) AS days_since_last_sale,
      COALESCE(dormancy_capped, 0) AS dormancy_capped,
      COALESCE(feat_store_spike_pct, 0) AS feat_store_spike_pct,
      COALESCE(feat_store_promo_day, 0) AS feat_store_promo_day,
      COALESCE(feat_seasonal_lift, 1.0) AS feat_seasonal_lift,
      COALESCE(feat_had_recent_spike, 0) AS feat_had_recent_spike,
      COALESCE(feat_historical_spike_prob, 0) AS feat_historical_spike_prob
    FROM `myforecastingsales.forecasting.v_trainval_lgbm_v3`
    WHERE fold_id = 'F1'
      AND split_role = 'VAL'
      AND tier_name = 'T1_MATURE'
  )
) pred
ON v.store_id = pred.store_id AND v.sku_id = pred.sku_id AND v.date = pred.date
WHERE v.fold_id = 'F1' AND v.split_role = 'VAL' AND v.tier_name = 'T1_MATURE';


-- =============================================================================
-- STEP 5: COMPARE RESULTS
-- =============================================================================

-- Daily WFA comparison
SELECT
  'BASELINE (no spike features)' AS model,
  COUNT(*) AS rows,
  ROUND(SUM(y), 0) AS total_actual,
  ROUND(SUM(y_pred), 0) AS total_predicted,
  ROUND(100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS daily_wmape,
  ROUND(100.0 - 100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS daily_wfa
FROM `myforecastingsales.forecasting.spike_test_baseline_predictions`

UNION ALL

SELECT
  'WITH SPIKE FEATURES' AS model,
  COUNT(*) AS rows,
  ROUND(SUM(y), 0) AS total_actual,
  ROUND(SUM(y_pred), 0) AS total_predicted,
  ROUND(100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS daily_wmape,
  ROUND(100.0 - 100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS daily_wfa
FROM `myforecastingsales.forecasting.spike_test_with_spikes_predictions`;


-- Weekly Store WFA comparison
WITH weekly_base AS (
  SELECT
    store_id,
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(WEEK FROM date) AS week,
    SUM(y) AS y,
    SUM(y_pred) AS y_pred
  FROM `myforecastingsales.forecasting.spike_test_baseline_predictions`
  GROUP BY 1, 2, 3
),
weekly_spike AS (
  SELECT
    store_id,
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(WEEK FROM date) AS week,
    SUM(y) AS y,
    SUM(y_pred) AS y_pred
  FROM `myforecastingsales.forecasting.spike_test_with_spikes_predictions`
  GROUP BY 1, 2, 3
)
SELECT
  'BASELINE (no spike features)' AS model,
  'Weekly Store' AS aggregation,
  ROUND(100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS wmape,
  ROUND(100.0 - 100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS wfa
FROM weekly_base

UNION ALL

SELECT
  'WITH SPIKE FEATURES' AS model,
  'Weekly Store' AS aggregation,
  ROUND(100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS wmape,
  ROUND(100.0 - 100.0 * SUM(ABS(y - y_pred)) / NULLIF(SUM(y), 0), 2) AS wfa
FROM weekly_spike;
