-- =============================================================================
-- STEP 10: TRAIN MODELS USING BIGQUERY ML
-- =============================================================================
-- Uses BOOSTED_TREE_REGRESSOR (XGBoost-based, similar to LightGBM)
-- Trains per-tier models using the validated v_trainval_lgbm_v2 view
-- =============================================================================

-- =============================================================================
-- PART A: T1_MATURE MODEL (F1 Fold - 104M train rows)
-- =============================================================================
CREATE OR REPLACE MODEL `myforecastingsales.forecasting.model_t1_mature_f1_v1`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['y'],
  data_split_method = 'CUSTOM',
  data_split_col = 'is_val',
  booster_type = 'GBTREE',
  num_parallel_tree = 1,
  max_iterations = 100,
  learn_rate = 0.05,
  min_split_loss = 0,
  max_tree_depth = 8,
  subsample = 0.8,
  colsample_bytree = 0.8,
  colsample_bylevel = 1.0,
  l1_reg = 0,
  l2_reg = 1,
  early_stop = TRUE,
  min_rel_progress = 0.001,
  enable_global_explain = TRUE
) AS
SELECT
  y,
  CASE WHEN split_role = 'VAL' THEN TRUE ELSE FALSE END AS is_val,

  -- Calendar features
  dow,
  is_weekend,
  week_of_year,
  month,
  year,
  day_of_year,
  sin_doy,
  cos_doy,
  sin_dow,
  cos_dow,

  -- Closure features
  is_store_closed,
  days_to_next_closure,
  days_from_prev_closure,
  is_closure_week,

  -- Lag features
  COALESCE(lag_1, 0) AS lag_1,
  COALESCE(lag_7, 0) AS lag_7,
  COALESCE(lag_14, 0) AS lag_14,
  COALESCE(lag_28, 0) AS lag_28,
  COALESCE(lag_56, 0) AS lag_56,

  -- Rolling features
  COALESCE(roll_mean_7, 0) AS roll_mean_7,
  COALESCE(roll_sum_7, 0) AS roll_sum_7,
  COALESCE(roll_mean_28, 0) AS roll_mean_28,
  COALESCE(roll_sum_28, 0) AS roll_sum_28,
  COALESCE(roll_std_28, 0) AS roll_std_28,

  -- Sparse/intermittency features
  COALESCE(nz_rate_7, 0) AS nz_rate_7,
  COALESCE(nz_rate_28, 0) AS nz_rate_28,
  COALESCE(roll_mean_pos_28, 0) AS roll_mean_pos_28,

  -- Recency features
  COALESCE(days_since_last_sale_asof, 999) AS days_since_last_sale_asof,
  COALESCE(dormancy_capped, 365) AS dormancy_capped,
  COALESCE(zero_run_length_asof, 0) AS zero_run_length_asof,
  COALESCE(last_sale_qty_asof, 0) AS last_sale_qty_asof,

  -- IDs as categorical (BQML handles automatically)
  store_id,
  sku_id

FROM `myforecastingsales.forecasting.v_trainval_lgbm_v2`
WHERE fold_id = 'F1' AND tier_name = 'T1_MATURE';


-- =============================================================================
-- PART B: T2_GROWING MODEL (G1 Fold - 9M train rows)
-- =============================================================================
CREATE OR REPLACE MODEL `myforecastingsales.forecasting.model_t2_growing_g1_v1`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['y'],
  data_split_method = 'CUSTOM',
  data_split_col = 'is_val',
  booster_type = 'GBTREE',
  num_parallel_tree = 1,
  max_iterations = 100,
  learn_rate = 0.05,
  max_tree_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  l2_reg = 1,
  early_stop = TRUE,
  enable_global_explain = TRUE
) AS
SELECT
  y,
  CASE WHEN split_role = 'VAL' THEN TRUE ELSE FALSE END AS is_val,

  -- Calendar features (reduced)
  dow,
  is_weekend,
  month,
  year,
  day_of_year,
  sin_doy,
  cos_doy,

  -- Closure features
  is_store_closed,
  is_closure_week,

  -- Lag features (reduced)
  COALESCE(lag_1, 0) AS lag_1,
  COALESCE(lag_7, 0) AS lag_7,
  COALESCE(lag_14, 0) AS lag_14,
  COALESCE(lag_28, 0) AS lag_28,

  -- Rolling features (reduced)
  COALESCE(roll_mean_7, 0) AS roll_mean_7,
  COALESCE(roll_mean_28, 0) AS roll_mean_28,
  COALESCE(nz_rate_28, 0) AS nz_rate_28,

  -- Recency features
  COALESCE(days_since_last_sale_asof, 999) AS days_since_last_sale_asof,
  COALESCE(zero_run_length_asof, 0) AS zero_run_length_asof,

  -- IDs
  store_id,
  sku_id

FROM `myforecastingsales.forecasting.v_trainval_lgbm_v2`
WHERE fold_id = 'G1' AND tier_name = 'T2_GROWING';


-- =============================================================================
-- PART C: T3_COLD_START MODEL (C1 Fold - 641K train rows)
-- =============================================================================
CREATE OR REPLACE MODEL `myforecastingsales.forecasting.model_t3_coldstart_c1_v1`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['y'],
  data_split_method = 'CUSTOM',
  data_split_col = 'is_val',
  booster_type = 'GBTREE',
  num_parallel_tree = 1,
  max_iterations = 50,
  learn_rate = 0.1,
  max_tree_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  l2_reg = 2,
  early_stop = TRUE,
  enable_global_explain = TRUE
) AS
SELECT
  y,
  CASE WHEN split_role = 'VAL' THEN TRUE ELSE FALSE END AS is_val,

  -- Calendar features (minimal)
  dow,
  is_weekend,
  month,
  day_of_year,
  sin_doy,
  cos_doy,

  -- Closure features
  is_store_closed,
  is_closure_week,

  -- Lag features (minimal)
  COALESCE(lag_1, 0) AS lag_1,
  COALESCE(lag_7, 0) AS lag_7,
  COALESCE(lag_14, 0) AS lag_14,

  -- Rolling features (minimal)
  COALESCE(roll_mean_28, 0) AS roll_mean_28,
  COALESCE(nz_rate_28, 0) AS nz_rate_28,

  -- Recency
  COALESCE(days_since_last_sale_asof, 999) AS days_since_last_sale_asof,

  -- IDs
  store_id,
  sku_id

FROM `myforecastingsales.forecasting.v_trainval_lgbm_v2`
WHERE fold_id = 'C1' AND tier_name = 'T3_COLD_START';


-- =============================================================================
-- PART D: EVALUATION QUERIES (run after training completes)
-- =============================================================================

-- D1: Model training info
SELECT * FROM ML.TRAINING_INFO(MODEL `myforecastingsales.forecasting.model_t1_mature_f1_v1`);
SELECT * FROM ML.TRAINING_INFO(MODEL `myforecastingsales.forecasting.model_t2_growing_g1_v1`);
SELECT * FROM ML.TRAINING_INFO(MODEL `myforecastingsales.forecasting.model_t3_coldstart_c1_v1`);

-- D2: Feature importance (global explain)
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `myforecastingsales.forecasting.model_t1_mature_f1_v1`);
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `myforecastingsales.forecasting.model_t2_growing_g1_v1`);
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `myforecastingsales.forecasting.model_t3_coldstart_c1_v1`);

-- D3: Evaluate on VAL set with custom metrics
-- (Run after predictions are generated)
