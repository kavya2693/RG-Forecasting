-- =============================================================================
-- STEP 7B: FEATURE ENGINEERING V2 (ENHANCED, LEAKAGE-PROOF)
-- =============================================================================
-- Improvements over V1:
-- 1) Yearly/weekly seasonality (sin/cos encodings)
-- 2) Closure proximity features (days to/from closure)
-- 3) Recency buckets + capped dormancy
-- 4) Intermittency severity features (nz_rate_7, roll_mean_pos_28)
-- 5) Safe transforms (log1p) for model stability
-- =============================================================================

-- =============================================================================
-- PART A: CREATE GOLD_PANEL_FEATURES_V2
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.gold_panel_features_v2`
PARTITION BY date
CLUSTER BY store_id, sku_id
AS
WITH
-- Get closure dates for proximity calculation
closures AS (
  SELECT closure_date FROM `myforecastingsales.forecasting.dim_store_closures_manual`
),

-- Calculate next/prev closure for each date
closure_proximity AS (
  SELECT
    d.date,
    -- Next closure (min closure_date >= date)
    (SELECT MIN(c.closure_date) FROM closures c WHERE c.closure_date >= d.date) AS next_closure_date,
    -- Prev closure (max closure_date <= date)
    (SELECT MAX(c.closure_date) FROM closures c WHERE c.closure_date <= d.date) AS prev_closure_date
  FROM `myforecastingsales.forecasting.dim_dates_daily` d
),

-- Base features from V1
base AS (
  SELECT
    f.*,
    -- Add intermittency features (causal)
    AVG(CASE WHEN sales_filled > 0 THEN 1.0 ELSE 0.0 END) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS nz_rate_7,

    -- Rolling mean of positive sales only (causal)
    AVG(IF(sales_filled > 0, sales_filled, NULL)) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS roll_mean_pos_28

  FROM `myforecastingsales.forecasting.gold_panel_features_v1` f
)

SELECT
  b.*,

  -- =========================================================================
  -- 1) YEARLY SEASONALITY ENCODINGS
  -- =========================================================================
  SIN(2 * ACOS(-1) * day_of_year / 365.25) AS sin_doy,
  COS(2 * ACOS(-1) * day_of_year / 365.25) AS cos_doy,

  -- =========================================================================
  -- 2) WEEKLY SEASONALITY ENCODINGS
  -- =========================================================================
  SIN(2 * ACOS(-1) * dow / 7.0) AS sin_dow,
  COS(2 * ACOS(-1) * dow / 7.0) AS cos_dow,

  -- =========================================================================
  -- 3) CLOSURE PROXIMITY FEATURES
  -- =========================================================================
  -- Days to next closure (capped at 60, 60 if none)
  LEAST(
    COALESCE(DATE_DIFF(cp.next_closure_date, b.date, DAY), 60),
    60
  ) AS days_to_next_closure,

  -- Days from prev closure (capped at 60, 60 if none)
  LEAST(
    COALESCE(DATE_DIFF(b.date, cp.prev_closure_date, DAY), 60),
    60
  ) AS days_from_prev_closure,

  -- Is closure week (within ±3 days of any closure)
  CASE
    WHEN COALESCE(DATE_DIFF(cp.next_closure_date, b.date, DAY), 999) <= 3 THEN 1
    WHEN COALESCE(DATE_DIFF(b.date, cp.prev_closure_date, DAY), 999) <= 3 THEN 1
    ELSE 0
  END AS is_closure_week,

  -- =========================================================================
  -- 4) RECENCY/DORMANCY BUCKETS
  -- =========================================================================
  -- Categorical bucket
  CASE
    WHEN days_since_last_sale_asof <= 7 THEN 'D0_7'
    WHEN days_since_last_sale_asof <= 28 THEN 'D8_28'
    WHEN days_since_last_sale_asof <= 56 THEN 'D29_56'
    WHEN days_since_last_sale_asof <= 182 THEN 'D57_182'
    ELSE 'D183_PLUS'
  END AS dormancy_bucket,

  -- Numeric capped at 365
  LEAST(days_since_last_sale_asof, 365) AS dormancy_capped,

  -- =========================================================================
  -- 5) SAFE TRANSFORMS FOR MODEL STABILITY
  -- =========================================================================
  LOG(1 + sales_filled) AS y_log1p,
  LOG(1 + COALESCE(lag_7, 0)) AS lag_7_log1p,
  LOG(1 + COALESCE(roll_mean_28, 0)) AS roll_mean_28_log1p,
  LOG(1 + COALESCE(roll_mean_pos_28, 0)) AS roll_mean_pos_28_log1p

FROM base b
LEFT JOIN closure_proximity cp ON b.date = cp.date;


-- =============================================================================
-- PART C: VALIDATION CHECKS
-- =============================================================================

-- V1: Verify sin/cos ranges within [-1, 1]
SELECT
  'sin_doy' AS feature,
  MIN(sin_doy) AS min_val,
  MAX(sin_doy) AS max_val,
  CASE WHEN MIN(sin_doy) >= -1 AND MAX(sin_doy) <= 1 THEN 'PASS' ELSE 'FAIL' END AS check
FROM `myforecastingsales.forecasting.gold_panel_features_v2`
UNION ALL
SELECT 'cos_doy', MIN(cos_doy), MAX(cos_doy),
  CASE WHEN MIN(cos_doy) >= -1 AND MAX(cos_doy) <= 1 THEN 'PASS' ELSE 'FAIL' END
FROM `myforecastingsales.forecasting.gold_panel_features_v2`
UNION ALL
SELECT 'sin_dow', MIN(sin_dow), MAX(sin_dow),
  CASE WHEN MIN(sin_dow) >= -1 AND MAX(sin_dow) <= 1 THEN 'PASS' ELSE 'FAIL' END
FROM `myforecastingsales.forecasting.gold_panel_features_v2`
UNION ALL
SELECT 'cos_dow', MIN(cos_dow), MAX(cos_dow),
  CASE WHEN MIN(cos_dow) >= -1 AND MAX(cos_dow) <= 1 THEN 'PASS' ELSE 'FAIL' END
FROM `myforecastingsales.forecasting.gold_panel_features_v2`;

-- V2: Verify closure proximity ranges
SELECT
  'days_to_next_closure' AS feature,
  MIN(days_to_next_closure) AS min_val,
  MAX(days_to_next_closure) AS max_val,
  CASE WHEN MIN(days_to_next_closure) >= 0 AND MAX(days_to_next_closure) <= 60 THEN 'PASS' ELSE 'FAIL' END AS check
FROM `myforecastingsales.forecasting.gold_panel_features_v2`
UNION ALL
SELECT 'days_from_prev_closure',
  MIN(days_from_prev_closure), MAX(days_from_prev_closure),
  CASE WHEN MIN(days_from_prev_closure) >= 0 AND MAX(days_from_prev_closure) <= 60 THEN 'PASS' ELSE 'FAIL' END
FROM `myforecastingsales.forecasting.gold_panel_features_v2`;

-- V3: Verify closure_week correctly flags ±3 days around closures
SELECT
  'is_closure_week on actual closures' AS check_name,
  COUNT(*) AS total_closure_days,
  COUNTIF(f.is_closure_week = 1) AS flagged_as_closure_week,
  CASE WHEN COUNTIF(f.is_closure_week = 1) = COUNT(*) THEN 'PASS' ELSE 'FAIL' END AS check
FROM `myforecastingsales.forecasting.gold_panel_features_v2` f
WHERE f.is_store_closed = 1;

-- V4: Summary stats for new features
SELECT
  COUNT(*) AS total_rows,
  ROUND(AVG(nz_rate_7), 4) AS avg_nz_rate_7,
  ROUND(AVG(roll_mean_pos_28), 4) AS avg_roll_mean_pos_28,
  ROUND(AVG(dormancy_capped), 2) AS avg_dormancy_capped,
  COUNTIF(is_closure_week = 1) AS closure_week_rows,
  ROUND(AVG(days_to_next_closure), 2) AS avg_days_to_next_closure
FROM `myforecastingsales.forecasting.gold_panel_features_v2`;

-- V5: Dormancy bucket distribution
SELECT
  dormancy_bucket,
  COUNT(*) AS n_rows,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
FROM `myforecastingsales.forecasting.gold_panel_features_v2`
GROUP BY dormancy_bucket
ORDER BY dormancy_bucket;


-- =============================================================================
-- PART D: UPDATED BASELINE EVALUATION WITH METRIC GUARDS
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.baseline_smart_pred_v2` AS
SELECT
  store_id,
  sku_id,
  date,
  tier_name,
  is_store_closed,
  days_since_last_sale_asof,
  dormancy_bucket,

  -- True value
  sales_filled AS y_true,

  -- Individual baseline predictions (non-negative)
  GREATEST(COALESCE(lag_7, 0), 0) AS pred_lag7,
  GREATEST(COALESCE(roll_mean_28, 0), 0) AS pred_roll28,

  -- Smart baseline: closure=0, active=roll28, sparse=lag7
  CASE
    WHEN is_store_closed = 1 THEN 0.0
    WHEN days_since_last_sale_asof <= 28 THEN GREATEST(COALESCE(roll_mean_28, 0), 0)
    ELSE GREATEST(COALESCE(lag_7, 0), 0)
  END AS pred_smart

FROM `myforecastingsales.forecasting.gold_panel_features_v2`
WHERE date BETWEEN '2025-07-03' AND '2025-12-17'
  AND tier_name IN ('T1_MATURE', 'T2_GROWING', 'T3_COLD_START');


-- =============================================================================
-- EVALUATION WITH METRIC GUARDS + ZEROACC + MAE_NONZERO
-- =============================================================================

-- E1: OVERALL METRICS (with guards)
SELECT
  'pred_lag7' AS model,
  COUNT(*) AS n_obs,
  SUM(y_true) AS sum_actual,
  ROUND(AVG(ABS(y_true - pred_lag7)), 4) AS MAE,
  -- WMAPE only if sum_actual > 0
  CASE WHEN SUM(y_true) > 0
    THEN ROUND(100.0 * SUM(ABS(y_true - pred_lag7)) / SUM(y_true), 2)
    ELSE NULL
  END AS WMAPE_pct,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_lag7), 2))), 4) AS RMSLE,
  -- ZeroAccuracy: correctly predict zeros
  ROUND(AVG(CASE WHEN y_true = 0 AND pred_lag7 < 0.5 THEN 1.0 ELSE 0.0 END), 4) AS ZeroAcc,
  -- MAE on non-zero actuals only
  ROUND(AVG(CASE WHEN y_true > 0 THEN ABS(y_true - pred_lag7) END), 4) AS MAE_nonzero
FROM `myforecastingsales.forecasting.baseline_smart_pred_v2`

UNION ALL

SELECT
  'pred_roll28', COUNT(*), SUM(y_true),
  ROUND(AVG(ABS(y_true - pred_roll28)), 4),
  CASE WHEN SUM(y_true) > 0 THEN ROUND(100.0 * SUM(ABS(y_true - pred_roll28)) / SUM(y_true), 2) ELSE NULL END,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_roll28), 2))), 4),
  ROUND(AVG(CASE WHEN y_true = 0 AND pred_roll28 < 0.5 THEN 1.0 ELSE 0.0 END), 4),
  ROUND(AVG(CASE WHEN y_true > 0 THEN ABS(y_true - pred_roll28) END), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v2`

UNION ALL

SELECT
  'pred_smart', COUNT(*), SUM(y_true),
  ROUND(AVG(ABS(y_true - pred_smart)), 4),
  CASE WHEN SUM(y_true) > 0 THEN ROUND(100.0 * SUM(ABS(y_true - pred_smart)) / SUM(y_true), 2) ELSE NULL END,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_smart), 2))), 4),
  ROUND(AVG(CASE WHEN y_true = 0 AND pred_smart < 0.5 THEN 1.0 ELSE 0.0 END), 4),
  ROUND(AVG(CASE WHEN y_true > 0 THEN ABS(y_true - pred_smart) END), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v2`

ORDER BY model;


-- E2: BY DORMANCY BUCKET (with metric guards)
SELECT
  dormancy_bucket,
  'pred_lag7' AS model,
  COUNT(*) AS n_obs,
  ROUND(SUM(y_true), 0) AS sum_actual,
  ROUND(AVG(ABS(y_true - pred_lag7)), 4) AS MAE,
  CASE WHEN SUM(y_true) > 0
    THEN ROUND(100.0 * SUM(ABS(y_true - pred_lag7)) / SUM(y_true), 2)
    ELSE NULL
  END AS WMAPE_pct,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_lag7), 2))), 4) AS RMSLE,
  ROUND(AVG(CASE WHEN y_true = 0 AND pred_lag7 < 0.5 THEN 1.0 ELSE 0.0 END), 4) AS ZeroAcc,
  ROUND(AVG(CASE WHEN y_true > 0 THEN ABS(y_true - pred_lag7) END), 4) AS MAE_nonzero
FROM `myforecastingsales.forecasting.baseline_smart_pred_v2`
GROUP BY dormancy_bucket

UNION ALL

SELECT
  dormancy_bucket, 'pred_roll28', COUNT(*), ROUND(SUM(y_true), 0),
  ROUND(AVG(ABS(y_true - pred_roll28)), 4),
  CASE WHEN SUM(y_true) > 0 THEN ROUND(100.0 * SUM(ABS(y_true - pred_roll28)) / SUM(y_true), 2) ELSE NULL END,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_roll28), 2))), 4),
  ROUND(AVG(CASE WHEN y_true = 0 AND pred_roll28 < 0.5 THEN 1.0 ELSE 0.0 END), 4),
  ROUND(AVG(CASE WHEN y_true > 0 THEN ABS(y_true - pred_roll28) END), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v2`
GROUP BY dormancy_bucket

UNION ALL

SELECT
  dormancy_bucket, 'pred_smart', COUNT(*), ROUND(SUM(y_true), 0),
  ROUND(AVG(ABS(y_true - pred_smart)), 4),
  CASE WHEN SUM(y_true) > 0 THEN ROUND(100.0 * SUM(ABS(y_true - pred_smart)) / SUM(y_true), 2) ELSE NULL END,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_smart), 2))), 4),
  ROUND(AVG(CASE WHEN y_true = 0 AND pred_smart < 0.5 THEN 1.0 ELSE 0.0 END), 4),
  ROUND(AVG(CASE WHEN y_true > 0 THEN ABS(y_true - pred_smart) END), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v2`
GROUP BY dormancy_bucket

ORDER BY dormancy_bucket, model;


-- E3: BY TIER (with metric guards)
SELECT
  tier_name,
  'pred_smart' AS model,
  COUNT(*) AS n_obs,
  ROUND(SUM(y_true), 0) AS sum_actual,
  ROUND(AVG(ABS(y_true - pred_smart)), 4) AS MAE,
  CASE WHEN SUM(y_true) > 0
    THEN ROUND(100.0 * SUM(ABS(y_true - pred_smart)) / SUM(y_true), 2)
    ELSE NULL
  END AS WMAPE_pct,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_smart), 2))), 4) AS RMSLE,
  ROUND(AVG(CASE WHEN y_true = 0 AND pred_smart < 0.5 THEN 1.0 ELSE 0.0 END), 4) AS ZeroAcc,
  ROUND(AVG(CASE WHEN y_true > 0 THEN ABS(y_true - pred_smart) END), 4) AS MAE_nonzero
FROM `myforecastingsales.forecasting.baseline_smart_pred_v2`
GROUP BY tier_name
ORDER BY CASE tier_name WHEN 'T1_MATURE' THEN 1 WHEN 'T2_GROWING' THEN 2 WHEN 'T3_COLD_START' THEN 3 END;
