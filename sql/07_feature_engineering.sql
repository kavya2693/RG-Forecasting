-- =============================================================================
-- STEP 7: FEATURE ENGINEERING (CAUSAL) + SMART BASELINE
-- =============================================================================
-- All features are CAUSAL: use only information up to date-1
-- No leakage: all windows use "ROWS BETWEEN X PRECEDING AND 1 PRECEDING"
-- Reusable for any fold/split without recompute
-- =============================================================================

-- =============================================================================
-- PART A: DAILY FEATURES TABLE (CAUSAL)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.gold_panel_features_v1`
PARTITION BY date
CLUSTER BY store_id, sku_id
AS
WITH base AS (
  SELECT
    -- Keys
    g.store_id,
    g.sku_id,
    g.date,

    -- Target
    g.sales_filled,
    g.is_observed,

    -- Flags
    g.was_negative,
    g.is_covid_period,
    g.is_covid_panic_spike,
    g.is_extreme_spike,

    -- Calendar
    g.dow,
    g.week_of_year,
    g.month,
    g.year,
    g.is_weekend,
    g.day_of_month,
    g.day_of_year,
    g.quarter,
    g.week_of_month,
    g.is_month_start,
    g.is_month_end,

    -- Closures
    g.is_store_closed,
    g.closure_name,

    -- Tier info (static as-of 2025-12-17)
    t.tier_name,
    t.is_intermittent,
    t.history_days AS series_history_days,
    t.nz_rate AS series_nz_rate,
    t.ADI AS series_ADI

  FROM `myforecastingsales.forecasting.gold_panel_spine_enriched` g
  JOIN `myforecastingsales.forecasting.series_tiers_asof_20251217` t
    USING (store_id, sku_id)
),

with_features AS (
  SELECT
    b.*,

    -- =========================================================================
    -- LAG FEATURES (CAUSAL: shifted by at least 1 day)
    -- =========================================================================
    LAG(sales_filled, 1) OVER w AS lag_1,
    LAG(sales_filled, 7) OVER w AS lag_7,
    LAG(sales_filled, 14) OVER w AS lag_14,
    LAG(sales_filled, 28) OVER w AS lag_28,
    LAG(sales_filled, 56) OVER w AS lag_56,

    -- =========================================================================
    -- ROLLING FEATURES (CAUSAL: exclude current day with "1 PRECEDING")
    -- =========================================================================
    -- Rolling mean (7-day and 28-day)
    AVG(sales_filled) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS roll_mean_7,

    AVG(sales_filled) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS roll_mean_28,

    -- Rolling sum (7-day and 28-day)
    SUM(sales_filled) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS roll_sum_7,

    SUM(sales_filled) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS roll_sum_28,

    -- Rolling std (28-day)
    STDDEV_POP(sales_filled) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS roll_std_28,

    -- Non-zero rate (28-day)
    AVG(CASE WHEN sales_filled > 0 THEN 1.0 ELSE 0.0 END) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS nz_rate_28,

    -- =========================================================================
    -- RECENCY / DORMANCY AS-OF DATE (DYNAMIC, CAUSAL)
    -- =========================================================================
    -- Last positive sale date (as of yesterday)
    MAX(IF(sales_filled > 0, date, NULL)) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS last_pos_date_asof,

    -- First date in series (for zero-run calculation)
    MIN(date) OVER (PARTITION BY store_id, sku_id) AS series_first_date,

    -- Last sale quantity (as of yesterday)
    LAST_VALUE(IF(sales_filled > 0, sales_filled, NULL) IGNORE NULLS) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS last_sale_qty_asof

  FROM base b
  WINDOW w AS (PARTITION BY store_id, sku_id ORDER BY date)
)

SELECT
  * EXCEPT(series_first_date, last_pos_date_asof),

  -- Keep last_pos_date_asof for reference
  last_pos_date_asof,

  -- Days since last sale (as-of date, dynamic)
  CASE
    WHEN last_pos_date_asof IS NULL THEN 999999
    ELSE DATE_DIFF(date, last_pos_date_asof, DAY)
  END AS days_since_last_sale_asof,

  -- Zero-run length as-of date
  CASE
    WHEN last_pos_date_asof IS NULL THEN DATE_DIFF(date, series_first_date, DAY)
    ELSE GREATEST(DATE_DIFF(date, last_pos_date_asof, DAY) - 1, 0)
  END AS zero_run_length_asof

FROM with_features;

-- =============================================================================
-- VALIDATION QUERIES FOR PART A
-- =============================================================================

-- V1: Confirm lag columns are NULL for early days (expected)
SELECT
  'lag_1' AS feature,
  COUNTIF(lag_1 IS NULL) AS null_count,
  COUNT(*) AS total_rows
FROM `myforecastingsales.forecasting.gold_panel_features_v1`
UNION ALL
SELECT 'lag_7', COUNTIF(lag_7 IS NULL), COUNT(*)
FROM `myforecastingsales.forecasting.gold_panel_features_v1`
UNION ALL
SELECT 'lag_28', COUNTIF(lag_28 IS NULL), COUNT(*)
FROM `myforecastingsales.forecasting.gold_panel_features_v1`
UNION ALL
SELECT 'roll_mean_28', COUNTIF(roll_mean_28 IS NULL), COUNT(*)
FROM `myforecastingsales.forecasting.gold_panel_features_v1`;

-- V2: Confirm features are causal (lag_1 on day N should equal sales_filled on day N-1)
SELECT
  COUNT(*) AS total_checked,
  COUNTIF(
    f1.lag_1 = f2.sales_filled OR (f1.lag_1 IS NULL AND f2.sales_filled IS NULL)
  ) AS correct_lag1,
  COUNTIF(
    f1.lag_1 != f2.sales_filled
  ) AS incorrect_lag1
FROM `myforecastingsales.forecasting.gold_panel_features_v1` f1
JOIN `myforecastingsales.forecasting.gold_panel_features_v1` f2
  ON f1.store_id = f2.store_id
  AND f1.sku_id = f2.sku_id
  AND f1.date = DATE_ADD(f2.date, INTERVAL 1 DAY)
WHERE f1.lag_1 IS NOT NULL;

-- V3: Summary stats
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT CONCAT(store_id, '|', sku_id)) AS total_series,
  MIN(date) AS min_date,
  MAX(date) AS max_date,
  ROUND(AVG(CASE WHEN lag_7 IS NOT NULL THEN 1 ELSE 0 END), 4) AS pct_with_lag7,
  ROUND(AVG(CASE WHEN roll_mean_28 IS NOT NULL THEN 1 ELSE 0 END), 4) AS pct_with_roll28
FROM `myforecastingsales.forecasting.gold_panel_features_v1`;


-- =============================================================================
-- PART B: SMART BASELINE PREDICTIONS (F1 FOLD: 2025-07-03 to 2025-12-17)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.baseline_smart_pred_v1` AS
SELECT
  store_id,
  sku_id,
  date,
  tier_name,
  is_store_closed,
  days_since_last_sale_asof,

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
  END AS pred_smart,

  -- Dormancy bucket for analysis
  CASE
    WHEN days_since_last_sale_asof <= 28 THEN '1_0-28d'
    WHEN days_since_last_sale_asof <= 56 THEN '2_29-56d'
    WHEN days_since_last_sale_asof <= 182 THEN '3_57-182d'
    ELSE '4_>182d'
  END AS dormancy_bucket

FROM `myforecastingsales.forecasting.gold_panel_features_v1`
WHERE date BETWEEN '2025-07-03' AND '2025-12-17'
  AND tier_name IN ('T1_MATURE', 'T2_GROWING', 'T3_COLD_START');


-- =============================================================================
-- PART C: EVALUATION REPORTS
-- =============================================================================

-- C1: OVERALL METRICS
SELECT
  'pred_lag7' AS model,
  COUNT(*) AS n_obs,
  ROUND(AVG(ABS(y_true - pred_lag7)), 4) AS MAE,
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_lag7)), SUM(y_true)), 2) AS WMAPE_pct,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_lag7), 2))), 4) AS RMSLE
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`

UNION ALL

SELECT
  'pred_roll28' AS model,
  COUNT(*),
  ROUND(AVG(ABS(y_true - pred_roll28)), 4),
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_roll28)), SUM(y_true)), 2),
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_roll28), 2))), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`

UNION ALL

SELECT
  'pred_smart' AS model,
  COUNT(*),
  ROUND(AVG(ABS(y_true - pred_smart)), 4),
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_smart)), SUM(y_true)), 2),
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_smart), 2))), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`;


-- C2: METRICS BY TIER
SELECT
  tier_name,
  'pred_lag7' AS model,
  COUNT(*) AS n_obs,
  ROUND(AVG(ABS(y_true - pred_lag7)), 4) AS MAE,
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_lag7)), SUM(y_true)), 2) AS WMAPE_pct,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_lag7), 2))), 4) AS RMSLE
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`
GROUP BY tier_name

UNION ALL

SELECT
  tier_name,
  'pred_roll28',
  COUNT(*),
  ROUND(AVG(ABS(y_true - pred_roll28)), 4),
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_roll28)), SUM(y_true)), 2),
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_roll28), 2))), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`
GROUP BY tier_name

UNION ALL

SELECT
  tier_name,
  'pred_smart',
  COUNT(*),
  ROUND(AVG(ABS(y_true - pred_smart)), 4),
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_smart)), SUM(y_true)), 2),
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_smart), 2))), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`
GROUP BY tier_name

ORDER BY
  CASE tier_name
    WHEN 'T1_MATURE' THEN 1
    WHEN 'T2_GROWING' THEN 2
    WHEN 'T3_COLD_START' THEN 3
  END,
  model;


-- C3: METRICS BY DORMANCY BUCKET
SELECT
  dormancy_bucket,
  'pred_lag7' AS model,
  COUNT(*) AS n_obs,
  ROUND(AVG(ABS(y_true - pred_lag7)), 4) AS MAE,
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_lag7)), SUM(y_true)), 2) AS WMAPE_pct,
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_lag7), 2))), 4) AS RMSLE
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`
GROUP BY dormancy_bucket

UNION ALL

SELECT
  dormancy_bucket,
  'pred_roll28',
  COUNT(*),
  ROUND(AVG(ABS(y_true - pred_roll28)), 4),
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_roll28)), SUM(y_true)), 2),
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_roll28), 2))), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`
GROUP BY dormancy_bucket

UNION ALL

SELECT
  dormancy_bucket,
  'pred_smart',
  COUNT(*),
  ROUND(AVG(ABS(y_true - pred_smart)), 4),
  ROUND(100.0 * SAFE_DIVIDE(SUM(ABS(y_true - pred_smart)), SUM(y_true)), 2),
  ROUND(SQRT(AVG(POW(LOG(1 + y_true) - LOG(1 + pred_smart), 2))), 4)
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`
GROUP BY dormancy_bucket

ORDER BY dormancy_bucket, model;


-- C4: CLOSURE SANITY CHECK (y_true=0 and pred_smart=0 on closure days)
SELECT
  'Closure Sanity Check' AS check_name,
  COUNT(*) AS closure_rows,
  COUNTIF(y_true = 0) AS y_true_zero,
  COUNTIF(pred_smart = 0) AS pred_smart_zero,
  CASE
    WHEN COUNTIF(y_true = 0) = COUNT(*) AND COUNTIF(pred_smart = 0) = COUNT(*)
    THEN 'PASS' ELSE 'FAIL'
  END AS check_result
FROM `myforecastingsales.forecasting.baseline_smart_pred_v1`
WHERE is_store_closed = 1;
