-- =============================================================================
-- BASELINE-2: PER-FOLD TIER TABLES (NO LEAKAGE)
-- =============================================================================
-- Creates separate tier tables for each fold cutoff date to prevent look-ahead bias.
--
-- Fold cutoffs:
--   F1: 2025-06-26 (train end)
--   F2: 2025-03-10 (train end)
--   F3: 2024-11-07 (train end)
--   G1: 2025-06-26 (T2)
--   G2: 2025-03-10 (T2)
--   C1: 2025-06-26 (T3)
--
-- Each tier table uses ONLY data up to that fold's cutoff date.
-- =============================================================================

-- =============================================================================
-- HELPER: Create tier table for a specific cutoff date
-- =============================================================================

-- F1 FOLD: As-of 2025-06-26
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.series_tiers_asof_20250626` AS
WITH series_stats AS (
  SELECT
    store_id,
    sku_id,
    MIN(date) AS first_date,
    MAX(CASE WHEN sales_filled > 0 THEN date END) AS last_pos_date,
    COUNT(*) AS history_days,
    COUNTIF(sales_filled > 0) AS n_pos_days,
    SUM(sales_filled) AS sum_sales,
    AVG(sales_filled) AS avg_sales_all_days,
    AVG(CASE WHEN sales_filled > 0 THEN sales_filled END) AS avg_sales_pos_days,
    SAFE_DIVIDE(COUNTIF(sales_filled > 0), COUNT(*)) AS nz_rate,
    -- ADI = Average Demand Interval
    SAFE_DIVIDE(COUNT(*), NULLIF(COUNTIF(sales_filled > 0), 0)) AS ADI,
    DATE_DIFF(DATE('2025-06-26'), MAX(CASE WHEN sales_filled > 0 THEN date END), DAY) AS days_since_last_sale
  FROM `myforecastingsales.forecasting.gold_panel_spine`
  WHERE date <= '2025-06-26'  -- ONLY data up to F1 cutoff
  GROUP BY store_id, sku_id
)
SELECT
  store_id,
  sku_id,
  CASE
    WHEN n_pos_days = 0 OR days_since_last_sale > 182 THEN 'T0_EXCLUDED'
    WHEN history_days >= 728 THEN 'T1_MATURE'
    WHEN history_days >= 182 THEN 'T2_GROWING'
    ELSE 'T3_COLD_START'
  END AS tier_name,
  history_days,
  n_pos_days,
  nz_rate,
  ADI,
  days_since_last_sale,
  sum_sales,
  IF(ADI >= 1.32, 1, 0) AS is_intermittent,
  '2025-06-26' AS asof_date
FROM series_stats;


-- F2 FOLD: As-of 2025-03-10
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.series_tiers_asof_20250310` AS
WITH series_stats AS (
  SELECT
    store_id,
    sku_id,
    MIN(date) AS first_date,
    MAX(CASE WHEN sales_filled > 0 THEN date END) AS last_pos_date,
    COUNT(*) AS history_days,
    COUNTIF(sales_filled > 0) AS n_pos_days,
    SUM(sales_filled) AS sum_sales,
    AVG(sales_filled) AS avg_sales_all_days,
    AVG(CASE WHEN sales_filled > 0 THEN sales_filled END) AS avg_sales_pos_days,
    SAFE_DIVIDE(COUNTIF(sales_filled > 0), COUNT(*)) AS nz_rate,
    SAFE_DIVIDE(COUNT(*), NULLIF(COUNTIF(sales_filled > 0), 0)) AS ADI,
    DATE_DIFF(DATE('2025-03-10'), MAX(CASE WHEN sales_filled > 0 THEN date END), DAY) AS days_since_last_sale
  FROM `myforecastingsales.forecasting.gold_panel_spine`
  WHERE date <= '2025-03-10'  -- ONLY data up to F2 cutoff
  GROUP BY store_id, sku_id
)
SELECT
  store_id,
  sku_id,
  CASE
    WHEN n_pos_days = 0 OR days_since_last_sale > 182 THEN 'T0_EXCLUDED'
    WHEN history_days >= 728 THEN 'T1_MATURE'
    WHEN history_days >= 182 THEN 'T2_GROWING'
    ELSE 'T3_COLD_START'
  END AS tier_name,
  history_days,
  n_pos_days,
  nz_rate,
  ADI,
  days_since_last_sale,
  sum_sales,
  IF(ADI >= 1.32, 1, 0) AS is_intermittent,
  '2025-03-10' AS asof_date
FROM series_stats;


-- F3 FOLD: As-of 2024-11-07
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.series_tiers_asof_20241107` AS
WITH series_stats AS (
  SELECT
    store_id,
    sku_id,
    MIN(date) AS first_date,
    MAX(CASE WHEN sales_filled > 0 THEN date END) AS last_pos_date,
    COUNT(*) AS history_days,
    COUNTIF(sales_filled > 0) AS n_pos_days,
    SUM(sales_filled) AS sum_sales,
    AVG(sales_filled) AS avg_sales_all_days,
    AVG(CASE WHEN sales_filled > 0 THEN sales_filled END) AS avg_sales_pos_days,
    SAFE_DIVIDE(COUNTIF(sales_filled > 0), COUNT(*)) AS nz_rate,
    SAFE_DIVIDE(COUNT(*), NULLIF(COUNTIF(sales_filled > 0), 0)) AS ADI,
    DATE_DIFF(DATE('2024-11-07'), MAX(CASE WHEN sales_filled > 0 THEN date END), DAY) AS days_since_last_sale
  FROM `myforecastingsales.forecasting.gold_panel_spine`
  WHERE date <= '2024-11-07'  -- ONLY data up to F3 cutoff
  GROUP BY store_id, sku_id
)
SELECT
  store_id,
  sku_id,
  CASE
    WHEN n_pos_days = 0 OR days_since_last_sale > 182 THEN 'T0_EXCLUDED'
    WHEN history_days >= 728 THEN 'T1_MATURE'
    WHEN history_days >= 182 THEN 'T2_GROWING'
    ELSE 'T3_COLD_START'
  END AS tier_name,
  history_days,
  n_pos_days,
  nz_rate,
  ADI,
  days_since_last_sale,
  sum_sales,
  IF(ADI >= 1.32, 1, 0) AS is_intermittent,
  '2024-11-07' AS asof_date
FROM series_stats;


-- =============================================================================
-- VALIDATION: Compare tier stability across folds
-- =============================================================================

-- V1: Tier counts per fold
SELECT 'F1 (2025-06-26)' AS fold, tier_name, COUNT(*) AS series_count
FROM `myforecastingsales.forecasting.series_tiers_asof_20250626`
GROUP BY tier_name
UNION ALL
SELECT 'F2 (2025-03-10)', tier_name, COUNT(*)
FROM `myforecastingsales.forecasting.series_tiers_asof_20250310`
GROUP BY tier_name
UNION ALL
SELECT 'F3 (2024-11-07)', tier_name, COUNT(*)
FROM `myforecastingsales.forecasting.series_tiers_asof_20241107`
GROUP BY tier_name
ORDER BY fold, tier_name;

-- V2: Tier migration between F3 and F1 (shows how many series changed tier)
SELECT
  f3.tier_name AS tier_f3,
  f1.tier_name AS tier_f1,
  COUNT(*) AS series_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
FROM `myforecastingsales.forecasting.series_tiers_asof_20241107` f3
JOIN `myforecastingsales.forecasting.series_tiers_asof_20250626` f1
  USING (store_id, sku_id)
GROUP BY f3.tier_name, f1.tier_name
ORDER BY f3.tier_name, f1.tier_name;
