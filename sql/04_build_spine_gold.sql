-- =============================================================================
-- STEP 4: BUILD THE SPINE (GOLD)
-- =============================================================================
-- Creates complete daily grid for all store×sku pairs.
--
-- Rules:
-- 1. Activation: series activates on first date it appears
-- 2. No pre-activation rows (no fake pre-history)
-- 3. Post-activation: create row for every date from first_date to global_end_date
-- 4. Missing rows → sales_raw/sales_clean=NULL, sales_filled=0, is_observed=0
--
-- Tables Created:
-- A1. dim_dates_daily       - Calendar dimension
-- A2. active_pairs_store_sku - Unique pairs with first/last dates
-- A3. spine_store_sku_date  - Complete date grid per pair
-- A4. gold_panel_spine      - Final GOLD table (partitioned + clustered)
-- =============================================================================

-- =============================================================================
-- A1: dim_dates_daily - Calendar dimension table
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.dim_dates_daily` AS
WITH date_range AS (
  SELECT date
  FROM UNNEST(GENERATE_DATE_ARRAY('2019-01-02', '2025-12-17', INTERVAL 1 DAY)) AS date
)
SELECT
  date,
  EXTRACT(DAYOFWEEK FROM date) AS dow,  -- 1=Sunday, 7=Saturday
  EXTRACT(WEEK FROM date) AS week_of_year,
  EXTRACT(MONTH FROM date) AS month,
  EXTRACT(YEAR FROM date) AS year,
  CASE WHEN EXTRACT(DAYOFWEEK FROM date) IN (1, 7) THEN TRUE ELSE FALSE END AS is_weekend
FROM date_range;

-- =============================================================================
-- A2: active_pairs_store_sku - Unique store×sku pairs with first/last dates
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.active_pairs_store_sku` AS
SELECT
  store_id,
  sku_id,
  MIN(date) AS first_date,
  MAX(date) AS last_date,
  DATE_DIFF(MAX(date), MIN(date), DAY) + 1 AS history_days
FROM `myforecastingsales.forecasting.sales_daily_clean`
GROUP BY store_id, sku_id;

-- =============================================================================
-- A3: spine_store_sku_date - Complete date grid per pair (first_date → global_end_date)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.spine_store_sku_date` AS
WITH global_params AS (
  SELECT DATE('2025-12-17') AS global_end_date
)
SELECT
  p.store_id,
  p.sku_id,
  d.date
FROM `myforecastingsales.forecasting.active_pairs_store_sku` p
CROSS JOIN `myforecastingsales.forecasting.dim_dates_daily` d
CROSS JOIN global_params g
WHERE d.date >= p.first_date
  AND d.date <= g.global_end_date;

-- =============================================================================
-- A4: gold_panel_spine - Final GOLD table with all fields
-- Partitioned by date, clustered by store_id, sku_id
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.gold_panel_spine`
PARTITION BY date
CLUSTER BY store_id, sku_id
AS
SELECT
  -- Keys
  s.store_id,
  s.sku_id,
  s.date,

  -- Sales values
  c.sales_raw,                                          -- NULL if missing obs
  c.sales_clean,                                        -- NULL if missing obs
  COALESCE(c.sales_clean, 0) AS sales_filled,           -- Always >= 0

  -- Flags (default 0 for missing obs)
  COALESCE(c.was_negative, 0) AS was_negative,
  COALESCE(c.is_covid_period, 0) AS is_covid_period,
  COALESCE(c.is_covid_panic_spike, 0) AS is_covid_panic_spike,
  COALESCE(c.is_extreme_spike, 0) AS is_extreme_spike,

  -- Observation indicator
  CASE WHEN c.sales_clean IS NULL THEN 0 ELSE 1 END AS is_observed,

  -- Calendar fields from dim_dates_daily
  d.dow,
  d.week_of_year,
  d.month,
  d.year,
  d.is_weekend

FROM `myforecastingsales.forecasting.spine_store_sku_date` s
LEFT JOIN `myforecastingsales.forecasting.sales_daily_clean` c
  ON s.store_id = c.store_id
  AND s.sku_id = c.sku_id
  AND s.date = c.date
JOIN `myforecastingsales.forecasting.dim_dates_daily` d
  ON s.date = d.date;

-- =============================================================================
-- VALIDATION QUERIES (GATES)
-- =============================================================================

-- B1: Gold panel rowcount equals SUM over pairs of (global_end_date - first_date + 1)
SELECT
  (SELECT COUNT(*) FROM `myforecastingsales.forecasting.gold_panel_spine`) AS actual_rows,
  (SELECT SUM(DATE_DIFF(DATE('2025-12-17'), first_date, DAY) + 1)
   FROM `myforecastingsales.forecasting.active_pairs_store_sku`) AS expected_rows,
  CASE
    WHEN (SELECT COUNT(*) FROM `myforecastingsales.forecasting.gold_panel_spine`) =
         (SELECT SUM(DATE_DIFF(DATE('2025-12-17'), first_date, DAY) + 1)
          FROM `myforecastingsales.forecasting.active_pairs_store_sku`)
    THEN 'PASS' ELSE 'FAIL'
  END AS check_b1;

-- B2: No missing dates for each (store_id, sku_id) between first_date and global_end_date
WITH pair_date_counts AS (
  SELECT
    g.store_id,
    g.sku_id,
    COUNT(*) AS actual_days,
    DATE_DIFF(DATE('2025-12-17'), p.first_date, DAY) + 1 AS expected_days
  FROM `myforecastingsales.forecasting.gold_panel_spine` g
  JOIN `myforecastingsales.forecasting.active_pairs_store_sku` p
    ON g.store_id = p.store_id AND g.sku_id = p.sku_id
  GROUP BY g.store_id, g.sku_id, p.first_date
)
SELECT
  COUNTIF(actual_days != expected_days) AS pairs_with_missing_dates,
  COUNT(*) AS total_pairs,
  CASE WHEN COUNTIF(actual_days != expected_days) = 0 THEN 'PASS' ELSE 'FAIL' END AS check_b2
FROM pair_date_counts;

-- B3: For observed rows, gold_panel_spine.sales_clean equals sales_daily_clean.sales_clean exactly
SELECT
  COUNTIF(g.sales_clean != c.sales_clean) AS mismatched_rows,
  COUNT(*) AS total_observed_rows,
  CASE WHEN COUNTIF(g.sales_clean != c.sales_clean) = 0 THEN 'PASS' ELSE 'FAIL' END AS check_b3
FROM `myforecastingsales.forecasting.gold_panel_spine` g
JOIN `myforecastingsales.forecasting.sales_daily_clean` c
  ON g.store_id = c.store_id
  AND g.sku_id = c.sku_id
  AND g.date = c.date
WHERE g.is_observed = 1;

-- B4: sales_filled is never NULL and always >= 0
SELECT
  COUNTIF(sales_filled IS NULL) AS null_sales_filled,
  COUNTIF(sales_filled < 0) AS negative_sales_filled,
  COUNT(*) AS total_rows,
  CASE
    WHEN COUNTIF(sales_filled IS NULL) = 0 AND COUNTIF(sales_filled < 0) = 0
    THEN 'PASS' ELSE 'FAIL'
  END AS check_b4
FROM `myforecastingsales.forecasting.gold_panel_spine`;

-- B5: For missing observations: is_observed=0, sales_raw NULL, sales_clean NULL, sales_filled=0, flags=0
SELECT
  COUNTIF(is_observed = 0 AND sales_raw IS NOT NULL) AS bad_sales_raw,
  COUNTIF(is_observed = 0 AND sales_clean IS NOT NULL) AS bad_sales_clean,
  COUNTIF(is_observed = 0 AND sales_filled != 0) AS bad_sales_filled,
  COUNTIF(is_observed = 0 AND was_negative != 0) AS bad_was_negative,
  COUNTIF(is_observed = 0 AND is_covid_period != 0) AS bad_covid_period,
  COUNTIF(is_observed = 0 AND is_covid_panic_spike != 0) AS bad_covid_spike,
  COUNTIF(is_observed = 0 AND is_extreme_spike != 0) AS bad_extreme_spike,
  COUNTIF(is_observed = 0) AS total_missing_obs,
  CASE
    WHEN COUNTIF(is_observed = 0 AND sales_raw IS NOT NULL) = 0
     AND COUNTIF(is_observed = 0 AND sales_clean IS NOT NULL) = 0
     AND COUNTIF(is_observed = 0 AND sales_filled != 0) = 0
     AND COUNTIF(is_observed = 0 AND was_negative != 0) = 0
     AND COUNTIF(is_observed = 0 AND is_covid_period != 0) = 0
     AND COUNTIF(is_observed = 0 AND is_covid_panic_spike != 0) = 0
     AND COUNTIF(is_observed = 0 AND is_extreme_spike != 0) = 0
    THEN 'PASS' ELSE 'FAIL'
  END AS check_b5
FROM `myforecastingsales.forecasting.gold_panel_spine`;

-- B6: Summary stats
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT store_id) AS stores,
  COUNT(DISTINCT sku_id) AS skus,
  COUNT(DISTINCT date) AS days,
  MIN(date) AS min_date,
  MAX(date) AS max_date,
  SUM(is_observed) AS observed_rows,
  SUM(1 - is_observed) AS filled_rows,
  ROUND(100.0 * SUM(is_observed) / COUNT(*), 2) AS obs_rate_pct
FROM `myforecastingsales.forecasting.gold_panel_spine`;
