-- =============================================================================
-- STEP 3: APPLY CLEANING + FLAGS (SILVER+) — LOCKED
-- =============================================================================
-- Input: sales_daily
-- Output: sales_daily_clean with cleaning flags
-- =============================================================================

CREATE OR REPLACE TABLE `myforecastingsales.forecasting.sales_daily_clean` AS
WITH global_stats AS (
  SELECT APPROX_QUANTILES(saleqty, 10000)[OFFSET(9990)] AS p99_9
  FROM `myforecastingsales.forecasting.sales_daily`
  WHERE saleqty > 0
)
SELECT
  store_id,
  sku_id,
  date,
  saleqty AS sales_raw,

  -- Rule 1: Negative sales clipped to 0
  GREATEST(saleqty, 0) AS sales_clean,
  IF(saleqty < 0, 1, 0) AS was_negative,

  -- Rule 2a: COVID period flag (DIAGNOSTIC ONLY)
  IF(date BETWEEN "2020-03-15" AND "2021-06-30", 1, 0) AS is_covid_period,

  -- Rule 2b: COVID panic spike (FOR DOWNWEIGHT/CAP)
  IF(date BETWEEN "2020-03-15" AND "2021-06-30" AND saleqty > (SELECT p99_9 FROM global_stats), 1, 0) AS is_covid_panic_spike,

  -- Rule 3: Extreme spike (absolute threshold >10,000)
  IF(saleqty > 10000, 1, 0) AS is_extreme_spike

FROM `myforecastingsales.forecasting.sales_daily`, global_stats;

-- =============================================================================
-- VALIDATION QUERIES (GATES)
-- =============================================================================

-- Gate 1: UNIQUE KEYS
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT CONCAT(store_id, '|', sku_id, '|', CAST(date AS STRING))) AS distinct_keys,
  COUNT(*) = COUNT(DISTINCT CONCAT(store_id, '|', sku_id, '|', CAST(date AS STRING))) AS keys_unique
FROM `myforecastingsales.forecasting.sales_daily_clean`;

-- Gate 2: All sales_clean >= 0
SELECT
  COUNTIF(sales_clean < 0) AS negative_clean_count,
  MIN(sales_clean) AS min_sales_clean
FROM `myforecastingsales.forecasting.sales_daily_clean`;

-- Gate 3: Top 10 sales have is_extreme_spike=1
SELECT sales_clean, is_extreme_spike
FROM `myforecastingsales.forecasting.sales_daily_clean`
ORDER BY sales_clean DESC
LIMIT 10;

-- Gate 4: Flag counts
SELECT
  COUNT(*) AS total_rows,
  SUM(was_negative) AS was_negative_count,
  SUM(is_covid_period) AS is_covid_period_count,
  SUM(is_covid_panic_spike) AS is_covid_panic_spike_count,
  SUM(is_extreme_spike) AS is_extreme_spike_count
FROM `myforecastingsales.forecasting.sales_daily_clean`;
