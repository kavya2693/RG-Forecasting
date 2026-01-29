-- =============================================================================
-- STEP 2: DEDUPLICATE/AGGREGATE → DAILY GRAIN (SILVER)
-- =============================================================================
-- Input: sales_raw
-- Output: sales_daily at (store_id, sku_id, date) grain
-- =============================================================================

CREATE OR REPLACE TABLE `myforecastingsales.forecasting.sales_daily` AS
SELECT
  CAST(store_id AS STRING) AS store_id,
  CAST(item_id AS STRING) AS sku_id,
  DATE(date) AS date,
  CAST(SUM(sales) AS INT64) AS saleqty
FROM `myforecastingsales.forecasting.sales_raw`
GROUP BY store_id, item_id, date;

-- =============================================================================
-- VALIDATION QUERIES (GATES)
-- =============================================================================

-- Gate: COUNT(*) == COUNT(DISTINCT store_id, sku_id, date)
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT CONCAT(store_id, '|', sku_id, '|', CAST(date AS STRING))) AS distinct_keys,
  COUNT(*) = COUNT(DISTINCT CONCAT(store_id, '|', sku_id, '|', CAST(date AS STRING))) AS keys_unique
FROM `myforecastingsales.forecasting.sales_daily`;

-- Gate: No null keys
SELECT
  COUNTIF(store_id IS NULL) AS null_store_id,
  COUNTIF(sku_id IS NULL) AS null_sku_id,
  COUNTIF(date IS NULL) AS null_date
FROM `myforecastingsales.forecasting.sales_daily`;
