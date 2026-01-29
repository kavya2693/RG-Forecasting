-- =============================================================================
-- STEP 1: INGEST RAW → BIGQUERY BRONZE TABLES
-- =============================================================================
-- Input: Raw CSVs from GCS
-- Output: sales_raw, sku_attr_raw (bronze tables)
-- =============================================================================

-- 1A: Load sales_raw
-- Run via: bq load or Python client
-- Schema: store_id STRING, item_id STRING, date DATE, sales INT64

-- 1B: Load sku_attr_raw
-- Run via: bq load or Python client
-- Schema: sku_id INT64, local_imported_attribute STRING

-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- Check row counts
SELECT 'sales_raw' AS table_name, COUNT(*) AS row_count
FROM `myforecastingsales.forecasting.sales_raw`
UNION ALL
SELECT 'sku_attr' AS table_name, COUNT(*) AS row_count
FROM `myforecastingsales.forecasting.sku_attr`;

-- Check null keys in sales_raw
SELECT
  COUNTIF(store_id IS NULL) AS null_store_id,
  COUNTIF(item_id IS NULL) AS null_item_id,
  COUNTIF(date IS NULL) AS null_date,
  COUNTIF(sales IS NULL) AS null_sales
FROM `myforecastingsales.forecasting.sales_raw`;

-- Check null keys in sku_attr
SELECT
  COUNTIF(sku_id IS NULL) AS null_sku_id,
  COUNTIF(local_imported_attribute IS NULL) AS null_attr
FROM `myforecastingsales.forecasting.sku_attr`;
