-- =============================================================================
-- STEP 5: PRE-TIERING FEATURES
-- =============================================================================
-- Creates calendar dimension, closure calendar, enriched spine, and series stats
-- Must run AFTER Step 4 (Build Spine Gold)
--
-- Tables Created:
-- A. dim_dates_daily          - Extended calendar (2019-01-02 to 2026-06-03)
-- B. dim_store_closures_manual - 24 exact closure dates (NewYear, Christmas, GoodFriday)
-- C. gold_panel_spine_enriched - Spine with calendar + closure flags
-- D. series_stats_pre_tiering  - One row per store×sku for tiering
-- =============================================================================

-- =============================================================================
-- PART A: dim_dates_daily (2019-01-02 to 2026-06-03)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.dim_dates_daily` AS
WITH date_range AS (
  SELECT date
  FROM UNNEST(GENERATE_DATE_ARRAY('2019-01-02', '2026-06-03', INTERVAL 1 DAY)) AS date
)
SELECT
  date,
  CAST(FORMAT_DATE('%Y%m%d', date) AS INT64) AS date_id,
  EXTRACT(DAYOFWEEK FROM date) AS dow,  -- 1=Sunday, 7=Saturday
  EXTRACT(WEEK FROM date) AS week_of_year,
  EXTRACT(MONTH FROM date) AS month,
  EXTRACT(YEAR FROM date) AS year,
  CASE WHEN EXTRACT(DAYOFWEEK FROM date) IN (1, 7) THEN TRUE ELSE FALSE END AS is_weekend,
  EXTRACT(DAY FROM date) AS day_of_month,
  EXTRACT(DAYOFYEAR FROM date) AS day_of_year,
  EXTRACT(QUARTER FROM date) AS quarter,
  DATE_TRUNC(date, WEEK(MONDAY)) AS week_start_date,
  DATE_ADD(DATE_TRUNC(date, WEEK(MONDAY)), INTERVAL 6 DAY) AS week_end_date,
  1 + CAST(FLOOR((EXTRACT(DAY FROM date) - 1) / 7) AS INT64) AS week_of_month,
  EXTRACT(DAY FROM date) = 1 AS is_month_start,
  date = LAST_DAY(date) AS is_month_end,
  EXTRACT(MONTH FROM date) IN (1, 4, 7, 10) AND EXTRACT(DAY FROM date) = 1 AS is_quarter_start,
  (EXTRACT(MONTH FROM date) IN (3, 6, 9, 12) AND date = LAST_DAY(date)) AS is_quarter_end,
  EXTRACT(MONTH FROM date) = 1 AND EXTRACT(DAY FROM date) = 1 AS is_year_start,
  EXTRACT(MONTH FROM date) = 12 AND EXTRACT(DAY FROM date) = 31 AS is_year_end,
  (MOD(EXTRACT(YEAR FROM date), 400) = 0 OR
   (MOD(EXTRACT(YEAR FROM date), 4) = 0 AND MOD(EXTRACT(YEAR FROM date), 100) != 0)) AS is_leap_year,
  (EXTRACT(MONTH FROM date) = 2 AND EXTRACT(DAY FROM date) = 29) AS is_feb29
FROM date_range;

-- =============================================================================
-- PART B: dim_store_closures_manual (24 exact dates)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.dim_store_closures_manual` AS

-- New Year: Jan 1 for years 2019-2026 (8 rows)
SELECT DATE('2019-01-01') AS closure_date, 'NewYear' AS closure_name, 1 AS is_store_closed
UNION ALL SELECT DATE('2020-01-01'), 'NewYear', 1
UNION ALL SELECT DATE('2021-01-01'), 'NewYear', 1
UNION ALL SELECT DATE('2022-01-01'), 'NewYear', 1
UNION ALL SELECT DATE('2023-01-01'), 'NewYear', 1
UNION ALL SELECT DATE('2024-01-01'), 'NewYear', 1
UNION ALL SELECT DATE('2025-01-01'), 'NewYear', 1
UNION ALL SELECT DATE('2026-01-01'), 'NewYear', 1

UNION ALL

-- Christmas: Dec 25 for years 2019-2026 (8 rows)
SELECT DATE('2019-12-25'), 'Christmas', 1
UNION ALL SELECT DATE('2020-12-25'), 'Christmas', 1
UNION ALL SELECT DATE('2021-12-25'), 'Christmas', 1
UNION ALL SELECT DATE('2022-12-25'), 'Christmas', 1
UNION ALL SELECT DATE('2023-12-25'), 'Christmas', 1
UNION ALL SELECT DATE('2024-12-25'), 'Christmas', 1
UNION ALL SELECT DATE('2025-12-25'), 'Christmas', 1
UNION ALL SELECT DATE('2026-12-25'), 'Christmas', 1

UNION ALL

-- Good Friday: forced exact dates (8 rows) - NO EASTER ALGORITHM
SELECT DATE('2019-04-19'), 'GoodFriday', 1
UNION ALL SELECT DATE('2020-04-10'), 'GoodFriday', 1
UNION ALL SELECT DATE('2021-04-02'), 'GoodFriday', 1
UNION ALL SELECT DATE('2022-04-15'), 'GoodFriday', 1
UNION ALL SELECT DATE('2023-04-07'), 'GoodFriday', 1
UNION ALL SELECT DATE('2024-03-29'), 'GoodFriday', 1
UNION ALL SELECT DATE('2025-04-18'), 'GoodFriday', 1
UNION ALL SELECT DATE('2026-04-03'), 'GoodFriday', 1;

-- =============================================================================
-- PART C: gold_panel_spine_enriched
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.gold_panel_spine_enriched`
PARTITION BY date
CLUSTER BY store_id, sku_id
AS
SELECT
  -- Keys
  g.store_id,
  g.sku_id,
  g.date,

  -- Sales values
  g.sales_raw,
  g.sales_clean,
  g.sales_filled,

  -- Observation indicator
  g.is_observed,

  -- Flags
  g.was_negative,
  g.is_covid_period,
  g.is_covid_panic_spike,
  g.is_extreme_spike,

  -- Calendar fields from dim_dates_daily
  d.date_id,
  d.dow,
  d.week_of_year,
  d.month,
  d.year,
  d.is_weekend,
  d.day_of_month,
  d.day_of_year,
  d.quarter,
  d.week_start_date,
  d.week_end_date,
  d.week_of_month,
  d.is_month_start,
  d.is_month_end,
  d.is_quarter_start,
  d.is_quarter_end,
  d.is_year_start,
  d.is_year_end,
  d.is_leap_year,
  d.is_feb29,

  -- Closure fields
  COALESCE(c.is_store_closed, 0) AS is_store_closed,
  c.closure_name

FROM `myforecastingsales.forecasting.gold_panel_spine` g
JOIN `myforecastingsales.forecasting.dim_dates_daily` d
  ON g.date = d.date
LEFT JOIN `myforecastingsales.forecasting.dim_store_closures_manual` c
  ON g.date = c.closure_date;

-- =============================================================================
-- PART D: series_stats_pre_tiering (one row per store×sku)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.series_stats_pre_tiering` AS
WITH
-- Filter to history only
history AS (
  SELECT *
  FROM `myforecastingsales.forecasting.gold_panel_spine_enriched`
  WHERE date <= '2025-12-17'
),

-- Basic stats per series
basic_stats AS (
  SELECT
    store_id,
    sku_id,
    MIN(date) AS start_date_in_spine,
    MAX(date) AS end_date_in_spine,
    MIN(IF(sales_filled > 0, date, NULL)) AS first_pos_date,
    MAX(IF(sales_filled > 0, date, NULL)) AS last_pos_date,
    COUNT(*) AS n_days_total,
    COUNTIF(sales_filled > 0) AS n_pos_days,
    SUM(sales_filled) AS sum_sales,
    AVG(sales_filled) AS avg_sales_all_days,
    MAX(sales_filled) AS max_sales
  FROM history
  GROUP BY store_id, sku_id
),

-- Positive-day stats
pos_stats AS (
  SELECT
    store_id,
    sku_id,
    AVG(sales_filled) AS avg_sales_pos_days,
    STDDEV_POP(sales_filled) AS stddev_sales_pos_days,
    APPROX_QUANTILES(sales_filled, 100)[OFFSET(50)] AS p50_sales_pos_days,
    APPROX_QUANTILES(sales_filled, 100)[OFFSET(90)] AS p90_sales_pos_days
  FROM history
  WHERE sales_filled > 0
  GROUP BY store_id, sku_id
),

-- Zero-run calculation using gaps-and-islands
with_grp AS (
  SELECT
    store_id,
    sku_id,
    date,
    sales_filled,
    SUM(IF(sales_filled > 0, 1, 0)) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
    ) AS grp
  FROM history
),

zero_runs AS (
  SELECT
    store_id,
    sku_id,
    MAX(run_length) AS max_zero_run_length
  FROM (
    SELECT
      store_id,
      sku_id,
      grp,
      COUNT(*) AS run_length
    FROM with_grp
    WHERE sales_filled = 0
    GROUP BY store_id, sku_id, grp
  )
  GROUP BY store_id, sku_id
),

-- Last zero run
last_pos_per_series AS (
  SELECT store_id, sku_id, MAX(IF(sales_filled > 0, date, NULL)) AS last_pos_date
  FROM history
  GROUP BY store_id, sku_id
),

last_zero_run AS (
  SELECT
    h.store_id,
    h.sku_id,
    COUNTIF(h.sales_filled = 0 AND h.date > lp.last_pos_date) AS last_zero_run_length
  FROM history h
  LEFT JOIN last_pos_per_series lp ON h.store_id = lp.store_id AND h.sku_id = lp.sku_id
  GROUP BY h.store_id, h.sku_id
)

SELECT
  b.store_id,
  b.sku_id,

  -- Lifecycle
  b.start_date_in_spine,
  b.end_date_in_spine,
  b.first_pos_date,
  b.last_pos_date,
  DATE_DIFF(b.end_date_in_spine, b.start_date_in_spine, DAY) + 1 AS history_days,
  b.n_days_total,
  b.n_pos_days,
  SAFE_DIVIDE(b.n_pos_days, b.n_days_total) AS nz_rate,
  IF(b.last_pos_date IS NULL, NULL, DATE_DIFF(b.end_date_in_spine, b.last_pos_date, DAY)) AS days_since_last_sale,

  -- Magnitude
  b.sum_sales,
  ROUND(b.avg_sales_all_days, 4) AS avg_sales_all_days,
  ROUND(p.avg_sales_pos_days, 4) AS avg_sales_pos_days,
  p.p50_sales_pos_days,
  p.p90_sales_pos_days,
  b.max_sales,

  -- Intermittency
  SAFE_DIVIDE(b.n_days_total, NULLIF(b.n_pos_days, 0)) AS ADI,
  POW(SAFE_DIVIDE(p.stddev_sales_pos_days, NULLIF(p.avg_sales_pos_days, 0)), 2) AS CV2,
  IF(SAFE_DIVIDE(b.n_days_total, NULLIF(b.n_pos_days, 0)) >= 1.32, 1, 0) AS intermittent_flag,

  -- Zero runs
  COALESCE(z.max_zero_run_length, 0) AS max_zero_run_length,
  COALESCE(lz.last_zero_run_length, 0) AS last_zero_run_length

FROM basic_stats b
LEFT JOIN pos_stats p ON b.store_id = p.store_id AND b.sku_id = p.sku_id
LEFT JOIN zero_runs z ON b.store_id = z.store_id AND b.sku_id = z.sku_id
LEFT JOIN last_zero_run lz ON b.store_id = lz.store_id AND b.sku_id = lz.sku_id;

-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- A: dim_dates_daily
SELECT
  COUNT(*) AS total_rows,
  DATE_DIFF(DATE('2026-06-03'), DATE('2019-01-02'), DAY) + 1 AS expected_rows,
  COUNTIF(is_feb29) AS feb29_count
FROM `myforecastingsales.forecasting.dim_dates_daily`;

-- B: dim_store_closures_manual
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT closure_date) AS distinct_dates
FROM `myforecastingsales.forecasting.dim_store_closures_manual`;

-- C: gold_panel_spine_enriched
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT date) AS dates,
  COUNTIF(is_store_closed = 1) AS closure_rows
FROM `myforecastingsales.forecasting.gold_panel_spine_enriched`;

-- D: series_stats_pre_tiering
SELECT
  COUNT(*) AS total_series,
  COUNTIF(intermittent_flag = 1) AS intermittent_series,
  ROUND(AVG(nz_rate), 4) AS avg_nz_rate,
  ROUND(AVG(ADI), 2) AS avg_ADI
FROM `myforecastingsales.forecasting.series_stats_pre_tiering`;

-- =============================================================================
-- FORECAST OVERRIDE TEMPLATE (for Step 12)
-- =============================================================================
-- Apply to forecast_raw table with (store_id, sku_id, date, yhat_model):
--
-- SELECT
--   f.store_id,
--   f.sku_id,
--   f.date,
--   f.yhat_model,
--   CASE WHEN c.is_store_closed = 1 THEN 0.0 ELSE f.yhat_model END AS yhat_final,
--   COALESCE(c.is_store_closed, 0) AS is_store_closed,
--   c.closure_name
-- FROM forecast_raw f
-- LEFT JOIN `myforecastingsales.forecasting.dim_store_closures_manual` c
--   ON f.date = c.closure_date;
