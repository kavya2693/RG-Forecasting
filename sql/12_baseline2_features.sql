-- =============================================================================
-- BASELINE-2: FEATURE ENGINEERING WITH SPIKE + VELOCITY FEATURES
-- =============================================================================
-- Extends gold_panel_features_v1 with:
-- 1. Original 30 numeric features (CAUSAL)
-- 2. Spike features (5 new)
-- 3. Velocity features (4 new)
-- Total: 39 numeric features + 2 categorical
-- =============================================================================

-- =============================================================================
-- STEP 1: Pre-compute spike statistics (historical, no leakage)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.baseline2_spike_stats` AS
WITH daily_with_spike AS (
  SELECT
    store_id,
    sku_id,
    date,
    sales_filled,
    -- Is this day a spike? (sales > 2x rolling mean AND sales > 1)
    CASE
      WHEN sales_filled > 1
       AND sales_filled > 2 * AVG(sales_filled) OVER (
           PARTITION BY store_id, sku_id ORDER BY date
           ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
         )
      THEN 1 ELSE 0
    END AS is_spike
  FROM `myforecastingsales.forecasting.gold_panel_spine`
),
store_daily_spike AS (
  -- Per store-day: what % of SKUs are spiking?
  SELECT
    store_id,
    date,
    SAFE_DIVIDE(SUM(is_spike), COUNT(*)) AS store_spike_pct,
    CASE WHEN SAFE_DIVIDE(SUM(is_spike), COUNT(*)) > 0.15 THEN 1 ELSE 0 END AS is_promo_day
  FROM daily_with_spike
  GROUP BY store_id, date
),
sku_spike_history AS (
  -- Per SKU: historical spike probability
  SELECT
    sku_id,
    SAFE_DIVIDE(SUM(is_spike), COUNT(*)) AS historical_spike_prob
  FROM daily_with_spike
  WHERE date < '2025-06-26'  -- Use train data only
  GROUP BY sku_id
)
SELECT
  d.store_id,
  d.sku_id,
  d.date,
  d.is_spike,
  s.store_spike_pct,
  s.is_promo_day,
  h.historical_spike_prob,
  -- Had recent spike in last 7 days?
  MAX(d.is_spike) OVER (
    PARTITION BY d.store_id, d.sku_id ORDER BY d.date
    ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
  ) AS had_recent_spike
FROM daily_with_spike d
JOIN store_daily_spike s USING (store_id, date)
LEFT JOIN sku_spike_history h USING (sku_id);


-- =============================================================================
-- STEP 2: Pre-compute velocity statistics (gap patterns)
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.baseline2_velocity_stats` AS
WITH sale_gaps AS (
  -- For each sale, compute gap since previous sale
  SELECT
    store_id,
    sku_id,
    date,
    sales_filled,
    LAG(date) OVER (PARTITION BY store_id, sku_id ORDER BY date) AS prev_sale_date,
    DATE_DIFF(date, LAG(date) OVER (PARTITION BY store_id, sku_id ORDER BY date), DAY) AS gap_days
  FROM `myforecastingsales.forecasting.gold_panel_spine`
  WHERE sales_filled > 0
),
series_velocity AS (
  -- Per series: median gap, sale frequency
  SELECT
    store_id,
    sku_id,
    COUNT(*) AS n_sales,
    APPROX_QUANTILES(gap_days, 100)[OFFSET(50)] AS median_gap,
    SAFE_DIVIDE(COUNT(*), DATE_DIFF(MAX(date), MIN(date), DAY) + 1) AS sale_frequency
  FROM sale_gaps
  WHERE gap_days IS NOT NULL
  GROUP BY store_id, sku_id
)
SELECT
  g.store_id,
  g.sku_id,
  g.date,
  g.sales_filled,
  v.median_gap,
  v.sale_frequency,
  -- Velocity segment
  CASE
    WHEN v.sale_frequency >= 0.30 THEN 'FAST'
    WHEN v.sale_frequency >= 0.10 THEN 'MEDIUM'
    WHEN v.sale_frequency >= 0.03 THEN 'SLOW'
    ELSE 'VERY_SLOW'
  END AS velocity_segment
FROM `myforecastingsales.forecasting.gold_panel_spine` g
LEFT JOIN series_velocity v USING (store_id, sku_id);


-- =============================================================================
-- STEP 3: Create unified Baseline-2 feature table
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.gold_panel_features_baseline2`
PARTITION BY date
CLUSTER BY store_id, sku_id
AS
SELECT
  -- Keys
  f.store_id,
  f.sku_id,
  f.date,

  -- Target
  f.sales_filled AS y,

  -- =========================================================================
  -- ORIGINAL 30 FEATURES (from baseline-1)
  -- =========================================================================
  -- Calendar (5)
  f.dow,
  CASE WHEN f.is_weekend THEN 1 ELSE 0 END AS is_weekend,
  f.week_of_year,
  f.month,
  EXTRACT(DAYOFYEAR FROM f.date) AS day_of_year,

  -- Cyclical encodings (4)
  SIN(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM f.date) / 365) AS sin_doy,
  COS(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM f.date) / 365) AS cos_doy,
  SIN(2 * ACOS(-1) * f.dow / 7) AS sin_dow,
  COS(2 * ACOS(-1) * f.dow / 7) AS cos_dow,

  -- Store closure (4)
  CASE WHEN f.is_store_closed THEN 1 ELSE 0 END AS is_store_closed,
  COALESCE(f.days_to_next_closure, 999) AS days_to_next_closure,
  COALESCE(f.days_from_prev_closure, 999) AS days_from_prev_closure,
  CASE WHEN f.is_closure_week THEN 1 ELSE 0 END AS is_closure_week,

  -- Lag features (5) - CAUSAL
  f.lag_1,
  f.lag_7,
  f.lag_14,
  f.lag_28,
  f.lag_56,

  -- Rolling features (5) - CAUSAL (1 PRECEDING)
  f.roll_mean_7,
  f.roll_sum_7,
  f.roll_mean_28,
  f.roll_sum_28,
  f.roll_std_28,

  -- Non-zero rates (3)
  f.nz_rate_7,
  f.nz_rate_28,
  COALESCE(f.roll_mean_pos_28, 0) AS roll_mean_pos_28,

  -- Dormancy/recency (4) - CAUSAL (asof yesterday)
  f.days_since_last_sale_asof,
  LEAST(f.days_since_last_sale_asof, 365) AS dormancy_capped,
  f.zero_run_length_asof,
  COALESCE(f.last_sale_qty_asof, 0) AS last_sale_qty_asof,

  -- =========================================================================
  -- SPIKE FEATURES (5 new)
  -- =========================================================================
  COALESCE(sp.store_spike_pct, 0) AS feat_store_spike_pct,
  COALESCE(sp.is_promo_day, 0) AS feat_store_promo_day,
  -- Seasonal lift: avg sales in this week vs overall avg
  SAFE_DIVIDE(
    AVG(f.sales_filled) OVER (PARTITION BY f.sku_id, f.week_of_year),
    AVG(f.sales_filled) OVER (PARTITION BY f.sku_id)
  ) AS feat_seasonal_lift,
  COALESCE(sp.had_recent_spike, 0) AS feat_had_recent_spike,
  COALESCE(sp.historical_spike_prob, 0) AS feat_historical_spike_prob,

  -- =========================================================================
  -- VELOCITY FEATURES (4 new)
  -- =========================================================================
  COALESCE(v.sale_frequency, 0) AS feat_sale_frequency,
  -- Gap vs median: how many median gaps since last sale
  SAFE_DIVIDE(f.days_since_last_sale_asof, NULLIF(v.median_gap, 0)) AS feat_gap_vs_median,
  -- Is overdue: current gap > 1.5x median gap
  CASE
    WHEN f.days_since_last_sale_asof > 1.5 * COALESCE(v.median_gap, 9999) THEN 1
    ELSE 0
  END AS feat_is_overdue,
  -- Gap pressure: approaches 1 as gap approaches median
  1 - EXP(-0.5 * SAFE_DIVIDE(f.days_since_last_sale_asof, NULLIF(v.median_gap, 1))) AS feat_gap_pressure,

  -- Velocity segment as categorical
  COALESCE(v.velocity_segment, 'VERY_SLOW') AS feat_velocity_segment,

  -- =========================================================================
  -- TIER INFO (will be joined per-fold in training)
  -- =========================================================================
  f.tier_name,
  f.is_intermittent

FROM `myforecastingsales.forecasting.gold_panel_features_v1` f
LEFT JOIN `myforecastingsales.forecasting.baseline2_spike_stats` sp
  ON f.store_id = sp.store_id AND f.sku_id = sp.sku_id AND f.date = sp.date
LEFT JOIN `myforecastingsales.forecasting.baseline2_velocity_stats` v
  ON f.store_id = v.store_id AND f.sku_id = v.sku_id AND f.date = v.date;


-- =============================================================================
-- STEP 4: Create training views with per-fold tiers
-- =============================================================================

-- F1 Training View (uses tiers as-of 2025-06-26)
CREATE OR REPLACE VIEW `myforecastingsales.forecasting.v_trainval_baseline2_f1` AS
SELECT
  f.*,
  t.tier_name AS tier_name_perfold,
  t.is_intermittent AS is_intermittent_perfold
FROM `myforecastingsales.forecasting.gold_panel_features_baseline2` f
JOIN `myforecastingsales.forecasting.series_tiers_asof_20250626` t
  USING (store_id, sku_id)
WHERE t.tier_name = 'T1_MATURE'
  AND f.date BETWEEN '2019-01-02' AND '2025-12-17';

-- F2 Training View (uses tiers as-of 2025-03-10)
CREATE OR REPLACE VIEW `myforecastingsales.forecasting.v_trainval_baseline2_f2` AS
SELECT
  f.*,
  t.tier_name AS tier_name_perfold,
  t.is_intermittent AS is_intermittent_perfold
FROM `myforecastingsales.forecasting.gold_panel_features_baseline2` f
JOIN `myforecastingsales.forecasting.series_tiers_asof_20250310` t
  USING (store_id, sku_id)
WHERE t.tier_name = 'T1_MATURE'
  AND f.date BETWEEN '2019-01-02' AND '2025-08-31';

-- F3 Training View (uses tiers as-of 2024-11-07)
CREATE OR REPLACE VIEW `myforecastingsales.forecasting.v_trainval_baseline2_f3` AS
SELECT
  f.*,
  t.tier_name AS tier_name_perfold,
  t.is_intermittent AS is_intermittent_perfold
FROM `myforecastingsales.forecasting.gold_panel_features_baseline2` f
JOIN `myforecastingsales.forecasting.series_tiers_asof_20241107` t
  USING (store_id, sku_id)
WHERE t.tier_name = 'T1_MATURE'
  AND f.date BETWEEN '2019-01-02' AND '2025-04-30';


-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- V1: Feature counts
SELECT
  COUNT(*) AS total_rows,
  COUNTIF(feat_store_spike_pct > 0) AS rows_with_spike_pct,
  COUNTIF(feat_sale_frequency > 0) AS rows_with_velocity,
  COUNTIF(feat_gap_vs_median IS NOT NULL) AS rows_with_gap_ratio,
  AVG(feat_store_spike_pct) AS avg_spike_pct,
  AVG(feat_sale_frequency) AS avg_sale_freq
FROM `myforecastingsales.forecasting.gold_panel_features_baseline2`
WHERE tier_name = 'T1_MATURE';

-- V2: Velocity segment distribution
SELECT
  feat_velocity_segment,
  COUNT(*) AS row_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
FROM `myforecastingsales.forecasting.gold_panel_features_baseline2`
WHERE tier_name = 'T1_MATURE'
GROUP BY feat_velocity_segment
ORDER BY feat_velocity_segment;
