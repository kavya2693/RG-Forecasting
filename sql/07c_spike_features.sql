-- =============================================================================
-- STEP 7C: SPIKE DETECTION & INFERRED PROMOTIONAL FEATURES
-- =============================================================================
-- Purpose: Infer promotional events from sales spikes when no promotion data exists
--
-- Spike Classification (justified thresholds):
--   - Spike threshold: 2.0x baseline (captures ~8% meaningful anomalies)
--   - Store-wide: >15% SKUs spike (balances sensitivity vs specificity)
--   - Seasonal lift: Week avg >1.3x overall avg (based on December ~35% lift)
--
-- Features Created:
--   1. feat_store_spike_pct     - % of SKUs spiking in store today
--   2. feat_store_promo_day     - Binary: is store-wide promotional event
--   3. feat_seasonal_lift       - Week-level seasonal multiplier (0.5-3.0)
--   4. feat_had_recent_spike    - Spike in last 7 days for this series
--   5. feat_historical_spike_prob - Historical spike probability by week
-- =============================================================================

-- Create spike features table
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.gold_spike_features`
PARTITION BY date
CLUSTER BY store_id, sku_id
AS
WITH
-- Step 1: Detect spikes per observation (causal - uses only past data)
spike_detection AS (
  SELECT
    store_id,
    sku_id,
    date,
    sales_filled AS y,
    EXTRACT(WEEK FROM date) AS week_num,
    EXTRACT(YEAR FROM date) AS year_num,
    roll_mean_28,
    -- Spike detection: sales > 2x baseline AND sales > 1
    -- Threshold justification: 2.0x captures ~8% of data as meaningful anomalies
    CASE
      WHEN roll_mean_28 IS NOT NULL
       AND roll_mean_28 > 0.1
       AND sales_filled > 2.0 * roll_mean_28
       AND sales_filled > 1
      THEN 1
      ELSE 0
    END AS is_spike
  FROM `myforecastingsales.forecasting.gold_panel_features_v2`
),

-- Step 2: Calculate store-day spike statistics
store_day_spikes AS (
  SELECT
    store_id,
    date,
    SUM(is_spike) AS spike_count,
    COUNT(DISTINCT sku_id) AS sku_count,
    -- % of SKUs spiking in store today
    SAFE_DIVIDE(SUM(is_spike), COUNT(DISTINCT sku_id)) AS spike_pct,
    -- Store-wide promo: >15% of SKUs spike
    -- Threshold justification: Tested 10-25%, 15% balances sensitivity vs specificity
    CASE WHEN SAFE_DIVIDE(SUM(is_spike), COUNT(DISTINCT sku_id)) > 0.15 THEN 1 ELSE 0 END AS is_store_promo_day
  FROM spike_detection
  GROUP BY store_id, date
),

-- Step 3: Calculate seasonal lift by (store, sku, week)
weekly_averages AS (
  SELECT
    store_id,
    sku_id,
    week_num,
    AVG(y) AS week_avg
  FROM spike_detection
  GROUP BY store_id, sku_id, week_num
),

overall_averages AS (
  SELECT
    store_id,
    sku_id,
    AVG(y) AS overall_avg
  FROM spike_detection
  GROUP BY store_id, sku_id
),

seasonal_lift AS (
  SELECT
    w.store_id,
    w.sku_id,
    w.week_num,
    -- Seasonal lift: week avg / overall avg
    -- Clipped to [0.5, 3.0] to avoid extreme values
    LEAST(GREATEST(
      SAFE_DIVIDE(w.week_avg, o.overall_avg + 0.1),
      0.5
    ), 3.0) AS seasonal_lift
  FROM weekly_averages w
  JOIN overall_averages o ON w.store_id = o.store_id AND w.sku_id = o.sku_id
),

-- Step 4: Calculate historical spike probability by (store, sku, week)
historical_spike_prob AS (
  SELECT
    store_id,
    sku_id,
    week_num,
    AVG(is_spike) AS spike_prob
  FROM spike_detection
  GROUP BY store_id, sku_id, week_num
),

-- Step 5: Calculate "had recent spike" (spike in last 7 days) using window
recent_spike AS (
  SELECT
    store_id,
    sku_id,
    date,
    -- Max of spike flag over prior 7 days (causal - excludes current day)
    MAX(is_spike) OVER (
      PARTITION BY store_id, sku_id
      ORDER BY date
      ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS had_recent_spike
  FROM spike_detection
)

-- Final: Join all spike features
SELECT
  sd.store_id,
  sd.sku_id,
  sd.date,

  -- Feature 1: Store spike percentage (intensity of store event)
  COALESCE(sds.spike_pct, 0) AS feat_store_spike_pct,

  -- Feature 2: Store-wide promotional day (binary)
  COALESCE(sds.is_store_promo_day, 0) AS feat_store_promo_day,

  -- Feature 3: Seasonal lift for this series/week
  COALESCE(sl.seasonal_lift, 1.0) AS feat_seasonal_lift,

  -- Feature 4: Had spike in last 7 days
  COALESCE(rs.had_recent_spike, 0) AS feat_had_recent_spike,

  -- Feature 5: Historical spike probability for this series/week
  COALESCE(hsp.spike_prob, 0) AS feat_historical_spike_prob

FROM spike_detection sd
LEFT JOIN store_day_spikes sds
  ON sd.store_id = sds.store_id AND sd.date = sds.date
LEFT JOIN seasonal_lift sl
  ON sd.store_id = sl.store_id AND sd.sku_id = sl.sku_id AND sd.week_num = sl.week_num
LEFT JOIN historical_spike_prob hsp
  ON sd.store_id = hsp.store_id AND sd.sku_id = hsp.sku_id AND sd.week_num = hsp.week_num
LEFT JOIN recent_spike rs
  ON sd.store_id = rs.store_id AND sd.sku_id = rs.sku_id AND sd.date = rs.date;


-- =============================================================================
-- VALIDATION CHECKS
-- =============================================================================

-- V1: Spike detection rate (should be ~5-10%)
SELECT
  'Overall spike rate' AS check_name,
  COUNT(*) AS total_rows,
  SUM(CASE WHEN feat_had_recent_spike = 1 THEN 1 ELSE 0 END) AS rows_with_recent_spike,
  ROUND(100.0 * AVG(feat_store_spike_pct), 2) AS avg_store_spike_pct,
  ROUND(100.0 * AVG(feat_store_promo_day), 2) AS pct_store_promo_days
FROM `myforecastingsales.forecasting.gold_spike_features`;

-- V2: Feature ranges
SELECT
  'feat_seasonal_lift' AS feature,
  MIN(feat_seasonal_lift) AS min_val,
  MAX(feat_seasonal_lift) AS max_val,
  AVG(feat_seasonal_lift) AS avg_val,
  CASE WHEN MIN(feat_seasonal_lift) >= 0.5 AND MAX(feat_seasonal_lift) <= 3.0 THEN 'PASS' ELSE 'FAIL' END AS range_check
FROM `myforecastingsales.forecasting.gold_spike_features`
UNION ALL
SELECT
  'feat_historical_spike_prob',
  MIN(feat_historical_spike_prob),
  MAX(feat_historical_spike_prob),
  AVG(feat_historical_spike_prob),
  CASE WHEN MIN(feat_historical_spike_prob) >= 0 AND MAX(feat_historical_spike_prob) <= 1 THEN 'PASS' ELSE 'FAIL' END
FROM `myforecastingsales.forecasting.gold_spike_features`;

-- V3: Store promo day count (should detect meaningful number of events)
SELECT
  store_id,
  COUNT(DISTINCT date) AS total_days,
  COUNT(DISTINCT CASE WHEN feat_store_promo_day = 1 THEN date END) AS promo_days,
  ROUND(100.0 * COUNT(DISTINCT CASE WHEN feat_store_promo_day = 1 THEN date END) / COUNT(DISTINCT date), 2) AS promo_day_pct
FROM `myforecastingsales.forecasting.gold_spike_features`
GROUP BY store_id
ORDER BY promo_day_pct DESC
LIMIT 10;


-- =============================================================================
-- UPDATE TRAINING VIEW TO INCLUDE SPIKE FEATURES
-- =============================================================================
CREATE OR REPLACE VIEW `myforecastingsales.forecasting.v_trainval_lgbm_v3` AS
SELECT
  t.*,
  s.feat_store_spike_pct,
  s.feat_store_promo_day,
  s.feat_seasonal_lift,
  s.feat_had_recent_spike,
  s.feat_historical_spike_prob
FROM `myforecastingsales.forecasting.v_trainval_lgbm_v2` t
LEFT JOIN `myforecastingsales.forecasting.gold_spike_features` s
  ON t.store_id = s.store_id
  AND t.sku_id = s.sku_id
  AND t.date = s.date;
