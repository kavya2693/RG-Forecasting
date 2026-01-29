-- =============================================================================
-- STEP 6: TIERING (MUTUALLY EXCLUSIVE)
-- =============================================================================
-- As-of date: 2025-12-17
--
-- Thresholds:
-- T1_MIN_HISTORY_DAYS = 728 (2 years)
-- T2_MIN_HISTORY_DAYS = 182 (6 months)
-- RECENCY_THRESHOLD_DAYS = 182 (6 months)
-- ADI_INTERMITTENT_THRESHOLD = 1.32
--
-- Tier Rules (strict order):
-- T0_EXCLUDED: n_pos_days = 0 OR days_since_last_sale > 182
-- T1_MATURE: history_days >= 728 AND NOT T0
-- T2_GROWING: history_days >= 182 AND < 728 AND NOT T0
-- T3_COLD_START: history_days < 182 AND NOT T0
--
-- Intermittency is a FLAG, not a tier: is_intermittent = 1 if ADI >= 1.32
-- =============================================================================

-- =============================================================================
-- A: Create Tier Table
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.series_tiers_asof_20251217` AS
SELECT
  store_id,
  sku_id,

  -- Tier assignment (mutually exclusive, strict order)
  CASE
    -- T0_EXCLUDED: no positive sales OR inactive > 182 days
    WHEN n_pos_days = 0 OR days_since_last_sale > 182 THEN 'T0_EXCLUDED'
    -- T1_MATURE: 2+ years history
    WHEN history_days >= 728 THEN 'T1_MATURE'
    -- T2_GROWING: 6mo-2yr history
    WHEN history_days >= 182 THEN 'T2_GROWING'
    -- T3_COLD_START: < 6mo history
    ELSE 'T3_COLD_START'
  END AS tier_name,

  -- T0 reason (NEVER_SOLD vs DORMANT_182D)
  CASE
    WHEN n_pos_days = 0 THEN 'NEVER_SOLD'
    WHEN n_pos_days > 0 AND days_since_last_sale > 182 THEN 'DORMANT_182D'
    ELSE NULL
  END AS t0_reason,

  -- Lifecycle stats
  history_days,
  n_pos_days,
  nz_rate,
  ADI,
  CV2,
  first_pos_date,
  last_pos_date,
  days_since_last_sale,

  -- Zero run stats
  max_zero_run_length,
  last_zero_run_length,

  -- Intermittency flag (property, not a tier)
  IF(ADI >= 1.32, 1, 0) AS is_intermittent,

  -- Magnitude stats
  sum_sales,
  avg_sales_all_days,
  avg_sales_pos_days,
  p50_sales_pos_days,
  p90_sales_pos_days,
  max_sales

FROM `myforecastingsales.forecasting.series_stats_pre_tiering`;

-- =============================================================================
-- B: Create View (daily panel with tiers)
-- =============================================================================
CREATE OR REPLACE VIEW `myforecastingsales.forecasting.v_daily_with_tiers` AS
SELECT
  g.*,
  t.tier_name,
  t.is_intermittent,
  t.history_days AS series_history_days,
  t.n_pos_days AS series_n_pos_days,
  t.nz_rate AS series_nz_rate,
  t.ADI AS series_ADI,
  t.CV2 AS series_CV2
FROM `myforecastingsales.forecasting.gold_panel_spine_enriched` g
JOIN `myforecastingsales.forecasting.series_tiers_asof_20251217` t
  USING (store_id, sku_id);

-- =============================================================================
-- C1: Series count by tier_name
-- =============================================================================
SELECT
  tier_name,
  COUNT(*) AS series_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`
GROUP BY tier_name
ORDER BY
  CASE tier_name
    WHEN 'T0_EXCLUDED' THEN 0
    WHEN 'T1_MATURE' THEN 1
    WHEN 'T2_GROWING' THEN 2
    WHEN 'T3_COLD_START' THEN 3
  END;

-- =============================================================================
-- C2: Series count by tier_name x is_intermittent
-- =============================================================================
SELECT
  tier_name,
  is_intermittent,
  COUNT(*) AS series_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total
FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`
GROUP BY tier_name, is_intermittent
ORDER BY
  CASE tier_name
    WHEN 'T0_EXCLUDED' THEN 0
    WHEN 'T1_MATURE' THEN 1
    WHEN 'T2_GROWING' THEN 2
    WHEN 'T3_COLD_START' THEN 3
  END,
  is_intermittent;

-- =============================================================================
-- C3: Sales share by tier (history window only)
-- =============================================================================
SELECT
  tier_name,
  tier_sales,
  ROUND(100.0 * tier_sales / total_sales, 2) AS sales_share_pct
FROM (
  SELECT
    t.tier_name,
    SUM(g.sales_filled) AS tier_sales,
    SUM(SUM(g.sales_filled)) OVER() AS total_sales
  FROM `myforecastingsales.forecasting.gold_panel_spine_enriched` g
  JOIN `myforecastingsales.forecasting.series_tiers_asof_20251217` t
    USING (store_id, sku_id)
  WHERE g.date <= '2025-12-17'
  GROUP BY t.tier_name
)
ORDER BY
  CASE tier_name
    WHEN 'T0_EXCLUDED' THEN 0
    WHEN 'T1_MATURE' THEN 1
    WHEN 'T2_GROWING' THEN 2
    WHEN 'T3_COLD_START' THEN 3
  END;

-- =============================================================================
-- C4: Inactivity buckets
-- =============================================================================
SELECT
  inactivity_bucket,
  series_count,
  ROUND(100.0 * series_count / SUM(series_count) OVER(), 2) AS pct
FROM (
  SELECT
    CASE
      WHEN n_pos_days = 0 THEN '1_never_sold'
      WHEN days_since_last_sale <= 28 THEN '2_0-28_days'
      WHEN days_since_last_sale <= 56 THEN '3_29-56_days'
      WHEN days_since_last_sale <= 182 THEN '4_57-182_days'
      ELSE '5_>182_days'
    END AS inactivity_bucket,
    COUNT(*) AS series_count
  FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`
  GROUP BY 1
)
ORDER BY inactivity_bucket;

-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- D1: No duplicate series
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT CONCAT(store_id, '|', sku_id)) AS distinct_series,
  CASE WHEN COUNT(*) = COUNT(DISTINCT CONCAT(store_id, '|', sku_id))
       THEN 'PASS' ELSE 'FAIL' END AS no_duplicates_check
FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`;

-- D2: Total count = 116,975
SELECT
  COUNT(*) AS total_series,
  CASE WHEN COUNT(*) = 116975 THEN 'PASS' ELSE 'FAIL' END AS count_check
FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`;

-- D3: T0_EXCLUDED logic check
SELECT
  (SELECT COUNT(*) FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`
   WHERE tier_name = 'T0_EXCLUDED') AS t0_count,
  (SELECT COUNT(*) FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`
   WHERE n_pos_days = 0 OR days_since_last_sale > 182) AS expected_t0_count,
  CASE
    WHEN (SELECT COUNT(*) FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`
          WHERE tier_name = 'T0_EXCLUDED') =
         (SELECT COUNT(*) FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`
          WHERE n_pos_days = 0 OR days_since_last_sale > 182)
    THEN 'PASS' ELSE 'FAIL'
  END AS t0_logic_check;

-- D4: Tier thresholds validation
SELECT
  COUNTIF(tier_name = 'T1_MATURE' AND history_days < 728) AS t1_bad_history,
  COUNTIF(tier_name = 'T2_GROWING' AND (history_days < 182 OR history_days >= 728)) AS t2_bad_history,
  COUNTIF(tier_name = 'T3_COLD_START' AND history_days >= 182) AS t3_bad_history,
  CASE
    WHEN COUNTIF(tier_name = 'T1_MATURE' AND history_days < 728) = 0
     AND COUNTIF(tier_name = 'T2_GROWING' AND (history_days < 182 OR history_days >= 728)) = 0
     AND COUNTIF(tier_name = 'T3_COLD_START' AND history_days >= 182) = 0
    THEN 'PASS' ELSE 'FAIL'
  END AS threshold_check
FROM `myforecastingsales.forecasting.series_tiers_asof_20251217`;
