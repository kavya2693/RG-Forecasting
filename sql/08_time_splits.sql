-- =============================================================================
-- STEP 8: TIME SPLITS (LEAKAGE-PROOF) FOR 168-DAY HORIZON
-- =============================================================================
-- Forecast horizon: 168 days (24 weeks) starting 2025-12-18
-- Validation mimics actual forecast horizon length
-- 7-day embargo between train_end and val_start
--
-- Folds by Tier:
-- F1, F2, F3 for T1_MATURE (3 folds, rolling origin)
-- G1, G2 for T2_GROWING (2 folds)
-- C1 for T3_COLD_START (1 fold)
-- =============================================================================

-- =============================================================================
-- A: Create Time Splits Dimension Table
-- =============================================================================
CREATE OR REPLACE TABLE `myforecastingsales.forecasting.dim_time_splits_168d` AS

-- T1_MATURE: 3 folds (F1, F2, F3)
SELECT 'F1' AS fold_id, 'T1_MATURE' AS tier,
       DATE('2019-01-02') AS train_start,
       DATE('2025-06-26') AS train_end,
       DATE('2025-07-03') AS val_start,
       DATE('2025-12-17') AS val_end,
       168 AS val_days,
       7 AS embargo_days
UNION ALL
SELECT 'F2', 'T1_MATURE',
       DATE('2019-01-02'), DATE('2025-03-10'),
       DATE('2025-03-17'), DATE('2025-08-31'),
       168, 7
UNION ALL
SELECT 'F3', 'T1_MATURE',
       DATE('2019-01-02'), DATE('2024-11-07'),
       DATE('2024-11-14'), DATE('2025-04-30'),
       168, 7

UNION ALL

-- T2_GROWING: 2 folds (G1, G2)
SELECT 'G1', 'T2_GROWING',
       DATE('2023-07-06'), DATE('2025-06-26'),
       DATE('2025-07-03'), DATE('2025-12-17'),
       168, 7
UNION ALL
SELECT 'G2', 'T2_GROWING',
       DATE('2023-03-20'), DATE('2025-03-10'),
       DATE('2025-03-17'), DATE('2025-08-31'),
       168, 7

UNION ALL

-- T3_COLD_START: 1 fold (C1)
SELECT 'C1', 'T3_COLD_START',
       DATE('2025-01-02'), DATE('2025-06-26'),
       DATE('2025-07-03'), DATE('2025-12-17'),
       168, 7;

-- =============================================================================
-- B: Create View for Panel with Split Labels
-- =============================================================================
CREATE OR REPLACE VIEW `myforecastingsales.forecasting.v_panel_with_split_labels` AS
SELECT
  g.*,
  t.tier_name,
  t.is_intermittent,
  s.fold_id,
  CASE
    WHEN g.date BETWEEN s.train_start AND s.train_end THEN 'train'
    WHEN g.date BETWEEN s.val_start AND s.val_end THEN 'val'
    ELSE 'exclude'
  END AS split_label
FROM `myforecastingsales.forecasting.gold_panel_spine_enriched` g
JOIN `myforecastingsales.forecasting.series_tiers_asof_20251217` t
  USING (store_id, sku_id)
JOIN `myforecastingsales.forecasting.dim_time_splits_168d` s
  ON t.tier_name = s.tier;

-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- V1: Splits table summary
SELECT * FROM `myforecastingsales.forecasting.dim_time_splits_168d`
ORDER BY tier, fold_id;

-- V2: No overlap between train and val
SELECT
  fold_id,
  tier,
  train_end,
  val_start,
  DATE_DIFF(val_start, train_end, DAY) AS gap_days,
  CASE WHEN DATE_DIFF(val_start, train_end, DAY) >= 7 THEN 'PASS' ELSE 'FAIL' END AS embargo_check
FROM `myforecastingsales.forecasting.dim_time_splits_168d`;

-- V3: Validation horizon = 168 days
SELECT
  fold_id,
  tier,
  val_start,
  val_end,
  DATE_DIFF(val_end, val_start, DAY) + 1 AS actual_val_days,
  CASE WHEN DATE_DIFF(val_end, val_start, DAY) + 1 = 168 THEN 'PASS' ELSE 'FAIL' END AS horizon_check
FROM `myforecastingsales.forecasting.dim_time_splits_168d`;

-- V4: Row counts by fold and split
SELECT
  fold_id,
  split_label,
  COUNT(*) AS row_count,
  COUNT(DISTINCT CONCAT(store_id, '|', sku_id)) AS series_count
FROM `myforecastingsales.forecasting.v_panel_with_split_labels`
WHERE split_label IN ('train', 'val')
GROUP BY fold_id, split_label
ORDER BY fold_id, split_label;
