-- =============================================================================
-- STEP 9: TRAINING VIEW FOR LIGHTGBM (v_trainval_lgbm_v2)
-- =============================================================================
-- Joins gold_panel_features_v2 with time splits for fold-based training
-- Excludes T0_EXCLUDED tier
-- Keeps closure rows (is_store_closed=1) with y=0 for model to learn closure pattern
-- =============================================================================

CREATE OR REPLACE VIEW `myforecastingsales.forecasting.v_trainval_lgbm_v2` AS
SELECT
  -- Fold/Split info
  ts.fold_id,
  ts.split_role,
  ts.tier_name AS fold_tier,

  -- Keys
  f.store_id,
  f.sku_id,
  f.date,

  -- Target
  f.sales_filled AS y,

  -- Tier info (from features table)
  f.tier_name,
  f.is_intermittent,
  f.series_history_days,
  f.series_nz_rate,
  f.series_ADI,

  -- Flags (for weighting)
  f.is_observed,
  f.was_negative,
  f.is_covid_period,
  f.is_covid_panic_spike,
  f.is_extreme_spike,

  -- Calendar features
  f.dow,
  f.week_of_year,
  f.month,
  f.year,
  f.is_weekend,
  f.day_of_month,
  f.day_of_year,
  f.quarter,
  f.week_of_month,
  f.is_month_start,
  f.is_month_end,

  -- Closure features
  f.is_store_closed,
  f.closure_name,
  f.days_to_next_closure,
  f.days_from_prev_closure,
  f.is_closure_week,

  -- Lag features
  f.lag_1,
  f.lag_7,
  f.lag_14,
  f.lag_28,
  f.lag_56,

  -- Rolling features
  f.roll_mean_7,
  f.roll_sum_7,
  f.roll_mean_28,
  f.roll_sum_28,
  f.roll_std_28,
  f.nz_rate_28,

  -- V2 features
  f.nz_rate_7,
  f.roll_mean_pos_28,
  f.sin_doy,
  f.cos_doy,
  f.sin_dow,
  f.cos_dow,
  f.dormancy_bucket,
  f.dormancy_capped,

  -- Recency features
  f.days_since_last_sale_asof,
  f.zero_run_length_asof,
  f.last_sale_qty_asof,

  -- Safe transforms
  f.y_log1p,
  f.lag_7_log1p,
  f.roll_mean_28_log1p,
  f.roll_mean_pos_28_log1p

FROM `myforecastingsales.forecasting.gold_panel_features_v2` f
INNER JOIN `myforecastingsales.forecasting.dim_time_splits_168d` ts
  ON f.tier_name = ts.tier_name
  AND f.date >= ts.train_start
  AND f.date <= ts.val_end
  AND (
    (f.date <= ts.train_end AND ts.split_role = 'TRAIN')
    OR (f.date >= ts.val_start AND f.date <= ts.val_end AND ts.split_role = 'VAL')
  )
WHERE f.tier_name != 'T0_EXCLUDED';


-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- V1: Count by fold and split
SELECT
  fold_id,
  tier_name,
  split_role,
  COUNT(*) AS n_rows,
  COUNT(DISTINCT CONCAT(store_id, '|', sku_id)) AS n_series,
  MIN(date) AS min_date,
  MAX(date) AS max_date
FROM `myforecastingsales.forecasting.v_trainval_lgbm_v2`
GROUP BY fold_id, tier_name, split_role
ORDER BY fold_id, tier_name, split_role;

-- V2: Example query for F1 fold, T1_MATURE tier
-- SELECT *
-- FROM `myforecastingsales.forecasting.v_trainval_lgbm_v2`
-- WHERE fold_id = 'F1' AND tier_name = 'T1_MATURE' AND split_role = 'TRAIN'
-- LIMIT 100;
