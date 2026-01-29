"""
Full Test: Spike Features Across All Segments + Weekly Metrics
==============================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

OUTPUT_DIR = Path("/tmp/spike_analysis")

def load_prepared_data():
    """Load data with spike features already computed."""
    # Re-run the spike analysis pipeline
    from spike_analysis import load_data, detect_spikes, classify_spikes, create_spike_features, assign_segments

    train, val = load_data()
    train = detect_spikes(train)
    val = detect_spikes(val)
    train = classify_spikes(train)
    val = classify_spikes(val)
    train, new_features = create_spike_features(train)
    val, _ = create_spike_features(val)
    train = assign_segments(train)
    val = assign_segments(val)

    return train, val, new_features


def compute_all_metrics(y_true, y_pred, val_df):
    """Compute daily and weekly metrics."""
    target = 'y' if 'y' in val_df.columns else 'qty'

    results = {}

    # Daily
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1)
    results['daily_sku_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Weekly SKU-Store
    val_df = val_df.copy()
    val_df['y_pred'] = y_pred
    weekly = val_df.groupby(['store_id', 'sku_id', 'year', 'week']).agg({target: 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly[target] - weekly['y_pred'])) / max(np.sum(weekly[target]), 1)
    results['weekly_sku_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Weekly Store
    weekly_store = val_df.groupby(['store_id', 'year', 'week']).agg({target: 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_store[target] - weekly_store['y_pred'])) / max(np.sum(weekly_store[target]), 1)
    results['weekly_store'] = {'wmape': wmape, 'wfa': 100 - wmape}

    # Weekly Total
    weekly_total = val_df.groupby(['year', 'week']).agg({target: 'sum', 'y_pred': 'sum'}).reset_index()
    wmape = 100 * np.sum(np.abs(weekly_total[target] - weekly_total['y_pred'])) / max(np.sum(weekly_total[target]), 1)
    results['weekly_total'] = {'wmape': wmape, 'wfa': 100 - wmape}

    return results


def test_segment(train, val, new_features, segment):
    """Test baseline vs spike features for a segment."""
    target = 'y' if 'y' in train.columns else 'qty'

    train_seg = train[train['segment'] == segment].copy()
    val_seg = val[val['segment'] == segment].copy()

    if len(train_seg) < 100 or len(val_seg) < 100:
        return None

    # Features
    base_features = [
        'dow', 'is_weekend', 'day_of_year', 'sin_doy', 'cos_doy',
        'lag_7', 'lag_14', 'roll_mean_28', 'nz_rate_28', 'roll_mean_pos_28',
    ]
    cat_features = ['sku_id', 'store_id']

    for col in cat_features:
        train_seg[col] = train_seg[col].astype('category')
        val_seg[col] = val_seg[col].astype('category')

    available_base = [f for f in base_features if f in train_seg.columns]
    available_new = [f for f in new_features if f in train_seg.columns]

    for col in available_base + available_new:
        train_seg[col] = train_seg[col].fillna(0)
        val_seg[col] = val_seg[col].fillna(0)

    y_train = train_seg[target].values
    y_val = val_seg[target].values

    results = {'segment': segment, 'train_size': len(train_seg), 'val_size': len(val_seg)}

    # Baseline
    X_train_base = train_seg[available_base + cat_features]
    X_val_base = val_seg[available_base + cat_features]

    clf = lgb.LGBMClassifier(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1, random_state=42)
    clf.fit(X_train_base, (y_train > 0).astype(int))
    prob = clf.predict_proba(X_val_base)[:, 1]

    train_nz = train_seg[train_seg[target] > 0]
    reg = lgb.LGBMRegressor(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1, random_state=42)
    reg.fit(train_nz[available_base + cat_features], np.log1p(train_nz[target]))
    pred = np.expm1(reg.predict(X_val_base))

    y_pred_base = np.where(prob > 0.6, pred, 0)
    y_pred_base = np.maximum(0, y_pred_base)

    results['baseline'] = compute_all_metrics(y_val, y_pred_base, val_seg)

    # With spike features
    all_features = available_base + available_new + cat_features
    X_train_spike = train_seg[all_features]
    X_val_spike = val_seg[all_features]

    clf2 = lgb.LGBMClassifier(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1, random_state=42)
    clf2.fit(X_train_spike, (y_train > 0).astype(int))
    prob2 = clf2.predict_proba(X_val_spike)[:, 1]

    reg2 = lgb.LGBMRegressor(num_leaves=63, n_estimators=300, verbose=-1, n_jobs=-1, random_state=42)
    reg2.fit(train_nz[all_features], np.log1p(train_nz[target]))
    pred2 = np.expm1(reg2.predict(X_val_spike))

    y_pred_spike = np.where(prob2 > 0.6, pred2, 0)
    y_pred_spike = np.maximum(0, y_pred_spike)

    results['with_spikes'] = compute_all_metrics(y_val, y_pred_spike, val_seg)

    return results


def main():
    print("=" * 70)
    print("FULL TEST: SPIKE FEATURES ACROSS ALL SEGMENTS")
    print("=" * 70)

    print("\nLoading and preparing data...")
    train, val, new_features = load_prepared_data()

    all_results = {}

    for segment in ['A', 'B', 'C']:
        print(f"\n{'='*50}")
        print(f"SEGMENT {segment}")
        print("=" * 50)

        results = test_segment(train, val, new_features, segment)
        if results:
            all_results[segment] = results

            print(f"\n  {'Metric':<20} {'Baseline':>12} {'+ Spikes':>12} {'Change':>12}")
            print("  " + "-" * 56)

            for level in ['daily_sku_store', 'weekly_sku_store', 'weekly_store', 'weekly_total']:
                base_wfa = results['baseline'][level]['wfa']
                spike_wfa = results['with_spikes'][level]['wfa']
                change = spike_wfa - base_wfa
                print(f"  {level:<20} {base_wfa:>11.2f}% {spike_wfa:>11.2f}% {change:>+11.2f}pp")

    # Save
    with open(OUTPUT_DIR / 'full_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Impact of Spike Features")
    print("=" * 70)
    print(f"\n{'Segment':<10} {'Daily WFA Δ':>15} {'Weekly Store WFA Δ':>20}")
    print("-" * 45)
    for seg in ['A', 'B', 'C']:
        if seg in all_results:
            daily_change = all_results[seg]['with_spikes']['daily_sku_store']['wfa'] - all_results[seg]['baseline']['daily_sku_store']['wfa']
            weekly_change = all_results[seg]['with_spikes']['weekly_store']['wfa'] - all_results[seg]['baseline']['weekly_store']['wfa']
            print(f"{seg:<10} {daily_change:>+14.2f}pp {weekly_change:>+19.2f}pp")

    print(f"\nResults saved to {OUTPUT_DIR / 'full_test_results.json'}")


if __name__ == "__main__":
    main()
