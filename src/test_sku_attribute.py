"""
Test Adding SKU Local/Imported Attribute
========================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

TRAIN_FILE = '/tmp/c1_data/f1_train_500k.csv'
VAL_FILE = '/tmp/c1_data/f1_val_500k.csv'
SKU_ATTR_FILE = '/Users/srikavya/Documents/Claude-Projects/RG-Forecasting/sku_list_attribute.csv'

FEATURES = [
    'dow', 'is_weekend', 'week_of_year', 'month', 'day_of_year',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
    'is_store_closed', 'days_to_next_closure', 'days_from_prev_closure', 'is_closure_week',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
    'roll_mean_7', 'roll_sum_7', 'roll_mean_28', 'roll_sum_28', 'roll_std_28',
    'nz_rate_7', 'nz_rate_28', 'roll_mean_pos_28',
    'days_since_last_sale_asof', 'dormancy_capped', 'zero_run_length_asof', 'last_sale_qty_asof'
]
CAT_FEATURES = ['store_id', 'sku_id']


def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)

    # Load SKU attributes
    sku_attr = pd.read_csv(SKU_ATTR_FILE)
    sku_attr['sku_id'] = sku_attr['sku_id'].astype(str)

    # Create is_local feature (L=1, I=0, LI=1)
    sku_attr['is_local'] = sku_attr['local_imported_attribute'].apply(
        lambda x: 1 if x in ['L', 'LI'] else 0
    )
    sku_attr['is_imported'] = sku_attr['local_imported_attribute'].apply(
        lambda x: 1 if x in ['I', 'LI'] else 0
    )

    print(f"  SKU attributes: {len(sku_attr)} SKUs")
    print(f"  Local: {sku_attr['is_local'].sum()}, Imported: {sku_attr['is_imported'].sum()}")

    # Merge with train and val
    train['sku_id'] = train['sku_id'].astype(str)
    val['sku_id'] = val['sku_id'].astype(str)

    train = train.merge(sku_attr[['sku_id', 'is_local', 'is_imported']], on='sku_id', how='left')
    val = val.merge(sku_attr[['sku_id', 'is_local', 'is_imported']], on='sku_id', how='left')

    # Fill missing with 0
    train['is_local'] = train['is_local'].fillna(0).astype(int)
    train['is_imported'] = train['is_imported'].fillna(0).astype(int)
    val['is_local'] = val['is_local'].fillna(0).astype(int)
    val['is_imported'] = val['is_imported'].fillna(0).astype(int)

    print(f"  Train matched: {train['is_local'].sum()} local, {train['is_imported'].sum()} imported")

    for col in FEATURES:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)

    print(f"  Train: {len(train):,} rows")
    print(f"  Val: {len(val):,} rows")
    return train, val


def run_c1_b1(train, val, features, cat_features, label=""):
    """Run C1+B1 approach."""
    train = train.copy()
    val_copy = val.copy()
    train['y_binary'] = (train['y'] > 0).astype(int)

    for col in cat_features:
        train[col] = train[col].astype('category')
        val_copy[col] = val_copy[col].astype('category')

    X_train = train[features + cat_features]
    X_val = val_copy[features + cat_features]
    y_val = val_copy['y'].values
    is_closed = val_copy['is_store_closed'].values

    # Stage 1: Binary classifier
    params_clf = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 100,
        'verbose': -1,
        'n_jobs': -1
    }

    train_clf = lgb.Dataset(X_train, label=train['y_binary'], categorical_feature=cat_features)
    clf_model = lgb.train(params_clf, train_clf, num_boost_round=150)
    prob_nonzero = clf_model.predict(X_val)

    # Stage 2: Log-transform regressor on non-zeros
    train_nz = train[train['y'] > 0].copy()
    train_nz['y_log'] = np.log1p(train_nz['y'])

    params_reg = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 30,
        'lambda_l2': 1.5,
        'verbose': -1,
        'n_jobs': -1
    }

    X_train_nz = train_nz[features + cat_features]
    train_reg = lgb.Dataset(X_train_nz, label=train_nz['y_log'], categorical_feature=cat_features)
    reg_model = lgb.train(params_reg, train_reg, num_boost_round=300)

    y_pred_log = reg_model.predict(X_val)
    y_pred_value = np.expm1(y_pred_log)

    # Combine with threshold 0.7
    y_pred = np.where(prob_nonzero > 0.7, y_pred_value, 0)
    y_pred = np.maximum(0, y_pred)
    y_pred[is_closed == 1] = 0

    # Calculate metrics
    mae = np.mean(np.abs(y_val - y_pred))
    wmape = 100 * np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    return wmape, mae, clf_model, reg_model


def main():
    train, val = load_data()

    results = []

    # A: Without SKU attribute (baseline)
    print("\n" + "="*60)
    print("A: C1+B1 WITHOUT SKU ATTRIBUTE (baseline)")
    print("="*60)
    wmape, mae, _, _ = run_c1_b1(train, val, FEATURES, CAT_FEATURES)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'A: Without attribute', 'WMAPE': wmape, 'MAE': mae})

    # B: With is_local only
    print("\n" + "="*60)
    print("B: C1+B1 + is_local FEATURE")
    print("="*60)
    features_b = FEATURES + ['is_local']
    wmape, mae, _, _ = run_c1_b1(train, val, features_b, CAT_FEATURES)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'B: + is_local', 'WMAPE': wmape, 'MAE': mae})

    # C: With both is_local and is_imported
    print("\n" + "="*60)
    print("C: C1+B1 + is_local + is_imported")
    print("="*60)
    features_c = FEATURES + ['is_local', 'is_imported']
    wmape, mae, clf_model, reg_model = run_c1_b1(train, val, features_c, CAT_FEATURES)
    print(f"  WMAPE: {wmape:.2f}%, MAE: {mae:.4f}")
    results.append({'Approach': 'C: + both attributes', 'WMAPE': wmape, 'MAE': mae})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: SKU ATTRIBUTE COMPARISON")
    print("="*60)
    print(f"\n{'Approach':<25} {'WMAPE':>10} {'MAE':>10} {'vs Baseline':>12}")
    print("-" * 60)

    baseline_wmape = results[0]['WMAPE']
    for r in sorted(results, key=lambda x: x['WMAPE']):
        diff = r['WMAPE'] - baseline_wmape
        diff_str = f"{diff:+.2f}pp" if diff != 0 else "baseline"
        print(f"{r['Approach']:<25} {r['WMAPE']:>9.2f}% {r['MAE']:>10.4f} {diff_str:>12}")

    # Feature importance for best model
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Regressor - Top 15)")
    print("="*60)
    importance = pd.DataFrame({
        'feature': features_c + CAT_FEATURES,
        'importance': reg_model.feature_importance()
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:>8}")

    return results


if __name__ == "__main__":
    main()
