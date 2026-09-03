#!/usr/bin/env python3
"""
Phase 7+8 End-to-End Test: Enhanced Model Zoo + Model Combination
Full pipeline: EDA → FA → Segmentation → Features → ALL Models → Combination → Reconciliation

Tests every new model type and combination method on DACH SKIN_CARE.
"""
import warnings
warnings.filterwarnings('ignore')

import json, logging, os, time
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

DATA_PATH = "sourcedata/DACH/SKIN_CARE.csv"
KEY_COL = "Model_Hierarchy"
DATE_COL = "Shipment_week"
TARGET_COL = "Actuals"
HIER_COL = "APG"
FORECAST_LAG = 4


def _prepare_data():
    """Load data and create train/val features."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    period_totals = df.groupby(DATE_COL)[TARGET_COL].sum()
    cutoff = str(period_totals[period_totals > 0].index.max())
    hist_df = df[df[DATE_COL] <= cutoff].copy()

    periods = sorted(hist_df[DATE_COL].unique())
    val_size = 13
    train_end = periods[-val_size - 1]
    val_start = periods[-val_size]

    train_df = hist_df[hist_df[DATE_COL] <= train_end].copy()
    val_df = hist_df[hist_df[DATE_COL] >= val_start].copy()

    for part in [train_df, val_df]:
        part.sort_values([KEY_COL, DATE_COL], inplace=True)
        for lag in [1, 2, 4, 8, 13]:
            part[f'{TARGET_COL}_lag_{lag}'] = part.groupby(KEY_COL)[TARGET_COL].shift(lag)
        for w in [4, 13]:
            part[f'{TARGET_COL}_roll_mean_{w}'] = part.groupby(KEY_COL)[TARGET_COL].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    feature_cols = [c for c in train_df.columns
                    if c.startswith(f'{TARGET_COL}_lag_') or c.startswith(f'{TARGET_COL}_roll_')]

    return df, hist_df, train_df, val_df, feature_cols, train_end, cutoff


def main():
    t0 = time.time()
    df, hist_df, train_df, val_df, feature_cols, train_end, cutoff = _prepare_data()

    y_val = val_df[TARGET_COL].fillna(0).values
    logger.info(f"Data: {len(train_df):,} train, {len(val_df):,} val, {len(feature_cols)} features")

    results = {}  # model_name → {wape, time, ...}

    # =========================================================================
    # PHASE 7 TEST 1: CatBoost with Key Embeddings
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: CatBoost Embedding Model")
    logger.info("=" * 70)

    from utils.enhanced_model_zoo import train_catboost_embedding_model
    t1 = time.time()
    cb_model, cb_meta = train_catboost_embedding_model(
        train_df, val_df, feature_cols, TARGET_COL, KEY_COL,
        iterations=400, learning_rate=0.05, depth=6,
    )
    results['catboost_embedding'] = {'wape': cb_meta['val_wape'], 'time': time.time() - t1}
    # Quick prediction test
    sample_X = val_df[feature_cols].fillna(0).values[:5]
    sample_keys = val_df[KEY_COL].values[:5]
    preds = cb_model.predict(sample_X, keys=sample_keys)
    assert len(preds) == 5 and all(p >= 0 for p in preds)
    logger.info(f"Test 1 PASSED in {results['catboost_embedding']['time']:.1f}s — wape={cb_meta['val_wape']:.4f}")

    # =========================================================================
    # PHASE 7 TEST 2: Quantile Regression
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Quantile Regression (median + intervals)")
    logger.info("=" * 70)

    from utils.enhanced_model_zoo import train_quantile_model
    t2 = time.time()
    q_model, q_meta = train_quantile_model(
        train_df, val_df, feature_cols, TARGET_COL,
        quantiles=[0.1, 0.5, 0.9], model_family='lightgbm',
    )
    results['quantile_regression'] = {'wape': q_meta['val_wape'], 'time': time.time() - t2}
    # Test intervals
    point, lower, upper = q_model.predict_intervals(sample_X)
    logger.info(f"  Sample interval: [{lower[0]:.1f}, {point[0]:.1f}, {upper[0]:.1f}]")
    logger.info(f"  80% PI coverage: {q_meta['interval_coverage']:.1%}")
    logger.info(f"Test 2 PASSED in {results['quantile_regression']['time']:.1f}s — wape={q_meta['val_wape']:.4f}")

    # =========================================================================
    # PHASE 7 TEST 3: Conformal Residual Boosting
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Conformal Residual Boosting")
    logger.info("=" * 70)

    from utils.enhanced_model_zoo import train_conformal_residual_boost
    t3 = time.time()
    conf_model, conf_meta = train_conformal_residual_boost(
        train_df, val_df, feature_cols, TARGET_COL,
        model_family='lightgbm',
    )
    results['conformal_boost'] = {'wape': conf_meta['val_wape_combined'], 'time': time.time() - t3}
    # Test conformal intervals
    point_c, lower_c, upper_c = conf_model.predict_intervals(sample_X, confidence=0.9)
    logger.info(f"  Sample conformal interval: [{lower_c[0]:.1f}, {point_c[0]:.1f}, {upper_c[0]:.1f}]")
    logger.info(f"  90% coverage: {conf_meta['conformal_coverage_90']:.1%}")
    logger.info(f"Test 3 PASSED in {results['conformal_boost']['time']:.1f}s — "
                f"wape={conf_meta['val_wape_combined']:.4f} (base={conf_meta['val_wape_base']:.4f})")

    # =========================================================================
    # PHASE 3 BASELINES: Global-Local + Mixed-Effects (for combination)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("BASELINES: Hierarchical Models (from Phase 3)")
    logger.info("=" * 70)

    from utils.hierarchical_model_training import train_global_local_model, train_mixed_effects_model
    gl_model, gl_meta = train_global_local_model(
        train_df, val_df, feature_cols, TARGET_COL, KEY_COL,
    )
    results['global_local'] = {'wape': gl_meta['val_wape'], 'time': 0}

    me_model, me_meta = train_mixed_effects_model(
        train_df, val_df, feature_cols, TARGET_COL, KEY_COL, min_key_samples=20,
    )
    results['mixed_effects'] = {'wape': me_meta['val_wape'], 'time': 0}

    # Standard LightGBM baseline
    import lightgbm as lgb
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[TARGET_COL].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values

    lgbm_base = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, verbosity=-1,
    )
    lgbm_base.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    lgbm_wape = float(np.sum(np.abs(y_val - np.maximum(lgbm_base.predict(X_val), 0))) /
                       (np.sum(np.abs(y_val)) + 1e-10))
    results['lgbm_baseline'] = {'wape': lgbm_wape, 'time': 0}

    # =========================================================================
    # PHASE 8 TEST 4: Multi-Level Forecast Combination
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Multi-Level Forecast Combination")
    logger.info("=" * 70)

    from utils.model_combination import combine_multi_level_forecasts
    t4 = time.time()

    # Get predictions from all models
    forecasts_dict = {
        'global_local': gl_model.predict(X_val, keys=val_df[KEY_COL].values),
        'catboost': cb_model.predict(X_val, keys=val_df[KEY_COL].values),
        'lgbm': np.maximum(lgbm_base.predict(X_val), 0),
        'quantile': q_model.predict(X_val),
    }

    # Key observation counts
    key_obs = train_df[KEY_COL].value_counts().to_dict()

    for method in ['optimal', 'inverse_error', 'data_adaptive']:
        combined, combo_meta = combine_multi_level_forecasts(
            forecasts=forecasts_dict,
            actuals=y_val,
            keys=val_df[KEY_COL].values,
            key_obs_counts=key_obs,
            method=method,
        )
        results[f'combination_{method}'] = {'wape': combo_meta['combined_wape'], 'time': 0}

    logger.info(f"Test 4 PASSED in {time.time() - t4:.1f}s")

    # =========================================================================
    # PHASE 8 TEST 5: Diverse Stacking Ensemble
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Diverse Stacking Ensemble")
    logger.info("=" * 70)

    from utils.model_combination import train_diverse_stacking
    t5 = time.time()

    stacked_model, stack_meta = train_diverse_stacking(
        train_df, val_df, feature_cols, TARGET_COL, KEY_COL,
    )
    results['stacked_ensemble'] = {'wape': stack_meta['val_wape'], 'time': time.time() - t5}

    stacked_preds = stacked_model.predict(X_val)
    assert len(stacked_preds) == len(y_val)
    logger.info(f"Test 5 PASSED in {results['stacked_ensemble']['time']:.1f}s — wape={stack_meta['val_wape']:.4f}")

    # =========================================================================
    # PHASE 3 TEST 6: MinT Reconciliation (on stacked predictions)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: MinT Reconciliation on Final Predictions")
    logger.info("=" * 70)

    from utils.hierarchical_reconciliation import run_hierarchical_reconciliation
    t6 = time.time()

    # Create forecast DataFrame
    forecast_rows = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        forecast_rows.append({
            KEY_COL: row[KEY_COL],
            DATE_COL: row[DATE_COL],
            'predicted': float(stacked_preds[i]),
        })
    forecast_df = pd.DataFrame(forecast_rows)

    reconciled_df = run_hierarchical_reconciliation(
        forecasts_df=forecast_df,
        source_df=hist_df,
        hierarchy_col=HIER_COL,
        key_col=KEY_COL,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        predicted_col='predicted',
        method='mint_shrink',
        train_cutoff=train_end,
    )

    recon_preds = reconciled_df['predicted'].values
    recon_wape = float(np.sum(np.abs(y_val - recon_preds)) / (np.sum(np.abs(y_val)) + 1e-10))
    results['stacked+mint'] = {'wape': recon_wape, 'time': time.time() - t6}
    logger.info(f"Test 6 PASSED in {results['stacked+mint']['time']:.1f}s — "
                f"wape={recon_wape:.4f}")

    # =========================================================================
    # FINAL SUMMARY — ALL MODELS COMPARED
    # =========================================================================
    total = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("ALL PHASE 7+8 TESTS PASSED")
    logger.info("=" * 70)
    logger.info(f"Total time: {total:.1f}s\n")

    # Sort by WAPE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['wape'])

    logger.info(f"{'Rank':<5} {'Model':<30} {'Val WAPE':<12} {'Time (s)':<10}")
    logger.info("-" * 57)
    for rank, (name, info) in enumerate(sorted_results, 1):
        marker = " ★" if name in ('stacked_ensemble', 'stacked+mint', 'combination_optimal') else ""
        logger.info(f"  {rank:<3} {name:<30} {info['wape']:<12.4f} {info.get('time', 0):<10.1f}{marker}")

    best_name, best_info = sorted_results[0]
    worst_name, worst_info = sorted_results[-1]
    baseline = results['lgbm_baseline']['wape']

    logger.info(f"\n  Best:     {best_name} (WAPE={best_info['wape']:.4f})")
    logger.info(f"  Baseline: lgbm_baseline (WAPE={baseline:.4f})")
    logger.info(f"  Worst:    {worst_name} (WAPE={worst_info['wape']:.4f})")
    logger.info(f"\n  New models (Phase 7): catboost_embedding, quantile_regression, conformal_boost")
    logger.info(f"  Combinations (Phase 8): optimal/inverse_error/data_adaptive, stacked_ensemble")
    logger.info(f"  Reconciliation: stacked + MinT shrinkage")


if __name__ == '__main__':
    main()
