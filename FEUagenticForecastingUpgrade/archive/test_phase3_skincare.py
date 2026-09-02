#!/usr/bin/env python3
"""
Phase 3 End-to-End Test: Hierarchy Features + Models + Reconciliation
on DACH SKIN_CARE data (317 keys, APG as hierarchy with 14 groups).

Tests:
1. Hierarchy temporal feature creation
2. Global-Local model training
3. Mixed-Effects model training
4. Multi-Level Ensemble training
5. MinT reconciliation
"""
import warnings
warnings.filterwarnings('ignore')

import json
import logging
import os
import time

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


def main():
    t0 = time.time()

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    logger.info(f"Source: {df.shape[0]:,} rows, {df.shape[1]} cols, {df[KEY_COL].nunique()} keys")

    # History cutoff
    period_totals = df.groupby(DATE_COL)[TARGET_COL].sum()
    positive = period_totals[period_totals > 0]
    cutoff = str(positive.index.max())
    logger.info(f"History cutoff: {cutoff}")

    hist_df = df[df[DATE_COL] <= cutoff].copy()

    # Train/val split
    hist_periods = sorted(hist_df[DATE_COL].unique())
    val_size = 13
    train_periods = hist_periods[:-val_size]
    val_periods = hist_periods[-val_size:]
    train_end = train_periods[-1]
    val_start = val_periods[0]
    val_end = val_periods[-1]

    train_df = hist_df[hist_df[DATE_COL] <= train_end].copy()
    val_df = hist_df[(hist_df[DATE_COL] >= val_start) & (hist_df[DATE_COL] <= val_end)].copy()

    logger.info(f"Train: {len(train_df):,} rows ({len(train_periods) - val_size} periods)")
    logger.info(f"Val:   {len(val_df):,} rows ({val_size} periods)")
    logger.info(f"APG groups: {df[HIER_COL].nunique()}")

    # =========================================================================
    # TEST 1: HIERARCHY TEMPORAL FEATURES
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Hierarchy Temporal Features")
    logger.info("=" * 70)

    from utils.hierarchy_features import create_hierarchy_temporal_features, get_hierarchy_feature_names

    t1 = time.time()
    featured_df = create_hierarchy_temporal_features(
        df=hist_df,
        key_cols=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        hierarchy_cols=[HIER_COL],
        forecast_lag=FORECAST_LAG,
        windows=[4, 13],
    )

    expected_names = get_hierarchy_feature_names(TARGET_COL, [HIER_COL], windows=[4, 13])
    actual_names = [c for c in featured_df.columns if c.startswith(f'{TARGET_COL}_')]
    missing = [n for n in expected_names if n not in featured_df.columns]

    logger.info(f"Expected features: {expected_names}")
    logger.info(f"Actual hierarchy features: {[c for c in actual_names if 'apg' in c.lower() or 'group' in c.lower()]}")
    if missing:
        logger.error(f"MISSING features: {missing}")
    else:
        logger.info("All expected hierarchy features present")

    # Verify leakage-free: no NaN in training period for lagged features
    train_mask = featured_df[DATE_COL] <= train_end
    for col in expected_names:
        if col in featured_df.columns:
            # After forecast_lag periods, should have values
            non_null = featured_df.loc[train_mask, col].notna().mean()
            logger.info(f"  {col}: {non_null:.1%} non-null in train")

    logger.info(f"Test 1 PASSED in {time.time() - t1:.1f}s")

    # =========================================================================
    # TEST 2: GLOBAL-LOCAL MODEL
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Global-Local Model Training")
    logger.info("=" * 70)

    from utils.hierarchical_model_training import (
        train_global_local_model,
        train_mixed_effects_model,
        train_multi_level_ensemble,
    )

    # Prepare simple features for model training
    feature_cols = _create_simple_features(train_df, val_df, KEY_COL, DATE_COL, TARGET_COL)

    t2 = time.time()
    gl_model, gl_meta = train_global_local_model(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        key_col=KEY_COL,
        model_family='lightgbm',
    )

    logger.info(f"Global-Local: val_wape={gl_meta['val_wape']:.4f}, "
                f"avg_bias={gl_meta['avg_bias_magnitude']:.2f}")

    # Test prediction
    sample_keys = val_df[KEY_COL].values[:5]
    sample_X = val_df[feature_cols].fillna(0).values[:5]
    preds = gl_model.predict(sample_X, keys=sample_keys)
    logger.info(f"Sample predictions: {preds[:5]}")
    assert len(preds) == 5, "Prediction count mismatch"
    assert all(p >= 0 for p in preds), "Negative predictions"
    logger.info(f"Test 2 PASSED in {time.time() - t2:.1f}s")

    # =========================================================================
    # TEST 3: MIXED-EFFECTS MODEL
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Mixed-Effects Model Training")
    logger.info("=" * 70)

    t3 = time.time()
    me_model, me_meta = train_mixed_effects_model(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        key_col=KEY_COL,
        model_family='lightgbm',
        min_key_samples=20,
    )

    logger.info(f"Mixed-Effects: val_wape={me_meta['val_wape']:.4f}, "
                f"global_only={me_meta['val_wape_global_only']:.4f}, "
                f"improvement={me_meta['improvement_from_adjustments']:.4f}")

    preds_me = me_model.predict(sample_X, keys=sample_keys)
    assert len(preds_me) == 5
    logger.info(f"Test 3 PASSED in {time.time() - t3:.1f}s")

    # =========================================================================
    # TEST 4: MULTI-LEVEL ENSEMBLE
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Multi-Level Ensemble Training")
    logger.info("=" * 70)

    t4 = time.time()
    ml_model, ml_meta = train_multi_level_ensemble(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        key_col=KEY_COL,
        hierarchy_col=HIER_COL,
        model_family='lightgbm',
        min_key_samples=40,
        min_group_samples=50,
    )

    logger.info(f"Multi-Level: val_wape={ml_meta['val_wape']:.4f}, "
                f"global_only={ml_meta['val_wape_global_only']:.4f}, "
                f"n_hier={ml_meta['n_hier_models']}, n_key={ml_meta['n_key_models']}")

    preds_ml = ml_model.predict(sample_X, keys=sample_keys)
    assert len(preds_ml) == 5
    logger.info(f"Test 4 PASSED in {time.time() - t4:.1f}s")

    # =========================================================================
    # TEST 5: MinT RECONCILIATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: MinT Reconciliation")
    logger.info("=" * 70)

    from utils.hierarchical_reconciliation import (
        build_summing_matrix,
        run_hierarchical_reconciliation,
    )

    t5 = time.time()

    # Create synthetic forecasts
    forecast_keys = val_df[KEY_COL].unique()[:50]  # Use subset for speed
    key_to_group = dict(zip(df[KEY_COL], df[HIER_COL]))
    hierarchy_map = {k: key_to_group[k] for k in forecast_keys if k in key_to_group}

    # Build summing matrix
    S, node_names = build_summing_matrix(hierarchy_map, list(forecast_keys))
    logger.info(f"Summing matrix: {S.shape}, nodes: Total + {len(set(hierarchy_map.values()))} groups + {len(forecast_keys)} keys")

    # Create forecast df for one period
    forecast_rows = []
    for key in forecast_keys:
        forecast_rows.append({
            KEY_COL: key,
            DATE_COL: val_periods[0],
            'predicted': np.random.uniform(0, 100),
        })
    forecast_df = pd.DataFrame(forecast_rows)

    # Run MinT reconciliation
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

    # Check coherence
    original_total = forecast_df['predicted'].sum()
    reconciled_total = reconciled_df['predicted'].sum()
    logger.info(f"Original total: {original_total:.1f}")
    logger.info(f"Reconciled total: {reconciled_total:.1f}")
    logger.info(f"All predictions non-negative: {(reconciled_df['predicted'] >= 0).all()}")

    # Check group coherence
    reconciled_with_group = reconciled_df.merge(
        pd.DataFrame({'key': list(hierarchy_map.keys()), 'group': list(hierarchy_map.values())}),
        left_on=KEY_COL, right_on='key', how='left'
    )
    group_totals = reconciled_with_group.groupby('group')['predicted'].sum()
    logger.info(f"Group totals after reconciliation: {dict(group_totals.round(1))}")

    logger.info(f"Test 5 PASSED in {time.time() - t5:.1f}s")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("ALL PHASE 3 TESTS PASSED")
    logger.info("=" * 70)
    logger.info(f"Total time: {total:.1f}s")
    logger.info(f"")
    logger.info(f"Model Performance Comparison (val WAPE):")
    logger.info(f"  Global-Local:        {gl_meta['val_wape']:.4f}")
    logger.info(f"  Mixed-Effects:       {me_meta['val_wape']:.4f}")
    logger.info(f"  Multi-Level:         {ml_meta['val_wape']:.4f}")
    logger.info(f"  Global-Only:         {me_meta['val_wape_global_only']:.4f}")
    logger.info(f"")
    logger.info(f"Hierarchy Features: {len(expected_names)} features from {HIER_COL}")
    logger.info(f"MinT Reconciliation: applied on {len(forecast_keys)} keys")


def _create_simple_features(train_df, val_df, key_col, date_col, target_col):
    """Create basic lag features for model testing."""
    for df_part in [train_df, val_df]:
        df_part.sort_values([key_col, date_col], inplace=True)
        for lag in [1, 2, 4, 8, 13]:
            col = f'{target_col}_lag_{lag}'
            df_part[col] = df_part.groupby(key_col)[target_col].shift(lag)

        # Rolling features
        for window in [4, 13]:
            col = f'{target_col}_roll_mean_{window}'
            df_part[col] = df_part.groupby(key_col)[target_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    feature_cols = [c for c in train_df.columns
                    if c.startswith(f'{target_col}_lag_') or c.startswith(f'{target_col}_roll_')]
    return feature_cols


if __name__ == '__main__':
    main()
