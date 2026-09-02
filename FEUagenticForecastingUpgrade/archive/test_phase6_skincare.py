#!/usr/bin/env python3
"""
Phase 6 End-to-End Test: Rich Feature Engineering
on DACH SKIN_CARE data (317 keys).

Tests:
1. Regime/state features (growth, volatility, momentum, level shift, zero-rate)
2. Cross-key relative features (overall, segment, hierarchy-relative)
3. History-only feature embeddings (from future-unknown features)
4. Full integration: all rich features together
5. Feature impact: compare model accuracy with vs without rich features
"""
import warnings
warnings.filterwarnings('ignore')

import json
import logging
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
    # LOAD & PREPARE
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
    hist_df = df[df[DATE_COL] <= cutoff].copy()

    # Train/val split
    periods = sorted(hist_df[DATE_COL].unique())
    val_size = 13
    train_end = periods[-val_size - 1]
    val_start = periods[-val_size]

    train_df = hist_df[hist_df[DATE_COL] <= train_end].copy()
    val_df = hist_df[hist_df[DATE_COL] >= val_start].copy()

    logger.info(f"History: {len(periods)} periods, cutoff={cutoff}")
    logger.info(f"Train: {len(train_df):,} rows up to {train_end}")
    logger.info(f"Val:   {len(val_df):,} rows from {val_start}")

    # Load feature availability context
    with open('artifacts_test_skincare/feature_availability_output/feature_availability_to_feature_context.json') as f:
        fa_ctx = json.load(f)
    history_only_feats = fa_ctx.get('history_only_features', [])
    logger.info(f"History-only features: {history_only_feats}")

    # =========================================================================
    # TEST 1: REGIME FEATURES
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Regime / State Features")
    logger.info("=" * 70)

    from utils.rich_features import create_regime_features

    t1 = time.time()
    regime_df = create_regime_features(
        df=hist_df.copy(),
        key_cols=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        forecast_lag=FORECAST_LAG,
        short_window=4,
        long_window=26,
    )

    regime_cols = [c for c in regime_df.columns if c.startswith('regime_')]
    logger.info(f"Regime features created: {regime_cols}")

    # Validate: check non-null rate in training period
    train_mask = regime_df[DATE_COL] <= train_end
    for col in regime_cols:
        fill_rate = regime_df.loc[train_mask, col].notna().mean()
        unique = regime_df[col].nunique()
        logger.info(f"  {col}: {fill_rate:.1%} non-null, {unique} unique values")

    assert len(regime_cols) >= 8, f"Expected >=8 regime features, got {len(regime_cols)}"
    logger.info(f"Test 1 PASSED in {time.time() - t1:.1f}s — {len(regime_cols)} features")

    # =========================================================================
    # TEST 2: CROSS-KEY RELATIVE FEATURES
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Cross-Key Relative Features")
    logger.info("=" * 70)

    from utils.rich_features import create_relative_features

    t2 = time.time()

    # Add a fake segment column for testing
    np.random.seed(42)
    keys = hist_df[KEY_COL].unique()
    key_to_seg = {k: f'seg_{i % 5}' for i, k in enumerate(keys)}
    test_df = hist_df.copy()
    test_df['segment_id'] = test_df[KEY_COL].map(key_to_seg)

    rel_df = create_relative_features(
        df=test_df,
        key_cols=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        forecast_lag=FORECAST_LAG,
        segment_col='segment_id',
        hierarchy_cols=[HIER_COL],
    )

    rel_cols = [c for c in rel_df.columns if c.startswith('rel_') or c.startswith('rank_')
                or c.startswith('dist_') or c.startswith('relative_')]
    logger.info(f"Relative features created: {rel_cols}")

    for col in rel_cols:
        fill_rate = rel_df.loc[rel_df[DATE_COL] <= train_end, col].notna().mean()
        logger.info(f"  {col}: {fill_rate:.1%} non-null")

    assert len(rel_cols) >= 6, f"Expected >=6 relative features, got {len(rel_cols)}"
    logger.info(f"Test 2 PASSED in {time.time() - t2:.1f}s — {len(rel_cols)} features")

    # =========================================================================
    # TEST 3: HISTORY-ONLY FEATURE EMBEDDINGS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: History-Only Feature Embeddings")
    logger.info("=" * 70)

    from utils.rich_features import create_history_embeddings

    t3 = time.time()
    emb_df = create_history_embeddings(
        df=hist_df.copy(),
        key_cols=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        future_unknown_features=history_only_feats,
        train_end=train_end,
    )

    emb_cols = [c for c in emb_df.columns if '_hist_' in c or '_target_corr' in c]
    logger.info(f"Embedding features created: {emb_cols}")

    for col in emb_cols:
        vals = emb_df[col].dropna()
        logger.info(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    assert len(emb_cols) >= 4, f"Expected >=4 embedding features, got {len(emb_cols)}"
    logger.info(f"Test 3 PASSED in {time.time() - t3:.1f}s — {len(emb_cols)} features")

    # =========================================================================
    # TEST 4: ALL RICH FEATURES TOGETHER
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: All Rich Features Together (create_all_rich_features)")
    logger.info("=" * 70)

    from utils.rich_features import create_all_rich_features

    t4 = time.time()
    full_df = hist_df.copy()
    full_df['segment_id'] = full_df[KEY_COL].map(key_to_seg)

    enriched_df, rich_meta = create_all_rich_features(
        df=full_df,
        key_cols=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        forecast_lag=FORECAST_LAG,
        train_end=train_end,
        segment_col='segment_id',
        hierarchy_cols=[HIER_COL],
        future_unknown_features=history_only_feats,
        enable_regime=True,
        enable_relative=True,
        enable_history_embeddings=True,
    )

    logger.info(f"\nRich feature summary:")
    logger.info(f"  Regime:     {rich_meta['n_regime']} features")
    logger.info(f"  Relative:   {rich_meta['n_relative']} features")
    logger.info(f"  Embeddings: {rich_meta['n_embeddings']} features")
    logger.info(f"  TOTAL:      {rich_meta['total_features_added']} new features")

    assert rich_meta['total_features_added'] >= 15, \
        f"Expected >=15 rich features, got {rich_meta['total_features_added']}"
    logger.info(f"Test 4 PASSED in {time.time() - t4:.1f}s")

    # =========================================================================
    # TEST 5: FEATURE IMPACT — Model accuracy with vs without rich features
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Feature Impact — With vs Without Rich Features")
    logger.info("=" * 70)

    t5 = time.time()

    # Prepare base features (lags + rolling)
    for part in [train_df, val_df]:
        part.sort_values([KEY_COL, DATE_COL], inplace=True)
        for lag in [1, 2, 4, 8, 13]:
            part[f'{TARGET_COL}_lag_{lag}'] = part.groupby(KEY_COL)[TARGET_COL].shift(lag)
        for w in [4, 13]:
            part[f'{TARGET_COL}_roll_mean_{w}'] = part.groupby(KEY_COL)[TARGET_COL].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    base_feature_cols = [c for c in train_df.columns
                         if c.startswith(f'{TARGET_COL}_lag_') or c.startswith(f'{TARGET_COL}_roll_')]

    # Train model WITHOUT rich features
    import lightgbm as lgb
    X_train_base = train_df[base_feature_cols].fillna(0).values
    y_train = train_df[TARGET_COL].fillna(0).values
    X_val_base = val_df[base_feature_cols].fillna(0).values
    y_val = val_df[TARGET_COL].fillna(0).values

    model_base = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, verbosity=-1,
    )
    model_base.fit(X_train_base, y_train,
                   eval_set=[(X_val_base, y_val)],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
    preds_base = np.maximum(model_base.predict(X_val_base), 0)
    wape_base = float(np.sum(np.abs(y_val - preds_base)) / (np.sum(np.abs(y_val)) + 1e-10))

    # Now add rich features to train/val
    train_rich = train_df.copy()
    train_rich['segment_id'] = train_rich[KEY_COL].map(key_to_seg)
    val_rich = val_df.copy()
    val_rich['segment_id'] = val_rich[KEY_COL].map(key_to_seg)

    # Combine for rich feature creation (need both periods)
    combined = pd.concat([train_rich, val_rich], ignore_index=True)
    combined, _ = create_all_rich_features(
        df=combined, key_cols=[KEY_COL], date_col=DATE_COL, target_col=TARGET_COL,
        forecast_lag=FORECAST_LAG, train_end=train_end, segment_col='segment_id',
        hierarchy_cols=[HIER_COL], future_unknown_features=history_only_feats,
    )

    # Split back
    train_enriched = combined[combined[DATE_COL] <= train_end]
    val_enriched = combined[combined[DATE_COL] >= val_start]

    # Get rich feature columns
    rich_cols = [c for c in combined.columns if c.startswith('regime_') or c.startswith('rel_')
                 or c.startswith('rank_') or c.startswith('dist_') or c.startswith('relative_')
                 or '_hist_' in c or '_target_corr' in c]
    all_feature_cols = base_feature_cols + [c for c in rich_cols if c in combined.columns]

    X_train_rich = train_enriched[all_feature_cols].fillna(0).values
    X_val_rich = val_enriched[all_feature_cols].fillna(0).values

    model_rich = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, verbosity=-1,
    )
    model_rich.fit(X_train_rich, train_enriched[TARGET_COL].fillna(0).values,
                   eval_set=[(X_val_rich, val_enriched[TARGET_COL].fillna(0).values)],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
    preds_rich = np.maximum(model_rich.predict(X_val_rich), 0)
    wape_rich = float(np.sum(np.abs(y_val - preds_rich)) / (np.sum(np.abs(y_val)) + 1e-10))

    improvement = wape_base - wape_rich
    improvement_pct = improvement / wape_base * 100

    logger.info(f"\n{'Metric':<30} {'Without Rich':<15} {'With Rich':<15} {'Change':<15}")
    logger.info("-" * 75)
    logger.info(f"{'Val WAPE':<30} {wape_base:<15.4f} {wape_rich:<15.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
    logger.info(f"{'Features used':<30} {len(base_feature_cols):<15} {len(all_feature_cols):<15}")

    # Feature importance of rich features
    importances = model_rich.feature_importances_
    feat_imp = sorted(zip(all_feature_cols, importances), key=lambda x: -x[1])
    logger.info(f"\nTop 15 features by importance (with rich features):")
    for feat, imp in feat_imp[:15]:
        marker = " ★" if feat in rich_cols else ""
        logger.info(f"  {feat:<40} {imp:>6.0f}{marker}")

    n_rich_in_top15 = sum(1 for f, _ in feat_imp[:15] if f in rich_cols)
    logger.info(f"\n{n_rich_in_top15}/15 top features are rich features (★)")

    logger.info(f"Test 5 PASSED in {time.time() - t5:.1f}s")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("ALL PHASE 6 TESTS PASSED")
    logger.info("=" * 70)
    logger.info(f"Total time: {total:.1f}s")
    logger.info(f"")
    logger.info(f"Phase 6 Features:")
    logger.info(f"  Regime features:            {rich_meta['n_regime']}")
    logger.info(f"  Cross-key relative:         {rich_meta['n_relative']}")
    logger.info(f"  History embeddings:          {rich_meta['n_embeddings']}")
    logger.info(f"  Total rich features:         {rich_meta['total_features_added']}")
    logger.info(f"")
    logger.info(f"Impact on accuracy:")
    logger.info(f"  WAPE without rich features:  {wape_base:.4f}")
    logger.info(f"  WAPE with rich features:     {wape_rich:.4f}")
    logger.info(f"  Improvement:                 {improvement:+.4f} ({improvement_pct:+.1f}%)")
    logger.info(f"  Rich features in top 15:     {n_rich_in_top15}/15")


if __name__ == '__main__':
    main()
