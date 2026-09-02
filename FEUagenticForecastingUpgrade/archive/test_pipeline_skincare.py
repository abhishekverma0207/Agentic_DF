#!/usr/bin/env python3
"""
End-to-end test: EDA → Feature Availability → Segmentation
on DACH SKIN_CARE data (317 keys, small enough for fast iteration).

This runs WITHOUT any LLM — purely deterministic pipeline.
"""
import warnings
warnings.filterwarnings('ignore')

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "sourcedata/DACH/SKIN_CARE.csv"
ARTIFACT_DIR = "artifacts_test_skincare"
KEY_COL = "Model_Hierarchy"
DATE_COL = "Shipment_week"
TARGET_COL = "Actuals"
TIME_FORMAT = "year_week"
FORECAST_HORIZON = 13

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def main():
    t0 = time.time()

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: Loading source data")
    logger.info("=" * 70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"Keys: {df[KEY_COL].nunique()}")
    logger.info(f"Periods: {df[DATE_COL].nunique()} ({df[DATE_COL].min()} to {df[DATE_COL].max()})")

    # =========================================================================
    # STEP 2: RUN EDA (deterministic — compute per-key metrics)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Computing per-key metrics (EDA)")
    logger.info("=" * 70)

    eda_dir = os.path.join(ARTIFACT_DIR, "eda_output")
    os.makedirs(eda_dir, exist_ok=True)

    # Filter to history only
    period_totals = df.groupby(DATE_COL)[TARGET_COL].sum()
    positive_periods = period_totals[period_totals > 0].index.tolist()
    history_cutoff = max(positive_periods) if positive_periods else df[DATE_COL].max()
    hist_df = df[df[DATE_COL] <= history_cutoff]

    logger.info(f"History cutoff: {history_cutoff}")
    logger.info(f"History rows: {len(hist_df):,}")

    # Compute per-key metrics
    def compute_adi(series):
        nonzero = series[series > 0]
        if len(nonzero) <= 1:
            return float(len(series))
        return float(len(series)) / max(len(nonzero), 1)

    per_key_list = []
    for key_val, group in hist_df.groupby(KEY_COL):
        vals = group[TARGET_COL].fillna(0)
        row = {
            KEY_COL: key_val,
            'n_obs': len(vals),
            'mean': vals.mean(),
            'std': vals.std(),
            'min': vals.min(),
            'max': vals.max(),
            'sum': vals.sum(),
            'median': vals.median(),
            'cv': vals.std() / (vals.mean() + 1e-10),
            'zero_count': (vals == 0).sum(),
            'zero_fraction': (vals == 0).mean(),
            'adi': compute_adi(vals),
            'skewness': vals.skew(),
        }
        row['cv2'] = row['cv'] ** 2

        # Trend strength (simple linear regression R²)
        if len(vals) >= 10:
            x = np.arange(len(vals), dtype=float)
            try:
                slope, intercept = np.polyfit(x, vals.values, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((vals.values - y_pred) ** 2)
                ss_tot = np.sum((vals.values - vals.mean()) ** 2)
                row['trend_strength'] = max(0, 1 - ss_res / (ss_tot + 1e-10))
            except Exception:
                row['trend_strength'] = 0.0
        else:
            row['trend_strength'] = 0.0

        # Autocorrelation at lag 1
        if len(vals) >= 5:
            try:
                row['autocorr_lag1'] = float(vals.autocorr(lag=1))
            except Exception:
                row['autocorr_lag1'] = 0.0
        else:
            row['autocorr_lag1'] = 0.0

        per_key_list.append(row)

    per_key = pd.DataFrame(per_key_list)

    # Save EDA outputs
    per_key.to_csv(os.path.join(eda_dir, "per_key_metrics.csv"), index=False)

    # Compute volume tier
    high_thresh = per_key['mean'].quantile(0.7)
    low_thresh = per_key['mean'].quantile(0.3)
    per_key['volume_tier'] = pd.cut(per_key['mean'], bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                      labels=['low', 'medium', 'high'])

    # Create EDA summary
    eda_summary = {
        'n_series': len(per_key),
        'history_cutoff': str(history_cutoff),
        'avg_cv': float(per_key['cv'].mean()),
        'avg_zero_fraction': float(per_key['zero_fraction'].mean()),
        'avg_adi': float(per_key['adi'].mean()),
    }
    with open(os.path.join(eda_dir, "eda_summary.json"), 'w') as f:
        json.dump(eda_summary, f, indent=2)

    logger.info(f"Per-key metrics: {per_key.shape}")
    logger.info(f"  Avg CV: {per_key['cv'].mean():.2f}")
    logger.info(f"  Avg zero_fraction: {per_key['zero_fraction'].mean():.2%}")
    logger.info(f"  Avg ADI: {per_key['adi'].mean():.2f}")

    t1 = time.time()
    logger.info(f"EDA complete in {t1 - t0:.1f}s")

    # =========================================================================
    # STEP 3: FEATURE AVAILABILITY DETECTION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Feature Availability Detection")
    logger.info("=" * 70)

    from utils.feature_availability import run_feature_availability_pipeline

    # Get all candidate feature columns
    exclude_cols = {KEY_COL, DATE_COL, TARGET_COL}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    logger.info(f"Candidate feature columns: {len(feature_cols)}")

    fa_output_dir = os.path.join(ARTIFACT_DIR, "feature_availability_output")

    fa_result = run_feature_availability_pipeline(
        df=df,
        key_cols=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        feature_cols=feature_cols,
        time_format=TIME_FORMAT,
        config_train_end=str(history_cutoff),
        output_dir=fa_output_dir,
    )

    t2 = time.time()
    logger.info(f"Feature Availability complete in {t2 - t1:.1f}s")

    # Load context for next step
    fa_context_path = os.path.join(fa_output_dir, "feature_availability_to_feature_context.json")
    with open(fa_context_path) as f:
        fa_context = json.load(f)

    # =========================================================================
    # STEP 4: ENRICHED SEGMENTATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Enriched Segmentation (Phase 2)")
    logger.info("=" * 70)

    from utils.segmentation import (
        create_segmentation_features,
        get_recommended_clustering_features,
        scale_features,
        cluster_gaussian_mixture,
        enforce_minimum_segment_size,
        create_hybrid_segments,
        compute_segment_profiles,
        assign_model_recommendations,
        create_feature_recommendations,
    )
    from utils.segmentation_enhanced import enrich_segmentation_features

    # Step 4a: Create base segmentation features
    logger.info("[4a] Creating base segmentation features...")
    seg_df = create_segmentation_features(
        per_key_metrics=per_key,
        key_cols=[KEY_COL],
        include_volume=True,
        include_intermittency=True,
        include_variability=True,
        include_temporal=True,
        include_external=True,
        time_format=TIME_FORMAT,
    )
    logger.info(f"Base segmentation features: {len(seg_df.columns)} columns")

    # Step 4b: Enrich with Phase 2 features
    logger.info("[4b] Enriching with Phase 2 features (external response, seasonality, trends, hierarchy)...")
    seg_df, enrichment_meta = enrich_segmentation_features(
        per_key_metrics=seg_df,
        source_df=df,
        key_cols=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        time_format=TIME_FORMAT,
        feature_availability_context=fa_context,
        auto_detect_hierarchy=True,
        compute_response_profiles=True,
        compute_seasonality=True,
        compute_trends=True,
        compute_importance=True,
    )
    logger.info(f"Enriched: {enrichment_meta['n_features_before']} → {enrichment_meta['n_features_after']} columns")
    logger.info(f"Hierarchy detected: {enrichment_meta.get('hierarchy_detected', [])}")

    # Step 4c: Select clustering features
    logger.info("[4c] Selecting clustering features...")
    clustering_features = get_recommended_clustering_features(seg_df, strategy='comprehensive')
    logger.info(f"Clustering features ({len(clustering_features)}): {clustering_features}")

    # Step 4d: Scale and cluster — DATA-ADAPTIVE parameters
    logger.info("[4d] Scaling and clustering...")
    valid_features = [f for f in clustering_features if f in seg_df.columns and pd.api.types.is_numeric_dtype(seg_df[f])]
    logger.info(f"Valid numeric features for clustering: {len(valid_features)}")

    X_scaled, scaler = scale_features(seg_df, valid_features, scaler_type='robust')
    logger.info(f"Scaled matrix: {X_scaled.shape}")

    # Data-adaptive cluster range: 317 keys → [3..12]
    n_keys = len(seg_df)
    if n_keys < 50:
        k_range = list(range(2, 6))
        min_seg_pct = 0.03
    elif n_keys < 200:
        k_range = list(range(3, 9))
        min_seg_pct = 0.05
    elif n_keys < 500:
        k_range = list(range(3, 13))
        min_seg_pct = 0.05
    elif n_keys < 2000:
        k_range = list(range(4, 16))
        min_seg_pct = 0.03
    else:
        k_range = list(range(5, 21))
        min_seg_pct = 0.02
    logger.info(f"Data-adaptive: {n_keys} keys → k_range={k_range}, min_seg_pct={min_seg_pct:.0%}")

    labels, clusterer, metrics = cluster_gaussian_mixture(
        X_scaled, n_range=k_range, auto_select=True
    )
    logger.info(f"GMM clustering: {metrics.n_clusters} clusters, silhouette={metrics.silhouette:.3f}")

    # Step 4e: Enforce minimum size — data-adaptive
    min_size = max(5, int(n_keys * min_seg_pct))
    min_final = max(3, metrics.n_clusters // 2)
    logger.info(f"Enforcement: min_size={min_size}, min_final_segments={min_final}")
    labels = enforce_minimum_segment_size(X_scaled, labels, min_pct=min_seg_pct, min_final_segments=min_final)
    n_after = len(set(labels))
    logger.info(f"After size enforcement: {n_after} clusters")

    if 'volume_tier' in seg_df.columns and 'demand_pattern' in seg_df.columns:
        labels, descriptions = create_hybrid_segments(
            seg_df, labels, use_volume_tier=True, use_demand_pattern=True,
            min_segment_size=min_size, min_final_segments=min_final,
        )
        logger.info(f"Hybrid segments: {len(set(labels))} segments")
    else:
        descriptions = {}

    # Step 4f: Assign segments and save
    seg_df['segment_id'] = labels
    seg_output_dir = os.path.join(ARTIFACT_DIR, "seg_output")
    os.makedirs(seg_output_dir, exist_ok=True)
    seg_df.to_csv(os.path.join(seg_output_dir, "per_key_with_segments.csv"), index=False)

    # Compute profiles
    profile_features = [c for c in valid_features if c in seg_df.columns]
    profiles = compute_segment_profiles(seg_df, np.array(labels), profile_features, key_cols=[KEY_COL])
    with open(os.path.join(seg_output_dir, "segment_profiles.json"), 'w') as f:
        json.dump(profiles, f, indent=2, default=str)

    t3 = time.time()
    logger.info(f"Segmentation complete in {t3 - t2:.1f}s")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = t3 - t0
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Category: SKIN_CARE ({per_key.shape[0]} keys)")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"")
    logger.info(f"EDA:                   {t1 - t0:.1f}s")
    logger.info(f"Feature Availability:  {t2 - t1:.1f}s")
    logger.info(f"Segmentation:          {t3 - t2:.1f}s")
    logger.info(f"")
    logger.info(f"Feature Availability:")
    logger.info(f"  Known in future:     {fa_result.n_known_in_future}")
    logger.info(f"  History only:        {fa_result.n_history_only}")
    logger.info(f"  Partially known:     {fa_result.n_partially_known}")
    logger.info(f"  Excluded:            {fa_result.n_excluded}")
    logger.info(f"")
    logger.info(f"Segmentation:")
    logger.info(f"  Total segments:      {len(set(labels))}")
    logger.info(f"  Clustering features: {len(valid_features)}")
    logger.info(f"  Silhouette score:    {metrics.silhouette:.3f}")
    logger.info(f"  Enriched features:   {len(enrichment_meta.get('features_added', []))}")
    logger.info(f"  Hierarchy detected:  {[h['column'] for h in enrichment_meta.get('hierarchy_detected', [])]}")
    logger.info(f"")
    logger.info(f"Segment distribution:")
    for seg_id in sorted(set(labels)):
        count = sum(1 for l in labels if l == seg_id)
        pct = count / len(labels) * 100
        logger.info(f"  Segment {seg_id}: {count} keys ({pct:.1f}%)")
    logger.info(f"")
    logger.info(f"Output files:")
    for root, dirs, files in os.walk(ARTIFACT_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            logger.info(f"  {os.path.relpath(fpath, ARTIFACT_DIR)}: {size / 1024:.1f} KB")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
