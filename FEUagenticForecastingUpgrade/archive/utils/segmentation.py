"""
World-Class Demand Segmentation Utilities for Demand Forecasting.

This module provides COMPREHENSIVE, PRODUCTION-READY segmentation capabilities
that go FAR BEYOND simple ADI/CV2 classification.

DESIGN PHILOSOPHY:
==================
1. MULTI-DIMENSIONAL SEGMENTATION: Not just intermittency, but volume, variability,
   trend, seasonality, and forecastability.
2. HIERARCHICAL APPROACH: First volume tier, then pattern, then refinement.
3. BUSINESS INTERPRETABILITY: Segments must be meaningful for business users.
4. UTILITY-FIRST: Agents call functions, don't generate code from scratch.

KEY SEGMENTATION DIMENSIONS:
============================
1. VOLUME TIER: High/Medium/Low volume (business priority)
2. DEMAND PATTERN: Smooth/Erratic/Intermittent/Lumpy (Syntetos-Boylan)
3. VARIABILITY: Stable/Moderate/Variable (coefficient of variation)
4. TREND: Trending-Up/Stable/Trending-Down
5. SEASONALITY: Strong/Moderate/Weak/None
6. FORECASTABILITY: Easy/Medium/Hard (based on predictability metrics)

USAGE:
======
>>> from utils.segmentation import (
...     # HIGH-LEVEL ONE-CALL PIPELINE
...     run_segmentation_pipeline,
...
...     # MULTI-DIMENSIONAL SEGMENTATION
...     create_segmentation_features,
...     segment_by_volume_tier,
...     segment_by_demand_pattern,
...     segment_hierarchical,
...
...     # CLUSTERING ALGORITHMS
...     cluster_kmeans, cluster_gaussian_mixture, cluster_hdbscan,
...     cluster_hierarchical_optimal,
...
...     # SEGMENT PROFILING
...     compute_segment_profiles, profile_segments_detailed,
...     create_segment_summary, label_segments_automatically,
...
...     # MODEL ASSIGNMENT
...     assign_model_recommendations, create_training_strategy,
...
...     # VALIDATION & VISUALIZATION
...     validate_segments, plot_segment_analysis,
... )

Author: FEU-Agentic-Forecasting Demand Forecasting System
Version: 2.0 - World-Class Multi-Dimensional Segmentation
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Syntetos-Boylan Thresholds (EXACT)
# =============================================================================

ADI_THRESHOLD = 1.32  # Average Demand Interval threshold (weekly data)
CV2_THRESHOLD = 0.49  # Squared Coefficient of Variation threshold (weekly data)

# Monthly-adapted thresholds: with ~12 observations, ADI and CV² are naturally
# inflated. A series with 6 non-zero months already has ADI=2.0 (>1.32).
# Relaxed thresholds prevent over-classifying monthly data as lumpy.
ADI_THRESHOLD_MONTHLY = 2.0   # Monthly: allow up to 50% zeros before intermittent
CV2_THRESHOLD_MONTHLY = 0.75  # Monthly: CV² inflated with few observations

# Volume tier thresholds (percentile-based)
VOLUME_HIGH_PERCENTILE = 0.70  # Top 30% are high volume
VOLUME_LOW_PERCENTILE = 0.30   # Bottom 30% are low volume

# Minimum segment size
DEFAULT_MIN_SEGMENT_PCT = 0.10  # 10% minimum (but capped at 80 - see MIN_SEGMENT_SIZE_CAP)
MIN_SEGMENT_SIZE_CAP = 80  # Absolute maximum for min segment size (allows more granular segments)

# Variability thresholds
CV_LOW_THRESHOLD = 0.5    # CV < 0.5 is stable
CV_HIGH_THRESHOLD = 1.5   # CV > 1.5 is highly variable


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClusteringMetrics:
    """Metrics for evaluating clustering quality."""
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    n_clusters: int
    n_noise: int = 0


@dataclass
class SegmentationResult:
    """Complete result container for segmentation pipeline."""
    labels: np.ndarray
    n_segments: int
    segment_profiles: Dict[str, Any]
    segment_distribution: Dict[str, float]
    segment_labels: Dict[str, str]  # segment_id -> human-readable label
    clustering_metrics: Dict[str, float]
    model_recommendations: Dict[str, Any]
    feature_recommendations: Dict[str, Any]
    scaler: Any = None
    clusterer: Any = None
    segmentation_features_used: List[str] = field(default_factory=list)


# =============================================================================
# 1. FEATURE ENGINEERING FOR SEGMENTATION
# =============================================================================

def create_segmentation_features(
    per_key_metrics: pd.DataFrame,
    key_cols: List[str],
    include_volume: bool = True,
    include_intermittency: bool = True,
    include_variability: bool = True,
    include_temporal: bool = True,
    include_external: bool = True,
    external_feature_cols: Optional[List[str]] = None,
    price_agg_cols: Optional[List[str]] = None,
    promo_agg_cols: Optional[List[str]] = None,
    categorical_encoded_cols: Optional[List[str]] = None,
    min_variance_threshold: float = 1e-6,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create comprehensive segmentation features from per-key metrics.

    This function computes ALL dimensions needed for world-class segmentation:
    - Volume metrics (mean, total, frequency)
    - Intermittency metrics (ADI, CV2, zero_fraction)
    - Variability metrics (CV, std, range)
    - Temporal metrics (trend, seasonality if available)
    - External features (price sensitivity, promo responsiveness)

    Parameters
    ----------
    per_key_metrics : pd.DataFrame
        DataFrame with per-key metrics from EDA (must have: mean, std, cv, adi, cv2, zero_fraction)
    key_cols : List[str]
        Key identifier columns
    include_volume : bool
        Include volume-based features
    include_intermittency : bool
        Include intermittency features (ADI, CV2)
    include_variability : bool
        Include variability features
    include_temporal : bool
        Include temporal features if available
    include_external : bool
        Include external feature aggregates (price, promo) if available
    external_feature_cols : List[str], optional
        DEPRECATED: Use price_agg_cols/promo_agg_cols instead.
        Explicit list of external feature columns to use. If None, uses config-driven columns.
    price_agg_cols : List[str], optional
        List of price aggregate column names from compute_demand_characteristics
    promo_agg_cols : List[str], optional
        List of promo aggregate column names from compute_demand_characteristics
    categorical_encoded_cols : List[str], optional
        List of categorical encoded column names (e.g., promo_type_encoded_mean)
    min_variance_threshold : float
        Minimum variance for including a feature (default 1e-6)

    Returns
    -------
    pd.DataFrame
        DataFrame with comprehensive segmentation features

    Example
    -------
    >>> seg_features = create_segmentation_features(per_key_metrics, ['key'],
    ...     price_agg_cols=['price_unit_price_mean', 'price_unit_price_cv'],
    ...     promo_agg_cols=['promo_discount_mean', 'promo_discount_frequency'])
    >>> print(seg_features.columns)
    """
    df = per_key_metrics.copy()

    # Ensure we have the key columns
    for col in key_cols:
        if col not in df.columns:
            raise ValueError(f"Key column '{col}' not found in per_key_metrics")

    features_created = []

    # -------------------------------------------------------------------------
    # VOLUME FEATURES
    # -------------------------------------------------------------------------
    if include_volume:
        # Mean demand (already exists as 'mean')
        if 'mean' in df.columns:
            df['volume_mean'] = df['mean'].fillna(0)
            features_created.append('volume_mean')

        # Total demand
        if 'sum' in df.columns:
            df['volume_total'] = df['sum'].fillna(0)
            features_created.append('volume_total')
        elif 'mean' in df.columns and 'n_obs' in df.columns:
            df['volume_total'] = df['mean'] * df['n_obs']
            features_created.append('volume_total')

        # Demand frequency (non-zero periods / total periods)
        if 'zero_fraction' in df.columns:
            df['demand_frequency'] = 1 - df['zero_fraction'].fillna(0)
            features_created.append('demand_frequency')
        elif 'zero_count' in df.columns and 'n_obs' in df.columns:
            df['demand_frequency'] = 1 - (df['zero_count'] / df['n_obs'])
            features_created.append('demand_frequency')

        # Volume tier (computed dynamically based on percentiles)
        if 'volume_mean' in df.columns:
            high_threshold = df['volume_mean'].quantile(VOLUME_HIGH_PERCENTILE)
            low_threshold = df['volume_mean'].quantile(VOLUME_LOW_PERCENTILE)

            df['volume_tier'] = pd.cut(
                df['volume_mean'],
                bins=[-np.inf, low_threshold, high_threshold, np.inf],
                labels=['low', 'medium', 'high']
            )
            df['volume_tier_numeric'] = df['volume_tier'].map({
                'low': 0, 'medium': 1, 'high': 2
            }).fillna(1)
            features_created.extend(['volume_tier', 'volume_tier_numeric'])

    # -------------------------------------------------------------------------
    # INTERMITTENCY FEATURES (Syntetos-Boylan)
    # -------------------------------------------------------------------------
    if include_intermittency:
        # ADI - Average Demand Interval
        if 'adi' in df.columns:
            df['adi_clean'] = df['adi'].fillna(1).clip(lower=0.1)
            # Log transform for better clustering (ADI can have large range)
            df['adi_log'] = np.log1p(df['adi_clean'])
            features_created.extend(['adi_clean', 'adi_log'])

        # CV² - Squared Coefficient of Variation
        if 'cv2' in df.columns:
            df['cv2_clean'] = df['cv2'].fillna(0).clip(lower=0, upper=10)
            features_created.append('cv2_clean')
        elif 'cv' in df.columns:
            df['cv2_clean'] = (df['cv'].fillna(0) ** 2).clip(lower=0, upper=10)
            features_created.append('cv2_clean')

        # Zero fraction
        if 'zero_fraction' in df.columns:
            df['zero_fraction_clean'] = df['zero_fraction'].fillna(0).clip(0, 1)
            features_created.append('zero_fraction_clean')

        # Demand pattern classification (Syntetos-Boylan)
        if 'adi_clean' in df.columns and 'cv2_clean' in df.columns:
            _tf = time_format
            # Vectorised replacement for the row-wise .apply that used to
            # be: df.apply(lambda r: classify_demand_pattern(...), axis=1).
            # On a 100k-key frame this was a 100k Python function call
            # loop; np.select is two boolean broadcasts. 10-50x faster.
            #
            # Logic mirrors classify_demand_pattern exactly:
            #   adi <= thr & cv2 <= thr -> smooth
            #   adi <= thr & cv2 >  thr -> erratic
            #   adi >  thr & cv2 <= thr -> intermittent
            #   adi >  thr & cv2 >  thr -> lumpy
            adi_thr = ADI_THRESHOLD_MONTHLY if _tf == 'year_month' else ADI_THRESHOLD
            cv2_thr = CV2_THRESHOLD_MONTHLY if _tf == 'year_month' else CV2_THRESHOLD
            adi_arr = df['adi_clean'].values
            cv2_arr = df['cv2_clean'].values
            df['demand_pattern'] = np.select(
                [
                    (adi_arr <= adi_thr) & (cv2_arr <= cv2_thr),
                    (adi_arr <= adi_thr) & (cv2_arr >  cv2_thr),
                    (adi_arr >  adi_thr) & (cv2_arr <= cv2_thr),
                    (adi_arr >  adi_thr) & (cv2_arr >  cv2_thr),
                ],
                ['smooth', 'erratic', 'intermittent', 'lumpy'],
                default='smooth',
            )
            # Numeric encoding for clustering
            pattern_map = {'smooth': 0, 'erratic': 1, 'intermittent': 2, 'lumpy': 3}
            df['demand_pattern_numeric'] = df['demand_pattern'].map(pattern_map).fillna(0)
            features_created.extend(['demand_pattern', 'demand_pattern_numeric'])

            # Pattern family: binary indicator (0=regular: smooth/erratic, 1=sparse: intermittent/lumpy)
            # This is a stronger clustering signal than the 4-way ordinal
            family_map = {'smooth': 0, 'erratic': 0, 'intermittent': 1, 'lumpy': 1}
            df['pattern_family'] = df['demand_pattern'].map(family_map).fillna(0).astype(int)
            features_created.append('pattern_family')

        # Composite intermittency score (continuous 0-1 scale)
        # Combines ADI, zero_fraction, and CV² into a single continuous metric
        # that gives the clustering algorithm a meaningful distance for intermittency
        # severity — keys with ADI=5.0 and ADI=1.5 are far apart even though both
        # may be classified as "intermittent"
        _adi_thresh = ADI_THRESHOLD_MONTHLY if time_format == 'year_month' else ADI_THRESHOLD
        _cv2_thresh = CV2_THRESHOLD_MONTHLY if time_format == 'year_month' else CV2_THRESHOLD

        has_adi = 'adi_clean' in df.columns
        has_zf = 'zero_fraction_clean' in df.columns
        has_cv2 = 'cv2_clean' in df.columns

        if has_adi and has_zf and has_cv2:
            # Normalize each component to 0-1 scale, capped at 3x threshold
            adi_normalized = (df['adi_clean'] / _adi_thresh).clip(upper=3.0) / 3.0
            cv2_normalized = (df['cv2_clean'] / _cv2_thresh).clip(upper=3.0) / 3.0
            zf_component = df['zero_fraction_clean']  # already 0-1

            # Weighted combination: ADI most important, then zero_fraction, then CV²
            df['intermittency_score'] = (
                0.5 * adi_normalized +
                0.3 * zf_component +
                0.2 * cv2_normalized
            ).clip(0, 1)
            features_created.append('intermittency_score')
            logger.info(
                f"Created intermittency_score: mean={df['intermittency_score'].mean():.3f}, "
                f"std={df['intermittency_score'].std():.3f}"
            )

    # -------------------------------------------------------------------------
    # VARIABILITY FEATURES
    # -------------------------------------------------------------------------
    if include_variability:
        # CV - Coefficient of Variation
        if 'cv' in df.columns:
            df['cv_clean'] = df['cv'].fillna(0).clip(lower=0, upper=10)
            features_created.append('cv_clean')

        # Standard deviation (normalized by mean for comparability)
        if 'std' in df.columns and 'mean' in df.columns:
            df['std_normalized'] = df['std'] / (df['mean'].abs() + 1e-10)
            df['std_normalized'] = df['std_normalized'].clip(0, 10)
            features_created.append('std_normalized')

        # Range (max - min)
        if 'max' in df.columns and 'min' in df.columns:
            df['demand_range'] = df['max'] - df['min']
            df['demand_range_normalized'] = df['demand_range'] / (df['mean'].abs() + 1e-10)
            features_created.extend(['demand_range', 'demand_range_normalized'])

        # Variability tier
        if 'cv_clean' in df.columns:
            df['variability_tier'] = pd.cut(
                df['cv_clean'],
                bins=[-np.inf, CV_LOW_THRESHOLD, CV_HIGH_THRESHOLD, np.inf],
                labels=['stable', 'moderate', 'variable']
            )
            df['variability_tier_numeric'] = df['variability_tier'].map({
                'stable': 0, 'moderate': 1, 'variable': 2
            }).fillna(1)
            features_created.extend(['variability_tier', 'variability_tier_numeric'])

    # -------------------------------------------------------------------------
    # TEMPORAL FEATURES (if available)
    # -------------------------------------------------------------------------
    if include_temporal:
        # Trend strength (if computed in EDA)
        if 'trend_strength' in df.columns:
            df['trend_strength_clean'] = df['trend_strength'].fillna(0).clip(0, 1)
            features_created.append('trend_strength_clean')

        # Seasonal strength (if computed in EDA)
        if 'seasonal_strength' in df.columns:
            df['seasonal_strength_clean'] = df['seasonal_strength'].fillna(0).clip(0, 1)
            features_created.append('seasonal_strength_clean')

        # Autocorrelation at lag 1 (if computed in EDA)
        if 'autocorr_lag1' in df.columns:
            df['autocorr_lag1_clean'] = df['autocorr_lag1'].fillna(0).clip(-1, 1)
            features_created.append('autocorr_lag1_clean')

        # History length (n_obs)
        if 'n_obs' in df.columns:
            df['history_length'] = df['n_obs'].fillna(0)
            features_created.append('history_length')

    # -------------------------------------------------------------------------
    # DERIVED COMPOSITE FEATURES
    # -------------------------------------------------------------------------

    # Forecastability score (composite) - ENHANCED with more components
    forecastability_components = []
    weights = []

    if 'demand_frequency' in df.columns:
        forecastability_components.append(df['demand_frequency'])
        weights.append(0.3)  # High weight - frequency is critical
    if 'cv_clean' in df.columns:
        # Inverse CV - lower variability = higher forecastability
        cv_score = 1 - (df['cv_clean'] / (df['cv_clean'].max() + 1e-10))
        forecastability_components.append(cv_score.clip(0, 1))
        weights.append(0.25)
    if 'autocorr_lag1_clean' in df.columns:
        # Higher autocorrelation = more predictable
        forecastability_components.append(df['autocorr_lag1_clean'].abs())
        weights.append(0.2)
    if 'trend_strength_clean' in df.columns:
        # Trend presence helps prediction
        forecastability_components.append(df['trend_strength_clean'])
        weights.append(0.15)
    if 'seasonal_strength_clean' in df.columns:
        # Seasonality helps prediction
        forecastability_components.append(df['seasonal_strength_clean'])
        weights.append(0.1)

    if forecastability_components:
        # Weighted average if we have weights, otherwise simple mean
        if len(weights) == len(forecastability_components):
            weights = np.array(weights) / sum(weights)  # Normalize
            df['forecastability_score'] = sum(w * c for w, c in zip(weights, forecastability_components))
        else:
            df['forecastability_score'] = np.mean(forecastability_components, axis=0)
        df['forecastability_score'] = df['forecastability_score'].fillna(0.5).clip(0, 1)
        features_created.append('forecastability_score')

    # -------------------------------------------------------------------------
    # ADVANCED FEATURES - Lifecycle, Stability, Complexity
    # -------------------------------------------------------------------------

    # Demand Lifecycle Stage (growth, mature, decline)
    if 'trend_strength' in df.columns or 'trend_strength_clean' in df.columns:
        trend_col = 'trend_strength_clean' if 'trend_strength_clean' in df.columns else 'trend_strength'
        # Positive trend = growth, near-zero = mature, negative = decline
        # Assuming trend_strength is absolute, we need direction if available
        if 'trend_direction' in df.columns:
            # Vectorised replacement for the row-wise .apply.
            # Original logic:
            #   trend_dir > 0.1 AND trend_strength > 0.3 -> growth
            #   trend_dir < -0.1 AND trend_strength > 0.3 -> decline
            #   else -> mature
            tdir = df['trend_direction'].values
            tstr = df[trend_col].values
            df['lifecycle_stage'] = np.select(
                [
                    (tdir > 0.1)  & (tstr > 0.3),
                    (tdir < -0.1) & (tstr > 0.3),
                ],
                ['growth', 'decline'],
                default='mature',
            )
        else:
            # Without direction, use trend strength as proxy for maturity
            df['lifecycle_stage'] = pd.cut(
                df[trend_col].fillna(0),
                bins=[-np.inf, 0.2, 0.5, np.inf],
                labels=['mature', 'transitioning', 'dynamic']
            )
        df['lifecycle_stage_numeric'] = df['lifecycle_stage'].map(
            {'mature': 0, 'transitioning': 1, 'dynamic': 2, 'growth': 2, 'decline': 2}
        ).fillna(1)
        features_created.extend(['lifecycle_stage', 'lifecycle_stage_numeric'])

    # Demand Stability Score (inverse of combined variability measures)
    stability_components = []
    if 'cv_clean' in df.columns:
        stability_components.append(1 - df['cv_clean'].clip(0, 3) / 3)
    if 'zero_fraction_clean' in df.columns:
        stability_components.append(1 - df['zero_fraction_clean'])

    if stability_components:
        df['stability_score'] = np.mean(stability_components, axis=0).clip(0, 1)
        features_created.append('stability_score')

    # Forecasting Complexity Score (higher = harder to forecast)
    complexity_components = []
    if 'cv_clean' in df.columns:
        complexity_components.append(df['cv_clean'].clip(0, 3) / 3)  # High CV = complex
    if 'zero_fraction_clean' in df.columns:
        complexity_components.append(df['zero_fraction_clean'])  # High zeros = complex
    if 'demand_pattern_numeric' in df.columns:
        complexity_components.append(df['demand_pattern_numeric'] / 3)  # Lumpy = most complex

    if complexity_components:
        df['complexity_score'] = np.mean(complexity_components, axis=0).clip(0, 1)
        features_created.append('complexity_score')

    # -------------------------------------------------------------------------
    # EXTERNAL FEATURE METRICS (Price, Promo, etc.)
    # These help identify price-sensitive vs base-demand segments
    # USES CONFIG-DRIVEN COLUMN LISTS (not pattern detection)
    # -------------------------------------------------------------------------
    if include_external:
        # Helper function to validate feature has sufficient variance
        def _validate_variance(col: str, data: pd.DataFrame) -> bool:
            """Check if column has sufficient variance."""
            if col not in data.columns:
                return False
            col_data = data[col].dropna()
            if len(col_data) == 0:
                return False
            try:
                variance = col_data.var()
                return variance >= min_variance_threshold
            except Exception:
                return False

        # Collect all valid external feature columns from config-driven lists
        valid_price_cols = []
        valid_promo_cols = []

        # Use config-driven column lists if provided
        if price_agg_cols:
            valid_price_cols = [c for c in price_agg_cols if c in df.columns and _validate_variance(c, df)]
            if valid_price_cols:
                logger.info(f"Using {len(valid_price_cols)} config-defined price aggregate columns")

        if promo_agg_cols:
            valid_promo_cols = [c for c in promo_agg_cols if c in df.columns and _validate_variance(c, df)]
            if valid_promo_cols:
                logger.info(f"Using {len(valid_promo_cols)} config-defined promo aggregate columns")

        # Fallback: use external_feature_cols if provided (deprecated but supported)
        if not valid_price_cols and not valid_promo_cols and external_feature_cols:
            logger.warning("Using deprecated external_feature_cols parameter. Please use price_agg_cols/promo_agg_cols.")
            for col in external_feature_cols:
                if col in df.columns and _validate_variance(col, df):
                    if 'price' in col.lower():
                        valid_price_cols.append(col)
                    elif any(p in col.lower() for p in ['promo', 'discount', 'promotion']):
                        valid_promo_cols.append(col)

        # Process PRICE features
        if valid_price_cols:
            # Price level (use first mean column)
            price_mean_cols = [c for c in valid_price_cols if 'mean' in c.lower()]
            if price_mean_cols:
                df['price_level'] = df[price_mean_cols[0]].fillna(df[price_mean_cols[0]].median())
                features_created.append('price_level')

                # Price tier (high/medium/low)
                try:
                    price_high = df['price_level'].quantile(0.7)
                    price_low = df['price_level'].quantile(0.3)
                    df['price_tier'] = pd.cut(
                        df['price_level'],
                        bins=[-np.inf, price_low, price_high, np.inf],
                        labels=['low', 'medium', 'high']
                    )
                    df['price_tier_numeric'] = df['price_tier'].map({
                        'low': 0, 'medium': 1, 'high': 2
                    }).fillna(1)
                    features_created.extend(['price_tier', 'price_tier_numeric'])
                except Exception as e:
                    logger.debug(f"Could not create price_tier: {e}")

            # Price variability (CV of price)
            price_cv_cols = [c for c in valid_price_cols if 'cv' in c.lower() or 'std' in c.lower()]
            if price_cv_cols:
                df['price_variability'] = df[price_cv_cols[0]].fillna(0).clip(0, 5)
                features_created.append('price_variability')

        # Process PROMO features
        if valid_promo_cols:
            # Promo intensity (average promo level or frequency)
            promo_mean_cols = [c for c in valid_promo_cols if 'mean' in c.lower() or 'frequency' in c.lower()]
            if promo_mean_cols:
                df['promo_intensity'] = df[promo_mean_cols[0]].fillna(0).clip(0, 1)
                features_created.append('promo_intensity')

                # Promo tier (high/medium/low promo activity)
                try:
                    promo_high = df['promo_intensity'].quantile(0.7)
                    promo_low = df['promo_intensity'].quantile(0.3)
                    df['promo_tier'] = pd.cut(
                        df['promo_intensity'],
                        bins=[-np.inf, promo_low, promo_high, np.inf],
                        labels=['low', 'medium', 'high']
                    )
                    df['promo_tier_numeric'] = df['promo_tier'].map({
                        'low': 0, 'medium': 1, 'high': 2
                    }).fillna(0)
                    features_created.extend(['promo_tier', 'promo_tier_numeric'])
                except Exception as e:
                    logger.debug(f"Could not create promo_tier: {e}")

        # External feature sensitivity composite score
        # Combines price and promo to identify externally-driven vs base-demand keys
        external_sensitivity_components = []
        if 'price_variability' in df.columns:
            external_sensitivity_components.append(df['price_variability'].clip(0, 1))
        if 'promo_intensity' in df.columns:
            external_sensitivity_components.append(df['promo_intensity'])

        if external_sensitivity_components:
            df['external_sensitivity_score'] = np.mean(external_sensitivity_components, axis=0).clip(0, 1)
            features_created.append('external_sensitivity_score')

            # Create external sensitivity tier
            try:
                ext_high = df['external_sensitivity_score'].quantile(0.7)
                ext_low = df['external_sensitivity_score'].quantile(0.3)
                df['external_sensitivity_tier'] = pd.cut(
                    df['external_sensitivity_score'],
                    bins=[-np.inf, ext_low, ext_high, np.inf],
                    labels=['base_demand', 'moderate', 'externally_driven']
                )
                df['external_sensitivity_numeric'] = df['external_sensitivity_tier'].map({
                    'base_demand': 0, 'moderate': 1, 'externally_driven': 2
                }).fillna(1)
                features_created.extend(['external_sensitivity_tier', 'external_sensitivity_numeric'])
            except Exception as e:
                logger.debug(f"Could not create external_sensitivity_tier: {e}")

        # ---------------------------------------------------------------------
        # CATEGORICAL EXTERNAL FEATURE ENCODING
        # These are pre-computed in compute_demand_characteristics as encoded aggregates
        # Here we create normalized/tiered versions for clustering
        # USES CONFIG-DRIVEN COLUMN LISTS (not pattern detection)
        # ---------------------------------------------------------------------

        # Use config-driven categorical encoded columns if provided
        cat_encoded_cols = []
        if categorical_encoded_cols:
            cat_encoded_cols = [c for c in categorical_encoded_cols if c in df.columns and _validate_variance(c, df)]
            if cat_encoded_cols:
                logger.info(f"Using {len(cat_encoded_cols)} config-defined categorical encoded columns")

        if cat_encoded_cols:
            for col in cat_encoded_cols:
                try:
                    # Clean and normalize the encoded values
                    col_clean = f"{col}_clean"
                    df[col_clean] = df[col].fillna(0).clip(0, None)

                    # For encoded_mean columns, create tiered features
                    if '_encoded_mean' in col:
                        base_name = col.replace('_encoded_mean', '')
                        try:
                            high_thresh = df[col_clean].quantile(0.7)
                            low_thresh = df[col_clean].quantile(0.3)
                            tier_col = f"{base_name}_cat_tier"
                            tier_numeric_col = f"{base_name}_cat_tier_numeric"

                            df[tier_col] = pd.cut(
                                df[col_clean],
                                bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                labels=['low_freq_type', 'medium_freq_type', 'high_freq_type']
                            )
                            df[tier_numeric_col] = df[tier_col].map({
                                'low_freq_type': 0, 'medium_freq_type': 1, 'high_freq_type': 2
                            }).fillna(1)
                            features_created.extend([col_clean, tier_numeric_col])
                        except Exception as tier_e:
                            # Just use the clean column
                            features_created.append(col_clean)
                            logger.debug(f"Could not create tier for {col}: {tier_e}")
                    else:
                        features_created.append(col_clean)

                except Exception as e:
                    logger.debug(f"Could not process categorical encoded column {col}: {e}")

            # Create categorical diversity score (composite)
            # Higher diversity = more varied category usage = potentially more complex
            diversity_cols = [c for c in df.columns if '_diversity' in c.lower() and c.endswith('_clean')]
            if len(diversity_cols) > 1:
                try:
                    # Normalize each diversity column and average
                    normalized_diversity = []
                    for dc in diversity_cols:
                        dc_max = df[dc].max()
                        if dc_max > 0:
                            normalized_diversity.append(df[dc] / dc_max)
                    if normalized_diversity:
                        df['categorical_diversity_score'] = np.mean(normalized_diversity, axis=0).clip(0, 1)
                        features_created.append('categorical_diversity_score')
                except Exception as e:
                    logger.debug(f"Could not create categorical_diversity_score: {e}")

    # Log information
    logger.info(f"Created {len(features_created)} segmentation features: {features_created}")

    return df


def filter_low_variance_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_variance_ratio: float = 0.001,
    min_unique_ratio: float = 0.01,
) -> List[str]:
    """
    Filter out low-variance and near-constant features before clustering.

    Low-variance features cause numerical issues in clustering algorithms
    (singular covariance matrices in GMM, poor separation in K-Means).
    This function identifies and removes such features EARLY in the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features
    feature_cols : List[str]
        List of feature column names to evaluate
    min_variance_ratio : float
        Minimum variance as ratio of max variance across features.
        Features with variance < max_variance * min_variance_ratio are removed.
        Default 0.001 (0.1% of max variance).
    min_unique_ratio : float
        Minimum ratio of unique values to total rows.
        Features where unique_count / n_rows < min_unique_ratio are removed.
        Default 0.01 (1% unique values).

    Returns
    -------
    List[str]
        Filtered list of feature columns with low-variance features removed.

    Example
    -------
    >>> features = ['volume_mean', 'cv_clean', 'lifecycle_stage_numeric']
    >>> filtered = filter_low_variance_features(df, features)
    >>> print(f"Kept {len(filtered)} of {len(features)} features")
    """
    if not feature_cols:
        return feature_cols

    # Only evaluate NUMERIC features that exist in df
    # Skip categorical columns - they don't support variance computation
    valid_features = []
    for f in feature_cols:
        if f not in df.columns:
            continue
        # Check if column is numeric (not categorical/object)
        dtype = df[f].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_categorical_dtype(dtype):
            valid_features.append(f)
        else:
            logger.debug(f"Skipping non-numeric feature '{f}' (dtype={dtype}) in variance filter")

    if not valid_features:
        logger.warning("No valid numeric features found in DataFrame for variance filtering")
        # Return original features that exist in df
        return [f for f in feature_cols if f in df.columns]

    n_rows = len(df)
    removed_features = []
    kept_features = []

    # Compute variance for all numeric features
    variances = {}
    unique_counts = {}
    for feat in valid_features:
        col_data = df[feat].dropna()
        if len(col_data) > 0:
            try:
                variances[feat] = float(col_data.var())
            except (TypeError, ValueError):
                # Fallback if var() still fails for some reason
                variances[feat] = 0
                logger.warning(f"Could not compute variance for '{feat}', defaulting to 0")
            unique_counts[feat] = col_data.nunique()
        else:
            variances[feat] = 0
            unique_counts[feat] = 0

    max_variance = max(variances.values()) if variances else 0
    variance_threshold = max_variance * min_variance_ratio

    for feat in valid_features:
        var = variances.get(feat, 0)
        n_unique = unique_counts.get(feat, 0)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0

        # Remove if: (1) variance too low OR (2) too few unique values
        is_low_variance = var < variance_threshold
        is_near_constant = unique_ratio < min_unique_ratio

        if is_low_variance or is_near_constant:
            removed_features.append({
                'feature': feat,
                'variance': var,
                'variance_ratio': var / (max_variance + 1e-10),
                'unique_count': n_unique,
                'unique_ratio': unique_ratio,
                'reason': 'low_variance' if is_low_variance else 'near_constant'
            })
        else:
            kept_features.append(feat)

    # Log removed features with details
    if removed_features:
        logger.warning(
            f"Removed {len(removed_features)} low-variance/constant features from clustering:"
        )
        for rf in removed_features:
            logger.warning(
                f"  - {rf['feature']}: variance_ratio={rf['variance_ratio']:.6f}, "
                f"unique_ratio={rf['unique_ratio']:.4f} ({rf['unique_count']} unique values), "
                f"reason={rf['reason']}"
            )

    logger.info(
        f"Feature variance filter: kept {len(kept_features)} of {len(valid_features)} features"
    )

    return kept_features


def enrich_with_eda_insights(
    df: pd.DataFrame,
    eda_dir: str,
    key_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich segmentation features with exhaustive EDA insights.

    This function reads ALL exhaustive analysis outputs from EDA and adds
    relevant features for state-of-the-art segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with basic segmentation features
    eda_dir : str
        Path to EDA output directory
    key_cols : List[str]
        Key identifier columns

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Enriched DataFrame and dictionary of EDA insights for strategy
    """
    eda_insights = {
        'seasonality': {},
        'trend': {},
        'changepoints': {},
        'autocorrelation': {},
        'data_profile': {},
    }

    features_added = []

    # -------------------------------------------------------------------------
    # 1. Load Seasonality Analysis
    # -------------------------------------------------------------------------
    seasonality_path = os.path.join(eda_dir, 'seasonality_analysis.json')
    if os.path.exists(seasonality_path):
        try:
            with open(seasonality_path) as f:
                seasonality = json.load(f)
            eda_insights['seasonality'] = seasonality

            # Use STRENGTH-BASED metrics for seasonality
            has_seasonality_pct = seasonality.get('has_seasonality_pct', 0)
            avg_seasonal_strength = seasonality.get('avg_seasonal_strength', 0)

            # Flag for clustering strategy: require both pct AND strength
            df['_global_has_seasonality'] = (has_seasonality_pct > 0.3) and (avg_seasonal_strength > 0.15)

            # If dominant period detected, flag series likely to have it
            dominant_period = seasonality.get('dominant_periods', {})
            if dominant_period:
                df['_detected_seasonal_period'] = max(dominant_period.keys(), key=lambda k: dominant_period[k]) if dominant_period else None

            logger.info(f"Loaded seasonality insights: {has_seasonality_pct*100:.0f}% series show seasonality (avg_strength={avg_seasonal_strength:.2f})")
        except Exception as e:
            logger.warning(f"Could not load seasonality analysis: {e}")

    # -------------------------------------------------------------------------
    # 2. Load Trend Analysis
    # -------------------------------------------------------------------------
    trend_path = os.path.join(eda_dir, 'trend_analysis.json')
    if os.path.exists(trend_path):
        try:
            with open(trend_path) as f:
                trend = json.load(f)
            eda_insights['trend'] = trend

            # Add trend distribution insights
            avg_trend = trend.get('avg_trend_strength', 0)
            strongly_trending = trend.get('strongly_trending_pct', 0)

            # Flag for clustering strategy: require meaningful trend presence
            df['_high_trend_data'] = (strongly_trending > 0.2) or (avg_trend > 0.3)

            logger.info(f"Loaded trend insights: avg_strength={avg_trend:.2f}, strongly_trending={strongly_trending*100:.0f}%")
        except Exception as e:
            logger.warning(f"Could not load trend analysis: {e}")

    # -------------------------------------------------------------------------
    # 3. Load Changepoint Analysis
    # -------------------------------------------------------------------------
    changepoint_path = os.path.join(eda_dir, 'changepoint_analysis.json')
    if os.path.exists(changepoint_path):
        try:
            with open(changepoint_path) as f:
                changepoints = json.load(f)
            eda_insights['changepoints'] = changepoints

            # Use SIGNIFICANCE-BASED metrics for changepoints
            pct_with_changepoints = changepoints.get('pct_with_changepoints', 0)
            pct_significant = changepoints.get('pct_with_significant_changepoints', 0)
            avg_mean_shift = changepoints.get('avg_mean_shift_ratio', 0)
            avg_changepoints = changepoints.get('avg_changepoints_per_series', 0)

            # Flag for clustering strategy: require significant changepoints
            df['_regime_switching_likely'] = (pct_significant > 0.3) and (avg_mean_shift > 0.25)

            logger.info(f"Loaded changepoint insights: {pct_significant*100:.0f}% have SIGNIFICANT changepoints (avg_shift={avg_mean_shift:.2f})")
        except Exception as e:
            logger.warning(f"Could not load changepoint analysis: {e}")

    # -------------------------------------------------------------------------
    # 4. Load Autocorrelation Summary
    # -------------------------------------------------------------------------
    acf_path = os.path.join(eda_dir, 'autocorrelation_summary.csv')
    if os.path.exists(acf_path):
        try:
            acf_df = pd.read_csv(acf_path)

            # Extract significant lags
            if 'significant_acf_lags' in acf_df.columns:
                all_lags = acf_df['significant_acf_lags'].dropna().astype(str).str.split(',').explode()
                try:
                    all_lags = all_lags[all_lags != ''].astype(int)
                    significant_lags = all_lags.value_counts().head(5).index.tolist()
                    eda_insights['autocorrelation']['significant_lags'] = significant_lags
                except:
                    significant_lags = []

            # Merge ACF features if key column matches
            key_col = key_cols[0] if len(key_cols) == 1 else '_combined_key'
            if key_col in acf_df.columns and key_col in df.columns:
                acf_features = ['lag1_acf', 'lag1_pacf']
                available_acf = [c for c in acf_features if c in acf_df.columns]
                if available_acf:
                    df = df.merge(
                        acf_df[[key_col] + available_acf],
                        on=key_col,
                        how='left',
                        suffixes=('', '_eda')
                    )
                    features_added.extend(available_acf)

            logger.info(f"Loaded ACF insights: {len(eda_insights['autocorrelation'].get('significant_lags', []))} significant lags")
        except Exception as e:
            logger.warning(f"Could not load ACF summary: {e}")

    # -------------------------------------------------------------------------
    # 5. Load Data Profile
    # -------------------------------------------------------------------------
    profile_path = os.path.join(eda_dir, 'data_profile.json')
    if os.path.exists(profile_path):
        try:
            with open(profile_path) as f:
                profile = json.load(f)
            eda_insights['data_profile'] = profile

            # Extract key profile info for strategy
            primary_pattern = profile.get('primary_demand_pattern', 'mixed')
            granularity = profile.get('time_granularity', 'weekly')

            logger.info(f"Loaded data profile: pattern={primary_pattern}, granularity={granularity}")
        except Exception as e:
            logger.warning(f"Could not load data profile: {e}")

    if features_added:
        logger.info(f"Added {len(features_added)} features from EDA insights: {features_added}")

    return df, eda_insights


def classify_demand_pattern(
    adi: float,
    cv2: float,
    adi_threshold: float = None,
    cv2_threshold: float = None,
    time_format: str = 'year_week',
) -> str:
    """
    Classify demand pattern using Syntetos-Boylan framework.

    The Syntetos-Boylan classification divides demand into 4 quadrants:

                    |  ADI <= thresh  |  ADI > thresh
    ----------------+-----------------+-----------------
    CV² <= thresh   |     SMOOTH      |   INTERMITTENT
    CV² > thresh    |     ERRATIC     |      LUMPY

    Thresholds adapt to time_format:
    - year_week: ADI=1.32, CV²=0.49 (standard Syntetos-Boylan)
    - year_month: ADI=2.0, CV²=0.75 (relaxed for few observations)

    Parameters
    ----------
    adi : float
        Average Demand Interval (mean periods between non-zero demands)
    cv2 : float
        Squared Coefficient of Variation
    adi_threshold : float, optional
        ADI threshold (auto-resolved from time_format if None)
    cv2_threshold : float, optional
        CV² threshold (auto-resolved from time_format if None)
    time_format : str
        'year_week' or 'year_month'

    Returns
    -------
    str
        Demand pattern: 'smooth', 'erratic', 'intermittent', or 'lumpy'
    """
    if adi_threshold is None:
        adi_threshold = ADI_THRESHOLD_MONTHLY if time_format == 'year_month' else ADI_THRESHOLD
    if cv2_threshold is None:
        cv2_threshold = CV2_THRESHOLD_MONTHLY if time_format == 'year_month' else CV2_THRESHOLD

    if adi <= adi_threshold:
        return 'erratic' if cv2 > cv2_threshold else 'smooth'
    else:
        return 'lumpy' if cv2 > cv2_threshold else 'intermittent'


def get_recommended_clustering_features(
    df: pd.DataFrame,
    strategy: str = 'comprehensive',
    eda_insights: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Get recommended features for clustering based on strategy and EDA insights.

    This function is ADAPTIVE - it adjusts feature selection based on:
    - What EDA found (seasonality, trend, changepoints)
    - The data characteristics
    - The chosen strategy

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with segmentation features
    strategy : str
        'comprehensive': Use all available dimensions (recommended)
        'adaptive': Let EDA insights guide feature selection
        'volume_focus': Prioritize volume-based segmentation
        'pattern_focus': Prioritize demand pattern segmentation
        'minimal': Use minimal features for simple segmentation
    eda_insights : Dict[str, Any], optional
        EDA insights from enrich_with_eda_insights() for adaptive selection

    Returns
    -------
    List[str]
        List of feature names to use for clustering

    Example
    -------
    >>> features = get_recommended_clustering_features(df, 'adaptive', eda_insights)
    """
    # Define feature priorities by category
    volume_features = ['volume_mean', 'volume_total', 'demand_frequency', 'volume_tier_numeric']
    intermittency_features = ['adi_log', 'cv2_clean', 'zero_fraction_clean', 'demand_pattern_numeric', 'intermittency_score', 'pattern_family']
    variability_features = ['cv_clean', 'std_normalized', 'variability_tier_numeric']
    temporal_features = ['trend_strength_clean', 'seasonal_strength_clean', 'autocorr_lag1_clean']
    advanced_features = ['forecastability_score', 'stability_score', 'complexity_score', 'lifecycle_stage_numeric']
    # External feature aggregates for price/promo sensitivity segmentation
    external_features = ['price_level', 'price_tier_numeric', 'price_variability',
                         'promo_intensity', 'promo_tier_numeric',
                         'external_sensitivity_score', 'external_sensitivity_numeric']

    # Dynamically detect categorical external features from the DataFrame
    # These are created from config-driven columns in compute_demand_characteristics
    # Look for columns ending with _cat_tier_numeric, _encoded_mean_clean, or categorical_diversity_score
    categorical_external_features = []
    for col in df.columns:
        if col.endswith('_cat_tier_numeric') or col.endswith('_encoded_mean_clean'):
            categorical_external_features.append(col)
    if 'categorical_diversity_score' in df.columns:
        categorical_external_features.append('categorical_diversity_score')

    # Phase 2: Dynamically detect enriched segmentation features
    # These are added by segmentation_enhanced.py:enrich_segmentation_features()
    enriched_response_features = [c for c in df.columns if c.startswith('ext_response_') or c.startswith('resp_')]
    enriched_seasonality_features = [c for c in df.columns if c.startswith('seasonal_amplitude_') or c == 'seasonal_total_power' or c == 'seasonal_power_normalized']
    enriched_trend_features = [c for c in df.columns if c in ('trend_slope_normalized', 'trend_direction', 'trend_acceleration', 'recent_vs_longterm_ratio')]
    enriched_hierarchy_features = [c for c in df.columns if c.startswith('hier_')]
    enriched_importance_features = [c for c in df.columns if c.startswith('imp_')]

    has_enriched = any([enriched_response_features, enriched_seasonality_features,
                        enriched_trend_features, enriched_hierarchy_features])

    if strategy == 'adaptive' and eda_insights:
        # =====================================================================
        # ADAPTIVE STRATEGY - MULTI-DIMENSIONAL SEGMENTATION
        # =====================================================================
        # The goal of segmentation is to create HOMOGENEOUS groups where series
        # within each segment behave similarly. This requires INCLUDING all
        # relevant dimensions - not being conservative about them.
        #
        # Even if only 8% of series have seasonality, we WANT to separate those
        # from non-seasonal series. Even if average trend is low, some series
        # DO have trends and should be grouped together.
        #
        # Key dimensions for demand forecasting segmentation:
        # 1. VOLUME: How much demand (volume_mean, demand_frequency)
        # 2. INTERMITTENCY: How sparse the demand (adi, zero_fraction, cv)
        # 3. VARIABILITY: How volatile the demand (cv, std_normalized)
        # 4. TEMPORAL: Trend, seasonality, autocorrelation patterns
        # 5. FORECASTABILITY: How predictable (forecastability_score, complexity)
        # =====================================================================

        # Start with comprehensive multi-dimensional feature set
        priority_order = [
            # Volume dimension
            'volume_mean',
            'demand_frequency',
            # Intermittency dimension (Syntetos-Boylan) - continuous score + components
            'intermittency_score',
            'pattern_family',
            'adi_log',
            'cv_clean',
            'zero_fraction_clean',
            # Variability dimension
            'variability_tier_numeric',
            # Forecastability dimension - ALWAYS include
            'forecastability_score',
            'complexity_score',
        ]

        # ALWAYS include temporal features - they create important segment distinctions
        # Even low average values mean SOME series have these patterns
        priority_order.extend([
            'trend_strength_clean',
            'seasonal_strength_clean',
            'autocorr_lag1_clean',
        ])

        # Log what EDA found (for debugging) but DON'T skip features based on thresholds
        seasonality_info = eda_insights.get('seasonality', {})
        avg_seasonal_strength = seasonality_info.get('avg_seasonal_strength', 0)
        has_seasonality_pct = seasonality_info.get('has_seasonality_pct', 0)
        logger.info(f"Seasonality in data: avg_strength={avg_seasonal_strength:.2f}, pct={has_seasonality_pct:.1%} - INCLUDED for segmentation")

        trend_info = eda_insights.get('trend', {})
        strongly_trending_pct = trend_info.get('strongly_trending_pct', 0)
        avg_trend_strength = trend_info.get('avg_trend_strength', 0)
        logger.info(f"Trend in data: strongly_trending={strongly_trending_pct:.1%}, avg_strength={avg_trend_strength:.2f} - INCLUDED for segmentation")

        # Add lifecycle stage if changepoints detected (any level)
        changepoints = eda_insights.get('changepoints', {})
        pct_with_cp = changepoints.get('pct_with_changepoints', 0)
        if pct_with_cp > 0.1:  # If >10% have changepoints, include lifecycle
            priority_order.append('lifecycle_stage_numeric')
            logger.info(f"Changepoints detected ({pct_with_cp:.1%}) - including lifecycle_stage")

        # Add stability score for regime identification
        priority_order.append('stability_score')

        # EXTERNAL FEATURES - Critical for price-sensitive vs base-demand segmentation
        # These help identify keys that respond differently to price/promo changes
        priority_order.extend([
            'external_sensitivity_score',
            'price_tier_numeric',
            'promo_intensity',
        ])
        logger.info("Including external sensitivity features for price/promo-aware segmentation")

        # CATEGORICAL EXTERNAL FEATURES - Dynamically detected from config-driven columns
        # These capture the "type" of promotion/price change, not just the intensity
        if categorical_external_features:
            priority_order.extend(categorical_external_features)
            logger.info(f"Including {len(categorical_external_features)} categorical external feature encodings for clustering")
        else:
            logger.debug("No categorical external features detected in DataFrame")

        # PHASE 2 ENRICHED FEATURES - External response, seasonality shape, trend, hierarchy
        n_enriched = 0
        if enriched_response_features:
            # Use aggregate response metrics (not per-feature correlations to keep dimensionality down)
            for f in ['ext_response_mean_abs_corr', 'ext_response_max_abs_corr', 'ext_response_n_significant']:
                if f in df.columns:
                    priority_order.append(f)
                    n_enriched += 1
        if enriched_seasonality_features:
            # Use normalized power and top harmonics
            for f in ['seasonal_power_normalized', 'seasonal_amplitude_1', 'seasonal_phase_1']:
                if f in df.columns:
                    priority_order.append(f)
                    n_enriched += 1
        if enriched_trend_features:
            for f in enriched_trend_features:
                priority_order.append(f)
                n_enriched += 1
        if enriched_hierarchy_features:
            # Include up to 4 hierarchy features (share and rank for top 2 hierarchies)
            for f in enriched_hierarchy_features[:4]:
                priority_order.append(f)
                n_enriched += 1
        if enriched_importance_features:
            for f in ['imp_concentration', 'imp_external_pct', 'imp_lag_pct']:
                if f in df.columns:
                    priority_order.append(f)
                    n_enriched += 1
        if n_enriched > 0:
            logger.info(f"Including {n_enriched} Phase 2 enriched features for clustering")

        n_dims = 7 + (1 if has_enriched else 0)
        logger.info(f"Adaptive strategy: {len(priority_order)} features across {n_dims}+ dimensions")

    elif strategy == 'comprehensive':
        # Use all available features from multiple dimensions INCLUDING external features
        priority_order = (
            ['volume_mean', 'cv_clean', 'adi_log', 'zero_fraction_clean',
             'demand_frequency', 'intermittency_score', 'pattern_family',
             'forecastability_score', 'variability_tier_numeric'] +
            temporal_features + ['stability_score', 'complexity_score'] +
            # External features for price/promo sensitivity (numeric)
            ['external_sensitivity_score', 'price_tier_numeric', 'promo_intensity', 'price_variability'] +
            # Categorical external feature encodings - dynamically detected
            categorical_external_features +
            # Phase 2 enriched features
            ['ext_response_mean_abs_corr', 'ext_response_max_abs_corr',
             'seasonal_power_normalized', 'seasonal_amplitude_1',
             'trend_slope_normalized', 'trend_direction', 'trend_acceleration',
             'recent_vs_longterm_ratio', 'imp_concentration', 'imp_external_pct'] +
            enriched_hierarchy_features[:4]
        )
    elif strategy == 'volume_focus':
        priority_order = volume_features + ['cv_clean', 'demand_pattern_numeric']
    elif strategy == 'pattern_focus':
        priority_order = intermittency_features + ['volume_tier_numeric']
    else:  # minimal
        priority_order = ['volume_mean', 'cv_clean', 'zero_fraction_clean']

    # Filter to features that exist in the DataFrame
    available_features = [f for f in priority_order if f in df.columns]

    # Remove duplicates while preserving order
    seen = set()
    available_features = [f for f in available_features if not (f in seen or seen.add(f))]

    if len(available_features) < 3:
        # Fallback to any numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['segment_id', 'cluster', 'label'] + list(df.select_dtypes(include=['object', 'category']).columns)
        exclude += [c for c in df.columns if c.startswith('_')]  # Exclude internal columns
        available_features = [c for c in numeric_cols if c not in exclude][:6]

    logger.info(f"Recommended clustering features ({strategy}): {available_features}")
    return available_features


# =============================================================================
# 2. SCALING FUNCTIONS
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = 'robust',
    handle_missing: str = 'median',
) -> Tuple[np.ndarray, Any]:
    """
    Scale features for clustering with robust handling of outliers.

    NOTE: Low-variance features should be filtered out BEFORE calling this
    function using filter_low_variance_features(). This function assumes
    all features passed to it are valid for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Columns to scale (should be pre-filtered for low variance)
    scaler_type : str
        Type of scaler: 'robust' (recommended), 'standard', 'minmax'
    handle_missing : str
        How to handle missing values: 'median', 'mean', 'zero'

    Returns
    -------
    Tuple[np.ndarray, Any]
        Scaled features array and fitted scaler object

    Example
    -------
    >>> # First filter low-variance features
    >>> features = filter_low_variance_features(df, ['cv', 'mean', 'zero_fraction'])
    >>> # Then scale
    >>> X_scaled, scaler = scale_features(df, features)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scalers = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,  # Best for demand data with outliers
    }

    scaler_class = scalers.get(scaler_type, RobustScaler)
    scaler = scaler_class()

    X = df[feature_cols].copy()

    # Handle missing values
    if handle_missing == 'median':
        X = X.fillna(X.median())
    elif handle_missing == 'mean':
        X = X.fillna(X.mean())
    else:
        X = X.fillna(0)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled = scaler.fit_transform(X.values)

    return X_scaled, scaler


# =============================================================================
# 3. CLUSTERING ALGORITHMS
# =============================================================================

def cluster_kmeans(
    X_scaled: np.ndarray,
    k: int = None,
    k_range: List[int] = None,
    auto_select: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, Any, ClusteringMetrics]:
    """
    Cluster using K-Means with automatic k selection.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    k : int, optional
        Fixed number of clusters
    k_range : List[int], optional
        Range of k values to try (default [3, 4, 5, 6, 7])
    auto_select : bool
        If True, automatically select best k via silhouette
    random_state : int
        Random state for reproducibility

    Returns
    -------
    Tuple[np.ndarray, Any, ClusteringMetrics]
        Labels, fitted KMeans, and metrics
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    if k_range is None:
        k_range = [3, 4, 5, 6, 7]

    if k is not None and not auto_select:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
    else:
        best_score = -1
        best_model = None
        best_k = k_range[0]

        for k_val in k_range:
            if k_val >= len(X_scaled):
                continue

            kmeans = KMeans(n_clusters=k_val, random_state=random_state, n_init=10)
            labels_temp = kmeans.fit_predict(X_scaled)

            if len(set(labels_temp)) > 1:
                sil = silhouette_score(X_scaled, labels_temp)
                if sil > best_score:
                    best_score = sil
                    best_model = kmeans
                    best_k = k_val

        if best_model is None:
            kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
        else:
            kmeans = best_model
            labels = kmeans.labels_

    # Compute metrics
    n_clusters = len(set(labels))
    if n_clusters > 1:
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
    else:
        sil, ch, db = 0.0, 0.0, float('inf')

    metrics = ClusteringMetrics(
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        n_clusters=n_clusters,
    )

    return labels, kmeans, metrics


def cluster_gaussian_mixture(
    X_scaled: np.ndarray,
    n_components: int = None,
    n_range: List[int] = None,
    covariance_type: str = 'full',
    auto_select: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, Any, ClusteringMetrics]:
    """
    Cluster using Gaussian Mixture Model with BIC-based selection.

    GMM is excellent for demand segmentation because:
    - Handles overlapping clusters (fuzzy boundaries)
    - Provides probability assignments
    - Works well with non-spherical clusters

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    n_components : int, optional
        Fixed number of components
    n_range : List[int], optional
        Range to try (default [3, 4, 5, 6, 7])
    covariance_type : str
        'full', 'tied', 'diag', 'spherical'
    auto_select : bool
        If True, select best n via BIC
    random_state : int
        Random state

    Returns
    -------
    Tuple[np.ndarray, Any, ClusteringMetrics]
        Labels, fitted GMM, and metrics
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    if n_range is None:
        n_range = [3, 4, 5, 6, 7]

    if n_components is not None and not auto_select:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=5,
        )
        labels = gmm.fit_predict(X_scaled)
    else:
        best_bic = float('inf')
        best_model = None

        for n in n_range:
            if n >= len(X_scaled):
                continue

            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                random_state=random_state,
                n_init=5,
            )
            gmm.fit(X_scaled)
            bic = gmm.bic(X_scaled)

            if bic < best_bic:
                best_bic = bic
                best_model = gmm

        if best_model is None:
            gmm = GaussianMixture(n_components=3, random_state=random_state)
            labels = gmm.fit_predict(X_scaled)
        else:
            gmm = best_model
            labels = gmm.predict(X_scaled)

    # Compute metrics
    n_clusters = len(set(labels))
    if n_clusters > 1:
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
    else:
        sil, ch, db = 0.0, 0.0, float('inf')

    metrics = ClusteringMetrics(
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        n_clusters=n_clusters,
    )

    return labels, gmm, metrics


def cluster_hdbscan(
    X_scaled: np.ndarray,
    min_pct: float = DEFAULT_MIN_SEGMENT_PCT,
    min_samples: int = None,
    cluster_selection_method: str = 'eom',
) -> Tuple[np.ndarray, Any, ClusteringMetrics]:
    """
    Cluster using HDBSCAN with automatic cluster detection.

    HDBSCAN is excellent for:
    - Automatic cluster count determination
    - Handling varying density
    - Identifying outliers

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    min_pct : float
        Minimum cluster size as percentage (default 10%)
    min_samples : int, optional
        HDBSCAN min_samples (default: auto)
    cluster_selection_method : str
        'eom' or 'leaf'

    Returns
    -------
    Tuple[np.ndarray, Any, ClusteringMetrics]
        Labels, fitted clusterer, and metrics
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        logger.warning("HDBSCAN not available, falling back to KMeans")
        return cluster_kmeans(X_scaled, k_range=[4, 5, 6])

    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    n_samples = len(X_scaled)
    # Use min of cap (50) or percentage, with floor of 5
    pct_based = int(n_samples * min_pct)
    min_cluster_size = max(min(MIN_SEGMENT_SIZE_CAP, pct_based), 5)

    if min_samples is None:
        min_samples = max(5, int(n_samples * 0.01))

    # core_dist_n_jobs capped at 4 to avoid oversubscription with
    # LightGBM threading later in the pipeline. HDBSCAN's distance
    # computation scales sublinearly past 4 threads anyway, so the
    # cap costs almost nothing and prevents context-switch thrashing
    # on 16-core Databricks drivers.
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=4,
    )

    labels = clusterer.fit_predict(X_scaled)

    # Handle noise points (label -1)
    n_noise = (labels == -1).sum()

    if n_noise > 0 and n_noise < n_samples:
        valid_labels = sorted([l for l in set(labels) if l >= 0])
        if valid_labels:
            centroids = np.array([
                X_scaled[labels == l].mean(axis=0) for l in valid_labels
            ])

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(centroids)

            noise_mask = labels == -1
            noise_points = X_scaled[noise_mask]
            _, nearest_idx = nn.kneighbors(noise_points)

            labels[noise_mask] = np.array(valid_labels)[nearest_idx.flatten()]

    # Renumber to consecutive integers
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    label_map[-1] = 0 if not label_map else max(label_map.values())
    labels = np.array([label_map.get(l, 0) for l in labels])

    # Compute metrics
    n_clusters = len(set(labels))
    if n_clusters > 1:
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
    else:
        sil, ch, db = 0.0, 0.0, float('inf')

    metrics = ClusteringMetrics(
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        n_clusters=n_clusters,
        n_noise=n_noise,
    )

    return labels, clusterer, metrics


def cluster_hierarchical_optimal(
    X_scaled: np.ndarray,
    n_range: List[int] = None,
    linkage: str = 'ward',
    min_pct: float = DEFAULT_MIN_SEGMENT_PCT,
) -> Tuple[np.ndarray, Any, ClusteringMetrics]:
    """
    Hierarchical clustering with optimal cluster selection.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    n_range : List[int], optional
        Range of clusters to try
    linkage : str
        Linkage type: 'ward', 'complete', 'average', 'single'
    min_pct : float
        Minimum segment size

    Returns
    -------
    Tuple[np.ndarray, Any, ClusteringMetrics]
        Labels, fitted model, and metrics
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    if n_range is None:
        n_range = [3, 4, 5, 6, 7]

    best_score = -1
    best_model = None
    best_labels = None

    for n in n_range:
        if n >= len(X_scaled):
            continue

        model = AgglomerativeClustering(n_clusters=n, linkage=linkage)
        labels_temp = model.fit_predict(X_scaled)

        if len(set(labels_temp)) > 1:
            sil = silhouette_score(X_scaled, labels_temp)
            if sil > best_score:
                best_score = sil
                best_model = model
                best_labels = labels_temp

    if best_model is None:
        model = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        labels = model.fit_predict(X_scaled)
    else:
        model = best_model
        labels = best_labels

    # Compute metrics
    n_clusters = len(set(labels))
    if n_clusters > 1:
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
    else:
        sil, ch, db = 0.0, 0.0, float('inf')

    metrics = ClusteringMetrics(
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        n_clusters=n_clusters,
    )

    return labels, model, metrics


# =============================================================================
# 4. SEGMENT SIZE ENFORCEMENT
# =============================================================================

def enforce_minimum_segment_size(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    min_pct: float = DEFAULT_MIN_SEGMENT_PCT,
    min_final_segments: int = 2,
) -> np.ndarray:
    """
    Enforce minimum segment size by merging small segments.

    This function intelligently merges segments that are too small while
    preserving meaningful segmentation. It guarantees at least min_final_segments
    segments (default 2) to avoid collapsing everything into a single segment.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix (for computing centroids)
    labels : np.ndarray
        Current segment labels
    min_pct : float
        Minimum segment size as percentage (default 10%)
    min_final_segments : int
        Minimum number of segments to preserve (default 2).
        Setting to 1 allows merging all into single segment.

    Returns
    -------
    np.ndarray
        Labels with small segments merged

    Example
    -------
    >>> labels = enforce_minimum_segment_size(X_scaled, labels, min_pct=0.10)
    """
    labels = labels.copy()
    n_samples = len(labels)

    # Calculate min_size as the SMALLER of: 50 or 10% of total
    # This allows more granular segments for larger datasets
    pct_based_min = int(n_samples * min_pct)
    min_size = min(MIN_SEGMENT_SIZE_CAP, pct_based_min)

    # Track initial state for logging
    initial_segments = len(set(labels))
    logger.info(f"enforce_minimum_segment_size: starting with {initial_segments} segments, min_size={min_size} (min of {MIN_SEGMENT_SIZE_CAP} or {min_pct*100:.0f}%={pct_based_min})")

    iteration = 0
    max_iterations = 50  # Safety limit

    while iteration < max_iterations:
        iteration += 1
        seg_counts = pd.Series(labels).value_counts()
        n_current_segments = len(seg_counts)

        # Find segments below minimum size
        small_segs = seg_counts[seg_counts < min_size].index.tolist()

        if not small_segs:
            # All segments meet minimum size requirement
            logger.info(f"enforce_minimum_segment_size: converged after {iteration} iterations, {n_current_segments} segments")
            break

        # CRITICAL: Preserve minimum number of segments
        # Don't merge if we'd drop below min_final_segments
        if n_current_segments <= min_final_segments:
            logger.warning(
                f"enforce_minimum_segment_size: stopping at {n_current_segments} segments "
                f"(min_final_segments={min_final_segments}). Some segments below {min_pct*100:.0f}%."
            )
            break

        large_segs = seg_counts[seg_counts >= min_size].index.tolist()

        if not large_segs:
            # All segments below threshold
            # Instead of merging ALL into one, try relaxing the constraint progressively
            # Find the two largest segments and keep them separate
            if n_current_segments >= min_final_segments:
                top_segments = seg_counts.nlargest(min_final_segments).index.tolist()
                # Merge everything else into the nearest top segment
                other_segs = [s for s in seg_counts.index if s not in top_segments]

                if other_segs:
                    # Compute centroids for top segments
                    top_centroids = {
                        seg: X_scaled[labels == seg].mean(axis=0)
                        for seg in top_segments
                    }

                    for other_seg in other_segs:
                        other_centroid = X_scaled[labels == other_seg].mean(axis=0)
                        # Find nearest top segment
                        best_target = min(
                            top_segments,
                            key=lambda s: np.linalg.norm(other_centroid - top_centroids[s])
                        )
                        labels[labels == other_seg] = best_target

                    logger.info(
                        f"enforce_minimum_segment_size: merged {len(other_segs)} small segments "
                        f"into {len(top_segments)} largest segments"
                    )
                break
            else:
                # Fewer than min_final_segments, nothing to preserve
                labels[:] = 0
                logger.warning("enforce_minimum_segment_size: all segments too small, merged into one")
                break

        # Merge smallest segment into nearest large segment
        smallest = min(small_segs, key=lambda s: seg_counts[s])
        small_centroid = X_scaled[labels == smallest].mean(axis=0)

        best_target = None
        best_dist = float('inf')

        for large_seg in large_segs:
            large_centroid = X_scaled[labels == large_seg].mean(axis=0)
            dist = np.linalg.norm(small_centroid - large_centroid)
            if dist < best_dist:
                best_dist = dist
                best_target = large_seg

        if best_target is not None:
            labels[labels == smallest] = best_target
            logger.debug(f"Merged segment {smallest} ({seg_counts[smallest]} samples) into {best_target}")

    # Renumber to consecutive integers
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    final_segments = len(set(labels))
    logger.info(f"enforce_minimum_segment_size: {initial_segments} → {final_segments} segments")

    return labels


def validate_segment_sizes(
    labels: np.ndarray,
    min_pct: float = DEFAULT_MIN_SEGMENT_PCT,
) -> Dict[str, Any]:
    """
    Validate that all segments meet minimum size requirements.

    Parameters
    ----------
    labels : np.ndarray
        Segment labels
    min_pct : float
        Minimum required percentage

    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    n_samples = len(labels)
    seg_counts = pd.Series(labels).value_counts()
    seg_pcts = seg_counts / n_samples

    small_segments = seg_pcts[seg_pcts < min_pct].index.tolist()

    return {
        'valid': len(small_segments) == 0,
        'n_segments': len(seg_counts),
        'min_pct_required': min_pct,
        'min_count_required': int(n_samples * min_pct),
        'small_segments': small_segments,
        'segment_sizes': seg_counts.to_dict(),
        'segment_percentages': (seg_pcts * 100).round(2).to_dict(),
    }


# =============================================================================
# 5. SEGMENT PROFILING
# =============================================================================

def compute_segment_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    key_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute comprehensive segment profiles with relative indices.

    Relative Index = segment_mean / overall_mean
    - > 1.0: segment is above average
    - < 1.0: segment is below average
    - = 1.0: segment matches average

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with segment features
    labels : np.ndarray
        Segment labels
    feature_cols : List[str]
        Feature columns to profile
    key_cols : List[str], optional
        Key columns (for context)

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Comprehensive profile for each segment
    """
    df = df.copy()
    df['segment_id'] = labels

    # Filter feature cols to those that exist and are numeric
    numeric_features = [c for c in feature_cols if c in df.columns]
    if not numeric_features:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [c for c in numeric_features if c != 'segment_id'][:10]

    overall_means = df[numeric_features].mean()
    overall_stds = df[numeric_features].std()

    profiles = {}
    total_count = len(df)

    for seg_id in sorted(df['segment_id'].unique()):
        seg_data = df[df['segment_id'] == seg_id]
        seg_means = seg_data[numeric_features].mean()

        # Relative indices (clamped to [-10, 10] to prevent explosion when overall mean ≈ 0)
        relative_indices = {}
        for feat in numeric_features:
            if abs(overall_means[feat]) > 1e-6:
                raw = seg_means[feat] / overall_means[feat]
                relative_indices[feat] = round(max(-10.0, min(10.0, raw)), 3)
            else:
                relative_indices[feat] = 0.0 if abs(seg_means[feat]) < 1e-6 else 1.0

        # Determine dominant demand pattern
        if 'demand_pattern' in seg_data.columns:
            pattern_dist = seg_data['demand_pattern'].value_counts(normalize=True)
            dominant_pattern = pattern_dist.index[0] if len(pattern_dist) > 0 else 'unknown'
            pattern_distribution = pattern_dist.to_dict()
        else:
            dominant_pattern = 'unknown'
            pattern_distribution = {}

        # Determine volume tier
        if 'volume_tier' in seg_data.columns:
            volume_dist = seg_data['volume_tier'].value_counts(normalize=True)
            dominant_volume = str(volume_dist.index[0]) if len(volume_dist) > 0 else 'medium'
        else:
            dominant_volume = 'medium'

        profiles[str(seg_id)] = {
            'segment_id': int(seg_id),
            'size': len(seg_data),
            'pct_of_total': round(100 * len(seg_data) / total_count, 2),
            'relative_indices': relative_indices,
            'means': seg_means.to_dict(),
            'dominant_pattern': dominant_pattern,
            'pattern_distribution': {str(k): round(v * 100, 1) for k, v in pattern_distribution.items()},
            'dominant_volume_tier': dominant_volume,
        }

    return profiles


def label_segments_automatically(
    profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generate human-readable labels for segments based on their profiles.

    Labels are constructed from:
    - Volume tier (High/Med/Low)
    - Demand pattern (Smooth/Erratic/Intermittent/Lumpy)
    - Variability (Stable/Variable)

    Parameters
    ----------
    profiles : Dict[str, Dict[str, Any]]
        Segment profiles from compute_segment_profiles

    Returns
    -------
    Dict[str, str]
        Mapping of segment_id to descriptive label

    Example
    -------
    >>> labels = label_segments_automatically(profiles)
    >>> print(labels)
    # {'0': 'High-Vol-Smooth', '1': 'Low-Vol-Lumpy', ...}
    """
    labels = {}

    for seg_id, profile in profiles.items():
        parts = []

        # Volume tier
        volume = profile.get('dominant_volume_tier', 'medium')
        if volume == 'high':
            parts.append('High-Vol')
        elif volume == 'low':
            parts.append('Low-Vol')
        else:
            parts.append('Med-Vol')

        # Demand pattern
        pattern = profile.get('dominant_pattern', 'unknown')
        if pattern == 'smooth':
            parts.append('Smooth')
        elif pattern == 'erratic':
            parts.append('Erratic')
        elif pattern == 'intermittent':
            parts.append('Intermittent')
        elif pattern == 'lumpy':
            parts.append('Lumpy')

        # Variability (from relative indices)
        rel_idx = profile.get('relative_indices', {})
        cv_idx = rel_idx.get('cv_clean', rel_idx.get('cv', 1.0))
        if cv_idx > 1.3:
            parts.append('Variable')
        elif cv_idx < 0.7:
            parts.append('Stable')

        labels[seg_id] = '-'.join(parts) if parts else f'Segment-{seg_id}'

    return labels


# =============================================================================
# 6. MODEL RECOMMENDATIONS
# =============================================================================

def assign_model_recommendations(
    profiles: Dict[str, Dict[str, Any]],
    allowed_model_families: List[str],
    enable_deep_models: bool = False,
) -> Dict[str, Any]:
    """
    Assign STATE-OF-THE-ART model recommendations for each segment based on profiles.

    This function provides comprehensive, actionable model recommendations that
    downstream Training crews can use directly. Recommendations are based on:
    - Demand pattern (Syntetos-Boylan classification)
    - Volume tier (affects model complexity and data availability)
    - Variability (affects regularization needs and loss function selection)
    - Forecastability score (indicates expected model performance)

    Parameters
    ----------
    profiles : Dict[str, Dict[str, Any]]
        Segment profiles from compute_segment_profiles()
    allowed_model_families : List[str]
        List of allowed model families from config
    enable_deep_models : bool
        Whether deep learning models are enabled

    Returns
    -------
    Dict[str, Any]
        Comprehensive model recommendations per segment including:
        - Primary and fallback models
        - Hyperparameter search spaces
        - Loss function recommendations
        - Validation strategy
        - Expected performance ranges
    """
    # ==========================================================================
    # STATE-OF-THE-ART MODEL MAPPINGS BY DEMAND PATTERN
    # ==========================================================================
    # CRITICAL: FEATURE-BASED MODELS ONLY - Univariate models are BANNED
    #
    # Research-backed rationale:
    #   1. Tree-based models with features consistently outperform univariate methods
    #   2. Feature engineering captures business context (price, promo, events)
    #   3. GBM models handle mixed feature types naturally
    #   4. Two-stage models (occurrence + severity) best for intermittent demand
    #
    # ALLOWED: lightgbm, xgboost, catboost, random_forest, zero_inflated, hurdle_model, tweedie
    # BANNED: croston, sba, tsb, imapa, ets, theta, arima, sarima, prophet (univariate)

    pattern_models = {
        'smooth': {
            'primary': ['lightgbm', 'xgboost'],
            'secondary': ['catboost', 'random_forest'],
            'loss_function': 'mse',
            'tweedie_power': None,
        },
        'erratic': {
            'primary': ['xgboost', 'lightgbm'],
            'secondary': ['catboost', 'quantile_regression'],
            'loss_function': 'huber',  # More robust to outliers
            'tweedie_power': None,
        },
        'intermittent': {
            'primary': ['lightgbm', 'xgboost'],  # With tweedie loss
            'secondary': ['zero_inflated', 'hurdle_model'],
            'loss_function': 'tweedie',
            'tweedie_power': 1.5,  # Between Poisson (1) and Gamma (2)
        },
        'lumpy': {
            'primary': ['lightgbm', 'xgboost'],  # With tweedie loss
            'secondary': ['hurdle_model', 'zero_inflated'],
            'loss_function': 'tweedie',
            'tweedie_power': 1.7,  # Closer to Gamma for high variance
        },
    }

    # Expected WAPE ranges based on academic benchmarks
    wape_ranges = {
        'smooth': {'min': 15, 'typical': 25, 'max': 35},
        'erratic': {'min': 25, 'typical': 40, 'max': 55},
        'intermittent': {'min': 35, 'typical': 50, 'max': 70},
        'lumpy': {'min': 45, 'typical': 65, 'max': 90},
    }

    # Regularization strength by pattern (higher = more regularization needed)
    regularization_strength = {
        'smooth': 'low',
        'erratic': 'high',
        'intermittent': 'medium',
        'lumpy': 'high',
    }

    recommendations = {
        'metadata': {
            'allowed_families': allowed_model_families,
            'enable_deep_models': enable_deep_models,
            'generation_method': 'state_of_the_art_segmentation_utility',
        },
        'segment_strategies': {},
        'global_training_guidance': {
            'cross_validation_folds': 5,
            'time_series_gap': 1,  # Gap between train and validation to prevent leakage
            'early_stopping_rounds': 50,
            'feature_importance_method': 'gain',  # or 'shap' for interpretability
        },
    }

    for seg_id, profile in profiles.items():
        pattern = profile.get('dominant_pattern', 'smooth')
        volume = profile.get('dominant_volume_tier', 'medium')
        size = profile.get('size', 100)
        rel_indices = profile.get('relative_indices', {})

        # Get pattern-specific model config
        pattern_config = pattern_models.get(pattern, pattern_models['smooth'])

        # Build ordered list of candidate models
        candidates = pattern_config['primary'] + pattern_config['secondary']

        # Filter to allowed models
        recommended = [m for m in candidates if m.lower() in [a.lower() for a in allowed_model_families]]

        # Add deep models if enabled and appropriate
        if enable_deep_models and pattern in ['smooth', 'erratic'] and size >= 100:
            deep_candidates = ['tft', 'nbeats', 'deepar', 'lstm']
            deep_allowed = [m for m in deep_candidates if m.lower() in [a.lower() for a in allowed_model_families]]
            recommended.extend(deep_allowed)

        # Fallback to lightgbm if nothing matched
        if not recommended:
            recommended = ['lightgbm']

        # =======================================================================
        # HYPERPARAMETER SEARCH SPACE - Segment-specific optimization
        # =======================================================================
        hyperparam_search_space = _get_hyperparameter_search_space(
            pattern=pattern,
            volume=volume,
            size=size,
            rel_indices=rel_indices,
        )

        # =======================================================================
        # LOSS FUNCTION CONFIGURATION
        # =======================================================================
        loss_config = {
            'primary_loss': pattern_config['loss_function'],
            'tweedie_power': pattern_config['tweedie_power'],
            'alternative_losses': ['mse', 'mae'] if pattern in ['smooth', 'erratic'] else ['tweedie', 'poisson'],
            'sample_weights': 'recency' if pattern in ['erratic', 'lumpy'] else None,
        }

        # =======================================================================
        # VALIDATION STRATEGY
        # =======================================================================
        if pattern in ['smooth', 'erratic']:
            validation_config = {
                'strategy': 'expanding_window',
                'n_splits': 5,
                'min_train_size': max(52, size // 3),  # At least 1 year or 1/3 of data
                'gap': 1,  # 1-period gap to prevent leakage
            }
        else:  # intermittent, lumpy
            validation_config = {
                'strategy': 'holdout',  # More stable for sparse data
                'test_size': 13,  # ~3 months
                'gap': 1,
            }

        # =======================================================================
        # FEATURE SELECTION GUIDANCE FOR TRAINING
        # =======================================================================
        feature_selection = {
            'max_features': 50 if volume == 'high' else (30 if volume == 'medium' else 20),
            'min_importance_threshold': 0.001,
            'correlation_threshold': 0.95,
            'remove_constant': True,
            'priority_features': _get_priority_features(pattern),
        }

        # =======================================================================
        # ENSEMBLE STRATEGY
        # =======================================================================
        ensemble_config = {
            'use_ensemble': len(recommended) >= 2 and size >= 50,
            'ensemble_method': 'weighted_average',
            'weight_by': 'validation_score',
            'max_models_in_ensemble': 3,
        }

        recommendations['segment_strategies'][seg_id] = {
            # Basic info
            'segment_id': seg_id,
            'segment_label': profile.get('label', f'Segment-{seg_id}'),
            'segment_size': size,
            'pct_of_total': profile.get('pct_of_total', 0),

            # Pattern characteristics
            'dominant_pattern': pattern,
            'dominant_volume': volume,
            'pattern_distribution': profile.get('pattern_distribution', {}),

            # Model recommendations (ordered by priority)
            'recommended_models': recommended[:5],
            'primary_model': recommended[0],
            'fallback_model': recommended[1] if len(recommended) > 1 else recommended[0],

            # Hyperparameters
            'hyperparameter_search_space': hyperparam_search_space,

            # Loss and validation
            'loss_config': loss_config,
            'validation_config': validation_config,
            'regularization_strength': regularization_strength.get(pattern, 'medium'),

            # Feature selection
            'feature_selection': feature_selection,

            # Ensemble
            'ensemble_config': ensemble_config,

            # Expected performance
            'expected_wape_range': wape_ranges.get(pattern, wape_ranges['smooth']),

            # Human-readable guidance
            'training_notes': _get_segment_training_notes(pattern, volume, size),
        }

    return recommendations


def _get_hyperparameter_search_space(
    pattern: str,
    volume: str,
    size: int,
    rel_indices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Generate segment-specific hyperparameter search spaces.

    Returns search spaces optimized for the segment's characteristics.
    """
    # Base configuration for tree-based models
    base_config = {
        'lightgbm': {
            'num_leaves': [15, 31, 63] if volume == 'high' else [7, 15, 31],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 500, 1000] if volume == 'high' else [100, 200, 500],
            'min_child_samples': [5, 10, 20],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0, 0.1, 1.0],
        },
        'xgboost': {
            'max_depth': [4, 6, 8] if volume == 'high' else [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 500, 1000] if volume == 'high' else [100, 200, 500],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1, 5, 10],
        },
        'catboost': {
            'depth': [4, 6, 8] if volume == 'high' else [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [200, 500, 1000] if volume == 'high' else [100, 200, 500],
            'l2_leaf_reg': [1, 3, 5, 10],
            'bagging_temperature': [0, 0.5, 1.0],
        },
    }

    # Adjust for pattern-specific needs
    search_space = base_config.copy()

    if pattern in ['erratic', 'lumpy']:
        # Need more regularization for high-variance patterns
        for model in search_space:
            if 'reg_lambda' in search_space[model]:
                search_space[model]['reg_lambda'] = [1, 5, 10, 20]
            if 'l2_leaf_reg' in search_space[model]:
                search_space[model]['l2_leaf_reg'] = [3, 5, 10, 20]

    if pattern in ['intermittent', 'lumpy']:
        # Add tweedie-specific parameters
        search_space['tweedie_variance_power'] = [1.5, 1.6, 1.7, 1.8, 1.9]

    # Add data size consideration
    if size < 100:
        # Small segment - reduce complexity
        for model in search_space:
            if isinstance(search_space[model], dict):
                if 'n_estimators' in search_space[model]:
                    search_space[model]['n_estimators'] = [50, 100, 200]
                if 'iterations' in search_space[model]:
                    search_space[model]['iterations'] = [50, 100, 200]

    return search_space


def _get_priority_features(pattern: str) -> List[str]:
    """Get priority feature patterns for each demand pattern."""
    base_features = [
        'lag_1', 'lag_2', 'lag_4',
        'rolling_mean_4', 'rolling_mean_13',
        'month', 'week_of_year', 'day_of_week',
    ]

    if pattern == 'smooth':
        return base_features + [
            'rolling_std_4', 'rolling_std_13',
            'trend_', 'seasonal_',
            'year', 'quarter',
        ]
    elif pattern == 'erratic':
        return base_features + [
            'rolling_std_', 'rolling_max_', 'rolling_min_',
            'momentum_', 'volatility_',
            'ewm_mean_', 'ewm_std_',
        ]
    elif pattern == 'intermittent':
        return base_features + [
            'demand_occurred', 'periods_since_nonzero',
            'rolling_nonzero_count_', 'avg_nonzero_demand',
            'demand_frequency_',
        ]
    else:  # lumpy
        return base_features + [
            'demand_occurred', 'periods_since_nonzero',
            'rolling_nonzero_count_', 'avg_nonzero_demand',
            'rolling_std_', 'spike_indicator',
        ]


def _get_segment_training_notes(pattern: str, volume: str, size: int) -> List[str]:
    """Generate actionable training notes for a segment."""
    notes = []

    # Pattern-specific guidance
    if pattern == 'smooth':
        notes.append("Use standard MSE/MAE loss - demand is well-behaved")
        notes.append("Include trend and seasonal features for best results")
        notes.append("Time series cross-validation recommended")
    elif pattern == 'erratic':
        notes.append("Use Huber loss or MAE to handle outliers gracefully")
        notes.append("Include volatility and momentum features")
        notes.append("Consider quantile regression for uncertainty estimation")
        notes.append("Higher regularization needed to prevent overfitting to noise")
    elif pattern == 'intermittent':
        notes.append("Use Tweedie loss (power 1.5) for zero-inflated data")
        notes.append("Include demand occurrence probability features")
        notes.append("Two-stage approach: P(demand > 0) × E[demand | demand > 0]")
        notes.append("Holdout validation more stable than time series CV")
    elif pattern == 'lumpy':
        notes.append("HARDEST pattern - set realistic expectations")
        notes.append("Use Tweedie loss (power 1.7) for extreme zero-inflation")
        notes.append("Two-stage model essential: occurrence + severity")
        notes.append("Focus on demand occurrence accuracy as primary metric")
        notes.append("High regularization critical to prevent overfitting")

    # Volume-specific guidance
    if volume == 'high':
        notes.append(f"High volume ({size} keys) - can support complex models")
        notes.append("Consider ensemble of top 2-3 models")
    elif volume == 'low':
        notes.append(f"Low volume ({size} keys) - keep model simple")
        notes.append("Limit max features to 20, use strong regularization")
    else:
        notes.append(f"Medium volume ({size} keys) - balanced approach")

    # Size-specific guidance
    if size < 50:
        notes.append("SMALL SEGMENT - consider merging with similar segment")
        notes.append("Use simple model with few features")

    return notes


def create_feature_recommendations(
    profiles: Dict[str, Dict[str, Any]],
    time_format: str = 'year_week',
    eda_insights: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create STATE-OF-THE-ART feature engineering recommendations per segment.

    This function provides comprehensive, actionable feature engineering guidance
    that downstream Feature Engineering crews can use directly. Recommendations are
    tailored to each segment's demand pattern and characteristics.

    Parameters
    ----------
    profiles : Dict[str, Dict[str, Any]]
        Segment profiles from compute_segment_profiles()
    time_format : str
        Time format ('year_week' or 'year_month')
    eda_insights : Dict[str, Any], optional
        EDA insights including seasonality, trend, and changepoint analysis

    Returns
    -------
    Dict[str, Any]
        Comprehensive feature recommendations per segment including:
        - Lag feature specifications
        - Rolling window configurations
        - Calendar feature selections
        - Intermittency feature flags
        - Feature interaction guidance
        - Transformation recommendations
    """
    eda_insights = eda_insights or {}

    # ==========================================================================
    # TIME-FORMAT-SPECIFIC CONFIGURATIONS
    # ==========================================================================
    if time_format == 'year_week':
        # Weekly data configuration
        base_config = {
            'seasonal_period': 52,
            'target_lags': [1, 2, 4, 8, 13, 26, 52],
            'rolling_windows': [4, 8, 13, 26, 52],
            'ewm_spans': [4, 13, 26],
            'diff_lags': [1, 4, 52],
        }
    else:
        # Monthly data configuration
        base_config = {
            'seasonal_period': 12,
            'target_lags': [1, 2, 3, 6, 12],
            'rolling_windows': [3, 6, 12],
            'ewm_spans': [3, 6, 12],
            'diff_lags': [1, 3, 12],
        }

    # ==========================================================================
    # GLOBAL RECOMMENDATIONS
    # ==========================================================================
    recommendations = {
        'global_recommendations': {
            'time_format': time_format,
            'seasonal_period': base_config['seasonal_period'],

            # Lag features (shifted to prevent leakage - CRITICAL)
            'target_lags': base_config['target_lags'],
            'lag_features_must_shift': True,  # All lag features MUST be shift(1) at minimum

            # Rolling statistics windows
            'rolling_windows': base_config['rolling_windows'],
            'rolling_stats': ['mean', 'std', 'min', 'max', 'sum', 'skew'],
            'rolling_min_periods': 1,  # Handle series starts gracefully

            # Exponential weighted moving averages
            'ewm_spans': base_config['ewm_spans'],
            'ewm_stats': ['mean', 'std'],

            # Differencing for stationarity
            'diff_lags': base_config['diff_lags'],

            # Calendar features
            'calendar_features': {
                'week_of_year': time_format == 'year_week',
                'month': True,
                'quarter': True,
                'year': True,
                'day_of_week': time_format == 'year_week',
                'is_month_start': True,
                'is_month_end': True,
                'is_quarter_start': True,
                'is_quarter_end': True,
                'is_year_start': True,
                'is_year_end': True,
            },

            # Cyclical encoding for calendar features
            'use_cyclical_encoding': True,
            'cyclical_features': ['week_of_year', 'month', 'day_of_week'],

            # Feature interaction guidance
            'create_interactions': True,
            'interaction_features': [
                ('lag_1', 'rolling_mean_4'),
                ('month', 'rolling_mean_13'),
            ],

            # Leakage prevention - CRITICAL
            'leakage_prevention': {
                'all_target_features_must_lag': True,
                'minimum_lag': 1,
                'exclude_future_information': True,
            },
        },
        'segment_recommendations': {},
    }

    # Add EDA-informed global recommendations
    seasonality_info = eda_insights.get('seasonality', {})
    trend_info = eda_insights.get('trend', {})
    changepoint_info = eda_insights.get('changepoints', {})
    acf_info = eda_insights.get('autocorrelation', {})

    # Adjust global config based on EDA insights
    if seasonality_info:
        avg_seasonal_strength = seasonality_info.get('avg_seasonal_strength', 0)
        has_seasonality_pct = seasonality_info.get('has_seasonality_pct', 0)
        recommendations['global_recommendations']['seasonality_detected'] = (
            avg_seasonal_strength > 0.15 and has_seasonality_pct > 0.3
        )
        dominant_periods = seasonality_info.get('dominant_periods', {})
        if dominant_periods:
            recommendations['global_recommendations']['dominant_seasonal_periods'] = list(dominant_periods.keys())

    if acf_info:
        significant_lags = acf_info.get('significant_lags', [])
        if significant_lags:
            # Add ACF-informed lags to the recommendation
            current_lags = set(recommendations['global_recommendations']['target_lags'])
            acf_lags = set([int(l) for l in significant_lags if isinstance(l, (int, float))])
            recommendations['global_recommendations']['acf_informed_lags'] = list(acf_lags)

    # ==========================================================================
    # PER-SEGMENT FEATURE RECOMMENDATIONS
    # ==========================================================================
    for seg_id, profile in profiles.items():
        pattern = profile.get('dominant_pattern', 'smooth')
        volume = profile.get('dominant_volume_tier', 'medium')
        rel_indices = profile.get('relative_indices', {})

        seg_rec = _get_segment_feature_config(
            pattern=pattern,
            volume=volume,
            rel_indices=rel_indices,
            base_config=base_config,
            seasonality_info=seasonality_info,
            time_format=time_format,
        )

        seg_rec['dominant_pattern'] = pattern
        seg_rec['segment_id'] = seg_id
        seg_rec['segment_label'] = profile.get('label', f'Segment-{seg_id}')

        recommendations['segment_recommendations'][seg_id] = seg_rec

    return recommendations


def _get_segment_feature_config(
    pattern: str,
    volume: str,
    rel_indices: Dict[str, float],
    base_config: Dict[str, Any],
    seasonality_info: Dict[str, Any],
    time_format: str,
) -> Dict[str, Any]:
    """
    Generate segment-specific feature configuration based on pattern.

    Returns a comprehensive feature engineering specification that the
    Feature Engineering crew can use directly.
    """
    seasonal_period = base_config['seasonal_period']

    # Pattern-specific configurations
    if pattern == 'smooth':
        config = {
            # Lag features - use full range for smooth patterns
            'target_lags': base_config['target_lags'],
            'feature_lags': [1, 2, 4, seasonal_period],

            # Rolling features - comprehensive for smooth data
            'rolling_windows': base_config['rolling_windows'],
            'rolling_stats': ['mean', 'std', 'min', 'max'],
            'rolling_agg_functions': {
                'mean': True,
                'std': True,
                'min': True,
                'max': True,
                'skew': volume == 'high',  # Only for high volume
            },

            # EWM features - smooth data benefits from EWM
            'ewm_spans': base_config['ewm_spans'],
            'ewm_stats': ['mean', 'std'],

            # Trend/Seasonal features
            'trend_features': True,
            'seasonal_features': True,
            'fourier_features': seasonality_info.get('has_seasonality_pct', 0) > 0.3,
            'fourier_order': 3 if time_format == 'year_week' else 2,

            # Calendar features
            'calendar_features': True,
            'calendar_feature_list': ['month', 'quarter', 'year', 'week_of_year'],

            # Intermittency features - NOT needed for smooth
            'intermittency_features': False,

            # Transformations
            'log_transform': False,
            'box_cox_transform': False,
            'difference_transform': False,

            # Feature limits
            'max_lag_features': 20,
            'max_rolling_features': 20,
            'max_total_features': 60 if volume == 'high' else 40,

            # Notes for Feature Engineering crew
            'engineering_notes': [
                "Smooth pattern - standard features work well",
                "Include full lag range for capturing autocorrelation",
                "Trend and seasonal decomposition recommended",
                "Calendar features important for capturing patterns",
            ],
        }

    elif pattern == 'erratic':
        config = {
            # Lag features - similar to smooth but with volatility focus
            'target_lags': base_config['target_lags'],
            'feature_lags': [1, 2, 4, 8, seasonal_period],

            # Rolling features - include volatility measures
            'rolling_windows': base_config['rolling_windows'],
            'rolling_stats': ['mean', 'std', 'min', 'max', 'range'],
            'rolling_agg_functions': {
                'mean': True,
                'std': True,  # CRITICAL for erratic
                'min': True,
                'max': True,
                'range': True,  # max - min captures volatility
                'coefficient_of_variation': volume == 'high',
            },

            # EWM features - important for erratic patterns
            'ewm_spans': base_config['ewm_spans'],
            'ewm_stats': ['mean', 'std'],

            # Volatility-specific features
            'volatility_features': True,
            'momentum_features': True,

            # Trend/Seasonal features
            'trend_features': True,
            'seasonal_features': True,
            'fourier_features': False,  # Less reliable for erratic

            # Calendar features
            'calendar_features': True,
            'calendar_feature_list': ['month', 'quarter', 'week_of_year'],

            # Intermittency features - NOT needed
            'intermittency_features': False,

            # Transformations - log helps with variance
            'log_transform': True,
            'log_transform_method': 'log1p',  # log(1+x) handles zeros
            'box_cox_transform': False,
            'difference_transform': True,  # Helps with non-stationarity

            # Feature limits - slightly fewer due to noise
            'max_lag_features': 15,
            'max_rolling_features': 20,
            'max_total_features': 50 if volume == 'high' else 35,

            # Notes
            'engineering_notes': [
                "Erratic pattern - include volatility/variance features",
                "Log transform recommended to stabilize variance",
                "Rolling std and range features capture variability",
                "Consider differencing for stationarity",
                "Momentum features (change from previous periods) useful",
            ],
        }

    elif pattern == 'intermittent':
        config = {
            # Lag features - shorter lags for sparse data
            'target_lags': [1, 2, 4, 8, 13],
            'feature_lags': [1, 2, 4],

            # Rolling features - focus on sums and counts
            'rolling_windows': [4, 8, 13, 26],
            'rolling_stats': ['mean', 'sum', 'count_nonzero'],
            'rolling_agg_functions': {
                'mean': True,
                'sum': True,
                'count_nonzero': True,  # Count of non-zero periods
                'nonzero_mean': True,   # Mean when demand > 0
            },

            # EWM features - limited
            'ewm_spans': [4, 13],
            'ewm_stats': ['mean'],

            # Trend/Seasonal features - less useful
            'trend_features': False,
            'seasonal_features': False,
            'fourier_features': False,

            # Calendar features - may still help
            'calendar_features': True,
            'calendar_feature_list': ['month', 'quarter'],

            # INTERMITTENCY FEATURES - CRITICAL
            'intermittency_features': True,
            'intermittency_feature_list': [
                'demand_occurred',           # Binary: did demand occur?
                'periods_since_nonzero',     # Intervals since last demand
                'rolling_demand_frequency',  # Frequency in rolling window
                'avg_nonzero_demand',        # Average when demand > 0
                'std_nonzero_demand',        # Std when demand > 0
                'demand_arrival_rate',       # Estimated arrival rate
            ],

            # Transformations
            'log_transform': False,  # Don't log sparse data
            'box_cox_transform': False,
            'difference_transform': False,

            # Feature limits - keep simple
            'max_lag_features': 10,
            'max_rolling_features': 15,
            'max_total_features': 35 if volume == 'high' else 25,

            # Notes
            'engineering_notes': [
                "INTERMITTENT pattern - intermittency features CRITICAL",
                "Include demand_occurred binary for occurrence modeling",
                "periods_since_nonzero captures inter-arrival times",
                "Rolling count_nonzero shows demand frequency",
                "Consider two-stage: P(demand>0) then E[demand|demand>0]",
                "DO NOT use log transform on sparse data",
            ],
        }

    else:  # lumpy
        config = {
            # Lag features - very short due to extreme sparsity
            'target_lags': [1, 2, 4, 8],
            'feature_lags': [1, 2],

            # Rolling features - focus on occurrence patterns
            'rolling_windows': [4, 8, 13],
            'rolling_stats': ['sum', 'count_nonzero', 'max'],
            'rolling_agg_functions': {
                'sum': True,
                'count_nonzero': True,
                'max': True,  # Captures spikes
                'nonzero_mean': True,
                'spike_indicator': True,  # Large deviations from mean
            },

            # EWM features - minimal
            'ewm_spans': [4, 8],
            'ewm_stats': ['mean'],

            # Trend/Seasonal features - not reliable
            'trend_features': False,
            'seasonal_features': False,
            'fourier_features': False,

            # Calendar features
            'calendar_features': True,
            'calendar_feature_list': ['month', 'quarter'],

            # INTERMITTENCY FEATURES - CRITICAL FOR LUMPY
            'intermittency_features': True,
            'intermittency_feature_list': [
                'demand_occurred',
                'periods_since_nonzero',
                'rolling_demand_frequency',
                'avg_nonzero_demand',
                'std_nonzero_demand',
                'demand_arrival_rate',
                'cv_nonzero',              # CV of non-zero demands (high for lumpy)
                'spike_count',             # Number of demand spikes
            ],

            # Transformations
            'log_transform': False,
            'box_cox_transform': False,
            'difference_transform': False,

            # Feature limits - keep very simple
            'max_lag_features': 8,
            'max_rolling_features': 12,
            'max_total_features': 30 if volume == 'high' else 20,

            # Notes
            'engineering_notes': [
                "LUMPY pattern - hardest to forecast",
                "Intermittency features MANDATORY",
                "Focus on occurrence prediction first",
                "Include spike/outlier detection features",
                "Two-stage model essential: occurrence + severity",
                "Keep total features LOW - high regularization needed",
                "DO NOT use log transform - too many zeros",
            ],
        }

    return config


def create_segmentation_context_files(
    seg_dir: str,
    eda_dir: str,
    profiles: Dict[str, Dict[str, Any]],
    model_recs: Dict[str, Any],
    feature_recs: Dict[str, Any],
    clustering_metrics: Dict[str, Any],
    allowed_models: List[str],
) -> Tuple[str, str]:
    """
    Create deterministic context files for downstream crews.

    This function creates GROUNDED context files that ONLY use data
    from the actual utility outputs - NO hallucinated content.

    Parameters
    ----------
    seg_dir : str
        Segmentation output directory
    eda_dir : str
        EDA output directory
    profiles : Dict[str, Dict[str, Any]]
        Segment profiles from compute_segment_profiles()
    model_recs : Dict[str, Any]
        Model recommendations from assign_model_recommendations()
    feature_recs : Dict[str, Any]
        Feature recommendations from create_feature_recommendations()
    clustering_metrics : Dict[str, Any]
        Clustering metrics dictionary
    allowed_models : List[str]
        List of allowed model families

    Returns
    -------
    Tuple[str, str]
        Paths to (feature_context_path, training_context_path)
    """
    import json
    import os

    # Read EDA context files if they exist
    eda_features = {}
    eda_training = {}
    eda_feat_path = os.path.join(eda_dir, 'eda_to_feature_context.json')
    eda_train_path = os.path.join(eda_dir, 'eda_to_training_context.json')

    if os.path.exists(eda_feat_path):
        with open(eda_feat_path) as f:
            eda_features = json.load(f)
    if os.path.exists(eda_train_path):
        with open(eda_train_path) as f:
            eda_training = json.load(f)

    n_segments = clustering_metrics.get('n_segments', len(profiles))
    segment_labels = clustering_metrics.get('segment_labels', {})

    # Get EDA insights that were stored during clustering
    eda_insights = clustering_metrics.get('eda_insights', {})
    seasonality_info = eda_insights.get('seasonality', {})
    trend_info = eda_insights.get('trend', {})
    changepoint_info = eda_insights.get('changepoints', {})
    acf_info = eda_insights.get('autocorrelation', {})

    # =================================================================
    # CREATE FEATURE CONTEXT - GROUNDED IN ACTUAL UTILITY OUTPUTS
    # =================================================================
    feat_context = {
        'source': 'Segmentation Utilities (Deterministic)',
        'n_segments': n_segments,
        'segment_labels': segment_labels,
        # Use ACTUAL feature_recs from create_feature_recommendations()
        'segment_feature_strategies': feature_recs.get('segment_recommendations', {}),
        'global_recommendations': feature_recs.get('global_recommendations', {}),
        # EDA top features (if available)
        'eda_top_features': eda_features.get('top_features_by_importance', []),
        # EDA lag recommendations
        'lag_recommendations': {
            **eda_features.get('lag_recommendations', {}),
            'acf_informed_lags': acf_info.get('significant_lags', []),
        },
        'rolling_features': eda_features.get('rolling_features', {}),
        'transformation': eda_features.get('transformation', {}),
        # EDA insights for feature engineering decisions
        'eda_insights': {
            'seasonality': {
                'has_seasonality_pct': seasonality_info.get('has_seasonality_pct', 0),
                'dominant_period': seasonality_info.get('dominant_periods', {}),
            },
            'trend': {
                'avg_trend_strength': trend_info.get('avg_trend_strength', 0),
                'strongly_trending_pct': trend_info.get('strongly_trending_pct', 0),
            },
            'changepoints': {
                'pct_with_changepoints': changepoint_info.get('pct_with_changepoints', 0),
            },
        },
    }

    # Add per-segment guidance based on ACTUAL profiles
    feat_context['per_segment_guidance'] = {}
    for seg_id, profile in profiles.items():
        pattern = profile.get('dominant_pattern', 'smooth')
        feat_context['per_segment_guidance'][seg_id] = {
            'pattern': pattern,
            'intermittency_features': pattern in ['intermittent', 'lumpy'],
            'trend_features': pattern in ['smooth', 'erratic'],
            'seasonal_features': seasonality_info.get('has_seasonality_pct', 0) > 0.3,
        }

    # =================================================================
    # CREATE TRAINING CONTEXT - GROUNDED IN ACTUAL UTILITY OUTPUTS
    # =================================================================
    train_context = {
        'source': 'Segmentation Utilities (Deterministic)',
        'n_segments': n_segments,
        'segment_labels': segment_labels,
        # Use ACTUAL model_recs from assign_model_recommendations()
        'segment_model_strategy': model_recs.get('segment_strategies', {}),
        'allowed_model_families': allowed_models,
        'clustering_quality': {
            'algorithm': clustering_metrics.get('algorithm'),
            'silhouette': clustering_metrics.get('silhouette'),
            'n_segments': n_segments,
            'adaptive_features_used': clustering_metrics.get('adaptive_features_used', False),
        },
        # Loss function recommendations from EDA or defaults
        'loss_function_recommendations': eda_training.get('loss_function_recommendations', {
            'default': 'tweedie' if any(
                p.get('dominant_pattern') in ['intermittent', 'lumpy']
                for p in profiles.values()
            ) else 'mse',
            'for_intermittent': 'tweedie',
            'for_smooth': 'mse',
        }),
        # Evaluation metrics from EDA or defaults
        'evaluation_metrics': eda_training.get('evaluation_metrics', {
            'primary': 'wape',
            'secondary': ['mae', 'rmse', 'bias'],
        }),
    }

    # Add per-segment training guidance based on ACTUAL profiles
    train_context['per_segment_training'] = {}
    for seg_id, profile in profiles.items():
        pattern = profile.get('dominant_pattern', 'smooth')
        train_context['per_segment_training'][seg_id] = {
            'pattern': pattern,
            'recommended_loss': 'tweedie' if pattern in ['intermittent', 'lumpy'] else 'mse',
            'validation_strategy': 'holdout' if pattern in ['intermittent', 'lumpy'] else 'time_series_split',
            'expected_difficulty': 'high' if pattern == 'lumpy' else (
                'medium' if pattern in ['erratic', 'intermittent'] else 'low'
            ),
        }

    # Save context files
    feat_context_path = os.path.join(seg_dir, 'segmentation_to_feature_context.json')
    train_context_path = os.path.join(seg_dir, 'segmentation_to_training_context.json')

    with open(feat_context_path, 'w') as f:
        json.dump(feat_context, f, indent=2, default=str)
    with open(train_context_path, 'w') as f:
        json.dump(train_context, f, indent=2, default=str)

    return feat_context_path, train_context_path


# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def setup_matplotlib():
    """Setup matplotlib for non-interactive use."""
    import matplotlib
    matplotlib.use('Agg')


def plot_segment_distribution(
    labels: np.ndarray,
    output_path: str,
    segment_labels: Dict[str, str] = None,
    min_pct_line: float = DEFAULT_MIN_SEGMENT_PCT,
    figsize: Tuple[int, int] = (12, 6),
) -> str:
    """
    Create bar chart showing segment size distribution.

    Parameters
    ----------
    labels : np.ndarray
        Segment labels
    output_path : str
        Path to save the chart
    segment_labels : Dict[str, str], optional
        Human-readable labels for segments
    min_pct_line : float
        Draw horizontal line at minimum percentage
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved chart
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    seg_counts = pd.Series(labels).value_counts().sort_index()
    seg_pcts = seg_counts / len(labels) * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Color based on whether segment meets minimum
    colors = ['#2ecc71' if p >= min_pct_line * 100 else '#e74c3c' for p in seg_pcts]

    # Use human-readable labels if provided
    if segment_labels:
        x_labels = [segment_labels.get(str(i), f'Segment {i}') for i in seg_counts.index]
    else:
        x_labels = [f'Segment {i}' for i in seg_counts.index]

    bars = ax.bar(x_labels, seg_pcts, color=colors, edgecolor='black', alpha=0.8)

    # Add count labels on bars
    for bar, count, pct in zip(bars, seg_counts, seg_pcts):
        ax.annotate(
            f'{count:,}\n({pct:.1f}%)',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom', fontsize=9,
        )

    # Add minimum line
    if min_pct_line:
        ax.axhline(y=min_pct_line * 100, color='red', linestyle='--',
                   label=f'Minimum {min_pct_line * 100:.0f}%', linewidth=2)

    ax.set_xlabel('Segment', fontsize=12)
    ax.set_ylabel('Percentage of Total', fontsize=12)
    ax.set_title('Segment Size Distribution', fontsize=14, fontweight='bold')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_segment_heatmap(
    profiles: Dict[str, Dict[str, Any]],
    output_path: str,
    feature_cols: List[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> str:
    """
    Create heatmap showing relative indices for all segments.

    Parameters
    ----------
    profiles : Dict[str, Dict[str, Any]]
        Segment profiles
    output_path : str
        Path to save
    feature_cols : List[str], optional
        Features to include (auto-detected if not specified)
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved chart
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract feature columns from profiles
    if feature_cols is None:
        sample_profile = list(profiles.values())[0]
        feature_cols = list(sample_profile.get('relative_indices', {}).keys())[:10]

    # Build matrix
    segments = list(profiles.keys())
    matrix = np.zeros((len(segments), len(feature_cols)))

    for i, seg in enumerate(segments):
        indices = profiles[seg].get('relative_indices', {})
        for j, feat in enumerate(feature_cols):
            matrix[i, j] = indices.get(feat, 1.0)

    # Create labels
    y_labels = [f"Seg {s}: {profiles[s].get('dominant_pattern', '?')[:3].upper()}"
                for s in segments]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        xticklabels=[f.replace('_', '\n') for f in feature_cols],
        yticklabels=y_labels,
        cmap='RdYlGn',
        center=1.0,
        ax=ax,
        cbar_kws={'label': 'Relative Index (1.0 = average)'},
        linewidths=0.5,
    )

    # Rotate x-axis labels for readability when many features
    if len(feature_cols) > 8:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    ax.set_title('Segment Feature Profiles (Relative to Overall Average)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Segments', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_segment_scatter_2d(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    segment_labels: Dict[str, str] = None,
    method: str = 'PCA',
    figsize: Tuple[int, int] = (12, 10),
) -> str:
    """
    Create 2D scatter plot of segments using dimensionality reduction.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    labels : np.ndarray
        Segment labels
    output_path : str
        Path to save
    segment_labels : Dict[str, str], optional
        Human-readable labels
    method : str
        'PCA' or 'UMAP'
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved chart
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Reduce to 2D
    if method == 'PCA':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X_scaled)
        var_explained = reducer.explained_variance_ratio_
        axis_labels = [f'PC1 ({var_explained[0]:.1%})', f'PC2 ({var_explained[1]:.1%})']
    else:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X_scaled)
            axis_labels = ['UMAP1', 'UMAP2']
        except ImportError:
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X_scaled)
            var_explained = reducer.explained_variance_ratio_
            axis_labels = [f'PC1 ({var_explained[0]:.1%})', f'PC2 ({var_explained[1]:.1%})']

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        if segment_labels:
            leg_label = segment_labels.get(str(label), f'Segment {label}')
        else:
            leg_label = f'Segment {label}'

        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color], label=leg_label, alpha=0.7, s=50, edgecolors='white'
        )

        # Add centroid marker
        centroid = X_2d[mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=[color], s=200, marker='*',
                   edgecolors='black', linewidths=2)

    ax.set_xlabel(axis_labels[0], fontsize=12)
    ax.set_ylabel(axis_labels[1], fontsize=12)
    ax.set_title(f'Segment Visualization ({method})', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_demand_pattern_distribution(
    df: pd.DataFrame,
    labels: np.ndarray,
    output_path: str,
    figsize: Tuple[int, int] = (14, 6),
) -> str:
    """
    Create stacked bar chart showing demand pattern distribution per segment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'demand_pattern' column
    labels : np.ndarray
        Segment labels
    output_path : str
        Path to save
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved chart
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    if 'demand_pattern' not in df.columns:
        logger.warning("No 'demand_pattern' column found, skipping chart")
        return output_path

    df_temp = df.copy()
    df_temp['segment_id'] = labels

    # Create crosstab
    cross = pd.crosstab(df_temp['segment_id'], df_temp['demand_pattern'], normalize='index') * 100

    # Ensure all patterns present
    patterns = ['smooth', 'erratic', 'intermittent', 'lumpy']
    for p in patterns:
        if p not in cross.columns:
            cross[p] = 0
    cross = cross[patterns]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = {'smooth': '#2ecc71', 'erratic': '#f1c40f',
              'intermittent': '#e67e22', 'lumpy': '#e74c3c'}

    cross.plot(kind='bar', stacked=True, ax=ax,
               color=[colors[p] for p in patterns], edgecolor='white')

    ax.set_xlabel('Segment', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Demand Pattern Distribution by Segment', fontsize=14, fontweight='bold')
    ax.legend(title='Pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels([f'Segment {i}' for i in cross.index], rotation=45, ha='right')

    # Add percentage labels
    for i, seg in enumerate(cross.index):
        y_offset = 0
        for pattern in patterns:
            pct = cross.loc[seg, pattern]
            if pct > 5:  # Only label if > 5%
                ax.text(i, y_offset + pct/2, f'{pct:.0f}%',
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            y_offset += pct

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_silhouette_analysis(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    figsize: Tuple[int, int] = (12, 8),
) -> str:
    """
    Create silhouette analysis plot for cluster quality assessment.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    labels : np.ndarray
        Cluster labels
    output_path : str
        Path to save
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved chart
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_samples, silhouette_score

    n_clusters = len(set(labels))

    if n_clusters < 2:
        logger.warning("Need at least 2 clusters for silhouette analysis")
        return output_path

    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    avg_score = silhouette_score(X_scaled, labels)

    fig, ax = plt.subplots(figsize=figsize)

    y_lower = 10
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    for i, color in zip(range(n_clusters), colors):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_silhouette_values,
            facecolor=color, edgecolor=color, alpha=0.7,
        )

        # Label
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Seg {i}', fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_score:.3f}')

    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Segment', fontsize=12)
    ax.set_title('Silhouette Analysis by Segment', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

# =============================================================================
# 7.5 HYBRID SEGMENTATION
# =============================================================================

def create_hybrid_segments(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    use_volume_tier: bool = True,
    use_demand_pattern: bool = True,
    min_segment_size: int = 10,
    min_final_segments: int = 3,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Create hybrid segments by combining cluster labels with known business dimensions.

    This creates more meaningful segments by crossing:
    - Statistical clusters (from clustering algorithm)
    - Volume tier (low/medium/high) if available
    - Demand pattern (smooth/erratic/intermittent/lumpy) if available

    The result is a richer segmentation that respects both statistical patterns
    AND business-meaningful dimensions.

    IMPORTANT: This function intelligently merges small segments while preserving
    at least min_final_segments segments to maintain meaningful segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume_tier and/or demand_pattern columns
    cluster_labels : np.ndarray
        Labels from clustering algorithm
    use_volume_tier : bool
        Whether to include volume_tier in hybrid segments
    use_demand_pattern : bool
        Whether to include demand_pattern in hybrid segments
    min_segment_size : int
        Minimum keys per segment; smaller segments are merged
    min_final_segments : int
        Minimum number of hybrid segments to preserve (default 3).
        Set to 2 for smaller datasets.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, str]]
        - hybrid_labels: New segment labels
        - segment_descriptions: Human-readable description of each segment
    """
    df = df.copy()
    df['_cluster'] = cluster_labels

    # Build hybrid segment key
    components = ['_cluster']
    component_names = ['Cluster']

    if use_volume_tier and 'volume_tier' in df.columns:
        components.append('volume_tier')
        component_names.append('Volume')
        logger.info("Including volume_tier in hybrid segmentation")

    if use_demand_pattern and 'demand_pattern' in df.columns:
        components.append('demand_pattern')
        component_names.append('Pattern')
        logger.info("Including demand_pattern in hybrid segmentation")

    # If no additional dimensions available, just return cluster labels
    if len(components) == 1:
        logger.info("No additional dimensions for hybrid segmentation, using cluster labels only")
        return cluster_labels, {str(i): f'Segment-{i}' for i in set(cluster_labels)}

    # Create hybrid segment string
    df['_hybrid_key'] = df[components].astype(str).agg('_'.join, axis=1)

    # Get unique hybrid segments and their sizes
    hybrid_counts = df['_hybrid_key'].value_counts()
    initial_segment_count = len(hybrid_counts)
    logger.info(f"Hybrid segmentation initial: {initial_segment_count} segments from {len(set(cluster_labels))} clusters")

    # Identify small segments that need merging
    small_segments = hybrid_counts[hybrid_counts < min_segment_size].index.tolist()

    # -------------------------------------------------------------------------
    # PATTERN-FAMILY-AWARE MERGING
    # -------------------------------------------------------------------------
    # Define demand pattern families to prevent inappropriate mixing:
    #   Family "regular": smooth + erratic (consistent demand occurrence)
    #   Family "sparse":  intermittent + lumpy (sporadic demand)
    # Small segments should ONLY merge into segments of the same family
    # to prevent lumpy keys being pooled with smooth keys in a segment model.
    # -------------------------------------------------------------------------
    REGULAR_PATTERNS = {'smooth', 'erratic'}
    SPARSE_PATTERNS = {'intermittent', 'lumpy'}

    def _get_pattern_family(hybrid_key: str) -> str:
        """Extract demand pattern family from hybrid key string."""
        parts = hybrid_key.split('_')
        # demand_pattern is the last component in cluster_volume_pattern format
        for part in reversed(parts):
            part_lower = part.lower()
            if part_lower in REGULAR_PATTERNS:
                return 'regular'
            if part_lower in SPARSE_PATTERNS:
                return 'sparse'
        return 'unknown'

    def _find_merge_target(small_seg, candidates, hybrid_counts_map):
        """Find best merge target respecting pattern family.

        Priority: 1) same cluster + same family, 2) any cluster + same family,
        3) same cluster any family (last resort with warning).
        """
        cluster_id = small_seg.split('_')[0]
        seg_family = _get_pattern_family(small_seg)

        # Priority 1: same cluster + same pattern family
        same_cluster_same_family = [
            s for s in candidates
            if s.startswith(f"{cluster_id}_") and _get_pattern_family(s) == seg_family
        ]
        if same_cluster_same_family:
            return max(same_cluster_same_family, key=lambda s: hybrid_counts_map.get(s, 0))

        # Priority 2: any cluster + same pattern family
        any_cluster_same_family = [
            s for s in candidates
            if _get_pattern_family(s) == seg_family
        ]
        if any_cluster_same_family:
            return max(any_cluster_same_family, key=lambda s: hybrid_counts_map.get(s, 0))

        # Priority 3 (last resort): same cluster, any family — log warning
        same_cluster_any = [s for s in candidates if s.startswith(f"{cluster_id}_")]
        if same_cluster_any:
            target = max(same_cluster_any, key=lambda s: hybrid_counts_map.get(s, 0))
            logger.warning(
                f"Pattern-aware merge: No same-family target for '{small_seg}' "
                f"(family={seg_family}). Merging into '{target}' (cross-family)."
            )
            return target

        # Absolute fallback: globally largest
        if candidates:
            target = max(candidates, key=lambda s: hybrid_counts_map.get(s, 0))
            logger.warning(
                f"Pattern-aware merge: No same-cluster or same-family target for "
                f"'{small_seg}'. Merging into globally largest '{target}'."
            )
            return target

        return None

    # CRITICAL: Preserve at least min_final_segments
    # Only merge if we have more segments than minimum
    hybrid_counts_map = hybrid_counts.to_dict()

    if small_segments:
        # Calculate how many segments would remain after merging all small ones
        large_segments = [s for s in hybrid_counts.index if s not in small_segments]
        n_large = len(large_segments)

        if n_large >= min_final_segments:
            # Safe to merge all small segments
            logger.info(f"Merging {len(small_segments)} small hybrid segments (< {min_segment_size} keys) with pattern-aware logic")
            for small_seg in small_segments:
                target = _find_merge_target(small_seg, large_segments, hybrid_counts_map)
                if target:
                    df.loc[df['_hybrid_key'] == small_seg, '_hybrid_key'] = target
        else:
            # Not enough large segments - only merge the smallest ones to reach min_final_segments
            n_to_merge = max(0, initial_segment_count - min_final_segments)

            if n_to_merge > 0:
                # Sort by size and merge only the smallest ones
                segments_by_size = hybrid_counts.sort_values().index.tolist()
                segments_to_merge = segments_by_size[:n_to_merge]

                logger.info(
                    f"Hybrid segmentation: merging only {len(segments_to_merge)} smallest segments "
                    f"to preserve {min_final_segments} segments (pattern-aware)"
                )

                remaining_segs = [s for s in hybrid_counts.index if s not in segments_to_merge]
                for small_seg in segments_to_merge:
                    target = _find_merge_target(small_seg, remaining_segs, hybrid_counts_map)
                    if target:
                        df.loc[df['_hybrid_key'] == small_seg, '_hybrid_key'] = target
            else:
                logger.info(
                    f"Hybrid segmentation: keeping all {initial_segment_count} segments "
                    f"(already at or below minimum {min_final_segments})"
                )

    # Create numeric labels
    unique_hybrids = sorted(df['_hybrid_key'].unique())
    hybrid_to_label = {h: i for i, h in enumerate(unique_hybrids)}
    hybrid_labels = df['_hybrid_key'].map(hybrid_to_label).values

    # Create human-readable descriptions
    segment_descriptions = {}
    for hybrid_key, label in hybrid_to_label.items():
        parts = hybrid_key.split('_')
        desc_parts = []
        for i, (part, name) in enumerate(zip(parts, component_names)):
            if name == 'Cluster':
                desc_parts.append(f"C{part}")
            elif name == 'Volume':
                desc_parts.append(f"{part.title()}-Vol")
            elif name == 'Pattern':
                desc_parts.append(part.title())
        segment_descriptions[str(label)] = '-'.join(desc_parts)

    n_hybrid = len(unique_hybrids)
    n_clusters = len(set(cluster_labels))
    logger.info(f"Hybrid segmentation: {n_clusters} clusters → {n_hybrid} hybrid segments (min preserved: {min_final_segments})")

    return hybrid_labels, segment_descriptions


# =============================================================================
# 8. HIGH-LEVEL PIPELINE
# =============================================================================

def run_segmentation_pipeline(
    per_key_metrics: pd.DataFrame,
    key_cols: List[str],
    output_dir: str,
    allowed_model_families: List[str] = None,
    enable_deep_models: bool = False,
    time_format: str = 'year_week',
    clustering_method: str = 'gmm',
    n_clusters_range: List[int] = None,
    min_segment_pct: float = DEFAULT_MIN_SEGMENT_PCT,
    create_visualizations: bool = True,
    eda_dir: str = None,
    use_adaptive_features: bool = True,
    use_hybrid_segmentation: bool = True,
    hybrid_use_volume_tier: bool = True,
    hybrid_use_demand_pattern: bool = True,
    # Phase 2: Enhanced segmentation parameters
    source_df: Optional[pd.DataFrame] = None,
    date_col: Optional[str] = None,
    target_col: Optional[str] = None,
    feature_availability_context: Optional[Dict[str, Any]] = None,
    enable_enriched_features: bool = True,
    hierarchy_cols: Optional[List[str]] = None,
) -> SegmentationResult:
    """
    Run COMPLETE world-class segmentation pipeline in ONE call.

    This pipeline:
    1. Creates comprehensive segmentation features (volume, pattern, variability)
    2. Scales features using robust scaling
    3. Clusters using the specified method
    4. Enforces minimum segment sizes
    5. Profiles each segment
    6. Generates human-readable labels
    7. Creates model recommendations
    8. Creates feature engineering recommendations
    9. Generates visualizations
    10. Saves all outputs

    Parameters
    ----------
    per_key_metrics : pd.DataFrame
        Per-key metrics from EDA (must have: mean, std, cv, adi, cv2, zero_fraction)
    key_cols : List[str]
        Key identifier columns
    output_dir : str
        Output directory for all files
    allowed_model_families : List[str], optional
        Allowed model families for recommendations
    enable_deep_models : bool
        Whether deep learning models are enabled
    time_format : str
        'year_week' or 'year_month' (affects lag recommendations)
    clustering_method : str
        'gmm' (recommended), 'kmeans', 'hdbscan', 'hierarchical'
    n_clusters_range : List[int], optional
        Range of clusters to try (default [3, 4, 5, 6, 7])
    min_segment_pct : float
        Minimum segment size percentage (default 10%)
    create_visualizations : bool
        Whether to create charts
    eda_dir : str, optional
        EDA output directory for enrichment with EDA insights (seasonality, trend, etc.)
    use_adaptive_features : bool
        Whether to use adaptive feature selection based on EDA insights (default True)
    use_hybrid_segmentation : bool
        Whether to combine clusters with known business dimensions (volume_tier, demand_pattern)
        This creates richer, more meaningful segments (default True)
    hybrid_use_volume_tier : bool
        Include volume_tier in hybrid segmentation (default True)
    hybrid_use_demand_pattern : bool
        Include demand_pattern in hybrid segmentation (default True)
    source_df : pd.DataFrame, optional
        Raw source data. When provided with enable_enriched_features=True,
        enables Phase 2 enriched segmentation: external response profiles,
        seasonality shape features, trend direction, hierarchy features,
        and feature importance profiles.
    feature_availability_context : Dict, optional
        Context from Feature Availability Detection. Used to identify which
        external features to compute response profiles for.
    enable_enriched_features : bool
        When True and source_df is provided, computes Phase 2 enriched
        segmentation features (default True).
    hierarchy_cols : List[str], optional
        Explicit hierarchy columns. If None, auto-detected from data.

    Returns
    -------
    SegmentationResult
        Complete segmentation results

    Example
    -------
    >>> result = run_segmentation_pipeline(
    ...     per_key_metrics=per_key_df,
    ...     key_cols=['key'],
    ...     output_dir='seg_output',
    ...     allowed_model_families=['lightgbm', 'croston', 'sba'],
    ... )
    >>> print(f"Created {result.n_segments} segments")
    """
    os.makedirs(output_dir, exist_ok=True)

    # Defaults
    if allowed_model_families is None:
        allowed_model_families = ['lightgbm', 'xgboost', 'catboost', 'croston', 'sba', 'prophet']

    # -------------------------------------------------------------------------
    # DATA-ADAPTIVE CLUSTER RANGE AND MINIMUM SEGMENT SIZE
    # -------------------------------------------------------------------------
    # The old hardcoded [3..7] range and 10% minimum were too conservative.
    # With enriched features we can support more segments.
    #
    # Rules:
    #   n_keys < 50   → k ∈ [2..5],   min_segment = max(5, 3% of keys)
    #   n_keys < 200  → k ∈ [3..8],   min_segment = max(5, 5% of keys)
    #   n_keys < 500  → k ∈ [3..12],  min_segment = max(8, 5% of keys)
    #   n_keys < 2000 → k ∈ [4..15],  min_segment = max(10, 3% of keys)
    #   n_keys >= 2000→ k ∈ [5..20],  min_segment = max(15, 2% of keys)
    # -------------------------------------------------------------------------
    n_total_keys = len(per_key_metrics)

    # Validate minimum keys for clustering
    if n_total_keys < 2:
        logger.warning(f"Only {n_total_keys} keys — too few for clustering. Assigning all to segment 0.")
        per_key_metrics['segment_id'] = 0
        # Save minimal outputs and return
        os.makedirs(output_dir, exist_ok=True)
        per_key_metrics.to_csv(os.path.join(output_dir, 'per_key_with_segments.csv'), index=False)
        return SegmentationResult(
            labels=np.zeros(n_total_keys, dtype=int), n_segments=1,
            segment_profiles={'0': {'size': n_total_keys, 'pct_of_total': 100}},
            segment_distribution={'0': 1.0}, segment_labels={'0': 'all_keys'},
            clustering_metrics={'silhouette': 0, 'n_clusters': 1},
            model_recommendations={}, feature_recommendations={},
        )

    if n_clusters_range is None:
        # 2026-04 OVER-SEGMENTATION FIX: cap the upper bound aggressively.
        # Empirically on COOKING_AID (7120 keys) the old range(5, 21) picked
        # 18 clusters with silhouette=-0.21 (WORSE than random); every
        # segment was labeled 'lumpy' so segment-level demand-pattern
        # signals were useless. The underlying feature space is only 3-6
        # dims (intermittency / CV / volume / trend / hierarchy share) —
        # it does not support 18 well-separated clusters.
        #
        # New ranges cap at ~8 clusters even for very large populations.
        # With the variance-filter fix above putting 5-6 clean features
        # back in play, this should land on 4-6 clusters with healthy
        # silhouette values.
        if n_total_keys < 50:
            n_clusters_range = list(range(2, 5))
            adaptive_min_pct = 0.05
        elif n_total_keys < 200:
            n_clusters_range = list(range(2, 6))
            adaptive_min_pct = 0.05
        elif n_total_keys < 500:
            n_clusters_range = list(range(3, 7))
            adaptive_min_pct = 0.05
        elif n_total_keys < 2000:
            n_clusters_range = list(range(3, 8))
            adaptive_min_pct = 0.04
        else:
            # Previously up to 20; now capped at 8 to prevent the
            # negative-silhouette, all-lumpy pathology seen on UK.
            n_clusters_range = list(range(3, 9))
            adaptive_min_pct = 0.04

        # Use the data-adaptive min_pct unless user explicitly set it
        if min_segment_pct == DEFAULT_MIN_SEGMENT_PCT:
            min_segment_pct = adaptive_min_pct
            logger.info(
                f"Data-adaptive settings: {n_total_keys} keys → "
                f"k_range={n_clusters_range}, min_segment_pct={min_segment_pct:.0%}"
            )

    logger.info(f"Starting segmentation pipeline with {n_total_keys} series")
    logger.info(f"Cluster range: {n_clusters_range}, min segment: {min_segment_pct:.0%}")
    logger.info(f"Input per_key_metrics has {len(per_key_metrics.columns)} columns: {list(per_key_metrics.columns)}")

    # -------------------------------------------------------------------------
    # STEP 1: Create comprehensive segmentation features
    # IMPORTANT: This preserves ALL original columns from per_key_metrics
    # -------------------------------------------------------------------------

    # Dynamically detect external feature aggregate columns from per_key_metrics
    # These are created by compute_demand_characteristics with config-driven features
    all_cols = per_key_metrics.columns.tolist()

    # Detect price aggregate columns (created from config.price_features)
    price_agg_cols = [c for c in all_cols if c.startswith('price_') and
                      any(suffix in c for suffix in ['mean', 'std', 'cv', 'min', 'max', 'frequency'])]

    # Detect promo aggregate columns (created from config.promo_features)
    promo_agg_cols = [c for c in all_cols if c.startswith('promo_') and
                      any(suffix in c for suffix in ['mean', 'std', 'cv', 'min', 'max', 'frequency'])]

    # Detect categorical encoded columns (created from config categorical features)
    categorical_encoded_cols = [c for c in all_cols if
                                '_encoded_mean' in c or '_encoded_max' in c or '_diversity' in c]

    if price_agg_cols:
        logger.info(f"Detected {len(price_agg_cols)} price aggregate columns for segmentation")
    if promo_agg_cols:
        logger.info(f"Detected {len(promo_agg_cols)} promo aggregate columns for segmentation")
    if categorical_encoded_cols:
        logger.info(f"Detected {len(categorical_encoded_cols)} categorical encoded columns for segmentation")

    df = create_segmentation_features(
        per_key_metrics=per_key_metrics,
        key_cols=key_cols,
        include_volume=True,
        include_intermittency=True,
        include_variability=True,
        include_temporal=True,
        include_external=True,
        price_agg_cols=price_agg_cols if price_agg_cols else None,
        promo_agg_cols=promo_agg_cols if promo_agg_cols else None,
        categorical_encoded_cols=categorical_encoded_cols if categorical_encoded_cols else None,
        time_format=time_format,
    )
    logger.info(f"After create_segmentation_features: {len(df.columns)} columns")

    # -------------------------------------------------------------------------
    # STEP 1.5: Enrich with EDA insights (if eda_dir provided)
    # -------------------------------------------------------------------------
    eda_insights = {}
    if eda_dir and os.path.exists(eda_dir):
        try:
            df, eda_insights = enrich_with_eda_insights(df, eda_dir, key_cols)
            logger.info(f"Enriched with EDA insights: {len(df.columns)} features")
        except Exception as e:
            logger.warning(f"Could not enrich with EDA insights: {e}")

    # -------------------------------------------------------------------------
    # STEP 1.7: Phase 2 - Enriched segmentation features (external response,
    #           seasonality shape, trend direction, hierarchy, importance)
    # -------------------------------------------------------------------------
    enrichment_metadata = {}
    if enable_enriched_features and source_df is not None:
        try:
            from utils.segmentation_enhanced import enrich_segmentation_features

            _date_col = date_col or (source_df.columns[1] if len(source_df.columns) > 1 else None)
            _target_col = target_col or (source_df.columns[2] if len(source_df.columns) > 2 else None)
            if _date_col and _target_col and _date_col in source_df.columns and _target_col in source_df.columns:
                logger.info("Running Phase 2 enriched segmentation features...")
                df, enrichment_metadata = enrich_segmentation_features(
                    per_key_metrics=df,
                    source_df=source_df,
                    key_cols=key_cols,
                    date_col=_date_col,
                    target_col=_target_col,
                    time_format=time_format,
                    feature_availability_context=feature_availability_context,
                    hierarchy_cols=hierarchy_cols,
                    auto_detect_hierarchy=True,
                    compute_response_profiles=True,
                    compute_seasonality=True,
                    compute_trends=True,
                    compute_importance=True,
                )
            else:
                logger.warning(f"Cannot run Phase 2 enrichment: date_col={_date_col}, target_col={_target_col}")
            logger.info(f"Phase 2 enrichment added {len(enrichment_metadata.get('features_added', []))} features")
        except Exception as e:
            logger.warning(f"Phase 2 enrichment failed (non-critical, continuing with base features): {e}")
    elif source_df is None:
        logger.info("No source_df provided — skipping Phase 2 enriched features")
    else:
        logger.info("Enriched features disabled — using base segmentation features only")

    # -------------------------------------------------------------------------
    # STEP 2: Get recommended clustering features
    # -------------------------------------------------------------------------
    if use_adaptive_features and eda_insights:
        clustering_features = get_recommended_clustering_features(df, strategy='adaptive', eda_insights=eda_insights)
    else:
        clustering_features = get_recommended_clustering_features(df, strategy='comprehensive')
    logger.info(f"Initial clustering features ({len(clustering_features)}): {clustering_features}")

    # -------------------------------------------------------------------------
    # STEP 2.5: Scale features FIRST, then filter low-variance
    # -------------------------------------------------------------------------
    # IMPORTANT: We scale BEFORE variance filtering because raw features have
    # vastly different scales (e.g., volume_mean=125000 variance vs cv=1.7).
    # Without scaling first, the variance filter would eliminate all normalized
    # features since their variance appears tiny compared to raw volume.
    #
    # FILTER-DECISION BUG FIX (2026-04): We previously used RobustScaler for
    # the filter-decision and a RELATIVE threshold (max_scaled_var * 0.001).
    # RobustScaler uses IQR, so heavy-tailed features (e.g., hier_market_name_
    # _share with a few extreme values) end up with scaled variance of 10³–10⁴
    # while well-behaved normalized features sit near 1. The relative 0.1%-
    # of-max threshold then balloons to ~2-10, KILLING every normalized
    # feature — which is why COOKING_AID's 18-segment solution was built on
    # only 3 features and produced silhouette=-0.21 (worse than random).
    #
    # Fix: use StandardScaler (z-score) for the filter-decision — every
    # feature has population variance of ~1 after scaling, so an ABSOLUTE
    # variance floor (0.01) is robust regardless of the input distribution.
    # We still scale with RobustScaler for the actual clustering input later
    # (STEP 3) because it's more robust to outliers during GMM/KMeans fit.
    from sklearn.preprocessing import RobustScaler, StandardScaler
    valid_features = [f for f in clustering_features if f in df.columns]
    if len(valid_features) < 3:
        logger.warning(f"Only {len(valid_features)} features exist in df. Using available features.")
        valid_features = [f for f in ['volume_mean', 'cv_clean', 'zero_fraction_clean', 'adi_log'] if f in df.columns]

    # Filter to only NUMERIC features (exclude categorical)
    numeric_features = []
    for f in valid_features:
        dtype = df[f].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_categorical_dtype(dtype):
            numeric_features.append(f)
        else:
            logger.debug(f"Excluding non-numeric feature '{f}' (dtype={dtype}) from variance scaling")

    if len(numeric_features) < 3:
        logger.warning(f"Only {len(numeric_features)} numeric features. Using all available.")
        numeric_features = [f for f in ['volume_mean', 'cv_clean', 'zero_fraction_clean', 'adi_log',
                                        'forecastability_score', 'demand_frequency'] if f in df.columns]

    valid_features = numeric_features

    # StandardScaler (z-score) for filter-decision: each feature has ~unit
    # variance post-scaling, so an ABSOLUTE floor is meaningful.
    _filter_scaler = StandardScaler()
    X_temp = df[valid_features].fillna(df[valid_features].median())
    X_scaled_temp = _filter_scaler.fit_transform(X_temp)

    import numpy as np
    scaled_variances = np.var(X_scaled_temp, axis=0)
    # ABSOLUTE variance floor — a z-scored feature with variance < 0.01 has
    # essentially collapsed to a constant (std < 0.1 after z-scoring means
    # the raw feature is tiny compared to a single outlier that dominated
    # the mean/std fit). Previously we used max*0.001 which degenerates on
    # heavy-tailed inputs and throws away healthy features.
    variance_threshold = 0.01

    # Also check unique ratio
    n_rows = len(df)
    kept_features = []
    removed_features = []
    removed_details: List[str] = []

    for i, feat in enumerate(valid_features):
        scaled_var = scaled_variances[i]
        n_unique = df[feat].nunique()
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0

        # Keep if: variance is sufficient AND enough unique values
        is_low_variance = scaled_var < variance_threshold
        is_near_constant = unique_ratio < 0.01  # 1% unique values

        if is_low_variance or is_near_constant:
            removed_features.append(feat)
            removed_details.append(
                f"{feat} (z_var={scaled_var:.4f}, unique_ratio={unique_ratio:.3f})"
            )
        else:
            kept_features.append(feat)

    if removed_features:
        logger.info(
            f"Variance filter (z-score based, threshold={variance_threshold}) removed "
            f"{len(removed_features)} features: {removed_details}"
        )

    clustering_features = kept_features

    # Ensure we have at least 3 features for meaningful clustering
    if len(clustering_features) < 3:
        logger.warning(
            f"Only {len(clustering_features)} features passed scaled variance filter. "
            f"Adding back important features to reach minimum of 3."
        )
        # Add back important features that exist
        important_features = ['volume_mean', 'cv_clean', 'zero_fraction_clean', 'adi_log',
                             'forecastability_score', 'complexity_score', 'demand_frequency']
        for feat in important_features:
            if feat in df.columns and feat not in clustering_features:
                clustering_features.append(feat)
                if len(clustering_features) >= 5:  # Aim for 5 features minimum
                    break

    logger.info(f"Final clustering features ({len(clustering_features)}): {clustering_features}")

    # -------------------------------------------------------------------------
    # STEP 3: Scale features
    # -------------------------------------------------------------------------
    X_scaled, scaler = scale_features(df, clustering_features, scaler_type='robust')
    logger.info(f"Scaled features: shape {X_scaled.shape}")

    # -------------------------------------------------------------------------
    # STEP 4: Run clustering
    # -------------------------------------------------------------------------
    # Normalize clustering method name to handle various aliases
    method_lower = clustering_method.lower().strip()
    gmm_aliases = ['gmm', 'gaussianmixture', 'gaussian_mixture', 'gaussian-mixture', 'mixture']
    kmeans_aliases = ['kmeans', 'k-means', 'k_means']
    hdbscan_aliases = ['hdbscan', 'dbscan']

    if method_lower in gmm_aliases:
        logger.info(f"Using GMM clustering (method='{clustering_method}')")
        labels, clusterer, metrics = cluster_gaussian_mixture(
            X_scaled, n_range=n_clusters_range, auto_select=True
        )
    elif method_lower in kmeans_aliases:
        logger.info(f"Using KMeans clustering (method='{clustering_method}')")
        labels, clusterer, metrics = cluster_kmeans(
            X_scaled, k_range=n_clusters_range, auto_select=True
        )
    elif method_lower in hdbscan_aliases:
        logger.info(f"Using HDBSCAN clustering (method='{clustering_method}')")
        labels, clusterer, metrics = cluster_hdbscan(
            X_scaled, min_pct=min_segment_pct
        )
    else:  # hierarchical or unknown
        logger.info(f"Using Hierarchical clustering (method='{clustering_method}')")
        labels, clusterer, metrics = cluster_hierarchical_optimal(
            X_scaled, n_range=n_clusters_range
        )

    logger.info(f"Initial clustering: {metrics.n_clusters} clusters, silhouette={metrics.silhouette:.3f}")

    # Save original labels before size enforcement (for building the mapping)
    original_labels = labels.copy()

    # -------------------------------------------------------------------------
    # STEP 5: Enforce minimum segment size
    # -------------------------------------------------------------------------
    # Dynamically determine minimum final segments based on data size
    # and the number of clusters the algorithm actually found
    n_keys_for_enforcement = len(df)
    actual_n_clusters = metrics.n_clusters
    if n_keys_for_enforcement >= 1000:
        min_final_segments = max(5, actual_n_clusters // 2)
    elif n_keys_for_enforcement >= 500:
        min_final_segments = max(4, actual_n_clusters // 2)
    elif n_keys_for_enforcement >= 200:
        min_final_segments = max(3, actual_n_clusters // 2)
    elif n_keys_for_enforcement >= 50:
        min_final_segments = max(3, actual_n_clusters // 2)
    else:
        min_final_segments = 2

    logger.info(f"Dataset has {n_keys_for_enforcement} keys, GMM found {actual_n_clusters} clusters, requiring min_final_segments={min_final_segments}")

    labels = enforce_minimum_segment_size(
        X_scaled, labels,
        min_pct=min_segment_pct,
        min_final_segments=min_final_segments
    )
    n_clusters_after_enforcement = len(set(labels))
    logger.info(f"After size enforcement: {n_clusters_after_enforcement} base clusters")

    # Build mapping from original cluster model predictions to final segment IDs
    # This is needed for inference when we predict new keys with the saved cluster model
    cluster_to_segment_map = {}
    for orig, final in zip(original_labels, labels):
        if orig not in cluster_to_segment_map:
            cluster_to_segment_map[int(orig)] = int(final)
    logger.info(f"Cluster to segment mapping: {cluster_to_segment_map}")

    # -------------------------------------------------------------------------
    # STEP 5.5: HYBRID SEGMENTATION (combine clusters with business dimensions)
    # -------------------------------------------------------------------------
    hybrid_descriptions = None
    if use_hybrid_segmentation:
        # Check if we have business dimensions available
        has_volume_tier = 'volume_tier' in df.columns and hybrid_use_volume_tier
        has_demand_pattern = 'demand_pattern' in df.columns and hybrid_use_demand_pattern

        if has_volume_tier or has_demand_pattern:
            logger.info("Creating hybrid segments combining clusters with business dimensions...")
            # Use min of cap (50) or percentage, with floor of 5
            pct_based = int(len(df) * min_segment_pct)
            min_hybrid_size = max(min(MIN_SEGMENT_SIZE_CAP, pct_based), 5)

            # For hybrid segmentation, allow more segments since we're adding dimensions
            # But still maintain a reasonable minimum
            hybrid_min_segments = max(min_final_segments, n_clusters_after_enforcement)

            labels, hybrid_descriptions = create_hybrid_segments(
                df=df,
                cluster_labels=labels,
                use_volume_tier=has_volume_tier,
                use_demand_pattern=has_demand_pattern,
                min_segment_size=min_hybrid_size,
                min_final_segments=hybrid_min_segments,
            )
            n_segments = len(set(labels))
            logger.info(f"Hybrid segmentation: {n_clusters_after_enforcement} clusters → {n_segments} segments")
        else:
            n_segments = n_clusters_after_enforcement
            logger.info("No business dimensions available for hybrid segmentation, using cluster labels only")
    else:
        n_segments = n_clusters_after_enforcement

    # -------------------------------------------------------------------------
    # STEP 6: Add segment labels to DataFrame
    # -------------------------------------------------------------------------
    df['segment_id'] = labels

    # -------------------------------------------------------------------------
    # STEP 7: Compute segment profiles
    # -------------------------------------------------------------------------
    profiles = compute_segment_profiles(df, labels, clustering_features, key_cols)

    # -------------------------------------------------------------------------
    # STEP 8: Generate human-readable labels
    # -------------------------------------------------------------------------
    if hybrid_descriptions:
        # Use hybrid descriptions as segment labels
        segment_labels = hybrid_descriptions
    else:
        segment_labels = label_segments_automatically(profiles)

    # Add labels to profiles
    for seg_id in profiles:
        profiles[seg_id]['label'] = segment_labels.get(seg_id, f'Segment-{seg_id}')

    # -------------------------------------------------------------------------
    # STEP 9: Create model recommendations
    # -------------------------------------------------------------------------
    model_recommendations = assign_model_recommendations(
        profiles, allowed_model_families, enable_deep_models
    )

    # -------------------------------------------------------------------------
    # STEP 10: Create feature recommendations
    # -------------------------------------------------------------------------
    feature_recommendations = create_feature_recommendations(profiles, time_format)

    # -------------------------------------------------------------------------
    # STEP 11: Save outputs
    # -------------------------------------------------------------------------
    # Standardise key column to 'key' — downstream pipeline expects this name
    actual_key_col = key_cols[0] if isinstance(key_cols, list) else key_cols
    if actual_key_col != 'key' and actual_key_col in df.columns and 'key' not in df.columns:
        df = df.rename(columns={actual_key_col: 'key'})

    # Per-key with segments - MUST include ALL original EDA metrics plus segmentation features
    logger.info(f"Saving per_key_with_segments.csv with {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"  Columns: {list(df.columns)}")
    df.to_csv(os.path.join(output_dir, 'per_key_with_segments.csv'), index=False)

    # Segment profiles
    with open(os.path.join(output_dir, 'segment_profiles.json'), 'w') as f:
        json.dump(profiles, f, indent=2, default=str)

    # Model recommendations
    with open(os.path.join(output_dir, 'model_recommendations.json'), 'w') as f:
        json.dump(model_recommendations, f, indent=2, default=str)

    # Feature recommendations
    with open(os.path.join(output_dir, 'feature_recommendations.json'), 'w') as f:
        json.dump(feature_recommendations, f, indent=2, default=str)

    # Clustering metrics
    # NOTE: clustering_metrics.json is saved AFTER visualizations are created
    # so we can include visualization file list. See STEP 13 visualization section.

    # Save scaler and clusterer
    try:
        import joblib
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(clusterer, os.path.join(output_dir, 'cluster_model.joblib'))

        # Save the cluster-to-segment mapping for inference
        # This maps the raw cluster model predictions to final segment IDs
        with open(os.path.join(output_dir, 'cluster_to_segment_map.json'), 'w') as f:
            json.dump(cluster_to_segment_map, f, indent=2)
        logger.info(f"Saved cluster_to_segment_map.json with {len(cluster_to_segment_map)} mappings")
    except Exception as e:
        logger.warning(f"Could not save models: {e}")

    # -------------------------------------------------------------------------
    # STEP 12: Create STATE-OF-THE-ART context files for downstream crews
    # -------------------------------------------------------------------------
    # These context files contain everything downstream crews need to make
    # intelligent, segment-aware decisions.

    # ==========================================================================
    # SEGMENTATION → FEATURE ENGINEERING CONTEXT
    # ==========================================================================
    # Contains everything the Feature Engineering crew needs to create
    # segment-appropriate features.

    # Build backward-compatible segment_profiles for Feature Crew
    # (Feature Crew code expects segment_profiles with 'demand_pattern' field)
    segment_profiles_compat = {}
    for seg_id, profile in profiles.items():
        segment_profiles_compat[seg_id] = {
            'demand_pattern': profile.get('dominant_pattern', 'smooth'),  # Required by Feature Crew
            'dominant_pattern': profile.get('dominant_pattern', 'smooth'),
            'volume_tier': profile.get('dominant_volume_tier', 'medium'),
            'size': profile.get('size', 0),
            'pct_of_total': profile.get('pct_of_total', 0),
            'label': profile.get('label', f'Segment-{seg_id}'),
        }

    seg_to_feat_context = {
        'source': 'segmentation_pipeline_v2',
        'generation_method': 'state_of_the_art_deterministic',

        # Segment summary
        'n_segments': n_segments,
        'segment_labels': segment_labels,
        'segment_distribution': {str(s): round((labels == s).sum() / len(labels), 4) for s in sorted(set(labels))},

        # BACKWARD COMPATIBLE: segment_profiles for existing Feature Crew code
        'segment_profiles': segment_profiles_compat,

        # GLOBAL FEATURE ENGINEERING GUIDANCE
        'global_recommendations': feature_recommendations.get('global_recommendations', {}),

        # PER-SEGMENT FEATURE STRATEGIES (comprehensive new format)
        'segment_feature_strategies': feature_recommendations.get('segment_recommendations', {}),

        # Quick reference: which segments need intermittency features
        'intermittency_segments': [
            seg_id for seg_id, rec in feature_recommendations.get('segment_recommendations', {}).items()
            if rec.get('intermittency_features', False)
        ],

        # Quick reference: which segments need log transform
        'log_transform_segments': [
            seg_id for seg_id, rec in feature_recommendations.get('segment_recommendations', {}).items()
            if rec.get('log_transform', False)
        ],

        # Quick reference: total intermittency percentage (for Feature Crew)
        'intermittency_summary': {
            'n_intermittent_segments': len([
                seg_id for seg_id, rec in feature_recommendations.get('segment_recommendations', {}).items()
                if rec.get('dominant_pattern') in ['intermittent', 'lumpy']
            ]),
            'intermittency_pct': sum(
                profile.get('pct_of_total', 0)
                for seg_id, profile in profiles.items()
                if profile.get('dominant_pattern') in ['intermittent', 'lumpy']
            ) / 100.0,  # As decimal
        },

        # EDA insights that affect feature engineering
        'eda_insights': eda_insights,

        # Clustering info for context
        'clustering_info': {
            'algorithm': clustering_method,
            'features_used': clustering_features,
            'silhouette_score': metrics.silhouette,
            'hybrid_segmentation_used': use_hybrid_segmentation,
        },
    }
    with open(os.path.join(output_dir, 'segmentation_to_feature_context.json'), 'w') as f:
        json.dump(seg_to_feat_context, f, indent=2, default=str)
    logger.info("Created segmentation_to_feature_context.json for Feature Engineering crew")

    # ==========================================================================
    # SEGMENTATION → TRAINING CONTEXT
    # ==========================================================================
    # Contains everything the Training crew needs for segment-aware model training.
    seg_to_train_context = {
        'source': 'segmentation_pipeline_v2',
        'generation_method': 'state_of_the_art_deterministic',

        # Segment summary
        'n_segments': n_segments,
        'segment_labels': segment_labels,
        'segment_distribution': {str(s): round((labels == s).sum() / len(labels), 4) for s in sorted(set(labels))},

        # PER-SEGMENT MODEL STRATEGIES (comprehensive)
        'segment_model_strategy': model_recommendations.get('segment_strategies', {}),

        # GLOBAL TRAINING GUIDANCE
        'global_training_guidance': model_recommendations.get('global_training_guidance', {
            'cross_validation_folds': 5,
            'time_series_gap': 1,
            'early_stopping_rounds': 50,
        }),

        # Allowed model families (from config)
        'allowed_model_families': model_recommendations.get('metadata', {}).get('allowed_families', []),
        'enable_deep_models': model_recommendations.get('metadata', {}).get('enable_deep_models', False),

        # Clustering quality for confidence assessment
        'clustering_quality': {
            'algorithm': clustering_method,
            'silhouette': metrics.silhouette,
            'calinski_harabasz': metrics.calinski_harabasz,
            'davies_bouldin': metrics.davies_bouldin,
            'n_segments': n_segments,
            'adaptive_features_used': use_adaptive_features and bool(eda_insights),
        },

        # Quick reference: segments by difficulty
        'segments_by_difficulty': {
            'easy': [seg_id for seg_id, strat in model_recommendations.get('segment_strategies', {}).items()
                     if strat.get('dominant_pattern') == 'smooth'],
            'medium': [seg_id for seg_id, strat in model_recommendations.get('segment_strategies', {}).items()
                       if strat.get('dominant_pattern') in ['erratic', 'intermittent']],
            'hard': [seg_id for seg_id, strat in model_recommendations.get('segment_strategies', {}).items()
                     if strat.get('dominant_pattern') == 'lumpy'],
        },

        # Quick reference: segments needing tweedie loss
        'tweedie_loss_segments': [
            seg_id for seg_id, strat in model_recommendations.get('segment_strategies', {}).items()
            if strat.get('loss_config', {}).get('primary_loss') == 'tweedie'
        ],

        # EDA insights for training decisions
        'eda_insights': eda_insights,
    }
    with open(os.path.join(output_dir, 'segmentation_to_training_context.json'), 'w') as f:
        json.dump(seg_to_train_context, f, indent=2, default=str)
    logger.info("Created segmentation_to_training_context.json for Training crew")

    # -------------------------------------------------------------------------
    # STEP 13: Create visualizations
    # -------------------------------------------------------------------------
    visualization_files = []
    visualization_errors = []
    if create_visualizations:
        logger.info("Creating segmentation visualizations...")

        # Ensure matplotlib is configured for non-interactive use
        try:
            setup_matplotlib()
        except Exception as e:
            logger.warning(f"Could not setup matplotlib: {e}")

        # Create visualizations directly (not using lambdas to avoid closure issues)
        # Each visualization is in its own try-except block

        # 1. Segment Distribution
        try:
            dist_path = os.path.join(output_dir, 'segment_distribution.png')
            plot_segment_distribution(
                labels=labels,
                output_path=dist_path,
                segment_labels=segment_labels,
                min_pct_line=min_segment_pct
            )
            visualization_files.append('segment_distribution.png')
            logger.info("  Created: segment_distribution.png")
        except Exception as e:
            import traceback
            error_msg = f"segment_distribution.png: {str(e)}"
            visualization_errors.append(error_msg)
            logger.error(f"  FAILED to create segment_distribution.png: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")

        # 2. Segment Heatmap
        try:
            heatmap_path = os.path.join(output_dir, 'segment_heatmap.png')
            features_for_heatmap = clustering_features  # Show ALL features used in clustering (including external)
            heatmap_width = max(14, len(features_for_heatmap) * 1.6)
            plot_segment_heatmap(
                profiles=profiles,
                output_path=heatmap_path,
                feature_cols=features_for_heatmap,
                figsize=(heatmap_width, 8),
            )
            visualization_files.append('segment_heatmap.png')
            logger.info("  Created: segment_heatmap.png")
        except Exception as e:
            import traceback
            error_msg = f"segment_heatmap.png: {str(e)}"
            visualization_errors.append(error_msg)
            logger.error(f"  FAILED to create segment_heatmap.png: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")

        # 3. Segment Scatter PCA
        try:
            scatter_path = os.path.join(output_dir, 'segment_scatter_pca.png')
            plot_segment_scatter_2d(
                X_scaled=X_scaled,
                labels=labels,
                output_path=scatter_path,
                segment_labels=segment_labels,
                method='PCA'
            )
            visualization_files.append('segment_scatter_pca.png')
            logger.info("  Created: segment_scatter_pca.png")
        except Exception as e:
            import traceback
            error_msg = f"segment_scatter_pca.png: {str(e)}"
            visualization_errors.append(error_msg)
            logger.error(f"  FAILED to create segment_scatter_pca.png: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")

        # 4. Demand Pattern Distribution (optional - only if demand_pattern column exists)
        if 'demand_pattern' in df.columns:
            try:
                pattern_path = os.path.join(output_dir, 'demand_pattern_by_segment.png')
                plot_demand_pattern_distribution(
                    df=df,
                    labels=labels,
                    output_path=pattern_path
                )
                visualization_files.append('demand_pattern_by_segment.png')
                logger.info("  Created: demand_pattern_by_segment.png")
            except Exception as e:
                import traceback
                error_msg = f"demand_pattern_by_segment.png: {str(e)}"
                visualization_errors.append(error_msg)
                logger.error(f"  FAILED to create demand_pattern_by_segment.png: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")

        # 5. Silhouette Analysis (only if more than 1 segment)
        if n_segments > 1:
            try:
                silhouette_path = os.path.join(output_dir, 'silhouette_analysis.png')
                plot_silhouette_analysis(
                    X_scaled=X_scaled,
                    labels=labels,
                    output_path=silhouette_path
                )
                visualization_files.append('silhouette_analysis.png')
                logger.info("  Created: silhouette_analysis.png")
            except Exception as e:
                import traceback
                error_msg = f"silhouette_analysis.png: {str(e)}"
                visualization_errors.append(error_msg)
                logger.error(f"  FAILED to create silhouette_analysis.png: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")

        total_viz_attempted = 3 + (1 if 'demand_pattern' in df.columns else 0) + (1 if n_segments > 1 else 0)
        if visualization_files:
            logger.info(f"Created {len(visualization_files)}/{total_viz_attempted} visualizations successfully")
        else:
            logger.error("NO visualizations were created successfully!")
            if visualization_errors:
                logger.error(f"Visualization errors: {visualization_errors}")

    # -------------------------------------------------------------------------
    # Build result object
    # -------------------------------------------------------------------------
    seg_dist = {str(s): round((labels == s).sum() / len(labels), 4) for s in sorted(set(labels))}

    # -------------------------------------------------------------------------
    # STEP 14: Save clustering_metrics.json (after visualizations so we can include viz list)
    # -------------------------------------------------------------------------
    clustering_summary = {
        'algorithm': clustering_method,
        'n_segments': n_segments,
        'silhouette': metrics.silhouette,
        'calinski_harabasz': metrics.calinski_harabasz,
        'davies_bouldin': metrics.davies_bouldin,
        'features_used': clustering_features,
        'segment_distribution': {str(s): round((labels == s).sum() / len(labels) * 100, 2)
                                  for s in sorted(set(labels))},
        'segment_labels': segment_labels,
        'eda_insights': eda_insights,  # Store EDA insights for downstream context files
        'adaptive_features_used': use_adaptive_features and bool(eda_insights),
        'visualizations_created': visualization_files,  # Track what visualizations were created
    }
    with open(os.path.join(output_dir, 'clustering_metrics.json'), 'w') as f:
        json.dump(clustering_summary, f, indent=2)
    logger.info(f"Saved clustering_metrics.json with {n_segments} segments")

    # -------------------------------------------------------------------------
    # Build result object
    # -------------------------------------------------------------------------
    result = SegmentationResult(
        labels=labels,
        n_segments=n_segments,
        segment_profiles=profiles,
        segment_distribution=seg_dist,
        segment_labels=segment_labels,
        clustering_metrics={
            'silhouette': metrics.silhouette,
            'calinski_harabasz': metrics.calinski_harabasz,
            'davies_bouldin': metrics.davies_bouldin,
        },
        model_recommendations=model_recommendations,
        feature_recommendations=feature_recommendations,
        scaler=scaler,
        clusterer=clusterer,
        segmentation_features_used=clustering_features,
    )

    logger.info(f"Segmentation complete: {n_segments} segments created")
    logger.info(f"Segment distribution: {seg_dist}")
    logger.info(f"Segment labels: {segment_labels}")

    return result


# =============================================================================
# 9. UTILITY EXPORTS
# =============================================================================

# For backwards compatibility, export key items at module level
__all__ = [
    # Constants
    'ADI_THRESHOLD',
    'CV2_THRESHOLD',
    'DEFAULT_MIN_SEGMENT_PCT',
    'MIN_SEGMENT_SIZE_CAP',

    # Data classes
    'ClusteringMetrics',
    'SegmentationResult',

    # Feature engineering
    'create_segmentation_features',
    'enrich_with_eda_insights',  # NEW: Enrich with exhaustive EDA analysis
    'classify_demand_pattern',
    'get_recommended_clustering_features',
    'filter_low_variance_features',  # NEW: Filter out constant/low-variance features

    # Scaling
    'scale_features',

    # Clustering algorithms
    'cluster_kmeans',
    'cluster_gaussian_mixture',
    'cluster_hdbscan',
    'cluster_hierarchical_optimal',

    # Segment size enforcement
    'enforce_minimum_segment_size',
    'validate_segment_sizes',

    # Hybrid segmentation
    'create_hybrid_segments',

    # Profiling
    'compute_segment_profiles',
    'label_segments_automatically',

    # Model recommendations
    'assign_model_recommendations',
    'create_feature_recommendations',

    # Visualization
    'plot_segment_distribution',
    'plot_segment_heatmap',
    'plot_segment_scatter_2d',
    'plot_demand_pattern_distribution',
    'plot_silhouette_analysis',

    # High-level pipeline
    'run_segmentation_pipeline',
]
