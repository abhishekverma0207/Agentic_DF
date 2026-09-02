"""
Validation-Learned Bias Calibration Module.

This module implements a data-driven bias calibration system that:
1. Learns calibration factors from validation period using recursive forecasting
2. Groups keys by (model_level, zero_fraction_bucket) for finer-grained calibration
3. Applies learned calibration factors during production inference

The system is completely generic and adapts to any dataset - no hard-coded thresholds.
Calibration factors are computed as: sum(actual) / sum(predicted) for each bucket.

Usage:
    from utils.bias_calibration import (
        learn_bias_calibration,
        apply_bias_calibration,
        load_calibration_factors,
        save_calibration_factors,
    )

    # Learn calibration from validation
    calibration = learn_bias_calibration(
        val_predictions_df,
        key_metadata_df,
        n_buckets=5,
    )

    # Apply to new predictions
    calibrated_preds = apply_bias_calibration(
        predictions_df,
        calibration,
        key_metadata_df,
    )

Author: FEU-Agentic-Forecasting Demand Forecasting System
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BiasCalibration:
    """
    Container for learned bias calibration factors.

    Attributes:
        enabled: Whether calibration is enabled
        method: Calibration method used (e.g., 'segment_x_zero_fraction_bucket')
        n_buckets: Number of zero_fraction buckets
        bucket_edges: Percentile edges for zero_fraction buckets [0.0, q1, q2, ..., 1.0]
        factors: Dict mapping 'segment_zf_bucket' -> calibration_factor
        fallback_factors: Dict mapping 'segment' -> factor (used when bucket has no data)
        global_factor: Global fallback factor
        learned_from: Data source ('validation')
        val_period: Validation period range string
        stats: Additional statistics about the calibration
        factor_min: Minimum allowed calibration factor (prevents over-correction downward)
        factor_max: Maximum allowed calibration factor (prevents blow-up)
    """
    enabled: bool = True
    method: str = 'segment_x_zero_fraction_bucket'
    n_buckets: int = 5
    bucket_edges: List[float] = field(default_factory=list)
    factors: Dict[str, float] = field(default_factory=dict)
    fallback_factors: Dict[str, float] = field(default_factory=dict)
    global_factor: float = 1.0
    learned_from: str = 'validation'
    val_period: str = ''
    stats: Dict[str, Any] = field(default_factory=dict)
    # Configurable bounds to prevent extreme corrections
    factor_min: float = 0.2  # Don't reduce predictions by more than 80%
    factor_max: float = 2.0  # Don't increase predictions by more than 2x

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasCalibration':
        """Create from dictionary."""
        # Handle older saved calibrations that don't have factor_min/factor_max
        if 'factor_min' not in data:
            data['factor_min'] = 0.2
        if 'factor_max' not in data:
            data['factor_max'] = 2.0
        return cls(**data)

    def get_factor(self, segment: str, zf_bucket: int) -> float:
        """
        Get calibration factor for a (segment, zf_bucket) combination.
        Falls back to segment factor, then global factor.

        IMPORTANT: All returned factors are clipped to [factor_min, factor_max]
        to prevent extreme corrections that could blow up predictions.
        """
        # Try exact match
        key = f"{segment}_zf_{zf_bucket}"
        if key in self.factors:
            factor = self.factors[key]
        # Fallback to segment factor
        elif segment in self.fallback_factors:
            factor = self.fallback_factors[segment]
        # Global fallback
        else:
            factor = self.global_factor

        # CRITICAL: Always clip to prevent extreme corrections
        return max(self.factor_min, min(self.factor_max, factor))


# =============================================================================
# BUCKET ASSIGNMENT
# =============================================================================

def compute_zero_fraction_buckets(
    zero_fractions: np.ndarray,
    n_buckets: int = 5,
) -> Tuple[np.ndarray, List[float]]:
    """
    Compute zero_fraction bucket assignments using percentile-based edges.

    Parameters
    ----------
    zero_fractions : np.ndarray
        Array of zero_fraction values for each key (0.0 to 1.0)
    n_buckets : int
        Number of buckets to create

    Returns
    -------
    bucket_assignments : np.ndarray
        Bucket index (0 to n_buckets-1) for each key
    bucket_edges : List[float]
        Percentile edges used [0.0, edge1, edge2, ..., 1.0]
    """
    # Compute percentile edges
    percentiles = np.linspace(0, 100, n_buckets + 1)
    bucket_edges = [0.0] + [np.percentile(zero_fractions, p) for p in percentiles[1:-1]] + [1.0]

    # Ensure edges are monotonically increasing (handle ties)
    for i in range(1, len(bucket_edges)):
        if bucket_edges[i] <= bucket_edges[i-1]:
            bucket_edges[i] = bucket_edges[i-1] + 1e-10

    # Assign buckets using digitize (returns 1-indexed, we want 0-indexed)
    bucket_assignments = np.digitize(zero_fractions, bucket_edges[1:-1], right=True)

    # Clip to valid range
    bucket_assignments = np.clip(bucket_assignments, 0, n_buckets - 1)

    return bucket_assignments, bucket_edges


def assign_bucket(zero_fraction: float, bucket_edges: List[float]) -> int:
    """
    Assign a single zero_fraction value to a bucket.

    Parameters
    ----------
    zero_fraction : float
        Zero fraction value (0.0 to 1.0)
    bucket_edges : List[float]
        Bucket edges from compute_zero_fraction_buckets

    Returns
    -------
    bucket : int
        Bucket index (0 to n_buckets-1)
    """
    n_buckets = len(bucket_edges) - 1
    for i in range(1, len(bucket_edges)):
        if zero_fraction <= bucket_edges[i]:
            return min(i - 1, n_buckets - 1)
    return n_buckets - 1


# =============================================================================
# CALIBRATION LEARNING
# =============================================================================

def learn_bias_calibration(
    val_predictions_df: pd.DataFrame,
    key_metadata_df: pd.DataFrame,
    n_buckets: int = 5,
    pred_col: str = 'predicted',
    actual_col: str = 'actual',
    key_col: str = 'key',
    model_level_col: str = 'model_level',
    zero_fraction_col: str = 'zero_fraction',
    segment_col: str = 'segment_id',
    min_samples_per_bucket: int = 10,
    val_period: str = '',
    factor_min: float = 0.2,
    factor_max: float = 2.0,
) -> BiasCalibration:
    """
    Learn bias calibration factors from validation predictions.

    This function:
    1. Computes zero_fraction bucket edges from key metadata
    2. Assigns each key to a (segment, zf_bucket) group - ALWAYS uses segment, not model_level
    3. Computes calibration factor = sum(actual) / sum(predicted) per group
    4. Computes fallback factors at segment level and global level
    5. Clips all factors to [factor_min, factor_max] to prevent extreme corrections

    IMPORTANT: Calibration is ALWAYS computed at segment level (not model_level) because:
    - model_level can be individual keys for some keys (key-level models)
    - Key-level calibration with only 7 validation periods is statistically unreliable
    - Segment-level calibration provides more robust factors with sufficient samples

    Parameters
    ----------
    val_predictions_df : pd.DataFrame
        Validation predictions with columns: key, predicted, actual, model_level
    key_metadata_df : pd.DataFrame
        Key metadata with columns: key, zero_fraction, segment_id
        (typically from per_key_with_segments.csv)
    n_buckets : int
        Number of zero_fraction buckets
    pred_col : str
        Column name for predictions
    actual_col : str
        Column name for actuals
    key_col : str
        Column name for key identifier
    model_level_col : str
        Column name for model level/group (used for logging only, not for grouping)
    zero_fraction_col : str
        Column name for zero fraction in metadata
    segment_col : str
        Column name for segment/cluster ID in metadata (USED FOR CALIBRATION GROUPING)
    min_samples_per_bucket : int
        Minimum samples required to compute bucket-specific factor
    val_period : str
        Validation period description (for metadata)
    factor_min : float
        Minimum allowed calibration factor (default 0.2 = reduce by max 80%)
    factor_max : float
        Maximum allowed calibration factor (default 2.0 = increase by max 2x)

    Returns
    -------
    BiasCalibration
        Learned calibration factors (keyed by segment, not model_level)
    """
    logger.info(f"Learning bias calibration from {len(val_predictions_df)} validation predictions")
    logger.info(f"Calibration will be grouped by segment_id (not model_level) for robustness")

    # Validate inputs
    if val_predictions_df.empty:
        logger.warning("Empty validation predictions, returning default calibration")
        return BiasCalibration(enabled=False, stats={'error': 'empty_predictions'})

    required_cols = [key_col, pred_col, actual_col]
    missing = [c for c in required_cols if c not in val_predictions_df.columns]
    if missing:
        logger.warning(f"Missing columns in predictions: {missing}")
        return BiasCalibration(enabled=False, stats={'error': f'missing_columns: {missing}'})

    if zero_fraction_col not in key_metadata_df.columns:
        logger.warning(f"Missing {zero_fraction_col} in key metadata")
        return BiasCalibration(enabled=False, stats={'error': f'missing {zero_fraction_col}'})

    # Check for segment_id in metadata - REQUIRED for robust calibration
    if segment_col not in key_metadata_df.columns:
        logger.warning(f"Missing {segment_col} in key metadata - calibration may be less robust")
        # Fall back to model_level from predictions if segment not available
        segment_col_to_use = model_level_col
        use_segment_from_metadata = False
    else:
        segment_col_to_use = segment_col
        use_segment_from_metadata = True
        logger.info(f"Using {segment_col} from metadata for calibration grouping")

    # Create key-to-metadata mappings
    key_to_zf = dict(zip(
        key_metadata_df[key_col].astype(str),
        key_metadata_df[zero_fraction_col].fillna(0).values
    ))

    # Create key-to-segment mapping from metadata
    if use_segment_from_metadata:
        key_to_segment = dict(zip(
            key_metadata_df[key_col].astype(str),
            key_metadata_df[segment_col].astype(str).values
        ))
    else:
        key_to_segment = {}

    # Get zero_fraction for each key in predictions
    unique_keys = val_predictions_df[key_col].astype(str).unique()
    key_zero_fractions = np.array([key_to_zf.get(k, 0.5) for k in unique_keys])

    # Compute bucket edges and assignments
    bucket_assignments, bucket_edges = compute_zero_fraction_buckets(
        key_zero_fractions, n_buckets
    )

    # Create key -> bucket mapping
    key_to_bucket = dict(zip(unique_keys, bucket_assignments))

    logger.info(f"Bucket edges: {[f'{e:.3f}' for e in bucket_edges]}")

    # Add bucket and segment assignment to predictions
    preds_df = val_predictions_df.copy()
    preds_df['zf_bucket'] = preds_df[key_col].astype(str).map(key_to_bucket).fillna(n_buckets // 2).astype(int)
    preds_df['zero_fraction'] = preds_df[key_col].astype(str).map(key_to_zf).fillna(0.5)

    # CRITICAL: Add segment_id for grouping (from metadata, not model_level)
    if use_segment_from_metadata:
        preds_df['calibration_segment'] = preds_df[key_col].astype(str).map(key_to_segment).fillna('default')
    elif model_level_col in preds_df.columns:
        # Fallback: use model_level but warn about potential key-level issues
        preds_df['calibration_segment'] = preds_df[model_level_col].astype(str)
        logger.warning("Using model_level for calibration - may include key-level models with few samples")
    else:
        preds_df['calibration_segment'] = 'default'

    # Log segment distribution
    segment_counts = preds_df['calibration_segment'].value_counts()
    logger.info(f"Calibration segments: {dict(segment_counts)}")

    # Compute calibration factors per (segment, zf_bucket) - NOT model_level!
    factors = {}
    factor_stats = {}

    logger.info(f"Calibration factor bounds: [{factor_min}, {factor_max}]")

    for (seg, bucket), group_df in preds_df.groupby(['calibration_segment', 'zf_bucket']):
        seg_str = str(seg)
        bucket_int = int(bucket)
        key = f"{seg_str}_zf_{bucket_int}"

        sum_actual = group_df[actual_col].sum()
        sum_pred = group_df[pred_col].sum()
        n_samples = len(group_df)
        n_keys = group_df[key_col].nunique()

        if n_samples >= min_samples_per_bucket and sum_pred > 1e-10:
            raw_factor = sum_actual / sum_pred
            # Clip to configurable range to avoid extreme corrections
            factor = np.clip(raw_factor, factor_min, factor_max)
            factors[key] = float(factor)
            factor_stats[key] = {
                'n_samples': int(n_samples),
                'n_keys': int(n_keys),
                'sum_actual': float(sum_actual),
                'sum_pred': float(sum_pred),
                'raw_factor': float(raw_factor),  # Before clipping
                'factor': float(factor),  # After clipping
                'was_clipped': bool(raw_factor != factor),  # Convert numpy bool to Python bool
                'mean_zero_fraction': float(group_df['zero_fraction'].mean()),
            }
            logger.debug(f"  {key}: factor={factor:.3f} (n={n_samples}, keys={n_keys}, sum_a={sum_actual:.0f}, sum_p={sum_pred:.0f})")
        else:
            logger.debug(f"  {key}: skipped (n={n_samples}, sum_pred={sum_pred:.0f})")

    # Compute fallback factors at segment level (not model_level)
    fallback_factors = {}
    for seg, seg_df in preds_df.groupby('calibration_segment'):
        seg_str = str(seg)
        sum_actual = seg_df[actual_col].sum()
        sum_pred = seg_df[pred_col].sum()

        if sum_pred > 1e-10:
            factor = np.clip(sum_actual / sum_pred, factor_min, factor_max)
            fallback_factors[seg_str] = float(factor)

    # Compute global factor
    global_sum_actual = preds_df[actual_col].sum()
    global_sum_pred = preds_df[pred_col].sum()
    global_factor = 1.0
    if global_sum_pred > 1e-10:
        global_factor = np.clip(global_sum_actual / global_sum_pred, factor_min, factor_max)

    # Count how many factors were clipped
    n_clipped = sum(1 for s in factor_stats.values() if s.get('was_clipped', False))

    # Compute summary statistics - ensure all values are native Python types for JSON serialization
    stats = {
        'n_predictions': int(len(preds_df)),
        'n_keys': int(len(unique_keys)),
        'n_factors_learned': int(len(factors)),
        'n_fallback_factors': int(len(fallback_factors)),
        'n_factors_clipped': int(n_clipped),
        'factor_bounds': {'min': float(factor_min), 'max': float(factor_max)},
        'global_factor': float(global_factor),
        'global_sum_actual': float(global_sum_actual),
        'global_sum_pred': float(global_sum_pred),
        'global_bias_pct': float((global_sum_pred - global_sum_actual) / max(global_sum_actual, 1) * 100),
        'bucket_stats': factor_stats,
        'bucket_distribution': {
            f'zf_{i}': int((preds_df['zf_bucket'] == i).sum())
            for i in range(n_buckets)
        },
    }

    logger.info(f"Learned {len(factors)} calibration factors, {len(fallback_factors)} fallbacks")
    logger.info(f"Factors clipped to bounds: {n_clipped}")
    logger.info(f"Global factor: {global_factor:.3f} (bias: {stats['global_bias_pct']:.1f}%)")

    return BiasCalibration(
        enabled=True,
        method='segment_x_zero_fraction_bucket',
        n_buckets=n_buckets,
        bucket_edges=bucket_edges,
        factors=factors,
        fallback_factors=fallback_factors,
        global_factor=global_factor,
        learned_from='validation',
        val_period=val_period,
        stats=stats,
        factor_min=factor_min,
        factor_max=factor_max,
    )


# =============================================================================
# CALIBRATION APPLICATION
# =============================================================================

def apply_bias_calibration(
    predictions_df: pd.DataFrame,
    calibration: BiasCalibration,
    key_metadata_df: pd.DataFrame,
    pred_col: str = 'predicted',
    key_col: str = 'key',
    model_level_col: str = 'model_level',
    zero_fraction_col: str = 'zero_fraction',
    segment_col: str = 'segment_id',
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply learned bias calibration factors to predictions.

    IMPORTANT: Calibration factors are looked up by SEGMENT (not model_level) because:
    - model_level can be individual keys for some keys (key-level models)
    - Calibration was learned at segment level for statistical robustness
    - segment_id is obtained from key_metadata_df

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions to calibrate (must have key, predicted columns)
    calibration : BiasCalibration
        Learned calibration factors (keyed by segment_id, not model_level)
    key_metadata_df : pd.DataFrame
        Key metadata with zero_fraction and segment_id columns
    pred_col : str
        Column name for predictions
    key_col : str
        Column name for key identifier
    model_level_col : str
        Column name for model level (NOT used for factor lookup - only for logging)
    zero_fraction_col : str
        Column name for zero fraction in metadata
    segment_col : str
        Column name for segment/cluster ID in metadata (USED for factor lookup)
    inplace : bool
        If True, modify predictions_df in place

    Returns
    -------
    pd.DataFrame
        Predictions with calibration applied
    """
    if not calibration.enabled:
        logger.info("Bias calibration disabled, returning predictions unchanged")
        return predictions_df if inplace else predictions_df.copy()

    if not inplace:
        predictions_df = predictions_df.copy()

    # Create key-to-metadata mappings
    key_to_zf = dict(zip(
        key_metadata_df[key_col].astype(str),
        key_metadata_df[zero_fraction_col].fillna(0).values
    ))

    # Create key-to-segment mapping (CRITICAL for correct factor lookup)
    if segment_col in key_metadata_df.columns:
        key_to_segment = dict(zip(
            key_metadata_df[key_col].astype(str),
            key_metadata_df[segment_col].astype(str).values
        ))
    else:
        # Fallback: use model_level from predictions (may include key-level)
        logger.warning(f"segment_col '{segment_col}' not found in metadata, using model_level")
        key_to_segment = {}

    # Compute bucket for each key and apply calibration
    calibrated_preds = []
    factors_applied = []

    for idx, row in predictions_df.iterrows():
        key = str(row[key_col])
        pred = row[pred_col]

        # Get segment for this key (from metadata, not model_level)
        if key in key_to_segment:
            segment = key_to_segment[key]
        elif model_level_col in predictions_df.columns:
            segment = str(row[model_level_col])
        else:
            segment = 'default'

        # Get zero_fraction and bucket
        zf = key_to_zf.get(key, 0.5)
        bucket = assign_bucket(zf, calibration.bucket_edges)

        # Get calibration factor using SEGMENT (not model_level)
        factor = calibration.get_factor(segment, bucket)

        # Apply calibration
        calibrated_pred = max(0.0, pred * factor)

        calibrated_preds.append(calibrated_pred)
        factors_applied.append(factor)

    # Update predictions
    predictions_df[pred_col] = calibrated_preds
    predictions_df['calibration_factor'] = factors_applied

    logger.info(f"Applied calibration to {len(predictions_df)} predictions")
    logger.info(f"Factor range: [{min(factors_applied):.3f}, {max(factors_applied):.3f}]")

    return predictions_df


def apply_bias_calibration_fast(
    predictions_df: pd.DataFrame,
    calibration: BiasCalibration,
    key_to_zf: Dict[str, float],
    key_to_segment: Optional[Dict[str, str]] = None,
    pred_col: str = 'predicted',
    key_col: str = 'key',
    model_level_col: str = 'model_level',
) -> pd.DataFrame:
    """
    Vectorized version of apply_bias_calibration for better performance.

    IMPORTANT: Uses key_to_segment mapping to look up calibration factors by segment,
    not by model_level. This ensures key-level models still use segment-level calibration.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions to calibrate
    calibration : BiasCalibration
        Learned calibration factors (keyed by segment)
    key_to_zf : Dict[str, float]
        Pre-computed key -> zero_fraction mapping
    key_to_segment : Dict[str, str], optional
        Pre-computed key -> segment_id mapping. If None, falls back to model_level.
    pred_col : str
        Column name for predictions
    key_col : str
        Column name for key identifier
    model_level_col : str
        Column name for model level (used as fallback if key_to_segment not provided)

    Returns
    -------
    pd.DataFrame
        Predictions with calibration applied (copy)
    """
    if not calibration.enabled:
        return predictions_df.copy()

    df = predictions_df.copy()

    # Vectorized zero_fraction lookup
    df['_zf'] = df[key_col].astype(str).map(key_to_zf).fillna(0.5)

    # Vectorized segment lookup (from key_to_segment mapping, not model_level)
    if key_to_segment:
        df['_segment'] = df[key_col].astype(str).map(key_to_segment)
        # Fallback to model_level if key not found in segment mapping
        if model_level_col in df.columns:
            df['_segment'] = df['_segment'].fillna(df[model_level_col].astype(str))
        else:
            df['_segment'] = df['_segment'].fillna('default')
    elif model_level_col in df.columns:
        df['_segment'] = df[model_level_col].astype(str)
    else:
        df['_segment'] = 'default'

    # Vectorized bucket assignment
    df['_bucket'] = df['_zf'].apply(lambda zf: assign_bucket(zf, calibration.bucket_edges))

    # Vectorized factor lookup using SEGMENT (not model_level)
    def get_factor(row):
        return calibration.get_factor(str(row['_segment']), int(row['_bucket']))

    df['calibration_factor'] = df.apply(get_factor, axis=1)

    # Apply calibration
    df[pred_col] = (df[pred_col] * df['calibration_factor']).clip(lower=0)

    # Clean up temporary columns
    df.drop(columns=['_zf', '_bucket', '_segment'], inplace=True)

    return df


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_calibration_factors(
    calibration: BiasCalibration,
    output_path: str,
) -> None:
    """
    Save calibration factors to JSON file.

    Parameters
    ----------
    calibration : BiasCalibration
        Calibration to save
    output_path : str
        Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(calibration.to_dict(), f, indent=2)

    logger.info(f"Saved calibration factors to {output_path}")


def load_calibration_factors(
    input_path: str,
) -> BiasCalibration:
    """
    Load calibration factors from JSON file.

    Parameters
    ----------
    input_path : str
        Input file path

    Returns
    -------
    BiasCalibration
        Loaded calibration factors
    """
    if not os.path.exists(input_path):
        logger.warning(f"Calibration file not found: {input_path}")
        return BiasCalibration(enabled=False, stats={'error': 'file_not_found'})

    with open(input_path, 'r') as f:
        data = json.load(f)

    calibration = BiasCalibration.from_dict(data)
    logger.info(f"Loaded calibration factors from {input_path}")
    logger.info(f"  {len(calibration.factors)} factors, global={calibration.global_factor:.3f}")

    return calibration


# =============================================================================
# VALIDATION PREDICTION GENERATION (FOR CALIBRATION LEARNING)
# =============================================================================

def generate_validation_predictions(
    trained_models: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    key_col: str = 'key',
    date_col: str = None,
    model_level_col: str = 'model_level',
    key_to_model_group: Optional[Dict[str, str]] = None,
    forecast_horizon: int = 8,
    model_feature_columns: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Generate recursive predictions on validation data for calibration learning.

    This function computes calibration factors from validation data:
    - Uses recursive forecasting to avoid feature leakage
    - Returns predictions with key, actual, predicted, model_level

    Parameters
    ----------
    trained_models : Dict[str, Any]
        Mapping of model_level -> trained model (or model artifact dict)
    train_df : pd.DataFrame
        Training data (for historical actuals)
    val_df : pd.DataFrame
        Validation data to generate predictions for
    feature_columns : List[str]
        Fallback feature columns if model-specific not available
    target_col : str
        Target column name
    key_col : str
        Key column name
    date_col : str
        Date column name
    model_level_col : str
        Model level column name
    key_to_model_group : Dict[str, str], optional
        Mapping of key -> model_group. If None, uses model_level from data.
    forecast_horizon : int
        Maximum forecast horizon
    model_feature_columns : Dict[str, List[str]], optional
        Per-model feature columns. Keys are model_group names, values are
        the feature columns that model was trained with. This is CRITICAL
        for correct predictions since each model may have different features.

    Returns
    -------
    pd.DataFrame
        Predictions with columns: key, {date_col}, predicted, actual, model_level
    """
    import re

    # Resolve date_col: try to detect from DataFrame columns if not provided
    if date_col is None:
        if 'year_month' in val_df.columns:
            date_col = 'year_month'
        elif 'year_week' in val_df.columns:
            date_col = 'year_week'
        else:
            date_col = 'year_week'

    logger.info(f"Generating validation predictions for calibration...")
    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    logger.info(f"Models available: {list(trained_models.keys())}")
    if model_feature_columns:
        logger.info(f"Per-model feature columns provided for {len(model_feature_columns)} models")

    # Identify feature types for recursive updating
    def identify_lag_cols(cols):
        """
        Identify lag columns and whether they need log transformation.
        Returns list of (col_name, lag_num, is_log_transformed, is_binary_indicator).

        Handles:
        - Pure lags: PS_actual_sales_lag_1 -> raw value
        - Log lags: PS_actual_sales_log_lag_1 -> log1p(raw value)
        - Seasonal lags: PS_actual_sales_log_seasonal_lag_52 -> log1p(raw value)
        - Binary indicators: PS_actual_sales_lag1_demand_occurred -> 1 if lag > 0 else 0
        """
        lag_cols = []
        # Patterns to exclude (derived features - handled separately)
        derived_patterns = ['lag_diff', 'lag_ratio', 'lag_pct_change', 'lag_direction', 'yoy_lag', 'qoq_lag']

        for col in cols:
            col_lower = col.lower()

            # Skip derived features (handled by identify_derived_lag_cols)
            if any(p in col_lower for p in derived_patterns):
                continue

            # Check for binary indicator pattern (e.g., lag1_demand_occurred)
            is_binary = 'demand_occurred' in col_lower

            # Match lag number
            match = re.search(r'lag[_]?(\d+)(?:[_]|$)', col, re.IGNORECASE)
            if match:
                lag_num = int(match.group(1))
                # Check if this is a log-transformed lag feature
                # Includes: _log_lag, log_lag, log_seasonal_lag
                is_log = '_log_' in col_lower and 'lag' in col_lower
                lag_cols.append((col, lag_num, is_log, is_binary))

        return lag_cols

    def identify_derived_lag_cols(cols):
        """
        Identify derived lag columns that need to be computed from base lags.
        Returns list of (col_name, lag_num, derived_type) where derived_type is:
        - 'diff': lag_N - lag_(N+1)
        - 'pct_change': (lag_N - lag_(N+1)) / lag_(N+1)
        - 'direction': sign(lag_N - lag_(N+1))

        These features are computed from the difference between consecutive lags
        and need to be recomputed during recursive forecasting.
        """
        derived_cols = []

        for col in cols:
            col_lower = col.lower()

            # Match lag_diff_N pattern
            match_diff = re.search(r'lag[_]?diff[_]?(\d+)', col_lower)
            if match_diff:
                lag_num = int(match_diff.group(1))
                derived_cols.append((col, lag_num, 'diff'))
                continue

            # Match lag_pct_change_N pattern
            match_pct = re.search(r'lag[_]?pct[_]?change[_]?(\d+)', col_lower)
            if match_pct:
                lag_num = int(match_pct.group(1))
                derived_cols.append((col, lag_num, 'pct_change'))
                continue

            # Match lag_direction_N pattern
            match_dir = re.search(r'lag[_]?direction[_]?(\d+)', col_lower)
            if match_dir:
                lag_num = int(match_dir.group(1))
                derived_cols.append((col, lag_num, 'direction'))
                continue

        return derived_cols

    def identify_rolling_cols(cols):
        """
        Identify rolling window columns and whether they need log transformation.
        Returns list of (col_name, window, agg, is_log_transformed).

        Handles multiple patterns:
        - roll_N_agg: PS_actual_sales_log_roll_4_mean
        - ewm_N: PS_actual_sales_log_ewm_4
        - ewm_std_N: PS_actual_sales_log_ewm_std_4
        """
        rolling_cols = []

        for col in cols:
            col_lower = col.lower()

            # Check if this is a log-transformed feature
            is_log = '_log_' in col_lower

            # Pattern 1: roll_N_agg (e.g., log_roll_4_mean, log_roll_13_std)
            for agg in ['mean', 'std', 'min', 'max', 'sum', 'median']:
                # Match patterns like: roll_4_mean, roll_13_std
                pattern = rf'roll[_]?(\d+)[_]?{agg}'
                match = re.search(pattern, col_lower)
                if match:
                    window = int(match.group(1))
                    rolling_cols.append((col, window, agg, is_log))
                    break

            # Pattern 2: ewm_N (exponential weighted mean)
            # e.g., log_ewm_4, log_ewm_13
            if 'ewm' in col_lower and col not in [c[0] for c in rolling_cols]:
                # Check for ewm_std_N pattern first (e.g., log_ewm_std_4)
                match_std = re.search(r'ewm[_]?std[_]?(\d+)', col_lower)
                if match_std:
                    window = int(match_std.group(1))
                    rolling_cols.append((col, window, 'ewm_std', is_log))
                else:
                    # Regular ewm_N pattern
                    match_ewm = re.search(r'ewm[_]?(\d+)', col_lower)
                    if match_ewm:
                        window = int(match_ewm.group(1))
                        rolling_cols.append((col, window, 'ewm', is_log))

        return rolling_cols

    # Helper functions for recursive feature updates
    def compute_lag_value(vals, lag_num, is_log=False, is_binary=False):
        """Compute lag value, optionally log-transformed or as binary indicator."""
        if len(vals) >= lag_num:
            raw_val = vals[-lag_num]
            if is_binary:
                # Binary indicator: 1 if demand occurred, 0 otherwise
                return 1.0 if raw_val > 0 else 0.0
            if is_log:
                return np.log1p(max(0, raw_val))  # log1p(x) = log(1+x), handles 0 gracefully
            return raw_val
        return 0.0

    def compute_rolling_agg(vals, window, agg, is_log=False):
        """Compute rolling aggregate, optionally on log-transformed values."""
        if len(vals) < 1:
            return 0.0

        window_vals = vals[-min(window, len(vals)):]

        # Apply log transform if needed
        if is_log:
            window_vals = [np.log1p(max(0, v)) for v in window_vals]

        if agg == 'mean' or agg == 'ewm':
            return np.mean(window_vals) if window_vals else 0.0
        elif agg == 'std' or agg == 'ewm_std':
            return np.std(window_vals) if len(window_vals) > 1 else 0.0
        elif agg == 'sum':
            return np.sum(window_vals)
        elif agg == 'min':
            return np.min(window_vals) if window_vals else 0.0
        elif agg == 'max':
            return np.max(window_vals) if window_vals else 0.0
        elif agg == 'median':
            return np.median(window_vals) if window_vals else 0.0
        return np.mean(window_vals) if window_vals else 0.0

    def compute_derived_lag(vals, lag_num, derived_type):
        """
        Compute derived lag features (diff, pct_change, direction).

        These are computed from the difference between lag_N and lag_(N+1).
        For example, lag_diff_1 = lag_1 - lag_2 = vals[-1] - vals[-2]

        The lag_num represents the first lag in the difference:
        - lag_diff_1 = lag_1 - lag_2
        - lag_diff_4 = lag_4 - lag_5
        - etc.
        """
        # Need at least lag_num + 1 values
        # lag_diff_1 needs vals[-1] and vals[-2], so need 2 values
        # lag_diff_4 needs vals[-4] and vals[-5], so need 5 values
        needed = lag_num + 1
        if len(vals) < needed:
            return 0.0

        lag_current = vals[-lag_num]
        lag_prev = vals[-(lag_num + 1)]

        if derived_type == 'diff':
            return lag_current - lag_prev
        elif derived_type == 'pct_change':
            if abs(lag_prev) < 1e-10:
                return 0.0
            return (lag_current - lag_prev) / lag_prev
        elif derived_type == 'direction':
            diff = lag_current - lag_prev
            if diff > 0:
                return 1.0
            elif diff < 0:
                return -1.0
            return 0.0
        return 0.0

    # Get unique keys
    unique_keys = val_df[key_col].unique()
    logger.info(f"Processing {len(unique_keys)} unique keys")

    # Build key-to-model mapping
    if key_to_model_group is None:
        key_to_model_group = {}
        if model_level_col in val_df.columns:
            for _, row in val_df[[key_col, model_level_col]].drop_duplicates().iterrows():
                key_to_model_group[str(row[key_col])] = str(row[model_level_col])

    # Combine train and val for full data access
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    full_df = full_df.sort_values([key_col, date_col])

    # Get validation periods
    val_periods = sorted(val_df[date_col].unique())

    all_predictions = []
    keys_processed = 0
    keys_skipped_no_model = 0
    keys_skipped_no_features = 0
    keys_skipped_no_history = 0

    for key in unique_keys:
        # Get model for this key
        key_str = str(key)
        model_group = key_to_model_group.get(key_str, 'default')

        if model_group not in trained_models:
            # Try to find any available model
            if trained_models:
                model_group = list(trained_models.keys())[0]
            else:
                keys_skipped_no_model += 1
                continue

        model_or_artifact = trained_models[model_group]

        # Extract model and feature_columns from artifact if needed
        if isinstance(model_or_artifact, dict):
            model = model_or_artifact.get('model', model_or_artifact)
            # Use feature columns from artifact if available
            model_feat_cols = model_or_artifact.get('feature_columns', None)
        else:
            model = model_or_artifact
            model_feat_cols = None

        # Priority for feature columns:
        # 1. From model artifact (most reliable)
        # 2. From model_feature_columns dict
        # 3. Fallback to global feature_columns
        if model_feat_cols:
            feat_cols_for_key = model_feat_cols
        elif model_feature_columns and model_group in model_feature_columns:
            feat_cols_for_key = model_feature_columns[model_group]
        else:
            feat_cols_for_key = feature_columns

        # Get full data for this key
        key_full = full_df[full_df[key_col] == key].sort_values(date_col)
        key_val = val_df[val_df[key_col] == key].sort_values(date_col)

        if len(key_val) == 0:
            continue

        # Get training history (up to first validation period)
        first_val_period = key_val[date_col].min()
        key_train = key_full[key_full[date_col] < first_val_period]

        if len(key_train) == 0:
            keys_skipped_no_history += 1
            continue

        # Historical actuals
        historical_actuals = key_train[target_col].values.tolist()

        # Check if model is a univariate dict (croston, sba, tsb, imapa)
        is_univariate_dict = isinstance(model, dict) and 'forecast' in model

        if is_univariate_dict:
            # Use recursive re-estimation for univariate models
            from utils.model_training import _recursive_forecast_univariate
            hist_arr = np.array(historical_actuals, dtype=float)
            n_steps = min(len(key_val), forecast_horizon)
            preds = _recursive_forecast_univariate(model, hist_arr, n_steps)
            for step_idx, (_, row) in enumerate(key_val.iterrows()):
                if step_idx >= n_steps:
                    break
                all_predictions.append({
                    key_col: key,
                    date_col: row[date_col],
                    'predicted': float(preds[step_idx]),
                    'actual': row[target_col],
                    'model_level': model_group,
                })
            keys_processed += 1
            continue

        # Prepare feature columns - use model-specific features
        avail_cols = [c for c in feat_cols_for_key if c in key_val.columns]
        if not avail_cols:
            keys_skipped_no_features += 1
            continue

        # Compute lag/rolling/derived cols for THIS model's feature set
        lag_cols = identify_lag_cols(avail_cols)
        roll_cols = identify_rolling_cols(avail_cols)
        derived_lag_cols = identify_derived_lag_cols(avail_cols)

        col_to_idx = {c: i for i, c in enumerate(avail_cols)}
        base_features = key_val[avail_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        # Generate recursive predictions
        for step_idx, (_, row) in enumerate(key_val.iterrows()):
            if step_idx >= len(base_features) or step_idx >= forecast_horizon:
                break

            # Start with base features
            features = base_features[step_idx].copy()

            # Update lag features (with log transformation and binary indicators where needed)
            for col, lag_num, is_log, is_binary in lag_cols:
                if col in col_to_idx:
                    idx = col_to_idx[col]
                    features[idx] = compute_lag_value(historical_actuals, lag_num, is_log, is_binary)

            # Update derived lag features (diff, pct_change, direction)
            for col, lag_num, derived_type in derived_lag_cols:
                if col in col_to_idx:
                    idx = col_to_idx[col]
                    features[idx] = compute_derived_lag(historical_actuals, lag_num, derived_type)

            # Update rolling features (with log transformation where needed)
            for col, window, agg, is_log in roll_cols:
                if col in col_to_idx:
                    idx = col_to_idx[col]
                    features[idx] = compute_rolling_agg(historical_actuals, window, agg, is_log)

            # Make prediction
            X = features.reshape(1, -1)
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    if isinstance(pred, np.ndarray):
                        pred = pred[0] if len(pred) == 1 else pred.mean()
                else:
                    pred = 0.0
            except Exception as e:
                # Log the exception for first few failures to help debugging
                if keys_processed < 3:
                    logger.warning(f"Prediction failed for key {key_str}: {e}")
                pred = 0.0

            pred = max(0.0, float(pred))

            # Append prediction to history for next step
            historical_actuals.append(pred)

            # Store result
            all_predictions.append({
                key_col: key,
                date_col: row[date_col],
                'predicted': pred,
                'actual': row[target_col],
                'model_level': model_group,
            })

        keys_processed += 1

    predictions_df = pd.DataFrame(all_predictions)
    logger.info(f"Generated {len(predictions_df)} validation predictions from {keys_processed} keys")
    if keys_skipped_no_model > 0:
        logger.warning(f"Skipped {keys_skipped_no_model} keys (no model)")
    if keys_skipped_no_features > 0:
        logger.warning(f"Skipped {keys_skipped_no_features} keys (no features matched)")
    if keys_skipped_no_history > 0:
        logger.warning(f"Skipped {keys_skipped_no_history} keys (no training history)")

    return predictions_df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'BiasCalibration',

    # Bucket computation
    'compute_zero_fraction_buckets',
    'assign_bucket',

    # Learning
    'learn_bias_calibration',

    # Validation prediction generation
    'generate_validation_predictions',

    # Application
    'apply_bias_calibration',
    'apply_bias_calibration_fast',

    # Persistence
    'save_calibration_factors',
    'load_calibration_factors',
]
