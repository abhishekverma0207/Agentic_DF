# utils/inference.py
"""
FEU-Agentic-Forecasting Inference Pipeline - Production Forecasting with New Key Handling

This module provides the production inference pipeline for generating forecasts
for all keys in the test/inference period, with intelligent handling of new keys
(keys present in inference data but not in the original training manifest).

Pipeline Phases:
================
1. DETECT NEW KEYS: Identifies keys in inference period not in training manifest
2. SEGMENT ASSIGNMENT: Computes segmentation features for new keys, assigns to segments
3. MANIFEST UPDATE: Updates training_manifest.csv with new keys
4. FEATURE REGENERATION: Re-runs feature engineering with latest data (if enabled)
   - Controlled by config.design.regenerate_features_on_inference (default: True)
   - Ensures lag features, rolling averages reflect most recent data
   - Falls back to pre-computed features if disabled or on failure
5. MODEL RETRAINING: Retrains all models with data up to val_end
   - Uses existing model_type and hyperparameters from final_model_specs.json
6. FORECAST GENERATION: Recursive multi-step forecasts over inference period
7. BIAS CALIBRATION: Applies validation-learned calibration factors (if enabled)

Configuration:
==============
Key settings in config.yaml -> design section:
- regenerate_features_on_inference: bool (default: True)
  When True, regenerates all features using latest source data before retraining.
  When False, uses pre-computed feature CSVs from training (faster but stale).

- apply_bias_calibration: bool (default: True)
  When True, applies bias correction factors learned from validation period.

Usage:
======
    from utils.inference import run_inference_pipeline, InferenceResult
    from config.schema import load_config_from_yaml

    config = load_config_from_yaml("config/config.yaml")
    result = run_inference_pipeline(config)

    # Check results
    if result.success:
        print(f"Forecasts saved to: {result.forecasts_path}")
        print(f"Total keys: {result.total_keys}")
        print(f"New keys assigned: {result.new_keys_count}")
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config.schema import DemandForecastConfig
from utils.backtesting import increment_period, parse_period, format_period

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InferenceResult:
    """Result container for inference pipeline."""
    success: bool
    forecasts_path: str = ""
    forecasts_parquet_path: str = ""
    summary_path: str = ""
    total_keys: int = 0
    new_keys_count: int = 0
    existing_keys_count: int = 0
    dead_keys_count: int = 0
    new_keys_by_segment: Dict[str, int] = field(default_factory=dict)
    models_retrained: int = 0
    forecast_horizon: int = 0
    total_forecasts: int = 0
    error_message: str = ""
    # Paths to artifacts
    updated_manifest_path: str = ""
    backup_manifest_path: str = ""
    retrained_models_dir: str = ""


@dataclass
class NewKeyAssignment:
    """Assignment info for a new key."""
    key: str
    segment_id: int
    model_level: str  # Will be segment_id (new keys always use segment models)
    intermittency_class: str
    segmentation_features: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# PHASE 1: NEW KEY DETECTION AND DEAD KEY REMOVAL
# =============================================================================

# Default dead key detection parameters
DEFAULT_DEAD_KEY_LOOKBACK = 52  # periods (overridden per time_format at call site)
DEFAULT_DEAD_KEY_THRESHOLD = 0.0


def detect_dead_keys(
    source_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    key_col: str,
    target_col: str,
    date_col: str,
    cutoff_date: str,
    lookback_periods: int = DEFAULT_DEAD_KEY_LOOKBACK,
    threshold: float = DEFAULT_DEAD_KEY_THRESHOLD,
) -> List[str]:
    """
    Detect "dead" keys - manifest keys with no recent demand before cutoff_date.

    These are keys that existed in the original training but have become inactive
    (e.g., discontinued products).

    Parameters
    ----------
    source_df : pd.DataFrame
        Full source data
    manifest_df : pd.DataFrame
        Training manifest with key assignments
    key_col : str
        Key column name
    target_col : str
        Target (demand) column name
    date_col : str
        Date column name
    cutoff_date : str
        End date for dead key detection (typically week before test_start)
    lookback_periods : int
        Number of recent periods to check for activity
    threshold : float
        Sum threshold (at or below this = dead)

    Returns
    -------
    List[str]
        List of dead key values
    """
    # Handle type conversion for date comparison
    date_col_dtype = source_df[date_col].dtype
    if pd.api.types.is_integer_dtype(date_col_dtype):
        cutoff_val = int(cutoff_date)
    else:
        cutoff_val = str(cutoff_date)

    # Filter data up to cutoff
    df_before_cutoff = source_df[source_df[date_col] <= cutoff_val].copy()

    # Get manifest keys only
    manifest_keys = set(manifest_df['key'].unique())
    df_manifest_keys = df_before_cutoff[df_before_cutoff[key_col].isin(manifest_keys)]

    if len(df_manifest_keys) == 0:
        logger.warning("No data found for manifest keys before cutoff date")
        return []

    # Get most recent periods before cutoff
    sorted_dates = sorted(df_manifest_keys[date_col].unique())
    recent_dates = sorted_dates[-lookback_periods:] if len(sorted_dates) >= lookback_periods else sorted_dates

    recent_df = df_manifest_keys[df_manifest_keys[date_col].isin(recent_dates)]

    # Sum demand by key
    key_sums = recent_df.groupby(key_col)[target_col].sum()

    # Keys with demand at or below threshold are dead
    dead_keys = key_sums[key_sums <= threshold].index.tolist()

    # Also check for manifest keys that have NO data at all in the recent period
    keys_with_data = set(recent_df[key_col].unique())
    keys_without_data = manifest_keys - keys_with_data
    dead_keys.extend(list(keys_without_data))

    # Remove duplicates
    dead_keys = list(set(dead_keys))

    logger.info(f"Dead key detection (lookback={lookback_periods}, threshold={threshold}):")
    logger.info(f"  - Keys with zero/low demand: {len(key_sums[key_sums <= threshold])}")
    logger.info(f"  - Keys with no recent data: {len(keys_without_data)}")
    logger.info(f"  - Total dead keys: {len(dead_keys)}")

    return dead_keys


def detect_new_keys(
    source_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    key_col: str,
    target_col: str,
    test_start: str,
    test_end: str,
    date_col: str,
    dead_key_lookback: int = DEFAULT_DEAD_KEY_LOOKBACK,
    dead_key_threshold: float = DEFAULT_DEAD_KEY_THRESHOLD,
) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
    """
    Detect new keys, existing keys, and dead keys for forward forecasting.

    This function:
    1. Detects dead keys (manifest keys with no recent demand before test_start)
    2. Removes dead keys from consideration
    3. Identifies new keys (in test but not in manifest)
    4. Identifies existing keys (in test and in manifest, excluding dead keys)

    Parameters
    ----------
    source_df : pd.DataFrame
        Full source data
    manifest_df : pd.DataFrame
        Training manifest with key assignments
    key_col : str
        Key column name
    target_col : str
        Target (demand) column name
    test_start : str
        Test period start
    test_end : str
        Test period end
    date_col : str
        Date column name
    dead_key_lookback : int
        Number of periods to look back for dead key detection
    dead_key_threshold : float
        Demand sum threshold for dead key detection

    Returns
    -------
    Tuple[List[str], List[str], List[str], pd.DataFrame]
        - new_keys: Keys in test but not in manifest
        - existing_keys: Keys in both test and manifest (excluding dead keys)
        - dead_keys: Keys in manifest but inactive/discontinued
        - test_df: Filtered test period data
    """
    # Handle type conversion for date comparison
    # The date column might be int64 (202529) or string ("202529")
    date_col_dtype = source_df[date_col].dtype

    if pd.api.types.is_integer_dtype(date_col_dtype):
        # Convert string dates to int for comparison
        test_start_val = int(test_start)
        test_end_val = int(test_end)
    else:
        # Keep as string
        test_start_val = str(test_start)
        test_end_val = str(test_end)

    # Step 1: Detect dead keys using data up to the week before test_start
    # Get the week before test_start
    all_dates = sorted(source_df[date_col].unique())
    test_start_idx = None
    for i, d in enumerate(all_dates):
        if d >= test_start_val:
            test_start_idx = i
            break

    if test_start_idx is not None and test_start_idx > 0:
        cutoff_date = str(all_dates[test_start_idx - 1])
    else:
        # Fallback: use test_start itself if no prior date available
        cutoff_date = str(test_start)
        logger.warning(f"No date found before test_start, using test_start as cutoff: {cutoff_date}")

    logger.info(f"Dead key detection cutoff date: {cutoff_date}")

    dead_keys_from_manifest = detect_dead_keys(
        source_df=source_df,
        manifest_df=manifest_df,
        key_col=key_col,
        target_col=target_col,
        date_col=date_col,
        cutoff_date=cutoff_date,
        lookback_periods=dead_key_lookback,
        threshold=dead_key_threshold,
    )

    # Step 2: Filter source data to test period
    test_df = source_df[
        (source_df[date_col] >= test_start_val) &
        (source_df[date_col] <= test_end_val)
    ].copy()

    # Step 3: Get unique keys in test data
    test_keys = set(test_df[key_col].unique())

    # Step 4: Detect keys with NO history before test period
    # These keys cannot be forecasted because we have no data to base predictions on
    # Handle type conversion for cutoff date
    if pd.api.types.is_integer_dtype(date_col_dtype):
        cutoff_val = int(cutoff_date)
    else:
        cutoff_val = str(cutoff_date)

    pre_test_df = source_df[source_df[date_col] <= cutoff_val]
    keys_with_history = set(pre_test_df[key_col].unique())
    keys_without_history = test_keys - keys_with_history

    logger.info(f"Keys without any history before test period: {len(keys_without_history)}")

    # Step 5: Get keys in manifest (excluding dead keys)
    manifest_keys = set(manifest_df['key'].unique())
    active_manifest_keys = manifest_keys - set(dead_keys_from_manifest)

    # Step 6: Identify new and existing keys (excluding keys without history)
    # New keys must have history to be forecastable
    potential_new_keys = test_keys - manifest_keys
    new_keys = list(potential_new_keys - keys_without_history)
    new_keys_without_history = list(potential_new_keys & keys_without_history)

    existing_keys = list(test_keys & active_manifest_keys)

    # Combine all dead keys: manifest dead keys + keys without history
    dead_keys = list(set(dead_keys_from_manifest) | keys_without_history)

    logger.info(f"Key detection summary:")
    logger.info(f"  - Test period keys: {len(test_keys)}")
    logger.info(f"  - Original manifest keys: {len(manifest_keys)}")
    logger.info(f"  - Dead keys from manifest (inactive): {len(dead_keys_from_manifest)}")
    logger.info(f"  - Keys without history (new but no data): {len(keys_without_history)}")
    logger.info(f"  - Total dead keys: {len(dead_keys)}")
    logger.info(f"  - Active manifest keys: {len(active_manifest_keys)}")
    logger.info(f"  - New keys (with history, forecastable): {len(new_keys)}")
    logger.info(f"  - Existing keys (in test and active manifest): {len(existing_keys)}")

    return new_keys, existing_keys, dead_keys, test_df


def create_dead_key_forecasts(
    dead_keys: List[str],
    test_start: str,
    test_end: str,
    key_col: str,
    date_col: str,
    target_col: str,
    forecast_horizon: int,
    source_df: pd.DataFrame = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create zero forecasts for dead keys, with actual values from source data.

    Dead keys are keys that existed in the training manifest but have become
    inactive (no recent demand). They get zero forecasts for all periods,
    but actual values are pulled from source data if available.

    Parameters
    ----------
    dead_keys : List[str]
        List of dead key values
    test_start : str
        Test period start (e.g., "202529" for weekly or "202506" for monthly)
    test_end : str
        Test period end (e.g., "202541" for weekly or "202512" for monthly)
    key_col : str
        Key column name
    date_col : str
        Date column name
    target_col : str
        Target column name for actuals
    forecast_horizon : int
        Number of periods to forecast
    source_df : pd.DataFrame, optional
        Source data to look up actual values
    time_format : str
        'year_week' (YYYYWW, max 52) or 'year_month' (YYYYMM, max 12)

    Returns
    -------
    pd.DataFrame
        DataFrame with zero forecasts for all dead keys
    """
    if not dead_keys:
        return pd.DataFrame()

    # Generate forecast periods from test_start using generic period functions
    try:
        forecast_periods = []
        current_period = str(test_start)

        for _ in range(forecast_horizon):
            # Keep periods in their original format (string for YYYY-WW, int for YYYYWW)
            try:
                forecast_periods.append(int(current_period))
            except ValueError:
                forecast_periods.append(current_period)  # Keep as string for dash format
            current_period = increment_period(current_period, 1, time_format)
    except (ValueError, IndexError):
        # Fallback: just use test_start as single period
        logger.warning(f"Could not parse test_start '{test_start}', using single period")
        forecast_periods = [test_start]

    # Build actuals lookup if source_df provided
    # IMPORTANT: Filter to only forecast periods to avoid looking up actuals from wrong time periods
    actuals_lookup = {}
    if source_df is not None and target_col in source_df.columns:
        dead_keys_set = set(dead_keys)
        forecast_periods_set = set(forecast_periods)

        # Filter to dead keys AND forecast periods only
        dead_key_data = source_df[
            (source_df[key_col].isin(dead_keys_set)) &
            (source_df[date_col].isin(forecast_periods_set))
        ]
        for _, row in dead_key_data.iterrows():
            key = row[key_col]
            period = row[date_col]
            actual = row[target_col]
            if pd.notna(actual):
                actuals_lookup[(key, period)] = float(actual)

    # Create forecast rows for all dead keys
    rows = []
    for key in dead_keys:
        for step_idx, period in enumerate(forecast_periods):
            # Look up actual value, default to 0.0 if not found
            actual_value = actuals_lookup.get((key, period), 0.0)

            rows.append({
                key_col: key,
                date_col: period,
                'origin_period': forecast_periods[0],
                'SnapshotTimePeriod': forecast_periods[0],
                'forecast_step': step_idx + 1,
                'lag': step_idx,
                'predicted': 0.0,
                'actual': actual_value,
                'model_level': 'dead_key',
                'model_name': 'dead_key',
                'model_params': '{}',
                'is_new_key': False,
                'is_dead_key': True,
            })

    dead_key_df = pd.DataFrame(rows)
    logger.info(f"Created {len(dead_key_df)} zero forecasts for {len(dead_keys)} dead keys")

    return dead_key_df


# =============================================================================
# PHASE 2: SEGMENTATION FEATURES FOR NEW KEYS
# =============================================================================

def compute_segmentation_features_for_new_keys(
    source_df: pd.DataFrame,
    new_keys: List[str],
    key_col: str,
    target_col: str,
    date_col: str,
    data_end: str,
    clustering_features: List[str],
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Compute segmentation features for new keys using data up to data_end.

    This mimics what the EDA crew computes in per_key_metrics.csv.

    Parameters
    ----------
    source_df : pd.DataFrame
        Full source data
    new_keys : List[str]
        List of new keys to compute features for
    key_col : str
        Key column name
    target_col : str
        Target column name
    date_col : str
        Date column name
    data_end : str
        End date for feature computation (typically val_end to avoid leakage)
    clustering_features : List[str]
        Features used in original clustering (from clustering_metrics.json)

    Returns
    -------
    pd.DataFrame
        DataFrame with segmentation features for each new key
    """
    if not new_keys:
        return pd.DataFrame()

    # Handle type conversion for date comparison
    date_col_dtype = source_df[date_col].dtype
    if pd.api.types.is_integer_dtype(date_col_dtype):
        data_end_val = int(data_end)
    else:
        data_end_val = str(data_end)

    # Filter data to keys and date range
    df = source_df[
        (source_df[key_col].isin(new_keys)) &
        (source_df[date_col] <= data_end_val)
    ].copy()

    if len(df) == 0:
        logger.warning(f"No historical data found for new keys before {data_end}")
        return pd.DataFrame()

    # Compute per-key metrics (same as EDA)
    results = []

    for key in new_keys:
        key_data = df[df[key_col] == key][target_col].values

        if len(key_data) == 0:
            # Key has no history - use defaults
            logger.warning(f"Key {key} has no history before {data_end}, using defaults")
            results.append({
                'key': key,
                'n_obs': 0,
                'mean': 0.0,
                'std': 0.0,
                'cv': 1.0,
                'adi': 1.4,  # Above threshold = intermittent
                'cv2': 0.5,  # Above threshold = variable
                'zero_fraction': 1.0,
                'demand_frequency': 0.0,
                'forecastability_score': 0.3,
                'trend_strength': 0.0,
                'seasonal_strength': 0.0,
                'autocorr_lag1': 0.0,
                'skewness': 0.0,
                'intermittency_class': 'lumpy',
            })
            continue

        # Basic statistics
        n_obs = len(key_data)
        mean_val = np.mean(key_data)
        std_val = np.std(key_data)

        # Coefficient of variation
        cv = std_val / mean_val if mean_val > 0 else 0.0
        cv2 = cv ** 2

        # Zero fraction and demand frequency
        zero_count = np.sum(key_data == 0)
        zero_fraction = zero_count / n_obs if n_obs > 0 else 0.0
        demand_frequency = 1.0 - zero_fraction

        # ADI (Average Demand Interval)
        nonzero_indices = np.where(key_data > 0)[0]
        if len(nonzero_indices) > 1:
            intervals = np.diff(nonzero_indices)
            adi = np.mean(intervals)
        else:
            adi = n_obs  # No demand or single demand

        # Skewness
        if std_val > 0 and n_obs > 2:
            skewness = float(pd.Series(key_data).skew())
        else:
            skewness = 0.0

        # Autocorrelation lag 1
        if n_obs > 2:
            series = pd.Series(key_data)
            autocorr = series.autocorr(lag=1)
            autocorr_lag1 = autocorr if not pd.isna(autocorr) else 0.0
        else:
            autocorr_lag1 = 0.0

        # Simple trend strength (linear regression slope normalized)
        if n_obs > 3:
            x = np.arange(n_obs)
            try:
                slope = np.polyfit(x, key_data, 1)[0]
                trend_strength = abs(slope) / (mean_val + 1e-10)
            except:
                trend_strength = 0.0
        else:
            trend_strength = 0.0

        # Seasonal strength (simple approximation)
        # Use time-format-aware seasonal period: 12 for monthly, 52 for weekly
        _seasonal_period = 12 if time_format == 'year_month' else 52
        seasonal_strength = 0.0
        if n_obs >= _seasonal_period:
            # Check if there's a yearly pattern
            try:
                series = pd.Series(key_data)
                seasonal_acf = series.autocorr(lag=_seasonal_period)
                seasonal_strength = seasonal_acf if not pd.isna(seasonal_acf) else 0.0
            except:
                seasonal_strength = 0.0

        # Forecastability score (composite)
        forecastability_score = (
            0.3 * (1 - zero_fraction) +  # More non-zero = more forecastable
            0.2 * (1 - min(cv, 2.0) / 2.0) +  # Lower CV = more forecastable
            0.2 * max(0, autocorr_lag1) +  # Positive autocorr = more forecastable
            0.15 * min(trend_strength, 1.0) +  # Some trend = forecastable
            0.15 * abs(seasonal_strength)  # Seasonality = forecastable
        )

        # Intermittency class (Syntetos-Boylan classification)
        # ADI threshold = 1.32, CV2 threshold = 0.49
        if adi < 1.32 and cv2 < 0.49:
            intermittency_class = 'smooth'
        elif adi < 1.32 and cv2 >= 0.49:
            intermittency_class = 'erratic'
        elif adi >= 1.32 and cv2 < 0.49:
            intermittency_class = 'intermittent'
        else:
            intermittency_class = 'lumpy'

        results.append({
            'key': key,
            'n_obs': n_obs,
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'adi': adi,
            'cv2': cv2,
            'zero_fraction': zero_fraction,
            'demand_frequency': demand_frequency,
            'forecastability_score': forecastability_score,
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'autocorr_lag1': autocorr_lag1,
            'skewness': skewness,
            'intermittency_class': intermittency_class,
        })

    features_df = pd.DataFrame(results)
    logger.info(f"Computed segmentation features for {len(features_df)} new keys")

    return features_df


# =============================================================================
# PHASE 3: SEGMENT ASSIGNMENT FOR NEW KEYS
# =============================================================================

def assign_segments_to_new_keys(
    new_key_features: pd.DataFrame,
    cluster_model_path: str,
    scaler_path: str,
    clustering_features: List[str],
    existing_segments: List[int],
    cluster_to_segment_map_path: Optional[str] = None,
) -> List[NewKeyAssignment]:
    """
    Assign new keys to existing segments using saved cluster model.

    Parameters
    ----------
    new_key_features : pd.DataFrame
        Segmentation features for new keys
    cluster_model_path : str
        Path to saved cluster model (cluster_model.joblib)
    scaler_path : str
        Path to saved scaler (scaler.joblib)
    clustering_features : List[str]
        Features used in clustering
    existing_segments : List[int]
        List of existing segment IDs (to validate assignments)
    cluster_to_segment_map_path : str, optional
        Path to cluster_to_segment_map.json (maps raw cluster predictions to final segment IDs)

    Returns
    -------
    List[NewKeyAssignment]
        Segment assignments for each new key
    """
    if len(new_key_features) == 0:
        return []

    # Load saved models with validation
    if not os.path.exists(cluster_model_path):
        logger.error(
            f"Cluster model not found at {cluster_model_path}. "
            f"Cannot assign segments to new keys. Run segmentation crew first."
        )
        # Return empty assignments - new keys will be handled as unassigned
        return []

    if not os.path.exists(scaler_path):
        logger.error(
            f"Scaler not found at {scaler_path}. "
            f"Cannot assign segments to new keys. Run segmentation crew first."
        )
        return []

    try:
        cluster_model = joblib.load(cluster_model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        logger.error(f"Failed to load cluster model or scaler: {e}")
        return []

    # Load cluster-to-segment mapping if available
    # This maps the raw cluster model predictions to final segment IDs
    # (accounts for segment merging and renumbering during training)
    cluster_to_segment_map = None
    if cluster_to_segment_map_path and os.path.exists(cluster_to_segment_map_path):
        with open(cluster_to_segment_map_path, 'r') as f:
            cluster_to_segment_map = json.load(f)
        # Convert string keys to int (JSON serializes int keys as strings)
        cluster_to_segment_map = {int(k): int(v) for k, v in cluster_to_segment_map.items()}
        logger.info(f"Loaded cluster-to-segment mapping: {cluster_to_segment_map}")

    # Prepare features for clustering
    # Check which clustering features are available
    available_features = [f for f in clustering_features if f in new_key_features.columns]
    missing_features = [f for f in clustering_features if f not in new_key_features.columns]

    if missing_features:
        logger.warning(f"Missing clustering features (using 0): {missing_features}")
        for f in missing_features:
            new_key_features[f] = 0.0

    # Extract and scale features
    X = new_key_features[clustering_features].values

    # Handle NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # Scale features using saved scaler
    X_scaled = scaler.transform(X)

    # Predict clusters
    predicted_segments = cluster_model.predict(X_scaled)

    # Get probabilities if available (GMM has predict_proba)
    # This allows us to assign to the most probable VALID segment if predicted segment doesn't exist
    segment_probs = None
    if hasattr(cluster_model, 'predict_proba'):
        segment_probs = cluster_model.predict_proba(X_scaled)

    # Validate assignments are in existing segments
    valid_segments = set(existing_segments)

    # Count invalid segment predictions (log once at the end)
    invalid_segment_count = 0

    assignments = []
    # IMPORTANT: Use enumerate to get positional index, not DataFrame index
    # predicted_segments is a numpy array with positional indices 0, 1, 2, ...
    # but df.iterrows() returns DataFrame indices which may be non-sequential
    for i, (df_idx, row) in enumerate(new_key_features.iterrows()):
        key = row['key']
        raw_cluster = int(predicted_segments[i])

        # Apply cluster-to-segment mapping if available
        # This maps raw cluster model predictions to final segment IDs
        if cluster_to_segment_map is not None and raw_cluster in cluster_to_segment_map:
            predicted_segment = cluster_to_segment_map[raw_cluster]
        else:
            predicted_segment = raw_cluster

        # If predicted segment STILL doesn't exist (shouldn't happen with proper mapping),
        # find the most probable VALID segment as fallback
        if predicted_segment not in valid_segments:
            invalid_segment_count += 1

            if segment_probs is not None:
                # Get probabilities for this sample and find most probable valid segment
                probs = segment_probs[i]
                # Sort segments by probability (descending) and pick first valid one
                sorted_segments = np.argsort(probs)[::-1]
                for seg in sorted_segments:
                    # Apply mapping to each candidate segment too
                    mapped_seg = cluster_to_segment_map.get(int(seg), int(seg)) if cluster_to_segment_map else int(seg)
                    if mapped_seg in valid_segments:
                        predicted_segment = mapped_seg
                        break
                else:
                    # Fallback: no valid segment found in probabilities, use first valid
                    predicted_segment = existing_segments[0]
            else:
                # No probabilities available (e.g., KMeans), use first valid segment
                predicted_segment = existing_segments[0]

        assignment = NewKeyAssignment(
            key=key,
            segment_id=predicted_segment,
            model_level=str(predicted_segment),  # New keys always use segment-level models
            intermittency_class=row.get('intermittency_class', 'lumpy'),
            segmentation_features={f: float(row[f]) for f in clustering_features if f in row},
        )
        assignments.append(assignment)

    # Log summary
    segment_counts = {}
    for a in assignments:
        segment_counts[a.segment_id] = segment_counts.get(a.segment_id, 0) + 1
    logger.info(f"New key segment assignments: {segment_counts}")

    # Log if any keys were remapped from invalid segments
    if invalid_segment_count > 0:
        logger.info(f"Remapped {invalid_segment_count} keys from invalid segments to nearest valid segment")

    return assignments


# =============================================================================
# PHASE 4: UPDATE TRAINING MANIFEST
# =============================================================================

def update_training_manifest(
    manifest_path: str,
    new_key_assignments: List[NewKeyAssignment],
    output_dir: str,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Update training manifest with new keys and create backup.

    Parameters
    ----------
    manifest_path : str
        Path to original training_manifest.csv
    new_key_assignments : List[NewKeyAssignment]
        Segment assignments for new keys
    output_dir : str
        Directory for output files

    Returns
    -------
    Tuple[pd.DataFrame, str, str]
        - Updated manifest DataFrame
        - Path to backup file
        - Path to updated manifest
    """
    # Load original manifest
    manifest_df = pd.read_csv(manifest_path)

    # Create backup
    backup_path = os.path.join(output_dir, "training_manifest_backup.csv")
    shutil.copy(manifest_path, backup_path)
    logger.info(f"Created manifest backup: {backup_path}")

    if not new_key_assignments:
        logger.info("No new keys to add to manifest")
        return manifest_df, backup_path, manifest_path

    # Create records for new keys
    new_records = []
    for assignment in new_key_assignments:
        record = {
            'key': assignment.key,
            'segment_id': assignment.segment_id,
            'model_level': assignment.model_level,
            'model_strategy': 'segment_pooled',
            'allocation_rationale': 'new_key_assigned_to_segment',
            'allocation_confidence': 0.5,  # Default confidence for new keys
            'demand_pattern': assignment.intermittency_class,
            'intermittency_class': assignment.intermittency_class,
            'model_group': str(assignment.segment_id),  # Use segment_id as model_group
            'feature_file': 'train_features.csv',
        }
        new_records.append(record)

    # Append to manifest
    new_keys_df = pd.DataFrame(new_records)

    # Ensure columns match
    for col in manifest_df.columns:
        if col not in new_keys_df.columns:
            new_keys_df[col] = None

    # Reorder columns to match original
    new_keys_df = new_keys_df[manifest_df.columns]

    # Concatenate
    updated_manifest = pd.concat([manifest_df, new_keys_df], ignore_index=True)

    # Save updated manifest (overwrite original)
    updated_manifest.to_csv(manifest_path, index=False)
    logger.info(f"Updated manifest with {len(new_key_assignments)} new keys: {manifest_path}")

    return updated_manifest, backup_path, manifest_path


# =============================================================================
# PHASE 4.5: REGENERATE FEATURES WITH LATEST DATA
# =============================================================================

def regenerate_features_for_inference(
    config: DemandForecastConfig,
    manifest_df: pd.DataFrame,
    feature_dir: str,
    key_col: str,
    date_col: str,
    target_col: str,
    source_df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Regenerate features using the latest source data up to val_end.

    This ensures that lag features, rolling averages, and derived features
    reflect the most recent data available before the inference period.

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object
    manifest_df : pd.DataFrame
        Updated training manifest with all keys (including new keys)
    feature_dir : str
        Directory containing feature files (will be updated in-place)
    key_col : str
        Key column name
    date_col : str
        Date column name
    target_col : str
        Target column name
    source_df : pd.DataFrame, optional
        Pre-loaded source data. If None, loads from config.input_data_path.
        Passing this avoids redundant disk reads when the caller already has the data.

    Returns
    -------
    bool
        True if features were regenerated successfully
    """
    from utils.feature_engineering import run_leakage_free_feature_pipeline

    logger.info("Regenerating features with latest data...")

    # Use pre-loaded source data if available, otherwise load from disk
    if source_df is None:
        from utils.agent_utilities import load_source_data
        source_df = load_source_data(config.input_data_path)
    else:
        source_df = source_df.copy()  # Don't mutate caller's DataFrame

    # Ensure 'key' column exists — all intermediate files use standardised 'key' column
    if 'key' not in source_df.columns:
        if len(config.prediction_key_cols) == 1 and config.prediction_key_cols[0] in source_df.columns:
            source_df['key'] = source_df[config.prediction_key_cols[0]]
        else:
            source_df['key'] = source_df[config.prediction_key_cols].astype(str).agg('_'.join, axis=1)

    # Get all keys from manifest (including new keys)
    all_keys = manifest_df['key'].unique()
    logger.info(f"Processing {len(all_keys)} keys from manifest")

    # Filter source data to only include keys in manifest (use 'key' column, not key_col)
    source_df = source_df[source_df['key'].isin(all_keys)]
    logger.info(f"Filtered to {len(source_df)} rows for {len(all_keys)} keys")

    # Load segment info from seg_output
    seg_dir = os.path.join(config.artifact_base_path, "seg_output")
    per_key_segments_path = os.path.join(seg_dir, "per_key_with_segments.csv")

    segment_col = None
    demand_pattern_col = None

    if os.path.exists(per_key_segments_path):
        segments_df = pd.read_csv(per_key_segments_path)

        # Merge segment info into source data
        segment_cols_to_merge = ['key']
        if 'segment_id' in segments_df.columns:
            segment_cols_to_merge.append('segment_id')
            segment_col = 'segment_id'
        if 'demand_pattern' in segments_df.columns:
            segment_cols_to_merge.append('demand_pattern')
            demand_pattern_col = 'demand_pattern'
        if 'intermittency_class' in segments_df.columns:
            segment_cols_to_merge.append('intermittency_class')

        # Keep only the columns we need and drop duplicates on key
        segments_df = segments_df[segment_cols_to_merge].drop_duplicates(subset=['key'])

        # Merge
        source_df = source_df.merge(segments_df, on='key', how='left')
        logger.info("Merged segment information from per_key_with_segments.csv")

    # Also try to merge from manifest if segment info not in seg_output
    if segment_col is None and 'segment_id' in manifest_df.columns:
        manifest_info = manifest_df[['key', 'segment_id']].drop_duplicates()
        source_df = source_df.merge(manifest_info, on='key', how='left', suffixes=('', '_manifest'))
        if 'segment_id_manifest' in source_df.columns:
            source_df['segment_id'] = source_df['segment_id'].fillna(source_df['segment_id_manifest'])
            source_df.drop(columns=['segment_id_manifest'], inplace=True)
        segment_col = 'segment_id'

    # Gather categorical columns from config (all feature groups)
    categorical_cols = list(config.categorical_feature_cols or [])

    # Add price categoricals
    if config.price_features and config.price_features.categorical:
        categorical_cols.extend(config.price_features.categorical)

    # Add promo categoricals
    if config.promo_features and config.promo_features.categorical:
        categorical_cols.extend(config.promo_features.categorical)

    # Add holiday categoricals
    if config.holiday_features and config.holiday_features.categorical:
        categorical_cols.extend(config.holiday_features.categorical)

    # Add weather categoricals
    if config.weather_features and config.weather_features.categorical:
        categorical_cols.extend(config.weather_features.categorical)

    # Filter to columns that exist in source_df
    categorical_cols = [c for c in categorical_cols if c in source_df.columns]

    logger.info(f"Using {len(categorical_cols)} categorical columns for encoding")

    # =========================================================================
    # Load feature strategy decision from training to ensure consistent features.
    # Training uses run_leakage_free_feature_pipeline via feature_reasoning.py
    # which produces log-transformed features, external feature lags, etc.
    # Inference must use the SAME pipeline to avoid feature mismatch.
    # =========================================================================
    strategy_path = os.path.join(config.artifact_base_path, "feature_output", "feature_strategy_decision.json")
    strategy_params = {}
    if os.path.exists(strategy_path):
        with open(strategy_path, 'r') as f:
            strategy_params = json.load(f)
        logger.info(f"Loaded feature strategy from: {strategy_path}")
    else:
        logger.warning(f"Feature strategy not found at {strategy_path}, using defaults")

    forecast_lag = getattr(config.design, 'forecast_lag', 4)
    allowed_external_cols = list(set(
        (config.all_numeric_features() if hasattr(config, 'all_numeric_features') else [])
        + (config.all_categorical_features() if hasattr(config, 'all_categorical_features') else [])
    ))

    # =========================================================================
    # CRITICAL: Resolve hierarchy_cols from the PERSISTED ARTIFACT (single
    # source of truth), NOT from feature_strategy_decision.json.
    #
    # FeatureStrategyDecision does not serialize hierarchy_cols, so the
    # strategy JSON written at training time always has hierarchy_cols=null.
    # Reading it from there silently disables hierarchy-level temporal
    # features during inference regeneration — which then overwrites
    # train_features.csv WITHOUT the hier features the trained models
    # expect. The retrained models in Phase 5 end up with a different
    # feature set than the original model specs.
    # =========================================================================
    _inf_hierarchy_cols: list = []
    if getattr(config.design, 'enable_hierarchy_features', True):
        try:
            from utils.hierarchy_resolution import resolve_hierarchies
            _seg_dir = os.path.join(config.artifact_base_path, 'seg_output')
            _h = resolve_hierarchies(config=config, source_df=source_df, seg_dir=_seg_dir)
            _inf_hierarchy_cols = list(_h.product) if _h.product else list(_h.flat)
            if _inf_hierarchy_cols:
                logger.info(
                    f"Resolved hierarchy_cols for inference regen (source={_h.source}): "
                    f"{_inf_hierarchy_cols}"
                )
        except Exception as _hcol_err:
            logger.warning(f"Hierarchy resolution for inference regen failed: {_hcol_err}")

    # Run feature pipeline (same leakage-free pipeline as training)
    inf_tf = getattr(config, 'time_format', 'year_week')
    if inf_tf in ('auto', 'date'):
        inf_tf = 'year_week'
    try:
        result = run_leakage_free_feature_pipeline(
            df=source_df,
            key_cols=[key_col],
            date_col=date_col,
            target_col=target_col,
            train_start=config.train_start,
            train_end=config.train_end,
            val_start=config.val_start,
            val_end=config.val_end,
            test_start=config.test_start,
            test_end=config.test_end,
            forecast_lag=forecast_lag,
            segment_col=segment_col,
            categorical_cols=categorical_cols if categorical_cols else None,
            output_dir=feature_dir,
            # Strategy parameters from training
            target_lags=strategy_params.get('target_lags'),
            rolling_windows=strategy_params.get('rolling_windows'),
            rolling_stats=strategy_params.get('rolling_stats'),
            include_intermittency_features=strategy_params.get('include_intermittency_features', False),
            apply_log_transform=strategy_params.get('apply_log_transform', False),
            apply_differencing=strategy_params.get('apply_differencing', False),
            include_trend_features=strategy_params.get('include_trend_features', True),
            include_fourier_features=strategy_params.get('has_strong_seasonality', True),
            fourier_order=strategy_params.get('fourier_order', 3),
            seasonal_period=strategy_params.get('seasonal_period', 12 if getattr(config, 'time_format', 'year_week') == 'year_month' else 52),
            allowed_external_cols=allowed_external_cols if allowed_external_cols else None,
            external_feature_lags=strategy_params.get('feature_lags'),
            # Phase 3: Hierarchy columns — resolved from the persisted
            # artifact above so inference regen matches training exactly.
            hierarchy_cols=_inf_hierarchy_cols if _inf_hierarchy_cols else strategy_params.get('hierarchy_cols'),
            # Phase 6: Rich features + history embeddings
            enable_rich_features=strategy_params.get('enable_rich_features', True),
            future_unknown_features=strategy_params.get('future_unknown_features'),
        )

        logger.info(f"Feature regeneration complete: {result.n_features_created} features created")
        logger.info(f"  Train: {result.n_rows_train} rows")
        logger.info(f"  Val: {result.n_rows_val} rows")
        logger.info(f"  Test: {result.n_rows_test} rows")

        return True

    except Exception as e:
        logger.error(f"Feature regeneration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# =============================================================================
# PHASE 5: RETRAIN MODELS
# =============================================================================

def retrain_all_models(
    config: DemandForecastConfig,
    manifest_df: pd.DataFrame,
    model_specs: Dict[str, Any],
    feature_dir: str,
    model_dir: str,
    target_col: str,
    key_col: str,
    date_col: str,
    train_cutoff: Optional[str] = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Retrain all models using existing model_type and hyperparameters.

    Models are trained on data up to train_cutoff (or val_end if not specified).

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object
    manifest_df : pd.DataFrame
        Updated training manifest
    model_specs : Dict
        Model specifications from final_model_specs.json
    feature_dir : str
        Directory containing feature files
    model_dir : str
        Directory to save retrained models
    target_col : str
        Target column name
    key_col : str
        Key column name
    date_col : str
        Date column name
    train_cutoff : str, optional
        Custom cutoff date for training data. If provided, includes data up to
        this period (inclusive). If None, uses all train + val data.
        Used by backtesting to roll forward the training window.

    Returns
    -------
    Tuple[Dict[str, Any], int]
        - Dictionary of retrained models {model_level: model}
        - Number of models retrained
    """
    from utils.model_training import (
        train_lightgbm, train_xgboost, train_catboost, train_random_forest,
        train_zero_inflated, train_hurdle_model, train_tweedie,
        train_weighted_ensemble, train_model_by_name, TRAINING_REGISTRY,
        MULTI_HORIZON_AVAILABLE, UNIVARIATE_MODELS,
    )

    # Import multi-horizon training if available
    if MULTI_HORIZON_AVAILABLE:
        from utils.multi_horizon_training import (
            train_multi_horizon_lightgbm,
            train_multi_horizon_xgboost,
            train_multi_horizon_ensemble,
        )

    # Load feature data (format-agnostic — parquet preferred, CSV fallback).
    from utils.feature_io import read_features_intermediate
    train_df = read_features_intermediate(feature_dir, "train_features")
    val_df   = read_features_intermediate(feature_dir, "val_features")

    # If train_cutoff is specified, include test data and filter by cutoff.
    # This is used by backtesting to roll forward the training window.
    if train_cutoff is not None:
        test_df = read_features_intermediate(feature_dir, "test_features")
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Filter to data up to train_cutoff (inclusive)
        # Handle int, float, and string date formats
        date_col_dtype = combined_df[date_col].dtype
        if pd.api.types.is_integer_dtype(date_col_dtype) or pd.api.types.is_float_dtype(date_col_dtype):
            # Convert cutoff to numeric (handles both int and float columns)
            cutoff_val = float(train_cutoff)
        else:
            cutoff_val = str(train_cutoff)

        combined_df = combined_df[combined_df[date_col] <= cutoff_val]
        logger.info(f"Training data up to {train_cutoff}: {len(combined_df)} rows")
    else:
        # Standard inference: combine train and val
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        logger.info(f"Combined training data (train + val): {len(combined_df)} rows")

    # CRITICAL: Sort by key + date so the temporal 80/20 split below is correct.
    # Without this, pd.concat ordering is fragile and the "last 20%" could include
    # rows from random time periods instead of the most recent data.
    combined_df = combined_df.sort_values([key_col, date_col]).reset_index(drop=True)

    # Get feature columns from model specs
    feature_columns = model_specs.get('feature_columns', [])
    if not feature_columns:
        # Fallback: infer from data
        exclude_cols = {target_col, key_col, date_col, 'split', 'model_level', 'model_group',
                       'segment_id', 'intermittency_class', 'demand_pattern', 'label'}
        feature_columns = [c for c in combined_df.columns if c not in exclude_cols and not c.endswith('_log')]
        logger.warning(f"No feature_columns in specs, inferred {len(feature_columns)} features")

    # Filter to available features and pad missing ones with zeros
    available_features = [c for c in feature_columns if c in combined_df.columns]
    missing_features = [c for c in feature_columns if c not in combined_df.columns]
    if missing_features:
        logger.warning(
            f"FEATURE MISMATCH: Only {len(available_features)}/{len(feature_columns)} features available.\n"
            f"Missing features (first 10): {missing_features[:10]}\n"
            f"Padding missing features with zeros to maintain model compatibility."
        )
        # Pad missing features with zeros to ensure consistent feature count
        for mf in missing_features:
            combined_df[mf] = 0.0
        available_features = feature_columns  # Now all features are available

    # Get global hyperparameters
    global_hp = model_specs.get('hyperparameters', {})
    global_model_type = model_specs.get('model_type', 'lightgbm')

    # Get per-model-group specs
    model_list = model_specs.get('models', [])
    model_group_specs = {}
    for spec in model_list:
        mg = str(spec.get('model_group', ''))
        if mg:
            mg_spec = {
                'model_type': spec.get('model_type', global_model_type),
                'hyperparameters': spec.get('hyperparameters', global_hp),
            }
            # Preserve multi-horizon metadata for retraining
            if spec.get('is_multi_horizon'):
                mg_spec['is_multi_horizon'] = True
                mg_spec['multi_horizon_config'] = spec.get('multi_horizon_config', {})
            model_group_specs[mg] = mg_spec

    # Get unique model_levels from manifest
    model_levels = manifest_df['model_level'].unique()
    logger.info(f"Retraining {len(model_levels)} model levels")

    retrained_models = {}
    models_retrained = 0

    for ml in model_levels:
        ml_str = str(ml)

        try:
            # Get keys for this model level
            ml_keys = manifest_df[manifest_df['model_level'] == ml]['key'].unique()

            # Filter training data to these keys
            ml_data = combined_df[combined_df[key_col].isin(ml_keys)]

            if len(ml_data) < 10:
                logger.warning(f"Model level {ml_str} has only {len(ml_data)} rows, skipping — "
                               f"keys will fall back to nearest available model at forecast time")
                continue

            # Sort per-segment data by date to ensure temporal ordering for the split
            ml_data = ml_data.sort_values([key_col, date_col])

            # Prepare features and target
            X = ml_data[available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
            y = ml_data[target_col].values

            # Split for validation (last 20% chronologically)
            val_size = max(int(len(X) * 0.2), 1)
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]

            # Get model spec for this level
            ml_spec = model_group_specs.get(ml_str, {})
            model_type = ml_spec.get('model_type', global_model_type)
            hp = ml_spec.get('hyperparameters', global_hp).copy()

            # Remove ensemble-specific params for non-ensemble training
            hp_clean = {k: v for k, v in hp.items()
                       if k not in ('member_params', 'ensemble_weights', 'is_ensemble', 'ensemble_info')}

            # Sanitize classification-specific hyperparameters that may leak
            # from ordinal_regression or discrete_classifier model specs
            classification_only_params = {
                'num_class', 'num_classes', 'is_unbalance', 'scale_pos_weight',
                'class_weight', 'multi_strategy',
                # Discrete demand metadata (not model params)
                'n_classes', 'unique_values', 'train_accuracy', 'val_accuracy',
                'n_unique_values', 'base_unit', 'cardinality_category',
                'val_wape_before_snap', 'val_wape_after_snap',
            }
            hp_clean = {k: v for k, v in hp_clean.items()
                       if k not in classification_only_params}

            # Reset classification objectives to regression
            if hp_clean.get('objective') in (
                'multiclass', 'multiclassova', 'binary', 'cross_entropy',
                'cross_entropy_lambda', 'binary:logistic', 'multi:softmax',
                'multi:softprob',
            ):
                hp_clean['objective'] = 'regression'

            # Reset classification metrics that conflict with regression objectives
            classification_metrics = {
                'multi_logloss', 'multi_error', 'auc_mu', 'binary_logloss',
                'binary_error', 'cross_entropy', 'cross_entropy_lambda',
                'auc', 'average_precision',
            }
            if hp_clean.get('metric') in classification_metrics:
                hp_clean.pop('metric')
            # Also handle list-form metrics
            if isinstance(hp_clean.get('metric'), list):
                hp_clean['metric'] = [m for m in hp_clean['metric'] if m not in classification_metrics]
                if not hp_clean['metric']:
                    hp_clean.pop('metric')

            # Tweedie/Poisson objectives require non-negative targets
            obj = hp_clean.get('objective', hp.get('objective', ''))
            if obj in ('tweedie', 'poisson') or model_type.lower() == 'tweedie':
                y_train = np.clip(y_train, 0, None)
                y_val = np.clip(y_val, 0, None)

            # Train model based on type
            model = None

            # Handle univariate models (croston, sba, tsb, imapa, etc.)
            # These only need target values, not features
            if model_type.lower() in UNIVARIATE_MODELS:
                result = train_model_by_name(model_type, None, y_train, None, y_val)
                model = result.model

            # Handle multi-horizon models
            elif model_type.startswith('multi_horizon') and MULTI_HORIZON_AVAILABLE:
                is_multi_horizon = ml_spec.get('is_multi_horizon', model_specs.get('is_multi_horizon', False))
                multi_horizon_config = ml_spec.get('multi_horizon_config', model_specs.get('multi_horizon_config', {}))
                strategy = multi_horizon_config.get('strategy', 'direct_separate')
                max_horizon = multi_horizon_config.get('max_horizon', config.forecast_horizon)
                target_horizon = multi_horizon_config.get('target_horizon', config.forecast_horizon)

                if model_type == 'multi_horizon_lightgbm':
                    result = train_multi_horizon_lightgbm(
                        X_train, y_train, X_val, y_val,
                        max_horizon=max_horizon,
                        target_horizon=target_horizon,
                        strategy=strategy,
                        params=hp_clean,
                    )
                elif model_type == 'multi_horizon_xgboost':
                    result = train_multi_horizon_xgboost(
                        X_train, y_train, X_val, y_val,
                        max_horizon=max_horizon,
                        target_horizon=target_horizon,
                        strategy=strategy,
                        params=hp_clean,
                    )
                elif model_type == 'multi_horizon_ensemble':
                    result = train_multi_horizon_ensemble(
                        X_train, y_train, X_val, y_val,
                        max_horizon=max_horizon,
                        target_horizon=target_horizon,
                    )
                else:
                    # Fallback to multi_horizon_lightgbm
                    result = train_multi_horizon_lightgbm(
                        X_train, y_train, X_val, y_val,
                        max_horizon=max_horizon,
                        target_horizon=target_horizon,
                        strategy=strategy,
                        params=hp_clean,
                    )

                model = result.model
                logger.info(f"Model {ml_str}: Trained multi-horizon {model_type} with target_horizon={target_horizon}")

            elif 'ensemble' in model_type.lower() or '+' in model_type:
                # Ensemble model
                if '+' in model_type:
                    # Strip prefixes like 'ensemble_', 'pattern_ensemble_', etc.
                    cleaned_type = model_type
                    for prefix in ['pattern_ensemble_', 'ensemble_']:
                        cleaned_type = cleaned_type.replace(prefix, '')
                    component_types = [t.strip() for t in cleaned_type.split('+')]
                else:
                    component_types = ['lightgbm', 'xgboost', 'catboost']

                member_params = hp.get('member_params', {})
                # Sanitize per-component params: strip classification metrics/objectives
                # that may leak from original training (e.g., discrete_classifier specs)
                _cls_objectives = {'multiclass', 'multiclassova', 'binary', 'cross_entropy',
                                   'cross_entropy_lambda', 'binary:logistic', 'multi:softmax', 'multi:softprob'}
                _cls_metrics = {'multi_logloss', 'multi_error', 'auc_mu', 'binary_logloss',
                                'binary_error', 'cross_entropy', 'cross_entropy_lambda', 'auc', 'average_precision'}
                _cls_metadata = {'num_class', 'num_classes', 'n_classes', 'unique_values',
                                 'is_unbalance', 'class_weight', 'multi_strategy',
                                 'train_accuracy', 'val_accuracy', 'n_unique_values',
                                 'base_unit', 'cardinality_category',
                                 'val_wape_before_snap', 'val_wape_after_snap'}
                sanitized_member_params = {}
                for comp_type, comp_hp in member_params.items():
                    if isinstance(comp_hp, dict):
                        cleaned = {k: v for k, v in comp_hp.items() if k not in _cls_metadata}
                        if cleaned.get('objective') in _cls_objectives:
                            cleaned['objective'] = 'regression'
                        m = cleaned.get('metric')
                        if isinstance(m, str) and m in _cls_metrics:
                            cleaned.pop('metric')
                        elif isinstance(m, list):
                            cleaned['metric'] = [x for x in m if x not in _cls_metrics]
                            if not cleaned['metric']:
                                cleaned.pop('metric')
                        sanitized_member_params[comp_type] = cleaned
                    else:
                        sanitized_member_params[comp_type] = comp_hp
                member_params = sanitized_member_params
                weights = hp.get('ensemble_weights', None)

                result = train_weighted_ensemble(
                    X_train, y_train, X_val, y_val,
                    model_types=component_types,
                    member_params=member_params,
                    weights=weights,
                )
                model = result.model

            elif model_type.lower() in TRAINING_REGISTRY:
                result = train_model_by_name(model_type, X_train, y_train, X_val, y_val, params=hp_clean)
                model = result.model

            else:
                # Default to LightGBM
                result = train_lightgbm(X_train, y_train, X_val, y_val, params=hp_clean)
                model = result.model

            if model is not None:
                # STATE-OF-THE-ART: Apply threshold calibration for zero-inflated/hurdle models
                if model_type.lower() in ['zero_inflated', 'hurdle_model']:
                    try:
                        from utils.state_of_art_training import (
                            calibrate_zero_inflated_threshold,
                            optimize_hurdle_threshold,
                        )

                        if model_type.lower() == 'zero_inflated':
                            calibrated_threshold, threshold_info = calibrate_zero_inflated_threshold(
                                model=model,
                                X_val=X_val,
                                y_val=y_val,
                            )
                        else:
                            zero_frac = float((y_train == 0).mean())
                            calibrated_threshold, threshold_info = optimize_hurdle_threshold(
                                model=model,
                                X_val=X_val,
                                y_val=y_val,
                                zero_fraction=zero_frac,
                            )

                        if calibrated_threshold and hasattr(model, 'zero_threshold'):
                            model.zero_threshold = calibrated_threshold
                            logger.debug(f"Model {ml_str}: Calibrated threshold to {calibrated_threshold:.2f}")
                    except Exception as thresh_err:
                        logger.debug(f"Threshold calibration skipped for {ml_str}: {thresh_err}")

                # Save model
                model_path = os.path.join(model_dir, f"{ml_str}_model.pkl")
                joblib.dump(model, model_path)

                retrained_models[ml_str] = {
                    'model': model,
                    'model_type': model_type,
                    'model_path': model_path,
                    'hyperparameters': hp,
                    'n_train_rows': len(X_train),
                }
                models_retrained += 1

                if models_retrained % 10 == 0:
                    logger.info(f"Retrained {models_retrained} models...")

        except Exception as e:
            logger.error(f"Failed to retrain model level {ml_str}: {e}")
            continue

    logger.info(f"Successfully retrained {models_retrained} models")

    return retrained_models, models_retrained


# =============================================================================
# PHASE 6 (variant): DIRECT MULTI-HORIZON FORWARD FORECASTS
# =============================================================================


def _generate_direct_multihorizon_forecasts(
    config: "DemandForecastConfig",
    feature_dir: str,
    seg_dir: str,
    output_dir: str,
    key_col: str,
    date_col: str,
    target_col: str,
    backtest_origin: Optional[int] = None,
    cached_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Direct multi-horizon forward forecasts.

    Replaces the recursive-feedback path with H dedicated horizon heads.
    Reads the same pre-computed feature files (`train/val/test_features.csv`)
    so there is no redundant feature engineering work, and it respects the
    same key/time/target columns the rest of the pipeline uses.

    Two operating modes:

    - **Forward inference** (``backtest_origin=None``, default):
      Combined training panel = train + val.  Predicts from
      ``config.val_end``.  Used by `run_inference_pipeline`.

    - **Rolling-origin backtest** (``backtest_origin`` set to the
      origin's train_cutoff as YYYYWW int):
      Combined training panel = train + val + test, but each horizon
      head is only trained on rows whose target week falls at or
      before the backtest origin.  The prediction origin is
      ``backtest_origin``.  Used by
      `utils.backtesting._generate_origin_forecasts`.

    Returns a DataFrame with columns:
      [key_col, date_col (target period), predicted, actual, origin_period,
       forecast_step, lag, model_level, model_name, model_params,
       is_new_key, is_dead_key]

    `actual` is populated from the (key, target_week, target_col) triples
    present in the combined feature panel when available, falling back to
    0.0 for weeks whose actuals aren't in the panel yet (forward inference).
    """
    import json as _json

    from utils.direct_multihorizon import (  # local import to avoid hard dep
        DirectMHConfig,
        train_direct_multihorizon,
        predict_direct_multihorizon,
        augment_features,
    )

    # --- load features (cached reuse across backtest origins when available)
    if cached_features and all(k in cached_features for k in ("train", "val", "test")):
        logger.info("DMH: using cached feature frames (cross-origin reuse)")
        train_df = cached_features["train"]
        val_df = cached_features["val"]
        test_df = cached_features["test"]
        _features_already_augmented = bool(cached_features.get("_augmented"))
    else:
        # Format-agnostic: helpers prefer parquet, fall back to CSV.
        # `low_memory=False` is silently ignored on the parquet path
        # (parquet is dtype-preserving so the kwarg has no meaning).
        from utils.feature_io import (
            features_intermediate_exists, read_features_intermediate,
        )
        for n in ("train_features", "val_features", "test_features"):
            if not features_intermediate_exists(feature_dir, n):
                raise FileNotFoundError(
                    f"Missing required feature file: "
                    f"{os.path.join(feature_dir, n)}.[parquet|csv]"
                )

        logger.info("DMH: loading feature files...")
        train_df = read_features_intermediate(feature_dir, "train_features", low_memory=False)
        val_df   = read_features_intermediate(feature_dir, "val_features",   low_memory=False)
        test_df  = read_features_intermediate(feature_dir, "test_features",  low_memory=False)
        _features_already_augmented = False

    # ── Defensive: drop all-null padding rows (key/period missing). A
    # feature-engineering path can append empty rows (observed on TH data:
    # key=None, year_week=None) that poison the year_week->int casts in the
    # augmenters below and mix str/float dtypes. Valid panels have none, so
    # this is a harmless no-op for UK/DE.
    def _drop_null_period_rows(_df, _name):
        if _df is None or len(_df) == 0 or "year_week" not in _df.columns:
            return _df
        _mask = _df["year_week"].notna()
        if "key" in _df.columns:
            _mask = _mask & _df["key"].notna()
        _n = int((~_mask).sum())
        if _n:
            logger.info("DMH: dropped %d null key/period padding rows from %s", _n, _name)
            return _df[_mask].copy()
        return _df
    train_df = _drop_null_period_rows(train_df, "train")
    val_df = _drop_null_period_rows(val_df, "val")
    test_df = _drop_null_period_rows(test_df, "test")

    # Apply the engineered-feature augmenters that proved beneficial in
    # the 36-iteration sweep (utils/direct_multihorizon:augment_features):
    #   - per-key trajectory (trend, volatility, acceleration)
    #   - APG-level weekly sum (shifted 1 week to avoid leak)
    #   - target-week calendar+promo+weather features for horizon=h
    # Gated by new flags so operators can toggle individually.
    #
    # Column names are derived from the config so the augmenters work
    # unchanged on any market:
    #   - apg_col is taken from `dmh_apg_col` OR the first entry of
    #     `imputation_level` (TH uses "APGDescription", UK uses "APG_code")
    #   - forward-feature candidates are the union of price/promo/holiday/
    #     weather categorical+numeric feature lists declared in the config,
    #     plus a small calendar default set.  This means TH's promo
    #     columns (BOGOF_PROMO, Monsoon, ...) and UK's (promo_shipment_flag,
    #     WTHR_avgTemp, ...) are both picked up automatically without any
    #     code change.
    # If the caller already augmented once (cross-origin cached features),
    # we must NOT augment again - it would double-add columns.
    _in_backtest = backtest_origin is not None
    _bt_enable_augmenters = bool(getattr(config.design, "dmh_backtest_enable_augmenters", True))
    _skip_augmenters = _features_already_augmented or (_in_backtest and not _bt_enable_augmenters)

    enable_traj = (not _skip_augmenters) and bool(getattr(config.design, "dmh_enable_trajectory_features", True))
    enable_apg = (not _skip_augmenters) and bool(getattr(config.design, "dmh_enable_apg_features", True))
    enable_fwd = (not _skip_augmenters) and bool(getattr(config.design, "dmh_enable_forward_features", True))

    # Derive apg_col in priority order: explicit dmh_apg_col -> imputation_level[0] -> "APG_code".
    apg_col = str(getattr(config.design, "dmh_apg_col", "") or "").strip()
    if not apg_col:
        imp_level = getattr(config.design, "imputation_level", None) or []
        if isinstance(imp_level, (list, tuple)) and imp_level:
            apg_col = str(imp_level[0])
    if not apg_col:
        apg_col = "APG_code"

    # Collect forward-feature candidate column names from config feature
    # declarations.  Each config attribute can be one of:
    #   - a plain list (e.g. numeric_feature_cols)
    #   - a dict with {numeric: [], categorical: []} keys
    #   - a Pydantic FeatureBlock with .numeric / .categorical attributes
    # Duplicates + columns missing in the feature panel are handled inside
    # the augmenter via `_FWD_CANDIDATE_COLS` -> `present` filtering.
    def _collect_feats(obj):
        out = []
        if obj is None:
            return out
        if isinstance(obj, list):
            return [str(c) for c in obj]
        if isinstance(obj, dict):
            for k in ("numeric", "categorical"):
                v = obj.get(k) or []
                if isinstance(v, list):
                    out.extend(str(c) for c in v)
            return out
        # Pydantic-style object with .numeric / .categorical attributes.
        for attr in ("numeric", "categorical"):
            v = getattr(obj, attr, None) or []
            if isinstance(v, list):
                out.extend(str(c) for c in v)
        return out

    fwd_candidates: List[str] = []
    for attr in ("price_features", "promo_features", "holiday_features", "weather_features"):
        fwd_candidates.extend(_collect_feats(getattr(config, attr, None)))
    # Plus always-safe calendar defaults (these are auto-produced by the
    # feature pipeline for every market).
    fwd_candidates.extend([
        "week_of_year", "week_of_year_sin", "week_of_year_cos",
        "month", "month_sin", "month_cos", "quarter",
        "is_holiday", "holiday_flag", "weeks_to_nearest_holiday", "season",
        "holiday", "holidays",
    ])
    # Dedup preserving order
    seen = set()
    fwd_candidates = [c for c in fwd_candidates if not (c in seen or seen.add(c))]

    if enable_traj or enable_apg or enable_fwd:
        logger.info(
            "DMH: augmenting features (trajectory=%s, apg=%s [apg_col=%s], forward=%s [%d candidates])",
            enable_traj, enable_apg, apg_col, enable_fwd, len(fwd_candidates),
        )
        aug_kwargs = dict(
            key_col=key_col,
            target_col=target_col,
            apg_col=apg_col,
            enable_trajectory=enable_traj,
            enable_apg=enable_apg,
            enable_forward=enable_fwd,
            forward_candidate_cols=fwd_candidates,
        )
        train_df = augment_features(train_df, horizon=4, **aug_kwargs)
        val_df = augment_features(val_df, horizon=4, **aug_kwargs)
        test_df = augment_features(test_df, horizon=4, **aug_kwargs)

    # --- segment map for calibration (optional)
    key_to_segment: Dict[str, str] = {}
    per_key_path = os.path.join(seg_dir, "per_key_with_segments.csv")
    if os.path.exists(per_key_path):
        seg_df = pd.read_csv(per_key_path)
        if "segment_id" in seg_df.columns:
            key_to_segment = dict(
                zip(seg_df[key_col].astype(str), seg_df["segment_id"].astype(str))
            )

    # --- build config (apply backtest-specific overrides when running inside
    # rolling-origin backtest).  These tighten the training cost per origin
    # without affecting forward-inference.
    def _bt_override(bt_field, base_field, default, sentinel=None):
        if not _in_backtest:
            return default
        bt_val = getattr(config.design, bt_field, sentinel)
        if isinstance(bt_val, str):
            return bt_val.strip() or default
        if isinstance(bt_val, (list, tuple)):
            return list(bt_val) if bt_val else default
        if isinstance(bt_val, (int, float)):
            return bt_val if bt_val != sentinel else default
        return default if bt_val is None else bt_val

    # Horizons: prefer dmh_backtest_horizons if set (list), else dmh_horizons, else 1..H
    base_horizons = list(getattr(config.design, "dmh_horizons", []) or []) or list(
        range(1, int(config.forecast_horizon) + 1)
    )
    if _in_backtest:
        bt_horizons = list(getattr(config.design, "dmh_backtest_horizons", []) or [])
        horizons = bt_horizons if bt_horizons else base_horizons
    else:
        horizons = base_horizons

    # Objective: override at backtest if dmh_backtest_objective is set
    base_objective = getattr(config.design, "dmh_objective", "auto")
    if _in_backtest:
        bt_obj = str(getattr(config.design, "dmh_backtest_objective", "") or "").strip()
        objective = bt_obj if bt_obj else base_objective
    else:
        objective = base_objective

    # n_seeds: override at backtest if dmh_backtest_n_seeds > 0
    base_n_seeds = int(getattr(config.design, "dmh_n_seeds", 1))
    if _in_backtest:
        bt_seeds = int(getattr(config.design, "dmh_backtest_n_seeds", 0) or 0)
        n_seeds = bt_seeds if bt_seeds > 0 else base_n_seeds
    else:
        n_seeds = base_n_seeds

    # top_k_features: override at backtest if dmh_backtest_top_k_features >= 0
    base_top_k = int(getattr(config.design, "dmh_top_k_features", 0))
    if _in_backtest:
        bt_top_k = int(getattr(config.design, "dmh_backtest_top_k_features", -1))
        top_k_features = bt_top_k if bt_top_k >= 0 else base_top_k
    else:
        top_k_features = base_top_k

    if _in_backtest:
        logger.info(
            "DMH backtest overrides: horizons=%s, objective=%s, n_seeds=%d, top_k=%d, augmenters=%s",
            horizons, objective, n_seeds, top_k_features,
            "on" if (enable_traj or enable_apg or enable_fwd) else "off",
        )

    cfg = DirectMHConfig(
        key_col=key_col,
        time_col=date_col,
        target_col=target_col,
        horizons=horizons,
        objective=objective,
        tweedie_variance_power=float(
            getattr(config.design, "dmh_tweedie_variance_power", 1.3)
        ),
        learning_rate=float(getattr(config.design, "dmh_learning_rate", 0.05)),
        num_leaves=int(getattr(config.design, "dmh_num_leaves", 63)),
        min_data_in_leaf=int(getattr(config.design, "dmh_min_data_in_leaf", 200)),
        lambda_l2=float(getattr(config.design, "dmh_lambda_l2", 1.0)),
        num_boost_round=int(getattr(config.design, "dmh_num_boost_round", 3000)),
        early_stopping_rounds=int(getattr(config.design, "dmh_early_stopping_rounds", 150)),
        enable_bias_calibration=bool(config.design.apply_bias_calibration),
        recency_halflife_weeks=float(
            getattr(config.design, "dmh_recency_halflife_weeks", 26.0)
        ),
        sply_blend_alpha=float(getattr(config.design, "dmh_sply_blend_alpha", 1.0)),
        n_seeds=n_seeds,
        top_k_features=top_k_features,
        smooth_window=int(getattr(config.design, "dmh_smooth_window", 0)),
        # Driver-side parallelism across horizons. 0 → auto (half cpu_count(),
        # capped at len(horizons)). See DirectMHConfig.horizon_workers and the
        # schema description for the full rationale.  Identical numerical
        # outputs since horizons are independent; speedup ~ horizon_workers.
        horizon_workers=int(getattr(config.design, "dmh_horizon_workers", 0)),
    )

    # --- origin selection
    def _to_int_period(v):
        try:
            return int(str(v).replace("-", ""))
        except Exception:
            return None

    if backtest_origin is not None:
        origin_week = int(backtest_origin)
        logger.info(
            "DMH: rolling-origin backtest mode, origin=%s (train+val+test combined "
            "as training panel, per-horizon filter: origin+h <= %s)",
            origin_week, origin_week,
        )
    else:
        origin_week = _to_int_period(config.val_end)
        if origin_week is None:
            raise ValueError(
                f"Could not parse config.val_end={config.val_end!r} into YYYYWW int"
            )
        logger.info(
            "DMH: forward inference mode, origin=%s (val_end)", origin_week,
        )

    # --- train + predict
    logger.info(
        "DMH: training %d horizon heads with objective=%s, recency=%.1f",
        len(cfg.horizons),
        cfg.objective,
        cfg.recency_halflife_weeks,
    )
    # In backtest mode, include test_df in the training panel so later origins
    # can train on earlier test weeks whose actuals would legitimately be
    # observable by that origin.  In forward-inference mode, train + val only.
    extra_features = test_df if backtest_origin is not None else None
    artifacts = train_direct_multihorizon(
        train_df, val_df, cfg,
        key_to_segment=key_to_segment,
        extra_features=extra_features,
        backtest_origin=origin_week if backtest_origin is not None else None,
    )

    forecasts = predict_direct_multihorizon(
        test_df,
        artifacts,
        cfg,
        origin_week=origin_week,
        key_to_segment=key_to_segment,
        apply_calibration=cfg.enable_bias_calibration,
        fallback_features=val_df,
    )

    # --- persist artifacts for reproducibility
    dmh_dir = os.path.join(output_dir, "direct_mh")
    try:
        artifacts.save(dmh_dir)
        with open(os.path.join(dmh_dir, "direct_mh_training_cfg.json"), "w") as f:
            _json.dump(
                {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.__dict__.items()},
                f,
                default=str,
                indent=2,
            )
    except Exception as e:
        logger.warning("DMH: could not save artifacts to %s: %s", dmh_dir, e)

    # --- shape into the forward-forecast schema used by the rest of inference
    # All period columns are coerced to str so that the output is compatible
    # with the schema produced by `create_dead_key_forecasts` and
    # `batch_recursive_forecast` - pyarrow/parquet writers require a single
    # dtype per column, and the downstream `pd.concat` with dead-key
    # forecasts would otherwise produce mixed int/str object columns.
    origin_period_str = str(origin_week)

    # Build the per-row model_params JSON string DEFENSIVELY.  Two
    # historical failure modes:
    #   1. Some `cfg.objective` values are objects (auto-ensemble dicts,
    #      LightGBM Booster refs, etc.) that json.dumps trips on with
    #      "Circular reference detected" — bubbling up to the outer
    #      catch in run_inference_pipeline and forcing a recursive
    #      fallback that costs hours.
    #   2. numpy scalars (np.int64, np.float64) coming back from cfg
    #      attribute access aren't always JSON-serialisable on older
    #      Python/json combinations.
    # Coerce to native primitives, and on any failure log loudly and
    # substitute a minimal placeholder so the DataFrame build still
    # succeeds — model_params is metadata, not a forecast input.
    try:
        _model_params_str = _json.dumps(
            {
                "objective": str(cfg.objective),
                "num_leaves": int(cfg.num_leaves),
                "min_data_in_leaf": int(cfg.min_data_in_leaf),
                "recency_halflife_weeks": float(cfg.recency_halflife_weeks),
            },
            default=str,
        )
    except Exception as _mp_exc:
        logger.warning(
            "DMH: model_params JSON serialisation failed (%s: %s); "
            "using minimal placeholder so the DMH path can still complete",
            type(_mp_exc).__name__, _mp_exc,
        )
        _model_params_str = '{"objective":"direct_mh","note":"params serialisation failed"}'

    out = pd.DataFrame(
        {
            key_col: forecasts[cfg.key_col].astype(str).values,
            date_col: forecasts["target_week"].astype(str).values,
            "predicted": forecasts["predicted"].astype(float).values,
            "origin_period": [origin_period_str] * len(forecasts),
            "SnapshotTimePeriod": [origin_period_str] * len(forecasts),
            "forecast_step": forecasts["horizon"].astype(int).values,
            "lag": forecasts["horizon"].astype(int).values,
            "model_level": ["direct_mh"] * len(forecasts),
            "model_name": [f"direct_mh_{cfg.objective}"] * len(forecasts),
            "model_params": [_model_params_str] * len(forecasts),
            "is_new_key": [False] * len(forecasts),
            "is_dead_key": [False] * len(forecasts),
        }
    )

    # --- populate real `actual` from the combined feature panel.  DMH trains
    # on features that contain the target column; we merge those values back
    # onto the output on (key, target_week) so backtest WMAPE/bias metrics
    # work end-to-end without the caller doing the merge themselves.  Missing
    # rows (forward-inference forecast weeks that have no actual yet) get 0.
    try:
        combined_hist = pd.concat(
            [train_df, val_df, test_df], ignore_index=True, sort=False
        )
        if target_col in combined_hist.columns:
            actual_lookup = (
                combined_hist[[key_col, date_col, target_col]]
                .assign(**{
                    key_col: lambda d: d[key_col].astype(str),
                    date_col: lambda d: d[date_col].astype(str),
                })
                .drop_duplicates(subset=[key_col, date_col], keep="last")
                .rename(columns={target_col: "_actual_lookup"})
            )
            out = out.merge(actual_lookup, on=[key_col, date_col], how="left")
            out["actual"] = pd.to_numeric(out["_actual_lookup"], errors="coerce").fillna(0.0).astype(float)
            out = out.drop(columns=["_actual_lookup"])
        else:
            out["actual"] = 0.0
    except Exception as _e:
        logger.warning("DMH: could not merge actuals (%s); defaulting to 0", _e)
        out["actual"] = 0.0

    # Belt-and-braces: force final dtypes.
    for _str_col in ("origin_period", "SnapshotTimePeriod", "model_level",
                      "model_name", "model_params", date_col):
        out[_str_col] = out[_str_col].astype(str)
    return out, len(out)


# =============================================================================
# PHASE 6: GENERATE FORWARD FORECASTS
# =============================================================================

def generate_forward_forecasts(
    config: DemandForecastConfig,
    retrained_models: Dict[str, Any],
    manifest_df: pd.DataFrame,
    model_specs: Dict[str, Any],
    feature_dir: str,
    output_dir: str,
    apply_bias_calibration: bool = False,
    calibration_factors: Optional[Dict[str, Any]] = None,
    segment_calibrations: Optional[Dict[str, Any]] = None,
    source_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Generate recursive forward forecasts for all keys over the test period.

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object
    retrained_models : Dict
        Retrained models {model_level: model_info}
    manifest_df : pd.DataFrame
        Updated training manifest
    model_specs : Dict
        Model specifications
    feature_dir : str
        Directory containing feature files
    output_dir : str
        Output directory for forecasts
    apply_bias_calibration : bool
        Whether to apply bias calibration
    calibration_factors : Dict, optional
        Basic bias calibration factors (from bias_calibration.json)
    segment_calibrations : Dict, optional
        STATE-OF-THE-ART segment-aware calibration factors (from segment_calibrations.json)

    Returns
    -------
    Tuple[pd.DataFrame, int]
        - DataFrame with all forecasts
        - Total number of forecasts generated
    """
    key_col = config.prediction_key_cols[0] if len(config.prediction_key_cols) == 1 else 'key'
    date_col = config.timestamp_col
    forecast_horizon = config.forecast_horizon

    # Load feature files once (format-agnostic: parquet preferred,
    # CSV fallback).
    from utils.feature_io import read_features_intermediate
    train_df = read_features_intermediate(feature_dir, "train_features")
    val_df   = read_features_intermediate(feature_dir, "val_features")
    test_df  = read_features_intermediate(feature_dir, "test_features")

    # Combine all data for historical lookback
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    del train_df, val_df  # Free memory early
    full_df = full_df.sort_values([key_col, date_col])

    # Get test periods and origin
    test_periods = sorted(test_df[date_col].unique())
    del test_df  # Free memory
    logger.info(f"Test periods: {len(test_periods)} ({test_periods[0]} to {test_periods[-1]})")

    origin_period = test_periods[0]
    forecast_periods = test_periods[:forecast_horizon]

    # History cutoff: everything strictly before the origin period
    # We use < origin semantics, so set cutoff to origin - 1 step
    # The batch engine uses <= cutoff, so we need the last period before origin
    date_col_dtype = full_df[date_col].dtype
    if pd.api.types.is_float_dtype(date_col_dtype) or pd.api.types.is_integer_dtype(date_col_dtype):
        all_periods_sorted = sorted(full_df[date_col].unique())
        origin_val = float(origin_period) if pd.api.types.is_float_dtype(date_col_dtype) else int(origin_period)
        periods_before_origin = [p for p in all_periods_sorted if p < origin_val]
        history_cutoff = periods_before_origin[-1] if periods_before_origin else origin_val
    else:
        all_periods_sorted = sorted(full_df[date_col].unique())
        periods_before_origin = [p for p in all_periods_sorted if str(p) < str(origin_period)]
        history_cutoff = periods_before_origin[-1] if periods_before_origin else origin_period

    # =========================================================================
    # INTELLIGENT FEATURE IMPUTATION (if enabled)
    # =========================================================================
    target_col = config.target_col
    tf = getattr(config, 'time_format', 'year_week')
    if tf in ('auto', 'date'):
        tf = 'year_week'

    key_fallback_features = None
    imputation_report = None

    if config.design.enable_feature_imputation:
        logger.info("[IMPUTATION] Intelligent feature imputation enabled")

        # Get feature columns from model specs
        feature_columns = model_specs.get('feature_columns', [])
        available_features = [c for c in feature_columns if c in full_df.columns]

        # Identify recursive columns (these are updated in the loop, not imputed here)
        lag_cols = _identify_lag_cols(available_features)
        roll_cols = _identify_rolling_cols(available_features)
        derived_cols = _identify_derived_cols(available_features, time_format=tf)
        sparsity_cols = _identify_sparsity_cols(available_features, time_format=tf)

        lag_col_names = {c[0] for c in lag_cols}
        roll_col_names = {c[0] for c in roll_cols}
        derived_col_names = {c[0] for c in derived_cols}
        sparsity_col_names = {c[0] for c in sparsity_cols}

        # Classify features
        feature_classification = _classify_feature_columns(
            feature_cols=available_features,
            config=config,
            lag_col_names=lag_col_names,
            roll_col_names=roll_col_names,
            derived_col_names=derived_col_names,
            sparsity_col_names=sparsity_col_names,
        )

        # Passthrough features: external numeric + categorical + key static
        passthrough_features = (
            feature_classification.get('external_numeric', [])
            + feature_classification.get('categorical', [])
            + feature_classification.get('key_static', [])
        )

        if passthrough_features:
            # Extract forecast-period rows for detection
            forecast_df = full_df[full_df[date_col].isin(forecast_periods)]

            # Phase 1: Detect which features need imputation per period
            features_to_impute = detect_features_needing_imputation(
                forecast_df=forecast_df,
                key_col=key_col,
                date_col=date_col,
                feature_cols=passthrough_features,
                forecast_periods=forecast_periods,
                threshold=config.design.imputation_missing_threshold,
            )

            if features_to_impute:
                n_feats = len(set(f for feats in features_to_impute.values() for f in feats))
                n_periods = len(features_to_impute)
                logger.info(f"[IMPUTATION] Detected {n_feats} features needing imputation "
                            f"across {n_periods} periods")

                # Detect key-invariant categoricals
                hist_df = full_df[full_df[date_col] <= history_cutoff]
                key_invariant_cats = _detect_key_invariant_categoricals(
                    history_df=hist_df,
                    key_col=key_col,
                    categorical_cols=feature_classification.get('categorical', []),
                    min_history=config.design.imputation_min_history_for_invariant,
                )

                if key_invariant_cats:
                    n_keys_with_inv = len(key_invariant_cats)
                    logger.info(f"[IMPUTATION] {n_keys_with_inv} keys have invariant categoricals")

                # Phase 2: Apply tiered imputation
                full_df, imputation_report = impute_forecast_features(
                    full_df=full_df,
                    forecast_periods=forecast_periods,
                    history_cutoff=history_cutoff,
                    features_to_impute=features_to_impute,
                    key_col=key_col,
                    date_col=date_col,
                    target_col=target_col,
                    time_format=tf,
                    manifest_df=manifest_df,
                    config=config,
                    feature_classification=feature_classification,
                    key_invariant_cats=key_invariant_cats,
                    source_df=source_df,
                )

                # Log summary
                if imputation_report:
                    for feat, stats in imputation_report.get('per_feature', {}).items():
                        total = sum(stats.values())
                        if total > 0:
                            logger.info(
                                f"[IMPUTATION]   {feat}: T1={stats['tier1_sply']} "
                                f"T2={stats['tier2_hierarchy']} inv={stats['invariant']} "
                                f"blank={stats['kept_blank']} unfilled={stats['unfilled']}"
                            )
            else:
                logger.info("[IMPUTATION] No passthrough features flagged for imputation")
        else:
            logger.info("[IMPUTATION] No passthrough features found in feature set")

        # Phase 3: Build recursive fallback maps for short-history keys
        if config.design.imputation_enable_recursive_fallbacks:
            key_fallback_features = build_recursive_feature_fallbacks(
                full_df=full_df,
                key_col=key_col,
                date_col=date_col,
                target_col=target_col,
                history_cutoff=history_cutoff,
                time_format=tf,
                manifest_df=manifest_df,
                lag_cols=lag_cols,
                derived_cols=derived_cols,
                config=config,
                forecast_periods=forecast_periods,
            )
            if key_fallback_features:
                logger.info(f"[IMPUTATION] Built recursive fallbacks for {len(key_fallback_features)} short-history keys")

        # Save imputation report
        if imputation_report and config.design.imputation_log_details:
            try:
                report_path = save_imputation_report(imputation_report, output_dir)
                logger.info(f"[IMPUTATION] Report saved: {report_path}")
            except Exception as e:
                logger.warning(f"[IMPUTATION] Failed to save report: {e}")

    # Delegate to batch engine
    return batch_recursive_forecast(
        config=config,
        retrained_models=retrained_models,
        manifest_df=manifest_df,
        model_specs=model_specs,
        full_df=full_df,
        forecast_periods=forecast_periods,
        origin_period=origin_period,
        snapshot_period=origin_period,
        history_cutoff=history_cutoff,
        apply_bias_calibration=apply_bias_calibration,
        calibration_factors=calibration_factors,
        segment_calibrations=segment_calibrations,
        max_workers=2,
        key_fallback_features=key_fallback_features,
    )


# =============================================================================
# BATCH RECURSIVE FORECAST ENGINE
# =============================================================================

def _forecast_stats_univariate(model: Any, model_type: str, hist_arr: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Generate forecasts from statistical time series models (ARIMA, ETS, Theta, TBATS, Prophet, BSTS).

    These models use their native forecast API (not sklearn-style predict(X)).
    Falls back to mean of history if the model's forecast API fails.
    """
    try:
        if model_type in ('arima', 'sarima'):
            # pmdarima auto_arima: model.predict(n_periods=N)
            preds = model.predict(n_periods=n_steps)
            return np.clip(np.asarray(preds, dtype=float).flatten()[:n_steps], 0, None)

        elif model_type in ('ets', 'theta'):
            # statsmodels ExponentialSmoothing / ThetaModel: model.forecast(N)
            preds = model.forecast(n_steps)
            return np.clip(np.asarray(preds, dtype=float).flatten()[:n_steps], 0, None)

        elif model_type == 'tbats':
            if hasattr(model, 'forecast'):
                preds = model.forecast(steps=n_steps)
            else:
                preds = model.predict(n_periods=n_steps)
            return np.clip(np.asarray(preds, dtype=float).flatten()[:n_steps], 0, None)

        elif model_type == 'prophet':
            # Facebook Prophet needs a DataFrame with 'ds' column
            last_date = pd.Timestamp('2025-01-01')
            future = pd.date_range(start=last_date, periods=n_steps + 1, freq='W')[1:]
            future_df = pd.DataFrame({'ds': future})
            forecast = model.predict(future_df)
            preds = forecast['yhat'].values[:n_steps]
            return np.clip(np.asarray(preds, dtype=float), 0, None)

        elif model_type == 'bsts':
            # orbit-ml DLT model
            last_date = pd.Timestamp('2025-01-01')
            future = pd.date_range(start=last_date, periods=n_steps + 1, freq='W')[1:]
            future_df = pd.DataFrame({'ds': future})
            pred_df = model.predict(future_df)
            col = 'prediction' if 'prediction' in pred_df.columns else pred_df.columns[0]
            preds = pred_df[col].values[:n_steps]
            return np.clip(np.asarray(preds, dtype=float), 0, None)

    except Exception as e:
        logger.debug(f"Stats univariate forecast failed for {model_type}: {e}")

    # Fallback: repeat mean of history
    fallback_val = max(0.0, float(np.mean(hist_arr))) if len(hist_arr) > 0 else 0.0
    return np.full(n_steps, fallback_val)


def _batch_predict_with_model(model: Any, X_batch: np.ndarray, model_type: str = None, horizon: int = None, keys: list = None) -> np.ndarray:
    """
    Batch prediction for feature-based models. Returns array of predictions.

    Handles the same model types as _predict_with_model but operates on
    a batch of samples (N_keys x N_features) returning N_keys predictions.

    Parameters
    ----------
    model : Any
        Trained model (may be HierarchicalModelWrapper, QuantileModelWrapper, etc.)
    X_batch : np.ndarray
        Feature matrix (n_samples, n_features)
    model_type : str, optional
        Model type string for dispatch
    horizon : int, optional
        Forecast horizon for multi-horizon models
    keys : list, optional
        Key identifiers per row (required for hierarchical models)
    """
    try:
        n_samples = X_batch.shape[0]

        # Phase 3/7/8: Hierarchical/enhanced/combination models with keys= support
        if hasattr(model, 'model_type') and model.model_type in (
            'global_local', 'mixed_effects', 'multi_level_ensemble',
            'catboost_embedding', 'quantile_regression', 'conformal_boost',
            'stacked_ensemble',
        ):
            preds = model.predict(X_batch, keys=keys)
            if isinstance(preds, np.ndarray):
                return np.maximum(preds.flatten()[:n_samples], 0)
            return np.maximum(np.full(n_samples, float(preds)), 0)

        # Multi-horizon models
        if model_type and model_type.startswith('multi_horizon'):
            if hasattr(model, 'predict') and hasattr(model, 'target_horizon'):
                pred = model.predict(X_batch, horizon=horizon)
                if isinstance(pred, np.ndarray):
                    return pred.flatten()[:n_samples]
                return np.full(n_samples, float(pred))

        if hasattr(model, 'target_horizon') and hasattr(model, 'max_horizon') and hasattr(model, 'strategy'):
            pred = model.predict(X_batch, horizon=horizon)
            if isinstance(pred, np.ndarray):
                return pred.flatten()[:n_samples]
            return np.full(n_samples, float(pred))

        # Dict-based models
        if isinstance(model, dict):
            if 'classifier' in model and 'unique_values' in model:
                from utils.model_selection_intelligence import snap_to_discrete_values
                classifier = model['classifier']
                unique_values = model['unique_values']
                if 'class_to_value' in model:
                    pred_classes = classifier.predict(X_batch)
                    class_to_value = model['class_to_value']
                    return np.array([class_to_value.get(int(c), unique_values[0]) for c in pred_classes], dtype=float)
                else:
                    probs = classifier.predict_proba(X_batch)
                    pred = np.sum(probs * unique_values, axis=1)
                    snap = model.get('snap_to_valid', True)
                    if snap:
                        pred = snap_to_discrete_values(pred, unique_values, method='nearest')
                    return pred.astype(float)

            elif 'regressor' in model and 'unique_values' in model:
                from utils.model_selection_intelligence import snap_to_discrete_values
                regressor = model['regressor']
                unique_values = model['unique_values']
                pred_cont = regressor.predict(X_batch)
                pred = snap_to_discrete_values(pred_cont, unique_values, method='nearest')
                return pred.astype(float)

            elif model.get('type') == 'weighted_ensemble':
                component_models = model.get('component_models', [])
                weights = model.get('weights', [])
                if not component_models or not weights:
                    return np.zeros(n_samples)
                weighted_pred = np.zeros(n_samples)
                for (mt, comp_model), weight in zip(component_models, weights):
                    try:
                        if isinstance(comp_model, dict):
                            comp_pred = np.full(n_samples, float(comp_model.get('forecast', 0.0)))
                        elif hasattr(comp_model, 'predict'):
                            cp = comp_model.predict(X_batch)
                            comp_pred = cp.flatten()[:n_samples] if isinstance(cp, np.ndarray) else np.full(n_samples, float(cp))
                        else:
                            continue
                        weighted_pred += weight * comp_pred
                    except Exception:
                        pass
                return weighted_pred

            elif model.get('type') == 'constant':
                return np.full(n_samples, float(model.get('value', 0.0)))

            elif model.get('type') in ('croston', 'sba', 'tsb', 'imapa'):
                return np.full(n_samples, float(model.get('forecast', 0.0)))

            else:
                val = float(model.get('forecast', model.get('value', 0.0)))
                return np.full(n_samples, val)

        # XGBoost Booster
        elif hasattr(model, 'save_model') and not hasattr(model, 'fit'):
            import xgboost as xgb
            dmat = xgb.DMatrix(X_batch)
            return model.predict(dmat)

        # Standard sklearn-like
        elif hasattr(model, 'predict'):
            pred = model.predict(X_batch)
            if isinstance(pred, np.ndarray):
                return pred.flatten()[:n_samples]
            return np.full(n_samples, float(pred))

    except Exception as e:
        logger.debug(f"Batch prediction error: {e}")

    return np.zeros(X_batch.shape[0])


def _compute_ewm_direct(vals: List[float], span: int) -> float:
    """Fast EWM mean using direct alpha math (no pd.Series overhead)."""
    if not vals:
        return 0.0
    alpha = 2.0 / (span + 1)
    ewm = vals[0]
    for v in vals[1:]:
        ewm = alpha * v + (1 - alpha) * ewm
    return ewm


def _compute_ewm_std_direct(vals: List[float], span: int) -> float:
    """Fast EWM std using direct alpha math (no pd.Series overhead)."""
    if len(vals) <= 1:
        return 0.0
    alpha = 2.0 / (span + 1)
    # Two-pass: compute EWM mean first, then EWM variance
    ewm_mean = vals[0]
    ewm_var = 0.0
    for v in vals[1:]:
        diff = v - ewm_mean
        ewm_mean = alpha * v + (1 - alpha) * ewm_mean
        ewm_var = (1 - alpha) * (ewm_var + alpha * diff * diff)
    return max(0.0, ewm_var) ** 0.5


def _compute_rolling_agg_fast(vals: List[float], window: int, agg: str, is_log: bool = False) -> float:
    """Compute rolling aggregate — fast version using direct math for EWM."""
    if len(vals) < 1:
        return 0.0

    window_vals = vals[-min(window, len(vals)):]

    if is_log:
        window_vals = [np.log1p(max(0, v)) for v in window_vals]

    if agg == 'mean':
        return np.mean(window_vals)
    elif agg == 'ewm_mean':
        return _compute_ewm_direct(window_vals, window)
    elif agg == 'std':
        return np.std(window_vals) if len(window_vals) > 1 else 0.0
    elif agg == 'ewm_std':
        return _compute_ewm_std_direct(window_vals, window)
    elif agg == 'sum':
        return np.sum(window_vals)
    elif agg == 'min':
        return np.min(window_vals)
    elif agg == 'max':
        return np.max(window_vals)
    elif agg == 'median':
        return np.median(window_vals)
    return np.mean(window_vals) if window_vals else 0.0


def _forecast_model_level_keys(
    keys: List[str],
    model_info: Dict[str, Any],
    key_groups: Dict[str, pd.DataFrame],
    full_df_forecast_groups: Dict[str, pd.DataFrame],
    target_col: str,
    date_col: str,
    origin_period: Any,
    snapshot_period: Any,
    forecast_horizon: int,
    available_features: List[str],
    lag_cols: list,
    roll_cols: list,
    derived_cols: list,
    sparsity_cols: list,
    apply_bias_calibration: bool,
    calibration_factors: Optional[Dict],
    segment_calibrations: Optional[Dict],
    key_to_segment_id: Dict[str, str],
    new_keys_set: set,
    model_level: str,
    model_params_str: str,
    forecast_periods: list,
    key_fallback_features: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict]:
    """
    Generate recursive forecasts for all keys assigned to a single model level.

    For feature-based models, at each forecast step:
    1. Update lag/rolling/derived/sparsity features per-key (must be per-key due to recursive history)
    2. Batch-predict all keys in one model.predict() call
    3. Append predictions to each key's history for next step

    This is the core performance improvement: one predict() call for N keys
    instead of N predict() calls.
    """
    from utils.model_training import _recursive_forecast_univariate

    model = model_info['model']
    model_type = model_info['model_type']
    forecasts = []

    # -------------------------------------------------------------------------
    # UNIVARIATE MODELS — no features needed, forecast from history only
    # -------------------------------------------------------------------------
    # Croston-family dict models: use _recursive_forecast_univariate
    is_croston_family = isinstance(model, dict) and model.get('type') in ('croston', 'sba', 'tsb', 'imapa')

    # Statistical time series models: ARIMA, ETS, Theta, TBATS, Prophet, BSTS
    # These have .forecast() or .predict(n_periods=) API, not sklearn .predict(X)
    _STATSMODEL_TYPES = {'arima', 'sarima', 'ets', 'theta', 'tbats', 'prophet', 'bsts'}
    is_stats_univariate = model_type.lower() in _STATSMODEL_TYPES and not isinstance(model, dict)

    # Dict fallback models from failed training (e.g. {'type': 'arima_fallback', 'mean': 5.0})
    is_fallback_dict = isinstance(model, dict) and model.get('type', '').endswith('_fallback')

    if is_croston_family or is_stats_univariate or is_fallback_dict:
        for key in keys:
            key_hist_df = key_groups.get(key)
            if key_hist_df is None or len(key_hist_df) == 0:
                continue

            historical_actuals = key_hist_df[target_col].values.tolist()
            key_fc_df = full_df_forecast_groups.get(key)
            if key_fc_df is None or len(key_fc_df) == 0:
                continue

            hist_arr = np.array(historical_actuals, dtype=float)
            n_steps = min(len(key_fc_df), forecast_horizon)

            # Generate predictions based on model type
            if is_croston_family:
                uni_preds = _recursive_forecast_univariate(model, hist_arr, n_steps)
            elif is_fallback_dict:
                # Fallback dicts just repeat the mean
                fallback_val = float(model.get('mean', model.get('forecast', model.get('value', 0.0))))
                uni_preds = np.full(n_steps, max(0.0, fallback_val))
            else:
                # Statistical models: use their native forecast API
                uni_preds = _forecast_stats_univariate(model, model_type.lower(), hist_arr, n_steps)

            is_new_key = key in new_keys_set

            # Compute zero_fraction once for calibration
            key_zero_fraction = None
            if apply_bias_calibration and (calibration_factors or segment_calibrations):
                key_zero_fraction = (key_hist_df[target_col] == 0).mean()

            for step_idx, (_, row) in enumerate(key_fc_df.iterrows()):
                if step_idx >= n_steps:
                    break
                pred = float(uni_preds[step_idx])

                if apply_bias_calibration and (calibration_factors or segment_calibrations):
                    key_segment_id = key_to_segment_id.get(key, model_level)
                    pred = _apply_bias_calibration(
                        pred, key, key_segment_id, calibration_factors,
                        segment_calibrations=segment_calibrations,
                        zero_fraction=key_zero_fraction,
                        lag=step_idx if 'step_idx' in dir() else None,
                        lag_calibration_factors=lag_calibration_factors if 'lag_calibration_factors' in dir() else None,
                    )

                actual_value = row.get(target_col, 0)
                if pd.isna(actual_value):
                    actual_value = 0.0
                else:
                    actual_value = float(actual_value)

                forecasts.append({
                    'key': key,
                    date_col: row[date_col],
                    'origin_period': origin_period,
                    'SnapshotTimePeriod': snapshot_period,
                    'forecast_step': step_idx + 1,
                    'lag': step_idx,
                    'predicted': pred,
                    'actual': actual_value,
                    'model_level': model_level,
                    'model_name': model_type,
                    'model_params': model_params_str,
                    'is_new_key': is_new_key,
                    'is_dead_key': False,
                })
        return forecasts

    # -------------------------------------------------------------------------
    # FEATURE-BASED MODELS — batch predict at each step
    # -------------------------------------------------------------------------

    # Pre-filter to valid keys (have history and forecast data)
    valid_keys = []
    key_histories = {}       # key -> list of actuals
    key_fc_data = {}         # key -> DataFrame of forecast rows
    key_base_features = {}   # key -> 2D numpy array (steps x features)
    key_zero_fractions = {}  # key -> float (for calibration)

    for key in keys:
        key_hist_df = key_groups.get(key)
        if key_hist_df is None or len(key_hist_df) == 0:
            continue

        key_fc_df = full_df_forecast_groups.get(key)
        if key_fc_df is None or len(key_fc_df) == 0:
            continue

        avail_cols = [c for c in available_features if c in key_fc_df.columns]
        if not avail_cols:
            continue

        valid_keys.append(key)
        key_histories[key] = key_hist_df[target_col].values.tolist()
        key_fc_data[key] = key_fc_df
        key_base_features[key] = key_fc_df[avail_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        # Always compute zero fractions — needed for zero-snapping and calibration
        key_zero_fractions[key] = (key_hist_df[target_col] == 0).mean()

    if not valid_keys:
        return forecasts

    # Column index mapping (same for all keys since features are identical)
    sample_fc = key_fc_data[valid_keys[0]]
    avail_cols = [c for c in available_features if c in sample_fc.columns]
    col_to_idx = {c: i for i, c in enumerate(avail_cols)}

    # Pre-filter feature update columns to only those that exist
    active_lag_cols = [(col, lag_num, is_log, is_binary) for col, lag_num, is_log, is_binary in lag_cols if col in col_to_idx]
    active_roll_cols = [(col, window, agg, is_log) for col, window, agg, is_log in roll_cols if col in col_to_idx]
    active_derived_cols = [(col, feat_type, param, is_log) for col, feat_type, param, is_log in derived_cols if col in col_to_idx]
    active_sparsity_cols = [(col, feat_type, param) for col, feat_type, param in sparsity_cols if col in col_to_idx]

    # Determine max steps — use the maximum across keys so short keys don't truncate others
    per_key_steps = {k: min(forecast_horizon, len(key_base_features[k])) for k in valid_keys}
    max_steps = max(per_key_steps.values()) if per_key_steps else 0

    # Sprint 3 B3: Initialise error correctors from training residuals
    error_correctors = {}
    try:
        from utils.sprint3_features import RecursiveErrorCorrector
        # Compute training residuals: for each key, diff between last few actuals and model predictions
        for key in valid_keys:
            hist = key_histories.get(key, [])
            if len(hist) >= 10:
                recent = np.array(hist[-10:])
                # Simple residual estimate: recent mean vs overall trend
                recent_mean = np.mean(recent[-4:])
                older_mean = np.mean(recent[-8:-4]) if len(recent) >= 8 else recent_mean
                residual = recent_mean - older_mean  # Recent deviation from trend
                error_correctors[key] = RecursiveErrorCorrector(alpha=0.2, initial_correction=residual * 0.3)
    except Exception:
        error_correctors = {}

    # Step-by-step recursive forecasting with BATCH prediction
    for step_idx in range(max_steps):
        # Determine which keys are still active at this step
        active_keys_at_step = [k for k in valid_keys if step_idx < per_key_steps[k]]
        if not active_keys_at_step:
            break

        # Build feature matrix for active keys at this step
        n_active = len(active_keys_at_step)
        n_features = len(avail_cols)
        X_batch = np.zeros((n_active, n_features), dtype=np.float64)

        for i, key in enumerate(active_keys_at_step):
            features = key_base_features[key][step_idx].copy()
            hist = key_histories[key]
            fallbacks = key_fallback_features.get(key, {}) if key_fallback_features else {}

            # Update lag features (with fallback for short-history keys)
            for col, lag_num, is_log, is_binary in active_lag_cols:
                idx = col_to_idx[col]
                val = _compute_lag_value(hist, lag_num, is_log, is_binary)
                if val == 0.0 and len(hist) < lag_num and col in fallbacks:
                    val = fallbacks[col]
                features[idx] = val

            # Update rolling features (uses fast direct math for EWM)
            for col, window, agg, is_log in active_roll_cols:
                idx = col_to_idx[col]
                features[idx] = _compute_rolling_agg_fast(hist, window, agg, is_log)

            # Update derived features (with fallback for short-history keys)
            for col, feat_type, param, is_log in active_derived_cols:
                idx = col_to_idx[col]
                val = _compute_derived_feature(hist, feat_type, param, is_log)
                if val == 0.0 and col in fallbacks:
                    needed = 1 + param if feat_type in ('yoy_lag_diff', 'yoy_lag_ratio', 'qoq_lag_diff') else param + 1
                    if len(hist) < needed:
                        val = fallbacks[col]
                features[idx] = val

            # Update sparsity features
            for col, feat_type, param in active_sparsity_cols:
                idx = col_to_idx[col]
                features[idx] = _compute_sparsity_feature(hist, feat_type, param)

            X_batch[i] = features

        # BATCH PREDICT — one call for all active keys at this step
        current_horizon_step = step_idx + 1
        preds = _batch_predict_with_model(model, X_batch, model_type=model_type, horizon=current_horizon_step)
        # Floor near-zero numerical artifacts: values in (0, 0.5) are artifacts
        # from zero-inflated/hurdle models (e.g. 1.85e-15) that corrupt recursive
        # lag features. Force them to clean zero to prevent cascade errors.
        preds = np.where(preds < 0.5, 0.0, preds)

        # Apply calibration and store results
        for i, key in enumerate(active_keys_at_step):
            pred = float(preds[i])

            if apply_bias_calibration and (calibration_factors or segment_calibrations):
                key_segment_id = key_to_segment_id.get(key, model_level)
                pred = _apply_bias_calibration(
                    pred, key, key_segment_id, calibration_factors,
                    segment_calibrations=segment_calibrations,
                    zero_fraction=key_zero_fractions.get(key),
                    lag=step_idx,
                    lag_calibration_factors=lag_calibration_factors if 'lag_calibration_factors' in dir() else None,
                )

            # Sprint 3 B3: Error correction — adjust for accumulated prediction bias
            # Uses exponentially decaying correction from training residuals
            if 'error_correctors' in dir() and key in error_correctors:
                pred = error_correctors[key].adjust_prediction(pred)

            # Zero-snap: for intermittent keys, snap low predictions to zero.
            # Benchmark predicts 0 for 22.9% of rows; we only do 2.4%.
            # This is the #1 driver of the WAPE gap.
            zf = key_zero_fractions.get(key, 0.0)
            if zf > 0.3 and pred > 0:
                hist = key_histories[key]
                hist_nonzero = [v for v in hist if v > 0]
                if hist_nonzero:
                    hist_mean_nz = sum(hist_nonzero) / len(hist_nonzero)
                    # Threshold scales with zero-fraction:
                    #   zf=0.3 → threshold = 0.15 * mean (conservative)
                    #   zf=0.7 → threshold = 0.35 * mean (moderate)
                    #   zf=0.9 → threshold = 0.45 * mean (aggressive)
                    snap_threshold = zf * 0.5 * hist_mean_nz
                    if pred < snap_threshold:
                        pred = 0.0
                elif pred < 1.0:
                    # All-zero history — snap anything small
                    pred = 0.0

            # Clamp prediction to historical range before appending to recursive
            # history. Prevents runaway predictions from corrupting lag features.
            hist = key_histories[key]
            hist_nonzero = [abs(v) for v in hist if v != 0]
            hist_max = max(hist_nonzero) if hist_nonzero else 0.0
            if hist_max > 0:
                pred = min(pred, hist_max * 3.0)  # Allow up to 3x historical max
            pred = max(pred, 0.0)

            # Append to history for recursive update
            key_histories[key].append(pred)

            # Get actual value
            fc_df = key_fc_data[key]
            if step_idx < len(fc_df):
                row = fc_df.iloc[step_idx]
                actual_value = row.get(target_col, 0)
                if pd.isna(actual_value):
                    actual_value = 0.0
                else:
                    actual_value = float(actual_value)

                forecasts.append({
                    'key': key,
                    date_col: row[date_col],
                    'origin_period': origin_period,
                    'SnapshotTimePeriod': snapshot_period,
                    'forecast_step': step_idx + 1,
                    'lag': step_idx,
                    'predicted': pred,
                    'actual': actual_value,
                    'model_level': model_level,
                    'model_name': model_type,
                    'model_params': model_params_str,
                    'is_new_key': key in new_keys_set,
                    'is_dead_key': False,
                })

    return forecasts


def batch_recursive_forecast(
    config: DemandForecastConfig,
    retrained_models: Dict[str, Any],
    manifest_df: pd.DataFrame,
    model_specs: Dict[str, Any],
    full_df: pd.DataFrame,
    forecast_periods: list,
    origin_period: Any,
    snapshot_period: Any = None,
    history_cutoff: Any = None,
    apply_bias_calibration: bool = False,
    calibration_factors: Optional[Dict[str, Any]] = None,
    segment_calibrations: Optional[Dict[str, Any]] = None,
    max_workers: int = 8,
    key_fallback_features: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    High-performance batch recursive forecast engine.

    Used by BOTH inference (generate_forward_forecasts) and backtesting
    (_generate_origin_forecasts) to ensure consistent, optimized forecasting.

    Key optimizations:
    1. Pre-groups full_df by key using groupby (O(1) lookup vs O(N) filtering)
    2. Groups keys by model_level for batch prediction (1 predict call per segment per step)
    3. Uses direct alpha math for EWM (no pd.Series overhead)
    4. Pre-serializes json once per model level
    5. Parallelizes across model levels using joblib (max_workers cores)

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object
    retrained_models : Dict
        {model_level: {'model': model, 'model_type': str, 'hyperparameters': dict}}
    manifest_df : pd.DataFrame
        Training manifest with key, model_level, segment_id columns
    model_specs : Dict
        Model specifications with feature_columns
    full_df : pd.DataFrame
        Pre-loaded combined DataFrame (train + val + test), sorted by [key_col, date_col]
    forecast_periods : list
        Periods to forecast (already limited to forecast_horizon)
    origin_period : Any
        Forecast origin period (for output metadata)
    snapshot_period : Any, optional
        Snapshot period (defaults to first forecast period)
    history_cutoff : Any, optional
        Cutoff for history. If None, uses everything before first forecast period.
    apply_bias_calibration : bool
        Whether to apply bias calibration
    calibration_factors, segment_calibrations : Dict, optional
        Calibration data
    max_workers : int
        Max parallel workers (default 8)

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Forecast DataFrame and total forecast count
    """
    from joblib import Parallel, delayed

    target_col = config.target_col
    key_col = config.prediction_key_cols[0] if len(config.prediction_key_cols) == 1 else 'key'
    date_col = config.timestamp_col
    forecast_horizon = config.forecast_horizon

    if snapshot_period is None:
        snapshot_period = forecast_periods[0] if forecast_periods else origin_period

    if history_cutoff is None:
        history_cutoff = origin_period

    # Get feature columns and pad missing ones with zeros for model compatibility
    feature_columns = model_specs.get('feature_columns', [])
    available_features = [c for c in feature_columns if c in full_df.columns]
    missing_fc_features = [c for c in feature_columns if c not in full_df.columns]
    if missing_fc_features:
        logger.warning(f"Forecast: {len(missing_fc_features)} features missing, padding with zeros")
        for mf in missing_fc_features:
            full_df[mf] = 0.0
        available_features = feature_columns

    # Get unique keys from manifest
    unique_keys = manifest_df['key'].unique()
    logger.info(f"Batch forecast engine: {len(unique_keys)} keys, {len(forecast_periods)} periods, {len(retrained_models)} models")

    # Build key-to-model mapping
    key_to_model_level = dict(zip(manifest_df['key'], manifest_df['model_level'].astype(str)))

    # Build key-to-segment_id mapping for calibration
    key_to_segment_id = {}
    if 'segment_id' in manifest_df.columns:
        key_to_segment_id = dict(zip(manifest_df['key'], manifest_df['segment_id'].astype(str)))
    else:
        key_to_segment_id = key_to_model_level.copy()

    # Mark new keys
    new_keys_set = set()
    if 'allocation_rationale' in manifest_df.columns:
        new_keys_set = set(manifest_df[manifest_df['allocation_rationale'] == 'new_key_assigned_to_segment']['key'])

    # Identify feature update columns
    tf = getattr(config, 'time_format', 'year_week')
    if tf in ('auto', 'date'):
        tf = 'year_week'
    lag_cols = _identify_lag_cols(available_features)
    roll_cols = _identify_rolling_cols(available_features)
    derived_cols = _identify_derived_cols(available_features, time_format=tf)
    sparsity_cols = _identify_sparsity_cols(available_features, time_format=tf)

    logger.info(f"Features: {len(lag_cols)} lags, {len(roll_cols)} rolling, {len(derived_cols)} derived, {len(sparsity_cols)} sparsity")

    # =========================================================================
    # PRE-GROUP full_df by key — O(1) lookup instead of O(N) boolean filtering
    # =========================================================================
    # Handle date column dtype for comparisons
    date_col_dtype = full_df[date_col].dtype
    if pd.api.types.is_float_dtype(date_col_dtype):
        history_cutoff_val = float(history_cutoff)
        forecast_periods_val = [float(w) for w in forecast_periods]
    elif pd.api.types.is_integer_dtype(date_col_dtype):
        history_cutoff_val = int(history_cutoff)
        forecast_periods_val = [int(w) for w in forecast_periods]
    else:
        history_cutoff_val = history_cutoff
        forecast_periods_val = forecast_periods

    forecast_periods_set = set(forecast_periods_val)

    # Pre-group: split full_df into history and forecast per key
    grouped = full_df.groupby(key_col)
    key_history_groups = {}    # key -> DataFrame (history only, <= cutoff)
    key_forecast_groups = {}   # key -> DataFrame (forecast periods only)

    for key in unique_keys:
        try:
            kdf = grouped.get_group(key)
        except KeyError:
            continue
        hist = kdf[kdf[date_col] <= history_cutoff_val]
        if len(hist) == 0:
            continue
        fc = kdf[kdf[date_col].isin(forecast_periods_set)].sort_values(date_col)
        if len(fc) == 0:
            continue
        key_history_groups[key] = hist
        key_forecast_groups[key] = fc

    logger.info(f"Pre-grouped: {len(key_history_groups)} keys with history+forecast data")

    # =========================================================================
    # GROUP KEYS BY MODEL LEVEL for batch prediction
    # =========================================================================
    model_level_keys = {}  # model_level -> list of keys
    fallback_count = 0
    for key in key_history_groups:
        ml = key_to_model_level.get(key, 'default')
        if ml not in retrained_models:
            # Fallback: use the segment_id from manifest to find the closest model
            # (segment_id often equals model_level for segment-pooled models)
            seg_id = key_to_segment_id.get(key, '')
            if seg_id in retrained_models:
                ml = seg_id
            elif retrained_models:
                # Ultimate fallback: use the model with the most keys (likely the biggest segment)
                ml = max(retrained_models.keys(), key=lambda m: len(model_level_keys.get(m, [])))
                if not ml or ml not in retrained_models:
                    ml = list(retrained_models.keys())[0]
            else:
                continue
            fallback_count += 1
        if ml not in model_level_keys:
            model_level_keys[ml] = []
        model_level_keys[ml].append(key)
    if fallback_count > 0:
        logger.info(f"  {fallback_count} keys reassigned to fallback models (original model level missing)")

    # Pre-serialize json per model level
    model_params_cache = {}
    for ml, model_info in retrained_models.items():
        model_params_cache[ml] = json.dumps(model_info.get('hyperparameters', {}))

    logger.info(f"Model levels to forecast: {len(model_level_keys)} "
                f"(keys per level: {', '.join(f'{ml}={len(ks)}' for ml, ks in model_level_keys.items())})")

    # =========================================================================
    # PARALLEL FORECAST per model level
    # =========================================================================
    n_model_levels = len(model_level_keys)
    # Only parallelize if multiple model levels and enough keys to justify overhead
    use_parallel = n_model_levels > 1 and len(key_history_groups) > 100
    effective_workers = min(max_workers, n_model_levels) if use_parallel else 1

    # Use threading backend to avoid memory duplication.
    # loky (multiprocessing) serializes full_df + models into each worker process,
    # which can easily OOM on large datasets. Threading shares memory.
    parallel_backend = 'threading'

    # Build common kwargs dict — passed explicitly to avoid closure pickle issues with joblib
    common_kwargs = dict(
        target_col=target_col,
        date_col=date_col,
        origin_period=origin_period,
        snapshot_period=snapshot_period,
        forecast_horizon=forecast_horizon,
        available_features=available_features,
        lag_cols=lag_cols,
        roll_cols=roll_cols,
        derived_cols=derived_cols,
        sparsity_cols=sparsity_cols,
        apply_bias_calibration=apply_bias_calibration,
        calibration_factors=calibration_factors,
        segment_calibrations=segment_calibrations,
        key_to_segment_id=key_to_segment_id,
        new_keys_set=new_keys_set,
        key_fallback_features=key_fallback_features,
    )

    if use_parallel:
        logger.info(f"Parallel forecasting with {effective_workers} workers ({parallel_backend} backend) "
                     f"across {n_model_levels} model levels")
        # Subset key dicts per model level — avoids serializing ALL keys to every worker
        results = Parallel(n_jobs=effective_workers, backend=parallel_backend, verbose=0)(
            delayed(_forecast_model_level_keys)(
                keys=ml_keys,
                model_info=retrained_models[ml],
                key_groups={k: key_history_groups[k] for k in ml_keys if k in key_history_groups},
                full_df_forecast_groups={k: key_forecast_groups[k] for k in ml_keys if k in key_forecast_groups},
                model_level=ml,
                model_params_str=model_params_cache.get(ml, '{}'),
                forecast_periods=forecast_periods_val,
                **common_kwargs,
            )
            for ml, ml_keys in model_level_keys.items()
        )
    else:
        logger.info(f"Sequential forecasting across {n_model_levels} model level(s)")
        results = []
        for ml_idx, (ml, ml_keys) in enumerate(model_level_keys.items()):
            logger.info(f"  Model level {ml}: {len(ml_keys)} keys")
            result = _forecast_model_level_keys(
                keys=ml_keys,
                model_info=retrained_models[ml],
                key_groups=key_history_groups,
                full_df_forecast_groups=key_forecast_groups,
                model_level=ml,
                model_params_str=model_params_cache.get(ml, '{}'),
                forecast_periods=forecast_periods_val,
                **common_kwargs,
            )
            results.append(result)
            logger.info(f"  Completed {ml}: {len(result)} forecasts")

    # Combine all results
    all_forecasts = []
    for result in results:
        all_forecasts.extend(result)

    forecasts_df = pd.DataFrame(all_forecasts)
    total_forecasts = len(forecasts_df)

    logger.info(f"Generated {total_forecasts} forecasts for {len(key_history_groups)} keys")

    return forecasts_df, total_forecasts


# =============================================================================
# HELPER FUNCTIONS FOR RECURSIVE FORECASTING
# =============================================================================

def _identify_lag_cols(cols: List[str]) -> List[Tuple[str, int, bool, bool]]:
    """Identify lag columns: (col_name, lag_num, is_log, is_binary)."""
    lag_cols = []
    derived_patterns = ['lag_diff', 'lag_ratio', 'lag_pct_change', 'lag_direction', 'yoy_lag', 'qoq_lag']

    for col in cols:
        col_lower = col.lower()

        # Skip derived features
        if any(p in col_lower for p in derived_patterns):
            continue

        is_binary = 'demand_occurred' in col_lower

        match = re.search(r'lag[_]?(\d+)(?:[_]|$)', col, re.IGNORECASE)
        if match:
            lag_num = int(match.group(1))
            is_log = '_log_' in col_lower and 'lag' in col_lower
            lag_cols.append((col, lag_num, is_log, is_binary))

    return lag_cols


def _identify_rolling_cols(cols: List[str]) -> List[Tuple[str, int, str, bool]]:
    """Identify rolling columns: (col_name, window, agg, is_log)."""
    roll_cols = []

    for col in cols:
        col_lower = col.lower()
        is_log = '_log_' in col_lower

        for agg in ['mean', 'std', 'min', 'max', 'sum', 'median']:
            pattern = rf'roll[_]?(\d+)[_]?{agg}'
            match = re.search(pattern, col_lower)
            if match:
                window = int(match.group(1))
                roll_cols.append((col, window, agg, is_log))
                break

        if 'ewm' in col_lower and col not in [c[0] for c in roll_cols]:
            match_std = re.search(r'ewm[_]?std[_]?(\d+)', col_lower)
            if match_std:
                window = int(match_std.group(1))
                roll_cols.append((col, window, 'ewm_std', is_log))
            else:
                match_ewm = re.search(r'ewm[_]?(\d+)', col_lower)
                if match_ewm:
                    window = int(match_ewm.group(1))
                    roll_cols.append((col, window, 'ewm_mean', is_log))

    return roll_cols


def _identify_derived_cols(cols: List[str], time_format: str = 'year_week') -> List[Tuple[str, str, Any, bool]]:
    """Identify derived columns: (col_name, feat_type, param, is_log)."""
    derived_cols = []
    _yoy_period = 12 if time_format == 'year_month' else 52
    _qoq_period = 3 if time_format == 'year_month' else 13

    for col in cols:
        col_lower = col.lower()
        is_log = '_log_' in col_lower

        if re.search(r'lag_diff[_]?(\d+)$', col_lower):
            match = re.search(r'lag_diff[_]?(\d+)$', col_lower)
            derived_cols.append((col, 'lag_diff', int(match.group(1)), is_log))
        elif re.search(r'lag_pct_change[_]?(\d+)$', col_lower):
            match = re.search(r'lag_pct_change[_]?(\d+)$', col_lower)
            derived_cols.append((col, 'lag_pct_change', int(match.group(1)), is_log))
        elif re.search(r'lag_direction[_]?(\d+)$', col_lower):
            match = re.search(r'lag_direction[_]?(\d+)$', col_lower)
            derived_cols.append((col, 'lag_direction', int(match.group(1)), is_log))
        elif 'yoy_lag_diff' in col_lower:
            derived_cols.append((col, 'yoy_lag_diff', _yoy_period, is_log))
        elif 'yoy_lag_ratio' in col_lower:
            derived_cols.append((col, 'yoy_lag_ratio', _yoy_period, is_log))
        elif 'qoq_lag_diff' in col_lower:
            derived_cols.append((col, 'qoq_lag_diff', _qoq_period, is_log))

    return derived_cols


def _compute_lag_value(vals: List[float], lag_num: int, is_log: bool = False, is_binary: bool = False) -> float:
    """Compute lag value."""
    if len(vals) >= lag_num:
        raw_val = vals[-lag_num]
        if is_binary:
            return 1.0 if raw_val > 0 else 0.0
        if is_log:
            return np.log1p(max(0, raw_val))
        return raw_val
    return 0.0


def _compute_rolling_agg(vals: List[float], window: int, agg: str, is_log: bool = False) -> float:
    """Compute rolling aggregate."""
    if len(vals) < 1:
        return 0.0

    window_vals = vals[-min(window, len(vals)):]

    if is_log:
        window_vals = [np.log1p(max(0, v)) for v in window_vals]

    if agg == 'mean':
        return np.mean(window_vals) if window_vals else 0.0
    elif agg == 'ewm_mean':
        if not window_vals:
            return 0.0
        return pd.Series(window_vals).ewm(span=window, min_periods=1).mean().iloc[-1]
    elif agg == 'std':
        return np.std(window_vals) if len(window_vals) > 1 else 0.0
    elif agg == 'ewm_std':
        if len(window_vals) <= 1:
            return 0.0
        return pd.Series(window_vals).ewm(span=window, min_periods=1).std().iloc[-1]
    elif agg == 'sum':
        return np.sum(window_vals)
    elif agg == 'min':
        return np.min(window_vals) if window_vals else 0.0
    elif agg == 'max':
        return np.max(window_vals) if window_vals else 0.0
    elif agg == 'median':
        return np.median(window_vals) if window_vals else 0.0
    return np.mean(window_vals) if window_vals else 0.0


def _compute_derived_feature(vals: List[float], feat_type: str, param: Any, is_log: bool = False) -> float:
    """Compute derived feature."""
    if feat_type in ('lag_diff', 'lag_pct_change', 'lag_direction'):
        lag_num = param
        needed = lag_num + 1
        if len(vals) < needed:
            return 0.0

        lag_current = vals[-lag_num]
        lag_prev = vals[-(lag_num + 1)]

        if is_log:
            lag_current = np.log1p(max(0, lag_current))
            lag_prev = np.log1p(max(0, lag_prev))

        if feat_type == 'lag_diff':
            return lag_current - lag_prev
        elif feat_type == 'lag_pct_change':
            return (lag_current - lag_prev) / (abs(lag_prev) + 1e-10)
        elif feat_type == 'lag_direction':
            diff = lag_current - lag_prev
            return 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)

    elif feat_type == 'yoy_lag_diff':
        # param = yoy_period (12 for monthly, 52 for weekly)
        yoy_lag = 1 + param  # lag_1 vs lag_(1+period)
        if len(vals) < yoy_lag:
            return 0.0
        val_short = _compute_lag_value(vals, 1, is_log, False)
        val_long = _compute_lag_value(vals, yoy_lag, is_log, False)
        return val_short - val_long

    elif feat_type == 'yoy_lag_ratio':
        yoy_lag = 1 + param
        if len(vals) < yoy_lag:
            return 0.0
        val_short = _compute_lag_value(vals, 1, is_log, False)
        val_long = _compute_lag_value(vals, yoy_lag, is_log, False)
        return val_short / (abs(val_long) + 1e-10)

    elif feat_type == 'qoq_lag_diff':
        # param = qoq_period (3 for monthly, 13 for weekly)
        qoq_lag = 1 + param
        if len(vals) < qoq_lag:
            return 0.0
        val_short = _compute_lag_value(vals, 1, is_log, False)
        val_long = _compute_lag_value(vals, qoq_lag, is_log, False)
        return val_short - val_long

    return 0.0


def _identify_sparsity_cols(cols: List[str], time_format: str = 'year_week') -> List[Tuple[str, str, int]]:
    """
    Identify sparsity/intermittency columns for recursive update.

    Returns: List of (col_name, feat_type, window_or_param)

    Feature types:
    - 'weeks_since_demand': periods since last non-zero
    - 'demand_prob': rolling demand probability
    - 'typical_order': typical order size metrics
    - 'demand_streak': consecutive demand/no-demand periods
    - 'demand_freq': demand frequency
    - 'periods_since': periods since non-zero
    - 'avg_nonzero': average of non-zero values
    - 'last_nonzero': last non-zero value
    """
    sparsity_cols = []

    for col in cols:
        col_lower = col.lower()

        # Weeks since demand (windowed)
        match = re.search(r'weeks_since_demand[_]?(\d+)w', col_lower)
        if match:
            window = int(match.group(1))
            sparsity_cols.append((col, 'weeks_since_demand', window))
            continue

        # Demand probability (windowed)
        match = re.search(r'demand_prob[_]?(\d+)w', col_lower)
        if match:
            window = int(match.group(1))
            sparsity_cols.append((col, 'demand_prob', window))
            continue

        # Demand frequency (windowed)
        match = re.search(r'demand_freq[_]?(\d+)w', col_lower)
        if match:
            window = int(match.group(1))
            sparsity_cols.append((col, 'demand_freq', window))
            continue

        # Typical order (windowed or expanding)
        if 'typical_order' in col_lower:
            match = re.search(r'typical_order[_]?(\d+)w', col_lower)
            if match:
                window = int(match.group(1))
                sparsity_cols.append((col, 'typical_order_rolling', window))
            elif 'median' in col_lower:
                sparsity_cols.append((col, 'typical_order_median', 0))
            elif 'mean' in col_lower:
                sparsity_cols.append((col, 'typical_order_mean', 0))
            continue

        # Order size CV
        if 'order_size_cv' in col_lower:
            sparsity_cols.append((col, 'order_size_cv', 0))
            continue

        # Demand streak features
        if 'demand_streak' in col_lower:
            sparsity_cols.append((col, 'demand_streak', 0))
            continue
        if 'demand_run_length' in col_lower:
            sparsity_cols.append((col, 'demand_run', 0))
            continue
        if 'no_demand_run_length' in col_lower:
            sparsity_cols.append((col, 'no_demand_run', 0))
            continue

        # Demand consistency (half-year window: 6 for monthly, 26 for weekly)
        if 'demand_consistency' in col_lower:
            _consistency_window = 6 if time_format == 'year_month' else 26
            sparsity_cols.append((col, 'demand_consistency', _consistency_window))
            continue

        # Periods since nonzero (from intermittency features)
        if 'periods_since_nonzero' in col_lower:
            sparsity_cols.append((col, 'periods_since', 0))
            continue

        # Average nonzero
        if 'avg_nonzero' in col_lower:
            sparsity_cols.append((col, 'avg_nonzero', 0))
            continue

        # Last nonzero
        if 'last_nonzero' in col_lower:
            sparsity_cols.append((col, 'last_nonzero', 0))
            continue

        # ADI expanding
        if 'adi_expanding' in col_lower:
            sparsity_cols.append((col, 'adi_expanding', 0))
            continue

        # EWM demand
        match = re.search(r'demand_ewm[_]?(\d+)', col_lower)
        if match:
            span = int(match.group(1))
            sparsity_cols.append((col, 'demand_ewm', span))
            continue

    return sparsity_cols


def _compute_sparsity_feature(vals: List[float], feat_type: str, param: int) -> float:
    """
    Compute sparsity/intermittency feature value from historical actuals.

    Parameters
    ----------
    vals : List[float]
        Historical values (including predictions during recursive forecast)
    feat_type : str
        Type of feature to compute
    param : int
        Window size or other parameter

    Returns
    -------
    float
        Computed feature value
    """
    if len(vals) == 0:
        return 0.0

    # Shift by 1 to avoid leakage (we use historical values, not current)
    if len(vals) < 2:
        shifted_vals = []
    else:
        shifted_vals = vals[:-1]  # All except the last (most recent)

    if feat_type == 'weeks_since_demand':
        # Count periods since last non-zero, capped at window
        count = 0
        for v in reversed(shifted_vals):
            if v > 0:
                break
            count += 1
        return min(count, param) if param > 0 else count

    elif feat_type == 'periods_since':
        # Same as weeks_since but uncapped
        count = 0
        for v in reversed(shifted_vals):
            if v > 0:
                break
            count += 1
        return count

    elif feat_type in ('demand_prob', 'demand_freq'):
        # Rolling mean of demand occurrence
        window = param if param > 0 else len(shifted_vals)
        if len(shifted_vals) == 0:
            return 0.5
        window_vals = shifted_vals[-window:]
        demand_occurred = [1.0 if v > 0 else 0.0 for v in window_vals]
        return np.mean(demand_occurred) if demand_occurred else 0.5

    elif feat_type == 'typical_order_median':
        nonzero = [v for v in shifted_vals if v > 0]
        return np.median(nonzero) if nonzero else 0.0

    elif feat_type == 'typical_order_mean':
        nonzero = [v for v in shifted_vals if v > 0]
        return np.mean(nonzero) if nonzero else 0.0

    elif feat_type == 'typical_order_rolling':
        window = param if param > 0 else len(shifted_vals)
        window_vals = shifted_vals[-window:]
        nonzero = [v for v in window_vals if v > 0]
        return np.median(nonzero) if nonzero else 0.0

    elif feat_type == 'order_size_cv':
        nonzero = [v for v in shifted_vals if v > 0]
        if len(nonzero) < 2:
            return 1.0  # Default CV
        std = np.std(nonzero)
        mean = np.mean(nonzero)
        return min(std / (mean + 1e-10), 10.0)

    elif feat_type == 'demand_streak':
        if len(shifted_vals) == 0:
            return 0
        streak = 0
        last_state = None
        for v in shifted_vals:
            current_state = 'demand' if v > 0 else 'no_demand'
            if last_state is None:
                streak = 1
            elif current_state == last_state:
                streak += 1
            else:
                streak = 1
            last_state = current_state
        return streak if last_state == 'demand' else -streak

    elif feat_type == 'demand_run':
        if len(shifted_vals) == 0:
            return 0
        run = 0
        for v in reversed(shifted_vals):
            if v > 0:
                run += 1
            else:
                break
        return run

    elif feat_type == 'no_demand_run':
        if len(shifted_vals) == 0:
            return 0
        run = 0
        for v in reversed(shifted_vals):
            if v <= 0:
                run += 1
            else:
                break
        return run

    elif feat_type == 'demand_consistency':
        window = param if param > 0 else 26
        window_vals = shifted_vals[-window:]
        nonzero_idx = [i for i, v in enumerate(window_vals) if v > 0]
        if len(nonzero_idx) < 2:
            return 0.5 if nonzero_idx else 0.0
        intervals = np.diff(nonzero_idx)
        if len(intervals) == 0:
            return 0.5
        interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        return 1.0 / (1.0 + interval_cv)

    elif feat_type == 'avg_nonzero':
        nonzero = [v for v in shifted_vals if v > 0]
        return np.mean(nonzero) if nonzero else 0.0

    elif feat_type == 'last_nonzero':
        for v in reversed(shifted_vals):
            if v > 0:
                return v
        return 0.0

    elif feat_type == 'adi_expanding':
        nonzero_idx = [i for i, v in enumerate(shifted_vals) if v > 0]
        if len(nonzero_idx) <= 1:
            return len(shifted_vals) if len(nonzero_idx) == 0 else 1.0
        return np.mean(np.diff(nonzero_idx))

    elif feat_type == 'demand_ewm':
        span = param if param > 0 else 4
        if len(shifted_vals) == 0:
            return 0.0
        # Simple EWM calculation
        alpha = 2.0 / (span + 1)
        ewm = shifted_vals[0]
        for v in shifted_vals[1:]:
            ewm = alpha * v + (1 - alpha) * ewm
        return ewm

    return 0.0


# =============================================================================
# INTELLIGENT FEATURE IMPUTATION ENGINE
# =============================================================================
# Detects systematically missing/zero features in forecast periods and imputes
# them using a tiered strategy. Handles both "passthrough" features (external
# numerics, categoricals) and recursive feature fallbacks (long lags, YoY).
# =============================================================================

def _sply_period(period, time_format: str):
    """
    Compute Same-Period-Last-Year for a given period.

    Parameters
    ----------
    period : int or str
        Period value (YYYYWW, YYYYMM, or 'YYYY-WW'/'YYYY-MM' dash format)
    time_format : str
        'year_week' or 'year_month'

    Returns
    -------
    int or str
        The corresponding period from one year ago (same format as input)
    """
    s = str(period)
    if '-' in s:
        parts = s.split('-')
        year, sub = int(parts[0]), int(parts[1])
        prev_year = year - 1
        if time_format == 'year_week' and sub > 52:
            sub = 52
        return f"{prev_year}-{sub:02d}"
    else:
        p = int(float(s))
        year = p // 100
        sub = p % 100
        prev_year = year - 1
        if time_format == 'year_week' and sub > 52:
            sub = 52
        return prev_year * 100 + sub


def _period_within_year(period) -> int:
    """Extract the sub-period (week or month number) from YYYYWW/YYYYMM or YYYY-WW."""
    s = str(period)
    if '-' in s:
        return int(s.split('-')[1])
    return int(float(s)) % 100


def _classify_feature_columns(
    feature_cols: List[str],
    config: DemandForecastConfig,
    lag_col_names: set,
    roll_col_names: set,
    derived_col_names: set,
    sparsity_col_names: set,
) -> Dict[str, List[str]]:
    """
    Classify feature columns into categories for targeted imputation.

    Returns dict with keys:
    - 'external_numeric': External numeric features (price, promo, weather raw + lags)
    - 'categorical': Categorical features (encoded)
    - 'key_static': Key-level static features (key_mean, key_cv, etc.)
    - 'calendar': Calendar/Fourier/cyclical (never imputed)
    - 'recursive': Features that the recursive engine updates (lags, rolling, etc.)
    """
    recursive_cols = lag_col_names | roll_col_names | derived_col_names | sparsity_col_names

    # Known external numeric columns from config
    config_numeric = set(config.all_numeric_features())
    config_categorical = set(config.all_categorical_features())

    # Calendar/Fourier/cyclical patterns — never need imputation
    calendar_patterns = [
        'year', 'month', 'week_of_year', 'day_of_week', 'quarter',
        'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
        'fourier_sin', 'fourier_cos', 'cyclical_sin', 'cyclical_cos',
        'week_sin', 'week_cos', 'month_sin', 'month_cos',
    ]
    key_static_patterns = [
        'key_mean', 'key_std', 'key_min', 'key_max', 'key_cv', 'key_zero_fraction',
        'key_median', 'key_skew', 'key_kurtosis',
    ]

    classified = {
        'external_numeric': [],
        'categorical': [],
        'key_static': [],
        'calendar': [],
        'recursive': [],
    }

    for col in feature_cols:
        col_lower = col.lower()

        if col in recursive_cols:
            classified['recursive'].append(col)
            continue

        # Calendar/Fourier
        if any(p in col_lower for p in calendar_patterns):
            classified['calendar'].append(col)
            continue

        # Key-level static
        if any(p in col_lower for p in key_static_patterns):
            classified['key_static'].append(col)
            continue

        # Check if this is an encoded categorical or external lag of a categorical
        # Encoded categoricals often have suffixes like _encoded, _label, _ord
        is_categorical = False
        for cat_col in config_categorical:
            if col.startswith(cat_col) or col == cat_col:
                is_categorical = True
                break
        if is_categorical:
            classified['categorical'].append(col)
            continue

        # Check if this matches a config numeric or is a lag of one
        is_ext_numeric = False
        for num_col in config_numeric:
            if col == num_col or col.startswith(num_col + '_lag_') or col.startswith(num_col + '_lag'):
                is_ext_numeric = True
                break
        if is_ext_numeric:
            classified['external_numeric'].append(col)
            continue

        # Anything else that isn't recursive or calendar — treat as external numeric
        # (conservative: prefer imputing over leaving as zero)
        classified['external_numeric'].append(col)

    return classified


def detect_features_needing_imputation(
    forecast_df: pd.DataFrame,
    key_col: str,
    date_col: str,
    feature_cols: List[str],
    forecast_periods: List,
    threshold: float = 0.90,
) -> Dict[Any, List[str]]:
    """
    For each forecast period, identify features where >= threshold fraction
    of keys have value == 0 or NaN.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing only forecast period rows (test_features)
    key_col : str
        Key column name
    date_col : str
        Date column name
    feature_cols : List[str]
        Feature columns to check (should exclude calendar/recursive)
    forecast_periods : List
        List of forecast period values
    threshold : float
        Fraction threshold to trigger imputation (default 0.90)

    Returns
    -------
    Dict[period, List[feature_name]]
        Features flagged for imputation per period
    """
    if not feature_cols:
        return {}

    result = {}
    for period in forecast_periods:
        period_df = forecast_df[forecast_df[date_col] == period]
        n_keys = len(period_df)
        if n_keys == 0:
            continue

        flagged = []
        for col in feature_cols:
            if col not in period_df.columns:
                flagged.append(col)
                continue
            vals = period_df[col]
            n_zero_or_missing = ((vals == 0) | vals.isna()).sum()
            if n_zero_or_missing / n_keys >= threshold:
                flagged.append(col)

        if flagged:
            result[period] = flagged

    return result


def _detect_key_invariant_categoricals(
    history_df: pd.DataFrame,
    key_col: str,
    categorical_cols: List[str],
    min_history: int = 13,
) -> Dict[str, Dict[str, Any]]:
    """
    Detect categorical features that are constant (key-invariant) per key.

    A categorical is key-invariant for a key if across all of the key's
    training periods, it has only one unique non-null value, and the key
    has at least `min_history` periods.

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical data (train periods)
    key_col : str
        Key column name
    categorical_cols : List[str]
        Categorical feature columns to check
    min_history : int
        Minimum periods of history to declare invariance

    Returns
    -------
    Dict[key, Dict[feature, constant_value]]
        For each key, the categorical features that are invariant and their values
    """
    if not categorical_cols:
        return {}

    avail_cols = [c for c in categorical_cols if c in history_df.columns]
    if not avail_cols:
        return {}

    result = {}
    for key, grp in history_df.groupby(key_col):
        if len(grp) < min_history:
            continue
        key_invariants = {}
        for col in avail_cols:
            unique_vals = grp[col].dropna().unique()
            if len(unique_vals) == 1:
                key_invariants[col] = unique_vals[0]
        if key_invariants:
            result[key] = key_invariants

    return result


def impute_forecast_features(
    full_df: pd.DataFrame,
    forecast_periods: List,
    history_cutoff: Any,
    features_to_impute: Dict[Any, List[str]],
    key_col: str,
    date_col: str,
    target_col: str,
    time_format: str,
    manifest_df: pd.DataFrame,
    config: DemandForecastConfig,
    feature_classification: Dict[str, List[str]],
    key_invariant_cats: Dict[str, Dict[str, Any]],
    source_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply tiered imputation to forecast period features in full_df.

    Numeric features:
      Tier 1: SPLY per key (same key, same period last year)
      Tier 2: Category hierarchy group median at SPLY period
              (grouped by imputation_level columns from config)

    Categorical features:
      1. Key-invariant: forward-fill constant value
      2. SPLY per key (blank SPLY on full-history key → keep blank)
      3. Category hierarchy group mode at SPLY period

    Parameters
    ----------
    full_df : pd.DataFrame
        Combined train+val+test DataFrame (modified in place)
    forecast_periods : List
        Periods to impute
    history_cutoff : Any
        Last period of historical data
    features_to_impute : Dict[period, List[feature]]
        Output from detect_features_needing_imputation()
    key_col, date_col, target_col : str
        Column names
    time_format : str
        'year_week' or 'year_month'
    manifest_df : pd.DataFrame
        Training manifest with segment_id per key
    config : DemandForecastConfig
        Configuration object
    feature_classification : Dict[str, List[str]]
        Classified features from _classify_feature_columns()
    key_invariant_cats : Dict[key, Dict[feature, value]]
        Key-invariant categorical values
    source_df : pd.DataFrame, optional
        Original source data for building hierarchy group mapping

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Modified full_df and imputation report statistics
    """
    periods_per_year = config.periods_per_year
    categorical_cols_set = set(feature_classification.get('categorical', []))

    # Split historical data
    hist_df = full_df[full_df[date_col] <= history_cutoff].copy()

    # Build period→SPLY mapping
    sply_map = {p: _sply_period(p, time_format) for p in forecast_periods}
    sply_periods_needed = set(sply_map.values())

    # Pre-compute: per-key historical data for SPLY lookups
    hist_grouped = hist_df.groupby(key_col)

    # Collect all features that need imputation across all periods
    all_features_to_impute = set()
    for period, feats in features_to_impute.items():
        all_features_to_impute.update(feats)

    numeric_to_impute = [f for f in all_features_to_impute if f not in categorical_cols_set]
    categorical_to_impute = [f for f in all_features_to_impute if f in categorical_cols_set]

    # =========================================================================
    # TIER 2: Category hierarchy group medians/modes at SPLY
    # =========================================================================
    imputation_level = getattr(config.design, 'imputation_level', [])
    key_to_group = {}  # key → hierarchy group tuple
    hierarchy_sply_medians = {}  # {group_tuple: {sply_period: {feature: median}}}
    hierarchy_sply_modes = {}    # {group_tuple: {sply_period: {feature: mode}}}

    if imputation_level:
        # Build key→group mapping from source_df (has the raw hierarchy columns)
        _hierarchy_source = source_df if source_df is not None else full_df
        _avail_hier_cols = [c for c in imputation_level if c in _hierarchy_source.columns]

        if _avail_hier_cols:
            if len(_avail_hier_cols) < len(imputation_level):
                missing = set(imputation_level) - set(_avail_hier_cols)
                logger.warning(
                    f"[IMPUTATION] imputation_level columns not found in data: {missing}. "
                    f"Using available: {_avail_hier_cols}"
                )

            # Build key→group: take the latest non-null row per key
            _grp_df = (
                _hierarchy_source[[key_col] + _avail_hier_cols]
                .drop_duplicates(subset=[key_col], keep='last')
            )
            for _, row in _grp_df.iterrows():
                key_to_group[row[key_col]] = tuple(
                    str(row[c]) if pd.notna(row[c]) else '' for c in _avail_hier_cols
                )

            logger.info(
                f"[IMPUTATION] Hierarchy grouping on {_avail_hier_cols}: "
                f"{len(set(key_to_group.values()))} unique groups across "
                f"{len(key_to_group)} keys"
            )

            # Add hierarchy group column to historical data for SPLY lookups
            hist_df['_hier_group'] = hist_df[key_col].map(
                lambda k: key_to_group.get(k, ())
            )
            sply_hist = hist_df[hist_df[date_col].isin(sply_periods_needed)]

            if len(sply_hist) > 0:
                # Numeric medians per (group, sply_period)
                for (grp, period), sub in sply_hist.groupby(['_hier_group', date_col]):
                    if grp not in hierarchy_sply_medians:
                        hierarchy_sply_medians[grp] = {}
                    feat_medians = {}
                    for feat in numeric_to_impute:
                        if feat in sub.columns:
                            nonzero = sub[feat][sub[feat] != 0].dropna()
                            if len(nonzero) > 0:
                                feat_medians[feat] = float(nonzero.median())
                    hierarchy_sply_medians[grp][period] = feat_medians

                # Categorical modes per (group, sply_period)
                if categorical_to_impute:
                    for (grp, period), sub in sply_hist.groupby(['_hier_group', date_col]):
                        if grp not in hierarchy_sply_modes:
                            hierarchy_sply_modes[grp] = {}
                        feat_modes = {}
                        for feat in categorical_to_impute:
                            if feat in sub.columns:
                                vals = sub[feat].dropna()
                                vals = vals[vals != 0]
                                if len(vals) > 0:
                                    feat_modes[feat] = vals.mode().iloc[0]
                        hierarchy_sply_modes[grp][period] = feat_modes
        else:
            logger.warning(
                f"[IMPUTATION] None of the imputation_level columns {imputation_level} "
                f"found in data. Tier 2 hierarchy imputation disabled."
            )

    # --- Tracking for diagnostics ---
    report = {
        'periods_analyzed': len(forecast_periods),
        'periods_with_imputations': len(features_to_impute),
        'total_features_flagged': len(all_features_to_impute),
        'numeric_features_flagged': len(numeric_to_impute),
        'categorical_features_flagged': len(categorical_to_impute),
        'imputation_level': imputation_level,
        'per_period': {},
        'per_feature': {},
    }

    for feat in all_features_to_impute:
        report['per_feature'][feat] = {
            'tier1_sply': 0, 'tier2_hierarchy': 0,
            'invariant': 0, 'kept_blank': 0, 'unfilled': 0,
        }

    # =========================================================================
    # MAIN IMPUTATION LOOP — per forecast period
    # =========================================================================
    for period in forecast_periods:
        if period not in features_to_impute:
            continue

        feats_this_period = features_to_impute[period]
        sply_period = sply_map[period]

        period_stats = {
            'features_imputed': feats_this_period,
            'tier1': 0, 'tier2': 0,
            'invariant': 0, 'kept_blank': 0,
        }

        period_mask = full_df[date_col] == period
        period_indices = full_df.index[period_mask]

        for idx in period_indices:
            key = full_df.at[idx, key_col]
            hier_group = key_to_group.get(key, ())

            try:
                key_hist = hist_grouped.get_group(key)
            except KeyError:
                key_hist = pd.DataFrame()

            key_sply_row = key_hist[key_hist[date_col] == sply_period] if len(key_hist) > 0 else pd.DataFrame()
            key_has_full_year = len(key_hist) >= periods_per_year

            for feat in feats_this_period:
                if feat not in full_df.columns:
                    continue

                current_val = full_df.at[idx, feat]
                if not (current_val == 0 or pd.isna(current_val)):
                    continue

                is_categorical = feat in categorical_cols_set

                if is_categorical:
                    imputed = _impute_categorical_feature(
                        key=key, feat=feat,
                        key_hist=key_hist, key_sply_row=key_sply_row,
                        key_has_full_year=key_has_full_year,
                        key_invariant_cats=key_invariant_cats,
                        hier_group=hier_group, sply_period=sply_period,
                        hierarchy_sply_modes=hierarchy_sply_modes,
                        period_stats=period_stats, report=report,
                    )
                else:
                    imputed = _impute_numeric_feature(
                        key=key, feat=feat,
                        key_sply_row=key_sply_row,
                        hier_group=hier_group, sply_period=sply_period,
                        hierarchy_sply_medians=hierarchy_sply_medians,
                        config=config, key_hist=key_hist,
                        period_stats=period_stats, report=report,
                    )

                if imputed is not None:
                    full_df.at[idx, feat] = imputed

        report['per_period'][str(period)] = period_stats

    return full_df, report


def _impute_numeric_feature(
    key: str,
    feat: str,
    key_sply_row: pd.DataFrame,
    hier_group: tuple,
    sply_period: int,
    hierarchy_sply_medians: Dict,
    config: DemandForecastConfig,
    key_hist: pd.DataFrame,
    period_stats: Dict,
    report: Dict,
) -> Optional[float]:
    """
    Impute a single numeric feature value.

    Tier 1: SPLY per key (same key, same period last year)
    Tier 2: Category hierarchy group median at SPLY period
    """
    # --- Tier 1: SPLY per key ---
    if len(key_sply_row) > 0 and feat in key_sply_row.columns:
        sply_val = key_sply_row.iloc[0].get(feat)
        if sply_val is not None and not pd.isna(sply_val) and sply_val != 0:
            if config.design.imputation_enable_trend_adjustment and feat in key_hist.columns:
                sply_val = _apply_trend_adjustment(key_hist, feat, sply_val, config)
            period_stats['tier1'] += 1
            report['per_feature'][feat]['tier1_sply'] += 1
            return float(sply_val)

    # --- Tier 2: Category hierarchy group median at SPLY ---
    if hier_group:
        grp_medians = hierarchy_sply_medians.get(hier_group, {}).get(sply_period, {})
        if feat in grp_medians:
            period_stats['tier2'] += 1
            report['per_feature'][feat]['tier2_hierarchy'] += 1
            return float(grp_medians[feat])

    report['per_feature'][feat]['unfilled'] += 1
    return None


def _impute_categorical_feature(
    key: str,
    feat: str,
    key_hist: pd.DataFrame,
    key_sply_row: pd.DataFrame,
    key_has_full_year: bool,
    key_invariant_cats: Dict[str, Dict[str, Any]],
    hier_group: tuple,
    sply_period: int,
    hierarchy_sply_modes: Dict,
    period_stats: Dict,
    report: Dict,
) -> Optional[Any]:
    """
    Impute a single categorical feature value.

    Priority:
    1. Key-invariant (constant across all training periods) → forward-fill
    2. SPLY per key (blank SPLY on full-history key → keep blank)
    3. Category hierarchy group mode at SPLY period
    """
    # --- Priority 1: Key-invariant check ---
    key_invs = key_invariant_cats.get(key, {})
    if feat in key_invs:
        period_stats['invariant'] += 1
        report['per_feature'][feat]['invariant'] += 1
        return key_invs[feat]

    # --- Priority 2: SPLY per key ---
    if key_has_full_year:
        if len(key_sply_row) > 0 and feat in key_sply_row.columns:
            sply_val = key_sply_row.iloc[0].get(feat)
            if sply_val is not None and not pd.isna(sply_val) and sply_val != 0:
                period_stats['tier1'] += 1
                report['per_feature'][feat]['tier1_sply'] += 1
                return sply_val
            else:
                period_stats['kept_blank'] += 1
                report['per_feature'][feat]['kept_blank'] += 1
                return None
        else:
            period_stats['kept_blank'] += 1
            report['per_feature'][feat]['kept_blank'] += 1
            return None

    # --- Priority 3: Category hierarchy group mode at SPLY ---
    if hier_group:
        grp_modes = hierarchy_sply_modes.get(hier_group, {}).get(sply_period, {})
        if feat in grp_modes:
            period_stats['tier2'] += 1
            report['per_feature'][feat]['tier2_hierarchy'] += 1
            return grp_modes[feat]

    report['per_feature'][feat]['unfilled'] += 1
    return None


def _apply_trend_adjustment(
    key_hist: pd.DataFrame,
    feat: str,
    sply_val: float,
    config: DemandForecastConfig,
) -> float:
    """
    Adjust SPLY value for YoY trend when enabled.

    Computes growth rate from the key's recent vs older history of the feature
    and applies: imputed = sply_value × (1 + yoy_growth_rate), clamped to [0.5, 2.0].
    """
    if feat not in key_hist.columns or len(key_hist) < 4:
        return sply_val

    periods_per_year = config.periods_per_year
    vals = key_hist[feat].values
    n = len(vals)

    if n >= periods_per_year:
        recent = np.mean(vals[-periods_per_year // 2:])
        older = np.mean(vals[-periods_per_year: -periods_per_year // 2])
    else:
        half = max(n // 2, 1)
        recent = np.mean(vals[-half:])
        older = np.mean(vals[:-half]) if n > half else recent

    if older != 0 and not np.isnan(older) and not np.isnan(recent):
        growth_rate = (recent - older) / (abs(older) + 1e-10)
        # Clamp growth rate to prevent extreme adjustments
        growth_rate = np.clip(growth_rate, -0.5, 1.0)
        return sply_val * (1.0 + growth_rate)

    return sply_val


def build_recursive_feature_fallbacks(
    full_df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    history_cutoff: Any,
    time_format: str,
    manifest_df: pd.DataFrame,
    lag_cols: List[Tuple],
    derived_cols: List[Tuple],
    config: DemandForecastConfig,
    forecast_periods: List,
) -> Dict[str, Dict[str, float]]:
    """
    Pre-compute fallback values for recursive features that return 0.0
    when key history is too short (e.g., lag_52 on a 30-week key).

    Uses tiered strategy:
    1. SPLY data from this key (partial — e.g., if lag_52 fails but lag_26 works)
    2. Segment peer median of the feature at the same seasonal position
    3. Global population median

    Parameters
    ----------
    full_df : pd.DataFrame
        Combined historical DataFrame
    key_col, date_col, target_col : str
        Column names
    history_cutoff : Any
        Last historical period
    time_format : str
        'year_week' or 'year_month'
    manifest_df : pd.DataFrame
        Training manifest with segment info
    lag_cols : List[Tuple]
        Identified lag columns: (col_name, lag_num, is_log, is_binary)
    derived_cols : List[Tuple]
        Identified derived columns: (col_name, feat_type, param, is_log)
    config : DemandForecastConfig
        Config object
    forecast_periods : List
        Forecast period values

    Returns
    -------
    Dict[key, Dict[feature_col, fallback_value]]
        Fallback values per key per feature
    """
    periods_per_year = config.periods_per_year

    # Only need fallbacks for features requiring long history
    long_lag_cols = [(col, lag, is_log, is_binary)
                     for col, lag, is_log, is_binary in lag_cols
                     if lag >= periods_per_year // 2]  # lags >= half-year
    long_derived = [(col, ft, param, is_log)
                    for col, ft, param, is_log in derived_cols
                    if ft in ('yoy_lag_diff', 'yoy_lag_ratio', 'qoq_lag_diff')]

    if not long_lag_cols and not long_derived:
        return {}

    hist_df = full_df[full_df[date_col] <= history_cutoff]

    # Determine grouping column: segment_id if available, else model_level
    _seg_col = 'segment_id' if 'segment_id' in manifest_df.columns else 'model_level'

    # Build key→segment mapping
    key_to_segment = dict(zip(manifest_df['key'], manifest_df[_seg_col].astype(str)))

    # Pre-compute: segment-level medians for each long feature
    # For each segment, compute the median of each long-lag feature across all keys
    # that have sufficient history
    _seg_lookup = manifest_df[['key', _seg_col]].copy()
    _seg_lookup.columns = ['key', 'segment_id']
    if key_col != 'key':
        _seg_lookup = _seg_lookup.rename(columns={'key': key_col})
    _seg_lookup = _seg_lookup.drop_duplicates()

    # Drop existing segment_id from hist_df to avoid suffix conflicts on merge
    _hist_for_seg_merge2 = hist_df.drop(columns=['segment_id'], errors='ignore')
    hist_with_seg = _hist_for_seg_merge2.merge(_seg_lookup, on=key_col, how='left')
    hist_with_seg['segment_id'] = hist_with_seg['segment_id'].fillna('unknown').astype(str)

    feature_cols_needed = [col for col, _, _, _ in long_lag_cols] + [col for col, _, _, _ in long_derived]
    feature_cols_needed = [c for c in feature_cols_needed if c in hist_df.columns]

    # Segment medians from keys with full history
    segment_feature_medians = {}  # {segment: {feature: median}}
    if feature_cols_needed:
        # Only use keys with sufficient history (>= periods_per_year)
        key_lengths = hist_df.groupby(key_col).size()
        full_hist_keys = set(key_lengths[key_lengths >= periods_per_year].index)

        if full_hist_keys:
            full_hist_seg = hist_with_seg[hist_with_seg[key_col].isin(full_hist_keys)]
            for seg, grp in full_hist_seg.groupby('segment_id'):
                seg_meds = {}
                for feat in feature_cols_needed:
                    if feat in grp.columns:
                        nonzero = grp[feat][grp[feat] != 0].dropna()
                        if len(nonzero) > 0:
                            seg_meds[feat] = float(nonzero.median())
                segment_feature_medians[seg] = seg_meds

    # Global medians (Tier 3 fallback)
    global_feature_medians = {}
    for feat in feature_cols_needed:
        nonzero = hist_df[feat][hist_df[feat] != 0].dropna()
        if len(nonzero) > 0:
            global_feature_medians[feat] = float(nonzero.median())

    # Now build per-key fallback maps
    key_fallbacks = {}  # {key: {feature_col: fallback_value}}
    key_lengths = hist_df.groupby(key_col).size()
    short_history_keys = key_lengths[key_lengths < periods_per_year].index

    for key in short_history_keys:
        segment = key_to_segment.get(key, 'unknown')
        fallbacks = {}

        for col, lag, is_log, is_binary in long_lag_cols:
            hist_len = key_lengths.get(key, 0)
            if hist_len >= lag:
                continue  # This key has enough history, no fallback needed

            # Tier 1: Segment median
            seg_meds = segment_feature_medians.get(segment, {})
            if col in seg_meds:
                fallbacks[col] = seg_meds[col]
                continue

            # Tier 2: Global median
            if col in global_feature_medians:
                fallbacks[col] = global_feature_medians[col]
                continue

        for col, feat_type, param, is_log in long_derived:
            hist_len = key_lengths.get(key, 0)
            needed = 1 + param  # e.g., yoy needs 1 + 52 = 53
            if hist_len >= needed:
                continue

            seg_meds = segment_feature_medians.get(segment, {})
            if col in seg_meds:
                fallbacks[col] = seg_meds[col]
                continue
            if col in global_feature_medians:
                fallbacks[col] = global_feature_medians[col]
                continue

        if fallbacks:
            key_fallbacks[key] = fallbacks

    return key_fallbacks


def save_imputation_report(
    report: Dict[str, Any],
    output_dir: str,
    filename: str = "imputation_report.json",
) -> str:
    """Save imputation diagnostics report as JSON."""
    report_path = os.path.join(output_dir, filename)
    # Convert any numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_report = json.loads(json.dumps(report, default=_convert))
    with open(report_path, 'w') as f:
        json.dump(clean_report, f, indent=2)
    return report_path


def _predict_with_model(model: Any, X: np.ndarray, model_type: str = None, horizon: int = None) -> float:
    """
    Make prediction with various model types.

    Handles:
    - Standard sklearn-style models (.predict method)
    - XGBoost Boosters
    - Weighted ensembles
    - Discrete demand models (ordinal_regression, discrete_classifier, hybrid_discrete)
    - Intermittent demand models (croston, sba, tsb, imapa)
    - Multi-horizon models (multi_horizon_lightgbm, multi_horizon_xgboost, multi_horizon_ensemble)

    Parameters
    ----------
    model : Any
        Trained model object or model container dict
    X : np.ndarray
        Feature matrix for prediction (single sample or batch)
    model_type : str, optional
        Model type hint (e.g., 'ordinal_regression', 'discrete_classifier', 'multi_horizon_lightgbm')
        Helps with specialized model handling
    horizon : int, optional
        Forecast horizon for multi-horizon models. If None, uses model's target_horizon.

    Returns
    -------
    float
        Predicted value (single prediction)
    """
    try:
        # Handle multi-horizon models (MultiHorizonModel wrapper)
        # Check by model_type hint or by presence of multi-horizon attributes
        if model_type and model_type.startswith('multi_horizon'):
            # Multi-horizon model - use horizon parameter or default to target_horizon
            if hasattr(model, 'predict') and hasattr(model, 'target_horizon'):
                pred = model.predict(X, horizon=horizon)
                if isinstance(pred, np.ndarray):
                    return float(pred[0]) if len(pred) == 1 else float(pred.mean())
                return float(pred)
            # Could also be the inner model (MultiHorizonModel wrapper)
            elif hasattr(model, 'models') and hasattr(model, 'strategy'):
                pred = model.predict(X, horizon=horizon)
                if isinstance(pred, np.ndarray):
                    return float(pred[0]) if len(pred) == 1 else float(pred.mean())
                return float(pred)

        # Check for MultiHorizonModel by duck-typing (if model_type not provided)
        if hasattr(model, 'target_horizon') and hasattr(model, 'max_horizon') and hasattr(model, 'strategy'):
            pred = model.predict(X, horizon=horizon)
            if isinstance(pred, np.ndarray):
                return float(pred[0]) if len(pred) == 1 else float(pred.mean())
            return float(pred)

        # Handle discrete demand model containers
        # These are dict structures with 'classifier' or 'regressor' keys
        if isinstance(model, dict):
            # Check for discrete model types
            if 'classifier' in model and 'unique_values' in model:
                # Ordinal regression or discrete classifier
                from utils.model_selection_intelligence import snap_to_discrete_values

                classifier = model['classifier']
                unique_values = model['unique_values']

                if 'class_to_value' in model:
                    # Discrete classifier - predict class and map to value
                    pred_class = classifier.predict(X)
                    class_to_value = model['class_to_value']
                    pred = class_to_value.get(int(pred_class[0]), unique_values[0])
                else:
                    # Ordinal regression - use expected value from probabilities
                    probs = classifier.predict_proba(X)
                    pred = np.sum(probs * unique_values, axis=1)
                    snap = model.get('snap_to_valid', True)
                    if snap:
                        pred = snap_to_discrete_values(pred, unique_values, method='nearest')
                    pred = pred[0] if isinstance(pred, np.ndarray) else pred

                return float(pred)

            elif 'regressor' in model and 'unique_values' in model:
                # Hybrid discrete - regression + snap to valid values
                from utils.model_selection_intelligence import snap_to_discrete_values

                regressor = model['regressor']
                unique_values = model['unique_values']

                pred_cont = regressor.predict(X)
                pred = snap_to_discrete_values(pred_cont, unique_values, method='nearest')
                pred = pred[0] if isinstance(pred, np.ndarray) else pred

                return float(pred)

            elif model.get('type') == 'weighted_ensemble':
                # Weighted ensemble
                component_models = model.get('component_models', [])
                weights = model.get('weights', [])
                if not component_models or not weights:
                    return 0.0

                weighted_pred = 0.0
                for (mt, comp_model), weight in zip(component_models, weights):
                    try:
                        if isinstance(comp_model, dict):
                            # Univariate model stored as dict — use stored forecast
                            comp_pred = float(comp_model.get('forecast', 0.0))
                        elif hasattr(comp_model, 'predict'):
                            comp_pred = comp_model.predict(X)
                            if isinstance(comp_pred, np.ndarray):
                                comp_pred = comp_pred[0] if len(comp_pred) == 1 else comp_pred.mean()
                        else:
                            continue
                        weighted_pred += weight * comp_pred
                    except:
                        pass
                return weighted_pred

            elif model.get('type') == 'constant':
                # Constant prediction (fallback for single-class scenarios)
                return float(model.get('value', 0.0))

            elif model.get('type') in ('croston', 'sba', 'tsb', 'imapa'):
                # Intermittent demand models - return stored forecast
                return float(model.get('forecast', 0.0))

            else:
                # Unknown dict type, try to get a forecast/value
                if 'forecast' in model:
                    return float(model['forecast'])
                elif 'value' in model:
                    return float(model['value'])
                logger.debug(f"Unknown dict model type: {model.get('type', 'unknown')}")
                return 0.0

        elif hasattr(model, 'save_model') and not hasattr(model, 'fit'):
            # XGBoost Booster
            import xgboost as xgb
            dmat = xgb.DMatrix(X)
            return model.predict(dmat)[0]

        elif hasattr(model, 'predict'):
            pred = model.predict(X)
            if isinstance(pred, np.ndarray):
                return pred[0] if len(pred) == 1 else pred.mean()
            return pred

    except Exception as e:
        logger.debug(f"Prediction error: {e}")

    return 0.0


def _assign_zero_fraction_bucket(zero_fraction: float, bucket_edges: List[float]) -> int:
    """
    Assign a zero_fraction value to a bucket based on percentile edges.

    This matches the bucket assignment logic in utils/bias_calibration.py

    Parameters
    ----------
    zero_fraction : float
        Zero fraction value (0.0 to 1.0)
    bucket_edges : List[float]
        Bucket edges from calibration (e.g., [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    Returns
    -------
    int
        Bucket index (0 to n_buckets-1)
    """
    if not bucket_edges or len(bucket_edges) < 2:
        return 0

    n_buckets = len(bucket_edges) - 1
    for i in range(1, len(bucket_edges)):
        if zero_fraction <= bucket_edges[i]:
            return min(i - 1, n_buckets - 1)
    return n_buckets - 1


def _apply_bias_calibration(
    pred: float,
    key: str,
    segment_id: str,
    calibration_factors: Dict[str, Any],
    segment_calibrations: Optional[Dict[str, Any]] = None,
    zero_fraction: Optional[float] = None,
    lag: Optional[int] = None,
    lag_calibration_factors: Optional[Dict[str, float]] = None,
) -> float:
    """
    Apply bias calibration factor to prediction.

    CORRECTED: Uses the SAME calibration approach as training:
    1. Uses segment_id (not model_level) for calibration lookup
    2. Uses bucket_edges from bias_calibration.json for zero_fraction bucketing
    3. Looks up factor using "{segment}_zf_{bucket}" key format
    4. Falls back to segment fallback factor, then global factor
    5. Clips factors to [factor_min, factor_max] bounds

    Parameters
    ----------
    pred : float
        Raw prediction value
    key : str
        Key identifier
    segment_id : str
        Segment ID for this key (from manifest's segment_id column)
        IMPORTANT: This should be segment_id, NOT model_level
    calibration_factors : Dict
        Bias calibration factors (from bias_calibration.json) with structure:
        - factors: Dict[str, float] mapping "{segment}_zf_{bucket}" -> factor
        - fallback_factors: Dict[str, float] mapping segment -> factor
        - global_factor: float
        - bucket_edges: List[float] for zero_fraction bucketing
        - factor_min, factor_max: Clipping bounds
    segment_calibrations : Dict, optional
        Segment-aware calibration factors (from segment_calibrations.json)
        Used as secondary fallback if primary calibration doesn't have the segment
    zero_fraction : float, optional
        Zero fraction for this key (used for bucket lookup)

    Returns
    -------
    float
        Calibrated prediction
    """
    factor = 1.0

    # Get clipping bounds from calibration (or use defaults)
    factor_min = 0.2
    factor_max = 2.0
    if calibration_factors:
        factor_min = calibration_factors.get('factor_min', 0.2)
        factor_max = calibration_factors.get('factor_max', 2.0)

    # Priority 0: Lag-specific calibration (Phase 4 — more precise than segment×zf)
    # Lookup: "{segment}_lag_{lag}" → factor, fallback to "global_lag_{lag}"
    if lag is not None and lag_calibration_factors:
        lag_key = f"{segment_id}_lag_{lag}"
        if lag_key in lag_calibration_factors:
            factor = float(np.clip(lag_calibration_factors[lag_key], factor_min, factor_max))
            return max(pred * factor, 0)
        global_lag_key = f"global_lag_{lag}"
        if global_lag_key in lag_calibration_factors:
            factor = float(np.clip(lag_calibration_factors[global_lag_key], factor_min, factor_max))
            return max(pred * factor, 0)

    # Priority 1: Use bias_calibration.json with proper bucket assignment
    # This matches the exact approach used in training (utils/bias_calibration.py)
    if calibration_factors and calibration_factors.get('enabled', True):
        factors_dict = calibration_factors.get('factors', {})
        fallback_factors = calibration_factors.get('fallback_factors', {})
        global_factor = calibration_factors.get('global_factor', 1.0)
        bucket_edges = calibration_factors.get('bucket_edges', [])

        # Assign bucket based on zero_fraction
        if zero_fraction is not None and bucket_edges:
            zf_bucket = _assign_zero_fraction_bucket(zero_fraction, bucket_edges)
        else:
            # Default to middle bucket if no zero_fraction available
            n_buckets = calibration_factors.get('n_buckets', 5)
            zf_bucket = n_buckets // 2

        # Try exact match: "{segment}_zf_{bucket}"
        lookup_key = f"{segment_id}_zf_{zf_bucket}"
        if lookup_key in factors_dict:
            factor = factors_dict[lookup_key]
        # Fallback to segment factor
        elif segment_id in fallback_factors:
            factor = fallback_factors[segment_id]
        # Global fallback
        else:
            factor = global_factor

    # Priority 2: Fall back to segment_calibrations.json (if bias_calibration.json didn't have factors)
    elif segment_calibrations and segment_id in segment_calibrations:
        seg_cal = segment_calibrations[segment_id]

        # Try to find bucket-specific factor if zero_fraction is available
        if zero_fraction is not None and 'calibration_factors' in seg_cal:
            bucket_factors = seg_cal['calibration_factors']
            if isinstance(bucket_factors, dict):
                # Try to find matching bucket by zero_fraction range
                best_bucket = None
                for bucket_key, bucket_val in bucket_factors.items():
                    if isinstance(bucket_val, dict) and 'zero_fraction_range' in bucket_val:
                        zf_range = bucket_val['zero_fraction_range']
                        if zf_range[0] <= zero_fraction < zf_range[1]:
                            factor = bucket_val.get('factor', 1.0)
                            best_bucket = bucket_key
                            break

                if best_bucket is None:
                    factor = seg_cal.get('global_factor', 1.0)
            else:
                factor = seg_cal.get('global_factor', 1.0)
        else:
            factor = seg_cal.get('global_factor', 1.0)

    # Ensure factor is a valid number
    if not isinstance(factor, (int, float)) or np.isnan(factor):
        factor = 1.0

    # CRITICAL: Clip factor to bounds (same as training)
    factor = max(factor_min, min(factor_max, factor))

    return pred * factor


# =============================================================================
# DOCUMENTATION GENERATION
# =============================================================================

def generate_inference_documentation(
    inference_result: InferenceResult,
    forecasts_df: pd.DataFrame,
    config: DemandForecastConfig,
    output_dir: Optional[str] = None,
) -> str:
    """
    Generate comprehensive markdown documentation explaining inference insights.

    This function creates a detailed guide for data scientists that explains:
    - What the inference pipeline did
    - New key handling and segment assignments
    - Dead key detection and handling
    - Forecast accuracy expectations
    - Key learnings and recommendations

    Parameters
    ----------
    inference_result : InferenceResult
        Result from run_inference_pipeline()
    forecasts_df : pd.DataFrame
        Generated forecasts
    config : DemandForecastConfig
        Configuration object
    output_dir : str, optional
        Override output directory (default: model_artifacts/)

    Returns
    -------
    str
        Path to the generated documentation file
    """
    artifact_base = config.artifact_base_path
    model_dir = output_dir or os.path.join(artifact_base, "model_artifacts")
    os.makedirs(model_dir, exist_ok=True)

    doc_path = os.path.join(model_dir, "INFERENCE_INSIGHTS_GUIDE.md")

    # Compute metrics for documentation
    metrics_summary = {}
    horizon_metrics = {}
    model_metrics = {}
    new_key_metrics = {}

    if len(forecasts_df) > 0:
        actual = forecasts_df['actual'].values
        predicted = forecasts_df['predicted'].values
        actual_sum = actual.sum()

        # Overall metrics
        if actual_sum > 0:
            metrics_summary['overall_wape'] = float(np.abs(predicted - actual).sum() / actual_sum)
            metrics_summary['overall_bias_pct'] = float((predicted.sum() - actual_sum) / actual_sum * 100)
        else:
            metrics_summary['overall_wape'] = 0.0
            metrics_summary['overall_bias_pct'] = 0.0

        metrics_summary['overall_mae'] = float(np.abs(predicted - actual).mean())

        # By horizon (lag)
        if 'lag' in forecasts_df.columns:
            for lag in sorted(forecasts_df['lag'].unique()):
                lag_df = forecasts_df[forecasts_df['lag'] == lag]
                lag_actual = lag_df['actual'].values
                lag_pred = lag_df['predicted'].values
                lag_actual_sum = lag_actual.sum()

                if lag_actual_sum > 0:
                    wape = float(np.abs(lag_pred - lag_actual).sum() / lag_actual_sum)
                    bias = float((lag_pred.sum() - lag_actual_sum) / lag_actual_sum * 100)
                else:
                    wape = 0.0
                    bias = 0.0

                horizon_metrics[int(lag)] = {
                    'wape': wape,
                    'bias_pct': bias,
                    'mae': float(np.abs(lag_pred - lag_actual).mean()),
                    'n_forecasts': len(lag_df),
                }

        # By model level
        if 'model_level' in forecasts_df.columns:
            for ml in forecasts_df['model_level'].unique():
                ml_df = forecasts_df[forecasts_df['model_level'] == ml]
                ml_actual = ml_df['actual'].values
                ml_pred = ml_df['predicted'].values
                ml_actual_sum = ml_actual.sum()

                if ml_actual_sum > 0:
                    wape = float(np.abs(ml_pred - ml_actual).sum() / ml_actual_sum)
                    bias = float((ml_pred.sum() - ml_actual_sum) / ml_actual_sum * 100)
                else:
                    wape = 0.0
                    bias = 0.0

                model_metrics[str(ml)] = {
                    'wape': wape,
                    'bias_pct': bias,
                    'n_forecasts': len(ml_df),
                }

        # New vs existing key performance
        if 'is_new_key' in forecasts_df.columns:
            new_df = forecasts_df[forecasts_df['is_new_key'] == True]
            existing_df = forecasts_df[forecasts_df['is_new_key'] == False]

            for label, df in [('new_keys', new_df), ('existing_keys', existing_df)]:
                if len(df) > 0:
                    df_actual = df['actual'].values
                    df_pred = df['predicted'].values
                    df_actual_sum = df_actual.sum()

                    if df_actual_sum > 0:
                        wape = float(np.abs(df_pred - df_actual).sum() / df_actual_sum)
                        bias = float((df_pred.sum() - df_actual_sum) / df_actual_sum * 100)
                    else:
                        wape = 0.0
                        bias = 0.0

                    new_key_metrics[label] = {
                        'wape': wape,
                        'bias_pct': bias,
                        'n_forecasts': len(df),
                        'n_unique_keys': df['key'].nunique() if 'key' in df.columns else 0,
                    }

    # Generate documentation
    doc_content = f"""# Inference Pipeline Insights Guide

## Executive Summary

This document provides a comprehensive analysis of the production inference pipeline results, explaining the methodology, key findings, and actionable insights for demand forecasting.

| Metric | Value |
|--------|-------|
| **Total Keys** | {inference_result.total_keys:,} |
| **New Keys Detected** | {inference_result.new_keys_count:,} |
| **Existing Keys** | {inference_result.existing_keys_count:,} |
| **Dead Keys** | {inference_result.dead_keys_count:,} |
| **Models Retrained** | {inference_result.models_retrained} |
| **Forecast Horizon** | {inference_result.forecast_horizon} periods |
| **Total Forecasts** | {inference_result.total_forecasts:,} |

---

## 1. Pipeline Overview

The inference pipeline is the production forecasting system that:

1. **Detects New Keys**: Identifies products/locations in the inference period that weren't in the original training
2. **Handles Dead Keys**: Detects inactive keys (no recent demand) and assigns zero forecasts
3. **Segments New Keys**: Uses the trained clustering model to assign new keys to existing segments
4. **Retrains Models**: Updates all segment models with the latest data
5. **Generates Forecasts**: Produces recursive multi-step forecasts for all active keys
6. **Applies Calibration**: Uses segment-aware bias calibration to improve accuracy

### Pipeline Configuration

| Setting | Value |
|---------|-------|
| Test Period | {config.test_start} to {config.test_end} |
| Forecast Horizon | {config.forecast_horizon} periods |
| Feature Regeneration | {'Enabled' if config.design.regenerate_features_on_inference else 'Disabled'} |
| Bias Calibration | {'Enabled' if config.design.apply_bias_calibration else 'Disabled'} |

---

## 2. Key Detection Analysis

### Key Categories

| Category | Count | Description |
|----------|-------|-------------|
| **Existing Keys** | {inference_result.existing_keys_count:,} | Keys in both training manifest and inference period |
| **New Keys** | {inference_result.new_keys_count:,} | Keys in inference period but not in training |
| **Dead Keys** | {inference_result.dead_keys_count:,} | Keys with no recent demand (assigned zero forecast) |

"""

    if inference_result.new_keys_by_segment:
        doc_content += """### New Key Segment Assignments

New keys were assigned to existing segments based on their demand characteristics:

| Segment | New Keys Assigned |
|---------|-------------------|
"""
        for seg, count in sorted(inference_result.new_keys_by_segment.items(), key=lambda x: -x[1]):
            doc_content += f"| {seg} | {count:,} |\n"

        doc_content += """
**Why Segment Assignment Matters**: New keys inherit the model and hyperparameters of their assigned segment. Keys with similar demand patterns (intermittency, variability, seasonality) are grouped together for more accurate forecasting.
"""

    doc_content += f"""

### Dead Key Handling

Dead keys are identified using a lookback analysis:
- **Lookback Period**: {DEFAULT_DEAD_KEY_LOOKBACK} periods before inference start
- **Activity Threshold**: Total demand ≤ {DEFAULT_DEAD_KEY_THRESHOLD}
- **Result**: {inference_result.dead_keys_count:,} keys marked as dead

Dead keys receive a **zero forecast** for all periods, with actual values captured from source data for accuracy tracking.

---

## 3. Forecast Accuracy Analysis

### Overall Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **WAPE** | {metrics_summary.get('overall_wape', 0):.2%} | {'Excellent (<10%)' if metrics_summary.get('overall_wape', 1) < 0.10 else 'Good (10-20%)' if metrics_summary.get('overall_wape', 1) < 0.20 else 'Moderate (20-30%)' if metrics_summary.get('overall_wape', 1) < 0.30 else 'Needs Improvement (>30%)'} |
| **Bias** | {metrics_summary.get('overall_bias_pct', 0):+.1f}% | {'Over-forecasting' if metrics_summary.get('overall_bias_pct', 0) > 2 else 'Under-forecasting' if metrics_summary.get('overall_bias_pct', 0) < -2 else 'Well-calibrated'} |
| **MAE** | {metrics_summary.get('overall_mae', 0):.2f} | Average absolute error per forecast |

"""

    if new_key_metrics:
        doc_content += """### New vs Existing Key Performance

| Key Type | WAPE | Bias | Forecasts | Unique Keys |
|----------|------|------|-----------|-------------|
"""
        for key_type, km in new_key_metrics.items():
            doc_content += f"| {key_type.replace('_', ' ').title()} | {km['wape']:.2%} | {km['bias_pct']:+.1f}% | {km['n_forecasts']:,} | {km['n_unique_keys']:,} |\n"

        doc_content += """
**Insight**: Comparing new key vs existing key performance helps assess how well the segment assignment process works for cold-start forecasting.
"""

    if horizon_metrics:
        doc_content += """
### Accuracy by Forecast Horizon

| Horizon | WAPE | Bias | MAE | Forecasts |
|---------|------|------|-----|-----------|
"""
        for horizon, hm in sorted(horizon_metrics.items()):
            doc_content += f"| Step {horizon} | {hm['wape']:.2%} | {hm['bias_pct']:+.1f}% | {hm['mae']:.2f} | {hm['n_forecasts']:,} |\n"

    if model_metrics:
        doc_content += """
### Performance by Model Level (Segment)

| Model Level | WAPE | Bias | Forecasts |
|-------------|------|------|-----------|
"""
        for ml, mm in sorted(model_metrics.items(), key=lambda x: x[1]['wape']):
            doc_content += f"| {ml} | {mm['wape']:.2%} | {mm['bias_pct']:+.1f}% | {mm['n_forecasts']:,} |\n"

    doc_content += f"""

---

## 4. Model Retraining Summary

The inference pipeline retrained **{inference_result.models_retrained}** models using:

- **Training Data**: All historical data up to validation end ({config.val_end})
- **Feature Regeneration**: {'Latest source data used for fresh features' if config.design.regenerate_features_on_inference else 'Pre-computed features from training pipeline'}
- **Hyperparameters**: Preserved from original training (no re-tuning)
- **Model Types**: Preserved from original training (same architecture)

### Why Retrain?

1. **Data Freshness**: Incorporate most recent demand patterns
2. **New Key Coverage**: Include new keys in model training
3. **Lag Feature Accuracy**: Ensure recursive forecasting uses correct historical values

---

## 5. Bias Calibration Analysis

"""

    if config.design.apply_bias_calibration:
        doc_content += """Bias calibration is **ENABLED** for this inference run.

### What is Segment-Aware Bias Calibration?

The state-of-the-art calibration approach uses:

1. **Segment-Level Factors**: Different calibration multipliers for each segment
2. **Zero-Fraction Bucketing**: Further refinement based on demand sparsity
3. **Validation-Learned**: Factors computed from validation period performance

### Calibration Impact

Bias calibration adjusts raw model predictions to correct systematic over/under-forecasting:

```
calibrated_forecast = raw_forecast × calibration_factor
```

For segments that tend to over-forecast, factor < 1.0
For segments that tend to under-forecast, factor > 1.0

"""
    else:
        doc_content += """Bias calibration is **DISABLED** for this inference run.

Consider enabling `apply_bias_calibration: True` in config.yaml to improve forecast accuracy through systematic bias correction.
"""

    doc_content += f"""

---

## 6. Key Learnings & Recommendations

### What This Inference Run Tells Us

"""

    # Generate data-driven recommendations
    recommendations = []

    # New key performance
    if 'new_keys' in new_key_metrics and 'existing_keys' in new_key_metrics:
        new_wape = new_key_metrics['new_keys']['wape']
        existing_wape = new_key_metrics['existing_keys']['wape']
        if new_wape > existing_wape * 1.5:
            recommendations.append(f"**Cold-Start Challenge**: New keys have {new_wape:.2%} WAPE vs {existing_wape:.2%} for existing keys. Consider more conservative forecasts for new keys or enhanced segment similarity scoring.")
        else:
            recommendations.append(f"**Effective Segment Assignment**: New keys perform comparably to existing keys ({new_wape:.2%} vs {existing_wape:.2%} WAPE). The clustering-based segment assignment is working well.")

    # Dead key handling
    if inference_result.dead_keys_count > 0:
        dead_pct = inference_result.dead_keys_count / (inference_result.total_keys + inference_result.dead_keys_count) * 100
        recommendations.append(f"**Product Lifecycle**: {inference_result.dead_keys_count:,} keys ({dead_pct:.1f}%) detected as dead/discontinued. Monitor for false positives if recent demand resumption is expected.")

    # Bias check
    if abs(metrics_summary.get('overall_bias_pct', 0)) > 5:
        bias_dir = "over" if metrics_summary.get('overall_bias_pct', 0) > 0 else "under"
        recommendations.append(f"**Bias Alert**: Model tends to {bias_dir}-forecast by {abs(metrics_summary.get('overall_bias_pct', 0)):.1f}%. Review calibration factors or model selection.")
    else:
        recommendations.append("**Well-Calibrated**: Overall bias within acceptable range. Current calibration approach is effective.")

    for i, rec in enumerate(recommendations, 1):
        doc_content += f"{i}. {rec}\n"

    doc_content += f"""

### Actionable Recommendations

1. **For Production Monitoring**:
   - Track forecast accuracy vs this baseline ({metrics_summary.get('overall_wape', 0):.2%} WAPE)
   - Alert if accuracy degrades by >20% from this benchmark
   - Monitor new key performance separately

2. **For Continuous Improvement**:
   - Re-run inference pipeline periodically to incorporate new data
   - Review segment assignments for persistently poor-performing keys
   - Consider per-key overrides for strategic products

3. **For Business Users**:
   - Use horizon-specific accuracy for planning (Step 1 is most accurate)
   - Apply safety stock based on forecast uncertainty
   - Flag new key forecasts for manual review if high-stakes

---

## 7. Output Files Reference

| File | Description |
|------|-------------|
| `inference_forecast.csv` | All generated forecasts with metadata |
| `inference_summary.json` | Pipeline execution summary |
| `training_manifest_backup.csv` | Backup of manifest before new key additions |

### Forecast File Schema

| Column | Description |
|--------|-------------|
| `key` | Unique product/location identifier |
| `{config.timestamp_col}` | Forecast period |
| `origin_period` | When forecast was made |
| `forecast_step` / `lag` | Horizon step (1 = next period) |
| `predicted` | Forecasted demand |
| `actual` | Actual demand (for accuracy tracking) |
| `model_level` | Segment/model used |
| `is_new_key` | Whether key was newly detected |
| `is_dead_key` | Whether key is inactive |

---

*Generated by HarmonIQ Demand-IQ Inference Pipeline*
*Documentation helps data scientists understand forecast generation and make informed decisions*
"""

    # Write documentation
    with open(doc_path, 'w') as f:
        f.write(doc_content)

    logger.info(f"Generated inference documentation: {doc_path}")

    return doc_path


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def run_inference_pipeline(
    config: DemandForecastConfig,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> InferenceResult:
    """
    Run the complete inference pipeline with intelligent new key handling.

    This is the main entry point for generating production forecasts.

    Pipeline Steps:
    ---------------
    1. PHASE 1: Detect new keys (in inference period but not in training_manifest)
    2. PHASE 2-3: For new keys - compute segmentation features, assign to existing segments
    3. PHASE 4: Backup and update training_manifest with new keys
    4. PHASE 4.5: Regenerate features with latest data (if regenerate_features_on_inference=True)
       - Re-runs feature engineering pipeline using source data up to val_end
       - Updates train_features.csv, val_features.csv, test_features.csv
       - Ensures lag features and rolling averages reflect most recent data
    5. PHASE 5: Retrain all models using existing model_type and hyperparameters
       - Uses fresh features (if regenerated) or pre-computed features
       - Trains on data up to val_end
    6. PHASE 6: Generate recursive multi-step forecasts for inference period
    7. PHASE 7: Apply validation-learned bias calibration factors (if enabled)
    8. Save inference_forecast.csv and summary

    Configuration Options:
    ----------------------
    - config.design.regenerate_features_on_inference (default: True)
      When True, regenerates all features before retraining to use latest data.
    - config.design.apply_bias_calibration (default: False)
      When True, applies bias correction learned from validation period.

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object
    output_dir : str, optional
        Override output directory (default: model_artifacts/)
    verbose : bool
        Enable verbose logging

    Returns
    -------
    InferenceResult
        Results including paths to all output files
    """
    start_time = time.time()

    # Setup paths
    artifact_base = config.artifact_base_path
    seg_dir = os.path.join(artifact_base, "seg_output")
    feature_dir = os.path.join(artifact_base, "feature_output")
    model_dir = output_dir or os.path.join(artifact_base, "model_artifacts")

    os.makedirs(model_dir, exist_ok=True)

    # Column names from config
    key_col = config.prediction_key_cols[0] if len(config.prediction_key_cols) == 1 else 'key'
    target_col = config.target_col
    date_col = config.timestamp_col

    logger.info("=" * 60)
    logger.info("INFERENCE PIPELINE")
    logger.info("=" * 60)

    try:
        # =====================================================================
        # PHASE 1: Load artifacts and detect new keys
        # =====================================================================
        logger.info("\n[PHASE 1] Loading artifacts and detecting new keys...")

        # Load source data (supports both CSV and Parquet)
        from utils.agent_utilities import load_source_data
        source_df = load_source_data(config.input_data_path)

        # Create composite key if needed
        if 'key' not in source_df.columns:
            if len(config.prediction_key_cols) == 1 and config.prediction_key_cols[0] in source_df.columns:
                source_df['key'] = source_df[config.prediction_key_cols[0]]
            else:
                source_df['key'] = source_df[config.prediction_key_cols].astype(str).agg('_'.join, axis=1)

        # Validate forward_forecast_exclude_col if configured
        exclude_col = getattr(config.design, 'forward_forecast_exclude_col', '')
        if exclude_col:
            if exclude_col not in source_df.columns:
                raise ValueError(
                    f"forward_forecast_exclude_col='{exclude_col}' is configured but "
                    f"not found in source data. Available columns: "
                    f"{sorted(source_df.columns.tolist())}. "
                    f"Either add this column to your data or set "
                    f"forward_forecast_exclude_col to '' (empty) in your config."
                )
            logger.info(
                f"Forward forecast exclude column '{exclude_col}' found in source data. "
                f"Keys with {exclude_col}=1 will have predictions zeroed before reconciliation."
            )

        # Load training manifest
        manifest_path = os.path.join(feature_dir, "training_manifest.csv")
        manifest_df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest: {len(manifest_df)} keys")

        # Detect new keys, existing keys, and dead keys
        # Use time-format-aware lookback: 52 periods for weekly, 12 for monthly
        time_aware_defaults = config.get_time_aware_defaults()
        dead_key_lookback = time_aware_defaults['dead_key_lookback']
        new_keys, existing_keys, dead_keys, test_df = detect_new_keys(
            source_df=source_df,
            manifest_df=manifest_df,
            key_col=key_col,
            target_col=target_col,
            test_start=config.test_start,
            test_end=config.test_end,
            date_col=date_col,
            dead_key_lookback=dead_key_lookback,
        )

        total_keys = len(new_keys) + len(existing_keys)
        logger.info(f"Dead keys (will get zero forecast): {len(dead_keys)}")

        # =====================================================================
        # PHASE 2 & 3: Handle new keys (if any)
        # =====================================================================
        new_key_assignments = []
        new_keys_by_segment = {}

        if new_keys:
            logger.info(f"\n[PHASE 2] Computing segmentation features for {len(new_keys)} new keys...")

            # Load clustering configuration with error handling
            clustering_metrics_path = os.path.join(seg_dir, "clustering_metrics.json")
            if not os.path.exists(clustering_metrics_path):
                logger.warning(
                    f"clustering_metrics.json not found at {clustering_metrics_path}. "
                    f"New keys ({len(new_keys)}) will use default clustering features. "
                    f"Run segmentation crew first if proper clustering is needed."
                )
                clustering_metrics = {}
            else:
                try:
                    with open(clustering_metrics_path, 'r') as f:
                        clustering_metrics = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load clustering_metrics.json: {e}. Using defaults.")
                    clustering_metrics = {}

            clustering_features = clustering_metrics.get('features_used', [
                'mean', 'cv', 'adi', 'zero_fraction', 'demand_frequency',
                'forecastability_score', 'trend_strength', 'seasonal_strength',
                'autocorr_lag1', 'skewness'
            ])

            # Compute segmentation features for new keys
            new_key_features = compute_segmentation_features_for_new_keys(
                source_df=source_df,
                new_keys=new_keys,
                key_col=key_col,
                target_col=target_col,
                date_col=date_col,
                data_end=config.val_end,  # Use data up to val_end only
                clustering_features=clustering_features,
                time_format=config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week',
            )

            if len(new_key_features) > 0:
                logger.info(f"\n[PHASE 3] Assigning segments to new keys...")

                # Get existing segments
                _new_key_seg_col = 'segment_id' if 'segment_id' in manifest_df.columns else 'model_level'
                existing_segments = list(manifest_df[_new_key_seg_col].unique())

                # Assign segments
                cluster_model_path = os.path.join(seg_dir, "cluster_model.joblib")
                scaler_path = os.path.join(seg_dir, "scaler.joblib")
                cluster_to_segment_map_path = os.path.join(seg_dir, "cluster_to_segment_map.json")

                new_key_assignments = assign_segments_to_new_keys(
                    new_key_features=new_key_features,
                    cluster_model_path=cluster_model_path,
                    scaler_path=scaler_path,
                    clustering_features=clustering_features,
                    existing_segments=existing_segments,
                    cluster_to_segment_map_path=cluster_to_segment_map_path,
                )

                # Count by segment
                for a in new_key_assignments:
                    seg_key = str(a.segment_id)
                    new_keys_by_segment[seg_key] = new_keys_by_segment.get(seg_key, 0) + 1
        else:
            logger.info("\n[PHASE 2-3] No new keys detected, skipping segmentation")

        # =====================================================================
        # PHASE 4: Update training manifest
        # =====================================================================
        logger.info(f"\n[PHASE 4] Updating training manifest...")

        updated_manifest, backup_path, updated_manifest_path = update_training_manifest(
            manifest_path=manifest_path,
            new_key_assignments=new_key_assignments,
            output_dir=model_dir,
        )

        # =====================================================================
        # PHASE 4.5: Regenerate features with latest data (if enabled)
        # =====================================================================
        regenerate_features = config.design.regenerate_features_on_inference

        if regenerate_features:
            logger.info(f"\n[PHASE 4.5] Regenerating features with latest data...")

            success = regenerate_features_for_inference(
                config=config,
                manifest_df=updated_manifest,
                feature_dir=feature_dir,
                key_col=key_col,
                date_col=date_col,
                target_col=target_col,
                source_df=source_df,
            )

            if not success:
                logger.warning("Feature regeneration failed, falling back to pre-computed features")
            else:
                logger.info("Feature regeneration complete - models will use fresh features")
        else:
            logger.info("\n[PHASE 4.5] Skipping feature regeneration (regenerate_features_on_inference=False)")
            logger.info("  Using pre-computed features from training pipeline")

        # =====================================================================
        # PHASE 5: Retrain models
        # =====================================================================
        logger.info(f"\n[PHASE 5] Retraining models...")

        # Load model specs
        specs_path = os.path.join(model_dir, "final_model_specs.json")
        with open(specs_path, 'r') as f:
            model_specs = json.load(f)

        retrained_models, models_retrained = retrain_all_models(
            config=config,
            manifest_df=updated_manifest,
            model_specs=model_specs,
            feature_dir=feature_dir,
            model_dir=model_dir,
            target_col=target_col,
            key_col=key_col,
            date_col=date_col,
        )

        # =====================================================================
        # PHASE 6: Generate inference forecasts
        # =====================================================================
        logger.info(f"\n[PHASE 6] Generating inference forecasts...")

        # Load bias calibration factors if enabled
        apply_bias_calibration = config.design.apply_bias_calibration
        calibration_factors = None
        segment_calibrations = None

        if apply_bias_calibration:
            # Load basic bias calibration factors
            calibration_path = os.path.join(model_dir, "bias_calibration.json")
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r') as f:
                    calibration_factors = json.load(f)
                logger.info("Loaded bias calibration factors")

            # STATE-OF-THE-ART: Load segment-aware calibration factors
            segment_cal_path = os.path.join(model_dir, "segment_calibrations.json")
            if os.path.exists(segment_cal_path):
                with open(segment_cal_path, 'r') as f:
                    segment_calibrations = json.load(f)
                logger.info(f"Loaded segment-aware calibration for {len(segment_calibrations)} segments")

            # Proceed if we have either type of calibration
            if not calibration_factors and not segment_calibrations:
                logger.warning(
                    "=" * 60 + "\n"
                    "WARNING: Bias calibration is ENABLED in config but NO calibration factors were found!\n"
                    f"  Checked: {calibration_path}\n"
                    f"  Checked: {segment_cal_path}\n"
                    "  Forecasts will NOT be calibrated and may have systematic bias.\n"
                    "  To fix: Run training crew with apply_bias_calibration=true to generate factors.\n"
                    "=" * 60
                )
                apply_bias_calibration = False

        # ---------------------------------------------------------------
        # Branch: direct multi-horizon vs legacy recursive forecasting.
        # ---------------------------------------------------------------
        use_direct_mh = bool(getattr(config.design, "use_direct_multi_horizon", False))
        if use_direct_mh:
            logger.info(
                "\n[PHASE 6] Using DIRECT MULTI-HORIZON inference "
                "(config.design.use_direct_multi_horizon=True)"
            )
            try:
                forecasts_df, total_forecasts = _generate_direct_multihorizon_forecasts(
                    config=config,
                    feature_dir=feature_dir,
                    seg_dir=seg_dir,
                    output_dir=model_dir,
                    key_col=key_col,
                    date_col=date_col,
                    target_col=target_col,
                )
                logger.info(
                    "Direct multi-horizon produced %d forecasts", total_forecasts
                )
            except Exception as e:
                # Log the full traceback (not just str(e)) so we can see
                # exactly which line in the DMH path raised — without
                # this, errors like "Circular reference detected" hide
                # their origin and we end up silently falling back to
                # the slow recursive path forever.
                import traceback as _tb
                logger.error(
                    "Direct multi-horizon path failed (%s: %s), "
                    "falling back to recursive.  Full traceback:\n%s",
                    type(e).__name__,
                    e,
                    _tb.format_exc(),
                )
                use_direct_mh = False

        if not use_direct_mh:
            forecasts_df, total_forecasts = generate_forward_forecasts(
                config=config,
                retrained_models=retrained_models,
                manifest_df=updated_manifest,
                model_specs=model_specs,
                feature_dir=feature_dir,
                output_dir=model_dir,
                apply_bias_calibration=apply_bias_calibration,
                calibration_factors=calibration_factors,
                segment_calibrations=segment_calibrations,
                source_df=source_df,
            )

        # Add zero forecasts for dead keys
        if dead_keys:
            logger.info(f"Creating zero forecasts for {len(dead_keys)} dead keys...")
            dead_key_forecasts = create_dead_key_forecasts(
                dead_keys=dead_keys,
                test_start=config.test_start,
                test_end=config.test_end,
                key_col=key_col,
                date_col=date_col,
                target_col=target_col,
                forecast_horizon=config.forecast_horizon,
                source_df=source_df,
                time_format=config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week',
            )
            if not dead_key_forecasts.empty:
                forecasts_df = pd.concat([forecasts_df, dead_key_forecasts], ignore_index=True)
                total_forecasts += len(dead_key_forecasts)
                logger.info(f"Added {len(dead_key_forecasts)} zero forecasts for dead keys")

        # =====================================================================
        # PHASE 6.4: ZERO OUT EXCLUDED KEYS (before reconciliation)
        # =====================================================================
        exclude_col = getattr(config.design, 'forward_forecast_exclude_col', '')
        if exclude_col and len(forecasts_df) > 0:
            logger.info(f"\n[PHASE 6.4] Zeroing predictions for keys with {exclude_col}=1...")

            # The source data contains rows for the test/inference periods with
            # the exclude flag at the (key, period) level. Merge the flag onto
            # forecasts so each (key, period) gets its own exclude decision.
            exclude_lookup = (
                source_df[[key_col, date_col, exclude_col]]
                .drop_duplicates(subset=[key_col, date_col], keep='last')
            )

            pre_exclude_count = (forecasts_df['predicted'] > 0).sum()
            forecasts_df = forecasts_df.merge(
                exclude_lookup, on=[key_col, date_col], how='left'
            )
            # If a (key, period) has no row in source → keep prediction (fill 0)
            forecasts_df[exclude_col] = forecasts_df[exclude_col].fillna(0)

            # Zero out predictions where exclude flag = 1
            exclude_mask = forecasts_df[exclude_col] == 1
            n_excluded = exclude_mask.sum()
            n_keys_excluded = forecasts_df.loc[exclude_mask, key_col].nunique()
            forecasts_df.loc[exclude_mask, 'predicted'] = 0.0

            # Drop the temporary exclude column from forecasts
            forecasts_df = forecasts_df.drop(columns=[exclude_col])

            logger.info(
                f"Excluded {n_excluded} (key, period) predictions across "
                f"{n_keys_excluded} keys with {exclude_col}=1. "
                f"Active predictions: "
                f"{pre_exclude_count} → {(forecasts_df['predicted'] > 0).sum()}"
            )

        # =====================================================================
        # PHASE 6.45: YOY TREND ADJUSTMENT (before reconciliation)
        # =====================================================================
        enable_yoy_trend = getattr(config.design, 'enable_yoy_trend_adjustment', False)
        yoy_trend_diagnostics = {}

        if enable_yoy_trend and len(forecasts_df) > 0:
            logger.info("\n[PHASE 6.45] Applying YOY trend adjustment...")
            try:
                from utils.reconciliation import apply_yoy_trend_adjustment

                tf = config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week'

                forecasts_df, yoy_trend_diagnostics = apply_yoy_trend_adjustment(
                    forecasts_df=forecasts_df,
                    source_df=source_df,
                    date_col=date_col,
                    target_col=target_col,
                    key_col=key_col,
                    test_start=config.test_start,
                    forecast_horizon=config.forecast_horizon,
                    time_format=tf,
                    min_history_periods=getattr(config.design, 'yoy_trend_min_history_periods', 104),
                    scaling_min=getattr(config.design, 'yoy_trend_scaling_min', 0.5),
                    scaling_max=getattr(config.design, 'yoy_trend_scaling_max', 2.0),
                )
                logger.info(f"YOY trend adjustment complete: {yoy_trend_diagnostics.get('keys_adjusted', 0)} keys adjusted")
            except Exception as e:
                logger.warning(f"YOY trend adjustment failed: {e}. Using original forecasts.")
                yoy_trend_diagnostics = {'yoy_trend_adjustment_applied': False, 'error': str(e)}
        else:
            if not enable_yoy_trend:
                logger.info("YOY trend adjustment is disabled in config")

        # =====================================================================
        # PHASE 6.5: TOP-DOWN CATEGORY RECONCILIATION
        # =====================================================================
        enable_reconciliation = getattr(config.design, 'enable_top_down_reconciliation', False)
        reconciliation_diagnostics = {}

        if enable_reconciliation and len(forecasts_df) > 0:
            logger.info("\n[PHASE 6.5] Applying top-down category reconciliation...")
            try:
                from utils.reconciliation import train_and_reconcile

                tf = config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week'
                forecast_periods = sorted(forecasts_df[date_col].unique().tolist())

                # Phase 3: Use configured reconciliation method and hierarchy column.
                # Resolve hierarchies via the single-source-of-truth resolver so
                # feature-engineering, training and reconciliation all agree on
                # which columns are hierarchies for this run.
                _recon_method = getattr(config.design, 'reconciliation_method', 'top_down')
                _hier_col = None
                try:
                    from utils.hierarchy_resolution import resolve_hierarchies
                    _seg_dir = os.path.join(config.artifact_base_path, 'seg_output')
                    _h = resolve_hierarchies(config=config, source_df=source_df, seg_dir=_seg_dir)
                    _hier_col = _h.primary_product_col
                    if _recon_method != 'top_down' and not _hier_col:
                        logger.warning(
                            "reconciliation_method=%s is configured but no hierarchy "
                            "column was resolved — reconciliation will fall back to "
                            "top_down. Set design.hierarchy_detection.candidate_cols "
                            "or design.hierarchy_cols to enable %s.",
                            _recon_method, _recon_method,
                        )
                except Exception as _hx:
                    logger.warning("Hierarchy resolution failed in inference: %s", _hx)
                    _hier_cols = getattr(config.design, 'hierarchy_cols', [])
                    if isinstance(_hier_cols, dict):
                        _hier_col = (_hier_cols.get('product') or [None])[-1] if _hier_cols.get('product') else None
                    elif _hier_cols:
                        _hier_col = _hier_cols[-1]

                forecasts_df, reconciliation_diagnostics = train_and_reconcile(
                    forecasts_df=forecasts_df,
                    source_df=source_df,
                    target_col=target_col,
                    date_col=date_col,
                    key_col=key_col,
                    forecast_periods=forecast_periods,
                    time_format=tf,
                    train_cutoff=config.val_end,
                    ratio_min=getattr(config.design, 'reconciliation_ratio_min', 0.5),
                    ratio_max=getattr(config.design, 'reconciliation_ratio_max', 2.0),
                    trust_band=getattr(config.design, 'reconciliation_trust_band', 0.1),
                    changepoint_prior_scale=getattr(config.design, 'reconciliation_changepoint_prior', 0.05),
                    yoy_max_deviation=getattr(config.design, 'reconciliation_yoy_max_deviation', 0.15),
                    method=_recon_method,
                    hierarchy_col=_hier_col,
                )
                logger.info(f"Reconciliation complete: {reconciliation_diagnostics}")
            except Exception as e:
                logger.warning(f"Top-down reconciliation failed: {e}. Using original forecasts.")
                reconciliation_diagnostics = {'reconciliation_applied': False, 'error': str(e)}
        else:
            if not enable_reconciliation:
                logger.info("Top-down reconciliation is disabled in config")

        # =====================================================================
        # PHASE 6.7: SPLY POST-PREDICTION BIAS CORRECTION (final adjustment)
        # =====================================================================
        enable_sply = getattr(config.design, 'enable_sply_correction', True)
        sply_diagnostics = {}

        if enable_sply and len(forecasts_df) > 0:
            logger.info("\n[PHASE 6.7] Applying SPLY (Same Period Last Year) bias correction...")
            try:
                from utils.sply_correction import apply_sply_correction, learn_sply_blend_weights

                tf = config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week'

                # forecasts_df always uses 'key' as the column name (from batch_recursive_forecast),
                # but source_df uses key_col (e.g. 'Model_Hierarchy'). Align both to 'key'.
                sply_key_col = 'key'
                sply_source = source_df
                if key_col != 'key' and key_col in source_df.columns and 'key' not in source_df.columns:
                    sply_source = source_df.copy()
                    sply_source['key'] = sply_source[key_col]

                # Learn blend weights from validation predictions (if available)
                sply_weights = None
                val_preds_path = os.path.join(model_dir, "validation_predictions.csv")
                if os.path.exists(val_preds_path):
                    val_preds_df = pd.read_csv(val_preds_path)
                    # Fix column name mismatch: validation_predictions.csv uses
                    # 'actual' but config target_col may be different (e.g. 'Actuals')
                    if 'actual' in val_preds_df.columns and target_col not in val_preds_df.columns:
                        val_preds_df[target_col] = val_preds_df['actual']
                    if 'predicted' in val_preds_df.columns and sply_key_col in val_preds_df.columns:
                        sply_weights = learn_sply_blend_weights(
                            val_predictions_df=val_preds_df,
                            source_df=sply_source,
                            key_col=sply_key_col,
                            date_col=date_col,
                            target_col=target_col,
                            predicted_col='predicted',
                            time_format=tf,
                            train_end=config.train_end,
                        )

                forecasts_df, sply_diagnostics = apply_sply_correction(
                    forecasts_df=forecasts_df,
                    source_df=sply_source,
                    key_col=sply_key_col,
                    date_col=date_col,
                    target_col=target_col,
                    predicted_col='predicted',
                    blend_weights=sply_weights,
                    time_format=tf,
                    train_end=config.train_end,
                )
                logger.info(f"SPLY correction: {sply_diagnostics.get('n_rows_corrected', 0)} rows, "
                             f"impact={sply_diagnostics.get('correction_impact_pct', 0):+.1f}%")
            except Exception as e:
                logger.warning(f"SPLY correction failed: {e}. Using uncorrected forecasts.")
                sply_diagnostics = {'sply_correction_applied': False, 'error': str(e)}
        else:
            if not enable_sply:
                logger.info("SPLY correction is disabled in config")

        # =====================================================================
        # PHASE 6.8: MOQ (Minimum Order Quantity) post-processing
        # =====================================================================
        # Adds a ``prediction_post_moq`` column alongside ``predicted``: for
        # SKUs whose historical order pattern shows a strong fixed-multiple
        # MOQ (NPD / reintroduced / highly-promoted / MOQ-driven items), the
        # forecast is rounded to that multiple so the DIQ-facing output
        # respects the real ordering rhythm. For SKUs without an MOQ
        # pattern the column mirrors ``predicted`` — downstream can read
        # ``prediction_post_moq`` unconditionally.
        if getattr(config.design, 'apply_moq_postprocessing', True):
            try:
                from utils.moq_postprocessing import apply_moq_postprocessing
                forecasts_df = apply_moq_postprocessing(
                    forecasts_df,
                    config,
                    history_df=source_df if 'source_df' in locals() else None,
                    forecast_col='predicted',
                    output_col='prediction_post_moq',
                )
            except Exception as moq_err:
                logger.warning(
                    f"MOQ post-processing failed: {moq_err}. "
                    "Forecasts will be saved without the prediction_post_moq column."
                )
        else:
            logger.info("MOQ post-processing is disabled in config")

        # =====================================================================
        # DEAD-KEY CLEANUP (single simple filter — runs after all post-processing)
        # =====================================================================
        # Per user requirement (2026-05-06): "in the inference_forecast.csv we
        # already have the dead-key indicator, just remove rows where it's a
        # dead key but the forecast is positive — that's it."
        #
        # Trust the `is_dead_key` column that's already on the frame.  Any
        # row whose key is flagged as dead anywhere in this output should
        # only contribute zero forecasts — the dead-key zero is what we
        # want, the non-zero rows for those keys (whether duplicates from
        # the model path, or zeros that got recomputed by yoy/recon/sply/
        # moq) are removed.
        #
        # This single filter handles BOTH scenarios that earlier code
        # tried to address with multiple layers:
        #   * Duplicate (one zero + one non-zero for same key, period):
        #     the zero row stays, the non-zero row is dropped because
        #     the key is in the dead set
        #   * Zero-got-recomputed-to-non-zero by post-processing:
        #     the row is dropped because is_dead_key=True survives the
        #     recomputation (post-processing modifies `predicted`,
        #     not `is_dead_key`)
        #
        # Result: each (key, period) for a dead key has at most one
        # zero forecast row, and no non-zero phantom demand.
        if 'is_dead_key' in forecasts_df.columns and len(forecasts_df) > 0:
            _dead_keys_in_output = set(
                forecasts_df.loc[forecasts_df['is_dead_key'] == True, key_col]
                .astype(str)
                .unique()
            )
            if _dead_keys_in_output:
                _bogus = (
                    forecasts_df[key_col].astype(str).isin(_dead_keys_in_output)
                    & (forecasts_df['predicted'] > 0)
                )
                _n_bogus = int(_bogus.sum())
                if _n_bogus > 0:
                    forecasts_df = forecasts_df[~_bogus].reset_index(drop=True)
                    logger.info(
                        f"Dead-key cleanup: removed {_n_bogus} positive-"
                        f"forecast rows for {len(_dead_keys_in_output)} "
                        f"keys flagged as dead (is_dead_key=True ⟹ "
                        f"predicted should be 0)"
                    )

        # =====================================================================
        # PHASE 7: Save results
        # =====================================================================
        logger.info(f"\n[PHASE 7] Saving results...")

        # Save forecasts (CSV always goes to model_artifacts/)
        forecasts_path = os.path.join(model_dir, "inference_forecast.csv")
        forecasts_df.to_csv(forecasts_path, index=False)
        logger.info(f"Saved forecasts CSV: {forecasts_path}")

        # If output_folder_path is configured, also write a Parquet copy there
        parquet_forecasts_path = ""
        output_folder = getattr(config, 'output_folder_path', '')
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            parquet_forecasts_path = os.path.join(output_folder, "inference_forecast.parquet")
            forecasts_df.to_parquet(parquet_forecasts_path, index=False, engine="pyarrow")
            logger.info(f"Saved forecasts Parquet: {parquet_forecasts_path}")

        # Create summary
        elapsed_time = time.time() - start_time

        output_files = {
            'forecasts': forecasts_path,
            'manifest_backup': backup_path,
            'updated_manifest': updated_manifest_path,
        }
        if parquet_forecasts_path:
            output_files['forecasts_parquet'] = parquet_forecasts_path

        summary = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_time,
            'total_keys': total_keys,
            'new_keys_count': len(new_keys),
            'existing_keys_count': len(existing_keys),
            'dead_keys_count': len(dead_keys),
            'new_keys_by_segment': new_keys_by_segment,
            'models_retrained': models_retrained,
            'forecast_horizon': config.forecast_horizon,
            'total_forecasts': total_forecasts,
            'test_period': {
                'start': config.test_start,
                'end': config.test_end,
            },
            'bias_calibration_applied': apply_bias_calibration,
            'yoy_trend_adjustment': yoy_trend_diagnostics,
            'reconciliation': reconciliation_diagnostics,
            'output_files': output_files,
        }

        summary_path = os.path.join(model_dir, "inference_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_path}")

        # =====================================================================
        # SUCCESS
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total active keys: {total_keys}")
        logger.info(f"New keys: {len(new_keys)}")
        logger.info(f"Existing keys: {len(existing_keys)}")
        logger.info(f"Dead keys (zero forecast): {len(dead_keys)}")
        logger.info(f"Models retrained: {models_retrained}")
        logger.info(f"Forecasts generated: {total_forecasts}")
        logger.info(f"Elapsed time: {elapsed_time:.1f}s")

        # Create result object
        result = InferenceResult(
            success=True,
            forecasts_path=forecasts_path,
            forecasts_parquet_path=parquet_forecasts_path,
            summary_path=summary_path,
            total_keys=total_keys,
            new_keys_count=len(new_keys),
            existing_keys_count=len(existing_keys),
            dead_keys_count=len(dead_keys),
            new_keys_by_segment=new_keys_by_segment,
            models_retrained=models_retrained,
            forecast_horizon=config.forecast_horizon,
            total_forecasts=total_forecasts,
            updated_manifest_path=updated_manifest_path,
            backup_manifest_path=backup_path,
            retrained_models_dir=model_dir,
        )

        # Generate documentation insights guide (if enabled)
        enable_insights = getattr(config.design, 'enable_insights_reports', False) if hasattr(config, 'design') else False
        if enable_insights:
            try:
                doc_path = generate_inference_documentation(
                    inference_result=result,
                    forecasts_df=forecasts_df,
                    config=config,
                    output_dir=model_dir,
                )
                logger.info(f"Generated insights guide: {doc_path}")
            except Exception as doc_err:
                logger.warning(f"Documentation generation failed: {doc_err}")
        else:
            logger.info("SKIPPING inference insights guide (enable_insights_reports=False)")

        return result

    except Exception as e:
        import traceback
        error_msg = f"Inference pipeline failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)

        return InferenceResult(
            success=False,
            error_message=error_msg,
        )


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# These aliases maintain compatibility with existing code that may reference
# the old "forward_forecast" naming convention.

ForwardForecastResult = InferenceResult
run_forward_forecast_pipeline = run_inference_pipeline
