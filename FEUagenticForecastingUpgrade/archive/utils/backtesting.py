# utils/backtesting.py
"""
Rolling-Origin Backtesting Module

This module provides rolling-origin (walk-forward) backtesting functionality
that iteratively runs the inference pipeline, each time rolling the training
cutoff forward by one period.

Rolling-Origin Backtesting:
- Origin 1: Train up to val_end, forecast test_start to test_start + horizon
- Origin 2: Train up to test_start, forecast test_start+1 to test_start+1 + horizon
- Origin 3: Train up to test_start+1, forecast test_start+2 to test_start+2 + horizon
- ... continues until no more viable forecast periods remain

This produces multi-origin forecasts that can be used to:
1. Assess forecast accuracy at different forecast horizons
2. Evaluate model stability over time
3. Generate more robust accuracy metrics

Usage:
    from utils.backtesting import run_rolling_origin_backtest, BacktestResult
    from config.schema import load_config_from_yaml

    config = load_config_from_yaml("config/config.yaml")
    result = run_rolling_origin_backtest(config)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.schema import DemandForecastConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BacktestResult:
    """Result container for rolling-origin backtest."""
    success: bool
    forecasts_path: str = ""
    summary_path: str = ""
    metrics_path: str = ""
    total_origins: int = 0
    total_forecasts: int = 0
    forecast_horizon: int = 0
    origins: List[str] = field(default_factory=list)
    error_message: str = ""
    elapsed_seconds: float = 0.0
    # Per-origin details
    origin_details: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class OriginResult:
    """Result for a single origin in the backtest."""
    origin_period: str
    train_cutoff: str
    forecast_start: str
    forecast_end: str
    num_forecasts: int
    elapsed_seconds: float
    success: bool
    error_message: str = ""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_time_format(period_str: str) -> str:
    """
    Auto-detect whether a period string is YYYYWW (year_week) or YYYYMM (year_month).

    Logic: If the last 2 digits are > 12, it must be a week number.
    If <= 12, we check context but default to year_week for backward compatibility
    unless explicitly configured.

    Parameters
    ----------
    period_str : str
        A 6-digit period string like '202501' or '202545'

    Returns
    -------
    str
        'year_week' or 'year_month'
    """
    s = str(period_str).strip()
    # Handle dash-separated
    if '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            sub = int(parts[1])
            return 'year_month' if sub <= 12 and len(parts[1]) <= 2 else 'year_week'
    # Handle 6-digit
    s_clean = s.replace(".", "").split(".")[0]
    if len(s_clean) == 6 and s_clean.isdigit():
        sub = int(s_clean[4:])
        if sub > 12:
            return 'year_week'
        return 'year_month'
    # Default to year_week for backward compatibility
    return 'year_week'


def parse_period(period: str) -> Tuple[int, int]:
    """Parse a YYYYWW, YYYYMM, YYYY-WW, or YYYY-MM string to (year, sub_period) tuple."""
    s = str(period).strip()
    # Handle dash-separated: YYYY-WW or YYYY-MM (e.g., "2025-45", "2025-03")
    if '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
    # Handle 6-digit: YYYYWW or YYYYMM (e.g., "202545")
    s_clean = s.replace(".", "").split(".")[0]
    if len(s_clean) == 6 and s_clean.isdigit():
        return int(s_clean[:4]), int(s_clean[4:])
    # Handle 7-digit with leading zero stripped: try as int
    try:
        val = int(float(s_clean))
        if val > 100000:
            return val // 100, val % 100
    except (ValueError, TypeError):
        pass
    raise ValueError(f"Invalid period format: {period}")


# Backward-compatible alias
parse_year_week = parse_period


def format_period(year: int, sub: int, use_dash: bool = False) -> str:
    """Format (year, sub_period) to period string.

    If use_dash=True: returns 'YYYY-WW' (e.g., '2025-45')
    If use_dash=False: returns 'YYYYWW' (e.g., '202545')
    """
    if use_dash:
        return f"{year}-{sub:02d}"
    return f"{year}{sub:02d}"


# Backward-compatible alias
format_year_week = format_period


def increment_period(period: str, steps: int = 1, time_format: str = 'year_week') -> str:
    """
    Increment a period value by n steps.

    Parameters
    ----------
    period : str
        Period string (YYYYWW or YYYYMM)
    steps : int
        Number of periods to increment
    time_format : str
        'year_week' (max 52) or 'year_month' (max 12)

    Returns
    -------
    str
        Incremented period string
    """
    year, sub = parse_period(period)
    max_sub = 12 if time_format == 'year_month' else 52

    for _ in range(steps):
        sub += 1
        if sub > max_sub:
            sub = 1
            year += 1

    # Preserve input format (dash vs no-dash)
    use_dash = '-' in str(period)
    return format_period(year, sub, use_dash=use_dash)


# Backward-compatible alias (defaults to year_week)
def increment_year_week(yw: str, weeks: int = 1) -> str:
    """Increment a year_week value by n weeks. Legacy wrapper."""
    return increment_period(yw, weeks, time_format='year_week')


def decrement_period(period: str, steps: int = 1, time_format: str = 'year_week') -> str:
    """
    Decrement a period value by n steps.

    Parameters
    ----------
    period : str
        Period string (YYYYWW or YYYYMM)
    steps : int
        Number of periods to decrement
    time_format : str
        'year_week' (max 52) or 'year_month' (max 12)

    Returns
    -------
    str
        Decremented period string
    """
    year, sub = parse_period(period)
    max_sub = 12 if time_format == 'year_month' else 52

    for _ in range(steps):
        sub -= 1
        if sub < 1:
            sub = max_sub
            year -= 1

    return format_period(year, sub)


# Backward-compatible alias
def decrement_year_week(yw: str, weeks: int = 1) -> str:
    """Decrement a year_week value by n weeks. Legacy wrapper."""
    return decrement_period(yw, weeks, time_format='year_week')


def count_periods_between(start: str, end: str, time_format: str = 'year_week') -> int:
    """Count number of periods between two values (inclusive)."""
    start_year, start_sub = parse_period(start)
    end_year, end_sub = parse_period(end)
    max_sub = 12 if time_format == 'year_month' else 52

    start_total = start_year * max_sub + start_sub
    end_total = end_year * max_sub + end_sub

    return end_total - start_total + 1


# Backward-compatible alias
def count_weeks_between(start_yw: str, end_yw: str) -> int:
    """Count weeks between two year_week values. Legacy wrapper."""
    return count_periods_between(start_yw, end_yw, time_format='year_week')


# ---------------------------------------------------------------------------
# Schema normalizer
# ---------------------------------------------------------------------------

# Columns that MUST be string across the pipeline so the parquet writer
# doesn't blow up with 'Expected bytes, got a int object'.  Period columns
# drift between str and int when forecasts from different origins / dead-key
# generators / source-data merges are concatenated, so we force them to str
# at one chokepoint before each origin's forecasts_df leaves
# `run_single_origin_inference`.
_PERIOD_STR_COLS: Tuple[str, ...] = (
    "origin_period",
    "SnapshotTimePeriod",
    "model_level",
    "model_name",
    "model_params",
)

_NUMERIC_FLOAT_COLS: Tuple[str, ...] = ("predicted", "actual")

_NUMERIC_INT_COLS: Tuple[str, ...] = ("forecast_step", "lag", "origin_idx")

_BOOL_COLS: Tuple[str, ...] = ("is_new_key", "is_dead_key")


def _normalize_forecasts_schema(df: "pd.DataFrame", date_col: str) -> "pd.DataFrame":
    """Force a consistent dtype on every column that crosses the legacy /
    DMH / dead-key concat boundary.  Returns the same df in-place-mutated.

    Rules applied (only to columns that exist):
      - date_col + period/string-label columns -> str
      - numeric prediction columns             -> float (NaN-safe)
      - ordinal step/lag columns               -> int (NaN-filled)
      - boolean flags                          -> bool
    """
    if df is None or len(df) == 0:
        return df

    if date_col in df.columns:
        df[date_col] = df[date_col].astype(str)

    for col in _PERIOD_STR_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in _NUMERIC_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in _NUMERIC_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df


def generate_period_range(start: str, end: str, time_format: str = 'year_week') -> List[str]:
    """
    Generate list of period values from start to end (inclusive).

    Parameters
    ----------
    start : str
        Start period (YYYYWW or YYYYMM)
    end : str
        End period (YYYYWW or YYYYMM)
    time_format : str
        'year_week' or 'year_month'

    Returns
    -------
    List[str]
        List of period strings
    """
    max_periods = 200 if time_format == 'year_week' else 120
    periods = []
    current = start

    while True:
        periods.append(current)
        if current == end:
            break
        current = increment_period(current, 1, time_format)

        if len(periods) > max_periods:
            logger.warning(f"Period range exceeded {max_periods}, stopping")
            break

    return periods


# Backward-compatible alias
def generate_week_range(start_yw: str, end_yw: str) -> List[str]:
    """Generate week range. Legacy wrapper."""
    return generate_period_range(start_yw, end_yw, time_format='year_week')


# =============================================================================
# FEATURE REGENERATION FOR BACKTESTING
# =============================================================================

def regenerate_features_for_backtest_origin(
    config: DemandForecastConfig,
    manifest_df: pd.DataFrame,
    train_cutoff: str,
    output_dir: str,
    key_col: str,
    date_col: str,
    target_col: str,
    source_df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Regenerate features for a specific backtest origin using data up to train_cutoff.

    This ensures that lag features, rolling averages, and derived features are
    computed using only data available at the origin point, preventing data leakage.

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object
    manifest_df : pd.DataFrame
        Training manifest with all keys
    train_cutoff : str
        The cutoff date for this backtest origin (e.g., "202528")
    output_dir : str
        Directory to save regenerated feature files
    key_col : str
        Key column name
    date_col : str
        Date column name
    target_col : str
        Target column name
    source_df : pd.DataFrame, optional
        Pre-loaded + memory-optimised source dataframe. When provided the
        function skips its own ``load_source_data`` call — this is the
        fast path used by ``run_rolling_origin_backtest`` to amortise the
        cost of parsing a 1+ GB source CSV across every origin.

    Returns
    -------
    bool
        True if features were regenerated successfully
    """
    from utils.feature_engineering import run_leakage_free_feature_pipeline

    logger.info(f"  Regenerating features with data up to {train_cutoff}...")

    # Use pre-loaded source_df when available (SW1: amortise CSV parse cost
    # across origins). Work on a lightweight copy so in-place operations
    # inside the pipeline don't leak back to the shared instance held by
    # the outer backtest driver.
    if source_df is None:
        from utils.agent_utilities import load_source_data
        source_df = load_source_data(config.input_data_path)
    else:
        # Shallow copy — column arrays are still shared until this
        # function modifies them, at which point pandas does copy-on-write.
        source_df = source_df.copy(deep=False)

    # Ensure 'key' column exists — all intermediate files use standardised 'key'
    if 'key' not in source_df.columns:
        if len(config.prediction_key_cols) == 1 and config.prediction_key_cols[0] in source_df.columns:
            source_df['key'] = source_df[config.prediction_key_cols[0]]
        else:
            source_df['key'] = source_df[config.prediction_key_cols].astype(str).agg('_'.join, axis=1)

    # Get all keys from manifest
    all_keys = manifest_df['key'].unique()

    # Filter source data to only include keys in manifest
    source_df = source_df[source_df['key'].isin(all_keys)]

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

        segments_df = segments_df[segment_cols_to_merge].drop_duplicates(subset=['key'])

        # Drop columns from source_df that will be added from segments_df to avoid duplicates
        cols_to_drop = [c for c in segment_cols_to_merge if c != 'key' and c in source_df.columns]
        if cols_to_drop:
            logger.debug(f"Dropping existing columns before segment merge: {cols_to_drop}")
            source_df = source_df.drop(columns=cols_to_drop)

        source_df = source_df.merge(segments_df, on='key', how='left')

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
    if config.price_features and config.price_features.categorical:
        categorical_cols.extend(config.price_features.categorical)
    if config.promo_features and config.promo_features.categorical:
        categorical_cols.extend(config.promo_features.categorical)
    if config.holiday_features and config.holiday_features.categorical:
        categorical_cols.extend(config.holiday_features.categorical)
    if config.weather_features and config.weather_features.categorical:
        categorical_cols.extend(config.weather_features.categorical)
    categorical_cols = [c for c in categorical_cols if c in source_df.columns]

    # For backtesting, we need to adjust the date ranges based on train_cutoff
    # The test period starts after train_cutoff
    tf = getattr(config, 'time_format', 'year_week')
    if tf == 'auto' or tf == 'date':
        tf = 'year_week'  # Default to year_week for auto/date
    forecast_start = increment_period(train_cutoff, 1, time_format=tf)

    # Calculate a reasonable test_end (train_cutoff + forecast_horizon)
    test_end = increment_period(train_cutoff, config.forecast_horizon, time_format=tf)

    # =========================================================================
    # Load feature strategy decision from training to ensure consistent features.
    # Training uses run_leakage_free_feature_pipeline via feature_reasoning.py
    # which produces log-transformed features, external feature lags, etc.
    # Backtesting must use the SAME pipeline to avoid feature mismatch.
    # =========================================================================
    strategy_path = os.path.join(config.artifact_base_path, "feature_output", "feature_strategy_decision.json")
    strategy_params = {}
    if os.path.exists(strategy_path):
        with open(strategy_path, 'r') as f:
            strategy_params = json.load(f)
        logger.info(f"  Loaded feature strategy from: {strategy_path}")
    else:
        logger.warning(f"  Feature strategy not found at {strategy_path}, using defaults")

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
    # features in the regenerated df — which is a SILENT FEATURE MISMATCH
    # vs. the trained model specs (segment-pooled global_local models list
    # 12 hierarchy features each). The result: retrained models at each
    # backtest origin are trained on a DIFFERENT feature set than the
    # original models, degrading accuracy without any warning.
    # =========================================================================
    _bt_hierarchy_cols: list = []
    if getattr(config.design, 'enable_hierarchy_features', True):
        try:
            from utils.hierarchy_resolution import resolve_hierarchies
            _seg_dir = os.path.join(config.artifact_base_path, 'seg_output')
            _h = resolve_hierarchies(config=config, source_df=source_df, seg_dir=_seg_dir)
            _bt_hierarchy_cols = list(_h.product) if _h.product else list(_h.flat)
            if _bt_hierarchy_cols:
                logger.info(
                    f"  Resolved hierarchy_cols for backtest regen (source={_h.source}): "
                    f"{_bt_hierarchy_cols}"
                )
        except Exception as _hcol_err:
            logger.warning(f"  Hierarchy resolution for backtest regen failed: {_hcol_err}")

    try:
        result = run_leakage_free_feature_pipeline(
            df=source_df,
            key_cols=[key_col],
            date_col=date_col,
            target_col=target_col,
            train_start=config.train_start,
            train_end=config.train_end,
            val_start=config.val_start,
            val_end=train_cutoff,  # Use train_cutoff as the val_end for this origin
            test_start=forecast_start,
            test_end=test_end,
            forecast_lag=forecast_lag,
            segment_col=segment_col,
            categorical_cols=categorical_cols if categorical_cols else None,
            output_dir=output_dir,
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
            # artifact above so backtest regen matches training exactly.
            hierarchy_cols=_bt_hierarchy_cols if _bt_hierarchy_cols else strategy_params.get('hierarchy_cols'),
            # Phase 6: Rich features + history embeddings
            enable_rich_features=strategy_params.get('enable_rich_features', True),
            future_unknown_features=strategy_params.get('future_unknown_features'),
        )

        logger.info(f"  Feature regeneration complete: {result.n_features_created} features")
        # Free source_df after feature generation to reclaim memory
        del source_df
        gc.collect()
        return True

    except Exception as e:
        logger.error(f"  Feature regeneration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        del source_df
        gc.collect()
        return False


# =============================================================================
# CORE BACKTESTING LOGIC
# =============================================================================

def generate_backtest_origins(
    val_end: str,
    test_start: str,
    test_end: str,
    forecast_horizon: int,
    time_format: str = 'year_week',
) -> List[Dict[str, str]]:
    """
    Generate list of backtest origins with their train cutoffs and forecast windows.

    Each origin specifies:
    - train_cutoff: Last period to include in training (inclusive)
    - forecast_start: First period to forecast
    - forecast_end: Last period to forecast

    Parameters
    ----------
    val_end : str
        End of validation period (YYYYWW or YYYYMM)
    test_start : str
        Start of test period (YYYYWW or YYYYMM)
    test_end : str
        End of test period (YYYYWW or YYYYMM)
    forecast_horizon : int
        Number of periods to forecast from each origin
    time_format : str
        'year_week' or 'year_month'

    Returns
    -------
    List[Dict[str, str]]
        List of origin configurations
    """
    origins = []

    # Origin 0: Standard inference (train up to val_end)
    test_periods = generate_period_range(test_start, test_end, time_format)
    actual_forecast_end_0 = test_periods[min(forecast_horizon - 1, len(test_periods) - 1)]

    origins.append({
        'origin_idx': 0,
        'origin_period': test_start,
        'train_cutoff': val_end,
        'forecast_start': test_start,
        'forecast_end': actual_forecast_end_0,
    })

    # Subsequent origins: Roll forward one period at a time
    current_train_cutoff = test_start
    origin_idx = 1

    while True:
        forecast_start = increment_period(current_train_cutoff, 1, time_format)

        # Check if we still have enough test periods left
        if forecast_start > test_end:
            break

        # Calculate forecast end (limited by test_end and horizon)
        remaining = generate_period_range(forecast_start, test_end, time_format)
        if len(remaining) == 0:
            break

        forecast_end = remaining[min(forecast_horizon - 1, len(remaining) - 1)]

        origins.append({
            'origin_idx': origin_idx,
            'origin_period': forecast_start,
            'train_cutoff': current_train_cutoff,
            'forecast_start': forecast_start,
            'forecast_end': forecast_end,
        })

        current_train_cutoff = increment_period(current_train_cutoff, 1, time_format)
        origin_idx += 1

        # Safety check
        if origin_idx > 100:
            logger.warning("Exceeded 100 origins, stopping")
            break

    return origins


def run_single_origin_inference(
    config: DemandForecastConfig,
    origin_config: Dict[str, Any],
    output_dir: str,
    verbose: bool = False,
    source_df: Optional[pd.DataFrame] = None,
    cached_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, OriginResult]:
    """
    Run inference for a single backtest origin.

    This function:
    1. Temporarily modifies config to use the origin's train cutoff
    2. Runs the inference pipeline
    3. Returns forecasts with origin metadata

    Parameters
    ----------
    config : DemandForecastConfig
        Base configuration
    origin_config : Dict
        Origin configuration with train_cutoff, forecast_start, forecast_end
    output_dir : str
        Directory for temporary outputs
    verbose : bool
        Enable verbose logging
    source_df : pd.DataFrame, optional
        Pre-loaded source data. When supplied the function skips its own
        ``load_source_data`` call. Provided by
        :func:`run_rolling_origin_backtest` so the 1+ GB source CSV is
        parsed exactly once per backtest instead of once per origin.

    Returns
    -------
    Tuple[pd.DataFrame, OriginResult]
        - DataFrame with forecasts for this origin
        - OriginResult with metadata
    """
    from utils.inference import (
        detect_new_keys,
        create_dead_key_forecasts,
        retrain_all_models,
    )

    start_time = time.time()

    origin_idx = origin_config['origin_idx']
    origin_period = origin_config['origin_period']
    train_cutoff = origin_config['train_cutoff']
    forecast_start = origin_config['forecast_start']
    forecast_end = origin_config['forecast_end']

    logger.info(f"\n{'='*60}")
    logger.info(f"ORIGIN {origin_idx}: {origin_period}")
    logger.info(f"  Train cutoff: {train_cutoff}")
    logger.info(f"  Forecast: {forecast_start} to {forecast_end}")
    logger.info(f"{'='*60}")

    try:
        # Setup paths
        artifact_base = config.artifact_base_path
        seg_dir = os.path.join(artifact_base, "seg_output")
        feature_dir = os.path.join(artifact_base, "feature_output")
        model_dir = os.path.join(artifact_base, "model_artifacts")

        # Column names
        key_col = config.prediction_key_cols[0] if len(config.prediction_key_cols) == 1 else 'key'
        target_col = config.target_col
        date_col = config.timestamp_col

        # Load source data (supports both CSV and Parquet). Skip the
        # expensive re-parse if the outer backtest driver already handed
        # us a pre-loaded DataFrame (SW1).
        if source_df is None:
            from utils.agent_utilities import load_source_data
            source_df = load_source_data(config.input_data_path)
        else:
            # Work on a shallow copy so per-origin mutations don't leak
            # back to the outer backtest driver's shared instance.
            source_df = source_df.copy(deep=False)

        # Ensure 'key' column exists — all intermediate files use standardised 'key'
        if 'key' not in source_df.columns:
            if len(config.prediction_key_cols) == 1 and config.prediction_key_cols[0] in source_df.columns:
                source_df['key'] = source_df[config.prediction_key_cols[0]]
            else:
                source_df['key'] = source_df[config.prediction_key_cols].astype(str).agg('_'.join, axis=1)

        # Validate forward_forecast_exclude_col if configured
        exclude_col = getattr(config.design, 'forward_forecast_exclude_col', '')
        if exclude_col and exclude_col not in source_df.columns:
            raise ValueError(
                f"forward_forecast_exclude_col='{exclude_col}' is configured but "
                f"not found in source data. Available columns: "
                f"{sorted(source_df.columns.tolist())}. "
                f"Either add this column to your data or set "
                f"forward_forecast_exclude_col to '' (empty) in your config."
            )

        # Load original manifest
        manifest_path = os.path.join(feature_dir, "training_manifest.csv")
        manifest_df = pd.read_csv(manifest_path)

        # Load model specs
        model_specs_path = os.path.join(model_dir, "final_model_specs.json")
        with open(model_specs_path, 'r') as f:
            model_specs = json.load(f)

        # Detect keys for this origin's forecast period
        # Use time-format-aware lookback: 52 periods for weekly, 12 for monthly
        time_aware_defaults = config.get_time_aware_defaults()
        new_keys, existing_keys, dead_keys, _ = detect_new_keys(
            source_df=source_df,
            manifest_df=manifest_df,
            key_col=key_col,
            target_col=target_col,
            test_start=forecast_start,
            test_end=forecast_end,
            date_col=date_col,
            dead_key_lookback=time_aware_defaults['dead_key_lookback'],
        )

        logger.info(f"  Keys: {len(existing_keys)} existing, {len(new_keys)} new, {len(dead_keys)} dead")

        # For backtesting, we typically don't add new keys - use existing manifest
        # But we do retrain models with data up to train_cutoff

        # Determine which feature directory to use
        # If regenerate_features_on_backtest is enabled, regenerate features for this origin
        regenerate_features = getattr(config.design, 'regenerate_features_on_backtest', True)
        effective_feature_dir = feature_dir  # Default to original pre-computed features

        if regenerate_features:
            logger.info(f"  Regenerating features with data up to {train_cutoff}...")
            origin_feature_dir = os.path.join(output_dir, "features")
            os.makedirs(origin_feature_dir, exist_ok=True)

            success = regenerate_features_for_backtest_origin(
                config=config,
                manifest_df=manifest_df,
                train_cutoff=train_cutoff,
                output_dir=origin_feature_dir,
                key_col=key_col,
                date_col=date_col,
                target_col=target_col,
                # SW1: forward the already-loaded source_df so this origin
                # doesn't re-parse the 1+ GB CSV for a second time.
                source_df=source_df,
            )

            if success:
                effective_feature_dir = origin_feature_dir
                logger.info(f"  Using regenerated features from: {origin_feature_dir}")
            else:
                logger.warning(f"  Feature regeneration failed, falling back to pre-computed features")

        # Retrain models with data up to train_cutoff
        retrained_models, models_retrained = retrain_all_models(
            config=config,
            manifest_df=manifest_df,
            model_specs=model_specs,
            feature_dir=effective_feature_dir,  # Use regenerated features if available
            model_dir=output_dir,  # Use temp output dir
            target_col=target_col,
            key_col=key_col,
            date_col=date_col,
            train_cutoff=train_cutoff,  # KEY: Use origin's train cutoff
        )

        logger.info(f"  Retrained {models_retrained} models")

        # Load bias calibration if available
        apply_bias_calibration = config.design.apply_bias_calibration
        calibration_factors = None
        segment_calibrations = None

        if apply_bias_calibration:
            # Load basic bias calibration factors
            calibration_path = os.path.join(model_dir, "bias_calibration.json")
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r') as f:
                    calibration_factors = json.load(f)

            # STATE-OF-THE-ART: Load segment-aware calibration factors
            segment_cal_path = os.path.join(model_dir, "segment_calibrations.json")
            if os.path.exists(segment_cal_path):
                with open(segment_cal_path, 'r') as f:
                    segment_calibrations = json.load(f)

        # Generate forecasts for this origin
        forecasts_df, num_forecasts = _generate_origin_forecasts(
            config=config,
            retrained_models=retrained_models,
            manifest_df=manifest_df,
            model_specs=model_specs,
            feature_dir=effective_feature_dir,  # Use regenerated features if available
            origin_period=origin_period,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            train_cutoff=train_cutoff,
            apply_bias_calibration=apply_bias_calibration,
            calibration_factors=calibration_factors,
            segment_calibrations=segment_calibrations,
            cached_features=cached_features,
        )

        # Add dead key forecasts
        if dead_keys:
            dead_forecasts = create_dead_key_forecasts(
                dead_keys=dead_keys,
                test_start=forecast_start,
                test_end=forecast_end,
                key_col=key_col,
                date_col=date_col,
                target_col=target_col,
                forecast_horizon=config.forecast_horizon,
                source_df=source_df,
                time_format=config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week',
            )
            if not dead_forecasts.empty:
                # Add origin info to dead key forecasts
                dead_forecasts['origin_period'] = origin_period
                dead_forecasts['origin_idx'] = origin_idx
                forecasts_df = pd.concat([forecasts_df, dead_forecasts], ignore_index=True)
                num_forecasts += len(dead_forecasts)

        # ZERO OUT EXCLUDED KEYS (before reconciliation)
        exclude_col = getattr(config.design, 'forward_forecast_exclude_col', '')
        if exclude_col and len(forecasts_df) > 0:
            exclude_lookup = (
                source_df[[key_col, date_col, exclude_col]]
                .drop_duplicates(subset=[key_col, date_col], keep='last')
            )
            # Coerce join keys to a consistent str dtype so the merge doesn't
            # produce mixed int/str object columns (source_df typically has
            # year_week as int, DMH forecasts have it as str - merging them
            # otherwise silently corrupts downstream comparisons).
            exclude_lookup[date_col] = exclude_lookup[date_col].astype(str)
            exclude_lookup[key_col] = exclude_lookup[key_col].astype(str)
            forecasts_df[date_col] = forecasts_df[date_col].astype(str)
            forecasts_df[key_col] = forecasts_df[key_col].astype(str)
            forecasts_df = forecasts_df.merge(
                exclude_lookup, on=[key_col, date_col], how='left'
            )
            forecasts_df[exclude_col] = forecasts_df[exclude_col].fillna(0)
            exclude_mask = forecasts_df[exclude_col] == 1
            n_excluded = exclude_mask.sum()
            forecasts_df.loc[exclude_mask, 'predicted'] = 0.0
            forecasts_df = forecasts_df.drop(columns=[exclude_col])
            if n_excluded > 0:
                logger.info(f"  Origin {origin_idx}: zeroed {n_excluded} excluded (key, period) predictions")

        # YOY TREND ADJUSTMENT (before reconciliation)
        enable_yoy_trend = getattr(config.design, 'enable_yoy_trend_adjustment', False)
        if enable_yoy_trend and len(forecasts_df) > 0:
            try:
                from utils.reconciliation import apply_yoy_trend_adjustment

                tf = config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week'

                forecasts_df, _yoy_trend_diag = apply_yoy_trend_adjustment(
                    forecasts_df=forecasts_df,
                    source_df=source_df,
                    date_col=date_col,
                    target_col=target_col,
                    key_col=key_col,
                    test_start=forecast_start,
                    forecast_horizon=config.forecast_horizon,
                    time_format=tf,
                    min_history_periods=getattr(config.design, 'yoy_trend_min_history_periods', 104),
                    scaling_min=getattr(config.design, 'yoy_trend_scaling_min', 0.5),
                    scaling_max=getattr(config.design, 'yoy_trend_scaling_max', 2.0),
                )
                logger.info(
                    f"  Origin {origin_idx} YOY trend: "
                    f"{_yoy_trend_diag.get('keys_adjusted', 0)} keys adjusted"
                )
            except Exception as e:
                logger.warning(f"  Origin {origin_idx} YOY trend adjustment failed: {e}")

        # TOP-DOWN CATEGORY RECONCILIATION (per origin)
        enable_reconciliation = getattr(config.design, 'enable_top_down_reconciliation', False)
        if enable_reconciliation and len(forecasts_df) > 0:
            # Before reconciliation: coerce date_col to str on both sides so
            # the comparison inside `_coerce_cutoff` / `train_and_reconcile`
            # (which pulls cutoffs from source_df) doesn't hit
            # "'<' not supported between int and str" when concat of DMH +
            # dead-key forecasts produced a mixed-dtype object column.
            forecasts_df = _normalize_forecasts_schema(forecasts_df, date_col)
            try:
                from utils.reconciliation import train_and_reconcile

                tf = config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week'
                origin_forecast_periods = sorted(forecasts_df[date_col].unique().tolist())

                # Phase 3: Use configured reconciliation method and hierarchy column.
                # Resolve via hierarchy_resolution so we get the same answer as
                # feature engineering / inference / training.
                _recon_method = getattr(config.design, 'reconciliation_method', 'top_down')
                _hier_col = None
                try:
                    from utils.hierarchy_resolution import resolve_hierarchies
                    _seg_dir = os.path.join(config.artifact_base_path, 'seg_output')
                    _h = resolve_hierarchies(config=config, source_df=source_df, seg_dir=_seg_dir)
                    _hier_col = _h.primary_product_col
                    if _recon_method != 'top_down' and not _hier_col:
                        logger.warning(
                            "Origin %s: reconciliation_method=%s but no hierarchy "
                            "column resolved — falling back to top_down.",
                            origin_idx, _recon_method,
                        )
                except Exception as _hx:
                    logger.warning("Hierarchy resolution failed in backtesting: %s", _hx)
                    _hier_cols = getattr(config.design, 'hierarchy_cols', [])
                    if isinstance(_hier_cols, dict):
                        _hier_col = (_hier_cols.get('product') or [None])[-1] if _hier_cols.get('product') else None
                    elif _hier_cols:
                        _hier_col = _hier_cols[-1]

                forecasts_df, _recon_diag = train_and_reconcile(
                    forecasts_df=forecasts_df,
                    source_df=source_df,
                    target_col=target_col,
                    date_col=date_col,
                    key_col=key_col,
                    forecast_periods=origin_forecast_periods,
                    time_format=tf,
                    train_cutoff=train_cutoff,
                    ratio_min=getattr(config.design, 'reconciliation_ratio_min', 0.5),
                    ratio_max=getattr(config.design, 'reconciliation_ratio_max', 2.0),
                    trust_band=getattr(config.design, 'reconciliation_trust_band', 0.1),
                    changepoint_prior_scale=getattr(config.design, 'reconciliation_changepoint_prior', 0.05),
                    yoy_max_deviation=getattr(config.design, 'reconciliation_yoy_max_deviation', 0.15),
                    method=_recon_method,
                    hierarchy_col=_hier_col,
                )
                logger.info(f"  Origin {origin_idx} reconciliation: {_recon_diag.get('reconciliation_applied', False)}")
            except Exception as e:
                logger.warning(f"  Origin {origin_idx} reconciliation failed: {e}")

        # SPLY correction (final bias adjustment, after reconciliation)
        enable_sply = getattr(config.design, 'enable_sply_correction', True)
        if enable_sply and len(forecasts_df) > 0:
            try:
                from utils.sply_correction import apply_sply_correction
                tf = config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week'
                # forecasts_df always uses 'key' as the column name (from batch_recursive_forecast),
                # but source_df uses key_col (e.g. 'Model_Hierarchy'). Align both to 'key'.
                sply_source = source_df
                sply_key_col = 'key'
                if key_col != 'key' and 'key' not in forecasts_df.columns:
                    # forecasts_df doesn't have 'key' — rename key_col to 'key' in forecasts
                    forecasts_df = forecasts_df.rename(columns={key_col: 'key'})
                if key_col != 'key' and key_col in sply_source.columns and 'key' not in sply_source.columns:
                    sply_source = sply_source.copy()
                    sply_source['key'] = sply_source[key_col]
                forecasts_df, _sply_diag = apply_sply_correction(
                    forecasts_df=forecasts_df,
                    source_df=sply_source,
                    key_col=sply_key_col, date_col=date_col, target_col=target_col,
                    predicted_col='predicted', time_format=tf, default_alpha=0.7,
                )
                logger.info(f"  Origin {origin_idx} SPLY: {_sply_diag.get('n_rows_corrected', 0)} rows corrected")
            except Exception as e:
                logger.debug(f"  Origin {origin_idx} SPLY correction skipped: {e}")

        # Add origin index to all forecasts
        forecasts_df['origin_idx'] = origin_idx

        # Memory cleanup: free large objects no longer needed
        del source_df, retrained_models, model_specs
        if 'segments_df' in dir():
            del segments_df
        gc.collect()

        elapsed = time.time() - start_time

        logger.info(f"  Generated {num_forecasts} forecasts in {elapsed:.1f}s")

        # Final chokepoint: normalise dtypes so the downstream parquet writer
        # in `run_rolling_origin_backtest` doesn't get tripped up by mixed
        # int/str period columns (SnapshotTimePeriod, origin_period, year_week)
        # introduced by the concat of DMH/recursive + dead-key forecasts and
        # the various merges with source_df (where year_week is int in CSV).
        forecasts_df = _normalize_forecasts_schema(forecasts_df, date_col)

        return forecasts_df, OriginResult(
            origin_period=origin_period,
            train_cutoff=train_cutoff,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            num_forecasts=num_forecasts,
            elapsed_seconds=elapsed,
            success=True,
        )

    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        error_msg = f"Origin {origin_idx} failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)

        return pd.DataFrame(), OriginResult(
            origin_period=origin_period,
            train_cutoff=train_cutoff,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            num_forecasts=0,
            elapsed_seconds=elapsed,
            success=False,
            error_message=str(e),
        )


def _generate_origin_forecasts(
    config: DemandForecastConfig,
    retrained_models: Dict[str, Any],
    manifest_df: pd.DataFrame,
    model_specs: Dict[str, Any],
    feature_dir: str,
    origin_period: str,
    forecast_start: str,
    forecast_end: str,
    train_cutoff: str,
    apply_bias_calibration: bool = False,
    calibration_factors: Optional[Dict] = None,
    segment_calibrations: Optional[Dict] = None,
    cached_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Generate forecasts for a specific origin.

    Two paths are supported, selected by `config.design.use_direct_multi_horizon`:
      - False (default): recursive forecasting via `batch_recursive_forecast`
      - True: direct multi-horizon forecasting via the same
        `_generate_direct_multihorizon_forecasts` helper used by forward
        inference, so backtest accuracy matches production accuracy.
    """
    from utils.inference import batch_recursive_forecast

    key_col = config.prediction_key_cols[0] if len(config.prediction_key_cols) == 1 else 'key'
    date_col = config.timestamp_col
    target_col = config.target_col

    # ---------------------------------------------------------------
    # Direct Multi-Horizon branch: identical logic to inference PHASE 6.
    # This makes the backtest an honest dress rehearsal for production
    # when DMH is enabled in the config.
    #
    # Proper rolling-origin semantics: we pass the origin's `train_cutoff`
    # to the DMH helper as an explicit backtest_origin.  Inside the
    # helper, the combined training panel becomes train + val + test
    # (not just train + val), and for each horizon h the training rows
    # are filtered to `origin_week + h <= backtest_origin` so the
    # target label would have been observable at this origin.  Every
    # origin thus retrains with the data IT would actually have access
    # to, never more, never less.
    # ---------------------------------------------------------------
    use_dmh = bool(getattr(config.design, 'use_direct_multi_horizon', False))
    if use_dmh:
        from utils.inference import _generate_direct_multihorizon_forecasts

        try:
            backtest_origin_int = int(str(train_cutoff).replace("-", ""))
        except Exception as _e:
            raise ValueError(
                f"train_cutoff={train_cutoff!r} could not be parsed as YYYYWW int: {_e}"
            )

        seg_dir_path = os.path.join(config.artifact_base_path, "seg_output")
        forecasts_df, total = _generate_direct_multihorizon_forecasts(
            config=config,
            feature_dir=feature_dir,
            seg_dir=seg_dir_path,
            output_dir=os.path.join(config.artifact_base_path, "backtest_output",
                                     f"origin_{origin_period}_dmh"),
            key_col=key_col,
            date_col=date_col,
            target_col=target_col,
            backtest_origin=backtest_origin_int,
            cached_features=cached_features,
        )

        # Filter to the forecast window for this origin
        tf = getattr(config, 'time_format', 'year_week')
        if tf in ('auto', 'date'):
            tf = 'year_week'
        forecast_weeks = generate_period_range(forecast_start, forecast_end, time_format=tf)
        forecast_weeks = forecast_weeks[:config.forecast_horizon]
        forecasts_df[date_col] = forecasts_df[date_col].astype(str)
        forecast_weeks_str = [str(w) for w in forecast_weeks]
        forecasts_df = forecasts_df[forecasts_df[date_col].isin(forecast_weeks_str)].reset_index(drop=True)

        # Ensure the origin metadata matches what the backtest loop expects.
        origin_period_str = str(origin_period)
        forecasts_df['origin_period'] = origin_period_str
        forecasts_df['SnapshotTimePeriod'] = origin_period_str
        # actual is now populated by _generate_direct_multihorizon_forecasts
        # from the combined feature panel; if somehow missing, fall back.
        if 'actual' not in forecasts_df.columns:
            forecasts_df['actual'] = 0.0
        # Normalise dtypes so subsequent concat with create_dead_key_forecasts
        # (str) and merges with source_df (int year_week) don't produce mixed
        # int/str object columns that break pyarrow later.
        forecasts_df = _normalize_forecasts_schema(forecasts_df, date_col)
        return forecasts_df, len(forecasts_df)

    # ---------------------------------------------------------------
    # Legacy recursive path
    # ---------------------------------------------------------------

    # Load feature files (format-agnostic: prefers parquet, falls back to CSV).
    from utils.feature_io import read_features_intermediate
    train_df = read_features_intermediate(feature_dir, "train_features")
    val_df   = read_features_intermediate(feature_dir, "val_features")
    test_df  = read_features_intermediate(feature_dir, "test_features")

    # Combine all data
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    del train_df, val_df, test_df
    gc.collect()
    full_df = full_df.sort_values([key_col, date_col])

    # Get forecast periods for this origin
    tf = getattr(config, 'time_format', 'year_week')
    if tf in ('auto', 'date'):
        tf = 'year_week'
    forecast_weeks = generate_period_range(forecast_start, forecast_end, time_format=tf)
    forecast_weeks = forecast_weeks[:config.forecast_horizon]

    # Convert to numeric for comparison based on the actual date column dtype
    date_col_dtype = full_df[date_col].dtype
    if pd.api.types.is_float_dtype(date_col_dtype):
        train_cutoff_val = float(train_cutoff)
        forecast_weeks_val = [float(w) for w in forecast_weeks]
    elif pd.api.types.is_integer_dtype(date_col_dtype):
        train_cutoff_val = int(train_cutoff)
        forecast_weeks_val = [int(w) for w in forecast_weeks]
    else:
        train_cutoff_val = train_cutoff
        forecast_weeks_val = forecast_weeks

    snapshot_period = forecast_weeks_val[0] if forecast_weeks_val else forecast_start

    # Delegate to batch engine
    return batch_recursive_forecast(
        config=config,
        retrained_models=retrained_models,
        manifest_df=manifest_df,
        model_specs=model_specs,
        full_df=full_df,
        forecast_periods=forecast_weeks_val,
        origin_period=origin_period,
        snapshot_period=snapshot_period,
        history_cutoff=train_cutoff_val,
        apply_bias_calibration=apply_bias_calibration,
        calibration_factors=calibration_factors,
        segment_calibrations=segment_calibrations,
        max_workers=8,
    )


# =============================================================================
# MAIN BACKTEST FUNCTION
# =============================================================================

def run_rolling_origin_backtest(
    config: DemandForecastConfig,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    origin_callback: Optional[Callable[[int, Dict[str, Any], Dict[str, float]], None]] = None,
) -> BacktestResult:
    """
    Run rolling-origin (walk-forward) backtesting.

    This function:
    1. Generates backtest origins based on test period
    2. For each origin, retrains models and generates forecasts
    3. Combines all forecasts into a single output
    4. Computes accuracy metrics across origins and horizons

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object
    output_dir : str, optional
        Output directory (default: {artifact_base}/backtest_output)
    verbose : bool
        Enable verbose logging
    origin_callback : Callable, optional
        Callback function called after each origin completes.
        Signature: callback(origin_idx, origin_result_dict, metrics_dict)
        Used for email notifications or progress tracking.

    Returns
    -------
    BacktestResult
        Results including paths to all output files
    """
    start_time = time.time()

    # Setup output directory
    artifact_base = config.artifact_base_path
    backtest_dir = output_dir or os.path.join(artifact_base, "backtest_output")
    os.makedirs(backtest_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ROLLING-ORIGIN BACKTEST")
    logger.info("=" * 70)
    logger.info(f"Test Period: {config.test_start} to {config.test_end}")
    logger.info(f"Forecast Horizon: {config.forecast_horizon}")
    logger.info(f"Output Directory: {backtest_dir}")

    try:
        # Determine time format
        tf = getattr(config, 'time_format', 'year_week')
        if tf in ('auto', 'date'):
            tf = 'year_week'

        # Generate backtest origins
        origins = generate_backtest_origins(
            val_end=config.val_end,
            test_start=config.test_start,
            test_end=config.test_end,
            forecast_horizon=config.forecast_horizon,
            time_format=tf,
        )

        # Cap the origin list when the user wants a shorter backtest.
        #
        # Semantics: we keep the N origins whose forecast at the benchmark
        # evaluation lag (config.design.forecast_lag, typically 4) lands on
        # each of the LATEST N periods of test data. This mirrors how the
        # benchmark report is scored — per-period lag-4 WAPE over recent
        # weeks — so the smaller backtest stays directly comparable to the
        # full one.
        #
        # Mapping (for UK COOKING_AID, test_end=202614, forecast_lag=4,
        # max_backtest_origins=8):
        #   Target periods (lag-4 forecasts land here):
        #       202607, 202608, ..., 202614   ← latest 8 test weeks
        #   Required origins (train_cutoff = target - forecast_lag):
        #       202603, 202604, ..., 202610
        #
        # Naive `origins[-N:]` (which I had before) would instead pick
        # train_cutoffs 202606..202613 — and origins with train_cutoff >
        # test_end − forecast_lag cannot produce a lag-4 forecast inside
        # the test window at all, so they'd be scored on shorter horizons
        # and dilute the lag-4 comparison.
        _max_origins = getattr(config.design, 'max_backtest_origins', None)
        if _max_origins is not None and len(origins) > _max_origins:
            _eval_lag = int(getattr(config.design, 'forecast_lag', 4))
            _full_n = len(origins)

            # Only keep origins whose lag-_eval_lag forecast is within
            # test_end — i.e. train_cutoff + eval_lag <= test_end. This
            # guarantees every kept origin actually CAN produce the
            # benchmark-relevant forecast.
            def _lag_eval_period(origin: Dict[str, Any]) -> str:
                return increment_period(origin['train_cutoff'], _eval_lag, tf)

            valid_for_eval = [o for o in origins if _lag_eval_period(o) <= config.test_end]

            if len(valid_for_eval) == 0:
                # No origin has a lag-eval_lag forecast in-range — fall
                # back to naive latest-N on the full list so the backtest
                # still runs instead of aborting. This happens when
                # forecast_lag >= span(test_start..test_end), which is
                # misconfigured but recoverable.
                logger.warning(
                    f"No origin has a lag-{_eval_lag} forecast within test_end="
                    f"{config.test_end}; falling back to naive latest-{_max_origins} selection."
                )
                origins = origins[-_max_origins:]
            else:
                # Take the LAST _max_origins whose lag-eval_lag forecast
                # lands inside the test window — i.e. the most recent N
                # weeks of lag-eval_lag forecasts.
                origins = valid_for_eval[-_max_origins:]

            # Re-number origin_idx so downstream code that uses it as a
            # sequential counter stays consistent (0..N-1). The underlying
            # train_cutoff / forecast_start / forecast_end are unchanged.
            for _new_idx, _o in enumerate(origins):
                _o['origin_idx'] = _new_idx

            _first_cutoff = origins[0]['train_cutoff']
            _last_cutoff = origins[-1]['train_cutoff']
            _first_eval = increment_period(_first_cutoff, _eval_lag, tf)
            _last_eval = increment_period(_last_cutoff, _eval_lag, tf)
            logger.info(
                f"\nCapping backtest origins from {_full_n} to {len(origins)} "
                f"(config.design.max_backtest_origins={_max_origins}, "
                f"forecast_lag={_eval_lag}). Kept origins whose lag-{_eval_lag} "
                f"forecast covers the latest {len(origins)} test weeks: "
                f"train_cutoffs {_first_cutoff} → {_last_cutoff}, "
                f"lag-{_eval_lag} targets {_first_eval} → {_last_eval}."
            )

        logger.info(f"\nGenerated {len(origins)} backtest origins:")
        for origin in origins:
            logger.info(f"  Origin {origin['origin_idx']}: "
                       f"train_cutoff={origin['train_cutoff']}, "
                       f"forecast={origin['forecast_start']} to {origin['forecast_end']}")

        # =================================================================
        # SW1: Load the source DataFrame ONCE up front, not once per origin
        # (and previously, twice per origin: once in run_single_origin_
        # inference + once inside regenerate_features_for_backtest_origin).
        # For UK CONDIMENT that's a 1.2 GB CSV read — saves ~30 s × 2 reads
        # × N_origins every backtest.
        #
        # ME1: While we're at it, cast wide object columns to ``category``
        # and downcast integer columns. Source_df drops from ~4 GB to ~1 GB
        # for UK CONDIMENT, which cascades into every downstream copy.
        # =================================================================
        logger.info("\n[SW1] Loading source data once for all origins...")
        _t_load = time.time()
        from utils.agent_utilities import load_source_data
        from utils.period_utils import normalise_period_column
        source_df_shared = load_source_data(config.input_data_path)
        source_df_shared = normalise_period_column(source_df_shared, config.timestamp_col)
        # Force the period column to string dtype so every downstream
        # comparison against cfg.train_end / val_end / test_start / test_end
        # works regardless of how the CSV was parsed. When the column
        # contains only digits (e.g. 202614) pandas infers int64 on
        # read_csv; ``normalise_period_column`` leaves int columns
        # unchanged. Without this cast, comparisons against the
        # string-typed config values raise TypeError.
        if not pd.api.types.is_object_dtype(source_df_shared[config.timestamp_col]) \
                and not pd.api.types.is_string_dtype(source_df_shared[config.timestamp_col]):
            source_df_shared[config.timestamp_col] = source_df_shared[config.timestamp_col].astype(str)
        logger.info(
            "  Loaded source data in %.1f s: %d rows × %d cols",
            time.time() - _t_load, len(source_df_shared), len(source_df_shared.columns),
        )
        try:
            from utils.dataframe_memory import optimise_dataframe_memory
            # Protect the schema columns — downstream code checks dtypes on
            # these (timestamps as int64, target as numeric, etc.).
            protect = set(config.prediction_key_cols) | {
                config.timestamp_col, config.target_col,
            }
            source_df_shared, _mem_diag = optimise_dataframe_memory(
                source_df_shared,
                protect_cols=protect,
                log_prefix="backtest source_df",
            )
            if _mem_diag.get("saved_bytes", 0):
                logger.info(
                    "  [ME1] saved %.0f MB (%.1f%%) via category / int downcast",
                    _mem_diag["saved_bytes"] / 1024 / 1024,
                    _mem_diag.get("saved_pct", 0.0),
                )
        except Exception as _mem_exc:
            logger.warning("ME1 memory optimisation skipped: %s", _mem_exc)

        # Run inference for each origin. Save each origin's forecasts as
        # separate temp Parquet files (SW2 — was CSV before), then combine
        # at the end. Parquet is typically 3-10× faster to read/write and
        # significantly smaller on disk, which matters for the combine step
        # (every per-origin file is round-tripped through disk).
        origin_results = {}
        forecasts_path = os.path.join(backtest_dir, "backtest_forecasts.csv")
        origin_forecast_files = []  # per-origin temp paths (now Parquet)
        total_forecasts = 0

        # SW2: prefer Parquet for intermediates if pyarrow is installed,
        # otherwise gracefully fall back to CSV. The combine step auto-
        # detects format from the extension.
        try:
            import pyarrow  # noqa: F401
            _intermediate_ext = ".parquet"
        except ImportError:
            logger.warning("pyarrow not installed — falling back to CSV per-origin intermediates")
            _intermediate_ext = ".csv"

        # Preload + augment feature frames once per backtest when DMH is
        # enabled and the cache flag is on.  Each origin otherwise reloads
        # and re-augments the train/val/test CSVs (~60-120s wasted per
        # origin on UK/TH category CSVs).  We do this BEFORE the origin
        # loop and pass the in-memory frames through `cached_features`.
        cached_features: Optional[Dict[str, pd.DataFrame]] = None
        _use_dmh = bool(getattr(config.design, 'use_direct_multi_horizon', False))
        _enable_cache = bool(getattr(config.design, 'dmh_cache_features_across_origins', True))
        if _use_dmh and _enable_cache:
            try:
                artifact_base = config.artifact_base_path
                feat_dir = os.path.join(artifact_base, "feature_output")
                # Format-agnostic existence + read via the helpers.  This
                # keeps the cache logic working whether feature engineering
                # wrote parquet (default) or CSV (fallback).
                from utils.feature_io import (
                    features_intermediate_exists, read_features_intermediate,
                )
                if all(
                    features_intermediate_exists(feat_dir, n)
                    for n in ("train_features", "val_features", "test_features")
                ):
                    logger.info(
                        "[DMH cache] preloading feature files once for all %d origins...",
                        len(origins),
                    )
                    _t_cache = time.time()
                    _tr = read_features_intermediate(feat_dir, "train_features", low_memory=False)
                    _va = read_features_intermediate(feat_dir, "val_features",   low_memory=False)
                    _te = read_features_intermediate(feat_dir, "test_features",  low_memory=False)

                    # Run augmenters once too (they're origin-invariant: they
                    # operate on the whole (key, year_week) panel per frame).
                    _bt_aug = bool(getattr(config.design, 'dmh_backtest_enable_augmenters', True))
                    if _bt_aug:
                        from utils.direct_multihorizon import augment_features
                        _key_col = config.prediction_key_cols[0] if len(config.prediction_key_cols) == 1 else 'key'
                        _target_col = config.target_col

                        # Resolve apg_col
                        _apg = str(getattr(config.design, 'dmh_apg_col', '') or '').strip()
                        if not _apg:
                            _imp = getattr(config.design, 'imputation_level', None) or []
                            if isinstance(_imp, (list, tuple)) and _imp:
                                _apg = str(_imp[0])
                        if not _apg:
                            _apg = 'APG_code'

                        # Build forward candidates from config feature lists
                        def _coll(obj):
                            out = []
                            if obj is None: return out
                            if isinstance(obj, list): return [str(c) for c in obj]
                            if isinstance(obj, dict):
                                for k in ('numeric','categorical'):
                                    v = obj.get(k) or []
                                    if isinstance(v, list): out.extend(str(c) for c in v)
                                return out
                            for a in ('numeric','categorical'):
                                v = getattr(obj, a, None) or []
                                if isinstance(v, list): out.extend(str(c) for c in v)
                            return out
                        _fwd = []
                        for a in ('price_features','promo_features','holiday_features','weather_features'):
                            _fwd.extend(_coll(getattr(config, a, None)))
                        _fwd.extend(['week_of_year','week_of_year_sin','week_of_year_cos',
                                     'month','month_sin','month_cos','quarter',
                                     'is_holiday','holiday_flag','weeks_to_nearest_holiday',
                                     'season','holiday','holidays'])
                        _seen = set()
                        _fwd = [c for c in _fwd if not (c in _seen or _seen.add(c))]
                        _aug_kw = dict(
                            key_col=_key_col, target_col=_target_col, apg_col=_apg,
                            enable_trajectory=bool(getattr(config.design, 'dmh_enable_trajectory_features', True)),
                            enable_apg=bool(getattr(config.design, 'dmh_enable_apg_features', True)),
                            enable_forward=bool(getattr(config.design, 'dmh_enable_forward_features', True)),
                            forward_candidate_cols=_fwd,
                        )
                        _tr = augment_features(_tr, horizon=4, **_aug_kw)
                        _va = augment_features(_va, horizon=4, **_aug_kw)
                        _te = augment_features(_te, horizon=4, **_aug_kw)

                    cached_features = {
                        'train': _tr,
                        'val': _va,
                        'test': _te,
                        '_augmented': _bt_aug,
                    }
                    logger.info(
                        "[DMH cache] ready in %.1fs: train=%d rows, val=%d rows, test=%d rows (augmented=%s)",
                        time.time() - _t_cache, len(_tr), len(_va), len(_te), _bt_aug,
                    )
            except Exception as _cache_exc:
                logger.warning(
                    "[DMH cache] preload failed, falling back to per-origin loading: %s",
                    _cache_exc,
                )
                cached_features = None

        for origin_config in origins:
            origin_idx = origin_config['origin_idx']

            # Create temp directory for this origin
            origin_temp_dir = os.path.join(backtest_dir, f"origin_{origin_idx}_temp")
            os.makedirs(origin_temp_dir, exist_ok=True)

            # Run inference for this origin — pass the shared source_df and
            # the cached feature frames so this origin skips the expensive
            # CSV reload and augmenter passes.
            forecasts_df, origin_result = run_single_origin_inference(
                config=config,
                origin_config=origin_config,
                output_dir=origin_temp_dir,
                verbose=verbose,
                source_df=source_df_shared,
                cached_features=cached_features,
            )

            if origin_result.success and len(forecasts_df) > 0:
                # SW2: write per-origin forecasts as Parquet (fast + small).
                origin_path = os.path.join(
                    backtest_dir,
                    f"_origin_{origin_idx}_forecasts{_intermediate_ext}",
                )
                if _intermediate_ext == ".parquet":
                    forecasts_df.to_parquet(origin_path, index=False, engine="pyarrow")
                else:
                    forecasts_df.to_csv(origin_path, index=False)
                origin_forecast_files.append(origin_path)
                total_forecasts += len(forecasts_df)

            origin_result_dict = {
                'origin_period': origin_result.origin_period,
                'train_cutoff': origin_result.train_cutoff,
                'forecast_start': origin_result.forecast_start,
                'forecast_end': origin_result.forecast_end,
                'num_forecasts': origin_result.num_forecasts,
                'elapsed_seconds': origin_result.elapsed_seconds,
                'success': origin_result.success,
                'error_message': origin_result.error_message,
            }
            origin_results[str(origin_idx)] = origin_result_dict

            # Compute metrics for this origin and call callback if provided
            if origin_callback and origin_result.success and len(forecasts_df) > 0:
                try:
                    # Compute quick metrics for this origin
                    actual = forecasts_df['actual'].values
                    predicted = forecasts_df['predicted'].values
                    actual_sum = actual.sum()

                    if actual_sum > 0:
                        wape = np.abs(predicted - actual).sum() / actual_sum
                        bias_pct = (predicted.sum() - actual_sum) / actual_sum * 100
                    else:
                        wape = 0.0
                        bias_pct = 0.0

                    mae = np.abs(predicted - actual).mean()

                    origin_metrics = {
                        'wape': float(wape),
                        'bias_pct': float(bias_pct),
                        'mae': float(mae),
                        'n_forecasts': len(forecasts_df),
                    }

                    # Call the callback (e.g., send email notification)
                    origin_callback(origin_idx + 1, origin_result_dict, origin_metrics)
                except Exception as e:
                    logger.warning(f"Origin callback failed: {e}")

            # Cleanup temp directory and free memory
            try:
                shutil.rmtree(origin_temp_dir)
            except Exception:
                pass

            # Explicit memory cleanup between origins to prevent OOM
            del forecasts_df, origin_result
            gc.collect()
            logger.debug(f"  Memory cleanup after origin {origin_idx}")

        # Release the shared source_df now that all origins are done
        try:
            del source_df_shared
        except Exception:
            pass
        gc.collect()

        # Combine all origin forecast files into one (single write, no append).
        # Auto-detect format per file (SW2 writes Parquet when pyarrow is
        # available, otherwise CSV — mixed-extension directories are fine).
        if origin_forecast_files:
            chunks = []
            for f in origin_forecast_files:
                if f.endswith(".parquet"):
                    chunks.append(pd.read_parquet(f))
                else:
                    chunks.append(pd.read_csv(f))
            combined_forecasts = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()

            # -------------------------------------------------------------
            # MOQ (Minimum Order Quantity) post-processing.
            # Adds a ``prediction_post_moq`` column so downstream DIQ can
            # use MOQ-rounded forecasts directly. Detection uses
            # training-period history only (up to train_end) which is
            # always strictly before every backtest origin, so this stays
            # leakage-free across origins. See utils/moq_postprocessing.py.
            # -------------------------------------------------------------
            if getattr(config.design, 'apply_moq_postprocessing', True):
                try:
                    from utils.moq_postprocessing import apply_moq_postprocessing
                    combined_forecasts = apply_moq_postprocessing(
                        combined_forecasts,
                        config,
                        # source_df_shared is already released above; the
                        # utility loads config.input_data_path on demand.
                        history_df=None,
                        forecast_col='predicted',
                        output_col='prediction_post_moq',
                    )
                except Exception as moq_err:
                    logger.warning(
                        f"MOQ post-processing failed on backtest output: "
                        f"{moq_err}. Saving forecasts without prediction_post_moq."
                    )
            else:
                logger.info("MOQ post-processing is disabled in config")

            # The canonical output remains CSV for downstream compatibility
            # (compare_benchmark.py and similar utilities expect CSV).
            combined_forecasts.to_csv(forecasts_path, index=False)

            # Cleanup temp origin files (Parquet or CSV)
            for f in origin_forecast_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
        else:
            combined_forecasts = pd.DataFrame()

        logger.info(f"\nCombined {total_forecasts} forecasts from {len(origins)} origins")
        logger.info(f"Saved forecasts: {forecasts_path}")

        # Compute and save metrics
        metrics_path = os.path.join(backtest_dir, "backtest_metrics.csv")
        if len(combined_forecasts) > 0:
            metrics_df = compute_backtest_metrics(combined_forecasts, config)
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved metrics: {metrics_path}")

        # Create summary
        elapsed_time = time.time() - start_time

        summary = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_time,
            'total_origins': len(origins),
            'total_forecasts': total_forecasts,
            'forecast_horizon': config.forecast_horizon,
            'test_period': {
                'start': config.test_start,
                'end': config.test_end,
            },
            'origins': [o['origin_period'] for o in origins],
            'origin_details': origin_results,
            'output_files': {
                'forecasts': forecasts_path,
                'metrics': metrics_path,
            },
        }

        summary_path = os.path.join(backtest_dir, "backtest_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_path}")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total Origins: {len(origins)}")
        logger.info(f"Total Forecasts: {total_forecasts}")
        logger.info(f"Elapsed Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

        # Create result object first
        result = BacktestResult(
            success=True,
            forecasts_path=forecasts_path,
            summary_path=summary_path,
            metrics_path=metrics_path,
            total_origins=len(origins),
            total_forecasts=total_forecasts,
            forecast_horizon=config.forecast_horizon,
            origins=[o['origin_period'] for o in origins],
            elapsed_seconds=elapsed_time,
            origin_details=origin_results,
        )

        # Generate documentation insights guide (if enabled)
        enable_insights = getattr(config.design, 'enable_insights_reports', False) if hasattr(config, 'design') else False
        if enable_insights:
            try:
                doc_path = generate_backtest_documentation(
                    backtest_result=result,
                    forecasts_df=combined_forecasts,
                    config=config,
                    output_dir=backtest_dir,
                )
                logger.info(f"Generated insights guide: {doc_path}")
            except Exception as doc_err:
                logger.warning(f"Documentation generation failed: {doc_err}")
        else:
            logger.info("SKIPPING backtest insights guide (enable_insights_reports=False)")

        return BacktestResult(
            success=True,
            forecasts_path=forecasts_path,
            summary_path=summary_path,
            metrics_path=metrics_path,
            total_origins=len(origins),
            total_forecasts=total_forecasts,
            forecast_horizon=config.forecast_horizon,
            origins=[o['origin_period'] for o in origins],
            elapsed_seconds=elapsed_time,
            origin_details=origin_results,
        )

    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        error_msg = f"Backtest failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)

        return BacktestResult(
            success=False,
            error_message=error_msg,
            elapsed_seconds=elapsed_time,
        )


# =============================================================================
# DOCUMENTATION GENERATION
# =============================================================================

def generate_backtest_documentation(
    backtest_result: BacktestResult,
    forecasts_df: pd.DataFrame,
    config: DemandForecastConfig,
    output_dir: Optional[str] = None,
) -> str:
    """
    Generate comprehensive markdown documentation explaining backtesting insights.

    This function creates a detailed guide for data scientists that explains:
    - What the rolling-origin backtesting did
    - Data-driven insights observed across origins
    - Forecast accuracy degradation by horizon
    - Model stability analysis
    - Key learnings and recommendations

    Parameters
    ----------
    backtest_result : BacktestResult
        Result from run_rolling_origin_backtest()
    forecasts_df : pd.DataFrame
        Combined forecasts from all origins
    config : DemandForecastConfig
        Configuration object
    output_dir : str, optional
        Override output directory (default: backtest_output/)

    Returns
    -------
    str
        Path to the generated documentation file
    """
    artifact_base = config.artifact_base_path
    backtest_dir = output_dir or os.path.join(artifact_base, "backtest_output")
    os.makedirs(backtest_dir, exist_ok=True)

    doc_path = os.path.join(backtest_dir, "BACKTESTING_INSIGHTS_GUIDE.md")

    # Compute metrics for documentation
    metrics_summary = {}
    horizon_metrics = {}
    origin_metrics = {}
    model_metrics = {}

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
        metrics_summary['overall_rmse'] = float(np.sqrt(((predicted - actual) ** 2).mean()))

        # By horizon (lag)
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

        # By origin
        for origin in sorted(forecasts_df['origin_period'].unique()):
            origin_df = forecasts_df[forecasts_df['origin_period'] == origin]
            orig_actual = origin_df['actual'].values
            orig_pred = origin_df['predicted'].values
            orig_actual_sum = orig_actual.sum()

            if orig_actual_sum > 0:
                wape = float(np.abs(orig_pred - orig_actual).sum() / orig_actual_sum)
                bias = float((orig_pred.sum() - orig_actual_sum) / orig_actual_sum * 100)
            else:
                wape = 0.0
                bias = 0.0

            origin_metrics[str(origin)] = {
                'wape': wape,
                'bias_pct': bias,
                'mae': float(np.abs(orig_pred - orig_actual).mean()),
                'n_forecasts': len(origin_df),
            }

        # By model level
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

    # Generate documentation
    doc_content = f"""# Backtesting Insights Guide

## Executive Summary

This document provides a comprehensive analysis of the rolling-origin backtesting results, explaining the methodology, key findings, and actionable insights for demand forecasting.

| Metric | Value |
|--------|-------|
| **Total Origins** | {backtest_result.total_origins} |
| **Total Forecasts** | {backtest_result.total_forecasts:,} |
| **Forecast Horizon** | {backtest_result.forecast_horizon} periods |
| **Test Period** | {config.test_start} to {config.test_end} |
| **Execution Time** | {backtest_result.elapsed_seconds:.1f}s ({backtest_result.elapsed_seconds/60:.1f} min) |

---

## 1. What is Rolling-Origin Backtesting?

Rolling-origin (or walk-forward) backtesting is a rigorous methodology for evaluating forecast accuracy that:

1. **Mimics Production Forecasting**: Each origin simulates making forecasts at a specific point in time using only data available at that moment
2. **Avoids Data Leakage**: Training data is strictly limited to periods before the forecast origin
3. **Tests Model Stability**: Multiple origins reveal how consistently the model performs over time
4. **Evaluates Horizon Degradation**: Shows how accuracy changes as forecast horizon increases

### Origin Structure

"""

    # Add origin details
    if backtest_result.origin_details:
        doc_content += "| Origin | Train Cutoff | Forecast Start | Forecast End | Forecasts | Status |\n"
        doc_content += "|--------|-------------|----------------|--------------|-----------|--------|\n"
        for idx, (origin_idx, details) in enumerate(sorted(backtest_result.origin_details.items(), key=lambda x: int(x[0]))):
            status = "✅" if details.get('success', False) else "❌"
            doc_content += f"| {origin_idx} | {details.get('train_cutoff', 'N/A')} | {details.get('forecast_start', 'N/A')} | {details.get('forecast_end', 'N/A')} | {details.get('num_forecasts', 0):,} | {status} |\n"

    doc_content += f"""

---

## 2. Overall Performance Summary

### Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **WAPE** | {metrics_summary.get('overall_wape', 0):.2%} | {'Excellent (<10%)' if metrics_summary.get('overall_wape', 1) < 0.10 else 'Good (10-20%)' if metrics_summary.get('overall_wape', 1) < 0.20 else 'Moderate (20-30%)' if metrics_summary.get('overall_wape', 1) < 0.30 else 'Needs Improvement (>30%)'} |
| **Bias** | {metrics_summary.get('overall_bias_pct', 0):+.1f}% | {'Over-forecasting' if metrics_summary.get('overall_bias_pct', 0) > 2 else 'Under-forecasting' if metrics_summary.get('overall_bias_pct', 0) < -2 else 'Well-calibrated'} |
| **MAE** | {metrics_summary.get('overall_mae', 0):.2f} | Average absolute error per forecast |
| **RMSE** | {metrics_summary.get('overall_rmse', 0):.2f} | Root mean squared error (penalizes large errors) |

### What These Metrics Mean

- **WAPE (Weighted Absolute Percentage Error)**: Primary accuracy metric. Lower is better. Weighted by actual demand, so high-volume items contribute more.
- **Bias**: Systematic over/under-forecasting tendency. Positive = over-forecast, negative = under-forecast.
- **MAE**: Average magnitude of forecast errors in original units.
- **RMSE**: Similar to MAE but gives more weight to large errors.

---

## 3. Forecast Horizon Degradation Analysis

One of the most important insights from backtesting is understanding how forecast accuracy degrades as we predict further into the future.

### Accuracy by Horizon Step

| Horizon | WAPE | Bias | MAE | Forecasts | Degradation |
|---------|------|------|-----|-----------|-------------|
"""

    # Add horizon metrics
    base_wape = horizon_metrics.get(1, {}).get('wape', 0) if horizon_metrics else 0
    for horizon, hm in sorted(horizon_metrics.items()):
        degradation = ((hm['wape'] - base_wape) / base_wape * 100) if base_wape > 0 else 0
        deg_indicator = "—" if horizon == 1 else f"+{degradation:.0f}%" if degradation > 0 else f"{degradation:.0f}%"
        doc_content += f"| Step {horizon} | {hm['wape']:.2%} | {hm['bias_pct']:+.1f}% | {hm['mae']:.2f} | {hm['n_forecasts']:,} | {deg_indicator} |\n"

    doc_content += """
### Key Observations

"""

    # Add horizon-specific insights
    if len(horizon_metrics) >= 2:
        first_horizon = min(horizon_metrics.keys())
        last_horizon = max(horizon_metrics.keys())
        first_wape = horizon_metrics[first_horizon]['wape']
        last_wape = horizon_metrics[last_horizon]['wape']
        degradation_total = ((last_wape - first_wape) / first_wape * 100) if first_wape > 0 else 0

        doc_content += f"""1. **Horizon Degradation Rate**: WAPE increases by approximately {degradation_total:.0f}% from step 1 to step {last_horizon}
2. **Step 1 Accuracy**: {first_wape:.2%} WAPE - this represents our best-case accuracy
3. **Final Step Accuracy**: {last_wape:.2%} WAPE - this represents accuracy at maximum horizon
"""

        if degradation_total > 50:
            doc_content += f"""4. **Recommendation**: Consider reducing forecast horizon or using different models for longer horizons
"""
        elif degradation_total < 20:
            doc_content += f"""4. **Observation**: Model maintains good accuracy across the full horizon - indicates stable demand patterns
"""

    doc_content += f"""

---

## 4. Model Stability Across Origins

Analyzing performance across different forecast origins reveals model stability over time.

### Performance by Origin

| Origin Period | WAPE | Bias | Forecasts |
|---------------|------|------|-----------|
"""

    for origin, om in sorted(origin_metrics.items()):
        doc_content += f"| {origin} | {om['wape']:.2%} | {om['bias_pct']:+.1f}% | {om['n_forecasts']:,} |\n"

    # Calculate stability metrics
    if len(origin_metrics) >= 2:
        origin_wapes = [om['wape'] for om in origin_metrics.values()]
        wape_std = np.std(origin_wapes)
        wape_mean = np.mean(origin_wapes)
        cv_origins = wape_std / wape_mean if wape_mean > 0 else 0

        doc_content += f"""
### Stability Analysis

| Stability Metric | Value | Interpretation |
|------------------|-------|----------------|
| **WAPE Std Dev** | {wape_std:.4f} | Variation in accuracy across origins |
| **WAPE CV** | {cv_origins:.2%} | {'Highly stable' if cv_origins < 0.10 else 'Stable' if cv_origins < 0.20 else 'Moderate variability' if cv_origins < 0.30 else 'High variability'} |
| **Best Origin** | {min(origin_metrics.items(), key=lambda x: x[1]['wape'])[0]} | Lowest WAPE |
| **Worst Origin** | {max(origin_metrics.items(), key=lambda x: x[1]['wape'])[0]} | Highest WAPE |

"""

    doc_content += f"""

---

## 5. Model-Level Performance

Different model levels (segments) may have varying forecast accuracy.

### Performance by Model Level

| Model Level | WAPE | Bias | Forecasts |
|-------------|------|------|-----------|
"""

    for ml, mm in sorted(model_metrics.items(), key=lambda x: x[1]['wape']):
        doc_content += f"| {ml} | {mm['wape']:.2%} | {mm['bias_pct']:+.1f}% | {mm['n_forecasts']:,} |\n"

    # Identify underperformers
    if model_metrics:
        avg_wape = np.mean([mm['wape'] for mm in model_metrics.values()])
        underperformers = [(ml, mm) for ml, mm in model_metrics.items() if mm['wape'] > avg_wape * 1.3]

        if underperformers:
            doc_content += f"""
### Underperforming Segments

The following model levels have WAPE > 30% above average ({avg_wape:.2%}):

"""
            for ml, mm in underperformers:
                doc_content += f"- **{ml}**: {mm['wape']:.2%} WAPE ({((mm['wape'] - avg_wape) / avg_wape * 100):.0f}% above average)\n"

    doc_content += f"""

---

## 6. Key Learnings & Recommendations

### What We Learned

"""

    # Generate data-driven recommendations
    recommendations = []

    # Check horizon degradation
    if len(horizon_metrics) >= 2:
        first_wape = horizon_metrics[min(horizon_metrics.keys())]['wape']
        last_wape = horizon_metrics[max(horizon_metrics.keys())]['wape']
        if last_wape > first_wape * 1.5:
            recommendations.append("**Horizon Impact**: Significant accuracy degradation at longer horizons. Consider ensemble approaches with different models for short vs. long horizons.")
        else:
            recommendations.append("**Horizon Stability**: Model maintains reasonable accuracy across the forecast horizon. Current approach is well-suited for the configured horizon.")

    # Check bias
    if abs(metrics_summary.get('overall_bias_pct', 0)) > 5:
        bias_dir = "over" if metrics_summary.get('overall_bias_pct', 0) > 0 else "under"
        recommendations.append(f"**Bias Correction Needed**: Model tends to {bias_dir}-forecast by {abs(metrics_summary.get('overall_bias_pct', 0)):.1f}%. Segment-aware bias calibration is recommended.")
    else:
        recommendations.append("**Well-Calibrated**: Model shows minimal systematic bias. Current calibration approach is effective.")

    # Check stability
    if len(origin_metrics) >= 2:
        origin_wapes = [om['wape'] for om in origin_metrics.values()]
        if np.std(origin_wapes) / np.mean(origin_wapes) > 0.25:
            recommendations.append("**Temporal Instability**: Performance varies significantly across origins. Consider adaptive learning or more recent data weighting.")
        else:
            recommendations.append("**Temporal Stability**: Model performs consistently across different forecast origins. Training approach is robust.")

    for i, rec in enumerate(recommendations, 1):
        doc_content += f"{i}. {rec}\n"

    doc_content += f"""

### Actionable Recommendations

1. **For Production Deployment**:
   - Use the overall WAPE ({metrics_summary.get('overall_wape', 0):.2%}) as the expected accuracy benchmark
   - Set safety stock levels based on horizon-specific error rates
   - Monitor for drift by comparing production accuracy to these backtest results

2. **For Model Improvement**:
   - Focus on underperforming segments identified above
   - Consider feature engineering for segments with high bias
   - Evaluate alternative models for segments with WAPE > 30%

3. **For Forecast Consumers**:
   - Communicate that accuracy degrades with forecast horizon
   - Provide confidence intervals based on RMSE at each horizon
   - Re-forecast frequently to leverage more recent data

---

## 7. Technical Details

### Configuration Used

| Setting | Value |
|---------|-------|
| Forecast Horizon | {config.forecast_horizon} periods |
| Test Start | {config.test_start} |
| Test End | {config.test_end} |
| Validation End | {config.val_end} |
| Feature Regeneration | {'Enabled' if getattr(config.design, 'regenerate_features_on_backtest', True) else 'Disabled'} |
| Bias Calibration | {'Enabled' if config.design.apply_bias_calibration else 'Disabled'} |

### Output Files

| File | Description |
|------|-------------|
| `backtest_forecasts.csv` | All forecasts from all origins |
| `backtest_metrics.csv` | Computed metrics at various aggregation levels |
| `backtest_summary.json` | Summary statistics and metadata |

---

*Generated by HarmonIQ Demand-IQ Backtesting Pipeline*
*Documentation helps data scientists understand forecast reliability and make informed decisions*
"""

    # Write documentation
    with open(doc_path, 'w') as f:
        f.write(doc_content)

    logger.info(f"Generated backtesting documentation: {doc_path}")

    return doc_path


def compute_backtest_metrics(
    forecasts_df: pd.DataFrame,
    config: DemandForecastConfig,
) -> pd.DataFrame:
    """
    Compute accuracy metrics from backtest forecasts.

    Computes metrics by:
    - Overall
    - By origin
    - By forecast horizon (lag)
    - By model level

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Combined forecasts from all origins
    config : DemandForecastConfig
        Configuration object

    Returns
    -------
    pd.DataFrame
        Metrics at various aggregation levels
    """
    if len(forecasts_df) == 0:
        return pd.DataFrame()

    metrics_rows = []

    # Helper to compute WAPE
    def compute_wape(df):
        actual_sum = df['actual'].sum()
        if actual_sum == 0:
            return np.nan
        abs_error = (df['predicted'] - df['actual']).abs().sum()
        return abs_error / actual_sum

    # Helper to compute other metrics
    def compute_metrics(df, group_name, group_value):
        if len(df) == 0:
            return None

        actual = df['actual'].values
        predicted = df['predicted'].values

        # Avoid division by zero
        actual_sum = actual.sum()
        if actual_sum == 0:
            wape = np.nan
        else:
            wape = np.abs(predicted - actual).sum() / actual_sum

        mae = np.abs(predicted - actual).mean()
        mse = ((predicted - actual) ** 2).mean()
        rmse = np.sqrt(mse)

        # Bias (positive = over-forecast)
        bias = (predicted - actual).sum() / max(len(df), 1)
        bias_pct = (predicted.sum() - actual_sum) / max(actual_sum, 1) * 100

        return {
            'group_type': group_name,
            'group_value': str(group_value),
            'n_forecasts': len(df),
            'actual_sum': actual_sum,
            'predicted_sum': predicted.sum(),
            'wape': wape,
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'bias_pct': bias_pct,
        }

    # Overall metrics
    overall = compute_metrics(forecasts_df, 'overall', 'all')
    if overall:
        metrics_rows.append(overall)

    # By origin
    for origin in forecasts_df['origin_period'].unique():
        origin_df = forecasts_df[forecasts_df['origin_period'] == origin]
        m = compute_metrics(origin_df, 'origin', origin)
        if m:
            metrics_rows.append(m)

    # By forecast horizon (lag)
    for lag in sorted(forecasts_df['lag'].unique()):
        lag_df = forecasts_df[forecasts_df['lag'] == lag]
        m = compute_metrics(lag_df, 'horizon', int(lag))
        if m:
            metrics_rows.append(m)

    # By model level
    for model_level in forecasts_df['model_level'].unique():
        level_df = forecasts_df[forecasts_df['model_level'] == model_level]
        m = compute_metrics(level_df, 'model_level', model_level)
        if m:
            metrics_rows.append(m)

    return pd.DataFrame(metrics_rows)
