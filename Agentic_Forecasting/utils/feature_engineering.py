# utils/feature_engineering.py
"""
State-of-the-Art Feature Engineering Utilities
==============================================

This module provides COMPREHENSIVE, WORLD-CLASS feature engineering functions
for demand forecasting. Agents MUST use these pre-built functions instead of
writing code from scratch - this dramatically reduces context usage.

Categories:
    1. Lag Features (standard, combinations, momentum)
    2. Rolling Features (standard, expanding, EWM)
    3. Calendar & Cyclical Features (Fourier, sin/cos)
    4. Trend & Decomposition Features (STL, detrending)
    5. Intermittency Features (sparse demand modeling)
    6. Cross-Sectional Features (relative to segment/global)
    7. Categorical Encoding (target, ordinal, one-hot, hash)
    8. Feature Selection (correlation, SHAP, variance)
    9. Data Quality & Validation
    10. High-Level Orchestration (complete pipeline in one call)

Usage:
    from utils.feature_engineering import (
        # HIGH-LEVEL - USE THESE FOR COMPLETE FEATURE ENGINEERING:
        create_complete_feature_set,
        run_feature_pipeline,

        # Individual utilities if needed:
        create_lag_features, create_momentum_features,
        create_rolling_features, create_ewm_features,
        create_fourier_features, create_trend_features,
        create_cross_sectional_features,
        auto_select_features,
        ...
    )

    # ONE FUNCTION FOR EVERYTHING:
    result = run_feature_pipeline(
        df=raw_df,
        key_cols=['key'],
        date_col='year_week',
        target_col='sales',
        segment_col='segment_id',
        ...
    )

NOTE: If you add a new feature type, add the corresponding function here.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# DEAD-FEATURE PRUNING
# =============================================================================
# Drop columns that carry zero predictive signal across the entire training
# panel BEFORE writing parquet artifacts.  Run once during feature engineering
# so that every downstream consumer (training pipeline, DMH inference, recursive
# fallback, validation prediction generation, backtest origin retraining) reads
# the slim version directly off disk — no pruning logic in the hot path.
#
# Why this is the right architectural choice (per user request 2026-05-05):
#
#   * Single source of truth.  One decision, one place; downstream code
#     doesn't need to know about it.
#   * Smaller parquet on UC volumes.  Faster reads everywhere, less
#     storage cost, less network when Spark broadcasts the train DF.
#   * Inference-time consistency.  The model was trained on a slim
#     feature set; the parquet IS the contract — anyone reading
#     train_features.parquet gets exactly the columns the model
#     expects, no per-stage filter logic.
#   * Memory amplification savings on the Spark dispatcher.  When
#     mapInPandas tasks read sc.broadcast(train_df), each Python
#     worker process unpickles its OWN copy (no cross-process shared
#     memory).  With 16 simultaneous tasks per worker, every byte
#     trimmed off the broadcast saves 16x its size in worker RAM.
#     Empirically on UK COOKING AID (730k rows × 654 cols, ~2 GB
#     in-memory): 62% of cols are all-null/all-zero/all-constant in
#     train, freeing ~542 MB of broadcast = ~8.5 GB per worker once
#     amplified.  This is what makes the pipeline fit in 128 GB
#     workers without OOM.
#
# Critical correctness note (from user discussion 2026-05-05):
#
#   The "dead" decision MUST be made on the GLOBAL training panel,
#   never per key.  A feature can be all-zero for some keys (e.g.
#   a promotion that never ran on a particular SKU) but have
#   meaningful values on other keys (where the promo did run).
#   Per-key pruning would silently destroy signal.  Pooled
#   train_df is the right scope: if a column has zero variance
#   across ALL keys combined, no model trained on this dataset
#   could possibly use it.
#
# Leakage-safety:
#
#   The decision is made on TRAIN ONLY (val_df and test_df are not
#   inspected for the prune choice).  The same column list is then
#   dropped from val/test for consistency — this is safe because a
#   tree-based model that saw a column constant in training cannot
#   have learned anything useful from it, so its values in val/test
#   are irrelevant to the predictions.
# =============================================================================

def prune_dead_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    *,
    protected_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Drop columns with zero predictive signal across the training panel.

    A column is considered "dead" if, in ``train_df``, it satisfies any of:
      * all values are null
      * all (non-null) values are exactly 0 (numeric-typed columns only)
      * exactly one unique non-null value (constant column)

    The same dropped-column list is applied to ``val_df`` and ``test_df`` so
    the three frames stay schema-aligned for downstream consumers.

    Parameters
    ----------
    train_df : DataFrame
        Source of truth for the prune decision.  Inspected to identify
        which columns carry no signal.
    val_df : DataFrame
        Validation panel — only mutated by dropping the same columns.
    test_df : DataFrame, optional
        Test panel — same treatment as val_df.  Pass ``None`` if the
        caller doesn't have a test split (forecast-only mode).
    protected_cols : list of str, optional
        Column names that must NEVER be dropped even if they look dead
        (typically the key, date, target, and model_level columns).

    Returns
    -------
    (train_pruned, val_pruned, test_pruned, manifest)
        Pruned frames + a dict describing what was dropped.  The
        manifest is JSON-serialisable and intended to be persisted to
        ``dead_features_dropped.json`` alongside the parquet artifacts.
    """
    protected = set(protected_cols or [])

    all_null: List[str] = []
    all_zero: List[str] = []
    all_constant: List[Tuple[str, str]] = []  # (col, value) — keep value for diagnostics

    for col in train_df.columns:
        if col in protected:
            continue
        s = train_df[col]
        # All-null fast path — avoids dropna() materialisation when the
        # whole column is missing.
        n_nulls = s.isna().sum()
        if n_nulls == len(s):
            all_null.append(col)
            continue
        non_null = s.dropna()
        if non_null.empty:
            # Pathologically all-null after dropna (shouldn't happen given
            # the n_nulls check above, but defensive).
            all_null.append(col)
            continue
        # Single unique value → constant or all-zero.
        nunique = non_null.nunique()
        if nunique == 1:
            val = non_null.iloc[0]
            try:
                # Numeric zero: separate bucket so the diagnostic log can
                # distinguish "promo dummy that never fired" (all zero)
                # from "category metadata" (constant string).
                if isinstance(val, (int, float, np.integer, np.floating)) and val == 0:
                    all_zero.append(col)
                else:
                    all_constant.append((col, str(val)))
            except Exception:
                # Defensive: anything we can't classify as zero falls
                # into the constant bucket so it's still dropped.
                all_constant.append((col, repr(val)))

    dropped = all_null + all_zero + [c for c, _ in all_constant]

    if not dropped:
        manifest = {
            "dropped_total": 0,
            "dropped_pct": 0.0,
            "all_null": [],
            "all_zero": [],
            "all_constant": [],
            "kept_count": len(train_df.columns),
            "original_count": len(train_df.columns),
            "protected_cols": sorted(protected),
        }
        return train_df, val_df, test_df, manifest

    train_pruned = train_df.drop(columns=dropped, errors='ignore')
    val_pruned = val_df.drop(columns=dropped, errors='ignore')
    test_pruned = (
        test_df.drop(columns=dropped, errors='ignore')
        if test_df is not None else None
    )

    manifest = {
        "dropped_total": len(dropped),
        "dropped_pct": round(len(dropped) / len(train_df.columns), 4),
        "original_count": len(train_df.columns),
        "kept_count": len(train_pruned.columns),
        "all_null": all_null,
        "all_zero": all_zero,
        "all_constant": [{"column": c, "value": v} for c, v in all_constant],
        "protected_cols": sorted(protected),
    }

    return train_pruned, val_pruned, test_pruned, manifest


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    feature_type: str  # 'lag', 'rolling', 'calendar', 'categorical', etc.
    source_column: str
    params: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.0


@dataclass
class FeatureEngineeringResult:
    """Result from feature engineering pipeline."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    encoding_specs: Dict[str, Any]
    feature_metadata: Dict[str, Any]
    n_features_created: int
    n_rows_train: int
    n_rows_val: int
    n_rows_test: int


# =============================================================================
# 0. DATA TYPE VALIDATION & CONVERSION (Config-Driven)
# =============================================================================

def validate_and_convert_column_types(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    drop_unconvertible: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and convert column types based on config declarations.

    For columns declared as NUMERIC in config:
    - If already numeric, keep as is
    - If not numeric, try to convert using pd.to_numeric(errors='coerce')
    - If conversion fails (>50% values become NaN), drop the column

    For columns declared as CATEGORICAL in config:
    - If already string/object, keep as is
    - If numeric, convert to string
    - If conversion produces invalid results, drop the column

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numeric_cols : List[str]
        Columns declared as numeric in config
    categorical_cols : List[str]
        Columns declared as categorical in config
    drop_unconvertible : bool
        If True, drop columns that can't be converted. If False, raise error.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Cleaned DataFrame with correct column types
        - Report dict with conversion details:
          - 'numeric_validated': columns that were already numeric
          - 'numeric_converted': columns successfully converted to numeric
          - 'numeric_dropped': columns that couldn't be converted to numeric
          - 'categorical_validated': columns that were already categorical
          - 'categorical_converted': columns successfully converted to categorical
          - 'categorical_dropped': columns that couldn't be converted
          - 'missing_cols': columns in config but not in data

    Example
    -------
    >>> df, report = validate_and_convert_column_types(
    ...     df,
    ...     numeric_cols=['price', 'quantity'],
    ...     categorical_cols=['category', 'region']
    ... )
    >>> print(report['numeric_dropped'])  # Columns that failed conversion
    """
    df = df.copy()
    report = {
        'numeric_validated': [],
        'numeric_converted': [],
        'numeric_dropped': [],
        'categorical_validated': [],
        'categorical_converted': [],
        'categorical_dropped': [],
        'missing_cols': [],
    }

    # Track columns to drop at the end
    cols_to_drop = []

    # -------------------------------------------------------------------------
    # Validate NUMERIC columns
    # -------------------------------------------------------------------------
    for col in numeric_cols:
        if col not in df.columns:
            report['missing_cols'].append(col)
            logger.debug(f"Numeric column '{col}' not found in data - skipping")
            continue

        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            report['numeric_validated'].append(col)
            continue

        # Try to convert to numeric
        original_non_null = df[col].notna().sum()
        converted = pd.to_numeric(df[col], errors='coerce')
        converted_non_null = converted.notna().sum()

        # Check if conversion retained at least 50% of non-null values
        if original_non_null > 0 and converted_non_null / original_non_null >= 0.5:
            df[col] = converted
            report['numeric_converted'].append(col)
            logger.info(f"Converted '{col}' to numeric ({converted_non_null}/{original_non_null} values retained)")
        else:
            report['numeric_dropped'].append(col)
            if drop_unconvertible:
                cols_to_drop.append(col)
                logger.warning(
                    f"Dropping '{col}' - declared as numeric but only "
                    f"{converted_non_null}/{original_non_null} values convertible"
                )
            else:
                raise ValueError(
                    f"Column '{col}' declared as numeric in config but cannot be converted. "
                    f"Only {converted_non_null}/{original_non_null} values are numeric."
                )

    # -------------------------------------------------------------------------
    # Validate CATEGORICAL columns
    # -------------------------------------------------------------------------
    for col in categorical_cols:
        if col not in df.columns:
            report['missing_cols'].append(col)
            logger.debug(f"Categorical column '{col}' not found in data - skipping")
            continue

        # Check if already string/object
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            report['categorical_validated'].append(col)
            continue

        # Check if categorical dtype
        if pd.api.types.is_categorical_dtype(df[col]):
            report['categorical_validated'].append(col)
            continue

        # Try to convert to string (for numeric columns declared as categorical)
        try:
            # Convert to string, handling NaN
            df[col] = df[col].astype(str).replace('nan', pd.NA).replace('None', pd.NA)
            report['categorical_converted'].append(col)
            logger.info(f"Converted '{col}' to categorical (string)")
        except Exception as e:
            report['categorical_dropped'].append(col)
            if drop_unconvertible:
                cols_to_drop.append(col)
                logger.warning(f"Dropping '{col}' - declared as categorical but conversion failed: {e}")
            else:
                raise ValueError(
                    f"Column '{col}' declared as categorical in config but cannot be converted: {e}"
                )

    # Drop unconvertible columns
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
        logger.info(f"Dropped {len(cols_to_drop)} unconvertible columns: {cols_to_drop}")

    # Log summary
    logger.info(
        f"Column type validation complete: "
        f"{len(report['numeric_validated'])} numeric OK, "
        f"{len(report['numeric_converted'])} numeric converted, "
        f"{len(report['numeric_dropped'])} numeric dropped, "
        f"{len(report['categorical_validated'])} categorical OK, "
        f"{len(report['categorical_converted'])} categorical converted, "
        f"{len(report['categorical_dropped'])} categorical dropped"
    )

    return df, report


# =============================================================================
# 1. LAG FEATURES
# =============================================================================

def create_lag_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    lags: List[int] = None,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create lag features for time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must be sorted by date within groups)
    group_cols : str or List[str]
        Columns to group by (e.g., key columns)
    target_col : str
        Column to create lags for
    lags : List[int]
        List of lag periods (default: [1, 2, 4, 8, 13, 26, 52])
    prefix : str, optional
        Prefix for lag column names (defaults to target_col)

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features added

    Example
    -------
    >>> df = create_lag_features(df, ['key'], 'sales', lags=[1, 4, 52])
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    if lags is None:
        lags = [1, 2, 3, 6, 12] if time_format == 'year_month' else [1, 2, 4, 8, 13, 26, 52]

    for lag in lags:
        col_name = f'{prefix}_lag_{lag}'
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)

    return df


def create_lag_combinations(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    lag_pairs: List[Tuple[int, int]] = None,
    operations: List[str] = None,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create lag combination features (differences, ratios, products).

    State-of-the-art: Capture momentum and trend signals from lag interactions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with lag features already created
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column name
    lag_pairs : List[Tuple[int, int]]
        Pairs of lags to combine (default: [(1,4), (1,52), (4,52), (1,13)])
    operations : List[str]
        Operations: 'diff', 'ratio', 'pct_change' (default: all)
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with lag combination features

    Example
    -------
    >>> df = create_lag_combinations(df, ['key'], 'sales')
    """
    df = df.copy()
    prefix = prefix or target_col
    if lag_pairs is None:
        if time_format == 'year_month':
            lag_pairs = [(1, 3), (1, 6), (1, 12), (3, 12)]
        else:
            lag_pairs = [(1, 4), (1, 52), (4, 52), (1, 13), (4, 13)]
    operations = operations or ['diff', 'ratio']

    for lag_short, lag_long in lag_pairs:
        col_short = f'{prefix}_lag_{lag_short}'
        col_long = f'{prefix}_lag_{lag_long}'

        if col_short not in df.columns or col_long not in df.columns:
            continue

        if 'diff' in operations:
            df[f'{prefix}_lag_diff_{lag_short}_{lag_long}'] = df[col_short] - df[col_long]

        if 'ratio' in operations:
            df[f'{prefix}_lag_ratio_{lag_short}_{lag_long}'] = (
                df[col_short] / (df[col_long] + 1e-10)
            )

        if 'pct_change' in operations:
            df[f'{prefix}_lag_pct_chg_{lag_short}_{lag_long}'] = (
                (df[col_short] - df[col_long]) / (df[col_long] + 1e-10)
            )

    return df


def create_momentum_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    periods: List[int] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create momentum features (financial-style indicators adapted for demand).

    State-of-the-art: Captures acceleration/deceleration of demand.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column
    periods : List[int]
        Periods for momentum calculation (default: [4, 13, 26])
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with momentum features

    Example
    -------
    >>> df = create_momentum_features(df, ['key'], 'sales')
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    periods = periods or [4, 13, 26]

    for period in periods:
        # Rate of Change (ROC)
        df[f'{prefix}_roc_{period}'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).pct_change(periods=period)
        )

        # Momentum (difference from n periods ago)
        df[f'{prefix}_momentum_{period}'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1) - x.shift(1 + period)
        )

        # Acceleration (change in momentum)
        df[f'{prefix}_acceleration_{period}'] = df.groupby(group_cols)[f'{prefix}_momentum_{period}'].transform(
            lambda x: x.diff()
        )

    return df


def create_seasonal_lag_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    seasonal_periods: List[int] = None,
    include_yoy: bool = True,
    include_qoq: bool = True,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create seasonal lag features (YoY, QoQ comparisons) WITHOUT using current target.

    CRITICAL RULE: No feature can use the current row's target value.
    All features use shift(1) as the reference point:
    - yoy_lag_diff = lag_1 - lag_53 (NOT current - lag_52)
    - qoq_lag_diff = lag_1 - lag_14 (NOT current - lag_13)

    During recursive forecasting, these features MUST be recalculated using
    predictions instead of actuals for validation/test periods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column
    seasonal_periods : List[int]
        Seasonal periods (default: [52] for weekly data)
    include_yoy : bool
        Include Year-over-Year features
    include_qoq : bool
        Include Quarter-over-Quarter features
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with seasonal lag features (no current-target leakage)
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    if seasonal_periods is None:
        seasonal_periods = [12] if time_format == 'year_month' else [52]

    for period in seasonal_periods:
        # Seasonal lag (these are already safe - just shifted values)
        df[f'{prefix}_seasonal_lag_{period}'] = df.groupby(group_cols)[target_col].shift(period)

        # LEAKAGE-FREE Year-over-Year: lag_1 vs lag_(1+period)
        # For weekly data (period=52): lag_1 vs lag_53
        # For monthly data (period=12): lag_1 vs lag_13
        if include_yoy and period in (52, 12):
            lag_short = 1
            lag_long = 1 + period  # 53 for weekly, 13 for monthly

            lag_short_col = f'{prefix}_lag_{lag_short}'
            lag_long_col = f'{prefix}_lag_{lag_long}'

            if lag_short_col not in df.columns:
                df[lag_short_col] = df.groupby(group_cols)[target_col].shift(lag_short)
            if lag_long_col not in df.columns:
                df[lag_long_col] = df.groupby(group_cols)[target_col].shift(lag_long)

            df[f'{prefix}_yoy_lag_diff'] = df[lag_short_col] - df[lag_long_col]
            df[f'{prefix}_yoy_lag_ratio'] = (
                df[lag_short_col] / (df[lag_long_col].abs() + 1e-10)
            )

        # LEAKAGE-FREE Quarter-over-Quarter: lag_1 vs lag_(1+quarter)
        # For weekly data (period=52): lag_1 vs lag_14 (13 weeks = 1 quarter)
        # For monthly data (period=12): lag_1 vs lag_4 (3 months = 1 quarter)
        if include_qoq and period in (52, 12):
            quarter_periods = 13 if period == 52 else 3
            lag_short = 1
            lag_long = 1 + quarter_periods  # 14 for weekly, 4 for monthly

            lag_short_col = f'{prefix}_lag_{lag_short}'
            lag_long_col = f'{prefix}_lag_{lag_long}'

            if lag_short_col not in df.columns:
                df[lag_short_col] = df.groupby(group_cols)[target_col].shift(lag_short)
            if lag_long_col not in df.columns:
                df[lag_long_col] = df.groupby(group_cols)[target_col].shift(lag_long)

            df[f'{prefix}_qoq_lag_diff'] = df[lag_short_col] - df[lag_long_col]

    return df


# =============================================================================
# 2. ROLLING FEATURES
# =============================================================================

def create_rolling_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    windows: List[int] = None,
    aggregations: List[str] = None,
    shift_periods: int = 1,
    min_periods: int = 1,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create rolling window features for time series.

    IMPORTANT: Uses shift(shift_periods) to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must be sorted by date within groups)
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Column to compute rolling features for
    windows : List[int]
        List of window sizes (default: [4, 8, 13, 26, 52])
    aggregations : List[str]
        Aggregation functions (default: ['mean', 'std', 'min', 'max'])
    shift_periods : int
        Periods to shift before rolling (default 1 to prevent leakage)
    min_periods : int
        Minimum periods for valid calculation
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features added

    Example
    -------
    >>> df = create_rolling_features(df, ['key'], 'sales', windows=[4, 13])
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    if windows is None:
        windows = [3, 6, 12] if time_format == 'year_month' else [4, 8, 13, 26, 52]
    aggregations = aggregations or ['mean', 'std', 'min', 'max']

    for window in windows:
        for agg in aggregations:
            col_name = f'{prefix}_roll_{window}_{agg}'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(shift_periods).rolling(window, min_periods=min_periods).agg(agg)
            )

    return df


def create_ewm_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    spans: List[int] = None,
    shift_periods: int = 1,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create Exponential Weighted Mean features.

    State-of-the-art: EWM gives more weight to recent observations,
    better capturing recent trends than simple rolling averages.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Column to compute EWM for
    spans : List[int]
        List of EWM spans (default: [4, 8, 13, 26])
    shift_periods : int
        Periods to shift before EWM
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with EWM features added

    Example
    -------
    >>> df = create_ewm_features(df, ['key'], 'sales', spans=[4, 13])
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    spans = spans or [4, 8, 13, 26]

    for span in spans:
        # EWM mean
        col_name = f'{prefix}_ewm_{span}'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).ewm(span=span, min_periods=1).mean()
        )

        # EWM std (volatility measure)
        col_name_std = f'{prefix}_ewm_std_{span}'
        df[col_name_std] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).ewm(span=span, min_periods=2).std()
        )

    return df


def create_expanding_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    aggregations: List[str] = None,
    shift_periods: int = 1,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create expanding window features (cumulative statistics).

    State-of-the-art: Captures lifetime/historical patterns per key.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Column to compute expanding features for
    aggregations : List[str]
        Aggregation functions (default: ['mean', 'std', 'min', 'max', 'sum'])
    shift_periods : int
        Periods to shift
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with expanding features

    Example
    -------
    >>> df = create_expanding_features(df, ['key'], 'sales')
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    aggregations = aggregations or ['mean', 'std', 'min', 'max']

    for agg in aggregations:
        col_name = f'{prefix}_expanding_{agg}'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).expanding(min_periods=1).agg(agg)
        )

    # Expanding count (history length)
    df[f'{prefix}_expanding_count'] = df.groupby(group_cols).cumcount()

    return df


def create_rolling_volatility_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    windows: List[int] = None,
    shift_periods: int = 1,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling volatility/variability features.

    State-of-the-art: Volatility features help models adapt to demand uncertainty.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column
    windows : List[int]
        Window sizes (default: [4, 13, 26])
    shift_periods : int
        Periods to shift
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with volatility features

    Example
    -------
    >>> df = create_rolling_volatility_features(df, ['key'], 'sales')
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    windows = windows or [4, 13, 26]

    for window in windows:
        # Coefficient of variation (CV)
        mean_col = f'{prefix}_roll_{window}_mean'
        std_col = f'{prefix}_roll_{window}_std'

        if mean_col not in df.columns:
            df[mean_col] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(shift_periods).rolling(window, min_periods=1).mean()
            )
        if std_col not in df.columns:
            df[std_col] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(shift_periods).rolling(window, min_periods=2).std()
            )

        df[f'{prefix}_cv_{window}'] = df[std_col] / (df[mean_col] + 1e-10)

        # Range (max - min)
        df[f'{prefix}_range_{window}'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).rolling(window, min_periods=1).max() -
                      x.shift(shift_periods).rolling(window, min_periods=1).min()
        )

        # IQR (interquartile range)
        iqr_min_periods = min(4, window)
        df[f'{prefix}_iqr_{window}'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).rolling(window, min_periods=iqr_min_periods).quantile(0.75) -
                      x.shift(shift_periods).rolling(window, min_periods=iqr_min_periods).quantile(0.25)
        )

    return df


# =============================================================================
# 3. CALENDAR & CYCLICAL FEATURES
# =============================================================================

def create_calendar_features(
    df: pd.DataFrame,
    date_col: str,
    date_format: str = None,
) -> pd.DataFrame:
    """
    Create comprehensive calendar features from a date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Date column name
    date_format : str, optional
        Date format for parsing (e.g., '%Y%m%d', 'YYYYWW')

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar features added

    Example
    -------
    >>> df = create_calendar_features(df, 'date')
    """
    df = df.copy()

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        sample_val = str(df[date_col].iloc[0])
        str_len = len(sample_val)

        # Try to parse 6-digit format: YYYYWW (year_week) or YYYYMM (year_month)
        if str_len == 6 and sample_val.isdigit():
            sub_period = int(sample_val[4:])
            df['_year'] = df[date_col].astype(str).str[:4].astype(int)
            df['_sub'] = df[date_col].astype(str).str[4:].astype(int)

            if sub_period > 12:
                # YYYYWW format (week > 12 confirms weekly)
                df['_date'] = pd.to_datetime(
                    df['_year'].astype(str) + df['_sub'].astype(str).str.zfill(2) + '1',
                    format='%Y%W%w'
                )
            else:
                # Could be YYYYMM or YYYYWW with low week number
                # Use date_format hint if available, otherwise try month first
                if date_format and 'month' in str(date_format).lower():
                    # Explicitly month format
                    df['_date'] = pd.to_datetime(
                        df['_year'].astype(str) + '-' + df['_sub'].astype(str).str.zfill(2) + '-01',
                        format='%Y-%m-%d'
                    )
                else:
                    # Check if ANY value in column has sub > 12 → must be weeks
                    max_sub = df['_sub'].max()
                    if max_sub > 12:
                        df['_date'] = pd.to_datetime(
                            df['_year'].astype(str) + df['_sub'].astype(str).str.zfill(2) + '1',
                            format='%Y%W%w'
                        )
                    else:
                        # All values <= 12: treat as month (YYYYMM)
                        df['_date'] = pd.to_datetime(
                            df['_year'].astype(str) + '-' + df['_sub'].astype(str).str.zfill(2) + '-01',
                            format='%Y-%m-%d'
                        )
            df = df.drop(columns=['_year', '_sub'])
            date_col_parsed = '_date'
        # Try to parse year-week format (YYYY-WW - 7 chars with hyphen)
        elif str_len == 7 and '-' in sample_val:
            parts = df[date_col].astype(str).str.split('-', expand=True)
            df['_year'] = parts[0].astype(int)
            df['_week'] = parts[1].astype(int)
            # Create date from ISO week: year + week + day 1 (Monday)
            df['_date'] = pd.to_datetime(
                df['_year'].astype(str) + df['_week'].astype(str).str.zfill(2) + '1',
                format='%G%V%u'  # ISO year, ISO week, weekday (1=Monday)
            )
            df = df.drop(columns=['_year', '_week'])
            date_col_parsed = '_date'
        else:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
            date_col_parsed = date_col
    else:
        date_col_parsed = date_col

    # Basic calendar features
    df['year'] = df[date_col_parsed].dt.year
    df['month'] = df[date_col_parsed].dt.month
    df['week_of_year'] = df[date_col_parsed].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df[date_col_parsed].dt.dayofweek
    df['quarter'] = df[date_col_parsed].dt.quarter
    df['day_of_year'] = df[date_col_parsed].dt.dayofyear

    # Special period flags
    df['is_month_start'] = df[date_col_parsed].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col_parsed].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col_parsed].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col_parsed].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_col_parsed].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_col_parsed].dt.is_year_end.astype(int)

    # Week position within month
    df['week_of_month'] = (df['day_of_year'] - 1) // 7 % 5 + 1

    # Clean up temporary column
    if date_col_parsed == '_date':
        df = df.drop(columns=['_date'])

    return df


def create_cyclical_features(
    df: pd.DataFrame,
    column: str,
    period: int,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create cyclical (sin/cos) encoding for a periodic feature.

    State-of-the-art: Captures circular nature of time (week 52 is close to week 1).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to encode
    period : int
        Period length (e.g., 52 for weeks, 12 for months)
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with cyclical features

    Example
    -------
    >>> df = create_cyclical_features(df, 'week_of_year', period=52)
    """
    df = df.copy()
    prefix = prefix or column

    df[f'{prefix}_sin'] = np.sin(2 * np.pi * df[column] / period)
    df[f'{prefix}_cos'] = np.cos(2 * np.pi * df[column] / period)

    return df


def create_fourier_features(
    df: pd.DataFrame,
    date_col: str,
    periods: List[int] = None,
    n_harmonics: int = 3,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create Fourier features for capturing complex seasonality.

    State-of-the-art: Multiple harmonics capture different frequency patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Date column (or week_of_year column)
    periods : List[int]
        Seasonal periods to model (default: [52, 26, 13] for weekly data)
    n_harmonics : int
        Number of harmonics per period (default: 3)

    Returns
    -------
    pd.DataFrame
        DataFrame with Fourier features

    Example
    -------
    >>> df = create_fourier_features(df, 'week_of_year')
    """
    df = df.copy()
    if periods is None:
        periods = [12, 6, 4] if time_format == 'year_month' else [52, 26, 13]

    # Determine time index
    if time_format == 'year_month':
        # For monthly data, use month (1-12) as the time index
        if df[date_col].dtype == 'int64':
            t = (df[date_col] % 100).values  # Extract month from YYYYMM
        else:
            t = pd.to_datetime(df[date_col].astype(str).str[:6], format='%Y%m').dt.month.values
    elif 'week_of_year' in df.columns:
        t = df['week_of_year'].values
    elif df[date_col].dtype == 'int64':
        # YYYYWW format
        t = (df[date_col] % 100).values  # Extract week
    else:
        t = pd.to_datetime(df[date_col]).dt.isocalendar().week.values

    for period in periods:
        for k in range(1, n_harmonics + 1):
            df[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
            df[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)

    return df


def _compute_easter(year: int) -> 'datetime.date':
    """Compute Easter Sunday date using the Anonymous Gregorian algorithm."""
    import datetime
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return datetime.date(year, month, day + 1)


def build_dach_holiday_calendar(
    year_weeks: list,
    date_col: str = 'year_week',
) -> pd.DataFrame:
    """
    Build a DACH (Germany/Austria/Switzerland) holiday calendar at weekly
    granularity.  Returns a DataFrame with one row per year_week containing
    holiday indicator features that can be merged directly into any DataFrame
    keyed on ``date_col`` in 'YYYY-WW' format.

    Features created:
        is_holiday          – 1 if the week contains a major DACH holiday
        holiday_type        – categorical: christmas / new_year / easter / other / none
        is_pre_holiday_1w   – 1 week before a holiday week
        is_pre_holiday_2w   – 2 weeks before a holiday week
        is_post_holiday_1w  – 1 week after a holiday week
        is_post_holiday_2w  – 2 weeks after a holiday week
        weeks_to_nearest_holiday – signed integer distance to closest holiday week
    """
    import datetime

    # Deduplicate and sort input weeks
    unique_weeks = sorted(set(year_weeks))

    # Parse year_week strings → (year, week) tuples and Monday dates
    week_dates = {}  # year_week -> Monday of that ISO week
    for yw in unique_weeks:
        parts = yw.split('-')
        yr, wk = int(parts[0]), int(parts[1])
        # Monday of ISO week
        monday = datetime.date.fromisocalendar(yr, wk, 1)
        week_dates[yw] = (yr, wk, monday)

    # Determine year range (extend ±1 year for pre/post window)
    years = set()
    for yr, wk, _ in week_dates.values():
        years.add(yr)
    min_year = min(years) - 1
    max_year = max(years) + 1

    # Build holiday list: (date, holiday_type)
    holidays = []
    for yr in range(min_year, max_year + 1):
        # Fixed-date holidays
        holidays.append((datetime.date(yr, 12, 24), 'christmas'))  # Christmas Eve
        holidays.append((datetime.date(yr, 12, 25), 'christmas'))  # Christmas Day
        holidays.append((datetime.date(yr, 12, 26), 'christmas'))  # St. Stephen's
        holidays.append((datetime.date(yr, 12, 31), 'new_year'))   # New Year's Eve
        holidays.append((datetime.date(yr, 1, 1), 'new_year'))     # New Year's Day
        holidays.append((datetime.date(yr, 1, 6), 'other'))        # Epiphany (parts of DE)
        holidays.append((datetime.date(yr, 5, 1), 'other'))        # Labour Day
        holidays.append((datetime.date(yr, 10, 3), 'other'))       # German Unity Day
        holidays.append((datetime.date(yr, 11, 1), 'other'))       # All Saints' Day
        holidays.append((datetime.date(yr, 10, 31), 'other'))      # Reformation Day (parts of DE)

        # Easter-dependent movable holidays
        easter = _compute_easter(yr)
        td = datetime.timedelta
        holidays.append((easter - td(days=2), 'easter'))           # Good Friday
        holidays.append((easter, 'easter'))                        # Easter Sunday
        holidays.append((easter + td(days=1), 'easter'))           # Easter Monday
        holidays.append((easter + td(days=39), 'other'))           # Ascension Day
        holidays.append((easter + td(days=49), 'other'))           # Whit Sunday
        holidays.append((easter + td(days=50), 'other'))           # Whit Monday
        holidays.append((easter + td(days=60), 'other'))           # Corpus Christi

    # Map holidays → ISO week strings, keeping the highest-priority holiday_type per week
    # Priority: christmas > new_year > easter > other
    type_priority = {'christmas': 3, 'new_year': 2, 'easter': 1, 'other': 0}
    holiday_weeks = {}  # year_week_str -> holiday_type
    for hdate, htype in holidays:
        iso_yr, iso_wk, _ = hdate.isocalendar()
        yw_str = f"{iso_yr}-{iso_wk:02d}"
        existing = holiday_weeks.get(yw_str)
        if existing is None or type_priority[htype] > type_priority[existing]:
            holiday_weeks[yw_str] = htype

    # Build the output DataFrame
    rows = []
    for yw in unique_weeks:
        htype = holiday_weeks.get(yw, 'none')
        rows.append({
            date_col: yw,
            'is_holiday': 1 if htype != 'none' else 0,
            'holiday_type': htype,
        })
    cal = pd.DataFrame(rows)

    # Pre/post holiday windows and distance — operate on the sorted calendar
    cal = cal.sort_values(date_col).reset_index(drop=True)
    is_h = cal['is_holiday'].values

    # Find indices of holiday weeks
    h_indices = set(i for i, v in enumerate(is_h) if v == 1)

    pre1 = np.zeros(len(cal), dtype=int)
    pre2 = np.zeros(len(cal), dtype=int)
    post1 = np.zeros(len(cal), dtype=int)
    post2 = np.zeros(len(cal), dtype=int)
    dist = np.full(len(cal), 999, dtype=int)

    for hi in h_indices:
        dist[hi] = 0
        if hi - 1 >= 0 and hi - 1 not in h_indices:
            pre1[hi - 1] = 1
        if hi - 2 >= 0 and hi - 2 not in h_indices:
            pre2[hi - 2] = 1
        if hi + 1 < len(cal) and hi + 1 not in h_indices:
            post1[hi + 1] = 1
        if hi + 2 < len(cal) and hi + 2 not in h_indices:
            post2[hi + 2] = 1

    # Compute distance to nearest holiday for all rows
    for i in range(len(cal)):
        if i in h_indices:
            continue
        min_dist = 999
        for hi in h_indices:
            d = abs(i - hi)
            if d < abs(min_dist):
                min_dist = d
        dist[i] = min_dist

    cal['is_pre_holiday_1w'] = pre1
    cal['is_pre_holiday_2w'] = pre2
    cal['is_post_holiday_1w'] = post1
    cal['is_post_holiday_2w'] = post2
    cal['weeks_to_nearest_holiday'] = dist

    return cal


def create_holiday_features(
    df: pd.DataFrame,
    date_col: str,
    holiday_df: Optional[pd.DataFrame] = None,
    holiday_col: str = 'is_holiday',
    pre_holiday_windows: List[int] = None,
    post_holiday_windows: List[int] = None,
) -> pd.DataFrame:
    """
    Create holiday and event features by merging a holiday calendar.

    If ``holiday_df`` is None and the date column looks like 'year_week'
    (format YYYY-WW), a DACH holiday calendar is automatically generated.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Date column (supports 'YYYY-WW' week format)
    holiday_df : pd.DataFrame, optional
        DataFrame with date and holiday indicator columns.  If None and
        date_col contains week strings, builds DACH calendar automatically.
    holiday_col : str
        Name of holiday indicator column
    pre_holiday_windows : List[int]
        Windows before holiday (default: [1, 2, 3])
    post_holiday_windows : List[int]
        Windows after holiday (default: [1, 2])

    Returns
    -------
    pd.DataFrame
        DataFrame with holiday features added
    """
    df = df.copy()

    # Auto-build DACH calendar when no holiday_df is provided
    if holiday_df is None and date_col in df.columns:
        sample_vals = df[date_col].dropna().astype(str).head(5).tolist()
        looks_like_week = any('-' in v and len(v.split('-')) == 2 for v in sample_vals)
        if looks_like_week:
            year_weeks = df[date_col].dropna().astype(str).unique().tolist()
            holiday_df = build_dach_holiday_calendar(year_weeks, date_col)

    if holiday_df is not None:
        # Drop columns that will be merged to avoid duplicates
        merge_cols = [c for c in holiday_df.columns if c != date_col and c in df.columns]
        if merge_cols:
            df = df.drop(columns=merge_cols)
        df = df.merge(holiday_df, on=date_col, how='left')
        # Fill NaN for any rows that didn't match
        for col in holiday_df.columns:
            if col == date_col:
                continue
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('none')
            else:
                df[col] = df[col].fillna(0)
    else:
        # Fallback: create empty features
        df['is_holiday'] = 0
        df['holiday_type'] = 'none'
        df['is_pre_holiday_1w'] = 0
        df['is_pre_holiday_2w'] = 0
        df['is_post_holiday_1w'] = 0
        df['is_post_holiday_2w'] = 0
        df['weeks_to_nearest_holiday'] = 99

    return df


# =============================================================================
# 4. TREND & DECOMPOSITION FEATURES
# =============================================================================

def create_trend_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    diff_periods: List[int] = None,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create trend-related features (differencing, percent change) WITHOUT using current target.

    CRITICAL RULE: No feature can use the current row's target value.
    All features use shift(1) minimum - they compare lag_1 vs lag_(1+period).

    During recursive forecasting, these features MUST be recalculated using
    predictions instead of actuals for validation/test periods.

    Features created:
    - {prefix}_lag_diff_{period}: lag_1 - lag_(1+period)
    - {prefix}_lag_pct_change_{period}: (lag_1 - lag_(1+period)) / |lag_(1+period)|
    - {prefix}_lag_direction_{period}: sign of lag_diff

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column
    diff_periods : List[int]
        Differencing periods (default: [1, 4, 13, 52])
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with trend features (no current-target leakage)
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    if diff_periods is None:
        diff_periods = [1, 3, 6, 12] if time_format == 'year_month' else [1, 4, 13, 52]

    for period in diff_periods:
        # LEAKAGE-FREE: Use shift(1) as the "current known" value
        # lag_diff_period = lag_1 - lag_(1 + period)
        # This compares the most recent known value to the value (period) steps before it
        lag_short = 1  # Always use lag_1 as the reference point
        lag_long = 1 + period

        lag_short_col = f'{prefix}_lag_{lag_short}'
        lag_long_col = f'{prefix}_lag_{lag_long}'

        # Create lag columns if they don't exist
        if lag_short_col not in df.columns:
            df[lag_short_col] = df.groupby(group_cols)[target_col].shift(lag_short)
        if lag_long_col not in df.columns:
            df[lag_long_col] = df.groupby(group_cols)[target_col].shift(lag_long)

        # Lagged difference (lag_K - lag_(K+period))
        df[f'{prefix}_lag_diff_{period}'] = df[lag_short_col] - df[lag_long_col]

        # Lagged percent change
        df[f'{prefix}_lag_pct_change_{period}'] = (
            (df[lag_short_col] - df[lag_long_col]) / (df[lag_long_col].abs() + 1e-10)
        )

        # Sign of lagged change (direction)
        df[f'{prefix}_lag_direction_{period}'] = np.sign(df[f'{prefix}_lag_diff_{period}']).fillna(0)

    return df


def create_detrended_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    method: str = 'linear',
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create detrended target and residual features.

    State-of-the-art: Remove trend to help models focus on patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column
    method : str
        Detrending method: 'linear', 'rolling', 'ewm'
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with detrended features

    Example
    -------
    >>> df = create_detrended_features(df, ['key'], 'sales', method='linear')
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    def detrend_linear(series):
        """Remove linear trend from series."""
        x = np.arange(len(series))
        valid_mask = ~np.isnan(series)
        if valid_mask.sum() < 2:
            return series
        coeffs = np.polyfit(x[valid_mask], series[valid_mask], 1)
        trend = np.polyval(coeffs, x)
        return series - trend

    def detrend_rolling(series, window=13):
        """Remove rolling mean trend."""
        trend = series.rolling(window, min_periods=1, center=True).mean()
        return series - trend

    if method == 'linear':
        df[f'{prefix}_detrended'] = df.groupby(group_cols)[target_col].transform(detrend_linear)
    elif method == 'rolling':
        df[f'{prefix}_detrended'] = df.groupby(group_cols)[target_col].transform(
            lambda x: detrend_rolling(x)
        )
    elif method == 'ewm':
        df[f'{prefix}_trend_ewm'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.ewm(span=13, min_periods=1).mean()
        )
        df[f'{prefix}_detrended'] = df[target_col] - df[f'{prefix}_trend_ewm']

    return df


def create_stl_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    seasonal_period: int = None,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create STL (Seasonal-Trend decomposition) features.

    State-of-the-art: STL separates trend, seasonality, and residual components.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column
    seasonal_period : int
        Seasonal period (default: 52 for weekly data)
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with STL decomposition features

    Example
    -------
    >>> df = create_stl_features(df, ['key'], 'sales')
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    if seasonal_period is None:
        seasonal_period = 12 if time_format == 'year_month' else 52

    try:
        from statsmodels.tsa.seasonal import STL

        def stl_decompose(series):
            """Perform STL decomposition on a single series."""
            # Need at least 2 seasonal periods for STL
            if len(series.dropna()) < 2 * seasonal_period:
                return pd.DataFrame({
                    'trend': [np.nan] * len(series),
                    'seasonal': [np.nan] * len(series),
                    'residual': [np.nan] * len(series),
                }, index=series.index)

            try:
                stl = STL(series.dropna(), seasonal=seasonal_period + 1, robust=True)
                result = stl.fit()
                return pd.DataFrame({
                    'trend': result.trend,
                    'seasonal': result.seasonal,
                    'residual': result.resid,
                }, index=series.dropna().index)
            except Exception:
                return pd.DataFrame({
                    'trend': [np.nan] * len(series),
                    'seasonal': [np.nan] * len(series),
                    'residual': [np.nan] * len(series),
                }, index=series.index)

        # Apply STL per group
        stl_results = df.groupby(group_cols)[target_col].apply(stl_decompose)

        # Flatten and merge back
        for component in ['trend', 'seasonal', 'residual']:
            col_name = f'{prefix}_stl_{component}'
            if isinstance(stl_results, pd.DataFrame):
                df[col_name] = stl_results[component].values
            else:
                df[col_name] = np.nan

    except ImportError:
        logger.warning("statsmodels not available for STL decomposition. Skipping.")

    return df


# =============================================================================
# 5. INTERMITTENCY FEATURES
# =============================================================================

def create_intermittency_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    windows: List[int] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create features for intermittent demand modeling WITHOUT using current target.

    CRITICAL RULE: No feature can use the current row's target value.
    All features use shift(1) minimum.

    During recursive forecasting, these features MUST be recalculated using
    predictions instead of actuals for validation/test periods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Columns to group by
    target_col : str
        Target column
    windows : List[int]
        Windows for demand frequency calculation
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with intermittency features (no current-target leakage)
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    windows = windows or [4, 13, 26]

    # LEAKAGE-FREE: Binary demand occurred using shift(1) - did demand occur last period?
    df[f'{prefix}_lag1_demand_occurred'] = (df.groupby(group_cols)[target_col].shift(1) > 0).astype(int)

    # Periods since last non-zero demand (using shift(1) history)
    def periods_since_nonzero_shifted(x):
        """Count periods since last non-zero, using shift(1) to avoid current value."""
        shifted = x.shift(1)
        result = []
        count = 0
        for val in shifted:
            if pd.isna(val):
                result.append(np.nan)
            elif val > 0:
                result.append(count)
                count = 0
            else:
                result.append(count)
                count += 1
        return result

    df[f'{prefix}_periods_since_nonzero'] = df.groupby(group_cols)[target_col].transform(
        periods_since_nonzero_shifted
    )

    # Demand frequency in windows (using lagged demand_occurred, with additional shift)
    for window in windows:
        col_name = f'{prefix}_demand_freq_{window}w'
        # Rolling mean of whether demand occurred, shifted by 1 to not include current
        df[col_name] = df.groupby(group_cols)[f'{prefix}_lag1_demand_occurred'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # Average non-zero demand (expanding, using shift(1))
    df[f'{prefix}_avg_nonzero'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).expanding().mean()
    )

    # Std of non-zero demand (using shift(1))
    df[f'{prefix}_std_nonzero'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).expanding().std()
    )

    # Last non-zero demand value (using shift(1))
    df[f'{prefix}_last_nonzero'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).ffill()
    )

    # ADI (Average Demand Interval) - expanding (using shift(1))
    def compute_adi_shifted(series):
        """Compute ADI using shift(1) to avoid current value."""
        shifted = series.shift(1)
        result = []
        for i in range(len(shifted)):
            if i < 1:
                result.append(np.nan)
                continue
            sub = shifted.iloc[:i+1].dropna()
            non_zero_idx = np.where(sub.values > 0)[0]
            if len(non_zero_idx) <= 1:
                result.append(len(sub) if len(non_zero_idx) == 0 else 1)
            else:
                result.append(np.mean(np.diff(non_zero_idx)))
        return result

    df[f'{prefix}_adi_expanding'] = df.groupby(group_cols)[target_col].transform(
        compute_adi_shifted
    )

    return df


def create_sparsity_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    windows: List[int] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create advanced sparsity features for intermittent/lumpy demand modeling.

    These features are specifically designed for sparse demand patterns where
    demand occurs sporadically with variable quantities.

    Features Created:
    - weeks_since_last_demand: Periods since last non-zero demand (rolling)
    - demand_occurrence_prob: Probability of demand occurring (rolling by window)
    - typical_order_size: Median/mean of non-zero orders (robust to outliers)
    - demand_streak: Consecutive periods with/without demand
    - order_size_volatility: CV of non-zero order sizes
    - demand_recency_weighted: Exponentially weighted recency of demand

    CRITICAL RULE: No feature uses the current row's target value.
    All features use shift(1) minimum to avoid leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must be sorted by date within groups)
    group_cols : str or List[str]
        Columns to group by (key columns)
    target_col : str
        Target column name
    windows : List[int]
        Windows for rolling calculations (default: [4, 8, 13, 26])
    prefix : str, optional
        Prefix for column names (defaults to target_col)

    Returns
    -------
    pd.DataFrame
        DataFrame with sparsity features added

    Example
    -------
    >>> df = create_sparsity_features(df, ['key'], 'sales', windows=[4, 13, 26])
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    windows = windows or [4, 8, 13, 26]

    # =========================================================================
    # 1. WEEKS SINCE LAST DEMAND (rolling, per window)
    # =========================================================================
    # First compute the cumulative periods since last nonzero (already done in intermittency)
    # But we'll add window-specific versions that cap at window size

    def periods_since_nonzero_capped(series, cap):
        """Count periods since last non-zero demand, capped at window size."""
        shifted = series.shift(1)
        result = []
        count = 0
        for val in shifted:
            if pd.isna(val):
                result.append(np.nan)
            elif val > 0:
                result.append(min(count, cap))
                count = 0
            else:
                count += 1
                result.append(min(count, cap))
        return result

    for window in windows:
        col_name = f'{prefix}_weeks_since_demand_{window}w'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x, w=window: periods_since_nonzero_capped(x, w)
        )

    # =========================================================================
    # 2. DEMAND OCCURRENCE PROBABILITY (rolling by window)
    # =========================================================================
    # Binary indicator of demand (shifted to avoid leakage)
    demand_binary_col = f'{prefix}_demand_binary_shifted'
    df[demand_binary_col] = (df.groupby(group_cols)[target_col].shift(1) > 0).astype(float)

    for window in windows:
        col_name = f'{prefix}_demand_prob_{window}w'
        df[col_name] = df.groupby(group_cols)[demand_binary_col].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )

    # Clean up temp column
    df = df.drop(columns=[demand_binary_col])

    # =========================================================================
    # 3. TYPICAL ORDER SIZE (multiple robust measures)
    # =========================================================================
    # Median of non-zero orders (more robust than mean)
    df[f'{prefix}_typical_order_median'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).expanding(min_periods=1).median()
    )

    # Mean of non-zero orders (for comparison)
    df[f'{prefix}_typical_order_mean'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).expanding(min_periods=1).mean()
    )

    # Rolling typical order size (more recent-weighted)
    for window in [13, 26]:
        col_name = f'{prefix}_typical_order_{window}w'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x, w=window: x.shift(1).where(x.shift(1) > 0).rolling(w, min_periods=1).median()
        )

    # =========================================================================
    # 4. ORDER SIZE VOLATILITY (CV of non-zero orders)
    # =========================================================================
    # Standard deviation of non-zero orders
    nonzero_std = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).expanding(min_periods=2).std()
    )

    nonzero_mean = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).expanding(min_periods=1).mean()
    )

    df[f'{prefix}_order_size_cv'] = nonzero_std / (nonzero_mean + 1e-10)
    df[f'{prefix}_order_size_cv'] = df[f'{prefix}_order_size_cv'].clip(0, 10)

    # =========================================================================
    # 5. DEMAND STREAK FEATURES
    # =========================================================================
    def compute_demand_streak(series):
        """Compute consecutive demand/no-demand streak length."""
        shifted = series.shift(1)
        result = []
        streak = 0
        last_state = None  # None, 'demand', 'no_demand'

        for val in shifted:
            if pd.isna(val):
                result.append(0)
                continue

            current_state = 'demand' if val > 0 else 'no_demand'

            if last_state is None:
                streak = 1
            elif current_state == last_state:
                streak += 1
            else:
                streak = 1

            # Positive for demand streaks, negative for no-demand streaks
            result.append(streak if current_state == 'demand' else -streak)
            last_state = current_state

        return result

    df[f'{prefix}_demand_streak'] = df.groupby(group_cols)[target_col].transform(
        compute_demand_streak
    )

    # Separate positive (demand) and negative (no-demand) streaks
    df[f'{prefix}_demand_run_length'] = df[f'{prefix}_demand_streak'].clip(lower=0)
    df[f'{prefix}_no_demand_run_length'] = (-df[f'{prefix}_demand_streak']).clip(lower=0)

    # =========================================================================
    # 6. DEMAND RECENCY WEIGHTED AVERAGE
    # =========================================================================
    # Exponentially weighted average of demand (more weight on recent periods)
    for span in [4, 13]:
        col_name = f'{prefix}_demand_ewm_{span}'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x, s=span: x.shift(1).ewm(span=s, adjust=False, min_periods=1).mean()
        )

    # =========================================================================
    # 7. DEMAND CONSISTENCY SCORE
    # =========================================================================
    # How consistently does demand occur? (low variance in inter-demand intervals)
    def compute_demand_consistency(series, window=26):
        """Compute consistency of demand intervals within a rolling window."""
        shifted = series.shift(1)
        result = []

        for i in range(len(shifted)):
            if i < 2:
                result.append(0.5)  # Default neutral score
                continue

            # Look at last 'window' periods
            start_idx = max(0, i - window)
            sub = shifted.iloc[start_idx:i+1].values

            # Find non-zero indices
            nonzero_idx = np.where(sub > 0)[0]

            if len(nonzero_idx) < 2:
                # Not enough demand events to measure consistency
                result.append(0.0 if len(nonzero_idx) == 0 else 0.5)
            else:
                # Compute CV of intervals
                intervals = np.diff(nonzero_idx)
                if len(intervals) > 0:
                    interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                    # Convert to 0-1 score (lower CV = higher consistency)
                    consistency = 1.0 / (1.0 + interval_cv)
                    result.append(consistency)
                else:
                    result.append(0.5)

        return result

    df[f'{prefix}_demand_consistency'] = df.groupby(group_cols)[target_col].transform(
        compute_demand_consistency
    )

    # =========================================================================
    # 8. FILL MISSING VALUES
    # =========================================================================
    # Fill NaN values with sensible defaults
    sparsity_cols = [col for col in df.columns if prefix in col and any(
        feat in col for feat in ['weeks_since', 'demand_prob', 'typical_order',
                                  'order_size_cv', 'demand_streak', 'run_length',
                                  'demand_ewm', 'demand_consistency']
    )]

    for col in sparsity_cols:
        if 'prob' in col or 'consistency' in col:
            df[col] = df[col].fillna(0.5)  # Neutral probability
        elif 'cv' in col:
            df[col] = df[col].fillna(1.0)  # Default CV
        elif 'streak' in col or 'run_length' in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(0)

    return df


# =============================================================================
# 6. CROSS-SECTIONAL FEATURES
# =============================================================================

def create_cross_sectional_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    date_col: str,
    segment_col: Optional[str] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create cross-sectional features (relative to peers).

    State-of-the-art: Captures how a key performs relative to its segment/overall.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Key columns
    target_col : str
        Target column
    date_col : str
        Date column
    segment_col : str, optional
        Segment column for segment-relative features
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with cross-sectional features

    Example
    -------
    >>> df = create_cross_sectional_features(df, ['key'], 'sales', 'date', 'segment_id')
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    # Overall statistics per time period
    period_stats = df.groupby(date_col)[target_col].agg(['mean', 'std', 'median']).reset_index()
    period_stats.columns = [date_col, f'{prefix}_period_mean', f'{prefix}_period_std', f'{prefix}_period_median']

    # Drop existing columns to avoid duplicates on merge
    cols_to_add = [c for c in period_stats.columns if c != date_col]
    existing_cols = [c for c in cols_to_add if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    df = df.merge(period_stats, on=date_col, how='left')

    # Relative to overall mean
    df[f'{prefix}_rel_to_period_mean'] = df[target_col] / (df[f'{prefix}_period_mean'] + 1e-10)
    df[f'{prefix}_diff_from_period_mean'] = df[target_col] - df[f'{prefix}_period_mean']

    # Z-score relative to period
    df[f'{prefix}_zscore_period'] = (
        (df[target_col] - df[f'{prefix}_period_mean']) /
        (df[f'{prefix}_period_std'] + 1e-10)
    )

    # Segment-relative features
    if segment_col and segment_col in df.columns:
        segment_period_stats = df.groupby([segment_col, date_col])[target_col].agg(['mean', 'std']).reset_index()
        segment_period_stats.columns = [segment_col, date_col, f'{prefix}_seg_period_mean', f'{prefix}_seg_period_std']

        # Drop existing columns to avoid duplicates on merge
        seg_cols_to_add = [f'{prefix}_seg_period_mean', f'{prefix}_seg_period_std']
        existing_seg_cols = [c for c in seg_cols_to_add if c in df.columns]
        if existing_seg_cols:
            df = df.drop(columns=existing_seg_cols)

        df = df.merge(segment_period_stats, on=[segment_col, date_col], how='left')

        df[f'{prefix}_rel_to_segment'] = df[target_col] / (df[f'{prefix}_seg_period_mean'] + 1e-10)
        df[f'{prefix}_diff_from_segment'] = df[target_col] - df[f'{prefix}_seg_period_mean']
        df[f'{prefix}_zscore_segment'] = (
            (df[target_col] - df[f'{prefix}_seg_period_mean']) /
            (df[f'{prefix}_seg_period_std'] + 1e-10)
        )

    # Rank within period
    df[f'{prefix}_rank_in_period'] = df.groupby(date_col)[target_col].rank(method='dense', pct=True)

    return df


def create_key_level_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create key-level (static) aggregate features.

    State-of-the-art: Captures lifetime characteristics of each key.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Key columns
    target_col : str
        Target column
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with key-level features merged back

    Example
    -------
    >>> df = create_key_level_features(df, ['key'], 'sales')
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    # Compute key-level statistics
    key_stats = df.groupby(group_cols)[target_col].agg([
        ('key_mean', 'mean'),
        ('key_std', 'std'),
        ('key_min', 'min'),
        ('key_max', 'max'),
        ('key_sum', 'sum'),
        ('key_count', 'count'),
        ('key_nonzero_count', lambda x: (x > 0).sum()),
    ]).reset_index()

    # Compute CV and zero fraction
    key_stats['key_cv'] = key_stats['key_std'] / (key_stats['key_mean'] + 1e-10)
    key_stats['key_zero_fraction'] = 1 - (key_stats['key_nonzero_count'] / key_stats['key_count'])

    # Rename columns with prefix
    rename_dict = {col: f'{prefix}_{col}' for col in key_stats.columns if col not in group_cols}
    key_stats = key_stats.rename(columns=rename_dict)

    # Drop existing columns to avoid duplicates on merge
    key_cols_to_add = [c for c in key_stats.columns if c not in group_cols]
    existing_key_cols = [c for c in key_cols_to_add if c in df.columns]
    if existing_key_cols:
        df = df.drop(columns=existing_key_cols)

    # Merge back
    df = df.merge(key_stats, on=group_cols, how='left')

    return df


# =============================================================================
# 7. CATEGORICAL ENCODING
# =============================================================================

def target_encode(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    alpha: float = 10.0,
    fit_data: Optional[pd.DataFrame] = None,
    encoding_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Apply target encoding with smoothing.

    State-of-the-art: Regularized target encoding prevents overfitting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to encode
    target_col : str
        Target column
    alpha : float
        Smoothing parameter (higher = more regularization)
    fit_data : pd.DataFrame, optional
        Data to fit encoding on (for train/inference split)
    encoding_map : Dict, optional
        Pre-computed encoding map (for inference)

    Returns
    -------
    Tuple[pd.Series, Dict[str, float]]
        Encoded series and encoding map

    Example
    -------
    >>> encoded, mapping = target_encode(df, 'category', 'sales')
    """
    if encoding_map is not None:
        # Apply existing encoding (inference mode)
        global_mean = encoding_map.get('__global_mean__', 0)
        encoded = df[col].map(encoding_map).fillna(global_mean)
        return encoded, encoding_map

    # Fit encoding (training mode)
    fit_df = fit_data if fit_data is not None else df
    global_mean = fit_df[target_col].mean()

    agg = fit_df.groupby(col)[target_col].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)

    encoding_map = smooth.to_dict()
    encoding_map['__global_mean__'] = global_mean

    encoded = df[col].map(smooth).fillna(global_mean)

    return encoded, encoding_map


def frequency_encode(
    df: pd.DataFrame,
    col: str,
    fit_data: Optional[pd.DataFrame] = None,
    encoding_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Apply frequency encoding (count-based).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to encode
    fit_data : pd.DataFrame, optional
        Data to fit encoding on
    encoding_map : Dict, optional
        Pre-computed encoding map

    Returns
    -------
    Tuple[pd.Series, Dict[str, float]]
        Encoded series and encoding map

    Example
    -------
    >>> encoded, mapping = frequency_encode(df, 'category')
    """
    if encoding_map is not None:
        encoded = df[col].map(encoding_map).fillna(0)
        return encoded, encoding_map

    fit_df = fit_data if fit_data is not None else df
    freq = fit_df[col].value_counts(normalize=True)
    encoding_map = freq.to_dict()

    encoded = df[col].map(encoding_map).fillna(0)

    return encoded, encoding_map


def ordinal_encode(
    df: pd.DataFrame,
    col: str,
    order_map: Optional[Dict[str, int]] = None,
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Apply ordinal encoding with optional explicit ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to encode
    order_map : Dict[str, int], optional
        Explicit ordering. If None, uses alphabetical order.

    Returns
    -------
    Tuple[pd.Series, Dict[str, int]]
        Encoded series and encoding map

    Example
    -------
    >>> encoded, mapping = ordinal_encode(df, 'tier', {'A': 4, 'B': 3, 'C': 2, 'D': 1})
    """
    if order_map is None:
        unique_vals = sorted(df[col].dropna().unique())
        order_map = {v: i for i, v in enumerate(unique_vals)}

    encoded = df[col].map(order_map).fillna(-1).astype(int)

    return encoded, order_map


def hash_encode(
    df: pd.DataFrame,
    col: str,
    n_components: int = 8,
) -> pd.DataFrame:
    """
    Apply hash encoding for high-cardinality categoricals.

    State-of-the-art: Fixed-size encoding regardless of cardinality.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to encode
    n_components : int
        Number of hash buckets

    Returns
    -------
    pd.DataFrame
        DataFrame with hash-encoded columns

    Example
    -------
    >>> df = hash_encode(df, 'product_id', n_components=8)
    """
    df = df.copy()

    for i in range(n_components):
        df[f'{col}_hash_{i}'] = df[col].apply(
            lambda x: (hash(str(x) + str(i)) % n_components) == i
        ).astype(int)

    return df


def encode_all_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    encoding_strategy: Dict[str, str] = None,
    fit_data: Optional[pd.DataFrame] = None,
    existing_specs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode all categorical columns using appropriate strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    categorical_cols : List[str]
        List of categorical columns
    target_col : str
        Target column (for target encoding)
    encoding_strategy : Dict[str, str], optional
        Strategy per column: 'target', 'frequency', 'ordinal', 'onehot', 'hash'
        Default: auto-detect based on cardinality
    fit_data : pd.DataFrame, optional
        Data to fit encodings on
    existing_specs : Dict, optional
        Pre-computed encoding specs (for inference)

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Encoded DataFrame and encoding specifications

    Example
    -------
    >>> df, specs = encode_all_categoricals(df, ['cat1', 'cat2'], 'sales')
    """
    df = df.copy()
    encoding_specs = existing_specs or {
        'target_encoding_maps': {},
        'frequency_encoding_maps': {},
        'ordinal_encoding_maps': {},
        'onehot_columns': [],
        'hash_columns': [],
        'binary_columns': [],
    }

    encoding_strategy = encoding_strategy or {}

    # Track strategy used for each column (for consistency across train/val/test)
    if 'column_strategies' not in encoding_specs:
        encoding_specs['column_strategies'] = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        # Use existing strategy if available (for val/test consistency with training)
        existing_strategy = encoding_specs['column_strategies'].get(col)

        if existing_strategy is not None:
            strategy = existing_strategy
        else:
            # Determine strategy from data (training phase)
            n_unique = df[col].nunique()

            if col in encoding_strategy:
                strategy = encoding_strategy[col]
            elif n_unique <= 2:
                strategy = 'binary'
            elif n_unique <= 10:
                strategy = 'onehot'
            elif n_unique <= 50:
                strategy = 'target'
            else:
                strategy = 'hash'

            # Store strategy for later use on val/test
            encoding_specs['column_strategies'][col] = strategy

        if strategy == 'binary':
            # Handle binary columns properly - they may be string categories, not just 0/1
            col_dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(col_dtype):
                # Already numeric - just fill NaN and convert to int
                df[col] = df[col].fillna(0).astype(int)
            else:
                # String/object categorical - need label encoding for binary values
                # Check for existing label map (for val/test consistency)
                if 'label_encoding_maps' not in encoding_specs:
                    encoding_specs['label_encoding_maps'] = {}

                existing_map = encoding_specs['label_encoding_maps'].get(col)

                if existing_map is not None:
                    # APPLY EXISTING SPEC: Use same mapping as training
                    df[col] = df[col].map(existing_map).fillna(0).astype(int)
                else:
                    # FIT: Create mapping from current data
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 2:
                        # Create a mapping: first unique value -> 0, second -> 1
                        label_map = {val: idx for idx, val in enumerate(sorted(unique_vals, key=str))}
                        encoding_specs['label_encoding_maps'][col] = label_map
                        # Apply the mapping
                        df[col] = df[col].map(label_map).fillna(0).astype(int)
                    else:
                        # Fallback: use frequency encoding if somehow more than 2 values
                        encoded, mapping = frequency_encode(df, col, fit_data=fit_data)
                        df[col] = encoded.fillna(0).astype(int)
                        encoding_specs['frequency_encoding_maps'][col] = mapping
            if col not in encoding_specs['binary_columns']:
                encoding_specs['binary_columns'].append(col)

        elif strategy == 'onehot':
            # One-hot encoding with FIT on training, APPLY on val/test
            # Key principle: Training defines the columns, val/test gets same columns (with 0s for missing categories)
            if 'onehot_column_mapping' not in encoding_specs:
                encoding_specs['onehot_column_mapping'] = {}
            if 'onehot_categories' not in encoding_specs:
                encoding_specs['onehot_categories'] = {}

            existing_categories = encoding_specs['onehot_categories'].get(col)

            if existing_categories is not None:
                # APPLY (val/test): Create indicator columns matching training exactly
                # Use the stored categories to create consistent dummy columns
                # Convert column to categorical with the training categories
                cat_series = pd.Categorical(df[col], categories=existing_categories)
                dummies = pd.get_dummies(cat_series, prefix=col, drop_first=True)
                # int8 downcast: pd.get_dummies returns bool/uint8 in modern
                # pandas but historically defaulted to float64, eating 8 bytes
                # per binary indicator. Force int8 explicitly so a one-hot
                # column with ~1000 categories takes 1 GB instead of 8 GB
                # on a 1M-row frame. LightGBM accepts int8 natively.
                dummies = dummies.astype('int8')
                # This automatically creates all columns from training categories
                # Values not in training categories become all 0s (no indicator set)
                df = pd.concat([df, dummies], axis=1)
                dummies_cols = list(dummies.columns)
            else:
                # FIT (training): Get unique categories, create dummies, and store
                # Get categories (excluding NaN), sorted for consistency
                categories = sorted([c for c in df[col].dropna().unique()], key=str)
                encoding_specs['onehot_categories'][col] = categories

                # Create dummies using these categories
                cat_series = pd.Categorical(df[col], categories=categories)
                dummies = pd.get_dummies(cat_series, prefix=col, drop_first=True)
                # int8 downcast — see comment above. Saves up to 8x memory
                # on the dummies frame versus the float64 default.
                dummies = dummies.astype('int8')
                encoding_specs['onehot_column_mapping'][col] = list(dummies.columns)
                df = pd.concat([df, dummies], axis=1)
                dummies_cols = list(dummies.columns)

            # Track all onehot columns created (for backward compatibility)
            for dummy_col in dummies_cols:
                if dummy_col not in encoding_specs['onehot_columns']:
                    encoding_specs['onehot_columns'].append(dummy_col)

        elif strategy == 'target':
            existing_map = encoding_specs['target_encoding_maps'].get(col)
            encoded, mapping = target_encode(df, col, target_col, fit_data=fit_data, encoding_map=existing_map)
            df[f'{col}_encoded'] = encoded
            encoding_specs['target_encoding_maps'][col] = mapping

        elif strategy == 'frequency':
            existing_map = encoding_specs['frequency_encoding_maps'].get(col)
            encoded, mapping = frequency_encode(df, col, fit_data=fit_data, encoding_map=existing_map)
            df[f'{col}_freq'] = encoded
            encoding_specs['frequency_encoding_maps'][col] = mapping

        elif strategy == 'hash':
            df = hash_encode(df, col, n_components=8)
            encoding_specs['hash_columns'].append(col)

    return df, encoding_specs


# =============================================================================
# 8. FEATURE SELECTION
# =============================================================================

def select_features_by_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_features: int = 50,
    correlation_threshold: float = 0.95,
) -> List[str]:
    """
    Select features based on correlation with target and inter-feature correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        List of feature columns to consider
    target_col : str
        Target column
    max_features : int
        Maximum features to select
    correlation_threshold : float
        Threshold for dropping highly correlated features

    Returns
    -------
    List[str]
        List of selected feature names

    Example
    -------
    >>> selected = select_features_by_correlation(df, feature_cols, 'sales', max_features=30)
    """
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return []

    # Compute correlation with target
    target_corr = df[feature_cols].corrwith(df[target_col]).abs()
    target_corr = target_corr.dropna().sort_values(ascending=False)

    # Select top features by target correlation
    selected = list(target_corr.head(max_features * 2).index)

    # Remove highly correlated features (keep the one with higher target correlation)
    if len(selected) > 1:
        corr_matrix = df[selected].corr().abs()
        to_drop = set()

        for i in range(len(selected)):
            if selected[i] in to_drop:
                continue
            for j in range(i + 1, len(selected)):
                if selected[j] in to_drop:
                    continue
                if corr_matrix.loc[selected[i], selected[j]] > correlation_threshold:
                    # Drop the one with lower target correlation
                    if target_corr[selected[i]] >= target_corr[selected[j]]:
                        to_drop.add(selected[j])
                    else:
                        to_drop.add(selected[i])

        selected = [f for f in selected if f not in to_drop]

    return selected[:max_features]


def select_features_by_variance(
    df: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float = 0.01,
) -> List[str]:
    """
    Select features with sufficient variance.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        List of feature columns
    variance_threshold : float
        Minimum variance threshold

    Returns
    -------
    List[str]
        List of selected feature names

    Example
    -------
    >>> selected = select_features_by_variance(df, feature_cols)
    """
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return []

    variances = df[feature_cols].var()
    selected = variances[variances > variance_threshold].index.tolist()

    return selected


def select_features_shap(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 50,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features using SHAP importance.

    State-of-the-art: Model-agnostic feature importance.

    Parameters
    ----------
    model : trained model
        Trained model object
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target
    max_features : int
        Maximum features to select

    Returns
    -------
    Tuple[List[str], pd.DataFrame]
        Selected feature names and importance DataFrame

    Example
    -------
    >>> selected, importance = select_features_shap(model, X_train, y_train)
    """
    try:
        import shap

        # Create explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X)

        # Compute SHAP values (sample if too large)
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Compute mean absolute SHAP
        mean_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_shap,
        }).sort_values('importance', ascending=False)

        selected = importance_df.head(max_features)['feature'].tolist()

        return selected, importance_df

    except ImportError:
        logger.warning("SHAP not available. Using correlation-based selection.")
        selected = select_features_by_correlation(
            pd.concat([X, y], axis=1),
            list(X.columns),
            y.name,
            max_features=max_features
        )
        return selected, pd.DataFrame()


def auto_select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_features: int = 50,
    method: str = 'correlation',
    model = None,
) -> List[str]:
    """
    Automatically select best features using specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        List of feature columns
    target_col : str
        Target column
    max_features : int
        Maximum features to select
    method : str
        Selection method: 'correlation', 'variance', 'shap'
    model : trained model, optional
        Model for SHAP-based selection

    Returns
    -------
    List[str]
        List of selected feature names

    Example
    -------
    >>> selected = auto_select_features(df, feature_cols, 'sales', method='correlation')
    """
    if method == 'variance':
        return select_features_by_variance(df, feature_cols)

    elif method == 'shap' and model is not None:
        selected, _ = select_features_shap(model, df[feature_cols], df[target_col], max_features)
        return selected

    else:  # correlation (default)
        return select_features_by_correlation(df, feature_cols, target_col, max_features)


# =============================================================================
# 9. DATA QUALITY & VALIDATION
# =============================================================================

def check_feature_quality(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Check data quality of features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Feature columns to check

    Returns
    -------
    Dict[str, Any]
        Quality report

    Example
    -------
    >>> quality = check_feature_quality(df, feature_cols)
    """
    feature_cols = [c for c in feature_cols if c in df.columns]

    report = {
        'n_features': len(feature_cols),
        'n_rows': len(df),
        'missing_pct': {},
        'zero_pct': {},
        'inf_count': {},
        'constant_features': [],
        'high_missing_features': [],
    }

    for col in feature_cols:
        # Missing percentage
        missing_pct = df[col].isna().mean()
        report['missing_pct'][col] = round(missing_pct, 4)

        # Zero percentage
        zero_pct = (df[col] == 0).mean()
        report['zero_pct'][col] = round(zero_pct, 4)

        # Inf count
        if df[col].dtype in [np.float64, np.float32]:
            inf_count = np.isinf(df[col]).sum()
            report['inf_count'][col] = int(inf_count)

        # Constant check
        if df[col].nunique() <= 1:
            report['constant_features'].append(col)

        # High missing
        if missing_pct > 0.5:
            report['high_missing_features'].append(col)

    return report


def remove_highly_correlated_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    correlation_threshold: float = 0.95,
    prefer_simpler: bool = True,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Remove highly correlated features to reduce redundancy.

    When two features have correlation >= threshold, one is dropped.
    By default, we prefer simpler features (e.g., rolling_mean over ewm).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Feature columns to check
    correlation_threshold : float
        Correlation threshold above which features are considered redundant
    prefer_simpler : bool
        If True, prefer simpler features (e.g., drop ewm if roll exists)

    Returns
    -------
    Tuple[List[str], List[Tuple[str, str, float]]]
        - Filtered feature columns
        - List of dropped pairs with correlation values

    Example
    -------
    >>> filtered_cols, dropped = remove_highly_correlated_features(df, cols, 0.95)
    """
    numeric_cols = [c for c in feature_cols if c in df.columns and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    if len(numeric_cols) < 2:
        return feature_cols, []

    # Compute correlation matrix
    try:
        corr_matrix = df[numeric_cols].corr().abs()
    except Exception as e:
        logger.warning(f"Could not compute correlation matrix: {e}")
        return feature_cols, []

    # Find highly correlated pairs
    dropped_pairs = []
    to_drop = set()

    # Priority for keeping features (lower = keep, higher = drop)
    # Prefer simpler features: rolling > ewm > expanding
    def feature_priority(col: str) -> int:
        col_lower = col.lower()
        if '_ewm_' in col_lower:
            return 3  # Higher priority to drop
        if '_expanding_' in col_lower:
            return 2
        if '_roll_' in col_lower or '_rolling_' in col_lower:
            return 1  # Prefer to keep
        return 0  # Keep by default

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            if col1 in to_drop or col2 in to_drop:
                continue

            corr_val = corr_matrix.iloc[i, j]
            if corr_val >= correlation_threshold:
                # Decide which to drop
                if prefer_simpler:
                    p1, p2 = feature_priority(col1), feature_priority(col2)
                    if p1 > p2:
                        drop_col = col1
                    elif p2 > p1:
                        drop_col = col2
                    else:
                        # Same priority, drop the one that appears later
                        drop_col = col2
                else:
                    # Drop the second one
                    drop_col = col2

                to_drop.add(drop_col)
                dropped_pairs.append((col1, col2, round(corr_val, 4)))

    filtered_cols = [c for c in feature_cols if c not in to_drop]

    if dropped_pairs:
        logger.info(f"Removed {len(to_drop)} highly correlated features (r >= {correlation_threshold})")
        for c1, c2, r in dropped_pairs[:5]:  # Log first 5
            logger.debug(f"  Dropped due to correlation: {c1} ~ {c2} (r={r})")

    return filtered_cols, dropped_pairs


def validate_no_target_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, Any]:
    """
    Validate that no features appear to use the current target value.

    CRITICAL CHECK: This function identifies potential leakage by detecting:
    1. Features named after target without 'lag', 'roll', 'ewm', 'expanding' keywords
    2. Features with perfect or near-perfect correlation with target
    3. Features that contain the exact target column name without transformation

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Feature columns to check
    target_col : str
        Target column name

    Returns
    -------
    Dict[str, Any]
        Validation report with potential leakage warnings
    """
    report = {
        'potential_leaking_features': [],
        'high_correlation_features': [],
        'warnings': [],
        'is_valid': True,
    }

    # Patterns that indicate safe features (use historical data)
    safe_patterns = ['lag', 'roll', 'ewm', 'expanding', 'shift', 'seasonal_lag',
                     'period_mean', 'period_std', 'seg_period', 'key_mean', 'key_std',
                     'key_min', 'key_max', 'key_cv', 'key_count', 'key_nonzero',
                     'key_zero_fraction', 'demand_freq',
                     # Sparsity pattern features (computed from historical lags)
                     'demand_prob', 'typical_order', 'demand_streak',
                     'demand_consistency', 'demand_ewm',
                     'weeks_since', 'order_size_cv', 'run_length',
                     'zero_run', 'nonzero_ratio', 'intermittent',
                     'periods_since', 'avg_nonzero', 'std_nonzero', 'last_nonzero',
                     'adi_expanding', 'encoded', '_sin', '_cos', 'fourier',
                     'week_of_year', 'month', 'quarter', 'year', 'is_holiday',
                     # Phase 3: Hierarchy temporal features (all use shift(forecast_lag))
                     'group_total', 'group_mean', 'share_lag', 'rank_in_group',
                     'group_trend', 'rel_to_group',
                     # Phase 6: Regime, relative, and history embedding features
                     'regime_', 'rel_to_', 'rel_share_', 'rank_in_', 'rank_overall',
                     'dist_from_', 'relative_momentum', '_hist_mean', '_hist_std',
                     '_hist_median', '_target_corr', '_hist_trend']

    # Patterns that indicate UNSAFE features (use current target)
    unsafe_patterns = [
        f'{target_col}_log',
        f'{target_col}_demand_occurred',  # Without 'lag' prefix
        f'{target_col}_zscore_period',
        f'{target_col}_rel_to_period_mean',
        f'{target_col}_diff_from_period_mean',
        f'{target_col}_rank_in_period',
        f'{target_col}_rel_to_segment',
        f'{target_col}_diff_from_segment',
        f'{target_col}_zscore_segment',
        f'{target_col}_yoy_diff',  # Without 'lag'
        f'{target_col}_yoy_ratio',
        f'{target_col}_qoq_diff',
    ]

    # Also check for diff/pct_change without 'lag' prefix
    for period in [1, 3, 4, 6, 12, 13, 52]:
        unsafe_patterns.extend([
            f'{target_col}_diff_{period}',
            f'{target_col}_pct_change_{period}',
            f'{target_col}_direction_{period}',
        ])

    for col in feature_cols:
        col_lower = col.lower()
        target_lower = target_col.lower()

        # Check against explicit unsafe patterns
        if col in unsafe_patterns:
            report['potential_leaking_features'].append(col)
            report['warnings'].append(f"LEAKAGE: '{col}' uses current target value")
            report['is_valid'] = False
            continue

        # Check if feature contains target name but no safe pattern
        if target_lower in col_lower:
            has_safe_pattern = any(pattern in col_lower for pattern in safe_patterns)
            if not has_safe_pattern:
                report['potential_leaking_features'].append(col)
                report['warnings'].append(
                    f"POTENTIAL LEAKAGE: '{col}' contains target name but no safe pattern"
                )

    # Check for high correlation with target (potential leakage indicator)
    if target_col in df.columns:
        for col in feature_cols:
            if col in df.columns and col != target_col:
                try:
                    corr = df[col].corr(df[target_col])
                    if abs(corr) > 0.95:  # Very high correlation
                        report['high_correlation_features'].append({
                            'feature': col,
                            'correlation': round(corr, 4),
                        })
                        report['warnings'].append(
                            f"HIGH CORRELATION: '{col}' has {corr:.4f} correlation with target"
                        )
                except Exception:
                    pass  # Skip if correlation can't be computed

    if report['potential_leaking_features']:
        logger.warning(f"LEAKAGE DETECTED: {len(report['potential_leaking_features'])} features may use current target")
        for feat in report['potential_leaking_features']:
            logger.warning(f"  - {feat}")

    return report


# ─────────────────────────────────────────────────────────────────────
# fill_missing_features — helpers
# ─────────────────────────────────────────────────────────────────────
# The pre-existing implementation assumed columns were either numeric
# or generic-object.  After `_optimize_dtypes_inplace` in diq_runner.py
# started converting low-cardinality strings to `pd.Categorical` for
# memory, the old `fillna(mode_val.iloc[0] if … else 0)` pattern would
# crash with:
#
#   TypeError: Cannot setitem on a Categorical with a new category (0),
#              set the categories first
#
# on every CONDIMENT-style run where a Categorical feature column
# happened to be all-NaN in test_df.  The two helpers below replace the
# unsafe pattern with dtype-aware, gracefully-degrading equivalents.

def _pick_fill_value(col_data: pd.Series, numeric_strategy: str = "median"):
    """Pick a fill value that's *compatible with the column's dtype*.

    Returns ``np.nan`` when no sensible value can be picked — callers
    should leave the column as-is in that case (LightGBM / XGBoost
    handle NaN natively, so leaving NaN is safe downstream).
    """
    dtype = col_data.dtype

    # Categorical — must be one of the existing categories
    if isinstance(dtype, pd.CategoricalDtype):
        mode = col_data.mode()
        if len(mode) > 0:
            return mode.iloc[0]
        cats = col_data.cat.categories
        if len(cats) > 0:
            return cats[0]
        return np.nan

    # Datetime — first non-null, else NaT
    if pd.api.types.is_datetime64_any_dtype(dtype):
        non_null = col_data.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else pd.NaT

    # Bool — mode, else False
    if pd.api.types.is_bool_dtype(dtype):
        mode = col_data.mode()
        return bool(mode.iloc[0]) if len(mode) > 0 else False

    # Numeric — strategy-driven (median/mean), else 0
    if pd.api.types.is_numeric_dtype(dtype):
        v = col_data.median() if numeric_strategy == "median" else col_data.mean()
        return v if pd.notna(v) else 0

    # Object / string / unknown — mode, else first non-null, else ""
    mode = col_data.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    non_null = col_data.dropna()
    if len(non_null) > 0:
        return non_null.iloc[0]
    return ""


def _safe_fillna(df: pd.DataFrame, col: str, value) -> bool:
    """In-place safe fillna on ``df[col]``.

    Handles ``pd.Categorical`` columns gracefully: if ``value`` isn't
    already in the categories, attempts to add it; if that fails,
    coerces the column to ``object`` dtype (sacrificing memory for
    that one column rather than crashing the pipeline); if even that
    fails, leaves NaN and logs a warning.

    Returns ``True`` if any fill actually happened, ``False`` if the
    column was left untouched.
    """
    col_data = df[col]
    if isinstance(col_data.dtype, pd.CategoricalDtype):
        cats = col_data.cat.categories
        if value in cats:
            df[col] = col_data.fillna(value)
            return True
        try:
            df[col] = col_data.cat.add_categories([value]).fillna(value)
            return True
        except (TypeError, ValueError):
            try:
                logger.debug(
                    f"_safe_fillna: {col!r} → coercing Categorical to "
                    f"object to accept fill value {value!r}"
                )
                df[col] = col_data.astype("object").fillna(value)
                return True
            except Exception as e:
                logger.warning(
                    f"_safe_fillna: could not fill Categorical column "
                    f"{col!r} with {value!r}: {e}.  Leaving NaN."
                )
                return False

    try:
        df[col] = col_data.fillna(value)
        return True
    except Exception as e:
        logger.warning(
            f"_safe_fillna: fillna failed on {col!r}: {e}.  Leaving NaN."
        )
        return False


def fill_missing_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    strategy: str = 'median',
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fill missing values in features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Feature columns to fill
    strategy : str
        Fill strategy: 'median', 'mean', 'zero', 'ffill', 'group_median'
    group_cols : List[str], optional
        Columns for group-based filling

    Returns
    -------
    pd.DataFrame
        DataFrame with filled values

    Notes
    -----
    Dtype-aware: per-column fill values are chosen to be compatible
    with the column's dtype (see ``_pick_fill_value`` / ``_safe_fillna``
    above).  Categorical columns that are entirely NaN are LEFT NaN —
    they carry no signal, and downstream model code (LightGBM / XGBoost
    / direct multi-horizon) handles NaN natively.

    Example
    -------
    >>> df = fill_missing_features(df, feature_cols, strategy='median')
    """
    df = df.copy()

    # Handle duplicate column names - keep first occurrence only
    if df.columns.duplicated().any():
        logger.warning(f"Duplicate columns detected: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]

    feature_cols = [c for c in feature_cols if c in df.columns]

    for col in feature_cols:
        # Ensure we're working with a Series, not a DataFrame (handles edge cases)
        col_data = df[col]
        if isinstance(col_data, pd.DataFrame):
            # If duplicate columns exist, take the first one
            col_data = col_data.iloc[:, 0]

        if col_data.isna().sum() == 0:
            continue

        # All-NaN columns carry zero signal — leave NaN.  Downstream
        # model code handles NaN natively; trying to impute would just
        # introduce arbitrary noise (and trip the Categorical-dtype
        # safety net for no benefit).
        if col_data.isna().all():
            logger.debug(
                f"fill_missing_features: column {col!r} is all-NaN; "
                f"leaving as-is (no meaningful imputation)."
            )
            continue

        if strategy == 'zero':
            _safe_fillna(df, col, 0)

        elif strategy in ('mean', 'median'):
            value = _pick_fill_value(col_data, numeric_strategy=strategy)
            if pd.notna(value) or value == "" or value is False:
                # Note: pd.notna(False) is True; the explicit checks
                # above keep boolean / empty-string fills working.
                _safe_fillna(df, col, value)

        elif strategy == 'ffill':
            try:
                if group_cols:
                    df[col] = df.groupby(group_cols)[col].ffill()
                else:
                    df[col] = col_data.ffill()
            except Exception as e:
                logger.warning(
                    f"fill_missing_features: ffill failed on {col!r}: {e}"
                )

        elif strategy == 'group_median' and group_cols:
            if pd.api.types.is_numeric_dtype(col_data):
                df[col] = df.groupby(group_cols)[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill any remaining with global median (safe for numeric)
                global_median = df[col].median()
                if pd.notna(global_median):
                    _safe_fillna(df, col, global_median)
            else:
                # Non-numeric path: per-group mode; remaining NaN gets
                # the dtype-aware fallback below.  We don't pre-pick
                # `0` inside the transform because that lambda runs on
                # a Categorical and would explode the old way.
                def _group_fill(x):
                    mode = x.mode()
                    if len(mode) > 0:
                        return x.fillna(mode.iloc[0])
                    return x  # leave NaN; outer _safe_fillna handles
                df[col] = df.groupby(group_cols)[col].transform(_group_fill)
                if df[col].isna().any():
                    value = _pick_fill_value(df[col])
                    if pd.notna(value) or value == "" or value is False:
                        _safe_fillna(df, col, value)

    return df


def clip_outliers(
    df: pd.DataFrame,
    feature_cols: List[str],
    method: str = 'iqr',
    factor: float = 3.0,
) -> pd.DataFrame:
    """
    Clip outliers in features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Feature columns to clip
    method : str
        Outlier detection method: 'iqr', 'std', 'percentile'
    factor : float
        Factor for IQR/std method, or percentile value

    Returns
    -------
    pd.DataFrame
        DataFrame with clipped values

    Example
    -------
    >>> df = clip_outliers(df, feature_cols, method='iqr', factor=3.0)
    """
    df = df.copy()
    feature_cols = [c for c in feature_cols if c in df.columns]

    for col in feature_cols:
        if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            continue

        if method == 'iqr':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr

        elif method == 'std':
            mean = df[col].mean()
            std = df[col].std()
            lower = mean - factor * std
            upper = mean + factor * std

        elif method == 'percentile':
            lower = df[col].quantile(factor / 100)
            upper = df[col].quantile(1 - factor / 100)

        else:
            continue

        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


# =============================================================================
# 10. HIGH-LEVEL ORCHESTRATION
# =============================================================================

def create_complete_feature_set(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    date_col: str,
    target_col: str,
    segment_col: Optional[str] = None,
    demand_pattern: str = 'smooth',
    include_lags: bool = True,
    include_rolling: bool = True,
    include_ewm: bool = True,
    include_calendar: bool = True,
    include_fourier: bool = True,
    include_trend: bool = True,
    include_intermittency: bool = False,
    include_cross_sectional: bool = True,
    lag_periods: List[int] = None,
    rolling_windows: List[int] = None,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create a complete feature set with one function call.

    This is the MAIN function agents should use for feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    segment_col : str, optional
        Segment column
    demand_pattern : str
        Demand pattern: 'smooth', 'erratic', 'intermittent', 'lumpy'
    include_lags : bool
        Include lag features
    include_rolling : bool
        Include rolling features
    include_ewm : bool
        Include EWM features
    include_calendar : bool
        Include calendar features
    include_fourier : bool
        Include Fourier features
    include_trend : bool
        Include trend features
    include_intermittency : bool
        Include intermittency features (auto-enabled for intermittent/lumpy)
    include_cross_sectional : bool
        Include cross-sectional features
    lag_periods : List[int], optional
        Specific lag periods
    rolling_windows : List[int], optional
        Specific rolling windows
    prefix : str, optional
        Prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with all features

    Example
    -------
    >>> df = create_complete_feature_set(
    ...     df, ['key'], 'year_week', 'sales',
    ...     demand_pattern='smooth'
    ... )
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    # Default parameters based on demand pattern AND time format
    is_monthly = (time_format == 'year_month')
    if demand_pattern in ['intermittent', 'lumpy']:
        include_intermittency = True
        if lag_periods is None:
            lag_periods = [1, 2, 3, 6, 12] if is_monthly else [1, 2, 4, 8, 13]
        if rolling_windows is None:
            rolling_windows = [3, 6, 12] if is_monthly else [4, 8, 13]
    else:
        if lag_periods is None:
            lag_periods = [1, 2, 3, 6, 12] if is_monthly else [1, 2, 4, 8, 13, 26, 52]
        if rolling_windows is None:
            rolling_windows = [3, 6, 12] if is_monthly else [4, 8, 13, 26, 52]

    # Apply features
    if include_lags:
        df = create_lag_features(df, group_cols, target_col, lags=lag_periods, prefix=prefix, time_format=time_format)
        df = create_lag_combinations(df, group_cols, target_col, prefix=prefix, time_format=time_format)
        df = create_momentum_features(df, group_cols, target_col, prefix=prefix)
        df = create_seasonal_lag_features(df, group_cols, target_col, prefix=prefix, time_format=time_format)

    if include_rolling:
        df = create_rolling_features(df, group_cols, target_col, windows=rolling_windows, prefix=prefix, time_format=time_format)
        df = create_rolling_volatility_features(df, group_cols, target_col, windows=rolling_windows[:3], prefix=prefix)

    if include_ewm:
        df = create_ewm_features(df, group_cols, target_col, prefix=prefix)

    if include_calendar:
        df = create_calendar_features(df, date_col)
        if is_monthly and 'month' in df.columns:
            df = create_cyclical_features(df, 'month', period=12)
        elif 'week_of_year' in df.columns:
            df = create_cyclical_features(df, 'week_of_year', period=52)
            if 'month' in df.columns:
                df = create_cyclical_features(df, 'month', period=12)

    if include_fourier:
        df = create_fourier_features(df, date_col, time_format=time_format)

    if include_trend:
        df = create_trend_features(df, group_cols, target_col, prefix=prefix, time_format=time_format)

    if include_intermittency:
        df = create_intermittency_features(df, group_cols, target_col, prefix=prefix)
        # Also add sparsity features for better intermittent/lumpy demand modeling
        df = create_sparsity_features(df, group_cols, target_col, prefix=prefix)

    if include_cross_sectional:
        df = create_cross_sectional_features(df, group_cols, target_col, date_col, segment_col, prefix=prefix)
        df = create_key_level_features(df, group_cols, target_col, prefix=prefix)

    # Add expanding features
    df = create_expanding_features(df, group_cols, target_col, prefix=prefix)

    return df


def run_feature_pipeline(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    segment_col: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    demand_pattern_col: Optional[str] = None,
    output_dir: Optional[str] = None,
    time_format: str = 'year_week',
) -> FeatureEngineeringResult:
    """
    Run complete feature engineering pipeline with one function call.

    This is the HIGHEST-LEVEL function - does EVERYTHING in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    train_start, train_end : str
        Training date range
    val_start, val_end : str
        Validation date range
    test_start, test_end : str
        Test date range
    segment_col : str, optional
        Segment column
    categorical_cols : List[str], optional
        Categorical columns to encode
    demand_pattern_col : str, optional
        Column with demand pattern classification
    output_dir : str, optional
        Directory to save feature files

    Returns
    -------
    FeatureEngineeringResult
        Complete feature engineering result

    Example
    -------
    >>> result = run_feature_pipeline(
    ...     df=raw_df,
    ...     key_cols=['key'],
    ...     date_col='year_week',
    ...     target_col='sales',
    ...     train_start='202001', train_end='202352',
    ...     val_start='202401', val_end='202426',
    ...     test_start='202427', test_end='202452',
    ... )
    >>> print(f"Created {result.n_features_created} features")
    """
    logger.info("Starting feature engineering pipeline...")

    # Sort by key and date
    df = df.sort_values(key_cols + [date_col]).reset_index(drop=True)

    # Determine demand pattern per row if available
    default_pattern = 'smooth'
    if demand_pattern_col and demand_pattern_col in df.columns:
        # Use first row's pattern as default
        default_pattern = df[demand_pattern_col].mode().iloc[0] if len(df) > 0 else 'smooth'

    # Create features
    logger.info(f"Creating feature set (time_format={time_format})...")
    df = create_complete_feature_set(
        df,
        group_cols=key_cols,
        date_col=date_col,
        target_col=target_col,
        segment_col=segment_col,
        demand_pattern=default_pattern,
        include_intermittency=(default_pattern in ['intermittent', 'lumpy']),
        time_format=time_format,
    )

    # Encode categoricals
    encoding_specs = {}
    if categorical_cols:
        logger.info(f"Encoding {len(categorical_cols)} categorical columns...")
        df, encoding_specs = encode_all_categoricals(df, categorical_cols, target_col)

    # Split by date
    # Handle type mismatch: convert date range params to match df[date_col] dtype
    logger.info("Splitting data by date...")
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        # Integer column (e.g., year_week as 202521)
        ts = int(train_start) if isinstance(train_start, str) else train_start
        te = int(train_end) if isinstance(train_end, str) else train_end
        vs = int(val_start) if isinstance(val_start, str) else val_start
        ve = int(val_end) if isinstance(val_end, str) else val_end
        tss = int(test_start) if isinstance(test_start, str) else test_start
        tse = int(test_end) if isinstance(test_end, str) else test_end
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        ts, te = str(train_start), str(train_end)
        vs, ve = str(val_start), str(val_end)
        tss, tse = str(test_start), str(test_end)
    else:
        ts, te = train_start, train_end
        vs, ve = val_start, val_end
        tss, tse = test_start, test_end

    train_df = df[(df[date_col] >= ts) & (df[date_col] <= te)].copy()
    val_df = df[(df[date_col] >= vs) & (df[date_col] <= ve)].copy()
    test_df = df[(df[date_col] >= tss) & (df[date_col] <= tse)].copy()

    # Identify feature columns
    exclude_cols = set(key_cols + [date_col, target_col])
    if segment_col:
        exclude_cols.add(segment_col)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Fill missing values
    logger.info("Filling missing values...")
    train_df = fill_missing_features(train_df, feature_cols, strategy='median')
    val_df = fill_missing_features(val_df, feature_cols, strategy='median')
    test_df = fill_missing_features(test_df, feature_cols, strategy='median')

    # Check quality
    quality = check_feature_quality(train_df, feature_cols)
    logger.info(f"Feature quality: {len(quality['constant_features'])} constant features found")

    # Remove constant features from feature list AND from DataFrames
    constant_cols = quality['constant_features']
    if constant_cols:
        logger.info(f"Dropping constant features: {constant_cols}")
        # Drop from all DataFrames
        cols_to_drop = [c for c in constant_cols if c in train_df.columns]
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop)
            val_df = val_df.drop(columns=cols_to_drop)
            test_df = test_df.drop(columns=cols_to_drop)
    feature_cols = [c for c in feature_cols if c not in constant_cols]

    # Remove highly correlated features (e.g., rolling_mean vs ewm with r >= 0.95)
    n_before_corr = len(feature_cols)
    feature_cols, corr_dropped_pairs = remove_highly_correlated_features(
        train_df, feature_cols, correlation_threshold=0.95, prefer_simpler=True
    )
    n_after_corr = len(feature_cols)
    if n_before_corr > n_after_corr:
        logger.info(f"Removed {n_before_corr - n_after_corr} highly correlated features")
        # Drop from DataFrames
        cols_to_drop_corr = [c for c in train_df.columns if c not in feature_cols and c not in key_cols + [date_col, target_col, segment_col]]
        cols_to_drop_corr = [c for c in cols_to_drop_corr if c in train_df.columns]
        if cols_to_drop_corr:
            train_df = train_df.drop(columns=cols_to_drop_corr, errors='ignore')
            val_df = val_df.drop(columns=cols_to_drop_corr, errors='ignore')
            test_df = test_df.drop(columns=cols_to_drop_corr, errors='ignore')
        quality['correlated_pairs_dropped'] = corr_dropped_pairs

    # Create metadata
    feature_metadata = {
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'target_col': target_col,
        'key_cols': key_cols,
        'date_col': date_col,
        'segment_col': segment_col,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'quality_report': quality,
    }

    # Save if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Dead-feature pruning: drop columns with zero predictive signal
        # across the GLOBAL training panel (all-null / all-zero / all-
        # constant columns).  See `prune_dead_features` docstring at the
        # top of this module for the full rationale and leakage-safety
        # argument.  Done BEFORE writing parquet so every downstream
        # consumer reads the slim version directly off disk.
        #
        # Protected columns (NEVER dropped even if they look dead):
        #   1. Pipeline plumbing — key, date, target, segment, model_level.
        #   2. User-listed columns from the function signature
        #      (`categorical_cols`, `demand_pattern_col`).
        #
        # Why protect (2): a column the caller explicitly named is part
        # of the downstream contract — other code paths may try to
        # access it directly (df[col], df.merge(other, on=col), encoder
        # lookups, etc.).  If it happens to be all-zero/constant for
        # *this particular category*, the right behaviour is "keep it
        # but it'll add no signal", NOT "drop it and crash downstream".
        # Memory cost of preserving constants is negligible — pandas
        # category-dtype + parquet dictionary encoding compress them to
        # near-zero on disk.
        _protected = [c for c in (key_cols or []) if c]
        if date_col:
            _protected.append(date_col)
        if target_col:
            _protected.append(target_col)
        if segment_col:
            _protected.append(segment_col)
        # Keep model_level if present — used as the per-MG groupby key
        # in the training pipeline; never want to drop it accidentally.
        if 'model_level' in train_df.columns:
            _protected.append('model_level')
        if categorical_cols:
            _protected.extend(categorical_cols)
        if demand_pattern_col:
            _protected.append(demand_pattern_col)
        train_df, val_df, test_df, _prune_manifest = prune_dead_features(
            train_df, val_df, test_df, protected_cols=_protected,
        )
        if _prune_manifest['dropped_total'] > 0:
            logger.info(
                f"Dead-feature pruning: dropped {_prune_manifest['dropped_total']}/"
                f"{_prune_manifest['original_count']} cols "
                f"({_prune_manifest['dropped_pct']:.1%}) — "
                f"{len(_prune_manifest['all_null'])} all-null, "
                f"{len(_prune_manifest['all_zero'])} all-zero, "
                f"{len(_prune_manifest['all_constant'])} all-constant"
            )
            with open(os.path.join(output_dir, 'dead_features_dropped.json'), 'w') as _f:
                json.dump(_prune_manifest, _f, indent=2, default=str)

        # Parquet by default — 10-30x faster write, 3-5x smaller on disk,
        # dtype-preserving (year_week stays int, no string-coercion on
        # round-trip). Falls back to CSV transparently if pyarrow is
        # missing. See utils/feature_io.py for the full rationale.
        from utils.feature_io import write_features_intermediate
        write_features_intermediate(train_df, output_dir, 'train_features')
        write_features_intermediate(val_df,   output_dir, 'val_features')
        write_features_intermediate(test_df,  output_dir, 'test_features')

        with open(os.path.join(output_dir, 'feature_metadata.json'), 'w') as f:
            json.dump(feature_metadata, f, indent=2, default=str)

        with open(os.path.join(output_dir, 'categorical_encoding_specs.json'), 'w') as f:
            json.dump(encoding_specs, f, indent=2, default=str)

        logger.info(f"Saved feature files to {output_dir}")

    logger.info(f"Feature pipeline complete: {len(feature_cols)} features created")

    return FeatureEngineeringResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        encoding_specs=encoding_specs,
        feature_metadata=feature_metadata,
        n_features_created=len(feature_cols),
        n_rows_train=len(train_df),
        n_rows_val=len(val_df),
        n_rows_test=len(test_df),
    )


# =============================================================================
# 11. LEAKAGE-FREE FEATURE ENGINEERING FOR LAG-K FORECASTING
# =============================================================================
# CRITICAL: These functions prevent data leakage for proper lag-K forecasting
# =============================================================================

def create_lag_features_leakage_free(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    date_col: str,
    train_end: Union[str, int],
    forecast_lag: int,
    lags: List[int] = None,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create lag features that are SAFE from data leakage for lag-K forecasting.

    For lag-K forecasting:
    - Lags >= K: Use actual values (available at forecast time)
    - Lags < K: Set to NaN for val/test (will be filled with predictions during recursive forecasting)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by date within groups
    group_cols : str or List[str]
        Key columns for grouping
    target_col : str
        Target column name
    date_col : str
        Date column name
    train_end : str or int
        Last date in training period
    forecast_lag : int
        The lag-K value for forecasting (e.g., 4 for 4-week ahead)
    lags : List[int]
        Lag periods to create
    prefix : str
        Prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features (some NaN for val/test short lags)
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    # CRITICAL: Always include lag_1 for recursive forecasting
    # lag_1 is essential because during recursive prediction:
    # - Step 1: lag_1 = last known actual
    # - Step 2+: lag_1 = previous prediction
    # This enables true step-by-step forecasting without using future actuals
    if lags is None:
        # Always start with lag_1, then add other useful lags
        lags = [1, 2, 3, 6, 12] if time_format == 'year_month' else [1, 2, 4, 8, 13, 26, 52]

    # Ensure lag_1 is always included
    if 1 not in lags:
        lags = [1] + list(lags)
    lags = sorted(set(lags))

    # Identify training vs non-training rows
    # Handle type mismatch: convert train_end to match df[date_col] dtype
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        # Integer column (e.g., year_week as 202521)
        train_end_typed = int(train_end) if isinstance(train_end, str) else train_end
        is_train = df[date_col] <= train_end_typed
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        # String column
        is_train = df[date_col].astype(str) <= str(train_end)
    else:
        # Datetime or other - try direct comparison
        is_train = df[date_col] <= train_end

    for lag in lags:
        col_name = f'{prefix}_lag_{lag}'
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)

        # IMPORTANT: For val/test periods, short lags (< forecast_lag) will contain
        # values that wouldn't be known at forecast time. However, we still CREATE
        # these features because they will be UPDATED during recursive forecasting
        # using predictions. We set them to NaN here as a placeholder.
        # The recursive forecaster will fill them with (history + predictions).
        if lag < forecast_lag:
            df.loc[~is_train, col_name] = np.nan

    logger.info(f"Created {len(lags)} lag features including lag_1 for recursive forecasting")
    return df


def create_rolling_features_leakage_free(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    forecast_lag: int,
    windows: List[int] = None,
    aggregations: List[str] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling features for recursive forecasting.

    CRITICAL: Rolling features use shift(1) so they can be properly updated
    during recursive forecasting. The recursive forecaster will recalculate
    these using (history + predictions so far).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Key columns
    target_col : str
        Target column
    forecast_lag : int
        Forecast horizon (not used for shift - we always use shift(1))
    windows : List[int]
        Window sizes
    aggregations : List[str]
        Aggregation functions
    prefix : str
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    windows = windows or [4, 8, 13, 26]
    aggregations = aggregations or ['mean', 'std', 'min', 'max']

    # CRITICAL: Always use shift(1) so features can be updated during recursive forecasting
    # During training: uses actual values (t-1, t-2, ...)
    # During recursive forecast: updated using predictions
    shift_periods = 1

    for window in windows:
        for agg in aggregations:
            col_name = f'{prefix}_roll_{window}_{agg}'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(shift_periods).rolling(window, min_periods=1).agg(agg)
            )

    logger.info(f"Created rolling features with shift(1) for recursive forecasting")
    return df


def create_seasonal_features_leakage_free(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    forecast_lag: int,
    seasonal_periods: List[int] = None,
    prefix: Optional[str] = None,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Create seasonal features WITHOUT using current row's target value.

    Instead of:
        yoy_diff = current_value - value_52_weeks_ago  (LEAKAGE!)
    We use:
        yoy_lag_diff = value_K_weeks_ago - value_(52+K)_weeks_ago

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Key columns
    target_col : str
        Target column
    forecast_lag : int
        Forecast horizon
    seasonal_periods : List[int]
        Seasonal periods (default [52] for weekly)
    prefix : str
        Prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with seasonal features (leakage-free)
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    _default_period = 12 if time_format == 'year_month' else 52
    seasonal_periods = seasonal_periods or [_default_period]

    for period in seasonal_periods:
        # Seasonal lag (safe - this is just shift by period)
        df[f'{prefix}_seasonal_lag_{period}'] = df.groupby(group_cols)[target_col].shift(period)

        # CRITICAL: Use shift(1) for all derived features so they can be updated
        # during recursive forecasting
        # YoY change: lag_1 vs lag_(1+52) = lag_1 vs lag_53
        lag_recent = df.groupby(group_cols)[target_col].shift(1)
        lag_old = df.groupby(group_cols)[target_col].shift(1 + period)  # 1 + 52 = 53

        df[f'{prefix}_yoy_lag_diff'] = lag_recent - lag_old
        df[f'{prefix}_yoy_lag_ratio'] = lag_recent / (lag_old.abs() + 1e-10)

        # QoQ using shift(1): lag_1 vs lag_(1+13) = lag_1 vs lag_14
        if period == 52:
            lag_qtr_recent = df.groupby(group_cols)[target_col].shift(1)
            lag_qtr_old = df.groupby(group_cols)[target_col].shift(1 + 13)  # = lag_14
            df[f'{prefix}_qoq_lag_diff'] = lag_qtr_recent - lag_qtr_old

    logger.info(f"Created seasonal features with shift(1) for recursive forecasting")
    return df


def create_cross_sectional_features_leakage_free(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    date_col: str,
    forecast_lag: int,
    segment_col: Optional[str] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create cross-sectional features WITHOUT using current row's target.

    Uses lagged values for all comparisons instead of current values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Key columns
    target_col : str
        Target column
    date_col : str
        Date column
    forecast_lag : int
        Forecast horizon
    segment_col : str, optional
        Segment column
    prefix : str
        Prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with cross-sectional features (leakage-free)
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    # CRITICAL: Use shift(1) for consistency with other features
    # Cross-sectional features compare a key's lag_1 value to other keys' lag_1 values
    # Note: These features are STATIC during recursive forecasting (we don't re-compute
    # cross-sectional stats from predictions). They represent "where this key stood
    # relative to others based on last known actual".
    df['_lagged_target'] = df.groupby(group_cols)[target_col].shift(1)

    # Compute period statistics using lag_1 values only
    period_stats = df.groupby(date_col)['_lagged_target'].agg(['mean', 'std', 'median'])
    period_stats.columns = ['_period_mean', '_period_std', '_period_median']
    period_stats = period_stats.reset_index()

    # Drop existing temp columns to avoid duplicates
    temp_cols = ['_period_mean', '_period_std', '_period_median']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')

    df = df.merge(period_stats, on=date_col, how='left')

    # Relative features using lag_1 values
    df[f'{prefix}_rel_to_period_mean'] = df['_lagged_target'] / (df['_period_mean'].abs() + 1e-10)
    df[f'{prefix}_diff_from_period_mean'] = df['_lagged_target'] - df['_period_mean']
    df[f'{prefix}_zscore_period'] = (df['_lagged_target'] - df['_period_mean']) / (df['_period_std'].abs() + 1e-10)

    # Segment-relative features
    if segment_col and segment_col in df.columns:
        seg_period_stats = df.groupby([segment_col, date_col])['_lagged_target'].agg(['mean', 'std'])
        seg_period_stats.columns = ['_seg_period_mean', '_seg_period_std']
        seg_period_stats = seg_period_stats.reset_index()

        # Drop existing temp columns to avoid duplicates
        seg_temp_cols = ['_seg_period_mean', '_seg_period_std']
        df = df.drop(columns=[c for c in seg_temp_cols if c in df.columns], errors='ignore')

        df = df.merge(seg_period_stats, on=[segment_col, date_col], how='left')

        df[f'{prefix}_rel_to_segment'] = df['_lagged_target'] / (df['_seg_period_mean'].abs() + 1e-10)
        df[f'{prefix}_diff_from_segment'] = df['_lagged_target'] - df['_seg_period_mean']
        df[f'{prefix}_zscore_segment'] = (
            (df['_lagged_target'] - df['_seg_period_mean']) / (df['_seg_period_std'].abs() + 1e-10)
        )
        df = df.drop(columns=['_seg_period_mean', '_seg_period_std'], errors='ignore')

    # Rank using lag_1 values
    df[f'{prefix}_rank_in_period'] = df.groupby(date_col)['_lagged_target'].rank(method='dense', pct=True)

    # Clean up
    df = df.drop(columns=['_lagged_target', '_period_mean', '_period_std', '_period_median'], errors='ignore')

    logger.info(f"Created cross-sectional features with shift(1)")
    return df


def create_key_level_features_leakage_free(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    date_col: str,
    train_end: Union[str, int],
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create key-level features using ONLY training data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : str or List[str]
        Key columns
    target_col : str
        Target column
    date_col : str
        Date column
    train_end : str or int
        Last training date
    prefix : str
        Prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with key-level features
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    # Filter to training data only
    # Handle type mismatch: convert train_end to match df[date_col] dtype
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        train_end_typed = int(train_end) if isinstance(train_end, str) else train_end
        train_mask = df[date_col] <= train_end_typed
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        train_mask = df[date_col].astype(str) <= str(train_end)
    else:
        train_mask = df[date_col] <= train_end

    train_df = df[train_mask]

    # Compute stats on training data only
    key_stats = train_df.groupby(group_cols)[target_col].agg([
        ('key_mean', 'mean'),
        ('key_std', 'std'),
        ('key_min', 'min'),
        ('key_max', 'max'),
        ('key_count', 'count'),
        ('key_nonzero_count', lambda x: (x > 0).sum()),
    ]).reset_index()

    key_stats['key_cv'] = key_stats['key_std'] / (key_stats['key_mean'] + 1e-10)
    key_stats['key_zero_fraction'] = 1 - (key_stats['key_nonzero_count'] / key_stats['key_count'])

    # Rename with prefix
    rename_dict = {col: f'{prefix}_{col}' for col in key_stats.columns if col not in group_cols}
    key_stats = key_stats.rename(columns=rename_dict)

    # Drop existing columns to avoid duplicates on merge
    key_cols_to_add = [c for c in key_stats.columns if c not in group_cols]
    existing_key_cols = [c for c in key_cols_to_add if c in df.columns]
    if existing_key_cols:
        df = df.drop(columns=existing_key_cols)

    # Merge to ALL data
    df = df.merge(key_stats, on=group_cols, how='left')

    logger.info(f"Created key-level features (training data only)")
    return df


def run_leakage_free_feature_pipeline(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    forecast_lag: int,
    segment_col: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    # NEW: Context-driven parameters from EDA analysis
    eda_context: Optional[Dict[str, Any]] = None,
    # OR explicit parameters (used if eda_context not provided)
    target_lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    rolling_stats: Optional[List[str]] = None,
    include_intermittency_features: Optional[bool] = None,
    apply_log_transform: bool = False,
    apply_differencing: bool = False,
    include_trend_features: bool = True,
    include_fourier_features: bool = True,
    fourier_order: int = 3,
    seasonal_period: int = None,
    # NEW: Restrict to config-specified columns only
    allowed_external_cols: Optional[List[str]] = None,
    # NEW: Lags for external numeric features (price, promo, etc.)
    external_feature_lags: Optional[List[int]] = None,
    # Phase 3: Hierarchy columns for hierarchy-level temporal features
    hierarchy_cols: Optional[List[str]] = None,
    # Phase 6: Rich feature engineering
    enable_rich_features: bool = True,
    future_unknown_features: Optional[List[str]] = None,
) -> FeatureEngineeringResult:
    """
    Run LEAKAGE-FREE feature engineering pipeline for lag-K forecasting.

    This is the RECOMMENDED function for production use. It ensures:
    1. No lag features < K use actual values in val/test
    2. All rolling features are shifted by K
    3. No features use current row's target value
    4. Key-level stats computed on training data only

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    train_start, train_end : str
        Training date range
    val_start, val_end : str
        Validation date range
    test_start, test_end : str
        Test date range
    forecast_lag : int
        Forecast horizon (lag-K)
    segment_col : str, optional
        Segment column
    categorical_cols : List[str], optional
        Categorical columns to encode
    output_dir : str, optional
        Directory to save outputs
    eda_context : Dict, optional
        EDA context from eda_to_feature_context.json. If provided, extracts:
        - lag_recommendations.target_lags
        - rolling_features.windows, rolling_features.stats
        - intermittency_features.needed
        - transformation.log_transform, transformation.differencing
        - lag_recommendations.seasonal_period
    target_lags : List[int], optional
        Specific lags to create (from EDA recommendations)
    rolling_windows : List[int], optional
        Rolling window sizes (from EDA recommendations)
    rolling_stats : List[str], optional
        Rolling statistics to compute (from EDA recommendations)
    include_intermittency_features : bool, optional
        Whether to include intermittency features (from EDA intermittency_features.needed)
    apply_log_transform : bool
        Apply log1p transformation to target (from EDA transformation.log_transform)
    apply_differencing : bool
        Apply differencing (from EDA transformation.differencing)
    include_trend_features : bool
        Include trend features
    include_fourier_features : bool
        Include Fourier features
    fourier_order : int
        Order for Fourier features
    seasonal_period : int
        Seasonal period (from EDA lag_recommendations.seasonal_period)
    allowed_external_cols : List[str], optional
        List of column names from source data that are allowed to be used as features.
        This should be the union of all feature columns from config.yaml:
        numeric_feature_cols + price_features + promo_features + holiday_features + weather_features + categorical_feature_cols
        If None, all columns from the dataframe are allowed (legacy behavior).
        If provided, only these columns (plus key, date, target, segment) will be retained.
    external_feature_lags : List[int], optional
        Lags to create for external numeric features (price, promo, weather, etc.).
        For example, [1, 4] would create lag_1 and lag_4 for each numeric external feature.
        These capture delayed effects of external factors on demand.
        Default is None (no lags for external features).
        Recommended values: [1, 4] for weekly data (previous week, previous month).

    Returns
    -------
    FeatureEngineeringResult
        Feature engineering result
    """
    logger.info(f"Starting LEAKAGE-FREE feature pipeline (forecast_lag={forecast_lag})")

    # Validate period format consistency between config dates and data column
    from utils.period_utils import normalise_period_column, normalise_period
    df = normalise_period_column(df, date_col)
    train_start = normalise_period(train_start) if train_start else train_start
    train_end = normalise_period(train_end) if train_end else train_end
    val_start = normalise_period(val_start) if val_start else val_start
    val_end = normalise_period(val_end) if val_end else val_end
    test_start = normalise_period(test_start) if test_start else test_start
    test_end = normalise_period(test_end) if test_end else test_end

    # Validate forecast_lag against available history
    n_periods = df[date_col].nunique()
    if forecast_lag >= n_periods:
        raise ValueError(
            f"forecast_lag={forecast_lag} >= number of unique periods ({n_periods}). "
            f"Reduce forecast_lag or provide more history."
        )

    # =========================================================================
    # FILTER TO ALLOWED COLUMNS FROM CONFIG (if specified)
    # =========================================================================
    # This ensures only columns explicitly listed in config.yaml are used as features.
    # Engineered features (lag, rolling, etc.) are still created from the target column.
    if allowed_external_cols is not None:
        # Build set of columns to keep: required + allowed external
        required_cols = set(key_cols + [date_col, target_col])
        if segment_col:
            required_cols.add(segment_col)
        if categorical_cols:
            required_cols.update(categorical_cols)
        # Hierarchy columns must be retained for hierarchy-level feature creation
        if hierarchy_cols:
            required_cols.update(hierarchy_cols)

        # Filter allowed_external_cols to only those present in df
        allowed_in_df = [c for c in allowed_external_cols if c in df.columns]
        cols_to_keep = list(required_cols.union(set(allowed_in_df)))

        # Log what we're filtering
        original_cols = set(df.columns)
        dropped_cols = original_cols - set(cols_to_keep)
        if dropped_cols:
            logger.info(f"Filtering to config-specified columns only:")
            logger.info(f"  Original columns: {len(original_cols)}")
            logger.info(f"  Allowed external: {len(allowed_in_df)}")
            logger.info(f"  Dropped columns: {len(dropped_cols)}")
            logger.debug(f"  Dropped: {list(dropped_cols)[:10]}...")  # Log first 10

        df = df[cols_to_keep].copy()
        logger.info(f"Retained {len(cols_to_keep)} columns (required + config-specified)")

        # Log external features that will be used
        external_numeric = [c for c in allowed_in_df if c not in (categorical_cols or [])]
        external_categorical = [c for c in allowed_in_df if c in (categorical_cols or [])]
        logger.info(f"  External numeric features: {len(external_numeric)}")
        logger.info(f"  External categorical features: {len(external_categorical)}")
        if external_numeric:
            logger.debug(f"  Numeric: {external_numeric[:10]}{'...' if len(external_numeric) > 10 else ''}")
        if external_categorical:
            logger.debug(f"  Categorical: {external_categorical[:10]}{'...' if len(external_categorical) > 10 else ''}")

    # =========================================================================
    # EXTRACT CONTEXT-DRIVEN PARAMETERS FROM EDA CONTEXT
    # =========================================================================
    if eda_context:
        logger.info("Using EDA context for feature engineering parameters")

        # Extract lag recommendations
        lag_recs = eda_context.get('lag_recommendations', {})
        if target_lags is None:
            target_lags = lag_recs.get('target_lags')
        if seasonal_period is None:  # Use default only if not overridden
            _is_monthly = (date_col == 'year_month')
            seasonal_period = lag_recs.get('seasonal_period', 12 if _is_monthly else 52)

        # Extract rolling recommendations
        rolling_recs = eda_context.get('rolling_features', {})
        if rolling_windows is None:
            rolling_windows = rolling_recs.get('windows')
        if rolling_stats is None:
            rolling_stats = rolling_recs.get('stats')

        # Extract intermittency recommendation
        intermittency_recs = eda_context.get('intermittency_features', {})
        if include_intermittency_features is None:
            include_intermittency_features = intermittency_recs.get('needed', False)

        # Extract transformation recommendations
        transform_recs = eda_context.get('transformation', {})
        if not apply_log_transform:  # Only use context if not already set
            apply_log_transform = transform_recs.get('log_transform', False)
        if not apply_differencing:
            apply_differencing = transform_recs.get('differencing', False)

        logger.info(f"  EDA target_lags: {target_lags}")
        logger.info(f"  EDA rolling_windows: {rolling_windows}")
        logger.info(f"  EDA intermittency_needed: {include_intermittency_features}")
        logger.info(f"  EDA log_transform: {apply_log_transform}")

    # Resolve None seasonal_period to time-format-aware default
    _is_monthly = (date_col == 'year_month')
    if seasonal_period is None:
        seasonal_period = 12 if _is_monthly else 52

    # Sort by key and date
    df = df.sort_values(key_cols + [date_col]).reset_index(drop=True)

    # =========================================================================
    # USE CONTEXT-DRIVEN LAGS OR DEFAULTS
    # =========================================================================
    if target_lags:
        # Use EDA-recommended lags, but ensure all are >= forecast_lag
        lags = sorted(set([l for l in target_lags if l >= forecast_lag]))
        # Always include forecast_lag itself
        if forecast_lag not in lags:
            lags = [forecast_lag] + lags
        logger.info(f"Using EDA-recommended lags (filtered for forecast_lag={forecast_lag}): {lags}")
    else:
        # Default lags that prioritize >= forecast_lag (time-format-aware)
        if _is_monthly:
            lags = [forecast_lag, forecast_lag + 1, forecast_lag + 2, forecast_lag + 3, 6, 12]
        else:
            lags = [forecast_lag, forecast_lag + 1, forecast_lag + 2, forecast_lag + 4, 13, 26, 52]
        lags = sorted(set([l for l in lags if l >= 1]))
        logger.info(f"Using default lags: {lags}")

    # =========================================================================
    # APPLY LOG TRANSFORM IF RECOMMENDED BY EDA
    # =========================================================================
    original_target_col = target_col
    if apply_log_transform:
        logger.info("Applying log1p transformation (EDA recommendation)")
        df[f'{target_col}_log'] = np.log1p(df[target_col].clip(lower=0))
        # Use log-transformed target for lag/rolling features
        target_for_features = f'{target_col}_log'
    else:
        target_for_features = target_col

    # 1. Lag features (leakage-free)
    df = create_lag_features_leakage_free(
        df, key_cols, target_for_features, date_col, train_end, forecast_lag, lags
    )

    # 2. Rolling features (shifted by forecast_lag) - USE EDA-RECOMMENDED WINDOWS
    if rolling_windows:
        # Filter windows to be reasonable (>= forecast_lag for safety)
        windows_to_use = [w for w in rolling_windows if w >= 2]
        logger.info(f"Using EDA-recommended rolling windows: {windows_to_use}")
    else:
        windows_to_use = [4, 8, 13, 26]
        logger.info(f"Using default rolling windows: {windows_to_use}")

    df = create_rolling_features_leakage_free(
        df, key_cols, target_for_features, forecast_lag, windows=windows_to_use
    )

    # 3. EWM features - use shift(1) for recursive forecasting consistency
    df = create_ewm_features(df, key_cols, target_for_features, shift_periods=1)

    # 4. Seasonal features (leakage-free) - USE EDA-RECOMMENDED SEASONAL PERIOD
    df = create_seasonal_features_leakage_free(
        df, key_cols, target_for_features, forecast_lag, seasonal_periods=[seasonal_period]
    )

    # 5. Cross-sectional features (leakage-free, static during recursive forecasting)
    try:
        df = create_cross_sectional_features_leakage_free(
            df, key_cols, target_col, date_col, forecast_lag, segment_col
        )
        logger.info("Created cross-sectional features")
    except Exception as e:
        logger.warning(f"Cross-sectional features failed (non-critical): {e}")

    # 5a. Promo-lift features (leakage-free, rolling per key, shifted by
    # forecast_lag). Gives the model the per-key magnitude of promo
    # response as an explicit regressor rather than making it learn that
    # from binary flags.
    try:
        from utils.promo_features import create_promo_lift_features
        df = create_promo_lift_features(
            df, key_cols=key_cols, target_col=target_col, date_col=date_col,
            forecast_lag=forecast_lag,
        )
    except Exception as e:
        logger.warning(f"Promo-lift features failed (non-critical): {e}")

    # 5b. Hierarchy-level temporal features (Phase 3)
    if hierarchy_cols:
        try:
            from utils.hierarchy_features import create_hierarchy_temporal_features
            logger.info(f"Creating hierarchy temporal features for: {hierarchy_cols}")
            df = create_hierarchy_temporal_features(
                df, key_cols, date_col, target_col, hierarchy_cols, forecast_lag,
                windows=[4, 13], stats=['sum', 'mean'],
            )
        except Exception as e:
            logger.warning(f"Hierarchy features failed (non-critical): {e}")

    # 5c. Rich features: regime, cross-key relative, history embeddings (Phase 6)
    if enable_rich_features:
        try:
            from utils.rich_features import create_all_rich_features
            logger.info("Creating Phase 6 rich features (regime, relative, history embeddings)...")
            _seasonal = seasonal_period if seasonal_period else (12 if 'month' in str(date_col).lower() else 52)
            df, rich_meta = create_all_rich_features(
                df=df, key_cols=key_cols, date_col=date_col, target_col=target_col,
                forecast_lag=forecast_lag, train_end=train_end,
                segment_col=segment_col, hierarchy_cols=hierarchy_cols,
                future_unknown_features=future_unknown_features,
                enable_regime=True, enable_relative=True,
                enable_history_embeddings=bool(future_unknown_features),
                seasonal_period=_seasonal,
            )
            logger.info(f"Rich features added: {rich_meta.get('total_features_added', 0)}")
        except Exception as e:
            logger.warning(f"Rich features failed (non-critical): {e}")

    # 6. Key-level features (training data only)
    df = create_key_level_features_leakage_free(
        df, key_cols, target_col, date_col, train_end
    )

    # =========================================================================
    # INTERMITTENCY FEATURES - ONLY IF RECOMMENDED BY EDA
    # =========================================================================
    if include_intermittency_features:
        logger.info("Creating intermittency features (EDA recommendation: high zero fraction)")
        # Intermittency features use shift(1) - will be updated during recursive forecasting
        df = create_intermittency_features(df, key_cols, target_col)
        # Also add advanced sparsity features for better sparse demand modeling
        logger.info("Creating sparsity features (weeks_since_demand, demand_prob, typical_order_size)")
        df = create_sparsity_features(df, key_cols, target_col)

    # =========================================================================
    # TREND FEATURES - IF ENABLED
    # =========================================================================
    if include_trend_features:
        # Trend features use shift(1) - will be updated during recursive forecasting
        df = create_trend_features(df, key_cols, target_col)

    # 7. Calendar features (no leakage)
    df = create_calendar_features(df, date_col)
    if 'week_of_year' in df.columns:
        df = create_cyclical_features(df, 'week_of_year', period=seasonal_period)
    if 'month' in df.columns:
        df = create_cyclical_features(df, 'month', period=12)

    # 7b. DACH Holiday features (no leakage — based on calendar only)
    logger.info("Creating DACH holiday features (is_holiday, holiday_type, pre/post windows)")
    df = create_holiday_features(df, date_col)
    # Encode holiday_type as numeric for LightGBM/CatBoost compatibility
    if 'holiday_type' in df.columns:
        _htype_map = {'none': 0, 'other': 1, 'easter': 2, 'new_year': 3, 'christmas': 4}
        df['holiday_type'] = df['holiday_type'].map(_htype_map).fillna(0).astype(int)
    holiday_cols_added = [c for c in ['is_holiday', 'holiday_type', 'is_pre_holiday_1w',
                                       'is_pre_holiday_2w', 'is_post_holiday_1w',
                                       'is_post_holiday_2w', 'weeks_to_nearest_holiday']
                          if c in df.columns]
    logger.info(f"  Holiday features added: {holiday_cols_added}")

    # 8. Fourier features (no leakage) - USE EDA-RECOMMENDED SEASONAL PERIOD
    if include_fourier_features:
        # Fixed: Use correct parameter names (periods=[list], n_harmonics=int)
        df = create_fourier_features(df, date_col, periods=[seasonal_period], n_harmonics=fourier_order)

    # =========================================================================
    # 8.5. EXTERNAL FEATURE LAGS - Create lagged versions of external numeric features
    # =========================================================================
    # This captures delayed effects of external factors on demand:
    # - Price lag: "Was there a price change recently that's still affecting demand?"
    # - Promo lag: "Did last week's promo create a demand spike or dip?"
    # - Weather lag: "Did last week's weather affect shopping patterns?"
    external_lag_features_created = 0
    if external_feature_lags and allowed_external_cols:
        # Identify numeric external features (exclude categoricals which will be encoded)
        external_numeric_cols = [c for c in allowed_external_cols
                                 if c in df.columns
                                 and c not in (categorical_cols or [])
                                 and c not in key_cols + [date_col, target_col]
                                 and pd.api.types.is_numeric_dtype(df[c])]

        if external_numeric_cols:
            logger.info(f"Creating lags {external_feature_lags} for {len(external_numeric_cols)} external numeric features")

            # Sort by key and date for proper lag creation
            df = df.sort_values(key_cols + [date_col]).reset_index(drop=True)

            for col in external_numeric_cols:
                for lag in external_feature_lags:
                    lag_col_name = f'{col}_lag_{lag}'
                    # Create lag within each key group
                    df[lag_col_name] = df.groupby(key_cols)[col].shift(lag)
                    external_lag_features_created += 1

            logger.info(f"  Created {external_lag_features_created} external feature lag columns")
            logger.debug(f"  External features with lags: {external_numeric_cols[:5]}{'...' if len(external_numeric_cols) > 5 else ''}")
        else:
            logger.info("No numeric external features found for lag creation")

    # 8b. Promo Adstock / decay curve features (Sprint 3 B8)
    if allowed_external_cols:
        promo_cols = [c for c in allowed_external_cols
                      if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                      and any(p in c.lower() for p in ['promo', 'discount', 'promotion', 'flag'])]
        if promo_cols:
            try:
                from utils.sprint3_features import create_adstock_features
                df = create_adstock_features(df, key_cols, promo_cols, decay_rates=[0.3, 0.6, 0.9])
            except Exception as e:
                logger.warning(f"Adstock features failed (non-critical): {e}")

    # 9. Split by date FIRST (before categorical encoding to prevent leakage)
    # Handle type mismatch: convert date range params to match df[date_col] dtype
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        # Integer column (e.g., year_week as 202521)
        ts = int(train_start) if isinstance(train_start, str) else train_start
        te = int(train_end) if isinstance(train_end, str) else train_end
        vs = int(val_start) if isinstance(val_start, str) else val_start
        ve = int(val_end) if isinstance(val_end, str) else val_end
        tss = int(test_start) if isinstance(test_start, str) else test_start
        tse = int(test_end) if isinstance(test_end, str) else test_end
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        # String column
        ts, te = str(train_start), str(train_end)
        vs, ve = str(val_start), str(val_end)
        tss, tse = str(test_start), str(test_end)
    else:
        # Datetime or other
        ts, te = train_start, train_end
        vs, ve = val_start, val_end
        tss, tse = test_start, test_end

    train_df = df[(df[date_col] >= ts) & (df[date_col] <= te)].copy()
    val_df = df[(df[date_col] >= vs) & (df[date_col] <= ve)].copy()
    test_df = df[(df[date_col] >= tss) & (df[date_col] <= tse)].copy()

    # 10. Encode categoricals - FIT ON TRAINING DATA ONLY to prevent leakage
    encoding_specs = {}
    if categorical_cols:
        logger.info(f"Encoding {len(categorical_cols)} categorical columns (fit on training data only)")
        # First, fit encodings on training data
        train_df, encoding_specs = encode_all_categoricals(
            train_df, categorical_cols, target_col, fit_data=train_df
        )
        # Then apply same encodings to val and test (using existing specs)
        val_df, _ = encode_all_categoricals(
            val_df, categorical_cols, target_col, existing_specs=encoding_specs
        )
        test_df, _ = encode_all_categoricals(
            test_df, categorical_cols, target_col, existing_specs=encoding_specs
        )

    # 11. Identify feature columns from the ENCODED train_df (not original df)
    # This ensures one-hot encoded columns are included
    exclude_cols = set(key_cols + [date_col, target_col])
    if segment_col:
        exclude_cols.add(segment_col)
    # Also exclude original categorical columns (we use encoded versions)
    if categorical_cols:
        exclude_cols.update(categorical_cols)
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # =========================================================================
    # REMOVE LEAKING FEATURES - Features that use current target values
    # These features contain information about the target that wouldn't be
    # available at prediction time.
    #
    # NOTE: With the leakage-free feature functions, most of these should NOT
    # be created anymore. This filter serves as a safety net for any legacy
    # features or features created by the non-leakage-free functions.
    # =========================================================================
    leaking_patterns = [
        f'{target_col}_log',  # Direct log transform of target (not lagged)
        f'{target_col}_demand_occurred',  # Binary flag using current target (OLD - now we use lag_demand_occurred)
        f'{target_col}_zscore_period',  # Z-score using current period data
        f'{target_col}_rel_to_period_mean',  # Relative to current period mean
        f'{target_col}_diff_from_period_mean',  # Diff from current period mean
        f'{target_col}_rank_in_period',  # Rank using current period data
        f'{target_col}_rel_to_segment',  # Relative to current segment mean
        f'{target_col}_diff_from_segment',  # Diff from current segment mean
        f'{target_col}_zscore_segment',  # Z-score using current segment data
        # OLD yoy/qoq features that use current target (now replaced with yoy_lag_diff, qoq_lag_diff)
        f'{target_col}_yoy_diff',  # current - lag_52 (LEAKAGE)
        f'{target_col}_yoy_ratio',  # current / lag_52 (LEAKAGE)
        f'{target_col}_qoq_diff',  # current - lag_13 (LEAKAGE)
    ]

    # Also remove diff/pct_change features that don't include 'lag' (they use current values)
    # The new leakage-free versions use 'lag_diff', 'lag_pct_change', 'lag_direction'
    leaking_diff_patterns = [
        f'{target_col}_diff_1', f'{target_col}_diff_4', f'{target_col}_diff_13', f'{target_col}_diff_52',
        f'{target_col}_pct_change_1', f'{target_col}_pct_change_4', f'{target_col}_pct_change_13', f'{target_col}_pct_change_52',
        f'{target_col}_direction_1', f'{target_col}_direction_4', f'{target_col}_direction_13', f'{target_col}_direction_52',
    ]
    leaking_patterns.extend(leaking_diff_patterns)

    original_count = len(feature_cols)
    feature_cols = [c for c in feature_cols if c not in leaking_patterns]
    removed_count = original_count - len(feature_cols)

    if removed_count > 0:
        logger.warning(f"LEAKAGE PREVENTION: Removed {removed_count} features that use current target values")
        logger.debug(f"Removed features: {[c for c in leaking_patterns if c in df.columns]}")

    # 12. VALIDATE NO TARGET LEAKAGE
    # This is a critical check to ensure no features use current target value
    leakage_report = validate_no_target_leakage(train_df, feature_cols, target_col)
    if leakage_report['potential_leaking_features']:
        # Remove any detected leaking features
        for feat in leakage_report['potential_leaking_features']:
            if feat in feature_cols:
                feature_cols.remove(feat)
                logger.warning(f"Removed leaking feature: {feat}")

    # 13. Check quality and remove constant features
    quality = check_feature_quality(train_df, feature_cols)
    feature_cols = [c for c in feature_cols if c not in quality.get('constant_features', [])]

    # 14. Fill missing values
    train_df = fill_missing_features(train_df, feature_cols, strategy='median')
    val_df = fill_missing_features(val_df, feature_cols, strategy='median')
    test_df = fill_missing_features(test_df, feature_cols, strategy='median')

    # 14. Create metadata - INCLUDING CONTEXT-DRIVEN PARAMETERS USED
    feature_metadata = {
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'target_col': target_col,
        'key_cols': key_cols,
        'date_col': date_col,
        'segment_col': segment_col,
        'forecast_lag': forecast_lag,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'quality_report': quality,
        'leakage_free': True,
        # NEW: Record context-driven parameters used
        'eda_context_used': eda_context is not None,
        'lag_config': {
            'target_lags_used': lags,
            'from_eda': target_lags is not None,
        },
        'rolling_config': {
            'windows_used': windows_to_use,
            'from_eda': rolling_windows is not None,
        },
        'intermittency_config': {
            'features_included': include_intermittency_features or False,
            'from_eda': eda_context is not None and eda_context.get('intermittency_features', {}).get('needed') is not None,
        },
        'transformation_config': {
            'log_transform_applied': apply_log_transform,
            'differencing_applied': apply_differencing,
        },
        'seasonal_period': seasonal_period,
        'feature_categories': {
            'lag_features': len([c for c in feature_cols if 'lag_' in c or '_lag' in c]),
            'rolling_features': len([c for c in feature_cols if 'rolling_' in c or '_roll_' in c]),
            'ewm_features': len([c for c in feature_cols if 'ewm_' in c]),
            'calendar_features': len([c for c in feature_cols if any(x in c for x in ['week', 'month', 'quarter', 'year', 'dow'])]),
            'intermittency_features': len([c for c in feature_cols if any(x in c for x in ['zero', 'nonzero', 'occurrence'])]),
            'trend_features': len([c for c in feature_cols if 'trend' in c]),
            # External features: features from config that are not engineered
            'external_features': len([c for c in feature_cols
                                      if c in (allowed_external_cols or [])
                                      and not any(x in c for x in ['lag_', '_lag', 'rolling_', '_roll_', 'ewm_', 'fourier_', 'trend', '_encoded'])]),
            # External lag features: lagged versions of external variables (price, promo, etc.)
            'external_lag_features': external_lag_features_created,
        },
        # Track which external features were actually used
        'external_features_used': [c for c in feature_cols
                                   if c in (allowed_external_cols or [])
                                   and not any(x in c for x in ['lag_', '_lag', 'rolling_', '_roll_', 'ewm_', 'fourier_', 'trend', '_encoded'])][:20],
        # Track external feature lag configuration
        'external_feature_lag_config': {
            'lags_used': external_feature_lags,
            'n_features_created': external_lag_features_created,
        },
    }

    # 15. Save if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Dead-feature pruning: drop columns with zero predictive signal
        # across the GLOBAL training panel (all-null / all-zero / all-
        # constant columns).  See `prune_dead_features` docstring at the
        # top of this module for the full rationale and leakage-safety
        # argument.  Done BEFORE writing parquet so every downstream
        # consumer reads the slim version directly off disk — including
        # this leakage-free pipeline's primary consumer (the per-MG
        # training loop) AND the DMH inference path (which re-reads the
        # same parquet artifacts to retrain horizon-specific heads).
        #
        # Protected columns (NEVER dropped even if they look dead):
        #   1. Pipeline plumbing — key, date, target, segment, model_level.
        #   2. EVERY user-listed column from the function signature:
        #      categorical_cols, allowed_external_cols, hierarchy_cols,
        #      future_unknown_features.
        #
        # Why protect (2): a column the caller explicitly named is part
        # of the downstream contract.  Examples:
        #   - `categorical_cols` columns get target/freq/ordinal encoded
        #     and the encoder spec is persisted to disk; if the source
        #     column is dropped, the encoder load at inference time
        #     fails with KeyError.
        #   - `hierarchy_cols` are used by hierarchical_model_training
        #     (Phase 3 models — global_local, mixed_effects) to pool
        #     across keys; missing → cannot build the hierarchy index.
        #   - `allowed_external_cols` is the user's whitelist of which
        #     non-engineered columns to keep — dropping any of them
        #     overrides user intent.
        #   - `future_unknown_features` is the per-horizon DMH augmenter's
        #     forward-looking feature set — name mismatch causes silent
        #     skip in DMH training.
        # Memory cost of preserving constants is negligible (pandas
        # category-dtype + parquet dictionary encoding compress them to
        # near-zero on disk).  Empirically the savings drop only a
        # few percentage points: most of the dead columns are FE-
        # generated derivatives (one-hot encoded promo dummies, lag
        # versions of empty source cols, etc.) that no caller named.
        _protected = list(key_cols or [])
        if date_col:
            _protected.append(date_col)
        if target_col:
            _protected.append(target_col)
        if segment_col:
            _protected.append(segment_col)
        if 'model_level' in train_df.columns:
            _protected.append('model_level')
        if categorical_cols:
            _protected.extend(categorical_cols)
        if allowed_external_cols:
            _protected.extend(allowed_external_cols)
        if hierarchy_cols:
            # hierarchy_cols may be List[str] (most common at this layer)
            # or a {level: List[str]} dict from the YAML.  Flatten both.
            if isinstance(hierarchy_cols, dict):
                for _v in hierarchy_cols.values():
                    if isinstance(_v, (list, tuple)):
                        _protected.extend(_v)
                    elif isinstance(_v, str):
                        _protected.append(_v)
            elif isinstance(hierarchy_cols, (list, tuple)):
                _protected.extend(hierarchy_cols)
            elif isinstance(hierarchy_cols, str):
                _protected.append(hierarchy_cols)
        if future_unknown_features:
            _protected.extend(future_unknown_features)
        train_df, val_df, test_df, _prune_manifest = prune_dead_features(
            train_df, val_df, test_df, protected_cols=_protected,
        )
        if _prune_manifest['dropped_total'] > 0:
            logger.info(
                f"Dead-feature pruning: dropped {_prune_manifest['dropped_total']}/"
                f"{_prune_manifest['original_count']} cols "
                f"({_prune_manifest['dropped_pct']:.1%}) — "
                f"{len(_prune_manifest['all_null'])} all-null, "
                f"{len(_prune_manifest['all_zero'])} all-zero, "
                f"{len(_prune_manifest['all_constant'])} all-constant"
            )
            with open(os.path.join(output_dir, 'dead_features_dropped.json'), 'w') as _f:
                json.dump(_prune_manifest, _f, indent=2, default=str)

        # Parquet by default (see utils/feature_io.py for the rationale).
        # Falls back to CSV transparently if pyarrow is unavailable.
        from utils.feature_io import write_features_intermediate
        write_features_intermediate(train_df, output_dir, 'train_features')
        write_features_intermediate(val_df,   output_dir, 'val_features')
        write_features_intermediate(test_df,  output_dir, 'test_features')

        with open(os.path.join(output_dir, 'feature_metadata.json'), 'w') as f:
            json.dump(feature_metadata, f, indent=2, default=str)

        # Save encoding specs for inference (needed to apply same encoding to new data)
        if encoding_specs:
            with open(os.path.join(output_dir, 'encoding_specs.json'), 'w') as f:
                json.dump(encoding_specs, f, indent=2, default=str)
            logger.info(f"Saved encoding_specs.json for inference consistency")

        # Save feature strategy decision — CRITICAL for backtesting feature regeneration.
        # Without this file, backtesting uses default parameters instead of the exact
        # same parameters training used, causing feature mismatch → invalid predictions.
        strategy_decision = {
            'target_lags': target_lags,
            'rolling_windows': rolling_windows,
            'rolling_stats': rolling_stats,
            'include_intermittency_features': include_intermittency_features,
            'apply_log_transform': apply_log_transform,
            'apply_differencing': apply_differencing,
            'include_trend_features': include_trend_features,
            'include_fourier_features': include_fourier_features,
            'fourier_order': fourier_order,
            'seasonal_period': seasonal_period,
            'forecast_lag': forecast_lag,
            'allowed_external_cols': allowed_external_cols,
            'external_feature_lags': external_feature_lags,
            'hierarchy_cols': hierarchy_cols,
            'enable_rich_features': enable_rich_features,
            'future_unknown_features': future_unknown_features,
            'n_features_created': len(feature_cols),
            'feature_cols': feature_cols,
        }
        with open(os.path.join(output_dir, 'feature_strategy_decision.json'), 'w') as f:
            json.dump(strategy_decision, f, indent=2, default=str)
        logger.info("Saved feature_strategy_decision.json for backtesting consistency")

        logger.info(f"Saved leakage-free feature files to {output_dir}")

    logger.info(f"Leakage-free feature pipeline complete: {len(feature_cols)} features")

    return FeatureEngineeringResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        encoding_specs=encoding_specs,
        feature_metadata=feature_metadata,
        n_features_created=len(feature_cols),
        n_rows_train=len(train_df),
        n_rows_val=len(val_df),
        n_rows_test=len(test_df),
    )


# =============================================================================
# 12. EDA-DRIVEN FEATURE ENRICHMENT (STATE-OF-THE-ART)
# =============================================================================
# These functions leverage exhaustive EDA analysis for intelligent feature creation
# =============================================================================


@dataclass
class EDAFeatureInsights:
    """Insights from EDA analysis for feature engineering."""
    # ACF-informed lag configuration (defaults are for weekly; override for monthly)
    acf_significant_lags: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 13, 26, 52])
    acf_optimal_lag_by_segment: Dict[str, List[int]] = field(default_factory=dict)

    # Seasonality insights
    detected_seasonal_period: int = 12  # Default 12; overridden by EDA detection
    has_seasonality_pct: float = 0.0
    seasonality_by_segment: Dict[str, bool] = field(default_factory=dict)

    # Trend insights
    avg_trend_strength: float = 0.0
    strongly_trending_pct: float = 0.0
    trend_by_segment: Dict[str, float] = field(default_factory=dict)

    # Changepoint insights
    pct_with_changepoints: float = 0.0
    changepoint_dates: List[str] = field(default_factory=list)
    changepoints_by_key: Dict[str, List[str]] = field(default_factory=dict)

    # Cross-correlation insights (for hierarchical data)
    correlated_series_pairs: List[Tuple[str, str, float]] = field(default_factory=list)

    # Recommended configurations
    recommended_rolling_windows: List[int] = field(default_factory=lambda: [4, 13, 26, 52])
    recommended_fourier_order: int = 3
    needs_intermittency_features: bool = True
    needs_trend_features: bool = True
    needs_changepoint_indicators: bool = False


def load_eda_insights_for_features(
    eda_dir: str,
) -> EDAFeatureInsights:
    """
    Load ALL exhaustive EDA outputs and extract insights for feature engineering.

    This function reads:
    - autocorrelation_summary.csv - ACF/PACF for optimal lag selection
    - seasonality_analysis.json - FFT-detected seasonal periods
    - trend_analysis.json - Trend strength per key
    - changepoint_analysis.json - Structural breaks
    - cross_series_correlation.json - Cross-key correlations
    - eda_to_feature_context.json - Core EDA recommendations

    Parameters
    ----------
    eda_dir : str
        Path to EDA output directory

    Returns
    -------
    EDAFeatureInsights
        Comprehensive insights for feature engineering
    """
    insights = EDAFeatureInsights()

    # =========================================================================
    # 1. Load ACF/PACF Summary - Critical for lag selection
    # =========================================================================
    acf_path = os.path.join(eda_dir, 'autocorrelation_summary.csv')
    if os.path.exists(acf_path):
        try:
            acf_df = pd.read_csv(acf_path)

            # Extract significant lags from ACF analysis
            if 'significant_lags' in acf_df.columns:
                # Aggregate significant lags across all keys
                all_significant_lags = set()
                for lags_str in acf_df['significant_lags'].dropna():
                    if isinstance(lags_str, str):
                        try:
                            lags = json.loads(lags_str.replace("'", '"'))
                            all_significant_lags.update(lags)
                        except:
                            pass
                if all_significant_lags:
                    insights.acf_significant_lags = sorted(list(all_significant_lags))[:15]

            # Build per-segment optimal lags
            if 'segment_id' in acf_df.columns and 'significant_lags' in acf_df.columns:
                for seg_id in acf_df['segment_id'].unique():
                    seg_lags = set()
                    seg_data = acf_df[acf_df['segment_id'] == seg_id]
                    for lags_str in seg_data['significant_lags'].dropna():
                        if isinstance(lags_str, str):
                            try:
                                lags = json.loads(lags_str.replace("'", '"'))
                                seg_lags.update(lags)
                            except:
                                pass
                    if seg_lags:
                        insights.acf_optimal_lag_by_segment[str(seg_id)] = sorted(list(seg_lags))[:10]

            logger.info(f"Loaded ACF insights: {len(insights.acf_significant_lags)} significant lags")
        except Exception as e:
            logger.warning(f"Error loading ACF summary: {e}")

    # =========================================================================
    # 2. Load Seasonality Analysis - For adaptive Fourier features
    # =========================================================================
    seasonality_path = os.path.join(eda_dir, 'seasonality_analysis.json')
    if os.path.exists(seasonality_path):
        try:
            with open(seasonality_path) as f:
                seasonality = json.load(f)

            insights.has_seasonality_pct = seasonality.get('has_seasonality_pct', 0)

            # Get dominant period
            dominant_periods = seasonality.get('dominant_periods', {})
            if dominant_periods:
                # Find most common period
                period_counts = {}
                for key, period in dominant_periods.items():
                    period_counts[period] = period_counts.get(period, 0) + 1
                if period_counts:
                    insights.detected_seasonal_period = max(period_counts, key=period_counts.get)

            # Per-segment seasonality
            if 'seasonality_by_segment' in seasonality:
                insights.seasonality_by_segment = seasonality['seasonality_by_segment']

            # Adjust Fourier order based on seasonality strength
            if insights.has_seasonality_pct > 0.7:
                insights.recommended_fourier_order = 5
            elif insights.has_seasonality_pct > 0.4:
                insights.recommended_fourier_order = 3
            else:
                insights.recommended_fourier_order = 2

            logger.info(f"Loaded seasonality: {insights.has_seasonality_pct*100:.0f}% seasonal, period={insights.detected_seasonal_period}")
        except Exception as e:
            logger.warning(f"Error loading seasonality analysis: {e}")

    # =========================================================================
    # 3. Load Trend Analysis - For trend features configuration
    # =========================================================================
    trend_path = os.path.join(eda_dir, 'trend_analysis.json')
    if os.path.exists(trend_path):
        try:
            with open(trend_path) as f:
                trend_analysis = json.load(f)

            insights.avg_trend_strength = trend_analysis.get('avg_trend_strength', 0)
            insights.strongly_trending_pct = trend_analysis.get('strongly_trending_pct', 0)

            # Per-segment trend
            if 'trend_by_segment' in trend_analysis:
                insights.trend_by_segment = trend_analysis['trend_by_segment']

            # Configure trend features based on findings
            insights.needs_trend_features = insights.strongly_trending_pct > 0.2

            logger.info(f"Loaded trend: {insights.strongly_trending_pct*100:.0f}% strongly trending")
        except Exception as e:
            logger.warning(f"Error loading trend analysis: {e}")

    # =========================================================================
    # 4. Load Changepoint Analysis - For regime indicators
    # =========================================================================
    changepoint_path = os.path.join(eda_dir, 'changepoint_analysis.json')
    if os.path.exists(changepoint_path):
        try:
            with open(changepoint_path) as f:
                changepoints = json.load(f)

            insights.pct_with_changepoints = changepoints.get('pct_with_changepoints', 0)

            # Extract common changepoint dates
            if 'common_changepoint_dates' in changepoints:
                insights.changepoint_dates = changepoints['common_changepoint_dates']

            # Per-key changepoints
            if 'changepoints_by_key' in changepoints:
                insights.changepoints_by_key = changepoints['changepoints_by_key']

            # Enable changepoint indicators if many series have them
            insights.needs_changepoint_indicators = insights.pct_with_changepoints > 0.3

            logger.info(f"Loaded changepoints: {insights.pct_with_changepoints*100:.0f}% have structural breaks")
        except Exception as e:
            logger.warning(f"Error loading changepoint analysis: {e}")

    # =========================================================================
    # 5. Load Cross-Series Correlation - For peer features
    # =========================================================================
    cross_corr_path = os.path.join(eda_dir, 'cross_series_correlation.json')
    if os.path.exists(cross_corr_path):
        try:
            with open(cross_corr_path) as f:
                cross_corr = json.load(f)

            if 'highly_correlated_pairs' in cross_corr:
                insights.correlated_series_pairs = [
                    (p['key1'], p['key2'], p['correlation'])
                    for p in cross_corr['highly_correlated_pairs'][:50]
                ]

            logger.info(f"Loaded cross-correlation: {len(insights.correlated_series_pairs)} correlated pairs")
        except Exception as e:
            logger.warning(f"Error loading cross-correlation: {e}")

    # =========================================================================
    # 6. Load Core EDA Context - For intermittency and rolling config
    # =========================================================================
    eda_context_path = os.path.join(eda_dir, 'eda_to_feature_context.json')
    if os.path.exists(eda_context_path):
        try:
            with open(eda_context_path) as f:
                eda_context = json.load(f)

            # Intermittency configuration
            intermittency = eda_context.get('intermittency_features', {})
            insights.needs_intermittency_features = intermittency.get('needed', True)

            # Rolling window configuration
            rolling = eda_context.get('rolling_features', {})
            if 'windows' in rolling:
                insights.recommended_rolling_windows = rolling['windows']

            logger.info(f"Loaded EDA context: intermittency={insights.needs_intermittency_features}")
        except Exception as e:
            logger.warning(f"Error loading EDA context: {e}")

    return insights


def get_adaptive_lag_config(
    insights: EDAFeatureInsights,
    segment_id: Optional[str] = None,
    demand_pattern: Optional[str] = None,
    default_lags: List[int] = None,
    time_format: str = 'year_week',
) -> List[int]:
    """
    Get adaptive lag configuration based on EDA insights.

    Parameters
    ----------
    insights : EDAFeatureInsights
        EDA insights object
    segment_id : str, optional
        Segment ID for segment-specific lags
    demand_pattern : str, optional
        Demand pattern (smooth, erratic, intermittent, lumpy)
    default_lags : List[int], optional
        Default lags if no insights available

    Returns
    -------
    List[int]
        Recommended lag values
    """
    _is_monthly = (time_format == 'year_month')
    if _is_monthly:
        default_lags = default_lags or [1, 2, 3, 6, 12]
    else:
        default_lags = default_lags or [1, 2, 4, 8, 13, 26, 52]

    # Priority 1: Segment-specific ACF lags
    _essential = [1, 3, 12] if _is_monthly else [1, 4, 52]
    if segment_id and str(segment_id) in insights.acf_optimal_lag_by_segment:
        seg_lags = insights.acf_optimal_lag_by_segment[str(segment_id)]
        # Merge with essential lags
        merged = sorted(set(seg_lags + _essential))
        return merged[:12]  # Limit to 12 lags

    # Priority 2: Global ACF significant lags
    if insights.acf_significant_lags:
        merged = sorted(set(insights.acf_significant_lags + _essential))
        return merged[:12]

    # Priority 3: Demand pattern-based defaults
    if demand_pattern:
        pattern = demand_pattern.lower()
        if _is_monthly:
            if pattern == 'smooth':
                return [1, 2, 3, 6, 12]
            elif pattern == 'erratic':
                return [1, 2, 3, 6, 12, 24]  # Include longer history
            elif pattern in ['intermittent', 'lumpy']:
                return [1, 3, 6, 12]  # Skip short lags that may be zero
        else:
            if pattern == 'smooth':
                return [1, 2, 4, 8, 13, 26, 52]
            elif pattern == 'erratic':
                return [1, 2, 4, 8, 13, 26, 52, 104]  # Include longer history
            elif pattern in ['intermittent', 'lumpy']:
                return [1, 4, 13, 26, 52]  # Skip short lags that may be zero

    return default_lags


def get_adaptive_rolling_config(
    insights: EDAFeatureInsights,
    segment_id: Optional[str] = None,
    demand_pattern: Optional[str] = None,
    time_format: str = 'year_week',
) -> Tuple[List[int], List[str]]:
    """
    Get adaptive rolling window configuration based on EDA insights.

    Parameters
    ----------
    insights : EDAFeatureInsights
        EDA insights object
    segment_id : str, optional
        Segment ID for segment-specific config
    demand_pattern : str, optional
        Demand pattern

    Returns
    -------
    Tuple[List[int], List[str]]
        (rolling windows, rolling statistics)
    """
    # Base windows from insights (time-format-aware defaults)
    _is_monthly = (time_format == 'year_month')
    _default_windows = [3, 6, 12] if _is_monthly else [4, 13, 26, 52]
    windows = insights.recommended_rolling_windows or _default_windows

    # Adapt based on demand pattern
    if demand_pattern:
        pattern = demand_pattern.lower()
        if _is_monthly:
            if pattern == 'smooth':
                windows = [3, 6, 12]
                stats = ['mean', 'std', 'min', 'max']
            elif pattern == 'erratic':
                windows = [3, 6, 12]
                stats = ['mean', 'std', 'min', 'max', 'median']
            elif pattern in ['intermittent', 'lumpy']:
                windows = [3, 6, 12]
                stats = ['mean', 'std', 'sum']
            else:
                stats = ['mean', 'std', 'min', 'max']
        else:
            if pattern == 'smooth':
                # Shorter windows for smooth demand
                windows = [4, 8, 13, 26]
                stats = ['mean', 'std', 'min', 'max']
            elif pattern == 'erratic':
                # Medium windows, include volatility measures
                windows = [4, 13, 26, 52]
                stats = ['mean', 'std', 'min', 'max', 'median']
            elif pattern in ['intermittent', 'lumpy']:
                # Longer windows for sparse data stability
                windows = [8, 13, 26, 52]
                stats = ['mean', 'std', 'sum']  # Sum is meaningful for sparse
            else:
                stats = ['mean', 'std', 'min', 'max']
    else:
        stats = ['mean', 'std', 'min', 'max']

    return windows, stats


def create_changepoint_indicator_features(
    df: pd.DataFrame,
    date_col: str,
    changepoint_dates: List[str],
    prefix: str = 'cp',
) -> pd.DataFrame:
    """
    Create indicator features for detected changepoints.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Date column name
    changepoint_dates : List[str]
        List of changepoint dates detected by EDA
    prefix : str
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with changepoint indicator features
    """
    df = df.copy()

    if not changepoint_dates:
        return df

    # Convert date column to comparable format
    try:
        # Handle different date formats
        sample_val = str(df[date_col].iloc[0])
        if len(sample_val) == 6 and sample_val.isdigit():
            # YYYYWW format
            df['_date_numeric'] = df[date_col].astype(int)
        else:
            df['_date_numeric'] = pd.to_datetime(df[date_col]).astype(int) // 10**9

        for i, cp_date in enumerate(changepoint_dates[:5]):  # Limit to 5 changepoints
            try:
                if len(str(cp_date)) == 6:
                    cp_numeric = int(cp_date)
                else:
                    cp_numeric = pd.to_datetime(cp_date).timestamp()

                # Create post-changepoint indicator
                df[f'{prefix}_post_{i+1}'] = (df['_date_numeric'] >= cp_numeric).astype(int)

                # Create time-since-changepoint feature
                df[f'{prefix}_time_since_{i+1}'] = (df['_date_numeric'] - cp_numeric).clip(lower=0)

            except Exception as e:
                logger.warning(f"Error creating changepoint feature for {cp_date}: {e}")

        df = df.drop(columns=['_date_numeric'], errors='ignore')

    except Exception as e:
        logger.warning(f"Error creating changepoint features: {e}")

    return df


def create_eda_driven_features(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    eda_insights: EDAFeatureInsights,
    segment_col: Optional[str] = None,
    forecast_lag: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create features using EDA-driven insights for STATE-OF-THE-ART feature engineering.

    This function adapts feature creation based on what EDA discovered:
    - ACF-informed lag selection
    - Adaptive rolling windows per demand pattern
    - Seasonal features with detected periods
    - Changepoint indicators
    - Trend features for trending series

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    eda_insights : EDAFeatureInsights
        Insights loaded from EDA exhaustive analysis
    segment_col : str, optional
        Segment column for segment-specific features
    forecast_lag : int
        Forecast horizon (for leakage prevention)

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - DataFrame with EDA-driven features
        - Feature creation metadata
    """
    df = df.copy()
    metadata = {
        'eda_driven': True,
        'features_created': [],
        'lag_config': {},
        'rolling_config': {},
        'seasonal_config': {},
        'changepoint_features': [],
    }

    key_col = key_cols[0] if len(key_cols) == 1 else tuple(key_cols)

    # Ensure sorted by date within keys
    df = df.sort_values(key_cols + [date_col])

    # =========================================================================
    # 1. ACF-INFORMED LAG FEATURES
    # =========================================================================
    adaptive_lags = get_adaptive_lag_config(eda_insights)
    metadata['lag_config']['adaptive_lags'] = adaptive_lags

    # Adjust lags for forecast horizon
    safe_lags = [lag for lag in adaptive_lags if lag >= forecast_lag]
    if not safe_lags:
        safe_lags = [forecast_lag, forecast_lag + 1, forecast_lag + 4]

    df = create_lag_features(df, key_cols, target_col, lags=safe_lags)
    metadata['features_created'].extend([f'{target_col}_lag_{lag}' for lag in safe_lags])

    # =========================================================================
    # 2. ADAPTIVE ROLLING FEATURES
    # =========================================================================
    windows, stats = get_adaptive_rolling_config(eda_insights)
    metadata['rolling_config'] = {'windows': windows, 'stats': stats}

    df = create_rolling_features(
        df, key_cols, target_col,
        windows=windows,
        aggregations=stats,
        shift_periods=forecast_lag,
    )

    # =========================================================================
    # 3. SEASONAL FEATURES (with detected period)
    # =========================================================================
    if eda_insights.has_seasonality_pct > 0.2:
        seasonal_period = eda_insights.detected_seasonal_period
        metadata['seasonal_config'] = {
            'period': seasonal_period,
            'fourier_order': eda_insights.recommended_fourier_order,
        }

        # Fourier features with adaptive order
        df = create_fourier_features(
            df, date_col,
            period=seasonal_period,
            order=eda_insights.recommended_fourier_order,
        )

        # Seasonal lag
        if seasonal_period >= forecast_lag:
            df = create_seasonal_lag_features(
                df, key_cols, target_col,
                seasonal_periods=[seasonal_period],
            )

    # =========================================================================
    # 4. CHANGEPOINT INDICATOR FEATURES
    # =========================================================================
    if eda_insights.needs_changepoint_indicators and eda_insights.changepoint_dates:
        df = create_changepoint_indicator_features(
            df, date_col, eda_insights.changepoint_dates
        )
        metadata['changepoint_features'] = [
            col for col in df.columns if col.startswith('cp_')
        ]

    # =========================================================================
    # 5. TREND FEATURES (if significant trending detected)
    # =========================================================================
    if eda_insights.needs_trend_features:
        df = create_trend_features(df, key_cols, date_col, target_col)

    # =========================================================================
    # 6. INTERMITTENCY FEATURES (if needed)
    # =========================================================================
    if eda_insights.needs_intermittency_features:
        df = create_intermittency_features(df, key_cols, target_col)
        # Also add advanced sparsity features
        df = create_sparsity_features(df, key_cols, target_col)

    return df, metadata


def enrich_features_with_eda_insights(
    df: pd.DataFrame,
    eda_dir: str,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    segment_col: Optional[str] = None,
    forecast_lag: int = 1,
) -> Tuple[pd.DataFrame, EDAFeatureInsights, Dict[str, Any]]:
    """
    MAIN ENTRY POINT: Enrich feature engineering with exhaustive EDA insights.

    This is the STATE-OF-THE-ART function that:
    1. Loads all EDA exhaustive analyses (ACF, seasonality, trend, changepoints)
    2. Adapts feature creation based on discovered patterns
    3. Returns enriched features with metadata

    Use this function in the Feature Engineering crew to leverage EDA insights.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (raw data or partially processed)
    eda_dir : str
        Path to EDA output directory
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    segment_col : str, optional
        Segment column
    forecast_lag : int
        Forecast horizon

    Returns
    -------
    Tuple[pd.DataFrame, EDAFeatureInsights, Dict[str, Any]]
        - Enriched DataFrame with EDA-driven features
        - EDA insights object
        - Feature creation metadata
    """
    # Load all EDA insights
    insights = load_eda_insights_for_features(eda_dir)

    # Create EDA-driven features
    df_enriched, metadata = create_eda_driven_features(
        df=df,
        key_cols=key_cols,
        date_col=date_col,
        target_col=target_col,
        eda_insights=insights,
        segment_col=segment_col,
        forecast_lag=forecast_lag,
    )

    logger.info(
        f"EDA-enriched features: {len(metadata['features_created'])} lag features, "
        f"windows={metadata['rolling_config'].get('windows', [])}, "
        f"changepoints={len(metadata.get('changepoint_features', []))}"
    )

    return df_enriched, insights, metadata


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'FeatureSpec',
    'FeatureEngineeringResult',

    # Data type validation
    'validate_and_convert_column_types',

    # Lag features
    'create_lag_features',
    'create_lag_combinations',
    'create_momentum_features',
    'create_seasonal_lag_features',

    # Rolling features
    'create_rolling_features',
    'create_ewm_features',
    'create_expanding_features',
    'create_rolling_volatility_features',

    # Calendar & cyclical
    'create_calendar_features',
    'create_cyclical_features',
    'create_fourier_features',
    'create_holiday_features',

    # Trend & decomposition
    'create_trend_features',
    'create_detrended_features',
    'create_stl_features',

    # Intermittency & Sparsity
    'create_intermittency_features',
    'create_sparsity_features',

    # Cross-sectional
    'create_cross_sectional_features',
    'create_key_level_features',

    # Categorical encoding
    'target_encode',
    'frequency_encode',
    'ordinal_encode',
    'hash_encode',
    'encode_all_categoricals',

    # Feature selection
    'select_features_by_correlation',
    'select_features_by_variance',
    'select_features_shap',
    'auto_select_features',

    # Data quality
    'check_feature_quality',
    'fill_missing_features',
    'clip_outliers',

    # High-level orchestration
    'create_complete_feature_set',
    'run_feature_pipeline',

    # LEAKAGE-FREE FUNCTIONS (RECOMMENDED FOR PRODUCTION)
    'create_lag_features_leakage_free',
    'create_rolling_features_leakage_free',
    'create_seasonal_features_leakage_free',
    'create_cross_sectional_features_leakage_free',
    'create_key_level_features_leakage_free',
    'run_leakage_free_feature_pipeline',

    # EDA-DRIVEN FEATURE ENRICHMENT (STATE-OF-THE-ART)
    'EDAFeatureInsights',
    'load_eda_insights_for_features',
    'get_adaptive_lag_config',
    'get_adaptive_rolling_config',
    'create_changepoint_indicator_features',
    'create_eda_driven_features',
    'enrich_features_with_eda_insights',
]
