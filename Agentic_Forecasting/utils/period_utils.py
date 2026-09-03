# utils/period_utils.py
"""
Shared utilities for period/date handling across the pipeline.

Handles ALL common date/period formats:
- YYYY-WW (dash-separated week, e.g., "2025-9" → "2025-09")
- YYYYWW (numeric, e.g., 202509)
- YYYY-MM (dash-separated month, e.g., "2025-3" → "2025-03")
- YYYYMM (numeric, e.g., 202503)
- Real dates: "2025-01-15", "01/15/2025", "15-Jan-2025" → converted to YYYY-WW or YYYY-MM
- Pandas Timestamps → converted to YYYY-WW or YYYY-MM

The system auto-detects the format and normalises everything to a comparable string.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


def normalise_period(period: Any) -> str:
    """
    Normalise a period value to a zero-padded string safe for lexicographic comparison.

    Examples:
        "2025-9"   -> "2025-09"
        "2025-10"  -> "2025-10"
        202509     -> "202509"
        "2025-52"  -> "2025-52"
        "2023-10"  -> "2023-10"

    Parameters
    ----------
    period : Any
        Period value (string, int, float)

    Returns
    -------
    str
        Normalised string suitable for comparison
    """
    s = str(period).strip()

    # Numeric without separator (YYYYWW or YYYYMM)
    try:
        v = int(float(s))
        return str(v)
    except (ValueError, TypeError):
        pass

    # Dash-separated: YYYY-W or YYYY-WW or YYYY-M or YYYY-MM
    if '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            year = parts[0]
            sub = parts[1].zfill(2)  # Zero-pad: "9" -> "09"
            return f"{year}-{sub}"

    # Fallback: return as-is
    return s


def detect_and_convert_date_format(
    df: pd.DataFrame,
    col: str,
    target_format: str = 'year_week',
) -> Tuple[pd.DataFrame, str]:
    """
    Auto-detect date format and convert to standard period format.

    Handles:
    - Already YYYY-WW or YYYYWW → normalise only
    - Already YYYY-MM or YYYYMM → normalise only
    - Real dates (2025-01-15, Timestamps) → convert to YYYY-WW or YYYY-MM
    - String dates (01/15/2025, 15-Jan-2025) → parse then convert

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify
    col : str
        Date/period column
    target_format : str
        'year_week' or 'year_month'

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (modified DataFrame, detected_format description)
    """
    if col not in df.columns:
        return df, 'column_missing'

    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
    if sample is None:
        return df, 'empty'

    # Already numeric (YYYYWW or YYYYMM)
    if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
        return df, 'numeric_period'

    s = str(sample)

    # Check if it's already YYYY-WW format (e.g., "2025-09", "2025-3")
    if '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            # Already YYYY-WW or YYYY-MM — just normalise
            return df, 'dash_period'

        # Could be a real date: 2025-01-15
        if len(parts) == 3:
            try:
                import pandas as pd_internal
                dates = pd_internal.to_datetime(df[col], errors='coerce')
                n_valid = dates.notna().sum()
                if n_valid > len(df) * 0.5:
                    # Successfully parsed as dates — convert to period format
                    if target_format == 'year_week':
                        df[col] = dates.dt.isocalendar().year.astype(str) + '-' + \
                                  dates.dt.isocalendar().week.astype(str).str.zfill(2)
                    else:
                        df[col] = dates.dt.strftime('%Y-%m')
                    logger.info(f"Converted real dates to {target_format}: {s} → {df[col].iloc[0]}")
                    return df, 'real_date_converted'
            except Exception:
                pass

    # Try parsing as general date string (01/15/2025, 15-Jan-2025, etc.)
    try:
        import pandas as pd_internal
        dates = pd_internal.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
        n_valid = dates.notna().sum()
        if n_valid > len(df) * 0.5:
            if target_format == 'year_week':
                df[col] = dates.dt.isocalendar().year.astype(str) + '-' + \
                          dates.dt.isocalendar().week.astype(str).str.zfill(2)
            else:
                df[col] = dates.dt.strftime('%Y-%m')
            logger.info(f"Parsed and converted dates to {target_format}: {s} → {df[col].iloc[0]}")
            return df, 'parsed_date_converted'
    except Exception:
        pass

    # Check if it's already a Timestamp column
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        if target_format == 'year_week':
            df[col] = df[col].dt.isocalendar().year.astype(str) + '-' + \
                      df[col].dt.isocalendar().week.astype(str).str.zfill(2)
        else:
            df[col] = df[col].dt.strftime('%Y-%m')
        return df, 'timestamp_converted'

    return df, 'unknown_format'


def normalise_period_column(df: pd.DataFrame, col: str, time_format: str = 'year_week') -> pd.DataFrame:
    """
    Normalise an entire period column in-place for safe comparison.

    Handles ALL date formats:
    1. Dash-separated periods (2025-9 → 2025-09)
    2. Real dates (2025-01-15 → 2025-03 as year-week)
    3. Timestamps → converted to year-week or year-month
    4. Numeric periods (202509) → left as-is

    After this call, standard string comparison (<=, >=, sort) works correctly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify
    col : str
        Column name to normalise
    time_format : str
        Target format: 'year_week' or 'year_month'

    Returns
    -------
    pd.DataFrame
        Same DataFrame with normalised column
    """
    if col not in df.columns:
        return df

    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
    if sample is None:
        return df

    # First: try auto-detecting and converting real dates
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        df, fmt = detect_and_convert_date_format(df, col, time_format)
        return df

    s = str(sample)

    # Check if it's a real date (has 3 parts with dash, or contains /)
    if ('-' in s and len(s.split('-')) == 3) or '/' in s:
        df, fmt = detect_and_convert_date_format(df, col, time_format)
        if fmt.endswith('converted'):
            # Apply normalisation to the converted values
            df[col] = df[col].astype(str).apply(normalise_period)
            return df

    # Standard normalisation for dash-separated periods
    if '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            df[col] = df[col].astype(str).apply(normalise_period)

    return df


def period_le(a: Any, b: Any) -> bool:
    """Compare two periods: a <= b, handling YYYY-WW correctly."""
    return normalise_period(a) <= normalise_period(b)


def period_gt(a: Any, b: Any) -> bool:
    """Compare two periods: a > b, handling YYYY-WW correctly."""
    return normalise_period(a) > normalise_period(b)


def sort_periods(periods: List[Any]) -> List[Any]:
    """Sort periods correctly regardless of format."""
    return sorted(periods, key=lambda p: normalise_period(p))


def _validate_data_quality(df, cfg):
    """Pre-flight data quality checks. Warns or raises on issues."""
    issues = []

    # Check target has valid values
    target = df[cfg.target_col]
    if target.isna().all():
        raise ValueError(f"Target column '{cfg.target_col}' is ALL NaN")
    if (target < 0).any():
        n_neg = (target < 0).sum()
        issues.append(f"WARNING: {n_neg} negative values in target '{cfg.target_col}' — clipping to 0")
        df[cfg.target_col] = target.clip(lower=0)

    # Check sufficient history
    n_periods = df[cfg.timestamp_col].nunique()
    if n_periods < 10:
        raise ValueError(f"Only {n_periods} unique periods — need at least 10 for forecasting")

    # Check for missing periods (gaps in time series)
    periods = sorted(df[cfg.timestamp_col].unique())
    # Can't check for gaps without knowing the period type, but log the range
    key_col = cfg.prediction_key_cols[0]
    n_keys = df[key_col].nunique()
    expected_rows = n_keys * n_periods
    actual_rows = len(df)
    completeness = actual_rows / expected_rows if expected_rows > 0 else 0
    if completeness < 0.5:
        issues.append(f"WARNING: Data is only {completeness:.0%} complete ({actual_rows:,} rows vs {expected_rows:,} expected for {n_keys} keys × {n_periods} periods)")

    # Check for high cardinality categoricals
    for col in df.select_dtypes(include=['object']).columns:
        n_unique = df[col].nunique()
        if n_unique > 10000:
            issues.append(f"WARNING: Column '{col}' has {n_unique:,} unique values — may cause memory issues with encoding")

    for issue in issues:
        logger.warning(issue)


def bootstrap_config(config_yaml_path: str):
    """
    Load config, auto-detect splits if empty, and normalise source data periods.

    Call this at the top of any standalone runner (run_eda.py, run_feature.py, etc.)
    to ensure consistent state regardless of whether the full pipeline or a single
    stage is being run.

    Returns
    -------
    Tuple[DemandForecastConfig, pd.DataFrame]
        (config, source_df with normalised periods)
    """
    import logging
    logger = logging.getLogger(__name__)

    from config.schema import load_config_from_yaml
    cfg = load_config_from_yaml(config_yaml_path)

    # Load source data with validation
    from utils.agent_utilities import load_source_data
    import os
    if not os.path.exists(cfg.input_data_path):
        raise FileNotFoundError(
            f"Data file not found: {cfg.input_data_path}\n"
            f"Check input_data_path in {config_yaml_path}"
        )
    source_df = load_source_data(cfg.input_data_path)

    # Validate required columns exist
    for col in cfg.prediction_key_cols + [cfg.timestamp_col, cfg.target_col]:
        if col not in source_df.columns:
            raise ValueError(
                f"Required column '{col}' not found in data. "
                f"Available columns: {list(source_df.columns)[:20]}"
            )

    # Auto-detect date format and convert if needed (handles real dates → YYYY-WW)
    target_fmt = 'year_week' if cfg.time_format in ('year_week', 'auto') else 'year_month'
    source_df, detected_fmt = detect_and_convert_date_format(source_df, cfg.timestamp_col, target_fmt)
    if detected_fmt.endswith('converted'):
        logger.info(f"Date format auto-detected and converted: {detected_fmt}")

    # Normalise period column for safe comparison
    source_df = normalise_period_column(source_df, cfg.timestamp_col, target_fmt)

    # Pre-flight data validation
    _validate_data_quality(source_df, cfg)

    # Auto-detect splits if not specified
    if not all([cfg.train_start, cfg.train_end, cfg.val_start, cfg.val_end]):
        logger.info("Auto-detecting train/val/test splits from data...")
        cfg.auto_detect_splits(source_df)
        logger.info(f"Splits: train={cfg.train_start}..{cfg.train_end}, "
                     f"val={cfg.val_start}..{cfg.val_end}, "
                     f"test={cfg.test_start}..{cfg.test_end}")

    # Enforce run_mode guards
    if hasattr(cfg, 'run_mode'):
        logger.info(f"Run mode: {cfg.run_mode} "
                     f"(backtest={cfg.should_backtest}, forecast={cfg.should_forward_forecast})")

    return cfg, source_df
