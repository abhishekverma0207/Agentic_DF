# utils/dead_key_handler.py
"""
Dead Key Detection and Handling for Demand Forecasting
=======================================================

This module implements the "dead key" rule for demand forecasting:

DEFINITION:
- A "dead key" is a product-customer combination that has had NO SALES for
  26 consecutive periods (weeks) at the END of the training period.

RULES:
1. For each key, only keep history up to 26 periods after its last non-zero sale
2. Keys that end training period with 26+ consecutive zeros are marked as "dead"
3. Dead keys are EXCLUDED from: EDA analysis, Segmentation, Feature Engineering, Model Training
4. Dead keys are INCLUDED in final forecasts with prediction = 0

This prevents dead keys from:
- Skewing demand pattern distributions (inflating "lumpy" counts)
- Wasting compute on training models for inactive keys
- Affecting model accuracy metrics

USAGE:
    from utils.dead_key_handler import (
        identify_dead_keys,
        filter_dead_keys_from_df,
        create_dead_key_forecasts,
        DeadKeyConfig,
        DeadKeySummary,
    )

    # Identify dead keys
    dead_keys, summary = identify_dead_keys(
        df, key_col='key', date_col='year_week', target_col='PS_actual_sales',
        train_end='202521', consecutive_zero_threshold=26
    )

    # Filter them out for analysis
    active_df = filter_dead_keys_from_df(df, dead_keys, key_col='key')

    # Create zero forecasts for dead keys
    dead_forecasts = create_dead_key_forecasts(dead_keys, forecast_periods)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DeadKeyConfig:
    """Configuration for dead key detection."""
    # Number of consecutive zero periods to consider a key "dead"
    consecutive_zero_threshold: int = 26

    # Only consider keys dead if zeros are at the END of training period
    require_trailing_zeros: bool = True

    # Maximum history to keep after last non-zero sale (for truncation)
    max_history_after_last_sale: int = 26

    # Whether to save dead key list to file
    save_dead_key_list: bool = True

    # Column names (can be overridden)
    key_col: str = 'key'
    date_col: str = 'year_week'
    target_col: str = 'PS_actual_sales'


@dataclass
class DeadKeySummary:
    """Summary statistics for dead key detection."""
    total_keys: int
    dead_keys: int
    active_keys: int
    dead_key_percentage: float
    total_rows_original: int
    total_rows_after_truncation: int
    rows_removed: int
    rows_removed_percentage: float
    dead_key_list: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_keys': self.total_keys,
            'dead_keys': self.dead_keys,
            'active_keys': self.active_keys,
            'dead_key_percentage': round(self.dead_key_percentage, 2),
            'total_rows_original': self.total_rows_original,
            'total_rows_after_truncation': self.total_rows_after_truncation,
            'rows_removed': self.rows_removed,
            'rows_removed_percentage': round(self.rows_removed_percentage, 2),
            'dead_key_count': len(self.dead_key_list),
            # Don't include full list in dict (can be very long)
        }

    def __str__(self) -> str:
        return (
            f"Dead Key Summary:\n"
            f"  Total keys: {self.total_keys}\n"
            f"  Dead keys: {self.dead_keys} ({self.dead_key_percentage:.1f}%)\n"
            f"  Active keys: {self.active_keys}\n"
            f"  Rows removed: {self.rows_removed} ({self.rows_removed_percentage:.1f}%)\n"
        )


# =============================================================================
# CORE DETECTION FUNCTIONS
# =============================================================================

def identify_dead_keys(
    df: pd.DataFrame,
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    train_end: Optional[Union[str, int]] = None,
    consecutive_zero_threshold: int = 26,
) -> Tuple[Set[str], DeadKeySummary]:
    """
    Identify "dead keys" - keys with N consecutive zeros at end of training period.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    key_col : str
        Column name for the key identifier
    date_col : str
        Column name for the date/period
    target_col : str
        Column name for the target variable (sales)
    train_end : str or int, optional
        End of training period. If None, uses max date in data
    consecutive_zero_threshold : int
        Number of consecutive zeros to consider a key "dead"

    Returns
    -------
    Tuple[Set[str], DeadKeySummary]
        Set of dead key identifiers and summary statistics
    """
    if df.empty:
        return set(), DeadKeySummary(0, 0, 0, 0.0, 0, 0, 0, 0.0, [])

    # Ensure date column is sortable
    df = df.copy()
    df[date_col] = df[date_col].astype(str)

    # Filter to training period if specified
    if train_end is not None:
        train_end = str(train_end)
        df_train = df[df[date_col] <= train_end].copy()
    else:
        df_train = df.copy()

    if df_train.empty:
        all_keys = set(df[key_col].unique())
        return set(), DeadKeySummary(len(all_keys), 0, len(all_keys), 0.0, len(df), len(df), 0, 0.0, [])

    # Sort by key and date
    df_train = df_train.sort_values([key_col, date_col])

    dead_keys = set()
    all_keys = set(df_train[key_col].unique())

    for key in all_keys:
        key_data = df_train[df_train[key_col] == key].sort_values(date_col)

        if len(key_data) == 0:
            continue

        # Get target values
        values = key_data[target_col].fillna(0).values

        # Count trailing zeros
        trailing_zeros = 0
        for val in reversed(values):
            if val == 0 or pd.isna(val):
                trailing_zeros += 1
            else:
                break

        # Mark as dead if enough trailing zeros
        if trailing_zeros >= consecutive_zero_threshold:
            dead_keys.add(key)

    # Calculate summary
    total_keys = len(all_keys)
    dead_count = len(dead_keys)
    active_count = total_keys - dead_count
    dead_pct = (dead_count / total_keys * 100) if total_keys > 0 else 0.0

    # Row counts (original vs after removing dead key data)
    original_rows = len(df)
    rows_after = len(df[~df[key_col].isin(dead_keys)])
    rows_removed = original_rows - rows_after
    rows_removed_pct = (rows_removed / original_rows * 100) if original_rows > 0 else 0.0

    summary = DeadKeySummary(
        total_keys=total_keys,
        dead_keys=dead_count,
        active_keys=active_count,
        dead_key_percentage=dead_pct,
        total_rows_original=original_rows,
        total_rows_after_truncation=rows_after,
        rows_removed=rows_removed,
        rows_removed_percentage=rows_removed_pct,
        dead_key_list=list(dead_keys),
    )

    logger.info(f"Dead key detection: {dead_count}/{total_keys} keys are dead ({dead_pct:.1f}%)")

    return dead_keys, summary


def truncate_key_history(
    df: pd.DataFrame,
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    max_periods_after_last_sale: int = 26,
) -> pd.DataFrame:
    """
    Truncate each key's history to max N periods after its last non-zero sale.

    This removes irrelevant trailing zeros that add noise to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for target variable
    max_periods_after_last_sale : int
        Maximum periods to keep after last non-zero sale

    Returns
    -------
    pd.DataFrame
        Truncated dataframe
    """
    if df.empty:
        return df

    df = df.copy()
    df[date_col] = df[date_col].astype(str)
    df = df.sort_values([key_col, date_col])

    result_dfs = []

    for key in df[key_col].unique():
        key_data = df[df[key_col] == key].copy()

        # Find last non-zero sale
        non_zero_mask = key_data[target_col].fillna(0) > 0

        if non_zero_mask.any():
            last_sale_idx = key_data[non_zero_mask].index[-1]
            last_sale_pos = key_data.index.get_loc(last_sale_idx)

            # Keep up to max_periods_after_last_sale after last sale
            cutoff_pos = min(last_sale_pos + max_periods_after_last_sale + 1, len(key_data))
            key_data = key_data.iloc[:cutoff_pos]

        result_dfs.append(key_data)

    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    return df.iloc[:0]  # Return empty df with same structure


def filter_dead_keys_from_df(
    df: pd.DataFrame,
    dead_keys: Set[str],
    key_col: str = 'key',
) -> pd.DataFrame:
    """
    Remove dead keys from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    dead_keys : Set[str]
        Set of dead key identifiers
    key_col : str
        Column name for key identifier

    Returns
    -------
    pd.DataFrame
        Dataframe with dead keys removed
    """
    if not dead_keys:
        return df
    return df[~df[key_col].isin(dead_keys)].copy()


def get_active_keys(
    df: pd.DataFrame,
    dead_keys: Set[str],
    key_col: str = 'key',
) -> Set[str]:
    """Get set of active (non-dead) keys."""
    all_keys = set(df[key_col].unique())
    return all_keys - dead_keys


# =============================================================================
# FORECAST GENERATION FOR DEAD KEYS
# =============================================================================

def create_dead_key_forecasts(
    dead_keys: Set[str],
    forecast_periods: List[Union[str, int]],
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    prediction_col: str = 'prediction',
) -> pd.DataFrame:
    """
    Create zero forecasts for dead keys.

    Parameters
    ----------
    dead_keys : Set[str]
        Set of dead key identifiers
    forecast_periods : List[str or int]
        List of forecast periods (e.g., ['202529', '202530', ...])
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for actual target (will be NaN for forecasts)
    prediction_col : str
        Column name for predictions

    Returns
    -------
    pd.DataFrame
        Dataframe with zero forecasts for all dead keys and periods
    """
    if not dead_keys or not forecast_periods:
        return pd.DataFrame(columns=[key_col, date_col, target_col, prediction_col, 'is_dead_key'])

    records = []
    for key in dead_keys:
        for period in forecast_periods:
            records.append({
                key_col: key,
                date_col: str(period),
                target_col: np.nan,  # Actual not known yet
                prediction_col: 0.0,  # Always predict 0 for dead keys
                'is_dead_key': True,
            })

    return pd.DataFrame(records)


def merge_dead_key_forecasts(
    model_forecasts: pd.DataFrame,
    dead_key_forecasts: pd.DataFrame,
    key_col: str = 'key',
    date_col: str = 'year_week',
) -> pd.DataFrame:
    """
    Merge model forecasts with dead key zero forecasts.

    Parameters
    ----------
    model_forecasts : pd.DataFrame
        Forecasts from trained models (active keys only)
    dead_key_forecasts : pd.DataFrame
        Zero forecasts for dead keys
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period

    Returns
    -------
    pd.DataFrame
        Combined forecasts with is_dead_key column
    """
    if model_forecasts.empty and dead_key_forecasts.empty:
        return pd.DataFrame()

    # Add is_dead_key column to model forecasts if not present
    if 'is_dead_key' not in model_forecasts.columns:
        model_forecasts = model_forecasts.copy()
        model_forecasts['is_dead_key'] = False

    # Concatenate
    combined = pd.concat([model_forecasts, dead_key_forecasts], ignore_index=True)

    # Sort by key and date
    combined = combined.sort_values([key_col, date_col])

    return combined


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def save_dead_key_summary(
    summary: DeadKeySummary,
    output_path: str,
) -> None:
    """Save dead key summary to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)
    logger.info(f"Saved dead key summary to {output_path}")


def save_dead_key_list(
    dead_keys: Set[str],
    output_path: str,
) -> None:
    """Save dead key list to text file (one key per line)."""
    with open(output_path, 'w') as f:
        for key in sorted(dead_keys):
            f.write(f"{key}\n")
    logger.info(f"Saved {len(dead_keys)} dead keys to {output_path}")


def load_dead_key_list(
    input_path: str,
) -> Set[str]:
    """Load dead key list from text file."""
    dead_keys = set()
    try:
        with open(input_path, 'r') as f:
            for line in f:
                key = line.strip()
                if key:
                    dead_keys.add(key)
        logger.info(f"Loaded {len(dead_keys)} dead keys from {input_path}")
    except FileNotFoundError:
        logger.warning(f"Dead key file not found: {input_path}")
    return dead_keys


def load_dead_key_summary(
    input_path: str,
) -> Optional[Dict[str, Any]]:
    """Load dead key summary from JSON file."""
    try:
        with open(input_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Dead key summary not found: {input_path}")
        return None


# =============================================================================
# CONVENIENCE FUNCTION FOR FULL PIPELINE
# =============================================================================

def process_dead_keys_for_eda(
    df: pd.DataFrame,
    config: DeadKeyConfig,
    train_end: Optional[Union[str, int]] = None,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Set[str], DeadKeySummary]:
    """
    Full dead key processing for EDA phase.

    1. Identifies dead keys
    2. Optionally truncates history
    3. Filters dead keys from analysis data
    4. Saves summary and list if output_dir provided

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data
    config : DeadKeyConfig
        Configuration for dead key detection
    train_end : str or int, optional
        End of training period
    output_dir : str, optional
        Directory to save dead key files

    Returns
    -------
    Tuple[pd.DataFrame, Set[str], DeadKeySummary]
        - Filtered dataframe (active keys only, for analysis)
        - Set of dead keys
        - Summary statistics
    """
    import os

    # Step 1: Identify dead keys
    dead_keys, summary = identify_dead_keys(
        df,
        key_col=config.key_col,
        date_col=config.date_col,
        target_col=config.target_col,
        train_end=train_end,
        consecutive_zero_threshold=config.consecutive_zero_threshold,
    )

    # Step 2: Truncate history (optional - removes trailing zeros beyond threshold)
    if config.max_history_after_last_sale > 0:
        df_truncated = truncate_key_history(
            df,
            key_col=config.key_col,
            date_col=config.date_col,
            target_col=config.target_col,
            max_periods_after_last_sale=config.max_history_after_last_sale,
        )
    else:
        df_truncated = df

    # Step 3: Filter out dead keys for analysis
    df_active = filter_dead_keys_from_df(
        df_truncated,
        dead_keys,
        key_col=config.key_col,
    )

    # Step 4: Save outputs if directory provided
    if output_dir and config.save_dead_key_list:
        os.makedirs(output_dir, exist_ok=True)
        save_dead_key_summary(summary, os.path.join(output_dir, 'dead_key_summary.json'))
        save_dead_key_list(dead_keys, os.path.join(output_dir, 'dead_keys.txt'))

    return df_active, dead_keys, summary


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_no_dead_keys_in_training(
    df: pd.DataFrame,
    dead_keys: Set[str],
    key_col: str = 'key',
    raise_error: bool = True,
) -> bool:
    """
    Validate that no dead keys are present in training data.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe
    dead_keys : Set[str]
        Set of dead keys
    key_col : str
        Column name for key identifier
    raise_error : bool
        Whether to raise error if dead keys found

    Returns
    -------
    bool
        True if no dead keys found
    """
    if not dead_keys:
        return True

    keys_in_df = set(df[key_col].unique())
    overlap = keys_in_df & dead_keys

    if overlap:
        msg = f"Found {len(overlap)} dead keys in training data: {list(overlap)[:5]}..."
        if raise_error:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    return True


# =============================================================================
# DYNAMIC/ROLLING DEAD KEY DETECTION
# =============================================================================

def is_key_dead_at_period(
    df: pd.DataFrame,
    key: str,
    as_of_period: Union[str, int],
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    consecutive_zero_threshold: int = 26,
) -> bool:
    """
    Check if a specific key is "dead" as of a given period.

    A key is dead if it has N consecutive zeros ending at or before the given period.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data
    key : str
        The key to check
    as_of_period : str or int
        The period to evaluate dead status at
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for target variable
    consecutive_zero_threshold : int
        Number of consecutive zeros to be considered dead

    Returns
    -------
    bool
        True if key is dead as of the given period
    """
    # Get data for this key up to the given period
    as_of_str = str(as_of_period)
    key_data = df[(df[key_col] == key) & (df[date_col].astype(str) <= as_of_str)]

    if len(key_data) == 0:
        return False  # No data, not considered dead

    # Sort by date and get target values
    key_data = key_data.sort_values(date_col)
    values = key_data[target_col].fillna(0).values

    # Count trailing zeros
    trailing_zeros = 0
    for val in reversed(values):
        if val == 0:
            trailing_zeros += 1
        else:
            break

    return trailing_zeros >= consecutive_zero_threshold


def get_dead_keys_at_period(
    df: pd.DataFrame,
    as_of_period: Union[str, int],
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    consecutive_zero_threshold: int = 26,
) -> Set[str]:
    """
    Get all dead keys as of a specific period.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data
    as_of_period : str or int
        The period to evaluate dead status at
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for target variable
    consecutive_zero_threshold : int
        Number of consecutive zeros to be considered dead

    Returns
    -------
    Set[str]
        Set of keys that are dead as of the given period
    """
    as_of_str = str(as_of_period)
    df_filtered = df[df[date_col].astype(str) <= as_of_str]

    if df_filtered.empty:
        return set()

    all_keys = df_filtered[key_col].unique()
    dead_keys = set()

    for key in all_keys:
        if is_key_dead_at_period(
            df, key, as_of_period,
            key_col=key_col, date_col=date_col, target_col=target_col,
            consecutive_zero_threshold=consecutive_zero_threshold
        ):
            dead_keys.add(key)

    return dead_keys


def compute_dead_key_status_by_period(
    df: pd.DataFrame,
    periods: List[Union[str, int]],
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    consecutive_zero_threshold: int = 26,
) -> Dict[str, Set[str]]:
    """
    Compute dead key status for each period (rolling/dynamic).

    This is useful for forecasting where dead key status changes over time.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data
    periods : List[str or int]
        List of periods to evaluate
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for target variable
    consecutive_zero_threshold : int
        Number of consecutive zeros to be considered dead

    Returns
    -------
    Dict[str, Set[str]]
        Dictionary mapping period -> set of dead keys at that period
    """
    dead_keys_by_period = {}

    for period in periods:
        period_str = str(period)
        dead_keys_by_period[period_str] = get_dead_keys_at_period(
            df, period,
            key_col=key_col, date_col=date_col, target_col=target_col,
            consecutive_zero_threshold=consecutive_zero_threshold
        )

    return dead_keys_by_period


def create_dynamic_dead_key_forecasts(
    df: pd.DataFrame,
    forecast_periods: List[Union[str, int]],
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    consecutive_zero_threshold: int = 26,
) -> pd.DataFrame:
    """
    Create zero forecasts for keys that are dead at each forecast period.

    This handles the dynamic nature of dead keys - a key may become dead
    during the forecast period.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data (used to determine dead status)
    forecast_periods : List[str or int]
        List of forecast periods
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for target variable
    consecutive_zero_threshold : int
        Number of consecutive zeros to be considered dead

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: key, date, prediction, is_dead_key, became_dead_at
    """
    if df.empty or not forecast_periods:
        return pd.DataFrame(columns=[key_col, date_col, 'prediction', 'is_dead_key', 'became_dead_at'])

    # Get all keys
    all_keys = set(df[key_col].unique())

    # Compute dead status at each period
    dead_by_period = compute_dead_key_status_by_period(
        df, forecast_periods,
        key_col=key_col, date_col=date_col, target_col=target_col,
        consecutive_zero_threshold=consecutive_zero_threshold
    )

    records = []
    for period in forecast_periods:
        period_str = str(period)
        dead_keys = dead_by_period.get(period_str, set())

        for key in dead_keys:
            # Find when this key became dead (first period where it was dead)
            became_dead_at = None
            for p in sorted(dead_by_period.keys()):
                if key in dead_by_period[p]:
                    became_dead_at = p
                    break

            records.append({
                key_col: key,
                date_col: period_str,
                'prediction': 0.0,
                'is_dead_key': True,
                'became_dead_at': became_dead_at,
            })

    return pd.DataFrame(records)


def get_key_status_at_period(
    df: pd.DataFrame,
    key: str,
    as_of_period: Union[str, int],
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    consecutive_zero_threshold: int = 26,
) -> Dict[str, Any]:
    """
    Get detailed status for a key at a specific period.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data
    key : str
        The key to check
    as_of_period : str or int
        The period to evaluate at
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for target variable
    consecutive_zero_threshold : int
        Number of consecutive zeros to be considered dead

    Returns
    -------
    Dict[str, Any]
        Status dictionary with keys: is_dead, trailing_zeros, last_sale_period, periods_since_last_sale
    """
    as_of_str = str(as_of_period)
    key_data = df[(df[key_col] == key) & (df[date_col].astype(str) <= as_of_str)]

    if len(key_data) == 0:
        return {
            'is_dead': False,
            'trailing_zeros': 0,
            'last_sale_period': None,
            'periods_since_last_sale': None,
            'total_periods': 0,
        }

    key_data = key_data.sort_values(date_col)
    values = key_data[target_col].fillna(0).values
    periods = key_data[date_col].values

    # Count trailing zeros
    trailing_zeros = 0
    for val in reversed(values):
        if val == 0:
            trailing_zeros += 1
        else:
            break

    # Find last non-zero sale
    non_zero_mask = values > 0
    if non_zero_mask.any():
        last_sale_idx = np.where(non_zero_mask)[0][-1]
        last_sale_period = str(periods[last_sale_idx])
        periods_since_last_sale = len(values) - last_sale_idx - 1
    else:
        last_sale_period = None
        periods_since_last_sale = len(values)  # All zeros

    return {
        'is_dead': trailing_zeros >= consecutive_zero_threshold,
        'trailing_zeros': trailing_zeros,
        'last_sale_period': last_sale_period,
        'periods_since_last_sale': periods_since_last_sale,
        'total_periods': len(values),
    }


def filter_predictions_for_dead_keys(
    predictions_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    key_col: str = 'key',
    date_col: str = 'year_week',
    target_col: str = 'PS_actual_sales',
    pred_col: str = 'prediction',
    consecutive_zero_threshold: int = 26,
) -> pd.DataFrame:
    """
    Post-process predictions to set dead key predictions to 0.

    This is useful when you have predictions from a model but want to
    override dead key predictions to 0.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Dataframe with predictions (key, date, prediction columns)
    historical_df : pd.DataFrame
        Historical data used to determine dead status
    key_col : str
        Column name for key identifier
    date_col : str
        Column name for date/period
    target_col : str
        Column name for target variable
    pred_col : str
        Column name for predictions
    consecutive_zero_threshold : int
        Number of consecutive zeros to be considered dead

    Returns
    -------
    pd.DataFrame
        Predictions with dead key predictions set to 0
    """
    if predictions_df.empty:
        return predictions_df

    result = predictions_df.copy()
    result['is_dead_key'] = False

    # Get unique prediction periods
    pred_periods = result[date_col].unique()

    # For each period, determine which keys are dead
    for period in pred_periods:
        period_str = str(period)

        # Get dead keys as of this period (looking at history up to the period before)
        dead_keys = get_dead_keys_at_period(
            historical_df, period,
            key_col=key_col, date_col=date_col, target_col=target_col,
            consecutive_zero_threshold=consecutive_zero_threshold
        )

        if dead_keys:
            # Set predictions to 0 for dead keys at this period
            mask = (result[date_col].astype(str) == period_str) & (result[key_col].isin(dead_keys))
            result.loc[mask, pred_col] = 0.0
            result.loc[mask, 'is_dead_key'] = True

    logger.info(f"Post-processed {result['is_dead_key'].sum()} predictions for dead keys (set to 0)")

    return result


def save_dead_keys_by_period(
    dead_keys_by_period: Dict[str, Set[str]],
    output_path: str,
) -> None:
    """Save dead keys by period to JSON file."""
    # Convert sets to lists for JSON serialization
    serializable = {k: list(v) for k, v in dead_keys_by_period.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved dead keys by period to {output_path}")


def load_dead_keys_by_period(
    input_path: str,
) -> Dict[str, Set[str]]:
    """Load dead keys by period from JSON file."""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()}
    except FileNotFoundError:
        logger.warning(f"Dead keys by period file not found: {input_path}")
        return {}
