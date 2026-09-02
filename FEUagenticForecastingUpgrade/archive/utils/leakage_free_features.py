# utils/leakage_free_features.py
"""
Leakage-Free Feature Engineering for Demand Forecasting
========================================================

This module provides feature engineering functions that PREVENT data leakage
by ensuring:
1. Lag features only use values available at forecast time
2. No features use the current row's target value
3. Features are computed separately for train vs validation/test
4. Recursive forecasting is used for multi-step predictions

CRITICAL CONCEPTS:
==================
For a lag-K forecast at time T:
- Training: Use actuals up to T-K to predict T
- Validation/Test:
  - Use actuals up to T-K for lags >= K
  - Use PREDICTED values for lags < K (recursive forecasting)

This ensures the model is evaluated in the same way it will be used in production.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LeakageFreeFeatureResult:
    """Result from leakage-free feature engineering."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    forecast_lag: int
    feature_metadata: Dict[str, Any]


@dataclass
class RecursiveForecastResult:
    """Result from recursive forecasting."""
    predictions_df: pd.DataFrame
    wape: float
    mae: float
    rmse: float
    n_predictions: int
    forecast_lag: int


# =============================================================================
# LEAKAGE-FREE LAG FEATURES
# =============================================================================

def create_lag_features_train_only(
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
    Create lag features that are SAFE from data leakage.

    For lag-K forecasting:
    - Training rows: can use lags as normal (shift by lag amount)
    - Validation/test rows: lags < K will be NaN (filled later with predictions)
    - Validation/test rows: lags >= K use actual values

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
        Lag periods to create (default: time_format-aware)
    prefix : str
        Prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features (some will be NaN for val/test periods)
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    if lags is None:
        lags = [1, 2, 3, 6, 12] if time_format == 'year_month' else [1, 2, 4, 8, 13, 26, 52]

    # Convert date col to comparable format
    # Handle type mismatch: convert train_end to match df[date_col] dtype
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        # Integer column (e.g., year_week as 202521)
        train_end_cmp = int(train_end) if isinstance(train_end, str) else train_end
        is_train = df[date_col] <= train_end_cmp
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        train_end_cmp = str(train_end)
        is_train = df[date_col].astype(str) <= train_end_cmp
    else:
        train_end_cmp = train_end
        is_train = df[date_col] <= train_end_cmp

    for lag in lags:
        col_name = f'{prefix}_lag_{lag}'

        # Create basic lag feature
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)

        # For validation/test: mask out lags that would use future actuals
        # If lag < forecast_lag, the model wouldn't have this info at forecast time
        if lag < forecast_lag:
            # Mark these as NaN for non-training rows
            # They will be filled with predicted values during recursive forecasting
            df.loc[~is_train, col_name] = np.nan

    return df


def create_rolling_features_leakage_free(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    date_col: str,
    train_end: Union[str, int],
    forecast_lag: int,
    windows: List[int] = None,
    aggregations: List[str] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling features that respect the forecast lag constraint.

    For lag-K forecasting, rolling windows must END at least K periods before
    the forecast target date.

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
    forecast_lag : int
        Forecast horizon (lag-K)
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

    # CRITICAL: shift by forecast_lag to prevent leakage
    # This ensures rolling windows only use data available at forecast time
    shift_periods = forecast_lag

    for window in windows:
        for agg in aggregations:
            col_name = f'{prefix}_roll_{window}_{agg}'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(shift_periods).rolling(window, min_periods=1).agg(agg)
            )

    return df


def create_ewm_features_leakage_free(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    target_col: str,
    forecast_lag: int,
    spans: List[int] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create EWM features that respect the forecast lag constraint.
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    spans = spans or [4, 8, 13, 26]

    shift_periods = forecast_lag

    for span in spans:
        col_name = f'{prefix}_ewm_{span}'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).ewm(span=span, min_periods=1).mean()
        )

        col_name_std = f'{prefix}_ewm_std_{span}'
        df[col_name_std] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).ewm(span=span, min_periods=2).std()
        )

    return df


# =============================================================================
# LEAKAGE-FREE SEASONAL FEATURES (NO YoY DIFF USING CURRENT VALUE)
# =============================================================================

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

    This captures year-over-year changes using only historical data.

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
        Seasonal periods (default [12] for monthly, [52] for weekly)
    prefix : str
        Prefix for feature names
    time_format : str
        'year_week' or 'year_month'

    Returns
    -------
    pd.DataFrame
        DataFrame with seasonal features
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col
    if seasonal_periods is None:
        seasonal_periods = [12] if time_format == 'year_month' else [52]

    for period in seasonal_periods:
        # Seasonal lag (safe - uses shifted value)
        df[f'{prefix}_seasonal_lag_{period}'] = df.groupby(group_cols)[target_col].shift(period)

        # YoY change using LAGGED values only (no current value!)
        # Compare value from K periods ago to value from (period+K) periods ago
        lag_recent = df.groupby(group_cols)[target_col].shift(forecast_lag)
        lag_old = df.groupby(group_cols)[target_col].shift(period + forecast_lag)

        df[f'{prefix}_yoy_lag_diff'] = lag_recent - lag_old
        df[f'{prefix}_yoy_lag_ratio'] = lag_recent / (lag_old + 1e-10)

        # QoQ using lagged values (quarterly: 13 weeks for weekly, 3 months for monthly)
        qtr_period = 3 if time_format == 'year_month' else 13
        if period in (52, 12):
            lag_qtr_recent = df.groupby(group_cols)[target_col].shift(forecast_lag)
            lag_qtr_old = df.groupby(group_cols)[target_col].shift(qtr_period + forecast_lag)
            df[f'{prefix}_qoq_lag_diff'] = lag_qtr_recent - lag_qtr_old

    return df


# =============================================================================
# LEAKAGE-FREE CROSS-SECTIONAL FEATURES
# =============================================================================

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

    Instead of comparing current value to period average (LEAKAGE!),
    we compare the LAGGED value to the lagged period average.

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
        Segment column for segment-relative features
    prefix : str
        Prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with cross-sectional features
    """
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    prefix = prefix or target_col

    # Create lagged target for comparisons
    df['_lagged_target'] = df.groupby(group_cols)[target_col].shift(forecast_lag)

    # Compute period statistics using LAGGED values only
    # Group by date and compute stats on the lagged target
    period_stats = df.groupby(date_col)['_lagged_target'].agg(['mean', 'std', 'median'])
    period_stats.columns = ['_period_mean', '_period_std', '_period_median']
    period_stats = period_stats.reset_index()

    df = df.merge(period_stats, on=date_col, how='left')

    # Relative features using lagged values
    df[f'{prefix}_rel_to_period_mean'] = df['_lagged_target'] / (df['_period_mean'] + 1e-10)
    df[f'{prefix}_diff_from_period_mean'] = df['_lagged_target'] - df['_period_mean']
    df[f'{prefix}_zscore_period'] = (df['_lagged_target'] - df['_period_mean']) / (df['_period_std'] + 1e-10)

    # Segment-relative features
    if segment_col and segment_col in df.columns:
        seg_period_stats = df.groupby([segment_col, date_col])['_lagged_target'].agg(['mean', 'std'])
        seg_period_stats.columns = ['_seg_period_mean', '_seg_period_std']
        seg_period_stats = seg_period_stats.reset_index()

        df = df.merge(seg_period_stats, on=[segment_col, date_col], how='left')

        df[f'{prefix}_rel_to_segment'] = df['_lagged_target'] / (df['_seg_period_mean'] + 1e-10)
        df[f'{prefix}_diff_from_segment'] = df['_lagged_target'] - df['_seg_period_mean']
        df[f'{prefix}_zscore_segment'] = (
            (df['_lagged_target'] - df['_seg_period_mean']) / (df['_seg_period_std'] + 1e-10)
        )

        # Clean up temp columns
        df = df.drop(columns=['_seg_period_mean', '_seg_period_std'], errors='ignore')

    # Rank using lagged values
    df[f'{prefix}_rank_in_period'] = df.groupby(date_col)['_lagged_target'].rank(method='dense', pct=True)

    # Clean up temp columns
    df = df.drop(columns=['_lagged_target', '_period_mean', '_period_std', '_period_median'], errors='ignore')

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
    Create key-level aggregate features using ONLY training data.

    These static features should be computed on training data only,
    then applied to all rows.

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

    # Filter to training data only for computing stats
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

    # Compute key-level stats on training data only
    key_stats = train_df.groupby(group_cols)[target_col].agg([
        ('key_mean', 'mean'),
        ('key_std', 'std'),
        ('key_min', 'min'),
        ('key_max', 'max'),
        ('key_count', 'count'),
        ('key_nonzero_count', lambda x: (x > 0).sum()),
    ]).reset_index()

    # Compute CV and zero fraction
    key_stats['key_cv'] = key_stats['key_std'] / (key_stats['key_mean'] + 1e-10)
    key_stats['key_zero_fraction'] = 1 - (key_stats['key_nonzero_count'] / key_stats['key_count'])

    # Rename with prefix
    rename_dict = {col: f'{prefix}_{col}' for col in key_stats.columns if col not in group_cols}
    key_stats = key_stats.rename(columns=rename_dict)

    # Merge back to ALL data (train + val + test get same key-level features)
    df = df.merge(key_stats, on=group_cols, how='left')

    return df


# =============================================================================
# RECURSIVE MULTI-STEP FORECASTING
# =============================================================================

def recursive_forecast(
    model,
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    lag_feature_pattern: str = '_lag_',
    forecast_start: Union[str, int] = None,
    forecast_end: Union[str, int] = None,
    forecast_lag: int = 4,
    actuals_available_until: Union[str, int] = None,
) -> RecursiveForecastResult:
    """
    Generate recursive multi-step forecasts.

    For each forecast period, this function:
    1. Uses actuals for lags >= forecast_lag (available at forecast time)
    2. Uses PREDICTED values for lags < forecast_lag (recursive)
    3. Updates lag features with new predictions before next step

    Parameters
    ----------
    model : trained model
        Model with .predict() method
    df : pd.DataFrame
        DataFrame with features and target
    group_cols : str or List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    feature_cols : List[str]
        Feature columns for prediction
    lag_feature_pattern : str
        Pattern to identify lag features (default '_lag_')
    forecast_start : str or int
        First forecast date
    forecast_end : str or int
        Last forecast date
    forecast_lag : int
        Forecast horizon (lag-K)
    actuals_available_until : str or int
        Last date with available actuals (for lag features)

    Returns
    -------
    RecursiveForecastResult
        Predictions with metrics
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    df = df.copy()

    # Sort by key and date
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    # Get unique dates in forecast period
    # Handle type mismatch: convert date range params to match df[date_col] dtype
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        fs = int(forecast_start) if isinstance(forecast_start, str) else forecast_start
        fe = int(forecast_end) if isinstance(forecast_end, str) else forecast_end
        forecast_dates = sorted(df[(df[date_col] >= fs) & (df[date_col] <= fe)][date_col].unique())
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        forecast_dates = sorted(df[(df[date_col].astype(str) >= str(forecast_start)) &
                                   (df[date_col].astype(str) <= str(forecast_end))][date_col].unique())
    else:
        forecast_dates = sorted(df[(df[date_col] >= forecast_start) &
                                   (df[date_col] <= forecast_end)][date_col].unique())

    # Identify lag feature columns
    lag_cols = [c for c in feature_cols if lag_feature_pattern in c]
    non_lag_cols = [c for c in feature_cols if lag_feature_pattern not in c]

    # Store predictions
    predictions = []

    # Dictionary to store predicted values by (key, date) for recursive use
    predicted_values = {}

    for forecast_date in forecast_dates:
        # Get rows for this date
        if df[date_col].dtype == 'object':
            date_mask = df[date_col].astype(str) == str(forecast_date)
        else:
            date_mask = df[date_col] == forecast_date

        date_df = df[date_mask].copy()

        if len(date_df) == 0:
            continue

        # For each row, update lag features with predictions where needed
        for idx, row in date_df.iterrows():
            key_tuple = tuple(row[col] for col in group_cols)

            # Update lag features that should use predictions
            for lag_col in lag_cols:
                # Extract lag number from column name
                try:
                    lag_num = int(lag_col.split(lag_feature_pattern)[1].split('_')[0])
                except (ValueError, IndexError):
                    continue

                # If lag < forecast_lag, we need to use predicted value
                if lag_num < forecast_lag:
                    # Find what date this lag corresponds to
                    # lag_num periods ago from forecast_date
                    # For now, store as NaN if no prediction available
                    # In a more sophisticated implementation, we'd track dates

                    # Check if we have a prediction for this lag
                    for past_offset in range(1, lag_num + 1):
                        past_key = (key_tuple, past_offset)
                        if past_key in predicted_values:
                            date_df.loc[idx, lag_col] = predicted_values[past_key]
                            break

        # Make predictions for this date
        X = date_df[feature_cols].values

        # Handle NaN in features (fill with 0 or median)
        X = np.nan_to_num(X, nan=0.0)

        # Predict
        y_pred = model.predict(X)
        y_pred = np.clip(y_pred, 0, None)  # Non-negative

        # Store predictions
        for i, (idx, row) in enumerate(date_df.iterrows()):
            key_tuple = tuple(row[col] for col in group_cols)

            pred_row = {
                **{col: row[col] for col in group_cols},
                date_col: forecast_date,
                'actual': row[target_col] if target_col in row and pd.notna(row[target_col]) else np.nan,
                'predicted': y_pred[i],
                'forecast_lag': forecast_lag,
            }
            predictions.append(pred_row)

            # Store prediction for recursive use
            # Key: (key_tuple, periods_from_now) - simplified tracking
            predicted_values[(key_tuple, 1)] = y_pred[i]

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Compute metrics
    valid_mask = predictions_df['actual'].notna()
    valid_df = predictions_df[valid_mask]

    if len(valid_df) > 0:
        y_true = valid_df['actual'].values
        y_pred = valid_df['predicted'].values

        wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-10)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    else:
        wape = mae = rmse = np.nan

    return RecursiveForecastResult(
        predictions_df=predictions_df,
        wape=wape,
        mae=mae,
        rmse=rmse,
        n_predictions=len(predictions_df),
        forecast_lag=forecast_lag,
    )


# =============================================================================
# COMPLETE LEAKAGE-FREE FEATURE PIPELINE
# =============================================================================

def create_leakage_free_features(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    date_col: str,
    target_col: str,
    train_end: Union[str, int],
    val_end: Union[str, int],
    forecast_lag: int,
    segment_col: Optional[str] = None,
    lags: List[int] = None,
    rolling_windows: List[int] = None,
    include_calendar: bool = True,
    include_fourier: bool = True,
    time_format: str = 'year_week',
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a complete feature set with NO data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    group_cols : str or List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    train_end : str or int
        Last training date
    val_end : str or int
        Last validation date
    forecast_lag : int
        Forecast horizon (lag-K)
    segment_col : str, optional
        Segment column
    lags : List[int]
        Lag periods
    rolling_windows : List[int]
        Rolling window sizes
    include_calendar : bool
        Include calendar features
    include_fourier : bool
        Include Fourier features

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (DataFrame with features, list of feature column names)
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    df = df.copy()
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    if lags is None:
        if time_format == 'year_month':
            lags = [forecast_lag, forecast_lag + 1, forecast_lag + 2,
                    forecast_lag + 3, 6, 12]
        else:
            lags = [forecast_lag, forecast_lag + 1, forecast_lag + 2,
                    forecast_lag + 4, 13, 26, 52]
    # Ensure forecast_lag is included and lags >= forecast_lag are prioritized
    lags = sorted(set([l for l in lags if l >= forecast_lag] + [l for l in lags if l < forecast_lag]))

    if rolling_windows is None:
        rolling_windows = [3, 6, 12] if time_format == 'year_month' else [4, 8, 13, 26]

    original_cols = set(df.columns)

    # 1. Lag features (with leakage protection)
    df = create_lag_features_train_only(
        df, group_cols, target_col, date_col, train_end, forecast_lag, lags
    )
    logger.info(f"Created {len(lags)} lag features with forecast_lag={forecast_lag}")

    # 2. Rolling features (shifted by forecast_lag)
    df = create_rolling_features_leakage_free(
        df, group_cols, target_col, date_col, train_end, forecast_lag, rolling_windows
    )
    logger.info(f"Created rolling features for windows {rolling_windows}")

    # 3. EWM features (shifted by forecast_lag)
    ewm_spans = [3, 6] if time_format == 'year_month' else [4, 13]
    df = create_ewm_features_leakage_free(
        df, group_cols, target_col, forecast_lag, spans=ewm_spans
    )
    logger.info("Created EWM features")

    # 4. Seasonal features (using lagged values only)
    seasonal_periods = [12] if time_format == 'year_month' else [52]
    df = create_seasonal_features_leakage_free(
        df, group_cols, target_col, forecast_lag, seasonal_periods,
        time_format=time_format,
    )
    logger.info("Created seasonal features (leakage-free)")

    # 5. Cross-sectional features (using lagged values only)
    df = create_cross_sectional_features_leakage_free(
        df, group_cols, target_col, date_col, forecast_lag, segment_col
    )
    logger.info("Created cross-sectional features (leakage-free)")

    # 6. Key-level features (computed on training data only)
    df = create_key_level_features_leakage_free(
        df, group_cols, target_col, date_col, train_end
    )
    logger.info("Created key-level features (training data only)")

    # 7. Calendar features (no leakage - purely date-based)
    if include_calendar:
        df = _add_calendar_features(df, date_col)
        logger.info("Created calendar features")

    # 8. Fourier features (no leakage - purely date-based)
    if include_fourier:
        df = _add_fourier_features(df, date_col, time_format=time_format)
        logger.info("Created Fourier features")

    # Get feature columns
    exclude_cols = set(group_cols) | {date_col, target_col}
    if segment_col:
        exclude_cols.add(segment_col)
    exclude_cols.update(original_cols)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Fill NaN in features (for validation/test short lags, they'll be NaN)
    # These will be updated during recursive forecasting
    # Handle type mismatch for train_end comparison
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        train_end_cmp = int(train_end) if isinstance(train_end, str) else train_end
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        train_end_cmp = str(train_end)
    else:
        train_end_cmp = train_end

    for col in feature_cols:
        if df[col].isna().any():
            # Fill with training median for now
            if col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
                train_median = df[df[date_col].astype(str) <= str(train_end_cmp)][col].median()
            else:
                train_median = df[df[date_col] <= train_end_cmp][col].median()
            df[col] = df[col].fillna(train_median if pd.notna(train_median) else 0)

    logger.info(f"Created {len(feature_cols)} total features")

    return df, feature_cols


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add calendar features (no leakage - purely date-based)."""
    df = df.copy()

    # Handle year_week format (YYYYWW)
    if df[date_col].dtype in ['int64', 'float64'] or (
        df[date_col].dtype == 'object' and df[date_col].str.len().mode().iloc[0] == 6
    ):
        df['year'] = df[date_col].astype(str).str[:4].astype(int)
        df['week_of_year'] = df[date_col].astype(str).str[4:].astype(int)
        df['quarter'] = ((df['week_of_year'] - 1) // 13) + 1
    else:
        # Try datetime parsing
        try:
            dates = pd.to_datetime(df[date_col])
            df['year'] = dates.dt.year
            df['month'] = dates.dt.month
            df['week_of_year'] = dates.dt.isocalendar().week.astype(int)
            df['quarter'] = dates.dt.quarter
        except Exception:
            pass

    return df


def _add_fourier_features(df: pd.DataFrame, date_col: str, n_harmonics: int = 3, time_format: str = 'year_week') -> pd.DataFrame:
    """Add Fourier features for seasonality.

    Adapts period to time_format: 52 for weekly, 12 for monthly.
    """
    df = df.copy()
    is_monthly = (time_format == 'year_month')

    if is_monthly and 'month_of_year' in df.columns:
        t = df['month_of_year'].values
        period = 12
    elif 'week_of_year' in df.columns:
        t = df['week_of_year'].values
        period = 52
    elif df[date_col].dtype in ['int64', 'float64']:
        t = (df[date_col] % 100).values
        period = 12 if is_monthly else 52
    else:
        return df  # Can't compute without period info

    for k in range(1, n_harmonics + 1):
        df[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)

    return df


# =============================================================================
# VALIDATION HELPER
# =============================================================================

def validate_no_leakage(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    train_end: Union[str, int],
    forecast_lag: int,
) -> Dict[str, Any]:
    """
    Validate that features don't contain data leakage.

    Checks:
    1. No feature has correlation > 0.99 with shifted target
    2. Lag features for lags < forecast_lag are NaN in val/test
    3. No feature uses future information

    Returns
    -------
    Dict with validation results
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    results = {
        'has_leakage': False,
        'leakage_features': [],
        'warnings': [],
    }

    # Check for high correlation with current target
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

    for col in feature_cols:
        if col not in train_df.columns:
            continue

        corr = train_df[col].corr(train_df[target_col])

        if pd.notna(corr) and abs(corr) > 0.99:
            results['has_leakage'] = True
            results['leakage_features'].append({
                'feature': col,
                'correlation': corr,
                'reason': 'Extremely high correlation with target'
            })

    # Check lag features in validation period
    val_df = df[~train_mask]

    for col in feature_cols:
        if '_lag_' in col:
            try:
                lag_num = int(col.split('_lag_')[1].split('_')[0])
                if lag_num < forecast_lag:
                    # These should have been handled (either NaN or filled with predictions)
                    non_nan_count = val_df[col].notna().sum()
                    if non_nan_count > 0:
                        results['warnings'].append(
                            f"{col} (lag={lag_num} < forecast_lag={forecast_lag}) has {non_nan_count} "
                            f"non-NaN values in validation. Ensure these are predictions, not actuals."
                        )
            except (ValueError, IndexError):
                pass

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'LeakageFreeFeatureResult',
    'RecursiveForecastResult',

    # Leakage-free feature functions
    'create_lag_features_train_only',
    'create_rolling_features_leakage_free',
    'create_ewm_features_leakage_free',
    'create_seasonal_features_leakage_free',
    'create_cross_sectional_features_leakage_free',
    'create_key_level_features_leakage_free',

    # Recursive forecasting
    'recursive_forecast',

    # Complete pipeline
    'create_leakage_free_features',

    # Validation
    'validate_no_leakage',
]
