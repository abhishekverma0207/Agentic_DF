# utils/recursive_forecasting.py
"""
Recursive Forecasting Utility
=============================

This module provides proper multi-step ahead forecasting WITHOUT feature leakage.

THE PROBLEM:
When Feature Crew computes lag and rolling features for test data, it uses ACTUAL
future values. If Training Crew just calls model.predict(test_features), it's
cheating - the model sees future information in the features.

THE SOLUTION:
Recursive forecasting:
1. Start with only known actuals (train for val, train+val for test)
2. Predict one step at a time
3. Use each prediction to update lag/rolling features for the next step
4. Never use actual future values in features

USAGE:
    from utils.recursive_forecasting import RecursiveForecaster, generate_recursive_predictions

    # Option 1: Use the RecursiveForecaster class
    forecaster = RecursiveForecaster(
        model=trained_model,
        lag_columns=['qty_lag_1', 'qty_lag_2', 'qty_lag_4'],
        rolling_columns=['qty_roll_4_mean', 'qty_roll_13_mean'],
        target_column='qty',
    )
    predictions = forecaster.forecast(
        historical_actuals=train_actuals,
        future_features=test_static_features,
        horizon=8,
    )

    # Option 2: Use the convenience function
    results_df = generate_recursive_predictions(
        model=trained_model,
        train_data=train_df,
        forecast_data=test_df,
        key_columns=['product_id', 'store_id'],
        date_column='year_week',
        target_column='qty',
        feature_columns=feature_cols,
        lag_columns=lag_cols,
        rolling_columns=roll_cols,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RecursiveForecastConfig:
    """Configuration for recursive forecasting."""
    # Dead key detection
    dead_key_lookback: int = None  # Periods to check for dead key (auto: 12 monthly, 52 weekly)
    dead_key_threshold: float = 0.0  # If sum of last N periods <= this, key is dead

    # Prediction clipping
    clip_min: float = 0.0  # Minimum prediction value (no negative demand)
    clip_max: Optional[float] = None  # Maximum prediction value (optional)

    # Feature update behavior
    use_predictions_for_lags: bool = True  # Update lags with predictions
    use_predictions_for_rolling: bool = True  # Update rolling features with predictions

    # Time format for adaptive defaults
    time_format: str = 'year_week'

    def __post_init__(self):
        """Resolve None defaults based on time_format."""
        if self.dead_key_lookback is None:
            self.dead_key_lookback = 12 if self.time_format == 'year_month' else 52


# =============================================================================
# CORE RECURSIVE FORECASTER
# =============================================================================

class RecursiveForecaster:
    """
    Handles recursive multi-step ahead forecasting.

    This class properly updates lag and rolling features step-by-step
    to avoid feature leakage during prediction.
    """

    def __init__(
        self,
        model: Any,
        target_column: str,
        lag_columns: List[str],
        rolling_columns: List[str],
        static_columns: List[str],
        config: Optional[RecursiveForecastConfig] = None,
    ):
        """
        Initialize the recursive forecaster.

        Parameters
        ----------
        model : fitted model
            Must have a predict() method
        target_column : str
            Name of the target column (e.g., 'qty', 'PS_actual_sales')
        lag_columns : List[str]
            Columns representing lag features (e.g., 'qty_lag_1', 'qty_lag_4')
        rolling_columns : List[str]
            Columns representing rolling features (e.g., 'qty_roll_4_mean')
        static_columns : List[str]
            Columns that don't need updating (calendar, price, promo, etc.)
        config : RecursiveForecastConfig, optional
            Configuration options
        """
        self.model = model
        self.target_column = target_column
        self.lag_columns = lag_columns
        self.rolling_columns = rolling_columns
        self.static_columns = static_columns
        self.config = config or RecursiveForecastConfig()

        # Parse lag numbers from column names
        self._lag_map = self._parse_lag_columns(lag_columns)
        # Parse rolling specs from column names
        self._rolling_map = self._parse_rolling_columns(rolling_columns)
        # Parse derived features (lag_diff, lag_ratio, yoy_lag_diff, etc.)
        all_cols = lag_columns + rolling_columns + static_columns
        self._derived_map = self._parse_derived_columns(all_cols)

    def _parse_lag_columns(self, columns: List[str]) -> Dict[str, int]:
        """
        Parse lag column names to extract lag numbers.

        Examples:
            'qty_lag_1' -> 1
            'PS_actual_sales_lag_4' -> 4
            'target_lag1' -> 1

        NOTE: Derived features like 'lag_diff_1_4' or 'lag_ratio_1_52' are handled
        separately by _parse_derived_columns.
        """
        lag_map = {}

        # Patterns for derived features that should be skipped here
        derived_patterns = ['lag_diff', 'lag_ratio', 'lag_pct_change', 'lag_direction',
                           'yoy_lag', 'qoq_lag']

        for col in columns:
            col_lower = col.lower()

            # Skip derived features (handled separately)
            if any(pattern in col_lower for pattern in derived_patterns):
                continue

            # Try to find simple lag number in column name (lag_N or lagN)
            # Match the LAST number after 'lag' to handle cases like 'target_lag_1'
            match = re.search(r'lag[_]?(\d+)(?:[_]|$)', col, re.IGNORECASE)
            if match:
                lag_map[col] = int(match.group(1))
            else:
                # Only warn for columns that look like they should be lags but can't be parsed
                logger.debug(f"Could not parse lag number from column: {col}")
        return lag_map

    def _parse_derived_columns(self, columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Parse derived feature columns (lag_diff, lag_ratio, yoy_lag_diff, etc.)
        to extract the lag values needed for recalculation.

        Examples:
            'qty_lag_diff_4' -> {'type': 'diff', 'lag_short': 4, 'lag_long': 8}
            'qty_lag_ratio_1_52' -> {'type': 'ratio', 'lag_short': 1, 'lag_long': 52}
            'qty_yoy_lag_diff' -> {'type': 'yoy_diff', 'lag_short': K, 'lag_long': K+52}
            'qty_lag_pct_change_4' -> {'type': 'pct_change', 'lag_short': 4, 'lag_long': 8}
        """
        derived_map = {}
        target_lower = self.target_column.lower()

        for col in columns:
            col_lower = col.lower()

            # Skip if not related to target
            if target_lower not in col_lower and 'lag' not in col_lower:
                continue

            # Pattern: {prefix}_lag_diff_{period} (e.g., qty_lag_diff_4)
            # Represents lag_K - lag_(K+period) where K is forecast_lag
            match = re.search(r'lag_diff[_]?(\d+)$', col_lower)
            if match:
                period = int(match.group(1))
                # These use forecast_lag as short, forecast_lag + period as long
                # We'll use a placeholder - actual lag values computed at runtime
                derived_map[col] = {
                    'type': 'lag_diff',
                    'period': period,
                }
                continue

            # Pattern: {prefix}_lag_pct_change_{period}
            match = re.search(r'lag_pct_change[_]?(\d+)$', col_lower)
            if match:
                period = int(match.group(1))
                derived_map[col] = {
                    'type': 'lag_pct_change',
                    'period': period,
                }
                continue

            # Pattern: {prefix}_lag_direction_{period}
            match = re.search(r'lag_direction[_]?(\d+)$', col_lower)
            if match:
                period = int(match.group(1))
                derived_map[col] = {
                    'type': 'lag_direction',
                    'period': period,
                }
                continue

            # Pattern: {prefix}_lag_ratio_{short}_{long}
            match = re.search(r'lag_ratio[_]?(\d+)[_](\d+)$', col_lower)
            if match:
                derived_map[col] = {
                    'type': 'lag_ratio',
                    'lag_short': int(match.group(1)),
                    'lag_long': int(match.group(2)),
                }
                continue

            # Pattern: {prefix}_yoy_lag_diff (year-over-year using lags)
            # Period is 12 for monthly, 52 for weekly
            _yoy_period = 12 if self.config.time_format == 'year_month' else 52
            _qoq_period = 3 if self.config.time_format == 'year_month' else 13
            if 'yoy_lag_diff' in col_lower:
                derived_map[col] = {
                    'type': 'yoy_lag_diff',
                    'period': _yoy_period,
                }
                continue

            # Pattern: {prefix}_yoy_lag_ratio
            if 'yoy_lag_ratio' in col_lower:
                derived_map[col] = {
                    'type': 'yoy_lag_ratio',
                    'period': _yoy_period,
                }
                continue

            # Pattern: {prefix}_qoq_lag_diff (quarter-over-quarter using lags)
            if 'qoq_lag_diff' in col_lower:
                derived_map[col] = {
                    'type': 'qoq_lag_diff',
                    'period': _qoq_period,
                }
                continue

        return derived_map

    def _parse_rolling_columns(self, columns: List[str]) -> Dict[str, Tuple[int, str]]:
        """
        Parse rolling/EWM column names to extract window/span size and aggregation.

        Examples:
            'qty_roll_4_mean' -> (4, 'mean')
            'PS_actual_sales_rolling_13_std' -> (13, 'std')
            'qty_ewm_4' -> (4, 'ewm_mean')
            'qty_ewm_std_8' -> (8, 'ewm_std')
        """
        rolling_map = {}
        for col in columns:
            col_lower = col.lower()

            # Pattern 1: EWM features - {prefix}_ewm_{span} or {prefix}_ewm_std_{span}
            if 'ewm' in col_lower:
                # Try to match ewm_std pattern first
                match_std = re.search(r'ewm[_]?std[_]?(\d+)', col, re.IGNORECASE)
                if match_std:
                    span = int(match_std.group(1))
                    rolling_map[col] = (span, 'ewm_std')
                    continue

                # Then try regular ewm pattern
                match_ewm = re.search(r'ewm[_]?(\d+)', col, re.IGNORECASE)
                if match_ewm:
                    span = int(match_ewm.group(1))
                    rolling_map[col] = (span, 'ewm_mean')
                    continue

            # Pattern 2: Rolling features - {prefix}_roll_{window}_{agg}
            match = re.search(r'roll(?:ing)?[_]?(\d+)[_]?(mean|std|min|max|sum|median)?', col, re.IGNORECASE)
            if match:
                window = int(match.group(1))
                agg = match.group(2).lower() if match.group(2) else 'mean'
                rolling_map[col] = (window, agg)
            else:
                logger.debug(f"Could not parse rolling/ewm spec from column: {col}")
        return rolling_map

    def _is_dead_key(self, historical_actuals: np.ndarray) -> bool:
        """
        Check if a key is 'dead' (no sales for extended period).

        A dead key gets zero predictions without running the model.
        """
        lookback = self.config.dead_key_lookback
        threshold = self.config.dead_key_threshold

        if len(historical_actuals) < lookback:
            return False  # Not enough history to determine

        recent_sum = np.sum(historical_actuals[-lookback:])
        return recent_sum <= threshold

    def _is_log_feature(self, col_name: str) -> bool:
        """Check if a feature is log-transformed (based on naming convention)."""
        return '_log_' in col_name.lower()

    def _get_values_for_feature(self, all_values: List[float], col_name: str) -> List[float]:
        """
        Get the appropriate values for a feature, applying log1p if needed.

        For log-transformed features (e.g., 'PS_actual_sales_log_lag_1'),
        we need to apply log1p to the raw values before computing the lag/rolling.
        """
        if self._is_log_feature(col_name):
            # Apply log1p transformation: log(1 + x) for x >= 0
            return [np.log1p(max(0, v)) for v in all_values]
        return all_values

    def _compute_lag_value(self, all_values: List[float], lag: int) -> float:
        """Get the value at a specific lag position."""
        if len(all_values) >= lag:
            return all_values[-lag]
        return 0.0

    def _compute_rolling_value(self, all_values: List[float], window: int, agg: str) -> float:
        """Compute a rolling aggregate over the most recent values."""
        if len(all_values) < window:
            # Not enough values - use what we have
            window_values = all_values
        else:
            window_values = all_values[-window:]

        if len(window_values) == 0:
            return 0.0

        window_array = np.array(window_values)

        if agg == 'mean':
            return np.mean(window_array)
        elif agg == 'std':
            return np.std(window_array) if len(window_array) > 1 else 0.0
        elif agg == 'min':
            return np.min(window_array)
        elif agg == 'max':
            return np.max(window_array)
        elif agg == 'sum':
            return np.sum(window_array)
        elif agg == 'median':
            return np.median(window_array)
        elif agg == 'ewm_mean':
            # EWM mean: use pandas for proper exponential weighting
            # span parameter maps to window size
            series = pd.Series(all_values)
            return series.ewm(span=window, min_periods=1).mean().iloc[-1]
        elif agg == 'ewm_std':
            # EWM std
            series = pd.Series(all_values)
            ewm_std = series.ewm(span=window, min_periods=2).std().iloc[-1]
            return ewm_std if not np.isnan(ewm_std) else 0.0
        else:
            return np.mean(window_array)

    def _update_features(
        self,
        base_features: np.ndarray,
        feature_names: List[str],
        all_values: List[float],
    ) -> np.ndarray:
        """
        Update lag, rolling, and derived features based on current value history.

        CRITICAL: During recursive forecasting, we update features using
        (historical actuals + predictions so far). All features that use
        shift(k) on the target must be recalculated.

        The key insight: features are created using shift(1) as the reference,
        so during step t of recursive forecasting:
        - lag_1 = prediction from step t-1 (or last actual if t=1)
        - lag_2 = prediction from step t-2 (or actual)
        - etc.

        Parameters
        ----------
        base_features : np.ndarray
            Original feature values (1D array for single row)
        feature_names : List[str]
            Names of all features in order
        all_values : List[float]
            Historical actuals + predictions so far

        Returns
        -------
        np.ndarray
            Updated feature values
        """
        updated = base_features.copy()

        # 1. Update simple lag features
        # For log-transformed features, we need to use log1p(values) instead of raw values
        if self.config.use_predictions_for_lags:
            for col, lag in self._lag_map.items():
                if col in feature_names:
                    idx = feature_names.index(col)
                    # Get appropriately transformed values (log1p for log features)
                    values_to_use = self._get_values_for_feature(all_values, col)
                    updated[idx] = self._compute_lag_value(values_to_use, lag)

        # 2. Update rolling features
        # For log-transformed features, we need to use log1p(values) instead of raw values
        if self.config.use_predictions_for_rolling:
            for col, (window, agg) in self._rolling_map.items():
                if col in feature_names:
                    idx = feature_names.index(col)
                    # Get appropriately transformed values (log1p for log features)
                    values_to_use = self._get_values_for_feature(all_values, col)
                    updated[idx] = self._compute_rolling_value(values_to_use, window, agg)

        # 3. Update derived features (lag_diff, lag_ratio, yoy_lag_diff, etc.)
        # These are computed from lag values, so must be recalculated after lags are updated
        # IMPORTANT: All derived features use shift(1) as reference, so lag_short=1
        # For log-transformed features, we need to use log1p(values)
        for col, spec in self._derived_map.items():
            if col not in feature_names:
                continue

            idx = feature_names.index(col)
            feat_type = spec['type']

            # Get appropriately transformed values (log1p for log features)
            values_to_use = self._get_values_for_feature(all_values, col)

            if feat_type == 'lag_diff':
                # lag_diff_period = lag_1 - lag_(1+period)
                period = spec['period']
                lag_short = 1  # Always use lag_1 as reference
                lag_long = 1 + period
                val_short = self._compute_lag_value(values_to_use, lag_short)
                val_long = self._compute_lag_value(values_to_use, lag_long)
                updated[idx] = val_short - val_long

            elif feat_type == 'lag_pct_change':
                # lag_pct_change_period = (lag_1 - lag_(1+period)) / |lag_(1+period)|
                period = spec['period']
                lag_short = 1
                lag_long = 1 + period
                val_short = self._compute_lag_value(values_to_use, lag_short)
                val_long = self._compute_lag_value(values_to_use, lag_long)
                updated[idx] = (val_short - val_long) / (abs(val_long) + 1e-10)

            elif feat_type == 'lag_direction':
                # Sign of lag_diff
                period = spec['period']
                lag_short = 1
                lag_long = 1 + period
                val_short = self._compute_lag_value(values_to_use, lag_short)
                val_long = self._compute_lag_value(values_to_use, lag_long)
                diff = val_short - val_long
                updated[idx] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)

            elif feat_type == 'lag_ratio':
                # Ratio of two specific lags (as specified in feature name)
                lag_short = spec['lag_short']
                lag_long = spec['lag_long']
                val_short = self._compute_lag_value(values_to_use, lag_short)
                val_long = self._compute_lag_value(values_to_use, lag_long)
                updated[idx] = val_short / (abs(val_long) + 1e-10)

            elif feat_type == 'yoy_lag_diff':
                # Year-over-year: lag_1 - lag_(1+period)
                _yoy_period = 12 if self.config.time_format == 'year_month' else 52
                lag_short = 1
                lag_long = 1 + _yoy_period
                val_short = self._compute_lag_value(values_to_use, lag_short)
                val_long = self._compute_lag_value(values_to_use, lag_long)
                updated[idx] = val_short - val_long

            elif feat_type == 'yoy_lag_ratio':
                # Year-over-year ratio: lag_1 / lag_(1+period)
                _yoy_period = 12 if self.config.time_format == 'year_month' else 52
                lag_short = 1
                lag_long = 1 + _yoy_period
                val_short = self._compute_lag_value(values_to_use, lag_short)
                val_long = self._compute_lag_value(values_to_use, lag_long)
                updated[idx] = val_short / (abs(val_long) + 1e-10)

            elif feat_type == 'qoq_lag_diff':
                # Quarter-over-quarter: lag_1 - lag_(1+period)
                _qoq_period = 3 if self.config.time_format == 'year_month' else 13
                lag_short = 1
                lag_long = 1 + _qoq_period
                val_short = self._compute_lag_value(values_to_use, lag_short)
                val_long = self._compute_lag_value(values_to_use, lag_long)
                updated[idx] = val_short - val_long

        return updated

    def forecast(
        self,
        historical_actuals: np.ndarray,
        future_features: np.ndarray,
        feature_names: List[str],
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate recursive forecasts for a single time series.

        CRITICAL: This method updates ALL shift-based features using
        (historical actuals + predictions so far) at each step.

        Parameters
        ----------
        historical_actuals : np.ndarray
            Historical target values (actuals known at forecast time)
        future_features : np.ndarray
            Feature matrix for future periods, shape (horizon, n_features)
            Lag/rolling columns will be overwritten with values from
            history + predictions
        feature_names : List[str]
            Names of features in order
        horizon : int, optional
            Number of periods to forecast (defaults to rows in future_features)

        Returns
        -------
        np.ndarray
            Predictions for each future period
        """
        if horizon is None:
            horizon = len(future_features)

        # Check for dead key
        if self._is_dead_key(historical_actuals):
            logger.debug("Dead key detected - returning zero predictions")
            return np.zeros(horizon)

        # Initialize value history with known actuals
        all_values = list(historical_actuals)
        predictions = []

        for t in range(min(horizon, len(future_features))):
            # Get base features for this period
            base_features = future_features[t]

            # Update ALL shift-based features using (history + predictions so far)
            # This ensures lag_1 = most recent prediction, lag_2 = prediction before that, etc.
            updated_features = self._update_features(
                base_features, feature_names, all_values
            )

            # Make prediction
            X = updated_features.reshape(1, -1)
            pred = self.model.predict(X)[0]

            # Clip prediction
            pred = np.clip(pred, self.config.clip_min, self.config.clip_max)

            # Store prediction
            predictions.append(pred)

            # Add prediction to value history for next iteration
            all_values.append(pred)

        return np.array(predictions)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def identify_lag_columns(columns: List[str], target_column: str) -> List[str]:
    """
    Automatically identify SIMPLE lag columns from a list of column names.

    This function identifies direct lag features like 'target_lag_1', 'target_lag_4'.
    It EXCLUDES derived features like 'lag_diff_1_4' or 'lag_ratio_1_52' because
    these are calculated from multiple lags and cannot be updated step-by-step.

    Parameters
    ----------
    columns : List[str]
        All column names
    target_column : str
        Name of the target column

    Returns
    -------
    List[str]
        Columns that appear to be simple lag features (excludable for recursive update)
    """
    lag_cols = []
    target_lower = target_column.lower()

    # Patterns to EXCLUDE (derived features that aren't simple lags)
    exclude_patterns = ['lag_diff', 'lag_ratio', 'diff_', 'ratio_', 'pct_change']

    for col in columns:
        col_lower = col.lower()

        # Skip derived features (diff, ratio between lags)
        if any(pattern in col_lower for pattern in exclude_patterns):
            continue

        # Match patterns like: target_lag_1, target_lag1, lag_1, lag1
        # Must have 'lag' followed by a single number
        if 'lag' in col_lower:
            # Check if it matches simple lag pattern: lag_N or lagN (single number)
            if re.search(r'lag[_]?\d+$', col_lower) or re.search(r'lag[_]?\d+[_]', col_lower):
                # Check if it's related to target
                if target_lower in col_lower or 'lag_' in col_lower:
                    lag_cols.append(col)

    # Log warning if no lag columns found to help diagnose issues
    if len(lag_cols) == 0:
        lag_containing = [c for c in columns if 'lag' in c.lower()]
        if lag_containing:
            logger.debug(f"No simple lag cols found. Columns with 'lag': {lag_containing[:5]} (showing first 5)")
        else:
            logger.debug(f"No lag columns found in {len(columns)} features. Sample cols: {columns[:5]}")

    return lag_cols


def identify_rolling_columns(columns: List[str], target_column: str) -> List[str]:
    """
    Automatically identify rolling/window/EWM columns from a list of column names.

    Parameters
    ----------
    columns : List[str]
        All column names
    target_column : str
        Name of the target column

    Returns
    -------
    List[str]
        Columns that appear to be rolling or EWM features
    """
    roll_cols = []
    target_lower = target_column.lower()

    for col in columns:
        col_lower = col.lower()
        # Match patterns like: target_roll_4_mean, rolling_13_std, target_ewm_4, target_ewm_std_8
        if 'roll' in col_lower or 'rolling' in col_lower or 'window' in col_lower or 'ewm' in col_lower:
            if target_lower in col_lower or re.search(r'(roll(?:ing)?|ewm)[_]?\d+', col_lower):
                roll_cols.append(col)

    return roll_cols


def generate_recursive_predictions(
    model: Any,
    train_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    key_columns: List[str],
    date_column: str,
    target_column: str,
    feature_columns: List[str],
    lag_columns: Optional[List[str]] = None,
    rolling_columns: Optional[List[str]] = None,
    val_data: Optional[pd.DataFrame] = None,
    is_test: bool = False,
    config: Optional[RecursiveForecastConfig] = None,
) -> pd.DataFrame:
    """
    Generate recursive predictions for all keys in the forecast data.

    This is the main convenience function for generating proper multi-step
    forecasts without feature leakage.

    Parameters
    ----------
    model : fitted model
        Must have predict() method
    train_data : pd.DataFrame
        Training data with actuals
    forecast_data : pd.DataFrame
        Data for periods to forecast (features will be updated recursively)
    key_columns : List[str]
        Columns identifying unique series
    date_column : str
        Column with date/period values
    target_column : str
        Column to forecast
    feature_columns : List[str]
        All feature columns to use
    lag_columns : List[str], optional
        Lag feature columns (auto-detected if None)
    rolling_columns : List[str], optional
        Rolling feature columns (auto-detected if None)
    val_data : pd.DataFrame, optional
        Validation data (if is_test=True, val actuals are included in history)
    is_test : bool
        If True, include val_data actuals in history
    config : RecursiveForecastConfig, optional
        Configuration options

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: key_columns, date_column, 'actual', 'predicted', 'is_dead_key'
    """
    config = config or RecursiveForecastConfig()

    # Auto-detect lag/rolling columns if not provided
    if lag_columns is None:
        lag_columns = identify_lag_columns(feature_columns, target_column)
        logger.info(f"Auto-detected lag columns: {lag_columns}")

    if rolling_columns is None:
        rolling_columns = identify_rolling_columns(feature_columns, target_column)
        logger.info(f"Auto-detected rolling columns: {rolling_columns}")

    # Static columns = all features that aren't lag or rolling
    static_columns = [c for c in feature_columns if c not in lag_columns and c not in rolling_columns]

    # Create forecaster
    forecaster = RecursiveForecaster(
        model=model,
        target_column=target_column,
        lag_columns=lag_columns,
        rolling_columns=rolling_columns,
        static_columns=static_columns,
        config=config,
    )

    # Combine historical data
    if is_test and val_data is not None:
        historical_data = pd.concat([train_data, val_data], ignore_index=True)
    else:
        historical_data = train_data

    # Get unique keys
    key_col_str = key_columns if isinstance(key_columns, str) else key_columns[0] if len(key_columns) == 1 else None

    if key_col_str and key_col_str in forecast_data.columns:
        unique_keys = forecast_data[key_col_str].unique()
        key_is_single = True
    else:
        # Multiple key columns
        unique_keys = forecast_data[key_columns].drop_duplicates().values.tolist()
        key_is_single = False

    all_results = []
    dead_key_count = 0

    for key in unique_keys:
        # Filter data for this key
        if key_is_single:
            hist_mask = historical_data[key_col_str] == key
            forecast_mask = forecast_data[key_col_str] == key
        else:
            hist_mask = (historical_data[key_columns] == key).all(axis=1)
            forecast_mask = (forecast_data[key_columns] == key).all(axis=1)

        key_hist = historical_data[hist_mask].sort_values(date_column)
        key_forecast = forecast_data[forecast_mask].sort_values(date_column)

        if len(key_forecast) == 0:
            continue

        # Get historical actuals
        historical_actuals = key_hist[target_column].values

        # Get future features
        future_features = key_forecast[feature_columns].values

        # Generate recursive predictions
        predictions = forecaster.forecast(
            historical_actuals=historical_actuals,
            future_features=future_features,
            feature_names=feature_columns,
        )

        # Check if dead key
        is_dead = forecaster._is_dead_key(historical_actuals)
        if is_dead:
            dead_key_count += 1

        # Build result records
        for i, (_, row) in enumerate(key_forecast.iterrows()):
            result = {}

            # Add key columns
            if key_is_single:
                result[key_col_str] = key
            else:
                for j, col in enumerate(key_columns):
                    result[col] = key[j]

            # Add date, actual, prediction
            result[date_column] = row[date_column]
            result['actual'] = row[target_column] if target_column in row else np.nan
            result['predicted'] = predictions[i] if i < len(predictions) else np.nan
            result['is_dead_key'] = is_dead

            all_results.append(result)

    logger.info(f"Generated predictions for {len(unique_keys)} keys ({dead_key_count} dead keys)")

    return pd.DataFrame(all_results)


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Weighted Absolute Percentage Error.

    WAPE = sum(|actual - predicted|) / sum(|actual|)

    This is the primary metric for demand forecasting evaluation.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true)) + 1e-10  # Avoid division by zero

    return numerator / denominator


def validate_recursive_predictions(
    predictions_df: pd.DataFrame,
    expected_horizon: int,
    key_columns: List[str],
) -> Dict[str, Any]:
    """
    Validate that predictions were generated correctly.

    Checks:
    1. Correct number of predictions per key
    2. No duplicate date-key combinations
    3. Predictions are non-negative
    4. Some variation in predictions (not all constant)

    Returns
    -------
    Dict with validation results
    """
    results = {
        'is_valid': True,
        'issues': [],
        'stats': {},
    }

    # Check predictions per key
    key_col = key_columns[0] if len(key_columns) == 1 else tuple(key_columns)

    if isinstance(key_col, str):
        preds_per_key = predictions_df.groupby(key_col).size()
    else:
        preds_per_key = predictions_df.groupby(list(key_columns)).size()

    if preds_per_key.min() < expected_horizon:
        results['issues'].append(f"Some keys have fewer than {expected_horizon} predictions")

    # Check for negative predictions
    if 'predicted' in predictions_df.columns:
        n_negative = (predictions_df['predicted'] < 0).sum()
        if n_negative > 0:
            results['issues'].append(f"{n_negative} negative predictions found")
            results['is_valid'] = False

    # Check for variation
    if 'predicted' in predictions_df.columns:
        pred_std = predictions_df['predicted'].std()
        if pred_std < 1e-10:
            results['issues'].append("All predictions are identical (no variation)")

    # Compute stats
    if 'predicted' in predictions_df.columns and 'actual' in predictions_df.columns:
        valid_mask = predictions_df['actual'].notna() & predictions_df['predicted'].notna()
        if valid_mask.sum() > 0:
            wape = compute_wape(
                predictions_df.loc[valid_mask, 'actual'].values,
                predictions_df.loc[valid_mask, 'predicted'].values,
            )
            results['stats']['wape'] = wape

    results['stats']['n_predictions'] = len(predictions_df)
    results['stats']['n_keys'] = preds_per_key.size
    results['stats']['n_dead_keys'] = predictions_df['is_dead_key'].sum() if 'is_dead_key' in predictions_df.columns else 0

    if results['issues']:
        logger.warning(f"Prediction validation issues: {results['issues']}")

    return results


# =============================================================================
# FEATURE LEAKAGE DETECTION
# =============================================================================

def detect_feature_leakage(
    predictions_df: pd.DataFrame,
    key_columns: List[str],
    date_column: str,
) -> Dict[str, Any]:
    """
    Detect potential feature leakage in predictions.

    Signs of leakage:
    1. Predictions are too similar to actuals (suspiciously good WAPE)
    2. Lag patterns don't match recursive structure
    3. Predictions for different horizons are independent (no correlation)

    Returns
    -------
    Dict with leakage detection results
    """
    results = {
        'leakage_suspected': False,
        'evidence': [],
        'metrics': {},
    }

    if 'predicted' not in predictions_df.columns or 'actual' not in predictions_df.columns:
        return results

    # Check 1: Is WAPE suspiciously low for test data?
    valid_mask = predictions_df['actual'].notna()
    if valid_mask.sum() > 10:
        wape = compute_wape(
            predictions_df.loc[valid_mask, 'actual'].values,
            predictions_df.loc[valid_mask, 'predicted'].values,
        )
        results['metrics']['wape'] = wape

        # WAPE < 1% on test data is suspicious for most demand forecasting
        if wape < 0.01:
            results['leakage_suspected'] = True
            results['evidence'].append(f"WAPE is suspiciously low: {wape:.4f}")

    # Check 2: Look for perfect prediction patterns
    if valid_mask.sum() > 0:
        correlation = np.corrcoef(
            predictions_df.loc[valid_mask, 'actual'].values,
            predictions_df.loc[valid_mask, 'predicted'].values,
        )[0, 1]
        results['metrics']['correlation'] = correlation

        # Correlation > 0.99 on test data is suspicious
        if correlation > 0.99:
            results['leakage_suspected'] = True
            results['evidence'].append(f"Actual-Predicted correlation is suspiciously high: {correlation:.4f}")

    # Check 3: Are predictions constant across horizons? (OK for univariate, suspicious for feature-based)
    key_col = key_columns[0] if len(key_columns) == 1 else None

    if key_col and key_col in predictions_df.columns:
        # Sample a few keys
        sample_keys = predictions_df[key_col].unique()[:10]

        constant_keys = 0
        for key in sample_keys:
            key_preds = predictions_df[predictions_df[key_col] == key]['predicted'].values
            if len(key_preds) > 1 and np.std(key_preds) < 1e-8:
                constant_keys += 1

        if constant_keys > len(sample_keys) * 0.8:
            results['evidence'].append(f"Most keys have constant predictions (may indicate non-recursive forecasting)")

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RecursiveForecaster',
    'RecursiveForecastConfig',
    'generate_recursive_predictions',
    'identify_lag_columns',
    'identify_rolling_columns',
    'compute_wape',
    'validate_recursive_predictions',
    'detect_feature_leakage',
]
