"""
State-of-the-Art Model Diagnostics Utilities for Demand Forecasting.

This module provides comprehensive diagnostic utilities for analyzing
forecast model performance at all levels of granularity.

DESIGN PRINCIPLE: Every function is generic, robust, and works on any data.

USAGE:
>>> from utils.diagnostics import (
...     # High-level pipeline (does EVERYTHING)
...     run_diagnostic_pipeline,
...
...     # Core metrics
...     compute_wape, compute_mae, compute_rmse, compute_bias, compute_smape,
...     compute_mape, compute_mase, compute_all_metrics,
...
...     # Grouped metrics
...     compute_metrics_by_group, compute_wape_by_group,
...     compute_metrics_at_multiple_levels,
...
...     # Error decomposition
...     decompose_forecast_error, compute_bias_variance_decomposition,
...     analyze_systematic_vs_random_error,
...
...     # Temporal analysis
...     analyze_error_over_time, detect_error_seasonality,
...     compute_rolling_metrics, analyze_forecast_horizon_decay,
...
...     # Top/Bottom analysis
...     identify_top_bottom_performers, analyze_performer_characteristics,
...     explain_performance_drivers,
...
...     # Root cause analysis
...     diagnose_forecast_failures, identify_error_patterns,
...     analyze_error_by_demand_pattern,
...
...     # Comparison utilities
...     compare_models, compare_segments, compare_periods,
...     compute_model_lift,
...
...     # Visualization
...     plot_actual_vs_predicted, plot_error_distribution,
...     plot_error_over_time, plot_segment_comparison,
...     plot_wape_heatmap, plot_forecast_fan_chart,
...
...     # Outlier detection
...     detect_forecast_outliers, identify_anomalous_errors,
...
...     # Model verdict
...     generate_model_verdict, create_recommendations,
...     assess_deployment_readiness,
...
...     # Reporting
...     create_diagnostic_report, create_executive_summary,
... )

Author: FEU-Agentic-Forecasting Demand Forecasting System
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# WAPE thresholds for quality assessment
WAPE_EXCELLENT = 0.10  # < 10%
WAPE_GOOD = 0.20       # < 20%
WAPE_ACCEPTABLE = 0.30 # < 30%
WAPE_POOR = 0.50       # < 50%
# > 50% is considered very poor

# Syntetos-Boylan thresholds
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetricsResult:
    """Container for comprehensive metrics."""
    wape: float
    mae: float
    rmse: float
    bias: float
    smape: float
    mape: Optional[float] = None
    mase: Optional[float] = None
    n_obs: int = 0
    sum_actual: float = 0.0
    sum_predicted: float = 0.0
    quality_tier: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'wape': self.wape,
            'mae': self.mae,
            'rmse': self.rmse,
            'bias': self.bias,
            'smape': self.smape,
            'mape': self.mape,
            'mase': self.mase,
            'n_obs': self.n_obs,
            'sum_actual': self.sum_actual,
            'sum_predicted': self.sum_predicted,
            'quality_tier': self.quality_tier,
        }


@dataclass
class ErrorDecomposition:
    """Decomposition of forecast error."""
    total_error: float
    bias_component: float
    variance_component: float
    bias_pct: float
    variance_pct: float
    systematic_error: float
    random_error: float


@dataclass
class PerformerAnalysis:
    """Analysis of top/bottom performers."""
    key: Any
    wape: float
    rank: int
    demand_pattern: str
    volume_tier: str
    characteristics: Dict[str, Any]
    explanation: str


@dataclass
class ModelVerdict:
    """Overall model verdict and recommendations."""
    overall_wape: float
    quality_tier: str
    deployment_ready: bool
    confidence_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    segment_verdicts: Dict[str, str]


@dataclass
class DiagnosticPipelineResult:
    """Complete diagnostic pipeline results."""
    overall_metrics: MetricsResult
    segment_metrics: pd.DataFrame
    key_metrics: pd.DataFrame
    top_performers: List[PerformerAnalysis]
    bottom_performers: List[PerformerAnalysis]
    error_decomposition: ErrorDecomposition
    temporal_analysis: Dict[str, Any]
    model_verdict: ModelVerdict
    output_dir: str


# =============================================================================
# 1. CORE METRICS
# =============================================================================

def compute_wape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|actual - predicted|) / sum(|actual|)

    This is the PRIMARY metric for demand forecasting as it:
    1. Handles zeros gracefully
    2. Weights errors by actual volume
    3. Is interpretable as % of total demand

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        WAPE value (0 to inf, lower is better)

    Example
    -------
    >>> wape = compute_wape(df['actual'], df['predicted'])
    >>> print(f"WAPE: {wape:.2%}")
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    abs_error = np.sum(np.abs(y_true - y_pred))
    total_actual = np.sum(np.abs(y_true))

    if total_actual < 1e-10:
        return 0.0 if abs_error < 1e-10 else 1.0

    return abs_error / total_actual


def compute_mae(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Mean Absolute Error (MAE).

    MAE = mean(|actual - predicted|)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    RMSE = sqrt(mean((actual - predicted)^2))

    More sensitive to outliers than MAE.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_bias(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Bias (mean error).

    Positive = over-predicting (forecast too high)
    Negative = under-predicting (forecast too low)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(y_pred - y_true))


def compute_smape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE = mean(2 * |actual - pred| / (|actual| + |pred|))

    Bounded between 0 and 2 (or 0-200%).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    mask = denominator > 1e-10
    if not np.any(mask):
        return 0.0

    smape_values = np.zeros_like(y_true, dtype=float)
    smape_values[mask] = 2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]

    return float(np.mean(smape_values))


def compute_mape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    handle_zeros: str = 'exclude',
) -> Optional[float]:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    MAPE = mean(|actual - pred| / |actual|) * 100

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    handle_zeros : str
        How to handle zero actuals: 'exclude', 'epsilon', 'none'

    Returns
    -------
    float or None
        MAPE value (returns None if cannot compute)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if handle_zeros == 'exclude':
        mask = np.abs(y_true) > 1e-10
        if not np.any(mask):
            return None
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    elif handle_zeros == 'epsilon':
        y_true = np.clip(np.abs(y_true), 1e-10, None)

    if len(y_true) == 0:
        return None

    mape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    return float(mape)


def compute_mase(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    seasonality: int = 1,
) -> Optional[float]:
    """
    Compute Mean Absolute Scaled Error (MASE).

    MASE scales errors by the in-sample naive forecast error.
    MASE < 1 means better than naive.

    Parameters
    ----------
    y_true : array-like
        Actual values (test period)
    y_pred : array-like
        Predicted values
    y_train : array-like, optional
        Training data for scaling
    seasonality : int
        Seasonality period for naive forecast

    Returns
    -------
    float or None
        MASE value (None if cannot compute)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_train is None:
        return None

    y_train = np.asarray(y_train).flatten()

    # Compute naive forecast error (seasonal random walk)
    if len(y_train) <= seasonality:
        return None

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale < 1e-10:
        return None

    mae = np.mean(np.abs(y_true - y_pred))
    mase = mae / scale

    return float(mase)


def compute_all_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_train: Optional[Union[pd.Series, np.ndarray]] = None,
) -> MetricsResult:
    """
    Compute all standard metrics at once.

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    y_train : array-like, optional
        Training data for MASE computation

    Returns
    -------
    MetricsResult
        Container with all metrics

    Example
    -------
    >>> metrics = compute_all_metrics(df['actual'], df['predicted'])
    >>> print(f"WAPE: {metrics.wape:.2%}, Quality: {metrics.quality_tier}")
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    wape = compute_wape(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    bias = compute_bias(y_true, y_pred)
    smape = compute_smape(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    mase = compute_mase(y_true, y_pred, y_train) if y_train is not None else None

    # Determine quality tier
    if wape <= WAPE_EXCELLENT:
        quality_tier = 'EXCELLENT'
    elif wape <= WAPE_GOOD:
        quality_tier = 'GOOD'
    elif wape <= WAPE_ACCEPTABLE:
        quality_tier = 'ACCEPTABLE'
    elif wape <= WAPE_POOR:
        quality_tier = 'POOR'
    else:
        quality_tier = 'VERY_POOR'

    return MetricsResult(
        wape=wape,
        mae=mae,
        rmse=rmse,
        bias=bias,
        smape=smape,
        mape=mape,
        mase=mase,
        n_obs=len(y_true),
        sum_actual=float(np.sum(y_true)),
        sum_predicted=float(np.sum(y_pred)),
        quality_tier=quality_tier,
    )


def classify_forecast_quality(wape: float) -> str:
    """
    Classify forecast quality based on WAPE.

    Parameters
    ----------
    wape : float
        WAPE value

    Returns
    -------
    str
        Quality tier: 'EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'VERY_POOR'
    """
    if wape <= WAPE_EXCELLENT:
        return 'EXCELLENT'
    elif wape <= WAPE_GOOD:
        return 'GOOD'
    elif wape <= WAPE_ACCEPTABLE:
        return 'ACCEPTABLE'
    elif wape <= WAPE_POOR:
        return 'POOR'
    else:
        return 'VERY_POOR'


# =============================================================================
# 2. GROUPED METRICS
# =============================================================================

def compute_metrics_by_group(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    actual_col: str,
    pred_col: str,
    include_count: bool = True,
) -> pd.DataFrame:
    """
    Compute comprehensive metrics by group(s).

    CRITICAL: Aggregates errors FIRST, then computes metrics.
    This is the correct method for grouped metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    group_cols : str or List[str]
        Column(s) to group by
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    include_count : bool
        Whether to include observation counts

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics per group

    Example
    -------
    >>> metrics_by_segment = compute_metrics_by_group(df, 'segment_id', 'actual', 'predicted')
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    df = df.copy()

    # Compute error columns
    df['_abs_error'] = np.abs(df[actual_col] - df[pred_col])
    df['_sq_error'] = (df[actual_col] - df[pred_col]) ** 2
    df['_error'] = df[pred_col] - df[actual_col]

    # Aggregate
    agg_dict = {
        '_abs_error': 'sum',
        '_sq_error': 'sum',
        '_error': 'sum',
        actual_col: 'sum',
        pred_col: 'sum',
    }

    if include_count:
        agg_dict['_count'] = (actual_col, 'count')

    result = df.groupby(group_cols).agg(**{
        'sum_abs_error': ('_abs_error', 'sum'),
        'sum_sq_error': ('_sq_error', 'sum'),
        'sum_error': ('_error', 'sum'),
        'sum_actual': (actual_col, 'sum'),
        'sum_predicted': (pred_col, 'sum'),
        'n_obs': (actual_col, 'count'),
    }).reset_index()

    # Compute metrics
    result['wape'] = result['sum_abs_error'] / (np.abs(result['sum_actual']) + 1e-10)
    result['mae'] = result['sum_abs_error'] / result['n_obs']
    result['rmse'] = np.sqrt(result['sum_sq_error'] / result['n_obs'])
    result['bias'] = result['sum_error'] / result['n_obs']
    result['bias_pct'] = result['sum_error'] / (np.abs(result['sum_actual']) + 1e-10) * 100

    # Add quality tier
    result['quality_tier'] = result['wape'].apply(classify_forecast_quality)

    return result


def compute_wape_by_group(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Compute WAPE by group(s) - simplified version.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    group_cols : str or List[str]
        Column(s) to group by
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    pd.DataFrame
        DataFrame with WAPE per group
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    df = df.copy()
    df['_abs_error'] = np.abs(df[actual_col] - df[pred_col])

    result = df.groupby(group_cols).agg({
        '_abs_error': 'sum',
        actual_col: 'sum',
    }).reset_index()

    result['wape'] = result['_abs_error'] / (np.abs(result[actual_col]) + 1e-10)
    result = result.rename(columns={actual_col: 'sum_actual', '_abs_error': 'sum_abs_error'})

    return result


def compute_metrics_at_multiple_levels(
    df: pd.DataFrame,
    level_configs: List[Dict[str, Any]],
    actual_col: str,
    pred_col: str,
) -> Dict[str, pd.DataFrame]:
    """
    Compute metrics at multiple aggregation levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    level_configs : List[Dict]
        List of level configurations:
        [{'name': 'overall', 'group_cols': None},
         {'name': 'segment', 'group_cols': ['segment_id']},
         {'name': 'key', 'group_cols': ['key']}]
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of metrics DataFrames by level name

    Example
    -------
    >>> levels = [
    ...     {'name': 'overall', 'group_cols': None},
    ...     {'name': 'segment', 'group_cols': ['segment_id']},
    ...     {'name': 'key', 'group_cols': ['key']},
    ... ]
    >>> metrics = compute_metrics_at_multiple_levels(df, levels, 'actual', 'predicted')
    """
    results = {}

    for config in level_configs:
        name = config['name']
        group_cols = config.get('group_cols')

        if group_cols is None:
            # Overall metrics
            metrics = compute_all_metrics(df[actual_col], df[pred_col])
            results[name] = pd.DataFrame([metrics.to_dict()])
        else:
            results[name] = compute_metrics_by_group(df, group_cols, actual_col, pred_col)

    return results


# =============================================================================
# 3. ERROR DECOMPOSITION
# =============================================================================

def decompose_forecast_error(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> ErrorDecomposition:
    """
    Decompose forecast error into bias and variance components.

    Total Error = Bias^2 + Variance + Irreducible Error

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    ErrorDecomposition
        Error decomposition results

    Example
    -------
    >>> decomp = decompose_forecast_error(df['actual'], df['predicted'])
    >>> print(f"Bias component: {decomp.bias_pct:.1f}%")
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Total error (MSE)
    errors = y_true - y_pred
    total_error = np.mean(errors ** 2)

    # Bias component
    bias = np.mean(y_pred - y_true)
    bias_component = bias ** 2

    # Variance component
    variance_component = np.var(errors)

    # Percentages
    if total_error > 1e-10:
        bias_pct = (bias_component / total_error) * 100
        variance_pct = (variance_component / total_error) * 100
    else:
        bias_pct = 0.0
        variance_pct = 0.0

    # Systematic vs random
    # Systematic: consistent over/under prediction
    # Random: unpredictable variation
    systematic_error = np.abs(bias)
    random_error = np.std(errors)

    return ErrorDecomposition(
        total_error=total_error,
        bias_component=bias_component,
        variance_component=variance_component,
        bias_pct=bias_pct,
        variance_pct=variance_pct,
        systematic_error=systematic_error,
        random_error=random_error,
    )


def compute_bias_variance_decomposition(
    df: pd.DataFrame,
    group_col: str,
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Compute bias-variance decomposition by group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    group_col : str
        Grouping column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    pd.DataFrame
        DataFrame with bias-variance decomposition per group
    """
    results = []

    for group in df[group_col].unique():
        subset = df[df[group_col] == group]
        decomp = decompose_forecast_error(subset[actual_col], subset[pred_col])

        results.append({
            group_col: group,
            'total_mse': decomp.total_error,
            'bias_component': decomp.bias_component,
            'variance_component': decomp.variance_component,
            'bias_pct': decomp.bias_pct,
            'variance_pct': decomp.variance_pct,
            'systematic_error': decomp.systematic_error,
            'random_error': decomp.random_error,
        })

    return pd.DataFrame(results)


def analyze_systematic_vs_random_error(
    df: pd.DataFrame,
    key_col: str,
    actual_col: str,
    pred_col: str,
) -> Dict[str, Any]:
    """
    Analyze the split between systematic and random errors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    key_col : str
        Key column for grouping
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    Dict[str, Any]
        Analysis of systematic vs random errors
    """
    # Compute bias per key
    key_bias = df.groupby(key_col).apply(
        lambda x: np.mean(x[pred_col] - x[actual_col])
    ).reset_index(name='bias')

    # Keys with systematic bias
    positive_bias_keys = key_bias[key_bias['bias'] > 0]
    negative_bias_keys = key_bias[key_bias['bias'] < 0]

    # Compute overall systematic vs random
    overall_decomp = decompose_forecast_error(df[actual_col], df[pred_col])

    return {
        'overall_decomposition': {
            'systematic_error': overall_decomp.systematic_error,
            'random_error': overall_decomp.random_error,
            'bias_pct': overall_decomp.bias_pct,
            'variance_pct': overall_decomp.variance_pct,
        },
        'keys_with_positive_bias': len(positive_bias_keys),
        'keys_with_negative_bias': len(negative_bias_keys),
        'mean_positive_bias': float(positive_bias_keys['bias'].mean()) if len(positive_bias_keys) > 0 else 0,
        'mean_negative_bias': float(negative_bias_keys['bias'].mean()) if len(negative_bias_keys) > 0 else 0,
        'bias_consistency': 'SYSTEMATIC' if overall_decomp.bias_pct > 30 else 'RANDOM',
    }


# =============================================================================
# 4. TEMPORAL ANALYSIS
# =============================================================================

def analyze_error_over_time(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Analyze forecast error over time periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals, predictions, and dates
    date_col : str
        Date column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    pd.DataFrame
        Error metrics by time period
    """
    return compute_metrics_by_group(df, date_col, actual_col, pred_col)


def detect_error_seasonality(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    seasonality_periods: List[int] = None,
) -> Dict[str, Any]:
    """
    Detect seasonality in forecast errors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    date_col : str
        Date column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    seasonality_periods : List[int], optional
        Periods to test (default: [4, 13, 26, 52])

    Returns
    -------
    Dict[str, Any]
        Seasonality analysis results
    """
    if seasonality_periods is None:
        seasonality_periods = [4, 13, 26, 52]

    df = df.copy().sort_values(date_col)

    # Compute error
    df['_error'] = df[pred_col] - df[actual_col]

    # Aggregate error by period
    period_errors = df.groupby(date_col)['_error'].mean().values

    if len(period_errors) < max(seasonality_periods) * 2:
        return {'detected_seasonality': None, 'insufficient_data': True}

    results = {}

    for period in seasonality_periods:
        if len(period_errors) >= period * 2:
            # Simple autocorrelation test
            shifted = period_errors[period:]
            original = period_errors[:-period]

            correlation = np.corrcoef(original, shifted)[0, 1]
            results[f'period_{period}_correlation'] = float(correlation)

    # Find strongest seasonality
    if results:
        strongest_period = max(results.keys(), key=lambda k: abs(results[k]))
        strongest_correlation = results[strongest_period]
    else:
        strongest_period = None
        strongest_correlation = 0

    return {
        'period_correlations': results,
        'strongest_seasonality': strongest_period,
        'strongest_correlation': strongest_correlation,
        'detected_seasonality': abs(strongest_correlation) > 0.3 if strongest_correlation else False,
    }


def compute_rolling_metrics(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    window: int = 4,
) -> pd.DataFrame:
    """
    Compute rolling window metrics over time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    date_col : str
        Date column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    window : int
        Rolling window size

    Returns
    -------
    pd.DataFrame
        Rolling metrics by date
    """
    df = df.copy().sort_values(date_col)

    # Compute cumulative error metrics
    df['_abs_error'] = np.abs(df[actual_col] - df[pred_col])

    # Rolling aggregation
    period_data = df.groupby(date_col).agg({
        '_abs_error': 'sum',
        actual_col: 'sum',
    }).reset_index()

    period_data['rolling_abs_error'] = period_data['_abs_error'].rolling(window, min_periods=1).sum()
    period_data['rolling_actual'] = period_data[actual_col].rolling(window, min_periods=1).sum()
    period_data['rolling_wape'] = period_data['rolling_abs_error'] / (period_data['rolling_actual'] + 1e-10)

    return period_data


def analyze_forecast_horizon_decay(
    df: pd.DataFrame,
    horizon_col: str,
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Analyze how forecast accuracy decays with horizon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with forecast horizon information
    horizon_col : str
        Column indicating forecast horizon (1, 2, 3, etc.)
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    pd.DataFrame
        Metrics by forecast horizon
    """
    return compute_metrics_by_group(df, horizon_col, actual_col, pred_col)


# =============================================================================
# 5. TOP/BOTTOM PERFORMER ANALYSIS
# =============================================================================

def identify_top_bottom_performers(
    df: pd.DataFrame,
    key_col: str,
    actual_col: str,
    pred_col: str,
    n: int = 10,
    min_volume: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify top N and bottom N performers based on WAPE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    key_col : str
        Key column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    n : int
        Number of top/bottom to return
    min_volume : float, optional
        Minimum volume to consider (filters low-volume keys)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Top performers (best WAPE) and bottom performers (worst WAPE)

    Example
    -------
    >>> top, bottom = identify_top_bottom_performers(df, 'key', 'actual', 'predicted', n=10)
    """
    # Compute metrics by key
    key_metrics = compute_metrics_by_group(df, key_col, actual_col, pred_col)

    # Filter by minimum volume if specified
    if min_volume is not None:
        key_metrics = key_metrics[key_metrics['sum_actual'] >= min_volume]

    # Rank by WAPE
    key_metrics = key_metrics.sort_values('wape')
    key_metrics['rank'] = range(1, len(key_metrics) + 1)

    # Get top and bottom
    top_performers = key_metrics.head(n).copy()
    bottom_performers = key_metrics.tail(n).sort_values('wape', ascending=False).copy()

    # Add performance label
    top_performers['performance'] = 'TOP'
    bottom_performers['performance'] = 'BOTTOM'

    return top_performers, bottom_performers


def analyze_performer_characteristics(
    performers_df: pd.DataFrame,
    characteristics_df: pd.DataFrame,
    key_col: str,
    characteristic_cols: List[str],
) -> pd.DataFrame:
    """
    Analyze characteristics of top/bottom performers.

    Parameters
    ----------
    performers_df : pd.DataFrame
        DataFrame of top or bottom performers (from identify_top_bottom_performers)
    characteristics_df : pd.DataFrame
        DataFrame with key characteristics (e.g., from per_key_metrics)
    key_col : str
        Key column for joining
    characteristic_cols : List[str]
        Characteristic columns to analyze

    Returns
    -------
    pd.DataFrame
        Performers with their characteristics

    Example
    -------
    >>> top_with_chars = analyze_performer_characteristics(
    ...     top_performers, per_key_metrics, 'key', ['cv', 'zero_fraction', 'mean_demand']
    ... )
    """
    # Join characteristics
    merged = performers_df.merge(
        characteristics_df[[key_col] + characteristic_cols],
        on=key_col,
        how='left'
    )

    return merged


def explain_performance_drivers(
    top_performers: pd.DataFrame,
    bottom_performers: pd.DataFrame,
    characteristic_cols: List[str],
) -> Dict[str, Any]:
    """
    Explain what drives good vs poor performance.

    Parameters
    ----------
    top_performers : pd.DataFrame
        Top performers with characteristics
    bottom_performers : pd.DataFrame
        Bottom performers with characteristics
    characteristic_cols : List[str]
        Characteristic columns to analyze

    Returns
    -------
    Dict[str, Any]
        Analysis of performance drivers

    Example
    -------
    >>> drivers = explain_performance_drivers(top_df, bottom_df, ['cv', 'zero_fraction'])
    """
    analysis = {}

    for col in characteristic_cols:
        if col not in top_performers.columns or col not in bottom_performers.columns:
            continue

        top_mean = top_performers[col].mean()
        bottom_mean = bottom_performers[col].mean()

        # Statistical test (if enough samples)
        if len(top_performers) >= 5 and len(bottom_performers) >= 5:
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(
                    top_performers[col].dropna(),
                    bottom_performers[col].dropna(),
                    equal_var=False
                )
                significant = p_value < 0.05
            except Exception:
                t_stat, p_value, significant = None, None, None
        else:
            t_stat, p_value, significant = None, None, None

        analysis[col] = {
            'top_mean': float(top_mean) if pd.notna(top_mean) else None,
            'bottom_mean': float(bottom_mean) if pd.notna(bottom_mean) else None,
            'difference': float(bottom_mean - top_mean) if pd.notna(top_mean) and pd.notna(bottom_mean) else None,
            'significant_difference': significant,
            'p_value': float(p_value) if p_value is not None else None,
        }

    return analysis


# =============================================================================
# 6. ROOT CAUSE ANALYSIS
# =============================================================================

def diagnose_forecast_failures(
    df: pd.DataFrame,
    key_col: str,
    actual_col: str,
    pred_col: str,
    wape_threshold: float = 0.50,
    characteristic_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Diagnose why forecasts are failing for certain keys.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals, predictions, and characteristics
    key_col : str
        Key column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    wape_threshold : float
        WAPE threshold for "failure" (default 50%)
    characteristic_cols : List[str], optional
        Columns to analyze for patterns

    Returns
    -------
    Dict[str, Any]
        Diagnosis of forecast failures
    """
    # Compute key metrics
    key_metrics = compute_metrics_by_group(df, key_col, actual_col, pred_col)

    # Identify failures
    failures = key_metrics[key_metrics['wape'] > wape_threshold]
    successes = key_metrics[key_metrics['wape'] <= wape_threshold]

    n_failures = len(failures)
    n_total = len(key_metrics)
    failure_rate = n_failures / n_total if n_total > 0 else 0

    diagnosis = {
        'n_failures': n_failures,
        'n_successes': len(successes),
        'failure_rate': failure_rate,
        'wape_threshold': wape_threshold,
        'failure_keys': failures[key_col].tolist()[:20],  # Top 20 failures
        'failure_characteristics': {},
    }

    # Analyze failure characteristics
    if characteristic_cols:
        failure_chars = df[df[key_col].isin(failures[key_col])]
        success_chars = df[df[key_col].isin(successes[key_col])]

        for col in characteristic_cols:
            if col in df.columns:
                fail_mean = failure_chars[col].mean()
                success_mean = success_chars[col].mean()

                diagnosis['failure_characteristics'][col] = {
                    'failure_mean': float(fail_mean) if pd.notna(fail_mean) else None,
                    'success_mean': float(success_mean) if pd.notna(success_mean) else None,
                }

    return diagnosis


def identify_error_patterns(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    pattern_cols: List[str],
) -> pd.DataFrame:
    """
    Identify patterns in forecast errors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    pattern_cols : List[str]
        Columns to analyze for error patterns

    Returns
    -------
    pd.DataFrame
        Error patterns by feature combinations
    """
    results = []

    for col in pattern_cols:
        if col not in df.columns:
            continue

        # Compute metrics by this column
        metrics = compute_metrics_by_group(df, col, actual_col, pred_col)

        for _, row in metrics.iterrows():
            results.append({
                'pattern_column': col,
                'pattern_value': row[col],
                'wape': row['wape'],
                'n_obs': row['n_obs'],
                'sum_actual': row['sum_actual'],
            })

    return pd.DataFrame(results).sort_values('wape', ascending=False)


def analyze_error_by_demand_pattern(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    demand_pattern_col: str = 'demand_pattern',
) -> pd.DataFrame:
    """
    Analyze forecast errors by demand pattern (smooth, erratic, intermittent, lumpy).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals, predictions, and demand pattern
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    demand_pattern_col : str
        Demand pattern column

    Returns
    -------
    pd.DataFrame
        Metrics by demand pattern
    """
    if demand_pattern_col not in df.columns:
        logger.warning(f"Column {demand_pattern_col} not found in DataFrame")
        return pd.DataFrame()

    return compute_metrics_by_group(df, demand_pattern_col, actual_col, pred_col)


# =============================================================================
# 7. MODEL COMPARISON
# =============================================================================

def compare_models(
    results: Dict[str, pd.DataFrame],
    actual_col: str,
    pred_col: str,
    model_name_col: str = 'model_name',
) -> pd.DataFrame:
    """
    Compare multiple models' performance.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Dictionary of model results: {'model_name': forecast_df}
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    model_name_col : str
        Column name for model identifier

    Returns
    -------
    pd.DataFrame
        Comparison of model performance

    Example
    -------
    >>> comparison = compare_models(
    ...     {'lightgbm': lgb_df, 'croston': croston_df},
    ...     'actual', 'predicted'
    ... )
    """
    comparisons = []

    for model_name, df in results.items():
        metrics = compute_all_metrics(df[actual_col], df[pred_col])

        comparisons.append({
            model_name_col: model_name,
            'wape': metrics.wape,
            'mae': metrics.mae,
            'rmse': metrics.rmse,
            'bias': metrics.bias,
            'smape': metrics.smape,
            'n_obs': metrics.n_obs,
            'quality_tier': metrics.quality_tier,
        })

    comparison_df = pd.DataFrame(comparisons).sort_values('wape')
    comparison_df['rank'] = range(1, len(comparison_df) + 1)

    return comparison_df


def compare_segments(
    df: pd.DataFrame,
    segment_col: str,
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Compare performance across segments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    segment_col : str
        Segment column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    pd.DataFrame
        Segment comparison with rankings
    """
    metrics = compute_metrics_by_group(df, segment_col, actual_col, pred_col)
    metrics = metrics.sort_values('wape')
    metrics['rank'] = range(1, len(metrics) + 1)

    return metrics


def compare_periods(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Compare performance across time periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    date_col : str
        Date column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column

    Returns
    -------
    pd.DataFrame
        Period comparison
    """
    return compute_metrics_by_group(df, date_col, actual_col, pred_col)


def compute_model_lift(
    model_wape: float,
    baseline_wape: float,
) -> Dict[str, float]:
    """
    Compute lift of model over baseline.

    Parameters
    ----------
    model_wape : float
        Model's WAPE
    baseline_wape : float
        Baseline WAPE (e.g., naive forecast)

    Returns
    -------
    Dict[str, float]
        Lift metrics
    """
    absolute_improvement = baseline_wape - model_wape
    relative_improvement = absolute_improvement / baseline_wape if baseline_wape > 0 else 0

    return {
        'model_wape': model_wape,
        'baseline_wape': baseline_wape,
        'absolute_improvement': absolute_improvement,
        'relative_improvement_pct': relative_improvement * 100,
        'lift_ratio': baseline_wape / model_wape if model_wape > 0 else float('inf'),
    }


# =============================================================================
# 8. OUTLIER DETECTION
# =============================================================================

def detect_forecast_outliers(
    df: pd.DataFrame,
    key_col: str,
    actual_col: str,
    pred_col: str,
    method: str = 'iqr',
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Detect outlier forecast errors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    key_col : str
        Key column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    method : str
        'iqr' (Interquartile Range) or 'zscore'
    threshold : float
        Threshold for outlier detection (1.5 for IQR, 3 for z-score)

    Returns
    -------
    pd.DataFrame
        Outlier observations with flags
    """
    df = df.copy()
    df['_abs_error'] = np.abs(df[actual_col] - df[pred_col])

    if method == 'iqr':
        q1 = df['_abs_error'].quantile(0.25)
        q3 = df['_abs_error'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        df['is_outlier'] = (df['_abs_error'] < lower_bound) | (df['_abs_error'] > upper_bound)
        df['outlier_score'] = (df['_abs_error'] - q3) / iqr

    elif method == 'zscore':
        mean_error = df['_abs_error'].mean()
        std_error = df['_abs_error'].std()

        df['zscore'] = (df['_abs_error'] - mean_error) / (std_error + 1e-10)
        df['is_outlier'] = np.abs(df['zscore']) > threshold
        df['outlier_score'] = df['zscore']

    else:
        raise ValueError(f"Unknown method: {method}")

    return df[df['is_outlier']]


def identify_anomalous_errors(
    df: pd.DataFrame,
    key_col: str,
    actual_col: str,
    pred_col: str,
    n_anomalies: int = 20,
) -> pd.DataFrame:
    """
    Identify most anomalous forecast errors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    key_col : str
        Key column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    n_anomalies : int
        Number of top anomalies to return

    Returns
    -------
    pd.DataFrame
        Top anomalous observations
    """
    df = df.copy()
    df['_abs_error'] = np.abs(df[actual_col] - df[pred_col])
    df['_pct_error'] = df['_abs_error'] / (np.abs(df[actual_col]) + 1e-10)

    # Sort by absolute error
    return df.nlargest(n_anomalies, '_abs_error')


# =============================================================================
# 9. MODEL VERDICT
# =============================================================================

def generate_model_verdict(
    overall_metrics: MetricsResult,
    segment_metrics: pd.DataFrame,
    error_decomposition: ErrorDecomposition,
    target_wape: float = 0.20,
) -> ModelVerdict:
    """
    Generate comprehensive model verdict with recommendations.

    Parameters
    ----------
    overall_metrics : MetricsResult
        Overall performance metrics
    segment_metrics : pd.DataFrame
        Per-segment metrics
    error_decomposition : ErrorDecomposition
        Error decomposition results
    target_wape : float
        Target WAPE for deployment readiness

    Returns
    -------
    ModelVerdict
        Comprehensive verdict with recommendations

    Example
    -------
    >>> verdict = generate_model_verdict(metrics, seg_metrics, decomp)
    >>> print(f"Deployment ready: {verdict.deployment_ready}")
    """
    strengths = []
    weaknesses = []
    recommendations = []

    # Overall performance assessment
    wape = overall_metrics.wape
    quality_tier = overall_metrics.quality_tier

    if wape <= WAPE_EXCELLENT:
        strengths.append(f"Excellent overall accuracy (WAPE: {wape:.1%})")
    elif wape <= WAPE_GOOD:
        strengths.append(f"Good overall accuracy (WAPE: {wape:.1%})")
    elif wape <= WAPE_ACCEPTABLE:
        weaknesses.append(f"Acceptable but improvable accuracy (WAPE: {wape:.1%})")
        recommendations.append("Consider additional feature engineering or model tuning")
    else:
        weaknesses.append(f"Poor accuracy requires attention (WAPE: {wape:.1%})")
        recommendations.append("Investigate data quality and model selection")

    # Bias assessment
    if error_decomposition.bias_pct > 30:
        if overall_metrics.bias > 0:
            weaknesses.append(f"Systematic over-prediction (Bias: {overall_metrics.bias:.2f})")
            recommendations.append("Adjust for positive bias in forecasts")
        else:
            weaknesses.append(f"Systematic under-prediction (Bias: {overall_metrics.bias:.2f})")
            recommendations.append("Adjust for negative bias in forecasts")
    else:
        strengths.append("Low systematic bias")

    # Segment analysis
    if len(segment_metrics) > 0 and 'wape' in segment_metrics.columns:
        best_segment_wape = segment_metrics['wape'].min()
        worst_segment_wape = segment_metrics['wape'].max()
        segment_spread = worst_segment_wape - best_segment_wape

        if segment_spread > 0.20:
            weaknesses.append(f"High variance across segments ({segment_spread:.1%} spread)")
            recommendations.append("Focus on improving worst-performing segments")
        else:
            strengths.append("Consistent performance across segments")

        # Identify problematic segments
        poor_segments = segment_metrics[segment_metrics['wape'] > WAPE_POOR]
        if len(poor_segments) > 0:
            # Safely find segment column (avoid IndexError if only metric columns exist)
            metric_cols = {'wape', 'mae', 'rmse', 'bias', 'n_obs', 'sum_actual', 'sum_predicted',
                          'sum_abs_error', 'sum_sq_error', 'sum_error', 'bias_pct', 'quality_tier'}
            segment_cols = [c for c in segment_metrics.columns if c not in metric_cols]
            if segment_cols:
                segment_col = segment_cols[0]
            weaknesses.append(f"{len(poor_segments)} segments with WAPE > {WAPE_POOR:.0%}")

    # Deployment readiness
    deployment_ready = wape <= target_wape and error_decomposition.bias_pct < 50

    # Confidence score (0-100)
    confidence_score = max(0, min(100, (1 - wape) * 100 - error_decomposition.bias_pct / 2))

    # Segment verdicts
    segment_verdicts = {}
    if len(segment_metrics) > 0 and 'wape' in segment_metrics.columns:
        # Safely find segment column (avoid IndexError if only metric columns exist)
        metric_cols = {'wape', 'mae', 'rmse', 'bias', 'n_obs', 'sum_actual', 'sum_predicted',
                      'sum_abs_error', 'sum_sq_error', 'sum_error', 'bias_pct', 'quality_tier'}
        segment_cols = [c for c in segment_metrics.columns if c not in metric_cols]
        if segment_cols:
            segment_col = segment_cols[0]
            for _, row in segment_metrics.iterrows():
                segment_verdicts[str(row[segment_col])] = classify_forecast_quality(row['wape'])

    return ModelVerdict(
        overall_wape=wape,
        quality_tier=quality_tier,
        deployment_ready=deployment_ready,
        confidence_score=confidence_score,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
        segment_verdicts=segment_verdicts,
    )


def create_recommendations(
    verdict: ModelVerdict,
    top_performers: pd.DataFrame,
    bottom_performers: pd.DataFrame,
) -> List[str]:
    """
    Create actionable recommendations based on diagnostics.

    Parameters
    ----------
    verdict : ModelVerdict
        Model verdict
    top_performers : pd.DataFrame
        Top performing keys
    bottom_performers : pd.DataFrame
        Bottom performing keys

    Returns
    -------
    List[str]
        List of recommendations
    """
    recommendations = list(verdict.recommendations)

    # Add recommendations based on performers
    if len(bottom_performers) > 0 and 'wape' in bottom_performers.columns:
        avg_bottom_wape = bottom_performers['wape'].mean()
        if avg_bottom_wape > 0.80:
            recommendations.append(
                f"Bottom {len(bottom_performers)} keys have {avg_bottom_wape:.1%} average WAPE - "
                "consider specialized models or exclusion"
            )

    # Add deployment recommendations
    if verdict.deployment_ready:
        recommendations.append("Model is ready for deployment with monitoring")
    else:
        recommendations.append("Model requires improvement before deployment")

    return recommendations


def assess_deployment_readiness(
    overall_wape: float,
    target_wape: float = 0.20,
    bias: float = 0.0,
    max_bias_pct: float = 10.0,
    segment_wapes: Optional[List[float]] = None,
    max_segment_wape: float = 0.50,
) -> Dict[str, Any]:
    """
    Assess if model is ready for production deployment.

    Parameters
    ----------
    overall_wape : float
        Overall WAPE
    target_wape : float
        Maximum acceptable WAPE
    bias : float
        Model bias
    max_bias_pct : float
        Maximum acceptable bias percentage
    segment_wapes : List[float], optional
        WAPE per segment
    max_segment_wape : float
        Maximum acceptable segment WAPE

    Returns
    -------
    Dict[str, Any]
        Deployment readiness assessment
    """
    checks = {
        'overall_wape_check': overall_wape <= target_wape,
        'bias_check': True,  # Will update below
        'segment_check': True,  # Will update below
    }

    issues = []

    # Overall WAPE check
    if not checks['overall_wape_check']:
        issues.append(f"Overall WAPE ({overall_wape:.1%}) exceeds target ({target_wape:.1%})")

    # Bias check
    # bias_pct would need sum_actual to compute properly
    # For simplicity, use absolute bias threshold
    if abs(bias) > max_bias_pct:
        checks['bias_check'] = False
        direction = 'over' if bias > 0 else 'under'
        issues.append(f"Systematic {direction}-prediction detected (Bias: {bias:.2f})")

    # Segment check
    if segment_wapes:
        worst_segment = max(segment_wapes)
        if worst_segment > max_segment_wape:
            checks['segment_check'] = False
            issues.append(f"Worst segment WAPE ({worst_segment:.1%}) exceeds threshold ({max_segment_wape:.1%})")

    all_passed = all(checks.values())

    return {
        'deployment_ready': all_passed,
        'checks': checks,
        'issues': issues,
        'overall_wape': overall_wape,
        'target_wape': target_wape,
    }


# =============================================================================
# 10. VISUALIZATION
# =============================================================================

def setup_matplotlib():
    """Setup matplotlib for non-interactive use."""
    import matplotlib
    matplotlib.use('Agg')


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = 'Actual vs Predicted',
    figsize: Tuple[int, int] = (10, 8),
    sample_size: int = 5000,
) -> str:
    """
    Create scatter plot of actual vs predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    output_path : str
        Path to save plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    sample_size : int
        Maximum points to plot (for performance)

    Returns
    -------
    str
        Path to saved plot
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Sample if too many points
    if len(y_true) > sample_size:
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.3, s=10)

    # Add diagonal line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()

    # Add metrics annotation
    wape = compute_wape(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    ax.annotate(f'WAPE: {wape:.2%}\nMAE: {mae:.2f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = 'Error Distribution',
    figsize: Tuple[int, int] = (12, 5),
) -> str:
    """
    Plot distribution of forecast errors.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    output_path : str
        Path to save plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved plot
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Error distribution (can be negative)
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', label='Zero error')
    axes[0].axvline(x=np.mean(errors), color='green', linestyle='-', label=f'Mean: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Error (Predicted - Actual)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()

    # Absolute error distribution
    axes[1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(x=np.mean(abs_errors), color='red', linestyle='-', label=f'MAE: {np.mean(abs_errors):.2f}')
    axes[1].axvline(x=np.median(abs_errors), color='green', linestyle='--', label=f'Median: {np.median(abs_errors):.2f}')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Error Distribution')
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_error_over_time(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    output_path: str,
    title: str = 'Forecast Error Over Time',
    figsize: Tuple[int, int] = (14, 6),
) -> str:
    """
    Plot forecast error over time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    date_col : str
        Date column
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    output_path : str
        Path to save plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved plot
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    # Compute metrics by period
    period_metrics = compute_metrics_by_group(df, date_col, actual_col, pred_col)
    period_metrics = period_metrics.sort_values(date_col)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(range(len(period_metrics)), period_metrics['wape'], 'b-o', label='WAPE')

    # Add threshold lines
    ax.axhline(y=WAPE_GOOD, color='green', linestyle='--', alpha=0.7, label=f'Good ({WAPE_GOOD:.0%})')
    ax.axhline(y=WAPE_ACCEPTABLE, color='orange', linestyle='--', alpha=0.7, label=f'Acceptable ({WAPE_ACCEPTABLE:.0%})')
    ax.axhline(y=WAPE_POOR, color='red', linestyle='--', alpha=0.7, label=f'Poor ({WAPE_POOR:.0%})')

    ax.set_xlabel('Time Period')
    ax.set_ylabel('WAPE')
    ax.set_title(title)
    ax.legend()

    # Set x-tick labels
    n_ticks = min(10, len(period_metrics))
    tick_positions = np.linspace(0, len(period_metrics) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(period_metrics[date_col].iloc[i])[:10] for i in tick_positions], rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_segment_comparison(
    segment_metrics: pd.DataFrame,
    segment_col: str,
    output_path: str,
    title: str = 'Segment Performance Comparison',
    figsize: Tuple[int, int] = (12, 6),
) -> str:
    """
    Plot segment comparison bar chart.

    Parameters
    ----------
    segment_metrics : pd.DataFrame
        Segment metrics DataFrame
    segment_col : str
        Segment column name
    output_path : str
        Path to save plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved plot
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    segment_metrics = segment_metrics.sort_values('wape')

    fig, ax = plt.subplots(figsize=figsize)

    colors = []
    for w in segment_metrics['wape']:
        if w <= WAPE_GOOD:
            colors.append('green')
        elif w <= WAPE_ACCEPTABLE:
            colors.append('orange')
        else:
            colors.append('red')

    bars = ax.barh(segment_metrics[segment_col].astype(str), segment_metrics['wape'], color=colors, alpha=0.8)

    # Add value labels
    for bar, wape in zip(bars, segment_metrics['wape']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{wape:.1%}', va='center', fontsize=9)

    ax.set_xlabel('WAPE')
    ax.set_ylabel('Segment')
    ax.set_title(title)

    # Add threshold lines
    ax.axvline(x=WAPE_GOOD, color='green', linestyle='--', alpha=0.5, label=f'Good')
    ax.axvline(x=WAPE_ACCEPTABLE, color='orange', linestyle='--', alpha=0.5, label=f'Acceptable')

    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_wape_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    actual_col: str,
    pred_col: str,
    output_path: str,
    title: str = 'WAPE Heatmap',
    figsize: Tuple[int, int] = (12, 8),
) -> str:
    """
    Create heatmap of WAPE by two dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    row_col : str
        Column for heatmap rows
    col_col : str
        Column for heatmap columns
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    output_path : str
        Path to save plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved plot
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Compute WAPE by combination
    metrics = compute_metrics_by_group(df, [row_col, col_col], actual_col, pred_col)

    # Pivot to matrix
    pivot = metrics.pivot(index=row_col, columns=col_col, values='wape')

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'WAPE'})

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# 11. REPORTING
# =============================================================================

def create_diagnostic_report(
    pipeline_result: DiagnosticPipelineResult,
    output_path: str,
) -> str:
    """
    Create comprehensive diagnostic report (JSON).

    Parameters
    ----------
    pipeline_result : DiagnosticPipelineResult
        Results from run_diagnostic_pipeline
    output_path : str
        Path to save report

    Returns
    -------
    str
        Path to saved report
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'overall_metrics': pipeline_result.overall_metrics.to_dict(),
        'error_decomposition': {
            'total_error': pipeline_result.error_decomposition.total_error,
            'bias_component': pipeline_result.error_decomposition.bias_component,
            'variance_component': pipeline_result.error_decomposition.variance_component,
            'bias_pct': pipeline_result.error_decomposition.bias_pct,
            'variance_pct': pipeline_result.error_decomposition.variance_pct,
        },
        'model_verdict': {
            'overall_wape': pipeline_result.model_verdict.overall_wape,
            'quality_tier': pipeline_result.model_verdict.quality_tier,
            'deployment_ready': pipeline_result.model_verdict.deployment_ready,
            'confidence_score': pipeline_result.model_verdict.confidence_score,
            'strengths': pipeline_result.model_verdict.strengths,
            'weaknesses': pipeline_result.model_verdict.weaknesses,
            'recommendations': pipeline_result.model_verdict.recommendations,
        },
        'temporal_analysis': pipeline_result.temporal_analysis,
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return output_path


def create_executive_summary(
    overall_metrics: MetricsResult,
    verdict: ModelVerdict,
    n_segments: int,
    n_keys: int,
) -> str:
    """
    Create executive summary text.

    Parameters
    ----------
    overall_metrics : MetricsResult
        Overall metrics
    verdict : ModelVerdict
        Model verdict
    n_segments : int
        Number of segments
    n_keys : int
        Number of keys

    Returns
    -------
    str
        Executive summary text
    """
    summary = f"""
FORECAST MODEL DIAGNOSTIC SUMMARY
=================================

OVERALL PERFORMANCE
- WAPE: {overall_metrics.wape:.1%}
- Quality Tier: {overall_metrics.quality_tier}
- MAE: {overall_metrics.mae:.2f}
- Bias: {overall_metrics.bias:+.2f}

COVERAGE
- Segments: {n_segments}
- Keys: {n_keys:,}
- Observations: {overall_metrics.n_obs:,}

DEPLOYMENT READINESS
- Ready: {'Yes' if verdict.deployment_ready else 'No'}
- Confidence Score: {verdict.confidence_score:.0f}/100

STRENGTHS
{chr(10).join(f'- {s}' for s in verdict.strengths) if verdict.strengths else '- None identified'}

AREAS FOR IMPROVEMENT
{chr(10).join(f'- {w}' for w in verdict.weaknesses) if verdict.weaknesses else '- None identified'}

RECOMMENDATIONS
{chr(10).join(f'- {r}' for r in verdict.recommendations) if verdict.recommendations else '- None'}
"""
    return summary.strip()


# =============================================================================
# 12. HIGH-LEVEL PIPELINE
# =============================================================================

def run_diagnostic_pipeline(
    forecast_df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    key_col: str,
    date_col: str,
    output_dir: str,
    segment_col: Optional[str] = None,
    characteristic_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    n_top_bottom: int = 10,
    create_visualizations: bool = True,
) -> DiagnosticPipelineResult:
    """
    Run complete diagnostic pipeline in one call.

    This is the MAIN function that does EVERYTHING:
    1. Computes metrics at all levels (overall, segment, key)
    2. Identifies top/bottom performers
    3. Decomposes errors (bias vs variance)
    4. Analyzes temporal patterns
    5. Generates model verdict
    6. Creates visualizations
    7. Saves all outputs

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame with actuals and predictions
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    key_col : str
        Key column
    date_col : str
        Date column
    output_dir : str
        Output directory for all files
    segment_col : str, optional
        Segment column
    characteristic_cols : List[str], optional
        Columns with key characteristics
    categorical_cols : List[str], optional
        Categorical columns to analyze
    n_top_bottom : int
        Number of top/bottom performers
    create_visualizations : bool
        Whether to create charts

    Returns
    -------
    DiagnosticPipelineResult
        Complete diagnostic results

    Example
    -------
    >>> result = run_diagnostic_pipeline(
    ...     forecast_df=df,
    ...     actual_col='actual',
    ...     pred_col='predicted',
    ...     key_col='key',
    ...     date_col='year_week',
    ...     output_dir='diagnostic_output',
    ...     segment_col='segment_id',
    ... )
    >>> print(f"WAPE: {result.overall_metrics.wape:.2%}")
    >>> print(f"Deployment ready: {result.model_verdict.deployment_ready}")
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting diagnostic pipeline...")

    # 1. Overall metrics
    overall_metrics = compute_all_metrics(forecast_df[actual_col], forecast_df[pred_col])
    logger.info(f"Overall WAPE: {overall_metrics.wape:.2%}")

    # 2. Segment metrics
    if segment_col and segment_col in forecast_df.columns:
        segment_metrics = compute_metrics_by_group(forecast_df, segment_col, actual_col, pred_col)
    else:
        segment_metrics = pd.DataFrame()

    # 3. Key metrics
    key_metrics = compute_metrics_by_group(forecast_df, key_col, actual_col, pred_col)

    # 4. Top/Bottom performers
    top_performers, bottom_performers = identify_top_bottom_performers(
        key_metrics, key_col, actual_col='sum_actual', pred_col='sum_predicted',
        n=n_top_bottom
    )
    # Note: We need to recompute since the function signature changed
    top_performers, bottom_performers = identify_top_bottom_performers(
        forecast_df, key_col, actual_col, pred_col, n=n_top_bottom
    )

    # 5. Error decomposition
    error_decomposition = decompose_forecast_error(forecast_df[actual_col], forecast_df[pred_col])

    # 6. Temporal analysis
    temporal_metrics = analyze_error_over_time(forecast_df, date_col, actual_col, pred_col)
    temporal_analysis = {
        'n_periods': len(temporal_metrics),
        'wape_trend': temporal_metrics['wape'].tolist() if len(temporal_metrics) > 0 else [],
        'best_period_wape': float(temporal_metrics['wape'].min()) if len(temporal_metrics) > 0 else None,
        'worst_period_wape': float(temporal_metrics['wape'].max()) if len(temporal_metrics) > 0 else None,
    }

    # 7. Model verdict
    model_verdict = generate_model_verdict(
        overall_metrics, segment_metrics, error_decomposition
    )

    # 8. Save outputs
    key_metrics.to_csv(os.path.join(output_dir, 'key_diagnostics.csv'), index=False)

    if len(segment_metrics) > 0:
        segment_metrics.to_csv(os.path.join(output_dir, 'segment_diagnostics.csv'), index=False)

    temporal_metrics.to_csv(os.path.join(output_dir, 'temporal_diagnostics.csv'), index=False)

    top_performers.to_csv(os.path.join(output_dir, 'top_performers.csv'), index=False)
    bottom_performers.to_csv(os.path.join(output_dir, 'bottom_performers.csv'), index=False)

    # Save summary JSON
    summary = {
        'overall_metrics': overall_metrics.to_dict(),
        'model_verdict': {
            'quality_tier': model_verdict.quality_tier,
            'deployment_ready': model_verdict.deployment_ready,
            'confidence_score': model_verdict.confidence_score,
            'strengths': model_verdict.strengths,
            'weaknesses': model_verdict.weaknesses,
            'recommendations': model_verdict.recommendations,
        },
        'error_decomposition': {
            'bias_pct': error_decomposition.bias_pct,
            'variance_pct': error_decomposition.variance_pct,
        },
    }

    with open(os.path.join(output_dir, 'diagnostics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # 9. Create visualizations
    if create_visualizations:
        try:
            plot_actual_vs_predicted(
                forecast_df[actual_col].values,
                forecast_df[pred_col].values,
                os.path.join(output_dir, 'actual_vs_predicted.png'),
            )

            plot_error_distribution(
                forecast_df[actual_col].values,
                forecast_df[pred_col].values,
                os.path.join(output_dir, 'error_distribution.png'),
            )

            plot_error_over_time(
                forecast_df, date_col, actual_col, pred_col,
                os.path.join(output_dir, 'error_over_time.png'),
            )

            if len(segment_metrics) > 0:
                plot_segment_comparison(
                    segment_metrics, segment_col,
                    os.path.join(output_dir, 'segment_comparison.png'),
                )

            logger.info("Visualizations created successfully")
        except Exception as e:
            logger.warning(f"Could not create some visualizations: {e}")

    # Build result (simplified PerformerAnalysis for now)
    top_analyses = []
    bottom_analyses = []

    logger.info(f"Diagnostic pipeline complete: WAPE={overall_metrics.wape:.2%}")

    return DiagnosticPipelineResult(
        overall_metrics=overall_metrics,
        segment_metrics=segment_metrics,
        key_metrics=key_metrics,
        top_performers=top_analyses,
        bottom_performers=bottom_analyses,
        error_decomposition=error_decomposition,
        temporal_analysis=temporal_analysis,
        model_verdict=model_verdict,
        output_dir=output_dir,
    )
