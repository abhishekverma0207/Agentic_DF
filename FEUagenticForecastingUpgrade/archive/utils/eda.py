"""
State-of-the-Art EDA Utilities Module
=====================================
Comprehensive Exploratory Data Analysis functions for demand forecasting.

This module provides WORLD-CLASS EDA capabilities that agents can call
with minimal code. Each function is designed to work on ANY dataset.

Categories:
1. Data Quality Analysis
2. Distribution Analysis
3. Outlier Detection
4. Time Series Decomposition
5. Stationarity & Trend Analysis
6. Autocorrelation Analysis
7. Seasonality Analysis
8. Changepoint Detection
9. Feature Importance Analysis
10. Predictability Metrics
11. Cross-Series Analysis
12. One-Call EDA Pipeline

Usage:
    from utils.eda import (
        run_eda_pipeline,  # ONE-CALL complete EDA
        analyze_data_quality,
        detect_outliers,
        decompose_time_series,
        compute_acf_pacf,
        detect_changepoints,
        compute_feature_importance_ensemble,
        compute_predictability_score,
        generate_eda_report,
    )
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Suppress ALL warnings for cleaner output
warnings.filterwarnings('ignore')
# Suppress specific library warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*p-value.*')
warnings.filterwarnings('ignore', message='.*interpolation.*')
warnings.filterwarnings('ignore', message='.*KPSS.*')
warnings.filterwarnings('ignore', message='.*stationary.*')

# Suppress statsmodels warnings at module level
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

logger = logging.getLogger(__name__)
logging.getLogger('statsmodels').setLevel(logging.ERROR)


# =============================================================================
# 1. DATA QUALITY ANALYSIS
# =============================================================================

def analyze_data_quality(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive data quality analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    key_columns : List[str], optional
        Key identifier columns
    target_col : str, optional
        Target column for specific analysis
    date_col : str, optional
        Date column for temporal analysis

    Returns
    -------
    Dict[str, Any]
        Comprehensive data quality report

    Example
    -------
    >>> quality = analyze_data_quality(df, ['key'], 'sales', 'date')
    >>> print(f"Quality Score: {quality['overall_score']}/100")
    """
    report = {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'column_types': {},
        'missing_analysis': {},
        'duplicate_analysis': {},
        'value_range_analysis': {},
        'quality_flags': [],
        'overall_score': 100,
    }

    # Column type analysis
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        report['column_types'][col] = {
            'dtype': dtype,
            'n_unique': n_unique,
            'unique_ratio': round(n_unique / len(df), 4) if len(df) > 0 else 0,
        }

    # Missing data analysis
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    cols_with_missing = missing_counts[missing_counts > 0]

    report['missing_analysis'] = {
        'total_missing_cells': int(missing_counts.sum()),
        'total_cells': len(df) * len(df.columns),
        'missing_percentage': round(missing_counts.sum() / (len(df) * len(df.columns)) * 100, 2),
        'columns_with_missing': {col: {'count': int(missing_counts[col]), 'pct': float(missing_pct[col])}
                                  for col in cols_with_missing.index},
    }

    if report['missing_analysis']['missing_percentage'] > 10:
        report['quality_flags'].append(f"High missing data: {report['missing_analysis']['missing_percentage']:.1f}%")
        report['overall_score'] -= 15
    elif report['missing_analysis']['missing_percentage'] > 5:
        report['quality_flags'].append(f"Moderate missing data: {report['missing_analysis']['missing_percentage']:.1f}%")
        report['overall_score'] -= 5

    # Duplicate analysis
    n_duplicates = df.duplicated().sum()
    report['duplicate_analysis'] = {
        'duplicate_rows': int(n_duplicates),
        'duplicate_percentage': round(n_duplicates / len(df) * 100, 2) if len(df) > 0 else 0,
    }

    if key_columns:
        key_duplicates = df.duplicated(subset=key_columns, keep=False).sum()
        report['duplicate_analysis']['key_duplicates'] = int(key_duplicates)

    if report['duplicate_analysis']['duplicate_percentage'] > 5:
        report['quality_flags'].append(f"High duplicates: {report['duplicate_analysis']['duplicate_percentage']:.1f}%")
        report['overall_score'] -= 10

    # Target column analysis
    if target_col and target_col in df.columns:
        target_series = df[target_col]
        report['target_analysis'] = {
            'min': float(target_series.min()),
            'max': float(target_series.max()),
            'mean': float(target_series.mean()),
            'median': float(target_series.median()),
            'std': float(target_series.std()),
            'zero_count': int((target_series == 0).sum()),
            'zero_percentage': round((target_series == 0).mean() * 100, 2),
            'negative_count': int((target_series < 0).sum()),
            'has_negatives': bool((target_series < 0).any()),
        }

        if report['target_analysis']['has_negatives']:
            report['quality_flags'].append("Target contains negative values")
            report['overall_score'] -= 10

    # Date column analysis
    if date_col and date_col in df.columns:
        date_series = df[date_col]
        report['temporal_analysis'] = {
            'min_date': str(date_series.min()),
            'max_date': str(date_series.max()),
            'n_unique_dates': int(date_series.nunique()),
        }

    # Ensure score is within bounds
    report['overall_score'] = max(0, min(100, report['overall_score']))

    return report


def analyze_missing_patterns(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze missing data patterns (MCAR, MAR, MNAR).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, optional
        Target column for correlation analysis

    Returns
    -------
    Dict[str, Any]
        Missing pattern analysis

    Example
    -------
    >>> patterns = analyze_missing_patterns(df, 'sales')
    >>> print(patterns['likely_mechanism'])
    """
    missing_matrix = df.isnull()
    missing_counts = missing_matrix.sum()
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()

    if not cols_with_missing:
        return {'has_missing': False, 'likely_mechanism': 'N/A'}

    result = {
        'has_missing': True,
        'columns_with_missing': cols_with_missing,
        'missing_correlations': {},
        'pattern_analysis': {},
    }

    # Check if missingness in one column correlates with missingness in another
    if len(cols_with_missing) > 1:
        missing_corr = missing_matrix[cols_with_missing].corr()
        result['missing_correlations'] = missing_corr.to_dict()

    # Check if missingness correlates with other variables (MAR indicator)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for missing_col in cols_with_missing[:5]:  # Limit to first 5 columns
        col_analysis = {}
        missing_indicator = missing_matrix[missing_col].astype(int)

        for num_col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            if num_col != missing_col and missing_indicator.sum() > 0:
                try:
                    # Point-biserial correlation
                    non_missing_mask = ~df[num_col].isnull()
                    if non_missing_mask.sum() > 10:
                        corr, pval = stats.pointbiserialr(
                            missing_indicator[non_missing_mask],
                            df.loc[non_missing_mask, num_col]
                        )
                        if abs(corr) > 0.1:
                            col_analysis[num_col] = {'correlation': round(corr, 3), 'p_value': round(pval, 4)}
                except Exception:
                    pass

        result['pattern_analysis'][missing_col] = col_analysis

    # Determine likely mechanism
    has_correlated_missing = any(
        len(v) > 0 for v in result['pattern_analysis'].values()
    )

    if has_correlated_missing:
        result['likely_mechanism'] = 'MAR (Missing At Random - correlated with other variables)'
    else:
        result['likely_mechanism'] = 'MCAR (Missing Completely At Random) or MNAR'

    return result


def recommend_imputation_strategy(
    df: pd.DataFrame,
    col: str,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recommend imputation strategy for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to analyze
    target_col : str, optional
        Target column for context

    Returns
    -------
    Dict[str, Any]
        Imputation recommendations

    Example
    -------
    >>> strategy = recommend_imputation_strategy(df, 'price')
    >>> print(strategy['recommended_method'])
    """
    series = df[col]
    missing_pct = series.isnull().mean() * 100

    if missing_pct == 0:
        return {'column': col, 'has_missing': False, 'recommended_method': 'N/A'}

    recommendations = {
        'column': col,
        'has_missing': True,
        'missing_percentage': round(missing_pct, 2),
        'dtype': str(series.dtype),
    }

    # Numeric column strategies
    if np.issubdtype(series.dtype, np.number):
        skewness = series.skew() if series.notna().sum() > 3 else 0

        if missing_pct < 5:
            if abs(skewness) < 0.5:
                recommendations['recommended_method'] = 'mean'
                recommendations['reasoning'] = 'Low missing %, symmetric distribution'
            else:
                recommendations['recommended_method'] = 'median'
                recommendations['reasoning'] = 'Low missing %, skewed distribution'
        elif missing_pct < 20:
            recommendations['recommended_method'] = 'interpolate'
            recommendations['reasoning'] = 'Moderate missing %, time series context likely'
        else:
            recommendations['recommended_method'] = 'multiple_imputation'
            recommendations['reasoning'] = 'High missing %, complex imputation needed'

        recommendations['fill_value'] = round(series.median(), 4)

    # Categorical column strategies
    else:
        mode_val = series.mode().iloc[0] if len(series.mode()) > 0 else None

        if missing_pct < 10:
            recommendations['recommended_method'] = 'mode'
            recommendations['reasoning'] = 'Low missing %, use most frequent value'
        else:
            recommendations['recommended_method'] = 'new_category'
            recommendations['reasoning'] = 'High missing %, create "Unknown" category'

        recommendations['fill_value'] = str(mode_val) if mode_val is not None else 'Unknown'

    return recommendations


# =============================================================================
# 2. DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_distribution(
    series: pd.Series,
    name: str = 'Variable',
) -> Dict[str, Any]:
    """
    Comprehensive distribution analysis.

    Parameters
    ----------
    series : pd.Series
        Numeric series to analyze
    name : str
        Variable name for reporting

    Returns
    -------
    Dict[str, Any]
        Distribution analysis results

    Example
    -------
    >>> dist = analyze_distribution(df['sales'], 'Sales')
    >>> print(f"Distribution: {dist['best_fit_distribution']}")
    """
    series = series.dropna()

    if len(series) < 10:
        return {'name': name, 'error': 'Insufficient data (< 10 observations)'}

    result = {
        'name': name,
        'n_observations': len(series),
        'basic_stats': {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'range': float(series.max() - series.min()),
        },
        'shape_stats': {
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'cv': float(series.std() / (series.mean() + 1e-10)),
        },
        'percentiles': {
            'p5': float(series.quantile(0.05)),
            'p25': float(series.quantile(0.25)),
            'p50': float(series.quantile(0.50)),
            'p75': float(series.quantile(0.75)),
            'p95': float(series.quantile(0.95)),
            'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
        },
    }

    # Normality tests
    if len(series) >= 20:
        try:
            # Shapiro-Wilk test (best for n < 5000)
            if len(series) < 5000:
                stat, p_value = stats.shapiro(series.sample(min(5000, len(series))))
                result['normality_tests'] = {
                    'shapiro_wilk': {'statistic': float(round(stat, 4)), 'p_value': float(round(p_value, 4))},
                }

            # D'Agostino-Pearson test
            stat, p_value = stats.normaltest(series)
            if 'normality_tests' not in result:
                result['normality_tests'] = {}
            result['normality_tests']['dagostino_pearson'] = {
                'statistic': float(round(stat, 4)),
                'p_value': float(round(p_value, 4)),
            }

            # Interpretation
            is_normal = float(result['normality_tests']['dagostino_pearson']['p_value']) > 0.05
            result['is_approximately_normal'] = bool(is_normal)
        except Exception:
            result['normality_tests'] = {'error': 'Could not perform normality tests'}
            result['is_approximately_normal'] = False

    # Distribution shape interpretation
    skew = result['shape_stats']['skewness']
    kurt = result['shape_stats']['kurtosis']

    if abs(skew) < 0.5:
        skew_interp = 'symmetric'
    elif skew > 0:
        skew_interp = 'right-skewed (positive)'
    else:
        skew_interp = 'left-skewed (negative)'

    if abs(kurt) < 1:
        kurt_interp = 'mesokurtic (normal-like tails)'
    elif kurt > 0:
        kurt_interp = 'leptokurtic (heavy tails)'
    else:
        kurt_interp = 'platykurtic (light tails)'

    result['interpretation'] = {
        'skewness': skew_interp,
        'kurtosis': kurt_interp,
    }

    # Best-fit distribution (simplified)
    if result.get('is_approximately_normal', False):
        result['best_fit_distribution'] = 'normal'
    elif (series >= 0).all() and skew > 1:
        result['best_fit_distribution'] = 'lognormal_or_exponential'
    elif (series >= 0).all():
        result['best_fit_distribution'] = 'gamma_or_weibull'
    else:
        result['best_fit_distribution'] = 'unknown'

    return result


def compute_distribution_statistics_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
) -> pd.DataFrame:
    """
    Compute distribution statistics grouped by specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Columns to group by
    value_col : str
        Column to analyze

    Returns
    -------
    pd.DataFrame
        Distribution statistics by group

    Example
    -------
    >>> stats_df = compute_distribution_statistics_by_group(df, ['segment'], 'sales')
    """
    agg_funcs = {
        'count': 'count',
        'mean': 'mean',
        'std': 'std',
        'min': 'min',
        'p25': lambda x: x.quantile(0.25),
        'median': 'median',
        'p75': lambda x: x.quantile(0.75),
        'max': 'max',
        'skew': 'skew',
    }

    result = df.groupby(group_cols)[value_col].agg(
        count='count',
        mean='mean',
        std='std',
        min='min',
        p25=lambda x: x.quantile(0.25),
        median='median',
        p75=lambda x: x.quantile(0.75),
        max='max',
    ).reset_index()

    result['cv'] = result['std'] / (result['mean'] + 1e-10)
    result['iqr'] = result['p75'] - result['p25']

    return result


# =============================================================================
# 3. OUTLIER DETECTION
# =============================================================================

def detect_outliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Detect outliers using multiple methods.

    Parameters
    ----------
    series : pd.Series
        Series to analyze
    method : str
        Detection method: 'iqr', 'zscore', 'mad', 'isolation_forest'
    threshold : float
        Threshold for outlier detection (method-specific)
        - IQR: multiplier (default 1.5, use 3.0 for extreme)
        - Z-score: number of std devs (default 3.0)
        - MAD: multiplier (default 3.0)

    Returns
    -------
    Dict[str, Any]
        Outlier detection results

    Example
    -------
    >>> outliers = detect_outliers(df['sales'], method='iqr')
    >>> print(f"Outliers: {outliers['n_outliers']} ({outliers['outlier_percentage']:.1f}%)")
    """
    series = series.dropna()

    if len(series) < 10:
        return {'error': 'Insufficient data', 'n_outliers': 0}

    if method == 'iqr':
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series))
        outlier_mask = z_scores > threshold
        lower_bound = series.mean() - threshold * series.std()
        upper_bound = series.mean() + threshold * series.std()

    elif method == 'mad':
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / (mad + 1e-10)
        outlier_mask = np.abs(modified_z) > threshold
        lower_bound = median - threshold * mad / 0.6745
        upper_bound = median + threshold * mad / 0.6745

    elif method == 'isolation_forest':
        try:
            from sklearn.ensemble import IsolationForest
            # Cap n_jobs at 4 to leave headroom for LightGBM's
            # num_threads=cpu_count(). On a 16-core driver, n_jobs=-1
            # would oversubscribe the moment any LGBM fit overlaps.
            clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=4)
            preds = clf.fit_predict(series.values.reshape(-1, 1))
            outlier_mask = pd.Series(preds == -1, index=series.index)
            lower_bound = None
            upper_bound = None
        except ImportError:
            return detect_outliers(series, method='iqr', threshold=threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    n_outliers = outlier_mask.sum()

    result = {
        'method': method,
        'threshold': threshold,
        'n_outliers': int(n_outliers),
        'outlier_percentage': round(n_outliers / len(series) * 100, 2),
        'outlier_indices': series.index[outlier_mask].tolist()[:100],  # Limit to 100
        'bounds': {
            'lower': float(lower_bound) if lower_bound is not None else None,
            'upper': float(upper_bound) if upper_bound is not None else None,
        },
    }

    if n_outliers > 0:
        outlier_values = series[outlier_mask]
        result['outlier_stats'] = {
            'min': float(outlier_values.min()),
            'max': float(outlier_values.max()),
            'mean': float(outlier_values.mean()),
        }

    return result


def detect_outliers_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    method: str = 'iqr',
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Detect outliers within each group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Columns to group by
    value_col : str
        Column to analyze
    method : str
        Detection method
    threshold : float
        Detection threshold

    Returns
    -------
    pd.DataFrame
        DataFrame with 'is_outlier' column added

    Example
    -------
    >>> df = detect_outliers_by_group(df, ['key'], 'sales')
    >>> outlier_count = df['is_outlier'].sum()
    """
    df = df.copy()
    df['is_outlier'] = False

    def mark_outliers(group):
        series = group[value_col]
        if len(series) < 10:
            return pd.Series(False, index=group.index)

        if method == 'iqr':
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            return (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            return z_scores > threshold
        else:
            return pd.Series(False, index=group.index)

    df['is_outlier'] = df.groupby(group_cols, group_keys=False).apply(
        lambda g: mark_outliers(g)
    )

    return df


def detect_multivariate_outliers(
    df: pd.DataFrame,
    feature_cols: List[str],
    contamination: float = 0.1,
) -> Dict[str, Any]:
    """
    Detect multivariate outliers using Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Columns to use for outlier detection
    contamination : float
        Expected proportion of outliers

    Returns
    -------
    Dict[str, Any]
        Multivariate outlier detection results

    Example
    -------
    >>> outliers = detect_multivariate_outliers(df, ['cv', 'mean', 'zero_fraction'])
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # Suppress warnings during sklearn operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Prepare data
        X = df[feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Isolation Forest. n_jobs=4 (not -1) prevents oversubscription
        # against LightGBM's full-cores threading later in the pipeline.
        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=4,
        )
        predictions = clf.fit_predict(X_scaled)
        scores = clf.decision_function(X_scaled)

        outlier_mask = predictions == -1

    return {
        'n_outliers': int(outlier_mask.sum()),
        'outlier_percentage': round(outlier_mask.sum() / len(df) * 100, 2),
        'outlier_indices': df.index[outlier_mask].tolist(),
        'anomaly_scores': scores.tolist(),
        'feature_cols_used': feature_cols,
    }


# =============================================================================
# 4. TIME SERIES DECOMPOSITION
# =============================================================================

def decompose_time_series(
    series: pd.Series,
    period: int = 52,
    model: str = 'additive',
    method: str = 'stl',
) -> Dict[str, Any]:
    """
    Decompose time series into trend, seasonal, and residual components.

    Parameters
    ----------
    series : pd.Series
        Time series to decompose
    period : int
        Seasonal period (52 for weekly, 12 for monthly)
    model : str
        Decomposition model: 'additive' or 'multiplicative'
    method : str
        Decomposition method: 'stl' (robust) or 'classical'

    Returns
    -------
    Dict[str, Any]
        Decomposition results with trend, seasonal, residual

    Example
    -------
    >>> decomp = decompose_time_series(df['sales'], period=52)
    >>> print(f"Trend strength: {decomp['metrics']['trend_strength']:.2f}")
    """
    series = series.dropna()

    if len(series) < period * 2:
        return {
            'error': f'Insufficient data: need at least {period * 2} observations',
            'n_observations': len(series),
        }

    result = {
        'method': method,
        'model': model,
        'period': period,
        'n_observations': len(series),
    }

    try:
        # Suppress statsmodels warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if method == 'stl':
                from statsmodels.tsa.seasonal import STL
                stl = STL(series, period=period, robust=True)
                decomposition = stl.fit()

                result['components'] = {
                    'trend': decomposition.trend.tolist(),
                    'seasonal': decomposition.seasonal.tolist(),
                    'residual': decomposition.resid.tolist(),
                }

            else:  # classical
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(series, model=model, period=period)

                result['components'] = {
                    'trend': decomposition.trend.dropna().tolist(),
                    'seasonal': decomposition.seasonal.dropna().tolist(),
                    'residual': decomposition.resid.dropna().tolist(),
                }

        # Compute strength metrics
        trend = np.array(result['components']['trend'])
        seasonal = np.array(result['components']['seasonal'])
        residual = np.array(result['components']['residual'])

        # Remove NaN values
        valid_mask = ~(np.isnan(trend) | np.isnan(seasonal) | np.isnan(residual))
        trend = trend[valid_mask]
        seasonal = seasonal[valid_mask]
        residual = residual[valid_mask]

        if len(residual) > 0:
            var_residual = np.var(residual)
            var_trend_residual = np.var(trend + residual) if len(trend) > 0 else var_residual
            var_seasonal_residual = np.var(seasonal + residual) if len(seasonal) > 0 else var_residual

            # Strength metrics (0 to 1)
            trend_strength = max(0, 1 - var_residual / (var_trend_residual + 1e-10))
            seasonal_strength = max(0, 1 - var_residual / (var_seasonal_residual + 1e-10))

            result['metrics'] = {
                'trend_strength': round(float(trend_strength), 4),
                'seasonal_strength': round(float(seasonal_strength), 4),
                'residual_variance': round(float(var_residual), 4),
                'trend_variance': round(float(np.var(trend)), 4),
                'seasonal_variance': round(float(np.var(seasonal)), 4),
            }

    except Exception as e:
        result['error'] = str(e)

    return result


def compute_trend_strength(series: pd.Series, window: int = 52) -> float:
    """
    Compute trend strength using variance decomposition.

    Parameters
    ----------
    series : pd.Series
        Time series
    window : int
        Moving average window

    Returns
    -------
    float
        Trend strength (0 to 1)

    Example
    -------
    >>> strength = compute_trend_strength(df['sales'])
    >>> print(f"Trend strength: {strength:.2f}")
    """
    series = series.dropna()

    if len(series) < window * 2:
        return 0.0

    # Compute trend using moving average
    trend = series.rolling(window=window, center=True).mean()
    detrended = series - trend

    # Remove NaN values
    valid_mask = ~(trend.isna() | detrended.isna())
    detrended_valid = detrended[valid_mask]
    series_valid = series[valid_mask]

    if len(detrended_valid) < 10:
        return 0.0

    # Trend strength
    var_detrended = detrended_valid.var()
    var_original = series_valid.var()

    strength = max(0, 1 - var_detrended / (var_original + 1e-10))

    return round(float(strength), 4)


def compute_seasonal_strength(
    series: pd.Series,
    period: int = 52,
) -> float:
    """
    Compute seasonal strength using variance decomposition.

    Parameters
    ----------
    series : pd.Series
        Time series
    period : int
        Seasonal period

    Returns
    -------
    float
        Seasonal strength (0 to 1)

    Example
    -------
    >>> strength = compute_seasonal_strength(df['sales'], period=52)
    """
    series = series.dropna()

    if len(series) < period * 2:
        return 0.0

    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(series, period=period, robust=True)
        decomposition = stl.fit()

        seasonal = decomposition.seasonal
        residual = decomposition.resid

        valid_mask = ~(seasonal.isna() | residual.isna())
        seasonal_valid = seasonal[valid_mask]
        residual_valid = residual[valid_mask]

        if len(residual_valid) < 10:
            return 0.0

        var_residual = residual_valid.var()
        var_seasonal_residual = (seasonal_valid + residual_valid).var()

        strength = max(0, 1 - var_residual / (var_seasonal_residual + 1e-10))

        return round(float(strength), 4)

    except Exception:
        return 0.0


# =============================================================================
# 5. STATIONARITY & TREND ANALYSIS
# =============================================================================

def comprehensive_stationarity_test(
    series: pd.Series,
    significance: float = 0.05,
) -> Dict[str, Any]:
    """
    Comprehensive stationarity testing with multiple tests.

    Parameters
    ----------
    series : pd.Series
        Time series to test
    significance : float
        Significance level

    Returns
    -------
    Dict[str, Any]
        Stationarity test results with interpretation

    Example
    -------
    >>> result = comprehensive_stationarity_test(df['sales'])
    >>> print(f"Status: {result['final_verdict']}")
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    series = series.dropna()

    if len(series) < 20:
        return {
            'error': 'Insufficient data',
            'final_verdict': 'UNKNOWN',
        }

    result = {
        'n_observations': len(series),
        'significance_level': significance,
        'tests': {},
    }

    # ADF Test (null: unit root / non-stationary)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_result = adfuller(series, autolag='AIC')
        result['tests']['adf'] = {
            'statistic': round(adf_result[0], 4),
            'p_value': round(adf_result[1], 4),
            'lags_used': adf_result[2],
            'critical_values': {k: round(v, 4) for k, v in adf_result[4].items()},
            'rejects_null': adf_result[1] < significance,
            'interpretation': 'Stationary' if adf_result[1] < significance else 'Non-stationary (unit root)',
        }
    except Exception as e:
        result['tests']['adf'] = {'error': str(e)}

    # KPSS Test (null: stationary) - SUPPRESS WARNINGS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(series, regression='c', nlags='auto')
        result['tests']['kpss'] = {
            'statistic': round(kpss_result[0], 4),
            'p_value': round(kpss_result[1], 4),
            'lags_used': kpss_result[2],
            'critical_values': {k: round(v, 4) for k, v in kpss_result[3].items()},
            'rejects_null': kpss_result[1] < significance,
            'interpretation': 'Non-stationary' if kpss_result[1] < significance else 'Stationary',
        }
    except Exception as e:
        result['tests']['kpss'] = {'error': str(e)}

    # Combined interpretation
    adf_stationary = result['tests'].get('adf', {}).get('rejects_null', False)
    kpss_stationary = not result['tests'].get('kpss', {}).get('rejects_null', True)

    if adf_stationary and kpss_stationary:
        result['final_verdict'] = 'STATIONARY'
        result['recommendation'] = 'No transformation needed'
    elif not adf_stationary and not kpss_stationary:
        result['final_verdict'] = 'NON_STATIONARY'
        result['recommendation'] = 'Apply differencing (d=1)'
    elif adf_stationary and not kpss_stationary:
        result['final_verdict'] = 'TREND_STATIONARY'
        result['recommendation'] = 'Detrend the series'
    else:
        result['final_verdict'] = 'DIFFERENCE_STATIONARY'
        result['recommendation'] = 'Apply differencing (d=1)'

    return result


def test_stationarity_batch(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Test stationarity for multiple time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Key columns
    target_col : str
        Target column
    significance : float
        Significance level

    Returns
    -------
    pd.DataFrame
        Stationarity results per series

    Example
    -------
    >>> results = test_stationarity_batch(df, ['key'], 'sales')
    >>> print(f"Stationary: {(results['verdict'] == 'STATIONARY').mean():.1%}")
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    results = []

    # Suppress ALL warnings during batch processing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for keys, group in df.groupby(group_cols):
            series = group[target_col].dropna()

            if len(series) < 20:
                row = {**dict(zip(group_cols, [keys] if not isinstance(keys, tuple) else keys)),
                       'n_obs': len(series), 'verdict': 'INSUFFICIENT_DATA'}
                results.append(row)
                continue

            row = dict(zip(group_cols, [keys] if not isinstance(keys, tuple) else keys))
            row['n_obs'] = len(series)

            try:
                adf_p = adfuller(series, autolag='AIC')[1]
                kpss_p = kpss(series, regression='c', nlags='auto')[1]

                row['adf_pvalue'] = round(adf_p, 4)
                row['kpss_pvalue'] = round(kpss_p, 4)

                adf_stationary = adf_p < significance
                kpss_stationary = kpss_p > significance

                if adf_stationary and kpss_stationary:
                    row['verdict'] = 'STATIONARY'
                elif not adf_stationary and not kpss_stationary:
                    row['verdict'] = 'NON_STATIONARY'
                elif adf_stationary:
                    row['verdict'] = 'TREND_STATIONARY'
                else:
                    row['verdict'] = 'DIFFERENCE_STATIONARY'

            except Exception:
                row['verdict'] = 'ERROR'

            results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# 6. AUTOCORRELATION ANALYSIS
# =============================================================================

def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 52,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute ACF and PACF with significance bounds.

    Parameters
    ----------
    series : pd.Series
        Time series
    nlags : int
        Number of lags
    alpha : float
        Significance level for confidence bounds

    Returns
    -------
    Dict[str, Any]
        ACF and PACF results

    Example
    -------
    >>> acf_pacf = compute_acf_pacf(df['sales'], nlags=52)
    >>> print(f"Significant ACF lags: {acf_pacf['significant_acf_lags']}")
    """
    from statsmodels.tsa.stattools import acf, pacf

    series = series.dropna()

    if len(series) < nlags + 10:
        nlags = max(10, len(series) - 10)

    if len(series) < 20:
        return {'error': 'Insufficient data', 'n_observations': len(series)}

    result = {
        'n_observations': len(series),
        'nlags': nlags,
    }

    try:
        # Suppress statsmodels warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ACF
            acf_values, acf_confint = acf(series, nlags=nlags, alpha=alpha)

            result['acf'] = {
                'values': [round(v, 4) for v in acf_values],
                'confidence_lower': [round(c[0], 4) for c in acf_confint],
                'confidence_upper': [round(c[1], 4) for c in acf_confint],
            }

            # Identify significant lags (outside confidence bounds)
            significant_acf = [
                i for i in range(1, len(acf_values))
                if acf_values[i] < acf_confint[i][0] or acf_values[i] > acf_confint[i][1]
            ]
            result['significant_acf_lags'] = significant_acf[:20]  # Limit to top 20

            # PACF
            pacf_values, pacf_confint = pacf(series, nlags=nlags, alpha=alpha)

            result['pacf'] = {
                'values': [round(v, 4) for v in pacf_values],
                'confidence_lower': [round(c[0], 4) for c in pacf_confint],
                'confidence_upper': [round(c[1], 4) for c in pacf_confint],
            }

            significant_pacf = [
                i for i in range(1, len(pacf_values))
                if pacf_values[i] < pacf_confint[i][0] or pacf_values[i] > pacf_confint[i][1]
            ]
            result['significant_pacf_lags'] = significant_pacf[:20]

            # Key metrics
            result['lag1_acf'] = round(float(acf_values[1]), 4) if len(acf_values) > 1 else 0.0
            result['lag1_pacf'] = round(float(pacf_values[1]), 4) if len(pacf_values) > 1 else 0.0

            # Model suggestions based on ACF/PACF patterns
            if len(significant_pacf) <= 3 and len(significant_acf) > 5:
                result['suggested_model'] = f'AR({max(significant_pacf) if significant_pacf else 1})'
            elif len(significant_acf) <= 3 and len(significant_pacf) > 5:
                result['suggested_model'] = f'MA({max(significant_acf) if significant_acf else 1})'
            else:
                result['suggested_model'] = 'ARMA or ARIMA'

    except Exception as e:
        result['error'] = str(e)

    return result


def compute_autocorrelation_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    lags: List[int] = [1, 4, 13, 26, 52],
) -> pd.DataFrame:
    """
    Compute autocorrelation summary for multiple series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Key columns
    target_col : str
        Target column
    lags : List[int]
        Lags to compute

    Returns
    -------
    pd.DataFrame
        Autocorrelation summary per series

    Example
    -------
    >>> acf_summary = compute_autocorrelation_summary(df, ['key'], 'sales')
    """
    results = []

    for keys, group in df.groupby(group_cols):
        series = group[target_col].dropna()

        row = dict(zip(group_cols, [keys] if not isinstance(keys, tuple) else keys))
        row['n_obs'] = len(series)

        if len(series) < max(lags) + 5:
            for lag in lags:
                row[f'acf_lag{lag}'] = None
            results.append(row)
            continue

        for lag in lags:
            if len(series) > lag:
                acf_val = series.autocorr(lag=lag)
                row[f'acf_lag{lag}'] = round(float(acf_val), 4) if not pd.isna(acf_val) else None
            else:
                row[f'acf_lag{lag}'] = None

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# 7. SEASONALITY ANALYSIS
# =============================================================================

def detect_seasonality(
    series: pd.Series,
    max_period: int = 104,
    min_seasonal_strength: float = 0.1,
    min_power_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    Detect seasonal patterns using spectral analysis with significance thresholds.

    This function now uses STRENGTH-BASED detection, not just presence detection.
    A series is considered seasonal only if:
    1. The dominant period's power exceeds min_power_ratio of total spectral power
    2. The computed seasonal_strength exceeds min_seasonal_strength

    Parameters
    ----------
    series : pd.Series
        Time series
    max_period : int
        Maximum period to consider
    min_seasonal_strength : float
        Minimum seasonal strength (0-1) to consider seasonality significant.
        Default 0.1 (10% of variance explained by seasonal component).
    min_power_ratio : float
        Minimum ratio of dominant period's power to total power.
        Default 0.05 (5% of spectral power concentrated at dominant frequency).

    Returns
    -------
    Dict[str, Any]
        Seasonality detection results including:
        - has_seasonality: bool (True only if SIGNIFICANT seasonality detected)
        - seasonal_strength: float (0-1, strength of seasonal pattern)
        - dominant_period: int or None
        - detected_periods: list of top periods with power

    Example
    -------
    >>> seasonality = detect_seasonality(df['sales'])
    >>> print(f"Dominant period: {seasonality['dominant_period']}")
    >>> print(f"Seasonal strength: {seasonality['seasonal_strength']:.2f}")
    """
    series = series.dropna()

    if len(series) < max_period:
        max_period = len(series) // 2

    if len(series) < 20:
        return {'error': 'Insufficient data', 'has_seasonality': False, 'seasonal_strength': 0.0}

    result = {
        'n_observations': len(series),
        'max_period_tested': max_period,
        'seasonal_strength': 0.0,
    }

    try:
        # FFT-based periodicity detection
        centered_series = series.values - series.mean()
        fft_vals = np.fft.fft(centered_series)
        power_spectrum = np.abs(fft_vals[:len(series)//2])**2

        # Skip frequency 0 (DC component)
        power_spectrum[0] = 0

        # Total spectral power (excluding DC)
        total_power = power_spectrum.sum()

        # Find dominant frequencies
        freq_indices = np.argsort(power_spectrum)[::-1][:5]

        # Convert to periods and compute power ratios
        periods = []
        for idx in freq_indices:
            if idx > 0:
                period = len(series) / idx
                if 2 <= period <= max_period:
                    power_ratio = float(power_spectrum[idx]) / (total_power + 1e-10)
                    periods.append({
                        'period': round(period, 1),
                        'power': round(float(power_spectrum[idx]), 2),
                        'power_ratio': round(power_ratio, 4),
                    })

        result['detected_periods'] = periods[:3]  # Top 3

        # Compute seasonal strength using autocorrelation at dominant period
        # This is more robust than just FFT power
        seasonal_strength = 0.0
        dominant_period = None
        dominant_power_ratio = 0.0

        if periods:
            dominant_period = int(round(periods[0]['period']))
            dominant_power_ratio = periods[0]['power_ratio']

            # Compute seasonal strength as autocorrelation at dominant period
            if dominant_period < len(series):
                acf_at_period = series.autocorr(lag=dominant_period)
                if not pd.isna(acf_at_period):
                    # Seasonal strength: positive autocorrelation at seasonal lag
                    seasonal_strength = max(0, float(acf_at_period))

            # Also consider the power ratio as part of strength
            # Combine autocorrelation and power ratio for robust measurement
            seasonal_strength = (seasonal_strength + dominant_power_ratio) / 2

        result['dominant_period'] = dominant_period
        result['seasonal_strength'] = round(seasonal_strength, 4)
        result['dominant_power_ratio'] = round(dominant_power_ratio, 4)

        # SIGNIFICANT seasonality: both power ratio AND strength must exceed thresholds
        result['has_seasonality'] = (
            seasonal_strength >= min_seasonal_strength and
            dominant_power_ratio >= min_power_ratio
        )

        # Common period detection (for reference)
        common_periods = [7, 12, 13, 26, 52]
        result['common_period_scores'] = {}

        for p in common_periods:
            if p < len(series) // 2:
                # Check autocorrelation at this lag
                acf_val = series.autocorr(lag=p) if len(series) > p else 0
                result['common_period_scores'][p] = round(float(acf_val), 4) if not pd.isna(acf_val) else 0

    except Exception as e:
        result['error'] = str(e)
        result['has_seasonality'] = False
        result['seasonal_strength'] = 0.0

    return result


def compute_seasonal_indices(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    period_col: str = 'week_of_year',
) -> pd.DataFrame:
    """
    Compute seasonal indices (average value by period).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Date column
    target_col : str
        Target column
    period_col : str
        Period column (e.g., 'week_of_year', 'month')

    Returns
    -------
    pd.DataFrame
        Seasonal indices

    Example
    -------
    >>> indices = compute_seasonal_indices(df, 'date', 'sales', 'week_of_year')
    """
    if period_col not in df.columns:
        # Try to extract from date
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            if period_col == 'week_of_year':
                df[period_col] = df[date_col].dt.isocalendar().week
            elif period_col == 'month':
                df[period_col] = df[date_col].dt.month

    overall_mean = df[target_col].mean()

    seasonal = df.groupby(period_col)[target_col].agg(['mean', 'std', 'count']).reset_index()
    seasonal['seasonal_index'] = seasonal['mean'] / (overall_mean + 1e-10)
    seasonal['relative_std'] = seasonal['std'] / (seasonal['mean'] + 1e-10)

    return seasonal


# =============================================================================
# 8. CHANGEPOINT DETECTION
# =============================================================================

def detect_changepoints(
    series: pd.Series,
    model: str = 'rbf',
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
    min_segment_length: int = 10,
    min_mean_shift_ratio: float = 0.2,
) -> Dict[str, Any]:
    """
    Detect structural changepoints using PELT algorithm with significance filtering.

    This function now uses CONSERVATIVE detection to avoid false positives:
    1. Higher default penalty (BIC-based) to reduce over-segmentation
    2. Minimum segment length to avoid detecting noise as changepoints
    3. Post-filtering to remove changepoints without significant mean shifts

    Parameters
    ----------
    series : pd.Series
        Time series
    model : str
        Cost model: 'l1', 'l2', 'rbf' (default)
    penalty : float, optional
        Penalty value for PELT. If None, uses BIC-based penalty:
        penalty = 3 * log(n) * variance (more conservative than default)
    n_bkps : int, optional
        Number of breakpoints to detect (overrides penalty)
    min_segment_length : int
        Minimum observations between changepoints (default 10)
    min_mean_shift_ratio : float
        Minimum ratio of mean shift to series std to consider significant.
        Default 0.2 means the mean must shift by at least 20% of std.

    Returns
    -------
    Dict[str, Any]
        Changepoint detection results including:
        - n_changepoints: int (only SIGNIFICANT changepoints)
        - significant_changepoints: bool (True if meaningful regime changes)
        - avg_mean_shift_ratio: float (average magnitude of mean shifts)

    Example
    -------
    >>> cp = detect_changepoints(df['sales'])
    >>> print(f"Significant changepoints: {cp['n_changepoints']}")
    """
    try:
        import ruptures as rpt
    except ImportError:
        return {'error': 'ruptures library not installed', 'n_changepoints': 0, 'significant_changepoints': False}

    series = series.dropna()

    if len(series) < 20:
        return {'error': 'Insufficient data', 'n_changepoints': 0, 'significant_changepoints': False}

    signal = series.values.reshape(-1, 1)
    series_std = series.std()
    series_mean = series.mean()

    result = {
        'n_observations': len(series),
        'model': model,
        'significant_changepoints': False,
    }

    try:
        # Use PELT algorithm
        algo = rpt.Pelt(model=model, min_size=min_segment_length).fit(signal)

        if n_bkps is not None:
            # Use Binseg if exact number needed
            algo = rpt.Binseg(model=model, min_size=min_segment_length).fit(signal)
            breakpoints = algo.predict(n_bkps=n_bkps)
        else:
            # Auto-detect with CONSERVATIVE BIC-based penalty
            # Using 3x multiplier to reduce false positives (standard BIC uses 2x)
            if penalty is None:
                penalty = 3 * np.log(len(signal)) * np.var(signal)
            breakpoints = algo.predict(pen=penalty)

        # Remove the last point (always equals length)
        breakpoints = [bp for bp in breakpoints if bp < len(series)]

        # Compute segment statistics BEFORE filtering
        segments = []
        prev_bp = 0
        for bp in breakpoints + [len(series)]:
            segment = series.iloc[prev_bp:bp]
            if len(segment) > 0:
                segments.append({
                    'start': prev_bp,
                    'end': bp,
                    'length': len(segment),
                    'mean': round(float(segment.mean()), 4),
                    'std': round(float(segment.std()), 4),
                })
            prev_bp = bp

        # POST-FILTER: Keep only changepoints with significant mean shifts
        # A changepoint is significant if the mean shift between adjacent segments
        # is at least min_mean_shift_ratio * series_std
        significant_breakpoints = []
        mean_shift_ratios = []

        for i, bp in enumerate(breakpoints):
            if i + 1 < len(segments):
                mean_before = segments[i]['mean']
                mean_after = segments[i + 1]['mean']
                mean_shift = abs(mean_after - mean_before)
                shift_ratio = mean_shift / (series_std + 1e-10)
                mean_shift_ratios.append(shift_ratio)

                if shift_ratio >= min_mean_shift_ratio:
                    significant_breakpoints.append(bp)

        result['n_changepoints_raw'] = len(breakpoints)
        result['n_changepoints'] = len(significant_breakpoints)
        result['changepoint_indices'] = significant_breakpoints
        result['avg_mean_shift_ratio'] = round(float(np.mean(mean_shift_ratios)), 4) if mean_shift_ratios else 0.0

        # Significant changepoints: at least one changepoint with meaningful mean shift
        result['significant_changepoints'] = len(significant_breakpoints) > 0

        # Get dates if index is datetime
        if hasattr(series.index, 'tolist'):
            result['changepoint_positions'] = [series.index[bp] for bp in significant_breakpoints if bp < len(series.index)]

        result['segments'] = segments

    except Exception as e:
        result['error'] = str(e)
        result['n_changepoints'] = 0
        result['significant_changepoints'] = False

    return result


def detect_level_shifts(
    series: pd.Series,
    window: int = 13,
    threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Detect level shifts using rolling statistics.

    Parameters
    ----------
    series : pd.Series
        Time series
    window : int
        Rolling window size
    threshold : float
        Z-score threshold for shift detection

    Returns
    -------
    Dict[str, Any]
        Level shift detection results

    Example
    -------
    >>> shifts = detect_level_shifts(df['sales'], window=13)
    >>> print(f"Level shifts detected: {shifts['n_shifts']}")
    """
    series = series.dropna()

    if len(series) < window * 3:
        return {'error': 'Insufficient data', 'n_shifts': 0}

    # Compute rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Compute difference in means
    mean_diff = rolling_mean.diff(window)

    # Z-score of mean difference
    z_scores = mean_diff / (rolling_std + 1e-10)

    # Detect shifts
    shift_mask = np.abs(z_scores) > threshold
    shift_indices = series.index[shift_mask].tolist()

    result = {
        'n_shifts': len(shift_indices),
        'shift_indices': shift_indices[:20],  # Limit
        'threshold': threshold,
        'window': window,
    }

    if shift_indices:
        result['shift_magnitudes'] = [
            round(float(mean_diff.loc[idx]), 4)
            for idx in shift_indices[:20]
        ]

    return result


# =============================================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def compute_feature_importance_ensemble(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute ensemble feature importance using multiple methods.

    Methods:
    - Random Forest importance
    - Pearson/Spearman correlation (robust to NaN/constant columns)
    - Mutual information
    - Target encoding correlation (for categoricals)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Target column
    feature_cols : List[str]
        Numeric feature columns
    categorical_cols : List[str], optional
        Categorical columns for separate analysis

    Returns
    -------
    pd.DataFrame
        Feature importance rankings with ensemble_score combining all methods

    Example
    -------
    >>> importance = compute_feature_importance_ensemble(df, 'sales', numeric_cols)
    >>> top_features = importance.head(10)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression

    results = []

    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return pd.DataFrame()

    # Helper to safely get numeric value (handle NaN/inf)
    def safe_val(v, default=0.0):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return default
        return float(v)

    # Suppress warnings during all sklearn operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Prepare data - more robust handling
        X = df[feature_cols].copy()
        y = df[target_col].fillna(0).values

        # Replace inf with NaN, then fill NaN with 0
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Identify constant columns (these will have 0 correlation and 0 MI)
        constant_cols = [c for c in feature_cols if X[c].nunique() <= 1]
        variable_cols = [c for c in feature_cols if c not in constant_cols]

        # Random Forest importance (works on all columns including constants).
        # n_jobs=4 (not -1) so we don't fight LightGBM for cores.
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=4)
        rf.fit(X, y)
        rf_importance = dict(zip(feature_cols, rf.feature_importances_))

        # Correlation - compute robustly, handling constant/near-constant columns
        correlations = {}
        for col in feature_cols:
            try:
                # Check if column has variance
                col_std = df[col].std()
                if col_std is None or np.isnan(col_std) or col_std < 1e-10:
                    correlations[col] = 0.0
                else:
                    corr_val = df[col].corr(df[target_col])
                    correlations[col] = safe_val(abs(corr_val), 0.0)
            except Exception:
                correlations[col] = 0.0

        # Mutual Information (only on variable columns)
        mi_importance = {col: 0.0 for col in feature_cols}  # default 0
        if variable_cols:
            try:
                X_var = X[variable_cols].values
                mi_scores = mutual_info_regression(X_var, y, random_state=42)
                for i, col in enumerate(variable_cols):
                    mi_importance[col] = safe_val(mi_scores[i], 0.0)
            except Exception:
                pass  # Keep defaults of 0

        # Normalize each method (robust to NaN)
        def normalize(d):
            # Filter out NaN/inf values before finding max
            valid_vals = [safe_val(v, 0.0) for v in d.values()]
            max_val = max(valid_vals) if valid_vals else 1.0
            if max_val < 1e-10:
                max_val = 1.0  # Avoid division by zero
            return {k: safe_val(v, 0.0) / max_val for k, v in d.items()}

        rf_norm = normalize(rf_importance)
        corr_norm = normalize(correlations)
        mi_norm = normalize(mi_importance)

        # Build results for numeric features
        for col in feature_cols:
            rf_val = safe_val(rf_importance.get(col, 0), 0.0)
            corr_val = safe_val(correlations.get(col, 0), 0.0)
            mi_val = safe_val(mi_importance.get(col, 0), 0.0)
            rf_n = safe_val(rf_norm.get(col, 0), 0.0)
            corr_n = safe_val(corr_norm.get(col, 0), 0.0)
            mi_n = safe_val(mi_norm.get(col, 0), 0.0)

            # Ensemble score: average of the three normalized scores
            ensemble = (rf_n + corr_n + mi_n) / 3.0

            results.append({
                'feature': col,
                'type': 'numeric',
                'rf_importance': round(rf_val, 6),
                'correlation': round(corr_val, 6),
                'mutual_info': round(mi_val, 6),
                'rf_normalized': round(rf_n, 4),
                'corr_normalized': round(corr_n, 4),
                'mi_normalized': round(mi_n, 4),
                'ensemble_score': round(ensemble, 4),
            })

        # Categorical features - use target encoding correlation + categorical MI
        if categorical_cols:
            categorical_cols = [c for c in categorical_cols if c in df.columns]

            # Prepare encoded versions for MI
            cat_encoded = {}
            for col in categorical_cols:
                try:
                    target_means = df.groupby(col)[target_col].mean()
                    cat_encoded[col] = df[col].map(target_means).fillna(0).values
                except Exception:
                    cat_encoded[col] = np.zeros(len(df))

            # Compute MI for categorical features (on encoded values)
            cat_mi = {}
            if categorical_cols:
                try:
                    cat_X = np.column_stack([cat_encoded[c] for c in categorical_cols])
                    cat_mi_scores = mutual_info_regression(cat_X, y, random_state=42)
                    for i, col in enumerate(categorical_cols):
                        cat_mi[col] = safe_val(cat_mi_scores[i], 0.0)
                except Exception:
                    for col in categorical_cols:
                        cat_mi[col] = 0.0

            # Normalize categorical MI
            cat_mi_max = max(cat_mi.values()) if cat_mi.values() else 1.0
            if cat_mi_max < 1e-10:
                cat_mi_max = 1.0

            for col in categorical_cols:
                # Target encoding correlation
                try:
                    te_corr = np.corrcoef(cat_encoded[col], y)[0, 1]
                    te_corr = safe_val(abs(te_corr), 0.0)
                except Exception:
                    te_corr = 0.0

                mi_val = safe_val(cat_mi.get(col, 0), 0.0)
                mi_norm_val = mi_val / cat_mi_max

                # Lift for binary
                lift = None
                try:
                    if df[col].nunique() == 2:
                        values = df[col].dropna().unique()
                        if len(values) == 2:
                            mean_0 = df[df[col] == values[0]][target_col].mean()
                            mean_1 = df[df[col] == values[1]][target_col].mean()
                            if mean_0 and mean_0 > 0:
                                lift = mean_1 / mean_0
                except Exception:
                    lift = None

                # Ensemble for categorical: weighted combo of correlation and MI
                # (RF importance is 0 for categoricals since we don't include them in RF)
                ensemble = (te_corr + mi_norm_val) / 2.0

                results.append({
                    'feature': col,
                    'type': 'categorical',
                    'rf_importance': 0.0,
                    'correlation': round(te_corr, 6),
                    'mutual_info': round(mi_val, 6),
                    'rf_normalized': 0.0,
                    'corr_normalized': round(te_corr, 4),
                    'mi_normalized': round(mi_norm_val, 4),
                    'ensemble_score': round(ensemble, 4),
                    'lift_ratio': round(lift, 4) if lift and not np.isnan(lift) else None,
                })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('ensemble_score', ascending=False).reset_index(drop=True)

    return result_df


def compute_feature_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    method: str = 'pearson',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute feature correlation matrix and detect multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Feature columns
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Correlation matrix and multicollinearity report

    Example
    -------
    >>> corr_matrix, report = compute_feature_correlations(df, feature_cols)
    >>> print(f"High correlation pairs: {report['high_correlation_pairs']}")
    """
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return pd.DataFrame(), {'error': 'No valid feature columns'}

    corr_matrix = df[feature_cols].corr(method=method)

    # Detect high correlations
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'feature1': feature_cols[i],
                    'feature2': feature_cols[j],
                    'correlation': round(corr_val, 4),
                })

    report = {
        'n_features': len(feature_cols),
        'method': method,
        'high_correlation_pairs': high_corr_pairs,
        'n_high_correlations': len(high_corr_pairs),
        'has_multicollinearity': len(high_corr_pairs) > 0,
    }

    if high_corr_pairs:
        report['recommended_drops'] = [p['feature2'] for p in high_corr_pairs][:5]

    return corr_matrix, report


# =============================================================================
# 10. PREDICTABILITY METRICS
# =============================================================================

def compute_predictability_score(
    series: pd.Series,
    method: str = 'composite',
) -> Dict[str, Any]:
    """
    Compute predictability score for a time series.

    Parameters
    ----------
    series : pd.Series
        Time series
    method : str
        Scoring method: 'composite', 'entropy', 'acf'

    Returns
    -------
    Dict[str, Any]
        Predictability metrics

    Example
    -------
    >>> pred = compute_predictability_score(df['sales'])
    >>> print(f"Predictability: {pred['score']:.2f}")
    """
    series = series.dropna()

    if len(series) < 20:
        return {'error': 'Insufficient data', 'score': 0.5}

    result = {
        'n_observations': len(series),
        'method': method,
    }

    # ACF contribution (higher lag-1 autocorrelation = more predictable)
    acf_lag1 = series.autocorr(lag=1)
    acf_contrib = min(1, max(0, (abs(acf_lag1) + 1) / 2)) if not pd.isna(acf_lag1) else 0.5
    result['acf_lag1'] = round(float(acf_lag1) if not pd.isna(acf_lag1) else 0, 4)
    result['acf_contribution'] = round(acf_contrib, 4)

    # CV contribution (lower CV = more predictable)
    cv = series.std() / (series.mean() + 1e-10)
    cv_contrib = 1 / (1 + cv)
    result['cv'] = round(float(cv), 4)
    result['cv_contribution'] = round(float(cv_contrib), 4)

    # Zero fraction contribution (less zeros = more predictable, typically)
    zero_frac = (series == 0).mean()
    zero_contrib = 1 - zero_frac
    result['zero_fraction'] = round(float(zero_frac), 4)
    result['zero_contribution'] = round(float(zero_contrib), 4)

    # Composite score
    if method == 'composite':
        score = (acf_contrib * 0.4 + cv_contrib * 0.4 + zero_contrib * 0.2)
    elif method == 'acf':
        score = acf_contrib
    else:
        score = (acf_contrib + cv_contrib) / 2

    result['score'] = round(float(score), 4)

    # Interpretation
    if score > 0.7:
        result['interpretation'] = 'High predictability'
    elif score > 0.4:
        result['interpretation'] = 'Moderate predictability'
    else:
        result['interpretation'] = 'Low predictability (consider simpler models)'

    return result


def compute_predictability_batch(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Compute predictability scores for multiple series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Key columns
    target_col : str
        Target column

    Returns
    -------
    pd.DataFrame
        Predictability scores per series

    Example
    -------
    >>> pred_df = compute_predictability_batch(df, ['key'], 'sales')
    """
    results = []

    for keys, group in df.groupby(group_cols):
        series = group[target_col].dropna()

        row = dict(zip(group_cols, [keys] if not isinstance(keys, tuple) else keys))
        row['n_obs'] = len(series)

        if len(series) < 20:
            row['predictability_score'] = None
            results.append(row)
            continue

        pred_result = compute_predictability_score(series)
        row['predictability_score'] = pred_result.get('score')
        row['acf_lag1'] = pred_result.get('acf_lag1')
        row['cv'] = pred_result.get('cv')

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# 11. CROSS-SERIES ANALYSIS
# =============================================================================

def compute_cross_series_correlation(
    df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Compute correlation between different time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    key_col : str
        Key column identifying series
    date_col : str
        Date column
    target_col : str
        Target column
    top_n : int
        Number of series to analyze

    Returns
    -------
    Dict[str, Any]
        Cross-series correlation analysis

    Example
    -------
    >>> cross_corr = compute_cross_series_correlation(df, 'key', 'date', 'sales')
    """
    # Pivot to wide format
    keys = df[key_col].value_counts().head(top_n).index.tolist()
    subset = df[df[key_col].isin(keys)]

    pivot = subset.pivot_table(
        index=date_col,
        columns=key_col,
        values=target_col,
        aggfunc='sum'
    ).fillna(0)

    if pivot.shape[1] < 2:
        return {'error': 'Insufficient series for cross-correlation'}

    # Compute correlation matrix
    corr_matrix = pivot.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'series1': corr_matrix.columns[i],
                    'series2': corr_matrix.columns[j],
                    'correlation': round(corr_val, 4),
                })

    # Cluster similar series
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    try:
        dist_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(dist_matrix, 0)
        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method='average')
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')

        cluster_map = {}
        for key, cluster in zip(corr_matrix.columns, clusters):
            cluster_map.setdefault(int(cluster), []).append(key)

    except Exception:
        cluster_map = {}

    return {
        'n_series_analyzed': len(keys),
        'correlation_matrix': corr_matrix.to_dict(),
        'high_correlation_pairs': sorted(high_corr_pairs, key=lambda x: -abs(x['correlation'])),
        'n_highly_correlated': len(high_corr_pairs),
        'series_clusters': cluster_map,
    }


def test_granger_causality(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_cols: Optional[List[str]] = None,
    max_lag: int = 4,
) -> pd.DataFrame:
    """
    Test Granger causality from features to target.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Target column
    feature_cols : List[str]
        Feature columns to test
    group_cols : List[str], optional
        If provided, aggregate first
    max_lag : int
        Maximum lag for Granger test

    Returns
    -------
    pd.DataFrame
        Granger causality test results

    Example
    -------
    >>> granger = test_granger_causality(df, 'sales', ['price', 'promo'])
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    feature_cols = [c for c in feature_cols if c in df.columns]

    if group_cols:
        df = df.groupby(group_cols)[[target_col] + feature_cols].sum().reset_index()

    results = []

    # Suppress warnings during Granger tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for col in feature_cols:
            try:
                # Prepare data
                test_data = df[[target_col, col]].dropna()

                if len(test_data) < max_lag * 2 + 10:
                    results.append({
                        'feature': col,
                        'error': 'Insufficient data',
                    })
                    continue

                # Run Granger test
                test_results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                # Extract p-values for each lag
                p_values = {
                    lag: round(test_results[lag][0]['ssr_ftest'][1], 4)
                    for lag in range(1, max_lag + 1)
                }

                min_p = min(p_values.values())
                best_lag = min(p_values.keys(), key=lambda k: p_values[k])

                results.append({
                    'feature': col,
                    'p_values_by_lag': p_values,
                    'min_p_value': min_p,
                    'best_lag': best_lag,
                    'granger_causes': min_p < 0.05,
                })

            except Exception as e:
                results.append({
                    'feature': col,
                    'error': str(e),
                })

    return pd.DataFrame(results)


# =============================================================================
# 12. DETAILED RECOMMENDATION GENERATORS
# =============================================================================

def generate_segmentation_recommendations(
    chars_df: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    summary: Dict[str, Any],
    period: int = 52,
) -> Dict[str, Any]:
    """
    Generate detailed segmentation recommendations based on EDA results.

    This creates actionable guidance for the Segmentation Crew including:
    - Recommended clustering features
    - Optimal number of clusters
    - Segment-specific model recommendations
    - Clustering algorithm suggestions

    Parameters
    ----------
    chars_df : pd.DataFrame
        Demand characteristics DataFrame with ADI, CV², intermittency_class
    stationarity_df : pd.DataFrame
        Stationarity test results
    summary : Dict[str, Any]
        EDA summary statistics
    period : int
        Seasonal period (52 for weekly, 12 for monthly)

    Returns
    -------
    Dict[str, Any]
        Comprehensive segmentation recommendations
    """
    recommendations = {
        'data_summary': {},
        'clustering_features': {},
        'cluster_count_recommendation': {},
        'algorithm_recommendation': {},
        'segment_model_hints': {},
        'preprocessing_recommendations': {},
    }

    # === Data Summary ===
    total_series = len(chars_df)
    lumpy_pct = float((chars_df['intermittency_class'] == 'lumpy').mean())
    intermittent_pct = float((chars_df['intermittency_class'] == 'intermittent').mean())
    erratic_pct = float((chars_df['intermittency_class'] == 'erratic').mean())
    smooth_pct = float((chars_df['intermittency_class'] == 'smooth').mean())

    recommendations['data_summary'] = {
        'total_series': total_series,
        'demand_pattern_distribution': {
            'smooth': round(smooth_pct, 4),
            'erratic': round(erratic_pct, 4),
            'intermittent': round(intermittent_pct, 4),
            'lumpy': round(lumpy_pct, 4),
        },
        'avg_cv': round(float(chars_df['cv'].mean()), 4),
        'avg_adi': round(float(chars_df['adi'].mean()), 4),
        'avg_zero_fraction': round(float(chars_df['zero_fraction'].mean()), 4),
        'cv_std': round(float(chars_df['cv'].std()), 4),
        'heterogeneity_score': round(float(chars_df['cv'].std() / (chars_df['cv'].mean() + 1e-10)), 4),
    }

    # === Clustering Features Recommendation ===
    # Primary features for clustering (always use these)
    primary_features = [
        {'name': 'cv', 'importance': 'high', 'reason': 'Captures demand variability - key for model selection'},
        {'name': 'adi', 'importance': 'high', 'reason': 'Captures intermittency pattern - essential for Syntetos-Boylan'},
        {'name': 'zero_fraction', 'importance': 'high', 'reason': 'Indicates sparsity - drives zero-inflated model need'},
    ]

    # Secondary features (add based on data characteristics)
    secondary_features = []

    if 'predictability_score' in chars_df.columns:
        secondary_features.append({
            'name': 'predictability_score',
            'importance': 'medium',
            'reason': 'Indicates forecastability - helps identify difficult series'
        })

    if 'mean' in chars_df.columns:
        secondary_features.append({
            'name': 'mean',
            'importance': 'medium',
            'reason': 'Volume-based segmentation - high/low volume may need different approaches',
            'transform': 'log1p recommended for skewed distributions'
        })

    # Add stationarity-based features
    if 'verdict' in stationarity_df.columns:
        non_stationary_pct = float((stationarity_df['verdict'] == 'NON_STATIONARY').mean())
        if non_stationary_pct > 0.3:
            secondary_features.append({
                'name': 'is_stationary',
                'importance': 'medium',
                'reason': f'{non_stationary_pct:.1%} series are non-stationary - affects differencing needs',
                'derive_from': "stationarity_df['verdict'] == 'STATIONARY'"
            })

    recommendations['clustering_features'] = {
        'primary_features': primary_features,
        'secondary_features': secondary_features,
        'feature_scaling': 'StandardScaler recommended - features have different scales',
        'missing_value_handling': 'Impute with median before clustering',
    }

    # === Cluster Count Recommendation ===
    # Base recommendation on data size and heterogeneity
    heterogeneity = recommendations['data_summary']['heterogeneity_score']

    if total_series < 100:
        min_clusters, max_clusters, recommended = 2, 4, 3
        reasoning = "Small dataset - fewer clusters to ensure adequate samples per cluster"
    elif total_series < 500:
        min_clusters, max_clusters, recommended = 3, 6, 4
        reasoning = "Medium dataset - moderate cluster count for balance"
    elif total_series < 2000:
        min_clusters, max_clusters, recommended = 4, 8, 5
        reasoning = "Large dataset - more clusters capture heterogeneity"
    else:
        min_clusters, max_clusters, recommended = 5, 10, 6
        reasoning = "Very large dataset - consider more granular segmentation"

    # Adjust based on heterogeneity
    if heterogeneity > 1.5:
        recommended = min(recommended + 1, max_clusters)
        reasoning += f"; High heterogeneity (CV std/mean = {heterogeneity:.2f}) suggests more clusters"
    elif heterogeneity < 0.5:
        recommended = max(recommended - 1, min_clusters)
        reasoning += f"; Low heterogeneity suggests fewer clusters needed"

    # If dominated by one pattern, fewer clusters may suffice
    max_pattern_pct = max(smooth_pct, erratic_pct, intermittent_pct, lumpy_pct)
    if max_pattern_pct > 0.7:
        recommended = max(recommended - 1, min_clusters)
        reasoning += f"; One pattern dominates ({max_pattern_pct:.1%}) - fewer clusters may suffice"

    recommendations['cluster_count_recommendation'] = {
        'min_clusters': min_clusters,
        'max_clusters': max_clusters,
        'recommended_clusters': recommended,
        'reasoning': reasoning,
        'validation_method': 'Use silhouette score and elbow method to validate',
    }

    # === Algorithm Recommendation ===
    if total_series > 5000:
        algo = 'MiniBatchKMeans'
        algo_reason = "Large dataset - MiniBatchKMeans for efficiency"
    elif heterogeneity > 1.0:
        algo = 'GaussianMixture'
        algo_reason = "High heterogeneity - GMM captures non-spherical clusters"
    else:
        algo = 'KMeans'
        algo_reason = "Standard dataset - KMeans is robust and interpretable"

    recommendations['algorithm_recommendation'] = {
        'primary_algorithm': algo,
        'reasoning': algo_reason,
        'alternatives': [
            {'name': 'KMeans', 'when': 'Baseline, interpretable clusters'},
            {'name': 'GaussianMixture', 'when': 'Non-spherical clusters, soft assignments'},
            {'name': 'HDBSCAN', 'when': 'Unknown cluster count, noise detection'},
        ],
        'parameters': {
            'KMeans': {'n_init': 10, 'max_iter': 300, 'random_state': 42},
            'GaussianMixture': {'n_init': 5, 'covariance_type': 'full', 'random_state': 42},
        }
    }

    # === Segment-Specific Model Hints ===
    # Based on Syntetos-Boylan classification
    model_hints = {
        'smooth': {
            'description': 'Regular, predictable demand (low ADI, low CV²)',
            'recommended_models': ['lightgbm', 'xgboost', 'sarima', 'ets', 'prophet'],
            'model_priority': 'Tree-based models excel; statistical models also work well',
            'feature_engineering': {
                'lags': [1, 2, 4, period // 4, period // 2, period],
                'rolling_windows': [4, 8, period // 2, period],
                'use_log_transform': False,
                'differencing': 'optional'
            },
            'expected_accuracy': 'High (WAPE < 30%)',
        },
        'erratic': {
            'description': 'High variability but consistent occurrence (low ADI, high CV²)',
            'recommended_models': ['lightgbm', 'xgboost', 'catboost', 'theta', 'bsts'],
            'model_priority': 'Tree-based models handle variability well; robust statistical methods',
            'feature_engineering': {
                'lags': [1, 2, 4, period // 4, period],
                'rolling_windows': [4, 8, 13, period],
                'use_log_transform': True,
                'differencing': 'recommended if non-stationary'
            },
            'expected_accuracy': 'Medium (WAPE 30-50%)',
        },
        'intermittent': {
            'description': 'Sporadic demand with consistent quantity (high ADI, low CV²)',
            'recommended_models': ['croston', 'sba', 'tsb', 'zero_inflated', 'lightgbm'],
            'model_priority': 'Intermittent specialists first; tree-based with zero features',
            'feature_engineering': {
                'lags': [1, 2, 4, period // 2, period],
                'rolling_windows': [period // 2, period],
                'use_log_transform': False,
                'add_zero_features': True,
                'features_to_add': ['time_since_last_nonzero', 'nonzero_rate', 'demand_occurrence_flag']
            },
            'expected_accuracy': 'Medium-Low (WAPE 40-60%)',
        },
        'lumpy': {
            'description': 'Sporadic AND highly variable demand (high ADI, high CV²)',
            'recommended_models': ['tsb', 'imapa', 'hurdle_model', 'zero_inflated', 'tweedie'],
            'model_priority': 'Two-stage models (occurrence + quantity); specialized lumpy methods',
            'feature_engineering': {
                'lags': [1, 2, 4, period // 2, period],
                'rolling_windows': [period // 2, period],
                'use_log_transform': True,
                'add_zero_features': True,
                'features_to_add': ['time_since_last_nonzero', 'nonzero_rate', 'demand_occurrence_flag', 'avg_nonzero_demand']
            },
            'expected_accuracy': 'Low (WAPE 50-80%)',
        },
    }

    # Add counts per pattern
    for pattern in model_hints:
        count = int((chars_df['intermittency_class'] == pattern).sum())
        model_hints[pattern]['series_count'] = count
        model_hints[pattern]['percentage'] = round(count / total_series, 4) if total_series > 0 else 0

    recommendations['segment_model_hints'] = model_hints

    # === Preprocessing Recommendations ===
    recommendations['preprocessing_recommendations'] = {
        'scaling': {
            'method': 'StandardScaler',
            'apply_to': 'All numeric clustering features',
            'reason': 'Features have different scales (CV: 0-5, ADI: 1-100+, zero_fraction: 0-1)'
        },
        'outlier_handling': {
            'method': 'clip' if chars_df['cv'].quantile(0.99) > 5 else 'none',
            'cv_clip_upper': float(chars_df['cv'].quantile(0.99)),
            'adi_clip_upper': float(chars_df['adi'].quantile(0.99)),
            'reason': 'Extreme outliers can dominate clustering'
        },
        'log_transform_candidates': ['mean', 'std'] if 'mean' in chars_df.columns else [],
    }

    return recommendations


def generate_feature_recommendations(
    chars_df: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    feature_importance_df: Optional[pd.DataFrame],
    summary: Dict[str, Any],
    period: int = 52,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate detailed feature engineering recommendations based on EDA results.

    This creates actionable guidance for the Feature Engineering Crew including:
    - Lag feature recommendations
    - Rolling window features
    - Interaction features
    - Encoding recommendations
    - Feature selection guidance

    Parameters
    ----------
    chars_df : pd.DataFrame
        Demand characteristics DataFrame
    stationarity_df : pd.DataFrame
        Stationarity test results
    feature_importance_df : pd.DataFrame, optional
        Feature importance from EDA
    summary : Dict[str, Any]
        EDA summary statistics
    period : int
        Seasonal period (52 for weekly, 12 for monthly)
    numeric_features : List[str], optional
        List of numeric features
    categorical_features : List[str], optional
        List of categorical features

    Returns
    -------
    Dict[str, Any]
        Comprehensive feature engineering recommendations
    """
    recommendations = {
        'data_characteristics': {},
        'lag_features': {},
        'rolling_features': {},
        'time_features': {},
        'intermittency_features': {},
        'interaction_features': {},
        'encoding_recommendations': {},
        'transformation_recommendations': {},
        'feature_selection': {},
        'top_features': [],
    }

    # === Data Characteristics Summary ===
    avg_cv = float(chars_df['cv'].mean())
    avg_zero_fraction = float(chars_df['zero_fraction'].mean())
    lumpy_intermittent_pct = float((chars_df['intermittency_class'].isin(['lumpy', 'intermittent'])).mean())

    non_stationary_pct = 0.0
    if 'verdict' in stationarity_df.columns:
        non_stationary_pct = float((stationarity_df['verdict'] == 'NON_STATIONARY').mean())

    recommendations['data_characteristics'] = {
        'avg_cv': round(avg_cv, 4),
        'avg_zero_fraction': round(avg_zero_fraction, 4),
        'lumpy_intermittent_pct': round(lumpy_intermittent_pct, 4),
        'non_stationary_pct': round(non_stationary_pct, 4),
        'seasonal_period': period,
        'time_format': 'weekly' if period == 52 else 'monthly' if period == 12 else 'custom',
    }

    # === Lag Feature Recommendations ===
    # Base lags - always include
    if period == 52:  # Weekly data
        base_lags = [1, 2, 4, 8, 13, 26, 52]
        lag_descriptions = {
            1: 'Previous week - captures short-term momentum',
            2: 'Two weeks ago - bi-weekly patterns',
            4: 'One month ago - monthly cycle',
            8: 'Two months ago - bi-monthly patterns',
            13: 'Quarter ago - quarterly seasonality',
            26: 'Half year ago - semi-annual patterns',
            52: 'Same week last year - annual seasonality (CRITICAL)',
        }
    else:  # Monthly or custom
        base_lags = [1, 2, 3, 6, 12]
        lag_descriptions = {
            1: 'Previous period - short-term momentum',
            2: 'Two periods ago - bi-period patterns',
            3: 'Three periods ago - quarterly proxy',
            6: 'Six periods ago - semi-annual',
            12: 'Same period last year - annual seasonality (CRITICAL)',
        }

    # Prioritize lags based on data characteristics
    if lumpy_intermittent_pct > 0.5:
        # For intermittent demand, longer lags matter more
        priority_lags = [l for l in base_lags if l >= period // 4]
        secondary_lags = [l for l in base_lags if l < period // 4]
    else:
        # For smooth demand, recent lags matter more
        priority_lags = [l for l in base_lags if l <= period // 4]
        secondary_lags = [l for l in base_lags if l > period // 4]

    recommendations['lag_features'] = {
        'target_lags': {
            'all_lags': base_lags,
            'priority_lags': priority_lags,
            'secondary_lags': secondary_lags,
            'descriptions': lag_descriptions,
        },
        'feature_lags': {
            'apply_to': numeric_features[:10] if numeric_features else [],  # Top 10 features
            'recommended_lags': [1, period // 4, period] if period >= 12 else [1, 2],
            'reason': 'Leading indicators may predict target with delay'
        },
        'implementation': {
            'naming_convention': '{feature}_lag_{n}',
            'handle_missing': 'Leave as NaN for tree models; impute with 0 for linear models',
        }
    }

    # === Rolling Window Features ===
    if period == 52:
        rolling_windows = [4, 8, 13, 26, 52]
        window_descriptions = {
            4: 'Monthly rolling - smooths weekly noise',
            8: 'Bi-monthly rolling - medium-term trend',
            13: 'Quarterly rolling - seasonal baseline',
            26: 'Semi-annual rolling - longer trend',
            52: 'Annual rolling - year-over-year comparison',
        }
    else:
        rolling_windows = [3, 6, 12]
        window_descriptions = {
            3: 'Quarter rolling - smooths monthly noise',
            6: 'Semi-annual rolling - medium trend',
            12: 'Annual rolling - year baseline',
        }

    rolling_stats = ['mean', 'std', 'min', 'max']
    if avg_zero_fraction > 0.3:
        rolling_stats.append('count_nonzero')  # Important for intermittent demand

    recommendations['rolling_features'] = {
        'windows': rolling_windows,
        'window_descriptions': window_descriptions,
        'statistics': rolling_stats,
        'priority_windows': rolling_windows[:3],  # First 3 are highest priority
        'apply_to_features': numeric_features[:5] if numeric_features else [],
        'implementation': {
            'naming_convention': '{feature}_roll_{window}_{stat}',
            'min_periods': 'Set to window // 2 to handle edge cases',
        }
    }

    # === Time Features ===
    time_features = {
        'cyclical_encoding': {
            'week_of_year': {'sin': True, 'cos': True, 'reason': 'Captures annual seasonality'},
            'month': {'sin': True, 'cos': True, 'reason': 'Captures monthly patterns'},
            'quarter': {'sin': True, 'cos': True, 'reason': 'Captures quarterly patterns'},
        },
        'categorical_time': {
            'is_month_start': 'Beginning of month effects',
            'is_month_end': 'End of month effects (often higher demand)',
            'is_quarter_end': 'Quarter-end effects (business cycles)',
            'is_year_end': 'Year-end effects (holiday, budget flush)',
        },
        'trend_features': {
            'time_index': 'Linear trend capture',
            'time_index_squared': 'Non-linear trend (only if polynomial trend suspected)',
        }
    }

    if non_stationary_pct > 0.5:
        time_features['trend_features']['recommendation'] = 'Include time_index - many series have trend'
    else:
        time_features['trend_features']['recommendation'] = 'Optional - most series are stationary'

    recommendations['time_features'] = time_features

    # === Intermittency Features ===
    if avg_zero_fraction > 0.1 or lumpy_intermittent_pct > 0.2:
        recommendations['intermittency_features'] = {
            'essential': True,
            'features_to_create': [
                {
                    'name': 'time_since_last_nonzero',
                    'description': 'Periods since last positive demand',
                    'importance': 'high',
                    'implementation': 'Group by key, find last nonzero index, compute diff'
                },
                {
                    'name': 'demand_occurrence_rate',
                    'description': 'Rolling proportion of non-zero periods',
                    'importance': 'high',
                    'implementation': 'Rolling mean of (target > 0) indicator'
                },
                {
                    'name': 'avg_nonzero_demand',
                    'description': 'Rolling average of demand when non-zero',
                    'importance': 'medium',
                    'implementation': 'Rolling mean where target > 0, forward fill'
                },
                {
                    'name': 'demand_streak',
                    'description': 'Consecutive periods with/without demand',
                    'importance': 'medium',
                    'implementation': 'Cumsum with reset on sign change'
                },
            ],
            'recommended_windows': [period // 4, period // 2, period],
        }
    else:
        recommendations['intermittency_features'] = {
            'essential': False,
            'reason': f'Low zero fraction ({avg_zero_fraction:.1%}) - intermittency features less critical',
            'optional_features': ['demand_occurrence_rate'],
        }

    # === Interaction Features ===
    interaction_candidates = []

    # Price × Promo interactions (if price and promo features exist)
    price_features = [f for f in (numeric_features or []) if 'price' in f.lower()]
    promo_features = [f for f in (numeric_features or []) if 'promo' in f.lower()]

    if price_features and promo_features:
        interaction_candidates.append({
            'features': [price_features[0], promo_features[0]],
            'type': 'multiplication',
            'name': 'price_promo_interaction',
            'reason': 'Price sensitivity may vary with promotion status'
        })

    # Lag × seasonality interactions
    interaction_candidates.append({
        'features': ['target_lag_1', 'week_of_year_sin'],
        'type': 'multiplication',
        'name': 'momentum_seasonality_interaction',
        'reason': 'Momentum effect may vary by season'
    })

    recommendations['interaction_features'] = {
        'candidates': interaction_candidates,
        'recommendation': 'Create selectively - too many interactions cause overfitting',
        'max_interactions': 5,
    }

    # === Encoding Recommendations ===
    encoding_recs = {}

    for cat_col in (categorical_features or []):
        # Heuristic: high cardinality → target encoding, low → one-hot
        encoding_recs[cat_col] = {
            'recommended_encoding': 'target_encoding',  # Default
            'reason': 'Unknown cardinality - target encoding is safe default',
            'alternatives': ['one_hot', 'label_encoding'],
        }

    recommendations['encoding_recommendations'] = {
        'categorical_features': encoding_recs,
        'general_guidance': {
            'high_cardinality': 'Target encoding or embedding (>20 unique values)',
            'medium_cardinality': 'Target encoding or one-hot (5-20 unique values)',
            'low_cardinality': 'One-hot encoding (<5 unique values)',
            'binary': 'Label encoding (0/1)',
        },
        'target_encoding_params': {
            'smoothing': 10,
            'min_samples_leaf': 5,
            'cv_folds': 5,
        }
    }

    # === Transformation Recommendations ===
    transformations = {}

    # Log transform recommendation
    if avg_cv > 1.5:
        transformations['log_transform'] = {
            'recommended': True,
            'apply_to': 'target and high-variance numeric features',
            'reason': f'High CV ({avg_cv:.2f}) indicates right-skewed distribution',
            'implementation': 'np.log1p(x) to handle zeros'
        }
    else:
        transformations['log_transform'] = {
            'recommended': False,
            'reason': f'CV ({avg_cv:.2f}) is moderate - log transform optional'
        }

    # Differencing recommendation
    if non_stationary_pct > 0.5:
        transformations['differencing'] = {
            'recommended': True,
            'order': 1,
            'reason': f'{non_stationary_pct:.1%} series are non-stationary',
            'apply_to': 'target variable for statistical models'
        }
    else:
        transformations['differencing'] = {
            'recommended': False,
            'reason': f'Only {non_stationary_pct:.1%} series are non-stationary'
        }

    recommendations['transformation_recommendations'] = transformations

    # === Top Features from Importance Analysis ===
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        top_n = min(20, len(feature_importance_df))
        top_features = feature_importance_df.head(top_n)[['feature', 'ensemble_score']].to_dict('records')
        recommendations['top_features'] = top_features
        recommendations['feature_selection'] = {
            'method': 'ensemble_importance',
            'top_features_count': top_n,
            'threshold': float(feature_importance_df['ensemble_score'].quantile(0.5)),
            'recommendation': f'Prioritize top {top_n} features; consider dropping features with score < threshold'
        }
    else:
        recommendations['feature_selection'] = {
            'method': 'not_available',
            'recommendation': 'Run feature importance analysis to guide selection'
        }

    return recommendations


# =============================================================================
# 13. ONE-CALL EDA PIPELINE
# =============================================================================

def run_eda_pipeline(
    df: pd.DataFrame,
    key_columns: List[str],
    date_col: str,
    target_col: str,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    output_dir: str = 'eda_output',
    period: int = 52,
    verbose: bool = False,
    train_end: Optional[str] = None,
    dead_key_threshold: int = 26,
    exhaustive: bool = True,
    price_features_numeric: Optional[List[str]] = None,
    price_features_categorical: Optional[List[str]] = None,
    promo_features_numeric: Optional[List[str]] = None,
    promo_features_categorical: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run COMPREHENSIVE, STATE-OF-THE-ART EDA pipeline with ONE function call.

    This is the MAIN ENTRY POINT for EDA - call this for exhaustive analysis.

    PHILOSOPHY:
    - Run ALL applicable analyses regardless of detected patterns
    - Profile data for context, not for limiting analysis
    - Generate SMART RECOMMENDATIONS that filter/prioritize based on findings
    - All insights flow to downstream crews (segmentation, feature engineering, training)

    DEAD KEY HANDLING:
    This pipeline automatically identifies and excludes "dead keys" - product-customer
    combinations with 26+ consecutive zeros at the end of the training period.
    Dead keys are saved to dead_keys.txt and dead_key_summary.json but excluded
    from all analysis and charts.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    key_columns : List[str]
        Key identifier columns
    date_col : str
        Date column
    target_col : str
        Target column
    numeric_features : List[str], optional
        Numeric feature columns
    categorical_features : List[str], optional
        Categorical feature columns
    output_dir : str
        Output directory for results
    period : int
        Seasonal period (52 for weekly, 12 for monthly)
    verbose : bool
        If True, print progress messages. Default False for agent use.
    train_end : str, optional
        End of training period for dead key detection. If None, uses max date in data.
    dead_key_threshold : int
        Number of consecutive zeros to consider a key "dead". Default 26.
    exhaustive : bool
        If True, run ALL analyses for comprehensive insights. Default True.
    price_features_numeric : List[str], optional
        Numeric price feature columns from config (for segmentation-aware EDA)
    price_features_categorical : List[str], optional
        Categorical price feature columns from config
    promo_features_numeric : List[str], optional
        Numeric promo feature columns from config
    promo_features_categorical : List[str], optional
        Categorical promo feature columns from config

    Returns
    -------
    Dict[str, Any]
        Comprehensive EDA results with paths to saved files and smart recommendations

    Example
    -------
    >>> result = run_eda_pipeline(
    ...     df, ['key'], 'date', 'sales',
    ...     numeric_features=['price', 'promo'],
    ...     output_dir='eda_output',
    ...     train_end='202521',
    ... )
    >>> print(f"EDA complete: {result['summary']['total_series']} series analyzed")
    >>> print(f"Dead keys: {result['dead_key_summary']['dead_keys']}")
    >>> print(f"Recommendations: {result['recommendations']['executive_summary']}")
    """
    import os

    # Suppress all warnings during pipeline execution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'charts'), exist_ok=True)

        total_steps = 14 if exhaustive else 8  # More steps for exhaustive mode

        result = {
            'status': 'success',
            'input_shape': df.shape,
            'key_columns': key_columns,
            'target_col': target_col,
            'output_dir': output_dir,
            'files_created': [],
            'exhaustive_mode': exhaustive,
        }

        # Import utilities
        from utils.agent_utilities import (
            save_csv, save_json,
            compute_demand_characteristics, add_demand_classification,
            setup_matplotlib, create_scatter_plot, create_bar_chart, create_histogram,
            ADI_THRESHOLD, CV2_THRESHOLD,
        )

        # Import dead key handler
        from utils.dead_key_handler import (
            identify_dead_keys, filter_dead_keys_from_df,
            save_dead_key_summary, save_dead_key_list,
        )

        # Import new profiler and recommendations modules
        from utils.data_profiler import profile_data, DataProfile
        from utils.eda_recommendations import generate_all_recommendations, save_recommendations

        try:
            # Create combined key column if multiple key columns
            if len(key_columns) == 1:
                key_col = key_columns[0]
            else:
                df = df.copy()
                df['_combined_key'] = df[key_columns].astype(str).agg('_'.join, axis=1)
                key_col = '_combined_key'

            # =====================================================================
            # STEP 0: DEAD KEY DETECTION - CRITICAL FIRST STEP
            # =====================================================================
            if verbose:
                print(f"0/{total_steps} Detecting dead keys...")

            dead_keys, dead_key_summary = identify_dead_keys(
                df,
                key_col=key_col,
                date_col=date_col,
                target_col=target_col,
                train_end=train_end,
                consecutive_zero_threshold=dead_key_threshold,
            )

            # Save dead key outputs
            save_dead_key_summary(dead_key_summary, os.path.join(output_dir, 'dead_key_summary.json'))
            save_dead_key_list(dead_keys, os.path.join(output_dir, 'dead_keys.txt'))
            result['files_created'].extend(['dead_key_summary.json', 'dead_keys.txt'])
            result['dead_key_summary'] = dead_key_summary.to_dict()

            if verbose:
                print(f"   Dead keys: {dead_key_summary.dead_keys}/{dead_key_summary.total_keys} "
                      f"({dead_key_summary.dead_key_percentage:.1f}%)")

            # Filter out dead keys for analysis
            df_active = filter_dead_keys_from_df(df, dead_keys, key_col=key_col)

            if verbose:
                print(f"   Active data: {len(df_active)} rows ({len(df) - len(df_active)} removed)")

            # Update key_col reference for further processing
            if key_col == '_combined_key':
                # Map back to original key columns for compatibility
                pass  # keep using _combined_key internally

            # Store original df for reference
            df_original = df
            df = df_active  # Use active data for all subsequent analysis

            # =====================================================================
            # STEP 0.5: DATA PROFILING - Comprehensive characterization
            # =====================================================================
            if verbose:
                print(f"0.5/{total_steps} Profiling data characteristics...")

            try:
                data_profile = profile_data(
                    df, key_columns, date_col, target_col,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                )
                # Save profile
                profile_dict = data_profile.to_dict()
                save_json(profile_dict, os.path.join(output_dir, 'data_profile.json'))
                result['files_created'].append('data_profile.json')
                result['data_profile'] = profile_dict

                if verbose:
                    print(f"   Primary pattern: {data_profile.primary_demand_pattern.value}")
                    print(f"   Granularity: {data_profile.time_granularity.value}")
                    print(f"   Pattern mix: Smooth {data_profile.pct_smooth*100:.0f}%, "
                          f"Erratic {data_profile.pct_erratic*100:.0f}%, "
                          f"Intermittent {data_profile.pct_intermittent*100:.0f}%, "
                          f"Lumpy {data_profile.pct_lumpy*100:.0f}%")

                # Update period based on detected granularity if not explicitly set
                if period == 52 and data_profile.detected_period:
                    period = data_profile.detected_period
                    if verbose:
                        print(f"   Auto-detected period: {period}")
            except Exception as e:
                logger.warning(f"Data profiling failed: {e}. Continuing with basic analysis.")
                data_profile = None
                result['data_profile'] = {'error': str(e)}
            # 1. Data Quality Analysis
            if verbose:
                print("1/8 Analyzing data quality...")
            quality = analyze_data_quality(df, key_columns, target_col, date_col)
            save_json(quality, os.path.join(output_dir, 'data_quality.json'))
            result['files_created'].append('data_quality.json')
            result['data_quality_score'] = quality['overall_score']

            # 2. Demand Characteristics (Syntetos-Boylan)
            if verbose:
                print("2/8 Computing demand characteristics...")
            chars = compute_demand_characteristics(
                df, key_columns, target_col,
                price_features_numeric=price_features_numeric,
                price_features_categorical=price_features_categorical,
                promo_features_numeric=promo_features_numeric,
                promo_features_categorical=promo_features_categorical,
            )
            chars = add_demand_classification(chars)

            # Add demand_pattern as alias for intermittency_class (for downstream compatibility)
            if 'intermittency_class' in chars.columns:
                chars['demand_pattern'] = chars['intermittency_class']

            # Add additional metrics
            pred_scores = compute_predictability_batch(df, key_columns, target_col)
            chars = chars.merge(pred_scores[key_columns + ['predictability_score']], on=key_columns, how='left')

            # Standardise key column to 'key' for downstream consistency
            actual_key = key_columns[0] if len(key_columns) == 1 else None
            if actual_key and actual_key != 'key' and actual_key in chars.columns and 'key' not in chars.columns:
                chars = chars.rename(columns={actual_key: 'key'})

            save_csv(chars, os.path.join(output_dir, 'per_key_metrics.csv'))
            result['files_created'].append('per_key_metrics.csv')

            # 3. Stationarity Testing
            if verbose:
                print("3/8 Testing stationarity...")
            stationarity = test_stationarity_batch(df, key_columns, target_col)
            save_csv(stationarity, os.path.join(output_dir, 'stationarity_results.csv'))
            result['files_created'].append('stationarity_results.csv')

            # 4. Feature Importance
            if numeric_features:
                if verbose:
                    print("4/8 Computing feature importance...")
                importance = compute_feature_importance_ensemble(
                    df, target_col, numeric_features, categorical_features
                )
                save_csv(importance, os.path.join(output_dir, 'feature_importance.csv'))
                result['files_created'].append('feature_importance.csv')
            elif verbose:
                print("4/8 Skipping feature importance (no numeric features specified)")

            # 5. Distribution Analysis
            if verbose:
                print("5/8 Analyzing distributions...")
            dist_analysis = analyze_distribution(df[target_col], 'target')
            save_json(dist_analysis, os.path.join(output_dir, 'target_distribution.json'))
            result['files_created'].append('target_distribution.json')

            # 6. Create Visualizations
            if verbose:
                print("6/8 Creating visualizations...")
            setup_matplotlib()

            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)

            # Syntetos-Boylan scatter
            try:
                create_scatter_plot(
                    chars['adi'], chars['cv2'],
                    'Syntetos-Boylan Demand Classification',
                    os.path.join(charts_dir, 'syntetos_boylan.png'),
                    xlabel='ADI (Average Demand Interval)',
                    ylabel='CV² (Squared Coefficient of Variation)',
                    hue=chars['intermittency_class'],
                    add_quadrants=True,
                    x_threshold=ADI_THRESHOLD,
                    y_threshold=CV2_THRESHOLD,
                )
                result['files_created'].append('charts/syntetos_boylan.png')
            except Exception as e:
                logger.warning(f"Failed to create syntetos_boylan chart: {e}")

            # Demand pattern distribution
            try:
                dist_df = chars['intermittency_class'].value_counts().reset_index()
                dist_df.columns = ['class', 'count']
                create_bar_chart(
                    dist_df, 'class', 'count',
                    'Demand Pattern Distribution',
                    os.path.join(charts_dir, 'demand_patterns.png'),
                )
                result['files_created'].append('charts/demand_patterns.png')
            except Exception as e:
                logger.warning(f"Failed to create demand_patterns chart: {e}")

            # CV Distribution
            try:
                create_histogram(
                    chars['cv'],
                    'Coefficient of Variation Distribution',
                    os.path.join(charts_dir, 'cv_distribution.png'),
                    xlabel='CV',
                )
                result['files_created'].append('charts/cv_distribution.png')
            except Exception as e:
                logger.warning(f"Failed to create cv_distribution chart: {e}")

            # Zero Fraction Distribution
            try:
                create_histogram(
                    chars['zero_fraction'],
                    'Zero Fraction Distribution',
                    os.path.join(charts_dir, 'zero_fraction_distribution.png'),
                    xlabel='Zero Fraction',
                )
                result['files_created'].append('charts/zero_fraction_distribution.png')
            except Exception as e:
                logger.warning(f"Failed to create zero_fraction_distribution chart: {e}")

            # Stationarity Distribution
            try:
                if 'verdict' in stationarity.columns:
                    stat_dist = stationarity['verdict'].value_counts().reset_index()
                    stat_dist.columns = ['verdict', 'count']
                    create_bar_chart(
                        stat_dist, 'verdict', 'count',
                        'Stationarity Distribution',
                        os.path.join(charts_dir, 'stationarity_distribution.png'),
                    )
                    result['files_created'].append('charts/stationarity_distribution.png')
            except Exception as e:
                logger.warning(f"Failed to create stationarity_distribution chart: {e}")

            # Predictability Distribution
            try:
                if 'predictability_score' in chars.columns:
                    valid_pred = chars['predictability_score'].dropna()
                    if len(valid_pred) > 0:
                        create_histogram(
                            valid_pred,
                            'Predictability Score Distribution',
                            os.path.join(charts_dir, 'predictability_distribution.png'),
                            xlabel='Predictability Score',
                        )
                        result['files_created'].append('charts/predictability_distribution.png')
            except Exception as e:
                logger.warning(f"Failed to create predictability_distribution chart: {e}")

            # Feature Importance Bar Charts - Separate for Numeric and Categorical
            try:
                importance_path = os.path.join(output_dir, 'feature_importance.csv')
                if os.path.exists(importance_path):
                    importance_df = pd.read_csv(importance_path)

                    # Split by type
                    importance_df_c = importance_df[importance_df['type'] == 'categorical'].copy()
                    importance_df_c.sort_values(by='ensemble_score', ascending=False, inplace=True)

                    importance_df_n = importance_df[importance_df['type'] == 'numeric'].copy()
                    importance_df_n.sort_values(by='ensemble_score', ascending=False, inplace=True)

                    # Chart 1: Top Categorical Features
                    if len(importance_df_c) > 0:
                        top_n = min(20, len(importance_df_c))
                        top_features = importance_df_c.head(top_n)[['feature', 'ensemble_score']]
                        create_bar_chart(
                            top_features, 'feature', 'ensemble_score',
                            f'Top {top_n} Categorical Feature Importance',
                            os.path.join(charts_dir, 'feature_importance_categorical.png'),
                            horizontal=True,
                        )
                        result['files_created'].append('charts/feature_importance_categorical.png')

                    # Chart 2: Top Numeric Features
                    if len(importance_df_n) > 0:
                        top_n = min(20, len(importance_df_n))
                        top_features = importance_df_n.head(top_n)[['feature', 'ensemble_score']]
                        create_bar_chart(
                            top_features, 'feature', 'ensemble_score',
                            f'Top {top_n} Numeric Feature Importance',
                            os.path.join(charts_dir, 'feature_importance_numeric.png'),
                            horizontal=True,
                        )
                        result['files_created'].append('charts/feature_importance_numeric.png')

                    # Chart 3: Overall Top Features (Combined with type label)
                    if len(importance_df) > 0:
                        importance_df_all = importance_df.sort_values(by='ensemble_score', ascending=False).copy()
                        top_n = min(25, len(importance_df_all))
                        top_all = importance_df_all.head(top_n)[['feature', 'ensemble_score', 'type']].copy()
                        # Add type indicator to feature name
                        top_all['feature_label'] = top_all.apply(
                            lambda x: f"{x['feature']} (C)" if x['type'] == 'categorical' else f"{x['feature']} (N)",
                            axis=1
                        )
                        create_bar_chart(
                            top_all, 'feature_label', 'ensemble_score',
                            f'Top {top_n} Features Overall (N=Numeric, C=Categorical)',
                            os.path.join(charts_dir, 'feature_importance_all.png'),
                            horizontal=True,
                        )
                        result['files_created'].append('charts/feature_importance_all.png')

            except Exception as e:
                logger.warning(f"Failed to create feature_importance charts: {e}")

            # =====================================================================
            # EXHAUSTIVE ANALYSES (Steps 7-10) - Run ALL analyses regardless of pattern
            # Philosophy: Analyze everything, let recommendations filter what's relevant
            # =====================================================================
            exhaustive_results = {}

            if exhaustive:
                # 7. AUTOCORRELATION ANALYSIS - Aggregate ACF/PACF summary
                if verbose:
                    print(f"7/{total_steps} Computing autocorrelation summary (ACF/PACF)...")
                try:
                    # Generate lags list based on period (e.g., [1, 4, 13, 26, 52] for weekly)
                    max_lag = min(period, 52)
                    acf_lags = [1, 4, 13, 26, 52] if period >= 52 else [1, max(2, period//4), max(4, period//2), period]
                    acf_lags = [l for l in acf_lags if l <= max_lag]
                    acf_summary = compute_autocorrelation_summary(df, key_columns, target_col, lags=acf_lags)
                    save_csv(acf_summary, os.path.join(output_dir, 'autocorrelation_summary.csv'))
                    result['files_created'].append('autocorrelation_summary.csv')
                    # Column names are acf_lag1, acf_lag4, etc. (not lag1_acf)
                    exhaustive_results['autocorrelation'] = {
                        'avg_lag1_acf': float(acf_summary['acf_lag1'].mean()) if 'acf_lag1' in acf_summary.columns else None,
                        # Find significant lags (absolute ACF > 0.2) for each series
                        'significant_lags': [],
                        'n_series_analyzed': len(acf_summary),
                    }
                    # Identify which lags have strong autocorrelation on average
                    acf_cols = [c for c in acf_summary.columns if c.startswith('acf_lag')]
                    for col in acf_cols:
                        avg_acf = acf_summary[col].dropna().abs().mean()
                        if avg_acf > 0.2:
                            lag_num = int(col.replace('acf_lag', ''))
                            exhaustive_results['autocorrelation']['significant_lags'].append(lag_num)
                    if verbose:
                        print(f"   ACF/PACF computed for {len(acf_summary)} series")
                except Exception as e:
                    logger.warning(f"ACF/PACF analysis failed: {e}")
                    exhaustive_results['autocorrelation'] = {'error': str(e)}

                # 8. SEASONALITY DETECTION - Detect seasonal patterns across series
                if verbose:
                    print(f"8/{total_steps} Detecting seasonality patterns...")
                try:
                    # Sample series for seasonality detection (for efficiency)
                    unique_keys = df[key_col].unique()
                    sample_size = min(100, len(unique_keys))
                    sample_keys = np.random.choice(unique_keys, size=sample_size, replace=False)

                    seasonality_results = []
                    for key in sample_keys:
                        series_data = df[df[key_col] == key][target_col]
                        if len(series_data) >= 20:
                            try:
                                season_result = detect_seasonality(series_data, max_period=period)
                                season_result['key'] = key
                                seasonality_results.append(season_result)
                            except:
                                pass

                    if seasonality_results:
                        seasonality_df = pd.DataFrame(seasonality_results)

                        # Compute average common_period_scores across all series
                        # common_period_scores is a dict column - need to aggregate manually
                        avg_period_scores = {}
                        if 'common_period_scores' in seasonality_df.columns:
                            all_scores = seasonality_df['common_period_scores'].dropna().tolist()
                            if all_scores:
                                # Get all periods from first result
                                periods = list(all_scores[0].keys()) if all_scores[0] else []
                                for p in periods:
                                    scores = [s.get(p, 0) for s in all_scores if isinstance(s, dict)]
                                    if scores:
                                        avg_period_scores[p] = round(sum(scores) / len(scores), 4)

                        # Compute average seasonal strength (new metric for adaptive feature selection)
                        avg_seasonal_strength = 0.0
                        if 'seasonal_strength' in seasonality_df.columns:
                            avg_seasonal_strength = float(seasonality_df['seasonal_strength'].mean())

                        save_json({
                            'sample_size': sample_size,
                            'has_seasonality_pct': float(seasonality_df['has_seasonality'].mean()) if 'has_seasonality' in seasonality_df.columns else 0,
                            'avg_seasonal_strength': round(avg_seasonal_strength, 4),
                            'dominant_periods': seasonality_df['dominant_period'].value_counts().head(5).to_dict() if 'dominant_period' in seasonality_df.columns else {},
                            'common_period_scores': avg_period_scores,
                        }, os.path.join(output_dir, 'seasonality_analysis.json'))
                        result['files_created'].append('seasonality_analysis.json')
                        exhaustive_results['seasonality'] = {
                            'has_seasonality_pct': float(seasonality_df['has_seasonality'].mean()) if 'has_seasonality' in seasonality_df.columns else 0,
                            'avg_seasonal_strength': round(avg_seasonal_strength, 4),
                            'dominant_period': int(seasonality_df['dominant_period'].mode().iloc[0]) if 'dominant_period' in seasonality_df.columns and len(seasonality_df['dominant_period'].dropna()) > 0 else None,
                        }
                        if verbose:
                            print(f"   {exhaustive_results['seasonality']['has_seasonality_pct']*100:.0f}% series show significant seasonality (avg strength: {avg_seasonal_strength:.2f})")
                except Exception as e:
                    logger.warning(f"Seasonality detection failed: {e}")
                    exhaustive_results['seasonality'] = {'error': str(e)}

                # 9. TREND STRENGTH ANALYSIS - Compute trend strength for series
                if verbose:
                    print(f"9/{total_steps} Analyzing trend strength...")
                try:
                    trend_results = []
                    unique_keys = df[key_col].unique()
                    sample_size = min(200, len(unique_keys))
                    sample_keys = np.random.choice(unique_keys, size=sample_size, replace=False)

                    for key in sample_keys:
                        series_data = df[df[key_col] == key][target_col]
                        if len(series_data) >= 10:
                            try:
                                trend_strength = compute_trend_strength(series_data, window=min(period // 2, 26))
                                trend_results.append({'key': key, 'trend_strength': trend_strength})
                            except:
                                pass

                    if trend_results:
                        trend_df = pd.DataFrame(trend_results)
                        exhaustive_results['trend'] = {
                            'avg_trend_strength': float(trend_df['trend_strength'].mean()),
                            'strongly_trending_pct': float((trend_df['trend_strength'] > 0.5).mean()),
                            'weak_trend_pct': float((trend_df['trend_strength'] < 0.2).mean()),
                            'n_series_analyzed': len(trend_df),
                        }
                        save_json(exhaustive_results['trend'], os.path.join(output_dir, 'trend_analysis.json'))
                        result['files_created'].append('trend_analysis.json')
                        if verbose:
                            print(f"   Avg trend strength: {exhaustive_results['trend']['avg_trend_strength']:.2f}")
                except Exception as e:
                    logger.warning(f"Trend analysis failed: {e}")
                    exhaustive_results['trend'] = {'error': str(e)}

                # 10. CHANGEPOINT DETECTION (Sampled) - Detect structural changes
                if verbose:
                    print(f"10/{total_steps} Detecting changepoints...")
                try:
                    changepoint_results = []
                    unique_keys = df[key_col].unique()
                    sample_size = min(50, len(unique_keys))  # Smaller sample - more expensive
                    sample_keys = np.random.choice(unique_keys, size=sample_size, replace=False)

                    for key in sample_keys:
                        series_data = df[df[key_col] == key][target_col]
                        if len(series_data) >= 30:
                            try:
                                cp_result = detect_changepoints(series_data, model='l2', n_bkps=3)
                                cp_result['key'] = key
                                changepoint_results.append(cp_result)
                            except:
                                pass

                    if changepoint_results:
                        cp_df = pd.DataFrame(changepoint_results)
                        avg_changepoints = cp_df['n_changepoints'].mean() if 'n_changepoints' in cp_df.columns else 0

                        # Track SIGNIFICANT changepoints (those with meaningful mean shifts)
                        pct_with_significant = 0.0
                        avg_mean_shift_ratio = 0.0
                        if 'significant_changepoints' in cp_df.columns:
                            pct_with_significant = float(cp_df['significant_changepoints'].mean())
                        if 'avg_mean_shift_ratio' in cp_df.columns:
                            avg_mean_shift_ratio = float(cp_df['avg_mean_shift_ratio'].mean())

                        exhaustive_results['changepoints'] = {
                            'avg_changepoints_per_series': float(avg_changepoints),
                            'pct_with_changepoints': float((cp_df['n_changepoints'] > 0).mean()) if 'n_changepoints' in cp_df.columns else 0,
                            'pct_with_significant_changepoints': round(pct_with_significant, 4),
                            'avg_mean_shift_ratio': round(avg_mean_shift_ratio, 4),
                            'n_series_analyzed': len(cp_df),
                        }
                        save_json(exhaustive_results['changepoints'], os.path.join(output_dir, 'changepoint_analysis.json'))
                        result['files_created'].append('changepoint_analysis.json')
                        if verbose:
                            print(f"   {pct_with_significant*100:.0f}% series have SIGNIFICANT changepoints (avg shift ratio: {avg_mean_shift_ratio:.2f})")
                except Exception as e:
                    logger.warning(f"Changepoint detection failed: {e}")
                    exhaustive_results['changepoints'] = {'error': str(e)}

                # 11. CROSS-SERIES CORRELATION (for hierarchical data)
                if verbose:
                    print(f"11/{total_steps} Analyzing cross-series correlations...")
                try:
                    if len(key_columns) > 1:
                        # For multi-level hierarchies, compute correlation within groups
                        cross_corr = compute_cross_series_correlation(
                            df, key_columns, target_col, date_col,
                            sample_size=min(50, df[key_col].nunique())
                        )
                        exhaustive_results['cross_correlation'] = cross_corr
                        save_json(cross_corr, os.path.join(output_dir, 'cross_series_correlation.json'))
                        result['files_created'].append('cross_series_correlation.json')
                        if verbose:
                            print(f"   Avg correlation: {cross_corr.get('avg_correlation', 'N/A')}")
                    else:
                        exhaustive_results['cross_correlation'] = {'note': 'Single-level hierarchy - no cross-series correlation computed'}
                except Exception as e:
                    logger.warning(f"Cross-series correlation failed: {e}")
                    exhaustive_results['cross_correlation'] = {'error': str(e)}

            result['exhaustive_results'] = exhaustive_results

            # 12. Summary Statistics (updated step number)
            if verbose:
                print(f"{'12' if exhaustive else '7'}/{total_steps} Computing summary statistics...")

            # Get date range from data
            date_min = str(df[date_col].min()) if date_col in df.columns else None
            date_max = str(df[date_col].max()) if date_col in df.columns else None
            n_periods = df[date_col].nunique() if date_col in df.columns else None

            summary = {
                'total_series': len(chars),
                'total_rows': len(df),
                'date_range': {'start': date_min, 'end': date_max, 'n_periods': n_periods},
                'seasonal_period': period,
                'intermittency_distribution': chars['intermittency_class'].value_counts(normalize=True).to_dict(),
                'lumpy_intermittent_pct': float((chars['intermittency_class'].isin(['lumpy', 'intermittent'])).mean()),
                'stationarity_distribution': stationarity['verdict'].value_counts(normalize=True).to_dict() if 'verdict' in stationarity else {},
                'avg_cv': float(chars['cv'].mean()),
                'avg_zero_fraction': float(chars['zero_fraction'].mean()),
                'avg_predictability': float(chars['predictability_score'].mean()) if 'predictability_score' in chars else None,
                'data_quality_score': quality['overall_score'],
                # Feature counts for analyst
                'numeric_features_analyzed': len(numeric_features) if numeric_features else 0,
                'categorical_features_analyzed': len(categorical_features) if categorical_features else 0,
                'numeric_features': numeric_features if numeric_features else [],
                'categorical_features': categorical_features if categorical_features else [],
                # DEAD KEY SUMMARY - CRITICAL for downstream crews
                'dead_key_summary': {
                    'total_keys_in_data': dead_key_summary.total_keys,
                    'dead_keys': dead_key_summary.dead_keys,
                    'active_keys': dead_key_summary.active_keys,
                    'dead_key_percentage': dead_key_summary.dead_key_percentage,
                    'dead_key_threshold': dead_key_threshold,
                    'note': 'Dead keys are excluded from all analysis. They will be forecasted as 0.',
                },
                # Exhaustive analysis results summary
                'exhaustive_mode': exhaustive,
                'exhaustive_results_summary': {
                    k: {kk: vv for kk, vv in v.items() if kk != 'error'}
                    for k, v in exhaustive_results.items()
                    if isinstance(v, dict) and 'error' not in v
                } if exhaustive else {},
            }

            save_json(summary, os.path.join(output_dir, 'eda_summary.json'))
            result['files_created'].append('eda_summary.json')
            result['summary'] = summary

            # 13. Generate Comprehensive Markdown Report
            if verbose:
                print(f"{'13' if exhaustive else '8'}/{total_steps} Generating EDA report...")
            report = generate_eda_report(
                chars, stationarity, summary, quality,
                output_path=os.path.join(output_dir, 'eda_report.md'),
                exhaustive_results=exhaustive_results if exhaustive else None,
            )
            result['files_created'].append('eda_report.md')

            # =====================================================================
            # NOTE ON CONTEXT FILES FOR DOWNSTREAM CREWS
            # =====================================================================
            # The context files (eda_to_segmentation_context.json, eda_to_feature_context.json,
            # eda_to_training_context.json) are created by the EDA ANALYST AGENT, not here.
            #
            # This is INTENTIONAL because:
            # 1. The agent reads and INTERPRETS the raw outputs with expert understanding
            # 2. The agent can make nuanced, contextual recommendations
            # 3. The agent adds value by understanding patterns a static function cannot
            # 4. The agent can adapt recommendations based on edge cases and combinations
            #
            # The pipeline's job is to produce comprehensive RAW OUTPUTS that the analyst
            # can then interpret intelligently.
            # =====================================================================

            # Store data profile in result for analyst to use
            if data_profile is not None:
                # Update profile with stationarity info from EDA results
                if 'verdict' in stationarity.columns:
                    non_stationary_pct = (stationarity['verdict'] == 'NON_STATIONARY').mean()
                    data_profile.pct_non_stationary = non_stationary_pct
                result['data_profile'] = data_profile.to_dict()

            if verbose:
                print("=" * 60)
                print(f"EDA PIPELINE COMPLETE!")
                print(f"Files saved to: {output_dir}")
                print(f"Total files created: {len(result['files_created'])}")
                print("=" * 60)
                print("\n📋 RAW OUTPUTS READY FOR EDA ANALYST:")
                print("   The EDA Analyst agent will now interpret these results")
                print("   and create intelligent context files for downstream crews.")

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()

    return result


def generate_eda_report(
    chars_df: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    summary: Dict[str, Any],
    quality: Dict[str, Any],
    output_path: str = 'eda_report.md',
    exhaustive_results: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate comprehensive EDA markdown report with all analysis results.

    This report includes:
    - Executive summary of data characteristics
    - Data quality analysis
    - Demand pattern distribution (Syntetos-Boylan)
    - Stationarity analysis
    - Per-series metrics summary
    - Exhaustive analysis results (if available)
    - Recommendations for downstream crews

    Parameters
    ----------
    chars_df : pd.DataFrame
        Demand characteristics DataFrame
    stationarity_df : pd.DataFrame
        Stationarity results DataFrame
    summary : Dict[str, Any]
        Summary statistics
    quality : Dict[str, Any]
        Data quality report
    output_path : str
        Output file path
    exhaustive_results : Dict[str, Any], optional
        Results from exhaustive analyses (ACF/PACF, seasonality, trend, changepoints)

    Returns
    -------
    str
        Path to generated report
    """
    report = []

    report.append("# Exploratory Data Analysis Report\n")
    report.append("*Auto-generated by FEU-Agentic-Forecasting State-of-the-Art EDA Pipeline*\n\n")

    if exhaustive_results:
        report.append("> **Exhaustive Analysis Mode**: All applicable analyses were run regardless of detected patterns.\n")
        report.append("> Smart recommendations filter and prioritize based on actual findings.\n\n")
    else:
        report.append("> **Note**: This is the EDA output. See `eda_to_*_context.json` files for\n")
        report.append("> actionable recommendations for downstream crews.\n\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"- **Total Series Analyzed**: {summary.get('total_series', 'N/A')}\n")
    report.append(f"- **Total Rows**: {summary.get('total_rows', 'N/A'):,}\n")
    report.append(f"- **Date Range**: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}\n")
    report.append(f"- **Seasonal Period**: {summary.get('seasonal_period', 'N/A')}\n")
    report.append(f"- **Data Quality Score**: {quality.get('overall_score', 'N/A')}/100\n")
    report.append(f"- **Lumpy/Intermittent Series**: {summary.get('lumpy_intermittent_pct', 0):.1%}\n")
    report.append(f"- **Average CV**: {summary.get('avg_cv', 0):.2f}\n")
    report.append(f"- **Average Zero Fraction**: {summary.get('avg_zero_fraction', 0):.1%}\n")
    if summary.get('avg_predictability'):
        report.append(f"- **Average Predictability Score**: {summary.get('avg_predictability', 0):.2f}\n")
    report.append(f"- **Numeric Features**: {summary.get('numeric_features_analyzed', 0)}\n")
    report.append(f"- **Categorical Features**: {summary.get('categorical_features_analyzed', 0)}\n")
    report.append("\n")

    # Data Quality
    report.append("## Data Quality\n")
    report.append(f"- **Overall Score**: {quality.get('overall_score', 'N/A')}/100\n")
    if quality.get('quality_flags'):
        report.append("- **Quality Flags**:\n")
        for flag in quality['quality_flags']:
            report.append(f"  - ⚠️ {flag}\n")
    else:
        report.append("- **Quality Flags**: None - data quality is good\n")
    report.append("\n")

    # Demand Pattern Distribution
    report.append("## Demand Pattern Distribution (Syntetos-Boylan)\n")
    report.append("| Pattern | Count | Percentage | Description |\n")
    report.append("|---------|-------|------------|-------------|\n")
    dist = summary.get('intermittency_distribution', {})
    pattern_descriptions = {
        'smooth': 'Regular, predictable (low ADI, low CV²)',
        'erratic': 'High variability (low ADI, high CV²)',
        'intermittent': 'Sporadic occurrence (high ADI, low CV²)',
        'lumpy': 'Sporadic AND variable (high ADI, high CV²)',
    }
    for pattern, pct in sorted(dist.items()):
        count = int(pct * summary.get('total_series', 0))
        desc = pattern_descriptions.get(pattern, '')
        report.append(f"| {pattern} | {count:,} | {pct:.1%} | {desc} |\n")
    report.append("\n")

    # Stationarity Summary
    report.append("## Stationarity Analysis\n")
    if 'verdict' in stationarity_df.columns:
        stat_dist = stationarity_df['verdict'].value_counts(normalize=True)
        report.append("| Status | Percentage | Implication |\n")
        report.append("|--------|------------|-------------|\n")
        status_implications = {
            'STATIONARY': 'No differencing needed',
            'NON_STATIONARY': 'Consider first differencing',
            'TREND_STATIONARY': 'Detrend the series',
            'DIFFERENCE_STATIONARY': 'Apply differencing',
            'INSUFFICIENT_DATA': 'Cannot determine',
            'ERROR': 'Test failed',
        }
        for status, pct in stat_dist.items():
            impl = status_implications.get(status, '')
            report.append(f"| {status} | {pct:.1%} | {impl} |\n")
    report.append("\n")

    # Per-Key Metrics Summary Statistics
    report.append("## Per-Series Metrics Summary\n")
    report.append("| Metric | Mean | Std | Min | Median | Max |\n")
    report.append("|--------|------|-----|-----|--------|-----|\n")
    for col in ['cv', 'adi', 'cv2', 'zero_fraction', 'mean', 'std']:
        if col in chars_df.columns:
            vals = chars_df[col].dropna()
            if len(vals) > 0:
                report.append(f"| {col} | {vals.mean():.3f} | {vals.std():.3f} | {vals.min():.3f} | {vals.median():.3f} | {vals.max():.3f} |\n")
    report.append("\n")

    # Available columns for analyst
    report.append("## Available Metrics for Analysis\n")
    report.append("The following metrics are available in `per_key_metrics.csv` for the Analyst:\n\n")
    report.append("**Per-Series Metrics:**\n")
    for col in chars_df.columns:
        report.append(f"- `{col}`\n")
    report.append("\n")

    # =========================================================================
    # EXHAUSTIVE ANALYSIS RESULTS
    # =========================================================================
    if exhaustive_results:
        report.append("## Exhaustive Analysis Results\n\n")
        report.append("The following advanced analyses were performed on sampled series:\n\n")

        # Autocorrelation
        if 'autocorrelation' in exhaustive_results:
            acf_res = exhaustive_results['autocorrelation']
            report.append("### Autocorrelation Analysis (ACF/PACF)\n")
            if 'error' not in acf_res:
                report.append(f"- **Average Lag-1 ACF**: {acf_res.get('avg_lag1_acf', 'N/A'):.3f}\n")
                report.append(f"- **Series Analyzed**: {acf_res.get('n_series_analyzed', 'N/A')}\n")
            else:
                report.append(f"- ⚠️ Analysis failed: {acf_res.get('error')}\n")
            report.append("\n")

        # Seasonality
        if 'seasonality' in exhaustive_results:
            season_res = exhaustive_results['seasonality']
            report.append("### Seasonality Detection\n")
            if 'error' not in season_res:
                has_season = season_res.get('has_seasonality_pct', 0)
                report.append(f"- **Series with Seasonality**: {has_season*100:.1f}%\n")
                if season_res.get('dominant_period'):
                    report.append(f"- **Dominant Period**: {season_res.get('dominant_period')}\n")
            else:
                report.append(f"- ⚠️ Analysis failed: {season_res.get('error')}\n")
            report.append("\n")

        # Trend
        if 'trend' in exhaustive_results:
            trend_res = exhaustive_results['trend']
            report.append("### Trend Analysis\n")
            if 'error' not in trend_res:
                report.append(f"- **Average Trend Strength**: {trend_res.get('avg_trend_strength', 'N/A'):.2f}\n")
                report.append(f"- **Strongly Trending**: {trend_res.get('strongly_trending_pct', 0)*100:.1f}%\n")
                report.append(f"- **Weak/No Trend**: {trend_res.get('weak_trend_pct', 0)*100:.1f}%\n")
            else:
                report.append(f"- ⚠️ Analysis failed: {trend_res.get('error')}\n")
            report.append("\n")

        # Changepoints
        if 'changepoints' in exhaustive_results:
            cp_res = exhaustive_results['changepoints']
            report.append("### Changepoint Detection\n")
            if 'error' not in cp_res:
                report.append(f"- **Series with Changepoints**: {cp_res.get('pct_with_changepoints', 0)*100:.1f}%\n")
                report.append(f"- **Avg Changepoints per Series**: {cp_res.get('avg_changepoints_per_series', 'N/A'):.1f}\n")
            else:
                report.append(f"- ⚠️ Analysis failed: {cp_res.get('error')}\n")
            report.append("\n")

        # Cross-correlation
        if 'cross_correlation' in exhaustive_results:
            cc_res = exhaustive_results['cross_correlation']
            report.append("### Cross-Series Correlation\n")
            if 'error' not in cc_res and 'note' not in cc_res:
                report.append(f"- **Average Correlation**: {cc_res.get('avg_correlation', 'N/A')}\n")
            elif 'note' in cc_res:
                report.append(f"- {cc_res.get('note')}\n")
            else:
                report.append(f"- ⚠️ Analysis failed: {cc_res.get('error')}\n")
            report.append("\n")

    report.append("---\n\n")
    report.append("## Output Files Summary\n\n")

    report.append("**Core Analysis Files:**\n")
    report.append("- `per_key_metrics.csv` - Per-series demand characteristics\n")
    report.append("- `stationarity_results.csv` - Stationarity test results\n")
    report.append("- `feature_importance.csv` - Feature importance rankings\n")
    report.append("- `eda_summary.json` - Summary statistics\n")
    report.append("- `data_quality.json` - Data quality details\n")
    report.append("- `data_profile.json` - Comprehensive data profile\n")
    report.append("- `charts/` - Visualization outputs\n\n")

    if exhaustive_results:
        report.append("**Exhaustive Analysis Files:**\n")
        report.append("- `autocorrelation_summary.csv` - ACF/PACF analysis\n")
        report.append("- `seasonality_analysis.json` - Seasonality detection\n")
        report.append("- `trend_analysis.json` - Trend strength analysis\n")
        report.append("- `changepoint_analysis.json` - Changepoint detection\n")
        if 'cross_correlation' in exhaustive_results and 'error' not in exhaustive_results.get('cross_correlation', {}):
            report.append("- `cross_series_correlation.json` - Cross-series analysis\n")
        report.append("\n")

    report.append("**Context Files (Created by EDA Analyst Agent):**\n")
    report.append("The EDA Analyst agent will interpret the above results and create:\n")
    report.append("- `eda_to_segmentation_context.json` - Segmentation crew guidance\n")
    report.append("- `eda_to_feature_context.json` - Feature engineering crew guidance\n")
    report.append("- `eda_to_training_context.json` - Training crew guidance\n\n")
    report.append("> **Note**: Context files are created by the EDA Analyst agent who interprets\n")
    report.append("> the raw outputs with expert understanding and generates intelligent,\n")
    report.append("> contextual recommendations for downstream crews.\n\n")

    report.append("---\n")
    report.append("*EDA Pipeline Complete. Raw outputs ready for EDA Analyst interpretation.*\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    return output_path


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data Quality
    'analyze_data_quality',
    'analyze_missing_patterns',
    'recommend_imputation_strategy',

    # Distribution Analysis
    'analyze_distribution',
    'compute_distribution_statistics_by_group',

    # Outlier Detection
    'detect_outliers',
    'detect_outliers_by_group',
    'detect_multivariate_outliers',

    # Time Series Decomposition
    'decompose_time_series',
    'compute_trend_strength',
    'compute_seasonal_strength',

    # Stationarity
    'comprehensive_stationarity_test',
    'test_stationarity_batch',

    # Autocorrelation
    'compute_acf_pacf',
    'compute_autocorrelation_summary',

    # Seasonality
    'detect_seasonality',
    'compute_seasonal_indices',

    # Changepoint Detection
    'detect_changepoints',
    'detect_level_shifts',

    # Feature Importance
    'compute_feature_importance_ensemble',
    'compute_feature_correlations',

    # Predictability
    'compute_predictability_score',
    'compute_predictability_batch',

    # Cross-Series Analysis
    'compute_cross_series_correlation',
    'test_granger_causality',

    # Recommendation Generators
    'generate_segmentation_recommendations',
    'generate_feature_recommendations',

    # Pipeline
    'run_eda_pipeline',
    'generate_eda_report',
]
