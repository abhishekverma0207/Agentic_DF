"""
Data Profiler Module for Adaptive EDA
======================================

This module provides comprehensive data profiling that characterizes the data
WITHOUT limiting which analyses run. The philosophy is:

1. RUN ALL APPLICABLE ANALYSES - Don't pre-filter based on detected patterns
2. PROFILE FOR CONTEXT - Understand data characteristics to contextualize findings
3. SMART RECOMMENDATIONS - Use detected characteristics to prioritize and filter
   recommendations for downstream crews (segmentation, feature engineering, training)

The profiler output helps interpret EDA results and generate targeted recommendations,
but does NOT restrict which analyses are performed.

Key Insight: A dataset often has MIXED patterns (some series smooth, some intermittent,
some lumpy). Running all analyses ensures we capture insights across all segments.
The recommendations then highlight what's most relevant based on the distribution.

Usage:
    from utils.data_profiler import profile_data, DataProfile

    profile = profile_data(df, key_columns, date_col, target_col)
    print(f"Primary Pattern: {profile.primary_demand_pattern}")
    print(f"Pattern Distribution: {profile.pattern_distribution}")

    # Profile informs recommendations, not analysis selection
    recommendations = profile.generate_downstream_recommendations()
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS FOR DATA CLASSIFICATION
# =============================================================================

class DemandPatternType(Enum):
    """Primary demand pattern classification (Syntetos-Boylan based)."""
    SMOOTH = "smooth"           # Low ADI, low CV² - regular, predictable
    ERRATIC = "erratic"         # Low ADI, high CV² - frequent but variable
    INTERMITTENT = "intermittent"  # High ADI, low CV² - sporadic but consistent size
    LUMPY = "lumpy"             # High ADI, high CV² - sporadic AND variable
    MIXED = "mixed"             # Heterogeneous across series


class TimeGranularity(Enum):
    """Time series granularity."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class HierarchyType(Enum):
    """Hierarchy structure type."""
    SINGLE_LEVEL = "single_level"    # One key column
    MULTI_LEVEL = "multi_level"      # Multiple key columns (e.g., customer-product)
    HIERARCHICAL = "hierarchical"    # Tree structure with parent-child relationships


class VolumeCategory(Enum):
    """Volume-based classification."""
    HIGH_VOLUME = "high_volume"      # Frequent, large quantities
    MEDIUM_VOLUME = "medium_volume"
    LOW_VOLUME = "low_volume"        # Sparse, small quantities


class TemporalCharacteristic(Enum):
    """Dominant temporal characteristic."""
    STRONGLY_TRENDING = "strongly_trending"
    MODERATELY_TRENDING = "moderately_trending"
    STRONGLY_SEASONAL = "strongly_seasonal"
    MODERATELY_SEASONAL = "moderately_seasonal"
    STATIONARY = "stationary"
    MIXED_TEMPORAL = "mixed_temporal"


# =============================================================================
# SYNTETOS-BOYLAN THRESHOLDS
# =============================================================================
ADI_THRESHOLD = 1.32   # Average Demand Interval threshold
CV2_THRESHOLD = 0.49   # Squared Coefficient of Variation threshold


# =============================================================================
# DATA PROFILE DATACLASS
# =============================================================================

@dataclass
class DataProfile:
    """
    Comprehensive data profile for contextualizing EDA results.

    This profile does NOT limit which analyses run - it provides context
    for interpreting results and generating smart recommendations.
    """
    # =========================================================================
    # METADATA
    # =========================================================================
    total_rows: int = 0
    total_series: int = 0
    n_time_periods: int = 0
    date_range: Dict[str, str] = field(default_factory=dict)

    # =========================================================================
    # TIME GRANULARITY (detected from data)
    # =========================================================================
    time_granularity: TimeGranularity = TimeGranularity.WEEKLY
    detected_period: int = 52  # Seasonal period based on granularity

    # =========================================================================
    # DEMAND PATTERN ANALYSIS (Syntetos-Boylan based)
    # =========================================================================
    pattern_distribution: Dict[str, float] = field(default_factory=dict)
    primary_demand_pattern: DemandPatternType = DemandPatternType.MIXED

    # Detailed percentages
    pct_smooth: float = 0.0
    pct_erratic: float = 0.0
    pct_intermittent: float = 0.0
    pct_lumpy: float = 0.0

    # Aggregated intermittency metrics
    avg_zero_fraction: float = 0.0
    median_zero_fraction: float = 0.0
    avg_adi: float = 0.0
    avg_cv2: float = 0.0
    avg_cv: float = 0.0

    # =========================================================================
    # VOLUME ANALYSIS
    # =========================================================================
    volume_category: VolumeCategory = VolumeCategory.MEDIUM_VOLUME
    avg_demand_per_period: float = 0.0
    median_demand_per_period: float = 0.0
    total_demand_volume: float = 0.0
    avg_series_length: float = 0.0

    pct_high_volume: float = 0.0
    pct_medium_volume: float = 0.0
    pct_low_volume: float = 0.0

    # =========================================================================
    # TEMPORAL ANALYSIS
    # =========================================================================
    temporal_characteristic: TemporalCharacteristic = TemporalCharacteristic.MIXED_TEMPORAL
    avg_trend_strength: float = 0.0
    avg_seasonal_strength: float = 0.0
    pct_non_stationary: float = 0.0
    pct_stationary: float = 0.0

    # =========================================================================
    # HIERARCHY ANALYSIS
    # =========================================================================
    hierarchy_type: HierarchyType = HierarchyType.SINGLE_LEVEL
    hierarchy_levels: List[str] = field(default_factory=list)
    n_hierarchy_levels: int = 1
    series_per_level: Dict[str, int] = field(default_factory=dict)

    # =========================================================================
    # FEATURE ANALYSIS
    # =========================================================================
    has_price_features: bool = False
    has_promo_features: bool = False
    has_holiday_features: bool = False
    has_weather_features: bool = False
    n_numeric_features: int = 0
    n_categorical_features: int = 0
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)

    # =========================================================================
    # DATA QUALITY
    # =========================================================================
    missing_pct: float = 0.0
    duplicate_pct: float = 0.0
    has_negatives: bool = False
    quality_score: int = 100
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for JSON serialization."""
        return {
            'metadata': {
                'total_rows': self.total_rows,
                'total_series': self.total_series,
                'n_time_periods': self.n_time_periods,
                'date_range': self.date_range,
            },
            'time_granularity': {
                'granularity': self.time_granularity.value,
                'detected_period': self.detected_period,
            },
            'demand_pattern_analysis': {
                'primary_pattern': self.primary_demand_pattern.value,
                'pattern_distribution': self.pattern_distribution,
                'pct_smooth': round(self.pct_smooth, 4),
                'pct_erratic': round(self.pct_erratic, 4),
                'pct_intermittent': round(self.pct_intermittent, 4),
                'pct_lumpy': round(self.pct_lumpy, 4),
                'avg_zero_fraction': round(self.avg_zero_fraction, 4),
                'median_zero_fraction': round(self.median_zero_fraction, 4),
                'avg_adi': round(self.avg_adi, 4),
                'avg_cv2': round(self.avg_cv2, 4),
                'avg_cv': round(self.avg_cv, 4),
            },
            'volume_analysis': {
                'volume_category': self.volume_category.value,
                'avg_demand_per_period': round(self.avg_demand_per_period, 2),
                'median_demand_per_period': round(self.median_demand_per_period, 2),
                'total_demand_volume': round(self.total_demand_volume, 2),
                'avg_series_length': round(self.avg_series_length, 2),
                'pct_high_volume': round(self.pct_high_volume, 4),
                'pct_medium_volume': round(self.pct_medium_volume, 4),
                'pct_low_volume': round(self.pct_low_volume, 4),
            },
            'temporal_analysis': {
                'temporal_characteristic': self.temporal_characteristic.value,
                'avg_trend_strength': round(self.avg_trend_strength, 4),
                'avg_seasonal_strength': round(self.avg_seasonal_strength, 4),
                'pct_non_stationary': round(self.pct_non_stationary, 4),
                'pct_stationary': round(self.pct_stationary, 4),
            },
            'hierarchy_analysis': {
                'hierarchy_type': self.hierarchy_type.value,
                'hierarchy_levels': self.hierarchy_levels,
                'n_hierarchy_levels': self.n_hierarchy_levels,
                'series_per_level': self.series_per_level,
            },
            'feature_analysis': {
                'has_price_features': self.has_price_features,
                'has_promo_features': self.has_promo_features,
                'has_holiday_features': self.has_holiday_features,
                'has_weather_features': self.has_weather_features,
                'n_numeric_features': self.n_numeric_features,
                'n_categorical_features': self.n_categorical_features,
            },
            'data_quality': {
                'missing_pct': round(self.missing_pct, 2),
                'duplicate_pct': round(self.duplicate_pct, 2),
                'has_negatives': self.has_negatives,
                'quality_score': self.quality_score,
                'quality_flags': self.quality_flags,
            },
        }


# =============================================================================
# PROFILING FUNCTIONS
# =============================================================================

def detect_time_granularity(
    df: pd.DataFrame,
    date_col: str,
) -> Tuple[TimeGranularity, int]:
    """Detect time series granularity from data."""
    try:
        dates = df[date_col].dropna().unique()
        sample_date = dates[0]

        # Check for year_week format (integer like 202401)
        if isinstance(sample_date, (int, np.integer)) or (
            isinstance(sample_date, str) and sample_date.isdigit() and len(str(sample_date)) == 6
        ):
            return TimeGranularity.WEEKLY, 52

        # Check for year_month format
        if isinstance(sample_date, (int, np.integer)):
            val = int(sample_date)
            period_part = val % 100
            if 1 <= period_part <= 12:
                return TimeGranularity.MONTHLY, 12
            elif 1 <= period_part <= 53:
                return TimeGranularity.WEEKLY, 52

        # Try to parse as datetime
        try:
            dates_parsed = pd.to_datetime(dates)
            dates_sorted = np.sort(dates_parsed)

            if len(dates_sorted) > 1:
                gaps = np.diff(dates_sorted)
                median_gap_days = np.median([g.days for g in gaps])

                if median_gap_days <= 1.5:
                    return TimeGranularity.DAILY, 365
                elif median_gap_days <= 8:
                    return TimeGranularity.WEEKLY, 52
                elif median_gap_days <= 35:
                    return TimeGranularity.MONTHLY, 12
                elif median_gap_days <= 95:
                    return TimeGranularity.QUARTERLY, 4
                else:
                    return TimeGranularity.YEARLY, 1
        except Exception:
            pass

        # Default based on number of unique dates
        n_unique = len(dates)
        if n_unique > 200:
            return TimeGranularity.DAILY, 365
        elif n_unique > 50:
            return TimeGranularity.WEEKLY, 52
        else:
            return TimeGranularity.MONTHLY, 12

    except Exception as e:
        logger.warning(f"Could not detect time granularity: {e}")
        return TimeGranularity.WEEKLY, 52


def detect_hierarchy_structure(
    df: pd.DataFrame,
    key_columns: List[str],
) -> Tuple[HierarchyType, Dict[str, Any]]:
    """Detect hierarchy structure from key columns."""
    n_levels = len(key_columns)

    hierarchy_info = {
        'levels': key_columns,
        'n_unique': {},
        'total_combinations': df.groupby(key_columns).ngroups if n_levels > 0 else 0,
    }

    for col in key_columns:
        if col in df.columns:
            hierarchy_info['n_unique'][col] = df[col].nunique()

    if n_levels == 1:
        return HierarchyType.SINGLE_LEVEL, hierarchy_info

    # Check if hierarchical (parent-child relationship)
    if n_levels >= 2:
        level1, level2 = key_columns[0], key_columns[1]
        level2_per_level1 = df.groupby(level1)[level2].nunique()
        if level2_per_level1.max() > 1:
            return HierarchyType.HIERARCHICAL, hierarchy_info

    return HierarchyType.MULTI_LEVEL, hierarchy_info


def compute_series_statistics(
    df: pd.DataFrame,
    key_columns: List[str],
    target_col: str,
) -> pd.DataFrame:
    """Compute comprehensive statistics for each series including Syntetos-Boylan."""
    grouped = df.groupby(key_columns)[target_col]

    # Basic aggregations
    stats = grouped.agg([
        ('count', 'count'),
        ('sum', 'sum'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
    ]).reset_index()

    # Zero count and fraction
    zero_counts = df[df[target_col] == 0].groupby(key_columns).size().reset_index(name='zero_count')
    stats = stats.merge(zero_counts, on=key_columns, how='left')
    stats['zero_count'] = stats['zero_count'].fillna(0)
    stats['zero_fraction'] = stats['zero_count'] / stats['count']

    # CV and CV²
    stats['cv'] = stats['std'] / (stats['mean'] + 1e-10)
    stats['cv2'] = stats['cv'] ** 2

    # ADI (Average Demand Interval)
    def compute_adi(group):
        values = group[target_col].values
        non_zero_indices = np.where(values > 0)[0]
        if len(non_zero_indices) <= 1:
            return float(len(values))
        intervals = np.diff(non_zero_indices)
        return float(np.mean(intervals)) if len(intervals) > 0 else float(len(values))

    adi_df = df.groupby(key_columns).apply(compute_adi).reset_index(name='adi')
    stats = stats.merge(adi_df, on=key_columns, how='left')
    stats['adi'] = stats['adi'].fillna(1)

    # Syntetos-Boylan classification
    def classify_sb(row):
        if row['adi'] < ADI_THRESHOLD and row['cv2'] < CV2_THRESHOLD:
            return 'smooth'
        elif row['adi'] < ADI_THRESHOLD and row['cv2'] >= CV2_THRESHOLD:
            return 'erratic'
        elif row['adi'] >= ADI_THRESHOLD and row['cv2'] < CV2_THRESHOLD:
            return 'intermittent'
        else:
            return 'lumpy'

    stats['pattern'] = stats.apply(classify_sb, axis=1)

    # Volume tier classification
    mean_vals = stats['mean']
    q33 = mean_vals.quantile(0.33)
    q67 = mean_vals.quantile(0.67)
    stats['volume_tier'] = pd.cut(
        mean_vals,
        bins=[-np.inf, q33, q67, np.inf],
        labels=['low', 'medium', 'high']
    )

    return stats


def compute_temporal_characteristics_sample(
    df: pd.DataFrame,
    key_columns: List[str],
    target_col: str,
    date_col: str,
    period: int = 52,
    sample_size: int = 100,
) -> Dict[str, float]:
    """Compute temporal characteristics on a sample of series for efficiency."""
    from statsmodels.tsa.stattools import adfuller

    # Create combined key
    if len(key_columns) == 1:
        df = df.copy()
        df['_key'] = df[key_columns[0]]
    else:
        df = df.copy()
        df['_key'] = df[key_columns].astype(str).agg('_'.join, axis=1)

    unique_keys = df['_key'].unique()
    if len(unique_keys) > sample_size:
        sampled_keys = np.random.choice(unique_keys, sample_size, replace=False)
    else:
        sampled_keys = unique_keys

    trend_strengths = []
    seasonal_strengths = []
    stationarity_results = []

    for key in sampled_keys:
        series = df[df['_key'] == key].sort_values(date_col)[target_col].values

        if len(series) < max(20, period):
            continue

        # Trend strength via moving average decomposition
        try:
            window = min(period // 2, len(series) // 4)
            if window >= 3:
                trend = pd.Series(series).rolling(window=window, center=True).mean().dropna().values
                if len(trend) > 10:
                    idx_start = window // 2
                    idx_end = idx_start + len(trend)
                    if idx_end <= len(series):
                        detrended = series[idx_start:idx_end] - trend
                        var_detrended = np.var(detrended)
                        var_original = np.var(series)
                        strength = max(0, 1 - var_detrended / (var_original + 1e-10))
                        trend_strengths.append(strength)
        except Exception:
            pass

        # Stationarity test
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_result = adfuller(series, autolag='AIC', maxlag=min(10, len(series)//4))
                is_stationary = adf_result[1] < 0.05
                stationarity_results.append(is_stationary)
        except Exception:
            pass

    return {
        'avg_trend_strength': float(np.mean(trend_strengths)) if trend_strengths else 0.0,
        'avg_seasonal_strength': 0.0,
        'pct_stationary': float(np.mean(stationarity_results)) if stationarity_results else 0.5,
        'sample_size': len(sampled_keys),
    }


def detect_feature_presence(
    df: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect presence of different feature types by column name patterns."""
    all_cols_lower = [c.lower() for c in df.columns]

    price_keywords = ['price', 'cost', 'msrp', 'srp', 'unit_price', 'avg_price']
    promo_keywords = ['promo', 'discount', 'offer', 'sale', 'promotion', 'campaign', 'deal']
    holiday_keywords = ['holiday', 'christmas', 'easter', 'thanksgiving', 'festival', 'event']
    weather_keywords = ['weather', 'temperature', 'temp', 'rain', 'humidity', 'precip', 'snow']

    return {
        'has_price': any(any(kw in col for kw in price_keywords) for col in all_cols_lower),
        'has_promo': any(any(kw in col for kw in promo_keywords) for col in all_cols_lower),
        'has_holiday': any(any(kw in col for kw in holiday_keywords) for col in all_cols_lower),
        'has_weather': any(any(kw in col for kw in weather_keywords) for col in all_cols_lower),
        'n_numeric': len(numeric_features) if numeric_features else 0,
        'n_categorical': len(categorical_features) if categorical_features else 0,
    }


def determine_primary_pattern(pattern_distribution: Dict[str, float]) -> DemandPatternType:
    """Determine primary demand pattern from distribution."""
    if not pattern_distribution:
        return DemandPatternType.MIXED

    dominant = max(pattern_distribution, key=pattern_distribution.get)
    dominant_pct = pattern_distribution.get(dominant, 0)

    if dominant_pct > 0.5:
        return DemandPatternType[dominant.upper()]
    return DemandPatternType.MIXED


def determine_volume_category(
    avg_demand: float,
    zero_fraction: float,
    pct_high: float,
) -> VolumeCategory:
    """Determine overall volume category."""
    if zero_fraction > 0.5:
        return VolumeCategory.LOW_VOLUME
    elif pct_high > 0.4 and zero_fraction < 0.2:
        return VolumeCategory.HIGH_VOLUME
    else:
        return VolumeCategory.MEDIUM_VOLUME


def determine_temporal_characteristic(
    trend_strength: float,
    seasonal_strength: float,
    pct_non_stationary: float,
) -> TemporalCharacteristic:
    """Determine dominant temporal characteristic."""
    if trend_strength > 0.7:
        return TemporalCharacteristic.STRONGLY_TRENDING
    elif trend_strength > 0.4:
        return TemporalCharacteristic.MODERATELY_TRENDING
    elif seasonal_strength > 0.7:
        return TemporalCharacteristic.STRONGLY_SEASONAL
    elif seasonal_strength > 0.4:
        return TemporalCharacteristic.MODERATELY_SEASONAL
    elif pct_non_stationary < 0.3:
        return TemporalCharacteristic.STATIONARY
    else:
        return TemporalCharacteristic.MIXED_TEMPORAL


# =============================================================================
# MAIN PROFILING FUNCTION
# =============================================================================

def profile_data(
    df: pd.DataFrame,
    key_columns: List[str],
    date_col: str,
    target_col: str,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    temporal_sample_size: int = 100,
) -> DataProfile:
    """
    Profile data comprehensively for adaptive EDA.

    This is the MAIN ENTRY POINT for data profiling. It characterizes the data
    WITHOUT limiting which analyses run - all applicable analyses should still
    be executed. The profile informs recommendations, not analysis selection.

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
    temporal_sample_size : int
        Sample size for temporal analysis (for efficiency)

    Returns
    -------
    DataProfile
        Comprehensive data profile
    """
    logger.info("=" * 60)
    logger.info("DATA PROFILING: Analyzing data characteristics")
    logger.info("=" * 60)

    profile = DataProfile()

    # METADATA
    profile.total_rows = len(df)
    profile.total_series = df.groupby(key_columns).ngroups
    profile.n_time_periods = df[date_col].nunique()
    profile.date_range = {
        'start': str(df[date_col].min()),
        'end': str(df[date_col].max()),
    }

    logger.info(f"Data: {profile.total_rows:,} rows, {profile.total_series:,} series, "
                f"{profile.n_time_periods} periods")

    # TIME GRANULARITY
    profile.time_granularity, profile.detected_period = detect_time_granularity(df, date_col)
    logger.info(f"Time granularity: {profile.time_granularity.value} (period={profile.detected_period})")

    # HIERARCHY STRUCTURE
    profile.hierarchy_type, hierarchy_info = detect_hierarchy_structure(df, key_columns)
    profile.hierarchy_levels = key_columns
    profile.n_hierarchy_levels = len(key_columns)
    profile.series_per_level = hierarchy_info.get('n_unique', {})
    logger.info(f"Hierarchy: {profile.hierarchy_type.value} ({profile.n_hierarchy_levels} levels)")

    # SERIES STATISTICS
    logger.info("Computing per-series statistics...")
    series_stats = compute_series_statistics(df, key_columns, target_col)

    # Pattern distribution
    pattern_counts = series_stats['pattern'].value_counts(normalize=True)
    profile.pattern_distribution = pattern_counts.to_dict()
    profile.pct_smooth = profile.pattern_distribution.get('smooth', 0)
    profile.pct_erratic = profile.pattern_distribution.get('erratic', 0)
    profile.pct_intermittent = profile.pattern_distribution.get('intermittent', 0)
    profile.pct_lumpy = profile.pattern_distribution.get('lumpy', 0)
    profile.primary_demand_pattern = determine_primary_pattern(profile.pattern_distribution)

    logger.info(f"Pattern distribution: Smooth {profile.pct_smooth*100:.1f}%, "
                f"Erratic {profile.pct_erratic*100:.1f}%, "
                f"Intermittent {profile.pct_intermittent*100:.1f}%, "
                f"Lumpy {profile.pct_lumpy*100:.1f}%")

    # Intermittency metrics
    profile.avg_zero_fraction = series_stats['zero_fraction'].mean()
    profile.median_zero_fraction = series_stats['zero_fraction'].median()
    profile.avg_adi = series_stats['adi'].mean()
    profile.avg_cv2 = series_stats['cv2'].mean()
    profile.avg_cv = series_stats['cv'].mean()

    # VOLUME ANALYSIS
    profile.avg_demand_per_period = series_stats['mean'].mean()
    profile.median_demand_per_period = series_stats['mean'].median()
    profile.total_demand_volume = series_stats['sum'].sum()
    profile.avg_series_length = series_stats['count'].mean()

    volume_counts = series_stats['volume_tier'].value_counts(normalize=True)
    profile.pct_high_volume = volume_counts.get('high', 0)
    profile.pct_medium_volume = volume_counts.get('medium', 0)
    profile.pct_low_volume = volume_counts.get('low', 0)
    profile.volume_category = determine_volume_category(
        profile.avg_demand_per_period, profile.avg_zero_fraction, profile.pct_high_volume
    )

    # TEMPORAL CHARACTERISTICS
    logger.info(f"Computing temporal characteristics (sampling {temporal_sample_size} series)...")
    temporal_stats = compute_temporal_characteristics_sample(
        df, key_columns, target_col, date_col,
        period=profile.detected_period, sample_size=temporal_sample_size
    )
    profile.avg_trend_strength = temporal_stats['avg_trend_strength']
    profile.avg_seasonal_strength = temporal_stats['avg_seasonal_strength']
    profile.pct_stationary = temporal_stats['pct_stationary']
    profile.pct_non_stationary = 1 - profile.pct_stationary
    profile.temporal_characteristic = determine_temporal_characteristic(
        profile.avg_trend_strength, profile.avg_seasonal_strength, profile.pct_non_stationary
    )

    # FEATURE ANALYSIS
    feature_info = detect_feature_presence(df, numeric_features, categorical_features)
    profile.has_price_features = feature_info['has_price']
    profile.has_promo_features = feature_info['has_promo']
    profile.has_holiday_features = feature_info['has_holiday']
    profile.has_weather_features = feature_info['has_weather']
    profile.n_numeric_features = feature_info['n_numeric']
    profile.n_categorical_features = feature_info['n_categorical']
    profile.numeric_features = numeric_features or []
    profile.categorical_features = categorical_features or []

    # DATA QUALITY
    profile.missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    profile.duplicate_pct = df.duplicated().sum() / len(df) * 100
    profile.has_negatives = (df[target_col] < 0).any()
    profile.quality_score = 100
    if profile.missing_pct > 10:
        profile.quality_score -= 15
        profile.quality_flags.append(f"High missing data: {profile.missing_pct:.1f}%")
    if profile.duplicate_pct > 5:
        profile.quality_score -= 10
        profile.quality_flags.append(f"High duplicates: {profile.duplicate_pct:.1f}%")
    if profile.has_negatives:
        profile.quality_score -= 10
        profile.quality_flags.append("Target contains negative values")
    profile.quality_score = max(0, min(100, profile.quality_score))

    logger.info("=" * 60)
    logger.info("DATA PROFILING COMPLETE")
    logger.info("=" * 60)

    return profile


def save_profile(profile: DataProfile, output_path: str) -> None:
    """Save data profile to JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(profile.to_dict(), f, indent=2, default=str)
    logger.info(f"Profile saved to: {output_path}")
