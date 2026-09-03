"""
EDA Recommendations Generator
==============================

This module generates SMART, ACTIONABLE recommendations for downstream crews
based on comprehensive EDA results. The philosophy is:

1. ALL ANALYSES ARE RUN - We don't pre-filter
2. RECOMMENDATIONS ARE THE SMART FILTER - Based on what the data actually shows
3. PATTERN-SPECIFIC GUIDANCE - Different advice for different segments
4. ACTIONABLE OUTPUT - Directly usable by segmentation, feature engineering, and training crews

Key outputs:
- Segmentation recommendations: clustering features, algorithm, number of clusters
- Feature engineering recommendations: lags, rolling windows, transformations
- Training recommendations: model families, loss functions, evaluation metrics

Usage:
    from utils.eda_recommendations import generate_all_recommendations
    from utils.data_profiler import profile_data

    profile = profile_data(df, key_columns, date_col, target_col)
    eda_results = run_eda_pipeline(...)

    recommendations = generate_all_recommendations(profile, eda_results)
    save_recommendations(recommendations, output_dir)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.data_profiler import DataProfile, TimeGranularity, DemandPatternType

logger = logging.getLogger(__name__)


# =============================================================================
# RECOMMENDATION DATACLASSES
# =============================================================================

@dataclass
class SegmentationRecommendations:
    """Recommendations for the Segmentation Crew."""
    # Clustering approach
    recommended_algorithm: str = "GaussianMixture"
    algorithm_rationale: str = ""

    # Features for clustering
    primary_features: List[str] = field(default_factory=list)
    secondary_features: List[str] = field(default_factory=list)
    feature_rationale: str = ""

    # Number of clusters
    n_clusters_min: int = 3
    n_clusters_max: int = 7
    n_clusters_recommended: int = 5
    cluster_rationale: str = ""

    # Multi-dimensional clustering dimensions
    clustering_dimensions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-pattern model guidance (for ALL patterns, since data is mixed)
    per_pattern_guidance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Key insights that drive segmentation
    key_insights: List[str] = field(default_factory=list)

    # Warnings/considerations
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'clustering_config': {
                'algorithm': self.recommended_algorithm,
                'algorithm_rationale': self.algorithm_rationale,
                'n_clusters': {
                    'min': self.n_clusters_min,
                    'max': self.n_clusters_max,
                    'recommended': self.n_clusters_recommended,
                    'rationale': self.cluster_rationale,
                },
            },
            'features': {
                'primary': self.primary_features,
                'secondary': self.secondary_features,
                'rationale': self.feature_rationale,
            },
            'clustering_dimensions': self.clustering_dimensions,
            'per_pattern_guidance': self.per_pattern_guidance,
            'key_insights': self.key_insights,
            'warnings': self.warnings,
        }


@dataclass
class FeatureEngineeringRecommendations:
    """Recommendations for the Feature Engineering Crew."""
    # Lag configuration
    target_lags: List[int] = field(default_factory=list)
    feature_lags: List[int] = field(default_factory=list)
    seasonal_period: int = 52
    lag_rationale: str = ""

    # Rolling window configuration
    rolling_windows: List[int] = field(default_factory=list)
    rolling_statistics: List[str] = field(default_factory=list)
    rolling_rationale: str = ""

    # Transformations
    transformations: List[str] = field(default_factory=list)
    transformation_rationale: str = ""

    # Intermittency features (if needed)
    intermittency_features_needed: bool = False
    intermittency_features: List[str] = field(default_factory=list)
    intermittency_rationale: str = ""

    # Trend/Seasonal features
    trend_features_needed: bool = False
    trend_features: List[str] = field(default_factory=list)
    seasonal_features_needed: bool = False
    seasonal_features: List[str] = field(default_factory=list)
    temporal_rationale: str = ""

    # Calendar features based on granularity
    calendar_features: List[str] = field(default_factory=list)

    # Interaction features
    interaction_features: List[str] = field(default_factory=list)
    interaction_rationale: str = ""

    # Feature importance (from EDA)
    top_features_by_importance: List[str] = field(default_factory=list)

    # Key insights
    key_insights: List[str] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'lag_config': {
                'target_lags': self.target_lags,
                'feature_lags': self.feature_lags,
                'seasonal_period': self.seasonal_period,
                'rationale': self.lag_rationale,
            },
            'rolling_config': {
                'windows': self.rolling_windows,
                'statistics': self.rolling_statistics,
                'rationale': self.rolling_rationale,
            },
            'transformations': {
                'recommended': self.transformations,
                'rationale': self.transformation_rationale,
            },
            'intermittency_features': {
                'needed': self.intermittency_features_needed,
                'features': self.intermittency_features,
                'rationale': self.intermittency_rationale,
            },
            'temporal_features': {
                'trend_needed': self.trend_features_needed,
                'trend_features': self.trend_features,
                'seasonal_needed': self.seasonal_features_needed,
                'seasonal_features': self.seasonal_features,
                'rationale': self.temporal_rationale,
            },
            'calendar_features': self.calendar_features,
            'interaction_features': {
                'features': self.interaction_features,
                'rationale': self.interaction_rationale,
            },
            'top_features_by_importance': self.top_features_by_importance,
            'key_insights': self.key_insights,
            'warnings': self.warnings,
        }


@dataclass
class TrainingRecommendations:
    """Recommendations for the Training Crew."""
    # Model families (feature-based only - univariate banned)
    recommended_model_families: List[str] = field(default_factory=list)
    model_rationale: str = ""

    # Loss function
    recommended_loss: str = "mse"
    loss_rationale: str = ""

    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=list)
    primary_metric: str = "wape"
    metric_rationale: str = ""

    # Cross-validation strategy
    cv_strategy: str = "time_series_split"
    cv_rationale: str = ""

    # Per-segment model recommendations
    per_segment_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Hyperparameter guidance
    hyperparameter_guidance: Dict[str, Any] = field(default_factory=dict)

    # Sample weighting
    use_sample_weights: bool = False
    sample_weight_strategy: str = ""
    sample_weight_rationale: str = ""

    # Expected performance by segment
    expected_performance: Dict[str, str] = field(default_factory=dict)

    # Key insights
    key_insights: List[str] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_config': {
                'recommended_families': self.recommended_model_families,
                'rationale': self.model_rationale,
            },
            'loss_config': {
                'recommended': self.recommended_loss,
                'rationale': self.loss_rationale,
            },
            'evaluation_config': {
                'metrics': self.evaluation_metrics,
                'primary_metric': self.primary_metric,
                'rationale': self.metric_rationale,
            },
            'cv_config': {
                'strategy': self.cv_strategy,
                'rationale': self.cv_rationale,
            },
            'per_segment_models': self.per_segment_models,
            'hyperparameter_guidance': self.hyperparameter_guidance,
            'sample_weighting': {
                'use_weights': self.use_sample_weights,
                'strategy': self.sample_weight_strategy,
                'rationale': self.sample_weight_rationale,
            },
            'expected_performance': self.expected_performance,
            'key_insights': self.key_insights,
            'warnings': self.warnings,
        }


@dataclass
class AllRecommendations:
    """Container for all recommendations."""
    segmentation: SegmentationRecommendations = field(default_factory=SegmentationRecommendations)
    feature_engineering: FeatureEngineeringRecommendations = field(default_factory=FeatureEngineeringRecommendations)
    training: TrainingRecommendations = field(default_factory=TrainingRecommendations)

    # Overall key insights (top-level summary)
    executive_summary: List[str] = field(default_factory=list)

    # Data profile summary
    data_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'executive_summary': self.executive_summary,
            'data_summary': self.data_summary,
            'segmentation': self.segmentation.to_dict(),
            'feature_engineering': self.feature_engineering.to_dict(),
            'training': self.training.to_dict(),
        }


# =============================================================================
# RECOMMENDATION GENERATORS
# =============================================================================

def generate_segmentation_recommendations(
    profile: DataProfile,
    eda_results: Optional[Dict[str, Any]] = None,
) -> SegmentationRecommendations:
    """
    Generate segmentation recommendations based on data profile and EDA results.

    These recommendations are used by the Segmentation Crew to:
    1. Select clustering algorithm
    2. Choose features for clustering
    3. Determine number of clusters
    4. Map segments to model families
    """
    rec = SegmentationRecommendations()

    # =========================================================================
    # CLUSTERING ALGORITHM SELECTION
    # =========================================================================
    pct_intermittent_lumpy = profile.pct_intermittent + profile.pct_lumpy

    if pct_intermittent_lumpy > 0.4:
        rec.recommended_algorithm = "GaussianMixture"
        rec.algorithm_rationale = (
            f"GMM recommended: {pct_intermittent_lumpy*100:.1f}% of series are intermittent/lumpy. "
            "GMM handles overlapping clusters better than K-Means for non-spherical distributions."
        )
    elif profile.avg_cv > 1.5:
        rec.recommended_algorithm = "GaussianMixture"
        rec.algorithm_rationale = (
            f"GMM recommended: High variability (CV={profile.avg_cv:.2f}) suggests heterogeneous patterns. "
            "GMM's soft clustering captures uncertainty in segment boundaries."
        )
    else:
        rec.recommended_algorithm = "GaussianMixture"
        rec.algorithm_rationale = (
            "GMM recommended as default: Provides soft cluster assignments and handles varied cluster shapes."
        )

    # =========================================================================
    # CLUSTERING FEATURES
    # =========================================================================
    # Primary features - ALWAYS use these
    rec.primary_features = ['mean', 'cv', 'adi', 'zero_fraction']
    rec.feature_rationale = (
        "Primary features capture volume (mean), variability (cv), intermittency (adi), "
        "and sparsity (zero_fraction) - the four key dimensions of demand behavior."
    )

    # Secondary features based on what analysis revealed
    secondary = []
    if profile.avg_trend_strength > 0.3:
        secondary.append('trend_strength')
    if profile.avg_seasonal_strength > 0.3:
        secondary.append('seasonal_strength')
    secondary.extend(['autocorr_lag1', 'skewness', 'predictability_score'])
    rec.secondary_features = secondary

    # =========================================================================
    # CLUSTERING DIMENSIONS (Multi-dimensional approach)
    # =========================================================================
    rec.clustering_dimensions = {
        'volume_dimension': {
            'features': ['mean', 'sum', 'demand_frequency'],
            'purpose': 'Separate high vs low volume series',
            'weight': 'HIGH' if profile.pct_high_volume > 0.3 else 'MEDIUM',
        },
        'intermittency_dimension': {
            'features': ['adi', 'cv2', 'zero_fraction'],
            'purpose': 'Syntetos-Boylan classification',
            'weight': 'HIGH' if pct_intermittent_lumpy > 0.3 else 'MEDIUM',
        },
        'variability_dimension': {
            'features': ['cv', 'std', 'range_normalized'],
            'purpose': 'Stable vs variable demand',
            'weight': 'HIGH' if profile.avg_cv > 1.0 else 'MEDIUM',
        },
        'temporal_dimension': {
            'features': ['trend_strength', 'seasonal_strength', 'autocorr_lag1'],
            'purpose': 'Trending vs stationary patterns',
            'weight': 'HIGH' if profile.pct_non_stationary > 0.4 else 'LOW',
        },
    }

    # =========================================================================
    # NUMBER OF CLUSTERS
    # =========================================================================
    # Base on heterogeneity
    if profile.primary_demand_pattern == DemandPatternType.MIXED:
        rec.n_clusters_recommended = 6
        rec.cluster_rationale = (
            "Mixed demand patterns detected - 6 clusters recommended to capture "
            "heterogeneity across smooth, erratic, intermittent, and lumpy patterns."
        )
    elif profile.avg_cv > 1.5:
        rec.n_clusters_recommended = 6
        rec.cluster_rationale = (
            f"High variability (CV={profile.avg_cv:.2f}) suggests heterogeneous demand - "
            "6 clusters for fine-grained segmentation."
        )
    elif pct_intermittent_lumpy > 0.5:
        rec.n_clusters_recommended = 5
        rec.cluster_rationale = (
            f"{pct_intermittent_lumpy*100:.1f}% intermittent/lumpy - "
            "5 clusters to separate by intermittency severity."
        )
    else:
        rec.n_clusters_recommended = 4
        rec.cluster_rationale = (
            "Relatively homogeneous demand - 4 clusters sufficient for practical segmentation."
        )

    rec.n_clusters_min = 3
    rec.n_clusters_max = 8

    # =========================================================================
    # PER-PATTERN MODEL GUIDANCE
    # =========================================================================
    rec.per_pattern_guidance = {
        'smooth': {
            'description': f'{profile.pct_smooth*100:.1f}% of series',
            'characteristics': 'Low ADI (<1.32), Low CV² (<0.49) - Regular, predictable demand',
            'recommended_models': ['lightgbm', 'xgboost', 'catboost', 'random_forest'],
            'expected_wape': '15-30%',
            'feature_priority': 'Standard lags, rolling means, calendar features',
            'loss_function': 'mse',
        },
        'erratic': {
            'description': f'{profile.pct_erratic*100:.1f}% of series',
            'characteristics': 'Low ADI (<1.32), High CV² (>=0.49) - Frequent but highly variable',
            'recommended_models': ['xgboost', 'lightgbm', 'catboost'],
            'expected_wape': '30-50%',
            'feature_priority': 'Volatility features, robust rolling stats, log transform',
            'loss_function': 'huber',
        },
        'intermittent': {
            'description': f'{profile.pct_intermittent*100:.1f}% of series',
            'characteristics': 'High ADI (>=1.32), Low CV² (<0.49) - Sporadic but consistent size',
            'recommended_models': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model'],
            'expected_wape': '40-60%',
            'feature_priority': 'Intermittency features, demand occurrence indicators',
            'loss_function': 'tweedie',
        },
        'lumpy': {
            'description': f'{profile.pct_lumpy*100:.1f}% of series',
            'characteristics': 'High ADI (>=1.32), High CV² (>=0.49) - Sporadic AND variable',
            'recommended_models': ['lightgbm', 'xgboost', 'zero_inflated', 'tweedie'],
            'expected_wape': '50-80%',
            'feature_priority': 'Occurrence + quantity features, bootstrap for uncertainty',
            'loss_function': 'tweedie',
        },
    }

    # =========================================================================
    # KEY INSIGHTS
    # =========================================================================
    insights = []

    if pct_intermittent_lumpy > 0.5:
        insights.append(
            f"⚠️ INTERMITTENT DEMAND DOMINANT: {pct_intermittent_lumpy*100:.1f}% of series. "
            "Prioritize segments by intermittency severity for model selection."
        )

    if profile.pct_smooth > 0.5:
        insights.append(
            f"✓ SMOOTH DEMAND DOMINANT: {profile.pct_smooth*100:.1f}% of series. "
            "Standard forecasting approaches will work well for majority."
        )

    if profile.primary_demand_pattern == DemandPatternType.MIXED:
        insights.append(
            "⚠️ MIXED PATTERNS: No single pattern dominates. "
            "Segment-specific modeling is CRITICAL for good performance."
        )

    if profile.avg_zero_fraction > 0.3:
        insights.append(
            f"⚠️ HIGH ZERO PREVALENCE: {profile.avg_zero_fraction*100:.1f}% zeros on average. "
            "Consider two-stage models or zero-inflated approaches for affected segments."
        )

    rec.key_insights = insights

    # =========================================================================
    # WARNINGS
    # =========================================================================
    warnings = []
    if profile.n_hierarchy_levels > 1:
        warnings.append(
            f"Multi-level hierarchy detected ({profile.n_hierarchy_levels} levels). "
            "Consider hierarchical reconciliation after segment-level forecasting."
        )

    if profile.total_series < 100:
        warnings.append(
            f"Small number of series ({profile.total_series}). "
            "Consider reducing number of clusters to ensure statistical robustness."
        )

    rec.warnings = warnings

    return rec


def generate_feature_engineering_recommendations(
    profile: DataProfile,
    eda_results: Optional[Dict[str, Any]] = None,
) -> FeatureEngineeringRecommendations:
    """
    Generate feature engineering recommendations based on data profile and EDA results.

    These recommendations are used by the Feature Engineering Crew to:
    1. Create appropriate lags
    2. Build rolling window features
    3. Apply transformations
    4. Add intermittency/temporal features
    """
    rec = FeatureEngineeringRecommendations()

    # =========================================================================
    # LAG CONFIGURATION (based on granularity)
    # =========================================================================
    if profile.time_granularity == TimeGranularity.DAILY:
        rec.target_lags = [1, 2, 3, 7, 14, 21, 28]
        rec.feature_lags = [1, 7, 28]
        rec.seasonal_period = 7
        rec.lag_rationale = (
            "Daily data: Include day-of-week patterns (lag 7), bi-weekly (lag 14), "
            "and monthly patterns (lag 28)."
        )
    elif profile.time_granularity == TimeGranularity.WEEKLY:
        rec.target_lags = [1, 2, 4, 13, 26, 52]
        rec.feature_lags = [1, 4, 52]
        rec.seasonal_period = 52
        rec.lag_rationale = (
            "Weekly data: Include weekly momentum (lag 1-2), monthly (lag 4), "
            "quarterly (lag 13), semi-annual (lag 26), and year-over-year (lag 52)."
        )
    elif profile.time_granularity == TimeGranularity.MONTHLY:
        rec.target_lags = [1, 2, 3, 6, 12]
        rec.feature_lags = [1, 3, 12]
        rec.seasonal_period = 12
        rec.lag_rationale = (
            "Monthly data: Include recent momentum (lag 1-3), semi-annual (lag 6), "
            "and year-over-year (lag 12)."
        )
    else:
        rec.target_lags = [1, 2, 4]
        rec.feature_lags = [1, 4]
        rec.seasonal_period = 4
        rec.lag_rationale = "Quarterly/Yearly data: Focus on year-over-year patterns."

    # =========================================================================
    # ROLLING WINDOW CONFIGURATION
    # =========================================================================
    if profile.time_granularity == TimeGranularity.DAILY:
        rec.rolling_windows = [7, 14, 28, 56]
    elif profile.time_granularity == TimeGranularity.WEEKLY:
        rec.rolling_windows = [4, 13, 26, 52]
    elif profile.time_granularity == TimeGranularity.MONTHLY:
        rec.rolling_windows = [3, 6, 12]
    else:
        rec.rolling_windows = [2, 4]

    rec.rolling_statistics = ['mean', 'std', 'min', 'max']
    rec.rolling_rationale = (
        f"Rolling windows capture short, medium, and long-term patterns. "
        f"Windows: {rec.rolling_windows} (periods: {profile.time_granularity.value})."
    )

    # =========================================================================
    # TRANSFORMATIONS
    # =========================================================================
    transformations = []
    rationale_parts = []

    # Log transform for high variability
    if profile.avg_cv > 1.5:
        transformations.append('log_transform')
        rationale_parts.append(f"Log transform: High CV ({profile.avg_cv:.2f}) indicates skewed distributions")

    # Differencing for non-stationarity
    if profile.pct_non_stationary > 0.3:
        transformations.append('differencing')
        rationale_parts.append(f"Differencing: {profile.pct_non_stationary*100:.1f}% series non-stationary")

    # Robust scaling for outliers
    if profile.avg_cv > 1.0:
        transformations.append('robust_scaling')
        rationale_parts.append("Robust scaling: Handle outliers in high-variability data")

    rec.transformations = transformations if transformations else ['none_needed']
    rec.transformation_rationale = "; ".join(rationale_parts) if rationale_parts else "No transformations needed"

    # =========================================================================
    # INTERMITTENCY FEATURES
    # =========================================================================
    pct_intermittent_lumpy = profile.pct_intermittent + profile.pct_lumpy

    if profile.avg_zero_fraction > 0.1:
        rec.intermittency_features_needed = True
        rec.intermittency_features = [
            'time_since_last_nonzero',
            'demand_occurrence_rate',
            'cumsum_nonzero',
            'rolling_nonzero_count',
            'avg_demand_when_positive',
            'demand_recency',
        ]
        rec.intermittency_rationale = (
            f"Intermittency features REQUIRED: {profile.avg_zero_fraction*100:.1f}% zeros on average, "
            f"{pct_intermittent_lumpy*100:.1f}% series are intermittent/lumpy. "
            "These features capture demand occurrence patterns critical for sparse data."
        )
    else:
        rec.intermittency_features_needed = False
        rec.intermittency_rationale = (
            f"Intermittency features optional: Only {profile.avg_zero_fraction*100:.1f}% zeros. "
            "Standard lag features sufficient."
        )

    # =========================================================================
    # TREND/SEASONAL FEATURES
    # =========================================================================
    if profile.avg_trend_strength > 0.3:
        rec.trend_features_needed = True
        rec.trend_features = ['trend_slope', 'detrended_value', 'trend_direction', 'trend_strength_local']
    else:
        rec.trend_features_needed = False
        rec.trend_features = []

    if profile.avg_seasonal_strength > 0.3 or profile.time_granularity in [TimeGranularity.WEEKLY, TimeGranularity.MONTHLY]:
        rec.seasonal_features_needed = True
        rec.seasonal_features = ['seasonal_index', 'deseasonalized_value', 'period_of_year']
    else:
        rec.seasonal_features_needed = False
        rec.seasonal_features = []

    rec.temporal_rationale = (
        f"Trend strength: {profile.avg_trend_strength:.2f} ({'INCLUDE trend features' if rec.trend_features_needed else 'optional'}). "
        f"Seasonal strength: {profile.avg_seasonal_strength:.2f} ({'INCLUDE seasonal features' if rec.seasonal_features_needed else 'optional'})."
    )

    # =========================================================================
    # CALENDAR FEATURES (based on granularity)
    # =========================================================================
    if profile.time_granularity == TimeGranularity.DAILY:
        rec.calendar_features = [
            'day_of_week', 'is_weekend', 'day_of_month', 'week_of_year',
            'month', 'is_month_start', 'is_month_end', 'is_quarter_end'
        ]
    elif profile.time_granularity == TimeGranularity.WEEKLY:
        rec.calendar_features = [
            'week_of_year', 'month', 'quarter', 'year',
            'is_year_start', 'is_year_end', 'weeks_until_year_end'
        ]
    elif profile.time_granularity == TimeGranularity.MONTHLY:
        rec.calendar_features = [
            'month', 'quarter', 'year', 'is_year_end', 'is_quarter_end'
        ]
    else:
        rec.calendar_features = ['quarter', 'year']

    # =========================================================================
    # INTERACTION FEATURES
    # =========================================================================
    interactions = []
    interaction_rationale = []

    if profile.has_price_features and profile.has_promo_features:
        interactions.append('price_promo_interaction')
        interaction_rationale.append("Price × Promo: Capture promotion price sensitivity")

    if profile.has_price_features:
        interactions.extend(['price_lag_ratio', 'price_change_indicator'])
        interaction_rationale.append("Price dynamics: Capture price changes over time")

    if profile.has_promo_features:
        interactions.append('promo_lag_interaction')
        interaction_rationale.append("Promo carryover: Capture promotion aftereffects")

    rec.interaction_features = interactions
    rec.interaction_rationale = "; ".join(interaction_rationale) if interaction_rationale else "No interactions needed"

    # =========================================================================
    # TOP FEATURES FROM EDA
    # =========================================================================
    if eda_results and 'feature_importance' in eda_results:
        top_features = eda_results['feature_importance']
        if isinstance(top_features, list):
            rec.top_features_by_importance = top_features[:20]
        elif isinstance(top_features, dict):
            rec.top_features_by_importance = list(top_features.keys())[:20]

    # =========================================================================
    # KEY INSIGHTS
    # =========================================================================
    insights = []

    if rec.intermittency_features_needed:
        insights.append(
            f"🔴 CRITICAL: Intermittency features are REQUIRED for {pct_intermittent_lumpy*100:.1f}% of series. "
            "Include time_since_last_nonzero, demand_occurrence_rate in feature set."
        )

    if profile.avg_cv > 1.5:
        insights.append(
            f"⚠️ HIGH VARIABILITY: CV={profile.avg_cv:.2f}. Apply log transform before creating lag features."
        )

    if profile.pct_non_stationary > 0.4:
        insights.append(
            f"⚠️ NON-STATIONARITY: {profile.pct_non_stationary*100:.1f}% series need differencing or trend features."
        )

    if profile.has_price_features:
        insights.append("✓ Price features available - include price elasticity features.")

    if profile.has_promo_features:
        insights.append("✓ Promo features available - include promotion uplift features.")

    rec.key_insights = insights

    # =========================================================================
    # WARNINGS
    # =========================================================================
    warnings = []

    if profile.avg_series_length < 52:
        warnings.append(
            f"Short series (avg {profile.avg_series_length:.0f} periods). "
            "Reduce max lag to avoid excessive NaN from lag features."
        )

    if profile.missing_pct > 5:
        warnings.append(
            f"Missing data ({profile.missing_pct:.1f}%). "
            "Ensure proper imputation before creating lag/rolling features."
        )

    rec.warnings = warnings

    return rec


def generate_training_recommendations(
    profile: DataProfile,
    eda_results: Optional[Dict[str, Any]] = None,
) -> TrainingRecommendations:
    """
    Generate model training recommendations based on data profile and EDA results.

    These recommendations are used by the Training Crew to:
    1. Select model families
    2. Choose loss functions
    3. Set up evaluation
    4. Configure per-segment training
    """
    rec = TrainingRecommendations()

    pct_intermittent_lumpy = profile.pct_intermittent + profile.pct_lumpy

    # =========================================================================
    # MODEL FAMILIES (Feature-based only - univariate BANNED)
    # =========================================================================
    # Base models for all cases
    base_models = ['lightgbm', 'xgboost', 'catboost']

    if pct_intermittent_lumpy > 0.3:
        rec.recommended_model_families = base_models + ['zero_inflated', 'tweedie', 'hurdle_model']
        rec.model_rationale = (
            f"Intermittent-aware models needed: {pct_intermittent_lumpy*100:.1f}% series are intermittent/lumpy. "
            "Include zero_inflated and tweedie regression alongside tree-based models."
        )
    elif profile.avg_cv > 1.5:
        rec.recommended_model_families = base_models
        rec.model_rationale = (
            f"High variability (CV={profile.avg_cv:.2f}): Tree-based models with robust loss functions. "
            "LightGBM and XGBoost handle outliers well with Huber loss."
        )
    else:
        rec.recommended_model_families = base_models + ['random_forest']
        rec.model_rationale = (
            "Standard tree-based models work well for smooth demand patterns."
        )

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================
    if pct_intermittent_lumpy > 0.3:
        rec.recommended_loss = 'tweedie'
        rec.loss_rationale = (
            f"Tweedie loss for zero-inflated data: {pct_intermittent_lumpy*100:.1f}% intermittent/lumpy series. "
            "Tweedie handles zeros naturally without log(0) issues."
        )
    elif profile.avg_cv > 1.5:
        rec.recommended_loss = 'huber'
        rec.loss_rationale = (
            f"Huber loss for robustness: High CV ({profile.avg_cv:.2f}) indicates outliers. "
            "Huber provides smooth transition between L1 and L2 loss."
        )
    else:
        rec.recommended_loss = 'mse'
        rec.loss_rationale = "MSE loss for smooth demand patterns."

    # =========================================================================
    # EVALUATION METRICS
    # =========================================================================
    rec.evaluation_metrics = ['wape', 'mape', 'rmse', 'bias', 'mae']
    rec.primary_metric = 'wape'

    if pct_intermittent_lumpy > 0.3:
        rec.evaluation_metrics.extend(['zero_accuracy', 'conditional_wape', 'mase'])
        rec.metric_rationale = (
            "Include zero_accuracy and conditional_wape for intermittent series evaluation. "
            "WAPE as primary metric (robust to zeros), supplemented by segment-specific metrics."
        )
    else:
        rec.metric_rationale = (
            "WAPE as primary metric (scale-independent). RMSE for sensitivity to large errors."
        )

    # =========================================================================
    # CROSS-VALIDATION
    # =========================================================================
    rec.cv_strategy = 'time_series_split'
    rec.cv_rationale = (
        "Time series split preserves temporal ordering. Use expanding window with "
        f"{profile.detected_period // 4} period gap between train and validation."
    )

    # =========================================================================
    # PER-SEGMENT MODELS
    # =========================================================================
    rec.per_segment_models = {
        'smooth_segments': {
            'models': ['lightgbm', 'xgboost', 'catboost', 'random_forest'],
            'loss': 'mse',
            'expected_wape': '15-30%',
            'notes': f'~{profile.pct_smooth*100:.0f}% of series',
        },
        'erratic_segments': {
            'models': ['xgboost', 'lightgbm', 'catboost'],
            'loss': 'huber',
            'expected_wape': '30-50%',
            'notes': f'~{profile.pct_erratic*100:.0f}% of series - use robust loss',
        },
        'intermittent_segments': {
            'models': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model'],
            'loss': 'tweedie',
            'expected_wape': '40-60%',
            'notes': f'~{profile.pct_intermittent*100:.0f}% of series - consider two-stage approach',
        },
        'lumpy_segments': {
            'models': ['lightgbm', 'xgboost', 'zero_inflated', 'tweedie'],
            'loss': 'tweedie',
            'expected_wape': '50-80%',
            'notes': f'~{profile.pct_lumpy*100:.0f}% of series - hardest to forecast accurately',
        },
    }

    # =========================================================================
    # HYPERPARAMETER GUIDANCE
    # =========================================================================
    rec.hyperparameter_guidance = {
        'lightgbm': {
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'max_depth': [-1, 10, 15],
            'min_data_in_leaf': [20, 50, 100],
            'feature_fraction': [0.7, 0.8, 0.9],
            'notes': 'Start with default, tune learning_rate first',
        },
        'xgboost': {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'notes': 'XGBoost more regularization-focused than LightGBM',
        },
    }

    # =========================================================================
    # SAMPLE WEIGHTING
    # =========================================================================
    if profile.avg_zero_fraction > 0.3:
        rec.use_sample_weights = True
        rec.sample_weight_strategy = 'inverse_frequency'
        rec.sample_weight_rationale = (
            f"Sample weighting recommended: {profile.avg_zero_fraction*100:.1f}% zeros. "
            "Upweight non-zero observations to improve positive demand prediction."
        )
    else:
        rec.use_sample_weights = False
        rec.sample_weight_rationale = "Sample weighting not needed for balanced data."

    # =========================================================================
    # EXPECTED PERFORMANCE
    # =========================================================================
    rec.expected_performance = {
        'overall': f"WAPE 25-45% (blended across segments)",
        'smooth': "WAPE 15-30%",
        'erratic': "WAPE 30-50%",
        'intermittent': "WAPE 40-60%",
        'lumpy': "WAPE 50-80%",
        'notes': (
            f"Based on pattern distribution: "
            f"Smooth {profile.pct_smooth*100:.0f}%, Erratic {profile.pct_erratic*100:.0f}%, "
            f"Intermittent {profile.pct_intermittent*100:.0f}%, Lumpy {profile.pct_lumpy*100:.0f}%"
        ),
    }

    # =========================================================================
    # KEY INSIGHTS
    # =========================================================================
    insights = []

    if pct_intermittent_lumpy > 0.5:
        insights.append(
            f"🔴 MAJORITY INTERMITTENT/LUMPY ({pct_intermittent_lumpy*100:.1f}%): "
            "Expect lower accuracy. Focus on zero_accuracy + conditional_wape, not just WAPE."
        )

    if profile.pct_smooth > 0.5:
        insights.append(
            f"✓ MAJORITY SMOOTH ({profile.pct_smooth*100:.1f}%): "
            "Standard tree-based models should achieve WAPE < 30%."
        )

    if profile.primary_demand_pattern == DemandPatternType.MIXED:
        insights.append(
            "⚠️ MIXED PATTERNS: Train separate models per segment for best performance."
        )

    rec.key_insights = insights

    # =========================================================================
    # WARNINGS
    # =========================================================================
    warnings = []

    warnings.append(
        "🚫 BANNED MODELS: croston, sba, tsb, imapa, arima, ets, theta, prophet (univariate). "
        "Use feature-based models ONLY."
    )

    if profile.total_series < 50:
        warnings.append(
            f"Small dataset ({profile.total_series} series). "
            "Consider pooled training across segments for statistical power."
        )

    rec.warnings = warnings

    return rec


# =============================================================================
# MAIN RECOMMENDATION GENERATOR
# =============================================================================

def generate_all_recommendations(
    profile: DataProfile,
    eda_results: Optional[Dict[str, Any]] = None,
) -> AllRecommendations:
    """
    Generate all recommendations for downstream crews.

    This is the MAIN ENTRY POINT for generating recommendations.

    Parameters
    ----------
    profile : DataProfile
        Data profile from profile_data()
    eda_results : Dict[str, Any], optional
        Results from run_eda_pipeline()

    Returns
    -------
    AllRecommendations
        Comprehensive recommendations for all downstream crews
    """
    logger.info("=" * 60)
    logger.info("GENERATING RECOMMENDATIONS FOR DOWNSTREAM CREWS")
    logger.info("=" * 60)

    rec = AllRecommendations()

    # Generate individual recommendations
    rec.segmentation = generate_segmentation_recommendations(profile, eda_results)
    rec.feature_engineering = generate_feature_engineering_recommendations(profile, eda_results)
    rec.training = generate_training_recommendations(profile, eda_results)

    # Data summary
    rec.data_summary = {
        'total_series': profile.total_series,
        'total_rows': profile.total_rows,
        'time_granularity': profile.time_granularity.value,
        'primary_pattern': profile.primary_demand_pattern.value,
        'pattern_distribution': {
            'smooth': f"{profile.pct_smooth*100:.1f}%",
            'erratic': f"{profile.pct_erratic*100:.1f}%",
            'intermittent': f"{profile.pct_intermittent*100:.1f}%",
            'lumpy': f"{profile.pct_lumpy*100:.1f}%",
        },
        'avg_zero_fraction': f"{profile.avg_zero_fraction*100:.1f}%",
        'avg_cv': f"{profile.avg_cv:.2f}",
        'data_quality_score': profile.quality_score,
    }

    # Executive summary
    pct_intermittent_lumpy = profile.pct_intermittent + profile.pct_lumpy

    summary = []

    summary.append(
        f"📊 DATA OVERVIEW: {profile.total_series:,} series, {profile.total_rows:,} rows, "
        f"{profile.time_granularity.value} granularity"
    )

    if pct_intermittent_lumpy > 0.5:
        summary.append(
            f"⚠️ INTERMITTENT DEMAND DOMINANT: {pct_intermittent_lumpy*100:.1f}% of series are intermittent/lumpy. "
            "Use zero-aware models and intermittency features."
        )
    elif profile.pct_smooth > 0.5:
        summary.append(
            f"✓ SMOOTH DEMAND DOMINANT: {profile.pct_smooth*100:.1f}% of series are smooth. "
            "Standard approaches will work well."
        )
    else:
        summary.append(
            f"🔀 MIXED PATTERNS: Smooth {profile.pct_smooth*100:.0f}%, Erratic {profile.pct_erratic*100:.0f}%, "
            f"Intermittent {profile.pct_intermittent*100:.0f}%, Lumpy {profile.pct_lumpy*100:.0f}%. "
            "Segment-specific modeling recommended."
        )

    if profile.avg_cv > 1.5:
        summary.append(
            f"⚠️ HIGH VARIABILITY: CV={profile.avg_cv:.2f}. Apply log transforms and use robust models."
        )

    if profile.pct_non_stationary > 0.4:
        summary.append(
            f"📈 TRENDING DATA: {profile.pct_non_stationary*100:.1f}% non-stationary. Include trend features."
        )

    rec.executive_summary = summary

    logger.info("Recommendations generated successfully")
    logger.info("=" * 60)

    return rec


def save_recommendations(
    recommendations: AllRecommendations,
    output_dir: str,
) -> Dict[str, str]:
    """
    Save recommendations to JSON files for downstream crews.

    Creates:
    - eda_to_segmentation_context.json
    - eda_to_feature_context.json
    - eda_to_training_context.json
    - eda_recommendations_full.json

    Returns
    -------
    Dict[str, str]
        Paths to created files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Segmentation context
    seg_path = os.path.join(output_dir, 'eda_to_segmentation_context.json')
    with open(seg_path, 'w') as f:
        json.dump({
            'source': 'eda_recommendations_generator',
            'data_summary': recommendations.data_summary,
            **recommendations.segmentation.to_dict()
        }, f, indent=2, default=str)
    paths['segmentation'] = seg_path
    logger.info(f"Saved: {seg_path}")

    # Feature engineering context
    feat_path = os.path.join(output_dir, 'eda_to_feature_context.json')
    with open(feat_path, 'w') as f:
        json.dump({
            'source': 'eda_recommendations_generator',
            'data_summary': recommendations.data_summary,
            **recommendations.feature_engineering.to_dict()
        }, f, indent=2, default=str)
    paths['feature_engineering'] = feat_path
    logger.info(f"Saved: {feat_path}")

    # Training context
    train_path = os.path.join(output_dir, 'eda_to_training_context.json')
    with open(train_path, 'w') as f:
        json.dump({
            'source': 'eda_recommendations_generator',
            'data_summary': recommendations.data_summary,
            **recommendations.training.to_dict()
        }, f, indent=2, default=str)
    paths['training'] = train_path
    logger.info(f"Saved: {train_path}")

    # Full recommendations
    full_path = os.path.join(output_dir, 'eda_recommendations_full.json')
    with open(full_path, 'w') as f:
        json.dump(recommendations.to_dict(), f, indent=2, default=str)
    paths['full'] = full_path
    logger.info(f"Saved: {full_path}")

    return paths
