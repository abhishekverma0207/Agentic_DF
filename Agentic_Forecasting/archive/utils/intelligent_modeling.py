# utils/intelligent_modeling.py
"""
Intelligent Modeling Framework
==============================

This module provides a world-class forecasting decision framework that:

1. INTELLIGENTLY CLASSIFIES KEYS for individual vs segment-level modeling
2. PERFORMS SOPHISTICATED FEATURE SELECTION using multiple techniques
3. ENSURES ONLY FEATURE-BASED MODELS are used (no univariate methods)
4. PROVIDES ADAPTIVE MODEL SELECTION based on data characteristics

Key Philosophy:
- Every model MUST use features (lag, rolling, calendar, external)
- Univariate models (croston, tsb, ets, arima, theta) are EXCLUDED
- Keys with sufficient data get individual models
- Keys with limited data are pooled into segments for shared learning
- Feature selection is dynamic, not static

Author: FEU-Agentic-Forecasting Demand Intelligence
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION: ALLOWED MODELS (FEATURE-BASED ONLY)
# =============================================================================

class ModelType(Enum):
    """Model types - ALL are feature-based."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    ZERO_INFLATED = "zero_inflated"  # Two-stage: occurrence + severity with features
    HURDLE_MODEL = "hurdle_model"    # Hurdle with feature-based components
    TWEEDIE = "tweedie"              # GLM with features


# Models that are ALLOWED - all use features
ALLOWED_MODELS = {
    ModelType.LIGHTGBM.value,
    ModelType.XGBOOST.value,
    ModelType.CATBOOST.value,
    ModelType.RANDOM_FOREST.value,
    ModelType.ZERO_INFLATED.value,
    ModelType.HURDLE_MODEL.value,
    ModelType.TWEEDIE.value,
}

# Models that are BANNED - univariate methods that ignore features
BANNED_MODELS = {
    'croston', 'sba', 'tsb', 'imapa',  # Intermittent specialists (univariate)
    'arima', 'sarima', 'ets', 'theta', 'tbats', 'bsts',  # Statistical (univariate)
    'prophet',  # While prophet can use regressors, it's primarily univariate
    'naive', 'seasonal_naive', 'mean',  # Baseline methods
}

# Model performance ranking (empirical - tree-based models dominate)
MODEL_PRIORITY = [
    'lightgbm',      # Fast, handles categoricals, great with features
    'xgboost',       # Robust, excellent regularization
    'catboost',      # Best categorical handling, slower
    'zero_inflated', # Two-stage for intermittent, uses features
    'hurdle_model',  # Alternative two-stage approach
    'random_forest', # Ensemble baseline
    'tweedie',       # GLM for zeros + continuous
]


# =============================================================================
# KEY MODELING STRATEGY CLASSIFICATION
# =============================================================================

class ModelingStrategy(Enum):
    """How a key should be modeled."""
    INDIVIDUAL = "individual"      # Enough data for standalone model
    SEGMENT_POOLED = "segment"     # Pool with similar keys for shared model
    HIERARCHICAL = "hierarchical"  # Segment model + key-level adjustment


@dataclass
class FeatureCharacteristics:
    """Feature-level characteristics for a key - CRITICAL for modeling decisions."""
    n_features_available: int           # Total features available for this key
    n_features_with_variance: int       # Features with non-zero variance
    feature_completeness: float         # Fraction of non-missing feature values (0-1)
    external_feature_coverage: float    # Fraction of external features available (0-1)
    feature_target_correlation_max: float  # Max absolute correlation with target
    feature_target_correlation_mean: float # Mean absolute correlation with target
    n_lag_features: int                 # Number of lag features available
    n_external_features: int            # Number of external features (price, promo, etc.)
    has_price_features: bool            # Price data available
    has_promo_features: bool            # Promo data available
    has_calendar_features: bool         # Calendar features available


@dataclass
class KeyCharacteristics:
    """Characteristics of a time series key for modeling decisions."""
    key: str
    n_observations: int
    n_nonzero: int
    zero_fraction: float
    cv: float                      # Coefficient of variation
    mean_value: float
    total_volume: float
    trend_strength: float
    seasonal_strength: float
    autocorr_lag1: float
    forecastability_score: float
    data_quality_score: float      # Missing values, outliers

    # NEW: Feature-level characteristics
    feature_chars: Optional[FeatureCharacteristics] = None

    # Time format for adaptive thresholds (weekly vs monthly)
    time_format: str = 'year_week'

    # Computed properties
    @property
    def is_high_volume(self) -> bool:
        """High volume keys have more reliable patterns."""
        return self.total_volume > 1000 and self.mean_value > 10

    @property
    def has_sufficient_history(self) -> bool:
        """Sufficient history: 8 months for monthly, 52 weeks for weekly."""
        threshold = 8 if self.time_format == 'year_month' else 52
        return self.n_observations >= threshold

    @property
    def has_sufficient_nonzero(self) -> bool:
        """Sufficient non-zero obs: 4 for monthly, 20 for weekly."""
        threshold = 4 if self.time_format == 'year_month' else 20
        return self.n_nonzero >= threshold

    @property
    def is_forecastable(self) -> bool:
        """Score > 0.3 indicates learnable patterns."""
        return self.forecastability_score > 0.3

    @property
    def is_stable(self) -> bool:
        """Low CV indicates stable, predictable demand."""
        return self.cv < 1.5

    @property
    def has_rich_features(self) -> bool:
        """Key has rich feature set suitable for individual modeling."""
        if self.feature_chars is None:
            return False
        fc = self.feature_chars
        return (
            fc.n_features_with_variance >= 20 and
            fc.feature_completeness >= 0.8 and
            fc.feature_target_correlation_max >= 0.3
        )

    @property
    def has_external_features(self) -> bool:
        """Key has external features (price, promo) for richer models."""
        if self.feature_chars is None:
            return False
        return self.feature_chars.n_external_features >= 3

    @property
    def feature_quality_score(self) -> float:
        """Composite feature quality score (0-1)."""
        if self.feature_chars is None:
            return 0.5  # Default mid-score if no feature info
        fc = self.feature_chars

        # Components
        completeness_score = fc.feature_completeness
        variance_score = min(fc.n_features_with_variance / 30, 1.0)
        correlation_score = min(fc.feature_target_correlation_max / 0.5, 1.0)
        external_score = fc.external_feature_coverage

        # Weighted combination
        return (
            0.30 * completeness_score +
            0.25 * variance_score +
            0.25 * correlation_score +
            0.20 * external_score
        )


@dataclass
class ModelingDecision:
    """Decision about how to model a key."""
    key: str
    strategy: ModelingStrategy
    model_group: str              # Which group to train with
    recommended_models: List[str]
    feature_priority: str         # 'full', 'reduced', 'minimal'
    n_features_max: int           # Max features to use
    rationale: str
    confidence: float             # 0-1 confidence in decision


def classify_key_modeling_strategy(
    characteristics: KeyCharacteristics,
    segment_id: Optional[int] = None,
    min_obs_individual: int = None,
    min_nonzero_individual: int = None,
    high_volume_threshold: float = 5000,
    min_feature_quality_individual: float = 0.6,
) -> ModelingDecision:
    """
    Classify how a key should be modeled based on its characteristics AND FEATURES.

    Decision Tree (incorporating feature quality):
    1. HIGH VOLUME + SUFFICIENT HISTORY + RICH FEATURES → INDIVIDUAL
    2. MEDIUM DATA + GOOD FEATURES + SOME PATTERNS → HIERARCHICAL
    3. LIMITED DATA / SPARSE / POOR FEATURES → SEGMENT_POOLED

    CRITICAL: Feature quality is now a key factor in the decision:
    - Keys with poor feature coverage should be pooled (can't learn from features individually)
    - Keys with rich features and good correlations are candidates for individual models
    - Feature variance matters - zero-variance features provide no learning signal

    Parameters
    ----------
    characteristics : KeyCharacteristics
        Computed characteristics for this key (including feature_chars)
    segment_id : int, optional
        Pre-assigned segment from clustering
    min_obs_individual : int
        Minimum observations for individual model
    min_nonzero_individual : int
        Minimum non-zero observations
    high_volume_threshold : float
        Volume threshold for high-value classification
    min_feature_quality_individual : float
        Minimum feature quality score for individual modeling (0-1)

    Returns
    -------
    ModelingDecision
        Complete decision about modeling strategy
    """
    key = characteristics.key
    rationale_parts = []

    # Resolve time-format-aware defaults
    tf = getattr(characteristics, 'time_format', 'year_week')
    if min_obs_individual is None:
        min_obs_individual = 12 if tf == 'year_month' else 52
    if min_nonzero_individual is None:
        min_nonzero_individual = 6 if tf == 'year_month' else 20

    # ==========================================================================
    # SCORE COMPONENTS - Now includes FEATURE QUALITY
    # ==========================================================================

    # Time series quality scores
    history_score = min(characteristics.n_observations / min_obs_individual, 1.0)
    nonzero_score = min(characteristics.n_nonzero / min_nonzero_individual, 1.0)
    volume_score = min(characteristics.total_volume / high_volume_threshold, 1.0)
    stability_score = 1.0 / (1.0 + characteristics.cv)  # Lower CV = higher score
    pattern_score = characteristics.forecastability_score
    quality_score = characteristics.data_quality_score

    # NEW: Feature quality score (critical for individual modeling decision)
    feature_score = characteristics.feature_quality_score

    # Check specific feature conditions
    has_rich_features = characteristics.has_rich_features
    has_external_features = characteristics.has_external_features

    # ==========================================================================
    # WEIGHTED COMPOSITE SCORE - Features now contribute 15%
    # ==========================================================================
    composite_score = (
        0.15 * history_score +
        0.15 * nonzero_score +
        0.10 * volume_score +
        0.10 * stability_score +
        0.15 * pattern_score +
        0.10 * quality_score +
        0.15 * feature_score +  # NEW: Feature quality contribution
        0.10 * (1.0 if has_external_features else 0.0)  # Bonus for external features
    )

    # ==========================================================================
    # DECISION LOGIC - Now considers feature quality
    # ==========================================================================

    # INDIVIDUAL: Needs both good time series AND good features
    if (characteristics.has_sufficient_history and
        characteristics.has_sufficient_nonzero and
        characteristics.is_high_volume and
        feature_score >= min_feature_quality_individual and  # NEW: Feature quality check
        has_rich_features and  # NEW: Rich features required
        composite_score >= 0.65):

        strategy = ModelingStrategy.INDIVIDUAL
        model_group = f"key_{key}"
        feature_priority = 'full'
        n_features_max = 100
        rationale_parts.append(f"High-quality key (score={composite_score:.2f})")
        rationale_parts.append(f"Volume={characteristics.total_volume:.0f}, Obs={characteristics.n_observations}")
        rationale_parts.append(f"Rich features (quality={feature_score:.2f})")
        if has_external_features:
            rationale_parts.append("Has price/promo features")
        confidence = composite_score

    # HIERARCHICAL: Medium data with decent features
    elif (characteristics.has_sufficient_history and
          characteristics.has_sufficient_nonzero and
          feature_score >= 0.4 and  # NEW: Minimum feature quality for hierarchical
          composite_score >= 0.45):

        strategy = ModelingStrategy.HIERARCHICAL
        model_group = f"segment_{segment_id}" if segment_id is not None else f"key_{key}"
        feature_priority = 'full' if feature_score >= 0.6 else 'standard'
        n_features_max = 80 if feature_score >= 0.6 else 60
        rationale_parts.append(f"Medium-quality key (score={composite_score:.2f})")
        rationale_parts.append(f"Feature quality={feature_score:.2f}")
        rationale_parts.append("Using hierarchical: segment baseline + key adjustment")
        confidence = composite_score

    # SEGMENT_POOLED: Limited data, sparse, or poor features
    else:
        strategy = ModelingStrategy.SEGMENT_POOLED
        model_group = f"segment_{segment_id}" if segment_id is not None else "segment_default"

        # Adjust feature priority based on feature quality
        if feature_score < 0.3:
            feature_priority = 'minimal'
            n_features_max = 30
            rationale_parts.append(f"Poor feature quality ({feature_score:.2f}) - using minimal features")
        elif composite_score < 0.3:
            feature_priority = 'reduced'
            n_features_max = 50
            rationale_parts.append(f"Limited data (score={composite_score:.2f})")
        else:
            feature_priority = 'standard'
            n_features_max = 60
            rationale_parts.append(f"Moderate quality (score={composite_score:.2f})")

        rationale_parts.append(f"Pooling with segment {segment_id} for shared learning")
        confidence = max(0.3, composite_score)

    # ==========================================================================
    # MODEL RECOMMENDATIONS - Based on demand pattern AND feature availability
    # ==========================================================================

    # Base recommendations on demand pattern
    if characteristics.zero_fraction > 0.5:
        # High zeros - use zero-handling models first
        base_models = ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'catboost']
    elif characteristics.cv > 2.0:
        # High variability - robust tree models
        base_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'tweedie']
    else:
        # Stable demand - any tree model works well
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'zero_inflated']

    # NEW: Adjust based on feature availability
    if has_external_features and feature_score >= 0.6:
        # Rich external features - tree models excel
        recommended = ['lightgbm', 'xgboost', 'catboost'] + [m for m in base_models if m not in ['lightgbm', 'xgboost', 'catboost']]
        rationale_parts.append("Prioritizing tree models for external features")
    elif feature_score < 0.4:
        # Poor features - simpler models might do better
        recommended = ['lightgbm', 'random_forest', 'tweedie', 'xgboost', 'catboost']
        rationale_parts.append("Using regularized models for limited features")
    else:
        recommended = base_models

    return ModelingDecision(
        key=key,
        strategy=strategy,
        model_group=model_group,
        recommended_models=recommended,
        feature_priority=feature_priority,
        n_features_max=n_features_max,
        rationale=" | ".join(rationale_parts),
        confidence=confidence,
    )


def compute_key_characteristics(
    df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    key: str,
    time_format: str = 'year_week',
) -> KeyCharacteristics:
    """
    Compute comprehensive characteristics for a single key.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    key_col : str
        Column identifying keys
    date_col : str
        Date/period column
    target_col : str
        Target variable column
    key : str
        Specific key to analyze
    time_format : str
        'year_week' or 'year_month' — affects seasonal lag (52 vs 12)

    Returns
    -------
    KeyCharacteristics
        Computed characteristics
    """
    key_data = df[df[key_col] == key].sort_values(date_col)
    values = key_data[target_col].values

    n_obs = len(values)
    n_nonzero = np.sum(values > 0)
    zero_fraction = 1.0 - (n_nonzero / n_obs) if n_obs > 0 else 1.0

    nonzero_values = values[values > 0]
    mean_val = np.mean(nonzero_values) if len(nonzero_values) > 0 else 0
    std_val = np.std(nonzero_values) if len(nonzero_values) > 1 else 0
    cv = std_val / mean_val if mean_val > 0 else 0

    total_volume = np.sum(values)

    # Trend strength (linear regression R²)
    if n_obs > 2:
        x = np.arange(n_obs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(x, values)[0, 1] if np.std(values) > 0 else 0
            trend_strength = abs(corr) if not np.isnan(corr) else 0
    else:
        trend_strength = 0

    # Seasonal strength (autocorrelation at seasonal lag)
    # Adapt to time format: 12 for monthly, 52 for weekly
    seasonal_lag = 12 if time_format == 'year_month' else 52
    if n_obs > seasonal_lag + 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(values[:-seasonal_lag], values[seasonal_lag:])[0, 1]
            seasonal_strength = abs(corr) if not np.isnan(corr) else 0
    else:
        seasonal_strength = 0

    # Lag-1 autocorrelation
    if n_obs > 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(values[:-1], values[1:])[0, 1]
            autocorr_lag1 = corr if not np.isnan(corr) else 0
    else:
        autocorr_lag1 = 0

    # Forecastability score (combination of pattern strength)
    forecastability = (
        0.3 * min(autocorr_lag1, 1.0) +
        0.3 * trend_strength +
        0.2 * seasonal_strength +
        0.2 * (1.0 - zero_fraction)
    )

    # Data quality score
    missing_rate = key_data[target_col].isna().mean() if target_col in key_data else 0
    data_quality = 1.0 - missing_rate

    return KeyCharacteristics(
        key=key,
        n_observations=n_obs,
        n_nonzero=n_nonzero,
        zero_fraction=zero_fraction,
        cv=cv,
        mean_value=mean_val,
        total_volume=total_volume,
        trend_strength=trend_strength,
        seasonal_strength=seasonal_strength,
        autocorr_lag1=autocorr_lag1,
        forecastability_score=forecastability,
        data_quality_score=data_quality,
        feature_chars=None,  # Can be added later with compute_feature_characteristics
        time_format=time_format,
    )


def compute_feature_characteristics(
    df: pd.DataFrame,
    key_col: str,
    target_col: str,
    key: str,
    feature_cols: Optional[List[str]] = None,
    external_feature_patterns: Optional[List[str]] = None,
) -> FeatureCharacteristics:
    """
    Compute feature-level characteristics for a single key.

    This analyzes the QUALITY and AVAILABILITY of features for a key,
    which is critical for deciding whether the key should be modeled individually.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features
    key_col : str
        Column identifying keys
    target_col : str
        Target variable column
    key : str
        Specific key to analyze
    feature_cols : List[str], optional
        List of feature columns (auto-detected if None)
    external_feature_patterns : List[str], optional
        Patterns to identify external features (default: price, promo, weather, etc.)

    Returns
    -------
    FeatureCharacteristics
        Computed feature characteristics
    """
    key_data = df[df[key_col] == key].copy()

    # Default external feature patterns
    if external_feature_patterns is None:
        external_feature_patterns = [
            'price', 'promo', 'promotion', 'discount',
            'weather', 'temp', 'temperature',
            'holiday', 'event', 'store', 'distribution'
        ]

    # Auto-detect feature columns if not provided
    if feature_cols is None:
        exclude_cols = {key_col, target_col, 'year_week', 'year_month', 'date', 'split', 'model_group', 'segment_id'}
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    n_features_available = len(feature_cols)

    # Count features with non-zero variance for this key
    n_features_with_variance = 0
    for col in feature_cols:
        if col in key_data.columns:
            try:
                if key_data[col].std() > 1e-10:
                    n_features_with_variance += 1
            except:
                pass

    # Feature completeness (fraction of non-missing values)
    if len(key_data) > 0 and len(feature_cols) > 0:
        missing_rate = key_data[feature_cols].isna().mean().mean()
        feature_completeness = 1.0 - missing_rate
    else:
        feature_completeness = 0.0

    # Identify feature types
    lag_features = [c for c in feature_cols if 'lag' in c.lower() and target_col.lower() in c.lower()]
    external_features = []
    for col in feature_cols:
        col_lower = col.lower()
        if any(pat in col_lower for pat in external_feature_patterns):
            external_features.append(col)

    n_lag_features = len(lag_features)
    n_external_features = len(external_features)

    # Specific feature type checks
    has_price_features = any('price' in c.lower() for c in feature_cols)
    has_promo_features = any(any(p in c.lower() for p in ['promo', 'promotion', 'discount']) for c in feature_cols)
    has_calendar_features = any(any(p in c.lower() for p in ['week', 'month', 'quarter', 'year', 'holiday', 'fourier']) for c in feature_cols)

    # External feature coverage
    expected_external_count = 10  # Expected number of external features in a rich dataset
    external_feature_coverage = min(n_external_features / expected_external_count, 1.0)

    # Compute feature-target correlations
    target_values = key_data[target_col].values if target_col in key_data.columns else np.array([])
    correlations = []

    if len(target_values) > 5 and np.std(target_values) > 1e-10:
        for col in feature_cols[:50]:  # Limit to first 50 for speed
            if col in key_data.columns:
                try:
                    feat_values = key_data[col].fillna(0).values
                    if np.std(feat_values) > 1e-10:
                        corr = np.corrcoef(feat_values, target_values)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                except:
                    pass

    feature_target_correlation_max = max(correlations) if correlations else 0.0
    feature_target_correlation_mean = np.mean(correlations) if correlations else 0.0

    return FeatureCharacteristics(
        n_features_available=n_features_available,
        n_features_with_variance=n_features_with_variance,
        feature_completeness=feature_completeness,
        external_feature_coverage=external_feature_coverage,
        feature_target_correlation_max=feature_target_correlation_max,
        feature_target_correlation_mean=feature_target_correlation_mean,
        n_lag_features=n_lag_features,
        n_external_features=n_external_features,
        has_price_features=has_price_features,
        has_promo_features=has_promo_features,
        has_calendar_features=has_calendar_features,
    )


def compute_key_characteristics_with_features(
    df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    key: str,
    feature_cols: Optional[List[str]] = None,
    time_format: str = 'year_week',
) -> KeyCharacteristics:
    """
    Compute FULL key characteristics including feature analysis.

    This is the recommended function for production use as it includes
    both time series and feature-level characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features
    key_col : str
        Column identifying keys
    date_col : str
        Date/period column
    target_col : str
        Target variable column
    key : str
        Specific key to analyze
    feature_cols : List[str], optional
        List of feature columns (auto-detected if None)
    time_format : str
        'year_week' or 'year_month' — forwarded to compute_key_characteristics

    Returns
    -------
    KeyCharacteristics
        Full characteristics including feature_chars
    """
    # Get base characteristics
    base_chars = compute_key_characteristics(df, key_col, date_col, target_col, key, time_format=time_format)

    # Compute feature characteristics
    feature_chars = compute_feature_characteristics(
        df, key_col, target_col, key, feature_cols
    )

    # Return with feature characteristics attached
    return KeyCharacteristics(
        key=base_chars.key,
        n_observations=base_chars.n_observations,
        n_nonzero=base_chars.n_nonzero,
        zero_fraction=base_chars.zero_fraction,
        cv=base_chars.cv,
        mean_value=base_chars.mean_value,
        total_volume=base_chars.total_volume,
        trend_strength=base_chars.trend_strength,
        seasonal_strength=base_chars.seasonal_strength,
        autocorr_lag1=base_chars.autocorr_lag1,
        forecastability_score=base_chars.forecastability_score,
        data_quality_score=base_chars.data_quality_score,
        feature_chars=feature_chars,
        time_format=time_format,
    )


# =============================================================================
# SOPHISTICATED FEATURE SELECTION
# =============================================================================

@dataclass
class FeatureSelectionResult:
    """Result of feature selection process."""
    selected_features: List[str]
    feature_importance: Dict[str, float]
    selection_method: str
    n_original: int
    n_selected: int
    rationale: str


def select_features_tree_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    n_top: int = 50,
    importance_threshold: float = 0.001,
) -> FeatureSelectionResult:
    """
    Select features using tree-based importance (fast, effective).

    Uses LightGBM for fast importance computation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target values
    feature_cols : List[str]
        Feature column names
    n_top : int
        Maximum features to select
    importance_threshold : float
        Minimum importance to include

    Returns
    -------
    FeatureSelectionResult
        Selected features with importance scores
    """
    try:
        import lightgbm as lgb

        # Quick model for importance
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        }

        # Prepare data
        X_array = X[feature_cols].values

        model = lgb.LGBMRegressor(**params)
        model.fit(X_array, y)

        # Get importance
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_cols, importance))

        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])

        # Filter by threshold and n_top
        selected = []
        for feat, imp in sorted_features:
            if imp >= importance_threshold and len(selected) < n_top:
                selected.append(feat)

        # Ensure at least some features
        if len(selected) < 10:
            selected = [f for f, _ in sorted_features[:10]]

        return FeatureSelectionResult(
            selected_features=selected,
            feature_importance={f: feature_importance[f] for f in selected},
            selection_method='lightgbm_importance',
            n_original=len(feature_cols),
            n_selected=len(selected),
            rationale=f"Selected top {len(selected)} features by LightGBM importance",
        )

    except Exception as e:
        logger.warning(f"Tree importance failed: {e}, using correlation fallback")
        return select_features_correlation(X, y, feature_cols, n_top)


def select_features_correlation(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    n_top: int = 50,
) -> FeatureSelectionResult:
    """
    Select features using correlation with target (fast fallback).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target values
    feature_cols : List[str]
        Feature column names
    n_top : int
        Maximum features to select

    Returns
    -------
    FeatureSelectionResult
        Selected features with correlation scores
    """
    correlations = {}
    for col in feature_cols:
        try:
            if X[col].std() > 0:
                corr = np.corrcoef(X[col].fillna(0).values, y)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
        except:
            pass

    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: -x[1])
    selected = [f for f, _ in sorted_features[:n_top]]

    return FeatureSelectionResult(
        selected_features=selected,
        feature_importance={f: correlations.get(f, 0) for f in selected},
        selection_method='correlation',
        n_original=len(feature_cols),
        n_selected=len(selected),
        rationale=f"Selected top {len(selected)} features by correlation with target",
    )


def select_features_recursive_elimination(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    n_top: int = 50,
) -> FeatureSelectionResult:
    """
    Select features using recursive feature elimination.

    More sophisticated but slower - use for high-value keys.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target values
    feature_cols : List[str]
        Feature column names
    n_top : int
        Target number of features

    Returns
    -------
    FeatureSelectionResult
        Selected features
    """
    try:
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor

        # Use RF for RFE
        estimator = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )

        X_array = X[feature_cols].fillna(0).values

        selector = RFE(
            estimator,
            n_features_to_select=min(n_top, len(feature_cols)),
            step=0.1,
        )
        selector.fit(X_array, y)

        selected = [f for f, s in zip(feature_cols, selector.support_) if s]
        ranking = dict(zip(feature_cols, selector.ranking_))

        return FeatureSelectionResult(
            selected_features=selected,
            feature_importance={f: 1.0/ranking[f] for f in selected},
            selection_method='recursive_elimination',
            n_original=len(feature_cols),
            n_selected=len(selected),
            rationale=f"Selected {len(selected)} features via RFE",
        )

    except Exception as e:
        logger.warning(f"RFE failed: {e}, using tree importance fallback")
        return select_features_tree_importance(X, y, feature_cols, n_top)


def select_features_hybrid(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    n_top: int = 50,
    strategy: str = 'balanced',
) -> FeatureSelectionResult:
    """
    Hybrid feature selection combining multiple methods.

    This is the recommended approach for production.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target values
    feature_cols : List[str]
        Feature column names
    n_top : int
        Target number of features
    strategy : str
        'fast' (correlation + importance),
        'balanced' (importance + remove correlated),
        'thorough' (RFE + importance + correlation)

    Returns
    -------
    FeatureSelectionResult
        Selected features with combined scores
    """
    if strategy == 'fast':
        # Just use tree importance
        return select_features_tree_importance(X, y, feature_cols, n_top)

    elif strategy == 'balanced':
        # Tree importance + remove highly correlated features
        importance_result = select_features_tree_importance(
            X, y, feature_cols, n_top=min(n_top * 2, len(feature_cols))
        )

        # Remove highly correlated features
        selected = _remove_correlated_features(
            X, importance_result.selected_features, threshold=0.95
        )

        # Trim to n_top
        if len(selected) > n_top:
            # Keep by importance
            importance = importance_result.feature_importance
            selected = sorted(selected, key=lambda f: -importance.get(f, 0))[:n_top]

        return FeatureSelectionResult(
            selected_features=selected,
            feature_importance={f: importance_result.feature_importance.get(f, 0) for f in selected},
            selection_method='hybrid_balanced',
            n_original=len(feature_cols),
            n_selected=len(selected),
            rationale=f"Hybrid: importance + decorrelation → {len(selected)} features",
        )

    elif strategy == 'thorough':
        # Multiple methods, take intersection
        importance_result = select_features_tree_importance(X, y, feature_cols, n_top * 2)
        corr_result = select_features_correlation(X, y, feature_cols, n_top * 2)

        # Find features that both methods agree on
        importance_set = set(importance_result.selected_features[:n_top])
        corr_set = set(corr_result.selected_features[:n_top])

        # Intersection + top from importance
        agreed = list(importance_set & corr_set)
        remaining = [f for f in importance_result.selected_features if f not in agreed]
        selected = agreed + remaining[:n_top - len(agreed)]

        return FeatureSelectionResult(
            selected_features=selected,
            feature_importance={f: importance_result.feature_importance.get(f, 0) for f in selected},
            selection_method='hybrid_thorough',
            n_original=len(feature_cols),
            n_selected=len(selected),
            rationale=f"Thorough hybrid: {len(agreed)} agreed + {len(selected)-len(agreed)} importance",
        )

    else:
        return select_features_tree_importance(X, y, feature_cols, n_top)


def _remove_correlated_features(
    X: pd.DataFrame,
    features: List[str],
    threshold: float = 0.95
) -> List[str]:
    """Remove highly correlated features, keeping the first one."""
    if len(features) <= 1:
        return features

    try:
        X_subset = X[features].fillna(0)
        corr_matrix = X_subset.corr().abs()

        # Track features to drop
        to_drop = set()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if corr_matrix.iloc[i, j] > threshold:
                    to_drop.add(features[j])

        return [f for f in features if f not in to_drop]
    except:
        return features


# =============================================================================
# INTELLIGENT FEATURE CATEGORIZATION
# =============================================================================

def categorize_features(feature_cols: List[str], target_col: str) -> Dict[str, List[str]]:
    """
    Categorize features into meaningful groups for selection.

    Parameters
    ----------
    feature_cols : List[str]
        All feature column names
    target_col : str
        Target column name

    Returns
    -------
    Dict[str, List[str]]
        Features grouped by category
    """
    categories = {
        'lag_features': [],        # Direct lags of target
        'rolling_features': [],    # Rolling statistics
        'calendar_features': [],   # Time-based features
        'external_features': [],   # Price, promo, weather, etc.
        'intermittency_features': [],  # Zero-handling features
        'derived_features': [],    # Combinations, ratios
        'categorical_encoded': [], # Encoded categoricals
        'unknown': [],
    }

    target_lower = target_col.lower()

    for col in feature_cols:
        col_lower = col.lower()

        if 'lag' in col_lower and target_lower in col_lower:
            if 'diff' in col_lower or 'ratio' in col_lower:
                categories['derived_features'].append(col)
            else:
                categories['lag_features'].append(col)
        elif 'roll' in col_lower or 'rolling' in col_lower or 'ewm' in col_lower:
            categories['rolling_features'].append(col)
        elif any(cal in col_lower for cal in ['week', 'month', 'year', 'quarter', 'season', 'holiday', 'fourier', 'sin_', 'cos_']):
            categories['calendar_features'].append(col)
        elif any(ext in col_lower for ext in ['price', 'promo', 'weather', 'temp', 'store', 'distribution']):
            categories['external_features'].append(col)
        elif any(inter in col_lower for inter in ['zero', 'nonzero', 'occurrence', 'intermittent', 'since_last']):
            categories['intermittency_features'].append(col)
        elif 'encoded' in col_lower or col_lower.endswith('_enc'):
            categories['categorical_encoded'].append(col)
        elif 'diff' in col_lower or 'ratio' in col_lower or 'momentum' in col_lower:
            categories['derived_features'].append(col)
        else:
            categories['unknown'].append(col)

    return categories


def get_priority_features(
    categories: Dict[str, List[str]],
    strategy: str = 'balanced',
    max_per_category: int = 15,
) -> List[str]:
    """
    Get priority features based on category and strategy.

    Parameters
    ----------
    categories : Dict[str, List[str]]
        Categorized features
    strategy : str
        'minimal' (only essential), 'balanced' (good mix), 'full' (all)
    max_per_category : int
        Maximum features per category

    Returns
    -------
    List[str]
        Priority-ordered features
    """
    priority_order = [
        'lag_features',          # Most predictive for demand
        'rolling_features',      # Smooth trends
        'external_features',     # Business drivers
        'intermittency_features',  # Zero handling
        'calendar_features',     # Seasonality
        'derived_features',      # Momentum signals
        'categorical_encoded',   # Cross-sectional
        'unknown',
    ]

    if strategy == 'minimal':
        limits = {
            'lag_features': 6,
            'rolling_features': 4,
            'external_features': 5,
            'calendar_features': 4,
            'intermittency_features': 3,
            'derived_features': 3,
            'categorical_encoded': 3,
            'unknown': 2,
        }
    elif strategy == 'balanced':
        limits = {
            'lag_features': 10,
            'rolling_features': 8,
            'external_features': 10,
            'calendar_features': 6,
            'intermittency_features': 5,
            'derived_features': 5,
            'categorical_encoded': 6,
            'unknown': 5,
        }
    else:  # full
        limits = {cat: max_per_category for cat in priority_order}

    selected = []
    for cat in priority_order:
        cat_features = categories.get(cat, [])
        selected.extend(cat_features[:limits.get(cat, max_per_category)])

    return selected


# =============================================================================
# STATE-OF-THE-ART MODEL LEVEL ALLOCATION
# =============================================================================

# Minimum thresholds for individual model viability (WEEKLY defaults)
# For monthly data, use get_obs_thresholds(time_format) to get adapted values
MIN_OBS_FOR_INDIVIDUAL = 13  # At least ~3 months of weekly data (aggressive: more keys get individual models)
MIN_NONZERO_FOR_INDIVIDUAL = 3  # At least 3 non-zero observations (captures sparse/intermittent keys)
IDEAL_OBS_FOR_INDIVIDUAL = 104  # 2 years is ideal


def get_obs_thresholds(time_format: str = 'year_week') -> Dict[str, int]:
    """Get observation thresholds adapted to time format (weekly vs monthly).

    Returns
    -------
    Dict with keys: min_obs, min_nonzero, ideal_obs
    """
    if time_format == 'year_month':
        return {
            'min_obs': 8,         # ~8 months (relaxed: datasets often have <12 months after split)
            'min_nonzero': 4,     # At least 4 non-zero observations
            'ideal_obs': 24,      # 2 years is ideal
        }
    else:
        return {
            'min_obs': MIN_OBS_FOR_INDIVIDUAL,       # 13 (~3 months weekly)
            'min_nonzero': MIN_NONZERO_FOR_INDIVIDUAL,  # 3
            'ideal_obs': IDEAL_OBS_FOR_INDIVIDUAL,   # 104 (2 years weekly)
        }


def compute_segment_relative_metrics(df: pd.DataFrame, segment_col: str = 'segment_id') -> pd.DataFrame:
    """
    Compute segment-relative metrics for sophisticated uniqueness scoring.

    This computes how different each key is from its segment peers, which is
    crucial for determining if a key truly needs individual modeling or can
    benefit from pooling with similar keys.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with key characteristics and segment assignments
    segment_col : str
        Column name for segment assignment

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added segment-relative columns:
        - _seg_cv_zscore: How many std deviations from segment mean CV
        - _seg_volume_zscore: How many std deviations from segment mean volume
        - _seg_zeros_zscore: How many std deviations from segment zero_fraction
        - _seg_forecastability_zscore: How different forecastability is from segment
        - _is_segment_outlier: Boolean flag if key is outlier within segment
        - _segment_homogeneity: How similar segment members are (affects pooling benefit)
    """
    df = df.copy()

    # Identify numeric columns for segment statistics
    cv_col = 'cv_clean' if 'cv_clean' in df.columns else 'cv'
    vol_col = 'volume_mean' if 'volume_mean' in df.columns else ('mean' if 'mean' in df.columns else None)
    zeros_col = 'zero_fraction_clean' if 'zero_fraction_clean' in df.columns else 'zero_fraction'
    forecast_col = 'forecastability_score'

    # Compute segment-level statistics
    segment_stats = df.groupby(segment_col).agg({
        cv_col: ['mean', 'std', 'median'] if cv_col in df.columns else [],
        zeros_col: ['mean', 'std', 'median'] if zeros_col in df.columns else [],
        forecast_col: ['mean', 'std'] if forecast_col in df.columns else [],
    })

    # Flatten column names
    segment_stats.columns = ['_'.join(col).strip('_') for col in segment_stats.columns]
    segment_stats = segment_stats.reset_index()

    # Merge segment stats back
    df = df.merge(segment_stats, on=segment_col, how='left', suffixes=('', '_seg'))

    # Compute z-scores (how different from segment mean)
    # Use robust calculation to handle edge cases
    def safe_zscore(value, mean, std):
        if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std == 0:
            return 0.0
        return (value - mean) / max(std, 0.001)

    # CV z-score within segment
    if cv_col in df.columns and f'{cv_col}_mean' in df.columns:
        df['_seg_cv_zscore'] = df.apply(
            lambda row: safe_zscore(
                row.get(cv_col, 0),
                row.get(f'{cv_col}_mean', 0),
                row.get(f'{cv_col}_std', 1)
            ), axis=1
        )
    else:
        df['_seg_cv_zscore'] = 0.0

    # Volume z-score within segment
    if vol_col and vol_col in df.columns:
        vol_stats = df.groupby(segment_col)[vol_col].agg(['mean', 'std']).reset_index()
        vol_stats.columns = [segment_col, '_vol_seg_mean', '_vol_seg_std']
        df = df.merge(vol_stats, on=segment_col, how='left')
        df['_seg_volume_zscore'] = df.apply(
            lambda row: safe_zscore(
                row.get(vol_col, 0),
                row.get('_vol_seg_mean', 0),
                row.get('_vol_seg_std', 1)
            ), axis=1
        )
    else:
        df['_seg_volume_zscore'] = 0.0

    # Zero fraction z-score within segment
    if zeros_col in df.columns and f'{zeros_col}_mean' in df.columns:
        df['_seg_zeros_zscore'] = df.apply(
            lambda row: safe_zscore(
                row.get(zeros_col, 0),
                row.get(f'{zeros_col}_mean', 0),
                row.get(f'{zeros_col}_std', 1)
            ), axis=1
        )
    else:
        df['_seg_zeros_zscore'] = 0.0

    # Forecastability z-score within segment
    if forecast_col in df.columns and f'{forecast_col}_mean' in df.columns:
        df['_seg_forecastability_zscore'] = df.apply(
            lambda row: safe_zscore(
                row.get(forecast_col, 0.5),
                row.get(f'{forecast_col}_mean', 0.5),
                row.get(f'{forecast_col}_std', 0.1)
            ), axis=1
        )
    else:
        df['_seg_forecastability_zscore'] = 0.0

    # Compute segment homogeneity (how similar segment members are)
    # Low CV-of-CV within segment = homogeneous = pooling benefits more
    if cv_col in df.columns:
        seg_homogeneity = df.groupby(segment_col)[cv_col].agg(
            lambda x: 1.0 / (1.0 + x.std() / max(x.mean(), 0.01)) if len(x) > 1 else 0.5
        ).reset_index()
        seg_homogeneity.columns = [segment_col, '_segment_homogeneity']
        df = df.merge(seg_homogeneity, on=segment_col, how='left')
    else:
        df['_segment_homogeneity'] = 0.5

    # Identify segment outliers (keys that are very different from peers)
    # A key is an outlier if any z-score exceeds 2.0 standard deviations
    df['_is_segment_outlier'] = (
        (df['_seg_cv_zscore'].abs() > 2.0) |
        (df['_seg_volume_zscore'].abs() > 2.5) |  # Volume can vary more
        (df['_seg_zeros_zscore'].abs() > 2.0)
    )

    logger.info(f"Computed segment-relative metrics: {df['_is_segment_outlier'].sum()} outliers identified")

    return df


@dataclass
class ModelLevelAllocationResult:
    """
    Result of model level allocation for all keys.

    This is the primary output used by the training manifest to determine
    how each key should be modeled (individual vs segment-pooled).
    """
    allocations: pd.DataFrame  # DataFrame with key, model_level, strategy, rationale, confidence
    summary: Dict[str, Any]    # Summary statistics
    individual_keys: List[str]  # Keys assigned to individual models
    segment_keys: Dict[str, List[str]]  # Segment -> list of keys
    allocation_rationale: str   # Human-readable explanation


def compute_individual_model_score(
    row: pd.Series,
    weights: Optional[Dict[str, float]] = None,
    use_segment_relative: bool = True,
    time_format: str = 'year_week',
) -> float:
    """
    Compute a sophisticated score indicating whether a key should get an individual model.

    This uses a multi-factor scoring approach considering:
    1. Data sufficiency (history length, non-zero observations) - GATE FACTOR
    2. Volume importance (business impact)
    3. Forecastability (learnable patterns)
    4. Uniqueness (how different from segment peers) - NOW SEGMENT-RELATIVE
    5. Predictability (signal quality)
    6. Stability vs Complexity balance
    7. Demand pattern strategy (NEW - intermittent/lumpy benefit from pooling)

    Higher score = more likely to benefit from individual model.

    IMPORTANT: Keys with insufficient data are penalized heavily as individual
    models will likely overfit. Thresholds adapt to time_format (weekly vs monthly).

    Parameters
    ----------
    row : pd.Series
        Row from per_key_with_segments.csv with all characteristics
        Should include segment-relative metrics (_seg_cv_zscore, etc.) if available
    weights : Dict[str, float], optional
        Custom weights for each factor
    use_segment_relative : bool
        If True, use segment-relative metrics for uniqueness (recommended)
    time_format : str
        'year_week' or 'year_month' — adapts observation thresholds

    Returns
    -------
    float
        Score between 0 and 1, where higher = prefer individual model
    """
    # Default weights (empirically tuned for demand forecasting)
    # NOTE: Added demand_pattern_strategy factor
    default_weights = {
        'data_sufficiency': 0.20,
        'volume_importance': 0.12,
        'forecastability': 0.18,
        'uniqueness': 0.15,
        'predictability': 0.12,
        'stability': 0.08,
        'autocorrelation': 0.05,
        'demand_pattern_strategy': 0.10,  # NEW: Considers pooling benefits for intermittent
    }
    w = weights or default_weights

    # =========================================================================
    # FACTOR 1: DATA SUFFICIENCY (20%)
    # More data = better individual model training
    # =========================================================================
    # Support multiple column name conventions from segmentation
    n_obs = row.get('n_obs', row.get('history_length', row.get('n_periods', 0)))
    zero_frac_for_calc = row.get('zero_fraction', row.get('zero_fraction_clean', 0.5))
    n_nonzero = int(n_obs * (1 - zero_frac_for_calc)) if n_obs > 0 else 0

    # Score based on observations — adapt to time_format
    thresholds = get_obs_thresholds(time_format)
    ideal_obs = thresholds['ideal_obs']  # 104 weekly, 24 monthly
    ideal_nonzero = 50 if time_format != 'year_month' else 15
    obs_score = min(n_obs / ideal_obs, 1.0) if n_obs > 0 else 0.0
    nonzero_score = min(n_nonzero / ideal_nonzero, 1.0) if n_nonzero > 0 else 0.0
    data_sufficiency = 0.6 * obs_score + 0.4 * nonzero_score

    # =========================================================================
    # FACTOR 2: VOLUME IMPORTANCE (15%)
    # High-volume keys have bigger business impact, deserve individual attention
    # =========================================================================
    # Support multiple column name conventions: mean, volume_mean, avg
    mean_val = row.get('mean', row.get('volume_mean', row.get('avg', 0)))
    total_vol = row.get('sum', row.get('volume_total', row.get('total', 0)))
    volume_tier = row.get('volume_tier', None)

    # Tiered scoring - only use if volume_tier is actually set
    tier_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.9}
    if volume_tier is not None:
        tier_score = tier_scores.get(str(volume_tier).lower(), 0.5)
    else:
        # Infer tier from mean value if not explicitly set
        if mean_val > 100:
            tier_score = 0.9
        elif mean_val > 10:
            tier_score = 0.5
        else:
            tier_score = 0.2

    # Continuous score based on mean (log scale for robustness)
    if mean_val > 0:
        log_mean = np.log1p(mean_val)
        mean_score = min(log_mean / np.log1p(1000), 1.0)  # 1000 mean = max score
    else:
        mean_score = 0.0

    volume_importance = 0.4 * tier_score + 0.6 * mean_score

    # =========================================================================
    # FACTOR 3: FORECASTABILITY (20%)
    # Keys with learnable patterns benefit more from individual models
    # =========================================================================
    forecastability = row.get('forecastability_score', 0.5)
    # Clamp and normalize
    forecastability = max(0, min(forecastability, 1.0))

    # =========================================================================
    # FACTOR 4: UNIQUENESS / OUTLIER SCORE (15%)
    # Keys that are different from their SEGMENT PEERS need individual models
    # ENHANCED: Now uses segment-relative z-scores when available
    # =========================================================================
    cv = row.get('cv', row.get('cv_clean', 1.0))
    zero_frac = row.get('zero_fraction', row.get('zero_fraction_clean', 0.5))
    adi = row.get('adi', row.get('adi_clean', 1.0))

    # Check if segment-relative metrics are available (from compute_segment_relative_metrics)
    has_seg_relative = '_seg_cv_zscore' in row.index if hasattr(row, 'index') else '_seg_cv_zscore' in row

    if use_segment_relative and has_seg_relative:
        # USE SEGMENT-RELATIVE UNIQUENESS (PREFERRED)
        # A key is unique if it's statistically different from its segment peers
        seg_cv_z = abs(row.get('_seg_cv_zscore', 0))
        seg_vol_z = abs(row.get('_seg_volume_zscore', 0))
        seg_zeros_z = abs(row.get('_seg_zeros_zscore', 0))
        is_segment_outlier = row.get('_is_segment_outlier', False)

        # Score based on how many standard deviations from segment mean
        # |z| > 2 = outlier, |z| > 1.5 = moderately different, |z| < 1 = typical
        uniqueness = 0.0
        if seg_cv_z > 2.0:
            uniqueness += 0.35
        elif seg_cv_z > 1.5:
            uniqueness += 0.2
        elif seg_cv_z > 1.0:
            uniqueness += 0.1

        if seg_vol_z > 2.5:  # Volume can naturally vary more
            uniqueness += 0.30
        elif seg_vol_z > 1.5:
            uniqueness += 0.15

        if seg_zeros_z > 2.0:
            uniqueness += 0.25
        elif seg_zeros_z > 1.5:
            uniqueness += 0.12

        # Bonus for being a clear segment outlier
        if is_segment_outlier:
            uniqueness = min(uniqueness + 0.1, 1.0)

        # Consider segment homogeneity: in homogeneous segments, even small
        # differences matter more; in heterogeneous segments, differences are normal
        seg_homogeneity = row.get('_segment_homogeneity', 0.5)
        if seg_homogeneity > 0.7:  # Very homogeneous segment
            uniqueness *= 1.2  # Amplify uniqueness score
        elif seg_homogeneity < 0.3:  # Very heterogeneous segment
            uniqueness *= 0.8  # Dampen uniqueness score (differences are normal)

    else:
        # FALLBACK: Use global thresholds (original logic)
        # Smooth demand (low CV, low zeros) with others = pool together
        # Erratic/lumpy with extreme values = likely unique, benefit from individual
        is_extreme_cv = cv > 2.5 or cv < 0.3
        is_extreme_zeros = zero_frac > 0.8 or zero_frac < 0.05
        is_extreme_adi = adi > 10 or adi < 1.1

        uniqueness = 0.0
        if is_extreme_cv:
            uniqueness += 0.4
        if is_extreme_zeros:
            uniqueness += 0.3
        if is_extreme_adi:
            uniqueness += 0.3

    uniqueness = min(uniqueness, 1.0)  # Cap at 1.0

    # =========================================================================
    # FACTOR 5: PREDICTABILITY (15%)
    # High predictability = cleaner signal = individual model can capture nuances
    # =========================================================================
    # Use predictability_score, or fall back to stability_score, or derive from autocorr
    predictability = row.get('predictability_score', None)
    if predictability is None:
        predictability = row.get('stability_score', None)
    if predictability is None:
        # Derive from autocorrelation if available - high autocorr = more predictable
        autocorr_val = row.get('autocorr_lag1', row.get('autocorr_lag1_clean', None))
        if autocorr_val is not None and not np.isnan(autocorr_val):
            predictability = 0.3 + 0.7 * abs(autocorr_val)  # Scale 0.3-1.0
        else:
            # Last resort: derive from CV - low CV = more predictable
            cv_val = row.get('cv', row.get('cv_clean', 1.0))
            predictability = max(0.1, 1.0 - min(cv_val / 3.0, 0.9))
    # Clamp and normalize
    predictability = max(0, min(float(predictability), 1.0))

    # =========================================================================
    # FACTOR 6: STABILITY (10%)
    # Stable demand patterns are easier to model individually
    # Highly erratic might benefit from pooling for regularization
    # =========================================================================
    variability_tier = row.get('variability_tier', 'moderate')
    tier_stability = {
        'stable': 0.9,
        'moderate': 0.6,
        'variable': 0.3,
    }
    stability = tier_stability.get(str(variability_tier).lower(), 0.5)

    # Adjust: If CV is very high, reduce stability score
    if cv > 3.0:
        stability *= 0.7

    # =========================================================================
    # FACTOR 7: AUTOCORRELATION (5%)
    # Strong autocorrelation = predictable patterns = individual model benefits
    # =========================================================================
    autocorr = row.get('autocorr_lag1', row.get('autocorr_lag1_clean', 0))
    autocorr = abs(autocorr) if not np.isnan(autocorr) else 0
    autocorr_score = min(autocorr, 1.0)

    # =========================================================================
    # FACTOR 8: DEMAND PATTERN STRATEGY (10%) - NEW
    # Considers whether the demand pattern benefits from pooling or individual
    #
    # CRITICAL INSIGHT:
    # - SMOOTH demand: Individual models can capture nuances well
    # - ERRATIC demand: May benefit from pooling for regularization
    # - INTERMITTENT/LUMPY: Strongly benefit from pooling due to sparse signal
    #   (individual models overfit on limited non-zero observations)
    # =========================================================================
    demand_pattern = str(row.get('demand_pattern', row.get('intermittency_class', 'unknown'))).lower()

    # Score: higher = better for individual, lower = better for pooling
    demand_pattern_scores = {
        'smooth': 0.85,      # Smooth demand benefits from individual attention
        'erratic': 0.50,     # Erratic may benefit from pooling for regularization
        'intermittent': 0.25, # Intermittent strongly benefits from pooling
        'lumpy': 0.20,       # Lumpy strongly benefits from pooling
        'unknown': 0.50,     # Default neutral
    }
    demand_pattern_strategy = demand_pattern_scores.get(demand_pattern, 0.50)

    # CRITICAL: Apply data sufficiency penalty for intermittent/lumpy
    # If non-zero observations are low AND pattern is sparse, pooling is essential
    min_nonzero_thresh = thresholds['min_nonzero']
    if demand_pattern in ('intermittent', 'lumpy') and n_nonzero < min_nonzero_thresh:
        demand_pattern_strategy *= 0.5  # Strong penalty - push toward pooling

    # =========================================================================
    # HARD GATE: MINIMUM DATA REQUIREMENT
    # Keys without sufficient data should NOT get individual models regardless
    # of other factors - they will overfit
    # Thresholds adapt to time_format via get_obs_thresholds()
    # =========================================================================
    min_obs_thresh = thresholds['min_obs']  # 52 weekly, 12 monthly
    data_gate_penalty = 1.0
    if n_obs < min_obs_thresh:
        # Significant penalty for insufficient observations
        data_gate_penalty = max(0.3, n_obs / min_obs_thresh)
    if n_nonzero < min_nonzero_thresh:
        # Additional penalty for insufficient non-zero observations
        data_gate_penalty *= max(0.4, n_nonzero / min_nonzero_thresh)

    # =========================================================================
    # WEIGHTED COMBINATION
    # =========================================================================
    raw_score = (
        w['data_sufficiency'] * data_sufficiency +
        w['volume_importance'] * volume_importance +
        w['forecastability'] * forecastability +
        w['uniqueness'] * uniqueness +
        w['predictability'] * predictability +
        w['stability'] * stability +
        w['autocorrelation'] * autocorr_score +
        w.get('demand_pattern_strategy', 0.10) * demand_pattern_strategy
    )

    # Apply data gate penalty (multiplicative to ensure minimum data is met)
    score = raw_score * data_gate_penalty

    return min(max(score, 0.0), 1.0)


def allocate_model_levels(
    seg_df: pd.DataFrame,
    key_col: str = 'key',
    segment_col: str = 'segment_id',
    min_individual_score: float = 0.65,
    max_individual_pct: float = 0.30,
    min_segment_size: int = 10,
    volume_override_quantile: float = 0.93,
    forecastability_override: float = 0.75,
    min_nonzero_obs_for_individual: int = 52,
    max_zero_fraction_for_individual: float = 0.70,
    top_volume_bypass_quantile: Optional[float] = 0.80,
    use_hierarchical: bool = False,
    verbose: bool = True,
    time_format: str = 'year_week',
) -> ModelLevelAllocationResult:
    """
    STATE-OF-THE-ART model level allocation for demand forecasting.

    This function uses a sophisticated multi-factor approach to determine
    whether each key should be modeled individually or as part of a segment.

    Decision Factors:
    1. Individual Model Score (composite of 7 factors)
    2. Volume Override (top keys always get individual models)
    3. Forecastability Override (highly forecastable keys get individual)
    4. Segment Size Constraints (ensure viable segment models)
    5. Demand Pattern Considerations (lumpy/intermittent handling)

    Parameters
    ----------
    seg_df : pd.DataFrame
        DataFrame with key characteristics from segmentation
        Required columns: key, segment_id
        Recommended columns: mean, cv, zero_fraction, forecastability_score,
                           predictability_score, n_obs, demand_pattern, etc.
    key_col : str
        Column name for key identifier
    segment_col : str
        Column name for segment assignment
    min_individual_score : float
        Minimum score (0-1) required for individual model allocation
    max_individual_pct : float
        Maximum percentage of keys that can get individual models
        Prevents over-allocation which would defeat pooling benefits
    min_segment_size : int
        Minimum keys per segment to ensure viable segment models
    volume_override_quantile : float
        Keys above this volume quantile ALWAYS get individual models
    forecastability_override : float
        Keys with forecastability above this ALWAYS get individual models
    use_hierarchical : bool
        If True, use hierarchical modeling for borderline keys
    verbose : bool
        Print allocation summary

    Returns
    -------
    ModelLevelAllocationResult
        Complete allocation result with assignments, summary, and rationale

    Example
    -------
    >>> result = allocate_model_levels(seg_df, min_individual_score=0.6)
    >>> print(f"Individual models: {len(result.individual_keys)}")
    >>> print(f"Segment models: {len(result.segment_keys)}")
    """
    logger.info("Starting state-of-the-art model level allocation...")

    df = seg_df.copy()
    n_keys = len(df)

    # =========================================================================
    # STEP 0: Normalize segment_col to handle different formats
    # =========================================================================
    # Handle case where segment_col doesn't exist but 'segment' does
    if segment_col not in df.columns:
        if 'segment' in df.columns:
            df[segment_col] = df['segment']
            logger.info(f"Copied 'segment' to '{segment_col}'")
        else:
            # Create a single segment if no segment info exists
            df[segment_col] = 0
            logger.warning(f"No segment column found - assigning all keys to segment 0")

    # Normalize segment IDs (convert "S1", "S2" to "0", "1" etc.)
    if df[segment_col].dtype == object:
        unique_segs = df[segment_col].unique()
        # Check if segments look like "S1", "S2", "S3" pattern
        if any(str(s).startswith('S') and str(s)[1:].isdigit() for s in unique_segs):
            # Extract numeric part and convert to int
            df[segment_col] = df[segment_col].apply(
                lambda x: int(str(x)[1:]) - 1 if str(x).startswith('S') and str(x)[1:].isdigit() else x
            )
            logger.info(f"Normalized segment IDs from S1/S2/S3 format to 0/1/2 format")

    # Ensure segment_col is numeric if possible
    try:
        df[segment_col] = pd.to_numeric(df[segment_col], errors='coerce').fillna(df[segment_col])
    except Exception:
        pass

    logger.info(f"Segment distribution: {df[segment_col].value_counts().to_dict()}")

    # =========================================================================
    # STEP 0.5: Compute segment-relative metrics for sophisticated uniqueness
    # This enables the scoring function to determine if a key is truly unique
    # WITHIN its segment, not just globally extreme
    # =========================================================================
    logger.info("Computing segment-relative metrics for uniqueness scoring...")
    df = compute_segment_relative_metrics(df, segment_col=segment_col)

    # =========================================================================
    # STEP 1: Compute individual model scores for all keys
    # Now uses enhanced scoring with segment-relative uniqueness and demand pattern strategy
    # =========================================================================
    logger.info("Computing individual model scores with segment-aware uniqueness...")
    df['_individual_score'] = df.apply(
        lambda row: compute_individual_model_score(row, use_segment_relative=True, time_format=time_format), axis=1
    )

    # =========================================================================
    # STEP 2: Apply volume override (top keys ALWAYS get individual models)
    # =========================================================================
    # Check for volume column variants (volume_mean is common from EDA/segmentation)
    if 'volume_mean' in df.columns:
        volume_col = 'volume_mean'
    elif 'mean' in df.columns:
        volume_col = 'mean'
    else:
        volume_col = 'sum'
    if volume_col in df.columns:
        volume_threshold = df[volume_col].quantile(volume_override_quantile)
        df['_volume_override'] = df[volume_col] >= volume_threshold
        logger.info(f"Volume override threshold: {volume_threshold:.2f} "
                   f"({df['_volume_override'].sum()} keys)")

        # Top-volume bypass: keys above this (typically stricter) quantile
        # bypass the sparsity gate entirely — they're important enough
        # that a key-specific model is worth the risk even on lumpy
        # demand. Only requires the basic min_obs gate.
        if top_volume_bypass_quantile is not None:
            _bypass_threshold = df[volume_col].quantile(top_volume_bypass_quantile)
            df['_top_volume_bypass'] = df[volume_col] >= _bypass_threshold
            logger.info(
                f"Top-volume bypass threshold: {_bypass_threshold:.2f} "
                f"(quantile={top_volume_bypass_quantile}, "
                f"{df['_top_volume_bypass'].sum()} keys will bypass sparsity gate)"
            )
        else:
            df['_top_volume_bypass'] = False
    else:
        df['_volume_override'] = False
        df['_top_volume_bypass'] = False

    # =========================================================================
    # STEP 3: Apply forecastability override
    # =========================================================================
    if 'forecastability_score' in df.columns:
        df['_forecastability_override'] = df['forecastability_score'] >= forecastability_override
        logger.info(f"Forecastability override: {df['_forecastability_override'].sum()} keys")
    else:
        df['_forecastability_override'] = False

    # =========================================================================
    # STEP 3.5: Data Sufficiency Gate - CRITICAL
    # Keys without sufficient data should NEVER get individual models
    # regardless of other factors (they will overfit)
    # =========================================================================
    n_obs_col = 'n_obs' if 'n_obs' in df.columns else ('history_length' if 'history_length' in df.columns else None)
    zero_frac_col = 'zero_fraction_clean' if 'zero_fraction_clean' in df.columns else 'zero_fraction'

    if n_obs_col and n_obs_col in df.columns:
        # Compute non-zero observations
        df['_n_nonzero'] = df.apply(
            lambda row: int(row.get(n_obs_col, 0) * (1 - row.get(zero_frac_col, 0.5))), axis=1
        )
        # Gate: Must have minimum observations AND minimum non-zero
        # Adapt thresholds to time_format (weekly vs monthly)
        _thresholds = get_obs_thresholds(time_format)
        _min_obs = _thresholds['min_obs']
        # Respect user-tunable floor from config if stricter than the
        # time-format default.
        _min_nonzero = max(
            int(_thresholds['min_nonzero']),
            int(min_nonzero_obs_for_individual),
        )

        # Strict gate (zero-fraction + min-nonzero): applies to the
        # long tail of low/medium-volume keys.
        strict_nonzero_ok = df['_n_nonzero'] >= _min_nonzero
        if zero_frac_col in df.columns and max_zero_fraction_for_individual < 1.0:
            strict_zf_ok = df[zero_frac_col].fillna(1.0) <= max_zero_fraction_for_individual
        else:
            strict_zf_ok = pd.Series(True, index=df.index)

        min_obs_ok = df[n_obs_col] >= _min_obs

        # Composite rule:
        #   - top-volume bypass keys need only min_obs
        #   - every other key needs min_obs AND strict nonzero AND strict zf
        strict_pass = min_obs_ok & strict_nonzero_ok & strict_zf_ok
        df['_data_sufficient'] = (
            df.get('_top_volume_bypass', False) & min_obs_ok
        ) | strict_pass

        n_bypass = int((df.get('_top_volume_bypass', pd.Series(False, index=df.index))
                        & min_obs_ok
                        & ~strict_pass).sum())
        n_insufficient = int((~df['_data_sufficient']).sum())
        logger.info(
            f"Data sufficiency gate: {n_insufficient} keys insufficient for individual models "
            f"(require min_obs >= {_min_obs} AND "
            f"[(non_zero >= {_min_nonzero} AND zero_frac <= {max_zero_fraction_for_individual:.2f}) "
            f"OR top_volume_bypass], time_format={time_format})"
        )
        if n_bypass > 0:
            logger.info(
                f"  Top-volume bypass promoted {n_bypass} lumpy-but-high-volume keys "
                f"past the sparsity gate (top_volume_bypass_quantile="
                f"{top_volume_bypass_quantile})."
            )
    else:
        df['_data_sufficient'] = True  # If we can't check, allow all

    # =========================================================================
    # STEP 4: Determine initial allocation
    # =========================================================================
    # Score-based: above threshold gets individual
    score_qualified = df['_individual_score'] >= min_individual_score

    # Override-based: volume or forecastability override. Top-volume bypass
    # also counts as an override — for a key in the top bypass quantile
    # we want to force individual modelling regardless of score (it's the
    # whole point of widening allocation for the high-impact tier). The
    # relaxed data_sufficient gate above already lets these keys past
    # sparsity; without adding them here they'd still need to pass the
    # score threshold to become individual.
    override_qualified = (
        df['_volume_override']
        | df['_forecastability_override']
        | df.get('_top_volume_bypass', False)
    )

    # Combined initial qualification - BUT data sufficiency gate takes precedence
    # Even volume/forecastability overrides are blocked if data is insufficient
    df['_initial_individual'] = (score_qualified | override_qualified) & df['_data_sufficient']

    # Log if any overrides were blocked by data sufficiency
    n_override_blocked = ((df['_volume_override'] | df['_forecastability_override']) & ~df['_data_sufficient']).sum()
    if n_override_blocked > 0:
        logger.warning(f"{n_override_blocked} keys qualified by override but blocked by data sufficiency gate")

    # =========================================================================
    # STEP 5: Apply constraints (max individual %, min segment size)
    # =========================================================================
    max_individual_count = int(n_keys * max_individual_pct)

    # If too many qualified, prioritize by score
    n_initial_individual = df['_initial_individual'].sum()
    if n_initial_individual > max_individual_count:
        logger.info(f"Constraining individual models: {n_initial_individual} -> {max_individual_count}")
        # Keep top by score, but always keep override keys. Top-volume
        # bypass keys count as overrides here — the whole point of the
        # bypass is that these keys are too important to let the cap
        # push them back to pooled.
        _override_mask = (
            df['_volume_override']
            | df['_forecastability_override']
            | df.get('_top_volume_bypass', False)
        )
        override_keys = df[_override_mask][key_col].tolist()
        n_override = len(override_keys)

        # Remaining slots for score-based
        remaining_slots = max(0, max_individual_count - n_override)

        # Get top non-override keys by score
        non_override_df = df[~_override_mask]
        top_by_score = non_override_df.nlargest(remaining_slots, '_individual_score')[key_col].tolist()

        individual_keys_set = set(override_keys + top_by_score)
        df['_final_individual'] = df[key_col].isin(individual_keys_set)

        # If override_keys alone exceed the cap, honour them all but log
        # that the cap was breached by business-critical overrides.
        if n_override > max_individual_count:
            logger.warning(
                f"Override tier has {n_override} keys, which exceeds the "
                f"max_individual_pct cap of {max_individual_count}. Keeping "
                f"all overrides (bypass tier is non-negotiable); no score-"
                f"based individuals added."
            )
    else:
        df['_final_individual'] = df['_initial_individual']

    # =========================================================================
    # STEP 6: Ensure minimum segment sizes
    # =========================================================================
    # Check if any segment becomes too small after individual allocation
    segment_sizes = df[~df['_final_individual']].groupby(segment_col).size()
    small_segments = segment_sizes[segment_sizes < min_segment_size].index.tolist()

    if small_segments:
        logger.info(f"Adjusting for segment size constraints: {len(small_segments)} small segments")
        for seg in small_segments:
            seg_mask = (df[segment_col] == seg) & df['_final_individual']
            seg_individual_count = seg_mask.sum()

            if seg_individual_count > 0:
                # Move lowest-scored individual keys back to segment
                seg_individual_df = df[seg_mask].nsmallest(
                    max(0, seg_individual_count - 2), '_individual_score'
                )
                df.loc[seg_individual_df.index, '_final_individual'] = False

    # =========================================================================
    # STEP 7: Create final model_level assignments
    # =========================================================================
    def assign_model_level(row):
        if row['_final_individual']:
            return row[key_col]  # Individual model uses key as model_level
        else:
            return str(row[segment_col])  # Segment model uses segment_id

    def assign_strategy(row):
        if row['_final_individual']:
            # Volume override (top ~7% by default) trumps bypass label so
            # existing downstream filters that key off 'individual_volume'
            # keep working. Bypass-only tier (7-10% band, or whatever lies
            # between override and bypass quantiles) gets its own label.
            if row['_volume_override']:
                return 'individual_volume'
            elif row.get('_top_volume_bypass', False):
                return 'individual_top_volume_bypass'
            elif row['_forecastability_override']:
                return 'individual_forecastability'
            else:
                return 'individual_score'
        else:
            return 'segment_pooled'

    def assign_rationale(row):
        parts = []
        parts.append(f"score={row['_individual_score']:.3f}")

        # Include segment-relative uniqueness info if available
        if '_is_segment_outlier' in row.index:
            if row.get('_is_segment_outlier', False):
                parts.append("segment_outlier")

        if row['_final_individual']:
            if row['_volume_override']:
                parts.append("high_volume_override")
            elif row['_forecastability_override']:
                parts.append("high_forecastability")
            else:
                parts.append("above_score_threshold")
        else:
            parts.append(f"pooled_with_segment_{row[segment_col]}")

            # Explain why pooled
            if '_data_sufficient' in row.index and not row.get('_data_sufficient', True):
                parts.append("insufficient_data")
            elif row['_individual_score'] < min_individual_score:
                parts.append("below_threshold")
            else:
                parts.append("constrained_by_limits")

            # Add demand pattern context
            demand_pattern = str(row.get('demand_pattern', '')).lower()
            if demand_pattern in ('intermittent', 'lumpy'):
                parts.append(f"{demand_pattern}_benefits_from_pooling")

        return " | ".join(parts)

    df['model_level'] = df.apply(assign_model_level, axis=1)
    df['model_strategy'] = df.apply(assign_strategy, axis=1)
    df['allocation_rationale'] = df.apply(assign_rationale, axis=1)
    df['allocation_confidence'] = df['_individual_score']

    # =========================================================================
    # STEP 8: Build result
    # =========================================================================
    individual_keys = df[df['_final_individual']][key_col].tolist()
    segment_keys = {}
    for seg in df[segment_col].unique():
        seg_keys = df[(df[segment_col] == seg) & (~df['_final_individual'])][key_col].tolist()
        if seg_keys:
            segment_keys[str(seg)] = seg_keys

    # Summary statistics
    n_individual = len(individual_keys)
    n_segment_models = len(segment_keys)
    n_pooled = n_keys - n_individual

    # Count keys by demand pattern
    demand_pattern_counts = {}
    if 'demand_pattern' in df.columns:
        demand_pattern_counts = df['demand_pattern'].value_counts().to_dict()

    # Count segment outliers
    n_segment_outliers = int(df['_is_segment_outlier'].sum()) if '_is_segment_outlier' in df.columns else 0

    # Count data insufficient keys
    n_data_insufficient = int((~df['_data_sufficient']).sum()) if '_data_sufficient' in df.columns else 0

    summary = {
        'total_keys': n_keys,
        'individual_models': n_individual,
        'individual_pct': n_individual / n_keys * 100,
        'segment_models': n_segment_models,
        'pooled_keys': n_pooled,
        'avg_segment_size': n_pooled / max(n_segment_models, 1),
        'score_distribution': {
            'mean': df['_individual_score'].mean(),
            'median': df['_individual_score'].median(),
            'std': df['_individual_score'].std(),
            'min': df['_individual_score'].min(),
            'max': df['_individual_score'].max(),
        },
        'volume_overrides': int(df['_volume_override'].sum()),
        'forecastability_overrides': int(df['_forecastability_override'].sum()),
        'top_volume_bypass_keys': int(df.get('_top_volume_bypass', pd.Series(False, index=df.index)).sum()),
        # NEW: Enhanced allocation metrics
        'data_insufficient_keys': n_data_insufficient,
        'segment_outliers_identified': n_segment_outliers,
        'demand_pattern_distribution': demand_pattern_counts,
        'allocation_params': {
            'min_individual_score': min_individual_score,
            'max_individual_pct': max_individual_pct,
            'min_segment_size': min_segment_size,
            'volume_override_quantile': volume_override_quantile,
            'forecastability_override': forecastability_override,
            'min_obs_for_individual': _thresholds['min_obs'] if n_obs_col and n_obs_col in df.columns else None,
            'min_nonzero_for_individual': (
                max(int(_thresholds['min_nonzero']), int(min_nonzero_obs_for_individual))
                if n_obs_col and n_obs_col in df.columns else None
            ),
            'max_zero_fraction_for_individual': max_zero_fraction_for_individual,
            'top_volume_bypass_quantile': top_volume_bypass_quantile,
            'time_format': time_format,
        }
    }

    # Human-readable rationale
    allocation_rationale = (
        f"Model Level Allocation Summary (Enhanced with Segment-Relative Scoring):\n"
        f"- Total keys: {n_keys}\n"
        f"- Individual models: {n_individual} ({n_individual/n_keys*100:.1f}%)\n"
        f"- Segment models: {n_segment_models} (covering {n_pooled} pooled keys)\n"
        f"- Volume overrides: {summary['volume_overrides']}\n"
        f"- Forecastability overrides: {summary['forecastability_overrides']}\n"
        f"- Data insufficient (blocked): {n_data_insufficient}\n"
        f"- Segment outliers identified: {n_segment_outliers}\n"
        f"- Average segment size: {summary['avg_segment_size']:.1f}\n"
        f"- Score distribution: mean={summary['score_distribution']['mean']:.3f}, "
        f"median={summary['score_distribution']['median']:.3f}\n"
        f"- Demand patterns: {demand_pattern_counts}"
    )

    if verbose:
        logger.info(allocation_rationale)

    # Clean output DataFrame
    output_cols = [key_col, segment_col, 'model_level', 'model_strategy',
                   'allocation_rationale', 'allocation_confidence']
    if 'demand_pattern' in df.columns:
        output_cols.append('demand_pattern')
    if 'intermittency_class' in df.columns:
        output_cols.append('intermittency_class')

    allocations_df = df[[c for c in output_cols if c in df.columns]].copy()

    return ModelLevelAllocationResult(
        allocations=allocations_df,
        summary=summary,
        individual_keys=individual_keys,
        segment_keys=segment_keys,
        allocation_rationale=allocation_rationale,
    )


# =============================================================================
# MODEL-LEVEL-AWARE FEATURE SELECTION
# =============================================================================
#
# PHILOSOPHY:
# Individual key models and segment models have fundamentally different needs:
#
# INDIVIDUAL MODELS (key-specific):
#   - PRIORITIZE: Key's own historical patterns (lags, rolling stats)
#   - PRIORITIZE: Key-specific external features (price, promo when key-specific)
#   - DE-PRIORITIZE: Cross-segment aggregation features
#   - DE-PRIORITIZE: Segment-level statistics (mean, percentile, etc.)
#   - RATIONALE: The model sees only one key's data, cross-key features add noise
#
# SEGMENT MODELS (pooled keys):
#   - PRIORITIZE: Cross-key segment features (segment means, percentiles)
#   - PRIORITIZE: Key identifiers for within-segment discrimination
#   - INCLUDE: Key-level lags and rolling stats
#   - INCLUDE: Features that help distinguish keys within segment
#   - RATIONALE: Model learns patterns across similar keys, needs discriminators
#
# =============================================================================

@dataclass
class ModelLevelFeatureConfig:
    """
    Configuration for model-level-specific feature selection.

    This determines which features are used for individual vs segment models.
    """
    model_level: str  # 'individual' or 'segment' or specific model_group name
    model_strategy: str  # 'individual_score', 'individual_volume', 'segment_pooled', etc.

    # Feature category priorities (1=highest, 5=lowest, 0=exclude)
    lag_priority: int = 1        # Target lag features (lag_1, lag_2, etc.)
    rolling_priority: int = 1    # Rolling statistics (rolling_mean_4w, etc.)
    calendar_priority: int = 2   # Calendar features (month, week, quarter)
    external_priority: int = 2   # External features (price, promo, weather)
    intermittency_priority: int = 2  # Intermittency features (zeros handling)
    segment_agg_priority: int = 3    # Segment-level aggregations
    key_identifier_priority: int = 4  # Key identifiers (for segment models)

    # Maximum features per category
    max_lag_features: int = 20
    max_rolling_features: int = 25
    max_calendar_features: int = 15
    max_external_features: int = 40
    max_intermittency_features: int = 10
    max_segment_agg_features: int = 0  # 0 for individual, >0 for segment
    max_key_identifier_features: int = 0  # 0 for individual, >0 for segment

    # Total max features
    max_total_features: int = 100


def get_feature_config_for_model_level(
    model_strategy: str,
    is_individual: bool = None,
    n_keys_in_group: int = 1,
    demand_pattern: str = None,
) -> ModelLevelFeatureConfig:
    """
    Get feature configuration based on model strategy AND demand pattern.

    This function now considers DEMAND PATTERN to adjust feature priorities:
    - Smooth/Erratic: Standard lag/rolling features, lower intermittency priority
    - Intermittent/Lumpy: Higher intermittency priority, adjusted lag features

    Parameters
    ----------
    model_strategy : str
        The model strategy: 'individual_score', 'individual_volume',
        'individual_forecastability', 'segment_pooled', 'hierarchical'
    is_individual : bool, optional
        Override to force individual or segment config
    n_keys_in_group : int
        Number of keys in the model group (used for segment sizing)
    demand_pattern : str, optional
        Demand pattern: 'smooth', 'erratic', 'intermittent', 'lumpy'
        Used to adjust feature priorities based on data characteristics.

    Returns
    -------
    ModelLevelFeatureConfig
        Configuration for feature selection
    """
    # Determine if this is an individual or segment model
    if is_individual is None:
        is_individual = model_strategy.startswith('individual')

    # Normalize demand pattern
    pattern = (demand_pattern or '').lower()
    is_sparse = pattern in ['intermittent', 'lumpy']
    is_smooth = pattern in ['smooth']
    is_erratic = pattern in ['erratic']

    if is_individual:
        # =======================================================================
        # INDIVIDUAL MODEL CONFIG
        # Focus on key's own patterns, minimize cross-key features
        # Adjust based on demand pattern
        # =======================================================================

        # Base priorities
        lag_priority = 1
        rolling_priority = 1
        calendar_priority = 2
        external_priority = 2
        intermittency_priority = 2

        # Base limits
        max_lag = 25
        max_rolling = 30
        max_calendar = 12
        max_external = 35
        max_intermittency = 10

        # Adjust for demand pattern
        if is_sparse:
            # Intermittent/Lumpy: Prioritize intermittency features
            intermittency_priority = 1  # Elevate to top priority
            max_intermittency = 15  # More intermittency features
            max_lag = 20  # Fewer lag features (less useful for sparse data)
            max_rolling = 20  # Fewer rolling (sparse data = noisy rolling stats)
            logger.debug(f"Individual model: Sparse pattern detected, boosting intermittency features")
        elif is_smooth:
            # Smooth: Standard approach, minimize intermittency
            intermittency_priority = 4  # Lower priority
            max_intermittency = 5  # Fewer intermittency features
            max_lag = 30  # More lag features (very useful for smooth series)
            max_rolling = 35  # More rolling (reliable for smooth series)
            logger.debug(f"Individual model: Smooth pattern detected, boosting lag/rolling features")
        elif is_erratic:
            # Erratic: More rolling features to smooth volatility
            rolling_priority = 1
            max_rolling = 35
            max_lag = 20  # Lags less useful for erratic
            logger.debug(f"Individual model: Erratic pattern detected, boosting rolling features")

        return ModelLevelFeatureConfig(
            model_level='individual',
            model_strategy=model_strategy,

            lag_priority=lag_priority,
            rolling_priority=rolling_priority,
            calendar_priority=calendar_priority,
            external_priority=external_priority,
            intermittency_priority=intermittency_priority,

            # LOW/NO priority for cross-key features
            segment_agg_priority=0,  # EXCLUDE segment aggregations
            key_identifier_priority=0,  # EXCLUDE key identifiers (model sees only one key)

            # Feature limits adjusted by pattern
            max_lag_features=max_lag,
            max_rolling_features=max_rolling,
            max_calendar_features=max_calendar,
            max_external_features=max_external,
            max_intermittency_features=max_intermittency,
            max_segment_agg_features=0,  # NO segment aggregations
            max_key_identifier_features=0,  # NO key identifiers

            max_total_features=100,
        )
    else:
        # =======================================================================
        # SEGMENT MODEL CONFIG
        # Include cross-key features and key discriminators
        # Adjust based on segment's dominant demand pattern
        # =======================================================================
        # Scale features based on segment size
        segment_scaling = min(n_keys_in_group / 50, 1.5)  # Scale up for larger segments

        # Base priorities
        lag_priority = 1
        rolling_priority = 1
        calendar_priority = 2
        external_priority = 2
        intermittency_priority = 2
        segment_agg_priority = 3
        key_identifier_priority = 3

        # Base limits
        max_lag = 20
        max_rolling = 20
        max_calendar = 10
        max_external = 30
        max_intermittency = 8

        # Adjust for demand pattern
        if is_sparse:
            # Intermittent/Lumpy segment: Prioritize intermittency
            intermittency_priority = 1  # Elevate to top priority
            max_intermittency = 12  # More intermittency features
            max_lag = 15  # Fewer lag features
            max_rolling = 15  # Fewer rolling features
            logger.debug(f"Segment model: Sparse pattern detected, boosting intermittency features")
        elif is_smooth:
            # Smooth segment: Standard approach
            intermittency_priority = 4  # Lower priority
            max_intermittency = 4  # Fewer intermittency features
            max_lag = 25  # More lag features
            max_rolling = 25  # More rolling features
            logger.debug(f"Segment model: Smooth pattern detected, boosting lag/rolling features")
        elif is_erratic:
            # Erratic segment: More rolling to capture volatility
            rolling_priority = 1
            max_rolling = 25
            max_lag = 15
            logger.debug(f"Segment model: Erratic pattern detected, boosting rolling features")

        return ModelLevelFeatureConfig(
            model_level='segment',
            model_strategy=model_strategy,

            lag_priority=lag_priority,
            rolling_priority=rolling_priority,
            calendar_priority=calendar_priority,
            external_priority=external_priority,
            intermittency_priority=intermittency_priority,
            segment_agg_priority=segment_agg_priority,
            key_identifier_priority=key_identifier_priority,

            # Feature limits adjusted by pattern
            max_lag_features=max_lag,
            max_rolling_features=max_rolling,
            max_calendar_features=max_calendar,
            max_external_features=max_external,
            max_intermittency_features=max_intermittency,
            max_segment_agg_features=int(15 * segment_scaling),  # Scale with segment size
            max_key_identifier_features=int(5 * segment_scaling),  # Key discriminators

            max_total_features=120,  # Can use more features with pooled data
        )


def categorize_features_for_model_level(
    feature_cols: List[str],
    key_col: str = 'key',
) -> Dict[str, List[str]]:
    """
    Categorize features for model-level-aware selection.

    This categorization is specifically designed to separate:
    1. Key-specific features (useful for individual models)
    2. Cross-key/segment features (useful for segment models)

    Parameters
    ----------
    feature_cols : List[str]
        All available feature columns
    key_col : str
        Name of the key column (to identify key identifiers)

    Returns
    -------
    Dict[str, List[str]]
        Features grouped by category:
        - 'lag_features': Target lag features
        - 'rolling_features': Rolling statistics
        - 'calendar_features': Calendar/time features
        - 'external_features': External predictors (price, promo, weather, etc.)
        - 'intermittency_features': Intermittency handling features
        - 'segment_agg_features': Segment-level aggregations
        - 'key_identifier_features': Key identifiers for segment models
        - 'other_features': Uncategorized features
    """
    categories = {
        'lag_features': [],
        'rolling_features': [],
        'calendar_features': [],
        'external_features': [],
        'intermittency_features': [],
        'segment_agg_features': [],
        'key_identifier_features': [],
        'other_features': [],
    }

    for col in feature_cols:
        col_lower = col.lower()

        # =====================================================================
        # LAG FEATURES - Key's own historical values
        # =====================================================================
        if any(p in col_lower for p in [
            'lag_', '_lag_', 'lag1', 'lag2', 'lag4', 'lag13', 'lag26', 'lag52',
            '_t-1', '_t-2', '_t-4', '_t-52',
            'yoy_', 'year_over_year', 'qoq_', 'quarter_over_quarter',
        ]):
            categories['lag_features'].append(col)

        # =====================================================================
        # ROLLING FEATURES - Key's rolling statistics
        # =====================================================================
        elif any(p in col_lower for p in [
            'rolling_', 'roll_', 'ewm_', 'ema_', 'ewma_',
            '_mean_', '_std_', '_min_', '_max_', '_sum_',
            '_4w', '_13w', '_26w', '_52w',
            'window_', 'moving_avg', 'moving_std',
        ]):
            categories['rolling_features'].append(col)

        # =====================================================================
        # CALENDAR FEATURES - Time-based patterns
        # =====================================================================
        elif any(p in col_lower for p in [
            'month', 'week', 'year', 'quarter', 'day_of',
            'is_month_', 'is_quarter_', 'is_year_',
            'sin_', 'cos_',  # Cyclical encodings
            'fourier_',  # Fourier features
            'season', 'holiday',
        ]):
            categories['calendar_features'].append(col)

        # =====================================================================
        # INTERMITTENCY FEATURES - Zero/sparse demand handling
        # =====================================================================
        elif any(p in col_lower for p in [
            'zero_', 'nonzero', 'demand_occur', 'periods_since',
            'adi_', 'demand_freq', 'intermit',
            'last_nonzero', 'time_since_last',
        ]):
            categories['intermittency_features'].append(col)

        # =====================================================================
        # SEGMENT AGGREGATION FEATURES - Cross-key features (for segment models)
        # =====================================================================
        elif any(p in col_lower for p in [
            'segment_', 'cluster_', 'group_',
            '_percentile', '_quantile',
            '_segment_mean', '_segment_std', '_segment_median',
            '_cluster_mean', '_cluster_std',
            '_cohort_', '_peer_',
            'relative_to_segment', 'vs_segment',
        ]):
            categories['segment_agg_features'].append(col)

        # =====================================================================
        # KEY IDENTIFIER FEATURES - For segment models to distinguish keys
        # =====================================================================
        elif any(p in col_lower for p in [
            'key_', key_col.lower(),
            '_encoded', '_target_enc', '_hash',
            'entity_', 'sku_', 'product_', 'customer_',
            # Categorical encodings of key identifiers
        ]) and not any(p in col_lower for p in ['lag_', 'rolling_', 'mean_', 'std_']):
            categories['key_identifier_features'].append(col)

        # =====================================================================
        # EXTERNAL FEATURES - Price, promo, weather, etc.
        # =====================================================================
        elif any(p in col_lower for p in [
            'price', 'promo', 'discount', 'promotion',
            'weather', 'temp', 'wthr_',
            'pos_', 'nielsen_', 'distribution',
            'store', 'feature_', 'display',
            'shipment', 'mechanic', 'objective',
            'h_', 'holiday', 'sunshine',
        ]):
            categories['external_features'].append(col)

        # =====================================================================
        # OTHER FEATURES - Uncategorized
        # =====================================================================
        else:
            categories['other_features'].append(col)

    return categories


def select_features_for_model_level(
    all_features: List[str],
    model_strategy: str,
    n_keys_in_group: int = 1,
    key_col: str = 'key',
    feature_importance: Optional[Dict[str, float]] = None,
    correlation_with_target: Optional[Dict[str, float]] = None,
    demand_pattern: str = None,
    verbose: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Select features appropriate for the model level (individual vs segment).

    This is the main entry point for model-level-aware feature selection.
    Now includes DEMAND PATTERN awareness to adjust feature priorities.

    Parameters
    ----------
    all_features : List[str]
        All available feature columns
    model_strategy : str
        Model strategy: 'individual_score', 'individual_volume',
        'individual_forecastability', 'segment_pooled', 'hierarchical'
    n_keys_in_group : int
        Number of keys in the model group
    key_col : str
        Name of the key column
    feature_importance : Dict[str, float], optional
        Pre-computed feature importances (from previous model or EDA)
    correlation_with_target : Dict[str, float], optional
        Pre-computed correlations with target
    demand_pattern : str, optional
        Demand pattern: 'smooth', 'erratic', 'intermittent', 'lumpy'
        Used to adjust feature priorities:
        - Smooth: More lag/rolling features, fewer intermittency
        - Intermittent/Lumpy: More intermittency features, fewer lag/rolling
        - Erratic: More rolling features to smooth volatility
    verbose : bool
        Print selection details

    Returns
    -------
    Tuple[List[str], Dict[str, Any]]
        - selected_features: List of selected feature names
        - selection_metadata: Details about the selection process
    """
    # Get configuration for this model level AND demand pattern
    config = get_feature_config_for_model_level(
        model_strategy=model_strategy,
        n_keys_in_group=n_keys_in_group,
        demand_pattern=demand_pattern,
    )

    # Categorize features
    categories = categorize_features_for_model_level(all_features, key_col)

    # Selection metadata
    metadata = {
        'model_strategy': model_strategy,
        'demand_pattern': demand_pattern,
        'is_individual': config.model_level == 'individual',
        'n_keys_in_group': n_keys_in_group,
        'total_features_available': len(all_features),
        'features_by_category': {k: len(v) for k, v in categories.items()},
        'config': {
            'max_total_features': config.max_total_features,
            'segment_agg_priority': config.segment_agg_priority,
            'key_identifier_priority': config.key_identifier_priority,
            'intermittency_priority': config.intermittency_priority,
            'pattern_adjusted': demand_pattern is not None,
        },
    }

    # =========================================================================
    # PRIORITY-BASED SELECTION
    # =========================================================================
    selected = []
    selection_details = {}

    # Define selection order based on priority
    category_config = [
        ('lag_features', config.lag_priority, config.max_lag_features),
        ('rolling_features', config.rolling_priority, config.max_rolling_features),
        ('calendar_features', config.calendar_priority, config.max_calendar_features),
        ('external_features', config.external_priority, config.max_external_features),
        ('intermittency_features', config.intermittency_priority, config.max_intermittency_features),
        ('segment_agg_features', config.segment_agg_priority, config.max_segment_agg_features),
        ('key_identifier_features', config.key_identifier_priority, config.max_key_identifier_features),
        ('other_features', 5, 10),  # Low priority catchall
    ]

    # Sort by priority (lower number = higher priority)
    category_config.sort(key=lambda x: (x[1], -x[2]))  # Primary: priority, Secondary: max features

    for cat_name, priority, max_features in category_config:
        if priority == 0:
            # Priority 0 = exclude entirely
            selection_details[cat_name] = {
                'available': len(categories[cat_name]),
                'selected': 0,
                'excluded': True,
                'reason': 'Priority 0 (excluded for this model level)',
            }
            continue

        cat_features = categories[cat_name]

        # If we have feature importance, use it to rank within category
        if feature_importance and cat_features:
            cat_features = sorted(
                cat_features,
                key=lambda f: feature_importance.get(f, 0),
                reverse=True
            )
        elif correlation_with_target and cat_features:
            cat_features = sorted(
                cat_features,
                key=lambda f: abs(correlation_with_target.get(f, 0)),
                reverse=True
            )

        # Select up to max_features
        to_select = cat_features[:max_features]
        selected.extend(to_select)

        selection_details[cat_name] = {
            'available': len(categories[cat_name]),
            'selected': len(to_select),
            'priority': priority,
            'max_allowed': max_features,
        }

    # =========================================================================
    # ENFORCE TOTAL LIMIT
    # =========================================================================
    if len(selected) > config.max_total_features:
        # Already sorted by priority, just truncate
        selected = selected[:config.max_total_features]
        metadata['truncated'] = True
        metadata['truncated_from'] = len(selected)

    # Remove duplicates while preserving order
    seen = set()
    selected_unique = []
    for f in selected:
        if f not in seen:
            seen.add(f)
            selected_unique.append(f)
    selected = selected_unique

    metadata['n_features_selected'] = len(selected)
    metadata['selection_details'] = selection_details

    if verbose:
        logger.info(f"Feature selection for {model_strategy} ({config.model_level} model):")
        logger.info(f"  Total available: {len(all_features)}")
        logger.info(f"  Selected: {len(selected)}")
        for cat_name, details in selection_details.items():
            logger.info(f"  - {cat_name}: {details['selected']}/{details['available']}")

    return selected, metadata


def get_excluded_features_for_individual_models(
    all_features: List[str],
    key_col: str = 'key',
) -> Tuple[List[str], List[str]]:
    """
    Identify features that should be EXCLUDED for individual key models.

    These are features that are only meaningful in a cross-key context and
    would add noise to individual key models.

    Parameters
    ----------
    all_features : List[str]
        All available feature columns
    key_col : str
        Name of the key column

    Returns
    -------
    Tuple[List[str], List[str]]
        - features_to_exclude: Features to exclude for individual models
        - features_to_keep: Features to keep for individual models
    """
    categories = categorize_features_for_model_level(all_features, key_col)

    # Features to exclude for individual models
    exclude_categories = ['segment_agg_features', 'key_identifier_features']

    features_to_exclude = []
    for cat in exclude_categories:
        features_to_exclude.extend(categories[cat])

    features_to_keep = [f for f in all_features if f not in features_to_exclude]

    return features_to_exclude, features_to_keep


def get_segment_specific_features(
    all_features: List[str],
    key_col: str = 'key',
) -> List[str]:
    """
    Identify features that are ONLY useful for segment models.

    These features help the segment model distinguish between keys within
    the segment and leverage cross-key patterns.

    Parameters
    ----------
    all_features : List[str]
        All available feature columns
    key_col : str
        Name of the key column

    Returns
    -------
    List[str]
        Features that are specifically useful for segment models
    """
    categories = categorize_features_for_model_level(all_features, key_col)

    segment_specific = []
    segment_specific.extend(categories['segment_agg_features'])
    segment_specific.extend(categories['key_identifier_features'])

    return segment_specific


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Model configuration
    'ALLOWED_MODELS',
    'BANNED_MODELS',
    'MODEL_PRIORITY',
    'ModelType',

    # Key classification
    'ModelingStrategy',
    'KeyCharacteristics',
    'FeatureCharacteristics',  # NEW: Feature-level characteristics
    'ModelingDecision',
    'classify_key_modeling_strategy',
    'compute_key_characteristics',
    'compute_feature_characteristics',  # NEW: Compute feature characteristics
    'compute_key_characteristics_with_features',  # NEW: Full characteristics

    # STATE-OF-THE-ART Model Level Allocation
    'ModelLevelAllocationResult',
    'compute_individual_model_score',
    'allocate_model_levels',
    'get_obs_thresholds',

    # Feature selection
    'FeatureSelectionResult',
    'select_features_tree_importance',
    'select_features_correlation',
    'select_features_recursive_elimination',
    'select_features_hybrid',

    # Feature categorization
    'categorize_features',
    'get_priority_features',

    # MODEL-LEVEL-AWARE FEATURE SELECTION (NEW)
    'ModelLevelFeatureConfig',
    'get_feature_config_for_model_level',
    'categorize_features_for_model_level',
    'select_features_for_model_level',
    'get_excluded_features_for_individual_models',
    'get_segment_specific_features',
]
