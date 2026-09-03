# utils/state_of_art_training.py
"""
State-of-the-Art Training Enhancements for Demand Forecasting.

This module implements cutting-edge training capabilities that leverage
ALL upstream context (EDA, Segmentation, Features) for optimal model training.

STATE-OF-THE-ART FEATURES:
==========================
1. SEGMENT-SPECIFIC HYPERPARAMETERS: Hyperparameters derived from segment characteristics
2. EDA-ENHANCED META-LEARNING: Meta-features include EDA insights for smarter model selection
3. PATTERN-AWARE ENSEMBLE WEIGHTING: Ensemble weights consider demand patterns
4. ADAPTIVE ZERO-INFLATED THRESHOLD CALIBRATION: Learn optimal thresholds from validation
5. SEGMENT-AWARE BIAS CALIBRATION: Per-segment calibration factors

USAGE:
------
>>> from utils.state_of_art_training import (
...     # Segment-specific hyperparameters
...     get_segment_specific_hyperparams,
...     HyperparameterProfile,
...
...     # Enhanced meta-learning
...     build_enhanced_meta_features,
...     EnhancedMetaFeatures,
...
...     # Pattern-aware ensemble
...     compute_pattern_aware_ensemble_weights,
...
...     # Adaptive threshold calibration
...     calibrate_zero_inflated_threshold,
...     optimize_hurdle_threshold,
...
...     # Segment-aware bias calibration
...     learn_segment_bias_calibration,
...     apply_segment_bias_calibration,
... )

Author: FEU-Agentic-Forecasting Demand Intelligence
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HyperparameterProfile:
    """
    Segment-specific hyperparameter profile derived from data characteristics.

    This profile is computed from:
    - Segment statistics (zero_fraction, CV, skewness, series_length)
    - EDA insights (seasonality, trend, changepoints)
    - Demand pattern classification
    """
    # Segment identifier
    segment_id: str
    demand_pattern: str  # smooth, erratic, intermittent, lumpy

    # LightGBM hyperparameters
    lgb_num_leaves: int = 31
    lgb_min_child_samples: int = 20
    lgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 500
    lgb_reg_lambda: float = 0.1
    lgb_reg_alpha: float = 0.1
    lgb_max_depth: int = 8
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8

    # XGBoost hyperparameters
    xgb_max_depth: int = 8
    xgb_min_child_weight: int = 3
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 500
    xgb_reg_lambda: float = 0.1
    xgb_reg_alpha: float = 0.1
    xgb_gamma: float = 0.0
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # CatBoost hyperparameters
    cat_depth: int = 8
    cat_learning_rate: float = 0.05
    cat_iterations: int = 500
    cat_l2_leaf_reg: float = 3.0

    # Random Forest hyperparameters
    rf_n_estimators: int = 300
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2

    # Zero-Inflated / Hurdle model hyperparameters
    zi_threshold: float = 0.5  # Probability threshold for zero prediction
    hurdle_threshold: float = 0.5

    # Training configuration
    early_stopping_rounds: int = 50

    # Metadata
    derivation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'segment_id': self.segment_id,
            'demand_pattern': self.demand_pattern,
            'lightgbm': {
                'num_leaves': self.lgb_num_leaves,
                'min_child_samples': self.lgb_min_child_samples,
                'learning_rate': self.lgb_learning_rate,
                'n_estimators': self.lgb_n_estimators,
                'reg_lambda': self.lgb_reg_lambda,
                'reg_alpha': self.lgb_reg_alpha,
                'max_depth': self.lgb_max_depth,
                'subsample': self.lgb_subsample,
                'colsample_bytree': self.lgb_colsample_bytree,
            },
            'xgboost': {
                'max_depth': self.xgb_max_depth,
                'min_child_weight': self.xgb_min_child_weight,
                'learning_rate': self.xgb_learning_rate,
                'n_estimators': self.xgb_n_estimators,
                'reg_lambda': self.xgb_reg_lambda,
                'reg_alpha': self.xgb_reg_alpha,
                'gamma': self.xgb_gamma,
                'subsample': self.xgb_subsample,
                'colsample_bytree': self.xgb_colsample_bytree,
            },
            'catboost': {
                'depth': self.cat_depth,
                'learning_rate': self.cat_learning_rate,
                'iterations': self.cat_iterations,
                'l2_leaf_reg': self.cat_l2_leaf_reg,
            },
            'random_forest': {
                'n_estimators': self.rf_n_estimators,
                'max_depth': self.rf_max_depth,
                'min_samples_split': self.rf_min_samples_split,
                'min_samples_leaf': self.rf_min_samples_leaf,
            },
            'zero_inflated': {
                'threshold': self.zi_threshold,
            },
            'hurdle_model': {
                'threshold': self.hurdle_threshold,
            },
            'early_stopping_rounds': self.early_stopping_rounds,
            'derivation_reason': self.derivation_reason,
        }

    def get_params_for_model(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters for a specific model type."""
        if model_type == 'lightgbm':
            return {
                'num_leaves': self.lgb_num_leaves,
                'min_child_samples': self.lgb_min_child_samples,
                'learning_rate': self.lgb_learning_rate,
                'n_estimators': self.lgb_n_estimators,
                'reg_lambda': self.lgb_reg_lambda,
                'reg_alpha': self.lgb_reg_alpha,
                'max_depth': self.lgb_max_depth,
                'subsample': self.lgb_subsample,
                'colsample_bytree': self.lgb_colsample_bytree,
            }
        elif model_type == 'xgboost':
            return {
                'max_depth': self.xgb_max_depth,
                'min_child_weight': self.xgb_min_child_weight,
                'learning_rate': self.xgb_learning_rate,
                'n_estimators': self.xgb_n_estimators,
                'reg_lambda': self.xgb_reg_lambda,
                'reg_alpha': self.xgb_reg_alpha,
                'gamma': self.xgb_gamma,
                'subsample': self.xgb_subsample,
                'colsample_bytree': self.xgb_colsample_bytree,
            }
        elif model_type == 'catboost':
            return {
                'depth': self.cat_depth,
                'learning_rate': self.cat_learning_rate,
                'iterations': self.cat_iterations,
                'l2_leaf_reg': self.cat_l2_leaf_reg,
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': self.rf_n_estimators,
                'max_depth': self.rf_max_depth,
                'min_samples_split': self.rf_min_samples_split,
                'min_samples_leaf': self.rf_min_samples_leaf,
            }
        else:
            return {}


@dataclass
class EnhancedMetaFeatures:
    """
    Enhanced meta-features that include EDA insights for intelligent model selection.

    This extends the basic MetaFeatures with:
    - EDA-derived characteristics (seasonality strength, trend coefficient)
    - Segment-specific statistics
    - Feature quality indicators
    """
    # Basic statistics (from time series)
    n_observations: int
    n_nonzero: int
    zero_fraction: float
    mean: float
    std: float
    cv: float  # Coefficient of variation

    # Demand pattern classification
    demand_pattern: str  # smooth, erratic, intermittent, lumpy
    adi: float  # Average Demand Interval
    cv2: float  # Squared CV (for Syntetos-Boylan)

    # Time series characteristics
    trend_strength: float
    autocorr_lag1: float
    autocorr_lag7: float
    entropy: float
    skewness: float
    kurtosis: float

    # ======== NEW: EDA-derived features ========
    seasonality_strength: float = 0.0  # From EDA
    seasonal_period: int = 12  # Detected period (default 12; overridden by EDA)
    has_strong_seasonality: bool = False
    trend_coefficient: float = 0.0  # Linear trend slope
    changepoint_count: int = 0  # Number of detected changepoints
    changepoint_pct: float = 0.0  # % of series with changepoints

    # ======== NEW: Segment-specific features ========
    segment_zero_fraction: float = 0.0
    segment_cv: float = 1.0
    segment_skewness: float = 0.0
    segment_avg_volume: float = 0.0
    segment_size: int = 1  # Number of keys in segment
    expected_difficulty: str = 'medium'  # low, medium, high

    # ======== NEW: Feature quality indicators ========
    n_features: int = 0
    has_external_features: bool = False
    feature_quality_score: float = 0.5
    n_lag_features: int = 0
    n_rolling_features: int = 0
    n_calendar_features: int = 0
    acf_informed_lags: bool = False

    # Derived scores
    forecastability: float = 0.5
    model_complexity_recommendation: str = 'medium'  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_observations': self.n_observations,
            'n_nonzero': self.n_nonzero,
            'zero_fraction': self.zero_fraction,
            'mean': self.mean,
            'std': self.std,
            'cv': self.cv,
            'demand_pattern': self.demand_pattern,
            'adi': self.adi,
            'cv2': self.cv2,
            'trend_strength': self.trend_strength,
            'autocorr_lag1': self.autocorr_lag1,
            'autocorr_lag7': self.autocorr_lag7,
            'entropy': self.entropy,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            # EDA features
            'seasonality_strength': self.seasonality_strength,
            'seasonal_period': self.seasonal_period,
            'has_strong_seasonality': self.has_strong_seasonality,
            'trend_coefficient': self.trend_coefficient,
            'changepoint_count': self.changepoint_count,
            'changepoint_pct': self.changepoint_pct,
            # Segment features
            'segment_zero_fraction': self.segment_zero_fraction,
            'segment_cv': self.segment_cv,
            'segment_skewness': self.segment_skewness,
            'segment_avg_volume': self.segment_avg_volume,
            'segment_size': self.segment_size,
            'expected_difficulty': self.expected_difficulty,
            # Feature quality
            'n_features': self.n_features,
            'has_external_features': self.has_external_features,
            'feature_quality_score': self.feature_quality_score,
            'n_lag_features': self.n_lag_features,
            'n_rolling_features': self.n_rolling_features,
            'n_calendar_features': self.n_calendar_features,
            'acf_informed_lags': self.acf_informed_lags,
            # Derived
            'forecastability': self.forecastability,
            'model_complexity_recommendation': self.model_complexity_recommendation,
        }


@dataclass
class SegmentBiasCalibration:
    """
    Per-segment bias calibration factors learned from validation data.
    """
    segment_id: str
    demand_pattern: str

    # Calibration factors by zero_fraction bucket
    calibration_factors: Dict[str, float] = field(default_factory=dict)

    # Global segment calibration factor
    global_factor: float = 1.0

    # Statistics
    n_keys_calibrated: int = 0
    avg_bias_before: float = 0.0  # Mean(predicted - actual) / Mean(actual)
    avg_bias_after: float = 0.0

    # Bucket definitions
    bucket_edges: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0])


# =============================================================================
# UNIVERSAL MODEL PREDICTION HELPER
# =============================================================================

def _predict_model(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Get predictions from any model type (sklearn-style or univariate dict).

    Univariate dict models (croston, sba, tsb, imapa) have no .predict()
    method — they store a constant 'forecast' value. This helper returns
    that constant broadcast to the right length so ensemble code works.

    Parameters
    ----------
    model : Any
        Trained model object or univariate dict
    X : np.ndarray
        Feature matrix (used for length and by sklearn-style models)

    Returns
    -------
    np.ndarray
        Predictions array of length len(X)
    """
    if isinstance(model, dict) and 'forecast' in model:
        return np.full(len(X), float(model['forecast']))
    return model.predict(X)


# =============================================================================
# PATTERN-AWARE ENSEMBLE CLASS (Module-level for pickle serialization)
# =============================================================================

class PatternAwareEnsemble:
    """
    Ensemble model that uses pattern-aware weighting.

    This class is defined at module level to ensure it can be pickled/serialized
    by joblib.dump() when saving trained models. Nested class definitions inside
    functions cannot be pickled by Python's pickle module.
    """

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        """
        Initialize the pattern-aware ensemble.

        Parameters
        ----------
        models : Dict[str, Any]
            Dictionary mapping model names to trained model objects
        weights : Dict[str, float]
            Dictionary mapping model names to their ensemble weights
        """
        self.models = models
        self.weights = weights
        self.model_type = 'pattern_aware_ensemble'

    def predict(self, X) -> np.ndarray:
        """
        Generate predictions using weighted ensemble of models.

        Parameters
        ----------
        X : array-like
            Feature matrix for prediction

        Returns
        -------
        np.ndarray
            Weighted ensemble predictions (clipped to non-negative)
        """
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 0)
            if w > 0:
                preds += w * np.clip(_predict_model(model, X), 0, None)
        return preds

    def __repr__(self) -> str:
        """String representation for debugging."""
        model_names = list(self.models.keys())
        return f"PatternAwareEnsemble(models={model_names}, weights={self.weights})"


# =============================================================================
# 1. SEGMENT-SPECIFIC HYPERPARAMETERS
# =============================================================================

def get_segment_specific_hyperparams(
    segment_profile: Dict[str, Any],
    eda_insights: Dict[str, Any],
    demand_pattern: str,
    n_features: int = 50,
    series_length: int = 100,
    time_format: str = 'year_week',
) -> HyperparameterProfile:
    """
    Generate hyperparameter profile from segment characteristics and EDA insights.

    This is a STATE-OF-THE-ART approach that derives hyperparameters from data
    rather than using one-size-fits-all defaults.

    Parameters
    ----------
    segment_profile : Dict
        Segment statistics from segmentation crew:
        - zero_fraction: Fraction of zeros in segment
        - cv: Coefficient of variation
        - skewness: Skewness of demand distribution
        - avg_volume: Average demand volume
        - n_keys: Number of keys in segment
    eda_insights : Dict
        EDA insights from EDA crew:
        - seasonality.avg_strength: Average seasonal strength
        - seasonality.dominant_period: Detected seasonal period
        - trend.avg_strength: Average trend strength
        - changepoints.pct_significant: % with significant changepoints
    demand_pattern : str
        'smooth', 'erratic', 'intermittent', or 'lumpy'
    n_features : int
        Number of features available
    series_length : int
        Average length of time series in segment

    Returns
    -------
    HyperparameterProfile
        Segment-specific hyperparameter configuration
    """
    # Extract segment characteristics
    zero_frac = segment_profile.get('zero_fraction', 0.0)
    cv = segment_profile.get('cv', segment_profile.get('coefficient_of_variation', 1.0))
    skewness = segment_profile.get('skewness', 0.0)
    avg_volume = segment_profile.get('avg_volume', segment_profile.get('mean_volume', 100))
    n_keys = segment_profile.get('n_keys', segment_profile.get('segment_size', 1))
    segment_id = segment_profile.get('segment_id', 'unknown')

    # Extract EDA insights
    seasonality_info = eda_insights.get('seasonality', {})
    seasonal_strength = seasonality_info.get('avg_strength', seasonality_info.get('avg_seasonal_strength', 0.0))
    # Use time_format-aware default: 12 for monthly, 52 for weekly
    _default_period = 12 if time_format == 'year_month' else 52
    seasonal_period = seasonality_info.get('dominant_period', _default_period)

    trend_info = eda_insights.get('trend', {})
    trend_strength = trend_info.get('avg_strength', trend_info.get('avg_trend_strength', 0.0))

    changepoint_info = eda_insights.get('changepoints', {})
    changepoint_pct = changepoint_info.get('pct_significant', changepoint_info.get('pct_with_changepoints', 0.0))

    # Build derivation reason for transparency
    reasons = []

    # =========================================================================
    # LIGHTGBM HYPERPARAMETERS
    # =========================================================================

    # num_leaves: Smaller for sparse data, larger for complex patterns
    if zero_frac > 0.5:
        lgb_num_leaves = 15  # Sparse data needs simpler trees
        reasons.append(f"lgb_num_leaves=15 (zero_frac={zero_frac:.2f})")
    elif seasonal_strength > 0.3:
        lgb_num_leaves = 63  # Complex seasonality needs more leaves
        reasons.append(f"lgb_num_leaves=63 (seasonal_strength={seasonal_strength:.2f})")
    else:
        lgb_num_leaves = 31  # Default

    # min_child_samples: Higher for sparse data to prevent overfitting
    lgb_min_child_samples = max(10, min(50, int(zero_frac * 100)))
    if zero_frac > 0.3:
        reasons.append(f"lgb_min_child_samples={lgb_min_child_samples} (sparse data)")

    # learning_rate: Lower for volatile data, higher for smooth
    if demand_pattern in ['erratic', 'lumpy']:
        lgb_learning_rate = 0.02  # Slower learning for volatile patterns
        reasons.append(f"lgb_learning_rate=0.02 (volatile pattern: {demand_pattern})")
    elif series_length < 50:
        lgb_learning_rate = 0.03  # Slower for limited data
        reasons.append(f"lgb_learning_rate=0.03 (short series: {series_length})")
    else:
        lgb_learning_rate = 0.05

    # n_estimators: More for seasonal/complex data
    if seasonal_strength > 0.3 or trend_strength > 0.3:
        lgb_n_estimators = 800
        reasons.append(f"lgb_n_estimators=800 (seasonal/trend complexity)")
    elif series_length < 30:
        lgb_n_estimators = 300  # Less for short series
    else:
        lgb_n_estimators = 500

    # Regularization: Stronger for high-variance data
    lgb_reg_lambda = 0.1 * (1 + cv)
    lgb_reg_alpha = 0.05 * (1 + cv)
    if cv > 1.5:
        reasons.append(f"lgb_reg_lambda={lgb_reg_lambda:.2f} (high cv={cv:.2f})")

    # max_depth: Deeper for more data
    lgb_max_depth = min(10, max(4, int(np.log2(series_length))))

    # subsample/colsample: Lower for overfitting-prone scenarios
    if n_keys < 10:  # Small segment, higher overfitting risk
        lgb_subsample = 0.7
        lgb_colsample_bytree = 0.7
        reasons.append("lgb_subsample=0.7 (small segment)")
    else:
        lgb_subsample = 0.8
        lgb_colsample_bytree = 0.8

    # =========================================================================
    # XGBOOST HYPERPARAMETERS
    # =========================================================================

    # max_depth: Similar logic to LightGBM
    xgb_max_depth = lgb_max_depth

    # min_child_weight: Higher for sparse (analogous to min_child_samples)
    xgb_min_child_weight = max(1, int(zero_frac * 30))

    # learning_rate, n_estimators: Same as LightGBM
    xgb_learning_rate = lgb_learning_rate
    xgb_n_estimators = lgb_n_estimators

    # Regularization
    xgb_reg_lambda = lgb_reg_lambda
    xgb_reg_alpha = lgb_reg_alpha

    # gamma (min split loss): Higher for noisy data
    xgb_gamma = 0.1 if cv > 2.0 else 0.0
    if xgb_gamma > 0:
        reasons.append(f"xgb_gamma={xgb_gamma} (high variance)")

    # =========================================================================
    # CATBOOST HYPERPARAMETERS
    # =========================================================================

    cat_depth = lgb_max_depth
    cat_learning_rate = lgb_learning_rate
    cat_iterations = lgb_n_estimators
    cat_l2_leaf_reg = 3.0 * (1 + cv / 2)

    # =========================================================================
    # RANDOM FOREST HYPERPARAMETERS
    # =========================================================================

    rf_n_estimators = min(500, max(100, lgb_n_estimators // 2))
    rf_max_depth = min(20, lgb_max_depth * 2)  # RF can be deeper
    rf_min_samples_split = max(2, lgb_min_child_samples // 4)
    rf_min_samples_leaf = max(1, rf_min_samples_split // 2)

    # =========================================================================
    # ZERO-INFLATED / HURDLE THRESHOLDS
    # =========================================================================

    # Higher zero_fraction → need higher threshold to predict zeros
    if zero_frac >= 0.7:
        zi_threshold = 0.6
        hurdle_threshold = 0.65
        reasons.append(f"zi_threshold=0.6 (high zero_frac={zero_frac:.2f})")
    elif zero_frac >= 0.5:
        zi_threshold = 0.5
        hurdle_threshold = 0.55
    elif zero_frac >= 0.3:
        zi_threshold = 0.4
        hurdle_threshold = 0.45
    else:
        zi_threshold = 0.3
        hurdle_threshold = 0.35

    # =========================================================================
    # EARLY STOPPING
    # =========================================================================

    # More rounds for complex patterns, fewer for simple/short data
    if seasonal_strength > 0.3 or demand_pattern in ['erratic', 'lumpy']:
        early_stopping_rounds = 100
    elif series_length < 30:
        early_stopping_rounds = 30
    else:
        early_stopping_rounds = 50

    return HyperparameterProfile(
        segment_id=segment_id,
        demand_pattern=demand_pattern,
        lgb_num_leaves=lgb_num_leaves,
        lgb_min_child_samples=lgb_min_child_samples,
        lgb_learning_rate=lgb_learning_rate,
        lgb_n_estimators=lgb_n_estimators,
        lgb_reg_lambda=lgb_reg_lambda,
        lgb_reg_alpha=lgb_reg_alpha,
        lgb_max_depth=lgb_max_depth,
        lgb_subsample=lgb_subsample,
        lgb_colsample_bytree=lgb_colsample_bytree,
        xgb_max_depth=xgb_max_depth,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_learning_rate=xgb_learning_rate,
        xgb_n_estimators=xgb_n_estimators,
        xgb_reg_lambda=xgb_reg_lambda,
        xgb_reg_alpha=xgb_reg_alpha,
        xgb_gamma=xgb_gamma,
        xgb_subsample=lgb_subsample,
        xgb_colsample_bytree=lgb_colsample_bytree,
        cat_depth=cat_depth,
        cat_learning_rate=cat_learning_rate,
        cat_iterations=cat_iterations,
        cat_l2_leaf_reg=cat_l2_leaf_reg,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_split=rf_min_samples_split,
        rf_min_samples_leaf=rf_min_samples_leaf,
        zi_threshold=zi_threshold,
        hurdle_threshold=hurdle_threshold,
        early_stopping_rounds=early_stopping_rounds,
        derivation_reason="; ".join(reasons) if reasons else "default profile",
    )


def compute_all_segment_hyperparams(
    segment_profiles: Dict[str, Dict[str, Any]],
    eda_insights: Dict[str, Any],
    feature_context: Optional[Dict[str, Any]] = None,
    time_format: str = 'year_week',
) -> Dict[str, HyperparameterProfile]:
    """
    Compute hyperparameter profiles for all segments.

    Parameters
    ----------
    segment_profiles : Dict[str, Dict]
        Segment ID -> segment profile dictionary
    eda_insights : Dict
        EDA insights (seasonality, trend, changepoints)
    feature_context : Dict, optional
        Feature context with n_features, etc.

    Returns
    -------
    Dict[str, HyperparameterProfile]
        Segment ID -> HyperparameterProfile
    """
    n_features = 50  # Default
    if feature_context:
        feat_summary = feature_context.get('feature_summary', {})
        n_features = feat_summary.get('total_features', feat_summary.get('n_features', 50))

    profiles = {}
    for seg_id, seg_profile in segment_profiles.items():
        demand_pattern = seg_profile.get('demand_pattern', seg_profile.get('dominant_pattern', 'smooth'))
        series_length = seg_profile.get('avg_series_length', 100)

        profiles[seg_id] = get_segment_specific_hyperparams(
            segment_profile=seg_profile,
            eda_insights=eda_insights,
            demand_pattern=demand_pattern,
            n_features=n_features,
            series_length=series_length,
            time_format=time_format,
        )

    logger.info(f"Computed hyperparameter profiles for {len(profiles)} segments")
    return profiles


# =============================================================================
# 2. EDA-ENHANCED META-LEARNING
# =============================================================================

def build_enhanced_meta_features(
    y: np.ndarray,
    eda_insights: Dict[str, Any],
    segment_profile: Dict[str, Any],
    feature_context: Optional[Dict[str, Any]] = None,
    time_format: str = 'year_week',
) -> EnhancedMetaFeatures:
    """
    Build enhanced meta-features that include EDA insights for intelligent model selection.

    This is STATE-OF-THE-ART because it combines:
    - Time series statistics (traditional meta-features)
    - EDA-derived insights (seasonality, trend, changepoints)
    - Segment-specific characteristics
    - Feature quality indicators

    Parameters
    ----------
    y : np.ndarray
        Target time series values
    eda_insights : Dict
        EDA insights from EDA crew
    segment_profile : Dict
        Segment profile from segmentation crew
    feature_context : Dict, optional
        Feature context from feature crew

    Returns
    -------
    EnhancedMetaFeatures
        Comprehensive meta-features for model selection
    """
    from scipy import stats

    y = np.asarray(y).flatten()
    n = len(y)

    # =========================================================================
    # BASIC TIME SERIES STATISTICS
    # =========================================================================
    n_nonzero = np.sum(y > 0)
    zero_fraction = 1 - (n_nonzero / n) if n > 0 else 1.0
    mean_val = np.mean(y)
    std_val = np.std(y)
    cv = std_val / mean_val if mean_val > 1e-10 else 0.0

    # Syntetos-Boylan classification
    nonzero_idx = np.where(y > 0)[0]
    if len(nonzero_idx) > 1:
        intervals = np.diff(nonzero_idx)
        adi = np.mean(intervals) if len(intervals) > 0 else n
    else:
        adi = n
    cv2 = cv ** 2

    # Demand pattern classification
    ADI_THRESHOLD = 1.32
    CV2_THRESHOLD = 0.49
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_pattern = 'smooth'
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        demand_pattern = 'erratic'
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_pattern = 'intermittent'
    else:
        demand_pattern = 'lumpy'

    # Higher-order statistics
    try:
        skewness = float(stats.skew(y[y > 0])) if n_nonzero > 2 else 0.0
    except:
        skewness = 0.0

    try:
        kurtosis = float(stats.kurtosis(y[y > 0])) if n_nonzero > 3 else 0.0
    except:
        kurtosis = 0.0

    # Trend strength (simple linear regression slope)
    if n >= 4:
        x = np.arange(n)
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            trend_strength = min(1.0, abs(slope * n) / (mean_val + 1e-10))
            trend_coefficient = float(slope)
        except:
            trend_strength = 0.0
            trend_coefficient = 0.0
    else:
        trend_strength = 0.0
        trend_coefficient = 0.0

    # Autocorrelation
    autocorr_lag1 = 0.0
    autocorr_lag7 = 0.0
    if n > 2:
        try:
            autocorr_lag1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
            if np.isnan(autocorr_lag1):
                autocorr_lag1 = 0.0
        except:
            pass
    if n > 8:
        try:
            autocorr_lag7 = float(np.corrcoef(y[:-7], y[7:])[0, 1])
            if np.isnan(autocorr_lag7):
                autocorr_lag7 = 0.0
        except:
            pass

    # Entropy
    y_nonzero = y[y > 0]
    if len(y_nonzero) > 1:
        try:
            hist, _ = np.histogram(y_nonzero, bins=min(10, len(y_nonzero)))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = float(-np.sum(hist * np.log(hist)))
            entropy = min(entropy / np.log(10), 1.0)
        except:
            entropy = 0.0
    else:
        entropy = 0.0

    # =========================================================================
    # EDA-DERIVED FEATURES
    # =========================================================================
    seasonality_info = eda_insights.get('seasonality', {})
    seasonality_strength = seasonality_info.get('avg_strength',
                            seasonality_info.get('avg_seasonal_strength', 0.0))
    _default_period = 12 if time_format == 'year_month' else 52
    seasonal_period = seasonality_info.get('dominant_period', _default_period)
    has_strong_seasonality = seasonality_strength > 0.3

    changepoint_info = eda_insights.get('changepoints', {})
    changepoint_count = changepoint_info.get('avg_changepoints', 0)
    changepoint_pct = changepoint_info.get('pct_significant',
                        changepoint_info.get('pct_with_changepoints', 0.0))

    # =========================================================================
    # SEGMENT-SPECIFIC FEATURES
    # =========================================================================
    segment_zero_fraction = segment_profile.get('zero_fraction', zero_fraction)
    segment_cv = segment_profile.get('cv', segment_profile.get('coefficient_of_variation', cv))
    segment_skewness = segment_profile.get('skewness', skewness)
    segment_avg_volume = segment_profile.get('avg_volume', segment_profile.get('mean_volume', mean_val))
    segment_size = segment_profile.get('n_keys', segment_profile.get('segment_size', 1))
    expected_difficulty = segment_profile.get('expected_difficulty', 'medium')

    # =========================================================================
    # FEATURE QUALITY INDICATORS
    # =========================================================================
    n_features = 0
    has_external_features = False
    feature_quality_score = 0.5
    n_lag_features = 0
    n_rolling_features = 0
    n_calendar_features = 0
    acf_informed_lags = False

    if feature_context:
        feat_summary = feature_context.get('feature_summary', {})
        n_features = feat_summary.get('total_features', feat_summary.get('n_features', 0))

        categories = feat_summary.get('feature_categories', {})
        n_lag_features = categories.get('lag_features', 0)
        n_rolling_features = categories.get('rolling_features', 0)
        n_calendar_features = categories.get('calendar_features', 0)

        adaptive_config = feat_summary.get('adaptive_config', {})
        acf_informed_lags = adaptive_config.get('acf_informed_lags', False)

        has_external_features = categories.get('external_features', 0) > 0

        # Compute feature quality score
        feature_quality_score = min(1.0, (
            0.3 * min(1.0, n_features / 50) +
            0.2 * min(1.0, n_lag_features / 10) +
            0.2 * min(1.0, n_rolling_features / 5) +
            0.2 * (1.0 if acf_informed_lags else 0.5) +
            0.1 * (1.0 if has_external_features else 0.5)
        ))

    # =========================================================================
    # DERIVED SCORES
    # =========================================================================

    # Forecastability score
    forecastability = (
        0.25 * (1 - zero_fraction) +  # More non-zero is better
        0.20 * min(1.0, max(0, autocorr_lag1)) +  # Higher autocorr is better
        0.15 * (1 - min(1.0, cv / 3)) +  # Lower CV is better
        0.15 * min(1.0, n / 52) +  # More data is better
        0.15 * feature_quality_score +  # Better features is better
        0.10 * (0.5 + 0.5 * seasonality_strength)  # Seasonality helps if present
    )

    # Model complexity recommendation
    if n < 30 or zero_fraction > 0.7:
        model_complexity_recommendation = 'low'
    elif seasonality_strength > 0.3 or cv > 2.0 or n > 200:
        model_complexity_recommendation = 'high'
    else:
        model_complexity_recommendation = 'medium'

    return EnhancedMetaFeatures(
        n_observations=n,
        n_nonzero=n_nonzero,
        zero_fraction=zero_fraction,
        mean=mean_val,
        std=std_val,
        cv=cv,
        demand_pattern=demand_pattern,
        adi=adi,
        cv2=cv2,
        trend_strength=trend_strength,
        autocorr_lag1=autocorr_lag1,
        autocorr_lag7=autocorr_lag7,
        entropy=entropy,
        skewness=skewness,
        kurtosis=kurtosis,
        seasonality_strength=seasonality_strength,
        seasonal_period=seasonal_period,
        has_strong_seasonality=has_strong_seasonality,
        trend_coefficient=trend_coefficient,
        changepoint_count=int(changepoint_count),
        changepoint_pct=changepoint_pct,
        segment_zero_fraction=segment_zero_fraction,
        segment_cv=segment_cv,
        segment_skewness=segment_skewness,
        segment_avg_volume=segment_avg_volume,
        segment_size=segment_size,
        expected_difficulty=expected_difficulty,
        n_features=n_features,
        has_external_features=has_external_features,
        feature_quality_score=feature_quality_score,
        n_lag_features=n_lag_features,
        n_rolling_features=n_rolling_features,
        n_calendar_features=n_calendar_features,
        acf_informed_lags=acf_informed_lags,
        forecastability=forecastability,
        model_complexity_recommendation=model_complexity_recommendation,
    )


def rank_models_with_enhanced_features(
    meta_features: EnhancedMetaFeatures,
    available_models: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Rank models using enhanced meta-features that include EDA insights.

    This is STATE-OF-THE-ART because it considers:
    - Demand pattern fit
    - Seasonality handling capability
    - Zero handling for intermittent data
    - Feature quality utilization
    - Complexity vs data size match

    Parameters
    ----------
    meta_features : EnhancedMetaFeatures
        Enhanced meta-features including EDA insights
    available_models : List[str], optional
        List of available models to rank

    Returns
    -------
    List[Tuple[str, float]]
        List of (model_name, score) sorted by score (descending)
    """
    if available_models is None:
        available_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest',
                           'zero_inflated', 'hurdle_model', 'tweedie']

    # Model characteristics (extended)
    MODEL_CHARS = {
        'lightgbm': {
            'best_for': ['smooth', 'erratic'],
            'handles_zeros': True,
            'handles_seasonality': True,  # Via features
            'training_speed': 1.0,  # Relative (1.0 = fast)
            'complexity': 'medium',
            'min_samples': 30,
        },
        'xgboost': {
            'best_for': ['erratic', 'smooth'],
            'handles_zeros': True,
            'handles_seasonality': True,
            'handles_outliers': True,  # Better with Huber loss
            'training_speed': 0.8,
            'complexity': 'medium',
            'min_samples': 30,
        },
        'catboost': {
            'best_for': ['smooth', 'erratic'],
            'handles_zeros': True,
            'handles_seasonality': True,
            'handles_categoricals': True,
            'training_speed': 0.5,
            'complexity': 'medium',
            'min_samples': 50,
        },
        'random_forest': {
            'best_for': ['smooth'],
            'handles_zeros': True,
            'handles_seasonality': False,  # Less effective
            'training_speed': 1.0,
            'complexity': 'low',
            'min_samples': 20,
        },
        'zero_inflated': {
            'best_for': ['intermittent', 'lumpy'],
            'handles_zeros': True,  # Explicitly designed for zeros
            'handles_seasonality': True,
            'training_speed': 0.7,
            'complexity': 'high',
            'min_samples': 50,
        },
        'hurdle_model': {
            'best_for': ['lumpy', 'intermittent'],
            'handles_zeros': True,
            'handles_seasonality': True,
            'training_speed': 0.7,
            'complexity': 'high',
            'min_samples': 50,
        },
        'tweedie': {
            'best_for': ['lumpy', 'intermittent'],
            'handles_zeros': True,  # Native zero handling
            'handles_seasonality': True,
            'training_speed': 0.9,
            'complexity': 'low',
            'min_samples': 30,
        },
    }

    scores = {}

    for model in available_models:
        if model not in MODEL_CHARS:
            continue

        chars = MODEL_CHARS[model]
        score = 0.0

        # 1. DEMAND PATTERN FIT (weight: 2.5)
        if meta_features.demand_pattern in chars.get('best_for', []):
            score += 2.5
        elif meta_features.demand_pattern in ['intermittent', 'lumpy']:
            # Penalize models not designed for sparse data
            if meta_features.demand_pattern not in chars.get('best_for', []):
                score -= 0.5

        # 2. ZERO HANDLING (weight: 1.5)
        if meta_features.zero_fraction > 0.3:
            if chars.get('handles_zeros', False):
                score += 1.5
            else:
                score -= 1.0

            # Extra boost for models explicitly designed for zeros
            if model in ['zero_inflated', 'hurdle_model', 'tweedie'] and meta_features.zero_fraction > 0.5:
                score += 1.0

        # 3. SEASONALITY HANDLING (weight: 1.0) - NEW: Uses EDA insight
        if meta_features.has_strong_seasonality:
            if chars.get('handles_seasonality', False):
                score += 1.0 * meta_features.seasonality_strength
            else:
                score -= 0.5

        # 4. HIGH VARIANCE / OUTLIER HANDLING (weight: 0.8)
        if meta_features.cv > 1.5:
            if chars.get('handles_outliers', chars.get('handles_zeros', False)):
                score += 0.8
            else:
                score -= 0.3

        # 5. COMPLEXITY vs DATA SIZE MATCH (weight: 1.0)
        complexity = chars.get('complexity', 'medium')
        recommended = meta_features.model_complexity_recommendation
        if complexity == recommended:
            score += 1.0
        elif (complexity == 'high' and recommended == 'low') or \
             (complexity == 'low' and recommended == 'high'):
            score -= 0.5

        # 6. MINIMUM SAMPLES CHECK (weight: -2.0 if violated)
        min_samples = chars.get('min_samples', 20)
        if meta_features.n_observations < min_samples:
            score -= 2.0

        # 7. FEATURE QUALITY UTILIZATION (weight: 0.5)
        if meta_features.feature_quality_score > 0.7:
            # Models that can better utilize features
            if model in ['lightgbm', 'xgboost', 'catboost']:
                score += 0.5 * meta_features.feature_quality_score

        # 8. CHANGEPOINT HANDLING (weight: 0.3) - NEW: Uses EDA insight
        if meta_features.changepoint_pct > 0.3:
            # Tree-based models handle regime changes well
            if model in ['lightgbm', 'xgboost', 'catboost']:
                score += 0.3

        # 9. SEGMENT SIZE CONSIDERATION (weight: 0.3)
        if meta_features.segment_size > 50:
            # Larger segments benefit from more complex models
            if model in ['catboost', 'zero_inflated', 'hurdle_model']:
                score += 0.3

        scores[model] = score

    # Sort by score (descending)
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    return ranked


# =============================================================================
# 3. PATTERN-AWARE ENSEMBLE WEIGHTING
# =============================================================================

def compute_pattern_aware_ensemble_weights(
    candidate_results: Dict[str, Any],
    demand_pattern: str,
    forecast_horizon: int = 8,
    meta_features: Optional[EnhancedMetaFeatures] = None,
) -> Dict[str, float]:
    """
    Compute ensemble weights that consider demand pattern and forecast horizon.

    This is STATE-OF-THE-ART because:
    - Different models excel at different patterns
    - Model performance degrades differently with horizon
    - Zero-handling models should have higher weight for intermittent data

    Parameters
    ----------
    candidate_results : Dict[str, Any]
        Model name -> TrainingResult with val_wape
    demand_pattern : str
        'smooth', 'erratic', 'intermittent', or 'lumpy'
    forecast_horizon : int
        Number of steps ahead
    meta_features : EnhancedMetaFeatures, optional
        Enhanced meta-features for additional context

    Returns
    -------
    Dict[str, float]
        Model name -> normalized weight (sums to 1.0)
    """
    # Pattern-specific model boosts
    PATTERN_BOOSTS = {
        'smooth': {
            'lightgbm': 1.1,
            'xgboost': 1.1,
            'catboost': 1.0,
            'random_forest': 0.9,
            'tweedie': 0.8,
        },
        'erratic': {
            'xgboost': 1.2,  # Better with outliers
            'lightgbm': 1.1,
            'catboost': 1.1,
            'random_forest': 0.8,
            'tweedie': 0.9,
        },
        'intermittent': {
            'zero_inflated': 1.4,  # Explicitly designed for zeros
            'hurdle_model': 1.3,
            'lightgbm': 1.0,
            'xgboost': 1.0,
            'tweedie': 1.2,
        },
        'lumpy': {
            'hurdle_model': 1.4,  # Better for lumpy (structural zeros)
            'zero_inflated': 1.3,
            'tweedie': 1.2,
            'xgboost': 1.0,
            'lightgbm': 1.0,
        },
    }

    # Horizon decay factors (some models degrade faster)
    HORIZON_DECAY = {
        'lightgbm': 0.98,  # Decays slowly
        'xgboost': 0.98,
        'catboost': 0.97,
        'random_forest': 0.95,  # Decays faster
        'zero_inflated': 0.96,
        'hurdle_model': 0.96,
        'tweedie': 0.97,
    }

    # Get pattern boosts
    boosts = PATTERN_BOOSTS.get(demand_pattern, {})

    # Compute base weights (inverse WAPE)
    inv_wapes = {}
    for model, result in candidate_results.items():
        if result is None:
            continue

        # Get validation WAPE
        if hasattr(result, 'val_wape'):
            wape = result.val_wape
        elif isinstance(result, dict):
            wape = result.get('val_wape', result.get('wape', 1.0))
        else:
            wape = 1.0

        # Avoid division by zero
        inv_wapes[model] = 1.0 / max(wape, 0.01)

    if not inv_wapes:
        return {}

    # Apply pattern boosts
    for model in inv_wapes:
        boost = boosts.get(model, 1.0)
        inv_wapes[model] *= boost

    # Apply horizon decay
    if forecast_horizon > 1:
        for model in inv_wapes:
            decay = HORIZON_DECAY.get(model, 0.97)
            # Decay factor increases with horizon
            inv_wapes[model] *= (decay ** (forecast_horizon - 1))

    # Additional adjustments based on meta-features
    if meta_features is not None:
        # Boost zero-handling models for high zero fraction
        if meta_features.zero_fraction > 0.5:
            for model in ['zero_inflated', 'hurdle_model', 'tweedie']:
                if model in inv_wapes:
                    inv_wapes[model] *= (1.0 + meta_features.zero_fraction * 0.3)

        # Boost models for high seasonality
        if meta_features.has_strong_seasonality:
            for model in ['lightgbm', 'xgboost', 'catboost']:
                if model in inv_wapes:
                    inv_wapes[model] *= (1.0 + meta_features.seasonality_strength * 0.2)

    # Normalize to sum to 1.0
    total = sum(inv_wapes.values())
    weights = {m: w / total for m, w in inv_wapes.items()}

    return weights


def create_pattern_aware_ensemble(
    candidate_results: Dict[str, Any],
    demand_pattern: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    top_k: int = 3,
    meta_features: Optional[EnhancedMetaFeatures] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a pattern-aware ensemble from candidate models.

    Parameters
    ----------
    candidate_results : Dict[str, TrainingResult]
        Model name -> training result with model object
    demand_pattern : str
        Demand pattern for this segment
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
    top_k : int
        Number of top models to include in ensemble
    meta_features : EnhancedMetaFeatures, optional
        Enhanced meta-features

    Returns
    -------
    Tuple[EnsembleModel, Dict]
        (ensemble model, ensemble info dict)
    """
    from utils.model_training import compute_wape

    # Get pattern-aware weights
    weights = compute_pattern_aware_ensemble_weights(
        candidate_results=candidate_results,
        demand_pattern=demand_pattern,
        meta_features=meta_features,
    )

    # Select top-k models by weight
    sorted_models = sorted(weights.items(), key=lambda x: -x[1])
    selected_models = [m for m, _ in sorted_models[:top_k] if m in candidate_results]

    if len(selected_models) < 2:
        # Not enough models for ensemble
        if selected_models:
            best_model = selected_models[0]
            return candidate_results[best_model].model, {
                'is_ensemble': False,
                'selected_model': best_model,
                'reason': 'insufficient_candidates',
            }
        return None, {'error': 'no_valid_candidates'}

    # Renormalize weights for selected models
    selected_weights = {m: weights[m] for m in selected_models}
    total = sum(selected_weights.values())
    selected_weights = {m: w / total for m, w in selected_weights.items()}

    # Build ensemble predictions on validation set
    ensemble_preds = np.zeros(len(y_val))
    models_in_ensemble = []

    for model_name in selected_models:
        result = candidate_results[model_name]
        model = result.model if hasattr(result, 'model') else result

        try:
            preds = _predict_model(model, X_val)
            preds = np.clip(preds, 0, None)
            ensemble_preds += selected_weights[model_name] * preds
            models_in_ensemble.append(model_name)
        except Exception as e:
            logger.warning(f"Ensemble prediction failed for {model_name}: {e}")

    if not models_in_ensemble:
        return None, {'error': 'all_predictions_failed'}

    # Compute ensemble WAPE
    ensemble_wape = compute_wape(y_val, ensemble_preds)

    # Create ensemble model wrapper using module-level PatternAwareEnsemble class
    # (Module-level class is required for pickle serialization by joblib.dump())
    ensemble_models = {
        m: candidate_results[m].model if hasattr(candidate_results[m], 'model') else candidate_results[m]
        for m in models_in_ensemble
    }

    ensemble = PatternAwareEnsemble(ensemble_models, selected_weights)

    info = {
        'is_ensemble': True,
        'model_types': models_in_ensemble,
        'weights': selected_weights,
        'ensemble_wape': ensemble_wape,
        'demand_pattern': demand_pattern,
        'method': 'pattern_aware_weighting',
    }

    return ensemble, info


# =============================================================================
# 4. ADAPTIVE ZERO-INFLATED THRESHOLD CALIBRATION
# =============================================================================

def calibrate_zero_inflated_threshold(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: Optional[List[float]] = None,
    metric: str = 'wape',
) -> Tuple[float, Dict[str, Any]]:
    """
    Calibrate the optimal probability threshold for zero-inflated models.

    Instead of using fixed rules, this learns the optimal threshold from
    validation data that minimizes the chosen metric (WAPE by default).

    Parameters
    ----------
    model : ZeroInflatedModel or HurdleModel
        Trained two-stage model with predict method
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
    thresholds : List[float], optional
        Thresholds to try. Default: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    metric : str
        Metric to optimize: 'wape', 'mae', or 'zero_accuracy'

    Returns
    -------
    Tuple[float, Dict]
        (optimal_threshold, calibration_info)
    """
    from utils.model_training import compute_wape, compute_mae

    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    y_val = np.asarray(y_val).flatten()

    # Check if model supports threshold-based prediction
    if not hasattr(model, 'predict'):
        logger.warning("Model doesn't have predict method, returning default threshold")
        return 0.5, {'error': 'no_predict_method'}

    results = []

    for thresh in thresholds:
        try:
            # Try to pass threshold to predict
            if hasattr(model, 'zero_threshold'):
                # Temporarily set threshold
                original_threshold = model.zero_threshold
                model.zero_threshold = thresh
                preds = model.predict(X_val)
                model.zero_threshold = original_threshold
            else:
                # Try passing as argument
                try:
                    preds = model.predict(X_val, zero_threshold=thresh)
                except TypeError:
                    preds = model.predict(X_val)
                    # Can't vary threshold, break early
                    break

            preds = np.clip(preds, 0, None)

            # Compute metrics
            wape = compute_wape(y_val, preds)
            mae = compute_mae(y_val, preds)

            # Compute zero accuracy
            actual_zeros = y_val == 0
            pred_zeros = preds < 0.5  # Consider < 0.5 as predicting zero
            zero_accuracy = np.mean(actual_zeros == pred_zeros)

            results.append({
                'threshold': thresh,
                'wape': wape,
                'mae': mae,
                'zero_accuracy': zero_accuracy,
            })

        except Exception as e:
            logger.warning(f"Threshold calibration failed for {thresh}: {e}")

    if not results:
        return 0.5, {'error': 'all_thresholds_failed'}

    # Select optimal threshold based on metric
    if metric == 'wape':
        best = min(results, key=lambda x: x['wape'])
    elif metric == 'mae':
        best = min(results, key=lambda x: x['mae'])
    elif metric == 'zero_accuracy':
        best = max(results, key=lambda x: x['zero_accuracy'])
    else:
        best = min(results, key=lambda x: x['wape'])

    optimal_threshold = best['threshold']

    # Compare with default (0.5) performance
    default_result = next((r for r in results if r['threshold'] == 0.5), results[0])
    improvement = (default_result['wape'] - best['wape']) / max(default_result['wape'], 0.01)

    info = {
        'optimal_threshold': optimal_threshold,
        'optimal_wape': best['wape'],
        'optimal_mae': best['mae'],
        'optimal_zero_accuracy': best['zero_accuracy'],
        'default_wape': default_result['wape'],
        'improvement_pct': improvement * 100,
        'all_results': results,
        'metric_optimized': metric,
    }

    logger.info(f"Calibrated threshold: {optimal_threshold:.2f} (WAPE improvement: {improvement*100:.1f}%)")

    return optimal_threshold, info


def optimize_hurdle_threshold(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    zero_fraction: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """
    Optimize threshold for hurdle models, considering the zero fraction.

    Hurdle models need different thresholds than zero-inflated models
    because they model "crossing the hurdle" (any demand vs no demand).

    Parameters
    ----------
    model : HurdleModel
        Trained hurdle model
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
    zero_fraction : float
        Fraction of zeros in the training data

    Returns
    -------
    Tuple[float, Dict]
        (optimal_threshold, info)
    """
    # Generate threshold candidates around the zero_fraction
    # Logic: If 70% of data is zeros, we should be more conservative (higher threshold)
    base_threshold = 0.3 + 0.4 * zero_fraction  # Range: 0.3 to 0.7

    # Create candidates around the base
    thresholds = [
        max(0.2, base_threshold - 0.2),
        max(0.2, base_threshold - 0.1),
        base_threshold,
        min(0.8, base_threshold + 0.1),
        min(0.8, base_threshold + 0.2),
    ]
    thresholds = sorted(set(thresholds))

    optimal, info = calibrate_zero_inflated_threshold(
        model=model,
        X_val=X_val,
        y_val=y_val,
        thresholds=thresholds,
        metric='wape',
    )

    info['zero_fraction'] = zero_fraction
    info['base_threshold'] = base_threshold

    return optimal, info


# =============================================================================
# 5. SEGMENT-AWARE BIAS CALIBRATION
# =============================================================================

def learn_segment_bias_calibration(
    segment_id: str,
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    key_zero_fractions: Optional[np.ndarray] = None,
    demand_pattern: str = 'smooth',
    n_buckets: int = 5,
    min_samples_per_bucket: int = 10,
) -> SegmentBiasCalibration:
    """
    Learn bias calibration factors for a specific segment.

    This is STATE-OF-THE-ART because:
    - Calibration is per-segment (not global)
    - Factors vary by zero_fraction bucket (different behavior for sparse vs dense keys)
    - Bounded to prevent extreme corrections

    Parameters
    ----------
    segment_id : str
        Segment identifier
    y_actual : np.ndarray
        Actual values from validation set
    y_predicted : np.ndarray
        Model predictions on validation set
    key_zero_fractions : np.ndarray, optional
        Zero fraction for each observation's key
    demand_pattern : str
        Segment's demand pattern
    n_buckets : int
        Number of zero_fraction buckets
    min_samples_per_bucket : int
        Minimum samples required per bucket

    Returns
    -------
    SegmentBiasCalibration
        Calibration factors for the segment
    """
    y_actual = np.asarray(y_actual).flatten()
    y_predicted = np.asarray(y_predicted).flatten()

    # Compute overall bias
    total_actual = np.sum(y_actual)
    total_pred = np.sum(y_predicted)

    if total_actual < 1e-10:
        # No actual demand, can't calibrate
        return SegmentBiasCalibration(
            segment_id=segment_id,
            demand_pattern=demand_pattern,
            calibration_factors={},
            global_factor=1.0,
            n_keys_calibrated=0,
        )

    global_bias = (total_pred - total_actual) / total_actual
    global_factor = total_actual / total_pred if total_pred > 1e-10 else 1.0

    # Bound global factor
    global_factor = max(0.2, min(2.0, global_factor))

    calibration_factors = {}
    bucket_edges = np.linspace(0, 1, n_buckets + 1)

    if key_zero_fractions is not None:
        key_zero_fractions = np.asarray(key_zero_fractions).flatten()

        # Compute calibration factor per bucket
        for i in range(n_buckets):
            low, high = bucket_edges[i], bucket_edges[i + 1]
            bucket_name = f"zf_{low:.1f}_{high:.1f}"

            mask = (key_zero_fractions >= low) & (key_zero_fractions < high)
            if i == n_buckets - 1:  # Include upper bound in last bucket
                mask = (key_zero_fractions >= low) & (key_zero_fractions <= high)

            if np.sum(mask) >= min_samples_per_bucket:
                bucket_actual = np.sum(y_actual[mask])
                bucket_pred = np.sum(y_predicted[mask])

                if bucket_actual > 1e-10 and bucket_pred > 1e-10:
                    factor = bucket_actual / bucket_pred
                    factor = max(0.2, min(2.0, factor))  # Bound
                    calibration_factors[bucket_name] = factor
                else:
                    calibration_factors[bucket_name] = global_factor
            else:
                # Not enough samples, use global factor
                calibration_factors[bucket_name] = global_factor

    return SegmentBiasCalibration(
        segment_id=segment_id,
        demand_pattern=demand_pattern,
        calibration_factors=calibration_factors,
        global_factor=global_factor,
        n_keys_calibrated=len(y_actual),
        avg_bias_before=global_bias,
        avg_bias_after=0.0,  # Will be computed after application
        bucket_edges=list(bucket_edges),
    )


def apply_segment_bias_calibration(
    predictions: np.ndarray,
    calibration: SegmentBiasCalibration,
    key_zero_fractions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply segment-specific bias calibration to predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions to calibrate
    calibration : SegmentBiasCalibration
        Learned calibration factors
    key_zero_fractions : np.ndarray, optional
        Zero fraction for each prediction's key

    Returns
    -------
    np.ndarray
        Calibrated predictions
    """
    predictions = np.asarray(predictions).flatten().copy()

    if key_zero_fractions is not None and calibration.calibration_factors:
        key_zero_fractions = np.asarray(key_zero_fractions).flatten()
        bucket_edges = calibration.bucket_edges
        n_buckets = len(bucket_edges) - 1

        for i in range(n_buckets):
            low, high = bucket_edges[i], bucket_edges[i + 1]
            bucket_name = f"zf_{low:.1f}_{high:.1f}"

            mask = (key_zero_fractions >= low) & (key_zero_fractions < high)
            if i == n_buckets - 1:
                mask = (key_zero_fractions >= low) & (key_zero_fractions <= high)

            factor = calibration.calibration_factors.get(bucket_name, calibration.global_factor)
            predictions[mask] *= factor
    else:
        # Apply global factor
        predictions *= calibration.global_factor

    # Ensure non-negative
    predictions = np.clip(predictions, 0, None)

    return predictions


def compute_all_segment_calibrations(
    segment_predictions: Dict[str, Dict[str, np.ndarray]],
    segment_profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, SegmentBiasCalibration]:
    """
    Compute bias calibration for all segments.

    Parameters
    ----------
    segment_predictions : Dict[str, Dict]
        segment_id -> {'y_actual': array, 'y_predicted': array, 'zero_fractions': array}
    segment_profiles : Dict[str, Dict]
        segment_id -> segment profile with demand_pattern

    Returns
    -------
    Dict[str, SegmentBiasCalibration]
        segment_id -> calibration
    """
    calibrations = {}

    for seg_id, pred_data in segment_predictions.items():
        y_actual = pred_data.get('y_actual', np.array([]))
        y_predicted = pred_data.get('y_predicted', np.array([]))
        zero_fractions = pred_data.get('zero_fractions')

        profile = segment_profiles.get(seg_id, {})
        demand_pattern = profile.get('demand_pattern', 'smooth')

        if len(y_actual) > 0:
            calibrations[seg_id] = learn_segment_bias_calibration(
                segment_id=seg_id,
                y_actual=y_actual,
                y_predicted=y_predicted,
                key_zero_fractions=zero_fractions,
                demand_pattern=demand_pattern,
            )

    logger.info(f"Computed bias calibration for {len(calibrations)} segments")
    return calibrations


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'HyperparameterProfile',
    'EnhancedMetaFeatures',
    'SegmentBiasCalibration',

    # 1. Segment-specific hyperparameters
    'get_segment_specific_hyperparams',
    'compute_all_segment_hyperparams',

    # 2. EDA-enhanced meta-learning
    'build_enhanced_meta_features',
    'rank_models_with_enhanced_features',

    # 3. Pattern-aware ensemble
    'compute_pattern_aware_ensemble_weights',
    'create_pattern_aware_ensemble',

    # 4. Adaptive threshold calibration
    'calibrate_zero_inflated_threshold',
    'optimize_hurdle_threshold',

    # 5. Segment-aware bias calibration
    'learn_segment_bias_calibration',
    'apply_segment_bias_calibration',
    'compute_all_segment_calibrations',
]
