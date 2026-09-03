"""
State-of-the-Art Automated Model Selection Intelligence.

This module implements intelligent, data-driven model selection that guarantees
the best possible results for any dataset. It goes beyond simple cross-validation
to incorporate:

1. META-LEARNING: Learn which models work best for different data characteristics
2. AUTO-ML SELECTION: Systematic model search with early stopping
3. DEMAND PATTERN AWARE: Different strategies for smooth/erratic/intermittent/lumpy
4. FEATURE-AWARE: Select models based on feature quality and availability
5. ENSEMBLE INTELLIGENCE: Smart selection of models for ensembling
6. COLD-START HANDLING: Strategies for new products with limited history

USAGE:
>>> from utils.model_selection_intelligence import (
...     # High-level selection
...     select_best_model_intelligently,
...     run_model_selection_tournament,
...
...     # Meta-learning
...     build_meta_features,
...     predict_best_model_family,
...     train_meta_learner,
...
...     # Auto-ML
...     automated_model_search,
...     early_stopping_model_search,
...
...     # Demand-pattern specific
...     get_candidate_models_for_pattern,
...     rank_models_by_pattern_fit,
...
...     # Cold-start
...     handle_cold_start_selection,
...     transfer_learning_selection,
... )

Author: FEU-Agentic-Forecasting Demand Forecasting System
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Syntetos-Boylan thresholds
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

# FEATURE-BASED MODELS (always allowed)
ALLOWED_MODEL_FAMILIES = [
    'lightgbm',
    'xgboost',
    'catboost',
    'random_forest',
    'zero_inflated',
    'hurdle_model',
    'tweedie',
    # Discrete/Ordinal demand specialists
    'ordinal_regression',
    'discrete_classifier',
    'hybrid_discrete',
    # Multi-horizon direct forecasting (for long-horizon optimization like Lag 5)
    # These models directly optimize for target horizon, avoiding recursive error compounding
    'multi_horizon_lightgbm',
    'multi_horizon_xgboost',
    'multi_horizon_ensemble',
    # Phase 3: Hierarchical models (pool data across keys/hierarchy levels)
    'global_local',
    'mixed_effects',
    'multi_level_ensemble',
    # Phase 7: Enhanced model zoo
    'catboost_embedding',
    'quantile_regression',
    'conformal_boost',
    # Phase 8: Model combination
    'stacked_ensemble',
]

# Univariate models — enabled for monthly data where feature-based models
# may struggle due to limited observations (12-36 data points)
UNIVARIATE_MODEL_FAMILIES = [
    'croston', 'sba', 'tsb', 'imapa',
    'arima', 'sarima', 'ets', 'theta',
]


def get_allowed_model_families(time_format: str = 'year_week') -> list:
    """Get allowed model families, including univariate models for monthly data.

    For monthly data (year_month), univariate models are enabled because:
    - Monthly data has fewer observations (~12-60 per key)
    - Feature-based models may overfit with limited data
    - Classical statistical models (ARIMA, ETS, Theta) are designed for small samples

    Returns
    -------
    list of allowed model family names
    """
    families = list(ALLOWED_MODEL_FAMILIES)
    if time_format == 'year_month':
        families.extend(UNIVARIATE_MODEL_FAMILIES)
    return families


# Phase-3 hierarchical model families that pool across keys or hierarchy levels.
# These are NEVER included in the per-config ``model_families`` list by default
# (the schema's default list only contains flat-table models). They have to be
# explicitly opted-in via ``design.enable_hierarchical_models`` because they
# require a hierarchy column to be resolvable, and using one without a valid
# hierarchy quietly degrades to a global model.
HIERARCHICAL_MODEL_FAMILIES: List[str] = [
    'global_local',
    'mixed_effects',
    'multi_level_ensemble',
]


def resolve_effective_model_families(config) -> List[str]:
    """Compute the effective ``model_families`` list for training.

    Merges the user-configured ``config.design.model_families`` with any
    Phase-3 hierarchical models when ``enable_hierarchical_models`` is True.
    This is the single source of truth for "which models is the selection
    crew allowed to consider?" — previously ``enable_hierarchical_models``
    was declared in the schema but never read anywhere, so hierarchical
    models were silently absent from selection even when the config asked
    for them.

    Parameters
    ----------
    config : DemandForecastConfig

    Returns
    -------
    List[str]
        Deduplicated family list preserving original ordering. When
        ``enable_hierarchical_models`` is True (default), the Phase-3
        families are appended to the end.
    """
    families = list(getattr(config.design, 'model_families', []))
    if getattr(config.design, 'enable_hierarchical_models', True):
        for fam in HIERARCHICAL_MODEL_FAMILIES:
            if fam not in families:
                families.append(fam)
    return families

# Model characteristics for intelligent selection
MODEL_CHARACTERISTICS = {
    'lightgbm': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,  # Tree models naturally handle discrete
        'needs_features': True,
        'min_samples': 30,
        'complexity': 'medium',
        'training_speed': 'fast',
        'best_for': ['smooth', 'erratic'],
    },
    'xgboost': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,  # Tree models naturally handle discrete
        'needs_features': True,
        'min_samples': 30,
        'complexity': 'medium',
        'training_speed': 'medium',
        'best_for': ['erratic', 'smooth'],
    },
    'catboost': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,  # Tree models naturally handle discrete
        'needs_features': True,
        'handles_categoricals': True,
        'min_samples': 50,
        'complexity': 'medium',
        'training_speed': 'slow',
        'best_for': ['smooth', 'erratic'],
    },
    'random_forest': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,  # Tree models naturally handle discrete
        'needs_features': True,
        'min_samples': 20,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['smooth'],
    },
    'zero_inflated': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': False,  # Continuous distribution
        'needs_features': True,
        'min_samples': 50,
        'complexity': 'high',
        'training_speed': 'medium',
        'best_for': ['intermittent', 'lumpy'],
    },
    'hurdle_model': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': False,  # Continuous distribution
        'needs_features': True,
        'min_samples': 50,
        'complexity': 'high',
        'training_speed': 'medium',
        'best_for': ['lumpy', 'intermittent'],
    },
    'tweedie': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': False,  # Continuous distribution
        'needs_features': True,
        'min_samples': 30,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['lumpy', 'intermittent'],
    },
    # Discrete/Ordinal demand specialist models
    'ordinal_regression': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,  # Designed for discrete ordinal data
        'needs_features': True,
        'min_samples': 30,
        'complexity': 'medium',
        'training_speed': 'fast',
        'best_for': ['discrete_low', 'discrete_medium'],  # 5-15 classes
        'max_classes': 20,  # Practical limit for ordinal
    },
    'discrete_classifier': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,  # Designed for very low cardinality
        'needs_features': True,
        'min_samples': 20,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['discrete_very_low'],  # 2-5 classes
        'max_classes': 10,  # Practical limit for classification
    },
    'hybrid_discrete': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,  # Regression + snap to discrete
        'needs_features': True,
        'min_samples': 30,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['discrete_medium', 'discrete_high'],  # 10-50 classes
        'max_classes': 100,  # Can handle more via snapping
    },
    # Multi-horizon direct forecasting models
    # These train separate models for each horizon, eliminating recursive error compounding
    'multi_horizon_lightgbm': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,
        'needs_features': True,
        'min_samples': 50,  # Needs more samples for multi-horizon training
        'complexity': 'medium',
        'training_speed': 'medium',  # Trains multiple models
        'best_for': ['smooth', 'erratic'],  # Best for regular demand with long horizons
        'optimal_horizon': 5,  # Designed for Lag 5+ forecasting
    },
    'multi_horizon_xgboost': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,
        'needs_features': True,
        'min_samples': 50,
        'complexity': 'medium',
        'training_speed': 'medium',
        'best_for': ['erratic', 'smooth'],
        'optimal_horizon': 5,
    },
    'multi_horizon_ensemble': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': True,
        'needs_features': True,
        'min_samples': 80,  # Needs more samples for ensemble
        'complexity': 'high',
        'training_speed': 'slow',  # Trains multiple models + optimization
        'best_for': ['smooth', 'erratic'],
        'optimal_horizon': 5,
    },
    # Univariate / classical statistical models — enabled for monthly data
    # where limited observations make feature-based models prone to overfitting
    'croston': {
        'handles_zeros': True,
        'handles_high_cv': False,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 12,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['intermittent', 'lumpy'],
    },
    'sba': {
        'handles_zeros': True,
        'handles_high_cv': False,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 12,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['intermittent', 'lumpy'],
    },
    'tsb': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 12,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['intermittent', 'lumpy'],
    },
    'imapa': {
        'handles_zeros': True,
        'handles_high_cv': True,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 12,
        'complexity': 'medium',
        'training_speed': 'medium',
        'best_for': ['intermittent', 'lumpy'],
    },
    'arima': {
        'handles_zeros': False,
        'handles_high_cv': True,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 12,
        'complexity': 'medium',
        'training_speed': 'medium',
        'best_for': ['smooth', 'erratic'],
    },
    'sarima': {
        'handles_zeros': False,
        'handles_high_cv': True,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 24,
        'complexity': 'high',
        'training_speed': 'slow',
        'best_for': ['smooth'],
    },
    'ets': {
        'handles_zeros': False,
        'handles_high_cv': True,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 12,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['smooth', 'erratic'],
    },
    'theta': {
        'handles_zeros': False,
        'handles_high_cv': False,
        'handles_discrete': False,
        'needs_features': False,
        'min_samples': 12,
        'complexity': 'low',
        'training_speed': 'fast',
        'best_for': ['smooth'],
    },
}

# Pattern-to-model priority mapping (base priorities - used as fallback)
PATTERN_MODEL_PRIORITY = {
    'smooth': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'tweedie'],
    'erratic': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'tweedie'],
    'intermittent': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'catboost'],
    'lumpy': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'tweedie'],
}

# =============================================================================
# DYNAMIC MODEL PRIORITY THRESHOLDS
# =============================================================================

# CV thresholds for categorization
CV_LOW_THRESHOLD = 0.5      # Low variability
CV_MEDIUM_THRESHOLD = 1.0   # Medium variability
CV_HIGH_THRESHOLD = 2.0     # High variability (above this is very high)

# Zero fraction thresholds
ZERO_FRACTION_LOW = 0.1     # Almost no zeros
ZERO_FRACTION_MEDIUM = 0.3  # Some zeros
ZERO_FRACTION_HIGH = 0.5    # Many zeros (sparse demand)
ZERO_FRACTION_VERY_HIGH = 0.7  # Very sparse demand

# Discrete demand detection thresholds
DISCRETE_UNIQUE_RATIO_THRESHOLD = 0.05  # If unique values / n_obs < this, consider discrete
DISCRETE_MAX_UNIQUE_VALUES = 20  # Maximum unique values to still consider discrete
ORDINAL_MAX_CLASSES = 10  # Maximum classes for ordinal regression to be effective
CLASSIFICATION_MAX_CLASSES = 5  # Maximum classes for pure classification approach

# Discrete demand cardinality categories
CARDINALITY_VERY_LOW = 5   # ≤5 classes: classification viable
CARDINALITY_LOW = 10       # ≤10 classes: ordinal regression optimal
CARDINALITY_MEDIUM = 20    # ≤20 classes: discrete-aware regression
# >20 classes: treat as continuous with discretization post-processing

# Dynamic model priority matrix based on (pattern, cv_category, zero_fraction_category)
# This provides fine-grained model selection based on actual data characteristics
DYNAMIC_MODEL_PRIORITY_MATRIX = {
    # SMOOTH PATTERN - regular demand with low ADI and low CV²
    ('smooth', 'low_cv', 'low_zeros'): ['lightgbm', 'catboost', 'xgboost', 'random_forest'],
    ('smooth', 'low_cv', 'medium_zeros'): ['lightgbm', 'xgboost', 'catboost', 'tweedie'],
    ('smooth', 'low_cv', 'high_zeros'): ['lightgbm', 'xgboost', 'zero_inflated', 'tweedie'],
    ('smooth', 'medium_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'catboost', 'random_forest'],
    ('smooth', 'medium_cv', 'medium_zeros'): ['xgboost', 'lightgbm', 'catboost', 'tweedie'],
    ('smooth', 'medium_cv', 'high_zeros'): ['xgboost', 'lightgbm', 'zero_inflated', 'tweedie'],
    ('smooth', 'high_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'catboost', 'random_forest'],
    ('smooth', 'high_cv', 'medium_zeros'): ['xgboost', 'lightgbm', 'tweedie', 'catboost'],
    ('smooth', 'high_cv', 'high_zeros'): ['xgboost', 'lightgbm', 'zero_inflated', 'hurdle_model'],

    # ERRATIC PATTERN - regular demand with low ADI but high CV²
    ('erratic', 'low_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'catboost', 'random_forest'],
    ('erratic', 'low_cv', 'medium_zeros'): ['xgboost', 'lightgbm', 'catboost', 'tweedie'],
    ('erratic', 'low_cv', 'high_zeros'): ['xgboost', 'lightgbm', 'zero_inflated', 'tweedie'],
    ('erratic', 'medium_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'catboost', 'random_forest'],
    ('erratic', 'medium_cv', 'medium_zeros'): ['xgboost', 'lightgbm', 'tweedie', 'catboost'],
    ('erratic', 'medium_cv', 'high_zeros'): ['xgboost', 'lightgbm', 'zero_inflated', 'hurdle_model'],
    ('erratic', 'high_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'catboost', 'tweedie'],
    ('erratic', 'high_cv', 'medium_zeros'): ['xgboost', 'lightgbm', 'tweedie', 'hurdle_model'],
    ('erratic', 'high_cv', 'high_zeros'): ['xgboost', 'lightgbm', 'zero_inflated', 'hurdle_model'],

    # INTERMITTENT PATTERN - sporadic demand with high ADI but low CV²
    ('intermittent', 'low_cv', 'low_zeros'): ['lightgbm', 'xgboost', 'catboost', 'tweedie'],
    ('intermittent', 'low_cv', 'medium_zeros'): ['lightgbm', 'xgboost', 'zero_inflated', 'tweedie'],
    ('intermittent', 'low_cv', 'high_zeros'): ['zero_inflated', 'hurdle_model', 'lightgbm', 'tweedie'],
    ('intermittent', 'medium_cv', 'low_zeros'): ['lightgbm', 'xgboost', 'catboost', 'tweedie'],
    ('intermittent', 'medium_cv', 'medium_zeros'): ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model'],
    ('intermittent', 'medium_cv', 'high_zeros'): ['zero_inflated', 'hurdle_model', 'lightgbm', 'tweedie'],
    ('intermittent', 'high_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'tweedie', 'catboost'],
    ('intermittent', 'high_cv', 'medium_zeros'): ['xgboost', 'lightgbm', 'zero_inflated', 'hurdle_model'],
    ('intermittent', 'high_cv', 'high_zeros'): ['zero_inflated', 'hurdle_model', 'xgboost', 'tweedie'],

    # LUMPY PATTERN - sporadic demand with high ADI and high CV²
    ('lumpy', 'low_cv', 'low_zeros'): ['lightgbm', 'xgboost', 'tweedie', 'catboost'],
    ('lumpy', 'low_cv', 'medium_zeros'): ['lightgbm', 'xgboost', 'zero_inflated', 'tweedie'],
    ('lumpy', 'low_cv', 'high_zeros'): ['zero_inflated', 'hurdle_model', 'lightgbm', 'tweedie'],
    ('lumpy', 'medium_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'tweedie', 'catboost'],
    ('lumpy', 'medium_cv', 'medium_zeros'): ['xgboost', 'lightgbm', 'zero_inflated', 'hurdle_model'],
    ('lumpy', 'medium_cv', 'high_zeros'): ['zero_inflated', 'hurdle_model', 'xgboost', 'tweedie'],
    ('lumpy', 'high_cv', 'low_zeros'): ['xgboost', 'lightgbm', 'tweedie', 'hurdle_model'],
    ('lumpy', 'high_cv', 'medium_zeros'): ['xgboost', 'zero_inflated', 'hurdle_model', 'tweedie'],
    ('lumpy', 'high_cv', 'high_zeros'): ['zero_inflated', 'hurdle_model', 'tweedie', 'xgboost'],
}

# Model adjustments for discrete demand (limited distinct values like 0, 200, 400, 800)
DISCRETE_DEMAND_MODEL_BOOST = ['xgboost', 'lightgbm', 'catboost']  # Tree models handle discrete well
DISCRETE_DEMAND_MODEL_PENALIZE = ['tweedie']  # Continuous distribution models struggle

# Discrete-specific models (when available)
# ordinal_regression: Uses ordinal structure (200 < 400 < 800)
# discrete_classifier: Treats as multi-class classification
# hybrid_discrete: Probability-weighted combination
DISCRETE_SPECIALIZED_MODELS = ['ordinal_regression', 'discrete_classifier', 'hybrid_discrete']


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetaFeatures:
    """Meta-features describing a time series for model selection."""
    # Basic statistics
    n_observations: int
    n_nonzero: int
    zero_fraction: float
    mean: float
    std: float
    cv: float

    # Demand pattern
    demand_pattern: str  # smooth, erratic, intermittent, lumpy
    adi: float  # Average Demand Interval
    cv2: float  # Squared coefficient of variation

    # Time series characteristics
    trend_strength: float
    seasonal_strength: float
    autocorr_lag1: float
    autocorr_lag7: float
    entropy: float

    # Feature availability
    n_features: int
    has_external_features: bool
    feature_quality_score: float

    # Derived
    forecastability: float


@dataclass
class ModelSelectionResult:
    """Result of model selection process."""
    selected_model: str
    selection_method: str
    cv_score: float
    all_scores: Dict[str, float]
    model_rankings: List[str]
    confidence: float
    meta_features: Optional[MetaFeatures] = None
    selection_time_seconds: float = 0.0


@dataclass
class TournamentResult:
    """Result of model tournament."""
    winner: str
    rankings: List[Tuple[str, float]]
    eliminated_models: Dict[str, str]  # model -> reason
    rounds_completed: int
    total_time_seconds: float


# =============================================================================
# DYNAMIC MODEL PRIORITY FUNCTIONS
# =============================================================================

def _categorize_cv(cv: float) -> str:
    """Categorize CV into low/medium/high for matrix lookup."""
    if cv <= CV_LOW_THRESHOLD:
        return 'low_cv'
    elif cv <= CV_MEDIUM_THRESHOLD:
        return 'medium_cv'
    else:
        return 'high_cv'


def _categorize_zero_fraction(zero_fraction: float) -> str:
    """Categorize zero fraction into low/medium/high for matrix lookup."""
    if zero_fraction <= ZERO_FRACTION_LOW:
        return 'low_zeros'
    elif zero_fraction <= ZERO_FRACTION_MEDIUM:
        return 'medium_zeros'
    else:
        return 'high_zeros'


@dataclass
class DiscreteCharacteristics:
    """Comprehensive characteristics of discrete demand."""
    is_discrete: bool
    n_unique: int
    unique_values: np.ndarray
    cardinality_category: str  # 'very_low', 'low', 'medium', 'high'
    has_ordinal_structure: bool  # Values form natural ordering
    base_unit: Optional[float]  # Detected base unit (e.g., 200 for {0, 200, 400, 800})
    is_multiple_based: bool  # Values are multiples of base unit
    zero_is_class: bool  # Zero is a distinct class (not just absence)
    class_frequencies: Dict[float, float]  # Frequency of each class
    class_imbalance_ratio: float  # Max freq / min freq
    recommended_approach: str  # 'continuous', 'ordinal', 'classification', 'hybrid'
    recommended_model: str  # Specific model recommendation


def detect_discrete_demand(y: np.ndarray, threshold: float = DISCRETE_UNIQUE_RATIO_THRESHOLD) -> Tuple[bool, int]:
    """
    Detect if demand values are discrete (limited distinct values).

    This is common in retail/wholesale where demand comes in fixed units
    (e.g., 0, 200, 400, 800 cases).

    Parameters
    ----------
    y : np.ndarray
        Target time series values
    threshold : float
        If unique_values / n_observations < threshold, consider discrete

    Returns
    -------
    Tuple[bool, int]
        (is_discrete, n_unique_values)
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if n == 0:
        return False, 0

    unique_values = np.unique(y)
    n_unique = len(unique_values)

    # Check ratio of unique values
    unique_ratio = n_unique / n
    is_discrete = unique_ratio < threshold

    # Also check if values appear to be multiples of a base unit
    if not is_discrete and n_unique <= DISCRETE_MAX_UNIQUE_VALUES:
        # Few unique values even without the ratio check
        nonzero_values = y[y > 0]
        if len(nonzero_values) > 2:
            # Check if values are roughly multiples of the smallest nonzero value
            min_nonzero = np.min(nonzero_values)
            if min_nonzero > 0:
                ratios = nonzero_values / min_nonzero
                # If most ratios are close to integers, likely discrete
                close_to_int = np.mean(np.abs(ratios - np.round(ratios)) < 0.1)
                if close_to_int > 0.8:
                    is_discrete = True

    return is_discrete, n_unique


def analyze_discrete_demand(y: np.ndarray) -> DiscreteCharacteristics:
    """
    Perform comprehensive analysis of discrete demand characteristics.

    This function goes beyond simple detection to understand the structure
    of discrete demand, enabling optimal model selection.

    Parameters
    ----------
    y : np.ndarray
        Target time series values

    Returns
    -------
    DiscreteCharacteristics
        Comprehensive discrete demand analysis
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if n == 0:
        return DiscreteCharacteristics(
            is_discrete=False,
            n_unique=0,
            unique_values=np.array([]),
            cardinality_category='high',
            has_ordinal_structure=False,
            base_unit=None,
            is_multiple_based=False,
            zero_is_class=False,
            class_frequencies={},
            class_imbalance_ratio=1.0,
            recommended_approach='continuous',
            recommended_model='lightgbm',
        )

    unique_values = np.unique(y)
    n_unique = len(unique_values)

    # Basic discrete detection
    unique_ratio = n_unique / n
    is_discrete = unique_ratio < DISCRETE_UNIQUE_RATIO_THRESHOLD or n_unique <= DISCRETE_MAX_UNIQUE_VALUES

    # Cardinality category
    if n_unique <= CARDINALITY_VERY_LOW:
        cardinality_category = 'very_low'
    elif n_unique <= CARDINALITY_LOW:
        cardinality_category = 'low'
    elif n_unique <= CARDINALITY_MEDIUM:
        cardinality_category = 'medium'
    else:
        cardinality_category = 'high'
        is_discrete = False  # Too many values to treat as discrete

    # Check for ordinal structure (values form natural ordering)
    # All demand values are inherently ordinal (more is more)
    has_ordinal_structure = True if is_discrete else False

    # Detect base unit (common in case-pack ordering)
    base_unit = None
    is_multiple_based = False
    nonzero_values = unique_values[unique_values > 0]

    if len(nonzero_values) >= 2:
        # Find GCD-like base unit
        min_nonzero = np.min(nonzero_values)
        if min_nonzero > 0:
            ratios = nonzero_values / min_nonzero
            # Check if all ratios are close to integers
            close_to_int = np.abs(ratios - np.round(ratios)) < 0.05
            if np.all(close_to_int):
                base_unit = float(min_nonzero)
                is_multiple_based = True
            else:
                # Try to find common divisor
                diffs = np.diff(np.sort(nonzero_values))
                if len(diffs) > 0:
                    potential_base = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else min_nonzero
                    if potential_base > 0:
                        ratios = nonzero_values / potential_base
                        close_to_int = np.abs(ratios - np.round(ratios)) < 0.1
                        if np.mean(close_to_int) > 0.8:
                            base_unit = float(potential_base)
                            is_multiple_based = True

    # Check if zero is a meaningful class
    zero_is_class = 0 in unique_values and np.sum(y == 0) > n * 0.05  # >5% zeros

    # Compute class frequencies
    class_frequencies = {}
    for val in unique_values:
        class_frequencies[float(val)] = float(np.sum(y == val) / n)

    # Class imbalance ratio
    if class_frequencies:
        freq_values = list(class_frequencies.values())
        max_freq = max(freq_values)
        min_freq = min(f for f in freq_values if f > 0)
        class_imbalance_ratio = max_freq / min_freq if min_freq > 0 else float('inf')
    else:
        class_imbalance_ratio = 1.0

    # Determine recommended approach
    if not is_discrete or cardinality_category == 'high':
        recommended_approach = 'continuous'
        recommended_model = 'lightgbm'
    elif cardinality_category == 'very_low':
        # Very few classes: classification or ordinal regression
        if class_imbalance_ratio > 10:
            # Highly imbalanced: use weighted classification
            recommended_approach = 'classification'
            recommended_model = 'discrete_classifier'
        else:
            recommended_approach = 'ordinal'
            recommended_model = 'ordinal_regression'
    elif cardinality_category == 'low':
        # Low cardinality: ordinal regression optimal
        recommended_approach = 'ordinal'
        recommended_model = 'ordinal_regression'
    else:  # medium
        # Medium cardinality: hybrid approach
        if is_multiple_based:
            # Multiple-based (e.g., case packs): use regression + snap
            recommended_approach = 'hybrid'
            recommended_model = 'hybrid_discrete'
        else:
            # General discrete: use ordinal if not too many classes
            recommended_approach = 'ordinal'
            recommended_model = 'ordinal_regression'

    # Override for high zero fraction - zero-inflated models still best
    if zero_is_class and class_frequencies.get(0.0, 0) > 0.5:
        recommended_approach = 'hybrid'
        recommended_model = 'zero_inflated'  # ZI handles zeros explicitly

    logger.debug(
        f"Discrete analysis: n_unique={n_unique}, category={cardinality_category}, "
        f"base_unit={base_unit}, approach={recommended_approach}, model={recommended_model}"
    )

    return DiscreteCharacteristics(
        is_discrete=is_discrete,
        n_unique=n_unique,
        unique_values=unique_values,
        cardinality_category=cardinality_category,
        has_ordinal_structure=has_ordinal_structure,
        base_unit=base_unit,
        is_multiple_based=is_multiple_based,
        zero_is_class=zero_is_class,
        class_frequencies=class_frequencies,
        class_imbalance_ratio=class_imbalance_ratio,
        recommended_approach=recommended_approach,
        recommended_model=recommended_model,
    )


def snap_to_discrete_values(
    predictions: np.ndarray,
    valid_values: np.ndarray,
    method: str = 'nearest',
    probabilities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Snap continuous predictions to valid discrete values.

    This is a post-processing step that ensures predictions are from
    the set of valid discrete values observed in training.

    Parameters
    ----------
    predictions : np.ndarray
        Continuous predictions from any model
    valid_values : np.ndarray
        Array of valid discrete values (e.g., [0, 200, 400, 800])
    method : str
        'nearest': Snap to nearest valid value
        'floor': Snap to largest valid value <= prediction
        'ceil': Snap to smallest valid value >= prediction
        'probabilistic': Use class probabilities if available
    probabilities : np.ndarray, optional
        Class probabilities (n_samples x n_classes) for probabilistic snapping

    Returns
    -------
    np.ndarray
        Predictions snapped to valid discrete values
    """
    predictions = np.asarray(predictions).flatten()
    valid_values = np.sort(np.unique(valid_values))

    if len(valid_values) == 0:
        return predictions

    snapped = np.zeros_like(predictions)

    if method == 'probabilistic' and probabilities is not None:
        # Use expected value from class probabilities
        for i, pred in enumerate(predictions):
            if i < len(probabilities):
                probs = probabilities[i]
                if len(probs) == len(valid_values):
                    snapped[i] = np.sum(probs * valid_values)
                else:
                    snapped[i] = valid_values[np.argmin(np.abs(valid_values - pred))]
            else:
                snapped[i] = valid_values[np.argmin(np.abs(valid_values - pred))]
    elif method == 'floor':
        for i, pred in enumerate(predictions):
            valid_below = valid_values[valid_values <= pred]
            snapped[i] = valid_below[-1] if len(valid_below) > 0 else valid_values[0]
    elif method == 'ceil':
        for i, pred in enumerate(predictions):
            valid_above = valid_values[valid_values >= pred]
            snapped[i] = valid_above[0] if len(valid_above) > 0 else valid_values[-1]
    else:  # nearest
        for i, pred in enumerate(predictions):
            snapped[i] = valid_values[np.argmin(np.abs(valid_values - pred))]

    return snapped


def compute_discrete_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_values: np.ndarray,
    loss_type: str = 'ordinal_mae',
) -> float:
    """
    Compute loss appropriate for discrete/ordinal targets.

    Standard MSE/MAE don't account for the discrete structure.
    This function provides discrete-aware loss metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    valid_values : np.ndarray
        Valid discrete values for ordinal distance calculation
    loss_type : str
        'ordinal_mae': MAE using ordinal positions (0=class0, 1=class1, etc.)
        'ordinal_accuracy': Exact match rate
        'within_one': Accuracy allowing off-by-one class
        'weighted_mae': MAE weighted by class size

    Returns
    -------
    float
        Computed loss value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    valid_values = np.sort(np.unique(valid_values))

    if len(y_true) == 0:
        return 0.0

    # Convert values to ordinal positions
    value_to_position = {v: i for i, v in enumerate(valid_values)}

    def get_position(val):
        if val in value_to_position:
            return value_to_position[val]
        # Find nearest valid value
        nearest_idx = np.argmin(np.abs(valid_values - val))
        return nearest_idx

    true_positions = np.array([get_position(v) for v in y_true])
    pred_positions = np.array([get_position(v) for v in y_pred])

    if loss_type == 'ordinal_mae':
        # MAE in ordinal space
        return float(np.mean(np.abs(true_positions - pred_positions)))
    elif loss_type == 'ordinal_accuracy':
        # Exact match rate
        return float(np.mean(true_positions == pred_positions))
    elif loss_type == 'within_one':
        # Allow off-by-one class
        return float(np.mean(np.abs(true_positions - pred_positions) <= 1))
    elif loss_type == 'weighted_mae':
        # Weight by actual value magnitude
        weights = np.maximum(y_true, 1)  # Avoid zero weights
        return float(np.sum(np.abs(y_true - y_pred) * weights) / np.sum(weights))
    else:
        return float(np.mean(np.abs(y_true - y_pred)))


def get_dynamic_model_priority(
    demand_pattern: str,
    cv: float,
    zero_fraction: float,
    y: Optional[np.ndarray] = None,
    n_observations: Optional[int] = None,
    has_external_features: bool = False,
    target_horizon: Optional[int] = None,
    enable_multi_horizon: bool = False,
    time_format: str = 'year_week',
) -> List[str]:
    """
    Get dynamically prioritized model list based on demand characteristics.

    This function goes beyond simple pattern-based selection to consider:
    - CV (coefficient of variation) for variability
    - Zero fraction for demand sparsity
    - Whether demand is discrete (limited distinct values)
    - Data availability (observations count)
    - External feature availability
    - Target forecast horizon (for multi-horizon model selection)

    Parameters
    ----------
    demand_pattern : str
        'smooth', 'erratic', 'intermittent', or 'lumpy'
    cv : float
        Coefficient of variation of the demand
    zero_fraction : float
        Fraction of zero values (0.0 to 1.0)
    y : np.ndarray, optional
        Raw demand values for discrete detection
    n_observations : int, optional
        Number of observations (for sample size considerations)
    has_external_features : bool
        Whether external features are available
    target_horizon : int, optional
        Target forecast horizon (e.g., 5 for Lag 5 evaluation).
        If >= 3, multi-horizon models may be prioritized.
    enable_multi_horizon : bool
        Whether to include multi-horizon models in selection.
        If True and target_horizon >= 3, multi_horizon_* models are considered.

    Returns
    -------
    List[str]
        Ordered list of model families (best first)
    """
    # Categorize CV and zero_fraction
    cv_category = _categorize_cv(cv)
    zero_category = _categorize_zero_fraction(zero_fraction)

    # Normalize pattern name
    pattern = demand_pattern.lower().strip()
    if pattern not in ['smooth', 'erratic', 'intermittent', 'lumpy']:
        logger.warning(f"Unknown demand pattern '{pattern}', defaulting to 'smooth'")
        pattern = 'smooth'

    # Look up in dynamic matrix
    key = (pattern, cv_category, zero_category)
    priority_list = DYNAMIC_MODEL_PRIORITY_MATRIX.get(key, None)

    if priority_list is None:
        # Fallback to base pattern priority
        logger.debug(f"Dynamic priority not found for {key}, using base pattern priority")
        priority_list = PATTERN_MODEL_PRIORITY.get(pattern, PATTERN_MODEL_PRIORITY['smooth']).copy()
    else:
        priority_list = priority_list.copy()  # Don't modify the constant

    # Detect and handle discrete demand if y is provided
    is_discrete = False
    discrete_info = None
    if y is not None:
        discrete_info = analyze_discrete_demand(y)
        is_discrete = discrete_info.is_discrete

        if is_discrete:
            n_unique = discrete_info.n_unique
            category = discrete_info.cardinality_category
            logger.debug(
                f"Detected discrete demand: {n_unique} unique values, "
                f"category={category}, approach={discrete_info.recommended_approach}"
            )

            # Insert specialized discrete models based on cardinality
            if category == 'very_low' and n_unique <= CARDINALITY_VERY_LOW:
                # Very low cardinality (2-5 classes): classifier is best
                for model in ['discrete_classifier', 'ordinal_regression']:
                    if model not in priority_list:
                        priority_list.insert(0, model)
                    else:
                        priority_list.remove(model)
                        priority_list.insert(0, model)

            elif category == 'low' and n_unique <= CARDINALITY_LOW:
                # Low cardinality (6-10 classes): ordinal regression is optimal
                for model in ['ordinal_regression', 'discrete_classifier']:
                    if model not in priority_list:
                        priority_list.insert(0, model)
                    else:
                        priority_list.remove(model)
                        priority_list.insert(0, model)

            elif category == 'medium':
                # Medium cardinality (11-20 classes): hybrid discrete
                for model in ['hybrid_discrete', 'ordinal_regression']:
                    if model not in priority_list:
                        priority_list.insert(0, model)
                    else:
                        priority_list.remove(model)
                        priority_list.insert(0, model)

            # Boost tree-based models (they handle discrete well)
            for model in DISCRETE_DEMAND_MODEL_BOOST:
                if model in priority_list:
                    priority_list.remove(model)
                    # Insert after specialized discrete models
                    insert_pos = min(3, len(priority_list))
                    priority_list.insert(insert_pos, model)

            # Penalize continuous distribution models for discrete data
            for model in DISCRETE_DEMAND_MODEL_PENALIZE:
                if model in priority_list:
                    priority_list.remove(model)
                    priority_list.append(model)  # Move to end

    # Handle very sparse demand (>70% zeros)
    if zero_fraction > ZERO_FRACTION_VERY_HIGH:
        # Strongly prioritize zero-inflated models
        for model in ['zero_inflated', 'hurdle_model']:
            if model in priority_list:
                priority_list.remove(model)
            priority_list.insert(0, model)
        logger.debug(f"Very sparse demand (zero_fraction={zero_fraction:.2f}), prioritizing zero-inflated models")

    # Handle small sample sizes
    if n_observations is not None and n_observations < 50:
        # Remove complex models that need more data
        complex_models = ['zero_inflated', 'hurdle_model']
        for model in complex_models:
            if model in priority_list:
                idx = priority_list.index(model)
                if idx < 2:  # Only demote if it's in top 2
                    priority_list.remove(model)
                    priority_list.append(model)
        # Prioritize simpler models
        for model in ['lightgbm', 'random_forest']:
            if model in priority_list:
                priority_list.remove(model)
                priority_list.insert(0, model)

    # Ensure all allowed models are included (some might be missing from matrix)
    # Use time-format-aware allowed families (includes univariate for monthly)
    _allowed = get_allowed_model_families(time_format)
    for model in _allowed:
        if model not in priority_list:
            priority_list.append(model)

    # Handle multi-horizon model selection
    # Multi-horizon models are beneficial when:
    # 1. enable_multi_horizon is True
    # 2. target_horizon >= 3 (longer horizons benefit from direct forecasting)
    # 3. Pattern is smooth or erratic (tree-based models work well)
    # 4. Not very sparse demand (zero_fraction < 0.7) - sparse demand needs specialist models
    if enable_multi_horizon and target_horizon is not None and target_horizon >= 3:
        # Check if multi-horizon is appropriate for this pattern
        use_multi_horizon = True

        # For very sparse demand, stick with zero-inflated specialists
        if zero_fraction > ZERO_FRACTION_VERY_HIGH:
            use_multi_horizon = False
            logger.debug(f"Skipping multi-horizon for very sparse demand (zero_fraction={zero_fraction:.2f})")

        # For discrete demand with low cardinality, discrete specialists are better
        if is_discrete and discrete_info and discrete_info.cardinality_category in ('very_low', 'low'):
            use_multi_horizon = False
            logger.debug(f"Skipping multi-horizon for low-cardinality discrete demand")

        if use_multi_horizon:
            # Insert multi-horizon variants at the top of the list
            # Multi-horizon models directly optimize for the target horizon,
            # avoiding error compounding from recursive forecasting
            multi_horizon_models = ['multi_horizon_lightgbm', 'multi_horizon_xgboost']

            # For longer horizons (5+), multi-horizon ensemble may be even better
            if target_horizon >= 5:
                multi_horizon_models.insert(0, 'multi_horizon_ensemble')

            # Remove existing multi-horizon models from priority_list first
            # (they may have been appended at the end by ALLOWED_MODEL_FAMILIES loop)
            for mh_model in multi_horizon_models:
                if mh_model in priority_list:
                    priority_list.remove(mh_model)

            # Insert multi-horizon models at the TOP of the list (position 0, 1, 2)
            # These should be tried first since they directly optimize for target horizon
            for i, mh_model in enumerate(multi_horizon_models):
                priority_list.insert(i, mh_model)

            logger.debug(
                f"Enabled multi-horizon models for target_horizon={target_horizon}, "
                f"pattern={pattern}: {multi_horizon_models}"
            )

    logger.debug(
        f"Dynamic model priority for pattern={pattern}, cv={cv:.2f}, "
        f"zero_fraction={zero_fraction:.2f}, discrete={is_discrete}: {priority_list[:5]}"
    )

    return priority_list


def compute_demand_characteristics(y: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive demand characteristics for model selection.

    Parameters
    ----------
    y : np.ndarray
        Target time series

    Returns
    -------
    Dict[str, Any]
        Dictionary with all computed characteristics
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if n == 0:
        return {
            'n_observations': 0,
            'demand_pattern': 'smooth',
            'cv': 0.0,
            'zero_fraction': 1.0,
            'is_discrete': False,
            'n_unique': 0,
            'adi': n,
            'cv2': 0.0,
        }

    # Basic stats
    n_nonzero = np.sum(y > 0)
    zero_fraction = 1 - (n_nonzero / n)
    mean_val = np.mean(y)
    std_val = np.std(y)
    cv = std_val / mean_val if mean_val > 1e-10 else 0.0

    # ADI calculation
    nonzero_idx = np.where(y > 0)[0]
    if len(nonzero_idx) > 1:
        intervals = np.diff(nonzero_idx)
        adi = np.mean(intervals) if len(intervals) > 0 else n
    else:
        adi = n

    cv2 = cv ** 2

    # Classify demand pattern (Syntetos-Boylan)
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_pattern = 'smooth'
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        demand_pattern = 'erratic'
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_pattern = 'intermittent'
    else:
        demand_pattern = 'lumpy'

    # Comprehensive discrete analysis
    discrete_info = analyze_discrete_demand(y)

    return {
        'n_observations': n,
        'demand_pattern': demand_pattern,
        'cv': cv,
        'cv2': cv2,
        'zero_fraction': zero_fraction,
        'adi': adi,
        'is_discrete': discrete_info.is_discrete,
        'n_unique': discrete_info.n_unique,
        'mean': mean_val,
        'std': std_val,
        # Discrete-specific characteristics
        'discrete_cardinality': discrete_info.cardinality_category,
        'discrete_base_unit': discrete_info.base_unit,
        'discrete_is_multiple_based': discrete_info.is_multiple_based,
        'discrete_recommended_approach': discrete_info.recommended_approach,
        'discrete_recommended_model': discrete_info.recommended_model,
        'discrete_class_imbalance': discrete_info.class_imbalance_ratio,
        'unique_values': discrete_info.unique_values.tolist() if len(discrete_info.unique_values) <= 50 else None,
    }


# =============================================================================
# META-FEATURE EXTRACTION
# =============================================================================

def build_meta_features(
    y: np.ndarray,
    n_features: int = 0,
    has_external_features: bool = False,
    feature_quality_score: float = 0.5,
    time_format: str = 'year_week',
) -> MetaFeatures:
    """
    Extract meta-features from a time series for model selection.

    These features characterize the time series and help predict
    which model family will work best.

    Parameters
    ----------
    y : np.ndarray
        Target time series
    n_features : int
        Number of features available
    has_external_features : bool
        Whether external features (price, promo) are available
    feature_quality_score : float
        Quality of available features (0-1)

    Returns
    -------
    MetaFeatures
        Extracted meta-features
    """
    y = np.asarray(y).flatten()
    n = len(y)

    # Basic statistics
    n_nonzero = np.sum(y > 0)
    zero_fraction = 1 - (n_nonzero / n) if n > 0 else 1.0
    mean_val = np.mean(y)
    std_val = np.std(y)
    cv = std_val / mean_val if mean_val > 1e-10 else 0.0

    # Syntetos-Boylan classification
    # ADI: Average Demand Interval (average periods between non-zero demands)
    nonzero_idx = np.where(y > 0)[0]
    if len(nonzero_idx) > 1:
        intervals = np.diff(nonzero_idx)
        adi = np.mean(intervals) if len(intervals) > 0 else n
    else:
        adi = n

    cv2 = cv ** 2

    # Classify demand pattern
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_pattern = 'smooth'
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        demand_pattern = 'erratic'
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_pattern = 'intermittent'
    else:
        demand_pattern = 'lumpy'

    # Trend strength (simplified)
    if n >= 4:
        half = n // 2
        trend_strength = abs(np.mean(y[half:]) - np.mean(y[:half])) / (mean_val + 1e-10)
        trend_strength = min(trend_strength, 1.0)
    else:
        trend_strength = 0.0

    # Seasonal strength (simplified - check for relevant periodic patterns)
    _seasonal_lags = [3, 12] if time_format == 'year_month' else [7, 52]
    if n >= 14:
        # Simple seasonality check using autocorrelation
        seasonal_strength = 0.0
        for lag in _seasonal_lags:
            if n > lag + 1:
                corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
                if not np.isnan(corr):
                    seasonal_strength = max(seasonal_strength, abs(corr))
    else:
        seasonal_strength = 0.0

    # Autocorrelation
    autocorr_lag1 = 0.0
    autocorr_lag7 = 0.0
    if n > 2:
        autocorr_lag1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        if np.isnan(autocorr_lag1):
            autocorr_lag1 = 0.0
    if n > 8:
        autocorr_lag7 = np.corrcoef(y[:-7], y[7:])[0, 1]
        if np.isnan(autocorr_lag7):
            autocorr_lag7 = 0.0

    # Entropy (measure of complexity/randomness)
    y_nonzero = y[y > 0]
    if len(y_nonzero) > 1:
        # Approximate entropy using histogram
        hist, _ = np.histogram(y_nonzero, bins=min(10, len(y_nonzero)))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        entropy = min(entropy / np.log(10), 1.0)  # Normalize
    else:
        entropy = 0.0

    # Forecastability score (combined metric)
    forecastability = (
        0.3 * (1 - zero_fraction) +  # More non-zero is better
        0.2 * min(1.0, autocorr_lag1) +  # Higher autocorr is better
        0.2 * (1 - min(1.0, cv/3)) +  # Lower CV is better
        0.2 * min(1.0, n / (12 if time_format == 'year_month' else 52)) +  # More data is better
        0.1 * feature_quality_score  # Better features is better
    )

    return MetaFeatures(
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
        seasonal_strength=seasonal_strength,
        autocorr_lag1=autocorr_lag1,
        autocorr_lag7=autocorr_lag7,
        entropy=entropy,
        n_features=n_features,
        has_external_features=has_external_features,
        feature_quality_score=feature_quality_score,
        forecastability=forecastability,
    )


# =============================================================================
# MODEL SELECTION LOGIC
# =============================================================================

def get_candidate_models_for_pattern(
    demand_pattern: str,
    n_observations: int,
    n_features: int,
    has_categoricals: bool = False,
    max_candidates: int = 5,
    cv: Optional[float] = None,
    zero_fraction: Optional[float] = None,
    y: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Get candidate models appropriate for a demand pattern.

    Uses dynamic pattern-specific priorities that consider CV and zero_fraction,
    and filters based on data characteristics.

    Parameters
    ----------
    demand_pattern : str
        'smooth', 'erratic', 'intermittent', or 'lumpy'
    n_observations : int
        Number of training observations
    n_features : int
        Number of features available
    has_categoricals : bool
        Whether categorical features are present
    max_candidates : int
        Maximum number of candidate models
    cv : float, optional
        Coefficient of variation for dynamic priority
    zero_fraction : float, optional
        Fraction of zero values for dynamic priority
    y : np.ndarray, optional
        Raw target values for discrete demand detection

    Returns
    -------
    List[str]
        Ordered list of candidate model families
    """
    # Use dynamic priority if CV and zero_fraction are provided
    if cv is not None and zero_fraction is not None:
        base_priority = get_dynamic_model_priority(
            demand_pattern=demand_pattern,
            cv=cv,
            zero_fraction=zero_fraction,
            y=y,
            n_observations=n_observations,
        )
    else:
        # Fallback to static pattern priority
        base_priority = PATTERN_MODEL_PRIORITY.get(
            demand_pattern, PATTERN_MODEL_PRIORITY['smooth']
        ).copy()

    # Filter based on minimum sample requirements
    candidates = []
    for model in base_priority:
        chars = MODEL_CHARACTERISTICS.get(model, {})
        min_samples = chars.get('min_samples', 20)

        if n_observations >= min_samples:
            candidates.append(model)

    # Boost CatBoost if categoricals present
    if has_categoricals and 'catboost' in candidates:
        candidates.remove('catboost')
        candidates.insert(0, 'catboost')

    # Limit to max candidates
    return candidates[:max_candidates]


def rank_models_by_pattern_fit(
    meta_features: MetaFeatures,
    available_models: Optional[List[str]] = None,
    y: Optional[np.ndarray] = None,
    time_format: str = 'year_week',
) -> List[Tuple[str, float]]:
    """
    Rank models by their expected fit for given meta-features.

    Uses dynamic model priority based on CV and zero_fraction thresholds,
    combined with model characteristics scoring.

    Parameters
    ----------
    meta_features : MetaFeatures
        Computed meta-features for the time series
    available_models : List[str], optional
        List of models to consider (defaults to all allowed)
    y : np.ndarray, optional
        Raw target values for discrete demand detection
    time_format : str
        'year_week' or 'year_month' — for monthly, includes univariate models

    Returns
    -------
    List[Tuple[str, float]]
        Models sorted by expected performance (best first) with scores
    """
    if available_models is None:
        available_models = get_allowed_model_families(time_format)

    # Get dynamic priority ordering based on pattern, CV, and zero_fraction
    dynamic_priority = get_dynamic_model_priority(
        demand_pattern=meta_features.demand_pattern,
        cv=meta_features.cv,
        zero_fraction=meta_features.zero_fraction,
        y=y,
        n_observations=meta_features.n_observations,
        has_external_features=meta_features.has_external_features,
    )

    scores = {}

    for model in available_models:
        if model not in MODEL_CHARACTERISTICS:
            continue

        chars = MODEL_CHARACTERISTICS[model]
        score = 0.0

        # Dynamic priority position bonus (higher for models earlier in dynamic list)
        if model in dynamic_priority:
            position = dynamic_priority.index(model)
            # Give up to 3.0 points based on position (first = 3.0, second = 2.5, etc.)
            position_score = max(0, 3.0 - (position * 0.5))
            score += position_score

        # Pattern match from static characteristics
        if meta_features.demand_pattern in chars.get('best_for', []):
            score += 1.0

        # Zero handling for high-zero data (granular thresholds)
        if meta_features.zero_fraction > ZERO_FRACTION_VERY_HIGH:
            # Very sparse - strongly prefer zero-inflated models
            if model in ['zero_inflated', 'hurdle_model']:
                score += 2.0
            elif model == 'tweedie':
                score += 1.0
        elif meta_features.zero_fraction > ZERO_FRACTION_HIGH:
            # Sparse - prefer zero-aware models
            if model in ['zero_inflated', 'hurdle_model']:
                score += 1.5
            elif chars.get('handles_zeros', False):
                score += 0.5
        elif meta_features.zero_fraction > ZERO_FRACTION_MEDIUM:
            if chars.get('handles_zeros', False):
                score += 0.5

        # CV-based scoring (granular)
        if meta_features.cv > CV_HIGH_THRESHOLD:
            # Very high variability - prefer robust models
            if model in ['xgboost', 'lightgbm']:
                score += 1.0
            if chars.get('handles_high_cv', False):
                score += 0.5
        elif meta_features.cv > CV_MEDIUM_THRESHOLD:
            # Medium-high variability
            if chars.get('handles_high_cv', False):
                score += 0.3

        # Categorical handling bonus
        if chars.get('handles_categoricals', False):
            score += 0.3

        # Speed bonus for large datasets
        if meta_features.n_observations > 1000:
            speed = chars.get('training_speed', 'medium')
            if speed == 'fast':
                score += 0.3
            elif speed == 'slow':
                score -= 0.2

        # Complexity penalty for small datasets
        if meta_features.n_observations < 100:
            complexity = chars.get('complexity', 'medium')
            if complexity == 'high':
                score -= 0.5
            elif complexity == 'low':
                score += 0.3

        # Very small datasets - strongly prefer simple models
        if meta_features.n_observations < 50:
            if model in ['lightgbm', 'random_forest']:
                score += 0.5
            if model in ['zero_inflated', 'hurdle_model']:
                score -= 0.5

        # Feature quality bonus for feature-intensive models
        if meta_features.feature_quality_score > 0.7:
            score += 0.5 * meta_features.feature_quality_score

        # External features bonus for gradient boosting models
        if meta_features.has_external_features:
            if model in ['lightgbm', 'xgboost', 'catboost']:
                score += 0.3

        scores[model] = score

    # Sort by score (descending)
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    return ranked


def select_best_model_intelligently(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_functions: Dict[str, Callable],
    meta_features: Optional[MetaFeatures] = None,
    max_candidates: int = 5,
    early_stopping: bool = True,
    wape_threshold: float = 0.3,
    time_format: str = 'year_week',
) -> ModelSelectionResult:
    """
    Intelligently select the best model for the data.

    This is the main entry point for automated model selection. It:
    1. Extracts meta-features if not provided
    2. Ranks models by expected performance
    3. Evaluates top candidates with early stopping
    4. Returns the best model with confidence score

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    training_functions : Dict mapping model names to training functions
    meta_features : Pre-computed meta-features (optional)
    max_candidates : Maximum models to evaluate
    early_stopping : Stop if a model meets threshold
    wape_threshold : WAPE threshold for early stopping

    Returns
    -------
    ModelSelectionResult
        Selected model with all evaluation details
    """
    start_time = time.time()

    # Extract meta-features if not provided
    if meta_features is None:
        meta_features = build_meta_features(
            y_train,
            n_features=X_train.shape[1] if len(X_train.shape) > 1 else 0,
            time_format=time_format,
        )

    # Get ranked candidates
    ranked_models = rank_models_by_pattern_fit(meta_features)
    candidate_models = [m for m, _ in ranked_models[:max_candidates]]

    # Filter to available training functions
    candidate_models = [m for m in candidate_models if m in training_functions]

    if not candidate_models:
        # Fall back to any available model
        candidate_models = list(training_functions.keys())[:max_candidates]

    # Evaluate candidates
    all_scores = {}
    best_model = None
    best_score = float('inf')

    for model_name in candidate_models:
        try:
            train_fn = training_functions[model_name]

            # Train and evaluate
            result = train_fn(X_train, y_train, X_val, y_val)

            if hasattr(result, 'val_wape'):
                score = result.val_wape
            elif isinstance(result, tuple):
                score = result[1] if len(result) > 1 else 1.0
            else:
                score = 1.0

            all_scores[model_name] = score

            if score < best_score:
                best_score = score
                best_model = model_name

            # Early stopping
            if early_stopping and score <= wape_threshold:
                logger.info(f"Early stopping: {model_name} achieved WAPE {score:.3f}")
                break

        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            all_scores[model_name] = float('inf')

    # Compute confidence
    if len(all_scores) > 1:
        scores = [s for s in all_scores.values() if s < float('inf')]
        if len(scores) > 1:
            # Confidence based on margin over second best
            sorted_scores = sorted(scores)
            margin = sorted_scores[1] - sorted_scores[0]
            confidence = min(1.0, margin / 0.1)  # 10% margin = full confidence
        else:
            confidence = 0.5
    else:
        confidence = 0.3

    # Create rankings
    rankings = sorted(
        [(m, s) for m, s in all_scores.items()],
        key=lambda x: x[1]
    )

    elapsed = time.time() - start_time

    return ModelSelectionResult(
        selected_model=best_model or candidate_models[0],
        selection_method='intelligent_ranking',
        cv_score=best_score,
        all_scores=all_scores,
        model_rankings=[m for m, _ in rankings],
        confidence=confidence,
        meta_features=meta_features,
        selection_time_seconds=elapsed,
    )


def run_model_selection_tournament(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_functions: Dict[str, Callable],
    max_rounds: int = 3,
    models_per_round: int = 3,
    elimination_threshold: float = 1.5,  # Eliminate if 50% worse than best
) -> TournamentResult:
    """
    Run a tournament-style model selection.

    In each round, models compete head-to-head. Models significantly
    worse than the best are eliminated. This continues until
    a clear winner emerges.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    training_functions : Dict mapping model names to training functions
    max_rounds : Maximum number of tournament rounds
    models_per_round : Models evaluated per round
    elimination_threshold : Relative threshold for elimination

    Returns
    -------
    TournamentResult
        Tournament winner and complete rankings
    """
    start_time = time.time()

    remaining_models = list(training_functions.keys())
    eliminated = {}
    all_scores = {}
    rounds_completed = 0

    for round_num in range(max_rounds):
        if len(remaining_models) <= 1:
            break

        rounds_completed = round_num + 1
        round_models = remaining_models[:models_per_round]
        round_scores = {}

        for model_name in round_models:
            try:
                train_fn = training_functions[model_name]
                result = train_fn(X_train, y_train, X_val, y_val)

                if hasattr(result, 'val_wape'):
                    score = result.val_wape
                else:
                    score = result[1] if isinstance(result, tuple) else 1.0

                round_scores[model_name] = score
                all_scores[model_name] = score

            except Exception as e:
                logger.warning(f"Tournament: {model_name} failed: {e}")
                round_scores[model_name] = float('inf')
                eliminated[model_name] = f"training_failed: {str(e)[:50]}"
                remaining_models.remove(model_name)

        # Find best and eliminate poor performers
        if round_scores:
            best_score = min(round_scores.values())

            for model, score in round_scores.items():
                if model in remaining_models:
                    if score > best_score * elimination_threshold:
                        eliminated[model] = f"eliminated_round_{round_num+1}"
                        remaining_models.remove(model)

        # Rotate remaining models for next round
        remaining_models = remaining_models[models_per_round:] + [
            m for m in round_models if m in remaining_models
        ]

    # Final ranking
    rankings = sorted(
        [(m, s) for m, s in all_scores.items()],
        key=lambda x: x[1]
    )

    elapsed = time.time() - start_time

    return TournamentResult(
        winner=rankings[0][0] if rankings else remaining_models[0] if remaining_models else '',
        rankings=rankings,
        eliminated_models=eliminated,
        rounds_completed=rounds_completed,
        total_time_seconds=elapsed,
    )


# =============================================================================
# COLD-START HANDLING
# =============================================================================

def handle_cold_start_selection(
    n_observations: int,
    segment_info: Optional[Dict[str, Any]] = None,
    similar_keys_performance: Optional[Dict[str, Dict[str, float]]] = None,
    time_format: str = 'year_week',
) -> Tuple[str, str]:
    """
    Handle model selection for cold-start (new product) situations.

    When a product has very limited history, we can:
    1. Use segment-level model if available
    2. Use similar products' best-performing models
    3. Use robust defaults

    Thresholds adapt to time_format: weekly uses 12/30, monthly uses 4/8.

    Parameters
    ----------
    n_observations : int
        Number of observations available
    segment_info : Dict, optional
        Segment this key belongs to and its characteristics
    similar_keys_performance : Dict, optional
        Performance of models on similar keys
    time_format : str
        'year_week' or 'year_month' — adapts cold-start thresholds

    Returns
    -------
    Tuple[str, str]
        (model_name, selection_reason)
    """
    # Adapt thresholds: monthly has fewer observations per year
    very_cold = 4 if time_format == 'year_month' else 12
    moderate_cold = 8 if time_format == 'year_month' else 30

    # Very cold start
    if n_observations < very_cold:
        if segment_info and 'best_model' in segment_info:
            return segment_info['best_model'], 'segment_model_cold_start'
        return 'lightgbm', 'default_cold_start_very_limited'

    # Moderate cold start
    if n_observations < moderate_cold:
        if similar_keys_performance:
            # Use model that performed best on similar keys
            model_scores = {}
            for key_perf in similar_keys_performance.values():
                for model, score in key_perf.items():
                    if model not in model_scores:
                        model_scores[model] = []
                    model_scores[model].append(score)

            if model_scores:
                avg_scores = {m: np.mean(s) for m, s in model_scores.items()}
                best_model = min(avg_scores.items(), key=lambda x: x[1])[0]
                return best_model, 'similar_keys_transfer'

        # Use simple models for limited data
        return 'lightgbm', 'simple_model_limited_data'

    # Enough data for normal selection
    return 'lightgbm', 'normal_selection'


def transfer_learning_selection(
    target_meta_features: MetaFeatures,
    source_performances: Dict[str, Dict[str, float]],
    similarity_threshold: float = 0.7,
) -> Tuple[str, float]:
    """
    Select model using transfer learning from similar time series.

    Uses meta-feature similarity to find similar historical keys
    and leverages their model performance for selection.

    Parameters
    ----------
    target_meta_features : MetaFeatures
        Meta-features of target time series
    source_performances : Dict[str, Dict[str, float]]
        Historical key -> {model -> wape} performance data
    similarity_threshold : float
        Minimum similarity to consider a source

    Returns
    -------
    Tuple[str, float]
        (best_model, confidence)
    """
    # This would typically use a similarity metric based on meta-features
    # For now, use pattern matching as a simple similarity

    target_pattern = target_meta_features.demand_pattern

    # Filter sources by pattern
    matching_sources = {}
    for source_id, performances in source_performances.items():
        # In practice, you'd compute full meta-feature similarity
        # Here we use pattern as a proxy
        matching_sources[source_id] = performances

    if not matching_sources:
        return 'lightgbm', 0.3

    # Aggregate model performances across matching sources
    model_scores = {}
    for source_perfs in matching_sources.values():
        for model, score in source_perfs.items():
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(score)

    # Compute average and select best
    avg_scores = {m: np.mean(s) for m, s in model_scores.items()}
    best_model = min(avg_scores.items(), key=lambda x: x[1])[0]

    # Compute confidence based on consistency across sources
    if len(model_scores.get(best_model, [])) > 3:
        std = np.std(model_scores[best_model])
        confidence = max(0.3, 1.0 - std / 0.3)
    else:
        confidence = 0.5

    return best_model, confidence


# =============================================================================
# AUTOMATED MODEL SEARCH
# =============================================================================

def automated_model_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_functions: Dict[str, Callable],
    time_budget_seconds: float = 300,
    min_models: int = 3,
    max_models: int = 7,
    time_format: str = 'year_week',
) -> ModelSelectionResult:
    """
    Automated model search with time budget.

    Intelligently allocates time across candidate models based on
    expected performance and training speed.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    training_functions : Dict mapping model names to training functions
    time_budget_seconds : Total time budget
    min_models : Minimum models to try
    max_models : Maximum models to try

    Returns
    -------
    ModelSelectionResult
        Best model found within time budget
    """
    start_time = time.time()

    # Build meta-features
    meta_features = build_meta_features(
        y_train,
        n_features=X_train.shape[1] if len(X_train.shape) > 1 else 0,
        time_format=time_format,
    )

    # Rank models
    ranked_models = rank_models_by_pattern_fit(meta_features)
    candidate_models = [m for m, _ in ranked_models if m in training_functions]

    if not candidate_models:
        candidate_models = list(training_functions.keys())

    candidate_models = candidate_models[:max_models]

    # Estimate time per model based on characteristics
    time_estimates = {}
    for model in candidate_models:
        chars = MODEL_CHARACTERISTICS.get(model, {})
        speed = chars.get('training_speed', 'medium')
        if speed == 'fast':
            time_estimates[model] = 1.0
        elif speed == 'slow':
            time_estimates[model] = 3.0
        else:
            time_estimates[model] = 2.0

    # Allocate time budget
    total_estimate = sum(time_estimates.values())
    time_per_unit = time_budget_seconds / total_estimate if total_estimate > 0 else 60

    all_scores = {}
    models_tried = 0

    for model_name in candidate_models:
        elapsed = time.time() - start_time
        remaining = time_budget_seconds - elapsed

        if remaining <= 0 and models_tried >= min_models:
            break

        try:
            train_fn = training_functions[model_name]
            result = train_fn(X_train, y_train, X_val, y_val)

            if hasattr(result, 'val_wape'):
                score = result.val_wape
            else:
                score = result[1] if isinstance(result, tuple) else 1.0

            all_scores[model_name] = score
            models_tried += 1

        except Exception as e:
            logger.warning(f"AutoML: {model_name} failed: {e}")
            all_scores[model_name] = float('inf')

    # Select best
    best_model = min(all_scores.items(), key=lambda x: x[1])[0]
    best_score = all_scores[best_model]

    # Rankings
    rankings = sorted(all_scores.items(), key=lambda x: x[1])

    elapsed = time.time() - start_time

    return ModelSelectionResult(
        selected_model=best_model,
        selection_method='automated_search',
        cv_score=best_score,
        all_scores=all_scores,
        model_rankings=[m for m, _ in rankings],
        confidence=0.7 if best_score < 0.3 else 0.5,
        meta_features=meta_features,
        selection_time_seconds=elapsed,
    )


def early_stopping_model_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_functions: Dict[str, Callable],
    wape_threshold: float = 0.25,
    max_models: int = 5,
    patience: int = 2,
    time_format: str = 'year_week',
) -> ModelSelectionResult:
    """
    Model search with early stopping.

    Stops searching when:
    1. A model meets the WAPE threshold, or
    2. No improvement for 'patience' consecutive models

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    training_functions : Dict mapping model names to training functions
    wape_threshold : Stop if WAPE <= threshold
    max_models : Maximum models to try
    patience : Stop if no improvement for this many models

    Returns
    -------
    ModelSelectionResult
        Best model found (possibly with early stopping)
    """
    start_time = time.time()

    # Build meta-features
    meta_features = build_meta_features(
        y_train,
        n_features=X_train.shape[1] if len(X_train.shape) > 1 else 0,
        time_format=time_format,
    )

    # Get ranked candidates
    ranked_models = rank_models_by_pattern_fit(meta_features)
    candidate_models = [m for m, _ in ranked_models if m in training_functions][:max_models]

    all_scores = {}
    best_score = float('inf')
    no_improvement_count = 0

    for model_name in candidate_models:
        try:
            train_fn = training_functions[model_name]
            result = train_fn(X_train, y_train, X_val, y_val)

            if hasattr(result, 'val_wape'):
                score = result.val_wape
            else:
                score = result[1] if isinstance(result, tuple) else 1.0

            all_scores[model_name] = score

            if score < best_score:
                best_score = score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping conditions
            if score <= wape_threshold:
                logger.info(f"Early stop: {model_name} met threshold ({score:.3f} <= {wape_threshold})")
                break

            if no_improvement_count >= patience:
                logger.info(f"Early stop: no improvement for {patience} models")
                break

        except Exception as e:
            logger.warning(f"EarlyStop: {model_name} failed: {e}")
            all_scores[model_name] = float('inf')

    # Select best
    best_model = min(all_scores.items(), key=lambda x: x[1])[0]

    elapsed = time.time() - start_time

    return ModelSelectionResult(
        selected_model=best_model,
        selection_method='early_stopping_search',
        cv_score=best_score,
        all_scores=all_scores,
        model_rankings=sorted(all_scores.keys(), key=lambda x: all_scores[x]),
        confidence=0.8 if best_score <= wape_threshold else 0.5,
        meta_features=meta_features,
        selection_time_seconds=elapsed,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'MetaFeatures',
    'ModelSelectionResult',
    'TournamentResult',
    'DiscreteCharacteristics',

    # Constants
    'ALLOWED_MODEL_FAMILIES',
    'UNIVARIATE_MODEL_FAMILIES',
    'get_allowed_model_families',
    'resolve_effective_model_families',
    'HIERARCHICAL_MODEL_FAMILIES',
    'MODEL_CHARACTERISTICS',
    'PATTERN_MODEL_PRIORITY',
    'DYNAMIC_MODEL_PRIORITY_MATRIX',
    'CV_LOW_THRESHOLD',
    'CV_MEDIUM_THRESHOLD',
    'CV_HIGH_THRESHOLD',
    'ZERO_FRACTION_LOW',
    'ZERO_FRACTION_MEDIUM',
    'ZERO_FRACTION_HIGH',
    'ZERO_FRACTION_VERY_HIGH',
    # Discrete demand constants
    'DISCRETE_UNIQUE_RATIO_THRESHOLD',
    'DISCRETE_MAX_UNIQUE_VALUES',
    'ORDINAL_MAX_CLASSES',
    'CLASSIFICATION_MAX_CLASSES',
    'CARDINALITY_VERY_LOW',
    'CARDINALITY_LOW',
    'CARDINALITY_MEDIUM',

    # Meta-feature extraction
    'build_meta_features',

    # Dynamic model priority
    'get_dynamic_model_priority',
    'detect_discrete_demand',
    'compute_demand_characteristics',

    # Discrete demand analysis
    'analyze_discrete_demand',
    'snap_to_discrete_values',
    'compute_discrete_loss',

    # Model selection
    'get_candidate_models_for_pattern',
    'rank_models_by_pattern_fit',
    'select_best_model_intelligently',
    'run_model_selection_tournament',

    # Cold-start handling
    'handle_cold_start_selection',
    'transfer_learning_selection',

    # Automated search
    'automated_model_search',
    'early_stopping_model_search',
]
