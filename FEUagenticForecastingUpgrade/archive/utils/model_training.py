# utils/model_training.py
"""
Pre-Canned Model Training & Hyperparameter Tuning Functions
============================================================

This module provides ready-to-use training functions for ALL model families
defined in config.yaml. Agents MUST use these functions instead of writing
training code from scratch - this dramatically reduces context usage.

Model Categories:
    1. Tree-Based: lightgbm, xgboost, catboost, random_forest
    2. Statistical: arima, sarima, ets, theta, tbats
    3. Bayesian/Probabilistic: bsts, prophet
    4. Intermittent Specialists: croston, sba, tsb, imapa
    5. Compound/Hurdle: zero_inflated, hurdle_model, tweedie
    6. Deep Learning: tft, lstm, nbeats, deepar, wavenet
    7. Ensemble: weighted_ensemble, stacking

Usage:
    from utils.model_training import (
        train_lightgbm, train_xgboost, train_catboost, train_random_forest,
        train_croston, train_sba, train_tsb, train_imapa,
        train_zero_inflated, train_hurdle_model, train_tweedie,
        train_arima, train_sarima, train_ets, train_theta, train_tbats,
        train_prophet, train_bsts,
        train_weighted_ensemble, train_stacking,
        train_model_by_name, train_best_model_for_segment,
        tune_model_hyperparameters,
    )

    # Train a specific model
    model, metrics = train_lightgbm(X_train, y_train, X_val, y_val)

    # Train best model for demand pattern
    model, metrics = train_best_model_for_segment(
        X_train, y_train, X_val, y_val,
        demand_pattern='intermittent',
        allowed_families=['croston', 'sba', 'lightgbm']
    )

NOTE: If you add a new model type to config.yaml, you MUST add a corresponding
train_<model_name>() function here.

MULTI-HORIZON TRAINING:
For forecasts evaluated at longer horizons (e.g., Lag 5), use the multi-horizon
training functions which directly optimize for the target horizon:
    - train_multi_horizon_lightgbm: LightGBM with direct multi-step training
    - train_multi_horizon_xgboost: XGBoost with direct multi-step training
    - train_multi_horizon_ensemble: Ensemble optimized for target horizon
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
import joblib

# Import multi-horizon training module
try:
    from utils.multi_horizon_training import (
        train_multi_horizon_lightgbm,
        train_multi_horizon_xgboost,
        train_multi_horizon_ensemble,
        train_multi_horizon_model,
        MultiHorizonTrainingResult,
        MultiHorizonModel,
        DEFAULT_HORIZON_WEIGHTS,
        LAG5_FOCUSED_WEIGHTS,
    )
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False

# Suppress common warnings during training
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
except (ImportError, NameError):
    pass

logger = logging.getLogger(__name__)

# Try to import ConvergenceWarning
try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    ConvergenceWarning = UserWarning


# =============================================================================
# DATA CLASSES FOR TRAINING RESULTS
# =============================================================================

@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any  # Trained model object
    model_type: str
    hyperparameters: Dict[str, Any]
    train_wape: float
    val_wape: float
    train_mae: float = 0.0
    val_mae: float = 0.0
    train_rmse: float = 0.0
    val_rmse: float = 0.0
    training_time_seconds: float = 0.0
    is_feature_based: bool = True
    feature_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (excluding model object)."""
        return {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'train_wape': self.train_wape,
            'val_wape': self.val_wape,
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'train_rmse': self.train_rmse,
            'val_rmse': self.val_rmse,
            'training_time_seconds': self.training_time_seconds,
            'is_feature_based': self.is_feature_based,
        }


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""
    best_model: Any
    best_params: Dict[str, Any]
    best_val_wape: float
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0
    tuning_time_seconds: float = 0.0


# =============================================================================
# TWO-STAGE MODEL WRAPPER CLASSES (with predict method)
# =============================================================================

class ZeroInflatedModel:
    """
    Wrapper for zero-inflated two-stage model with predict() method.

    Stage 1: Classification - P(demand > 0)
    Stage 2: Regression - E[demand | demand > 0]

    Prediction modes (controlled by zero_threshold):
    - zero_threshold = None: Classic P(>0) * E[|>0] (soft multiplication)
    - zero_threshold = 0.5: If P(>0) < 0.5 → predict 0, else predict E[|>0]
    - zero_threshold = 'auto': Automatically set based on zero_fraction

    The threshold approach is better for highly intermittent data where
    the classic P*E approach often over-forecasts when actual = 0.
    """

    def __init__(self, classifier, regressor, mean_positive: float,
                 classifier_type: str = 'lightgbm', regressor_type: str = 'lightgbm',
                 zero_fraction: float = 0.0, zero_threshold: float = None):
        self.classifier = classifier
        self.regressor = regressor
        self.mean_positive = mean_positive
        self.classifier_type = classifier_type
        self.regressor_type = regressor_type
        self.model_type = 'zero_inflated'
        self.zero_fraction = zero_fraction
        # Set default threshold based on zero_fraction
        self.zero_threshold = self._compute_optimal_threshold(zero_fraction) if zero_threshold == 'auto' else zero_threshold

    def _compute_optimal_threshold(self, zero_fraction: float) -> float:
        """
        Compute optimal probability threshold based on zero_fraction.

        Higher zero_fraction → higher threshold (more conservative, predict more zeros)

        Rationale:
        - If 80% of actuals are zero, we should only predict positive when very confident
        - If 20% of actuals are zero, we can be more liberal with positive predictions
        """
        if zero_fraction >= 0.7:
            return 0.6  # Very conservative - need 60% confidence to predict positive
        elif zero_fraction >= 0.5:
            return 0.5  # Balanced
        elif zero_fraction >= 0.3:
            return 0.4  # More liberal
        else:
            return 0.3  # Low intermittency - be liberal

    def predict(self, X: np.ndarray, zero_threshold: float = None) -> np.ndarray:
        """
        Generate predictions using two-stage approach.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        zero_threshold : float, optional
            Override threshold for this prediction. If None, uses instance threshold.
            If instance threshold is also None, uses classic P*E approach.

        Returns
        -------
        np.ndarray
            Predictions (non-negative)
        """
        X = np.asarray(X)

        # Use provided threshold, else instance threshold, else None (classic P*E)
        threshold = zero_threshold if zero_threshold is not None else self.zero_threshold

        # Stage 1: Get probability of positive demand
        if hasattr(self.classifier, 'predict_proba'):
            prob_positive = self.classifier.predict_proba(X)[:, 1]
        else:
            prob_positive = self.classifier.predict(X).astype(float)

        # Stage 2: Get expected value given positive demand
        if self.regressor is not None:
            expected_positive = self.regressor.predict(X)
        else:
            expected_positive = np.full(len(X), self.mean_positive)

        # Apply threshold if specified (recommended for intermittent data)
        if threshold is not None:
            # Hard threshold: below threshold → 0, above → use regressor output
            predictions = np.where(
                prob_positive < threshold,
                0.0,
                np.clip(expected_positive, 0, None)
            )
        else:
            # Classic soft multiplication: P(positive) * E[value|positive]
            predictions = prob_positive * np.clip(expected_positive, 0, None)

        return predictions


class HurdleModel:
    """
    Wrapper for hurdle two-stage model with predict() method.

    Similar to zero-inflated but uses balanced class weights for better
    zero/non-zero separation. The "hurdle" is whether any demand occurs.

    Prediction modes (controlled by zero_threshold):
    - zero_threshold = None: Classic P(>0) * E[|>0] (soft multiplication)
    - zero_threshold = 0.5: If P(>0) < 0.5 → predict 0, else predict E[|>0]
    - zero_threshold = 'auto': Automatically set based on zero_fraction

    The threshold approach is better for lumpy demand where the decision
    to "cross the hurdle" should be more decisive.
    """

    def __init__(self, classifier, regressor, mean_positive: float,
                 classifier_type: str = 'lightgbm', regressor_type: str = 'lightgbm',
                 zero_fraction: float = 0.0, zero_threshold: float = None):
        self.classifier = classifier
        self.regressor = regressor
        self.mean_positive = mean_positive
        self.classifier_type = classifier_type
        self.regressor_type = regressor_type
        self.model_type = 'hurdle_model'
        self.zero_fraction = zero_fraction
        # Set default threshold based on zero_fraction
        self.zero_threshold = self._compute_optimal_threshold(zero_fraction) if zero_threshold == 'auto' else zero_threshold

    def _compute_optimal_threshold(self, zero_fraction: float) -> float:
        """
        Compute optimal probability threshold based on zero_fraction.

        For Hurdle models (typically used for lumpy demand), we use slightly
        higher thresholds than ZeroInflated because lumpy demand has more
        structural zeros that we should respect.
        """
        if zero_fraction >= 0.7:
            return 0.65  # Very conservative for lumpy data
        elif zero_fraction >= 0.5:
            return 0.55  # Moderately conservative
        elif zero_fraction >= 0.3:
            return 0.45  # Balanced
        else:
            return 0.35  # Lower intermittency

    def predict(self, X: np.ndarray, zero_threshold: float = None) -> np.ndarray:
        """
        Generate predictions using two-stage hurdle approach.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        zero_threshold : float, optional
            Override threshold for this prediction. If None, uses instance threshold.
            If instance threshold is also None, uses classic P*E approach.

        Returns
        -------
        np.ndarray
            Predictions (non-negative)
        """
        X = np.asarray(X)

        # Use provided threshold, else instance threshold, else None (classic P*E)
        threshold = zero_threshold if zero_threshold is not None else self.zero_threshold

        # Stage 1: Get probability of positive demand
        if hasattr(self.classifier, 'predict_proba'):
            prob_positive = self.classifier.predict_proba(X)[:, 1]
        else:
            prob_positive = self.classifier.predict(X).astype(float)

        # Stage 2: Get expected value given positive demand
        if self.regressor is not None:
            expected_positive = self.regressor.predict(X)
        else:
            expected_positive = np.full(len(X), self.mean_positive)

        # Apply threshold if specified (recommended for lumpy data)
        if threshold is not None:
            # Hard threshold: below threshold → 0 (don't cross the hurdle)
            predictions = np.where(
                prob_positive < threshold,
                0.0,
                np.clip(expected_positive, 0, None)
            )
        else:
            # Classic soft multiplication: P(positive) * E[value|positive]
            predictions = prob_positive * np.clip(expected_positive, 0, None)

        return predictions


# =============================================================================
# METRIC COMPUTATION HELPERS
# =============================================================================

def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    total_actual = np.sum(np.abs(y_true))
    if total_actual < 1e-10:
        return 0.0 if np.sum(np.abs(y_pred)) < 1e-10 else 1.0
    return np.sum(np.abs(y_true - y_pred)) / total_actual


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics at once."""
    return {
        'wape': compute_wape(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'rmse': compute_rmse(y_true, y_pred),
    }


# =============================================================================
# POST-PREDICTION ZERO CLIPPING (ADI-Based)
# =============================================================================

def apply_adi_based_zero_clip(
    predictions: np.ndarray,
    key_adi: float = None,
    key_zero_fraction: float = None,
    recent_predictions: List[float] = None,
    clip_threshold: float = None,
    min_adi_for_clipping: float = 2.0,
) -> np.ndarray:
    """
    Post-prediction zero clipping based on key's ADI (Average Demand Interval).

    For intermittent keys (high ADI), if predictions are below a threshold AND
    the recent pattern suggests zeros are continuing, clip predictions to 0.

    This is a simple, effective post-processing step that catches many
    over-forecasted zeros that tree-based models tend to produce.

    Parameters
    ----------
    predictions : np.ndarray
        Raw model predictions (can be scalar or array)
    key_adi : float, optional
        Key's Average Demand Interval. Higher ADI = more intermittent.
        If None, no clipping is applied.
    key_zero_fraction : float, optional
        Key's zero fraction from training data. Used as fallback if ADI not available.
    recent_predictions : List[float], optional
        Recent predictions for this key (for pattern detection).
        If provided and most are near-zero, more likely to clip.
    clip_threshold : float, optional
        Threshold below which predictions are clipped to 0.
        If None, auto-computed from ADI or zero_fraction.
    min_adi_for_clipping : float, default=2.0
        Minimum ADI required to apply any clipping.
        Keys with ADI < 2 are considered non-intermittent.

    Returns
    -------
    np.ndarray
        Predictions with low values clipped to 0 for intermittent keys

    Examples
    --------
    >>> preds = np.array([0.5, 2.0, 0.3, 15.0])
    >>> apply_adi_based_zero_clip(preds, key_adi=4.0)
    array([ 0. ,  2. ,  0. , 15. ])

    >>> # Key with low ADI - no clipping
    >>> apply_adi_based_zero_clip(preds, key_adi=1.5)
    array([ 0.5,  2. ,  0.3, 15. ])
    """
    predictions = np.asarray(predictions).flatten()

    # No key-level info → return as-is
    if key_adi is None and key_zero_fraction is None:
        return predictions

    # Determine if this key is intermittent enough to warrant clipping
    is_intermittent = False
    if key_adi is not None and key_adi >= min_adi_for_clipping:
        is_intermittent = True
    elif key_zero_fraction is not None and key_zero_fraction >= 0.3:
        is_intermittent = True

    if not is_intermittent:
        return predictions  # Not intermittent enough, don't clip

    # Compute clip threshold if not provided
    if clip_threshold is None:
        if key_adi is not None and key_adi > 0:
            # Higher ADI → lower threshold (more aggressive clipping)
            # ADI = 2: threshold ≈ 0.5
            # ADI = 4: threshold ≈ 0.25
            # ADI = 10: threshold ≈ 0.1
            clip_threshold = 1.0 / key_adi
        elif key_zero_fraction is not None:
            # Higher zero_fraction → lower threshold
            if key_zero_fraction >= 0.7:
                clip_threshold = 0.5
            elif key_zero_fraction >= 0.5:
                clip_threshold = 0.3
            else:
                clip_threshold = 0.2
        else:
            clip_threshold = 0.3  # Default

    # Additional check: if recent predictions were mostly zeros, be more aggressive
    recent_zero_boost = 1.0
    if recent_predictions is not None and len(recent_predictions) >= 2:
        recent_preds = np.asarray(recent_predictions[-5:])  # Last 5
        n_recent_zeros = np.sum(recent_preds < clip_threshold)
        if n_recent_zeros >= len(recent_preds) * 0.6:  # 60%+ were zeros
            recent_zero_boost = 1.5  # More aggressive clipping

    adjusted_threshold = clip_threshold * recent_zero_boost

    # Apply clipping
    clipped = np.where(predictions < adjusted_threshold, 0.0, predictions)

    return clipped


def apply_batch_adi_zero_clip(
    predictions_df,
    key_col: str,
    pred_col: str = 'predicted',
    key_metadata: dict = None,
    adi_col: str = 'adi',
    zero_fraction_col: str = 'zero_fraction',
) -> None:
    """
    Apply ADI-based zero clipping to a batch of predictions (in-place).

    This function modifies the predictions_df in-place, applying zero-clipping
    to each key based on its ADI or zero_fraction.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions. Modified in-place.
    key_col : str
        Column name for key identifier
    pred_col : str
        Column name for predictions (default: 'predicted')
    key_metadata : dict, optional
        Dictionary mapping key → {adi: float, zero_fraction: float}
        If None, will try to use columns in predictions_df
    adi_col : str
        Column name for ADI in predictions_df or key_metadata
    zero_fraction_col : str
        Column name for zero_fraction in predictions_df or key_metadata

    Returns
    -------
    None (modifies predictions_df in-place)

    Also adds column 'zero_clipped' indicating which predictions were clipped.
    """
    import pandas as pd

    if pred_col not in predictions_df.columns:
        logger.warning(f"Prediction column '{pred_col}' not found, skipping zero clipping")
        return

    # Track clipping
    predictions_df['zero_clipped'] = False
    clip_count = 0

    for key in predictions_df[key_col].unique():
        key_mask = predictions_df[key_col] == key
        key_preds = predictions_df.loc[key_mask, pred_col].values

        # Get key metadata
        key_adi = None
        key_zf = None

        if key_metadata is not None and key in key_metadata:
            key_adi = key_metadata[key].get('adi') or key_metadata[key].get(adi_col)
            key_zf = key_metadata[key].get('zero_fraction') or key_metadata[key].get(zero_fraction_col)
        elif adi_col in predictions_df.columns:
            key_adi = predictions_df.loc[key_mask, adi_col].iloc[0] if key_mask.any() else None
        if zero_fraction_col in predictions_df.columns and key_zf is None:
            key_zf = predictions_df.loc[key_mask, zero_fraction_col].iloc[0] if key_mask.any() else None

        # Apply clipping
        clipped_preds = apply_adi_based_zero_clip(
            key_preds,
            key_adi=key_adi,
            key_zero_fraction=key_zf,
        )

        # Track which were clipped
        was_clipped = (key_preds != clipped_preds) & (clipped_preds == 0)
        clip_count += np.sum(was_clipped)

        predictions_df.loc[key_mask, pred_col] = clipped_preds
        predictions_df.loc[key_mask, 'zero_clipped'] = was_clipped

    if clip_count > 0:
        logger.info(f"ADI-based zero clipping: {clip_count} predictions clipped to 0")


def _categorize_feature_counts(feature_cols: List[str]) -> Dict[str, int]:
    """
    Categorize feature columns by type and return counts.

    This provides a summary of feature types for the model spec without
    listing all individual columns. Categories are detected by naming patterns.

    Categories:
    - lag_features: Columns with '_lag_' pattern (e.g., qty_lag_4)
    - rolling_features: Columns with '_roll_' or '_rolling_' pattern
    - ewm_features: Columns with '_ewm_' pattern
    - calendar_features: Columns like week_of_year, month, day_of_week, etc.
    - cyclical_features: Columns with '_sin' or '_cos' suffix
    - fourier_features: Columns with 'fourier_' pattern
    - seasonal_features: Columns with '_seasonal_' pattern
    - key_level_features: Columns with '_key_' or key-level stats pattern
    - cross_sectional_features: Columns with 'cross_' or '_pct_' pattern
    - trend_features: Columns with '_trend_' pattern
    - intermittency_features: Columns with 'zero_', 'adi_', 'cv2_' pattern
    - external_features: All other columns (from config: price, promo, weather, etc.)
    """
    categories = {
        'lag_features': 0,
        'rolling_features': 0,
        'ewm_features': 0,
        'calendar_features': 0,
        'cyclical_features': 0,
        'fourier_features': 0,
        'seasonal_features': 0,
        'key_level_features': 0,
        'cross_sectional_features': 0,
        'trend_features': 0,
        'intermittency_features': 0,
        'external_features': 0,
    }

    calendar_patterns = ['week_of_year', 'month', 'day_of_week', 'quarter', 'year', 'is_weekend']

    for col in feature_cols:
        col_lower = col.lower()

        if '_lag_' in col_lower and '_lag_diff' not in col_lower and '_lag_ratio' not in col_lower:
            categories['lag_features'] += 1
        elif '_roll_' in col_lower or '_rolling_' in col_lower:
            categories['rolling_features'] += 1
        elif '_ewm_' in col_lower:
            categories['ewm_features'] += 1
        elif col_lower.endswith('_sin') or col_lower.endswith('_cos'):
            categories['cyclical_features'] += 1
        elif 'fourier_' in col_lower:
            categories['fourier_features'] += 1
        elif '_seasonal_' in col_lower or '_yoy_' in col_lower or '_qoq_' in col_lower:
            categories['seasonal_features'] += 1
        elif any(p in col_lower for p in calendar_patterns):
            categories['calendar_features'] += 1
        elif '_key_' in col_lower or col_lower.endswith('_mean') or col_lower.endswith('_std'):
            # Key-level stats like key_mean, key_std, key_cv
            if not ('_roll_' in col_lower or '_rolling_' in col_lower):
                categories['key_level_features'] += 1
            else:
                categories['rolling_features'] += 1
        elif 'cross_' in col_lower or '_pct_' in col_lower or '_rank_' in col_lower:
            categories['cross_sectional_features'] += 1
        elif '_trend_' in col_lower:
            categories['trend_features'] += 1
        elif any(p in col_lower for p in ['zero_frac', 'adi_', 'cv2_', 'intermit', '_demand_']):
            categories['intermittency_features'] += 1
        else:
            # External features from config (price, promo, weather, etc.)
            categories['external_features'] += 1

    return categories


# =============================================================================
# TREE-BASED MODEL TRAINING FUNCTIONS
# =============================================================================

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    **kwargs
) -> TrainingResult:
    """
    Train LightGBM model with optional custom parameters.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    params : Custom hyperparameters (merged with defaults)
    early_stopping_rounds : Rounds for early stopping

    Returns
    -------
    TrainingResult with trained model and metrics
    """
    import lightgbm as lgb
    import time

    start_time = time.time()

    # Default parameters (battle-tested for demand forecasting)
    default_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
    }

    # Merge with custom params
    final_params = {**default_params, **(params or {}), **kwargs}
    n_estimators = final_params.pop('n_estimators', 500)

    # Remove XGBoost-specific params that are incompatible with LightGBM
    xgb_only_params = ['tree_method', 'grow_policy', 'eval_metric', 'gamma',
                       'scale_pos_weight', 'base_score', 'booster', 'missing']
    for p in xgb_only_params:
        final_params.pop(p, None)
    # Ensure LightGBM-compatible objective
    if final_params.get('objective') in ['reg:squarederror', 'reg:linear', 'binary:logistic']:
        final_params['objective'] = 'regression'

    # Remove classification objectives incompatible with LGBMRegressor
    if final_params.get('objective') in ('multiclass', 'multiclassova', 'binary',
                                          'cross_entropy', 'cross_entropy_lambda'):
        final_params['objective'] = 'regression'
        final_params.pop('num_class', None)
        final_params.pop('num_classes', None)

    # Remove classification metrics incompatible with regression objective.
    # This MUST be done here (not just in retrain_all_models hp_clean) because
    # train_lightgbm can be called from multiple entry points (ensemble member_params,
    # train_model_by_name, etc.) that may pass unsanitized params.
    _classification_metrics = {
        'multi_logloss', 'multi_error', 'auc_mu', 'binary_logloss',
        'binary_error', 'cross_entropy', 'cross_entropy_lambda',
        'auc', 'average_precision',
    }
    _metric = final_params.get('metric')
    if isinstance(_metric, str) and _metric in _classification_metrics:
        final_params['metric'] = 'mae'
    elif isinstance(_metric, list):
        final_params['metric'] = [m for m in _metric if m not in _classification_metrics]
        if not final_params['metric']:
            final_params['metric'] = 'mae'

    # Remove discrete demand metadata that may leak from original training specs
    _discrete_metadata = {
        'n_classes', 'unique_values', 'train_accuracy', 'val_accuracy',
        'n_unique_values', 'base_unit', 'cardinality_category',
        'val_wape_before_snap', 'val_wape_after_snap',
        'is_unbalance', 'class_weight', 'multi_strategy',
    }
    for p in _discrete_metadata:
        final_params.pop(p, None)

    # Tweedie/Poisson objectives require non-negative targets
    if final_params.get('objective') in ('tweedie', 'poisson'):
        y_train = np.clip(np.asarray(y_train).flatten(), 0, None)
        y_val = np.clip(np.asarray(y_val).flatten(), 0, None)

    # Create model
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **final_params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )

    # Compute metrics
    train_pred = np.clip(model.predict(X_train), 0, None)
    val_pred = np.clip(model.predict(X_val), 0, None)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='lightgbm',
        hyperparameters=final_params,
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    **kwargs
) -> TrainingResult:
    """Train XGBoost model with optimal defaults for demand forecasting."""
    import xgboost as xgb
    import time

    start_time = time.time()

    default_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'random_state': 42,
        'n_jobs': -1,
    }

    final_params = {**default_params, **(params or {}), **kwargs}

    # Remove LightGBM-specific params that are incompatible with XGBoost
    lgb_only_params = ['num_leaves', 'min_child_samples', 'min_split_gain',
                       'bagging_fraction', 'feature_fraction', 'bagging_freq',
                       'tweedie_variance_power', 'metric']
    for p in lgb_only_params:
        final_params.pop(p, None)
    # Ensure XGBoost-compatible objective
    if final_params.get('objective') in ['regression', 'tweedie', 'binary', 'mse', 'mae']:
        final_params['objective'] = 'reg:squarederror'

    model = xgb.XGBRegressor(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    train_pred = np.clip(model.predict(X_train), 0, None)
    val_pred = np.clip(model.predict(X_val), 0, None)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='xgboost',
        hyperparameters=final_params,
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    **kwargs
) -> TrainingResult:
    """Train CatBoost model with optimal defaults."""
    from catboost import CatBoostRegressor
    import time

    start_time = time.time()

    default_params = {
        'iterations': 500,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False,
        'early_stopping_rounds': early_stopping_rounds,
    }

    final_params = {**default_params, **(params or {}), **kwargs}

    # Remove params incompatible with CatBoost
    incompatible_params = ['objective', 'n_estimators', 'max_depth', 'num_leaves',
                           'min_child_samples', 'subsample', 'colsample_bytree',
                           'reg_alpha', 'reg_lambda', 'n_jobs', 'verbosity',
                           'tree_method', 'grow_policy', 'eval_metric', 'gamma',
                           'tweedie_variance_power', 'metric', 'min_child_weight']
    for p in incompatible_params:
        final_params.pop(p, None)

    model = CatBoostRegressor(**final_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    train_pred = np.clip(model.predict(X_train), 0, None)
    val_pred = np.clip(model.predict(X_val), 0, None)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='catboost',
        hyperparameters=final_params,
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> TrainingResult:
    """Train Random Forest model with optimal defaults."""
    from sklearn.ensemble import RandomForestRegressor
    import time

    start_time = time.time()

    default_params = {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
    }

    final_params = {**default_params, **(params or {}), **kwargs}

    # Remove params incompatible with RandomForest (sklearn)
    incompatible_params = ['objective', 'learning_rate', 'num_leaves', 'subsample',
                           'colsample_bytree', 'reg_alpha', 'reg_lambda', 'verbosity',
                           'tree_method', 'grow_policy', 'eval_metric', 'gamma',
                           'tweedie_variance_power', 'metric', 'min_child_weight',
                           'iterations', 'depth', 'l2_leaf_reg', 'verbose']
    for p in incompatible_params:
        final_params.pop(p, None)

    model = RandomForestRegressor(**final_params)
    model.fit(X_train, y_train)

    train_pred = np.clip(model.predict(X_train), 0, None)
    val_pred = np.clip(model.predict(X_val), 0, None)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='random_forest',
        hyperparameters=final_params,
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


# =============================================================================
# INTERMITTENT DEMAND SPECIALIST TRAINING FUNCTIONS
# =============================================================================

def train_croston(
    y_train: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 0.1,
    **kwargs
) -> TrainingResult:
    """
    Train Croston's method for intermittent demand.

    Croston's method separately forecasts:
    1. Demand size (when demand occurs)
    2. Inter-arrival interval (time between demands)
    Final forecast = demand_size / interval

    Best for: INTERMITTENT demand (low CV, high zero%)
    """
    import time

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()

    # Fit Croston's method
    non_zero_idx = np.where(y_train > 0)[0]

    if len(non_zero_idx) == 0:
        # No demand - predict zero
        demand_level = 0
        interval = float('inf')
    else:
        # Initialize
        demand_level = y_train[non_zero_idx[0]]
        interval = non_zero_idx[0] + 1 if non_zero_idx[0] > 0 else 1

        # Exponential smoothing updates
        for i in range(1, len(non_zero_idx)):
            idx = non_zero_idx[i]
            prev_idx = non_zero_idx[i - 1]

            demand_level = alpha * y_train[idx] + (1 - alpha) * demand_level
            inter_arrival = idx - prev_idx
            interval = alpha * inter_arrival + (1 - alpha) * interval

    # Calculate forecast
    if interval == 0 or interval == float('inf') or demand_level == 0:
        forecast = 0
    else:
        forecast = demand_level / interval

    # Store model as dict (Croston is simple enough)
    model = {
        'type': 'croston',
        'alpha': alpha,
        'demand_level': demand_level,
        'interval': interval,
        'forecast': max(0, forecast),
    }

    # Compute metrics
    train_pred = np.full(len(y_train), max(0, forecast))
    val_pred = np.full(len(y_val), max(0, forecast))

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='croston',
        hyperparameters={'alpha': alpha},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


def train_sba(
    y_train: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 0.1,
    **kwargs
) -> TrainingResult:
    """
    Train Syntetos-Boylan Approximation (SBA) - bias-corrected Croston.

    SBA corrects for positive bias in Croston's method:
    forecast = croston_forecast * (1 - alpha/2)
    """
    import time

    start_time = time.time()

    # First get Croston result
    croston_result = train_croston(y_train, y_val, alpha)
    croston_forecast = croston_result.model['forecast']

    # Apply SBA correction
    correction_factor = 1 - alpha / 2
    sba_forecast = croston_forecast * correction_factor

    # Update model
    model = {
        'type': 'sba',
        'alpha': alpha,
        'demand_level': croston_result.model['demand_level'],
        'interval': croston_result.model['interval'],
        'croston_forecast': croston_forecast,
        'correction_factor': correction_factor,
        'forecast': max(0, sba_forecast),
    }

    # Recompute metrics with SBA forecast
    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()

    train_pred = np.full(len(y_train), max(0, sba_forecast))
    val_pred = np.full(len(y_val), max(0, sba_forecast))

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='sba',
        hyperparameters={'alpha': alpha},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


def train_tsb(
    y_train: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.1,
    **kwargs
) -> TrainingResult:
    """
    Train Teunter-Syntetos-Babai (TSB) method with demand probability decay.

    TSB models demand probability decay during periods of no demand.
    Better for items with obsolescence risk.

    Best for: LUMPY demand or items with potential obsolescence
    """
    import time

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()

    # Initialize
    non_zero = y_train[y_train > 0]
    if len(non_zero) == 0:
        demand_level = 0
        demand_prob = 0
    else:
        demand_level = np.mean(non_zero)
        demand_prob = len(non_zero) / len(y_train)

        # Exponential smoothing with probability decay
        for val in y_train:
            if val > 0:
                demand_level = alpha * val + (1 - alpha) * demand_level
                demand_prob = alpha * 1.0 + (1 - alpha) * demand_prob
            else:
                demand_prob = (1 - beta) * demand_prob

    # TSB forecast
    forecast = demand_level * demand_prob

    model = {
        'type': 'tsb',
        'alpha': alpha,
        'beta': beta,
        'demand_level': demand_level,
        'demand_prob': demand_prob,
        'forecast': max(0, forecast),
    }

    train_pred = np.full(len(y_train), max(0, forecast))
    val_pred = np.full(len(y_val), max(0, forecast))

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='tsb',
        hyperparameters={'alpha': alpha, 'beta': beta},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


def train_imapa(
    y_train: np.ndarray,
    y_val: np.ndarray,
    aggregation_levels: List[int] = None,
    base_method: str = 'croston',
    **kwargs
) -> TrainingResult:
    """
    Train IMAPA (Intermittent Multiple Aggregation Prediction Algorithm).

    IMAPA uses temporal aggregation at multiple levels to reduce noise,
    then reconciles forecasts.

    Best for: Very sparse LUMPY demand with high noise
    """
    import time

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()

    # Determine sensible aggregation levels based on series length
    if aggregation_levels is None:
        n = len(y_train)
        if n <= 18:  # monthly-scale data
            aggregation_levels = [1, 2, 3]
        else:
            aggregation_levels = [1, 4, 13]  # weekly: weekly, monthly, quarterly

    level_forecasts = []
    level_weights = {}

    for level in aggregation_levels:
        n_complete = (len(y_train) // level) * level
        if n_complete < level:
            continue

        y_trimmed = y_train[:n_complete]
        y_agg = y_trimmed.reshape(-1, level).sum(axis=1)

        # Fit Croston/SBA at this level
        if base_method == 'sba':
            result = train_sba(y_agg, y_agg[:1])
        else:
            result = train_croston(y_agg, y_agg[:1])

        agg_forecast = result.model['forecast']
        base_forecast = agg_forecast / level  # Disaggregate

        # Weight by non-zero ratio
        non_zero_ratio = np.mean(y_agg > 0)
        level_weights[level] = non_zero_ratio
        level_forecasts.append((base_forecast, non_zero_ratio))

    # Normalize weights and compute final forecast
    total_weight = sum(w for _, w in level_forecasts) if level_forecasts else 1
    if total_weight > 0:
        final_forecast = sum(f * w / total_weight for f, w in level_forecasts)
    else:
        final_forecast = 0

    # Ensure scalar (sum of arrays can produce array)
    final_forecast = float(np.maximum(0, final_forecast)) if isinstance(final_forecast, np.ndarray) else max(0, float(final_forecast))

    model = {
        'type': 'imapa',
        'aggregation_levels': aggregation_levels,
        'base_method': base_method,
        'level_weights': level_weights,
        'forecast': final_forecast,
    }

    train_pred = np.full(len(y_train), final_forecast)
    val_pred = np.full(len(y_val), final_forecast)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='imapa',
        hyperparameters={'aggregation_levels': aggregation_levels, 'base_method': base_method},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


# =============================================================================
# COMPOUND / HURDLE MODEL TRAINING FUNCTIONS
# =============================================================================

def train_zero_inflated(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classifier_type: str = 'lightgbm',
    regressor_type: str = 'lightgbm',
    zero_threshold: float = 'auto',
    **kwargs
) -> TrainingResult:
    """
    Train two-stage zero-inflated model.

    Stage 1: Classification - P(demand > 0)
    Stage 2: Regression - E[demand | demand > 0]

    Prediction uses probability thresholding to reduce over-forecasting of zeros:
    - If P(demand > 0) < threshold → predict 0
    - Else → predict E[demand | demand > 0]

    Parameters
    ----------
    zero_threshold : float or 'auto'
        Probability threshold for predicting zero. If 'auto', computed from zero_fraction.
        Higher threshold = more conservative (predict more zeros).

    Best for: INTERMITTENT/LUMPY demand with many zeros
    """
    import time

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)

    # Compute zero_fraction from training data
    zero_fraction = float(np.sum(y_train == 0) / len(y_train)) if len(y_train) > 0 else 0.0
    logger.info(f"Zero-Inflated Model: training zero_fraction = {zero_fraction:.2%}")

    # Stage 1: Binary classifier
    y_binary = (y_train > 0).astype(int)

    if classifier_type == 'lightgbm':
        import lightgbm as lgb
        classifier = lgb.LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1, random_state=42)
    else:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(max_iter=1000, random_state=42)

    classifier.fit(X_train, y_binary)

    # Stage 2: Regressor on positive cases
    positive_mask = y_train > 0
    mean_positive = np.mean(y_train[positive_mask]) if np.sum(positive_mask) > 0 else 0

    regressor = None
    if np.sum(positive_mask) > 10:
        X_pos = X_train[positive_mask]
        y_pos = y_train[positive_mask]

        if regressor_type == 'lightgbm':
            import lightgbm as lgb
            regressor = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1, random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

        regressor.fit(X_pos, y_pos)

    # Create wrapper model with predict() method
    # Pass zero_fraction for automatic threshold computation
    model = ZeroInflatedModel(
        classifier=classifier,
        regressor=regressor,
        mean_positive=mean_positive,
        classifier_type=classifier_type,
        regressor_type=regressor_type,
        zero_fraction=zero_fraction,
        zero_threshold=zero_threshold,
    )

    logger.info(f"Zero-Inflated Model: using zero_threshold = {model.zero_threshold}")

    # Compute metrics using the model's predict method
    train_pred = np.clip(model.predict(X_train), 0, None)
    val_pred = np.clip(model.predict(X_val), 0, None)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='zero_inflated',
        hyperparameters={
            'classifier_type': classifier_type,
            'regressor_type': regressor_type,
            'zero_fraction': zero_fraction,
            'zero_threshold': model.zero_threshold,
        },
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def train_hurdle_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classifier_type: str = 'lightgbm',
    regressor_type: str = 'lightgbm',
    zero_threshold: float = 'auto',
    **kwargs
) -> TrainingResult:
    """
    Train Hurdle model - similar to zero-inflated with balanced class weights.

    The "hurdle" is whether any demand occurs at all.
    Uses balanced class weights for better zero/non-zero separation.

    Prediction uses probability thresholding to reduce over-forecasting of zeros:
    - If P(demand > 0) < threshold → predict 0 (don't cross the hurdle)
    - Else → predict E[demand | demand > 0]

    Parameters
    ----------
    zero_threshold : float or 'auto'
        Probability threshold for predicting zero. If 'auto', computed from zero_fraction.
        Hurdle models use slightly higher thresholds than ZeroInflated.

    Best for: LUMPY demand where zeros are structurally different
    """
    import time

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)

    # Compute zero_fraction from training data
    zero_fraction = float(np.sum(y_train == 0) / len(y_train)) if len(y_train) > 0 else 0.0
    logger.info(f"Hurdle Model: training zero_fraction = {zero_fraction:.2%}")

    # Stage 1: Binary classifier with balanced weights
    y_binary = (y_train > 0).astype(int)

    if classifier_type == 'lightgbm':
        import lightgbm as lgb
        classifier = lgb.LGBMClassifier(
            n_estimators=150, max_depth=6,
            class_weight='balanced',
            verbose=-1, random_state=42
        )
    else:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

    classifier.fit(X_train, y_binary)

    # Stage 2: Regressor on positive cases
    positive_mask = y_train > 0
    mean_positive = np.mean(y_train[positive_mask]) if np.sum(positive_mask) > 0 else 0

    regressor = None
    if np.sum(positive_mask) > 10:
        X_pos = X_train[positive_mask]
        y_pos = y_train[positive_mask]

        if regressor_type == 'lightgbm':
            import lightgbm as lgb
            regressor = lgb.LGBMRegressor(n_estimators=150, max_depth=6, verbose=-1, random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

        regressor.fit(X_pos, y_pos)

    # Create wrapper model with predict() method
    # Pass zero_fraction for automatic threshold computation
    model = HurdleModel(
        classifier=classifier,
        regressor=regressor,
        mean_positive=mean_positive,
        classifier_type=classifier_type,
        regressor_type=regressor_type,
        zero_fraction=zero_fraction,
        zero_threshold=zero_threshold,
    )

    logger.info(f"Hurdle Model: using zero_threshold = {model.zero_threshold}")

    # Compute metrics using the model's predict method
    train_pred = np.clip(model.predict(X_train), 0, None)
    val_pred = np.clip(model.predict(X_val), 0, None)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='hurdle_model',
        hyperparameters={
            'classifier_type': classifier_type,
            'regressor_type': regressor_type,
            'zero_fraction': zero_fraction,
            'zero_threshold': model.zero_threshold,
        },
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def get_optimal_tweedie_power(zero_fraction: float) -> float:
    """
    Compute optimal Tweedie variance power based on zero_fraction.

    Tweedie power parameter controls the distribution shape:
    - Power = 1: Poisson-like (more zero-mass, discrete-ish)
    - Power = 2: Gamma-like (no zeros, continuous positive)
    - Between 1-2: Mix of point mass at zero + continuous positive

    Higher zero_fraction → power closer to 1 (more zero-mass)
    Lower zero_fraction → power closer to 2 (less zero-mass)

    Parameters
    ----------
    zero_fraction : float
        Fraction of zeros in the target variable (0 to 1)

    Returns
    -------
    float
        Optimal Tweedie variance power between 1.1 and 1.9
    """
    if zero_fraction >= 0.7:
        return 1.1  # Very close to Poisson - lots of zeros
    elif zero_fraction >= 0.5:
        return 1.3  # High zero-mass
    elif zero_fraction >= 0.3:
        return 1.5  # Balanced (default)
    elif zero_fraction >= 0.15:
        return 1.7  # Lower zero-mass
    else:
        return 1.9  # Close to Gamma - very few zeros


def train_tweedie(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tweedie_variance_power: float = 'auto',
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> TrainingResult:
    """
    Train Tweedie regression using LightGBM.

    Tweedie distribution naturally handles zero-inflated continuous data.
    Power parameter between 1 and 2 for mix of zeros and positive values.

    Parameters
    ----------
    tweedie_variance_power : float or 'auto'
        If 'auto', dynamically computed from zero_fraction:
        - Higher zero_fraction → power closer to 1 (more zero-mass)
        - Lower zero_fraction → power closer to 2 (less zero-mass)

    Best for: INTERMITTENT/LUMPY demand (handles zeros natively)
    """
    import lightgbm as lgb
    import time

    start_time = time.time()

    y_train = np.clip(np.asarray(y_train).flatten(), 0, None)
    y_val = np.clip(np.asarray(y_val).flatten(), 0, None)

    # Compute zero_fraction from training data
    zero_fraction = float(np.sum(y_train == 0) / len(y_train)) if len(y_train) > 0 else 0.0

    # Dynamic power tuning based on zero_fraction
    if tweedie_variance_power == 'auto':
        power = get_optimal_tweedie_power(zero_fraction)
        logger.info(f"Tweedie: zero_fraction = {zero_fraction:.2%} → using power = {power}")
    else:
        power = tweedie_variance_power
        logger.info(f"Tweedie: using user-specified power = {power}")

    default_params = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'tweedie',
        'tweedie_variance_power': power,
        'verbosity': -1,
        'random_state': 42,
    }

    final_params = {**default_params, **(params or {}), **kwargs}
    # Ensure power is consistent even if params override
    final_params['tweedie_variance_power'] = power
    # CRITICAL: Force tweedie objective - other model params may contain incompatible objectives
    final_params['objective'] = 'tweedie'
    # Remove XGBoost-specific params that are incompatible with LightGBM
    xgb_only_params = ['tree_method', 'grow_policy', 'eval_metric', 'gamma', 'subsample',
                       'colsample_bytree', 'colsample_bylevel', 'reg_alpha', 'reg_lambda',
                       'scale_pos_weight', 'base_score', 'booster', 'missing']
    for p in xgb_only_params:
        final_params.pop(p, None)

    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    train_pred = np.clip(model.predict(X_train), 0, None)
    val_pred = np.clip(model.predict(X_val), 0, None)

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    # Add zero_fraction to hyperparameters for tracking
    final_params['zero_fraction'] = zero_fraction

    return TrainingResult(
        model=model,
        model_type='tweedie',
        hyperparameters=final_params,
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


# =============================================================================
# DISCRETE/ORDINAL DEMAND MODEL TRAINING FUNCTIONS
# =============================================================================
# These models are designed for low-cardinality targets where standard
# regression may not capture the discrete structure optimally.

def train_ordinal_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    snap_to_valid: bool = True,
    **kwargs
) -> TrainingResult:
    """
    Train ordinal regression for discrete demand with natural ordering.

    This model treats demand as ordinal categories (0 < 200 < 400 < 800)
    rather than continuous values. It predicts class probabilities and
    uses the ordinal structure for better predictions.

    Uses LightGBM with multi-class classification, then converts
    predicted class probabilities to expected demand values.

    Best for: DISCRETE demand with 5-15 unique values

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and target
    X_val, y_val : array-like
        Validation features and target
    params : dict, optional
        Additional LightGBM parameters
    snap_to_valid : bool
        If True, final predictions are snapped to valid discrete values

    Returns
    -------
    TrainingResult
        Model and metrics
    """
    import lightgbm as lgb
    import time
    from utils.model_selection_intelligence import snap_to_discrete_values

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()

    # Get unique values (classes) from training data
    unique_values = np.sort(np.unique(y_train))
    n_classes = len(unique_values)

    if n_classes < 2:
        # Fall back to regression if only one class
        logger.warning("Only one unique value - falling back to mean prediction")
        mean_val = np.mean(y_train)
        return TrainingResult(
            model={'type': 'constant', 'value': mean_val, 'unique_values': unique_values},
            model_type='ordinal_regression',
            hyperparameters={'n_classes': 1, 'fallback': 'constant'},
            train_wape=0.0,
            val_wape=compute_all_metrics(y_val, np.full_like(y_val, mean_val))['wape'],
            training_time_seconds=time.time() - start_time,
            is_feature_based=True,
        )

    if n_classes > 50:
        logger.warning(f"Too many classes ({n_classes}) for ordinal regression, consider using continuous regression")

    # Create class mapping
    value_to_class = {v: i for i, v in enumerate(unique_values)}
    class_to_value = {i: v for i, v in enumerate(unique_values)}

    # Convert targets to class labels
    y_train_class = np.array([value_to_class.get(v, 0) for v in y_train])
    y_val_class = np.array([value_to_class.get(v, np.argmin(np.abs(unique_values - v))) for v in y_val])

    # Handle class imbalance with class weights
    class_counts = np.bincount(y_train_class, minlength=n_classes)
    class_weights = np.where(class_counts > 0, len(y_train) / (n_classes * class_counts), 1.0)

    default_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'multiclass',
        'num_class': n_classes,
        'verbosity': -1,
        'random_state': 42,
        'class_weight': 'balanced',
    }

    final_params = {**default_params, **(params or {}), **kwargs}
    # Ensure multi-class settings
    final_params['objective'] = 'multiclass'
    final_params['num_class'] = n_classes

    # Strip regression metrics that conflict with multiclass objective
    _regression_only_metrics = {'mae', 'mse', 'rmse', 'mape', 'huber', 'fair', 'poisson', 'tweedie', 'gamma'}
    _m = final_params.get('metric')
    if isinstance(_m, str) and _m in _regression_only_metrics:
        final_params.pop('metric')
    elif isinstance(_m, list):
        final_params['metric'] = [m for m in _m if m not in _regression_only_metrics]
        if not final_params['metric']:
            final_params.pop('metric')

    # Strip non-classifier params that may leak from retrain hp_clean
    for _p in ('n_classes', 'unique_values', 'train_accuracy', 'val_accuracy',
               'n_unique_values', 'base_unit', 'cardinality_category',
               'val_wape_before_snap', 'val_wape_after_snap',
               'scale_pos_weight'):
        final_params.pop(_p, None)

    model = lgb.LGBMClassifier(**{k: v for k, v in final_params.items() if k != 'num_class'})
    model.set_params(num_class=n_classes)

    # Use sample weights for imbalance
    sample_weights = np.array([class_weights[c] for c in y_train_class])

    model.fit(
        X_train, y_train_class,
        eval_set=[(X_val, y_val_class)],
        sample_weight=sample_weights,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Get class probabilities
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)

    # Convert probabilities to expected value predictions
    # This uses the ordinal structure by computing probability-weighted expected value
    def probs_to_expected_value(probs, class_values):
        """Convert class probabilities to expected demand value."""
        return np.sum(probs * class_values, axis=1)

    train_pred = probs_to_expected_value(train_probs, unique_values)
    val_pred = probs_to_expected_value(val_probs, unique_values)

    # Optionally snap to valid values
    if snap_to_valid:
        train_pred = snap_to_discrete_values(train_pred, unique_values, method='nearest')
        val_pred = snap_to_discrete_values(val_pred, unique_values, method='nearest')

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    # Store metadata for inference
    model_container = {
        'classifier': model,
        'unique_values': unique_values,
        'value_to_class': value_to_class,
        'class_to_value': class_to_value,
        'snap_to_valid': snap_to_valid,
        'n_classes': n_classes,
    }

    return TrainingResult(
        model=model_container,
        model_type='ordinal_regression',
        hyperparameters={
            **final_params,
            'n_classes': n_classes,
            'unique_values': unique_values.tolist(),
            'snap_to_valid': snap_to_valid,
        },
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def train_discrete_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    use_class_weights: bool = True,
    **kwargs
) -> TrainingResult:
    """
    Train a pure classification model for very low cardinality targets.

    This treats each demand level as a separate class and predicts
    the most likely class. Best for targets with ≤5 unique values.

    Best for: VERY LOW cardinality discrete demand (2-5 unique values)

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and target
    X_val, y_val : array-like
        Validation features and target
    params : dict, optional
        Additional LightGBM parameters
    use_class_weights : bool
        If True, use balanced class weights for imbalanced classes

    Returns
    -------
    TrainingResult
        Model and metrics
    """
    import lightgbm as lgb
    import time

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()

    # Get unique values (classes) from training data
    unique_values = np.sort(np.unique(y_train))
    n_classes = len(unique_values)

    logger.info(f"Discrete Classifier: {n_classes} classes detected: {unique_values[:10]}...")

    if n_classes < 2:
        mean_val = np.mean(y_train)
        return TrainingResult(
            model={'type': 'constant', 'value': mean_val},
            model_type='discrete_classifier',
            hyperparameters={'n_classes': 1, 'fallback': 'constant'},
            train_wape=0.0,
            val_wape=compute_all_metrics(y_val, np.full_like(y_val, mean_val))['wape'],
            training_time_seconds=time.time() - start_time,
            is_feature_based=True,
        )

    # Create class mapping
    value_to_class = {v: i for i, v in enumerate(unique_values)}
    class_to_value = {i: v for i, v in enumerate(unique_values)}

    # Convert targets to class labels
    y_train_class = np.array([value_to_class.get(v, 0) for v in y_train])
    y_val_class = np.array([value_to_class.get(v, np.argmin(np.abs(unique_values - v))) for v in y_val])

    default_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'multiclass',
        'verbosity': -1,
        'random_state': 42,
    }

    if use_class_weights:
        default_params['class_weight'] = 'balanced'

    final_params = {**default_params, **(params or {}), **kwargs}
    final_params['objective'] = 'multiclass'

    # Strip regression metrics that conflict with multiclass objective
    # (may leak from hp_clean sanitization that reset objective to 'regression')
    _regression_only_metrics = {'mae', 'mse', 'rmse', 'mape', 'huber', 'fair', 'poisson', 'tweedie', 'gamma'}
    _m = final_params.get('metric')
    if isinstance(_m, str) and _m in _regression_only_metrics:
        final_params.pop('metric')  # Let LGBMClassifier use its default (multi_logloss)
    elif isinstance(_m, list):
        final_params['metric'] = [m for m in _m if m not in _regression_only_metrics]
        if not final_params['metric']:
            final_params.pop('metric')

    # Strip non-classifier params that may leak from retrain hp_clean
    for _p in ('n_classes', 'unique_values', 'train_accuracy', 'val_accuracy',
               'n_unique_values', 'base_unit', 'cardinality_category',
               'val_wape_before_snap', 'val_wape_after_snap',
               'scale_pos_weight', 'num_class'):
        final_params.pop(_p, None)

    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_train, y_train_class,
        eval_set=[(X_val, y_val_class)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Predict most likely class
    train_pred_class = model.predict(X_train)
    val_pred_class = model.predict(X_val)

    # Convert class predictions back to values
    train_pred = np.array([class_to_value[c] for c in train_pred_class])
    val_pred = np.array([class_to_value[c] for c in val_pred_class])

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    # Compute classification accuracy as additional metric
    train_accuracy = np.mean(train_pred_class == y_train_class)
    val_accuracy = np.mean(val_pred_class == y_val_class)

    model_container = {
        'classifier': model,
        'unique_values': unique_values,
        'value_to_class': value_to_class,
        'class_to_value': class_to_value,
        'n_classes': n_classes,
    }

    return TrainingResult(
        model=model_container,
        model_type='discrete_classifier',
        hyperparameters={
            **final_params,
            'n_classes': n_classes,
            'unique_values': unique_values.tolist(),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
        },
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def train_hybrid_discrete(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    base_unit: Optional[float] = None,
    **kwargs
) -> TrainingResult:
    """
    Train a hybrid model for discrete demand with base unit structure.

    This model combines:
    1. Continuous regression for magnitude prediction
    2. Post-processing to snap to valid discrete values

    Optimal for discrete demand that follows case-pack ordering
    (e.g., 0, 200, 400, 600, 800 where base_unit=200).

    Best for: MEDIUM cardinality discrete demand (10-20 unique values)

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and target
    X_val, y_val : array-like
        Validation features and target
    params : dict, optional
        Additional model parameters
    base_unit : float, optional
        If known, the base unit for discretization (e.g., 200 for case packs)

    Returns
    -------
    TrainingResult
        Model and metrics
    """
    import lightgbm as lgb
    import time
    from utils.model_selection_intelligence import (
        analyze_discrete_demand,
        snap_to_discrete_values,
    )

    start_time = time.time()

    y_train = np.asarray(y_train).flatten()
    y_val = np.asarray(y_val).flatten()

    # Analyze discrete structure
    discrete_info = analyze_discrete_demand(y_train)
    unique_values = discrete_info.unique_values

    # Detect base unit if not provided
    if base_unit is None:
        base_unit = discrete_info.base_unit

    logger.info(
        f"Hybrid Discrete: {discrete_info.n_unique} values, "
        f"base_unit={base_unit}, category={discrete_info.cardinality_category}"
    )

    # Train continuous regression model
    default_params = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'regression',
        'verbosity': -1,
        'random_state': 42,
    }

    final_params = {**default_params, **(params or {}), **kwargs}

    model = lgb.LGBMRegressor(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Get continuous predictions
    train_pred_cont = model.predict(X_train)
    val_pred_cont = model.predict(X_val)

    # Snap to discrete values
    train_pred = snap_to_discrete_values(train_pred_cont, unique_values, method='nearest')
    val_pred = snap_to_discrete_values(val_pred_cont, unique_values, method='nearest')

    # Compute metrics on snapped predictions
    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    # Also compute metrics on continuous for comparison
    train_metrics_cont = compute_all_metrics(y_train, train_pred_cont)
    val_metrics_cont = compute_all_metrics(y_val, val_pred_cont)

    model_container = {
        'regressor': model,
        'unique_values': unique_values,
        'base_unit': base_unit,
        'discrete_info': {
            'n_unique': discrete_info.n_unique,
            'cardinality_category': discrete_info.cardinality_category,
            'is_multiple_based': discrete_info.is_multiple_based,
        },
    }

    return TrainingResult(
        model=model_container,
        model_type='hybrid_discrete',
        hyperparameters={
            **final_params,
            'n_unique_values': len(unique_values),
            'base_unit': base_unit,
            'cardinality_category': discrete_info.cardinality_category,
            'val_wape_before_snap': val_metrics_cont['wape'],
            'val_wape_after_snap': val_metrics['wape'],
        },
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def predict_with_discrete_model(
    model_container: Dict[str, Any],
    X: np.ndarray,
    model_type: str,
    return_probabilities: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate predictions from discrete demand models.

    This handles the different model types (ordinal, classifier, hybrid)
    and their specific prediction logic.

    Parameters
    ----------
    model_container : dict
        Model container from training (contains model + metadata)
    X : np.ndarray
        Feature matrix for prediction
    model_type : str
        'ordinal_regression', 'discrete_classifier', or 'hybrid_discrete'
    return_probabilities : bool
        If True, also return class probabilities (for ordinal/classifier)

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        Predictions, and optionally class probabilities
    """
    from utils.model_selection_intelligence import snap_to_discrete_values

    if model_type == 'ordinal_regression':
        classifier = model_container['classifier']
        unique_values = model_container['unique_values']
        snap = model_container.get('snap_to_valid', True)

        # Get class probabilities
        probs = classifier.predict_proba(X)

        # Expected value from probabilities
        predictions = np.sum(probs * unique_values, axis=1)

        if snap:
            predictions = snap_to_discrete_values(predictions, unique_values, method='nearest')

        if return_probabilities:
            return predictions, probs
        return predictions

    elif model_type == 'discrete_classifier':
        classifier = model_container['classifier']
        class_to_value = model_container['class_to_value']

        # Predict class
        pred_class = classifier.predict(X)
        predictions = np.array([class_to_value[c] for c in pred_class])

        if return_probabilities:
            probs = classifier.predict_proba(X)
            return predictions, probs
        return predictions

    elif model_type == 'hybrid_discrete':
        regressor = model_container['regressor']
        unique_values = model_container['unique_values']

        # Continuous prediction + snap
        pred_cont = regressor.predict(X)
        predictions = snap_to_discrete_values(pred_cont, unique_values, method='nearest')

        if return_probabilities:
            # No probabilities for hybrid, return uniform
            n_classes = len(unique_values)
            probs = np.ones((len(X), n_classes)) / n_classes
            return predictions, probs
        return predictions

    else:
        raise ValueError(f"Unknown discrete model type: {model_type}")


# =============================================================================
# CLASSICAL STATISTICAL MODEL TRAINING FUNCTIONS
# =============================================================================

def train_arima(
    y_train: np.ndarray,
    y_val: np.ndarray,
    seasonal: bool = False,
    m: int = 1,
    **kwargs
) -> TrainingResult:
    """
    Train ARIMA model using auto_arima for order selection.

    Best for: SMOOTH demand with trend
    """
    import time

    start_time = time.time()

    y_train = np.clip(np.asarray(y_train).flatten(), 0, None)
    y_val = np.asarray(y_val).flatten()

    try:
        from pmdarima import auto_arima

        model = auto_arima(
            y_train,
            seasonal=seasonal,
            m=m if seasonal else 1,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_q=3, max_d=2,
            max_P=2, max_Q=2, max_D=1,
            stepwise=True,
            n_fits=10,
        )

        train_pred = np.clip(model.predict_in_sample(), 0, None)
        val_pred = np.clip(model.predict(n_periods=len(y_val)), 0, None)
        use_fallback = False

    except Exception as e:
        logger.warning(f"ARIMA failed: {e}. Using mean fallback.")
        model = {'type': 'arima_fallback', 'mean': np.mean(y_train)}
        train_pred = np.full(len(y_train), np.mean(y_train))
        val_pred = np.full(len(y_val), np.mean(y_train))
        use_fallback = True

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='arima',
        hyperparameters={'seasonal': seasonal, 'm': m, 'fallback': use_fallback},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


def train_sarima(
    y_train: np.ndarray,
    y_val: np.ndarray,
    m: int = None,
    time_format: str = 'year_week',
    **kwargs
) -> TrainingResult:
    """Train SARIMA (seasonal ARIMA) model."""
    if m is None:
        m = 12 if time_format == 'year_month' else 52
    return train_arima(y_train, y_val, seasonal=True, m=m, **kwargs)


def train_ets(
    y_train: np.ndarray,
    y_val: np.ndarray,
    trend: str = 'add',
    seasonal: str = None,
    seasonal_periods: int = None,
    time_format: str = 'year_week',
    **kwargs
) -> TrainingResult:
    """
    Train ETS (Exponential Smoothing) model.

    Best for: SMOOTH demand with clear patterns
    """
    if seasonal_periods is None:
        seasonal_periods = 12 if time_format == 'year_month' else 52
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import time

    start_time = time.time()

    y_train = np.clip(np.asarray(y_train).flatten(), 0.01, None)
    y_val = np.asarray(y_val).flatten()

    try:
        if seasonal and len(y_train) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                y_train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method='estimated',
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                y_train,
                trend=trend,
                seasonal=None,
                initialization_method='estimated',
            ).fit(optimized=True)

        train_pred = np.clip(model.fittedvalues, 0, None)
        val_pred = np.clip(model.forecast(len(y_val)), 0, None)
        use_fallback = False

    except Exception as e:
        logger.warning(f"ETS failed: {e}. Using mean fallback.")
        model = {'type': 'ets_fallback', 'mean': np.mean(y_train)}
        train_pred = np.full(len(y_train), np.mean(y_train))
        val_pred = np.full(len(y_val), np.mean(y_train))
        use_fallback = True

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='ets',
        hyperparameters={'trend': trend, 'seasonal': seasonal, 'seasonal_periods': seasonal_periods, 'fallback': use_fallback},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


def train_theta(
    y_train: np.ndarray,
    y_val: np.ndarray,
    **kwargs
) -> TrainingResult:
    """
    Train Theta method - simple but effective.

    Best for: ERRATIC demand with trend
    """
    from statsmodels.tsa.forecasting.theta import ThetaModel
    import time

    start_time = time.time()

    y_train = np.clip(np.asarray(y_train).flatten(), 0, None)
    y_val = np.asarray(y_val).flatten()

    try:
        model = ThetaModel(y_train).fit()
        train_pred = np.clip(model.fittedvalues, 0, None)
        val_pred = np.clip(model.forecast(len(y_val)), 0, None)
        use_fallback = False

    except Exception as e:
        logger.warning(f"Theta failed: {e}. Using mean fallback.")
        model = {'type': 'theta_fallback', 'mean': np.mean(y_train)}
        train_pred = np.full(len(y_train), np.mean(y_train))
        val_pred = np.full(len(y_val), np.mean(y_train))
        use_fallback = True

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='theta',
        hyperparameters={'fallback': use_fallback},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


def train_tbats(
    y_train: np.ndarray,
    y_val: np.ndarray,
    seasonal_periods: List[int] = None,
    **kwargs
) -> TrainingResult:
    """
    Train TBATS model for complex seasonality.

    Best for: SMOOTH demand with multiple seasonalities
    """
    import time

    start_time = time.time()

    y_train = np.clip(np.asarray(y_train).flatten(), 0, None)
    y_val = np.asarray(y_val).flatten()
    _default_period = 12 if kwargs.get('time_format', 'year_week') == 'year_month' else 52
    seasonal_periods = seasonal_periods or [_default_period]

    try:
        from tbats import TBATS

        estimator = TBATS(seasonal_periods=seasonal_periods)
        model = estimator.fit(y_train)

        train_pred = np.clip(model.y_hat, 0, None)
        val_pred = np.clip(model.forecast(steps=len(y_val)), 0, None)
        use_fallback = False

    except Exception as e:
        logger.warning(f"TBATS failed: {e}. Using ETS fallback.")
        result = train_ets(y_train, y_val)
        return TrainingResult(
            model=result.model,
            model_type='tbats',
            hyperparameters={'seasonal_periods': seasonal_periods, 'fallback': True},
            train_wape=result.train_wape,
            val_wape=result.val_wape,
            train_mae=result.train_mae,
            val_mae=result.val_mae,
            train_rmse=result.train_rmse,
            val_rmse=result.val_rmse,
            training_time_seconds=time.time() - start_time,
            is_feature_based=False,
        )

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='tbats',
        hyperparameters={'seasonal_periods': seasonal_periods, 'fallback': use_fallback},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


# =============================================================================
# BAYESIAN / PROBABILISTIC MODEL TRAINING FUNCTIONS
# =============================================================================

def train_prophet(
    y_train: np.ndarray,
    y_val: np.ndarray,
    dates_train: Optional[np.ndarray] = None,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    **kwargs
) -> TrainingResult:
    """
    Train Facebook Prophet model.

    Best for: SMOOTH demand with multiple seasonalities
    """
    import time

    start_time = time.time()

    y_train = np.clip(np.asarray(y_train).flatten(), 0, None)
    y_val = np.asarray(y_val).flatten()

    try:
        from prophet import Prophet

        # Create dates if not provided
        if dates_train is None:
            dates_train = pd.date_range(end='2025-01-01', periods=len(y_train), freq='W')

        df = pd.DataFrame({
            'ds': pd.to_datetime(dates_train),
            'y': y_train
        })

        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
        )
        model.fit(df)

        # In-sample predictions
        train_forecast = model.predict(df)
        train_pred = np.clip(train_forecast['yhat'].values, 0, None)

        # Future predictions
        future = model.make_future_dataframe(periods=len(y_val), freq='W')
        val_forecast = model.predict(future)
        val_pred = np.clip(val_forecast['yhat'].values[-len(y_val):], 0, None)
        use_fallback = False

    except Exception as e:
        logger.warning(f"Prophet failed: {e}. Using ETS fallback.")
        result = train_ets(y_train, y_val)
        return TrainingResult(
            model=result.model,
            model_type='prophet',
            hyperparameters={'yearly_seasonality': yearly_seasonality, 'weekly_seasonality': weekly_seasonality, 'fallback': True},
            train_wape=result.train_wape,
            val_wape=result.val_wape,
            train_mae=result.train_mae,
            val_mae=result.val_mae,
            train_rmse=result.train_rmse,
            val_rmse=result.val_rmse,
            training_time_seconds=time.time() - start_time,
            is_feature_based=False,
        )

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='prophet',
        hyperparameters={'yearly_seasonality': yearly_seasonality, 'weekly_seasonality': weekly_seasonality, 'fallback': use_fallback},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


def train_bsts(
    y_train: np.ndarray,
    y_val: np.ndarray,
    dates_train: Optional[np.ndarray] = None,
    seasonality: int = None,
    time_format: str = 'year_week',
    **kwargs
) -> TrainingResult:
    """
    Train BSTS (Bayesian Structural Time Series) using orbit-ml.

    Best for: ERRATIC demand with uncertainty quantification
    """
    import time

    start_time = time.time()

    if seasonality is None:
        seasonality = 12 if time_format == 'year_month' else 52

    y_train = np.clip(np.asarray(y_train).flatten(), 0.01, None)
    y_val = np.asarray(y_val).flatten()

    try:
        from orbit.models import DLT

        if dates_train is None:
            dates_train = pd.date_range(end='2025-01-01', periods=len(y_train), freq='W')

        df = pd.DataFrame({
            'ds': pd.to_datetime(dates_train),
            'y': y_train
        })

        model = DLT(
            response_col='y',
            date_col='ds',
            seasonality=seasonality,
        )
        model.fit(df)

        # In-sample predictions
        train_pred_df = model.predict(df)
        train_pred = np.clip(train_pred_df['prediction'].values, 0, None)

        # Future predictions
        last_date = df['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=len(y_val), freq='W')
        future_df = pd.DataFrame({'ds': future_dates})
        val_pred_df = model.predict(future_df)
        val_pred = np.clip(val_pred_df['prediction'].values, 0, None)
        use_fallback = False

    except Exception as e:
        logger.warning(f"BSTS/DLT failed: {e}. Using ETS fallback.")
        result = train_ets(y_train, y_val)
        return TrainingResult(
            model=result.model,
            model_type='bsts',
            hyperparameters={'seasonality': seasonality, 'fallback': True},
            train_wape=result.train_wape,
            val_wape=result.val_wape,
            train_mae=result.train_mae,
            val_mae=result.val_mae,
            train_rmse=result.train_rmse,
            val_rmse=result.val_rmse,
            training_time_seconds=time.time() - start_time,
            is_feature_based=False,
        )

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='bsts',
        hyperparameters={'seasonality': seasonality, 'fallback': use_fallback},
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=False,
    )


# =============================================================================
# ENSEMBLE MODEL TRAINING FUNCTIONS
# =============================================================================

def train_weighted_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_types: List[str] = None,
    weights: Optional[List[float]] = None,
    member_params: Optional[Dict[str, Dict[str, Any]]] = None,
    optimization_method: str = 'cross_validated',
    **kwargs
) -> TrainingResult:
    """
    Train weighted ensemble of multiple models with OPTIMAL WEIGHT FINDING.

    Uses sophisticated ensemble optimization from forecast_optimization.py:
    - 'cross_validated': Cross-validated optimal weights (prevents overfitting)
    - 'constrained': Constrained optimization minimizing WAPE
    - 'inverse_mse': Simple inverse MSE weighting (fast fallback)
    - 'equal': Equal weights (baseline)

    If weights are explicitly provided, uses those instead of optimization.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    model_types : List of model types to include in ensemble
    weights : Optional explicit weights (bypasses optimization)
    member_params : Optional dict mapping model_type -> hyperparameters
                    e.g., {'lightgbm': {'max_depth': 8}, 'tweedie': {...}}
    optimization_method : How to find optimal weights

    Returns
    -------
    TrainingResult with ensemble model and optimal weights
    """
    import time
    from utils.forecast_optimization import (
        find_optimal_ensemble_weights,
        cross_validated_ensemble_weights,
    )

    start_time = time.time()

    model_types = model_types or ['lightgbm', 'xgboost', 'catboost']
    member_params = member_params or {}

    # Train each component model with its specific hyperparameters
    component_results = []
    for model_type in model_types:
        try:
            # Get hyperparameters for this specific model type
            model_hp = member_params.get(model_type, {})
            result = train_model_by_name(model_type, X_train, y_train, X_val, y_val, params=model_hp)
            component_results.append((model_type, result))
        except Exception as e:
            logger.warning(f"Failed to train {model_type} for ensemble: {e}")

    if not component_results:
        logger.warning("No models trained successfully for ensemble. Using LightGBM only.")
        result = train_lightgbm(X_train, y_train, X_val, y_val)
        return result

    # Get validation predictions for weight optimization
    model_val_predictions = {}
    model_train_predictions = {}
    for model_type, result in component_results:
        m = result.model
        if isinstance(m, dict):
            # Dict-based models: discrete_classifier, hybrid_discrete, constant, univariate
            if 'classifier' in m and 'class_to_value' in m:
                # discrete_classifier dict
                pred_val = np.array([m['class_to_value'].get(int(c), 0) for c in m['classifier'].predict(X_val)], dtype=float)
                pred_train = np.array([m['class_to_value'].get(int(c), 0) for c in m['classifier'].predict(X_train)], dtype=float)
            elif 'regressor' in m and 'unique_values' in m:
                # hybrid_discrete dict
                from utils.model_selection_intelligence import snap_to_discrete_values
                pred_val = snap_to_discrete_values(m['regressor'].predict(X_val), m['unique_values'], method='nearest').astype(float)
                pred_train = snap_to_discrete_values(m['regressor'].predict(X_train), m['unique_values'], method='nearest').astype(float)
            elif m.get('type') == 'weighted_ensemble' and 'component_models' in m:
                # Nested ensemble — compute weighted sum of components
                pred_val = np.zeros(len(y_val))
                pred_train = np.zeros(len(y_train))
                for (_, comp_model), w in zip(m['component_models'], m.get('weights', [])):
                    if hasattr(comp_model, 'predict'):
                        pred_val += w * comp_model.predict(X_val)
                        pred_train += w * comp_model.predict(X_train)
                    elif isinstance(comp_model, dict):
                        fv = float(comp_model.get('forecast', comp_model.get('value', np.mean(y_train))))
                        pred_val += w * fv
                        pred_train += w * fv
            else:
                # Generic dict fallback (constant, univariate, etc.)
                forecast = float(m.get('forecast', m.get('value', m.get('mean', np.mean(y_train)))))
                pred_val = np.full(len(y_val), forecast)
                pred_train = np.full(len(y_train), forecast)
            model_val_predictions[model_type] = pred_val
            model_train_predictions[model_type] = pred_train
        elif hasattr(m, 'predict'):
            model_val_predictions[model_type] = m.predict(X_val)
            model_train_predictions[model_type] = m.predict(X_train)
        else:
            forecast = float(np.mean(y_train))
            model_val_predictions[model_type] = np.full(len(y_val), forecast)
            model_train_predictions[model_type] = np.full(len(y_train), forecast)

    # Find optimal weights using sophisticated optimization
    if weights is None:
        try:
            if optimization_method == 'cross_validated':
                # Cross-validated weights prevent overfitting to validation set
                weights_result = cross_validated_ensemble_weights(
                    model_val_predictions, y_val, n_folds=5, metric='wape'
                )
                weights_dict = weights_result.weights
                cv_score = weights_result.cv_score
                effective_n = weights_result.effective_n_models
                logger.info(f"CV optimal weights found: {weights_dict}, CV WAPE: {cv_score:.4f}, effective models: {effective_n:.1f}")

            elif optimization_method == 'constrained':
                # Constrained optimization on validation set
                weights_result = find_optimal_ensemble_weights(
                    model_val_predictions, y_val,
                    method='constrained_optimization',
                    metric='wape',
                    min_weight=0.0,
                    max_models=len(component_results)
                )
                weights_dict = weights_result.weights
                logger.info(f"Constrained optimal weights: {weights_dict}")

            elif optimization_method == 'inverse_mse':
                # Simple inverse MSE (fast)
                weights_result = find_optimal_ensemble_weights(
                    model_val_predictions, y_val,
                    method='inverse_mse',
                    metric='wape'
                )
                weights_dict = weights_result.weights
                logger.info(f"Inverse MSE weights: {weights_dict}")

            else:  # 'equal'
                weights_dict = {m: 1.0/len(component_results) for m, _ in component_results}
                logger.info("Using equal weights")

            # Convert dict to list in component order
            weights = [weights_dict.get(m, 0.0) for m, _ in component_results]

        except Exception as e:
            logger.warning(f"Optimal weight finding failed: {e}. Using inverse WAPE fallback.")
            # Fallback to simple inverse WAPE
            wapes = [r.val_wape for _, r in component_results]
            inv_wapes = [1.0 / (w + 0.01) for w in wapes]
            total = sum(inv_wapes)
            weights = [w / total for w in inv_wapes]
            weights_dict = {m: w for (m, _), w in zip(component_results, weights)}

    else:
        # Weights were provided explicitly
        weights_dict = {m: w for (m, _), w in zip(component_results, weights)}

    # Combine predictions using optimal weights
    train_pred = np.zeros(len(y_train))
    val_pred = np.zeros(len(y_val))

    for (model_type, result), weight in zip(component_results, weights):
        train_pred += weight * model_train_predictions[model_type]
        val_pred += weight * model_val_predictions[model_type]

    train_pred = np.clip(train_pred, 0, None)
    val_pred = np.clip(val_pred, 0, None)

    model = {
        'type': 'weighted_ensemble',
        'component_models': [(m, r.model) for m, r in component_results],
        'weights': weights,
        'weights_dict': weights_dict,
        'model_types': [m for m, _ in component_results],
        'optimization_method': optimization_method,
    }

    train_metrics = compute_all_metrics(y_train, train_pred)
    val_metrics = compute_all_metrics(y_val, val_pred)

    return TrainingResult(
        model=model,
        model_type='weighted_ensemble',
        hyperparameters={
            'model_types': [m for m, _ in component_results],
            'weights': weights,
            'weights_dict': weights_dict,
            'optimization_method': optimization_method,
        },
        train_wape=train_metrics['wape'],
        val_wape=val_metrics['wape'],
        train_mae=train_metrics['mae'],
        val_mae=val_metrics['mae'],
        train_rmse=train_metrics['rmse'],
        val_rmse=val_metrics['rmse'],
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


def train_stacking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_model_types: List[str] = None,
    meta_model_type: str = 'lightgbm',
    **kwargs
) -> TrainingResult:
    """
    Train stacking ensemble with meta-learner.

    Base model predictions become features for the meta-model.
    """
    import time
    from sklearn.model_selection import cross_val_predict

    start_time = time.time()

    base_model_types = base_model_types or ['lightgbm', 'xgboost', 'random_forest']

    # Train base models and get OOF predictions
    base_models = []
    train_meta_features = []
    val_meta_features = []

    for model_type in base_model_types:
        try:
            result = train_model_by_name(model_type, X_train, y_train, X_val, y_val)
            base_models.append((model_type, result.model))

            # Get predictions for meta-features
            if result.is_feature_based:
                train_pred = result.model.predict(X_train)
                val_pred = result.model.predict(X_val)
            else:
                forecast = result.model.get('forecast', np.mean(y_train))
                train_pred = np.full(len(y_train), forecast)
                val_pred = np.full(len(y_val), forecast)

            train_meta_features.append(train_pred)
            val_meta_features.append(val_pred)

        except Exception as e:
            logger.warning(f"Failed to train base model {model_type}: {e}")

    if not base_models:
        logger.warning("No base models trained. Using LightGBM only.")
        return train_lightgbm(X_train, y_train, X_val, y_val)

    # Stack meta-features
    X_meta_train = np.column_stack([X_train] + [f.reshape(-1, 1) for f in train_meta_features])
    X_meta_val = np.column_stack([X_val] + [f.reshape(-1, 1) for f in val_meta_features])

    # Train meta-model
    if meta_model_type == 'lightgbm':
        meta_result = train_lightgbm(X_meta_train, y_train, X_meta_val, y_val,
                                      params={'n_estimators': 100, 'max_depth': 4})
    else:
        meta_result = train_xgboost(X_meta_train, y_train, X_meta_val, y_val,
                                     params={'n_estimators': 100, 'max_depth': 4})

    model = {
        'type': 'stacking',
        'base_models': base_models,
        'meta_model': meta_result.model,
        'meta_model_type': meta_model_type,
        'base_model_types': [m for m, _ in base_models],
    }

    return TrainingResult(
        model=model,
        model_type='stacking',
        hyperparameters={'base_model_types': [m for m, _ in base_models], 'meta_model_type': meta_model_type},
        train_wape=meta_result.train_wape,
        val_wape=meta_result.val_wape,
        train_mae=meta_result.train_mae,
        val_mae=meta_result.val_mae,
        train_rmse=meta_result.train_rmse,
        val_rmse=meta_result.val_rmse,
        training_time_seconds=time.time() - start_time,
        is_feature_based=True,
    )


# =============================================================================
# DEEP LEARNING MODEL TRAINING FUNCTIONS (Stubs with fallbacks)
# =============================================================================

def train_tft(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    **kwargs
) -> TrainingResult:
    """Train TFT (Temporal Fusion Transformer) - falls back to LightGBM."""
    logger.warning("TFT requires pytorch-forecasting. Using LightGBM fallback.")
    result = train_lightgbm(X_train, y_train, X_val, y_val)
    result.model_type = 'tft'
    result.hyperparameters['fallback'] = True
    return result


def train_lstm(
    y_train: np.ndarray,
    y_val: np.ndarray,
    **kwargs
) -> TrainingResult:
    """Train LSTM - falls back to ETS."""
    logger.warning("LSTM requires darts/pytorch. Using ETS fallback.")
    result = train_ets(y_train, y_val)
    result.model_type = 'lstm'
    result.hyperparameters['fallback'] = True
    return result


def train_nbeats(
    y_train: np.ndarray,
    y_val: np.ndarray,
    **kwargs
) -> TrainingResult:
    """Train N-BEATS - falls back to Theta."""
    logger.warning("N-BEATS requires darts/pytorch. Using Theta fallback.")
    result = train_theta(y_train, y_val)
    result.model_type = 'nbeats'
    result.hyperparameters['fallback'] = True
    return result


def train_deepar(
    y_train: np.ndarray,
    y_val: np.ndarray,
    **kwargs
) -> TrainingResult:
    """Train DeepAR - falls back to Prophet."""
    logger.warning("DeepAR requires gluonts/pytorch. Using Prophet fallback.")
    result = train_prophet(y_train, y_val)
    result.model_type = 'deepar'
    result.hyperparameters['fallback'] = True
    return result


def train_wavenet(
    y_train: np.ndarray,
    y_val: np.ndarray,
    **kwargs
) -> TrainingResult:
    """Train WaveNet - falls back to SARIMA."""
    logger.warning("WaveNet requires gluonts/pytorch. Using SARIMA fallback.")
    result = train_sarima(y_train, y_val)
    result.model_type = 'wavenet'
    result.hyperparameters['fallback'] = True
    return result


# =============================================================================
# MODEL TRAINING REGISTRY
# =============================================================================

# Mapping from model name to training function
TRAINING_REGISTRY: Dict[str, Callable] = {
    # Tree-based (feature-based)
    'lightgbm': train_lightgbm,
    'xgboost': train_xgboost,
    'catboost': train_catboost,
    'random_forest': train_random_forest,

    # Intermittent specialists (univariate)
    'croston': train_croston,
    'sba': train_sba,
    'tsb': train_tsb,
    'imapa': train_imapa,

    # Compound / hurdle (feature-based)
    'zero_inflated': train_zero_inflated,
    'hurdle_model': train_hurdle_model,
    'tweedie': train_tweedie,

    # Discrete demand specialists (feature-based)
    # For low-cardinality targets that shouldn't be treated as continuous
    'ordinal_regression': train_ordinal_regression,
    'discrete_classifier': train_discrete_classifier,
    'hybrid_discrete': train_hybrid_discrete,

    # Classical statistical (univariate)
    'arima': train_arima,
    'sarima': train_sarima,
    'ets': train_ets,
    'theta': train_theta,
    'tbats': train_tbats,

    # Bayesian / probabilistic (univariate)
    'prophet': train_prophet,
    'bsts': train_bsts,

    # Ensemble (feature-based)
    'weighted_ensemble': train_weighted_ensemble,
    'stacking': train_stacking,

    # Deep learning (stubs)
    'tft': train_tft,
    'lstm': train_lstm,
    'nbeats': train_nbeats,
    'deepar': train_deepar,
    'wavenet': train_wavenet,
}

# Add multi-horizon training functions if available
if MULTI_HORIZON_AVAILABLE:
    TRAINING_REGISTRY.update({
        # Multi-horizon direct forecasting (optimized for longer horizons like Lag 5)
        'multi_horizon_lightgbm': train_multi_horizon_lightgbm,
        'multi_horizon_xgboost': train_multi_horizon_xgboost,
        'multi_horizon_ensemble': train_multi_horizon_ensemble,
    })

# =========================================================================
# Phase 3/7/8: Register hierarchical, enhanced, and combination models.
#
# These models have different signatures (they need DataFrames with key/hierarchy
# columns, not just X/y arrays). We register adapter functions that bridge
# the standard registry interface to the actual training functions.
#
# The adapter pattern: the registry function receives (X_train, y_train, X_val,
# y_val, **kwargs) where kwargs may contain 'train_df', 'val_df', 'key_col',
# 'hierarchy_col', 'feature_cols'. If these are absent, the adapter creates
# a basic DataFrame from X/y to call the underlying function.
# =========================================================================

def _adapt_hierarchical_training(train_func, X_train, y_train, X_val, y_val, **kwargs):
    """Adapter: creates DataFrames from X/y if train_df not provided."""
    train_df = kwargs.pop('train_df', None)
    val_df = kwargs.pop('val_df', None)
    feature_cols = kwargs.pop('feature_cols', None)
    key_col = kwargs.pop('key_col', 'key')
    target_col = kwargs.pop('target_col', 'target')

    if train_df is None:
        # Build DataFrame from arrays
        if feature_cols is None:
            feature_cols = [f'f{i}' for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df[target_col] = y_train
        train_df[key_col] = 'key_0'  # Single-key fallback
        val_df = pd.DataFrame(X_val, columns=feature_cols)
        val_df[target_col] = y_val
        val_df[key_col] = 'key_0'

    wrapper, meta = train_func(
        train_df=train_df, val_df=val_df,
        feature_cols=feature_cols or [c for c in train_df.columns if c not in [key_col, target_col]],
        target_col=target_col, key_col=key_col, **kwargs,
    )
    return TrainingResult(
        model=wrapper, model_type=meta.get('model_type', 'hierarchical'),
        train_wape=meta.get('train_wape', meta.get('val_wape', 1.0)),
        val_wape=meta.get('val_wape', 1.0),
        hyperparameters=meta, training_time_seconds=meta.get('training_time', 0),
    )


try:
    from utils.hierarchical_model_training import (
        train_global_local_model, train_mixed_effects_model, train_multi_level_ensemble,
    )
    TRAINING_REGISTRY['global_local'] = lambda X_tr, y_tr, X_v, y_v, **kw: _adapt_hierarchical_training(train_global_local_model, X_tr, y_tr, X_v, y_v, **kw)
    TRAINING_REGISTRY['mixed_effects'] = lambda X_tr, y_tr, X_v, y_v, **kw: _adapt_hierarchical_training(train_mixed_effects_model, X_tr, y_tr, X_v, y_v, **kw)
    TRAINING_REGISTRY['multi_level_ensemble'] = lambda X_tr, y_tr, X_v, y_v, **kw: _adapt_hierarchical_training(train_multi_level_ensemble, X_tr, y_tr, X_v, y_v, **kw)
    logger.info("Registered Phase 3 hierarchical models: global_local, mixed_effects, multi_level_ensemble")
except ImportError:
    logger.debug("Phase 3 hierarchical models not available (missing hierarchical_model_training)")

try:
    from utils.enhanced_model_zoo import (
        train_catboost_embedding_model, train_quantile_model, train_conformal_residual_boost,
    )
    TRAINING_REGISTRY['catboost_embedding'] = lambda X_tr, y_tr, X_v, y_v, **kw: _adapt_hierarchical_training(train_catboost_embedding_model, X_tr, y_tr, X_v, y_v, **kw)
    TRAINING_REGISTRY['quantile_regression'] = lambda X_tr, y_tr, X_v, y_v, **kw: _adapt_hierarchical_training(train_quantile_model, X_tr, y_tr, X_v, y_v, **kw)
    TRAINING_REGISTRY['conformal_boost'] = lambda X_tr, y_tr, X_v, y_v, **kw: _adapt_hierarchical_training(train_conformal_residual_boost, X_tr, y_tr, X_v, y_v, **kw)
    logger.info("Registered Phase 7 enhanced models: catboost_embedding, quantile_regression, conformal_boost")
except ImportError:
    logger.debug("Phase 7 enhanced models not available (missing enhanced_model_zoo)")

try:
    from utils.model_combination import train_diverse_stacking
    TRAINING_REGISTRY['stacked_ensemble'] = lambda X_tr, y_tr, X_v, y_v, **kw: _adapt_hierarchical_training(train_diverse_stacking, X_tr, y_tr, X_v, y_v, **kw)
    logger.info("Registered Phase 8 combination model: stacked_ensemble")
except ImportError:
    logger.debug("Phase 8 stacked_ensemble not available (missing model_combination)")


# Models that require feature matrices (X)
FEATURE_BASED_MODELS = {
    'lightgbm', 'xgboost', 'catboost', 'random_forest',
    'zero_inflated', 'hurdle_model', 'tweedie',
    'ordinal_regression', 'discrete_classifier', 'hybrid_discrete',
    'weighted_ensemble', 'stacking', 'tft',
    # Multi-horizon models (feature-based, optimized for longer horizons)
    'multi_horizon_lightgbm', 'multi_horizon_xgboost', 'multi_horizon_ensemble',
    # Phase 3/7/8: Hierarchical, enhanced, and combination models
    'global_local', 'mixed_effects', 'multi_level_ensemble',
    'catboost_embedding', 'quantile_regression', 'conformal_boost',
    'stacked_ensemble',
}

# Models that are univariate (y only)
UNIVARIATE_MODELS = {
    'croston', 'sba', 'tsb', 'imapa',
    'arima', 'sarima', 'ets', 'theta', 'tbats',
    'prophet', 'bsts',
    'lstm', 'nbeats', 'deepar', 'wavenet',
}


def is_feature_based(model_type: str) -> bool:
    """Check if a model type requires features (X)."""
    return model_type.lower() in FEATURE_BASED_MODELS


def train_model_by_name(
    model_type: str,
    X_train: Optional[np.ndarray],
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: np.ndarray,
    **kwargs
) -> TrainingResult:
    """
    Train any model by name using the training registry.

    Parameters
    ----------
    model_type : str
        Name of model (e.g., 'lightgbm', 'croston', 'prophet')
    X_train, X_val : Feature matrices (for feature-based models)
    y_train, y_val : Target arrays
    **kwargs : Additional model-specific parameters

    Returns
    -------
    TrainingResult
    """
    model_type = model_type.lower()

    if model_type not in TRAINING_REGISTRY:
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {', '.join(sorted(TRAINING_REGISTRY.keys()))}")

    train_func = TRAINING_REGISTRY[model_type]

    if is_feature_based(model_type):
        if X_train is None or X_val is None:
            raise ValueError(f"Model '{model_type}' requires features (X_train, X_val)")
        result = train_func(X_train, y_train, X_val, y_val, **kwargs)
    else:
        result = train_func(y_train, y_val, **kwargs)

    # Adapt MultiHorizonTrainingResult to standard TrainingResult
    # so the rest of the pipeline (which expects .val_wape) works seamlessly
    if not isinstance(result, TrainingResult) and hasattr(result, 'val_target_wape'):
        result = TrainingResult(
            model=result.model,
            model_type=result.model_type,
            hyperparameters=result.hyperparameters,
            train_wape=result.train_target_wape,
            val_wape=result.val_target_wape,
            is_feature_based=result.is_feature_based,
        )

    return result


# =============================================================================
# HIGH-LEVEL TRAINING FUNCTIONS
# =============================================================================

def train_best_model_for_segment(
    X_train: Optional[np.ndarray],
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: np.ndarray,
    demand_pattern: str,
    allowed_families: List[str],
    max_candidates: int = 5,
    ensure_baseline: bool = True,
    forecast_horizon: Optional[int] = None,
    enable_multi_horizon: bool = False,
) -> Tuple[TrainingResult, str]:
    """
    Train multiple models and select the best one based on validation WAPE.

    IMPORTANT: For intermittent/lumpy patterns, this function ensures BOTH:
    1. Specialist models are tried (Croston/SBA/TSB for intermittent)
    2. A baseline model (LightGBM) is ALWAYS included for comparison

    Parameters
    ----------
    X_train, X_val : Feature matrices (optional for univariate models)
    y_train, y_val : Target arrays
    demand_pattern : str
        One of: 'smooth', 'erratic', 'intermittent', 'lumpy'
    allowed_families : List[str]
        Model families to consider (from config.yaml)
    max_candidates : int
        Maximum number of models to try
    ensure_baseline : bool
        Always include LightGBM as baseline comparison
    forecast_horizon : int, optional
        Config forecast horizon, passed as max_horizon to multi-horizon models.
    enable_multi_horizon : bool
        If True, include multi-horizon direct forecasting models for suitable patterns.

    Returns
    -------
    Tuple[TrainingResult, str]
        (best_result, best_model_type)
    """
    # Get recommended models for this pattern
    from utils.model_registry import get_recommended_models

    candidates = get_recommended_models(
        demand_pattern,
        allowed_families=allowed_families,
        max_models=max_candidates,
        ensure_baseline=ensure_baseline,
        forecast_horizon=forecast_horizon,
        enable_multi_horizon=enable_multi_horizon,
    )

    if not candidates:
        logger.warning(f"No models available for {demand_pattern}. Using LightGBM.")
        candidates = ['lightgbm']

    best_result = None
    best_model_type = None
    all_results = []

    for model_type in candidates:
        try:
            # Pass max_horizon/target_horizon for multi-horizon models
            extra_kwargs = {}
            if model_type.startswith('multi_horizon') and forecast_horizon is not None:
                extra_kwargs['max_horizon'] = forecast_horizon
                extra_kwargs['target_horizon'] = forecast_horizon
            result = train_model_by_name(model_type, X_train, y_train, X_val, y_val, **extra_kwargs)
            all_results.append((model_type, result))
            logger.info(f"  {model_type}: val_WAPE = {result.val_wape:.4f}")

            if best_result is None or result.val_wape < best_result.val_wape:
                best_result = result
                best_model_type = model_type

        except Exception as e:
            logger.warning(f"Failed to train {model_type}: {e}")
            continue

    if best_result is None:
        # Fallback
        logger.warning("All models failed. Using simple mean.")
        mean_val = np.mean(y_train)
        best_result = TrainingResult(
            model={'type': 'mean_fallback', 'mean': mean_val},
            model_type='mean_fallback',
            hyperparameters={},
            train_wape=compute_wape(y_train, np.full(len(y_train), mean_val)),
            val_wape=compute_wape(y_val, np.full(len(y_val), mean_val)),
            is_feature_based=False,
        )
        best_model_type = 'mean_fallback'

    return best_result, best_model_type


def train_all_model_groups(
    train_manifest: pd.DataFrame,
    feature_dir: str,
    model_dir: str,
    segmentation_context: Dict[str, Any],
    allowed_families: List[str],
    target_col: str = 'target',
    max_candidates: int = 5,
    forecast_horizon: Optional[int] = None,
    enable_multi_horizon: bool = False,
) -> Dict[str, Any]:
    """
    Train models for ALL model groups in manifest.

    This is the MAIN function agents should call for full training.

    Parameters
    ----------
    train_manifest : pd.DataFrame
        Manifest with 'model_group' column listing all groups to train
    feature_dir : str
        Directory containing {model_group}_{split}_features.csv files
    model_dir : str
        Directory to save trained models
    segmentation_context : Dict
        Context from segmentation crew with demand patterns
    allowed_families : List[str]
        Allowed model families from config
    target_col : str
        Name of target column in feature files
    max_candidates : int
        Max models to try per group
    forecast_horizon : int, optional
        Number of forecast periods. Enables multi-horizon models when >= 3.
    enable_multi_horizon : bool
        If True, include multi-horizon direct forecasting models for suitable patterns.

    Returns
    -------
    Dict with:
        - 'model_specs': List of per-group specs
        - 'overall_wape': Weighted average WAPE across groups
        - 'models_trained': Count of successfully trained groups
    """
    from utils.agent_utilities import load_csv, save_csv, save_json, SmartPrinter

    printer = SmartPrinter(max_prints=50)
    os.makedirs(model_dir, exist_ok=True)

    model_groups = train_manifest['model_group'].unique()
    printer.print(f"Training models for {len(model_groups)} model groups...")

    model_specs = []
    total_actuals = 0
    total_errors = 0

    for mg in model_groups:
        try:
            # Load feature files — format-agnostic via the feature_io helpers
            # so the per-model-group split files can be parquet too.
            from utils.feature_io import read_features_intermediate
            train_df = read_features_intermediate(feature_dir, f'{mg}_train_features')
            val_df   = read_features_intermediate(feature_dir, f'{mg}_val_features')

            # Get feature columns (exclude target and metadata)
            exclude_cols = {target_col, 'key', 'year_week', 'split', 'model_group', 'segment_id'}
            feature_cols = [c for c in train_df.columns if c not in exclude_cols]

            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_val = val_df[feature_cols].values
            y_val = val_df[target_col].values

            # Get demand pattern from segmentation context
            seg_info = segmentation_context.get('model_recommendations_by_segment', {}).get(mg, {})
            demand_pattern = seg_info.get('primary_demand_pattern', 'smooth')

            # Train best model
            result, best_type = train_best_model_for_segment(
                X_train, y_train, X_val, y_val,
                demand_pattern=demand_pattern,
                allowed_families=allowed_families,
                max_candidates=max_candidates,
                forecast_horizon=forecast_horizon,
                enable_multi_horizon=enable_multi_horizon,
            )

            # Save model
            model_path = os.path.join(model_dir, f'{mg}_model.pkl')
            joblib.dump(result.model, model_path)

            # Record spec
            spec = {
                'model_group': mg,
                'model_type': best_type,
                'demand_pattern': demand_pattern,
                'val_wape': result.val_wape,
                'train_wape': result.train_wape,
                'hyperparameters': result.hyperparameters,
                'feature_cols': feature_cols,
                'model_path': model_path,
            }

            # Add multi-horizon metadata so inference retraining can use it
            if best_type.startswith('multi_horizon'):
                spec['is_multi_horizon'] = True
                spec['multi_horizon_config'] = {
                    'strategy': 'direct_separate',
                    'max_horizon': forecast_horizon or 10,
                    'target_horizon': forecast_horizon or 10,
                }

            model_specs.append(spec)

            # Accumulate for overall WAPE
            total_actuals += np.sum(np.abs(y_val))
            total_errors += result.val_wape * np.sum(np.abs(y_val))

            printer.print(f"  {mg}: {best_type} WAPE={result.val_wape:.3f}")

        except Exception as e:
            logger.error(f"Failed to train model for {mg}: {e}")
            printer.print(f"  {mg}: FAILED - {str(e)[:50]}")
            model_specs.append({
                'model_group': mg,
                'model_type': 'failed',
                'error': str(e),
            })

    # Calculate overall WAPE
    overall_wape = total_errors / total_actuals if total_actuals > 0 else 1.0

    # Save model specs
    save_json(model_specs, os.path.join(model_dir, 'model_specs.json'))

    # Save summary
    summary = {
        'overall_val_wape': overall_wape,
        'models_trained': sum(1 for s in model_specs if s.get('model_type') != 'failed'),
        'models_failed': sum(1 for s in model_specs if s.get('model_type') == 'failed'),
        'total_model_groups': len(model_groups),
    }
    save_json(summary, os.path.join(model_dir, 'training_summary.json'))

    printer.print(f"\nOverall val WAPE: {overall_wape:.3f}")
    printer.print(f"Trained: {summary['models_trained']}/{summary['total_model_groups']} groups")

    return {
        'model_specs': model_specs,
        'overall_wape': overall_wape,
        'models_trained': summary['models_trained'],
        'summary': summary,
    }


# =============================================================================
# HYPERPARAMETER TUNING WITH OPTUNA
# =============================================================================

def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    timeout: int = 300,
) -> TuningResult:
    """
    Tune LightGBM hyperparameters using Optuna.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials
    timeout : int
        Maximum tuning time in seconds

    Returns
    -------
    TuningResult with best model and parameters
    """
    import optuna
    import lightgbm as lgb
    import time

    start_time = time.time()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'objective': 'regression',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        val_pred = np.clip(model.predict(X_val), 0, None)
        wape = compute_wape(y_val, val_pred)

        return wape

    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # Train final model with best params
    best_params = study.best_params
    best_params['objective'] = 'regression'
    best_params['verbosity'] = -1
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False)])

    val_pred = np.clip(best_model.predict(X_val), 0, None)
    best_wape = compute_wape(y_val, val_pred)

    return TuningResult(
        best_model=best_model,
        best_params=best_params,
        best_val_wape=best_wape,
        all_trials=[{'params': t.params, 'value': t.value} for t in study.trials],
        n_trials=len(study.trials),
        tuning_time_seconds=time.time() - start_time,
    )


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    timeout: int = 300,
) -> TuningResult:
    """Tune XGBoost hyperparameters using Optuna."""
    import optuna
    import xgboost as xgb
    import time

    start_time = time.time()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'objective': 'reg:squarederror',
            'verbosity': 0,
            'random_state': 42,
            'n_jobs': -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_pred = np.clip(model.predict(X_val), 0, None)
        return compute_wape(y_val, val_pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['verbosity'] = 0
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = np.clip(best_model.predict(X_val), 0, None)
    best_wape = compute_wape(y_val, val_pred)

    return TuningResult(
        best_model=best_model,
        best_params=best_params,
        best_val_wape=best_wape,
        all_trials=[{'params': t.params, 'value': t.value} for t in study.trials],
        n_trials=len(study.trials),
        tuning_time_seconds=time.time() - start_time,
    )


def tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    timeout: int = 300,
) -> TuningResult:
    """Tune CatBoost hyperparameters using Optuna."""
    import optuna
    from catboost import CatBoostRegressor
    import time

    start_time = time.time()

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
        }

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)

        val_pred = np.clip(model.predict(X_val), 0, None)
        return compute_wape(y_val, val_pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    best_params['allow_writing_files'] = False

    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    val_pred = np.clip(best_model.predict(X_val), 0, None)
    best_wape = compute_wape(y_val, val_pred)

    return TuningResult(
        best_model=best_model,
        best_params=best_params,
        best_val_wape=best_wape,
        all_trials=[{'params': t.params, 'value': t.value} for t in study.trials],
        n_trials=len(study.trials),
        tuning_time_seconds=time.time() - start_time,
    )


def tune_tweedie(X_train, y_train, X_val, y_val, n_trials=20, timeout=200):
    """Tune Tweedie model: variance power + LightGBM base params."""
    import lightgbm as lgb
    best_wape = float('inf')
    best_model = None
    best_params = {}

    for power in [1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9]:
        for lr in [0.03, 0.05, 0.08]:
            for leaves in [15, 31, 63]:
                m = lgb.LGBMRegressor(
                    objective='tweedie', tweedie_variance_power=power,
                    n_estimators=400, learning_rate=lr, num_leaves=leaves,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1,
                )
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
                preds = np.maximum(m.predict(X_val), 0)
                total = np.sum(np.abs(y_val))
                wape = float(np.sum(np.abs(y_val - preds)) / total) if total > 1e-10 else 1.0
                if wape < best_wape:
                    best_wape = wape
                    best_model = m
                    best_params = {'tweedie_variance_power': power, 'learning_rate': lr, 'num_leaves': leaves}
    return TuningResult(best_model=best_model, best_params=best_params,
                         best_val_wape=best_wape, n_trials=len([1.05,1.1,1.2,1.3,1.5,1.7,1.9])*9, tuning_time_seconds=0)


def tune_zero_inflated(X_train, y_train, X_val, y_val, n_trials=15, timeout=200):
    """Tune zero-inflated model: threshold + base LightGBM params."""
    best_wape = float('inf')
    best_model = None
    best_params = {}

    zero_frac = float((y_train == 0).mean())
    thresholds = [max(0.1, zero_frac - 0.2), zero_frac - 0.1, zero_frac, min(0.95, zero_frac + 0.1)]

    for threshold in thresholds:
        for lr in [0.03, 0.05]:
            try:
                result = train_zero_inflated(X_train, y_train, X_val, y_val,
                                              zero_threshold=threshold, learning_rate=lr)
                if result.val_wape < best_wape:
                    best_wape = result.val_wape
                    best_model = result.model
                    best_params = {'zero_threshold': threshold, 'learning_rate': lr}
            except Exception:
                continue
    if best_model is None:
        result = train_zero_inflated(X_train, y_train, X_val, y_val)
        return TuningResult(best_model=result.model, best_params={}, best_val_wape=result.val_wape, n_trials=1, tuning_time_seconds=0)
    return TuningResult(best_model=best_model, best_params=best_params,
                         best_val_wape=best_wape, n_trials=len(thresholds)*2, tuning_time_seconds=0)


def tune_hurdle_model(X_train, y_train, X_val, y_val, n_trials=15, timeout=200):
    """Tune hurdle model: threshold + base params."""
    best_wape = float('inf')
    best_model = None
    best_params = {}

    zero_frac = float((y_train == 0).mean())
    thresholds = [max(0.1, zero_frac - 0.15), zero_frac, min(0.95, zero_frac + 0.1)]

    for threshold in thresholds:
        for lr in [0.03, 0.05]:
            try:
                result = train_hurdle_model(X_train, y_train, X_val, y_val,
                                             zero_threshold=threshold, learning_rate=lr)
                if result.val_wape < best_wape:
                    best_wape = result.val_wape
                    best_model = result.model
                    best_params = {'zero_threshold': threshold, 'learning_rate': lr}
            except Exception:
                continue
    if best_model is None:
        result = train_hurdle_model(X_train, y_train, X_val, y_val)
        return TuningResult(best_model=result.model, best_params={}, best_val_wape=result.val_wape, n_trials=1, tuning_time_seconds=0)
    return TuningResult(best_model=best_model, best_params=best_params,
                         best_val_wape=best_wape, n_trials=len(thresholds)*2, tuning_time_seconds=0)


def tune_model_hyperparameters(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    timeout: int = 300,
) -> TuningResult:
    """
    Tune hyperparameters for supported model types.

    Supports: lightgbm, xgboost, catboost, tweedie, zero_inflated, hurdle_model
    """
    model_type = model_type.lower()

    tuning_registry = {
        'lightgbm': tune_lightgbm,
        'xgboost': tune_xgboost,
        'catboost': tune_catboost,
        'tweedie': tune_tweedie,
        'zero_inflated': tune_zero_inflated,
        'hurdle_model': tune_hurdle_model,
    }

    if model_type in tuning_registry:
        return tuning_registry[model_type](X_train, y_train, X_val, y_val, n_trials, timeout)
    else:
        logger.warning(f"Tuning not implemented for {model_type}. Using defaults.")
        result = train_model_by_name(model_type, X_train, y_train, X_val, y_val)
        return TuningResult(
            best_model=result.model,
            best_params=result.hyperparameters,
            best_val_wape=result.val_wape,
            n_trials=0,
            tuning_time_seconds=0,
        )


# =============================================================================
# MODEL PERSISTENCE UTILITIES
# =============================================================================

def save_model(model: Any, path: str) -> None:
    """Save model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")


def load_model(path: str) -> Any:
    """Load model from disk."""
    return joblib.load(path)


def save_training_result(result: TrainingResult, model_path: str, metadata_path: str) -> None:
    """Save training result (model + metadata)."""
    save_model(result.model, model_path)
    with open(metadata_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


# =============================================================================
# POST-TRAINING OPTIMIZATION
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of post-training optimization."""
    original_predictions: np.ndarray
    optimized_predictions: np.ndarray
    original_wape: float
    optimized_wape: float
    improvement_pct: float
    bias_correction_applied: bool
    calibration_applied: bool
    bias_info: Optional[Dict] = None
    calibration_method: Optional[str] = None


def apply_post_training_optimization(
    predictions: np.ndarray,
    actuals: np.ndarray,
    test_predictions: Optional[np.ndarray] = None,
    enable_bias_correction: bool = True,
    enable_calibration: bool = True,
    calibration_method: str = 'isotonic',
    significance_level: float = 0.05,
) -> OptimizationResult:
    """
    Apply post-training optimization: bias correction and calibration.

    This is applied AFTER model training to improve forecast accuracy.
    Uses sophisticated techniques from forecast_optimization.py.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions on validation/training set (used for fitting corrections)
    actuals : np.ndarray
        Actual values corresponding to predictions
    test_predictions : np.ndarray, optional
        Predictions to apply corrections to. If None, applies to `predictions`
    enable_bias_correction : bool
        Whether to apply bias correction
    enable_calibration : bool
        Whether to apply forecast calibration
    calibration_method : str
        Calibration method: 'isotonic', 'platt', 'linear'
    significance_level : float
        Significance threshold for bias detection

    Returns
    -------
    OptimizationResult
        Optimized predictions with diagnostics

    Example
    -------
    >>> # After training a model
    >>> val_pred = model.predict(X_val)
    >>> opt_result = apply_post_training_optimization(val_pred, y_val)
    >>> print(f"WAPE improved from {opt_result.original_wape:.2%} to {opt_result.optimized_wape:.2%}")
    >>>
    >>> # Apply same corrections to test predictions
    >>> test_pred = model.predict(X_test)
    >>> test_opt = apply_post_training_optimization(val_pred, y_val, test_predictions=test_pred)
    """
    from utils.forecast_optimization import (
        detect_systematic_bias,
        apply_bias_correction,
        calibrate_forecasts,
    )

    # Original metrics
    original_wape = compute_wape(actuals, predictions)

    # Determine what to optimize
    target_predictions = test_predictions if test_predictions is not None else predictions.copy()
    optimized = target_predictions.copy()

    bias_info = None
    bias_applied = False
    calibration_applied = False

    # Step 1: Bias Correction
    if enable_bias_correction:
        try:
            bias_result = detect_systematic_bias(
                predictions, actuals,
                test_type='both',
                significance_level=significance_level
            )

            if bias_result.correction_applied:
                optimized = apply_bias_correction(optimized, bias_result)
                bias_applied = True
                bias_info = {
                    'additive_bias': float(bias_result.additive_bias),
                    'multiplicative_bias': float(bias_result.multiplicative_bias),
                    'correction_method': bias_result.correction_method,
                    'significance': float(bias_result.bias_significance),
                }
                logger.info(f"Bias correction applied: {bias_result.correction_method}, "
                           f"additive={bias_result.additive_bias:.4f}, "
                           f"multiplicative={bias_result.multiplicative_bias:.4f}")
        except Exception as e:
            logger.warning(f"Bias correction failed: {e}")

    # Step 2: Calibration
    if enable_calibration:
        try:
            optimized = calibrate_forecasts(
                predictions=predictions,
                actuals=actuals,
                test_predictions=optimized,
                method=calibration_method
            )
            calibration_applied = True
            logger.info(f"Calibration applied using {calibration_method} method")
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")

    # Ensure non-negative
    optimized = np.maximum(optimized, 0)

    # Compute optimized metrics (only if optimizing on same set)
    if test_predictions is None:
        optimized_wape = compute_wape(actuals, optimized)
        improvement = (original_wape - optimized_wape) / original_wape * 100 if original_wape > 0 else 0
    else:
        # Can't compute improvement without test actuals
        optimized_wape = original_wape  # Placeholder
        improvement = 0.0

    return OptimizationResult(
        original_predictions=predictions,
        optimized_predictions=optimized,
        original_wape=original_wape,
        optimized_wape=optimized_wape,
        improvement_pct=improvement,
        bias_correction_applied=bias_applied,
        calibration_applied=calibration_applied,
        bias_info=bias_info,
        calibration_method=calibration_method if calibration_applied else None,
    )


def optimize_model_predictions(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    enable_bias_correction: bool = True,
    enable_calibration: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to optimize predictions for a trained model.

    Parameters
    ----------
    model : trained model with predict method
    X_train, y_train : Training data (for fitting calibration)
    X_val, y_val : Validation data (for fitting corrections)
    X_test : Test data to generate optimized predictions for

    Returns
    -------
    Tuple[np.ndarray, Dict]
        Optimized predictions and optimization info
    """
    # Get raw predictions
    if hasattr(model, 'predict'):
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test) if X_test is not None else None
    else:
        # For dict-based models (univariate)
        forecast = model.get('forecast', 0)
        train_pred = np.full(len(y_train), forecast)
        val_pred = np.full(len(y_val), forecast)
        test_pred = np.full(len(X_test), forecast) if X_test is not None else None

    # Apply optimization
    opt_result = apply_post_training_optimization(
        predictions=val_pred,
        actuals=y_val,
        test_predictions=test_pred if test_pred is not None else val_pred,
        enable_bias_correction=enable_bias_correction,
        enable_calibration=enable_calibration,
    )

    info = {
        'original_val_wape': opt_result.original_wape,
        'optimized_val_wape': opt_result.optimized_wape,
        'improvement_pct': opt_result.improvement_pct,
        'bias_correction_applied': opt_result.bias_correction_applied,
        'calibration_applied': opt_result.calibration_applied,
        'bias_info': opt_result.bias_info,
    }

    return opt_result.optimized_predictions, info


# =============================================================================
# FULL TRAINING PIPELINE - Single entry point for training crew
# =============================================================================

@dataclass
class FullTrainingResult:
    """Result from run_full_training_pipeline."""
    models_trained: int
    models_failed: int
    overall_wape: float  # Validation WAPE (used for model selection)
    overall_test_wape: float = 0.0  # Test WAPE (holdout evaluation)
    model_specs: List[Dict[str, Any]] = field(default_factory=list)
    final_specs_path: str = ''
    diagnostic_context_path: str = ''
    success: bool = False
    error_message: Optional[str] = None
    # NEW: Detailed metrics
    val_wape_by_step: Dict[int, float] = field(default_factory=dict)  # WAPE by forecast step
    test_wape_by_step: Dict[int, float] = field(default_factory=dict)
    forecast_horizon: int = 8
    recursive_validation_used: bool = False
    # NEW: Ensemble info
    ensemble_created: bool = False
    ensemble_metadata: Dict[str, Any] = field(default_factory=dict)
    # NEW: All candidate comparisons
    candidate_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # STATE-OF-THE-ART: Walk-Forward Cross-Validation summary
    walk_forward_cv_enabled: bool = False
    walk_forward_cv_summary: Dict[str, Any] = field(default_factory=dict)
    concept_drift_detected: bool = False
    drift_affected_segments: List[str] = field(default_factory=list)


def run_full_training_pipeline(
    feature_dir: str,
    model_dir: str,
    target_col: str,
    strategy_path: Optional[str] = None,
    enable_meta_learning: bool = True,
    enable_ensemble_optimization: bool = True,
    enable_bias_correction: bool = True,
    enable_forecast_calibration: bool = True,
    prediction_intervals: bool = False,
    prediction_interval_confidence: float = 0.9,
    max_prints: int = 15,
    # NEW: Context-driven parameters from upstream analysis
    segmentation_context: Optional[Dict[str, Any]] = None,
    feature_context: Optional[Dict[str, Any]] = None,
    # OR explicit context-driven parameters
    segment_model_strategies: Optional[Dict[str, Dict]] = None,
    demand_pattern_distribution: Optional[Dict[str, int]] = None,
    n_features: Optional[int] = None,
    feature_categories: Optional[Dict[str, int]] = None,
    # Hyperparameter hints from context
    early_stopping_rounds: int = 50,
    max_candidates_per_group: int = 3,
    # NEW: Multi-step forecasting parameters
    forecast_horizon: int = 8,  # Number of steps to forecast (t+1 to t+horizon)
    use_recursive_validation: bool = True,  # Use recursive forecasting for validation WAPE
    key_col: str = 'key',  # Key column name for grouping
    date_col: str = None,  # Date column name (auto-resolved from time_format if None)
    ensemble_top_k: int = 3,  # Number of models to include in ensemble
    # STATE-OF-THE-ART: Walk-Forward Cross-Validation parameters
    enable_walk_forward_cv: bool = True,  # Enable temporal CV for robust model selection
    walk_forward_n_folds: int = 5,  # Number of temporal folds
    walk_forward_min_train_periods: Optional[int] = None,  # Auto: 52 for weekly, 12 for monthly
    walk_forward_strategy: str = 'expanding',  # 'expanding' or 'rolling'
    walk_forward_rolling_window: Optional[int] = None,  # Auto: 104 for weekly, 24 for monthly
    walk_forward_optimize_ensemble: bool = True,  # Optimize ensemble weights across folds
    walk_forward_detect_drift: bool = True,  # Detect concept drift over time
    # Time format for period-aware defaults
    time_format: str = 'year_week',  # 'year_week' or 'year_month'
    # Bias calibration parameters
    apply_bias_calibration: bool = True,  # Enable bias calibration from validation predictions
    bias_calibration_buckets: int = 5,  # Number of zero_fraction buckets for calibration
    bias_calibration_factor_min: float = 0.2,  # Min calibration factor (don't reduce by >80%)
    bias_calibration_factor_max: float = 2.0,  # Max calibration factor (don't increase by >2x)
    # ── Per-model-group parallelism (added 2026-04 for DIQ scale) ──
    # When >1, the per-mg training loop dispatches each group to a Ray
    # remote task with `num_cpus=cpu_count // workers` so OpenMP
    # threading inside LightGBM/XGBoost stays within budget.  Default
    # 1 = sequential (existing behaviour).
    parallel_training_workers: int = 1,
    # When set, validate each candidate using ONLY the recursive WAPE
    # at this specific lag, instead of the average over all lags up to
    # forecast_horizon.  The walk-forward still has to traverse lag
    # 1..N (each step's prediction feeds the next step's lag features
    # for the 1-step-ahead candidates), but stops at `lag` instead of
    # going all the way to forecast_horizon.  For DIQ's lag-4 backtest,
    # set this to 5 — gives ~2.6x recursive-eval speedup AND aligns
    # the selection metric with the downstream metric.
    recursive_validation_lag: Optional[int] = None,
    # When True, the inference pipeline will use Direct Multi-Horizon
    # (DMH) — train one model per horizon h that predicts y[t+h]
    # directly, no recursive feedback.  When that's the primary
    # inference path, the per-MG candidate models trained here are
    # only used for (a) bias-calibration validation predictions and
    # (b) the recursive-forecast fallback if DMH crashes.  In both
    # of those uses, single-step direct-WAPE is a perfectly good
    # candidate-selection metric — there's no point spending 13x
    # extra compute walking each candidate forward 13 steps to score
    # multi-step performance that DMH supersedes anyway.  When this
    # flag is True we therefore AUTO-SKIP the recursive eval below
    # regardless of `use_recursive_validation`, recovering the ~13x
    # training speedup without changing accuracy semantics for the
    # primary forecast path.
    use_direct_multi_horizon: bool = False,
) -> FullTrainingResult:
    """
    Run the complete training pipeline for all model groups.

    This is the MAIN entry point for the Training Crew executor task.
    It handles:
    1. Loading training strategy and feature data
    2. Meta-learning model selection (if enabled)
    3. Training models for each model_level
    4. Ensemble optimization (if enabled)
    5. Post-training optimization (bias correction, calibration)
    6. Saving all outputs (models, specs, diagnostic context)

    Parameters
    ----------
    feature_dir : str
        Directory containing train_features.csv, val_features.csv, training_manifest.csv
    model_dir : str
        Directory to save trained models and outputs
    target_col : str
        Name of target column
    strategy_path : str, optional
        Path to training_strategy.json. If None, uses default strategy.
    enable_meta_learning : bool
        Use meta-features to intelligently select models
    enable_ensemble_optimization : bool
        Optimize ensemble weights for top models
    enable_bias_correction : bool
        Apply bias correction post-training
    enable_forecast_calibration : bool
        Apply forecast calibration post-training
    prediction_intervals : bool
        Generate prediction intervals
    prediction_interval_confidence : float
        Confidence level for prediction intervals
    max_prints : int
        Maximum print statements for output control
    segmentation_context : Dict, optional
        Context from segmentation_to_training_context.json. If provided, extracts:
        - segment_model_strategy: Per-segment model recommendations
        - clustering_quality: Quality metrics from segmentation
    feature_context : Dict, optional
        Context from feature_to_training_context.json. If provided, extracts:
        - feature_summary: Feature counts and categories
        - model_recommendations_by_segment: Model hints from feature analysis
    segment_model_strategies : Dict, optional
        Explicit per-segment model strategies (primary model, hyperparams)
    demand_pattern_distribution : Dict, optional
        Distribution of demand patterns across segments
    n_features : int, optional
        Total number of features (for hyperparameter tuning)
    feature_categories : Dict, optional
        Breakdown of feature types (lag, rolling, calendar, etc.)
    early_stopping_rounds : int
        Early stopping rounds for boosting models (from context)
    max_candidates_per_group : int
        Maximum candidate models to try per group (from context)

    Returns
    -------
    FullTrainingResult
        Complete training results including paths to all outputs
    """
    import time
    import traceback
    from utils.agent_utilities import load_csv, load_json, save_json, SmartPrinter
    from utils.model_selection_intelligence import (
        build_meta_features, rank_models_by_pattern_fit,
        get_candidate_models_for_pattern, PATTERN_MODEL_PRIORITY
    )
    from utils.forecast_optimization import cross_validated_ensemble_weights
    from utils.intelligent_modeling import (
        select_features_for_model_level,
        categorize_features_for_model_level,
    )
    # STATE-OF-THE-ART: Import enhanced training capabilities
    from utils.state_of_art_training import (
        get_segment_specific_hyperparams,
        compute_all_segment_hyperparams,
        build_enhanced_meta_features,
        rank_models_with_enhanced_features,
        compute_pattern_aware_ensemble_weights,
        create_pattern_aware_ensemble,
        calibrate_zero_inflated_threshold,
        optimize_hurdle_threshold,
        learn_segment_bias_calibration,
        apply_segment_bias_calibration,
        compute_all_segment_calibrations,
        HyperparameterProfile,
        EnhancedMetaFeatures,
        SegmentBiasCalibration,
    )
    # STATE-OF-THE-ART: Walk-Forward Cross-Validation for robust model selection
    from utils.walk_forward_cv import (
        WalkForwardCV,
        WalkForwardCVResult,
        walk_forward_model_selection,
        create_period_index,
        detect_concept_drift,
    )

    printer = SmartPrinter(max_prints=max_prints)
    os.makedirs(model_dir, exist_ok=True)

    # =========================================================================
    # TIME-FORMAT-AWARE DEFAULTS
    # Walk-forward CV params adapt to weekly (52 periods/yr) vs monthly (12/yr)
    # =========================================================================
    is_monthly = (time_format == 'year_month')
    if date_col is None:
        date_col = 'year_month' if is_monthly else 'year_week'
    if walk_forward_min_train_periods is None:
        walk_forward_min_train_periods = 12 if is_monthly else 52
    if walk_forward_rolling_window is None:
        walk_forward_rolling_window = 24 if is_monthly else 104
    logger.info(f"Walk-forward CV: min_train={walk_forward_min_train_periods}, "
                f"rolling_window={walk_forward_rolling_window} (time_format={time_format})")

    # =========================================================================
    # EXTRACT CONTEXT-DRIVEN PARAMETERS FROM UPSTREAM CONTEXT
    # =========================================================================
    if segmentation_context:
        logger.info("Using segmentation context for training parameters")
        # Extract segment-level model strategies
        if segment_model_strategies is None:
            segment_model_strategies = segmentation_context.get('segment_model_strategy', {})
        # Extract clustering quality for confidence-based decisions
        clustering_quality = segmentation_context.get('clustering_quality', {})
        if clustering_quality.get('silhouette', 0) < 0.3:
            logger.warning("Low silhouette score - segments may be poorly defined")

    if feature_context:
        logger.info("Using feature context for training parameters")
        # Extract feature summary
        feat_summary = feature_context.get('feature_summary', {})
        if n_features is None:
            n_features = feat_summary.get('total_features', 0)
        if feature_categories is None:
            feature_categories = feat_summary.get('feature_categories', {})
        # Extract model recommendations from feature analysis
        if segment_model_strategies is None:
            segment_model_strategies = feature_context.get('model_recommendations_by_segment', {})

        # Log feature context info
        if n_features:
            printer.print(f"Feature context: {n_features} features")
            if feature_categories:
                lag_count = feature_categories.get('lag_features', 0)
                roll_count = feature_categories.get('rolling_features', 0)
                printer.print(f"  Lag: {lag_count}, Rolling: {roll_count}")

    # =========================================================================
    # STATE-OF-THE-ART: LOSS FUNCTION MAPPING
    # Maps recommended_loss from strategy to LightGBM/XGBoost objectives
    # =========================================================================
    LOSS_TO_OBJECTIVE = {
        'mse': {'lightgbm': 'regression', 'xgboost': 'reg:squarederror', 'catboost': 'RMSE'},
        'mae': {'lightgbm': 'regression_l1', 'xgboost': 'reg:absoluteerror', 'catboost': 'MAE'},
        'huber': {'lightgbm': 'huber', 'xgboost': 'reg:squarederror', 'catboost': 'Huber:delta=1.0'},  # XGB doesn't have huber natively
        'tweedie': {'lightgbm': 'tweedie', 'xgboost': 'reg:tweedie', 'catboost': 'Tweedie:variance_power=1.5'},
    }

    def get_loss_params(recommended_loss: str, model_type: str) -> Dict[str, Any]:
        """Get loss/objective parameters for a specific model type based on recommended loss."""
        if recommended_loss not in LOSS_TO_OBJECTIVE:
            return {}  # Use default

        objective_map = LOSS_TO_OBJECTIVE[recommended_loss]
        params = {}

        if model_type == 'lightgbm':
            params['objective'] = objective_map.get('lightgbm', 'regression')
            if recommended_loss == 'tweedie':
                params['tweedie_variance_power'] = 1.5  # Good default for intermittent demand
            elif recommended_loss == 'huber':
                params['alpha'] = 0.9  # Huber delta equivalent
        elif model_type == 'xgboost':
            params['objective'] = objective_map.get('xgboost', 'reg:squarederror')
            if recommended_loss == 'tweedie':
                params['tweedie_variance_power'] = 1.5
        elif model_type == 'catboost':
            params['loss_function'] = objective_map.get('catboost', 'RMSE')

        return params

    # =========================================================================
    # Phase-3 hierarchical models (global_local, mixed_effects,
    # multi_level_ensemble) require a real DataFrame with key_col and
    # hierarchy_col — the (X_train, y_train, X_val, y_val) matrix-only
    # contract used by most flat models is insufficient for them. The
    # two lookup sets below let create_training_fn route these models
    # through a DataFrame-aware training closure.
    # =========================================================================
    _HIERARCHICAL_MODELS = {'global_local', 'mixed_effects', 'multi_level_ensemble'}

    # =========================================================================
    # STATE-OF-THE-ART: Enhanced training function creation with segment-specific hyperparameters
    # =========================================================================
    def create_training_fn(
        model_type: str,
        loss_params: Dict[str, Any] = None,
        hyperparam_profile: HyperparameterProfile = None,
        early_stopping_rounds: int = 50,
        # NEW: DataFrame-level context required by Phase-3 hierarchical
        # models. Passed in from the per-segment training loop so
        # ``_adapt_hierarchical_training`` can build a proper
        # (key_col, hierarchy_col) indexed DataFrame rather than a
        # single-key fallback. When these are None / hierarchy_col is
        # not available, hierarchical model types return a no-op closure
        # so the candidate is recorded as failed and the pipeline moves
        # on without raising.
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        key_col: str = 'key',
        target_col_ctx: str = 'target',
        hierarchy_col: Optional[str] = None,
    ):
        """
        Create a training function with optional loss parameters and segment-specific hyperparameters.

        STATE-OF-THE-ART: This now supports:
        - Loss-specific parameters (tweedie, huber, mse)
        - Segment-specific hyperparameters from HyperparameterProfile
        - Early stopping configuration
        - Phase-3 hierarchical models (global_local, mixed_effects,
          multi_level_ensemble) when a hierarchy column is available
        """
        if loss_params is None:
            loss_params = {}

        # Get segment-specific hyperparameters if profile provided
        segment_params = {}
        if hyperparam_profile is not None:
            segment_params = hyperparam_profile.get_params_for_model(model_type)
            early_stopping_rounds = hyperparam_profile.early_stopping_rounds
            logger.debug(f"Using segment-specific params for {model_type}: {list(segment_params.keys())}")

        # Merge parameters: segment_params (base) + loss_params (override)
        final_params = {**segment_params, **loss_params}

        # ---- Phase-3 hierarchical models branch ----------------------------
        # Require (train_df, val_df, hierarchy_col) AND > 1 distinct key in
        # the training group. If any of those is missing we return a closure
        # that logs a skip and returns None — matching how other failed
        # candidates behave in the outer orchestrator.
        if model_type in _HIERARCHICAL_MODELS:
            reason = None
            if train_df is None or val_df is None or feature_cols is None:
                reason = "missing train/val DataFrame or feature_cols"
            elif hierarchy_col is None:
                reason = "no hierarchy_col resolved (segmentation artifact empty?)"
            elif hierarchy_col not in train_df.columns:
                reason = f"hierarchy_col '{hierarchy_col}' not in train_df columns"
            elif key_col not in train_df.columns:
                reason = f"key_col '{key_col}' not in train_df columns"
            elif train_df[key_col].nunique() < 2:
                reason = "only 1 unique key in training group (key-level model — hierarchical would be degenerate)"
            elif train_df[hierarchy_col].nunique() < 2:
                reason = f"only 1 unique '{hierarchy_col}' value in training group (no hierarchy to exploit)"

            if reason is not None:
                msg = f"{model_type}: skipped — {reason}"

                def _skip_fn(X_tr, y_tr, X_v, y_v):
                    logger.info(msg)
                    return None

                return _skip_fn

            hierarchy_fn = TRAINING_REGISTRY[model_type]
            _train_df = train_df
            _val_df = val_df
            _feature_cols = list(feature_cols)
            _key_col = key_col
            _target_col = target_col_ctx
            _hier_col = hierarchy_col
            # Freeze the merged loss+segment-specific params at closure
            # creation time. These flow through _adapt_hierarchical_training
            # into the underlying trainer, which now translates them into
            # LightGBM / XGBoost objective kwargs (tweedie, poisson, etc.).
            #
            # BUG FIX 2026-04: previously this closure only forwarded the
            # DataFrame context and silently dropped loss_params, so every
            # segment-pooled `global_local` model trained with LightGBM's
            # default squared-error objective — despite the segmentation
            # strategy recommending Tweedie(power=1.7) for all segments.
            # On zero-inflated lumpy demand (95% of UK keys) MSE is the
            # worst possible loss for WAPE because it chases the mean that
            # outliers pull upward, dominating pooled training.
            _hier_loss_params = dict(final_params) if final_params else {}
            _hier_early_stop = early_stopping_rounds

            def _hier_fn(X_tr, y_tr, X_v, y_v):
                return hierarchy_fn(
                    X_tr, y_tr, X_v, y_v,
                    train_df=_train_df,
                    val_df=_val_df,
                    feature_cols=_feature_cols,
                    key_col=_key_col,
                    target_col=_target_col,
                    hierarchy_col=_hier_col,
                    loss_params=_hier_loss_params,
                    early_stopping_rounds=_hier_early_stop,
                )

            return _hier_fn

        if model_type == 'lightgbm':
            return lambda X_tr, y_tr, X_v, y_v: train_lightgbm(
                X_tr, y_tr, X_v, y_v, params=final_params, early_stopping_rounds=early_stopping_rounds
            )
        elif model_type == 'xgboost':
            return lambda X_tr, y_tr, X_v, y_v: train_xgboost(
                X_tr, y_tr, X_v, y_v, params=final_params, early_stopping_rounds=early_stopping_rounds
            )
        elif model_type == 'catboost':
            return lambda X_tr, y_tr, X_v, y_v: train_catboost(
                X_tr, y_tr, X_v, y_v, params=final_params, early_stopping_rounds=early_stopping_rounds
            )
        elif model_type == 'random_forest':
            return lambda X_tr, y_tr, X_v, y_v: train_random_forest(X_tr, y_tr, X_v, y_v, params=final_params)
        elif model_type == 'zero_inflated':
            # Pass threshold from profile if available
            threshold = hyperparam_profile.zi_threshold if hyperparam_profile else 0.5
            return lambda X_tr, y_tr, X_v, y_v: train_zero_inflated(X_tr, y_tr, X_v, y_v, zero_threshold=threshold)
        elif model_type == 'hurdle_model':
            threshold = hyperparam_profile.hurdle_threshold if hyperparam_profile else 0.5
            return lambda X_tr, y_tr, X_v, y_v: train_hurdle_model(X_tr, y_tr, X_v, y_v, hurdle_threshold=threshold)
        elif model_type == 'tweedie':
            return lambda X_tr, y_tr, X_v, y_v: train_tweedie(X_tr, y_tr, X_v, y_v)
        elif model_type in UNIVARIATE_MODELS:
            # Univariate models only need y, not X
            univariate_fn = TRAINING_REGISTRY.get(model_type)
            if univariate_fn:
                return lambda X_tr, y_tr, X_v, y_v: univariate_fn(y_tr, y_v)
            return lambda X_tr, y_tr, X_v, y_v: train_lightgbm(
                X_tr, y_tr, X_v, y_v, params=final_params, early_stopping_rounds=early_stopping_rounds
            )
        else:
            return lambda X_tr, y_tr, X_v, y_v: train_lightgbm(
                X_tr, y_tr, X_v, y_v, params=final_params, early_stopping_rounds=early_stopping_rounds
            )

    # Default training functions (without loss-specific params)
    TRAINING_FUNCTIONS = {
        'lightgbm': lambda X_tr, y_tr, X_v, y_v: train_lightgbm(X_tr, y_tr, X_v, y_v),
        'xgboost': lambda X_tr, y_tr, X_v, y_v: train_xgboost(X_tr, y_tr, X_v, y_v),
        'catboost': lambda X_tr, y_tr, X_v, y_v: train_catboost(X_tr, y_tr, X_v, y_v),
        'random_forest': lambda X_tr, y_tr, X_v, y_v: train_random_forest(X_tr, y_tr, X_v, y_v),
        'zero_inflated': lambda X_tr, y_tr, X_v, y_v: train_zero_inflated(X_tr, y_tr, X_v, y_v),
        'hurdle_model': lambda X_tr, y_tr, X_v, y_v: train_hurdle_model(X_tr, y_tr, X_v, y_v),
        'tweedie': lambda X_tr, y_tr, X_v, y_v: train_tweedie(X_tr, y_tr, X_v, y_v),
        # Discrete/Ordinal demand models
        'ordinal_regression': lambda X_tr, y_tr, X_v, y_v: train_ordinal_regression(X_tr, y_tr, X_v, y_v),
        'discrete_classifier': lambda X_tr, y_tr, X_v, y_v: train_discrete_classifier(X_tr, y_tr, X_v, y_v),
        'hybrid_discrete': lambda X_tr, y_tr, X_v, y_v: train_hybrid_discrete(X_tr, y_tr, X_v, y_v),
    }

    # Intermittent demand specialists — available for all time formats.
    # Croston/SBA produce flat-rate forecasts repeated across all horizons,
    # which is appropriate for highly intermittent/lumpy individual keys.
    TRAINING_FUNCTIONS.update({
        'croston': lambda X_tr, y_tr, X_v, y_v: train_croston(y_tr, y_v),
        'sba': lambda X_tr, y_tr, X_v, y_v: train_sba(y_tr, y_v),
    })

    # Phase 3/7/8: Add hierarchical, enhanced, and combination models from TRAINING_REGISTRY
    for model_name in ['global_local', 'mixed_effects', 'multi_level_ensemble',
                        'catboost_embedding', 'quantile_regression', 'conformal_boost',
                        'stacked_ensemble']:
        if model_name in TRAINING_REGISTRY and model_name not in TRAINING_FUNCTIONS:
            TRAINING_FUNCTIONS[model_name] = TRAINING_REGISTRY[model_name]

    # Multi-horizon models
    for model_name in ['multi_horizon_lightgbm', 'multi_horizon_xgboost', 'multi_horizon_ensemble']:
        if model_name in TRAINING_REGISTRY and model_name not in TRAINING_FUNCTIONS:
            TRAINING_FUNCTIONS[model_name] = TRAINING_REGISTRY[model_name]

    # For monthly data, add univariate models — they work well with limited observations
    if is_monthly:
        TRAINING_FUNCTIONS.update({
            'croston': lambda X_tr, y_tr, X_v, y_v: train_croston(y_tr, y_v),
            'sba': lambda X_tr, y_tr, X_v, y_v: train_sba(y_tr, y_v),
            'tsb': lambda X_tr, y_tr, X_v, y_v: train_tsb(y_tr, y_v),
            'imapa': lambda X_tr, y_tr, X_v, y_v: train_imapa(y_tr, y_v),
            'arima': lambda X_tr, y_tr, X_v, y_v: train_arima(y_tr, y_v),
            'sarima': lambda X_tr, y_tr, X_v, y_v: train_sarima(y_tr, y_v, m=12),
            'ets': lambda X_tr, y_tr, X_v, y_v: train_ets(y_tr, y_v, seasonal_periods=12),
            'theta': lambda X_tr, y_tr, X_v, y_v: train_theta(y_tr, y_v),
        })
        logger.info(f"Monthly mode: enabled {len(TRAINING_FUNCTIONS) - 11} univariate models in TRAINING_FUNCTIONS")

    # =====================================================================
    # Phase-3 hierarchical models: resolve the hierarchy column ONCE here
    # from the artifact that ``run_segmentation.py`` (or the deterministic
    # pipeline's segmentation stage) writes to ``seg_output/hierarchy_
    # detection.json``. We read from the artifact rather than re-running
    # the detector so every stage sees the exact same answer.
    # =====================================================================
    _hierarchy_col: Optional[str] = None
    try:
        from utils.hierarchy_resolution import HierarchyResolution, ARTIFACT_FILENAME
        _model_dir_norm = model_dir.rstrip('/')
        _seg_dir = os.path.join(os.path.dirname(_model_dir_norm), 'seg_output')
        _art_path = os.path.join(_seg_dir, ARTIFACT_FILENAME)
        if os.path.exists(_art_path):
            import json as _json
            with open(_art_path) as _f:
                _hres = HierarchyResolution.from_dict(_json.load(_f))
            _hierarchy_col = _hres.primary_product_col
            logger.info(
                "Training: resolved hierarchy_col=%r from %s — Phase-3 "
                "hierarchical models will have access to %d distinct groups "
                "at full-catalogue scale.",
                _hierarchy_col, _art_path,
                len(_hres.product) + len(_hres.customer),
            )
        else:
            logger.info(
                "Training: no hierarchy artifact at %s — Phase-3 hierarchical "
                "models (multi_level_ensemble, global_local, mixed_effects) "
                "will be skipped gracefully.", _art_path,
            )
    except Exception as _hx:
        logger.warning(
            "Training: failed to load hierarchy artifact (%s) — hierarchical "
            "models will be skipped.", _hx,
        )

    try:
        # Load strategy - CRITICAL for determining what to train
        if strategy_path:
            if not os.path.exists(strategy_path):
                error_msg = (
                    f"CRITICAL: Training strategy file not found at {strategy_path}\n"
                    "This file should have been created by the Training Planner task.\n"
                    "Please check that Task 1 (Planner) executed successfully."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            strategy = load_json(strategy_path)
            printer.print(f"Loaded training strategy")
        else:
            strategy = {'model_groups': {}}
            printer.print("Using default training strategy")

        model_groups_config = strategy.get('model_groups', {})

        # Load manifest with validation
        manifest_path = os.path.join(feature_dir, 'training_manifest.csv')
        manifest = load_csv(manifest_path)

        # VALIDATION: Check manifest is not empty
        if manifest is None or len(manifest) == 0:
            raise ValueError(f"Training manifest is empty or could not be loaded from {manifest_path}")

        # Determine model level column
        if 'model_level' in manifest.columns:
            ml_col = 'model_level'
        elif 'model_group' in manifest.columns:
            ml_col = 'model_group'
        else:
            raise ValueError("Manifest must have 'model_level' or 'model_group' column")

        # VALIDATION: Check model groups exist
        model_groups = manifest[ml_col].unique()
        if len(model_groups) == 0:
            raise ValueError(f"No model groups found in manifest column '{ml_col}'")

        printer.print(f"Training {len(model_groups)} model levels...")

        # Load feature data (train, val, and test) — format-agnostic:
        # parquet preferred, CSV fallback for legacy artifacts.
        from utils.feature_io import (
            features_intermediate_exists, read_features_intermediate,
        )
        train_df_all = read_features_intermediate(feature_dir, 'train_features')
        val_df_all   = read_features_intermediate(feature_dir, 'val_features')

        # Try to load test data (may not exist)
        test_df_all = None
        if features_intermediate_exists(feature_dir, 'test_features'):
            test_df_all = read_features_intermediate(feature_dir, 'test_features')
            printer.print(f"Loaded train: {len(train_df_all)}, val: {len(val_df_all)}, test: {len(test_df_all)} rows")
        else:
            printer.print(f"Loaded train: {len(train_df_all)}, val: {len(val_df_all)} rows (no test data)")

        # ── Effective recursive-validation flag ──────────────────────
        # When DMH is the primary inference path, skip the recursive
        # walk-forward inside the per-MG candidate eval — see the
        # `use_direct_multi_horizon` parameter docstring above for the
        # full rationale.  This single line is responsible for the
        # ~13x training speedup when DMH is on.
        _effective_recursive_validation = (
            use_recursive_validation and not use_direct_multi_horizon
        )
        if use_recursive_validation and use_direct_multi_horizon:
            printer.print(
                f"Forecast horizon: {forecast_horizon} steps, "
                f"Recursive validation: AUTO-SKIPPED "
                f"(use_direct_multi_horizon=True; "
                f"per-MG candidates will be selected on direct 1-step WAPE — "
                f"DMH supersedes the multi-step path at inference, so "
                f"the 13x walk-forward cost adds no information)"
            )
        else:
            printer.print(
                f"Forecast horizon: {forecast_horizon} steps, "
                f"Recursive validation: {_effective_recursive_validation}"
            )

        # ── Effective walk-forward-CV flag ────────────────────────────
        # Walk-Forward CV trains `walk_forward_n_folds` (default 5) extra
        # LightGBM/XGBoost models PER candidate PER MG to pick the most
        # robust model.  When DMH is the inference path, the per-MG
        # CV-selected model is never deployed — DMH retrains all
        # horizon heads from scratch on the combined train+val panel at
        # inference time.  The CV picks are only consumed for diagnostic
        # logs and the recursive-fallback path that activates if DMH
        # crashes.  Spending ~30 LightGBM fits per MG to inform a
        # diagnostic is enormous waste at fleet scale; this flag
        # auto-skips it when DMH is on (mirroring the recursive-eval
        # auto-skip above).  The skip is a ~5x training speedup on its
        # own — combined with a `max_candidates_per_group` cap below
        # the per-MG fit count drops from ~38 to ~3.
        #
        # If DMH is OFF (legacy recursive inference), WF-CV stays
        # enabled because the per-MG CV-selected model IS what runs at
        # inference, and CV-selection vs single-split selection is a
        # real accuracy difference.
        _effective_walk_forward_cv = (
            enable_walk_forward_cv and not use_direct_multi_horizon
        )
        if enable_walk_forward_cv and use_direct_multi_horizon:
            printer.print(
                f"Walk-Forward CV: AUTO-SKIPPED "
                f"(use_direct_multi_horizon=True; "
                f"per-MG CV-selected model is never deployed when DMH is "
                f"the inference path — DMH retrains all horizon heads "
                f"from scratch.  Spending ~{walk_forward_n_folds} folds × "
                f"~6 candidates per MG to inform a diagnostic log line is "
                f"~5x training speedup foregone)"
            )
        else:
            printer.print(
                f"Walk-Forward CV: {_effective_walk_forward_cv}"
            )

        # Ensure model_level column exists in feature data
        if ml_col not in train_df_all.columns:
            if 'key' in manifest.columns and 'key' in train_df_all.columns:
                train_df_all = train_df_all.merge(
                    manifest[['key', ml_col]].drop_duplicates(), on='key', how='left'
                )
                val_df_all = val_df_all.merge(
                    manifest[['key', ml_col]].drop_duplicates(), on='key', how='left'
                )
                if test_df_all is not None:
                    test_df_all = test_df_all.merge(
                        manifest[['key', ml_col]].drop_duplicates(), on='key', how='left'
                    )

                # VALIDATION: Check for keys that didn't match in manifest (NaN values after left merge)
                train_unmatched = train_df_all[train_df_all[ml_col].isna()]['key'].nunique() if 'key' in train_df_all.columns else 0
                if train_unmatched > 0:
                    logger.warning(f"Found {train_unmatched} keys in train data not present in manifest. These will be skipped.")
                    train_df_all = train_df_all[train_df_all[ml_col].notna()]

                val_unmatched = val_df_all[val_df_all[ml_col].isna()]['key'].nunique() if 'key' in val_df_all.columns else 0
                if val_unmatched > 0:
                    logger.warning(f"Found {val_unmatched} keys in validation data not present in manifest. These will be skipped.")
                    val_df_all = val_df_all[val_df_all[ml_col].notna()]

                if test_df_all is not None:
                    test_unmatched = test_df_all[test_df_all[ml_col].isna()]['key'].nunique() if 'key' in test_df_all.columns else 0
                    if test_unmatched > 0:
                        logger.warning(f"Found {test_unmatched} keys in test data not present in manifest. These will be skipped.")
                        test_df_all = test_df_all[test_df_all[ml_col].notna()]

        # Columns to exclude from features
        # CRITICAL: Also exclude {target_col}_log which is the log-transformed target
        # This column contains the same information as the target and causes severe leakage!
        exclude_cols = {
            target_col, key_col, date_col, 'split', 'model_level', 'model_group',
            'segment_id', 'intermittency_class', 'demand_pattern', 'label',
            f'{target_col}_log',  # Log-transformed target - SEVERE LEAKAGE if included!
        }

        # Track results
        model_specs = []
        total_val_actuals = 0
        total_val_errors = 0
        total_test_actuals = 0
        total_test_errors = 0
        models_trained = 0
        models_failed = 0
        all_candidate_comparisons = {}  # Store all candidate results for analysis

        # =====================================================================
        # STATE-OF-THE-ART: COMPUTE SEGMENT-SPECIFIC HYPERPARAMETER PROFILES
        # =====================================================================
        # Pre-compute hyperparameter profiles for all segments using EDA + Segmentation context
        segment_hyperparam_profiles: Dict[str, HyperparameterProfile] = {}

        # Extract EDA insights from strategy (passed from Feature crew)
        eda_insights_for_hyperparams = strategy.get('eda_driven_config', {})
        if not eda_insights_for_hyperparams:
            eda_insights_for_hyperparams = strategy.get('eda_insights_for_training', {})

        # Convert to format expected by get_segment_specific_hyperparams
        eda_for_hyperparams = {
            'seasonality': {
                'avg_strength': eda_insights_for_hyperparams.get('seasonal_period_detected', 0) > 0 and 0.5 or 0.0,
                'dominant_period': eda_insights_for_hyperparams.get('seasonal_period_detected', 12 if time_format == 'year_month' else 52),
            },
            'trend': {
                'avg_strength': eda_insights_for_hyperparams.get('include_trend', False) and 0.3 or 0.0,
            },
            'changepoints': {
                'pct_significant': eda_insights_for_hyperparams.get('include_changepoints', False) and 0.3 or 0.0,
            },
        }

        # Compute profiles for each model group
        for mg_id, mg_config_item in model_groups_config.items():
            demand_pattern = mg_config_item.get('demand_pattern', 'smooth')
            expected_difficulty = mg_config_item.get('expected_difficulty', 'medium')

            # Build segment profile from config
            segment_profile = {
                'segment_id': mg_id,
                'demand_pattern': demand_pattern,
                'zero_fraction': 0.3 if demand_pattern in ['intermittent', 'lumpy'] else 0.1,
                'cv': 1.5 if demand_pattern in ['erratic', 'lumpy'] else 0.8,
                'expected_difficulty': expected_difficulty,
                'n_keys': 10,  # Default
            }

            # Override with actual segment stats if available in strategy
            segment_stats = mg_config_item.get('segment_stats', {})
            if segment_stats:
                segment_profile.update({
                    'zero_fraction': segment_stats.get('zero_fraction', segment_profile['zero_fraction']),
                    'cv': segment_stats.get('cv', segment_stats.get('coefficient_of_variation', segment_profile['cv'])),
                    'n_keys': segment_stats.get('n_keys', segment_profile['n_keys']),
                })

            try:
                segment_hyperparam_profiles[mg_id] = get_segment_specific_hyperparams(
                    segment_profile=segment_profile,
                    eda_insights=eda_for_hyperparams,
                    demand_pattern=demand_pattern,
                    n_features=n_features or 50,
                    series_length=segment_stats.get('avg_series_length', 100),
                    time_format=time_format,
                )
                logger.debug(f"Computed hyperparams for {mg_id}: {segment_hyperparam_profiles[mg_id].derivation_reason}")
            except Exception as hp_err:
                logger.warning(f"Failed to compute hyperparams for {mg_id}: {hp_err}")
                # Will fall back to default params

        if segment_hyperparam_profiles:
            printer.print(f"STATE-OF-THE-ART: Computed segment-specific hyperparameters for {len(segment_hyperparam_profiles)} segments")

        # Track segment calibrations for post-training bias correction
        segment_calibration_data: Dict[str, Dict[str, np.ndarray]] = {}

        # ============================================================
        # PARALLEL DISPATCH SETUP — Spark > Threads > Sequential
        # ============================================================
        # The per-model-group training loop is dispatched via one of
        # three backends:
        #
        #   1. **Spark `mapInPandas`** (preferred when a SparkSession
        #      is active — i.e. running on Databricks).  Big DataFrames
        #      are broadcast to all executors ONCE.  Each task picks
        #      one model group, runs the trainer, and returns a pickled
        #      result.  Spark schedules tasks across all executor cores
        #      so work fans out across the whole cluster (driver +
        #      every worker node), not just the driver.
        #
        #   2. **ThreadPoolExecutor** (fallback when no SparkSession).
        #      LightGBM / XGBoost release the GIL during fit, so threads
        #      give real parallelism on a single driver.  No process
        #      isolation, but no pickling overhead either.
        #
        #   3. **Sequential** (parallel_training_workers <= 1).
        #      Identical behaviour to the legacy code path.
        #
        # Each `_run_one_mg(mg)` call is INDEPENDENT: it accumulates
        # ALL its mutations into a local `_local_result` dict (no
        # nonlocal, no shared state, no locks) and returns the dict.
        # The dispatcher merges results back into the outer-scope
        # counters/lists/dicts in the main thread after all workers
        # finish.  This works identically for sequential, threads,
        # and Spark — Spark's process boundary just means we couldn't
        # mutate outer scope from a worker even if we wanted to.
        #
        # Why Spark over Ray on Databricks:
        #   * Spark UDFs run in the notebook-scoped Python env, so a
        #     single `%pip install <wheel>` in the notebook is enough
        #     to ship our package to executors — no cluster init
        #     script required.  Ray workers run in the cluster-level
        #     system Python and need init-script wheel installs to
        #     reach.
        #   * Spark is the native execution layer on Databricks; no
        #     extra runtime to attach to, no `runtime_env` URI quirks.
        #   * For multi-second-or-longer training tasks, Spark task
        #     overhead (~1-2s) is negligible vs Ray's ~10ms.
        # ============================================================
        import os as _os_pll

        # CRITICAL: suppress the libomp/libgomp double-load abort.  The
        # user's stack imports both TensorFlow (libomp via oneDNN) and
        # LightGBM/XGBoost (libgomp).  When BOTH are loaded in one
        # process and any concurrent OMP work happens, glibc aborts
        # with SIGSEGV.  This env var tells libomp to coexist with
        # libgomp instead of dying.  Set BEFORE the fits run.
        _os_pll.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        # CLOSED-STREAM HARDENING.
        #
        # In a Databricks notebook spawned via `dbutils.notebook.run`,
        # ipykernel's OutStream.write raises
        # ``ValueError: I/O operation on closed file`` when its
        # pub_thread is None.  Some libraries (py4j-backed Spark
        # callbacks among them) capture sys.stdout/stderr early and
        # write through that captured reference later — even after
        # we swap sys.stdout/stderr at the Python level, those
        # captured references still point at the broken OutStream.
        #
        # The robust fix is to monkey-patch OutStream.write at the
        # CLASS level so EVERY instance (including the references
        # any library has already captured) tolerates the closed
        # state.  Idempotent — leaves a no-op if already patched.
        try:
            from ipykernel.iostream import OutStream as _IpyOutStream
            if not getattr(_IpyOutStream, "_diq_safe_patched", False):
                _orig_write = _IpyOutStream.write

                def _safe_write(self, string):
                    try:
                        return _orig_write(self, string)
                    except ValueError as _exc:
                        if "closed file" in str(_exc):
                            return 0  # silent drop, was the bug
                        raise
                    except Exception:
                        return 0

                _IpyOutStream.write = _safe_write
                _IpyOutStream._diq_safe_patched = True
        except Exception:
            # ipykernel may not be present (running under something
            # other than a Databricks/Jupyter notebook) — that's fine,
            # the error this guards against can't happen there.
            pass

        _ptw = max(1, int(parallel_training_workers))
        _orig_omp = _os_pll.environ.get("OMP_NUM_THREADS")

        # Try Spark first.  On Databricks, a SparkSession is always
        # active in the notebook process — `getActiveSession()` returns
        # it without spinning up a new one.  Off Databricks (local
        # dev / CI), it returns None and we fall through to threads
        # or sequential.
        #
        # IMPORTANT: when the user has TF + LightGBM/XGBoost in the
        # same env (typical Databricks ML runtime), threads ARE NOT
        # SAFE — the libomp/libgomp combination crashes with SIGSEGV
        # under any concurrent OMP work.  We previously fell back to
        # ThreadPool capped at 8; user reported it STILL segfaulted.
        # Sequential is the only safe fallback.  Speed is reduced to
        # legacy levels but at least the run doesn't crash.
        _spark_available = False
        _spark_session = None
        if _ptw > 1:
            try:
                from pyspark.sql import SparkSession as _SparkSession
                _spark_session = _SparkSession.getActiveSession()
                if _spark_session is None:
                    # Off-Databricks fallback: try to create one.  If
                    # this works we're probably in a local Spark dev
                    # env.  If it doesn't, fall through to threads.
                    try:
                        _spark_session = _SparkSession.builder.getOrCreate()
                    except Exception:
                        _spark_session = None
                if _spark_session is not None:
                    _spark_available = True

                    # ════════════════════════════════════════════════
                    # Probe cluster size (worker_count × executor cores)
                    # so we can size per-task cores against the whole
                    # cluster, not the driver's local cpu_count.
                    #
                    # The naive approach (`getExecutorMemoryStatus()`)
                    # FAILS on freshly-started job clusters: executors
                    # don't register with the driver until the first
                    # Spark action runs, so right after SparkSession
                    # creation the API returns just the driver.  We
                    # try multiple methods in priority order:
                    #
                    #   1. Databricks cluster tags
                    #      (`clusterAllNumberOfWorkers`) — published
                    #      at SparkSession start, doesn't require
                    #      executors to have registered yet.  THIS IS
                    #      THE RELIABLE ONE on Databricks.
                    #   2. spark.executor.instances × spark.executor.cores
                    #   3. defaultParallelism (works once executors
                    #      register, which is unreliable pre-action).
                    #   4. getExecutorMemoryStatus enumeration (same
                    #      registration dependency).
                    #
                    # The last action probe is also useful as a
                    # cross-check log line.
                    # ════════════════════════════════════════════════
                    _sc = _spark_session.sparkContext
                    _total_cores = 0
                    _executor_count = 0
                    _sizing_method = "driver-fallback"

                    def _conf_int(key, default=0):
                        try:
                            return int(_spark_session.conf.get(key, str(default)))
                        except Exception:
                            return default

                    _exec_cores = _conf_int("spark.executor.cores", 0)

                    # Method 1: Databricks cluster tag (most reliable
                    # before any Spark action has run).
                    _db_workers = _conf_int(
                        "spark.databricks.clusterUsageTags.clusterAllNumberOfWorkers",
                        0,
                    )
                    if _db_workers > 0 and _exec_cores > 0:
                        _executor_count = _db_workers
                        _total_cores = _db_workers * _exec_cores
                        _sizing_method = "databricks cluster tag"

                    # Method 2: spark.executor.instances (set by
                    # Databricks for fixed-size clusters).
                    if _total_cores <= 0:
                        _exec_instances = _conf_int(
                            "spark.executor.instances", 0
                        )
                        if _exec_instances > 0 and _exec_cores > 0:
                            _executor_count = _exec_instances
                            _total_cores = _exec_instances * _exec_cores
                            _sizing_method = "spark.executor.instances"

                    # Method 3: defaultParallelism — total task slots
                    # the scheduler thinks it has.  Works when method
                    # 1+2 didn't (e.g. some autoscaling configs).
                    if _total_cores <= 0:
                        try:
                            _total_cores = int(_sc.defaultParallelism)
                            if _total_cores > 0:
                                _sizing_method = "defaultParallelism"
                        except Exception:
                            pass

                    # Method 4: enumerate registered executors
                    # (last-ditch — only works if any prior code path
                    # forced a Spark action so executors are up).
                    if _total_cores <= 0:
                        try:
                            _executors = list(
                                _sc.getExecutorMemoryStatus().keys()
                            )
                            _executor_count = max(0, len(_executors) - 1)
                            if _executor_count > 0 and _exec_cores > 0:
                                _total_cores = _executor_count * _exec_cores
                                _sizing_method = "getExecutorMemoryStatus"
                        except Exception:
                            pass

                    printer.print(
                        f"  Spark cluster: {_executor_count} executors, "
                        f"~{_total_cores} cores total "
                        f"(detected via {_sizing_method})"
                    )

                    # Size per-task cores from CLUSTER total, not driver.
                    # If every probe method failed, fall back to the
                    # driver's cpu_count and warn loudly — sub-optimal
                    # but won't crash.
                    if _total_cores > 0:
                        _cores_for_sizing = _total_cores
                    else:
                        _cores_for_sizing = _os_pll.cpu_count() or 16
                        printer.print(
                            f"  WARNING: could not detect cluster size; "
                            f"falling back to driver cpu_count={_cores_for_sizing}. "
                            f"This will under-utilise the cluster.  Workaround: "
                            f"set `parallel_training_workers` >= cluster_cores in "
                            f"your config to force per_task=1 across enough "
                            f"partitions to fill the cluster."
                        )
                    _per_task_cores = max(1, _cores_for_sizing // _ptw)
                    printer.print(
                        f"  PARALLEL TRAINING (Spark): {_ptw} tasks x "
                        f"{_per_task_cores} cores/task "
                        f"(sizing against {_cores_for_sizing} cores)"
                    )
            except ImportError:
                printer.print("  PySpark not installed")
            except Exception as _spark_exc:
                printer.print(
                    f"  Spark session check failed: "
                    f"{type(_spark_exc).__name__}: {_spark_exc}"
                )
                _spark_available = False

        if _ptw > 1 and not _spark_available:
            # FALLBACK: SEQUENTIAL.  Threads were tried in earlier
            # commits but produced SIGSEGV even at the conservative
            # 8-worker cap.  Root cause: TensorFlow (libomp via oneDNN)
            # and LightGBM/XGBoost (libgomp) coexist in the user's
            # environment, and any concurrent OMP work between them
            # crashes glibc.  Even setting KMP_DUPLICATE_LIB_OK (which
            # we now do above) only suppresses the warning; the
            # underlying race remains.
            #
            # Process isolation (Spark on a cluster) is the ONLY safe
            # parallel option when TF + LightGBM/XGBoost share an
            # environment.  If Spark is unreachable, we run
            # sequentially.  Slower but doesn't crash.
            printer.print(
                f"  WARNING: parallel_training_workers={_ptw} but "
                "Spark is unavailable.  Falling back to SEQUENTIAL "
                "training instead of ThreadPool — concurrent threads "
                "+ LightGBM/TF combination has been observed to "
                "SIGSEGV on Databricks ML runtime.\n"
                "  Spark should be available automatically on "
                "Databricks.  If you're seeing this message there, "
                "check that the notebook is attached to a running "
                "cluster."
            )
            _ptw = 1  # Force sequential dispatch path below.

        def _run_one_mg(mg, train_df_all, val_df_all, test_df_all):
            # CRITICAL: train/val/test_df_all are PARAMETERS (not closure
            # captures) so cloudpickle of this function doesn't include
            # the ~2-3 GB of DataFrames.  Spark serialises closure
            # captures with each task; capturing the frames via closure
            # would push the per-task payload to 2.5+ GiB.  By making
            # them parameters, the function body's references to
            # `train_df_all` etc. resolve to LOCAL parameter names
            # instead of enclosing-scope free variables, so cloudpickle
            # sees a tiny function.
            #
            # The Spark dispatcher uses sparkContext.broadcast() to
            # publish each frame to every executor ONCE and passes the
            # broadcast `.value` as the actual argument from the
            # mapInPandas dispatch closure.  The sequential path passes
            # the frames directly.  Either way the function itself
            # never carries the data.
            #
            # NO `nonlocal` — every accumulator is shadowed by a fresh
            # local at the top of the function so the body's
            # `models_trained += 1` etc. write to the local copy.  The
            # closure returns these locals as a `_local_result` dict;
            # the dispatcher aggregates them back into the outer-scope
            # state in the main thread / main process.
            #
            # This pattern works identically for sequential, threads,
            # and Spark — there's no shared mutable state to worry
            # about crossing process boundaries.
            models_trained = 0
            models_failed = 0
            total_val_actuals = 0.0
            total_val_errors = 0.0
            total_test_actuals = 0.0
            total_test_errors = 0.0
            model_specs = []
            all_candidate_comparisons = {}
            segment_calibration_data = {}

            mg_str = str(mg)
            mg_start = time.time()

            try:
                # Get config for this group
                mg_config = model_groups_config.get(mg_str, {})
                demand_pattern = mg_config.get('demand_pattern', 'smooth')

                # STATE-OF-THE-ART: Extract per-segment loss function and validation strategy
                recommended_loss = mg_config.get('recommended_loss', 'mse')
                validation_strategy = mg_config.get('validation_strategy', 'time_series_split')
                expected_difficulty = mg_config.get('expected_difficulty', 'medium')

                # Log the state-of-the-art config being used
                logger.info(f"Model {mg}: pattern={demand_pattern}, loss={recommended_loss}, validation={validation_strategy}")

                # Filter data for this model level
                if ml_col in train_df_all.columns:
                    train_df = train_df_all[train_df_all[ml_col] == mg].copy()
                    val_df = val_df_all[val_df_all[ml_col] == mg].copy()
                else:
                    train_df = train_df_all.copy()
                    val_df = val_df_all.copy()

                if len(train_df) == 0 or len(val_df) == 0:
                    raise ValueError(f"No data for model level {mg}")

                # Get feature columns - exclude metadata columns
                # CRITICAL: Only use columns that exist in BOTH train_df AND val_df
                # This ensures consistency when one-hot encoded columns might differ
                train_feature_cols = set(c for c in train_df.columns if c not in exclude_cols)
                val_feature_cols = set(c for c in val_df.columns if c not in exclude_cols)
                common_feature_cols = train_feature_cols.intersection(val_feature_cols)

                # Log any column mismatches for debugging
                train_only = train_feature_cols - val_feature_cols
                val_only = val_feature_cols - train_feature_cols
                if train_only or val_only:
                    logger.warning(f"Model level {mg}: Column mismatch detected!")
                    if train_only:
                        logger.warning(f"  Train-only columns ({len(train_only)}): {list(train_only)[:5]}...")
                    if val_only:
                        logger.warning(f"  Val-only columns ({len(val_only)}): {list(val_only)[:5]}...")

                feature_cols = [c for c in train_df.columns if c in common_feature_cols]

                # Filter to only numeric columns (feature engineering should have validated types,
                # but we still filter here as a safety net for any metadata columns that slipped through)
                # CRITICAL: Check numeric dtype in BOTH train_df AND val_df to avoid type mismatches
                # (e.g., a column might be float64 in train but object in val due to different values)
                numeric_cols = [
                    c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(val_df[c])
                ]
                non_numeric_dropped = len(feature_cols) - len(numeric_cols)
                if non_numeric_dropped > 0:
                    logger.debug(f"Model level {mg}: Filtered out {non_numeric_dropped} non-numeric columns")
                feature_cols = numeric_cols

                # Drop columns with >50% NaN in training data
                nan_threshold = 0.5
                valid_cols = []
                high_nan_count = 0
                for col in feature_cols:
                    nan_pct = train_df[col].isna().mean()
                    if nan_pct <= nan_threshold:
                        valid_cols.append(col)
                    else:
                        high_nan_count += 1
                        logger.debug(f"Dropping column {col} with {nan_pct*100:.1f}% NaN")

                feature_cols = valid_cols

                if high_nan_count > 0:
                    logger.info(f"Model level {mg}: Dropped {high_nan_count} columns with >50% NaN")

                if len(feature_cols) == 0:
                    raise ValueError(f"No valid numeric features for model level {mg}")

                # =====================================================================
                # MODEL-LEVEL-AWARE FEATURE SELECTION
                # Individual models use different features than segment models:
                # - Individual: Focus on key's own patterns (lags, rolling stats)
                # - Segment: Include cross-key features and key discriminators
                # =====================================================================
                n_features_before_selection = len(feature_cols)

                # Determine model strategy from manifest
                # Look for model_strategy column or infer from model_level pattern
                model_strategy = 'segment_pooled'  # Default to segment
                n_keys_in_group = len(train_df[key_col].unique()) if key_col in train_df.columns else 1

                if 'model_strategy' in manifest.columns:
                    # Get dominant strategy for this model group
                    mg_manifest = manifest[manifest[ml_col] == mg]
                    if len(mg_manifest) > 0:
                        # Safely get mode value with proper null handling
                        strategy_values = mg_manifest['model_strategy'].dropna()
                        if len(strategy_values) > 0:
                            mode_result = strategy_values.mode()
                            model_strategy = mode_result.iloc[0] if len(mode_result) > 0 else 'segment_pooled'
                        else:
                            model_strategy = 'segment_pooled'
                elif mg_str.startswith('key_') or n_keys_in_group == 1:
                    # Infer individual model from naming convention or single key
                    model_strategy = 'individual_score'

                # Apply model-level-aware feature selection
                # NOTE: demand_pattern is available from mg_config (extracted earlier)
                try:
                    selected_features, selection_metadata = select_features_for_model_level(
                        all_features=feature_cols,
                        model_strategy=model_strategy,
                        n_keys_in_group=n_keys_in_group,
                        key_col=key_col,
                        demand_pattern=demand_pattern,  # PATTERN-AWARE: Adjusts feature priorities
                        verbose=False,  # Don't spam logs
                    )

                    # Only use selection if it's meaningful (selected at least some features)
                    if len(selected_features) >= 5:
                        feature_cols = selected_features
                        is_individual = selection_metadata.get('is_individual', False)
                        model_type_desc = 'individual key' if is_individual else 'segment pooled'
                        n_excluded = n_features_before_selection - len(feature_cols)
                        if n_excluded > 0:
                            logger.info(
                                f"Model level {mg} ({model_type_desc}): Feature selection "
                                f"{n_features_before_selection} → {len(feature_cols)} features "
                                f"(-{n_excluded} {'cross-key' if is_individual else 'key-specific'})"
                            )
                except Exception as feat_sel_err:
                    logger.warning(f"Model level {mg}: Feature selection failed ({feat_sel_err}), using all features")

                # =====================================================================
                # MINIMUM SAMPLE SIZE CHECK - Most ML models require at least 2 samples
                # =====================================================================
                MIN_TRAIN_SAMPLES = 2
                if len(train_df) < MIN_TRAIN_SAMPLES:
                    logger.warning(f"Model level {mg}: Only {len(train_df)} training samples, need at least {MIN_TRAIN_SAMPLES}. Using naive fallback.")
                    # Use naive model for very small datasets
                    y_train_mean = train_df[target_col].mean() if len(train_df) > 0 else 0.0
                    model_specs.append({
                        'model_group': mg_str,
                        'model_type': 'naive_mean',
                        'val_wape': 1.0,
                        'val_wape_direct': 1.0,
                        'val_wape_recursive': 1.0,
                        'test_wape': None,
                        'train_wape': 0.0,
                        'n_train_samples': len(train_df),
                        'n_val_samples': len(val_df),
                        'n_features': len(feature_cols),
                        'feature_columns': feature_cols,
                        'naive_prediction': float(y_train_mean),
                        'status': 'fallback',
                        'reason': f'Too few training samples ({len(train_df)} < {MIN_TRAIN_SAMPLES})',
                    })
                    models_trained += 1
                    # Return now — the dispatcher will aggregate deltas.
                    return {
                        'models_trained': models_trained, 'models_failed': models_failed,
                        'total_val_actuals': total_val_actuals, 'total_val_errors': total_val_errors,
                        'total_test_actuals': total_test_actuals, 'total_test_errors': total_test_errors,
                        'model_specs': model_specs,
                        'all_candidate_comparisons': all_candidate_comparisons,
                        'segment_calibration_data': segment_calibration_data,
                        'mg_str': mg_str,
                    }

                # =====================================================================
                # DATA SANITIZATION - Handle inf, -inf, and very large values
                # =====================================================================
                # Replace infinity with NaN first, then fill NaN with 0
                # Also clip very large values that can cause float32 overflow
                MAX_FLOAT32 = 3.4e38  # Max float32 value

                train_features = train_df[feature_cols].copy()
                val_features = val_df[feature_cols].copy()

                # Replace inf/-inf with NaN
                train_features = train_features.replace([np.inf, -np.inf], np.nan)
                val_features = val_features.replace([np.inf, -np.inf], np.nan)

                # Clip extreme values that would overflow float32
                for col in feature_cols:
                    if train_features[col].abs().max() > MAX_FLOAT32:
                        train_features[col] = train_features[col].clip(-MAX_FLOAT32, MAX_FLOAT32)
                    if val_features[col].abs().max() > MAX_FLOAT32:
                        val_features[col] = val_features[col].clip(-MAX_FLOAT32, MAX_FLOAT32)

                # Fill NaN with 0
                X_train = train_features.fillna(0).values.astype(np.float32)
                X_val = val_features.fillna(0).values.astype(np.float32)

                y_train = train_df[target_col].replace([np.inf, -np.inf], np.nan).fillna(0).values
                y_val = val_df[target_col].replace([np.inf, -np.inf], np.nan).fillna(0).values

                # =====================================================================
                # CONSTANT FEATURE CHECK - Remove features with zero variance
                # =====================================================================
                non_constant_mask = np.std(X_train, axis=0) > 1e-10
                if not non_constant_mask.all():
                    n_constant = (~non_constant_mask).sum()
                    logger.debug(f"Model level {mg}: Removing {n_constant} constant features")
                    X_train = X_train[:, non_constant_mask]
                    X_val = X_val[:, non_constant_mask]
                    feature_cols = [c for c, keep in zip(feature_cols, non_constant_mask) if keep]

                if len(feature_cols) == 0:
                    logger.warning(f"Model level {mg}: All features are constant. Using naive fallback.")
                    y_train_mean = y_train.mean() if len(y_train) > 0 else 0.0
                    model_specs.append({
                        'model_group': mg_str,
                        'model_type': 'naive_mean',
                        'val_wape': 1.0,
                        'val_wape_direct': 1.0,
                        'val_wape_recursive': 1.0,
                        'test_wape': None,
                        'train_wape': 0.0,
                        'n_train_samples': len(X_train),
                        'n_val_samples': len(X_val),
                        'n_features': 0,
                        'feature_columns': [],
                        'naive_prediction': float(y_train_mean),
                        'status': 'fallback',
                        'reason': 'All features are constant',
                    })
                    models_trained += 1
                    return {
                        'models_trained': models_trained, 'models_failed': models_failed,
                        'total_val_actuals': total_val_actuals, 'total_val_errors': total_val_errors,
                        'total_test_actuals': total_test_actuals, 'total_test_errors': total_test_errors,
                        'model_specs': model_specs,
                        'all_candidate_comparisons': all_candidate_comparisons,
                        'segment_calibration_data': segment_calibration_data,
                        'mg_str': mg_str,
                    }

                logger.info(f"Model level {mg}: {len(feature_cols)} features, {len(X_train)} train, {len(X_val)} val rows")

                # Select candidate models using dynamic priority based on CV/zero_fraction
                if enable_meta_learning and len(y_train) > 10:
                    try:
                        # Check for external features (price, promo, etc.)
                        external_feature_names = ['price', 'promo', 'discount', 'promotion', 'unit_price']
                        has_external = any(
                            any(ext in col.lower() for ext in external_feature_names)
                            for col in feature_cols
                        )

                        meta_features = build_meta_features(
                            y=y_train,
                            n_features=len(feature_cols),
                            has_external_features=has_external,
                            feature_quality_score=0.7,
                            time_format=time_format,
                        )
                        # Pass y_train for discrete demand detection, time_format for monthly univariate
                        ranked = rank_models_by_pattern_fit(meta_features, y=y_train, time_format=time_format)

                        # Sprint 3 B7: Re-score with enriched segmentation features
                        try:
                            from utils.sprint3_features import score_model_with_enrichment
                            # Extract enrichment info from segmentation context
                            seg_info = segmentation_context.get('segment_profiles', {}).get(mg_str, {})
                            enriched_meta = {
                                'hierarchy_depth': len(segmentation_context.get('hierarchy_cols', [])),
                                'hierarchy_group_size': seg_info.get('size', 0),
                                'external_response_strength': seg_info.get('external_sensitivity_score', 0),
                                'zero_fraction': meta_features.zero_fraction,
                                'regime_volatility': seg_info.get('variability_tier_numeric', 0) / 2.0,
                                'seasonality_shape': seg_info.get('dominant_seasonality', 'unknown'),
                            }
                            ranked = [(name, score_model_with_enrichment(name, enriched_meta, score))
                                      for name, score in ranked]
                            ranked.sort(key=lambda x: -x[1])
                        except Exception:
                            pass  # Fall back to base ranking

                        candidates = [m for m, _ in ranked[:5] if m in TRAINING_FUNCTIONS]
                        if not candidates:
                            candidates = ['lightgbm', 'xgboost', 'catboost']

                        # Phase 3/7/8: Always include enhanced models as additional candidates
                        # These are trained alongside the pattern-selected candidates
                        enhanced_models = ['quantile_regression', 'global_local', 'conformal_boost']
                        for em in enhanced_models:
                            if em in TRAINING_FUNCTIONS and em not in candidates:
                                candidates.append(em)

                        logger.debug(
                            f"Model {mg}: Dynamic selection - pattern={meta_features.demand_pattern}, "
                            f"cv={meta_features.cv:.2f}, zero_frac={meta_features.zero_fraction:.2f}, "
                            f"candidates={candidates[:3]}"
                        )
                    except Exception as e:
                        logger.debug(f"Meta-learning failed: {e}, using pattern priority")
                        candidates = PATTERN_MODEL_PRIORITY.get(
                            demand_pattern, ['lightgbm', 'xgboost', 'catboost']
                        )[:5]
                else:
                    # Non-meta-learning path: still check for discrete demand
                    from utils.model_selection_intelligence import detect_discrete_demand
                    is_disc, n_uniq = detect_discrete_demand(y_train)
                    base_candidates = PATTERN_MODEL_PRIORITY.get(
                        demand_pattern, ['lightgbm', 'xgboost', 'catboost']
                    )[:5]
                    if is_disc and n_uniq <= 25:
                        # Inject discrete models at top based on cardinality
                        if n_uniq <= 5:
                            disc_models = ['discrete_classifier', 'ordinal_regression']
                        elif n_uniq <= 10:
                            disc_models = ['ordinal_regression', 'discrete_classifier']
                        else:
                            disc_models = ['hybrid_discrete', 'ordinal_regression']
                        disc_models = [m for m in disc_models if m in TRAINING_FUNCTIONS]
                        candidates = disc_models + [m for m in base_candidates if m not in disc_models]
                        logger.info(
                            f"Model {mg}: Discrete demand detected ({n_uniq} unique values), "
                            f"injecting discrete models: {disc_models}"
                        )
                    else:
                        candidates = base_candidates

                # Override with config if specified, but preserve discrete detection
                if mg_config.get('candidate_models'):
                    config_candidates = [m for m in mg_config['candidate_models'] if m in TRAINING_FUNCTIONS]
                    if not config_candidates:
                        config_candidates = ['lightgbm']

                    # If we detected discrete demand above but the strategy doesn't
                    # include discrete models, inject them into the strategy's list
                    discrete_model_names = {'discrete_classifier', 'ordinal_regression', 'hybrid_discrete'}
                    has_discrete_in_config = any(m in discrete_model_names for m in config_candidates)
                    has_discrete_in_candidates = any(m in discrete_model_names for m in candidates)

                    if has_discrete_in_candidates and not has_discrete_in_config:
                        # Preserve the discrete models we detected from y_train
                        disc_from_detection = [m for m in candidates if m in discrete_model_names]
                        candidates = disc_from_detection + [m for m in config_candidates if m not in discrete_model_names]
                        logger.info(
                            f"Model {mg}: Strategy override preserved discrete models: {disc_from_detection}"
                        )
                    else:
                        candidates = config_candidates

                # =============================================================
                # RESTRICT SEGMENT (POOLED) MODELS TO TREE-BASED FAMILIES ONLY
                # Two-stage models (zero_inflated, hurdle, tweedie) consistently
                # show WAPE >100% at segment level because they over-complicate
                # pooled prediction across diverse zero-fraction patterns.
                # =============================================================
                TREE_BASED_MODELS = {'lightgbm', 'xgboost', 'catboost', 'random_forest'}
                HARMFUL_SEGMENT_MODELS = {'zero_inflated', 'hurdle_model', 'tweedie'}
                if model_strategy == 'segment_pooled' and n_keys_in_group > 1:
                    filtered = [m for m in candidates if m not in HARMFUL_SEGMENT_MODELS]
                    if filtered:
                        if candidates != filtered:
                            removed = [m for m in candidates if m in HARMFUL_SEGMENT_MODELS]
                            logger.info(
                                f"Model {mg}: Segment-pooled — removed harmful models {removed}, "
                                f"keeping {filtered}"
                            )
                        candidates = filtered
                    else:
                        # All candidates were harmful — fall back to tree-based
                        candidates = ['lightgbm', 'xgboost', 'catboost']
                        logger.info(f"Model {mg}: Segment-pooled — all candidates harmful, using tree-based fallback")

                # =============================================================
                # CROSTON/SBA FALLBACK FOR HIGHLY INTERMITTENT INDIVIDUAL KEYS
                # For keys with zero_fraction > 0.8, inject Croston/SBA as
                # candidates. These produce flat-rate forecasts appropriate for
                # sparse demand where tree-based models may overfit to zeros.
                # =============================================================
                if model_strategy == 'individual_score' and n_keys_in_group == 1:
                    try:
                        # Always compute from current key's data (meta_features could be stale from prior iteration)
                        zf = float((y_train == 0).mean())
                        if zf > 0.8:
                            intermittent_models = ['croston', 'sba']
                            intermittent_models = [m for m in intermittent_models if m in TRAINING_FUNCTIONS and m not in candidates]
                            if intermittent_models:
                                candidates = intermittent_models + candidates
                                logger.info(
                                    f"Model {mg}: Highly intermittent (zero_frac={zf:.2f}), "
                                    f"injecting {intermittent_models}"
                                )
                    except Exception:
                        pass  # Don't fail model selection for this injection

                # =====================================================================
                # CAP CANDIDATES PER `max_candidates_per_group`
                # =====================================================================
                # Historical bug: this parameter was declared, documented, and
                # passed through deterministic_pipeline.py — but never actually
                # APPLIED to the candidates list.  The only `[:3]` slice in this
                # file was in a log-message format string at line ~5350 (display
                # truncation, not list mutation).  Result: users setting
                # `max_candidates_per_group: 3` in YAML were still training ~8
                # candidates per MG (5 ranked + 3 always-on enhanced models
                # appended at line ~5345).
                #
                # Applied AFTER all candidate-list manipulation is complete
                # (meta-learning rank, enhanced-model append, pattern fallback,
                # discrete injection, segment-pooled filter, intermittent
                # prepend) so we cap the FINAL ordered list — preserving the
                # selection priority each branch decided on.
                #
                # When `max_candidates_per_group <= 0` we treat it as "no cap"
                # (legacy unbounded behaviour) so this is a no-op for callers
                # that haven't set it explicitly.
                if max_candidates_per_group and max_candidates_per_group > 0:
                    if len(candidates) > max_candidates_per_group:
                        _dropped = candidates[max_candidates_per_group:]
                        candidates = candidates[:max_candidates_per_group]
                        logger.info(
                            f"Model {mg}: capping candidates to "
                            f"{max_candidates_per_group} (kept {candidates}, "
                            f"dropped {_dropped})"
                        )

                # =====================================================================
                # STATE-OF-THE-ART: WALK-FORWARD CROSS-VALIDATION FOR MODEL SELECTION
                # Provides temporally-aware, robust model selection with drift detection.
                # Uses _effective_walk_forward_cv (computed once near the top of
                # run_full_training_pipeline as
                # `enable_walk_forward_cv AND NOT use_direct_multi_horizon`) so
                # the entire CV block short-circuits when DMH is the inference path.
                # =====================================================================
                walk_forward_cv_result = None
                walk_forward_cv_used = False

                if _effective_walk_forward_cv:
                    try:
                        # Combine train and val for Walk-Forward CV
                        # CV will create its own temporal splits
                        combined_df = pd.concat([train_df, val_df], ignore_index=True)

                        # Check if we have period_idx column, if not create it from date_col
                        if 'period_idx' not in combined_df.columns and date_col in combined_df.columns:
                            combined_df['period_idx'] = create_period_index(
                                combined_df, date_col, time_format=time_format
                            )

                        if 'period_idx' in combined_df.columns:
                            n_unique_periods = combined_df['period_idx'].nunique()

                            # Check if we have enough periods for meaningful CV
                            min_required = walk_forward_min_train_periods + forecast_horizon + walk_forward_n_folds
                            if n_unique_periods >= min_required:
                                logger.info(f"Model {mg}: Running Walk-Forward CV ({walk_forward_n_folds} folds, {n_unique_periods} periods)")

                                # Create Walk-Forward CV instance
                                wf_cv = WalkForwardCV(
                                    n_splits=walk_forward_n_folds,
                                    forecast_horizon=forecast_horizon,
                                    min_train_periods=walk_forward_min_train_periods,
                                    strategy=walk_forward_strategy,
                                    rolling_window=walk_forward_rolling_window,
                                )

                                # Create model function wrapper for CV
                                def create_cv_model_fn(model_type_name, loss_p, hp_profile):
                                    """Create a model function for Walk-Forward CV evaluation."""
                                    def cv_model_fn(X_t, y_t, config):
                                        tfn = create_training_fn(
                                            model_type=model_type_name,
                                            loss_params=loss_p,
                                            hyperparam_profile=hp_profile,
                                            early_stopping_rounds=mg_config.get('recommended_hyperparams', {}).get('early_stopping_rounds', 50),
                                        )
                                        # For CV we need a lightweight eval, create minimal val set
                                        n_val = max(1, len(y_t) // 10)
                                        X_t_train = X_t.iloc[:-n_val] if n_val < len(X_t) else X_t
                                        y_t_train = y_t.iloc[:-n_val] if n_val < len(y_t) else y_t
                                        X_t_val = X_t.iloc[-n_val:] if n_val < len(X_t) else X_t.iloc[-1:]
                                        y_t_val = y_t.iloc[-n_val:] if n_val < len(y_t) else y_t.iloc[-1:]
                                        result = tfn(X_t_train.values, y_t_train.values, X_t_val.values, y_t_val.values)
                                        return result.model if result else None
                                    return cv_model_fn

                                # Prepare feature matrix for CV
                                X_cv = combined_df[feature_cols + ['period_idx']].copy()
                                y_cv = combined_df[target_col].copy()

                                # Build model configs for CV
                                cv_model_configs = []
                                _skipped_hierarchical = []
                                for cand_type in candidates:
                                    if cand_type not in TRAINING_FUNCTIONS:
                                        continue
                                    # Walk-Forward CV splits (X, y) matrices per fold;
                                    # Phase-3 hierarchical models (global_local,
                                    # mixed_effects, multi_level_ensemble) need the
                                    # full DataFrame with key_col + hierarchy_col to
                                    # pool across keys, which doesn't translate to
                                    # matrix-only folds. They're already evaluated in
                                    # the main training loop where the full DataFrames
                                    # are available, so we skip them in CV rather than
                                    # letting them silently return None and break the
                                    # downstream model.predict() call.
                                    if cand_type in _HIERARCHICAL_MODELS:
                                        _skipped_hierarchical.append(cand_type)
                                        continue
                                    cv_loss_params = get_loss_params(recommended_loss, cand_type)
                                    cv_hp_profile = segment_hyperparam_profiles.get(mg_str)
                                    cv_model_configs.append({
                                        'name': cand_type,
                                        'model_type': cand_type,
                                        'loss_params': cv_loss_params,
                                        'hp_profile': cv_hp_profile,
                                    })
                                if _skipped_hierarchical:
                                    logger.info(
                                        "Model %s: WF-CV skipping %d hierarchical model(s) %s "
                                        "(evaluated only in main training loop where DataFrames are available)",
                                        mg, len(_skipped_hierarchical), _skipped_hierarchical,
                                    )

                                if cv_model_configs:
                                    # Run Walk-Forward CV for all candidates
                                    cv_model_results = {}
                                    for cv_config in cv_model_configs:
                                        cv_model_fn = create_cv_model_fn(
                                            cv_config['model_type'],
                                            cv_config['loss_params'],
                                            cv_config['hp_profile'],
                                        )
                                        # Horizon weights: heavily weight lag 4 (the evaluation lag)
                                        # Lag 0 = first week after cutoff, lag 4 = 5th week
                                        # We optimise for lag 4 performance since benchmarks measure at lag 4
                                        _horizon_weights = {
                                            1: 0.05, 2: 0.08, 3: 0.12,
                                            4: 0.40,  # Lag 4 gets 40% weight — primary evaluation metric
                                            5: 0.15, 6: 0.10, 7: 0.05, 8: 0.05,
                                        }
                                        # Trim to actual forecast_horizon
                                        _horizon_weights = {k: v for k, v in _horizon_weights.items() if k <= forecast_horizon}

                                        try:
                                            cv_result = wf_cv.evaluate_model(
                                                X=X_cv,
                                                y=y_cv,
                                                model_fn=cv_model_fn,
                                                model_name=cv_config['name'],
                                                model_config=cv_config,
                                                period_col='period_idx',
                                                feature_cols=feature_cols,
                                                store_predictions=walk_forward_optimize_ensemble,
                                                horizon_weights=_horizon_weights,
                                            )
                                            cv_model_results[cv_config['name']] = cv_result
                                        except Exception as cv_err:
                                            logger.warning(f"Walk-Forward CV failed for {cv_config['name']}: {cv_err}")

                                    if cv_model_results:
                                        # Create WalkForwardCVResult and select best model
                                        walk_forward_cv_result = WalkForwardCVResult(
                                            segment_id=mg_str,
                                            n_folds=walk_forward_n_folds,
                                            forecast_horizon=forecast_horizon,
                                            min_train_periods=walk_forward_min_train_periods,
                                            model_results=cv_model_results,
                                        )
                                        walk_forward_cv_result.select_best_model()

                                        # Detect concept drift if enabled
                                        if walk_forward_detect_drift:
                                            for model_name, model_result in cv_model_results.items():
                                                drift_info = detect_concept_drift(model_result.fold_wapes)
                                                if drift_info['drift_detected']:
                                                    logger.warning(
                                                        f"Model {mg} {model_name}: Concept drift detected "
                                                        f"(trend={drift_info['trend']:.4f}, {drift_info['recommendation']})"
                                                    )
                                                    walk_forward_cv_result.drift_models.append(model_name)

                                        walk_forward_cv_result.any_drift_detected = len(walk_forward_cv_result.drift_models) > 0
                                        walk_forward_cv_used = True

                                        # Log CV results
                                        logger.info(
                                            f"Model {mg}: Walk-Forward CV best={walk_forward_cv_result.best_model_name} "
                                            f"(avg_WAPE={walk_forward_cv_result.best_avg_wape:.4f} ± {walk_forward_cv_result.best_std_wape:.4f})"
                                        )

                            else:
                                logger.info(f"Model {mg}: Not enough periods for Walk-Forward CV ({n_unique_periods} < {min_required})")
                        else:
                            logger.warning(f"Model {mg}: Cannot create period_idx for Walk-Forward CV")

                    except Exception as wf_err:
                        logger.warning(f"Model {mg}: Walk-Forward CV failed, falling back to single split: {wf_err}")
                        walk_forward_cv_used = False

                # =====================================================================
                # TRAIN CANDIDATES AND SELECT BEST BASED ON RECURSIVE VALIDATION WAPE
                # STATE-OF-THE-ART: Use per-segment loss functions from strategy
                # =====================================================================
                # IMPORTANT: We evaluate ALL candidates using recursive forecasting to
                # select the model that will perform best in real-world multi-step scenarios.
                # Direct WAPE can be misleading as it uses actual future values in lag features.
                candidate_results = {}
                candidate_recursive_wape = {}  # Store recursive WAPE for each candidate

                for model_type in candidates:
                    if model_type not in TRAINING_FUNCTIONS:
                        logger.warning(f"Model type {model_type} not in TRAINING_FUNCTIONS")
                        continue
                    try:
                        # STATE-OF-THE-ART: Get loss-specific parameters for this model type
                        loss_params = get_loss_params(recommended_loss, model_type)

                        # Log when using non-default loss function
                        if loss_params and recommended_loss != 'mse':
                            logger.info(f"  {mg_str} {model_type}: Using {recommended_loss} loss with params {loss_params}")

                        # STATE-OF-THE-ART: Get segment-specific hyperparameter profile
                        hyperparam_profile = segment_hyperparam_profiles.get(mg_str)
                        if hyperparam_profile:
                            logger.debug(f"  {mg_str} {model_type}: Using segment-specific hyperparameters")

                        # Create training function with loss parameters AND segment-specific
                        # hyperparameters. Phase-3 hierarchical models additionally receive
                        # the per-segment train_df / val_df / feature_cols / key_col / target_col
                        # and the resolved hierarchy_col so they can actually train instead
                        # of silently failing with a missing-argument TypeError.
                        training_fn = create_training_fn(
                            model_type=model_type,
                            loss_params=loss_params,
                            hyperparam_profile=hyperparam_profile,
                            early_stopping_rounds=mg_config.get('recommended_hyperparams', {}).get('early_stopping_rounds', 50),
                            train_df=train_df,
                            val_df=val_df,
                            feature_cols=feature_cols,
                            key_col=key_col,
                            target_col_ctx=target_col,
                            hierarchy_col=_hierarchy_col,
                        )
                        result = training_fn(X_train, y_train, X_val, y_val)
                        if result is not None:
                            candidate_results[model_type] = result

                            # Compute recursive WAPE for this candidate if enabled.
                            #
                            # When `recursive_validation_lag` is set, two things change:
                            #   1. the walk-forward stops at that lag (saves cost)
                            #   2. the SELECTION metric is wape_by_step[lag], not the
                            #      full-horizon average — so candidates are picked on
                            #      the exact lag the downstream metric cares about
                            #      (e.g. DIQ's lag-4 backtest -> set lag=5)
                            #
                            # Uses _effective_recursive_validation (computed once near
                            # the start of run_full_training_pipeline) — that flag is
                            # `use_recursive_validation AND NOT use_direct_multi_horizon`,
                            # so when DMH is the inference path this entire block
                            # short-circuits to the direct-WAPE branch.
                            if _effective_recursive_validation:
                                _eval_horizon = (
                                    min(forecast_horizon, recursive_validation_lag)
                                    if recursive_validation_lag is not None
                                    else forecast_horizon
                                )
                                try:
                                    recursive_eval = evaluate_model_recursive(
                                        model=result.model,
                                        train_df=train_df,
                                        eval_df=val_df,
                                        feature_columns=feature_cols,
                                        target_col=target_col,
                                        key_col=key_col,
                                        date_col=date_col,
                                        forecast_horizon=_eval_horizon,
                                        time_format=time_format,
                                    )
                                    # When `recursive_validation_lag` is set, score on
                                    # the SPECIFIC lag's WAPE, not the average over all
                                    # lags 1..N.  Falls back to overall wape if the
                                    # specific lag isn't in wape_by_step (shouldn't
                                    # happen if forecast_horizon >= lag).
                                    if recursive_validation_lag is not None:
                                        _wape = recursive_eval.wape_by_step.get(
                                            recursive_validation_lag, recursive_eval.wape,
                                        )
                                    else:
                                        _wape = recursive_eval.wape
                                    candidate_recursive_wape[model_type] = {
                                        'wape': _wape,
                                        'wape_by_step': recursive_eval.wape_by_step,
                                    }
                                    logger.debug(
                                        f"Model {mg} {model_type}: direct={result.val_wape:.4f}, "
                                        f"recursive_score={_wape:.4f} "
                                        f"(lag={recursive_validation_lag if recursive_validation_lag else 'avg'})"
                                    )
                                except Exception as e:
                                    logger.warning(f"Recursive eval failed for {model_type} on {mg}: {e}")
                                    # Fall back to direct WAPE
                                    candidate_recursive_wape[model_type] = {
                                        'wape': result.val_wape,
                                        'wape_by_step': {},
                                    }
                            else:
                                # Use direct WAPE if recursive validation disabled
                                candidate_recursive_wape[model_type] = {
                                    'wape': result.val_wape,
                                    'wape_by_step': {},
                                }
                    except Exception as e:
                        logger.warning(f"Failed {model_type} for {mg}: {type(e).__name__}: {e}")

                if len(candidate_results) == 0:
                    raise ValueError(f"All candidate models failed for {mg}")

                # =====================================================================
                # STATE-OF-THE-ART: MODEL SELECTION WITH WALK-FORWARD CV INTEGRATION
                # Use CV results for robust model selection when available
                # =====================================================================
                if walk_forward_cv_used and walk_forward_cv_result is not None:
                    # Use Walk-Forward CV results for model selection (more robust)
                    best_type = walk_forward_cv_result.best_model_name

                    # Ensure best_type has a trained result (may not if CV failed for some models)
                    if best_type not in candidate_results:
                        logger.warning(f"Model {mg}: CV best '{best_type}' not in candidates, falling back to recursive WAPE")
                        best_type = min(
                            candidate_recursive_wape.keys(),
                            key=lambda m: candidate_recursive_wape[m]['wape']
                        )

                    best_result = candidate_results[best_type]
                    best_wape = best_result.val_wape  # Direct WAPE for reference
                    recursive_val_wape = candidate_recursive_wape.get(best_type, {}).get('wape', best_result.val_wape)
                    recursive_val_wape_by_step = candidate_recursive_wape.get(best_type, {}).get('wape_by_step', {})

                    # Use CV average WAPE as the more honest performance estimate
                    cv_avg_wape = walk_forward_cv_result.best_avg_wape
                    cv_std_wape = walk_forward_cv_result.best_std_wape

                    logger.info(
                        f"Model {mg}: Selected {best_type} via Walk-Forward CV "
                        f"(CV_WAPE={cv_avg_wape:.4f}±{cv_std_wape:.4f}, recursive={recursive_val_wape:.4f})"
                    )

                else:
                    # Fallback: Select best model based on RECURSIVE WAPE (not direct WAPE)
                    best_type = min(
                        candidate_recursive_wape.keys(),
                        key=lambda m: candidate_recursive_wape[m]['wape']
                    )
                    best_result = candidate_results[best_type]
                    best_wape = best_result.val_wape  # Direct WAPE for reference
                    recursive_val_wape = candidate_recursive_wape[best_type]['wape']
                    recursive_val_wape_by_step = candidate_recursive_wape[best_type]['wape_by_step']
                    cv_avg_wape = None
                    cv_std_wape = None

                    logger.info(f"Model {mg}: Selected {best_type} (recursive WAPE={recursive_val_wape:.4f}, direct={best_wape:.4f})")

                # Store candidate comparison for this model group (now includes recursive WAPE + CV metrics)
                mg_candidate_comparison = {}
                for mtype, result in candidate_results.items():
                    if result is not None:
                        cv_metrics = {}
                        if walk_forward_cv_result and mtype in walk_forward_cv_result.model_results:
                            cv_model_res = walk_forward_cv_result.model_results[mtype]
                            cv_metrics = {
                                'cv_avg_wape': cv_model_res.avg_wape,
                                'cv_std_wape': cv_model_res.std_wape,
                                'cv_fold_wapes': cv_model_res.fold_wapes,
                                'cv_wape_trend': cv_model_res.wape_trend,
                                'cv_drift_detected': cv_model_res.concept_drift_detected,
                            }
                        mg_candidate_comparison[mtype] = {
                            'val_wape_direct': result.val_wape,
                            'val_wape_recursive': candidate_recursive_wape.get(mtype, {}).get('wape', result.val_wape),
                            'train_wape': result.train_wape,
                            'selected': mtype == best_type,
                            **cv_metrics,
                        }
                all_candidate_comparisons[mg_str] = mg_candidate_comparison

                # Use recursive WAPE for model selection (or CV WAPE if available)
                val_wape_for_selection = recursive_val_wape

                # =====================================================================
                # STATE-OF-THE-ART: PATTERN-AWARE ENSEMBLE CREATION
                # Uses demand pattern to weight models appropriately
                # =====================================================================
                ensemble_model = None
                ensemble_info = {}
                # Disable pattern ensemble for segment-pooled models: the ensemble
                # often overrides a good CV-selected model (e.g. catboost) with a
                # worse blend (e.g. xgboost+zero_inflated+hurdle), increasing WAPE.
                # Individual key models can still benefit from ensembling.
                skip_ensemble = (model_strategy == 'segment_pooled' and n_keys_in_group > 1)
                if enable_ensemble_optimization and len(candidate_results) >= 2 and not skip_ensemble:
                    try:
                        # STATE-OF-THE-ART: Build enhanced meta-features for pattern-aware ensemble
                        enhanced_meta = None
                        try:
                            # Build segment profile for enhanced meta-features
                            segment_profile_for_meta = {
                                'zero_fraction': 0.3 if demand_pattern in ['intermittent', 'lumpy'] else 0.1,
                                'cv': 1.5 if demand_pattern in ['erratic', 'lumpy'] else 0.8,
                                'segment_size': len(train_df[key_col].unique()) if key_col in train_df.columns else 1,
                                'expected_difficulty': expected_difficulty,
                            }
                            enhanced_meta = build_enhanced_meta_features(
                                y=y_train,
                                eda_insights=eda_for_hyperparams,
                                segment_profile=segment_profile_for_meta,
                                feature_context=feature_context,
                                time_format=time_format,
                            )
                        except Exception as meta_err:
                            logger.debug(f"Enhanced meta-features failed: {meta_err}")

                        # STATE-OF-THE-ART: Create pattern-aware ensemble
                        ensemble_model, ensemble_info = create_pattern_aware_ensemble(
                            candidate_results=candidate_results,
                            demand_pattern=demand_pattern,
                            X_val=X_val,
                            y_val=y_val,
                            top_k=min(ensemble_top_k, len(candidate_results)),
                            meta_features=enhanced_meta,
                        )

                        # Check if ensemble beats best single model
                        if ensemble_info.get('is_ensemble', False) and ensemble_info.get('ensemble_wape', float('inf')) < val_wape_for_selection:
                            logger.info(f"Model {mg}: Pattern-aware ensemble ({ensemble_info['ensemble_wape']:.4f}) beats best single ({val_wape_for_selection:.4f})")
                            # Use ensemble as the final model
                            final_model = ensemble_model
                            final_model_type = f"pattern_ensemble_{'+'.join(ensemble_info.get('model_types', []))}"
                            final_val_wape = ensemble_info['ensemble_wape']
                            ensemble_info['method'] = 'pattern_aware_weighting'
                        else:
                            final_model = best_result.model
                            final_model_type = best_type
                            final_val_wape = val_wape_for_selection
                    except Exception as e:
                        logger.warning(f"Pattern-aware ensemble creation failed for {mg}: {e}")
                        # Fall back to standard ensemble
                        try:
                            ensemble_model, ensemble_info = create_ensemble_from_candidates(
                                candidate_results=candidate_results,
                                feature_columns=feature_cols,
                                X_val=X_val,
                                y_val=y_val,
                                top_k=min(ensemble_top_k, len(candidate_results)),
                                optimization_method='inverse_wape',
                            )
                            if ensemble_info['ensemble_wape'] < val_wape_for_selection:
                                final_model = ensemble_model
                                final_model_type = f"ensemble_{'+'.join(ensemble_info['model_types'])}"
                                final_val_wape = ensemble_info['ensemble_wape']
                            else:
                                final_model = best_result.model
                                final_model_type = best_type
                                final_val_wape = val_wape_for_selection
                        except:
                            final_model = best_result.model
                            final_model_type = best_type
                            final_val_wape = val_wape_for_selection
                else:
                    final_model = best_result.model
                    final_model_type = best_type
                    final_val_wape = val_wape_for_selection

                # =====================================================================
                # STATE-OF-THE-ART: CALIBRATE ZERO-INFLATED/HURDLE THRESHOLDS
                # Learn optimal thresholds from validation data instead of using fixed rules
                # =====================================================================
                calibrated_threshold = None
                if final_model_type in ['zero_inflated', 'hurdle_model']:
                    try:
                        if final_model_type == 'zero_inflated':
                            calibrated_threshold, threshold_info = calibrate_zero_inflated_threshold(
                                model=final_model,
                                X_val=X_val,
                                y_val=y_val,
                                metric='wape',
                            )
                        else:
                            # Get zero fraction for hurdle threshold optimization
                            zero_frac = (y_train == 0).mean()
                            calibrated_threshold, threshold_info = optimize_hurdle_threshold(
                                model=final_model,
                                X_val=X_val,
                                y_val=y_val,
                                zero_fraction=zero_frac,
                            )

                        if calibrated_threshold and hasattr(final_model, 'zero_threshold'):
                            final_model.zero_threshold = calibrated_threshold
                            logger.info(f"Model {mg}: Calibrated {final_model_type} threshold to {calibrated_threshold:.2f} "
                                       f"(improvement: {threshold_info.get('improvement_pct', 0):.1f}%)")
                    except Exception as thresh_err:
                        logger.debug(f"Threshold calibration failed for {mg}: {thresh_err}")

                # =====================================================================
                # Sprint 3 B9: FEATURE IMPORTANCE PRUNING
                # Remove low-importance features to reduce noise and speed up inference
                # =====================================================================
                try:
                    from utils.sprint3_features import prune_features_by_importance
                    # Skip pruning retrain for Phase-3 hierarchical winners. The
                    # pruning step calls create_training_fn to retrain with a
                    # reduced feature set; for hierarchical models that requires
                    # threading train_df/val_df context through, and more
                    # fundamentally a global ``feature_importances_`` attribute
                    # does not capture the per-group sub-model importances of
                    # a multi-level ensemble — pruning by the wrapper's
                    # aggregate importance would discard features that are
                    # critical for some groups.
                    _skip_pruning_retrain = final_model_type in _HIERARCHICAL_MODELS
                    if hasattr(final_model, 'feature_importances_') and len(feature_cols) > 20 and not _skip_pruning_retrain:
                        pruned_cols, importance_dict = prune_features_by_importance(
                            model=final_model, feature_cols=feature_cols,
                            X_val=X_val, y_val=y_val,
                            min_importance_pct=0.3, max_features=80,
                        )
                        if len(pruned_cols) < len(feature_cols):
                            # Retrain with pruned features for potentially better performance
                            pruned_idx = [feature_cols.index(c) for c in pruned_cols if c in feature_cols]
                            X_train_pruned = X_train[:, pruned_idx]
                            X_val_pruned = X_val[:, pruned_idx]
                            try:
                                retrain_fn = create_training_fn(
                                    final_model_type, loss_params=loss_params,
                                    hyperparam_profile=segment_hyperparam_profiles.get(demand_pattern),
                                )
                                pruned_result = retrain_fn(X_train_pruned, y_train, X_val_pruned, y_val)
                                if pruned_result.val_wape <= final_val_wape * 1.02:
                                    # Pruned model is at most 2% worse — use it (faster inference)
                                    final_model = pruned_result.model
                                    feature_cols = pruned_cols
                                    X_train = X_train_pruned
                                    X_val = X_val_pruned
                                    logger.info(f"Model {mg}: Using pruned feature set ({len(pruned_cols)} features)")
                            except Exception:
                                pass  # Keep original model
                except Exception as prune_err:
                    logger.debug(f"Feature pruning skipped for {mg}: {prune_err}")

                # =====================================================================
                # STATE-OF-THE-ART: COLLECT DATA FOR SEGMENT-AWARE BIAS CALIBRATION
                # =====================================================================
                try:
                    # Get predictions on validation set for bias calibration
                    if isinstance(final_model, dict) and 'forecast' in final_model:
                        val_preds = np.full(len(y_val), float(final_model['forecast']))
                    else:
                        val_preds = final_model.predict(X_val)
                    val_preds = np.clip(val_preds, 0, None)

                    # DYNAMIC zero_fraction: Compute from training data for each key
                    # This ensures calibration uses actual training period behavior
                    val_zero_fractions = None
                    if key_col in val_df.columns and key_col in train_df.columns:
                        # Compute zero_fraction per key from training data
                        key_zf = train_df.groupby(key_col)[target_col].apply(
                            lambda x: (x == 0).mean()
                        ).to_dict()
                        val_zero_fractions = val_df[key_col].map(key_zf).fillna(0.0).values

                    # Store for post-training calibration
                    segment_calibration_data[mg_str] = {
                        'y_actual': y_val,
                        'y_predicted': val_preds,
                        'zero_fractions': val_zero_fractions,
                        'demand_pattern': demand_pattern,
                    }
                except Exception as cal_collect_err:
                    logger.debug(f"Calibration data collection failed for {mg}: {cal_collect_err}")

                # =====================================================================
                # TEST EVALUATION (if test data available)
                # =====================================================================
                test_wape = None
                test_wape_by_step = {}
                if test_df_all is not None and ml_col in test_df_all.columns:
                    test_df = test_df_all[test_df_all[ml_col] == mg].copy()
                    if len(test_df) > 0:
                        try:
                            # Combine train + val for historical data when evaluating test
                            train_val_combined = pd.concat([train_df, val_df], ignore_index=True)
                            test_eval = evaluate_model_recursive(
                                model=final_model,
                                train_df=train_val_combined,
                                eval_df=test_df,
                                feature_columns=feature_cols,
                                target_col=target_col,
                                key_col=key_col,
                                date_col=date_col,
                                forecast_horizon=forecast_horizon,
                                time_format=time_format,
                            )
                            test_wape = test_eval.wape
                            test_wape_by_step = test_eval.wape_by_step
                            logger.info(f"Model {mg}: Test WAPE={test_wape:.4f}")
                        except Exception as e:
                            logger.warning(f"Test evaluation failed for {mg}: {e}")

                # =====================================================================
                # SAVE MODEL ARTIFACT WITH FULL METADATA + HYPERPARAMETERS
                # =====================================================================
                # Get hyperparameters from best result for saving in artifact
                artifact_hyperparameters = best_result.hyperparameters if best_result else {}

                model_path = os.path.join(model_dir, f'{mg_str}_model.pkl')
                model_artifact = {
                    'model': final_model,
                    'feature_columns': feature_cols,
                    'model_type': final_model_type,
                    'hyperparameters': artifact_hyperparameters,  # Add hyperparameters!
                    'target_column': target_col,
                    'key_column': key_col,
                    'date_column': date_col,
                    'forecast_horizon': forecast_horizon,
                    'is_ensemble': isinstance(final_model, EnsembleModel),
                    'ensemble_info': ensemble_info if isinstance(final_model, EnsembleModel) else {},
                }
                joblib.dump(model_artifact, model_path)

                # Save feature columns as separate JSON for easy inspection
                feature_cols_path = model_path.replace('.pkl', '_features.json')
                save_json({
                    'feature_columns': feature_cols,
                    'model_type': final_model_type,
                    'hyperparameters': artifact_hyperparameters,  # Add here too
                    'forecast_horizon': forecast_horizon,
                }, feature_cols_path)

                # =====================================================================
                # RECORD COMPREHENSIVE MODEL SPEC WITH HYPERPARAMETERS
                # =====================================================================
                # Get hyperparameters from the best result (or ensemble info)
                if isinstance(final_model, EnsembleModel):
                    # For ensemble, store member hyperparameters
                    model_hyperparameters = {
                        'is_ensemble': True,
                        'ensemble_weights': ensemble_info.get('weights', {}),
                        'member_params': {
                            mtype: candidate_results[mtype].hyperparameters
                            for mtype in ensemble_info.get('model_types', [])
                            if mtype in candidate_results and candidate_results[mtype] is not None
                        }
                    }
                else:
                    # For single model, store its hyperparameters
                    model_hyperparameters = best_result.hyperparameters if best_result else {}

                # Build Walk-Forward CV summary for spec
                walk_forward_cv_summary = None
                if walk_forward_cv_used and walk_forward_cv_result is not None:
                    walk_forward_cv_summary = {
                        'enabled': True,
                        'n_folds': walk_forward_cv_result.n_folds,
                        'strategy': walk_forward_strategy,
                        'best_model': walk_forward_cv_result.best_model_name,
                        'best_avg_wape': walk_forward_cv_result.best_avg_wape,
                        'best_std_wape': walk_forward_cv_result.best_std_wape,
                        'expected_wape_range': list(walk_forward_cv_result.expected_wape_range),
                        'concept_drift_detected': walk_forward_cv_result.any_drift_detected,
                        'drift_affected_models': walk_forward_cv_result.drift_models,
                        'ensemble_weights': walk_forward_cv_result.ensemble_weights if walk_forward_optimize_ensemble else {},
                        'ensemble_avg_wape': walk_forward_cv_result.ensemble_avg_wape if walk_forward_optimize_ensemble else None,
                        'model_summary': {
                            name: {
                                'avg_wape': res.avg_wape,
                                'std_wape': res.std_wape,
                                'wape_trend': res.wape_trend,
                                'fold_wapes': res.fold_wapes,
                            }
                            for name, res in walk_forward_cv_result.model_results.items()
                        }
                    }

                spec = {
                    'model_group': mg_str,
                    'model_type': final_model_type,
                    'demand_pattern': demand_pattern,
                    # STATE-OF-THE-ART: Loss function and validation strategy from strategy
                    'recommended_loss': recommended_loss,
                    'validation_strategy': validation_strategy,
                    'expected_difficulty': expected_difficulty,
                    # HYPERPARAMETERS - Critical for deterministic retraining!
                    'hyperparameters': model_hyperparameters,
                    # Validation metrics (used for model selection)
                    'val_wape': final_val_wape,
                    'val_wape_direct': best_wape,  # Direct prediction WAPE
                    'val_wape_recursive': recursive_val_wape,  # Recursive forecast WAPE
                    'val_wape_by_step': recursive_val_wape_by_step,
                    # STATE-OF-THE-ART: Walk-Forward Cross-Validation metrics
                    'walk_forward_cv': walk_forward_cv_summary,
                    'cv_avg_wape': cv_avg_wape,  # More honest WAPE estimate from CV
                    'cv_std_wape': cv_std_wape,  # WAPE variability across time periods
                    # Test metrics (holdout evaluation)
                    'test_wape': test_wape,
                    'test_wape_by_step': test_wape_by_step,
                    # Training info
                    'train_wape': best_result.train_wape,
                    'n_train_samples': len(y_train),
                    'n_val_samples': len(y_val),
                    'n_features': len(feature_cols),
                    # Feature columns - categorized for clarity
                    # Full list needed for inference, categories for understanding
                    'feature_columns': feature_cols,
                    'n_features_by_category': _categorize_feature_counts(feature_cols),
                    'forecast_horizon': forecast_horizon,
                    # Candidate comparison
                    'candidates_tried': list(candidate_results.keys()),
                    'candidate_results': mg_candidate_comparison,
                    'best_single_model': best_type,
                    # Ensemble info
                    'is_ensemble': isinstance(final_model, EnsembleModel),
                    'ensemble_info': ensemble_info,
                    # Paths
                    'model_path': model_path,
                    'feature_columns_path': feature_cols_path,
                    'status': 'trained',
                    'training_time_sec': time.time() - mg_start,
                }
                model_specs.append(spec)
                models_trained += 1

                # Accumulate for overall WAPE (validation) — local-only,
                # the dispatcher aggregates after parallel calls finish.
                val_sum = np.sum(np.abs(y_val))
                total_val_actuals += val_sum
                total_val_errors += final_val_wape * val_sum

                # Accumulate for overall test WAPE
                if test_wape is not None and test_df_all is not None:
                    test_df = test_df_all[test_df_all[ml_col] == mg]
                    if len(test_df) > 0:
                        test_sum = np.sum(np.abs(test_df[target_col].fillna(0)))
                        total_test_actuals += test_sum
                        total_test_errors += test_wape * test_sum

                # Stash a one-line success summary for the dispatcher to
                # log after the call completes (the dispatcher's running
                # global count is the right number to gate log frequency).
                _success_summary = (
                    final_model_type, final_val_wape,
                    test_wape if test_wape is not None else None,
                )
            except Exception as e:
                logger.error(f"Failed model {mg}: {e}")
                model_specs.append({
                    'model_group': mg_str,
                    'status': 'failed',
                    'error': str(e)[:200],
                })
                models_failed += 1
                _success_summary = None
            return {
                'models_trained': models_trained, 'models_failed': models_failed,
                'total_val_actuals': total_val_actuals, 'total_val_errors': total_val_errors,
                'total_test_actuals': total_test_actuals, 'total_test_errors': total_test_errors,
                'model_specs': model_specs,
                'all_candidate_comparisons': all_candidate_comparisons,
                'segment_calibration_data': segment_calibration_data,
                'mg_str': mg_str,
                'success_summary': _success_summary,
            }
        # ─── end of closure _run_one_mg ───

        # ============================================================
        # DISPATCHER: Spark > Threads > Sequential
        # ============================================================
        # Each backend produces a list of `_local_result` dicts (one
        # per model_group); we aggregate them into the outer-scope
        # state in the main thread after dispatch finishes.
        # ============================================================
        _all_results: List[dict] = []
        try:
            if _spark_available:
                # Spark dispatch — distributes work across every
                # executor on the cluster.  Each task is a separate
                # Python process for proper isolation.
                #
                # Implementation:
                #   1. Broadcast train/val/test_df_all to every
                #      executor ONCE.  Without this, every task
                #      would receive its own pickled copy of the
                #      multi-GB DataFrames in its closure capture.
                #   2. Build a tiny Spark DataFrame with one row per
                #      model_group.  Repartition to _ptw partitions
                #      so concurrent tasks are capped at _ptw (each
                #      Spark task processes its partition's rows
                #      sequentially within the executor process).
                #   3. mapInPandas runs our dispatch closure on each
                #      partition; for each model_group in that
                #      partition's pandas frame we call _run_one_mg
                #      with the broadcast values, then pickle the
                #      result dict into a single binary column.
                #   4. collect() pulls results back to the driver,
                #      which we unpickle into _all_results.
                #
                # Closure capture concerns:
                #   * The dispatch function captures `_run_one`
                #     (alias for `_run_one_mg`), the broadcast
                #     handles, and `_per_task_cores`.  Spark
                #     cloudpickles the dispatch function and all
                #     its captures — _run_one's enclosing scope
                #     captures (model_groups_config, etc.) ride
                #     along.  The big DataFrames go via broadcast,
                #     not closure capture, so the serialised
                #     function stays small.
                #   * Per-task env vars (OMP_NUM_THREADS etc.) are
                #     set INSIDE the dispatch closure on the
                #     executor — Spark doesn't have Ray's
                #     `runtime_env` mechanism but setting env vars
                #     at the start of the task body is good enough
                #     because LightGBM / XGBoost read them at
                #     fit-time, not at import-time.
                import pickle as _pkl
                from pyspark.sql.types import (
                    StructType as _StructType,
                    StructField as _StructField,
                    StringType as _StringType,
                    BinaryType as _BinaryType,
                )

                _sc = _spark_session.sparkContext
                # Re-derive per-task cores against the cluster (not the
                # driver), using the same multi-method probe as the
                # availability check above.  See the long comment in
                # Region A for why driver cpu_count alone would be
                # wrong, and why we need to try multiple methods.
                def _conf_int_dispatch(key, default=0):
                    try:
                        return int(_spark_session.conf.get(key, str(default)))
                    except Exception:
                        return default

                _exec_cores_d = _conf_int_dispatch("spark.executor.cores", 0)
                _total_cores_dispatch = 0

                # Method 1: Databricks cluster tag.
                _db_workers_d = _conf_int_dispatch(
                    "spark.databricks.clusterUsageTags.clusterAllNumberOfWorkers",
                    0,
                )
                if _db_workers_d > 0 and _exec_cores_d > 0:
                    _total_cores_dispatch = _db_workers_d * _exec_cores_d

                # Method 2: spark.executor.instances.
                if _total_cores_dispatch <= 0:
                    _exec_inst_d = _conf_int_dispatch(
                        "spark.executor.instances", 0
                    )
                    if _exec_inst_d > 0 and _exec_cores_d > 0:
                        _total_cores_dispatch = _exec_inst_d * _exec_cores_d

                # Method 3: defaultParallelism.
                if _total_cores_dispatch <= 0:
                    try:
                        _total_cores_dispatch = int(_sc.defaultParallelism)
                    except Exception:
                        _total_cores_dispatch = 0

                # Method 4: registered-executor enumeration.
                if _total_cores_dispatch <= 0:
                    try:
                        _exec_keys = list(
                            _sc.getExecutorMemoryStatus().keys()
                        )
                        _exec_n = max(0, len(_exec_keys) - 1)
                        if _exec_n > 0 and _exec_cores_d > 0:
                            _total_cores_dispatch = _exec_n * _exec_cores_d
                    except Exception:
                        pass

                _cores_for_sizing_dispatch = (
                    _total_cores_dispatch
                    if _total_cores_dispatch > 0
                    else (_os_pll.cpu_count() or 16)
                )
                _per_task_cores = max(
                    1, _cores_for_sizing_dispatch // _ptw
                )

                printer.print(
                    "  Spark: broadcasting train/val/test DataFrames "
                    "to all executors..."
                )
                _train_bc = _sc.broadcast(train_df_all)
                _val_bc = _sc.broadcast(val_df_all)
                _test_bc = _sc.broadcast(test_df_all)
                printer.print("  Spark: broadcast complete; dispatching tasks...")

                # Local capture for cloudpickle.  Spark serialises
                # _spark_dispatch and any names it references; making
                # `_run_one` a plain local makes the capture
                # straightforward (no enclosing-scope free-var look-up
                # at unpickle time).
                _run_one = _run_one_mg

                # ════════════════════════════════════════════════════
                # Make `utils.*` importable on Spark executors WITHOUT
                # requiring `%pip install` of the repo.
                #
                # On Databricks DBR 14+ ML the user's `/Workspace/...`
                # tree is FUSE-mounted on every cluster node (driver
                # and workers).  Files there can be `open()`-ed from
                # any process on the cluster.  What's NOT automatic is
                # `sys.path` — Spark executor Python processes start
                # fresh and don't inherit the driver's `sys.path`.
                #
                # The wrapper notebook does `sys.path.insert(0,
                # REPO_PATH)` on the driver, which is why the driver
                # can `from utils.X import Y` without an install.  But
                # the closure body has dynamic imports like
                # `from utils.sprint3_features import …` that execute
                # ON THE EXECUTOR — so the executor needs the repo on
                # ITS sys.path too.
                #
                # We capture the repo root here (resolved from this
                # module's __file__: utils/model_training.py → parent
                # is `utils/` → parent of that is the repo root), then
                # the dispatch closure inserts it into sys.path on
                # each executor BEFORE doing any `utils.*` import.
                # The string is captured-by-value into the cloudpickled
                # closure, so executors get the resolved path even
                # though `__file__` itself isn't valid on workers.
                #
                # If FUSE on Workspace turns out to be flaky on a
                # given DBR/cluster combination, the user can fall
                # back to `%pip install --no-deps /Workspace/.../REPO`
                # in the notebook — and this block becomes a harmless
                # no-op (the path is already importable via the
                # installed package, the sys.path entry is a duplicate
                # that Python ignores).
                # ════════════════════════════════════════════════════
                _repo_path_for_executors = _os_pll.path.dirname(
                    _os_pll.path.dirname(_os_pll.path.abspath(__file__))
                )
                printer.print(
                    f"  Spark: executors will sys.path-insert "
                    f"{_repo_path_for_executors!r} so `utils.*` resolves"
                )

                _result_schema = _StructType([
                    _StructField("mg_id", _StringType(), nullable=False),
                    _StructField("payload", _BinaryType(), nullable=False),
                ])

                # Closure runs on Spark executors.  Must be picklable;
                # any references it holds (including _run_one and
                # _repo_path_for_executors) get serialised with it.
                # Imports are inside the function body so we don't
                # depend on the executor having the exact same
                # import-time state as the driver.
                def _spark_dispatch(iterator):
                    import os as _os_w
                    import sys as _sys_w
                    import pickle as _pkl_w
                    import traceback as _tb_w
                    import pandas as _pd_w

                    # ── Make `utils.*` importable on this executor ──
                    # See the long comment above the closure for why
                    # this is needed.  Idempotent: a duplicate sys.path
                    # entry is harmless if the package is already
                    # importable via some other mechanism (e.g. a
                    # `%pip install <repo>` in the notebook).
                    if (
                        _repo_path_for_executors
                        and _repo_path_for_executors not in _sys_w.path
                    ):
                        _sys_w.path.insert(0, _repo_path_for_executors)

                    # Pin every threading layer to the per-task budget
                    # so multiple OMP/MKL libs in the user's stack
                    # don't oversubscribe the executor's CPUs.  Set
                    # BEFORE the heavy imports so libs that read these
                    # at import-time pick up the right values.
                    _os_w.environ["OMP_NUM_THREADS"] = str(_per_task_cores)
                    _os_w.environ["MKL_NUM_THREADS"] = str(_per_task_cores)
                    _os_w.environ["OPENBLAS_NUM_THREADS"] = str(_per_task_cores)
                    _os_w.environ["VECLIB_MAXIMUM_THREADS"] = str(_per_task_cores)
                    _os_w.environ["NUMEXPR_NUM_THREADS"] = str(_per_task_cores)
                    # libomp+libgomp coexistence (suppress abort).
                    _os_w.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                    # Force MKL to use libgomp instead of libomp so
                    # TF (libomp) and MKL (now libgomp) don't both try
                    # to be the canonical OMP runtime.
                    _os_w.environ["MKL_THREADING_LAYER"] = "GNU"
                    # Quieten TF startup chatter.
                    _os_w.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                    # Disable TF's oneDNN custom ops (which load extra
                    # libomp) — small numerical precision difference,
                    # big stability win.
                    _os_w.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
                    # Suppress HuggingFace tokenizer parallelism
                    # warnings (in case crewai etc imports one).
                    _os_w.environ["TOKENIZERS_PARALLELISM"] = "false"
                    # Disable MLflow autologging in workers.
                    _os_w.environ["MLFLOW_AUTOLOGGING_DISABLE"] = "true"

                    for pdf in iterator:
                        out_rows = []
                        for _, row in pdf.iterrows():
                            mg = row["mg_id"]
                            try:
                                r = _run_one(
                                    mg,
                                    _train_bc.value,
                                    _val_bc.value,
                                    _test_bc.value,
                                )
                            except BaseException as _exc:
                                # Wrap any failure into a result dict
                                # so the driver can see it after
                                # collect() rather than crash the
                                # whole stage.
                                r = {
                                    "models_trained": 0,
                                    "models_failed": 1,
                                    "total_val_actuals": 0.0,
                                    "total_val_errors": 0.0,
                                    "total_test_actuals": 0.0,
                                    "total_test_errors": 0.0,
                                    "model_specs": [{
                                        "model_group": str(mg),
                                        "status": "failed",
                                        "error": (
                                            f"{type(_exc).__name__}: "
                                            f"{str(_exc)[:200]}"
                                        ),
                                        "traceback": _tb_w.format_exc()[:2000],
                                    }],
                                    "all_candidate_comparisons": {},
                                    "segment_calibration_data": {},
                                    "mg_str": str(mg),
                                    "success_summary": None,
                                }
                            out_rows.append({
                                "mg_id": str(mg),
                                "payload": _pkl_w.dumps(r),
                            })
                        if out_rows:
                            yield _pd_w.DataFrame(out_rows)

                # One row per model group; repartition caps concurrency
                # at _ptw (or len(model_groups), whichever is smaller).
                # When len(model_groups) > _ptw, Spark will iterate the
                # extra groups serially within each partition's task —
                # so total tasks running concurrently stays bounded by
                # _ptw and per-task cores stays bounded by _per_task_cores.
                _mg_rows = [(str(_mg),) for _mg in model_groups]
                _mg_sdf = _spark_session.createDataFrame(_mg_rows, ["mg_id"])
                _n_partitions = max(1, min(_ptw, len(_mg_rows)))
                _mg_sdf = _mg_sdf.repartition(_n_partitions)

                _result_sdf = _mg_sdf.mapInPandas(
                    _spark_dispatch, schema=_result_schema
                )

                # collect() materialises everything to driver memory.
                # Each row is a (mg_id, pickled-result-dict) pair —
                # the result dicts are small (counts + small lists),
                # so this is fine even for 500+ groups.
                try:
                    _collected = _result_sdf.collect()
                finally:
                    # Always release broadcasts so executor memory frees.
                    for _bc in (_train_bc, _val_bc, _test_bc):
                        try:
                            _bc.unpersist()
                        except Exception:
                            pass

                # Unpickle each result.  An unpickle failure means the
                # task wrote something we can't read — log + skip; the
                # aggregator below tolerates None entries.
                for _row in _collected:
                    try:
                        _all_results.append(_pkl.loads(_row["payload"]))
                    except Exception as _unp_exc:
                        logger.warning(
                            f"Failed to unpickle Spark result for "
                            f"mg={_row['mg_id']}: {_unp_exc}"
                        )
                        _all_results.append(None)
            elif _ptw > 1:
                # ThreadPoolExecutor fallback — works on a single
                # driver only.  LightGBM / XGBoost release the GIL,
                # so this still gives real parallelism.
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=_ptw) as _ex:
                    _futures = {
                        _ex.submit(_run_one_mg, _mg, train_df_all, val_df_all, test_df_all): _mg
                        for _mg in model_groups
                    }
                    for _f in as_completed(_futures):
                        try:
                            _all_results.append(_f.result())
                        except Exception as _exc:
                            logger.error(f"Parallel worker failed for mg={_futures[_f]}: {_exc}")
            else:
                # Sequential — same wall-clock as the legacy path.
                # Pass DFs explicitly (matches the closure's parameter
                # signature so its body uses the local-bound names
                # rather than enclosing-scope free variables).
                for _mg in model_groups:
                    _all_results.append(
                        _run_one_mg(_mg, train_df_all, val_df_all, test_df_all)
                    )
        finally:
            # Restore OMP_NUM_THREADS so this change doesn't leak
            # into subsequent stages of the pipeline.  Spark sets
            # OMP_NUM_THREADS per-task in the executor process, so
            # only the threads-fallback path mutates the driver's
            # global env.
            if _ptw > 1 and not _spark_available:
                if _orig_omp is None:
                    _os_pll.environ.pop("OMP_NUM_THREADS", None)
                else:
                    _os_pll.environ["OMP_NUM_THREADS"] = _orig_omp

        # Aggregate per-mg deltas into outer-scope state.  This is
        # the SINGLE point of mutation regardless of dispatch backend.
        for _r in _all_results:
            if not _r:
                continue
            models_trained += _r.get('models_trained', 0)
            models_failed += _r.get('models_failed', 0)
            total_val_actuals += _r.get('total_val_actuals', 0.0)
            total_val_errors += _r.get('total_val_errors', 0.0)
            total_test_actuals += _r.get('total_test_actuals', 0.0)
            total_test_errors += _r.get('total_test_errors', 0.0)
            model_specs.extend(_r.get('model_specs', []))
            for _k, _v in _r.get('all_candidate_comparisons', {}).items():
                all_candidate_comparisons[_k] = _v
            for _k, _v in _r.get('segment_calibration_data', {}).items():
                segment_calibration_data[_k] = _v
            # Per-mg success log: gate by global count after merge.
            _ss = _r.get('success_summary')
            if _ss is not None and (models_trained <= 5 or models_trained % 10 == 0):
                _model_type, _val_wape, _test_wape = _ss
                _test_info = f", test={_test_wape:.3f}" if _test_wape is not None else ""
                printer.print(
                    f"  {_r.get('mg_str')}: {_model_type} val_WAPE={_val_wape:.3f}{_test_info}"
                )

        # Calculate overall WAPE (validation)
        overall_wape = total_val_errors / total_val_actuals if total_val_actuals > 0 else 1.0
        overall_test_wape = total_test_errors / total_test_actuals if total_test_actuals > 0 else None

        # =====================================================================
        # BUILD KEY-TO-MODEL-GROUP MAPPING for pipeline consumption
        # =====================================================================
        # This tells the pipeline which keys should use which model
        key_to_model_group = {}
        model_group_to_keys = {}

        if 'key' in manifest.columns and ml_col in manifest.columns:
            for _, row in manifest[['key', ml_col]].drop_duplicates().iterrows():
                key_val = str(row['key'])
                mg_val = str(row[ml_col])
                key_to_model_group[key_val] = mg_val

                if mg_val not in model_group_to_keys:
                    model_group_to_keys[mg_val] = []
                model_group_to_keys[mg_val].append(key_val)

        logger.info(f"Built key-to-model mapping: {len(key_to_model_group)} keys -> {len(model_group_to_keys)} groups")

        # Add keys_covered to each model spec
        for spec in model_specs:
            mg = spec.get('model_group', '')
            spec['keys_covered'] = model_group_to_keys.get(mg, [])
            spec['n_keys'] = len(spec['keys_covered'])

        # =====================================================================
        # VALIDATION-LEARNED BIAS CALIBRATION
        # =====================================================================
        # If enabled, learn calibration factors from validation predictions
        # to correct systematic over/under-forecasting bias.
        # Calibration is computed per (model_level, zero_fraction_bucket).
        # =====================================================================
        bias_calibration_data = None
        # Use parameters passed to function (these come from config via training crew)
        apply_bias_calibration_flag = apply_bias_calibration
        # Note: bias_calibration_buckets, bias_calibration_factor_min, bias_calibration_factor_max
        # are now function parameters, no need to redefine them

        logger.info(f"Bias calibration settings (from parameters):")
        logger.info(f"  apply_bias_calibration: {apply_bias_calibration_flag}")
        logger.info(f"  bias_calibration_buckets: {bias_calibration_buckets}")
        logger.info(f"  bias_calibration_factor_bounds: [{bias_calibration_factor_min}, {bias_calibration_factor_max}]")

        # =====================================================================
        # GENERATE VALIDATION PREDICTIONS (ALWAYS - for diagnostic crew)
        # This is done regardless of bias calibration settings, because the
        # diagnostic crew needs validation predictions to analyze model performance.
        # =====================================================================
        try:
            from utils.bias_calibration import (
                learn_bias_calibration,
                generate_validation_predictions,
                save_calibration_factors,
                BiasCalibration,
            )

            printer.print(f"\n{'='*60}")
            printer.print("GENERATING VALIDATION PREDICTIONS FOR DIAGNOSTICS")
            printer.print(f"{'='*60}")

            # Load trained models for validation prediction generation
            # CRITICAL: Pass full model artifacts so feature_columns are available
            trained_models_for_cal = {}
            model_feature_columns = {}
            models_to_load = [s for s in model_specs if s.get('status') == 'trained' and s.get('model_path')]
            models_load_failed = 0

            printer.print(f"Loading {len(models_to_load)} trained models for validation predictions...")

            for spec in models_to_load:
                try:
                    model_artifact = joblib.load(spec['model_path'])
                    mg_key = spec.get('model_group', '')
                    # Pass full artifact dict, not just the model
                    # The artifact contains both 'model' and 'feature_columns'
                    trained_models_for_cal[mg_key] = model_artifact
                    # Also build feature_columns mapping from spec
                    if spec.get('feature_columns'):
                        model_feature_columns[mg_key] = spec['feature_columns']
                except Exception as load_err:
                    models_load_failed += 1
                    logger.error(f"Could not load model for calibration from {spec['model_path']}: {load_err}")

            logger.info(f"Loaded {len(trained_models_for_cal)} models for validation predictions")
            logger.info(f"Feature columns available for {len(model_feature_columns)} models")

            # CRITICAL: Log if no models loaded - this is a serious problem
            if len(trained_models_for_cal) == 0 and len(models_to_load) > 0:
                logger.error(f"CRITICAL: All {len(models_to_load)} model loads FAILED! Cannot generate validation predictions.")
                printer.print(f"ERROR: Failed to load any models ({models_load_failed} failed)")
            elif models_load_failed > 0:
                printer.print(f"WARNING: {models_load_failed}/{len(models_to_load)} models failed to load")

            # =========================================================
            # COMPUTE KEY METADATA FROM TRAINING DATA (not from EDA/segmentation files)
            # This ensures zero_fraction reflects the actual training period behavior
            # =========================================================
            logger.info("Computing key metadata from training data...")

            # Compute zero_fraction from training data for each key
            key_zero_fractions = train_df_all.groupby(key_col).apply(
                lambda g: (g[target_col] == 0).mean()
            ).reset_index()
            key_zero_fractions.columns = [key_col, 'zero_fraction']

            # Get segment_id from training manifest if available
            # NOTE: The variable is 'manifest' not 'train_manifest' in run_full_training_pipeline
            if 'segment_id' in manifest.columns:
                key_segments = manifest[[key_col, 'segment_id']].drop_duplicates()
                key_metadata_df = key_zero_fractions.merge(key_segments, on=key_col, how='left')
                key_metadata_df['segment_id'] = key_metadata_df['segment_id'].fillna('default').astype(str)
            else:
                key_metadata_df = key_zero_fractions.copy()
                key_metadata_df['segment_id'] = 'default'

            logger.info(f"Computed key metadata for {len(key_metadata_df)} keys")
            logger.info(f"  Avg zero_fraction: {key_metadata_df['zero_fraction'].mean():.3f}")
            logger.info(f"  Segments: {key_metadata_df['segment_id'].nunique()}")

            if trained_models_for_cal:
                # Build a fallback `feature_cols` list for
                # generate_validation_predictions().  The per-model
                # authoritative columns come via `model_feature_columns`
                # below (a dict keyed by model_group); `feature_columns`
                # is only used when a model is missing from that dict.
                #
                # Historical context: this used to inherit `feature_cols`
                # from the LAST model group's dispatch loop iteration —
                # working only because the loop was inline and Python
                # leaks for-loop locals to the enclosing scope.  When
                # dispatch was refactored to a closure (_run_one_mg),
                # the closure-local `feature_cols` stopped leaking, and
                # this line raised NameError as soon as a run actually
                # reached it.  We now compute the fallback explicitly
                # from train_df_all's numeric columns, excluding the
                # key/date/target/model-level identifiers.
                _exclude_for_fallback = {key_col, date_col, target_col}
                if ml_col:
                    _exclude_for_fallback.add(ml_col)
                feature_cols = [
                    c for c in train_df_all.columns
                    if c not in _exclude_for_fallback
                    and pd.api.types.is_numeric_dtype(train_df_all[c])
                ]

                # Generate validation predictions using recursive forecasting
                # CRITICAL: Pass model_feature_columns so each model uses correct features
                val_predictions = generate_validation_predictions(
                    trained_models=trained_models_for_cal,
                    train_df=train_df_all,
                    val_df=val_df_all,
                    feature_columns=feature_cols,  # Fallback only
                    target_col=target_col,
                    key_col=key_col,
                    date_col=date_col,
                    model_level_col=ml_col,
                    key_to_model_group=key_to_model_group,
                    forecast_horizon=forecast_horizon,
                    model_feature_columns=model_feature_columns,  # Per-model features
                )

                if len(val_predictions) > 0:
                    # =========================================================
                    # SAVE VALIDATION PREDICTIONS FOR DIAGNOSTIC CREW
                    # This file is ALWAYS saved so diagnostics can analyze model
                    # performance even when bias calibration is disabled.
                    # =========================================================
                    val_predictions_path = os.path.join(model_dir, 'validation_predictions.csv')
                    val_predictions.to_csv(val_predictions_path, index=False)
                    printer.print(f"Saved validation predictions: {len(val_predictions)} rows -> validation_predictions.csv")

                    # =========================================================
                    # BIAS CALIBRATION (only if enabled)
                    # zero_fraction is now always computed from training data above
                    # =========================================================
                    if apply_bias_calibration_flag:
                        printer.print(f"\n{'='*60}")
                        printer.print("LEARNING BIAS CALIBRATION FROM VALIDATION")
                        printer.print(f"{'='*60}")

                        # Learn calibration factors
                        # IMPORTANT: Use segment_id for grouping, NOT model_level
                        # This ensures key-level models still get segment-level calibration
                        val_period_str = f"{val_df_all[date_col].min()}-{val_df_all[date_col].max()}"

                        calibration = learn_bias_calibration(
                            val_predictions_df=val_predictions,
                            key_metadata_df=key_metadata_df,
                            n_buckets=bias_calibration_buckets,
                            pred_col='predicted',
                            actual_col='actual',
                            key_col=key_col,
                            model_level_col='model_level',  # For logging only
                            zero_fraction_col='zero_fraction',
                            segment_col='segment_id',  # CRITICAL: Use segment for calibration grouping
                            min_samples_per_bucket=10,
                            val_period=val_period_str,
                            factor_min=bias_calibration_factor_min,
                            factor_max=bias_calibration_factor_max,
                        )

                        if calibration.enabled:
                            # Save calibration factors
                            calibration_path = os.path.join(model_dir, 'bias_calibration.json')
                            save_calibration_factors(calibration, calibration_path)

                            # Store for inclusion in final_specs
                            bias_calibration_data = calibration.to_dict()

                            # Print basic calibration info first
                            printer.print(f"Learned {len(calibration.factors)} calibration factors")
                            printer.print(f"Global bias factor: {calibration.global_factor:.3f}")

                            # Compute validation WAPE before and after calibration
                            try:
                                from utils.bias_calibration import apply_bias_calibration

                                # Metrics before calibration
                                sum_actual = float(val_predictions['actual'].sum())
                                sum_pred_before = float(val_predictions['predicted'].sum())
                                bias_before_pct = (sum_pred_before - sum_actual) / max(sum_actual, 1) * 100
                                # WAPE = sum(|actual - predicted|) / sum(actual) - row-level absolute errors
                                abs_errors_before = (val_predictions['actual'] - val_predictions['predicted']).abs().sum()
                                wape_before = float(abs_errors_before / max(sum_actual, 1))

                                # Apply calibration to validation predictions
                                val_predictions_calibrated = apply_bias_calibration(
                                    predictions_df=val_predictions.copy(),
                                    calibration=calibration,
                                    key_metadata_df=key_metadata_df,
                                    pred_col='predicted',
                                    key_col=key_col,
                                    segment_col='segment_id',
                                )

                                # Metrics after calibration
                                sum_pred_after = float(val_predictions_calibrated['predicted'].sum())
                                bias_after_pct = (sum_pred_after - sum_actual) / max(sum_actual, 1) * 100
                                # WAPE after calibration - row-level absolute errors
                                abs_errors_after = (val_predictions['actual'] - val_predictions_calibrated['predicted']).abs().sum()
                                wape_after = float(abs_errors_after / max(sum_actual, 1))

                                printer.print(f"Validation bias BEFORE calibration: {bias_before_pct:.1f}%")
                                printer.print(f"Validation bias AFTER calibration: {bias_after_pct:.1f}%")
                                printer.print(f"Validation WAPE: {wape_before:.4f} -> {wape_after:.4f} (after calibration)")
                            except Exception as post_cal_err:
                                logger.warning(f"Post-calibration metrics failed: {post_cal_err}")
                                import traceback
                                logger.debug(traceback.format_exc())
                                # Still print basic info from calibration stats
                                if calibration.stats.get('global_bias_pct'):
                                    printer.print(f"Validation bias: {calibration.stats['global_bias_pct']:.1f}%")
                        else:
                            printer.print("Calibration learning failed - see logs")
                    else:
                        printer.print("Bias calibration disabled in config - skipping")
                else:
                    printer.print("No validation predictions generated")
            else:
                printer.print("No trained models available for validation predictions")

        except ImportError as ie:
            logger.warning(f"Bias calibration module not available: {ie}")
        except Exception as cal_err:
            # CRITICAL: Log the full error - this is essential for debugging
            logger.error(f"VALIDATION PREDICTIONS/CALIBRATION FAILED: {cal_err}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            printer.print(f"ERROR: Validation predictions failed - {cal_err}")

        # =====================================================================
        # STATE-OF-THE-ART: SEGMENT-AWARE BIAS CALIBRATION
        # This is a more sophisticated calibration that learns per-segment
        # correction factors based on demand patterns and zero fractions
        # =====================================================================
        segment_calibrations = {}
        if segment_calibration_data and len(segment_calibration_data) > 0:
            try:
                # Build segment profiles for calibration
                segment_profiles_for_cal = {}
                for mg_id, mg_config_item in model_groups_config.items():
                    segment_profiles_for_cal[mg_id] = {
                        'demand_pattern': mg_config_item.get('demand_pattern', 'smooth'),
                    }

                # Compute segment-aware calibrations
                segment_calibrations = compute_all_segment_calibrations(
                    segment_predictions=segment_calibration_data,
                    segment_profiles=segment_profiles_for_cal,
                )

                if segment_calibrations:
                    # Save segment calibrations
                    segment_cal_path = os.path.join(model_dir, 'segment_calibrations.json')
                    segment_cal_data = {
                        seg_id: {
                            'segment_id': cal.segment_id,
                            'demand_pattern': cal.demand_pattern,
                            'global_factor': cal.global_factor,
                            'calibration_factors': cal.calibration_factors,
                            'n_keys_calibrated': cal.n_keys_calibrated,
                            'avg_bias_before': cal.avg_bias_before,
                        }
                        for seg_id, cal in segment_calibrations.items()
                    }
                    save_json(segment_cal_data, segment_cal_path)
                    printer.print(f"STATE-OF-THE-ART: Learned segment-aware calibration for {len(segment_calibrations)} segments")

                    # Log summary of calibration factors
                    factors_summary = [
                        f"{seg_id}: {cal.global_factor:.3f}"
                        for seg_id, cal in list(segment_calibrations.items())[:3]
                    ]
                    if factors_summary:
                        printer.print(f"  Sample factors: {', '.join(factors_summary)}")

            except Exception as seg_cal_err:
                logger.warning(f"Segment-aware calibration failed: {seg_cal_err}")
                import traceback
                logger.debug(traceback.format_exc())

        # WAPE INTERPRETATION (Adjusted for Intermittent Demand & Multi-Step Horizon):
        # For highly intermittent demand at granular level with 8-week forward horizon,
        # WAPE thresholds are adjusted upward. WAPE < 100% is considered acceptable
        # because intermittent time series have many zeros and inherently high error rates.
        #
        # WAPE is expressed as a ratio where 0.0 = perfect, 1.0 = 100% error
        # - WAPE < 0.30 (30%): EXCELLENT - exceptional for intermittent demand
        # - WAPE 0.30-0.50 (30-50%): GOOD - strong performance for multi-step forecasting
        # - WAPE 0.50-0.70 (50-70%): FAIR - acceptable for intermittent demand
        # - WAPE 0.70-1.00 (70-100%): ACCEPTABLE - expected for highly intermittent series
        # - WAPE 1.00-1.50 (100-150%): POOR - higher than expected, review needed
        # - WAPE > 1.50 (150%+): CRITICAL - significantly worse than baseline
        def interpret_wape(wape_value: float) -> tuple:
            """Return (status, severity, description) adjusted for intermittent demand."""
            wape_pct = wape_value * 100
            if wape_value < 0.30:
                return ('excellent', 'info', f'{wape_pct:.1f}% error - exceptional for intermittent demand')
            elif wape_value < 0.50:
                return ('good', 'info', f'{wape_pct:.1f}% error - strong multi-step performance')
            elif wape_value < 0.70:
                return ('fair', 'info', f'{wape_pct:.1f}% error - acceptable for intermittent demand')
            elif wape_value < 1.00:
                return ('acceptable', 'info', f'{wape_pct:.1f}% error - expected for highly intermittent series')
            elif wape_value < 1.50:
                return ('poor', 'warning', f'{wape_pct:.1f}% error - higher than expected, review models')
            else:
                return ('critical', 'error', f'{wape_pct:.1f}% error - significantly worse than baseline')

        wape_status, wape_severity, wape_description = interpret_wape(overall_wape)

        # Log warning for poor performance
        if wape_severity in ('warning', 'error', 'critical'):
            logger.warning(f"TRAINING QUALITY ALERT: Overall WAPE = {overall_wape:.4f} ({wape_description})")
            printer.print(f"\n⚠️  WARNING: {wape_description}")

        # Interpret test WAPE if available
        test_wape_status = None
        test_wape_description = None
        if overall_test_wape is not None:
            test_wape_status, _, test_wape_description = interpret_wape(overall_test_wape)

        # =====================================================================
        # IDENTIFY BEST MODEL FOR GLOBAL/SINGLE-MODEL PIPELINE USE
        # =====================================================================
        # Find the best performing model spec (by validation WAPE)
        trained_specs = [s for s in model_specs if s.get('status') == 'trained']
        best_model_spec = None
        if trained_specs:
            best_model_spec = min(trained_specs, key=lambda s: s.get('val_wape', float('inf')))

        # Create "global" spec for pipelines that need a single model configuration
        # This is what the deterministic pipeline generators will use
        global_spec = {}
        if best_model_spec:
            is_ensemble = best_model_spec.get('is_ensemble', False)
            ensemble_info = best_model_spec.get('ensemble_info', {})

            feature_cols_list = best_model_spec.get('feature_columns', [])
            # Check if this is a multi-horizon model
            model_type = best_model_spec.get('model_type', 'lightgbm')
            is_multi_horizon = model_type.startswith('multi_horizon')

            # Extract multi-horizon config if applicable
            multi_horizon_config = {}
            if is_multi_horizon:
                multi_horizon_config = {
                    'strategy': best_model_spec.get('strategy', 'direct_separate'),
                    'max_horizon': best_model_spec.get('max_horizon', forecast_horizon),
                    'target_horizon': best_model_spec.get('target_horizon', forecast_horizon),
                    'horizon_weights': best_model_spec.get('horizon_weights', {}),
                }

            global_spec = {
                'model_type': model_type,
                'hyperparameters': best_model_spec.get('hyperparameters', {}),
                'feature_columns': feature_cols_list,
                'n_features': len(feature_cols_list),
                'n_features_by_category': _categorize_feature_counts(feature_cols_list),
                'target_column': target_col,
                'key_column': key_col,
                'date_column': date_col,
                'forecast_horizon': forecast_horizon,
                'validation_wape': best_model_spec.get('val_wape'),
                'test_wape': best_model_spec.get('test_wape'),
                'model_path': best_model_spec.get('model_path'),
                'is_ensemble': is_ensemble,
                'best_model_group': best_model_spec.get('model_group'),
                # ENSEMBLE INFO - for pipeline to retrain ensemble properly
                'ensemble_info': ensemble_info if is_ensemble else {},
                # For ensembles: model_types, weights are in ensemble_info
                # For ensembles: member hyperparameters are in hyperparameters['member_params']
                # MULTI-HORIZON INFO - for pipeline to retrain multi-horizon models properly
                'is_multi_horizon': is_multi_horizon,
                'multi_horizon_config': multi_horizon_config if is_multi_horizon else {},
            }

        # Save final model specs with interpreted status
        final_specs = {
            # TOP-LEVEL GLOBAL SPEC - for deterministic pipeline generators
            # Pipeline can read model_type, hyperparameters, feature_columns directly
            'model_type': global_spec.get('model_type', 'lightgbm'),
            'hyperparameters': global_spec.get('hyperparameters', {}),
            'feature_columns': global_spec.get('feature_columns', []),
            'n_features': global_spec.get('n_features', 0),
            'n_features_by_category': global_spec.get('n_features_by_category', {}),
            'target_column': target_col,
            'key_column': key_col,
            'date_column': date_col,
            'forecast_horizon': forecast_horizon,
            'validation_wape': global_spec.get('validation_wape'),
            'test_wape': global_spec.get('test_wape'),
            'best_model_group': global_spec.get('best_model_group'),
            'best_model_path': global_spec.get('model_path'),
            # ENSEMBLE SUPPORT - pipeline needs to know if this is an ensemble
            'is_ensemble': global_spec.get('is_ensemble', False),
            'ensemble_info': global_spec.get('ensemble_info', {}),
            # MULTI-HORIZON SUPPORT - pipeline needs to know if this is a multi-horizon model
            # Multi-horizon models train separate models for each forecast horizon (Lag 1, 2, 3, 4, 5)
            # This eliminates error compounding from recursive forecasting
            'is_multi_horizon': global_spec.get('is_multi_horizon', False),
            'multi_horizon_config': global_spec.get('multi_horizon_config', {}),
            # For multi-horizon retraining, pipeline should:
            # 1. Read multi_horizon_config['strategy'] (direct_separate, horizon_weighted, etc.)
            # 2. Read multi_horizon_config['max_horizon'] for how many horizon models to train
            # 3. Read multi_horizon_config['target_horizon'] for primary evaluation horizon
            # 4. Read multi_horizon_config['horizon_weights'] for loss weighting
            # 5. Use train_multi_horizon_* functions from multi_horizon_training module
            # For ensemble retraining, pipeline should:
            # 1. Read ensemble_info['model_types'] to know which models to train
            # 2. Read hyperparameters['member_params'][model_type] for each member's params
            # 3. Read ensemble_info['weights'] for combination weights
            # 4. Retrain each member model, then combine with saved weights
            # Per-model specs (for multi-model pipelines)
            'models': model_specs,
            # Validation metrics (used for model selection)
            'overall_val_wape': overall_wape,
            'overall_val_wape_pct': overall_wape * 100,
            'val_wape_status': wape_status,
            'val_wape_severity': wape_severity,
            'val_wape_interpretation': wape_description,
            # Test metrics (holdout evaluation)
            'overall_test_wape': overall_test_wape,
            'overall_test_wape_pct': overall_test_wape * 100 if overall_test_wape else None,
            'test_wape_status': test_wape_status,
            'test_wape_interpretation': test_wape_description,
            # Training summary
            'models_trained': models_trained,
            'models_failed': models_failed,
            # Report the EFFECTIVE flag, not the raw user setting — when
            # DMH auto-skips the walk-forward, downstream metadata should
            # reflect that "no recursive eval was performed."
            'recursive_validation_used': _effective_recursive_validation,
            # Candidate comparisons
            'candidate_comparisons': all_candidate_comparisons,
            # Config
            'config': {
                'enable_meta_learning': enable_meta_learning,
                'enable_ensemble_optimization': enable_ensemble_optimization,
                'enable_bias_correction': enable_bias_correction,
                'enable_forecast_calibration': enable_forecast_calibration,
                'forecast_horizon': forecast_horizon,
                'use_recursive_validation': use_recursive_validation,
                'ensemble_top_k': ensemble_top_k,
            },
            # STATE-OF-THE-ART: EDA-driven configuration summary from strategy
            'eda_driven_config': strategy.get('eda_driven_config', {}),
            'eda_insights_for_training': strategy.get('eda_insights_for_training', {}),
            'loss_function_distribution': strategy.get('segmentation_context_summary', {}).get('loss_function_distribution', {}),
            'validation_strategy_distribution': strategy.get('segmentation_context_summary', {}).get('validation_strategy_distribution', {}),
            # KEY-TO-MODEL ROUTING - Critical for pipeline to know which model to use for each key
            'key_to_model_group': key_to_model_group,  # {key: model_group}
            'model_group_to_keys': model_group_to_keys,  # {model_group: [keys]}
            'n_keys': len(key_to_model_group),
            'n_model_groups': len(model_group_to_keys),
            # Model level column name (for referencing in data)
            'model_level_column': ml_col,
            # BIAS CALIBRATION - Learned correction factors for systematic bias
            # Apply these factors during inference: calibrated_pred = raw_pred * factor
            'bias_calibration': bias_calibration_data,
            'bias_calibration_enabled': bias_calibration_data is not None and bias_calibration_data.get('enabled', False),
            # STATE-OF-THE-ART: Segment-aware calibration factors
            # Each segment has its own calibration factor based on demand pattern and zero fraction
            'segment_calibrations': {
                seg_id: {
                    'global_factor': cal.global_factor,
                    'demand_pattern': cal.demand_pattern,
                    'calibration_factors': cal.calibration_factors,
                }
                for seg_id, cal in segment_calibrations.items()
            } if segment_calibrations else {},
            'segment_calibration_enabled': len(segment_calibrations) > 0,
            # STATE-OF-THE-ART: Segment-specific hyperparameter profiles used
            'segment_hyperparam_profiles': {
                seg_id: profile.to_dict()
                for seg_id, profile in segment_hyperparam_profiles.items()
            } if segment_hyperparam_profiles else {},
            'state_of_art_features_used': {
                'segment_specific_hyperparams': len(segment_hyperparam_profiles) > 0,
                'pattern_aware_ensemble': enable_ensemble_optimization,
                'enhanced_meta_learning': enable_meta_learning,
                'adaptive_threshold_calibration': True,  # Always attempted for ZI/Hurdle
                'segment_aware_bias_calibration': len(segment_calibrations) > 0,
            },
        }
        final_specs_path = os.path.join(model_dir, 'final_model_specs.json')
        save_json(final_specs, final_specs_path)

        # Log what was saved for pipeline consumption
        logger.info(f"Saved final_model_specs.json with global spec: model_type={global_spec.get('model_type')}, "
                    f"n_features={len(global_spec.get('feature_columns', []))}, "
                    f"hyperparameters={bool(global_spec.get('hyperparameters'))}")

        # Create diagnostic context using ContextBuilder for self-documenting schema
        from utils.context_schema import ContextBuilder, SemanticTypes

        diag_ctx = ContextBuilder(
            context_type='training_to_diagnostic',
            source_crew='training_crew',
            target_crews=['diagnostic_crew']
        )

        diag_ctx.add_field(
            key='overall_performance',
            value={
                'overall_wape': overall_wape,
                'overall_wape_pct': overall_wape * 100,
                'wape_status': wape_status,
                'wape_severity': wape_severity,
                'wape_interpretation': wape_description,
                'models_trained': models_trained,
                'models_failed': models_failed,
                'training_status': 'completed' if wape_severity != 'critical' else 'completed_with_critical_issues',
            },
            description='Overall training performance metrics including WAPE, model counts, and status',
            semantic_type=SemanticTypes.TRAINING_RESULTS,
            required=True,
            required_by=['diagnostic_crew']
        )

        # Interpret per-model WAPE
        # NOTE: model specs use 'val_wape' and 'test_wape', not just 'wape'
        model_performance_list = []
        for s in model_specs:
            # Use val_wape as the primary metric (test_wape is for holdout evaluation)
            model_wape = s.get('val_wape', s.get('wape', 1.0))
            test_wape = s.get('test_wape', None)
            m_status, m_severity, m_desc = interpret_wape(model_wape) if model_wape is not None else ('failed', 'error', 'No WAPE computed')
            model_performance_list.append({
                'model_group': s['model_group'],
                'model_type': s.get('model_type', 'failed'),
                'val_wape': model_wape,
                'val_wape_pct': model_wape * 100 if model_wape is not None else None,
                'test_wape': test_wape,
                'test_wape_pct': test_wape * 100 if test_wape is not None else None,
                'wape_status': m_status,
                'wape_severity': m_severity,
                'status': s.get('status', 'failed'),
                'n_train': s.get('n_train_samples'),
                'n_val': s.get('n_val_samples'),
                'n_features': s.get('n_features'),
                'demand_pattern': s.get('demand_pattern'),
            })

        diag_ctx.add_field(
            key='model_performance',
            value=model_performance_list,
            description='Per-model performance metrics including WAPE and status',
            semantic_type=SemanticTypes.MODEL_PERFORMANCE,
            required=True,
            required_by=['diagnostic_crew']
        )

        # Identify underperformers with severity-based thresholds
        # ADJUSTED FOR INTERMITTENT DEMAND: Higher thresholds since intermittent series
        # inherently have high WAPE due to many zeros and sparse events.
        # Use val_wape for underperformer analysis
        critical_groups = [s['model_group'] for s in model_specs if s.get('val_wape', s.get('wape', 1.0)) >= 1.5]  # 150%+
        poor_groups = [s['model_group'] for s in model_specs if 1.0 <= s.get('val_wape', s.get('wape', 1.0)) < 1.5]  # 100-150%
        acceptable_groups = [s['model_group'] for s in model_specs if 0.7 <= s.get('val_wape', s.get('wape', 1.0)) < 1.0]  # 70-100%
        underperformer_analysis = {
            'critical_wape_groups': critical_groups,  # WAPE >= 150% (truly poor for intermittent)
            'poor_wape_groups': poor_groups,  # WAPE 100-150% (needs review)
            'acceptable_wape_groups': acceptable_groups,  # WAPE 70-100% (normal for intermittent)
            'total_critical': len(critical_groups),
            'total_poor': len(poor_groups),
            'total_acceptable': len(acceptable_groups),
            'total_flagged': len(critical_groups) + len(poor_groups),  # Only flag truly problematic ones
        }
        diag_ctx.add_field(
            key='underperformer_analysis',
            value=underperformer_analysis,
            description='Analysis of underperforming models by severity (critical >=100%, very_poor 80-100%, poor 50-80%)',
            semantic_type=SemanticTypes.UNDERPERFORMER_ANALYSIS,
        )

        # Build recommendations based on severity (ADJUSTED FOR INTERMITTENT DEMAND)
        # For intermittent demand with 8-week horizon, WAPE < 100% is acceptable
        recommendations = []
        if critical_groups:
            recommendations.append(
                f"CRITICAL: {len(critical_groups)} models have WAPE >= 150% - "
                "significantly worse than baseline, investigate data quality and feature relevance"
            )
        if poor_groups:
            recommendations.append(
                f"REVIEW: {len(poor_groups)} models have WAPE 100-150% - "
                "higher than expected for intermittent demand, consider model tuning"
            )
        # For intermittent demand, only flag if WAPE > 150% as truly critical
        if overall_wape > 1.5:
            recommendations.insert(0,
                f"🚨 OVERALL WAPE = {overall_wape*100:.1f}% (>150%) - "
                "Models are significantly underperforming. Review data quality and model selection."
            )
        elif overall_wape > 1.0:
            recommendations.append(
                f"NOTE: Overall WAPE = {overall_wape*100:.1f}% (>100%) - "
                "This is expected for highly intermittent demand with multi-step forecasting."
            )
        failed_groups = [s for s in model_specs if s.get('status') == 'failed']
        if failed_groups:
            recommendations.append(
                f"{len(failed_groups)} model groups failed to train - review data quality and model compatibility"
            )

        diag_ctx.add_field(
            key='diagnostic_priorities',
            value=recommendations,
            description='Priority list for diagnostic crew based on training results severity',
            semantic_type=SemanticTypes.DIAGNOSTIC_PRIORITIES,
        )

        diag_ctx.add_metadata(run_id=f'train_{models_trained}_models')
        diagnostic_path = os.path.join(model_dir, 'training_to_diagnostic_context.json')
        diag_ctx.save(diagnostic_path)

        # =====================================================================
        # STATE-OF-THE-ART: Aggregate Walk-Forward CV summary across all segments
        # =====================================================================
        wf_cv_summary = {}
        wf_cv_drift_segments = []
        wf_cv_any_enabled = False

        for spec in model_specs:
            if spec.get('walk_forward_cv') and spec['walk_forward_cv'].get('enabled'):
                wf_cv_any_enabled = True
                mg = spec['model_group']
                cv_data = spec['walk_forward_cv']
                wf_cv_summary[mg] = {
                    'best_model': cv_data.get('best_model'),
                    'avg_wape': cv_data.get('best_avg_wape'),
                    'std_wape': cv_data.get('best_std_wape'),
                    'expected_range': cv_data.get('expected_wape_range'),
                    'drift_detected': cv_data.get('concept_drift_detected', False),
                }
                if cv_data.get('concept_drift_detected'):
                    wf_cv_drift_segments.append(mg)

        # Log Walk-Forward CV summary
        if wf_cv_any_enabled:
            n_with_cv = len(wf_cv_summary)
            n_with_drift = len(wf_cv_drift_segments)
            avg_cv_wape = np.mean([v['avg_wape'] for v in wf_cv_summary.values() if v['avg_wape'] is not None])
            printer.print(f"\nWalk-Forward CV: {n_with_cv} segments evaluated, avg CV WAPE={avg_cv_wape:.4f}")
            if n_with_drift > 0:
                printer.print(f"  ⚠️  Concept drift detected in {n_with_drift} segments: {wf_cv_drift_segments[:3]}")

        # Clear status reporting with WAPE interpretation.  Reflect the
        # EFFECTIVE recursive-validation status (the auto-skip when DMH
        # is on is the headline speedup we want visible at the end of
        # training too, not just at the start).
        printer.print(f"\n{'='*60}")
        printer.print(f"TRAINING COMPLETE: {models_trained}/{len(model_groups)} models")
        if use_recursive_validation and use_direct_multi_horizon:
            printer.print(
                f"Forecast Horizon: {forecast_horizon} steps, "
                f"Recursive Validation: AUTO-SKIPPED "
                f"(use_direct_multi_horizon=True; ~13x training speedup applied)"
            )
        else:
            printer.print(
                f"Forecast Horizon: {forecast_horizon} steps, "
                f"Recursive Validation: {_effective_recursive_validation}"
            )
        if wf_cv_any_enabled:
            printer.print(f"Walk-Forward CV: {len(wf_cv_summary)} segments with temporal validation")
        printer.print(f"Validation WAPE: {overall_wape:.4f} ({overall_wape*100:.1f}%)")
        if overall_test_wape is not None:
            printer.print(f"Test WAPE: {overall_test_wape:.4f} ({overall_test_wape*100:.1f}%)")
        printer.print(f"Status: {wape_status.upper()} - {wape_description}")
        if recommendations:
            printer.print(f"\nDiagnostic Priorities:")
            for rec in recommendations[:3]:
                printer.print(f"  • {rec[:100]}...")
        printer.print(f"{'='*60}")

        return FullTrainingResult(
            models_trained=models_trained,
            models_failed=models_failed,
            overall_wape=overall_wape,
            overall_test_wape=overall_test_wape if overall_test_wape is not None else 0.0,
            model_specs=model_specs,
            final_specs_path=final_specs_path,
            diagnostic_context_path=diagnostic_path,
            success=True,
            forecast_horizon=forecast_horizon,
            # Effective flag, see same comment further up.
            recursive_validation_used=_effective_recursive_validation,
            candidate_comparisons=all_candidate_comparisons,
            # STATE-OF-THE-ART: Walk-Forward CV summary
            walk_forward_cv_enabled=wf_cv_any_enabled,
            walk_forward_cv_summary=wf_cv_summary,
            concept_drift_detected=len(wf_cv_drift_segments) > 0,
            drift_affected_segments=wf_cv_drift_segments,
        )

    except Exception as e:
        error_msg = f"Training pipeline failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        printer.print(f"TRAINING FAILED: {str(e)[:100]}")

        return FullTrainingResult(
            models_trained=0,
            models_failed=0,
            overall_wape=1.0,
            overall_test_wape=0.0,
            success=False,
            error_message=error_msg,
        )


# =============================================================================
# MULTI-STEP RECURSIVE EVALUATION (LEAKAGE-FREE)
# =============================================================================

@dataclass
class RecursiveEvalResult:
    """Result of recursive multi-step evaluation."""
    wape: float
    mae: float
    rmse: float
    bias: float
    predictions: np.ndarray
    actuals: np.ndarray
    n_samples: int
    n_keys: int
    horizon: int
    wape_by_step: Dict[int, float] = field(default_factory=dict)  # WAPE for each forecast step


def _recursive_forecast_univariate(
    model: dict,
    historical_actuals: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Produce multi-step recursive forecasts for Croston-family univariate models.

    Instead of repeating a flat constant, this re-estimates the smoothing state
    after each prediction step:
      1. Compute current forecast from model state
      2. Append prediction to the running history
      3. Re-run exponential smoothing on extended history to update state
      4. Repeat for all horizon steps

    Supports: croston, sba, tsb, imapa.
    """
    model_type = model.get('type', 'croston')
    alpha = model.get('alpha', 0.1)
    history = np.asarray(historical_actuals, dtype=float).flatten().copy()

    predictions = np.empty(horizon, dtype=float)

    for step in range(horizon):
        if model_type in ('croston', 'sba'):
            # --- Croston smoothing on current history ---
            non_zero_idx = np.where(history > 0)[0]
            if len(non_zero_idx) == 0:
                demand_level = 0.0
                interval = float('inf')
            else:
                demand_level = float(history[non_zero_idx[0]])
                interval = float(non_zero_idx[0] + 1) if non_zero_idx[0] > 0 else 1.0
                for i in range(1, len(non_zero_idx)):
                    idx = non_zero_idx[i]
                    prev_idx = non_zero_idx[i - 1]
                    demand_level = alpha * history[idx] + (1 - alpha) * demand_level
                    inter_arrival = float(idx - prev_idx)
                    interval = alpha * inter_arrival + (1 - alpha) * interval

            if interval == 0 or interval == float('inf') or demand_level == 0:
                forecast = 0.0
            else:
                forecast = demand_level / interval

            if model_type == 'sba':
                forecast *= (1 - alpha / 2)

            forecast = max(0.0, forecast)

        elif model_type == 'tsb':
            # --- TSB smoothing on current history ---
            beta = model.get('beta', 0.1)
            non_zero = history[history > 0]
            if len(non_zero) == 0:
                demand_level = 0.0
                demand_prob = 0.0
            else:
                demand_level = float(np.mean(non_zero))
                demand_prob = float(len(non_zero)) / len(history)
                for val in history:
                    if val > 0:
                        demand_level = alpha * val + (1 - alpha) * demand_level
                        demand_prob = alpha * 1.0 + (1 - alpha) * demand_prob
                    else:
                        demand_prob = (1 - beta) * demand_prob
            forecast = max(0.0, demand_level * demand_prob)

        elif model_type == 'imapa':
            # --- IMAPA: multi-level aggregation with re-estimation ---
            aggregation_levels = model.get('aggregation_levels', [1, 2, 3])
            base_method = model.get('base_method', 'croston')
            level_forecasts = []
            for level in aggregation_levels:
                n_complete = (len(history) // level) * level
                if n_complete < level:
                    continue
                y_agg = history[:n_complete].reshape(-1, level).sum(axis=1)
                # Croston on aggregated series
                nz = np.where(y_agg > 0)[0]
                if len(nz) == 0:
                    agg_fc = 0.0
                else:
                    dl = float(y_agg[nz[0]])
                    iv = float(nz[0] + 1) if nz[0] > 0 else 1.0
                    for i in range(1, len(nz)):
                        dl = alpha * y_agg[nz[i]] + (1 - alpha) * dl
                        ia = float(nz[i] - nz[i - 1])
                        iv = alpha * ia + (1 - alpha) * iv
                    agg_fc = dl / iv if iv > 0 else 0.0
                    if base_method == 'sba':
                        agg_fc *= (1 - alpha / 2)
                base_fc = agg_fc / level
                w = float(np.mean(y_agg > 0))
                level_forecasts.append((base_fc, w))
            total_w = sum(w for _, w in level_forecasts) if level_forecasts else 1.0
            if total_w > 0 and level_forecasts:
                forecast = sum(f * w / total_w for f, w in level_forecasts)
            else:
                forecast = 0.0
            forecast = max(0.0, float(forecast))

        else:
            # Fallback: use stored flat forecast
            forecast = float(model.get('forecast', 0.0))

        predictions[step] = forecast
        # Extend history with the prediction for the next step
        history = np.append(history, forecast)

    return predictions


def evaluate_model_recursive(
    model: Any,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    key_col: str = 'key',
    date_col: str = None,
    forecast_horizon: int = 8,
    time_format: str = 'year_week',
) -> RecursiveEvalResult:
    """
    Evaluate a model using proper recursive multi-step forecasting.

    This function computes WAPE without feature leakage by:
    1. For each key, using only historical actuals up to the forecast origin
    2. Generating predictions step-by-step from t+1 to t+forecast_horizon
    3. Updating lag/rolling features recursively using predictions (not actuals)

    Parameters
    ----------
    model : fitted model
        Must have predict() method
    train_df : pd.DataFrame
        Training data (historical actuals)
    eval_df : pd.DataFrame
        Evaluation data (validation or test)
    feature_columns : List[str]
        Feature columns used by the model
    target_col : str
        Target column name
    key_col : str
        Key column name
    date_col : str
        Date column name
    forecast_horizon : int
        Number of steps to forecast (t+1 to t+horizon)

    Returns
    -------
    RecursiveEvalResult
        Comprehensive evaluation results
    """
    from utils.recursive_forecasting import (
        RecursiveForecaster, RecursiveForecastConfig,
        identify_lag_columns, identify_rolling_columns
    )

    # Resolve date_col from time_format if not provided
    if date_col is None:
        date_col = 'year_month' if time_format == 'year_month' else 'year_week'

    # Check if model is a univariate dict (croston, sba, tsb, imapa, etc.)
    # These models use recursive re-estimation: at each forecast step, re-fit
    # the smoothing on (actuals + prior predictions) to produce evolving forecasts.
    is_univariate_dict = isinstance(model, dict) and 'forecast' in model

    if not is_univariate_dict:
        # Identify lag and rolling columns for recursive updating
        lag_columns = identify_lag_columns(feature_columns, target_col)
        rolling_columns = identify_rolling_columns(feature_columns, target_col)
        static_columns = [c for c in feature_columns if c not in lag_columns and c not in rolling_columns]

        logger.info(f"Recursive eval: {len(lag_columns)} lag cols, {len(rolling_columns)} rolling cols")

        # Create forecaster
        config = RecursiveForecastConfig(clip_min=0.0, time_format=time_format)
        forecaster = RecursiveForecaster(
            model=model,
            target_column=target_col,
            lag_columns=lag_columns,
            rolling_columns=rolling_columns,
            static_columns=static_columns,
            config=config,
        )
    else:
        forecaster = None
        logger.info(f"Recursive eval: univariate model ({model.get('type', 'unknown')}), "
                     f"using recursive re-estimation over {forecast_horizon} steps")

    # Get unique keys
    unique_keys = eval_df[key_col].unique()

    all_predictions = []
    all_actuals = []
    step_predictions = {i: [] for i in range(1, forecast_horizon + 1)}
    step_actuals = {i: [] for i in range(1, forecast_horizon + 1)}

    for key in unique_keys:
        # Get training history for this key
        key_train = train_df[train_df[key_col] == key].sort_values(date_col)
        key_eval = eval_df[eval_df[key_col] == key].sort_values(date_col)

        if len(key_train) == 0 or len(key_eval) == 0:
            continue

        # Historical actuals (known at forecast time)
        historical_actuals = key_train[target_col].values

        # Limit evaluation to forecast_horizon periods
        key_eval = key_eval.head(forecast_horizon)

        if is_univariate_dict:
            # Univariate: recursive re-estimation at each forecast step
            predictions = _recursive_forecast_univariate(
                model=model,
                historical_actuals=historical_actuals,
                horizon=len(key_eval),
            )
        else:
            # Feature-based: recursive forecasting with feature updates
            # Sanitize: replace inf/-inf with NaN, then fill with 0
            future_features = key_eval[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).values

            # Generate recursive predictions
            predictions = forecaster.forecast(
                historical_actuals=historical_actuals,
                future_features=future_features,
                feature_names=feature_columns,
                horizon=len(key_eval),
            )

        actuals = key_eval[target_col].values

        all_predictions.extend(predictions)
        all_actuals.extend(actuals)

        # Track by step
        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            step_num = i + 1
            if step_num <= forecast_horizon:
                step_predictions[step_num].append(pred)
                step_actuals[step_num].append(actual)

    # Compute overall metrics
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    total_actuals = np.sum(np.abs(all_actuals))
    total_errors = np.sum(np.abs(all_predictions - all_actuals))

    wape = total_errors / total_actuals if total_actuals > 0 else 1.0
    mae = np.mean(np.abs(all_predictions - all_actuals))
    rmse = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
    bias = np.mean(all_predictions - all_actuals)

    # Compute WAPE by forecast step
    wape_by_step = {}
    for step in range(1, forecast_horizon + 1):
        if step_actuals[step]:
            step_total = np.sum(np.abs(step_actuals[step]))
            step_error = np.sum(np.abs(np.array(step_predictions[step]) - np.array(step_actuals[step])))
            wape_by_step[step] = step_error / step_total if step_total > 0 else 1.0

    return RecursiveEvalResult(
        wape=wape,
        mae=mae,
        rmse=rmse,
        bias=bias,
        predictions=all_predictions,
        actuals=all_actuals,
        n_samples=len(all_actuals),
        n_keys=len(unique_keys),
        horizon=forecast_horizon,
        wape_by_step=wape_by_step,
    )


# =============================================================================
# ENSEMBLE MODEL CREATION
# =============================================================================

@dataclass
class EnsembleModel:
    """Ensemble model that combines multiple base models."""
    models: List[Any]
    weights: List[float]
    model_types: List[str]
    feature_columns: List[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")

        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            # Handle univariate dict models (croston, sba, tsb, imapa)
            if isinstance(model, dict) and 'forecast' in model:
                preds = np.full(len(X), float(model['forecast']))
            else:
                preds = model.predict(X)
            predictions += weight * preds
        return np.clip(predictions, 0, None)


def create_ensemble_from_candidates(
    candidate_results: Dict[str, TrainingResult],
    feature_columns: List[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    top_k: int = 3,
    optimization_method: str = 'inverse_wape',
) -> Tuple[EnsembleModel, Dict[str, Any]]:
    """
    Create an ensemble model from top-k candidate models.

    Parameters
    ----------
    candidate_results : Dict[str, TrainingResult]
        Results from training each candidate model
    feature_columns : List[str]
        Feature column names
    X_val, y_val : np.ndarray
        Validation data for weight optimization
    top_k : int
        Number of models to include in ensemble
    optimization_method : str
        'inverse_wape': Weight by 1/WAPE
        'equal': Equal weights
        'optimized': Optimize weights using validation data

    Returns
    -------
    Tuple[EnsembleModel, Dict[str, Any]]
        Ensemble model and metadata about ensemble creation
    """
    # Sort by validation WAPE
    sorted_candidates = sorted(
        [(name, result) for name, result in candidate_results.items() if result is not None],
        key=lambda x: x[1].val_wape
    )

    # Take top-k
    top_candidates = sorted_candidates[:top_k]

    if len(top_candidates) == 0:
        raise ValueError("No valid candidates for ensemble")

    models = [result.model for name, result in top_candidates]
    model_types = [name for name, result in top_candidates]
    wapes = [result.val_wape for name, result in top_candidates]

    # Calculate weights
    if optimization_method == 'inverse_wape':
        # Weight inversely proportional to WAPE
        inverse_wapes = [1.0 / max(w, 0.01) for w in wapes]
        total = sum(inverse_wapes)
        weights = [w / total for w in inverse_wapes]

    elif optimization_method == 'equal':
        weights = [1.0 / len(models)] * len(models)

    elif optimization_method == 'optimized':
        # Optimize weights using validation data
        try:
            from scipy.optimize import minimize

            def ensemble_wape(w):
                w = np.abs(w) / np.sum(np.abs(w))  # Normalize
                preds = np.zeros(len(y_val))
                for model, weight in zip(models, w):
                    if isinstance(model, dict) and 'forecast' in model:
                        preds += weight * np.full(len(y_val), float(model['forecast']))
                    else:
                        preds += weight * model.predict(X_val)
                preds = np.clip(preds, 0, None)
                return compute_wape(y_val, preds)

            # Initial weights from inverse WAPE
            x0 = np.array([1.0 / max(w, 0.01) for w in wapes])
            x0 = x0 / x0.sum()

            result = minimize(ensemble_wape, x0, method='Nelder-Mead')
            weights = list(np.abs(result.x) / np.sum(np.abs(result.x)))
        except Exception as e:
            logger.warning(f"Weight optimization failed, using inverse WAPE: {e}")
            inverse_wapes = [1.0 / max(w, 0.01) for w in wapes]
            total = sum(inverse_wapes)
            weights = [w / total for w in inverse_wapes]
    else:
        weights = [1.0 / len(models)] * len(models)

    ensemble = EnsembleModel(
        models=models,
        weights=weights,
        model_types=model_types,
        feature_columns=feature_columns,
    )

    # Evaluate ensemble on validation
    ensemble_preds = ensemble.predict(X_val)
    ensemble_wape = compute_wape(y_val, ensemble_preds)

    metadata = {
        'n_models': len(models),
        'model_types': model_types,
        'weights': weights,
        'individual_wapes': wapes,
        'ensemble_wape': ensemble_wape,
        'optimization_method': optimization_method,
        'improvement_over_best': wapes[0] - ensemble_wape if wapes else 0,
    }

    logger.info(f"Created ensemble: {model_types} with weights {[f'{w:.3f}' for w in weights]}")
    logger.info(f"Ensemble WAPE: {ensemble_wape:.4f} vs best single: {wapes[0]:.4f}")

    return ensemble, metadata


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'TrainingResult',
    'TuningResult',
    'OptimizationResult',
    'RecursiveEvalResult',
    'EnsembleModel',
    'ZeroInflatedModel',
    'HurdleModel',

    # Metrics
    'compute_wape',
    'compute_mae',
    'compute_rmse',
    'compute_all_metrics',

    # Post-prediction zero clipping
    'apply_adi_based_zero_clip',
    'apply_batch_adi_zero_clip',
    'get_optimal_tweedie_power',

    # Recursive evaluation
    'evaluate_model_recursive',

    # Ensemble
    'create_ensemble_from_candidates',

    # Tree-based training
    'train_lightgbm',
    'train_xgboost',
    'train_catboost',
    'train_random_forest',

    # Intermittent specialists
    'train_croston',
    'train_sba',
    'train_tsb',
    'train_imapa',

    # Compound / hurdle
    'train_zero_inflated',
    'train_hurdle_model',
    'train_tweedie',

    # Classical statistical
    'train_arima',
    'train_sarima',
    'train_ets',
    'train_theta',
    'train_tbats',

    # Bayesian / probabilistic
    'train_prophet',
    'train_bsts',

    # Ensemble
    'train_weighted_ensemble',
    'train_stacking',

    # Deep learning (stubs)
    'train_tft',
    'train_lstm',
    'train_nbeats',
    'train_deepar',
    'train_wavenet',

    # Registry and utilities
    'TRAINING_REGISTRY',
    'FEATURE_BASED_MODELS',
    'UNIVARIATE_MODELS',
    'is_feature_based',
    'train_model_by_name',

    # High-level functions
    'train_best_model_for_segment',
    'train_all_model_groups',

    # Hyperparameter tuning
    'tune_lightgbm',
    'tune_xgboost',
    'tune_catboost',
    'tune_model_hyperparameters',

    # Post-training optimization
    'apply_post_training_optimization',
    'optimize_model_predictions',

    # Model persistence
    'save_model',
    'load_model',
    'save_training_result',

    # Full training pipeline (main entry point)
    'FullTrainingResult',
    'run_full_training_pipeline',

    # Multi-horizon training (optimized for longer horizons like Lag 5)
    'train_multi_horizon_lightgbm',
    'train_multi_horizon_xgboost',
    'train_multi_horizon_ensemble',
    'train_multi_horizon_model',
    'MultiHorizonTrainingResult',
    'MultiHorizonModel',
    'DEFAULT_HORIZON_WEIGHTS',
    'LAG5_FOCUSED_WEIGHTS',
    'MULTI_HORIZON_AVAILABLE',
]
