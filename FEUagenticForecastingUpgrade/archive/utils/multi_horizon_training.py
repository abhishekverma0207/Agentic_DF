"""
Multi-Horizon Direct Forecasting Training Module.
==================================================

This module implements direct multi-step forecasting that optimizes for
longer forecast horizons (e.g., Lag 5) rather than just 1-step-ahead predictions.

Key Insight:
Traditional recursive forecasting trains a 1-step-ahead model and applies it
recursively, but errors compound at longer horizons. Direct forecasting trains
separate models (or a single model with multi-output) to predict each horizon
directly, eliminating error compounding.

Horizon-Weighted Loss:
Since the user's forecasts are evaluated at Lag 5, we weight the loss function
to penalize errors at longer horizons more heavily:
- Lag 1: weight 0.1 (less important)
- Lag 2: weight 0.15
- Lag 3: weight 0.2
- Lag 4: weight 0.25
- Lag 5: weight 0.3 (most important)

Training Strategies:
1. Direct Multi-Output: Single model with multiple outputs for all horizons
2. Direct Separate: Separate model per horizon (ensemble at inference)
3. Horizon-Weighted Single: Single model trained with horizon-weighted samples

Usage:
    from utils.multi_horizon_training import (
        create_multi_horizon_dataset,
        train_multi_horizon_lightgbm,
        train_multi_horizon_xgboost,
        MultiHorizonModel,
        horizon_weighted_wape,
    )

    # Create multi-horizon targets
    X_multi, y_multi, horizon_weights = create_multi_horizon_dataset(
        X_train, y_train, max_horizon=5, target_horizon=5
    )

    # Train horizon-optimized model
    result = train_multi_horizon_lightgbm(
        X_train, y_train, X_val, y_val,
        max_horizon=5, target_horizon=5
    )

Author: HarmonIQ Demand-IQ System
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Static 5-horizon weights kept for backward compatibility
_DEFAULT_5_WEIGHTS = {
    1: 0.10, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.30,
}
_LAG5_FOCUSED_5_WEIGHTS = {
    1: 0.05, 2: 0.10, 3: 0.15, 4: 0.25, 5: 0.45,
}
_BALANCED_5_WEIGHTS = {
    1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20, 5: 0.20,
}


def build_horizon_weights(
    max_horizon: int,
    scheme: str = 'default',
    eval_lags: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Build horizon weights for any max_horizon, not just 5.

    Schemes
    -------
    'default'      : linearly increasing weights toward the last horizon.
    'eval_focused' : heavy weight on eval_lags (default lags 4-5),
                     remaining weight spread across other horizons.
    'balanced'     : equal weight for every horizon.

    Parameters
    ----------
    max_horizon : int
        Number of horizons (1 … max_horizon).
    scheme : str
        One of 'default', 'eval_focused', 'balanced'.
    eval_lags : list[int], optional
        Horizons to emphasise under 'eval_focused'. Defaults to
        the last two horizons [max_horizon-1, max_horizon].

    Returns
    -------
    Dict[int, float]  mapping horizon → weight (sums to 1.0).
    """
    if max_horizon <= 0:
        raise ValueError(f"max_horizon must be > 0, got {max_horizon}")

    if max_horizon == 1:
        return {1: 1.0}

    if scheme == 'balanced':
        w = 1.0 / max_horizon
        return {h: w for h in range(1, max_horizon + 1)}

    if scheme == 'eval_focused':
        if eval_lags is None:
            eval_lags = [max(1, max_horizon - 1), max_horizon]
        eval_lags = [l for l in eval_lags if 1 <= l <= max_horizon]
        if not eval_lags:
            eval_lags = [max_horizon]
        # 60 % to eval lags, 40 % to the rest
        eval_total = 0.60
        rest_total = 0.40
        n_eval = len(eval_lags)
        n_rest = max_horizon - n_eval
        weights = {}
        for h in range(1, max_horizon + 1):
            if h in eval_lags:
                weights[h] = eval_total / n_eval
            else:
                weights[h] = rest_total / max(n_rest, 1)
        return weights

    # 'default': linearly increasing
    raw = {h: h for h in range(1, max_horizon + 1)}
    total = sum(raw.values())
    return {h: v / total for h, v in raw.items()}


# Module-level defaults (5-horizon for backward compatibility)
DEFAULT_HORIZON_WEIGHTS = _DEFAULT_5_WEIGHTS
LAG5_FOCUSED_WEIGHTS = _LAG5_FOCUSED_5_WEIGHTS
BALANCED_HORIZON_WEIGHTS = _BALANCED_5_WEIGHTS


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MultiHorizonTrainingResult:
    """Result container for multi-horizon training."""
    model: Any  # Trained model or MultiHorizonModel wrapper
    model_type: str
    strategy: str  # 'direct_separate', 'direct_multi_output', 'horizon_weighted'
    horizon_weights: Dict[int, float]
    max_horizon: int
    target_horizon: int  # Primary evaluation horizon

    # Metrics per horizon
    train_wape_by_horizon: Dict[int, float]
    val_wape_by_horizon: Dict[int, float]

    # Weighted metrics (using horizon weights)
    train_weighted_wape: float
    val_weighted_wape: float

    # Target horizon metrics (the one that matters most)
    train_target_wape: float
    val_target_wape: float

    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    is_feature_based: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'model_type': self.model_type,
            'strategy': self.strategy,
            'horizon_weights': self.horizon_weights,
            'max_horizon': self.max_horizon,
            'target_horizon': self.target_horizon,
            'train_wape_by_horizon': self.train_wape_by_horizon,
            'val_wape_by_horizon': self.val_wape_by_horizon,
            'train_weighted_wape': self.train_weighted_wape,
            'val_weighted_wape': self.val_weighted_wape,
            'train_target_wape': self.train_target_wape,
            'val_target_wape': self.val_target_wape,
            'hyperparameters': self.hyperparameters,
            'training_time_seconds': self.training_time_seconds,
        }


class MultiHorizonModel:
    """
    Wrapper for multi-horizon forecasting models.

    This wrapper handles different training strategies:
    1. Direct Separate: Dictionary of {horizon: model}
    2. Direct Multi-Output: Single model with multi-output
    3. Horizon-Weighted Single: Single model trained with weighted samples

    The predict() method returns predictions for the target horizon by default,
    or predictions for all horizons when requested.
    """

    def __init__(
        self,
        models: Union[Any, Dict[int, Any]],
        strategy: str,
        max_horizon: int = 5,
        target_horizon: int = 5,
        horizon_weights: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize MultiHorizonModel.

        Parameters
        ----------
        models : Union[Any, Dict[int, Any]]
            For 'direct_separate': Dict mapping horizon to model
            For other strategies: Single model object
        strategy : str
            One of 'direct_separate', 'direct_multi_output', 'horizon_weighted'
        max_horizon : int
            Maximum forecast horizon
        target_horizon : int
            Primary horizon for evaluation (e.g., 5 for Lag 5)
        horizon_weights : Dict[int, float], optional
            Weights for each horizon (for weighted metrics)
        """
        self.models = models
        self.strategy = strategy
        self.max_horizon = max_horizon
        self.target_horizon = target_horizon
        self.horizon_weights = horizon_weights or DEFAULT_HORIZON_WEIGHTS
        self.model_type = 'multi_horizon'

    def predict(
        self,
        X: np.ndarray,
        horizon: Optional[int] = None,
        return_all_horizons: bool = False,
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Generate predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction
        horizon : int, optional
            Specific horizon to predict. If None, uses target_horizon.
        return_all_horizons : bool
            If True, return predictions for all horizons as dict.

        Returns
        -------
        Union[np.ndarray, Dict[int, np.ndarray]]
            Predictions array, or dict of {horizon: predictions}
        """
        X = np.asarray(X)

        if return_all_horizons:
            return self._predict_all_horizons(X)

        target = horizon or self.target_horizon
        return self._predict_single_horizon(X, target)

    def _predict_single_horizon(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Predict for a single horizon."""
        if self.strategy == 'direct_separate':
            if horizon not in self.models:
                raise ValueError(f"No model trained for horizon {horizon}")
            model = self.models[horizon]
        else:
            model = self.models

        if self.strategy == 'direct_multi_output':
            # Multi-output model returns all horizons
            all_preds = model.predict(X)
            if len(all_preds.shape) == 1:
                # Single output - return as is
                return np.clip(all_preds, 0, None)
            # Multi-output - select the horizon column (0-indexed)
            horizon_idx = horizon - 1
            return np.clip(all_preds[:, horizon_idx], 0, None)
        else:
            # Single-output model
            return np.clip(model.predict(X), 0, None)

    def _predict_all_horizons(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Predict for all horizons."""
        predictions = {}

        if self.strategy == 'direct_separate':
            for h in range(1, self.max_horizon + 1):
                if h in self.models:
                    predictions[h] = np.clip(self.models[h].predict(X), 0, None)
        elif self.strategy == 'direct_multi_output':
            all_preds = self.models.predict(X)
            if len(all_preds.shape) == 1:
                # Single output model
                predictions[self.target_horizon] = np.clip(all_preds, 0, None)
            else:
                for h in range(1, min(self.max_horizon + 1, all_preds.shape[1] + 1)):
                    predictions[h] = np.clip(all_preds[:, h - 1], 0, None)
        else:
            # Horizon-weighted single model - only target horizon
            predictions[self.target_horizon] = np.clip(self.models.predict(X), 0, None)

        return predictions


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    total_actual = np.sum(np.abs(y_true))
    if total_actual < 1e-10:
        return 0.0 if np.sum(np.abs(y_pred)) < 1e-10 else 1.0
    return np.sum(np.abs(y_true - y_pred)) / total_actual


def horizon_weighted_wape(
    y_true_by_horizon: Dict[int, np.ndarray],
    y_pred_by_horizon: Dict[int, np.ndarray],
    horizon_weights: Optional[Dict[int, float]] = None,
) -> float:
    """
    Compute horizon-weighted WAPE.

    Combines WAPE across horizons using specified weights.

    Parameters
    ----------
    y_true_by_horizon : Dict[int, np.ndarray]
        True values for each horizon
    y_pred_by_horizon : Dict[int, np.ndarray]
        Predicted values for each horizon
    horizon_weights : Dict[int, float], optional
        Weights for each horizon. Defaults to DEFAULT_HORIZON_WEIGHTS.

    Returns
    -------
    float
        Weighted average WAPE across horizons
    """
    weights = horizon_weights or DEFAULT_HORIZON_WEIGHTS

    total_weight = 0.0
    weighted_wape = 0.0

    for h, y_true in y_true_by_horizon.items():
        if h not in y_pred_by_horizon:
            continue
        y_pred = y_pred_by_horizon[h]
        w = weights.get(h, 1.0 / len(y_true_by_horizon))

        wape = compute_wape(y_true, y_pred)
        weighted_wape += w * wape
        total_weight += w

    return weighted_wape / total_weight if total_weight > 0 else 0.0


def horizon_weighted_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    """Compute weighted MSE for gradient boosting custom objective."""
    errors = (y_true - y_pred) ** 2
    return np.sum(sample_weights * errors) / np.sum(sample_weights)


# =============================================================================
# DATASET CREATION FOR MULTI-HORIZON TRAINING
# =============================================================================

def create_multi_horizon_targets(
    y: np.ndarray,
    max_horizon: int = 5,
) -> np.ndarray:
    """
    Create multi-horizon target matrix from a time series.

    For each position t, creates targets [y_{t+1}, y_{t+2}, ..., y_{t+H}]

    Parameters
    ----------
    y : np.ndarray
        Time series array of length T
    max_horizon : int
        Maximum forecast horizon

    Returns
    -------
    np.ndarray
        Target matrix of shape (T - max_horizon, max_horizon)
        Each row is [y_{t+1}, y_{t+2}, ..., y_{t+H}]
    """
    y = np.asarray(y).flatten()
    T = len(y)

    if T <= max_horizon:
        raise ValueError(f"Time series length {T} must be > max_horizon {max_horizon}")

    n_samples = T - max_horizon
    targets = np.zeros((n_samples, max_horizon))

    for h in range(max_horizon):
        targets[:, h] = y[h + 1 : n_samples + h + 1]

    return targets


def create_multi_horizon_dataset(
    X: np.ndarray,
    y: np.ndarray,
    max_horizon: int = 5,
    target_horizon: int = 5,
    weight_scheme: str = 'default',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Create dataset for multi-horizon training.

    For 'direct_separate' strategy: Returns aligned X, y_multi for training
    separate models per horizon.

    For 'horizon_weighted' strategy: Returns expanded X, y with sample weights
    that emphasize samples predicting the target horizon.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (T, n_features)
    y : np.ndarray
        Target array of shape (T,)
    max_horizon : int
        Maximum forecast horizon
    target_horizon : int
        Primary evaluation horizon (for weighting)
    weight_scheme : str
        'default': Use DEFAULT_HORIZON_WEIGHTS
        'lag5_focused': Use LAG5_FOCUSED_WEIGHTS
        'balanced': Use BALANCED_HORIZON_WEIGHTS

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]
        (X_aligned, y_multi, sample_weights, horizon_weights)
        - X_aligned: Features aligned with multi-horizon targets
        - y_multi: Multi-horizon targets (n_samples, max_horizon)
        - sample_weights: Weights for each (sample, horizon) pair
        - horizon_weights: The weight scheme used
    """
    X = np.asarray(X)
    y = np.asarray(y).flatten()

    # Select weight scheme – use dynamic builder so it scales to any max_horizon
    if weight_scheme == 'lag5_focused' or weight_scheme == 'eval_focused':
        horizon_weights = build_horizon_weights(max_horizon, scheme='eval_focused')
    elif weight_scheme == 'balanced':
        horizon_weights = build_horizon_weights(max_horizon, scheme='balanced')
    else:
        horizon_weights = build_horizon_weights(max_horizon, scheme='default')

    # Create multi-horizon targets
    y_multi = create_multi_horizon_targets(y, max_horizon)

    # Align X with targets (remove last max_horizon rows)
    n_samples = y_multi.shape[0]
    X_aligned = X[:n_samples]

    # Create sample weights (one weight per sample, based on target_horizon importance)
    # For direct training, we weight samples uniformly but train separate models
    sample_weights = np.ones(n_samples)

    return X_aligned, y_multi, sample_weights, horizon_weights


def create_horizon_weighted_expanded_dataset(
    X: np.ndarray,
    y: np.ndarray,
    max_horizon: int = 5,
    horizon_weights: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an expanded dataset where each sample is replicated for each horizon.

    This enables training a single model with horizon as a feature, where
    samples are weighted by horizon importance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (T, n_features)
    y : np.ndarray
        Target array (T,)
    max_horizon : int
        Maximum forecast horizon
    horizon_weights : Dict[int, float], optional
        Weights for each horizon

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (X_expanded, y_expanded, sample_weights)
        - X_expanded: Features with horizon indicator appended (n_samples * H, n_features + 1)
        - y_expanded: Corresponding targets
        - sample_weights: Weights for each sample
    """
    weights = horizon_weights or DEFAULT_HORIZON_WEIGHTS

    X = np.asarray(X)
    y = np.asarray(y).flatten()

    # Create multi-horizon targets
    y_multi = create_multi_horizon_targets(y, max_horizon)
    n_samples = y_multi.shape[0]
    X_aligned = X[:n_samples]

    # Expand dataset: each sample replicated for each horizon
    n_expanded = n_samples * max_horizon
    n_features = X_aligned.shape[1]

    X_expanded = np.zeros((n_expanded, n_features + 1))  # +1 for horizon feature
    y_expanded = np.zeros(n_expanded)
    sample_weights = np.zeros(n_expanded)

    idx = 0
    for h in range(1, max_horizon + 1):
        for i in range(n_samples):
            X_expanded[idx, :n_features] = X_aligned[i]
            X_expanded[idx, n_features] = h  # Horizon indicator
            y_expanded[idx] = y_multi[i, h - 1]
            sample_weights[idx] = weights.get(h, 1.0 / max_horizon)
            idx += 1

    return X_expanded, y_expanded, sample_weights


# =============================================================================
# CUSTOM OBJECTIVE FUNCTIONS FOR TREE MODELS
# =============================================================================

def create_horizon_weighted_objective_lightgbm(
    horizon_weights: Dict[int, float],
    horizon_indices: np.ndarray,
) -> Callable:
    """
    Create a horizon-weighted custom objective for LightGBM.

    This objective function weighs samples based on which horizon they predict,
    giving more importance to longer horizons.

    Parameters
    ----------
    horizon_weights : Dict[int, float]
        Weights for each horizon
    horizon_indices : np.ndarray
        Array indicating which horizon each sample corresponds to

    Returns
    -------
    Callable
        Custom objective function for LightGBM
    """
    # Precompute sample weights based on horizon
    sample_weights = np.array([
        horizon_weights.get(int(h), 0.2) for h in horizon_indices
    ])

    def horizon_weighted_mse_obj(y_pred: np.ndarray, train_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Horizon-weighted MSE objective.

        Returns gradient and hessian weighted by horizon importance.
        """
        y_true = train_data.get_label()

        # Gradient: d/dy_pred of 0.5 * w * (y_true - y_pred)^2 = -w * (y_true - y_pred)
        residual = y_true - y_pred
        grad = -sample_weights * residual

        # Hessian: d^2/dy_pred^2 = w
        hess = sample_weights.copy()

        return grad, hess

    return horizon_weighted_mse_obj


def create_horizon_weighted_objective_xgboost(
    horizon_weights: Dict[int, float],
    horizon_indices: np.ndarray,
) -> Callable:
    """
    Create a horizon-weighted custom objective for XGBoost.

    Parameters
    ----------
    horizon_weights : Dict[int, float]
        Weights for each horizon
    horizon_indices : np.ndarray
        Array indicating which horizon each sample corresponds to

    Returns
    -------
    Callable
        Custom objective function for XGBoost
    """
    sample_weights = np.array([
        horizon_weights.get(int(h), 0.2) for h in horizon_indices
    ])

    def horizon_weighted_mse_obj(y_pred: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
        """Horizon-weighted MSE objective for XGBoost."""
        y_true = dtrain.get_label()

        residual = y_true - y_pred
        grad = -sample_weights * residual
        hess = sample_weights.copy()

        return grad, hess

    return horizon_weighted_mse_obj


# =============================================================================
# DIRECT SEPARATE HORIZON TRAINING
# =============================================================================

def train_direct_separate_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_horizon: int = 5,
    target_horizon: int = 5,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> MultiHorizonTrainingResult:
    """
    Train separate LightGBM models for each horizon.

    This is the "Direct" forecasting approach where each horizon h
    has its own model that directly predicts y_{t+h}.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    max_horizon : int
        Maximum forecast horizon to train for
    target_horizon : int
        Primary evaluation horizon (weights reporting)
    params : Dict, optional
        LightGBM hyperparameters
    early_stopping_rounds : int
        Early stopping patience

    Returns
    -------
    MultiHorizonTrainingResult
        Contains MultiHorizonModel with trained models per horizon
    """
    import lightgbm as lgb

    start_time = time.time()

    # Create multi-horizon targets
    y_train_multi = create_multi_horizon_targets(y_train, max_horizon)
    y_val_multi = create_multi_horizon_targets(y_val, max_horizon)

    n_train = y_train_multi.shape[0]
    n_val = y_val_multi.shape[0]

    X_train_aligned = X_train[:n_train]
    X_val_aligned = X_val[:n_val]

    # Default parameters
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
    final_params = {**default_params, **(params or {})}
    n_estimators = final_params.pop('n_estimators', 500)

    # Train model for each horizon
    models = {}
    train_wape_by_horizon = {}
    val_wape_by_horizon = {}

    for h in range(1, max_horizon + 1):
        logger.debug(f"Training LightGBM for horizon {h}...")

        y_h_train = y_train_multi[:, h - 1]
        y_h_val = y_val_multi[:, h - 1]

        model = lgb.LGBMRegressor(n_estimators=n_estimators, **final_params)
        model.fit(
            X_train_aligned, y_h_train,
            eval_set=[(X_val_aligned, y_h_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )

        models[h] = model

        # Compute metrics for this horizon
        train_pred = np.clip(model.predict(X_train_aligned), 0, None)
        val_pred = np.clip(model.predict(X_val_aligned), 0, None)

        train_wape_by_horizon[h] = compute_wape(y_h_train, train_pred)
        val_wape_by_horizon[h] = compute_wape(y_h_val, val_pred)

    # Build dynamic weights that scale to any max_horizon
    hw = build_horizon_weights(max_horizon, scheme='default')

    # Create wrapper model
    multi_model = MultiHorizonModel(
        models=models,
        strategy='direct_separate',
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        horizon_weights=hw,
    )

    # Compute weighted metrics
    train_weighted_wape = sum(
        hw.get(h, 1.0 / max_horizon) * train_wape_by_horizon[h]
        for h in range(1, max_horizon + 1)
    )
    val_weighted_wape = sum(
        hw.get(h, 1.0 / max_horizon) * val_wape_by_horizon[h]
        for h in range(1, max_horizon + 1)
    )

    return MultiHorizonTrainingResult(
        model=multi_model,
        model_type='lightgbm_multi_horizon',
        strategy='direct_separate',
        horizon_weights=hw,
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        train_wape_by_horizon=train_wape_by_horizon,
        val_wape_by_horizon=val_wape_by_horizon,
        train_weighted_wape=train_weighted_wape,
        val_weighted_wape=val_weighted_wape,
        train_target_wape=train_wape_by_horizon.get(target_horizon, 0.0),
        val_target_wape=val_wape_by_horizon.get(target_horizon, 0.0),
        hyperparameters=final_params,
        training_time_seconds=time.time() - start_time,
    )


def train_direct_separate_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_horizon: int = 5,
    target_horizon: int = 5,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> MultiHorizonTrainingResult:
    """
    Train separate XGBoost models for each horizon.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    max_horizon : int
        Maximum forecast horizon
    target_horizon : int
        Primary evaluation horizon
    params : Dict, optional
        XGBoost hyperparameters
    early_stopping_rounds : int
        Early stopping patience

    Returns
    -------
    MultiHorizonTrainingResult
    """
    import xgboost as xgb

    start_time = time.time()

    # Create multi-horizon targets
    y_train_multi = create_multi_horizon_targets(y_train, max_horizon)
    y_val_multi = create_multi_horizon_targets(y_val, max_horizon)

    n_train = y_train_multi.shape[0]
    n_val = y_val_multi.shape[0]

    X_train_aligned = X_train[:n_train]
    X_val_aligned = X_val[:n_val]

    # Default parameters
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
    final_params = {**default_params, **(params or {})}

    # Train model for each horizon
    models = {}
    train_wape_by_horizon = {}
    val_wape_by_horizon = {}

    for h in range(1, max_horizon + 1):
        logger.debug(f"Training XGBoost for horizon {h}...")

        y_h_train = y_train_multi[:, h - 1]
        y_h_val = y_val_multi[:, h - 1]

        model = xgb.XGBRegressor(**final_params)
        model.fit(
            X_train_aligned, y_h_train,
            eval_set=[(X_val_aligned, y_h_val)],
            verbose=False
        )

        models[h] = model

        # Compute metrics
        train_pred = np.clip(model.predict(X_train_aligned), 0, None)
        val_pred = np.clip(model.predict(X_val_aligned), 0, None)

        train_wape_by_horizon[h] = compute_wape(y_h_train, train_pred)
        val_wape_by_horizon[h] = compute_wape(y_h_val, val_pred)

    # Build dynamic weights
    hw = build_horizon_weights(max_horizon, scheme='default')

    # Create wrapper
    multi_model = MultiHorizonModel(
        models=models,
        strategy='direct_separate',
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        horizon_weights=hw,
    )

    # Weighted metrics
    train_weighted_wape = sum(
        hw.get(h, 1.0 / max_horizon) * train_wape_by_horizon[h]
        for h in range(1, max_horizon + 1)
    )
    val_weighted_wape = sum(
        hw.get(h, 1.0 / max_horizon) * val_wape_by_horizon[h]
        for h in range(1, max_horizon + 1)
    )

    return MultiHorizonTrainingResult(
        model=multi_model,
        model_type='xgboost_multi_horizon',
        strategy='direct_separate',
        horizon_weights=hw,
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        train_wape_by_horizon=train_wape_by_horizon,
        val_wape_by_horizon=val_wape_by_horizon,
        train_weighted_wape=train_weighted_wape,
        val_weighted_wape=val_weighted_wape,
        train_target_wape=train_wape_by_horizon.get(target_horizon, 0.0),
        val_target_wape=val_wape_by_horizon.get(target_horizon, 0.0),
        hyperparameters=final_params,
        training_time_seconds=time.time() - start_time,
    )


# =============================================================================
# HORIZON-WEIGHTED SINGLE MODEL TRAINING
# =============================================================================

def train_horizon_weighted_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_horizon: int = 5,
    target_horizon: int = 5,
    weight_scheme: str = 'lag5_focused',
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> MultiHorizonTrainingResult:
    """
    Train a single LightGBM model with horizon as a feature and weighted samples.

    This approach:
    1. Expands the dataset: each sample is replicated for each horizon
    2. Adds horizon as a feature
    3. Weights samples by horizon importance

    The result is a single model that learns to predict any horizon,
    with emphasis on the target horizon.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    max_horizon : int
        Maximum forecast horizon
    target_horizon : int
        Primary evaluation horizon
    weight_scheme : str
        'default', 'lag5_focused', or 'balanced'
    params : Dict, optional
        LightGBM hyperparameters
    early_stopping_rounds : int
        Early stopping patience

    Returns
    -------
    MultiHorizonTrainingResult
    """
    import lightgbm as lgb

    start_time = time.time()

    # Select horizon weights – dynamic for any max_horizon
    if weight_scheme == 'lag5_focused' or weight_scheme == 'eval_focused':
        horizon_weights = build_horizon_weights(max_horizon, scheme='eval_focused')
    elif weight_scheme == 'balanced':
        horizon_weights = build_horizon_weights(max_horizon, scheme='balanced')
    else:
        horizon_weights = build_horizon_weights(max_horizon, scheme='default')

    # Create expanded datasets
    X_train_exp, y_train_exp, weights_train = create_horizon_weighted_expanded_dataset(
        X_train, y_train, max_horizon, horizon_weights
    )
    X_val_exp, y_val_exp, weights_val = create_horizon_weighted_expanded_dataset(
        X_val, y_val, max_horizon, horizon_weights
    )

    # Default parameters
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
    final_params = {**default_params, **(params or {})}
    n_estimators = final_params.pop('n_estimators', 500)

    # Train with sample weights
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **final_params)
    model.fit(
        X_train_exp, y_train_exp,
        sample_weight=weights_train,
        eval_set=[(X_val_exp, y_val_exp)],
        eval_sample_weight=[weights_val],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )

    # Compute metrics per horizon
    # Need to create aligned data for each horizon
    y_train_multi = create_multi_horizon_targets(y_train, max_horizon)
    y_val_multi = create_multi_horizon_targets(y_val, max_horizon)

    n_train = y_train_multi.shape[0]
    n_val = y_val_multi.shape[0]

    X_train_aligned = X_train[:n_train]
    X_val_aligned = X_val[:n_val]

    train_wape_by_horizon = {}
    val_wape_by_horizon = {}

    for h in range(1, max_horizon + 1):
        # Create features with horizon indicator
        X_train_h = np.column_stack([X_train_aligned, np.full(n_train, h)])
        X_val_h = np.column_stack([X_val_aligned, np.full(n_val, h)])

        train_pred = np.clip(model.predict(X_train_h), 0, None)
        val_pred = np.clip(model.predict(X_val_h), 0, None)

        train_wape_by_horizon[h] = compute_wape(y_train_multi[:, h - 1], train_pred)
        val_wape_by_horizon[h] = compute_wape(y_val_multi[:, h - 1], val_pred)

    # Create wrapper (needs to handle horizon feature internally)
    class HorizonAwareModel:
        """Wrapper that adds horizon feature before prediction."""
        def __init__(self, base_model, target_horizon):
            self.base_model = base_model
            self.target_horizon = target_horizon
            self.model_type = 'horizon_weighted_lightgbm'

        def predict(self, X, horizon=None):
            h = horizon if horizon is not None else self.target_horizon
            X = np.asarray(X)
            X_h = np.column_stack([X, np.full(X.shape[0], h)])
            return np.clip(self.base_model.predict(X_h), 0, None)

    wrapped_model = HorizonAwareModel(model, target_horizon)

    multi_model = MultiHorizonModel(
        models=wrapped_model,
        strategy='horizon_weighted',
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        horizon_weights=horizon_weights,
    )

    # Weighted metrics
    train_weighted_wape = sum(
        horizon_weights.get(h, 0.2) * train_wape_by_horizon[h]
        for h in range(1, max_horizon + 1)
    )
    val_weighted_wape = sum(
        horizon_weights.get(h, 0.2) * val_wape_by_horizon[h]
        for h in range(1, max_horizon + 1)
    )

    return MultiHorizonTrainingResult(
        model=multi_model,
        model_type='lightgbm_horizon_weighted',
        strategy='horizon_weighted',
        horizon_weights=horizon_weights,
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        train_wape_by_horizon=train_wape_by_horizon,
        val_wape_by_horizon=val_wape_by_horizon,
        train_weighted_wape=train_weighted_wape,
        val_weighted_wape=val_weighted_wape,
        train_target_wape=train_wape_by_horizon.get(target_horizon, 0.0),
        val_target_wape=val_wape_by_horizon.get(target_horizon, 0.0),
        hyperparameters=final_params,
        training_time_seconds=time.time() - start_time,
    )


# =============================================================================
# TARGET HORIZON FOCUSED TRAINING
# =============================================================================

def train_target_horizon_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_horizon: int = 5,
    model_type: str = 'lightgbm',
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> MultiHorizonTrainingResult:
    """
    Train a single model specifically for the target horizon.

    This is the simplest direct forecasting approach: train a model
    that directly predicts y_{t+H} instead of y_{t+1}.

    Advantages:
    - No error compounding from recursive prediction
    - Model learns patterns specific to the target horizon
    - Simple and computationally efficient

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    target_horizon : int
        The forecast horizon to optimize for (e.g., 5 for Lag 5)
    model_type : str
        'lightgbm' or 'xgboost'
    params : Dict, optional
        Model hyperparameters
    early_stopping_rounds : int
        Early stopping patience

    Returns
    -------
    MultiHorizonTrainingResult
    """
    start_time = time.time()

    # Create target for the specific horizon
    y_train_h = y_train[target_horizon:]  # y_{t+H}
    y_val_h = y_val[target_horizon:]

    n_train = len(y_train_h)
    n_val = len(y_val_h)

    X_train_aligned = X_train[:n_train]
    X_val_aligned = X_val[:n_val]

    if model_type == 'lightgbm':
        import lightgbm as lgb

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
        final_params = {**default_params, **(params or {})}
        n_estimators = final_params.pop('n_estimators', 500)

        model = lgb.LGBMRegressor(n_estimators=n_estimators, **final_params)
        model.fit(
            X_train_aligned, y_train_h,
            eval_set=[(X_val_aligned, y_val_h)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
    else:  # xgboost
        import xgboost as xgb

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
        final_params = {**default_params, **(params or {})}

        model = xgb.XGBRegressor(**final_params)
        model.fit(
            X_train_aligned, y_train_h,
            eval_set=[(X_val_aligned, y_val_h)],
            verbose=False
        )

    # Compute metrics
    train_pred = np.clip(model.predict(X_train_aligned), 0, None)
    val_pred = np.clip(model.predict(X_val_aligned), 0, None)

    train_wape = compute_wape(y_train_h, train_pred)
    val_wape = compute_wape(y_val_h, val_pred)

    # Wrap in MultiHorizonModel
    multi_model = MultiHorizonModel(
        models={target_horizon: model},
        strategy='direct_separate',
        max_horizon=target_horizon,
        target_horizon=target_horizon,
        horizon_weights={target_horizon: 1.0},
    )

    return MultiHorizonTrainingResult(
        model=multi_model,
        model_type=f'{model_type}_target_horizon',
        strategy='target_horizon_only',
        horizon_weights={target_horizon: 1.0},
        max_horizon=target_horizon,
        target_horizon=target_horizon,
        train_wape_by_horizon={target_horizon: train_wape},
        val_wape_by_horizon={target_horizon: val_wape},
        train_weighted_wape=train_wape,
        val_weighted_wape=val_wape,
        train_target_wape=train_wape,
        val_target_wape=val_wape,
        hyperparameters=final_params,
        training_time_seconds=time.time() - start_time,
    )


# =============================================================================
# HIGH-LEVEL TRAINING FUNCTIONS
# =============================================================================

def train_multi_horizon_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_horizon: int = 5,
    target_horizon: int = 5,
    strategy: str = 'direct_separate',
    weight_scheme: str = 'lag5_focused',
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> MultiHorizonTrainingResult:
    """
    Train multi-horizon LightGBM with specified strategy.

    This is the main entry point for multi-horizon LightGBM training.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    max_horizon : int
        Maximum forecast horizon to train for
    target_horizon : int
        Primary evaluation horizon (e.g., 5 for Lag 5)
    strategy : str
        'direct_separate': Train separate model per horizon (recommended)
        'horizon_weighted': Single model with horizon feature and weighted samples
        'target_only': Train only for target horizon
    weight_scheme : str
        'default', 'lag5_focused', or 'balanced'
    params : Dict, optional
        LightGBM hyperparameters
    early_stopping_rounds : int
        Early stopping patience

    Returns
    -------
    MultiHorizonTrainingResult
    """
    if strategy == 'direct_separate':
        return train_direct_separate_lightgbm(
            X_train, y_train, X_val, y_val,
            max_horizon=max_horizon,
            target_horizon=target_horizon,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
    elif strategy == 'horizon_weighted':
        return train_horizon_weighted_lightgbm(
            X_train, y_train, X_val, y_val,
            max_horizon=max_horizon,
            target_horizon=target_horizon,
            weight_scheme=weight_scheme,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
    elif strategy == 'target_only':
        return train_target_horizon_model(
            X_train, y_train, X_val, y_val,
            target_horizon=target_horizon,
            model_type='lightgbm',
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'direct_separate', 'horizon_weighted', or 'target_only'")


def train_multi_horizon_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_horizon: int = 5,
    target_horizon: int = 5,
    strategy: str = 'direct_separate',
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> MultiHorizonTrainingResult:
    """
    Train multi-horizon XGBoost with specified strategy.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    max_horizon : int
        Maximum forecast horizon
    target_horizon : int
        Primary evaluation horizon
    strategy : str
        'direct_separate' or 'target_only'
    params : Dict, optional
        XGBoost hyperparameters
    early_stopping_rounds : int
        Early stopping patience

    Returns
    -------
    MultiHorizonTrainingResult
    """
    if strategy == 'direct_separate':
        return train_direct_separate_xgboost(
            X_train, y_train, X_val, y_val,
            max_horizon=max_horizon,
            target_horizon=target_horizon,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
    elif strategy == 'target_only':
        return train_target_horizon_model(
            X_train, y_train, X_val, y_val,
            target_horizon=target_horizon,
            model_type='xgboost',
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'direct_separate' or 'target_only'")


def train_multi_horizon_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_horizon: int = 5,
    target_horizon: int = 5,
    model_types: List[str] = None,
    optimize_weights: bool = True,
) -> MultiHorizonTrainingResult:
    """
    Train an ensemble of multi-horizon models optimized for target horizon.

    This trains multiple model types (LightGBM, XGBoost) with multi-horizon
    strategies and combines them with optimized weights.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    max_horizon : int
        Maximum forecast horizon
    target_horizon : int
        Primary evaluation horizon
    model_types : List[str], optional
        Model types to include. Default: ['lightgbm', 'xgboost']
    optimize_weights : bool
        Whether to optimize ensemble weights on validation data

    Returns
    -------
    MultiHorizonTrainingResult
    """
    start_time = time.time()

    if model_types is None:
        model_types = ['lightgbm', 'xgboost']

    # Train each model type
    results = []
    for model_type in model_types:
        if model_type == 'lightgbm':
            result = train_direct_separate_lightgbm(
                X_train, y_train, X_val, y_val,
                max_horizon=max_horizon,
                target_horizon=target_horizon,
            )
        elif model_type == 'xgboost':
            result = train_direct_separate_xgboost(
                X_train, y_train, X_val, y_val,
                max_horizon=max_horizon,
                target_horizon=target_horizon,
            )
        else:
            continue
        results.append((model_type, result))

    if not results:
        raise ValueError("No models trained successfully")

    # If only one model, return it
    if len(results) == 1:
        return results[0][1]

    # Optimize ensemble weights based on target horizon validation performance
    if optimize_weights:
        weights = _optimize_ensemble_weights(
            [r[1] for r in results],
            X_val, y_val,
            target_horizon,
        )
    else:
        weights = [1.0 / len(results)] * len(results)

    # Create ensemble model
    class MultiHorizonEnsemble:
        """Ensemble of multi-horizon models."""
        def __init__(self, models_and_weights, target_horizon):
            self.models_and_weights = models_and_weights
            self.target_horizon = target_horizon
            self.model_type = 'multi_horizon_ensemble'

        def predict(self, X, horizon=None):
            h = horizon if horizon is not None else self.target_horizon
            X = np.asarray(X)

            predictions = np.zeros(X.shape[0])
            total_weight = 0.0

            for (model, weight) in self.models_and_weights:
                try:
                    pred = model.predict(X, horizon=h)
                    predictions += weight * pred
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Ensemble member prediction failed: {e}")
                    continue

            if total_weight > 0:
                predictions /= total_weight

            return np.clip(predictions, 0, None)

    models_and_weights = [(r[1].model, w) for r, w in zip(results, weights)]
    ensemble = MultiHorizonEnsemble(models_and_weights, target_horizon)

    # Compute ensemble metrics
    y_val_multi = create_multi_horizon_targets(y_val, max_horizon)
    n_val = y_val_multi.shape[0]
    X_val_aligned = X_val[:n_val]

    val_wape_by_horizon = {}
    for h in range(1, max_horizon + 1):
        val_pred = ensemble.predict(X_val_aligned, horizon=h)
        val_wape_by_horizon[h] = compute_wape(y_val_multi[:, h - 1], val_pred)

    hw = build_horizon_weights(max_horizon, scheme='default')
    val_weighted_wape = sum(
        hw.get(h, 1.0 / max_horizon) * val_wape_by_horizon[h]
        for h in range(1, max_horizon + 1)
    )

    # Wrap in MultiHorizonModel
    multi_model = MultiHorizonModel(
        models=ensemble,
        strategy='ensemble',
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        horizon_weights=hw,
    )

    return MultiHorizonTrainingResult(
        model=multi_model,
        model_type='multi_horizon_ensemble',
        strategy='ensemble',
        horizon_weights=hw,
        max_horizon=max_horizon,
        target_horizon=target_horizon,
        train_wape_by_horizon={},  # Not computed for ensemble
        val_wape_by_horizon=val_wape_by_horizon,
        train_weighted_wape=0.0,
        val_weighted_wape=val_weighted_wape,
        train_target_wape=0.0,
        val_target_wape=val_wape_by_horizon.get(target_horizon, 0.0),
        hyperparameters={'ensemble_weights': {r[0]: w for r, w in zip(results, weights)}},
        training_time_seconds=time.time() - start_time,
    )


def _optimize_ensemble_weights(
    results: List[MultiHorizonTrainingResult],
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_horizon: int,
) -> List[float]:
    """
    Optimize ensemble weights based on validation performance.

    Uses simple grid search to find weights that minimize WAPE
    at the target horizon.
    """
    from scipy.optimize import minimize

    n_models = len(results)

    if n_models == 1:
        return [1.0]

    # Get target horizon targets
    y_val_h = y_val[target_horizon:]
    n_val = len(y_val_h)
    X_val_aligned = X_val[:n_val]

    # Get predictions from each model
    predictions = []
    for result in results:
        try:
            pred = result.model.predict(X_val_aligned, horizon=target_horizon)
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Ensemble weight optimization: model prediction failed: {e}")
            predictions.append(np.zeros(n_val))

    predictions = np.array(predictions)  # (n_models, n_samples)

    def objective(weights):
        """WAPE of weighted ensemble."""
        weights = np.abs(weights)  # Ensure positive
        weights = weights / np.sum(weights)  # Normalize

        ensemble_pred = np.sum(weights[:, None] * predictions, axis=0)
        return compute_wape(y_val_h, ensemble_pred)

    # Initial weights (equal)
    x0 = np.ones(n_models) / n_models

    # Optimize
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 100}
    )

    weights = np.abs(result.x)
    weights = weights / np.sum(weights)

    return weights.tolist()


# =============================================================================
# COMPATIBILITY WITH EXISTING TRAINING PIPELINE
# =============================================================================

def train_multi_horizon_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'lightgbm',
    max_horizon: int = 5,
    target_horizon: int = 5,
    strategy: str = 'direct_separate',
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> MultiHorizonTrainingResult:
    """
    Generic multi-horizon training function compatible with existing pipeline.

    This function provides a unified interface for multi-horizon training
    that matches the signature of existing training functions.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    model_type : str
        'lightgbm', 'xgboost', or 'ensemble'
    max_horizon : int
        Maximum forecast horizon
    target_horizon : int
        Primary evaluation horizon
    strategy : str
        Training strategy
    params : Dict, optional
        Model hyperparameters
    **kwargs
        Additional arguments passed to training function

    Returns
    -------
    MultiHorizonTrainingResult
    """
    if model_type == 'lightgbm':
        return train_multi_horizon_lightgbm(
            X_train, y_train, X_val, y_val,
            max_horizon=max_horizon,
            target_horizon=target_horizon,
            strategy=strategy,
            params=params,
            **kwargs
        )
    elif model_type == 'xgboost':
        return train_multi_horizon_xgboost(
            X_train, y_train, X_val, y_val,
            max_horizon=max_horizon,
            target_horizon=target_horizon,
            strategy=strategy,
            params=params,
            **kwargs
        )
    elif model_type == 'ensemble':
        return train_multi_horizon_ensemble(
            X_train, y_train, X_val, y_val,
            max_horizon=max_horizon,
            target_horizon=target_horizon,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants & weight builder
    'DEFAULT_HORIZON_WEIGHTS',
    'LAG5_FOCUSED_WEIGHTS',
    'BALANCED_HORIZON_WEIGHTS',
    'build_horizon_weights',

    # Data classes
    'MultiHorizonTrainingResult',
    'MultiHorizonModel',

    # Metrics
    'compute_wape',
    'horizon_weighted_wape',

    # Dataset creation
    'create_multi_horizon_targets',
    'create_multi_horizon_dataset',
    'create_horizon_weighted_expanded_dataset',

    # Direct separate training
    'train_direct_separate_lightgbm',
    'train_direct_separate_xgboost',

    # Horizon-weighted training
    'train_horizon_weighted_lightgbm',

    # Target horizon training
    'train_target_horizon_model',

    # High-level functions
    'train_multi_horizon_lightgbm',
    'train_multi_horizon_xgboost',
    'train_multi_horizon_ensemble',
    'train_multi_horizon_model',
]
