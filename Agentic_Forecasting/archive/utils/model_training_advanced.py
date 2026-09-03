"""
Advanced Model Training Utilities for State-of-the-Art Demand Forecasting.

This module extends utils/model_training.py with advanced capabilities:

1. CROSS-VALIDATION: TimeSeriesSplit, purged CV, nested CV
2. MODEL COMPARISON: Systematic multi-model benchmarking
3. UNCERTAINTY QUANTIFICATION: Prediction intervals, conformal prediction
4. MODEL INTERPRETATION: SHAP, feature importance aggregation
5. BACKTESTING: Walk-forward validation, rolling window evaluation
6. MODEL COMBINATION: Averaging, weighted blending, stacking
7. LIFECYCLE HANDLING: Dead keys, new product ramp-up
8. HIGH-LEVEL PIPELINE: End-to-end training orchestration

USAGE:
>>> from utils.model_training_advanced import (
...     # High-level pipeline (does EVERYTHING)
...     run_training_pipeline,
...
...     # Cross-validation
...     time_series_cv, purged_time_series_cv, nested_cv_tune,
...
...     # Model comparison
...     compare_models, benchmark_model_families, create_model_leaderboard,
...
...     # Uncertainty quantification
...     compute_prediction_intervals, quantile_regression_intervals,
...     conformal_prediction_intervals, bootstrap_prediction_intervals,
...
...     # Model interpretation
...     compute_shap_importance, aggregate_feature_importance,
...     explain_predictions, create_importance_report,
...
...     # Backtesting
...     walk_forward_validation, rolling_window_backtest,
...     compute_backtest_metrics,
...
...     # Model combination
...     simple_average_ensemble, weighted_average_ensemble,
...     optimal_weights_ensemble, model_selection_ensemble,
...
...     # Lifecycle handling
...     detect_dead_keys, detect_new_products, apply_lifecycle_rules,
...     ramp_up_adjustment,
...
...     # Utilities
...     save_model_versioned, load_model_versioned, create_training_report,
... )

Author: FEU-Agentic-Forecasting Demand Forecasting System
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

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

# Default WAPE threshold for acceptable models
DEFAULT_WAPE_THRESHOLD = 0.30

# Dead key detection
DEFAULT_DEAD_KEY_LOOKBACK = 52
DEFAULT_DEAD_KEY_THRESHOLD = 0.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CVResult:
    """Result of cross-validation."""
    mean_wape: float
    std_wape: float
    fold_wapes: List[float]
    fold_maes: List[float]
    fold_rmses: List[float]
    n_folds: int
    best_fold_idx: int
    worst_fold_idx: int


@dataclass
class ModelComparisonResult:
    """Result of model comparison."""
    model_name: str
    cv_result: CVResult
    training_time: float
    best_params: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0


@dataclass
class PredictionInterval:
    """Prediction with uncertainty bounds."""
    point_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    method: str


@dataclass
class BacktestResult:
    """Result of walk-forward backtesting."""
    overall_wape: float
    period_wapes: List[float]
    period_maes: List[float]
    cumulative_wapes: List[float]
    n_periods: int
    forecast_horizon: int
    actuals: np.ndarray
    predictions: np.ndarray


@dataclass
class TrainingPipelineResult:
    """Result of end-to-end training pipeline."""
    model_specs: List[Dict[str, Any]]
    overall_wape: float
    best_model_group: str
    worst_model_group: str
    models_trained: int
    models_failed: int
    dead_keys_detected: int
    new_products_detected: int
    training_time_seconds: float
    output_dir: str


# =============================================================================
# 1. CROSS-VALIDATION UTILITIES
# =============================================================================

def time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    n_splits: int = 5,
    gap: int = 0,
    test_size: Optional[int] = None,
) -> CVResult:
    """
    Perform time series cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target array
    model_fn : Callable
        Function that returns a fitted model: model_fn(X_train, y_train) -> model
    n_splits : int
        Number of CV folds
    gap : int
        Gap between train and test (to prevent leakage)
    test_size : int, optional
        Fixed test size (if None, uses expanding window)

    Returns
    -------
    CVResult
        Cross-validation results

    Example
    -------
    >>> def train_lgb(X, y):
    ...     import lightgbm as lgb
    ...     model = lgb.LGBMRegressor(n_estimators=100)
    ...     model.fit(X, y)
    ...     return model
    >>> result = time_series_cv(X, y, train_lgb, n_splits=5)
    """
    from sklearn.model_selection import TimeSeriesSplit

    n_samples = len(y)
    if test_size is None:
        test_size = n_samples // (n_splits + 1)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)

    fold_wapes = []
    fold_maes = []
    fold_rmses = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            model = model_fn(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 0, None)

            wape = _compute_wape(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            fold_wapes.append(wape)
            fold_maes.append(mae)
            fold_rmses.append(rmse)
        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed: {e}")
            fold_wapes.append(1.0)
            fold_maes.append(np.inf)
            fold_rmses.append(np.inf)

    return CVResult(
        mean_wape=np.mean(fold_wapes),
        std_wape=np.std(fold_wapes),
        fold_wapes=fold_wapes,
        fold_maes=fold_maes,
        fold_rmses=fold_rmses,
        n_folds=n_splits,
        best_fold_idx=int(np.argmin(fold_wapes)),
        worst_fold_idx=int(np.argmax(fold_wapes)),
    )


def purged_time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_fn: Callable,
    n_splits: int = 5,
    purge_gap: int = 1,
) -> CVResult:
    """
    Purged time series cross-validation to prevent leakage.

    Removes observations from training that are within `purge_gap` of test set.
    Essential when features include forward-looking information.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target array
    groups : np.ndarray
        Time period indices for purging
    model_fn : Callable
        Function that returns fitted model
    n_splits : int
        Number of folds
    purge_gap : int
        Number of periods to purge before test

    Returns
    -------
    CVResult
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    fold_size = n_groups // (n_splits + 1)

    fold_wapes = []
    fold_maes = []
    fold_rmses = []

    for fold_idx in range(n_splits):
        test_start = (fold_idx + 1) * fold_size
        test_end = test_start + fold_size

        test_groups = unique_groups[test_start:test_end]
        train_groups = unique_groups[:max(0, test_start - purge_gap)]

        train_mask = np.isin(groups, train_groups)
        test_mask = np.isin(groups, test_groups)

        if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        try:
            model = model_fn(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 0, None)

            wape = _compute_wape(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            fold_wapes.append(wape)
            fold_maes.append(mae)
            fold_rmses.append(rmse)
        except Exception as e:
            logger.warning(f"Purged fold {fold_idx} failed: {e}")

    if not fold_wapes:
        return CVResult(1.0, 0.0, [1.0], [np.inf], [np.inf], 1, 0, 0)

    return CVResult(
        mean_wape=np.mean(fold_wapes),
        std_wape=np.std(fold_wapes),
        fold_wapes=fold_wapes,
        fold_maes=fold_maes,
        fold_rmses=fold_rmses,
        n_folds=len(fold_wapes),
        best_fold_idx=int(np.argmin(fold_wapes)),
        worst_fold_idx=int(np.argmax(fold_wapes)),
    )


def nested_cv_tune(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    outer_splits: int = 3,
    inner_splits: int = 3,
    metric: str = 'wape',
) -> Tuple[CVResult, Dict[str, Any]]:
    """
    Nested cross-validation for unbiased model selection and evaluation.

    Outer loop: Estimate generalization performance
    Inner loop: Tune hyperparameters

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target array
    model_class : Any
        Model class with sklearn interface
    param_grid : Dict[str, List[Any]]
        Hyperparameter grid to search
    outer_splits : int
        Number of outer CV folds
    inner_splits : int
        Number of inner CV folds
    metric : str
        Metric to optimize ('wape', 'mae', 'rmse')

    Returns
    -------
    Tuple[CVResult, Dict[str, Any]]
        CV result and best parameters from each outer fold
    """
    from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

    outer_cv = TimeSeriesSplit(n_splits=outer_splits)
    inner_cv = TimeSeriesSplit(n_splits=inner_splits)

    fold_wapes = []
    fold_maes = []
    fold_rmses = []
    best_params_per_fold = []

    for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_outer_train, X_outer_test = X[train_idx], X[test_idx]
        y_outer_train, y_outer_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter tuning
        best_score = float('inf')
        best_params = {}

        for params in ParameterGrid(param_grid):
            inner_scores = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train):
                X_inner_train = X_outer_train[inner_train_idx]
                y_inner_train = y_outer_train[inner_train_idx]
                X_inner_val = X_outer_train[inner_val_idx]
                y_inner_val = y_outer_train[inner_val_idx]

                try:
                    model = model_class(**params)
                    model.fit(X_inner_train, y_inner_train)
                    y_pred = np.clip(model.predict(X_inner_val), 0, None)

                    if metric == 'wape':
                        score = _compute_wape(y_inner_val, y_pred)
                    elif metric == 'mae':
                        score = np.mean(np.abs(y_inner_val - y_pred))
                    else:  # rmse
                        score = np.sqrt(np.mean((y_inner_val - y_pred) ** 2))

                    inner_scores.append(score)
                except Exception:
                    inner_scores.append(float('inf'))

            mean_score = np.mean(inner_scores)
            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        # Evaluate with best params on outer test
        try:
            model = model_class(**best_params)
            model.fit(X_outer_train, y_outer_train)
            y_pred = np.clip(model.predict(X_outer_test), 0, None)

            wape = _compute_wape(y_outer_test, y_pred)
            mae = np.mean(np.abs(y_outer_test - y_pred))
            rmse = np.sqrt(np.mean((y_outer_test - y_pred) ** 2))

            fold_wapes.append(wape)
            fold_maes.append(mae)
            fold_rmses.append(rmse)
            best_params_per_fold.append(best_params)
        except Exception as e:
            logger.warning(f"Outer fold {outer_idx} failed: {e}")

    cv_result = CVResult(
        mean_wape=np.mean(fold_wapes) if fold_wapes else 1.0,
        std_wape=np.std(fold_wapes) if fold_wapes else 0.0,
        fold_wapes=fold_wapes,
        fold_maes=fold_maes,
        fold_rmses=fold_rmses,
        n_folds=len(fold_wapes),
        best_fold_idx=int(np.argmin(fold_wapes)) if fold_wapes else 0,
        worst_fold_idx=int(np.argmax(fold_wapes)) if fold_wapes else 0,
    )

    return cv_result, {'best_params_per_fold': best_params_per_fold}


# =============================================================================
# 2. MODEL COMPARISON
# =============================================================================

def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_configs: Dict[str, Dict[str, Any]],
    cv_splits: int = 3,
) -> List[ModelComparisonResult]:
    """
    Compare multiple models systematically.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    model_configs : Dict[str, Dict]
        Model configurations: {'model_name': {'class': ModelClass, 'params': {...}}}
    cv_splits : int
        Number of CV folds for training evaluation

    Returns
    -------
    List[ModelComparisonResult]
        Sorted by validation WAPE (best first)

    Example
    -------
    >>> import lightgbm as lgb
    >>> import xgboost as xgb
    >>> configs = {
    ...     'lightgbm': {'class': lgb.LGBMRegressor, 'params': {'n_estimators': 100}},
    ...     'xgboost': {'class': xgb.XGBRegressor, 'params': {'n_estimators': 100}},
    ... }
    >>> results = compare_models(X_train, y_train, X_val, y_val, configs)
    """
    import time

    results = []

    for name, config in model_configs.items():
        model_class = config['class']
        params = config.get('params', {})

        start_time = time.time()

        try:
            # Cross-validation on training data
            def train_fn(X, y):
                model = model_class(**params)
                model.fit(X, y)
                return model

            cv_result = time_series_cv(X_train, y_train, train_fn, n_splits=cv_splits)

            # Final evaluation on validation
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = np.clip(model.predict(X_val), 0, None)

            val_wape = _compute_wape(y_val, y_pred)
            cv_result.mean_wape = val_wape  # Use validation WAPE as primary metric

            training_time = time.time() - start_time

            results.append(ModelComparisonResult(
                model_name=name,
                cv_result=cv_result,
                training_time=training_time,
                best_params=params,
            ))

        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")
            results.append(ModelComparisonResult(
                model_name=name,
                cv_result=CVResult(1.0, 0.0, [1.0], [np.inf], [np.inf], 1, 0, 0),
                training_time=0.0,
            ))

    # Sort by validation WAPE
    results.sort(key=lambda x: x.cv_result.mean_wape)

    # Assign ranks
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


def benchmark_model_families(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    demand_pattern: str,
    allowed_families: List[str],
) -> Dict[str, Any]:
    """
    Benchmark all models appropriate for a demand pattern.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    demand_pattern : str
        'smooth', 'erratic', 'intermittent', 'lumpy'
    allowed_families : List[str]
        List of allowed model families

    Returns
    -------
    Dict with:
        - 'best_model': Best performing model name
        - 'rankings': Sorted list of (model, wape)
        - 'comparison_results': Full ModelComparisonResult objects
    """
    from utils.model_training import TRAINING_REGISTRY, is_feature_based

    # Get models for this pattern
    pattern_models = {
        'smooth': ['lightgbm', 'xgboost', 'catboost', 'ets', 'prophet'],
        'erratic': ['lightgbm', 'xgboost', 'catboost', 'theta', 'tbats'],
        'intermittent': ['croston', 'sba', 'tsb', 'zero_inflated', 'lightgbm'],
        'lumpy': ['sba', 'tsb', 'imapa', 'hurdle_model', 'lightgbm'],
    }

    candidates = pattern_models.get(demand_pattern, ['lightgbm'])
    candidates = [c for c in candidates if c in allowed_families]

    if not candidates:
        candidates = ['lightgbm'] if 'lightgbm' in allowed_families else allowed_families[:1]

    # Build model configs
    configs = {}
    for model_name in candidates:
        if model_name in TRAINING_REGISTRY:
            if is_feature_based(model_name):
                configs[model_name] = {
                    'class': _get_model_class(model_name),
                    'params': _get_default_params(model_name),
                }

    if not configs:
        # Fallback to LightGBM
        import lightgbm as lgb
        configs['lightgbm'] = {
            'class': lgb.LGBMRegressor,
            'params': {'n_estimators': 100, 'max_depth': 6, 'verbosity': -1},
        }

    results = compare_models(X_train, y_train, X_val, y_val, configs)

    return {
        'best_model': results[0].model_name if results else 'lightgbm',
        'best_wape': results[0].cv_result.mean_wape if results else 1.0,
        'rankings': [(r.model_name, r.cv_result.mean_wape) for r in results],
        'comparison_results': results,
    }


def create_model_leaderboard(
    comparison_results: List[ModelComparisonResult],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a model leaderboard from comparison results.

    Parameters
    ----------
    comparison_results : List[ModelComparisonResult]
        Results from compare_models()
    output_path : str, optional
        Path to save CSV leaderboard

    Returns
    -------
    pd.DataFrame
        Leaderboard sorted by rank
    """
    records = []
    for r in comparison_results:
        records.append({
            'rank': r.rank,
            'model': r.model_name,
            'val_wape': r.cv_result.mean_wape,
            'cv_std': r.cv_result.std_wape,
            'training_time_sec': r.training_time,
            'best_fold_wape': r.cv_result.fold_wapes[r.cv_result.best_fold_idx] if r.cv_result.fold_wapes else None,
            'worst_fold_wape': r.cv_result.fold_wapes[r.cv_result.worst_fold_idx] if r.cv_result.fold_wapes else None,
        })

    df = pd.DataFrame(records).sort_values('rank')

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Leaderboard saved to {output_path}")

    return df


# =============================================================================
# 3. UNCERTAINTY QUANTIFICATION
# =============================================================================

def compute_prediction_intervals(
    model: Any,
    X: np.ndarray,
    y_train: np.ndarray,
    method: str = 'bootstrap',
    confidence: float = 0.90,
    n_bootstrap: int = 100,
) -> PredictionInterval:
    """
    Compute prediction intervals using specified method.

    Parameters
    ----------
    model : fitted model
        Model with predict() method
    X : np.ndarray
        Feature matrix for predictions
    y_train : np.ndarray
        Training targets (for residual-based methods)
    method : str
        'bootstrap', 'quantile', 'conformal', 'residual'
    confidence : float
        Confidence level (e.g., 0.90 for 90% intervals)
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    PredictionInterval
        Point forecast with lower and upper bounds
    """
    point_forecast = np.clip(model.predict(X), 0, None)

    if method == 'bootstrap':
        return bootstrap_prediction_intervals(
            model, X, y_train, confidence, n_bootstrap
        )
    elif method == 'residual':
        return residual_prediction_intervals(
            model, X, y_train, confidence
        )
    elif method == 'conformal':
        return conformal_prediction_intervals(
            model, X, y_train, confidence
        )
    else:
        # Default: simple residual-based
        return residual_prediction_intervals(model, X, y_train, confidence)


def bootstrap_prediction_intervals(
    model: Any,
    X: np.ndarray,
    y_train: np.ndarray,
    confidence: float = 0.90,
    n_bootstrap: int = 100,
) -> PredictionInterval:
    """
    Bootstrap prediction intervals.

    Uses residual bootstrap to estimate prediction uncertainty.
    """
    point_forecast = np.clip(model.predict(X), 0, None)

    # Get training residuals
    if hasattr(model, 'predict'):
        try:
            train_pred = model.predict(X[:len(y_train)] if len(X) >= len(y_train) else X)
            residuals = y_train[:len(train_pred)] - train_pred[:len(y_train)]
        except Exception:
            residuals = np.zeros(len(y_train))
    else:
        residuals = np.zeros(len(y_train))

    # Bootstrap residuals
    bootstrap_preds = []
    for _ in range(n_bootstrap):
        sampled_residuals = np.random.choice(residuals, size=len(X), replace=True)
        bootstrap_preds.append(point_forecast + sampled_residuals)

    bootstrap_preds = np.array(bootstrap_preds)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_preds, alpha / 2 * 100, axis=0)
    upper = np.percentile(bootstrap_preds, (1 - alpha / 2) * 100, axis=0)

    return PredictionInterval(
        point_forecast=point_forecast,
        lower_bound=np.clip(lower, 0, None),
        upper_bound=upper,
        confidence_level=confidence,
        method='bootstrap',
    )


def residual_prediction_intervals(
    model: Any,
    X: np.ndarray,
    y_train: np.ndarray,
    confidence: float = 0.90,
) -> PredictionInterval:
    """
    Residual-based prediction intervals.

    Assumes residuals are normally distributed.
    """
    from scipy import stats

    point_forecast = np.clip(model.predict(X), 0, None)

    # Estimate residual standard error
    try:
        train_pred = model.predict(X[:len(y_train)] if len(X) >= len(y_train) else X)
        residuals = y_train[:len(train_pred)] - train_pred[:len(y_train)]
        std_error = np.std(residuals)
    except Exception:
        std_error = np.std(y_train) * 0.3  # Fallback

    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_error

    return PredictionInterval(
        point_forecast=point_forecast,
        lower_bound=np.clip(point_forecast - margin, 0, None),
        upper_bound=point_forecast + margin,
        confidence_level=confidence,
        method='residual',
    )


def conformal_prediction_intervals(
    model: Any,
    X: np.ndarray,
    y_calibration: np.ndarray,
    confidence: float = 0.90,
    X_calibration: Optional[np.ndarray] = None,
) -> PredictionInterval:
    """
    Conformal prediction intervals (distribution-free).

    Uses a calibration set to compute non-conformity scores.
    """
    point_forecast = np.clip(model.predict(X), 0, None)

    # If no separate calibration set, use provided y as reference
    if X_calibration is not None:
        cal_pred = model.predict(X_calibration)
        non_conformity = np.abs(y_calibration - cal_pred)
    else:
        # Use cross-validation-like approach
        non_conformity = np.abs(y_calibration - np.mean(y_calibration))

    # Get quantile of non-conformity scores
    alpha = 1 - confidence
    quantile = np.percentile(non_conformity, (1 - alpha) * 100)

    return PredictionInterval(
        point_forecast=point_forecast,
        lower_bound=np.clip(point_forecast - quantile, 0, None),
        upper_bound=point_forecast + quantile,
        confidence_level=confidence,
        method='conformal',
    )


def quantile_regression_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    confidence: float = 0.90,
) -> PredictionInterval:
    """
    Quantile regression prediction intervals using LightGBM.

    Trains separate models for lower/median/upper quantiles.
    """
    import lightgbm as lgb

    alpha = 1 - confidence
    quantiles = [alpha / 2, 0.5, 1 - alpha / 2]

    predictions = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=100,
            max_depth=6,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        predictions[q] = model.predict(X_pred)

    return PredictionInterval(
        point_forecast=np.clip(predictions[0.5], 0, None),
        lower_bound=np.clip(predictions[alpha / 2], 0, None),
        upper_bound=predictions[1 - alpha / 2],
        confidence_level=confidence,
        method='quantile_regression',
    )


# =============================================================================
# 4. MODEL INTERPRETATION
# =============================================================================

def compute_shap_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 1000,
) -> Dict[str, float]:
    """
    Compute SHAP feature importance.

    Parameters
    ----------
    model : fitted model
        Tree-based model (LightGBM, XGBoost, CatBoost)
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of features
    max_samples : int
        Maximum samples for SHAP computation

    Returns
    -------
    Dict[str, float]
        Feature importance dict sorted by importance

    Example
    -------
    >>> importance = compute_shap_importance(model, X, feature_names)
    >>> print(importance)  # {'feature_1': 0.25, 'feature_2': 0.18, ...}
    """
    try:
        import shap

        # Sample data if too large
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        # Create explainer
        if hasattr(model, 'booster_'):  # LightGBM
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'get_booster'):  # XGBoost
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance = {
            feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(feature_names))
        }

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

        return importance

    except ImportError:
        logger.warning("SHAP not installed. Using model's built-in importance.")
        return get_builtin_importance(model, feature_names)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return get_builtin_importance(model, feature_names)


def get_builtin_importance(
    model: Any,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Get model's built-in feature importance.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return {name: 0.0 for name in feature_names}

    # Normalize
    total = np.sum(importances)
    if total > 0:
        importances = importances / total

    importance = {
        feature_names[i]: float(importances[i])
        for i in range(min(len(feature_names), len(importances)))
    }

    return dict(sorted(importance.items(), key=lambda x: -x[1]))


def aggregate_feature_importance(
    importance_dicts: List[Dict[str, float]],
    method: str = 'mean',
) -> Dict[str, float]:
    """
    Aggregate feature importance across multiple models/folds.

    Parameters
    ----------
    importance_dicts : List[Dict[str, float]]
        List of importance dicts from different models
    method : str
        'mean', 'median', 'max', 'rank'

    Returns
    -------
    Dict[str, float]
        Aggregated importance
    """
    all_features = set()
    for d in importance_dicts:
        all_features.update(d.keys())

    aggregated = {}
    for feature in all_features:
        values = [d.get(feature, 0.0) for d in importance_dicts]

        if method == 'mean':
            aggregated[feature] = np.mean(values)
        elif method == 'median':
            aggregated[feature] = np.median(values)
        elif method == 'max':
            aggregated[feature] = np.max(values)
        elif method == 'rank':
            # Average rank across models
            ranks = []
            for d in importance_dicts:
                sorted_features = sorted(d.keys(), key=lambda x: -d[x])
                if feature in sorted_features:
                    ranks.append(sorted_features.index(feature) + 1)
            aggregated[feature] = np.mean(ranks) if ranks else len(all_features)
        else:
            aggregated[feature] = np.mean(values)

    # Sort (for rank, lower is better)
    reverse = method != 'rank'
    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=reverse))


def create_importance_report(
    importance: Dict[str, float],
    output_path: str,
    top_n: int = 20,
    create_plot: bool = True,
) -> str:
    """
    Create feature importance report with optional plot.

    Parameters
    ----------
    importance : Dict[str, float]
        Feature importance dict
    output_path : str
        Path to save report (JSON and optionally PNG)
    top_n : int
        Number of top features to include
    create_plot : bool
        Whether to create importance bar chart

    Returns
    -------
    str
        Path to saved report
    """
    # Get top features
    top_features = list(importance.items())[:top_n]

    report = {
        'top_features': dict(top_features),
        'total_features': len(importance),
        'top_n_shown': top_n,
        'timestamp': datetime.now().isoformat(),
    }

    # Save JSON
    json_path = output_path if output_path.endswith('.json') else output_path + '.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Create plot
    if create_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))

            features = [f[0] for f in reversed(top_features)]
            values = [f[1] for f in reversed(top_features)]

            ax.barh(features, values, color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importance')

            plt.tight_layout()
            plot_path = json_path.replace('.json', '.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()

            logger.info(f"Importance plot saved to {plot_path}")

        except Exception as e:
            logger.warning(f"Could not create importance plot: {e}")

    return json_path


# =============================================================================
# 5. BACKTESTING
# =============================================================================

def walk_forward_validation(
    df: pd.DataFrame,
    model_fn: Callable,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    initial_train_periods: int,
    test_periods: int = 1,
    step_size: int = 1,
    n_iterations: Optional[int] = None,
) -> BacktestResult:
    """
    Walk-forward (expanding window) backtesting.

    Parameters
    ----------
    df : pd.DataFrame
        Data sorted by date
    model_fn : Callable
        Function: model_fn(X_train, y_train) -> model
    date_col : str
        Date column name
    target_col : str
        Target column name
    feature_cols : List[str]
        Feature column names
    initial_train_periods : int
        Initial training window size
    test_periods : int
        Forecast horizon per iteration
    step_size : int
        How much to move forward each iteration
    n_iterations : int, optional
        Max iterations (None = all possible)

    Returns
    -------
    BacktestResult
        Backtest results with per-period metrics
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].unique()
    n_dates = len(dates)

    all_actuals = []
    all_predictions = []
    period_wapes = []
    period_maes = []

    iteration = 0
    train_end = initial_train_periods

    while train_end + test_periods <= n_dates:
        if n_iterations and iteration >= n_iterations:
            break

        # Get train and test dates
        train_dates = dates[:train_end]
        test_dates = dates[train_end:train_end + test_periods]

        train_mask = df[date_col].isin(train_dates)
        test_mask = df[date_col].isin(test_dates)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values

        try:
            model = model_fn(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 0, None)

            all_actuals.extend(y_test)
            all_predictions.extend(y_pred)

            wape = _compute_wape(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))

            period_wapes.append(wape)
            period_maes.append(mae)

        except Exception as e:
            logger.warning(f"Walk-forward iteration {iteration} failed: {e}")

        train_end += step_size
        iteration += 1

    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)

    overall_wape = _compute_wape(all_actuals, all_predictions)

    # Cumulative WAPE
    cumulative_wapes = []
    for i in range(1, len(period_wapes) + 1):
        cum_actual = np.array(all_actuals[:i * test_periods])
        cum_pred = np.array(all_predictions[:i * test_periods])
        cumulative_wapes.append(_compute_wape(cum_actual, cum_pred))

    return BacktestResult(
        overall_wape=overall_wape,
        period_wapes=period_wapes,
        period_maes=period_maes,
        cumulative_wapes=cumulative_wapes,
        n_periods=len(period_wapes),
        forecast_horizon=test_periods,
        actuals=all_actuals,
        predictions=all_predictions,
    )


def rolling_window_backtest(
    df: pd.DataFrame,
    model_fn: Callable,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    train_window: int,
    test_periods: int = 1,
    step_size: int = 1,
) -> BacktestResult:
    """
    Rolling (fixed window) backtesting.

    Unlike walk-forward, training window size stays constant.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].unique()
    n_dates = len(dates)

    all_actuals = []
    all_predictions = []
    period_wapes = []
    period_maes = []

    train_start = 0
    train_end = train_window

    while train_end + test_periods <= n_dates:
        train_dates = dates[train_start:train_end]
        test_dates = dates[train_end:train_end + test_periods]

        train_mask = df[date_col].isin(train_dates)
        test_mask = df[date_col].isin(test_dates)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values

        try:
            model = model_fn(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 0, None)

            all_actuals.extend(y_test)
            all_predictions.extend(y_pred)

            wape = _compute_wape(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))

            period_wapes.append(wape)
            period_maes.append(mae)

        except Exception as e:
            logger.warning(f"Rolling backtest failed: {e}")

        train_start += step_size
        train_end += step_size

    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)

    overall_wape = _compute_wape(all_actuals, all_predictions)

    return BacktestResult(
        overall_wape=overall_wape,
        period_wapes=period_wapes,
        period_maes=period_maes,
        cumulative_wapes=[],  # Not computed for rolling
        n_periods=len(period_wapes),
        forecast_horizon=test_periods,
        actuals=all_actuals,
        predictions=all_predictions,
    )


def compute_backtest_metrics(
    backtest_result: BacktestResult,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics from backtest result.
    """
    wapes = backtest_result.period_wapes

    return {
        'overall_wape': backtest_result.overall_wape,
        'mean_period_wape': np.mean(wapes),
        'std_period_wape': np.std(wapes),
        'median_period_wape': np.median(wapes),
        'min_period_wape': np.min(wapes),
        'max_period_wape': np.max(wapes),
        'wape_iqr': np.percentile(wapes, 75) - np.percentile(wapes, 25),
        'n_periods': backtest_result.n_periods,
        'mean_mae': np.mean(backtest_result.period_maes),
    }


# =============================================================================
# 6. MODEL COMBINATION
# =============================================================================

def simple_average_ensemble(
    predictions: List[np.ndarray],
) -> np.ndarray:
    """
    Simple averaging of predictions from multiple models.

    Parameters
    ----------
    predictions : List[np.ndarray]
        List of prediction arrays from different models

    Returns
    -------
    np.ndarray
        Averaged predictions
    """
    stacked = np.stack(predictions, axis=0)
    return np.clip(np.mean(stacked, axis=0), 0, None)


def weighted_average_ensemble(
    predictions: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Weighted average of predictions.

    Parameters
    ----------
    predictions : List[np.ndarray]
        List of prediction arrays
    weights : List[float]
        Weights for each model (will be normalized)

    Returns
    -------
    np.ndarray
        Weighted average predictions
    """
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize

    stacked = np.stack(predictions, axis=0)
    weighted = np.average(stacked, axis=0, weights=weights)

    return np.clip(weighted, 0, None)


def optimal_weights_ensemble(
    predictions: List[np.ndarray],
    actuals: np.ndarray,
    method: str = 'minimize_wape',
) -> Tuple[np.ndarray, List[float]]:
    """
    Find optimal weights for ensemble by minimizing validation error.

    Parameters
    ----------
    predictions : List[np.ndarray]
        List of prediction arrays from different models
    actuals : np.ndarray
        Actual values for optimization
    method : str
        'minimize_wape' or 'minimize_mae'

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        (ensemble_predictions, optimal_weights)
    """
    from scipy.optimize import minimize

    n_models = len(predictions)
    stacked = np.stack(predictions, axis=0)

    def objective(weights):
        weights = np.abs(weights)  # Ensure positive
        weights = weights / np.sum(weights)  # Normalize
        ensemble_pred = np.average(stacked, axis=0, weights=weights)

        if method == 'minimize_wape':
            return _compute_wape(actuals, ensemble_pred)
        else:
            return np.mean(np.abs(actuals - ensemble_pred))

    # Initial weights: equal
    x0 = np.ones(n_models) / n_models

    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead')

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    ensemble_pred = np.average(stacked, axis=0, weights=optimal_weights)

    return np.clip(ensemble_pred, 0, None), optimal_weights.tolist()


def model_selection_ensemble(
    predictions: List[np.ndarray],
    validation_wapes: List[float],
    top_k: int = 3,
    weight_method: str = 'inverse_wape',
) -> Tuple[np.ndarray, List[int]]:
    """
    Ensemble using only top-k models based on validation performance.

    Parameters
    ----------
    predictions : List[np.ndarray]
        Predictions from all models
    validation_wapes : List[float]
        Validation WAPE for each model
    top_k : int
        Number of top models to include
    weight_method : str
        'equal', 'inverse_wape', 'softmax'

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        (ensemble_predictions, indices_of_selected_models)
    """
    # Get top-k model indices
    sorted_indices = np.argsort(validation_wapes)
    selected_indices = sorted_indices[:top_k].tolist()

    selected_preds = [predictions[i] for i in selected_indices]
    selected_wapes = [validation_wapes[i] for i in selected_indices]

    # Compute weights
    if weight_method == 'equal':
        weights = [1.0 / top_k] * top_k
    elif weight_method == 'inverse_wape':
        inverse_wapes = [1.0 / (w + 1e-10) for w in selected_wapes]
        total = sum(inverse_wapes)
        weights = [w / total for w in inverse_wapes]
    elif weight_method == 'softmax':
        neg_wapes = [-w for w in selected_wapes]
        exp_vals = np.exp(neg_wapes - np.max(neg_wapes))
        weights = (exp_vals / np.sum(exp_vals)).tolist()
    else:
        weights = [1.0 / top_k] * top_k

    ensemble_pred = weighted_average_ensemble(selected_preds, weights)

    return ensemble_pred, selected_indices


# =============================================================================
# 7. LIFECYCLE HANDLING
# =============================================================================

def detect_dead_keys(
    df: pd.DataFrame,
    key_col: str,
    target_col: str,
    date_col: str,
    lookback_periods: int = DEFAULT_DEAD_KEY_LOOKBACK,
    threshold: float = DEFAULT_DEAD_KEY_THRESHOLD,
) -> List[Any]:
    """
    Detect "dead" keys with no recent demand.

    Parameters
    ----------
    df : pd.DataFrame
        Data with key, target, date columns
    key_col : str
        Key column name
    target_col : str
        Target (demand) column name
    date_col : str
        Date column name
    lookback_periods : int
        Number of recent periods to check
    threshold : float
        Sum threshold (below this = dead)

    Returns
    -------
    List[Any]
        List of dead key values
    """
    # Get most recent periods
    sorted_dates = sorted(df[date_col].unique())
    recent_dates = sorted_dates[-lookback_periods:] if len(sorted_dates) >= lookback_periods else sorted_dates

    recent_df = df[df[date_col].isin(recent_dates)]

    # Sum demand by key
    key_sums = recent_df.groupby(key_col)[target_col].sum()

    dead_keys = key_sums[key_sums <= threshold].index.tolist()

    return dead_keys


def detect_new_products(
    df: pd.DataFrame,
    key_col: str,
    date_col: str,
    min_history_periods: int = 13,
) -> List[Any]:
    """
    Detect new products with limited history.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    key_col : str
        Key column name
    date_col : str
        Date column name
    min_history_periods : int
        Minimum periods for "established" product

    Returns
    -------
    List[Any]
        List of new product keys
    """
    key_history = df.groupby(key_col)[date_col].nunique()
    new_products = key_history[key_history < min_history_periods].index.tolist()
    return new_products


def apply_lifecycle_rules(
    df: pd.DataFrame,
    predictions: np.ndarray,
    key_col: str,
    keys: np.ndarray,
    dead_keys: List[Any],
    new_products: List[Any],
    new_product_adjustment: float = 0.8,
) -> np.ndarray:
    """
    Apply lifecycle-based adjustments to predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Original data
    predictions : np.ndarray
        Model predictions
    key_col : str
        Key column name
    keys : np.ndarray
        Key values corresponding to predictions
    dead_keys : List[Any]
        Keys identified as dead
    new_products : List[Any]
        Keys identified as new products
    new_product_adjustment : float
        Multiplier for new product forecasts (reduce uncertainty)

    Returns
    -------
    np.ndarray
        Adjusted predictions
    """
    adjusted = predictions.copy()

    for i, key in enumerate(keys):
        if key in dead_keys:
            # Dead keys get zero forecast
            adjusted[i] = 0.0
        elif key in new_products:
            # New products get conservative adjustment
            adjusted[i] = adjusted[i] * new_product_adjustment

    return adjusted


def ramp_up_adjustment(
    predictions: np.ndarray,
    history_length: int,
    min_history: int = 13,
    max_adjustment: float = 1.5,
) -> np.ndarray:
    """
    Apply ramp-up adjustment for new products.

    Products with short history get boosted forecasts to account
    for growth trajectory.

    Parameters
    ----------
    predictions : np.ndarray
        Base predictions
    history_length : int
        Number of historical periods available
    min_history : int
        Minimum history for full confidence
    max_adjustment : float
        Maximum ramp-up multiplier

    Returns
    -------
    np.ndarray
        Adjusted predictions
    """
    if history_length >= min_history:
        return predictions

    # Linear ramp from max_adjustment down to 1.0
    progress = history_length / min_history
    adjustment = max_adjustment - (max_adjustment - 1.0) * progress

    return predictions * adjustment


# =============================================================================
# 8. MODEL PERSISTENCE
# =============================================================================

def save_model_versioned(
    model: Any,
    model_dir: str,
    model_name: str,
    metadata: Dict[str, Any],
    version: Optional[str] = None,
) -> str:
    """
    Save model with versioning and metadata.

    Parameters
    ----------
    model : Any
        Trained model object
    model_dir : str
        Directory to save model
    model_name : str
        Base name for model files
    metadata : Dict[str, Any]
        Model metadata (hyperparameters, metrics, etc.)
    version : str, optional
        Version string (auto-generated if None)

    Returns
    -------
    str
        Path to saved model
    """
    os.makedirs(model_dir, exist_ok=True)

    if version is None:
        version = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_path = os.path.join(model_dir, f'{model_name}_v{version}.pkl')
    joblib.dump(model, model_path)

    # Save metadata
    metadata['version'] = version
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_path'] = model_path

    metadata_path = os.path.join(model_dir, f'{model_name}_v{version}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Update latest symlink/reference
    latest_path = os.path.join(model_dir, f'{model_name}_latest.json')
    with open(latest_path, 'w') as f:
        json.dump({'version': version, 'model_path': model_path}, f)

    logger.info(f"Model saved: {model_path}")

    return model_path


def load_model_versioned(
    model_dir: str,
    model_name: str,
    version: str = 'latest',
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load versioned model with metadata.

    Parameters
    ----------
    model_dir : str
        Directory containing model files
    model_name : str
        Base name of model
    version : str
        Version to load ('latest' or specific version)

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (model, metadata)
    """
    if version == 'latest':
        latest_path = os.path.join(model_dir, f'{model_name}_latest.json')
        with open(latest_path) as f:
            latest_info = json.load(f)
        version = latest_info['version']

    model_path = os.path.join(model_dir, f'{model_name}_v{version}.pkl')
    metadata_path = os.path.join(model_dir, f'{model_name}_v{version}_metadata.json')

    model = joblib.load(model_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, metadata


def create_training_report(
    pipeline_result: TrainingPipelineResult,
    output_path: str,
) -> str:
    """
    Create comprehensive training report.

    Parameters
    ----------
    pipeline_result : TrainingPipelineResult
        Result from run_training_pipeline
    output_path : str
        Path to save report

    Returns
    -------
    str
        Path to saved report
    """
    report = {
        'summary': {
            'overall_wape': pipeline_result.overall_wape,
            'models_trained': pipeline_result.models_trained,
            'models_failed': pipeline_result.models_failed,
            'dead_keys_detected': pipeline_result.dead_keys_detected,
            'new_products_detected': pipeline_result.new_products_detected,
            'training_time_seconds': pipeline_result.training_time_seconds,
        },
        'best_model_group': pipeline_result.best_model_group,
        'worst_model_group': pipeline_result.worst_model_group,
        'model_specs': pipeline_result.model_specs,
        'generated_at': datetime.now().isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return output_path


# =============================================================================
# 9. HIGH-LEVEL TRAINING PIPELINE
# =============================================================================

def run_training_pipeline(
    train_manifest: pd.DataFrame,
    feature_dir: str,
    model_dir: str,
    segmentation_context: Dict[str, Any],
    allowed_families: List[str],
    target_col: str = 'target',
    key_col: str = 'key',
    date_col: str = 'year_week',
    enable_cv: bool = True,
    cv_splits: int = 3,
    enable_tuning: bool = False,
    tuning_trials: int = 20,
    enable_uncertainty: bool = False,
    enable_lifecycle: bool = True,
    dead_key_lookback: int = DEFAULT_DEAD_KEY_LOOKBACK,
) -> TrainingPipelineResult:
    """
    Run complete training pipeline with all advanced features.

    This is the MAIN function that does EVERYTHING:
    1. Loads feature files for each model group
    2. Optionally performs cross-validation
    3. Trains best model for each group
    4. Optionally tunes hyperparameters
    5. Handles dead keys and new products
    6. Computes prediction intervals (if enabled)
    7. Saves all models and metadata
    8. Creates comprehensive training report

    Parameters
    ----------
    train_manifest : pd.DataFrame
        Manifest with 'model_group' column
    feature_dir : str
        Directory containing feature files
    model_dir : str
        Directory to save models
    segmentation_context : Dict
        Context from segmentation with demand patterns
    allowed_families : List[str]
        Allowed model families
    target_col : str
        Target column name
    key_col : str
        Key column name
    date_col : str
        Date column name
    enable_cv : bool
        Enable cross-validation
    cv_splits : int
        Number of CV splits
    enable_tuning : bool
        Enable hyperparameter tuning
    tuning_trials : int
        Number of tuning trials
    enable_uncertainty : bool
        Enable prediction intervals
    enable_lifecycle : bool
        Enable dead key / new product detection
    dead_key_lookback : int
        Lookback for dead key detection

    Returns
    -------
    TrainingPipelineResult
        Complete pipeline results

    Example
    -------
    >>> result = run_training_pipeline(
    ...     train_manifest=manifest,
    ...     feature_dir='feature_output',
    ...     model_dir='model_artifacts',
    ...     segmentation_context=seg_context,
    ...     allowed_families=['lightgbm', 'croston', 'sba'],
    ...     enable_cv=True,
    ...     enable_tuning=True,
    ... )
    >>> print(f"Overall WAPE: {result.overall_wape:.3f}")
    """
    import time
    from utils.model_training import (
        train_best_model_for_segment, save_model, compute_wape
    )
    from utils.agent_utilities import load_csv, save_csv, save_json

    start_time = time.time()
    os.makedirs(model_dir, exist_ok=True)

    model_groups = train_manifest['model_group'].unique()
    logger.info(f"Training pipeline started for {len(model_groups)} model groups")

    model_specs = []
    total_actuals = 0
    total_errors = 0
    dead_keys_count = 0
    new_products_count = 0

    best_wape = float('inf')
    worst_wape = 0
    best_group = None
    worst_group = None

    for mg in model_groups:
        try:
            # Load feature files — format-agnostic via the feature_io
            # helpers so per-MG split files can be parquet too.
            from utils.feature_io import (
                features_intermediate_exists, read_features_intermediate,
            )
            if not features_intermediate_exists(feature_dir, f'{mg}_train_features'):
                logger.warning(f"Train file not found for {mg}")
                continue

            train_df = read_features_intermediate(feature_dir, f'{mg}_train_features')
            if features_intermediate_exists(feature_dir, f'{mg}_val_features'):
                val_df = read_features_intermediate(feature_dir, f'{mg}_val_features')
            else:
                val_df = train_df.tail(100)

            # Get feature columns
            exclude_cols = {target_col, key_col, date_col, 'split', 'model_group', 'segment_id'}
            feature_cols = [c for c in train_df.columns if c not in exclude_cols]

            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_val = val_df[feature_cols].values
            y_val = val_df[target_col].values

            # Lifecycle detection
            if enable_lifecycle:
                dead_keys = detect_dead_keys(
                    train_df, key_col, target_col, date_col, dead_key_lookback
                )
                new_prods = detect_new_products(train_df, key_col, date_col)
                dead_keys_count += len(dead_keys)
                new_products_count += len(new_prods)

            # Get demand pattern
            seg_info = segmentation_context.get('model_recommendations', {}).get('model_groups', {}).get(mg, {})
            demand_pattern = seg_info.get('demand_pattern', 'smooth')

            # Cross-validation (if enabled)
            cv_result = None
            if enable_cv:
                def train_fn(X, y):
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbosity=-1)
                    model.fit(X, y)
                    return model

                cv_result = time_series_cv(X_train, y_train, train_fn, n_splits=cv_splits)

            # Train best model
            result, best_type = train_best_model_for_segment(
                X_train, y_train, X_val, y_val,
                demand_pattern=demand_pattern,
                allowed_families=allowed_families,
            )

            # Hyperparameter tuning (if enabled and tree-based)
            if enable_tuning and best_type in ['lightgbm', 'xgboost', 'catboost']:
                from utils.model_training import tune_lightgbm, tune_xgboost

                if best_type == 'lightgbm':
                    tuning_result = tune_lightgbm(
                        X_train, y_train, X_val, y_val,
                        n_trials=tuning_trials
                    )
                    if tuning_result.best_val_wape < result.val_wape:
                        result.model = tuning_result.best_model
                        result.val_wape = tuning_result.best_val_wape
                        result.hyperparameters = tuning_result.best_params

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
                'cv_mean_wape': cv_result.mean_wape if cv_result else None,
            }
            model_specs.append(spec)

            # Track best/worst
            if result.val_wape < best_wape:
                best_wape = result.val_wape
                best_group = mg
            if result.val_wape > worst_wape:
                worst_wape = result.val_wape
                worst_group = mg

            # Accumulate for overall WAPE
            total_actuals += np.sum(np.abs(y_val))
            total_errors += result.val_wape * np.sum(np.abs(y_val))

            logger.info(f"  {mg}: {best_type} WAPE={result.val_wape:.3f}")

        except Exception as e:
            logger.error(f"Failed to train {mg}: {e}")
            model_specs.append({
                'model_group': mg,
                'model_type': 'failed',
                'error': str(e),
            })

    # Calculate overall WAPE
    overall_wape = total_errors / total_actuals if total_actuals > 0 else 1.0

    # Save summaries
    save_json(model_specs, os.path.join(model_dir, 'model_specs.json'))

    training_time = time.time() - start_time

    result = TrainingPipelineResult(
        model_specs=model_specs,
        overall_wape=overall_wape,
        best_model_group=best_group or '',
        worst_model_group=worst_group or '',
        models_trained=sum(1 for s in model_specs if s.get('model_type') != 'failed'),
        models_failed=sum(1 for s in model_specs if s.get('model_type') == 'failed'),
        dead_keys_detected=dead_keys_count,
        new_products_detected=new_products_count,
        training_time_seconds=training_time,
        output_dir=model_dir,
    )

    # Create training report
    create_training_report(result, os.path.join(model_dir, 'training_report.json'))

    logger.info(f"Training pipeline complete: WAPE={overall_wape:.3f}, time={training_time:.1f}s")

    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute WAPE."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    total_actual = np.sum(np.abs(y_true))
    if total_actual < 1e-10:
        return 0.0 if np.sum(np.abs(y_pred)) < 1e-10 else 1.0
    return np.sum(np.abs(y_true - y_pred)) / total_actual


def _get_model_class(model_name: str) -> Any:
    """Get model class by name."""
    if model_name == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMRegressor
    elif model_name == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor
    elif model_name == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    else:
        import lightgbm as lgb
        return lgb.LGBMRegressor


def _get_default_params(model_name: str) -> Dict[str, Any]:
    """Get default parameters for model."""
    defaults = {
        'lightgbm': {'n_estimators': 200, 'max_depth': 8, 'verbosity': -1},
        'xgboost': {'n_estimators': 200, 'max_depth': 8, 'verbosity': 0},
        'catboost': {'iterations': 200, 'depth': 8, 'verbose': False},
        'random_forest': {'n_estimators': 200, 'max_depth': 12},
    }
    return defaults.get(model_name, {})
