"""
State-of-the-Art Forecast Optimization Utilities.

This module provides advanced forecast optimization capabilities:

1. OPTIMAL ENSEMBLE WEIGHTS: Find best combination of models
2. BIAS CORRECTION: Detect and correct systematic forecast errors
3. FORECAST COMBINATION: Smart model averaging with optimal weights
4. CONFORMAL PREDICTION: Distribution-free prediction intervals
5. FORECAST CALIBRATION: Ensure probabilistic calibration

These techniques are proven in academic literature and M-competitions to
significantly improve forecast accuracy.

USAGE:
>>> from utils.forecast_optimization import (
...     # Ensemble optimization
...     find_optimal_ensemble_weights,
...     cross_validated_ensemble_weights,
...     constrained_optimal_weights,
...
...     # Bias correction
...     detect_systematic_bias,
...     apply_bias_correction,
...     rolling_bias_correction,
...
...     # Forecast combination
...     combine_forecasts_optimally,
...     inverse_mse_weights,
...     trimmed_mean_combination,
...
...     # Prediction intervals
...     conformal_prediction_intervals,
...     quantile_regression_intervals,
...     bootstrap_intervals,
...
...     # Calibration
...     calibrate_forecasts,
...     isotonic_calibration,
...     platt_scaling,
... )

Author: FEU-Agentic-Forecasting Demand Forecasting System
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum weight for ensemble members
MIN_ENSEMBLE_WEIGHT = 0.01

# Maximum number of models in ensemble
MAX_ENSEMBLE_SIZE = 10

# Default confidence level for prediction intervals
DEFAULT_CONFIDENCE_LEVEL = 0.95


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnsembleWeights:
    """Optimal ensemble weights."""
    weights: Dict[str, float]
    method: str
    cv_score: float
    n_models: int
    effective_n_models: float  # Accounts for weight concentration


@dataclass
class BiasCorrection:
    """Bias correction parameters."""
    additive_bias: float
    multiplicative_bias: float
    correction_method: str
    bias_significance: float  # p-value
    correction_applied: bool


@dataclass
class PredictionInterval:
    """Prediction interval result."""
    point_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    method: str
    coverage_empirical: Optional[float] = None


# =============================================================================
# 1. OPTIMAL ENSEMBLE WEIGHTS
# =============================================================================

def find_optimal_ensemble_weights(
    model_predictions: Dict[str, np.ndarray],
    actuals: np.ndarray,
    method: str = 'constrained_optimization',
    metric: str = 'wape',
    min_weight: float = 0.0,
    max_models: int = MAX_ENSEMBLE_SIZE,
) -> EnsembleWeights:
    """
    Find optimal weights for combining multiple model predictions.

    This is crucial for state-of-the-art forecasting as model combination
    almost always improves on individual models (M4 competition finding).

    Parameters
    ----------
    model_predictions : Dict[str, np.ndarray]
        Model name -> prediction array mapping
    actuals : np.ndarray
        Actual values
    method : str
        Optimization method: 'constrained_optimization', 'inverse_mse', 'equal'
    metric : str
        Metric to optimize: 'wape', 'mse', 'mae'
    min_weight : float
        Minimum weight for each model (0 allows dropping models)
    max_models : int
        Maximum number of models to include

    Returns
    -------
    EnsembleWeights
        Optimal weights with diagnostics

    Example
    -------
    >>> predictions = {
    ...     'lightgbm': lgb_pred,
    ...     'xgboost': xgb_pred,
    ...     'catboost': cat_pred,
    ... }
    >>> weights = find_optimal_ensemble_weights(predictions, actuals)
    >>> ensemble_pred = sum(w * predictions[m] for m, w in weights.weights.items())
    """
    model_names = list(model_predictions.keys())
    n_models = len(model_names)

    if n_models == 0:
        return EnsembleWeights(
            weights={}, method=method, cv_score=float('inf'),
            n_models=0, effective_n_models=0.0
        )

    if n_models == 1:
        return EnsembleWeights(
            weights={model_names[0]: 1.0}, method='single_model',
            cv_score=_compute_metric(model_predictions[model_names[0]], actuals, metric),
            n_models=1, effective_n_models=1.0
        )

    # Stack predictions
    pred_matrix = np.column_stack([model_predictions[m] for m in model_names])

    if method == 'equal':
        weights = {m: 1.0/n_models for m in model_names}
        combined = pred_matrix.mean(axis=1)
        score = _compute_metric(combined, actuals, metric)

    elif method == 'inverse_mse':
        weights = inverse_mse_weights(model_predictions, actuals)
        combined = sum(weights[m] * model_predictions[m] for m in model_names)
        score = _compute_metric(combined, actuals, metric)

    elif method == 'constrained_optimization':
        weights, score = constrained_optimal_weights(
            pred_matrix, actuals, model_names, metric, min_weight, max_models
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute effective number of models (inverse Herfindahl)
    weight_values = list(weights.values())
    herfindahl = sum(w**2 for w in weight_values)
    effective_n = 1.0 / herfindahl if herfindahl > 0 else 0.0

    return EnsembleWeights(
        weights=weights,
        method=method,
        cv_score=score,
        n_models=len([w for w in weight_values if w > min_weight]),
        effective_n_models=effective_n,
    )


def inverse_mse_weights(
    model_predictions: Dict[str, np.ndarray],
    actuals: np.ndarray,
) -> Dict[str, float]:
    """
    Compute weights inversely proportional to MSE.

    Simple but effective method: models with lower error get higher weights.
    """
    mses = {}
    for name, pred in model_predictions.items():
        mse = np.mean((pred - actuals) ** 2)
        mses[name] = mse + 1e-10  # Avoid division by zero

    # Inverse MSE
    inv_mses = {name: 1.0/mse for name, mse in mses.items()}
    total = sum(inv_mses.values())

    return {name: w/total for name, w in inv_mses.items()}


def constrained_optimal_weights(
    pred_matrix: np.ndarray,
    actuals: np.ndarray,
    model_names: List[str],
    metric: str = 'wape',
    min_weight: float = 0.0,
    max_models: int = MAX_ENSEMBLE_SIZE,
) -> Tuple[Dict[str, float], float]:
    """
    Find optimal weights using constrained optimization.

    Constraints:
    - Weights sum to 1
    - Each weight >= min_weight
    - At most max_models with non-zero weight
    """
    n_models = len(model_names)

    def objective(w):
        combined = pred_matrix @ w
        return _compute_metric(combined, actuals, metric)

    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
    ]

    # Bounds
    bounds = [(min_weight, 1.0) for _ in range(n_models)]

    try:
        result = optimize.minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )

        weights = result.x
        score = result.fun

    except Exception as e:
        logger.warning(f"Optimization failed: {e}, using equal weights")
        weights = x0
        score = objective(x0)

    # Enforce max_models by zeroing smallest weights
    if max_models < n_models:
        sorted_idx = np.argsort(weights)
        n_to_zero = n_models - max_models
        weights[sorted_idx[:n_to_zero]] = 0.0
        # Renormalize
        weights = weights / weights.sum()

    # Round very small weights to zero
    weights[weights < min_weight] = 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(n_models) / n_models

    return {model_names[i]: weights[i] for i in range(n_models)}, score


def cross_validated_ensemble_weights(
    model_predictions: Dict[str, np.ndarray],
    actuals: np.ndarray,
    n_folds: int = 5,
    metric: str = 'wape',
) -> EnsembleWeights:
    """
    Find ensemble weights using cross-validation to prevent overfitting.

    This prevents the weights from overfitting to the validation set
    by computing optimal weights on CV folds and averaging.
    """
    model_names = list(model_predictions.keys())
    n = len(actuals)

    fold_weights = []
    fold_scores = []

    fold_size = n // n_folds

    for i in range(n_folds):
        # Validation fold
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n

        # Training folds
        train_idx = list(range(0, val_start)) + list(range(val_end, n))
        val_idx = list(range(val_start, val_end))

        # Compute weights on training folds
        train_preds = {m: model_predictions[m][train_idx] for m in model_names}
        train_actuals = actuals[train_idx]

        weights_result = find_optimal_ensemble_weights(
            train_preds, train_actuals, method='constrained_optimization', metric=metric
        )
        fold_weights.append(weights_result.weights)

        # Evaluate on validation fold
        val_preds = {m: model_predictions[m][val_idx] for m in model_names}
        val_actuals = actuals[val_idx]

        combined = sum(weights_result.weights[m] * val_preds[m] for m in model_names)
        fold_scores.append(_compute_metric(combined, val_actuals, metric))

    # Average weights across folds
    avg_weights = {}
    for m in model_names:
        avg_weights[m] = np.mean([fw.get(m, 0.0) for fw in fold_weights])

    # Normalize
    total = sum(avg_weights.values())
    if total > 0:
        avg_weights = {m: w/total for m, w in avg_weights.items()}

    # Compute effective n
    weight_values = list(avg_weights.values())
    herfindahl = sum(w**2 for w in weight_values if w > 0)
    effective_n = 1.0 / herfindahl if herfindahl > 0 else 0.0

    return EnsembleWeights(
        weights=avg_weights,
        method='cross_validated',
        cv_score=np.mean(fold_scores),
        n_models=len([w for w in weight_values if w > 0.01]),
        effective_n_models=effective_n,
    )


def _compute_metric(pred: np.ndarray, actual: np.ndarray, metric: str) -> float:
    """Compute forecast metric."""
    if metric == 'wape':
        total_actual = np.sum(np.abs(actual))
        if total_actual < 1e-10:
            return 0.0 if np.sum(np.abs(pred)) < 1e-10 else 1.0
        return np.sum(np.abs(actual - pred)) / total_actual
    elif metric == 'mse':
        return np.mean((pred - actual) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(pred - actual))
    elif metric == 'rmse':
        return np.sqrt(np.mean((pred - actual) ** 2))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# 2. BIAS CORRECTION
# =============================================================================

def detect_systematic_bias(
    predictions: np.ndarray,
    actuals: np.ndarray,
    test_type: str = 'both',
    significance_level: float = 0.05,
) -> BiasCorrection:
    """
    Detect systematic bias in forecasts.

    Tests for both additive bias (constant offset) and multiplicative
    bias (proportional error). Returns correction parameters if significant.

    Parameters
    ----------
    predictions : np.ndarray
        Forecast values
    actuals : np.ndarray
        Actual values
    test_type : str
        'additive', 'multiplicative', or 'both'
    significance_level : float
        p-value threshold for significance

    Returns
    -------
    BiasCorrection
        Detected bias and correction parameters
    """
    errors = actuals - predictions
    n = len(errors)

    # Additive bias: mean error != 0
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    t_stat = mean_error / (std_error / np.sqrt(n)) if std_error > 0 else 0
    p_value_additive = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    additive_significant = p_value_additive < significance_level

    # Multiplicative bias: regression of actual on predicted
    # actual = alpha + beta * predicted + error
    # If beta != 1, there's multiplicative bias
    if np.std(predictions) > 1e-10:
        slope, intercept, r_value, p_value_mult, std_err = stats.linregress(
            predictions, actuals
        )
        multiplicative_bias = slope
    else:
        multiplicative_bias = 1.0
        p_value_mult = 1.0

    multiplicative_significant = (
        p_value_mult < significance_level and
        abs(multiplicative_bias - 1.0) > 0.05
    )

    # Determine correction method
    if additive_significant and multiplicative_significant:
        method = 'both'
    elif additive_significant:
        method = 'additive'
    elif multiplicative_significant:
        method = 'multiplicative'
    else:
        method = 'none'

    return BiasCorrection(
        additive_bias=mean_error if additive_significant else 0.0,
        multiplicative_bias=multiplicative_bias if multiplicative_significant else 1.0,
        correction_method=method,
        bias_significance=min(p_value_additive, p_value_mult),
        correction_applied=method != 'none',
    )


def apply_bias_correction(
    predictions: np.ndarray,
    bias_correction: BiasCorrection,
) -> np.ndarray:
    """
    Apply bias correction to forecasts.

    Order of operations:
    1. Apply multiplicative correction first (if needed)
    2. Apply additive correction second (if needed)
    """
    corrected = predictions.copy()

    if bias_correction.correction_method in ['multiplicative', 'both']:
        # Adjust for multiplicative bias
        # If slope is 1.1, predictions are too high -> divide by 1.1
        if abs(bias_correction.multiplicative_bias) > 1e-10:
            corrected = corrected / bias_correction.multiplicative_bias

    if bias_correction.correction_method in ['additive', 'both']:
        # Adjust for additive bias
        # If mean error is positive (under-prediction), add bias
        corrected = corrected + bias_correction.additive_bias

    # Ensure non-negative
    corrected = np.maximum(corrected, 0)

    return corrected


def rolling_bias_correction(
    predictions: np.ndarray,
    actuals: np.ndarray,
    window_size: int = 52,
    min_periods: int = 12,
) -> np.ndarray:
    """
    Apply rolling bias correction.

    This allows the bias correction to adapt over time as the
    forecast errors change (e.g., due to concept drift).
    """
    n = len(predictions)
    corrected = predictions.copy()

    for i in range(min_periods, n):
        start = max(0, i - window_size)

        # Compute bias on recent history
        recent_pred = predictions[start:i]
        recent_actual = actuals[start:i]

        bias_correction = detect_systematic_bias(recent_pred, recent_actual)

        # Apply correction to current prediction
        if bias_correction.correction_applied:
            corrected[i] = apply_bias_correction(
                np.array([predictions[i]]), bias_correction
            )[0]

    return corrected


# =============================================================================
# 3. FORECAST COMBINATION
# =============================================================================

def combine_forecasts_optimally(
    model_predictions: Dict[str, np.ndarray],
    actuals: Optional[np.ndarray] = None,
    method: str = 'auto',
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Combine multiple forecasts using the optimal method.

    If actuals are available, uses cross-validated optimal weights.
    Otherwise, uses equal weights or inverse variance.

    Parameters
    ----------
    model_predictions : Dict[str, np.ndarray]
        Model predictions
    actuals : np.ndarray, optional
        Actual values for weight optimization
    method : str
        'auto', 'optimal', 'equal', 'inverse_variance', 'trimmed_mean'

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        Combined forecast and weights used
    """
    model_names = list(model_predictions.keys())

    if len(model_names) == 0:
        return np.array([]), {}

    if len(model_names) == 1:
        return model_predictions[model_names[0]], {model_names[0]: 1.0}

    if method == 'auto':
        if actuals is not None:
            method = 'optimal'
        else:
            method = 'equal'

    if method == 'optimal' and actuals is not None:
        weights_result = cross_validated_ensemble_weights(model_predictions, actuals)
        weights = weights_result.weights
    elif method == 'inverse_variance':
        weights = inverse_mse_weights(
            model_predictions,
            actuals if actuals is not None else np.zeros(len(next(iter(model_predictions.values()))))
        )
    elif method == 'trimmed_mean':
        combined, weights = trimmed_mean_combination(model_predictions, trim_pct=0.2)
        return combined, weights
    else:  # equal
        weights = {m: 1.0/len(model_names) for m in model_names}

    combined = sum(weights.get(m, 0) * model_predictions[m] for m in model_names)

    return combined, weights


def trimmed_mean_combination(
    model_predictions: Dict[str, np.ndarray],
    trim_pct: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Combine forecasts using trimmed mean.

    For each time point, trims the highest and lowest predictions
    before averaging. Robust to outlier models.
    """
    model_names = list(model_predictions.keys())
    n_models = len(model_names)
    n_trim = max(1, int(n_models * trim_pct))

    if n_models <= 2:
        # Can't trim, use simple mean
        combined = np.mean([model_predictions[m] for m in model_names], axis=0)
        weights = {m: 1.0/n_models for m in model_names}
        return combined, weights

    # Stack predictions
    pred_stack = np.column_stack([model_predictions[m] for m in model_names])

    # Trim and average for each time point
    n_points = pred_stack.shape[0]
    combined = np.zeros(n_points)

    for i in range(n_points):
        sorted_preds = np.sort(pred_stack[i, :])
        trimmed = sorted_preds[n_trim:-n_trim] if n_trim > 0 else sorted_preds
        combined[i] = np.mean(trimmed)

    # Approximate weights (equal for retained models)
    weights = {m: 1.0/(n_models - 2*n_trim) for m in model_names}

    return combined, weights


# =============================================================================
# 4. CONFORMAL PREDICTION INTERVALS
# =============================================================================

def conformal_prediction_intervals(
    train_predictions: np.ndarray,
    train_actuals: np.ndarray,
    test_predictions: np.ndarray,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    method: str = 'absolute',
) -> PredictionInterval:
    """
    Compute distribution-free prediction intervals using conformal prediction.

    Conformal prediction provides valid prediction intervals without
    distributional assumptions, making it ideal for demand forecasting.

    Parameters
    ----------
    train_predictions : np.ndarray
        Predictions on calibration set
    train_actuals : np.ndarray
        Actuals on calibration set
    test_predictions : np.ndarray
        Predictions for which to compute intervals
    confidence_level : float
        Desired coverage (e.g., 0.95 for 95% intervals)
    method : str
        'absolute': |actual - pred|
        'signed': actual - pred (can be asymmetric)
        'normalized': (actual - pred) / pred (scale-independent)

    Returns
    -------
    PredictionInterval
        Point forecasts with lower and upper bounds
    """
    # Compute conformity scores on calibration set
    if method == 'absolute':
        scores = np.abs(train_actuals - train_predictions)
    elif method == 'signed':
        scores = train_actuals - train_predictions
    elif method == 'normalized':
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.abs((train_actuals - train_predictions) / (train_predictions + 1e-10))
            scores = np.nan_to_num(scores, nan=0, posinf=1, neginf=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find quantile for desired coverage
    n = len(scores)
    q = np.ceil((n + 1) * confidence_level) / n
    q = min(q, 1.0)

    if method == 'signed':
        # Asymmetric bounds
        lower_q = (1 - confidence_level) / 2
        upper_q = 1 - lower_q
        lower_threshold = np.quantile(scores, lower_q)
        upper_threshold = np.quantile(scores, upper_q)

        lower_bound = test_predictions + lower_threshold
        upper_bound = test_predictions + upper_threshold
    else:
        # Symmetric bounds
        threshold = np.quantile(scores, q)

        if method == 'normalized':
            lower_bound = test_predictions - threshold * np.abs(test_predictions)
            upper_bound = test_predictions + threshold * np.abs(test_predictions)
        else:
            lower_bound = test_predictions - threshold
            upper_bound = test_predictions + threshold

    # Ensure non-negative for demand
    lower_bound = np.maximum(lower_bound, 0)

    return PredictionInterval(
        point_forecast=test_predictions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
        method=f'conformal_{method}',
    )


def bootstrap_intervals(
    model_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_bootstrap: int = 100,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PredictionInterval:
    """
    Compute prediction intervals using bootstrap.

    Trains the model on multiple bootstrap samples and uses
    the distribution of predictions to form intervals.
    """
    n_train = len(y_train)
    n_test = len(X_test)

    predictions_matrix = np.zeros((n_bootstrap, n_test))

    for b in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_train, size=n_train, replace=True)
        X_b = X_train[idx]
        y_b = y_train[idx]

        # Train model
        try:
            model = model_fn(X_b, y_b)
            predictions_matrix[b, :] = model.predict(X_test)
        except Exception:
            predictions_matrix[b, :] = np.nan

    # Remove failed bootstraps
    valid_mask = ~np.isnan(predictions_matrix).any(axis=1)
    predictions_matrix = predictions_matrix[valid_mask]

    if len(predictions_matrix) < 10:
        # Not enough valid bootstraps
        point_forecast = np.mean(predictions_matrix, axis=0) if len(predictions_matrix) > 0 else np.zeros(n_test)
        return PredictionInterval(
            point_forecast=point_forecast,
            lower_bound=point_forecast,
            upper_bound=point_forecast,
            confidence_level=confidence_level,
            method='bootstrap_failed',
        )

    # Compute percentiles
    alpha = (1 - confidence_level) / 2
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100

    point_forecast = np.median(predictions_matrix, axis=0)
    lower_bound = np.percentile(predictions_matrix, lower_pct, axis=0)
    upper_bound = np.percentile(predictions_matrix, upper_pct, axis=0)

    # Ensure non-negative
    lower_bound = np.maximum(lower_bound, 0)

    return PredictionInterval(
        point_forecast=point_forecast,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
        method='bootstrap',
    )


def quantile_regression_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PredictionInterval:
    """
    Compute prediction intervals using quantile regression.

    Trains separate models for the lower and upper quantiles
    of the target distribution.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available for quantile regression")
        point_pred = np.zeros(len(X_test))
        return PredictionInterval(
            point_forecast=point_pred,
            lower_bound=point_pred,
            upper_bound=point_pred,
            confidence_level=confidence_level,
            method='quantile_regression_failed',
        )

    alpha = (1 - confidence_level) / 2

    # Point forecast (median)
    model_median = lgb.LGBMRegressor(
        objective='quantile', alpha=0.5, n_estimators=100, verbosity=-1
    )
    model_median.fit(X_train, y_train)
    point_forecast = model_median.predict(X_test)

    # Lower quantile
    model_lower = lgb.LGBMRegressor(
        objective='quantile', alpha=alpha, n_estimators=100, verbosity=-1
    )
    model_lower.fit(X_train, y_train)
    lower_bound = model_lower.predict(X_test)

    # Upper quantile
    model_upper = lgb.LGBMRegressor(
        objective='quantile', alpha=1-alpha, n_estimators=100, verbosity=-1
    )
    model_upper.fit(X_train, y_train)
    upper_bound = model_upper.predict(X_test)

    # Ensure non-negative and monotonicity
    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = np.maximum(upper_bound, lower_bound)
    point_forecast = np.clip(point_forecast, lower_bound, upper_bound)

    return PredictionInterval(
        point_forecast=point_forecast,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
        method='quantile_regression',
    )


# =============================================================================
# 5. FORECAST CALIBRATION
# =============================================================================

def calibrate_forecasts(
    predictions: np.ndarray,
    actuals: np.ndarray,
    test_predictions: np.ndarray,
    method: str = 'isotonic',
) -> np.ndarray:
    """
    Calibrate forecasts to improve reliability.

    Calibration ensures that predicted values match actual values
    on average (e.g., when we predict 100, the average actual is 100).

    Parameters
    ----------
    predictions : np.ndarray
        Training predictions for calibration
    actuals : np.ndarray
        Training actuals for calibration
    test_predictions : np.ndarray
        Predictions to calibrate
    method : str
        'isotonic': Isotonic regression (non-parametric, monotonic)
        'platt': Platt scaling (logistic-like)
        'linear': Simple linear recalibration

    Returns
    -------
    np.ndarray
        Calibrated predictions
    """
    if method == 'isotonic':
        return isotonic_calibration(predictions, actuals, test_predictions)
    elif method == 'platt':
        return platt_scaling(predictions, actuals, test_predictions)
    elif method == 'linear':
        return linear_calibration(predictions, actuals, test_predictions)
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def isotonic_calibration(
    train_predictions: np.ndarray,
    train_actuals: np.ndarray,
    test_predictions: np.ndarray,
) -> np.ndarray:
    """
    Isotonic regression calibration.

    Fits a non-decreasing function that maps predictions to calibrated values.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(train_predictions, train_actuals)

    calibrated = iso.predict(test_predictions)

    # Ensure non-negative
    calibrated = np.maximum(calibrated, 0)

    return calibrated


def platt_scaling(
    train_predictions: np.ndarray,
    train_actuals: np.ndarray,
    test_predictions: np.ndarray,
) -> np.ndarray:
    """
    Platt scaling calibration.

    Fits a sigmoid-like transformation (for demand, we use a modified version).
    """
    from sklearn.linear_model import LinearRegression

    # For demand forecasting, we use linear regression with log transform
    log_pred = np.log1p(train_predictions)
    log_actual = np.log1p(train_actuals)

    lr = LinearRegression()
    lr.fit(log_pred.reshape(-1, 1), log_actual)

    log_test = np.log1p(test_predictions)
    calibrated_log = lr.predict(log_test.reshape(-1, 1))
    calibrated = np.expm1(calibrated_log)

    # Ensure non-negative
    calibrated = np.maximum(calibrated, 0)

    return calibrated


def linear_calibration(
    train_predictions: np.ndarray,
    train_actuals: np.ndarray,
    test_predictions: np.ndarray,
) -> np.ndarray:
    """
    Simple linear recalibration.

    Fits actual = a + b * predicted and applies to test predictions.
    """
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(train_predictions.reshape(-1, 1), train_actuals)

    calibrated = lr.predict(test_predictions.reshape(-1, 1))

    # Ensure non-negative
    calibrated = np.maximum(calibrated, 0)

    return calibrated


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'EnsembleWeights',
    'BiasCorrection',
    'PredictionInterval',

    # Ensemble optimization
    'find_optimal_ensemble_weights',
    'inverse_mse_weights',
    'constrained_optimal_weights',
    'cross_validated_ensemble_weights',

    # Bias correction
    'detect_systematic_bias',
    'apply_bias_correction',
    'rolling_bias_correction',

    # Forecast combination
    'combine_forecasts_optimally',
    'trimmed_mean_combination',

    # Prediction intervals
    'conformal_prediction_intervals',
    'bootstrap_intervals',
    'quantile_regression_intervals',

    # Calibration
    'calibrate_forecasts',
    'isotonic_calibration',
    'platt_scaling',
    'linear_calibration',
]
