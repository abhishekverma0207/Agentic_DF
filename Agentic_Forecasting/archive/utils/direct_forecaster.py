# utils/direct_forecaster.py
"""
Phase 4: Direct Forecasting & Hybrid Recursive-Direct Strategy.

The key insight: recursive forecasting compounds prediction errors at longer
lags. At lag 4, all lag features (lag_1, lag_2, lag_3) are filled with
predictions — each carrying error from prior steps.

**Direct forecasting** eliminates this by training SEPARATE models for each
horizon. The lag-4 model uses only features available at the forecast origin
(lag_4+, rolling stats shifted by 4, etc.) — no intermediate predictions.

**Hybrid strategy** combines the best of both:
- Lag 1-2: Recursive (highest accuracy for short horizons)
- Lag 3+: Direct (avoids error compounding)
- Blending zone: weighted average of both

This module provides:
1. `DirectForecaster` — generates predictions from pre-trained direct models
2. `HybridForecaster` — combines recursive + direct with lag-adaptive weights
3. `train_direct_models_per_lag()` — trains separate models for each target lag
4. `compute_lag_specific_calibration()` — learns bias factors per (segment, lag)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HybridForecastConfig:
    """Configuration for hybrid recursive-direct forecasting."""

    # Below this lag, use recursive forecasting (more accurate for short horizons)
    recursive_max_lag: int = 2

    # Above this lag, use direct forecasting (avoids error compounding)
    direct_min_lag: int = 3

    # Blending weights for transition zone (if recursive_max_lag >= direct_min_lag)
    # At exactly direct_min_lag: blend_alpha * direct + (1-blend_alpha) * recursive
    blend_alpha_at_transition: float = 0.6

    # How fast to increase direct weight beyond transition
    # At lag = direct_min_lag + k: alpha = min(1.0, blend_alpha + k * alpha_increment)
    alpha_increment_per_lag: float = 0.15

    # Whether to clip predictions to [0, +inf)
    clip_min: float = 0.0

    def get_direct_weight(self, lag: int) -> float:
        """Get the weight for direct forecast at a given lag."""
        if lag <= self.recursive_max_lag:
            return 0.0  # Pure recursive
        if lag >= self.direct_min_lag:
            k = lag - self.direct_min_lag
            return min(1.0, self.blend_alpha_at_transition + k * self.alpha_increment_per_lag)
        # Transition zone
        return self.blend_alpha_at_transition * 0.5


# =============================================================================
# DIRECT FORECASTER
# =============================================================================

class DirectForecaster:
    """
    Generate forecasts using pre-trained direct (per-horizon) models.

    Each model was trained to predict target[t+h] from features[t], where h
    is the specific horizon. No intermediate predictions are needed.

    Parameters
    ----------
    models : Dict[int, Any]
        {lag: trained_model} — one model per target horizon
    feature_cols : List[str]
        Feature columns the models expect
    """

    def __init__(
        self,
        models: Dict[int, Any],
        feature_cols: List[str],
        clip_min: float = 0.0,
    ):
        self.models = models  # {lag: model}
        self.feature_cols = feature_cols
        self.clip_min = clip_min
        self.available_lags = sorted(models.keys())

        logger.info(f"DirectForecaster initialized with models for lags: {self.available_lags}")

    def forecast_at_lag(
        self,
        X_origin: np.ndarray,
        lag: int,
    ) -> np.ndarray:
        """
        Predict at a specific lag using the direct model for that lag.

        Parameters
        ----------
        X_origin : np.ndarray
            Feature values at the forecast origin. Shape (n_keys, n_features)
            or (n_features,) for single key.
        lag : int
            Target horizon (e.g., 4 for 4-step-ahead)

        Returns
        -------
        np.ndarray
            Predictions, shape (n_keys,) or scalar
        """
        if lag not in self.models:
            # Fall back to nearest available lag
            nearest = min(self.available_lags, key=lambda l: abs(l - lag))
            logger.debug(f"No direct model for lag {lag}, using lag {nearest}")
            lag = nearest

        model = self.models[lag]
        X = np.atleast_2d(X_origin)
        preds = model.predict(X)
        return np.maximum(preds, self.clip_min)

    def forecast_all_lags(
        self,
        X_origin: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Predict at all available lags from the same origin features."""
        results = {}
        X = np.atleast_2d(X_origin)
        for lag in self.available_lags:
            results[lag] = np.maximum(self.models[lag].predict(X), self.clip_min)
        return results


# =============================================================================
# HYBRID FORECASTER
# =============================================================================

class HybridForecaster:
    """
    Combines recursive and direct forecasting with lag-adaptive weights.

    For short horizons (lag 1-2): uses recursive forecasting
    For longer horizons (lag 3+): uses direct forecasting
    In between: weighted blend that increases direct weight with lag

    Parameters
    ----------
    recursive_forecaster : Any
        RecursiveForecaster instance (or any object with .forecast())
    direct_forecaster : DirectForecaster
        DirectForecaster instance
    config : HybridForecastConfig
        Blending configuration
    """

    def __init__(
        self,
        recursive_forecaster: Any,
        direct_forecaster: Optional[DirectForecaster],
        config: Optional[HybridForecastConfig] = None,
    ):
        self.recursive = recursive_forecaster
        self.direct = direct_forecaster
        self.config = config or HybridForecastConfig()

    def forecast(
        self,
        historical_actuals: np.ndarray,
        future_features: np.ndarray,
        feature_names: List[str],
        horizon: int,
        origin_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate hybrid forecasts combining recursive and direct approaches.

        Parameters
        ----------
        historical_actuals : np.ndarray
            Historical actual values for this key
        future_features : np.ndarray
            Future feature matrix, shape (horizon, n_features)
        feature_names : List[str]
            Feature column names
        horizon : int
            Number of periods to forecast
        origin_features : np.ndarray, optional
            Features at the forecast origin (for direct models).
            If None, uses future_features[0].

        Returns
        -------
        np.ndarray
            Combined predictions, shape (horizon,)
        """
        # Always get recursive forecasts (needed for short lags and blending)
        recursive_preds = self.recursive.forecast(
            historical_actuals, future_features, feature_names, horizon
        )

        # If no direct models, return pure recursive
        if self.direct is None or not self.direct.available_lags:
            return recursive_preds

        # Get origin features for direct models
        if origin_features is None:
            origin_features = future_features[0]

        # Combine lag by lag
        combined = np.zeros(horizon)
        for step in range(horizon):
            lag = step + 1  # lag is 1-indexed
            w_direct = self.config.get_direct_weight(lag)

            if w_direct <= 0 or lag not in self.direct.models:
                # Pure recursive
                combined[step] = recursive_preds[step]
            elif w_direct >= 1.0:
                # Pure direct
                combined[step] = self.direct.forecast_at_lag(origin_features, lag)[0]
            else:
                # Blend
                pred_recursive = recursive_preds[step]
                pred_direct = self.direct.forecast_at_lag(origin_features, lag)[0]
                combined[step] = w_direct * pred_direct + (1 - w_direct) * pred_recursive

        return np.maximum(combined, self.config.clip_min)


# =============================================================================
# TRAINING: DIRECT MODELS PER LAG
# =============================================================================

def train_direct_models_per_lag(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    key_col: str,
    date_col: str,
    max_horizon: int = 8,
    min_lag: int = 1,
    model_family: str = 'lightgbm',
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """
    Train separate models for each forecast horizon (direct forecasting).

    For each lag h in [min_lag, max_horizon]:
    - Creates target: y[t+h] for each (key, t) pair
    - Trains model using features at time t to predict y[t+h]
    - Features are the SAME for all lags (computed at the origin time t)

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data sorted by [key, date]
    val_df : pd.DataFrame
        Validation data
    feature_cols : List[str]
        Feature columns (available at time t)
    target_col : str
        Target column
    key_col : str
        Key column
    date_col : str
        Date column
    max_horizon : int
        Maximum forecast horizon
    min_lag : int
        Minimum lag to train direct model for (default 1)
    model_family : str
        'lightgbm' or 'xgboost'

    Returns
    -------
    Tuple[Dict[int, model], Dict[str, Any]]
        ({lag: trained_model}, metadata with per-lag WAPE)
    """
    logger.info(f"Training direct models for lags {min_lag}-{max_horizon} ({model_family})")

    models = {}
    metadata = {'per_lag_wape': {}, 'per_lag_n_samples': {}}

    # For each lag, create the shifted target and train
    for lag in range(min_lag, max_horizon + 1):
        # Create lag-h target: for each row at time t, target is value at t+lag
        train_targets = []
        train_features = []
        val_targets = []
        val_features = []

        for key_val, group in train_df.groupby(key_col):
            group = group.sort_values(date_col)
            y = group[target_col].values
            X = group[feature_cols].fillna(0).values

            # For direct forecasting: feature[t] → target[t+lag]
            if len(y) > lag:
                train_features.append(X[:len(y) - lag])
                train_targets.append(y[lag:])

        if not train_features:
            logger.warning(f"No training samples for lag {lag}, skipping")
            continue

        X_train = np.vstack(train_features)
        y_train = np.concatenate(train_targets)

        # Validation: same logic
        for key_val, group in val_df.groupby(key_col):
            group = group.sort_values(date_col)
            y = group[target_col].values
            X = group[feature_cols].fillna(0).values
            if len(y) > lag:
                val_features.append(X[:len(y) - lag])
                val_targets.append(y[lag:])

        if val_features:
            X_val = np.vstack(val_features)
            y_val = np.concatenate(val_targets)
        else:
            # No validation data for this lag — use last 10% of training as holdout
            # This is suboptimal but avoids using the same data for train AND val
            n_holdout = max(10, len(X_train) // 10)
            X_val, y_val = X_train[-n_holdout:], y_train[-n_holdout:]
            logger.warning(f"  Lag {lag}: no val data, using last {n_holdout} train rows as holdout")

        # Train model
        model = _train_single_model(X_train, y_train, X_val, y_val, model_family)
        models[lag] = model

        # Compute validation WAPE
        val_preds = np.maximum(model.predict(X_val), 0)
        total_actual = np.sum(np.abs(y_val))
        wape = float(np.sum(np.abs(y_val - val_preds)) / (total_actual + 1e-10))

        metadata['per_lag_wape'][lag] = wape
        metadata['per_lag_n_samples'][lag] = len(y_train)
        logger.info(f"  Lag {lag}: {len(y_train):,} samples, val_wape={wape:.4f}")

    metadata['n_models'] = len(models)
    metadata['model_family'] = model_family

    return models, metadata


# =============================================================================
# LAG-SPECIFIC BIAS CALIBRATION
# =============================================================================

def compute_lag_specific_calibration(
    backtest_df: pd.DataFrame,
    actual_col: str = 'actual',
    predicted_col: str = 'predicted',
    lag_col: str = 'lag',
    segment_col: str = 'segment_id',
    factor_min: float = 0.3,
    factor_max: float = 3.0,
) -> Dict[str, float]:
    """
    Compute bias calibration factors per (segment, lag) combination.

    This extends the existing segment × zero_fraction calibration with a lag
    dimension, so lag 1 and lag 4 can have different correction factors.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Backtesting results with actual, predicted, lag, and segment columns
    actual_col, predicted_col, lag_col, segment_col : str
        Column names
    factor_min, factor_max : float
        Clipping bounds for calibration factors

    Returns
    -------
    Dict[str, float]
        Calibration factors keyed by f"{segment}_lag_{lag}"
    """
    if lag_col not in backtest_df.columns:
        logger.warning(f"No '{lag_col}' column in backtest_df, cannot compute lag-specific calibration")
        return {}

    factors = {}
    n_computed = 0

    for (seg, lag), group in backtest_df.groupby([segment_col, lag_col]):
        sum_actual = group[actual_col].sum()
        sum_pred = group[predicted_col].sum()

        if sum_pred < 1e-10:
            factor = 1.0
        else:
            factor = sum_actual / sum_pred

        factor = float(np.clip(factor, factor_min, factor_max))
        key = f"{seg}_lag_{int(lag)}"
        factors[key] = factor
        n_computed += 1

    # Also compute per-lag global factors (fallback when segment unknown)
    for lag, group in backtest_df.groupby(lag_col):
        sum_actual = group[actual_col].sum()
        sum_pred = group[predicted_col].sum()
        if sum_pred > 1e-10:
            factors[f"global_lag_{int(lag)}"] = float(
                np.clip(sum_actual / sum_pred, factor_min, factor_max)
            )

    logger.info(f"Computed {n_computed} lag-specific calibration factors "
                f"across {backtest_df[segment_col].nunique()} segments × "
                f"{backtest_df[lag_col].nunique()} lags")

    return factors


def apply_lag_specific_calibration(
    predictions_df: pd.DataFrame,
    lag_factors: Dict[str, float],
    predicted_col: str = 'predicted',
    lag_col: str = 'lag',
    segment_col: str = 'segment_id',
) -> pd.DataFrame:
    """
    Apply lag-specific calibration factors to predictions.

    Lookup priority:
    1. segment × lag specific: f"{segment}_lag_{lag}"
    2. global lag: f"global_lag_{lag}"
    3. No adjustment (factor = 1.0)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with lag and segment columns
    lag_factors : Dict[str, float]
        Calibration factors from compute_lag_specific_calibration()
    predicted_col, lag_col, segment_col : str
        Column names

    Returns
    -------
    pd.DataFrame
        Calibrated predictions
    """
    if not lag_factors:
        return predictions_df

    result = predictions_df.copy()

    def _get_factor(row):
        seg = str(row.get(segment_col, 'unknown'))
        lag = int(row.get(lag_col, 1))
        # Try segment-specific first
        key = f"{seg}_lag_{lag}"
        if key in lag_factors:
            return lag_factors[key]
        # Fallback to global lag
        key = f"global_lag_{lag}"
        if key in lag_factors:
            return lag_factors[key]
        return 1.0

    factors = result.apply(_get_factor, axis=1)
    result[predicted_col] = result[predicted_col] * factors
    result[predicted_col] = result[predicted_col].clip(lower=0)

    applied = (factors != 1.0).sum()
    logger.info(f"Applied lag-specific calibration to {applied}/{len(result)} predictions")

    return result


# =============================================================================
# HELPERS
# =============================================================================

def _train_single_model(X_train, y_train, X_val, y_val, model_family='lightgbm'):
    """Train a single model."""
    if model_family == 'lightgbm':
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, verbosity=-1,
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            return model
        except ImportError:
            pass

    if model_family == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, verbosity=0,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return model
        except ImportError:
            pass

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
