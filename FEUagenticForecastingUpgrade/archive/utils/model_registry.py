# utils/model_registry.py
"""
State-of-the-Art Forecasting Model Registry
============================================

This module provides a unified interface for ALL model families used in demand forecasting.
Agents MUST use these implementations to ensure:
1. Proper model selection based on demand patterns
2. Consistent API across all model types
3. Correct handling of intermittent demand (Croston, SBA, TSB)
4. Proper recursive forecasting during prediction

Model Categories:
    1. Tree-Based (universal): lightgbm, xgboost, catboost, random_forest
    2. Statistical (smooth demand): arima, sarima, ets, theta, tbats
    3. Bayesian/Probabilistic: bsts, prophet
    4. Intermittent Specialists: croston, sba, tsb, imapa
    5. Compound/Hurdle (zero-inflated): zero_inflated, hurdle_model, tweedie
    6. Deep Learning: tft, lstm, nbeats, deepar, wavenet
    7. Ensemble: weighted_ensemble, stacking

Pattern-to-Model Recommendations:
    SMOOTH (low CV, low zero%): sarima, ets, prophet, lightgbm, xgboost
    ERRATIC (high CV, low zero%): xgboost, lightgbm, theta, bsts
    INTERMITTENT (low CV, high zero%): croston, sba, tsb, zero_inflated
    LUMPY (high CV, high zero%): hurdle_model, zero_inflated, tsb, imapa

Usage:
    from utils.model_registry import get_model, get_recommended_models, MODEL_REGISTRY

    # Get recommended models for a demand pattern
    models = get_recommended_models('intermittent', allowed_families=['croston', 'sba', 'lightgbm'])

    # Create and train a model
    model = get_model('croston', alpha=0.1)
    model.fit(y_train, X_train)
    predictions = model.predict(horizon=8, X=X_test)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class ModelMetadata:
    """Metadata about a trained model."""
    model_type: str
    demand_pattern: str  # smooth, erratic, intermittent, lumpy
    hyperparameters: Dict[str, Any]
    is_feature_based: bool  # True for tree-based, False for univariate
    supports_recursive: bool  # True if model can be used in recursive forecasting
    val_wape: Optional[float] = None
    training_samples: Optional[int] = None


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.

    All forecasters MUST implement:
    - fit(): Train the model
    - predict(): Generate predictions
    - get_metadata(): Return model metadata

    Forecasters can be either:
    - Feature-based (tree models): Use X features for prediction
    - Univariate (statistical): Use only historical y values
    """

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.hyperparameters = kwargs
        self.is_fitted = False
        self._feature_names: Optional[List[str]] = None

    @abstractmethod
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'BaseForecaster':
        """
        Fit the model.

        Parameters
        ----------
        y : np.ndarray
            Target values (historical demand)
        X : np.ndarray, optional
            Feature matrix (for feature-based models)

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        horizon : int
            Number of periods to forecast
        X : np.ndarray, optional
            Feature matrix for prediction periods (for feature-based models)

        Returns
        -------
        np.ndarray
            Predicted values, shape (horizon,) or (n_samples,)
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        pass

    def is_feature_based(self) -> bool:
        """Return True if model uses features, False if univariate."""
        return False

    def supports_recursive(self) -> bool:
        """Return True if model supports recursive multi-step forecasting."""
        return True


# =============================================================================
# TREE-BASED MODELS (Feature-based, Universal)
# =============================================================================

class LightGBMForecaster(BaseForecaster):
    """LightGBM gradient boosting regressor."""

    def __init__(self, **kwargs):
        # Default hyperparameters
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
        }
        # Override with user params
        params = {**self.default_params, **kwargs}
        super().__init__('lightgbm', **params)
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'LightGBMForecaster':
        import lightgbm as lgb

        if X is None:
            raise ValueError("LightGBM requires feature matrix X")

        self.model = lgb.LGBMRegressor(**self.hyperparameters)
        self.model.fit(X, y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            raise ValueError("LightGBM requires feature matrix X for prediction")

        preds = self.model.predict(X)
        return np.clip(preds, 0, None)  # No negative demand

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='lightgbm',
            demand_pattern='universal',
            hyperparameters=self.hyperparameters,
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


class XGBoostForecaster(BaseForecaster):
    """XGBoost gradient boosting regressor."""

    def __init__(self, **kwargs):
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'objective': 'reg:squarederror',
            'verbosity': 0,
            'random_state': 42,
            'n_jobs': -1,
        }
        params = {**self.default_params, **kwargs}
        super().__init__('xgboost', **params)
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'XGBoostForecaster':
        import xgboost as xgb

        if X is None:
            raise ValueError("XGBoost requires feature matrix X")

        self.model = xgb.XGBRegressor(**self.hyperparameters)
        self.model.fit(X, y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            raise ValueError("XGBoost requires feature matrix X for prediction")

        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='xgboost',
            demand_pattern='universal',
            hyperparameters=self.hyperparameters,
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


class CatBoostForecaster(BaseForecaster):
    """CatBoost gradient boosting regressor."""

    def __init__(self, **kwargs):
        self.default_params = {
            'iterations': 200,
            'depth': 8,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
        }
        params = {**self.default_params, **kwargs}
        super().__init__('catboost', **params)
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'CatBoostForecaster':
        from catboost import CatBoostRegressor

        if X is None:
            raise ValueError("CatBoost requires feature matrix X")

        self.model = CatBoostRegressor(**self.hyperparameters)
        self.model.fit(X, y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            raise ValueError("CatBoost requires feature matrix X for prediction")

        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='catboost',
            demand_pattern='universal',
            hyperparameters=self.hyperparameters,
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


class RandomForestForecaster(BaseForecaster):
    """Random Forest regressor."""

    def __init__(self, **kwargs):
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
        }
        params = {**self.default_params, **kwargs}
        super().__init__('random_forest', **params)
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'RandomForestForecaster':
        from sklearn.ensemble import RandomForestRegressor

        if X is None:
            raise ValueError("RandomForest requires feature matrix X")

        self.model = RandomForestRegressor(**self.hyperparameters)
        self.model.fit(X, y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            raise ValueError("RandomForest requires feature matrix X for prediction")

        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='random_forest',
            demand_pattern='universal',
            hyperparameters=self.hyperparameters,
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


# =============================================================================
# INTERMITTENT DEMAND SPECIALISTS (Univariate, for sparse/lumpy demand)
# =============================================================================

class CrostonForecaster(BaseForecaster):
    """
    Croston's method for intermittent demand forecasting.

    Croston's method separately forecasts:
    1. Demand size (when demand occurs)
    2. Inter-arrival interval (time between demands)

    Final forecast = demand_size / interval

    Best for: INTERMITTENT demand (low CV, high zero%)
    """

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__('croston', alpha=alpha, **kwargs)
        self.alpha = alpha
        self.demand_level = None
        self.interval = None
        self._last_demand_idx = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'CrostonForecaster':
        """
        Fit Croston's method using exponential smoothing on:
        - Non-zero demand values
        - Inter-demand intervals
        """
        y = np.asarray(y).flatten()
        non_zero_idx = np.where(y > 0)[0]

        if len(non_zero_idx) == 0:
            # No demand at all - predict zero
            self.demand_level = 0
            self.interval = float('inf')
            self._last_demand_idx = len(y)
            self.is_fitted = True
            self._training_samples = len(y)
            return self

        # Initialize with first non-zero demand
        self.demand_level = y[non_zero_idx[0]]
        self.interval = non_zero_idx[0] + 1 if non_zero_idx[0] > 0 else 1

        # Update with exponential smoothing
        for i in range(1, len(non_zero_idx)):
            idx = non_zero_idx[i]
            prev_idx = non_zero_idx[i - 1]

            # Update demand level (exponential smoothing on non-zero values)
            self.demand_level = self.alpha * y[idx] + (1 - self.alpha) * self.demand_level

            # Update interval (exponential smoothing on inter-arrival times)
            inter_arrival = idx - prev_idx
            self.interval = self.alpha * inter_arrival + (1 - self.alpha) * self.interval

        self._last_demand_idx = non_zero_idx[-1]
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate flat forecast for h periods."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.interval == 0 or self.interval == float('inf') or self.demand_level == 0:
            return np.zeros(horizon)

        # Croston forecast: expected demand per period
        forecast = self.demand_level / self.interval
        return np.full(horizon, max(0, forecast))

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='croston',
            demand_pattern='intermittent',
            hyperparameters={'alpha': self.alpha},
            is_feature_based=False,
            supports_recursive=False,  # Flat forecast
            training_samples=getattr(self, '_training_samples', None),
        )


class SBAForecaster(CrostonForecaster):
    """
    Syntetos-Boylan Approximation (SBA) - bias-corrected Croston.

    SBA corrects for the positive bias in Croston's method by applying
    a debiasing factor: forecast = croston_forecast * (1 - alpha/2)

    Best for: INTERMITTENT demand with more accuracy than Croston
    """

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.model_type = 'sba'

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate SBA forecast with bias correction."""
        croston_forecast = super().predict(horizon, X)

        # SBA debiasing correction
        correction_factor = 1 - self.alpha / 2
        return croston_forecast * correction_factor

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='sba',
            demand_pattern='intermittent',
            hyperparameters={'alpha': self.alpha},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class TSBForecaster(BaseForecaster):
    """
    Teunter-Syntetos-Babai (TSB) method with demand probability decay.

    TSB differs from Croston/SBA by modeling demand probability decay:
    - Probability of demand decreases during periods of no demand
    - Better for items with obsolescence risk

    Separately smooths:
    1. Demand size (when demand occurs)
    2. Demand probability (decays when no demand)

    Best for: LUMPY demand or items with potential obsolescence
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1, **kwargs):
        super().__init__('tsb', alpha=alpha, beta=beta, **kwargs)
        self.alpha = alpha  # Smoothing for demand size
        self.beta = beta    # Decay rate for demand probability
        self.demand_level = None
        self.demand_prob = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'TSBForecaster':
        """
        Fit TSB method:
        - Demand level: exponential smoothing on non-zero values
        - Demand probability: increases on demand, decays on no-demand
        """
        y = np.asarray(y).flatten()

        # Initialize
        non_zero = y[y > 0]
        if len(non_zero) == 0:
            self.demand_level = 0
            self.demand_prob = 0
            self.is_fitted = True
            self._training_samples = len(y)
            return self

        self.demand_level = np.mean(non_zero)
        self.demand_prob = len(non_zero) / len(y)

        # Update with exponential smoothing
        for val in y:
            if val > 0:
                # Demand occurred: update demand level and increase probability
                self.demand_level = self.alpha * val + (1 - self.alpha) * self.demand_level
                self.demand_prob = self.alpha * 1.0 + (1 - self.alpha) * self.demand_prob
            else:
                # No demand: decay probability (demand level stays same)
                self.demand_prob = (1 - self.beta) * self.demand_prob

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate TSB forecast for h periods."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # TSB forecast: demand_level * demand_probability
        forecast = self.demand_level * self.demand_prob
        return np.full(horizon, max(0, forecast))

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='tsb',
            demand_pattern='lumpy',
            hyperparameters={'alpha': self.alpha, 'beta': self.beta},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class IMAPAForecaster(BaseForecaster):
    """
    Intermittent Multiple Aggregation Prediction Algorithm (IMAPA).

    IMAPA uses temporal aggregation at multiple levels to reduce
    noise in intermittent demand, then reconciles forecasts.

    Process:
    1. Aggregate demand at multiple time scales (weekly, monthly, etc.)
    2. Apply forecasting method at each level
    3. Reconcile forecasts across levels

    Best for: Very sparse LUMPY demand with high noise
    """

    def __init__(self, aggregation_levels: List[int] = None, base_method: str = 'croston', **kwargs):
        super().__init__('imapa', aggregation_levels=aggregation_levels, base_method=base_method, **kwargs)
        self.aggregation_levels = aggregation_levels or [1, 4, 13]  # weekly, monthly, quarterly
        self.base_method = base_method
        self.level_models = {}
        self.level_weights = {}

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'IMAPAForecaster':
        """Fit models at each aggregation level."""
        y = np.asarray(y).flatten()

        for level in self.aggregation_levels:
            # Aggregate to this level
            n_complete = (len(y) // level) * level
            if n_complete < level:
                continue

            y_trimmed = y[:n_complete]
            y_agg = y_trimmed.reshape(-1, level).sum(axis=1)

            # Fit a Croston/SBA model at this level
            if self.base_method == 'sba':
                model = SBAForecaster(alpha=0.1)
            else:
                model = CrostonForecaster(alpha=0.1)

            model.fit(y_agg)
            self.level_models[level] = model

            # Weight based on number of non-zero observations
            non_zero_ratio = np.mean(y_agg > 0)
            self.level_weights[level] = non_zero_ratio

        # Normalize weights
        total_weight = sum(self.level_weights.values())
        if total_weight > 0:
            self.level_weights = {k: v / total_weight for k, v in self.level_weights.items()}

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate IMAPA forecast by reconciling across levels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if not self.level_models:
            return np.zeros(horizon)

        # Get forecasts from each level and disaggregate
        level_forecasts = []
        for level, model in self.level_models.items():
            # Forecast at aggregated level
            agg_horizon = max(1, (horizon + level - 1) // level)
            agg_forecast = model.predict(agg_horizon)[0]

            # Disaggregate to base level (uniform distribution)
            base_forecast = agg_forecast / level

            weight = self.level_weights.get(level, 1.0 / len(self.level_models))
            level_forecasts.append(base_forecast * weight)

        # Weighted average
        final_forecast = sum(level_forecasts)
        return np.full(horizon, max(0, final_forecast))

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='imapa',
            demand_pattern='lumpy',
            hyperparameters={
                'aggregation_levels': self.aggregation_levels,
                'base_method': self.base_method,
            },
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


# =============================================================================
# COMPOUND / HURDLE MODELS (Two-stage: occurrence + severity)
# =============================================================================

class ZeroInflatedForecaster(BaseForecaster):
    """
    Two-stage zero-inflated model.

    Stage 1: Classification - P(demand > 0)
    Stage 2: Regression - E[demand | demand > 0]

    Final prediction = P(demand > 0) * E[demand | demand > 0]

    Best for: INTERMITTENT/LUMPY demand with many zeros
    """

    def __init__(self, classifier_type: str = 'logistic', regressor_type: str = 'lightgbm', **kwargs):
        super().__init__('zero_inflated', classifier_type=classifier_type, regressor_type=regressor_type, **kwargs)
        self.classifier_type = classifier_type
        self.regressor_type = regressor_type
        self.classifier = None
        self.regressor = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'ZeroInflatedForecaster':
        """
        Fit two-stage model:
        1. Classifier on all data (predict zero vs non-zero)
        2. Regressor on positive cases only
        """
        if X is None:
            raise ValueError("ZeroInflated requires feature matrix X")

        y = np.asarray(y).flatten()
        X = np.asarray(X)

        # Stage 1: Binary classifier
        y_binary = (y > 0).astype(int)

        if self.classifier_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            import lightgbm as lgb
            self.classifier = lgb.LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1, random_state=42)

        self.classifier.fit(X, y_binary)

        # Stage 2: Regressor on positive cases
        positive_mask = y > 0
        if np.sum(positive_mask) > 10:  # Enough positive samples
            X_pos = X[positive_mask]
            y_pos = y[positive_mask]

            if self.regressor_type == 'lightgbm':
                import lightgbm as lgb
                self.regressor = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1, random_state=42)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                self.regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

            self.regressor.fit(X_pos, y_pos)
        else:
            # Not enough positive samples - use mean
            self._mean_positive = np.mean(y[positive_mask]) if np.sum(positive_mask) > 0 else 0
            self.regressor = None

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate zero-inflated predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            raise ValueError("ZeroInflated requires feature matrix X for prediction")

        X = np.asarray(X)

        # Stage 1: Probability of non-zero demand
        if hasattr(self.classifier, 'predict_proba'):
            prob_positive = self.classifier.predict_proba(X)[:, 1]
        else:
            prob_positive = self.classifier.predict(X).astype(float)

        # Stage 2: Expected demand given positive
        if self.regressor is not None:
            expected_positive = self.regressor.predict(X)
        else:
            expected_positive = np.full(len(X), getattr(self, '_mean_positive', 0))

        # Combined forecast
        forecast = prob_positive * np.clip(expected_positive, 0, None)
        return np.clip(forecast, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='zero_inflated',
            demand_pattern='intermittent',
            hyperparameters={
                'classifier_type': self.classifier_type,
                'regressor_type': self.regressor_type,
            },
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


class HurdleModelForecaster(BaseForecaster):
    """
    Hurdle model - similar to zero-inflated but with different interpretation.

    The "hurdle" is whether any demand occurs at all:
    Stage 1: Binary classification - Did we cross the hurdle?
    Stage 2: Truncated regression - How much demand, given we crossed?

    Best for: LUMPY demand where zeros are structurally different
    """

    def __init__(self, classifier_type: str = 'lightgbm', regressor_type: str = 'lightgbm', **kwargs):
        super().__init__('hurdle_model', classifier_type=classifier_type, regressor_type=regressor_type, **kwargs)
        self.classifier_type = classifier_type
        self.regressor_type = regressor_type
        self.classifier = None
        self.regressor = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'HurdleModelForecaster':
        """Fit hurdle model with stronger classification emphasis."""
        if X is None:
            raise ValueError("HurdleModel requires feature matrix X")

        y = np.asarray(y).flatten()
        X = np.asarray(X)

        # Stage 1: Binary classifier with balanced weights for zeros
        y_binary = (y > 0).astype(int)

        if self.classifier_type == 'lightgbm':
            import lightgbm as lgb
            # Use class weights to handle imbalanced data
            self.classifier = lgb.LGBMClassifier(
                n_estimators=150, max_depth=6,
                class_weight='balanced',
                verbose=-1, random_state=42
            )
        else:
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

        self.classifier.fit(X, y_binary)

        # Stage 2: Regressor on positive cases with emphasis on magnitude
        positive_mask = y > 0
        if np.sum(positive_mask) > 10:
            X_pos = X[positive_mask]
            y_pos = y[positive_mask]

            if self.regressor_type == 'lightgbm':
                import lightgbm as lgb
                self.regressor = lgb.LGBMRegressor(
                    n_estimators=150, max_depth=6,
                    objective='regression',
                    verbose=-1, random_state=42
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                self.regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

            self.regressor.fit(X_pos, y_pos)
        else:
            self._mean_positive = np.mean(y[positive_mask]) if np.sum(positive_mask) > 0 else 0
            self.regressor = None

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate hurdle model predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            raise ValueError("HurdleModel requires feature matrix X for prediction")

        X = np.asarray(X)

        # Stage 1: Probability of crossing hurdle
        if hasattr(self.classifier, 'predict_proba'):
            prob_positive = self.classifier.predict_proba(X)[:, 1]
        else:
            prob_positive = self.classifier.predict(X).astype(float)

        # Stage 2: Expected demand given positive
        if self.regressor is not None:
            expected_positive = self.regressor.predict(X)
        else:
            expected_positive = np.full(len(X), getattr(self, '_mean_positive', 0))

        # Combined forecast
        forecast = prob_positive * np.clip(expected_positive, 0, None)
        return np.clip(forecast, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='hurdle_model',
            demand_pattern='lumpy',
            hyperparameters={
                'classifier_type': self.classifier_type,
                'regressor_type': self.regressor_type,
            },
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


class TweedieForecaster(BaseForecaster):
    """
    Tweedie regression for zero-inflated continuous data.

    Tweedie distribution with power parameter between 1 and 2 naturally
    handles the mix of zeros and positive values in demand data.

    Uses LightGBM with tweedie objective for efficiency.

    Best for: INTERMITTENT/LUMPY demand (handles zeros natively)
    """

    def __init__(self, tweedie_variance_power: float = 1.5, **kwargs):
        super().__init__('tweedie', tweedie_variance_power=tweedie_variance_power, **kwargs)
        self.tweedie_power = tweedie_variance_power
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'objective': 'tweedie',
            'tweedie_variance_power': tweedie_variance_power,
            'verbosity': -1,
            'random_state': 42,
        }
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'TweedieForecaster':
        """Fit Tweedie model using LightGBM."""
        import lightgbm as lgb

        if X is None:
            raise ValueError("Tweedie requires feature matrix X")

        y = np.asarray(y).flatten()
        X = np.asarray(X)

        # Tweedie requires y >= 0
        y = np.clip(y, 0, None)

        params = {**self.default_params, **self.hyperparameters}
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X, y)

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate Tweedie predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            raise ValueError("Tweedie requires feature matrix X for prediction")

        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='tweedie',
            demand_pattern='intermittent',
            hyperparameters={'tweedie_variance_power': self.tweedie_power},
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


# =============================================================================
# CLASSICAL STATISTICAL MODELS (Univariate, for smooth demand)
# =============================================================================

class ARIMAForecaster(BaseForecaster):
    """
    ARIMA model for smooth demand patterns.
    Uses auto_arima for automatic order selection.

    Best for: SMOOTH demand with trend
    """

    def __init__(self, seasonal: bool = False, m: int = 1, **kwargs):
        super().__init__('arima', seasonal=seasonal, m=m, **kwargs)
        self.seasonal = seasonal
        self.m = m
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'ARIMAForecaster':
        """Fit ARIMA using pmdarima auto_arima."""
        try:
            from pmdarima import auto_arima
        except ImportError:
            logger.warning("pmdarima not installed. Using simple exponential smoothing fallback.")
            self._use_fallback = True
            self._mean = np.mean(y)
            self.is_fitted = True
            self._training_samples = len(y)
            return self

        y = np.asarray(y).flatten()

        # Handle zeros and negative values
        y = np.clip(y, 0, None)

        try:
            self.model = auto_arima(
                y,
                seasonal=self.seasonal,
                m=self.m if self.seasonal else 1,
                suppress_warnings=True,
                error_action='ignore',
                max_p=3, max_q=3, max_d=2,
                max_P=2, max_Q=2, max_D=1,
                stepwise=True,
                n_fits=10,
            )
            self._use_fallback = False
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}. Using fallback.")
            self._use_fallback = True
            self._mean = np.mean(y)

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate ARIMA forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if getattr(self, '_use_fallback', False):
            return np.full(horizon, max(0, self._mean))

        try:
            forecast = self.model.predict(n_periods=horizon)
            return np.clip(forecast, 0, None)
        except Exception:
            return np.full(horizon, max(0, getattr(self, '_mean', 0)))

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='arima',
            demand_pattern='smooth',
            hyperparameters={'seasonal': self.seasonal, 'm': self.m},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class SARIMAForecaster(ARIMAForecaster):
    """SARIMA model - ARIMA with seasonality."""

    def __init__(self, m: int = 52, **kwargs):
        super().__init__(seasonal=True, m=m, **kwargs)
        self.model_type = 'sarima'

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='sarima',
            demand_pattern='smooth',
            hyperparameters={'seasonal': True, 'm': self.m},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class ETSForecaster(BaseForecaster):
    """
    Exponential Smoothing (ETS) model.

    Handles error, trend, and seasonality components.
    Best for: SMOOTH demand with clear patterns
    """

    def __init__(self, trend: str = 'add', seasonal: str = None, seasonal_periods: int = 52, **kwargs):
        super().__init__('ets', trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, **kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'ETSForecaster':
        """Fit ETS model."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = np.asarray(y).flatten()
        y = np.clip(y, 0.01, None)  # ETS needs positive values

        try:
            # Determine if we have enough data for seasonality
            if self.seasonal and len(y) >= 2 * self.seasonal_periods:
                self.model = ExponentialSmoothing(
                    y,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                    initialization_method='estimated',
                ).fit(optimized=True)
            else:
                # No seasonality if not enough data
                self.model = ExponentialSmoothing(
                    y,
                    trend=self.trend,
                    seasonal=None,
                    initialization_method='estimated',
                ).fit(optimized=True)
            self._use_fallback = False
        except Exception as e:
            logger.warning(f"ETS fitting failed: {e}. Using fallback.")
            self._use_fallback = True
            self._mean = np.mean(y)

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate ETS forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if getattr(self, '_use_fallback', False):
            return np.full(horizon, max(0, self._mean))

        try:
            forecast = self.model.forecast(horizon)
            return np.clip(forecast, 0, None)
        except Exception:
            return np.full(horizon, max(0, getattr(self, '_mean', 0)))

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='ets',
            demand_pattern='smooth',
            hyperparameters={
                'trend': self.trend,
                'seasonal': self.seasonal,
                'seasonal_periods': self.seasonal_periods,
            },
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class ThetaForecaster(BaseForecaster):
    """
    Theta method - decomposes series into trend and seasonality.

    Simple but effective for M3 competition.
    Best for: ERRATIC demand with trend
    """

    def __init__(self, **kwargs):
        super().__init__('theta', **kwargs)
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'ThetaForecaster':
        """Fit Theta model."""
        from statsmodels.tsa.forecasting.theta import ThetaModel

        y = np.asarray(y).flatten()
        y = np.clip(y, 0, None)

        try:
            self.model = ThetaModel(y).fit()
            self._use_fallback = False
        except Exception as e:
            logger.warning(f"Theta fitting failed: {e}. Using fallback.")
            self._use_fallback = True
            self._mean = np.mean(y)

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate Theta forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if getattr(self, '_use_fallback', False):
            return np.full(horizon, max(0, self._mean))

        try:
            forecast = self.model.forecast(horizon)
            return np.clip(forecast, 0, None)
        except Exception:
            return np.full(horizon, max(0, getattr(self, '_mean', 0)))

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='theta',
            demand_pattern='erratic',
            hyperparameters={},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class TBATSForecaster(BaseForecaster):
    """
    TBATS - handles complex seasonality patterns.

    T: Trigonometric seasonality
    B: Box-Cox transformation
    A: ARMA errors
    T: Trend
    S: Seasonal

    Best for: SMOOTH demand with multiple seasonalities
    """

    def __init__(self, seasonal_periods: List[int] = None, **kwargs):
        super().__init__('tbats', seasonal_periods=seasonal_periods, **kwargs)
        self.seasonal_periods = seasonal_periods or [52]
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'TBATSForecaster':
        """Fit TBATS model."""
        try:
            from tbats import TBATS
        except ImportError:
            logger.warning("tbats not installed. Using ETS fallback.")
            self._fallback = ETSForecaster()
            self._fallback.fit(y)
            self.is_fitted = True
            self._training_samples = len(y)
            return self

        y = np.asarray(y).flatten()
        y = np.clip(y, 0, None)

        try:
            estimator = TBATS(seasonal_periods=self.seasonal_periods)
            self.model = estimator.fit(y)
            self._use_fallback = False
        except Exception as e:
            logger.warning(f"TBATS fitting failed: {e}. Using ETS fallback.")
            self._fallback = ETSForecaster()
            self._fallback.fit(y)
            self._use_fallback = True

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate TBATS forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if getattr(self, '_use_fallback', False) or hasattr(self, '_fallback'):
            return self._fallback.predict(horizon)

        try:
            forecast = self.model.forecast(steps=horizon)
            return np.clip(forecast, 0, None)
        except Exception:
            return np.zeros(horizon)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='tbats',
            demand_pattern='smooth',
            hyperparameters={'seasonal_periods': self.seasonal_periods},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


# =============================================================================
# BAYESIAN / PROBABILISTIC MODELS
# =============================================================================

class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet for time series forecasting.

    Handles trend, seasonality, and holidays well.
    Best for: SMOOTH demand with multiple seasonalities
    """

    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = False, **kwargs):
        super().__init__('prophet', yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, **kwargs)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model = None
        self._dates = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, dates: Optional[np.ndarray] = None) -> 'ProphetForecaster':
        """
        Fit Prophet model.

        Parameters
        ----------
        y : np.ndarray
            Target values
        X : np.ndarray, optional
            Not used (Prophet is univariate with dates)
        dates : np.ndarray, optional
            Date values for Prophet's 'ds' column
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.warning("prophet not installed. Using ETS fallback.")
            self._fallback = ETSForecaster()
            self._fallback.fit(y)
            self.is_fitted = True
            self._training_samples = len(y)
            return self

        y = np.asarray(y).flatten()
        y = np.clip(y, 0, None)

        # Create dates if not provided
        if dates is None:
            dates = pd.date_range(end='2025-01-01', periods=len(y), freq='W')

        self._dates = dates

        # Prepare Prophet dataframe
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': y
        })

        try:
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=False,
            )
            self.model.fit(df)
            self._use_fallback = False
        except Exception as e:
            logger.warning(f"Prophet fitting failed: {e}. Using ETS fallback.")
            self._fallback = ETSForecaster()
            self._fallback.fit(y)
            self._use_fallback = True

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate Prophet forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if getattr(self, '_use_fallback', False) or hasattr(self, '_fallback'):
            return self._fallback.predict(horizon)

        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=horizon, freq='W')
            forecast = self.model.predict(future)

            # Return only the forecast periods
            predictions = forecast['yhat'].values[-horizon:]
            return np.clip(predictions, 0, None)
        except Exception:
            return np.zeros(horizon)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='prophet',
            demand_pattern='smooth',
            hyperparameters={
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
            },
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class BSTSForecaster(BaseForecaster):
    """
    Bayesian Structural Time Series.

    Uses orbit-ml as Python implementation.
    Best for: ERRATIC demand with uncertainty quantification
    """

    def __init__(self, seasonality: int = 52, **kwargs):
        super().__init__('bsts', seasonality=seasonality, **kwargs)
        self.seasonality = seasonality
        self.model = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, dates: Optional[np.ndarray] = None) -> 'BSTSForecaster':
        """Fit BSTS using orbit-ml DLT model."""
        try:
            from orbit.models import DLT
        except ImportError:
            logger.warning("orbit-ml not installed. Using ETS fallback.")
            self._fallback = ETSForecaster()
            self._fallback.fit(y)
            self.is_fitted = True
            self._training_samples = len(y)
            return self

        y = np.asarray(y).flatten()
        y = np.clip(y, 0.01, None)  # Orbit needs positive values

        # Create dates if not provided
        if dates is None:
            dates = pd.date_range(end='2025-01-01', periods=len(y), freq='W')

        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': y
        })

        try:
            self.model = DLT(
                response_col='y',
                date_col='ds',
                seasonality=self.seasonality,
            )
            self.model.fit(df)
            self._last_date = df['ds'].max()
            self._use_fallback = False
        except Exception as e:
            logger.warning(f"BSTS/DLT fitting failed: {e}. Using ETS fallback.")
            self._fallback = ETSForecaster()
            self._fallback.fit(y)
            self._use_fallback = True

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate BSTS forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if getattr(self, '_use_fallback', False) or hasattr(self, '_fallback'):
            return self._fallback.predict(horizon)

        try:
            future_dates = pd.date_range(start=self._last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W')
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = self.model.predict(future_df)
            predictions = forecast['prediction'].values
            return np.clip(predictions, 0, None)
        except Exception:
            return np.zeros(horizon)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='bsts',
            demand_pattern='erratic',
            hyperparameters={'seasonality': self.seasonality},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


# =============================================================================
# ENSEMBLE MODELS
# =============================================================================

class WeightedEnsembleForecaster(BaseForecaster):
    """
    Weighted ensemble of multiple forecasters.

    Combines predictions from multiple models using weights.
    """

    def __init__(self, models: List[BaseForecaster] = None, weights: List[float] = None, **kwargs):
        super().__init__('weighted_ensemble', **kwargs)
        self.models = models or []
        self.weights = weights

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'WeightedEnsembleForecaster':
        """Fit all component models."""
        for model in self.models:
            if model.is_feature_based():
                model.fit(y, X)
            else:
                model.fit(y)

        # If weights not provided, use equal weights
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate weighted ensemble prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = []
        for model, weight in zip(self.models, self.weights):
            if model.is_feature_based():
                pred = model.predict(horizon, X)
            else:
                pred = model.predict(horizon)
            predictions.append(pred * weight)

        ensemble_pred = np.sum(predictions, axis=0)
        return np.clip(ensemble_pred, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='weighted_ensemble',
            demand_pattern='universal',
            hyperparameters={
                'component_models': [m.model_type for m in self.models],
                'weights': self.weights,
            },
            is_feature_based=any(m.is_feature_based() for m in self.models),
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return any(m.is_feature_based() for m in self.models)


class StackingForecaster(BaseForecaster):
    """
    Stacking ensemble with meta-learner.

    Uses base model predictions as features for a meta-model.
    """

    def __init__(self, base_models: List[BaseForecaster] = None, meta_model: BaseForecaster = None, **kwargs):
        super().__init__('stacking', **kwargs)
        self.base_models = base_models or []
        self.meta_model = meta_model or LightGBMForecaster(n_estimators=50, max_depth=3)

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'StackingForecaster':
        """Fit base models and meta-learner."""
        # Fit base models
        for model in self.base_models:
            if model.is_feature_based():
                model.fit(y, X)
            else:
                model.fit(y)

        # Generate base model predictions for meta-learner training
        # (In practice, you'd use cross-validation to avoid leakage)
        base_preds = []
        for model in self.base_models:
            if model.is_feature_based():
                pred = model.predict(horizon=1, X=X)
            else:
                # For univariate models, use fitted values
                pred = model.predict(horizon=len(y))

            # Ensure correct shape
            if len(pred) != len(y):
                pred = np.full(len(y), np.mean(pred))
            base_preds.append(pred)

        # Stack base predictions
        meta_features = np.column_stack(base_preds)
        if X is not None:
            meta_features = np.hstack([meta_features, X])

        # Fit meta-learner
        self.meta_model.fit(y, meta_features)

        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate stacking prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        base_preds = []
        for model in self.base_models:
            if model.is_feature_based():
                pred = model.predict(horizon, X)
            else:
                pred = model.predict(horizon)
            base_preds.append(pred)

        # Stack and predict with meta-learner
        meta_features = np.column_stack(base_preds)
        if X is not None:
            meta_features = np.hstack([meta_features, X])

        final_pred = self.meta_model.predict(horizon, meta_features)
        return np.clip(final_pred, 0, None)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='stacking',
            demand_pattern='universal',
            hyperparameters={
                'base_models': [m.model_type for m in self.base_models],
                'meta_model': self.meta_model.model_type,
            },
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


# =============================================================================
# DEEP LEARNING MODELS (Stubs - require enable_deep_models=True)
# =============================================================================

class DeepLearningForecaster(BaseForecaster):
    """
    Base class for deep learning forecasters.
    These require additional dependencies and enable_deep_models=True.
    """

    def __init__(self, model_type: str, **kwargs):
        super().__init__(model_type, **kwargs)
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if deep learning dependencies are available."""
        try:
            import torch
            self._has_torch = True
        except ImportError:
            self._has_torch = False
            logger.warning(f"PyTorch not available for {self.model_type}. Using fallback.")


class TFTForecaster(DeepLearningForecaster):
    """Temporal Fusion Transformer - stub implementation."""

    def __init__(self, **kwargs):
        super().__init__('tft', **kwargs)
        self._fallback = LightGBMForecaster()

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'TFTForecaster':
        logger.warning("TFT requires pytorch-forecasting. Using LightGBM fallback.")
        if X is not None:
            self._fallback.fit(y, X)
        else:
            self._mean = np.mean(y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        if X is not None and self._fallback.is_fitted:
            return self._fallback.predict(horizon, X)
        return np.full(horizon, max(0, getattr(self, '_mean', 0)))

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='tft',
            demand_pattern='universal',
            hyperparameters={},
            is_feature_based=True,
            supports_recursive=True,
            training_samples=getattr(self, '_training_samples', None),
        )

    def is_feature_based(self) -> bool:
        return True


class LSTMForecaster(DeepLearningForecaster):
    """LSTM - stub implementation."""

    def __init__(self, **kwargs):
        super().__init__('lstm', **kwargs)
        self._fallback = ETSForecaster()

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'LSTMForecaster':
        logger.warning("LSTM requires darts/pytorch. Using ETS fallback.")
        self._fallback.fit(y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        return self._fallback.predict(horizon)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='lstm',
            demand_pattern='universal',
            hyperparameters={},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class NBEATSForecaster(DeepLearningForecaster):
    """N-BEATS - stub implementation."""

    def __init__(self, **kwargs):
        super().__init__('nbeats', **kwargs)
        self._fallback = ThetaForecaster()

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'NBEATSForecaster':
        logger.warning("N-BEATS requires darts/pytorch. Using Theta fallback.")
        self._fallback.fit(y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        return self._fallback.predict(horizon)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='nbeats',
            demand_pattern='universal',
            hyperparameters={},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class DeepARForecaster(DeepLearningForecaster):
    """DeepAR - stub implementation."""

    def __init__(self, **kwargs):
        super().__init__('deepar', **kwargs)
        self._fallback = ProphetForecaster()

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'DeepARForecaster':
        logger.warning("DeepAR requires gluonts/pytorch. Using Prophet fallback.")
        self._fallback.fit(y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        return self._fallback.predict(horizon)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='deepar',
            demand_pattern='universal',
            hyperparameters={},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


class WaveNetForecaster(DeepLearningForecaster):
    """WaveNet - stub implementation."""

    def __init__(self, **kwargs):
        super().__init__('wavenet', **kwargs)
        self._fallback = SARIMAForecaster()

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'WaveNetForecaster':
        logger.warning("WaveNet requires gluonts/pytorch. Using SARIMA fallback.")
        self._fallback.fit(y)
        self.is_fitted = True
        self._training_samples = len(y)
        return self

    def predict(self, horizon: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        return self._fallback.predict(horizon)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type='wavenet',
            demand_pattern='universal',
            hyperparameters={},
            is_feature_based=False,
            supports_recursive=False,
            training_samples=getattr(self, '_training_samples', None),
        )


# =============================================================================
# MODEL REGISTRY & FACTORY
# =============================================================================

# Complete model registry mapping model names to classes
MODEL_REGISTRY: Dict[str, type] = {
    # Tree-based (universal)
    'lightgbm': LightGBMForecaster,
    'xgboost': XGBoostForecaster,
    'catboost': CatBoostForecaster,
    'random_forest': RandomForestForecaster,

    # Intermittent specialists
    'croston': CrostonForecaster,
    'sba': SBAForecaster,
    'tsb': TSBForecaster,
    'imapa': IMAPAForecaster,

    # Compound / hurdle
    'zero_inflated': ZeroInflatedForecaster,
    'hurdle_model': HurdleModelForecaster,
    'tweedie': TweedieForecaster,

    # Classical statistical
    'arima': ARIMAForecaster,
    'sarima': SARIMAForecaster,
    'ets': ETSForecaster,
    'theta': ThetaForecaster,
    'tbats': TBATSForecaster,

    # Bayesian / probabilistic
    'prophet': ProphetForecaster,
    'bsts': BSTSForecaster,

    # Ensemble
    'weighted_ensemble': WeightedEnsembleForecaster,
    'stacking': StackingForecaster,

    # Deep learning (stubs)
    'tft': TFTForecaster,
    'lstm': LSTMForecaster,
    'nbeats': NBEATSForecaster,
    'deepar': DeepARForecaster,
    'wavenet': WaveNetForecaster,
}

# Demand pattern to recommended models mapping
# NOTE: For intermittent/lumpy patterns, BOTH specialists AND tree-based baseline (lightgbm) are included
# The system should try BOTH and select based on WAPE comparison
PATTERN_MODEL_RECOMMENDATIONS: Dict[str, List[str]] = {
    'smooth': ['sarima', 'ets', 'prophet', 'lightgbm', 'xgboost', 'tbats', 'theta'],
    'erratic': ['xgboost', 'lightgbm', 'theta', 'bsts', 'catboost', 'random_forest'],
    # Intermittent: Try specialists FIRST but ALWAYS include lightgbm as baseline
    'intermittent': ['croston', 'sba', 'tsb', 'lightgbm', 'zero_inflated', 'tweedie'],
    # Lumpy: Try specialists FIRST but ALWAYS include lightgbm as baseline
    'lumpy': ['tsb', 'hurdle_model', 'zero_inflated', 'lightgbm', 'imapa', 'croston'],
}

# Baseline models that should ALWAYS be tried regardless of demand pattern
# This ensures we have a universal comparison point
BASELINE_MODELS: List[str] = ['lightgbm']

# Specialist models for intermittent/lumpy demand that MUST be tried
INTERMITTENT_SPECIALISTS: List[str] = ['croston', 'sba', 'tsb']
LUMPY_SPECIALISTS: List[str] = ['tsb', 'hurdle_model', 'zero_inflated', 'imapa']

# Discrete demand specialists for low-cardinality targets
DISCRETE_SPECIALISTS: List[str] = ['ordinal_regression', 'discrete_classifier', 'hybrid_discrete']


def get_model(model_type: str, **kwargs) -> BaseForecaster:
    """
    Factory function to create a model instance.

    Parameters
    ----------
    model_type : str
        Name of the model (must be in MODEL_REGISTRY)
    **kwargs
        Model-specific hyperparameters

    Returns
    -------
    BaseForecaster
        Instantiated model ready for fitting

    Examples
    --------
    >>> model = get_model('croston', alpha=0.15)
    >>> model = get_model('lightgbm', n_estimators=300, max_depth=10)
    """
    model_type = model_type.lower()

    if model_type not in MODEL_REGISTRY:
        available = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")

    return MODEL_REGISTRY[model_type](**kwargs)


def get_recommended_models(
    demand_pattern: str,
    allowed_families: Optional[List[str]] = None,
    max_models: int = 5,
    ensure_baseline: bool = True,
    forecast_horizon: Optional[int] = None,
    enable_multi_horizon: bool = False,
    discrete_cardinality: Optional[str] = None,
) -> List[str]:
    """
    Get recommended models for a demand pattern.

    IMPORTANT: For intermittent/lumpy patterns, this function ensures BOTH:
    1. Specialist models (Croston/SBA/TSB) are included
    2. A baseline model (LightGBM) is ALWAYS included for comparison

    This ensures we never miss cases where a tree-based model might outperform
    the specialist models.

    Parameters
    ----------
    demand_pattern : str
        One of: 'smooth', 'erratic', 'intermittent', 'lumpy'
    allowed_families : List[str], optional
        Only return models in this list (from config.yaml model_families)
    max_models : int
        Maximum number of models to return
    ensure_baseline : bool
        If True (default), always include at least one baseline model (lightgbm)
        This ensures specialists are compared against a strong baseline
    forecast_horizon : int, optional
        Number of periods to forecast. Multi-horizon models are included when >= 3.
    enable_multi_horizon : bool
        If True, include multi-horizon direct forecasting models for suitable patterns.
        Multi-horizon models directly optimize for each horizon step, avoiding error
        compounding from recursive one-step-ahead forecasting.
    discrete_cardinality : str, optional
        Discrete demand cardinality category: 'very_low', 'low', 'medium', 'high', or None.
        When set to 'very_low', 'low', or 'medium', discrete specialist models
        (ordinal_regression, discrete_classifier, hybrid_discrete) are injected
        at the top of the recommendation list.

    Returns
    -------
    List[str]
        Ordered list of recommended model names with baseline guaranteed

    Examples
    --------
    >>> get_recommended_models('intermittent', allowed_families=['croston', 'sba', 'lightgbm'])
    ['croston', 'sba', 'lightgbm']  # lightgbm included as baseline

    >>> get_recommended_models('smooth', forecast_horizon=10, enable_multi_horizon=True)
    ['multi_horizon_ensemble', 'multi_horizon_lightgbm', 'sarima', 'ets', ...]

    >>> get_recommended_models('smooth', discrete_cardinality='very_low')
    ['discrete_classifier', 'ordinal_regression', 'sarima', 'ets', ...]
    """
    demand_pattern = demand_pattern.lower()

    if demand_pattern not in PATTERN_MODEL_RECOMMENDATIONS:
        logger.warning(f"Unknown demand pattern: {demand_pattern}. Using universal models.")
        recommendations = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'ets']
    else:
        recommendations = list(PATTERN_MODEL_RECOMMENDATIONS[demand_pattern])  # Make a copy

    # Filter by allowed families if specified
    if allowed_families:
        allowed_lower = [m.lower() for m in allowed_families]
        recommendations = [m for m in recommendations if m.lower() in allowed_lower]

    # -------------------------------------------------------------------------
    # MULTI-HORIZON MODEL INJECTION
    # -------------------------------------------------------------------------
    # Multi-horizon models are beneficial when:
    # 1. enable_multi_horizon is True
    # 2. forecast_horizon >= 3 (longer horizons benefit from direct forecasting)
    # 3. Pattern is smooth or erratic (tree-based models work well)
    # 4. NOT intermittent/lumpy (sparse demand needs specialist models)
    if enable_multi_horizon and forecast_horizon is not None and forecast_horizon >= 3:
        if demand_pattern in ('smooth', 'erratic'):
            multi_horizon_candidates = ['multi_horizon_lightgbm', 'multi_horizon_xgboost']

            # For longer horizons (5+), ensemble of direct models is even more beneficial
            if forecast_horizon >= 5:
                multi_horizon_candidates.insert(0, 'multi_horizon_ensemble')

            # Filter by allowed families
            if allowed_families:
                allowed_lower = [m.lower() for m in allowed_families]
                multi_horizon_candidates = [
                    m for m in multi_horizon_candidates if m.lower() in allowed_lower
                ]

            # Insert multi-horizon models at the TOP of recommendations
            # They should be tried first since they directly optimize for target horizon
            for i, mh_model in enumerate(multi_horizon_candidates):
                if mh_model not in recommendations:
                    recommendations.insert(i, mh_model)

            if multi_horizon_candidates:
                logger.info(
                    f"Multi-horizon models injected for {demand_pattern} pattern "
                    f"(horizon={forecast_horizon}): {multi_horizon_candidates}"
                )

    # -------------------------------------------------------------------------
    # DISCRETE MODEL INJECTION
    # -------------------------------------------------------------------------
    # For keys with low-cardinality discrete demand, inject discrete specialist
    # models at the top of recommendations. These models treat demand as ordinal
    # classes rather than continuous values, which is more appropriate for
    # case-pack ordering patterns (e.g., 0, 200, 400, 800).
    if discrete_cardinality and discrete_cardinality in ('very_low', 'low', 'medium'):
        if discrete_cardinality == 'very_low':
            # 2-5 unique values: pure classification is best
            discrete_candidates = ['discrete_classifier', 'ordinal_regression']
        elif discrete_cardinality == 'low':
            # 6-10 unique values: ordinal regression preserves ordering
            discrete_candidates = ['ordinal_regression', 'discrete_classifier']
        else:  # medium (11-25)
            # Higher cardinality: hybrid approach (regression + snap)
            discrete_candidates = ['hybrid_discrete', 'ordinal_regression']

        # Filter by allowed families
        if allowed_families:
            allowed_lower = [m.lower() for m in allowed_families]
            discrete_candidates = [
                m for m in discrete_candidates if m.lower() in allowed_lower
            ]

        # Insert discrete models at the TOP of recommendations
        for i, d_model in enumerate(discrete_candidates):
            if d_model not in recommendations:
                recommendations.insert(i, d_model)

        if discrete_candidates:
            logger.info(
                f"Discrete models injected for {discrete_cardinality} cardinality "
                f"({demand_pattern} pattern): {discrete_candidates}"
            )

    # CRITICAL: For intermittent/lumpy patterns, ensure baseline model is included
    # This guarantees we compare specialists against a strong tree-based model
    if ensure_baseline and demand_pattern in ['intermittent', 'lumpy']:
        # Check if any baseline model is already in the list
        has_baseline = any(m.lower() in [b.lower() for b in BASELINE_MODELS] for m in recommendations)

        if not has_baseline:
            # Find an allowed baseline model
            if allowed_families:
                allowed_lower = [m.lower() for m in allowed_families]
                available_baselines = [b for b in BASELINE_MODELS if b.lower() in allowed_lower]
            else:
                available_baselines = BASELINE_MODELS

            # Add baseline if available
            if available_baselines:
                baseline = available_baselines[0]
                logger.info(f"Adding baseline model '{baseline}' for {demand_pattern} pattern comparison")
                recommendations.append(baseline)

    return recommendations[:max_models]


def get_models_for_comparison(
    demand_pattern: str,
    allowed_families: Optional[List[str]] = None,
    max_specialists: int = 3,
    max_baselines: int = 1,
) -> Dict[str, List[str]]:
    """
    Get models for fair comparison, ensuring both specialists and baselines.

    This function explicitly separates specialist models from baseline models
    to ensure fair comparison during model selection.

    Parameters
    ----------
    demand_pattern : str
        One of: 'smooth', 'erratic', 'intermittent', 'lumpy'
    allowed_families : List[str], optional
        Only return models in this list
    max_specialists : int
        Maximum number of specialist models
    max_baselines : int
        Maximum number of baseline models

    Returns
    -------
    Dict with keys:
        'specialists': List of specialist model names
        'baselines': List of baseline model names
        'all': Combined list (specialists first, then baselines)

    Examples
    --------
    >>> get_models_for_comparison('intermittent', allowed_families=['croston', 'sba', 'tsb', 'lightgbm'])
    {
        'specialists': ['croston', 'sba', 'tsb'],
        'baselines': ['lightgbm'],
        'all': ['croston', 'sba', 'tsb', 'lightgbm']
    }
    """
    demand_pattern = demand_pattern.lower()

    # Determine which specialists to use
    if demand_pattern == 'intermittent':
        specialists = INTERMITTENT_SPECIALISTS.copy()
    elif demand_pattern == 'lumpy':
        specialists = LUMPY_SPECIALISTS.copy()
    else:
        # For smooth/erratic, treat pattern-specific models as "specialists"
        specialists = PATTERN_MODEL_RECOMMENDATIONS.get(demand_pattern, []).copy()
        # Remove baselines from specialists
        specialists = [m for m in specialists if m not in BASELINE_MODELS]

    baselines = BASELINE_MODELS.copy()

    # Filter by allowed families
    if allowed_families:
        allowed_lower = [m.lower() for m in allowed_families]
        specialists = [m for m in specialists if m.lower() in allowed_lower]
        baselines = [m for m in baselines if m.lower() in allowed_lower]

    # Limit counts
    specialists = specialists[:max_specialists]
    baselines = baselines[:max_baselines]

    # Ensure no duplicates
    all_models = specialists + [b for b in baselines if b not in specialists]

    return {
        'specialists': specialists,
        'baselines': baselines,
        'all': all_models,
    }


def is_feature_based_model(model_type: str) -> bool:
    """Check if a model type requires features (X) or is univariate."""
    model_type = model_type.lower()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create temporary instance to check
    temp_model = MODEL_REGISTRY[model_type]()
    return temp_model.is_feature_based()


def get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available models.

    Returns
    -------
    Dict with model names as keys and info dicts as values
    """
    info = {}
    for name, cls in MODEL_REGISTRY.items():
        try:
            temp = cls()
            info[name] = {
                'class': cls.__name__,
                'is_feature_based': temp.is_feature_based(),
                'supports_recursive': temp.supports_recursive(),
                'docstring': cls.__doc__.split('\n')[0] if cls.__doc__ else '',
            }
        except Exception:
            info[name] = {
                'class': cls.__name__,
                'is_feature_based': 'unknown',
                'supports_recursive': 'unknown',
            }
    return info


# =============================================================================
# HELPER FUNCTIONS FOR TRAINING CREW
# =============================================================================

def train_and_select_best_model(
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_train: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    demand_pattern: str,
    allowed_families: List[str],
    max_candidates: int = 5,
) -> Tuple[BaseForecaster, float, str]:
    """
    Train multiple models and select the best one based on validation WAPE.

    Parameters
    ----------
    y_train : np.ndarray
        Training target values
    y_val : np.ndarray
        Validation target values
    X_train : np.ndarray, optional
        Training features
    X_val : np.ndarray, optional
        Validation features
    demand_pattern : str
        Demand pattern for model recommendations
    allowed_families : List[str]
        Allowed model families from config
    max_candidates : int
        Maximum number of models to try

    Returns
    -------
    Tuple[BaseForecaster, float, str]
        (best_model, best_wape, model_type)
    """
    # Get recommended models
    candidates = get_recommended_models(demand_pattern, allowed_families, max_candidates)

    if not candidates:
        logger.warning(f"No models available for {demand_pattern}. Using LightGBM.")
        candidates = ['lightgbm']

    best_model = None
    best_wape = float('inf')
    best_type = None

    for model_type in candidates:
        try:
            model = get_model(model_type)

            # Fit model
            if model.is_feature_based():
                if X_train is None:
                    logger.warning(f"Skipping {model_type}: requires features but X_train is None")
                    continue
                model.fit(y_train, X_train)
                preds = model.predict(len(y_val), X_val)
            else:
                model.fit(y_train)
                preds = model.predict(len(y_val))

            # Compute WAPE
            wape = np.sum(np.abs(y_val - preds)) / (np.sum(np.abs(y_val)) + 1e-10)

            logger.info(f"  {model_type}: WAPE = {wape:.4f}")

            if wape < best_wape:
                best_wape = wape
                best_model = model
                best_type = model_type

        except Exception as e:
            logger.warning(f"Failed to train {model_type}: {e}")
            continue

    if best_model is None:
        # Fallback to simple average
        logger.warning("All models failed. Using simple mean fallback.")
        from sklearn.dummy import DummyRegressor
        fallback = DummyRegressor(strategy='mean')
        fallback.fit(X_train if X_train is not None else np.zeros((len(y_train), 1)), y_train)
        return fallback, 1.0, 'fallback'

    return best_model, best_wape, best_type


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    'BaseForecaster',
    'ModelMetadata',

    # Tree-based
    'LightGBMForecaster',
    'XGBoostForecaster',
    'CatBoostForecaster',
    'RandomForestForecaster',

    # Intermittent specialists
    'CrostonForecaster',
    'SBAForecaster',
    'TSBForecaster',
    'IMAPAForecaster',

    # Compound / hurdle
    'ZeroInflatedForecaster',
    'HurdleModelForecaster',
    'TweedieForecaster',

    # Classical statistical
    'ARIMAForecaster',
    'SARIMAForecaster',
    'ETSForecaster',
    'ThetaForecaster',
    'TBATSForecaster',

    # Bayesian / probabilistic
    'ProphetForecaster',
    'BSTSForecaster',

    # Ensemble
    'WeightedEnsembleForecaster',
    'StackingForecaster',

    # Deep learning
    'TFTForecaster',
    'LSTMForecaster',
    'NBEATSForecaster',
    'DeepARForecaster',
    'WaveNetForecaster',

    # Registry & factory
    'MODEL_REGISTRY',
    'PATTERN_MODEL_RECOMMENDATIONS',
    'BASELINE_MODELS',
    'INTERMITTENT_SPECIALISTS',
    'LUMPY_SPECIALISTS',
    'get_model',
    'get_recommended_models',
    'get_models_for_comparison',
    'is_feature_based_model',
    'get_model_info',
    'train_and_select_best_model',
]
