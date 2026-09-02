# utils/target_transforms.py
"""
Target Transformation Utility for Demand Forecasting
=====================================================

This module provides stationarity-aware transformations for the target variable.
Transformations are learned from EDA and applied during feature engineering and training.

Key Transformations:
    1. LOG TRANSFORM: For high-variance data (CV > 1.5)
       - Stabilizes variance
       - Makes multiplicative effects additive
       - Use log1p to handle zeros

    2. DIFFERENCING: For non-stationary data
       - First differencing: y_diff = y_t - y_{t-1}
       - Seasonal differencing: y_diff = y_t - y_{t-s}
       - Removes trends and seasonality

    3. BOX-COX: For general variance stabilization
       - Optimal power transform
       - Requires positive values

    4. STANDARDIZATION: For neural network models
       - Z-score normalization
       - Per-key or global

The transformation pipeline is:
    1. EDA Crew detects: stationarity type, CV, seasonality
    2. EDA creates: transformation_recommendations in eda_to_feature_context.json
    3. Feature Crew applies: transforms to target based on recommendations
    4. Training Crew trains: on transformed target
    5. Prediction: inverse transform predictions back to original scale

Usage:
    from utils.target_transforms import (
        get_recommended_transforms,
        TransformPipeline,
        apply_transforms,
        inverse_transforms,
    )

    # Get transforms based on EDA recommendations
    transforms = get_recommended_transforms(
        stationarity_type='non_stationary',
        cv=2.5,
        zero_fraction=0.3,
        seasonality_period=52,
    )

    # Apply transforms
    pipeline = TransformPipeline(transforms)
    y_transformed = pipeline.fit_transform(y_train)

    # After training and prediction
    y_pred_original = pipeline.inverse_transform(y_pred_transformed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
import warnings

logger = logging.getLogger(__name__)

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# STATIONARITY TYPES AND RECOMMENDATIONS
# =============================================================================

@dataclass
class StationarityInfo:
    """Information about a series' stationarity."""
    stationarity_type: str  # stationary, non_stationary, trend_stationary, difference_stationary
    adf_pvalue: float = 0.0
    kpss_pvalue: float = 0.0
    is_stationary: bool = False
    requires_differencing: bool = False
    differencing_order: int = 0
    seasonal_period: Optional[int] = None
    requires_seasonal_diff: bool = False


# Mapping from stationarity type to recommended transforms
STATIONARITY_TRANSFORMS: Dict[str, List[str]] = {
    'stationary': [],  # No transform needed
    'non_stationary': ['first_difference'],  # Needs differencing
    'trend_stationary': ['detrend'],  # Remove trend
    'difference_stationary': ['first_difference'],  # Needs differencing
    'seasonal_non_stationary': ['seasonal_difference'],  # Seasonal differencing
}

# CV thresholds for log transform
CV_THRESHOLD_LOG = 1.5  # If CV > this, apply log transform
CV_THRESHOLD_BOXCOX = 2.0  # If CV > this, consider Box-Cox

# Zero fraction thresholds
ZERO_FRACTION_HIGH = 0.3  # If >30% zeros, handle specially


# =============================================================================
# BASE TRANSFORM CLASS
# =============================================================================

class BaseTransform(ABC):
    """Abstract base class for all transformations."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self._params: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, y: np.ndarray) -> 'BaseTransform':
        """Fit the transform parameters."""
        pass

    @abstractmethod
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply the transformation."""
        pass

    @abstractmethod
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Reverse the transformation."""
        pass

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y)
        return self.transform(y)

    def get_params(self) -> Dict[str, Any]:
        """Get the fitted parameters."""
        return self._params.copy()

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters (for loading saved transforms)."""
        self._params = params
        self.is_fitted = True


# =============================================================================
# SPECIFIC TRANSFORMS
# =============================================================================

class LogTransform(BaseTransform):
    """
    Log transformation for variance stabilization.

    Uses log1p(y) to handle zeros: log(1 + y)
    Inverse: expm1(y) = exp(y) - 1
    """

    def __init__(self, offset: float = 1.0):
        super().__init__('log')
        self.offset = offset  # Added to y before log (default 1 for log1p)

    def fit(self, y: np.ndarray) -> 'LogTransform':
        """Log transform doesn't need fitting."""
        self.is_fitted = True
        self._params = {'offset': self.offset}
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply log1p transform."""
        y = np.asarray(y)
        # Clip to non-negative
        y_clipped = np.maximum(y, 0)
        return np.log1p(y_clipped)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Apply expm1 (inverse of log1p)."""
        y = np.asarray(y)
        result = np.expm1(y)
        # Clip to non-negative
        return np.maximum(result, 0)


class FirstDifferenceTransform(BaseTransform):
    """
    First-order differencing: y_diff = y_t - y_{t-1}

    Removes trend and makes non-stationary series stationary.
    """

    def __init__(self):
        super().__init__('first_difference')
        self._first_value: Optional[float] = None

    def fit(self, y: np.ndarray) -> 'FirstDifferenceTransform':
        """Store first value for inverse transform."""
        y = np.asarray(y)
        self._first_value = float(y[0]) if len(y) > 0 else 0.0
        self.is_fitted = True
        self._params = {'first_value': self._first_value}
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply first differencing."""
        y = np.asarray(y)
        if len(y) <= 1:
            return np.array([0.0])
        return np.diff(y)

    def inverse_transform(self, y_diff: np.ndarray) -> np.ndarray:
        """Cumulative sum to reverse differencing."""
        y_diff = np.asarray(y_diff)
        if self._first_value is None:
            self._first_value = 0.0
        return np.concatenate([[self._first_value], self._first_value + np.cumsum(y_diff)])


class SeasonalDifferenceTransform(BaseTransform):
    """
    Seasonal differencing: y_diff = y_t - y_{t-s}

    Removes seasonal patterns.
    """

    def __init__(self, period: int = 52):
        super().__init__('seasonal_difference')
        self.period = period
        self._initial_values: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> 'SeasonalDifferenceTransform':
        """Store initial season values for inverse transform."""
        y = np.asarray(y)
        if len(y) >= self.period:
            self._initial_values = y[:self.period].copy()
        else:
            self._initial_values = y.copy()
        self.is_fitted = True
        self._params = {
            'period': self.period,
            'initial_values': self._initial_values.tolist(),
        }
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply seasonal differencing."""
        y = np.asarray(y)
        if len(y) <= self.period:
            return np.zeros(len(y))
        return y[self.period:] - y[:-self.period]

    def inverse_transform(self, y_diff: np.ndarray) -> np.ndarray:
        """Reverse seasonal differencing."""
        y_diff = np.asarray(y_diff)
        if self._initial_values is None:
            self._initial_values = np.zeros(self.period)

        n_total = len(y_diff) + self.period
        result = np.zeros(n_total)
        result[:self.period] = self._initial_values[:self.period]

        for i in range(len(y_diff)):
            result[self.period + i] = y_diff[i] + result[i]

        return result


class DetrendTransform(BaseTransform):
    """
    Linear detrending: removes linear trend from series.

    y_detrended = y - (a + b*t)
    """

    def __init__(self):
        super().__init__('detrend')
        self._intercept: float = 0.0
        self._slope: float = 0.0

    def fit(self, y: np.ndarray) -> 'DetrendTransform':
        """Fit linear trend."""
        y = np.asarray(y)
        t = np.arange(len(y))

        if len(y) > 1:
            # Simple linear regression
            t_mean = t.mean()
            y_mean = y.mean()
            self._slope = np.sum((t - t_mean) * (y - y_mean)) / (np.sum((t - t_mean)**2) + 1e-10)
            self._intercept = y_mean - self._slope * t_mean

        self.is_fitted = True
        self._params = {
            'intercept': self._intercept,
            'slope': self._slope,
        }
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Remove trend."""
        y = np.asarray(y)
        t = np.arange(len(y))
        trend = self._intercept + self._slope * t
        return y - trend

    def inverse_transform(self, y_detrended: np.ndarray) -> np.ndarray:
        """Add trend back."""
        y_detrended = np.asarray(y_detrended)
        t = np.arange(len(y_detrended))
        trend = self._intercept + self._slope * t
        return y_detrended + trend


class StandardizeTransform(BaseTransform):
    """
    Z-score standardization: (y - mean) / std

    Useful for neural network models.
    """

    def __init__(self):
        super().__init__('standardize')
        self._mean: float = 0.0
        self._std: float = 1.0

    def fit(self, y: np.ndarray) -> 'StandardizeTransform':
        """Compute mean and std."""
        y = np.asarray(y)
        self._mean = float(np.mean(y))
        self._std = float(np.std(y))
        if self._std < 1e-10:
            self._std = 1.0

        self.is_fitted = True
        self._params = {
            'mean': self._mean,
            'std': self._std,
        }
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Standardize."""
        y = np.asarray(y)
        return (y - self._mean) / self._std

    def inverse_transform(self, y_std: np.ndarray) -> np.ndarray:
        """De-standardize."""
        y_std = np.asarray(y_std)
        return y_std * self._std + self._mean


class BoxCoxTransform(BaseTransform):
    """
    Box-Cox transformation for variance stabilization.

    Requires all values to be positive.
    """

    def __init__(self, lmbda: Optional[float] = None):
        super().__init__('boxcox')
        self._lmbda = lmbda
        self._offset: float = 0.0

    def fit(self, y: np.ndarray) -> 'BoxCoxTransform':
        """Fit optimal lambda."""
        y = np.asarray(y)

        # Ensure positive values
        self._offset = max(0, -np.min(y) + 1)
        y_pos = y + self._offset

        if self._lmbda is None:
            try:
                from scipy import stats
                _, self._lmbda = stats.boxcox(y_pos + 1e-10)
            except Exception:
                self._lmbda = 0.0  # Log transform

        self.is_fitted = True
        self._params = {
            'lmbda': self._lmbda,
            'offset': self._offset,
        }
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply Box-Cox."""
        y = np.asarray(y)
        y_pos = y + self._offset

        if abs(self._lmbda) < 1e-10:
            return np.log(y_pos + 1e-10)
        else:
            return (np.power(y_pos + 1e-10, self._lmbda) - 1) / self._lmbda

    def inverse_transform(self, y_bc: np.ndarray) -> np.ndarray:
        """Inverse Box-Cox."""
        y_bc = np.asarray(y_bc)

        if abs(self._lmbda) < 1e-10:
            return np.exp(y_bc) - self._offset
        else:
            result = np.power(y_bc * self._lmbda + 1, 1 / self._lmbda)
            return result - self._offset


# =============================================================================
# TRANSFORM FACTORY
# =============================================================================

TRANSFORM_REGISTRY: Dict[str, type] = {
    'log': LogTransform,
    'log1p': LogTransform,
    'first_difference': FirstDifferenceTransform,
    'difference': FirstDifferenceTransform,
    'seasonal_difference': SeasonalDifferenceTransform,
    'detrend': DetrendTransform,
    'standardize': StandardizeTransform,
    'boxcox': BoxCoxTransform,
}


def get_transform(name: str, **kwargs) -> BaseTransform:
    """Factory function to create a transform."""
    name_lower = name.lower()
    if name_lower not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORM_REGISTRY.keys())}")
    return TRANSFORM_REGISTRY[name_lower](**kwargs)


# =============================================================================
# TRANSFORM PIPELINE
# =============================================================================

class TransformPipeline:
    """
    Pipeline to apply multiple transforms in sequence.

    Inverse transforms are applied in reverse order.
    """

    def __init__(self, transforms: Optional[List[Union[str, BaseTransform]]] = None):
        """
        Initialize pipeline.

        Parameters
        ----------
        transforms : List of transform names or instances
            E.g., ['log', 'first_difference'] or [LogTransform(), FirstDifferenceTransform()]
        """
        self.transforms: List[BaseTransform] = []
        if transforms:
            for t in transforms:
                if isinstance(t, str):
                    self.transforms.append(get_transform(t))
                else:
                    self.transforms.append(t)

        self.is_fitted = False

    def add_transform(self, transform: Union[str, BaseTransform]) -> 'TransformPipeline':
        """Add a transform to the pipeline."""
        if isinstance(transform, str):
            self.transforms.append(get_transform(transform))
        else:
            self.transforms.append(transform)
        self.is_fitted = False
        return self

    def fit(self, y: np.ndarray) -> 'TransformPipeline':
        """Fit all transforms in sequence."""
        y_current = np.asarray(y)

        for transform in self.transforms:
            transform.fit(y_current)
            y_current = transform.transform(y_current)

        self.is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply all transforms."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        y_current = np.asarray(y)
        for transform in self.transforms:
            y_current = transform.transform(y_current)

        return y_current

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse transforms in reverse order."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        y_current = np.asarray(y)

        # Apply inverse transforms in REVERSE order
        for transform in reversed(self.transforms):
            y_current = transform.inverse_transform(y_current)

        return y_current

    def get_config(self) -> Dict[str, Any]:
        """Get serializable config for saving."""
        return {
            'transforms': [
                {
                    'name': t.name,
                    'params': t.get_params(),
                }
                for t in self.transforms
            ]
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TransformPipeline':
        """Load pipeline from saved config."""
        pipeline = cls()
        for t_config in config.get('transforms', []):
            transform = get_transform(t_config['name'])
            transform.set_params(t_config['params'])
            pipeline.transforms.append(transform)
        pipeline.is_fitted = True
        return pipeline


# =============================================================================
# RECOMMENDATION FUNCTIONS
# =============================================================================

def get_recommended_transforms(
    stationarity_type: str = 'unknown',
    cv: float = 0.0,
    zero_fraction: float = 0.0,
    seasonality_period: Optional[int] = None,
    for_model_type: str = 'tree',
) -> List[str]:
    """
    Get recommended transforms based on data characteristics.

    This function encapsulates the business logic for choosing transforms
    based on EDA findings.

    Parameters
    ----------
    stationarity_type : str
        From EDA: 'stationary', 'non_stationary', 'trend_stationary', 'difference_stationary'
    cv : float
        Coefficient of variation from EDA
    zero_fraction : float
        Fraction of zeros in the series
    seasonality_period : int, optional
        Detected seasonality period (e.g., 52 for weekly data)
    for_model_type : str
        Target model type: 'tree', 'statistical', 'neural'

    Returns
    -------
    List[str]
        Ordered list of transform names to apply
    """
    transforms = []

    # 1. Log transform for high variance
    if cv > CV_THRESHOLD_LOG and zero_fraction < ZERO_FRACTION_HIGH:
        transforms.append('log')
        logger.info(f"Recommending log transform (CV={cv:.2f} > {CV_THRESHOLD_LOG})")

    # 2. Stationarity-based transforms
    stationarity_type = stationarity_type.lower().replace('-', '_').replace(' ', '_')

    if stationarity_type in ['non_stationary', 'difference_stationary']:
        transforms.append('first_difference')
        logger.info(f"Recommending first differencing (stationarity={stationarity_type})")

    elif stationarity_type == 'trend_stationary':
        transforms.append('detrend')
        logger.info("Recommending detrending (trend_stationary)")

    # 3. Seasonal differencing if seasonal and not stationary
    if seasonality_period and stationarity_type != 'stationary':
        # Only for statistical models; trees can learn seasonality
        if for_model_type == 'statistical':
            transforms.append('seasonal_difference')
            logger.info(f"Recommending seasonal differencing (period={seasonality_period})")

    # 4. Standardization for neural networks
    if for_model_type == 'neural':
        transforms.append('standardize')
        logger.info("Recommending standardization (neural network)")

    return transforms


def create_transform_pipeline_from_eda(
    eda_context: Dict[str, Any],
    demand_pattern: str = 'smooth',
    model_type: str = 'tree',
) -> TransformPipeline:
    """
    Create a transform pipeline from EDA context file.

    Parameters
    ----------
    eda_context : dict
        Content of eda_to_feature_context.json or consolidated_feature_context.json
    demand_pattern : str
        The demand pattern for this segment/key
    model_type : str
        Target model type

    Returns
    -------
    TransformPipeline
        Configured pipeline ready to fit
    """
    # Extract transformation recommendations
    transform_recs = eda_context.get('transformation_recommendations', {})

    # Extract stationarity info
    stationarity_type = eda_context.get('stationarity_summary', {}).get('dominant_type', 'unknown')
    non_stationary_pct = eda_context.get('stationarity_summary', {}).get('non_stationary_pct', 0.0)

    # Extract CV info
    cv_stats = eda_context.get('cv_statistics', {})
    avg_cv = cv_stats.get('mean', 0.0) if isinstance(cv_stats, dict) else 0.0

    # Get explicit recommendations if available
    log_transform = transform_recs.get('log_transform', False)
    differencing = transform_recs.get('differencing', False)

    # Build transforms list
    transforms = []

    # Log transform
    if log_transform or avg_cv > CV_THRESHOLD_LOG:
        transforms.append('log')

    # Differencing
    if differencing or non_stationary_pct > 0.5 or stationarity_type in ['non_stationary', 'difference_stationary']:
        transforms.append('first_difference')
    elif stationarity_type == 'trend_stationary':
        transforms.append('detrend')

    # Standardization for neural nets
    if model_type == 'neural':
        transforms.append('standardize')

    logger.info(f"Created transform pipeline: {transforms} for pattern={demand_pattern}, model={model_type}")

    return TransformPipeline(transforms)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_transforms(
    y: np.ndarray,
    transforms: List[str],
    fit: bool = True,
) -> Tuple[np.ndarray, TransformPipeline]:
    """
    Convenience function to apply transforms.

    Parameters
    ----------
    y : array-like
        Target values
    transforms : list of str
        Transform names to apply
    fit : bool
        Whether to fit the pipeline (True for training, False for prediction)

    Returns
    -------
    y_transformed : np.ndarray
        Transformed values
    pipeline : TransformPipeline
        Fitted pipeline (for later inverse transform)
    """
    pipeline = TransformPipeline(transforms)

    if fit:
        y_transformed = pipeline.fit_transform(y)
    else:
        y_transformed = pipeline.transform(y)

    return y_transformed, pipeline


def inverse_transforms(
    y_transformed: np.ndarray,
    pipeline: TransformPipeline,
) -> np.ndarray:
    """
    Convenience function to inverse transform.

    Parameters
    ----------
    y_transformed : array-like
        Transformed values
    pipeline : TransformPipeline
        Fitted pipeline from apply_transforms

    Returns
    -------
    y_original : np.ndarray
        Values in original scale
    """
    return pipeline.inverse_transform(y_transformed)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    'BaseTransform',
    'TransformPipeline',

    # Specific transforms
    'LogTransform',
    'FirstDifferenceTransform',
    'SeasonalDifferenceTransform',
    'DetrendTransform',
    'StandardizeTransform',
    'BoxCoxTransform',

    # Factory and registry
    'TRANSFORM_REGISTRY',
    'get_transform',

    # Recommendation functions
    'get_recommended_transforms',
    'create_transform_pipeline_from_eda',

    # Convenience functions
    'apply_transforms',
    'inverse_transforms',

    # Configuration
    'StationarityInfo',
    'STATIONARITY_TRANSFORMS',
    'CV_THRESHOLD_LOG',
    'CV_THRESHOLD_BOXCOX',
    'ZERO_FRACTION_HIGH',
]
