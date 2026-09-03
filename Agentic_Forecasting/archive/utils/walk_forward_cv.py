# utils/walk_forward_cv.py
"""
Walk-Forward Cross-Validation for Time Series Demand Forecasting.

This module implements temporally-aware cross-validation that:
1. Respects time ordering (no future data leakage)
2. Provides realistic performance estimates
3. Detects concept drift over time
4. Enables robust model selection and hyperparameter tuning
5. Optimizes ensemble weights across multiple time periods

Key Concepts:
- Expanding Window: Train on [0:t], validate on [t:t+h]
- Rolling Window: Train on [t-w:t], validate on [t:t+h] (fixed window size)
- Walk-Forward: Iterate origin t forward through time

Usage:
    from utils.walk_forward_cv import WalkForwardCV, walk_forward_model_selection

    cv = WalkForwardCV(n_splits=5, forecast_horizon=8, min_train_periods=52)
    results = cv.evaluate_models(X, y, candidate_models, segment_id='seg_0')
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Represents a single walk-forward CV fold."""
    fold_idx: int
    train_start: int  # Period index (inclusive)
    train_end: int    # Period index (exclusive)
    val_start: int    # Period index (inclusive)
    val_end: int      # Period index (exclusive)
    train_periods: int
    val_periods: int


@dataclass
class FoldResult:
    """Results from evaluating a model on a single fold."""
    fold_idx: int
    model_name: str
    wape: float
    mape: float
    rmse: float
    mae: float
    bias: float  # Mean error (positive = over-forecast)
    n_train_samples: int
    n_val_samples: int
    train_time_seconds: float
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None


@dataclass
class ModelCVResult:
    """Aggregated CV results for a single model."""
    model_name: str
    model_config: Dict[str, Any]
    fold_results: List[FoldResult]

    # Aggregated metrics
    avg_wape: float = 0.0
    std_wape: float = 0.0
    avg_mape: float = 0.0
    avg_rmse: float = 0.0
    avg_mae: float = 0.0
    avg_bias: float = 0.0

    # Stability metrics
    wape_trend: float = 0.0  # Positive = degrading over time
    concept_drift_detected: bool = False

    # Per-fold WAPEs for analysis
    fold_wapes: List[float] = field(default_factory=list)

    # Multi-horizon metrics (Sprint 2 B1)
    horizon_weighted_wape: float = 0.0  # Weighted WAPE across horizons
    per_horizon_wape: Dict[int, float] = field(default_factory=dict)  # lag → WAPE

    def __post_init__(self):
        if self.fold_results:
            self._compute_aggregates()

    def _compute_aggregates(self):
        """Compute aggregate statistics from fold results."""
        wapes = [r.wape for r in self.fold_results]
        self.fold_wapes = wapes
        self.avg_wape = np.mean(wapes)
        self.std_wape = np.std(wapes)
        self.avg_mape = np.mean([r.mape for r in self.fold_results])
        self.avg_rmse = np.mean([r.rmse for r in self.fold_results])
        self.avg_mae = np.mean([r.mae for r in self.fold_results])
        self.avg_bias = np.mean([r.bias for r in self.fold_results])

        # Compute WAPE trend (linear regression slope)
        if len(wapes) >= 3:
            x = np.arange(len(wapes))
            self.wape_trend = np.polyfit(x, wapes, 1)[0]
            # Concept drift if WAPE increases by >2% per fold on average
            self.concept_drift_detected = self.wape_trend > 0.02


@dataclass
class WalkForwardCVResult:
    """Complete results from walk-forward cross-validation."""
    segment_id: str
    n_folds: int
    forecast_horizon: int
    min_train_periods: int

    # Results per model
    model_results: Dict[str, ModelCVResult] = field(default_factory=dict)

    # Best model selection
    best_model_name: str = ""
    best_model_config: Dict[str, Any] = field(default_factory=dict)
    best_avg_wape: float = float('inf')
    best_std_wape: float = float('inf')

    # Ensemble weights (if multiple models)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    ensemble_avg_wape: float = float('inf')

    # Concept drift summary
    any_drift_detected: bool = False
    drift_models: List[str] = field(default_factory=list)

    # Honest performance estimate
    expected_wape_range: Tuple[float, float] = (0.0, 1.0)

    def select_best_model(self):
        """Select best model based on horizon-weighted WAPE (if available), with stability as tiebreaker."""
        if not self.model_results:
            return

        # Prefer horizon-weighted WAPE (puts more weight on lag 4/5)
        # Fall back to avg_wape if horizon-weighted not computed
        def _selection_score(item):
            name, result = item
            hw = result.horizon_weighted_wape
            if hw > 0:
                return (hw, result.std_wape)
            return (result.avg_wape, result.std_wape)

        sorted_models = sorted(self.model_results.items(), key=_selection_score)

        best_name, best_result = sorted_models[0]
        self.best_model_name = best_name
        self.best_model_config = best_result.model_config
        self.best_avg_wape = best_result.avg_wape
        self.best_std_wape = best_result.std_wape

        # Set expected WAPE range (mean ± 1.5 std)
        lower = max(0, self.best_avg_wape - 1.5 * self.best_std_wape)
        upper = min(1, self.best_avg_wape + 1.5 * self.best_std_wape)
        self.expected_wape_range = (lower, upper)

        # Check for concept drift
        self.drift_models = [
            name for name, result in self.model_results.items()
            if result.concept_drift_detected
        ]
        self.any_drift_detected = len(self.drift_models) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'segment_id': self.segment_id,
            'n_folds': self.n_folds,
            'forecast_horizon': self.forecast_horizon,
            'min_train_periods': self.min_train_periods,
            'best_model': {
                'name': self.best_model_name,
                'config': self.best_model_config,
                'avg_wape': self.best_avg_wape,
                'std_wape': self.best_std_wape,
                'expected_wape_range': list(self.expected_wape_range),
            },
            'ensemble': {
                'weights': self.ensemble_weights,
                'avg_wape': self.ensemble_avg_wape,
            },
            'concept_drift': {
                'detected': self.any_drift_detected,
                'affected_models': self.drift_models,
            },
            'model_summary': {
                name: {
                    'avg_wape': result.avg_wape,
                    'std_wape': result.std_wape,
                    'wape_trend': result.wape_trend,
                    'fold_wapes': result.fold_wapes,
                    'concept_drift': result.concept_drift_detected,
                }
                for name, result in self.model_results.items()
            }
        }


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation for time series.

    This implements expanding window CV where:
    - Each fold trains on all data up to origin_t
    - Validates on [origin_t : origin_t + forecast_horizon]
    - Origin moves forward through time

    Parameters
    ----------
    n_splits : int
        Number of CV folds (default: 5)
    forecast_horizon : int
        Number of periods to forecast ahead (default: 8)
    min_train_periods : int
        Minimum training periods required (default: 52, ~1 year)
    gap : int
        Gap between train end and validation start (default: 0)
        Use gap > 0 if there's a delay in data availability
    strategy : str
        'expanding' (default) or 'rolling'
        - expanding: train on all history up to origin
        - rolling: train on fixed window before origin
    rolling_window : int
        Window size for rolling strategy (default: 104, ~2 years)
    """

    def __init__(
        self,
        n_splits: int = 5,
        forecast_horizon: int = 8,
        min_train_periods: int = 52,
        gap: int = 0,
        strategy: str = 'expanding',
        rolling_window: int = 104,
    ):
        self.n_splits = n_splits
        self.forecast_horizon = forecast_horizon
        self.min_train_periods = min_train_periods
        self.gap = gap
        self.strategy = strategy
        self.rolling_window = rolling_window

        if strategy not in ('expanding', 'rolling'):
            raise ValueError(f"strategy must be 'expanding' or 'rolling', got {strategy}")

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        period_col: str = 'period_idx',
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation indices for each fold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with period column
        period_col : str
            Column name containing period indices (0, 1, 2, ...)

        Yields
        ------
        train_idx, val_idx : tuple of np.ndarray
            Boolean masks for train and validation sets
        """
        if period_col not in X.columns:
            raise ValueError(f"Column '{period_col}' not found in X. Available: {list(X.columns)}")

        periods = X[period_col].values
        unique_periods = np.sort(np.unique(periods))
        n_periods = len(unique_periods)

        # Calculate available periods for CV
        # Need: min_train + gap + forecast_horizon + (n_splits - 1) * step
        required_periods = self.min_train_periods + self.gap + self.forecast_horizon
        if n_periods < required_periods:
            logger.warning(
                f"Not enough periods for walk-forward CV: {n_periods} < {required_periods}. "
                f"Reducing n_splits or min_train_periods may help."
            )
            # Fallback to single split
            yield self._single_split(X, periods, unique_periods)
            return

        # Calculate step size between fold origins
        available_for_cv = n_periods - self.min_train_periods - self.gap - self.forecast_horizon
        step_size = max(1, available_for_cv // self.n_splits)

        for fold_idx in range(self.n_splits):
            # Origin is the last training period
            origin_period_idx = self.min_train_periods + (fold_idx * step_size)
            origin_period = unique_periods[origin_period_idx]

            # Validation period range
            val_start_period_idx = origin_period_idx + self.gap
            val_end_period_idx = min(val_start_period_idx + self.forecast_horizon, n_periods)
            val_start_period = unique_periods[val_start_period_idx]
            val_end_period = unique_periods[val_end_period_idx - 1] if val_end_period_idx > val_start_period_idx else val_start_period

            # Training period range
            if self.strategy == 'expanding':
                train_start_period = unique_periods[0]
            else:  # rolling
                train_start_idx = max(0, origin_period_idx - self.rolling_window)
                train_start_period = unique_periods[train_start_idx]
            train_end_period = origin_period

            # Create boolean masks
            train_mask = (periods >= train_start_period) & (periods < train_end_period)
            val_mask = (periods >= val_start_period) & (periods <= val_end_period)

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                logger.warning(f"Fold {fold_idx}: Empty train or val set, skipping")
                continue

            yield train_mask, val_mask

    def _single_split(
        self,
        X: pd.DataFrame,
        periods: np.ndarray,
        unique_periods: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to single train/val split when not enough data for CV."""
        n_periods = len(unique_periods)
        split_idx = max(1, n_periods - self.forecast_horizon)
        split_period = unique_periods[split_idx]

        train_mask = periods < split_period
        val_mask = periods >= split_period

        return train_mask, val_mask

    def get_fold_info(
        self,
        X: pd.DataFrame,
        period_col: str = 'period_idx',
    ) -> List[WalkForwardFold]:
        """Get detailed information about each fold."""
        periods = X[period_col].values
        unique_periods = np.sort(np.unique(periods))
        n_periods = len(unique_periods)

        folds = []
        for fold_idx, (train_mask, val_mask) in enumerate(self.split(X, period_col)):
            train_periods = periods[train_mask]
            val_periods = periods[val_mask]

            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=int(train_periods.min()),
                train_end=int(train_periods.max()) + 1,
                val_start=int(val_periods.min()),
                val_end=int(val_periods.max()) + 1,
                train_periods=len(np.unique(train_periods)),
                val_periods=len(np.unique(val_periods)),
            )
            folds.append(fold)

        return folds

    def evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_fn: Callable,
        model_name: str,
        model_config: Dict[str, Any],
        period_col: str = 'period_idx',
        feature_cols: Optional[List[str]] = None,
        store_predictions: bool = False,
        horizon_weights: Optional[Dict[int, float]] = None,
    ) -> ModelCVResult:
        """
        Evaluate a single model across all CV folds.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model_fn : Callable
            Function that creates and fits model: model_fn(X_train, y_train, config) -> fitted model
        model_name : str
            Name for this model configuration
        model_config : dict
            Configuration passed to model_fn
        period_col : str
            Column name for period indices
        feature_cols : list, optional
            Feature columns to use. If None, use all except period_col
        store_predictions : bool
            Whether to store predictions in results (memory intensive)

        Returns
        -------
        ModelCVResult
            Cross-validation results for this model
        """
        import time

        if feature_cols is None:
            feature_cols = [c for c in X.columns if c != period_col]

        fold_results = []
        fold_lag_wapes = []  # Store per-fold per-lag WAPEs for horizon weighting

        for fold_idx, (train_mask, val_mask) in enumerate(self.split(X, period_col)):
            X_train = X.loc[train_mask, feature_cols]
            y_train = y[train_mask]
            X_val = X.loc[val_mask, feature_cols]
            y_val = y[val_mask]

            # Train model
            start_time = time.time()
            try:
                model = model_fn(X_train, y_train, model_config)
                train_time = time.time() - start_time

                # Predict
                predictions = model.predict(X_val)
                predictions = np.maximum(predictions, 0)  # Ensure non-negative

            except Exception as e:
                logger.warning(f"Model {model_name} fold {fold_idx} failed: {e}")
                fold_lag_wapes.append({})
                continue

            # Calculate metrics
            actuals = y_val.values
            metrics = self._compute_metrics(actuals, predictions)

            # Multi-horizon: compute per-lag WAPE within THIS fold
            this_fold_lag_wape = {}
            if period_col in X.columns and horizon_weights:
                val_periods = sorted(X.loc[val_mask, period_col].unique())
                for lag_idx, period in enumerate(val_periods):
                    period_mask = X.loc[val_mask, period_col] == period
                    a = actuals[period_mask.values]
                    p = predictions[period_mask.values]
                    total = np.sum(np.abs(a))
                    if total > 1e-10:
                        this_fold_lag_wape[lag_idx + 1] = float(np.sum(np.abs(a - p)) / total)
            fold_lag_wapes.append(this_fold_lag_wape)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                model_name=model_name,
                wape=metrics['wape'],
                mape=metrics['mape'],
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                bias=metrics['bias'],
                n_train_samples=len(y_train),
                n_val_samples=len(y_val),
                train_time_seconds=train_time,
                predictions=predictions if store_predictions else None,
                actuals=actuals if store_predictions else None,
            )
            fold_results.append(fold_result)

        result = ModelCVResult(
            model_name=model_name,
            model_config=model_config,
            fold_results=fold_results,
        )

        # Compute horizon-weighted WAPE if weights provided
        # Uses per-fold per-lag WAPEs stored in fold_lag_wapes (NOT the loop variable)
        if horizon_weights and fold_results:
            all_lag_wapes = {}
            for fold_lw in fold_lag_wapes:
                for lag, w in fold_lw.items():
                    all_lag_wapes.setdefault(lag, []).append(w)

            for lag in all_lag_wapes:
                result.per_horizon_wape[lag] = float(np.mean(all_lag_wapes[lag]))

            # Compute horizon-weighted WAPE
            weighted_sum = 0.0
            weight_total = 0.0
            for lag, weight in horizon_weights.items():
                if lag in result.per_horizon_wape:
                    weighted_sum += weight * result.per_horizon_wape[lag]
                    weight_total += weight
            if weight_total > 0:
                result.horizon_weighted_wape = weighted_sum / weight_total
            else:
                result.horizon_weighted_wape = result.avg_wape

        return result

    def evaluate_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_configs: List[Dict[str, Any]],
        model_fn: Callable,
        segment_id: str = 'default',
        period_col: str = 'period_idx',
        feature_cols: Optional[List[str]] = None,
        optimize_ensemble: bool = True,
    ) -> WalkForwardCVResult:
        """
        Evaluate multiple models and select the best.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model_configs : list of dict
            List of model configurations, each with 'name' and other params
        model_fn : Callable
            Function to create/fit models
        segment_id : str
            Identifier for this segment
        period_col : str
            Column name for period indices
        feature_cols : list, optional
            Feature columns to use
        optimize_ensemble : bool
            Whether to optimize ensemble weights

        Returns
        -------
        WalkForwardCVResult
            Complete CV results with best model selection
        """
        result = WalkForwardCVResult(
            segment_id=segment_id,
            n_folds=self.n_splits,
            forecast_horizon=self.forecast_horizon,
            min_train_periods=self.min_train_periods,
        )

        # Evaluate each model
        for config in model_configs:
            model_name = config.get('name', 'unnamed')
            logger.info(f"Evaluating model: {model_name}")

            model_result = self.evaluate_model(
                X=X,
                y=y,
                model_fn=model_fn,
                model_name=model_name,
                model_config=config,
                period_col=period_col,
                feature_cols=feature_cols,
                store_predictions=optimize_ensemble,
            )
            result.model_results[model_name] = model_result

        # Select best model
        result.select_best_model()

        # Optimize ensemble weights if requested
        if optimize_ensemble and len(result.model_results) > 1:
            self._optimize_ensemble_weights(result)

        return result

    def _compute_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, float]:
        """Compute forecast accuracy metrics."""
        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)

        # Filter out zeros for percentage metrics
        non_zero_mask = actuals > 0

        # WAPE (Weighted Absolute Percentage Error)
        wape = np.sum(np.abs(actuals - predictions)) / max(np.sum(np.abs(actuals)), 1e-10)

        # MAPE (only for non-zero actuals)
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask]))
        else:
            mape = 0.0

        # RMSE
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

        # MAE
        mae = np.mean(np.abs(actuals - predictions))

        # Bias (mean error)
        bias = np.mean(predictions - actuals)

        return {
            'wape': float(wape),
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
        }

    def _optimize_ensemble_weights(self, result: WalkForwardCVResult) -> None:
        """
        Optimize ensemble weights across all folds.

        Uses average optimal weights from each fold to get robust ensemble.
        """
        from scipy.optimize import minimize

        model_names = list(result.model_results.keys())
        n_models = len(model_names)

        if n_models < 2:
            return

        # Collect predictions from all models for each fold
        fold_optimal_weights = []

        for fold_idx in range(self.n_splits):
            # Get predictions for this fold from each model
            fold_predictions = {}
            fold_actuals = None

            for model_name, model_result in result.model_results.items():
                for fold_result in model_result.fold_results:
                    if fold_result.fold_idx == fold_idx and fold_result.predictions is not None:
                        fold_predictions[model_name] = fold_result.predictions
                        if fold_actuals is None:
                            fold_actuals = fold_result.actuals

            if len(fold_predictions) < 2 or fold_actuals is None:
                continue

            # Optimize weights for this fold
            def ensemble_wape(weights):
                ensemble_pred = sum(
                    w * fold_predictions[name]
                    for w, name in zip(weights, model_names)
                    if name in fold_predictions
                )
                return np.sum(np.abs(fold_actuals - ensemble_pred)) / max(np.sum(np.abs(fold_actuals)), 1e-10)

            # Constraints: weights sum to 1, all >= 0
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n_models)]
            initial_weights = np.ones(n_models) / n_models

            try:
                res = minimize(
                    ensemble_wape,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                )
                if res.success:
                    fold_optimal_weights.append(res.x)
            except Exception as e:
                logger.warning(f"Ensemble optimization failed for fold {fold_idx}: {e}")

        if not fold_optimal_weights:
            # Fallback to equal weights
            result.ensemble_weights = {name: 1.0 / n_models for name in model_names}
            return

        # Average weights across folds
        avg_weights = np.mean(fold_optimal_weights, axis=0)
        avg_weights = avg_weights / avg_weights.sum()  # Normalize

        result.ensemble_weights = {
            name: float(w) for name, w in zip(model_names, avg_weights)
        }

        # Estimate ensemble WAPE
        ensemble_wapes = []
        for fold_idx in range(self.n_splits):
            fold_predictions = {}
            fold_actuals = None

            for model_name, model_result in result.model_results.items():
                for fold_result in model_result.fold_results:
                    if fold_result.fold_idx == fold_idx and fold_result.predictions is not None:
                        fold_predictions[model_name] = fold_result.predictions
                        if fold_actuals is None:
                            fold_actuals = fold_result.actuals

            if fold_predictions and fold_actuals is not None:
                ensemble_pred = sum(
                    result.ensemble_weights.get(name, 0) * pred
                    for name, pred in fold_predictions.items()
                )
                wape = np.sum(np.abs(fold_actuals - ensemble_pred)) / max(np.sum(np.abs(fold_actuals)), 1e-10)
                ensemble_wapes.append(wape)

        if ensemble_wapes:
            result.ensemble_avg_wape = float(np.mean(ensemble_wapes))


def create_period_index(
    df: pd.DataFrame,
    time_col: str,
    time_format: str = 'year_week',
) -> pd.Series:
    """
    Create sequential period index from time column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time column
    time_col : str
        Name of time column
    time_format : str
        Format of time column: 'year_week', 'year_month', 'date'

    Returns
    -------
    pd.Series
        Integer period index (0, 1, 2, ...)
    """
    times = df[time_col]

    if time_format == 'year_week':
        # Assume format like 202301, 202302, etc.
        unique_times = sorted(times.unique())
    elif time_format == 'year_month':
        unique_times = sorted(times.unique())
    elif time_format == 'date':
        unique_times = sorted(pd.to_datetime(times).unique())
    else:
        unique_times = sorted(times.unique())

    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    return times.map(time_to_idx)


def walk_forward_model_selection(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_models: List[Dict[str, Any]],
    model_fn: Callable,
    segment_id: str = 'default',
    n_folds: int = 5,
    forecast_horizon: int = 8,
    min_train_periods: int = 52,
    period_col: str = 'period_idx',
    feature_cols: Optional[List[str]] = None,
    optimize_ensemble: bool = True,
    strategy: str = 'expanding',
) -> WalkForwardCVResult:
    """
    Convenience function for walk-forward model selection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with period_col
    y : pd.Series
        Target variable
    candidate_models : list of dict
        Model configurations to evaluate
    model_fn : Callable
        Function to create/fit models
    segment_id : str
        Segment identifier
    n_folds : int
        Number of CV folds
    forecast_horizon : int
        Forecast horizon in periods
    min_train_periods : int
        Minimum training periods
    period_col : str
        Period column name
    feature_cols : list, optional
        Feature columns to use
    optimize_ensemble : bool
        Whether to optimize ensemble weights
    strategy : str
        'expanding' or 'rolling'

    Returns
    -------
    WalkForwardCVResult
        Complete CV results
    """
    cv = WalkForwardCV(
        n_splits=n_folds,
        forecast_horizon=forecast_horizon,
        min_train_periods=min_train_periods,
        strategy=strategy,
    )

    return cv.evaluate_models(
        X=X,
        y=y,
        model_configs=candidate_models,
        model_fn=model_fn,
        segment_id=segment_id,
        period_col=period_col,
        feature_cols=feature_cols,
        optimize_ensemble=optimize_ensemble,
    )


def detect_concept_drift(
    fold_wapes: List[float],
    threshold: float = 0.02,
) -> Dict[str, Any]:
    """
    Detect concept drift from fold WAPE progression.

    Parameters
    ----------
    fold_wapes : list of float
        WAPE values for each fold (ordered by time)
    threshold : float
        WAPE increase per fold that triggers drift detection

    Returns
    -------
    dict
        Drift detection results
    """
    if len(fold_wapes) < 3:
        return {
            'drift_detected': False,
            'trend': 0.0,
            'confidence': 'low',
            'recommendation': 'Not enough folds for drift detection',
        }

    # Linear regression slope
    x = np.arange(len(fold_wapes))
    slope, intercept = np.polyfit(x, fold_wapes, 1)

    # R-squared for confidence
    y_pred = slope * x + intercept
    ss_res = np.sum((np.array(fold_wapes) - y_pred) ** 2)
    ss_tot = np.sum((np.array(fold_wapes) - np.mean(fold_wapes)) ** 2)
    r_squared = 1 - (ss_res / max(ss_tot, 1e-10))

    drift_detected = slope > threshold and r_squared > 0.5

    confidence = 'high' if r_squared > 0.7 else 'medium' if r_squared > 0.4 else 'low'

    if drift_detected:
        if slope > 0.05:
            recommendation = 'Severe drift: Consider retraining more frequently or using more recent data only'
        else:
            recommendation = 'Moderate drift: Monitor performance and consider shorter retraining cycles'
    else:
        recommendation = 'No significant drift detected'

    return {
        'drift_detected': drift_detected,
        'trend': float(slope),
        'r_squared': float(r_squared),
        'confidence': confidence,
        'recommendation': recommendation,
        'fold_wapes': fold_wapes,
    }


# =============================================================================
# Integration helpers for model_training.py
# =============================================================================

def create_lightgbm_model_fn():
    """Create a model function for LightGBM."""
    def model_fn(X_train, y_train, config):
        import lightgbm as lgb

        params = {
            'objective': config.get('objective', 'regression'),
            'metric': 'mae',
            'learning_rate': config.get('learning_rate', 0.05),
            'num_leaves': config.get('num_leaves', 31),
            'min_child_samples': config.get('min_child_samples', 20),
            'reg_alpha': config.get('reg_alpha', 0.1),
            'reg_lambda': config.get('reg_lambda', 0.1),
            'n_estimators': config.get('n_estimators', 200),
            'verbose': -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return model

    return model_fn


def create_xgboost_model_fn():
    """Create a model function for XGBoost."""
    def model_fn(X_train, y_train, config):
        import xgboost as xgb

        params = {
            'objective': config.get('objective', 'reg:squarederror'),
            'learning_rate': config.get('learning_rate', 0.05),
            'max_depth': config.get('max_depth', 6),
            'min_child_weight': config.get('min_child_weight', 1),
            'reg_alpha': config.get('reg_alpha', 0.1),
            'reg_lambda': config.get('reg_lambda', 0.1),
            'n_estimators': config.get('n_estimators', 200),
            'verbosity': 0,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model

    return model_fn


def create_catboost_model_fn():
    """Create a model function for CatBoost."""
    def model_fn(X_train, y_train, config):
        from catboost import CatBoostRegressor

        params = {
            'loss_function': config.get('loss_function', 'RMSE'),
            'learning_rate': config.get('learning_rate', 0.05),
            'depth': config.get('depth', 6),
            'l2_leaf_reg': config.get('l2_leaf_reg', 3),
            'iterations': config.get('iterations', 200),
            'verbose': False,
        }

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        return model

    return model_fn


def get_default_candidate_models(
    demand_pattern: str = 'smooth',
    loss_function: str = 'mse',
) -> List[Dict[str, Any]]:
    """
    Get default candidate model configurations based on demand pattern.

    Parameters
    ----------
    demand_pattern : str
        Demand pattern: 'smooth', 'erratic', 'intermittent', 'lumpy'
    loss_function : str
        Recommended loss function from segmentation

    Returns
    -------
    list of dict
        Model configurations for walk-forward CV
    """
    # Base configurations
    lightgbm_base = {
        'name': 'lightgbm',
        'model_type': 'lightgbm',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'n_estimators': 200,
    }

    xgboost_base = {
        'name': 'xgboost',
        'model_type': 'xgboost',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'n_estimators': 200,
    }

    catboost_base = {
        'name': 'catboost',
        'model_type': 'catboost',
        'learning_rate': 0.05,
        'depth': 6,
        'iterations': 200,
    }

    # Adjust for demand pattern
    if demand_pattern in ('intermittent', 'lumpy'):
        # More regularization for sparse data
        lightgbm_base['min_child_samples'] = 30
        lightgbm_base['reg_alpha'] = 0.2
        xgboost_base['min_child_weight'] = 3
        xgboost_base['reg_alpha'] = 0.2
        catboost_base['l2_leaf_reg'] = 5

        if loss_function == 'tweedie':
            lightgbm_base['objective'] = 'tweedie'
            lightgbm_base['tweedie_variance_power'] = 1.5

    elif demand_pattern == 'erratic':
        # Use Huber loss for robustness
        if loss_function == 'huber':
            lightgbm_base['objective'] = 'huber'
            lightgbm_base['alpha'] = 0.9
            xgboost_base['objective'] = 'reg:pseudohubererror'

    return [lightgbm_base, xgboost_base, catboost_base]
