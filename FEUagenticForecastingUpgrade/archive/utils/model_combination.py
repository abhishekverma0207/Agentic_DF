# utils/model_combination.py
"""
Phase 8: Model Combination at Multiple Levels.

Two combination strategies:

1. **Multi-Level Forecast Combination** — Combine forecasts from key-level,
   segment-level, and category-level models using data-adaptive weights.
   Keys with more data trust key-level models; sparse keys trust aggregate.

2. **Diverse Stacking** — Train base models from different families
   (LightGBM, XGBoost, CatBoost, quantile, global, direct, hierarchical)
   and combine with a meta-learner. The meta-learner learns which model
   to trust for each data pattern.

Both produce a single final forecast per (key, period).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 1. MULTI-LEVEL FORECAST COMBINATION
# =============================================================================

def combine_multi_level_forecasts(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    keys: np.ndarray,
    key_obs_counts: Dict[str, int],
    method: str = 'data_adaptive',
    lambda_param: float = 100.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Combine forecasts from multiple hierarchy levels.

    Parameters
    ----------
    forecasts : Dict[str, np.ndarray]
        {level_name: predictions} e.g., {'key': preds, 'segment': preds, 'global': preds}
    actuals : np.ndarray
        Actual values (for weight optimization if method='optimal')
    keys : np.ndarray
        Key identifiers for each prediction row
    key_obs_counts : Dict[str, int]
        Number of training observations per key
    method : str
        'optimal': minimize validation WAPE via constrained optimization
        'inverse_error': weight by inverse of validation WAPE
        'data_adaptive': more data → more weight to granular level
    lambda_param : float
        Shrinkage parameter for data_adaptive (default 100)

    Returns
    -------
    Tuple[np.ndarray, Dict]
        (combined_predictions, metadata with weights)
    """
    level_names = list(forecasts.keys())
    n_levels = len(level_names)

    if n_levels == 0:
        raise ValueError("No forecasts to combine")
    if n_levels == 1:
        return list(forecasts.values())[0], {'method': 'single_model', 'weights': {level_names[0]: 1.0}}

    logger.info(f"Combining {n_levels} forecast levels: {level_names} (method={method})")

    n_samples = len(actuals)

    if method == 'optimal':
        weights = _optimize_combination_weights(forecasts, actuals)
    elif method == 'inverse_error':
        weights = _inverse_error_weights(forecasts, actuals)
    elif method == 'data_adaptive':
        weights = _data_adaptive_weights(forecasts, keys, key_obs_counts, lambda_param)
    else:
        # Equal weights
        weights = {name: 1.0 / n_levels for name in level_names}

    # Apply weights — can be per-key (data_adaptive) or global
    if isinstance(list(weights.values())[0], dict):
        # Per-key weights: weights = {level: {key: weight}}
        combined = np.zeros(n_samples)
        for i in range(n_samples):
            key = keys[i]
            total_w = 0
            for level in level_names:
                w = weights[level].get(key, weights[level].get('_default', 1.0 / n_levels))
                combined[i] += w * forecasts[level][i]
                total_w += w
            if total_w > 0:
                combined[i] /= total_w
    else:
        # Global weights
        combined = np.zeros(n_samples)
        for level in level_names:
            combined += weights[level] * forecasts[level]

    combined = np.maximum(combined, 0)

    # Compute combined WAPE
    combined_wape = _compute_wape(actuals, combined)
    per_level_wape = {name: _compute_wape(actuals, forecasts[name]) for name in level_names}

    metadata = {
        'method': method,
        'weights': {k: (float(np.mean(list(v.values()))) if isinstance(v, dict) else float(v))
                    for k, v in weights.items()},
        'combined_wape': combined_wape,
        'per_level_wape': per_level_wape,
        'improvement_vs_best_single': min(per_level_wape.values()) - combined_wape,
    }

    logger.info(f"Combined WAPE: {combined_wape:.4f} "
                f"(best single: {min(per_level_wape.values()):.4f}, "
                f"improvement: {metadata['improvement_vs_best_single']:.4f})")

    return combined, metadata


def _optimize_combination_weights(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
) -> Dict[str, float]:
    """Find weights minimizing WAPE via constrained optimization."""
    from scipy.optimize import minimize

    level_names = list(forecasts.keys())
    n_levels = len(level_names)
    preds_matrix = np.column_stack([forecasts[name] for name in level_names])

    def objective(w):
        combined = preds_matrix @ w
        combined = np.maximum(combined, 0)
        total_actual = np.sum(np.abs(actuals)) + 1e-10
        return np.sum(np.abs(actuals - combined)) / total_actual

    # Constraints: weights sum to 1, each >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * n_levels
    x0 = np.ones(n_levels) / n_levels

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        weights = {name: float(result.x[i]) for i, name in enumerate(level_names)}
    else:
        logger.warning("Optimization failed, using equal weights")
        weights = {name: 1.0 / n_levels for name in level_names}

    logger.info(f"Optimal weights: {weights}")
    return weights


def _inverse_error_weights(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
) -> Dict[str, float]:
    """Weight by inverse of validation WAPE."""
    errors = {}
    for name, preds in forecasts.items():
        wape = _compute_wape(actuals, preds)
        errors[name] = max(wape, 0.001)  # Floor to avoid division by zero

    inv_errors = {name: 1.0 / err for name, err in errors.items()}
    total = sum(inv_errors.values())
    return {name: v / total for name, v in inv_errors.items()}


def _data_adaptive_weights(
    forecasts: Dict[str, np.ndarray],
    keys: np.ndarray,
    key_obs_counts: Dict[str, int],
    lambda_param: float,
) -> Dict[str, Dict[str, float]]:
    """Data-adaptive per-key weights: more data → more weight to granular level."""
    level_names = sorted(forecasts.keys())

    # Define granularity order (most granular first)
    granularity = {}
    for name in level_names:
        if 'key' in name.lower() or 'individual' in name.lower():
            granularity[name] = 3  # Most granular
        elif 'segment' in name.lower() or 'group' in name.lower() or 'hier' in name.lower():
            granularity[name] = 2
        else:  # global, category
            granularity[name] = 1

    weights = {name: {} for name in level_names}
    unique_keys = set(keys)

    for key in unique_keys:
        n_obs = key_obs_counts.get(key, 10)

        # Shrinkage: more data → more weight to granular
        key_trust = min(n_obs / lambda_param, 0.6)
        remaining = 1.0 - key_trust

        for name in level_names:
            g = granularity.get(name, 1)
            if g == 3:
                weights[name][key] = key_trust
            elif g == 2:
                weights[name][key] = remaining * 0.6
            else:
                weights[name][key] = remaining * 0.4

        # Normalize
        total = sum(weights[name][key] for name in level_names)
        if total > 0:
            for name in level_names:
                weights[name][key] /= total

    return weights


# =============================================================================
# 2. DIVERSE STACKING
# =============================================================================

class StackedEnsembleWrapper:
    """Wrapper for stacked ensemble with meta-learner."""

    def __init__(self, base_models: Dict[str, Any], meta_model: Any, feature_cols: List[str]):
        self.base_models = base_models
        self.meta_model = meta_model
        self.feature_cols = feature_cols
        self.model_type = 'stacked_ensemble'

    def predict(self, X, keys=None):
        X_arr = np.atleast_2d(X)
        # Get base model predictions
        meta_features = self._get_meta_features(X_arr, keys)
        preds = self.meta_model.predict(meta_features)
        return np.maximum(preds, 0)

    def _get_meta_features(self, X, keys=None):
        cols = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict'):
                if hasattr(model, 'model_type') and keys is not None:
                    preds = model.predict(X, keys=keys)
                else:
                    preds = model.predict(X)
                cols.append(np.maximum(preds, 0))
        return np.column_stack(cols) if cols else X


def train_diverse_stacking(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    key_col: str,
    base_model_configs: Optional[List[Dict[str, Any]]] = None,
    meta_model_family: str = 'lightgbm',
    **kwargs,
) -> Tuple[StackedEnsembleWrapper, Dict[str, Any]]:
    """
    Train a diverse stacking ensemble with base models from multiple families.

    Default base models:
    - LightGBM (MSE)
    - LightGBM (MAE / quantile median)
    - XGBoost
    - CatBoost (with key embedding)
    - RandomForest

    Meta-learner: LightGBM trained on base model predictions.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
    feature_cols : List[str]
    target_col, key_col : str
    base_model_configs : List[Dict], optional
        Custom base model specs. Default uses diverse set.
    meta_model_family : str
        Meta-learner family

    Returns
    -------
    Tuple[StackedEnsembleWrapper, Dict]
    """
    logger.info("Training diverse stacking ensemble...")

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df[target_col].fillna(0).values

    # Train base models
    base_models = {}
    base_val_preds = {}

    # 1. LightGBM MSE
    logger.info("  Base model 1: LightGBM (MSE)")
    import lightgbm as lgb
    m1 = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, verbosity=-1,
    )
    m1.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(30, verbose=False)])
    base_models['lgbm_mse'] = m1
    base_val_preds['lgbm_mse'] = np.maximum(m1.predict(X_val), 0)

    # 2. LightGBM Quantile Median
    logger.info("  Base model 2: LightGBM (Quantile 0.5)")
    m2 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.5,
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, verbosity=-1,
    )
    m2.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(30, verbose=False)])
    base_models['lgbm_median'] = m2
    base_val_preds['lgbm_median'] = np.maximum(m2.predict(X_val), 0)

    # 3. XGBoost
    logger.info("  Base model 3: XGBoost")
    try:
        import xgboost as xgb
        m3 = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, verbosity=0,
        )
        m3.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        base_models['xgboost'] = m3
        base_val_preds['xgboost'] = np.maximum(m3.predict(X_val), 0)
    except ImportError:
        logger.warning("  XGBoost not available, skipping")

    # 4. CatBoost
    logger.info("  Base model 4: CatBoost")
    try:
        from catboost import CatBoostRegressor
        m4 = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            l2_leaf_reg=3.0, verbose=0, early_stopping_rounds=30,
        )
        m4.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        base_models['catboost'] = m4
        base_val_preds['catboost'] = np.maximum(m4.predict(X_val), 0)
    except ImportError:
        logger.warning("  CatBoost not available, skipping")

    # 5. RandomForest (diversity: non-boosting)
    logger.info("  Base model 5: RandomForest")
    from sklearn.ensemble import RandomForestRegressor
    m5 = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    m5.fit(X_train, y_train)
    base_models['random_forest'] = m5
    base_val_preds['random_forest'] = np.maximum(m5.predict(X_val), 0)

    # Log individual WAPE
    logger.info("  Base model WAPEs:")
    for name, preds in base_val_preds.items():
        logger.info(f"    {name}: {_compute_wape(y_val, preds):.4f}")

    # Train meta-learner on out-of-fold predictions
    logger.info("  Training meta-learner...")

    # Create meta-features from training set using cross-validation
    from sklearn.model_selection import KFold
    n_folds = 3
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    meta_train = np.zeros((len(X_train), len(base_models)))
    model_names = list(base_models.keys())

    for fold_idx, (trn_idx, oof_idx) in enumerate(kf.split(X_train)):
        X_trn, X_oof = X_train[trn_idx], X_train[oof_idx]
        y_trn = y_train[trn_idx]

        for j, name in enumerate(model_names):
            # Retrain base model on fold
            fold_model = _clone_and_train(base_models[name], X_trn, y_trn, X_oof, y_train[oof_idx])
            meta_train[oof_idx, j] = np.maximum(fold_model.predict(X_oof), 0)

    # Meta-learner validation features
    meta_val = np.column_stack([base_val_preds[name] for name in model_names])

    # Train meta-learner
    meta_model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.1, num_leaves=15,
        min_child_samples=20, reg_lambda=3.0, verbosity=-1,
    )
    meta_model.fit(meta_train, y_train,
                   eval_set=[(meta_val, y_val)],
                   callbacks=[lgb.early_stopping(20, verbose=False)])

    # Final predictions
    stacked_preds = np.maximum(meta_model.predict(meta_val), 0)
    stacked_wape = _compute_wape(y_val, stacked_preds)

    # Meta-model feature importances (which base model matters most)
    importances = meta_model.feature_importances_
    model_weights = {name: float(importances[i]) for i, name in enumerate(model_names)}
    total_imp = sum(model_weights.values()) + 1e-10
    model_weights = {k: v / total_imp for k, v in model_weights.items()}

    wrapper = StackedEnsembleWrapper(base_models, meta_model, feature_cols)

    metadata = {
        'model_type': 'stacked_ensemble',
        'base_models': model_names,
        'n_base_models': len(base_models),
        'val_wape': stacked_wape,
        'per_model_wape': {name: _compute_wape(y_val, base_val_preds[name]) for name in model_names},
        'meta_model_weights': model_weights,
        'improvement_vs_best_base': min(
            _compute_wape(y_val, base_val_preds[name]) for name in model_names
        ) - stacked_wape,
    }

    logger.info(f"Stacked ensemble: val_wape={stacked_wape:.4f} "
                f"(best base: {min(metadata['per_model_wape'].values()):.4f}, "
                f"improvement: {metadata['improvement_vs_best_base']:.4f})")
    logger.info(f"  Meta-model weights: {model_weights}")

    return wrapper, metadata


def _clone_and_train(model, X_train, y_train, X_val, y_val):
    """Clone and retrain a model on a subset. Handles LightGBM, XGBoost, CatBoost, sklearn."""
    import lightgbm as lgb

    if isinstance(model, lgb.LGBMRegressor):
        params = model.get_params()
        clone = lgb.LGBMRegressor(**params)
        clone.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
        return clone

    try:
        import xgboost as xgb
        if isinstance(model, xgb.XGBRegressor):
            params = model.get_params()
            clone = xgb.XGBRegressor(**params)
            clone.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return clone
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor
        if isinstance(model, CatBoostRegressor):
            params = model.get_all_params()
            # Remove non-constructor params that CatBoost stores internally
            for key in ['model_size_reg', 'train_dir', 'gpu_ram_part']:
                params.pop(key, None)
            clone = CatBoostRegressor(**params)
            clone.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            return clone
    except (ImportError, Exception):
        pass

    # Fallback: sklearn-style clone
    from sklearn.base import clone as sk_clone
    try:
        clone = sk_clone(model)
        clone.fit(X_train, y_train)
        return clone
    except Exception:
        # Last resort: retrain from scratch with RandomForest
        from sklearn.ensemble import RandomForestRegressor
        clone = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=1)
        clone.fit(X_train, y_train)
        return clone


def _compute_wape(actuals, predictions):
    total = np.sum(np.abs(actuals))
    if total < 1e-10:
        return 1.0
    return float(np.sum(np.abs(actuals - predictions)) / total)
