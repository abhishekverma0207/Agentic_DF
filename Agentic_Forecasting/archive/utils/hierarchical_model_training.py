# utils/hierarchical_model_training.py
"""
Phase 3: Hierarchical Model Training.

Three model types that exploit hierarchy structure:

1. **Global-Local**: Single global model on ALL keys (with key identity as
   feature) + per-key bias correction. Maximizes data utilization.

2. **Mixed-Effects**: Global model (no key identity) + per-key residual
   adjustment models. Captures both global patterns and key-specific deviations.

3. **Multi-Level Ensemble**: Models at global, hierarchy-group, and key levels
   combined with data-adaptive weights. More data → more weight to key-level.

All models return a HierarchicalModelWrapper that provides a standard
.predict(X, keys=None) interface compatible with the existing inference pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# LOSS-PARAM TRANSLATION HELPERS
# =============================================================================
# 2026-04 FIX: hierarchical trainers (global_local / mixed_effects /
# multi_level_ensemble) previously ignored the loss_params dict from the
# segmentation strategy and always trained with LightGBM's default
# squared-error objective. For zero-inflated lumpy demand (95% of UK
# keys) this is catastrophic for WAPE because MSE chases the mean,
# which outliers pull upward, dominating pooled training.
#
# These helpers translate the generic loss_params dict
#   {'objective': 'tweedie', 'tweedie_variance_power': 1.7}
# into model-family-specific kwargs for LightGBM / XGBoost. If
# loss_params is empty or names an objective the family doesn't
# support, the defaults are returned unchanged.
# =============================================================================

_LGBM_SUPPORTED_OBJECTIVES = {
    'regression', 'regression_l1', 'mape', 'huber', 'quantile',
    'tweedie', 'poisson', 'gamma',
}
_XGB_OBJECTIVE_MAP = {
    'regression':         'reg:squarederror',
    'regression_l1':      'reg:absoluteerror',
    'mape':               'reg:squarederror',  # no native MAPE in xgb regressor
    'huber':              'reg:pseudohubererror',
    'quantile':           'reg:quantileerror',
    'tweedie':            'reg:tweedie',
    'poisson':            'count:poisson',
    'gamma':              'reg:gamma',
    # Pass-through for already-xgboost-style objectives
    'reg:squarederror':   'reg:squarederror',
    'reg:absoluteerror':  'reg:absoluteerror',
    'reg:tweedie':        'reg:tweedie',
    'reg:pseudohubererror': 'reg:pseudohubererror',
    'reg:gamma':          'reg:gamma',
    'count:poisson':      'count:poisson',
}


def _lgbm_kwargs_from_loss(loss_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate a generic loss_params dict into LightGBM LGBMRegressor kwargs.

    Keeps only the keys LightGBM understands; silently drops anything it
    doesn't (including XGBoost-specific keys, segment hyperparameters
    that slipped in, etc.). Returns {} when loss_params is None/empty.
    """
    if not loss_params:
        return {}
    out: Dict[str, Any] = {}
    obj = loss_params.get('objective')
    if obj:
        # Normalise xgboost-flavoured objectives back to the LGBM name
        _xgb_to_lgbm = {
            'reg:squarederror':     'regression',
            'reg:absoluteerror':    'regression_l1',
            'reg:tweedie':          'tweedie',
            'reg:pseudohubererror': 'huber',
            'reg:gamma':            'gamma',
            'count:poisson':        'poisson',
        }
        obj = _xgb_to_lgbm.get(obj, obj)
        if obj in _LGBM_SUPPORTED_OBJECTIVES:
            out['objective'] = obj
    # Loss-specific auxiliary params
    if out.get('objective') == 'tweedie':
        # Segmentation strategy ships this as 'tweedie_variance_power' OR
        # legacy 'tweedie_power'. LightGBM wants 'tweedie_variance_power'.
        tvp = loss_params.get('tweedie_variance_power',
                               loss_params.get('tweedie_power'))
        if tvp is not None:
            out['tweedie_variance_power'] = float(tvp)
    if out.get('objective') == 'huber':
        if 'alpha' in loss_params:
            out['alpha'] = float(loss_params['alpha'])
    if out.get('objective') == 'quantile':
        if 'alpha' in loss_params:
            out['alpha'] = float(loss_params['alpha'])
    return out


def _xgb_kwargs_from_loss(loss_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate a generic loss_params dict into XGBRegressor kwargs."""
    if not loss_params:
        return {}
    out: Dict[str, Any] = {}
    obj = loss_params.get('objective')
    if obj:
        xgb_obj = _XGB_OBJECTIVE_MAP.get(obj)
        if xgb_obj:
            out['objective'] = xgb_obj
    if out.get('objective') == 'reg:tweedie':
        tvp = loss_params.get('tweedie_variance_power',
                               loss_params.get('tweedie_power'))
        if tvp is not None:
            out['tweedie_variance_power'] = float(tvp)
    if out.get('objective') == 'reg:quantileerror':
        if 'alpha' in loss_params:
            out['quantile_alpha'] = float(loss_params['alpha'])
    return out


def _loss_needs_nonneg_target(loss_params: Optional[Dict[str, Any]]) -> bool:
    """Tweedie, Poisson, Gamma all require y >= 0."""
    if not loss_params:
        return False
    obj = (loss_params.get('objective') or '').lower()
    return obj in ('tweedie', 'poisson', 'gamma',
                   'reg:tweedie', 'count:poisson', 'reg:gamma')


# =============================================================================
# WRAPPER CLASS FOR INFERENCE COMPATIBILITY
# =============================================================================

class HierarchicalModelWrapper:
    """
    Wrapper providing .predict(X, keys=) interface for hierarchical models.

    This is pickle-serializable and compatible with the existing inference
    pipeline in utils/inference.py.
    """

    def __init__(self, model_dict: Dict[str, Any], model_type: str):
        self.model_dict = model_dict
        self.model_type = model_type

    def predict(self, X, keys=None):
        """
        Predict using the hierarchical model.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix. Shape (n_samples, n_features).
        keys : np.ndarray or List[str], optional
            Key identifiers for each row. Required for global_local and
            mixed_effects. If None, uses global model only.

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,)
        """
        if self.model_type == 'global_local':
            return self._predict_global_local(X, keys)
        elif self.model_type == 'mixed_effects':
            return self._predict_mixed_effects(X, keys)
        elif self.model_type == 'multi_level_ensemble':
            return self._predict_multi_level(X, keys)
        else:
            raise ValueError(f"Unknown hierarchical model type: {self.model_type}")

    def _predict_global_local(self, X, keys):
        global_model = self.model_dict['global_model']
        key_bias = self.model_dict.get('key_bias', {})
        key_encoder = self.model_dict.get('key_encoder', None)

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        # Encode keys and add as feature column.
        # When keys is None but the global model was trained with a key-encoding
        # column, we must still append a default encoding (-1 = unknown key) so
        # the feature count matches what the model expects.
        if key_encoder is not None and keys is not None:
            encoded = np.array([
                key_encoder.get(k, -1) for k in keys
            ]).reshape(-1, 1)
            X_with_key = np.hstack([X_arr, encoded])
        elif key_encoder is not None:
            # No keys provided but model expects key-encoding column —
            # use -1 (unknown key) as default to keep feature count consistent.
            default_encoded = np.full((X_arr.shape[0], 1), -1, dtype=float)
            X_with_key = np.hstack([X_arr, default_encoded])
        else:
            X_with_key = X_arr

        preds = global_model.predict(X_with_key)

        # Add per-key bias
        if keys is not None:
            bias = np.array([key_bias.get(k, 0.0) for k in keys])
            preds = preds + bias

        return np.maximum(preds, 0)

    def _predict_mixed_effects(self, X, keys):
        global_model = self.model_dict['global_model']
        adjustments = self.model_dict.get('key_adjustments', {})

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        # Global prediction
        preds = global_model.predict(X_arr)

        # Per-key adjustment
        if keys is not None:
            for i, key in enumerate(keys):
                if key in adjustments:
                    adj_model = adjustments[key]
                    if hasattr(adj_model, 'predict'):
                        adj = adj_model.predict(X_arr[i:i + 1])[0]
                    else:
                        adj = float(adj_model)  # Scalar bias fallback
                    preds[i] += adj

        return np.maximum(preds, 0)

    def _predict_multi_level(self, X, keys):
        global_model = self.model_dict['global_model']
        hier_models = self.model_dict.get('hier_models', {})
        key_models = self.model_dict.get('key_models', {})
        weights = self.model_dict.get('weights', {})
        key_to_group = self.model_dict.get('key_to_group', {})
        default_weights = self.model_dict.get('default_weights', (0.5, 0.3, 0.2))

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        preds = np.zeros(len(X_arr))

        for i in range(len(X_arr)):
            x_row = X_arr[i:i + 1]
            key = keys[i] if keys is not None else None

            # Get weights for this key
            w_g, w_h, w_k = weights.get(key, default_weights) if key else default_weights

            # Global prediction
            pred_global = global_model.predict(x_row)[0]

            # Hierarchy group prediction
            group = key_to_group.get(key) if key else None
            if group and group in hier_models:
                pred_hier = hier_models[group].predict(x_row)[0]
            else:
                pred_hier = pred_global
                w_g += w_h  # Redistribute hierarchy weight to global
                w_h = 0

            # Key-level prediction
            if key and key in key_models:
                pred_key = key_models[key].predict(x_row)[0]
            else:
                pred_key = pred_hier
                w_h += w_k  # Redistribute key weight to hierarchy
                w_k = 0

            # Normalize weights
            w_total = w_g + w_h + w_k
            if w_total > 0:
                preds[i] = (w_g * pred_global + w_h * pred_hier + w_k * pred_key) / w_total
            else:
                preds[i] = pred_global

        return np.maximum(preds, 0)


# =============================================================================
# 1. GLOBAL-LOCAL MODEL
# =============================================================================

def train_global_local_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    key_col: str,
    model_family: str = 'lightgbm',
    loss_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    **kwargs,
) -> Tuple[HierarchicalModelWrapper, Dict[str, Any]]:
    """
    Train a Global-Local model: single global model + per-key bias correction.

    The global model trains on ALL keys simultaneously, with key identity
    encoded as an integer feature. Per-key bias is the mean residual from
    training data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (all keys concatenated)
    val_df : pd.DataFrame
        Validation data
    feature_cols : List[str]
        Feature columns
    target_col : str
        Target column
    key_col : str
        Key column
    model_family : str
        'lightgbm' or 'xgboost'

    Returns
    -------
    Tuple[HierarchicalModelWrapper, Dict[str, Any]]
        (wrapped model, metadata dict with val_wape, etc.)
    """
    logger.info(f"Training Global-Local model ({model_family}) on {len(train_df)} rows, "
                f"{train_df[key_col].nunique()} keys")

    # Encode keys as integers
    unique_keys = sorted(train_df[key_col].unique())
    key_encoder = {k: i for i, k in enumerate(unique_keys)}

    train_encoded = np.array([key_encoder.get(k, -1) for k in train_df[key_col]])
    val_encoded = np.array([key_encoder.get(k, -1) for k in val_df[key_col]])

    X_train = np.hstack([train_df[feature_cols].fillna(0).values, train_encoded.reshape(-1, 1)])
    y_train = train_df[target_col].fillna(0).values
    X_val = np.hstack([val_df[feature_cols].fillna(0).values, val_encoded.reshape(-1, 1)])
    y_val = val_df[target_col].fillna(0).values

    # Tweedie / Poisson / Gamma objectives require y >= 0. The raw target
    # column should already be non-negative, but backtest-origin retrains
    # occasionally produce tiny negative lag-residuals after differencing
    # upstream; clip defensively so the booster doesn't error out.
    if _loss_needs_nonneg_target(loss_params):
        y_train = np.clip(y_train, 0, None)
        y_val = np.clip(y_val, 0, None)

    _applied_loss = None
    # Train global model
    if model_family == 'lightgbm':
        try:
            import lightgbm as lgb
            lgbm_kwargs = dict(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                verbosity=-1,
            )
            # Thread the strategy's loss_params into the booster. Tweedie
            # / Poisson / Huber / Quantile all land here.
            loss_kwargs = _lgbm_kwargs_from_loss(loss_params)
            if loss_kwargs:
                lgbm_kwargs.update(loss_kwargs)
                _applied_loss = loss_kwargs.get('objective', 'regression')
                logger.info(
                    f"  global_local: applying loss_params to LightGBM: {loss_kwargs}"
                )
            model = lgb.LGBMRegressor(**lgbm_kwargs)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        except ImportError:
            model = _train_sklearn_fallback(X_train, y_train)
    elif model_family == 'xgboost':
        try:
            import xgboost as xgb
            xgb_kwargs = dict(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                verbosity=0,
            )
            loss_kwargs = _xgb_kwargs_from_loss(loss_params)
            if loss_kwargs:
                xgb_kwargs.update(loss_kwargs)
                _applied_loss = loss_kwargs.get('objective', 'reg:squarederror')
                logger.info(
                    f"  global_local: applying loss_params to XGBoost: {loss_kwargs}"
                )
            model = xgb.XGBRegressor(**xgb_kwargs)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        except ImportError:
            model = _train_sklearn_fallback(X_train, y_train)
    else:
        model = _train_sklearn_fallback(X_train, y_train)

    # Compute per-key bias (mean residual on training data)
    train_preds = model.predict(X_train)
    train_residuals = y_train - train_preds

    key_bias = {}
    for key in unique_keys:
        mask = train_df[key_col].values == key
        if mask.any():
            key_bias[key] = float(np.mean(train_residuals[mask]))

    # Compute validation metrics
    val_preds = model.predict(X_val)
    val_bias = np.array([key_bias.get(k, 0.0) for k in val_df[key_col]])
    val_preds_adjusted = np.maximum(val_preds + val_bias, 0)

    val_wape = _compute_wape(y_val, val_preds_adjusted)

    wrapper = HierarchicalModelWrapper(
        model_dict={
            'global_model': model,
            'key_bias': key_bias,
            'key_encoder': key_encoder,
            'type': 'global_local',
        },
        model_type='global_local',
    )

    metadata = {
        'model_type': 'global_local',
        'model_family': model_family,
        'n_keys': len(unique_keys),
        'n_train_rows': len(train_df),
        'val_wape': val_wape,
        'avg_bias_magnitude': float(np.mean(np.abs(list(key_bias.values())))),
        'applied_loss': _applied_loss,
    }

    logger.info(f"Global-Local model trained: val_wape={val_wape:.4f}, "
                f"avg_bias={metadata['avg_bias_magnitude']:.2f}, "
                f"loss={_applied_loss or 'default'}")

    return wrapper, metadata


# =============================================================================
# 2. MIXED-EFFECTS MODEL
# =============================================================================

def train_mixed_effects_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    key_col: str,
    model_family: str = 'lightgbm',
    min_key_samples: int = 30,
    loss_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    **kwargs,
) -> Tuple[HierarchicalModelWrapper, Dict[str, Any]]:
    """
    Train a Mixed-Effects model: global base + per-key residual adjustments.

    Stage 1: Train global model on all data (no key identity).
    Stage 2: For each key with sufficient data, train a small adjustment
             model on the residuals (actual - global_prediction).
             Keys with < min_key_samples use global-only (no adjustment).

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Training and validation data
    feature_cols : List[str]
        Feature columns
    target_col : str
        Target column
    key_col : str
        Key column
    model_family : str
        Model family for global model
    min_key_samples : int
        Minimum samples per key to train adjustment model

    Returns
    -------
    Tuple[HierarchicalModelWrapper, Dict[str, Any]]
        (wrapped model, metadata)
    """
    logger.info(f"Training Mixed-Effects model on {len(train_df)} rows, "
                f"{train_df[key_col].nunique()} keys (min_key_samples={min_key_samples})")

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df[target_col].fillna(0).values

    # Tweedie / Poisson / Gamma require non-negative target
    if _loss_needs_nonneg_target(loss_params):
        y_train = np.clip(y_train, 0, None)
        y_val = np.clip(y_val, 0, None)

    _applied_loss = None
    # Stage 1: Global model
    if model_family == 'lightgbm':
        try:
            import lightgbm as lgb
            lgbm_kwargs = dict(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, verbosity=-1,
            )
            loss_kwargs = _lgbm_kwargs_from_loss(loss_params)
            if loss_kwargs:
                lgbm_kwargs.update(loss_kwargs)
                _applied_loss = loss_kwargs.get('objective')
                logger.info(
                    f"  mixed_effects: applying loss_params to LightGBM global: {loss_kwargs}"
                )
            global_model = lgb.LGBMRegressor(**lgbm_kwargs)
            global_model.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        except ImportError:
            global_model = _train_sklearn_fallback(X_train, y_train)
    else:
        global_model = _train_sklearn_fallback(X_train, y_train)

    # Stage 2: Per-key residual adjustments
    global_train_preds = global_model.predict(X_train)
    residuals = y_train - global_train_preds

    from sklearn.ensemble import RandomForestRegressor

    key_adjustments = {}
    n_with_model = 0
    n_with_bias = 0

    for key_val in train_df[key_col].unique():
        key_mask = train_df[key_col].values == key_val
        key_resid = residuals[key_mask]
        key_X = X_train[key_mask]

        if len(key_resid) >= min_key_samples:
            # Train small adjustment model
            adj_model = RandomForestRegressor(
                n_estimators=50, max_depth=3, random_state=42, n_jobs=1,
            )
            adj_model.fit(key_X, key_resid)
            key_adjustments[key_val] = adj_model
            n_with_model += 1
        elif len(key_resid) >= 5:
            # Simple bias (scalar mean residual)
            key_adjustments[key_val] = float(np.mean(key_resid))
            n_with_bias += 1
        # else: no adjustment (global only)

    # Validation
    val_global_preds = global_model.predict(X_val)
    val_adjusted = val_global_preds.copy()
    for i, key_val in enumerate(val_df[key_col].values):
        if key_val in key_adjustments:
            adj = key_adjustments[key_val]
            if hasattr(adj, 'predict'):
                val_adjusted[i] += adj.predict(X_val[i:i + 1])[0]
            else:
                val_adjusted[i] += float(adj)
    val_adjusted = np.maximum(val_adjusted, 0)

    val_wape = _compute_wape(y_val, val_adjusted)
    val_wape_global_only = _compute_wape(y_val, np.maximum(val_global_preds, 0))

    wrapper = HierarchicalModelWrapper(
        model_dict={
            'global_model': global_model,
            'key_adjustments': key_adjustments,
            'type': 'mixed_effects',
        },
        model_type='mixed_effects',
    )

    metadata = {
        'model_type': 'mixed_effects',
        'model_family': model_family,
        'n_keys_with_model': n_with_model,
        'n_keys_with_bias': n_with_bias,
        'n_keys_global_only': train_df[key_col].nunique() - n_with_model - n_with_bias,
        'val_wape': val_wape,
        'val_wape_global_only': val_wape_global_only,
        'improvement_from_adjustments': val_wape_global_only - val_wape,
        'applied_loss': _applied_loss,
    }

    logger.info(f"Mixed-Effects model: val_wape={val_wape:.4f} "
                f"(global_only={val_wape_global_only:.4f}, "
                f"improvement={metadata['improvement_from_adjustments']:.4f})")
    logger.info(f"  {n_with_model} keys with adjustment model, "
                f"{n_with_bias} with bias, "
                f"{metadata['n_keys_global_only']} global-only")

    return wrapper, metadata


# =============================================================================
# 3. MULTI-LEVEL ENSEMBLE
# =============================================================================

def train_multi_level_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    key_col: str,
    hierarchy_col: str,
    model_family: str = 'lightgbm',
    min_key_samples: int = 50,
    min_group_samples: int = 100,
    loss_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 30,
    **kwargs,
) -> Tuple[HierarchicalModelWrapper, Dict[str, Any]]:
    """
    Train Multi-Level Ensemble: global + hierarchy-group + key-level models.

    Models trained at three levels:
    1. Global: single model on all data
    2. Hierarchy group: one model per group (e.g., per APG)
    3. Key-level: individual models for keys with sufficient data

    Weights are data-adaptive: more key data → more weight to key model.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Training and validation data
    feature_cols : List[str]
        Feature columns
    target_col : str
        Target column
    key_col : str
        Key column
    hierarchy_col : str
        Hierarchy grouping column (e.g., 'APG')
    model_family : str
        Base model family
    min_key_samples : int
        Minimum samples to train key-level model
    min_group_samples : int
        Minimum samples to train group-level model

    Returns
    -------
    Tuple[HierarchicalModelWrapper, Dict[str, Any]]
        (wrapped model, metadata)
    """
    logger.info(f"Training Multi-Level Ensemble on {len(train_df)} rows, "
                f"{train_df[key_col].nunique()} keys, "
                f"{train_df[hierarchy_col].nunique()} {hierarchy_col} groups")

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df[target_col].fillna(0).values

    # Tweedie / Poisson / Gamma require non-negative target
    if _loss_needs_nonneg_target(loss_params):
        y_train = np.clip(y_train, 0, None)
        y_val = np.clip(y_val, 0, None)

    # Level 1: Global model
    logger.info("  Training global model...")
    global_model = _train_model(
        X_train, y_train, X_val, y_val, model_family,
        loss_params=loss_params, early_stopping_rounds=early_stopping_rounds,
    )

    # Level 2: Hierarchy group models
    logger.info("  Training hierarchy group models...")
    hier_models = {}
    key_to_group = dict(zip(train_df[key_col], train_df[hierarchy_col]))

    for group_val, group_df in train_df.groupby(hierarchy_col):
        if len(group_df) < min_group_samples:
            continue
        group_X = group_df[feature_cols].fillna(0).values
        group_y = group_df[target_col].fillna(0).values
        if _loss_needs_nonneg_target(loss_params):
            group_y = np.clip(group_y, 0, None)

        # Validation subset for this group
        val_group = val_df[val_df[hierarchy_col] == group_val]
        if len(val_group) < 5:
            val_gX, val_gy = X_val[:10], y_val[:10]  # Use global val as fallback
        else:
            val_gX = val_group[feature_cols].fillna(0).values
            val_gy = val_group[target_col].fillna(0).values
            if _loss_needs_nonneg_target(loss_params):
                val_gy = np.clip(val_gy, 0, None)

        hier_models[group_val] = _train_model(
            group_X, group_y, val_gX, val_gy, model_family,
            loss_params=loss_params, early_stopping_rounds=early_stopping_rounds,
        )

    logger.info(f"  Trained {len(hier_models)} group models")

    # Level 3: Key-level models
    logger.info("  Training key-level models...")
    key_models = {}
    max_key_models = 200  # Cap for efficiency

    key_counts = train_df[key_col].value_counts()
    eligible_keys = key_counts[key_counts >= min_key_samples].index[:max_key_models]

    for key_val in eligible_keys:
        key_mask = train_df[key_col] == key_val
        key_X = train_df.loc[key_mask, feature_cols].fillna(0).values
        key_y = train_df.loc[key_mask, target_col].fillna(0).values
        if _loss_needs_nonneg_target(loss_params):
            key_y = np.clip(key_y, 0, None)

        val_key = val_df[val_df[key_col] == key_val]
        if len(val_key) < 3:
            continue
        val_kX = val_key[feature_cols].fillna(0).values
        val_ky = val_key[target_col].fillna(0).values
        if _loss_needs_nonneg_target(loss_params):
            val_ky = np.clip(val_ky, 0, None)

        key_models[key_val] = _train_model(
            key_X, key_y, val_kX, val_ky, model_family,
            loss_params=loss_params, early_stopping_rounds=early_stopping_rounds,
        )

    logger.info(f"  Trained {len(key_models)} key-level models")

    # Compute data-adaptive weights per key
    weights = {}
    for key_val in train_df[key_col].unique():
        n_key = int((train_df[key_col] == key_val).sum())
        group = key_to_group.get(key_val)
        n_group = int((train_df[hierarchy_col] == group).sum()) if group else 0

        # Shrinkage weights: more data → more weight to granular level
        w_key = min(n_key / 200.0, 0.5) if key_val in key_models else 0.0
        w_hier = min(n_group / 1000.0, 0.3) if group in hier_models else 0.0
        w_global = max(0.1, 1.0 - w_key - w_hier)

        # Normalize
        w_total = w_global + w_hier + w_key
        weights[key_val] = (w_global / w_total, w_hier / w_total, w_key / w_total)

    # Default weights for new keys
    default_weights = (0.5, 0.3, 0.2)

    # Validation
    wrapper = HierarchicalModelWrapper(
        model_dict={
            'global_model': global_model,
            'hier_models': hier_models,
            'key_models': key_models,
            'weights': weights,
            'key_to_group': key_to_group,
            'default_weights': default_weights,
            'type': 'multi_level_ensemble',
        },
        model_type='multi_level_ensemble',
    )

    val_preds = wrapper.predict(X_val, keys=val_df[key_col].values)
    val_wape = _compute_wape(y_val, val_preds)

    # Also compute global-only WAPE for comparison
    val_global_only = np.maximum(global_model.predict(X_val), 0)
    val_wape_global = _compute_wape(y_val, val_global_only)

    metadata = {
        'model_type': 'multi_level_ensemble',
        'model_family': model_family,
        'n_hier_models': len(hier_models),
        'n_key_models': len(key_models),
        'val_wape': val_wape,
        'val_wape_global_only': val_wape_global,
        'improvement': val_wape_global - val_wape,
        'applied_loss': (loss_params or {}).get('objective'),
    }

    logger.info(f"Multi-Level Ensemble: val_wape={val_wape:.4f} "
                f"(global_only={val_wape_global:.4f}, "
                f"improvement={metadata['improvement']:.4f})")

    return wrapper, metadata


# =============================================================================
# HELPERS
# =============================================================================

def _train_model(
    X_train, y_train, X_val, y_val,
    model_family: str = 'lightgbm',
    loss_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 30,
):
    """Train a single model using the specified family.

    When ``loss_params`` is provided (e.g. Tweedie for lumpy demand), the
    booster's objective / auxiliary params are overridden accordingly.
    Used by train_multi_level_ensemble for global / hierarchy-group /
    key-level base learners — all three tiers now honour the strategy.
    """
    if model_family == 'lightgbm':
        try:
            import lightgbm as lgb
            lgbm_kwargs = dict(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, verbosity=-1,
            )
            loss_kwargs = _lgbm_kwargs_from_loss(loss_params)
            if loss_kwargs:
                lgbm_kwargs.update(loss_kwargs)
            model = lgb.LGBMRegressor(**lgbm_kwargs)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            return model
        except ImportError:
            pass

    if model_family == 'xgboost':
        try:
            import xgboost as xgb
            xgb_kwargs = dict(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, verbosity=0,
            )
            loss_kwargs = _xgb_kwargs_from_loss(loss_params)
            if loss_kwargs:
                xgb_kwargs.update(loss_kwargs)
            model = xgb.XGBRegressor(**xgb_kwargs)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return model
        except ImportError:
            pass

    return _train_sklearn_fallback(X_train, y_train)


def _train_sklearn_fallback(X_train, y_train):
    """Fallback to sklearn RandomForest if LightGBM/XGBoost not available."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=1,  # n_jobs=1 for pickle serialization
    )
    model.fit(X_train, y_train)
    return model


def _compute_wape(actuals, predictions):
    """Compute Weighted Absolute Percentage Error."""
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    total_actual = np.sum(np.abs(actuals))
    if total_actual < 1e-10:
        return 1.0
    return float(np.sum(np.abs(actuals - predictions)) / total_actual)
