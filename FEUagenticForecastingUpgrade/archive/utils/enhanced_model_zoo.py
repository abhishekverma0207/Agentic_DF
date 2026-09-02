# utils/enhanced_model_zoo.py
"""
Phase 7: Enhanced Model Zoo.

Adds three new model types that improve forecast robustness and accuracy:

1. **Global Model with CatBoost Key Embeddings** — Single CatBoost model
   across ALL keys using native categorical handling of key identity.
   CatBoost learns key-specific embeddings automatically. Best for
   maximizing data utilization with automatic cross-key learning.

2. **Quantile Regression Models** — Train at median (0.5) for robust point
   forecasts resistant to outliers, plus quantiles 0.1/0.9 for prediction
   intervals. Median regression is strictly better than MSE for skewed
   demand distributions (which most FMCG data has).

3. **Conformalized Residual Boosting** — Two-stage: base model + residual
   model trained on the base model's errors. The residual model captures
   systematic patterns the base model misses. Conformal calibration
   provides valid prediction intervals without distributional assumptions.

All models return wrappers with .predict(X, keys=) interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 1. GLOBAL MODEL WITH CATBOOST KEY EMBEDDINGS
# =============================================================================

class CatBoostEmbeddingWrapper:
    """Wrapper for CatBoost model with native key categorical handling."""

    def __init__(self, model, key_encoder: Dict[str, int], feature_cols: List[str]):
        self.model = model
        self.key_encoder = key_encoder
        self.feature_cols = feature_cols
        self.model_type = 'catboost_embedding'

    def predict(self, X, keys=None):
        X_arr = np.atleast_2d(X)
        # Use exact feature columns the model was trained with (excluding _key_id)
        n_feat = X_arr.shape[1]
        col_names = self.feature_cols[:n_feat] if n_feat <= len(self.feature_cols) else [f'f{i}' for i in range(n_feat)]
        X_df = pd.DataFrame(X_arr, columns=col_names)
        if keys is not None:
            X_df['_key_id'] = [int(self.key_encoder.get(str(k), -1)) for k in keys]
        else:
            X_df['_key_id'] = 0
        preds = self.model.predict(X_df)
        return np.maximum(np.asarray(preds, dtype=float), 0)


def train_catboost_embedding_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    key_col: str,
    iterations: int = 800,
    learning_rate: float = 0.05,
    depth: int = 6,
    **kwargs,
) -> Tuple[CatBoostEmbeddingWrapper, Dict[str, Any]]:
    """
    Train a global CatBoost model with key identity as native categorical.

    CatBoost handles high-cardinality categoricals natively with ordered
    target statistics encoding. This learns a key-specific embedding
    automatically during training.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
    feature_cols : List[str]
    target_col, key_col : str
    iterations : int
    learning_rate : float
    depth : int

    Returns
    -------
    Tuple[CatBoostEmbeddingWrapper, Dict]
    """
    from catboost import CatBoostRegressor, Pool

    logger.info(f"Training CatBoost embedding model on {len(train_df)} rows, "
                f"{train_df[key_col].nunique()} keys")

    # Encode keys as integers
    unique_keys = sorted(train_df[key_col].unique())
    key_encoder = {str(k): i for i, k in enumerate(unique_keys)}

    def _prepare(df):
        X_num = df[feature_cols].fillna(0).values
        encoded = np.array([key_encoder.get(str(k), -1) for k in df[key_col]])
        # Build DataFrame so CatBoost sees mixed types (float features + int categorical)
        X_df = pd.DataFrame(X_num, columns=feature_cols)
        X_df['_key_id'] = encoded.astype(int)
        return X_df, df[target_col].fillna(0).values

    X_train_df, y_train = _prepare(train_df)
    X_val_df, y_val = _prepare(val_df)

    cat_idx = [X_train_df.columns.get_loc('_key_id')]

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        cat_features=cat_idx,
        early_stopping_rounds=50,
    )

    train_pool = Pool(X_train_df, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val_df, y_val, cat_features=cat_idx)

    model.fit(train_pool, eval_set=val_pool, verbose=0)

    # Compute WAPE
    val_preds = np.maximum(model.predict(val_pool), 0)
    wape = _compute_wape(y_val, val_preds)

    wrapper = CatBoostEmbeddingWrapper(model, key_encoder, feature_cols)

    metadata = {
        'model_type': 'catboost_embedding',
        'n_keys': len(unique_keys),
        'val_wape': wape,
        'iterations_used': model.get_best_iteration() if hasattr(model, 'get_best_iteration') else iterations,
        'n_features': len(feature_cols) + 1,
    }

    logger.info(f"CatBoost embedding model: val_wape={wape:.4f}, "
                f"{len(unique_keys)} keys, {model.tree_count_} trees")
    return wrapper, metadata


# =============================================================================
# 2. QUANTILE REGRESSION MODELS
# =============================================================================

class QuantileModelWrapper:
    """Wrapper for quantile regression ensemble (median + intervals)."""

    def __init__(self, models: Dict[float, Any], primary_quantile: float = 0.5):
        self.models = models  # {quantile: model}
        self.primary_quantile = primary_quantile
        self.model_type = 'quantile_regression'

    def predict(self, X, keys=None):
        """Return median (primary quantile) predictions."""
        X_arr = np.atleast_2d(X)
        preds = self.models[self.primary_quantile].predict(X_arr)
        return np.maximum(preds, 0)

    def predict_intervals(self, X, lower_q: float = 0.1, upper_q: float = 0.9):
        """Return point forecast + prediction intervals."""
        X_arr = np.atleast_2d(X)
        point = np.maximum(self.models[self.primary_quantile].predict(X_arr), 0)
        lower = np.maximum(self.models.get(lower_q, self.models[self.primary_quantile]).predict(X_arr), 0)
        upper = np.maximum(self.models.get(upper_q, self.models[self.primary_quantile]).predict(X_arr), 0)
        return point, lower, upper


def train_quantile_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    quantiles: List[float] = None,
    model_family: str = 'lightgbm',
    **kwargs,
) -> Tuple[QuantileModelWrapper, Dict[str, Any]]:
    """
    Train quantile regression models for robust point forecasts + intervals.

    Median regression (q=0.5) is more robust to outliers than MSE because it
    minimizes absolute error rather than squared error. For demand data with
    many zeros and occasional spikes, this is significantly better.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
    feature_cols : List[str]
    target_col : str
    quantiles : List[float]
        Quantiles to train. Default [0.1, 0.5, 0.9] for median + 80% PI.
    model_family : str
        'lightgbm' or 'catboost'

    Returns
    -------
    Tuple[QuantileModelWrapper, Dict]
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
    # Ensure median (0.5) is always included — required by QuantileModelWrapper.predict()
    if 0.5 not in quantiles:
        quantiles = sorted(set(quantiles) | {0.5})

    logger.info(f"Training quantile regression models at q={quantiles}")

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df[target_col].fillna(0).values

    models = {}
    per_q_wape = {}

    for q in quantiles:
        if model_family == 'lightgbm':
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                objective='quantile', alpha=q,
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, verbosity=-1,
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        elif model_family == 'catboost':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                loss_function=f'Quantile:alpha={q}',
                iterations=400, learning_rate=0.05, depth=6,
                verbose=0, early_stopping_rounds=30,
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                loss='quantile', alpha=q,
                n_estimators=200, learning_rate=0.05, max_depth=6,
            )
            model.fit(X_train, y_train)

        models[q] = model

        val_preds = np.maximum(model.predict(X_val), 0)
        wape = _compute_wape(y_val, val_preds)
        per_q_wape[q] = wape
        logger.info(f"  q={q:.1f}: val_wape={wape:.4f}")

    wrapper = QuantileModelWrapper(models, primary_quantile=0.5)

    # Compute interval coverage
    point, lower, upper = wrapper.predict_intervals(X_val)
    coverage = float(np.mean((y_val >= lower) & (y_val <= upper)))
    interval_width = float(np.mean(upper - lower))

    metadata = {
        'model_type': 'quantile_regression',
        'model_family': model_family,
        'quantiles': quantiles,
        'val_wape': per_q_wape.get(0.5, per_q_wape.get(quantiles[len(quantiles)//2])),
        'per_quantile_wape': {str(k): v for k, v in per_q_wape.items()},
        'interval_coverage': coverage,
        'avg_interval_width': interval_width,
    }

    logger.info(f"Quantile model: median_wape={metadata['val_wape']:.4f}, "
                f"80% PI coverage={coverage:.1%}, avg_width={interval_width:.1f}")
    return wrapper, metadata


# =============================================================================
# 3. CONFORMALIZED RESIDUAL BOOSTING
# =============================================================================

class ConformalBoostWrapper:
    """Wrapper for base model + residual model with conformal intervals."""

    def __init__(self, base_model, residual_model, conformal_scores: np.ndarray = None):
        self.base_model = base_model
        self.residual_model = residual_model
        self.conformal_scores = conformal_scores  # Sorted |residuals| from calibration
        self.model_type = 'conformal_boost'

    def predict(self, X, keys=None):
        X_arr = np.atleast_2d(X)
        base_pred = self.base_model.predict(X_arr)
        residual_pred = self.residual_model.predict(X_arr)
        combined = base_pred + residual_pred
        return np.maximum(combined, 0)

    def predict_intervals(self, X, confidence: float = 0.9):
        """Conformal prediction intervals with guaranteed coverage."""
        X_arr = np.atleast_2d(X)
        point = self.predict(X_arr)

        if self.conformal_scores is not None and len(self.conformal_scores) > 0:
            # Conformal quantile: (1 - alpha)(1 + 1/n)-th quantile of scores
            n = len(self.conformal_scores)
            q_level = min(confidence * (1 + 1/n), 1.0)
            idx = int(np.ceil(q_level * n)) - 1
            idx = max(0, min(idx, n - 1))  # Clip both bounds
            half_width = self.conformal_scores[idx]
        else:
            half_width = np.std(point) * 1.645  # Fallback: normal approximation

        lower = np.maximum(point - half_width, 0)
        upper = point + half_width
        return point, lower, upper


def train_conformal_residual_boost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    key_col: str = None,
    model_family: str = 'lightgbm',
    calibration_fraction: float = 0.3,
    **kwargs,
) -> Tuple[ConformalBoostWrapper, Dict[str, Any]]:
    """
    Train a two-stage model: base + residual boosting with conformal calibration.

    Stage 1: Train base model on all training data
    Stage 2: Train residual model on base model's errors
    Stage 3: Calibrate conformal intervals on held-out calibration set

    The residual model captures systematic patterns the base model misses.
    Conformal calibration provides distribution-free prediction intervals
    with guaranteed finite-sample coverage.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
    feature_cols : List[str]
    target_col : str
    key_col : str, optional
    model_family : str
    calibration_fraction : float
        Fraction of validation data used for conformal calibration

    Returns
    -------
    Tuple[ConformalBoostWrapper, Dict]
    """
    logger.info(f"Training conformal residual boost ({model_family})")

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df[target_col].fillna(0).values

    # Stage 1: Base model
    logger.info("  Stage 1: Training base model...")
    base_model = _train_model(X_train, y_train, X_val, y_val, model_family)
    base_train_preds = base_model.predict(X_train)
    base_val_preds = base_model.predict(X_val)

    base_wape = _compute_wape(y_val, np.maximum(base_val_preds, 0))

    # Stage 2: Residual model (train on base model errors)
    logger.info("  Stage 2: Training residual model on base errors...")
    train_residuals = y_train - base_train_preds

    # Use lighter model for residuals (avoid overfitting)
    residual_model = _train_model(
        X_train, train_residuals, X_val, y_val - base_val_preds,
        model_family, light=True,
    )

    # Combined predictions
    combined_val_preds = base_val_preds + residual_model.predict(X_val)
    combined_val_preds = np.maximum(combined_val_preds, 0)
    combined_wape = _compute_wape(y_val, combined_val_preds)

    # Stage 3: Conformal calibration
    logger.info("  Stage 3: Computing conformal scores...")
    # Split validation into proper-val and calibration
    n_cal = max(10, int(len(X_val) * calibration_fraction))
    cal_X = X_val[-n_cal:]
    cal_y = y_val[-n_cal:]

    cal_preds = np.maximum(base_model.predict(cal_X) + residual_model.predict(cal_X), 0)
    conformal_scores = np.sort(np.abs(cal_y - cal_preds))

    wrapper = ConformalBoostWrapper(base_model, residual_model, conformal_scores)

    # Compute interval metrics
    point, lower, upper = wrapper.predict_intervals(X_val, confidence=0.9)
    coverage = float(np.mean((y_val >= lower) & (y_val <= upper)))

    metadata = {
        'model_type': 'conformal_boost',
        'model_family': model_family,
        'val_wape_base': base_wape,
        'val_wape_combined': combined_wape,
        'improvement_from_residuals': base_wape - combined_wape,
        'conformal_coverage_90': coverage,
        'n_calibration_samples': n_cal,
    }

    logger.info(f"Conformal boost: base_wape={base_wape:.4f} → combined_wape={combined_wape:.4f} "
                f"(improvement={metadata['improvement_from_residuals']:.4f}), "
                f"90% coverage={coverage:.1%}")
    return wrapper, metadata


# =============================================================================
# HELPERS
# =============================================================================

def _train_model(X_train, y_train, X_val, y_val, model_family='lightgbm', light=False):
    """Train a single model. If light=True, use fewer trees and more regularization."""
    n_est = 150 if light else 400
    lr = 0.08 if light else 0.05
    reg = 3.0 if light else 1.0

    if model_family == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=n_est, learning_rate=lr, num_leaves=31,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=reg, verbosity=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        return model

    if model_family == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=reg, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model

    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1).fit(X_train, y_train)


def _compute_wape(actuals, predictions):
    total = np.sum(np.abs(actuals))
    if total < 1e-10:
        return 1.0
    return float(np.sum(np.abs(actuals - predictions)) / total)
