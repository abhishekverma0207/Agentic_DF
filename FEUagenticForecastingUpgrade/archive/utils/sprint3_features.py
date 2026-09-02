# utils/sprint3_features.py
"""
Sprint 3: State-of-the-Art Forecasting Enhancements.

B2: Hierarchy feature updater for recursive forecasting
B3: Error correction mechanism (exponential smoothing of residuals)
B7: Enriched MetaFeatures for intelligent model selection
B8: Promo Adstock / decay curve feature engineering
B9: Feature importance-based pruning after training
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# B2: HIERARCHY FEATURE UPDATE DURING RECURSIVE FORECASTING
# =============================================================================

class HierarchyFeatureUpdater:
    """
    Updates hierarchy-level features (group_total, share, rank) during
    recursive multi-step forecasting.

    Without this, hierarchy features are frozen at their initial values,
    meaning at step 2+ the model sees stale group totals.

    This updater tracks predictions per key and recomputes group-level
    aggregates at each step.
    """

    def __init__(self, hierarchy_cols: List[str], key_to_group: Dict[str, Dict[str, str]]):
        """
        Parameters
        ----------
        hierarchy_cols : List[str]
            Hierarchy column names (e.g., ['SubCategory', 'VolumeSegment'])
        key_to_group : Dict[str, Dict[str, str]]
            {key: {hier_col: group_value}} mapping
        """
        self.hierarchy_cols = hierarchy_cols
        self.key_to_group = key_to_group
        self.step_predictions = {}  # {key: latest_prediction}

    def update_prediction(self, key: str, prediction: float):
        """Record a key's prediction for this step."""
        self.step_predictions[key] = prediction

    def get_updated_features(self, key: str, feature_names: List[str], feature_values: np.ndarray) -> np.ndarray:
        """
        Update hierarchy features in the feature vector based on accumulated predictions.

        Recomputes group_total and share using predictions from ALL keys in the group.
        """
        updated = feature_values.copy()

        for hier_col in self.hierarchy_cols:
            safe = hier_col.lower().replace(' ', '_')[:15]
            group_val = self.key_to_group.get(key, {}).get(hier_col)
            if group_val is None:
                continue

            # Find keys in same group
            group_keys = [k for k, mapping in self.key_to_group.items()
                          if mapping.get(hier_col) == group_val and k in self.step_predictions]

            if not group_keys:
                continue

            # Compute group total from predictions
            group_total = sum(self.step_predictions[k] for k in group_keys)
            key_pred = self.step_predictions.get(key, 0)

            # Update feature values
            for i, fname in enumerate(feature_names):
                fname_lower = fname.lower()
                if f'{safe}_group_total' in fname_lower:
                    updated[i] = group_total
                elif f'{safe}_group_mean' in fname_lower:
                    updated[i] = group_total / max(len(group_keys), 1)
                elif f'{safe}_share' in fname_lower:
                    updated[i] = key_pred / max(group_total, 1e-10)

        return updated

    def reset_step(self):
        """Reset predictions for next forecast step."""
        self.step_predictions.clear()


# =============================================================================
# B3: ERROR CORRECTION IN RECURSIVE FORECASTING
# =============================================================================

class RecursiveErrorCorrector:
    """
    Corrects accumulated prediction error during recursive forecasting.

    Uses exponential smoothing of recent forecast errors to adjust
    future predictions. If the model consistently over-predicts,
    the correction term becomes negative and pulls predictions down.

    error_t = actual_t - predicted_t  (only known for in-sample)
    correction_t = alpha * error_(t-1) + (1-alpha) * correction_(t-1)
    adjusted_pred_t = pred_t + correction_t

    For out-of-sample steps where error is unknown, the correction term
    decays exponentially via (1-alpha) factor.
    """

    def __init__(self, alpha: float = 0.3, initial_correction: float = 0.0):
        """
        Parameters
        ----------
        alpha : float
            Smoothing parameter (0-1). Higher = more reactive to recent errors.
        initial_correction : float
            Starting correction term.
        """
        self.alpha = alpha
        self.correction = initial_correction
        self.step_count = 0

    def update_with_error(self, error: float):
        """Update correction term with observed error."""
        self.correction = self.alpha * error + (1 - self.alpha) * self.correction

    def get_correction(self) -> float:
        """Get current correction term (decays each step without new error)."""
        # Decay correction for out-of-sample steps
        decay = (1 - self.alpha) ** self.step_count
        self.step_count += 1
        return self.correction * decay

    def adjust_prediction(self, raw_prediction: float) -> float:
        """Apply error correction to raw prediction."""
        corrected = raw_prediction + self.get_correction()
        return max(corrected, 0)  # Ensure non-negative

    @classmethod
    def from_training_residuals(cls, residuals: np.ndarray, alpha: float = 0.3) -> 'RecursiveErrorCorrector':
        """Create corrector initialised from training residuals."""
        if len(residuals) == 0:
            return cls(alpha=alpha)

        # Use exponentially weighted mean of recent residuals as initial correction
        weights = np.array([(1 - alpha) ** i for i in range(len(residuals))])[::-1]
        weights /= weights.sum()
        initial = float(np.sum(weights * residuals[-len(weights):]))

        return cls(alpha=alpha, initial_correction=initial)


# =============================================================================
# B7: ENRICHED META-FEATURES FOR MODEL SELECTION
# =============================================================================

def build_enriched_meta_features(
    y: np.ndarray,
    n_features: int = 0,
    has_external_features: bool = False,
    feature_quality_score: float = 0.5,
    time_format: str = 'year_week',
    # Sprint 3: Enriched fields
    hierarchy_depth: int = 0,
    hierarchy_group_size: float = 0.0,
    seasonality_shape: str = 'unknown',
    external_response_strength: float = 0.0,
    trend_direction: float = 0.0,
    regime_volatility: float = 0.0,
) -> Dict[str, Any]:
    """
    Build enriched meta-features that include segmentation context.

    Returns a dict (not MetaFeatures dataclass) with extra fields
    that can be used by an enhanced scoring function.
    """
    from utils.model_selection_intelligence import build_meta_features

    # Get base meta-features
    base = build_meta_features(y, n_features, has_external_features, feature_quality_score, time_format)

    # Enrich with Sprint 3 fields
    enriched = {
        'base': base,
        'demand_pattern': base.demand_pattern,
        'cv': base.cv,
        'zero_fraction': base.zero_fraction,
        'n_observations': base.n_observations,
        'has_external': has_external_features,
        # Sprint 3 enrichments
        'hierarchy_depth': hierarchy_depth,
        'hierarchy_group_size': hierarchy_group_size,
        'seasonality_shape': seasonality_shape,
        'external_response_strength': external_response_strength,
        'trend_direction': trend_direction,
        'regime_volatility': regime_volatility,
    }

    return enriched


def score_model_with_enrichment(
    model_name: str,
    enriched_meta: Dict[str, Any],
    base_score: float,
) -> float:
    """
    Adjust model score based on enriched meta-features.

    Applied AFTER rank_models_by_pattern_fit() to re-score models
    using hierarchy, seasonality, and external response information.
    """
    score = base_score

    # Hierarchy awareness: boost global/hierarchical models if hierarchy is deep
    hier_depth = enriched_meta.get('hierarchy_depth', 0)
    if hier_depth >= 2 and model_name in ('global_local', 'mixed_effects', 'multi_level_ensemble'):
        score *= 1.3  # 30% boost for hierarchical models when hierarchy exists

    # Strong external response: boost tree models that capture non-linear effects
    ext_strength = enriched_meta.get('external_response_strength', 0)
    if ext_strength > 0.3 and model_name in ('xgboost', 'lightgbm', 'catboost'):
        score *= 1.15  # 15% boost for tree models with strong external signal

    # High zero fraction: boost zero-handling models
    zf = enriched_meta.get('zero_fraction', 0)
    if zf > 0.5 and model_name in ('zero_inflated', 'hurdle_model', 'tweedie'):
        score *= (1 + zf * 0.5)  # Up to 50% boost

    # High regime volatility: boost robust models
    regime_vol = enriched_meta.get('regime_volatility', 0)
    if regime_vol > 0.5 and model_name in ('quantile_regression', 'conformal_boost'):
        score *= 1.2  # 20% boost for robust models under volatility

    # Strong seasonality: boost models that handle cyclical patterns
    season_shape = enriched_meta.get('seasonality_shape', 'unknown')
    if season_shape in ('strong_peak', 'multi_peak') and model_name in ('lightgbm', 'xgboost'):
        score *= 1.1  # 10% boost

    return score


# =============================================================================
# B8: PROMO ADSTOCK / DECAY CURVE FEATURES
# =============================================================================

def create_adstock_features(
    df: pd.DataFrame,
    key_cols: List[str],
    promo_cols: List[str],
    decay_rates: List[float] = None,
    max_lag: int = 8,
) -> pd.DataFrame:
    """
    Create Adstock (geometric decay) features for promo/price columns.

    Standard lag features treat each period independently. Adstock captures
    the lingering effect of promotions:

    adstock_t = promo_t + decay * promo_(t-1) + decay² * promo_(t-2) + ...

    This models:
    - Pre-promo dip (customers wait for promo)
    - Post-promo payback (demand drops after promo ends)
    - Diminishing returns (repeated promos have less impact)

    Parameters
    ----------
    df : pd.DataFrame
        Source data sorted by key + date
    key_cols : List[str]
        Key columns
    promo_cols : List[str]
        Promo/price columns to create Adstock features for
    decay_rates : List[float]
        Geometric decay rates to try (default [0.3, 0.5, 0.7, 0.9])
    max_lag : int
        Maximum lag for Adstock computation

    Returns
    -------
    pd.DataFrame
        DataFrame with Adstock features added
    """
    if decay_rates is None:
        decay_rates = [0.3, 0.5, 0.7, 0.9]

    valid_cols = [c for c in promo_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        return df

    logger.info(f"Creating Adstock features for {len(valid_cols)} promo columns × {len(decay_rates)} decay rates")

    key_col = key_cols[0] if len(key_cols) == 1 else key_cols
    features_added = 0

    for col in valid_cols[:10]:  # Cap at 10 promo columns
        col_short = col[:20]
        for decay in decay_rates:
            adstock_col = f'{col_short}_adstock_{int(decay*10)}'

            def _compute_adstock(series, _decay=decay, _max_lag=max_lag):
                result = series.copy().astype(float)
                for i in range(1, min(_max_lag, len(series))):
                    result += (_decay ** i) * series.shift(i).fillna(0)
                return result

            df[adstock_col] = df.groupby(key_col)[col].transform(_compute_adstock)
            features_added += 1

    # Also create promo saturation indicator: rolling count of promo periods
    for col in valid_cols[:5]:
        col_short = col[:20]
        if df[col].max() > 0:
            # Promo frequency in recent window (saturation proxy)
            df[f'{col_short}_freq_13w'] = df.groupby(key_col)[col].transform(
                lambda x: (x > 0).astype(float).rolling(13, min_periods=1).mean()
            )
            features_added += 1

    logger.info(f"Created {features_added} Adstock/promo features")
    return df


# =============================================================================
# B9: FEATURE IMPORTANCE-BASED PRUNING
# =============================================================================

def prune_features_by_importance(
    model: Any,
    feature_cols: List[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    min_importance_pct: float = 0.5,
    max_features: int = 100,
    method: str = 'model',
) -> Tuple[List[str], Dict[str, float]]:
    """
    Prune low-importance features after training.

    Removes features that contribute less than min_importance_pct of total
    importance. This reduces noise and speeds up inference.

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_cols : List[str]
        Feature column names
    X_val : np.ndarray
        Validation data
    y_val : np.ndarray
        Validation target
    min_importance_pct : float
        Minimum importance as % of total (features below this are pruned)
    max_features : int
        Maximum number of features to keep
    method : str
        'model' (use model.feature_importances_) or 'permutation'

    Returns
    -------
    Tuple[List[str], Dict[str, float]]
        (pruned_feature_cols, importance_dict)
    """
    importance_dict = {}

    if method == 'model' and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if len(importances) != len(feature_cols):
            logger.warning(f"Importance length mismatch: {len(importances)} vs {len(feature_cols)} features")
            return feature_cols, {}

        total = sum(importances)
        if total < 1e-10:
            return feature_cols, {}

        for i, col in enumerate(feature_cols):
            importance_dict[col] = float(importances[i] / total * 100)

    elif method == 'permutation':
        # Permutation importance — model-agnostic but slower
        baseline_preds = np.maximum(model.predict(X_val), 0)
        total_actual = np.sum(np.abs(y_val))
        baseline_wape = float(np.sum(np.abs(y_val - baseline_preds)) / total_actual) if total_actual > 1e-10 else 1.0

        for i, col in enumerate(feature_cols):
            X_permuted = X_val.copy()
            np.random.shuffle(X_permuted[:, i])
            perm_preds = np.maximum(model.predict(X_permuted), 0)
            perm_wape = float(np.sum(np.abs(y_val - perm_preds)) / total_actual) if total_actual > 1e-10 else 1.0
            importance_dict[col] = max(0, (perm_wape - baseline_wape) * 100)
    else:
        return feature_cols, {}

    if not importance_dict:
        return feature_cols, {}

    # Sort by importance (descending)
    sorted_features = sorted(importance_dict.items(), key=lambda x: -x[1])

    # Cumulative importance approach: keep features up to 95% cumulative importance
    # This is more robust than a fixed threshold — adapts to importance distribution
    CUMULATIVE_TARGET = 95.0  # Keep features covering 95% of total importance
    MIN_FEATURES = 15  # Always keep at least this many

    pruned = []
    cumulative = 0.0
    for col, imp in sorted_features:
        if cumulative < CUMULATIVE_TARGET or len(pruned) < MIN_FEATURES:
            pruned.append(col)
            cumulative += imp
        elif len(pruned) < max_features and imp >= min_importance_pct:
            pruned.append(col)
            cumulative += imp
        else:
            break

    n_pruned = len(feature_cols) - len(pruned)
    if n_pruned > 0:
        logger.info(f"Feature pruning: {len(feature_cols)} → {len(pruned)} features "
                     f"(removed {n_pruned} with importance < {min_importance_pct}%)")
        # Log top 10 and bottom 5
        logger.info(f"  Top 10: {[f'{c}:{v:.1f}%' for c, v in sorted_features[:10]]}")
        logger.info(f"  Pruned: {[f'{c}:{v:.2f}%' for c, v in sorted_features[-min(5, n_pruned):]]}")
    else:
        logger.info(f"Feature pruning: all {len(feature_cols)} features above {min_importance_pct}% threshold")

    return pruned, importance_dict
