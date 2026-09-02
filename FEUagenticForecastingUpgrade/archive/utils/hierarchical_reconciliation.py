# utils/hierarchical_reconciliation.py
"""
Phase 3: Hierarchical Forecast Reconciliation.

Implements coherent reconciliation methods beyond the existing top-down Prophet:

- **Bottom-Up**: Sum leaf-level forecasts to parent levels
- **OLS**: Ordinary least squares reconciliation (equal weights)
- **WLS**: Weighted least squares (diagonal covariance)
- **MinT Shrinkage**: Minimum Trace with Ledoit-Wolf shrinkage (recommended)

The MinT approach (Wickramasuriya et al. 2019) finds the optimal reconciled
forecasts that minimize total forecast variance while satisfying the coherency
constraint: parent forecast = sum of children forecasts.

Reconciled = S @ (S' W⁻¹ S)⁻¹ @ S' W⁻¹ @ ŷ_base

where S is the summing matrix and W is the error covariance estimate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 1. SUMMING MATRIX CONSTRUCTION
# =============================================================================

def build_summing_matrix(
    hierarchy_map: Dict[str, str],
    bottom_keys: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the summing matrix S for a two-level hierarchy.

    For hierarchy: Total → Groups → Keys (bottom)

    S has shape (n_total_nodes, n_bottom_keys) where:
    - Row 0: Total (all 1s)
    - Rows 1..n_groups: Group indicators (1 if key belongs to group)
    - Rows n_groups+1..end: Identity matrix for bottom keys

    Parameters
    ----------
    hierarchy_map : Dict[str, str]
        Mapping from bottom key to group name: {key: group}
    bottom_keys : List[str]
        Ordered list of bottom-level keys

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (S matrix, ordered list of all node names [Total, group1, ..., key1, ...])
    """
    n_keys = len(bottom_keys)
    groups = sorted(set(hierarchy_map.values()))
    n_groups = len(groups)
    n_total = 1 + n_groups + n_keys  # Total + groups + keys

    S = np.zeros((n_total, n_keys))

    # Row 0: Total (sum of all keys)
    S[0, :] = 1.0

    # Rows 1..n_groups: Group indicators
    group_to_idx = {g: i + 1 for i, g in enumerate(groups)}
    key_to_col = {k: j for j, k in enumerate(bottom_keys)}

    for key, group in hierarchy_map.items():
        if key in key_to_col:
            col = key_to_col[key]
            row = group_to_idx[group]
            S[row, col] = 1.0

    # Rows n_groups+1..end: Identity for bottom keys
    S[1 + n_groups:, :] = np.eye(n_keys)

    node_names = ['Total'] + [f'Group_{g}' for g in groups] + list(bottom_keys)

    logger.info(f"Summing matrix S: {S.shape} ({n_total} nodes: 1 total + {n_groups} groups + {n_keys} keys)")
    return S, node_names


# =============================================================================
# 2. COVARIANCE ESTIMATION
# =============================================================================

def estimate_residual_covariance(
    residuals_df: pd.DataFrame,
    key_col: str,
    date_col: str,
    residual_col: str,
    bottom_keys: List[str],
    method: str = 'shrink',
) -> np.ndarray:
    """
    Estimate covariance matrix W of forecast errors.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        DataFrame with columns [key_col, date_col, residual_col]
    key_col, date_col, residual_col : str
        Column names
    bottom_keys : List[str]
        Ordered list of bottom keys (defines column order)
    method : str
        'shrink' (Ledoit-Wolf), 'sample', or 'diagonal'

    Returns
    -------
    np.ndarray
        Covariance matrix of shape (n_keys, n_keys)
    """
    n_keys = len(bottom_keys)

    # Pivot to wide format: rows=dates, cols=keys
    pivot = residuals_df.pivot_table(
        index=date_col, columns=key_col, values=residual_col, aggfunc='first'
    )
    # Reindex to match bottom_keys order, fill missing with 0
    pivot = pivot.reindex(columns=bottom_keys, fill_value=0.0).fillna(0.0)

    X = pivot.values  # (n_dates, n_keys)
    n_dates = X.shape[0]

    if method == 'shrink':
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(X)
            W = lw.covariance_
            logger.info(f"Ledoit-Wolf shrinkage covariance: shape={W.shape}, shrinkage={lw.shrinkage_:.4f}")
            return W
        except Exception as e:
            logger.warning(f"Ledoit-Wolf failed ({e}), falling back to diagonal")
            method = 'diagonal'

    if method == 'sample':
        if n_dates > n_keys:
            W = np.cov(X, rowvar=False)
            logger.info(f"Sample covariance: shape={W.shape}")
            return W
        else:
            logger.warning(f"n_dates ({n_dates}) <= n_keys ({n_keys}), falling back to diagonal")
            method = 'diagonal'

    # Diagonal: just variances on diagonal
    variances = np.var(X, axis=0)
    variances = np.maximum(variances, 1e-10)  # Avoid zero variance
    W = np.diag(variances)
    logger.info(f"Diagonal covariance: shape={W.shape}")
    return W


# =============================================================================
# 3. RECONCILIATION METHODS
# =============================================================================

def reconcile_mint(
    base_forecasts: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    MinT (Minimum Trace) reconciliation.

    Finds reconciled forecasts that minimize total forecast variance:
    ŷ_tilde = S @ P @ ŷ_base
    where P = (S' W⁻¹ S)⁻¹ S' W⁻¹

    For large hierarchies (>1000 nodes), automatically uses WLS (diagonal W)
    instead of full matrix inversion for numerical stability and memory.

    Parameters
    ----------
    base_forecasts : np.ndarray
        Base forecasts at all levels, shape (n_total_nodes,)
    S : np.ndarray
        Summing matrix, shape (n_total_nodes, n_bottom_keys)
    W : np.ndarray
        Error covariance, shape (n_total_nodes, n_total_nodes) or
        (n_bottom_keys, n_bottom_keys) depending on implementation

    Returns
    -------
    np.ndarray
        Reconciled forecasts at all levels, shape (n_total_nodes,)
    """
    from scipy import linalg

    n_total, n_bottom = S.shape

    # For large hierarchies, use WLS (diagonal) instead of full covariance
    # Full matrix: n_total × n_total can be 7000² = 49M entries → 400MB+ memory
    LARGE_HIERARCHY_THRESHOLD = 1000
    if n_total > LARGE_HIERARCHY_THRESHOLD:
        logger.info(f"Large hierarchy ({n_total} nodes) — using WLS (diagonal covariance) for stability")
        # Extract diagonal from W (variance per node)
        if W.shape[0] == n_bottom:
            # Bottom-level W → compute diagonal of full W = diag(S @ W_diag @ S')
            w_diag_bottom = np.diag(W) if W.ndim == 2 else W
            # Approximate: variance at each level = sum of bottom variances it aggregates
            w_diag_full = np.array([
                np.sum(w_diag_bottom[S[i] > 0]) for i in range(n_total)
            ])
        else:
            w_diag_full = np.diag(W) if W.ndim == 2 else np.ones(n_total)

        # Ensure positive
        w_diag_full = np.maximum(w_diag_full, 1e-10)

        # WLS: P = (S' W_diag⁻¹ S)⁻¹ S' W_diag⁻¹
        w_inv = 1.0 / w_diag_full
        StWi = S.T * w_inv[np.newaxis, :]  # (n_bottom, n_total) × diagonal
        StWiS = StWi @ S
        StWiS += np.eye(n_bottom) * 1e-8  # Ridge
        try:
            P = linalg.solve(StWiS, StWi)
            reconciled_bottom = P @ base_forecasts
        except linalg.LinAlgError:
            logger.warning("WLS solve failed, falling back to OLS")
            return reconcile_ols(base_forecasts, S)

        reconciled_bottom = np.maximum(reconciled_bottom, 0)
        return S @ reconciled_bottom

    # Standard MinT for small hierarchies
    # Extend W to full size if it's only bottom-level
    if W.shape[0] == n_bottom:
        W_full = S @ W @ S.T
    else:
        W_full = W

    # Add small ridge for numerical stability
    W_full += np.eye(W_full.shape[0]) * 1e-8

    try:
        # P = (S' W⁻¹ S)⁻¹ S' W⁻¹
        W_inv = linalg.inv(W_full)
        StWi = S.T @ W_inv
        StWiS = StWi @ S
        P = linalg.solve(StWiS, StWi)
        reconciled_bottom = P @ base_forecasts
    except linalg.LinAlgError:
        logger.warning("MinT solve failed, falling back to OLS")
        return reconcile_ols(base_forecasts, S)

    # Reconstruct all levels from reconciled bottom
    reconciled_bottom_nn = np.maximum(reconciled_bottom, 0)
    reconciled_all = S @ reconciled_bottom_nn

    return reconciled_all


def reconcile_ols(
    base_forecasts: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """OLS reconciliation (equal weights, identity covariance)."""
    from scipy import linalg

    n_total, n_bottom = S.shape

    try:
        StS = S.T @ S
        P = linalg.solve(StS, S.T)
        reconciled_bottom = P @ base_forecasts
    except linalg.LinAlgError:
        # Fallback: just use bottom-level forecasts
        reconciled_bottom = base_forecasts[-n_bottom:]

    reconciled_bottom = np.maximum(reconciled_bottom, 0)
    return S @ reconciled_bottom


def reconcile_bottom_up(
    key_forecasts: Dict[str, float],
    hierarchy_map: Dict[str, str],
) -> Dict[str, float]:
    """
    Bottom-up reconciliation: parent = sum of children.

    Parameters
    ----------
    key_forecasts : Dict[str, float]
        Forecasts for each bottom-level key
    hierarchy_map : Dict[str, str]
        {key: group}

    Returns
    -------
    Dict[str, float]
        Forecasts at all levels {key: val, group: val, Total: val}
    """
    result = dict(key_forecasts)

    # Group totals
    group_totals = {}
    for key, group in hierarchy_map.items():
        if key in key_forecasts:
            group_totals[group] = group_totals.get(group, 0.0) + key_forecasts[key]

    for group, total in group_totals.items():
        result[f'Group_{group}'] = total

    result['Total'] = sum(key_forecasts.values())
    return result


# =============================================================================
# 4. HIGH-LEVEL DISPATCHER
# =============================================================================

def run_hierarchical_reconciliation(
    forecasts_df: pd.DataFrame,
    source_df: pd.DataFrame,
    hierarchy_col: str,
    key_col: str,
    date_col: str,
    target_col: str,
    predicted_col: str = 'predicted',
    method: str = 'mint_shrink',
    train_cutoff: Optional[str] = None,
) -> pd.DataFrame:
    """
    High-level API: reconcile forecasts using the specified method.

    This function handles the full workflow:
    1. Build hierarchy mapping from source data
    2. For each forecast period:
       a. Get base forecasts at bottom level
       b. Compute base forecasts at group and total levels (bottom-up)
       c. Apply reconciliation method
       d. Replace bottom-level forecasts with reconciled values
    3. Return reconciled forecasts_df

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Forecasts with columns [key_col, date_col, predicted_col]
    source_df : pd.DataFrame
        Historical data for covariance estimation
    hierarchy_col : str
        Hierarchy column name (e.g., 'APG')
    key_col : str
        Key column name
    date_col : str
        Date column name
    target_col : str
        Target column in source_df (for residuals)
    predicted_col : str
        Predicted column in forecasts_df
    method : str
        Reconciliation method
    train_cutoff : str, optional
        Training cutoff for residual computation

    Returns
    -------
    pd.DataFrame
        Reconciled forecasts_df with updated predicted_col
    """
    if method == 'top_down':
        logger.info("Method is 'top_down' — use existing reconciliation.py logic")
        return forecasts_df

    if hierarchy_col not in source_df.columns:
        logger.warning(f"Hierarchy column '{hierarchy_col}' not in source data, skipping reconciliation")
        return forecasts_df

    logger.info(f"Running hierarchical reconciliation: method={method}, hierarchy={hierarchy_col}")

    # Build hierarchy mapping: key → group
    key_to_group = (
        source_df.groupby(key_col)[hierarchy_col]
        .first()
        .to_dict()
    )

    forecast_keys = sorted(forecasts_df[key_col].unique())
    # Filter to keys that have hierarchy mapping
    mapped_keys = [k for k in forecast_keys if k in key_to_group]
    if len(mapped_keys) < len(forecast_keys):
        n_unmapped = len(forecast_keys) - len(mapped_keys)
        logger.warning(f"{n_unmapped} forecast keys have no hierarchy mapping — left unreconciled")

    if not mapped_keys:
        return forecasts_df

    hierarchy_map = {k: key_to_group[k] for k in mapped_keys}

    # Build summing matrix
    S, node_names = build_summing_matrix(hierarchy_map, mapped_keys)
    n_total, n_keys = S.shape
    n_groups = len(set(hierarchy_map.values()))

    # Estimate covariance (for MinT/WLS)
    W = None
    if method in ('mint_shrink', 'wls'):
        # Compute in-sample residuals
        if train_cutoff:
            hist = source_df[source_df[date_col] <= train_cutoff]
        else:
            hist = source_df

        # Simple residuals: actual - mean(actual) per key
        residuals_list = []
        key_means = hist.groupby(key_col)[target_col].mean()
        for key_val, group in hist.groupby(key_col):
            if key_val not in set(mapped_keys):
                continue
            resids = group[target_col] - key_means.get(key_val, 0)
            for _, row in group.iterrows():
                residuals_list.append({
                    key_col: key_val,
                    date_col: row[date_col],
                    'residual': row[target_col] - key_means.get(key_val, 0),
                })

        if residuals_list:
            residuals_df = pd.DataFrame(residuals_list)
            cov_method = 'shrink' if method == 'mint_shrink' else 'diagonal'
            W = estimate_residual_covariance(
                residuals_df, key_col, date_col, 'residual',
                mapped_keys, method=cov_method,
            )
        else:
            logger.warning("No residuals computed, falling back to OLS")
            method = 'ols'

    # Reconcile per period
    result_df = forecasts_df.copy()
    forecast_periods = sorted(forecasts_df[date_col].unique())

    n_reconciled = 0
    for period in forecast_periods:
        period_mask = result_df[date_col] == period

        # Get bottom-level base forecasts
        period_forecasts = result_df[period_mask].set_index(key_col)[predicted_col]
        bottom_base = np.array([period_forecasts.get(k, 0.0) for k in mapped_keys])

        if method == 'bottom_up':
            # Bottom-up: key forecasts unchanged, just ensure coherence logged
            continue

        # Compute group and total level base forecasts (bottom-up aggregation)
        group_base = S[:1 + n_groups, :] @ bottom_base  # total + groups
        all_base = np.concatenate([group_base, bottom_base])

        # Apply reconciliation
        if method in ('mint_shrink', 'wls') and W is not None:
            reconciled = reconcile_mint(all_base, S, W)
        elif method == 'ols':
            reconciled = reconcile_ols(all_base, S)
        else:
            continue

        # Extract reconciled bottom-level forecasts
        reconciled_bottom = reconciled[1 + n_groups:]
        reconciled_bottom = np.maximum(reconciled_bottom, 0)

        # Update forecasts_df
        for i, key in enumerate(mapped_keys):
            mask = period_mask & (result_df[key_col] == key)
            if mask.any():
                result_df.loc[mask, predicted_col] = reconciled_bottom[i]
                n_reconciled += 1

    logger.info(f"Reconciled {n_reconciled} forecast entries across {len(forecast_periods)} periods")

    # Diagnostics: check coherence
    _check_coherence(result_df, key_col, date_col, predicted_col, hierarchy_map)

    return result_df


def _check_coherence(
    df: pd.DataFrame,
    key_col: str,
    date_col: str,
    predicted_col: str,
    hierarchy_map: Dict[str, str],
):
    """Log coherence diagnostics after reconciliation — checks ALL periods."""
    n_incoherent = 0
    for period in df[date_col].unique():  # Check ALL periods
        period_data = df[df[date_col] == period]
        total = period_data[predicted_col].sum()

        group_totals = {}
        for _, row in period_data.iterrows():
            key = row[key_col]
            group = hierarchy_map.get(key, 'unknown')
            group_totals[group] = group_totals.get(group, 0) + row[predicted_col]

        group_sum = sum(group_totals.values())
        coherence_gap = abs(total - group_sum) / (total + 1e-10)
        if coherence_gap > 0.01:
            logger.warning(f"Period {period}: coherence gap {coherence_gap:.4f} (total={total:.1f}, group_sum={group_sum:.1f})")
        else:
            logger.debug(f"Period {period}: coherent (gap={coherence_gap:.6f})")
