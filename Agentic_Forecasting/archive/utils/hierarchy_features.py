# utils/hierarchy_features.py
"""
Phase 3: Hierarchy-Level Temporal Features for Demand Forecasting.

Creates time-varying features at hierarchy levels (e.g., Category, Brand/APG)
that capture macro patterns not visible at the individual key level:

- Group total demand (category-level signal, more stable than key-level)
- Key's share of group (market position, evolving over time)
- Group-level trend (macro growth/decline affecting all keys in the group)
- Rank within group (competitive position among peers)

All features are LEAKAGE-FREE: shifted by `forecast_lag` so at prediction
time t, only information from t - forecast_lag and earlier is used.

All features are STATIC during recursive forecasting: computed once from
actuals and NOT updated with predicted values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_hierarchy_temporal_features(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    hierarchy_cols: List[str],
    forecast_lag: int,
    windows: List[int] = None,
    stats: List[str] = None,
    max_hierarchy_cols: int = 2,
) -> pd.DataFrame:
    """
    Create leakage-free hierarchy-level temporal features.

    For each hierarchy column, computes per-(key, date) features that capture
    the key's relationship to its hierarchy group over time.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with key, date, target, and hierarchy columns.
        Must be sorted or sortable by key + date.
    key_cols : List[str]
        Key identifier columns
    date_col : str
        Date/period column
    target_col : str
        Target variable column
    hierarchy_cols : List[str]
        Hierarchy columns to create features for, ordered coarsest to finest.
        E.g., ['Category', 'APG']. Limited to max_hierarchy_cols.
    forecast_lag : int
        Forecast lag for leakage prevention (shift amount)
    windows : List[int], optional
        Rolling windows for trend features. Default [4, 13].
    stats : List[str], optional
        Aggregation stats. Default ['sum', 'mean'].
    max_hierarchy_cols : int
        Maximum hierarchy columns to process (default 2)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with hierarchy features added
    """
    if not hierarchy_cols:
        return df

    if windows is None:
        windows = [4, 13]
    if stats is None:
        stats = ['sum', 'mean']

    # Limit to max_hierarchy_cols to prevent feature explosion
    hier_cols = [c for c in hierarchy_cols if c in df.columns][:max_hierarchy_cols]
    if not hier_cols:
        logger.warning(f"None of hierarchy_cols {hierarchy_cols} found in DataFrame")
        return df

    key_col = key_cols[0] if len(key_cols) == 1 else None
    if key_col is None:
        logger.warning("Hierarchy features require single key column")
        return df

    df = df.sort_values(key_cols + [date_col])
    features_added = []

    for hier_col in hier_cols:
        safe = hier_col.lower().replace(' ', '_')[:15]
        prefix = f"{target_col}_{safe}"

        logger.info(f"Creating hierarchy features for '{hier_col}' "
                     f"({df[hier_col].nunique()} groups)")

        # ─── 1. GROUP AGGREGATES PER PERIOD ──────────────────────────
        # Compute group-level totals and means at each date
        group_agg = (
            df.groupby([hier_col, date_col])[target_col]
            .agg(['sum', 'mean', 'count'])
            .reset_index()
        )
        group_agg.columns = [hier_col, date_col,
                             f'_hier_{safe}_total',
                             f'_hier_{safe}_mean',
                             f'_hier_{safe}_count']

        # Merge back to key-level
        df = df.merge(group_agg, on=[hier_col, date_col], how='left')

        # ─── 2. LAG THE GROUP AGGREGATES (leakage prevention) ────────
        for agg_name in ['total', 'mean']:
            raw_col = f'_hier_{safe}_{agg_name}'
            lagged_col = f'{prefix}_group_{agg_name}_lag'

            # Shift within each key's time series by forecast_lag
            df[lagged_col] = (
                df.groupby(key_col)[raw_col]
                .shift(forecast_lag)
            )
            features_added.append(lagged_col)

        # ─── 3. SHARE OF GROUP (lagged) ─────────────────────────────
        # Key's demand / group total (market share proxy)
        lagged_target = df.groupby(key_col)[target_col].shift(forecast_lag)
        lagged_total = df[f'{prefix}_group_total_lag']

        share_col = f'{prefix}_share_lag'
        df[share_col] = (lagged_target / (lagged_total + 1e-10)).clip(0, 1)
        features_added.append(share_col)

        # ─── 4. RANK WITHIN GROUP (lagged) ──────────────────────────
        # Percentile rank of key within its group at each date
        rank_col = f'{prefix}_rank_in_group'
        df['_temp_lagged_target'] = lagged_target
        df[rank_col] = (
            df.groupby([hier_col, date_col])['_temp_lagged_target']
            .rank(pct=True, method='average')
        )
        df.drop(columns=['_temp_lagged_target'], inplace=True)
        features_added.append(rank_col)

        # ─── 5. GROUP TREND (rolling slope of group total) ──────────
        for window in windows:
            trend_col = f'{prefix}_group_trend_{window}'

            # Compute rolling mean of group total (already lagged)
            df[trend_col] = (
                df.groupby(key_col)[f'{prefix}_group_total_lag']
                .transform(lambda x: x.rolling(window, min_periods=max(2, window // 2)).apply(
                    _compute_slope, raw=True
                ))
            )
            features_added.append(trend_col)

        # ─── 6. RELATIVE TO GROUP MEAN (lagged) ─────────────────────
        rel_col = f'{prefix}_rel_to_group'
        df[rel_col] = (
            lagged_target / (df[f'{prefix}_group_mean_lag'] + 1e-10)
        ).clip(0, 20)
        features_added.append(rel_col)

        # ─── CLEANUP internal columns ────────────────────────────────
        internal = [c for c in df.columns if c.startswith(f'_hier_{safe}_')]
        df.drop(columns=internal, inplace=True, errors='ignore')

    # Fill NaN in new features with 0 (happens at start of series due to lag)
    for col in features_added:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logger.info(f"Created {len(features_added)} hierarchy temporal features: "
                f"{features_added[:8]}{'...' if len(features_added) > 8 else ''}")

    return df


def _compute_slope(values: np.ndarray) -> float:
    """Compute linear regression slope of an array. Used for group trend."""
    # Filter out NaN values before polyfit
    mask = ~np.isnan(values)
    clean = values[mask]
    n = len(clean)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    try:
        slope = np.polyfit(x, clean, 1)[0]
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def get_hierarchy_feature_names(
    target_col: str,
    hierarchy_cols: List[str],
    windows: List[int] = None,
) -> List[str]:
    """Return the list of feature names that create_hierarchy_temporal_features produces."""
    if windows is None:
        windows = [4, 13]

    names = []
    for hier_col in hierarchy_cols[:2]:
        safe = hier_col.lower().replace(' ', '_')[:15]
        prefix = f"{target_col}_{safe}"
        names.extend([
            f'{prefix}_group_total_lag',
            f'{prefix}_group_mean_lag',
            f'{prefix}_share_lag',
            f'{prefix}_rank_in_group',
            f'{prefix}_rel_to_group',
        ])
        for w in windows:
            names.append(f'{prefix}_group_trend_{w}')
    return names
