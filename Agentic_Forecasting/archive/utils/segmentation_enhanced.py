# utils/segmentation_enhanced.py
"""
Enhanced Segmentation Features for Agentic Demand Forecasting (Phase 2).

This module adds rich, external-feature-aware dimensions to segmentation:

1. **External Response Profiles** — Per-key measures of HOW each key responds to
   external drivers (price elasticity proxy, promo lift, feature correlations).
   Keys that respond similarly to promotions should be in the same segment.

2. **Seasonality Shape Features** — Fourier coefficients capturing the SHAPE of
   each key's seasonal pattern. Keys with summer peaks cluster separately from
   keys with winter peaks, even if both are "seasonal".

3. **Trend Direction Features** — Recent growth/decline direction and magnitude,
   not just whether a trend exists.

4. **Hierarchy-Aware Features** — If categorical columns define a hierarchy
   (e.g., brand → category → division), compute per-key features relative to
   hierarchy peers (share of category, relative volume).

5. **Feature Importance Profiles** — Quick per-key feature importance estimates
   using lightweight models, capturing WHAT DRIVES each key's demand.

All functions are DETERMINISTIC (no LLM). They take per_key_metrics or source
data and return enriched DataFrames with new feature columns.

Usage:
    from utils.segmentation_enhanced import enrich_segmentation_features

    enriched_df = enrich_segmentation_features(
        per_key_metrics=per_key_df,
        source_df=raw_data_df,
        key_cols=['key'],
        date_col='year_week',
        target_col='Actuals',
        feature_availability_context=fa_context,
    )
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. EXTERNAL RESPONSE PROFILE FEATURES
# =============================================================================

def compute_external_response_profiles(
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    external_numeric_cols: List[str],
    history_cutoff: Optional[str] = None,
    max_features: int = 20,
) -> pd.DataFrame:
    """
    Compute per-key external response profiles from raw time-series data.

    For each key and each external feature, computes:
    - Correlation with target (how strongly does this feature drive demand?)
    - Responsiveness magnitude (how much does demand change per unit feature change?)

    These become static key-level features for clustering.

    Parameters
    ----------
    source_df : pd.DataFrame
        Raw source data with time series
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    external_numeric_cols : List[str]
        Numeric external features to compute responses for
    history_cutoff : str, optional
        Only use data up to this period
    max_features : int
        Maximum number of external features to process (top by variance)

    Returns
    -------
    pd.DataFrame
        One row per key with response profile columns
    """
    logger.info(f"Computing external response profiles for {len(external_numeric_cols)} features...")

    df = source_df.copy()

    # Filter to history only
    if history_cutoff:
        try:
            cutoff = int(history_cutoff)
        except (ValueError, TypeError):
            cutoff = history_cutoff
        df = df[df[date_col] <= cutoff]

    if len(df) == 0:
        logger.warning("No data after history cutoff filter")
        return pd.DataFrame()

    # Select features that exist and have variance
    valid_cols = []
    for col in external_numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].dropna().nunique() < 3:
            continue
        valid_cols.append(col)

    if not valid_cols:
        logger.warning("No valid external numeric columns found for response profiles")
        return pd.DataFrame()

    # Limit to top features by cross-key variance
    if len(valid_cols) > max_features:
        variances = {col: df.groupby(key_cols)[col].mean().var() for col in valid_cols}
        valid_cols = sorted(variances, key=variances.get, reverse=True)[:max_features]
        logger.info(f"Selected top {max_features} external features by variance")

    key_col = key_cols[0] if len(key_cols) == 1 else key_cols

    results = []
    for key_val, group in df.groupby(key_cols):
        if len(group) < 10:
            continue

        row = {key_cols[0]: key_val} if isinstance(key_val, (str, int, float)) else dict(zip(key_cols, key_val))

        # Compute correlation with target for each external feature
        corr_values = []
        for col in valid_cols:
            series = group[col].dropna()
            target_aligned = group.loc[series.index, target_col]

            if len(series) < 5 or series.std() < 1e-10:
                corr = 0.0
            else:
                corr = series.corr(target_aligned)
                corr = float(corr) if pd.notna(corr) else 0.0

            row[f'resp_{col}_corr'] = np.clip(corr, -1, 1)
            corr_values.append(abs(corr))

        # Aggregate response metrics
        row['ext_response_mean_abs_corr'] = np.mean(corr_values) if corr_values else 0.0
        row['ext_response_max_abs_corr'] = np.max(corr_values) if corr_values else 0.0
        row['ext_response_n_significant'] = sum(1 for c in corr_values if c > 0.2)

        results.append(row)

    if not results:
        logger.warning("No keys with sufficient data for response profiles")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    logger.info(f"Computed response profiles for {len(result_df)} keys, {len(valid_cols)} features")

    return result_df


# =============================================================================
# 2. SEASONALITY SHAPE FEATURES
# =============================================================================

def compute_seasonality_shape_features(
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    seasonal_period: int = 52,
    n_harmonics: int = 3,
    history_cutoff: Optional[str] = None,
    min_obs: int = 26,
) -> pd.DataFrame:
    """
    Compute per-key seasonality shape features using Fourier decomposition.

    Instead of just "has seasonality: yes/no", this captures the SHAPE of the
    seasonal pattern — keys with summer peaks cluster separately from keys with
    winter peaks, even if both have similar seasonal_strength.

    Features created per key:
    - seasonal_amplitude_k (k=1..n_harmonics): magnitude of k-th harmonic
    - seasonal_phase_k: phase (timing) of k-th harmonic
    - seasonal_dominant_harmonic: which harmonic is strongest
    - seasonal_total_power: total seasonal power (sum of amplitudes)

    Parameters
    ----------
    source_df : pd.DataFrame
        Raw source data
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    seasonal_period : int
        Seasonal period (52 for weekly, 12 for monthly)
    n_harmonics : int
        Number of Fourier harmonics to extract
    history_cutoff : str, optional
        Only use data up to this period
    min_obs : int
        Minimum observations per key to compute features

    Returns
    -------
    pd.DataFrame
        One row per key with seasonality shape columns
    """
    logger.info(f"Computing seasonality shape features (period={seasonal_period}, harmonics={n_harmonics})...")

    df = source_df.copy()
    if history_cutoff:
        try:
            cutoff = int(history_cutoff)
        except (ValueError, TypeError):
            cutoff = history_cutoff
        df = df[df[date_col] <= cutoff]

    results = []
    for key_val, group in df.groupby(key_cols):
        group = group.sort_values(date_col)
        values = group[target_col].fillna(0).values

        if len(values) < min_obs:
            continue

        row = {key_cols[0]: key_val} if isinstance(key_val, (str, int, float)) else dict(zip(key_cols, key_val))

        # Compute FFT
        try:
            fft_vals = np.fft.rfft(values)
            freqs = np.fft.rfftfreq(len(values))
            amplitudes = np.abs(fft_vals) / len(values)

            # Extract harmonics at seasonal frequency and its multiples
            total_power = 0.0
            dominant_harmonic = 0
            dominant_amplitude = 0.0

            for k in range(1, n_harmonics + 1):
                # Find the FFT bin closest to k/seasonal_period frequency
                target_freq = k / seasonal_period
                freq_idx = np.argmin(np.abs(freqs - target_freq))

                amp = float(amplitudes[freq_idx]) if freq_idx < len(amplitudes) else 0.0
                phase = float(np.angle(fft_vals[freq_idx])) if freq_idx < len(fft_vals) else 0.0

                row[f'seasonal_amplitude_{k}'] = amp
                row[f'seasonal_phase_{k}'] = phase
                total_power += amp

                if amp > dominant_amplitude:
                    dominant_amplitude = amp
                    dominant_harmonic = k

            row['seasonal_total_power'] = total_power
            row['seasonal_dominant_harmonic'] = dominant_harmonic

            # Normalized seasonal power (relative to mean)
            mean_val = np.mean(np.abs(values)) + 1e-10
            row['seasonal_power_normalized'] = total_power / mean_val

        except Exception:
            # Default zeros
            for k in range(1, n_harmonics + 1):
                row[f'seasonal_amplitude_{k}'] = 0.0
                row[f'seasonal_phase_{k}'] = 0.0
            row['seasonal_total_power'] = 0.0
            row['seasonal_dominant_harmonic'] = 0
            row['seasonal_power_normalized'] = 0.0

        results.append(row)

    if not results:
        logger.warning("No keys with sufficient data for seasonality shape features")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    logger.info(f"Computed seasonality shape features for {len(result_df)} keys")
    return result_df


# =============================================================================
# 3. TREND DIRECTION FEATURES
# =============================================================================

def compute_trend_direction_features(
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    history_cutoff: Optional[str] = None,
    recent_window: int = 13,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Compute per-key trend direction and magnitude features.

    Goes beyond "trend_strength" by capturing:
    - Direction: growing vs declining
    - Magnitude: how fast
    - Recency: is the recent trend different from the long-term trend?
    - Acceleration: is the trend accelerating or decelerating?

    Parameters
    ----------
    source_df : pd.DataFrame
        Raw source data
    key_cols, date_col, target_col : str
        Column names
    history_cutoff : str, optional
        Cutoff
    recent_window : int
        Number of periods for "recent" trend (default 13 = quarterly)
    min_obs : int
        Minimum observations
    """
    logger.info(f"Computing trend direction features (recent_window={recent_window})...")

    df = source_df.copy()
    if history_cutoff:
        try:
            cutoff = int(history_cutoff)
        except (ValueError, TypeError):
            cutoff = history_cutoff
        df = df[df[date_col] <= cutoff]

    results = []
    for key_val, group in df.groupby(key_cols):
        group = group.sort_values(date_col)
        values = group[target_col].fillna(0).values

        if len(values) < min_obs:
            continue

        row = {key_cols[0]: key_val} if isinstance(key_val, (str, int, float)) else dict(zip(key_cols, key_val))

        try:
            x_full = np.arange(len(values), dtype=float)

            # Full-history trend (slope of linear fit)
            slope_full, intercept = np.polyfit(x_full, values, 1)
            mean_val = np.mean(values) + 1e-10
            row['trend_slope'] = float(slope_full)
            row['trend_slope_normalized'] = float(slope_full / mean_val)  # % change per period

            # Recent trend (last recent_window periods)
            recent_vals = values[-recent_window:]
            if len(recent_vals) >= 5:
                x_recent = np.arange(len(recent_vals), dtype=float)
                slope_recent = np.polyfit(x_recent, recent_vals, 1)[0]
                row['trend_slope_recent'] = float(slope_recent)
                row['trend_slope_recent_normalized'] = float(slope_recent / mean_val)
            else:
                row['trend_slope_recent'] = row['trend_slope']
                row['trend_slope_recent_normalized'] = row['trend_slope_normalized']

            # Trend direction: 1=growing, 0=stable, -1=declining
            norm_slope = row['trend_slope_normalized']
            if norm_slope > 0.005:   # >0.5% per period growth
                row['trend_direction'] = 1.0
            elif norm_slope < -0.005:
                row['trend_direction'] = -1.0
            else:
                row['trend_direction'] = 0.0

            # Trend acceleration: is recent trend faster or slower than overall?
            row['trend_acceleration'] = float(
                row['trend_slope_recent_normalized'] - row['trend_slope_normalized']
            )

            # Recent vs long-term ratio (captures trend regime change)
            recent_mean = np.mean(recent_vals) + 1e-10
            old_mean = np.mean(values[:-recent_window]) + 1e-10 if len(values) > recent_window else mean_val
            row['recent_vs_longterm_ratio'] = float(np.clip(recent_mean / old_mean, 0.1, 10.0))

        except (np.linalg.LinAlgError, ValueError):
            row['trend_slope'] = 0.0
            row['trend_slope_normalized'] = 0.0
            row['trend_slope_recent'] = 0.0
            row['trend_slope_recent_normalized'] = 0.0
            row['trend_direction'] = 0.0
            row['trend_acceleration'] = 0.0
            row['recent_vs_longterm_ratio'] = 1.0

        results.append(row)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    logger.info(f"Computed trend direction features for {len(result_df)} keys")
    return result_df


# =============================================================================
# 4. HIERARCHY-AWARE FEATURES
# =============================================================================

def detect_hierarchy_columns(
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    candidate_cols: Optional[List[str]] = None,
    max_cardinality: int = 500,
    min_cardinality: int = 2,
) -> List[Dict[str, Any]]:
    """Legacy flat-list hierarchy detector (kept for back-compat).

    **New callers should use** :func:`detect_hierarchy_chains` instead —
    that function applies name-priority scoring, collapses degenerate
    aliases, groups nested columns into explicit coarse→fine chains, and
    returns ``product`` / ``customer`` / ``other`` separately.

    This legacy entry point is preserved so existing callers keep working
    without behavioural change; it simply flattens the result of
    :func:`detect_hierarchy_chains` into a single list sorted by
    cardinality. When called with the legacy default ``min_cardinality=2``
    and no ``candidate_cols``, it ALSO emits a warning because this
    configuration is what produced the UK CONDIMENT bug where binary
    indicator columns were picked up as hierarchies.
    """
    if candidate_cols is None and min_cardinality < 3:
        logger.warning(
            "detect_hierarchy_columns called in unconstrained mode "
            "(candidate_cols=None, min_cardinality=%d). Binary indicator "
            "columns may be treated as hierarchies. Configure "
            "design.hierarchy_detection.candidate_cols to restrict the "
            "search to a curated list.",
            min_cardinality,
        )
    result = detect_hierarchy_chains(
        source_df=source_df,
        key_cols=key_cols,
        date_col=date_col,
        target_col=target_col,
        candidate_cols=candidate_cols,
        min_cardinality=min_cardinality,
        max_cardinality=max_cardinality,
        include_binary=(min_cardinality <= 2),
    )
    # Flatten chains in the order: product finest-first, customer,
    # then other. Callers that take ``[:3]`` therefore get the most
    # useful columns rather than the smallest cardinality ones.
    flat: List[Dict[str, Any]] = []
    for chain in result.get('product', []) + result.get('customer', []) + result.get('other', []):
        flat.append(chain)
    # Keep a stable sort by cardinality ascending *within* the combined
    # list so that legacy callers still see coarse→fine ordering.
    flat.sort(key=lambda x: x['cardinality'])
    if flat:
        logger.info("detect_hierarchy_columns returning %d candidates:", len(flat))
        for h in flat[:5]:
            logger.info(
                "  %s: %d groups (%.1f keys/group, score=%.2f)",
                h['column'], h['cardinality'], h['keys_per_group_avg'],
                h.get('score', 0.0),
            )
    return flat


# =============================================================================
# PHASE-3 HIERARCHY CHAIN DETECTOR (curated candidates + named chains)
# =============================================================================

def detect_hierarchy_chains(
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    candidate_cols: Optional[List[str]] = None,
    *,
    min_keys_per_group: int = 5,
    min_cardinality: int = 3,
    max_cardinality: int = 2000,
    require_coverage: float = 0.95,
    include_binary: bool = False,
    product_name_patterns: Optional[List[str]] = None,
    customer_name_patterns: Optional[List[str]] = None,
    collapse_duplicate_columns: bool = True,
    max_depth_per_chain: int = 3,
) -> Dict[str, Any]:
    """Detect named hierarchy chains against a curated candidate list.

    Implements the Phase-3 hierarchy design:

    1. Validate each candidate against the data (cardinality bounds, key
       coverage, functional dependency per key).
    2. Collapse columns carrying identical values into a single
       representative, picking the name that best matches the product
       pattern list.
    3. Build the full pairwise nesting matrix among valid candidates.
    4. Extract chains via greedy walks from the finest-cardinality column
       up to its strict parents; unconnected candidates become their own
       chain (parallel hierarchies).
    5. Classify each chain as ``product`` / ``customer`` / ``other`` using
       the name-prior heuristics.
    6. Score and rank columns within each chain.

    Parameters
    ----------
    source_df, key_cols, date_col, target_col
        Data and schema.
    candidate_cols : List[str], optional
        Columns to restrict the search to. When ``None`` every non-
        key/date/target column is inspected — equivalent to the legacy
        auto-detector. Listing a non-existent column produces a warning
        and that entry is skipped.
    min_keys_per_group, min_cardinality, max_cardinality,
    require_coverage, include_binary,
    product_name_patterns, customer_name_patterns,
    collapse_duplicate_columns, max_depth_per_chain
        See :class:`HierarchyDetectionConfig`.

    Returns
    -------
    Dict[str, Any]
        ``{
            'product':  [chain items...],
            'customer': [chain items...],
            'other':    [chain items...],
            'candidates_evaluated': List[str],
            'candidates_rejected': List[Dict[str, str]],  # column + reason
            'duplicates_collapsed': Dict[str, str],        # alias -> kept
        }``

        Each chain item is a dict with ``column``, ``cardinality``,
        ``keys_per_group_avg``, ``keys_per_group_min``,
        ``coverage``, ``name_match``, ``score`` fields.
    """
    key_col = key_cols[0] if len(key_cols) == 1 else None
    if key_col is None:
        logger.info("Multi-key columns — skipping hierarchy detection")
        return {
            'product': [], 'customer': [], 'other': [],
            'candidates_evaluated': [], 'candidates_rejected': [],
            'duplicates_collapsed': {},
        }

    if product_name_patterns is None:
        product_name_patterns = [
            "brand", "segment", "form", "category", "market",
            "family", "variant", "gtin", "cpl", "product",
        ]
    if customer_name_patterns is None:
        customer_name_patterns = [
            "apg", "customer", "account", "planning_customer",
            "retailer", "store", "channel", "banner",
        ]
    product_name_patterns = [p.lower() for p in product_name_patterns]
    customer_name_patterns = [p.lower() for p in customer_name_patterns]

    exclude = set(key_cols + [date_col, target_col])

    # -------- Step 1: resolve the candidate list ---------------------------
    evaluated: List[str] = []
    rejected: List[Dict[str, str]] = []

    if candidate_cols:
        # Strict mode — only these columns
        cols_to_check = []
        for c in candidate_cols:
            if c in exclude:
                rejected.append({'column': c, 'reason': 'is_key_date_or_target'})
                continue
            if c not in source_df.columns:
                logger.warning(
                    "hierarchy_detection.candidate_cols lists '%s' but "
                    "it is not in the source data — skipped.", c,
                )
                rejected.append({'column': c, 'reason': 'column_not_in_data'})
                continue
            cols_to_check.append(c)
        mode = "strict"
    else:
        cols_to_check = [c for c in source_df.columns if c not in exclude]
        mode = "auto"

    n_keys = source_df[key_col].nunique()
    n_rows = len(source_df)
    logger.info(
        "Hierarchy detection: mode=%s, %d candidate(s), %d keys, %d rows",
        mode, len(cols_to_check), n_keys, n_rows,
    )

    # -------- Step 2: per-column validity + metrics -----------------------
    valid: List[Dict[str, Any]] = []

    for col in cols_to_check:
        evaluated.append(col)

        try:
            series = source_df[col]
        except KeyError:
            rejected.append({'column': col, 'reason': 'column_not_in_data'})
            continue

        n_unique = series.nunique(dropna=True)
        n_non_null = int(series.notna().sum())
        coverage = n_non_null / max(1, n_rows)

        if n_unique < min_cardinality or (n_unique == 2 and not include_binary):
            rejected.append({
                'column': col,
                'reason': f"cardinality {n_unique} below min_cardinality={min_cardinality} "
                          + ("" if include_binary else "or binary-excluded"),
            })
            continue
        if n_unique > max_cardinality:
            rejected.append({
                'column': col,
                'reason': f"cardinality {n_unique} above max_cardinality={max_cardinality}",
            })
            continue
        if n_unique >= n_keys:
            rejected.append({
                'column': col,
                'reason': f"cardinality {n_unique} >= n_keys {n_keys} (does not group keys)",
            })
            continue
        if coverage < require_coverage:
            rejected.append({
                'column': col,
                'reason': f"coverage {coverage:.2%} < require_coverage={require_coverage:.0%}",
            })
            continue

        # Functional dependency test: each key → exactly one value
        try:
            key_to_val = source_df.groupby(key_col)[col].nunique(dropna=True)
            is_functional = (key_to_val <= 1).all()
        except Exception as exc:
            rejected.append({'column': col, 'reason': f"functional-dep test failed: {exc}"})
            continue
        if not is_functional:
            pct = (key_to_val <= 1).mean() * 100
            rejected.append({
                'column': col,
                'reason': f"not functional per key ({pct:.1f}% keys have single value)",
            })
            continue

        group_sizes = source_df.groupby(col)[key_col].nunique()
        median_group = int(group_sizes.median())
        if median_group < min_keys_per_group:
            rejected.append({
                'column': col,
                'reason': f"median group size {median_group} < "
                          f"min_keys_per_group={min_keys_per_group}",
            })
            continue

        lname = col.lower()
        prod_match = any(p in lname for p in product_name_patterns)
        cust_match = any(p in lname for p in customer_name_patterns)

        valid.append({
            'column': col,
            'cardinality': int(n_unique),
            'grouping_ratio': float(n_unique) / max(1, n_keys),
            'keys_per_group_avg': float(n_keys) / max(1, n_unique),
            'keys_per_group_min': int(group_sizes.min()),
            'keys_per_group_max': int(group_sizes.max()),
            'coverage': round(float(coverage), 4),
            'name_match': 'product' if prod_match else ('customer' if cust_match else 'other'),
        })

    if not valid:
        logger.info("No valid hierarchy columns after filtering.")
        return {
            'product': [], 'customer': [], 'other': [],
            'candidates_evaluated': evaluated,
            'candidates_rejected': rejected,
            'duplicates_collapsed': {},
        }

    # -------- Step 3: collapse duplicate columns --------------------------
    duplicates_collapsed: Dict[str, str] = {}
    if collapse_duplicate_columns and len(valid) > 1:
        # Two columns are duplicates if they induce the same partition on
        # keys — i.e., for every key, the pair (valA, valB) is constant.
        # A faster equivalence: both columns agree on a full catalogue of
        # (val_A, val_B) pairs that match the cardinalities.
        def _key_partition(col: str) -> pd.Series:
            return source_df.groupby(key_col)[col].first()

        partitions = {v['column']: _key_partition(v['column']) for v in valid}

        kept = []
        removed: Set[str] = set()
        # Sort candidates so that the "preferred" name (lowest index in
        # product_name_patterns) wins when collisions occur.
        name_priority_order = product_name_patterns + customer_name_patterns

        def _priority(name: str) -> int:
            ln = name.lower()
            for i, pat in enumerate(name_priority_order):
                if pat in ln:
                    return i
            return len(name_priority_order)

        valid_sorted = sorted(valid, key=lambda v: (_priority(v['column']), v['column']))
        for a in valid_sorted:
            if a['column'] in removed:
                continue
            kept.append(a)
            for b in valid_sorted:
                if b['column'] == a['column'] or b['column'] in removed:
                    continue
                if a['cardinality'] != b['cardinality']:
                    continue  # only identical partitions can be duplicates
                pa, pb = partitions[a['column']], partitions[b['column']]
                # Map pb → pa (any bijection between the two is fine)
                try:
                    joint = pd.concat([pa.rename('A'), pb.rename('B')], axis=1)
                    unique_pairs = joint.drop_duplicates()
                    if len(unique_pairs) == a['cardinality']:
                        # b is a relabelling of a → duplicate
                        duplicates_collapsed[b['column']] = a['column']
                        removed.add(b['column'])
                except Exception:
                    continue
        valid = kept
        if duplicates_collapsed:
            logger.info(
                "Collapsed %d duplicate column(s): %s",
                len(duplicates_collapsed),
                ", ".join(f"{k}→{v}" for k, v in duplicates_collapsed.items()),
            )

    # -------- Step 4: build nesting matrix + chains -----------------------
    # Nesting: column A is "under" column B when every key-value of A maps
    # to exactly one key-value of B (A is finer than B).
    col_vals: Dict[str, pd.Series] = {
        v['column']: source_df.groupby(key_col)[v['column']].first() for v in valid
    }
    parents: Dict[str, Set[str]] = {v['column']: set() for v in valid}
    for a in valid:
        for b in valid:
            if a['column'] == b['column']:
                continue
            # a is a potential CHILD of b, so a must be finer (more
            # distinct values) than b. Strict inequality so we don't treat
            # duplicates as parent/child (duplicates were collapsed).
            if a['cardinality'] <= b['cardinality']:
                continue
            # Check nesting: every value of A maps to exactly one value of B.
            nesting = (
                pd.concat([col_vals[a['column']].rename('A'),
                           col_vals[b['column']].rename('B')], axis=1)
                .drop_duplicates()
                .groupby('A')['B']
                .nunique()
            )
            if (nesting <= 1).all():
                parents[a['column']].add(b['column'])

    # Partition valid columns by label FIRST — a chain can only contain
    # columns that share a label (product / customer / other). This
    # prevents e.g. APG_code being absorbed into a product chain just
    # because a finer customer-level code (CPL) happens to carry a
    # product-sounding pattern in its name.
    by_label: Dict[str, List[Dict[str, Any]]] = {'product': [], 'customer': [], 'other': []}
    for v in valid:
        by_label[v['name_match']].append(v)

    def _build_chains(pool: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Build deepest coarse→fine chains within a single-label pool.

        For each leaf (starting from the finest-cardinality column), the
        immediate parent is the parent with the LARGEST cardinality still
        strictly smaller than the child's — i.e. the closest level above
        in the tree. Walking that pointer repeatedly yields the deepest
        possible chain.
        """
        pool_cols = {v['column'] for v in pool}
        card_of = {v['column']: v['cardinality'] for v in pool}
        by_card = sorted(pool, key=lambda v: v['cardinality'], reverse=True)  # fine→coarse
        chains_: List[List[Dict[str, Any]]] = []
        assigned_: Set[str] = set()
        for leaf in by_card:
            if leaf['column'] in assigned_:
                continue
            chain_nodes: List[str] = [leaf['column']]
            assigned_.add(leaf['column'])
            current = leaf['column']
            depth = 1
            while depth < max_depth_per_chain:
                # Immediate parent = parent with the LARGEST cardinality
                # (closest above) that is in the same label pool and not
                # yet assigned to another chain.
                parent_candidates = [
                    p for p in parents[current]
                    if p in pool_cols and p not in assigned_
                ]
                if not parent_candidates:
                    break
                parent = max(parent_candidates, key=lambda p: card_of[p])
                chain_nodes.append(parent)
                assigned_.add(parent)
                current = parent
                depth += 1
            chain_nodes.reverse()
            chain_items = [next(v for v in valid if v['column'] == c) for c in chain_nodes]
            chains_.append(chain_items)
        return chains_

    product_chains = _build_chains(by_label['product'])
    customer_chains = _build_chains(by_label['customer'])
    other_chains = _build_chains(by_label['other'])

    # For each named bucket, pick the deepest chain as the primary; any
    # additional chains (parallel hierarchies) get appended to 'other' so
    # they are still surfaced but don't hijack the primary label.
    def _pick_primary(chains_list: List[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        if not chains_list:
            return [], []
        sorted_chains = sorted(chains_list, key=lambda c: -len(c))
        return sorted_chains[0], sorted_chains[1:]

    named: Dict[str, List[Dict[str, Any]]] = {'product': [], 'customer': [], 'other': []}
    named['product'], product_extras = _pick_primary(product_chains)
    named['customer'], customer_extras = _pick_primary(customer_chains)
    named['other'], other_extras = _pick_primary(other_chains)

    # Flatten any extras into 'other' as individual leaf dicts
    for extras in (product_extras, customer_extras, other_extras):
        for chain in extras:
            named['other'].extend(chain)

    # -------- Step 6: score ------------------------------------------------
    for bucket in named.values():
        for i, item in enumerate(bucket):
            # Score rewards: decent keys_per_group, reasonable cardinality,
            # name match, and position in chain (coarsest first).
            card = item['cardinality']
            kpg = item['keys_per_group_avg']
            name_boost = 1.0 if item['name_match'] == 'product' else 0.9 if item['name_match'] == 'customer' else 0.7
            card_score = max(0.0, 1.0 - abs(np.log10(max(card, 1)) - 1.5))  # peaks around 30 groups
            kpg_score = max(0.0, 1.0 - abs(np.log10(max(kpg, 1)) - 1.5))     # peaks around 30 keys/group
            item['score'] = round(float(name_boost * (0.4 * card_score + 0.4 * kpg_score + 0.2 * (1.0 - i * 0.2))), 4)

    # Log a summary
    def _summarise(bucket: List[Dict[str, Any]]) -> str:
        if not bucket:
            return "(none)"
        return " → ".join(f"{it['column']}({it['cardinality']})" for it in bucket)

    logger.info("Hierarchy chains detected:")
    logger.info("  product  : %s", _summarise(named['product']))
    logger.info("  customer : %s", _summarise(named['customer']))
    if named['other']:
        logger.info("  other    : %s", _summarise(named['other']))
    if rejected:
        logger.info("  rejected : %d candidates", len(rejected))

    return {
        'product':  named['product'],
        'customer': named['customer'],
        'other':    named['other'],
        'candidates_evaluated': evaluated,
        'candidates_rejected':  rejected,
        'duplicates_collapsed': duplicates_collapsed,
    }


def compute_hierarchy_features(
    per_key_metrics: pd.DataFrame,
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    hierarchy_cols: List[str],
    history_cutoff: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute per-key features relative to hierarchy group.

    For each hierarchy column (e.g., 'Category', 'Brand'):
    - share_of_group: key's mean / group total mean (market share proxy)
    - rank_in_group: percentile rank within group
    - group_size: how many keys in the group
    - relative_cv: key's CV / group's average CV

    Parameters
    ----------
    per_key_metrics : pd.DataFrame
        Per-key metrics (must have 'mean', 'cv')
    source_df : pd.DataFrame
        Raw source data (to get hierarchy column values)
    key_cols : List[str]
        Key columns
    date_col, target_col : str
        Column names
    hierarchy_cols : List[str]
        Hierarchy columns to compute features for
    history_cutoff : str, optional
        Cutoff

    Returns
    -------
    pd.DataFrame
        Per-key metrics enriched with hierarchy features
    """
    logger.info(f"Computing hierarchy features for {len(hierarchy_cols)} hierarchy columns...")

    df = per_key_metrics.copy()
    key_col = key_cols[0] if len(key_cols) == 1 else key_cols

    # Get hierarchy mapping: key → hierarchy_value (from source data)
    for h_col in hierarchy_cols:
        if h_col in df.columns:
            continue  # Already present in per_key_metrics

        if h_col not in source_df.columns:
            logger.warning(f"Hierarchy column '{h_col}' not in source data, skipping")
            continue

        # Get key → hierarchy mapping (take mode/first for each key)
        key_mapping = source_df.groupby(key_col)[h_col].first().reset_index()
        df = df.merge(key_mapping, on=key_col, how='left')

    features_added = []

    for h_col in hierarchy_cols:
        if h_col not in df.columns:
            continue

        safe_name = h_col.lower().replace(' ', '_')[:20]

        try:
            # Group size
            group_sizes = df.groupby(h_col)[key_col].transform('count')
            df[f'hier_{safe_name}_group_size'] = group_sizes
            features_added.append(f'hier_{safe_name}_group_size')

            # Share of group (volume share)
            if 'mean' in df.columns:
                group_totals = df.groupby(h_col)['mean'].transform('sum')
                df[f'hier_{safe_name}_share'] = (df['mean'] / (group_totals + 1e-10)).clip(0, 1)
                features_added.append(f'hier_{safe_name}_share')

                # Rank within group
                df[f'hier_{safe_name}_rank'] = df.groupby(h_col)['mean'].rank(pct=True)
                features_added.append(f'hier_{safe_name}_rank')

            # Relative CV (key CV / group avg CV)
            if 'cv' in df.columns:
                group_avg_cv = df.groupby(h_col)['cv'].transform('mean')
                df[f'hier_{safe_name}_relative_cv'] = (
                    df['cv'] / (group_avg_cv + 1e-10)
                ).clip(0, 10)
                features_added.append(f'hier_{safe_name}_relative_cv')

        except Exception as e:
            logger.warning(f"Could not compute hierarchy features for {h_col}: {e}")

    logger.info(f"Added {len(features_added)} hierarchy features: {features_added[:10]}")
    return df


# =============================================================================
# 5. FEATURE IMPORTANCE PROFILES
# =============================================================================

def compute_feature_importance_profiles(
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    history_cutoff: Optional[str] = None,
    n_importance_dims: int = 5,
    sample_keys: int = 200,
) -> pd.DataFrame:
    """
    Compute lightweight feature importance profiles per key.

    For a sample of keys, trains a quick model and extracts feature importances.
    These are then aggregated into "importance profile" features that characterize
    what drives each key's demand.

    Features created:
    - imp_top1_feature: index of most important feature
    - imp_concentration: Herfindahl index of importances (concentrated vs spread)
    - imp_external_pct: % of importance from external features
    - imp_lag_pct: % of importance from lag features
    - imp_calendar_pct: % of importance from calendar features

    Parameters
    ----------
    source_df : pd.DataFrame
        Raw data
    key_cols, date_col, target_col : str
        Columns
    feature_cols : List[str]
        Features to compute importance for
    history_cutoff : str, optional
        Cutoff
    n_importance_dims : int
        Number of importance dimensions to produce
    sample_keys : int
        Maximum keys to sample for efficiency

    Returns
    -------
    pd.DataFrame
        One row per key with importance profile features
    """
    logger.info(f"Computing feature importance profiles (max {sample_keys} keys)...")

    df = source_df.copy()
    if history_cutoff:
        try:
            cutoff = int(history_cutoff)
        except (ValueError, TypeError):
            cutoff = history_cutoff
        df = df[df[date_col] <= cutoff]

    # Get valid feature columns
    valid_features = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_features) < 3:
        logger.warning("Not enough numeric features for importance profiles")
        return pd.DataFrame()

    key_col = key_cols[0] if len(key_cols) == 1 else key_cols

    # Sample keys for efficiency
    all_keys = df[key_col].unique()
    if len(all_keys) > sample_keys:
        rng = np.random.RandomState(42)
        sampled_keys = rng.choice(all_keys, sample_keys, replace=False)
        df = df[df[key_col].isin(sampled_keys)]

    results = []
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        logger.warning("sklearn not available for importance profiles")
        return pd.DataFrame()

    for key_val, group in df.groupby(key_cols):
        if len(group) < 15:
            continue

        row = {key_cols[0]: key_val} if isinstance(key_val, (str, int, float)) else dict(zip(key_cols, key_val))

        X = group[valid_features].fillna(0)
        y = group[target_col].fillna(0)

        if y.std() < 1e-10:
            continue

        try:
            rf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=1)
            rf.fit(X, y)
            importances = rf.feature_importances_

            # Concentration (Herfindahl index): high = one feature dominates
            hhi = float(np.sum(importances ** 2))
            row['imp_concentration'] = hhi

            # Top feature index
            row['imp_top1_idx'] = int(np.argmax(importances))
            row['imp_top1_value'] = float(importances[row['imp_top1_idx']])

            # Classify importance by feature type
            ext_importance = sum(
                importances[i] for i, f in enumerate(valid_features)
                if any(p in f.lower() for p in ['price', 'promo', 'discount', 'holiday', 'weather', 'temp'])
            )
            lag_importance = sum(
                importances[i] for i, f in enumerate(valid_features)
                if any(p in f.lower() for p in ['lag', 'roll', 'ewm', 'ma_'])
            )
            total_imp = sum(importances) + 1e-10
            row['imp_external_pct'] = float(ext_importance / total_imp)
            row['imp_lag_pct'] = float(lag_importance / total_imp)
            row['imp_other_pct'] = float(1.0 - row['imp_external_pct'] - row['imp_lag_pct'])

        except Exception:
            row['imp_concentration'] = 0.0
            row['imp_top1_idx'] = 0
            row['imp_top1_value'] = 0.0
            row['imp_external_pct'] = 0.0
            row['imp_lag_pct'] = 0.0
            row['imp_other_pct'] = 1.0

        results.append(row)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    logger.info(f"Computed importance profiles for {len(result_df)} keys")
    return result_df


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def enrich_segmentation_features(
    per_key_metrics: pd.DataFrame,
    source_df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    time_format: str = 'year_week',
    feature_availability_context: Optional[Dict[str, Any]] = None,
    hierarchy_cols: Optional[List[str]] = None,
    auto_detect_hierarchy: bool = True,
    compute_response_profiles: bool = True,
    compute_seasonality: bool = True,
    compute_trends: bool = True,
    compute_importance: bool = True,
    max_external_features: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Master function: Enrich per_key_metrics with all enhanced segmentation features.

    This is the single entry point called from run_segmentation_pipeline().
    It orchestrates all Phase 2 feature enrichments.

    Parameters
    ----------
    per_key_metrics : pd.DataFrame
        Base per-key metrics from EDA
    source_df : pd.DataFrame
        Raw source data (needed for time-series-level computations)
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    time_format : str
        'year_week' or 'year_month'
    feature_availability_context : Dict, optional
        Context from Feature Availability Detection
    hierarchy_cols : List[str], optional
        Explicit hierarchy columns. If None and auto_detect_hierarchy=True,
        will be auto-detected from data.
    auto_detect_hierarchy : bool
        Whether to auto-detect hierarchy columns
    compute_response_profiles : bool
        Compute external response profiles
    compute_seasonality : bool
        Compute seasonality shape features
    compute_trends : bool
        Compute trend direction features
    compute_importance : bool
        Compute feature importance profiles
    max_external_features : int
        Max external features for response profiles

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (enriched per_key_metrics, metadata about features added)
    """
    logger.info("=" * 60)
    logger.info("ENRICHED SEGMENTATION FEATURES (Phase 2)")
    logger.info("=" * 60)

    enriched = per_key_metrics.copy()
    metadata = {
        'features_added': [],
        'hierarchy_detected': [],
        'external_features_used': [],
        'n_features_before': len(per_key_metrics.columns),
        'n_features_after': 0,
    }

    key_col = key_cols[0] if len(key_cols) == 1 else key_cols

    # Determine history cutoff
    history_cutoff = None
    if feature_availability_context:
        hc = feature_availability_context.get('history_cutoff', {})
        history_cutoff = hc.get('detected', None)
    if not history_cutoff:
        # Auto-detect from data
        period_totals = source_df.groupby(date_col)[target_col].sum()
        positive = period_totals[period_totals > 0]
        if len(positive) > 0:
            history_cutoff = str(positive.index.max())

    logger.info(f"Using history cutoff: {history_cutoff}")

    # Determine external numeric features (from feature availability or auto-detect)
    external_numeric_cols = []
    if feature_availability_context:
        known = feature_availability_context.get('known_in_future_features', [])
        history_only = feature_availability_context.get('history_only_features', [])
        external_numeric_cols = known + history_only  # All features (both known and history-only)
        logger.info(f"Using {len(external_numeric_cols)} features from availability context")
    else:
        exclude = set(key_cols + [date_col, target_col])
        external_numeric_cols = [
            c for c in source_df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(source_df[c])
        ]
        logger.info(f"Auto-detected {len(external_numeric_cols)} numeric features from data")

    metadata['external_features_used'] = external_numeric_cols[:20]

    seasonal_period = 12 if time_format == 'year_month' else 52

    # -------------------------------------------------------------------------
    # 1. External Response Profiles
    # -------------------------------------------------------------------------
    if compute_response_profiles and external_numeric_cols:
        logger.info("\n[1/5] Computing external response profiles...")
        try:
            response_df = compute_external_response_profiles(
                source_df=source_df,
                key_cols=key_cols,
                date_col=date_col,
                target_col=target_col,
                external_numeric_cols=external_numeric_cols,
                history_cutoff=history_cutoff,
                max_features=max_external_features,
            )
            if len(response_df) > 0:
                new_cols = [c for c in response_df.columns if c not in key_cols]
                enriched = enriched.merge(response_df, on=key_col, how='left')
                for c in new_cols:
                    if pd.api.types.is_numeric_dtype(enriched[c]):
                        enriched[c] = enriched[c].fillna(0)
                metadata['features_added'].extend(new_cols)
                logger.info(f"  Added {len(new_cols)} response profile features")
        except Exception as e:
            logger.warning(f"  External response profiles failed: {e}")
    else:
        logger.info("\n[1/5] Skipping external response profiles (no external features)")

    # -------------------------------------------------------------------------
    # 2. Seasonality Shape Features
    # -------------------------------------------------------------------------
    if compute_seasonality:
        logger.info("\n[2/5] Computing seasonality shape features...")
        try:
            season_df = compute_seasonality_shape_features(
                source_df=source_df,
                key_cols=key_cols,
                date_col=date_col,
                target_col=target_col,
                seasonal_period=seasonal_period,
                n_harmonics=3,
                history_cutoff=history_cutoff,
            )
            if len(season_df) > 0:
                new_cols = [c for c in season_df.columns if c not in key_cols]
                enriched = enriched.merge(season_df, on=key_col, how='left')
                for c in new_cols:
                    if pd.api.types.is_numeric_dtype(enriched[c]):
                        enriched[c] = enriched[c].fillna(0)
                metadata['features_added'].extend(new_cols)
                logger.info(f"  Added {len(new_cols)} seasonality shape features")
        except Exception as e:
            logger.warning(f"  Seasonality shape features failed: {e}")

    # -------------------------------------------------------------------------
    # 3. Trend Direction Features
    # -------------------------------------------------------------------------
    if compute_trends:
        logger.info("\n[3/5] Computing trend direction features...")
        try:
            trend_df = compute_trend_direction_features(
                source_df=source_df,
                key_cols=key_cols,
                date_col=date_col,
                target_col=target_col,
                history_cutoff=history_cutoff,
                recent_window=seasonal_period // 4,  # Quarterly for weekly, monthly for monthly
            )
            if len(trend_df) > 0:
                new_cols = [c for c in trend_df.columns if c not in key_cols]
                enriched = enriched.merge(trend_df, on=key_col, how='left')
                for c in new_cols:
                    if pd.api.types.is_numeric_dtype(enriched[c]):
                        enriched[c] = enriched[c].fillna(0)
                metadata['features_added'].extend(new_cols)
                logger.info(f"  Added {len(new_cols)} trend direction features")
        except Exception as e:
            logger.warning(f"  Trend direction features failed: {e}")

    # -------------------------------------------------------------------------
    # 4. Hierarchy-Aware Features
    # -------------------------------------------------------------------------
    detected_hierarchy = hierarchy_cols or []
    if auto_detect_hierarchy and not detected_hierarchy:
        logger.info("\n[4/5] Auto-detecting hierarchy columns...")
        try:
            candidate_cats = [c for c in source_df.columns
                              if c not in set(key_cols + [date_col, target_col])
                              and (source_df[c].dtype == 'object' or source_df[c].nunique() < 500)]
            hier_info = detect_hierarchy_columns(
                source_df=source_df,
                key_cols=key_cols,
                date_col=date_col,
                target_col=target_col,
                candidate_cols=candidate_cats,
            )
            detected_hierarchy = [h['column'] for h in hier_info[:5]]  # Top 5
            metadata['hierarchy_detected'] = [
                {'column': h['column'], 'cardinality': h['cardinality']} for h in hier_info[:5]
            ]
        except Exception as e:
            logger.warning(f"  Hierarchy detection failed: {e}")

    if detected_hierarchy:
        logger.info(f"[4/5] Computing hierarchy features for: {detected_hierarchy}")
        try:
            enriched = compute_hierarchy_features(
                per_key_metrics=enriched,
                source_df=source_df,
                key_cols=key_cols,
                date_col=date_col,
                target_col=target_col,
                hierarchy_cols=detected_hierarchy,
                history_cutoff=history_cutoff,
            )
            hier_cols = [c for c in enriched.columns if c.startswith('hier_')]
            metadata['features_added'].extend(hier_cols)
            logger.info(f"  Added {len(hier_cols)} hierarchy features")
        except Exception as e:
            logger.warning(f"  Hierarchy features failed: {e}")
    else:
        logger.info("\n[4/5] No hierarchy columns detected/provided")

    # -------------------------------------------------------------------------
    # 5. Feature Importance Profiles (optional — expensive)
    # -------------------------------------------------------------------------
    if compute_importance and len(external_numeric_cols) >= 3:
        logger.info("\n[5/5] Computing feature importance profiles...")
        try:
            imp_df = compute_feature_importance_profiles(
                source_df=source_df,
                key_cols=key_cols,
                date_col=date_col,
                target_col=target_col,
                feature_cols=external_numeric_cols,
                history_cutoff=history_cutoff,
                sample_keys=min(200, per_key_metrics.shape[0]),
            )
            if len(imp_df) > 0:
                new_cols = [c for c in imp_df.columns if c not in key_cols]
                enriched = enriched.merge(imp_df, on=key_col, how='left')
                for c in new_cols:
                    if pd.api.types.is_numeric_dtype(enriched[c]):
                        enriched[c] = enriched[c].fillna(0)
                metadata['features_added'].extend(new_cols)
                logger.info(f"  Added {len(new_cols)} importance profile features")
        except Exception as e:
            logger.warning(f"  Feature importance profiles failed: {e}")
    else:
        logger.info("\n[5/5] Skipping importance profiles (insufficient features)")

    metadata['n_features_after'] = len(enriched.columns)
    n_new = metadata['n_features_after'] - metadata['n_features_before']

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ENRICHMENT COMPLETE: {n_new} new features added")
    logger.info(f"  Total columns: {metadata['n_features_before']} → {metadata['n_features_after']}")
    logger.info(f"  Hierarchy columns: {detected_hierarchy}")
    logger.info(f"{'=' * 60}")

    return enriched, metadata
