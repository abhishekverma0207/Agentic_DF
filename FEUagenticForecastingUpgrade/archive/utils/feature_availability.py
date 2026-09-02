# utils/feature_availability.py
"""
Intelligent Feature Availability Detection for Agentic Demand Forecasting.

This module automatically detects:
1. The history cutoff point (last period with actual demand data)
2. Which features are available in future periods (known_in_future)
3. Which features are only available in history (history_only)
4. Which features are partially available in future (partially_known)

It then produces a feature strategy that guides the feature engineering pipeline
to use each feature type appropriately:
- known_in_future: Use directly as regressors in train AND inference
- history_only: Create frozen embeddings (key-level aggregates, trends, correlations)
- partially_known: Use directly where available, impute intelligently beyond

This is a DETERMINISTIC module — no LLM calls. It uses data inspection only.

Usage:
    from utils.feature_availability import run_feature_availability_pipeline

    result = run_feature_availability_pipeline(
        df=source_df,
        key_cols=['key'],
        date_col='year_week',
        target_col='psbdl_actual_sales',
        feature_cols=['price_per_unit', 'promo_flag', 'store_count', ...],
        time_format='year_week',
        config_train_end='202521',  # Optional: validate against auto-detected cutoff
    )
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Thresholds for feature availability classification
FUTURE_KNOWN_MIN_FILL_RATE = 0.50   # >=50% non-null/non-zero in future → known_in_future
FUTURE_PARTIAL_MIN_FILL_RATE = 0.10  # 10-50% → partially_known
# Below 10% → history_only

# Minimum non-null rate in history to consider a feature useful at all
HISTORY_MIN_USEFUL_RATE = 0.05  # At least 5% non-null in history

# Columns that are always considered known in future (calendar-derived)
ALWAYS_KNOWN_PATTERNS = [
    'year', 'month', 'quarter', 'week', 'day', 'is_month', 'is_quarter',
    'is_year', 'fourier', 'sin_', 'cos_', 'holiday', 'season',
]

# Columns that are typically history-only (actuals of related series)
TYPICALLY_HISTORY_PATTERNS = [
    'actual', 'sales', 'demand', 'qty', 'units', 'revenue', 'volume',
]


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass
class FeatureAvailabilityResult:
    """Result from feature availability detection pipeline."""

    # Detected cutoff
    detected_history_cutoff: str = ""
    config_train_end: str = ""
    cutoff_match: bool = True  # True if detected matches config

    # Feature classifications
    known_in_future: List[str] = field(default_factory=list)
    history_only: List[str] = field(default_factory=list)
    partially_known: List[str] = field(default_factory=list)
    excluded_features: List[str] = field(default_factory=list)  # Too sparse in history

    # Detailed per-feature info
    feature_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Frozen embedding recommendations for history-only features
    frozen_embedding_features: Dict[str, List[str]] = field(default_factory=dict)

    # Summary statistics
    n_total_features: int = 0
    n_known_in_future: int = 0
    n_history_only: int = 0
    n_partially_known: int = 0
    n_excluded: int = 0

    # Metadata
    generated_at: str = ""
    n_keys: int = 0
    n_periods_history: int = 0
    n_periods_future: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Feature Availability Detection Summary",
            f"=" * 50,
            f"History cutoff (detected): {self.detected_history_cutoff}",
            f"History cutoff (config):   {self.config_train_end}",
            f"Cutoff match: {self.cutoff_match}",
            f"",
            f"Total features analyzed: {self.n_total_features}",
            f"  Known in future:    {self.n_known_in_future} ({', '.join(self.known_in_future[:5])}{'...' if len(self.known_in_future) > 5 else ''})",
            f"  History only:       {self.n_history_only} ({', '.join(self.history_only[:5])}{'...' if len(self.history_only) > 5 else ''})",
            f"  Partially known:    {self.n_partially_known} ({', '.join(self.partially_known[:5])}{'...' if len(self.partially_known) > 5 else ''})",
            f"  Excluded (sparse):  {self.n_excluded} ({', '.join(self.excluded_features[:5])}{'...' if len(self.excluded_features) > 5 else ''})",
            f"",
            f"History periods: {self.n_periods_history}",
            f"Future periods:  {self.n_periods_future}",
            f"Keys: {self.n_keys}",
        ]
        return "\n".join(lines)


@dataclass
class FrozenEmbeddingSpec:
    """Specification for a frozen embedding feature derived from a history-only feature."""
    source_feature: str
    embedding_type: str  # 'mean', 'std', 'trend', 'correlation', 'last_known', 'median'
    output_feature_name: str
    description: str


# =============================================================================
# CORE DETECTION FUNCTIONS
# =============================================================================

def detect_history_cutoff(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    time_format: str = 'year_week',
) -> Tuple[str, Dict[str, Any]]:
    """
    Auto-detect the history cutoff point from data.

    The cutoff is the LAST period where target has any positive values
    across ALL keys (i.e., where total demand > 0).

    Parameters
    ----------
    df : pd.DataFrame
        Source data with all periods (history + future)
    key_cols : List[str]
        Columns defining unique time series
    date_col : str
        Time period column
    target_col : str
        Demand/target column
    time_format : str
        'year_week' or 'year_month' or 'date'

    Returns
    -------
    Tuple[str, Dict]
        (cutoff_value, metadata_dict)
    """
    # Get total demand per period across all keys
    period_totals = df.groupby(date_col)[target_col].sum().reset_index()
    period_totals = period_totals.sort_values(date_col)

    # Find last period with positive total demand
    positive_periods = period_totals[period_totals[target_col] > 0]

    if positive_periods.empty:
        logger.warning("No periods with positive demand found!")
        cutoff = str(period_totals[date_col].max())
    else:
        cutoff = str(positive_periods[date_col].max())

    # Compute metadata
    all_periods = sorted(df[date_col].unique())
    cutoff_idx = list(all_periods).index(positive_periods[date_col].max()) if not positive_periods.empty else len(all_periods) - 1

    n_history_periods = cutoff_idx + 1
    n_future_periods = len(all_periods) - n_history_periods

    # Check for "tail" of zero demand (dead periods at end)
    last_few = period_totals.tail(10)
    zero_tail_count = 0
    for _, row in last_few.iloc[::-1].iterrows():
        if row[target_col] <= 0:
            zero_tail_count += 1
        else:
            break

    metadata = {
        'cutoff_value': cutoff,
        'n_total_periods': len(all_periods),
        'n_history_periods': n_history_periods,
        'n_future_periods': n_future_periods,
        'first_period': str(all_periods[0]),
        'last_period': str(all_periods[-1]),
        'zero_tail_periods': zero_tail_count,
        'total_demand_at_cutoff': float(positive_periods[target_col].iloc[-1]) if not positive_periods.empty else 0.0,
    }

    logger.info(f"Detected history cutoff: {cutoff}")
    logger.info(f"  History periods: {n_history_periods}, Future periods: {n_future_periods}")
    if zero_tail_count > 0:
        logger.info(f"  Zero-demand tail: {zero_tail_count} periods at end of data")

    return cutoff, metadata


def classify_feature_availability(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    history_cutoff: str,
    key_cols: Optional[List[str]] = None,
    future_known_threshold: float = FUTURE_KNOWN_MIN_FILL_RATE,
    partial_threshold: float = FUTURE_PARTIAL_MIN_FILL_RATE,
    history_useful_threshold: float = HISTORY_MIN_USEFUL_RATE,
    fa_config: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Classify features by availability in future periods (v2 — per-key aware).

    Rewrite of the original classifier that fixes three issues found in
    production runs:

    1. **Pattern override is first-class** — calendar/holiday/promo name
       patterns promote a feature to ``known_in_future`` before any
       sparsity gate runs. The old code checked sparsity first and
       ``continue``-d, so the override never ran for excluded features
       (e.g. ``holiday_christmas_day`` always got dropped).
    2. **Auto-indicator detection** — binary / low-cardinality categorical
       features are classified by their future presence instead of by a
       fill-rate threshold. A promo flag that fires 3% of the time stays
       a valuable regressor.
    3. **Per-key active-scope** — continuous features are scored on the
       fraction of *history-active* keys that also have future values,
       not on the global fill rate. This keeps features that are sparse
       across the catalogue but dense for the keys that actually use them.

    An optional ``fa_config`` (a ``FeatureAvailabilityConfig`` from
    ``config/schema.py``) overrides every threshold and the pattern list.
    When omitted, the legacy default thresholds are used so existing
    callers keep working unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Source data spanning history + future periods.
    date_col, target_col : str
        Period and target column names.
    feature_cols : List[str]
        Candidate feature columns to classify.
    history_cutoff : str
        Detected history cutoff value (all rows with period ≤ cutoff are
        considered history).
    key_cols : List[str], optional
        Key columns (excluded from classification).
    future_known_threshold, partial_threshold, history_useful_threshold :
        Legacy thresholds, ignored when ``fa_config`` is supplied.
    fa_config : FeatureAvailabilityConfig, optional
        Advanced tuning from config. When provided, the per-key scope and
        recency-gate logic are enabled and all thresholds come from this
        object.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Per-feature classification details.
    """
    from utils.period_utils import normalise_period, normalise_period_column

    # ------------------------------------------------------------------
    # 1. Resolve thresholds and patterns from either fa_config or legacy args
    # ------------------------------------------------------------------
    cfg = fa_config  # alias
    if cfg is not None:
        patterns = [p.lower() for p in cfg.always_known_patterns]
        cov_known = cfg.forward_coverage_known
        cov_partial = cfg.forward_coverage_partial
        min_active_keys = cfg.min_active_keys
        auto_indicator_enabled = cfg.auto_indicator_enabled
        max_obj_unique = cfg.max_unique_values_object
        treat_binary_indicator = cfg.treat_binary_as_indicator
        ind_min_fut_rows = cfg.indicator_min_future_nonzero_rows
        ind_min_hist_rows = cfg.indicator_min_history_nonzero_rows
        recency_enabled = cfg.recency_enabled
        recency_lookback = cfg.recency_lookback_periods
        recency_min_keys = cfg.recency_min_active_keys
        recency_min_rows = cfg.recency_min_nonzero_rows
    else:
        patterns = [p.lower() for p in ALWAYS_KNOWN_PATTERNS]
        # Legacy thresholds fall through to the passed-in arguments.
        cov_known = future_known_threshold
        cov_partial = partial_threshold
        min_active_keys = 1  # legacy: any active key keeps the feature
        auto_indicator_enabled = False
        max_obj_unique = 10
        treat_binary_indicator = False
        ind_min_fut_rows = 1
        ind_min_hist_rows = 1
        recency_enabled = False
        recency_lookback = 4
        recency_min_keys = 1
        recency_min_rows = 1

    exclude_cols = set([date_col, target_col] + (key_cols or []))
    features_to_check = [c for c in feature_cols if c in df.columns and c not in exclude_cols]

    # ------------------------------------------------------------------
    # 2. Slice history / future ONCE
    # ------------------------------------------------------------------
    df = normalise_period_column(df, date_col)
    cutoff_val = normalise_period(history_cutoff)

    # The date column may still be int64 if the source is all-numeric
    # (e.g. 202614). Compare via a string projection so cutoff_val (a string)
    # is comparable for every period value we ever see.
    _period_str = df[date_col].astype(str).map(normalise_period)
    history_mask = _period_str <= cutoff_val
    future_mask = _period_str > cutoff_val
    history_df = df[history_mask]
    future_df = df[future_mask]

    n_history_rows = len(history_df)
    n_future_rows = len(future_df)

    # --- Recency window: last N periods of history (sorted chronologically) ---
    recent_history_df: Optional[pd.DataFrame] = None
    if recency_enabled and n_history_rows > 0 and recency_lookback > 0:
        hist_periods_norm = _period_str[history_mask]
        hist_periods_sorted = sorted(hist_periods_norm.unique())
        last_n = set(hist_periods_sorted[-recency_lookback:])
        recent_history_df = history_df[hist_periods_norm.isin(last_n).values]

    # --- Resolve key column for per-key analysis ---
    effective_key = None
    if key_cols:
        for kc in key_cols:
            if kc in df.columns:
                effective_key = kc
                break
    if effective_key is None and 'key' in df.columns:
        effective_key = 'key'

    logger.info(
        "Classifying %d features (history=%d rows, future=%d rows, per-key=%s, recency=%s)",
        len(features_to_check), n_history_rows, n_future_rows,
        bool(effective_key), recency_enabled,
    )

    results: Dict[str, Dict[str, Any]] = {}

    for col in features_to_check:
        info: Dict[str, Any] = {
            'column': col,
            'dtype': str(df[col].dtype),
            'classification': 'unknown',
        }

        # ---- Type inspection --------------------------------------------
        series_all = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series_all)
        is_bool = pd.api.types.is_bool_dtype(series_all) or series_all.dtype == bool
        dropped = series_all.dropna()

        # Auto-indicator detection
        is_indicator = False
        if auto_indicator_enabled:
            if is_bool:
                is_indicator = True
            elif is_numeric and treat_binary_indicator and len(dropped) > 0:
                unique_vals = np.unique(dropped.to_numpy())
                # Accept {0} / {1} / {0,1} / {0.0,1.0}
                is_indicator = bool(
                    np.all(np.isin(unique_vals, [0, 1]))
                    or np.all(np.isin(unique_vals, [0.0, 1.0]))
                )
            elif not is_numeric:
                # Object / categorical with few unique values
                try:
                    nu = int(dropped.nunique())
                    is_indicator = nu <= max_obj_unique
                except Exception:
                    is_indicator = False

        # Pattern-match on column name
        col_lower = col.lower()
        pattern_matched: Optional[str] = None
        for pat in patterns:
            if pat in col_lower:
                pattern_matched = pat
                break

        # ---- Global / history stats --------------------------------------
        if n_history_rows > 0:
            hist_values = history_df[col]
            hist_non_null = int(hist_values.notna().sum())
            hist_fill_rate = hist_non_null / n_history_rows
            if is_numeric:
                hist_non_zero = int((hist_values.notna() & (hist_values != 0)).sum())
                hist_useful_rate = hist_non_zero / n_history_rows
            else:
                hist_non_zero = hist_non_null
                hist_useful_rate = hist_fill_rate
        else:
            hist_non_null = hist_non_zero = 0
            hist_fill_rate = hist_useful_rate = 0.0

        if n_future_rows > 0:
            fut_values = future_df[col]
            fut_non_null = int(fut_values.notna().sum())
            fut_fill_rate = fut_non_null / n_future_rows
            if is_numeric:
                fut_non_zero = int((fut_values.notna() & (fut_values != 0)).sum())
                fut_useful_rate = fut_non_zero / n_future_rows
            else:
                fut_non_zero = fut_non_null
                fut_useful_rate = fut_fill_rate
        else:
            fut_non_null = fut_non_zero = 0
            fut_fill_rate = fut_useful_rate = None

        info.update({
            'history_fill_rate': round(float(hist_fill_rate), 4),
            'history_useful_rate': round(float(hist_useful_rate), 4),
            'history_non_null_count': hist_non_null,
            'history_nonzero_count': hist_non_zero,
            'future_fill_rate': (round(float(fut_fill_rate), 4) if fut_fill_rate is not None else None),
            'future_useful_rate': (round(float(fut_useful_rate), 4) if fut_useful_rate is not None else None),
            'future_non_null_count': fut_non_null,
            'future_nonzero_count': fut_non_zero,
            'is_indicator': bool(is_indicator),
            'pattern_matched': pattern_matched,
        })

        # ---- Per-key active scope (continuous features) -----------------
        n_active_keys_hist = 0
        n_active_keys_fut = 0
        forward_key_coverage = 0.0
        if effective_key is not None and n_history_rows > 0:
            hist_nz_mask = (history_df[col].notna() & (history_df[col] != 0)) \
                if is_numeric else history_df[col].notna()
            hist_active_keys = set(history_df.loc[hist_nz_mask, effective_key].unique())
            n_active_keys_hist = len(hist_active_keys)
            if n_future_rows > 0 and n_active_keys_hist > 0:
                fut_nz_mask = (future_df[col].notna() & (future_df[col] != 0)) \
                    if is_numeric else future_df[col].notna()
                fut_active_keys = set(future_df.loc[fut_nz_mask, effective_key].unique())
                n_active_keys_fut = len(fut_active_keys & hist_active_keys)
                forward_key_coverage = n_active_keys_fut / max(1, n_active_keys_hist)
        info['n_active_keys_history'] = n_active_keys_hist
        info['n_active_keys_future'] = n_active_keys_fut
        info['forward_key_coverage'] = round(float(forward_key_coverage), 4)

        # ---- Recency gate (precomputed once per column) -----------------
        recency_passed = True
        if recency_enabled and recent_history_df is not None and effective_key is not None:
            recent_vals = recent_history_df[col]
            rmask = (recent_vals.notna() & (recent_vals != 0)) if is_numeric else recent_vals.notna()
            r_rows = int(rmask.sum())
            r_keys = int(recent_history_df.loc[rmask, effective_key].nunique())
            info['n_active_keys_recent'] = r_keys
            info['n_nonzero_rows_recent'] = r_rows
            recency_passed = (r_keys >= recency_min_keys) and (r_rows >= recency_min_rows)
        info['recency_passed'] = bool(recency_passed)

        # ==================================================================
        # 3. CLASSIFICATION CASCADE — first match wins
        # ==================================================================

        # a) Name pattern whitelist runs FIRST (this is the bug fix).
        if pattern_matched is not None and (hist_non_zero + fut_non_zero) > 0:
            info['classification'] = 'known_in_future'
            info['reason'] = (
                f"Pattern override ('{pattern_matched}') with "
                f"{hist_non_zero + fut_non_zero} non-zero observations"
            )
            results[col] = info
            continue

        # Pattern matched but zero signal anywhere — still safer to keep as
        # known_in_future so downstream calendar synthesis (or a human fix)
        # can populate it without needing a config change.
        if pattern_matched is not None:
            info['classification'] = 'known_in_future'
            info['reason'] = (
                f"Pattern override ('{pattern_matched}') even though no "
                f"non-zero rows observed (calendar/promo synth may populate)"
            )
            results[col] = info
            continue

        # b) No signal at all → excluded
        if hist_non_zero == 0 and fut_non_zero == 0:
            info['classification'] = 'excluded'
            info['reason'] = "No non-zero values in either history or future"
            results[col] = info
            continue

        # c) Auto-indicator: classify by future presence, not by fill rate.
        if is_indicator:
            if fut_non_zero >= ind_min_fut_rows:
                info['classification'] = 'known_in_future'
                info['reason'] = (
                    f"Auto-indicator with {fut_non_zero} non-zero future rows "
                    f"across {n_active_keys_fut} keys"
                )
                results[col] = info
                continue
            # No future presence — decide between history_only and excluded.
            if hist_non_zero >= ind_min_hist_rows and n_active_keys_hist >= min_active_keys:
                if recency_enabled and not recency_passed:
                    info['classification'] = 'excluded'
                    info['reason'] = (
                        f"Stale indicator: only {info.get('n_active_keys_recent', 0)} "
                        f"active keys in last {recency_lookback} history periods"
                    )
                else:
                    info['classification'] = 'history_only'
                    info['reason'] = (
                        f"Indicator with history signal but no future presence "
                        f"({n_active_keys_hist} history-active keys)"
                    )
                results[col] = info
                continue
            info['classification'] = 'excluded'
            info['reason'] = (
                f"Indicator with insufficient history signal "
                f"(active_keys={n_active_keys_hist}, nonzero_rows={hist_non_zero})"
            )
            results[col] = info
            continue

        # d) Continuous feature branch — use per-key coverage if available.
        use_per_key = effective_key is not None and cfg is not None
        if use_per_key:
            if n_active_keys_hist < min_active_keys:
                info['classification'] = 'excluded'
                info['reason'] = (
                    f"Only {n_active_keys_hist} active history keys "
                    f"(< min_active_keys={min_active_keys})"
                )
                results[col] = info
                continue

            if forward_key_coverage >= cov_known:
                info['classification'] = 'known_in_future'
                info['reason'] = (
                    f"Continuous feature with forward_key_coverage="
                    f"{forward_key_coverage:.1%} ≥ {cov_known:.0%}"
                )
                results[col] = info
                continue

            if forward_key_coverage >= cov_partial:
                info['classification'] = 'partially_known'
                info['reason'] = (
                    f"Continuous feature with forward_key_coverage="
                    f"{forward_key_coverage:.1%} in "
                    f"[{cov_partial:.0%}, {cov_known:.0%})"
                )
                # Determine how far forward the feature is populated
                try:
                    future_period_str = _period_str[future_mask]
                    future_periods = sorted(future_period_str.unique())
                    last_available = None
                    for period in future_periods:
                        pm = (future_period_str == period).values
                        pdata = future_df.loc[pm, col]
                        any_nz = (pdata.notna() & (pdata != 0)).any() if is_numeric else pdata.notna().any()
                        if any_nz:
                            last_available = period
                        else:
                            break
                    if last_available is not None:
                        info['known_until'] = str(last_available)
                except Exception:
                    pass
                results[col] = info
                continue

            # Falls through to history_only path below
            # (with optional recency gate).

        # Legacy / fallback path for callers without fa_config.
        if not use_per_key:
            # Legacy: fill-rate against history_useful_threshold.
            if hist_useful_rate < history_useful_threshold:
                info['classification'] = 'excluded'
                info['reason'] = (
                    f"Too sparse in history (useful_rate="
                    f"{hist_useful_rate:.2%} < {history_useful_threshold:.0%})"
                )
                results[col] = info
                continue
            if fut_useful_rate is None:
                info['classification'] = 'known_in_future'
                info['reason'] = 'No future periods in data — assuming available'
            elif fut_useful_rate >= future_known_threshold:
                info['classification'] = 'known_in_future'
                info['reason'] = (
                    f"Available in future (useful_rate={fut_useful_rate:.2%} "
                    f">= {future_known_threshold:.0%})"
                )
            elif fut_useful_rate >= partial_threshold:
                info['classification'] = 'partially_known'
                info['reason'] = f"Partially available in future (useful_rate={fut_useful_rate:.2%})"
            else:
                info['classification'] = 'history_only'
                info['reason'] = f"Not available in future (useful_rate={fut_useful_rate:.2%} < {partial_threshold:.0%})"
            results[col] = info
            continue

        # e) history_only with recency gate
        if recency_enabled and not recency_passed:
            info['classification'] = 'excluded'
            info['reason'] = (
                f"Stale feed: only {info.get('n_active_keys_recent', 0)} "
                f"active keys and {info.get('n_nonzero_rows_recent', 0)} "
                f"non-zero rows in last {recency_lookback} history periods"
            )
            results[col] = info
            continue

        info['classification'] = 'history_only'
        info['reason'] = (
            f"Continuous feature present in history but not forward "
            f"(active_keys_hist={n_active_keys_hist}, "
            f"forward_coverage={forward_key_coverage:.1%})"
        )
        results[col] = info

    # ------------------------------------------------------------------
    # 4. Summary logging
    # ------------------------------------------------------------------
    classifications: Dict[str, List[str]] = {}
    for col, info in results.items():
        cls = info['classification']
        classifications.setdefault(cls, []).append(col)
    for cls, cols in sorted(classifications.items()):
        logger.info(f"  {cls}: {len(cols)} features")

    return results


def generate_frozen_embeddings(
    feature_details: Dict[str, Dict[str, Any]],
    history_only_features: List[str],
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    history_cutoff: str,
) -> Dict[str, List[FrozenEmbeddingSpec]]:
    """
    Generate specifications for frozen embeddings from history-only features.

    For each history-only feature, determines which aggregate representations
    should be created as static key-level features:
    - mean: average value during training period
    - std: volatility during training period
    - trend: direction of change (slope of linear fit)
    - correlation: correlation with target during training
    - last_known: last non-null value before cutoff
    - median: median value during training

    Parameters
    ----------
    feature_details : Dict
        Per-feature classification from classify_feature_availability()
    history_only_features : List[str]
        Features classified as history_only
    df : pd.DataFrame
        Source data
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    history_cutoff : str
        History cutoff value

    Returns
    -------
    Dict[str, List[FrozenEmbeddingSpec]]
        Mapping from source feature to list of embedding specs
    """
    # Align cutoff dtype with the actual date column so we never cross
    # str / int. The function's docstring contract is
    # ``history_cutoff: str``, but the column may be int64 (numeric
    # year_week from a re-read CSV) or object (string year_week after
    # preprocessing). Dispatching on the column dtype keeps the
    # comparison well-defined in both regimes.
    _col_series = df[date_col]
    if pd.api.types.is_integer_dtype(_col_series):
        try:
            cutoff_val = int(history_cutoff)
        except (ValueError, TypeError):
            cutoff_val = history_cutoff
    elif pd.api.types.is_float_dtype(_col_series):
        try:
            cutoff_val = float(history_cutoff)
        except (ValueError, TypeError):
            cutoff_val = history_cutoff
    else:
        cutoff_val = str(history_cutoff)

    history_df = df[df[date_col] <= cutoff_val].copy()
    embeddings = {}

    for feat in history_only_features:
        if feat not in history_df.columns:
            continue

        specs = []

        is_numeric = pd.api.types.is_numeric_dtype(history_df[feat])

        if is_numeric:
            # Mean embedding
            specs.append(FrozenEmbeddingSpec(
                source_feature=feat,
                embedding_type='mean',
                output_feature_name=f'{feat}_hist_mean',
                description=f'Historical mean of {feat} per key (frozen at cutoff)',
            ))

            # Std embedding (captures volatility)
            specs.append(FrozenEmbeddingSpec(
                source_feature=feat,
                embedding_type='std',
                output_feature_name=f'{feat}_hist_std',
                description=f'Historical std of {feat} per key (frozen at cutoff)',
            ))

            # Trend embedding (slope of linear fit)
            specs.append(FrozenEmbeddingSpec(
                source_feature=feat,
                embedding_type='trend',
                output_feature_name=f'{feat}_hist_trend',
                description=f'Historical trend (slope) of {feat} per key',
            ))

            # Correlation with target
            specs.append(FrozenEmbeddingSpec(
                source_feature=feat,
                embedding_type='correlation',
                output_feature_name=f'{feat}_target_corr',
                description=f'Historical correlation of {feat} with {target_col} per key',
            ))

            # Last known value
            specs.append(FrozenEmbeddingSpec(
                source_feature=feat,
                embedding_type='last_known',
                output_feature_name=f'{feat}_last_known',
                description=f'Last non-null value of {feat} before cutoff per key',
            ))

        else:
            # Categorical: mode embedding
            specs.append(FrozenEmbeddingSpec(
                source_feature=feat,
                embedding_type='mode',
                output_feature_name=f'{feat}_hist_mode',
                description=f'Most frequent value of {feat} per key during history',
            ))

            # Unique count embedding
            specs.append(FrozenEmbeddingSpec(
                source_feature=feat,
                embedding_type='nunique',
                output_feature_name=f'{feat}_hist_nunique',
                description=f'Number of unique values of {feat} per key during history',
            ))

        embeddings[feat] = specs

    return embeddings


def compute_frozen_embedding_features(
    df: pd.DataFrame,
    embedding_specs: Dict[str, List[FrozenEmbeddingSpec]],
    key_cols: List[str],
    date_col: str,
    target_col: str,
    history_cutoff: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Actually compute frozen embedding features and merge them into the DataFrame.

    These are computed on HISTORY data only, then merged to ALL rows (including future).
    This ensures no leakage — the embeddings are static, computed once at cutoff time.

    Parameters
    ----------
    df : pd.DataFrame
        Full source data
    embedding_specs : Dict[str, List[FrozenEmbeddingSpec]]
        Specs from generate_frozen_embeddings()
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    history_cutoff : str
        Cutoff value

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (DataFrame with new columns, list of new column names)
    """
    # Align cutoff dtype with the actual date column so we never cross
    # str / int. The function's docstring contract is
    # ``history_cutoff: str``, but the column may be int64 (numeric
    # year_week from a re-read CSV) or object (string year_week after
    # preprocessing). Dispatching on the column dtype keeps the
    # comparison well-defined in both regimes.
    _col_series = df[date_col]
    if pd.api.types.is_integer_dtype(_col_series):
        try:
            cutoff_val = int(history_cutoff)
        except (ValueError, TypeError):
            cutoff_val = history_cutoff
    elif pd.api.types.is_float_dtype(_col_series):
        try:
            cutoff_val = float(history_cutoff)
        except (ValueError, TypeError):
            cutoff_val = history_cutoff
    else:
        cutoff_val = str(history_cutoff)

    history_df = df[df[date_col] <= cutoff_val].copy()
    new_columns = []
    key_col_str = key_cols[0] if len(key_cols) == 1 else key_cols

    for source_feat, specs in embedding_specs.items():
        if source_feat not in history_df.columns:
            logger.warning(f"Source feature '{source_feat}' not in DataFrame, skipping embeddings")
            continue

        for spec in specs:
            col_name = spec.output_feature_name

            try:
                if spec.embedding_type == 'mean':
                    agg = history_df.groupby(key_cols)[source_feat].mean()
                elif spec.embedding_type == 'std':
                    agg = history_df.groupby(key_cols)[source_feat].std().fillna(0)
                elif spec.embedding_type == 'median':
                    agg = history_df.groupby(key_cols)[source_feat].median()
                elif spec.embedding_type == 'last_known':
                    agg = history_df.sort_values(date_col).groupby(key_cols)[source_feat].last()
                elif spec.embedding_type == 'trend':
                    # Compute per-key linear trend slope
                    def _compute_slope(group):
                        vals = group[source_feat].dropna().values
                        if len(vals) < 3:
                            return 0.0
                        x = np.arange(len(vals), dtype=float)
                        try:
                            slope = np.polyfit(x, vals, 1)[0]
                            return float(slope)
                        except (np.linalg.LinAlgError, ValueError):
                            return 0.0

                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        agg = history_df.groupby(key_cols).apply(_compute_slope)

                elif spec.embedding_type == 'correlation':
                    # Per-key correlation with target
                    def _compute_corr(group):
                        if len(group) < 5:
                            return 0.0
                        corr = group[source_feat].corr(group[target_col])
                        return float(corr) if pd.notna(corr) else 0.0

                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        agg = history_df.groupby(key_cols).apply(_compute_corr)

                elif spec.embedding_type == 'mode':
                    agg = history_df.groupby(key_cols)[source_feat].agg(
                        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
                    )

                elif spec.embedding_type == 'nunique':
                    agg = history_df.groupby(key_cols)[source_feat].nunique()

                else:
                    logger.warning(f"Unknown embedding type: {spec.embedding_type}")
                    continue

                # Convert to DataFrame for merging
                agg_df = agg.reset_index()
                if isinstance(agg_df.columns[-1], int):
                    agg_df.columns = list(key_cols) + [col_name]
                else:
                    agg_df = agg_df.rename(columns={agg_df.columns[-1]: col_name})

                # Merge to full DataFrame
                df = df.merge(agg_df[key_cols + [col_name]], on=key_cols, how='left')

                # Fill NaN with 0 for numeric, 'unknown' for categorical
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    df[col_name] = df[col_name].fillna(0)

                new_columns.append(col_name)

            except Exception as e:
                logger.warning(f"Failed to compute embedding {col_name}: {e}")
                continue

    logger.info(f"Computed {len(new_columns)} frozen embedding features from {len(embedding_specs)} history-only features")
    return df, new_columns


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_feature_availability_pipeline(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    time_format: str = 'year_week',
    config_train_end: Optional[str] = None,
    future_known_threshold: float = FUTURE_KNOWN_MIN_FILL_RATE,
    partial_threshold: float = FUTURE_PARTIAL_MIN_FILL_RATE,
    history_useful_threshold: float = HISTORY_MIN_USEFUL_RATE,
    output_dir: Optional[str] = None,
    fa_config: Any = None,
) -> FeatureAvailabilityResult:
    """
    Run the complete feature availability detection pipeline.

    This is the main entry point for the feature availability module.
    It detects the history cutoff, classifies all features, generates
    frozen embedding specs, and saves results.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with all time periods
    key_cols : List[str]
        Columns defining unique time series
    date_col : str
        Time period column
    target_col : str
        Demand/target column
    feature_cols : List[str]
        Feature columns to analyze (all candidate features)
    time_format : str
        'year_week', 'year_month', or 'date'
    config_train_end : str, optional
        Train end from config — used for validation
    future_known_threshold : float
        Threshold for known_in_future classification
    partial_threshold : float
        Threshold for partially_known classification
    history_useful_threshold : float
        Minimum useful rate in history
    output_dir : str, optional
        Directory to save results

    Returns
    -------
    FeatureAvailabilityResult
        Complete detection results
    """
    logger.info("=" * 60)
    logger.info("FEATURE AVAILABILITY DETECTION PIPELINE")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Detect history cutoff
    # -------------------------------------------------------------------------
    logger.info("\n[Step 1/4] Detecting history cutoff...")
    detected_cutoff, cutoff_metadata = detect_history_cutoff(
        df=df,
        key_cols=key_cols,
        date_col=date_col,
        target_col=target_col,
        time_format=time_format,
    )

    # Validate against config
    cutoff_match = True
    if config_train_end:
        # Compare as strings (both should be same format)
        if str(detected_cutoff) != str(config_train_end):
            logger.warning(
                f"Detected cutoff ({detected_cutoff}) differs from config train_end ({config_train_end}). "
                f"Using detected cutoff for feature classification."
            )
            cutoff_match = False
        else:
            logger.info(f"Cutoff matches config train_end: {config_train_end}")

    # -------------------------------------------------------------------------
    # Step 2: Classify feature availability
    # -------------------------------------------------------------------------
    logger.info("\n[Step 2/4] Classifying feature availability...")
    feature_details = classify_feature_availability(
        df=df,
        date_col=date_col,
        target_col=target_col,
        feature_cols=feature_cols,
        history_cutoff=detected_cutoff,
        key_cols=key_cols,
        future_known_threshold=future_known_threshold,
        partial_threshold=partial_threshold,
        history_useful_threshold=history_useful_threshold,
        fa_config=fa_config,
    )

    # Group by classification
    known_in_future = []
    history_only = []
    partially_known = []
    excluded = []

    for col, info in feature_details.items():
        cls = info['classification']
        if cls == 'known_in_future':
            known_in_future.append(col)
        elif cls == 'history_only':
            history_only.append(col)
        elif cls == 'partially_known':
            partially_known.append(col)
        elif cls == 'excluded':
            excluded.append(col)

    # -------------------------------------------------------------------------
    # Step 3: Generate frozen embedding specs for history-only features
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 3/4] Generating frozen embedding specs for {len(history_only)} history-only features...")
    embedding_specs = generate_frozen_embeddings(
        feature_details=feature_details,
        history_only_features=history_only,
        df=df,
        key_cols=key_cols,
        date_col=date_col,
        target_col=target_col,
        history_cutoff=detected_cutoff,
    )

    # Convert embedding specs to serializable format
    frozen_embedding_features = {}
    for source_feat, specs in embedding_specs.items():
        frozen_embedding_features[source_feat] = [
            {
                'source_feature': s.source_feature,
                'embedding_type': s.embedding_type,
                'output_feature_name': s.output_feature_name,
                'description': s.description,
            }
            for s in specs
        ]

    # -------------------------------------------------------------------------
    # Step 4: Build result
    # -------------------------------------------------------------------------
    logger.info("\n[Step 4/4] Building result...")

    n_keys = df[key_cols].drop_duplicates().shape[0] if key_cols else 0

    result = FeatureAvailabilityResult(
        detected_history_cutoff=str(detected_cutoff),
        config_train_end=str(config_train_end) if config_train_end else "",
        cutoff_match=cutoff_match,
        known_in_future=sorted(known_in_future),
        history_only=sorted(history_only),
        partially_known=sorted(partially_known),
        excluded_features=sorted(excluded),
        feature_details={k: v for k, v in feature_details.items()},
        frozen_embedding_features=frozen_embedding_features,
        n_total_features=len(feature_details),
        n_known_in_future=len(known_in_future),
        n_history_only=len(history_only),
        n_partially_known=len(partially_known),
        n_excluded=len(excluded),
        generated_at=datetime.now().isoformat(),
        n_keys=n_keys,
        n_periods_history=cutoff_metadata.get('n_history_periods', 0),
        n_periods_future=cutoff_metadata.get('n_future_periods', 0),
    )

    # -------------------------------------------------------------------------
    # Save results if output_dir provided
    # -------------------------------------------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save full result
        result_path = os.path.join(output_dir, 'feature_availability_result.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved feature availability result to: {result_path}")

        # Save compact context for downstream crews
        context = _build_feature_availability_context(result)
        context_path = os.path.join(output_dir, 'feature_availability_to_feature_context.json')
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2, default=str)
        logger.info(f"Saved feature availability context to: {context_path}")

        # Save human-readable summary
        summary_path = os.path.join(output_dir, 'feature_availability_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(result.summary())
        logger.info(f"Saved summary to: {summary_path}")

    logger.info("\n" + result.summary())
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE AVAILABILITY DETECTION COMPLETE")
    logger.info("=" * 60)

    return result


def _build_feature_availability_context(result: FeatureAvailabilityResult) -> Dict[str, Any]:
    """
    Build a compact context JSON for the Feature Engineering crew.

    This context file is kept under 15 KB to fit in LLM context windows.
    """
    context = {
        'generated_at': result.generated_at,
        'context_type': 'feature_availability_to_feature',
        'source': 'feature_availability_pipeline',
        'target_crews': ['feature_crew'],

        # Cutoff information
        'history_cutoff': {
            'detected': result.detected_history_cutoff,
            'config_train_end': result.config_train_end,
            'match': result.cutoff_match,
            'n_history_periods': result.n_periods_history,
            'n_future_periods': result.n_periods_future,
        },

        # Classification summary
        'summary': {
            'total_features': result.n_total_features,
            'known_in_future': result.n_known_in_future,
            'history_only': result.n_history_only,
            'partially_known': result.n_partially_known,
            'excluded': result.n_excluded,
        },

        # Feature lists (compact)
        'known_in_future_features': result.known_in_future,
        'history_only_features': result.history_only,
        'partially_known_features': result.partially_known,
        'excluded_features': result.excluded_features,

        # Frozen embedding specs (compact: just feature names)
        'frozen_embedding_output_features': [
            spec['output_feature_name']
            for specs_list in result.frozen_embedding_features.values()
            for spec in specs_list
        ],

        # Strategy guidance for Feature Engineering crew
        'feature_strategy': {
            'known_in_future': 'Use directly as features in training AND inference. Apply standard lag/rolling engineering.',
            'history_only': 'Do NOT use raw values. Use frozen embeddings (key-level aggregates computed on history only).',
            'partially_known': 'Use directly where available. For periods beyond known_until, impute with segment-level historical mean.',
            'excluded': 'Do NOT use. These features are too sparse to be informative.',
        },
    }

    return context


# =============================================================================
# CONVENIENCE FUNCTION: Run from config
# =============================================================================

def run_feature_availability_from_config(
    config: Any,
    source_df: Optional[pd.DataFrame] = None,
) -> FeatureAvailabilityResult:
    """
    Run feature availability detection using a DemandForecastConfig object.

    This is the integration entry point used by crews and runners.

    Parameters
    ----------
    config : DemandForecastConfig
        Pipeline configuration
    source_df : pd.DataFrame, optional
        Pre-loaded source data. If None, loads from config.input_data_path.

    Returns
    -------
    FeatureAvailabilityResult
    """
    # Load data if not provided
    if source_df is None:
        from utils.agent_utilities import load_source_data
        logger.info(f"Loading source data from: {config.input_data_path}")
        source_df = load_source_data(config.input_data_path)

    # Inject deterministic calendar features BEFORE classification so the
    # classifier sees holiday / seasonal columns populated in both history
    # and future. Respect design.calendar_features for opt-out.
    _cal_cfg = getattr(getattr(config, 'design', None), 'calendar_features', None)
    _country = getattr(config, 'country', '') or ''
    if _cal_cfg is not None and getattr(_cal_cfg, 'enabled', False) and _country:
        try:
            from utils.calendar_features import inject_calendar_features
            _period_totals_pre = source_df.groupby(config.timestamp_col)[config.target_col].sum()
            _hc_pre = str(_period_totals_pre[_period_totals_pre > 0].index.max())
            source_df = inject_calendar_features(
                source_df,
                date_col=config.timestamp_col,
                time_format=getattr(config, 'time_format', 'year_week'),
                country=_country,
                subdivision=getattr(config, 'country_subdivision', None),
                overwrite_mode=getattr(_cal_cfg, 'overwrite_mode', 'always'),
                history_cutoff=_hc_pre,
                custom_events=getattr(_cal_cfg, 'custom_events', None) or None,
                lead_lag_windows=list(getattr(_cal_cfg, 'include_lead_lag_windows', [1, 2, 4])),
            )
        except Exception as _cal_exc:
            logger.warning("Calendar injection skipped inside FA config path: %s", _cal_exc)

    # Collect all feature columns from config
    all_feature_cols = []

    # Numeric features
    if hasattr(config, 'all_numeric_features'):
        all_feature_cols.extend(config.all_numeric_features())

    # Categorical features
    if hasattr(config, 'all_categorical_features'):
        all_feature_cols.extend(config.all_categorical_features())

    # Also check any additional columns in the DataFrame that aren't key/date/target
    exclude_cols = set(config.prediction_key_cols + [config.timestamp_col, config.target_col])
    for col in source_df.columns:
        if col not in exclude_cols and col not in all_feature_cols:
            all_feature_cols.append(col)

    # Remove duplicates while preserving order
    seen = set()
    unique_feature_cols = []
    for col in all_feature_cols:
        if col not in seen and col in source_df.columns:
            seen.add(col)
            unique_feature_cols.append(col)

    # Output directory
    output_dir = os.path.join(config.artifact_base_path, 'feature_availability_output')

    # Run pipeline — pass the per-design fa_config so the rewritten
    # classifier uses per-key scope + recency gate + pattern whitelist.
    _fa_cfg = getattr(getattr(config, 'design', None), 'feature_availability', None)

    # Refresh the candidate feature list AFTER calendar injection so any
    # newly added holiday / season / Fourier columns are scored.
    if _cal_cfg is not None and getattr(_cal_cfg, 'enabled', False) and _country:
        existing = set(unique_feature_cols)
        for col in source_df.columns:
            if col in existing or col in exclude_cols:
                continue
            unique_feature_cols.append(col)

    result = run_feature_availability_pipeline(
        df=source_df,
        key_cols=config.prediction_key_cols,
        date_col=config.timestamp_col,
        target_col=config.target_col,
        feature_cols=unique_feature_cols,
        time_format=getattr(config, 'time_format', 'year_week'),
        config_train_end=getattr(config, 'train_end', None),
        output_dir=output_dir,
        fa_config=_fa_cfg,
    )

    return result
