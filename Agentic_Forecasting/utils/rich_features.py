# utils/rich_features.py
"""
Phase 6: Richer Feature Engineering.

Three categories of features that significantly improve forecast accuracy:

1. **Regime/State Features** — Detect what regime each key is in at each
   time point: growth vs stable vs decline, high vs low volatility, seasonal
   phase. Models learn different patterns for different regimes.

2. **Cross-Key Relative Features** — Features measuring each key's performance
   relative to its peers (segment, hierarchy group). Captures competitive
   dynamics, market share shifts, and relative momentum.

3. **History-Only Feature Embeddings** — For features classified as
   `future_unknown` by Feature Availability Detection, convert them into
   static key-level embeddings (historical mean, std, trend, correlation
   with target). This preserves their signal without needing future values.

All features are LEAKAGE-FREE: use shift(forecast_lag) to prevent look-ahead.
All are STATIC during recursive forecasting (computed from actuals only).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 1. REGIME / STATE FEATURES
# =============================================================================

def create_regime_features(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    forecast_lag: int,
    short_window: int = 4,
    long_window: int = 26,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """
    Detect what regime each key is in at each time point.

    Regimes capture structural state changes that affect forecasting:
    - Growth regime: is recent demand above long-term average?
    - Volatility regime: is recent volatility above normal?
    - Momentum regime: is demand accelerating or decelerating?
    - Level shift indicator: has the level changed significantly recently?

    All features use shift(forecast_lag) for leakage prevention.

    Parameters
    ----------
    df : pd.DataFrame
        Source data sorted by key + date
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    forecast_lag : int
        Forecast lag for leakage prevention
    short_window : int
        Short rolling window (default 4 = ~1 month for weekly)
    long_window : int
        Long rolling window (default 26 = ~6 months for weekly)
    seasonal_period : int
        Seasonal period (52 for weekly, 12 for monthly)

    Returns
    -------
    pd.DataFrame
        DataFrame with regime features added
    """
    logger.info(f"Creating regime features (short={short_window}, long={long_window}, lag={forecast_lag})")

    key_col = key_cols[0] if len(key_cols) == 1 else key_cols
    features_added = []

    # Lagged target (leakage-free base for all computations)
    lagged = df.groupby(key_col)[target_col].shift(forecast_lag)

    # ─── GROWTH REGIME ──────────────────────────────────────────────────
    # Short MA vs Long MA: captures whether demand is above/below trend
    short_ma = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag).rolling(short_window, min_periods=1).mean()
    )
    long_ma = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag).rolling(long_window, min_periods=1).mean()
    )

    # Binary growth indicator
    df['regime_growth'] = (short_ma > long_ma).astype(np.float32)
    features_added.append('regime_growth')

    # Continuous growth ratio (how far above/below trend)
    df['regime_growth_ratio'] = (short_ma / (long_ma + 1e-10)).clip(0, 5).astype(np.float32)
    features_added.append('regime_growth_ratio')

    # ─── VOLATILITY REGIME ──────────────────────────────────────────────
    # Compares recent volatility to historical volatility
    short_vol = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag).rolling(short_window, min_periods=2).std()
    )
    long_vol = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag).rolling(long_window, min_periods=2).std()
    )

    df['regime_high_volatility'] = (short_vol > long_vol).astype(np.float32)
    features_added.append('regime_high_volatility')

    df['regime_volatility_ratio'] = (short_vol / (long_vol + 1e-10)).clip(0, 10).astype(np.float32)
    features_added.append('regime_volatility_ratio')

    # ─── MOMENTUM REGIME ────────────────────────────────────────────────
    # Rate of change of the moving average (acceleration/deceleration)
    ma_diff = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag).rolling(short_window, min_periods=1).mean().diff()
    )
    mean_abs = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag).rolling(long_window, min_periods=1).mean()
    )
    df['regime_momentum'] = (ma_diff / (mean_abs.abs() + 1e-10)).clip(-2, 2).astype(np.float32)
    features_added.append('regime_momentum')

    # Direction: +1 accelerating, -1 decelerating, 0 stable
    df['regime_direction'] = np.sign(df['regime_momentum']).astype(np.float32)
    features_added.append('regime_direction')

    # ─── LEVEL SHIFT INDICATOR ──────────────────────────────────────────
    # Detects sudden jumps: recent level vs prior level
    recent_level = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag).rolling(short_window, min_periods=1).mean()
    )
    prior_level = df.groupby(key_col)[target_col].transform(
        lambda x: x.shift(forecast_lag + short_window).rolling(short_window, min_periods=1).mean()
    )
    safe_prior = np.maximum(prior_level.abs(), 1e-6)  # Floor to avoid extreme ratios from near-zero baselines
    level_change = (recent_level - prior_level) / safe_prior
    df['regime_level_shift'] = level_change.clip(-3, 3).astype(np.float32)
    features_added.append('regime_level_shift')

    # ─── ZERO-DEMAND REGIME ─────────────────────────────────────────────
    # How many of the last N periods had zero demand? (intermittency state)
    for window in [short_window, long_window]:
        col = f'regime_zero_rate_{window}'
        df[col] = df.groupby(key_col)[target_col].transform(
            lambda x: (x.shift(forecast_lag) == 0).astype(float).rolling(window, min_periods=1).mean()
        ).astype(np.float32)
        features_added.append(col)

    # Fill NaN from initial periods
    for col in features_added:
        df[col] = df[col].fillna(0)

    logger.info(f"Created {len(features_added)} regime features: {features_added}")
    return df


# =============================================================================
# 2. CROSS-KEY RELATIVE FEATURES
# =============================================================================

def create_relative_features(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    forecast_lag: int,
    segment_col: Optional[str] = None,
    hierarchy_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Features measuring each key's performance relative to peers.

    Captures competitive dynamics: is this key gaining or losing share?
    Is it more or less volatile than its peers? Is it leading or lagging
    the group trend?

    Parameters
    ----------
    df : pd.DataFrame
        Source data
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    forecast_lag : int
        Forecast lag for leakage prevention
    segment_col : str, optional
        Segment column for segment-relative features
    hierarchy_cols : List[str], optional
        Hierarchy columns for hierarchy-relative features

    Returns
    -------
    pd.DataFrame
        DataFrame with relative features added
    """
    logger.info("Creating cross-key relative features...")

    key_col = key_cols[0] if len(key_cols) == 1 else key_cols
    features_added = []

    # Leakage-free lagged target
    df['_lagged_target'] = df.groupby(key_col)[target_col].shift(forecast_lag)

    # ─── OVERALL RELATIVE FEATURES ──────────────────────────────────────
    # Compare each key to the overall population at each time point
    period_stats = df.groupby(date_col)['_lagged_target'].agg(['mean', 'median', 'std'])
    period_stats.columns = ['_period_mean', '_period_median', '_period_std']
    df = df.merge(period_stats, on=date_col, how='left')

    # Relative to period mean
    df['rel_to_overall_mean'] = (
        df['_lagged_target'] / (df['_period_mean'] + 1e-10)
    ).clip(0, 20).astype(np.float32)
    features_added.append('rel_to_overall_mean')

    # Distance from period median (robust to outliers)
    df['dist_from_overall_median'] = (
        df['_lagged_target'] - df['_period_median']
    ).astype(np.float32)
    features_added.append('dist_from_overall_median')

    # Overall rank
    df['rank_overall'] = df.groupby(date_col)['_lagged_target'].rank(
        pct=True, method='average'
    ).astype(np.float32)
    features_added.append('rank_overall')

    # Cleanup period stats
    df.drop(columns=['_period_mean', '_period_median', '_period_std'], inplace=True, errors='ignore')

    # ─── SEGMENT-RELATIVE FEATURES ──────────────────────────────────────
    if segment_col and segment_col in df.columns:
        # Fill NaN segment values to prevent groupby silently dropping rows
        if df[segment_col].isna().any():
            logger.warning(f"Found {df[segment_col].isna().sum()} NaN in {segment_col}, filling with 'Unknown'")
            df[segment_col] = df[segment_col].fillna('Unknown')
        seg_stats = df.groupby([segment_col, date_col])['_lagged_target'].agg(['mean', 'median', 'std'])
        seg_stats.columns = ['_seg_mean', '_seg_median', '_seg_std']
        seg_stats = seg_stats.reset_index()
        df = df.merge(seg_stats, on=[segment_col, date_col], how='left')

        # Relative to segment mean
        df['rel_to_segment_mean'] = (
            df['_lagged_target'] / (df['_seg_mean'] + 1e-10)
        ).clip(0, 20).astype(np.float32)
        features_added.append('rel_to_segment_mean')

        # Distance from segment median
        df['dist_from_segment_median'] = (
            df['_lagged_target'] - df['_seg_median']
        ).astype(np.float32)
        features_added.append('dist_from_segment_median')

        # Rank within segment
        df['rank_in_segment'] = df.groupby([segment_col, date_col])['_lagged_target'].rank(
            pct=True, method='average'
        ).astype(np.float32)
        features_added.append('rank_in_segment')

        # Relative momentum: key's change vs segment's average change
        key_change = df.groupby(key_col)['_lagged_target'].diff()
        seg_change = df.groupby([segment_col, date_col])['_lagged_target'].transform(
            lambda x: x.diff().mean()
        )
        df['relative_momentum_seg'] = (
            (key_change - seg_change) / (df['_seg_mean'].abs() + 1e-10)
        ).clip(-5, 5).astype(np.float32)
        features_added.append('relative_momentum_seg')

        df.drop(columns=['_seg_mean', '_seg_median', '_seg_std'], inplace=True, errors='ignore')

    # ─── HIERARCHY-RELATIVE FEATURES ────────────────────────────────────
    if hierarchy_cols:
        for hier_col in hierarchy_cols[:2]:  # Cap at 2
            if hier_col not in df.columns:
                continue
            safe = hier_col.lower().replace(' ', '_')[:15]

            hier_stats = df.groupby([hier_col, date_col])['_lagged_target'].agg(['mean', 'sum'])
            hier_stats.columns = [f'_hier_{safe}_mean', f'_hier_{safe}_sum']
            hier_stats = hier_stats.reset_index()
            df = df.merge(hier_stats, on=[hier_col, date_col], how='left')

            # Share of hierarchy group
            col = f'rel_share_{safe}'
            df[col] = (
                df['_lagged_target'] / (df[f'_hier_{safe}_sum'] + 1e-10)
            ).clip(0, 1).astype(np.float32)
            features_added.append(col)

            # Relative to hierarchy mean
            col = f'rel_to_{safe}_mean'
            df[col] = (
                df['_lagged_target'] / (df[f'_hier_{safe}_mean'] + 1e-10)
            ).clip(0, 20).astype(np.float32)
            features_added.append(col)

            # Rank in hierarchy group
            col = f'rank_in_{safe}'
            df[col] = df.groupby([hier_col, date_col])['_lagged_target'].rank(
                pct=True, method='average'
            ).astype(np.float32)
            features_added.append(col)

            df.drop(columns=[f'_hier_{safe}_mean', f'_hier_{safe}_sum'],
                    inplace=True, errors='ignore')

    # Cleanup
    df.drop(columns=['_lagged_target'], inplace=True, errors='ignore')

    # Fill NaN
    for col in features_added:
        df[col] = df[col].fillna(0)

    logger.info(f"Created {len(features_added)} cross-key relative features: {features_added}")
    return df


# =============================================================================
# 3. HISTORY-ONLY FEATURE EMBEDDINGS
# =============================================================================

def create_history_embeddings(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    future_unknown_features: List[str],
    train_end: str,
    max_features: int = 20,
) -> pd.DataFrame:
    """
    Convert future-unknown features into static key-level embeddings.

    For features classified as `history_only` by Feature Availability Detection,
    we cannot use their raw values at inference time (they won't exist).
    Instead, we extract their historical signal as "frozen" aggregates:

    Per feature, per key:
    - {feat}_hist_mean: historical mean value
    - {feat}_hist_std: historical standard deviation
    - {feat}_hist_median: historical median
    - {feat}_target_corr: correlation with target (captures relationship strength)
    - {feat}_trend: linear slope during training (captures direction)

    These static features are merged onto ALL rows (train + val + test), so
    they're available during inference without needing future values.

    Parameters
    ----------
    df : pd.DataFrame
        Full data (train + val + test)
    key_cols : List[str]
        Key columns
    date_col : str
        Date column
    target_col : str
        Target column
    future_unknown_features : List[str]
        Features classified as history_only by Feature Availability Detection
    train_end : str
        Training end period (embeddings computed from data <= train_end only)
    max_features : int
        Maximum number of features to create embeddings for (default 20)

    Returns
    -------
    pd.DataFrame
        DataFrame with history embedding features added
    """
    # Filter to features that actually exist and are numeric
    valid_features = [
        f for f in future_unknown_features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ][:max_features]

    if not valid_features:
        logger.info("No valid future-unknown numeric features for history embeddings")
        return df

    logger.info(f"Creating history embeddings for {len(valid_features)} future-unknown features: "
                f"{valid_features[:5]}{'...' if len(valid_features) > 5 else ''}")

    key_col = key_cols[0] if len(key_cols) == 1 else key_cols

    # Filter to training period only
    try:
        cutoff = int(train_end)
    except (ValueError, TypeError):
        cutoff = train_end
    train_df = df[df[date_col] <= cutoff]

    if len(train_df) == 0:
        logger.warning("No training data for history embeddings")
        return df

    features_added = []

    for feat in valid_features:
        # Skip if feature is all null/zero in training
        feat_data = train_df[feat].dropna()
        if len(feat_data) == 0 or feat_data.std() < 1e-10:
            continue

        prefix = feat[:25]  # Truncate long names

        # ─── Static aggregates per key ──────────────────────────────
        try:
            key_stats = train_df.groupby(key_col)[feat].agg(['mean', 'std', 'median'])
            key_stats.columns = [
                f'{prefix}_hist_mean',
                f'{prefix}_hist_std',
                f'{prefix}_hist_median',
            ]
            key_stats = key_stats.reset_index()

            # Fill NaN std with 0 (keys with single observation)
            key_stats[f'{prefix}_hist_std'] = key_stats[f'{prefix}_hist_std'].fillna(0)

            df = df.merge(key_stats, on=key_col, how='left')
            features_added.extend([f'{prefix}_hist_mean', f'{prefix}_hist_std', f'{prefix}_hist_median'])
        except Exception as e:
            logger.debug(f"Could not compute stats for {feat}: {e}")
            continue

        # ─── Correlation with target (per key) ──────────────────────
        try:
            _apply_kwargs = {}
            if pd.__version__ >= '2.2.0':
                _apply_kwargs['include_groups'] = False
            key_corr = train_df.groupby(key_col).apply(
                lambda g: g[feat].corr(g[target_col]) if len(g) >= 5 else 0.0,
                **_apply_kwargs,
            )
            key_corr = key_corr.rename(f'{prefix}_target_corr').reset_index()
            key_corr[f'{prefix}_target_corr'] = (
                key_corr[f'{prefix}_target_corr'].fillna(0).clip(-1, 1)
            )
            df = df.merge(key_corr, on=key_col, how='left')
            features_added.append(f'{prefix}_target_corr')
        except Exception as e:
            logger.debug(f"Could not compute correlation for {feat}: {e}")

        # ─── Trend during training (per key) ────────────────────────
        try:
            def _compute_trend(g):
                vals = g[feat].fillna(0).values
                if len(vals) < 5:
                    return 0.0
                try:
                    x = np.arange(len(vals), dtype=float)
                    slope = np.polyfit(x, vals, 1)[0]
                    return float(slope)
                except (np.linalg.LinAlgError, ValueError):
                    return 0.0

            _apply_kwargs2 = {}
            if pd.__version__ >= '2.2.0':
                _apply_kwargs2['include_groups'] = False
            key_trend = train_df.groupby(key_col).apply(
                _compute_trend, **_apply_kwargs2,
            )
            key_trend = key_trend.rename(f'{prefix}_hist_trend').reset_index()
            df = df.merge(key_trend, on=key_col, how='left')
            features_added.append(f'{prefix}_hist_trend')
        except Exception as e:
            logger.debug(f"Could not compute trend for {feat}: {e}")

    # Fill NaN in new columns
    for col in features_added:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(np.float32)

    logger.info(f"Created {len(features_added)} history embedding features from "
                f"{len(valid_features)} future-unknown features")
    return df


# =============================================================================
# MASTER FUNCTION: Create all rich features
# =============================================================================

def create_all_rich_features(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    forecast_lag: int,
    train_end: str,
    segment_col: Optional[str] = None,
    hierarchy_cols: Optional[List[str]] = None,
    future_unknown_features: Optional[List[str]] = None,
    enable_regime: bool = True,
    enable_relative: bool = True,
    enable_history_embeddings: bool = True,
    seasonal_period: int = 52,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Master function: create all Phase 6 rich features in one call.

    Orchestrates regime, cross-key relative, and history embedding features.

    Parameters
    ----------
    df : pd.DataFrame
        Source data (train + val + test)
    key_cols, date_col, target_col : str
        Column names
    forecast_lag : int
        Forecast lag for leakage prevention
    train_end : str
        Training end period
    segment_col : str, optional
        Segment column for segment-relative features
    hierarchy_cols : List[str], optional
        Hierarchy columns for hierarchy-relative features
    future_unknown_features : List[str], optional
        Features classified as history_only by Feature Availability Detection
    enable_regime : bool
        Create regime/state features
    enable_relative : bool
        Create cross-key relative features
    enable_history_embeddings : bool
        Create history embeddings for future-unknown features
    seasonal_period : int
        52 for weekly, 12 for monthly

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (enriched DataFrame, metadata about features created)
    """
    logger.info("=" * 60)
    logger.info("PHASE 6: RICH FEATURE ENGINEERING")
    logger.info("=" * 60)

    n_cols_before = len(df.columns)
    metadata = {
        'regime_features': [],
        'relative_features': [],
        'history_embedding_features': [],
    }

    short_window = max(2, seasonal_period // 13)  # ~4 for weekly, ~1 for monthly
    long_window = seasonal_period // 2  # ~26 for weekly, ~6 for monthly

    # ─── 1. Regime Features ─────────────────────────────────────────────
    if enable_regime:
        logger.info("\n[1/3] Creating regime/state features...")
        before = set(df.columns)
        df = create_regime_features(
            df, key_cols, date_col, target_col, forecast_lag,
            short_window=short_window,
            long_window=long_window,
            seasonal_period=seasonal_period,
        )
        new_cols = [c for c in df.columns if c not in before]
        metadata['regime_features'] = new_cols

    # ─── 2. Cross-Key Relative Features ─────────────────────────────────
    if enable_relative:
        logger.info("\n[2/3] Creating cross-key relative features...")
        before = set(df.columns)
        df = create_relative_features(
            df, key_cols, date_col, target_col, forecast_lag,
            segment_col=segment_col,
            hierarchy_cols=hierarchy_cols,
        )
        new_cols = [c for c in df.columns if c not in before]
        metadata['relative_features'] = new_cols

    # ─── 3. History-Only Feature Embeddings ─────────────────────────────
    if enable_history_embeddings and future_unknown_features:
        logger.info("\n[3/3] Creating history embeddings for future-unknown features...")
        before = set(df.columns)
        df = create_history_embeddings(
            df, key_cols, date_col, target_col,
            future_unknown_features=future_unknown_features,
            train_end=train_end,
        )
        new_cols = [c for c in df.columns if c not in before]
        metadata['history_embedding_features'] = new_cols
    else:
        logger.info("\n[3/3] Skipping history embeddings (no future-unknown features)")

    n_cols_after = len(df.columns)
    total_new = n_cols_after - n_cols_before

    metadata['total_features_added'] = total_new
    metadata['n_regime'] = len(metadata['regime_features'])
    metadata['n_relative'] = len(metadata['relative_features'])
    metadata['n_embeddings'] = len(metadata['history_embedding_features'])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"RICH FEATURES COMPLETE: {total_new} new features")
    logger.info(f"  Regime:     {metadata['n_regime']}")
    logger.info(f"  Relative:   {metadata['n_relative']}")
    logger.info(f"  Embeddings: {metadata['n_embeddings']}")
    logger.info(f"{'=' * 60}")

    return df, metadata
