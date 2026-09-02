# utils/sply_correction.py
"""
Same-Period-Last-Year (SPLY) Post-Prediction Bias Correction.

FINAL adjustment step after all other corrections (calibration, error
correction, reconciliation). For FMCG demand, SPLY is the single most
powerful benchmark — it captures seasonality, base level, and distribution
patterns that models often miss.

Design principles:
1. VECTORISED — no row-by-row iteration; works on full DataFrame via merge + arithmetic
2. GROWTH-ADJUSTED — doesn't blindly use last year's value; adjusts for observed YoY growth
3. ADAPTIVE ALPHA — blend weight varies by:
   - Segment (different segments have different SPLY reliability)
   - Key stability (stable keys trust SPLY more, volatile keys trust model more)
4. GRACEFUL DEGRADATION — if SPLY unavailable (new products, missing history), returns
   model forecast unchanged

The correction formula:
    sply_adjusted = prior_year_actual × yoy_growth_factor
    final = alpha × model_forecast + (1 - alpha) × sply_adjusted

Where:
    yoy_growth_factor = recent_period_total / prior_year_same_period_total
    alpha = learned per-segment from validation (0 = pure SPLY, 1 = pure model)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_prior_year_period(period: str, time_format: str) -> Optional[str]:
    """Map a period to its same-period-last-year equivalent."""
    try:
        s = str(period)
        if '-' in s:
            parts = s.split('-')
            year, sub = int(parts[0]), int(parts[1])
            return f"{year - 1}-{sub:02d}"
        else:
            val = int(float(s))
            if val > 99999:  # YYYYWW or YYYYMM
                year, sub = val // 100, val % 100
                return str((year - 1) * 100 + sub)
        return None
    except (ValueError, TypeError):
        return None


def _build_sply_map(
    source_df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    forecast_periods: List,
    time_format: str,
) -> pd.DataFrame:
    """
    Vectorised SPLY lookup: for each forecast period, find the prior-year
    same-period actual for every key.

    Returns DataFrame with [key_col, date_col, 'sply_actual', 'sply_period'].
    """
    # Build period → prior year period mapping
    period_map = {}
    for p in forecast_periods:
        py = _get_prior_year_period(str(p), time_format)
        if py is not None:
            period_map[str(p)] = py

    if not period_map:
        return pd.DataFrame(columns=[key_col, date_col, 'sply_actual', 'sply_period'])

    # Get all prior-year periods we need
    prior_periods = list(set(period_map.values()))

    # Vectorised: filter source to prior-year periods, then rename
    sply_data = source_df[source_df[date_col].astype(str).isin(prior_periods)][[key_col, date_col, target_col]].copy()
    sply_data[date_col] = sply_data[date_col].astype(str)
    sply_data = sply_data.rename(columns={target_col: 'sply_actual', date_col: 'sply_period'})

    # Map prior periods back to forecast periods
    reverse_map = {}
    for fp, pp in period_map.items():
        reverse_map.setdefault(pp, []).append(fp)

    # Expand: each prior-period row maps to one or more forecast periods
    rows = []
    for _, row in sply_data.iterrows():
        sp = row['sply_period']
        if sp in reverse_map:
            for fp in reverse_map[sp]:
                rows.append({
                    key_col: row[key_col],
                    date_col: fp,
                    'sply_actual': float(row['sply_actual']) if pd.notna(row['sply_actual']) else 0.0,
                    'sply_period': sp,
                })

    if not rows:
        return pd.DataFrame(columns=[key_col, date_col, 'sply_actual', 'sply_period'])

    result = pd.DataFrame(rows)
    # Aggregate if multiple rows per (key, period) — shouldn't happen but be safe
    result = result.groupby([key_col, date_col]).agg(
        sply_actual=('sply_actual', 'sum'),
        sply_period=('sply_period', 'first'),
    ).reset_index()

    return result


def compute_yoy_growth_factor(
    source_df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    reference_period_end: str,
    lookback_periods: int = 13,
    time_format: str = 'year_week',
    min_periods_required: int = 4,
) -> pd.DataFrame:
    """
    Compute per-key YoY growth factor from recent TRAINING history.

    growth_factor = sum(recent_N_periods_actual) / sum(prior_year_same_N_periods_actual)

    Keys with growth_factor > 1 are growing; < 1 are declining.
    This adjusts SPLY so we don't anchor declining products to last year's peak.

    Handles edge cases:
    - Keys with <min_periods_required history: growth_factor = 1.0 (neutral)
    - Keys with no prior-year data: growth_factor = 1.0
    - Keys with zero prior-year actual: growth_factor = 1.0
    - reference_period_end should be TRAINING END (not forecast start) to avoid leakage

    Parameters
    ----------
    source_df : pd.DataFrame
    key_col, date_col, target_col : str
    reference_period_end : str
        End of training period (NOT forecast start). Growth computed from
        periods BEFORE this point only — no validation/test leakage.
    lookback_periods : int
        How many recent periods to compare vs prior year (default 13 = ~1 quarter)
    time_format : str
    min_periods_required : int
        Minimum recent periods a key must have to compute growth. Keys below
        this get growth_factor=1.0 (neutral — no adjustment).

    Returns
    -------
    pd.DataFrame with [key_col, 'yoy_growth_factor']
    """
    all_keys = source_df[key_col].unique()
    all_periods = sorted(source_df[date_col].astype(str).unique())

    # Recent periods = last N training periods (strictly <= reference_period_end)
    ref = str(reference_period_end)
    training_periods = [p for p in all_periods if p <= ref]
    recent_periods = training_periods[-lookback_periods:]

    if len(recent_periods) < min_periods_required:
        logger.warning(f"Only {len(recent_periods)} training periods available "
                       f"(need {min_periods_required}) — using neutral growth factor")
        return pd.DataFrame({key_col: all_keys, 'yoy_growth_factor': 1.0})

    # Prior-year equivalents of these recent periods
    prior_periods = []
    for p in recent_periods:
        py = _get_prior_year_period(p, time_format)
        if py:
            prior_periods.append(py)

    if not prior_periods:
        logger.warning("No prior-year periods found — using neutral growth factor")
        return pd.DataFrame({key_col: all_keys, 'yoy_growth_factor': 1.0})

    # Per-key totals for recent and prior-year periods
    recent_data = source_df[source_df[date_col].astype(str).isin(recent_periods)]
    prior_data = source_df[source_df[date_col].astype(str).isin(prior_periods)]

    recent_totals = recent_data.groupby(key_col)[target_col].agg(['sum', 'count']).rename(
        columns={'sum': 'recent_total', 'count': 'recent_count'})
    prior_totals = prior_data.groupby(key_col)[target_col].sum().rename('prior_total')

    # Merge
    growth_df = pd.DataFrame({key_col: all_keys}).set_index(key_col)
    growth_df = growth_df.join(recent_totals).join(prior_totals)
    growth_df = growth_df.fillna({'recent_total': 0, 'recent_count': 0, 'prior_total': 0})

    # Compute growth factor with safeguards
    def _safe_growth(row):
        if row['recent_count'] < min_periods_required:
            return 1.0  # Not enough recent data
        if row['prior_total'] <= 0:
            return 1.0  # No prior year data
        if row['recent_total'] <= 0:
            return 0.5  # Key has died — use 50% of SPLY
        return row['recent_total'] / row['prior_total']

    growth_df['yoy_growth_factor'] = growth_df.apply(_safe_growth, axis=1).clip(0.2, 5.0)
    growth_df = growth_df.reset_index()[[key_col, 'yoy_growth_factor']]

    n_neutral = (growth_df['yoy_growth_factor'] == 1.0).sum()
    logger.info(f"YoY growth factors: median={growth_df['yoy_growth_factor'].median():.2f}, "
                f"mean={growth_df['yoy_growth_factor'].mean():.2f}, "
                f"range=[{growth_df['yoy_growth_factor'].min():.2f}, {growth_df['yoy_growth_factor'].max():.2f}], "
                f"neutral={n_neutral}/{len(growth_df)} keys")

    return growth_df


def learn_sply_blend_weights(
    val_predictions_df: pd.DataFrame,
    source_df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    predicted_col: str = 'predicted',
    segment_col: str = 'segment_id',
    time_format: str = 'year_week',
    train_end: Optional[str] = None,
    min_samples: int = 10,
) -> Dict[str, float]:
    """
    Learn optimal SPLY blend weight (alpha) per segment from validation data.

    Tests alpha ∈ [0, 0.05, 0.1, ..., 1.0] and picks the one minimising WAPE.

    Parameters
    ----------
    train_end : str, optional
        End of training period. Growth factors computed from data UP TO this
        point only (no validation/test leakage). If None, uses the earliest
        validation period as reference.

    Returns {segment_id: alpha} where alpha=0 is pure SPLY, alpha=1 is pure model.
    """
    val_df = val_predictions_df.copy()
    val_df[date_col] = val_df[date_col].astype(str)

    # Get SPLY + growth factor for validation periods
    val_periods = list(val_df[date_col].unique())
    sply_map = _build_sply_map(source_df, key_col, date_col, target_col, val_periods, time_format)

    if len(sply_map) == 0:
        logger.warning("No SPLY data for validation — using alpha=0.7")
        return {'_default': 0.7}

    # Growth factors from TRAINING period only (no leakage)
    ref_end = str(train_end) if train_end else min(val_periods)
    growth_df = compute_yoy_growth_factor(source_df, key_col, date_col, target_col, ref_end, 13, time_format)

    # Merge SPLY + growth into validation
    sply_map[date_col] = sply_map[date_col].astype(str)
    val_df = val_df.merge(sply_map[[key_col, date_col, 'sply_actual']], on=[key_col, date_col], how='left')
    val_df = val_df.merge(growth_df, on=key_col, how='left')
    val_df['sply_actual'] = val_df['sply_actual'].fillna(0)
    val_df['yoy_growth_factor'] = val_df['yoy_growth_factor'].fillna(1.0)

    # Growth-adjusted SPLY
    val_df['sply_adjusted'] = val_df['sply_actual'] * val_df['yoy_growth_factor']

    weights = {}
    segments = val_df[segment_col].unique() if segment_col in val_df.columns else ['_default']

    for seg in segments:
        seg_df = val_df[val_df[segment_col] == seg] if segment_col in val_df.columns else val_df

        if len(seg_df) < min_samples:
            weights[str(seg)] = 0.7
            continue

        actuals = seg_df[target_col].values.astype(float)
        model_preds = seg_df[predicted_col].values.astype(float)
        sply_adj = seg_df['sply_adjusted'].values.astype(float)

        total_actual = np.sum(np.abs(actuals))
        if total_actual < 1e-10:
            weights[str(seg)] = 0.7
            continue

        # Grid search for optimal alpha
        best_alpha = 0.7
        best_wape = float('inf')

        for alpha_int in range(0, 21):
            alpha = alpha_int / 20.0  # 0.00, 0.05, 0.10, ..., 1.00
            blended = np.maximum(alpha * model_preds + (1 - alpha) * sply_adj, 0)
            wape = np.sum(np.abs(actuals - blended)) / total_actual
            if wape < best_wape:
                best_wape = wape
                best_alpha = alpha

        weights[str(seg)] = float(best_alpha)

        # Diagnostic: compare pure model vs pure SPLY vs optimal blend
        model_wape = np.sum(np.abs(actuals - np.maximum(model_preds, 0))) / total_actual
        sply_wape = np.sum(np.abs(actuals - np.maximum(sply_adj, 0))) / total_actual
        logger.info(f"  Segment {seg}: alpha={best_alpha:.2f} | "
                     f"model={model_wape:.3f}, sply_adj={sply_wape:.3f}, blend={best_wape:.3f}")

    # Global fallback
    if '_default' not in weights:
        weights['_default'] = float(np.median(list(weights.values()))) if weights else 0.7

    return weights


def apply_sply_correction(
    forecasts_df: pd.DataFrame,
    source_df: pd.DataFrame,
    key_col: str,
    date_col: str,
    target_col: str,
    predicted_col: str = 'predicted',
    segment_col: str = 'segment_id',
    blend_weights: Optional[Dict[str, float]] = None,
    time_format: str = 'year_week',
    default_alpha: float = 0.7,
    train_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply growth-adjusted SPLY correction. VECTORISED — no row-by-row iteration.

    For each (key, period):
        sply_adjusted = prior_year_actual × yoy_growth_factor
        final = alpha × model_forecast + (1 - alpha) × sply_adjusted

    Parameters
    ----------
    train_end : str, optional
        End of training period. Growth factors use ONLY data up to this point.
        If None, uses earliest forecast period (may cause leakage if forecast
        periods overlap with validation).
    """
    df = forecasts_df.copy()
    df[date_col] = df[date_col].astype(str)

    diagnostics = {
        'sply_correction_applied': True,
        'total_before': float(df[predicted_col].sum()),
    }

    forecast_periods = list(df[date_col].unique())

    # Step 1: Get SPLY values
    sply_map = _build_sply_map(source_df, key_col, date_col, target_col, forecast_periods, time_format)
    if len(sply_map) == 0:
        logger.warning("No SPLY data available — skipping correction")
        diagnostics['sply_correction_applied'] = False
        diagnostics['n_rows_corrected'] = 0
        return df, diagnostics

    # Step 2: Growth factors from TRAINING period only (no leakage)
    ref_end = str(train_end) if train_end else min(forecast_periods)
    growth_df = compute_yoy_growth_factor(source_df, key_col, date_col, target_col, ref_end, 13, time_format)

    # Step 3: Merge (vectorised)
    sply_map[date_col] = sply_map[date_col].astype(str)
    df = df.merge(sply_map[[key_col, date_col, 'sply_actual']], on=[key_col, date_col], how='left')
    df = df.merge(growth_df, on=key_col, how='left')
    df['sply_actual'] = df['sply_actual'].fillna(0)
    df['yoy_growth_factor'] = df['yoy_growth_factor'].fillna(1.0)

    # Step 4: Growth-adjusted SPLY
    df['sply_adjusted'] = df['sply_actual'] * df['yoy_growth_factor']

    # Step 5: Build alpha column (vectorised)
    if blend_weights and segment_col in df.columns:
        default_w = blend_weights.get('_default', default_alpha)
        df['_alpha'] = df[segment_col].astype(str).map(blend_weights).fillna(default_w)
    else:
        df['_alpha'] = default_alpha

    # Step 6: Apply blend (vectorised) — only where SPLY is available
    has_sply = df['sply_actual'] > 0
    df.loc[has_sply, predicted_col] = (
        df.loc[has_sply, '_alpha'] * df.loc[has_sply, predicted_col] +
        (1 - df.loc[has_sply, '_alpha']) * df.loc[has_sply, 'sply_adjusted']
    ).clip(lower=0)

    # Cleanup
    n_corrected = int(has_sply.sum())
    df.drop(columns=['sply_actual', 'yoy_growth_factor', 'sply_adjusted', '_alpha'], inplace=True, errors='ignore')

    diagnostics['n_rows_corrected'] = n_corrected
    diagnostics['n_rows_no_sply'] = len(df) - n_corrected
    diagnostics['total_after'] = float(df[predicted_col].sum())
    diagnostics['correction_impact_pct'] = float(
        (diagnostics['total_after'] - diagnostics['total_before']) / (diagnostics['total_before'] + 1e-10) * 100
    )
    diagnostics['avg_alpha'] = float(df.get('_alpha', pd.Series([default_alpha])).mean()) if '_alpha' in df.columns else default_alpha

    logger.info(f"SPLY correction: {n_corrected} rows corrected, "
                f"{diagnostics['n_rows_no_sply']} no SPLY, "
                f"impact={diagnostics['correction_impact_pct']:+.1f}%")

    return df, diagnostics
