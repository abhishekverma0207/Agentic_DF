# utils/reconciliation.py
"""
Top-Down Category-Level Forecast Reconciliation.

This module trains a Prophet model on the aggregate (category-level) demand
time series and uses its forecasts to proportionally scale key-level
forecasts so that they sum to the category total each period.

Why?
----
Key-level models (especially segment-pooled ones) can systematically
underforecast or overforecast because segments mix heterogeneous keys.
A category-level model sees the sum of all keys — an extremely stable
signal thanks to the law of large numbers — and captures trend and
seasonality much more reliably than individual key models.

The reconciliation ensures the total forecast is anchored to a robust
aggregate forecast while preserving each key's relative share.

Integration Points
-------------------
1. Inference:   after generate_forward_forecasts(), before saving
2. Backtesting: after _generate_origin_forecasts() per origin
3. Validation:  after generate_validation_predictions(), before
                learning bias calibration factors
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 1. TRAIN CATEGORY-LEVEL PROPHET MODEL
# =============================================================================

def train_category_model(
    source_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    key_col: str,
    train_cutoff: Optional[str] = None,
    time_format: str = 'year_week',
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
) -> object:
    """
    Train a Prophet model on the category-level aggregate time series.

    The model learns from total demand (sum across all keys per period),
    which already implicitly captures portfolio size changes. No key-count
    regressor is used — this prevents dangerous sensitivity to the gap
    between historically-active keys and forecast-active keys (tree models
    rarely predict exactly zero, inflating the apparent active-key count).

    Parameters
    ----------
    source_df : pd.DataFrame
        Full source data with all keys.
    target_col : str
        Target column name (e.g. 'qty').
    date_col : str
        Date/time period column (e.g. 'year_month', 'year_week').
    key_col : str
        Key column name for grouping.
    train_cutoff : str, optional
        If provided, only use data up to this period for training.
        If None, use all available data.
    time_format : str
        'year_week' or 'year_month'.
    yearly_seasonality : bool
        Whether to include yearly seasonality (default True).
    changepoint_prior_scale : float
        Flexibility of trend changepoints (higher = more flexible).

    Returns
    -------
    prophet.Prophet
        Fitted Prophet model on the aggregate series.
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning(
            "Prophet not installed. Install with: pip install prophet. "
            "Top-down reconciliation will be skipped."
        )
        return None

    df = source_df.copy()

    # Filter to training data if cutoff provided
    if train_cutoff is not None:
        # Coerce train_cutoff to match the date column's dtype.
        # Config values arrive as strings (e.g. "202606") but the source
        # data column may be int64. Without this cast, pandas raises
        # TypeError: Invalid comparison between dtype=int64 and str.
        cutoff_val = _coerce_cutoff(train_cutoff, df[date_col].dtype)
        df = df[df[date_col] <= cutoff_val]

    if len(df) == 0:
        logger.warning("No training data available for category model")
        return None

    # Aggregate to category level: sum target by period
    cat_df = df.groupby(date_col, as_index=False)[target_col].sum()
    cat_df = cat_df.sort_values(date_col).reset_index(drop=True)

    # Convert to Prophet format (ds, y)
    cat_df['ds'] = _period_to_date(cat_df[date_col], time_format)
    cat_df['y'] = cat_df[target_col].astype(float)

    if len(cat_df) < 4:
        logger.warning(f"Only {len(cat_df)} periods for category model — too few, skipping")
        return None

    logger.info(
        f"Training category Prophet model on {len(cat_df)} periods, "
        f"total demand range: [{cat_df['y'].min():.0f}, {cat_df['y'].max():.0f}]"
    )

    # Configure Prophet — pure time-series model on aggregate demand.
    # The total demand signal already captures portfolio changes (more keys
    # selling → higher total), so no key-count regressor is needed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode='multiplicative',
        )

        # Fit on aggregate demand only — trend + yearly seasonality
        train_data = cat_df[['ds', 'y']].dropna()

        if len(train_data) < 2:
            logger.warning(
                f"Only {len(train_data)} non-NaN rows after date conversion "
                f"(from {len(cat_df)} periods) — too few for Prophet, skipping"
            )
            return None

        model.fit(train_data)

    logger.info("Category Prophet model trained successfully")
    return model


# =============================================================================
# 2. GENERATE CATEGORY-LEVEL FORECASTS
# =============================================================================

def forecast_category_total(
    model: object,
    forecast_periods: list,
    time_format: str = 'year_week',
) -> pd.DataFrame:
    """
    Generate category-level forecasts for given periods.

    Parameters
    ----------
    model : prophet.Prophet
        Fitted Prophet model.
    forecast_periods : list
        List of period values to forecast (e.g. [202401, 202402, ...]).
    time_format : str
        'year_week' or 'year_month'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [period, category_forecast].
    """
    if model is None:
        return pd.DataFrame(columns=['period', 'category_forecast'])

    # Build future dataframe
    future_dates = _period_to_date(pd.Series(forecast_periods), time_format)

    # Validate dates — warn if any NaT (malformed periods)
    n_nat = future_dates.isna().sum()
    if n_nat > 0:
        logger.warning(
            f"forecast_category_total: {n_nat}/{len(future_dates)} periods "
            f"could not be converted to dates — they will be dropped"
        )
        valid_mask = ~future_dates.isna()
        future_dates = future_dates[valid_mask].reset_index(drop=True)
        forecast_periods = [p for p, v in zip(forecast_periods, valid_mask) if v]

    if len(future_dates) == 0:
        logger.warning("No valid forecast periods — skipping category forecast")
        return pd.DataFrame(columns=['period', 'category_forecast'])

    future_df = pd.DataFrame({'ds': future_dates})

    # Predict — pure trend + seasonality, no regressors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future_df)

    # Build result
    result = pd.DataFrame({
        'period': forecast_periods,
        'category_forecast': forecast['yhat'].values.clip(min=0),
    })

    logger.info(
        f"Category forecasts: {len(result)} periods, "
        f"range [{result['category_forecast'].min():.0f}, {result['category_forecast'].max():.0f}], "
        f"mean={result['category_forecast'].mean():.0f}"
    )

    return result


# =============================================================================
# 3. RECONCILE KEY-LEVEL FORECASTS
# =============================================================================

def reconcile_forecasts(
    forecasts_df: pd.DataFrame,
    category_forecasts: pd.DataFrame,
    date_col: str,
    key_col: str = 'key',
    predicted_col: str = 'predicted',
    ratio_min: float = 0.5,
    ratio_max: float = 2.0,
    trust_band: float = 0.1,
) -> pd.DataFrame:
    """
    Proportionally scale key-level forecasts to match category totals.

    For each forecast period:
    1. Compute the bottom-up total (sum of key-level forecasts)
    2. Compute the ratio: category_forecast / bottom_up_total
    3. If ratio is within the trust band (1-trust_band, 1+trust_band),
       don't adjust — key-level models are already coherent
    4. Otherwise, cap ratio to [ratio_min, ratio_max] and scale all
       active key forecasts proportionally

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Key-level forecasts with columns: [key_col, date_col, predicted_col].
    category_forecasts : pd.DataFrame
        Category totals with columns: [period, category_forecast].
    date_col : str
        Date column name.
    key_col : str
        Key column name.
    predicted_col : str
        Predicted value column name.
    ratio_min : float
        Minimum allowed scaling ratio (default 0.5 = max 50% reduction).
    ratio_max : float
        Maximum allowed scaling ratio (default 2.0 = max 2x increase).
    trust_band : float
        If ratio is within (1-trust_band, 1+trust_band), skip adjustment.

    Returns
    -------
    pd.DataFrame
        Reconciled forecasts with adjusted predicted column.
    """
    if category_forecasts is None or len(category_forecasts) == 0:
        logger.info("No category forecasts available — skipping reconciliation")
        return forecasts_df

    df = forecasts_df.copy()

    # Build lookup: period -> category_forecast
    cat_lookup = dict(zip(
        category_forecasts['period'].astype(str),
        category_forecasts['category_forecast']
    ))

    periods_adjusted = 0
    periods_in_trust = 0
    periods_capped = 0
    total_periods = 0
    ratios_applied = []

    for period in df[date_col].unique():
        period_str = str(period)
        if period_str not in cat_lookup:
            continue

        total_periods += 1
        cat_total = cat_lookup[period_str]

        # Get key-level forecasts for this period
        period_mask = df[date_col] == period
        period_preds = df.loc[period_mask, predicted_col]

        # Only consider active keys (positive forecast)
        active_mask = period_mask & (df[predicted_col] > 0)
        active_total = df.loc[active_mask, predicted_col].sum()

        if active_total <= 0:
            continue

        # Compute ratio
        ratio = cat_total / active_total

        # Trust band check: if ratio is close to 1.0, skip
        if (1.0 - trust_band) <= ratio <= (1.0 + trust_band):
            periods_in_trust += 1
            ratios_applied.append(1.0)
            continue

        # Cap ratio to prevent extreme corrections
        original_ratio = ratio
        ratio = max(ratio_min, min(ratio_max, ratio))
        if ratio != original_ratio:
            periods_capped += 1

        # Apply proportional scaling to active keys only
        df.loc[active_mask, predicted_col] = (
            df.loc[active_mask, predicted_col] * ratio
        ).clip(lower=0)

        periods_adjusted += 1
        ratios_applied.append(ratio)

    # Summary logging
    if total_periods > 0:
        avg_ratio = np.mean(ratios_applied) if ratios_applied else 1.0
        logger.info(
            f"Reconciliation: {total_periods} periods processed, "
            f"{periods_adjusted} adjusted, {periods_in_trust} within trust band, "
            f"{periods_capped} capped. Average ratio: {avg_ratio:.3f}"
        )
    else:
        logger.info("Reconciliation: no matching periods found")

    return df


# =============================================================================
# 3b. YoY SANITY CHECK — Post-reconciliation guardrail
# =============================================================================

def yoy_sanity_check(
    forecasts_df: pd.DataFrame,
    source_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    key_col: str,
    predicted_col: str = 'predicted',
    time_format: str = 'year_week',
    yoy_max_deviation: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Post-reconciliation YoY sanity check.

    Compares the reconciled category total forecast for each period to the
    prior-year same-period actual total. If the YoY change exceeds
    +/- yoy_max_deviation, dampens the forecast for that period back
    toward the prior-year level.

    This prevents the category model from producing forecasts that
    swing wildly relative to observed historical totals.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Reconciled key-level forecasts.
    source_df : pd.DataFrame
        Full source data (for computing prior-year actuals).
    date_col : str
        Date column name.
    target_col : str
        Target column name.
    key_col : str
        Key column name.
    predicted_col : str
        Predicted value column name.
    time_format : str
        'year_week' or 'year_month'.
    yoy_max_deviation : float
        Maximum allowed YoY deviation. 0.15 = +/-15%.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (adjusted_forecasts, yoy_diagnostics)
    """
    df = forecasts_df.copy()
    diagnostics = {
        'yoy_check_applied': True,
        'periods_checked': 0,
        'periods_dampened': 0,
        'periods_ok': 0,
        'period_details': [],
    }

    # Compute prior-year actual totals by period
    # Map each forecast period to its prior-year equivalent
    prior_year_totals = _compute_prior_year_totals(
        source_df=source_df,
        target_col=target_col,
        date_col=date_col,
        time_format=time_format,
    )

    if not prior_year_totals:
        logger.info("YoY check: No prior-year data available — skipping")
        diagnostics['yoy_check_applied'] = False
        return df, diagnostics

    for period in df[date_col].unique():
        period_str = str(period)
        prior_period = _get_prior_year_period(period_str, time_format)

        if prior_period not in prior_year_totals:
            continue

        diagnostics['periods_checked'] += 1
        prior_total = prior_year_totals[prior_period]

        if prior_total <= 0:
            continue

        # Compute current forecast total for this period
        active_mask = (df[date_col] == period) & (df[predicted_col] > 0)
        forecast_total = df.loc[active_mask, predicted_col].sum()

        if forecast_total <= 0:
            continue

        # YoY ratio
        yoy_ratio = forecast_total / prior_total
        yoy_change = yoy_ratio - 1.0  # e.g., +0.25 = +25% growth

        period_detail = {
            'period': period_str,
            'prior_period': prior_period,
            'prior_year_total': float(prior_total),
            'forecast_total': float(forecast_total),
            'yoy_change_pct': float(yoy_change * 100),
            'dampened': False,
        }

        if abs(yoy_change) > yoy_max_deviation:
            # Dampen: blend forecast total toward the allowed limit
            # If forecast is +30% YoY but max allowed is +15%,
            # set the allowed total to prior_year * (1 + 0.15) = prior_year * 1.15
            if yoy_change > 0:
                allowed_total = prior_total * (1.0 + yoy_max_deviation)
            else:
                allowed_total = prior_total * (1.0 - yoy_max_deviation)

            dampen_ratio = allowed_total / forecast_total

            df.loc[active_mask, predicted_col] = (
                df.loc[active_mask, predicted_col] * dampen_ratio
            ).clip(lower=0)

            new_total = df.loc[active_mask, predicted_col].sum()
            diagnostics['periods_dampened'] += 1
            period_detail['dampened'] = True
            period_detail['dampen_ratio'] = float(dampen_ratio)
            period_detail['adjusted_total'] = float(new_total)
            period_detail['adjusted_yoy_pct'] = float((new_total / prior_total - 1.0) * 100)

            logger.warning(
                f"YoY check: Period {period_str} forecast {forecast_total:.0f} is "
                f"{yoy_change*100:+.1f}% vs prior year {prior_total:.0f} "
                f"(exceeds +/-{yoy_max_deviation*100:.0f}%). "
                f"Dampened to {new_total:.0f} ({(new_total/prior_total-1)*100:+.1f}%)"
            )
        else:
            diagnostics['periods_ok'] += 1

        diagnostics['period_details'].append(period_detail)

    logger.info(
        f"YoY sanity check: {diagnostics['periods_checked']} periods checked, "
        f"{diagnostics['periods_ok']} OK, {diagnostics['periods_dampened']} dampened "
        f"(max allowed: +/-{yoy_max_deviation*100:.0f}%)"
    )

    return df, diagnostics


# =============================================================================
# 3c. YoY TREND ADJUSTMENT — Pre-reconciliation key-level scaling
# =============================================================================

def apply_yoy_trend_adjustment(
    forecasts_df: pd.DataFrame,
    source_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    key_col: str,
    test_start: str,
    forecast_horizon: int,
    time_format: str = 'year_week',
    min_history_periods: int = 104,
    scaling_min: float = 0.5,
    scaling_max: float = 2.0,
    predicted_col: str = 'predicted',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adjust key-level forecasts based on observed YOY growth trends.

    For each prediction key with sufficient history (>= min_history_periods
    before test_start), computes the year-over-year growth rate in the
    pre-period (last forecast_horizon periods before test_start) and
    scales the forward forecast total to exhibit the same growth rate
    relative to prior-year actuals for the forecast period.

    This preserves each key's weekly/monthly profile (relative period
    distribution) via uniform scaling — only the total level changes.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Key-level forecasts with columns [key_col, date_col, predicted_col].
    source_df : pd.DataFrame
        Full source data with historical actuals per key per period.
    date_col : str
        Date column name (e.g., 'year_week').
    target_col : str
        Target column name (e.g., 'Actuals').
    key_col : str
        Key column name (e.g., 'key').
    test_start : str
        First period of the forecast horizon (e.g., '202541').
    forecast_horizon : int
        Number of forecast periods (e.g., 13).
    time_format : str
        'year_week' or 'year_month'.
    min_history_periods : int
        Minimum historical periods required for eligibility. Default 104.
    scaling_min : float
        Floor for the scaling factor. Default 0.5.
    scaling_max : float
        Ceiling for the scaling factor. Default 2.0.
    predicted_col : str
        Name of the prediction column. Default 'predicted'.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (adjusted_forecasts, diagnostics_dict)
    """
    df = forecasts_df.copy()
    diagnostics = {
        'yoy_trend_adjustment_applied': True,
        'total_keys': 0,
        'keys_eligible': 0,
        'keys_adjusted': 0,
        'keys_skipped_insufficient_history': 0,
        'keys_skipped_zero_prior_year': 0,
        'keys_skipped_zero_forecast': 0,
        'keys_capped': 0,
        'key_details': [],
    }

    # Coerce test_start to match source_df dtype
    col_dtype = source_df[date_col].dtype
    test_start_coerced = _coerce_cutoff(test_start, col_dtype)

    # Get all unique keys from forecasts
    forecast_keys = df[key_col].unique()
    diagnostics['total_keys'] = len(forecast_keys)

    # Pre-compute: historical data (only rows before test_start)
    hist_df = source_df[source_df[date_col] < test_start_coerced]

    # Get sorted forecast periods from forecasts_df
    forecast_periods = sorted(df[date_col].unique())

    logger.info(
        f"YOY trend adjustment: {len(forecast_keys)} keys, "
        f"min_history={min_history_periods}, "
        f"scaling_bounds=[{scaling_min}, {scaling_max}]"
    )

    for key in forecast_keys:
        key_hist = hist_df[hist_df[key_col] == key]
        n_periods = len(key_hist[date_col].unique())

        # SKIP: insufficient history
        if n_periods < min_history_periods:
            diagnostics['keys_skipped_insufficient_history'] += 1
            continue

        diagnostics['keys_eligible'] += 1

        # --- Step A: Compute Pre-Period YOY Growth Rate ---
        # Pre-period = last forecast_horizon periods before test_start
        all_hist_periods = sorted(key_hist[date_col].unique())
        pre_period_dates = all_hist_periods[-forecast_horizon:]

        # Prior-year pre-period = shift each date back 1 year
        prior_year_pre_dates = []
        for p in pre_period_dates:
            py = _get_prior_year_period(str(p), time_format)
            if py:
                prior_year_pre_dates.append(_coerce_cutoff(py, col_dtype))

        pre_period_actual = float(
            key_hist[key_hist[date_col].isin(pre_period_dates)][target_col].sum()
        )

        prior_year_pre_actual = float(
            key_hist[key_hist[date_col].isin(prior_year_pre_dates)][target_col].sum()
        )

        # SKIP: prior-year pre-period has zero actuals
        if prior_year_pre_actual <= 0:
            diagnostics['keys_skipped_zero_prior_year'] += 1
            continue

        pre_period_yoy = pre_period_actual / prior_year_pre_actual

        # --- Step B: Compute Target Forecast Total ---
        # Prior-year forecast periods = shift each forecast period back 1 year
        prior_year_forecast_dates = []
        for p in forecast_periods:
            py = _get_prior_year_period(str(p), time_format)
            if py:
                prior_year_forecast_dates.append(_coerce_cutoff(py, col_dtype))

        # Look up prior-year actuals for the forecast period
        prior_year_forecast_actual = float(
            source_df[
                (source_df[key_col] == key) &
                (source_df[date_col].isin(prior_year_forecast_dates))
            ][target_col].sum()
        )

        # SKIP: prior-year forecast period has zero actuals
        if prior_year_forecast_actual <= 0:
            diagnostics['keys_skipped_zero_prior_year'] += 1
            continue

        target_forecast_total = prior_year_forecast_actual * pre_period_yoy

        # --- Step C: Scale Forecast ---
        key_mask = (df[key_col] == key) & (df[predicted_col] > 0)
        current_forecast_total = float(df.loc[key_mask, predicted_col].sum())

        # SKIP: current forecast total is zero
        if current_forecast_total <= 0:
            diagnostics['keys_skipped_zero_forecast'] += 1
            continue

        raw_scaling_factor = target_forecast_total / current_forecast_total

        # Apply guardrails
        capped = False
        scaling_factor = raw_scaling_factor
        if scaling_factor < scaling_min:
            scaling_factor = scaling_min
            capped = True
        elif scaling_factor > scaling_max:
            scaling_factor = scaling_max
            capped = True

        if capped:
            diagnostics['keys_capped'] += 1

        # Apply uniform scaling (preserves weekly profile)
        df.loc[key_mask, predicted_col] = (
            df.loc[key_mask, predicted_col] * scaling_factor
        ).clip(lower=0)

        diagnostics['keys_adjusted'] += 1

        # Per-key diagnostic detail
        key_detail = {
            'key': str(key),
            'n_history_periods': n_periods,
            'pre_period_actual': pre_period_actual,
            'prior_year_pre_actual': prior_year_pre_actual,
            'pre_period_yoy': float(pre_period_yoy),
            'prior_year_forecast_actual': prior_year_forecast_actual,
            'target_forecast_total': float(target_forecast_total),
            'original_forecast_total': current_forecast_total,
            'raw_scaling_factor': float(raw_scaling_factor),
            'applied_scaling_factor': float(scaling_factor),
            'capped': capped,
        }
        diagnostics['key_details'].append(key_detail)

        if abs(raw_scaling_factor - 1.0) > 0.1:
            logger.info(
                f"YOY trend: key={key}, pre_yoy={pre_period_yoy:.3f}, "
                f"scale={scaling_factor:.3f} "
                f"({'CAPPED' if capped else 'applied'}), "
                f"forecast {current_forecast_total:.0f} -> "
                f"{df.loc[key_mask, predicted_col].sum():.0f}"
            )

    logger.info(
        f"YOY trend adjustment: {diagnostics['keys_adjusted']}/{diagnostics['total_keys']} keys adjusted, "
        f"{diagnostics['keys_skipped_insufficient_history']} skipped (history), "
        f"{diagnostics['keys_skipped_zero_prior_year']} skipped (zero PY), "
        f"{diagnostics['keys_skipped_zero_forecast']} skipped (zero forecast), "
        f"{diagnostics['keys_capped']} capped to [{scaling_min}, {scaling_max}]"
    )

    return df, diagnostics


def _compute_prior_year_totals(
    source_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    time_format: str,
) -> Dict[str, float]:
    """Compute category total demand by period from source data."""
    totals = (
        source_df
        .groupby(date_col)[target_col]
        .sum()
    )
    return {str(k): float(v) for k, v in totals.items()}


def _get_prior_year_period(period_str: str, time_format: str) -> str:
    """
    Get the same period from the prior year.

    YYYYMM: 202403 -> 202303
    YYYYWW: 202415 -> 202315
    """
    try:
        period_int = int(period_str)
        year = period_int // 100
        sub_period = period_int % 100
        prior_year = year - 1
        return str(prior_year * 100 + sub_period)
    except (ValueError, TypeError):
        return ""


# =============================================================================
# 4. HIGH-LEVEL API: TRAIN + FORECAST + RECONCILE
# =============================================================================

def train_and_reconcile(
    forecasts_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    key_col: str,
    forecast_periods: list,
    time_format: str = 'year_week',
    train_cutoff: Optional[str] = None,
    ratio_min: float = 0.5,
    ratio_max: float = 2.0,
    trust_band: float = 0.1,
    changepoint_prior_scale: float = 0.05,
    yoy_max_deviation: float = 0.15,
    # Phase 3: Method selection and hierarchy support
    method: str = 'top_down',
    hierarchy_col: Optional[str] = None,
    predicted_col: str = 'predicted',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reconciliation dispatcher: top-down (legacy) or hierarchical methods.

    This is the main entry point for all integration points (inference,
    backtesting, validation).

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Key-level forecasts.
    source_df : pd.DataFrame
        Full source data for training the category model.
    target_col : str
        Target column name.
    date_col : str
        Date column name.
    key_col : str
        Key column name.
    forecast_periods : list
        Periods to forecast at category level.
    time_format : str
        'year_week' or 'year_month'.
    train_cutoff : str, optional
        Train on data up to this period.
    ratio_min : float
        Minimum reconciliation ratio.
    ratio_max : float
        Maximum reconciliation ratio.
    trust_band : float
        Trust band width (skip if ratio within 1 +/- trust_band).
    changepoint_prior_scale : float
        Prophet changepoint flexibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        (reconciled_forecasts, diagnostics_dict)
    """
    diagnostics = {}

    # ─── Phase 3: Dispatch to hierarchical reconciliation if not top_down ────
    if method != 'top_down' and hierarchy_col:
        logger.info(f"Using hierarchical reconciliation: method={method}, hierarchy={hierarchy_col}")
        try:
            from utils.hierarchical_reconciliation import run_hierarchical_reconciliation
            reconciled = run_hierarchical_reconciliation(
                forecasts_df=forecasts_df,
                source_df=source_df,
                hierarchy_col=hierarchy_col,
                key_col=key_col,
                date_col=date_col,
                target_col=target_col,
                predicted_col=predicted_col,
                method=method,
                train_cutoff=train_cutoff,
            )
            diagnostics['reconciliation_method'] = method
            diagnostics['reconciliation_applied'] = True
            return reconciled, diagnostics
        except Exception as e:
            logger.warning(f"Hierarchical reconciliation failed ({e}), falling back to top_down")
            diagnostics['hierarchical_reconciliation_error'] = str(e)
            diagnostics['reconciliation_method_attempted'] = method
            diagnostics['reconciliation_method_actual'] = 'top_down'

    # ─── Legacy top-down Prophet reconciliation ─────────────────────────────
    # Step 1: Train category model
    logger.info("=" * 50)
    logger.info("TOP-DOWN RECONCILIATION")
    logger.info("=" * 50)

    model = train_category_model(
        source_df=source_df,
        target_col=target_col,
        date_col=date_col,
        key_col=key_col,
        train_cutoff=train_cutoff,
        time_format=time_format,
        changepoint_prior_scale=changepoint_prior_scale,
    )

    if model is None:
        logger.warning("Category model training failed — returning original forecasts")
        diagnostics['reconciliation_applied'] = False
        return forecasts_df, diagnostics

    # Step 2: Generate category forecasts (pure trend + seasonality)
    cat_forecasts = forecast_category_total(
        model=model,
        forecast_periods=forecast_periods,
        time_format=time_format,
    )

    if cat_forecasts.empty:
        logger.warning("Category forecast generation failed — returning original forecasts")
        diagnostics['reconciliation_applied'] = False
        return forecasts_df, diagnostics

    # Compute pre-reconciliation bottom-up totals for diagnostics
    pre_totals = (
        forecasts_df[forecasts_df['predicted'] > 0]
        .groupby(date_col)['predicted']
        .sum()
    )

    # Step 3: Reconcile
    reconciled_df = reconcile_forecasts(
        forecasts_df=forecasts_df,
        category_forecasts=cat_forecasts,
        date_col=date_col,
        key_col=key_col,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        trust_band=trust_band,
    )

    # Step 4: YoY sanity check — ensure post-reconciliation totals don't
    # swing wildly relative to prior-year same-period actuals
    reconciled_df, yoy_diagnostics = yoy_sanity_check(
        forecasts_df=reconciled_df,
        source_df=source_df,
        date_col=date_col,
        target_col=target_col,
        key_col=key_col,
        time_format=time_format,
        yoy_max_deviation=yoy_max_deviation,
    )

    # Compute post-reconciliation totals for diagnostics (after YoY check)
    post_totals = (
        reconciled_df[reconciled_df['predicted'] > 0]
        .groupby(date_col)['predicted']
        .sum()
    )

    diagnostics['reconciliation_applied'] = True
    diagnostics['n_forecast_periods'] = len(cat_forecasts)
    diagnostics['category_forecast_mean'] = float(cat_forecasts['category_forecast'].mean())
    diagnostics['pre_reconciliation_mean_total'] = float(pre_totals.mean()) if len(pre_totals) > 0 else 0.0
    diagnostics['post_reconciliation_mean_total'] = float(post_totals.mean()) if len(post_totals) > 0 else 0.0
    diagnostics['yoy_sanity_check'] = yoy_diagnostics

    logger.info(
        f"Reconciliation diagnostics: "
        f"category_mean={diagnostics['category_forecast_mean']:.0f}, "
        f"pre_total={diagnostics['pre_reconciliation_mean_total']:.0f}, "
        f"post_total={diagnostics['post_reconciliation_mean_total']:.0f}, "
        f"yoy_dampened={yoy_diagnostics.get('periods_dampened', 0)}"
    )
    logger.info("=" * 50)

    return reconciled_df, diagnostics


# =============================================================================
# HELPER: Coerce cutoff value to match column dtype
# =============================================================================

def _coerce_cutoff(value, col_dtype):
    """Cast *value* to match *col_dtype* so pandas comparisons don't crash.

    Config values (e.g. val_end) arrive as Python strings ("202606") but the
    date column in a DataFrame may be int64 or float64.  A mixed-type
    comparison like ``int64_series <= "202606"`` raises
    ``TypeError: Invalid comparison between dtype=int64 and str``.
    """
    if pd.api.types.is_integer_dtype(col_dtype):
        return int(value)
    if pd.api.types.is_float_dtype(col_dtype):
        return float(value)
    return str(value)


# =============================================================================
# HELPER: Period string to datetime conversion
# =============================================================================

def _period_to_date(periods: pd.Series, time_format: str) -> pd.Series:
    """
    Convert period values (YYYYMM or YYYYWW) to datetime for Prophet.

    Parameters
    ----------
    periods : pd.Series
        Period values as int or str (e.g. 202401, 202452).
    time_format : str
        'year_month' or 'year_week'.

    Returns
    -------
    pd.Series
        Datetime values.
    """
    # Handle dash-separated formats (e.g. '2025-44', '2025-03') by stripping
    # the dash so they become plain YYYYWW / YYYYMM integers.
    periods_cleaned = periods.astype(str).str.strip()
    dash_mask = periods_cleaned.str.contains('-', na=False)
    if dash_mask.any():
        # Split on dash: '2025-44' -> year='2025', part='44'
        parts = periods_cleaned.str.split('-', n=1, expand=True)
        # Reconstruct as YYYYWW / YYYYMM (zero-pad the second part to 2 digits)
        periods_cleaned = pd.Series(
            np.where(
                dash_mask,
                parts[0].fillna('') + parts[1].fillna('').str.zfill(2),
                periods_cleaned,
            ),
            index=periods.index,
        )

    # Convert to int first to strip float decimals (e.g. 202608.0 -> 202608)
    # then to str.  Periods may arrive as float64 when the DataFrame column
    # was widened by a merge/join that introduced NaNs.
    periods_int = pd.to_numeric(periods_cleaned, errors='coerce').astype('Int64')
    periods_str = periods_int.astype(str)

    if time_format == 'year_month':
        # YYYYMM -> first day of month
        dates = pd.to_datetime(periods_str, format='%Y%m')
    else:
        # YYYYWW -> Monday of that ISO week
        def _week_to_date(yw):
            try:
                year = int(str(yw)[:4])
                week = int(str(yw)[4:])
                if week < 1:
                    return pd.NaT
                # ISO 8601 allows week 53 for some years; clamp to valid range
                max_week = pd.Timestamp(year=year, month=12, day=28).isocalendar()[1]
                week = min(week, max_week)
                return pd.Timestamp.fromisocalendar(year, week, 1)
            except (ValueError, TypeError, OverflowError):
                return pd.NaT
        dates = periods_str.apply(_week_to_date)

    return pd.Series(dates, index=periods.index)
