"""Leakage-free rolling promo-lift features.

``promo_lift`` quantifies, for each (key, period), how much the key's
historical sales have risen during promotional weeks compared to its
non-promo weeks. Using the lift as an explicit input feature is well
known to improve gradient-boosted demand models — the model no longer
has to learn the per-key promo elasticity from raw flags, it reads the
magnitude directly.

Strict leakage-freeness for rolling-origin evaluation
-----------------------------------------------------

For a training row with target week ``w``, the lift is computed using
**only** data from periods ``t ≤ w − forecast_lag``. Implementation-
wise this is a cumulative sum shifted by ``forecast_lag``, so the
computation is O(N) with a single pandas groupby pass — the critical
property for a 64 GB Databricks cluster running on ~1 M row
per-category extracts.

Returned features
-----------------

``promo_lift``
    mean(sales | on-promo history) / mean(sales | off-promo history),
    clipped to a configurable band and defaulted to 1.0 when history
    is absent or insufficient.

``promo_lift_sample_size``
    min(number-of-on-promo-history-rows, number-of-off-promo-history-
    rows). Gives the model a signal of how reliable the ``promo_lift``
    estimate is — small sample size means the model should downweight
    the feature.

``promo_lift_any_history``
    0/1 flag indicating whether the key has any history yet. Lets the
    model distinguish "lift = 1 because no info" from "lift = 1 because
    observed lift really is 1".

``promo_lift_recent_13w``
    Same formulation but restricted to the last 13 weeks of history
    (relative to target week). Captures regime shifts — a brand whose
    promo elasticity is decaying will show a lower recent lift than
    lifetime lift.

Design notes
------------

* The function auto-detects promo flag columns by name prefix (``promo_``
  by default). The auto-detection is limited to binary / indicator
  columns; continuous columns that happen to start with ``promo_``
  (like ``promo_cash_discount_per_case_on_invoice``) are excluded.
* An "any-promo-this-week" boolean is derived as the logical OR of all
  detected flags. Collapsing to a single boolean keeps the feature
  tractable and matches how promo events drive demand in practice.
* Clipping bounds (default 0.2–5.0) protect downstream models from
  extreme ratios produced by very thin key × flag combinations.

Typical call site
-----------------

>>> from utils.promo_features import create_promo_lift_features
>>> df = create_promo_lift_features(
...     df, key_cols=['key'], target_col='actual_sales',
...     date_col='year_week', forecast_lag=5,
... )
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Column-name prefixes that mark a candidate promo flag. Kept generic on
# purpose — the auto-detector then filters to binary/near-binary columns.
_PROMO_NAME_PREFIXES = (
    "promo_", "promotion_", "is_promo", "has_promo",
)

# Column names that LOOK like promos but are continuous (discount values,
# prices, lists, descriptions) and must NOT be OR'd into the any-promo
# boolean. These are name-substring matches, case-insensitive.
_PROMO_CONTINUOUS_BLACKLIST = (
    "cash_discount", "_per_case", "_price", "slogan",
    "list_of", "mechanic", "description", "start", "end",
    "features_shipped", "percentage",
)


def _detect_promo_flag_cols(df: pd.DataFrame) -> List[str]:
    """Return the sorted list of columns that look like binary promo flags.

    Criteria:
    * Column name starts with (or contains) one of the promo prefixes.
    * Column name does not match the continuous / descriptive blacklist.
    * Column is numeric and has non-null values strictly in ``{0, 1}``
      (after coercion). Float 0.0/1.0 is allowed.
    """
    detected: List[str] = []
    for col in df.columns:
        low = col.lower()
        if not any(pref in low for pref in _PROMO_NAME_PREFIXES):
            continue
        if any(bad in low for bad in _PROMO_CONTINUOUS_BLACKLIST):
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        uniq = pd.unique(non_null)
        if not np.all(np.isin(uniq, [0, 1, 0.0, 1.0])):
            continue
        detected.append(col)
    return sorted(detected)


def create_promo_lift_features(
    df: pd.DataFrame,
    key_cols: List[str],
    target_col: str,
    date_col: str,
    forecast_lag: int,
    promo_flag_cols: Optional[Iterable[str]] = None,
    min_history_periods: int = 8,
    recent_window: int = 13,
    lift_clip: Tuple[float, float] = (0.2, 5.0),
    output_prefix: str = "promo_lift",
) -> pd.DataFrame:
    """Append leakage-free promo-lift features to ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form data with one row per (key, period). Must contain
        ``target_col`` and a ``date_col`` sortable in chronological
        order.
    key_cols : List[str]
        Columns identifying the time series key. Typically ``['key']``.
    target_col : str
        Numeric target column (e.g. ``'actual_sales'``).
    date_col : str
        Time period column. Sorted lexicographically — since the upstream
        pipeline normalises periods to YYYYWW / YYYYMM, lexicographic
        order equals chronological order.
    forecast_lag : int
        Minimum forecast horizon (in periods). All running statistics are
        shifted by this amount so a row for target week ``w`` only sees
        data from periods ``t ≤ w − forecast_lag``.
    promo_flag_cols : Iterable[str], optional
        Explicit list of binary promo flag columns. When ``None`` the
        detector scans ``df`` for columns that match a promo name
        prefix and are binary. Columns not present in ``df`` are
        silently skipped.
    min_history_periods : int
        When the total number of historical periods (in either the
        on-promo or off-promo buckets) is below this threshold, the
        emitted ``promo_lift`` is forced to 1.0 and the
        ``promo_lift_any_history`` flag is 0. Prevents noisy lifts from
        tiny samples.
    recent_window : int
        Window (in periods) for the recent-lift variant. Default 13
        (one quarter of weekly data).
    lift_clip : (low, high)
        Multiplicative clipping range for the final lift. Applied after
        division. Typical range ``(0.2, 5.0)``.
    output_prefix : str
        Base name for emitted columns. Defaults to ``'promo_lift'``.

    Returns
    -------
    pd.DataFrame
        Same dataframe with four new columns:
        ``promo_lift``, ``promo_lift_sample_size``,
        ``promo_lift_any_history``, ``promo_lift_recent_13w``.
    """
    if not key_cols:
        logger.warning("promo_lift: no key columns — skipping")
        return df
    key_col = key_cols[0]
    if key_col not in df.columns or target_col not in df.columns or date_col not in df.columns:
        logger.warning("promo_lift: missing key/date/target column — skipping")
        return df

    # ---- 1. Resolve flag columns ---------------------------------------
    if promo_flag_cols is None:
        flag_cols = _detect_promo_flag_cols(df)
    else:
        flag_cols = [c for c in promo_flag_cols if c in df.columns]

    if not flag_cols:
        logger.info("promo_lift: no promo flag columns detected — emitting neutral features")
        df[f"{output_prefix}"] = 1.0
        df[f"{output_prefix}_sample_size"] = 0
        df[f"{output_prefix}_any_history"] = 0
        df[f"{output_prefix}_recent_{recent_window}w"] = 1.0
        return df

    logger.info(
        "promo_lift: using %d flag columns (first 5: %s)",
        len(flag_cols), flag_cols[:5],
    )

    # ---- 2. Sort once, build helper columns -----------------------------
    df = df.sort_values(key_cols + [date_col], kind="stable").reset_index(drop=True)

    target_safe = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # any_promo = 1 if any flag is 1 this row (logical OR across flags)
    flag_block = df[flag_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    any_promo = (flag_block.sum(axis=1) > 0).astype(np.int8)

    # ---- 3. Running sums per key, shifted by forecast_lag --------------
    # shift(forecast_lag) guarantees a row for week w sees only data at
    # t ≤ w − forecast_lag. Runs purely inside pandas — no python loops.
    grp = df.groupby(key_col, sort=False)

    on_sales_cum = (target_safe * any_promo).groupby(df[key_col]).cumsum()
    off_sales_cum = (target_safe * (1 - any_promo)).groupby(df[key_col]).cumsum()
    on_count_cum = any_promo.groupby(df[key_col]).cumsum()
    off_count_cum = (1 - any_promo).groupby(df[key_col]).cumsum()

    # Shift each cumulative so the "history" observed at row i excludes
    # rows i − forecast_lag + 1 … i. Forecast_lag=5 → shift by 5.
    lag = max(1, int(forecast_lag))
    on_sales_shift = on_sales_cum.groupby(df[key_col]).shift(lag).fillna(0.0)
    off_sales_shift = off_sales_cum.groupby(df[key_col]).shift(lag).fillna(0.0)
    on_count_shift = on_count_cum.groupby(df[key_col]).shift(lag).fillna(0).astype(np.int32)
    off_count_shift = off_count_cum.groupby(df[key_col]).shift(lag).fillna(0).astype(np.int32)

    # ---- 4. Compute lift + sample size + any-history flag --------------
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_on = np.where(on_count_shift > 0, on_sales_shift / np.maximum(on_count_shift, 1), np.nan)
        mean_off = np.where(off_count_shift > 0, off_sales_shift / np.maximum(off_count_shift, 1), np.nan)
        eps = 1e-9
        lift = mean_on / np.maximum(mean_off, eps)

    # Neutral defaults: when either bucket has zero history (or nearly
    # zero), we cannot estimate a lift — force to 1.0 and drop the
    # "has history" flag. Same when total history is below the minimum.
    total_hist = on_count_shift + off_count_shift
    insufficient = (
        (on_count_shift <= 0)
        | (off_count_shift <= 0)
        | (total_hist < int(min_history_periods))
        | np.isnan(lift)
    )
    lift = np.where(insufficient, 1.0, lift)

    # Clip extremes
    low, high = float(lift_clip[0]), float(lift_clip[1])
    lift = np.clip(lift, low, high)

    sample_size = np.minimum(on_count_shift, off_count_shift)

    df[f"{output_prefix}"] = lift.astype(np.float32)
    df[f"{output_prefix}_sample_size"] = sample_size.astype(np.int32)
    df[f"{output_prefix}_any_history"] = (~insufficient).astype(np.int8)

    # ---- 5. Recent-window variant (last ``recent_window`` periods) -----
    # Rolling on the shifted series: cumulative over the last N after
    # excluding the forecast_lag window. Done via a length-N rolling
    # sum of the on/off indicators (shifted by forecast_lag).
    target_shift = target_safe.groupby(df[key_col]).shift(lag).fillna(0.0)
    any_shift = any_promo.groupby(df[key_col]).shift(lag).fillna(0).astype(np.int32)

    on_sales_roll = (target_shift * any_shift).groupby(df[key_col]).rolling(
        window=recent_window, min_periods=1
    ).sum().reset_index(level=0, drop=True)
    off_sales_roll = (target_shift * (1 - any_shift)).groupby(df[key_col]).rolling(
        window=recent_window, min_periods=1
    ).sum().reset_index(level=0, drop=True)
    on_count_roll = any_shift.groupby(df[key_col]).rolling(
        window=recent_window, min_periods=1
    ).sum().reset_index(level=0, drop=True).astype(np.int32)
    off_count_roll = (1 - any_shift).groupby(df[key_col]).rolling(
        window=recent_window, min_periods=1
    ).sum().reset_index(level=0, drop=True).astype(np.int32)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_on_r = np.where(on_count_roll > 0,
                             on_sales_roll / np.maximum(on_count_roll, 1), np.nan)
        mean_off_r = np.where(off_count_roll > 0,
                              off_sales_roll / np.maximum(off_count_roll, 1), np.nan)
        lift_r = mean_on_r / np.maximum(mean_off_r, eps)

    insufficient_r = (
        (on_count_roll <= 0) | (off_count_roll <= 0) | np.isnan(lift_r)
    )
    lift_r = np.where(insufficient_r, 1.0, lift_r)
    lift_r = np.clip(lift_r, low, high)
    df[f"{output_prefix}_recent_{recent_window}w"] = lift_r.astype(np.float32)

    logger.info(
        "promo_lift: added 4 features. Coverage: %.1f%% rows have any history; "
        "mean lift=%.2f",
        100 * (df[f"{output_prefix}_any_history"].mean()),
        float(df.loc[df[f"{output_prefix}_any_history"] == 1, f"{output_prefix}"].mean()
              or 0.0),
    )
    return df


__all__ = ["create_promo_lift_features"]
