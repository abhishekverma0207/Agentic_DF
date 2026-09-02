"""
MOQ (Minimum Order Quantity) Post-Processing for Forecast Output.

Rounds point forecasts to multiples of a detected MOQ value for SKUs whose
historical sales show a strong MOQ ordering pattern (e.g. New-Product-
Development items, reintroduced products, highly-promoted items, or MOQ-
driven orders). These patterns are common on lumpy / intermittent demand
where the retailer reorders in fixed case-pack / pallet quantities — a
regression-style point forecast will predict a continuous value that, if
served unrounded, disagrees with the real ordering rhythm and inflates
per-period error.

How the detector works (ported from the legacy Lego `postprocessing_for_
strong_moq` function and hardened):

1. Collect the non-zero historical sales values for the key, up to the
   training cutoff. Require at least 30 non-zero observations and a total
   volume of 60 — smaller samples aren't stable enough to fit a fixed
   MOQ.
2. Compute p40..p80 of the non-zero sales. A strong MOQ is declared when
   either:
      p40 >= 48 AND (p70 % p40 == 0 OR p60 % p40 == 0)
   or p50 >= 48 AND (p70 % p50 == 0 OR p80 % p50 == 0)
   i.e. the smaller percentile evenly divides a higher percentile,
   meaning most real order sizes are integer multiples of the same base.
3. Given a detected MOQ ``m``, sweep ceiling-rounding thresholds t in
   [0.15, 0.9] in 0.05 steps. For each t, round the future forecasts to
   multiples of ``m`` (rounding up when fractional part > t) and compute
   the ratio of the mean future forecast to the mean recent-history sale.
   Pick the t whose ratio is closest to 1.0.
4. Emit those rounded values as ``prediction_post_moq`` for the key's
   future rows. For rows where MOQ was NOT detected (or the key has no
   qualifying history), ``prediction_post_moq`` defaults to the existing
   point forecast so downstream consumers can read this single column
   unconditionally.

Design choices vs. the legacy implementation:

- The Lego_segments gate (``05: NPD``, ``06: Reintroduced``, ``07: Highly
  Promoted``, ``09: MOQ Orders``) is dropped because the FEU Agentic
  pipeline doesn't produce that taxonomy. ``check_strong_moq`` is already
  a strict gate on its own — percentile divisibility + minimum volume
  rarely fires on smooth / erratic SKUs. If ``Lego_segments`` is present
  in the forecast DataFrame we still honour it as an additional filter
  (optional; controlled by ``honour_lego_segments_if_present``).
- ``prediction_post_moq`` ALWAYS populates (falls back to the raw
  forecast when MOQ isn't detected), unlike the legacy NaN-fallback.
  Downstream DIQ code can read one column and be done.
- Safe positional access via ``.iloc[0]``, proper logger instead of
  ``print(e)``, explicit NaN / empty-series guards, and a clean merge
  back into the full forecasts DataFrame rather than mutating in place.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The four Lego segment names that the legacy implementation whitelisted
# for MOQ post-processing. We accept this set as an OPTIONAL extra gate
# if the caller merges a Lego_segments column into the DataFrame.
_LEGO_MOQ_SEGMENTS = frozenset([
    "05: NPD (26-52)",
    "06: Reintroduced product",
    "07: Highly Promoted",
    "09: MOQ Orders",
])

# Minimum counts for MOQ detection. Below these, percentile estimates are
# unstable and we skip.
_MIN_NONZERO_OBSERVATIONS = 30
_MIN_NONZERO_VOLUME_SUM = 60

# Percentile floor for a value to be considered a candidate MOQ. 48 was
# chosen in the legacy code as roughly one pallet / one case-pack layer;
# below that you tend to be picking up noise rather than a packaging
# multiple. Kept for parity with the original behaviour.
_MOQ_CANDIDATE_FLOOR = 48

# Threshold sweep: which ceiling-rounding thresholds to try. Stepping
# from 0.9 down to 0.15 in 0.05 steps gives 16 candidates. The best
# threshold is the one whose rounded future-forecast mean is closest to
# the last 52 weeks' history mean.
_THRESHOLD_SWEEP = np.round(np.arange(0.9, 0.1, -0.05), 2)


# ---------------------------------------------------------------------------
# Detection primitives (pure functions, easy to unit-test)
# ---------------------------------------------------------------------------

# Relative tolerance for the "integer-multiple" check on percentile pairs.
# Real case-pack / pallet sales values are always exact integers, but we
# allow up to 2% drift on the ratio to survive floating-point jitter
# introduced by upstream aggregation (e.g. unit conversions, scaling).
_INT_MULTIPLE_TOL = 0.02


def _is_integer_multiple(larger: float, smaller: float, tol: float = _INT_MULTIPLE_TOL) -> bool:
    """True when ``larger`` is approximately an integer multiple (>= 1x)
    of ``smaller``. Uses a relative tolerance on the ratio so 119.98 is
    still "2x 60" — important because raw percentile values drift under
    upstream aggregation / float rounding even when the underlying
    orders are clean integers.
    """
    if smaller <= 0 or larger <= 0:
        return False
    ratio = larger / smaller
    rounded = round(ratio)
    if rounded < 1:
        return False
    return abs(ratio - rounded) / rounded <= tol


def check_strong_moq(past_sales) -> float:
    """Detect a strong MOQ pattern from a 1-D sequence of historical sales.

    Returns the detected MOQ value (p40 or p50 of non-zero sales) when a
    strong pattern is found, or 0.0 when no MOQ is detected. Ignores zeros
    in the input series.
    """
    nonzero = [float(x) for x in past_sales if x is not None and not (isinstance(x, float) and np.isnan(x)) and x > 0]
    if len(nonzero) <= _MIN_NONZERO_OBSERVATIONS or np.sum(nonzero) <= _MIN_NONZERO_VOLUME_SUM:
        return 0.0

    p40 = float(np.percentile(nonzero, 40))
    p50 = float(np.percentile(nonzero, 50))
    p60 = float(np.percentile(nonzero, 60))
    p70 = float(np.percentile(nonzero, 70))
    p80 = float(np.percentile(nonzero, 80))

    # Primary gate: p40 evenly divides p70 or p60
    if p40 >= _MOQ_CANDIDATE_FLOOR and (
        _is_integer_multiple(p70, p40) or _is_integer_multiple(p60, p40)
    ):
        return p40

    # Fallback: p50 evenly divides p70 or p80
    if p50 >= _MOQ_CANDIDATE_FLOOR and (
        _is_integer_multiple(p70, p50) or _is_integer_multiple(p80, p50)
    ):
        return p50

    return 0.0


def custom_round(value: float, fact: float, threshold: float) -> float:
    """Ceiling-round ``value`` up to the next multiple of ``fact`` whose
    fractional part exceeds ``threshold``, otherwise floor to the current
    multiple.

    Examples:
      custom_round(130, 60, 0.5) -> 120  (130/60 = 2.17, 0.17 < 0.5)
      custom_round(160, 60, 0.5) -> 180  (160/60 = 2.67, 0.67 > 0.5)
    """
    if fact <= 0 or not np.isfinite(value):
        return 0.0 if not np.isfinite(value) else float(value)
    if value <= 0:
        return 0.0
    dval = value / fact
    whole = int(dval)
    frac = dval - whole
    rounded = (whole + 1) if frac > threshold else whole
    return float(rounded * fact)


def _pick_best_threshold(
    moq: float,
    future_forecasts: np.ndarray,
    recent_history: np.ndarray,
) -> float:
    """For a given MOQ and a key's future-forecast + recent-history
    arrays, sweep rounding thresholds and pick the one whose rounded
    future-forecast mean is closest to the recent-history mean.

    Returns the chosen threshold. If the recent history mean is zero or
    the future is empty we return 0.5 as a sensible default.
    """
    future_forecasts = np.asarray(future_forecasts, dtype=float)
    recent_history = np.asarray(recent_history, dtype=float)

    if future_forecasts.size == 0 or recent_history.size == 0:
        return 0.5
    hist_mean = float(np.mean(recent_history))
    if not np.isfinite(hist_mean) or hist_mean == 0.0:
        return 0.5

    # Only the first 53 forecast weeks contribute to the ratio — that's
    # the comparable horizon for "annualised mean" vs the prior 52w of
    # history. Matches the legacy implementation.
    future_cmp = future_forecasts[:min(53, future_forecasts.size)]

    best_t, best_abs_diff = 0.5, np.inf
    for t in _THRESHOLD_SWEEP:
        rounded = np.array([custom_round(v, moq, t) for v in future_cmp], dtype=float)
        if rounded.size == 0:
            continue
        fut_mean = float(np.mean(rounded))
        ratio = fut_mean / hist_mean
        abs_diff = abs(1.0 - ratio)
        if abs_diff < best_abs_diff:
            best_abs_diff, best_t = abs_diff, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_moq_postprocessing(
    forecasts_df: pd.DataFrame,
    config,
    *,
    history_df: Optional[pd.DataFrame] = None,
    forecast_col: str = 'predicted',
    output_col: str = 'prediction_post_moq',
    history_till: Optional[str] = None,
    honour_lego_segments_if_present: bool = True,
) -> pd.DataFrame:
    """Add an MOQ-rounded forecast column to ``forecasts_df``.

    Parameters
    ----------
    forecasts_df
        DataFrame with one row per (key, forecast_period). Must contain a
        key column (``config.prediction_key_cols[0]`` or ``'key'``), a
        date/period column (``config.timestamp_col``), and ``forecast_col``.
    config
        The fully-resolved DemandForecastConfig. Used for key / date /
        target / train_end.
    history_df
        Optional raw source DataFrame with columns (key, date, target).
        If not provided, the function loads from ``config.input_data_path``
        via ``utils.agent_utilities.load_source_data``.
    forecast_col
        Name of the point-forecast column in ``forecasts_df``. Default
        ``'predicted'`` — the column produced by the FEU Agentic pipeline.
    output_col
        Name of the new column to add. Default ``'prediction_post_moq'``.
        Always populated (falls back to ``forecasts_df[forecast_col]``
        for keys where no MOQ is detected).
    history_till
        Optional override for the training-history cutoff used to detect
        MOQ patterns. Defaults to ``config.train_end`` which is correct
        for inference and backtest-with-fixed-history-window.
    honour_lego_segments_if_present
        If True (default) and the DataFrame carries a ``Lego_segments``
        column, restrict MOQ post-processing to the four legacy lego
        codes ("05: NPD (26-52)", "06: Reintroduced product", etc.).
        If the column is absent this flag is a no-op and every key is
        considered — the MOQ detector itself is the primary gate.

    Returns
    -------
    pd.DataFrame
        A copy of ``forecasts_df`` with ``output_col`` appended.
    """
    # --- Defensive column resolution -----------------------------------
    if forecast_col not in forecasts_df.columns:
        logger.warning(
            f"MOQ post-processing skipped: forecast column '{forecast_col}' "
            f"not in forecasts_df (columns: {list(forecasts_df.columns)[:10]}…)"
        )
        out = forecasts_df.copy()
        out[output_col] = np.nan
        return out

    # We use 'key' consistently — FEU standardises the key column to 'key'
    # post feature-engineering. Fall back to the configured prediction key.
    key_col = 'key' if 'key' in forecasts_df.columns else config.prediction_key_cols[0]
    date_col = config.timestamp_col
    target_col = config.target_col
    effective_history_till = history_till or getattr(config, 'train_end', None)

    if effective_history_till in (None, '', 'None'):
        logger.warning(
            "MOQ post-processing skipped: no effective history_till "
            "(config.train_end is empty). Set train_end or pass history_till."
        )
        out = forecasts_df.copy()
        out[output_col] = forecasts_df[forecast_col]
        return out

    # --- History source ------------------------------------------------
    if history_df is None:
        try:
            from utils.agent_utilities import load_source_data
            history_df = load_source_data(config.input_data_path)
        except Exception as e:
            logger.warning(
                f"MOQ post-processing: could not load source data "
                f"({type(e).__name__}: {e}). Passing through raw forecasts."
            )
            out = forecasts_df.copy()
            out[output_col] = forecasts_df[forecast_col]
            return out

    # Require the three columns we rely on. 'key' on source_df may or may
    # not be present — FEU synthesises it from prediction_key_cols when
    # missing, replicate that behaviour here so we don't need to depend
    # on upstream preprocessing order.
    if 'key' not in history_df.columns:
        if len(config.prediction_key_cols) == 1 and config.prediction_key_cols[0] in history_df.columns:
            history_df = history_df.copy()
            history_df['key'] = history_df[config.prediction_key_cols[0]]
        else:
            history_df = history_df.copy()
            history_df['key'] = history_df[config.prediction_key_cols].astype(str).agg('_'.join, axis=1)

    missing_hist_cols = [c for c in [date_col, target_col, 'key'] if c not in history_df.columns]
    if missing_hist_cols:
        logger.warning(
            f"MOQ post-processing skipped: history_df missing {missing_hist_cols}. "
            "Passing through raw forecasts."
        )
        out = forecasts_df.copy()
        out[output_col] = forecasts_df[forecast_col]
        return out

    # Normalise the period column on both sides (tolerate int vs str).
    # Compare as strings so '202548' <= '202548' holds regardless of
    # dtype; we never need arithmetic on these period values inside MOQ.
    history_df = history_df[[key_col, date_col, target_col]].copy()
    history_df[date_col] = history_df[date_col].astype(str)
    cutoff = str(effective_history_till)
    history_df = history_df[history_df[date_col] <= cutoff]
    if history_df.empty:
        logger.warning(
            f"MOQ post-processing: no history rows at or before {cutoff}. "
            "Passing through raw forecasts."
        )
        out = forecasts_df.copy()
        out[output_col] = forecasts_df[forecast_col]
        return out

    # --- Initialise output column with the pass-through default -------
    out = forecasts_df.copy()
    out[output_col] = out[forecast_col]

    # Build a boolean eligibility mask keyed on each forecast row.
    eligibility_mask = pd.Series(True, index=out.index)
    if honour_lego_segments_if_present and 'Lego_segments' in out.columns:
        eligibility_mask = out['Lego_segments'].astype(str).str.strip().isin(_LEGO_MOQ_SEGMENTS)
        n_elig_keys = out.loc[eligibility_mask, key_col].nunique()
        logger.info(
            f"MOQ post-processing: Lego_segments filter narrows universe to "
            f"{n_elig_keys} keys (across {_LEGO_MOQ_SEGMENTS})"
        )

    # Forecast rows we should consider for re-rounding: eligible AND date
    # strictly after history cutoff (we never rewrite pre-cutoff rows,
    # even if somehow present in forecasts_df).
    out['_moq_date_str'] = out[date_col].astype(str)
    future_row_mask = (out['_moq_date_str'] > cutoff) & eligibility_mask

    eligible_keys = out.loc[future_row_mask, key_col].unique()
    if len(eligible_keys) == 0:
        logger.info("MOQ post-processing: no eligible (key, future-period) rows; skipping.")
        out.drop(columns=['_moq_date_str'], inplace=True)
        return out

    # Pre-group the history once so each per-key loop is an O(n_obs) scan.
    history_by_key = {
        k: g[target_col].to_numpy(dtype=float)
        for k, g in history_df.groupby('key', sort=False)
    }

    n_keys_with_moq = 0
    n_rows_rewritten = 0
    moq_stats = []

    # --- Per-key MOQ detection + rounding -----------------------------
    for key_val in eligible_keys:
        past = history_by_key.get(key_val)
        if past is None or past.size == 0:
            continue
        moq = check_strong_moq(past)
        if moq <= 0:
            continue

        key_row_mask = (out[key_col] == key_val) & future_row_mask
        if not key_row_mask.any():
            continue

        future_forecasts = out.loc[key_row_mask, forecast_col].to_numpy(dtype=float)
        recent_history = past[max(0, past.size - 52):]

        best_t = _pick_best_threshold(moq, future_forecasts, recent_history)
        rounded = np.array([custom_round(v, moq, best_t) for v in future_forecasts], dtype=float)

        out.loc[key_row_mask, output_col] = rounded
        n_keys_with_moq += 1
        n_rows_rewritten += int(key_row_mask.sum())
        moq_stats.append((str(key_val), float(moq), float(best_t), float(recent_history.mean()) if recent_history.size else 0.0))

    out.drop(columns=['_moq_date_str'], inplace=True)

    logger.info(
        "MOQ post-processing complete: %d keys had a strong MOQ pattern, "
        "%d forecast rows were rounded (out of %d eligible future rows).",
        n_keys_with_moq,
        n_rows_rewritten,
        int(future_row_mask.sum()),
    )
    if moq_stats:
        # Log the 5 largest MOQ values for visibility (usually these are
        # the keys that matter most to the business — big pallet sizes).
        moq_stats.sort(key=lambda t: t[1], reverse=True)
        top = moq_stats[:5]
        logger.info("  Top 5 MOQ keys: %s",
                    [(k, f"moq={m:.0f}", f"thsd={t:.2f}") for k, m, t, _ in top])

    return out


__all__ = [
    'check_strong_moq',
    'custom_round',
    'apply_moq_postprocessing',
]
