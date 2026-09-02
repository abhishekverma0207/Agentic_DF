"""Lightweight DataFrame memory optimisations used by the backtester.

The UK source data routinely has ~60 object columns across 1.1 M rows
(brand names, APG descriptions, promo-feature strings, etc.). Pandas
stores each object value as a separate Python string with roughly
~45-60 bytes of overhead — for low-cardinality columns this is 30-50×
more than a ``category`` representation costs.

On a 64 GB Databricks cluster the backtester keeps several DataFrames
in flight per origin (source, train_features, val_features, test_features,
per-segment subsets). Shrinking source_df from ~4 GB to ~1 GB cascades
into every downstream copy and is frequently the difference between a
comfortable run and an OOM.

This module provides a single entry point :func:`optimise_dataframe_memory`
that is safe to call once at the top of a run. It is deliberately
conservative:

* Object columns are cast to ``category`` only when the distinct-value
  fraction is below ``max_category_ratio`` (default 0.5). For columns
  where nearly every row has a unique value (e.g. free-text descriptions
  that vary per row) ``category`` would actually be *larger* than
  ``object`` — we leave those alone.
* Float64 → float32 is applied only when every value fits in the
  float32 range with negligible precision loss (default off; opt-in
  via ``downcast_floats=True``) because some downstream utilities
  assume float64.
* Int64 is downcast to the smallest signed/unsigned integer that fits
  the observed range. This is always safe.

The function is a no-op for any DataFrame under a configurable byte
threshold, so calling it on small intermediates is free.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _deep_mem_bytes(df: pd.DataFrame) -> int:
    try:
        return int(df.memory_usage(index=True, deep=True).sum())
    except Exception:
        return 0


def optimise_dataframe_memory(
    df: pd.DataFrame,
    *,
    max_category_ratio: float = 0.5,
    downcast_integers: bool = True,
    downcast_floats: bool = False,
    protect_cols: Optional[Iterable[str]] = None,
    min_bytes_threshold: int = 100 * 1024 * 1024,   # 100 MB
    log_prefix: str = "optimise_dataframe_memory",
) -> Tuple[pd.DataFrame, dict]:
    """Shrink a DataFrame's memory footprint in-place where safe.

    Parameters
    ----------
    df : pd.DataFrame
        The frame to optimise. Modified in place (and returned).
    max_category_ratio : float, optional
        Object columns are cast to ``category`` only when
        ``nunique(dropna=True) / len(df) <= max_category_ratio``.
        Default 0.5. Set lower (e.g. 0.05) for a more aggressive policy.
    downcast_integers : bool, optional
        When True, integer columns are downcast to the smallest fitting
        ``np.int*`` / ``np.uint*`` type. Always safe. Default True.
    downcast_floats : bool, optional
        When True, ``float64`` columns are downcast to ``float32`` when
        safe. Default False — some downstream code assumes ``float64``.
    protect_cols : Iterable[str], optional
        Column names that must retain their original dtype. Typical
        examples: the timestamp / target / key columns.
    min_bytes_threshold : int
        Skip optimisation entirely if the frame is smaller than this
        many bytes. Default 100 MB — there's no point shrinking small
        intermediates.
    log_prefix : str
        Prefix for info / debug messages.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        The (modified) DataFrame and a small diagnostics dict with
        ``before_bytes``, ``after_bytes``, and counts of columns changed.
    """
    protect = set(protect_cols or [])

    before = _deep_mem_bytes(df)
    if before < min_bytes_threshold:
        return df, {
            "before_bytes": before,
            "after_bytes": before,
            "skipped": True,
            "reason": "below min_bytes_threshold",
        }

    n_cat = 0
    n_int_down = 0
    n_float_down = 0
    n_rows = len(df)

    for col in df.columns:
        if col in protect:
            continue
        s = df[col]
        dtype = s.dtype

        # ---- Object → category ---------------------------------------
        if dtype == object:
            try:
                nu = s.nunique(dropna=True)
                if nu > 0 and (nu / max(n_rows, 1)) <= max_category_ratio:
                    df[col] = s.astype("category")
                    n_cat += 1
            except Exception as exc:
                logger.debug("%s: skip %s (object→category failed: %s)", log_prefix, col, exc)
            continue

        # ---- Integer downcast ---------------------------------------
        if downcast_integers and pd.api.types.is_integer_dtype(dtype):
            try:
                # Choose signed or unsigned based on observed minimum
                s_min = s.min()
                s_max = s.max()
                if pd.isna(s_min) or pd.isna(s_max):
                    continue
                if s_min >= 0:
                    new = pd.to_numeric(s, downcast="unsigned")
                else:
                    new = pd.to_numeric(s, downcast="signed")
                if new.dtype != dtype:
                    df[col] = new
                    n_int_down += 1
            except Exception as exc:
                logger.debug("%s: skip %s (int downcast failed: %s)", log_prefix, col, exc)
            continue

        # ---- Float downcast (opt-in) --------------------------------
        if downcast_floats and pd.api.types.is_float_dtype(dtype):
            try:
                new = pd.to_numeric(s, downcast="float")
                if new.dtype != dtype:
                    df[col] = new
                    n_float_down += 1
            except Exception as exc:
                logger.debug("%s: skip %s (float downcast failed: %s)", log_prefix, col, exc)
            continue

    after = _deep_mem_bytes(df)
    saved = before - after
    pct = (100.0 * saved / max(before, 1)) if before else 0.0
    logger.info(
        "%s: %.0f MB → %.0f MB (%.1f%% saved). cat=%d int_down=%d float_down=%d",
        log_prefix, before / 1024 / 1024, after / 1024 / 1024, pct,
        n_cat, n_int_down, n_float_down,
    )

    return df, {
        "before_bytes": before,
        "after_bytes": after,
        "saved_bytes": saved,
        "saved_pct": round(pct, 2),
        "n_cat": n_cat,
        "n_int_down": n_int_down,
        "n_float_down": n_float_down,
        "skipped": False,
    }


__all__ = ["optimise_dataframe_memory"]
