"""
Forecast blending via WAPE minimisation with bias control.

Self-contained utility for fitting per-group blend weights between two
forecasts (e.g. Lego and DIQ) to minimise WAPE on actuals, with
configurable bias treatment.  Designed to be called manually on a
monthly cadence after a fresh DIQ run produces forecasts that need to
be blended with an external Lego forecast.

This module is deliberately decoupled from the rest of `utils/`:
nothing else in the pipeline imports it, and it imports only common
scientific-Python deps (numpy / pandas / pyspark / scipy) that are
already pinned in `requirements.txt`.  It is safe to ignore on
day-to-day runs and to call ad-hoc when you want to refresh weights.

Modes
-----
    'dominant' : minimise WAPE s.t. |bias_blend| <= min(|bias_a|, |bias_b|)
                 The blend is at least as unbiased as the better
                 component, with strictly lower (or equal) WAPE.
                 [DEFAULT — recommended for almost all use cases]
    'hard'     : minimise WAPE s.t. bias_blend == 0 exactly.
                 Use when downstream consumers require exact volume
                 reconciliation at the group level.
    'soft'     : minimise WAPE + lambda * |bias|. Tunable trade-off.
    'none'     : pure WAPE minimisation, no bias control.

The default fallback weights are (1.0, 0.0) — i.e. fall back to the
first forecast unmodified — which is the safest "do no harm" behaviour
when a group can't be optimised.  List the safer / baseline forecast
first in `forecast_cols`.

Typical usage
-------------
>>> from utils.forecast_blend import fit_forecast_blend
>>> weights = fit_forecast_blend(
...     df,                                  # spark DataFrame, long format
...     group_cols=["category", "lego_segment"],
...     forecast_cols=["Lego", "DIQ"],       # fallback => 100% Lego
...     actuals_col="Actuals",
...     time_col="year_week",
... )  # bias_mode='dominant' by default
>>> weights.show()

Cadence note
------------
Run manually after each fresh DIQ run produces a new monthly forecast
that needs to be blended with the latest Lego output.  The returned
DataFrame has one row per group, with weights + diagnostic metrics —
persist it (Delta / parquet) and apply the weights downstream as a
join on group_cols.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from scipy.optimize import minimize, minimize_scalar


__all__ = ["fit_forecast_blend", "save_weights", "load_weights"]


# Numerical threshold below which a fractional-bias bound is treated
# as zero (causing dominant mode to collapse to hard mode).
_M_TOL = 1e-6


# Default column renames applied at save-time so the persisted artifact
# matches the column naming downstream consumers expect.  In particular
# `w_DIQ -> diq_pcnt` matches the legacy lookup-table convention (where
# diq_pcnt was the per-cs_gtin weight on DIQ), making it a drop-in
# replacement for any apply code that joins on a column called
# "diq_pcnt".  Only applied to columns that exist; missing keys are
# silently ignored.
_DEFAULT_OUTPUT_COLUMN_RENAMES: dict = {
    "w_DIQ": "diq_pcnt",
}


# ---------- artifact helpers -----------------------------------------

def _infer_metadata_from_schema(columns: Sequence[str]) -> Tuple[list, list]:
    """Recover (group_cols, forecast_cols) from a fit_forecast_blend output schema.

    The output schema is built as:
        <group_cols>, w_<f1>, w_<f2>, wape_<f1>, wape_<f2>, wape_blend,
        bias_<f1>, bias_<f2>, bias_blend, n_obs, [n_periods,] status

    So any column starting with ``w_`` / ``wape_`` / ``bias_``, or
    matching one of the diagnostic-exact names, is NOT a group column.
    The forecast columns are recovered by stripping the ``w_`` prefix
    off the weight columns.
    """
    cols = list(columns)
    weight_cols = [c for c in cols if c.startswith("w_")]
    forecast_cols = [c[len("w_"):] for c in weight_cols]

    diagnostic_prefixes = ("w_", "wape_", "bias_")
    diagnostic_exact = {"n_obs", "n_periods", "status", "wape_blend", "bias_blend"}
    group_cols = [
        c for c in cols
        if not c.startswith(diagnostic_prefixes) and c not in diagnostic_exact
    ]
    return group_cols, forecast_cols


def save_weights(
    weights_df,
    output_path: str,
    *,
    snapshot_week: str,
    bias_mode: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    round_to: Optional[int] = 3,
    filename: Optional[str] = None,
    also_write_json: bool = False,
) -> str:
    """Persist fitted blend weights as a self-describing JSON manifest.

    Wraps the weights DataFrame with a metadata header (generated_at,
    snapshot_week, bias_mode, group_cols, forecast_cols, n_groups, ...)
    so the file is audit-friendly and downstream consumers can apply
    the weights with a one-line ``load_weights(...)`` call.

    Parameters
    ----------
    weights_df
        DataFrame returned by ``fit_forecast_blend`` (Spark or pandas).
        Columns are inspected to auto-infer ``group_cols`` and
        ``forecast_cols`` so they don't need to be passed explicitly.
    output_path
        Where to write the lookup CSV.  Two accepted forms:
        - DIRECTORY (most common): if ``output_path`` does NOT end in
          ``.csv``, it's treated as a directory and the file lands
          at ``{output_path}/{filename}``.
        - LITERAL PATH: if ``output_path`` ends in ``.csv``, it's
          used verbatim (caller controls everything).
    snapshot_week
        REQUIRED. The period label this manifest applies to — typically
        the year_week or year_month string from the run that produced
        the input forecasts (e.g. ``"202615"`` or ``"2026-15"``).
        This goes into BOTH the folder name (when ``use_snapshot_subdir``
        is True) AND the manifest header, so downstream consumers know
        which forecast period the weights were fitted for.  We do NOT
        derive this from the system clock — the run date and the
        forecast snapshot can legitimately differ (e.g. when
        backfilling weights for an earlier period).
    use_snapshot_subdir
        When True (default), append ``{snapshot_week}/weights.json``
        to ``output_path`` so monthly snapshots accumulate as siblings
        under a single root.  Set False if you want a flat layout or
        a custom filename.
    bias_mode
        Optional, recorded in the manifest for audit.  Pass the same
        ``bias_mode`` you passed to ``fit_forecast_blend``.
    extra_metadata
        Optional dict merged into the manifest header.  Use for
        run-specific context (input_path, run_id, etc.).  Reserved
        keys (``generated_at_utc``, ``schema``, ``weights``,
        ``n_groups``, ``snapshot_week``, ``bias_mode``,
        ``group_cols``, ``forecast_cols``) won't be overwritten.
    filename : str or None, default None
        Filename to write under ``output_path`` when it's a directory.
        When ``None`` (default), uses
        ``diq_gtin_lookup_{YYYYMMDD}.csv`` where YYYYMMDD is today's
        UTC date.  Pass a specific name (must end in ``.csv``) to
        override — useful for backfills where you want a specific
        date stamp not equal to today.
    also_write_json : bool, default False
        When True, ALSO write a JSON manifest sibling at the same
        path with ``.json`` extension.  The JSON contains the full
        metadata header (generated_at_utc, snapshot_week, bias_mode,
        group_cols, forecast_cols, schema, plus any
        extra_metadata) — useful when a downstream consumer needs
        the metadata, but for the typical lookup-CSV workflow the
        flat CSV is sufficient on its own.  Defaults to False so
        the per-month run produces a clean single-file artifact.
    round_to : int or None, default 3
        Decimal places to round all float-typed columns
        (``w_*``, ``wape_*``, ``bias_*``) to before writing the JSON
        and CSV.  Set to ``None`` to disable rounding and persist full
        floating-point precision.  The Spark/pandas DataFrame the
        caller passed in is NOT mutated — rounding is applied to a
        local copy only.

        Note on the simplex constraint: rounding two weights
        independently to 3 dp can in rare cases cause their sum to
        drift from exactly 1.000 by up to 0.001 (e.g.
        ``[0.6667, 0.3333]`` rounds to ``[0.667, 0.333]`` which sums
        to 1.000, but ``[0.6665, 0.3335]`` could round to
        ``[0.667, 0.334]``).  This is below downstream consumer
        tolerance and the audit log makes the original full-precision
        values traceable via the JSON if needed.

    Returns
    -------
    str
        Absolute path of the CSV file.  When ``also_write_json=True``
        a sibling JSON path (same basename, ``.json`` extension) is
        also written and printed to stdout for discoverability.

    Examples
    --------
    >>> weights = fit_forecast_blend(df, group_cols=['category'],
    ...                              forecast_cols=['Lego', 'DIQ'],
    ...                              actuals_col='Actuals')
    >>> save_weights(weights, '/Volumes/.../forecast_blend_weights',
    ...              snapshot_week='202615',
    ...              bias_mode='dominant',
    ...              extra_metadata={'input_path': '/Volumes/.../joined'})
    '/Volumes/.../forecast_blend_weights/202615/weights.json'
    """
    import json
    import os
    from datetime import datetime

    # 0. Validate required snapshot_week
    if snapshot_week is None or str(snapshot_week).strip() == "":
        raise ValueError(
            "snapshot_week is required.  Pass the year_week / period "
            "label this manifest applies to (e.g. snapshot_week='202615' "
            "or '2026-15').  We deliberately do not derive it from the "
            "system clock because run date and forecast snapshot can "
            "legitimately differ (e.g. backfills)."
        )
    snapshot_week = str(snapshot_week)

    # 1. Normalise to pandas (function works for both Spark and pandas inputs)
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        is_spark = isinstance(weights_df, SparkDataFrame)
    except ImportError:
        is_spark = False
    weights_pd = weights_df.toPandas() if is_spark else weights_df

    # 1b. Round all float columns for cleanliness in the saved artifacts.
    # Operate on a copy so the caller's DataFrame is never mutated — full
    # precision stays accessible if they want to inspect the in-memory
    # frame programmatically.
    if round_to is not None:
        weights_pd = weights_pd.copy()
        float_cols = weights_pd.select_dtypes(include=["float64", "float32"]).columns
        if len(float_cols) > 0:
            weights_pd[float_cols] = weights_pd[float_cols].round(int(round_to))

    # 1c. Capture schema metadata from the ORIGINAL pre-rename columns
    # so `forecast_cols` keeps its semantic names (e.g. ['Lego', 'DIQ'])
    # even when downstream renames hide them behind labels like
    # `diq_pcnt`.  This must happen BEFORE applying the rename map —
    # otherwise the inference helper, which looks for `w_*` columns,
    # would miss the renamed DIQ weight and report the wrong list.
    group_cols, forecast_cols = _infer_metadata_from_schema(weights_pd.columns)

    # 1d. Apply default output column renames.  In particular `w_DIQ`
    # becomes `diq_pcnt` so the saved file matches the legacy
    # `diq_pcnt` apply pattern downstream.  Only renames columns that
    # actually exist; safe no-op for non-DIQ forecast names.  The
    # rename map is also recorded in the manifest header so downstream
    # consumers can recover the mapping if they need to translate
    # semantic names ('DIQ') back to physical column names ('diq_pcnt').
    rename_map = {
        old: new
        for old, new in _DEFAULT_OUTPUT_COLUMN_RENAMES.items()
        if old in weights_pd.columns
    }
    if rename_map:
        # `weights_pd` is already a copy from the rounding step above
        # (or a fresh slice from `.toPandas()` if rounding was off);
        # safe to rename in place.
        weights_pd = weights_pd.rename(columns=rename_map)

    # 2. Resolve output path.  CSV is the primary artifact (matches
    # the legacy lookup-table convention used by downstream apply
    # code).  Default filename embeds today's UTC date in YYYYMMDD
    # format: `diq_gtin_lookup_{YYYYMMDD}.csv` — caller can override
    # via the `filename` parameter for backfills.
    if filename is None:
        _today = datetime.utcnow().strftime("%Y%m%d")
        _filename = f"diq_gtin_lookup_{_today}.csv"
    else:
        _filename = filename
        if not _filename.lower().endswith(".csv"):
            raise ValueError(
                f"filename must end in '.csv'; got {_filename!r}"
            )

    if output_path.lower().endswith(".csv"):
        csv_file = output_path
    else:
        csv_file = os.path.join(output_path, _filename)

    out_dir = os.path.dirname(csv_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 3. Write CSV (primary artifact — flat lookup table).
    weights_pd.to_csv(csv_file, index=False)

    # 4. Optionally also write a JSON manifest sibling for callers
    # who need the metadata header (snapshot_week, bias_mode,
    # group_cols, forecast_cols, schema, extra_metadata, etc.).
    if also_write_json:
        manifest = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "snapshot_week":    snapshot_week,
            "bias_mode":        bias_mode,
            "group_cols":       group_cols,
            "forecast_cols":    forecast_cols,
            "n_groups":         int(len(weights_pd)),
            "schema":           list(weights_pd.columns),  # post-rename names
            "column_renames":   rename_map if rename_map else None,
        }
        # Merge user-supplied metadata, never overwriting reserved keys
        if extra_metadata:
            reserved = set(manifest.keys()) | {"weights"}
            for k, v in extra_metadata.items():
                if k in reserved:
                    continue
                manifest[k] = v
        manifest["weights"] = weights_pd.to_dict(orient="records")

        json_file = os.path.splitext(csv_file)[0] + ".json"
        with open(json_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"Also wrote JSON: {os.path.abspath(json_file)}")

    return os.path.abspath(csv_file)


def load_weights(weights_path: str):
    """Load a weights manifest written by ``save_weights``.

    Returns
    -------
    (manifest, weights_pd)
        manifest : dict — the full JSON header (group_cols, forecast_cols,
                  bias_mode, generated_at_utc, ...).
        weights_pd : pandas.DataFrame — one row per group, ready to
                  ``spark.createDataFrame(...)`` for a join-based apply.

    Examples
    --------
    >>> manifest, weights = load_weights('/Volumes/.../2026-05/weights.json')
    >>> spark_weights = spark.createDataFrame(weights)
    >>> blended = next_forecast.join(spark_weights, on=manifest['group_cols'])
    """
    import json
    import pandas as pd

    with open(weights_path) as f:
        manifest = json.load(f)
    weights_pd = pd.DataFrame(manifest["weights"])
    return manifest, weights_pd


def fit_forecast_blend(
    df: DataFrame,
    *,
    group_cols: Sequence[str],
    forecast_cols: Sequence[str],
    actuals_col: str,
    time_col: Optional[str] = None,
    bias_mode: str = "dominant",
    bias_penalty: float = 1.0,
    weights_sum_to_one: bool = True,
    weight_bounds: Optional[Tuple[float, float]] = None,
    min_obs: int = 3,
    fallback_weights: Optional[Sequence[float]] = None,
    nonzero_forecasts_only: bool = True,
    zero_tol: float = 1e-9,
    max_periods: Optional[int] = 5,
) -> DataFrame:
    """
    Estimate per-group blend weights for two forecasts that minimise
    WAPE against actuals, with configurable bias treatment.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Long-format frame with one row per (group, time) observation.
    group_cols : sequence of str
        Columns defining the level at which weights are estimated.
        Examples: ['category'], ['category', 'lego_segment'],
        ['category', 'lego_segment', 'cs_gtin']. Weights are estimated
        independently per group.
    forecast_cols : sequence of str (length 2)
        Column names of the two forecasts to blend, e.g. ['Lego', 'DIQ'].
        Output columns are derived from these names.
        IMPORTANT: list the "safe" / baseline forecast first. The
        default fallback_weights of (1.0, 0.0) means groups that cannot
        be optimised default to 100% of forecast_cols[0].
    actuals_col : str
        Column name of the ground-truth actuals.
    time_col : str, optional
        Time period column. Used only for the n_periods diagnostic.
    bias_mode : {'dominant', 'hard', 'soft', 'none'}, default 'dominant'
        - 'dominant' : minimise WAPE s.t. the absolute fractional bias
                       of the blend is no worse than the better of the
                       two components. Two free parameters with one
                       linear-slab constraint. The recommended default —
                       cleanest "blend dominates each component on both
                       error and bias" guarantee, and almost always
                       feasible inside weight_bounds.
        - 'hard'     : enforce zero in-sample bias as an equality
                       constraint. One free parameter; bias = 0 by
                       construction whenever feasible.
        - 'soft'     : minimise WAPE + bias_penalty * |fractional_bias|.
                       Trades off WAPE and bias smoothly.
        - 'none'     : pure WAPE minimisation, no bias control.
    bias_penalty : float, default 1.0
        Scale of the bias term in 'soft' mode.
    weights_sum_to_one : bool, default True
        Constrain `w1 + w2 == 1` (proper convex blend / simplex).
        - True  : the default — weights represent a proportionality
                  between the two forecasts.  Every bias mode is
                  reformulated as a 1D optimisation on `w1` (with
                  `w2 = 1 - w1`), bias becomes a linear function of
                  `w1`, and the search is done with `minimize_scalar`
                  on a bounded interval.  Faster and more stable than
                  the unconstrained 2D version.  Default
                  `weight_bounds` becomes `(0.0, 1.0)` to keep `w2 >= 0`.
        - False : the legacy unconstrained 2D version.  `w1`, `w2`
                  are independent box-constrained variables that may
                  sum to more or less than 1 (allows extrapolation
                  beyond either forecast).  Default `weight_bounds`
                  is `(0.0, 3.0)`.
    weight_bounds : (float, float), optional
        Box constraints on each weight.  When `None`, defaults to
        `(0.0, 1.0)` if `weights_sum_to_one=True`, else `(0.0, 3.0)`.
        When `weights_sum_to_one=True`, values are silently clamped
        into `[0, 1]` to keep both weights non-negative.
    min_obs : int, default 3
        Groups with fewer rows fall back to fallback_weights.
        Note: this check applies AFTER the nonzero-forecast filter
        (see `nonzero_forecasts_only`), so a group with plenty of raw
        rows but few rows where both forecasts are non-zero may still
        fall back.
    fallback_weights : sequence of float (length 2), optional
        Used when a group cannot be optimised. Defaults to (1.0, 0.0):
        100% of the first forecast.
    nonzero_forecasts_only : bool, default True
        When True, restrict the fit to rows where BOTH forecasts are
        non-zero (using `zero_tol` as the threshold).  This is the
        right default when one of the forecasts emits an explicit
        zero for "no signal" cases (e.g. dead-key forecasts in the
        DIQ pipeline, or out-of-stock placeholders): including those
        zeros in the optimisation would skew the weights toward the
        forecast that has fewer zero rows, since the zero rows give
        the OTHER forecast a free WAPE-improvement when blended.
        All metrics (`wape_<f1>`, `wape_<f2>`, `wape_blend`,
        `bias_<f1>`, `bias_<f2>`, `bias_blend`) are then computed
        on the same filtered subset for consistency, and `n_obs`
        reports the post-filter row count actually used in the fit.
        Set to False to use every row regardless of forecast value.
    zero_tol : float, default 1e-9
        Magnitude below which a forecast value is treated as zero.
        Only used when `nonzero_forecasts_only=True`.
    max_periods : int or None, default 5
        Restrict the fit to the FIRST N distinct values of `time_col`
        per group, sorted ascending — typically the 5 nearest-horizon
        weeks where forecast accuracy matters most for short-term
        planning, and where Lego/DIQ both produce reliable point
        forecasts.  Pass ``None`` to use every period.  Requires
        `time_col` to be set; ignored otherwise.

        Filter ordering: `max_periods` runs FIRST, then
        `nonzero_forecasts_only`.  So with the defaults
        (``max_periods=5``, ``nonzero_forecasts_only=True``) the
        optimiser sees rows in the first 5 weeks where both
        forecasts are non-zero.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per group, with columns:
            <group_cols>,
            w_<f1>, w_<f2>,
            wape_<f1>, wape_<f2>, wape_blend,
            bias_<f1>, bias_<f2>, bias_blend,
            n_obs, [n_periods,] status

        Bias is reported as fractional bias:
            (sum(forecast) - sum(actual)) / sum(actual)

        `status` flags how each group was solved:
            'ok:dominant'              — solved in dominant mode
            'ok:hard'                  — solved in hard mode
            'ok:soft'                  — solved in soft mode
            'ok:none'                  — solved unconstrained
            'hard:dominant_degenerate' — dominant requested, M≈0,
                                         routed to hard
            'soft:hard_infeasible'     — hard requested, infeasible in
                                         box, routed to soft
            'soft:dominant_failed'     — dominant SLSQP failed, soft
                                         used as fallback
            'none:hard_degenerate'     — hard requested, degenerate sums
            'none:dominant_degenerate' — dominant requested, degenerate
                                         sums, routed to unconstrained
            'fallback:n_obs<{N}'       — too few rows, used fallback
            'fallback:zero_actuals'    — all actuals zero, used fallback
            'fallback:error:<Type>'    — exception raised, used fallback
    """
    # ----- input validation -----
    group_cols = list(group_cols)
    forecast_cols = list(forecast_cols)
    if not group_cols:
        raise ValueError("group_cols must contain at least one column")
    if len(forecast_cols) != 2:
        raise ValueError(
            f"forecast_cols must contain exactly 2 columns; got {forecast_cols}"
        )
    if bias_mode not in {"dominant", "hard", "soft", "none"}:
        raise ValueError(
            f"bias_mode must be 'dominant', 'hard', 'soft', or 'none'; "
            f"got {bias_mode!r}"
        )

    needed = set(group_cols) | set(forecast_cols) | {actuals_col}
    if time_col:
        needed.add(time_col)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in df: {sorted(missing)}")

    # Resolve weight_bounds default based on simplex flag.  When
    # weights_sum_to_one=True, w2 = 1 - w1, so w1 ∈ [0, 1] keeps both
    # weights non-negative; we clamp into that interval if the user
    # passed something wider (eg the pre-simplex default of (0, 3)).
    if weight_bounds is None:
        weight_bounds = (0.0, 1.0) if weights_sum_to_one else (0.0, 3.0)
    w_lo, w_hi = float(weight_bounds[0]), float(weight_bounds[1])
    if w_lo > w_hi:
        raise ValueError("weight_bounds must satisfy lo <= hi")
    if weights_sum_to_one:
        # Clamp to the simplex interior to ensure w1 ∈ [0, 1].
        w_lo = max(0.0, w_lo)
        w_hi = min(1.0, w_hi)
        if w_lo > w_hi:
            raise ValueError(
                "weight_bounds clamps to an empty interval under the "
                "simplex constraint; pass bounds inside [0, 1]."
            )
    if fallback_weights is None:
        fb1, fb2 = 1.0, 0.0
    else:
        if len(fallback_weights) != 2:
            raise ValueError("fallback_weights must have length 2")
        fb1, fb2 = float(fallback_weights[0]), float(fallback_weights[1])

    f1_col, f2_col = forecast_cols

    # ----- output schema -----
    group_fields = [StructField(g, df.schema[g].dataType) for g in group_cols]
    metric_fields = [
        StructField(f"w_{f1_col}",    DoubleType()),
        StructField(f"w_{f2_col}",    DoubleType()),
        StructField(f"wape_{f1_col}", DoubleType()),
        StructField(f"wape_{f2_col}", DoubleType()),
        StructField("wape_blend",     DoubleType()),
        StructField(f"bias_{f1_col}", DoubleType()),
        StructField(f"bias_{f2_col}", DoubleType()),
        StructField("bias_blend",     DoubleType()),
        StructField("n_obs",          IntegerType()),
    ]
    if time_col:
        metric_fields.append(StructField("n_periods", IntegerType()))
    metric_fields.append(StructField("status", StringType()))
    out_schema = StructType(group_fields + metric_fields)

    # ----- per-group fit -----
    def _fit(pdf: pd.DataFrame) -> pd.DataFrame:
        n_obs_raw = len(pdf)

        # Step 1 — restrict to the first `max_periods` distinct time
        # values per group, sorted ascending.  Done BEFORE the
        # nonzero filter so the "first 5 weeks" semantic refers to
        # the original time axis, not the post-zero-filter axis.
        # Skipped when max_periods is None or time_col isn't set.
        if max_periods is not None and time_col and n_obs_raw > 0:
            sorted_times = sorted(pdf[time_col].dropna().unique())
            if len(sorted_times) > max_periods:
                keep_times = set(sorted_times[:max_periods])
                pdf = pdf.loc[pdf[time_col].isin(keep_times)]

        y_full  = pdf[actuals_col].astype(float).to_numpy()
        f1_full = pdf[f1_col].astype(float).to_numpy()
        f2_full = pdf[f2_col].astype(float).to_numpy()
        n_after_period_filter = len(y_full)

        # Step 2 — filter to rows where BOTH forecasts are non-zero
        # (so a row with `f1=0 or f2=0` can't bias the optimisation
        # by giving the other forecast a free WAPE-improvement).
        # Applied to the actuals + both forecast arrays AND to pdf
        # (for the n_periods diagnostic) so every metric is
        # computed on the same subset the optimiser saw.
        if nonzero_forecasts_only and n_after_period_filter > 0:
            mask = (np.abs(f1_full) > zero_tol) & (np.abs(f2_full) > zero_tol)
            y, f1, f2 = y_full[mask], f1_full[mask], f2_full[mask]
            pdf_used = pdf.loc[mask]
        else:
            y, f1, f2, pdf_used = y_full, f1_full, f2_full, pdf

        n_obs = len(y)
        n_periods = int(pdf_used[time_col].nunique()) if time_col else None
        S_y, S_1, S_2 = float(y.sum()), float(f1.sum()), float(f2.sum())
        abs_y = float(np.abs(y).sum())

        wape = lambda f: float(np.abs(y - f).sum() / abs_y) if abs_y else float("nan")
        bias = lambda f: float((f.sum() - S_y) / S_y)        if S_y   else float("nan")

        w1, w2, status = fb1, fb2, "fallback"

        if n_obs < min_obs:
            # Distinguish "too few rows in source" from "filters
            # dropped most rows" — the latter is a real signal that
            # this group's forecasts overlap too rarely in the chosen
            # period window for a meaningful blend.
            filters_active = (
                (max_periods is not None and time_col is not None)
                or nonzero_forecasts_only
            )
            if filters_active and n_obs_raw >= min_obs:
                status = f"fallback:n_obs<{min_obs}_after_filters"
            else:
                status = f"fallback:n_obs<{min_obs}"
        elif abs_y == 0:
            status = "fallback:zero_actuals"
        else:
            try:
                w1, w2, status = _dispatch(
                    bias_mode=bias_mode,
                    y=y, f1=f1, f2=f2,
                    S_y=S_y, S_1=S_1, S_2=S_2, abs_y=abs_y,
                    w_lo=w_lo, w_hi=w_hi,
                    bias_penalty=bias_penalty,
                    fb1=fb1, fb2=fb2,
                    weights_sum_to_one=weights_sum_to_one,
                )
            except Exception as e:
                w1, w2 = fb1, fb2
                status = f"fallback:error:{type(e).__name__}"

        f_blend = w1 * f1 + w2 * f2
        row = {g: pdf[g].iloc[0] for g in group_cols}
        row[f"w_{f1_col}"]    = float(w1)
        row[f"w_{f2_col}"]    = float(w2)
        row[f"wape_{f1_col}"] = wape(f1)
        row[f"wape_{f2_col}"] = wape(f2)
        row["wape_blend"]     = wape(f_blend)
        row[f"bias_{f1_col}"] = bias(f1)
        row[f"bias_{f2_col}"] = bias(f2)
        row["bias_blend"]     = bias(f_blend)
        row["n_obs"]          = int(n_obs)
        if time_col:
            row["n_periods"]  = n_periods
        row["status"]         = status
        return pd.DataFrame([row])

    return df.groupBy(*group_cols).applyInPandas(_fit, schema=out_schema)


# ---------- mode dispatch ----------

def _dispatch(*, bias_mode, y, f1, f2, S_y, S_1, S_2, abs_y,
              w_lo, w_hi, bias_penalty, fb1, fb2,
              weights_sum_to_one):
    """Route a group to the right solver, with cross-mode fallbacks.

    When `weights_sum_to_one=True` we use the 1D-parameterised simplex
    solvers (`_solve_simplex_*`); otherwise the legacy 2D solvers.
    Both branches preserve identical bias-mode semantics and status
    strings — the simplex variants just append ":simplex" to make it
    clear from the audit log which branch ran.
    """

    if weights_sum_to_one:
        return _dispatch_simplex(
            bias_mode=bias_mode,
            y=y, f1=f1, f2=f2,
            S_y=S_y, S_1=S_1, S_2=S_2, abs_y=abs_y,
            w_lo=w_lo, w_hi=w_hi,
            bias_penalty=bias_penalty,
        )

    # ── legacy unconstrained 2D path ───────────────────────────────
    if bias_mode == "dominant":
        if S_y == 0 or S_1 == 0 or S_2 == 0:
            w1, w2 = _solve_unconstrained(y, f1, f2, w_lo, w_hi, abs_y, fb1, fb2)
            return w1, w2, "none:dominant_degenerate"

        bias_a = (S_1 - S_y) / S_y
        bias_b = (S_2 - S_y) / S_y
        M = min(abs(bias_a), abs(bias_b))

        if M < _M_TOL:
            # both components effectively unbiased → use hard solver (faster, exact)
            w1, w2, infeasible = _solve_hard(y, f1, f2, S_y, S_1, S_2, w_lo, w_hi)
            if infeasible:
                w1, w2 = _solve_soft(y, f1, f2, w_lo, w_hi, bias_penalty, S_y, abs_y, fb1, fb2)
                return w1, w2, "soft:hard_infeasible"
            return w1, w2, "hard:dominant_degenerate"

        try:
            w1, w2 = _solve_dominant(
                y, f1, f2, S_y, S_1, S_2, abs_y, M, bias_a, bias_b, w_lo, w_hi
            )
            return w1, w2, "ok:dominant"
        except Exception:
            w1, w2 = _solve_soft(y, f1, f2, w_lo, w_hi, bias_penalty, S_y, abs_y, fb1, fb2)
            return w1, w2, "soft:dominant_failed"

    if bias_mode == "hard":
        if S_1 == 0 or S_2 == 0 or S_y == 0:
            w1, w2 = _solve_unconstrained(y, f1, f2, w_lo, w_hi, abs_y, fb1, fb2)
            return w1, w2, "none:hard_degenerate"
        w1, w2, infeasible = _solve_hard(y, f1, f2, S_y, S_1, S_2, w_lo, w_hi)
        if infeasible:
            w1, w2 = _solve_soft(y, f1, f2, w_lo, w_hi, bias_penalty, S_y, abs_y, fb1, fb2)
            return w1, w2, "soft:hard_infeasible"
        return w1, w2, "ok:hard"

    if bias_mode == "soft":
        w1, w2 = _solve_soft(y, f1, f2, w_lo, w_hi, bias_penalty, S_y, abs_y, fb1, fb2)
        return w1, w2, "ok:soft"

    # 'none'
    w1, w2 = _solve_unconstrained(y, f1, f2, w_lo, w_hi, abs_y, fb1, fb2)
    return w1, w2, "ok:none"


def _dispatch_simplex(*, bias_mode, y, f1, f2, S_y, S_1, S_2, abs_y,
                      w_lo, w_hi, bias_penalty):
    """1D simplex routing (w2 = 1 - w1, w1 ∈ [w_lo, w_hi]).

    Same bias-mode semantics as the 2D dispatcher, but every solver is
    a 1D bounded scalar minimisation — much simpler and faster, and
    guarantees w1 + w2 == 1 by construction.
    """

    if bias_mode == "dominant":
        if S_y == 0:
            w1, w2 = _solve_simplex_unconstrained(y, f1, f2, abs_y, w_lo, w_hi)
            return w1, w2, "none:dominant_degenerate:simplex"

        bias_a = (S_1 - S_y) / S_y
        bias_b = (S_2 - S_y) / S_y
        M = min(abs(bias_a), abs(bias_b))

        if M < _M_TOL:
            w1, w2, infeasible = _solve_simplex_hard(S_y, S_1, S_2, w_lo, w_hi, y, f1, f2, abs_y)
            if infeasible:
                w1, w2 = _solve_simplex_soft(y, f1, f2, S_y, abs_y, w_lo, w_hi, bias_penalty)
                return w1, w2, "soft:hard_infeasible:simplex"
            return w1, w2, "hard:dominant_degenerate:simplex"

        try:
            w1, w2 = _solve_simplex_dominant(y, f1, f2, S_y, S_1, S_2, abs_y, M, w_lo, w_hi)
            return w1, w2, "ok:dominant:simplex"
        except Exception:
            w1, w2 = _solve_simplex_soft(y, f1, f2, S_y, abs_y, w_lo, w_hi, bias_penalty)
            return w1, w2, "soft:dominant_failed:simplex"

    if bias_mode == "hard":
        if S_y == 0:
            w1, w2 = _solve_simplex_unconstrained(y, f1, f2, abs_y, w_lo, w_hi)
            return w1, w2, "none:hard_degenerate:simplex"
        w1, w2, infeasible = _solve_simplex_hard(S_y, S_1, S_2, w_lo, w_hi, y, f1, f2, abs_y)
        if infeasible:
            w1, w2 = _solve_simplex_soft(y, f1, f2, S_y, abs_y, w_lo, w_hi, bias_penalty)
            return w1, w2, "soft:hard_infeasible:simplex"
        return w1, w2, "ok:hard:simplex"

    if bias_mode == "soft":
        w1, w2 = _solve_simplex_soft(y, f1, f2, S_y, abs_y, w_lo, w_hi, bias_penalty)
        return w1, w2, "ok:soft:simplex"

    # 'none'
    w1, w2 = _solve_simplex_unconstrained(y, f1, f2, abs_y, w_lo, w_hi)
    return w1, w2, "ok:none:simplex"


# ---------- solvers ----------

def _solve_dominant(y, f1, f2, S_y, S_1, S_2, abs_y, M, bias_a, bias_b, w_lo, w_hi):
    """Minimise WAPE s.t. |bias_blend| <= M (slab constraint, SLSQP)."""

    def loss(w):
        return float(np.abs(y - (w[0] * f1 + w[1] * f2)).sum() / abs_y)

    def bias_blend(w):
        return (w[0] * S_1 + w[1] * S_2 - S_y) / S_y

    constraints = [
        {"type": "ineq", "fun": lambda w: M - bias_blend(w)},  # blend bias <=  M
        {"type": "ineq", "fun": lambda w: M + bias_blend(w)},  # blend bias >= -M
    ]

    # Warm-start from the better-biased component (always feasible)
    if abs(bias_a) <= abs(bias_b):
        x0 = np.array([1.0, 0.0])
    else:
        x0 = np.array([0.0, 1.0])

    res = minimize(
        loss, x0=x0,
        bounds=[(w_lo, w_hi), (w_lo, w_hi)],
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 200},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP did not converge: {res.message}")
    return float(res.x[0]), float(res.x[1])


def _solve_hard(y, f1, f2, S_y, S_1, S_2, w_lo, w_hi):
    """
    Minimise WAPE s.t. bias_blend == 0 (equality, 1D parameterisation).
    Returns (w1, w2, infeasible_flag).
    """
    if S_1 > 0:
        lo = max(w_lo, (S_y - w_hi * S_2) / S_1)
        hi = min(w_hi, (S_y - w_lo * S_2) / S_1)
    else:  # S_1 < 0; flip
        lo = max(w_lo, (S_y - w_lo * S_2) / S_1)
        hi = min(w_hi, (S_y - w_hi * S_2) / S_1)

    if lo >= hi:
        return 1.0, 0.0, True

    def loss_1d(w):
        wb = (S_y - w * S_1) / S_2
        return float(np.abs(y - (w * f1 + wb * f2)).sum())

    res = minimize_scalar(loss_1d, bounds=(lo, hi), method="bounded")
    w1 = float(res.x)
    w2 = (S_y - w1 * S_1) / S_2
    return w1, w2, False


def _solve_soft(y, f1, f2, w_lo, w_hi, lam, S_y, abs_y, fb1, fb2):
    """Soft-penalty: minimise WAPE + lam * |fractional_bias|."""
    def loss(w):
        f = w[0] * f1 + w[1] * f2
        wape_v = np.abs(y - f).sum() / abs_y if abs_y else 0.0
        bias_v = (f.sum() - S_y) / S_y       if S_y   else 0.0
        return wape_v + lam * abs(bias_v)

    res = minimize(
        loss,
        x0=np.array([fb1, fb2]),
        bounds=[(w_lo, w_hi), (w_lo, w_hi)],
        method="L-BFGS-B",
    )
    return float(res.x[0]), float(res.x[1])


def _solve_unconstrained(y, f1, f2, w_lo, w_hi, abs_y, fb1, fb2):
    """Pure WAPE minimisation, no bias control."""
    def loss(w):
        f = w[0] * f1 + w[1] * f2
        return np.abs(y - f).sum() / abs_y if abs_y else 0.0

    res = minimize(
        loss,
        x0=np.array([fb1, fb2]),
        bounds=[(w_lo, w_hi), (w_lo, w_hi)],
        method="L-BFGS-B",
    )
    return float(res.x[0]), float(res.x[1])


# ---------- simplex solvers (w2 = 1 - w1, weights_sum_to_one=True) ----------

# Under the simplex constraint each solver becomes a 1D bounded scalar
# minimisation in `w1`, with `w2 = 1 - w1` derived implicitly.  Bias as
# a function of w1 is linear:
#
#     bias(w1) = (w1 * (S_1 - S_2) + S_2 - S_y) / S_y
#
# so the |bias(w1)| ≤ M slab constraint in dominant mode reduces to an
# interval intersection on w1, and the bias=0 hard constraint reduces
# to a closed-form unique point.  This is materially simpler than the
# 2D versions and converges to the global optimum without warm-starts.


def _simplex_bias_interval(S_y, S_1, S_2, M, w_lo, w_hi):
    """Solve |bias(w1)| ≤ M for w1, intersect with [w_lo, w_hi].

    Returns (lo, hi) — the interval where the bias constraint holds
    inside the box.  Caller checks lo >= hi to detect infeasibility.
    """
    if S_1 == S_2:
        # bias(w1) is constant in w1; either always in slab or never
        bias_const = (S_2 - S_y) / S_y
        if abs(bias_const) > M:
            return w_hi, w_lo  # infeasible (lo > hi)
        return w_lo, w_hi

    # |a*w1 + b| ≤ M, where a = (S_1 - S_2)/S_y, b = (S_2 - S_y)/S_y.
    # Solving:  -M - b ≤ a*w1 ≤ M - b
    a = (S_1 - S_2) / S_y
    b = (S_2 - S_y) / S_y
    if a > 0:
        lo_b = (-M - b) / a
        hi_b = ( M - b) / a
    else:
        lo_b = ( M - b) / a
        hi_b = (-M - b) / a

    return max(w_lo, lo_b), min(w_hi, hi_b)


def _solve_simplex_dominant(y, f1, f2, S_y, S_1, S_2, abs_y, M, w_lo, w_hi):
    """Simplex + dominant: minimise WAPE on w1 ∈ [w_lo, w_hi] s.t. |bias(w1)| ≤ M."""
    lo, hi = _simplex_bias_interval(S_y, S_1, S_2, M, w_lo, w_hi)
    if lo >= hi:
        # Not feasible inside the box — caller will fall back to soft
        raise RuntimeError("simplex dominant: bias slab empty inside box")

    def loss(w1):
        return float(np.abs(y - (w1 * f1 + (1 - w1) * f2)).sum() / abs_y)

    res = minimize_scalar(loss, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-9})
    w1 = float(res.x)
    return w1, 1.0 - w1


def _solve_simplex_hard(S_y, S_1, S_2, w_lo, w_hi, y, f1, f2, abs_y):
    """Simplex + hard (bias == 0): closed-form, unique solution if feasible.

    With w2 = 1 - w1, bias=0 means
        w1 * S_1 + (1 - w1) * S_2 = S_y
        w1 = (S_y - S_2) / (S_1 - S_2).
    If this lies inside [w_lo, w_hi], it's the answer (no minimisation
    needed — it's the only point in the simplex with bias=0).  If
    S_1 == S_2, bias is constant and we either accept any w1 (and
    minimise WAPE on the box) or report infeasible.

    Returns (w1, w2, infeasible_flag).
    """
    if S_1 == S_2:
        if abs(S_2 - S_y) < _M_TOL * max(abs(S_y), 1.0):
            # Whole interval is bias=0; pick the WAPE-minimising w1
            w1, w2 = _solve_simplex_unconstrained(y, f1, f2, abs_y, w_lo, w_hi)
            return w1, w2, False
        return 1.0, 0.0, True

    w1 = (S_y - S_2) / (S_1 - S_2)
    if w1 < w_lo - 1e-12 or w1 > w_hi + 1e-12:
        return 1.0, 0.0, True
    w1 = float(min(max(w1, w_lo), w_hi))
    return w1, 1.0 - w1, False


def _solve_simplex_soft(y, f1, f2, S_y, abs_y, w_lo, w_hi, lam):
    """Simplex + soft: minimise WAPE + lam * |fractional_bias|."""
    def loss(w1):
        f = w1 * f1 + (1 - w1) * f2
        wape_v = np.abs(y - f).sum() / abs_y if abs_y else 0.0
        bias_v = (f.sum() - S_y) / S_y       if S_y   else 0.0
        return wape_v + lam * abs(bias_v)

    res = minimize_scalar(loss, bounds=(w_lo, w_hi), method="bounded",
                          options={"xatol": 1e-9})
    w1 = float(res.x)
    return w1, 1.0 - w1


def _solve_simplex_unconstrained(y, f1, f2, abs_y, w_lo, w_hi):
    """Simplex + none: pure WAPE minimisation on the simplex."""
    def loss(w1):
        return float(np.abs(y - (w1 * f1 + (1 - w1) * f2)).sum() / abs_y) \
            if abs_y else 0.0

    res = minimize_scalar(loss, bounds=(w_lo, w_hi), method="bounded",
                          options={"xatol": 1e-9})
    w1 = float(res.x)
    return w1, 1.0 - w1
