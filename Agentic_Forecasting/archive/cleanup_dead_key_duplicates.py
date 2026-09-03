"""
Adhoc cleanup: remove duplicate (key, period) forecast rows where one
duplicate is zero and the other is non-zero.  The zero row(s) are kept;
the non-zero phantom forecast row(s) are dropped.

This is a one-shot cleanup utility for already-produced output files.
The pipeline itself (utils/inference.py:run_inference_pipeline) prevents
new duplicates from being written — see commit 24e243f.

Usage in a Databricks notebook (recommended):
    1. Open a new cell.
    2. Paste:
           %run /Workspace/.../Agentic_ForecastingGit/cleanup_dead_key_duplicates.py
       OR copy the function body inline.
    3. Edit the IN_PATH / KEY_COL / DATE_COL / PREDICTED_COL constants
       below (or call cleanup(...) directly with overrides).
    4. Run.

Usage as a CLI:
    python cleanup_dead_key_duplicates.py /path/to/inference_forecast.parquet
    python cleanup_dead_key_duplicates.py /path/to/inference_forecast.csv /optional/out/path.parquet

Output:
    A cleaned file written alongside the input with "_cleaned" suffix
    (or to OUT_PATH if specified).  Diagnostic prints: which keys had
    duplicates, how many rows were dropped.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd


# ─── CONFIGURE (edit to match your output) ───────────────────────────
# These are the defaults used when the script is run as a CLI without
# explicit args, OR when called from a notebook via cleanup() with no
# arguments.  Override either by editing these constants or by passing
# arguments to cleanup(...).
IN_PATH = "/Volumes/pds_feu_931272_dev/eu_uk/landing/UK/Experiments/DIQ_TEST/total_forecast_test/wDIQ/2026-11/DIQ/TF_DIQ/inference_forecast.parquet"
OUT_PATH: Optional[str] = None       # None → input + "_cleaned" suffix
KEY_COL = "key"
DATE_COL = "year_week"               # the period column name
PREDICTED_COL = "predicted"

# Also zero `prediction_post_moq` wherever `predicted == 0` for
# consistency (no business case for MOQ-rounding a zero up).  Set to
# False if your output doesn't have a `prediction_post_moq` column or
# you don't want that behaviour.
ALSO_CLEAN_POST_MOQ = True
POST_MOQ_COL = "prediction_post_moq"

# Treat tiny float-noise values as zero.  Set 0 to require exact
# equality with 0.  1e-9 catches floating-point residuals that are
# semantically zero but not bit-identical.
ZERO_TOL = 1e-9
# ─────────────────────────────────────────────────────────────────────


def cleanup(
    in_path: str = None,
    out_path: str = None,
    *,
    key_col: str = None,
    date_col: str = None,
    predicted_col: str = None,
    also_clean_post_moq: bool = None,
    post_moq_col: str = None,
    zero_tol: float = None,
) -> str:
    """Clean duplicate (key, period) forecast rows.

    For every (key, period) group with at least one zero forecast row,
    drop any non-zero rows.  The zero rows are kept as the canonical
    forecast for that key/period.

    Returns the path to the cleaned output file.
    """
    in_path = in_path or IN_PATH
    out_path = out_path or OUT_PATH
    key_col = key_col or KEY_COL
    date_col = date_col or DATE_COL
    predicted_col = predicted_col or PREDICTED_COL
    also_clean_post_moq = (
        ALSO_CLEAN_POST_MOQ if also_clean_post_moq is None else also_clean_post_moq
    )
    post_moq_col = post_moq_col or POST_MOQ_COL
    zero_tol = ZERO_TOL if zero_tol is None else zero_tol

    # ── 1. Load (parquet auto-detect, fallback to CSV) ────────────────
    if in_path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {in_path}")

    # Sanity-check required columns
    for required in (key_col, date_col, predicted_col):
        if required not in df.columns:
            raise ValueError(
                f"Required column '{required}' not found.  "
                f"Available columns: {list(df.columns)}"
            )

    # ── 2. Identify (key, period) groups with at least one zero row ───
    is_zero = df[predicted_col].abs() < zero_tol
    n_zeros_per_group = is_zero.groupby(
        [df[key_col], df[date_col]]
    ).transform("sum")

    # Total rows per group (for diagnostic)
    n_rows_per_group = df.assign(_one=1).groupby(
        [key_col, date_col]
    )["_one"].transform("sum")

    # Drop mask: row is in a group with at least one zero AND this row
    # is itself non-zero.
    drop_mask = (n_zeros_per_group > 0) & (~is_zero)
    n_drop = int(drop_mask.sum())

    # ── 3. Diagnostic ────────────────────────────────────────────────
    duplicate_groups = (
        df[n_rows_per_group > 1][[key_col, date_col]].drop_duplicates()
    )
    affected_groups = (
        df[drop_mask][[key_col, date_col]].drop_duplicates()
    )
    affected_keys = sorted(affected_groups[key_col].astype(str).unique())

    print()
    print("=== DIAGNOSTIC ===")
    print(f"  Total (key, period) groups        : {df.groupby([key_col, date_col]).ngroups:,}")
    print(f"  Groups with >1 row                : {len(duplicate_groups):,}")
    print(f"  Groups with zero+nonzero conflict : {len(affected_groups):,}")
    print(f"  Distinct affected keys            : {len(affected_keys):,}")
    print(f"  Non-zero rows to drop             : {n_drop:,}")

    if affected_keys:
        print()
        print("  Sample affected keys (and total rows for each in input):")
        for k in affected_keys[:20]:
            n_rows_for_key = int((df[key_col].astype(str) == k).sum())
            print(f"    {k:<60s}  ({n_rows_for_key} rows)")
        if len(affected_keys) > 20:
            print(f"    ... and {len(affected_keys) - 20} more")

    if n_drop == 0:
        print()
        print("Nothing to clean.  Output identical to input — not writing a new file.")
        return in_path

    # ── 4. Apply cleanup ──────────────────────────────────────────────
    cleaned = df[~drop_mask].reset_index(drop=True)

    # Optional: align prediction_post_moq to predicted (zero ⇒ zero)
    if also_clean_post_moq and post_moq_col in cleaned.columns:
        zero_pred_mask = cleaned[predicted_col].abs() < zero_tol
        n_post_moq_fixed = int(
            (cleaned.loc[zero_pred_mask, post_moq_col] != 0).sum()
        )
        cleaned.loc[zero_pred_mask, post_moq_col] = 0.0
        if n_post_moq_fixed:
            print()
            print(
                f"  Also zeroed {n_post_moq_fixed:,} '{post_moq_col}' values "
                f"where predicted==0 (consistency)."
            )

    # ── 5. Save ───────────────────────────────────────────────────────
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_cleaned{ext}"

    if out_path.endswith((".parquet", ".pq")):
        cleaned.to_parquet(out_path, index=False)
    else:
        cleaned.to_csv(out_path, index=False)

    print()
    print("=== SAVED ===")
    print(f"  Before: {len(df):,} rows")
    print(f"  After : {len(cleaned):,} rows  (dropped {len(df) - len(cleaned):,})")
    print(f"  Path  : {out_path}")
    return out_path


def cleanup_glob(pattern: str, **kwargs) -> list:
    """Convenience: run cleanup() over every file matching a glob.

    Example:
        cleanup_glob("/Volumes/.../TF_DIQ/2026-11/*_inference_forecast.parquet")
    """
    import glob
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files matched pattern: {pattern}")
        return []
    print(f"Cleaning {len(paths)} file(s) matching {pattern}\n")
    out_paths = []
    for p in paths:
        print(f"\n{'='*70}")
        print(f"FILE: {p}")
        print('=' * 70)
        out_paths.append(cleanup(p, **kwargs))
    return out_paths


if __name__ == "__main__":
    arg_in = sys.argv[1] if len(sys.argv) > 1 else IN_PATH
    arg_out = sys.argv[2] if len(sys.argv) > 2 else OUT_PATH
    cleanup(arg_in, arg_out)
