#!/usr/bin/env python3
"""
Databricks Entry Point for DIQ Forward-Forecast Runs
====================================================

**Standalone** runner for DIQ (Delivery Intelligence Quotient) — drives a
forward-forecast-only run across every requested UK category from a
single notebook or job, using the unified input table at
``{PARENT_DIR}/DIQ/INPUT_DATA_DELTA``.

This is intentionally separate from ``databricks_runner_uk.py`` (which
does per-category CLI-style runs with preprocessing + backtesting).
The DIQ flow does NO preprocessing (the upstream data is already
ready) and NO backtesting (only the forward forecast is needed).

Expected layout AFTER a run at snapshot_week=202615::

    {PARENT_DIR}/DIQ/
        INPUT_DATA_DELTA                     # caller-provided input table
        202615/                              # snapshot_week
            COOKING_AID/
                sourcedata/COOKING_AID.csv
                artifacts/model_artifacts/inference_forecast.csv
                ... (eda_output, seg_output, feature_output, model_artifacts)
            CONDIMENT/ ...
            (one folder per category in --category-list)
        TF_DIQ/
            202615/                          # snapshot_week (mirrors layout above)
                inference_forecast.parquet   # STACKED + MOQ post-processed

Three ways to invoke
--------------------

1. **From a Databricks notebook** (recommended) — see the companion
   ``databricks_diq_notebook.py`` template. The notebook pulls widget
   values and calls ``utils.diq_runner.run_diq_forecast`` directly.

2. **As a Databricks job / spark_python_task** passing flags::

       python databricks_runner_diq.py \\
           --parent-dir "/Volumes/.../uk_rerun_lego_features" \\
           --history-till 202614 \\
           --snapshot-week 202615 \\
           --category-list COOKING_AID,CONDIMENT,DEODORANT_FRAGRANCE,...

3. **Locally for smoke testing** — same flag shape, ``--skip-install``
   to skip the pip-install step.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# CRITICAL: disable OpenTelemetry / CrewAI telemetry BEFORE any imports
# that might pull them in. Same pattern as databricks_runner.py so we
# don't hit version-pin conflicts on Databricks runtime.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", "all")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency install (stripped-down — DIQ only needs pandas / pyarrow /
# pyspark / the checked-in utils, not the full CrewAI / litellm stack).
# We still offer a ``requirements.txt`` path for teams that want a
# single-source-of-truth install. Safe no-op when dependencies are
# already available.
# ---------------------------------------------------------------------------

def _install_dependencies(project_root: str, quiet: bool = True) -> bool:
    req = os.path.join(project_root, "requirements.txt")
    if not os.path.exists(req):
        logger.info("No requirements.txt found — skipping install.")
        return True
    logger.info("Installing Python dependencies from requirements.txt…")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "-r", req,
                "--upgrade", "--no-warn-script-location",
                *(["--quiet"] if quiet else []),
            ],
            timeout=1800, capture_output=True, text=True,
        )
        if result.returncode != 0 and result.stderr:
            logger.warning(f"pip stderr tail: {result.stderr[-800:]}")
        logger.info("Dependency install complete.")
        return True
    except Exception as e:
        logger.error(f"pip install failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DIQ forward-forecast runner for FEU-Agentic-Forecasting"
    )
    parser.add_argument(
        "--parent-dir", required=True,
        help="Root DBFS volume path — input at {parent_dir}/DIQ/INPUT_DATA_DELTA, "
             "outputs under {parent_dir}/DIQ/{snapshot_week}/{CATEGORY}/ and "
             "{parent_dir}/DIQ/TF_DIQ/.",
    )
    parser.add_argument(
        "--history-till", required=True,
        help="Last period of known actuals (YYYYWW or YYYYMM), e.g. '202614'. "
             "Becomes val_end for every category's runtime config.",
    )
    parser.add_argument(
        "--snapshot-week", required=True,
        help="First week to forecast (YYYYWW or YYYYMM), e.g. '202615'. Used as "
             "the folder name for this run's artifacts AND as test_start.",
    )
    parser.add_argument(
        "--category-list", required=True,
        help="Delimiter-separated list of category tokens (uppercase with "
             "underscores). Accepts ',' or '#' as separator. "
             "Example: 'COOKING_AID,CONDIMENT,DEODORANT_FRAGRANCE'",
    )
    parser.add_argument(
        "--input-subpath", default="DIQ/INPUT_DATA_DELTA",
        help="Relative path from --parent-dir to the unified input table "
             "(default: 'DIQ/INPUT_DATA_DELTA'). Delta or Parquet.",
    )
    parser.add_argument(
        "--category-column", default="category_name",
        help="Column name in the input table used to filter per category "
             "(default: 'category_name').",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Abort on the first category failure. Default behaviour is to "
             "log + continue so one broken category doesn't stop the rest.",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip the pip-install step. Use when the cluster image already "
             "has the dependencies.",
    )
    parser.add_argument(
        "--out-path", default="",
        help="Optional output path. When provided (non-blank), the parent "
             "directory of --out-path is used as the output dir and one "
             "Parquet per category is written there as "
             "{out_dir}/{CATEGORY}_inference_forecast.parquet. When blank, "
             "the default {parent_dir}/DIQ/TF_DIQ/{snapshot_week}/"
             "inference_forecast.parquet stacked layout is used.",
    )

    # --- UK stacked-ensemble engine (Pattern B - replaces the GBM pipeline for routed cats) ---
    parser.add_argument(
        "--uk-engine", action="store_true",
        help="Replace the per-category GBM pipeline with the UK stacked-ensemble engine "
             "(utils/uk_engine.py) for every category listed in --uk-route. Trains a pooled "
             "global LightGBM + per-category LightGBM ONCE per snapshot and blends with the "
             "LEGO prediction_rf column using frozen weights from --uk-weights-path. Default off.",
    )
    parser.add_argument(
        "--uk-route", default="",
        help="Comma/#-separated categories to route through the UK engine (widget tokens like "
             "'DEODORANT_FRAGRANCE' or human form 'DEODORANT & FRAGRANCE' both work).",
    )
    parser.add_argument(
        "--uk-weights-path", default="config/uk_ensemble_weights.yaml",
        help="Path to the frozen per-category blend weights YAML (relative to project_root).",
    )
    parser.add_argument(
        "--uk-lego-source", default="",
        help="Spark parquet directory holding prediction_rf for the snapshot. "
             "Typically {parent_dir}/TF_modelling_pristine. Required when --uk-engine is set.",
    )
    parser.add_argument(
        "--uk-num-threads", type=int, default=1,
        help="LightGBM threads per worker (kept at 1 for bit-identical determinism). "
             "Worker pool size is governed by the UK_ENGINE_WORKERS env var; set it on the cluster.",
    )

    args = parser.parse_args()

    # Normalise category list — accept both '#' (legacy widget shape) and ','
    sep = "#" if "#" in args.category_list else ","
    categories = [c.strip() for c in args.category_list.split(sep) if c.strip()]
    if not categories:
        logger.error("--category-list produced zero categories after split")
        sys.exit(2)

    logger.info("=" * 70)
    logger.info("FEU-Agentic-Forecasting: DIQ Runner")
    logger.info("=" * 70)
    logger.info(f"  parent_dir:    {args.parent_dir}")
    logger.info(f"  history_till:  {args.history_till}")
    logger.info(f"  snapshot_week: {args.snapshot_week}")
    logger.info(f"  categories:    {categories}")
    logger.info(f"  input_subpath: {args.input_subpath}")
    logger.info(f"  category_col:  {args.category_column}")

    # Project root — the dir this file lives in
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:  # notebook context without __file__
        project_root = os.getcwd()
    logger.info(f"  project_root:  {project_root}")

    if not args.skip_install:
        if not _install_dependencies(project_root):
            logger.error("Dependency install failed — aborting.")
            sys.exit(1)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Delegate to the orchestrator
    from utils.diq_runner import run_diq_forecast
    stacked = run_diq_forecast(
        parent_dir=args.parent_dir,
        history_till=args.history_till,
        snapshot_week=args.snapshot_week,
        category_list=categories,
        project_root=project_root,
        category_col=args.category_column,
        continue_on_category_failure=not args.fail_fast,
        out_path=args.out_path or None,
        # UK engine (Pattern B - replaces GBM for routed cats; default off keeps UK/DE byte-identical)
        enable_uk_engine=args.uk_engine,
        uk_route_categories=[c.strip() for c in args.uk_route.replace("#", ",").split(",") if c.strip()],
        uk_weights_path=args.uk_weights_path,
        uk_lego_source=args.uk_lego_source or None,
        uk_num_threads=args.uk_num_threads,
    )

    out_path_clean = (args.out_path or "").strip()
    output_loc = (
        str(Path(out_path_clean).parent) if out_path_clean
        else str(Path(args.parent_dir) / 'DIQ' / 'TF_DIQ' / args.snapshot_week)
    )
    logger.info("=" * 70)
    logger.info("DIQ Runner finished successfully")
    logger.info(f"  Stacked forecast rows: {len(stacked)}")
    logger.info(f"  Output location:       {output_loc}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
