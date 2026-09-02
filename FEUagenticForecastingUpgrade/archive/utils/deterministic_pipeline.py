"""Deterministic (no-LLM) end-to-end pipeline.

This module exposes :func:`run_full_deterministic_pipeline`, a single entry
point that reproduces the full forecasting pipeline — EDA → calendar
synthesis → feature availability → segmentation → feature engineering →
training → inference → backtesting — without invoking any LLM or CrewAI
agents. It calls the same low-level utility functions the crews call.

It is shared between ``databricks_runner.py`` and ``databricks_runner_uk.py``
so both runners get identical deterministic behaviour.

Usage
-----

>>> from config.schema import load_config_from_yaml
>>> from utils.deterministic_pipeline import run_full_deterministic_pipeline
>>> cfg = load_config_from_yaml("config/config_uk_condiment_databricks.yaml")
>>> run_full_deterministic_pipeline(cfg, clean_artifacts=True)

Author: FEU-Agentic-Forecasting Demand Forecasting System
"""

from __future__ import annotations

# CRITICAL: kill EVERY auto-instrumentation hook at module-import time
# before any downstream module (statsmodels, sklearn, lightgbm, ...) gets a
# chance to trigger Databricks's auto-logging hook OR CrewAI's
# OpenTelemetry tracer OR LiteLLM/Posthog/HuggingFace telemetry.
#
# This mirrors the env-var bootstrap at the top of
# databricks_runner*.py / utils/diq_runner.py so any caller that imports
# this module first still gets the same slow-path mitigation.
import os as _early_os
if _early_os.environ.get("DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG", "").lower() not in (
    "1", "true", "yes",
):
    _env_defaults = {
        "MLFLOW_AUTOLOGGING_DISABLE":          "1",
        "MLFLOW_DATABRICKS_AUTOLOG_DISABLED":  "true",
        "DISABLE_MLFLOW_INTEGRATION":          "TRUE",
        "OTEL_SDK_DISABLED":                   "true",
        "OTEL_TRACES_EXPORTER":                "none",
        "OTEL_METRICS_EXPORTER":               "none",
        "OTEL_LOGS_EXPORTER":                  "none",
        "CREWAI_TELEMETRY_OPT_OUT":            "true",
        "CREWAI_DISABLE_TELEMETRY":            "true",
        "CREWAI_DO_NOT_TRACK":                 "1",
        "LITELLM_TELEMETRY":                   "False",
        "TOKENIZERS_PARALLELISM":              "false",
        "POSTHOG_DISABLED":                    "true",
    }
    for _k, _v in _env_defaults.items():
        _early_os.environ.setdefault(_k, _v)
    del _env_defaults

    try:
        import mlflow as _mlflow
        try:
            _mlflow.autolog(disable=True, silent=True)
        except Exception:
            pass
        for _flavor in ("statsmodels", "sklearn", "lightgbm", "xgboost",
                         "pytorch", "catboost", "tensorflow", "spark",
                         "pyspark", "fastai", "h2o", "keras", "transformers"):
            try:
                _f = getattr(_mlflow, _flavor, None)
                if _f is not None:
                    _f.autolog(disable=True, silent=True)
            except Exception:
                pass
        try:
            from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS as _AI
            for _intg, _cfg in list(_AI.items()):
                if isinstance(_cfg, dict):
                    _cfg["disable"] = True
        except Exception:
            pass
        del _mlflow
    except ImportError:
        pass
del _early_os

import json
import logging
import os
import shutil
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_full_deterministic_pipeline(
    cfg: Any,
    clean_artifacts: bool = True,
) -> None:
    """Run the full no-LLM pipeline for a config.

    This function is intentionally sequential and inline. Each stage calls
    the same low-level utility the CrewAI crew would, so deterministic
    output is identical to "crews with LLM disabled".

    Parameters
    ----------
    cfg : DemandForecastConfig
        Fully-resolved configuration (after ``load_config_from_yaml`` and
        ``cfg.resolve_paths``).
    clean_artifacts : bool, optional
        If True, delete ``cfg.artifact_base_path`` before running so the
        pipeline starts from a clean slate. Default True (matches the
        legacy ``--no-llm`` behaviour in ``databricks_runner.py``).
    """
    import numpy as np  # noqa: F401  (used by sub-imports)
    import pandas as pd

    from utils.agent_utilities import load_source_data
    from utils.period_utils import normalise_period_column

    logger.info("=" * 60)
    logger.info("DETERMINISTIC PIPELINE (no LLM)")
    logger.info("=" * 60)

    if clean_artifacts:
        if os.path.exists(cfg.artifact_base_path):
            logger.info("Deleting existing artifacts: %s", cfg.artifact_base_path)
            shutil.rmtree(cfg.artifact_base_path, ignore_errors=True)
        os.makedirs(cfg.artifact_base_path, exist_ok=True)
        logger.info("Fresh artifacts directory: %s", cfg.artifact_base_path)

    source_df = load_source_data(cfg.input_data_path)
    source_df = normalise_period_column(source_df, cfg.timestamp_col)

    # Force the period column to string dtype after normalisation.
    # ``normalise_period_column`` only converts dash-separated or real
    # date formats; when the source CSV stores year_week as pure digits
    # (e.g. 202614) pandas infers int64 on read, and every downstream
    # comparison against cfg.train_end (which is a string from YAML)
    # would otherwise raise TypeError. The cast is idempotent for
    # already-string columns and costs a single pass over the column.
    if not pd.api.types.is_object_dtype(source_df[cfg.timestamp_col]) \
            and not pd.api.types.is_string_dtype(source_df[cfg.timestamp_col]):
        source_df[cfg.timestamp_col] = source_df[cfg.timestamp_col].astype(str)

    KEY_COL = cfg.prediction_key_cols[0]
    DATE_COL = cfg.timestamp_col
    TARGET_COL = cfg.target_col

    # ── Stage 1: EDA ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[1/7] EDA PIPELINE (deterministic)")
    logger.info("=" * 60)
    from utils.eda import run_eda_pipeline

    eda_dir = os.path.join(cfg.artifact_base_path, "eda_output")
    train_df = source_df[source_df[DATE_COL] <= cfg.train_end]
    run_eda_pipeline(
        df=train_df, key_columns=[KEY_COL], date_col=DATE_COL, target_col=TARGET_COL,
        numeric_features=cfg.all_numeric_features(),
        categorical_features=cfg.all_categorical_features(),
        output_dir=eda_dir,
        period=52 if cfg.time_format == 'year_week' else 12,
        train_end=cfg.train_end, dead_key_threshold=26, exhaustive=True,
    )
    logger.info("  EDA complete")

    # ── Stage 1.5: Calendar feature synthesis ────────────────────
    _cal_cfg = getattr(cfg.design, 'calendar_features', None)
    if _cal_cfg is not None and getattr(_cal_cfg, 'enabled', False) and cfg.country:
        logger.info("\n" + "=" * 60)
        logger.info("[1.5/7] CALENDAR FEATURE SYNTHESIS (deterministic)")
        logger.info("=" * 60)
        try:
            from utils.calendar_features import inject_calendar_features
            _period_totals_pre = source_df.groupby(DATE_COL)[TARGET_COL].sum()
            _hc_pre = str(_period_totals_pre[_period_totals_pre > 0].index.max())
            source_df = inject_calendar_features(
                source_df,
                date_col=DATE_COL,
                time_format=cfg.time_format,
                country=cfg.country,
                subdivision=getattr(cfg, 'country_subdivision', None),
                overwrite_mode=getattr(_cal_cfg, 'overwrite_mode', 'always'),
                history_cutoff=_hc_pre,
                custom_events=getattr(_cal_cfg, 'custom_events', None) or None,
                lead_lag_windows=list(getattr(_cal_cfg, 'include_lead_lag_windows', [1, 2, 4])),
            )
            logger.info("  Calendar injection complete (country=%s, subdiv=%s)",
                        cfg.country, getattr(cfg, 'country_subdivision', None))
        except Exception as exc:
            logger.warning("Calendar injection failed: %s — continuing without synthesised holidays.", exc)
    else:
        if not cfg.country:
            logger.info("Calendar synthesis skipped: country is not set in config")

    # ── Stage 2: Feature Availability ────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[2/7] FEATURE AVAILABILITY (deterministic)")
    logger.info("=" * 60)
    from utils.feature_availability import run_feature_availability_pipeline

    fa_dir = os.path.join(cfg.artifact_base_path, "feature_availability_output")
    exclude = {KEY_COL, DATE_COL, TARGET_COL}
    feature_cols_all = [c for c in source_df.columns if c not in exclude]
    period_totals = source_df.groupby(DATE_COL)[TARGET_COL].sum()
    history_cutoff = str(period_totals[period_totals > 0].index.max())

    fa_result = run_feature_availability_pipeline(
        df=source_df, key_cols=[KEY_COL], date_col=DATE_COL, target_col=TARGET_COL,
        feature_cols=feature_cols_all, time_format=cfg.time_format,
        config_train_end=history_cutoff, output_dir=fa_dir,
        fa_config=getattr(cfg.design, 'feature_availability', None),
    )
    logger.info(
        "  Known: %d, Partial: %d, History-only: %d, Excluded: %d",
        fa_result.n_known_in_future, fa_result.n_partially_known,
        fa_result.n_history_only, fa_result.n_excluded,
    )

    with open(os.path.join(fa_dir, 'feature_availability_to_feature_context.json')) as f:
        fa_context = json.load(f)

    # ── Stage 3: Segmentation ────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[3/7] SEGMENTATION (deterministic)")
    logger.info("=" * 60)
    from utils.segmentation import run_segmentation_pipeline
    from utils.hierarchy_resolution import resolve_hierarchies, save_resolution

    per_key = pd.read_csv(os.path.join(eda_dir, "per_key_metrics.csv"))
    _int_key = 'key' if 'key' in per_key.columns else KEY_COL

    seg_dir = os.path.join(cfg.artifact_base_path, "seg_output")
    os.makedirs(seg_dir, exist_ok=True)

    # Resolve hierarchies ONCE, persist to seg_output/hierarchy_detection.json
    # so every downstream stage (inference, backtesting, features, training)
    # reads the same answer and never re-detects.
    hierarchy_res = resolve_hierarchies(
        config=cfg, source_df=source_df, seg_dir=seg_dir, force_redetect=True,
    )
    save_resolution(hierarchy_res, seg_dir)
    # Legacy single-list for the current segmentation signature
    hierarchy_cols = hierarchy_res.flat[:3] if hierarchy_res.flat else []

    seg_result = run_segmentation_pipeline(
        per_key_metrics=per_key, key_cols=[_int_key], output_dir=seg_dir,
        time_format=cfg.time_format, eda_dir=eda_dir,
        use_adaptive_features=True, use_hybrid_segmentation=True,
        source_df=source_df, date_col=DATE_COL, target_col=TARGET_COL,
        feature_availability_context=fa_context, enable_enriched_features=True,
        hierarchy_cols=hierarchy_cols,
    )
    logger.info("  %d segments", seg_result.n_segments)
    logger.info(
        "  Hierarchy: product=%s, customer=%s (primary=%s)",
        hierarchy_res.product, hierarchy_res.customer, hierarchy_res.primary_product_col,
    )

    # ── Stage 4: Feature Engineering ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[4/7] FEATURE ENGINEERING (deterministic — same as crew)")
    logger.info("=" * 60)
    from crews.feature_crew import (
        _create_eda_aware_fallback_strategy,
        _run_deterministic_executor,
    )

    feat_dir = os.path.join(cfg.artifact_base_path, "feature_output")
    os.makedirs(feat_dir, exist_ok=True)

    strategy = _create_eda_aware_fallback_strategy(
        eda_dir=eda_dir, seg_dir=seg_dir, time_format=cfg.time_format,
    )
    strategy.save(os.path.join(feat_dir, 'feature_strategy_decision.json'))
    logger.info(
        "  Strategy: lags=%s, windows=%s",
        strategy.target_lags, strategy.rolling_windows,
    )

    exec_result = _run_deterministic_executor(
        config=cfg, strategy=strategy,
        feat_dir=feat_dir, seg_dir=seg_dir, eda_dir=eda_dir,
    )
    logger.info(
        "  Features: %s, Train: %s, Val: %s",
        exec_result.get('n_features', '?'),
        exec_result.get('n_rows_train', '?'),
        exec_result.get('n_rows_val', '?'),
    )

    # Standardise key column to 'key' in saved feature files.
    # Format-agnostic: works whether feature engineering wrote parquet
    # (default) or CSV (legacy).  The peek+rewrite preserves the
    # original format because write_features_intermediate writes
    # parquet by default and removes any stale alternate format.
    from utils.feature_io import (
        features_intermediate_exists,
        features_intermediate_path,
        read_features_intermediate,
        write_features_intermediate,
    )
    for _base in ('train_features', 'val_features', 'test_features'):
        if not features_intermediate_exists(feat_dir, _base):
            continue
        # Peek (1 row) is cheap on both formats and tells us if we even
        # need to do the rewrite.
        _fdf_peek = read_features_intermediate(feat_dir, _base, nrows=1)
        if KEY_COL in _fdf_peek.columns and 'key' not in _fdf_peek.columns:
            _fdf_full = read_features_intermediate(feat_dir, _base)
            _fdf_full.rename(columns={KEY_COL: 'key'}, inplace=True)
            # Re-write in whichever format was on disk so we don't
            # silently switch a CSV-only run to parquet mid-pipeline.
            _orig_path = features_intermediate_path(feat_dir, _base)
            _was_csv = bool(_orig_path) and str(_orig_path).lower().endswith('.csv')
            write_features_intermediate(
                _fdf_full, feat_dir, _base, prefer_parquet=not _was_csv,
            )
            logger.info("  Renamed %s → key in %s", KEY_COL, _orig_path.name if _orig_path else _base)

    # Create training manifest (model level allocation) — crew does this internally
    if not os.path.exists(os.path.join(feat_dir, "training_manifest.csv")):
        logger.info("  Creating training manifest...")
        seg_df = pd.read_csv(os.path.join(seg_dir, "per_key_with_segments.csv"))
        from utils.intelligent_modeling import allocate_model_levels
        _key_in_seg = 'key' if 'key' in seg_df.columns else KEY_COL
        _alloc_cfg = getattr(cfg.design, 'model_level_allocation', None)
        _alloc_kwargs: Dict[str, Any] = {}
        if _alloc_cfg is not None:
            _alloc_kwargs = dict(
                min_individual_score=_alloc_cfg.min_individual_score,
                max_individual_pct=_alloc_cfg.max_individual_pct,
                min_segment_size=_alloc_cfg.min_segment_size,
                volume_override_quantile=_alloc_cfg.volume_override_quantile,
                forecastability_override=_alloc_cfg.forecastability_override,
                min_nonzero_obs_for_individual=getattr(_alloc_cfg, 'min_nonzero_obs_for_individual', 52),
                max_zero_fraction_for_individual=getattr(_alloc_cfg, 'max_zero_fraction_for_individual', 0.70),
                top_volume_bypass_quantile=getattr(_alloc_cfg, 'top_volume_bypass_quantile', 0.80),
            )
        allocation = allocate_model_levels(
            seg_df=seg_df, key_col=_key_in_seg, segment_col='segment_id',
            time_format=cfg.time_format,
            **_alloc_kwargs,
        )
        manifest = allocation.allocations
        if 'model_level' not in manifest.columns:
            manifest['model_level'] = manifest.apply(
                lambda r: r[_key_in_seg]
                if r.get('model_strategy', '') == 'individual'
                else str(r.get('segment_id', '0')),
                axis=1,
            )
        manifest.to_csv(os.path.join(feat_dir, "training_manifest.csv"), index=False)
        logger.info("  Manifest: %d keys", len(manifest))

    # ── Stage 5: Training ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[5/7] MODEL TRAINING (deterministic)")
    logger.info("=" * 60)
    from utils.model_training import run_full_training_pipeline

    model_dir = os.path.join(cfg.artifact_base_path, "model_artifacts")
    seg_context: dict = {}
    seg_ctx_path = os.path.join(seg_dir, "segmentation_to_training_context.json")
    if os.path.exists(seg_ctx_path):
        with open(seg_ctx_path) as f:
            seg_context = json.load(f)

    _strategy_path = os.path.join(feat_dir, 'feature_strategy_decision.json')
    training_result = run_full_training_pipeline(
        feature_dir=feat_dir, model_dir=model_dir, target_col=TARGET_COL,
        strategy_path=_strategy_path if os.path.exists(_strategy_path) else None,
        segmentation_context=seg_context,
        enable_meta_learning=True, enable_ensemble_optimization=True,
        enable_bias_correction=True, enable_forecast_calibration=True,
        forecast_horizon=cfg.forecast_horizon, key_col='key', date_col=DATE_COL,
        time_format=cfg.time_format,
        apply_bias_calibration=cfg.design.apply_bias_calibration,
        enable_walk_forward_cv=cfg.design.enable_walk_forward_cv,
        # ── Speed knobs from cfg.design (added 2026-04 for DIQ scale) ──
        # use_recursive_validation default = True (existing behaviour)
        # max_candidates_per_group default = 3 (existing behaviour)
        # parallel_training_workers default = 1 (existing behaviour)
        # Setting any below default trades model breadth/accuracy for
        # wall-clock training time — see config/schema.py docstrings.
        use_recursive_validation=cfg.design.use_recursive_validation,
        max_candidates_per_group=cfg.design.max_candidates_per_group,
        parallel_training_workers=cfg.design.parallel_training_workers,
        recursive_validation_lag=cfg.design.recursive_validation_lag,
        # When DMH is the inference path, run_full_training_pipeline
        # will auto-skip the recursive walk-forward inside per-MG
        # candidate evaluation (the multi-step error signal it
        # produces is superseded by DMH's per-horizon heads at
        # inference, so it adds no information for selection).  This
        # is the ~13x training-speedup we want when DMH is on.
        use_direct_multi_horizon=cfg.design.use_direct_multi_horizon,
    )
    logger.info(
        "  Training complete: %d models, wape=%.4f",
        training_result.models_trained, training_result.overall_wape,
    )

    # ── Stage 6: Inference ───────────────────────────────────────
    if cfg.should_forward_forecast:
        logger.info("\n" + "=" * 60)
        logger.info("[6/7] INFERENCE (deterministic)")
        logger.info("=" * 60)
        try:
            from utils.inference import run_inference_pipeline
            run_inference_pipeline(config=cfg)
        except Exception as exc:
            logger.error("  Inference failed: %s", exc)

    # ── Stage 7: Backtesting ─────────────────────────────────────
    if cfg.should_backtest:
        logger.info("\n" + "=" * 60)
        logger.info("[7/7] BACKTESTING (deterministic)")
        logger.info("=" * 60)
        try:
            from utils.backtesting import run_rolling_origin_backtest
            bt_result = run_rolling_origin_backtest(config=cfg)
            n_orig = getattr(bt_result, 'n_origins', getattr(bt_result, 'total_origins', '?'))
            wape = getattr(bt_result, 'overall_wape', getattr(bt_result, 'wape', '?'))
            logger.info("  Backtesting complete: %s origins, wape=%s", n_orig, wape)
        except Exception as exc:
            logger.error("  Backtesting failed: %s", exc)

    logger.info("\n" + "=" * 60)
    logger.info("DETERMINISTIC PIPELINE COMPLETE")
    logger.info("Artifacts: %s", cfg.artifact_base_path)
    logger.info("=" * 60)


__all__ = ["run_full_deterministic_pipeline"]
