#!/usr/bin/env python3
"""
Full Deterministic Pipeline + Benchmark Comparison.

Calls the EXACT SAME functions the CrewAI agents call, in the same order,
with the same parameters — but without any LLM overhead. Zero agents, full power.

Pipeline:
  1. run_eda_pipeline()                    — same as EDA crew executor
  2. run_feature_availability_pipeline()   — same as FA crew
  3. run_segmentation_pipeline()           — same as Segmentation crew executor
  4. run_leakage_free_feature_pipeline()   — same as Feature crew executor
  5. run_full_training_pipeline()          — same as Training crew executor
  6. run_inference_pipeline()              — same as Inference stage
  7. run_rolling_origin_backtest()         — same as Backtesting stage
  8. Benchmark comparison + Excel report

Usage:
    source .venv/bin/activate
    python run_full_benchmark.py --category SKIN_CARE
    python run_full_benchmark.py --category DEODORANTS_FRAGRANCES
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

CATEGORY_TO_CONFIG = {
    'SKIN_CARE': 'config/config_de_skincare.yaml',
    'SKIN_CLEANSING': 'config/config_de_skincleansing.yaml',
    'DEODORANTS_FRAGRANCES': 'config/config_de_deo.yaml',
    'DRESSINGS': 'config/config_de_dressings.yaml',
    'FABRIC_CLEANING': 'config/config_de_fab_clean.yaml',
    'HAIR_CARE': 'config/config_de_haircare.yaml',
    'HEALTHY_SNACKING': 'config/config_de_healthy_snack.yaml',
    'HOME_HYGIENE': 'config/config_de_home_hygiene.yaml',
    'ORAL_CARE': 'config/config_de_oral_care.yaml',
    'SCRATCH_COOKING_AIDS': 'config/config_de_cooking_aids.yaml',
}

CATEGORY_TO_BENCH = {
    'SKIN_CARE': 'SKIN CARE', 'SKIN_CLEANSING': 'SKIN CLEANSING',
    'DEODORANTS_FRAGRANCES': 'DEODORANT & FRAGRANCE', 'DRESSINGS': 'CONDIMENT',
    'FABRIC_CLEANING': 'FABRIC CLEANING', 'HAIR_CARE': 'HAIR CARE',
    'HEALTHY_SNACKING': 'MINI MEAL', 'HOME_HYGIENE': 'HOME & HYGIENE',
    'ORAL_CARE': 'ORAL CARE', 'SCRATCH_COOKING_AIDS': 'COOKING AID',
}

BENCHMARK_PATH = "TmpValidationData/DACH/CurrentForecastBenchmark.csv"


def yw_numeric_to_dash(yw: int) -> str:
    """Convert YYYYWW → 'YYYY-WW'. E.g., 202552 → '2025-52'."""
    return f"{yw // 100}-{yw % 100:02d}"


def main():
    parser = argparse.ArgumentParser(description="Full deterministic pipeline + benchmark")
    parser.add_argument('--category', required=True, choices=list(CATEGORY_TO_CONFIG.keys()))
    parser.add_argument('--output-dir', default='TmpValidationData/DACH/benchmark_reports')
    args = parser.parse_args()

    category = args.category
    config_path = CATEGORY_TO_CONFIG[category]
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 90)
    logger.info(f"FULL DETERMINISTIC PIPELINE: {category}")
    logger.info(f"Same functions as CrewAI agents — zero LLM, full power")
    logger.info("=" * 90)

    # ── Load config + auto-detect splits ─────────────────────────────
    from config.schema import load_config_from_yaml
    from utils.period_utils import normalise_period_column
    from utils.agent_utilities import load_source_data

    cfg = load_config_from_yaml(config_path)
    KEY_COL = cfg.prediction_key_cols[0]
    DATE_COL = cfg.timestamp_col
    TARGET_COL = cfg.target_col

    source_df = load_source_data(cfg.input_data_path)
    source_df = normalise_period_column(source_df, DATE_COL)
    cfg.auto_detect_splits(source_df)

    artifact_dir = cfg.artifact_base_path
    os.makedirs(artifact_dir, exist_ok=True)

    logger.info(f"  Data: {len(source_df):,} rows, {source_df[KEY_COL].nunique()} keys")
    logger.info(f"  Splits: train={cfg.train_start}..{cfg.train_end}, val={cfg.val_start}..{cfg.val_end}, test={cfg.test_start}..{cfg.test_end}")

    # ─────────────────────────────────────────────────────────────────
    # STAGE 1: EDA — run_eda_pipeline()
    # ─────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info("[1/7] EDA PIPELINE")
    logger.info(f"{'='*90}")
    from utils.eda import run_eda_pipeline

    eda_dir = os.path.join(artifact_dir, "eda_output")
    train_df = source_df[source_df[DATE_COL] <= cfg.train_end]

    eda_result = run_eda_pipeline(
        df=train_df,
        key_columns=[KEY_COL],
        date_col=DATE_COL,
        target_col=TARGET_COL,
        numeric_features=cfg.all_numeric_features(),
        categorical_features=cfg.all_categorical_features(),
        output_dir=eda_dir,
        period=52 if cfg.time_format == 'year_week' else 12,
        train_end=cfg.train_end,
        dead_key_threshold=26,
        exhaustive=True,
    )
    logger.info(f"  EDA complete: {eda_result.get('n_series', '?')} series analysed")

    # ─────────────────────────────────────────────────────────────────
    # STAGE 2: FEATURE AVAILABILITY DETECTION
    # ─────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info("[2/7] FEATURE AVAILABILITY DETECTION")
    logger.info(f"{'='*90}")
    from utils.feature_availability import run_feature_availability_pipeline

    fa_dir = os.path.join(artifact_dir, "feature_availability_output")
    exclude = {KEY_COL, DATE_COL, TARGET_COL}
    feature_cols_all = [c for c in source_df.columns if c not in exclude]

    fa_result = run_feature_availability_pipeline(
        df=source_df, key_cols=[KEY_COL], date_col=DATE_COL, target_col=TARGET_COL,
        feature_cols=feature_cols_all, time_format=cfg.time_format,
        config_train_end=cfg.train_end, output_dir=fa_dir,
    )
    logger.info(f"  Known: {fa_result.n_known_in_future}, History-only: {fa_result.n_history_only}, Excluded: {fa_result.n_excluded}")

    with open(os.path.join(fa_dir, 'feature_availability_to_feature_context.json')) as f:
        fa_context = json.load(f)

    # ─────────────────────────────────────────────────────────────────
    # STAGE 3: SEGMENTATION — run_segmentation_pipeline()
    # ─────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info("[3/7] SEGMENTATION PIPELINE")
    logger.info(f"{'='*90}")
    from utils.segmentation import run_segmentation_pipeline
    from utils.segmentation_enhanced import detect_hierarchy_columns

    per_key_path = os.path.join(eda_dir, "per_key_metrics.csv")
    per_key = pd.read_csv(per_key_path)

    # Detect hierarchies
    hier_info = detect_hierarchy_columns(source_df, [KEY_COL], DATE_COL, TARGET_COL)
    hierarchy_cols = [h['column'] for h in hier_info[:3]]

    seg_dir = os.path.join(artifact_dir, "segmentation_output")
    seg_result = run_segmentation_pipeline(
        per_key_metrics=per_key,
        key_cols=[KEY_COL],
        output_dir=seg_dir,
        time_format=cfg.time_format,
        eda_dir=eda_dir,
        use_adaptive_features=True,
        use_hybrid_segmentation=True,
        # Phase 2: Enriched segmentation
        source_df=source_df,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        feature_availability_context=fa_context,
        enable_enriched_features=True,
        hierarchy_cols=hierarchy_cols,
    )
    logger.info(f"  {seg_result.n_segments} segments, silhouette={seg_result.clustering_metrics.get('silhouette', 0):.3f}")

    # ─────────────────────────────────────────────────────────────────
    # STAGE 4: FEATURE ENGINEERING — run_leakage_free_feature_pipeline()
    # ─────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info("[4/7] FEATURE ENGINEERING PIPELINE")
    logger.info(f"{'='*90}")
    from utils.feature_engineering import run_leakage_free_feature_pipeline

    # Merge segment_id into source data
    seg_df = pd.read_csv(os.path.join(seg_dir, "per_key_with_segments.csv"))
    seg_map = seg_df[[KEY_COL, 'segment_id']].drop_duplicates(KEY_COL)
    source_with_seg = source_df.merge(seg_map, on=KEY_COL, how='left')
    source_with_seg['segment_id'] = source_with_seg['segment_id'].fillna(0).astype(int).astype(str)

    known_features = fa_context.get('known_in_future_features', [])
    history_only = fa_context.get('history_only_features', [])
    forecast_lag = getattr(cfg.design, 'forecast_lag', 4)

    feat_dir = os.path.join(artifact_dir, "feature_output")
    fe_result = run_leakage_free_feature_pipeline(
        df=source_with_seg,
        key_cols=[KEY_COL], date_col=DATE_COL, target_col=TARGET_COL,
        train_start=cfg.train_start, train_end=cfg.train_end,
        val_start=cfg.val_start, val_end=cfg.val_end,
        test_start=cfg.test_start, test_end=cfg.test_end,
        forecast_lag=forecast_lag,
        segment_col='segment_id',
        target_lags=[1, 2, 4, 8, 13, 26],
        rolling_windows=[4, 8, 13, 26],
        rolling_stats=['mean', 'std'],
        seasonal_period=52 if cfg.time_format == 'year_week' else 12,
        allowed_external_cols=known_features,
        hierarchy_cols=hierarchy_cols[:2],
        enable_rich_features=True,
        future_unknown_features=history_only,
        output_dir=feat_dir,
    )

    # Filter to numeric features only
    feature_cols = [c for c in fe_result.feature_cols
                    if c in fe_result.train_df.columns
                    and pd.api.types.is_numeric_dtype(fe_result.train_df[c])]
    logger.info(f"  {len(feature_cols)} numeric features created")

    # Standardise key column to 'key' in feature files — downstream pipeline expects this
    for _df in [fe_result.train_df, fe_result.val_df, fe_result.test_df]:
        if KEY_COL != 'key' and KEY_COL in _df.columns and 'key' not in _df.columns:
            _df.rename(columns={KEY_COL: 'key'}, inplace=True)

    # Save features for training pipeline (parquet by default — see
    # utils/feature_io.py)
    from utils.feature_io import write_features_intermediate
    write_features_intermediate(fe_result.train_df, feat_dir, "train_features")
    write_features_intermediate(fe_result.val_df,   feat_dir, "val_features")
    write_features_intermediate(fe_result.test_df,  feat_dir, "test_features")

    # Create training manifest (model level allocation: individual vs segment-pooled)
    logger.info("  Creating training manifest (model level allocation)...")
    from utils.intelligent_modeling import allocate_model_levels
    allocation = allocate_model_levels(
        seg_df=seg_df,
        key_col=KEY_COL,
        segment_col='segment_id',
        time_format=cfg.time_format,
    )
    manifest = allocation.allocations  # DataFrame with key_col, model_level, strategy, etc.
    # Standardise key column to 'key' — the entire downstream pipeline expects this name
    if KEY_COL != 'key' and KEY_COL in manifest.columns:
        manifest = manifest.rename(columns={KEY_COL: 'key'})
    # Ensure required columns exist
    if 'model_level' not in manifest.columns and 'model_group' not in manifest.columns:
        manifest['model_level'] = manifest.apply(
            lambda r: r['key'] if r.get('strategy', '') == 'individual' else str(r.get('segment_id', '0')),
            axis=1
        )
    manifest.to_csv(os.path.join(feat_dir, "training_manifest.csv"), index=False)
    n_individual = len(allocation.individual_keys)
    n_segment = sum(len(v) for v in allocation.segment_keys.values())
    logger.info(f"  Manifest: {len(manifest)} keys, "
                f"{manifest.get('model_level', manifest.get('model_group', pd.Series())).nunique()} model levels, "
                f"individual={n_individual}, segment={n_segment}")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 5: MODEL TRAINING — run_full_training_pipeline()
    # ═════════════════════════════════════════════════════════════════
    # This is the SAME function the CrewAI training agent calls.
    # It handles: segment-pooled + individual model training,
    # walk-forward CV, pattern-aware model selection, ensemble creation,
    # bias calibration, and saves all artifacts needed by backtesting.
    logger.info(f"\n{'='*90}")
    logger.info("[5/7] MODEL TRAINING")
    logger.info(f"{'='*90}")

    model_dir = os.path.join(artifact_dir, "model_artifacts")
    os.makedirs(model_dir, exist_ok=True)

    # Load segmentation context
    seg_context_path = os.path.join(seg_dir, "segmentation_to_training_context.json")
    seg_context = {}
    if os.path.exists(seg_context_path):
        with open(seg_context_path) as f:
            seg_context = json.load(f)

    # Run the SAME training pipeline the CrewAI agent calls.
    # This handles everything: segment-pooled + individual models, walk-forward CV,
    # pattern-aware selection, ensemble creation, bias calibration, and saves ALL
    # artifacts needed by backtesting (final_model_specs.json, model pickles, etc.)
    from utils.model_training import run_full_training_pipeline

    training_result = run_full_training_pipeline(
        feature_dir=feat_dir,
        model_dir=model_dir,
        target_col=TARGET_COL,
        segmentation_context=seg_context,
        enable_meta_learning=True,
        enable_ensemble_optimization=True,
        enable_bias_correction=True,
        enable_forecast_calibration=True,
        forecast_horizon=cfg.forecast_horizon,
        key_col=KEY_COL,
        date_col=DATE_COL,
        time_format=cfg.time_format,
        apply_bias_calibration=cfg.design.apply_bias_calibration,
        enable_walk_forward_cv=cfg.design.enable_walk_forward_cv,
    )
    logger.info(f"  Training complete: {training_result.models_trained} models trained, "
                f"{training_result.models_failed} failed, "
                f"overall_wape={training_result.overall_wape:.4f}")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 6: ROLLING-ORIGIN BACKTESTING (using best model)
    # ═════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info("[6/7] ROLLING-ORIGIN BACKTESTING (full infrastructure)")
    logger.info(f"{'='*90}")
    logger.info("  Using run_rolling_origin_backtest() — the CORRECT pipeline:")
    logger.info("    - Feature regeneration at each origin (no data leakage)")
    logger.info("    - Segment-pooled model retraining (respects segmentation)")
    logger.info("    - Recursive multi-step forecasting (proper lag updates)")
    logger.info("    - Bias calibration + hierarchical reconciliation")

    from utils.backtesting import run_rolling_origin_backtest

    bt_result = run_rolling_origin_backtest(config=cfg, verbose=True)

    logger.info(f"  Backtesting complete: {bt_result.n_origins} origins, "
                f"overall_wape={bt_result.overall_wape:.4f}")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 7: INFERENCE — forward forecast (if run_mode requires it)
    # ═════════════════════════════════════════════════════════════════
    if cfg.should_forward_forecast:
        logger.info(f"\n{'='*90}")
        logger.info("[7/7] INFERENCE — forward forecast")
        logger.info(f"{'='*90}")
        from utils.inference import run_inference_pipeline
        run_inference_pipeline(config=cfg)
    else:
        logger.info(f"\n[7/7] Inference SKIPPED (run_mode={cfg.run_mode})")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 8: BENCHMARK COMPARISON
    # ═════════════════════════════════════════════════════════════════

    if os.path.exists(BENCHMARK_PATH) and category in CATEGORY_TO_BENCH:
        logger.info(f"\n{'='*90}")
        logger.info("[BENCHMARK] BENCHMARK COMPARISON")
        logger.info(f"{'='*90}")

        import subprocess
        result = subprocess.run(
            [sys.executable, "-u", "compare_benchmark.py",
             "--category", category, "--output-dir", args.output_dir],
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning("Benchmark comparison had issues — check report")
    else:
        logger.info("No benchmark file found, skipping comparison")

    total = time.time() - t0
    logger.info(f"\n{'='*90}")
    logger.info(f"COMPLETE: {category} in {total:.0f}s ({total/60:.1f} min)")
    logger.info(f"Artifacts: {artifact_dir}/")
    logger.info(f"{'='*90}")


if __name__ == '__main__':
    main()
