#!/usr/bin/env python3
"""
FEU-Agentic-Forecasting Demand Forecasting - Model Training Crew Runner
=========================================================

Runs ONLY the Model Training crew.

This runner stops after training - it does NOT run diagnostics or pipeline
generation. Use this for iterating on models.

Prerequisites:
    - EDA crew must have been run (eda_output/ must exist)
    - Segmentation crew must have been run (segmentation_output/ must exist)
    - Feature Engineering crew must have been run (feature_output/ must exist)

Usage:
    python run_training.py --config config/config.yaml

Outputs:
    - model_artifacts/final_model_specs.json
    - model_artifacts/model_*.pkl (trained model files)
    - model_artifacts/val_predictions.csv
    - model_artifacts/test_predictions.csv
    - model_artifacts/training_to_diagnostic_context.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Optional

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: str = "run_training.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# =============================================================================
# Prerequisite Check
# =============================================================================

def check_prerequisites(artifact_base_path: str) -> bool:
    """Check that EDA, Segmentation, and Feature Engineering crews have been run."""
    eda_output_dir = os.path.join(artifact_base_path, "eda_output")
    seg_output_dir = os.path.join(artifact_base_path, "seg_output")
    feature_output_dir = os.path.join(artifact_base_path, "feature_output")

    required_files = [
        # EDA outputs
        os.path.join(eda_output_dir, "eda_summary.json"),
        os.path.join(eda_output_dir, "per_key_metrics.csv"),
        # Segmentation outputs
        os.path.join(seg_output_dir, "per_key_with_segments.csv"),
        os.path.join(seg_output_dir, "feature_recommendations.json"),
        # Feature Engineering outputs — bare basename markers below
        # tolerate parquet OR csv (see _exists_either).
        os.path.join(feature_output_dir, "train_features"),
        os.path.join(feature_output_dir, "val_features"),
        os.path.join(feature_output_dir, "training_manifest.csv"),
    ]

    def _exists_either(p: str) -> bool:
        if "." in os.path.basename(p):
            return os.path.exists(p)
        return os.path.exists(p + ".parquet") or os.path.exists(p + ".csv")

    missing = [f for f in required_files if not _exists_either(f)]

    if missing:
        logging.error("Prerequisite check FAILED!")
        logging.error("The following required files are missing:")
        for f in missing:
            logging.error(f"  - {f}")
        logging.error("\nPlease run the preceding crews first:")
        logging.error("  1. python run_eda.py --config config/config.yaml")
        logging.error("  2. python run_segmentation.py --config config/config.yaml")
        logging.error("  3. python run_feature.py --config config/config.yaml")
        return False

    logging.info("Prerequisite check passed - EDA, Segmentation, and Feature outputs found")
    return True


# =============================================================================
# Model Training Crew Runner
# =============================================================================

def run_training_only(
    config_yaml_path: str,
    enable_trace: bool = True,
) -> bool:
    """
    Run only the Model Training crew.

    Parameters
    ----------
    config_yaml_path : str
        Path to config YAML file
    enable_trace : bool
        Whether to enable detailed CrewAI trace logging

    Returns
    -------
    bool
        True if completed successfully, False otherwise
    """
    from config.schema import load_config_from_yaml
    from config.llm_config import get_llm
    from crews.training_crew import run_training_crew

    start_time = datetime.now()

    logging.info("\n" + "="*70)
    logging.info("FEU-AGENTIC-FORECASTING MODEL TRAINING CREW - STANDALONE RUN")
    logging.info("="*70)
    logging.info(f"Config: {config_yaml_path}")
    logging.info(f"Trace logging: {'ENABLED' if enable_trace else 'DISABLED'}")
    logging.info("="*70 + "\n")

    try:
        # Load configuration
        logging.info("Loading configuration...")
        cfg = load_config_from_yaml(config_yaml_path)
        # Bootstrap: auto-detect splits + normalise periods
        from utils.period_utils import bootstrap_config as _bootstrap
        cfg, _src_df = _bootstrap(config_yaml_path)
        del _src_df  # Free memory; crews load data themselves

        llm = get_llm(config_path=config_yaml_path)
        logging.info("Configuration loaded\n")

        # Check prerequisites
        if not check_prerequisites(cfg.artifact_base_path):
            return False

        # Set up trace logging
        if enable_trace:
            try:
                from utils.trace_logging import (
                    setup_crewai_trace_logging,
                    set_crewai_logging_env_vars,
                )
                set_crewai_logging_env_vars()
                trace_dir = setup_crewai_trace_logging(
                    artifact_base_path=cfg.artifact_base_path,
                    log_level="DEBUG",
                    also_log_to_console=False,
                )
                logging.info(f"CrewAI trace logging enabled at: {trace_dir}")
            except ImportError as e:
                logging.warning(f"Could not enable trace logging: {e}")

        # Initialize cost tracking
        try:
            from utils.cost_tracking import (
                get_cost_tracker,
                setup_litellm_callbacks,
                cleanup_litellm_callbacks,
            )
            setup_litellm_callbacks()
            cost_tracker = get_cost_tracker()
            cost_tracker.start_pipeline()
            cost_tracker.start_crew("Model Training Crew")

            model_id = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
            cost_tracker.set_model(model_id.replace("bedrock/", ""))
            logging.info("Token tracking enabled\n")
        except ImportError:
            cost_tracker = None
            cleanup_litellm_callbacks = None

        # Run Model Training crew
        logging.info("Starting Model Training Crew...")
        logging.info("-" * 40)

        result = run_training_crew(llm=llm, config=cfg, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        # End cost tracking
        if cost_tracker:
            try:
                model_output_dir = os.path.join(cfg.artifact_base_path, "model_artifacts")
                crew_report = cost_tracker.end_crew("Model Training Crew", model_output_dir)
                logging.info(f"\nTokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output")
                logging.info(f"Est. Cost: ${crew_report.total_cost_usd:.4f}")
            except Exception as e:
                logging.warning(f"Could not save cost report: {e}")

        logging.info("\n" + "="*70)
        logging.info("MODEL TRAINING CREW COMPLETED SUCCESSFULLY")
        logging.info("="*70)
        logging.info(f"Duration: {duration / 60.0:.2f} minutes")
        logging.info(f"\nModel Training Outputs:")
        logging.info(f"  - Model specs: {result.final_model_specs_path}")
        logging.info(f"  - Model directory: {result.model_dir}")

        logging.info("\n" + "="*70)
        logging.info("NOTE: Diagnostics and Pipeline Generation are SKIPPED in this runner.")
        logging.info("To run full pipeline, use the main orchestrator or run individual crews.")
        logging.info("="*70 + "\n")

        # Cleanup
        if cleanup_litellm_callbacks:
            cleanup_litellm_callbacks()

        return True

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()

        logging.error("\n" + "="*70)
        logging.error("MODEL TRAINING CREW FAILED")
        logging.error("="*70)
        logging.error(f"Duration before failure: {duration / 60.0:.2f} minutes")
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
        logging.error("="*70 + "\n")

        try:
            from utils.cost_tracking import cleanup_litellm_callbacks
            cleanup_litellm_callbacks()
        except Exception:
            pass

        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run FEU-Agentic-Forecasting Model Training crew only (requires EDA, Segmentation, and Feature Engineering to have run)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file (e.g., config/config.yaml)'
    )
    parser.add_argument(
        '--no-trace',
        action='store_true',
        help='Disable detailed CrewAI trace logging'
    )

    args = parser.parse_args()

    setup_logging()

    success = run_training_only(
        config_yaml_path=args.config,
        enable_trace=not args.no_trace,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
