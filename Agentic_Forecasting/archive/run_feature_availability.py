#!/usr/bin/env python3
"""
FEU-Agentic-Forecasting - Feature Availability Detection Runner
================================================================

Runs the Feature Availability Detection crew standalone.
This step auto-detects which features are available in future periods
and generates frozen embedding specs for history-only features.

Prerequisites:
    - EDA crew must have been run (eda_output/ must exist) [optional]
    - Source data must be accessible

Usage:
    python run_feature_availability.py --config config/config.yaml

    # Skip the optional LLM analyst phase (faster, deterministic only):
    python run_feature_availability.py --config config/config.yaml --skip-llm

Outputs:
    - feature_availability_output/feature_availability_result.json
    - feature_availability_output/feature_availability_to_feature_context.json
    - feature_availability_output/feature_availability_summary.txt
    - feature_availability_output/analyst_recommendations.json (if LLM enabled)
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

def setup_logging(log_file: str = "run_feature_availability.log"):
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
# Feature Availability Detection Runner
# =============================================================================

def run_feature_availability_only(
    config_yaml_path: str,
    enable_trace: bool = True,
    skip_llm: bool = False,
) -> bool:
    """
    Run only the Feature Availability Detection crew.

    Parameters
    ----------
    config_yaml_path : str
        Path to config YAML file
    enable_trace : bool
        Whether to enable detailed CrewAI trace logging
    skip_llm : bool
        Skip the LLM analyst phase (use deterministic results only)

    Returns
    -------
    bool
        True if completed successfully, False otherwise
    """
    from config.schema import load_config_from_yaml
    from config.llm_config import get_llm
    from crews.feature_availability_crew import run_feature_availability_crew

    start_time = datetime.now()

    logging.info("\n" + "=" * 70)
    logging.info("FEU-AGENTIC-FORECASTING FEATURE AVAILABILITY DETECTION - STANDALONE")
    logging.info("=" * 70)
    logging.info(f"Config: {config_yaml_path}")
    logging.info(f"Trace logging: {'ENABLED' if enable_trace else 'DISABLED'}")
    logging.info(f"LLM analyst: {'DISABLED' if skip_llm else 'ENABLED'}")
    logging.info("=" * 70 + "\n")

    try:
        # Load configuration
        logging.info("Loading configuration...")
        cfg = load_config_from_yaml(config_yaml_path)
        # Bootstrap: auto-detect splits + normalise periods
        from utils.period_utils import bootstrap_config as _bootstrap
        cfg, _src_df = _bootstrap(config_yaml_path)
        del _src_df  # Free memory; crews load data themselves


        # Load LLM (needed for optional analyst phase)
        llm = None
        if not skip_llm:
            try:
                llm = get_llm(config_path=config_yaml_path)
                logging.info("LLM loaded for analyst phase")
            except Exception as e:
                logging.warning(f"Could not load LLM: {e}. Running deterministic only.")
                skip_llm = True

        logging.info("Configuration loaded\n")

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
        cost_tracker = None
        cleanup_litellm_callbacks = None
        try:
            from utils.cost_tracking import (
                get_cost_tracker,
                setup_litellm_callbacks,
                cleanup_litellm_callbacks as _cleanup,
            )
            cleanup_litellm_callbacks = _cleanup
            setup_litellm_callbacks()
            cost_tracker = get_cost_tracker()
            cost_tracker.start_pipeline()
            cost_tracker.start_crew("Feature Availability Detection")

            model_id = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
            cost_tracker.set_model(model_id.replace("bedrock/", ""))
            logging.info("Token tracking enabled\n")
        except ImportError:
            pass

        # Run Feature Availability Detection
        logging.info("Starting Feature Availability Detection...")
        logging.info("-" * 40)

        result = run_feature_availability_crew(
            llm=llm,
            config=cfg,
            config_yaml_path=config_yaml_path,
            skip_llm_analyst=skip_llm,
        )

        duration = (datetime.now() - start_time).total_seconds()

        # End cost tracking
        if cost_tracker:
            try:
                crew_report = cost_tracker.end_crew(
                    "Feature Availability Detection",
                    result.output_dir,
                )
                logging.info(f"\nTokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output")
                logging.info(f"Est. Cost: ${crew_report.total_cost_usd:.4f}")
            except Exception as e:
                logging.warning(f"Could not save cost report: {e}")

        if result.success:
            logging.info("\n" + "=" * 70)
            logging.info("FEATURE AVAILABILITY DETECTION COMPLETED SUCCESSFULLY")
            logging.info("=" * 70)
            logging.info(f"Duration: {duration / 60.0:.2f} minutes")
            logging.info(f"Features analyzed: {result.n_features_analyzed}")
            logging.info(f"  Known in future: {result.n_known_in_future}")
            logging.info(f"  History only:    {result.n_history_only}")
            logging.info(f"  Partially known: {result.n_partially_known}")
            logging.info(f"  Excluded:        {result.n_excluded}")
            logging.info(f"Detected cutoff:   {result.detected_cutoff}")
            logging.info(f"\nOutputs:")
            logging.info(f"  - Result:  {result.result_path}")
            logging.info(f"  - Context: {result.context_path}")
            logging.info(f"  - Summary: {result.summary_path}")
            logging.info("=" * 70 + "\n")
        else:
            logging.error(f"Feature Availability Detection failed: {result.error}")

        # Cleanup
        if cleanup_litellm_callbacks:
            cleanup_litellm_callbacks()

        return result.success

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()

        logging.error("\n" + "=" * 70)
        logging.error("FEATURE AVAILABILITY DETECTION FAILED")
        logging.error("=" * 70)
        logging.error(f"Duration before failure: {duration / 60.0:.2f} minutes")
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
        logging.error("=" * 70 + "\n")

        try:
            from utils.cost_tracking import cleanup_litellm_callbacks
            cleanup_litellm_callbacks()
        except Exception:
            pass

        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run FEU-Agentic-Forecasting Feature Availability Detection (auto-detects feature availability in future periods)'
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
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip the optional LLM analyst phase (faster, deterministic only)'
    )

    args = parser.parse_args()

    setup_logging()

    success = run_feature_availability_only(
        config_yaml_path=args.config,
        enable_trace=not args.no_trace,
        skip_llm=args.skip_llm,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
