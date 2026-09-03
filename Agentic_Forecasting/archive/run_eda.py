#!/usr/bin/env python3
"""
FEU-Agentic-Forecasting Demand Forecasting - EDA Crew Runner
==============================================

Runs ONLY the EDA (Exploratory Data Analysis) crew.

This is the first stage of the pipeline and has no dependencies on other stages.

Mode: DETERMINISTIC core analysis + LLM for context/rationale/insights
- Core EDA (per_key_metrics.csv, etc.) runs deterministically without LLM
- LLM Analyst generates:
  - Context files with intelligent rationale and recommendations
  - Exhaustive insights report for data scientists

Usage:
    python run_eda.py --config config/config.yaml

Outputs (Core - Deterministic):
    - eda_output/per_key_metrics.csv (27+ columns)
    - eda_output/eda_summary.json
    - eda_output/seasonality_analysis.json
    - eda_output/trend_analysis.json
    - eda_output/feature_importance.csv

Outputs (LLM-Generated):
    - eda_output/eda_to_segmentation_context.json
    - eda_output/eda_to_feature_context.json
    - eda_output/eda_to_training_context.json
    - eda_output/eda_insights_report.md (EXHAUSTIVE insights for data scientists)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: str = "run_eda.log"):
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
# EDA Crew Runner
# =============================================================================

def run_eda_only(
    config_yaml_path: str,
    enable_trace: bool = True,
) -> bool:
    """
    Run the EDA crew.

    Mode: DETERMINISTIC core + LLM context/rationale
    - Core EDA analysis runs deterministically (no LLM)
    - Context files are enhanced with LLM-generated rationale

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
    from crews.eda_crew import run_eda_deterministic

    start_time = datetime.now()

    logging.info("\n" + "="*70)
    logging.info("FEU-AGENTIC-FORECASTING EDA CREW - STANDALONE RUN")
    logging.info("="*70)
    logging.info(f"Config: {config_yaml_path}")
    logging.info("Mode: DETERMINISTIC core + LLM context/rationale")
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


        # Load LLM for context file creation
        llm = get_llm(config_path=config_yaml_path)
        logging.info("LLM loaded for context file creation")
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
                cleanup_litellm_callbacks as cleanup_func,
            )
            setup_litellm_callbacks()
            cost_tracker = get_cost_tracker()
            cost_tracker.start_pipeline()
            cost_tracker.start_crew("EDA Crew")

            model_id = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
            cost_tracker.set_model(model_id.replace("bedrock/", ""))
            cleanup_litellm_callbacks = cleanup_func
            logging.info("Token tracking enabled\n")
        except ImportError:
            pass

        # Run EDA - deterministic core + LLM for context
        logging.info("Starting EDA...")
        logging.info("  Core analysis: DETERMINISTIC (no LLM)")
        logging.info("  Context files: LLM Analyst (rationale + recommendations)")
        logging.info("-" * 40)

        result = run_eda_deterministic(
            config=cfg,
            use_agents_for_context=True,  # Always use LLM for context
            llm=llm,
        )

        duration = (datetime.now() - start_time).total_seconds()

        # End cost tracking
        if cost_tracker:
            try:
                eda_output_dir = os.path.join(cfg.artifact_base_path, "eda_output")
                crew_report = cost_tracker.end_crew("EDA Crew", eda_output_dir)
                logging.info(f"\nTokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output")
                logging.info(f"Est. Cost: ${crew_report.total_cost_usd:.4f}")
            except Exception as e:
                logging.warning(f"Could not save cost report: {e}")

        logging.info("\n" + "="*70)
        logging.info("EDA CREW COMPLETED SUCCESSFULLY")
        logging.info("="*70)
        logging.info(f"Duration: {duration / 60.0:.2f} minutes")
        logging.info(f"\nOutputs:")
        logging.info(f"  - Per-key metrics: {result.per_key_metrics_path}")
        logging.info(f"  - Global summary: {result.global_eda_summary_path}")
        logging.info(f"  - Context files:")
        logging.info(f"      - {result.eda_to_segmentation_context_path}")
        logging.info(f"      - {result.eda_dir}/eda_to_feature_context.json")
        logging.info(f"      - {result.eda_dir}/eda_to_training_context.json")
        logging.info(f"  - Insights report: {result.eda_insights_report_path}")
        logging.info("="*70 + "\n")

        # Cleanup
        if cleanup_litellm_callbacks:
            cleanup_litellm_callbacks()

        return True

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()

        logging.error("\n" + "="*70)
        logging.error("EDA CREW FAILED")
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
        description='Run FEU-Agentic-Forecasting EDA crew (deterministic core + LLM rationale)'
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

    success = run_eda_only(
        config_yaml_path=args.config,
        enable_trace=not args.no_trace,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
