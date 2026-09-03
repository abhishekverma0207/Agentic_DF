#!/usr/bin/env python3
"""
Databricks Entry Point for FEU-Agentic-Forecasting Pipeline

This script is the recommended entry point when running on Databricks.
It handles:
  1. Installing Python dependencies (for EDA and Inference stages)
  2. Databricks-specific path resolution
  3. Ensuring artifact directories exist
  4. Forwarding to the main runner.py

Usage (Databricks Job):
    python databricks_runner.py --config /Workspace/Repos/.../config/config.yaml

Usage (Databricks CLI):
    databricks jobs run-now --job-id <JOB_ID>

The config.yaml should have:
    llm_provider: "databricks"
    databricks_base_path: "/Volumes/catalog/schema/path"
"""

import sys
import os

# =============================================================================
# CRITICAL: Disable OpenTelemetry SDK to avoid version conflicts with Databricks
# This must be done BEFORE any imports that might trigger opentelemetry
# =============================================================================
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"  # Also disable CrewAI's telemetry

import argparse
import logging
import subprocess
import importlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def install_dependencies(project_root: str, force: bool = False) -> bool:
    """
    Install Python dependencies from requirements.txt.

    This function is called for EDA and Inference stages (the entry points).
    Other stages (segmentation, feature, training, etc.) assume dependencies
    are already installed by EDA.

    On Databricks, we use regular pip install (not --target) because:
    1. Job clusters have writable site-packages
    2. --target can cause issues with complex dependency chains
    3. Databricks handles package isolation at the cluster level

    Parameters
    ----------
    project_root : str
        Path to the project root containing requirements.txt
    force : bool
        If True, reinstall even if packages appear to be installed

    Returns
    -------
    bool
        True if installation succeeded or was skipped, False on error
    """
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"sys.path[0:5]: {sys.path[:5]}")

    # Check if crewai is already installed (quick check)
    if not force:
        try:
            importlib.invalidate_caches()
            import crewai
            logger.info(f"Dependencies already installed (crewai found at {crewai.__file__})")
            return True
        except ImportError:
            logger.info("crewai not found, installing dependencies...")
        except Exception as e:
            logger.info(f"crewai import error: {e}, will reinstall...")

    requirements_path = os.path.join(project_root, "requirements.txt")

    if not os.path.exists(requirements_path):
        logger.error(f"requirements.txt not found at: {requirements_path}")
        return False

    logger.info("=" * 60)
    logger.info("Installing Python dependencies from requirements.txt")
    logger.info("=" * 60)
    logger.info(f"Requirements file: {requirements_path}")

    try:
        # =================================================================
        # APPROACH: Use regular pip install (not --target)
        # On Databricks job clusters, site-packages is writable
        # This is more reliable than --target for complex packages
        # =================================================================
        logger.info("Installing from requirements.txt (this may take 5-10 minutes)...")
        logger.info("Using standard pip install (Databricks job cluster mode)...")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "-r", requirements_path,
             "--upgrade",
             "--no-warn-script-location",
             "--quiet"],  # Less verbose
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        if result.returncode == 0:
            logger.info("Dependencies installed successfully")
        else:
            logger.warning(f"Some packages may have failed (return code: {result.returncode})")
            if result.stderr:
                # Only show last 1000 chars of stderr to avoid noise
                logger.warning(f"stderr (last 1000 chars): {result.stderr[-1000:]}")

        # Clear any cached failed imports - must be done before re-importing
        logger.info("Clearing cached module imports...")
        modules_to_clear = [m for m in list(sys.modules.keys())
                          if 'crewai' in m or 'opentelemetry' in m or 'litellm' in m]
        for m in modules_to_clear:
            del sys.modules[m]
        logger.info(f"Cleared {len(modules_to_clear)} cached modules")

        # CRITICAL: Invalidate import caches after pip install
        importlib.invalidate_caches()
        logger.info("Import caches invalidated")

        # Quick verification of crewai
        logger.info("Verifying crewai installation...")
        try:
            import crewai
            logger.info(f"crewai: OK (version: {getattr(crewai, '__version__', 'unknown')})")
            logger.info(f"crewai location: {crewai.__file__}")
        except ImportError as e:
            logger.error(f"crewai import failed: {e}")
            logger.error(f"sys.path: {sys.path[:10]}")

            # Debug: Check what pip thinks is installed
            logger.info("Checking pip list for crewai...")
            pip_result = subprocess.run(
                [sys.executable, "-m", "pip", "list"],
                capture_output=True,
                text=True
            )
            crewai_lines = [line for line in pip_result.stdout.split('\n') if 'crewai' in line.lower()]
            logger.info(f"pip list crewai: {crewai_lines}")

            # Check pip show for location
            pip_show = subprocess.run(
                [sys.executable, "-m", "pip", "show", "crewai"],
                capture_output=True,
                text=True
            )
            logger.info(f"pip show crewai: {pip_show.stdout[:500]}")

            return False
        except Exception as e:
            logger.error(f"crewai import error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        # Also verify litellm (another critical dependency)
        try:
            import litellm
            logger.info(f"litellm: OK (version: {getattr(litellm, '__version__', 'unknown')})")
        except Exception as e:
            logger.warning(f"litellm import warning: {e}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("pip install timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Databricks entry point for FEU-Agentic-Forecasting Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["full", "eda", "feature_availability", "segmentation", "feature", "training", "inference", "diagnostic", "backtest"],
        default="full",
        help="Pipeline stage to run (default: full)"
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable CrewAI trace logging"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation (use if dependencies are pre-installed)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun all crews even if outputs already exist"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip crews that have already completed (default: True)"
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Disable skipping of completed crews (run all stages)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run deterministic pipeline WITHOUT any LLM/CrewAI agents. "
             "Calls the same functions the agents call, but directly. "
             "Faster, cheaper, more reliable. Recommended for production."
    )

    args = parser.parse_args()

    # Handle skip-completed logic
    if args.no_skip_completed:
        args.skip_completed = False

    logger.info("=" * 60)
    logger.info("FEU-Agentic-Forecasting: Databricks Runner")
    logger.info("=" * 60)

    # Validate config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    logger.info(f"Config file: {args.config}")
    logger.info(f"Stage: {args.stage}")

    # Determine project root
    # Note: __file__ is not defined in Databricks notebooks, so we derive project root from config path
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # In Databricks notebooks, __file__ is not defined
        # Derive project root from config path (config is in <project_root>/config/config.yaml)
        config_dir = os.path.dirname(os.path.abspath(args.config))
        project_root = os.path.dirname(config_dir)
        logger.info("Running in Databricks notebook context (__file__ not defined)")

    logger.info(f"Project root: {project_root}")

    # Install dependencies for entry-point stages (EDA, Inference, Full)
    # Other stages assume dependencies were installed by EDA
    if not args.skip_install and args.stage in ["eda", "inference", "full"]:
        logger.info(f"Stage '{args.stage}' requires dependency installation check")
        if not install_dependencies(project_root):
            logger.error("Failed to install dependencies. Exiting.")
            sys.exit(1)
    elif args.skip_install:
        logger.info("Skipping dependency installation (--skip-install flag)")
    else:
        logger.info(f"Stage '{args.stage}' assumes dependencies installed by prior EDA run")
        # Still do a quick check
        try:
            import crewai
            logger.info("Dependencies check: crewai found")
        except ImportError:
            logger.warning("crewai not found! Running dependency installation...")
            if not install_dependencies(project_root):
                logger.error("Failed to install dependencies. Exiting.")
                sys.exit(1)

    # Add project root to path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added to sys.path: {project_root}")

    # Load and validate config
    logger.info("Loading configuration...")
    from config.schema import load_config_from_yaml

    cfg = load_config_from_yaml(args.config)

    # Bootstrap: auto-detect splits + normalise periods
    from utils.period_utils import bootstrap_config as _bootstrap
    cfg, _src_df = _bootstrap(args.config)
    del _src_df

    logger.info(f"LLM Provider: {cfg.llm_provider}")
    logger.info(f"Input Data Path: {cfg.input_data_path}")
    logger.info(f"Artifact Base Path: {cfg.artifact_base_path}")

    if cfg.llm_provider == "databricks":
        logger.info(f"Databricks Base Path: {cfg.databricks_base_path}")

    # Ensure artifact directories exist
    logger.info("Ensuring artifact directories exist...")
    cfg.ensure_artifact_dirs()
    logger.info(f"Artifact directory ready: {cfg.artifact_base_path}")

    # Validate input data exists
    if not os.path.exists(cfg.input_data_path):
        logger.error(f"Input data not found: {cfg.input_data_path}")
        logger.error("Please ensure the data file exists at the specified path.")
        sys.exit(1)
    logger.info(f"Input data found: {cfg.input_data_path}")

    # Test LLM connection
    logger.info("Testing LLM connection...")
    try:
        from config.llm_config import get_llm
        llm = get_llm(config_path=args.config)
        logger.info(f"LLM initialized successfully: {type(llm).__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)

    # ==========================================================================
    # DIAGNOSTIC: Test LLM with tool calling BEFORE starting any crew
    # For AIFChatLLM (Databricks), the CodeExecutionTool is already registered
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC: Testing LLM with tool calling...")
    logger.info("=" * 60)

    try:
        from config.llm_config import AIFChatLLM

        # Check if this is an AIFChatLLM with registered tools
        if isinstance(llm, AIFChatLLM):
            logger.info(f"[DIAGNOSTIC] AIFChatLLM detected with {len(llm._registered_tools)} registered tool(s)")
            for tool_name in llm._registered_tools.keys():
                logger.info(f"[DIAGNOSTIC]   - {tool_name}")

            # Test the tool by asking the LLM to run code
            test_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the python_code_executor tool to run Python code."
                },
                {
                    "role": "user",
                    "content": "Run this Python code and tell me the result: print('Hello from diagnostic test!')"
                }
            ]

            logger.info("[DIAGNOSTIC] Sending test request...")
            response = llm.call(messages=test_messages)

            logger.info(f"[DIAGNOSTIC] Response length: {len(response) if response else 0} chars")
            logger.info(f"[DIAGNOSTIC] Response (first 500 chars): {response[:500] if response else 'EMPTY/NONE'}")

            if not response:
                logger.error("[DIAGNOSTIC] WARNING: LLM returned empty response!")
            elif "Hello from diagnostic test" in response:
                logger.info("[DIAGNOSTIC] SUCCESS: Tool was called and executed correctly!")
            else:
                logger.warning("[DIAGNOSTIC] Tool may not have been called. Response doesn't contain expected output.")
        else:
            # For Bedrock/other LLMs, just do a simple call
            logger.info(f"[DIAGNOSTIC] Non-AIFChatLLM detected: {type(llm).__name__}")
            response = llm.call("What is 2 + 2? Reply with just the number.")
            logger.info(f"[DIAGNOSTIC] Simple test response: {response[:100] if response else 'EMPTY'}")

    except Exception as e:
        logger.error(f"[DIAGNOSTIC] LLM test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("[DIAGNOSTIC] Continuing anyway to see if crew works...")

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC: LLM test complete")
    logger.info("=" * 60)

    # ==========================================================================
    # CREW OUTPUT VALIDATION - Skip completed crews or delete incomplete outputs
    # ==========================================================================
    from utils.crew_output_validator import (
        prepare_crew_for_run,
        get_crew_status_summary,
        validate_all_crews,
    )

    # Show current status of all crews
    logger.info("\n" + get_crew_status_summary(cfg.artifact_base_path))

    # Helper function to check if a crew should run
    def should_run_crew(crew_name: str) -> bool:
        """Check if a crew should run based on completion status."""
        if args.force_rerun:
            should_run, reason = prepare_crew_for_run(
                cfg.artifact_base_path, crew_name, force_rerun=True
            )
            logger.info(f"[{crew_name.upper()}] Force rerun: {reason}")
            return True

        if args.skip_completed:
            should_run, reason = prepare_crew_for_run(
                cfg.artifact_base_path, crew_name, force_rerun=False
            )
            if should_run:
                logger.info(f"[{crew_name.upper()}] Will run: {reason}")
            else:
                logger.info(f"[{crew_name.upper()}] SKIPPING: {reason}")
            return should_run

        # No skip logic - always run
        return True

    # Run the appropriate stage
    logger.info("=" * 60)
    logger.info(f"Starting pipeline stage: {args.stage}")
    logger.info("=" * 60)

    # =====================================================================
    # --no-llm MODE: Deterministic pipeline without any LLM agents
    # Calls the EXACT same functions the CrewAI agents call, directly.
    # =====================================================================
    if args.no_llm and args.stage == "full":
        # Delegates to the shared deterministic pipeline. Historically this
        # block was inline — it has been extracted to utils.deterministic_pipeline
        # so databricks_runner_uk.py can share it.
        from utils.deterministic_pipeline import run_full_deterministic_pipeline
        run_full_deterministic_pipeline(cfg, clean_artifacts=True)
        # Fall through — the dispatch cascade below is skipped because it
        # is wrapped in the elif chain, and the ``if`` block has consumed
        # args.stage == "full" when combined with --no-llm.
        pass  # sentinel: the legacy inline pipeline lived here

    elif args.stage == "full":
        # Run full pipeline with skip logic for each stage
        logger.info("Running full pipeline with skip/retry logic...")

        # Stage 1: EDA
        if should_run_crew("eda"):
            logger.info("=" * 60)
            logger.info("STAGE 1: EDA")
            logger.info("=" * 60)
            from run_eda import run_eda_only
            run_eda_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("STAGE 1: EDA - SKIPPED (already complete)")

        # Stage 2: Feature Availability Detection
        logger.info("=" * 60)
        logger.info("STAGE 2: FEATURE AVAILABILITY DETECTION")
        logger.info("=" * 60)
        from run_feature_availability import run_feature_availability_only
        run_feature_availability_only(args.config, enable_trace=not args.no_trace)

        # Stage 3: Segmentation
        if should_run_crew("segmentation"):
            logger.info("=" * 60)
            logger.info("STAGE 3: SEGMENTATION")
            logger.info("=" * 60)
            try:
                from run_segmentation import run_segmentation_only
                run_segmentation_only(args.config, enable_trace=not args.no_trace)
            except Exception as e:
                logger.error(f"STAGE 3 FAILED: {e}")
                logger.error("PIPELINE STOPPED — segmentation is required for all downstream stages")
                sys.exit(1)
        else:
            logger.info("STAGE 3: SEGMENTATION - SKIPPED (already complete)")

        # Stage 4: Feature Engineering
        if should_run_crew("feature"):
            logger.info("=" * 60)
            logger.info("STAGE 4: FEATURE ENGINEERING")
            logger.info("=" * 60)
            try:
                from run_feature import run_feature_only
                run_feature_only(args.config, enable_trace=not args.no_trace)
            except Exception as e:
                logger.error(f"STAGE 4 FAILED: {e}")
                logger.error("PIPELINE STOPPED — features are required for training")
                sys.exit(1)
        else:
            logger.info("STAGE 4: FEATURE ENGINEERING - SKIPPED (already complete)")

        # Stage 5: Training
        if should_run_crew("training"):
            logger.info("=" * 60)
            logger.info("STAGE 5: TRAINING")
            logger.info("=" * 60)
            try:
                from run_training import run_training_only
                run_training_only(args.config, enable_trace=not args.no_trace)
            except Exception as e:
                logger.error(f"STAGE 5 FAILED: {e}")
                logger.error("PIPELINE STOPPED — trained models are required for inference/backtesting")
                sys.exit(1)
        else:
            logger.info("STAGE 5: TRAINING - SKIPPED (already complete)")

        # Stage 6: Inference (if run_mode requires it)
        if cfg.should_forward_forecast:
            if should_run_crew("inference"):
                logger.info("=" * 60)
                logger.info("STAGE 6: INFERENCE")
                logger.info("=" * 60)
                try:
                    from run_inference import run_inference_only
                    run_inference_only(args.config)
                except Exception as e:
                    logger.error(f"STAGE 6 FAILED: {e}")
                    logger.warning("Inference failed but continuing to backtesting...")
            else:
                logger.info("STAGE 6: INFERENCE - SKIPPED (already complete)")

            # Stage 7: Diagnostics
            if should_run_crew("diagnostic"):
                logger.info("=" * 60)
                logger.info("STAGE 7: DIAGNOSTICS")
                logger.info("=" * 60)
                try:
                    from run_diagnostic import run_diagnostic_only
                    run_diagnostic_only(args.config, enable_trace=not args.no_trace)
                except Exception as e:
                    logger.error(f"STAGE 7 FAILED: {e}")
                    logger.warning("Diagnostics failed but continuing...")
            else:
                logger.info("STAGE 7: DIAGNOSTICS - SKIPPED (already complete)")
        else:
            logger.info("STAGE 6-7: Inference & Diagnostics - SKIPPED (run_mode)")

        # Stage 8: Backtesting (if run_mode requires it)
        if cfg.should_backtest:
            if should_run_crew("backtest"):
                logger.info("=" * 60)
                logger.info("STAGE 8: BACKTESTING")
                logger.info("=" * 60)
                try:
                    from run_backtesting import run_backtesting_only
                    run_backtesting_only(args.config, enable_trace=not args.no_trace)
                except Exception as e:
                    logger.error(f"STAGE 8 FAILED: {e}")
            else:
                logger.info("STAGE 8: BACKTESTING - SKIPPED (already complete)")
        else:
            logger.info("STAGE 8: Backtesting - SKIPPED (run_mode)")

        # Show final status
        logger.info("\n" + get_crew_status_summary(cfg.artifact_base_path))

    elif args.stage == "eda":
        if should_run_crew("eda"):
            from run_eda import run_eda_only
            run_eda_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("EDA stage skipped - output already complete")

    elif args.stage == "feature_availability":
        from run_feature_availability import run_feature_availability_only
        run_feature_availability_only(args.config, enable_trace=not args.no_trace)

    elif args.stage == "segmentation":
        if should_run_crew("segmentation"):
            from run_segmentation import run_segmentation_only
            run_segmentation_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("Segmentation stage skipped - output already complete")

    elif args.stage == "feature":
        if should_run_crew("feature"):
            from run_feature import run_feature_only
            run_feature_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("Feature Engineering stage skipped - output already complete")

    elif args.stage == "training":
        if should_run_crew("training"):
            from run_training import run_training_only
            run_training_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("Training stage skipped - output already complete")

    elif args.stage == "inference":
        from run_inference import run_inference_only
        run_inference_only(args.config)

    elif args.stage == "diagnostic":
        from run_diagnostic import run_diagnostic_only
        run_diagnostic_only(args.config, enable_trace=not args.no_trace)

    elif args.stage == "backtest":
        from run_backtesting import run_backtesting_only
        run_backtesting_only(args.config, enable_trace=not args.no_trace)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
