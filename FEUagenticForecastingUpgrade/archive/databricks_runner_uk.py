#!/usr/bin/env python3
"""
Databricks Entry Point for UK Categories — FEU-Agentic-Forecasting Pipeline

Standalone runner for UK data that adds a ``preprocessing`` stage on top of
the standard 7-stage pipeline.  The preprocessing stage:

  1. Reads a raw parquet snapshot from Databricks Volumes
  2. Filters by category_name
  3. Collapses one-hot indicator columns into categorical strings
  4. Cleans the year_week column (strips hyphens)
  5. Computes train/val/test time splits from --history-till
  6. Saves preprocessed CSV to the config's input_data_path
  7. Updates the config YAML with the computed time periods

All other stages (eda → backtest) behave identically to databricks_runner.py.

Usage
-----
# Step 1: Preprocess raw data
python databricks_runner_uk.py \\
  --config config/config_uk_deo.yaml \\
  --stage preprocessing \\
  --raw-data-path "/Volumes/.../DT/" \\
  --category-name "DEODORANT & FRAGRANCE" \\
  --history-till "202604" \\
  --skip-install

# Step 2: Run full pipeline (uses updated config + preprocessed data)
python databricks_runner_uk.py \\
  --config config/config_uk_deo.yaml \\
  --stage full
"""

import sys
import os

# =============================================================================
# CRITICAL: Disable OpenTelemetry SDK to avoid version conflicts with Databricks
# Must be set BEFORE any imports that might trigger opentelemetry.
# =============================================================================
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import argparse
import logging
import subprocess
import importlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dependency installation (identical to databricks_runner.py)
# ============================================================================

def install_dependencies(project_root: str, force: bool = False) -> bool:
    """Install Python dependencies from requirements.txt."""
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"sys.path[0:5]: {sys.path[:5]}")

    if not force:
        try:
            importlib.invalidate_caches()
            import crewai
            logger.info(
                f"Dependencies already installed (crewai found at {crewai.__file__})"
            )
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

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "-r", requirements_path,
                "--upgrade",
                "--no-warn-script-location",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode == 0:
            logger.info("Dependencies installed successfully")
        else:
            logger.warning(
                f"Some packages may have failed (return code: {result.returncode})"
            )
            if result.stderr:
                logger.warning(f"stderr (last 1000 chars): {result.stderr[-1000:]}")

        # Clear cached failed imports
        modules_to_clear = [
            m for m in list(sys.modules.keys())
            if "crewai" in m or "opentelemetry" in m or "litellm" in m
        ]
        for m in modules_to_clear:
            del sys.modules[m]
        logger.info(f"Cleared {len(modules_to_clear)} cached modules")

        importlib.invalidate_caches()

        # Verify crewai
        try:
            import crewai
            logger.info(
                f"crewai: OK (version: {getattr(crewai, '__version__', 'unknown')})"
            )
        except ImportError as e:
            logger.error(f"crewai import failed: {e}")
            return False

        # Verify litellm
        try:
            import litellm
            logger.info(
                f"litellm: OK (version: {getattr(litellm, '__version__', 'unknown')})"
            )
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


# ============================================================================
# UK Preprocessing utilities
# ============================================================================

# 12 prefix → output column mappings for one-hot collapse
_INDICATOR_PREFIX_MAP: List[tuple] = [
    ("promo_primary_mechanic_",             "promo_primary_mechanic"),
    ("promo_primary_objective_",            "promo_primary_objective"),
    ("promo_secondary_objective_",          "promo_secondary_objective"),
    ("promo_feature_",                      "promo_feature"),
    ("promo_secondary_mechanic_",           "promo_secondary_mechanic"),
    ("promo_instore_primary_mechanic_",     "promo_instore_primary_mechanic"),
    ("promo_instore_primary_objective_",    "promo_instore_primary_objective"),
    ("promo_instore_secondary_objective_",  "promo_instore_secondary_objective"),
    ("promo_instore_feature_",              "promo_instore_feature"),
    ("promo_instore_secondary_mechanic_",   "promo_instore_secondary_mechanic"),
    ("holiday_",                            "holiday"),
    ("season_",                             "season"),
]


def collapse_indicator_columns(
    df: pd.DataFrame,
    indicator_cols: List[str],
    out_col: str,
    prefix: Optional[str] = None,
    sep: str = "_",
    none_value: Optional[str] = None,
) -> pd.DataFrame:
    """Collapse multiple 0/1 indicator columns into a single string column.

    For each row, picks the suffix (mechanic name) of columns where value == 1.
    If multiple are 1, concatenates them with *sep*.
    If none are 1, sets *none_value* (default ``None`` → NaN).

    Returns *df* (same object) with the new column added.
    """
    cols = [c for c in indicator_cols if c in df.columns]
    if not cols:
        logger.warning(
            f"No indicator columns found for prefix '{prefix}' — skipping."
        )
        return df

    if prefix is None:
        labels = cols[:]
    else:
        labels = [
            c[len(prefix):] if c.startswith(prefix) else c for c in cols
        ]

    active = df[cols].fillna(0).astype(int).eq(1)
    label_arr = pd.Index(labels).to_numpy()
    mask_arr = active.to_numpy()

    collapsed = [
        (sep.join(label_arr[row_mask]) if row_mask.any() else none_value)
        for row_mask in mask_arr
    ]
    df[out_col] = collapsed
    return df


def _collapse_all_indicators(df: pd.DataFrame, all_columns: List[str]) -> pd.DataFrame:
    """Run collapse for all 12 prefix groups, then drop the original one-hot cols."""
    cols_to_drop: List[str] = []

    for prefix, out_col in _INDICATOR_PREFIX_MAP:
        indicator_cols = [c for c in all_columns if c.startswith(prefix)]
        if not indicator_cols:
            logger.info(f"  No columns for prefix '{prefix}' — skipping.")
            continue

        logger.info(f"  Collapsing {len(indicator_cols)} cols for '{prefix}' → '{out_col}'")
        df = collapse_indicator_columns(
            df,
            indicator_cols=indicator_cols,
            out_col=out_col,
            prefix=prefix,
            sep="_",
            none_value="NA",
        )
        cols_to_drop.extend(indicator_cols)

    # Drop original one-hot columns
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
        logger.info(f"  Dropped {len(existing_to_drop)} one-hot indicator columns.")

    return df


# ============================================================================
# Preprocessing stage
# ============================================================================

def run_preprocessing(
    config_yaml_path: str,
    raw_data_path: str,
    category_name: str,
    history_till: str,
) -> None:
    """UK-specific preprocessing: filter, collapse indicators, compute splits."""
    from config.schema import load_config_from_yaml
    from utils.backtesting import increment_period, decrement_period

    logger.info("=" * 60)
    logger.info("UK PREPROCESSING STAGE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load config & read raw parquet
    # ------------------------------------------------------------------
    logger.info(f"Loading config from: {config_yaml_path}")
    cfg = load_config_from_yaml(config_yaml_path)

    time_format = cfg.time_format
    forecast_horizon = cfg.forecast_horizon
    timestamp_col = cfg.timestamp_col

    logger.info(f"Time format: {time_format}")
    logger.info(f"Forecast horizon: {forecast_horizon}")
    logger.info(f"Timestamp column: {timestamp_col}")

    logger.info(f"Reading raw parquet from: {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    logger.info(f"Raw data shape: {df.shape}")

    # ------------------------------------------------------------------
    # Step 2: Filter by category
    # ------------------------------------------------------------------
    logger.info(f"Filtering by category_name == '{category_name}'")
    df = df[df["category_name"] == category_name].copy()
    logger.info(f"Filtered data shape: {df.shape}")

    if df.empty:
        raise ValueError(
            f"No data found for category_name='{category_name}'. "
            f"Available categories: {df['category_name'].unique().tolist() if 'category_name' in df.columns else 'column missing'}"
        )

    # ------------------------------------------------------------------
    # Step 3: Collapse one-hot indicator columns
    # ------------------------------------------------------------------
    logger.info("Collapsing one-hot indicator columns...")
    # Capture all column names BEFORE filtering so we find all indicator cols
    all_raw_columns = list(df.columns)
    df = _collapse_all_indicators(df, all_raw_columns)
    logger.info(f"Shape after collapse: {df.shape}")

    # ------------------------------------------------------------------
    # Step 4: Clean year_week / timestamp column
    # ------------------------------------------------------------------
    logger.info(f"Cleaning '{timestamp_col}' — stripping hyphens...")
    df[timestamp_col] = df[timestamp_col].astype(str).str.replace("-", "", regex=False)
    logger.info(
        f"  Period range: {df[timestamp_col].min()} → {df[timestamp_col].max()}"
    )

    # ------------------------------------------------------------------
    # Step 5: Compute time splits from --history-till
    # ------------------------------------------------------------------
    logger.info(f"Computing time splits from history_till={history_till}")

    val_periods = 3 if time_format == "year_month" else 13

    val_end = history_till
    val_start = decrement_period(val_end, val_periods - 1, time_format)
    train_end = decrement_period(val_start, 1, time_format)
    train_start = str(df[timestamp_col].min())
    test_start = increment_period(val_end, 1, time_format)
    test_end = increment_period(test_start, forecast_horizon - 1, time_format)

    logger.info(f"  train: {train_start} → {train_end}")
    logger.info(f"  val:   {val_start} → {val_end}")
    logger.info(f"  test:  {test_start} → {test_end}")

    # ------------------------------------------------------------------
    # Step 5.5: Calendar / holiday feature injection (Phase B2)
    # ------------------------------------------------------------------
    # Persist the synthesised calendar columns into the CSV so every
    # downstream stage (EDA, segmentation, feature engineering, training,
    # inference) sees the same canonical holiday values — not just the
    # feature-availability crew.
    _cal_cfg = getattr(cfg.design, "calendar_features", None)
    if _cal_cfg is not None and getattr(_cal_cfg, "enabled", False) and cfg.country:
        try:
            from utils.calendar_features import inject_calendar_features

            _period_totals_pre = df.groupby(timestamp_col)[cfg.target_col].sum()
            _hc_pre = str(
                _period_totals_pre[_period_totals_pre > 0].index.max()
            ) if (_period_totals_pre > 0).any() else None

            logger.info(
                "Injecting calendar features (country=%s, subdivision=%s, mode=%s)",
                cfg.country, cfg.country_subdivision,
                getattr(_cal_cfg, "overwrite_mode", "always"),
            )
            n_cols_before = df.shape[1]
            df = inject_calendar_features(
                df,
                date_col=timestamp_col,
                time_format=cfg.time_format,
                country=cfg.country,
                subdivision=cfg.country_subdivision,
                overwrite_mode=getattr(_cal_cfg, "overwrite_mode", "always"),
                history_cutoff=_hc_pre,
                custom_events=getattr(_cal_cfg, "custom_events", None) or None,
                lead_lag_windows=list(
                    getattr(_cal_cfg, "include_lead_lag_windows", [1, 2, 4])
                ),
            )
            logger.info(
                "Calendar features injected: %d → %d cols (+%d new)",
                n_cols_before, df.shape[1], df.shape[1] - n_cols_before,
            )
        except Exception as _cal_exc:
            logger.warning(
                "Calendar injection skipped during UK preprocessing: %s "
                "— downstream stages will fall back to in-memory synthesis.",
                _cal_exc,
            )
    else:
        if not cfg.country:
            logger.info(
                "Calendar injection skipped: config.country is not set"
            )

    # ------------------------------------------------------------------
    # Step 6: Save preprocessed data as CSV
    # ------------------------------------------------------------------
    output_path = cfg.input_data_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to: {output_path}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Step 7: Update config YAML with computed time periods
    # ------------------------------------------------------------------
    logger.info(f"Updating config YAML: {config_yaml_path}")
    yaml_path = Path(config_yaml_path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f)

    raw_yaml["train_start"] = train_start
    raw_yaml["train_end"] = train_end
    raw_yaml["val_start"] = val_start
    raw_yaml["val_end"] = val_end
    raw_yaml["test_start"] = test_start
    raw_yaml["test_end"] = test_end

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(raw_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Config YAML updated with new time splits.")
    logger.info("=" * 60)
    logger.info("UK PREPROCESSING COMPLETE")
    logger.info("=" * 60)


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Databricks entry point for UK categories — FEU-Agentic-Forecasting Pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config.yaml file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "preprocessing",
            "full",
            "eda",
            "feature_availability",
            "segmentation",
            "feature",
            "training",
            "inference",
            "diagnostic",
            "backtest",
        ],
        default="full",
        help="Pipeline stage to run (default: full)",
    )
    # UK preprocessing-specific arguments
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Path to raw parquet on Volumes (required for preprocessing)",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default=None,
        help='Category to filter, e.g. "DEODORANT & FRAGRANCE" (required for preprocessing)',
    )
    parser.add_argument(
        "--history-till",
        type=str,
        default=None,
        help="Last period of known data, YYYYWW or YYYYMM (required for preprocessing)",
    )
    # Standard pipeline flags
    parser.add_argument(
        "--no-trace", action="store_true", help="Disable CrewAI trace logging"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun all crews even if outputs already exist",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip crews that have already completed (default: True)",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Disable skipping of completed crews",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run deterministic pipeline WITHOUT any LLM/CrewAI agents. "
             "Calls the same functions the agents call, but directly. "
             "Faster, cheaper, more reliable. Recommended for production.",
    )

    args = parser.parse_args()

    if args.no_skip_completed:
        args.skip_completed = False

    logger.info("=" * 60)
    logger.info("FEU-Agentic-Forecasting: Databricks Runner (UK)")
    logger.info("=" * 60)

    # Validate config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    logger.info(f"Config file: {args.config}")
    logger.info(f"Stage: {args.stage}")

    # ------------------------------------------------------------------
    # Preprocessing stage
    # ------------------------------------------------------------------
    if args.stage == "preprocessing":
        # Validate preprocessing-specific args
        missing = []
        if not args.raw_data_path:
            missing.append("--raw-data-path")
        if not args.category_name:
            missing.append("--category-name")
        if not args.history_till:
            missing.append("--history-till")
        if missing:
            logger.error(
                f"Preprocessing stage requires: {', '.join(missing)}"
            )
            sys.exit(1)

        # Determine project root
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            config_dir = os.path.dirname(os.path.abspath(args.config))
            project_root = os.path.dirname(config_dir)

        # Install dependencies (preprocessing is the first stage on a fresh cluster)
        if not args.skip_install:
            logger.info("Preprocessing: installing dependencies...")
            if not install_dependencies(project_root):
                logger.error("Failed to install dependencies. Exiting.")
                sys.exit(1)

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        run_preprocessing(
            config_yaml_path=args.config,
            raw_data_path=args.raw_data_path,
            category_name=args.category_name,
            history_till=args.history_till,
        )
        return

    # ------------------------------------------------------------------
    # All other stages — same logic as databricks_runner.py
    # ------------------------------------------------------------------

    # Determine project root
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        project_root = os.path.dirname(config_dir)
        logger.info("Running in Databricks notebook context (__file__ not defined)")

    logger.info(f"Project root: {project_root}")

    # Install dependencies for entry-point stages
    if not args.skip_install and args.stage in ["eda", "inference", "full"]:
        logger.info(f"Stage '{args.stage}' requires dependency installation check")
        if not install_dependencies(project_root):
            logger.error("Failed to install dependencies. Exiting.")
            sys.exit(1)
    elif args.skip_install:
        logger.info("Skipping dependency installation (--skip-install flag)")
    else:
        logger.info(
            f"Stage '{args.stage}' assumes dependencies installed by prior EDA run"
        )
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

    # ------------------------------------------------------------------
    # --no-llm SHORT-CIRCUIT: run the deterministic pipeline and exit.
    # Must run BEFORE the LLM connection test so environments without
    # any LLM credentials can still execute the deterministic mode.
    # ------------------------------------------------------------------
    if args.no_llm and args.stage == "full":
        logger.info("=" * 60)
        logger.info("DETERMINISTIC MODE (--no-llm): No LLM agents, same functions")
        logger.info("=" * 60)
        from utils.deterministic_pipeline import run_full_deterministic_pipeline
        run_full_deterministic_pipeline(cfg, clean_artifacts=True)
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully (deterministic).")
        logger.info("=" * 60)
        return

    # Test LLM connection
    logger.info("Testing LLM connection...")
    try:
        from config.llm_config import get_llm
        llm = get_llm(config_path=args.config)
        logger.info(f"LLM initialized successfully: {type(llm).__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)

    # LLM diagnostic test
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC: Testing LLM with tool calling...")
    logger.info("=" * 60)

    try:
        from config.llm_config import AIFChatLLM

        if isinstance(llm, AIFChatLLM):
            logger.info(
                f"[DIAGNOSTIC] AIFChatLLM detected with "
                f"{len(llm._registered_tools)} registered tool(s)"
            )
            for tool_name in llm._registered_tools.keys():
                logger.info(f"[DIAGNOSTIC]   - {tool_name}")

            test_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the python_code_executor tool to run Python code.",
                },
                {
                    "role": "user",
                    "content": "Run this Python code and tell me the result: print('Hello from diagnostic test!')",
                },
            ]
            response = llm.call(messages=test_messages)
            logger.info(
                f"[DIAGNOSTIC] Response length: {len(response) if response else 0} chars"
            )
            if response and "Hello from diagnostic test" in response:
                logger.info("[DIAGNOSTIC] SUCCESS: Tool was called and executed correctly!")
            else:
                logger.warning("[DIAGNOSTIC] Tool may not have been called.")
        else:
            logger.info(f"[DIAGNOSTIC] Non-AIFChatLLM detected: {type(llm).__name__}")
            response = llm.call("What is 2 + 2? Reply with just the number.")
            logger.info(
                f"[DIAGNOSTIC] Simple test response: {response[:100] if response else 'EMPTY'}"
            )
    except Exception as e:
        logger.error(f"[DIAGNOSTIC] LLM test failed: {e}")
        logger.error("[DIAGNOSTIC] Continuing anyway...")

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC: LLM test complete")
    logger.info("=" * 60)

    # Crew output validation
    from utils.crew_output_validator import (
        prepare_crew_for_run,
        get_crew_status_summary,
    )

    logger.info("\n" + get_crew_status_summary(cfg.artifact_base_path))

    def should_run_crew(crew_name: str) -> bool:
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
        return True

    # ---------------------------------------------------------------
    # Run the appropriate stage
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Starting pipeline stage: {args.stage}")
    logger.info("=" * 60)

    if args.stage == "full":
        # Stage 1: EDA
        if should_run_crew("eda"):
            logger.info("=" * 60)
            logger.info("STAGE 1: EDA")
            logger.info("=" * 60)
            from run_eda import run_eda_only
            run_eda_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("STAGE 1: EDA - SKIPPED (already complete)")

        # Stage 1.5: Feature Availability Detection (Phase A classifier)
        # This MUST run before segmentation / feature engineering so the
        # feature_availability_to_feature_context.json artifact exists and
        # downstream crews only use features that are actually known in
        # the forecast horizon (plus frozen embeddings for history-only).
        if should_run_crew("feature_availability"):
            logger.info("=" * 60)
            logger.info("STAGE 1.5: FEATURE AVAILABILITY")
            logger.info("=" * 60)
            from run_feature_availability import run_feature_availability_only
            run_feature_availability_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info(
                "STAGE 1.5: FEATURE AVAILABILITY - SKIPPED (already complete)"
            )

        # Stage 2: Segmentation
        if should_run_crew("segmentation"):
            logger.info("=" * 60)
            logger.info("STAGE 2: SEGMENTATION")
            logger.info("=" * 60)
            from run_segmentation import run_segmentation_only
            run_segmentation_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("STAGE 2: SEGMENTATION - SKIPPED (already complete)")

        # Stage 3: Feature Engineering
        if should_run_crew("feature"):
            logger.info("=" * 60)
            logger.info("STAGE 3: FEATURE ENGINEERING")
            logger.info("=" * 60)
            from run_feature import run_feature_only
            run_feature_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("STAGE 3: FEATURE ENGINEERING - SKIPPED (already complete)")

        # Stage 4: Training
        if should_run_crew("training"):
            logger.info("=" * 60)
            logger.info("STAGE 4: TRAINING")
            logger.info("=" * 60)
            from run_training import run_training_only
            run_training_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("STAGE 4: TRAINING - SKIPPED (already complete)")

        logger.info("\n" + get_crew_status_summary(cfg.artifact_base_path))

    elif args.stage == "eda":
        if should_run_crew("eda"):
            from run_eda import run_eda_only
            run_eda_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info("EDA stage skipped - output already complete")

    elif args.stage == "feature_availability":
        if should_run_crew("feature_availability"):
            from run_feature_availability import run_feature_availability_only
            run_feature_availability_only(args.config, enable_trace=not args.no_trace)
        else:
            logger.info(
                "Feature availability stage skipped - output already complete"
            )

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
