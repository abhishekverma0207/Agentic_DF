# crews/training_crew.py
"""
STATE-OF-THE-ART Training Crew for Demand Forecasting

CRITICAL: FEATURE-BASED MODELS ONLY
This crew uses ONLY feature-based models that leverage the engineered features.
ALLOWED: lightgbm, xgboost, catboost, random_forest, zero_inflated, hurdle_model, tweedie
BANNED: croston, sba, tsb, imapa, arima, sarima, ets, theta, prophet, tbats, bsts

STATE-OF-THE-ART FEATURES:
==========================
This crew leverages ALL exhaustive EDA insights from upstream crews:

1. EDA-DRIVEN LOSS FUNCTIONS:
   - smooth → 'mse' (standard MSE works well)
   - erratic → 'huber' (robust to outliers)
   - intermittent → 'tweedie' (handles zeros properly)
   - lumpy → 'tweedie' (handles zeros + high variance)

2. VALIDATION STRATEGIES:
   - smooth/erratic with seasonality → 'seasonal_split'
   - smooth/erratic without seasonality → 'time_series_split'
   - intermittent/lumpy → 'holdout' (sparse data needs simple split)

3. ACF-INFORMED HYPERPARAMETERS:
   - Learning rates adjusted by expected difficulty
   - Early stopping rounds based on demand pattern complexity
   - Lag configurations from ACF/PACF analysis

3-AGENT PATTERN:
================
1. Training Planner: Reads ALL enriched context (eda_insights_for_training,
   per_segment_training_strategy, adaptive_config) and creates training_strategy.json
   with per-segment loss functions and validation strategies.

2. Training Executor: Uses run_full_training_pipeline with ALL context including
   per-segment loss functions, validation strategies, and EDA-driven hyperparameters.

3. Training Analyst: ENRICHES diagnostic context with EDA insights utilization,
   per-segment analysis vs expectations, and actionable recommendations.

CONTEXT FILES USED:
==================
- feature_to_training_context.json:
  * feature_summary with adaptive_config (ACF-informed lags, seasonal period)
  * eda_insights_for_training (seasonality, trend, changepoint recommendations)
  * per_segment_training_strategy (loss function, validation strategy, difficulty)
- segmentation_to_training_context.json:
  * segment_model_strategy (demand patterns, model recommendations)

PRIMARY OBJECTIVE: MINIMIZE WAPE (Weighted Absolute Percentage Error)
WAPE = sum(|actual - pred|) / sum(actual)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List

from crewai import Agent, Crew, Task, Process, LLM

from config.schema import DemandForecastConfig
from utils.code_execution_tool import CodeExecutionTool

logger = logging.getLogger(__name__)


class TrainingFailedError(Exception):
    """Raised when model training fails to produce required outputs."""
    pass


def _validate_planner_output(model_dir: str) -> None:
    """
    Validate that Planner task created training_strategy.json.

    This is called as a task callback after Planner completes.
    Raises TrainingFailedError if the file doesn't exist.

    ROOT CAUSE this addresses:
    - CrewAI agents sometimes describe code instead of executing it
    - The LLM may say "I'll create training_strategy.json" without actually running CodeExecutionTool
    - This validation catches that failure immediately instead of letting Executor fail
    """
    strategy_path = os.path.join(model_dir, 'training_strategy.json')
    if not os.path.exists(strategy_path):
        raise TrainingFailedError(
            f"CRITICAL: Planner task FAILED to create training_strategy.json\n\n"
            f"Expected file: {strategy_path}\n\n"
            "ROOT CAUSE: The LLM agent did NOT execute the code block.\n"
            "The agent likely described what should happen instead of running CodeExecutionTool.\n\n"
            "SOLUTION: Check the Planner task output in training_planning_report.md.\n"
            "If it contains description instead of execution output, this confirms the issue.\n\n"
            "This is a known issue with CrewAI agents. Re-running the pipeline may help,\n"
            "or try adjusting the LLM temperature/model settings."
        )
    logger.info(f"Planner validation passed: {strategy_path} exists")


@dataclass
class TrainingCrewResult:
    """Container with the main model training & selection artifact paths."""
    final_model_specs_path: str
    model_selection_report_markdown_path: str
    model_dir: str
    # Generated pipeline
    training_pipeline_script_path: str = ""
    # Focused context output
    training_to_diagnostic_context_path: str = ""
    # DETERMINISTIC CODE OUTPUT
    training_deterministic_code_path: str = ""
    # Cost tracking
    cost_report_path: str = ""


def _validate_training_completed(model_dir: str) -> None:
    """
    Validate that training actually completed and produced required outputs.

    Raises TrainingFailedError if validation fails.

    Checks:
    1. final_model_specs.json exists and contains actual model data
    2. training_to_diagnostic_context.json shows models_trained > 0
    3. At least one .pkl model file exists
    """
    import glob

    errors = []

    # Check 1: final_model_specs.json
    specs_path = os.path.join(model_dir, "final_model_specs.json")
    if not os.path.exists(specs_path):
        errors.append(f"Missing: {specs_path}")
    else:
        try:
            with open(specs_path, 'r') as f:
                specs = json.load(f)
            # Check if it contains actual model data (not just a stub)
            if not specs.get("models") and not specs.get("model_groups"):
                errors.append("final_model_specs.json is empty or missing 'models' key")
            # Check if overall_val_wape is None (indicates no training)
            # Note: model_training.py saves validation WAPE as 'overall_val_wape'
            if specs.get("overall_val_wape") is None and specs.get("overall_wape") is None:
                errors.append("final_model_specs.json has null overall_val_wape - training may not have run")
        except json.JSONDecodeError:
            errors.append("final_model_specs.json is not valid JSON")
        except Exception as e:
            errors.append(f"Error reading final_model_specs.json: {e}")

    # Check 2: training_to_diagnostic_context.json
    context_path = os.path.join(model_dir, "training_to_diagnostic_context.json")
    if os.path.exists(context_path):
        try:
            with open(context_path, 'r') as f:
                context = json.load(f)
            overall_perf = context.get("overall_performance", {})
            models_trained = overall_perf.get("models_trained", 0)
            training_status = overall_perf.get("training_status", "unknown")

            if models_trained == 0:
                errors.append(
                    f"training_to_diagnostic_context.json shows models_trained=0, "
                    f"training_status='{training_status}' - NO MODELS WERE TRAINED!"
                )
            elif training_status == "pending":
                errors.append(
                    f"training_to_diagnostic_context.json shows training_status='pending' - "
                    f"training was never executed!"
                )
        except Exception as e:
            logger.warning(f"Could not validate training_to_diagnostic_context.json: {e}")

    # Check 3: .pkl model files exist
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    models_exist = len(pkl_files) > 0
    if not models_exist:
        errors.append(f"No .pkl model files found in {model_dir} - NO MODELS WERE SAVED!")
    else:
        logger.info(f"Found {len(pkl_files)} model files in {model_dir}")

    # Check 4: training_execution_report.md has actual content
    # NOTE: This is a WARNING only, not an error. The execution report is generated by the LLM
    # and may fail due to content filtering or token limits, but the actual model training
    # (which is done by Python code) may still have succeeded.
    warnings = []
    exec_report_path = os.path.join(model_dir, "training_execution_report.md")
    if os.path.exists(exec_report_path):
        with open(exec_report_path, 'r') as f:
            content = f.read().strip()
        # Check for empty/error content
        if not content or len(content) < 100:
            warnings.append(f"training_execution_report.md is nearly empty ({len(content)} chars)")
        if "I apologize" in content or "empty response" in content.lower():
            warnings.append(
                "training_execution_report.md contains LLM error message - "
                "the LLM may have failed to generate the report"
            )

    # Log warnings but don't fail if models were actually trained
    if warnings:
        for w in warnings:
            logger.warning(f"Training report issue (non-fatal): {w}")

        # If models exist and specs are valid, these warnings are informational only
        if models_exist and not errors:
            logger.info(
                "Training completed successfully despite report generation issues. "
                "Model files and specs are valid."
            )

    # Raise error only for critical failures (missing models, invalid specs)
    if errors:
        error_msg = (
            "TRAINING VALIDATION FAILED - Model training did not complete successfully!\n\n"
            "Issues found:\n" +
            "\n".join(f"  - {e}" for e in errors) +
            "\n\nThis is likely due to:\n"
            "  1. AWS Bedrock content filtering blocked the LLM response\n"
            "  2. The LLM returned an empty response\n"
            "  3. The training code in the task description was not executed\n\n"
            "The pipeline cannot continue without trained models."
        )
        logger.error(error_msg)
        raise TrainingFailedError(error_msg)


def _get_output_path(absolute_path: str) -> str:
    """
    Get a safe path for CrewAI Task output_file parameter.

    CrewAI 1.9.1+ rejects paths with '..' (path traversal) for security.
    On Databricks, relative paths from cwd to /Volumes/ create '../..' paths.

    Solution: Use absolute paths which work for both Bedrock and Databricks.
    """
    return os.path.abspath(absolute_path)


def _create_training_planner_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Training Planner agent - reads ALL enriched context and creates
    STATE-OF-THE-ART training strategy with EDA-driven configuration.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="training_planner",
        role="Training Strategy Code Executor",
        goal=(
            "EXECUTE the exact Python code provided in the task description using CodeExecutionTool. "
            "DO NOT describe what the code does. DO NOT write your own code. "
            "JUST RUN the provided code block to create training_strategy.json."
        ),
        backstory=(
            "######################################################################\n"
            "#  CRITICAL: YOU MUST EXECUTE CODE USING CodeExecutionTool          #\n"
            "#  DO NOT DESCRIBE THE CODE OR OUTPUT - ACTUALLY RUN IT!            #\n"
            "#  THE TASK CONTAINS A COMPLETE PYTHON CODE BLOCK - RUN IT EXACTLY  #\n"
            "######################################################################\n\n"
            "## YOUR ONLY JOB:\n"
            "1. Find the ```python ... ``` code block in the task description\n"
            "2. Copy that EXACT code into CodeExecutionTool\n"
            "3. Run it - do NOT modify, simplify, or rewrite the code\n"
            "4. The code will create training_strategy.json\n\n"
            "## WHAT YOU MUST NOT DO:\n"
            "- Do NOT describe what the code should do\n"
            "- Do NOT write a JSON response describing the strategy\n"
            "- Do NOT skip the CodeExecutionTool call\n"
            "- Do NOT modify or rewrite the provided code\n\n"
            "## CORRECT BEHAVIOR:\n"
            "1. See task with Python code block\n"
            "2. Use CodeExecutionTool to execute that code\n"
            "3. Code creates training_strategy.json file\n"
            "4. Report that the file was created\n\n"
            "## INCORRECT BEHAVIOR (DO NOT DO THIS):\n"
            "- Responding with JSON describing the strategy\n"
            "- Describing what the strategy should look like\n"
            "- Not calling CodeExecutionTool at all\n\n"
            "## OUTPUT LIMIT: MAX 5 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_training_executor_agent(llm: LLM, allowed_model_families: List[str], protected_paths: list = None) -> Agent:
    """
    Create the Training Executor agent - orchestrates STATE-OF-THE-ART training
    with per-segment loss functions and validation strategies.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="training_executor",
        role="Training Pipeline Code Executor",
        goal=(
            "EXECUTE the exact Python code provided in the task description using CodeExecutionTool. "
            "DO NOT describe what the code does. DO NOT write your own code. "
            "JUST RUN the provided code block to train models."
        ),
        backstory=(
            "######################################################################\n"
            "#  CRITICAL: YOU MUST EXECUTE CODE USING CodeExecutionTool          #\n"
            "#  DO NOT DESCRIBE THE CODE OR OUTPUT - ACTUALLY RUN IT!            #\n"
            "#  THE TASK CONTAINS A COMPLETE PYTHON CODE BLOCK - RUN IT EXACTLY  #\n"
            "######################################################################\n\n"
            "## YOUR ONLY JOB:\n"
            "1. Find the ```python ... ``` code block in the task description\n"
            "2. Copy that EXACT code into CodeExecutionTool\n"
            "3. Run it - do NOT modify, simplify, or rewrite the code\n"
            "4. The code will train models and save them to files\n\n"
            "## WHAT YOU MUST NOT DO:\n"
            "- Do NOT describe what the code should do\n"
            "- Do NOT write a response describing what would happen\n"
            "- Do NOT skip the CodeExecutionTool call\n"
            "- Do NOT modify or rewrite the provided code\n\n"
            "## CORRECT BEHAVIOR:\n"
            "1. See task with Python code block\n"
            "2. Use CodeExecutionTool to execute that code\n"
            "3. Code runs training pipeline\n"
            "4. Report the training results\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n\n"
            "## OUTPUT LIMIT: MAX 10 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_training_analyst_agent(llm: LLM, allowed_model_families: List[str], protected_paths: list = None) -> Agent:
    """
    Create the Training Analyst agent - analyzes results and creates ENRICHED
    diagnostic context with EDA insights and actionable recommendations.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="training_analyst",
        role="Training Analysis Code Executor",
        goal=(
            "EXECUTE the exact Python code provided in the task description using CodeExecutionTool. "
            "DO NOT describe what the code does. DO NOT write your own code. "
            "JUST RUN the provided code block to analyze training results."
        ),
        backstory=(
            "######################################################################\n"
            "#  CRITICAL: YOU MUST EXECUTE CODE USING CodeExecutionTool          #\n"
            "#  DO NOT DESCRIBE THE CODE OR OUTPUT - ACTUALLY RUN IT!            #\n"
            "#  THE TASK CONTAINS A COMPLETE PYTHON CODE BLOCK - RUN IT EXACTLY  #\n"
            "######################################################################\n\n"
            "## YOUR ONLY JOB:\n"
            "1. Find the ```python ... ``` code block in the task description\n"
            "2. Copy that EXACT code into CodeExecutionTool\n"
            "3. Run it - do NOT modify, simplify, or rewrite the code\n"
            "4. The code will analyze results and save diagnostic context\n\n"
            "## WHAT YOU MUST NOT DO:\n"
            "- Do NOT describe what the code should do\n"
            "- Do NOT write a response describing the analysis\n"
            "- Do NOT skip the CodeExecutionTool call\n"
            "- Do NOT modify or rewrite the provided code\n\n"
            "## CORRECT BEHAVIOR:\n"
            "1. See task with Python code block\n"
            "2. Use CodeExecutionTool to execute that code\n"
            "3. Code enriches diagnostic context\n"
            "4. Report that the file was updated\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n\n"
            "## OUTPUT LIMIT: MAX 5 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_training_reviewer_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Training Reviewer agent that validates model training outputs.
    This agent is OPTIONAL and only used when config.design.enable_reviewer is True.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="training_reviewer",
        role="Model Training Quality Assurance & Validation Specialist",
        goal=(
            "Review and validate all model training outputs. Ensure models were trained "
            "with appropriate algorithms for each demand pattern and WAPE metrics are reasonable."
        ),
        backstory=(
            "You are a senior ML engineer specializing in model validation.\n\n"
            "######################################################################\n"
            "#  OUTPUT LIMIT: MAX 10 PRINT STATEMENTS                            #\n"
            "######################################################################\n"
            "```python\n"
            "# Validation pattern:\n"
            "files = ['file1.pkl', 'file2.json']\n"
            "exists = [f for f in files if os.path.exists(f)]\n"
            "print(f'Files: {len(exists)}/{len(files)} present')\n"
            "# Save report, print path only\n"
            "```\n"
            "######################################################################\n\n"
            "## YOUR MISSION\n"
            "1. Validate model files exist for all model_groups\n"
            "2. Verify WAPE metrics are reasonable (<100%)\n"
            "3. Check model selection matches demand patterns\n"
            "4. Create training_review_report.json with quality score (1-10)"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_training_documentation_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Training Documentation Agent that generates comprehensive markdown documentation.

    This agent uses CodeExecutionTool to ITERATIVELY explore training outputs
    and creates a comprehensive insights markdown report through multiple phases.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="training_documentation_agent",
        role="PhD-Level Model Training Insights Documentation Specialist",
        goal=(
            "EXECUTE Python code using CodeExecutionTool to explore training outputs "
            "and create MODEL_TRAINING_INSIGHTS_GUIDE.md. You MUST use the tool to run code - "
            "do NOT just describe or print the analysis."
        ),
        backstory=(
            "You are a PhD-level expert in machine learning model training and evaluation.\n\n"
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE CodeExecutionTool TO RUN ANALYSIS CODE    #\n"
            "#  EXPLORE DATA ITERATIVELY - RUN MULTIPLE CODE EXECUTIONS          #\n"
            "#  SAVE FINAL REPORT TO FILE - DO NOT JUST PRINT IT                 #\n"
            "######################################################################\n\n"
            "## YOUR EXPERTISE:\n"
            "- Demand forecasting model selection (LightGBM, XGBoost, CatBoost, Zero-Inflated)\n"
            "- Loss function selection (MSE, Huber, Tweedie) for different demand patterns\n"
            "- Ensemble methods and model combination strategies\n"
            "- WAPE interpretation and model performance analysis\n\n"
            "## HOW TO COMPLETE THIS TASK:\n"
            "1. Use CodeExecutionTool to run exploration code\n"
            "2. Run MULTIPLE code executions to build understanding\n"
            "3. Extract specific statistics (WAPE, model types, counts)\n"
            "4. Final execution: SAVE markdown report to file\n"
            "5. Print only confirmation messages (max 10 lines per execution)\n\n"
            "## OUTPUT LIMIT: MAX 10 PRINT STATEMENTS PER CODE EXECUTION\n"
            "## SUPPRESS WARNINGS: warnings.filterwarnings('ignore')"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def create_training_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> Crew:
    """
    Create the Training Crew with INTELLIGENT UTILITY ORCHESTRATION.

    This crew reads upstream context and orchestrates individual training utilities
    based on demand patterns from segmentation.

    The crew has three phases:
    1. PLANNING: Read upstream context, create training_strategy.json with per-segment model selections
    2. EXECUTION: Orchestrate individual training utilities (train_lightgbm, train_croston, etc.)
    3. ANALYSIS: Create intelligent diagnostic context for downstream crews

    PRIMARY OBJECTIVE: MINIMIZE WAPE (Weighted Absolute Percentage Error)
    """
    artifact_base = config.artifact_base_path
    eda_dir = os.path.join(artifact_base, "eda_output")
    feat_dir = os.path.join(artifact_base, "feature_output")
    seg_dir = os.path.join(artifact_base, "seg_output")
    model_dir = os.path.join(artifact_base, "model_artifacts")

    os.makedirs(model_dir, exist_ok=True)

    # Get safe output paths for CrewAI Task output_file
    # Note: CrewAI 1.9.1+ rejects relative paths with '..' for security
    model_dir_out = _get_output_path(model_dir)
    feat_dir_out = _get_output_path(feat_dir)
    seg_dir_out = _get_output_path(seg_dir)

    # Get config details
    target_col = config.target_column
    date_col = config.date_column

    # Get train/val/test date ranges from config
    train_start = config.train_start
    train_end = config.train_end
    val_start = config.val_start
    val_end = config.val_end
    test_start = config.test_start
    test_end = config.test_end

    # Get allowed model families from config.design (CRITICAL).
    # The resolver makes ``design.enable_hierarchical_models`` actually
    # extend the list with Phase-3 hierarchical families — previously that
    # flag was declared in the schema but ignored everywhere.
    from utils.model_selection_intelligence import resolve_effective_model_families
    allowed_model_families = resolve_effective_model_families(config)
    enable_deep_models = config.design.enable_deep_models

    # Filter out deep learning models if disabled
    deep_model_types = ['tft', 'lstm', 'nbeats', 'deepar', 'wavenet']
    if not enable_deep_models:
        allowed_model_families = [m for m in allowed_model_families if m.lower() not in deep_model_types]

    # Get HPO configuration from config.design
    max_model_candidates = getattr(config.design, 'max_model_candidates_per_group', 3)
    max_hparam_depth = getattr(config.design, 'max_hparam_search_depth', 10)

    # Get state-of-the-art optimization flags from config.design
    enable_bias_correction = getattr(config.design, 'enable_bias_correction', True)
    enable_ensemble_optimization = getattr(config.design, 'enable_ensemble_optimization', True)
    enable_forecast_calibration = getattr(config.design, 'enable_forecast_calibration', True)
    meta_learning_enabled = getattr(config.design, 'meta_learning_enabled', True)
    prediction_intervals = getattr(config.design, 'prediction_intervals', True)
    prediction_interval_confidence = getattr(config.design, 'prediction_interval_confidence', 0.95)

    # Get bias calibration settings from config.design
    apply_bias_calibration = getattr(config.design, 'apply_bias_calibration', True)
    bias_calibration_buckets = getattr(config.design, 'bias_calibration_buckets', 5)
    bias_calibration_factor_min = getattr(config.design, 'bias_calibration_factor_min', 0.2)
    bias_calibration_factor_max = getattr(config.design, 'bias_calibration_factor_max', 2.0)

    # Time format for period-aware training defaults (walk-forward CV, lags, etc.)
    time_format = config.time_format if config.time_format in ('year_week', 'year_month') else 'year_week'

    # WRITE PROTECTION: Prevent LLM agents from corrupting upstream outputs
    protected_dirs = [eda_dir, seg_dir, feat_dir]

    # Create agents
    planner = _create_training_planner_agent(llm, protected_paths=protected_dirs)
    executor = _create_training_executor_agent(llm, allowed_model_families, protected_paths=protected_dirs)
    analyst = _create_training_analyst_agent(llm, allowed_model_families, protected_paths=protected_dirs)

    # -------------------------------------------------------------------------
    # Task 1: Training Planning - READ ALL ENRICHED CONTEXT, CREATE STRATEGY
    # STATE-OF-THE-ART: Uses EDA insights, per-segment training strategies,
    # ACF-informed lags, loss function recommendations, and validation strategies
    # -------------------------------------------------------------------------
    task_plan = Task(
        name="create_training_strategy",
        description=(
            "# EXECUTE THIS CODE USING CodeExecutionTool\n\n"
            "**CRITICAL: You MUST use CodeExecutionTool to execute the Python code below.**\n"
            "**DO NOT describe what the code does. DO NOT write your own version.**\n"
            "**JUST COPY THE CODE BLOCK INTO CodeExecutionTool AND RUN IT.**\n\n"
            "The code creates training_strategy.json from enriched context.\n\n"
            "## PYTHON CODE TO EXECUTE:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n"
            "import pandas as pd\n"
            "from utils.agent_utilities import load_json, save_json, load_csv\n"
            "from utils.context_schema import ContextReader, SemanticTypes\n\n"
            f"feat_dir = '{feat_dir}'\n"
            f"seg_dir = '{seg_dir}'\n"
            f"model_dir = '{model_dir}'\n"
            f"allowed_families = {allowed_model_families}\n\n"
            "# =================================================================\n"
            "# READ ALL ENRICHED CONTEXT (STATE-OF-THE-ART)\n"
            "# =================================================================\n"
            "feat_ctx_path = os.path.join(feat_dir, 'feature_to_training_context.json')\n"
            "seg_ctx_path = os.path.join(seg_dir, 'segmentation_to_training_context.json')\n\n"
            "# Initialize enriched context containers\n"
            "feat_summary = {}\n"
            "adaptive_config = {}  # ACF-informed lags, seasonal period, fourier order\n"
            "eda_insights = {}     # Seasonality, trend, changepoint recommendations\n"
            "per_segment_training = {}  # Loss function, validation strategy per segment\n"
            "n_features = 0\n"
            "feature_categories = {}\n\n"
            "# Load ENRICHED feature context (STATE-OF-THE-ART)\n"
            "if os.path.exists(feat_ctx_path):\n"
            "    feat_reader = ContextReader(feat_ctx_path)\n"
            "    \n"
            "    # 1. Feature summary with ADAPTIVE config\n"
            "    feat_summary = feat_reader.get_by_type(SemanticTypes.FEATURE_SUMMARY) or feat_reader.get('feature_summary', {})\n"
            "    n_features = feat_summary.get('total_features', 0)\n"
            "    feature_categories = feat_summary.get('feature_categories', {})\n"
            "    adaptive_config = feat_summary.get('adaptive_config', {})\n"
            "    \n"
            "    # 2. EDA insights for training (CRITICAL for state-of-the-art)\n"
            "    eda_insights = feat_reader.get('eda_insights_for_training', {})\n"
            "    \n"
            "    # 3. Per-segment training strategy (loss, validation, difficulty)\n"
            "    per_segment_training = feat_reader.get_by_type(SemanticTypes.SEGMENT_TRAINING_STRATEGY) or feat_reader.get('per_segment_training_strategy', {})\n"
            "    \n"
            "    print(f'Loaded ENRICHED context: {n_features} features, ACF-informed={adaptive_config.get(\"acf_informed_lags\", False)}')\n"
            "    print(f'EDA insights: seasonality={eda_insights.get(\"seasonality\", {}).get(\"recommendation\", \"N/A\")}')\n"
            "    print(f'Per-segment strategies: {len(per_segment_training)} segments')\n\n"
            "# Load segmentation context (includes segment_model_strategy AND segment_profiles)\n"
            "segment_model_strategy = {}\n"
            "segment_profiles = {}  # STATE-OF-THE-ART: For segment statistics\n"
            "if os.path.exists(seg_ctx_path):\n"
            "    seg_reader = ContextReader(seg_ctx_path)\n"
            "    segment_model_strategy = seg_reader.get_by_type(SemanticTypes.SEGMENT_MODEL_STRATEGY) or seg_reader.get('segment_model_strategy', {})\n"
            "    # STATE-OF-THE-ART: Load segment profiles for detailed stats (zero_fraction, cv, etc.)\n"
            "    segment_profiles_path = os.path.join(seg_dir, 'segment_profiles.json')\n"
            "    if os.path.exists(segment_profiles_path):\n"
            "        segment_profiles = load_json(segment_profiles_path)\n"
            "        print(f'Loaded segment_profiles with {len(segment_profiles)} segments for hyperparameter derivation')\n\n"
            "# Read manifest to get model levels\n"
            "manifest = load_csv(os.path.join(feat_dir, 'training_manifest.csv'))\n"
            "model_groups = manifest['model_level'].unique()\n"
            "print(f'Found {len(model_groups)} unique model levels in manifest')\n\n"
            "# =================================================================\n"
            "# MODEL SELECTION: FEATURE-BASED MODELS ONLY (STATE-OF-THE-ART)\n"
            "# =================================================================\n"
            "from utils.intelligent_modeling import ALLOWED_MODELS, BANNED_MODELS\n\n"
            "# Pattern affects model ORDER (priority), not model TYPE\n"
            "# Loss function selection is now DATA-DRIVEN from EDA insights\n"
            f"_is_monthly = ('{time_format}' == 'year_month')\n"
            "if _is_monthly:\n"
            "    # Monthly: include univariate models (less data, statistical models shine)\n"
            "    pattern_to_models = {\n"
            "        'smooth': ['lightgbm', 'xgboost', 'ets', 'theta', 'arima', 'catboost'],\n"
            "        'erratic': ['xgboost', 'lightgbm', 'catboost', 'arima', 'ets', 'theta'],\n"
            "        'intermittent': ['lightgbm', 'croston', 'sba', 'tsb', 'imapa', 'zero_inflated'],\n"
            "        'lumpy': ['lightgbm', 'croston', 'sba', 'imapa', 'zero_inflated', 'hurdle_model'],\n"
            "    }\n"
            "else:\n"
            "    # Weekly: feature-based models only (more data available)\n"
            "    pattern_to_models = {\n"
            "        'smooth': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'tweedie'],\n"
            "        'erratic': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'tweedie'],\n"
            "        'intermittent': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'catboost'],\n"
            "        'lumpy': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'tweedie'],\n"
            "    }\n\n"
            "# Loss function selection based on demand pattern (STATE-OF-THE-ART)\n"
            "pattern_to_loss = {\n"
            "    'smooth': 'mse',\n"
            "    'erratic': 'huber',  # Robust to outliers\n"
            "    'intermittent': 'tweedie',  # Handles zeros\n"
            "    'lumpy': 'tweedie',  # Handles zeros + high variance\n"
            "}\n\n"
            "# Validation strategy selection (STATE-OF-THE-ART)\n"
            "# Use EDA insights for seasonal validation\n"
            "seasonality_info = eda_insights.get('seasonality', {})\n"
            "has_seasonality = seasonality_info.get('has_seasonality_pct', 0) > 0.5\n"
            "default_validation = 'seasonal_split' if has_seasonality else 'time_series_split'\n\n"
            "pattern_to_validation = {\n"
            "    'smooth': 'time_series_split' if not has_seasonality else 'seasonal_split',\n"
            "    'erratic': 'time_series_split',\n"
            "    'intermittent': 'holdout',  # Intermittent needs holdout\n"
            "    'lumpy': 'holdout',  # Lumpy needs holdout\n"
            "}\n\n"
            f"max_candidates = {max_model_candidates}  # From config.design.max_model_candidates_per_group\n\n"
            "# =================================================================\n"
            "# BUILD STRATEGY FOR EACH MODEL LEVEL (STATE-OF-THE-ART)\n"
            "# Uses: manifest intermittency_class + per_segment_training + EDA insights\n"
            "# =================================================================\n"
            "model_groups_config = {}\n"
            "demand_pattern_counts = {}\n"
            "loss_function_counts = {}\n"
            "validation_strategy_counts = {}\n\n"
            "for mg in model_groups:\n"
            "    mg_str = str(mg)\n"
            "    \n"
            "    # PRIMARY SOURCE: Get demand pattern from manifest's intermittency_class\n"
            "    mg_data = manifest[manifest['model_level'] == mg]\n"
            "    if 'intermittency_class' in mg_data.columns and len(mg_data) > 0:\n"
            "        mode_values = mg_data['intermittency_class'].mode()\n"
            "        demand_pattern = mode_values.iloc[0] if len(mode_values) > 0 else 'smooth'\n"
            "    else:\n"
            "        # Fallback to segment_model_strategy\n"
            "        seg_rec = segment_model_strategy.get(mg_str, {})\n"
            "        demand_pattern = seg_rec.get('dominant_pattern') or seg_rec.get('primary_demand_pattern') or 'smooth'\n"
            "    \n"
            "    demand_pattern = str(demand_pattern).lower()\n"
            "    demand_pattern_counts[demand_pattern] = demand_pattern_counts.get(demand_pattern, 0) + 1\n"
            "    \n"
            "    # Get candidate models for this pattern\n"
            "    candidates = pattern_to_models.get(demand_pattern, ['lightgbm'])\n"
            "    candidates = [m for m in candidates if m in allowed_families] or ['lightgbm']\n"
            "    \n"
            "    # DISCRETE DEMAND: Inject discrete models if segment has low-cardinality demand\n"
            "    if 'is_discrete' in mg_data.columns and 'n_unique' in mg_data.columns:\n"
            "        discrete_pct = float(mg_data['is_discrete'].mean())\n"
            "        avg_n_unique = float(mg_data['n_unique'].mean())\n"
            "        if discrete_pct > 0.3:  # >30% of keys in this segment are discrete\n"
            "            if avg_n_unique <= 5:\n"
            "                disc_models = ['discrete_classifier', 'ordinal_regression']\n"
            "            elif avg_n_unique <= 10:\n"
            "                disc_models = ['ordinal_regression', 'discrete_classifier']\n"
            "            else:\n"
            "                disc_models = ['hybrid_discrete', 'ordinal_regression']\n"
            "            disc_models = [m for m in disc_models if m in allowed_families]\n"
            "            # Inject at top of candidate list\n"
            "            candidates = disc_models + [m for m in candidates if m not in disc_models]\n"
            "            print(f'  Segment {mg_str}: discrete demand detected (pct={discrete_pct:.2f}, '\n"
            "                  f'avg_n_unique={avg_n_unique:.1f}), injecting {disc_models}')\n"
            "    \n"
            "    # STATE-OF-THE-ART: Get loss function from per_segment_training or pattern default\n"
            "    seg_training = per_segment_training.get(mg_str, {})\n"
            "    recommended_loss = seg_training.get('recommended_loss', pattern_to_loss.get(demand_pattern, 'mse'))\n"
            "    loss_function_counts[recommended_loss] = loss_function_counts.get(recommended_loss, 0) + 1\n"
            "    \n"
            "    # STATE-OF-THE-ART: Get validation strategy from per_segment_training or pattern default\n"
            "    validation_strategy = seg_training.get('validation_strategy', pattern_to_validation.get(demand_pattern, 'time_series_split'))\n"
            "    validation_strategy_counts[validation_strategy] = validation_strategy_counts.get(validation_strategy, 0) + 1\n"
            "    \n"
            "    # STATE-OF-THE-ART: Get expected difficulty\n"
            "    expected_difficulty = seg_training.get('expected_difficulty', 'medium')\n"
            "    \n"
            "    # STATE-OF-THE-ART: Extract segment statistics for hyperparameter derivation\n"
            "    # These stats enable data-driven hyperparameter selection\n"
            "    seg_stats = {}\n"
            "    \n"
            "    # Method 1: Extract from manifest (per-key metrics aggregated to segment)\n"
            "    if 'zero_fraction' in mg_data.columns:\n"
            "        seg_stats['zero_fraction'] = float(mg_data['zero_fraction'].mean())\n"
            "    if 'cv' in mg_data.columns:\n"
            "        seg_stats['cv'] = float(mg_data['cv'].mean())\n"
            "    elif 'coefficient_of_variation' in mg_data.columns:\n"
            "        seg_stats['cv'] = float(mg_data['coefficient_of_variation'].mean())\n"
            "    if 'skewness' in mg_data.columns:\n"
            "        seg_stats['skewness'] = float(mg_data['skewness'].mean())\n"
            "    seg_stats['n_keys'] = len(mg_data)\n"
            "    \n"
            "    # Count observations per key for avg_series_length\n"
            "    if 'n_obs' in mg_data.columns:\n"
            "        seg_stats['avg_series_length'] = float(mg_data['n_obs'].mean())\n"
            "    \n"
            "    # Method 2: Fallback to segment_profiles if manifest doesn't have stats\n"
            "    if mg_str in segment_profiles:\n"
            "        profile_means = segment_profiles[mg_str].get('means', {})\n"
            "        if 'zero_fraction' not in seg_stats and 'zero_fraction_clean' in profile_means:\n"
            "            seg_stats['zero_fraction'] = profile_means.get('zero_fraction_clean', 0.1)\n"
            "        if 'cv' not in seg_stats and 'cv_clean' in profile_means:\n"
            "            seg_stats['cv'] = profile_means.get('cv_clean', 0.5)\n"
            "        seg_stats['pct_of_total'] = segment_profiles[mg_str].get('pct_of_total', 0)\n"
            "    \n"
            "    # Defaults if still missing\n"
            "    seg_stats.setdefault('zero_fraction', 0.3 if demand_pattern in ['intermittent', 'lumpy'] else 0.1)\n"
            "    seg_stats.setdefault('cv', 1.5 if demand_pattern in ['erratic', 'lumpy'] else 0.5)\n"
            "    seg_stats.setdefault('skewness', 2.0 if demand_pattern == 'lumpy' else 1.0)\n"
            "    seg_stats.setdefault('avg_series_length', 100)\n"
            "    \n"
            "    model_groups_config[mg_str] = {\n"
            "        'demand_pattern': demand_pattern,\n"
            "        'candidate_models': candidates[:max_candidates],\n"
            "        'primary_model': candidates[0],\n"
            "        # STATE-OF-THE-ART: EDA-driven configuration\n"
            "        'recommended_loss': recommended_loss,\n"
            "        'validation_strategy': validation_strategy,\n"
            "        'expected_difficulty': expected_difficulty,\n"
            "        # STATE-OF-THE-ART: Segment statistics for hyperparameter derivation\n"
            "        'segment_stats': seg_stats,\n"
            "        'recommended_hyperparams': {\n"
            "            'early_stopping_rounds': 100 if expected_difficulty == 'high' else 50,\n"
            "            'learning_rate': 0.01 if expected_difficulty == 'high' else 0.05,\n"
            "        },\n"
            "    }\n\n"
            "print(f'Demand patterns: {demand_pattern_counts}')\n"
            "print(f'Loss functions: {loss_function_counts}')\n"
            "print(f'Validation strategies: {validation_strategy_counts}')\n"
            "# STATE-OF-THE-ART: Summarize segment stats extraction\n"
            "stats_with_data = sum(1 for mg in model_groups_config.values() if mg.get('segment_stats', {}).get('zero_fraction', 0) > 0)\n"
            "print(f'Segment stats extracted for {stats_with_data}/{len(model_groups_config)} segments (for hyperparameter derivation)')\n\n"
            "# =================================================================\n"
            "# CREATE STATE-OF-THE-ART STRATEGY WITH ALL EDA INSIGHTS\n"
            "# =================================================================\n"
            "strategy = {\n"
            "    'model_groups': model_groups_config,\n"
            "    'global_config': {\n"
            f"        'max_candidates_per_group': {max_model_candidates},\n"
            "        'early_stopping_rounds': 50,\n"
            f"        'allowed_families': {allowed_model_families},\n"
            "    },\n"
            "    # STATE-OF-THE-ART: EDA-driven configuration\n"
            "    'eda_driven_config': {\n"
            f"        'seasonal_period': adaptive_config.get('detected_seasonal_period', {12 if time_format == 'year_month' else 52}),\n"
            "        'acf_informed_lags': adaptive_config.get('acf_informed_lags', False),\n"
            "        'adaptive_lags': adaptive_config.get('adaptive_lags', []),\n"
            "        'fourier_order': adaptive_config.get('fourier_order', 3),\n"
            "        'changepoint_indicators': adaptive_config.get('changepoint_indicators', False),\n"
            "        'has_seasonality': has_seasonality,\n"
            "    },\n"
            "    # STATE-OF-THE-ART: EDA insights for training\n"
            "    'eda_insights_for_training': eda_insights,\n"
            "    # STATE-OF-THE-ART: Per-segment training strategies\n"
            "    'per_segment_training_strategy': per_segment_training,\n"
            "    'feature_context_summary': {\n"
            "        'n_features': n_features,\n"
            "        'feature_types': feature_categories,\n"
            "    },\n"
            "    'segmentation_context_summary': {\n"
            "        'n_segments': len(segment_model_strategy),\n"
            "        'demand_pattern_distribution': demand_pattern_counts,\n"
            "        'loss_function_distribution': loss_function_counts,\n"
            "        'validation_strategy_distribution': validation_strategy_counts,\n"
            "    },\n"
            "}\n\n"
            "# Save strategy\n"
            "save_json(strategy, os.path.join(model_dir, 'training_strategy.json'))\n"
            "print(f'Training strategy created for {len(model_groups)} model groups (STATE-OF-THE-ART)')\n"
            "print('Saved: training_strategy.json with EDA-driven loss functions and validation strategies')\n"
            "```\n\n"
            "Then tell Executor: 'Use training_strategy.json to orchestrate individual training utilities "
            "with per-segment loss functions and validation strategies.'"
        ),
        agent=planner,
        expected_output=(
            "Created training_strategy.json with STATE-OF-THE-ART configuration including:\n"
            "- Per-segment model selections based on demand patterns\n"
            "- EDA-driven loss function recommendations (tweedie for intermittent/lumpy, mse for smooth)\n"
            "- Validation strategy selections (holdout for sparse, time_series_split for smooth)\n"
            "- ACF-informed hyperparameter suggestions\n"
            "Executor should use individual training utilities with per-segment loss functions."
        ),
        output_file=os.path.join(model_dir_out, "training_planning_report.md"),
        # CRITICAL: Callback to validate Planner output BEFORE Executor starts
        callback=lambda output: _validate_planner_output(model_dir),
    )

    # -------------------------------------------------------------------------
    # Task 2: Training Execution - STATE-OF-THE-ART UTILITY ORCHESTRATION
    # Uses per-segment loss functions, validation strategies, and EDA config
    # -------------------------------------------------------------------------
    task_execute = Task(
        name="execute_training_orchestration",
        description=(
            "# EXECUTE THIS CODE USING CodeExecutionTool\n\n"
            "**CRITICAL: You MUST use CodeExecutionTool to execute the Python code below.**\n"
            "**DO NOT describe what the code does. DO NOT write your own version.**\n"
            "**JUST COPY THE CODE BLOCK INTO CodeExecutionTool AND RUN IT.**\n\n"
            "The code runs the full training pipeline with all context.\n\n"
            "## PYTHON CODE TO EXECUTE:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n"
            "from utils.model_training import run_full_training_pipeline\n"
            "from utils.context_schema import ContextReader, SemanticTypes\n"
            "from utils.agent_utilities import load_json\n\n"
            f"feat_dir = '{feat_dir}'\n"
            f"seg_dir = '{seg_dir}'\n"
            f"model_dir = '{model_dir}'\n"
            f"target_col = '{target_col}'\n"
            f"strategy_path = os.path.join(model_dir, 'training_strategy.json')\n\n"
            "# =================================================================\n"
            "# CRITICAL: Validate strategy file exists (created by Planner)\n"
            "# =================================================================\n"
            "if not os.path.exists(strategy_path):\n"
            "    raise FileNotFoundError(\n"
            "        f'CRITICAL: training_strategy.json not found at {strategy_path}\\n'\n"
            "        'This file should have been created by the Training Planner task.\\n'\n"
            "        'Please check that Task 1 (Planner) executed successfully.'\n"
            "    )\n\n"
            "# =================================================================\n"
            "# LOAD STATE-OF-THE-ART TRAINING STRATEGY (from Planner)\n"
            "# =================================================================\n"
            "strategy = load_json(strategy_path)\n"
            "eda_config = strategy.get('eda_driven_config', {})\n"
            "eda_insights = strategy.get('eda_insights_for_training', {})\n"
            "per_segment_strategy = strategy.get('per_segment_training_strategy', {})\n\n"
            f"print(f'Loaded strategy with EDA config: seasonal_period={{eda_config.get(\"seasonal_period\", {12 if time_format == 'year_month' else 52})}}')\n"
            "print(f'EDA insights available: {list(eda_insights.keys())}')\n"
            "print(f'Per-segment strategies: {len(per_segment_strategy)} segments')\n\n"
            "# =================================================================\n"
            "# LOAD UPSTREAM CONTEXT (Segmentation + Features)\n"
            "# =================================================================\n"
            "seg_context_path = os.path.join(seg_dir, 'segmentation_to_training_context.json')\n"
            "segmentation_context = None\n"
            "if os.path.exists(seg_context_path):\n"
            "    seg_reader = ContextReader(seg_context_path)\n"
            "    seg_defs = seg_reader.get_by_type(SemanticTypes.SEGMENT_DEFINITIONS) or seg_reader.get('segment_definitions', {})\n"
            "    segmentation_context = {\n"
            "        'n_segments': seg_defs.get('n_segments', 0),\n"
            "        'segment_labels': seg_defs.get('segment_labels', {}),\n"
            "        'segment_model_strategy': seg_reader.get_by_type(SemanticTypes.SEGMENT_MODEL_STRATEGY) or seg_reader.get('segment_model_strategy', {}),\n"
            "    }\n"
            "    print(f'Loaded segmentation context: {segmentation_context.get(\"n_segments\", \"?\")} segments')\n\n"
            "feat_context_path = os.path.join(feat_dir, 'feature_to_training_context.json')\n"
            "feature_context = None\n"
            "if os.path.exists(feat_context_path):\n"
            "    feat_reader = ContextReader(feat_context_path)\n"
            "    feat_summary = feat_reader.get_by_type(SemanticTypes.FEATURE_SUMMARY) or feat_reader.get('feature_summary', {})\n"
            "    feature_context = {\n"
            "        'feature_summary': feat_summary,\n"
            "        'model_level_summary': feat_reader.get_by_type(SemanticTypes.MODEL_LEVEL_SUMMARY) or feat_reader.get('model_level_summary', {}),\n"
            "        'model_recommendations': feat_reader.get_by_type(SemanticTypes.MODEL_RECOMMENDATIONS) or feat_reader.get('model_recommendations', {}),\n"
            "        # STATE-OF-THE-ART: Include EDA-driven config\n"
            "        'eda_driven_config': eda_config,\n"
            "        'per_segment_training_strategy': per_segment_strategy,\n"
            "    }\n"
            "    n_feat = feat_summary.get('total_features', 0)\n"
            "    print(f'Loaded feature context: {n_feat} features with EDA-driven config')\n\n"
            "# =================================================================\n"
            "# RUN STATE-OF-THE-ART TRAINING PIPELINE\n"
            "# =================================================================\n"
            "# The pipeline will use:\n"
            "# - Per-segment loss functions from strategy\n"
            "# - Validation strategies from strategy\n"
            "# - EDA-driven hyperparameters\n"
            "result = run_full_training_pipeline(\n"
            "    feature_dir=feat_dir,\n"
            "    model_dir=model_dir,\n"
            "    target_col=target_col,\n"
            "    strategy_path=strategy_path,\n"
            f"    enable_meta_learning={meta_learning_enabled},\n"
            f"    enable_ensemble_optimization={enable_ensemble_optimization},\n"
            f"    enable_bias_correction={enable_bias_correction},\n"
            f"    enable_forecast_calibration={enable_forecast_calibration},\n"
            f"    prediction_intervals={prediction_intervals},\n"
            f"    prediction_interval_confidence={prediction_interval_confidence},\n"
            "    # STATE-OF-THE-ART: Pass ALL enriched context\n"
            "    segmentation_context=segmentation_context,\n"
            "    feature_context=feature_context,\n"
            f"    early_stopping_rounds=50,\n"
            f"    max_candidates_per_group={max_model_candidates},\n"
            "    # Bias calibration settings from config\n"
            f"    apply_bias_calibration={apply_bias_calibration},\n"
            f"    bias_calibration_buckets={bias_calibration_buckets},\n"
            f"    bias_calibration_factor_min={bias_calibration_factor_min},\n"
            f"    bias_calibration_factor_max={bias_calibration_factor_max},\n"
            f"    # Time format for period-aware walk-forward CV defaults\n"
            f"    time_format='{time_format}',\n"
            f"    date_col='{config.date_column}',\n"
            f"    forecast_horizon={config.forecast_horizon},\n"
            ")\n\n"
            "# Report results with STATE-OF-THE-ART details\n"
            "if result.success:\n"
            "    print(f'STATE-OF-THE-ART Training SUCCESS: {result.models_trained} models trained')\n"
            "    print(f'Overall WAPE: {result.overall_wape:.4f}')\n"
            "    print(f'Failed: {result.models_failed} models')\n"
            "    print(f'Specs saved: {result.final_specs_path}')\n"
            "    print(f'Diagnostic context: {result.diagnostic_context_path}')\n"
            "    # Log loss function usage\n"
            "    loss_dist = strategy.get('segmentation_context_summary', {}).get('loss_function_distribution', {})\n"
            "    if loss_dist:\n"
            "        print(f'Loss functions used: {loss_dist}')\n"
            "else:\n"
            "    print(f'Training FAILED: {result.error_message}')\n"
            "```\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n\n"
            "## STATE-OF-THE-ART FEATURES:\n"
            "- Per-segment loss functions (tweedie for zeros, huber for outliers, mse for smooth)\n"
            "- Validation strategies adapted to demand patterns\n"
            "- EDA-driven hyperparameters (learning rate, early stopping)"
        ),
        agent=executor,
        expected_output=(
            "STATE-OF-THE-ART Training complete:\n"
            "- N models trained with overall WAPE of X.XX\n"
            "- Per-segment loss functions applied (tweedie/mse/huber)\n"
            "- Validation strategies applied (holdout/time_series_split)\n"
            "- Files created: final_model_specs.json, training_to_diagnostic_context.json"
        ),
        output_file=os.path.join(model_dir_out, "training_execution_report.md"),
        context=[task_plan],
    )

    # -------------------------------------------------------------------------
    # Task 3: Training Analysis - Create ENRICHED diagnostic context
    # STATE-OF-THE-ART: Includes EDA insights, per-segment analysis, and
    # actionable recommendations for Diagnostic crew
    # -------------------------------------------------------------------------
    task_analyze = Task(
        name="create_diagnostic_context",
        description=(
            "# EXECUTE THIS CODE USING CodeExecutionTool\n\n"
            "**CRITICAL: You MUST use CodeExecutionTool to execute the Python code below.**\n"
            "**DO NOT describe what the code does. DO NOT write your own version.**\n"
            "**JUST COPY THE CODE BLOCK INTO CodeExecutionTool AND RUN IT.**\n\n"
            "The code analyzes training results and enriches the diagnostic context.\n\n"
            "## PYTHON CODE TO EXECUTE:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n"
            "from utils.agent_utilities import load_json, save_json\n"
            "from utils.context_schema import ContextBuilder, SemanticTypes\n\n"
            f"model_dir = '{model_dir}'\n"
            f"feat_dir = '{feat_dir}'\n\n"
            "# =================================================================\n"
            "# LOAD ALL TRAINING OUTPUTS\n"
            "# =================================================================\n"
            "specs_path = os.path.join(model_dir, 'final_model_specs.json')\n"
            "context_path = os.path.join(model_dir, 'training_to_diagnostic_context.json')\n"
            "strategy_path = os.path.join(model_dir, 'training_strategy.json')\n\n"
            "specs = load_json(specs_path) if os.path.exists(specs_path) else {}\n"
            "existing_context = load_json(context_path) if os.path.exists(context_path) else {}\n"
            "strategy = load_json(strategy_path) if os.path.exists(strategy_path) else {}\n\n"
            "print(f'Models trained: {specs.get(\"models_trained\", 0)}')\n"
            "print(f'Overall WAPE: {specs.get(\"overall_wape\", \"N/A\")}')\n\n"
            "# =================================================================\n"
            "# ENRICH DIAGNOSTIC CONTEXT WITH EDA INSIGHTS (STATE-OF-THE-ART)\n"
            "# =================================================================\n"
            "eda_config = strategy.get('eda_driven_config', {})\n"
            "eda_insights = strategy.get('eda_insights_for_training', {})\n"
            "per_segment_strategy = strategy.get('per_segment_training_strategy', {})\n"
            "model_groups_config = strategy.get('model_groups', {})\n\n"
            "# Analyze per-segment performance vs expectations\n"
            "performance_vs_expectation = {}\n"
            "model_performance = existing_context.get('model_performance', {})\n"
            "for mg_str, config in model_groups_config.items():\n"
            "    expected_difficulty = config.get('expected_difficulty', 'medium')\n"
            "    mg_perf = model_performance.get(mg_str, {})\n"
            "    actual_wape = mg_perf.get('val_wape', 999)\n"
            "    \n"
            "    # Determine if performance matches expectations\n"
            "    if expected_difficulty == 'high' and actual_wape < 0.4:\n"
            "        status = 'better_than_expected'\n"
            "    elif expected_difficulty == 'low' and actual_wape > 0.3:\n"
            "        status = 'worse_than_expected'\n"
            "    else:\n"
            "        status = 'as_expected'\n"
            "    \n"
            "    performance_vs_expectation[mg_str] = {\n"
            "        'expected_difficulty': expected_difficulty,\n"
            "        'actual_wape': actual_wape,\n"
            "        'status': status,\n"
            "        'demand_pattern': config.get('demand_pattern', 'unknown'),\n"
            "        'loss_function_used': config.get('recommended_loss', 'mse'),\n"
            "        'validation_strategy': config.get('validation_strategy', 'time_series_split'),\n"
            "    }\n\n"
            "# Count performance categories\n"
            "perf_categories = {'better_than_expected': 0, 'as_expected': 0, 'worse_than_expected': 0}\n"
            "for v in performance_vs_expectation.values():\n"
            "    perf_categories[v['status']] = perf_categories.get(v['status'], 0) + 1\n"
            "print(f'Performance vs expectations: {perf_categories}')\n\n"
            "# =================================================================\n"
            "# GENERATE ACTIONABLE DIAGNOSTIC RECOMMENDATIONS\n"
            "# =================================================================\n"
            "diagnostic_recommendations = []\n\n"
            "# 1. Check for underperforming segments\n"
            "underperformers = [k for k, v in performance_vs_expectation.items() if v['status'] == 'worse_than_expected']\n"
            "if underperformers:\n"
            "    diagnostic_recommendations.append({\n"
            "        'priority': 'high',\n"
            "        'category': 'underperformance',\n"
            "        'segments': underperformers[:5],\n"
            "        'recommendation': 'Investigate feature quality or try alternative models for these segments',\n"
            "        'suggested_actions': [\n"
            "            'Check for data quality issues in these segments',\n"
            "            'Consider alternative loss functions',\n"
            "            'Review if demand pattern classification is correct',\n"
            "        ],\n"
            "    })\n\n"
            "# 2. Check seasonality handling\n"
            "seasonality_info = eda_insights.get('seasonality', {})\n"
            "if seasonality_info.get('has_seasonality_pct', 0) > 0.5:\n"
            "    diagnostic_recommendations.append({\n"
            "        'priority': 'medium',\n"
            "        'category': 'seasonality',\n"
            "        'info': f\"{seasonality_info.get('has_seasonality_pct', 0)*100:.0f}% of series have seasonality\",\n"
            "        'recommendation': 'Ensure seasonal validation was used for seasonal series',\n"
            f"        'seasonal_period': eda_config.get('seasonal_period', {12 if time_format == 'year_month' else 52}),\n"
            "    })\n\n"
            "# 3. Check changepoint handling\n"
            "changepoint_info = eda_insights.get('changepoints', {})\n"
            "if changepoint_info.get('pct_with_changepoints', 0) > 0.3:\n"
            "    diagnostic_recommendations.append({\n"
            "        'priority': 'medium',\n"
            "        'category': 'changepoints',\n"
            "        'info': f\"{changepoint_info.get('pct_with_changepoints', 0)*100:.0f}% of series have structural breaks\",\n"
            "        'recommendation': 'Consider weighting recent data more heavily for series with changepoints',\n"
            "        'changepoint_indicators_used': eda_config.get('changepoint_indicators', False),\n"
            "    })\n\n"
            "# 4. Loss function effectiveness\n"
            "loss_dist = strategy.get('segmentation_context_summary', {}).get('loss_function_distribution', {})\n"
            "if loss_dist:\n"
            "    diagnostic_recommendations.append({\n"
            "        'priority': 'info',\n"
            "        'category': 'loss_functions',\n"
            "        'distribution': loss_dist,\n"
            "        'recommendation': 'Review if tweedie loss improved performance for intermittent/lumpy segments',\n"
            "    })\n\n"
            "print(f'Generated {len(diagnostic_recommendations)} diagnostic recommendations')\n\n"
            "# =================================================================\n"
            "# ENRICH AND SAVE DIAGNOSTIC CONTEXT\n"
            "# =================================================================\n"
            "# Add enriched fields to existing context\n"
            "existing_context['eda_insights_used'] = {\n"
            "    'acf_informed_lags': eda_config.get('acf_informed_lags', False),\n"
            f"    'seasonal_period': eda_config.get('seasonal_period', {12 if time_format == 'year_month' else 52}),\n"
            "    'changepoint_indicators': eda_config.get('changepoint_indicators', False),\n"
            "    'fourier_order': eda_config.get('fourier_order', 3),\n"
            "}\n"
            "existing_context['per_segment_analysis'] = performance_vs_expectation\n"
            "existing_context['performance_summary'] = perf_categories\n"
            "existing_context['diagnostic_recommendations'] = diagnostic_recommendations\n"
            "existing_context['training_approach'] = 'state_of_the_art'\n\n"
            "# Save enriched context\n"
            "save_json(existing_context, context_path)\n"
            "print(f'ENRICHED diagnostic context saved to {context_path}')\n"
            "print('Training analysis complete with STATE-OF-THE-ART insights')\n"
            "```"
        ),
        agent=analyst,
        expected_output=(
            "Analyzed training outputs with STATE-OF-THE-ART insights:\n"
            "- Performance vs expectations analysis for each segment\n"
            "- EDA insights utilization summary\n"
            "- Actionable diagnostic recommendations\n"
            "ENRICHED training_to_diagnostic_context.json for Diagnostic crew."
        ),
        output_file=os.path.join(model_dir_out, "training_analysis_report.md"),
        context=[task_plan, task_execute],
    )

    # -------------------------------------------------------------------------
    # Task 4: Documentation - Generate comprehensive insights guide
    # Uses MULTI-PHASE ITERATIVE approach like EDA Insights Agent
    # -------------------------------------------------------------------------
    documentation_agent = _create_training_documentation_agent(llm, protected_paths=protected_dirs)
    task_document = Task(
        name="generate_training_documentation",
        description=(
            "# MODEL TRAINING INSIGHTS REPORT GENERATION\n\n"
            f"Analyze training outputs in `{model_dir}` and create a comprehensive insights report.\n\n"
            "## AVAILABLE FILES\n\n"
            f"- `{model_dir}/final_model_specs.json` - Model specifications and WAPE metrics\n"
            f"- `{model_dir}/training_strategy.json` - Training strategy with EDA-driven config\n"
            f"- `{model_dir}/training_to_diagnostic_context.json` - Diagnostic context and recommendations\n"
            f"- `{model_dir}/*.pkl` - Trained model artifacts\n\n"
            "## PHASE 1: EXPLORE TRAINING OUTPUTS\n\n"
            "Execute this code to understand the training results:\n\n"
            "```python\n"
            "import json\n"
            "import os\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n\n"
            f"model_dir = '{model_dir}'\n\n"
            "# List files\n"
            "files = [f for f in os.listdir(model_dir) if f.endswith(('.json', '.pkl', '.md'))]\n"
            "print(f'Available files: {{len(files)}}')\n"
            "for f in sorted(files)[:15]: print(f'  - {{f}}')\n\n"
            "# Load final model specs\n"
            "specs_path = os.path.join(model_dir, 'final_model_specs.json')\n"
            "with open(specs_path) as f:\n"
            "    specs = json.load(f)\n\n"
            "overall_wape = specs.get('overall_wape', 0)\n"
            "n_models = len(specs.get('model_specs', []))\n"
            "print(f'\\nOverall WAPE: {{overall_wape:.4f}} ({{overall_wape*100:.1f}}%)')\n"
            "print(f'Models trained: {{n_models}}')\n"
            "```\n\n"
            "## PHASE 2: ANALYZE MODEL PERFORMANCE BY SEGMENT\n\n"
            "Execute this code to analyze per-segment performance:\n\n"
            "```python\n"
            "import json\n"
            "import os\n\n"
            f"model_dir = '{model_dir}'\n\n"
            "# Load model specs\n"
            "with open(os.path.join(model_dir, 'final_model_specs.json')) as f:\n"
            "    specs = json.load(f)\n\n"
            "model_specs = specs.get('model_specs', [])\n\n"
            "print('=== MODEL PERFORMANCE BY SEGMENT ===')\n"
            "# Count by WAPE thresholds\n"
            "excellent = sum(1 for s in model_specs if s.get('val_wape', 1) < 0.2)\n"
            "good = sum(1 for s in model_specs if 0.2 <= s.get('val_wape', 1) < 0.35)\n"
            "acceptable = sum(1 for s in model_specs if 0.35 <= s.get('val_wape', 1) < 0.5)\n"
            "poor = sum(1 for s in model_specs if s.get('val_wape', 1) >= 0.5)\n\n"
            "print(f'Excellent (WAPE < 20%): {{excellent}}')\n"
            "print(f'Good (20-35%): {{good}}')\n"
            "print(f'Acceptable (35-50%): {{acceptable}}')\n"
            "print(f'Poor (>50%): {{poor}}')\n\n"
            "# Count by model type\n"
            "model_types = {{}}\n"
            "for s in model_specs:\n"
            "    mt = s.get('model_type', 'unknown')\n"
            "    model_types[mt] = model_types.get(mt, 0) + 1\n"
            "print(f'\\nModel types used: {{model_types}}')\n\n"
            "# Show top and bottom performers\n"
            "sorted_specs = sorted(model_specs, key=lambda x: x.get('val_wape', 1))\n"
            "print('\\nTop 3 performers:')\n"
            "for s in sorted_specs[:3]:\n"
            "    print(f\"  {{s.get('model_group', '?')}}: {{s.get('val_wape', 0):.4f}} ({{s.get('model_type', '?')}}})\")\n"
            "```\n\n"
            "## PHASE 3: ANALYZE TRAINING STRATEGY AND EDA CONFIG\n\n"
            "Execute this code to analyze the strategy used:\n\n"
            "```python\n"
            "import json\n"
            "import os\n\n"
            f"model_dir = '{model_dir}'\n\n"
            "# Load training strategy\n"
            "with open(os.path.join(model_dir, 'training_strategy.json')) as f:\n"
            "    strategy = json.load(f)\n\n"
            "eda_config = strategy.get('eda_driven_config', {{}})\n"
            "seg_summary = strategy.get('segmentation_context_summary', {{}})\n"
            "model_groups = strategy.get('model_groups', {{}})\n\n"
            "print('=== EDA-DRIVEN CONFIGURATION ===')\n"
            "print(f'Seasonal period: {{eda_config.get(\"seasonal_period\", \"N/A\")}}')\n"
            "print(f'ACF-informed lags: {{eda_config.get(\"acf_informed_lags\", False)}}')\n"
            "print(f'Has seasonality: {{eda_config.get(\"has_seasonality\", False)}}')\n\n"
            "print('\\n=== LOSS FUNCTION DISTRIBUTION ===')\n"
            "loss_dist = seg_summary.get('loss_function_distribution', {{}})\n"
            "for loss, count in loss_dist.items():\n"
            "    print(f'{{loss}}: {{count}} segments')\n\n"
            "print('\\n=== DEMAND PATTERN DISTRIBUTION ===')\n"
            "pattern_dist = seg_summary.get('demand_pattern_distribution', {{}})\n"
            "for pattern, count in pattern_dist.items():\n"
            "    print(f'{{pattern}}: {{count}} segments')\n"
            "```\n\n"
            "## PHASE 4: ANALYZE ENSEMBLE AND ADVANCED FEATURES\n\n"
            "Execute this code to analyze advanced features:\n\n"
            "```python\n"
            "import json\n"
            "import os\n\n"
            f"model_dir = '{model_dir}'\n\n"
            "# Load model specs\n"
            "with open(os.path.join(model_dir, 'final_model_specs.json')) as f:\n"
            "    specs = json.load(f)\n\n"
            "model_specs = specs.get('model_specs', [])\n"
            "sota = specs.get('state_of_art_features', {{}})\n\n"
            "# Count ensembles\n"
            "n_ensembles = sum(1 for s in model_specs if s.get('is_ensemble', False))\n"
            "n_single = len(model_specs) - n_ensembles\n\n"
            "print('=== ENSEMBLE ANALYSIS ===')\n"
            "print(f'Ensemble models: {{n_ensembles}}')\n"
            "print(f'Single models: {{n_single}}')\n\n"
            "# Show ensemble details\n"
            "for s in model_specs:\n"
            "    if s.get('is_ensemble'):\n"
            "        ensemble_info = s.get('ensemble_info', {{}})\n"
            "        models = ensemble_info.get('model_types', [])\n"
            "        print(f\"  {{s.get('model_group')}}: {{' + '.join(models)}}\")\n\n"
            "print('\\n=== STATE-OF-THE-ART FEATURES ===')\n"
            "print(f'Segment-specific hyperparams: {{sota.get(\"segment_specific_hyperparams\", False)}}')\n"
            "print(f'Pattern-aware ensemble: {{sota.get(\"pattern_aware_ensemble\", False)}}')\n"
            "print(f'Bias calibration: {{specs.get(\"bias_calibration_enabled\", False)}}')\n"
            "```\n\n"
            "## PHASE 5: GENERATE AND SAVE REPORT\n\n"
            "After gathering all insights, execute code to CREATE and SAVE the report.\n"
            "YOU MUST use the ACTUAL NUMBERS from your previous code executions.\n\n"
            "```python\n"
            "from datetime import datetime\n"
            "import json\n"
            "import os\n\n"
            f"model_dir = '{model_dir}'\n\n"
            "# Re-load all data to build comprehensive report\n"
            "with open(os.path.join(model_dir, 'final_model_specs.json')) as f:\n"
            "    specs = json.load(f)\n"
            "with open(os.path.join(model_dir, 'training_strategy.json')) as f:\n"
            "    strategy = json.load(f)\n\n"
            "# Extract statistics\n"
            "overall_wape = specs.get('overall_wape', 0)\n"
            "overall_test_wape = specs.get('overall_test_wape')\n"
            "model_specs_list = specs.get('model_specs', [])\n"
            "n_models = len(model_specs_list)\n"
            "training_time = specs.get('training_time_seconds', 0)\n"
            "sota = specs.get('state_of_art_features', {{}})\n"
            "bias_cal = specs.get('bias_calibration_enabled', False)\n"
            "eda_config = strategy.get('eda_driven_config', {{}})\n"
            "seg_summary = strategy.get('segmentation_context_summary', {{}})\n\n"
            "# Calculate performance buckets\n"
            "excellent = sum(1 for s in model_specs_list if s.get('val_wape', 1) < 0.2)\n"
            "good = sum(1 for s in model_specs_list if 0.2 <= s.get('val_wape', 1) < 0.35)\n"
            "acceptable = sum(1 for s in model_specs_list if 0.35 <= s.get('val_wape', 1) < 0.5)\n"
            "poor = sum(1 for s in model_specs_list if s.get('val_wape', 1) >= 0.5)\n\n"
            "# Count model types and ensembles\n"
            "model_types = {{}}\n"
            "n_ensembles = 0\n"
            "for s in model_specs_list:\n"
            "    mt = s.get('model_type', 'unknown')\n"
            "    model_types[mt] = model_types.get(mt, 0) + 1\n"
            "    if s.get('is_ensemble'): n_ensembles += 1\n\n"
            "# Build comprehensive markdown report\n"
            "quality_tier = 'EXCELLENT' if overall_wape < 0.2 else 'GOOD' if overall_wape < 0.35 else 'ACCEPTABLE' if overall_wape < 0.5 else 'NEEDS IMPROVEMENT'\n\n"
            "md = f'''# Model Training Insights Guide\n"
            "**Generated:** {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\n\n"
            "---\n\n"
            "## Executive Summary\n\n"
            "This report provides comprehensive analysis of the model training pipeline.\n"
            "**{{n_models}} models** were trained achieving an **overall validation WAPE of {{overall_wape:.1%}}** ({{quality_tier}}).\n\n"
            "The pipeline used **state-of-the-art techniques** including segment-specific hyperparameters,\n"
            "pattern-aware loss functions, and ensemble optimization.\n\n"
            "---\n\n"
            "## 1. Overall Performance\n\n"
            "| Metric | Value | Interpretation |\n"
            "|--------|-------|----------------|\n"
            "| Overall Validation WAPE | {{overall_wape:.1%}} | {{quality_tier}} |\n"
            "| Models Trained | {{n_models}} | Segment-level models |\n"
            "| Training Time | {{training_time:.1f}}s | Total duration |\n"
            "| Ensemble Models | {{n_ensembles}} | Pattern-aware ensembles |\n\n"
            "### Performance Distribution\n\n"
            "| Quality Tier | WAPE Range | Count | Percentage |\n"
            "|--------------|------------|-------|------------|\n"
            "| ✓ Excellent | < 20% | {{excellent}} | {{excellent/n_models*100:.1f}}% |\n"
            "| ✓ Good | 20-35% | {{good}} | {{good/n_models*100:.1f}}% |\n"
            "| ⚠ Acceptable | 35-50% | {{acceptable}} | {{acceptable/n_models*100:.1f}}% |\n"
            "| ✗ Poor | > 50% | {{poor}} | {{poor/n_models*100:.1f}}% |\n\n"
            "---\n\n"
            "## 2. Model Types Used\n\n"
            "| Model Type | Count | Percentage |\n"
            "|------------|-------|------------|\n'''\n\n"
            "for mt, count in sorted(model_types.items(), key=lambda x: -x[1]):\n"
            "    md += f'| {{mt}} | {{count}} | {{count/n_models*100:.1f}}% |\\n'\n\n"
            "md += f'''\\n\\n---\\n\\n"
            "## 3. EDA-Driven Configuration\n\n"
            "The training was configured based on upstream EDA insights:\n\n"
            "- **Seasonal Period:** {{eda_config.get('seasonal_period', 'N/A')}}\n"
            "- **ACF-Informed Lags:** {{eda_config.get('acf_informed_lags', False)}}\n"
            "- **Has Seasonality:** {{eda_config.get('has_seasonality', False)}}\n"
            "- **Fourier Order:** {{eda_config.get('fourier_order', 3)}}\n\n"
            "### Loss Function Distribution\n\n"
            "| Loss Function | Segments | Use Case |\n"
            "|---------------|----------|----------|\n'''\n\n"
            "loss_dist = seg_summary.get('loss_function_distribution', {{}})\n"
            "loss_descriptions = {{'mse': 'Standard regression', 'huber': 'Robust to outliers', 'tweedie': 'Handles zeros'}}\n"
            "for loss, count in loss_dist.items():\n"
            "    desc = loss_descriptions.get(loss, 'Custom')\n"
            "    md += f'| {{loss}} | {{count}} | {{desc}} |\\n'\n\n"
            "md += f'''\\n\\n---\\n\\n"
            "## 4. State-of-the-Art Features\n\n"
            "| Feature | Enabled | Description |\n"
            "|---------|---------|-------------|\n"
            "| Segment-Specific Hyperparams | {{'✓' if sota.get('segment_specific_hyperparams') else '✗'}} | Derived from segment statistics |\n"
            "| Pattern-Aware Ensemble | {{'✓' if sota.get('pattern_aware_ensemble') else '✗'}} | Weights by demand pattern |\n"
            "| Bias Calibration | {{'✓' if bias_cal else '✗'}} | Systematic error correction |\n"
            "| Recursive Validation | ✓ | Multi-step forecast evaluation |\n\n"
            "---\n\n"
            "## 5. Recommendations\n\n"
            "### For High-Performing Segments (WAPE < 35%)\n\n"
            "- These segments are well-modeled with current approach\n"
            "- Consider reducing model complexity to improve inference speed\n\n"
            "### For Underperforming Segments (WAPE > 50%)\n\n'''\n\n"
            "if poor > 0:\n"
            "    md += f'- **{{poor}} segments** have high WAPE - investigate data quality\\n'\n"
            "    md += '- Consider alternative model families (zero-inflated, hurdle)\\n'\n"
            "    md += '- Review feature engineering for these segments\\n\\n'\n"
            "else:\n"
            "    md += 'All segments performing within acceptable thresholds.\\n\\n'\n\n"
            "md += '''---\\n\\n"
            "*Generated by Training Insights Agent using iterative code execution.*\\n"
            "'''\n\n"
            "# SAVE THE REPORT TO FILE\n"
            "report_path = os.path.join(model_dir, 'MODEL_TRAINING_INSIGHTS_GUIDE.md')\n"
            "with open(report_path, 'w') as f:\n"
            "    f.write(md)\n\n"
            "print(f'Saved: MODEL_TRAINING_INSIGHTS_GUIDE.md ({{len(md):,}} bytes)')\n"
            "print(f'Overall WAPE: {{overall_wape:.1%}} ({{quality_tier}})')\n"
            "print(f'Models: {{n_models}}, Ensembles: {{n_ensembles}}')\n"
            "```\n\n"
            "## CRITICAL INSTRUCTIONS\n\n"
            "1. You MUST execute code using CodeExecutionTool\n"
            "2. Run Phases 1-4 to gather statistics BEFORE Phase 5\n"
            "3. In Phase 5, the code uses the ACTUAL numbers loaded from files\n"
            "4. The report MUST be saved to file (not just printed)\n"
            "5. Final report should be comprehensive with specific data"
        ),
        agent=documentation_agent,
        expected_output=(
            "Created MODEL_TRAINING_INSIGHTS_GUIDE.md - comprehensive documentation with "
            "performance metrics, state-of-the-art features, and segment analysis."
        ),
        output_file=os.path.join(model_dir_out, "training_documentation_report.md"),
        context=[task_plan, task_execute, task_analyze],
    )

    # -------------------------------------------------------------------------
    # OPTIONAL Tasks: Documentation & Reviewer
    # -------------------------------------------------------------------------
    enable_insights = getattr(config.design, 'enable_insights_reports', False)
    enable_reviewer = getattr(config.design, 'enable_reviewer', False)

    agents = [planner, executor, analyst]
    tasks = [task_plan, task_execute, task_analyze]

    if enable_insights:
        agents.append(documentation_agent)
        tasks.append(task_document)
    else:
        logger.info("SKIPPING training insights report (enable_insights_reports=False)")

    if enable_reviewer:
        reviewer = _create_training_reviewer_agent(llm, protected_paths=protected_dirs)
        task_review = Task(
            name="review_training_outputs",
            description=(
                "# TRAINING OUTPUT QUALITY REVIEW\n\n"
                "You are the Training Reviewer. Validate all outputs from the Training crew.\n\n"
                "## VALIDATION STEPS\n"
                "Write and execute code to validate:\n"
                "```python\n"
                "import os\n"
                "import json\n"
                f"model_dir = '{model_dir}'\n"
                "\n"
                "# Check required files exist\n"
                "required_files = [\n"
                "    'final_model_specs.json',\n"
                "    'training_to_diagnostic_context.json',\n"
                "    'training_strategy.json',\n"
                "]\n"
                "missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]\n"
                "print(f'Missing files: {missing}' if missing else 'All required files present')\n"
                "\n"
                "# Check model files exist\n"
                "model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]\n"
                "print(f'Model files: {len(model_files)}')\n"
                "\n"
                "# Validate WAPE metrics\n"
                "with open(os.path.join(model_dir, 'final_model_specs.json')) as f:\n"
                "    specs = json.load(f)\n"
                "overall_wape = specs.get('overall_wape', 0)\n"
                "if overall_wape > 1.0:\n"
                "    print(f'WARNING: Overall WAPE > 100%: {overall_wape:.1%}')\n"
                "else:\n"
                "    print(f'Overall WAPE: {overall_wape:.1%}')\n"
                "```\n\n"
                "## OUTPUT\n"
                f"Create `{model_dir}/training_review_report.json` with:\n"
                "- quality_score (1-10)\n"
                "- files_validated (list)\n"
                "- wape_validation (pass/fail)\n"
                "- model_selection_validation (pass/fail)\n"
                "- issues_found (list)\n"
                "- recommendations (list)\n"
            ),
            agent=reviewer,
            expected_output=(
                "Created training_review_report.json with quality score and validation results. "
                "All model files validated. Issues identified and documented."
            ),
            output_file=os.path.join(model_dir_out, "training_review_summary.md"),
            context=[task_analyze],
        )
        agents.append(reviewer)
        tasks.append(task_review)
        crew_name = "Training Crew (4-Agent Pattern with Reviewer)"
    else:
        crew_name = "Training Crew (3-Agent Pattern)"

    return Crew(
        name=crew_name,
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def run_training_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> TrainingCrewResult:
    """Run the training crew and return results."""
    from utils.cost_tracking import get_cost_tracker, extract_tokens_from_crew_result

    # ==========================================================================
    # CRITICAL: Validate input data file exists BEFORE running any crew tasks
    # ==========================================================================
    if not os.path.exists(config.input_data_path):
        raise TrainingFailedError(
            f"Input data file not found: {config.input_data_path}\n"
            "Please ensure the data file exists at the configured path."
        )

    # Start cost tracking
    tracker = get_cost_tracker()
    tracker.start_crew("Training Crew")

    # Get model ID from LLM if available
    model_id = getattr(llm, "model", "default")
    tracker.set_model(model_id)

    # ======================================================================
    # WRITE PROTECTION: Set protected_paths on the LLM's CodeExecutionTool
    # ======================================================================
    eda_dir = os.path.join(config.artifact_base_path, "eda_output")
    seg_dir = os.path.join(config.artifact_base_path, "seg_output")
    feat_dir = os.path.join(config.artifact_base_path, "feature_output")
    _code_tool = getattr(llm, '_code_execution_tool', None)
    if _code_tool is not None:
        _code_tool.protected_paths = [eda_dir, seg_dir, feat_dir]
        logger.info(f"Write protection set on LLM CodeExecutionTool: eda_output, seg_output, feature_output are READ-ONLY")

    crew = create_training_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)
    crew_result = crew.kickoff()

    model_dir = os.path.join(config.artifact_base_path, "model_artifacts")

    # Extract and record tokens from crew result
    tokens = extract_tokens_from_crew_result(crew_result)
    if tokens["total"] > 0:
        tracker.record_llm_call(
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            model=model_id,
        )

    # End tracking and save cost report
    cost_report = tracker.end_crew("Training Crew", model_dir)
    cost_report_path = os.path.join(model_dir, "training_cost.json")

    # CRITICAL: Validate that training actually completed successfully
    # This will raise TrainingFailedError if validation fails, causing the pipeline to fail
    # instead of silently continuing with no trained models
    logger.info("Validating training completion...")
    _validate_training_completed(model_dir)
    logger.info("Training validation passed - models were successfully trained")

    # Build result object with paths to all outputs
    return TrainingCrewResult(
        final_model_specs_path=os.path.join(model_dir, "final_model_specs.json"),
        model_selection_report_markdown_path=os.path.join(model_dir, "training_analysis_report.md"),
        model_dir=model_dir,
        training_pipeline_script_path=os.path.join(model_dir, "training_planning_report.md"),
        training_to_diagnostic_context_path=os.path.join(model_dir, "training_to_diagnostic_context.json"),
        training_deterministic_code_path=os.path.join(model_dir, "training_deterministic.py"),
        cost_report_path=cost_report_path,
    )
