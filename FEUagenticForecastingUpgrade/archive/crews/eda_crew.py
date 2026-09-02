# crews/eda_crew.py
"""
State-of-the-Art EDA Crew for Demand Forecasting.

This crew uses the 3-Agent Pattern with UTILITY-FIRST code execution:
1. EDA Planner: Discovers data characteristics dynamically and creates a comprehensive
   analysis plan with SPECIFIC instructions for what analyses to perform
2. EDA Executor: Uses pre-built utilities from utils/agent_utilities.py for all
   standard operations - Syntetos-Boylan classification, stationarity testing,
   feature importance, visualization. Only writes custom code when no utility exists.
3. EDA Analyst: Reads outputs and creates TARGETED context files for downstream crews
   - Segmentation guidance with clustering recommendations
   - Feature engineering guidance with lag/rolling window suggestions
   - Model family recommendations by demand pattern

Key Capabilities:
- Uses utils/agent_utilities.py for all standard EDA operations
- Pre-built functions for demand classification, stationarity testing, visualization
- Adapts to ANY demand forecasting dataset dynamically
- Generates state-of-the-art analyses with minimal code via utility calls

The crew reads config.yaml to understand features and MUST use utility functions.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from crewai import Agent, Crew, Task, Process, LLM

from config.schema import DemandForecastConfig
from utils.code_execution_tool import CodeExecutionTool

logger = logging.getLogger(__name__)


class EDAFailedError(Exception):
    """Raised when EDA fails to produce required outputs."""
    pass


def _validate_executor_output(eda_dir: str) -> None:
    """
    Validate that Executor task created all required EDA files WITH CORRECT CONTENT.
    Called as callback after Executor completes.

    This validates that the LLM actually ran the provided run_core_eda code
    and didn't invent its own simplified version.
    """
    import pandas as pd

    required_files = [
        ('per_key_metrics.csv', 'Per-series metrics for all time series'),
        ('eda_summary.json', 'EDA summary statistics'),  # Fixed: was global_eda_summary.json
    ]

    missing = []
    for filename, description in required_files:
        filepath = os.path.join(eda_dir, filename)
        if not os.path.exists(filepath):
            missing.append(f"  - {filename}: {description}")

    if missing:
        raise EDAFailedError(
            f"CRITICAL: EDA Executor FAILED to create required files\n\n"
            f"Missing files in {eda_dir}:\n" +
            "\n".join(missing) + "\n\n"
            "ROOT CAUSE: The LLM agent did NOT execute the code block.\n"
            "The agent likely described what should happen instead of running CodeExecutionTool.\n\n"
            "SOLUTION: Check eda_execution_report.md for the task output.\n"
            "If it contains description instead of execution output, this confirms the issue.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )

    # ==========================================================================
    # CRITICAL: Validate per_key_metrics.csv has RICH content
    # ==========================================================================
    per_key_path = os.path.join(eda_dir, 'per_key_metrics.csv')
    df = pd.read_csv(per_key_path)

    # Minimum required columns for proper model level allocation
    # Note: demand_pattern OR intermittency_class are acceptable (same semantic meaning)
    required_cols = ['mean', 'cv', 'zero_fraction', 'adi', 'volume_tier']
    pattern_cols = ['demand_pattern', 'intermittency_class']  # Either is acceptable

    missing_cols = [c for c in required_cols if c not in df.columns]
    has_pattern_col = any(c in df.columns for c in pattern_cols)

    if not has_pattern_col:
        missing_cols.append('demand_pattern (or intermittency_class)')

    if missing_cols:
        raise EDAFailedError(
            f"CRITICAL: per_key_metrics.csv is INCOMPLETE!\n\n"
            f"Missing columns: {missing_cols}\n"
            f"Actual columns ({len(df.columns)}): {list(df.columns)}\n\n"
            "ROOT CAUSE: The LLM agent wrote its own simplified EDA code\n"
            "instead of using the provided run_core_eda() function.\n\n"
            "The run_core_eda function produces 20+ columns including:\n"
            "mean, std, cv, adi, cv2, zero_fraction, demand_pattern, volume_tier,\n"
            "skewness, kurtosis, autocorr_lag1, trend_strength, seasonal_strength,\n"
            "predictability_score, and many more.\n\n"
            "SOLUTION: Re-run and ensure the agent executes the EXACT code block provided."
        )

    if len(df.columns) < 15:
        logger.warning(
            f"per_key_metrics.csv has only {len(df.columns)} columns. "
            f"Expected 20+. Agent may have used simplified code."
        )

    logger.info(f"EDA Executor validation passed: per_key_metrics.csv has {len(df.columns)} columns")
    logger.info(f"  Columns include: {list(df.columns)[:10]}...")


def _validate_analyst_output(eda_dir: str) -> None:
    """
    Validate that Analyst task created the context files for downstream crews.
    Called as callback after Analyst completes.
    """
    required_files = [
        ('eda_to_segmentation_context.json', 'Context for Segmentation crew'),
        ('eda_to_feature_context.json', 'Context for Feature Engineering crew'),
        ('eda_to_training_context.json', 'Context for Training crew'),
    ]

    missing = []
    for filename, description in required_files:
        filepath = os.path.join(eda_dir, filename)
        if not os.path.exists(filepath):
            missing.append(f"  - {filename}: {description}")

    if missing:
        raise EDAFailedError(
            f"CRITICAL: EDA Analyst FAILED to create context files\n\n"
            f"Missing files in {eda_dir}:\n" +
            "\n".join(missing) + "\n\n"
            "ROOT CAUSE: The LLM agent did NOT execute the code block.\n"
            "The agent likely described what should happen instead of running CodeExecutionTool.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )
    logger.info(f"EDA Analyst validation passed: all context files exist in {eda_dir}")


@dataclass
class EDACrewResult:
    eda_dir: str
    per_key_metrics_path: str
    global_eda_summary_path: str
    enhanced_eda_summary_path: str
    segmentation_suggestions_path: str
    feature_importances_rf_path: str
    feature_correlation_matrix_path: str
    # Advanced EDA outputs
    advanced_per_key_metrics_path: str
    granger_causality_path: str
    shap_interactions_path: str
    horizon_specific_importance_path: str
    cross_series_correlation_path: str
    charts_dir_path: str
    eda_report_markdown_path: str
    # New: Generated pipeline and context
    eda_pipeline_script_path: str = ""
    eda_to_segmentation_context_path: str = ""
    # DETERMINISTIC CODE OUTPUT for Pipeline Generator
    eda_deterministic_code_path: str = ""
    # LLM-generated exhaustive insights report
    eda_insights_report_path: str = ""
    # Cost tracking
    cost_report_path: str = ""


def _get_output_path(absolute_path: str) -> str:
    """
    Get a safe path for CrewAI Task output_file parameter.

    CrewAI 1.9.1+ rejects paths with '..' (path traversal) for security.
    On Databricks, relative paths from cwd to /Volumes/ create '../..' paths.

    Solution: Use absolute paths which are safe and always work.
    """
    # Return absolute path - CrewAI accepts these without security issues
    return os.path.abspath(absolute_path)


def _generate_enhanced_config_context(config: DemandForecastConfig) -> Dict[str, Any]:
    """
    Generate a comprehensive configuration context for the EDA agents.

    This extracts all feature groups and schema information from the config
    to enable fully dynamic, config-driven EDA.

    CRITICAL: This defines the ONLY columns agents are allowed to use.
    """
    # Extract all feature groups
    price_features_numeric = getattr(config.price_features, 'numeric', [])
    price_features_categorical = getattr(config.price_features, 'categorical', [])
    promo_features_numeric = getattr(config.promo_features, 'numeric', [])
    promo_features_categorical = getattr(config.promo_features, 'categorical', [])
    holiday_features_numeric = getattr(config.holiday_features, 'numeric', [])
    holiday_features_categorical = getattr(config.holiday_features, 'categorical', [])
    weather_features = getattr(config.weather_features, 'numeric', [])

    # Get ALL allowed columns from config
    all_allowed_cols = config.all_allowed_columns()
    feature_cols_only = config.get_feature_columns_only()

    return {
        "schema": {
            "data_path": config.input_data_path,
            "timestamp_col": config.timestamp_col,
            "target_col": config.target_col,
            "key_columns": list(config.prediction_key_cols),
            "time_format": getattr(config, 'time_format', 'year_week'),
            "forecast_horizon": getattr(config, 'forecast_horizon', 8),
            # Train/Val/Test splits
            "train_start": config.train_start,
            "train_end": config.train_end,
            "val_start": config.val_start,
            "val_end": config.val_end,
            "test_start": config.test_start,
            "test_end": config.test_end,
        },
        # CRITICAL: These are the ONLY columns agents can use
        "allowed_columns": {
            "all_columns": all_allowed_cols,
            "feature_columns_only": feature_cols_only,
            "numeric_features": config.all_numeric_features(),
            "categorical_features": config.all_categorical_features(),
        },
        "feature_groups": {
            "price_features_numeric": price_features_numeric,
            "price_features_categorical": price_features_categorical,
            "price_features": price_features_numeric,  # Legacy alias for backward compatibility
            "promo_features_numeric": promo_features_numeric,
            "promo_features_categorical": promo_features_categorical,
            "holiday_features_numeric": holiday_features_numeric,
            "holiday_features_categorical": holiday_features_categorical,
            "weather_features": weather_features,
        },
        "analysis_recommendations": {
            "enable_promo_analysis": len(promo_features_numeric + promo_features_categorical) > 0,
            "enable_price_analysis": len(price_features_numeric + price_features_categorical) > 0,
            "enable_holiday_analysis": len(holiday_features_numeric + holiday_features_categorical) > 0,
            "enable_weather_analysis": len(weather_features) > 0,
        },
        "decomposition_period": 52 if getattr(config, 'time_format', 'year_week') == 'year_week' else 12,
    }


def _create_eda_planner_agent(llm: LLM, config: DemandForecastConfig) -> Agent:
    """
    Create the EDA Planner agent - minimal role, just validates data exists.
    """
    code_tool = CodeExecutionTool()

    return Agent(
        name="eda_planner",
        role="EDA Data Validator",
        goal=(
            "Validate the data file exists and check basic data shape. "
            "Then tell the Executor to run run_eda_pipeline()."
        ),
        backstory=(
            "You validate data exists before EDA runs.\n\n"
            "## YOUR ONLY JOB:\n"
            "1. Check data file exists\n"
            "2. Load and print shape (1 line)\n"
            "3. Tell Executor to use run_eda_pipeline()\n\n"
            "## OUTPUT LIMIT: MAX 5 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_eda_executor_agent(llm: LLM, config: DemandForecastConfig) -> Agent:
    """
    Create the EDA Executor agent - runs run_eda_pipeline() in ONE call.
    """
    code_tool = CodeExecutionTool()

    return Agent(
        name="eda_executor",
        role="EDA Pipeline Executor",
        goal=(
            "Run run_eda_pipeline() from utils/eda.py to execute the ENTIRE EDA in ONE call. "
            "Do NOT write custom analysis code."
        ),
        backstory=(
            "You execute EDA using ONE pre-built function.\n\n"
            "## 🚨 USE run_eda_pipeline() - IT DOES EVERYTHING 🚨\n\n"
            "```python\n"
            "from utils.agent_utilities import load_csv\n"
            "from utils.eda import run_eda_pipeline\n\n"
            "df = load_csv(DATA_PATH)\n"
            "result = run_eda_pipeline(\n"
            "    df=df,\n"
            "    key_columns=KEY_COLUMNS,\n"
            "    date_col=DATE_COL,\n"
            "    target_col=TARGET_COL,\n"
            "    numeric_features=numeric_cols,\n"
            "    output_dir=OUTPUT_DIR,\n"
            "    verbose=False\n"
            ")\n"
            "print(f'EDA complete: {result[\"summary\"][\"total_series\"]} series')\n"
            "```\n\n"
            "## THIS ONE FUNCTION CREATES:\n"
            "- per_key_metrics.csv (Syntetos-Boylan classification)\n"
            "- stationarity_results.csv (ADF + KPSS tests)\n"
            "- feature_importance.csv (RF + Corr + MI ensemble)\n"
            "- eda_summary.json, eda_report.md\n"
            "- charts/syntetos_boylan.png, charts/demand_patterns.png\n\n"
            "## DO NOT:\n"
            "- Write custom ADI/CV² calculations\n"
            "- Write custom stationarity tests\n"
            "- Call individual functions - use run_eda_pipeline()\n\n"
            "## OUTPUT LIMIT: MAX 5 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_eda_analyst_agent(llm: LLM, config: DemandForecastConfig, allowed_model_families: list, enable_deep_models: bool) -> Agent:
    """
    Create the EDA Analyst agent - a PhD-level expert who autonomously interprets
    EDA results and creates intelligent recommendations for downstream crews.
    """
    code_tool = CodeExecutionTool()

    return Agent(
        name="eda_analyst",
        role="PhD-Level Demand Forecasting & Data Preparation Expert",
        goal=(
            "Read EDA outputs, extract key metrics, and create 2 context JSON files "
            "with intelligent recommendations. MINIMAL PRINT OUTPUT."
        ),
        backstory=(
            "You are a PhD-level expert in demand forecasting with 15+ years of experience.\n\n"
            "## CRITICAL: MINIMAL OUTPUT RULES\n"
            "######################################################################\n"
            "#  MAXIMUM 5 PRINT STATEMENTS TOTAL - NO EXCEPTIONS                  #\n"
            "#  NEVER print DataFrames, dicts, lists, or file contents            #\n"
            "#  NEVER use print() inside loops                                    #\n"
            "#  SUPPRESS ALL WARNINGS: warnings.filterwarnings('ignore')          #\n"
            "######################################################################\n\n"
            "## YOUR EXPERTISE:\n"
            "- Syntetos-Boylan: ADI>1.32 = intermittent, CV²>0.49 = erratic\n"
            "- Stationarity → model selection implications\n"
            "- Feature importance → engineering priorities\n\n"
            "## INTERPRETATION FRAMEWORK (FEATURE-BASED MODELS ONLY):\n"
            "- High lumpy/intermittent % → zero_inflated, hurdle_model, tweedie with features\n"
            "- High non-stationary % → trend features, differenced lags\n"
            "- High CV (>1.5) → log transform, robust scaling, XGBoost with regularization\n"
            "- BANNED: croston, sba, tsb, imapa, arima, ets, theta, prophet (univariate)\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n\n"
            "## SAFE CODE PATTERN:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json, pandas as pd\n"
            "# Read files silently, extract metrics, save JSON\n"
            "# ONLY 5 print() calls allowed!\n"
            "```"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_eda_reviewer_agent(llm: LLM, config: DemandForecastConfig) -> Agent:
    """
    Create the EDA Reviewer agent that validates and quality-checks the EDA outputs.
    This agent is OPTIONAL and only used when config.design.enable_reviewer is True.
    """
    code_tool = CodeExecutionTool()

    return Agent(
        name="eda_reviewer",
        role="EDA Quality Assurance & Validation Specialist",
        goal=(
            "Review and validate all EDA outputs for completeness, accuracy, and actionability. "
            "Identify any gaps, inconsistencies, or issues that need correction before "
            "downstream crews consume the EDA context files."
        ),
        backstory=(
            "You are a senior quality assurance specialist with deep expertise in "
            "data science and demand forecasting.\n\n"
            "######################################################################\n"
            "#  OUTPUT LIMIT: MAXIMUM 10 PRINT STATEMENTS                        #\n"
            "######################################################################\n"
            "```python\n"
            "# Validation pattern:\n"
            "import os, json\n"
            "files = ['file1.json', 'file2.csv']\n"
            "exists = [f for f in files if os.path.exists(f)]\n"
            "print(f'Files validated: {len(exists)}/{len(files)}')\n\n"
            "# Load and check keys (NOT print contents)\n"
            "with open('context.json') as f: d = json.load(f)\n"
            "print(f'Context keys present: {list(d.keys())}')\n\n"
            "# Save report\n"
            "report = {'score': 8, 'issues': []}\n"
            "with open('review.json', 'w') as f: json.dump(report, f)\n"
            "print('Saved: review.json')\n"
            "```\n"
            "######################################################################\n\n"
            "## YOUR MISSION\n"
            "1. **Validate Context Files**: Check eda_to_segmentation_context.json and "
            "   eda_to_feature_context.json contain all required fields.\n"
            "2. **Verify Recommendations**: Ensure model recommendations match patterns.\n"
            "3. **Quality Score**: Assign score (1-10) and save to eda_review_report.json."
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_eda_documentation_agent(llm: LLM) -> Agent:
    """
    Create the EDA Documentation Agent that generates comprehensive markdown documentation.
    This agent reads all EDA outputs and creates a detailed guide for data scientists.
    """
    code_tool = CodeExecutionTool()

    return Agent(
        name="eda_documentation_agent",
        role="EDA Insights Documentation Specialist",
        goal=(
            "EXECUTE Python code using CodeExecutionTool to create EDA_INSIGHTS_GUIDE.md. "
            "You MUST use the tool to run the code - do NOT just describe or print the markdown."
        ),
        backstory=(
            "You are an expert data science communicator.\n\n"
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE CodeExecutionTool TO RUN THE CODE BLOCK   #\n"
            "#  DO NOT just print or describe the markdown - SAVE IT TO FILE!    #\n"
            "######################################################################\n\n"
            "## HOW TO COMPLETE THIS TASK:\n"
            "1. Use CodeExecutionTool to execute the Python code in the task\n"
            "2. The code will SAVE the markdown to EDA_INSIGHTS_GUIDE.md\n"
            "3. Do NOT print the entire markdown content\n"
            "4. Only print confirmation messages (max 5 lines)\n\n"
            "## WRONG APPROACH (DO NOT DO THIS):\n"
            "- Printing the markdown content to stdout\n"
            "- Describing what the code would do\n"
            "- Showing the markdown without saving\n\n"
            "## CORRECT APPROACH:\n"
            "- Execute the code block using CodeExecutionTool\n"
            "- Code saves to file with: open(doc_path, 'w').write(md)\n"
            "- Print only: 'Saved: EDA_INSIGHTS_GUIDE.md'"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def create_eda_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> Crew:
    """
    Create the State-of-the-Art EDA Crew with the 3-Agent Pattern.

    This crew uses pre-built utilities from utils/agent_utilities.py for all
    standard EDA operations, minimizing code generation and context usage.

    The crew has three phases:
    1. Planning: Discovers data characteristics and creates DETAILED SPECIFICATIONS
       for state-of-the-art EDA analyses.
    2. Execution: Uses utility functions for Syntetos-Boylan classification,
       stationarity testing, visualization, etc. Only writes custom code when
       no utility exists for a specific operation.
    3. Analysis: Extracts TARGETED insights and creates ACTIONABLE context files
       for Segmentation and Feature Engineering crews.

    Key Capabilities:
    - Uses utils/agent_utilities.py for: compute_demand_characteristics,
      add_demand_classification, test_stationarity, create_bar_chart, etc.
    - Minimal code generation - most operations are single utility calls
    - Adapts to ANY demand forecasting dataset dynamically
    - Creates targeted guidance for downstream crews
    """
    artifact_base = config.artifact_base_path
    eda_dir = os.path.join(artifact_base, "eda_output")
    charts_dir = os.path.join(eda_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Get safe output path for CrewAI Task output_file
    # Note: CrewAI 1.9.1+ rejects relative paths with '..' for security
    eda_dir_out = _get_output_path(eda_dir)

    # Generate enhanced configuration context
    config_context = _generate_enhanced_config_context(config)

    # Get allowed model families from config.design (MUST be before agent creation)
    allowed_model_families = list(config.design.model_families)
    enable_deep_models = config.design.enable_deep_models

    # Filter out deep learning models if disabled
    deep_model_types = ['tft', 'lstm', 'nbeats', 'deepar', 'wavenet']
    if not enable_deep_models:
        allowed_model_families = [m for m in allowed_model_families if m.lower() not in deep_model_types]

    # Create agents
    planner = _create_eda_planner_agent(llm, config)
    executor = _create_eda_executor_agent(llm, config)
    analyst = _create_eda_analyst_agent(llm, config, allowed_model_families, enable_deep_models)

    # Get config details for the planner
    data_path = config.input_data_path
    date_col = config.date_column
    target_col = config.target_column
    key_columns = config.key_columns
    time_format = getattr(config, 'time_format', 'year_week')
    forecast_horizon = getattr(config, 'forecast_horizon', 8)
    numeric_features = config.all_numeric_features()
    categorical_features = config.all_categorical_features()

    # Get domain-specific features from config
    price_features_numeric = config_context["feature_groups"]["price_features_numeric"]
    price_features_categorical = config_context["feature_groups"]["price_features_categorical"]
    promo_features_numeric = config_context["feature_groups"]["promo_features_numeric"]
    promo_features_categorical = config_context["feature_groups"]["promo_features_categorical"]
    holiday_features = config_context["feature_groups"]["holiday_features_numeric"] + config_context["feature_groups"]["holiday_features_categorical"]
    weather_features = config_context["feature_groups"]["weather_features"]

    # Get train/val/test date ranges from config
    train_start = config.train_start
    train_end = config.train_end
    val_start = config.val_start
    val_end = config.val_end
    test_start = config.test_start
    test_end = config.test_end

    # Build the complete list of allowed columns
    all_allowed_columns = (
        list(key_columns) +
        [date_col, target_col] +
        numeric_features +
        categorical_features
    )

    # NOTE: No script template - agents generate ALL code autonomously

    # -------------------------------------------------------------------------
    # Task 1: EDA Planning - Just validate data exists
    # -------------------------------------------------------------------------
    task_plan = Task(
        name="validate_data",
        description=(
            "# VALIDATE DATA EXISTS\n\n"
            "Check the data file exists and print basic info.\n\n"
            "```python\n"
            "import os\n"
            "import pandas as pd\n\n"
            f"data_path = '{data_path}'\n"
            "if os.path.exists(data_path):\n"
            "    df = pd.read_csv(data_path)\n"
            f"    n_series = df.groupby({key_columns}).ngroups\n"
            "    print(f'Data validated: {len(df)} rows, {n_series} series')\n"
            "else:\n"
            "    print(f'ERROR: File not found: {data_path}')\n"
            "```\n\n"
            "Then tell Executor: 'Run run_eda_pipeline() with the config values.'"
        ),
        agent=planner,
        expected_output=(
            "Data validated: N rows, M series. Executor should run run_eda_pipeline()."
        ),
        output_file=os.path.join(eda_dir_out, "eda_strategy.md"),
    )

    # -------------------------------------------------------------------------
    # Task 2: EDA Execution - ONE CALL with ALL features on TRAINING DATA
    # -------------------------------------------------------------------------
    task_execute = Task(
        name="execute_eda_pipeline",
        description=(
            "# RUN EDA PIPELINE ON TRAINING DATA\n\n"
            "Execute this code to run the complete EDA with ALL features:\n\n"
            "```python\n"
            "import numpy as np\n"
            "from utils.agent_utilities import load_csv\n"
            "from utils.eda import run_eda_pipeline\n\n"
            f"df = load_csv('{data_path}')\n\n"
            "# FILTER TO TRAINING PERIOD ONLY (from config)\n"
            f"date_col = '{date_col}'\n"
            f"train_start_raw = '{train_start}'\n"
            f"train_end_raw = '{train_end}'\n\n"
            "# Infer dtype and convert train_start/train_end to match\n"
            "col_dtype = df[date_col].dtype\n"
            "if np.issubdtype(col_dtype, np.integer):\n"
            "    train_start = int(train_start_raw)\n"
            "    train_end = int(train_end_raw)\n"
            "elif np.issubdtype(col_dtype, np.floating):\n"
            "    train_start = float(train_start_raw)\n"
            "    train_end = float(train_end_raw)\n"
            "else:\n"
            "    train_start = str(train_start_raw)\n"
            "    train_end = str(train_end_raw)\n\n"
            "df = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)].copy()\n"
            "print(f'Training data: {len(df)} rows ({train_start} to {train_end})')\n\n"
            "# ALL numeric features from config\n"
            f"numeric_features = {numeric_features}\n\n"
            "# ALL categorical features from config\n"
            f"categorical_features = {categorical_features}\n\n"
            "# Filter to columns that exist in the data\n"
            "numeric_features = [c for c in numeric_features if c in df.columns]\n"
            "categorical_features = [c for c in categorical_features if c in df.columns]\n\n"
            "# Price and promo features from config (for external feature aggregation)\n"
            f"price_features_numeric = [c for c in {price_features_numeric} if c in df.columns]\n"
            f"price_features_categorical = [c for c in {price_features_categorical} if c in df.columns]\n"
            f"promo_features_numeric = [c for c in {promo_features_numeric} if c in df.columns]\n"
            f"promo_features_categorical = [c for c in {promo_features_categorical} if c in df.columns]\n\n"
            "result = run_eda_pipeline(\n"
            "    df=df,\n"
            f"    key_columns={key_columns},\n"
            f"    date_col='{date_col}',\n"
            f"    target_col='{target_col}',\n"
            "    numeric_features=numeric_features,\n"
            "    categorical_features=categorical_features,\n"
            f"    output_dir='{eda_dir}',\n"
            f"    period={52 if time_format == 'year_week' else 12},\n"
            "    verbose=False,\n"
            f"    train_end=train_end_raw,  # CRITICAL: Pass train_end for dead key detection\n"
            "    dead_key_threshold=26,  # Keys with 26+ consecutive zeros at end are 'dead'\n"
            "    # Config-driven external features for segmentation-aware EDA\n"
            "    price_features_numeric=price_features_numeric,\n"
            "    price_features_categorical=price_features_categorical,\n"
            "    promo_features_numeric=promo_features_numeric,\n"
            "    promo_features_categorical=promo_features_categorical,\n"
            ")\n\n"
            "# Check if pipeline succeeded\n"
            "if result.get('status') == 'error':\n"
            "    print(f'EDA ERROR: {result.get(\"error\", \"Unknown error\")}')\n"
            "else:\n"
            "    summary = result.get('summary', {})\n"
            "    dead_info = summary.get('dead_key_summary', {})\n"
            "    print(f'EDA complete: {summary.get(\"total_series\", \"?\")} active series')\n"
            "    print(f'Dead keys excluded: {dead_info.get(\"dead_keys\", 0)} ({dead_info.get(\"dead_key_percentage\", 0):.1f}%)')\n"
            "    print(f'Lumpy/intermittent: {summary.get(\"lumpy_intermittent_pct\", 0):.1%}')\n"
            "    print(f'Files created: {len(result.get(\"files_created\", []))}')\n"
            "```\n\n"
            "This creates all output files automatically. Do NOT write additional analysis code."
        ),
        agent=executor,
        expected_output=(
            "EDA complete on training data. N series analyzed. "
            "Files created in output directory."
        ),
        output_file=os.path.join(eda_dir_out, "eda_execution_report.md"),
        context=[task_plan],
        # CRITICAL: Callback to validate Executor output BEFORE Analyst starts
        callback=lambda output: _validate_executor_output(eda_dir),
    )

    # -------------------------------------------------------------------------
    # Task 3: EDA Analysis - Create INTELLIGENT context files from raw outputs
    # The agent reads ALL exhaustive analysis outputs and creates SMART context
    # files for downstream crews (segmentation, feature engineering, training)
    # -------------------------------------------------------------------------
    task_analyze = Task(
        name="create_intelligent_context",
        description=(
            "# CREATE INTELLIGENT CONTEXT FILES FOR DOWNSTREAM CREWS\n\n"
            "######################################################################\n"
            "#  MAXIMUM 5 PRINT STATEMENTS - SUPPRESS ALL WARNINGS                #\n"
            "######################################################################\n\n"
            f"## EDA OUTPUT DIRECTORY: `{eda_dir}`\n\n"
            "## YOUR JOB: Read ALL EDA outputs and create 3 intelligent context JSON files\n\n"
            "## RAW FILES TO ANALYZE (Core + Exhaustive):\n"
            "**Core Outputs:**\n"
            "- `per_key_metrics.csv` - Per-series metrics (cv, adi, cv2, mean, std, volume_tier, etc.)\n"
            "- `stationarity_results.csv` - Per-series stationarity (verdict, adf_stat, kpss_stat)\n"
            "- `feature_importance.csv` - Features ranked by ensemble_score\n"
            "- `eda_summary.json` - Aggregated statistics\n"
            "- `data_profile.json` - Comprehensive data profile (if exists)\n\n"
            "**Exhaustive Analysis Outputs (if available):**\n"
            "- `autocorrelation_summary.csv` - ACF/PACF analysis with significant lags\n"
            "- `seasonality_analysis.json` - Seasonality detection results\n"
            "- `trend_analysis.json` - Trend strength distribution\n"
            "- `changepoint_analysis.json` - Structural change detection\n\n"
            "## CONTEXT FILES TO CREATE:\n"
            "1. `eda_to_segmentation_context.json` - For Segmentation Crew\n"
            "2. `eda_to_feature_context.json` - For Feature Engineering Crew\n"
            "3. `eda_to_training_context.json` - For Training Crew\n\n"
            "## EXECUTION CODE - RUN THIS EXACTLY:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n"
            "import pandas as pd\n"
            "import numpy as np\n\n"
            f"eda_dir = '{eda_dir}'\n"
            f"time_format = '{time_format}'\n"
            f"allowed_models = {allowed_model_families}\n"
            f"numeric_features = {numeric_features}\n"
            f"categorical_features = {categorical_features}\n\n"
            "# =================================================================\n"
            "# READ ALL OUTPUTS (Core + Exhaustive)\n"
            "# =================================================================\n"
            "# Core outputs\n"
            "metrics = pd.read_csv(os.path.join(eda_dir, 'per_key_metrics.csv'))\n"
            "stationarity_path = os.path.join(eda_dir, 'stationarity_results.csv')\n"
            "stationarity = pd.read_csv(stationarity_path) if os.path.exists(stationarity_path) else pd.DataFrame()\n"
            "feat_imp_path = os.path.join(eda_dir, 'feature_importance.csv')\n"
            "feat_imp = pd.read_csv(feat_imp_path) if os.path.exists(feat_imp_path) else pd.DataFrame()\n"
            "with open(os.path.join(eda_dir, 'eda_summary.json')) as f:\n"
            "    summary = json.load(f)\n\n"
            "# Exhaustive analysis outputs (if available)\n"
            "data_profile = {}\n"
            "seasonality = {}\n"
            "trend_analysis = {}\n"
            "changepoint_analysis = {}\n"
            "acf_summary = pd.DataFrame()\n\n"
            "if os.path.exists(os.path.join(eda_dir, 'data_profile.json')):\n"
            "    with open(os.path.join(eda_dir, 'data_profile.json')) as f:\n"
            "        data_profile = json.load(f)\n"
            "if os.path.exists(os.path.join(eda_dir, 'seasonality_analysis.json')):\n"
            "    with open(os.path.join(eda_dir, 'seasonality_analysis.json')) as f:\n"
            "        seasonality = json.load(f)\n"
            "if os.path.exists(os.path.join(eda_dir, 'trend_analysis.json')):\n"
            "    with open(os.path.join(eda_dir, 'trend_analysis.json')) as f:\n"
            "        trend_analysis = json.load(f)\n"
            "if os.path.exists(os.path.join(eda_dir, 'changepoint_analysis.json')):\n"
            "    with open(os.path.join(eda_dir, 'changepoint_analysis.json')) as f:\n"
            "        changepoint_analysis = json.load(f)\n"
            "if os.path.exists(os.path.join(eda_dir, 'autocorrelation_summary.csv')):\n"
            "    acf_summary = pd.read_csv(os.path.join(eda_dir, 'autocorrelation_summary.csv'))\n\n"
            "n_series = len(metrics)\n"
            "period = 52 if time_format == 'year_week' else 12\n\n"
            "# =================================================================\n"
            "# ANALYZE DATA CHARACTERISTICS\n"
            "# =================================================================\n"
            "# Compute demand distribution\n"
            "demand_dist = metrics['intermittency_class'].value_counts(normalize=True).to_dict() if 'intermittency_class' in metrics.columns else {}\n\n"
            "# Compute stationarity distribution\n"
            "stat_dist = stationarity['verdict'].value_counts(normalize=True).to_dict() if 'verdict' in stationarity.columns else {}\n\n"
            "# Get seasonality and trend info\n"
            "has_seasonality_pct = seasonality.get('has_seasonality_pct', 0) if seasonality else 0\n"
            "dominant_period = seasonality.get('dominant_periods', {}).get(str(period), 0) if seasonality else 0\n"
            "avg_trend_strength = trend_analysis.get('avg_trend_strength', 0) if trend_analysis else 0\n"
            "strongly_trending_pct = trend_analysis.get('strongly_trending_pct', 0) if trend_analysis else 0\n"
            "pct_with_changepoints = changepoint_analysis.get('pct_with_changepoints', 0) if changepoint_analysis else 0\n\n"
            "# Get top features by importance\n"
            "top_features = []\n"
            "if len(feat_imp) > 0 and 'feature' in feat_imp.columns:\n"
            "    feat_imp_sorted = feat_imp.sort_values('ensemble_score', ascending=False)\n"
            "    top_features = feat_imp_sorted['feature'].head(20).tolist()\n\n"
            "# Get significant lags from ACF analysis\n"
            "significant_lags = []\n"
            "if len(acf_summary) > 0 and 'significant_acf_lags' in acf_summary.columns:\n"
            "    all_lags = acf_summary['significant_acf_lags'].dropna().astype(str).str.split(',').explode()\n"
            "    try:\n"
            "        all_lags = all_lags[all_lags != ''].astype(int)\n"
            "        significant_lags = all_lags.value_counts().head(10).index.tolist()\n"
            "    except:\n"
            "        pass\n\n"
            "# Determine recommended clustering features from per_key_metrics\n"
            "available_clustering_features = []\n"
            "priority_features = ['mean', 'cv', 'adi', 'zero_fraction', 'demand_frequency', 'forecastability_score',\n"
            "                     'trend_strength', 'seasonal_strength', 'autocorr_lag1', 'skewness']\n"
            "for f in priority_features:\n"
            "    if f in metrics.columns:\n"
            "        available_clustering_features.append(f)\n\n"
            "# Determine recommended algorithm based on data characteristics\n"
            "lumpy_pct = demand_dist.get('lumpy', 0) + demand_dist.get('intermittent', 0)\n"
            "recommended_algorithm = 'GaussianMixture'  # GMM generally better for mixed patterns\n\n"
            "# Determine cluster count based on heterogeneity\n"
            "avg_cv = metrics['cv'].mean() if 'cv' in metrics.columns else 0.5\n"
            "if avg_cv > 1.5:\n"
            "    k_recommended = 6  # High variability = more segments\n"
            "elif avg_cv > 0.8:\n"
            "    k_recommended = 5\n"
            "else:\n"
            "    k_recommended = 4\n\n"
            "# =================================================================\n"
            "# CREATE SEGMENTATION CONTEXT\n"
            "# =================================================================\n"
            "seg_context = {\n"
            "    'source': 'eda_analyst',\n"
            "    'data_summary': {\n"
            "        'total_series': n_series,\n"
            "        'demand_distribution': {str(k): round(v, 3) for k, v in demand_dist.items()},\n"
            "        'stationarity_distribution': {str(k): round(v, 3) for k, v in stat_dist.items()},\n"
            "        'avg_cv': round(avg_cv, 3),\n"
            "        'avg_zero_fraction': round(metrics['zero_fraction'].mean(), 3) if 'zero_fraction' in metrics.columns else 0,\n"
            "    },\n"
            "    'exhaustive_analysis_insights': {\n"
            "        'has_seasonality_pct': round(has_seasonality_pct, 3),\n"
            "        'avg_trend_strength': round(avg_trend_strength, 3),\n"
            "        'strongly_trending_pct': round(strongly_trending_pct, 3),\n"
            "        'pct_with_changepoints': round(pct_with_changepoints, 3),\n"
            "        'note': 'Use these insights to inform clustering strategy'\n"
            "    },\n"
            "    'clustering_dimensions': {\n"
            "        'volume_based': {'features': ['mean', 'sum', 'demand_frequency'], 'rationale': 'High vs low volume'},\n"
            "        'intermittency_based': {'features': ['adi', 'cv', 'cv2', 'zero_fraction'], 'rationale': 'Syntetos-Boylan'},\n"
            "        'variability_based': {'features': ['cv', 'std', 'range_normalized'], 'rationale': 'Stable vs variable'},\n"
            "        'temporal_based': {'features': ['trend_strength', 'seasonal_strength', 'autocorr_lag1'], 'rationale': 'Trending vs stable'},\n"
            "    },\n"
            "    'recommended_clustering_features': available_clustering_features,\n"
            "    'cluster_count': {\n"
            "        'min': 3,\n"
            "        'max': 7,\n"
            "        'recommended': k_recommended,\n"
            "        'rationale': f'Based on avg_cv={avg_cv:.2f}, {n_series} series, trend_pct={strongly_trending_pct:.1%}'\n"
            "    },\n"
            "    'algorithm': recommended_algorithm,\n"
            "    'per_segment_model_hints': {\n"
            "        # FEATURE-BASED MODELS ONLY - univariate models (croston, tsb, ets, theta, prophet) are BANNED\n"
            "        # All models MUST leverage engineered features (lags, rolling, calendar, external)\n"
            "        'smooth': {'models': ['lightgbm', 'xgboost', 'catboost', 'random_forest'], 'expected_wape': '15-30%'},\n"
            "        'erratic': {'models': ['xgboost', 'lightgbm', 'catboost', 'random_forest'], 'expected_wape': '30-50%'},\n"
            "        'intermittent': {'models': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model'], 'expected_wape': '40-60%'},\n"
            "        'lumpy': {'models': ['lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'tweedie'], 'expected_wape': '50-80%'}\n"
            "    },\n"
            "    'critical_warning': 'Use MULTI-DIMENSIONAL clustering (volume + pattern + variability), not just ADI/CV!'\n"
            "}\n\n"
            "# =================================================================\n"
            "# CREATE FEATURE ENGINEERING CONTEXT\n"
            "# =================================================================\n"
            "# Determine lag recommendations based on time format and ACF analysis\n"
            "if time_format == 'year_week':\n"
            "    base_target_lags = [1, 2, 4, 13, 26, 52]\n"
            "    base_feature_lags = [1, 4, 52]\n"
            "    rolling_windows = [4, 13, 26, 52]\n"
            "else:\n"
            "    base_target_lags = [1, 3, 6, 12]\n"
            "    base_feature_lags = [1, 3, 12]\n"
            "    rolling_windows = [3, 6, 12]\n\n"
            "# Enhance lag recommendations with significant lags from ACF analysis\n"
            "target_lags = list(set(base_target_lags + [l for l in significant_lags if l <= period]))\n"
            "target_lags.sort()\n"
            "feature_lags = base_feature_lags\n\n"
            "# Determine if intermittency features needed\n"
            "avg_zero_frac = metrics['zero_fraction'].mean() if 'zero_fraction' in metrics.columns else 0\n"
            "need_intermittency = avg_zero_frac > 0.2\n\n"
            "# Determine transformations\n"
            "need_log = avg_cv > 1.5\n"
            "stat_pct = stat_dist.get('NON_STATIONARY', stat_dist.get('non_stationary', 0))\n"
            "need_diff = stat_pct > 0.3 if stat_dist else False\n\n"
            "# Seasonal features based on seasonality detection\n"
            "need_seasonal_features = has_seasonality_pct > 0.3\n\n"
            "feat_context = {\n"
            "    'source': 'eda_analyst',\n"
            "    'feature_philosophy': 'Include ALL features initially - let model selection decide',\n"
            "    'exhaustive_analysis_insights': {\n"
            "        'has_seasonality_pct': round(has_seasonality_pct, 3),\n"
            "        'avg_trend_strength': round(avg_trend_strength, 3),\n"
            "        'pct_with_changepoints': round(pct_with_changepoints, 3),\n"
            "        'significant_acf_lags': significant_lags[:10],\n"
            "    },\n"
            "    'total_features': {\n"
            "        'numeric': len(numeric_features),\n"
            "        'categorical': len(categorical_features),\n"
            "        'total': len(numeric_features) + len(categorical_features)\n"
            "    },\n"
            "    'top_features_by_importance': top_features[:20],\n"
            "    'all_numeric_features': numeric_features,\n"
            "    'all_categorical_features': categorical_features,\n"
            "    'lag_recommendations': {\n"
            "        'target_lags': target_lags,\n"
            "        'feature_lags': feature_lags,\n"
            "        'seasonal_period': period,\n"
            "        'acf_informed_lags': significant_lags[:5] if significant_lags else [],\n"
            "        'rationale': f'{time_format} format, period={period}, enhanced with ACF analysis'\n"
            "    },\n"
            "    'rolling_features': {\n"
            "        'windows': rolling_windows,\n"
            "        'stats': ['mean', 'std', 'min', 'max']\n"
            "    },\n"
            "    'seasonal_features': {\n"
            "        'needed': need_seasonal_features,\n"
            "        'features': ['week_of_year', 'month', 'quarter', 'is_holiday_week'] if need_seasonal_features else [],\n"
            "        'seasonal_strength': round(has_seasonality_pct, 3)\n"
            "    },\n"
            "    'trend_features': {\n"
            "        'needed': strongly_trending_pct > 0.3,\n"
            "        'features': ['linear_trend', 'rolling_trend_slope'] if strongly_trending_pct > 0.3 else [],\n"
            "        'avg_trend_strength': round(avg_trend_strength, 3)\n"
            "    },\n"
            "    'intermittency_features': {\n"
            "        'needed': need_intermittency,\n"
            "        'avg_zero_fraction': round(avg_zero_frac, 3),\n"
            "        'features': ['time_since_last_nonzero', 'demand_occurrence_rate', 'cumsum_nonzero'] if need_intermittency else []\n"
            "    },\n"
            "    'transformation': {\n"
            "        'log_transform': need_log,\n"
            "        'differencing': need_diff,\n"
            "        'rationale': f'CV={avg_cv:.2f}, non_stationary={stat_pct:.1%}'\n"
            "    },\n"
            "    'categorical_encoding': 'target_encoding',\n"
            "    'interaction_features': ['price*promo', 'lag1*seasonal_indicator'] if 'price' in str(numeric_features) else []\n"
            "}\n\n"
            "# =================================================================\n"
            "# CREATE TRAINING CONTEXT - NEW!\n"
            "# =================================================================\n"
            "training_context = {\n"
            "    'source': 'eda_analyst',\n"
            "    'data_characteristics': {\n"
            "        'total_series': n_series,\n"
            "        'demand_distribution': {str(k): round(v, 3) for k, v in demand_dist.items()},\n"
            "        'avg_cv': round(avg_cv, 3),\n"
            "        'avg_zero_fraction': round(avg_zero_frac, 3),\n"
            "        'pct_non_stationary': round(stat_pct, 3) if stat_dist else 0,\n"
            "        'pct_with_seasonality': round(has_seasonality_pct, 3),\n"
            "        'pct_trending': round(strongly_trending_pct, 3),\n"
            "    },\n"
            "    'model_recommendations': {\n"
            "        'primary_family': 'gradient_boosting',\n"
            "        'models': ['lightgbm', 'xgboost', 'catboost'],\n"
            "        'rationale': 'Feature-based models that leverage engineered features'\n"
            "    },\n"
            "    'loss_function_recommendations': {\n"
            "        'default': 'tweedie' if lumpy_pct > 0.3 else 'mse',\n"
            "        'for_intermittent': 'tweedie',\n"
            "        'for_smooth': 'mse',\n"
            "        'rationale': f'Tweedie for {lumpy_pct*100:.0f}% intermittent/lumpy series'\n"
            "    },\n"
            "    'evaluation_metrics': {\n"
            "        'primary': 'wape',\n"
            "        'secondary': ['mae', 'rmse', 'bias'],\n"
            "        'per_segment': True\n"
            "    },\n"
            "    'per_segment_guidance': {\n"
            "        'smooth': {'loss': 'mse', 'expected_wape': '15-30%', 'models': ['lightgbm', 'xgboost']},\n"
            "        'erratic': {'loss': 'huber', 'expected_wape': '30-50%', 'models': ['xgboost', 'catboost']},\n"
            "        'intermittent': {'loss': 'tweedie', 'expected_wape': '40-60%', 'models': ['lightgbm', 'tweedie_gbm']},\n"
            "        'lumpy': {'loss': 'tweedie', 'expected_wape': '50-80%', 'models': ['lightgbm', 'tweedie_gbm']}\n"
            "    },\n"
            "    'hyperparameter_hints': {\n"
            "        'n_estimators': [100, 300, 500],\n"
            "        'learning_rate': [0.01, 0.05, 0.1],\n"
            "        'max_depth': [5, 7, 10],\n"
            "        'min_child_samples': [20, 50, 100] if lumpy_pct > 0.3 else [10, 20, 50]\n"
            "    },\n"
            "    'critical_warnings': [\n"
            "        'Use FEATURE-BASED models only (lightgbm, xgboost) - univariate models BANNED',\n"
            "        'Must leverage lag features, rolling features, calendar features',\n"
            "        f'High intermittency ({lumpy_pct*100:.0f}%) - consider Tweedie loss',\n"
            "    ]\n"
            "}\n\n"
            "# =================================================================\n"
            "# SAVE ALL 3 CONTEXT FILES\n"
            "# =================================================================\n"
            "with open(os.path.join(eda_dir, 'eda_to_segmentation_context.json'), 'w') as f:\n"
            "    json.dump(seg_context, f, indent=2, default=str)\n"
            "with open(os.path.join(eda_dir, 'eda_to_feature_context.json'), 'w') as f:\n"
            "    json.dump(feat_context, f, indent=2, default=str)\n"
            "with open(os.path.join(eda_dir, 'eda_to_training_context.json'), 'w') as f:\n"
            "    json.dump(training_context, f, indent=2, default=str)\n\n"
            "print(f'Analyzed {n_series} series with exhaustive insights')\n"
            "print(f'Seasonality: {has_seasonality_pct*100:.0f}%, Trend: {strongly_trending_pct*100:.0f}%')\n"
            "print(f'Recommended: {recommended_algorithm}, k={k_recommended}')\n"
            "print(f'Significant ACF lags: {significant_lags[:5]}')\n"
            "print('Created: eda_to_segmentation_context.json, eda_to_feature_context.json, eda_to_training_context.json')\n"
            "```\n"
        ),
        agent=analyst,
        expected_output=(
            "Created 3 intelligent context files based on exhaustive EDA analysis: "
            "eda_to_segmentation_context.json (clustering guidance), "
            "eda_to_feature_context.json (lag/rolling/seasonal features), "
            "eda_to_training_context.json (model/loss/metrics recommendations)."
        ),
        output_file=os.path.join(eda_dir_out, "eda_analysis_report.md"),
        context=[task_plan, task_execute],
        # CRITICAL: Callback to validate Analyst output for downstream crews
        callback=lambda output: _validate_analyst_output(eda_dir),
    )

    # -------------------------------------------------------------------------
    # Task 4: Documentation - Generate comprehensive insights guide
    # -------------------------------------------------------------------------
    documentation_agent = _create_eda_documentation_agent(llm)
    task_document = Task(
        name="generate_eda_documentation",
        description=(
            "# CREATE COMPREHENSIVE EDA INSIGHTS DOCUMENTATION\n\n"
            "######################################################################\n"
            "#  USE CodeExecutionTool TO RUN THIS CODE - DO NOT JUST PRINT IT!   #\n"
            "######################################################################\n\n"
            "Execute this code using CodeExecutionTool. The code will SAVE the\n"
            "markdown file to disk. Do NOT print the markdown content.\n\n"
            "## CODE TO EXECUTE (use CodeExecutionTool):\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from datetime import datetime\n\n"
            f"eda_dir = '{eda_dir}'\n\n"
            "# Load all EDA outputs\n"
            "metrics = pd.read_csv(os.path.join(eda_dir, 'per_key_metrics.csv'))\n"
            "eda_summary = json.load(open(os.path.join(eda_dir, 'eda_summary.json'))) if os.path.exists(os.path.join(eda_dir, 'eda_summary.json')) else {}\n"
            "seg_context = json.load(open(os.path.join(eda_dir, 'eda_to_segmentation_context.json'))) if os.path.exists(os.path.join(eda_dir, 'eda_to_segmentation_context.json')) else {}\n"
            "feat_context = json.load(open(os.path.join(eda_dir, 'eda_to_feature_context.json'))) if os.path.exists(os.path.join(eda_dir, 'eda_to_feature_context.json')) else {}\n"
            "train_context = json.load(open(os.path.join(eda_dir, 'eda_to_training_context.json'))) if os.path.exists(os.path.join(eda_dir, 'eda_to_training_context.json')) else {}\n\n"
            "# Extract key statistics\n"
            "n_series = len(metrics)\n"
            "avg_cv = metrics['cv'].mean() if 'cv' in metrics.columns else 0\n"
            "avg_zero_frac = metrics['zero_fraction'].mean() if 'zero_fraction' in metrics.columns else 0\n\n"
            "# Demand pattern distribution\n"
            "if 'intermittency_class' in metrics.columns:\n"
            "    pattern_dist = metrics['intermittency_class'].value_counts(normalize=True).to_dict()\n"
            "else:\n"
            "    pattern_dist = {}\n\n"
            "# Stationarity distribution\n"
            "if 'stationarity' in metrics.columns:\n"
            "    stat_dist = metrics['stationarity'].value_counts(normalize=True).to_dict()\n"
            "else:\n"
            "    stat_dist = {}\n\n"
            "# Extract exhaustive analysis insights\n"
            "exhaustive = seg_context.get('exhaustive_analysis_insights', {})\n"
            "seasonality_pct = exhaustive.get('has_seasonality_pct', 0)\n"
            "trend_pct = exhaustive.get('strongly_trending_pct', 0)\n"
            "changepoint_pct = exhaustive.get('pct_with_changepoints', 0)\n\n"
            "# Get recommendations\n"
            "cluster_rec = seg_context.get('cluster_count', {})\n"
            "algorithm_rec = seg_context.get('algorithm', 'gaussian_mixture')\n"
            "lag_rec = feat_context.get('lag_recommendations', {})\n"
            "model_rec = train_context.get('model_recommendations', {})\n\n"
            "# Build markdown document\n"
            "md = f'''# EDA Insights Guide\n"
            "**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "---\n\n"
            "## Executive Summary\n\n"
            "This document provides a comprehensive analysis of the demand forecasting dataset,\n"
            "explaining the key patterns detected and the reasoning behind each recommendation.\n\n"
            "### Key Metrics at a Glance\n\n"
            "| Metric | Value | Interpretation |\n"
            "|--------|-------|----------------|\n"
            "| Total Time Series | {n_series:,} | Number of unique keys analyzed |\n"
            "| Average CV | {avg_cv:.2f} | {'High variability - consider robust models' if avg_cv > 1.0 else 'Moderate variability'} |\n"
            "| Average Zero Fraction | {avg_zero_frac:.1%} | {'Significant intermittency - use zero-inflated models' if avg_zero_frac > 0.3 else 'Low intermittency'} |\n"
            "| Seasonality Detected | {seasonality_pct:.1%} | {'Strong seasonal patterns' if seasonality_pct > 0.5 else 'Limited seasonality'} |\n"
            "| Trending Series | {trend_pct:.1%} | {'Many series trending' if trend_pct > 0.3 else 'Most series stable'} |\n\n"
            "---\n\n"
            "## 1. Demand Pattern Analysis (Syntetos-Boylan Classification)\n\n"
            "The Syntetos-Boylan framework classifies demand into four categories based on:\n"
            "- **ADI (Average Demand Interval)**: Time between non-zero demands\n"
            "- **CV² (Squared Coefficient of Variation)**: Demand size variability\n\n"
            "### Pattern Distribution\n\n"
            "| Pattern | Percentage | Characteristics | Recommended Approach |\n"
            "|---------|------------|-----------------|---------------------|\n"
            "| Smooth | {pattern_dist.get(\"smooth\", 0):.1%} | Regular, predictable | Standard ML models (LightGBM) |\n"
            "| Erratic | {pattern_dist.get(\"erratic\", 0):.1%} | Frequent but variable | Robust models (XGBoost) |\n"
            "| Intermittent | {pattern_dist.get(\"intermittent\", 0):.1%} | Sparse but consistent | Zero-inflated models |\n"
            "| Lumpy | {pattern_dist.get(\"lumpy\", 0):.1%} | Sparse and variable | Hurdle/Tweedie models |\n\n"
            "**Key Insight:** '''\n\n"
            "# Add pattern insight\n"
            "dominant_pattern = max(pattern_dist.items(), key=lambda x: x[1])[0] if pattern_dist else 'unknown'\n"
            "lumpy_interm_pct = pattern_dist.get('lumpy', 0) + pattern_dist.get('intermittent', 0)\n"
            "pattern_insight = ''\n"
            "if lumpy_interm_pct > 0.4:\n"
            "    pattern_insight = f'With {lumpy_interm_pct:.1%} of series being intermittent/lumpy, special handling for zero-inflation is critical. Recommend Tweedie loss and zero-inflated models.'\n"
            "elif dominant_pattern == 'smooth':\n"
            "    pattern_insight = 'Dataset is dominated by smooth demand patterns, making it amenable to standard forecasting approaches.'\n"
            "else:\n"
            "    pattern_insight = f'Mixed demand patterns detected with {dominant_pattern} being most common. Segment-specific modeling recommended.'\n\n"
            "md += f'''{pattern_insight}\n\n"
            "---\n\n"
            "## 2. Temporal Pattern Analysis\n\n"
            "### Seasonality Analysis\n\n"
            "- **Series with Seasonality:** {seasonality_pct:.1%}\n"
            "- **Detected Period:** {lag_rec.get(\"seasonal_period\", \"N/A\")} periods\n"
            "- **Implication:** {\"Include Fourier features and seasonal lags\" if seasonality_pct > 0.3 else \"Seasonal features may have limited impact\"}\n\n"
            "### Trend Analysis\n\n"
            "- **Strongly Trending Series:** {trend_pct:.1%}\n"
            "- **Implication:** {\"Include trend features and consider detrending\" if trend_pct > 0.3 else \"Most series are stationary - minimal trend handling needed\"}\n\n"
            "### Stationarity Results\n\n'''\n\n"
            "# Add stationarity table\n"
            "stat_table = '| Status | Percentage |\\n|--------|------------|\\n'\n"
            "for status, pct in stat_dist.items():\n"
            "    stat_table += f'| {status} | {pct:.1%} |\\n'\n"
            "md += stat_table\n\n"
            "md += f'''\n\n"
            "### Changepoint Detection\n\n"
            "- **Series with Structural Breaks:** {changepoint_pct:.1%}\n"
            "- **Implication:** {\"Consider regime-switching or include changepoint indicators\" if changepoint_pct > 0.2 else \"Most series have stable patterns\"}\n\n"
            "---\n\n"
            "## 3. Feature Engineering Recommendations\n\n"
            "Based on the temporal patterns observed:\n\n"
            "### Lag Features\n"
            "- **Target Lags:** {lag_rec.get(\"target_lags\", [])}\n"
            "- **ACF-Informed Lags:** {lag_rec.get(\"acf_informed_lags\", [])}\n"
            "- **Rationale:** {lag_rec.get(\"rationale\", \"Based on data characteristics\")}\n\n"
            "### Rolling Window Features\n"
            "- **Windows:** {feat_context.get(\"rolling_features\", {{}}).get(\"windows\", [])}\n"
            "- **Statistics:** {feat_context.get(\"rolling_features\", {{}}).get(\"stats\", [])}\n\n"
            "### Special Features\n"
            "- **Seasonal Features Needed:** {feat_context.get(\"seasonal_features\", {{}}).get(\"needed\", False)}\n"
            "- **Intermittency Features Needed:** {feat_context.get(\"intermittency_features\", {{}}).get(\"needed\", False)}\n"
            "- **Log Transform Recommended:** {feat_context.get(\"transformation\", {{}}).get(\"log_transform\", False)}\n\n"
            "---\n\n"
            "## 4. Segmentation Recommendations\n\n"
            "### Clustering Strategy\n\n"
            "- **Recommended Algorithm:** {algorithm_rec}\n"
            "- **Suggested Clusters:** {cluster_rec.get(\"recommended\", \"N/A\")} (range: {cluster_rec.get(\"min\", 3)}-{cluster_rec.get(\"max\", 7)})\n"
            "- **Rationale:** {cluster_rec.get(\"rationale\", \"Based on data characteristics\")}\n\n"
            "### Clustering Dimensions\n\n"
            "The segmentation should consider multiple dimensions:\n"
            "1. **Volume-based**: High vs low demand series\n"
            "2. **Intermittency-based**: Syntetos-Boylan classification\n"
            "3. **Variability-based**: Stable vs volatile series\n"
            "4. **Temporal-based**: Trending vs stationary series\n\n"
            "---\n\n"
            "## 5. Model Training Recommendations\n\n"
            "### Primary Model Family\n"
            "- **Recommended:** {model_rec.get(\"primary_family\", \"gradient_boosting\")}\n"
            "- **Specific Models:** {model_rec.get(\"models\", [])}\n"
            "- **Rationale:** {model_rec.get(\"rationale\", \"Feature-based models for demand forecasting\")}\n\n"
            "### Loss Function by Pattern\n\n"
            "| Demand Pattern | Recommended Loss | Expected WAPE |\n"
            "|----------------|------------------|---------------|\n"
            "| Smooth | MSE | 15-30% |\n"
            "| Erratic | Huber | 30-50% |\n"
            "| Intermittent | Tweedie | 40-60% |\n"
            "| Lumpy | Tweedie | 50-80% |\n\n"
            "---\n\n"
            "## 6. Key Learnings & Takeaways\n\n"
            "### What the Data Tells Us\n\n'''\n\n"
            "# Generate key learnings\n"
            "learnings = []\n"
            "if avg_zero_frac > 0.3:\n"
            "    learnings.append(f'1. **High Zero-Inflation ({avg_zero_frac:.1%})**: Many series have sparse demand. Zero-inflated or hurdle models are essential.')\n"
            "if avg_cv > 1.5:\n"
            "    learnings.append(f'2. **High Variability (CV={avg_cv:.2f})**: Demand is highly variable. Use robust loss functions and regularization.')\n"
            "if seasonality_pct > 0.5:\n"
            "    learnings.append(f'3. **Strong Seasonality ({seasonality_pct:.1%})**: Include seasonal features and Fourier terms.')\n"
            "if trend_pct > 0.3:\n"
            "    learnings.append(f'4. **Trending Data ({trend_pct:.1%})**: Include trend features or consider differencing.')\n"
            "if lumpy_interm_pct > 0.4:\n"
            "    learnings.append(f'5. **Intermittent Demand ({lumpy_interm_pct:.1%})**: Segment-specific models with Tweedie loss recommended.')\n\n"
            "if not learnings:\n"
            "    learnings.append('1. Dataset appears well-behaved with standard demand patterns.')\n"
            "    learnings.append('2. Standard gradient boosting models should perform well.')\n\n"
            "md += '\\n'.join(learnings)\n\n"
            "md += f'''\n\n"
            "### Critical Warnings\n\n'''\n\n"
            "warnings = train_context.get('critical_warnings', [])\n"
            "for w in warnings:\n"
            "    md += f'- ⚠️ {w}\\n'\n\n"
            "md += f'''\n\n"
            "---\n\n"
            "*This documentation was auto-generated by the EDA Documentation Agent based on actual data analysis.*\n"
            "'''\n\n"
            "# Save documentation\n"
            "doc_path = os.path.join(eda_dir, 'EDA_INSIGHTS_GUIDE.md')\n"
            "with open(doc_path, 'w') as f:\n"
            "    f.write(md)\n\n"
            "print(f'Created EDA documentation: {n_series} series analyzed')\n"
            "print(f'Key patterns: {dominant_pattern} dominant, {lumpy_interm_pct:.1%} intermittent/lumpy')\n"
            "print(f'Saved: EDA_INSIGHTS_GUIDE.md')\n"
            "```"
        ),
        agent=documentation_agent,
        expected_output=(
            "Created EDA_INSIGHTS_GUIDE.md - a comprehensive documentation file with "
            "actual data-driven insights, pattern analysis, and recommendations."
        ),
        output_file=os.path.join(eda_dir_out, "eda_documentation_report.md"),
        context=[task_plan, task_execute, task_analyze],
    )

    # -------------------------------------------------------------------------
    # OPTIONAL Task 5: Reviewer Validation (if enable_reviewer is True)
    # -------------------------------------------------------------------------
    enable_reviewer = getattr(config.design, 'enable_reviewer', False)

    agents = [planner, executor, analyst, documentation_agent]
    tasks = [task_plan, task_execute, task_analyze, task_document]

    if enable_reviewer:
        reviewer = _create_eda_reviewer_agent(llm, config)
        task_review = Task(
            name="review_eda_outputs",
            description=(
                "# EDA OUTPUT QUALITY REVIEW\n\n"
                "You are the EDA Reviewer. Validate all outputs from the EDA crew including documentation.\n\n"
                "## VALIDATION STEPS\n"
                "Write and execute code to validate:\n"
                "```python\n"
                "import os\n"
                "import json\n"
                f"eda_dir = '{eda_dir}'\n"
                "\n"
                "# Check required files exist\n"
                "required_files = [\n"
                "    'eda_to_segmentation_context.json',\n"
                "    'eda_to_feature_context.json',\n"
                "    'eda_to_training_context.json',  # New context file\n"
                "    'eda_report.md'\n"
                "]\n"
                "missing = [f for f in required_files if not os.path.exists(os.path.join(eda_dir, f))]\n"
                "print(f'Missing files: {missing}' if missing else 'All required files present')\n"
                "\n"
                "# Validate JSON structures\n"
                "if os.path.exists(os.path.join(eda_dir, 'eda_to_segmentation_context.json')):\n"
                "    with open(os.path.join(eda_dir, 'eda_to_segmentation_context.json')) as f:\n"
                "        seg_ctx = json.load(f)\n"
                "    required_keys = ['cluster_recommendations', 'model_family_hints']\n"
                "    missing_keys = [k for k in required_keys if k not in seg_ctx]\n"
                "    print(f'Segmentation context missing: {missing_keys}' if missing_keys else 'Segmentation context valid')\n"
                "```\n\n"
                "## OUTPUT\n"
                f"Create `{eda_dir}/eda_review_report.json` with:\n"
                "- quality_score (1-10)\n"
                "- files_validated (list)\n"
                "- issues_found (list)\n"
                "- recommendations (list)\n"
            ),
            agent=reviewer,
            expected_output=(
                "Created eda_review_report.json with quality score and validation results. "
                "All required files validated. Issues identified and documented."
            ),
            output_file=os.path.join(eda_dir_out, "eda_review_summary.md"),
            context=[task_analyze],
        )
        agents.append(reviewer)
        tasks.append(task_review)
        crew_name = "Autonomous EDA Crew (4-Agent Pattern with Reviewer)"
    else:
        crew_name = "Autonomous EDA Crew (3-Agent Pattern)"

    return Crew(
        name=crew_name,
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def run_eda_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> EDACrewResult:
    from utils.cost_tracking import get_cost_tracker, extract_tokens_from_crew_result

    # ==========================================================================
    # CRITICAL: Validate input data file exists BEFORE running any crew tasks
    # This prevents wasted LLM calls when the data is missing
    # ==========================================================================
    if not os.path.exists(config.input_data_path):
        raise EDAFailedError(
            f"Input data file not found: {config.input_data_path}\n"
            "Please ensure the data file exists at the configured path.\n"
            "Check config.yaml 'input_data_path' setting."
        )

    # Start cost tracking
    tracker = get_cost_tracker()
    tracker.start_crew("EDA Crew")

    # Get model ID from LLM if available
    model_id = getattr(llm, "model", "default")
    tracker.set_model(model_id)

    crew = create_eda_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)
    crew_result = crew.kickoff()

    eda_dir = os.path.join(config.artifact_base_path, "eda_output")
    charts_dir = os.path.join(eda_dir, "charts")

    # Extract and record tokens from crew result
    tokens = extract_tokens_from_crew_result(crew_result)
    if tokens["total"] > 0:
        tracker.record_llm_call(
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            model=model_id,
        )

    # End tracking and save cost report
    cost_report = tracker.end_crew("EDA Crew", eda_dir)
    cost_report_path = os.path.join(eda_dir, "eda_cost.json")

    return EDACrewResult(
        eda_dir=eda_dir,
        per_key_metrics_path=os.path.join(eda_dir, "per_key_metrics.csv"),
        global_eda_summary_path=os.path.join(eda_dir, "eda_summary.json"),  # Fixed: was global_eda_summary.json
        enhanced_eda_summary_path=os.path.join(eda_dir, "enhanced_eda_summary.json"),
        segmentation_suggestions_path=os.path.join(eda_dir, "segmentation_suggestions.json"),
        feature_importances_rf_path=os.path.join(eda_dir, "feature_importances_rf.csv"),
        feature_correlation_matrix_path=os.path.join(eda_dir, "feature_correlation_matrix.csv"),
        # Advanced EDA outputs
        advanced_per_key_metrics_path=os.path.join(eda_dir, "advanced_per_key_metrics.csv"),
        granger_causality_path=os.path.join(eda_dir, "granger_causality.csv"),
        shap_interactions_path=os.path.join(eda_dir, "shap_interactions.json"),
        horizon_specific_importance_path=os.path.join(eda_dir, "horizon_specific_importance.json"),
        cross_series_correlation_path=os.path.join(eda_dir, "cross_series_correlation.json"),
        charts_dir_path=charts_dir,
        eda_report_markdown_path=os.path.join(eda_dir, "eda_report.md"),
        # New: Generated pipeline and context
        eda_pipeline_script_path=os.path.join(eda_dir, "eda_pipeline.py"),
        eda_to_segmentation_context_path=os.path.join(eda_dir, "eda_to_segmentation_context.json"),
        # DETERMINISTIC CODE OUTPUT for Pipeline Generator
        eda_deterministic_code_path=os.path.join(eda_dir, "eda_deterministic.py"),
        cost_report_path=cost_report_path,
    )


# =============================================================================
# DETERMINISTIC EDA - NO LLM INVOLVEMENT FOR CORE ANALYSIS
# =============================================================================

def run_eda_deterministic(
    config: DemandForecastConfig,
    llm: LLM,
    use_agents_for_context: bool = True,  # Kept for backwards compatibility, always True
) -> EDACrewResult:
    """
    Run EDA pipeline: DETERMINISTIC core + LLM for context/rationale.

    Mode:
    - Core EDA (per_key_metrics.csv, etc.) runs deterministically without LLM
    - Context files use LLM to generate intelligent rationale and recommendations

    This guarantees:
    - Consistent, correct core analysis output every time
    - No risk of agents writing simplified code for core analysis
    - Rich, intelligent rationale in context files from LLM

    Parameters
    ----------
    config : DemandForecastConfig
        Configuration object with data paths and settings
    llm : LLM
        LLM instance for context file creation

    Returns
    -------
    EDACrewResult
        Result object with paths to all created files

    Example
    -------
    >>> result = run_eda_deterministic(config, llm)
    >>> print(f"EDA complete: {result.per_key_metrics_path}")
    """
    import pandas as pd
    import numpy as np

    from utils.eda import run_eda_pipeline

    logger.info("="*70)
    logger.info("RUNNING DETERMINISTIC EDA (NO LLM FOR CORE ANALYSIS)")
    logger.info("="*70)

    # Validate input
    if not os.path.exists(config.input_data_path):
        raise EDAFailedError(
            f"Input data file not found: {config.input_data_path}\n"
            "Please ensure the data file exists at the configured path."
        )

    # Setup directories
    eda_dir = os.path.join(config.artifact_base_path, "eda_output")
    charts_dir = os.path.join(eda_dir, "charts")
    os.makedirs(eda_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # Load data (supports both CSV and Parquet)
    from utils.agent_utilities import load_source_data
    logger.info(f"Loading data from {config.input_data_path}")
    df = load_source_data(config.input_data_path)

    # Filter to training period
    date_col = config.date_column
    target_col = config.target_column
    train_start = config.train_start
    train_end = config.train_end

    # Convert to match column dtype
    col_dtype = df[date_col].dtype
    if np.issubdtype(col_dtype, np.integer):
        train_start = int(train_start)
        train_end = int(train_end)
    elif np.issubdtype(col_dtype, np.floating):
        train_start = float(train_start)
        train_end = float(train_end)
    else:
        train_start = str(train_start)
        train_end = str(train_end)

    df = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)].copy()
    logger.info(f"Filtered to training period: {len(df)} rows ({train_start} to {train_end})")

    # Get feature lists from config (these are methods, not properties)
    numeric_features = [c for c in config.all_numeric_features() if c in df.columns]
    categorical_features = [c for c in config.all_categorical_features() if c in df.columns]

    # Determine period from time_format
    period = 52 if config.time_format == 'year_week' else 12

    # Get dead key threshold from design config (with fallback)
    dead_key_threshold = getattr(config.design, 'dead_key_threshold', 26) if hasattr(config, 'design') else 26

    # Run the DETERMINISTIC EDA pipeline
    logger.info("Running run_eda_pipeline() - DETERMINISTIC")
    logger.info(f"  Key columns: {list(config.key_columns)}")
    logger.info(f"  Date column: {date_col}")
    logger.info(f"  Target column: {target_col}")
    logger.info(f"  Numeric features: {len(numeric_features)}")
    logger.info(f"  Categorical features: {len(categorical_features)}")
    logger.info(f"  Period: {period}")

    result = run_eda_pipeline(
        df=df,
        key_columns=list(config.key_columns),
        date_col=date_col,
        target_col=target_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        output_dir=eda_dir,
        period=period,
        verbose=True,
        train_end=str(config.train_end),
        dead_key_threshold=dead_key_threshold,
    )

    # Check result
    if result.get('status') == 'error':
        raise EDAFailedError(f"EDA pipeline failed: {result.get('error', 'Unknown error')}")

    logger.info(f"EDA pipeline completed successfully")
    logger.info(f"Files created: {result.get('files_created', [])}")

    # Validate per_key_metrics.csv has correct columns
    per_key_path = os.path.join(eda_dir, 'per_key_metrics.csv')
    if os.path.exists(per_key_path):
        per_key_df = pd.read_csv(per_key_path)
        logger.info(f"per_key_metrics.csv has {len(per_key_df.columns)} columns: {list(per_key_df.columns)}")

        # Verify required columns
        required_cols = ['mean', 'cv', 'zero_fraction', 'adi', 'volume_tier', 'demand_pattern']
        missing = [c for c in required_cols if c not in per_key_df.columns]
        if missing:
            logger.error(f"CRITICAL: Missing columns in per_key_metrics.csv: {missing}")
            raise EDAFailedError(f"per_key_metrics.csv missing required columns: {missing}")
    else:
        raise EDAFailedError(f"per_key_metrics.csv was not created!")

    # Create context files DETERMINISTICALLY (reliable)
    logger.info("Creating context files (deterministic)...")
    _create_context_files_deterministic(eda_dir, config, per_key_df)

    # =========================================================================
    # EDA INSIGHTS REPORT (optional - controlled by config)
    # =========================================================================
    enable_insights = getattr(config.design, 'enable_insights_reports', False)
    if enable_insights:
        import shutil
        # Backup critical files before LLM insights crew runs
        critical_files = [
            'per_key_metrics.csv', 'eda_to_segmentation_context.json',
            'eda_to_feature_context.json', 'eda_to_training_context.json',
            'eda_summary.json', 'data_quality.json', 'stationarity_results.csv',
            'feature_importance.csv', 'data_profile.json', 'seasonality_analysis.json',
            'trend_analysis.json', 'changepoint_analysis.json',
            'dead_key_summary.json', 'dead_keys.txt',
        ]
        backup_dir = os.path.join(eda_dir, '.backup_before_insights')
        os.makedirs(backup_dir, exist_ok=True)
        for fname in critical_files:
            src = os.path.join(eda_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(backup_dir, fname))
        logger.info(f"Backed up {len([f for f in critical_files if os.path.exists(os.path.join(eda_dir, f))])} critical files before insights crew")

        # Run EDA Insights crew
        n_series = len(per_key_df)
        _run_eda_insights_crew(eda_dir, config, llm, n_series)

        # Restore any files corrupted by the insights agent
        restored_count = 0
        for fname in critical_files:
            current_path = os.path.join(eda_dir, fname)
            backup_path = os.path.join(backup_dir, fname)
            if os.path.exists(backup_path):
                if not os.path.exists(current_path):
                    shutil.copy2(backup_path, current_path)
                    logger.warning(f"RESTORED deleted file: {fname}")
                    restored_count += 1
                else:
                    current_size = os.path.getsize(current_path)
                    backup_size = os.path.getsize(backup_path)
                    if backup_size > 0 and current_size < backup_size * 0.5:
                        shutil.copy2(backup_path, current_path)
                        logger.warning(f"RESTORED corrupted file: {fname} ({current_size}B -> {backup_size}B)")
                        restored_count += 1
        if restored_count > 0:
            logger.warning(f"RESTORED {restored_count} files corrupted by insights agent")
        shutil.rmtree(backup_dir, ignore_errors=True)
    else:
        logger.info("SKIPPING EDA insights report (enable_insights_reports=False)")

    logger.info("="*70)
    logger.info("DETERMINISTIC EDA COMPLETE")
    logger.info("="*70)

    return EDACrewResult(
        eda_dir=eda_dir,
        per_key_metrics_path=per_key_path,
        global_eda_summary_path=os.path.join(eda_dir, "eda_summary.json"),
        enhanced_eda_summary_path=os.path.join(eda_dir, "enhanced_eda_summary.json"),
        segmentation_suggestions_path=os.path.join(eda_dir, "segmentation_suggestions.json"),
        feature_importances_rf_path=os.path.join(eda_dir, "feature_importance.csv"),
        feature_correlation_matrix_path=os.path.join(eda_dir, "feature_correlation_matrix.csv"),
        advanced_per_key_metrics_path=os.path.join(eda_dir, "advanced_per_key_metrics.csv"),
        granger_causality_path=os.path.join(eda_dir, "granger_causality.csv"),
        shap_interactions_path=os.path.join(eda_dir, "shap_interactions.json"),
        horizon_specific_importance_path=os.path.join(eda_dir, "horizon_specific_importance.json"),
        cross_series_correlation_path=os.path.join(eda_dir, "cross_series_correlation.json"),
        charts_dir_path=charts_dir,
        eda_report_markdown_path=os.path.join(eda_dir, "eda_report.md"),
        eda_pipeline_script_path="",
        eda_to_segmentation_context_path=os.path.join(eda_dir, "eda_to_segmentation_context.json"),
        eda_deterministic_code_path="",
        eda_insights_report_path=os.path.join(eda_dir, "EDA_INSIGHTS_REPORT.md"),
        cost_report_path=os.path.join(eda_dir, "eda_crew_cost.json"),
    )


def _create_context_files_deterministic(
    eda_dir: str,
    config: DemandForecastConfig,
    per_key_df: 'pd.DataFrame',
) -> None:
    """
    Create context files for downstream crews DETERMINISTICALLY using Pydantic schemas.

    This creates standardized context files based on EDA metrics without
    any LLM interpretation. The context is based on statistical thresholds
    and best practices.

    Uses Pydantic schemas to ensure consistent, validated format.
    """
    import json
    import pandas as pd

    from config.schema import (
        EDAToSegmentationContext,
        EDAToFeatureContext,
        EDAToTrainingContext,
        DemandPatternDistribution,
        DataCharacteristics,
        SegmentationRecommendations,
        LagFeatureRecommendations,
        RollingFeatureRecommendations,
        IntermittencyHandling,
        ExternalFeatureRecommendations,
        ModelRecommendations,
        TrainingStrategy,
        TrainingDataCharacteristics,
    )

    # Load seasonality if exists
    seasonality = {}
    seasonality_path = os.path.join(eda_dir, 'seasonality_analysis.json')
    if os.path.exists(seasonality_path):
        with open(seasonality_path) as f:
            seasonality = json.load(f)

    # Compute statistics from per_key_metrics
    n_series = len(per_key_df)
    lumpy_pct = float((per_key_df['demand_pattern'] == 'lumpy').mean()) if 'demand_pattern' in per_key_df.columns else 0.0
    intermittent_pct = float((per_key_df['demand_pattern'] == 'intermittent').mean()) if 'demand_pattern' in per_key_df.columns else 0.0
    smooth_pct = float((per_key_df['demand_pattern'] == 'smooth').mean()) if 'demand_pattern' in per_key_df.columns else 0.0
    erratic_pct = float((per_key_df['demand_pattern'] == 'erratic').mean()) if 'demand_pattern' in per_key_df.columns else 0.0

    avg_cv = float(per_key_df['cv'].mean()) if 'cv' in per_key_df.columns else 1.0
    avg_zero_fraction = float(per_key_df['zero_fraction'].mean()) if 'zero_fraction' in per_key_df.columns else 0.3
    avg_adi = float(per_key_df['adi'].mean()) if 'adi' in per_key_df.columns else 1.5

    has_seasonality = seasonality.get('has_seasonality_pct', 0) > 0.3
    seasonal_period = seasonality.get('dominant_period', 52 if config.time_format == 'year_week' else 12)

    # Discrete demand detection
    discrete_pct = float(per_key_df['is_discrete'].mean()) if 'is_discrete' in per_key_df.columns else 0.0
    avg_n_unique = float(per_key_df['n_unique'].mean()) if 'n_unique' in per_key_df.columns else None

    # Determine dominant pattern
    pattern_pcts = {'smooth': smooth_pct, 'erratic': erratic_pct, 'intermittent': intermittent_pct, 'lumpy': lumpy_pct}
    dominant_pattern = max(pattern_pcts, key=pattern_pcts.get) if max(pattern_pcts.values()) > 0.4 else 'mixed'

    # =========================================================================
    # 1. EDA to Segmentation Context (using Pydantic schema)
    # =========================================================================
    segmentation_context = EDAToSegmentationContext(
        source="EDA (deterministic)",
        n_series=n_series,
        demand_pattern_distribution=DemandPatternDistribution(
            smooth=round(smooth_pct, 4),
            erratic=round(erratic_pct, 4),
            intermittent=round(intermittent_pct, 4),
            lumpy=round(lumpy_pct, 4),
        ),
        recommendations=SegmentationRecommendations(
            algorithm="gmm",  # GMM works best for mixed demand patterns
            n_clusters_range=[3, 4, 5, 6, 7],
            features_to_use=["volume_mean", "cv_clean", "zero_fraction_clean", "adi_log", "demand_frequency"],
            use_hybrid_segmentation=True,
            hybrid_dimensions=["volume_tier", "demand_pattern"],
            rationale="GMM recommended for mixed demand patterns. Hybrid segmentation combines statistical clusters with business dimensions.",
        ),
        data_characteristics=DataCharacteristics(
            avg_cv=round(avg_cv, 3),
            avg_zero_fraction=round(avg_zero_fraction, 3),
            avg_adi=round(avg_adi, 3),
            has_seasonality=has_seasonality,
            seasonal_period=seasonal_period,
            discrete_demand_pct=round(discrete_pct, 3),
        ),
    )

    with open(os.path.join(eda_dir, 'eda_to_segmentation_context.json'), 'w') as f:
        f.write(segmentation_context.model_dump_json(indent=2))

    # =========================================================================
    # 2. EDA to Feature Engineering Context (using Pydantic schema)
    # =========================================================================
    # Determine lag recommendations based on data characteristics
    if config.time_format == 'year_week':
        base_lags = [1, 2, 4, 8, 13, 26, 52]
        seasonal_lags = [52, 104] if has_seasonality else []
        rolling_windows = [4, 8, 13]
    else:
        base_lags = [1, 2, 3, 6, 12]
        seasonal_lags = [12, 24] if has_seasonality else []
        rolling_windows = [3, 6, 12]

    feature_context = EDAToFeatureContext(
        source="EDA (deterministic)",
        lag_features=LagFeatureRecommendations(
            recommended_lags=base_lags + seasonal_lags,
            seasonal_lags=seasonal_lags,
            rationale=f"Based on {config.time_format} granularity. Seasonal lags included: {has_seasonality}.",
        ),
        rolling_features=RollingFeatureRecommendations(
            windows=rolling_windows,
            aggregations=["mean", "std", "min", "max"],
            rationale=f"Standard rolling windows for {config.time_format} data.",
        ),
        intermittency_handling=IntermittencyHandling(
            high_intermittency_pct=round(lumpy_pct + intermittent_pct, 3),
            use_demand_probability_features=(lumpy_pct + intermittent_pct) > 0.3,
            use_zero_inflated_features=avg_zero_fraction > 0.3,
            rationale=f"Intermittency: {(lumpy_pct + intermittent_pct)*100:.1f}%, Zero fraction: {avg_zero_fraction*100:.1f}%",
        ),
        external_features=ExternalFeatureRecommendations(
            use_price_features="price" in str(config.all_numeric_features()).lower(),
            use_promo_features="promo" in str(config.all_categorical_features()).lower(),
            use_weather_features=False,
            recommended_lags=[0, 1, 2],
        ),
        analysis_summary="",
    )

    with open(os.path.join(eda_dir, 'eda_to_feature_context.json'), 'w') as f:
        f.write(feature_context.model_dump_json(indent=2))

    # =========================================================================
    # 3. EDA to Training Context (using Pydantic schema)
    # =========================================================================
    # Model recommendations based on demand patterns
    primary_models = ["lightgbm", "xgboost"]
    intermittency_specialists = []
    discrete_specialists = []
    seasonal_models = []

    # Add intermittency specialists if needed
    if (lumpy_pct + intermittent_pct) > 0.3:
        intermittency_specialists = ["croston", "sba", "tsb"]

    # Add discrete demand specialists if needed
    if discrete_pct > 0.1:  # More than 10% of keys have discrete demand
        discrete_specialists = ["ordinal_regression", "discrete_classifier", "hybrid_discrete"]

    # Add seasonal models if seasonality detected
    if has_seasonality:
        seasonal_models = ["sarima", "prophet", "tbats"]

    all_models = list(set(primary_models + intermittency_specialists + discrete_specialists + seasonal_models))

    training_context = EDAToTrainingContext(
        source="EDA (deterministic)",
        model_recommendations=ModelRecommendations(
            primary_models=primary_models,
            intermittency_specialists=intermittency_specialists,
            discrete_specialists=discrete_specialists,
            seasonal_models=seasonal_models,
            all_recommended=all_models,
            rationale=(
                f"Tree-based models for general forecasting. "
                f"Intermittency specialists for {(lumpy_pct+intermittent_pct)*100:.1f}% lumpy/intermittent. "
                f"Discrete specialists for {discrete_pct*100:.1f}% discrete demand keys. "
                f"Seasonal models: {'Yes' if has_seasonality else 'No'}."
            ),
        ),
        training_strategy=TrainingStrategy(
            use_segment_specific_models=True,
            use_ensemble=True,
            cv_folds=3,
            rationale="Segment-specific models to capture different demand patterns. Ensemble for robustness.",
        ),
        data_characteristics=TrainingDataCharacteristics(
            n_series=n_series,
            high_intermittency=(lumpy_pct + intermittent_pct) > 0.5,
            has_seasonality=has_seasonality,
            avg_cv=round(avg_cv, 3),
            has_discrete_demand=discrete_pct > 0.1,
            discrete_pct=round(discrete_pct, 3),
            avg_n_unique=round(avg_n_unique, 1) if avg_n_unique is not None else None,
        ),
        hyperparameter_hints={},
        analysis_summary="",
    )

    with open(os.path.join(eda_dir, 'eda_to_training_context.json'), 'w') as f:
        f.write(training_context.model_dump_json(indent=2))

    logger.info("Created SCHEMA-VALIDATED deterministic context files:")
    logger.info("  - eda_to_segmentation_context.json (EDAToSegmentationContext)")
    logger.info("  - eda_to_feature_context.json (EDAToFeatureContext)")
    logger.info("  - eda_to_training_context.json (EDAToTrainingContext)")


def _create_eda_insights_agent(llm: LLM) -> Agent:
    """
    Create the EDA Insights Documentation Agent.

    This agent uses CodeExecutionTool to iteratively explore EDA outputs
    and creates a comprehensive insights markdown report.
    """
    code_tool = CodeExecutionTool()

    return Agent(
        name="eda_insights_agent",
        role="PhD-Level Demand Forecasting EDA Insights Specialist",
        goal=(
            "EXECUTE Python code using CodeExecutionTool to explore EDA outputs "
            "and create EDA_INSIGHTS_REPORT.md. You MUST use the tool to run code - "
            "do NOT just describe or print the analysis."
        ),
        backstory=(
            "You are a PhD-level expert in demand forecasting with 15+ years experience.\n\n"
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE CodeExecutionTool TO RUN ANALYSIS CODE    #\n"
            "#  EXPLORE DATA ITERATIVELY - RUN MULTIPLE CODE EXECUTIONS          #\n"
            "#  SAVE FINAL REPORT TO FILE - DO NOT JUST PRINT IT                 #\n"
            "######################################################################\n\n"
            "## YOUR EXPERTISE:\n"
            "- Syntetos-Boylan: ADI>1.32 = intermittent, CV²>0.49 = erratic\n"
            "- Demand patterns: smooth, erratic, intermittent, lumpy\n"
            "- Model selection by pattern type\n"
            "- Feature engineering for time series\n\n"
            "## HOW TO COMPLETE THIS TASK:\n"
            "1. Use CodeExecutionTool to run exploration code\n"
            "2. Run MULTIPLE code executions to build understanding\n"
            "3. Extract specific statistics (counts, percentages, means)\n"
            "4. Final execution: SAVE markdown report to file\n"
            "5. Print only confirmation messages (max 10 lines per execution)\n\n"
            "## OUTPUT LIMIT: MAX 10 PRINT STATEMENTS PER CODE EXECUTION\n"
            "## SUPPRESS WARNINGS: warnings.filterwarnings('ignore')\n\n"
            "######################################################################\n"
            "#  CRITICAL: DO NOT OVERWRITE OR MODIFY ANY EXISTING FILES!          #\n"
            "#  You may ONLY create EDA_INSIGHTS_REPORT.md                        #\n"
            "#  NEVER write to per_key_metrics.csv or any .json context files     #\n"
            "#  NEVER recreate, simplify, or 'clean up' any existing EDA output   #\n"
            "#  These files were created by deterministic code and must NOT change #\n"
            "######################################################################"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _validate_insights_report(eda_dir: str) -> None:
    """
    Validate that insights report was created with sufficient content.

    Raises:
        EDAFailedError: If report doesn't exist or is insufficient
    """
    report_path = os.path.join(eda_dir, 'EDA_INSIGHTS_REPORT.md')

    if not os.path.exists(report_path):
        raise EDAFailedError(
            f"CRITICAL: EDA Insights Agent FAILED to create report\n\n"
            f"Expected file: {report_path}\n\n"
            "ROOT CAUSE: The agent did NOT execute the code to save the file.\n"
            "The agent likely described the analysis instead of running CodeExecutionTool.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )

    # Check content quality
    with open(report_path) as f:
        content = f.read()

    if len(content) < 2000:  # Minimum ~50 lines
        raise EDAFailedError(
            f"CRITICAL: EDA Insights Report is too short ({len(content)} chars)\n\n"
            "The agent likely did not complete the full analysis.\n"
            "Expected at least 2000 characters of insights."
        )

    # Check for placeholder text (indicates agent didn't fill in actual data)
    placeholders = ['[Write', '[Use', '[Based on', '[Recommend']
    placeholder_count = sum(1 for p in placeholders if p in content)
    if placeholder_count > 3:
        raise EDAFailedError(
            f"CRITICAL: EDA Insights Report contains {placeholder_count} placeholders\n\n"
            "The agent did not fill in actual statistics from the data.\n"
            "The report should contain specific numbers, not placeholder text."
        )

    logger.info(f"Insights report validation passed: {len(content):,} chars")


def _run_eda_insights_crew(
    eda_dir: str,
    config: DemandForecastConfig,
    llm: LLM,
    n_series: int,
) -> None:
    """
    Run EDA Insights Documentation crew.

    Creates EDA_INSIGHTS_REPORT.md through iterative code execution.
    The agent explores data files using CodeExecutionTool and builds
    a comprehensive insights report.

    Raises:
        EDAFailedError: If report generation fails
    """
    logger.info("="*70)
    logger.info("RUNNING EDA INSIGHTS CREW")
    logger.info("="*70)

    # Create the insights agent
    insights_agent = _create_eda_insights_agent(llm)

    # Build the task description with file paths
    task_description = f"""
# EDA INSIGHTS REPORT GENERATION

######################################################################
#  CRITICAL: READ-ONLY ACCESS TO ALL EXISTING FILES!                 #
#  You may ONLY CREATE one new file: EDA_INSIGHTS_REPORT.md          #
#  NEVER overwrite, modify, recreate, or 'clean up' existing files   #
#  NEVER write to per_key_metrics.csv, context .json files, etc.     #
#  These files were produced by deterministic code and MUST NOT change#
######################################################################

Analyze EDA outputs in `{eda_dir}` and create a comprehensive insights report.

## AVAILABLE FILES (READ-ONLY - DO NOT MODIFY!)

- `{eda_dir}/per_key_metrics.csv` - Main metrics for all {n_series:,} series
- `{eda_dir}/eda_summary.json` - Global summary statistics
- `{eda_dir}/seasonality_analysis.json` - Seasonality detection results
- `{eda_dir}/trend_analysis.json` - Trend analysis results
- `{eda_dir}/data_profile.json` - Data characterization
- `{eda_dir}/data_quality.json` - Data quality analysis
- `{eda_dir}/stationarity_results.csv` - Stationarity test results
- `{eda_dir}/feature_importance.csv` - Feature importance rankings

## CONFIG INFO

- Time format: {config.time_format}
- Key columns: {list(config.key_columns)}
- Target: {config.target_column}
- Training period: {config.train_start} to {config.train_end}
- Forecast horizon: {config.forecast_horizon}

## PHASE 1: EXPLORE DATA STRUCTURE

Execute this code to understand available data:

```python
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

eda_dir = '{eda_dir}'

# List files
files = [f for f in os.listdir(eda_dir) if f.endswith(('.csv', '.json', '.md'))]
print(f'Available files: {{len(files)}}')
for f in sorted(files)[:15]: print(f'  - {{f}}')

# Load main metrics
df = pd.read_csv(f'{{eda_dir}}/per_key_metrics.csv')
print(f'\\nPer-key metrics: {{df.shape[0]}} series, {{df.shape[1]}} columns')
print(f'Columns: {{list(df.columns)[:15]}}...')
```

## PHASE 2: DEMAND PATTERN ANALYSIS

Execute this code to analyze demand patterns:

```python
import pandas as pd
df = pd.read_csv('{eda_dir}/per_key_metrics.csv')

# Demand pattern distribution
print('=== DEMAND PATTERN DISTRIBUTION ===')
if 'demand_pattern' in df.columns:
    pattern_counts = df['demand_pattern'].value_counts()
    pattern_pcts = df['demand_pattern'].value_counts(normalize=True) * 100
    for p in pattern_counts.index:
        print(f'{{p}}: {{pattern_counts[p]:,}} series ({{pattern_pcts[p]:.1f}}%)')

# Volume tier distribution
print('\\n=== VOLUME TIER DISTRIBUTION ===')
if 'volume_tier' in df.columns:
    vol_counts = df['volume_tier'].value_counts()
    for v in vol_counts.index:
        print(f'{{v}}: {{vol_counts[v]:,}} series')
```

## PHASE 3: KEY METRICS ANALYSIS

Execute this code for detailed metrics:

```python
import pandas as pd
df = pd.read_csv('{eda_dir}/per_key_metrics.csv')

print('=== KEY METRICS ===')
if 'cv' in df.columns:
    print(f'CV - Mean: {{df["cv"].mean():.3f}}, Median: {{df["cv"].median():.3f}}, Max: {{df["cv"].max():.3f}}')
if 'zero_fraction' in df.columns:
    print(f'Zero Fraction - Mean: {{df["zero_fraction"].mean():.3f}}')
    high_intermittent = (df['zero_fraction'] > 0.5).sum()
    print(f'High intermittency (>50% zeros): {{high_intermittent}} series ({{high_intermittent/len(df)*100:.1f}}%)')
if 'adi' in df.columns:
    print(f'ADI - Mean: {{df["adi"].mean():.3f}}, Median: {{df["adi"].median():.3f}}')

# Cross-tabulation
print('\\n=== PATTERN x VOLUME CROSSTAB ===')
if 'demand_pattern' in df.columns and 'volume_tier' in df.columns:
    print(pd.crosstab(df['demand_pattern'], df['volume_tier'], margins=True))
```

## PHASE 4: LOAD JSON INSIGHTS

Execute this code to get additional context:

```python
import json
import os

eda_dir = '{eda_dir}'

def load_json(name):
    path = f'{{eda_dir}}/{{name}}'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {{}}

seasonality = load_json('seasonality_analysis.json')
trend = load_json('trend_analysis.json')
profile = load_json('data_profile.json')

print('=== SEASONALITY ===')
print(f'Has seasonality: {{seasonality.get("has_seasonality_pct", "N/A")}}')
print(f'Dominant period: {{seasonality.get("dominant_period", "N/A")}}')

print('\\n=== DATA PROFILE ===')
if profile:
    tg = profile.get('time_granularity', {{}})
    dq = profile.get('data_quality', {{}})
    print(f'Time granularity: {{tg.get("granularity", "N/A")}}')
    print(f'Data quality score: {{dq.get("quality_score", "N/A")}}')
```

## PHASE 5: GENERATE AND SAVE REPORT

After gathering all insights from the previous phases, execute code to CREATE and SAVE the report.

YOU MUST use the ACTUAL NUMBERS from your previous code executions. Do NOT use placeholders.

```python
from datetime import datetime
import os

eda_dir = '{eda_dir}'

# Build the markdown report using the statistics you gathered
# Replace ALL placeholders with actual numbers from your analysis

md = '''# EDA Insights Report
**Generated:** ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''
**Total Series:** {n_series:,}

---

## Executive Summary

[Write 3-5 paragraphs using the ACTUAL statistics from your analysis above.
Include specific numbers like: "X% of series are lumpy", "average CV is Y", etc.]

---

## 1. Demand Pattern Analysis

[Use the exact pattern_counts and pattern_pcts from Phase 2.
Create a markdown table with the actual numbers.]

---

## 2. Volume Tier Analysis

[Use the exact vol_counts from Phase 2.]

---

## 3. Intermittency Analysis

[Use CV, zero_fraction, ADI statistics from Phase 3.
Include the high_intermittent count.]

---

## 4. Seasonality & Trend Insights

[Use seasonality data from Phase 4.]

---

## 5. Cross-Segment Analysis

[Use the crosstab from Phase 3 to identify which pattern-volume
combinations are most common and which need special attention.]

---

## 6. Model Recommendations

Based on the demand patterns observed:
- For smooth series: LightGBM, XGBoost with MSE loss
- For erratic series: XGBoost, CatBoost with Huber loss
- For intermittent/lumpy series: Tweedie loss, zero-inflated models

---

## 7. Feature Engineering Recommendations

Based on the time format ({config.time_format}) and patterns:
- Lag features: [1, 2, 4, 13, 26, 52]
- Rolling windows: [4, 13, 26] periods
- Intermittency features if zero_fraction > 30%

---

## 8. Expected Challenges

[Based on the CV, zero_fraction, and data quality from your analysis.]

---

## 9. Performance Expectations

| Pattern | Expected WAPE |
|---------|---------------|
| Smooth | 15-30% |
| Erratic | 30-50% |
| Intermittent | 40-60% |
| Lumpy | 50-80% |

---

*Generated by EDA Insights Agent using iterative code execution.*
'''

# SAVE THE REPORT TO FILE
report_path = os.path.join(eda_dir, 'EDA_INSIGHTS_REPORT.md')
with open(report_path, 'w') as f:
    f.write(md)

print(f'\\nSaved: EDA_INSIGHTS_REPORT.md ({{len(md):,}} chars)')
print(f'Location: {{report_path}}')
```

## CRITICAL INSTRUCTIONS

1. You MUST execute code using CodeExecutionTool for EACH phase
2. Run Phases 1-4 FIRST to gather actual statistics
3. In Phase 5, replace ALL bracketed placeholders with REAL numbers from your analysis
4. The report MUST be saved to file using the code in Phase 5
5. Final report should have specific data-driven insights, not generic text
6. Print only confirmation after saving (not the entire report content)
7. NEVER OVERWRITE ANY EXISTING FILES - only create EDA_INSIGHTS_REPORT.md
8. NEVER write to per_key_metrics.csv, eda_summary.json, or any context .json files
9. NEVER recreate, simplify, or 'improve' any existing EDA output files
"""

    # Create the task
    insights_task = Task(
        name="generate_eda_insights_report",
        description=task_description,
        expected_output=(
            "Created EDA_INSIGHTS_REPORT.md - comprehensive insights report with "
            "demand pattern analysis, intermittency insights, and model recommendations."
        ),
        agent=insights_agent,
    )

    # Create and run crew
    insights_crew = Crew(
        name="EDA Insights Crew",
        agents=[insights_agent],
        tasks=[insights_task],
        process=Process.sequential,
        verbose=True,
    )

    # Run the crew (no try/except - let it fail if it fails)
    logger.info("Running EDA Insights agent...")
    insights_crew.kickoff()

    # Validate the output (raises EDAFailedError if invalid)
    _validate_insights_report(eda_dir)

    logger.info("="*70)
    logger.info("EDA INSIGHTS CREW COMPLETED SUCCESSFULLY")
    logger.info("="*70)


def _create_context_files_with_agents(
    eda_dir: str,
    config: DemandForecastConfig,
    llm: LLM,
) -> None:
    """
    Create context files and exhaustive insights report using LLM agent.

    This uses an "EDA Analyst" agent that:
    1. Reads all the deterministic EDA outputs
    2. Creates intelligent context files for downstream crews
    3. Generates an EXHAUSTIVE markdown insights report for data scientists

    The core EDA analysis is still deterministic - this only adds
    LLM interpretation for context files and insights report.
    """
    import json
    import pandas as pd
    import numpy as np

    logger.info("Creating context files and insights report with LLM Analyst agent...")

    # =========================================================================
    # Load ALL EDA outputs for comprehensive analysis
    # =========================================================================
    per_key_df = pd.read_csv(os.path.join(eda_dir, 'per_key_metrics.csv'))

    # Load all supporting files
    def load_json_safe(filepath):
        if os.path.exists(filepath):
            with open(filepath) as f:
                return json.load(f)
        return {}

    def load_csv_safe(filepath):
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None

    data_profile = load_json_safe(os.path.join(eda_dir, 'data_profile.json'))
    seasonality = load_json_safe(os.path.join(eda_dir, 'seasonality_analysis.json'))
    trend_analysis = load_json_safe(os.path.join(eda_dir, 'trend_analysis.json'))
    eda_summary_json = load_json_safe(os.path.join(eda_dir, 'eda_summary.json'))
    correlation_analysis = load_json_safe(os.path.join(eda_dir, 'correlation_analysis.json'))
    outlier_analysis = load_json_safe(os.path.join(eda_dir, 'outlier_analysis.json'))
    stationarity_results = load_json_safe(os.path.join(eda_dir, 'stationarity_results.json'))

    # Load CSV files
    feature_importance_df = load_csv_safe(os.path.join(eda_dir, 'feature_importance.csv'))
    correlation_matrix_df = load_csv_safe(os.path.join(eda_dir, 'feature_correlation_matrix.csv'))

    # =========================================================================
    # Compute comprehensive statistics
    # =========================================================================
    n_series = len(per_key_df)

    # Demand pattern distribution
    demand_patterns = per_key_df['demand_pattern'].value_counts(normalize=True).to_dict() if 'demand_pattern' in per_key_df.columns else {}
    demand_pattern_counts = per_key_df['demand_pattern'].value_counts().to_dict() if 'demand_pattern' in per_key_df.columns else {}

    # Volume tier distribution
    volume_tiers = per_key_df['volume_tier'].value_counts(normalize=True).to_dict() if 'volume_tier' in per_key_df.columns else {}
    volume_tier_counts = per_key_df['volume_tier'].value_counts().to_dict() if 'volume_tier' in per_key_df.columns else {}

    # Comprehensive metrics
    metric_cols = ['mean', 'std', 'cv', 'zero_fraction', 'adi', 'cv2', 'forecastability_score',
                   'trend_strength', 'seasonality_strength', 'entropy', 'stability']
    avg_metrics = {}
    median_metrics = {}
    std_metrics = {}
    min_metrics = {}
    max_metrics = {}
    percentile_25 = {}
    percentile_75 = {}

    for col in metric_cols:
        if col in per_key_df.columns:
            series = per_key_df[col].dropna()
            if len(series) > 0:
                avg_metrics[col] = round(float(series.mean()), 4)
                median_metrics[col] = round(float(series.median()), 4)
                std_metrics[col] = round(float(series.std()), 4)
                min_metrics[col] = round(float(series.min()), 4)
                max_metrics[col] = round(float(series.max()), 4)
                percentile_25[col] = round(float(series.quantile(0.25)), 4)
                percentile_75[col] = round(float(series.quantile(0.75)), 4)

    # Cross-tabulation of patterns and tiers
    pattern_tier_crosstab = {}
    if 'demand_pattern' in per_key_df.columns and 'volume_tier' in per_key_df.columns:
        crosstab = pd.crosstab(per_key_df['demand_pattern'], per_key_df['volume_tier'], normalize='all')
        pattern_tier_crosstab = crosstab.round(4).to_dict()

    # Feature importance summary
    feature_importance_summary = {}
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        if 'feature' in feature_importance_df.columns and 'importance' in feature_importance_df.columns:
            top_features = feature_importance_df.nlargest(10, 'importance')[['feature', 'importance']]
            feature_importance_summary = top_features.set_index('feature')['importance'].to_dict()

    # =========================================================================
    # Build comprehensive EDA summary for the agent
    # =========================================================================
    # Pre-compute percentage strings to avoid f-string dict comprehension issues
    demand_pattern_pcts = {k: f"{v*100:.1f}%" for k, v in demand_patterns.items()}
    volume_tier_pcts = {k: f"{v*100:.1f}%" for k, v in volume_tiers.items()}

    eda_full_summary = f"""
## COMPREHENSIVE EDA RESULTS

### 1. DATASET OVERVIEW
- **Total time series**: {n_series:,}
- **Time format**: {config.time_format}
- **Key columns**: {list(config.key_columns)}
- **Target variable**: {config.target_column}
- **Training period**: {config.train_start} to {config.train_end}
- **Forecast horizon**: {config.forecast_horizon} periods

### 2. DEMAND PATTERN DISTRIBUTION (Syntetos-Boylan Classification)
The Syntetos-Boylan classification categorizes demand based on:
- ADI (Average Demand Interval): measures intermittency
- CV² (Squared Coefficient of Variation): measures variability

**Distribution:**
{json.dumps(demand_pattern_counts, indent=2)}

**Percentages:**
{json.dumps(demand_pattern_pcts, indent=2)}

**Interpretation:**
- Smooth: Low variability, low intermittency (easiest to forecast)
- Erratic: High variability, low intermittency (challenging due to volatility)
- Intermittent: Low variability, high intermittency (sporadic demand)
- Lumpy: High variability AND high intermittency (most challenging)

### 3. VOLUME TIER DISTRIBUTION
**Counts:**
{json.dumps(volume_tier_counts, indent=2)}

**Percentages:**
{json.dumps(volume_tier_pcts, indent=2)}

### 4. PATTERN x TIER CROSS-TABULATION
{json.dumps(pattern_tier_crosstab, indent=2)}

### 5. KEY METRICS STATISTICS

| Metric | Mean | Median | Std | Min | 25% | 75% | Max |
|--------|------|--------|-----|-----|-----|-----|-----|
{chr(10).join([f"| {m} | {avg_metrics.get(m, 'N/A')} | {median_metrics.get(m, 'N/A')} | {std_metrics.get(m, 'N/A')} | {min_metrics.get(m, 'N/A')} | {percentile_25.get(m, 'N/A')} | {percentile_75.get(m, 'N/A')} | {max_metrics.get(m, 'N/A')} |" for m in metric_cols if m in avg_metrics])}

### 6. SEASONALITY ANALYSIS
{json.dumps(seasonality, indent=2) if seasonality else "Not available"}

### 7. TREND ANALYSIS
{json.dumps(trend_analysis, indent=2) if trend_analysis else "Not available"}

### 8. STATIONARITY RESULTS
{json.dumps(stationarity_results, indent=2) if stationarity_results else "Not available"}

### 9. CORRELATION ANALYSIS
{json.dumps(correlation_analysis, indent=2) if correlation_analysis else "Not available"}

### 10. OUTLIER ANALYSIS
{json.dumps(outlier_analysis, indent=2) if outlier_analysis else "Not available"}

### 11. TOP FEATURE IMPORTANCES (Random Forest)
{json.dumps(feature_importance_summary, indent=2) if feature_importance_summary else "Not available"}

### 12. DATA PROFILE
{json.dumps(data_profile, indent=2) if data_profile else "Not available"}

### 13. CONFIGURATION
- Numeric features: {config.all_numeric_features()}
- Categorical features: {config.all_categorical_features()}
"""

    # =========================================================================
    # Create the EDA Analyst agent
    # =========================================================================
    analyst_agent = Agent(
        role="Senior Data Science Analyst",
        goal="Analyze EDA results, create context files for downstream crews, and generate an exhaustive insights report",
        backstory="""You are a senior data scientist with 15+ years of experience in demand forecasting,
time series analysis, and machine learning. You have deep expertise in:
- Syntetos-Boylan demand classification and intermittent demand forecasting
- Time series decomposition, seasonality detection, and trend analysis
- Feature engineering for forecasting (lags, rolling windows, calendar features)
- Model selection for different demand patterns (tree-based, statistical, deep learning)
- Segmentation strategies for large-scale forecasting systems

Your reports are known for being thorough, insightful, and actionable. You explain not just
WHAT the data shows, but WHY it matters and HOW to use this information.""",
        llm=llm,
        tools=[CodeExecutionTool()],
        verbose=True,
        allow_delegation=False,
    )

    # =========================================================================
    # Task 1: Create Context Files
    # =========================================================================
    context_files_task = Task(
        description=f"""
Analyze the EDA results and create THREE context JSON files for downstream crews.

{eda_full_summary}

## YOUR TASK: Create Context Files

Execute Python code to create these three JSON files:

### 1. eda_to_segmentation_context.json
### 2. eda_to_feature_context.json
### 3. eda_to_training_context.json

```python
import json
import os

eda_dir = '{eda_dir}'

# 1. Segmentation Context
segmentation_context = {{
    "source": "EDA Analyst (LLM)",
    "analysis_summary": "<2-3 sentence summary of segmentation approach>",
    "n_series": {n_series},
    "demand_pattern_distribution": {json.dumps(demand_patterns)},
    "recommendations": {{
        "algorithm": "<gmm/kmeans/hdbscan - choose based on data>",
        "n_clusters_range": [<min>, <max>],
        "features_to_use": ["volume_mean", "cv_clean", "zero_fraction_clean", "adi_log", "demand_frequency"],
        "use_hybrid_segmentation": True,
        "hybrid_dimensions": ["volume_tier", "demand_pattern"],
        "rationale": "<detailed explanation of why this approach>"
    }},
    "data_characteristics": {{
        "avg_cv": {avg_metrics.get('cv', 1.0)},
        "avg_zero_fraction": {avg_metrics.get('zero_fraction', 0.3)},
        "avg_adi": {avg_metrics.get('adi', 1.5)},
        "has_seasonality": <True/False based on seasonality analysis>,
        "seasonal_period": <period if detected>
    }}
}}

with open(os.path.join(eda_dir, 'eda_to_segmentation_context.json'), 'w') as f:
    json.dump(segmentation_context, f, indent=2)

# 2. Feature Context
feature_context = {{
    "source": "EDA Analyst (LLM)",
    "analysis_summary": "<2-3 sentence summary of feature engineering approach>",
    "lag_features": {{
        "recommended_lags": [<based on time_format and seasonality>],
        "seasonal_lags": [<seasonal lags>],
        "rationale": "<why these specific lags>"
    }},
    "rolling_features": {{
        "windows": [<window sizes>],
        "aggregations": ["mean", "std", "min", "max"],
        "rationale": "<why these windows>"
    }},
    "intermittency_handling": {{
        "high_intermittency_pct": <% of lumpy+intermittent>,
        "use_demand_probability_features": <True if >30% intermittent>,
        "use_zero_inflated_features": <True if avg_zero_fraction > 0.3>,
        "rationale": "<based on demand patterns>"
    }},
    "external_features": {{
        "use_price_features": <True/False>,
        "use_promo_features": <True/False>,
        "recommended_lags": [0, 1, 2],
        "rationale": "<recommendations>"
    }}
}}

with open(os.path.join(eda_dir, 'eda_to_feature_context.json'), 'w') as f:
    json.dump(feature_context, f, indent=2)

# 3. Training Context
training_context = {{
    "source": "EDA Analyst (LLM)",
    "analysis_summary": "<2-3 sentence summary of model strategy>",
    "model_recommendations": {{
        "primary_models": ["lightgbm", "xgboost", <others based on data>],
        "intermittency_specialists": [<if needed: "croston", "sba", "tsb">],
        "seasonal_models": [<if seasonality: "sarima", "prophet", "tbats">],
        "all_recommended": [<full list>],
        "rationale": "<why these models for this data>"
    }},
    "training_strategy": {{
        "use_segment_specific_models": True,
        "use_ensemble": True,
        "cv_folds": 3,
        "rationale": "<why this strategy>"
    }},
    "data_characteristics": {{
        "n_series": {n_series},
        "high_intermittency": <True if lumpy+intermittent > 50%>,
        "has_seasonality": <True/False>,
        "avg_cv": {avg_metrics.get('cv', 1.0)}
    }},
    "hyperparameter_hints": {{
        "lightgbm": {{"num_leaves": "<range>", "learning_rate": "<range>"}},
        "xgboost": {{"max_depth": "<range>", "learning_rate": "<range>"}}
    }}
}}

with open(os.path.join(eda_dir, 'eda_to_training_context.json'), 'w') as f:
    json.dump(training_context, f, indent=2)

print("Context files created!")
```

Fill in ALL placeholders with real values based on the EDA results!
""",
        expected_output="Confirmation that all three context JSON files were created",
        agent=analyst_agent,
    )

    # =========================================================================
    # Task 2: Create Exhaustive Insights Report
    # =========================================================================
    insights_report_task = Task(
        description=f"""
Now create an EXHAUSTIVE markdown insights report for data scientists.

{eda_full_summary}

## YOUR TASK: Create eda_insights_report.md

This report should be COMPREHENSIVE and DETAILED - at least 500 lines.
A senior data scientist should be able to read this and fully understand:
1. The nature and characteristics of this demand data
2. The challenges and opportunities for forecasting
3. Specific recommendations with detailed rationale

Execute this Python code to create the report:

```python
import os

eda_dir = '{eda_dir}'

report = '''# Demand Forecasting EDA Insights Report
## Comprehensive Analysis for Data Scientists

**Generated by**: EDA Analyst (LLM)
**Dataset**: {config.input_data_path}
**Analysis Date**: <current date>

---

## Executive Summary

<Write 3-5 paragraphs summarizing:
- Key characteristics of the demand data
- Main forecasting challenges identified
- Top 3-5 actionable recommendations
- Expected forecasting difficulty level>

---

## 1. Dataset Overview

### 1.1 Data Dimensions
<Detailed description of the dataset size, structure, time range>

### 1.2 Key Columns Analysis
<Analysis of key columns: {config.key_columns}>

### 1.3 Target Variable: {config.target_column}
<Detailed analysis of the target variable distribution, range, characteristics>

---

## 2. Demand Pattern Analysis (Syntetos-Boylan Classification)

### 2.1 Classification Methodology
<Explain the Syntetos-Boylan framework:
- What ADI measures and its thresholds
- What CV² measures and its thresholds
- The 4 demand patterns and their characteristics>

### 2.2 Pattern Distribution in This Dataset
<Detailed breakdown of:
{json.dumps(demand_pattern_counts, indent=2)}

For each pattern, explain:
- Count and percentage
- What this means for forecasting
- Recommended approaches>

### 2.3 Pattern Implications for Forecasting
<For EACH pattern type, provide:
- Specific challenges
- Recommended model families
- Feature engineering considerations
- Expected accuracy ranges>

### 2.4 Intermittency Analysis
<Deep dive into intermittent and lumpy patterns:
- Total percentage of intermittent demand
- Characteristics of intermittent series
- Recommended specialized methods (Croston, SBA, TSB)>

---

## 3. Volume Tier Analysis

### 3.1 Volume Distribution
{json.dumps(volume_tier_counts, indent=2)}

### 3.2 Volume-Pattern Interaction
<Analysis of how volume tiers interact with demand patterns
- Which patterns are more common in which tiers
- Implications for segmented forecasting>

### 3.3 Volume-Based Recommendations
<Specific recommendations for each volume tier>

---

## 4. Statistical Metrics Deep Dive

### 4.1 Coefficient of Variation (CV)
- Mean CV: {avg_metrics.get('cv', 'N/A')}
- Median CV: {median_metrics.get('cv', 'N/A')}
- Range: {min_metrics.get('cv', 'N/A')} to {max_metrics.get('cv', 'N/A')}

<Interpretation:
- What this CV distribution tells us about demand volatility
- Comparison to industry benchmarks
- Implications for model selection>

### 4.2 Zero Fraction Analysis
- Mean zero fraction: {avg_metrics.get('zero_fraction', 'N/A')}
- Median: {median_metrics.get('zero_fraction', 'N/A')}

<Analysis of sparsity:
- Distribution of zero values
- Impact on standard forecasting methods
- Recommended handling approaches>

### 4.3 Average Demand Interval (ADI)
- Mean ADI: {avg_metrics.get('adi', 'N/A')}

<Interpretation:
- What ADI distribution reveals
- Comparison of ADI across segments>

### 4.4 Forecastability Score
<If available, analyze the forecastability distribution>

---

## 5. Seasonality Analysis

### 5.1 Seasonality Detection Results
{json.dumps(seasonality, indent=2) if seasonality else "Analysis not available"}

### 5.2 Seasonal Pattern Interpretation
<Detailed interpretation:
- Percentage of series with seasonality
- Dominant seasonal periods
- Strength of seasonality
- Recommendations for seasonal handling>

### 5.3 Calendar Effects
<Discussion of potential calendar effects:
- Day-of-week patterns
- Month-of-year patterns
- Holiday effects
- Promotional calendar alignment>

---

## 6. Trend Analysis

### 6.1 Trend Detection Results
{json.dumps(trend_analysis, indent=2) if trend_analysis else "Analysis not available"}

### 6.2 Trend Implications
<Analysis of:
- Percentage of series with significant trends
- Direction of trends (growing vs declining)
- Trend strength distribution
- Implications for forecasting>

---

## 7. Feature Analysis

### 7.1 Feature Importance Rankings
{json.dumps(feature_importance_summary, indent=2) if feature_importance_summary else "Not available"}

### 7.2 Feature Engineering Recommendations

#### 7.2.1 Lag Features
<Specific recommendations:
- Which lags to include and why
- Seasonal lags
- Interaction with demand patterns>

#### 7.2.2 Rolling Window Features
<Recommendations:
- Window sizes to use
- Aggregations to compute
- Pattern-specific considerations>

#### 7.2.3 Calendar Features
<What calendar features to engineer>

#### 7.2.4 External Feature Handling
<Recommendations for external regressors>

---

## 8. Segmentation Strategy

### 8.1 Recommended Segmentation Approach
<Detailed recommendations:
- Primary segmentation method
- Number of segments
- Features to use for clustering>

### 8.2 Segment-Specific Strategies
<For each expected segment type, recommend:
- Model families
- Feature emphasis
- Expected performance>

### 8.3 Hybrid Segmentation Rationale
<Explain why hybrid segmentation (statistical clusters + business dimensions) is/isn't recommended>

---

## 9. Model Recommendations

### 9.1 Primary Model Families
<Ranked recommendations with detailed rationale>

### 9.2 Intermittency Specialists
<If applicable, which specialized methods to use:
- Croston's method
- SBA (Syntetos-Boylan Approximation)
- TSB (Teunter-Syntetos-Babai)
- Zero-inflated models>

### 9.3 Ensemble Strategy
<Recommendations for ensembling:
- Which models to combine
- Weighting strategy
- Expected improvement>

### 9.4 Hyperparameter Guidelines
<Specific hyperparameter recommendations based on data characteristics>

---

## 10. Expected Challenges and Mitigations

### 10.1 Data Quality Challenges
<Identify and address:
- Missing data patterns
- Outliers
- Data inconsistencies>

### 10.2 Forecasting Challenges
<Specific challenges:
- High intermittency
- Volatility
- Trend changes
- New product introductions>

### 10.3 Recommended Mitigations
<Actionable mitigations for each challenge>

---

## 11. Performance Expectations

### 11.1 Expected Accuracy Ranges
<Realistic expectations:
- By segment
- By demand pattern
- By volume tier>

### 11.2 Key Performance Metrics
<Recommended evaluation metrics:
- For smooth demand
- For intermittent demand
- Overall portfolio metrics>

---

## 12. Implementation Roadmap

### 12.1 Priority 1: Quick Wins
<Features and models to implement first>

### 12.2 Priority 2: Core Implementation
<Main forecasting pipeline components>

### 12.3 Priority 3: Advanced Optimizations
<Longer-term improvements>

---

## Appendix A: Metric Definitions

<Define all metrics used in this report>

## Appendix B: Methodology Notes

<Explain methodologies used for analysis>

## Appendix C: Data Dictionary

<Column definitions from per_key_metrics.csv>

---

*This report was generated by the EDA Analyst agent to provide comprehensive insights for demand forecasting.*
'''

with open(os.path.join(eda_dir, 'eda_insights_report.md'), 'w') as f:
    f.write(report)

print("Exhaustive insights report created: eda_insights_report.md")
```

CRITICAL: Replace ALL placeholders with ACTUAL analysis and insights.
The report should be at least 500 lines with REAL, SPECIFIC insights.
Do NOT leave any angle brackets <> or placeholder text!
""",
        expected_output="Confirmation that eda_insights_report.md was created with comprehensive analysis",
        agent=analyst_agent,
    )

    # =========================================================================
    # Create crew with both tasks
    # =========================================================================
    analyst_crew = Crew(
        agents=[analyst_agent],
        tasks=[context_files_task, insights_report_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        # Run the analyst
        analyst_crew.kickoff()

        # Validate outputs exist
        required_files = [
            'eda_to_segmentation_context.json',
            'eda_to_feature_context.json',
            'eda_to_training_context.json',
            'eda_insights_report.md',
        ]

        missing_files = []
        for filename in required_files:
            filepath = os.path.join(eda_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            logger.warning(f"Analyst failed to create: {missing_files}")
            # Fall back to deterministic for context files
            if any('context' in f for f in missing_files):
                logger.info("Falling back to deterministic context files...")
                _create_context_files_deterministic(eda_dir, config, per_key_df)
        else:
            logger.info("LLM Analyst created all files successfully:")
            for f in required_files:
                filepath = os.path.join(eda_dir, f)
                size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                logger.info(f"  - {f} ({size:,} bytes)")

    except Exception as e:
        logger.warning(f"LLM Analyst failed: {e}. Falling back to deterministic context files.")
        _create_context_files_deterministic(eda_dir, config, per_key_df)
