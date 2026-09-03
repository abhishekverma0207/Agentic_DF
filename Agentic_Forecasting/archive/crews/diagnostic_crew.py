# crews/diagnostic_crew.py
"""
State-of-the-Art Diagnostic Crew for Demand Forecasting - INTELLIGENT UTILITY ORCHESTRATION

This crew uses the 3-Agent Pattern with INTELLIGENT utility function orchestration:
1. Diagnostic Planner: Reads training_to_diagnostic_context.json, extracts underperformer info,
   model specs, and demand patterns, then creates diagnostic_strategy.json specifying
   which diagnostic utilities to run and what to focus on.
2. Diagnostic Executor: Loads strategy and ORCHESTRATES individual diagnostic utilities
   (compute_metrics_by_group, identify_top_bottom_performers, diagnose_forecast_failures, etc.)
   based on the strategy.
3. Diagnostic Analyst: Creates INTELLIGENT feedback context with model verdict,
   underperformer analysis, and recommendations for future improvements.

KEY PRINCIPLE: Do NOT blindly call run_diagnostic_pipeline(). Instead:
- Read upstream context for underperformers and model selection issues
- Select appropriate diagnostic functions to analyze specific problems
- Orchestrate individual utility calls to create targeted insights

KEY DELIVERABLES:
- Overall and per-segment WAPE metrics
- Top 10 best performing keys with success factors
- Bottom 10 worst performing keys with failure analysis
- Model verdict: Was the approach appropriate for the data?
- Actionable recommendations for improvement
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from crewai import Agent, Crew, Task, Process, LLM

from config.schema import DemandForecastConfig
from utils.code_execution_tool import CodeExecutionTool

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticCrewResult:
    """Container with the main diagnostics artifact paths."""
    group_diagnostics_path: str
    segment_diagnostics_path: str
    key_diagnostics_path: str
    diagnostics_summary_path: str
    diagnostics_report_markdown_path: str
    diagnostics_dir: str
    # Top/Bottom analysis
    top_bottom_analysis_path: str = ""
    model_verdict_path: str = ""
    # Generated pipeline
    diagnostic_pipeline_script_path: str = ""
    # Focused context output
    diagnostic_feedback_context_path: str = ""
    # Cost tracking
    cost_report_path: str = ""


def _get_output_path(absolute_path: str) -> str:
    """
    Get a safe path for CrewAI Task output_file parameter.

    CrewAI 1.9.1+ rejects paths with '..' (path traversal) for security.
    On Databricks, relative paths from cwd to /Volumes/ create '../..' paths.

    Solution: Use absolute paths which work for both Bedrock and Databricks.
    """
    return os.path.abspath(absolute_path)


def _create_diagnostic_planner_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Diagnostic Planner agent - reads upstream context and creates diagnostic strategy.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="diagnostic_planner",
        role="Diagnostic Strategy Code Executor",
        goal=(
            "EXECUTE the exact Python code provided in the task description using CodeExecutionTool. "
            "DO NOT describe what the code does. DO NOT write your own code. "
            "JUST RUN the provided code block to create diagnostic_strategy.json."
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
            "4. The code will create diagnostic_strategy.json\n\n"
            "## WHAT YOU MUST NOT DO:\n"
            "- Do NOT describe what the code should do\n"
            "- Do NOT write a JSON response describing the strategy\n"
            "- Do NOT skip the CodeExecutionTool call\n"
            "- Do NOT modify or rewrite the provided code\n\n"
            "## CORRECT BEHAVIOR:\n"
            "1. See task with Python code block\n"
            "2. Use CodeExecutionTool to execute that code\n"
            "3. Code creates diagnostic_strategy.json file\n"
            "4. Report that the file was created\n\n"
            "## OUTPUT LIMIT: MAX 5 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_diagnostic_executor_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Diagnostic Executor agent - orchestrates individual diagnostic utilities.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="diagnostic_executor",
        role="Diagnostic Pipeline Code Executor",
        goal=(
            "EXECUTE the exact Python code provided in the task description using CodeExecutionTool. "
            "DO NOT describe what the code does. DO NOT write your own code. "
            "JUST RUN the provided code block to run diagnostics."
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
            "4. The code will run diagnostic utilities and save outputs\n\n"
            "## WHAT YOU MUST NOT DO:\n"
            "- Do NOT describe what the code should do\n"
            "- Do NOT write a response describing what would happen\n"
            "- Do NOT skip the CodeExecutionTool call\n"
            "- Do NOT modify or rewrite the provided code\n\n"
            "## CORRECT BEHAVIOR:\n"
            "1. See task with Python code block\n"
            "2. Use CodeExecutionTool to execute that code\n"
            "3. Code runs diagnostics and saves files\n"
            "4. Report the diagnostic results\n\n"
            "## OUTPUT LIMIT: MAX 10 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_diagnostic_analyst_agent(llm: LLM, allowed_model_families: list, protected_paths: list = None) -> Agent:
    """
    Create the Diagnostic Analyst agent - analyzes results and creates feedback context.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="diagnostic_analyst",
        role="Diagnostic Analysis Code Executor",
        goal=(
            "EXECUTE the exact Python code provided in the task description using CodeExecutionTool. "
            "DO NOT describe what the code does. DO NOT write your own code. "
            "JUST RUN the provided code block to create feedback context."
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
            "4. The code will analyze results and create feedback context\n\n"
            "## WHAT YOU MUST NOT DO:\n"
            "- Do NOT describe what the code should do\n"
            "- Do NOT write a response describing the analysis\n"
            "- Do NOT skip the CodeExecutionTool call\n"
            "- Do NOT modify or rewrite the provided code\n\n"
            "## CORRECT BEHAVIOR:\n"
            "1. See task with Python code block\n"
            "2. Use CodeExecutionTool to execute that code\n"
            "3. Code creates feedback context JSON\n"
            "4. Report that the file was created\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n\n"
            "## OUTPUT LIMIT: MAX 5 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_diagnostic_reviewer_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Diagnostic Reviewer agent that validates diagnostic outputs.
    This agent is OPTIONAL and only used when config.design.enable_reviewer is True.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="diagnostic_reviewer",
        role="Diagnostic Quality Assurance & Validation Specialist",
        goal=(
            "Review and validate all diagnostic outputs. Ensure WAPE metrics are correct, "
            "model verdicts are justified, and recommendations are actionable."
        ),
        backstory=(
            "You are a senior data scientist specializing in model diagnostics.\n\n"
            "######################################################################\n"
            "#  OUTPUT LIMIT: MAX 10 PRINT STATEMENTS                            #\n"
            "######################################################################\n"
            "```python\n"
            "# Validation pattern:\n"
            "files = ['file1.csv', 'file2.json']\n"
            "exists = [f for f in files if os.path.exists(f)]\n"
            "print(f'Files: {len(exists)}/{len(files)} present')\n"
            "# Save report, print path only\n"
            "```\n"
            "######################################################################\n\n"
            "## YOUR MISSION\n"
            "1. Validate diagnostic files exist\n"
            "2. Verify WAPE metrics are reasonable\n"
            "3. Check model verdict is justified\n"
            "4. Create diagnostic_review_report.json with quality score (1-10)"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_diagnostic_documentation_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Diagnostic Documentation Agent that generates comprehensive markdown documentation.

    This agent uses CodeExecutionTool to ITERATIVELY explore diagnostic outputs
    and creates a comprehensive insights markdown report through multiple phases.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="diagnostic_documentation_agent",
        role="PhD-Level Model Diagnostics Insights Documentation Specialist",
        goal=(
            "EXECUTE Python code using CodeExecutionTool to explore diagnostic outputs "
            "and create DIAGNOSTICS_INSIGHTS_GUIDE.md. You MUST use the tool to run code - "
            "do NOT just describe or print the analysis."
        ),
        backstory=(
            "You are a PhD-level expert in model diagnostics and performance analysis.\n\n"
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE CodeExecutionTool TO RUN ANALYSIS CODE    #\n"
            "#  EXPLORE DATA ITERATIVELY - RUN MULTIPLE CODE EXECUTIONS          #\n"
            "#  SAVE FINAL REPORT TO FILE - DO NOT JUST PRINT IT                 #\n"
            "######################################################################\n\n"
            "## YOUR EXPERTISE:\n"
            "- Model performance diagnostics (WAPE, MAE, RMSE, Bias)\n"
            "- Root cause analysis for underperforming forecasts\n"
            "- Segment-level performance comparison\n"
            "- Actionable recommendations for improvement\n\n"
            "## HOW TO COMPLETE THIS TASK:\n"
            "1. Use CodeExecutionTool to run exploration code\n"
            "2. Run MULTIPLE code executions to build understanding\n"
            "3. Extract specific statistics (WAPE, top/bottom performers, root causes)\n"
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


def create_diagnostic_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> Crew:
    """
    Create the Diagnostic Crew with INTELLIGENT UTILITY ORCHESTRATION.

    This crew reads upstream context and orchestrates individual diagnostic utilities
    based on training results and identified issues.

    The crew has three phases:
    1. PLANNING: Read upstream context, create diagnostic_strategy.json with focus areas
    2. EXECUTION: Orchestrate individual diagnostic utilities (compute_metrics_by_group, etc.)
    3. ANALYSIS: Create intelligent feedback context for future improvements

    KEY OUTPUTS:
    - key_diagnostics.csv: Per-key WAPE, MAE, RMSE, bias
    - segment_diagnostics.csv: Per-segment metrics
    - top_bottom_analysis.json: Top 10 / Bottom 10 with explanations
    - model_verdict.json: Assessment of modeling approach
    - diagnostic_feedback_context.json: Recommendations for retraining
    """
    artifact_base = config.artifact_base_path
    model_dir = os.path.join(artifact_base, "model_artifacts")
    seg_dir = os.path.join(artifact_base, "seg_output")
    feat_dir = os.path.join(artifact_base, "feature_output")
    eda_dir = os.path.join(artifact_base, "eda_output")
    diag_dir = os.path.join(artifact_base, "diagnostics_output")

    os.makedirs(diag_dir, exist_ok=True)

    # Get safe output path for CrewAI Task output_file
    # Note: CrewAI 1.9.1+ rejects relative paths with '..' for security
    diag_dir_out = _get_output_path(diag_dir)

    # Get config details
    target_col = config.target_column
    date_col = config.date_column
    key_columns = config.key_columns

    # Get categorical features from config for WAPE-by-feature analysis
    all_categorical_features = config.all_categorical_features()

    # Get allowed model families from config.design
    allowed_model_families = list(config.design.model_families)
    enable_deep_models = config.design.enable_deep_models

    # Filter out deep learning models if disabled
    deep_model_types = ['tft', 'lstm', 'nbeats', 'deepar', 'wavenet']
    if not enable_deep_models:
        allowed_model_families = [m for m in allowed_model_families if m.lower() not in deep_model_types]

    # WRITE PROTECTION: Prevent LLM agents from corrupting upstream outputs
    protected_dirs = [eda_dir, seg_dir, feat_dir, model_dir]

    # Create agents
    planner = _create_diagnostic_planner_agent(llm, protected_paths=protected_dirs)
    executor = _create_diagnostic_executor_agent(llm, protected_paths=protected_dirs)
    analyst = _create_diagnostic_analyst_agent(llm, allowed_model_families, protected_paths=protected_dirs)

    # -------------------------------------------------------------------------
    # Task 1: Diagnostic Planning - READ UPSTREAM CONTEXT, CREATE STRATEGY
    # -------------------------------------------------------------------------
    task_plan = Task(
        name="create_diagnostic_strategy",
        description=(
            "# EXECUTE THIS CODE USING CodeExecutionTool\n\n"
            "**CRITICAL: You MUST use CodeExecutionTool to execute the Python code below.**\n"
            "**DO NOT describe what the code does. DO NOT write your own version.**\n"
            "**JUST COPY THE CODE BLOCK INTO CodeExecutionTool AND RUN IT.**\n\n"
            "The code creates diagnostic_strategy.json from training context.\n\n"
            "## PYTHON CODE TO EXECUTE:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n"
            "from utils.agent_utilities import load_json, save_json\n"
            "from utils.context_schema import ContextReader, SemanticTypes\n\n"
            f"model_dir = '{model_dir}'\n"
            f"seg_dir = '{seg_dir}'\n"
            f"diag_dir = '{diag_dir}'\n\n"
            "# Read upstream context using ContextReader (schema-aware)\n"
            "training_ctx_path = os.path.join(model_dir, 'training_to_diagnostic_context.json')\n"
            "model_specs_path = os.path.join(model_dir, 'final_model_specs.json')\n\n"
            "# Load training context using ContextReader for robust key access\n"
            "overall_perf = {}\n"
            "underperformer_analysis = {}\n"
            "model_performance = []\n"
            "diagnostic_priorities = []\n"
            "if os.path.exists(training_ctx_path):\n"
            "    train_reader = ContextReader(training_ctx_path)\n"
            "    overall_perf = train_reader.get_by_type(SemanticTypes.TRAINING_RESULTS) or train_reader.get('overall_performance', {})\n"
            "    underperformer_analysis = train_reader.get_by_type(SemanticTypes.UNDERPERFORMER_ANALYSIS) or train_reader.get('underperformer_analysis', {})\n"
            "    model_performance = train_reader.get_by_type(SemanticTypes.MODEL_PERFORMANCE) or train_reader.get('model_performance', [])\n"
            "    diagnostic_priorities = train_reader.get_by_type(SemanticTypes.DIAGNOSTIC_PRIORITIES) or train_reader.get('diagnostic_priorities', [])\n\n"
            "model_specs = load_json(model_specs_path) if os.path.exists(model_specs_path) else {}\n\n"
            "# Extract key information\n"
            "high_wape_groups = underperformer_analysis.get('high_wape_groups', [])\n\n"
            "# Build per_model_perf from model_performance list\n"
            "per_model_perf = {}\n"
            "for mp in model_performance:\n"
            "    if isinstance(mp, dict):\n"
            "        mg = mp.get('model_group', 'unknown')\n"
            "        per_model_perf[mg] = mp\n\n"
            "# Model mismatches (not present in new schema, so empty by default)\n"
            "mismatched_models = []\n\n"
            "# Build diagnostic strategy\n"
            "strategy = {\n"
            "    'focus_segments': high_wape_groups[:5],  # Top 5 worst performers\n"
            "    'model_mismatches': mismatched_models,\n"
            "    'overall_wape': overall_perf.get('overall_wape', 1.0),\n"
            "    'analysis_priorities': [\n"
            "        'compute_metrics_by_group for segment-level metrics',\n"
            "        'identify_top_bottom_performers for key-level analysis',\n"
            "        'decompose_forecast_error for bias/variance',\n"
            "    ],\n"
            "    'expected_issues': underperformer_analysis.get('recommendations', {}),\n"
            "    'per_model_perf_summary': {\n"
            "        k: {'wape': v.get('wape'), 'model_type': v.get('model_type'), 'demand_pattern': v.get('demand_pattern')}\n"
            "        for k, v in per_model_perf.items()\n"
            "    },\n"
            "}\n\n"
            "# Add focus analysis based on issues found\n"
            "if high_wape_groups:\n"
            "    strategy['analysis_priorities'].append('diagnose_forecast_failures for underperformers')\n"
            "if mismatched_models:\n"
            "    strategy['analysis_priorities'].append('analyze_error_by_demand_pattern')\n\n"
            "# Save strategy\n"
            "save_json(strategy, os.path.join(diag_dir, 'diagnostic_strategy.json'))\n\n"
            "print(f'Diagnostic strategy created: {len(high_wape_groups)} focus segments')\n"
            "print(f'Model mismatches to investigate: {len(mismatched_models)}')\n"
            "print('Saved: diagnostic_strategy.json')\n"
            "```\n\n"
            "Then tell Executor: 'Use diagnostic_strategy.json to orchestrate individual diagnostic utilities.'"
        ),
        agent=planner,
        expected_output=(
            "Created diagnostic_strategy.json with focus segments, model mismatches, and analysis priorities. "
            "Executor should use individual diagnostic utilities based on the strategy."
        ),
        output_file=os.path.join(diag_dir_out, "diagnostic_planning_report.md"),
    )

    # -------------------------------------------------------------------------
    # Task 2: Diagnostic Execution - ORCHESTRATE INDIVIDUAL UTILITIES
    # -------------------------------------------------------------------------
    task_execute = Task(
        name="execute_diagnostic_orchestration",
        description=(
            "# EXECUTE THIS CODE USING CodeExecutionTool\n\n"
            "**CRITICAL: You MUST use CodeExecutionTool to execute the Python code below.**\n"
            "**DO NOT describe what the code does. DO NOT write your own version.**\n"
            "**JUST COPY THE CODE BLOCK INTO CodeExecutionTool AND RUN IT.**\n\n"
            "The code runs diagnostic utilities and saves outputs.\n\n"
            "## PYTHON CODE TO EXECUTE:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json, glob\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from utils.agent_utilities import load_csv, load_json, save_json, SmartPrinter\n"
            "from utils.diagnostics import (\n"
            "    compute_all_metrics, compute_metrics_by_group,\n"
            "    identify_top_bottom_performers, decompose_forecast_error,\n"
            "    diagnose_forecast_failures, generate_model_verdict,\n"
            "    classify_forecast_quality,\n"
            "    plot_actual_vs_predicted, plot_error_distribution,\n"
            "    plot_segment_comparison,\n"
            ")\n\n"
            f"model_dir = '{model_dir}'\n"
            f"seg_dir = '{seg_dir}'\n"
            f"diag_dir = '{diag_dir}'\n"
            f"target_col = '{target_col}'\n"
            f"key_col = '{key_columns[0]}'\n\n"
            "printer = SmartPrinter(max_prints=15)\n\n"
            "# Load strategy from Planner (with fallback if file doesn't exist)\n"
            "strategy_path = os.path.join(diag_dir, 'diagnostic_strategy.json')\n"
            "strategy = load_json(strategy_path) if os.path.exists(strategy_path) else {'focus_segments': [], 'per_model_perf_summary': {}}\n"
            "focus_segments = strategy.get('focus_segments', [])\n\n"
            "# PRIORITY: Load inference forecast results (true model performance on test data)\n"
            "# Backtest results use rolling-origin evaluation with recursive forecasting\n"
            "forecast_path = os.path.join(model_dir, 'inference_forecast.csv')\n"
            "forecasts = pd.DataFrame()\n"
            "if os.path.exists(forecast_path):\n"
            "    forecasts = pd.read_csv(forecast_path)\n"
            "    printer.print(f'Loaded inference forecasts: {len(forecasts)} rows (true model performance)')\n"
            "else:\n"
            "    # Fallback 1: validation_predictions.csv from training crew\n"
            "    val_pred_path = os.path.join(model_dir, 'validation_predictions.csv')\n"
            "    if os.path.exists(val_pred_path):\n"
            "        forecasts = pd.read_csv(val_pred_path)\n"
            "        printer.print(f'Loaded validation predictions: {len(forecasts)} rows (from training crew)')\n"
            "    else:\n"
            "        # Fallback 2: glob pattern for other forecast files\n"
            "        forecast_files = glob.glob(os.path.join(model_dir, '*_val_forecasts.csv'))\n"
            "        if not forecast_files:\n"
            "            forecast_files = glob.glob(os.path.join(model_dir, '*forecasts*.csv'))\n"
            "        all_forecasts = []\n"
            "        for f in forecast_files:\n"
            "            df = pd.read_csv(f)\n"
            "            all_forecasts.append(df)\n"
            "        forecasts = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()\n"
            "        printer.print(f'Loaded validation forecasts: {len(forecasts)} rows from {len(forecast_files)} files')\n\n"
            "# Identify columns - check multiple possible column names\n"
            "# actual column: 'target' (inference), 'actual' (validation_predictions), or config target_col\n"
            "if 'target' in forecasts.columns:\n"
            "    actual_col = 'target'\n"
            "elif 'actual' in forecasts.columns:\n"
            "    actual_col = 'actual'\n"
            "else:\n"
            "    actual_col = target_col\n"
            "pred_col = 'prediction' if 'prediction' in forecasts.columns else 'predicted'\n"
            "key_col_use = 'key' if 'key' in forecasts.columns else key_col\n\n"
            "# VALIDATION: Ensure required columns exist before proceeding\n"
            "if len(forecasts) > 0:\n"
            "    missing_cols = []\n"
            "    if actual_col not in forecasts.columns:\n"
            "        missing_cols.append(f'actual ({actual_col})')\n"
            "    if pred_col not in forecasts.columns:\n"
            "        missing_cols.append(f'prediction ({pred_col})')\n"
            "    if key_col_use not in forecasts.columns:\n"
            "        missing_cols.append(f'key ({key_col_use})')\n"
            "    if missing_cols:\n"
            "        printer.print(f'ERROR: Missing required columns: {missing_cols}')\n"
            "        printer.print(f'Available columns: {list(forecasts.columns)}')\n"
            "        # Set forecasts to empty to trigger placeholder creation\n"
            "        forecasts = pd.DataFrame()\n\n"
            "# Load segment info if available\n"
            "seg_path = os.path.join(seg_dir, 'per_key_with_segments.csv')\n"
            "if os.path.exists(seg_path) and 'segment_id' not in forecasts.columns:\n"
            "    seg_df = pd.read_csv(seg_path)\n"
            "    if key_col_use in forecasts.columns and 'key' in seg_df.columns:\n"
            "        # Select segment columns (exclude 'key' if merging on same column to avoid duplicates)\n"
            "        seg_cols = ['key', 'segment_id', 'intermittency_class', 'demand_pattern']\n"
            "        seg_cols = [c for c in seg_cols if c in seg_df.columns]\n"
            "        forecasts = forecasts.merge(seg_df[seg_cols], left_on=key_col_use, right_on='key', how='left', suffixes=('', '_seg'))\n"
            "        # Drop duplicate key column if created\n"
            "        if 'key_seg' in forecasts.columns:\n"
            "            forecasts = forecasts.drop(columns=['key_seg'])\n\n"
            "if len(forecasts) == 0:\n"
            "    printer.print('WARNING: No forecast data found - creating placeholder outputs')\n"
            "    # Create placeholder files so subsequent tasks don't fail\n"
            "    placeholder_summary = {\n"
            "        'overall_metrics': {'wape': 1.0, 'mae': 0, 'rmse': 0, 'bias': 0, 'smape': 0, 'quality_tier': 'UNKNOWN', 'n_obs': 0},\n"
            "        'error_decomposition': {'bias_pct': 0, 'variance_pct': 0, 'systematic_error': 0, 'random_error': 0},\n"
            "        'n_keys': 0,\n"
            "        'n_segments': 0,\n"
            "        'failure_analysis': {},\n"
            "    }\n"
            "    placeholder_verdict = {\n"
            "        'overall_wape': 1.0,\n"
            "        'quality_tier': 'UNKNOWN',\n"
            "        'deployment_ready': False,\n"
            "        'confidence_score': 0,\n"
            "        'strengths': [],\n"
            "        'weaknesses': ['No forecast data available - training has not completed'],\n"
            "        'recommendations': ['Run training crew first to generate model forecasts'],\n"
            "        'segment_verdicts': {},\n"
            "    }\n"
            "    placeholder_top_bottom = {\n"
            "        'top_performers': [],\n"
            "        'bottom_performers': [],\n"
            "        'analysis': {'top_avg_wape': None, 'bottom_avg_wape': None},\n"
            "    }\n"
            "    save_json(placeholder_summary, os.path.join(diag_dir, 'diagnostics_summary.json'))\n"
            "    save_json(placeholder_verdict, os.path.join(diag_dir, 'model_verdict.json'))\n"
            "    save_json(placeholder_top_bottom, os.path.join(diag_dir, 'top_bottom_analysis.json'))\n"
            "    pd.DataFrame(columns=['segment_id', 'wape', 'mae', 'rmse']).to_csv(os.path.join(diag_dir, 'segment_diagnostics.csv'), index=False)\n"
            "    pd.DataFrame(columns=['key', 'wape', 'mae', 'rmse']).to_csv(os.path.join(diag_dir, 'key_diagnostics.csv'), index=False)\n"
            "    printer.print('Created placeholder diagnostic files')\n"
            "else:\n"
            "    # 1. Compute overall metrics\n"
            "    overall = compute_all_metrics(forecasts[actual_col], forecasts[pred_col])\n"
            "    printer.print(f'Overall WAPE: {overall.wape:.2%}, Quality: {overall.quality_tier}')\n\n"
            "    # 2. Compute segment metrics (FIXED: segment_id is the actual column from segmentation)\n"
            "    seg_metrics = pd.DataFrame()\n"
            "    if 'segment_id' in forecasts.columns:\n"
            "        seg_metrics = compute_metrics_by_group(forecasts, 'segment_id', actual_col, pred_col)\n"
            "        seg_metrics.to_csv(os.path.join(diag_dir, 'segment_diagnostics.csv'), index=False)\n"
            "    elif 'model_group' in forecasts.columns:\n"
            "        seg_metrics = compute_metrics_by_group(forecasts, 'model_group', actual_col, pred_col)\n"
            "        seg_metrics.to_csv(os.path.join(diag_dir, 'segment_diagnostics.csv'), index=False)\n"
            "    elif 'model_level' in forecasts.columns:\n"
            "        seg_metrics = compute_metrics_by_group(forecasts, 'model_level', actual_col, pred_col)\n"
            "        seg_metrics.to_csv(os.path.join(diag_dir, 'segment_diagnostics.csv'), index=False)\n\n"
            "    # 3. Compute key-level metrics\n"
            "    key_metrics = compute_metrics_by_group(forecasts, key_col_use, actual_col, pred_col)\n"
            "    key_metrics.to_csv(os.path.join(diag_dir, 'key_diagnostics.csv'), index=False)\n\n"
            "    # 4. Identify top/bottom performers\n"
            "    top_df, bottom_df = identify_top_bottom_performers(\n"
            "        forecasts, key_col_use, actual_col, pred_col, n=10\n"
            "    )\n"
            "    top_df.to_csv(os.path.join(diag_dir, 'top_performers.csv'), index=False)\n"
            "    bottom_df.to_csv(os.path.join(diag_dir, 'bottom_performers.csv'), index=False)\n"
            "    \n"
            "    # Create top_bottom_analysis.json\n"
            "    top_bottom = {\n"
            "        'top_performers': top_df.to_dict('records'),\n"
            "        'bottom_performers': bottom_df.to_dict('records'),\n"
            "        'analysis': {\n"
            "            'top_avg_wape': float(top_df['wape'].mean()) if len(top_df) > 0 else None,\n"
            "            'bottom_avg_wape': float(bottom_df['wape'].mean()) if len(bottom_df) > 0 else None,\n"
            "        }\n"
            "    }\n"
            "    save_json(top_bottom, os.path.join(diag_dir, 'top_bottom_analysis.json'))\n\n"
            "    # 5. Error decomposition\n"
            "    decomp = decompose_forecast_error(forecasts[actual_col], forecasts[pred_col])\n\n"
            "    # 6. Diagnose failures in focus segments\n"
            "    failure_analysis = {}\n"
            "    for seg in focus_segments[:3]:  # Top 3 focus segments\n"
            "        seg_col = 'segment_id' if 'segment_id' in forecasts.columns else ('model_group' if 'model_group' in forecasts.columns else 'model_level')\n"
            "        if seg_col in forecasts.columns:\n"
            "            seg_df = forecasts[forecasts[seg_col] == seg]\n"
            "            if len(seg_df) > 0:\n"
            "                failures = diagnose_forecast_failures(\n"
            "                    seg_df, key_col_use, actual_col, pred_col, wape_threshold=0.5\n"
            "                )\n"
            "                failure_analysis[str(seg)] = failures\n\n"
            "    # 7. Generate model verdict\n"
            "    verdict = generate_model_verdict(overall, seg_metrics, decomp)\n"
            "    \n"
            "    # Save model verdict\n"
            "    verdict_dict = {\n"
            "        'overall_wape': verdict.overall_wape,\n"
            "        'quality_tier': verdict.quality_tier,\n"
            "        'deployment_ready': verdict.deployment_ready,\n"
            "        'confidence_score': verdict.confidence_score,\n"
            "        'strengths': verdict.strengths,\n"
            "        'weaknesses': verdict.weaknesses,\n"
            "        'recommendations': verdict.recommendations,\n"
            "        'segment_verdicts': verdict.segment_verdicts,\n"
            "    }\n"
            "    save_json(verdict_dict, os.path.join(diag_dir, 'model_verdict.json'))\n\n"
            "    # 8. Save diagnostics summary\n"
            "    summary = {\n"
            "        'overall_metrics': overall.to_dict(),\n"
            "        'error_decomposition': {\n"
            "            'bias_pct': decomp.bias_pct,\n"
            "            'variance_pct': decomp.variance_pct,\n"
            "            'systematic_error': decomp.systematic_error,\n"
            "            'random_error': decomp.random_error,\n"
            "        },\n"
            "        'n_keys': len(key_metrics),\n"
            "        'n_segments': len(seg_metrics) if len(seg_metrics) > 0 else 0,\n"
            "        'failure_analysis': failure_analysis,\n"
            "    }\n"
            "    save_json(summary, os.path.join(diag_dir, 'diagnostics_summary.json'))\n\n"
            "    # 9. Create visualizations\n"
            "    try:\n"
            "        plot_actual_vs_predicted(\n"
            "            forecasts[actual_col].values,\n"
            "            forecasts[pred_col].values,\n"
            "            os.path.join(diag_dir, 'actual_vs_predicted.png')\n"
            "        )\n"
            "        plot_error_distribution(\n"
            "            forecasts[actual_col].values,\n"
            "            forecasts[pred_col].values,\n"
            "            os.path.join(diag_dir, 'error_distribution.png')\n"
            "        )\n"
            "        if len(seg_metrics) > 0:\n"
            "            seg_col = 'segment_id' if 'segment_id' in seg_metrics.columns else ('model_group' if 'model_group' in seg_metrics.columns else 'model_level')\n"
            "            if seg_col in seg_metrics.columns:\n"
            "                plot_segment_comparison(\n"
            "                    seg_metrics, seg_col,\n"
            "                    os.path.join(diag_dir, 'segment_comparison.png')\n"
            "                )\n"
            "    except Exception:\n"
            "        pass  # Visualization failed but not critical\n\n"
            "    # Print summary (handle empty DataFrames)\n"
            "    top_wape = top_df['wape'].mean() if len(top_df) > 0 else 0.0\n"
            "    bottom_wape = bottom_df['wape'].mean() if len(bottom_df) > 0 else 0.0\n"
            "    printer.print(f'Top 10 avg WAPE: {top_wape:.2%}, Bottom 10 avg: {bottom_wape:.2%}')\n"
            "    printer.print(f'Deployment ready: {verdict.deployment_ready}')\n"
            "    printer.print('Saved: key_diagnostics.csv, segment_diagnostics.csv, model_verdict.json')\n"
            "```\n\n"
            "This orchestrates individual diagnostic utilities based on the strategy."
        ),
        agent=executor,
        expected_output=(
            "Diagnostics complete. Overall WAPE: X.XX%. Top 10 and Bottom 10 performers identified. "
            "Files created: key_diagnostics.csv, segment_diagnostics.csv, model_verdict.json"
        ),
        output_file=os.path.join(diag_dir_out, "diagnostic_execution_report.md"),
        context=[task_plan],
    )

    # -------------------------------------------------------------------------
    # Task 3: Diagnostic Analysis - Create INTELLIGENT feedback context
    # -------------------------------------------------------------------------
    task_analyze = Task(
        name="create_feedback_context",
        description=(
            "# EXECUTE THIS CODE USING CodeExecutionTool\n\n"
            "**CRITICAL: You MUST use CodeExecutionTool to execute the Python code below.**\n"
            "**DO NOT describe what the code does. DO NOT write your own version.**\n"
            "**JUST COPY THE CODE BLOCK INTO CodeExecutionTool AND RUN IT.**\n\n"
            "The code creates diagnostic_feedback_context.json with analysis results.\n\n"
            "## PYTHON CODE TO EXECUTE:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from utils.agent_utilities import load_json, save_json, load_csv\n\n"
            f"diag_dir = '{diag_dir}'\n"
            f"model_dir = '{model_dir}'\n\n"
            "# Helper function for safe JSON loading\n"
            "def safe_load_json(path, default=None):\n"
            "    if default is None:\n"
            "        default = {}\n"
            "    return load_json(path) if os.path.exists(path) else default\n\n"
            "# Read diagnostic outputs (with fallbacks if files don't exist)\n"
            "summary = safe_load_json(os.path.join(diag_dir, 'diagnostics_summary.json'), {'overall_metrics': {}})\n"
            "verdict = safe_load_json(os.path.join(diag_dir, 'model_verdict.json'), {'deployment_ready': False, 'confidence_score': 0, 'strengths': [], 'weaknesses': [], 'recommendations': [], 'segment_verdicts': {}})\n"
            "top_bottom = safe_load_json(os.path.join(diag_dir, 'top_bottom_analysis.json'), {'analysis': {}, 'top_performers': [], 'bottom_performers': []})\n\n"
            "# Read strategy for context\n"
            "strategy = safe_load_json(os.path.join(diag_dir, 'diagnostic_strategy.json'), {'focus_segments': [], 'per_model_perf_summary': {}})\n\n"
            "# Read model specs for model type analysis\n"
            "model_specs = safe_load_json(os.path.join(model_dir, 'final_model_specs.json'), {})\n"
            "specs_list = model_specs.get('model_specs', []) if isinstance(model_specs, dict) else []\n\n"
            "# STATE-OF-THE-ART: Read calibration and hyperparameter info\n"
            "segment_calibrations = load_json(os.path.join(model_dir, 'segment_calibrations.json')) if os.path.exists(os.path.join(model_dir, 'segment_calibrations.json')) else {}\n"
            "bias_calibration = load_json(os.path.join(model_dir, 'bias_calibration.json')) if os.path.exists(os.path.join(model_dir, 'bias_calibration.json')) else {}\n\n"
            "# Extract state-of-the-art features status\n"
            "sota_features = model_specs.get('state_of_art_features', {})\n"
            "segment_hyperparam_profiles = model_specs.get('segment_hyperparam_profiles', {})\n\n"
            "# Extract overall metrics\n"
            "overall = summary.get('overall_metrics', {})\n"
            "wape = overall.get('wape', 1.0)\n"
            "quality_tier = overall.get('quality_tier', 'POOR')\n\n"
            "# Analyze root causes\n"
            "root_causes = {'model_mismatch': [], 'feature_gaps': [], 'data_quality': []}\n"
            "per_model_perf = strategy.get('per_model_perf_summary', {})\n\n"
            "# Check for model mismatches - FEATURE-BASED MODELS ONLY\n"
            "# All patterns should use feature-based models (LightGBM, XGBoost, CatBoost, etc.)\n"
            "# Univariate models (TSB, Croston, SBA, ARIMA, Prophet) are BANNED\n"
            "pattern_to_recommended = {\n"
            "    'lumpy': {'lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'tweedie'},\n"
            "    'intermittent': {'lightgbm', 'xgboost', 'zero_inflated', 'hurdle_model', 'catboost'},\n"
            "    'erratic': {'xgboost', 'lightgbm', 'catboost', 'random_forest', 'tweedie'},\n"
            "    'smooth': {'lightgbm', 'xgboost', 'catboost', 'random_forest', 'tweedie'},\n"
            "}\n\n"
            "action_items = []\n"
            "for mg, perf in per_model_perf.items():\n"
            "    model_type = perf.get('model_type', 'unknown')\n"
            "    demand_pattern = perf.get('demand_pattern', 'smooth')\n"
            "    mg_wape = perf.get('wape', 0)\n"
            "    \n"
            "    recommended = pattern_to_recommended.get(demand_pattern, set())\n"
            "    if model_type not in recommended and model_type != 'lightgbm' and mg_wape and mg_wape > 0.3:\n"
            "        root_causes['model_mismatch'].append(mg)\n"
            "        rec_model = list(recommended)[0] if recommended else 'lightgbm'\n"
            "        action_items.append(f'Retrain {mg} with {rec_model} (current: {model_type}, pattern: {demand_pattern})')\n\n"
            "# Check for high WAPE without model mismatch (likely feature gaps)\n"
            "for mg, perf in per_model_perf.items():\n"
            "    mg_wape = perf.get('wape', 0)\n"
            "    if mg_wape and mg_wape > 0.4 and mg not in root_causes['model_mismatch']:\n"
            "        root_causes['feature_gaps'].append(mg)\n"
            "        action_items.append(f'Improve feature engineering for {mg} (WAPE: {mg_wape:.1%})')\n\n"
            "# Build feedback context\n"
            "feedback = {\n"
            "    'overall_assessment': {\n"
            "        'wape': wape,\n"
            "        'quality_tier': quality_tier,\n"
            "        'deployment_ready': verdict.get('deployment_ready', False),\n"
            "        'confidence_score': verdict.get('confidence_score', 0),\n"
            "    },\n"
            "    'model_verdict': {\n"
            "        'strengths': verdict.get('strengths', []),\n"
            "        'weaknesses': verdict.get('weaknesses', []),\n"
            "        'recommendations': verdict.get('recommendations', []),\n"
            "    },\n"
            "    'underperformer_root_causes': root_causes,\n"
            "    'top_bottom_summary': {\n"
            "        'top_avg_wape': top_bottom.get('analysis', {}).get('top_avg_wape'),\n"
            "        'bottom_avg_wape': top_bottom.get('analysis', {}).get('bottom_avg_wape'),\n"
            "    },\n"
            "    'error_decomposition': summary.get('error_decomposition', {}),\n"
            "    'action_items_for_pipeline': action_items[:10],  # Top 10 action items\n"
            "    'segment_verdicts': verdict.get('segment_verdicts', {}),\n"
            "    # STATE-OF-THE-ART: Track advanced features utilization\n"
            "    'state_of_art_features': {\n"
            "        'segment_specific_hyperparams_used': sota_features.get('segment_specific_hyperparams', False),\n"
            "        'pattern_aware_ensemble_used': sota_features.get('pattern_aware_ensemble', False),\n"
            "        'bias_calibration_enabled': bias_calibration.get('enabled', False),\n"
            "        'n_segments_calibrated': len(segment_calibrations),\n"
            "        'threshold_calibration_applied': sota_features.get('adaptive_threshold_calibration', False),\n"
            "        'segment_aware_bias_calibration': sota_features.get('segment_aware_bias_calibration', False),\n"
            "    },\n"
            "    # STATE-OF-THE-ART: Per-segment calibration factors for inspection\n"
            "    'segment_calibration_summary': {\n"
            "        seg_id: {\n"
            "            'global_factor': cal.get('global_factor', 1.0),\n"
            "            'demand_pattern': cal.get('demand_pattern', 'unknown'),\n"
            "        }\n"
            "        for seg_id, cal in list(segment_calibrations.items())[:5]  # Top 5 for brevity\n"
            "    } if segment_calibrations else {},\n"
            "}\n\n"
            "# Save feedback context\n"
            "save_json(feedback, os.path.join(diag_dir, 'diagnostic_feedback_context.json'))\n\n"
            "print(f'Analyzed: WAPE={wape:.2%}, Quality={quality_tier}')\n"
            "print(f'Root causes: {len(root_causes[\"model_mismatch\"])} model mismatches, {len(root_causes[\"feature_gaps\"])} feature gaps')\n"
            "print(f'SoTA features: bias_cal={bias_calibration.get(\"enabled\", False)}, seg_cal={len(segment_calibrations)} segments')\n"
            "print(f'Action items: {len(action_items)}')\n"
            "print('Created: diagnostic_feedback_context.json')\n"
            "```"
        ),
        agent=analyst,
        expected_output=(
            "Created diagnostic_feedback_context.json with overall assessment, "
            "model verdict, underperformer root causes, and prioritized action items for future improvements."
        ),
        output_file=os.path.join(diag_dir_out, "diagnostic_analysis_report.md"),
        context=[task_plan, task_execute],
    )

    # -------------------------------------------------------------------------
    # Task 4: Documentation - Generate comprehensive insights guide
    # -------------------------------------------------------------------------
    documentation_agent = _create_diagnostic_documentation_agent(llm, protected_paths=protected_dirs)
    task_document = Task(
        name="generate_diagnostic_documentation",
        description=(
            "# DIAGNOSTICS INSIGHTS REPORT GENERATION\n\n"
            f"Analyze diagnostic outputs in `{diag_dir}` and create a comprehensive insights report.\n\n"
            "## AVAILABLE FILES\n\n"
            f"- `{diag_dir}/diagnostics_summary.json` - Overall metrics and error decomposition\n"
            f"- `{diag_dir}/model_verdict.json` - Deployment readiness and confidence score\n"
            f"- `{diag_dir}/top_bottom_analysis.json` - Top 10 / Bottom 10 performers\n"
            f"- `{diag_dir}/diagnostic_feedback_context.json` - Root causes and action items\n"
            f"- `{diag_dir}/segment_diagnostics.csv` - Per-segment WAPE metrics\n"
            f"- `{diag_dir}/key_diagnostics.csv` - Per-key WAPE metrics\n\n"
            "## PHASE 1: EXPLORE DIAGNOSTIC OUTPUTS\n\n"
            "Execute this code to understand the diagnostic results:\n\n"
            "```python\n"
            "import json\n"
            "import os\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n\n"
            f"diag_dir = '{diag_dir}'\n\n"
            "# List files\n"
            "files = [f for f in os.listdir(diag_dir) if f.endswith(('.json', '.csv', '.md'))]\n"
            "print(f'Available diagnostic files: {{len(files)}}')\n"
            "for f in sorted(files)[:15]: print(f'  - {{f}}')\n\n"
            "# Load summary\n"
            "with open(os.path.join(diag_dir, 'diagnostics_summary.json')) as f:\n"
            "    summary = json.load(f)\n\n"
            "overall = summary.get('overall_metrics', {{}})\n"
            "overall_wape = overall.get('wape', 0)\n"
            "quality_tier = overall.get('quality_tier', 'UNKNOWN')\n"
            "print(f'\\nOverall WAPE: {{overall_wape:.4f}} ({{overall_wape*100:.1f}}%)')\n"
            "print(f'Quality Tier: {{quality_tier}}')\n"
            "```\n\n"
            "## PHASE 2: ANALYZE MODEL VERDICT\n\n"
            "Execute this code to analyze the model verdict:\n\n"
            "```python\n"
            "import json\n"
            "import os\n\n"
            f"diag_dir = '{diag_dir}'\n\n"
            "# Load model verdict\n"
            "with open(os.path.join(diag_dir, 'model_verdict.json')) as f:\n"
            "    verdict = json.load(f)\n\n"
            "print('=== MODEL VERDICT ===')\n"
            "print(f'Deployment Ready: {{verdict.get(\"deployment_ready\", False)}}')\n"
            "print(f'Confidence Score: {{verdict.get(\"confidence_score\", 0):.1f}}')\n\n"
            "print('\\nStrengths:')\n"
            "for s in verdict.get('strengths', [])[:5]:\n"
            "    print(f'  ✓ {{s}}')\n\n"
            "print('\\nWeaknesses:')\n"
            "for w in verdict.get('weaknesses', [])[:5]:\n"
            "    print(f'  ⚠ {{w}}')\n\n"
            "print('\\nRecommendations:')\n"
            "for r in verdict.get('recommendations', [])[:3]:\n"
            "    print(f'  → {{r}}')\n"
            "```\n\n"
            "## PHASE 3: ANALYZE TOP/BOTTOM PERFORMERS\n\n"
            "Execute this code to analyze performance distribution:\n\n"
            "```python\n"
            "import json\n"
            "import os\n\n"
            f"diag_dir = '{diag_dir}'\n\n"
            "# Load top/bottom analysis\n"
            "with open(os.path.join(diag_dir, 'top_bottom_analysis.json')) as f:\n"
            "    tb = json.load(f)\n\n"
            "analysis = tb.get('analysis', {{}})\n"
            "top_avg = analysis.get('top_avg_wape', 0)\n"
            "bottom_avg = analysis.get('bottom_avg_wape', 0)\n\n"
            "print('=== TOP/BOTTOM PERFORMER ANALYSIS ===')\n"
            "print(f'Top 10 Average WAPE: {{top_avg:.1%}}')\n"
            "print(f'Bottom 10 Average WAPE: {{bottom_avg:.1%}}')\n"
            "print(f'Performance Gap: {{(bottom_avg - top_avg):.1%}}')\n\n"
            "print('\\nTop 3 Performers:')\n"
            "for p in tb.get('top_performers', [])[:3]:\n"
            "    print(f\"  {{p.get('key', '?')}}: {{p.get('wape', 0):.1%}}\")\n\n"
            "print('\\nBottom 3 Performers:')\n"
            "for p in tb.get('bottom_performers', [])[:3]:\n"
            "    print(f\"  {{p.get('key', '?')}}: {{p.get('wape', 0):.1%}}\")\n"
            "```\n\n"
            "## PHASE 4: ANALYZE ROOT CAUSES AND SEGMENT PERFORMANCE\n\n"
            "Execute this code to analyze root causes:\n\n"
            "```python\n"
            "import json\n"
            "import os\n"
            "import pandas as pd\n\n"
            f"diag_dir = '{diag_dir}'\n\n"
            "# Load feedback context\n"
            "with open(os.path.join(diag_dir, 'diagnostic_feedback_context.json')) as f:\n"
            "    feedback = json.load(f)\n\n"
            "root_causes = feedback.get('underperformer_root_causes', {{}})\n"
            "action_items = feedback.get('action_items_for_pipeline', [])\n\n"
            "print('=== ROOT CAUSE ANALYSIS ===')\n"
            "print(f'Model Mismatches: {{len(root_causes.get(\"model_mismatch\", []))}}')\n"
            "print(f'Feature Gaps: {{len(root_causes.get(\"feature_gaps\", []))}}')\n"
            "print(f'Data Quality Issues: {{len(root_causes.get(\"data_quality\", []))}}')\n\n"
            "print(f'\\nAction Items: {{len(action_items)}}')\n"
            "for i, item in enumerate(action_items[:5], 1):\n"
            "    print(f'  {{i}}. {{item[:80]}}...' if len(item) > 80 else f'  {{i}}. {{item}}')\n\n"
            "# Load segment diagnostics\n"
            "seg_path = os.path.join(diag_dir, 'segment_diagnostics.csv')\n"
            "if os.path.exists(seg_path):\n"
            "    seg_df = pd.read_csv(seg_path)\n"
            "    print(f'\\nSegment Diagnostics: {{len(seg_df)}} segments')\n"
            "    if 'wape' in seg_df.columns:\n"
            "        good = (seg_df['wape'] < 0.35).sum()\n"
            "        fair = ((seg_df['wape'] >= 0.35) & (seg_df['wape'] < 0.5)).sum()\n"
            "        poor = (seg_df['wape'] >= 0.5).sum()\n"
            "        print(f'  Good (WAPE < 35%): {{good}}')\n"
            "        print(f'  Fair (35-50%): {{fair}}')\n"
            "        print(f'  Poor (>50%): {{poor}}')\n"
            "```\n\n"
            "## PHASE 5: GENERATE AND SAVE REPORT\n\n"
            "After gathering all insights, execute code to CREATE and SAVE the report.\n"
            "YOU MUST use the ACTUAL NUMBERS from your previous code executions.\n\n"
            "```python\n"
            "from datetime import datetime\n"
            "import json\n"
            "import os\n"
            "import pandas as pd\n\n"
            f"diag_dir = '{diag_dir}'\n\n"
            "# Re-load all data to build comprehensive report\n"
            "with open(os.path.join(diag_dir, 'diagnostics_summary.json')) as f:\n"
            "    summary = json.load(f)\n"
            "with open(os.path.join(diag_dir, 'model_verdict.json')) as f:\n"
            "    verdict = json.load(f)\n"
            "with open(os.path.join(diag_dir, 'top_bottom_analysis.json')) as f:\n"
            "    tb = json.load(f)\n"
            "with open(os.path.join(diag_dir, 'diagnostic_feedback_context.json')) as f:\n"
            "    feedback = json.load(f)\n\n"
            "# Extract statistics\n"
            "overall = summary.get('overall_metrics', {{}})\n"
            "overall_wape = overall.get('wape', 0)\n"
            "quality_tier = overall.get('quality_tier', 'UNKNOWN')\n"
            "deployment_ready = verdict.get('deployment_ready', False)\n"
            "confidence = verdict.get('confidence_score', 0)\n"
            "strengths = verdict.get('strengths', [])\n"
            "weaknesses = verdict.get('weaknesses', [])\n"
            "recommendations = verdict.get('recommendations', [])\n\n"
            "# Top/Bottom\n"
            "analysis = tb.get('analysis', {{}})\n"
            "top_avg = analysis.get('top_avg_wape', 0)\n"
            "bottom_avg = analysis.get('bottom_avg_wape', 0)\n"
            "top_performers = tb.get('top_performers', [])\n"
            "bottom_performers = tb.get('bottom_performers', [])\n\n"
            "# Root causes\n"
            "root_causes = feedback.get('underperformer_root_causes', {{}})\n"
            "action_items = feedback.get('action_items_for_pipeline', [])\n"
            "model_mismatch = root_causes.get('model_mismatch', [])\n"
            "feature_gaps = root_causes.get('feature_gaps', [])\n\n"
            "# Load segment diagnostics\n"
            "seg_path = os.path.join(diag_dir, 'segment_diagnostics.csv')\n"
            "seg_df = pd.read_csv(seg_path) if os.path.exists(seg_path) else pd.DataFrame()\n\n"
            "# Build comprehensive markdown report\n"
            "quality_label = '✓ Excellent' if overall_wape < 0.2 else '✓ Good' if overall_wape < 0.35 else '⚠ Fair' if overall_wape < 0.5 else '✗ Poor'\n\n"
            "md = f'''# Diagnostics Insights Guide\n"
            "**Generated:** {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\n\n"
            "---\n\n"
            "## Executive Summary\n\n"
            "This report provides comprehensive diagnostics analysis of the forecasting model.\n"
            "The model achieved an **overall WAPE of {{overall_wape:.1%}}** ({{quality_tier}}).\n\n"
            "**Deployment Readiness:** {{'✓ Yes' if deployment_ready else '✗ No'}} (Confidence: {{confidence:.0f}}/100)\n\n"
            "---\n\n"
            "## 1. Overall Performance\n\n"
            "| Metric | Value | Assessment |\n"
            "|--------|-------|------------|\n"
            "| Overall WAPE | {{overall_wape:.1%}} | {{quality_label}} |\n"
            "| Quality Tier | {{quality_tier}} | Model classification |\n"
            "| Deployment Ready | {{'✓ Yes' if deployment_ready else '✗ No'}} | Production readiness |\n"
            "| Confidence Score | {{confidence:.0f}}/100 | Assessment confidence |\n\n"
            "---\n\n"
            "## 2. Model Verdict\n\n"
            "### Strengths\n\n'''\n\n"
            "for s in strengths[:5]:\n"
            "    md += f'- ✓ {{s}}\\n'\n"
            "if not strengths:\n"
            "    md += '- No specific strengths identified\\n'\n\n"
            "md += f'''\\n### Weaknesses\\n\\n'''\n"
            "for w in weaknesses[:5]:\n"
            "    md += f'- ⚠ {{w}}\\n'\n"
            "if not weaknesses:\n"
            "    md += '- No significant weaknesses identified\\n'\n\n"
            "md += f'''\n\n---\n\n"
            "## 3. Top/Bottom Performer Analysis\n\n"
            "### Performance Gap\n\n"
            "| Metric | Top 10 | Bottom 10 | Gap |\n"
            "|--------|--------|-----------|-----|\n"
            "| Average WAPE | {{top_avg:.1%}} | {{bottom_avg:.1%}} | {{(bottom_avg - top_avg):.1%}} |\n\n"
            "### Top 5 Performers\n\n"
            "| Key | WAPE | MAE |\n"
            "|-----|------|-----|\n'''\n\n"
            "for p in top_performers[:5]:\n"
            "    md += f\"| {{p.get('key', '?')}} | {{p.get('wape', 0):.1%}} | {{p.get('mae', 0):.2f}} |\\n\"\n\n"
            "md += f'''\\n### Bottom 5 Performers\n\n"
            "| Key | WAPE | MAE |\n"
            "|-----|------|-----|\n'''\n\n"
            "for p in bottom_performers[:5]:\n"
            "    md += f\"| {{p.get('key', '?')}} | {{p.get('wape', 0):.1%}} | {{p.get('mae', 0):.2f}} |\\n\"\n\n"
            "md += f'''\n\n---\n\n"
            "## 4. Root Cause Analysis\n\n'''\n\n"
            "if model_mismatch:\n"
            "    md += f'### Model Mismatch Issues\\n\\n'\n"
            "    md += f'**Affected Segments ({len(model_mismatch)}):** {{model_mismatch[:5]}}\\n\\n'\n"
            "    md += 'These segments may be using inappropriate model types for their demand patterns.\\n\\n'\n"
            "if feature_gaps:\n"
            "    md += f'### Feature Engineering Gaps\\n\\n'\n"
            "    md += f'**Affected Segments ({len(feature_gaps)}):** {{feature_gaps[:5]}}\\n\\n'\n"
            "    md += 'These segments may benefit from additional or different features.\\n\\n'\n"
            "if not model_mismatch and not feature_gaps:\n"
            "    md += 'No significant root causes identified - model is performing well.\\n\\n'\n\n"
            "md += f'''\n---\n\n"
            "## 5. Segment-Level Performance\n\n'''\n\n"
            "if not seg_df.empty and 'wape' in seg_df.columns:\n"
            "    md += '| Segment | WAPE | Quality | Action Needed |\\n'\n"
            "    md += '|---------|------|---------|---------------|\\n'\n"
            "    for _, row in seg_df.head(10).iterrows():\n"
            "        seg = row.get('segment_id', row.get('model_group', row.get('model_level', 'unknown')))\n"
            "        wape = row.get('wape', 0)\n"
            "        quality = '✓ Good' if wape < 0.35 else '⚠ Fair' if wape < 0.5 else '✗ Poor'\n"
            "        action = 'None' if wape < 0.35 else 'Review features' if wape < 0.5 else 'Retrain with different model'\n"
            "        md += f'| {{seg}} | {{wape:.1%}} | {{quality}} | {{action}} |\\n'\n"
            "    md += '\\n'\n"
            "else:\n"
            "    md += 'Segment-level diagnostics not available.\\n\\n'\n\n"
            "md += f'''\n---\n\n"
            "## 6. Action Items & Recommendations\n\n"
            "### Prioritized Action Items\n\n'''\n\n"
            "for i, item in enumerate(action_items[:10], 1):\n"
            "    md += f'{{i}}. {{item}}\\n'\n"
            "if not action_items:\n"
            "    md += 'No critical action items - model is performing within acceptable thresholds.\\n'\n\n"
            "md += f'''\\n### General Recommendations\\n\\n'''\n"
            "for r in recommendations[:5]:\n"
            "    md += f'- 💡 {{r}}\\n'\n\n"
            "md += f'''\n\n---\n\n"
            "*Generated by Diagnostics Insights Agent using iterative code execution.*\\n"
            "'''\n\n"
            "# SAVE THE REPORT TO FILE\n"
            "report_path = os.path.join(diag_dir, 'DIAGNOSTICS_INSIGHTS_GUIDE.md')\n"
            "with open(report_path, 'w') as f:\n"
            "    f.write(md)\n\n"
            "print(f'Saved: DIAGNOSTICS_INSIGHTS_GUIDE.md ({{len(md):,}} bytes)')\n"
            "print(f'Overall WAPE: {{overall_wape:.1%}} ({{quality_tier}})')\n"
            "print(f'Deployment Ready: {{deployment_ready}}, Confidence: {{confidence:.0f}}')\n"
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
            "Created DIAGNOSTICS_INSIGHTS_GUIDE.md - comprehensive documentation with "
            "performance metrics, root causes, segment analysis, and action items."
        ),
        output_file=os.path.join(diag_dir_out, "diagnostic_documentation_report.md"),
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
        logger.info("SKIPPING diagnostic insights report (enable_insights_reports=False)")

    if enable_reviewer:
        reviewer = _create_diagnostic_reviewer_agent(llm, protected_paths=protected_dirs)
        task_review = Task(
            name="review_diagnostic_outputs",
            description=(
                "# DIAGNOSTIC OUTPUT QUALITY REVIEW\n\n"
                "You are the Diagnostic Reviewer. Validate all outputs from the Diagnostic crew.\n\n"
                "## VALIDATION STEPS\n"
                "Write and execute code to validate:\n"
                "```python\n"
                "import os\n"
                "import json\n"
                f"diag_dir = '{diag_dir}'\n"
                "\n"
                "# Check required files exist\n"
                "required_files = [\n"
                "    'key_diagnostics.csv',\n"
                "    'segment_diagnostics.csv',\n"
                "    'model_verdict.json',\n"
                "    'diagnostic_feedback_context.json',\n"
                "    'diagnostic_strategy.json',\n"
                "]\n"
                "missing = [f for f in required_files if not os.path.exists(os.path.join(diag_dir, f))]\n"
                "print(f'Missing files: {missing}' if missing else 'All required files present')\n"
                "\n"
                "# Validate WAPE metrics are reasonable\n"
                "with open(os.path.join(diag_dir, 'model_verdict.json')) as f:\n"
                "    verdict = json.load(f)\n"
                "overall_wape = verdict.get('overall_wape', 0)\n"
                "if overall_wape > 1.0:\n"
                "    print(f'WARNING: Overall WAPE > 100%: {overall_wape:.1%}')\n"
                "else:\n"
                "    print(f'Overall WAPE: {overall_wape:.1%}')\n"
                "```\n\n"
                "## OUTPUT\n"
                f"Create `{diag_dir}/diagnostic_review_report.json` with:\n"
                "- quality_score (1-10)\n"
                "- files_validated (list)\n"
                "- wape_validation (pass/fail)\n"
                "- verdict_justified (pass/fail)\n"
                "- issues_found (list)\n"
                "- recommendations (list)\n"
            ),
            agent=reviewer,
            expected_output=(
                "Created diagnostic_review_report.json with quality score and validation results. "
                "All diagnostic files validated. Issues identified and documented."
            ),
            output_file=os.path.join(diag_dir_out, "diagnostic_review_summary.md"),
            context=[task_analyze],
        )
        agents.append(reviewer)
        tasks.append(task_review)
        crew_name = "Diagnostic Crew (4-Agent Pattern with Reviewer)"
    else:
        crew_name = "Diagnostic Crew (3-Agent Pattern)"

    return Crew(
        name=crew_name,
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def run_diagnostic_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> DiagnosticCrewResult:
    """Run the diagnostic crew and return results."""
    from utils.cost_tracking import get_cost_tracker, extract_tokens_from_crew_result

    # ==========================================================================
    # CRITICAL: Validate input data file exists BEFORE running any crew tasks
    # ==========================================================================
    if not os.path.exists(config.input_data_path):
        raise DiagnosticFailedError(
            f"Input data file not found: {config.input_data_path}\n"
            "Please ensure the data file exists at the configured path."
        )

    # Start cost tracking
    tracker = get_cost_tracker()
    tracker.start_crew("Diagnostic Crew")

    # Get model ID from LLM if available
    model_id = getattr(llm, "model", "default")
    tracker.set_model(model_id)

    # ======================================================================
    # WRITE PROTECTION: Set protected_paths on the LLM's CodeExecutionTool
    # ======================================================================
    eda_dir = os.path.join(config.artifact_base_path, "eda_output")
    seg_dir = os.path.join(config.artifact_base_path, "seg_output")
    feat_dir = os.path.join(config.artifact_base_path, "feature_output")
    model_dir = os.path.join(config.artifact_base_path, "model_artifacts")
    _code_tool = getattr(llm, '_code_execution_tool', None)
    if _code_tool is not None:
        _code_tool.protected_paths = [eda_dir, seg_dir, feat_dir, model_dir]
        logger.info(f"Write protection set on LLM CodeExecutionTool: upstream dirs are READ-ONLY")

    crew = create_diagnostic_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)
    crew_result = crew.kickoff()

    diag_dir = os.path.join(config.artifact_base_path, "diagnostics_output")

    # Extract and record tokens from crew result
    tokens = extract_tokens_from_crew_result(crew_result)
    if tokens["total"] > 0:
        tracker.record_llm_call(
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            model=model_id,
        )

    # End tracking and save cost report
    cost_report = tracker.end_crew("Diagnostic Crew", diag_dir)
    cost_report_path = os.path.join(diag_dir, "diagnostic_cost.json")

    # Build result object with paths to all outputs
    return DiagnosticCrewResult(
        group_diagnostics_path=os.path.join(diag_dir, "group_diagnostics.csv"),
        segment_diagnostics_path=os.path.join(diag_dir, "segment_diagnostics.csv"),
        key_diagnostics_path=os.path.join(diag_dir, "key_diagnostics.csv"),
        diagnostics_summary_path=os.path.join(diag_dir, "diagnostics_summary.json"),
        diagnostics_report_markdown_path=os.path.join(diag_dir, "diagnostic_analysis_report.md"),
        diagnostics_dir=diag_dir,
        top_bottom_analysis_path=os.path.join(diag_dir, "top_bottom_analysis.json"),
        model_verdict_path=os.path.join(diag_dir, "model_verdict.json"),
        diagnostic_pipeline_script_path=os.path.join(diag_dir, "diagnostic_planning_report.md"),
        diagnostic_feedback_context_path=os.path.join(diag_dir, "diagnostic_feedback_context.json"),
        cost_report_path=cost_report_path,
    )
