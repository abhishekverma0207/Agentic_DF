# crews/segmentation_crew.py
"""
World-Class Segmentation Crew for Demand Forecasting.

This crew uses the 3-Agent Pattern with INTELLIGENT UTILITY ORCHESTRATION:
1. Segmentation Planner: Reads EDA context and creates an intelligent clustering strategy
   based on EDA recommendations (algorithm, features, cluster count).
2. Segmentation Executor: Intelligently calls individual utility functions based on EDA guidance
   - Creates MULTI-DIMENSIONAL features (volume, pattern, variability)
   - Uses recommended algorithm (GaussianMixture, KMeans, HDBSCAN, etc.)
   - Uses recommended features from EDA context
   - Enforces minimum segment size (10%)
3. Segmentation Analyst: Reads raw segmentation outputs and creates INTELLIGENT context files
   for downstream Feature and Training crews.

KEY DESIGN PRINCIPLES:
======================
1. INTELLIGENT ORCHESTRATION: Agents decide which utilities to call based on data characteristics
2. NO MONOLITHIC PIPELINES: Uses individual utility functions, not run_segmentation_pipeline()
3. EDA-DRIVEN DECISIONS: Reads EDA context and adapts strategy accordingly
4. MULTI-DIMENSIONAL SEGMENTATION: Volume, pattern, variability - not just ADI/CV2

AVAILABLE UTILITIES (from utils/segmentation.py):
================================================
- create_segmentation_features(): Create comprehensive features (volume, pattern, variability)
- classify_demand_pattern(): Syntetos-Boylan classification
- get_recommended_clustering_features(): Get features based on strategy
- scale_features(): Robust scaling for clustering
- cluster_kmeans(), cluster_gaussian_mixture(), cluster_hdbscan(), cluster_hierarchical_optimal()
- enforce_minimum_segment_size(): Merge small segments
- compute_segment_profiles(): Create segment profiles with relative indices
- label_segments_automatically(): Generate human-readable labels
- assign_model_recommendations(): Model recommendations per segment
- create_feature_recommendations(): Feature engineering recommendations per segment
- plot_*(): Visualization functions
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from crewai import Agent, Crew, Task, Process, LLM

from config.schema import DemandForecastConfig
from utils.code_execution_tool import CodeExecutionTool

logger = logging.getLogger(__name__)


class SegmentationFailedError(Exception):
    """Raised when segmentation fails to produce required outputs."""
    pass


def _validate_planner_output(seg_dir: str) -> None:
    """
    Validate that Planner task created clustering_strategy.json.
    Called as callback after Planner completes.
    """
    strategy_path = os.path.join(seg_dir, 'clustering_strategy.json')
    if not os.path.exists(strategy_path):
        raise SegmentationFailedError(
            f"CRITICAL: Segmentation Planner FAILED to create clustering_strategy.json\n\n"
            f"Expected file: {strategy_path}\n\n"
            "ROOT CAUSE: The LLM agent did NOT execute the code block.\n"
            "The agent likely described what should happen instead of running CodeExecutionTool.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )
    logger.info(f"Planner validation passed: {strategy_path} exists")


def _validate_executor_output(seg_dir: str) -> None:
    """
    Validate that Executor task created all required segmentation files WITH CORRECT FORMAT.
    Called as callback after Executor completes.

    This validates that the LLM actually ran the provided run_segmentation_pipeline code
    and didn't invent its own simplified version.
    """
    import json
    import pandas as pd

    required_files = [
        ('per_key_with_segments.csv', 'Segment assignments for all series'),
        ('segment_profiles.json', 'Segment profiles with statistics'),
        ('model_recommendations.json', 'Model recommendations per segment'),
        ('feature_recommendations.json', 'Feature recommendations per segment'),
        ('clustering_metrics.json', 'Clustering quality metrics'),
    ]

    missing = []
    for filename, description in required_files:
        filepath = os.path.join(seg_dir, filename)
        if not os.path.exists(filepath):
            missing.append(f"  - {filename}: {description}")

    if missing:
        raise SegmentationFailedError(
            f"CRITICAL: Segmentation Executor FAILED to create required files\n\n"
            f"Missing files in {seg_dir}:\n" +
            "\n".join(missing) + "\n\n"
            "ROOT CAUSE: The LLM agent did NOT execute the code block.\n"
            "The agent likely described what should happen instead of running CodeExecutionTool.\n\n"
            "SOLUTION: Check segmentation_execution_report.md for the task output.\n"
            "If it contains description instead of execution output, this confirms the issue.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )

    # Validate per_key_with_segments.csv has required columns (not just 4 columns!)
    per_key_path = os.path.join(seg_dir, 'per_key_with_segments.csv')
    try:
        df = pd.read_csv(per_key_path)
        # Note: demand_pattern OR intermittency_class are acceptable (same semantic meaning)
        required_cols = ['segment_id', 'volume_tier']
        pattern_cols = ['demand_pattern', 'intermittency_class']  # Either is acceptable

        missing_cols = [c for c in required_cols if c not in df.columns]
        has_pattern_col = any(c in df.columns for c in pattern_cols)

        if not has_pattern_col:
            missing_cols.append('demand_pattern (or intermittency_class)')

        if missing_cols:
            raise SegmentationFailedError(
                f"CRITICAL: per_key_with_segments.csv is INCOMPLETE!\n\n"
                f"Missing columns: {missing_cols}\n"
                f"Actual columns ({len(df.columns)}): {list(df.columns)}\n\n"
                "ROOT CAUSE: The LLM agent wrote its own simplified segmentation code\n"
                "instead of using the provided run_segmentation_pipeline() function.\n\n"
                "The utility function produces 30+ columns including demand_pattern,\n"
                "volume_tier, adi, cv2, and many segmentation features.\n\n"
                "SOLUTION: Re-run and ensure the agent executes the EXACT code block provided."
            )

        if len(df.columns) < 10:
            logger.warning(
                f"per_key_with_segments.csv has only {len(df.columns)} columns. "
                f"Expected 30+. Agent may have used simplified code."
            )
    except Exception as e:
        if 'INCOMPLETE' in str(e):
            raise
        logger.warning(f"Could not validate per_key_with_segments.csv columns: {e}")

    # Validate segment_profiles.json has correct structure
    profiles_path = os.path.join(seg_dir, 'segment_profiles.json')
    try:
        with open(profiles_path) as f:
            profiles = json.load(f)

        # Check that keys are numeric strings ("0", "1", "2"), not "S1", "S2", "S3"
        if profiles:
            first_key = list(profiles.keys())[0]
            if first_key.startswith('S') and first_key[1:].isdigit():
                raise SegmentationFailedError(
                    f"CRITICAL: segment_profiles.json has WRONG KEY FORMAT!\n\n"
                    f"Found keys like '{first_key}' - Expected keys like '0', '1', '2'\n\n"
                    "ROOT CAUSE: The LLM agent wrote its own segmentation code\n"
                    "instead of using the provided run_segmentation_pipeline() function.\n\n"
                    "SOLUTION: Re-run and ensure the agent executes the EXACT code block provided."
                )

            # Check for required profile fields
            sample_profile = profiles[first_key]
            expected_fields = ['dominant_pattern', 'size', 'pct_of_total']
            missing_fields = [f for f in expected_fields if f not in sample_profile]
            if missing_fields:
                logger.warning(
                    f"segment_profiles.json missing expected fields: {missing_fields}. "
                    f"Actual fields: {list(sample_profile.keys())}"
                )
    except Exception as e:
        if 'WRONG KEY FORMAT' in str(e):
            raise
        logger.warning(f"Could not validate segment_profiles.json structure: {e}")

    logger.info(f"Executor validation passed: all required files exist in {seg_dir}")


def _validate_analyst_output(seg_dir: str) -> None:
    """
    Validate that Analyst task created the context files for downstream crews.
    Called as callback after Analyst completes.
    """
    required_files = [
        ('segmentation_to_feature_context.json', 'Context for Feature Engineering crew'),
        ('segmentation_to_training_context.json', 'Context for Training crew'),
    ]

    missing = []
    for filename, description in required_files:
        filepath = os.path.join(seg_dir, filename)
        if not os.path.exists(filepath):
            missing.append(f"  - {filename}: {description}")

    if missing:
        raise SegmentationFailedError(
            f"CRITICAL: Segmentation Analyst FAILED to create context files\n\n"
            f"Missing files in {seg_dir}:\n" +
            "\n".join(missing) + "\n\n"
            "ROOT CAUSE: The LLM agent did NOT execute the code block.\n"
            "The agent likely described what should happen instead of running CodeExecutionTool.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )
    logger.info(f"Analyst validation passed: all context files exist in {seg_dir}")


@dataclass
class SegmentationCrewResult:
    seg_dir: str
    per_key_with_segments_path: str
    segmentation_metadata_path: str
    segmentation_preview_path: str
    # Advanced segmentation outputs
    modeling_strategy_path: str
    feature_recommendations_path: str
    # Charts
    cluster_sizes_chart_path: str
    segment_sizes_chart_path: str
    intermittency_mix_chart_path: str
    segmentation_report_markdown_path: str
    # Focused context outputs
    segmentation_to_feature_context_path: str
    segmentation_to_training_context_path: str
    # Generated pipeline script
    segmentation_pipeline_script_path: str = ""
    # Saved clustering models for production pipelines
    cluster_model_path: str = ""
    cluster_scaler_path: str = ""
    # DETERMINISTIC CODE OUTPUT for Pipeline Generator
    segmentation_deterministic_code_path: str = ""
    # Cost tracking
    cost_report_path: str = ""


def _get_output_path(absolute_path: str) -> str:
    """
    Get a safe path for CrewAI Task output_file parameter.

    CrewAI 1.9.1+ rejects paths with '..' (path traversal) for security.
    On Databricks, relative paths from cwd to /Volumes/ create '../..' paths.

    Solution: Use absolute paths which are safe and always work.
    """
    return os.path.abspath(absolute_path)


def _create_segmentation_planner_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Segmentation Planner agent - reads EDA context and creates intelligent strategy.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="segmentation_planner",
        role="Intelligent Segmentation Strategist",
        goal=(
            "EXECUTE the Python code provided in the task using CodeExecutionTool to create "
            "clustering_strategy.json. You MUST actually run the code, not describe it."
        ),
        backstory=(
            "######################################################################\n"
            "#  CRITICAL: YOU MUST EXECUTE CODE USING CodeExecutionTool          #\n"
            "#  DO NOT just describe what should happen - ACTUALLY RUN THE CODE! #\n"
            "#  DO NOT WRITE ANY FILES TO THE EDA OUTPUT DIRECTORY!              #\n"
            "######################################################################\n\n"
            "## YOUR JOB:\n"
            "1. Find the ```python ... ``` code block in the task\n"
            "2. Copy that EXACT code into CodeExecutionTool\n"
            "3. Execute it to create clustering_strategy.json\n\n"
            "## WHAT THE CODE DOES:\n"
            "- Loads EDA outputs (eda_to_segmentation_context.json, etc.)\n"
            "- Creates an adaptive clustering strategy\n"
            "- Saves clustering_strategy.json\n\n"
            "## CRITICAL: DO NOT write, create, or modify ANY files in the eda_output directory!\n"
            "## The eda_output directory is READ-ONLY. Only write to seg_output.\n\n"
            "## OUTPUT LIMIT: MAX 10 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_segmentation_executor_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Segmentation Executor agent - MUST use run_segmentation_pipeline utility.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="segmentation_executor",
        role="Segmentation Pipeline Executor",
        goal=(
            "EXECUTE the exact Python code provided in the task description using CodeExecutionTool. "
            "DO NOT write your own code. JUST RUN the provided code block."
        ),
        backstory=(
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE THE EXACT CODE PROVIDED IN THE TASK       #\n"
            "#  DO NOT INVENT YOUR OWN SEGMENTATION CODE!                         #\n"
            "#  THE TASK CONTAINS A COMPLETE PYTHON CODE BLOCK - RUN IT EXACTLY  #\n"
            "#  DO NOT WRITE ANY FILES TO THE EDA OUTPUT DIRECTORY!              #\n"
            "######################################################################\n\n"
            "## YOUR ONLY JOB:\n"
            "1. Find the ```python ... ``` code block in the task description\n"
            "2. Copy that EXACT code into CodeExecutionTool\n"
            "3. Run it - do NOT modify, simplify, or rewrite the code\n\n"
            "## WHAT THE CODE DOES (for your understanding):\n"
            "- Uses `run_segmentation_pipeline()` from utils/segmentation.py\n"
            "- Creates ALL features: volume, intermittency, variability, temporal\n"
            "- Produces per_key_with_segments.csv with 30+ columns (not just 4!)\n"
            "- Creates visualizations: .png files\n"
            "- Creates segment_profiles.json with keys '0', '1', '2' (not 'S1', 'S2'!)\n\n"
            "## WHAT YOU MUST NOT DO:\n"
            "- Do NOT write your own sklearn/clustering code\n"
            "- Do NOT create simplified versions\n"
            "- Do NOT skip the run_segmentation_pipeline call\n"
            "- Do NOT use only 2 features (cv, zero_fraction) - use ALL features\n"
            "- Do NOT write, create, or modify ANY files in the eda_output directory!\n\n"
            "## OUTPUT LIMIT: MAX 10 LINES OF PRINT OUTPUT"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_segmentation_analyst_agent(llm: LLM, allowed_model_families: list, enable_deep_models: bool, protected_paths: list = None) -> Agent:
    """
    Create the Segmentation Analyst agent - creates intelligent context files.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="segmentation_analyst",
        role="PhD-Level Demand Segmentation & Model Strategy Expert",
        goal=(
            "Read segmentation outputs, extract segment profiles, and create 2 context JSON files "
            "with intelligent recommendations for Feature Engineering and Training crews."
        ),
        backstory=(
            "You are a PhD-level expert in demand forecasting segmentation.\n\n"
            "## CRITICAL: MINIMAL OUTPUT RULES\n"
            "######################################################################\n"
            "#  MAXIMUM 5 PRINT STATEMENTS TOTAL - NO EXCEPTIONS                  #\n"
            "#  SUPPRESS ALL WARNINGS: warnings.filterwarnings('ignore')          #\n"
            "######################################################################\n\n"
            "## YOUR EXPERTISE:\n"
            "- Syntetos-Boylan: ADI>1.32 = intermittent, CV²>0.49 = erratic\n"
            "- Segment profiling: relative_index = segment_mean / overall_mean\n"
            "- Model selection (FEATURE-BASED ONLY):\n"
            "  smooth→lightgbm/xgboost, erratic→xgboost/catboost,\n"
            "  intermittent→zero_inflated/hurdle, lumpy→hurdle/tweedie\n"
            "- BANNED: croston, sba, tsb, imapa, arima, ets, theta, prophet\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n"
            f"## DEEP MODELS ENABLED: {enable_deep_models}"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_segmentation_reviewer_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """Create the Segmentation Reviewer agent that validates segmentation outputs."""
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="segmentation_reviewer",
        role="Segmentation Quality Assurance & Validation Specialist",
        goal=(
            "Review and validate all segmentation outputs for completeness, accuracy, and actionability."
        ),
        backstory=(
            "You are a senior quality assurance specialist.\n\n"
            "## OUTPUT LIMIT: MAX 10 PRINT STATEMENTS\n"
            "## VALIDATION:\n"
            "1. All segments >= 10% size\n"
            "2. Model recommendations match demand patterns\n"
            "3. Context files created correctly"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_segmentation_documentation_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Segmentation Documentation Agent that generates comprehensive markdown documentation.

    This agent uses CodeExecutionTool to ITERATIVELY explore segmentation outputs
    and creates a comprehensive insights markdown report through multiple phases.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="segmentation_documentation_agent",
        role="PhD-Level Segmentation Insights Documentation Specialist",
        goal=(
            "EXECUTE Python code using CodeExecutionTool to explore segmentation outputs "
            "and create SEGMENTATION_INSIGHTS_GUIDE.md. You MUST use the tool to run code - "
            "do NOT just describe or print the analysis."
        ),
        backstory=(
            "You are a PhD-level expert in demand segmentation and clustering.\n\n"
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE CodeExecutionTool TO RUN ANALYSIS CODE    #\n"
            "#  EXPLORE DATA ITERATIVELY - RUN MULTIPLE CODE EXECUTIONS          #\n"
            "#  SAVE FINAL REPORT TO FILE - DO NOT JUST PRINT IT                 #\n"
            "######################################################################\n\n"
            "## YOUR EXPERTISE:\n"
            "- Demand segmentation using clustering algorithms (GMM, K-Means, HDBSCAN)\n"
            "- Silhouette analysis and cluster quality metrics\n"
            "- Demand patterns: smooth, erratic, intermittent, lumpy\n"
            "- Model selection by segment characteristics\n\n"
            "## HOW TO COMPLETE THIS TASK:\n"
            "1. Use CodeExecutionTool to run exploration code\n"
            "2. Run MULTIPLE code executions to build understanding\n"
            "3. Extract specific statistics (counts, percentages, means)\n"
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


def create_segmentation_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> Crew:
    """
    Create the Segmentation Crew with INTELLIGENT utility orchestration.

    This crew READS EDA recommendations and uses individual utility functions
    to build appropriate segmentation - NOT just calling run_segmentation_pipeline().
    """
    artifact_base = config.artifact_base_path
    seg_dir = os.path.join(artifact_base, "seg_output")
    os.makedirs(seg_dir, exist_ok=True)

    eda_dir = os.path.join(artifact_base, "eda_output")

    # Get safe output paths for CrewAI Task output_file
    # Note: CrewAI 1.9.1+ rejects relative paths with '..' for security
    seg_dir_out = _get_output_path(seg_dir)
    eda_dir_out = _get_output_path(eda_dir)

    # Get allowed model families from config.design
    allowed_model_families = list(config.design.model_families)
    enable_deep_models = config.design.enable_deep_models

    # Filter out deep learning models if disabled
    deep_model_types = ['tft', 'lstm', 'nbeats', 'deepar', 'wavenet']
    if not enable_deep_models:
        allowed_model_families = [m for m in allowed_model_families if m.lower() not in deep_model_types]

    # Get config details
    key_columns = list(config.key_columns)
    time_format = getattr(config, 'time_format', 'year_week')

    # WRITE PROTECTION: Prevent LLM agents from corrupting EDA outputs
    # The segmentation agents should only READ from eda_output, never WRITE to it
    protected_dirs = [eda_dir]

    # Create agents with write-protection for EDA output directory
    planner = _create_segmentation_planner_agent(llm, protected_paths=protected_dirs)
    executor = _create_segmentation_executor_agent(llm, protected_paths=protected_dirs)
    analyst = _create_segmentation_analyst_agent(llm, allowed_model_families, enable_deep_models, protected_paths=protected_dirs)

    # -------------------------------------------------------------------------
    # Task 1: Segmentation Planning - READ ALL EDA OUTPUTS AND CREATE STRATEGY
    # -------------------------------------------------------------------------
    task_plan = Task(
        name="create_intelligent_strategy",
        description=(
            "# EXECUTE CODE TO CREATE CLUSTERING STRATEGY\n\n"
            "######################################################################\n"
            "# CRITICAL: YOU MUST USE CodeExecutionTool TO RUN THE CODE BELOW!   #\n"
            "# DO NOT just describe what the code does - ACTUALLY EXECUTE IT!     #\n"
            "# Copy the entire ```python ... ``` block into CodeExecutionTool    #\n"
            "######################################################################\n\n"
            "## REQUIRED ACTION:\n"
            "1. Copy the Python code block below\n"
            "2. Paste it into CodeExecutionTool\n"
            "3. Execute it - this will create clustering_strategy.json\n\n"
            "```python\n"
            "import os\n"
            "import json\n"
            "import pandas as pd\n\n"
            f"eda_dir = '{eda_dir}'\n"
            f"seg_dir = '{seg_dir}'\n\n"
            "# =================================================================\n"
            "# LOAD ALL EDA OUTPUTS (Core + Exhaustive)\n"
            "# =================================================================\n"
            "eda_ctx = {}\n"
            "seasonality = {}\n"
            "trend_analysis = {}\n"
            "changepoints = {}\n"
            "data_profile = {}\n\n"
            "# Core segmentation context\n"
            "eda_ctx_path = os.path.join(eda_dir, 'eda_to_segmentation_context.json')\n"
            "if os.path.exists(eda_ctx_path):\n"
            "    with open(eda_ctx_path) as f:\n"
            "        eda_ctx = json.load(f)\n\n"
            "# Exhaustive analysis outputs\n"
            "if os.path.exists(os.path.join(eda_dir, 'seasonality_analysis.json')):\n"
            "    with open(os.path.join(eda_dir, 'seasonality_analysis.json')) as f:\n"
            "        seasonality = json.load(f)\n"
            "if os.path.exists(os.path.join(eda_dir, 'trend_analysis.json')):\n"
            "    with open(os.path.join(eda_dir, 'trend_analysis.json')) as f:\n"
            "        trend_analysis = json.load(f)\n"
            "if os.path.exists(os.path.join(eda_dir, 'changepoint_analysis.json')):\n"
            "    with open(os.path.join(eda_dir, 'changepoint_analysis.json')) as f:\n"
            "        changepoints = json.load(f)\n"
            "if os.path.exists(os.path.join(eda_dir, 'data_profile.json')):\n"
            "    with open(os.path.join(eda_dir, 'data_profile.json')) as f:\n"
            "        data_profile = json.load(f)\n\n"
            "# =================================================================\n"
            "# EXTRACT INSIGHTS FOR ADAPTIVE STRATEGY\n"
            "# =================================================================\n"
            "# Core recommendations\n"
            "algorithm = eda_ctx.get('algorithm', 'GaussianMixture')\n"
            "features = eda_ctx.get('recommended_clustering_features', [])\n"
            "cluster_count = eda_ctx.get('cluster_count', {})\n"
            "k_min = cluster_count.get('min', 3)\n"
            "k_max = cluster_count.get('max', 8)\n"
            "k_rec = cluster_count.get('recommended', 5)\n\n"
            "# Exhaustive insights for adaptive clustering\n"
            "exhaustive_insights = eda_ctx.get('exhaustive_analysis_insights', {})\n"
            "has_seasonality_pct = exhaustive_insights.get('has_seasonality_pct', seasonality.get('has_seasonality_pct', 0))\n"
            "avg_trend_strength = exhaustive_insights.get('avg_trend_strength', trend_analysis.get('avg_trend_strength', 0))\n"
            "strongly_trending_pct = exhaustive_insights.get('strongly_trending_pct', trend_analysis.get('strongly_trending_pct', 0))\n"
            "pct_with_changepoints = exhaustive_insights.get('pct_with_changepoints', changepoints.get('pct_with_changepoints', 0))\n\n"
            "# Demand distribution\n"
            "demand_dist = eda_ctx.get('data_summary', {}).get('demand_distribution', {})\n"
            "lumpy_pct = demand_dist.get('lumpy', 0)\n"
            "smooth_pct = demand_dist.get('smooth', 0)\n\n"
            "print(f'EDA Insights Summary:')\n"
            "print(f'  Algorithm: {algorithm}, K: {k_min}-{k_max} (rec: {k_rec})')\n"
            "print(f'  Patterns: smooth={smooth_pct*100:.0f}%, lumpy={lumpy_pct*100:.0f}%')\n"
            "print(f'  Seasonality: {has_seasonality_pct*100:.0f}%, Trending: {strongly_trending_pct*100:.0f}%')\n"
            "print(f'  Changepoints: {pct_with_changepoints*100:.0f}% series')\n\n"
            "# =================================================================\n"
            "# CREATE ADAPTIVE CLUSTERING STRATEGY\n"
            "# =================================================================\n"
            "# Decide clustering strategy based on EDA insights\n"
            "use_adaptive = True  # Use adaptive feature selection\n\n"
            "# Adjust features based on what EDA found\n"
            "adaptive_features = list(features) if features else ['volume_mean', 'cv_clean', 'zero_fraction_clean']\n\n"
            "# Add seasonal features if seasonality detected\n"
            "if has_seasonality_pct > 0.3:\n"
            "    if 'seasonal_strength_clean' not in adaptive_features:\n"
            "        adaptive_features.append('seasonal_strength_clean')\n"
            "    print('  -> Adding seasonal features (high seasonality detected)')\n\n"
            "# Add trend features if significant trending\n"
            "if strongly_trending_pct > 0.3:\n"
            "    if 'trend_strength_clean' not in adaptive_features:\n"
            "        adaptive_features.append('trend_strength_clean')\n"
            "    print('  -> Adding trend features (significant trending detected)')\n\n"
            "# Add lifecycle/complexity if changepoints detected\n"
            "if pct_with_changepoints > 0.4:\n"
            "    adaptive_features.extend(['lifecycle_stage_numeric', 'complexity_score'])\n"
            "    print('  -> Adding lifecycle features (changepoints detected)')\n\n"
            "# Always include forecastability for segment quality\n"
            "if 'forecastability_score' not in adaptive_features:\n"
            "    adaptive_features.append('forecastability_score')\n\n"
            "strategy = {\n"
            "    'algorithm': algorithm,\n"
            "    'features': adaptive_features,\n"
            "    'k_min': k_min,\n"
            "    'k_max': k_max,\n"
            "    'k_recommended': k_rec,\n"
            "    'scaler': 'robust',\n"
            "    'min_segment_pct': 0.10,\n"
            "    'use_adaptive_features': use_adaptive,\n"
            "    # Store EDA insights for downstream context\n"
            "    'eda_insights': {\n"
            "        'has_seasonality_pct': has_seasonality_pct,\n"
            "        'strongly_trending_pct': strongly_trending_pct,\n"
            "        'pct_with_changepoints': pct_with_changepoints,\n"
            "        'lumpy_pct': lumpy_pct,\n"
            "        'smooth_pct': smooth_pct,\n"
            "    }\n"
            "}\n\n"
            "with open(os.path.join(seg_dir, 'clustering_strategy.json'), 'w') as f:\n"
            "    json.dump(strategy, f, indent=2, default=str)\n"
            "print(f'Strategy saved with {len(adaptive_features)} adaptive features')\n\n"
            "# Load per-key metrics to check shape\n"
            "per_key_path = os.path.join(eda_dir, 'per_key_metrics.csv')\n"
            "if os.path.exists(per_key_path):\n"
            "    df = pd.read_csv(per_key_path)\n"
            "    print(f'Per-key metrics: {len(df)} series, {len(df.columns)} columns')\n"
            "```\n\n"
            "Then tell Executor: 'Execute ADAPTIVE clustering using the strategy in clustering_strategy.json'"
        ),
        agent=planner,
        expected_output=(
            "All EDA outputs read (core + exhaustive). Adaptive strategy created with features based on "
            "detected seasonality, trend, and changepoints. Saved to clustering_strategy.json."
        ),
        output_file=os.path.join(seg_dir_out, "segmentation_strategy.md"),
        # CRITICAL: Callback to validate Planner output BEFORE Executor starts
        callback=lambda output: _validate_planner_output(seg_dir),
    )

    # -------------------------------------------------------------------------
    # Task 2: Segmentation Execution - USE run_segmentation_pipeline UTILITY
    # -------------------------------------------------------------------------
    task_execute = Task(
        name="execute_segmentation_pipeline",
        description=(
            "# EXECUTE THE SEGMENTATION PIPELINE CODE BELOW\n\n"
            "######################################################################\n"
            "# CRITICAL: COPY THE PYTHON CODE BLOCK BELOW INTO CodeExecutionTool  #\n"
            "# DO NOT WRITE YOUR OWN CODE - JUST RUN THIS EXACT CODE!             #\n"
            "######################################################################\n\n"
            "The code uses `run_segmentation_pipeline()` which creates:\n"
            "- per_key_with_segments.csv with 30+ columns (demand_pattern, volume_tier, etc.)\n"
            "- segment_profiles.json with keys '0', '1', '2' (NOT 'S1', 'S2', 'S3')\n"
            "- Visualizations: segment_distribution.png, segment_heatmap.png, etc.\n\n"
            "## COPY AND RUN THIS CODE EXACTLY:\n"
            "```python\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import os\n"
            "import json\n"
            "import pandas as pd\n"
            "from utils.segmentation import run_segmentation_pipeline\n\n"
            f"eda_dir = '{eda_dir}'\n"
            f"seg_dir = '{seg_dir}'\n"
            f"key_columns = {key_columns}\n"
            f"allowed_models = {allowed_model_families}\n"
            f"time_format = '{time_format}'\n"
            f"enable_deep_models = {enable_deep_models}\n\n"
            "# 1. Load strategy from Planner\n"
            "with open(os.path.join(seg_dir, 'clustering_strategy.json')) as f:\n"
            "    strategy = json.load(f)\n\n"
            "algorithm = strategy.get('algorithm', 'gmm').lower()\n"
            "k_min = strategy.get('k_min', 3)\n"
            "k_max = strategy.get('k_max', 7)\n"
            "min_pct = strategy.get('min_segment_pct', 0.10)\n"
            "use_adaptive = strategy.get('use_adaptive_features', True)\n"
            "print(f'Strategy: {algorithm}, k={k_min}-{k_max}, adaptive={use_adaptive}')\n\n"
            "# 2. Load per-key metrics from EDA\n"
            "per_key_df = pd.read_csv(os.path.join(eda_dir, 'per_key_metrics.csv'))\n"
            "print(f'Loaded {len(per_key_df)} series from EDA')\n\n"
            "# 2.5 Load source data for Phase 2 enriched segmentation\n"
            f"source_data_path = '{config.input_data_path}'\n"
            "try:\n"
            "    from utils.agent_utilities import load_source_data\n"
            "    source_df = load_source_data(source_data_path)\n"
            "    print(f'Loaded source data: {len(source_df)} rows for enriched segmentation')\n"
            "except Exception as e:\n"
            "    print(f'Could not load source data (Phase 2 enrichment will be skipped): {e}')\n"
            "    source_df = None\n\n"
            "# 2.6 Load feature availability context (if available)\n"
            f"fa_context_path = os.path.join('{config.artifact_base_path}', 'feature_availability_output', 'feature_availability_to_feature_context.json')\n"
            "fa_context = None\n"
            "if os.path.exists(fa_context_path):\n"
            "    with open(fa_context_path) as f:\n"
            "        fa_context = json.load(f)\n"
            "    print(f'Loaded feature availability context')\n"
            "else:\n"
            "    print('No feature availability context found (will auto-detect features)')\n\n"
            "# 3. Run the complete segmentation pipeline\n"
            "# This handles: features, EDA enrichment, Phase 2 enrichment, clustering, profiling, recommendations, saving\n"
            "result = run_segmentation_pipeline(\n"
            "    per_key_metrics=per_key_df,\n"
            "    key_cols=key_columns,\n"
            "    output_dir=seg_dir,\n"
            "    allowed_model_families=allowed_models,\n"
            "    enable_deep_models=enable_deep_models,\n"
            "    time_format=time_format,\n"
            "    clustering_method=algorithm,\n"
            "    n_clusters_range=list(range(k_min, k_max + 1)),\n"
            "    min_segment_pct=min_pct,\n"
            "    create_visualizations=True,\n"
            "    eda_dir=eda_dir,\n"
            "    use_adaptive_features=use_adaptive,\n"
            "    # HYBRID SEGMENTATION: Combine clusters with business dimensions\n"
            "    use_hybrid_segmentation=True,\n"
            "    hybrid_use_volume_tier=True,\n"
            "    hybrid_use_demand_pattern=True,\n"
            "    # Phase 2: Enriched segmentation with external features\n"
            "    source_df=source_df,\n"
            f"    date_col='{config.timestamp_col}',\n"
            f"    target_col='{config.target_col}',\n"
            "    feature_availability_context=fa_context,\n"
            "    enable_enriched_features=True,\n"
            ")\n\n"
            "print(f'Segmentation complete: {result.n_segments} segments')\n"
            "print(f'Silhouette score: {result.clustering_metrics.get(\"silhouette\", 0):.3f}')\n"
            "print(f'Files created in {seg_dir}')\n"
            "```\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n"
            f"## MINIMUM SEGMENT SIZE: 10%"
        ),
        agent=executor,
        expected_output=(
            "Segmentation pipeline complete. Created {N} segments with minimum 10% size each. "
            "Files: per_key_with_segments.csv, segment_profiles.json, model_recommendations.json, "
            "feature_recommendations.json, clustering_metrics.json, segmentation_to_feature_context.json, "
            "segmentation_to_training_context.json"
        ),
        output_file=os.path.join(seg_dir_out, "segmentation_execution_report.md"),
        context=[task_plan],
        # CRITICAL: Callback to validate Executor output BEFORE Analyst starts
        callback=lambda output: _validate_executor_output(seg_dir),
    )

    # -------------------------------------------------------------------------
    # Task 3: Segmentation Analysis - VALIDATE and SUMMARIZE context files
    # NOTE: Context files are now created by Executor using deterministic utility
    # -------------------------------------------------------------------------
    task_analyze = Task(
        name="validate_and_summarize",
        description=(
            "# VALIDATE SEGMENTATION OUTPUTS AND CREATE SUMMARY\n\n"
            "The Executor has already created context files using deterministic utilities.\n"
            "Your job is to VALIDATE these files exist and create a summary report.\n\n"
            "## DO NOT INVENT OR HALLUCINATE ANY FEATURES OR RECOMMENDATIONS!\n"
            "## ONLY READ AND REPORT WHAT IS IN THE ACTUAL FILES!\n\n"
            f"## SEGMENTATION OUTPUT DIRECTORY: `{seg_dir}`\n\n"
            "## EXECUTE THIS VALIDATION CODE:\n"
            "```python\n"
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import os, json\n\n"
            f"seg_dir = '{seg_dir}'\n\n"
            "# Validate all required files exist\n"
            "required_files = [\n"
            "    'per_key_with_segments.csv',\n"
            "    'segment_profiles.json',\n"
            "    'model_recommendations.json',\n"
            "    'feature_recommendations.json',\n"
            "    'clustering_metrics.json',\n"
            "    'segmentation_to_feature_context.json',\n"
            "    'segmentation_to_training_context.json',\n"
            "]\n\n"
            "missing = []\n"
            "for f in required_files:\n"
            "    path = os.path.join(seg_dir, f)\n"
            "    if not os.path.exists(path):\n"
            "        missing.append(f)\n\n"
            "if missing:\n"
            "    print(f'ERROR: Missing files: {missing}')\n"
            "else:\n"
            "    print('All required files present')\n\n"
            "# Read and summarize clustering metrics\n"
            "with open(os.path.join(seg_dir, 'clustering_metrics.json')) as f:\n"
            "    metrics = json.load(f)\n\n"
            "print(f'Algorithm: {metrics.get(\"algorithm\")}')\n"
            "print(f'Segments: {metrics.get(\"n_segments\")}')\n"
            "print(f'Silhouette: {metrics.get(\"silhouette\", 0):.3f}')\n"
            "print(f'Labels: {metrics.get(\"segment_labels\", {})}')\n"
            "```\n\n"
            "## YOUR OUTPUT:\n"
            "Report the validation results. Do NOT invent any features or recommendations.\n"
            "Just report what the files contain."
        ),
        agent=analyst,
        expected_output=(
            "Validated all segmentation output files. Context files created by utility function. "
            "N segments with silhouette score X. Files ready for downstream crews."
        ),
        output_file=os.path.join(seg_dir_out, "segmentation_analysis_report.md"),
        context=[task_plan, task_execute],
        # CRITICAL: Callback to validate Analyst output for downstream crews
        callback=lambda output: _validate_analyst_output(seg_dir),
    )

    # -------------------------------------------------------------------------
    # Task 4: Documentation - Generate comprehensive insights guide
    # Uses MULTI-PHASE ITERATIVE approach like EDA Insights Agent
    # -------------------------------------------------------------------------
    documentation_agent = _create_segmentation_documentation_agent(llm, protected_paths=protected_dirs)
    task_document = Task(
        name="generate_segmentation_documentation",
        description=(
            "# SEGMENTATION INSIGHTS REPORT GENERATION\n\n"
            f"Analyze segmentation outputs in `{seg_dir}` and create a comprehensive insights report.\n\n"
            "## AVAILABLE FILES\n\n"
            f"- `{seg_dir}/per_key_with_segments.csv` - Main segmentation results\n"
            f"- `{seg_dir}/segment_profiles.json` - Segment characteristics\n"
            f"- `{seg_dir}/model_recommendations.json` - Model recommendations by segment\n"
            f"- `{seg_dir}/clustering_metrics.json` - Clustering quality metrics\n"
            f"- `{seg_dir}/segmentation_to_feature_context.json` - Context for feature engineering\n"
            f"- `{seg_dir}/segmentation_to_training_context.json` - Context for training\n\n"
            "## PHASE 1: EXPLORE SEGMENTATION STRUCTURE\n\n"
            "Execute this code to understand the segmentation outputs:\n\n"
            "```python\n"
            "import pandas as pd\n"
            "import json\n"
            "import os\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n\n"
            f"seg_dir = '{seg_dir}'\n\n"
            "# List files\n"
            "files = [f for f in os.listdir(seg_dir) if f.endswith(('.csv', '.json', '.md', '.png'))]\n"
            "print(f'Available files: {{len(files)}}')\n"
            "for f in sorted(files)[:15]: print(f'  - {{f}}')\n\n"
            "# Load main segmentation results\n"
            "df = pd.read_csv(f'{{seg_dir}}/per_key_with_segments.csv')\n"
            "print(f'\\nSegmentation results: {{df.shape[0]}} series, {{df.shape[1]}} columns')\n"
            "print(f'Columns: {{list(df.columns)[:15]}}...')\n\n"
            "# Check segment distribution\n"
            "if 'segment_id' in df.columns:\n"
            "    n_segments = df['segment_id'].nunique()\n"
            "    print(f'Number of segments: {{n_segments}}')\n"
            "```\n\n"
            "## PHASE 2: CLUSTERING QUALITY ANALYSIS\n\n"
            "Execute this code to analyze clustering quality:\n\n"
            "```python\n"
            "import json\n"
            "import os\n\n"
            f"seg_dir = '{seg_dir}'\n\n"
            "# Load clustering metrics\n"
            "with open(f'{{seg_dir}}/clustering_metrics.json') as f:\n"
            "    metrics = json.load(f)\n\n"
            "print('=== CLUSTERING QUALITY METRICS ===')\n"
            "algorithm = metrics.get('algorithm', 'unknown')\n"
            "silhouette = metrics.get('silhouette', 0)\n"
            "calinski = metrics.get('calinski_harabasz', 0)\n"
            "davies = metrics.get('davies_bouldin', 0)\n"
            "n_segments = metrics.get('n_segments', 0)\n"
            "features_used = metrics.get('features_used', [])\n\n"
            "print(f'Algorithm: {{algorithm}}')\n"
            "print(f'Number of segments: {{n_segments}}')\n"
            "print(f'Silhouette Score: {{silhouette:.3f}} ({\"Excellent\" if silhouette > 0.5 else \"Good\" if silhouette > 0.3 else \"Fair\"})')\n"
            "print(f'Calinski-Harabasz: {{calinski:.1f}}')\n"
            "print(f'Davies-Bouldin: {{davies:.3f}}')\n"
            "print(f'\\nFeatures used ({len(features_used)}): {{features_used[:5]}}...')\n"
            "```\n\n"
            "## PHASE 3: SEGMENT PROFILE ANALYSIS\n\n"
            "Execute this code to analyze segment profiles:\n\n"
            "```python\n"
            "import pandas as pd\n"
            "import json\n\n"
            f"seg_dir = '{seg_dir}'\n\n"
            "# Load segment profiles\n"
            "with open(f'{{seg_dir}}/segment_profiles.json') as f:\n"
            "    profiles = json.load(f)\n\n"
            "# Load segmentation results for distribution\n"
            "df = pd.read_csv(f'{{seg_dir}}/per_key_with_segments.csv')\n"
            "n_series = len(df)\n"
            "seg_dist = df['segment_id'].value_counts().to_dict() if 'segment_id' in df.columns else {{}}\n\n"
            "print('=== SEGMENT DISTRIBUTION ===')\n"
            "for seg_id, count in sorted(seg_dist.items()):\n"
            "    pct = count / n_series * 100\n"
            "    profile = profiles.get(str(seg_id), {{}})\n"
            "    pattern = profile.get('dominant_pattern', 'unknown')\n"
            "    vol_tier = profile.get('dominant_volume_tier', 'medium')\n"
            "    print(f'Segment {{seg_id}}: {{count:,}} series ({{pct:.1f}}%) - {{pattern.title()}} / {{vol_tier.title()}}')\n\n"
            "print('\\n=== SEGMENT CHARACTERISTICS ===')\n"
            "for seg_id, profile in sorted(profiles.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):\n"
            "    means = profile.get('means', {{}})\n"
            "    cv_mean = means.get('cv_clean', 0)\n"
            "    zero_frac = means.get('zero_fraction_clean', 0)\n"
            "    print(f'Segment {{seg_id}}: CV={{cv_mean:.3f}}, Zero Fraction={{zero_frac:.3f}}')\n"
            "```\n\n"
            "## PHASE 4: MODEL RECOMMENDATIONS ANALYSIS\n\n"
            "Execute this code to analyze model recommendations:\n\n"
            "```python\n"
            "import json\n\n"
            f"seg_dir = '{seg_dir}'\n\n"
            "# Load model recommendations\n"
            "with open(f'{{seg_dir}}/model_recommendations.json') as f:\n"
            "    recs = json.load(f)\n\n"
            "print('=== MODEL RECOMMENDATIONS BY SEGMENT ===')\n"
            "segment_strategies = recs.get('segment_strategies', {{}})\n"
            "for seg_id, strategy in segment_strategies.items():\n"
            "    primary = strategy.get('primary_model', 'lightgbm')\n"
            "    loss_config = strategy.get('loss_config', {{}})\n"
            "    loss = loss_config.get('primary_loss', 'mse')\n"
            "    wape_range = strategy.get('expected_wape_range', {{}})\n"
            "    wape_str = f\"{{wape_range.get('min', 0)}}-{{wape_range.get('max', 0)}}%\" if wape_range else 'N/A'\n"
            "    pattern = strategy.get('dominant_pattern', 'unknown')\n"
            "    print(f'Segment {{seg_id}}: {{primary}} with {{loss}} loss, Expected WAPE: {{wape_str}} ({{pattern.title()}})')\n\n"
            "# Count model types\n"
            "model_counts = {{}}\n"
            "for strategy in segment_strategies.values():\n"
            "    model = strategy.get('primary_model', 'unknown')\n"
            "    model_counts[model] = model_counts.get(model, 0) + 1\n"
            "print(f'\\nModel distribution: {{model_counts}}')\n"
            "```\n\n"
            "## PHASE 5: GENERATE AND SAVE REPORT\n\n"
            "After gathering all insights from the previous phases, execute code to CREATE and SAVE the report.\n"
            "YOU MUST use the ACTUAL NUMBERS from your previous code executions. Do NOT use placeholders.\n\n"
            "```python\n"
            "from datetime import datetime\n"
            "import pandas as pd\n"
            "import json\n"
            "import os\n\n"
            f"seg_dir = '{seg_dir}'\n\n"
            "# Re-load all data to build comprehensive report\n"
            "df = pd.read_csv(f'{{seg_dir}}/per_key_with_segments.csv')\n"
            "with open(f'{{seg_dir}}/clustering_metrics.json') as f:\n"
            "    metrics = json.load(f)\n"
            "with open(f'{{seg_dir}}/segment_profiles.json') as f:\n"
            "    profiles = json.load(f)\n"
            "with open(f'{{seg_dir}}/model_recommendations.json') as f:\n"
            "    recs = json.load(f)\n\n"
            "# Extract statistics\n"
            "n_series = len(df)\n"
            "n_segments = df['segment_id'].nunique() if 'segment_id' in df.columns else 0\n"
            "algorithm = metrics.get('algorithm', 'unknown')\n"
            "silhouette = metrics.get('silhouette', 0)\n"
            "calinski = metrics.get('calinski_harabasz', 0)\n"
            "davies = metrics.get('davies_bouldin', 0)\n"
            "features_used = metrics.get('features_used', [])\n"
            "seg_dist = df['segment_id'].value_counts().to_dict() if 'segment_id' in df.columns else {{}}\n"
            "segment_strategies = recs.get('segment_strategies', {{}})\n\n"
            "# Build comprehensive markdown report\n"
            "md = f'''# Segmentation Insights Guide\n"
            "**Generated:** {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\n\n"
            "---\n\n"
            "## Executive Summary\n\n"
            "This report provides comprehensive analysis of the demand forecasting segmentation results.\n"
            "The data was segmented into **{{n_segments}} distinct clusters** using **{{algorithm}}** algorithm,\n"
            "achieving a silhouette score of **{{silhouette:.3f}}** ({\"excellent\" if silhouette > 0.5 else \"good\" if silhouette > 0.3 else \"fair\"} cluster separation).\n\n"
            "A total of **{{n_series:,}} time series** were analyzed and grouped based on their demand patterns,\n"
            "volume characteristics, and variability metrics.\n\n"
            "---\n\n"
            "## 1. Clustering Quality Analysis\n\n"
            "### Quality Metrics\n\n"
            "| Metric | Value | Quality | Interpretation |\n"
            "|--------|-------|---------|----------------|\n"
            "| Silhouette Score | {{silhouette:.3f}} | {\"✓ Excellent\" if silhouette > 0.5 else \"✓ Good\" if silhouette > 0.3 else \"⚠ Fair\"} | Cluster cohesion (-1 to 1, higher is better) |\n"
            "| Calinski-Harabasz | {{calinski:.1f}} | {\"✓ Good\" if calinski > 100 else \"Fair\"} | Between/within cluster variance ratio |\n"
            "| Davies-Bouldin | {{davies:.3f}} | {\"✓ Good\" if davies < 1 else \"Fair\"} | Cluster similarity (lower is better) |\n\n"
            "### Features Used for Clustering\\n\\n'''\n\n"
            "for i, feat in enumerate(features_used[:10], 1):\n"
            "    md += f'{{i}}. `{{feat}}`\\n'\n"
            "if len(features_used) > 10:\n"
            "    md += f'\\n... and {{len(features_used) - 10}} more features\\n'\n\n"
            "md += f'''\\n\\n---\\n\\n"
            "## 2. Segment Distribution\\n\\n"
            "| Segment | Size | % of Total | Dominant Pattern | Volume Tier |\\n"
            "|---------|------|------------|------------------|-------------|\\n'''\n\n"
            "for seg_id, count in sorted(seg_dist.items()):\n"
            "    pct = count / n_series * 100\n"
            "    profile = profiles.get(str(seg_id), {{}})\n"
            "    pattern = profile.get('dominant_pattern', 'unknown')\n"
            "    vol_tier = profile.get('dominant_volume_tier', 'medium')\n"
            "    md += f'| Segment {{seg_id}} | {{count:,}} | {{pct:.1f}}% | {{pattern.title()}} | {{vol_tier.title()}} |\\n'\n\n"
            "md += f'''\\n\\n---\\n\\n"
            "## 3. Detailed Segment Profiles\\n\\n'''\n\n"
            "for seg_id, profile in sorted(profiles.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):\n"
            "    dominant = profile.get('dominant_pattern', 'unknown')\n"
            "    vol_tier = profile.get('dominant_volume_tier', 'medium')\n"
            "    size = profile.get('size', seg_dist.get(int(seg_id) if seg_id.isdigit() else seg_id, 0))\n"
            "    pct = size / n_series * 100 if n_series > 0 else 0\n"
            "    cv_mean = profile.get('cv_mean', 0)\n"
            "    zero_frac = profile.get('zero_fraction_mean', 0)\n"
            "    pattern_dist = profile.get('pattern_distribution', {{}})\n"
            "    rel_indices = profile.get('relative_indices', {{}})\n\n"
            "    md += f'''### Segment {{seg_id}}: {{dominant.title()}} - {{vol_tier.title()}} Volume\\n\\n"
            "**Key Characteristics:**\\n"
            "- Size: {{size:,}} series ({{pct:.1f}}% of total)\\n"
            "- Dominant Pattern: {{dominant.title()}}\\n"
            "- Volume Tier: {{vol_tier.title()}}\\n"
            "- Average CV: {{cv_mean:.3f}}\\n"
            "- Average Zero Fraction: {{zero_frac:.3f}}\\n\\n'''\n\n"
            "    if pattern_dist:\n"
            "        pattern_str = ', '.join([f'{{k}}: {{v:.1f}}%' for k, v in pattern_dist.items()])\n"
            "        md += f'**Pattern Mix:** {{pattern_str}}\\n\\n'\n\n"
            "    if rel_indices:\n"
            "        md += '**Relative to Average:**\\n'\n"
            "        for feat, val in list(rel_indices.items())[:5]:\n"
            "            direction = '↑ above' if val > 1.1 else '↓ below' if val < 0.9 else '≈ near'\n"
            "            md += f'- {{feat}}: {{val:.2f}}x ({{direction}} average)\\n'\n"
            "        md += '\\n'\n\n"
            "md += f'''---\\n\\n"
            "## 4. Model Recommendations by Segment\\n\\n"
            "| Segment | Primary Model | Loss Function | Expected WAPE | Rationale |\\n"
            "|---------|---------------|---------------|---------------|-----------|\\n'''\n\n"
            "for seg_id, strategy in segment_strategies.items():\n"
            "    primary = strategy.get('primary_model', 'lightgbm')\n"
            "    loss_config = strategy.get('loss_config', {{}})\n"
            "    loss = loss_config.get('primary_loss', 'mse')\n"
            "    wape_range = strategy.get('expected_wape_range', {{}})\n"
            "    wape_str = f\"{{wape_range.get('min', 0)}}-{{wape_range.get('max', 0)}}%\" if wape_range else 'N/A'\n"
            "    pattern = strategy.get('dominant_pattern', 'unknown')\n"
            "    md += f'| {{seg_id}} | {{primary}} | {{loss}} | {{wape_str}} | {{pattern.title()}} pattern |\\n'\n\n"
            "md += f'''\\n\\n---\\n\\n"
            "## 5. Key Insights & Recommendations\\n\\n"
            "### Why These Segments?\\n\\n"
            "The segmentation groups series with similar:\\n"
            "1. **Demand patterns** (smooth, erratic, intermittent, lumpy)\\n"
            "2. **Volume levels** (high, medium, low)\\n"
            "3. **Variability** (stable vs volatile)\\n\\n"
            "### Actionable Insights\\n\\n'''\n\n"
            "# Generate insights based on actual data\n"
            "lumpy_segs = [s for s, p in profiles.items() if p.get('dominant_pattern') in ['lumpy', 'intermittent']]\n"
            "smooth_segs = [s for s, p in profiles.items() if p.get('dominant_pattern') == 'smooth']\n"
            "erratic_segs = [s for s, p in profiles.items() if p.get('dominant_pattern') == 'erratic']\n\n"
            "if lumpy_segs:\n"
            "    md += f'- **Intermittent Demand:** Segments {{lumpy_segs}} have sparse demand - use Tweedie loss and zero-inflated models.\\n'\n"
            "if smooth_segs:\n"
            "    md += f'- **Predictable Segments:** Segments {{smooth_segs}} have smooth demand - standard models will perform well.\\n'\n"
            "if erratic_segs:\n"
            "    md += f'- **High Variability:** Segments {{erratic_segs}} have erratic demand - consider robust loss functions.\\n'\n"
            "if silhouette < 0.3:\n"
            "    md += '- **Cluster Quality:** Consider adjusting segment count or features for better separation.\\n'\n\n"
            "md += f'''\\n\\n---\\n\\n"
            "*Generated by Segmentation Insights Agent using iterative code execution.*\\n"
            "'''\n\n"
            "# SAVE THE REPORT TO FILE\n"
            "report_path = os.path.join(seg_dir, 'SEGMENTATION_INSIGHTS_GUIDE.md')\n"
            "with open(report_path, 'w') as f:\n"
            "    f.write(md)\n\n"
            "print(f'Saved: SEGMENTATION_INSIGHTS_GUIDE.md ({{len(md):,}} bytes)')\n"
            "print(f'Segments: {{n_segments}}, Series: {{n_series:,}}')\n"
            "print(f'Silhouette: {{silhouette:.3f}}')\n"
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
            "Created SEGMENTATION_INSIGHTS_GUIDE.md - comprehensive documentation with "
            "segment profiles, clustering quality analysis, and model recommendations."
        ),
        output_file=os.path.join(seg_dir_out, "segmentation_documentation_report.md"),
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
        logger.info("SKIPPING segmentation insights report (enable_insights_reports=False)")

    if enable_reviewer:
        reviewer = _create_segmentation_reviewer_agent(llm, protected_paths=protected_dirs)
        task_review = Task(
            name="review_segmentation_outputs",
            description=(
                "# SEGMENTATION OUTPUT QUALITY REVIEW\n\n"
                f"Validate outputs in `{seg_dir}`:\n"
                "1. Check all segments >= 10% size\n"
                "2. Verify context files created\n"
                "3. Create segmentation_review_report.json"
            ),
            agent=reviewer,
            expected_output="Quality review complete. All segments validated.",
            output_file=os.path.join(seg_dir_out, "segmentation_review_summary.md"),
            context=[task_analyze],
        )
        agents.append(reviewer)
        tasks.append(task_review)
        crew_name = "Segmentation Crew (4-Agent Pattern with Reviewer)"
    else:
        crew_name = "Segmentation Crew (3-Agent Pattern)"

    return Crew(
        name=crew_name,
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def run_segmentation_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> SegmentationCrewResult:
    """Run the segmentation crew and return results."""
    from utils.cost_tracking import get_cost_tracker, extract_tokens_from_crew_result

    # ==========================================================================
    # CRITICAL: Validate input data file exists BEFORE running any crew tasks
    # ==========================================================================
    if not os.path.exists(config.input_data_path):
        raise SegmentationFailedError(
            f"Input data file not found: {config.input_data_path}\n"
            "Please ensure the data file exists at the configured path."
        )

    # Start cost tracking
    tracker = get_cost_tracker()
    tracker.start_crew("Segmentation Crew")

    # Get model ID from LLM if available
    model_id = getattr(llm, "model", "default")
    tracker.set_model(model_id)

    # ======================================================================
    # WRITE PROTECTION: Set protected_paths on the LLM's CodeExecutionTool
    # ======================================================================
    # The LLM handles tool calling internally (CrewAI doesn't pass tools to
    # custom BaseLLM). The actual CodeExecutionTool instance is stored on
    # llm._code_execution_tool. We set protected_paths here so the LLM's
    # agents cannot write to the EDA output directory.
    eda_dir = os.path.join(config.artifact_base_path, "eda_output")
    _code_tool = getattr(llm, '_code_execution_tool', None)
    if _code_tool is not None:
        _code_tool.protected_paths = [eda_dir]
        logger.info(f"Write protection set on LLM CodeExecutionTool: eda_output is READ-ONLY")
    else:
        logger.warning("LLM has no _code_execution_tool - write protection not available")

    # ======================================================================
    # DEFENSE-IN-DEPTH: Backup critical EDA files before LLM agents run
    # ======================================================================
    # The segmentation LLM agents have CodeExecutionTool and may write their
    # own code that overwrites EDA output files (e.g., per_key_metrics.csv).
    # Primary protection: CodeExecutionTool.protected_paths blocks writes.
    # Secondary protection: backup/restore ensures recovery if anything slips through.
    import shutil
    eda_dir = os.path.join(config.artifact_base_path, "eda_output")
    _eda_critical_files = [
        'per_key_metrics.csv',
        'eda_to_segmentation_context.json',
        'eda_to_feature_context.json',
        'eda_to_training_context.json',
        'eda_summary.json',
        'data_quality.json',
        'stationarity_results.csv',
        'feature_importance.csv',
        'data_profile.json',
        'seasonality_analysis.json',
        'trend_analysis.json',
        'changepoint_analysis.json',
        'dead_key_summary.json',
        'dead_keys.txt',
    ]
    _eda_backup_dir = os.path.join(eda_dir, '.backup_before_segmentation')
    os.makedirs(_eda_backup_dir, exist_ok=True)
    _eda_backup_sizes = {}
    for fname in _eda_critical_files:
        src = os.path.join(eda_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(_eda_backup_dir, fname))
            _eda_backup_sizes[fname] = os.path.getsize(src)
    logger.info(f"Backed up {len(_eda_backup_sizes)} EDA files before segmentation crew")

    crew = create_segmentation_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)
    crew_result = crew.kickoff()

    # ======================================================================
    # DEFENSE-IN-DEPTH: Restore any EDA files corrupted by LLM agents
    # ======================================================================
    _restored_count = 0
    for fname, orig_size in _eda_backup_sizes.items():
        current_path = os.path.join(eda_dir, fname)
        backup_path = os.path.join(_eda_backup_dir, fname)
        if not os.path.exists(current_path):
            # File was deleted — restore it
            shutil.copy2(backup_path, current_path)
            _restored_count += 1
            logger.warning(f"RESTORED deleted EDA file: {fname}")
        else:
            current_size = os.path.getsize(current_path)
            # If file shrunk to less than 50% of original, it was corrupted
            if orig_size > 0 and current_size < orig_size * 0.5:
                shutil.copy2(backup_path, current_path)
                _restored_count += 1
                logger.warning(
                    f"RESTORED corrupted EDA file: {fname} "
                    f"(was {current_size}B, restored to {orig_size}B)"
                )
    if _restored_count > 0:
        logger.warning(f"RESTORED {_restored_count} EDA files corrupted by segmentation agents")
    else:
        logger.info("EDA files intact after segmentation crew (no restoration needed)")

    seg_dir = os.path.join(config.artifact_base_path, "seg_output")

    # Extract and record tokens from crew result
    tokens = extract_tokens_from_crew_result(crew_result)
    if tokens["total"] > 0:
        tracker.record_llm_call(
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            model=model_id,
        )

    # End tracking and save cost report
    cost_report = tracker.end_crew("Segmentation Crew", seg_dir)
    cost_report_path = os.path.join(seg_dir, "segmentation_cost.json")

    # Build result object with paths to all outputs
    return SegmentationCrewResult(
        seg_dir=seg_dir,
        per_key_with_segments_path=os.path.join(seg_dir, "per_key_with_segments.csv"),
        segmentation_metadata_path=os.path.join(seg_dir, "segment_profiles.json"),
        segmentation_preview_path=os.path.join(seg_dir, "segment_distribution.png"),
        modeling_strategy_path=os.path.join(seg_dir, "model_recommendations.json"),
        feature_recommendations_path=os.path.join(seg_dir, "segmentation_to_feature_context.json"),
        cluster_sizes_chart_path=os.path.join(seg_dir, "segment_distribution.png"),
        segment_sizes_chart_path=os.path.join(seg_dir, "segment_distribution.png"),
        intermittency_mix_chart_path=os.path.join(seg_dir, "segment_radar.png"),
        segmentation_report_markdown_path=os.path.join(seg_dir, "segmentation_analysis_report.md"),
        segmentation_to_feature_context_path=os.path.join(seg_dir, "segmentation_to_feature_context.json"),
        segmentation_to_training_context_path=os.path.join(seg_dir, "segmentation_to_training_context.json"),
        segmentation_pipeline_script_path=os.path.join(seg_dir, "segmentation_strategy.md"),
        cluster_model_path=os.path.join(seg_dir, "cluster_model.joblib"),
        cluster_scaler_path=os.path.join(seg_dir, "scaler.joblib"),
        segmentation_deterministic_code_path=os.path.join(seg_dir, "segmentation_deterministic.py"),
        cost_report_path=cost_report_path,
    )
