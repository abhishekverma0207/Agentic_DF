# crews/feature_crew.py
"""
State-of-the-Art Intelligent Feature Engineering Crew
======================================================

This crew implements the REASONING + EXECUTION separation pattern:

1. Feature Strategist Agent (AI REASONING):
   - REASONS about EDA insights, segmentation, data characteristics
   - Makes INTELLIGENT DECISIONS about feature engineering strategy
   - Formulates HYPOTHESES about what features will be important
   - Outputs structured FeatureStrategyDecision

2. Feature Executor (DETERMINISTIC - NO LLM):
   - Pure Python function that EXECUTES the AI's decisions
   - No LLM involvement - guaranteed reliable execution
   - Calls existing feature engineering utilities

3. Feature Quality Analyst Agent (AI REASONING):
   - REASONS about feature quality metrics
   - VALIDATES the strategist's hypotheses
   - Provides qualitative assessment and recommendations
   - Outputs structured FeatureQualityAssessment

KEY INSIGHT: Separate WHAT (AI reasoning) from HOW (deterministic execution).
- AI DECIDES: "Use ACF-informed lags [1,4,13,52] because seasonality is 78%"
- Code EXECUTES: Creates those specific lags reliably
- AI VALIDATES: "Hypothesis H1 confirmed - lag_52 has high importance"

This prevents the common failure mode where LLMs describe code instead of running it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from crewai import Agent, Crew, Task, Process, LLM

from config.schema import DemandForecastConfig
from utils.code_execution_tool import CodeExecutionTool
from utils.feature_reasoning import (
    FeatureStrategyDecision,
    FeatureQualityAssessment,
    FeatureComplexity,
    InteractionEmphasis,
    EncodingStrategy,
    FeatureHypothesis,
    create_reasoning_context,
    execute_feature_strategy,
    validate_features_and_assess_quality,
    load_eda_insights,
    load_segmentation_insights,
    STRATEGIST_REASONING_PROMPT,
    ANALYST_REASONING_PROMPT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EDA-AWARE FALLBACK DEFAULTS
# =============================================================================

def _create_eda_aware_fallback_strategy(
    eda_dir: str,
    seg_dir: str,
    time_format: str = 'year_week',
) -> FeatureStrategyDecision:
    """
    Create a STATE-OF-THE-ART fallback strategy that uses ACTUAL EDA + SEGMENTATION CONTEXT.

    This is called when the LLM strategist fails to produce valid output.
    Instead of using generic defaults (which may not fit the data), we extract
    key characteristics from the EDA and SEGMENTATION outputs.

    IMPORTANT: This now uses the comprehensive segmentation_to_feature_context.json
    which contains state-of-the-art feature recommendations per segment.

    Returns
    -------
    FeatureStrategyDecision
        A strategy based on actual EDA and segmentation insights
    """
    # Load actual EDA insights
    eda_insights = load_eda_insights(eda_dir)
    seg_insights = load_segmentation_insights(seg_dir)

    # ==========================================================================
    # LOAD STATE-OF-THE-ART SEGMENTATION CONTEXT
    # ==========================================================================
    seg_context_path = os.path.join(seg_dir, 'segmentation_to_feature_context.json')
    seg_feature_context = {}
    if os.path.exists(seg_context_path):
        with open(seg_context_path) as f:
            seg_feature_context = json.load(f)
        logger.info("Loaded state-of-the-art segmentation feature context")

    # Get global recommendations from segmentation
    global_recs = seg_feature_context.get('global_recommendations', {})
    segment_strategies = seg_feature_context.get('segment_feature_strategies', {})
    segment_profiles = seg_feature_context.get('segment_profiles', {})
    intermittency_segments = seg_feature_context.get('intermittency_segments', [])
    log_transform_segments = seg_feature_context.get('log_transform_segments', [])
    intermittency_summary = seg_feature_context.get('intermittency_summary', {})

    # ==========================================================================
    # ACF-INFORMED LAGS (from EDA + segmentation global recommendations)
    # ==========================================================================
    acf_data = eda_insights.get('autocorrelation', {})
    significant_lags = acf_data.get('significant_lags', [])

    # Also check segmentation's ACF-informed lags
    seg_acf_lags = global_recs.get('acf_informed_lags', [])
    if seg_acf_lags:
        significant_lags = list(set(significant_lags + seg_acf_lags))

    if significant_lags:
        # Use ACF-informed lags, ensuring we have key lags
        target_lags = sorted(set([1, 4] + [int(l) for l in significant_lags[:10]]))[:12]
        feature_lags = [l for l in [1, 4] if l in significant_lags or l <= 4]
        acf_informed = True
        acf_reasoning = f"ACF-informed lags from EDA+Segmentation: {significant_lags[:10]}"
    else:
        # Use segmentation global recommendation or defaults
        _default_lags = [1, 2, 3, 6, 12] if time_format == 'year_month' else [1, 4, 13, 26, 52]
        target_lags = global_recs.get('target_lags', _default_lags)
        feature_lags = [1, 4]
        acf_informed = False
        acf_reasoning = "Using segmentation default lags (no significant ACF lags found)"

    # ==========================================================================
    # SEASONALITY (from EDA + segmentation)
    # ==========================================================================
    seas_data = eda_insights.get('seasonality', {})
    has_seasonality_pct = seas_data.get('pct_with_seasonality', seas_data.get('has_seasonality_pct', 0.0))
    avg_seasonal_strength = seas_data.get('avg_seasonal_strength', 0.0)
    _default_period = 12 if time_format == 'year_month' else 52
    dominant_period = seas_data.get('dominant_period', _default_period)

    # Check segmentation's seasonality recommendation
    seg_seasonality_detected = global_recs.get('seasonality_detected', False)
    seg_seasonal_periods = global_recs.get('dominant_seasonal_periods', [])

    # Use strength-based threshold (from our updated EDA)
    has_strong_seasonality = (
        (has_seasonality_pct > 0.3 and avg_seasonal_strength > 0.1) or
        seg_seasonality_detected
    )
    seasonal_period = dominant_period if dominant_period and dominant_period > 0 else (
        seg_seasonal_periods[0] if seg_seasonal_periods else _default_period
    )
    fourier_order = 5 if has_strong_seasonality else 3
    seasonality_reasoning = (
        f"EDA: {has_seasonality_pct*100:.0f}% seasonal (strength={avg_seasonal_strength:.2f}), "
        f"period={seasonal_period}. Segmentation detected: {seg_seasonality_detected}"
    )

    # ==========================================================================
    # TREND (from EDA)
    # ==========================================================================
    trend_data = eda_insights.get('trend', {})
    trend_strength = trend_data.get('avg_trend_strength', 0.3)
    has_trend_pct = trend_data.get('pct_with_trend', 0.5)
    include_trend = has_trend_pct > 0.3
    trend_reasoning = f"EDA: {has_trend_pct*100:.0f}% with trend (avg strength={trend_strength:.2f})"

    # ==========================================================================
    # INTERMITTENCY (from segmentation - STATE-OF-THE-ART)
    # ==========================================================================
    # Use the segmentation's intermittency analysis which is more accurate
    n_intermittent_segments = intermittency_summary.get('n_intermittent_segments', 0)
    seg_intermittency_pct = intermittency_summary.get('intermittency_pct', 0.0)

    # Also check EDA intermittency
    interm_data = eda_insights.get('intermittency', {})
    eda_intermittency_pct = interm_data.get('pct_intermittent', 0.0) + interm_data.get('pct_lumpy', 0.0)

    # Use the higher estimate
    intermittency_pct = max(seg_intermittency_pct, eda_intermittency_pct)
    include_intermittency = intermittency_pct > 0.1 or len(intermittency_segments) > 0

    intermittency_reasoning = (
        f"Segmentation: {n_intermittent_segments} intermittent segments ({seg_intermittency_pct*100:.0f}% of keys). "
        f"EDA: {eda_intermittency_pct*100:.0f}% intermittent/lumpy. "
        f"Include intermittency features: {include_intermittency}"
    )

    # ==========================================================================
    # CHANGEPOINTS (from EDA - significance-filtered)
    # ==========================================================================
    cp_data = eda_insights.get('changepoints', {})
    pct_with_changepoints = cp_data.get('pct_with_significant_changepoints',
                                         cp_data.get('pct_with_changepoints', 0.0))
    include_changepoints = pct_with_changepoints > 0.2
    changepoint_reasoning = f"EDA: {pct_with_changepoints*100:.0f}% with significant changepoints"

    # ==========================================================================
    # ROLLING WINDOWS (from segmentation global recommendations)
    # ==========================================================================
    seg_rolling_windows = global_recs.get('rolling_windows', [])
    if seg_rolling_windows:
        rolling_windows = seg_rolling_windows
        rolling_reasoning = f"Using segmentation-recommended rolling windows: {rolling_windows}"
    elif has_strong_seasonality and seasonal_period:
        # Adapt rolling windows to the seasonal period
        rolling_windows = sorted(set([4, int(seasonal_period/4), int(seasonal_period/2), seasonal_period]))
        rolling_reasoning = f"Rolling windows adapted to seasonal period {seasonal_period}"
    else:
        rolling_windows = [3, 6, 12] if time_format == 'year_month' else [4, 13, 26, 52]
        rolling_reasoning = f"Standard {'monthly' if time_format == 'year_month' else 'weekly'} rolling windows"

    # Get rolling stats from segmentation or use defaults
    rolling_stats = global_recs.get('rolling_stats', ['mean', 'std', 'min', 'max'])

    # ==========================================================================
    # LOG TRANSFORM (from segmentation - per segment analysis)
    # ==========================================================================
    # If any segment needs log transform, enable it
    apply_log_transform = len(log_transform_segments) > 0
    transformation_reasoning = (
        f"{len(log_transform_segments)} segments recommend log transform"
        if apply_log_transform else "No segments need log transform"
    )

    # ==========================================================================
    # COMPLEXITY AND EMPHASIS (from segmentation analysis)
    # ==========================================================================
    n_segments = seg_insights.get('n_segments', 1)

    # Determine complexity based on data characteristics
    if n_segments > 5 or intermittency_pct > 0.3:
        complexity_level = FeatureComplexity.COMPREHENSIVE
    elif has_strong_seasonality or include_changepoints:
        complexity_level = FeatureComplexity.STANDARD
    else:
        complexity_level = FeatureComplexity.MINIMAL

    # Determine emphasis based on dominant pattern across segments.
    # Note: the seasonal branch uses `SEASONAL_HEAVY` (the actual enum
    # member name in utils/feature_reasoning.py:54).  An earlier version
    # of this file referenced a non-existent `SEASONALITY` member, which
    # raised AttributeError only on categories that satisfied BOTH
    # `intermittency_pct <= 0.3` AND `has_strong_seasonality` —
    # exactly the laundry / fabric categories.
    if intermittency_pct > 0.3 or n_intermittent_segments > n_segments / 2:
        primary_emphasis = InteractionEmphasis.INTERMITTENCY
    elif has_strong_seasonality:
        primary_emphasis = InteractionEmphasis.SEASONAL_HEAVY
    else:
        primary_emphasis = InteractionEmphasis.BALANCED

    logger.info(f"Created STATE-OF-THE-ART EDA+Segmentation fallback strategy:")
    logger.info(f"  - Lags: {target_lags} (ACF-informed: {acf_informed})")
    logger.info(f"  - Rolling: {rolling_windows} with {rolling_stats}")
    logger.info(f"  - Seasonality: {has_strong_seasonality} (period={seasonal_period})")
    logger.info(f"  - Intermittency: {include_intermittency} ({intermittency_pct*100:.0f}%, {n_intermittent_segments} segments)")
    logger.info(f"  - Changepoints: {include_changepoints} ({pct_with_changepoints*100:.0f}%)")
    logger.info(f"  - Log transform: {apply_log_transform} ({len(log_transform_segments)} segments)")
    logger.info(f"  - Complexity: {complexity_level.value}, Emphasis: {primary_emphasis.value}")

    return FeatureStrategyDecision(
        strategy_id=f"segmentation_aware_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        complexity_level=complexity_level,
        primary_emphasis=primary_emphasis,
        encoding_strategy=EncodingStrategy.TARGET_ENCODING,
        target_lags=target_lags,
        feature_lags=feature_lags,
        acf_informed=acf_informed,
        acf_reasoning=acf_reasoning,
        rolling_windows=rolling_windows,
        rolling_stats=rolling_stats,
        rolling_reasoning=rolling_reasoning,
        seasonal_period=seasonal_period,
        fourier_order=fourier_order,
        has_strong_seasonality=has_strong_seasonality,
        seasonality_reasoning=seasonality_reasoning,
        include_trend_features=include_trend,
        trend_strength=trend_strength,
        trend_reasoning=trend_reasoning,
        include_intermittency_features=include_intermittency,
        intermittency_pct=intermittency_pct,
        intermittency_reasoning=intermittency_reasoning,
        include_changepoint_indicators=include_changepoints,
        changepoint_dates=[],  # Dates would need to be loaded separately
        changepoint_reasoning=changepoint_reasoning,
        apply_log_transform=apply_log_transform,
        transformation_reasoning=transformation_reasoning,
    )


class FeatureEngineeringFailedError(Exception):
    """Raised when feature engineering fails to produce required outputs."""
    pass


# =============================================================================
# VALIDATION CALLBACKS
# =============================================================================

def _validate_strategist_output(feat_dir: str) -> None:
    """
    Validate that Strategist created feature_strategy_decision.json.
    Called as callback after Strategist completes.
    """
    strategy_path = os.path.join(feat_dir, 'feature_strategy_decision.json')
    if not os.path.exists(strategy_path):
        raise FeatureEngineeringFailedError(
            f"CRITICAL: Feature Strategist FAILED to create feature_strategy_decision.json\n\n"
            f"Expected file: {strategy_path}\n\n"
            "ROOT CAUSE: The LLM agent did NOT output valid JSON strategy.\n"
            "Check the strategist output for the JSON decision.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )

    # Validate JSON structure
    try:
        with open(strategy_path) as f:
            strategy = json.load(f)
        required_keys = ['target_lags', 'rolling_windows', 'acf_reasoning']
        missing = [k for k in required_keys if k not in strategy]
        if missing:
            raise FeatureEngineeringFailedError(
                f"Strategy JSON missing required keys: {missing}\n"
                f"Please ensure the AI outputs all required decision fields."
            )
    except json.JSONDecodeError as e:
        raise FeatureEngineeringFailedError(
            f"Strategy JSON is invalid: {e}\n"
            f"File: {strategy_path}"
        )

    logger.info(f"Feature Strategist validation passed: {strategy_path} exists and is valid")


def _validate_executor_output(feat_dir: str) -> None:
    """
    Validate that Executor created all required feature files WITH CORRECT CONTENT.
    Called as callback after Executor completes.
    """
    import pandas as pd
    from utils.feature_io import (
        features_intermediate_exists, read_features_intermediate,
    )

    # Format-agnostic existence check: parquet OR csv counts as "present".
    # Required base names + their fallback file labels (for clear errors).
    required_features = [
        ('train_features', 'Training features'),
        ('val_features', 'Validation features'),
        ('test_features', 'Test features'),
    ]
    required_json = [
        ('feature_metadata.json', 'Feature metadata'),
    ]

    missing = []
    for base_name, description in required_features:
        if not features_intermediate_exists(feat_dir, base_name):
            missing.append(f"  - {base_name}.[parquet|csv]: {description}")
    for filename, description in required_json:
        if not os.path.exists(os.path.join(feat_dir, filename)):
            missing.append(f"  - {filename}: {description}")

    if missing:
        raise FeatureEngineeringFailedError(
            f"CRITICAL: Feature Executor FAILED to create required files\n\n"
            f"Missing files in {feat_dir}:\n" +
            "\n".join(missing) + "\n\n"
            "ROOT CAUSE: The deterministic executor encountered an error.\n"
            "Check the execution logs for details."
        )

    # Validate train_features has sufficient features (peek nrows=1 — works
    # for both parquet and csv).
    try:
        train_peek = read_features_intermediate(feat_dir, 'train_features', nrows=1)
        n_cols = len(train_peek.columns)
        # Read full only if we need the row count.  Parquet can also
        # report row count via metadata but pandas hides that, so we
        # just full-read when needed.
        train_df = read_features_intermediate(feat_dir, 'train_features')
        n_rows = len(train_df)

        if n_cols < 10:
            raise FeatureEngineeringFailedError(
                f"CRITICAL: train_features has only {n_cols} columns!\n"
                f"Expected at least 10+ engineered features.\n"
                f"Columns: {list(train_df.columns)}\n\n"
                "Check feature engineering pipeline for errors."
            )

        if n_rows == 0:
            raise FeatureEngineeringFailedError(
                f"CRITICAL: train_features is empty (0 rows)!\n"
                "Check date ranges and data filtering."
            )

        logger.info(f"Feature Executor validation passed: {n_rows} rows, {n_cols} columns in train_features")

    except FeatureEngineeringFailedError:
        raise
    except Exception as e:
        logger.warning(f"Could not validate train_features content: {e}")

    logger.info(f"Feature Executor validation passed: all required files exist in {feat_dir}")


def _validate_analyst_output(feat_dir: str) -> None:
    """
    Validate that Analyst created the context files for Training crew.
    Called as callback after Analyst completes.
    """
    required_files = [
        ('training_manifest.csv', 'Model-level assignments'),
        ('feature_to_training_context.json', 'Context for Training crew'),
        ('feature_quality_assessment.json', 'Quality assessment'),
    ]

    missing = []
    for filename, description in required_files:
        filepath = os.path.join(feat_dir, filename)
        if not os.path.exists(filepath):
            missing.append(f"  - {filename}: {description}")

    if missing:
        raise FeatureEngineeringFailedError(
            f"CRITICAL: Feature Analyst FAILED to create context files\n\n"
            f"Missing files in {feat_dir}:\n" +
            "\n".join(missing) + "\n\n"
            "ROOT CAUSE: The LLM agent did NOT output valid assessment.\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )
    logger.info(f"Feature Analyst validation passed: all context files exist in {feat_dir}")


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class FeatureCrewResult:
    """Container with the main feature engineering artifact paths."""
    feature_dir: str
    feature_metadata_path: str
    feature_quality_summary_path: str
    feature_report_markdown_path: str
    segmentation_integration_used: bool = True
    feature_pipeline_script_path: str = ""
    feature_to_training_context_path: str = ""
    feature_deterministic_code_path: str = ""
    cost_report_path: str = ""
    strategy_decision_path: str = ""
    quality_assessment_path: str = ""


# =============================================================================
# INTELLIGENT AGENT CREATION
# =============================================================================

def _create_feature_strategist_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Feature Strategist agent - AI REASONING to make decisions.

    This agent REASONS about data and makes INTELLIGENT DECISIONS.
    It does NOT execute code - it outputs a structured strategy.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="feature_strategist",
        role="Intelligent Feature Engineering Strategist",
        goal=(
            "REASON about EDA and Segmentation insights to make INTELLIGENT DECISIONS "
            "about feature engineering strategy. Output a structured JSON strategy."
        ),
        backstory=(
            "You are a PhD-level expert in time series feature engineering.\n\n"
            "## YOUR JOB: REASON and DECIDE (not execute code)\n\n"
            "You will analyze:\n"
            "1. Autocorrelation analysis (ACF/PACF) - which lags are significant?\n"
            "2. Seasonality detection (FFT) - what seasonal period? how strong?\n"
            "3. Trend analysis - what % of series have strong trends?\n"
            "4. Changepoint detection - structural breaks in the data?\n"
            "5. Demand patterns - intermittent, lumpy, smooth, erratic?\n\n"
            "Based on this analysis, you DECIDE:\n"
            "- What lags to use (justify with ACF analysis)\n"
            "- What rolling windows (justify with demand patterns)\n"
            "- Whether to include intermittency features (justify with intermittency %)\n"
            "- What seasonal features (justify with FFT analysis)\n"
            "- Form HYPOTHESES about what features will be important\n\n"
            "## CRITICAL: Output ONLY valid JSON\n"
            "Your entire response must be a valid JSON object.\n"
            "Do NOT include any text before or after the JSON.\n\n"
            + STRATEGIST_REASONING_PROMPT
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_feature_analyst_agent(llm: LLM, allowed_model_families: list, enable_deep_models: bool, protected_paths: list = None) -> Agent:
    """
    Create the Feature Quality Analyst agent - AI REASONING to validate.

    This agent REASONS about feature quality and validates hypotheses.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="feature_analyst",
        role="PhD-Level Feature Quality Analyst",
        goal=(
            "REASON about feature quality metrics and VALIDATE the strategist's hypotheses. "
            "Assess overall quality and provide recommendations."
        ),
        backstory=(
            "You are a PhD-level expert in feature quality assessment.\n\n"
            "## YOUR JOB: REASON and VALIDATE (not just report metrics)\n\n"
            "You will analyze:\n"
            "1. Feature counts by type (lag, rolling, seasonal, intermittency)\n"
            "2. Data quality (missing values, infinite values, constant features)\n"
            "3. Feature redundancy (highly correlated pairs)\n"
            "4. The strategist's hypotheses - were they confirmed?\n\n"
            "Based on this analysis, you ASSESS:\n"
            "- Overall feature quality (excellent/good/acceptable/poor) with reasoning\n"
            "- Which hypotheses were confirmed and which rejected\n"
            "- Specific concerns with reasoning\n"
            "- Specific recommendations for improvement\n\n"
            "## CRITICAL: Output ONLY valid JSON\n"
            "Your entire response must be a valid JSON object.\n"
            "Do NOT include any text before or after the JSON.\n\n"
            f"## ALLOWED MODELS: {allowed_model_families}\n"
            f"## DEEP MODELS ENABLED: {enable_deep_models}\n\n"
            + ANALYST_REASONING_PROMPT
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_feature_documentation_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Feature Engineering Documentation Agent.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="feature_documentation_agent",
        role="Feature Engineering Insights Documentation Specialist",
        goal=(
            "EXECUTE Python code using CodeExecutionTool to create FEATURE_ENGINEERING_INSIGHTS_GUIDE.md. "
            "You MUST use the tool to run the code - do NOT just describe or print the markdown."
        ),
        backstory=(
            "You are an expert at explaining feature engineering decisions.\n\n"
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE CodeExecutionTool TO RUN THE CODE BLOCK   #\n"
            "#  DO NOT just print or describe the markdown - SAVE IT TO FILE!    #\n"
            "######################################################################\n\n"
            "## HOW TO COMPLETE THIS TASK:\n"
            "1. Use CodeExecutionTool to execute the Python code in the task\n"
            "2. The code will SAVE the markdown to FEATURE_ENGINEERING_INSIGHTS_GUIDE.md\n"
            "3. Do NOT print the entire markdown content\n"
            "4. Only print confirmation messages (max 5 lines)\n\n"
            "## WRONG APPROACH (DO NOT DO THIS):\n"
            "- Printing the markdown content to stdout\n"
            "- Describing what the code would do\n"
            "- Showing the markdown without saving\n\n"
            "## CORRECT APPROACH:\n"
            "- Execute the code block using CodeExecutionTool\n"
            "- Code saves to file with: open(doc_path, 'w').write(md)\n"
            "- Print only: 'Saved: FEATURE_ENGINEERING_INSIGHTS_GUIDE.md'"
        ),
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )


def _create_feature_insights_agent(llm: LLM, protected_paths: list = None) -> Agent:
    """
    Create the Feature Engineering Insights Agent.

    This agent uses CodeExecutionTool to ITERATIVELY explore feature engineering outputs
    and creates a comprehensive insights markdown report through multiple phases.
    """
    code_tool = CodeExecutionTool(protected_paths=protected_paths or [])

    return Agent(
        name="feature_insights_agent",
        role="PhD-Level Feature Engineering Insights Documentation Specialist",
        goal=(
            "EXECUTE Python code using CodeExecutionTool to explore feature engineering outputs "
            "and create FEATURE_ENGINEERING_INSIGHTS_REPORT.md. You MUST use the tool to run code - "
            "do NOT just describe or print the analysis."
        ),
        backstory=(
            "You are a PhD-level expert in feature engineering for demand forecasting.\n\n"
            "######################################################################\n"
            "#  CRITICAL: YOU MUST USE CodeExecutionTool TO RUN ANALYSIS CODE    #\n"
            "#  EXPLORE DATA ITERATIVELY - RUN MULTIPLE CODE EXECUTIONS          #\n"
            "#  SAVE FINAL REPORT TO FILE - DO NOT JUST PRINT IT                 #\n"
            "######################################################################\n\n"
            "## YOUR EXPERTISE:\n"
            "- Lag feature engineering based on ACF analysis\n"
            "- Rolling window statistics for demand patterns\n"
            "- Seasonal features (Fourier, calendar components)\n"
            "- External features (price, promotion, holiday effects)\n"
            "- Model-level allocation (individual vs segment-pooled)\n\n"
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


class FeatureInsightsError(Exception):
    """Raised when feature insights generation fails."""
    pass


def _validate_feature_insights_report(feat_dir: str) -> None:
    """
    Validate that insights report was created with sufficient content.

    Raises:
        FeatureInsightsError: If report doesn't exist or is insufficient
    """
    report_path = os.path.join(feat_dir, 'FEATURE_ENGINEERING_INSIGHTS_REPORT.md')

    if not os.path.exists(report_path):
        raise FeatureInsightsError(
            f"CRITICAL: Feature Insights Agent FAILED to create report\n\n"
            f"Expected file: {report_path}\n\n"
            "ROOT CAUSE: The agent did NOT execute the code to save the file.\n"
            "The agent likely described the analysis instead of running CodeExecutionTool.\n\n"
            "This is a known issue with CrewAI agents. Re-running may help."
        )

    # Check content quality
    with open(report_path) as f:
        content = f.read()

    if len(content) < 3000:  # Minimum ~75 lines for comprehensive report
        raise FeatureInsightsError(
            f"CRITICAL: Feature Insights Report is too short ({len(content)} chars)\n\n"
            "The agent likely did not complete the full analysis.\n"
            "Expected at least 3000 characters of insights."
        )

    # Check for placeholder text
    placeholders = ['[Write', '[Use', '[Based on', '[Recommend', '[Insert', '[Add']
    placeholder_count = sum(1 for p in placeholders if p in content)
    if placeholder_count > 3:
        raise FeatureInsightsError(
            f"CRITICAL: Feature Insights Report contains {placeholder_count} placeholders\n\n"
            "The agent did not fill in actual statistics from the data.\n"
            "The report should contain specific numbers, not placeholder text."
        )

    logger.info(f"Feature insights report validation passed: {len(content):,} chars")


def _run_feature_insights_crew(
    feat_dir: str,
    seg_dir: str,
    eda_dir: str,
    config: DemandForecastConfig,
    llm: LLM,
) -> None:
    """
    Run Feature Engineering Insights Documentation crew.

    Creates FEATURE_ENGINEERING_INSIGHTS_REPORT.md through iterative code execution.
    The agent explores feature files using CodeExecutionTool and builds
    a comprehensive insights report.

    Raises:
        FeatureInsightsError: If report generation fails
    """
    logger.info("="*70)
    logger.info("RUNNING FEATURE ENGINEERING INSIGHTS CREW")
    logger.info("="*70)

    # Create the insights agent
    insights_agent = _create_feature_insights_agent(llm)

    # Build the task description with file paths
    task_description = f"""
# FEATURE ENGINEERING INSIGHTS REPORT GENERATION

Analyze feature engineering outputs in `{feat_dir}` and create a comprehensive insights report.

######################################################################
#  CRITICAL: DO NOT LOAD train_features.csv - IT MAY BE VERY LARGE!  #
#  USE ONLY JSON METADATA FILES AND SMALL CSVs FOR ANALYSIS          #
######################################################################

## AVAILABLE FILES (USE THESE - SMALL FILES)

### Feature Metadata (USE THESE - JSON files are small)
- `{feat_dir}/feature_metadata.json` - Feature column names and types (PREFERRED)
- `{feat_dir}/feature_strategy_decision.json` - AI strategy decisions
- `{feat_dir}/feature_quality_metrics.json` - Quality metrics with counts
- `{feat_dir}/feature_quality_assessment.json` - AI quality assessment
- `{feat_dir}/feature_to_training_context.json` - Context for training crew

### Small CSV Files (SAFE TO LOAD)
- `{feat_dir}/training_manifest.csv` - Model-level assignments (one row per key)
- `{seg_dir}/per_key_with_segments.csv` - Per-key metrics (one row per key)

### Segmentation Context (JSON - SAFE)
- `{seg_dir}/segment_profiles.json` - Segment characteristics
- `{seg_dir}/model_recommendations.json` - Model recommendations by segment

### EDA Context (JSON - SAFE)
- `{eda_dir}/eda_summary.json` - Overall data characteristics

### LARGE FILES - DO NOT LOAD FULLY
- `{feat_dir}/train_features.csv` - LARGE FILE! Use only if absolutely needed with nrows limit
- `{feat_dir}/val_features.csv` - LARGE FILE! Do not load
- `{feat_dir}/test_features.csv` - LARGE FILE! Do not load

## CONFIG INFO

- Time format: {config.time_format}
- Key columns: {list(config.key_columns)}
- Target: {config.target_column}
- Forecast horizon: {config.forecast_horizon}

## STRATEGY: Use metadata files to get all statistics - they contain pre-computed counts!

## PHASE 1: EXPLORE FEATURE STRUCTURE (FROM METADATA - NO LARGE FILE LOADING)

Execute this code to understand the feature engineering outputs using ONLY metadata files:

```python
import json
import os
import warnings
warnings.filterwarnings('ignore')

feat_dir = '{feat_dir}'

# List available files and their sizes
print('=== FEATURE ENGINEERING FILES ===')
for f in sorted(os.listdir(feat_dir)):
    if f.endswith(('.csv', '.json', '.md')):
        path = os.path.join(feat_dir, f)
        size_mb = os.path.getsize(path) / (1024*1024)
        size_str = f'{{size_mb:.1f}} MB' if size_mb > 1 else f'{{os.path.getsize(path)/1024:.0f}} KB'
        print(f'  {{f}}: {{size_str}}')

# Load feature metadata (SMALL JSON - contains all column names)
meta_path = os.path.join(feat_dir, 'feature_metadata.json')
with open(meta_path) as f:
    meta = json.load(f)

feature_cols = meta.get('feature_cols', [])
print(f'\\n=== FEATURE COUNTS FROM METADATA ===')
print(f'Total features: {{len(feature_cols)}}')

# Categorize features by analyzing column names (NO DATA LOADING)
lag_feats = [c for c in feature_cols if '_lag_' in c.lower() or c.endswith('_lag')]
roll_feats = [c for c in feature_cols if 'rolling_' in c.lower() or '_ma_' in c.lower() or '_std_' in c.lower()]
seasonal_feats = [c for c in feature_cols if 'fourier_' in c.lower() or 'seasonal_' in c.lower() or 'sin_' in c.lower() or 'cos_' in c.lower()]
calendar_feats = [c for c in feature_cols if any(x in c.lower() for x in ['week_of_year', 'month', 'quarter', 'day_of'])]
intermittency_feats = [c for c in feature_cols if any(x in c.lower() for x in ['demand_occurred', 'periods_since', 'zero_', 'nonzero'])]
external_feats = [c for c in feature_cols if any(x in c.lower() for x in ['price', 'promo', 'discount', 'holiday', 'weather'])]

print(f'Lag features: {{len(lag_feats)}}')
print(f'Rolling features: {{len(roll_feats)}}')
print(f'Seasonal/Fourier features: {{len(seasonal_feats)}}')
print(f'Calendar features: {{len(calendar_feats)}}')
print(f'Intermittency features: {{len(intermittency_feats)}}')
print(f'External features: {{len(external_feats)}}')

# Also load quality metrics (pre-computed counts)
quality_path = os.path.join(feat_dir, 'feature_quality_metrics.json')
if os.path.exists(quality_path):
    with open(quality_path) as f:
        quality = json.load(f)
    print(f'\\n=== PRE-COMPUTED QUALITY METRICS ===')
    print(f'Total features (from quality): {{quality.get("total_features", "N/A")}}')
    print(f'Missing %: {{quality.get("missing_pct", "N/A")}}')
    print(f'Constant features: {{len(quality.get("constant_features", []))}}')
```

## PHASE 2: TRAINING MANIFEST ANALYSIS

Execute this code to analyze model-level allocation decisions:

```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

feat_dir = '{feat_dir}'

# Load training manifest
manifest = pd.read_csv(f'{{feat_dir}}/training_manifest.csv')
print(f'=== TRAINING MANIFEST: {{len(manifest)}} keys ===')
print(f'Columns: {{list(manifest.columns)}}')

# Model strategy distribution
print('\\n=== MODEL STRATEGY DISTRIBUTION ===')
if 'model_strategy' in manifest.columns:
    strat_counts = manifest['model_strategy'].value_counts()
    for s, c in strat_counts.items():
        print(f'{{s}}: {{c}} keys ({{c/len(manifest)*100:.1f}}%)')

# Individual vs Segment-pooled
individual_mask = manifest['model_level'] == manifest['key']
n_individual = individual_mask.sum()
n_pooled = len(manifest) - n_individual
print(f'\\n=== ALLOCATION SUMMARY ===')
print(f'Individual models: {{n_individual}} ({{n_individual/len(manifest)*100:.1f}}%)')
print(f'Segment-pooled: {{n_pooled}} ({{n_pooled/len(manifest)*100:.1f}}%)')

# By demand pattern
if 'demand_pattern' in manifest.columns:
    print('\\n=== ALLOCATION BY DEMAND PATTERN ===')
    for pattern in manifest['demand_pattern'].unique():
        pattern_df = manifest[manifest['demand_pattern'] == pattern]
        ind = (pattern_df['model_level'] == pattern_df['key']).sum()
        print(f'{{pattern}}: {{len(pattern_df)}} total, {{ind}} individual ({{ind/len(pattern_df)*100:.1f}}%)')
```

## PHASE 3: EXTERNAL FEATURE ANALYSIS

Execute this code to analyze external feature usage:

```python
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

feat_dir = '{feat_dir}'

# Load feature metadata
with open(f'{{feat_dir}}/feature_metadata.json') as f:
    meta = json.load(f)

feature_cols = meta.get('feature_cols', [])

# Categorize external features
price_feats = [c for c in feature_cols if 'price' in c.lower()]
promo_feats = [c for c in feature_cols if 'promo' in c.lower() or 'discount' in c.lower()]
holiday_feats = [c for c in feature_cols if 'holiday' in c.lower()]

print('=== EXTERNAL FEATURES CREATED ===')
print(f'Price features: {{len(price_feats)}}')
if price_feats[:5]: print(f'  Examples: {{price_feats[:5]}}')

print(f'\\nPromo features: {{len(promo_feats)}}')
if promo_feats[:5]: print(f'  Examples: {{promo_feats[:5]}}')

print(f'\\nHoliday features: {{len(holiday_feats)}}')
if holiday_feats[:3]: print(f'  Examples: {{holiday_feats[:3]}}')

# Check for lagged versions
price_lags = [c for c in price_feats if 'lag' in c.lower()]
promo_lags = [c for c in promo_feats if 'lag' in c.lower()]
print(f'\\n=== LAGGED EXTERNAL FEATURES ===')
print(f'Price lags created: {{len(price_lags)}}')
print(f'Promo lags created: {{len(promo_lags)}}')
```

## PHASE 4: SEGMENT-LEVEL FEATURE ANALYSIS

Execute this code to analyze features by segment:

```python
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

seg_dir = '{seg_dir}'

# Load segment profiles
profiles_path = f'{{seg_dir}}/segment_profiles.json'
with open(profiles_path) as f:
    profiles = json.load(f)

print('=== SEGMENT PROFILES ===')
for seg_id, profile in profiles.items():
    n_keys = profile.get('n_keys', 0)
    dominant_pattern = profile.get('dominant_demand_pattern', 'unknown')
    cv_mean = profile.get('cv_mean', 0)
    zero_frac = profile.get('zero_fraction_mean', 0)
    print(f'\\nSegment {{seg_id}}: {{n_keys}} keys')
    print(f'  Dominant pattern: {{dominant_pattern}}')
    print(f'  Avg CV: {{cv_mean:.3f}}, Avg zero fraction: {{zero_frac:.3f}}')

# Load model recommendations
recs_path = f'{{seg_dir}}/model_recommendations.json'
with open(recs_path) as f:
    recs = json.load(f)

print('\\n=== SEGMENT MODEL RECOMMENDATIONS ===')
for seg_id, rec in recs.items():
    models = rec.get('recommended_models', [])[:3]
    print(f'Segment {{seg_id}}: {{models}}')
```

## PHASE 5: AI STRATEGY AND HYPOTHESIS ANALYSIS

Execute this code to analyze AI reasoning:

```python
import json
import warnings
warnings.filterwarnings('ignore')

feat_dir = '{feat_dir}'

# Load strategy decision
strategy_path = f'{{feat_dir}}/feature_strategy_decision.json'
with open(strategy_path) as f:
    strategy = json.load(f)

print('=== AI FEATURE STRATEGY ===')
print(f'Complexity: {{strategy.get("complexity_level", "N/A")}}')
print(f'Primary emphasis: {{strategy.get("primary_emphasis", "N/A")}}')
print(f'Target lags: {{strategy.get("target_lags", [])}}')
print(f'Feature lags (external): {{strategy.get("feature_lags", [])}}')
print(f'Rolling windows: {{strategy.get("rolling_windows", [])}}')
print(f'Seasonal period: {{strategy.get("seasonal_period", "N/A")}}')

print('\\n=== ACF REASONING ===')
print(strategy.get('acf_reasoning', 'N/A')[:300])

print('\\n=== HYPOTHESES ===')
hypotheses = strategy.get('hypotheses', [])
for h in hypotheses[:5]:
    print(f"{{h.get('hypothesis_id', '?')}}: {{h.get('description', 'N/A')[:80]}}...")

# Load quality assessment
assess_path = f'{{feat_dir}}/feature_quality_assessment.json'
with open(assess_path) as f:
    assessment = json.load(f)

print('\\n=== QUALITY ASSESSMENT ===')
print(f'Overall quality: {{assessment.get("overall_quality", "N/A")}}')
print(f'Confirmed hypotheses: {{assessment.get("hypotheses_confirmed", [])}}')
print(f'Rejected hypotheses: {{assessment.get("hypotheses_rejected", [])}}')
```

## PHASE 6: GENERATE AND SAVE COMPREHENSIVE REPORT

After gathering all insights from the previous phases, execute code to CREATE and SAVE the report.
The code below PROGRAMMATICALLY builds the markdown using data loaded from files - NO PLACEHOLDERS.

```python
from datetime import datetime
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

feat_dir = '{feat_dir}'
seg_dir = '{seg_dir}'

# Re-load all data to build comprehensive report
with open(f'{{feat_dir}}/feature_metadata.json') as f:
    meta = json.load(f)

manifest = pd.read_csv(f'{{feat_dir}}/training_manifest.csv')

with open(f'{{feat_dir}}/feature_strategy_decision.json') as f:
    strategy = json.load(f)

with open(f'{{seg_dir}}/segment_profiles.json') as f:
    profiles = json.load(f)

# Categorize features by analyzing column names
feature_cols = meta.get('feature_cols', [])
lag_feats = [c for c in feature_cols if '_lag_' in c.lower() or c.endswith('_lag')]
roll_feats = [c for c in feature_cols if 'rolling_' in c.lower() or '_ma_' in c.lower() or '_std_' in c.lower()]
seasonal_feats = [c for c in feature_cols if 'fourier_' in c.lower() or 'seasonal_' in c.lower() or 'sin_' in c.lower() or 'cos_' in c.lower()]
calendar_feats = [c for c in feature_cols if any(x in c.lower() for x in ['week_of_year', 'month', 'quarter', 'day_of'])]
intermittency_feats = [c for c in feature_cols if any(x in c.lower() for x in ['demand_occurred', 'periods_since', 'zero_', 'nonzero'])]
price_feats = [c for c in feature_cols if 'price' in c.lower()]
promo_feats = [c for c in feature_cols if 'promo' in c.lower() or 'discount' in c.lower()]
holiday_feats = [c for c in feature_cols if 'holiday' in c.lower()]
external_feats = price_feats + promo_feats + holiday_feats

# Training manifest analysis
n_keys = len(manifest)
if 'model_strategy' in manifest.columns:
    strat_counts = manifest['model_strategy'].value_counts().to_dict()
else:
    strat_counts = {{}}

# Individual vs pooled
if 'model_level' in manifest.columns and 'key' in manifest.columns:
    individual_mask = manifest['model_level'] == manifest['key']
    n_individual = individual_mask.sum()
    n_pooled = n_keys - n_individual
else:
    n_individual = 0
    n_pooled = n_keys

# Strategy details
complexity = strategy.get('complexity_level', 'N/A')
emphasis = strategy.get('primary_emphasis', 'N/A')
target_lags = strategy.get('target_lags', [])
feature_lags = strategy.get('feature_lags', [])
rolling_windows = strategy.get('rolling_windows', [])
acf_reasoning = strategy.get('acf_reasoning', 'N/A')

# Build comprehensive markdown report
md = f'''# Feature Engineering Insights Report

**Generated:** {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}

---

## Executive Summary

This report provides comprehensive analysis of the feature engineering process for demand forecasting.
A total of **{{len(feature_cols)}} features** were engineered across multiple categories including
lag features, rolling statistics, seasonal components, and external variables.

The model-level allocation assigned **{{n_individual}} keys ({{n_individual/n_keys*100:.1f}}%)** to individual models
and **{{n_pooled}} keys ({{n_pooled/n_keys*100:.1f}}%)** to segment-pooled models.

The AI strategy emphasized **{{emphasis}}** with **{{complexity}}** complexity level.

---

## 1. Feature Engineering Overview

### Features Created

| Category | Count | Description |
|----------|-------|-------------|
| Lag Features | {{len(lag_feats)}} | Historical target values |
| Rolling Features | {{len(roll_feats)}} | Moving averages and statistics |
| Seasonal/Fourier | {{len(seasonal_feats)}} | Fourier and seasonal components |
| Calendar Features | {{len(calendar_feats)}} | Week, month, quarter indicators |
| Intermittency Features | {{len(intermittency_feats)}} | Zero demand indicators |
| External Features | {{len(external_feats)}} | Price, promo, holiday effects |
| **Total** | **{{len(feature_cols)}}** | All engineered features |

### AI Strategy Decision

- **Complexity Level:** {{complexity}}
- **Primary Emphasis:** {{emphasis}}
- **Target Lags Selected:** {{target_lags}}
- **Feature Lags (External):** {{feature_lags}}
- **Rolling Windows:** {{rolling_windows}}

### ACF Reasoning

{{acf_reasoning[:500]}}

---

## 2. Training Manifest Analysis

### Model-Level Allocation

| Strategy | Count | Percentage |
|----------|-------|------------|'''

for strat, count in strat_counts.items():
    pct = count / n_keys * 100
    md += f'''
| {{strat}} | {{count}} | {{pct:.1f}}% |'''

md += f'''

### Summary

- **Total Keys:** {{n_keys:,}}
- **Individual Models:** {{n_individual}} ({{n_individual/n_keys*100:.1f}}%)
- **Segment Pooled:** {{n_pooled}} ({{n_pooled/n_keys*100:.1f}}%)

'''

# Add allocation by demand pattern if available
if 'demand_pattern' in manifest.columns:
    md += '''### Allocation by Demand Pattern

| Pattern | Total Keys | Individual | % Individual |
|---------|------------|------------|--------------|'''
    for pattern in manifest['demand_pattern'].unique():
        pattern_df = manifest[manifest['demand_pattern'] == pattern]
        if 'model_level' in manifest.columns and 'key' in manifest.columns:
            ind = (pattern_df['model_level'] == pattern_df['key']).sum()
        else:
            ind = 0
        md += f'''
| {{pattern.title()}} | {{len(pattern_df)}} | {{ind}} | {{ind/len(pattern_df)*100:.1f}}% |'''
    md += '''

'''

md += f'''---

## 3. External Feature Analysis

### Price Features ({{len(price_feats)}})

'''
if price_feats:
    for f in price_feats[:10]:
        md += f'- `{{f}}`\\n'
    if len(price_feats) > 10:
        md += f'... and {{len(price_feats) - 10}} more\\n'
else:
    md += 'No price features created.\\n'

md += f'''
### Promotion Features ({{len(promo_feats)}})

'''
if promo_feats:
    for f in promo_feats[:10]:
        md += f'- `{{f}}`\\n'
    if len(promo_feats) > 10:
        md += f'... and {{len(promo_feats) - 10}} more\\n'
else:
    md += 'No promotion features created.\\n'

md += f'''
### Holiday Features ({{len(holiday_feats)}})

'''
if holiday_feats:
    for f in holiday_feats[:5]:
        md += f'- `{{f}}`\\n'
else:
    md += 'No holiday features created.\\n'

md += f'''
---

## 4. Segment-Specific Analysis

'''

for seg_id, profile in sorted(profiles.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
    n_keys_seg = profile.get('n_keys', profile.get('size', 0))
    dominant_pattern = profile.get('dominant_demand_pattern', profile.get('dominant_pattern', 'unknown'))
    cv_mean = profile.get('cv_mean', 0)
    zero_frac = profile.get('zero_fraction_mean', 0)

    md += f'''### Segment {{seg_id}}: {{dominant_pattern.title()}}

- **Keys:** {{n_keys_seg:,}}
- **Dominant Pattern:** {{dominant_pattern.title()}}
- **Avg CV:** {{cv_mean:.3f}}
- **Avg Zero Fraction:** {{zero_frac:.3f}}

'''

md += f'''---

## 5. Recommendations for Training

### High Priority

1. **Individual models** should be used for keys with unique patterns and sufficient data
2. **Segment-pooled models** work well for keys with similar characteristics within segments

### By Demand Pattern

- **Smooth:** Standard models with MSE loss
- **Erratic:** Models with robust loss functions (Huber, Quantile)
- **Intermittent/Lumpy:** Tweedie loss, zero-inflated models

### External Feature Usage

- Price features: Use lagged versions to capture price elasticity effects
- Promo features: Include lag-0 and lag-1 for immediate and delayed effects
- Holiday features: Calendar indicators help capture seasonal patterns

---

*Generated by Feature Engineering Insights Agent using iterative code execution.*
'''

# SAVE THE REPORT TO FILE
report_path = os.path.join(feat_dir, 'FEATURE_ENGINEERING_INSIGHTS_REPORT.md')
with open(report_path, 'w') as f:
    f.write(md)

print(f'Saved: FEATURE_ENGINEERING_INSIGHTS_REPORT.md ({{len(md):,}} bytes)')
print(f'Total features: {{len(feature_cols)}}')
print(f'Individual models: {{n_individual}}, Pooled: {{n_pooled}}')
```

## CRITICAL INSTRUCTIONS

1. You MUST execute code using CodeExecutionTool
2. Run Phases 1-5 to gather statistics BEFORE Phase 6
3. In Phase 6, the code PROGRAMMATICALLY builds markdown using loaded data - no placeholders
4. The report MUST be saved to file (not just printed)
5. Final report should be comprehensive with specific data
"""

    # Create the task
    insights_task = Task(
        name="generate_feature_insights",
        description=task_description,
        agent=insights_agent,
        expected_output=(
            "Created FEATURE_ENGINEERING_INSIGHTS_REPORT.md - comprehensive documentation with "
            "feature analysis, training manifest insights, external feature usage, and "
            "segment-specific recommendations. All statistics are real numbers from data analysis."
        ),
    )

    # Create and run the crew
    crew = Crew(
        name="Feature Engineering Insights Crew",
        agents=[insights_agent],
        tasks=[insights_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    logger.info(f"Feature insights crew completed: {result}")

    # Validate the report was created
    _validate_feature_insights_report(feat_dir)

    logger.info("Feature Engineering Insights Report validated successfully")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_output_path(absolute_path: str) -> str:
    """Get a safe path for CrewAI Task output_file parameter."""
    return os.path.abspath(absolute_path)


def _parse_strategist_json_output(output_text: str, feat_dir: str, time_format: str = 'year_week') -> FeatureStrategyDecision:
    """
    Parse the Strategist's JSON output and create FeatureStrategyDecision.

    Handles various output formats from the LLM.
    """
    import re

    # Try to extract JSON from the output
    json_match = re.search(r'\{[\s\S]*\}', output_text)
    if json_match:
        json_str = json_match.group()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            data = json.loads(json_str)
    else:
        # Check if strategy file was already created
        strategy_path = os.path.join(feat_dir, 'feature_strategy_decision.json')
        if os.path.exists(strategy_path):
            with open(strategy_path) as f:
                data = json.load(f)
        else:
            raise ValueError("Could not find JSON in strategist output")

    # Build FeatureStrategyDecision from parsed data
    strategy = FeatureStrategyDecision(
        strategy_id=data.get('strategy_id', f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        timestamp=datetime.now().isoformat(),
        source="AI_REASONING",
        complexity_level=FeatureComplexity(data.get('complexity_level', 'standard')),
        primary_emphasis=InteractionEmphasis(data.get('primary_emphasis', 'balanced')),
        encoding_strategy=EncodingStrategy(data.get('encoding_strategy', 'target_encoding')),
        target_lags=data.get('target_lags', [1, 2, 3, 6, 12] if time_format == 'year_month' else [1, 4, 13, 26, 52]),
        feature_lags=data.get('feature_lags', [1, 2] if time_format == 'year_month' else [1, 4]),
        acf_informed=data.get('acf_informed', False),
        acf_reasoning=data.get('acf_reasoning', 'Default lags used'),
        rolling_windows=data.get('rolling_windows', [3, 6, 12] if time_format == 'year_month' else [4, 13, 26, 52]),
        rolling_stats=data.get('rolling_stats', ['mean', 'std', 'min', 'max']),
        rolling_reasoning=data.get('rolling_reasoning', 'Default windows used'),
        seasonal_period=data.get('seasonal_period', 12 if time_format == 'year_month' else 52),
        fourier_order=data.get('fourier_order', 3),
        has_strong_seasonality=data.get('has_strong_seasonality', True),
        seasonality_reasoning=data.get('seasonality_reasoning', 'Default seasonality assumed'),
        include_trend_features=data.get('include_trend_features', True),
        trend_strength=data.get('trend_strength', 0.3),
        trend_reasoning=data.get('trend_reasoning', 'Default trend features included'),
        include_intermittency_features=data.get('include_intermittency_features', True),
        intermittency_pct=data.get('intermittency_pct', 0.3),
        intermittency_reasoning=data.get('intermittency_reasoning', 'Default intermittency features included'),
        include_changepoint_indicators=data.get('include_changepoint_indicators', False),
        changepoint_dates=data.get('changepoint_dates', []),
        changepoint_reasoning=data.get('changepoint_reasoning', 'No changepoints detected'),
        apply_log_transform=data.get('apply_log_transform', False),
        apply_differencing=data.get('apply_differencing', False),
        transformation_reasoning=data.get('transformation_reasoning', 'No transformations applied'),
    )

    # Parse hypotheses
    if 'hypotheses' in data:
        for h in data['hypotheses']:
            strategy.hypotheses.append(FeatureHypothesis(
                hypothesis_id=h.get('hypothesis_id', 'H0'),
                description=h.get('description', ''),
                feature_types=h.get('feature_types', []),
                expected_importance=h.get('expected_importance', 'medium'),
                reasoning=h.get('reasoning', ''),
                data_evidence=h.get('data_evidence', {}),
            ))

    return strategy


def _run_deterministic_executor(
    config: DemandForecastConfig,
    strategy: FeatureStrategyDecision,
    feat_dir: str,
    seg_dir: str,
    eda_dir: str,
) -> Dict[str, Any]:
    """
    Run the DETERMINISTIC feature executor (NO LLM).

    This is pure Python that reliably executes the AI's strategy.
    """
    from utils.agent_utilities import load_csv

    logger.info("=" * 60)
    logger.info("DETERMINISTIC FEATURE EXECUTOR (No LLM)")
    logger.info("=" * 60)
    logger.info(f"Executing strategy: {strategy.strategy_id}")
    logger.info(f"  Complexity: {strategy.complexity_level.value}")
    logger.info(f"  Emphasis: {strategy.primary_emphasis.value}")
    logger.info(f"  Target lags: {strategy.target_lags}")
    logger.info(f"  External feature lags: {strategy.feature_lags}")
    logger.info(f"  Rolling windows: {strategy.rolling_windows}")
    logger.info(f"  Intermittency features: {strategy.include_intermittency_features}")

    # Load data
    data_path = config.input_data_path
    df = load_csv(data_path)

    # Load segment assignments
    seg_df = load_csv(os.path.join(seg_dir, 'per_key_with_segments.csv'))
    key_cols = config.key_columns
    key_col = key_cols[0] if len(key_cols) == 1 else key_cols

    # Filter dead keys from EDA
    dead_keys_path = os.path.join(eda_dir, 'dead_keys.txt')
    if os.path.exists(dead_keys_path):
        with open(dead_keys_path) as f:
            dead_keys = {line.strip() for line in f if line.strip()}
        if dead_keys and key_col in df.columns:
            original_len = len(df)
            df = df[~df[key_col].isin(dead_keys)].copy()
            logger.info(f"Filtered {original_len - len(df)} rows for {len(dead_keys)} dead keys")

    # Merge segment info — handle both 'key' and actual key column name (e.g., 'Model_Hierarchy')
    key_col = key_cols[0] if isinstance(key_cols, list) else key_cols
    merge_col = key_col if key_col in seg_df.columns else ('key' if 'key' in seg_df.columns else key_col)
    seg_cols = [merge_col] + [c for c in ['segment_id', 'model_group', 'demand_pattern'] if c in seg_df.columns]
    seg_cols = [c for c in seg_cols if c in seg_df.columns]  # Only keep columns that exist
    if merge_col in seg_df.columns and merge_col in df.columns:
        seg_merge = seg_df[seg_cols].drop_duplicates(subset=[merge_col])
        df = df.merge(seg_merge, on=merge_col, how='left')
    else:
        logger.warning(f"Cannot merge segments: '{merge_col}' not in both DataFrames. "
                       f"Source cols: {list(df.columns)[:5]}, Seg cols: {list(seg_df.columns)[:5]}")

    # =====================================================================
    # FEATURE AVAILABILITY INTEGRATION
    # =====================================================================
    # If Feature Availability Detection has run, use its results to:
    # 1. Only allow known_in_future + partially_known as direct features
    # 2. Compute frozen embeddings for history_only features
    # 3. Exclude features that are too sparse
    # =====================================================================
    fa_output_dir = os.path.join(config.artifact_base_path, 'feature_availability_output')
    fa_context_path = os.path.join(fa_output_dir, 'feature_availability_to_feature_context.json')
    frozen_embedding_cols = []
    history_only_features = []

    if os.path.exists(fa_context_path):
        logger.info("Feature Availability context found — applying intelligent feature filtering")
        with open(fa_context_path) as f:
            fa_context = json.load(f)

        known_features = fa_context.get('known_in_future_features', [])
        history_only_features = fa_context.get('history_only_features', [])
        partially_known_features = fa_context.get('partially_known_features', [])
        excluded_features = fa_context.get('excluded_features', [])

        # Direct features: only known_in_future + partially_known
        allowed_direct_features = list(set(known_features + partially_known_features))

        # Compute frozen embeddings for history-only features
        if history_only_features:
            logger.info(f"Computing frozen embeddings for {len(history_only_features)} history-only features...")
            try:
                fa_result_path = os.path.join(fa_output_dir, 'feature_availability_result.json')
                with open(fa_result_path) as f:
                    fa_result = json.load(f)
                frozen_specs = fa_result.get('frozen_embedding_features', {})

                if frozen_specs:
                    from utils.feature_availability import compute_frozen_embedding_features, FrozenEmbeddingSpec

                    # Convert dict specs back to FrozenEmbeddingSpec objects
                    typed_specs = {}
                    for source_feat, specs_list in frozen_specs.items():
                        typed_specs[source_feat] = [
                            FrozenEmbeddingSpec(**s) for s in specs_list
                        ]

                    df, frozen_embedding_cols = compute_frozen_embedding_features(
                        df=df,
                        embedding_specs=typed_specs,
                        key_cols=key_cols if isinstance(key_cols, list) else [key_cols],
                        date_col=config.date_column,
                        target_col=config.target_column,
                        history_cutoff=fa_context.get('history_cutoff', {}).get('detected', config.train_end),
                    )
                    logger.info(f"Computed {len(frozen_embedding_cols)} frozen embedding features")
            except Exception as e:
                logger.warning(f"Could not compute frozen embeddings: {e}")
                frozen_embedding_cols = []

        # Build final allowed_external_cols: direct features + frozen embeddings
        allowed_external = list(set(allowed_direct_features + frozen_embedding_cols))
        logger.info(f"Feature availability filtering: {len(allowed_external)} features allowed "
                     f"({len(allowed_direct_features)} direct + {len(frozen_embedding_cols)} frozen embeddings)")
        if excluded_features:
            logger.info(f"  Excluded {len(excluded_features)} sparse features")
    else:
        # No feature availability context — use config-specified features (legacy behavior)
        logger.info("No Feature Availability context found — using config-specified features")
        allowed_external = list(set(config.all_numeric_features() + config.all_categorical_features()))

    # Resolve hierarchy columns via the single-source-of-truth resolver.
    # Precedence: explicit config override → persisted artifact from
    # segmentation → curated candidate list → legacy full auto-detect.
    _hierarchy_cols: list = []
    if getattr(config.design, 'enable_hierarchy_features', True):
        try:
            from utils.hierarchy_resolution import resolve_hierarchies
            _seg_dir = os.path.join(config.artifact_base_path, 'seg_output')
            _h = resolve_hierarchies(config=config, source_df=df, seg_dir=_seg_dir)
            # Feature engineering wants a flat coarse→fine list. Product
            # hierarchy drives the primary features; if it's empty fall
            # back to the flat union (legacy behaviour for old configs).
            _hierarchy_cols = list(_h.product) if _h.product else list(_h.flat)
            if _hierarchy_cols:
                logger.info(
                    "Hierarchy columns resolved for features (source=%s): %s",
                    _h.source, _hierarchy_cols,
                )
        except Exception as e:
            logger.debug(f"Hierarchy resolution skipped: {e}")

    # Collect future-unknown features for Phase 6 history embeddings
    _future_unknown = history_only_features  # Initialized to [] above, populated if FA context exists

    # Execute using the strategy
    result = execute_feature_strategy(
        df=df,
        strategy=strategy,
        key_cols=key_cols if isinstance(key_cols, list) else [key_cols],
        date_col=config.date_column,
        target_col=config.target_column,
        train_start=config.train_start,
        train_end=config.train_end,
        val_start=config.val_start,
        val_end=config.val_end,
        test_start=config.test_start,
        test_end=config.test_end,
        forecast_lag=getattr(config.design, 'forecast_lag', 4),
        categorical_cols=config.all_categorical_features(),
        output_dir=feat_dir,
        allowed_external_cols=allowed_external,
        hierarchy_cols=_hierarchy_cols,
        future_unknown_features=_future_unknown,
    )

    logger.info("=" * 60)
    logger.info("DETERMINISTIC EXECUTOR COMPLETE")
    logger.info(f"  Train rows: {result.n_rows_train}")
    logger.info(f"  Val rows: {result.n_rows_val}")
    logger.info(f"  Test rows: {result.n_rows_test}")
    logger.info(f"  Features: {result.n_features_created}")
    logger.info("=" * 60)

    return {
        'n_rows_train': result.n_rows_train,
        'n_rows_val': result.n_rows_val,
        'n_rows_test': result.n_rows_test,
        'n_features': result.n_features_created,
        'feature_cols': result.feature_cols[:20],  # First 20
    }


def _enrich_segment_data_with_metrics(
    seg_df: pd.DataFrame,
    config: DemandForecastConfig,
    eda_dir: str,
) -> pd.DataFrame:
    """
    Enrich segment data with metrics needed for intelligent model level allocation.

    The allocate_model_levels function needs rich metrics like:
    - mean, sum (volume)
    - cv, zero_fraction (variability)
    - n_obs (data sufficiency)
    - forecastability_score (predictability)
    - demand_pattern, volume_tier (categorization)

    If the segmentation output is incomplete (e.g., LLM wrote simplified code),
    this function computes these metrics from the raw data.

    Parameters
    ----------
    seg_df : pd.DataFrame
        Segment assignments (may have limited columns)
    config : DemandForecastConfig
        Configuration with data paths
    eda_dir : str
        EDA output directory (may have per_key_metrics.csv)

    Returns
    -------
    pd.DataFrame
        Enriched segment data with all metrics needed for allocation
    """
    import numpy as np
    from utils.agent_utilities import load_csv

    logger.info(f"Enriching segment data. Current columns: {list(seg_df.columns)}")

    # Check if we already have rich metrics (handle _clean suffix variants)
    # The downstream allocate_model_levels function can work with these column name variants
    def has_metric(df, base_name):
        """Check if metric exists with any variant name."""
        variants = [base_name, f'{base_name}_clean', f'volume_{base_name}']
        return any(v in df.columns for v in variants)

    # Required metrics for model level allocation
    # Note: n_obs is helpful but not strictly required - allocation can work without it
    core_metrics = ['mean', 'cv', 'zero_fraction']
    has_core_metrics = all(has_metric(seg_df, col) for col in core_metrics)

    # Also check for volume_mean which is commonly used instead of mean
    has_volume = 'volume_mean' in seg_df.columns or 'mean' in seg_df.columns

    if has_core_metrics and has_volume:
        logger.info("Segment data already has rich metrics (possibly with _clean suffix) - no enrichment needed")
        return seg_df

    logger.info("Segment data may need enrichment - checking EDA metrics...")

    # Try to load per_key_metrics from EDA first (fastest)
    eda_metrics_path = os.path.join(eda_dir, 'per_key_metrics.csv')
    if os.path.exists(eda_metrics_path):
        eda_df = pd.read_csv(eda_metrics_path)
        logger.info(f"Loaded EDA metrics: {list(eda_df.columns)}")

        # Merge EDA metrics into seg_df
        key_col = 'key' if 'key' in seg_df.columns else seg_df.columns[0]
        eda_key_col = 'key' if 'key' in eda_df.columns else eda_df.columns[0]

        # Rename EDA key column if needed
        if eda_key_col != key_col:
            eda_df = eda_df.rename(columns={eda_key_col: key_col})

        # Get columns to merge (exclude key)
        merge_cols = [c for c in eda_df.columns if c != key_col and c not in seg_df.columns]

        if merge_cols:
            seg_df = seg_df.merge(
                eda_df[[key_col] + merge_cols],
                on=key_col,
                how='left'
            )
            logger.info(f"Merged {len(merge_cols)} columns from EDA: {merge_cols[:10]}...")

    # Re-check if we now have core metrics after EDA merge (with _clean suffix handling)
    has_core_metrics_now = all(has_metric(seg_df, col) for col in core_metrics)
    has_volume_now = 'volume_mean' in seg_df.columns or 'mean' in seg_df.columns

    if has_core_metrics_now and has_volume_now:
        logger.info("After EDA merge, segment data has all required metrics")
        return seg_df

    # If still missing critical metrics, compute from raw data
    # Check which specific metrics are truly missing (not just name variants)
    truly_missing = []
    for col in core_metrics:
        if not has_metric(seg_df, col):
            truly_missing.append(col)
    if not has_volume_now:
        truly_missing.append('volume/mean')

    if truly_missing:
        logger.info(f"Still missing metrics: {truly_missing}. Computing from raw data...")

        # Load raw data
        raw_df = load_csv(config.input_data_path)
        key_cols = config.key_columns
        target_col = config.target_column
        key_col = key_cols[0] if len(key_cols) == 1 else key_cols

        # Create key if multiple key columns
        if isinstance(key_col, list):
            raw_df['_key'] = raw_df[key_col].astype(str).agg('_'.join, axis=1)
            key_col = '_key'

        # Compute per-key metrics
        metrics = raw_df.groupby(key_col).agg(
            n_obs=(target_col, 'count'),
            mean=(target_col, 'mean'),
            std=(target_col, 'std'),
            sum=(target_col, 'sum'),
            min=(target_col, 'min'),
            max=(target_col, 'max'),
        ).reset_index()

        # Compute CV (coefficient of variation)
        metrics['cv'] = np.where(
            metrics['mean'] > 0,
            metrics['std'] / metrics['mean'],
            0
        )

        # Compute zero_fraction
        zero_counts = raw_df.groupby(key_col)[target_col].apply(
            lambda x: (x == 0).sum() / len(x)
        ).reset_index()
        zero_counts.columns = [key_col, 'zero_fraction']
        metrics = metrics.merge(zero_counts, on=key_col, how='left')

        # Compute ADI (Average Demand Interval)
        def calc_adi(series):
            nonzero_idx = np.where(series > 0)[0]
            if len(nonzero_idx) < 2:
                return np.nan
            intervals = np.diff(nonzero_idx)
            return intervals.mean() if len(intervals) > 0 else np.nan

        adi_vals = raw_df.groupby(key_col)[target_col].apply(calc_adi).reset_index()
        adi_vals.columns = [key_col, 'adi']
        metrics = metrics.merge(adi_vals, on=key_col, how='left')

        # Assign demand_pattern based on ADI and CV2
        metrics['cv2'] = metrics['cv'] ** 2
        metrics['demand_pattern'] = 'smooth'
        metrics.loc[(metrics['adi'] > 1.32) & (metrics['cv2'] <= 0.49), 'demand_pattern'] = 'intermittent'
        metrics.loc[(metrics['adi'] <= 1.32) & (metrics['cv2'] > 0.49), 'demand_pattern'] = 'erratic'
        metrics.loc[(metrics['adi'] > 1.32) & (metrics['cv2'] > 0.49), 'demand_pattern'] = 'lumpy'

        # Assign volume_tier
        vol_33 = metrics['mean'].quantile(0.33)
        vol_67 = metrics['mean'].quantile(0.67)
        metrics['volume_tier'] = pd.cut(
            metrics['mean'],
            bins=[-np.inf, vol_33, vol_67, np.inf],
            labels=['low', 'medium', 'high']
        ).astype(str)

        # Compute forecastability_score (based on CV and autocorrelation proxy)
        # Lower CV = more forecastable
        metrics['forecastability_score'] = np.clip(1.0 - metrics['cv'] / 3.0, 0.1, 0.9)

        # Merge computed metrics into seg_df
        seg_key_col = 'key' if 'key' in seg_df.columns else seg_df.columns[0]
        merge_cols = [c for c in metrics.columns if c != key_col and c not in seg_df.columns]

        if key_col != seg_key_col:
            metrics = metrics.rename(columns={key_col: seg_key_col})

        seg_df = seg_df.merge(
            metrics[[seg_key_col] + merge_cols],
            on=seg_key_col,
            how='left'
        )

        logger.info(f"Computed and merged {len(merge_cols)} metrics from raw data")

    logger.info(f"Enriched segment data now has {len(seg_df.columns)} columns: {list(seg_df.columns)[:15]}...")
    return seg_df


def _create_training_manifest_and_context(
    config: DemandForecastConfig,
    strategy: FeatureStrategyDecision,
    quality_assessment: FeatureQualityAssessment,
    feat_dir: str,
    seg_dir: str,
    eda_dir: str,
) -> None:
    """
    Create STATE-OF-THE-ART training_manifest.csv and feature_to_training_context.json.

    This function creates comprehensive context files that the Training crew can use
    directly. It includes:
    - Model-level allocation
    - AI strategy decisions with reasoning
    - Quality assessment
    - Segmentation-informed training strategies
    - EDA insights for training

    This is DETERMINISTIC - no LLM involved.
    """
    from utils.intelligent_modeling import allocate_model_levels, categorize_features_for_model_level
    from utils.context_schema import ContextBuilder, SemanticTypes

    # Load segment assignments
    seg_df = pd.read_csv(os.path.join(seg_dir, 'per_key_with_segments.csv'))

    # ==========================================================================
    # CRITICAL: Enrich segment data with metrics for intelligent allocation
    # ==========================================================================
    # If segmentation output is incomplete (e.g., LLM wrote simplified code),
    # we need to compute metrics from raw data to enable intelligent decisions
    seg_df = _enrich_segment_data_with_metrics(seg_df, config, eda_dir)

    # Load feature metadata
    with open(os.path.join(feat_dir, 'feature_metadata.json')) as f:
        feat_meta = json.load(f)

    # ==========================================================================
    # LOAD STATE-OF-THE-ART SEGMENTATION CONTEXT FOR TRAINING
    # ==========================================================================
    seg_train_context_path = os.path.join(seg_dir, 'segmentation_to_training_context.json')
    seg_train_context = {}
    if os.path.exists(seg_train_context_path):
        with open(seg_train_context_path) as f:
            seg_train_context = json.load(f)
        logger.info("Loaded state-of-the-art segmentation training context")

    # Extract key information from segmentation
    segment_model_strategy = seg_train_context.get('segment_model_strategy', {})
    segments_by_difficulty = seg_train_context.get('segments_by_difficulty', {})
    tweedie_loss_segments = seg_train_context.get('tweedie_loss_segments', [])
    global_training_guidance = seg_train_context.get('global_training_guidance', {})
    clustering_quality = seg_train_context.get('clustering_quality', {})

    # Load EDA insights for training
    eda_train_context_path = os.path.join(eda_dir, 'eda_to_training_context.json')
    eda_train_context = {}
    if os.path.exists(eda_train_context_path):
        with open(eda_train_context_path) as f:
            eda_train_context = json.load(f)

    # Model level allocation
    alloc_cfg = config.design.model_level_allocation
    allocation_result = allocate_model_levels(
        seg_df=seg_df,
        key_col='key',
        segment_col='segment_id',
        min_individual_score=alloc_cfg.min_individual_score,
        max_individual_pct=alloc_cfg.max_individual_pct,
        min_segment_size=alloc_cfg.min_segment_size,
        volume_override_quantile=alloc_cfg.volume_override_quantile,
        forecastability_override=alloc_cfg.forecastability_override,
        min_nonzero_obs_for_individual=getattr(alloc_cfg, 'min_nonzero_obs_for_individual', 52),
        max_zero_fraction_for_individual=getattr(alloc_cfg, 'max_zero_fraction_for_individual', 0.70),
        top_volume_bypass_quantile=getattr(alloc_cfg, 'top_volume_bypass_quantile', 0.80),
        verbose=True,
        time_format=config.time_format,
    )

    # Create training manifest
    manifest = allocation_result.allocations.copy()
    manifest['model_group'] = manifest['segment_id'].astype(str)
    mask_individual = manifest['model_level'] != manifest['segment_id'].astype(str)
    manifest.loc[mask_individual, 'model_group'] = manifest.loc[mask_individual, 'key']

    # Add demand pattern columns
    if 'intermittency_class' in seg_df.columns:
        manifest['intermittency_class'] = manifest['key'].map(seg_df.set_index('key')['intermittency_class'])
    if 'demand_pattern' in seg_df.columns:
        manifest['demand_pattern'] = manifest['key'].map(seg_df.set_index('key')['demand_pattern'])

    manifest['feature_file'] = 'train_features.csv'
    manifest.to_csv(os.path.join(feat_dir, 'training_manifest.csv'), index=False)

    # Feature categorization
    all_features = feat_meta.get('feature_cols', [])
    categories = categorize_features_for_model_level(all_features, key_col='key')
    cat_summary = {cat: len(cols) for cat, cols in categories.items()}

    # Build context using schema
    ctx = ContextBuilder(
        context_type='feature_to_training',
        source_crew='feature_crew',
        target_crews=['training_crew']
    )

    # Model-level allocation summary
    ctx.add_field(
        key='model_level_summary',
        value={
            'total_unique_levels': int(manifest['model_level'].nunique()),
            'group_models': allocation_result.summary['segment_models'],
            'individual_key_models': allocation_result.summary['individual_models'],
            'individual_pct': allocation_result.summary['individual_pct'],
            'avg_segment_size': allocation_result.summary['avg_segment_size'],
        },
        description='Model-level allocation summary',
        semantic_type=SemanticTypes.MODEL_LEVEL_SUMMARY,
        required=True,
        required_by=['training_crew']
    )

    # AI Strategy summary (from reasoning)
    ctx.add_field(
        key='ai_strategy_summary',
        value={
            'strategy_id': strategy.strategy_id,
            'complexity_level': strategy.complexity_level.value,
            'primary_emphasis': strategy.primary_emphasis.value,
            'acf_informed': strategy.acf_informed,
            'acf_reasoning': strategy.acf_reasoning,
            'seasonality_reasoning': strategy.seasonality_reasoning,
            'intermittency_reasoning': strategy.intermittency_reasoning,
            'hypotheses': [h.to_dict() for h in strategy.hypotheses],
        },
        description='AI reasoning summary for feature strategy',
        semantic_type=SemanticTypes.FEATURE_SUMMARY,
    )

    # Quality assessment summary (from analyst reasoning)
    ctx.add_field(
        key='quality_assessment',
        value={
            'overall_quality': quality_assessment.overall_quality,
            'quality_score': quality_assessment.quality_score,
            'hypotheses_confirmed': quality_assessment.hypotheses_confirmed,
            'hypotheses_rejected': quality_assessment.hypotheses_rejected,
            'concerns': quality_assessment.concerns,
            'recommendations': quality_assessment.recommendations,
            'reasoning_summary': quality_assessment.reasoning_summary,
        },
        description='AI quality assessment and hypothesis validation',
        semantic_type=SemanticTypes.QUALITY_ASSESSMENT,
    )

    # Feature summary with ADAPTIVE config
    ctx.add_field(
        key='feature_summary',
        value={
            'total_features': feat_meta.get('n_features', 0),
            'feature_cols': all_features[:20],
            'feature_categories': cat_summary,
            'adaptive_config': {
                'target_lags': strategy.target_lags,
                'rolling_windows': strategy.rolling_windows,
                'rolling_stats': strategy.rolling_stats,
                'seasonal_period': strategy.seasonal_period,
                'fourier_order': strategy.fourier_order,
                'acf_informed_lags': strategy.acf_informed,
                'include_intermittency': strategy.include_intermittency_features,
                'include_trend': strategy.include_trend_features,
            },
        },
        description='Feature summary with adaptive configuration',
        semantic_type=SemanticTypes.FEATURE_SUMMARY,
    )

    # ==========================================================================
    # STATE-OF-THE-ART: Per-segment training strategy from Segmentation
    # ==========================================================================
    ctx.add_field(
        key='per_segment_training_strategy',
        value=segment_model_strategy,
        description='Per-segment model strategy from segmentation (includes loss, validation, hyperparams)',
        semantic_type=SemanticTypes.SEGMENT_TRAINING_STRATEGY,
    )

    # Segments by difficulty for prioritization
    ctx.add_field(
        key='segments_by_difficulty',
        value=segments_by_difficulty,
        description='Segments categorized by forecasting difficulty (easy/medium/hard)',
        semantic_type=SemanticTypes.SEGMENT_DIFFICULTY,
    )

    # Quick reference: segments needing tweedie loss
    ctx.add_field(
        key='tweedie_loss_segments',
        value=tweedie_loss_segments,
        description='Segments that should use Tweedie loss (intermittent/lumpy patterns)',
        semantic_type=SemanticTypes.LOSS_RECOMMENDATION,
    )

    # Global training guidance from segmentation
    ctx.add_field(
        key='global_training_guidance',
        value={
            **global_training_guidance,
            'feature_engineering_emphasis': strategy.primary_emphasis.value,
            'feature_quality_score': quality_assessment.quality_score,
        },
        description='Global training guidance (CV folds, early stopping, etc.)',
        semantic_type=SemanticTypes.TRAINING_GUIDANCE,
    )

    # ==========================================================================
    # EDA INSIGHTS FOR TRAINING (seasonality, trend, changepoints)
    # ==========================================================================
    ctx.add_field(
        key='eda_insights_for_training',
        value={
            'seasonality': eda_train_context.get('seasonality_summary', {
                'has_seasonality_pct': 0,
                'avg_strength': 0,
                'dominant_period': strategy.seasonal_period,
                'recommendation': 'seasonal_split' if strategy.has_strong_seasonality else 'time_series_split',
            }),
            'trend': eda_train_context.get('trend_summary', {
                'pct_with_trend': 0,
                'avg_strength': strategy.trend_strength,
            }),
            'changepoints': eda_train_context.get('changepoint_summary', {
                'pct_with_changepoints': 0,
            }),
            'intermittency': {
                'pct_intermittent': strategy.intermittency_pct,
                'include_features': strategy.include_intermittency_features,
            },
        },
        description='EDA insights relevant for training decisions',
        semantic_type=SemanticTypes.EDA_INSIGHTS,
    )

    # Clustering quality from segmentation
    ctx.add_field(
        key='clustering_quality',
        value=clustering_quality,
        description='Clustering quality metrics from segmentation',
        semantic_type=SemanticTypes.CLUSTERING_QUALITY,
    )

    # Training instructions
    ctx.add_field(
        key='training_instructions',
        value=[
            'Read training_manifest.csv for model-level assignments',
            f'AI Strategy: {strategy.primary_emphasis.value} emphasis',
            f'Quality: {quality_assessment.overall_quality} (score: {quality_assessment.quality_score:.1f})',
            'Train ONE model per unique model_level',
            f'Use per_segment_training_strategy for segment-specific loss/validation',
            f'Segments needing Tweedie loss: {len(tweedie_loss_segments)}',
            f'Difficult segments (lumpy): {len(segments_by_difficulty.get("hard", []))}',
        ],
        description='Training instructions from AI reasoning + segmentation',
        semantic_type=SemanticTypes.TRAINING_INSTRUCTIONS,
    )

    # File paths
    ctx.add_field(
        key='file_paths',
        value={
            'training_manifest': os.path.join(feat_dir, 'training_manifest.csv'),
            'feature_dir': feat_dir,
            'seg_dir': seg_dir,
            'feature_files': ['train_features.csv', 'val_features.csv', 'test_features.csv'],
            'segmentation_training_context': seg_train_context_path,
        },
        description='Paths to training files',
        semantic_type=SemanticTypes.FILE_PATHS,
    )

    ctx.add_metadata(run_id=f"feat_{strategy.strategy_id}")
    ctx.save(os.path.join(feat_dir, 'feature_to_training_context.json'))

    logger.info(f"Created training manifest: {len(manifest)} keys")
    logger.info(f"Created STATE-OF-THE-ART feature_to_training_context.json")
    logger.info(f"  - Per-segment strategies: {len(segment_model_strategy)}")
    logger.info(f"  - Tweedie loss segments: {len(tweedie_loss_segments)}")
    logger.info(f"  - Difficult segments: {len(segments_by_difficulty.get('hard', []))}")


# =============================================================================
# CREW CREATION
# =============================================================================

def create_feature_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> Crew:
    """
    Create the Intelligent Feature Engineering Crew.

    This crew uses the REASONING + EXECUTION separation pattern:
    1. Strategist (AI REASONING): Decides what features to create
    2. Executor (DETERMINISTIC): Executes the strategy reliably
    3. Analyst (AI REASONING): Validates quality and hypotheses
    """
    artifact_base = config.artifact_base_path
    seg_dir = os.path.join(artifact_base, "seg_output")
    eda_dir = os.path.join(artifact_base, "eda_output")
    feat_dir = os.path.join(artifact_base, "feature_output")

    os.makedirs(feat_dir, exist_ok=True)

    # Get safe output paths
    feat_dir_out = _get_output_path(feat_dir)

    # Get config details for context
    data_path = config.input_data_path

    # Get allowed model families
    allowed_model_families = list(config.design.model_families)
    enable_deep_models = config.design.enable_deep_models

    deep_model_types = ['tft', 'lstm', 'nbeats', 'deepar', 'wavenet']
    if not enable_deep_models:
        allowed_model_families = [m for m in allowed_model_families if m.lower() not in deep_model_types]

    # WRITE PROTECTION: Prevent LLM agents from corrupting upstream outputs
    protected_dirs = [eda_dir, seg_dir]

    # Create agents with write-protection for upstream output directories
    strategist = _create_feature_strategist_agent(llm, protected_paths=protected_dirs)
    analyst = _create_feature_analyst_agent(llm, allowed_model_families, enable_deep_models, protected_paths=protected_dirs)

    # -------------------------------------------------------------------------
    # Task 1: AI REASONING - Feature Strategy Decision
    # -------------------------------------------------------------------------
    # Load context for the strategist
    reasoning_context = create_reasoning_context(eda_dir, seg_dir)

    task_strategize = Task(
        name="create_feature_strategy",
        description=(
            "# AI REASONING TASK: Create Feature Engineering Strategy\n\n"
            "You are the Feature Strategist. Your job is to REASON about the data\n"
            "and DECIDE what feature engineering strategy to use.\n\n"
            "## DATA CONTEXT (from EDA and Segmentation)\n\n"
            "```python\n"
            "import os, json\n"
            "import pandas as pd\n"
            f"eda_dir = '{eda_dir}'\n"
            f"seg_dir = '{seg_dir}'\n"
            f"feat_dir = '{feat_dir}'\n\n"
            "# Load EDA insights\n"
            "eda_ctx_path = os.path.join(eda_dir, 'eda_to_feature_context.json')\n"
            "eda_ctx = {}\n"
            "if os.path.exists(eda_ctx_path):\n"
            "    with open(eda_ctx_path) as f:\n"
            "        eda_ctx = json.load(f)\n\n"
            "# Load ACF analysis\n"
            "acf_path = os.path.join(eda_dir, 'autocorrelation_summary.csv')\n"
            "significant_lags = []\n"
            "if os.path.exists(acf_path):\n"
            "    acf_df = pd.read_csv(acf_path)\n"
            "    if 'significant_lags' in acf_df.columns:\n"
            "        all_lags = set()\n"
            "        for lags_str in acf_df['significant_lags'].dropna():\n"
            "            if isinstance(lags_str, str):\n"
            "                try:\n"
            "                    lags = json.loads(lags_str.replace(\"'\", '\"'))\n"
            "                    all_lags.update(lags)\n"
            "                except: pass\n"
            "        significant_lags = sorted(list(all_lags))[:15]\n"
            "print(f'ACF significant lags: {significant_lags}')\n\n"
            "# Load seasonality (using strength-based metrics)\n"
            "seas_path = os.path.join(eda_dir, 'seasonality_analysis.json')\n"
            "seasonality = {}\n"
            "if os.path.exists(seas_path):\n"
            "    with open(seas_path) as f:\n"
            "        seasonality = json.load(f)\n"
            "has_seasonality_pct = seasonality.get('pct_with_seasonality', seasonality.get('has_seasonality_pct', 0))\n"
            "avg_seasonal_strength = seasonality.get('avg_seasonal_strength', 0.0)\n"
            f"dominant_period = seasonality.get('dominant_period', {12 if config.time_format == 'year_month' else 52})\n"
            "# Use strength-based check: need both presence AND meaningful strength\n"
            "has_strong_seasonality = has_seasonality_pct > 0.3 and avg_seasonal_strength > 0.1\n"
            "print(f'Seasonality: {has_seasonality_pct*100:.0f}% seasonal (strength={avg_seasonal_strength:.2f}), period={dominant_period}')\n"
            "print(f'  Strong seasonality: {has_strong_seasonality}')\n\n"
            "# Load trend analysis\n"
            "trend_path = os.path.join(eda_dir, 'trend_analysis.json')\n"
            "trend = {}\n"
            "if os.path.exists(trend_path):\n"
            "    with open(trend_path) as f:\n"
            "        trend = json.load(f)\n"
            "strongly_trending_pct = trend.get('strongly_trending_pct', 0)\n"
            "print(f'Trend: {strongly_trending_pct*100:.0f}% strongly trending')\n\n"
            "# =================================================================\n"
            "# LOAD STATE-OF-THE-ART SEGMENTATION CONTEXT\n"
            "# =================================================================\n"
            "seg_ctx_path = os.path.join(seg_dir, 'segmentation_to_feature_context.json')\n"
            "seg_ctx = {}\n"
            "if os.path.exists(seg_ctx_path):\n"
            "    with open(seg_ctx_path) as f:\n"
            "        seg_ctx = json.load(f)\n\n"
            "# Get key information from segmentation\n"
            "n_segments = seg_ctx.get('n_segments', 0)\n"
            "segment_profiles = seg_ctx.get('segment_profiles', {})\n"
            "segment_feature_strategies = seg_ctx.get('segment_feature_strategies', {})\n"
            "global_recs = seg_ctx.get('global_recommendations', {})\n"
            "intermittency_segments = seg_ctx.get('intermittency_segments', [])\n"
            "log_transform_segments = seg_ctx.get('log_transform_segments', [])\n"
            "intermittency_summary = seg_ctx.get('intermittency_summary', {})\n\n"
            "# Calculate intermittency percentage from segment profiles\n"
            "intermittent_count = sum(1 for s in segment_profiles.values() \n"
            "                        if s.get('demand_pattern', '') in ['intermittent', 'lumpy'])\n"
            "intermittency_pct = intermittent_count / max(n_segments, 1)\n\n"
            "print(f'=== SEGMENTATION CONTEXT (STATE-OF-THE-ART) ===')\n"
            "print(f'Segments: {n_segments}')\n"
            "print(f'Intermittent segments: {len(intermittency_segments)} ({intermittency_pct*100:.0f}%)')\n"
            "print(f'Log transform segments: {len(log_transform_segments)}')\n\n"
            "# Print per-segment feature recommendations\n"
            "print(f'\\n=== PER-SEGMENT FEATURE STRATEGIES ===')\n"
            "for seg_id, strategy in segment_feature_strategies.items():\n"
            "    pattern = strategy.get('dominant_pattern', 'unknown')\n"
            "    interm_needed = strategy.get('intermittency_features', False)\n"
            "    log_needed = strategy.get('log_transform', False)\n"
            "    max_features = strategy.get('max_total_features', 50)\n"
            "    print(f'  Segment {seg_id}: pattern={pattern}, intermittency={interm_needed}, log={log_needed}, max_features={max_features}')\n\n"
            "# Print global recommendations from segmentation\n"
            "print(f'\\n=== GLOBAL RECOMMENDATIONS FROM SEGMENTATION ===')\n"
            "seg_target_lags = global_recs.get('target_lags', [])\n"
            "seg_rolling_windows = global_recs.get('rolling_windows', [])\n"
            "seg_rolling_stats = global_recs.get('rolling_stats', [])\n"
            "seg_seasonality = global_recs.get('seasonality_detected', False)\n"
            "seg_acf_lags = global_recs.get('acf_informed_lags', [])\n"
            "print(f'  Target lags: {seg_target_lags}')\n"
            "print(f'  Rolling windows: {seg_rolling_windows}')\n"
            "print(f'  Rolling stats: {seg_rolling_stats}')\n"
            "print(f'  ACF-informed lags: {seg_acf_lags}')\n"
            "print(f'  Seasonality detected: {seg_seasonality}')\n\n"
            "# Load changepoint analysis (using significance-filtered metrics)\n"
            "cp_path = os.path.join(eda_dir, 'changepoint_analysis.json')\n"
            "changepoints = {}\n"
            "if os.path.exists(cp_path):\n"
            "    with open(cp_path) as f:\n"
            "        changepoints = json.load(f)\n"
            "# Prefer the significance-filtered metric (pct_with_significant_changepoints)\n"
            "pct_with_changepoints = changepoints.get('pct_with_significant_changepoints', \n"
            "                                          changepoints.get('pct_with_changepoints', 0))\n"
            "changepoint_dates = changepoints.get('common_changepoint_dates', [])\n"
            "# Check if changepoints are truly significant\n"
            "has_significant_changepoints = pct_with_changepoints > 0.2\n"
            "print(f'Changepoints: {pct_with_changepoints*100:.0f}% with SIGNIFICANT changepoints')\n"
            "print(f'  Common dates: {changepoint_dates[:3]}')\n"
            "print(f'  Include changepoint features: {has_significant_changepoints}')\n\n"
            "print('\\n=== NOW REASON AND DECIDE ===')\n"
            "print('Based on the above data, output your JSON strategy decision.')\n"
            "```\n\n"
            "## YOUR REASONING TASK\n\n"
            "Based on the data context above, you must:\n"
            "1. REASON about what lags to use based on ACF analysis\n"
            "2. REASON about seasonal features based on FFT analysis\n"
            "3. REASON about intermittency features based on demand patterns\n"
            "4. Form HYPOTHESES about what features will be important\n"
            "5. Output a COMPLETE JSON strategy decision\n\n"
            "## OUTPUT: Valid JSON (no other text)\n\n"
            "After running the code above to see the data, output ONLY a valid JSON object:\n"
            "```json\n"
            "{\n"
            '  "strategy_id": "strategy_YYYYMMDD_HHMMSS",\n'
            '  "complexity_level": "standard",\n'
            '  "primary_emphasis": "balanced",\n'
            '  "encoding_strategy": "target_encoding",\n'
            f'  "target_lags": {[1, 2, 3, 6, 12] if config.time_format == "year_month" else [1, 4, 13, 26, 52]},\n'
            f'  "feature_lags": {[1, 2] if config.time_format == "year_month" else [1, 4]},\n'
            '  "acf_informed": true,\n'
            f'  "acf_reasoning": "ACF shows significant lags at {("1, 2, 3, 12" if config.time_format == "year_month" else "1, 4, 13, 52")}...",\n'
            f'  "rolling_windows": {[3, 6, 12] if config.time_format == "year_month" else [4, 13, 26, 52]},\n'
            '  "rolling_stats": ["mean", "std", "min", "max"],\n'
            '  "rolling_reasoning": "...",\n'
            f'  "seasonal_period": {12 if config.time_format == "year_month" else 52},\n'
            '  "fourier_order": 3,\n'
            '  "has_strong_seasonality": true,\n'
            '  "seasonality_reasoning": "...",\n'
            '  "include_trend_features": true,\n'
            '  "trend_strength": 0.35,\n'
            '  "trend_reasoning": "...",\n'
            '  "include_intermittency_features": true,\n'
            '  "intermittency_pct": 0.45,\n'
            '  "intermittency_reasoning": "...",\n'
            '  "include_changepoint_indicators": false,\n'
            '  "changepoint_dates": [],\n'
            '  "changepoint_reasoning": "...",\n'
            '  "apply_log_transform": false,\n'
            '  "apply_differencing": false,\n'
            '  "transformation_reasoning": "...",\n'
            '  "hypotheses": [\n'
            '    {\n'
            '      "hypothesis_id": "H1",\n'
            '      "description": "Lag-52 will be highly important due to annual seasonality",\n'
            '      "feature_types": ["lag", "seasonal"],\n'
            '      "expected_importance": "high",\n'
            '      "reasoning": "FFT shows strong annual pattern",\n'
            '      "data_evidence": {"seasonality_pct": 0.78}\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```\n"
        ),
        agent=strategist,
        expected_output=(
            "A complete JSON strategy decision with reasoning for all choices "
            "and at least 2 testable hypotheses about feature importance."
        ),
        output_file=os.path.join(feat_dir_out, "strategist_output.md"),
    )

    # -------------------------------------------------------------------------
    # Task 2: DETERMINISTIC EXECUTION (handled in run_feature_crew, not as a Task)
    # -------------------------------------------------------------------------
    # Note: The executor is NOT an LLM agent - it's a Python function
    # that runs AFTER the strategist completes. See run_feature_crew().

    # -------------------------------------------------------------------------
    # Task 3: AI REASONING - Feature Quality Analysis
    # -------------------------------------------------------------------------
    task_analyze = Task(
        name="analyze_feature_quality",
        description=(
            "# AI REASONING TASK: Assess Feature Quality and Validate Hypotheses\n\n"
            "You are the Feature Quality Analyst. Your job is to REASON about the\n"
            "feature engineering results and VALIDATE the strategist's hypotheses.\n\n"
            "## QUALITY METRICS (load from feature_quality_metrics.json)\n\n"
            "```python\n"
            "import os, json\n"
            f"feat_dir = '{feat_dir}'\n\n"
            "# Load quality metrics\n"
            "metrics_path = os.path.join(feat_dir, 'feature_quality_metrics.json')\n"
            "metrics = {}\n"
            "if os.path.exists(metrics_path):\n"
            "    with open(metrics_path) as f:\n"
            "        metrics = json.load(f)\n"
            "print(f'Total features: {metrics.get(\"total_features\", 0)}')\n"
            "print(f'Lag features: {metrics.get(\"lag_features\", 0)}')\n"
            "print(f'Rolling features: {metrics.get(\"rolling_features\", 0)}')\n"
            "print(f'Seasonal features: {metrics.get(\"seasonal_features\", 0)}')\n"
            "print(f'Intermittency features: {metrics.get(\"intermittency_features\", 0)}')\n"
            "print(f'Missing value %: {metrics.get(\"missing_pct\", 0):.2f}')\n"
            "print(f'Constant features: {metrics.get(\"constant_features\", [])}')\n"
            "print(f'High correlation pairs: {len(metrics.get(\"high_corr_pairs\", []))}')\n\n"
            "# Load strategist's hypotheses\n"
            "strategy_path = os.path.join(feat_dir, 'feature_strategy_decision.json')\n"
            "strategy = {}\n"
            "if os.path.exists(strategy_path):\n"
            "    with open(strategy_path) as f:\n"
            "        strategy = json.load(f)\n"
            "hypotheses = strategy.get('hypotheses', [])\n"
            "print(f'\\nHypotheses to validate: {len(hypotheses)}')\n"
            "for h in hypotheses:\n"
            "    print(f'  - {h.get(\"hypothesis_id\")}: {h.get(\"description\")}')\n\n"
            "print('\\n=== NOW REASON AND ASSESS ===')\n"
            "print('Based on the metrics above, output your JSON quality assessment.')\n"
            "```\n\n"
            "## YOUR REASONING TASK\n\n"
            "Based on the quality metrics, you must:\n"
            "1. ASSESS overall feature quality (excellent/good/acceptable/poor)\n"
            "2. VALIDATE each hypothesis - was it confirmed or rejected?\n"
            "3. Identify any CONCERNS with reasoning\n"
            "4. Provide specific RECOMMENDATIONS\n\n"
            "## OUTPUT: Valid JSON (no other text)\n\n"
            "```json\n"
            "{\n"
            '  "overall_quality": "good",\n'
            '  "quality_reasoning": "Features created successfully with low missing values...",\n'
            '  "hypotheses_confirmed": ["H1"],\n'
            '  "hypotheses_rejected": ["H2"],\n'
            '  "hypothesis_assessment": "H1 confirmed because lag_52 features were created...",\n'
            '  "concerns": [\n'
            '    "High correlation between lag_1 and lag_2 features"\n'
            '  ],\n'
            '  "recommendations": [\n'
            '    "Consider feature selection to reduce redundancy"\n'
            '  ],\n'
            '  "reasoning_summary": "Overall the feature engineering was successful..."\n'
            '}\n'
            "```\n"
        ),
        agent=analyst,
        expected_output=(
            "A complete JSON quality assessment with hypothesis validation "
            "and actionable recommendations."
        ),
        output_file=os.path.join(feat_dir_out, "analyst_output.md"),
        context=[task_strategize],
    )

    # -------------------------------------------------------------------------
    # Task 4: Documentation - Generate comprehensive insights guide
    # -------------------------------------------------------------------------
    documentation_agent = _create_feature_documentation_agent(llm)
    task_document = Task(
        name="generate_feature_documentation",
        description=(
            "# CREATE COMPREHENSIVE FEATURE ENGINEERING DOCUMENTATION\n\n"
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
            "from datetime import datetime\n\n"
            f"feat_dir = '{feat_dir}'\n"
            f"seg_dir = '{seg_dir}'\n"
            f"eda_dir = '{eda_dir}'\n\n"
            "# Load feature engineering outputs\n"
            "strategy = json.load(open(os.path.join(feat_dir, 'feature_strategy_decision.json'))) if os.path.exists(os.path.join(feat_dir, 'feature_strategy_decision.json')) else {}\n"
            "quality = json.load(open(os.path.join(feat_dir, 'feature_quality_metrics.json'))) if os.path.exists(os.path.join(feat_dir, 'feature_quality_metrics.json')) else {}\n"
            "assessment = json.load(open(os.path.join(feat_dir, 'feature_quality_assessment.json'))) if os.path.exists(os.path.join(feat_dir, 'feature_quality_assessment.json')) else {}\n"
            "train_context = json.load(open(os.path.join(feat_dir, 'feature_to_training_context.json'))) if os.path.exists(os.path.join(feat_dir, 'feature_to_training_context.json')) else {}\n\n"
            "# Extract key metrics\n"
            "total_features = quality.get('total_features', 0)\n"
            "lag_features = quality.get('lag_features', 0)\n"
            "rolling_features = quality.get('rolling_features', 0)\n"
            "seasonal_features = quality.get('seasonal_features', 0)\n"
            "intermittency_features = quality.get('intermittency_features', 0)\n"
            "missing_pct = quality.get('missing_pct', 0)\n\n"
            "# Strategy details\n"
            "complexity = strategy.get('complexity_level', 'standard')\n"
            "emphasis = strategy.get('primary_emphasis', 'balanced')\n"
            "target_lags = strategy.get('target_lags', [])\n"
            "acf_reasoning = strategy.get('acf_reasoning', '')\n"
            "seasonality_reasoning = strategy.get('seasonality_reasoning', '')\n"
            "hypotheses = strategy.get('hypotheses', [])\n\n"
            "# Quality assessment\n"
            "overall_quality = assessment.get('overall_quality', 'unknown')\n"
            "confirmed = assessment.get('hypotheses_confirmed', [])\n"
            "rejected = assessment.get('hypotheses_rejected', [])\n\n"
            "# Build markdown\n"
            "md = f'''# Feature Engineering Insights Guide\n"
            "**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "---\n\n"
            "## Executive Summary\n\n"
            "This document explains the feature engineering process, the reasoning behind\n"
            "each decision, and how upstream insights from EDA and Segmentation informed\n"
            "the feature creation strategy.\n\n"
            "### Feature Engineering Overview\n\n"
            "| Metric | Value | Notes |\n"
            "|--------|-------|-------|\n"
            "| Total Features | {total_features} | All engineered features |\n"
            "| Lag Features | {lag_features} | Historical demand values |\n"
            "| Rolling Features | {rolling_features} | Moving averages/statistics |\n"
            "| Seasonal Features | {seasonal_features} | Calendar and Fourier |\n"
            "| Intermittency Features | {intermittency_features} | Zero-inflation handling |\n"
            "| Missing Value % | {missing_pct:.2f}% | After feature creation |\n"
            "| Overall Quality | {overall_quality.upper()} | AI assessment |\n\n"
            "---\n\n"
            "## 1. Strategy Decision\n\n"
            "### Complexity Level: {complexity.upper()}\n\n"
            "**Why this complexity?**\n"
            "- Minimal: Basic lags only (simple data)\n"
            "- Standard: Full feature set (typical data)\n"
            "- Comprehensive: All features + interactions (complex data)\n\n"
            "The `{complexity}` level was chosen based on data characteristics.\n\n"
            "### Primary Emphasis: {emphasis.upper()}\n\n"
            "This indicates which feature types received special attention.\n\n"
            "---\n\n"
            "## 2. Lag Feature Selection\n\n"
            "### Selected Target Lags\n"
            "`{target_lags}`\n\n"
            "### ACF-Informed Reasoning\n"
            "{acf_reasoning if acf_reasoning else \"Based on standard lag patterns for the time format.\"}\n\n"
            "**Why These Lags Matter:**\n"
            "- **Lag-1**: Immediate dependency (previous period)\n"
            "- **Lag-4**: Monthly pattern (for weekly data)\n"
            "- **Lag-13**: Quarterly pattern\n"
            "- **Lag-52**: Annual seasonality\n\n"
            "---\n\n"
            "## 3. Seasonality Handling\n\n"
            "{seasonality_reasoning if seasonality_reasoning else \"Seasonal features created based on detected patterns.\"}\n\n"
            "---\n\n"
            "## 4. AI Hypotheses & Validation\n\n"
            "The AI Strategist formulated testable hypotheses about feature importance.\n\n"
            "### Hypotheses Formulated\n\n'''\n\n"
            "for h in hypotheses:\n"
            "    status = '✓ CONFIRMED' if h.get('hypothesis_id') in confirmed else '✗ REJECTED' if h.get('hypothesis_id') in rejected else '? UNTESTED'\n"
            "    md += f'''#### {h.get(\"hypothesis_id\", \"H?\")}: {status}\n"
            "**Hypothesis:** {h.get(\"description\", \"N/A\")}\n"
            "**Reasoning:** {h.get(\"reasoning\", \"N/A\")}\n"
            "**Expected Importance:** {h.get(\"expected_importance\", \"N/A\")}\n\n"
            "'''\n\n"
            "md += f'''\n"
            "---\n\n"
            "## 5. Quality Assessment\n\n"
            "### Overall Quality: {overall_quality.upper()}\n\n'''\n\n"
            "if assessment.get('concerns'):\n"
            "    md += '### Concerns Identified\\n\\n'\n"
            "    for c in assessment.get('concerns', []):\n"
            "        md += f'- ⚠️ {c}\\n'\n"
            "    md += '\\n'\n\n"
            "if assessment.get('recommendations'):\n"
            "    md += '### Recommendations\\n\\n'\n"
            "    for r in assessment.get('recommendations', []):\n"
            "        md += f'- 💡 {r}\\n'\n"
            "    md += '\\n'\n\n"
            "md += f'''\n"
            "---\n\n"
            "## 6. Key Learnings\n\n"
            "### What Worked Well\n\n'''\n\n"
            "if confirmed:\n"
            "    md += f'- Hypotheses {confirmed} were confirmed - the predicted feature importance held\\n'\n"
            "if intermittency_features > 0:\n"
            "    md += f'- {intermittency_features} intermittency features created for handling sparse demand\\n'\n"
            "if seasonal_features > 0:\n"
            "    md += f'- {seasonal_features} seasonal features capturing temporal patterns\\n'\n\n"
            "md += f'''\n\n"
            "### Areas for Improvement\n\n'''\n\n"
            "if rejected:\n"
            "    md += f'- Hypotheses {rejected} were rejected - revisit assumptions\\n'\n"
            "if missing_pct > 5:\n"
            "    md += f'- Missing value rate of {missing_pct:.1f}% is significant - consider imputation strategies\\n'\n\n"
            "md += f'''\n\n"
            "---\n\n"
            "*This documentation was auto-generated by the Feature Engineering Documentation Agent.*\n"
            "'''\n\n"
            "# Save documentation\n"
            "doc_path = os.path.join(feat_dir, 'FEATURE_ENGINEERING_INSIGHTS_GUIDE.md')\n"
            "with open(doc_path, 'w') as f:\n"
            "    f.write(md)\n\n"
            "print(f'Created Feature Engineering documentation: {total_features} features')\n"
            "print(f'Quality: {overall_quality}, Hypotheses: {len(confirmed)} confirmed, {len(rejected)} rejected')\n"
            "print(f'Saved: FEATURE_ENGINEERING_INSIGHTS_GUIDE.md')\n"
            "```"
        ),
        agent=documentation_agent,
        expected_output=(
            "Created FEATURE_ENGINEERING_INSIGHTS_GUIDE.md - comprehensive documentation with "
            "feature strategy reasoning, hypothesis validation, and quality assessment."
        ),
        output_file=os.path.join(feat_dir_out, "feature_documentation_report.md"),
        context=[task_strategize, task_analyze],
    )

    enable_insights = getattr(config.design, 'enable_insights_reports', False)
    if enable_insights:
        agents = [strategist, analyst, documentation_agent]
        tasks = [task_strategize, task_analyze, task_document]
    else:
        logger.info("SKIPPING feature insights report (enable_insights_reports=False)")
        agents = [strategist, analyst]
        tasks = [task_strategize, task_analyze]

    return Crew(
        name="Intelligent Feature Engineering Crew (Reasoning + Execution + Documentation)",
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


# =============================================================================
# RUN FUNCTION
# =============================================================================

def run_feature_crew(
    llm: LLM,
    config: DemandForecastConfig,
    config_yaml_path: str,
) -> FeatureCrewResult:
    """
    Run the Intelligent Feature Engineering Crew.

    This implements the REASONING + EXECUTION separation:
    1. Run Strategist (LLM) to get strategy decision
    2. Run DETERMINISTIC executor (no LLM) to create features
    3. Run Analyst (LLM) to validate quality
    """
    from utils.cost_tracking import get_cost_tracker, extract_tokens_from_crew_result

    # Validate input
    if not os.path.exists(config.input_data_path):
        raise FeatureEngineeringFailedError(
            f"Input data file not found: {config.input_data_path}\n"
            "Please ensure the data file exists at the configured path."
        )

    tracker = get_cost_tracker()
    tracker.start_crew("Feature Engineering Crew")

    model_id = getattr(llm, "model", "default")
    tracker.set_model(model_id)

    feat_dir = os.path.join(config.artifact_base_path, "feature_output")
    seg_dir = os.path.join(config.artifact_base_path, "seg_output")
    eda_dir = os.path.join(config.artifact_base_path, "eda_output")
    os.makedirs(feat_dir, exist_ok=True)

    # ======================================================================
    # WRITE PROTECTION: Set protected_paths on the LLM's CodeExecutionTool
    # ======================================================================
    _code_tool = getattr(llm, '_code_execution_tool', None)
    if _code_tool is not None:
        _code_tool.protected_paths = [eda_dir, seg_dir]
        logger.info(f"Write protection set on LLM CodeExecutionTool: eda_output, seg_output are READ-ONLY")

    # -------------------------------------------------------------------------
    # PHASE 1: AI REASONING - Strategist decides what features to create
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1: AI REASONING - Feature Strategy")
    logger.info("=" * 60)

    crew = create_feature_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

    # Run only the strategist task first
    strategist_task = crew.tasks[0]
    from crewai import Crew, Process
    strategist_crew = Crew(
        name="Strategist Only",
        agents=[crew.agents[0]],
        tasks=[strategist_task],
        process=Process.sequential,
        verbose=True,
    )
    strategist_result = strategist_crew.kickoff()

    # Extract tokens
    tokens = extract_tokens_from_crew_result(strategist_result)
    if tokens["total"] > 0:
        tracker.record_llm_call(
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            model=model_id,
        )

    # Parse strategist output
    try:
        strategy = _parse_strategist_json_output(
            str(strategist_result.raw) if hasattr(strategist_result, 'raw') else str(strategist_result),
            feat_dir,
            time_format=config.time_format
        )
        strategy.save(os.path.join(feat_dir, 'feature_strategy_decision.json'))
        logger.info(f"Strategist decision saved: {strategy.strategy_id}")
    except Exception as e:
        logger.error(f"Failed to parse strategist output: {e}")
        # Use EDA-AWARE fallback strategy instead of hardcoded defaults
        # This ensures the fallback strategy reflects actual data characteristics
        logger.info("Creating EDA-aware fallback strategy from actual EDA insights...")
        try:
            strategy = _create_eda_aware_fallback_strategy(eda_dir, seg_dir, time_format=config.time_format)
            logger.info(f"EDA-aware fallback created: {strategy.strategy_id}")
        except Exception as fallback_err:
            # Ultimate fallback if EDA loading fails too
            logger.warning(f"EDA-aware fallback failed ({fallback_err}), using minimal defaults")
            strategy = FeatureStrategyDecision(
                strategy_id=f"minimal_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                complexity_level=FeatureComplexity.MINIMAL,
                primary_emphasis=InteractionEmphasis.BALANCED,
                encoding_strategy=EncodingStrategy.TARGET_ENCODING,
                target_lags=[1, 2, 3] if config.time_format == 'year_month' else [1, 4, 13],
                feature_lags=[1],
                acf_informed=False,
                acf_reasoning="Minimal fallback - no EDA data available",
                rolling_windows=[3, 6] if config.time_format == 'year_month' else [4, 13],
                rolling_stats=['mean', 'std'],
                rolling_reasoning="Minimal rolling windows",
                seasonal_period=12 if config.time_format == 'year_month' else 52,
                fourier_order=2,
                has_strong_seasonality=False,
                seasonality_reasoning="Unknown - minimal fallback",
                include_trend_features=True,
                trend_strength=0.3,
                trend_reasoning="Default trend features",
                include_intermittency_features=False,
                intermittency_pct=0.0,
                intermittency_reasoning="Disabled in minimal fallback",
                include_changepoint_indicators=False,
                changepoint_dates=[],
                changepoint_reasoning="Disabled in minimal fallback",
            )
        strategy.save(os.path.join(feat_dir, 'feature_strategy_decision.json'))

    # -------------------------------------------------------------------------
    # PHASE 2: DETERMINISTIC EXECUTION - No LLM, pure Python
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2: DETERMINISTIC EXECUTION - Feature Creation")
    logger.info("=" * 60)

    exec_result = _run_deterministic_executor(
        config=config,
        strategy=strategy,
        feat_dir=feat_dir,
        seg_dir=seg_dir,
        eda_dir=eda_dir,
    )

    # Validate executor output
    _validate_executor_output(feat_dir)

    # Compute quality metrics for analyst
    quality_metrics = {
        'total_features': exec_result['n_features'],
        'lag_features': len([c for c in exec_result['feature_cols'] if '_lag_' in c]),
        'rolling_features': len([c for c in exec_result['feature_cols'] if '_rolling_' in c]),
        'seasonal_features': len([c for c in exec_result['feature_cols'] if 'fourier_' in c or 'seasonal_' in c]),
        # Count actual intermittency features created by create_intermittency_features()
        # These have patterns: demand_occurred, periods_since_nonzero, demand_freq_, avg_nonzero, std_nonzero
        'intermittency_features': len([c for c in exec_result['feature_cols']
                                        if any(p in c for p in ['demand_occurred', 'periods_since_nonzero',
                                                                'demand_freq_', 'avg_nonzero', 'std_nonzero'])]),
        'missing_pct': 0.0,  # Will be computed
        'constant_features': [],
        'high_corr_pairs': [],
    }

    # Compute actual quality metrics
    try:
        raw_assessment = validate_features_and_assess_quality(feat_dir, strategy)
        quality_metrics['missing_pct'] = raw_assessment.missing_value_pct
        quality_metrics['constant_features'] = raw_assessment.constant_features
        quality_metrics['high_corr_pairs'] = [(a, b, float(c)) for a, b, c in raw_assessment.high_correlation_pairs]
        quality_metrics['lag_features'] = raw_assessment.lag_features_count
        quality_metrics['rolling_features'] = raw_assessment.rolling_features_count
        quality_metrics['seasonal_features'] = raw_assessment.seasonal_features_count
        quality_metrics['intermittency_features'] = raw_assessment.intermittency_features_count
    except Exception as e:
        logger.warning(f"Could not compute full quality metrics: {e}")

    with open(os.path.join(feat_dir, 'feature_quality_metrics.json'), 'w') as f:
        json.dump(quality_metrics, f, indent=2)

    # -------------------------------------------------------------------------
    # PHASE 3: AI REASONING - Analyst validates quality
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 3: AI REASONING - Quality Assessment")
    logger.info("=" * 60)

    analyst_task = crew.tasks[1]
    analyst_crew = Crew(
        name="Analyst Only",
        agents=[crew.agents[1]],
        tasks=[analyst_task],
        process=Process.sequential,
        verbose=True,
    )
    analyst_result = analyst_crew.kickoff()

    # Extract tokens
    tokens = extract_tokens_from_crew_result(analyst_result)
    if tokens["total"] > 0:
        tracker.record_llm_call(
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            model=model_id,
        )

    # Parse analyst output
    try:
        import re
        analyst_text = str(analyst_result.raw) if hasattr(analyst_result, 'raw') else str(analyst_result)
        json_match = re.search(r'\{[\s\S]*\}', analyst_text)
        if json_match:
            assessment_data = json.loads(json_match.group())
        else:
            assessment_data = {}

        quality_assessment = FeatureQualityAssessment(
            overall_quality=assessment_data.get('overall_quality', 'good'),
            quality_score=raw_assessment.quality_score if 'raw_assessment' in dir() else 75.0,
            total_features_created=quality_metrics['total_features'],
            lag_features_count=quality_metrics['lag_features'],
            rolling_features_count=quality_metrics['rolling_features'],
            seasonal_features_count=quality_metrics['seasonal_features'],
            intermittency_features_count=quality_metrics['intermittency_features'],
            other_features_count=quality_metrics['total_features'] - quality_metrics['lag_features'] - quality_metrics['rolling_features'] - quality_metrics['seasonal_features'] - quality_metrics['intermittency_features'],
            missing_value_pct=quality_metrics['missing_pct'],
            infinite_value_pct=0.0,
            constant_features=quality_metrics['constant_features'],
            high_correlation_pairs=[(a, b, c) for a, b, c in quality_metrics['high_corr_pairs']],
            hypotheses_confirmed=assessment_data.get('hypotheses_confirmed', []),
            hypotheses_rejected=assessment_data.get('hypotheses_rejected', []),
            hypothesis_assessment=assessment_data.get('hypothesis_assessment', ''),
            concerns=assessment_data.get('concerns', []),
            recommendations=assessment_data.get('recommendations', []),
            reasoning_summary=assessment_data.get('reasoning_summary', ''),
        )
    except Exception as e:
        logger.warning(f"Could not parse analyst output: {e}")
        quality_assessment = FeatureQualityAssessment(
            overall_quality='good',
            quality_score=75.0,
            total_features_created=quality_metrics['total_features'],
            lag_features_count=quality_metrics['lag_features'],
            rolling_features_count=quality_metrics['rolling_features'],
            seasonal_features_count=quality_metrics['seasonal_features'],
            intermittency_features_count=quality_metrics['intermittency_features'],
            other_features_count=0,
            missing_value_pct=quality_metrics['missing_pct'],
            infinite_value_pct=0.0,
            constant_features=quality_metrics['constant_features'],
            high_correlation_pairs=[],
            hypotheses_confirmed=[],
            hypotheses_rejected=[],
            hypothesis_assessment='Assessment failed to parse',
            concerns=[],
            recommendations=[],
            reasoning_summary='Quality assessment completed with default values',
        )

    quality_assessment.save(os.path.join(feat_dir, 'feature_quality_assessment.json'))

    # -------------------------------------------------------------------------
    # PHASE 4: Create training manifest and context (DETERMINISTIC)
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 4: Create Training Manifest (Deterministic)")
    logger.info("=" * 60)

    try:
        _create_training_manifest_and_context(
            config=config,
            strategy=strategy,
            quality_assessment=quality_assessment,
            feat_dir=feat_dir,
            seg_dir=seg_dir,
            eda_dir=eda_dir,
        )
        logger.info("PHASE 4: Training manifest and context created successfully")
    except Exception as e:
        logger.error(f"PHASE 4 FAILED: Could not create training manifest: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to create training_manifest.csv: {e}") from e

    # Validate final outputs
    _validate_analyst_output(feat_dir)

    # -------------------------------------------------------------------------
    # PHASE 5: Run Feature Insights Crew (LLM-based comprehensive documentation)
    # -------------------------------------------------------------------------
    enable_insights = getattr(config.design, 'enable_insights_reports', False)
    if enable_insights:
        logger.info("=" * 60)
        logger.info("PHASE 5: Feature Engineering Insights Report (LLM)")
        logger.info("=" * 60)

        try:
            _run_feature_insights_crew(
                feat_dir=feat_dir,
                seg_dir=seg_dir,
                eda_dir=eda_dir,
                config=config,
                llm=llm,
            )
            logger.info("Feature insights report generated successfully")
        except FeatureInsightsError as e:
            logger.warning(f"Feature insights report generation failed: {e}")
            logger.warning("Continuing without insights report - this is non-critical")
        except Exception as e:
            logger.warning(f"Unexpected error in insights crew: {e}")
            logger.warning("Continuing without insights report - this is non-critical")
    else:
        logger.info("SKIPPING feature insights report (enable_insights_reports=False)")

    # End cost tracking
    cost_report = tracker.end_crew("Feature Engineering Crew", feat_dir)
    cost_report_path = os.path.join(feat_dir, "feature_cost.json")

    logger.info("=" * 60)
    logger.info("INTELLIGENT FEATURE ENGINEERING COMPLETE")
    logger.info(f"  Strategy: {strategy.strategy_id}")
    logger.info(f"  Quality: {quality_assessment.overall_quality} (score: {quality_assessment.quality_score:.1f})")
    logger.info(f"  Features: {quality_assessment.total_features_created}")
    logger.info(f"  Hypotheses confirmed: {len(quality_assessment.hypotheses_confirmed)}")
    logger.info(f"  Hypotheses rejected: {len(quality_assessment.hypotheses_rejected)}")
    logger.info("=" * 60)

    return FeatureCrewResult(
        feature_dir=feat_dir,
        feature_metadata_path=os.path.join(feat_dir, "feature_metadata.json"),
        feature_quality_summary_path=os.path.join(feat_dir, "feature_to_training_context.json"),
        feature_report_markdown_path=os.path.join(feat_dir, "strategist_output.md"),
        segmentation_integration_used=True,
        feature_pipeline_script_path=os.path.join(feat_dir, "feature_strategy_decision.json"),
        feature_to_training_context_path=os.path.join(feat_dir, "feature_to_training_context.json"),
        feature_deterministic_code_path=os.path.join(feat_dir, "feature_quality_metrics.json"),
        cost_report_path=cost_report_path,
        strategy_decision_path=os.path.join(feat_dir, 'feature_strategy_decision.json'),
        quality_assessment_path=os.path.join(feat_dir, 'feature_quality_assessment.json'),
    )
