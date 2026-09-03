# crews/feature_availability_crew.py
"""
Feature Availability Detection Crew for Agentic Demand Forecasting.

This crew runs BETWEEN the EDA crew and the Feature Engineering crew.
It has a HYBRID architecture:
  - Phase 1 (DETERMINISTIC): Auto-detect history cutoff, classify features,
    generate frozen embedding specs. No LLM calls needed.
  - Phase 2 (AGENTIC / LLM): An LLM analyst reviews the detection results,
    validates the classifications, and produces a refined context JSON with
    strategic recommendations for the Feature Engineering crew.

The Phase 2 LLM step is optional (controlled by config). When disabled,
the deterministic results are passed directly to the Feature Engineering crew.

This crew works identically on:
  - Local execution (with AWS Bedrock LLM)
  - Databricks execution (with Databricks AIF LLM)

Usage:
    from crews.feature_availability_crew import run_feature_availability_crew

    result = run_feature_availability_crew(
        llm=llm,
        config=cfg,
        config_yaml_path='config/config.yaml',
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATA CLASS
# =============================================================================

@dataclass
class FeatureAvailabilityCrewResult:
    """Result from the Feature Availability Detection crew."""
    output_dir: str = ""
    result_path: str = ""
    context_path: str = ""
    summary_path: str = ""
    n_features_analyzed: int = 0
    n_known_in_future: int = 0
    n_history_only: int = 0
    n_partially_known: int = 0
    n_excluded: int = 0
    detected_cutoff: str = ""
    duration_seconds: float = 0.0
    success: bool = False
    error: str = ""


# =============================================================================
# DETERMINISTIC PIPELINE (Phase 1 — No LLM)
# =============================================================================

def _run_deterministic_detection(
    config: Any,
    source_df: Optional[pd.DataFrame] = None,
) -> "FeatureAvailabilityResult":
    """
    Run the deterministic feature availability detection.

    This is the core logic — no LLM calls, pure data inspection.

    Parameters
    ----------
    config : DemandForecastConfig
        Pipeline configuration
    source_df : pd.DataFrame, optional
        Pre-loaded source data

    Returns
    -------
    FeatureAvailabilityResult
        Detection results
    """
    from utils.feature_availability import run_feature_availability_from_config

    return run_feature_availability_from_config(
        config=config,
        source_df=source_df,
    )


# =============================================================================
# AGENTIC ANALYST (Phase 2 — Optional LLM Review)
# =============================================================================

def _run_llm_analyst(
    llm: Any,
    config: Any,
    detection_result: Any,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run the LLM analyst to review and enrich detection results.

    The analyst:
    1. Reviews the feature classification and validates it
    2. Identifies potential misclassifications
    3. Adds strategic recommendations (which frozen embeddings are most valuable)
    4. Produces an enriched context for the Feature Engineering crew

    Parameters
    ----------
    llm : LLM
        CrewAI-compatible LLM (Bedrock or Databricks AIF)
    config : DemandForecastConfig
        Pipeline configuration
    detection_result : FeatureAvailabilityResult
        Results from deterministic detection
    output_dir : str
        Output directory for analyst artifacts

    Returns
    -------
    Dict[str, Any]
        Enriched context with LLM recommendations
    """
    try:
        from crewai import Agent, Task, Crew, Process
        from utils.code_execution_tool import CodeExecutionTool
    except ImportError:
        logger.warning("CrewAI not available — skipping LLM analyst, using deterministic results only")
        return {}

    code_tool = CodeExecutionTool()
    code_tool.protected_paths = []  # No protected paths needed

    # Build the analyst prompt with detection results summary
    summary = detection_result.summary()
    known_features = detection_result.known_in_future[:20]
    history_features = detection_result.history_only[:20]
    partial_features = detection_result.partially_known[:10]

    analyst_backstory = f"""You are a Feature Availability Analyst for demand forecasting.

You have received the results of an automated feature availability detection pipeline.
Your job is to:
1. Review the classifications and flag any that look suspicious
2. Recommend which frozen embeddings are most valuable for forecasting
3. Suggest feature engineering strategies based on availability patterns

## Detection Results Summary:
{summary}

## Known-in-Future Features (use directly):
{json.dumps(known_features, indent=2)}

## History-Only Features (use as frozen embeddings):
{json.dumps(history_features, indent=2)}

## Partially-Known Features:
{json.dumps(partial_features, indent=2)}

## Instructions:
Execute the following Python code to read the full detection result and produce recommendations:

```python
import json
import os

# Read full detection result
result_path = os.path.join('{output_dir}', 'feature_availability_result.json')
with open(result_path) as f:
    result = json.load(f)

# Build enriched recommendations
recommendations = {{
    'analyst_review': 'completed',
    'classification_validated': True,
    'high_value_frozen_embeddings': [],
    'feature_engineering_notes': [],
    'warnings': [],
}}

# Identify high-value frozen embeddings
# Features with strong correlation to target are most valuable as embeddings
feature_details = result.get('feature_details', {{}})
for feat, info in feature_details.items():
    if info.get('classification') == 'history_only':
        hist_rate = info.get('history_useful_rate', 0)
        if hist_rate > 0.3:
            recommendations['high_value_frozen_embeddings'].append(feat)

# Add strategic notes
n_known = result.get('n_known_in_future', 0)
n_history = result.get('n_history_only', 0)
n_total = result.get('n_total_features', 0)

if n_history > n_known:
    recommendations['feature_engineering_notes'].append(
        'More features are history-only than known-in-future. '
        'Frozen embeddings will be critical for model quality.'
    )

if n_known == 0:
    recommendations['warnings'].append(
        'No features detected as known-in-future! '
        'Model will rely entirely on lag/rolling features and frozen embeddings.'
    )

# Save enriched recommendations
rec_path = os.path.join('{output_dir}', 'analyst_recommendations.json')
with open(rec_path, 'w') as f:
    json.dump(recommendations, f, indent=2)

print(f"Analyst recommendations saved to: {{rec_path}}")
print(f"High-value frozen embeddings: {{len(recommendations['high_value_frozen_embeddings'])}}")
print(f"Warnings: {{len(recommendations['warnings'])}}")
```
"""

    analyst = Agent(
        name="feature_availability_analyst",
        role="Feature Availability Analyst",
        goal="Review feature availability detection results and provide strategic recommendations",
        backstory=analyst_backstory,
        tools=[code_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        respect_context_window=False,
    )

    task = Task(
        name="analyze_feature_availability",
        description=(
            "Execute the Python code in your backstory instructions to analyze "
            "the feature availability detection results and produce enriched recommendations. "
            "Run the code EXACTLY as specified."
        ),
        agent=analyst,
        expected_output="Feature availability analysis complete with recommendations saved",
        output_file=os.path.join(output_dir, "analyst_task_output.md"),
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        crew.kickoff()
    except Exception as e:
        logger.warning(f"LLM analyst failed (non-critical): {e}")
        return {}

    # Load analyst recommendations if available
    rec_path = os.path.join(output_dir, 'analyst_recommendations.json')
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            return json.load(f)
    return {}


# =============================================================================
# MAIN CREW ENTRY POINT
# =============================================================================

def run_feature_availability_crew(
    llm: Any,
    config: Any,
    config_yaml_path: str,
    source_df: Optional[pd.DataFrame] = None,
    skip_llm_analyst: bool = False,
) -> FeatureAvailabilityCrewResult:
    """
    Run the Feature Availability Detection crew.

    This is the main entry point — works on both local and Databricks.

    Phase 1 (always): Deterministic detection — auto-detect cutoff,
    classify features, generate frozen embedding specs.

    Phase 2 (optional): LLM analyst reviews results and enriches context
    with strategic recommendations.

    Parameters
    ----------
    llm : LLM
        CrewAI-compatible LLM (Bedrock or Databricks AIF)
    config : DemandForecastConfig
        Pipeline configuration
    config_yaml_path : str
        Path to config YAML (for logging)
    source_df : pd.DataFrame, optional
        Pre-loaded source data
    skip_llm_analyst : bool
        Skip LLM analyst phase (use deterministic results only)

    Returns
    -------
    FeatureAvailabilityCrewResult
    """
    start_time = datetime.now()
    output_dir = os.path.join(config.artifact_base_path, 'feature_availability_output')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FEATURE AVAILABILITY DETECTION CREW")
    logger.info("=" * 60)
    logger.info(f"Config: {config_yaml_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Auto-detect features: {getattr(config.design, 'auto_detect_features', True)}")
    logger.info(f"LLM analyst: {'DISABLED' if skip_llm_analyst else 'ENABLED'}")
    logger.info("=" * 60)

    result = FeatureAvailabilityCrewResult(output_dir=output_dir)

    try:
        # =====================================================================
        # Phase 1: Deterministic Detection (No LLM)
        # =====================================================================
        logger.info("\n[Phase 1] Running deterministic feature availability detection...")

        detection_result = _run_deterministic_detection(
            config=config,
            source_df=source_df,
        )

        result.n_features_analyzed = detection_result.n_total_features
        result.n_known_in_future = detection_result.n_known_in_future
        result.n_history_only = detection_result.n_history_only
        result.n_partially_known = detection_result.n_partially_known
        result.n_excluded = detection_result.n_excluded
        result.detected_cutoff = detection_result.detected_history_cutoff
        result.result_path = os.path.join(output_dir, 'feature_availability_result.json')
        result.context_path = os.path.join(output_dir, 'feature_availability_to_feature_context.json')
        result.summary_path = os.path.join(output_dir, 'feature_availability_summary.txt')

        logger.info(f"Phase 1 complete: {detection_result.n_total_features} features analyzed")
        logger.info(f"  Known in future: {detection_result.n_known_in_future}")
        logger.info(f"  History only:    {detection_result.n_history_only}")
        logger.info(f"  Partially known: {detection_result.n_partially_known}")
        logger.info(f"  Excluded:        {detection_result.n_excluded}")

        # =====================================================================
        # Phase 2: LLM Analyst (Optional)
        # =====================================================================
        if not skip_llm_analyst and llm is not None:
            logger.info("\n[Phase 2] Running LLM analyst for enriched recommendations...")
            try:
                analyst_recs = _run_llm_analyst(
                    llm=llm,
                    config=config,
                    detection_result=detection_result,
                    output_dir=output_dir,
                )
                if analyst_recs:
                    # Merge analyst recommendations into context
                    context_path = result.context_path
                    if os.path.exists(context_path):
                        with open(context_path) as f:
                            context = json.load(f)
                        context['analyst_recommendations'] = analyst_recs
                        with open(context_path, 'w') as f:
                            json.dump(context, f, indent=2, default=str)
                        logger.info("Analyst recommendations merged into context")
                    else:
                        logger.warning("Context file not found — saving analyst recommendations separately")
                        rec_path = os.path.join(output_dir, 'analyst_recommendations.json')
                        with open(rec_path, 'w') as f:
                            json.dump(analyst_recs, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"LLM analyst phase failed (non-critical): {e}")
                logger.warning("Continuing with deterministic results only")
        else:
            logger.info("\n[Phase 2] LLM analyst skipped")

        # =====================================================================
        # Done
        # =====================================================================
        duration = (datetime.now() - start_time).total_seconds()
        result.duration_seconds = duration
        result.success = True

        logger.info("\n" + "=" * 60)
        logger.info("FEATURE AVAILABILITY DETECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Results: {result.result_path}")
        logger.info(f"Context: {result.context_path}")
        logger.info("=" * 60)

        return result

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        result.duration_seconds = duration
        result.success = False
        result.error = str(e)

        logger.error(f"Feature Availability Detection FAILED: {e}")
        logger.error(traceback.format_exc())

        return result
