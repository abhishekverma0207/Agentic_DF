# utils/context_summarizer.py
"""
Intelligent Context Summarizer for Agentic Pipeline.

This module provides LLM-powered context summarization that intelligently
extracts key insights from upstream crew outputs before passing to downstream crews.

The summarizer uses a fast/cheap model (e.g., GPT-3.5 or Haiku) to analyze
raw JSON context and produce structured, actionable insights that the
executor agent can use for informed decision-making.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from crewai import LLM


# =============================================================================
# Summarizer Result Container
# =============================================================================

@dataclass
class ContextSummary:
    """Container for summarized context."""
    summary_text: str
    key_insights: List[str]
    recommendations: Dict[str, Any]
    raw_context_files: List[str]
    summarizer_model: str


# =============================================================================
# Context Loading Utilities
# =============================================================================

def _load_json_safely(path: str, max_chars: int = 8000) -> Tuple[Optional[Dict], str]:
    """Load JSON file safely with size limits."""
    if not os.path.exists(path):
        return None, ""

    try:
        with open(path, 'r') as f:
            content = f.read()

        # Truncate if too large
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [truncated]"

        data = json.loads(content[:content.rfind('}') + 1] if '... [truncated]' in content else content)
        return data, content
    except Exception as e:
        return None, f"Error loading {path}: {e}"


def _load_context_files(
    context_dir: str,
    context_filenames: List[str],
    max_chars_per_file: int = 4000,
) -> Tuple[Dict[str, Any], str]:
    """
    Load multiple context files from a directory.

    Returns tuple of (combined_data_dict, raw_text_for_summarization)
    """
    combined_data = {}
    raw_parts = []

    for filename in context_filenames:
        path = os.path.join(context_dir, filename)
        data, raw = _load_json_safely(path, max_chars_per_file)

        if data:
            combined_data[filename] = data
            raw_parts.append(f"=== {filename} ===\n{raw}")

    return combined_data, "\n\n".join(raw_parts)


# =============================================================================
# Summarization Prompts
# =============================================================================

SUMMARIZER_PROMPTS = {
    "eda_to_segmentation": """You are analyzing EDA (Exploratory Data Analysis) output to guide segmentation strategy.

CONTEXT TO ANALYZE:
{context}

EXTRACT AND SUMMARIZE:
1. **Data Characteristics**: Number of keys/series, time series length, data frequency
2. **Intermittency Distribution**: What % of keys are smooth/erratic/intermittent/lumpy?
3. **Clustering Features**: Key features identified for clustering (CV, zero_frac, etc.)
4. **Seasonality Patterns**: Weekly, monthly, yearly patterns detected (or none)
5. **Data Quality Warnings**: Any issues to be aware of

IMPORTANT: Do NOT recommend a specific number of clusters. EDA cannot determine the optimal
cluster count - that requires experimentation. Instead, suggest a RANGE based on data size.

OUTPUT FORMAT (be concise, <300 words):
## Key Insights
- Number of series: [N]
- Intermittency: [% smooth, % erratic, % intermittent, % lumpy]
- Dominant pattern: [which type dominates and implications]
- Seasonality/Trend: [detected or not]

## Clustering Guidance
- Suggested cluster range: [min-max based on data size, e.g., 2-5 for small data, 3-8 for larger]
- Primary clustering features: [list top features with reasoning]
- Clustering method: [kmeans/hierarchical recommendation]

## Data Quality Warnings
- [list any warnings from EDA that affect segmentation]

## Recommendations for Segmentation Agent
- [actionable guidance - the segmentation agent will determine optimal cluster count]
""",

    "segmentation_to_feature": """You are analyzing Segmentation output to guide Feature Engineering.

CONTEXT TO ANALYZE:
{context}

EXTRACT AND SUMMARIZE:
1. **Segment Overview**: Number of segments, model groups, keys per segment
2. **Segment Characteristics**: Dominant intermittency class, volume tier, complexity per segment
3. **Feature Recommendations**: What features are recommended for each segment type?
4. **Lag Patterns**: Recommended lags by segment/intermittency type
5. **Modeling Strategy**: Which segments need key-level vs cluster-level models?

OUTPUT FORMAT (be concise, <300 words):
## Segment Summary
- Total segments: [N], Model groups: [M]
- [brief description of segment distribution]

## Feature Engineering Strategy
- High-importance features: [list from EDA]
- Lag recommendations: [by intermittency type]
- Transformations needed: [log, diff, rolling, etc.]

## Per-Segment Notes
- [key insights for different segment types]
""",

    "feature_to_training": """You are analyzing Feature Engineering output to guide Model Training.

CONTEXT TO ANALYZE:
{context}

EXTRACT AND SUMMARIZE:
1. **Feature Set Overview**: Number of features per model group, key feature types
2. **Quality Metrics**: Feature quality scores, any issues detected
3. **Complexity Recommendations**: Simple/moderate/complex per segment
4. **Model Recommendations**: Suggested model families per segment type
5. **Training Considerations**: Data volume, special handling needed

OUTPUT FORMAT (be concise, <300 words):
## Feature Summary
- Model groups: [N], Average features per group: [M]
- Feature types: [lags, rolling, calendar, interactions, etc.]

## Training Strategy
- Model recommendations by complexity: [simple/moderate/complex mappings]
- Hyperparameter tuning: [enabled/disabled, scope]

## Special Considerations
- [any warnings, edge cases, or specific recommendations]
""",

    "training_to_diagnostic": """You are analyzing Model Training output to guide Diagnostics.

CONTEXT TO ANALYZE:
{context}

EXTRACT AND SUMMARIZE:
1. **Model Overview**: Number of models trained, model types used
2. **Performance Summary**: Validation metrics (RMSE, MAPE, etc.) by segment
3. **Best/Worst Performers**: Which segments performed well/poorly?
4. **Potential Issues**: Any training failures, convergence issues?
5. **Diagnostic Focus Areas**: Where should diagnostics focus?

OUTPUT FORMAT (be concise, <300 words):
## Training Summary
- Models trained: [N], Successful: [M]
- Model types: [list]

## Performance Overview
- Best performers: [segments/groups with best metrics]
- Areas of concern: [segments needing attention]

## Diagnostic Priorities
- [what to focus on in diagnostics]
- [specific metrics to evaluate]
""",
}


# =============================================================================
# Main Summarizer Function
# =============================================================================

def summarize_context_with_llm(
    context_dirs: Dict[str, str],
    context_type: str,
    llm: Optional[LLM] = None,
    fallback_to_heuristic: bool = True,
    verbose: bool = False,
) -> ContextSummary:
    """
    Intelligently summarize context using an LLM.

    Parameters
    ----------
    context_dirs : Dict[str, str]
        Mapping of context type to directory path, e.g.:
        {"eda": "/path/to/eda_output", "seg": "/path/to/seg_output"}
    context_type : str
        Type of summarization needed:
        - "eda_to_segmentation"
        - "segmentation_to_feature"
        - "feature_to_training"
        - "training_to_diagnostic"
    llm : LLM, optional
        CrewAI LLM to use for summarization. If None, uses heuristic fallback.
    fallback_to_heuristic : bool
        If True and LLM fails, fall back to simple heuristic extraction.
    verbose : bool
        If True, print progress messages.

    Returns
    -------
    ContextSummary
        Summarized context with key insights and recommendations.
    """
    # Define which files to load for each context type
    context_file_mapping = {
        "eda_to_segmentation": {
            "eda": ["eda_to_segmentation_context.json", "eda_context.json", "enhanced_eda_summary.json"],
        },
        "segmentation_to_feature": {
            "seg": ["segmentation_to_feature_context.json", "segmentation_context.json"],
            "eda": ["eda_to_feature_context.json", "eda_context.json"],
        },
        "feature_to_training": {
            "seg": ["segmentation_to_training_context.json", "segmentation_context.json"],
            "feat": ["feature_to_training_context.json", "feature_context.json"],
        },
        "training_to_diagnostic": {
            "model": ["training_to_diagnostic_context.json", "final_model_specs.json"],
            "seg": ["segmentation_context.json"],
        },
    }

    if context_type not in context_file_mapping:
        raise ValueError(f"Unknown context_type: {context_type}")

    # Load all relevant context files
    all_context = {}
    all_raw_text = []
    loaded_files = []

    for dir_key, filenames in context_file_mapping[context_type].items():
        if dir_key in context_dirs:
            data, raw = _load_context_files(context_dirs[dir_key], filenames)
            if data:
                all_context.update(data)
                all_raw_text.append(raw)
                loaded_files.extend([f for f in filenames if os.path.exists(os.path.join(context_dirs[dir_key], f))])

    if not all_context:
        return ContextSummary(
            summary_text="No context available from upstream crews.",
            key_insights=["No prior context found - using defaults"],
            recommendations={},
            raw_context_files=[],
            summarizer_model="none",
        )

    combined_raw = "\n\n".join(all_raw_text)

    # Try LLM summarization using CrewAI's LLM
    import sys
    print(f"[Summarizer] LLM provided: {llm is not None}, verbose: {verbose}", flush=True)

    if llm is not None:
        try:
            prompt = SUMMARIZER_PROMPTS[context_type].format(context=combined_raw)

            print(f"[Summarizer] Calling LLM to summarize {context_type} context...", flush=True)
            print(f"[Summarizer] Prompt length: {len(prompt)} chars", flush=True)
            sys.stdout.flush()

            # Use CrewAI LLM's call method - it accepts a string directly
            response = llm.call(messages=prompt)

            print(f"[Summarizer] LLM response received, type: {type(response)}", flush=True)

            # Extract the response text - response is typically a string
            if isinstance(response, str):
                summary_text = response
            elif hasattr(response, 'content'):
                summary_text = response.content
            else:
                summary_text = str(response)

            print(f"[Summarizer] Summary generated: {len(summary_text)} chars")

            # Extract key insights from the LLM response
            key_insights = _extract_bullet_points(summary_text)

            return ContextSummary(
                summary_text=summary_text,
                key_insights=key_insights,
                recommendations=_extract_recommendations(summary_text),
                raw_context_files=loaded_files,
                summarizer_model=getattr(llm, 'model', 'unknown'),
            )

        except Exception as e:
            print(f"[Summarizer] LLM call FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            if not fallback_to_heuristic:
                raise

    # Heuristic fallback if LLM not provided or fails
    print(f"[Summarizer] Using heuristic extraction for {context_type} context (LLM failed or not provided)", flush=True)

    return _heuristic_summarize(all_context, context_type, loaded_files, combined_raw)


def _extract_bullet_points(text: str) -> List[str]:
    """Extract bullet points from markdown text."""
    insights = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('- ') or line.startswith('* '):
            insights.append(line[2:].strip())
    return insights[:10]  # Limit to top 10


def _extract_recommendations(text: str) -> Dict[str, Any]:
    """Extract structured recommendations from summary text."""
    recs = {}

    # Try to find specific patterns
    if 'clusters:' in text.lower():
        # Extract cluster recommendation
        import re
        match = re.search(r'clusters?:\s*(\d+)', text, re.IGNORECASE)
        if match:
            recs['suggested_clusters'] = int(match.group(1))

    if 'features:' in text.lower() or 'features for' in text.lower():
        # Extract feature mentions
        match = re.search(r'features?[^:]*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
        if match:
            recs['key_features'] = [f.strip() for f in match.group(1).split(',')]

    return recs


def _heuristic_summarize(
    context: Dict[str, Any],
    context_type: str,
    loaded_files: List[str],
    raw_context: str = "",
) -> ContextSummary:
    """
    Heuristic-based summarization that extracts key statistics and patterns.

    The extracted insights are passed to the CrewAI context summarizer agent,
    which will then intelligently analyze them using the LLM.
    """
    insights = []
    recommendations = {}

    # Flatten nested context for easier extraction
    def _flatten(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            else:
                items[new_key] = v
        return dict(items)

    # Extract common patterns across context types
    for filename, data in context.items():
        if not isinstance(data, dict):
            continue

        # Look for common keys
        if 'n_segments' in data:
            insights.append(f"Segments: {data['n_segments']}")
        if 'n_clusters' in data:
            insights.append(f"Clusters: {data['n_clusters']}")
        if 'suggested_n_clusters' in data:
            recommendations['suggested_clusters'] = data['suggested_n_clusters']
            insights.append(f"Suggested clusters: {data['suggested_n_clusters']}")
        if 'intermittency_distribution' in data:
            dist = data['intermittency_distribution']
            insights.append(f"Intermittency: {dist}")
        if 'high_importance_features' in data:
            features = data['high_importance_features'][:5]
            recommendations['key_features'] = features
            insights.append(f"Top features: {features}")
        if 'lag_recommendations' in data:
            lags = data['lag_recommendations'].get('suggested_lags', [])
            recommendations['suggested_lags'] = lags
            insights.append(f"Suggested lags: {lags}")
        if 'model_groups' in data:
            n_groups = len(data['model_groups']) if isinstance(data['model_groups'], (list, dict)) else data['model_groups']
            insights.append(f"Model groups: {n_groups}")

        # Recursively check nested structures
        for key, value in data.items():
            if isinstance(value, dict):
                if 'total_segments' in value:
                    insights.append(f"Total segments: {value['total_segments']}")
                if 'total_keys' in value:
                    insights.append(f"Total keys: {value['total_keys']}")

    # Build summary text with extracted insights AND raw context for agent analysis
    summary_lines = [
        f"## Context Summary ({context_type})",
        "",
        "### Extracted Key Insights",
    ]
    for insight in insights[:10]:
        summary_lines.append(f"- {insight}")

    if recommendations:
        summary_lines.append("")
        summary_lines.append("### Extracted Recommendations")
        for key, value in recommendations.items():
            summary_lines.append(f"- {key}: {value}")

    # Include truncated raw context for the agent to analyze intelligently
    if raw_context:
        # Truncate raw context to reasonable size for agent consumption
        max_raw_chars = 6000
        truncated_raw = raw_context[:max_raw_chars]
        if len(raw_context) > max_raw_chars:
            truncated_raw += "\n... [truncated for brevity]"

        summary_lines.append("")
        summary_lines.append("### Raw Context Data (for detailed analysis)")
        summary_lines.append(truncated_raw)

    summary_lines.append("")
    summary_lines.append(f"*Source files: {', '.join(loaded_files)}*")

    return ContextSummary(
        summary_text="\n".join(summary_lines),
        key_insights=insights,
        recommendations=recommendations,
        raw_context_files=loaded_files,
        summarizer_model="heuristic+agent",
    )


# =============================================================================
# Convenience Functions for Each Crew
# =============================================================================

def summarize_for_segmentation(
    eda_dir: str,
    llm: Optional[LLM] = None,
    verbose: bool = False,
) -> ContextSummary:
    """Summarize EDA context for Segmentation crew."""
    return summarize_context_with_llm(
        context_dirs={"eda": eda_dir},
        context_type="eda_to_segmentation",
        llm=llm,
        verbose=verbose,
    )


def summarize_for_feature_engineering(
    eda_dir: str,
    seg_dir: str,
    llm: Optional[LLM] = None,
    verbose: bool = False,
) -> ContextSummary:
    """Summarize EDA and Segmentation context for Feature Engineering crew."""
    return summarize_context_with_llm(
        context_dirs={"eda": eda_dir, "seg": seg_dir},
        context_type="segmentation_to_feature",
        llm=llm,
        verbose=verbose,
    )


def summarize_for_training(
    seg_dir: str,
    feat_dir: str,
    llm: Optional[LLM] = None,
    verbose: bool = False,
) -> ContextSummary:
    """Summarize Segmentation and Feature context for Training crew."""
    return summarize_context_with_llm(
        context_dirs={"seg": seg_dir, "feat": feat_dir},
        context_type="feature_to_training",
        llm=llm,
        verbose=verbose,
    )


def summarize_for_diagnostics(
    model_dir: str,
    seg_dir: str,
    llm: Optional[LLM] = None,
    verbose: bool = False,
) -> ContextSummary:
    """Summarize Training and Segmentation context for Diagnostics crew."""
    return summarize_context_with_llm(
        context_dirs={"model": model_dir, "seg": seg_dir},
        context_type="training_to_diagnostic",
        llm=llm,
        verbose=verbose,
    )
