# utils/context_manager.py
"""
Central Context Manager for Agentic Demand Forecasting Pipeline.

This module manages the flow of compact, targeted context between crews:
- EDA → Segmentation (eda_to_segmentation_context.json)
- EDA → Feature Engineering (eda_to_feature_context.json)
- Segmentation → Feature Engineering (segmentation_to_feature_context.json)
- Segmentation → Training (segmentation_to_training_context.json)
- Feature Engineering → Training (feature_to_training_context.json)
- Training → Diagnostics (training_to_diagnostic_context.json)

Each context file is designed to be:
- Small (< 15 KB) to avoid context window overflow
- Targeted (only information relevant to the receiving crew)
- Structured (consistent schemas for reliable parsing)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import pandas as pd


# =============================================================================
# CONTEXT SIZE LIMITS
# =============================================================================

MAX_CONTEXT_SIZE_KB = 20  # Maximum context file size in KB
WARN_CONTEXT_SIZE_KB = 15  # Warn if context exceeds this size
ENFORCE_SIZE_LIMIT = True  # If True, truncate contexts that exceed MAX_CONTEXT_SIZE_KB


# =============================================================================
# CONTEXT FILE NAMES (Standardized)
# =============================================================================

CONTEXT_FILES = {
    # EDA outputs
    "eda_to_segmentation": "eda_to_segmentation_context.json",
    "eda_to_feature": "eda_to_feature_context.json",

    # Feature Availability Detection outputs (runs between EDA and Feature Engineering)
    "feature_availability_to_feature": "feature_availability_to_feature_context.json",

    # Segmentation outputs
    "segmentation_to_feature": "segmentation_to_feature_context.json",
    "segmentation_to_training": "segmentation_to_training_context.json",

    # Feature Engineering outputs
    "feature_to_training": "feature_to_training_context.json",

    # Training outputs
    "training_to_diagnostic": "training_to_diagnostic_context.json",

    # Diagnostic outputs (for feedback loop)
    "diagnostic_feedback": "diagnostic_feedback_context.json",
}


# =============================================================================
# CONTEXT DATA CLASSES
# =============================================================================

@dataclass
class EDAToSegmentationContext:
    """
    Context from EDA crew to Segmentation crew.

    Contains ONLY segmentation-relevant information:
    - Cluster recommendations
    - Data characteristics for segmentation decisions
    - NO feature importance, NO transformation recommendations
    """
    # Metadata
    generated_at: str = ""
    n_series: int = 0
    time_format: str = ""

    # Segmentation guidance
    suggested_n_clusters: int = 5
    clustering_features: List[str] = field(default_factory=list)
    primary_segmentation_drivers: List[str] = field(default_factory=list)
    segment_strategy: str = ""

    # Data characteristics (aggregated, not key-level)
    intermittency_distribution: Dict[str, float] = field(default_factory=dict)
    volume_distribution: Dict[str, float] = field(default_factory=dict)
    seasonality_detected: bool = False
    trend_detected: bool = False

    # Warnings relevant to segmentation
    segmentation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EDAToFeatureContext:
    """
    Context from EDA crew to Feature Engineering crew.

    Contains ONLY feature engineering-relevant information:
    - Feature importance rankings
    - Lag and transformation recommendations
    - NO segmentation suggestions, NO cluster recommendations
    """
    # Metadata
    generated_at: str = ""
    n_series: int = 0
    time_format: str = ""
    prediction_key_cols: List[str] = field(default_factory=list)
    target_col: str = ""
    timestamp_col: str = ""

    # Feature recommendations
    high_importance_features: List[str] = field(default_factory=list)  # Top 15 max
    lag_recommendations: Dict[str, Any] = field(default_factory=dict)
    rolling_window_recommendations: Dict[str, Any] = field(default_factory=dict)
    seasonal_features: List[str] = field(default_factory=list)
    interaction_features: List[str] = field(default_factory=list)

    # Transformation recommendations
    transformation_recommendations: Dict[str, Any] = field(default_factory=dict)

    # Feature availability
    price_features_available: bool = False
    promo_features_available: bool = False
    external_features_available: List[str] = field(default_factory=list)

    # Granger causal features (if available)
    granger_causal_features: List[str] = field(default_factory=list)

    # Warnings relevant to feature engineering
    feature_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationToFeatureContext:
    """
    Context from Segmentation crew to Feature Engineering crew.

    Contains per-segment feature engineering guidance:
    - Segment profiles (stationarity, distribution, intermittency)
    - Feature guidance per segment
    - NO model recommendations, NO hyperparameter suggestions
    """
    # Metadata
    generated_at: str = ""
    n_segments: int = 0
    n_keys: int = 0

    # Segment profiles (aggregated characteristics per segment)
    segment_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {segment_id: {dominant_stationarity, dominant_distribution,
    #             dominant_intermittency, avg_observations, complexity}}

    # Feature guidance per segment
    feature_guidance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {segment_id: {lag_strategy, lag_values, rolling_windows,
    #             transformations, seasonal_features, interaction_features}}

    # Stationarity x Distribution rules (lookup table for feature decisions)
    stationarity_distribution_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Warnings
    feature_engineering_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationToTrainingContext:
    """
    Context from Segmentation crew to Training crew.

    Contains per-segment model guidance:
    - Model family recommendations
    - Hyperparameter strategies
    - Training approaches
    - NO feature guidance, NO transformation details
    """
    # Metadata
    generated_at: str = ""
    n_segments: int = 0
    n_model_groups: int = 0

    # Model guidance per segment
    model_guidance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {segment_id: {recommended_families, model_complexity,
    #             hyperparameter_strategy, training_approach, loss_function,
    #             modeling_mode, key_level_strategy}}

    # Segment summary for training decisions
    segment_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {segment_id: {segment_size, avg_observations,
    #             dominant_intermittency, dominant_volume_tier}}

    # Modeling mode distribution
    modeling_mode_distribution: Dict[str, int] = field(default_factory=dict)

    # Warnings
    training_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureToTrainingContext:
    """
    Context from Feature Engineering crew to Training crew.

    Contains feature summary and quality information:
    - Feature counts per segment
    - Quality scores
    - Data summary
    - NO raw feature lists, NO transformation details
    """
    # Metadata
    generated_at: str = ""
    n_model_groups: int = 0
    total_features_created: int = 0
    time_format: str = ""

    # Segment feature summary (compact)
    segment_feature_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {segment_id: {n_features, feature_categories,
    #             transformations_applied}}

    # Feature quality summary
    feature_quality_summary: Dict[str, Any] = field(default_factory=dict)
    # Structure: {overall_quality_score, segments_with_issues,
    #             correlation_warnings}

    # Data summary
    data_summary: Dict[str, int] = field(default_factory=dict)
    # Structure: {train_rows, val_rows, test_rows, feature_columns}

    # Warnings
    feature_quality_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingToDiagnosticContext:
    """
    Context from Training crew to Diagnostic crew.

    Contains model performance summary:
    - Per-segment model info and metrics
    - Validation and test performance
    - NO full hyperparameters, NO training history
    """
    # Metadata
    generated_at: str = ""
    n_model_groups: int = 0

    # Model summary per segment
    model_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {segment_id: {model_family, val_metrics, test_metrics,
    #             modeling_mode, segment_size}}

    # Overall performance summary
    overall_performance: Dict[str, Any] = field(default_factory=dict)
    # Structure: {val_wape, val_mape, test_wape, test_mape,
    #             best_segment, worst_segment}

    # Segments needing attention
    weak_segments: List[str] = field(default_factory=list)

    # Warnings
    training_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticFeedbackContext:
    """
    Context from Diagnostic crew for potential feedback loop.

    Contains actionable recommendations:
    - Segments to retrain
    - Feature engineering feedback
    - Model selection feedback
    """
    # Metadata
    generated_at: str = ""

    # Retraining recommendations
    needs_retraining: bool = False
    segments_to_retrain: List[str] = field(default_factory=list)
    retraining_reasons: Dict[str, str] = field(default_factory=dict)

    # Segmentation feedback
    segmentation_feedback: List[str] = field(default_factory=list)

    # Feature engineering feedback
    feature_engineering_feedback: List[str] = field(default_factory=list)

    # Model selection feedback
    model_selection_feedback: List[str] = field(default_factory=list)

    # Overall assessment
    overall_quality: str = ""  # "good", "acceptable", "needs_improvement"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# SAVE/LOAD UTILITIES
# =============================================================================

def save_context(
    context: Union[
        EDAToSegmentationContext,
        EDAToFeatureContext,
        SegmentationToFeatureContext,
        SegmentationToTrainingContext,
        FeatureToTrainingContext,
        TrainingToDiagnosticContext,
        DiagnosticFeedbackContext,
        Dict[str, Any],
    ],
    output_path: str,
    context_name: str = "",
) -> str:
    """
    Save context to JSON file with size validation and enforcement.

    Parameters
    ----------
    context : context dataclass or dict
        Context to save
    output_path : str
        Path to save JSON file
    context_name : str
        Name for logging purposes

    Returns
    -------
    str
        Path to saved file
    """
    # Convert to dict if dataclass
    if hasattr(context, 'to_dict'):
        context_dict = context.to_dict()
    elif isinstance(context, dict):
        context_dict = context
    else:
        context_dict = asdict(context)

    # Add timestamp if not present
    if "generated_at" not in context_dict or not context_dict["generated_at"]:
        context_dict["generated_at"] = datetime.now().isoformat()

    # Serialize to JSON
    json_str = json.dumps(context_dict, indent=2, default=str)

    # Check size and enforce limit if needed
    size_kb = len(json_str.encode('utf-8')) / 1024

    if size_kb > MAX_CONTEXT_SIZE_KB and ENFORCE_SIZE_LIMIT:
        # Truncate context to fit within limit
        context_dict = _truncate_context(context_dict, MAX_CONTEXT_SIZE_KB)
        json_str = json.dumps(context_dict, indent=2, default=str)
        new_size_kb = len(json_str.encode('utf-8')) / 1024
        print(f"  TRUNCATED: Context '{context_name}' reduced from {size_kb:.1f} KB to {new_size_kb:.1f} KB")
        size_kb = new_size_kb
    elif size_kb > MAX_CONTEXT_SIZE_KB:
        print(f"  WARNING: Context '{context_name}' exceeds max size: {size_kb:.1f} KB > {MAX_CONTEXT_SIZE_KB} KB")
    elif size_kb > WARN_CONTEXT_SIZE_KB:
        print(f"  NOTICE: Context '{context_name}' is large: {size_kb:.1f} KB")

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    print(f"  Saved {context_name}: {output_path} ({size_kb:.1f} KB)")

    return output_path


def _truncate_context(context_dict: Dict[str, Any], max_size_kb: float) -> Dict[str, Any]:
    """
    Truncate context dictionary to fit within size limit.

    Truncation strategy:
    1. Truncate lists to max 10 items
    2. Truncate nested dicts to max 5 keys
    3. Truncate string values to max 500 chars
    4. Repeat until within limit
    """
    import copy
    result = copy.deepcopy(context_dict)

    # Track truncation limits - start aggressive
    list_limit = 10
    dict_limit = 5
    string_limit = 500

    while True:
        json_str = json.dumps(result, indent=2, default=str)
        size_kb = len(json_str.encode('utf-8')) / 1024

        if size_kb <= max_size_kb:
            break

        # Reduce limits and re-truncate
        list_limit = max(3, list_limit - 2)
        dict_limit = max(3, dict_limit - 1)
        string_limit = max(100, string_limit - 100)

        result = _apply_truncation(result, list_limit, dict_limit, string_limit)

        # Safety: if we've reduced to minimum limits, stop
        if list_limit <= 3 and dict_limit <= 3 and string_limit <= 100:
            break

    # Add truncation marker
    result["_truncated"] = True
    result["_truncation_note"] = "Context truncated to prevent overflow. See full artifacts on disk."

    return result


def _apply_truncation(
    obj: Any,
    list_limit: int,
    dict_limit: int,
    string_limit: int,
) -> Any:
    """Recursively apply truncation limits."""
    if isinstance(obj, dict):
        # Truncate dict keys
        keys = list(obj.keys())
        if len(keys) > dict_limit:
            # Keep important keys
            important_keys = [k for k in keys if k.startswith("_") or k in [
                "generated_at", "n_series", "n_segments", "n_model_groups",
                "needs_retraining", "overall_quality", "warnings"
            ]]
            other_keys = [k for k in keys if k not in important_keys]
            keys_to_keep = important_keys + other_keys[:dict_limit - len(important_keys)]
            obj = {k: obj[k] for k in keys_to_keep if k in obj}

        # Recursively truncate values
        return {k: _apply_truncation(v, list_limit, dict_limit, string_limit) for k, v in obj.items()}

    elif isinstance(obj, list):
        # Truncate list
        if len(obj) > list_limit:
            obj = obj[:list_limit]
        # Recursively truncate items
        return [_apply_truncation(item, list_limit, dict_limit, string_limit) for item in obj]

    elif isinstance(obj, str):
        # Truncate string
        if len(obj) > string_limit:
            return obj[:string_limit] + "..."
        return obj

    else:
        return obj


def load_context(
    context_path: str,
    context_name: str = "",
    required: bool = False,
) -> Dict[str, Any]:
    """
    Load context from JSON file.

    Parameters
    ----------
    context_path : str
        Path to JSON file
    context_name : str
        Name for logging purposes
    required : bool
        If True, raise error if file not found

    Returns
    -------
    dict
        Loaded context (empty dict if not found and not required)
    """
    if not os.path.exists(context_path):
        if required:
            raise FileNotFoundError(f"Required context not found: {context_path}")
        print(f"  Context not found (optional): {context_path}")
        return {}

    with open(context_path, 'r', encoding='utf-8') as f:
        context = json.load(f)

    size_kb = os.path.getsize(context_path) / 1024
    print(f"  Loaded {context_name}: {context_path} ({size_kb:.1f} KB)")

    return context


# =============================================================================
# CONTEXT PATH HELPERS
# =============================================================================

def get_context_path(output_dir: str, context_type: str) -> str:
    """
    Get the standardized path for a context file.

    Parameters
    ----------
    output_dir : str
        Directory where context should be saved
    context_type : str
        Type of context (key from CONTEXT_FILES)

    Returns
    -------
    str
        Full path to context file
    """
    if context_type not in CONTEXT_FILES:
        raise ValueError(f"Unknown context type: {context_type}. Valid types: {list(CONTEXT_FILES.keys())}")

    return os.path.join(output_dir, CONTEXT_FILES[context_type])


def validate_context_exists(
    context_path: str,
    context_name: str,
    raise_on_missing: bool = True,
) -> bool:
    """
    Validate that a context file exists and is valid JSON.

    Parameters
    ----------
    context_path : str
        Path to context file
    context_name : str
        Name for error messages
    raise_on_missing : bool
        If True, raise error if missing

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if not os.path.exists(context_path):
        if raise_on_missing:
            raise FileNotFoundError(f"Required context missing: {context_name} at {context_path}")
        return False

    try:
        with open(context_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        if raise_on_missing:
            raise ValueError(f"Invalid JSON in context {context_name}: {e}")
        return False


# =============================================================================
# CONTEXT SUMMARY UTILITIES
# =============================================================================

def summarize_context_for_prompt(
    context: Dict[str, Any],
    max_items: int = 5,
    max_chars: int = 500,
) -> str:
    """
    Create a compact summary of context for inclusion in prompts.

    Parameters
    ----------
    context : dict
        Context dictionary
    max_items : int
        Max items to show for lists
    max_chars : int
        Max characters for string values

    Returns
    -------
    str
        Compact summary string
    """
    lines = []

    for key, value in context.items():
        if key in ["generated_at"]:
            continue  # Skip metadata

        if isinstance(value, list):
            if len(value) > max_items:
                summary = f"{key}: [{', '.join(str(v)[:50] for v in value[:max_items])}, ...+{len(value)-max_items} more]"
            else:
                summary = f"{key}: {value}"
        elif isinstance(value, dict):
            n_keys = len(value)
            if n_keys > max_items:
                sample_keys = list(value.keys())[:max_items]
                summary = f"{key}: {{{', '.join(sample_keys)}, ...+{n_keys-max_items} more}}"
            else:
                summary = f"{key}: {list(value.keys())}"
        elif isinstance(value, str) and len(value) > max_chars:
            summary = f"{key}: {value[:max_chars]}..."
        else:
            summary = f"{key}: {value}"

        lines.append(summary)

    return "\n".join(lines)


def get_context_size_report(artifact_dir: str) -> Dict[str, float]:
    """
    Get sizes of all context files in a directory.

    Parameters
    ----------
    artifact_dir : str
        Base artifact directory

    Returns
    -------
    dict
        Mapping of context name to size in KB
    """
    sizes = {}

    subdirs = ["eda_output", "seg_output", "feature_output", "model_artifacts", "diagnostics"]

    for subdir in subdirs:
        full_dir = os.path.join(artifact_dir, subdir)
        if not os.path.exists(full_dir):
            continue

        for filename in os.listdir(full_dir):
            if filename.endswith("_context.json"):
                path = os.path.join(full_dir, filename)
                sizes[filename] = os.path.getsize(path) / 1024

    return sizes


def print_context_flow_summary(artifact_dir: str) -> None:
    """
    Print a summary of context flow for debugging.

    Parameters
    ----------
    artifact_dir : str
        Base artifact directory
    """
    print("\n" + "=" * 60)
    print("CONTEXT FLOW SUMMARY")
    print("=" * 60)

    sizes = get_context_size_report(artifact_dir)

    if not sizes:
        print("No context files found.")
        return

    total_size = sum(sizes.values())

    for name, size in sorted(sizes.items()):
        status = "OK" if size < WARN_CONTEXT_SIZE_KB else ("WARN" if size < MAX_CONTEXT_SIZE_KB else "LARGE")
        print(f"  [{status}] {name}: {size:.1f} KB")

    print("-" * 60)
    print(f"  Total context size: {total_size:.1f} KB")
    print("=" * 60)


# =============================================================================
# FOCUSED CONTEXT GENERATION UTILITIES
# =============================================================================

def create_eda_to_segmentation_context(
    eda_context: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create focused EDA context for Segmentation crew.

    This extracts ONLY segmentation-relevant information:
    - Cluster recommendations
    - Data characteristics for segmentation decisions
    - NO feature importance, NO transformation recommendations

    Parameters
    ----------
    eda_context : dict
        Full EDA context
    output_dir : str
        Output directory for context file

    Returns
    -------
    str
        Path to saved context file
    """
    seg_context = EDAToSegmentationContext(
        generated_at=datetime.now().isoformat(),
        n_series=eda_context.get("metadata", {}).get("n_series", 0),
        time_format=eda_context.get("metadata", {}).get("time_format", ""),

        # Segmentation guidance
        suggested_n_clusters=eda_context.get("segmentation_recommendations", {}).get("suggested_n_clusters", 5),
        clustering_features=eda_context.get("segmentation_recommendations", {}).get("clustering_features", [])[:10],
        primary_segmentation_drivers=eda_context.get("segmentation_recommendations", {}).get("primary_segmentation_drivers", []),
        segment_strategy=eda_context.get("segmentation_recommendations", {}).get("segment_strategy", ""),

        # Data characteristics (aggregated)
        intermittency_distribution=eda_context.get("data_characteristics", {}).get("intermittency_distribution", {}),
        volume_distribution=eda_context.get("data_characteristics", {}).get("volume_distribution", {}),
        seasonality_detected=eda_context.get("data_characteristics", {}).get("seasonality_detected", False),
        trend_detected=eda_context.get("data_characteristics", {}).get("trend_detected", False),

        # Warnings
        segmentation_warnings=[
            w for w in eda_context.get("warnings_and_notes", [])
            if any(kw in w.lower() for kw in ["segment", "cluster", "group", "lumpy", "intermittent"])
        ][:5],
    )

    output_path = get_context_path(output_dir, "eda_to_segmentation")
    return save_context(seg_context, output_path, "eda_to_segmentation")


def create_eda_to_feature_context(
    eda_context: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create focused EDA context for Feature Engineering crew.

    This extracts ONLY feature engineering-relevant information:
    - Feature importance rankings
    - Lag and transformation recommendations
    - NO segmentation suggestions, NO cluster recommendations

    Parameters
    ----------
    eda_context : dict
        Full EDA context
    output_dir : str
        Output directory for context file

    Returns
    -------
    str
        Path to saved context file
    """
    feat_recs = eda_context.get("feature_recommendations", {})
    metadata = eda_context.get("metadata", {})

    feat_context = EDAToFeatureContext(
        generated_at=datetime.now().isoformat(),
        n_series=metadata.get("n_series", 0),
        time_format=metadata.get("time_format", ""),
        prediction_key_cols=metadata.get("prediction_key_cols", []),
        target_col=metadata.get("target_col", ""),
        timestamp_col=metadata.get("timestamp_col", ""),

        # Feature recommendations
        high_importance_features=feat_recs.get("high_importance_features", [])[:15],
        lag_recommendations=feat_recs.get("lag_recommendations", {}),
        rolling_window_recommendations=feat_recs.get("rolling_window_recommendations", {}),
        seasonal_features=feat_recs.get("seasonal_features", []),
        interaction_features=feat_recs.get("interaction_features", []),

        # Transformation recommendations
        transformation_recommendations=feat_recs.get("transformation_recommendations", {}),

        # Feature availability
        price_features_available=feat_recs.get("price_features_available", False),
        promo_features_available=feat_recs.get("promo_features_available", False),
        external_features_available=feat_recs.get("external_features_available", [])[:10],

        # Granger causal features
        granger_causal_features=feat_recs.get("granger_causal_features", [])[:5],

        # Warnings
        feature_warnings=[
            w for w in eda_context.get("warnings_and_notes", [])
            if any(kw in w.lower() for kw in ["feature", "lag", "transform", "correlation", "season"])
        ][:5],
    )

    output_path = get_context_path(output_dir, "eda_to_feature")
    return save_context(feat_context, output_path, "eda_to_feature")


def create_segmentation_to_feature_context(
    segmentation_context: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create focused Segmentation context for Feature Engineering crew.

    This extracts ONLY feature engineering-relevant information:
    - Segment profiles (stationarity, distribution, intermittency)
    - Feature guidance per segment
    - NO model recommendations, NO hyperparameter suggestions

    Parameters
    ----------
    segmentation_context : dict
        Full segmentation context
    output_dir : str
        Output directory for context file

    Returns
    -------
    str
        Path to saved context file
    """
    metadata = segmentation_context.get("metadata", {})

    seg_feat_context = SegmentationToFeatureContext(
        generated_at=datetime.now().isoformat(),
        n_segments=metadata.get("n_segments", 0),
        n_keys=metadata.get("n_keys", 0),

        # Segment profiles (only relevant fields for feature engineering)
        segment_profiles={
            seg_id: {
                "segment_size": profile.get("segment_size", 0),
                "modeling_mode": profile.get("modeling_mode", "cluster_model"),
                "model_level": profile.get("model_level", "segment"),
                "dominant_intermittency": profile.get("dominant_intermittency", "unknown"),
                "dominant_stationarity": profile.get("dominant_stationarity", "unknown"),
                "dominant_distribution": profile.get("dominant_distribution", "unknown"),
                "stationarity_mix": profile.get("stationarity_mix", {}),
                "distribution_mix": profile.get("distribution_mix", {}),
                "avg_observations": profile.get("avg_observations", 0),
                "avg_zero_frac": profile.get("avg_zero_frac", 0),
                "avg_cv": profile.get("avg_cv", 0),
                "complexity": profile.get("complexity", "moderate"),
            }
            for seg_id, profile in segmentation_context.get("segment_profiles", {}).items()
        },

        # Feature guidance per segment
        feature_guidance=segmentation_context.get("feature_guidance", {}),

        # Stationarity x Distribution rules
        stationarity_distribution_rules=segmentation_context.get("stationarity_x_distribution_rules", {}),

        # Warnings
        feature_engineering_warnings=segmentation_context.get("warnings", [])[:5],
    )

    output_path = get_context_path(output_dir, "segmentation_to_feature")
    return save_context(seg_feat_context, output_path, "segmentation_to_feature")


def create_segmentation_to_training_context(
    segmentation_context: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create focused Segmentation context for Training crew.

    This extracts ONLY model training-relevant information:
    - Model family recommendations per segment
    - Hyperparameter strategies
    - Training approaches
    - NO feature guidance, NO transformation details

    Parameters
    ----------
    segmentation_context : dict
        Full segmentation context
    output_dir : str
        Output directory for context file

    Returns
    -------
    str
        Path to saved context file
    """
    metadata = segmentation_context.get("metadata", {})

    seg_train_context = SegmentationToTrainingContext(
        generated_at=datetime.now().isoformat(),
        n_segments=metadata.get("n_segments", 0),
        n_model_groups=metadata.get("n_model_groups", 0),

        # Model guidance per segment
        model_guidance=segmentation_context.get("model_guidance", {}),

        # Segment summary for training decisions (compact)
        segment_summary={
            seg_id: {
                "segment_size": profile.get("segment_size", 0),
                "avg_observations": profile.get("avg_observations", 0),
                "dominant_intermittency": profile.get("dominant_intermittency", "unknown"),
                "dominant_volume_tier": profile.get("dominant_volume_tier", "vol_mid"),
            }
            for seg_id, profile in segmentation_context.get("segment_profiles", {}).items()
        },

        # Modeling mode distribution
        modeling_mode_distribution={},  # Will be computed from segment profiles

        # Warnings
        training_warnings=[
            w for w in segmentation_context.get("warnings", [])
            if any(kw in w.lower() for kw in ["model", "train", "hyperparameter", "overfit"])
        ][:5],
    )

    # Compute modeling mode distribution
    for profile in segmentation_context.get("segment_profiles", {}).values():
        mode = profile.get("modeling_mode", "cluster_model")
        seg_train_context.modeling_mode_distribution[mode] = \
            seg_train_context.modeling_mode_distribution.get(mode, 0) + 1

    output_path = get_context_path(output_dir, "segmentation_to_training")
    return save_context(seg_train_context, output_path, "segmentation_to_training")


def create_feature_to_training_context(
    feature_context: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create focused Feature Engineering context for Training crew.

    This extracts ONLY training-relevant information:
    - Feature counts per segment
    - Quality scores
    - Data summary
    - NO raw feature lists, NO transformation details

    Parameters
    ----------
    feature_context : dict
        Full feature context
    output_dir : str
        Output directory for context file

    Returns
    -------
    str
        Path to saved context file
    """
    metadata = feature_context.get("metadata", {})

    feat_train_context = FeatureToTrainingContext(
        generated_at=datetime.now().isoformat(),
        n_model_groups=metadata.get("n_model_groups", 0),
        total_features_created=metadata.get("total_features_created", 0),
        time_format=metadata.get("time_format", ""),

        # Segment feature summary (compact)
        segment_feature_summary={
            seg_id: {
                "n_features": summary.get("n_features", 0),
                "feature_categories": summary.get("feature_categories", {}),
                "transformations_applied": summary.get("transformations_applied", []),
            }
            for seg_id, summary in feature_context.get("segment_feature_summary", {}).items()
        },

        # Feature quality summary
        feature_quality_summary=feature_context.get("feature_quality_summary", {}),

        # Data summary
        data_summary=feature_context.get("data_summary", {}),

        # Warnings
        feature_quality_warnings=feature_context.get("warnings", [])[:5],
    )

    output_path = get_context_path(output_dir, "feature_to_training")
    return save_context(feat_train_context, output_path, "feature_to_training")


def create_training_to_diagnostic_context(
    training_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create focused Training context for Diagnostic crew.

    This extracts ONLY diagnostic-relevant information:
    - Per-segment model info and metrics
    - Validation and test performance
    - NO full hyperparameters, NO training history

    Parameters
    ----------
    training_results : dict
        Training results with model specs and metrics
    output_dir : str
        Output directory for context file

    Returns
    -------
    str
        Path to saved context file
    """
    per_model_group = training_results.get("per_model_group", {})

    # Build model summary
    model_summary = {}
    weak_segments = []
    overall_val_wape = 0
    overall_val_mape = 0
    overall_test_wape = 0
    overall_test_mape = 0
    n_groups = len(per_model_group)

    best_segment = None
    best_val_wape = float('inf')
    worst_segment = None
    worst_val_wape = 0

    for seg_id, specs in per_model_group.items():
        metrics = specs.get("metric_values", {})
        val_metrics = metrics.get("val", {})
        test_metrics = metrics.get("test", {})

        model_summary[seg_id] = {
            "model_family": specs.get("model_family", "unknown"),
            "val_metrics": {
                "wape": val_metrics.get("wape", 0),
                "mape": val_metrics.get("mape", 0),
                "rmse": val_metrics.get("rmse", 0),
            },
            "test_metrics": {
                "wape": test_metrics.get("wape", 0),
                "mape": test_metrics.get("mape", 0),
                "rmse": test_metrics.get("rmse", 0),
            },
            "modeling_mode": specs.get("modeling_mode", "cluster_model"),
            "segment_size": specs.get("segment_coverage", {}).get("n_keys", 0),
        }

        # Track weak segments (WAPE > 30%)
        val_wape = val_metrics.get("wape", 0)
        if val_wape > 0.30:
            weak_segments.append(seg_id)

        # Track best/worst
        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_segment = seg_id
        if val_wape > worst_val_wape:
            worst_val_wape = val_wape
            worst_segment = seg_id

        overall_val_wape += val_metrics.get("wape", 0)
        overall_val_mape += val_metrics.get("mape", 0)
        overall_test_wape += test_metrics.get("wape", 0)
        overall_test_mape += test_metrics.get("mape", 0)

    # Average metrics
    if n_groups > 0:
        overall_val_wape /= n_groups
        overall_val_mape /= n_groups
        overall_test_wape /= n_groups
        overall_test_mape /= n_groups

    train_diag_context = TrainingToDiagnosticContext(
        generated_at=datetime.now().isoformat(),
        n_model_groups=n_groups,
        model_summary=model_summary,
        overall_performance={
            "val_wape": overall_val_wape,
            "val_mape": overall_val_mape,
            "test_wape": overall_test_wape,
            "test_mape": overall_test_mape,
            "best_segment": best_segment,
            "worst_segment": worst_segment,
        },
        weak_segments=weak_segments,
        training_warnings=[],
    )

    output_path = get_context_path(output_dir, "training_to_diagnostic")
    return save_context(train_diag_context, output_path, "training_to_diagnostic")


def create_diagnostic_feedback_context(
    diagnostics_summary: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create feedback context from Diagnostic crew for potential feedback loop.

    This extracts actionable recommendations:
    - Segments to retrain
    - Feature engineering feedback
    - Model selection feedback

    Parameters
    ----------
    diagnostics_summary : dict
        Diagnostics summary with recommendations
    output_dir : str
        Output directory for context file

    Returns
    -------
    str
        Path to saved context file
    """
    feedback_recs = diagnostics_summary.get("feedback_recommendations", {})
    flags = diagnostics_summary.get("flags", {})

    feedback_context = DiagnosticFeedbackContext(
        generated_at=datetime.now().isoformat(),

        # Retraining recommendations
        needs_retraining=flags.get("needs_retraining", False),
        segments_to_retrain=flags.get("groups_to_retrain", []),
        retraining_reasons={
            seg: "High WAPE or degraded performance"
            for seg in flags.get("groups_to_retrain", [])
        },

        # Segmentation feedback
        segmentation_feedback=feedback_recs.get("segmentation", [])[:5],

        # Feature engineering feedback
        feature_engineering_feedback=feedback_recs.get("feature_engineering", [])[:5],

        # Model selection feedback
        model_selection_feedback=feedback_recs.get("model_selection", [])[:5],

        # Overall assessment
        overall_quality=_determine_overall_quality(diagnostics_summary),
    )

    output_path = get_context_path(output_dir, "diagnostic_feedback")
    return save_context(feedback_context, output_path, "diagnostic_feedback")


def _determine_overall_quality(diagnostics_summary: Dict[str, Any]) -> str:
    """Determine overall quality based on diagnostics."""
    global_summary = diagnostics_summary.get("global_summary", {})

    # Get average WAPE
    test_wape = global_summary.get("test_wape", 0)

    if test_wape < 0.15:
        return "good"
    elif test_wape < 0.30:
        return "acceptable"
    else:
        return "needs_improvement"


# =============================================================================
# CONTEXT LOADING HELPERS
# =============================================================================

def load_eda_to_segmentation_context(artifact_dir: str) -> Dict[str, Any]:
    """Load EDA→Segmentation context if available."""
    eda_dir = os.path.join(artifact_dir, "eda_output")
    path = get_context_path(eda_dir, "eda_to_segmentation")
    return load_context(path, "eda_to_segmentation", required=False)


def load_eda_to_feature_context(artifact_dir: str) -> Dict[str, Any]:
    """Load EDA→Feature context if available."""
    eda_dir = os.path.join(artifact_dir, "eda_output")
    path = get_context_path(eda_dir, "eda_to_feature")
    return load_context(path, "eda_to_feature", required=False)


def load_segmentation_to_feature_context(artifact_dir: str) -> Dict[str, Any]:
    """Load Segmentation→Feature context if available."""
    seg_dir = os.path.join(artifact_dir, "seg_output")
    path = get_context_path(seg_dir, "segmentation_to_feature")
    return load_context(path, "segmentation_to_feature", required=False)


def load_segmentation_to_training_context(artifact_dir: str) -> Dict[str, Any]:
    """Load Segmentation→Training context if available."""
    seg_dir = os.path.join(artifact_dir, "seg_output")
    path = get_context_path(seg_dir, "segmentation_to_training")
    return load_context(path, "segmentation_to_training", required=False)


def load_feature_to_training_context(artifact_dir: str) -> Dict[str, Any]:
    """Load Feature→Training context if available."""
    feat_dir = os.path.join(artifact_dir, "feature_output")
    path = get_context_path(feat_dir, "feature_to_training")
    return load_context(path, "feature_to_training", required=False)


def load_training_to_diagnostic_context(artifact_dir: str) -> Dict[str, Any]:
    """Load Training→Diagnostic context if available."""
    model_dir = os.path.join(artifact_dir, "model_artifacts")
    path = get_context_path(model_dir, "training_to_diagnostic")
    return load_context(path, "training_to_diagnostic", required=False)


def load_diagnostic_feedback_context(artifact_dir: str) -> Dict[str, Any]:
    """Load Diagnostic Feedback context if available."""
    diag_dir = os.path.join(artifact_dir, "diagnostics")
    path = get_context_path(diag_dir, "diagnostic_feedback")
    return load_context(path, "diagnostic_feedback", required=False)
