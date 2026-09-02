# utils/crew_output_validator.py
"""
Crew Output Validation Utilities
================================

This module provides functions to validate that each crew has completed
successfully by checking for required output files.

Used by databricks_runner.py to:
1. Skip crews that have already completed
2. Delete incomplete output folders before retry

Required files per crew:
- EDA: per_key_metrics.csv, eda_summary.json, 3 context files
- Segmentation: per_key_with_segments.csv, segment_profiles.json, etc.
- Feature: train/val/test_features.csv, training_manifest.csv, etc.
- Training: final_model_specs.json, at least one .pkl model file
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of crew output validation."""
    crew_name: str
    is_complete: bool
    missing_files: List[str]
    existing_files: List[str]
    output_dir: str
    message: str

    def __str__(self) -> str:
        status = "COMPLETE" if self.is_complete else "INCOMPLETE"
        return f"{self.crew_name}: {status} - {self.message}"


# =============================================================================
# REQUIRED FILES PER CREW
# =============================================================================

EDA_REQUIRED_FILES = [
    # Executor outputs
    ("per_key_metrics.csv", "Per-series metrics for all time series"),
    ("eda_summary.json", "EDA summary statistics"),  # Fixed: was global_eda_summary.json
    # Analyst outputs (context files)
    ("eda_to_segmentation_context.json", "Context for Segmentation crew"),
    ("eda_to_feature_context.json", "Context for Feature Engineering crew"),
    ("eda_to_training_context.json", "Context for Training crew"),
]

SEGMENTATION_REQUIRED_FILES = [
    # Planner output
    ("clustering_strategy.json", "Adaptive clustering strategy"),
    # Executor outputs
    ("per_key_with_segments.csv", "Segment assignments for all series"),
    ("segment_profiles.json", "Segment profiles with statistics"),
    ("model_recommendations.json", "Model recommendations per segment"),
    ("feature_recommendations.json", "Feature recommendations per segment"),
    ("clustering_metrics.json", "Clustering quality metrics"),
    # Analyst outputs (context files)
    ("segmentation_to_feature_context.json", "Context for Feature Engineering crew"),
    ("segmentation_to_training_context.json", "Context for Training crew"),
]

FEATURE_AVAILABILITY_REQUIRED_FILES = [
    ("feature_availability_result.json", "Full FA detection result (per-feature classifications)"),
    ("feature_availability_to_feature_context.json", "Context for Feature crew — known / history_only / excluded lists"),
]

FEATURE_REQUIRED_FILES = [
    # Strategist output
    ("feature_strategy_decision.json", "AI reasoning decision with structured strategy"),
    # Executor outputs.  The train/val/test_features files are now
    # parquet by default but a legacy run could still have CSVs;
    # _validate_crew_output below matches either format for entries
    # listed by their bare base name (no extension).
    ("train_features", "Training features"),
    ("val_features", "Validation features"),
    ("test_features", "Test features"),
    ("feature_metadata.json", "Feature metadata"),
    # Analyst outputs
    ("training_manifest.csv", "Model-level assignments for training"),
    ("feature_to_training_context.json", "Context for Training crew"),
    ("feature_quality_assessment.json", "Quality assessment with hypothesis validation"),
]

TRAINING_REQUIRED_FILES = [
    # Planner output
    ("training_strategy.json", "Training strategy with per-segment loss functions"),
    # Executor outputs
    ("final_model_specs.json", "Model specifications with training results"),
    ("training_to_diagnostic_context.json", "Diagnostic context for downstream crews"),
    # Note: .pkl model files are checked separately
]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_eda_output(artifact_base: str) -> ValidationResult:
    """
    Validate that EDA crew has completed with all required files.

    Parameters
    ----------
    artifact_base : str
        Base path for artifacts (e.g., /Volumes/catalog/schema/artifacts)

    Returns
    -------
    ValidationResult
        Validation result with completion status
    """
    eda_dir = os.path.join(artifact_base, "eda_output")
    return _validate_crew_output("EDA", eda_dir, EDA_REQUIRED_FILES)


def validate_segmentation_output(artifact_base: str) -> ValidationResult:
    """
    Validate that Segmentation crew has completed with all required files.
    """
    seg_dir = os.path.join(artifact_base, "seg_output")
    return _validate_crew_output("Segmentation", seg_dir, SEGMENTATION_REQUIRED_FILES)


def validate_feature_output(artifact_base: str) -> ValidationResult:
    """
    Validate that Feature Engineering crew has completed with all required files.
    """
    feat_dir = os.path.join(artifact_base, "feature_output")
    return _validate_crew_output("Feature Engineering", feat_dir, FEATURE_REQUIRED_FILES)


def validate_feature_availability_output(artifact_base: str) -> ValidationResult:
    """
    Validate that Feature Availability detection has produced its artifacts.
    """
    fa_dir = os.path.join(artifact_base, "feature_availability_output")
    return _validate_crew_output(
        "Feature Availability", fa_dir, FEATURE_AVAILABILITY_REQUIRED_FILES
    )


def validate_training_output(artifact_base: str) -> ValidationResult:
    """
    Validate that Training crew has completed with all required files.

    Training has additional validation:
    - At least one .pkl model file must exist
    - final_model_specs.json must have valid content
    """
    model_dir = os.path.join(artifact_base, "model_artifacts")

    # First check basic files
    result = _validate_crew_output("Training", model_dir, TRAINING_REQUIRED_FILES)

    if not result.is_complete:
        return result

    # Additional validation: check for .pkl model files
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not pkl_files:
        result.is_complete = False
        result.missing_files.append("*.pkl (at least one model file)")
        result.message = f"Missing {len(result.missing_files)} files (no trained models found)"
        return result

    # Additional validation: check final_model_specs.json content
    specs_path = os.path.join(model_dir, "final_model_specs.json")
    try:
        with open(specs_path, 'r') as f:
            specs = json.load(f)

        # Check for valid model data
        models = specs.get('models', specs.get('model_groups', {}))
        if not models:
            result.is_complete = False
            result.message = "final_model_specs.json has no models"
            return result

        # Check for valid WAPE (not nan, not None)
        wape = specs.get('overall_val_wape', specs.get('overall_wape'))
        if wape is None or (isinstance(wape, float) and wape != wape):  # NaN check
            result.is_complete = False
            result.message = f"Training incomplete: WAPE is {wape}"
            return result

    except (json.JSONDecodeError, KeyError) as e:
        result.is_complete = False
        result.message = f"final_model_specs.json is invalid: {e}"
        return result

    result.message = f"Complete with {len(pkl_files)} trained models, WAPE={wape:.4f}"
    return result


def _validate_crew_output(
    crew_name: str,
    output_dir: str,
    required_files: List[Tuple[str, str]],
) -> ValidationResult:
    """
    Generic validation function for crew outputs.

    Parameters
    ----------
    crew_name : str
        Name of the crew (for logging)
    output_dir : str
        Path to the crew's output directory
    required_files : List[Tuple[str, str]]
        List of (filename, description) tuples

    Returns
    -------
    ValidationResult
        Validation result
    """
    missing = []
    existing = []

    # Check if directory exists
    if not os.path.exists(output_dir):
        return ValidationResult(
            crew_name=crew_name,
            is_complete=False,
            missing_files=[f[0] for f in required_files],
            existing_files=[],
            output_dir=output_dir,
            message=f"Output directory does not exist: {output_dir}",
        )

    # Check each required file.  Entries WITHOUT an extension are
    # matched leniently against {filename}.parquet OR {filename}.csv,
    # so the train/val/test_features intermediate files are accepted
    # in either format.  Entries WITH an extension (e.g.
    # "training_manifest.csv") are matched exactly as before.
    for filename, description in required_files:
        if "." not in os.path.basename(filename):
            # Format-agnostic lookup: check parquet then csv
            from utils.feature_io import features_intermediate_path
            actual = features_intermediate_path(output_dir, filename)
            if actual is not None and actual.stat().st_size > 0:
                existing.append(actual.name)
            elif actual is not None:
                missing.append(f"{actual.name} (empty)")
            else:
                missing.append(f"{filename}.[parquet|csv]")
            continue
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            # Check file is not empty
            if os.path.getsize(filepath) > 0:
                existing.append(filename)
            else:
                missing.append(f"{filename} (empty)")
        else:
            missing.append(filename)

    is_complete = len(missing) == 0

    if is_complete:
        message = f"All {len(required_files)} required files present"
    else:
        message = f"Missing {len(missing)} of {len(required_files)} files: {missing[:3]}{'...' if len(missing) > 3 else ''}"

    return ValidationResult(
        crew_name=crew_name,
        is_complete=is_complete,
        missing_files=missing,
        existing_files=existing,
        output_dir=output_dir,
        message=message,
    )


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

def delete_crew_output(artifact_base: str, crew: str) -> bool:
    """
    Delete the output directory for a specific crew.

    Parameters
    ----------
    artifact_base : str
        Base path for artifacts
    crew : str
        Crew name: 'eda', 'segmentation', 'feature', or 'training'

    Returns
    -------
    bool
        True if deletion succeeded, False otherwise
    """
    crew_dirs = {
        'eda': 'eda_output',
        'feature_availability': 'feature_availability_output',
        'segmentation': 'seg_output',
        'feature': 'feature_output',
        'training': 'model_artifacts',
    }

    if crew.lower() not in crew_dirs:
        logger.error(f"Unknown crew: {crew}")
        return False

    output_dir = os.path.join(artifact_base, crew_dirs[crew.lower()])

    if not os.path.exists(output_dir):
        logger.info(f"{crew} output directory does not exist: {output_dir}")
        return True

    try:
        logger.warning(f"Deleting incomplete {crew} output: {output_dir}")
        shutil.rmtree(output_dir)
        logger.info(f"Successfully deleted: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {output_dir}: {e}")
        return False


def validate_all_crews(artifact_base: str) -> Dict[str, ValidationResult]:
    """
    Validate all crew outputs and return results.

    Parameters
    ----------
    artifact_base : str
        Base path for artifacts

    Returns
    -------
    Dict[str, ValidationResult]
        Dictionary mapping crew name to validation result
    """
    return {
        'eda': validate_eda_output(artifact_base),
        'feature_availability': validate_feature_availability_output(artifact_base),
        'segmentation': validate_segmentation_output(artifact_base),
        'feature': validate_feature_output(artifact_base),
        'training': validate_training_output(artifact_base),
    }


def get_crew_status_summary(artifact_base: str) -> str:
    """
    Get a formatted summary of all crew statuses.

    Parameters
    ----------
    artifact_base : str
        Base path for artifacts

    Returns
    -------
    str
        Formatted status summary
    """
    results = validate_all_crews(artifact_base)

    lines = ["=" * 60, "CREW OUTPUT STATUS SUMMARY", "=" * 60]

    for crew, result in results.items():
        status = "COMPLETE" if result.is_complete else "INCOMPLETE"
        icon = "[OK]" if result.is_complete else "[XX]"
        lines.append(f"{icon} {crew.upper():15} : {status:10} - {result.message}")

    lines.append("=" * 60)

    complete_count = sum(1 for r in results.values() if r.is_complete)
    lines.append(f"Total: {complete_count}/{len(results)} crews complete")

    return "\n".join(lines)


# =============================================================================
# SKIP/RETRY LOGIC FOR DATABRICKS RUNNER
# =============================================================================

def should_skip_crew(artifact_base: str, crew: str) -> Tuple[bool, str]:
    """
    Determine if a crew should be skipped (already complete) or needs to run.

    Parameters
    ----------
    artifact_base : str
        Base path for artifacts
    crew : str
        Crew name: 'eda', 'segmentation', 'feature', or 'training'

    Returns
    -------
    Tuple[bool, str]
        (should_skip, reason)
        - should_skip: True if crew output is complete and should be skipped
        - reason: Human-readable reason for the decision
    """
    validators = {
        'eda': validate_eda_output,
        'feature_availability': validate_feature_availability_output,
        'segmentation': validate_segmentation_output,
        'feature': validate_feature_output,
        'training': validate_training_output,
    }

    if crew.lower() not in validators:
        return False, f"Unknown crew: {crew}"

    result = validators[crew.lower()](artifact_base)

    if result.is_complete:
        return True, f"Output complete: {result.message}"
    else:
        return False, f"Output incomplete: {result.message}"


def prepare_crew_for_run(artifact_base: str, crew: str, force_rerun: bool = False) -> Tuple[bool, str]:
    """
    Prepare a crew for running by checking completion or cleaning up incomplete output.

    This is the main function to call from databricks_runner.py.

    Logic:
    1. If force_rerun=True, delete existing output and return ready to run
    2. If crew output is complete, skip (don't run)
    3. If crew output is incomplete, delete and return ready to run

    Parameters
    ----------
    artifact_base : str
        Base path for artifacts
    crew : str
        Crew name: 'eda', 'segmentation', 'feature', or 'training'
    force_rerun : bool
        If True, always delete and rerun regardless of completion status

    Returns
    -------
    Tuple[bool, str]
        (should_run, reason)
        - should_run: True if the crew should be executed
        - reason: Human-readable reason for the decision
    """
    should_skip, reason = should_skip_crew(artifact_base, crew)

    if force_rerun:
        # Delete existing output and run
        delete_crew_output(artifact_base, crew)
        return True, f"Force rerun requested, deleted existing output"

    if should_skip:
        # Crew already complete, skip
        return False, reason

    # Crew incomplete - delete existing output and run
    delete_crew_output(artifact_base, crew)
    return True, f"Deleted incomplete output, ready to run. Reason: {reason}"
