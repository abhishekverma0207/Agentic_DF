# utils/context_schema.py
"""
Self-Documenting Context Schema for Inter-Crew Communication.

This module provides a standardized way to create context JSON files that:
1. Include schema documentation for each key
2. Are self-describing so downstream agents can understand the structure
3. Support flexible key naming while maintaining semantic meaning
4. Enable robust validation regardless of exact key names

USAGE:
------
Creating context (in upstream crew):
```python
from utils.context_schema import ContextBuilder

ctx = ContextBuilder("eda_to_segmentation")
ctx.add_field(
    key="clustering_recommendations",
    value={"algorithm": "kmeans", "k": 5},
    description="Recommended clustering configuration from EDA analysis",
    semantic_type="clustering_config",
    required_by=["segmentation_crew"]
)
ctx.save(output_path)
```

Consuming context (in downstream crew):
```python
from utils.context_schema import ContextReader

reader = ContextReader(context_path)
# Get by semantic type (preferred - robust to key naming)
clustering = reader.get_by_type("clustering_config")
# Or get by key name (fallback)
clustering = reader.get("clustering_recommendations")
# Get description for logging
desc = reader.get_description("clustering_recommendations")
```
"""

from __future__ import annotations
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC TYPES - Standard types for context fields
# =============================================================================

class SemanticTypes:
    """
    Standard semantic types for context fields.
    Using these enables robust lookup regardless of key naming.
    """
    # EDA Output Types
    LAG_RECOMMENDATIONS = "lag_recommendations"
    ROLLING_RECOMMENDATIONS = "rolling_recommendations"
    INTERMITTENCY_CONFIG = "intermittency_config"
    TRANSFORMATION_CONFIG = "transformation_config"
    CLUSTERING_CONFIG = "clustering_config"
    MODEL_FAMILY_HINTS = "model_family_hints"
    DATA_QUALITY_SUMMARY = "data_quality_summary"
    DEMAND_PATTERN_DISTRIBUTION = "demand_pattern_distribution"
    FEATURE_IMPORTANCE = "feature_importance"

    # Segmentation Output Types
    SEGMENT_DEFINITIONS = "segment_definitions"
    SEGMENT_MODEL_STRATEGY = "segment_model_strategy"
    CLUSTERING_QUALITY = "clustering_quality"
    SEGMENT_FEATURE_PRIORITIES = "segment_feature_priorities"

    # Feature Output Types
    FEATURE_SUMMARY = "feature_summary"
    FEATURE_CATEGORIES = "feature_categories"
    MODEL_LEVEL_SUMMARY = "model_level_summary"
    MODEL_RECOMMENDATIONS = "model_recommendations"
    TRAINING_INSTRUCTIONS = "training_instructions"
    FEATURE_SELECTION_CONFIG = "feature_selection_config"  # Model-level-aware feature selection
    QUALITY_ASSESSMENT = "quality_assessment"  # AI quality assessment from Feature Analyst
    EDA_INSIGHTS = "eda_insights"  # EDA insights for training strategy
    SEGMENT_TRAINING_STRATEGY = "segment_training_strategy"  # Per-segment training config
    SEGMENT_DIFFICULTY = "segment_difficulty"  # Segments by forecasting difficulty
    LOSS_RECOMMENDATION = "loss_recommendation"  # Loss function recommendations
    TRAINING_GUIDANCE = "training_guidance"  # Global training guidance

    # Training Output Types
    TRAINING_RESULTS = "training_results"
    MODEL_PERFORMANCE = "model_performance"
    UNDERPERFORMER_ANALYSIS = "underperformer_analysis"
    DIAGNOSTIC_PRIORITIES = "diagnostic_priorities"

    # Metadata Types
    METADATA = "metadata"
    FILE_PATHS = "file_paths"
    DATE_RANGES = "date_ranges"


@dataclass
class FieldSchema:
    """Schema definition for a single context field."""
    key: str
    description: str
    semantic_type: str
    value_type: str  # "dict", "list", "str", "int", "float", "bool"
    required: bool = False
    required_by: List[str] = field(default_factory=list)
    example: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "semantic_type": self.semantic_type,
            "value_type": self.value_type,
            "required": self.required,
            "required_by": self.required_by,
            "example": self.example,
        }


@dataclass
class ContextSchema:
    """Complete schema for a context file."""
    context_type: str  # e.g., "eda_to_segmentation", "feature_to_training"
    source_crew: str
    target_crews: List[str]
    version: str = "1.0"
    fields: Dict[str, FieldSchema] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_type": self.context_type,
            "source_crew": self.source_crew,
            "target_crews": self.target_crews,
            "version": self.version,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }


# =============================================================================
# CONTEXT BUILDER - For creating self-documenting context
# =============================================================================

class ContextBuilder:
    """
    Builder for creating self-documenting context JSON files.

    The resulting JSON has this structure:
    {
        "_schema": {
            "context_type": "eda_to_segmentation",
            "source_crew": "eda_crew",
            "target_crews": ["segmentation_crew"],
            "version": "1.0",
            "created_at": "2024-01-15T10:30:00",
            "fields": {
                "clustering_recommendations": {
                    "description": "Recommended clustering configuration",
                    "semantic_type": "clustering_config",
                    "value_type": "dict",
                    "required": true,
                    "required_by": ["segmentation_crew"]
                }
            }
        },
        "_semantic_index": {
            "clustering_config": "clustering_recommendations",
            "lag_recommendations": "lag_config"
        },
        "clustering_recommendations": {...actual data...},
        "lag_config": {...actual data...}
    }
    """

    def __init__(
        self,
        context_type: str,
        source_crew: str,
        target_crews: Optional[List[str]] = None,
    ):
        self.context_type = context_type
        self.source_crew = source_crew
        self.target_crews = target_crews or []
        self.fields: Dict[str, FieldSchema] = {}
        self.data: Dict[str, Any] = {}
        self.semantic_index: Dict[str, str] = {}  # semantic_type -> key

    def add_field(
        self,
        key: str,
        value: Any,
        description: str,
        semantic_type: str,
        required: bool = False,
        required_by: Optional[List[str]] = None,
        example: Optional[Any] = None,
    ) -> "ContextBuilder":
        """Add a field with its value and schema documentation."""
        # Determine value type
        if isinstance(value, dict):
            value_type = "dict"
        elif isinstance(value, list):
            value_type = "list"
        elif isinstance(value, bool):
            value_type = "bool"
        elif isinstance(value, int):
            value_type = "int"
        elif isinstance(value, float):
            value_type = "float"
        else:
            value_type = "str"

        # Create field schema
        self.fields[key] = FieldSchema(
            key=key,
            description=description,
            semantic_type=semantic_type,
            value_type=value_type,
            required=required,
            required_by=required_by or [],
            example=example,
        )

        # Store data
        self.data[key] = value

        # Update semantic index
        self.semantic_index[semantic_type] = key

        return self

    def add_metadata(
        self,
        key: str = "metadata",
        **kwargs
    ) -> "ContextBuilder":
        """Add metadata field with common info."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "source_crew": self.source_crew,
            "context_type": self.context_type,
            **kwargs
        }
        return self.add_field(
            key=key,
            value=metadata,
            description="Context metadata including creation time and source",
            semantic_type=SemanticTypes.METADATA,
        )

    def build(self) -> Dict[str, Any]:
        """Build the complete context dictionary with schema."""
        schema = ContextSchema(
            context_type=self.context_type,
            source_crew=self.source_crew,
            target_crews=self.target_crews,
            fields=self.fields,
        )

        result = {
            "_schema": {
                **schema.to_dict(),
                "created_at": datetime.now().isoformat(),
            },
            "_semantic_index": self.semantic_index,
            **self.data
        }

        return result

    def save(self, path: str) -> str:
        """Save context to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        context = self.build()
        with open(path, 'w') as f:
            json.dump(context, f, indent=2, default=str)

        logger.info(f"Saved self-documenting context to {path}")
        return path


# =============================================================================
# CONTEXT READER - For consuming self-documenting context
# =============================================================================

class ContextReader:
    """
    Reader for self-documenting context JSON files.

    Supports multiple ways to access data:
    1. By semantic type (most robust)
    2. By exact key name
    3. By fuzzy key matching
    """

    def __init__(self, path_or_dict: Union[str, Dict[str, Any]]):
        if isinstance(path_or_dict, str):
            with open(path_or_dict, 'r') as f:
                self.data = json.load(f)
            self.path = path_or_dict
        else:
            self.data = path_or_dict
            self.path = None

        self._schema = self.data.get("_schema", {})
        self._semantic_index = self.data.get("_semantic_index", {})

    @property
    def context_type(self) -> str:
        return self._schema.get("context_type", "unknown")

    @property
    def source_crew(self) -> str:
        return self._schema.get("source_crew", "unknown")

    @property
    def has_schema(self) -> bool:
        """Check if this context has embedded schema."""
        return bool(self._schema)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by exact key name."""
        return self.data.get(key, default)

    def get_by_type(self, semantic_type: str, default: Any = None) -> Any:
        """
        Get value by semantic type (most robust method).

        This is the PREFERRED way to access context data as it's
        independent of the exact key naming used by the LLM.
        """
        # First try semantic index
        if semantic_type in self._semantic_index:
            key = self._semantic_index[semantic_type]
            return self.data.get(key, default)

        # Fallback: search field schemas
        fields = self._schema.get("fields", {})
        for key, field_info in fields.items():
            if field_info.get("semantic_type") == semantic_type:
                return self.data.get(key, default)

        # Last resort: fuzzy matching on key names
        return self._fuzzy_match(semantic_type, default)

    def _fuzzy_match(self, semantic_type: str, default: Any = None) -> Any:
        """Fuzzy match semantic type to actual keys."""
        # Convert semantic_type to potential key patterns
        patterns = [
            semantic_type,
            semantic_type.replace("_", ""),
            semantic_type.replace("_recommendations", ""),
            semantic_type.replace("_config", ""),
            semantic_type.replace("_summary", ""),
        ]

        for key in self.data.keys():
            if key.startswith("_"):
                continue
            key_lower = key.lower()
            for pattern in patterns:
                if pattern in key_lower or key_lower in pattern:
                    return self.data[key]

        return default

    def get_description(self, key: str) -> Optional[str]:
        """Get the description for a field."""
        fields = self._schema.get("fields", {})
        if key in fields:
            return fields[key].get("description")
        return None

    def get_all_descriptions(self) -> Dict[str, str]:
        """Get all field descriptions."""
        fields = self._schema.get("fields", {})
        return {k: v.get("description", "") for k, v in fields.items()}

    def list_fields(self) -> List[str]:
        """List all data fields (excluding schema)."""
        return [k for k in self.data.keys() if not k.startswith("_")]

    def list_semantic_types(self) -> List[str]:
        """List all semantic types available."""
        return list(self._semantic_index.keys())

    def validate_required(self, required_types: List[str]) -> List[str]:
        """
        Validate that required semantic types are present.
        Returns list of missing types.
        """
        missing = []
        for sem_type in required_types:
            if self.get_by_type(sem_type) is None:
                missing.append(sem_type)
        return missing

    def to_summary(self) -> str:
        """Generate a human-readable summary of the context."""
        lines = [
            f"Context Type: {self.context_type}",
            f"Source Crew: {self.source_crew}",
            f"Has Schema: {self.has_schema}",
            "",
            "Fields:",
        ]

        for key in self.list_fields():
            desc = self.get_description(key) or "No description"
            value = self.data[key]
            if isinstance(value, dict):
                value_preview = f"dict with {len(value)} keys"
            elif isinstance(value, list):
                value_preview = f"list with {len(value)} items"
            else:
                value_preview = str(value)[:50]

            lines.append(f"  - {key}: {value_preview}")
            lines.append(f"    Description: {desc}")

        return "\n".join(lines)


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_context(
    context_path: str,
    required_semantic_types: Optional[List[str]] = None,
    required_keys: Optional[List[str]] = None,
) -> tuple[bool, List[str]]:
    """
    Validate a context file.

    Parameters
    ----------
    context_path : str
        Path to context JSON file
    required_semantic_types : List[str], optional
        List of semantic types that must be present
    required_keys : List[str], optional
        List of exact keys that must be present (fallback)

    Returns
    -------
    tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    if not os.path.exists(context_path):
        return False, [f"Context file not found: {context_path}"]

    try:
        reader = ContextReader(context_path)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Error reading context: {e}"]

    # Check for empty context
    if not reader.list_fields():
        errors.append("Context file has no data fields")

    # Check required semantic types
    if required_semantic_types:
        missing = reader.validate_required(required_semantic_types)
        if missing:
            errors.append(f"Missing required semantic types: {missing}")

    # Check required keys (fallback for legacy contexts)
    if required_keys:
        for key in required_keys:
            if reader.get(key) is None:
                # Try fuzzy match before reporting error
                found = False
                for actual_key in reader.list_fields():
                    if key.lower() in actual_key.lower() or actual_key.lower() in key.lower():
                        found = True
                        break
                if not found:
                    errors.append(f"Missing required key: {key}")

    return len(errors) == 0, errors


def read_context_robust(
    context_path: str,
    semantic_type: str,
    fallback_keys: Optional[List[str]] = None,
    default: Any = None,
) -> Any:
    """
    Robustly read a value from a context file.

    Tries in order:
    1. Semantic type lookup (if schema exists)
    2. Each fallback key in order
    3. Fuzzy matching on key names
    4. Returns default

    Parameters
    ----------
    context_path : str
        Path to context JSON file
    semantic_type : str
        Semantic type to look for
    fallback_keys : List[str], optional
        List of exact keys to try as fallback
    default : Any
        Default value if not found

    Returns
    -------
    Any
        The found value or default
    """
    if not os.path.exists(context_path):
        logger.warning(f"Context file not found: {context_path}")
        return default

    try:
        reader = ContextReader(context_path)

        # Try semantic type first
        value = reader.get_by_type(semantic_type)
        if value is not None:
            logger.debug(f"Found {semantic_type} via semantic lookup")
            return value

        # Try fallback keys
        if fallback_keys:
            for key in fallback_keys:
                value = reader.get(key)
                if value is not None:
                    logger.debug(f"Found {semantic_type} via fallback key: {key}")
                    return value

        logger.warning(f"Could not find {semantic_type} in {context_path}")
        return default

    except Exception as e:
        logger.error(f"Error reading context {context_path}: {e}")
        return default


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "ContextBuilder",
    "ContextReader",
    "FieldSchema",
    "ContextSchema",
    "SemanticTypes",
    # Functions
    "validate_context",
    "read_context_robust",
]
