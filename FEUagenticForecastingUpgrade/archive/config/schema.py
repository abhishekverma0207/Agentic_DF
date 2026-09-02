# config/schema.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Literal

import yaml
import pydantic as _pydantic
from pydantic import BaseModel, Field, PrivateAttr

# -----------------------------------------------------------------------------
# Pydantic v1 / v2 compatibility shim.
#
# Databricks DBR clusters (especially Python 3.11 runtimes) frequently ship
# with Pydantic v1 system-installed. This codebase is written for v2
# (`field_validator`, `ConfigDict`).  When running under v1 we shim:
#
#   - `field_validator(...)` -> Pydantic v1's `validator(...)` which has an
#     identical decorator signature for field-by-field validators that take
#     `(cls, v)` and return the coerced value.
#   - `ConfigDict(**kw)` -> a plain dict.  In Pydantic v1, model_config is
#     not a recognised class attribute - the inner `class Config:` is - so
#     the dict is silently ignored.  This loses the `extra="forbid"` /
#     `extra="ignore"` enforcement under v1, but the import succeeds and
#     models still validate, which is what matters for unblocking the
#     Databricks job.
#
# A loud warning is emitted ONCE so users know to upgrade.
# -----------------------------------------------------------------------------
_PYDANTIC_V2 = str(getattr(_pydantic, "VERSION", "1")).split(".")[0] == "2"

if _PYDANTIC_V2:
    from pydantic import ConfigDict, field_validator  # type: ignore
else:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Pydantic v1 detected (%s) - schema.py runs in compat mode. "
        "extra='forbid'/'ignore' enforcement is disabled.  For full "
        "validation install pydantic>=2.0 in the cluster.",
        getattr(_pydantic, "VERSION", "?"),
    )
    from pydantic import validator as _v1_validator  # type: ignore

    def field_validator(*fields, **kwargs):  # type: ignore
        """Pydantic v1 shim: drop v2-only kwargs and route to v1 validator."""
        # v2-only kwargs we have to swallow:
        for _v2_only in ("mode", "check_fields"):
            kwargs.pop(_v2_only, None)
        # v1 has `pre`, `always`, `each_item`, `allow_reuse` etc.; map mode
        if kwargs.pop("mode", None) == "before":
            kwargs.setdefault("pre", True)
        kwargs.setdefault("allow_reuse", True)
        return _v1_validator(*fields, **kwargs)

    def ConfigDict(**kwargs):  # type: ignore
        """Pydantic v1 shim: return a plain dict that gets attached to
        `model_config` and silently ignored by v1's validator machinery.

        We DO NOT inject a `class Config` attribute into the surrounding
        class because that would require a metaclass; instead we accept
        that under v1 we lose extra='forbid' enforcement.
        """
        return dict(kwargs)


# =============================================================================
# CONTEXT FILE SCHEMAS - Standardized formats for inter-crew communication
# =============================================================================

class SegmentationRecommendations(BaseModel):
    """Recommendations for segmentation crew."""
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["gmm", "kmeans", "hdbscan", "hierarchical"] = Field(
        default="gmm",
        description="Recommended clustering algorithm"
    )
    n_clusters_range: List[int] = Field(
        default=[3, 4, 5, 6, 7],
        description="Range of cluster counts to try"
    )
    features_to_use: List[str] = Field(
        default_factory=lambda: ["volume_mean", "cv_clean", "zero_fraction_clean", "adi_log", "demand_frequency"],
        description="Features to use for clustering"
    )
    use_hybrid_segmentation: bool = Field(
        default=True,
        description="Whether to combine clustering with business dimensions"
    )
    hybrid_dimensions: List[str] = Field(
        default_factory=lambda: ["volume_tier", "demand_pattern"],
        description="Business dimensions to use in hybrid segmentation"
    )
    intermittency_aware_merge: bool = Field(
        default=True,
        description=(
            "When True, segment merging respects demand pattern families: "
            "regular (smooth/erratic) and sparse (intermittent/lumpy) are never "
            "mixed during merge operations. This prevents lumpy keys from being "
            "pooled with smooth keys in segment-level models."
        ),
    )
    rationale: str = Field(
        default="",
        description="Explanation for the recommendations"
    )


class DataInsights(BaseModel):
    """Key insights about the data characteristics."""
    model_config = ConfigDict(extra="forbid")

    dominant_pattern: Literal["smooth", "erratic", "intermittent", "lumpy", "mixed"] = Field(
        default="mixed",
        description="Most common demand pattern"
    )
    complexity_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Overall forecasting complexity"
    )
    key_challenges: List[str] = Field(
        default_factory=list,
        description="Main challenges identified in the data"
    )


class DemandPatternDistribution(BaseModel):
    """Distribution of demand patterns (Syntetos-Boylan)."""
    model_config = ConfigDict(extra="forbid")

    smooth: float = Field(default=0.0, ge=0.0, le=1.0)
    erratic: float = Field(default=0.0, ge=0.0, le=1.0)
    intermittent: float = Field(default=0.0, ge=0.0, le=1.0)
    lumpy: float = Field(default=0.0, ge=0.0, le=1.0)


class DataCharacteristics(BaseModel):
    """Summary statistics about the data."""
    model_config = ConfigDict(extra="forbid")

    avg_cv: Optional[float] = Field(default=None, description="Average coefficient of variation")
    avg_zero_fraction: Optional[float] = Field(default=None, description="Average zero fraction")
    avg_adi: Optional[float] = Field(default=None, description="Average demand interval")
    has_seasonality: bool = Field(default=False, description="Whether seasonality was detected")
    seasonal_period: int = Field(default=52, description="Detected seasonal period")
    discrete_demand_pct: float = Field(default=0.0, description="Percentage of keys with discrete (low-cardinality) demand")


class EDAToSegmentationContext(BaseModel):
    """
    STANDARDIZED context file from EDA to Segmentation crew.

    This schema ensures consistent format that Segmentation crew can reliably parse.
    """
    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="EDA", description="Source of this context (EDA/EDA Analyst)")
    n_series: int = Field(description="Total number of time series")
    demand_pattern_distribution: DemandPatternDistribution = Field(
        default_factory=DemandPatternDistribution,
        description="Distribution of Syntetos-Boylan patterns"
    )
    recommendations: SegmentationRecommendations = Field(
        default_factory=SegmentationRecommendations,
        description="Segmentation recommendations"
    )
    data_characteristics: DataCharacteristics = Field(
        default_factory=DataCharacteristics,
        description="Summary data characteristics"
    )
    data_insights: Optional[DataInsights] = Field(
        default=None,
        description="Optional insights (if LLM analyst was used)"
    )


class LagFeatureRecommendations(BaseModel):
    """Lag feature recommendations."""
    model_config = ConfigDict(extra="forbid")

    recommended_lags: List[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 13, 26, 52],
        description="Recommended lag periods"
    )
    seasonal_lags: List[int] = Field(
        default_factory=list,
        description="Seasonal lag periods"
    )
    rationale: str = Field(default="", description="Explanation for lag choices")


class RollingFeatureRecommendations(BaseModel):
    """Rolling window feature recommendations."""
    model_config = ConfigDict(extra="forbid")

    windows: List[int] = Field(
        default_factory=lambda: [4, 8, 13],
        description="Rolling window sizes"
    )
    aggregations: List[str] = Field(
        default_factory=lambda: ["mean", "std", "min", "max"],
        description="Aggregation functions to apply"
    )
    rationale: str = Field(default="", description="Explanation for window choices")


class IntermittencyHandling(BaseModel):
    """Recommendations for handling intermittent demand."""
    model_config = ConfigDict(extra="forbid")

    high_intermittency_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of series with high intermittency"
    )
    use_demand_probability_features: bool = Field(
        default=False,
        description="Whether to create demand probability features"
    )
    use_zero_inflated_features: bool = Field(
        default=False,
        description="Whether to create zero-inflated features"
    )
    rationale: str = Field(default="", description="Explanation")


class ExternalFeatureRecommendations(BaseModel):
    """External feature recommendations."""
    model_config = ConfigDict(extra="forbid")

    use_price_features: bool = Field(default=False, description="Use price-related features")
    use_promo_features: bool = Field(default=False, description="Use promotion features")
    use_weather_features: bool = Field(default=False, description="Use weather features")
    recommended_lags: List[int] = Field(
        default_factory=lambda: [0, 1, 2],
        description="Lags for external features"
    )


class EDAToFeatureContext(BaseModel):
    """
    STANDARDIZED context file from EDA to Feature Engineering crew.

    This schema ensures consistent format that Feature Engineering crew can reliably parse.
    """
    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="EDA", description="Source of this context")
    lag_features: LagFeatureRecommendations = Field(
        default_factory=LagFeatureRecommendations,
        description="Lag feature recommendations"
    )
    rolling_features: RollingFeatureRecommendations = Field(
        default_factory=RollingFeatureRecommendations,
        description="Rolling feature recommendations"
    )
    intermittency_handling: IntermittencyHandling = Field(
        default_factory=IntermittencyHandling,
        description="Intermittency handling recommendations"
    )
    external_features: ExternalFeatureRecommendations = Field(
        default_factory=ExternalFeatureRecommendations,
        description="External feature recommendations"
    )
    analysis_summary: str = Field(
        default="",
        description="Optional summary from LLM analyst"
    )


class ModelRecommendations(BaseModel):
    """Model family recommendations."""
    model_config = ConfigDict(extra="forbid")

    primary_models: List[str] = Field(
        default_factory=lambda: ["lightgbm", "xgboost"],
        description="Primary recommended models (top 3-5)"
    )
    intermittency_specialists: List[str] = Field(
        default_factory=list,
        description="Models for intermittent demand (croston, sba, tsb)"
    )
    discrete_specialists: List[str] = Field(
        default_factory=list,
        description="Models for discrete/low-cardinality demand (ordinal_regression, discrete_classifier, hybrid_discrete)"
    )
    seasonal_models: List[str] = Field(
        default_factory=list,
        description="Models for seasonal data (sarima, prophet, tbats)"
    )
    all_recommended: List[str] = Field(
        default_factory=list,
        description="All recommended models"
    )
    rationale: str = Field(default="", description="Explanation for model choices")


class TrainingStrategy(BaseModel):
    """Training strategy recommendations."""
    model_config = ConfigDict(extra="forbid")

    use_segment_specific_models: bool = Field(
        default=True,
        description="Whether to train different models per segment"
    )
    use_ensemble: bool = Field(
        default=True,
        description="Whether to use ensemble methods"
    )
    cv_folds: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )
    rationale: str = Field(default="", description="Explanation")


class TrainingDataCharacteristics(BaseModel):
    """Data characteristics relevant for training."""
    model_config = ConfigDict(extra="forbid")

    n_series: int = Field(description="Total number of series")
    high_intermittency: bool = Field(default=False, description="High intermittency flag")
    has_seasonality: bool = Field(default=False, description="Seasonality detected")
    avg_cv: Optional[float] = Field(default=None, description="Average CV")
    has_discrete_demand: bool = Field(default=False, description="Some keys have discrete (low-cardinality) demand")
    discrete_pct: float = Field(default=0.0, description="Percentage of keys with discrete demand")
    avg_n_unique: Optional[float] = Field(default=None, description="Average unique target values per key")


class EDAToTrainingContext(BaseModel):
    """
    STANDARDIZED context file from EDA to Training crew.

    This schema ensures consistent format that Training crew can reliably parse.
    """
    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="EDA", description="Source of this context")
    model_recommendations: ModelRecommendations = Field(
        default_factory=ModelRecommendations,
        description="Model recommendations"
    )
    training_strategy: TrainingStrategy = Field(
        default_factory=TrainingStrategy,
        description="Training strategy recommendations"
    )
    data_characteristics: TrainingDataCharacteristics = Field(
        default_factory=lambda: TrainingDataCharacteristics(n_series=0),
        description="Data characteristics"
    )
    hyperparameter_hints: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional hyperparameter hints by model"
    )
    analysis_summary: str = Field(
        default="",
        description="Optional summary from LLM analyst"
    )


# =============================================================================
# ORIGINAL SCHEMA CLASSES
# =============================================================================

class FeatureGroup(BaseModel):
    """Domain-specific feature buckets to keep config organized."""
    model_config = ConfigDict(extra="ignore")

    numeric: List[str] = Field(default_factory=list)
    categorical: List[str] = Field(default_factory=list)


class CrewTimeouts(BaseModel):
    """Per-crew soft timeouts in seconds (used only as guidance in prompts/logging)."""
    model_config = ConfigDict(extra="ignore")

    eda: int = 600
    segmentation: int = 600
    feature: int = 900
    training: int = 1800
    diagnostics: int = 900


class ModelLevelAllocationConfig(BaseModel):
    """
    Configuration for the state-of-the-art model level allocation algorithm.

    This controls the sophisticated multi-factor algorithm that determines
    whether each key should get an individual model or be pooled with segment.

    Factors Considered:
    - Data sufficiency (history length, non-zero observations)
    - Volume importance (business impact)
    - Forecastability (learnable patterns)
    - Uniqueness (how different from segment peers)
    - Predictability (signal quality)
    - Stability vs complexity balance
    - Autocorrelation patterns
    """
    model_config = ConfigDict(extra="ignore")

    min_individual_score: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum composite score (0-1) for a key to get an individual model. "
            "Higher = more selective (fewer individual models). "
            "Score combines data sufficiency, volume, forecastability, uniqueness, etc."
        ),
    )

    max_individual_pct: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum percentage of keys that can get individual models. "
            "Prevents over-allocation which would defeat the benefits of pooling. "
            "Default 0.30 = max 30% of keys get individual models. "
            "Raised from the historical 0.15 because the per-key XGBoost "
            "benchmark (which gives 100% of keys their own model) was "
            "beating the pooled pipeline on WAPE — widening the individual "
            "tier closes part of that specialization gap without losing "
            "pooling benefits for the long tail of low-volume SKUs."
        ),
    )

    min_segment_size: int = Field(
        default=10,
        ge=2,
        description=(
            "Minimum keys per segment to ensure viable segment models. "
            "If allocating individual models would make a segment too small, "
            "some keys will be pushed back to segment pooling."
        ),
    )

    volume_override_quantile: float = Field(
        default=0.93,
        ge=0.5,
        le=1.0,
        description=(
            "Keys above this volume percentile ALWAYS get individual models, "
            "PROVIDED they also pass the data-sufficiency gate (min non-zero obs) "
            "AND stay under max_zero_fraction_for_individual. "
            "Default 0.93 = top 7% by volume are candidates for dedicated models. "
            "Previously 0.995 (top 0.5%) which was too strict on large catalogues "
            "— many high-volume but moderately-sparse SKUs were forced into segment "
            "pooling where segment-average patterns washed out their idiosyncratic "
            "signal. Tighter gates below prevent overfitting on sparse keys."
        ),
    )

    forecastability_override: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description=(
            "Keys with forecastability_score above this ALWAYS get individual models. "
            "Highly forecastable keys have strong patterns that benefit from dedicated models."
        ),
    )

    min_nonzero_obs_for_individual: int = Field(
        default=52,
        ge=0,
        description=(
            "Minimum non-zero observations a key must have to receive an individual model. "
            "Combined with the existing _data_sufficient gate, this prevents sparse keys "
            "from winning the volume/score override despite having too little signal to fit. "
            "Default 52 (one year of weekly non-zero data). Set to 0 to disable this gate."
        ),
    )

    max_zero_fraction_for_individual: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum zero_fraction a key can have and still qualify for an individual model. "
            "Even if volume is high, a key with >70%% zero weeks is typically lumpy/intermittent "
            "and benefits more from pooled estimation of the occurrence probability. "
            "Default 0.70. Set to 1.0 to disable this gate. "
            "NOTE: keys above top_volume_bypass_quantile bypass this gate entirely."
        ),
    )

    top_volume_bypass_quantile: Optional[float] = Field(
        default=0.80,
        ge=0.5,
        le=1.0,
        description=(
            "Keys whose volume is at or above this quantile BYPASS the "
            "zero-fraction and min-nonzero-obs gates entirely and go "
            "straight to individual modelling (they still need to meet "
            "the basic min-obs threshold so there's something to train on). "
            "Default 0.80 = top 20%% by volume are guaranteed individual "
            "models regardless of lumpiness. On UK COOKING_AID this "
            "promotes the high-business-impact keys that were previously "
            "stuck in pooled training because of their 80%%+ zero "
            "fractions, bringing the individual-model share from ~2%% to "
            "~20%% — meaningfully closer to the per-key XGBoost benchmark "
            "which individualises 100%% of keys. Set to None to disable."
        ),
    )


class DesignConfig(BaseModel):
    """
    Design-time controls for the agentic multi-crew workflow.

    Model Family Categories:
    ========================

    1. TREE-BASED / GRADIENT BOOSTING (Universal - works for all patterns):
       - lightgbm: Fast gradient boosting, handles sparse data well
       - xgboost: Robust gradient boosting, good for tabular features
       - catboost: Handles categoricals natively, robust to overfitting
       - random_forest: Ensemble method, less prone to overfitting

    2. CLASSICAL STATISTICAL (Best for smooth/regular demand):
       - arima: Autoregressive integrated moving average
       - sarima: Seasonal ARIMA for seasonal patterns
       - ets: Exponential smoothing (Error-Trend-Seasonality)
       - theta: Theta method (M3 competition winner)
       - tbats: Trigonometric seasonality, Box-Cox, ARMA errors

    3. BAYESIAN / PROBABILISTIC:
       - bsts: Bayesian Structural Time Series
       - prophet: Facebook's decomposable additive model

    4. INTERMITTENT DEMAND SPECIALISTS (For sparse/lumpy demand):
       - croston: Classic intermittent demand method
       - sba: Syntetos-Boylan Approximation (bias-corrected Croston)
       - tsb: Teunter-Syntetos-Babai (handles demand obsolescence)
       - imapa: Intermittent Multiple Aggregation Prediction Algorithm

    5. COMPOUND / HURDLE MODELS (Two-stage: occurrence + severity):
       - zero_inflated: Separate zero probability model + demand size model
       - hurdle_model: P(demand occurs) × E[demand | demand > 0]
       - tweedie: Single model for zero-inflated continuous (Tweedie distribution)

    6. DEEP LEARNING (Requires enable_deep_models=True):
       - tft: Temporal Fusion Transformer (interpretable attention-based)
       - lstm: Long Short-Term Memory networks
       - nbeats: Neural Basis Expansion Analysis
       - deepar: Amazon's probabilistic autoregressive RNN
       - wavenet: Dilated causal convolutions

    7. ENSEMBLE / META METHODS:
       - weighted_ensemble: Weighted combination of multiple models
       - stacking: Meta-learner trained on base model predictions

    Pattern-to-Model Recommendations:
    =================================
    - SMOOTH demand (low CV, low zero%): sarima, ets, prophet, lightgbm, xgboost
    - ERRATIC demand (high CV, low zero%): xgboost, lightgbm, theta, bsts
    - INTERMITTENT demand (low CV, high zero%): croston, sba, tsb, zero_inflated
    - LUMPY demand (high CV, high zero%): hurdle_model, zero_inflated, tsb, imapa
    """
    model_config = ConfigDict(extra="ignore")

    # -------------------------------------------------------------------------
    # Feature Availability Detection (Phase 1 Enhancement)
    # -------------------------------------------------------------------------
    enable_feature_availability_detection: bool = Field(
        default=True,
        description=(
            "When True, runs the Feature Availability Detection pipeline before "
            "Feature Engineering. This automatically: (1) detects the history cutoff "
            "from data, (2) classifies features as known_in_future / history_only / "
            "partially_known, (3) generates frozen embeddings for history-only features. "
            "This ensures no future-unavailable features leak into model training."
        ),
    )

    auto_detect_features: bool = Field(
        default=True,
        description=(
            "When True, automatically discover feature columns from the source data "
            "instead of requiring them in config. Every column that is not a key, "
            "timestamp, or target column is treated as a candidate feature. "
            "Feature Availability Detection then classifies each into known_in_future, "
            "history_only, or excluded. This allows minimal config with only: "
            "input_data_path, prediction_key_cols, timestamp_col, target_col, "
            "train/val/test dates, and forecast_horizon."
        ),
    )

    feature_availability_future_known_threshold: float = Field(
        default=0.50,
        ge=0.1,
        le=1.0,
        description=(
            "Minimum fill rate (non-null, non-zero) in future periods for a feature "
            "to be classified as 'known_in_future'. 0.50 = at least 50% of future rows "
            "must have meaningful values."
        ),
    )

    feature_availability_partial_threshold: float = Field(
        default=0.10,
        ge=0.01,
        le=0.5,
        description=(
            "Minimum fill rate in future periods for 'partially_known' classification. "
            "Features between this and future_known_threshold are partially available."
        ),
    )

    feature_availability_history_min_useful: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description=(
            "Minimum useful rate in history for a feature to be considered at all. "
            "Features below this threshold are excluded entirely."
        ),
    )

    max_model_candidates_per_group: int = 2
    max_hparam_search_depth: int = 10
    max_design_iterations: int = 2
    crew_timeouts: CrewTimeouts = Field(default_factory=CrewTimeouts)

    # Model Level Allocation (Individual vs Segment)
    model_level_allocation: ModelLevelAllocationConfig = Field(
        default_factory=ModelLevelAllocationConfig,
        description=(
            "Configuration for the sophisticated multi-factor model level allocation algorithm. "
            "Controls how keys are assigned to individual models vs segment-pooled models."
        ),
    )

    model_families: List[str] = Field(
        default_factory=lambda: [
            # Tree-based (universal)
            "lightgbm", "xgboost", "catboost", "random_forest",
            # Classical statistical
            "arima", "sarima", "ets", "theta", "tbats",
            # Bayesian/Probabilistic
            "bsts", "prophet",
            # Intermittent demand specialists
            "croston", "sba", "tsb", "imapa",
            # Compound/Hurdle models
            "zero_inflated", "hurdle_model", "tweedie",
            # Discrete demand specialists (for low-cardinality targets)
            "ordinal_regression", "discrete_classifier", "hybrid_discrete",
            # Deep learning (controlled by enable_deep_models)
            "tft", "lstm", "nbeats", "deepar", "wavenet",
            # Ensemble methods
            "weighted_ensemble", "stacking",
            # Multi-horizon direct forecasting (controlled by enable_multi_horizon)
            "multi_horizon_lightgbm", "multi_horizon_xgboost", "multi_horizon_ensemble",
        ],
        description="Available model families. Training crew selects appropriate models per segment pattern."
    )
    enable_deep_models: bool = True

    # -------------------------------------------------------------------------
    # Multi-Horizon Direct Forecasting
    # -------------------------------------------------------------------------
    enable_multi_horizon: bool = Field(
        default=True,
        description=(
            "When True, multi-horizon direct forecasting models (multi_horizon_lightgbm, "
            "multi_horizon_xgboost, multi_horizon_ensemble) are automatically included as "
            "candidates for smooth/erratic demand patterns when forecast_horizon >= 3. "
            "These models directly optimize for each horizon step, avoiding error compounding "
            "from recursive one-step-ahead forecasting."
        ),
    )

    # -------------------------------------------------------------------------
    # Reviewer agent toggle (reduces token usage when disabled)
    # -------------------------------------------------------------------------
    enable_reviewer: bool = Field(
        default=False,
        description=(
            "When False (default), skip the reviewer/validation agent in each crew. "
            "This reduces token usage and speeds up execution. "
            "Set to True for production runs where review/validation is critical."
        ),
    )

    # -------------------------------------------------------------------------
    # Insights report toggle (reduces token usage and prevents LLM file corruption)
    # -------------------------------------------------------------------------
    enable_insights_reports: bool = Field(
        default=False,
        description=(
            "When False (default), skip LLM-based insights/documentation agents in each crew. "
            "These agents generate markdown reports (EDA_INSIGHTS_REPORT.md, "
            "SEGMENTATION_INSIGHTS_GUIDE.md, etc.) but use CodeExecutionTool which "
            "risks overwriting core pipeline output files. "
            "Set to True only when you need detailed LLM-generated analysis reports."
        ),
    )

    # -------------------------------------------------------------------------
    # Diagnostic Metric Levels Configuration
    # -------------------------------------------------------------------------
    # These control how forecast accuracy/WAPE are calculated and reported.
    #
    # abs_error_calc_levels: Defines the granularity at which absolute errors
    # are computed FIRST. Each level is a list of column names that form a group.
    # Default is [["key", "time_period"]] which computes error per key per period.
    # Additional levels aggregate actuals and predictions first, then compute error.
    #
    # diagnostic_metric_levels: Defines how to GROUP/SUMMARIZE metrics for reporting.
    # Each level is a list of column names to group by when aggregating WAPE/Accuracy.
    # Default is [["segment_id"]] which reports metrics per segment.
    #
    # Example:
    #   abs_error_calc_levels: [["key", "time_period"], ["subgroup", "time_period"]]
    #   diagnostic_metric_levels: [["segment_id"], ["category", "brand"], ["category", "sub_category"]]
    #
    # This produces:
    #   - Error at key×time_period → WAPE by segment, by category×brand, by category×sub_category
    #   - Error at subgroup×time_period → WAPE by segment, by category×brand, by category×sub_category
    # -------------------------------------------------------------------------
    abs_error_calc_levels: List[List[str]] = Field(
        default_factory=lambda: [],
        description=(
            "Additional levels for absolute error calculation beyond the default key×time_period. "
            "Each entry is a list of column names defining the aggregation level. "
            "Example: [['subgroup', 'time_period'], ['store_group', 'subgroup', 'time_period']]"
        ),
    )

    diagnostic_metric_levels: List[List[str]] = Field(
        default_factory=lambda: [],
        description=(
            "Additional levels for WAPE/Accuracy metric aggregation beyond the default model grouping column. "
            "The default metric level is DISCOVERED from per_key_with_segments.csv (e.g., 'model_group', 'segment_id'). "
            "Each entry is a list of column names to group by for reporting. "
            "Example: [['category', 'brand'], ['category', 'sub_category'], ['category', 'sub_category', 'packsize']]"
        ),
    )

    # -------------------------------------------------------------------------
    # State-of-the-Art Optimization Settings
    # -------------------------------------------------------------------------
    # These settings enable advanced forecast optimization techniques from
    # utils/forecast_optimization.py and utils/model_selection_intelligence.py
    # -------------------------------------------------------------------------

    enable_bias_correction: bool = Field(
        default=True,
        description=(
            "Enable automatic detection and correction of systematic forecast bias. "
            "Detects both additive and multiplicative bias patterns and applies "
            "appropriate corrections to improve forecast accuracy."
        ),
    )

    enable_ensemble_optimization: bool = Field(
        default=True,
        description=(
            "Enable optimal ensemble weight finding using cross-validated optimization. "
            "Automatically finds the best combination of model predictions using "
            "constrained optimization (weights sum to 1, non-negative)."
        ),
    )

    enable_forecast_calibration: bool = Field(
        default=True,
        description=(
            "Enable forecast calibration using isotonic regression or Platt scaling. "
            "Ensures probabilistic forecasts are well-calibrated (predicted probability "
            "matches actual frequency)."
        ),
    )

    prediction_intervals: bool = Field(
        default=True,
        description=(
            "Generate prediction intervals using conformal prediction. "
            "Provides distribution-free, non-parametric uncertainty estimates "
            "with guaranteed coverage (unlike Gaussian assumptions)."
        ),
    )

    prediction_interval_confidence: float = Field(
        default=0.95,
        description=(
            "Confidence level for prediction intervals (0.8 to 0.99). "
            "0.95 means 95% of actual values should fall within the interval."
        ),
    )

    meta_learning_enabled: bool = Field(
        default=True,
        description=(
            "Enable meta-learning for model selection. Uses time series characteristics "
            "(ADI, CV, trend, seasonality) to predict which model family will perform "
            "best without exhaustive search. Speeds up model selection significantly."
        ),
    )

    early_stopping_rounds: int = Field(
        default=50,
        description=(
            "Early stopping rounds for model selection tournament. "
            "Stop evaluating a model if it hasn't improved in this many rounds. "
            "Reduces computation for clearly underperforming models."
        ),
    )

    max_ensemble_models: int = Field(
        default=5,
        description=(
            "Maximum number of models to include in ensemble. "
            "More models can improve accuracy but increase complexity and runtime."
        ),
    )

    # -------------------------------------------------------------------------
    # Validation-Learned Bias Calibration Settings
    # -------------------------------------------------------------------------
    # These settings control the bias calibration system that learns correction
    # factors from validation data and applies them during inference.
    # This addresses systematic over-forecasting, especially for zero-actual cases.
    # -------------------------------------------------------------------------

    apply_bias_calibration: bool = Field(
        default=False,
        description=(
            "Enable validation-learned bias calibration. When enabled, the system "
            "learns calibration factors per (model_level, zero_fraction_bucket) from "
            "validation period recursive forecasts, then applies them during "
            "production inference to correct systematic bias."
        ),
    )

    bias_calibration_buckets: int = Field(
        default=5,
        ge=2,
        le=10,
        description=(
            "Number of zero_fraction percentile buckets for bias calibration. "
            "Keys are grouped into buckets based on their zero_fraction percentile. "
            "5 = quintiles (20% each), 4 = quartiles, 10 = deciles. "
            "More buckets = finer calibration but fewer samples per bucket."
        ),
    )

    bias_calibration_factor_min: float = Field(
        default=0.2,
        ge=0.01,
        le=1.0,
        description=(
            "Minimum allowed calibration factor. This prevents over-correction "
            "that could reduce predictions by too much. Default 0.2 means predictions "
            "won't be reduced by more than 80%. Range: 0.01 to 1.0."
        ),
    )

    bias_calibration_factor_max: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description=(
            "Maximum allowed calibration factor. This prevents blow-up that could "
            "increase predictions too much. Default 2.0 means predictions won't be "
            "increased by more than 2x. Range: 1.0 to 10.0."
        ),
    )

    # -------------------------------------------------------------------------
    # Inference Feature Regeneration Settings
    # -------------------------------------------------------------------------
    # Controls whether inference pipeline regenerates features from scratch
    # using the latest data, or uses pre-computed feature files from training.
    # -------------------------------------------------------------------------

    regenerate_features_on_inference: bool = Field(
        default=True,
        description=(
            "When True, the inference pipeline will regenerate features using "
            "the latest source data (up to val_end) before retraining models. "
            "This ensures lag features, rolling averages, and derived features "
            "reflect the most recent data. When False, uses pre-computed feature "
            "files from the training pipeline (faster but potentially stale)."
        ),
    )

    regenerate_features_on_backtest: bool = Field(
        default=True,
        description=(
            "When True, the backtesting pipeline will regenerate features at each "
            "origin point using only data available up to that origin's cutoff. "
            "This prevents data leakage from future lag/rolling features and gives "
            "realistic backtest accuracy metrics. When False, uses pre-computed "
            "features (faster but may have optimistic/leaked results). "
            "WARNING: Setting to True significantly increases backtest runtime."
        ),
    )

    max_backtest_origins: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cap the number of rolling-origin backtest iterations. "
            "When set (e.g. 8), the pipeline evaluates only the first N origins "
            "starting from (val_end + 1) and stops — each origin still covers "
            "the full forecast_horizon so you keep multi-step accuracy metrics. "
            "Useful for faster iteration during development (UK COOKING_AID: "
            "18 origins × ~14 min = ~4.3 h; capping at 8 brings this to ~1.9 h). "
            "When None (default), generates one origin per period in "
            "[test_start, test_end], which is the original behaviour."
        ),
    )

    # -------------------------------------------------------------------------
    # Top-Down Category Reconciliation Settings
    # -------------------------------------------------------------------------
    # Trains a Prophet model on the aggregate (category-level) demand and
    # uses its period-wise forecasts to proportionally scale key-level
    # forecasts. This corrects systematic over/under-forecasting because
    # aggregate models are inherently more stable (law of large numbers)
    # and capture trend/seasonality more reliably.
    # -------------------------------------------------------------------------

    enable_top_down_reconciliation: bool = Field(
        default=True,
        description=(
            "Enable top-down hierarchical reconciliation of forecasts using "
            "a Prophet model on category-level aggregate demand. Ensures "
            "total forecast is anchored to a robust aggregate while "
            "preserving each key's relative share."
        ),
    )

    reconciliation_ratio_min: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description=(
            "Minimum allowed reconciliation ratio. 0.5 means forecasts "
            "won't be reduced by more than 50%. Prevents extreme downward "
            "corrections even if the category model predicts much less."
        ),
    )

    reconciliation_ratio_max: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description=(
            "Maximum allowed reconciliation ratio. 2.0 means forecasts "
            "won't be increased by more than 2x. Prevents extreme upward "
            "corrections."
        ),
    )

    reconciliation_trust_band: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description=(
            "If the reconciliation ratio is within 1.0 +/- this value, "
            "skip adjustment. 0.1 means if bottom-up total is within 10%% "
            "of category forecast, no scaling is applied."
        ),
    )

    reconciliation_changepoint_prior: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description=(
            "Prophet changepoint_prior_scale for the category model. "
            "Higher = more flexible trend (captures recent changes). "
            "Lower = smoother trend (more stable). Default 0.05."
        ),
    )

    reconciliation_yoy_max_deviation: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum allowed YoY deviation after reconciliation. "
            "After reconciliation, compares each period's forecast total "
            "to the same period last year's actual total. If the YoY change "
            "exceeds this threshold, the forecast is dampened back. "
            "0.20 = max +/-20% swing vs prior year."
        ),
    )

    forward_forecast_exclude_col: str = Field(
        default="",
        description=(
            "Optional column name in the source data containing a binary "
            "flag (1 = exclude, 0 = keep). When set, keys with this flag "
            "equal to 1 have their predictions zeroed out BEFORE category "
            "reconciliation runs. This ensures excluded keys do not inflate "
            "the bottom-up total used in reconciliation. If this column is "
            "configured but missing from the source data, the pipeline will "
            "fail immediately with a clear error."
        ),
    )

    # -------------------------------------------------------------------------
    # Direct Multi-Horizon Forecasting (upgrade over recursive inference)
    # -------------------------------------------------------------------------
    # When enabled, PHASE 6 of inference (and each origin of backtest) trains
    # ONE LightGBM per forecast horizon against the true y[t+h] and predicts
    # each lag directly - no recursive feedback of own predictions into lag
    # features.  Eliminates recursive error accumulation which is the dominant
    # failure mode on intermittent SKUs where the recursive pipeline validates
    # at ~10% WMAPE but tests at ~55%.
    # -------------------------------------------------------------------------

    use_direct_multi_horizon: bool = Field(
        default=False,
        description=(
            "Enable direct multi-horizon inference. Trains one model per "
            "horizon against the true y[t+h] target and predicts each lag "
            "directly (no recursive feedback). Recommended for intermittent "
            "demand where recursive predictions accumulate bias. When False, "
            "the existing recursive forecasting path is used."
        ),
    )

    dmh_objective: str = Field(
        default="auto",
        description=(
            "How the direct-horizon models are trained and combined. Options:\n"
            "  - 'auto' (default, CATEGORY-AGNOSTIC): trains a small panel of "
            "candidate heads (regression + quantile@0.70 + quantile@0.80) per "
            "horizon and LEARNS ensemble weights from validation OOF by "
            "minimising WMAPE + 0.3·|bias|, then shrinks 50% toward uniform. "
            "This lets the model pick the right bias-correction recipe for "
            "each dataset without hand-tuning per category.\n"
            "  - 'ensemble:regression=0.5,quantile@0.70=0.5' : explicit ensemble "
            "spec if you want deterministic weights.\n"
            "  - 'regression' / 'tweedie' / 'quantile' : single-objective."
        ),
    )

    dmh_tweedie_variance_power: float = Field(
        default=1.3, ge=1.0, le=2.0,
        description="Only used when dmh_objective='tweedie'.",
    )

    dmh_num_leaves: int = Field(default=63, ge=8, le=511)
    dmh_learning_rate: float = Field(default=0.05, gt=0, le=1.0)
    dmh_min_data_in_leaf: int = Field(
        default=200, ge=10,
        description="Stronger regularisation for intermittent SKUs. 200 works well on Unilever weekly data.",
    )
    dmh_lambda_l2: float = Field(default=1.0, ge=0)
    dmh_num_boost_round: int = Field(default=3000, ge=100)
    dmh_early_stopping_rounds: int = Field(default=150, ge=10)

    dmh_recency_halflife_weeks: float = Field(
        default=26.0, ge=0.0,
        description=(
            "Exponential weight decay on training rows where weight = "
            "0.5 ** ((max_week - row_week) / halflife). 26 weeks emphasises "
            "the last 6 months. Set to 0 to disable."
        ),
    )

    dmh_sply_blend_alpha: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Weight of model prediction in the final forecast; (1-alpha) goes to lag-52 SPLY reference. 1.0 disables blend.",
    )

    dmh_horizons: list = Field(
        default_factory=list,
        description="Explicit horizons. Empty -> [1..forecast_horizon].",
    )

    dmh_n_seeds: int = Field(
        default=1, ge=1, le=10,
        description="Multi-seed bagging. Training ~N× cost, ~0.5pp WMAPE gain typical.",
    )

    dmh_top_k_features: int = Field(
        default=0, ge=0,
        description="Feature-importance filter. 50 is a good default on 500-column panels.",
    )

    dmh_smooth_window: int = Field(
        default=0, ge=0, le=5,
        description="Horizon-smoothing half-width. 1 averages {h-1,h,h+1} predictions.",
    )

    dmh_horizon_workers: int = Field(
        default=0, ge=0, le=64,
        description=(
            "Driver-side parallelism for the per-horizon DMH training loop. "
            "DMH historically trained all H horizons sequentially on the "
            "driver while the Spark workers idled; on a 13-horizon, 3-seed, "
            "3-head 'auto' config that's 117 serial LightGBM fits. Each "
            "horizon is mathematically independent (no cross-horizon state), "
            "so running them concurrently via ThreadPoolExecutor produces "
            "bit-identical numerical outputs but cuts wall time roughly "
            "linearly in `horizon_workers` (bounded by driver cores). "
            "0 = auto (half the driver cores, capped at the horizon count). "
            "Per-fit LightGBM `num_threads` is automatically reduced so "
            "horizon_workers * num_threads ≈ driver cores (avoids OMP "
            "oversubscription)."
        ),
    )

    dmh_enable_trajectory_features: bool = Field(
        default=True,
        description="Per-key trajectory features (slope/volatility/acceleration/recent-to-long ratio).",
    )

    dmh_enable_apg_features: bool = Field(
        default=True,
        description="APG-level weekly sum (shifted 1 week) cross-sectional signal.",
    )

    dmh_enable_forward_features: bool = Field(
        default=True,
        description="Target-week calendar + promo + weather features (known-ahead signals).",
    )

    dmh_apg_col: str = Field(
        default="",
        description=(
            "Column name used by the APG-level weekly-sum augmenter. "
            "Empty means auto-derive: first entry of `imputation_level` if "
            "present, else fall back to 'APG_code'. Set explicitly to override "
            "for markets using non-standard APG labels (e.g. TH uses "
            "'APGDescription', DE uses 'APG_code', etc.)."
        ),
    )

    # -------------------------------------------------------------------------
    # DMH backtest-specific overrides (big runtime wins).
    #
    # At every backtest origin the DMH path would otherwise repeat the full
    # forward-inference training - H horizons × n_seeds × n_ensemble_heads
    # fits, plus a feature-importance probe per horizon, plus three
    # augmenter passes.  For a 13-horizon/8-origin/3-seed/3-head
    # configuration that's 936 LightGBM fits.
    #
    # These knobs override the same-named dmh_* fields ONLY while running
    # inside the backtest loop (backtest_origin is set).  Setting them
    # tighter than production lets you evaluate the metric you actually
    # care about without paying the full training cost.
    #
    # Typical lag-4 backtest recipe (≈13× faster):
    #   dmh_backtest_horizons: [4]          # only train h=4 per origin
    #   dmh_backtest_n_seeds: 1             # skip seed bagging
    #   dmh_backtest_top_k_features: 0      # skip importance probe
    #   dmh_backtest_objective: "regression"  # skip ensemble auto-tune
    # -------------------------------------------------------------------------

    dmh_backtest_horizons: list = Field(
        default_factory=list,
        description=(
            "Horizons to train at each backtest origin.  Empty -> inherit from "
            "`dmh_horizons` (falls back to [1..forecast_horizon]).  "
            "For a lag-N evaluation, set to [N] so each origin only trains the "
            "head whose forecasts are actually measured, saving H-1× compute."
        ),
    )

    dmh_backtest_n_seeds: int = Field(
        default=0, ge=0, le=10,
        description=(
            "Multi-seed bagging at backtest time.  0 -> inherit from "
            "`dmh_n_seeds`.  Set to 1 to drop bagging during backtest (one "
            "model per horizon) and keep bagging only for production forecasts."
        ),
    )

    dmh_backtest_top_k_features: int = Field(
        default=-1, ge=-1,
        description=(
            "Feature-importance probe top-K at backtest time.  -1 -> inherit "
            "from `dmh_top_k_features`.  Set to 0 to skip the probe entirely "
            "at backtest (saves ~30-60s per horizon per origin)."
        ),
    )

    dmh_backtest_objective: str = Field(
        default="",
        description=(
            "LightGBM objective override at backtest.  Empty -> inherit from "
            "`dmh_objective`.  Set to 'regression' to skip the 3-head ensemble "
            "auto-tune during backtest and cut training by ~3×, while keeping "
            "'auto' for the production forward forecast."
        ),
    )

    dmh_backtest_enable_augmenters: bool = Field(
        default=True,
        description=(
            "Whether to run the feature augmenters (trajectory, APG sum, "
            "forward target-week features) during backtest.  Disable to save "
            "~30-60s per origin at the cost of losing the augmenter signal."
        ),
    )

    dmh_cache_features_across_origins: bool = Field(
        default=True,
        description=(
            "When True, the backtest loader reads the train/val/test CSVs "
            "and applies augmenters ONCE per backtest run and passes the "
            "in-memory frames to every origin.  Otherwise each origin "
            "re-reads + re-augments.  Saves ~50-120s per origin on large UK "
            "and TH categories with zero accuracy change."
        ),
    )

    # -------------------------------------------------------------------------
    # YOY Trend Adjustment Settings (pre-reconciliation)
    # -------------------------------------------------------------------------
    # Adjusts key-level forecasts based on observed year-over-year growth
    # trends BEFORE top-down reconciliation runs. For each key with
    # sufficient history, computes the YOY growth rate in the pre-period
    # and scales the forecast total to exhibit the same growth rate
    # relative to prior-year actuals. Preserves the weekly profile
    # (relative distribution across periods) via uniform scaling.
    # -------------------------------------------------------------------------

    enable_yoy_trend_adjustment: bool = Field(
        default=False,
        description=(
            "Enable year-over-year trend adjustment of key-level forecasts. "
            "When enabled, each key's forward forecast total is scaled to "
            "reflect the same YOY growth rate observed in the pre-period "
            "(last forecast_horizon periods before test_start). "
            "This runs AFTER bias calibration and exclusion zeroing, but "
            "BEFORE top-down category reconciliation. Keys with insufficient "
            "history (< yoy_trend_min_history_periods) are left unchanged."
        ),
    )

    yoy_trend_min_history_periods: int = Field(
        default=104,
        ge=24,
        description=(
            "Minimum number of historical periods (before test_start) "
            "required for a key to qualify for YOY trend adjustment. "
            "Default 104 = 2 years of weekly data. For monthly data, "
            "use 24 (2 years). Keys with fewer periods are skipped."
        ),
    )

    yoy_trend_scaling_min: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description=(
            "Minimum allowed YOY trend scaling factor. 0.5 means the "
            "adjusted forecast total won't be less than 50%% of the "
            "original model forecast total. Prevents extreme downward "
            "corrections."
        ),
    )

    yoy_trend_scaling_max: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description=(
            "Maximum allowed YOY trend scaling factor. 2.0 means the "
            "adjusted forecast total won't exceed 200%% of the original "
            "model forecast total. Prevents extreme upward corrections."
        ),
    )

    # -------------------------------------------------------------------------
    # SPLY Post-Prediction Bias Correction
    # -------------------------------------------------------------------------

    enable_sply_correction: bool = Field(
        default=True,
        description=(
            "Enable Same-Period-Last-Year (SPLY) post-prediction correction. "
            "This is the FINAL adjustment step, applied after calibration, "
            "error correction, and reconciliation. For each (key, period), "
            "blends the model forecast with the prior-year same-period actual "
            "using a learned weight per segment. Highly effective for seasonal "
            "FMCG demand. The blend weight (alpha) is learned from validation "
            "data: final = alpha * model + (1-alpha) * SPLY."
        ),
    )

    apply_moq_postprocessing: bool = Field(
        default=True,
        description=(
            "Enable Minimum-Order-Quantity (MOQ) post-processing on forecast "
            "output. Detects SKUs whose historical order pattern has a strong "
            "fixed-case-pack / pallet multiple (common on NPD, reintroduced, "
            "highly-promoted, and MOQ-driven items) and rounds the forecast "
            "to that multiple for downstream DIQ consumption. Emits a new "
            "column ``prediction_post_moq`` alongside the regression-style "
            "``predicted`` column. The column is always populated — for "
            "keys without an MOQ pattern it simply mirrors ``predicted``, "
            "so DIQ can read it unconditionally. Detection uses training-"
            "period history only (up to ``train_end``), so it is leakage-"
            "free on both inference and rolling-origin backtests. Disable "
            "to leave the forecast column untouched."
        ),
    )

    # -------------------------------------------------------------------------
    # Phase 3: Hierarchical Modeling, Features & Reconciliation
    # -------------------------------------------------------------------------

    hierarchy_cols: Any = Field(
        default_factory=list,
        description=(
            "Hierarchy column names in source data. Accepts two shapes:\n"
            "  1. A flat list, ordered coarsest to finest (legacy form) — "
            "treated as a single 'product' chain.\n"
            "  2. A dict of named chains, e.g.\n"
            "     ``{'product': ['market_name','brand_name'], "
            "'customer': ['APG_code']}`` — supports multiple parallel "
            "hierarchies for MinT reconciliation.\n"
            "If empty, the detector runs against "
            "``hierarchy_detection.candidate_cols`` (recommended) and, if "
            "that is also empty, falls back to full auto-detection over "
            "every non-key / non-date / non-target column (legacy). "
            "Resulting chains are persisted to ``seg_output/"
            "hierarchy_detection.json`` and read by every downstream stage."
        ),
    )

    hierarchy_detection: "HierarchyDetectionConfig" = Field(
        default_factory=lambda: HierarchyDetectionConfig(),
        description=(
            "Configuration for the hierarchy detector. See "
            "HierarchyDetectionConfig for the knobs."
        ),
    )

    reconciliation_method: str = Field(
        default="mint_shrink",
        description=(
            "Reconciliation method for coherent forecasts. Options: "
            "'top_down' (Prophet aggregate + proportional scaling), "
            "'bottom_up' (sum leaf forecasts), "
            "'mint_shrink' (MinT with Ledoit-Wolf shrinkage — recommended), "
            "'wls' (weighted least squares), 'ols' (ordinary least squares)."
        ),
    )

    enable_hierarchical_models: bool = Field(
        default=True,
        description=(
            "Include hierarchical model types (global_local, mixed_effects, "
            "multi_level_ensemble) in model selection candidates. These pool "
            "data across keys and hierarchy levels for improved accuracy."
        ),
    )

    enable_hierarchy_features: bool = Field(
        default=True,
        description=(
            "Create hierarchy-level temporal features: group totals, group means, "
            "share-of-group, group trend, rank-in-group. All leakage-free. "
            "Requires hierarchy_cols to be set or auto-detected."
        ),
    )

    # -------------------------------------------------------------------------
    # Phase 6: Rich Feature Engineering
    # -------------------------------------------------------------------------

    enable_rich_features: bool = Field(
        default=True,
        description=(
            "Create Phase 6 rich features: regime/state detection, cross-key "
            "relative features, and history embeddings for future-unknown features. "
            "Adds ~15-30 features depending on data structure."
        ),
    )

    # -------------------------------------------------------------------------
    # Walk-Forward Cross-Validation Settings
    # -------------------------------------------------------------------------
    # Walk-Forward CV provides temporally-aware model selection and evaluation.
    # It trains on historical data and validates on future data, mimicking
    # real-world forecasting scenarios and preventing data leakage.
    # -------------------------------------------------------------------------

    enable_walk_forward_cv: bool = Field(
        default=True,
        description=(
            "Enable Walk-Forward Cross-Validation for model selection. "
            "When True, models are evaluated across multiple temporal folds, "
            "providing realistic performance estimates and detecting concept drift. "
            "When False, uses single train/validation split (faster but less robust)."
        ),
    )

    walk_forward_n_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description=(
            "Number of folds for Walk-Forward CV. More folds provide better "
            "performance estimates but increase training time. "
            "Recommended: 3-5 for most use cases."
        ),
    )

    walk_forward_min_train_periods: int = Field(
        default=52,
        ge=12,
        description=(
            "Minimum number of training periods before first validation. "
            "52 = 1 year of weekly data, 12 = 1 year of monthly data. "
            "Ensures enough data for initial model training."
        ),
    )

    walk_forward_strategy: str = Field(
        default="expanding",
        description=(
            "CV strategy: 'expanding' or 'rolling'. "
            "'expanding' trains on all data up to origin (recommended). "
            "'rolling' uses fixed window size (good for concept drift)."
        ),
    )

    walk_forward_rolling_window: int = Field(
        default=104,
        ge=26,
        description=(
            "Window size for 'rolling' strategy (in periods). "
            "104 = 2 years of weekly data. Only used when strategy='rolling'."
        ),
    )

    # ----------------------------------------------------------------
    # Per-model-group training speed knobs.
    #
    # `run_full_training_pipeline` accepts `use_recursive_validation`
    # and `max_candidates_per_group` parameters that BOTH multiply the
    # per-model fit count substantially:
    #   - use_recursive_validation=True  -> walk-forward eval per
    #     candidate (~13 retrains for a 13-week horizon, per candidate)
    #   - max_candidates_per_group=3     -> 3 candidate model types
    #     fit per group, the best is kept
    #
    # On a 500-model run the combined factor is ~3 x 13 = 39x more
    # work than a single fit per group.  These knobs make that
    # tunable from config so users can trade off accuracy vs. speed
    # without touching code.
    # ----------------------------------------------------------------

    use_recursive_validation: bool = Field(
        default=True,
        description=(
            "Use recursive (walk-forward) validation when scoring "
            "candidate models within each group.  Multiplies per-fit "
            "cost by the forecast horizon (13 for weekly, 3 for "
            "monthly).\n\n"
            "**Keep True for accuracy.**  Recursive validation scores "
            "each candidate the way it'll actually be used at inference: "
            "the model's own predictions feed back into its lag features "
            "for the next step, which is how multi-step forecasting "
            "errors compound.  Models that look great at direct 1-step "
            "prediction can degrade badly at the 13-week horizon, and "
            "selecting on direct WAPE alone hides that.\n\n"
            "Setting False is a real accuracy trade-off (typically a "
            "small WAPE delta on smooth keys, larger on intermittent "
            "/ lumpy keys) for ~13x training speedup.  Use only after "
            "explicitly validating the accuracy delta is acceptable "
            "for your category.  Consider "
            "`recursive_validation_max_horizon` instead — it caps the "
            "walk-forward at a target lag (e.g. 5) without disabling "
            "recursive validation entirely, preserving most of the "
            "accuracy gain at a fraction of the cost."
        ),
    )

    recursive_validation_lag: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Score each candidate model on the WAPE at THIS specific "
            "forecast lag, instead of the average over all lags up to "
            "forecast_horizon.\n\n"
            "Set to None (default) to score on the full-horizon average "
            "(legacy behaviour).\n"
            "Set to e.g. 5 to score on lag-5 WAPE only.\n\n"
            "Why this matters: the recursive eval still walks lag 1, 2, "
            "..., N (each step's prediction feeds the next step's lag "
            "features, so we can't skip them).  But the SELECTION "
            "METRIC is just `wape_by_step[lag]`, not the average.  Two "
            "wins:\n"
            "  1. Walk-forward stops at `lag` instead of forecast_horizon\n"
            "     (~2.6x faster for lag=5 on a 13-week horizon).\n"
            "  2. The selection metric is the EXACT lag the downstream "
            "     pipeline cares about — e.g. for DIQ's lag-4 backtest, "
            "     selecting on lag-5 WAPE picks the model that's best "
            "     at the relevant horizon, not best on average.\n\n"
            "Smarter trade-off than `use_recursive_validation=False`: "
            "preserves the recursive-error-compounding signal up to the "
            "lag you actually care about."
        ),
    )

    max_candidates_per_group: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "How many candidate model types to fit per model group "
            "before picking the best (e.g. lightgbm + xgboost + "
            "catboost).\n\n"
            "**Keep at 3+ for accuracy.**  Different demand patterns "
            "favour different model families (smooth -> LightGBM, "
            "intermittent -> tweedie / zero-inflated, lumpy -> "
            "xgboost+ordinal).  Per-segment candidate selection is one "
            "of the main reasons the pipeline outperforms a single-"
            "model baseline.  Reducing to 1 forces every segment to use "
            "the highest-prior model type, losing that adaptivity.\n\n"
            "Setting 1 is a real accuracy trade-off for ~3x training "
            "speedup.  Use only after validating per-category that the "
            "single best-prior model performs comparably to the sweep."
        ),
    )

    # ----------------------------------------------------------------
    # Per-model-group training PARALLELISM (Ray-based).
    #
    # Default is sequential (1 worker) — the current behaviour, no
    # change for users who haven't opted in.  Setting this to N>1
    # dispatches each model_group's training to a Ray task, with
    # `num_cpus=cpu_count // N` per task so OpenMP threading stays
    # within the per-task budget instead of all 16 cores fighting on
    # every concurrent fit.
    # ----------------------------------------------------------------

    parallel_training_workers: int = Field(
        default=1,
        ge=1,
        le=1024,
        description=(
            "Number of model groups to train in parallel — drives the "
            "Spark mapInPandas partition count in the per-MG training "
            "dispatcher.  1 = sequential (default, safest).\n\n"
            "Sizing guidance: set this equal to the cluster's total "
            "worker cores for maximum throughput.  E.g. on a 16-worker "
            "x 16-core cluster (256 cores), set to 256 — Spark gets "
            "256 partitions of ~2 MGs each, runs them all in parallel "
            "with 1 OMP thread per task (LightGBM is most efficient "
            "single-threaded; sub-linear scaling at higher thread "
            "counts).  Setting equal to half the cluster cores is also "
            "fine — Spark runs 2 task waves at 2 OMP threads each.\n\n"
            "The dispatcher auto-detects cluster cores via the "
            "Databricks cluster usage tags and sizes per-task OMP "
            "threads as `cluster_cores // parallel_training_workers`, "
            "so any setting up to the cluster total uses the cluster "
            "fully.  Going above the cluster total just queues extra "
            "Spark task waves — also fine, just no extra parallelism."
        ),
    )

    walk_forward_optimize_ensemble: bool = Field(
        default=True,
        description=(
            "Optimize ensemble weights across CV folds. "
            "When True, finds optimal model combination weights that are "
            "robust across different time periods."
        ),
    )

    walk_forward_detect_drift: bool = Field(
        default=True,
        description=(
            "Enable concept drift detection. Analyzes WAPE trend across folds "
            "to detect if model performance is degrading over time. "
            "Useful for identifying when retraining is needed."
        ),
    )

    # -------------------------------------------------------------------------
    # Inference Feature Imputation Settings
    # -------------------------------------------------------------------------
    # Controls intelligent imputation of missing/zero features during inference.
    # During forecast periods, external features (price, promo, weather) and
    # long-lag features may be 0 or missing. This system detects systematically
    # missing features per period and imputes them using a tiered strategy:
    #   Numeric:
    #     Tier 1: Same-Period-Last-Year (SPLY) per key
    #     Tier 2: Category hierarchy group median at SPLY period
    #   Categorical:
    #     Priority 1: Key-invariant forward-fill (constant across training)
    #     Priority 2: SPLY per key (blank on full-history key → keep blank)
    #     Priority 3: Category hierarchy group mode at SPLY period
    # -------------------------------------------------------------------------

    enable_feature_imputation: bool = Field(
        default=True,
        description=(
            "Enable intelligent feature imputation during inference. "
            "When True, detects features that are systematically 0/missing "
            "across keys in forecast periods and imputes them using a tiered "
            "strategy (SPLY per key → hierarchy group median/mode)."
        ),
    )

    imputation_missing_threshold: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description=(
            "Fraction of keys with 0/NaN for a feature at a given period to "
            "trigger imputation. 1.0 = only impute when ALL keys have the "
            "feature as 0 or NaN (safest — avoids imputing legitimately zero "
            "features like promos). Lower values are more aggressive."
        ),
    )

    imputation_level: list = Field(
        default_factory=list,
        description=(
            "List of category hierarchy columns used for Tier 2 imputation. "
            "After per-key SPLY (Tier 1), missing features are imputed using "
            "the median from keys sharing the same hierarchy group at the "
            "same-period-last-year. E.g. ['APG_code', 'Brand', 'form'] means "
            "impute with median from keys matching on APG_code + Brand + form "
            "at SPLY. If empty, falls back to segment-based imputation."
        ),
    )

    imputation_min_history_for_invariant: int = Field(
        default=13,
        ge=4,
        description=(
            "Minimum periods of history for a key to detect key-invariant "
            "categorical features. If a categorical has the same value for "
            "all periods across this many weeks/months, it is treated as a "
            "fixed attribute (e.g., brand, channel) and forward-filled."
        ),
    )

    imputation_enable_recursive_fallbacks: bool = Field(
        default=True,
        description=(
            "Enable fallback values for recursive feature computation "
            "when key history is too short (e.g., lag_52 on a 30-week key). "
            "Uses the same tiered strategy to provide a meaningful value "
            "instead of returning 0.0."
        ),
    )

    imputation_enable_trend_adjustment: bool = Field(
        default=False,
        description=(
            "When True, SPLY values are adjusted for YoY trend: "
            "imputed = sply_value × (1 + yoy_growth_rate). "
            "Prevents stale values when there is clear growth or decline. "
            "Disabled by default as it adds complexity."
        ),
    )

    imputation_log_details: bool = Field(
        default=True,
        description=(
            "Log detailed imputation diagnostics including per-period "
            "feature flags, tier coverage stats, and save an "
            "imputation_report.json alongside forecast output."
        ),
    )

    # -------------------------------------------------------------------------
    # Feature Availability Detection — advanced controls (Phase A classifier)
    # -------------------------------------------------------------------------
    # These override the defaults inside utils/feature_availability.py.
    # When feature_availability is left at defaults, the classifier
    # uses the hard-coded values baked into that module for backward
    # compatibility with legacy configs.
    # -------------------------------------------------------------------------

    feature_availability: "FeatureAvailabilityConfig" = Field(
        default_factory=lambda: FeatureAvailabilityConfig(),
        description=(
            "Advanced tuning for the feature-availability classifier: "
            "name-pattern whitelist, auto-indicator detection, per-key "
            "active-scope thresholds, and the recent-history staleness gate."
        ),
    )

    calendar_features: "CalendarFeaturesConfig" = Field(
        default_factory=lambda: CalendarFeaturesConfig(),
        description=(
            "Deterministic calendar/holiday feature injection. When enabled, "
            "the pipeline synthesises country-specific holiday flags and "
            "Fourier terms into the source dataframe before feature "
            "availability detection runs, closing the gap where upstream "
            "ETLs fail to populate calendar columns for the future horizon."
        ),
    )


# =============================================================================
# FEATURE AVAILABILITY v2 — CONFIGURABLE THRESHOLDS
# =============================================================================

DEFAULT_ALWAYS_KNOWN_PATTERNS: List[str] = [
    # Calendar
    "year", "month", "quarter", "week", "day",
    "fourier", "sin_", "cos_", "season",
    # Holidays
    "holiday", "xmas", "christmas", "easter", "jubilee",
    "bank_holiday", "pancake", "boxing", "new_year", "good_friday",
    # Promotions
    "promo_", "promotion", "mechanic", "slogan", "display",
    "deal", "price_cut", "coupon", "off_invoice", "wigig",
    # Events
    "event", "campaign", "launch", "market_event",
    # Generic indicators
    "_flag", "_ind", "_indicator", "is_", "has_",
    # Product lifecycle
    "delist", "transition", "relaunch", "ramp_up", "ramp_down",
    # Forward-known weather
    "wthr_", "weather_", "temp_", "rain_", "sunshine",
]


class FeatureAvailabilityConfig(BaseModel):
    """Tunable knobs for the feature-availability classifier (Phase A fix).

    Defaults preserve legacy behaviour where possible while enabling the
    per-key active-scope logic and the recency staleness gate by default —
    both additive rules that only rescue features that were previously
    dropped incorrectly.
    """

    model_config = ConfigDict(extra="ignore")

    # --- Name-pattern whitelist --------------------------------------------
    always_known_patterns: List[str] = Field(
        default_factory=lambda: list(DEFAULT_ALWAYS_KNOWN_PATTERNS),
        description=(
            "Lower-cased substrings; a feature whose name contains any of "
            "these is classified known_in_future if it has at least one "
            "non-zero observation somewhere in the data. Runs BEFORE any "
            "sparsity / recency gate."
        ),
    )

    # --- Auto-indicator detection ------------------------------------------
    auto_indicator_enabled: bool = Field(
        default=True,
        description=(
            "When True, any bool / int-binary / low-cardinality object "
            "column is treated as an indicator and classified by its "
            "future presence (not by a fill-rate threshold). "
            "This is what keeps sparse-but-planned promo flags alive."
        ),
    )
    max_unique_values_object: int = Field(
        default=10, ge=2,
        description=(
            "Object dtype columns with at most this many distinct non-null "
            "values are treated as categorical indicators."
        ),
    )
    treat_binary_as_indicator: bool = Field(
        default=True,
        description=(
            "Treat numeric columns whose non-null values are all in {0,1} "
            "as binary indicators regardless of fill rate."
        ),
    )

    # --- Continuous-feature thresholds (per-key scope) ---------------------
    forward_coverage_known: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description=(
            "Min fraction of history-active keys that also have a non-zero "
            "future value for the feature to be classified known_in_future."
        ),
    )
    forward_coverage_partial: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description=(
            "Min fraction of history-active keys with a non-zero future "
            "value for the feature to be classified partially_known. "
            "Below this, the feature falls through to history_only."
        ),
    )
    min_active_keys: int = Field(
        default=5, ge=1,
        description=(
            "Minimum number of keys with non-zero history values required "
            "to consider the feature usable at all. Below this, the "
            "feature is excluded as effectively unused."
        ),
    )

    # --- Indicator-specific thresholds -------------------------------------
    indicator_min_future_nonzero_rows: int = Field(
        default=1, ge=0,
        description=(
            "Minimum count of non-zero rows in the future window for an "
            "indicator to be classified known_in_future. Set to 0 to allow "
            "name-pattern overrides to rescue indicators with no future "
            "presence (e.g. when the ETL has a gap that calendar synthesis "
            "will fix later)."
        ),
    )
    indicator_min_history_nonzero_rows: int = Field(
        default=1, ge=0,
        description=(
            "Minimum count of non-zero rows in history for an indicator to "
            "be considered useful. Sparse indicators that fire at all in "
            "history still provide signal when they re-fire in future."
        ),
    )

    # --- Recent-history staleness gate ------------------------------------
    recency_enabled: bool = Field(
        default=True,
        description=(
            "When True, features that would be classified history_only or "
            "partially_known must have recent non-zero values in the last "
            "recency_lookback_periods of history. Features whose upstream "
            "feed went stale (e.g. Nielsen stops updating) are demoted to "
            "excluded so their frozen embeddings do not poison new products."
        ),
    )
    recency_lookback_periods: int = Field(
        default=4, ge=1,
        description=(
            "Number of most-recent history periods to inspect for the "
            "staleness gate. 4 means the last 4 weeks for year_week data "
            "or the last 4 months for year_month data."
        ),
    )
    recency_min_active_keys: int = Field(
        default=5, ge=1,
        description=(
            "Minimum keys with a non-zero value in the recency window for "
            "the feature to pass the staleness gate."
        ),
    )
    recency_min_nonzero_rows: int = Field(
        default=1, ge=0,
        description=(
            "Minimum total non-zero rows in the recency window for the "
            "feature to pass the staleness gate."
        ),
    )


class HierarchyDetectionConfig(BaseModel):
    """Controls how the pipeline selects hierarchical grouping columns.

    The forecasting stack uses hierarchies in three places that all MUST
    agree: hierarchy-level temporal features, hierarchical reconciliation
    (MinT / WLS), and multi-level model training. Leaving the decision to
    a generic "any low-cardinality categorical works" auto-detector was
    producing catastrophically wrong results (e.g. binary indicator flags
    being treated as product hierarchies for UK CONDIMENT).

    The preferred pattern is therefore: the user curates a list of
    candidate columns in ``candidate_cols`` that they know are meaningful
    business groupings (APG, brand, segment, forecast family, ...). The
    detector then works out nesting relationships among ONLY those
    columns, collapses degenerate aliases (segment_name / form_name /
    sub_form_name often carry identical values), and returns separate
    named hierarchy chains: ``product``, ``customer``, ``other``.
    """

    model_config = ConfigDict(extra="ignore")

    candidate_cols: List[str] = Field(
        default_factory=list,
        description=(
            "Columns the detector is ALLOWED to treat as hierarchy. "
            "When non-empty the detector only considers these. When "
            "empty the detector falls back to full auto-detect on every "
            "non-key/date/target column (legacy behaviour). Listing "
            "missing column names produces a warning, not an error."
        ),
    )

    min_keys_per_group: int = Field(
        default=5, ge=2,
        description=(
            "Reject a hierarchy column whose finest group has fewer "
            "than this many keys. Too-fine groups give hierarchical "
            "models unstable per-group parameters."
        ),
    )
    min_cardinality: int = Field(
        default=3, ge=2,
        description=(
            "Reject columns with fewer distinct values than this. "
            "Default 3 silently drops binary 0/1 indicator flags — set "
            "``include_binary: true`` to override."
        ),
    )
    max_cardinality: int = Field(
        default=2000, ge=10,
        description=(
            "Upper bound on distinct values. Allows SKU-level codes "
            "(e.g. cs_gtin with ~240 values) to pass while still "
            "rejecting near-unique columns that are effectively keys."
        ),
    )
    require_coverage: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description=(
            "Minimum fraction of keys that must have a non-null value "
            "in the column. Drops sparse metadata columns."
        ),
    )
    include_binary: bool = Field(
        default=False,
        description=(
            "When True, binary (cardinality=2) columns pass the "
            "cardinality gate. Default False so indicator flags like "
            "``ps_pos_retailers_mapped_keys`` never masquerade as "
            "hierarchies."
        ),
    )

    product_name_patterns: List[str] = Field(
        default_factory=lambda: [
            "brand", "segment", "form", "category", "market",
            "family", "variant", "gtin", "cpl", "product",
        ],
        description=(
            "Lower-cased substrings that, when found in a column name, "
            "classify the containing chain as the product hierarchy."
        ),
    )
    customer_name_patterns: List[str] = Field(
        default_factory=lambda: [
            "apg", "customer", "account", "planning_customer",
            "retailer", "store", "channel", "banner",
        ],
        description=(
            "Lower-cased substrings that classify a chain as the "
            "customer / distribution hierarchy."
        ),
    )

    collapse_duplicate_columns: bool = Field(
        default=True,
        description=(
            "When True, columns that carry identical values for every "
            "row (e.g. form_name == segment_name == sub_form_name) are "
            "collapsed to a single representative, using ``product_name_"
            "patterns`` priority order."
        ),
    )

    max_depth_per_chain: int = Field(
        default=5, ge=1,
        description=(
            "Maximum depth of any single hierarchy chain (coarse → "
            "fine). Downstream stages may further cap this (e.g. "
            "hierarchy_features.max_hierarchy_cols defaults to 2). "
            "Defaulting to 5 keeps coarse levels like market / category "
            "available for reconciliation without spamming features."
        ),
    )


class CalendarFeaturesConfig(BaseModel):
    """Configuration for the deterministic calendar/holiday synthesiser."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(
        default=True,
        description=(
            "When True, the pipeline injects country-specific calendar "
            "features (bank holidays, pancake day, Fourier terms, "
            "week-to-holiday distances) into the source dataframe before "
            "feature availability detection runs."
        ),
    )
    overwrite_mode: Literal["never", "future_only", "always"] = Field(
        default="always",
        description=(
            "How to handle pre-existing calendar columns in the CSV. "
            "'never' only creates new columns. 'future_only' overwrites "
            "only rows beyond the history cutoff (keeps training history "
            "unchanged). 'always' overwrites every row with the truth from "
            "the calendar — safest when upstream ETLs are known-unreliable."
        ),
    )
    include_lead_lag_windows: List[int] = Field(
        default_factory=lambda: [1, 2, 4],
        description=(
            "Window sizes (in periods) for pre-/post-holiday indicator "
            "features. Each N produces is_pre_holiday_<N>w and "
            "is_post_holiday_<N>w flags."
        ),
    )
    custom_events: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional per-run custom events. Key is the output column name "
            "(snake_case_flag). Value is either a list of ISO dates "
            "(one-off events) or a callable key registered in "
            "utils.calendar_features.COUNTRY_CUSTOM_EVENTS."
        ),
    )


class DemandForecastConfig(BaseModel):
    """
    Top-level configuration object shared by all crews and pipelines.
    """

    model_config = ConfigDict(extra="ignore")  # tolerate extra YAML keys without exploding

    # -------------------------------------------------------------------------
    # LLM Provider Configuration
    # -------------------------------------------------------------------------
    # Controls which LLM backend to use for the agentic crews.
    # Options:
    #   - "bedrock": AWS Bedrock with Claude (default, for local/AWS deployment)
    #   - "databricks": Databricks AIF with GPT-5 (for Databricks deployment)
    llm_provider: str = Field(
        default="bedrock",
        description=(
            "LLM provider to use for agentic crews. "
            "Options: 'bedrock' (AWS Claude) or 'databricks' (AIF GPT-5)"
        ),
    )

    # -------------------------------------------------------------------------
    # Databricks Path Configuration
    # -------------------------------------------------------------------------
    # Base path for Databricks Unity Catalog Volumes.
    # Only used when llm_provider="databricks".
    # When set, input_data_path and artifact_base_path are resolved relative to this.
    # Example: "/Volumes/pds_feu_931272_dev/data_science_team/raw_data"
    databricks_base_path: str = Field(
        default="",
        description=(
            "Base path for Databricks Unity Catalog Volumes. "
            "Only used when llm_provider='databricks'. "
            "When set, input_data_path and artifact_base_path are resolved relative to this base. "
            "Leave empty to use local filesystem paths."
        ),
    )

    # -------------------------------------------------------------------------
    # I/O paths
    # -------------------------------------------------------------------------
    # input_data_path supports:
    #   - Relative path: resolved against project root or databricks_base_path
    #     e.g. "sourcedata/sales.csv"
    #   - Absolute path: used as-is (for data outside the repository)
    #     e.g. "/mnt/shared/data/sales.parquet", "~/data/sales.csv"
    #   - Databricks paths: "/Volumes/...", "dbfs:/..."
    #   - Delta table path: directory containing _delta_log/
    #     e.g. "/mnt/delta/sales_table" or "sourcedata/sales_delta"
    # Supported formats: .csv, .parquet, .pq, delta (directory)
    input_data_path: str
    artifact_base_path: str

    # Optional output folder for the final inference forecast file.
    # When set, the inference pipeline writes a Parquet copy of the forecast
    # to this directory in addition to the CSV in model_artifacts/.
    # Supports the same path forms as input_data_path (absolute, relative, ~/...).
    # Leave blank or omit to disable the extra Parquet output.
    output_folder_path: str = Field(
        default="",
        description=(
            "Optional output folder for the final inference forecast Parquet file. "
            "When set, a Parquet copy of inference_forecast is written here "
            "in addition to the CSV in model_artifacts/. "
            "Supports absolute, relative, and ~/... paths."
        ),
    )

    # -------------------------------------------------------------------------
    # Core schema
    # -------------------------------------------------------------------------
    prediction_key_cols: List[str]
    timestamp_col: str
    target_col: str

    numeric_feature_cols: List[str] = Field(default_factory=list)
    categorical_feature_cols: List[str] = Field(default_factory=list)

    price_features: FeatureGroup = Field(default_factory=FeatureGroup)
    promo_features: FeatureGroup = Field(default_factory=FeatureGroup)
    holiday_features: FeatureGroup = Field(default_factory=FeatureGroup)
    weather_features: FeatureGroup = Field(default_factory=FeatureGroup)

    # -------------------------------------------------------------------------
    # Country / locale (drives calendar-feature synthesis)
    # -------------------------------------------------------------------------
    country: str = Field(
        default="",
        description=(
            "ISO-3166-ish country label used by the calendar-feature "
            "synthesiser to emit bank-holiday flags for the full horizon. "
            "Common values: 'UK', 'DE', 'TH', 'US', 'IN', 'FR', 'NL', 'JP'. "
            "Leave blank to skip calendar synthesis regardless of the "
            "design.calendar_features.enabled flag."
        ),
    )
    country_subdivision: Optional[str] = Field(
        default=None,
        description=(
            "Optional subdivision for countries where bank holidays differ "
            "by region. Pass the subdivision code expected by the "
            "'holidays' library. Examples: 'ENG' / 'SCT' / 'WLS' / 'NIR' "
            "for UK, 'BY' / 'BE' for DE Länder. When None, the country-"
            "wide defaults are used."
        ),
    )

    # -------------------------------------------------------------------------
    # Time format & forecasting horizon
    # -------------------------------------------------------------------------
    time_format: str = Field(
        default="auto",
        description=(
            "Allowed values:\n"
            "  - 'auto'\n"
            "  - 'date' (YYYY-MM-DD)\n"
            "  - 'year_week' (YYYYWW)\n"
            "  - 'year_month' (YYYYMM)\n"
        ),
    )

    forecast_horizon: int

    # -------------------------------------------------------------------------
    # Run Mode
    # -------------------------------------------------------------------------
    # Controls what the pipeline produces:
    #
    #   "backtest_and_forecast" (default):
    #       Trains models on history, runs rolling-origin backtesting to
    #       validate accuracy, AND generates forward forecasts into future
    #       periods (beyond the history cutoff). Full pipeline.
    #
    #   "forecast_only":
    #       Trains on ALL available history (up to cutoff) and generates
    #       forward forecasts into the future. No backtesting, no held-out
    #       test set. Use when you trust the model and need production
    #       forecasts. Train/val split is still used for model selection.
    #
    #   "backtest_only":
    #       Trains models and runs rolling-origin backtesting to evaluate
    #       accuracy on historical data. No forward forecast produced.
    #       Use when validating model quality before deploying.
    #
    # When run_mode is set and splits are left empty (""), auto-detection
    # adapts to the mode:
    #   - backtest_and_forecast: train + val from history, test = backtest
    #     window, forward forecast generated for future periods
    #   - forecast_only: train = all history except last forecast_horizon
    #     for validation, no test set needed
    #   - backtest_only: same as backtest_and_forecast but no forward forecast
    # -------------------------------------------------------------------------
    run_mode: str = Field(
        default="backtest_and_forecast",
        description=(
            "Pipeline execution mode. "
            "'backtest_and_forecast': full pipeline with backtesting + forward forecast. "
            "'forecast_only': train on all history, produce forward forecast, no backtest. "
            "'backtest_only': backtesting for accuracy validation, no forward forecast."
        ),
    )

    # -------------------------------------------------------------------------
    # Dataset splits (optional — auto-detected if omitted)
    # -------------------------------------------------------------------------
    # When left empty (""), the system auto-detects splits from data based
    # on run_mode:
    #
    #   backtest_and_forecast / backtest_only:
    #     1. Detects history cutoff (last period with actuals > 0)
    #     2. test = last (forecast_horizon + 5) periods of history
    #     3. val = prior forecast_horizon periods
    #     4. train = everything before val
    #
    #   forecast_only:
    #     1. Detects history cutoff
    #     2. val = last forecast_horizon periods of history
    #     3. train = everything before val
    #     4. test_start/end left empty (no test set)
    #
    # Users can override any/all splits explicitly regardless of run_mode.
    # -------------------------------------------------------------------------
    train_start: str = ""
    train_end: str = ""
    val_start: str = ""
    val_end: str = ""
    test_start: str = ""
    test_end: str = ""

    # -------------------------------------------------------------------------
    # Design-time configuration
    # -------------------------------------------------------------------------
    design: DesignConfig = Field(default_factory=DesignConfig)

    # -------------------------------------------------------------------------
    # Private (computed) path context
    # -------------------------------------------------------------------------
    _config_yaml_dir: Optional[Path] = PrivateAttr(default=None)
    _project_root: Optional[Path] = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("forecast_horizon")
    @classmethod
    def _check_positive_horizon(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("forecast_horizon must be > 0")
        return v

    @field_validator("prediction_key_cols")
    @classmethod
    def _check_key_cols(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("prediction_key_cols must not be empty")
        return v

    @field_validator("time_format")
    @classmethod
    def _validate_time_format(cls, v: str) -> str:
        allowed = {"auto", "date", "year_week", "year_month"}
        if v not in allowed:
            raise ValueError(f"time_format must be one of {allowed}, got '{v}'")
        return v

    @field_validator("run_mode")
    @classmethod
    def _validate_run_mode(cls, v: str) -> str:
        allowed = {"backtest_and_forecast", "forecast_only", "backtest_only"}
        if v not in allowed:
            raise ValueError(f"run_mode must be one of {allowed}, got '{v}'")
        return v

    # -------------------------------------------------------------------------
    # Backwards-compatible aliases (prevents agent failures)
    # -------------------------------------------------------------------------
    @property
    def key_cols(self) -> List[str]:
        return self.prediction_key_cols

    @property
    def key_columns(self) -> List[str]:
        return self.prediction_key_cols

    @property
    def time_column(self) -> str:
        return self.timestamp_col

    @property
    def date_column(self) -> str:
        return self.timestamp_col

    @property
    def target_column(self) -> str:
        return self.target_col

    @property
    def dataset_path(self) -> str:
        # some older code expects dataset_path
        return self.input_data_path

    # -------------------------------------------------------------------------
    # Run mode helpers
    # -------------------------------------------------------------------------
    @property
    def should_backtest(self) -> bool:
        """Whether backtesting should run."""
        return self.run_mode in ("backtest_and_forecast", "backtest_only")

    @property
    def should_forward_forecast(self) -> bool:
        """Whether forward forecast into future periods should run."""
        return self.run_mode in ("backtest_and_forecast", "forecast_only")

    @property
    def has_test_set(self) -> bool:
        """Whether a held-out test set is configured."""
        return bool(self.test_start and self.test_end)

    # -------------------------------------------------------------------------
    # Feature helpers
    # -------------------------------------------------------------------------
    def all_numeric_features(self) -> List[str]:
        buckets = [
            self.numeric_feature_cols,
            self.price_features.numeric,
            self.promo_features.numeric,
            self.holiday_features.numeric,
            self.weather_features.numeric,
        ]
        out: List[str] = []
        seen = set()
        for bucket in buckets:
            for col in bucket:
                if col not in seen:
                    out.append(col)
                    seen.add(col)
        return out

    def all_categorical_features(self) -> List[str]:
        buckets = [
            self.categorical_feature_cols,
            self.price_features.categorical,
            self.promo_features.categorical,
            self.holiday_features.categorical,
            self.weather_features.categorical,
        ]
        out: List[str] = []
        seen = set()
        for bucket in buckets:
            for col in bucket:
                if col not in seen:
                    out.append(col)
                    seen.add(col)
        return out

    def all_allowed_columns(self) -> List[str]:
        """
        Return ALL columns that are explicitly defined in the config.
        This is the ONLY set of columns that agents should use.

        Includes:
        - prediction_key_cols (key columns)
        - timestamp_col
        - target_col
        - all numeric features (from all buckets)
        - all categorical features (from all buckets)
        """
        cols = []
        seen = set()

        # Key columns
        for col in self.prediction_key_cols:
            if col not in seen:
                cols.append(col)
                seen.add(col)

        # Timestamp
        if self.timestamp_col not in seen:
            cols.append(self.timestamp_col)
            seen.add(self.timestamp_col)

        # Target
        if self.target_col not in seen:
            cols.append(self.target_col)
            seen.add(self.target_col)

        # All numeric features
        for col in self.all_numeric_features():
            if col not in seen:
                cols.append(col)
                seen.add(col)

        # All categorical features
        for col in self.all_categorical_features():
            if col not in seen:
                cols.append(col)
                seen.add(col)

        return cols

    def get_feature_columns_only(self) -> List[str]:
        """
        Return ONLY feature columns (excludes key, timestamp, and target).
        These are the columns that can be used as model inputs.
        """
        exclude = set(self.prediction_key_cols + [self.timestamp_col, self.target_col])
        return [col for col in self.all_allowed_columns() if col not in exclude]

    def auto_detect_splits(self, df: "pd.DataFrame") -> "DemandForecastConfig":
        """
        Auto-detect train/val/test splits from data when not specified.

        Behavior depends on run_mode:

        **backtest_and_forecast** / **backtest_only**:
          1. Detect history cutoff (last period with actuals > 0)
          2. test = last (forecast_horizon + 5) periods of history
          3. val = prior forecast_horizon periods
          4. train = everything before val
          (Forward forecast periods = all periods after history cutoff)

        **forecast_only**:
          1. Detect history cutoff
          2. val = last forecast_horizon periods of history
          3. train = everything before val
          4. test_start/end left empty (no held-out test set)
          (Forward forecast periods = all periods after history cutoff)

        Only fills in splits that are empty (""). Explicit values are preserved.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame with timestamp and target columns

        Returns
        -------
        self (mutated in place)
        """
        # If all needed splits are already specified, skip
        if self.run_mode == "forecast_only":
            if all([self.train_start, self.train_end, self.val_start, self.val_end]):
                return self
        else:
            if all([self.train_start, self.train_end, self.val_start,
                    self.val_end, self.test_start, self.test_end]):
                return self

        import logging
        logger = logging.getLogger(__name__)

        # Normalise periods for safe YYYY-WW comparison (e.g., "2025-9" → "2025-09")
        from utils.period_utils import normalise_period, sort_periods, normalise_period_column
        df = normalise_period_column(df, self.timestamp_col)

        # Get sorted periods (normalised)
        all_periods = sort_periods(df[self.timestamp_col].dropna().unique())

        # Find history cutoff: last period where total target > 0
        period_totals = df.groupby(self.timestamp_col)[self.target_col].sum()
        positive_periods = [p for p in all_periods if period_totals.get(p, 0) > 0]

        if not positive_periods:
            raise ValueError("No periods with positive target values found in data")

        history_periods = sorted(positive_periods)
        n_history = len(history_periods)
        future_periods = [p for p in all_periods if p > history_periods[-1]]

        logger.info(f"Auto-detecting splits (run_mode={self.run_mode}):")
        logger.info(f"  Total periods: {len(all_periods)}, History: {n_history}, Future: {len(future_periods)}")

        if self.run_mode == "forecast_only":
            # ─── FORECAST ONLY ───────────────────────────────────────
            # No test set needed. Use val for model selection only.
            val_size = min(self.forecast_horizon, n_history // 4)
            train_size = n_history - val_size

            if train_size < 10:
                raise ValueError(
                    f"Not enough history for forecast_only splits: {n_history} periods, "
                    f"need at least train(10) + val({val_size})"
                )

            train_periods_split = history_periods[:train_size]
            val_periods_split = history_periods[train_size:]

            if not self.train_start:
                self.train_start = str(train_periods_split[0])
            if not self.train_end:
                self.train_end = str(train_periods_split[-1])
            if not self.val_start:
                self.val_start = str(val_periods_split[0])
            if not self.val_end:
                self.val_end = str(val_periods_split[-1])
            # test splits intentionally left empty for forecast_only

            logger.info(f"  Train: {self.train_start} to {self.train_end} ({train_size} periods)")
            logger.info(f"  Val:   {self.val_start} to {self.val_end} ({val_size} periods)")
            logger.info(f"  Test:  (none — forecast_only mode)")
            if future_periods:
                logger.info(f"  Forward forecast: {future_periods[0]} to {future_periods[-1]} ({len(future_periods)} periods)")

        else:
            # ─── BACKTEST_AND_FORECAST / BACKTEST_ONLY ────────────────
            # Need train + val + test (test = backtest window)
            test_size = min(self.forecast_horizon + 5, n_history // 3)
            val_size = min(self.forecast_horizon, (n_history - test_size) // 4)
            train_size = n_history - val_size - test_size

            if train_size < 10:
                raise ValueError(
                    f"Not enough history for auto-splits: {n_history} periods, "
                    f"need at least train(10) + val({val_size}) + test({test_size})"
                )

            train_periods_split = history_periods[:train_size]
            val_periods_split = history_periods[train_size:train_size + val_size]
            test_periods_split = history_periods[train_size + val_size:]

            if not self.train_start:
                self.train_start = str(train_periods_split[0])
            if not self.train_end:
                self.train_end = str(train_periods_split[-1])
            if not self.val_start:
                self.val_start = str(val_periods_split[0])
            if not self.val_end:
                self.val_end = str(val_periods_split[-1])
            if not self.test_start:
                self.test_start = str(test_periods_split[0])
            if not self.test_end:
                self.test_end = str(test_periods_split[-1])

            logger.info(f"  Train: {self.train_start} to {self.train_end} ({train_size} periods)")
            logger.info(f"  Val:   {self.val_start} to {self.val_end} ({val_size} periods)")
            logger.info(f"  Test:  {self.test_start} to {self.test_end} ({len(test_periods_split)} periods)")
            if self.run_mode == "backtest_and_forecast" and future_periods:
                logger.info(f"  Forward forecast: {future_periods[0]} to {future_periods[-1]} ({len(future_periods)} periods)")
            elif self.run_mode == "backtest_only":
                logger.info(f"  Forward forecast: (none — backtest_only mode)")

        return self

    def auto_detect_all_features(self, df: "pd.DataFrame") -> List[str]:
        """
        Auto-detect feature columns from a DataFrame.

        When design.auto_detect_features is True, every column that is NOT a
        key column, timestamp column, or target column is treated as a
        candidate feature. This allows minimal config.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame

        Returns
        -------
        List[str]
            List of auto-detected candidate feature columns
        """
        exclude = set(self.prediction_key_cols + [self.timestamp_col, self.target_col])
        return [col for col in df.columns if col not in exclude]

    # -------------------------------------------------------------------------
    # Time-format-aware defaults
    # -------------------------------------------------------------------------
    @property
    def is_monthly(self) -> bool:
        """Whether the config uses monthly (YYYYMM) time format."""
        return self.time_format == 'year_month'

    @property
    def periods_per_year(self) -> int:
        """Number of sub-periods in a year: 52 for weekly, 12 for monthly."""
        return 12 if self.is_monthly else 52

    def get_time_aware_defaults(self) -> Dict[str, Any]:
        """
        Return time-format-appropriate default values for key parameters.

        For YYYYWW (weekly):
            seasonal_period=52, lags=[1,2,4,8,13,26,52],
            rolling_windows=[4,8,13], walk_forward_min_train=52,
            walk_forward_rolling_window=104, dead_key_lookback=52

        For YYYYMM (monthly):
            seasonal_period=12, lags=[1,2,3,6,12],
            rolling_windows=[3,6,12], walk_forward_min_train=12,
            walk_forward_rolling_window=24, dead_key_lookback=12
        """
        if self.is_monthly:
            return {
                'seasonal_period': 12,
                'recommended_lags': [1, 2, 3, 6, 12],
                'rolling_windows': [3, 6, 12],
                'walk_forward_min_train_periods': 12,
                'walk_forward_rolling_window': 24,
                'dead_key_lookback': 12,
            }
        else:
            return {
                'seasonal_period': 52,
                'recommended_lags': [1, 2, 4, 8, 13, 26, 52],
                'rolling_windows': [4, 8, 13],
                'walk_forward_min_train_periods': 52,
                'walk_forward_rolling_window': 104,
                'dead_key_lookback': 52,
            }

    # -------------------------------------------------------------------------
    # Path resolution (PERMANENT FIX)
    # -------------------------------------------------------------------------
    def resolve_paths(self, yaml_dir: Path, project_root: Optional[Path] = None) -> None:
        """
        Resolve config paths based on deployment environment.

        Absolute paths (including ~) are always preserved as-is, regardless
        of deployment mode.  This allows input_data_path to point anywhere
        on the filesystem or network mount.

        For local/AWS (llm_provider='bedrock'):
            Relative paths resolved against project root:
              input_data_path: "sourcedata/X.csv"  → <project_root>/sourcedata/X.csv
              input_data_path: "/mnt/data/X.parquet" → /mnt/data/X.parquet  (absolute, kept)
              input_data_path: "~/data/X.csv"        → /home/user/data/X.csv (expanded)

        For Databricks (llm_provider='databricks'):
            Relative paths resolved against databricks_base_path:
              input_data_path: "sourcedata/X.csv" → <databricks_base_path>/sourcedata/X.csv
            Absolute paths (/, /Volumes/, dbfs:/) are preserved as-is.

        Supported input formats: .csv, .parquet/.pq, delta (directory)
        """
        yaml_dir = Path(yaml_dir).expanduser().resolve()

        if project_root is None:
            # Heuristic: if config yaml is inside a folder named "config" or "configs",
            # your real project root is one level up.
            if yaml_dir.name.lower() in {"config", "configs"}:
                project_root = yaml_dir.parent
            else:
                project_root = yaml_dir

        project_root = Path(project_root).expanduser().resolve()

        self._config_yaml_dir = yaml_dir
        self._project_root = project_root

        # Check if we're in Databricks mode with a base path configured
        if self.llm_provider == "databricks" and self.databricks_base_path:
            self._resolve_databricks_paths()
        else:
            self._resolve_local_paths(project_root)

    def _resolve_local_paths(self, project_root: Path) -> None:
        """
        Resolve paths for local/AWS deployment (bedrock mode).
        Paths are resolved relative to project_root.
        """
        def _resolve(p: str) -> str:
            pp = Path(p).expanduser()
            if pp.is_absolute():
                return str(pp.resolve())
            # IMPORTANT: resolve relative paths against project_root (NOT yaml_dir)
            return str((project_root / pp).resolve())

        self.input_data_path = _resolve(self.input_data_path)
        self.artifact_base_path = _resolve(self.artifact_base_path)
        if self.output_folder_path:
            self.output_folder_path = _resolve(self.output_folder_path)

    def _resolve_databricks_paths(self) -> None:
        """
        Resolve paths for Databricks deployment.
        Paths are resolved relative to databricks_base_path.

        Absolute paths (starting with /Volumes/, dbfs:/, or /) are preserved as-is.
        """
        base_path = self.databricks_base_path.rstrip("/")

        def _resolve_databricks(p: str) -> str:
            # Check if path is already absolute (Databricks paths)
            if p.startswith(("/Volumes/", "dbfs:/", "/dbfs/")):
                return p
            # Check if it's a Unix absolute path
            if p.startswith("/"):
                return p
            # Relative path - resolve against databricks_base_path
            return f"{base_path}/{p}"

        self.input_data_path = _resolve_databricks(self.input_data_path)
        self.artifact_base_path = _resolve_databricks(self.artifact_base_path)
        if self.output_folder_path:
            self.output_folder_path = _resolve_databricks(self.output_folder_path)

    def ensure_artifact_dirs(self) -> None:
        """
        Ensure the artifact base directory exists.

        Uses os.makedirs which works for both:
        - Local/AWS filesystem paths
        - Databricks Unity Catalog Volumes (/Volumes/...)

        This method should be called after resolve_paths() and before any crew execution.
        """
        import os
        os.makedirs(self.artifact_base_path, exist_ok=True)


# Resolve forward-referenced types used by DesignConfig.feature_availability,
# DesignConfig.calendar_features, DesignConfig.hierarchy_detection, etc.
# Pydantic v2 calls this `model_rebuild`; v1 calls it `update_forward_refs`.
# Both must succeed - if forward-refs aren't resolved, model instantiation
# raises ConfigError("field 'hierarchy_detection' not yet prepared so type
# is still a ForwardRef") on the user's first config load.
try:
    if _PYDANTIC_V2:
        DesignConfig.model_rebuild()
        DemandForecastConfig.model_rebuild()
    else:
        # v1 needs update_forward_refs() and the localns needs to include
        # all the BaseModel subclasses that DesignConfig + DemandForecastConfig
        # reference by string ("HierarchyDetectionConfig", etc.).
        _localns = {k: v for k, v in globals().items()
                    if isinstance(v, type) and issubclass(v, BaseModel)}
        DesignConfig.update_forward_refs(**_localns)
        DemandForecastConfig.update_forward_refs(**_localns)
except Exception as _rebuild_exc:  # pragma: no cover — defensive
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Pydantic forward-ref rebuild failed (%s) - some configs may fail to load",
        _rebuild_exc,
    )


def load_config_from_yaml(path: str) -> DemandForecastConfig:
    """
    Load YAML config file from disk and return a validated DemandForecastConfig.
    ALSO resolves relative paths relative to the PROJECT ROOT (parent of config/).
    """
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = DemandForecastConfig(**raw)
    cfg.resolve_paths(yaml_path.parent)  # will resolve against parent-of-config/ by default
    return cfg
