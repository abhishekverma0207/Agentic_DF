# utils/feature_reasoning.py
"""
Intelligent Feature Engineering: REASONING + EXECUTION Separation
=================================================================

This module implements the STATE-OF-THE-ART pattern for intelligent feature engineering:

1. REASONING LAYER (AI Decisions):
   - FeatureStrategyDecision: What the AI DECIDES about features
   - FeatureQualityAssessment: AI VALIDATION of feature quality
   - The LLM REASONS about data characteristics and makes DECISIONS

2. EXECUTION LAYER (Deterministic):
   - execute_feature_strategy(): Deterministic code that EXECUTES the AI's decisions
   - No LLM involved in execution - pure Python

KEY INSIGHT: Separate WHAT (AI reasoning) from HOW (deterministic execution).
- AI decides: "Use ACF-informed lags [1,4,13,52] because seasonality is 78%"
- Code executes: Creates those specific lags reliably

This prevents the common failure mode where LLMs describe code instead of running it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS FOR INTELLIGENT DECISIONS
# =============================================================================

class FeatureComplexity(str, Enum):
    """Feature engineering complexity level - AI DECIDES this."""
    MINIMAL = "minimal"      # Only essential lags + rolling
    STANDARD = "standard"    # Standard feature set
    COMPREHENSIVE = "comprehensive"  # Full feature engineering
    ADAPTIVE = "adaptive"    # Segment-specific complexity


class InteractionEmphasis(str, Enum):
    """Which feature interactions to emphasize - AI DECIDES this."""
    LAG_HEAVY = "lag_heavy"           # Emphasize historical lags
    SEASONAL_HEAVY = "seasonal_heavy"  # Emphasize seasonal patterns
    TREND_HEAVY = "trend_heavy"        # Emphasize trend features
    INTERMITTENCY = "intermittency"    # Emphasize sparse demand features
    BALANCED = "balanced"              # Equal emphasis


class EncodingStrategy(str, Enum):
    """Categorical encoding strategy - AI DECIDES this."""
    TARGET_ENCODING = "target_encoding"  # Best for trees, handles high cardinality
    ONE_HOT = "one_hot"                   # Best for linear models, low cardinality
    ORDINAL = "ordinal"                   # For ordered categories
    HASH = "hash"                         # For very high cardinality


# =============================================================================
# AI REASONING OUTPUTS (What the AI DECIDES)
# =============================================================================

@dataclass
class FeatureHypothesis:
    """
    A hypothesis about what features will be important.

    The AI REASONS about the data and formulates hypotheses.
    These guide feature creation and will be validated later.
    """
    hypothesis_id: str
    description: str
    feature_types: List[str]  # e.g., ['lag', 'seasonal', 'interaction']
    expected_importance: str  # 'high', 'medium', 'low'
    reasoning: str  # WHY the AI thinks this is important
    data_evidence: Dict[str, Any]  # Supporting data from EDA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentFeatureStrategy:
    """
    Per-segment feature strategy - AI DECIDES this for each segment.

    Different demand patterns need different feature strategies:
    - Smooth demand: Standard lags, trend features
    - Erratic demand: Rolling stats, outlier indicators
    - Intermittent: Zero indicators, demand rate features
    - Lumpy: Intermittency + volume features
    """
    segment_id: str
    demand_pattern: str  # 'smooth', 'erratic', 'intermittent', 'lumpy'

    # AI-DECIDED feature priorities
    lag_priority: str  # 'high', 'medium', 'low'
    rolling_priority: str
    seasonal_priority: str
    trend_priority: str
    intermittency_priority: str

    # AI-DECIDED specific configurations
    recommended_lags: List[int]
    recommended_windows: List[int]
    needs_intermittency_features: bool
    needs_trend_decomposition: bool

    # AI reasoning
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureStrategyDecision:
    """
    The complete AI DECISION about feature engineering strategy.

    This is the OUTPUT of AI REASONING - the AI analyzes EDA outputs
    and DECIDES what feature engineering approach to take.

    The deterministic executor then IMPLEMENTS these decisions.
    """
    # Metadata (required fields first)
    strategy_id: str
    timestamp: str

    # Global feature engineering decisions
    complexity_level: FeatureComplexity
    primary_emphasis: InteractionEmphasis
    encoding_strategy: EncodingStrategy

    # ACF-informed lag decisions (AI reasons about which lags matter)
    target_lags: List[int]
    feature_lags: List[int]
    acf_informed: bool
    acf_reasoning: str  # WHY these lags were chosen

    # Rolling window decisions
    rolling_windows: List[int]
    rolling_stats: List[str]
    rolling_reasoning: str

    # Seasonal decisions (AI reasons about detected seasonality)
    seasonal_period: int
    fourier_order: int
    has_strong_seasonality: bool
    seasonality_reasoning: str

    # Trend decisions
    include_trend_features: bool
    trend_strength: float  # 0-1
    trend_reasoning: str

    # Intermittency decisions (critical for sparse demand)
    include_intermittency_features: bool
    intermittency_pct: float  # % of series that are intermittent
    intermittency_reasoning: str

    # Changepoint decisions (structural breaks)
    include_changepoint_indicators: bool
    changepoint_dates: List[str]
    changepoint_reasoning: str

    # Fields with defaults MUST come after fields without defaults
    source: str = "AI_REASONING"

    # Per-segment strategies (AI adapts per segment)
    segment_strategies: Dict[str, SegmentFeatureStrategy] = field(default_factory=dict)

    # Feature hypotheses (AI formulates and will validate)
    hypotheses: List[FeatureHypothesis] = field(default_factory=list)

    # Transformation decisions
    apply_log_transform: bool = False
    apply_differencing: bool = False
    transformation_reasoning: str = ""

    # Phase 3: Hierarchy columns (product/customer dimensions) used for
    # hierarchy-level temporal features. Persisted here so backtest and
    # inference regeneration re-create the SAME hier features the trained
    # models expect. Resolved from hierarchy_detection.json at training
    # time, defaults to empty list so older JSONs remain loadable.
    hierarchy_cols: List[str] = field(default_factory=list)

    # Phase 6: Future-unknown features that should be embedded for history
    # context but NOT used as live external features at forecast time.
    future_unknown_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert enums to strings
        d['complexity_level'] = self.complexity_level.value
        d['primary_emphasis'] = self.primary_emphasis.value
        d['encoding_strategy'] = self.encoding_strategy.value
        return d

    def save(self, filepath: str) -> None:
        """Save strategy decision to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved feature strategy decision to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeatureStrategyDecision':
        """Load strategy decision from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert string enums back
        data['complexity_level'] = FeatureComplexity(data['complexity_level'])
        data['primary_emphasis'] = InteractionEmphasis(data['primary_emphasis'])
        data['encoding_strategy'] = EncodingStrategy(data['encoding_strategy'])

        # Reconstruct segment strategies
        if 'segment_strategies' in data:
            for seg_id, seg_data in data['segment_strategies'].items():
                data['segment_strategies'][seg_id] = SegmentFeatureStrategy(**seg_data)

        # Reconstruct hypotheses
        if 'hypotheses' in data:
            data['hypotheses'] = [FeatureHypothesis(**h) for h in data['hypotheses']]

        return cls(**data)


@dataclass
class FeatureQualityAssessment:
    """
    AI ASSESSMENT of feature quality after execution.

    The AI VALIDATES that features were created correctly and
    assesses whether the hypotheses were confirmed.
    """
    # Overall assessment
    overall_quality: str  # 'excellent', 'good', 'acceptable', 'poor'
    quality_score: float  # 0-100

    # Feature counts
    total_features_created: int
    lag_features_count: int
    rolling_features_count: int
    seasonal_features_count: int
    intermittency_features_count: int
    other_features_count: int

    # Data quality checks
    missing_value_pct: float
    infinite_value_pct: float
    constant_features: List[str]  # Features with no variance
    high_correlation_pairs: List[Tuple[str, str, float]]  # Highly correlated features

    # Hypothesis validation
    hypotheses_confirmed: List[str]
    hypotheses_rejected: List[str]
    hypothesis_assessment: str  # AI reasoning about hypotheses

    # Concerns and recommendations
    concerns: List[str]
    recommendations: List[str]

    # AI reasoning summary
    reasoning_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str) -> None:
        """Save quality assessment to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# HELPER FUNCTIONS FOR AI REASONING
# =============================================================================

def load_eda_insights(eda_dir: str) -> Dict[str, Any]:
    """
    Load ALL EDA insights for AI reasoning.

    Returns a comprehensive dict with all EDA outputs that the AI
    should reason about when making feature engineering decisions.
    """
    insights = {
        'eda_context': {},
        'autocorrelation': {},
        'seasonality': {},
        'trend': {},
        'changepoints': {},
        'intermittency': {},
        'data_quality': {},
    }

    # Core EDA context
    eda_ctx_path = os.path.join(eda_dir, 'eda_to_feature_context.json')
    if os.path.exists(eda_ctx_path):
        with open(eda_ctx_path) as f:
            insights['eda_context'] = json.load(f)

    # Autocorrelation analysis (ACF/PACF)
    acf_path = os.path.join(eda_dir, 'autocorrelation_summary.csv')
    if os.path.exists(acf_path):
        acf_df = pd.read_csv(acf_path)
        # Extract significant lags
        all_lags = set()
        if 'significant_lags' in acf_df.columns:
            for lags_str in acf_df['significant_lags'].dropna():
                if isinstance(lags_str, str):
                    try:
                        lags = json.loads(lags_str.replace("'", '"'))
                        all_lags.update(lags)
                    except:
                        pass
        insights['autocorrelation'] = {
            'significant_lags': sorted(list(all_lags))[:15],
            'n_series_analyzed': len(acf_df),
            'has_acf_data': True,
        }

    # Seasonality detection
    seas_path = os.path.join(eda_dir, 'seasonality_analysis.json')
    if os.path.exists(seas_path):
        with open(seas_path) as f:
            insights['seasonality'] = json.load(f)

    # Trend analysis
    trend_path = os.path.join(eda_dir, 'trend_analysis.json')
    if os.path.exists(trend_path):
        with open(trend_path) as f:
            insights['trend'] = json.load(f)

    # Changepoint detection
    cp_path = os.path.join(eda_dir, 'changepoint_analysis.json')
    if os.path.exists(cp_path):
        with open(cp_path) as f:
            insights['changepoints'] = json.load(f)

    # Intermittency analysis
    interm_path = os.path.join(eda_dir, 'intermittency_analysis.json')
    if os.path.exists(interm_path):
        with open(interm_path) as f:
            insights['intermittency'] = json.load(f)

    # Data quality report
    dq_path = os.path.join(eda_dir, 'data_quality_report.json')
    if os.path.exists(dq_path):
        with open(dq_path) as f:
            insights['data_quality'] = json.load(f)

    return insights


def load_segmentation_insights(seg_dir: str) -> Dict[str, Any]:
    """
    Load ALL segmentation insights for AI reasoning.
    """
    insights = {
        'segment_profiles': {},
        'feature_recommendations': {},
        'model_recommendations': {},
        'segment_to_feature_context': {},
        'n_segments': 0,
        'segment_sizes': {},
    }

    # Segment profiles
    profiles_path = os.path.join(seg_dir, 'segment_profiles.json')
    if os.path.exists(profiles_path):
        with open(profiles_path) as f:
            insights['segment_profiles'] = json.load(f)

    # Feature recommendations
    feat_rec_path = os.path.join(seg_dir, 'feature_recommendations.json')
    if os.path.exists(feat_rec_path):
        with open(feat_rec_path) as f:
            insights['feature_recommendations'] = json.load(f)

    # Model recommendations
    model_rec_path = os.path.join(seg_dir, 'model_recommendations.json')
    if os.path.exists(model_rec_path):
        with open(model_rec_path) as f:
            insights['model_recommendations'] = json.load(f)

    # Context for feature crew
    ctx_path = os.path.join(seg_dir, 'segmentation_to_feature_context.json')
    if os.path.exists(ctx_path):
        with open(ctx_path) as f:
            insights['segment_to_feature_context'] = json.load(f)

    # Segment assignments
    seg_df_path = os.path.join(seg_dir, 'per_key_with_segments.csv')
    if os.path.exists(seg_df_path):
        seg_df = pd.read_csv(seg_df_path)
        if 'segment_id' in seg_df.columns:
            insights['n_segments'] = seg_df['segment_id'].nunique()
            insights['segment_sizes'] = seg_df['segment_id'].value_counts().to_dict()

    return insights


def create_reasoning_context(
    eda_dir: str,
    seg_dir: str,
) -> Dict[str, Any]:
    """
    Create comprehensive context for AI reasoning.

    This is what the AI "sees" when making feature engineering decisions.
    It includes all EDA and segmentation insights in a structured format.
    """
    eda_insights = load_eda_insights(eda_dir)
    seg_insights = load_segmentation_insights(seg_dir)

    # Structure context for AI reasoning
    context = {
        'data_characteristics': {
            'n_segments': seg_insights['n_segments'],
            'segment_sizes': seg_insights['segment_sizes'],
            'data_quality': eda_insights.get('data_quality', {}),
        },
        'temporal_patterns': {
            'autocorrelation': eda_insights.get('autocorrelation', {}),
            'seasonality': eda_insights.get('seasonality', {}),
            'trend': eda_insights.get('trend', {}),
            'changepoints': eda_insights.get('changepoints', {}),
        },
        'demand_patterns': {
            'intermittency': eda_insights.get('intermittency', {}),
            'segment_profiles': seg_insights.get('segment_profiles', {}),
        },
        'upstream_recommendations': {
            'eda_context': eda_insights.get('eda_context', {}),
            'feature_recommendations': seg_insights.get('feature_recommendations', {}),
            'model_recommendations': seg_insights.get('model_recommendations', {}),
        },
    }

    return context


# =============================================================================
# DETERMINISTIC EXECUTION (No LLM - Pure Python)
# =============================================================================

def execute_feature_strategy(
    df: pd.DataFrame,
    strategy: FeatureStrategyDecision,
    key_cols: List[str],
    date_col: str,
    target_col: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    forecast_lag: int,
    categorical_cols: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    allowed_external_cols: Optional[List[str]] = None,
    # Phase 3: Hierarchy columns for hierarchy-level temporal features
    hierarchy_cols: Optional[List[str]] = None,
    # Phase 6: Future-unknown features for history embeddings
    future_unknown_features: Optional[List[str]] = None,
) -> 'FeatureEngineeringResult':
    """
    DETERMINISTIC execution of the AI's feature strategy.

    This function IMPLEMENTS the AI's decisions without any LLM involvement.
    It's pure Python that reliably executes whatever strategy the AI decided.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    strategy : FeatureStrategyDecision
        The AI's DECIDED feature engineering strategy
    key_cols, date_col, target_col : str
        Standard column specifications
    train_start, train_end, etc : str
        Date range specifications
    forecast_lag : int
        Forecast lag for leakage prevention
    categorical_cols : List[str], optional
        Categorical columns to encode
    output_dir : str, optional
        Directory to save outputs
    allowed_external_cols : List[str], optional
        Allowed external feature columns

    Returns
    -------
    FeatureEngineeringResult
        Result object with train/val/test DataFrames and metadata
    """
    from utils.feature_engineering import run_leakage_free_feature_pipeline

    logger.info(f"Executing feature strategy: {strategy.strategy_id}")
    logger.info(f"  Complexity: {strategy.complexity_level.value}")
    logger.info(f"  Emphasis: {strategy.primary_emphasis.value}")
    logger.info(f"  Target lags: {strategy.target_lags}")
    logger.info(f"  External feature lags: {strategy.feature_lags}")
    logger.info(f"  Rolling windows: {strategy.rolling_windows}")

    # Build EDA context from AI decisions
    eda_context = {
        'lag_recommendations': {
            'target_lags': strategy.target_lags,
            'acf_informed': strategy.acf_informed,
            'seasonal_period': strategy.seasonal_period,
        },
        'rolling_features': {
            'windows': strategy.rolling_windows,
            'stats': strategy.rolling_stats,
        },
        'intermittency_features': {
            'needed': strategy.include_intermittency_features,
        },
        'transformation': {
            'log_transform': strategy.apply_log_transform,
            'differencing': strategy.apply_differencing,
        },
        'seasonal_config': {
            'period': strategy.seasonal_period,
            'fourier_order': strategy.fourier_order,
        },
    }

    # Execute deterministically using existing pipeline
    result = run_leakage_free_feature_pipeline(
        df=df,
        key_cols=key_cols,
        date_col=date_col,
        target_col=target_col,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        forecast_lag=forecast_lag,
        segment_col='segment_id' if 'segment_id' in df.columns else None,
        categorical_cols=categorical_cols,
        output_dir=output_dir,
        eda_context=eda_context,
        target_lags=strategy.target_lags,
        rolling_windows=strategy.rolling_windows,
        rolling_stats=strategy.rolling_stats,
        include_intermittency_features=strategy.include_intermittency_features,
        apply_log_transform=strategy.apply_log_transform,
        apply_differencing=strategy.apply_differencing,
        include_trend_features=strategy.include_trend_features,
        include_fourier_features=strategy.has_strong_seasonality,
        fourier_order=strategy.fourier_order,
        seasonal_period=strategy.seasonal_period,
        allowed_external_cols=allowed_external_cols,
        # Pass external feature lags from AI strategy (for price, promo, weather effects)
        external_feature_lags=strategy.feature_lags if strategy.feature_lags else None,
        # Phase 3: Hierarchy columns for hierarchy-level temporal features
        hierarchy_cols=hierarchy_cols,
        # Phase 6: Rich features + history embeddings
        enable_rich_features=True,
        future_unknown_features=future_unknown_features,
    )

    # Persist hierarchy / future-unknown feature lists onto the strategy
    # object BEFORE saving, so the JSON written here contains everything
    # backtest and inference regen need to re-create the same features.
    # Without this, hierarchy_cols=None flows into regen → hier temporal
    # features silently vanish → retrained/inference models train on a
    # different feature set than the original spec.
    if hierarchy_cols:
        strategy.hierarchy_cols = list(hierarchy_cols)
    if future_unknown_features:
        strategy.future_unknown_features = list(future_unknown_features)

    # Save strategy metadata
    if output_dir:
        strategy.save(os.path.join(output_dir, 'feature_strategy_decision.json'))

        # Enhance feature metadata with AI reasoning
        meta_path = os.path.join(output_dir, 'feature_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta['ai_strategy'] = {
                'strategy_id': strategy.strategy_id,
                'complexity_level': strategy.complexity_level.value,
                'primary_emphasis': strategy.primary_emphasis.value,
                'acf_reasoning': strategy.acf_reasoning,
                'seasonality_reasoning': strategy.seasonality_reasoning,
                'intermittency_reasoning': strategy.intermittency_reasoning,
                'hypotheses': [h.to_dict() for h in strategy.hypotheses],
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2, default=str)

    logger.info(f"Feature strategy executed: {result.n_features_created} features created")

    return result


def validate_features_and_assess_quality(
    feat_dir: str,
    strategy: FeatureStrategyDecision,
) -> FeatureQualityAssessment:
    """
    DETERMINISTIC validation of feature engineering outputs.

    This computes objective metrics that the AI Analyst will then
    REASON about to provide qualitative assessment.

    Returns raw metrics that the AI will interpret.
    """
    # Load feature files (format-agnostic — parquet preferred, CSV fallback).
    from utils.feature_io import (
        features_intermediate_exists, read_features_intermediate,
    )
    if not features_intermediate_exists(feat_dir, 'train_features'):
        raise FileNotFoundError(
            f"Training features not found: {feat_dir}/train_features.[parquet|csv]"
        )

    train_df = read_features_intermediate(feat_dir, 'train_features')

    # Count features by type
    all_cols = train_df.columns.tolist()
    lag_cols = [c for c in all_cols if '_lag_' in c]
    rolling_cols = [c for c in all_cols if '_rolling_' in c or '_ewm_' in c]
    seasonal_cols = [c for c in all_cols if 'fourier_' in c or 'seasonal_' in c or '_sin_' in c or '_cos_' in c]
    # Count actual intermittency features created by create_intermittency_features()
    # Patterns: demand_occurred, periods_since_nonzero, demand_freq_, avg_nonzero, std_nonzero
    # Exclude segment-level features like adi_, cv2_, zero_fraction_clean
    interm_patterns = ['demand_occurred', 'periods_since_nonzero', 'demand_freq_', 'avg_nonzero', 'std_nonzero']
    interm_cols = [c for c in all_cols if any(p in c for p in interm_patterns)]

    # Data quality metrics
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    missing_pct = train_df[numeric_cols].isna().mean().mean() * 100
    inf_pct = np.isinf(train_df[numeric_cols].select_dtypes(include=[np.number])).mean().mean() * 100

    # Find constant features
    constant_cols = []
    for col in numeric_cols:
        if train_df[col].nunique() <= 1:
            constant_cols.append(col)

    # Find highly correlated pairs (sample for efficiency)
    high_corr_pairs = []
    if len(numeric_cols) > 2:
        sample_cols = list(numeric_cols)[:50]  # Limit for efficiency
        corr_matrix = train_df[sample_cols].corr().abs()
        for i in range(len(sample_cols)):
            for j in range(i+1, len(sample_cols)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((sample_cols[i], sample_cols[j], corr_matrix.iloc[i, j]))

    # Overall quality score
    quality_score = 100.0
    quality_score -= missing_pct * 2  # Penalize missing values
    quality_score -= inf_pct * 5      # Penalize infinite values
    quality_score -= len(constant_cols) * 0.5  # Penalize constant features
    quality_score -= len(high_corr_pairs) * 0.2  # Penalize redundancy
    quality_score = max(0, min(100, quality_score))

    # Determine overall quality level
    if quality_score >= 90:
        overall_quality = 'excellent'
    elif quality_score >= 75:
        overall_quality = 'good'
    elif quality_score >= 50:
        overall_quality = 'acceptable'
    else:
        overall_quality = 'poor'

    # Build concerns list
    concerns = []
    if missing_pct > 5:
        concerns.append(f"High missing value rate: {missing_pct:.1f}%")
    if inf_pct > 0:
        concerns.append(f"Infinite values detected: {inf_pct:.2f}%")
    if len(constant_cols) > 5:
        concerns.append(f"Many constant features ({len(constant_cols)}): consider removing")
    if len(high_corr_pairs) > 10:
        concerns.append(f"High feature redundancy ({len(high_corr_pairs)} correlated pairs)")
    if len(lag_cols) == 0:
        concerns.append("No lag features created - critical for time series!")

    # Build recommendations
    recommendations = []
    if len(constant_cols) > 0:
        recommendations.append(f"Remove constant features: {constant_cols[:5]}")
    if len(high_corr_pairs) > 5:
        recommendations.append("Consider feature selection to reduce redundancy")
    if strategy.include_intermittency_features and len(interm_cols) == 0:
        recommendations.append("Intermittency features requested but not created - check input data")

    return FeatureQualityAssessment(
        overall_quality=overall_quality,
        quality_score=quality_score,
        total_features_created=len(all_cols),
        lag_features_count=len(lag_cols),
        rolling_features_count=len(rolling_cols),
        seasonal_features_count=len(seasonal_cols),
        intermittency_features_count=len(interm_cols),
        other_features_count=len(all_cols) - len(lag_cols) - len(rolling_cols) - len(seasonal_cols) - len(interm_cols),
        missing_value_pct=missing_pct,
        infinite_value_pct=inf_pct,
        constant_features=constant_cols,
        high_correlation_pairs=high_corr_pairs[:10],  # Limit output
        hypotheses_confirmed=[],  # AI will fill this
        hypotheses_rejected=[],   # AI will fill this
        hypothesis_assessment="",  # AI will provide reasoning
        concerns=concerns,
        recommendations=recommendations,
        reasoning_summary="",  # AI will provide summary
    )


# =============================================================================
# PROMPTS FOR AI REASONING (Used by Feature Crew)
# =============================================================================

STRATEGIST_REASONING_PROMPT = """
You are an expert Feature Engineering Strategist. Your job is to REASON about the data
and DECIDE what feature engineering strategy to use.

## YOUR TASK: Make INTELLIGENT DECISIONS about features

You will be given:
1. EDA insights (autocorrelation, seasonality, trends, changepoints)
2. Segmentation insights (demand patterns, segment profiles)
3. Data characteristics

You must REASON and DECIDE:
- What lags to use and WHY (justify with ACF analysis)
- What rolling windows to use and WHY
- Whether to include intermittency features and WHY
- What seasonal features to use and WHY (justify with FFT analysis)
- What feature complexity level and WHY
- Form HYPOTHESES about what will be important

## OUTPUT FORMAT

You MUST output a valid JSON object with this structure:
```json
{
  "strategy_id": "unique_id",
  "complexity_level": "minimal|standard|comprehensive|adaptive",
  "primary_emphasis": "lag_heavy|seasonal_heavy|trend_heavy|intermittency|balanced",
  "encoding_strategy": "target_encoding|one_hot|ordinal|hash",

  "target_lags": [1, 4, 13, 52],
  "feature_lags": [1, 4],
  "acf_informed": true,
  "acf_reasoning": "WHY you chose these lags based on ACF analysis",

  "rolling_windows": [4, 13, 26, 52],
  "rolling_stats": ["mean", "std", "min", "max"],
  "rolling_reasoning": "WHY you chose these windows",

  "seasonal_period": 52,
  "fourier_order": 3,
  "has_strong_seasonality": true,
  "seasonality_reasoning": "WHY based on FFT analysis",

  "include_trend_features": true,
  "trend_strength": 0.35,
  "trend_reasoning": "WHY based on trend analysis",

  "include_intermittency_features": true,
  "intermittency_pct": 0.45,
  "intermittency_reasoning": "WHY based on demand pattern analysis",

  "include_changepoint_indicators": false,
  "changepoint_dates": [],
  "changepoint_reasoning": "WHY based on changepoint detection",

  "apply_log_transform": false,
  "apply_differencing": false,
  "transformation_reasoning": "WHY or why not apply transformations",

  "hypotheses": [
    {
      "hypothesis_id": "H1",
      "description": "Lag-52 will be highly important due to strong annual seasonality",
      "feature_types": ["lag", "seasonal"],
      "expected_importance": "high",
      "reasoning": "FFT shows 78% of series have strong annual patterns",
      "data_evidence": {"seasonality_pct": 0.78}
    }
  ]
}
```

## CRITICAL RULES
1. Base ALL decisions on the provided data - no hallucinations
2. Justify EVERY decision with specific data evidence
3. Form testable hypotheses about feature importance
4. Output ONLY valid JSON - no explanatory text outside the JSON
"""


ANALYST_REASONING_PROMPT = """
You are an expert Feature Quality Analyst. Your job is to REASON about the feature
engineering results and VALIDATE the strategy.

## YOUR TASK: Assess feature quality and validate hypotheses

You will be given:
1. Feature quality metrics (counts, missing values, correlations)
2. The original strategy with hypotheses
3. Feature metadata

You must REASON and ASSESS:
- Is the overall feature quality acceptable? WHY?
- Were the strategy's hypotheses confirmed? Which ones and WHY?
- Are there any concerns about the features? What and WHY?
- What recommendations do you have for improvement?

## OUTPUT FORMAT

You MUST output a valid JSON object with this structure:
```json
{
  "overall_quality": "excellent|good|acceptable|poor",
  "quality_reasoning": "WHY you gave this rating",

  "hypotheses_confirmed": ["H1", "H3"],
  "hypotheses_rejected": ["H2"],
  "hypothesis_assessment": "Detailed reasoning about which hypotheses were confirmed/rejected and WHY",

  "concerns": [
    "Specific concern 1 with reasoning",
    "Specific concern 2 with reasoning"
  ],

  "recommendations": [
    "Specific actionable recommendation 1",
    "Specific actionable recommendation 2"
  ],

  "reasoning_summary": "Overall assessment summary with key insights"
}
```

## CRITICAL RULES
1. Base ALL assessments on the provided metrics - no hallucinations
2. Be SPECIFIC about concerns and recommendations
3. Validate hypotheses based on actual feature creation results
4. Output ONLY valid JSON - no explanatory text outside the JSON
"""
