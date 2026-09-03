# FEU-Agentic-Forecasting

Agentic demand forecasting pipeline for FMCG/CPG categories. Uses LLM-powered CrewAI agents for analysis and decision-making, with deterministic execution for core computation (EDA, feature engineering, model training, inference, backtesting).

The pipeline supports 35+ model families, automatic feature availability detection, hierarchical reconciliation, and multi-horizon direct forecasting. Designed for weekly DACH (Germany/Austria/Switzerland) category data but generalises to any key-level time series.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Run Modes](#run-modes)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Pipeline Stages](#pipeline-stages)
6. [Model Families](#model-families)
7. [Feature Engineering](#feature-engineering)
8. [Hierarchical Reconciliation](#hierarchical-reconciliation)
9. [Bias Calibration](#bias-calibration)
10. [Project Structure](#project-structure)
11. [Deployment](#deployment)

---

## Pipeline Overview

The pipeline executes 8 stages sequentially, with stages 6-8 conditional on `run_mode`:

```
1. EDA                      -- Syntetos-Boylan classification, seasonality, ACF, data profiling
2. Feature Availability     -- Auto-classify features as known_in_future / history_only / partially_known
3. Segmentation             -- GMM/KMeans/HDBSCAN clustering + hybrid business-dimension segments
4. Feature Engineering      -- Leakage-free lag, rolling, seasonal, hierarchy, regime, cross-key features
5. Model Training           -- Train 35+ model families per segment, walk-forward CV, model selection
6. Inference                -- Forward forecast generation + reconciliation (if run_mode != backtest_only)
7. Diagnostics              -- WAPE/accuracy analysis on inference results (if run_mode != backtest_only)
8. Backtesting              -- Rolling-origin backtesting (if run_mode != forecast_only)
```

Each stage writes artifacts to `{artifact_base_path}/{stage}_output/`. Inter-stage communication uses Pydantic-validated JSON context files (defined in `config/schema.py`).

---

## Run Modes

| Mode | Description | Stages Run |
|------|-------------|------------|
| `backtest_and_forecast` | Full pipeline: backtest + forward forecast (default) | 1-8 |
| `forecast_only` | Train on all history, produce forward forecast, no backtest | 1-7 |
| `backtest_only` | Backtesting for accuracy validation, no forward forecast | 1-5, 8 |

---

## Quick Start

### Full pipeline

```bash
python runner.py --config config/config_de_skincare.yaml
```

### With email notifications (Mac Outlook Desktop)

```bash
python runner.py --config config/config_de_skincare.yaml --email
python runner.py --config config/config_de_skincare.yaml --email --to user@example.com
```

### Individual stages

```bash
python run_eda.py --config config/config_de_skincare.yaml
python run_feature_availability.py --config config/config_de_skincare.yaml
python run_segmentation.py --config config/config_de_skincare.yaml
python run_feature.py --config config/config_de_skincare.yaml
python run_training.py --config config/config_de_skincare.yaml
python run_inference.py --config config/config_de_skincare.yaml
python run_diagnostic.py --config config/config_de_skincare.yaml
python run_backtesting.py --config config/config_de_skincare.yaml
```

### Databricks deployment

```bash
python databricks_runner.py --config /Workspace/Repos/.../config/config.yaml
```

---

## Configuration

Configuration is a simplified YAML file with extensive auto-detection. The system auto-detects train/val/test splits from data and hierarchy columns from the product master when left empty.

### Minimal config

```yaml
llm_provider: "bedrock"
input_data_path: "sourcedata/DACH/SKIN_CARE.csv"
artifact_base_path: "artifacts_dach_skincare"

prediction_key_cols:
  - "Model_Hierarchy"
timestamp_col: "Shipment_week"
target_col: "Actuals"
time_format: "year_week"
forecast_horizon: 13
run_mode: "backtest_and_forecast"

# Leave empty for auto-detection (recommended)
train_start: ""
train_end: ""
val_start: ""
val_end: ""
test_start: ""
test_end: ""
```

### Design section (key toggles)

```yaml
design:
  # Hierarchical modeling and features (Phase 3)
  enable_hierarchy_features: true
  enable_hierarchical_models: true
  reconciliation_method: "mint_shrink"
  hierarchy_cols: []                          # Auto-detected if empty

  # Feature availability detection (Phase 1)
  enable_feature_availability_detection: true
  auto_detect_features: true

  # Rich feature engineering (Phase 6)
  enable_rich_features: true

  # Multi-horizon direct forecasting (Phase 4)
  enable_multi_horizon: true

  # Walk-forward cross-validation
  enable_walk_forward_cv: true

  # Bias calibration
  apply_bias_calibration: true

  # Top-down category reconciliation (legacy fallback)
  enable_top_down_reconciliation: true

  # Reviewer agent (false = faster, fewer tokens)
  enable_reviewer: false
```

### Auto-detection behaviour

**Train/val/test splits** (when left empty):
- Detects history cutoff (last period with actuals > 0)
- `test` = last `(forecast_horizon + 5)` periods of history
- `val` = prior `forecast_horizon` periods
- `train` = everything before `val`
- For `forecast_only`: `val` = last `forecast_horizon` periods, no test set

**Hierarchy columns** (when `hierarchy_cols: []`):
- Auto-detected from product master columns: SubCategory, Brand, VolumeSegment, APG, etc.
- Used for hierarchy temporal features and reconciliation

**Features** (when `auto_detect_features: true`):
- Every column not in key/timestamp/target is a candidate feature
- Feature Availability Detection classifies each as known_in_future, history_only, partially_known, or excluded

### Period format

Supports YYYY-WW (dash-separated week) format via `period_utils.py`. Normalises periods for safe lexicographic comparison (e.g., `"2025-9"` becomes `"2025-09"`). Also supports YYYYWW, YYYY-MM, YYYYMM, and date strings.

### Pre-built category configs

10 DACH category configs are included:

```
config/config_de_cooking_aids.yaml
config/config_de_deo.yaml
config/config_de_dressings.yaml
config/config_de_fab_clean.yaml
config/config_de_haircare.yaml
config/config_de_healthy_snack.yaml
config/config_de_home_hygiene.yaml
config/config_de_oral_care.yaml
config/config_de_skincare.yaml
config/config_de_skincleansing.yaml
```

### LLM providers

| Provider | Config file | Use case |
|----------|-------------|----------|
| AWS Bedrock (Claude) | `config/llm_config.py` | Local / AWS deployment |
| Databricks AIF | `config/llm_config_databricks.py` | Databricks deployment |

Set `llm_provider: "bedrock"` or `llm_provider: "databricks"` in the YAML.

---

## Pipeline Stages

### 1. EDA

Deterministic analysis (no LLM for core computation). Produces:
- Syntetos-Boylan demand pattern classification (smooth/erratic/intermittent/lumpy)
- ACF-informed lag recommendations
- Seasonality detection and seasonal period
- Per-key metrics (`per_key_metrics.csv`)
- Context files for downstream crews (segmentation, feature, training recommendations)

### 2. Feature Availability Detection

Auto-detects the history cutoff from data and classifies every feature column:
- **known_in_future**: available during forecast periods (e.g., price plans, promo calendar)
- **history_only**: only in historical data, converted to frozen key-level embeddings
- **partially_known**: partially available in future, used where present and imputed beyond
- **excluded**: too sparse or uninformative

Thresholds are configurable (`feature_availability_future_known_threshold`, etc.).

### 3. Segmentation

Clustering + hybrid business-dimension segmentation:
- Algorithms: GMM (default), KMeans, HDBSCAN, Hierarchical
- Features: volume_mean, cv_clean, zero_fraction_clean, adi_log, demand_frequency
- Hybrid dimensions: volume_tier, demand_pattern
- Intermittency-aware merge: prevents mixing regular and sparse keys in the same segment
- Enhanced profiles: external response profiles, seasonality shape, hierarchy detection

### 4. Feature Engineering

Leakage-free feature pipeline with `shift(forecast_lag)` on all derived features:

- **Lag features**: ACF-informed (default: 1, 2, 4, 8, 13, 26, 52)
- **Rolling statistics**: mean, std, min, max over configurable windows (default: 4, 8, 13)
- **EWM features**: exponentially weighted moving averages
- **Seasonal features**: seasonal decomposition, Fourier terms
- **Calendar features**: week-of-year, month, quarter
- **Cross-sectional features**: rank within period, relative to period mean
- **Hierarchy temporal features** (Phase 3): group_total, group_mean, share_of_group, rank_in_group, group_trend at SubCategory/Brand/VolumeSegment levels
- **Regime features** (Phase 6): growth, volatility, momentum, level_shift, zero_rate
- **Cross-key relative features** (Phase 6): relative to overall mean, segment mean, hierarchy group
- **History embeddings** (Phase 6): frozen aggregates (mean, std, trend, correlation) from history-only features

### 5. Model Training

Trains models per segment with walk-forward cross-validation. Uses a model registry with 35+ families. Intelligent model-level allocation determines whether each key gets an individual model or is pooled with its segment (based on data sufficiency, volume importance, forecastability, uniqueness, and predictability scores).

### 6. Inference

Forward forecast generation with recursive multi-step forecasting. Includes:
- Feature regeneration from latest data
- Intelligent feature imputation (SPLY per key, hierarchy group median/mode)
- Bias calibration (segment x zero_fraction + lag-specific)
- Hierarchical reconciliation
- YoY trend adjustment (optional)
- Outputs CSV and optional Parquet

### 7. Diagnostics

Analyses inference results:
- WAPE and accuracy metrics at configurable aggregation levels
- Per-segment and per-group diagnostics
- Model verdict (deployment readiness assessment)
- Diagnostic charts

### 8. Backtesting

Rolling-origin backtesting across multiple forecast origins:
- Generates origins from validation end through test end
- Optional feature regeneration at each origin (leakage-free)
- Per-origin WAPE and bias metrics
- Aggregated summary across all origins

---

## Model Families

### Tree-Based (universal)

| Model | Description |
|-------|-------------|
| `lightgbm` | Fast gradient boosting, handles sparse data |
| `xgboost` | Robust gradient boosting for tabular features |
| `catboost` | Native categorical handling, robust to overfitting |
| `random_forest` | Ensemble method, less prone to overfitting |

### Zero-Handling / Compound

| Model | Description |
|-------|-------------|
| `zero_inflated` | Separate zero probability + demand size models |
| `hurdle_model` | P(demand occurs) x E[demand given demand > 0] |
| `tweedie` | Single model with Tweedie distribution for zero-inflated continuous |

### Multi-Horizon Direct (Phase 4)

| Model | Description |
|-------|-------------|
| `multi_horizon_lightgbm` | LightGBM with direct multi-step training |
| `multi_horizon_xgboost` | XGBoost with direct multi-step training |
| `multi_horizon_ensemble` | Ensemble optimized for target horizon |

### Hierarchical (Phase 3)

| Model | Description |
|-------|-------------|
| `global_local` | Global model on all keys + per-key bias correction |
| `mixed_effects` | Global model + per-key residual adjustment |
| `multi_level_ensemble` | Global + hierarchy + key level, data-adaptive weights |

### Enhanced (Phase 7)

| Model | Description |
|-------|-------------|
| `catboost_embedding` | CatBoost with native key categorical embeddings |
| `quantile_regression` | Median (0.5) for robust point forecast + 0.1/0.9 intervals |
| `conformal_boost` | Residual boosting + conformal prediction intervals |

### Model Combination (Phase 8)

| Model | Description |
|-------|-------------|
| `stacked_ensemble` | 5 diverse base models + LightGBM meta-learner |

Multi-level forecast combination is also available (key-level + segment-level + category-level with data-adaptive weights).

### Statistical / Univariate

| Model | Description |
|-------|-------------|
| `arima` | Autoregressive integrated moving average |
| `sarima` | Seasonal ARIMA |
| `ets` | Exponential smoothing (Error-Trend-Seasonality) |
| `theta` | Theta method |
| `tbats` | Trigonometric seasonality, Box-Cox, ARMA errors |
| `croston` | Classic intermittent demand method |
| `sba` | Syntetos-Boylan Approximation (bias-corrected Croston) |
| `tsb` | Teunter-Syntetos-Babai (handles demand obsolescence) |
| `imapa` | Intermittent Multiple Aggregation Prediction Algorithm |
| `prophet` | Facebook's decomposable additive model |
| `bsts` | Bayesian Structural Time Series |

### Deep Learning (optional, `enable_deep_models: true`)

| Model | Description |
|-------|-------------|
| `tft` | Temporal Fusion Transformer |
| `lstm` | Long Short-Term Memory networks |
| `nbeats` | Neural Basis Expansion Analysis |
| `deepar` | Probabilistic autoregressive RNN |
| `wavenet` | Dilated causal convolutions |

### Discrete Demand Specialists

| Model | Description |
|-------|-------------|
| `ordinal_regression` | For low-cardinality targets |
| `discrete_classifier` | Classification-based demand model |
| `hybrid_discrete` | Combined continuous + discrete approach |

---

## Feature Engineering

### Base Features

All features are leakage-free via `shift(forecast_lag)`. Static during recursive forecasting (computed from actuals only, not updated with predictions).

| Category | Features |
|----------|----------|
| Lag | ACF-informed lags (default: 1, 2, 4, 8, 13, 26, 52) |
| Rolling | Mean, std, min, max over windows (default: 4, 8, 13 periods) |
| EWM | Exponentially weighted moving averages |
| Seasonal | Seasonal decomposition, Fourier terms |
| Calendar | Week-of-year, month, quarter |
| Cross-sectional | Rank within period, relative to period mean |

### Hierarchy Temporal Features (Phase 3)

Created at SubCategory, Brand, VolumeSegment levels:

| Feature | Description |
|---------|-------------|
| `{group}_total` | Sum of target across all keys in hierarchy group |
| `{group}_mean` | Mean target across group |
| `share_of_{group}` | Key's share of group total (market position) |
| `rank_in_{group}` | Key's rank within group |
| `{group}_trend` | Group-level trend (macro growth/decline) |

### Regime Features (Phase 6)

| Feature | Description |
|---------|-------------|
| `growth_regime` | Is recent demand above long-term average? |
| `volatility_regime` | Is recent volatility above normal? |
| `momentum` | Is demand accelerating or decelerating? |
| `level_shift` | Has the level changed significantly? |
| `zero_rate` | Recent zero-demand frequency |

### Cross-Key Relative Features (Phase 6)

| Feature | Description |
|---------|-------------|
| Relative to overall | Key's performance vs all keys |
| Relative to segment | Key's performance vs segment peers |
| Relative to hierarchy group | Key's performance vs hierarchy group |

### History Embeddings (Phase 6)

For features classified as `history_only` by Feature Availability Detection, frozen key-level aggregates are created: historical mean, std, trend slope, and correlation with target. These preserve the feature's signal without requiring future values.

---

## Hierarchical Reconciliation

Five reconciliation methods ensure coherent forecasts across the product hierarchy:

| Method | Description | Notes |
|--------|-------------|-------|
| `mint_shrink` | MinT with Ledoit-Wolf shrinkage | **Default, recommended.** Minimises total forecast variance under coherency constraint. |
| `bottom_up` | Sum leaf-level forecasts to parents | Simple, no adjustment to leaf forecasts. |
| `wls` | Weighted least squares | Diagonal covariance weighting. |
| `ols` | Ordinary least squares | Equal weights. |
| `top_down` | Prophet aggregate + proportional scaling | Legacy method. Trains Prophet on category total, scales keys proportionally. |

Set via `reconciliation_method` in the design config. The summing matrix is built from the hierarchy map (keys grouped by hierarchy columns). MinT reconciliation implements the method from Wickramasuriya et al. (2019).

---

## Bias Calibration

Two complementary calibration systems correct systematic forecast bias:

### Segment x Zero-Fraction Calibration

Learns per (segment, zero_fraction_bucket) correction factors from validation data. Keys are bucketed by zero_fraction percentile (configurable, default 5 = quintiles). Applied multiplicatively to predictions.

### Lag-Specific Calibration (Phase 4)

Learns per (segment, lag) correction factors from validation recursive forecasts. Addresses the fact that longer lags have different systematic biases than shorter lags.

### Application Priority

During inference, calibration factors are applied with priority: lag-specific > segment x zero_fraction > global fallback. Both systems are integrated into the inference pipeline.

---

## Project Structure

```
config/
  schema.py                          -- Pydantic config (DemandForecastConfig, DesignConfig, context schemas)
  config_de_*.yaml                   -- 10 DACH category configs
  llm_config.py                      -- AWS Bedrock LLM provider
  llm_config_databricks.py           -- Databricks AIF LLM provider

crews/
  eda_crew.py                        -- EDA analysis crew
  feature_availability_crew.py       -- Feature availability detection crew
  segmentation_crew.py               -- Clustering + model allocation crew
  feature_crew.py                    -- Feature engineering crew
  training_crew.py                   -- Model training crew
  diagnostic_crew.py                 -- Model diagnostics crew

utils/
  # Core pipeline
  agent_utilities.py                 -- Shared utilities (load data, compute metrics)
  eda.py                             -- EDA pipeline (Syntetos-Boylan, seasonality, ACF)
  segmentation.py                    -- Clustering (GMM, KMeans, HDBSCAN) + hybrid segments
  feature_engineering.py             -- Leakage-free feature pipeline
  model_training.py                  -- Training registry (35+ models) + orchestration
  inference.py                       -- Forward forecast generation + reconciliation
  backtesting.py                     -- Rolling-origin backtesting
  recursive_forecasting.py           -- Recursive multi-step forecasting
  diagnostics.py                     -- Forecast accuracy diagnostics

  # Phase 1: Feature Availability
  feature_availability.py            -- Auto-detect known/unknown/partial features

  # Phase 2: Enriched Segmentation
  segmentation_enhanced.py           -- External response profiles, seasonality shape, hierarchy detection

  # Phase 3: Hierarchical
  hierarchy_features.py              -- Hierarchy temporal features (group totals, share, trend)
  hierarchical_reconciliation.py     -- MinT, bottom-up, WLS, OLS reconciliation
  hierarchical_model_training.py     -- Global-local, mixed-effects, multi-level ensemble

  # Phase 4: Direct Forecasting
  direct_forecaster.py               -- DirectForecaster, HybridForecaster, lag-specific calibration

  # Phase 6: Rich Features
  rich_features.py                   -- Regime, cross-key relative, history embeddings

  # Phase 7: Enhanced Models
  enhanced_model_zoo.py              -- CatBoost embeddings, quantile regression, conformal boost

  # Phase 8: Model Combination
  model_combination.py               -- Multi-level combination, diverse stacking ensemble

  # Infrastructure
  period_utils.py                    -- YYYY-WW normalisation, period comparison utilities
  bias_calibration.py                -- Segment x zero_fraction bias calibration
  reconciliation.py                  -- Top-down Prophet reconciliation (legacy)
  model_selection_intelligence.py    -- Dynamic model priority and meta-learning
  intelligent_modeling.py            -- Individual vs pooled model allocation
  context_manager.py                 -- Inter-crew context passing
  context_schema.py                  -- Context file schemas
  context_summarizer.py              -- Context summarisation for LLM prompts
  cost_tracking.py                   -- LLM token and cost tracking
  trace_logging.py                   -- CrewAI trace logging
  crew_output_validator.py           -- Crew output validation
  code_execution_tool.py             -- Code execution tool for agents
  data_profiler.py                   -- Data profiling utilities
  dead_key_handler.py                -- Dead/inactive key handling
  eda_recommendations.py             -- EDA-based recommendations
  feature_reasoning.py               -- Feature importance reasoning
  forecast_optimization.py           -- Forecast optimization techniques
  leakage_free_features.py           -- Feature leakage prevention
  model_registry.py                  -- Model registry
  model_training_advanced.py         -- Advanced training utilities
  multi_horizon_training.py          -- Multi-horizon model training
  output_suppressor.py               -- Output suppression for clean logs
  pipeline_generator.py              -- Pipeline generation utilities
  smart_output.py                    -- Smart output formatting
  state_of_art_training.py           -- State-of-the-art training techniques
  target_transforms.py               -- Target variable transformations
  walk_forward_cv.py                 -- Walk-forward cross-validation

# Entry points
runner.py                            -- Full pipeline runner with email notifications
databricks_runner.py                 -- Databricks entry point
databricks_runner_uk.py              -- Databricks entry point (UK variant)
run_eda.py                           -- EDA stage runner
run_feature_availability.py          -- Feature availability stage runner
run_segmentation.py                  -- Segmentation stage runner
run_feature.py                       -- Feature engineering stage runner
run_training.py                      -- Training stage runner
run_inference.py                     -- Inference stage runner
run_diagnostic.py                    -- Diagnostics stage runner
run_backtesting.py                   -- Backtesting stage runner

# Testing
tests/
  test_e2e_time_format.py            -- End-to-end time format tests
  test_e2e_multi_horizon.py          -- End-to-end multi-horizon tests
  test_e2e_imputation.py             -- End-to-end imputation tests
  test_e2e_data_formats.py           -- End-to-end data format tests
unit_test_runner.py                  -- Unit test runner
```

---

## Deployment

### Local (AWS Bedrock)

```yaml
llm_provider: "bedrock"
```

Requires AWS credentials configured for Bedrock access. The default model is Claude Sonnet.

```bash
python runner.py --config config/config_de_skincare.yaml
```

### Databricks

```yaml
llm_provider: "databricks"
databricks_base_path: "/Volumes/catalog/schema/path/"
```

Use `databricks_runner.py` which handles dependency installation and Databricks-specific path resolution:

```bash
python databricks_runner.py --config /Workspace/Repos/.../config/config.yaml
```

### Artifacts

All outputs are written to `{artifact_base_path}/`:

```
{artifact_base_path}/
  eda_output/                   -- EDA metrics, context files, reports
  feature_availability_output/  -- Feature classification, embedding specs
  segmentation_output/          -- Segment assignments, modeling strategy
  feature_output/               -- Feature metadata, quality summary
  model_artifacts/              -- Trained models, forecasts, model specs
  diagnostics_output/           -- WAPE metrics, diagnostic charts, verdict
  backtest_output/              -- Per-origin forecasts, aggregated metrics
  trace_logs/                   -- CrewAI trace logs (if enabled)
```
