# FEU-Agentic Forecasting: Visual Summary & Understanding

## Overview

FEU (Forecast Enhancement Upgrade) is an **agentic demand forecasting pipeline** for FMCG/CPG categories. It combines LLM-powered reasoning (CrewAI agents) with deterministic computation (core pipeline) to produce accurate, explainable forecasts across product hierarchies.

---

## Quick Visual Map

### What Problem Does It Solve?

**Fishbone Analysis:** The pipeline integrates multiple factors into one system:
- **Data Quality:** Clean actuals, correct timestamps, no missing data
- **Feature Engineering:** Leakage-free lags, seasonal decomposition, hierarchy features
- **Model Selection:** 35+ model families with intelligent allocation
- **Segmentation:** Smart clustering (GMM, KMeans, HDBSCAN) to group similar products
- **Bias Calibration:** Segment × zero-fraction and lag-specific correction
- **Reconciliation:** MinT shrinkage ensures hierarchy coherence
- **Validation:** Backtesting WAPE and diagnostic metrics

**Result:** Low WAPE (Weighted Absolute Percentage Error) forecasts with explainable model verdicts.

---

## 8-Stage Pipeline Flow

```
Stage 1: EDA
  └─→ Stage 2: Feature Availability
        └─→ Stage 3: Segmentation
              └─→ Stage 4: Feature Engineering
                    └─→ Stage 5: Model Training
                          ├─→ Stage 6: Inference (conditional)
                          │     └─→ Stage 7: Diagnostics (conditional)
                          └─→ Stage 8: Backtesting (conditional)
```

**Conditional Execution (Run Modes):**
- **`backtest_and_forecast`** (default): Run stages 1-8 (full pipeline)
- **`forecast_only`**: Run stages 1-7 (train on all history, no backtest)
- **`backtest_only`**: Run stages 1-5 + 8 (validation only, no forward forecast)

---

## Key Concepts Explained

### 1. EDA (Exploratory Data Analysis) — Stage 1

Deterministic analysis producing:
- **Syntetos-Boylan classification:** Categorizes demand patterns as smooth, erratic, intermittent, or lumpy
- **Seasonality detection:** Identifies seasonal periods (e.g., 52 weeks for annual patterns)
- **ACF analysis:** Informs optimal lag selection (autocorrelation function)
- **Per-key metrics:** Volume mean, CV, zero_fraction, intermittency patterns

Output: Recommendations for downstream stages (segmentation, feature engineering).

---

### 2. Feature Availability Detection — Stage 2

Auto-detects feature "knowability" on forecast date:
- **`known_in_future`:** Available during forecast (e.g., planned price, promo calendar)
- **`history_only`:** Not available in future → converted to frozen key-level embeddings
- **`partially_known`:** Available where provided, imputed beyond
- **`excluded`:** Too sparse or uninformative

Smart approach: Don't discard history-only features; preserve their signal via embeddings.

---

### 3. Segmentation — Stage 3

Groups similar keys to improve model training:
- **Algorithms:** GMM (default), KMeans, HDBSCAN, Hierarchical
- **Features:** volume_mean, CV, zero_fraction, demand_frequency, intermittency
- **Hybrid dimensions:** Business tiers (volume_tier, demand_pattern) merged with cluster assignments
- **Intermittency-aware:** Prevents mixing regular and sparse keys

Output: Segment assignments for model-level allocation (individual vs pooled models).

---

### 4. Feature Engineering — Stage 4

**Leakage-Free Design:** All features use `shift(forecast_lag)` to prevent future data from bleeding into training.

**Feature Categories:**

| Category | Examples | Purpose |
|----------|----------|---------|
| **Lag** | ACF-informed (1,2,4,8,13,26,52) | Captures recent history, seasonality |
| **Rolling** | Mean, std, min, max (4,8,13 windows) | Trend, volatility signals |
| **EWM** | Exponentially weighted moving average | Adaptive smoothing, trend capture |
| **Seasonal** | Decomposition, Fourier terms | Captures repeating patterns |
| **Calendar** | Week-of-year, month, quarter | Holidays, business cycles |
| **Cross-sectional** | Rank within period, relative to mean | Competitive positioning |
| **Hierarchy** | Group totals, share, rank, trend | Multi-level context |
| **Regime** | Growth, volatility, momentum, level_shift | State-dependent behavior |
| **Cross-Key Relative** | vs segment, vs category | Market dynamics |
| **History Embeddings** | Frozen aggregates (for history-only) | Preserved signal without future data |

**Static During Recursive Forecasting:** Features don't update with predictions; only with actual data.

---

### 5. Model Training — Stage 5

Trains **35+ model families** per segment:

**Tree-Based (Universal):**
- LightGBM, XGBoost, CatBoost, Random Forest

**Zero-Handling (for sparse demand):**
- Zero-inflated, Hurdle model, Tweedie

**Statistical (Univariate):**
- ARIMA, SARIMA, ETS, Prophet, TBATS, Croston, SBA, TSB, IMAPA

**Hierarchical (Multi-level):**
- Global-local, Mixed-effects, Multi-level ensemble

**Specialized:**
- Quantile regression, Conformal boost, CatBoost embeddings

**Multi-Horizon Direct:**
- Direct forecasting for all horizons simultaneously (avoids recursive error accumulation)

**Ensemble:**
- Stacked ensemble, Multi-level combination

**Deep Learning (optional):**
- TFT, LSTM, NBEATs, DeepAR, WaveNet

**Validation:** Walk-forward cross-validation on train/val splits; per-segment model selection via meta-learning.

---

### 6. Inference — Stage 6

Forward forecast generation:
1. Regenerate features from latest data
2. Handle missing future features intelligently (SPLY per key, hierarchy group median/mode)
3. Apply bias calibration (segment × zero_fraction + lag-specific)
4. Generate recursive multi-step forecasts
5. Reconcile forecasts to ensure hierarchy coherence
6. Optional: Apply YoY trend adjustment

Output: CSV/Parquet with reconciled forecasts for all horizons.

---

### 7. Diagnostics — Stage 7

Model performance analysis:
- **WAPE metrics:** Per-segment, per-hierarchy-group
- **Accuracy analysis:** Bias, coverage, outliers
- **Diagnostic charts:** WAPE trends, residual distributions
- **Model verdict:** Deployment readiness assessment

---

### 8. Backtesting — Stage 8

Rolling-origin validation:
- Multiple forecast origins (from val_end through test_end)
- Per-origin WAPE and bias metrics
- Aggregated summary (mean, std, percentiles)
- Optional feature regeneration at each origin (leakage-free)

Confirms model robustness before deployment.

---

## System Architecture

**4 Layers:**

1. **Input & Configuration**
   - CSV data, YAML config (auto-detect enabled)
   - LLM provider (AWS Bedrock or Databricks)
   - Schema validation (Pydantic)
   - Period utilities (YYYY-WW normalization)

2. **Analysis Crews (CrewAI Agents)**
   - EDA Crew, Feature Availability Crew, Segmentation Crew
   - Feature Engineering Crew, Training Crew, Diagnostics Crew
   - Hybrid: deterministic computation + LLM reasoning

3. **Core Utilities (Deterministic)**
   - Model registry (35+ families)
   - Walk-forward CV, inference, bias calibration
   - Reconciliation, backtesting, recursive forecasting
   - EDA, feature engineering, hierarchy features, rich features
   - Data management (data profiler, dead key handler, context manager)

4. **Deployment & Outputs**
   - Local (AWS Bedrock) or Databricks (AIF)
   - Artifacts: CSV, Parquet, JSON context files
   - Monitoring: logs, traces, cost tracking

---

## Configuration

**Minimal Config Example:**
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

# Auto-detection (recommended)
train_start: ""
train_end: ""
val_start: ""
val_end: ""
test_start: ""
test_end: ""
```

**Auto-Detection Behavior:**
- **Train/val/test splits:** Detects history cutoff, allocates periods intelligently
- **Hierarchy columns:** Finds SubCategory, Brand, VolumeSegment in data
- **Features:** Classifies all columns by availability type

---

## Hierarchical Reconciliation

Ensures forecasts are coherent (leaf-level sums equal parent-level forecasts).

**Methods:**
| Method | Approach | Best For |
|--------|----------|----------|
| **MinT Shrink** (default) | Ledoit-Wolf shrinkage under coherency constraint | Minimizes variance, recommended |
| **Bottom-up** | Sum leaf forecasts to parents | Simple, no adjustment |
| **WLS** | Weighted least squares (diagonal covariance) | Weighted coherency |
| **OLS** | Ordinary least squares (equal weights) | Basic coherency |
| **Top-down** | Prophet aggregate + proportional scaling | Legacy fallback |

---

## Bias Calibration

Two systems correct systematic bias:

### Segment × Zero-Fraction Calibration
- Learns per (segment, zero_fraction_bucket) correction from validation data
- Applied multiplicatively to predictions
- Captures segment-specific biases across demand sparsity levels

### Lag-Specific Calibration
- Per (segment, lag) correction factors
- Longer lags have different systematic biases
- Prioritized during inference

---

## Project Structure

```
config/                 → YAML configs, schemas, LLM providers
crews/                  → CrewAI agent definitions
utils/                  → Core utilities (deterministic)
  - eda.py, segmentation.py, feature_engineering.py
  - model_training.py, inference.py, backtesting.py
  - hierarchy_features.py, rich_features.py
  - bias_calibration.py, reconciliation.py
  - [many more specialized utilities]
runner.py               → Full pipeline entry point
databricks_runner.py    → Databricks deployment
run_*.py                → Individual stage runners
tests/                  → Unit & E2E tests
```

---

## Deployment

### Local (AWS Bedrock)
```bash
python runner.py --config config/config_de_skincare.yaml
# Optional: Email notifications
python runner.py --config config/config_de_skincare.yaml --email
```

### Databricks
```bash
python databricks_runner.py --config /Workspace/Repos/.../config.yaml
```

### Outputs
All artifacts → `{artifact_base_path}/`:
```
eda_output/
feature_availability_output/
segmentation_output/
feature_output/
model_artifacts/
diagnostics_output/
backtest_output/
trace_logs/
```

---

## Key Strengths

1. **Agentic Design:** LLM agents for analysis + deterministic core for reproducibility
2. **Feature Richness:** Base + hierarchy + regime + cross-key + embeddings → 50+ features per key
3. **Model Coverage:** 35+ families, auto-selection, hierarchical variants
4. **Leakage-Free:** All features shifted, recursive forecasting isolated
5. **Hierarchy Aware:** Multi-level reconciliation, group-level features
6. **Zero-Demand Handling:** Specialized models + intermittency-aware segmentation
7. **Auto-Detection:** Config simplicity via intelligent defaults
8. **Walk-Forward Validation:** Realistic backtesting with per-origin metrics
9. **Explainability:** Model verdicts, diagnostic charts, cost tracking
10. **Scalability:** Databricks integration, per-segment parallelization

---

## Quick Command Reference

```bash
# Full pipeline
python runner.py --config config/config_de_skincare.yaml

# With email
python runner.py --config config/config_de_skincare.yaml --email

# Individual stages
python run_eda.py --config config/config_de_skincare.yaml
python run_feature_availability.py --config config/config_de_skincare.yaml
python run_segmentation.py --config config/config_de_skincare.yaml
python run_feature.py --config config/config_de_skincare.yaml
python run_training.py --config config/config_de_skincare.yaml
python run_inference.py --config config/config_de_skincare.yaml
python run_diagnostic.py --config config/config_de_skincare.yaml
python run_backtesting.py --config config/config_de_skincare.yaml

# Databricks
python databricks_runner.py --config /Workspace/Repos/.../config.yaml
```

---

## Summary

FEU transforms raw time series data into production-ready forecasts through:

1. **Intelligent analysis** (CrewAI agents)
2. **Rich feature engineering** (leakage-free, multi-dimensional)
3. **Sophisticated model training** (35+ families, per-segment allocation)
4. **Robust validation** (walk-forward CV, backtesting)
5. **Explainability** (model verdicts, diagnostics)
6. **Hierarchy coherence** (MinT reconciliation)
7. **Bias correction** (multi-level calibration)
8. **Scalable deployment** (local or Databricks)

The result: **FMCG demand forecasts with low WAPE, explainable models, and production-ready confidence.**
