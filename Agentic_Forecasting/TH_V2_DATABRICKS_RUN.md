# Thailand v2 — Databricks run (GBM hybrid + deep TFT overlay)

The v2 forecast = the existing DIQ GBM pipeline for **every** category **plus** a global
**Temporal Fusion Transformer** that replaces the forecast for the **routed high-volume
categories** (where the deep model beats the per-horizon GBM on both t+4&5 and 13-week —
e.g. FOODS). It runs through the **same** `diq_runner` flow; the deep overlay is a single
config-gated stage that slots in after the per-category pipeline and before stack + MOQ, so
the output layout is unchanged. **Default OFF → UK/DE runs are byte-identical.**

## What runs (all steps)

1. `_load_and_join_inputs` → unified TH input (Spark → pandas), filtered to the categories.
2. Per category → `run_full_deterministic_pipeline` → `artifacts/model_artifacts/inference_forecast.csv` (the GBM/DMH forecast).
3. **NEW — deep overlay** (`utils.deep_forecast.apply_deep_overlay`): trains ONE global TFT across all series (known-future calendar/promo/holiday + static category covariates), then overwrites the `predicted` column of each **routed** category's `inference_forecast.csv` (schema preserved). Best-effort — any failure keeps the GBM forecast, so the run never regresses below the GBM baseline.
4. `_stack_and_moq` → MOQ post-processing → final `{...}/TF_DIQ/{snapshot}/inference_forecast.parquet` (unchanged).

## How to run

### A) Notebook (`databricks_diq_notebook.py`) — recommended
Set the widgets:
- `MARKET` = `th`
- `CONFIG_ROOT` = your UC-Volume DIQ_configs path (or blank for in-repo `config/`)
- `category_list` = `#`-separated TH category tokens (must match the input's `category_name` values, uppercased)
- `history_till` / `snapshot_week` (e.g. `202602` / `202603`)
- **`DEEP_OVERLAY`** = `true`
- **`DEEP_ROUTE`** = `FOODS#FABRIC_ENHANCERS#FACE#HAIR_CARE#BODY#DEODORANTS_AND_FRAGRANCES`  *(the categories the TFT wins; the rest keep the GBM hybrid)*
- **`DEEP_MODEL`** = `TFT` (or `NHITS` / `TiDE`), **`DEEP_MAX_STEPS`** = `2000`

### B) Job / CLI (`databricks_runner_diq.py`)
```
python databricks_runner_diq.py \
    --parent-dir "/Volumes/.../th_rerun" \
    --history-till 202602 --snapshot-week 202603 \
    --category-list "FOODS#FABRIC_CLEANING#HAIR_CARE#..." \
    --deep-overlay \
    --deep-route "FOODS,FABRIC_ENHANCERS,FACE,HAIR_CARE,BODY,DEODORANTS_AND_FRAGRANCES" \
    --deep-model TFT --deep-max-steps 2000
```

## Routing (why these categories)

The TFT is the better model on the dense high-volume categories (it cracked FOODS — wins both
t+4&5 and 13-week); the GBM hybrid (global pooled DMH + per-key rescue + adaptive SPLY) stays on
the categories it wins (FABRIC CLEANING via per-key rescue, HOME & HYGIENE, SKIN CLEANSING). Tune
`DEEP_ROUTE` per the latest scorecard.

## Cluster requirements

- `requirements.txt` now includes `neuralforecast>=3.0.0` (installed by `databricks_init.sh` / the
  runner's dependency step). It pulls `pytorch-lightning`, `ray`, `optuna`.
- **A GPU cluster is recommended** for the TFT (the model auto-selects CUDA/MPS/CPU; CPU works but
  is much slower). Training is one global model for the whole market, not per category.

## Code touched (all backward-compatible, deep overlay default OFF)

- `utils/deep_forecast.py` — NEW: `train_global_deep`, `apply_deep_overlay`.
- `utils/diq_runner.py` — `run_diq_forecast(...)` gains `enable_deep_overlay`, `deep_route_categories`, `deep_model`, `deep_max_steps`; the overlay stage runs after the category loop.
- `databricks_runner_diq.py` — `--deep-overlay/--deep-route/--deep-model/--deep-max-steps`.
- `databricks_diq_notebook.py` — `DEEP_OVERLAY/DEEP_ROUTE/DEEP_MODEL/DEEP_MAX_STEPS` widgets.
- `requirements.txt` — `neuralforecast`.
