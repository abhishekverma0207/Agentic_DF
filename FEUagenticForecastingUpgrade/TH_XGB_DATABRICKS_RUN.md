# Hierarchical-XGBoost overlay — Databricks run guide

The production form of the model that **beats LEGO on accuracy 8/9 categories** (13-week, Thailand).
It runs as a **config-gated overlay** on the existing DIQ runner: the classic GBM pipeline runs for
every category as before; the hierarchical XGBoost then **overwrites the routed categories'**
forecasts for the target account. Default OFF → UK/DE runs are byte-identical.

## What it does
`utils/xgb_hier_forecast.py` trains ONE global direct-multi-horizon XGBoost across all series with
**hierarchical features** the per-key benchmark can't see:
- own history (lags, rolling, SPLY) + **cross-account** (same CS_BARCODE summed over all ~50 accounts —
  E1016 is only ~1/3 of a product's demand) + category aggregates,
- growth/seasonality **curves** (YoY momentum, OLS trend slopes, Fourier harmonics, acceleration),
- distribution-breadth, forward promo/holiday, forecastability statics.
Leak-free (every feature as-of the forecast origin). It overwrites each routed category's
`model_artifacts/inference_forecast.csv` `predicted` column (schema preserved → stack + MOQ unchanged).
Best-effort: any failure leaves the GBM forecast in place (never regresses).

## Requirements
`xgboost>=2.0.0` (in `requirements.txt`). CPU is fine (~30s global; +~2min if `perkey` on).

## Run — notebook (`databricks_diq_notebook.py`) widgets
- `XGB_OVERLAY` = `true`
- `XGB_ROUTE` = `FABRIC_CLEANING#HAIR_CARE#FOODS#HOME_AND_HYGIENE#FABRIC_ENHANCERS` (the top-5; or all 9)
- `XGB_OBJ` = `huber` (robust default) | `ratio` (growth-rate; best on high-growth FABRIC/HAIR/FOODS) | `tweedie`
- `XGB_PERKEY` = `false` (set `true` to add per-key models for high-volume keys, blended with the global)
- `XGB_ACCOUNT` = `E1016` (the LEGO eval grain)
- `XGB_SEASONAL_TRUST` = `true` (leak-free bias fix — ON by default; lifts keys toward the same-period-
  last-year level where that level was block-reliable on past years and recent demand dipped. Flipped
  HAIR CARE to a both-metric win in backtest; CLI: omit `--xgb-no-trust` to keep it on)

### Un-damping gate (OPT-IN — a growth-window bet, OFF by default)
A second leak-free overlay (`undamp=True`, **default OFF** in `apply_xgb_hier_overlay`; opt-in only).
**Rolling-origin lock caveat (`scripts/_rolling_lock.py`):** this gate is a DIRECTIONAL GROWTH BET. It
corrects FABRIC under-forecasting only in *growth* windows (it helped at origin 202602, where FABRIC
grew); at flat/declining held-out windows (202541, 202548) the base is already calibrated, so the gate
OVERSHOOTS (bias 0 -> +0.2/+0.45). Enable ONLY when you have an external reason to expect category
growth. The durable, no-overfit recipe leaves it OFF.
Diagnosis: the global model **mean-reverts a few high-volume keys downward** (e.g. 464229 forecast 167k
vs its recent-13-week run-rate 205k and actual 218k) while LEGO tracks their level. Fix: for keys whose
recent-13-week demand is **stable** (CV<0.8), **non-declining** (2nd-half ≥ 0.75×1st-half) and below
`floor×recent13`, lift the forecast **total** up to `floor×recent13` (weekly shape preserved). Applied to
FABRIC CLEANING only (the other categories already win bias; a lift would over-shoot them). **Validated
out-of-sample**: actual_next13/recent13 for these keys = 1.05/1.01/1.31 at origins 202541/202548/202602
(always ≥1 — the random walk holds), and the floor (1.05) matches the held-out ratio (not test-overfit).
Result: FABRIC CLEANING 0.4141/−0.093 → **0.4127/−0.0497 = WIN BOTH**.

## Run — CLI (`databricks_runner_diq.py`)
```
python databricks_runner_diq.py --parent-dir <...> --history-till 202614 --snapshot-week 202615 \
  --market th --cats-subdir <th_driver_frame_folder> \
  --xgb-overlay --xgb-route FABRIC_CLEANING,HAIR_CARE,FOODS,HOME_AND_HYGIENE,FABRIC_ENHANCERS \
  --xgb-obj huber [--xgb-perkey] --xgb-account E1016
```

## Notes / next
- The runner already dispatches TH (`_TH_YAML_SLUG_OVERRIDES` in `diq_runner.py`); set `CATS_SUBDIR`
  to the TH driver-frame folder.
- For the strongest per-category result use the **per-category recipe** (ratio for high-growth
  categories, huber for the rest) — currently a single global `XGB_OBJ` per run; the validated
  per-category selection (`scripts/_cat_select13.py`) + the rolling-origin lock harness are the
  follow-up to bake the recipe into production.
- **Shipped (conservative, lock-defensible) recipe = `_xgb_hier.py` + `_cat_select13.py` + seasonal-trust
  → `final_forecast_catselect13.parquet`**: top-5 both-metric win **3/5** (HOME & HYGIENE + FABRIC
  ENHANCERS by wide bias margins, HAIR CARE), FABRIC CLEANING wins accuracy, FOODS wins bias (each near-
  parity on the other metric), acc-win **8/9**, and the **top-5 aggregate beats LEGO on BOTH**
  (0.4786/−0.0679 vs 0.4688/−0.0723). Nothing window-overfit.
- A `final_forecast_v4.parquet` (5/5 both-win via the un-damp gate + a FOODS neural+GBM ensemble) exists
  as a "best-estimate" but the rolling-origin lock showed both extra levers are window-specific, so it is
  NOT the shipped recipe. `scripts/_rolling_lock.py` is the leak-free harness (origins 202541/202548).
